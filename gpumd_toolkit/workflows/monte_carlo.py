"""4. Monte Carlo・組成自由度.

GPUMD の ``mc`` キーワードは MD の途中に MC 試行を挟み、原子の入れ替えや
化学種の変更をサンプリングする (NEP 専用)。

============================== ==================================================
アンサンブル                    使いどころ
============================== ==================================================
カノニカル (``canonical``)      組成固定。短距離規則度・偏析・サイト占有の平衡化
半グランドカノニカル (``sgc``)  化学ポテンシャル差 Δμ を与えて組成を動かす
分散拘束 SGC (``vcsgc``)        二相共存領域でも組成を安定にサンプリングできる
============================== ==================================================

``mcmc`` は ``time_step 0`` の NVE で MD を止め、GPUMD の ``mc canonical`` だけを
回す純粋な MCMC (原子種の交換のみ)。ASE calculator で回す
:class:`~gpumd_toolkit.mcmc.MetropolisMC` よりはるかに速い。

Examples
--------
>>> from gpumd_toolkit.workflows import MonteCarloCalculation
>>> calc = MonteCarloCalculation("POSCAR", "nep.txt", "runs/sqs")
>>> calc.canonical(temperature=1000, steps=100000, md_steps=100, mc_trials=200)
>>> calc.run()
>>> calc.concentrations().tail()
"""

from __future__ import annotations

import math
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from ..inputs import modifiers
from ..inputs.builder import MDStage
from ..inputs.dumps import DumpSettings
from ..md import GPUMDCalculation
from ..outputs import read_mcmd
from ..structure import StructureHandler

__all__ = ["MonteCarloCalculation"]


class MonteCarloCalculation(GPUMDCalculation):
    """MD + MC による組成・配置サンプリング。

    ``mc`` は MD ステージに付随するキーワードなので、まず MD ステージを
    作り、そこに ``mc`` 行を差し込む形になる。
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        #: MC で扱う元素の並び (``mcmd.out`` の列名に使う)
        self.mc_species: list[str] = []
        #: 直前のステージが ``mcmc`` (time_step 0) か
        self._after_mcmc = False

    def add_stage(self, stage: MDStage) -> "MonteCarloCalculation":
        # mcmc の time_step 0 は GPUMD では次の run に引き継がれるので、
        # 後続ステージが時間刻みを指定していなければ既定値に戻す
        if self._after_mcmc and stage.time_step is None:
            stage = replace(stage, time_step=self.builder.time_step)
        self._after_mcmc = False
        return super().add_stage(stage)

    def check_box_size(self, *, strict: bool = True) -> dict[str, float]:
        """MCMD が要求するセルサイズを満たしているか確認する。

        GPUMD は周期方向のセル厚みが ``2.5 * (rc_radial + 1)`` 以下だと
        ``Cannot use small box for MCMD.`` で即座に終了する。
        実行前にここで気づけるようにしておく。
        """
        potential = self.potential if isinstance(self.potential, str) else None
        if potential is None and isinstance(self.potential, list):
            first = self.potential[0]
            potential = first if isinstance(first, str) else first[0]
        cutoffs = StructureHandler.nep_cutoffs(potential) if potential else None
        if cutoffs is None:
            return {}
        required = 2.5 * (cutoffs[0] + 1.0)
        thickness = StructureHandler.cell_thickness(self.atoms)
        info = {
            "rc_radial": cutoffs[0],
            "required_thickness": required,
            "thickness_x": thickness[0],
            "thickness_y": thickness[1],
            "thickness_z": thickness[2],
        }
        too_small = [
            f"{axis}={value:.2f} Å"
            for axis, value, periodic in zip("xyz", thickness, self.atoms.pbc)
            if periodic and value <= required
        ]
        if too_small:
            message = (
                f"MCMD にはセル厚みが {required:.1f} Å より大きい必要があります"
                f" (NEP の radial cutoff {cutoffs[0]:g} Å)。"
                f" 不足: {', '.join(too_small)}。"
                " repeat= か min_cell_length= でスーパーセルを大きくしてください。"
            )
            if strict:
                raise ValueError(message)
            warnings.warn(message, RuntimeWarning)
        return info

    def _mc_stage(
        self,
        line: str,
        *,
        temperature: float,
        steps: int,
        temperature_end: float | None,
        pressure: float | None,
        thermostat: str,
        barostat: str,
        stage_kwargs: dict,
    ) -> "MonteCarloCalculation":
        self.check_box_size(strict=False)
        if pressure is None:
            self.nvt(
                temperature=temperature,
                temperature_end=temperature_end,
                steps=steps,
                thermostat=thermostat,
                **stage_kwargs,
            )
        else:
            self.npt(
                temperature=temperature,
                temperature_end=temperature_end,
                steps=steps,
                pressure=pressure,
                barostat=barostat,
                **stage_kwargs,
            )
        return self.add_commands(line)

    def canonical(
        self,
        *,
        temperature: float,
        steps: int,
        md_steps: int = 100,
        mc_trials: int = 100,
        temperature_end: float | None = None,
        pressure: float | None = None,
        group: Sequence[int] | None = None,
        thermostat: str = "nvt_nhc",
        barostat: str = "npt_scr",
        **stage_kwargs,
    ) -> "MonteCarloCalculation":
        """組成を保ったまま原子を入れ替えるカノニカル MC。

        ``md_steps`` ステップの MD ごとに ``mc_trials`` 回の交換を試す。
        ``temperature_end`` を与えると MC の温度も線形に変わるので、
        焼きなまし (SQS 生成・規則化) に使える。
        """
        line = modifiers.mc_canonical(
            md_steps, mc_trials, temperature, temperature_end, group=group
        )
        return self._mc_stage(
            line,
            temperature=temperature,
            steps=steps,
            temperature_end=temperature_end,
            pressure=pressure,
            thermostat=thermostat,
            barostat=barostat,
            stage_kwargs=stage_kwargs,
        )

    def mcmc(
        self,
        *,
        temperature: float,
        trials: int,
        trials_per_call: int = 1000,
        temperature_end: float | None = None,
        group: Sequence[int] | None = None,
        thermo_interval: int | None = 1,
        traj_interval: int | None = None,
        traj_properties: Sequence[str] = ("potential",),
    ) -> "MonteCarloCalculation":
        """MD を止めて原子種の交換だけを行う純粋な MCMC (カノニカル)。

        ``time_step 0`` の NVE ステージに ``mc canonical 1 <trials_per_call> ...``
        を付ける。GPUMD は各 MD ステップの位置更新と力計算の間に MC を挟むが、
        時間刻みが 0 なので座標は一切動かず、MC の交換だけが効く。

        Parameters
        ----------
        trials
            交換試行の総数。``trials_per_call`` の倍数に切り上げる。
        trials_per_call
            1 回の ``mc`` 呼び出し (= 1 MD ステップ) あたりの試行数。
            各ステップで力計算が 1 回走るので、大きいほど無駄が少ない。
            ``mcmd.out`` の受理率・温度の更新・出力はこの単位になる。
        temperature, temperature_end
            MC の温度 [K]。``temperature_end`` を与えると呼び出しごとに線形に変わる。
        thermo_interval, traj_interval
            ``thermo.out`` / ``dump.xyz`` の出力間隔 [呼び出し回数]。
            ``thermo.out`` のポテンシャルエネルギーがそのまま MCMC のエネルギー推移になる。
            ``traj_interval`` の既定は全体で約 10 フレーム。
        """
        if trials <= 0 or trials_per_call <= 0:
            raise ValueError("trials と trials_per_call は正の整数です。")
        self.check_box_size(strict=False)
        calls = math.ceil(trials / trials_per_call)
        if traj_interval is None:
            traj_interval = max(1, calls // 10)
        line = modifiers.mc_canonical(
            1, trials_per_call, temperature, temperature_end, group=group
        )
        T_end = temperature if temperature_end is None else temperature_end
        stage = MDStage(
            "nve",
            steps=calls,
            T_start=temperature,  # velocity 行用 (dt=0 なので運動には効かない)
            time_step=0.0,
            dump=DumpSettings(
                thermo_interval=thermo_interval,
                traj_interval=traj_interval,
                traj_properties=tuple(traj_properties),
            ),
            pre_commands=(line,),
            label=f"MCMC canonical {temperature:g}->{T_end:g} K ({calls * trials_per_call} trials)",
        )
        self.add_stage(stage)
        self._after_mcmc = True
        return self

    def semi_grand_canonical(
        self,
        *,
        temperature: float,
        steps: int,
        chemical_potentials: Mapping[str, float],
        md_steps: int = 100,
        mc_trials: int = 100,
        temperature_end: float | None = None,
        pressure: float | None = None,
        group: Sequence[int] | None = None,
        thermostat: str = "nvt_nhc",
        barostat: str = "npt_scr",
        **stage_kwargs,
    ) -> "MonteCarloCalculation":
        """化学ポテンシャル差 [eV] を与えて組成を動かす SGC-MC。

        差だけが意味を持つので、基準元素を 0 eV にしておけばよい。
        """
        self.mc_species = list(chemical_potentials)
        line = modifiers.mc_sgc(
            md_steps, mc_trials, temperature, chemical_potentials, temperature_end, group=group
        )
        return self._mc_stage(
            line,
            temperature=temperature,
            steps=steps,
            temperature_end=temperature_end,
            pressure=pressure,
            thermostat=thermostat,
            barostat=barostat,
            stage_kwargs=stage_kwargs,
        )

    def vcsgc(
        self,
        *,
        temperature: float,
        steps: int,
        phi: Mapping[str, float],
        kappa: float = 100.0,
        md_steps: int = 100,
        mc_trials: int = 100,
        temperature_end: float | None = None,
        pressure: float | None = None,
        group: Sequence[int] | None = None,
        thermostat: str = "nvt_nhc",
        barostat: str = "npt_scr",
        **stage_kwargs,
    ) -> "MonteCarloCalculation":
        """分散拘束 SGC-MC。二相共存領域でも組成が飛ばない。

        ``phi`` を -1.2 から +1.2 の範囲で振ると組成全域を掃ける。
        ``kappa`` は組成ゆらぎの拘束の強さ (目安 100)。
        """
        self.mc_species = list(phi)
        line = modifiers.mc_vcsgc(
            md_steps, mc_trials, temperature, phi, kappa, temperature_end, group=group
        )
        return self._mc_stage(
            line,
            temperature=temperature,
            steps=steps,
            temperature_end=temperature_end,
            pressure=pressure,
            thermostat=thermostat,
            barostat=barostat,
            stage_kwargs=stage_kwargs,
        )

    # ------------------------------------------------------------------ 結果
    def mcmd(self) -> pd.DataFrame:
        """``mcmd.out`` を読む (ステップ・受理率・各元素の濃度)。"""
        return read_mcmd(self.workdir / "mcmd.out", species=self.mc_species)

    def acceptance_ratio(self, *, drop_fraction: float = 0.3) -> float:
        """MC 試行の平均受理率 (0.2-0.5 程度なら健全)。"""
        frame = self.mcmd()
        tail = frame.iloc[int(len(frame) * drop_fraction) :]
        return float(tail["acceptance"].mean())

    def concentrations(self, *, drop_fraction: float = 0.3) -> pd.Series:
        """平衡化後の平均濃度。

        ``canonical`` は組成を変えないので GPUMD は濃度の列を書かない。
        その場合は空の Series が返る (組成は最初から分かっている)。
        """
        frame = self.mcmd()
        tail = frame.iloc[int(len(frame) * drop_fraction) :]
        columns = [c for c in frame.columns if c.startswith("c_")]
        return tail[columns].mean()

    def final_structure_composition(self) -> dict[str, int]:
        """最終構造 (``dump.xyz`` の最終フレーム) の元素数。"""
        atoms = self.result.final_structure() if self.result else None
        if atoms is None:
            raise RuntimeError("先に run() を実行してください。")
        counts: dict[str, int] = {}
        for symbol in atoms.get_chemical_symbols():
            counts[symbol] = counts.get(symbol, 0) + 1
        return counts

    def plot_mcmd(self, filename: str = "mcmd.png", *, dpi: int = 150) -> Path:
        """受理率と濃度の時間変化を描く。"""
        from ..plotting import label, setup_matplotlib

        setup_matplotlib()
        import matplotlib.pyplot as plt

        frame = self.mcmd()
        columns = [c for c in frame.columns if c.startswith("c_")]
        figure, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
        axes[0].plot(frame["step"], frame["acceptance"], lw=1.0)
        axes[0].set_ylabel(label("MC 受理率"))
        axes[0].set_ylim(0, 1)
        for column in columns:
            axes[1].plot(frame["step"], frame[column], lw=1.2, label=column[2:])
        axes[1].set_xlabel(label("MD ステップ"))
        axes[1].set_ylabel(label("濃度"))
        if columns:
            axes[1].legend(fontsize=8)
        figure.suptitle(label(f"MC/MD — {self.name}"))
        figure.tight_layout()
        output = self.workdir / filename
        figure.savefig(output, dpi=dpi)
        plt.close(figure)
        return output
