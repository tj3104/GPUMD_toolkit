"""12. NEP の不確かさ・Active Learning.

NEP を実用に耐える精度まで育てるには「MD を回して、学習集合から外挿して
いる構造を見つけ、それを DFT で計算して学習集合に足す」ループを回す。
GPUMD はこれを 2 通りで支援する。

============================================== ==============================
方式                                            特徴
============================================== ==============================
committee (``active``)                          複数の NEP の力のばらつきを
                                                不確かさとする。モデルを複数
                                                学習する必要がある。
外挿グレード (``compute_extrapolation``)        1 つの NEP と active set
                                                (ASI ファイル) から gamma を
                                                計算する。安価。
観測モード (``dump_observer``)                  複数 NEP の予測を並べて記録し、
                                                後からばらつきを解析する。
============================================== ==============================

``active`` と ``dump_observer`` は ``potential`` 行を複数書く必要がある。
その場合 :class:`~gpumd_toolkit.md.GPUMDCalculation` には
``potential=[["nep0.txt"], ["nep1.txt"], ...]`` の形で渡す。

Examples
--------
>>> from gpumd_toolkit.workflows import ActiveLearningCalculation
>>> calc = ActiveLearningCalculation(
...     "POSCAR", [["nep0.txt"], ["nep1.txt"], ["nep2.txt"]], "runs/al")
>>> calc.committee(temperature=1000, steps=100000, threshold=0.05, interval=10)
>>> calc.run(check=False)      # gamma_high で途中停止しても失敗扱いにしない
>>> calc.uncertainty_summary()
>>> calc.selected_structures()
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from ..inputs import dumps, modifiers
from ..md import GPUMDCalculation
from ..outputs import read_active
from ..postprocess import committee_uncertainty_summary

__all__ = ["ActiveLearningCalculation"]


class ActiveLearningCalculation(GPUMDCalculation):
    """NEP の不確かさ評価と on-the-fly active learning。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._threshold: float | None = None

    @property
    def n_potentials(self) -> int:
        """``run.in`` に書かれる ``potential`` 行の数。"""
        return self.builder.n_potentials

    # ------------------------------------------------------------ committee
    def committee(
        self,
        *,
        temperature: float,
        steps: int,
        threshold: float,
        interval: int = 10,
        temperature_end: float | None = None,
        pressure: float | None = None,
        thermostat: str = "nvt_nhc",
        has_velocity: bool = True,
        has_force: bool = True,
        has_uncertainty: bool = True,
        **kwargs,
    ) -> "ActiveLearningCalculation":
        """複数 NEP の力のばらつきを不確かさとして構造を選別する。

        不確かさが ``threshold`` [eV/Å] を超えた構造が ``active.xyz`` に
        追記される。MD は **1 番目の** ポテンシャルで進む。

        Parameters
        ----------
        threshold
            選別のしきい値 [eV/Å]。学習集合の力の RMSE の 2-3 倍が目安。
        """
        if self.n_potentials < 2:
            raise ValueError(
                "committee には 2 つ以上のポテンシャルが必要です。"
                ' potential=[["nep0.txt"], ["nep1.txt"], ...] の形で渡してください。'
            )
        if pressure is None:
            self.nvt(
                temperature=temperature,
                temperature_end=temperature_end,
                steps=steps,
                thermostat=thermostat,
                **kwargs,
            )
        else:
            self.npt(
                temperature=temperature,
                temperature_end=temperature_end,
                steps=steps,
                pressure=pressure,
                **kwargs,
            )
        self._threshold = threshold
        return self.add_commands(
            dumps.active(
                interval,
                threshold,
                has_velocity=has_velocity,
                has_force=has_force,
                has_uncertainty=has_uncertainty,
            )
        )

    def observe(
        self,
        *,
        temperature: float,
        steps: int,
        mode: str = "observe",
        interval_thermo: int = 100,
        interval_exyz: int = 1000,
        thermostat: str = "nvt_nhc",
        **kwargs,
    ) -> "ActiveLearningCalculation":
        """複数 NEP の予測を並べて記録する (``dump_observer``)。

        ``mode='observe'`` なら 1 番目のポテンシャルで MD を進めつつ
        全ポテンシャルの予測を記録、``'average'`` なら平均で MD を進める。
        """
        if self.n_potentials < 2:
            raise ValueError("dump_observer には 2 つ以上のポテンシャルが必要です。")
        self.nvt(temperature=temperature, steps=steps, thermostat=thermostat, **kwargs)
        return self.add_commands(
            dumps.dump_observer(mode, interval_thermo, interval_exyz)
        )

    # ------------------------------------------------------- 外挿グレード
    def extrapolation(
        self,
        *,
        temperature: float,
        steps: int,
        nep_file: str,
        asi_file: str,
        gamma_low: float = 5.0,
        gamma_high: float | None = 10.0,
        check_interval: int = 10,
        dump_interval: int = 10,
        thermostat: str = "nvt_nhc",
        **kwargs,
    ) -> "ActiveLearningCalculation":
        """active set からの外挿グレード gamma で構造を選別する。

        ``gamma_low`` を超えた構造は ``extrapolation_dump.xyz`` に書かれ、
        ``gamma_high`` を超えると MD が停止する (構造の破綻を防ぐ)。
        ``asi_file`` は nep_active (https://github.com/psn417/nep_active) で作る。
        """
        self.nvt(temperature=temperature, steps=steps, thermostat=thermostat, **kwargs)
        return self.add_commands(
            modifiers.compute_extrapolation(
                nep_file,
                asi_file,
                gamma_low=gamma_low,
                gamma_high=gamma_high,
                check_interval=check_interval,
                dump_interval=dump_interval,
            )
        )

    def explore(
        self,
        *,
        T_start: float,
        T_end: float,
        steps: int,
        threshold: float,
        interval: int = 10,
        pressure: float | None = None,
        **kwargs,
    ) -> "ActiveLearningCalculation":
        """昇温しながら未知領域を探索する (active learning の定石)。

        温度を上げるほど学習集合から外れた構造に出会いやすい。
        """
        return self.committee(
            temperature=T_start,
            temperature_end=T_end,
            steps=steps,
            threshold=threshold,
            interval=interval,
            pressure=pressure,
            **kwargs,
        )

    # ------------------------------------------------------------------ 結果
    def uncertainty(self) -> pd.DataFrame:
        """``active.out`` を読む (時刻 [fs] と不確かさ [eV/Å])。"""
        return read_active(self.workdir / "active.out")

    def uncertainty_summary(self, *, threshold: float | None = None) -> dict[str, float]:
        """不確かさの分布をまとめる (平均・中央値・95 パーセンタイル・超過割合)。"""
        return committee_uncertainty_summary(
            self.uncertainty(), threshold=threshold if threshold is not None else self._threshold
        )

    def selected_structures(self, index: str | int = ":"):
        """``active.xyz`` に選ばれた構造を読む (DFT 計算に回す候補)。"""
        from ase.io import read

        path = self.workdir / "active.xyz"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} がありません。"
                " しきい値を超える構造が現れなかったか、まだ実行していません。"
            )
        return read(str(path), index=index, format="extxyz")

    def extrapolated_structures(self, index: str | int = ":"):
        """``extrapolation_dump.xyz`` に選ばれた構造を読む。"""
        from ase.io import read

        path = self.workdir / "extrapolation_dump.xyz"
        if not path.is_file():
            raise FileNotFoundError(f"{path} がありません。")
        return read(str(path), index=index, format="extxyz")

    def observer_spread(self) -> pd.DataFrame:
        """``observer*.out`` を並べてポテンシャル間のばらつきを見る。"""
        from ..outputs import read_thermo

        files = sorted(self.workdir.glob("observer*.out"))
        if not files:
            raise FileNotFoundError(f"{self.workdir} に observer*.out がありません。")
        frames = {}
        for path in files:
            frame = read_thermo(path)
            frames[path.stem] = frame["potential_energy"] / self.n_atoms
        table = pd.DataFrame(frames)
        table["mean"] = table.mean(axis=1)
        table["std"] = table.std(axis=1, ddof=0)
        return table

    def write_candidates(
        self, output: Path | str = "candidates.xyz", *, stride: int = 1
    ) -> Path:
        """選ばれた構造を間引いて 1 ファイルにまとめる (DFT 投入用)。"""
        from ase.io import write

        structures = self.selected_structures()
        if not isinstance(structures, list):
            structures = [structures]
        selected = structures[::stride]
        path = Path(output)
        if not path.is_absolute():
            path = self.workdir / path
        write(str(path), selected, format="extxyz")
        return path

    # ------------------------------------------------------------------ 作図
    def plot_uncertainty(
        self, filename: str = "uncertainty.png", *, dpi: int = 150
    ) -> Path:
        """不確かさの時系列とヒストグラムを描く。"""
        from ..plotting import label, setup_matplotlib

        setup_matplotlib()
        import matplotlib.pyplot as plt

        frame = self.uncertainty()
        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(frame["time"] * 1e-3, frame["uncertainty"], lw=0.8)
        if self._threshold is not None:
            axes[0].axhline(
                self._threshold, color="C3", ls="--", lw=1.0,
                label=f"しきい値 {self._threshold:g} eV/Å",
            )
            axes[0].legend()
        axes[0].set_xlabel(label("時間 (ps)"))
        axes[0].set_ylabel(r"$\sigma_f$ (eV/Å)")
        axes[1].hist(frame["uncertainty"], bins=60, color="C0")
        if self._threshold is not None:
            axes[1].axvline(self._threshold, color="C3", ls="--", lw=1.0)
        axes[1].set_xlabel(r"$\sigma_f$ (eV/Å)")
        axes[1].set_ylabel(label("出現回数"))
        axes[1].set_yscale("log")
        figure.suptitle(label(f"committee 不確かさ — {self.name}"))
        figure.tight_layout()
        output = self.workdir / filename
        figure.savefig(output, dpi=dpi)
        plt.close(figure)
        return output
