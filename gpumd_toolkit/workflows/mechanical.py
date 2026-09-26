"""11. 機械的・外場・特殊操作.

========================================== ==================================
操作                                        GPUMD キーワード
========================================== ==================================
連続変形 (引張・圧縮・せん断)                ``deform``
セルの一括変形                               ``change_box``
原子の固定 / 等速駆動                        ``fix`` / ``move``
一定外力 / 電場                              ``add_force`` / ``add_efield``
ばね拘束 (steered MD)                        ``add_spring``
成膜・照射で原子を追加                       ``deposit``
電子阻止能 (照射損傷)                        ``electron_stop``
拡張サンプリング (メタダイナミクス等)        ``plumed``
分散力補正                                   ``dftd3``
運動量のリセット                             ``correct_velocity``
========================================== ==================================

Examples
--------
>>> from gpumd_toolkit.workflows import MechanicalCalculation
>>> calc = MechanicalCalculation("POSCAR", "nep.txt", "runs/tensile")
>>> calc.equilibrate(temperature=300, steps=20000, pressure=0.0)
>>> calc.tensile(temperature=300, steps=500000, strain_rate=1e8, axis="x")
>>> calc.run()
>>> calc.stress_strain().head()
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..groups import GroupingScheme, by_region
from ..inputs import modifiers
from ..md import GPUMDCalculation
from ..outputs import read_spring, read_thermo

__all__ = ["MechanicalCalculation"]

_AXIS_TO_VOIGT = {"x": "xx", "y": "yy", "z": "zz"}


class MechanicalCalculation(GPUMDCalculation):
    """変形・外場・特殊操作のワークフロー。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        #: 変形の軸 (応力-ひずみ曲線の作成に使う)
        self.deform_axis: str | None = None
        self._initial_length: float | None = None

    # ------------------------------------------------------------------ 準備
    def equilibrate(
        self,
        *,
        temperature: float,
        steps: int,
        pressure: float | None = 0.0,
        **kwargs,
    ) -> "MechanicalCalculation":
        """変形前に応力ゼロの状態へ緩和する。"""
        if pressure is None:
            self.nvt(temperature=temperature, steps=steps, **kwargs)
        else:
            self.npt(temperature=temperature, steps=steps, pressure=pressure, **kwargs)
        return self

    def add_grip_groups(
        self, *, axis: str = "x", thickness: float = 8.0
    ) -> "MechanicalCalculation":
        """両端に「つかみ部」のグループを作る (``fix`` / ``move`` 用)。

        グループ 0 = 下端、1 = 中央、2 = 上端。
        """
        index = {"x": 0, "y": 1, "z": 2}[axis]
        positions = self.atoms.get_positions()[:, index]
        low = float(positions.min()) + thickness
        high = float(positions.max()) - thickness
        if low >= high:
            raise ValueError("thickness が大きすぎて中央領域がありません。")
        scheme = self.grouping_scheme or GroupingScheme()
        scheme.add(by_region(self.atoms, [low, high], axis), "grips")
        self.grouping_scheme = scheme
        self.groupings = scheme.to_groupings()
        return self

    # ------------------------------------------------------------------ 変形
    def tensile(
        self,
        *,
        temperature: float,
        steps: int,
        strain_rate: float,
        axis: str = "x",
        ensemble: str = "npt_scr",
        lateral_pressure: float = 0.0,
        **kwargs,
    ) -> "MechanicalCalculation":
        """一軸引張 (負の ``strain_rate`` で圧縮)。

        Parameters
        ----------
        strain_rate
            ひずみ速度 [1/s]。時間刻みとセル長から Å/step に換算する。
        ensemble
            ``'npt_scr'`` なら横方向は圧力制御 (ポアソン収縮を許す)。
            ``'nvt_nhc'`` なら横方向も固定。

        Notes
        -----
        GPUMD の ``deform`` は等方 NPT と併用できない。3 成分指定の NPT を
        使い、変形する軸の圧力制御は GPUMD 側で自動的に無効になる。
        """
        if axis not in _AXIS_TO_VOIGT:
            raise ValueError("axis は 'x' / 'y' / 'z' です。")
        index = {"x": 0, "y": 1, "z": 2}[axis]
        length = float(np.linalg.norm(self.atoms.cell[index]))
        self._initial_length = length
        self.deform_axis = axis
        # ひずみ速度 [1/s] -> Å/step
        dt_s = self.builder.time_step * 1e-15
        rate = strain_rate * length * dt_s

        if ensemble in ("npt_scr", "npt_ber"):
            self.npt(
                temperature=temperature,
                steps=steps,
                pressure=(lateral_pressure,) * 3,
                barostat=ensemble,
                cell_mode="ortho",
                **kwargs,
            )
        else:
            self.nvt(temperature=temperature, steps=steps, thermostat=ensemble, **kwargs)
        return self.add_commands(modifiers.deform({_AXIS_TO_VOIGT[axis]: rate}))

    def shear(
        self,
        *,
        temperature: float,
        steps: int,
        rate: float,
        component: str = "xy",
        ensemble: str = "nvt_nhc",
        **kwargs,
    ) -> "MechanicalCalculation":
        """せん断変形。``component`` は ``'xy'`` / ``'xz'`` / ``'yz'``。

        ``rate`` はセル成分の変化率 [Å/step]。
        """
        if component not in ("xy", "xz", "yz"):
            raise ValueError("component は 'xy' / 'xz' / 'yz' です。")
        self.deform_axis = component
        self.nvt(temperature=temperature, steps=steps, thermostat=ensemble, **kwargs)
        return self.add_commands(modifiers.deform({component: rate}))

    def change_box(self, delta: float | Sequence[float]) -> "MechanicalCalculation":
        """セルを一度だけ変形する (原子座標もアフィン変換される)。"""
        return self.add_commands(modifiers.change_box(delta))

    def pull_grips(
        self,
        *,
        temperature: float,
        steps: int,
        velocity: Sequence[float],
        fixed_group: int = 0,
        moving_group: int = 2,
        grouping_method: int | None = None,
        thermostat: str = "nvt_nhc",
        **kwargs,
    ) -> "MechanicalCalculation":
        """片端を固定し、もう片端を等速で引く (境界駆動の引張)。

        :meth:`add_grip_groups` でグループを作ってから呼ぶ。
        NPT とは併用できない (GPUMD の制限)。
        """
        self.ensure_grouping()
        self.nvt(temperature=temperature, steps=steps, thermostat=thermostat, **kwargs)
        return self.add_commands(
            modifiers.fix(fixed_group, grouping_method),
            modifiers.move(moving_group, velocity, grouping_method),
        )

    # ------------------------------------------------------------------ 外場
    def constant_force(
        self,
        *,
        group_id: int,
        force: Sequence[float] | str,
        grouping_method: int = 0,
    ) -> "MechanicalCalculation":
        """直前のステージでグループに一定の力 [eV/Å] を加える。"""
        self.ensure_grouping()
        return self.add_commands(modifiers.add_force(grouping_method, group_id, force))

    def electric_field(
        self,
        *,
        group_id: int,
        field: Sequence[float] | str,
        grouping_method: int = 0,
        mode: str | None = None,
    ) -> "MechanicalCalculation":
        """直前のステージでグループに電場 [V/Å] をかける。"""
        self.ensure_grouping()
        return self.add_commands(
            modifiers.add_efield(grouping_method, group_id, field, mode)
        )

    def spring(self, **kwargs) -> "MechanicalCalculation":
        """ばね拘束を足す (引数は :func:`gpumd_toolkit.inputs.modifiers.add_spring`)。

        ゴースト原子を等速で動かせば steered MD になる。
        """
        self.ensure_grouping()
        return self.add_commands(modifiers.add_spring(**kwargs))

    # -------------------------------------------------------- 成膜・照射・その他
    def deposition(self, **kwargs) -> "MechanicalCalculation":
        """成膜・照射のように原子を周期的に追加する。

        引数は :func:`gpumd_toolkit.inputs.modifiers.deposit` と同じ。
        ``run.in`` 中で 1 回だけ使え、``run`` も 1 つだけになる制約がある。
        """
        return self.add_commands(modifiers.deposit(**kwargs))

    def electron_stopping(self, filename: str) -> "MechanicalCalculation":
        """高エネルギー原子に電子阻止能を効かせる。"""
        return self.add_commands(modifiers.electron_stop(filename))

    def plumed(
        self, plumed_file: str = "plumed.dat", *, interval: int = 1, restart: bool = False
    ) -> "MechanicalCalculation":
        """PLUMED を呼ぶ (メタダイナミクスなどの拡張サンプリング)。"""
        return self.add_commands(modifiers.plumed(plumed_file, interval, restart))

    def dispersion_correction(
        self, functional: str, potential_cutoff: float = 12.0, cn_cutoff: float = 6.0
    ) -> "MechanicalCalculation":
        """NEP に DFT-D3 分散力補正を足す (学習時に入っていない場合のみ)。"""
        return self.add_preamble(
            modifiers.dftd3(functional, potential_cutoff, cn_cutoff)
        )

    def reset_momentum(
        self, interval: int = 50, grouping_method: int | None = None
    ) -> "MechanicalCalculation":
        """一定間隔で並進・回転運動量をゼロに戻す (長時間 MD のドリフト対策)。"""
        return self.add_preamble(modifiers.correct_velocity(interval, grouping_method))

    # ------------------------------------------------------------------ 結果
    def stress_strain(self) -> pd.DataFrame:
        """``thermo.out`` から応力-ひずみ曲線を作る。

        ひずみは変形軸のセル長の相対変化、応力は対応する圧力成分の符号反転。
        """
        if self.deform_axis is None:
            raise RuntimeError("変形ステージがありません。tensile() / shear() を使ってください。")
        frame = read_thermo(self.workdir / "thermo.out")
        if self.deform_axis in _AXIS_TO_VOIGT:
            voigt = _AXIS_TO_VOIGT[self.deform_axis]
            component = {"xx": ("ax", "ay", "az"), "yy": ("bx", "by", "bz"),
                         "zz": ("cx", "cy", "cz")}[voigt]
            lengths = np.linalg.norm(frame[list(component)].to_numpy(), axis=1)
            reference = self._initial_length or float(lengths[0])
            strain = lengths / reference - 1.0
        else:
            voigt = self.deform_axis
            # せん断: 傾き成分 / 対応する辺の長さ
            mapping = {"xy": ("bx", ("ax", "ay", "az")),
                       "xz": ("cx", ("ax", "ay", "az")),
                       "yz": ("cy", ("bx", "by", "bz"))}
            tilt_column, base = mapping[voigt]
            base_length = np.linalg.norm(frame[list(base)].to_numpy(), axis=1)
            strain = frame[tilt_column].to_numpy() / base_length
            strain = strain - strain[0]
        return pd.DataFrame(
            {
                "strain": strain,
                "stress_GPa": -frame[f"P{voigt}"].to_numpy(),
                "temperature": frame["temperature"].to_numpy(),
                "potential_energy": frame["potential_energy"].to_numpy(),
            }
        )

    def mechanical_properties(self, *, elastic_strain: float = 0.02) -> dict[str, float]:
        """応力-ひずみ曲線からヤング率・降伏応力・最大応力を見積もる。

        Parameters
        ----------
        elastic_strain
            弾性域とみなすひずみの上限 (この範囲を直線フィットする)。
        """
        curve = self.stress_strain()
        elastic = curve[curve["strain"].abs() <= elastic_strain]
        if len(elastic) < 3:
            raise ValueError("弾性域の点が足りません。elastic_strain を広げてください。")
        modulus = float(np.polyfit(elastic["strain"], elastic["stress_GPa"], 1)[0])
        peak = int(curve["stress_GPa"].idxmax())
        return {
            "young_modulus_GPa": modulus,
            "max_stress_GPa": float(curve["stress_GPa"].iloc[peak]),
            "strain_at_max": float(curve["strain"].iloc[peak]),
            "final_strain": float(curve["strain"].iloc[-1]),
        }

    def spring_forces(self, filename: str | None = None) -> pd.DataFrame:
        """``spring_gm*_g*_s*.out`` を読む (steered MD の仕事の計算に使う)。"""
        if filename is None:
            candidates = sorted(self.workdir.glob("spring_gm*_g*_s*.out"))
            if not candidates:
                raise FileNotFoundError(f"{self.workdir} にばねの出力がありません。")
            filename = candidates[0].name
        return read_spring(self.workdir / filename)

    # ------------------------------------------------------------------ 作図
    def plot_stress_strain(
        self, filename: str = "stress_strain.png", *, dpi: int = 150
    ) -> Path:
        """応力-ひずみ曲線を描く。"""
        from ..plotting import label, setup_matplotlib

        setup_matplotlib()
        import matplotlib.pyplot as plt

        curve = self.stress_strain()
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.plot(curve["strain"] * 100, curve["stress_GPa"], lw=1.4)
        try:
            properties = self.mechanical_properties()
            axis.plot(
                curve["strain"].iloc[: len(curve) // 20] * 100,
                properties["young_modulus_GPa"] * curve["strain"].iloc[: len(curve) // 20],
                "--",
                color="C3",
                label=f"E = {properties['young_modulus_GPa']:.0f} GPa",
            )
            axis.legend()
        except ValueError:
            pass
        axis.set_xlabel(label("ひずみ (%)"))
        axis.set_ylabel(label("応力 (GPa)"))
        axis.set_title(label(f"応力-ひずみ ({self.deform_axis}) — {self.name}"))
        figure.tight_layout()
        output = self.workdir / filename
        figure.savefig(output, dpi=dpi)
        plt.close(figure)
        return output
