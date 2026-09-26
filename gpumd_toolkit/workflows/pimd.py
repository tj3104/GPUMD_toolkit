"""8. PIMD・核量子効果.

軽元素 (H, Li, C) や低温では、原子核を古典粒子として扱う近似が破れる。
GPUMD は 2 系統の取り込み方を持つ。

======================================= ======================================
手法                                     特徴
======================================= ======================================
経路積分 MD (``pimd`` / ``pimd_scr``)     厳密。ビーズ数 P に比例して重い。
                                         熱力学量 (エネルギー・圧力) が正しい。
RPMD / TRPMD                             近似的な量子ダイナミクス。
                                         振動スペクトル・拡散係数に使う。
量子熱浴 (``nvt_qtb`` / ``npt_qtb``)      1 レプリカで済む安価な近似。
                                         零点エネルギーを色つきノイズで入れる。
======================================= ======================================

.. note::
   QTB では零点エネルギーの分だけ ``thermo.out`` の運動温度が目標温度より
   高く出る (水 300 K で 1000 K 前後)。これは異常ではない。

Examples
--------
>>> from gpumd_toolkit.workflows import PathIntegralCalculation
>>> calc = PathIntegralCalculation("water.xyz", "nep.txt", "runs/pimd")
>>> calc.pimd(num_beads=32, temperature=300, steps=100000)
>>> calc.dump_beads(interval=1000)
>>> calc.run()
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..inputs import dumps
from ..inputs.ensembles import PIMD, QTB, RPMD, TRPMD
from ..md import GPUMDCalculation

__all__ = ["PathIntegralCalculation"]


class PathIntegralCalculation(GPUMDCalculation):
    """経路積分 MD と量子熱浴で核量子効果を扱うワークフロー。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        #: 設定したビーズ数 (``dump_beads`` の解析に使う)
        self.num_beads: int | None = None

    # ------------------------------------------------------------------ PIMD
    def pimd(
        self,
        *,
        num_beads: int,
        temperature: float,
        steps: int,
        temperature_end: float | None = None,
        pressure: float | Sequence[float] | None = None,
        stochastic_cell_rescaling: bool = False,
        eco_omega_max: float | None = None,
        T_coup: float = 100.0,
        tau_T: float | None = None,
        p_coup: float = 1000.0,
        tau_p: float | None = None,
        cell_mode: str = "iso",
        elastic_modulus: float | Sequence[float] = 100.0,
        **kwargs,
    ) -> "PathIntegralCalculation":
        """経路積分 MD を仕込む。

        Parameters
        ----------
        num_beads
            ring polymer のビーズ数 (128 以下の正の偶数)。
            軽元素・低温ほど多く必要 (水 300 K なら 32 前後が目安)。
        pressure
            与えると NPT になる。``stochastic_cell_rescaling=True`` で
            ``pimd_scr`` (Berendsen ではなく確率的セルリスケーリング)。
        eco_omega_max
            economised path integral の最大振動数 [cm^-1]。
            同じ精度を少ないビーズ数で得たいときに指定する。

        Notes
        -----
        ``num_beads`` を設定できるのは最初の PIMD の run だけで、
        途中で変えることはできない。
        """
        spec = PIMD(
            num_beads=num_beads,
            T_start=temperature,
            T_end=temperature_end,
            T_coup=T_coup,
            tau_T=tau_T,
            pressure=pressure,
            elastic_modulus=elastic_modulus,
            p_coup=p_coup,
            tau_p=tau_p,
            cell_mode=cell_mode,
            stochastic_cell_rescaling=stochastic_cell_rescaling,
            eco_omega_max=eco_omega_max,
        )
        self.num_beads = num_beads
        return self.add_ensemble(spec, steps=steps, **kwargs)

    def rpmd(
        self, *, num_beads: int, steps: int, thermostatted: bool = False, **kwargs
    ) -> "PathIntegralCalculation":
        """ring-polymer MD。``thermostatted=True`` で内部モードに熱浴をかける (TRPMD)。

        PIMD で平衡化したあとに使う (RPMD 自体は熱浴を持たない)。
        """
        spec = TRPMD(num_beads) if thermostatted else RPMD(num_beads)
        self.num_beads = num_beads
        return self.add_ensemble(spec, steps=steps, **kwargs)

    # ------------------------------------------------------------------ QTB
    def qtb(
        self,
        *,
        temperature: float,
        steps: int,
        temperature_end: float | None = None,
        pressure: float | Sequence[float] | None = None,
        direction: str | Sequence[str] = "iso",
        f_max: float | None = None,
        n_f: int | None = None,
        T_coup: float = 100.0,
        tau_T: float | None = None,
        p_period: float = 1000.0,
        tau_p: float | None = None,
        **kwargs,
    ) -> "PathIntegralCalculation":
        """量子熱浴 (QTB)。1 レプリカで零点振動を近似的に取り込む。

        Parameters
        ----------
        f_max
            QTB フィルタの最大振動数 [ps^-1]。系の最大フォノン振動数より
            大きく取る (水なら 200 で十分)。
        pressure
            与えると ``npt_qtb`` (MTTK 圧浴つき) になる。
        """
        spec = QTB(
            T_start=temperature,
            T_end=temperature_end,
            mode="nvt" if pressure is None else "npt",
            T_coup=T_coup,
            tau_T=tau_T,
            direction=direction,
            pressure=0.0 if pressure is None else pressure,
            p_period=p_period,
            tau_p=tau_p,
            f_max=f_max,
            n_f=n_f,
        )
        return self.add_ensemble(spec, steps=steps, **kwargs)

    # ------------------------------------------------------------------ 出力
    def dump_beads(
        self, *, interval: int, velocity: bool = False, force: bool = False
    ) -> "PathIntegralCalculation":
        """各ビーズの座標を ``beads_dump_<k>.xyz`` に出力する。

        重心 (全ビーズの平均) は通常の ``dump_xyz`` が出す。
        """
        return self.add_commands(
            dumps.dump_beads(interval, has_velocity=velocity, has_force=force)
        )

    # ------------------------------------------------------------------ 結果
    def bead_files(self) -> list[Path]:
        """``beads_dump_<k>.xyz`` の一覧 (ビーズ番号順)。"""
        files = sorted(
            self.workdir.glob("beads_dump_*.xyz"),
            key=lambda p: int(p.stem.rsplit("_", 1)[1]),
        )
        if not files:
            raise FileNotFoundError(
                f"{self.workdir} に beads_dump_*.xyz がありません。"
                " dump_beads() を仕込んで実行してください。"
            )
        return files

    def bead_trajectories(self, index: int | str = -1):
        """各ビーズの構造を読む (既定は最終フレーム)。"""
        from ..fastio import read_xyz_fast

        return [read_xyz_fast(path, index=index) for path in self.bead_files()]

    def gyration_radius(self) -> pd.DataFrame:
        """最終フレームの ring polymer 回転半径 [Å] を原子ごとに返す。

        量子的な非局在の大きさの指標。水素は重原子よりはるかに大きくなる。
        """
        beads = self.bead_trajectories(index=-1)
        if len(beads) < 2:
            raise ValueError("ビーズが 1 つしかありません。")
        positions = np.array([atoms.get_positions() for atoms in beads])
        centroid = positions.mean(axis=0)
        radius = np.sqrt(((positions - centroid) ** 2).sum(axis=2).mean(axis=0))
        return pd.DataFrame(
            {
                "symbol": beads[0].get_chemical_symbols(),
                "gyration_radius": radius,
            }
        )

    def quantum_vs_classical(self, classical_thermo: pd.DataFrame) -> pd.DataFrame:
        """同条件の古典 MD と熱力学量を並べて比較する。

        Parameters
        ----------
        classical_thermo
            古典 MD の :func:`gpumd_toolkit.outputs.read_thermo` の結果。
        """
        from ..outputs import read_thermo

        quantum = read_thermo(self.workdir / "thermo.out")
        columns = ["temperature", "kinetic_energy", "potential_energy", "pressure"]
        rows = []
        for column in columns:
            rows.append(
                {
                    "quantity": column,
                    "quantum": float(quantum[column].mean()),
                    "classical": float(classical_thermo[column].mean()),
                    "difference": float(
                        quantum[column].mean() - classical_thermo[column].mean()
                    ),
                }
            )
        return pd.DataFrame(rows)
