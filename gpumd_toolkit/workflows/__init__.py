"""GPUMD の機能分類ごとの高レベルワークフロー。

どのクラスも :class:`gpumd_toolkit.md.GPUMDCalculation` のサブクラスなので、
構造の読み込み・スーパーセル化・``run.in`` の生成・実行・解析は共通の作法で使える。

==== ======================================== ====================================
番号 分類                                      クラス
==== ======================================== ====================================
1    静的計算・構造計算                        :class:`~.static.StaticCalculation`
2    通常の MD・アンサンブル                   :class:`gpumd_toolkit.md.GPUMDCalculation`
3    自由エネルギー・熱力学的積分              :class:`~.free_energy.FreeEnergyCalculation`
4    Monte Carlo・組成自由度                   :class:`~.monte_carlo.MonteCarloCalculation`
\\-    熱輸送 (GK / HNEMD / NEMD / SHC)          :class:`~.transport.ThermalTransport`
6    拡散・イオン伝導・液体物性                :class:`~.diffusion.DiffusionCalculation`
7    電子・電気的性質                          :class:`~.electronic.ElectronicCalculation`
8    PIMD・核量子効果                          :class:`~.pimd.PathIntegralCalculation`
9    高圧・衝撃波                              :class:`~.shock.ShockCalculation`
10   Two-Temperature Model                     :class:`~.ttm.TwoTemperatureCalculation`
11   機械的・外場・特殊操作                    :class:`~.mechanical.MechanicalCalculation`
12   NEP の不確かさ・Active Learning           :class:`~.active_learning.ActiveLearningCalculation`
==== ======================================== ====================================
"""

from __future__ import annotations

from .active_learning import ActiveLearningCalculation
from .diffusion import DiffusionCalculation
from .electronic import ElectronicCalculation
from .free_energy import FreeEnergyCalculation, melting_point_from_curves
from .mechanical import MechanicalCalculation
from .monte_carlo import MonteCarloCalculation
from .pimd import PathIntegralCalculation
from .shock import ShockCalculation
from .static import StaticCalculation, band_path_kpoints, write_kpoints
from .transport import ThermalTransport
from .ttm import (
    TwoTemperatureCalculation,
    write_electron_properties_file,
    write_electron_temperature_file,
)

#: 分類番号 -> ワークフロークラス
WORKFLOWS = {
    1: StaticCalculation,
    3: FreeEnergyCalculation,
    4: MonteCarloCalculation,
    5: ThermalTransport,
    6: DiffusionCalculation,
    7: ElectronicCalculation,
    8: PathIntegralCalculation,
    9: ShockCalculation,
    10: TwoTemperatureCalculation,
    11: MechanicalCalculation,
    12: ActiveLearningCalculation,
}

__all__ = [
    "ActiveLearningCalculation",
    "DiffusionCalculation",
    "ElectronicCalculation",
    "FreeEnergyCalculation",
    "MechanicalCalculation",
    "MonteCarloCalculation",
    "PathIntegralCalculation",
    "ShockCalculation",
    "StaticCalculation",
    "ThermalTransport",
    "TwoTemperatureCalculation",
    "WORKFLOWS",
    "band_path_kpoints",
    "melting_point_from_curves",
    "write_electron_properties_file",
    "write_electron_temperature_file",
    "write_kpoints",
]
