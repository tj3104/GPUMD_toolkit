"""GPUMD を Python から使いこなすためのツールキット。

GPUMD が ``run.in`` で提供する機能を、分類ごとのワークフロークラスとして
一通り Python API にしてある。

主要クラス (機能分類は mission.md に対応)
------------------------------------------
:class:`~gpumd_toolkit.md.GPUMDCalculation`
    2. 通常の MD (NVE / NVT / NPT / NPH、昇温・降温・任意温度プロファイル)。
    すべてのワークフローの基底クラス。
:class:`~gpumd_toolkit.workflows.static.StaticCalculation`
    1. 一点計算・構造最適化・凝集エネルギー曲線・弾性定数・フォノン。
:class:`~gpumd_toolkit.workflows.free_energy.FreeEnergyCalculation`
    3. 熱力学的積分 (Frenkel-Ladd / UF / 可逆スケーリング / 断熱スイッチング)。
:class:`~gpumd_toolkit.workflows.monte_carlo.MonteCarloCalculation`
    4. MC/MD (canonical / SGC / VCSGC)。
:class:`~gpumd_toolkit.workflows.transport.ThermalTransport`
    熱輸送 (Green-Kubo / HNEMD / HNEMDEC / NEMD / SHC / GKMA / HNEMA)。
:class:`~gpumd_toolkit.workflows.diffusion.DiffusionCalculation`
    6. 拡散・イオン伝導・粘性率・RDF/ADF・配向秩序。
:class:`~gpumd_toolkit.workflows.electronic.ElectronicCalculation`
    7. LSQT 電子輸送・赤外/ラマンスペクトル・分極・外部電場。
:class:`~gpumd_toolkit.workflows.pimd.PathIntegralCalculation`
    8. PIMD / RPMD / TRPMD / 量子熱浴 (QTB)。
:class:`~gpumd_toolkit.workflows.shock.ShockCalculation`
    9. MSST / NPHug / NEMD ピストン / 段階加圧。
:class:`~gpumd_toolkit.workflows.ttm.TwoTemperatureCalculation`
    10. 二温度モデル・電子阻止能。
:class:`~gpumd_toolkit.workflows.mechanical.MechanicalCalculation`
    11. 引張・せん断・固定/駆動・外力/電場・ばね・成膜・PLUMED。
:class:`~gpumd_toolkit.workflows.active_learning.ActiveLearningCalculation`
    12. committee 不確かさ・外挿グレード・observer。
:class:`~gpumd_toolkit.examples.dpmd.DPMDExample`
    DPMD の example を実行し、予実 (期待値 vs 実測値) をとる。
:class:`~gpumd_toolkit.ase_interface.ASEMDRunner`
    ASE の calculator / 積分器で NEP を使う (CPU 版 NEP_CPU・pyNEP にも対応)。

補助
----
:mod:`~gpumd_toolkit.inputs`                        すべての ``run.in`` キーワード
:mod:`~gpumd_toolkit.outputs`                       すべての出力ファイルの読み込み
:mod:`~gpumd_toolkit.postprocess`                   出力から物理量を取り出す後処理
:mod:`~gpumd_toolkit.groups`                        grouping method の組み立て
:class:`~gpumd_toolkit.structure.StructureHandler`  POSCAR / CIF などの入出力
:mod:`~gpumd_toolkit.fastio`                        構造ファイルの高速読み込み
:class:`~gpumd_toolkit.profiles.TemperatureProfile` 温度プロファイル
:class:`~gpumd_toolkit.trajectory.TrajectoryConverter` XDATCAR / traj への変換
:class:`~gpumd_toolkit.analysis.MDAnalyzer`         thermo.out の解析と作図
:class:`~gpumd_toolkit.parallel.ParallelRunner`     joblib による並列実行
:class:`~gpumd_toolkit.validation.ValidationReport` 予実表
"""

from __future__ import annotations

__version__ = "0.2.0"

from . import groups, inputs, outputs, postprocess
from .analysis import MDAnalyzer, ThermoData, compare_runs
from .ase_interface import (
    ASEMDRunner,
    available_backends,
    axes_to_mask,
    create_nep_calculator,
)
from .config import GPUMDEnvironment
from .diagnostics import environment_report, format_report
from .fastio import (
    benchmark_readers,
    cif_to_xyz,
    clear_cache,
    convert_to_xyz,
    poscar_to_xyz,
    read_fast,
)
from .groups import GroupingScheme, NEMDLayout, nemd_layout
from .inputs import DumpSettings, MDStage, RunInputBuilder
from .md import GPUMDCalculation, GPUMDResult
from .outputs import OutputReader
from .parallel import JobSpec, ParallelRunner
from .postprocess import ElasticModuli, elastic_moduli
from .profiles import TemperatureProfile, TemperatureSegment
from .structure import StructureHandler
from .trajectory import TrajectoryConverter
from .validation import Expectation, ValidationReport
from .workflows import (
    ActiveLearningCalculation,
    DiffusionCalculation,
    ElectronicCalculation,
    FreeEnergyCalculation,
    MechanicalCalculation,
    MonteCarloCalculation,
    PathIntegralCalculation,
    ShockCalculation,
    StaticCalculation,
    ThermalTransport,
    TwoTemperatureCalculation,
)

__all__ = [
    "__version__",
    # サブモジュール
    "groups",
    "inputs",
    "outputs",
    "postprocess",
    # 基本
    "ASEMDRunner",
    "DumpSettings",
    "ElasticModuli",
    "Expectation",
    "GPUMDCalculation",
    "GPUMDEnvironment",
    "GPUMDResult",
    "GroupingScheme",
    "JobSpec",
    "MDAnalyzer",
    "MDStage",
    "NEMDLayout",
    "OutputReader",
    "ParallelRunner",
    "RunInputBuilder",
    "StructureHandler",
    "TemperatureProfile",
    "TemperatureSegment",
    "ThermoData",
    "TrajectoryConverter",
    "ValidationReport",
    # ワークフロー
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
    # 関数
    "available_backends",
    "axes_to_mask",
    "benchmark_readers",
    "cif_to_xyz",
    "clear_cache",
    "compare_runs",
    "convert_to_xyz",
    "create_nep_calculator",
    "elastic_moduli",
    "environment_report",
    "format_report",
    "nemd_layout",
    "poscar_to_xyz",
    "read_fast",
]


def __getattr__(name: str):
    # examples は calorine/GPUMD 依存が重いので遅延 import する
    if name in ("DPMDExample", "DPMDReference"):
        from .examples.dpmd import DPMDExample, DPMDReference

        return {"DPMDExample": DPMDExample, "DPMDReference": DPMDReference}[name]
    raise AttributeError(name)
