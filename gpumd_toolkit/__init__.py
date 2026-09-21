"""GPUMD を Python から簡単に使いこなすためのツールキット。

主要クラス
----------
:class:`~gpumd_toolkit.md.GPUMDCalculation`
    GPUMD 本体で MD (NVE / NVT / NPT、昇温・降温・任意温度プロファイル) を回す。
:class:`~gpumd_toolkit.examples.dpmd.DPMDExample`
    DPMD の example を実行し、予実 (期待値 vs 実測値) をとる。
:class:`~gpumd_toolkit.ase_interface.ASEMDRunner`
    ASE の calculator / 積分器で NEP を使う (CPU 版 NEP_CPU・pyNEP にも対応)。

補助
----
:class:`~gpumd_toolkit.structure.StructureHandler`   POSCAR / CIF などの入出力
:mod:`~gpumd_toolkit.fastio`                        構造ファイルの高速読み込み
:class:`~gpumd_toolkit.profiles.TemperatureProfile`  温度プロファイル
:class:`~gpumd_toolkit.trajectory.TrajectoryConverter` XDATCAR / traj への変換
:class:`~gpumd_toolkit.analysis.MDAnalyzer`          thermo.out の解析と作図
:class:`~gpumd_toolkit.parallel.ParallelRunner`      joblib による並列実行
:class:`~gpumd_toolkit.validation.ValidationReport`  予実表
"""

from __future__ import annotations

__version__ = "0.1.0"

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
from .inputs import DumpSettings, MDStage, RunInputBuilder
from .md import GPUMDCalculation, GPUMDResult
from .parallel import JobSpec, ParallelRunner
from .profiles import TemperatureProfile, TemperatureSegment
from .structure import StructureHandler
from .trajectory import TrajectoryConverter
from .validation import Expectation, ValidationReport

__all__ = [
    "__version__",
    "ASEMDRunner",
    "DumpSettings",
    "Expectation",
    "GPUMDCalculation",
    "GPUMDEnvironment",
    "GPUMDResult",
    "JobSpec",
    "MDAnalyzer",
    "MDStage",
    "ParallelRunner",
    "RunInputBuilder",
    "StructureHandler",
    "TemperatureProfile",
    "TemperatureSegment",
    "ThermoData",
    "TrajectoryConverter",
    "ValidationReport",
    "available_backends",
    "axes_to_mask",
    "benchmark_readers",
    "cif_to_xyz",
    "clear_cache",
    "compare_runs",
    "convert_to_xyz",
    "create_nep_calculator",
    "environment_report",
    "format_report",
    "poscar_to_xyz",
    "read_fast",
]


def __getattr__(name: str):
    # examples は calorine/GPUMD 依存が重いので遅延 import する
    if name in ("DPMDExample", "DPMDReference"):
        from .examples.dpmd import DPMDExample, DPMDReference

        return {"DPMDExample": DPMDExample, "DPMDReference": DPMDReference}[name]
    raise AttributeError(name)
