"""``run.in`` を組み立てるためのサブパッケージ。

.. list-table::
   :header-rows: 1

   * - モジュール
     - 役割
   * - :mod:`~gpumd_toolkit.inputs.builder`
     - :class:`~gpumd_toolkit.inputs.builder.MDStage` /
       :class:`~gpumd_toolkit.inputs.builder.RunInputBuilder` — 全体の骨組み
   * - :mod:`~gpumd_toolkit.inputs.ensembles`
     - すべての ``ensemble`` (QTB / NEMD 熱浴 / TTM / PIMD / TI / 衝撃波)
   * - :mod:`~gpumd_toolkit.inputs.computes`
     - すべての ``compute_*``
   * - :mod:`~gpumd_toolkit.inputs.dumps`
     - すべての ``dump_*`` と ``active``
   * - :mod:`~gpumd_toolkit.inputs.modifiers`
     - セットアップ・外場・MC・最小化などその他のキーワード
"""

from __future__ import annotations

from . import computes, dumps, ensembles, modifiers
from .builder import (
    ALL_ENSEMBLES,
    CELL_MODES,
    DUMP_PROPERTIES,
    FROZEN_MODULUS,
    MTTK_AXES,
    MTTK_DIRECTIONS,
    NPH_ENSEMBLES,
    NPT_ENSEMBLES,
    NVT_ENSEMBLES,
    VOIGT_LABELS,
    DumpSettings,
    MDStage,
    RunInputBuilder,
)
from .computes import (
    compute,
    compute_adf,
    compute_angular_rdf,
    compute_chunk,
    compute_cohesive,
    compute_dos,
    compute_dpdt,
    compute_elastic,
    compute_gkma,
    compute_hac,
    compute_hnema,
    compute_hnemd,
    compute_hnemdec,
    compute_ic,
    compute_lsqt,
    compute_msd,
    compute_orientorder,
    compute_phonon,
    compute_rdf,
    compute_sdc,
    compute_shc,
    compute_viscosity,
)
from .dumps import (
    active,
    dump_beads,
    dump_dipole,
    dump_netcdf,
    dump_observer,
    dump_polarizability,
    dump_restart,
    dump_shock_nemd,
    dump_thermo,
    dump_xyz,
)
from .ensembles import (
    HEAT_ENSEMBLES,
    PIMD_ENSEMBLES,
    QTB_ENSEMBLES,
    SHOCK_ENSEMBLES,
    STANDARD_ENSEMBLES,
    TI_ENSEMBLES,
    TTM_ENSEMBLES,
    Ensemble,
    HeatBath,
    HeatTTM,
    MSST,
    NPHug,
    PIMD,
    QTB,
    RPMD,
    TIAdiabaticSwitching,
    TIFixedLambda,
    TILiquid,
    TIReversibleScaling,
    TISpring,
    TRPMD,
    TTM,
    TTMOptions,
    Wall,
)
from .modifiers import (
    add_efield,
    add_force,
    add_spring,
    change_box,
    compute_extrapolation,
    correct_velocity,
    deform,
    deposit,
    dftd3,
    electron_stop,
    fix,
    kspace,
    mc_canonical,
    mc_sgc,
    mc_vcsgc,
    minimize,
    move,
    plumed,
    potential,
    replicate,
    time_step,
    velocity,
)

__all__ = [
    # サブモジュール
    "builder", "computes", "dumps", "ensembles", "modifiers",
    # 骨組み
    "MDStage", "RunInputBuilder", "DumpSettings",
    # 定数
    "ALL_ENSEMBLES", "NVT_ENSEMBLES", "NPT_ENSEMBLES", "NPH_ENSEMBLES",
    "STANDARD_ENSEMBLES", "QTB_ENSEMBLES", "HEAT_ENSEMBLES", "TTM_ENSEMBLES",
    "PIMD_ENSEMBLES", "TI_ENSEMBLES", "SHOCK_ENSEMBLES",
    "CELL_MODES", "VOIGT_LABELS", "MTTK_DIRECTIONS", "MTTK_AXES",
    "FROZEN_MODULUS", "DUMP_PROPERTIES",
    # ensemble クラス
    "Ensemble", "QTB", "HeatBath", "TTM", "TTMOptions", "HeatTTM",
    "PIMD", "RPMD", "TRPMD", "TISpring", "TILiquid",
    "TIAdiabaticSwitching", "TIReversibleScaling", "TIFixedLambda",
    "MSST", "NPHug", "Wall",
    # compute_*
    "compute", "compute_chunk", "compute_adf", "compute_angular_rdf",
    "compute_cohesive", "compute_dos", "compute_dpdt", "compute_elastic",
    "compute_gkma", "compute_hac", "compute_hnema", "compute_hnemd",
    "compute_hnemdec", "compute_ic", "compute_lsqt", "compute_msd",
    "compute_orientorder", "compute_phonon", "compute_rdf", "compute_sdc",
    "compute_shc", "compute_viscosity",
    # dump_*
    "dump_thermo", "dump_restart", "dump_xyz", "dump_netcdf", "dump_beads",
    "dump_observer", "dump_dipole", "dump_polarizability", "dump_shock_nemd",
    "active",
    # modifiers
    "replicate", "velocity", "correct_velocity", "potential", "dftd3",
    "kspace", "time_step", "change_box", "deform", "fix", "move",
    "add_force", "add_efield", "add_spring", "deposit", "electron_stop",
    "plumed", "compute_extrapolation", "mc_canonical", "mc_sgc", "mc_vcsgc",
    "minimize",
]

from . import builder  # noqa: E402  (循環参照を避けるため最後に置く)
