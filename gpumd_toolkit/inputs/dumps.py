"""GPUMD の出力 (``dump_*`` / ``active``) キーワード。

各関数は ``run.in`` に書く 1 行を文字列で返す。引数はその場で検証されるので、
GPUMD を起動してから「引数の個数が違う」と怒られることがない。

:class:`DumpSettings` は「毎ステージで書き直す必要がある基本の出力設定」
(``dump_thermo`` / ``dump_xyz`` / ``dump_restart``) をまとめたもの。
それ以外の dump は :attr:`DumpSettings.extra` か
:attr:`~gpumd_toolkit.inputs.builder.MDStage.pre_commands` に足す。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

__all__ = [
    "DumpSettings",
    "DUMP_PROPERTIES",
    "NETCDF_PROPERTIES",
    "dump_thermo",
    "dump_restart",
    "dump_xyz",
    "dump_netcdf",
    "dump_beads",
    "dump_observer",
    "dump_dipole",
    "dump_polarizability",
    "dump_shock_nemd",
    "active",
]

#: ``dump_xyz`` で出力できる per-atom 量
DUMP_PROPERTIES = (
    "mass",
    "velocity",
    "force",
    "potential",
    "virial",
    "charge",
    "bec",
    "group_labels",
    "unwrapped_position",
)

#: ``dump_netcdf`` で出力できる per-atom 量
NETCDF_PROPERTIES = ("velocity", "force", "unwrapped_position")


def _positive(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} は正の整数です (与えられたのは {value})。")
    return value


def _group_args(group: Sequence[int] | None) -> str:
    """``group <grouping_method> <group_id>`` を組み立てる。"""
    if group is None:
        return ""
    pair = tuple(int(v) for v in group)
    if len(pair) != 2:
        raise ValueError("group は (grouping_method, group_id) の 2 要素で指定します。")
    if pair[0] < 0:
        raise ValueError("grouping_method は 0 以上です。")
    return f"group {pair[0]} {pair[1]}"


# ------------------------------------------------------------------ 基本 3 種
def dump_thermo(interval: int) -> str:
    """``thermo.out`` (T, K, U, 応力 6 成分, セル 9 成分) を出力する。"""
    return f"dump_thermo {_positive(interval, 'interval')}"


def dump_restart(interval: int) -> str:
    """``restart.xyz`` を出力する (上書き)。リネームすれば ``model.xyz`` として再開できる。"""
    return f"dump_restart {_positive(interval, 'interval')}"


def dump_xyz(
    interval: int,
    filename: str = "dump.xyz",
    *,
    properties: Sequence[str] = (),
    precision: str = "single",
    group: Sequence[int] | None = None,
) -> str:
    """拡張 XYZ 形式でトラジェクトリを出力する。

    Parameters
    ----------
    interval
        出力間隔 [ステップ]。
    filename
        出力ファイル名。末尾が ``*`` ならフレームごとに 1 ファイル
        (``*`` がステップ数に置換される)。
    properties
        追加で書く per-atom 量 (:data:`DUMP_PROPERTIES`)。座標は常に出力される。
    precision
        ``'single'`` (9 桁) または ``'double'`` (17 桁)。
    group
        ``(grouping_method, group_id)``。指定するとそのグループだけ出力する。
    """
    bad = [p for p in properties if p not in DUMP_PROPERTIES]
    if bad:
        raise ValueError(
            f"dump_xyz で未対応の property です: {bad}\n"
            f"  使用可能: {', '.join(DUMP_PROPERTIES)}"
        )
    if precision not in ("single", "double"):
        raise ValueError("precision は 'single' か 'double' です。")
    parts = [f"dump_xyz {_positive(interval, 'interval')} {filename}"]
    group_args = _group_args(group)
    if group_args:
        parts.append(group_args)
    if precision != "single":
        parts.append(f"precision {precision}")
    if properties:
        parts.append(" ".join(properties))
    return " ".join(parts)


def dump_netcdf(
    interval: int,
    filename: str = "movie.nc",
    *,
    properties: Sequence[str] = (),
    precision: str = "single",
    group: Sequence[int] | None = None,
    compression: int | None = None,
) -> str:
    """AMBER NetCDF 形式でトラジェクトリを出力する (大規模系で xyz より軽い)。"""
    bad = [p for p in properties if p not in NETCDF_PROPERTIES]
    if bad:
        raise ValueError(
            f"dump_netcdf で未対応の property です: {bad}"
            f" (使用可能: {', '.join(NETCDF_PROPERTIES)})"
        )
    if precision not in ("single", "double"):
        raise ValueError("precision は 'single' か 'double' です。")
    parts = [f"dump_netcdf {_positive(interval, 'interval')} {filename}"]
    group_args = _group_args(group)
    if group_args:
        parts.append(group_args)
    if properties:
        parts.append(" ".join(properties))
    if precision != "single":
        parts.append(f"precision {precision}")
    if compression is not None:
        level = int(compression)
        if not 1 <= level <= 9:
            raise ValueError("compression (deflate level) は 1-9 です。")
        parts.append(f"compression deflate {level}")
    return " ".join(parts)


def dump_beads(
    interval: int, *, has_velocity: bool = False, has_force: bool = False
) -> str:
    """PIMD の各ビーズの座標を ``beads_dump_<k>.xyz`` に出力する。

    重心 (全ビーズ平均) は通常どおり :func:`dump_xyz` が出力する。
    """
    return (
        f"dump_beads {_positive(interval, 'interval')} "
        f"{int(bool(has_velocity))} {int(bool(has_force))}"
    )


def dump_observer(
    mode: str,
    interval_thermo: int,
    interval_exyz: int,
    *,
    has_velocity: bool = True,
    has_force: bool = True,
) -> str:
    """複数の NEP でエネルギー・力を「観測」する (committee/アンサンブル評価)。

    Parameters
    ----------
    mode
        ``'observe'``: 1 番目のポテンシャルで MD を進めつつ、全ポテンシャルの
        予測を ``observer<i>.out`` / ``observer<i>.xyz`` に書く。
        ``'average'``: 全ポテンシャルの平均で MD を進め、``observer.out`` /
        ``observer.xyz`` に書く。
    """
    if mode not in ("observe", "average"):
        raise ValueError("dump_observer の mode は 'observe' か 'average' です。")
    return (
        f"dump_observer {mode} {_positive(interval_thermo, 'interval_thermo')} "
        f"{_positive(interval_exyz, 'interval_exyz')} "
        f"{int(bool(has_velocity))} {int(bool(has_force))}"
    )


def dump_dipole(interval: int, nep_file: str) -> str:
    """双極子モーメント予測 NEP で ``dipole.out`` を出力する (赤外スペクトル用)。"""
    return f"dump_dipole {_positive(interval, 'interval')} {nep_file}"


def dump_polarizability(interval: int, nep_file: str) -> str:
    """分極率予測 NEP で ``polarizability.out`` を出力する (ラマンスペクトル用)。"""
    return f"dump_polarizability {_positive(interval, 'interval')} {nep_file}"


def dump_shock_nemd(interval: int, bin_size: float = 10.0) -> str:
    """NEMD 衝撃波の x 方向空間分布を出力する。

    ``temperature_hist.txt`` / ``pxx_hist.txt`` / ``pyy_hist.txt`` /
    ``pzz_hist.txt`` / ``density_hist.txt`` / ``vp_hist.txt`` が作られる。
    """
    if bin_size <= 0:
        raise ValueError("bin_size [Å] は正の値です。")
    return (
        f"dump_shock_nemd interval {_positive(interval, 'interval')} "
        f"bin_size {bin_size:g}"
    )


def active(
    interval: int,
    threshold: float,
    *,
    has_velocity: bool = True,
    has_force: bool = True,
    has_uncertainty: bool = True,
) -> str:
    """NEP committee による on-the-fly active learning。

    複数の ``potential`` 行を並べておくと、力の標準偏差 (不確かさ) が
    ``threshold`` [eV/Å] を超えた構造を ``active.xyz`` に追記する。
    ``active.out`` には時刻と不確かさが毎回書かれる。
    MD は **1 番目の** ポテンシャルで進む。
    """
    if threshold < 0:
        raise ValueError("threshold [eV/Å] は 0 以上です。")
    return (
        f"active {_positive(interval, 'interval')} {int(bool(has_velocity))} "
        f"{int(bool(has_force))} {int(bool(has_uncertainty))} {threshold:g}"
    )


@dataclass
class DumpSettings:
    """1 ステージ分の基本出力設定。

    GPUMD の ``dump_*`` は propagating でない (次の ``run`` に引き継がれない)
    ので、ステージごとに書き直す必要がある。本クラスはそれを自動化する。

    Parameters
    ----------
    thermo_interval
        ``thermo.out`` への出力間隔 [ステップ]。``None`` で出力しない。
    traj_interval
        トラジェクトリ (``dump_xyz``) の出力間隔。``None`` で出力しない。
    traj_file
        トラジェクトリのファイル名。
    traj_properties
        トラジェクトリに含める per-atom 量。座標は常に出力される。
    traj_precision
        ``'single'`` (既定, 9 桁) または ``'double'`` (17 桁)。
    traj_group
        ``(grouping_method, group_id)``。指定したグループだけ出力する。
    restart_interval
        ``restart.xyz`` の出力間隔。``None`` で出力しない。
    extra
        そのまま ``run.in`` に差し込む追加キーワード行。
        :mod:`~gpumd_toolkit.inputs.computes` の関数の戻り値を並べるとよい。
    """

    thermo_interval: int | None = 100
    traj_interval: int | None = 1000
    traj_file: str = "dump.xyz"
    traj_properties: Sequence[str] = ("velocity", "force", "potential")
    traj_precision: str = "single"
    traj_group: Sequence[int] | None = None
    restart_interval: int | None = None
    extra: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        bad = [p for p in self.traj_properties if p not in DUMP_PROPERTIES]
        if bad:
            raise ValueError(
                f"dump_xyz で未対応の property です: {bad}\n"
                f"  使用可能: {', '.join(DUMP_PROPERTIES)}"
            )
        if self.traj_precision not in ("single", "double"):
            raise ValueError("traj_precision は 'single' か 'double' です。")

    def to_lines(self) -> list[str]:
        lines: list[str] = []
        if self.thermo_interval:
            lines.append(dump_thermo(self.thermo_interval))
        if self.traj_interval:
            lines.append(
                dump_xyz(
                    self.traj_interval,
                    self.traj_file,
                    properties=tuple(self.traj_properties),
                    precision=self.traj_precision,
                    group=self.traj_group,
                )
            )
        if self.restart_interval:
            lines.append(dump_restart(self.restart_interval))
        lines.extend(self.extra)
        return lines

    def with_extra(self, *lines: str) -> "DumpSettings":
        """``extra`` に行を足した複製を返す。"""
        from dataclasses import replace

        return replace(self, extra=tuple(self.extra) + tuple(lines))
