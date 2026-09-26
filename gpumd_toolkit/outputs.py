"""GPUMD が書き出すすべての出力ファイルの読み込み。

GPUMD の出力はほぼ「ヘッダなしの数値テーブル」なので、列の意味を知らないと
使えない。本モジュールは各ファイルに正しい列名を与えた
:class:`pandas.DataFrame` を返す。

``OutputReader`` を使うと、計算ディレクトリを 1 つ指定するだけで
そこにある出力を名前で引ける。

Examples
--------
>>> from gpumd_toolkit.outputs import OutputReader
>>> reader = OutputReader("runs/si_nvt")
>>> reader.available()
['thermo.out', 'dump.xyz', 'msd.out']
>>> reader.msd().head()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "OutputReader",
    "THERMO_COLUMNS",
    "read_table",
    "read_thermo",
    "read_compute",
    "read_compute_chunk",
    "read_cohesive",
    "read_elastic",
    "read_omega2",
    "read_dos",
    "read_mvac",
    "read_sdc",
    "read_msd",
    "read_ic",
    "read_rdf",
    "read_adf",
    "read_angular_rdf",
    "read_orientorder",
    "read_hac",
    "read_kappa",
    "read_onsager",
    "read_shc",
    "read_viscosity",
    "read_heatmode",
    "read_kappamode",
    "read_lsqt",
    "read_dipole",
    "read_polarizability",
    "read_dpdt",
    "read_active",
    "read_mcmd",
    "read_ti_csv",
    "read_ti_yaml",
    "read_ttm_electron_temperature",
    "read_shock_profiles",
    "read_spring",
]

#: ``thermo.out`` の 18 列
THERMO_COLUMNS = [
    "temperature", "kinetic_energy", "potential_energy",
    "Pxx", "Pyy", "Pzz", "Pyz", "Pxz", "Pxy",
    "ax", "ay", "az", "bx", "by", "bz", "cx", "cy", "cz",
]

_XYZ = ("x", "y", "z")


def _resolve(path: Path | str, default_name: str | None = None) -> Path:
    path = Path(path).expanduser()
    if path.is_dir() and default_name:
        path = path / default_name
    if not path.is_file():
        raise FileNotFoundError(f"出力ファイルが見つかりません: {path}")
    return path


def read_table(
    path: Path | str,
    columns: Sequence[str] | None = None,
    *,
    default_name: str | None = None,
    comment: str | None = "#",
) -> pd.DataFrame:
    """空白区切りの数値テーブルを読む (列名は任意)。"""
    resolved = _resolve(path, default_name)
    frame = pd.read_csv(
        resolved, sep=r"\s+", header=None, comment=comment, engine="python"
    )
    if columns is not None:
        if len(columns) != frame.shape[1]:
            raise ValueError(
                f"{resolved.name}: 列数が想定と違います "
                f"(想定 {len(columns)}, 実際 {frame.shape[1]})。"
            )
        frame.columns = list(columns)
    return frame


def _named(prefix: str, suffixes: Iterable[str]) -> list[str]:
    return [f"{prefix}_{s}" for s in suffixes]


# ------------------------------------------------------------------ 基本
def read_thermo(path: Path | str = "thermo.out") -> pd.DataFrame:
    """``thermo.out``: 温度・運動/ポテンシャルエネルギー・応力 6 成分・セル 9 成分。

    体積 ``volume`` と静水圧 ``pressure`` を派生列として足す。
    """
    frame = read_table(path, THERMO_COLUMNS, default_name="thermo.out")
    cell = frame[["ax", "ay", "az", "bx", "by", "bz", "cx", "cy", "cz"]].to_numpy()
    matrices = cell.reshape(-1, 3, 3)
    frame["volume"] = np.abs(np.linalg.det(matrices))
    frame["pressure"] = frame[["Pxx", "Pyy", "Pzz"]].mean(axis=1)
    frame["total_energy"] = frame["kinetic_energy"] + frame["potential_energy"]
    return frame


def read_compute(
    path: Path | str = "compute.out",
    *,
    n_groups: int,
    quantities: Sequence[str],
) -> pd.DataFrame:
    """``compute.out``: グループごとの空間・時間平均量。

    GPUMD の列の並びは ``compute`` キーワードでの指定順ではなく固定
    (temperature, potential, force, virial, jp, jk, momentum の順)。
    温度を含む場合、最後の 2 列は熱源・熱浴の熱浴エネルギーになる。

    Parameters
    ----------
    n_groups
        grouping method のグループ数。
    quantities
        ``compute`` に渡した量 (順不同でよい)。
    """
    order = ("temperature", "potential", "force", "virial", "jp", "jk", "momentum")
    requested = [q for q in order if q in set(quantities)]
    unknown = set(quantities) - set(order)
    if unknown:
        raise ValueError(f"compute で未対応の量です: {sorted(unknown)}")
    columns: list[str] = []
    for quantity in requested:
        if quantity in ("temperature", "potential"):
            columns += [f"{quantity}_g{g}" for g in range(n_groups)]
        elif quantity == "virial":
            for component in ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"):
                columns += [f"virial_{component}_g{g}" for g in range(n_groups)]
        else:  # force / jp / jk / momentum は x,y,z の順に M 列ずつ
            for axis in _XYZ:
                columns += [f"{quantity}_{axis}_g{g}" for g in range(n_groups)]
    if "temperature" in requested:
        columns += ["thermostat_energy_source", "thermostat_energy_sink"]
    return read_table(path, columns, default_name="compute.out")


def read_compute_chunk(
    path: Path | str = "compute_chunk.out",
    *,
    n_dims: int = 1,
    quantities: Sequence[str] = ("temperature",),
) -> pd.DataFrame:
    """``compute_chunk.out``: 動的空間ビンごとの平均量。

    ``frame`` 列 (何回目の出力か) を自動で付ける。
    """
    columns = ["chunk_id"] + [f"coord{i + 1}" for i in range(n_dims)] + ["count"]
    columns += list(quantities)
    frame = read_table(path, columns, default_name="compute_chunk.out")
    n_chunks = int(frame["chunk_id"].max()) + 1
    frame.insert(0, "frame", np.arange(len(frame)) // n_chunks)
    return frame


# ------------------------------------------------------------------ 静的計算
def read_cohesive(path: Path | str = "cohesive.out") -> pd.DataFrame:
    """``cohesive.out``: スケーリング係数とポテンシャルエネルギー [eV]。"""
    return read_table(path, ["scale", "energy"], default_name="cohesive.out")


def read_elastic(path: Path | str = "elastic.out") -> np.ndarray:
    """``elastic.out``: 6x6 の弾性定数行列 C_ij [GPa]。"""
    resolved = _resolve(path, "elastic.out")
    matrix = np.loadtxt(resolved, comments="#")
    if matrix.shape != (6, 6):
        raise ValueError(f"{resolved.name}: 6x6 行列を期待しましたが {matrix.shape} でした。")
    return matrix


def read_omega2(path: Path | str = "omega2.out") -> pd.DataFrame:
    """``omega2.out``: フォノン分散。

    1 列目が高対称線に沿った距離、残りが omega^2 [THz^2]。
    ``nu_<i>`` 列 (振動数 [THz]、虚数は負値で表す) を派生列として足す。
    """
    frame = read_table(path, default_name="omega2.out")
    n_branches = frame.shape[1] - 1
    frame.columns = ["distance"] + [f"omega2_{i}" for i in range(n_branches)]
    for i in range(n_branches):
        omega2 = frame[f"omega2_{i}"].to_numpy()
        frame[f"nu_{i}"] = np.sign(omega2) * np.sqrt(np.abs(omega2)) / (2 * np.pi)
    return frame


# ------------------------------------------------------------------ 振動・拡散
def _split_groups(frame: pd.DataFrame, n_groups: int, base: list[str]) -> pd.DataFrame:
    """``time`` + (base 列 * n_groups) の形にリネームする。"""
    expected = 1 + len(base) * n_groups
    if frame.shape[1] != expected:
        raise ValueError(
            f"列数が合いません (想定 {expected} = 1 + {len(base)}*{n_groups},"
            f" 実際 {frame.shape[1]})。n_groups を確認してください。"
        )
    columns = ["time"]
    for g in range(n_groups):
        suffix = "" if n_groups == 1 else f"_g{g}"
        columns += [f"{name}{suffix}" for name in base]
    frame.columns = columns
    return frame


def read_dos(
    path: Path | str = "dos.out", *, n_groups: int = 1
) -> pd.DataFrame:
    """``dos.out``: フォノン状態密度。列は角振動数 [THz] と x/y/z 成分 [1/THz]。

    複数グループの場合は縦に連結されているので ``group`` 列を足す。
    """
    frame = read_table(path, ["omega", "dos_x", "dos_y", "dos_z"], default_name="dos.out")
    if n_groups > 1:
        if len(frame) % n_groups:
            raise ValueError("行数が n_groups で割り切れません。")
        n_points = len(frame) // n_groups
        frame.insert(0, "group", np.repeat(np.arange(n_groups), n_points))
    frame["nu"] = frame["omega"] / (2 * np.pi)
    return frame


def read_mvac(path: Path | str = "mvac.out") -> pd.DataFrame:
    """``mvac.out``: 質量重み付き速度自己相関 (規格化済み)。"""
    return read_table(
        path, ["time", "vac_x", "vac_y", "vac_z"], default_name="mvac.out"
    )


def read_sdc(path: Path | str = "sdc.out", *, n_groups: int = 1) -> pd.DataFrame:
    """``sdc.out``: 速度自己相関 [Å^2/ps^2] と自己拡散係数 [Å^2/ps]。"""
    frame = read_table(path, default_name="sdc.out")
    base = _named("vac", _XYZ) + _named("sdc", _XYZ)
    return _split_groups(frame, n_groups, base)


def read_msd(path: Path | str = "msd.out", *, n_groups: int | None = None) -> pd.DataFrame:
    """``msd.out``: 平均二乗変位 [Å^2] と自己拡散係数 [Å^2/ps]。

    ``all_groups`` を使った場合はグループ数を列数から推定する。
    """
    frame = read_table(path, default_name="msd.out")
    if n_groups is None:
        n_groups = (frame.shape[1] - 1) // 6
    base = _named("msd", _XYZ) + _named("sdc", _XYZ)
    frame = _split_groups(frame, n_groups, base)
    if n_groups == 1:
        frame["msd_total"] = frame[["msd_x", "msd_y", "msd_z"]].sum(axis=1)
    return frame


def read_ic(path: Path | str = "ic.out") -> pd.DataFrame:
    """``ic.out``: イオン伝導度 [mS/cm]。"""
    return read_table(path, ["time", "ic_x", "ic_y", "ic_z"], default_name="ic.out")


# ------------------------------------------------------------------ 構造相関
def read_rdf(path: Path | str = "rdf.out", *, pairs: Sequence[str] = ()) -> pd.DataFrame:
    """``rdf.out``: 動径分布関数。2 列目が全体、3 列目以降が partial。"""
    frame = read_table(path, default_name="rdf.out")
    n_partial = frame.shape[1] - 2
    names = list(pairs) if pairs else [f"g_{i}" for i in range(n_partial)]
    if len(names) != n_partial:
        raise ValueError(f"partial RDF は {n_partial} 列ありますが pairs は {len(names)} 個です。")
    frame.columns = ["radius", "total"] + names
    return frame


def read_adf(path: Path | str = "adf.out", *, labels: Sequence[str] = ()) -> pd.DataFrame:
    """``adf.out``: 角度分布関数 (1 列目が角度 [度])。"""
    frame = read_table(path, default_name="adf.out")
    n_curves = frame.shape[1] - 1
    names = list(labels) if labels else (
        ["total"] if n_curves == 1 else [f"adf_{i}" for i in range(n_curves)]
    )
    frame.columns = ["angle"] + names
    return frame


def read_angular_rdf(path: Path | str = "angular_rdf.out") -> pd.DataFrame:
    """``angular_rdf.out``: 角度依存 RDF g(r, theta)。ヘッダ行から列名を拾う。"""
    resolved = _resolve(path, "angular_rdf.out")
    with resolved.open() as handle:
        first = handle.readline()
    if first.lstrip().startswith("#"):
        names = first.lstrip("#").split()
        return pd.read_csv(resolved, sep=r"\s+", comment="#", names=names, engine="python")
    frame = read_table(resolved)
    frame.columns = ["radius", "theta", "total"] + [
        f"partial_{i}" for i in range(frame.shape[1] - 3)
    ]
    return frame


def read_orientorder(path: Path | str = "orientorder.out") -> pd.DataFrame:
    """``orientorder.out``: Steinhardt 秩序変数 (ステップごとのブロック構造)。"""
    resolved = _resolve(path, "orientorder.out")
    frames: list[pd.DataFrame] = []
    step: int | None = None
    columns: list[str] | None = None
    rows: list[list[float]] = []

    def flush() -> None:
        if step is not None and columns and rows:
            block = pd.DataFrame(rows, columns=columns)
            block.insert(0, "atom", np.arange(len(block)))
            block.insert(0, "step", step)
            frames.append(block)

    with resolved.open() as handle:
        for line in handle:
            tokens = line.split()
            if not tokens:
                continue
            if len(tokens) == 1 and tokens[0].lstrip("-").isdigit():
                flush()
                step, columns, rows = int(tokens[0]), None, []
            elif columns is None:
                columns = tokens
            else:
                rows.append([float(v) for v in tokens])
    flush()
    if not frames:
        raise ValueError(f"{resolved.name} を解釈できませんでした。")
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------ 熱輸送
def read_hac(path: Path | str = "hac.out") -> pd.DataFrame:
    """``hac.out``: 熱流自己相関と Green-Kubo 熱伝導率 [W/mK]。

    ``kappa_x`` / ``kappa_y`` は in + out の和として派生列に足す。
    """
    columns = [
        "time",
        "jx_in", "jx_out", "jy_in", "jy_out", "jz",
        "kappa_x_in", "kappa_x_out", "kappa_y_in", "kappa_y_out", "kappa_z",
    ]
    frame = read_table(path, columns, default_name="hac.out")
    frame["kappa_x"] = frame["kappa_x_in"] + frame["kappa_x_out"]
    frame["kappa_y"] = frame["kappa_y_in"] + frame["kappa_y_out"]
    return frame


def read_kappa(path: Path | str = "kappa.out") -> pd.DataFrame:
    """``kappa.out``: HNEMD の熱伝導率 [W/mK] の時系列。

    累積平均 ``kappa_*_cum`` を派生列として足す (収束の確認に使う)。
    """
    columns = ["kappa_x_in", "kappa_x_out", "kappa_y_in", "kappa_y_out", "kappa_z"]
    frame = read_table(path, columns, default_name="kappa.out")
    frame["kappa_x"] = frame["kappa_x_in"] + frame["kappa_x_out"]
    frame["kappa_y"] = frame["kappa_y_in"] + frame["kappa_y_out"]
    for name in ("kappa_x", "kappa_y", "kappa_z"):
        frame[f"{name}_cum"] = frame[name].expanding().mean()
    return frame


def read_onsager(path: Path | str = "onsager.out") -> pd.DataFrame:
    """``onsager.out``: HNEMDEC の Onsager 係数 L。"""
    frame = read_table(path, default_name="onsager.out")
    if frame.shape[1] % 3:
        raise ValueError("列数が 3 の倍数ではありません。")
    n_blocks = frame.shape[1] // 3
    names = ["qq"] + [f"{i}q" for i in range(1, n_blocks)]
    frame.columns = [f"L_{axis}_{name}" for name in names for axis in _XYZ]
    return frame


def read_shc(
    path: Path | str = "shc.out", *, Nc: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``shc.out``: 前半が相関関数 K(t)、後半がスペクトル熱流 J_q(omega)。

    Returns
    -------
    (correlation, spectrum)
        ``correlation``: time [ps], K_in, K_out, K
        ``spectrum``: omega [THz], Jq_in, Jq_out, Jq
    """
    resolved = _resolve(path, "shc.out")
    frame = read_table(resolved, ["x", "in", "out"])
    values = frame["x"].to_numpy()

    split: int | None = None
    if Nc is not None:
        split = 2 * int(Nc) - 1
    else:
        # GPUMD は "# num_correlation_rows <n>" をヘッダに書く
        for line in resolved.read_text().splitlines():
            if not line.startswith("#"):
                break
            tokens = line.lstrip("#").split()
            if len(tokens) >= 2 and tokens[0] == "num_correlation_rows":
                split = int(tokens[1])
                break
    if split is None:
        # ヘッダが無い古い版: K(t) は -t から +t へ等間隔、omega はそこで刻みが変わる
        steps = np.diff(values)
        if steps.size < 3:
            raise ValueError("shc.out の行数が少なすぎます。")
        reference = steps[0]
        changed = np.flatnonzero(np.abs(steps - reference) > 1e-6 * max(abs(reference), 1.0))
        if changed.size == 0:
            raise ValueError(
                "K(t) と J_q(omega) の境目を検出できません。Nc を指定してください。"
            )
        split = int(changed[0]) + 1
    if not 0 < split < len(frame):
        raise ValueError(
            f"K(t) の行数 {split} が shc.out の行数 {len(frame)} と矛盾します。"
        )
    correlation = frame.iloc[:split].copy()
    correlation.columns = ["time", "K_in", "K_out"]
    correlation["K"] = correlation["K_in"] + correlation["K_out"]
    spectrum = frame.iloc[split:].reset_index(drop=True).copy()
    spectrum.columns = ["omega", "Jq_in", "Jq_out"]
    spectrum["Jq"] = spectrum["Jq_in"] + spectrum["Jq_out"]
    return correlation.reset_index(drop=True), spectrum


def read_viscosity(path: Path | str = "viscosity.out") -> pd.DataFrame:
    """``viscosity.out``: 応力自己相関と粘性率 [Pa s]。

    せん断粘性率 ``eta_shear``、縦粘性率 ``eta_longitudinal``、
    体積粘性率 ``eta_bulk`` を派生列として足す。
    """
    components = ("xx", "yy", "zz", "xy", "xz", "yz", "yx", "zx", "zy")
    columns = ["time"] + [f"sac_{c}" for c in components] + [f"eta_{c}" for c in components]
    frame = read_table(path, columns, default_name="viscosity.out")
    frame["eta_shear"] = frame[["eta_xy", "eta_xz", "eta_yz"]].mean(axis=1)
    frame["eta_longitudinal"] = frame[["eta_xx", "eta_yy", "eta_zz"]].mean(axis=1)
    frame["eta_bulk"] = frame["eta_longitudinal"] - 4.0 / 3.0 * frame["eta_shear"]
    return frame


def _modal(path: Path | str, default_name: str, prefix: str) -> pd.DataFrame:
    frame = read_table(path, default_name=default_name)
    if frame.shape[1] != 5:
        raise ValueError(f"{default_name}: 5 列を期待しましたが {frame.shape[1]} 列でした。")
    frame.columns = [
        f"{prefix}_x_in", f"{prefix}_x_out",
        f"{prefix}_y_in", f"{prefix}_y_out", f"{prefix}_z",
    ]
    frame[f"{prefix}_x"] = frame[f"{prefix}_x_in"] + frame[f"{prefix}_x_out"]
    frame[f"{prefix}_y"] = frame[f"{prefix}_y_in"] + frame[f"{prefix}_y_out"]
    return frame


def read_heatmode(path: Path | str = "heatmode.out") -> pd.DataFrame:
    """``heatmode.out``: GKMA のモード分解熱流。"""
    return _modal(path, "heatmode.out", "J")


def read_kappamode(path: Path | str = "kappamode.out") -> pd.DataFrame:
    """``kappamode.out``: HNEMA のモード分解熱伝導率 [W/mK]。"""
    return _modal(path, "kappamode.out", "kappa")


# ------------------------------------------------------------------ 電子・電気
def read_lsqt(
    directory: Path | str = ".",
    *,
    E_1: float | None = None,
    E_2: float | None = None,
) -> dict[str, pd.DataFrame | np.ndarray]:
    """``lsqt_dos.out`` / ``lsqt_velocity.out`` / ``lsqt_sigma.out`` をまとめて読む。

    各ファイルは「行 = 時刻、列 = エネルギー点」の行列なので、
    時間平均を取った ``mean`` 列つきの DataFrame も返す。
    """
    directory = Path(directory).expanduser()
    result: dict[str, Any] = {}
    for key, filename, unit in (
        ("dos", "lsqt_dos.out", "states/atom/eV"),
        ("velocity", "lsqt_velocity.out", "m/s"),
        ("sigma", "lsqt_sigma.out", "S/m"),
    ):
        path = directory / filename if directory.is_dir() else Path(filename)
        if not path.is_file():
            continue
        matrix = np.loadtxt(path, ndmin=2)
        frame = pd.DataFrame({"mean": matrix.mean(axis=0), "std": matrix.std(axis=0)})
        if E_1 is not None and E_2 is not None:
            frame.insert(0, "energy", np.linspace(E_1, E_2, matrix.shape[1]))
        frame.attrs["unit"] = unit
        result[key] = frame
        result[f"{key}_raw"] = matrix
    if not result:
        raise FileNotFoundError(f"{directory} に lsqt_*.out がありません。")
    return result


def read_dipole(path: Path | str = "dipole.out") -> pd.DataFrame:
    """``dipole.out``: 双極子モーメント (赤外スペクトルの元データ)。"""
    return read_table(path, ["step", "mu_x", "mu_y", "mu_z"], default_name="dipole.out")


def read_polarizability(path: Path | str = "polarizability.out") -> pd.DataFrame:
    """``polarizability.out``: 分極率テンソル (ラマンスペクトルの元データ)。"""
    return read_table(
        path,
        ["step", "p_xx", "p_yy", "p_zz", "p_xy", "p_yz", "p_zx"],
        default_name="polarizability.out",
    )


def read_dpdt(path: Path | str = "dpdt.out") -> pd.DataFrame:
    """``dpdt.out``: 分極の時間微分 dP/dt [e A/fs] と分極 P [e A]。"""
    return read_table(
        path,
        ["time", "dPx_dt", "dPy_dt", "dPz_dt", "Px", "Py", "Pz"],
        default_name="dpdt.out",
    )


# ------------------------------------------------------- Active learning / MC
def read_active(path: Path | str = "active.out") -> pd.DataFrame:
    """``active.out``: 時刻 [fs] と committee 不確かさ [eV/Å]。"""
    return read_table(path, ["time", "uncertainty"], default_name="active.out")


def read_mcmd(
    path: Path | str = "mcmd.out", *, species: Sequence[str] = ()
) -> pd.DataFrame:
    """``mcmd.out``: MD ステップ・MC 受理率・各元素の濃度。"""
    frame = read_table(path, default_name="mcmd.out")
    n_species = frame.shape[1] - 2
    names = list(species) if species else [f"c_{i}" for i in range(n_species)]
    if len(names) != n_species:
        raise ValueError(f"濃度は {n_species} 列ありますが species は {len(names)} 個です。")
    frame.columns = ["step", "acceptance"] + [f"c_{s}" for s in names]
    return frame


# ------------------------------------------------------------ 熱力学的積分
def read_ti_csv(path: Path | str) -> pd.DataFrame:
    """``ti_spring.csv`` / ``ti_liquid.csv`` / ``ti_rs.csv`` / ``ti_as.csv`` / ``ti.csv``。

    どれもヘッダ付き CSV。往路と復路が縦に連結されている
    (``ti.csv`` を除く)。
    """
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"TI の出力が見つかりません: {resolved}")
    return pd.read_csv(resolved)


def read_ti_yaml(path: Path | str) -> dict[str, float]:
    """``ti_spring.yaml`` / ``ti_liquid.yaml``: 自由エネルギーのまとめ。

    ``E_Einstein`` / ``E_UFmodel`` (参照系)、``E_diff`` (差分)、
    ``F`` (Helmholtz)、``G`` (Gibbs)、``T``、``V``、``P`` [eV/Å^3]。
    """
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"TI の出力が見つかりません: {resolved}")
    data: dict[str, float] = {}
    for line in resolved.read_text().splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        try:
            data[key.strip()] = float(value.strip())
        except ValueError:
            continue
    return data


# ------------------------------------------------------------------ TTM / 衝撃波
@dataclass
class TTMSnapshots:
    """``ttm_electron_temperature.out`` の中身。"""

    grid: tuple[int, int, int]
    steps: list[int]
    #: ``temperatures[i]`` が i 番目のスナップショットの (nx, ny, nz) 配列 [K]
    temperatures: list[np.ndarray]
    header: dict[str, str]

    def as_frame(self) -> pd.DataFrame:
        """``step, ix, iy, iz, T_e`` の長形式 DataFrame。"""
        rows = []
        nx, ny, nz = self.grid
        for step, array in zip(self.steps, self.temperatures):
            for iz in range(nz):
                for iy in range(ny):
                    for ix in range(nx):
                        rows.append((step, ix + 1, iy + 1, iz + 1, array[ix, iy, iz]))
        return pd.DataFrame(rows, columns=["step", "ix", "iy", "iz", "T_e"])

    def profile(self, axis: str = "z") -> pd.DataFrame:
        """指定軸に沿った電子温度プロファイル (他の軸は平均)。"""
        index = {"x": 0, "y": 1, "z": 2}[axis]
        other = tuple(i for i in range(3) if i != index)
        data = {"index": np.arange(1, self.grid[index] + 1)}
        for step, array in zip(self.steps, self.temperatures):
            data[f"step_{step}"] = array.mean(axis=other)
        return pd.DataFrame(data)


def read_ttm_electron_temperature(
    path: Path | str = "ttm_electron_temperature.out",
) -> TTMSnapshots:
    """``ttm_electron_temperature.out``: 電子温度グリッドのスナップショット列。"""
    resolved = _resolve(path, "ttm_electron_temperature.out")
    header: dict[str, str] = {}
    grid = (1, 1, 1)
    steps: list[int] = []
    arrays: list[np.ndarray] = []
    current: np.ndarray | None = None

    for line in resolved.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            tokens = stripped.lstrip("#").split()
            if not tokens:
                continue
            if tokens[0] == "nx" and len(tokens) >= 6:
                grid = (int(tokens[1]), int(tokens[3]), int(tokens[5]))
                header["grid"] = f"{grid[0]}x{grid[1]}x{grid[2]}"
            elif tokens[0] == "step" and len(tokens) >= 2:
                current = np.zeros(grid, dtype=float)
                steps.append(int(tokens[1]))
                arrays.append(current)
            elif len(tokens) >= 2:
                header[tokens[0]] = " ".join(tokens[1:])
            continue
        if current is None:
            continue
        ix, iy, iz, value = stripped.split()[:4]
        current[int(ix) - 1, int(iy) - 1, int(iz) - 1] = float(value)
    if not steps:
        raise ValueError(f"{resolved.name} にスナップショットがありません。")
    return TTMSnapshots(grid=grid, steps=steps, temperatures=arrays, header=header)


#: :func:`read_shock_profiles` が読むファイルと単位
SHOCK_FILES = {
    "temperature": ("temperature_hist.txt", "K"),
    "pxx": ("pxx_hist.txt", "GPa"),
    "pyy": ("pyy_hist.txt", "GPa"),
    "pzz": ("pzz_hist.txt", "GPa"),
    "density": ("density_hist.txt", "g/cm^3"),
    "vp": ("vp_hist.txt", "km/s"),
}


def read_shock_profiles(
    directory: Path | str = ".", *, bin_size: float | None = None
) -> dict[str, pd.DataFrame]:
    """``dump_shock_nemd`` の x 方向空間分布ファイルをまとめて読む。

    各 DataFrame は「行 = 出力時刻、列 = x 方向のビン」。
    ``bin_size`` を渡すと列名がビン中心の座標 [Å] になる。
    """
    directory = Path(directory).expanduser()
    result: dict[str, pd.DataFrame] = {}
    for key, (filename, unit) in SHOCK_FILES.items():
        path = directory / filename
        if not path.is_file():
            continue
        matrix = np.loadtxt(path, ndmin=2)
        if bin_size:
            columns = [(i + 0.5) * bin_size for i in range(matrix.shape[1])]
        else:
            columns = list(range(matrix.shape[1]))
        frame = pd.DataFrame(matrix, columns=columns)
        frame.attrs["unit"] = unit
        result[key] = frame
    if not result:
        raise FileNotFoundError(f"{directory} に *_hist.txt がありません。")
    return result


def read_spring(path: Path | str) -> pd.DataFrame:
    """``spring_gm<m>_g<g>_s<id>.out``: ばね力と弾性エネルギー。"""
    return read_table(
        path,
        ["step", "mode", "Fx", "Fy", "Fz", "F_total", "energy"],
    )


# ------------------------------------------------------------------ まとめ役
class OutputReader:
    """1 つの計算ディレクトリの出力をまとめて扱う。

    メソッド名はファイル名から ``.out`` を取ったものに対応する。
    存在しないファイルを読もうとすると :class:`FileNotFoundError` になる。
    """

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory).expanduser().resolve()
        if not self.directory.is_dir():
            raise NotADirectoryError(f"ディレクトリがありません: {self.directory}")

    def __repr__(self) -> str:
        return f"OutputReader({self.directory}, {len(self.available())} files)"

    def path(self, name: str) -> Path:
        return self.directory / name

    def has(self, name: str) -> bool:
        return (self.directory / name).is_file()

    def available(self) -> list[str]:
        """読み込み対象になりうるファイルの一覧。"""
        patterns = ("*.out", "*.csv", "*.yaml", "*.xyz", "*_hist.txt", "*.nc")
        names: set[str] = set()
        for pattern in patterns:
            names.update(p.name for p in self.directory.glob(pattern))
        return sorted(names)

    # --- 個別 ---
    def thermo(self) -> pd.DataFrame:
        return read_thermo(self.path("thermo.out"))

    def compute(self, *, n_groups: int, quantities: Sequence[str]) -> pd.DataFrame:
        return read_compute(self.path("compute.out"), n_groups=n_groups, quantities=quantities)

    def compute_chunk(self, *, n_dims: int = 1, quantities=("temperature",)) -> pd.DataFrame:
        return read_compute_chunk(
            self.path("compute_chunk.out"), n_dims=n_dims, quantities=quantities
        )

    def cohesive(self) -> pd.DataFrame:
        return read_cohesive(self.path("cohesive.out"))

    def elastic(self) -> np.ndarray:
        return read_elastic(self.path("elastic.out"))

    def omega2(self) -> pd.DataFrame:
        return read_omega2(self.path("omega2.out"))

    def dos(self, *, n_groups: int = 1) -> pd.DataFrame:
        return read_dos(self.path("dos.out"), n_groups=n_groups)

    def mvac(self) -> pd.DataFrame:
        return read_mvac(self.path("mvac.out"))

    def sdc(self, *, n_groups: int = 1) -> pd.DataFrame:
        return read_sdc(self.path("sdc.out"), n_groups=n_groups)

    def msd(self, *, n_groups: int | None = None) -> pd.DataFrame:
        return read_msd(self.path("msd.out"), n_groups=n_groups)

    def ic(self) -> pd.DataFrame:
        return read_ic(self.path("ic.out"))

    def rdf(self, *, pairs: Sequence[str] = ()) -> pd.DataFrame:
        return read_rdf(self.path("rdf.out"), pairs=pairs)

    def adf(self, *, labels: Sequence[str] = ()) -> pd.DataFrame:
        return read_adf(self.path("adf.out"), labels=labels)

    def angular_rdf(self) -> pd.DataFrame:
        return read_angular_rdf(self.path("angular_rdf.out"))

    def orientorder(self) -> pd.DataFrame:
        return read_orientorder(self.path("orientorder.out"))

    def hac(self) -> pd.DataFrame:
        return read_hac(self.path("hac.out"))

    def kappa(self) -> pd.DataFrame:
        return read_kappa(self.path("kappa.out"))

    def onsager(self) -> pd.DataFrame:
        return read_onsager(self.path("onsager.out"))

    def shc(self, *, Nc: int | None = None):
        return read_shc(self.path("shc.out"), Nc=Nc)

    def viscosity(self) -> pd.DataFrame:
        return read_viscosity(self.path("viscosity.out"))

    def heatmode(self) -> pd.DataFrame:
        return read_heatmode(self.path("heatmode.out"))

    def kappamode(self) -> pd.DataFrame:
        return read_kappamode(self.path("kappamode.out"))

    def lsqt(self, **kwargs):
        return read_lsqt(self.directory, **kwargs)

    def dipole(self) -> pd.DataFrame:
        return read_dipole(self.path("dipole.out"))

    def polarizability(self) -> pd.DataFrame:
        return read_polarizability(self.path("polarizability.out"))

    def dpdt(self) -> pd.DataFrame:
        return read_dpdt(self.path("dpdt.out"))

    def active(self) -> pd.DataFrame:
        return read_active(self.path("active.out"))

    def mcmd(self, *, species: Sequence[str] = ()) -> pd.DataFrame:
        return read_mcmd(self.path("mcmd.out"), species=species)

    def ti(self, name: str) -> pd.DataFrame:
        """``ti_spring`` などの名前を渡して csv を読む。"""
        return read_ti_csv(self.path(f"{name}.csv"))

    def ti_summary(self, name: str) -> dict[str, float]:
        return read_ti_yaml(self.path(f"{name}.yaml"))

    def ttm(self) -> TTMSnapshots:
        return read_ttm_electron_temperature(self.path("ttm_electron_temperature.out"))

    def shock_profiles(self, *, bin_size: float | None = None) -> dict[str, pd.DataFrame]:
        return read_shock_profiles(self.directory, bin_size=bin_size)

    def spring(self, filename: str) -> pd.DataFrame:
        return read_spring(self.path(filename))
