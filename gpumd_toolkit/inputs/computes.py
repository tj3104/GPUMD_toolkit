"""GPUMD の ``compute_*`` / ``compute`` キーワード。

各関数は ``run.in`` に書く 1 行を文字列で返す。
``MDStage(pre_commands=[...])`` や ``DumpSettings(extra=[...])`` に並べて使う。

分類
----
静的・構造          :func:`compute_cohesive` :func:`compute_elastic` :func:`compute_phonon`
熱輸送 (平衡)       :func:`compute_hac` :func:`compute_viscosity`
熱輸送 (非平衡)     :func:`compute_hnemd` :func:`compute_hnemdec` :func:`compute_shc`
熱輸送 (モード分解) :func:`compute_gkma` :func:`compute_hnema` :func:`compute_dos`
拡散・液体          :func:`compute_msd` :func:`compute_sdc` :func:`compute_ic`
構造相関            :func:`compute_rdf` :func:`compute_adf` :func:`compute_angular_rdf`
                    :func:`compute_orientorder`
電子・電気          :func:`compute_lsqt` :func:`compute_dpdt`
空間平均            :func:`compute` :func:`compute_chunk`
"""

from __future__ import annotations

from typing import Mapping, Sequence

__all__ = [
    "COMPUTE_QUANTITIES",
    "CHUNK_QUANTITIES",
    "compute",
    "compute_chunk",
    "compute_adf",
    "compute_angular_rdf",
    "compute_cohesive",
    "compute_dos",
    "compute_dpdt",
    "compute_elastic",
    "compute_gkma",
    "compute_hac",
    "compute_hnema",
    "compute_hnemd",
    "compute_hnemdec",
    "compute_ic",
    "compute_lsqt",
    "compute_msd",
    "compute_orientorder",
    "compute_phonon",
    "compute_rdf",
    "compute_sdc",
    "compute_shc",
    "compute_viscosity",
]

#: :func:`compute` で指定できる量
COMPUTE_QUANTITIES = ("temperature", "potential", "force", "virial", "jp", "jk", "momentum")

#: :func:`compute_chunk` で指定できる量
CHUNK_QUANTITIES = (
    "temperature", "density/number", "density/mass",
    "vx", "vy", "vz", "fx", "fy", "fz",
)

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _positive(value, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} は正の整数です (与えられたのは {value})。")
    return value


def _axis_index(direction: str | int, name: str = "direction") -> int:
    if isinstance(direction, int):
        if direction not in (0, 1, 2):
            raise ValueError(f"{name} は 0 (x) / 1 (y) / 2 (z) です。")
        return direction
    key = str(direction).strip().lower()
    if key not in _AXIS_INDEX:
        raise ValueError(f"{name} は 'x' / 'y' / 'z' です。")
    return _AXIS_INDEX[key]


def _group_args(group: Sequence[int] | None) -> str:
    if group is None:
        return ""
    pair = tuple(int(v) for v in group)
    if len(pair) != 2:
        raise ValueError("group は (grouping_method, group_id) の 2 要素で指定します。")
    if pair[0] < 0:
        raise ValueError("grouping_method は 0 以上です。")
    return f"group {pair[0]} {pair[1]}"


def _force_vector(Fe: float | Sequence[float], direction: str | None) -> tuple[float, float, float]:
    """駆動力を (Fe_x, Fe_y, Fe_z) に展開する。"""
    if direction is not None:
        vector = [0.0, 0.0, 0.0]
        vector[_axis_index(direction)] = float(Fe)  # type: ignore[arg-type]
        return tuple(vector)  # type: ignore[return-value]
    if isinstance(Fe, (int, float)):
        raise ValueError("Fe をスカラーで与えるときは direction ('x'/'y'/'z') も指定してください。")
    seq = tuple(float(v) for v in Fe)
    if len(seq) != 3:
        raise ValueError("Fe は 3 成分 (Fe_x, Fe_y, Fe_z) で指定します。")
    return seq


# --------------------------------------------------------------- 空間・時間平均
def compute(
    grouping_method: int,
    sample_interval: int,
    output_interval: int,
    quantities: Sequence[str],
) -> str:
    """グループごとの空間・時間平均量を ``compute.out`` に出力する。

    NEMD の温度プロファイルや熱流の取得に使う。
    ``quantities`` は :data:`COMPUTE_QUANTITIES` から選ぶ。
    出力の並びは指定順ではなく GPUMD 側で固定されている点に注意。
    """
    items = tuple(quantities)
    if not items:
        raise ValueError("quantities を 1 つ以上指定してください。")
    bad = [q for q in items if q not in COMPUTE_QUANTITIES]
    if bad:
        raise ValueError(f"compute で未対応の量です: {bad} (使用可能: {COMPUTE_QUANTITIES})")
    if len(set(items)) != len(items):
        raise ValueError("quantities に重複があります。")
    if int(grouping_method) < 0:
        raise ValueError("grouping_method は 0 以上です。")
    return (
        f"compute {int(grouping_method)} {_positive(sample_interval, 'sample_interval')} "
        f"{_positive(output_interval, 'output_interval')} {' '.join(items)}"
    )


def compute_chunk(
    sample_interval: int,
    output_interval: int,
    bins: Sequence[tuple[str, float]],
    quantities: Sequence[str],
) -> str:
    """実時間の座標で動的に空間ビン分割した平均量を ``compute_chunk.out`` に出力する。

    :func:`compute` と違い、原子が拡散してビンをまたぐ系 (液体・多孔体・アモルファス)
    でも正しい空間プロファイルが取れる。

    Parameters
    ----------
    bins
        ``[('z', 1.0)]`` のように ``(軸, ビン幅 [Å])`` を 1-3 個。
    quantities
        :data:`CHUNK_QUANTITIES` から選ぶ。
    """
    bin_list = [(str(axis).lower(), float(delta)) for axis, delta in bins]
    if not 1 <= len(bin_list) <= 3:
        raise ValueError("bins は 1-3 個 (bin/1d, bin/2d, bin/3d) です。")
    axes = [axis for axis, _ in bin_list]
    if len(set(axes)) != len(axes):
        raise ValueError("bins の軸が重複しています。")
    for axis, delta in bin_list:
        _axis_index(axis, "bins の軸")
        if delta <= 0:
            raise ValueError("ビン幅 [Å] は正の値です。")
    items = tuple(quantities)
    bad = [q for q in items if q not in CHUNK_QUANTITIES]
    if bad:
        raise ValueError(f"compute_chunk で未対応の量です: {bad} (使用可能: {CHUNK_QUANTITIES})")
    if not items:
        raise ValueError("quantities を 1 つ以上指定してください。")
    style = f"bin/{len(bin_list)}d"
    spec = " ".join(f"{axis} lower {delta:g}" for axis, delta in bin_list)
    return (
        f"compute_chunk {_positive(sample_interval, 'sample_interval')} "
        f"{_positive(output_interval, 'output_interval')} {style} {spec} {' '.join(items)}"
    )


# ------------------------------------------------------------------ 静的・構造
def compute_cohesive(e1: float, e2: float, direction: str | int = "xyz") -> str:
    """凝集エネルギー曲線を ``cohesive.out`` に出力する (体積スキャン)。

    Parameters
    ----------
    e1, e2
        セルのスケーリング係数の下限・上限。``(e2 - e1) * 1000 + 1`` 点を計算する。
    direction
        スケーリング方向。``'x'``, ``'y'``, ``'z'``, ``'xy'``, ``'yz'``, ``'zx'``,
        ``'xyz'`` (等方) または 0-6 の整数。
    """
    directions = ("x", "y", "z", "xy", "yz", "zx", "xyz")
    if isinstance(direction, int):
        if not 0 <= direction <= 6:
            raise ValueError("direction は 0-6 です。")
        index = direction
    else:
        key = str(direction).strip().lower()
        if key not in directions:
            raise ValueError(f"direction は {directions} から選びます。")
        index = directions.index(key)
    if not 0 < e1 < e2:
        raise ValueError("0 < e1 < e2 である必要があります。")
    return f"compute_cohesive {e1:g} {e2:g} {index}"


def compute_elastic(strain: float = 0.01) -> str:
    """弾性定数行列 C_ij [GPa] を ``elastic.out`` に出力する。"""
    if not 0 < strain < 0.5:
        raise ValueError("strain は 0 より大きく 0.5 未満の値にしてください (目安 0.01)。")
    return f"compute_elastic {strain:g}"


def compute_phonon(displacement: float = 0.01) -> str:
    """有限変位法でフォノン分散を計算し ``D.out`` / ``omega2.out`` に出力する。

    ``kpoints.in`` を作業ディレクトリに置き、``run.in`` の先頭で
    :func:`~gpumd_toolkit.inputs.modifiers.replicate` を呼んでおく必要がある。
    """
    if displacement <= 0:
        raise ValueError("displacement [Å] は正の値です (目安 0.01)。")
    return f"compute_phonon {displacement:g}"


# ------------------------------------------------------------------ 熱輸送
def compute_hac(
    sampling_interval: int, correlation_steps: int, output_interval: int = 1
) -> str:
    """Green-Kubo 法: 熱流自己相関と熱伝導率を ``hac.out`` に出力する。"""
    return (
        f"compute_hac {_positive(sampling_interval, 'sampling_interval')} "
        f"{_positive(correlation_steps, 'correlation_steps')} "
        f"{_positive(output_interval, 'output_interval')}"
    )


def compute_viscosity(sampling_interval: int, correlation_steps: int) -> str:
    """Green-Kubo 法: 応力自己相関と粘性率を ``viscosity.out`` に出力する。"""
    return (
        f"compute_viscosity {_positive(sampling_interval, 'sampling_interval')} "
        f"{_positive(correlation_steps, 'correlation_steps')}"
    )


def compute_hnemd(
    output_interval: int,
    Fe: float | Sequence[float],
    direction: str | None = None,
) -> str:
    """HNEMD 法: 駆動力 Fe [1/Å] を加えて熱伝導率を ``kappa.out`` に出力する。

    駆動力は小さく (目安 1e-5 /Å 以下) し、必ず熱浴 (``nvt_nhc`` 推奨) を併用する。
    """
    fx, fy, fz = _force_vector(Fe, direction)
    if fx == fy == fz == 0.0:
        raise ValueError("Fe がゼロベクトルです。")
    return (
        f"compute_hnemd {_positive(output_interval, 'output_interval')} "
        f"{fx:g} {fy:g} {fz:g}"
    )


def compute_hnemdec(
    driving_force: int,
    output_interval: int,
    Fe: float | Sequence[float],
    direction: str | None = None,
) -> str:
    """HNEMDEC 法: 多成分系の Onsager 係数を ``onsager.out`` に出力する。

    Parameters
    ----------
    driving_force
        ``0`` なら熱流を散逸流にする (Fe の単位 1/Å)。
        正の整数 i なら i 番目の元素の運動量流を散逸流にする (Fe の単位 eV/Å)。
    """
    if int(driving_force) < 0:
        raise ValueError("driving_force は 0 以上の整数です。")
    fx, fy, fz = _force_vector(Fe, direction)
    return (
        f"compute_hnemdec {int(driving_force)} "
        f"{_positive(output_interval, 'output_interval')} {fx:g} {fy:g} {fz:g}"
    )


def compute_shc(
    sample_interval: int,
    Nc: int,
    direction: str | int,
    num_omega: int,
    max_omega: float,
    *,
    group: Sequence[int] | None = None,
) -> str:
    """スペクトル熱流 K(t), J_q(omega) を ``shc.out`` に出力する。"""
    sample_interval = _positive(sample_interval, "sample_interval")
    if sample_interval > 10:
        raise ValueError("compute_shc の sample_interval は 1-10 です。")
    Nc = _positive(Nc, "Nc")
    if not 100 <= Nc <= 1000:
        raise ValueError("compute_shc の Nc は 100-1000 です。")
    if max_omega <= 0:
        raise ValueError("max_omega [THz] は正の値です。")
    parts = [
        f"compute_shc {sample_interval} {Nc} {_axis_index(direction)} "
        f"{_positive(num_omega, 'num_omega')} {max_omega:g}"
    ]
    group_args = _group_args(group)
    if group_args:
        parts.append(group_args)
    return " ".join(parts)


def compute_gkma(
    sample_interval: int,
    first_mode: int,
    last_mode: int,
    bin_option: str = "f_bin_size",
    size: float = 1.0,
) -> str:
    """GKMA 法: モード分解した熱流を ``heatmode.out`` に出力する。

    ``eigenvector.in`` が必要。``bin_option`` は ``'bin_size'`` (モード数で分割) か
    ``'f_bin_size'`` (振動数 [THz] で分割)。
    """
    if bin_option not in ("bin_size", "f_bin_size"):
        raise ValueError("bin_option は 'bin_size' か 'f_bin_size' です。")
    if first_mode < 1 or last_mode < first_mode:
        raise ValueError("1 <= first_mode <= last_mode である必要があります。")
    value = f"{int(size)}" if bin_option == "bin_size" else f"{float(size):g}"
    return (
        f"compute_gkma {_positive(sample_interval, 'sample_interval')} "
        f"{int(first_mode)} {int(last_mode)} {bin_option} {value}"
    )


def compute_hnema(
    sample_interval: int,
    output_interval: int,
    Fe: float | Sequence[float],
    first_mode: int,
    last_mode: int,
    bin_option: str = "f_bin_size",
    size: float = 1.0,
    *,
    direction: str | None = None,
) -> str:
    """HNEMA 法: モード分解した熱伝導率を ``kappamode.out`` に出力する。"""
    if bin_option not in ("bin_size", "f_bin_size"):
        raise ValueError("bin_option は 'bin_size' か 'f_bin_size' です。")
    output_interval = _positive(output_interval, "output_interval")
    sample_interval = _positive(sample_interval, "sample_interval")
    if output_interval % sample_interval != 0:
        raise ValueError("sample_interval は output_interval の約数である必要があります。")
    if first_mode < 1 or last_mode < first_mode:
        raise ValueError("1 <= first_mode <= last_mode である必要があります。")
    fx, fy, fz = _force_vector(Fe, direction)
    value = f"{int(size)}" if bin_option == "bin_size" else f"{float(size):g}"
    return (
        f"compute_hnema {sample_interval} {output_interval} {fx:g} {fy:g} {fz:g} "
        f"{int(first_mode)} {int(last_mode)} {bin_option} {value}"
    )


def compute_dos(
    sample_interval: int,
    Nc: int,
    omega_max: float,
    *,
    group: Sequence[int] | None = None,
    num_dos_points: int | None = None,
) -> str:
    """速度自己相関からフォノン状態密度を ``dos.out`` / ``mvac.out`` に出力する。

    同じ run で :func:`compute_sdc` と併用することはできない。
    """
    if omega_max <= 0:
        raise ValueError("omega_max [THz] は正の値です (角振動数 2*pi*nu_max)。")
    parts = [
        f"compute_dos {_positive(sample_interval, 'sample_interval')} "
        f"{_positive(Nc, 'Nc')} {omega_max:g}"
    ]
    group_args = _group_args(group)
    if group_args:
        parts.append(group_args)
    if num_dos_points is not None:
        parts.append(f"num_dos_points {_positive(num_dos_points, 'num_dos_points')}")
    return " ".join(parts)


# ------------------------------------------------------------- 拡散・イオン伝導
def compute_msd(
    sample_interval: int,
    Nc: int,
    *,
    group: Sequence[int] | None = None,
    all_groups: int | None = None,
    save_every: int | None = None,
) -> str:
    """平均二乗変位と自己拡散係数を ``msd.out`` に出力する。

    ``all_groups`` に grouping method を渡すと、その方法の全グループについて
    MSD/SDC を計算する (分子ごとの MSD など)。``group`` とは併用できない。
    """
    if group is not None and all_groups is not None:
        raise ValueError("group と all_groups は同時に指定できません。")
    parts = [f"compute_msd {_positive(sample_interval, 'sample_interval')} {_positive(Nc, 'Nc')}"]
    group_args = _group_args(group)
    if group_args:
        parts.append(group_args)
    if all_groups is not None:
        if int(all_groups) < 0:
            raise ValueError("all_groups (grouping method) は 0 以上です。")
        parts.append(f"all_groups {int(all_groups)}")
    if save_every is not None:
        parts.append(f"save_every {_positive(save_every, 'save_every')}")
    return " ".join(parts)


def compute_sdc(
    sample_interval: int, Nc: int, *, group: Sequence[int] | None = None
) -> str:
    """速度自己相関から自己拡散係数を ``sdc.out`` に出力する。

    同じ run で :func:`compute_dos` と併用することはできない。
    """
    parts = [f"compute_sdc {_positive(sample_interval, 'sample_interval')} {_positive(Nc, 'Nc')}"]
    group_args = _group_args(group)
    if group_args:
        parts.append(group_args)
    return " ".join(parts)


def compute_ic(sample_interval: int, Nc: int, type_index: int, charge: float) -> str:
    """指定した元素のイオン伝導度を ``ic.out`` [mS/cm] に出力する。

    Parameters
    ----------
    type_index
        ポテンシャルファイルにおける元素の型番号 (0 始まり)。
    charge
        そのイオンの電荷 (電気素量単位)。
    """
    if int(type_index) < 0:
        raise ValueError("type_index は 0 以上です。")
    return (
        f"compute_ic {_positive(sample_interval, 'sample_interval')} "
        f"{_positive(Nc, 'Nc')} {int(type_index)} {charge:g}"
    )


# ------------------------------------------------------------------ 構造相関
def compute_rdf(cutoff: float, num_bins: int, interval: int) -> str:
    """動径分布関数を ``rdf.out`` に出力する (全ペアの partial RDF も自動で出る)。"""
    if cutoff <= 0:
        raise ValueError("cutoff [Å] は正の値です。")
    return (
        f"compute_rdf {cutoff:g} {_positive(num_bins, 'num_bins')} "
        f"{_positive(interval, 'interval')}"
    )


def compute_adf(
    interval: int,
    num_bins: int,
    rc_min: float | None = None,
    rc_max: float | None = None,
    *,
    triples: Sequence[tuple[int, int, int, float, float, float, float]] = (),
) -> str:
    """角度分布関数を ``adf.out`` に出力する。

    Parameters
    ----------
    rc_min, rc_max
        全体 ADF の距離範囲 [Å]。``triples`` を使う場合は指定しない。
    triples
        局所 ADF。``(itype, jtype, ktype, rc_min_j, rc_max_j, rc_min_k, rc_max_k)``
        のタプルを並べる。
    """
    head = f"compute_adf {_positive(interval, 'interval')} {_positive(num_bins, 'num_bins')}"
    if triples:
        if rc_min is not None or rc_max is not None:
            raise ValueError("triples を使う場合は rc_min / rc_max を指定しません。")
        parts = [head]
        for item in triples:
            if len(item) != 7:
                raise ValueError(
                    "triples の各要素は (itype, jtype, ktype, rc_min_j, rc_max_j,"
                    " rc_min_k, rc_max_k) の 7 要素です。"
                )
            i, j, k, rjmin, rjmax, rkmin, rkmax = item
            if rjmin >= rjmax or rkmin >= rkmax:
                raise ValueError("rc_min < rc_max である必要があります。")
            parts.append(
                f"{int(i)} {int(j)} {int(k)} {rjmin:g} {rjmax:g} {rkmin:g} {rkmax:g}"
            )
        return " ".join(parts)
    if rc_min is None or rc_max is None:
        raise ValueError("全体 ADF には rc_min と rc_max が必要です。")
    if rc_min >= rc_max:
        raise ValueError("rc_min < rc_max である必要があります。")
    return f"{head} {rc_min:g} {rc_max:g}"


def compute_angular_rdf(
    cutoff: float,
    r_num_bins: int,
    angular_num_bins: int,
    interval: int,
    *,
    pairs: Sequence[tuple[int, int]] = (),
) -> str:
    """角度依存 RDF g(r, theta) を ``angular_rdf.out`` に出力する。"""
    if cutoff <= 0:
        raise ValueError("cutoff [Å] は正の値です。")
    if len(pairs) > 6:
        raise ValueError("partial ARDF は最大 6 ペアまでです。")
    parts = [
        f"compute_angular_rdf {cutoff:g} {_positive(r_num_bins, 'r_num_bins')} "
        f"{_positive(angular_num_bins, 'angular_num_bins')} {_positive(interval, 'interval')}"
    ]
    for i, j in pairs:
        parts.append(f"atom {int(i)} {int(j)}")
    return " ".join(parts)


def compute_orientorder(
    interval: int,
    degrees: Sequence[int],
    *,
    mode: str = "cutoff",
    parameter: float = 4.0,
    average: bool = False,
    wl: bool = False,
    wl_hat: bool = False,
) -> str:
    """Steinhardt 配向秩序変数 q_l / w_l を ``orientorder.out`` に出力する。

    結晶核生成・結晶/非晶の判別に使う。

    Parameters
    ----------
    mode
        ``'cutoff'`` (近傍をカットオフ距離 [Å] で選ぶ) か
        ``'nnn'`` (近傍数で選ぶ)。
    parameter
        ``mode`` に対応する値 (カットオフ距離または近傍数)。
    degrees
        計算する球面調和関数の次数 l のリスト (例 ``[4, 6]``)。
    """
    if mode not in ("cutoff", "nnn"):
        raise ValueError("mode は 'cutoff' か 'nnn' です。")
    ls = [int(d) for d in degrees]
    if not ls:
        raise ValueError("degrees を 1 つ以上指定してください。")
    if any(d < 0 for d in ls):
        raise ValueError("degrees は 0 以上の整数です。")
    value = f"{int(parameter)}" if mode == "nnn" else f"{float(parameter):g}"
    return (
        f"compute_orientorder {_positive(interval, 'interval')} {mode} {value} "
        f"{len(ls)} {' '.join(str(d) for d in ls)} "
        f"{int(bool(average))} {int(bool(wl))} {int(bool(wl_hat))}"
    )


# ------------------------------------------------------------------ 電子・電気
def compute_lsqt(
    direction: str,
    num_moments: int,
    num_energies: int,
    E_1: float,
    E_2: float,
    E_max: float,
) -> str:
    """線形スケーリング量子輸送 (LSQT) で電子輸送を計算する。

    ``lsqt_dos.out`` / ``lsqt_velocity.out`` / ``lsqt_sigma.out`` を出力する。
    現状は炭素系の強束縛模型のみ (ハードコード)。
    """
    if direction not in ("x", "y", "z"):
        raise ValueError("transport_direction は 'x' / 'y' / 'z' です。")
    if E_1 >= E_2:
        raise ValueError("E_1 < E_2 である必要があります。")
    if E_max <= max(abs(E_1), abs(E_2)):
        raise ValueError("E_max は |E_1|, |E_2| より大きく取ってください。")
    return (
        f"compute_lsqt {direction} {_positive(num_moments, 'num_moments')} "
        f"{_positive(num_energies, 'num_energies')} {E_1:g} {E_2:g} {E_max:g}"
    )


def compute_dpdt(sampling_interval: int) -> str:
    """Born 有効電荷と速度から分極の時間微分を ``dpdt.out`` に出力する。

    BEC を学習した qNEP モデルでのみ意味がある (赤外スペクトル)。
    """
    return f"compute_dpdt {_positive(sampling_interval, 'sampling_interval')}"
