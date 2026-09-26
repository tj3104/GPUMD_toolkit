"""セットアップ・外場・特殊操作のキーワード。

``ensemble`` / ``compute_*`` / ``dump_*`` 以外の ``run.in`` キーワードを
すべてここで組み立てる。各関数は 1 行の文字列を返す。

分類
----
系の準備        :func:`replicate` :func:`velocity` :func:`correct_velocity`
                :func:`potential` :func:`dftd3` :func:`kspace`
セル操作        :func:`change_box` :func:`deform`
原子の拘束・駆動 :func:`fix` :func:`move` :func:`add_force` :func:`add_efield`
                :func:`add_spring` :func:`deposit` :func:`electron_stop`
組成自由度      :func:`mc_canonical` :func:`mc_sgc` :func:`mc_vcsgc`
不確かさ        :func:`compute_extrapolation`
外部プラグイン  :func:`plumed`
最適化          :func:`minimize`
"""

from __future__ import annotations

from typing import Mapping, Sequence

__all__ = [
    "replicate",
    "velocity",
    "correct_velocity",
    "potential",
    "dftd3",
    "kspace",
    "change_box",
    "deform",
    "fix",
    "move",
    "add_force",
    "add_efield",
    "add_spring",
    "deposit",
    "electron_stop",
    "plumed",
    "compute_extrapolation",
    "mc_canonical",
    "mc_sgc",
    "mc_vcsgc",
    "minimize",
    "time_step",
    "DEFORM_COMPONENTS",
    "MINIMIZE_METHODS",
]

#: :func:`deform` の一般形で指定できるセル成分
DEFORM_COMPONENTS = ("xx", "yy", "zz", "xy", "xz", "yz")

#: :func:`minimize` の手法
MINIMIZE_METHODS = ("fire", "sd")

_DIRECTION_INDEX = {"x": 0, "y": 1, "z": 2}


def _positive(value, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} は正の整数です (与えられたのは {value})。")
    return value


def _nonneg(value, name: str) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} は 0 以上の整数です (与えられたのは {value})。")
    return value


def _vector3(value: Sequence[float], name: str) -> tuple[float, float, float]:
    seq = tuple(float(v) for v in value)
    if len(seq) != 3:
        raise ValueError(f"{name} は 3 成分で指定します (与えられたのは {len(seq)} 個)。")
    return seq  # type: ignore[return-value]


# ------------------------------------------------------------------ 系の準備
def replicate(na: int, nb: int, nc: int) -> str:
    """セルを a, b, c 方向に複製する。``potential`` より **前** に置くこと。"""
    counts = [_positive(n, "replicate の複製数") for n in (na, nb, nc)]
    return f"replicate {counts[0]} {counts[1]} {counts[2]}"


def velocity(initial_temperature: float, seed: int | None = None) -> str:
    """初期速度を温度 [K] から与える。``seed`` を指定すると再現性が確保される。"""
    if initial_temperature < 0:
        raise ValueError("initial_temperature [K] は 0 以上です。")
    tail = f" seed {int(seed)}" if seed is not None else ""
    return f"velocity {initial_temperature:g}{tail}"


def correct_velocity(interval: int, grouping_method: int | None = None) -> str:
    """一定間隔で全体 (またはグループごと) の並進・回転運動量をゼロに戻す。"""
    interval = _positive(interval, "interval")
    if interval < 10:
        raise ValueError("correct_velocity の interval は 10 以上です (10-100 が目安)。")
    if grouping_method is None:
        return f"correct_velocity {interval}"
    return f"correct_velocity {interval} {_nonneg(grouping_method, 'grouping_method')}"


def potential(filename: str | Sequence[str]) -> str:
    """ポテンシャルファイルを指定する。複数並べると committee / observer になる。"""
    if isinstance(filename, (str, bytes)):
        return f"potential {filename}"
    return f"potential {' '.join(str(item) for item in filename)}"


def dftd3(functional: str, potential_cutoff: float, coordination_cutoff: float) -> str:
    """NEP に DFT-D3 (BJ damping) 分散力補正を足す。

    学習データに分散補正が入っている NEP に重ねると二重計上になるので注意。
    """
    if potential_cutoff <= 0 or coordination_cutoff <= 0:
        raise ValueError("カットオフ [Å] は正の値です。")
    return f"dftd3 {functional} {potential_cutoff:g} {coordination_cutoff:g}"


def kspace(method: str = "pppm") -> str:
    """静電相互作用の逆空間項の計算法を選ぶ (``ewald`` / ``pppm``)。"""
    if method not in ("ewald", "pppm"):
        raise ValueError("kspace の method は 'ewald' か 'pppm' です。")
    return f"kspace {method}"


def time_step(dt_fs: float, max_distance: float | None = None) -> str:
    """時間刻み [fs] を設定する (propagating: 次の run にも引き継がれる)。

    ``max_distance`` [Å] を与えると、1 ステップで原子が動ける距離を制限する
    (高エネルギー衝突・照射損傷で暴走を防ぐ)。
    """
    if dt_fs < 0:
        raise ValueError("time_step [fs] は 0 以上です。")
    if max_distance is None:
        return f"time_step {dt_fs:g}"
    if max_distance <= 0:
        raise ValueError("max_distance [Å] は正の値です。")
    return f"time_step {dt_fs:g} {max_distance:g}"


# ------------------------------------------------------------------ セル操作
def change_box(delta: float | Sequence[float]) -> str:
    """セルを一度だけ変形する (原子座標もアフィン変換される)。

    引数は 1 個 (等方、Å)、3 個 (dxx, dyy, dzz、Å)、
    6 個 (dxx, dyy, dzz [Å], eps_yz, eps_xz, eps_xy [無次元ひずみ]) のいずれか。
    6 個の場合、セルは三斜晶である必要がある。
    """
    if isinstance(delta, (int, float)):
        return f"change_box {float(delta):g}"
    seq = tuple(float(v) for v in delta)
    if len(seq) not in (1, 3, 6):
        raise ValueError("change_box の引数は 1 / 3 / 6 個です。")
    return "change_box " + " ".join(f"{v:g}" for v in seq)


def deform(
    rates: Mapping[str, float] | float,
    directions: Sequence[int] | None = None,
) -> str:
    """run の間セルを連続変形する (引張・圧縮・せん断)。

    Parameters
    ----------
    rates
        一般形なら ``{'xx': 1e-5, 'xy': 2e-5}`` のようにセル成分ごとの
        変化率 [Å/step]。旧来形 (4 引数) なら変化率のスカラー。
    directions
        旧来形で使う ``(deform_x, deform_y, deform_z)`` の 0/1 フラグ。
        ``rates`` がスカラーのときのみ指定する。

    Notes
    -----
    * NVE と標準 NVT では自由に使える。
    * NPT-Berendsen / NPT-SCR では等方圧力制御と併用できない。
    * MTTK 系のアンサンブルでは使えない。
    * 変形する方向は周期境界でなければならない。
    """
    if isinstance(rates, (int, float)):
        if directions is None:
            raise ValueError(
                "旧来形では directions=(deform_x, deform_y, deform_z) が必要です。"
                " 一般形を使うなら rates={'xx': 1e-5} のように渡してください。"
            )
        flags = tuple(int(bool(d)) for d in directions)
        if len(flags) != 3:
            raise ValueError("directions は 3 要素です。")
        if not any(flags):
            raise ValueError("directions が全て 0 です。")
        return f"deform {float(rates):g} {flags[0]} {flags[1]} {flags[2]}"
    if directions is not None:
        raise ValueError("一般形 (rates が辞書) では directions を指定しません。")
    items = list(rates.items())
    if not items:
        raise ValueError("deform する成分が指定されていません。")
    parts = ["deform"]
    seen: set[str] = set()
    for component, rate in items:
        key = str(component).lower()
        if key not in DEFORM_COMPONENTS:
            raise ValueError(
                f"deform の成分は {DEFORM_COMPONENTS} から選びます (与えられたのは '{component}')。"
            )
        if key in seen:
            raise ValueError(f"成分 '{key}' が重複しています。")
        seen.add(key)
        parts.append(f"{key} {float(rate):g}")
    return " ".join(parts)


# ------------------------------------------------------------ 原子の拘束・駆動
def fix(group_label: int, grouping_method: int | None = None) -> str:
    """グループの原子を固定する (速度・力を常にゼロにする)。"""
    if grouping_method is None:
        return f"fix {_nonneg(group_label, 'group_label')}"
    return (
        f"fix {_nonneg(grouping_method, 'grouping_method')} "
        f"{_nonneg(group_label, 'group_label')}"
    )


def move(
    group_label: int,
    velocity_vector: Sequence[float],
    grouping_method: int | None = None,
) -> str:
    """グループを一定速度 [Å/fs] で動かす (引張・せん断の境界駆動)。

    ``nvt_ber`` / ``nvt_nhc`` / ``nvt_bdp`` / ``heat_lan`` とのみ併用できる。
    :func:`fix` と併用する場合は同じ grouping method を使うこと。
    """
    vx, vy, vz = _vector3(velocity_vector, "velocity_vector")
    head = f"move {_nonneg(group_label, 'group_label')}"
    if grouping_method is not None:
        head = (
            f"move {_nonneg(grouping_method, 'grouping_method')} "
            f"{_nonneg(group_label, 'group_label')}"
        )
    return f"{head} {vx:g} {vy:g} {vz:g}"


def add_force(
    grouping_method: int,
    group_id: int,
    force: Sequence[float] | str,
) -> str:
    """グループの各原子に一定の力 [eV/Å] を加える。

    ``force`` にファイル名を渡すと、その中の力ベクトル列が周期的に適用される。
    """
    head = (
        f"add_force {_nonneg(grouping_method, 'grouping_method')} "
        f"{_nonneg(group_id, 'group_id')}"
    )
    if isinstance(force, str):
        return f"{head} {force}"
    fx, fy, fz = _vector3(force, "force")
    return f"{head} {fx:g} {fy:g} {fz:g}"


def add_efield(
    grouping_method: int,
    group_id: int,
    field: Sequence[float] | str,
    mode: str | None = None,
) -> str:
    """グループに電場 [V/Å] をかける。

    qNEP なら Born 有効電荷 (BEC) との内積、それ以外なら ``model.xyz`` の
    ``charge:R:1`` との積が力になる。``mode`` で ``'charge'`` / ``'bec'`` を明示できる。
    """
    if mode is not None and mode not in ("charge", "bec"):
        raise ValueError("add_efield の mode は 'charge' か 'bec' です。")
    head = (
        f"add_efield {_nonneg(grouping_method, 'grouping_method')} "
        f"{_nonneg(group_id, 'group_id')}"
    )
    if isinstance(field, str):
        body = field
    else:
        ex, ey, ez = _vector3(field, "field")
        body = f"{ex:g} {ey:g} {ez:g}"
    tail = f" {mode}" if mode else ""
    return f"{head} {body}{tail}"


def add_spring(
    mode: str,
    grouping_method: int,
    group_id: int,
    *,
    group_id_2: int | None = None,
    ghost_velocity: Sequence[float] = (0.0, 0.0, 0.0),
    spring_type: str = "couple",
    k: float | Sequence[float] = 10.0,
    R0: float = 0.0,
    offset: Sequence[float] = (0.0, 0.0, 0.0),
    continue_id: int | None = None,
) -> str:
    """ばねで拘束・駆動する (steered MD、自由エネルギー計算の引き込みなど)。

    Parameters
    ----------
    mode
        ``'ghost_com'`` (ゴースト原子 - グループ重心)、
        ``'ghost_atom'`` (ゴースト原子 - 各原子)、
        ``'com_com'`` (2 グループの重心どうし)。
    spring_type
        ``'couple'`` (距離に対する 1 本のばね、``k`` と ``R0`` を使う) か
        ``'decouple'`` (x/y/z 独立の 3 本、``k`` に 3 成分を渡す)。
    ghost_velocity
        ゴースト原子の移動速度 [Å/step] (``ghost_*`` モードのみ)。
    offset
        ゴースト原子の初期オフセット [Å] (``ghost_*`` モードのみ)。
    continue_id
        前の run のばね状態を引き継ぐ場合の spring_id。
    """
    if mode not in ("ghost_com", "ghost_atom", "com_com"):
        raise ValueError("mode は 'ghost_com' / 'ghost_atom' / 'com_com' です。")
    if spring_type not in ("couple", "decouple"):
        raise ValueError("spring_type は 'couple' か 'decouple' です。")

    if spring_type == "couple":
        if not isinstance(k, (int, float)) or float(k) <= 0:
            raise ValueError("couple の k [eV/Å^2] は正のスカラーです。")
        if R0 < 0:
            raise ValueError("R0 [Å] は 0 以上です。")
        spring_args = f"couple {float(k):g} {R0:g}"
    else:
        ks = (float(k), float(k), float(k)) if isinstance(k, (int, float)) else _vector3(k, "k")
        if any(v < 0 for v in ks):
            raise ValueError("decouple の k は 0 以上です。")
        spring_args = f"decouple {ks[0]:g} {ks[1]:g} {ks[2]:g}"

    if mode == "com_com":
        if group_id_2 is None:
            raise ValueError("com_com モードには group_id_2 が必要です。")
        if int(group_id_2) == int(group_id):
            raise ValueError("com_com の 2 つのグループは別々にしてください。")
        if continue_id is not None:
            raise ValueError("com_com モードは continue に対応していません。")
        return (
            f"add_spring com_com {_nonneg(grouping_method, 'grouping_method')} "
            f"{_nonneg(group_id, 'group_id')} {_nonneg(group_id_2, 'group_id_2')} {spring_args}"
        )

    vx, vy, vz = _vector3(ghost_velocity, "ghost_velocity")
    ox, oy, oz = _vector3(offset, "offset")
    tail = f" continue {_nonneg(continue_id, 'continue_id')}" if continue_id is not None else ""
    return (
        f"add_spring {mode} {_nonneg(grouping_method, 'grouping_method')} "
        f"{_nonneg(group_id, 'group_id')} {vx:g} {vy:g} {vz:g} {spring_args} "
        f"{ox:g} {oy:g} {oz:g}{tail}"
    )


def deposit(
    interval: int,
    direction: int | str,
    height_min: float,
    height_max: float | None = None,
    *,
    atoms: Sequence[tuple] = (),
    file: str | None = None,
    file_velocity: float | None = None,
) -> str:
    """成膜・照射のように原子を周期的に追加する。

    Parameters
    ----------
    direction
        ``'x'`` / ``'y'`` / ``'z'`` (または 0/1/2)。``file`` を使う場合のみ
        ``-1`` を指定でき、ファイル中の座標・速度をそのまま使う。
    atoms
        ``[('H', 10, -0.1), ('C', 5, -0.1, 12.0)]`` のように
        ``(元素, 個数, 速度 [Å/fs][, 質量])`` を並べる。
    file
        追加する原子を書いたファイル (``element x y z vx vy vz [mass]``)。

    Notes
    -----
    ``run.in`` 中で 1 回だけ、かつ ``run`` も 1 つだけのときに使える。
    ``run`` のステップ数は ``interval`` で割り切れる必要がある。
    """
    if direction == -1 or direction == "-1":
        if file is None:
            raise ValueError("direction=-1 はファイル指定 (file=...) のときだけ使えます。")
        if file_velocity is not None:
            raise ValueError("direction=-1 では file_velocity を指定できません。")
        index = -1
    elif isinstance(direction, int):
        if direction not in (0, 1, 2):
            raise ValueError("direction は 0 (x) / 1 (y) / 2 (z) / -1 です。")
        index = direction
    else:
        key = str(direction).strip().lower()
        if key not in _DIRECTION_INDEX:
            raise ValueError("direction は 'x' / 'y' / 'z' / -1 です。")
        index = _DIRECTION_INDEX[key]

    parts = [f"deposit {_positive(interval, 'interval')} {index} {float(height_min):g}"]
    if height_max is not None:
        if height_max < height_min:
            raise ValueError("height_max は height_min 以上です。")
        parts.append(f"{float(height_max):g}")

    if file is not None:
        if atoms:
            raise ValueError("atoms と file は同時に指定できません。")
        parts.append(f"file {file}")
        if file_velocity is not None:
            parts.append(f"{float(file_velocity):g}")
        return " ".join(parts)

    if not atoms:
        raise ValueError("atoms か file のどちらかを指定してください。")
    parts.append("atom")
    for item in atoms:
        if len(item) not in (3, 4):
            raise ValueError(
                "atoms の各要素は (元素, 個数, 速度) か (元素, 個数, 速度, 質量) です。"
            )
        element, count, velocity_value = item[0], item[1], item[2]
        parts.append(f"{element} {_positive(count, '個数')} {float(velocity_value):g}")
        if len(item) == 4:
            parts.append(f"{float(item[3]):g}")
    return " ".join(parts)


def electron_stop(filename: str) -> str:
    """高エネルギー原子に電子阻止能 (electronic stopping) を効かせる (照射損傷)。

    ``filename`` の 1 行目に ``N E_min E_max``、続けて N 行の阻止能を書く。
    列数はポテンシャルの元素数と一致させる。
    """
    return f"electron_stop {filename}"


def plumed(plumed_file: str, interval: int = 1, restart: bool = False) -> str:
    """PLUMED プラグインを呼ぶ (メタダイナミクスなどの拡張サンプリング)。

    バイアスをかける場合は ``interval`` を必ず 1 にすること。
    """
    return (
        f"plumed {plumed_file} {_positive(interval, 'interval')} {int(bool(restart))}"
    )


# ------------------------------------------------------------ NEP の不確かさ
def compute_extrapolation(
    nep_file: str,
    asi_file: str,
    *,
    gamma_low: float = 0.0,
    gamma_high: float | None = None,
    check_interval: int = 1,
    dump_interval: int = 1,
) -> str:
    """NEP の外挿グレード gamma (学習集合に対する不確かさ) を評価する。

    ``gamma_low`` を超えた構造を ``extrapolation_dump.xyz`` に書き、
    ``gamma_high`` を超えたら MD を止める。
    ``asi_file`` (Active Set Inversion) は nep_active スクリプトで作る。
    """
    if gamma_low < 0:
        raise ValueError("gamma_low は 0 以上です。")
    parts = [
        f"compute_extrapolation nep_file {nep_file} asi_file {asi_file}",
        f"gamma_low {gamma_low:g}",
    ]
    if gamma_high is not None:
        if gamma_high <= gamma_low:
            raise ValueError("gamma_high は gamma_low より大きくしてください。")
        parts.append(f"gamma_high {gamma_high:g}")
    parts.append(f"check_interval {_positive(check_interval, 'check_interval')}")
    parts.append(f"dump_interval {_positive(dump_interval, 'dump_interval')}")
    return " ".join(parts)


# ------------------------------------------------------------ Monte Carlo (組成)
def _mc_group(group: Sequence[int] | None) -> str:
    if group is None:
        return ""
    pair = tuple(int(v) for v in group)
    if len(pair) != 2:
        raise ValueError("group は (grouping_method, group_id) の 2 要素です。")
    return f"group {pair[0]} {pair[1]}"


def _mc_head(kind: str, md_steps: int, mc_trials: int, T_i: float, T_f: float | None) -> str:
    if T_f is None:
        T_f = T_i
    return (
        f"mc {kind} {_positive(md_steps, 'md_steps')} "
        f"{_positive(mc_trials, 'mc_trials')} {T_i:g} {T_f:g}"
    )


def _mc_species(species: Mapping[str, float], label: str) -> str:
    items = list(species.items())
    if not 2 <= len(items) <= 4:
        raise ValueError(f"{label} は 2-4 種類で指定します (与えられたのは {len(items)} 種)。")
    parts = [str(len(items))]
    for symbol, value in items:
        parts.append(f"{symbol} {float(value):g}")
    return " ".join(parts)


def mc_canonical(
    md_steps: int,
    mc_trials: int,
    T_i: float,
    T_f: float | None = None,
    *,
    group: Sequence[int] | None = None,
) -> str:
    """カノニカル MC (組成を変えずに原子を交換する) を MD に挟む。

    ``md_steps`` ステップごとに ``mc_trials`` 回の交換を試みる。
    短距離規則度や偏析の平衡化に使う。NEP 専用。
    """
    return " ".join(
        p for p in [_mc_head("canonical", md_steps, mc_trials, T_i, T_f), _mc_group(group)] if p
    )


def mc_sgc(
    md_steps: int,
    mc_trials: int,
    T_i: float,
    chemical_potentials: Mapping[str, float],
    T_f: float | None = None,
    *,
    group: Sequence[int] | None = None,
) -> str:
    """半グランドカノニカル MC。化学ポテンシャル差 [eV] を与えて組成を動かす。"""
    return " ".join(
        p
        for p in [
            _mc_head("sgc", md_steps, mc_trials, T_i, T_f),
            _mc_species(chemical_potentials, "chemical_potentials"),
            _mc_group(group),
        ]
        if p
    )


def mc_vcsgc(
    md_steps: int,
    mc_trials: int,
    T_i: float,
    phi: Mapping[str, float],
    kappa: float = 100.0,
    T_f: float | None = None,
    *,
    group: Sequence[int] | None = None,
) -> str:
    """分散拘束半グランドカノニカル MC。

    ``phi`` (目安 -1.2 ～ +1.2) が平均組成を、``kappa`` (目安 100) が
    組成のゆらぎを拘束する。二相共存領域でも安定にサンプリングできる。
    """
    if kappa <= 0:
        raise ValueError("kappa は正の値です (目安 100)。")
    return " ".join(
        p
        for p in [
            _mc_head("vcsgc", md_steps, mc_trials, T_i, T_f),
            _mc_species(phi, "phi"),
            f"{kappa:g}",
            _mc_group(group),
        ]
        if p
    )


# ------------------------------------------------------------------ 構造最適化
def minimize(
    method: str = "fire",
    force_tolerance: float = 1e-4,
    max_steps: int = 1000,
    *,
    box_change: bool = False,
    hydrostatic_strain: bool = False,
) -> str:
    """エネルギー最小化を行う。``potential`` より後に置くこと。

    Parameters
    ----------
    force_tolerance
        力の最大成分 [eV/Å] がこれを下回ると打ち切る。負値で必ず
        ``max_steps`` 回まわす。
    box_change
        セルも最適化するか (FIRE のみ)。
    hydrostatic_strain
        セル最適化を等方ひずみに限るか。
    """
    if method not in MINIMIZE_METHODS:
        raise ValueError(f"minimize の method は {MINIMIZE_METHODS} です。")
    if box_change and method != "fire":
        raise ValueError("box_change は FIRE 法でのみ使えます。")
    if hydrostatic_strain and not box_change:
        raise ValueError("hydrostatic_strain を使うには box_change=True が必要です。")
    parts = [f"minimize {method} {force_tolerance:g} {_positive(max_steps, 'max_steps')}"]
    if box_change:
        parts.append("1")
        if hydrostatic_strain:
            parts.append("1")
    return " ".join(parts)
