"""GPUMD の ``ensemble`` キーワード全種を組み立てる。

:mod:`~gpumd_toolkit.inputs.builder` の :class:`~gpumd_toolkit.inputs.builder.MDStage`
は「文字列のアンサンブル名 (nve / nvt_* / npt_* / nph_mttk)」をそのまま扱える。
それ以外の特殊なアンサンブル (熱伝導・PIMD・熱力学的積分・衝撃波・TTM・QTB) は
引数の構造がまったく違うので、本モジュールの :class:`Ensemble` サブクラスを
``MDStage(ensemble=...)`` に渡す形で指定する。

対応一覧
--------
=========================== ===============================================
分類                        クラス
=========================== ===============================================
標準 (NVE/NVT/NPT/NPH)      文字列指定 (:data:`STANDARD_ENSEMBLES`)
量子熱浴 (QTB)              :class:`QTB`
熱伝導 NEMD                 :class:`HeatBath`
Two-Temperature Model       :class:`TTM` / :class:`HeatTTM`
経路積分 (PIMD/RPMD/TRPMD)  :class:`PIMD` / :class:`RPMD` / :class:`TRPMD`
熱力学的積分 (自由エネルギー) :class:`TISpring` / :class:`TILiquid` /
                            :class:`TIAdiabaticSwitching` /
                            :class:`TIReversibleScaling` / :class:`TIFixedLambda`
衝撃波                      :class:`MSST` / :class:`NPHug` / :class:`Wall`
=========================== ===============================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

__all__ = [
    "Ensemble",
    "QTB",
    "HeatBath",
    "TTM",
    "TTMOptions",
    "HeatTTM",
    "PIMD",
    "RPMD",
    "TRPMD",
    "TISpring",
    "TILiquid",
    "TIAdiabaticSwitching",
    "TIReversibleScaling",
    "TIFixedLambda",
    "MSST",
    "NPHug",
    "Wall",
    "STANDARD_ENSEMBLES",
    "QTB_ENSEMBLES",
    "HEAT_ENSEMBLES",
    "PIMD_ENSEMBLES",
    "TI_ENSEMBLES",
    "SHOCK_ENSEMBLES",
    "TTM_ENSEMBLES",
    "MTTK_DIRECTIONS",
    "MTTK_AXES",
    "VOIGT_LABELS",
    "CELL_MODES",
    "FROZEN_MODULUS",
    "coupling_from_tau",
    "mttk_direction_args",
    "pressure_control_args",
    "normalize_axis",
    "mttk_axis",
]

# --------------------------------------------------------------------- 定数
#: ``MDStage(ensemble="...")`` に文字列で渡せる標準アンサンブル
STANDARD_ENSEMBLES = (
    "nve",
    "nvt_ber", "nvt_nhc", "nvt_bdp", "nvt_lan", "nvt_bao", "nvt_mttk",
    "npt_ber", "npt_scr", "npt_mttk",
    "nph_mttk",
)
QTB_ENSEMBLES = ("nvt_qtb", "npt_qtb")
HEAT_ENSEMBLES = ("heat_nhc", "heat_bdp", "heat_lan")
TTM_ENSEMBLES = ("ttm", "heat_ttm")
PIMD_ENSEMBLES = ("pimd", "pimd_scr", "rpmd", "trpmd")
TI_ENSEMBLES = ("ti", "ti_spring", "ti_liquid", "ti_as", "ti_rs")
SHOCK_ENSEMBLES = ("msst", "nphug", "wall_piston", "wall_mirror", "wall_harmonic")

#: ``npt_ber`` / ``npt_scr`` のセル自由度。値は圧力成分の数。
CELL_MODES = {"iso": 1, "ortho": 3, "tri": 6}

#: 6 成分指定の並び (GPUMD の順序)
VOIGT_LABELS = ("xx", "yy", "zz", "yz", "xz", "xy")

#: MTTK 系 (``npt_mttk`` / ``nph_mttk`` / ``npt_qtb`` / ``nphug`` / ``ti_as`` / ``ti_rs``) の一括指定
MTTK_DIRECTIONS = ("iso", "aniso", "tri")

#: MTTK 系で個別に指定できる成分
MTTK_AXES = ("x", "y", "z", "xy", "yz", "xz")

#: この値より大きい弾性率を渡すと GPUMD はその成分を固定する (> 2000 GPa)
FROZEN_MODULUS = 1.0e4

_AXIS_ALIASES = {
    "x": "xx", "xx": "xx",
    "y": "yy", "yy": "yy",
    "z": "zz", "zz": "zz",
    "yz": "yz", "zy": "yz",
    "xz": "xz", "zx": "xz",
    "xy": "xy", "yx": "xy",
}


def normalize_axis(name: str) -> str:
    """``'x'`` や ``'zy'`` を Voigt ラベル (``'xx'``, ``'yz'``) に正規化する。"""
    key = str(name).strip().lower()
    if key not in _AXIS_ALIASES:
        raise ValueError(
            f"未知の軸名 '{name}'。使用可能: x, y, z, yz, xz, xy (= xx, yy, zz, ...)"
        )
    return _AXIS_ALIASES[key]


def mttk_axis(name: str) -> str:
    """Voigt ラベルを MTTK 系の軸名 (``x`` / ``y`` / ``z`` / 剪断) に直す。"""
    voigt = normalize_axis(name)
    return {"xx": "x", "yy": "y", "zz": "z"}.get(voigt, voigt)


def _fmt(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def coupling_from_tau(
    tau_fs: float | None, fallback: float, time_step: float | None, name: str
) -> float:
    """時定数 [fs] を GPUMD の無次元カップリング定数 :math:`\\tau/\\Delta t` に直す。

    GPUMD の ``<T_coup>`` / ``<p_coup>`` / ``tperiod`` / ``pperiod`` はいずれも
    **時間刻みを単位とする無次元量** であって時間ではない。
    """
    if tau_fs is None:
        return float(fallback)
    if time_step is None or time_step <= 0:
        raise ValueError(
            f"{name} を fs で指定するには time_step が必要です。"
            " MDStage(time_step=...) か RunInputBuilder(time_step=...) を設定してください。"
        )
    value = float(tau_fs) / float(time_step)
    if value < 1.0:
        raise ValueError(
            f"{name}={tau_fs} fs は時間刻み {time_step} fs に対して短すぎます"
            f" (GPUMD は tau/dt >= 1 を要求)。"
        )
    return value


def _scalar(value: float | Sequence[float]) -> float:
    """静水圧としての代表値 (対角成分の平均)。"""
    if isinstance(value, (int, float)):
        return float(value)
    seq = tuple(float(v) for v in value)
    return sum(seq[:3]) / min(3, len(seq))


def _axis_values(value: float | Sequence[float], axes: Sequence[str]) -> tuple[float, ...]:
    if isinstance(value, (int, float)):
        return tuple(float(value) for _ in axes)
    seq = tuple(float(v) for v in value)
    if len(seq) == len(axes):
        return seq
    if len(seq) in (3, 6):  # Voigt 並びから該当軸を拾う
        return tuple(seq[VOIGT_LABELS.index(normalize_axis(a))] for a in axes)
    raise ValueError(
        f"pressure の成分数 {len(seq)} が direction の軸数 {len(axes)} と合いません。"
    )


def mttk_direction_args(
    direction: str | Sequence[str] | Mapping[str, float],
    pressure: float | Sequence[float] | None = 0.0,
    pressure_end: float | Sequence[float] | None = None,
    *,
    single_pressure: bool = False,
    pressure_pairs: bool = True,
) -> str:
    """MTTK 系の ``<direction> <p_1> [<p_2>]`` を組み立てる。

    Parameters
    ----------
    direction
        ``'iso'`` / ``'aniso'`` / ``'tri'``、``('x', 'y')`` のような軸のリスト、
        ``{'x': 5.0, 'z': 0.0}`` のような軸ごとの圧力 [GPa] のいずれか。
    pressure, pressure_end
        目標圧力 [GPa]。``pressure_end`` を省略すると ``pressure`` と同じ。
    single_pressure
        ``True`` なら ``<direction> <p>`` 形式 (``ti_rs`` 用)。
    pressure_pairs
        ``False`` なら方向だけを返す。
    """
    start = pressure if pressure is not None else 0.0
    end = pressure_end if pressure_end is not None else start

    if isinstance(direction, Mapping):
        pairs = [
            (mttk_axis(axis), float(value), float(value))
            for axis, value in direction.items()
        ]
    elif isinstance(direction, str) and direction in MTTK_DIRECTIONS:
        p1, p2 = _scalar(start), _scalar(end)
        if not pressure_pairs:
            return direction
        if single_pressure:
            return f"{direction} {p1:g}"
        return f"{direction} {p1:g} {p2:g}"
    else:
        axes = (direction,) if isinstance(direction, str) else tuple(direction)
        axes = tuple(mttk_axis(a) for a in axes)
        pairs = list(zip(axes, _axis_values(start, axes), _axis_values(end, axes)))

    if not pairs:
        raise ValueError("direction が空です。")
    for axis, _, _ in pairs:
        if axis not in MTTK_AXES:
            raise ValueError(f"MTTK 系で指定できない軸です: {axis} (使用可能: {MTTK_AXES})")
    if not pressure_pairs:
        return " ".join(axis for axis, _, _ in pairs)
    if single_pressure:
        return " ".join(f"{axis} {p1:g}" for axis, p1, _ in pairs)
    return " ".join(f"{axis} {p1:g} {p2:g}" for axis, p1, p2 in pairs)


def _expand(value: float | Sequence[float], n: int, *, pad: float) -> tuple[float, ...]:
    """スカラー / 3 成分 / 6 成分の値を n 成分に揃える。"""
    if isinstance(value, (int, float)):
        values = [float(value)] * 3 + [pad] * 3
    else:
        seq = [float(v) for v in value]
        if len(seq) == 1:
            values = seq * 3 + [pad] * 3
        elif len(seq) == 3:
            values = seq + [pad] * 3
        elif len(seq) == 6:
            values = seq
        else:
            raise ValueError(
                "pressure はスカラー、3 成分 (xx, yy, zz)、"
                f" 6 成分 (xx, yy, zz, yz, xz, xy) のいずれかです (与えられたのは {len(seq)} 個)。"
            )
    if n == 6:
        return tuple(values)
    if n == 3:
        return tuple(values[:3])
    return (sum(values[:3]) / 3.0,)


def pressure_control_args(
    pressure: float | Sequence[float],
    elastic_modulus: float | Sequence[float],
    p_coup: float,
    *,
    cell_mode: str = "iso",
    fixed_axes: Sequence[str] = (),
) -> str:
    """``npt_ber`` / ``npt_scr`` / ``pimd`` の圧力制御パラメータを組み立てる。

    ``fixed_axes`` に挙げた成分の弾性率は :data:`FROZEN_MODULUS` に置き換える
    (GPUMD は 2000 GPa 超の弾性率を見るとその成分のカップリングを 0 にする)。
    """
    if cell_mode not in CELL_MODES:
        raise ValueError(f"cell_mode は {tuple(CELL_MODES)} から選びます。")
    n = CELL_MODES[cell_mode]
    pressures = _expand(pressure, n, pad=0.0)
    mean_modulus = (
        float(elastic_modulus)
        if isinstance(elastic_modulus, (int, float))
        else sum(float(v) for v in elastic_modulus) / len(tuple(elastic_modulus))
    )
    moduli = list(_expand(elastic_modulus, n, pad=mean_modulus))
    for axis in {normalize_axis(a) for a in fixed_axes}:
        index = VOIGT_LABELS.index(axis)
        if index >= n:
            raise ValueError(f"軸 '{axis}' を固定するには cell_mode='tri' が必要です。")
        moduli[index] = FROZEN_MODULUS
    parts = [f"{p:g}" for p in pressures] + [f"{c:g}" for c in moduli] + [f"{p_coup:g}"]
    return " ".join(parts)


# --------------------------------------------------------------------- 基底
@dataclass
class Ensemble:
    """``ensemble`` 行を組み立てる基底クラス。

    サブクラスは :attr:`name` と :meth:`args` を実装する。
    """

    @property
    def name(self) -> str:  # pragma: no cover - サブクラスが実装
        raise NotImplementedError

    def args(self, time_step: float | None = None) -> str:  # pragma: no cover
        raise NotImplementedError

    def line(self, time_step: float | None = None) -> str:
        """``ensemble <name> <args>`` の 1 行を返す。"""
        args = self.args(time_step).strip()
        return f"ensemble {self.name} {args}".strip()

    def describe(self) -> str:
        """人が読む 1 行説明。"""
        return self.name

    def metadata(self) -> dict:
        """``metadata.json`` に残すための辞書表現。"""
        data = {"ensemble": self.name, "description": self.describe()}
        for key in ("T_start", "T_end", "temperature", "T_min", "T_max", "pressure",
                    "num_beads", "shock_velocity", "vp", "lambda_value"):
            value = getattr(self, key, None)
            if value is not None and not callable(value):
                data[key] = value
        return data


# ------------------------------------------------------------------ QTB (核量子効果)
@dataclass
class QTB(Ensemble):
    """Quantum Thermal Bath: 色つきノイズで零点振動を近似的に取り込む Langevin 熱浴。

    Parameters
    ----------
    mode
        ``'nvt'`` (``nvt_qtb``) または ``'npt'`` (``npt_qtb``)。
    T_start, T_end
        目標温度 [K]。``T_end`` 省略時は定温。
    T_coup, tau_T
        熱浴の時定数。無次元 (:math:`\\tau_T/\\Delta t`) か **fs**。
    direction, pressure, pressure_end
        ``npt_qtb`` の圧力制御 (MTTK と同じ書式)。
    p_period, tau_p
        圧浴の時定数 (``npt_qtb``、200 以上推奨)。
    f_max
        QTB フィルタの最大振動数 [ps^-1]。系の最大フォノン振動数より大きく取る。
    n_f
        フィルタの周波数点数。

    Notes
    -----
    QTB では零点エネルギーの分だけ ``thermo.out`` の運動温度が目標温度より
    高く出るのが正常な挙動 (水 300 K で 1000 K 程度)。
    """

    T_start: float = 300.0
    T_end: float | None = None
    mode: str = "nvt"
    T_coup: float = 100.0
    tau_T: float | None = None
    direction: str | Sequence[str] | Mapping[str, float] = "iso"
    pressure: float | Sequence[float] = 0.0
    pressure_end: float | Sequence[float] | None = None
    p_period: float = 1000.0
    tau_p: float | None = None
    f_max: float | None = None
    n_f: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("nvt", "npt"):
            raise ValueError("QTB の mode は 'nvt' か 'npt' です。")
        if self.T_end is None:
            self.T_end = self.T_start

    @property
    def name(self) -> str:
        return f"{self.mode}_qtb"

    def _filter_args(self) -> str:
        parts = []
        if self.f_max is not None:
            parts.append(f"f_max {self.f_max:g}")
        if self.n_f is not None:
            parts.append(f"N_f {int(self.n_f)}")
        return " ".join(parts)

    def args(self, time_step: float | None = None) -> str:
        t_coup = coupling_from_tau(self.tau_T, self.T_coup, time_step, "tau_T")
        if self.mode == "nvt":
            head = f"{self.T_start:g} {self.T_end:g} {t_coup:g}"
        else:
            p_period = coupling_from_tau(self.tau_p, self.p_period, time_step, "tau_p")
            if p_period < 200:
                raise ValueError("npt_qtb の pperiod は 200 ステップ以上にしてください。")
            direction = mttk_direction_args(
                self.direction, self.pressure, self.pressure_end
            )
            head = (
                f"{direction} temp {self.T_start:g} {self.T_end:g} "
                f"tperiod {t_coup:g} pperiod {p_period:g}"
            )
        return f"{head} {self._filter_args()}".strip()

    def describe(self) -> str:
        return f"QTB ({self.mode}) {self.T_start:g}->{self.T_end:g} K"


# --------------------------------------------------------- 熱伝導 NEMD (source/sink)
@dataclass
class HeatBath(Ensemble):
    """NEMD 熱伝導用の局所熱浴 (``heat_nhc`` / ``heat_bdp`` / ``heat_lan``)。

    ``label_source`` のグループを :math:`T + \\Delta T`、``label_sink`` を
    :math:`T - \\Delta T` に保つ。両ラベルは **grouping method 0** を指す。
    """

    temperature: float
    delta_T: float
    source: int
    sink: int
    method: str = "lan"
    T_coup: float = 100.0
    tau_T: float | None = None

    def __post_init__(self) -> None:
        if self.method not in ("nhc", "bdp", "lan"):
            raise ValueError("HeatBath の method は 'nhc' / 'bdp' / 'lan' です。")
        if int(self.source) == int(self.sink):
            raise ValueError("source と sink には別のグループラベルを指定してください。")
        if self.delta_T <= 0:
            raise ValueError("delta_T は正の値にしてください。")

    @property
    def name(self) -> str:
        return f"heat_{self.method}"

    def args(self, time_step: float | None = None) -> str:
        t_coup = coupling_from_tau(self.tau_T, self.T_coup, time_step, "tau_T")
        return (
            f"{self.temperature:g} {t_coup:g} {self.delta_T:g} "
            f"{int(self.source)} {int(self.sink)}"
        )

    def describe(self) -> str:
        return (
            f"NEMD 熱浴 {self.name}: source(g{self.source})="
            f"{self.temperature + self.delta_T:g} K / sink(g{self.sink})="
            f"{self.temperature - self.delta_T:g} K"
        )


# ------------------------------------------------------------ Two-Temperature Model
@dataclass
class TTMOptions:
    """:class:`TTM` / :class:`HeatTTM` の追加オプション。"""

    out_interval: int | None = None
    infile: str | None = None
    properties_file: str | None = None
    source: float | None = None
    active_x: str | int | None = None
    active_y: str | int | None = None
    active_z: str | int | None = None

    def to_args(self) -> str:
        parts: list[str] = []
        if self.out_interval is not None:
            parts += ["ttm_out_interval", str(int(self.out_interval))]
        if self.infile:
            parts += ["ttm_infile", str(self.infile)]
        if self.properties_file:
            parts += ["ttm_properties_file", str(self.properties_file)]
        if self.source is not None:
            parts += ["ttm_source", f"{float(self.source):g}"]
        for axis in ("x", "y", "z"):
            value = getattr(self, f"active_{axis}")
            if value is not None:
                parts += [f"ttm_active_{axis}", str(value)]
        return " ".join(parts)


@dataclass
class TTM(Ensemble):
    """二温度モデル (電子温度グリッドに結合した MD)。

    Parameters
    ----------
    grouping_method, group_id
        電子グリッドに結合する原子グループ。
    Ce
        電子 1 個あたりの比熱 [eV/K]。
    rho_e
        電子数密度 [1/Å^3]。``Ce * rho_e`` が体積比熱 [eV/(K Å^3)]。
    kappa_e
        電子熱伝導率 [eV/(ps K Å)]。
    gamma_p, gamma_s
        摩擦係数 [amu/ps]。
    v_0
        しきい速度 [Å/ps]。
    grid
        電子グリッドの分割数 ``(nx, ny, nz)``。
    T_e_init
        初期電子温度 [K]。
    options
        :class:`TTMOptions` (出力間隔・入力ファイル・熱源・活性領域)。
    """

    Ce: float
    rho_e: float
    kappa_e: float
    gamma_p: float
    gamma_s: float
    v_0: float
    grid: Sequence[int]
    T_e_init: float
    grouping_method: int = 0
    group_id: int = 0
    options: TTMOptions = field(default_factory=TTMOptions)

    def __post_init__(self) -> None:
        if len(tuple(self.grid)) != 3:
            raise ValueError("grid は (nx, ny, nz) の 3 要素です。")
        if any(int(n) < 1 for n in self.grid):
            raise ValueError("grid の各成分は 1 以上の整数です。")

    @property
    def name(self) -> str:
        return "ttm"

    def _core_args(self) -> str:
        nx, ny, nz = (int(n) for n in self.grid)
        return (
            f"{int(self.grouping_method)} {int(self.group_id)} "
            f"{self.Ce:g} {self.rho_e:g} {self.kappa_e:g} "
            f"{self.gamma_p:g} {self.gamma_s:g} {self.v_0:g} "
            f"{nx} {ny} {nz} {self.T_e_init:g}"
        )

    def args(self, time_step: float | None = None) -> str:
        return f"{self._core_args()} {self.options.to_args()}".strip()

    def describe(self) -> str:
        nx, ny, nz = (int(n) for n in self.grid)
        return f"TTM grid={nx}x{ny}x{nz} T_e(0)={self.T_e_init:g} K"


@dataclass
class HeatTTM(TTM):
    """局所 source/sink Langevin 熱浴 + 電子温度グリッド (``heat_ttm``)。"""

    temperature: float = 300.0
    delta_T: float = 10.0
    source: int = 0
    sink: int = 1
    T_coup: float = 100.0
    tau_T: float | None = None

    @property
    def name(self) -> str:
        return "heat_ttm"

    def args(self, time_step: float | None = None) -> str:
        t_coup = coupling_from_tau(self.tau_T, self.T_coup, time_step, "tau_T")
        head = (
            f"{self.temperature:g} {t_coup:g} {self.delta_T:g} "
            f"{int(self.source)} {int(self.sink)}"
        )
        return f"{head} {self._core_args()} {self.options.to_args()}".strip()

    def describe(self) -> str:
        return f"heat_ttm ({super().describe()}) T={self.temperature:g}±{self.delta_T:g} K"


# ------------------------------------------------------------------ 経路積分 (PIMD)
@dataclass
class PIMD(Ensemble):
    """経路積分 MD。核の量子効果 (零点振動・トンネル) を取り込む。

    Parameters
    ----------
    num_beads
        ring polymer のビーズ数。128 以下の正の偶数。
    T_start, T_end, T_coup, tau_T
        温度と熱浴の時定数。
    pressure
        与えると NPT になる (``pimd`` は Berendsen、``pimd_scr`` は
        stochastic cell rescaling の圧浴)。``None`` なら NVT。
    stochastic_cell_rescaling
        ``True`` で ``pimd_scr``。
    eco_omega_max
        economised path-integral (Eco) の最大振動数 [cm^-1]。
        ``None`` なら従来の Trotter 振動数。
    """

    num_beads: int
    T_start: float
    T_end: float | None = None
    T_coup: float = 100.0
    tau_T: float | None = None
    pressure: float | Sequence[float] | None = None
    elastic_modulus: float | Sequence[float] = 100.0
    p_coup: float = 1000.0
    tau_p: float | None = None
    cell_mode: str = "iso"
    fixed_axes: Sequence[str] = ()
    stochastic_cell_rescaling: bool = False
    eco_omega_max: float | None = None

    def __post_init__(self) -> None:
        beads = int(self.num_beads)
        if beads <= 0 or beads > 128 or beads % 2 != 0:
            raise ValueError("num_beads は 128 以下の正の偶数にしてください。")
        self.num_beads = beads
        if self.T_end is None:
            self.T_end = self.T_start
        if self.stochastic_cell_rescaling and self.pressure is None:
            raise ValueError("pimd_scr には pressure が必要です (圧力制御つき PIMD)。")

    @property
    def name(self) -> str:
        return "pimd_scr" if self.stochastic_cell_rescaling else "pimd"

    def args(self, time_step: float | None = None) -> str:
        t_coup = coupling_from_tau(self.tau_T, self.T_coup, time_step, "tau_T")
        parts = [f"{self.num_beads}", f"{self.T_start:g}", f"{self.T_end:g}", f"{t_coup:g}"]
        if self.pressure is not None:
            p_coup = coupling_from_tau(self.tau_p, self.p_coup, time_step, "tau_p")
            parts.append(
                pressure_control_args(
                    self.pressure,
                    self.elastic_modulus,
                    p_coup,
                    cell_mode=self.cell_mode,
                    fixed_axes=self.fixed_axes,
                )
            )
        if self.eco_omega_max is not None:
            parts.append(f"eco {self.eco_omega_max:g}")
        return " ".join(parts)

    def describe(self) -> str:
        kind = "NPT" if self.pressure is not None else "NVT"
        eco = f", eco {self.eco_omega_max:g} cm^-1" if self.eco_omega_max else ""
        return f"PIMD {self.num_beads} beads ({kind}){eco}"


@dataclass
class RPMD(Ensemble):
    """Ring-polymer MD (PIMD の NVE 版、熱浴なし)。"""

    num_beads: int

    def __post_init__(self) -> None:
        if int(self.num_beads) <= 0:
            raise ValueError("num_beads は正の整数です。")
        self.num_beads = int(self.num_beads)

    @property
    def name(self) -> str:
        return "rpmd"

    def args(self, time_step: float | None = None) -> str:
        return str(self.num_beads)

    def describe(self) -> str:
        return f"RPMD {self.num_beads} beads"


@dataclass
class TRPMD(RPMD):
    """内部モードに Langevin 熱浴をかけた ring-polymer MD。"""

    @property
    def name(self) -> str:
        return "trpmd"

    def describe(self) -> str:
        return f"TRPMD {self.num_beads} beads"


# --------------------------------------------------------- 熱力学的積分 (自由エネルギー)
def _switch_steps(t_equil: int | None, t_switch: int | None) -> tuple[int | None, int | None]:
    if t_equil is not None and t_equil < 0:
        raise ValueError("tequil は 0 以上です。")
    if t_switch is not None and t_switch <= 0:
        raise ValueError("tswitch は正の整数です。")
    return t_equil, t_switch


@dataclass
class _TIBase(Ensemble):
    """非平衡熱力学的積分の共通部分。"""

    t_equil: int | None = None
    t_switch: int | None = None

    @property
    def total_steps(self) -> int | None:
        """``run`` に渡すべきステップ数 (往復するので 2*(tequil+tswitch))。"""
        if self.t_equil is None or self.t_switch is None:
            return None
        return 2 * (int(self.t_equil) + int(self.t_switch))

    def _switch_args(self) -> str:
        parts = []
        if self.t_switch is not None:
            parts.append(f"tswitch {int(self.t_switch)}")
        if self.t_equil is not None:
            parts.append(f"tequil {int(self.t_equil)}")
        return " ".join(parts)


@dataclass
class TISpring(_TIBase):
    """Frenkel-Ladd 経路 (Einstein 結晶を参照系) による固体の自由エネルギー。

    ``ti_spring.csv`` (lambda, dlambda, pe, espring) と
    ``ti_spring.yaml`` (E_Einstein, E_diff, F, T, V, P, G) を出力する。

    Parameters
    ----------
    temperature
        温度 [K]。
    pressure
        Gibbs 自由エネルギーの計算に使う圧力 [GPa] (ダイナミクスには影響しない)。
    spring
        元素ごとのばね定数 [eV/Å^2]。``None`` なら GPUMD が MSD から自動決定。
    """

    temperature: float = 300.0
    pressure: float = 0.0
    T_coup: float = 100.0
    tau_T: float | None = None
    spring: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        _switch_steps(self.t_equil, self.t_switch)

    @property
    def name(self) -> str:
        return "ti_spring"

    def args(self, time_step: float | None = None) -> str:
        t_coup = coupling_from_tau(self.tau_T, self.T_coup, time_step, "tau_T")
        parts = [f"temp {self.temperature:g}", f"tperiod {t_coup:g}", self._switch_args()]
        parts.append(f"press {self.pressure:g}")
        if self.spring:
            # spring は必ず末尾に置く (GPUMD の仕様)
            springs = " ".join(f"{el} {float(k):g}" for el, k in self.spring.items())
            parts.append(f"spring {springs}")
        return " ".join(p for p in parts if p)

    def describe(self) -> str:
        return f"TI (Frenkel-Ladd) T={self.temperature:g} K, P={self.pressure:g} GPa"


@dataclass
class TILiquid(_TIBase):
    """Uhlenbeck-Ford 経路による液体の自由エネルギー。

    ``ti_liquid.csv`` (lambda, dlambda, pe, eUF) と ``ti_liquid.yaml`` を出力する。
    """

    temperature: float = 1000.0
    pressure: float = 0.0
    T_coup: float = 100.0
    tau_T: float | None = None
    sigma_sqrd: float = 2.0
    p: int = 100

    _ALLOWED_P = (1, 25, 50, 75, 100)

    def __post_init__(self) -> None:
        _switch_steps(self.t_equil, self.t_switch)
        if int(self.p) not in self._ALLOWED_P:
            raise ValueError(f"p は {self._ALLOWED_P} のいずれかです。")

    @property
    def name(self) -> str:
        return "ti_liquid"

    def args(self, time_step: float | None = None) -> str:
        t_coup = coupling_from_tau(self.tau_T, self.T_coup, time_step, "tau_T")
        return " ".join(
            p
            for p in [
                f"temp {self.temperature:g}",
                f"tperiod {t_coup:g}",
                self._switch_args(),
                f"press {self.pressure:g}",
                f"sigmasqrd {self.sigma_sqrd:g}",
                f"p {int(self.p)}",
            ]
            if p
        )

    def describe(self) -> str:
        return f"TI (Uhlenbeck-Ford, 液体) T={self.temperature:g} K"


@dataclass
class TIAdiabaticSwitching(_TIBase):
    """断熱スイッチング (AS) 経路: 等温線に沿って Gibbs 自由エネルギー G(P) を得る。

    ``ti_as.csv`` (p, V) を出力する。
    """

    temperature: float = 300.0
    p_min: float = 0.0
    p_max: float = 10.0
    direction: str | Sequence[str] = "aniso"
    T_coup: float = 100.0
    tau_T: float | None = None
    p_period: float = 1000.0
    tau_p: float | None = None

    def __post_init__(self) -> None:
        _switch_steps(self.t_equil, self.t_switch)

    @property
    def name(self) -> str:
        return "ti_as"

    def args(self, time_step: float | None = None) -> str:
        t_coup = coupling_from_tau(self.tau_T, self.T_coup, time_step, "tau_T")
        p_period = coupling_from_tau(self.tau_p, self.p_period, time_step, "tau_p")
        direction = mttk_direction_args(self.direction, pressure_pairs=False)
        return " ".join(
            p
            for p in [
                f"temp {self.temperature:g}",
                f"tperiod {t_coup:g}",
                f"{direction} {self.p_min:g} {self.p_max:g}",
                f"pperiod {p_period:g}",
                self._switch_args(),
            ]
            if p
        )

    def describe(self) -> str:
        return (
            f"TI (断熱スイッチング) T={self.temperature:g} K, "
            f"P: {self.p_min:g} -> {self.p_max:g} GPa"
        )


@dataclass
class TIReversibleScaling(_TIBase):
    """可逆スケーリング (RS) 経路: 等圧線に沿って Gibbs 自由エネルギー G(T) を得る。

    ``ti_rs.csv`` (lambda, dlambda, enthalpy) を出力する。
    """

    T_min: float = 300.0
    T_max: float = 3000.0
    pressure: float = 0.0
    direction: str | Sequence[str] = "aniso"
    T_coup: float = 100.0
    tau_T: float | None = None
    p_period: float = 1000.0
    tau_p: float | None = None

    def __post_init__(self) -> None:
        _switch_steps(self.t_equil, self.t_switch)
        if self.T_min <= 0:
            raise ValueError("T_min は正の値です (G は T0/lambda で外挿されます)。")

    @property
    def name(self) -> str:
        return "ti_rs"

    def args(self, time_step: float | None = None) -> str:
        t_coup = coupling_from_tau(self.tau_T, self.T_coup, time_step, "tau_T")
        p_period = coupling_from_tau(self.tau_p, self.p_period, time_step, "tau_p")
        direction = mttk_direction_args(
            self.direction, self.pressure, single_pressure=True
        )
        return " ".join(
            p
            for p in [
                f"temp {self.T_min:g} {self.T_max:g}",
                f"tperiod {t_coup:g}",
                direction,
                f"pperiod {p_period:g}",
                self._switch_args(),
            ]
            if p
        )

    def describe(self) -> str:
        return (
            f"TI (可逆スケーリング) T: {self.T_min:g} -> {self.T_max:g} K, "
            f"P={self.pressure:g} GPa"
        )


@dataclass
class TIFixedLambda(Ensemble):
    """lambda を固定した平衡熱力学的積分 (検証用)。``ti.csv`` (pe, espring) を出力する。"""

    lambda_value: float = 0.5
    temperature: float = 300.0
    T_coup: float = 100.0
    tau_T: float | None = None
    spring: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.lambda_value <= 1.0:
            raise ValueError("lambda は 0 以上 1 以下です。")

    @property
    def name(self) -> str:
        return "ti"

    def args(self, time_step: float | None = None) -> str:
        t_coup = coupling_from_tau(self.tau_T, self.T_coup, time_step, "tau_T")
        parts = [
            f"lambda {self.lambda_value:g}",
            f"temp {self.temperature:g}",
            f"tperiod {t_coup:g}",
        ]
        if self.spring:
            springs = " ".join(f"{el} {float(k):g}" for el, k in self.spring.items())
            parts.append(f"spring {springs}")
        return " ".join(parts)

    def describe(self) -> str:
        return f"TI (lambda={self.lambda_value:g} 固定) T={self.temperature:g} K"


# ------------------------------------------------------------------ 高圧・衝撃波
@dataclass
class MSST(Ensemble):
    """Multi-Scale Shock Technique: 小さいセルで定常衝撃波後方の状態を再現する。"""

    direction: str = "x"
    shock_velocity: float = 15.0
    qmass: float = 10000.0
    mu: float = 10.0
    tscale: float | None = None
    p0: float | None = None
    v0: float | None = None
    e0: float | None = None

    def __post_init__(self) -> None:
        if self.direction not in ("x", "y", "z"):
            raise ValueError("MSST の direction は 'x' / 'y' / 'z' です。")
        if self.shock_velocity <= 0:
            raise ValueError("shock_velocity [km/s] は正の値です。")

    @property
    def name(self) -> str:
        return "msst"

    def args(self, time_step: float | None = None) -> str:
        parts = [
            self.direction,
            f"{self.shock_velocity:g}",
            f"qmass {self.qmass:g}",
            f"mu {self.mu:g}",
        ]
        if self.tscale is not None:
            parts.append(f"tscale {self.tscale:g}")
        for key in ("p0", "v0", "e0"):
            value = getattr(self, key)
            if value is not None:
                parts.append(f"{key} {value:g}")
        return " ".join(parts)

    def describe(self) -> str:
        return f"MSST {self.direction} 方向 us={self.shock_velocity:g} km/s"


@dataclass
class NPHug(Ensemble):
    """Hugoniostat: 目標応力を与えて Hugoniot 上の状態に収束させる。"""

    direction: str | Sequence[str] = "iso"
    pressure: float = 300.0
    T_period: float = 100.0
    tau_T: float | None = None
    p_period: float = 1000.0
    tau_p: float | None = None
    p0: float | None = None
    v0: float | None = None
    e0: float | None = None

    @property
    def name(self) -> str:
        return "nphug"

    def args(self, time_step: float | None = None) -> str:
        t_period = coupling_from_tau(self.tau_T, self.T_period, time_step, "tau_T")
        p_period = coupling_from_tau(self.tau_p, self.p_period, time_step, "tau_p")
        direction = mttk_direction_args(self.direction, self.pressure, self.pressure)
        parts = [direction, f"tperiod {t_period:g}", f"pperiod {p_period:g}"]
        for key in ("p0", "v0", "e0"):
            value = getattr(self, key)
            if value is not None:
                parts.append(f"{key} {value:g}")
        return " ".join(parts)

    def describe(self) -> str:
        return f"NPHug P={self.pressure:g} GPa"


@dataclass
class Wall(Ensemble):
    """NEMD 衝撃波: 動く壁 (ピストン) で衝撃波を x 方向に発生させる。

    Parameters
    ----------
    kind
        ``'piston'`` (固定原子層) / ``'mirror'`` (運動量ミラー) /
        ``'harmonic'`` (調和ポテンシャル壁、ミラーより柔らかい)。
    vp
        ピストン速度 [km/s]。
    thickness
        ``'piston'`` の壁の厚み [Å]。
    k
        ``'harmonic'`` の壁のばね定数 [eV/Å^2]。

    Notes
    -----
    衝撃波は **x 方向** に伝播する。x 方向を非周期にしてはいけない
    (GPUMD が真空層を自動で付ける)。:func:`~gpumd_toolkit.inputs.dumps.dump_shock_nemd`
    と併用して空間分布を取るのが定石。
    """

    vp: float
    kind: str = "piston"
    thickness: float | None = None
    k: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("piston", "mirror", "harmonic"):
            raise ValueError("Wall の kind は 'piston' / 'mirror' / 'harmonic' です。")
        if self.vp <= 0:
            raise ValueError("vp [km/s] は正の値です。")
        if self.kind != "piston" and self.thickness is not None:
            raise ValueError("thickness は wall_piston でのみ指定できます。")
        if self.kind != "harmonic" and self.k is not None:
            raise ValueError("k は wall_harmonic でのみ指定できます。")

    @property
    def name(self) -> str:
        return f"wall_{self.kind}"

    def args(self, time_step: float | None = None) -> str:
        parts = [f"vp {self.vp:g}"]
        if self.thickness is not None:
            parts.append(f"thickness {self.thickness:g}")
        if self.k is not None:
            parts.append(f"k {self.k:g}")
        return " ".join(parts)

    def describe(self) -> str:
        return f"NEMD 衝撃波 ({self.name}) vp={self.vp:g} km/s"
