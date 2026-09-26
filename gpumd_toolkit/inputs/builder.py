"""``run.in`` の組み立て。

GPUMD の ``run.in`` は「キーワード行の並び + ``run`` で 1 ステージ」という
構造をとる。多くのキーワードは *propagating* ではない (次の ``run`` に
引き継がれない) ため、ステージごとに必要な出力設定を書き直す必要がある。
本モジュールはその煩雑さを吸収する。

アンサンブルの指定方法
----------------------
* **文字列** — 標準アンサンブル (:data:`ALL_ENSEMBLES`) はそのまま名前で指定する。
  NVE / NVT (6 種) / NPT (3 種) / NPH。
* **オブジェクト** — それ以外 (QTB・NEMD 熱浴・TTM・PIMD・熱力学的積分・衝撃波) は
  :mod:`gpumd_toolkit.inputs.ensembles` のクラスのインスタンスを渡す。

熱浴・圧浴の時定数
------------------
GPUMD の ``<T_coup>`` / ``<p_coup>`` / ``tperiod`` / ``pperiod`` はいずれも
**時間刻みを単位とする無次元量** (:math:`\\tau/\\Delta t`) であって時間ではない。
本モジュールでは無次元量をそのまま与えることも、``tau_T`` / ``tau_p`` に
**fs** で与えて自動換算させることもできる。

セル形状と変調軸
----------------
* ``npt_ber`` / ``npt_scr`` は圧力成分の数でセルの動き方が決まる。
  1 成分 = 等方 (直交セル必須)、3 成分 = 直方晶 (直交セル必須)、
  6 成分 = 三斜晶 (任意形状のセルに対応)。:attr:`MDStage.cell_mode` で選ぶ。
* ある成分の弾性率を 2000 GPa より大きくすると GPUMD はその成分の
  カップリングを 0 にする = **その軸を固定する**。
  :attr:`MDStage.fixed_axes` / :attr:`MDStage.free_axes` はこれを使う。
* ``npt_mttk`` は ``iso`` / ``aniso`` / ``tri`` のほか
  ``x`` ``y`` ``z`` ``xy`` ``yz`` ``xz`` を並べて成分ごとに指定できる。
  指定しなかった成分は固定される。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .dumps import DUMP_PROPERTIES, DumpSettings
from .ensembles import (
    CELL_MODES,
    FROZEN_MODULUS,
    MTTK_AXES,
    MTTK_DIRECTIONS,
    VOIGT_LABELS,
    Ensemble,
    coupling_from_tau,
    mttk_axis,
    mttk_direction_args,
    normalize_axis,
)
from .modifiers import minimize as _minimize_line
from .modifiers import potential as _potential_line
from .modifiers import replicate as _replicate_line
from .modifiers import velocity as _velocity_line

__all__ = [
    "DumpSettings",
    "MDStage",
    "RunInputBuilder",
    "NVT_ENSEMBLES",
    "NPT_ENSEMBLES",
    "NPH_ENSEMBLES",
    "ALL_ENSEMBLES",
    "CELL_MODES",
    "VOIGT_LABELS",
    "MTTK_DIRECTIONS",
    "MTTK_AXES",
    "FROZEN_MODULUS",
    "DUMP_PROPERTIES",
]

NVT_ENSEMBLES = ("nvt_ber", "nvt_nhc", "nvt_bdp", "nvt_lan", "nvt_bao", "nvt_mttk")
NPT_ENSEMBLES = ("npt_ber", "npt_scr", "npt_mttk")
NPH_ENSEMBLES = ("nph_mttk",)
#: 文字列で ``MDStage`` に渡せるアンサンブル
ALL_ENSEMBLES = ("nve",) + NVT_ENSEMBLES + NPT_ENSEMBLES + NPH_ENSEMBLES

_normalize_axis = normalize_axis
_mttk_axis = mttk_axis


def _bad_pressure(length: int):
    raise ValueError(
        f"pressure はスカラー、3 成分 (xx, yy, zz)、"
        f" 6 成分 (xx, yy, zz, yz, xz, xy) のいずれかです (与えられたのは {length} 個)。"
    )


def _mean(value: float | Sequence[float]) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    seq = tuple(float(v) for v in value)
    return sum(seq) / len(seq) if seq else 0.0


@dataclass
class MDStage:
    """1 つの ``run`` ブロック。

    Parameters
    ----------
    ensemble
        :data:`ALL_ENSEMBLES` のいずれかの名前、または
        :class:`gpumd_toolkit.inputs.ensembles.Ensemble` のインスタンス。
    steps
        ステップ数。
    T_start, T_end
        目標温度 [K]。NVE では不要。``T_end`` 省略時は ``T_start`` と同じ(定温)。
    T_coup
        温度カップリング定数 :math:`\\tau_T/\\Delta t` (無次元、GPUMD の慣習で ~100)。
        ``nvt_mttk`` / ``npt_mttk`` の ``tperiod`` も同じ意味。
    tau_T
        温度の時定数 :math:`\\tau_T` を **fs** で指定する場合はこちら。
        与えると ``T_coup = tau_T / time_step`` として上書きする。
    pressure
        目標圧力 [GPa]。スカラー (等方) か、3 成分 (xx, yy, zz)、
        6 成分 (xx, yy, zz, yz, xz, xy) のいずれか。
    pressure_end
        ``npt_mttk`` / ``nph_mttk`` で圧力を線形に変化させる場合の終圧 [GPa]。
    elastic_modulus
        ``npt_ber`` / ``npt_scr`` で必要な弾性率の概算値 [GPa]。
        ``pressure`` と同じ長さのスカラー/シーケンス。桁が合っていればよい。
    p_coup
        圧力カップリング定数 :math:`\\tau_p/\\Delta t` (``npt_ber`` / ``npt_scr``、
        無次元、目安 ~1000)。
    p_period
        ``npt_mttk`` / ``nph_mttk`` の ``pperiod`` :math:`\\tau_p/\\Delta t`
        (無次元、200 以上が推奨)。
    tau_p
        圧力の時定数 :math:`\\tau_p` を **fs** で指定する場合はこちら。
        与えると ``p_coup`` (ber/scr) または ``p_period`` (mttk) を上書きする。
    cell_mode
        ``npt_ber`` / ``npt_scr`` のセル自由度。
        ``'iso'`` (1 成分・直交セル) / ``'ortho'`` (3 成分・直交セル) /
        ``'tri'`` (6 成分・三斜晶セル可)。``None`` なら ``pressure`` の長さと
        ``fixed_axes`` から自動で決める。
    fixed_axes, free_axes
        変調を許す / 禁じる軸。``('x', 'y')`` や ``('xy', 'yz')`` のように指定する。
        ``npt_ber`` / ``npt_scr`` では固定軸の弾性率を 2000 GPa 超に設定して
        GPUMD 側のカップリングを 0 にすることで実現する。
        ``npt_mttk`` では ``free_axes`` がそのまま ``direction`` になる。
    mttk_direction
        ``npt_mttk`` / ``nph_mttk`` のセル変形方向。
        ``'iso'`` / ``'aniso'`` / ``'tri'`` のほか、
        ``('x', 'y')`` のような軸のリストや ``{'x': 5.0, 'y': 0.0}`` の
        ような軸ごとの圧力 [GPa] でも指定できる。
    time_step
        このステージの時間刻み [fs]。``None`` なら直前の設定を引き継ぐ。
    dump
        出力設定。``None`` なら :class:`RunInputBuilder` の既定値を使う。
    raw_ensemble_args
        ``ensemble <name> ...`` の引数を完全に手書きしたい場合に指定する。
    pre_commands / post_commands
        ``run`` の直前/直後に差し込む任意の行。
        :mod:`gpumd_toolkit.inputs.computes` や
        :mod:`gpumd_toolkit.inputs.modifiers` の戻り値を並べる。
    label
        ログ・解析でステージを識別する名前。
    """

    ensemble: str | Ensemble
    steps: int
    T_start: float | None = None
    T_end: float | None = None
    T_coup: float = 100.0
    tau_T: float | None = None
    pressure: float | Sequence[float] | None = None
    pressure_end: float | Sequence[float] | None = None
    elastic_modulus: float | Sequence[float] = 100.0
    p_coup: float = 1000.0
    p_period: float = 1000.0
    tau_p: float | None = None
    cell_mode: str | None = None
    fixed_axes: Sequence[str] = field(default_factory=tuple)
    free_axes: Sequence[str] | None = None
    mttk_direction: str | Sequence[str] | Mapping[str, float] = "iso"
    time_step: float | None = None
    dump: DumpSettings | None = None
    raw_ensemble_args: str | None = None
    pre_commands: Sequence[str] = field(default_factory=tuple)
    post_commands: Sequence[str] = field(default_factory=tuple)
    label: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.ensemble, Ensemble):
            if self.steps <= 0:
                raise ValueError("steps は正の整数である必要があります。")
            self.steps = int(self.steps)
            if not self.label:
                self.label = self.ensemble.describe()
            return
        if self.ensemble not in ALL_ENSEMBLES and self.raw_ensemble_args is None:
            raise ValueError(
                f"未知のアンサンブル '{self.ensemble}'。"
                f" 文字列で指定できるのは {', '.join(ALL_ENSEMBLES)} です。"
                " それ以外は gpumd_toolkit.inputs.ensembles のクラスを使ってください。"
            )
        if self.steps <= 0:
            raise ValueError("steps は正の整数である必要があります。")
        self.steps = int(self.steps)
        needs_temperature = self.ensemble not in ("nve",) + NPH_ENSEMBLES
        if needs_temperature and self.T_start is None:
            raise ValueError(f"{self.ensemble} には T_start が必要です。")
        if self.T_end is None:
            self.T_end = self.T_start
        if self.ensemble in NPT_ENSEMBLES + NPH_ENSEMBLES and self.pressure is None:
            self.pressure = 0.0
        if self.cell_mode is not None and self.cell_mode not in CELL_MODES:
            raise ValueError(f"cell_mode は {tuple(CELL_MODES)} から選びます。")
        if self.fixed_axes and self.free_axes is not None:
            raise ValueError("fixed_axes と free_axes は同時に指定できません。")
        self.fixed_axes = tuple(_normalize_axis(a) for a in self.fixed_axes)
        if self.free_axes is not None:
            self.free_axes = tuple(_normalize_axis(a) for a in self.free_axes)
        if not self.label:
            self.label = self._auto_label()

    # ------------------------------------------------------------------ 名前
    @property
    def ensemble_name(self) -> str:
        """``ensemble`` がオブジェクトでも文字列の名前を返す。"""
        if isinstance(self.ensemble, Ensemble):
            return self.ensemble.name
        return self.ensemble

    @property
    def ensemble_spec(self) -> Ensemble | None:
        """オブジェクト指定のアンサンブル (文字列指定なら ``None``)。"""
        return self.ensemble if isinstance(self.ensemble, Ensemble) else None

    def _auto_label(self) -> str:
        if self.ensemble == "nve":
            return "nve"
        if self.ensemble in NPH_ENSEMBLES:
            return f"{self.ensemble}"
        if abs((self.T_end or 0) - (self.T_start or 0)) < 1e-9:
            return f"{self.ensemble}_{self.T_start:g}K"
        return f"{self.ensemble}_{self.T_start:g}K_to_{self.T_end:g}K"

    # ------------------------------------------------------------------ helper
    @staticmethod
    def _as_tuple(value: float | Sequence[float], n: int) -> tuple[float, ...]:
        if isinstance(value, (int, float)):
            return tuple(float(value) for _ in range(n))
        seq = tuple(float(v) for v in value)
        if len(seq) != n:
            raise ValueError(f"{n} 成分が必要ですが {len(seq)} 個与えられました。")
        return seq

    def resolved_time_step(self, time_step: float | None = None) -> float | None:
        """このステージに実際に効く時間刻み [fs]。"""
        return self.time_step if self.time_step is not None else time_step

    def _coupling(
        self, tau_fs: float | None, fallback: float, time_step: float | None, name: str
    ) -> float:
        return coupling_from_tau(tau_fs, fallback, self.resolved_time_step(time_step), name)

    def temperature_coupling(self, time_step: float | None = None) -> float:
        """このステージの :math:`\\tau_T/\\Delta t`。"""
        return self._coupling(self.tau_T, self.T_coup, time_step, "tau_T")

    def pressure_coupling(self, time_step: float | None = None) -> float:
        """このステージの :math:`\\tau_p/\\Delta t` (ber/scr なら p_coup、mttk なら pperiod)。"""
        fallback = (
            self.p_period if self.ensemble_name.endswith("_mttk") else self.p_coup
        )
        return self._coupling(self.tau_p, fallback, time_step, "tau_p")

    # ------------------------------------------------------- 圧力成分とセル自由度
    def resolve_cell_mode(self) -> str:
        """``npt_ber`` / ``npt_scr`` の圧力成分の数 (セル自由度) を決める。"""
        if self.cell_mode is not None:
            return self.cell_mode
        axes = tuple(self.fixed_axes) + tuple(self.free_axes or ())
        if any(axis in ("yz", "xz", "xy") for axis in axes):
            return "tri"
        if axes:
            return "ortho"
        if self.pressure is None or isinstance(self.pressure, (int, float)):
            return "iso"
        length = len(tuple(self.pressure))
        return {1: "iso", 3: "ortho", 6: "tri"}.get(length) or _bad_pressure(length)

    @staticmethod
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
                _bad_pressure(len(seq))
        if n == 6:
            return tuple(values)
        if n == 3:
            return tuple(values[:3])
        # 等方: 静水圧 (対角成分の平均)
        return (sum(values[:3]) / 3.0,)

    def pressure_components(self) -> tuple[str, tuple[float, ...], tuple[float, ...]]:
        """``(cell_mode, 目標圧力, 弾性率)`` を返す (``npt_ber`` / ``npt_scr`` 用)。

        固定したい軸の弾性率は :data:`FROZEN_MODULUS` に置き換える。
        GPUMD は 2000 GPa を超える弾性率を見るとその成分のカップリングを 0 に
        するので、これが「変調可能な軸を選ぶ」正攻法になる。
        """
        mode = self.resolve_cell_mode()
        n = CELL_MODES[mode]
        pressures = self._expand(self.pressure if self.pressure is not None else 0.0, n, pad=0.0)
        moduli = list(self._expand(self.elastic_modulus, n, pad=_mean(self.elastic_modulus)))

        fixed = set(self.fixed_axes)
        if self.free_axes is not None:
            fixed = set(VOIGT_LABELS[:n]) - set(self.free_axes)
        if fixed:
            if n == 1:
                raise ValueError(
                    "等方 (cell_mode='iso') では軸を個別に固定できません。"
                    " cell_mode='ortho' か 'tri' を指定してください。"
                )
            for axis in sorted(fixed):
                index = VOIGT_LABELS.index(axis)
                if index >= n:
                    raise ValueError(
                        f"軸 '{axis}' を固定するには cell_mode='tri' が必要です。"
                    )
                moduli[index] = FROZEN_MODULUS
        return mode, pressures, tuple(moduli)

    # ------------------------------------------------------------- MTTK の方向
    def mttk_pressure_args(self) -> str:
        """``npt_mttk`` / ``nph_mttk`` の ``<direction> <p_1> <p_2>`` を組み立てる。"""
        direction = self.mttk_direction
        if self.free_axes is not None and direction in MTTK_DIRECTIONS:
            direction = tuple(_mttk_axis(a) for a in self.free_axes)
        return mttk_direction_args(direction, self.pressure, self.pressure_end)

    # ------------------------------------------------------------------ 出力
    def ensemble_line(self, time_step: float | None = None) -> str:
        spec = self.ensemble_spec
        if spec is not None:
            return spec.line(self.resolved_time_step(time_step))
        if self.raw_ensemble_args is not None:
            return f"ensemble {self.ensemble} {self.raw_ensemble_args}".strip()
        if self.ensemble == "nve":
            return "ensemble nve"

        t_coup = self.temperature_coupling(time_step)

        if self.ensemble == "nph_mttk":
            p_args = self.mttk_pressure_args()
            return (
                f"ensemble nph_mttk {p_args} "
                f"pperiod {self.pressure_coupling(time_step):g}"
            )

        T1, T2 = float(self.T_start), float(self.T_end)

        if self.ensemble == "nvt_mttk":
            return f"ensemble nvt_mttk temp {T1:g} {T2:g} tperiod {t_coup:g}"

        if self.ensemble in NVT_ENSEMBLES:
            return f"ensemble {self.ensemble} {T1:g} {T2:g} {t_coup:g}"

        # --- NPT ---
        if self.ensemble == "npt_mttk":
            p_args = self.mttk_pressure_args()
            return (
                f"ensemble npt_mttk temp {T1:g} {T2:g} {p_args} "
                f"tperiod {t_coup:g} pperiod {self.pressure_coupling(time_step):g}"
            )

        # npt_ber / npt_scr
        _, pressures, moduli = self.pressure_components()
        args = " ".join(f"{p:g}" for p in pressures)
        args += " " + " ".join(f"{c:g}" for c in moduli)
        args += f" {self.pressure_coupling(time_step):g}"
        return f"ensemble {self.ensemble} {T1:g} {T2:g} {t_coup:g} {args}"

    def to_lines(
        self, default_dump: DumpSettings | None = None, time_step: float | None = None
    ) -> list[str]:
        lines: list[str] = [f"# --- stage: {self.label} ---"]
        if self.time_step is not None:
            lines.append(f"time_step {self.time_step:g}")
        lines.append(self.ensemble_line(time_step))
        lines.extend(self.pre_commands)
        dump = self.dump if self.dump is not None else default_dump
        if dump is not None:
            lines.extend(dump.to_lines())
        lines.append(f"run {self.steps}")
        lines.extend(self.post_commands)
        return lines

    def describe_cell_control(self, time_step: float | None = None) -> str:
        """このステージがセルをどう動かすかを 1 行で説明する。"""
        if self.ensemble_spec is not None:
            return self.ensemble_spec.describe()
        if self.ensemble in NVT_ENSEMBLES or self.ensemble == "nve":
            return "セル固定"
        if self.ensemble.endswith("_mttk"):
            return f"mttk direction = {self.mttk_pressure_args()}"
        mode, pressures, moduli = self.pressure_components()
        labels = VOIGT_LABELS[: len(pressures)] if len(pressures) > 1 else ("hydro",)
        parts = [
            f"{label}: {'固定' if modulus >= 2.0e3 else f'{p:g} GPa'}"
            for label, p, modulus in zip(labels, pressures, moduli)
        ]
        return f"cell_mode={mode} ({', '.join(parts)})"

    def with_dump(self, dump: DumpSettings) -> "MDStage":
        return replace(self, dump=dump)

    def with_commands(self, *lines: str) -> "MDStage":
        """``pre_commands`` に行を足した複製を返す。"""
        return replace(self, pre_commands=tuple(self.pre_commands) + tuple(lines))

    def metadata(self) -> dict:
        """``metadata.json`` に残すための辞書表現。"""
        data = {
            "label": self.label,
            "ensemble": self.ensemble_name,
            "steps": self.steps,
        }
        spec = self.ensemble_spec
        if spec is not None:
            data.update(spec.metadata())
        else:
            data.update(
                {
                    "T_start": self.T_start,
                    "T_end": self.T_end,
                    "pressure": self.pressure,
                    "tau_T_fs": self.tau_T,
                    "tau_p_fs": self.tau_p,
                }
            )
        if self.pre_commands:
            data["pre_commands"] = list(self.pre_commands)
        if self.post_commands:
            data["post_commands"] = list(self.post_commands)
        return data


@dataclass
class RunInputBuilder:
    """``run.in`` 全体を組み立てる。

    Parameters
    ----------
    potential
        ポテンシャルファイル。複数渡すと ``potential`` 行が複数並ぶ
        (committee / ``dump_observer`` / ``active`` 用)。
        1 つのポテンシャルが複数引数を要るとき (DP など) は
        ``[["dp.txt", "model.pb"]]`` のように入れ子で渡す。
    replicate
        ``(na, nb, nc)``。``potential`` より前に ``replicate`` 行を書く。
    minimize
        ``(method, force_tolerance, max_steps)`` または
        ``(method, force_tolerance, max_steps, box_change, hydrostatic_strain)``。
    preamble
        ``potential`` / ``time_step`` の後、最初のステージの前に差し込む行。
        ``dftd3`` / ``kspace`` / ``correct_velocity`` などを置く。

    Examples
    --------
    >>> builder = RunInputBuilder(potential="nep.txt", time_step=1.0,
    ...                           initial_temperature=300)
    >>> builder.add_stage(MDStage("nvt_nhc", steps=10000, T_start=300))
    >>> print(builder.build())
    """

    potential: str | Sequence[str]
    time_step: float = 1.0
    initial_temperature: float | None = None
    velocity_seed: int | None = None
    minimize: tuple | None = None
    stages: list[MDStage] = field(default_factory=list)
    default_dump: DumpSettings = field(default_factory=DumpSettings)
    header_comments: Sequence[str] = field(default_factory=tuple)
    preamble: Sequence[str] = field(default_factory=tuple)
    replicate: Sequence[int] | None = None
    max_distance_per_step: float | None = None
    actions: Sequence[str] = field(default_factory=tuple)

    # ------------------------------------------------------------------ 構築
    def add_stage(self, stage: MDStage) -> "RunInputBuilder":
        self.stages.append(stage)
        return self

    def add_stages(self, stages: Iterable[MDStage]) -> "RunInputBuilder":
        for stage in stages:
            self.add_stage(stage)
        return self

    def add_preamble(self, *lines: str) -> "RunInputBuilder":
        """``dftd3`` / ``kspace`` / ``correct_velocity`` などを先頭に足す。"""
        self.preamble = tuple(self.preamble) + tuple(lines)
        return self

    def add_action(self, *lines: str) -> "RunInputBuilder":
        """``run`` を伴わずその場で実行されるキーワードを足す。

        ``compute_cohesive`` / ``compute_elastic`` / ``compute_phonon`` は
        GPUMD が構文解析した時点で実行されるので、``ensemble`` と ``run`` の
        ブロックを必要としない。
        """
        self.actions = tuple(self.actions) + tuple(lines)
        return self

    def set_minimize(
        self,
        method: str = "fire",
        force_tolerance: float = 1e-4,
        max_steps: int = 1000,
        *,
        box_change: bool = False,
        hydrostatic_strain: bool = False,
    ) -> "RunInputBuilder":
        if method not in ("fire", "sd"):
            raise ValueError("minimize の method は 'fire' か 'sd' です。")
        self.minimize = (
            method,
            force_tolerance,
            int(max_steps),
            bool(box_change),
            bool(hydrostatic_strain),
        )
        return self

    # ------------------------------------------------------------------ 出力
    def potential_lines(self) -> list[str]:
        """``potential`` 行 (複数ポテンシャルにも対応)。"""
        if isinstance(self.potential, (str, Path)):
            return [_potential_line(str(self.potential))]
        items = list(self.potential)
        if items and all(isinstance(item, (str, Path)) for item in items):
            # DP のように 1 つのポテンシャルが複数引数を取る場合との区別が
            # つかないため、従来どおり 1 行にまとめる。
            return [_potential_line([str(item) for item in items])]
        return [
            _potential_line(item if isinstance(item, (str, Path)) else [str(v) for v in item])
            for item in items
        ]

    @property
    def n_potentials(self) -> int:
        return len(self.potential_lines())

    def build(self) -> str:
        if not self.stages and not self.actions:
            raise ValueError(
                "ステージが 1 つもありません。add_stage() か add_action() を呼んでください。"
            )
        lines: list[str] = []
        for comment in self.header_comments:
            lines.append(f"# {comment}")
        if self.header_comments:
            lines.append("")
        if self.replicate is not None:
            na, nb, nc = (int(n) for n in self.replicate)
            lines.append(_replicate_line(na, nb, nc))
        lines.extend(self.potential_lines())
        if self.initial_temperature is not None:
            lines.append(_velocity_line(self.initial_temperature, self.velocity_seed))
        if self.max_distance_per_step is not None:
            lines.append(f"time_step {self.time_step:g} {self.max_distance_per_step:g}")
        else:
            lines.append(f"time_step {self.time_step:g}")
        lines.extend(self.preamble)
        if self.minimize is not None:
            method, tol, steps = self.minimize[0], self.minimize[1], self.minimize[2]
            box_change = bool(self.minimize[3]) if len(self.minimize) > 3 else False
            hydrostatic = bool(self.minimize[4]) if len(self.minimize) > 4 else False
            lines.append("")
            lines.append("# --- energy minimization ---")
            lines.append(
                _minimize_line(
                    method,
                    tol,
                    steps,
                    box_change=box_change,
                    hydrostatic_strain=hydrostatic,
                )
            )
        if self.actions:
            lines.append("")
            lines.append("# --- standalone actions (run 不要) ---")
            lines.extend(self.actions)
        time_step = self.time_step
        for stage in self.stages:
            lines.append("")
            lines.extend(stage.to_lines(self.default_dump, time_step))
            if stage.time_step is not None:
                time_step = stage.time_step
        return "\n".join(lines) + "\n"

    def write(self, path: Path | str) -> Path:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.build())
        return path

    @property
    def total_steps(self) -> int:
        return sum(stage.steps for stage in self.stages)

    def total_time_ps(self) -> float:
        total = 0.0
        dt = self.time_step
        for stage in self.stages:
            if stage.time_step is not None:
                dt = stage.time_step
            total += stage.steps * dt * 1e-3
        return total
