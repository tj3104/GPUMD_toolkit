"""``run.in`` の組み立て。

GPUMD の ``run.in`` は「キーワード行の並び + ``run`` で 1 ステージ」という
構造をとる。多くのキーワードは *propagating* ではない (次の ``run`` に
引き継がれない) ため、ステージごとに必要な出力設定を書き直す必要がある。
本モジュールはその煩雑さを吸収する。

対応アンサンブル
----------------
NVE   : ``nve``
NVT   : ``nvt_ber``, ``nvt_nhc``, ``nvt_bdp``, ``nvt_lan``, ``nvt_bao``, ``nvt_mttk``
NPT   : ``npt_ber``, ``npt_scr``, ``npt_mttk``
NPH   : ``nph_mttk``

熱浴・圧浴の時定数 (追加依頼 20260921-2)
-----------------------------------------
GPUMD の ``<T_coup>`` / ``<p_coup>`` / ``tperiod`` / ``pperiod`` はいずれも
**時間刻みを単位とする無次元量** (:math:`\\tau/\\Delta t`) であって時間ではない。
本モジュールでは無次元量をそのまま与えることも、``tau_T`` / ``tau_p`` に
**fs** で与えて自動換算させることもできる。

セル形状と変調軸 (追加依頼 20260921-3, 4)
-------------------------------------------
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

import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
]

NVT_ENSEMBLES = ("nvt_ber", "nvt_nhc", "nvt_bdp", "nvt_lan", "nvt_bao", "nvt_mttk")
NPT_ENSEMBLES = ("npt_ber", "npt_scr", "npt_mttk")
NPH_ENSEMBLES = ("nph_mttk",)
ALL_ENSEMBLES = ("nve",) + NVT_ENSEMBLES + NPT_ENSEMBLES + NPH_ENSEMBLES

#: ``npt_ber`` / ``npt_scr`` のセル自由度。成分数に対応する。
CELL_MODES = {"iso": 1, "ortho": 3, "tri": 6}

#: 6 成分指定の並び (GPUMD の順序)
VOIGT_LABELS = ("xx", "yy", "zz", "yz", "xz", "xy")

#: 軸名の別名 -> Voigt ラベル
_AXIS_ALIASES = {
    "x": "xx", "xx": "xx",
    "y": "yy", "yy": "yy",
    "z": "zz", "zz": "zz",
    "yz": "yz", "zy": "yz",
    "xz": "xz", "zx": "xz",
    "xy": "xy", "yx": "xy",
}

#: ``npt_mttk`` のセル変形モード
MTTK_DIRECTIONS = ("iso", "aniso", "tri")

#: ``npt_mttk`` で個別に指定できる成分
MTTK_AXES = ("x", "y", "z", "xy", "yz", "xz")

#: この値より大きい弾性率を渡すと GPUMD はその成分を固定する (> 2000 GPa)
FROZEN_MODULUS = 1.0e4


def _normalize_axis(name: str) -> str:
    key = str(name).strip().lower()
    if key not in _AXIS_ALIASES:
        raise ValueError(
            f"未知の軸名 '{name}'。使用可能: x, y, z, yz, xz, xy (= xx, yy, zz, ...)"
        )
    return _AXIS_ALIASES[key]


def _mttk_axis(name: str) -> str:
    """Voigt ラベルを ``npt_mttk`` の軸名 (``x`` / ``y`` / ``z`` / 剪断) に直す。"""
    voigt = _normalize_axis(name)
    return {"xx": "x", "yy": "y", "zz": "z"}.get(voigt, voigt)


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


def _scalar(value: float | Sequence[float]) -> float:
    """静水圧としての代表値 (対角成分の平均)。"""
    if isinstance(value, (int, float)):
        return float(value)
    seq = tuple(float(v) for v in value)
    return sum(seq[:3]) / min(3, len(seq))

#: dump_xyz で出力できる per-atom 量
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


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


@dataclass
class DumpSettings:
    """1 ステージ分の出力設定。

    Parameters
    ----------
    thermo_interval
        ``thermo.out`` への出力間隔 (ステップ)。``None`` で出力しない。
    traj_interval
        トラジェクトリ (``dump_xyz``) の出力間隔。``None`` で出力しない。
        mission.md の「任意ステップあたりで保存」はここで指定する。
    traj_file
        トラジェクトリのファイル名。
    traj_properties
        トラジェクトリに含める per-atom 量。座標は常に出力される。
    traj_precision
        ``'single'`` (既定, 9 桁) または ``'double'`` (17 桁)。
    restart_interval
        ``restart.xyz`` の出力間隔。``None`` で出力しない。
    extra
        そのまま ``run.in`` に差し込む追加キーワード行。
        例: ``["compute_msd 10 200", "compute_rdf 100 100 5.0"]``
    """

    thermo_interval: int | None = 100
    traj_interval: int | None = 1000
    traj_file: str = "dump.xyz"
    traj_properties: Sequence[str] = ("velocity", "force", "potential")
    traj_precision: str = "single"
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
            lines.append(f"dump_thermo {int(self.thermo_interval)}")
        if self.traj_interval:
            props = " ".join(self.traj_properties)
            precision = (
                "" if self.traj_precision == "single" else f" precision {self.traj_precision}"
            )
            lines.append(
                f"dump_xyz {int(self.traj_interval)} {self.traj_file}{precision}"
                + (f" {props}" if props else "")
            )
        if self.restart_interval:
            lines.append(f"dump_restart {int(self.restart_interval)}")
        lines.extend(self.extra)
        return lines


@dataclass
class MDStage:
    """1 つの ``run`` ブロック。

    Parameters
    ----------
    ensemble
        :data:`ALL_ENSEMBLES` のいずれか。
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
    label
        ログ・解析でステージを識別する名前。
    """

    ensemble: str
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
        if self.ensemble not in ALL_ENSEMBLES and self.raw_ensemble_args is None:
            raise ValueError(
                f"未知のアンサンブル '{self.ensemble}'。"
                f" 使用可能: {', '.join(ALL_ENSEMBLES)}"
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
        """時定数 [fs] を GPUMD の無次元カップリング定数 (tau/dt) に直す。"""
        if tau_fs is None:
            return float(fallback)
        dt = self.resolved_time_step(time_step)
        if dt is None or dt <= 0:
            raise ValueError(
                f"{name} を fs で指定するには time_step が必要です。"
                " MDStage(time_step=...) か RunInputBuilder(time_step=...) を設定してください。"
            )
        value = float(tau_fs) / float(dt)
        if value < 1.0:
            raise ValueError(
                f"{name}={tau_fs} fs は時間刻み {dt} fs に対して短すぎます"
                f" (GPUMD は tau/dt >= 1 を要求)。"
            )
        return value

    def temperature_coupling(self, time_step: float | None = None) -> float:
        """このステージの :math:`\\tau_T/\\Delta t`。"""
        return self._coupling(self.tau_T, self.T_coup, time_step, "tau_T")

    def pressure_coupling(self, time_step: float | None = None) -> float:
        """このステージの :math:`\\tau_p/\\Delta t` (ber/scr なら p_coup、mttk なら pperiod)。"""
        fallback = self.p_period if self.ensemble.endswith("_mttk") else self.p_coup
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

        start = self.pressure if self.pressure is not None else 0.0
        end = self.pressure_end if self.pressure_end is not None else start

        # 1) iso / aniso / tri : 静水圧をひとつだけ取る
        if isinstance(direction, str) and direction in MTTK_DIRECTIONS:
            p1, p2 = _scalar(start), _scalar(end)
            return f"{direction} {p1:g} {p2:g}"

        # 2) 軸ごとの指定
        if isinstance(direction, Mapping):
            pairs = [
                (_mttk_axis(axis), float(value), float(value))
                for axis, value in direction.items()
            ]
        else:
            axes = (direction,) if isinstance(direction, str) else tuple(direction)
            axes = tuple(_mttk_axis(a) for a in axes)
            starts = self._axis_values(start, axes)
            ends = self._axis_values(end, axes)
            pairs = list(zip(axes, starts, ends))
        if not pairs:
            raise ValueError("npt_mttk の direction が空です。")
        return " ".join(f"{axis} {p1:g} {p2:g}" for axis, p1, p2 in pairs)

    @staticmethod
    def _axis_values(
        value: float | Sequence[float], axes: Sequence[str]
    ) -> tuple[float, ...]:
        if isinstance(value, (int, float)):
            return tuple(float(value) for _ in axes)
        seq = tuple(float(v) for v in value)
        if len(seq) == len(axes):
            return seq
        if len(seq) in (3, 6):  # Voigt 並びから該当軸を拾う
            return tuple(seq[VOIGT_LABELS.index(_normalize_axis(a))] for a in axes)
        raise ValueError(
            f"pressure の成分数 {len(seq)} が direction の軸数 {len(axes)} と合いません。"
        )

    # ------------------------------------------------------------------ 出力
    def ensemble_line(self, time_step: float | None = None) -> str:
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


@dataclass
class RunInputBuilder:
    """``run.in`` 全体を組み立てる。

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
    minimize: tuple[str, float, int] | None = None
    stages: list[MDStage] = field(default_factory=list)
    default_dump: DumpSettings = field(default_factory=DumpSettings)
    header_comments: Sequence[str] = field(default_factory=tuple)
    preamble: Sequence[str] = field(default_factory=tuple)

    # ------------------------------------------------------------------ 構築
    def add_stage(self, stage: MDStage) -> "RunInputBuilder":
        self.stages.append(stage)
        return self

    def add_stages(self, stages: Iterable[MDStage]) -> "RunInputBuilder":
        for stage in stages:
            self.add_stage(stage)
        return self

    def set_minimize(
        self, method: str = "fire", force_tolerance: float = 1e-4, max_steps: int = 1000
    ) -> "RunInputBuilder":
        if method not in ("fire", "sd"):
            raise ValueError("minimize の method は 'fire' か 'sd' です。")
        self.minimize = (method, force_tolerance, int(max_steps))
        return self

    # ------------------------------------------------------------------ 出力
    def potential_lines(self) -> list[str]:
        if isinstance(self.potential, (str, Path)):
            return [f"potential {self.potential}"]
        return [f"potential {' '.join(str(p) for p in self.potential)}"]

    def build(self) -> str:
        if not self.stages:
            raise ValueError("ステージが 1 つもありません。add_stage() を呼んでください。")
        lines: list[str] = []
        for comment in self.header_comments:
            lines.append(f"# {comment}")
        if self.header_comments:
            lines.append("")
        lines.extend(self.potential_lines())
        if self.initial_temperature is not None:
            seed = f" seed {self.velocity_seed}" if self.velocity_seed is not None else ""
            lines.append(f"velocity {self.initial_temperature:g}{seed}")
        lines.append(f"time_step {self.time_step:g}")
        lines.extend(self.preamble)
        if self.minimize is not None:
            method, tol, steps = self.minimize
            lines.append("")
            lines.append("# --- energy minimization ---")
            lines.append(f"minimize {method} {tol:g} {steps}")
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
