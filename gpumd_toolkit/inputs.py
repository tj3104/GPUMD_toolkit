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
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "DumpSettings",
    "MDStage",
    "RunInputBuilder",
    "NVT_ENSEMBLES",
    "NPT_ENSEMBLES",
    "ALL_ENSEMBLES",
]

NVT_ENSEMBLES = ("nvt_ber", "nvt_nhc", "nvt_bdp", "nvt_lan", "nvt_bao", "nvt_mttk")
NPT_ENSEMBLES = ("npt_ber", "npt_scr", "npt_mttk")
ALL_ENSEMBLES = ("nve",) + NVT_ENSEMBLES + NPT_ENSEMBLES

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
        温度カップリング定数。Berendsen/NHC/BDP 系では「ステップ数」単位の
        無次元量 (GPUMD の慣習で ~100)。mttk 系では周期 [fs]。
    pressure
        目標圧力 [GPa]。スカラー (等方) か、3 成分 (xx, yy, zz)、
        6 成分 (xx, yy, zz, yz, xz, xy) のいずれか。
    elastic_modulus
        ``npt_ber`` / ``npt_scr`` で必要な弾性率の概算値 [GPa]。
        ``pressure`` と同じ長さのスカラー/シーケンス。
    p_coup
        圧力カップリング定数 (``npt_ber`` / ``npt_scr``)。
    p_period
        ``npt_mttk`` の barostat 周期 [fs]。
    mttk_direction
        ``npt_mttk`` のセル変形方向 (``iso`` / ``aniso`` / ``tri`` / ``x`` など)。
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
    pressure: float | Sequence[float] | None = None
    elastic_modulus: float | Sequence[float] = 100.0
    p_coup: float = 1000.0
    p_period: float = 1000.0
    mttk_direction: str = "iso"
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
        if self.ensemble != "nve" and self.T_start is None:
            raise ValueError(f"{self.ensemble} には T_start が必要です。")
        if self.T_end is None:
            self.T_end = self.T_start
        if self.ensemble in NPT_ENSEMBLES and self.pressure is None:
            self.pressure = 0.0
        if not self.label:
            self.label = self._auto_label()

    def _auto_label(self) -> str:
        if self.ensemble == "nve":
            return "nve"
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

    def ensemble_line(self) -> str:
        if self.raw_ensemble_args is not None:
            return f"ensemble {self.ensemble} {self.raw_ensemble_args}".strip()
        if self.ensemble == "nve":
            return "ensemble nve"

        T1, T2 = float(self.T_start), float(self.T_end)

        if self.ensemble == "nvt_mttk":
            return f"ensemble nvt_mttk temp {T1:g} {T2:g} tperiod {self.T_coup:g}"

        if self.ensemble in NVT_ENSEMBLES:
            return f"ensemble {self.ensemble} {T1:g} {T2:g} {self.T_coup:g}"

        # --- NPT ---
        if self.ensemble == "npt_mttk":
            press = self.pressure
            if isinstance(press, (int, float)):
                p_args = f"{self.mttk_direction} {float(press):g} {float(press):g}"
            else:
                press = tuple(float(p) for p in press)
                if len(press) == 3 and self.mttk_direction in ("aniso", "tri"):
                    # aniso/tri は開始圧・終了圧の 2 値のみを取るため平均で代表させる
                    mean = sum(press) / 3.0
                    p_args = f"{self.mttk_direction} {mean:g} {mean:g}"
                else:
                    labels = ("x", "y", "z")
                    p_args = " ".join(
                        f"{lab} {p:g} {p:g}" for lab, p in zip(labels, press)
                    )
            return (
                f"ensemble npt_mttk temp {T1:g} {T2:g} {p_args} "
                f"tperiod {self.T_coup:g} pperiod {self.p_period:g}"
            )

        # npt_ber / npt_scr
        press = self.pressure
        n = 1 if isinstance(press, (int, float)) else len(tuple(press))
        if n not in (1, 3, 6):
            raise ValueError("pressure はスカラー、3 成分、6 成分のいずれかです。")
        p_vals = self._as_tuple(press, n)
        c_vals = self._as_tuple(self.elastic_modulus, n)
        args = " ".join(f"{p:g}" for p in p_vals)
        args += " " + " ".join(f"{c:g}" for c in c_vals)
        args += f" {self.p_coup:g}"
        return f"ensemble {self.ensemble} {T1:g} {T2:g} {self.T_coup:g} {args}"

    def to_lines(self, default_dump: DumpSettings | None = None) -> list[str]:
        lines: list[str] = [f"# --- stage: {self.label} ---"]
        if self.time_step is not None:
            lines.append(f"time_step {self.time_step:g}")
        lines.append(self.ensemble_line())
        lines.extend(self.pre_commands)
        dump = self.dump if self.dump is not None else default_dump
        if dump is not None:
            lines.extend(dump.to_lines())
        lines.append(f"run {self.steps}")
        lines.extend(self.post_commands)
        return lines

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
        for stage in self.stages:
            lines.append("")
            lines.extend(stage.to_lines(self.default_dump))
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
