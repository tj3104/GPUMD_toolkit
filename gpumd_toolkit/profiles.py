"""温度プロファイル (昇温・降温・定温・任意プロファイル) の定義。

GPUMD の ``ensemble nvt_xxx <T1> <T2> <T_coup>`` は 1 つの ``run`` の間に
目標温度を ``T1`` から ``T2`` へ線形に変化させる。したがって任意の温度
プロファイルは「区分線形なセグメントの並び」= 複数の run ブロックとして
表現できる。本モジュールはその区分線形セグメントを組み立てる。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = ["TemperatureSegment", "TemperatureProfile"]


@dataclass
class TemperatureSegment:
    """1 つの ``run`` ブロックに対応する温度区間。"""

    T_start: float
    T_end: float
    steps: int
    label: str = ""

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError(f"steps は正の整数である必要があります: {self.steps}")
        if self.T_start < 0 or self.T_end < 0:
            raise ValueError("温度は非負である必要があります。")
        self.steps = int(self.steps)
        if not self.label:
            if abs(self.T_end - self.T_start) < 1e-9:
                self.label = f"hold_{self.T_start:g}K"
            elif self.T_end > self.T_start:
                self.label = f"heat_{self.T_start:g}K_to_{self.T_end:g}K"
            else:
                self.label = f"cool_{self.T_start:g}K_to_{self.T_end:g}K"

    @property
    def is_ramp(self) -> bool:
        return abs(self.T_end - self.T_start) > 1e-9

    def duration_ps(self, time_step_fs: float) -> float:
        return self.steps * time_step_fs * 1e-3

    def rate_K_per_ps(self, time_step_fs: float) -> float:
        duration = self.duration_ps(time_step_fs)
        return (self.T_end - self.T_start) / duration if duration > 0 else 0.0


@dataclass
class TemperatureProfile:
    """区分線形の温度プロファイル。

    Examples
    --------
    >>> TemperatureProfile.isothermal(300, 10000)
    >>> TemperatureProfile.heating(300, 1200, 50000)
    >>> TemperatureProfile.anneal(300, 1200, heat_steps=20000,
    ...                           hold_steps=20000, cool_steps=40000)
    >>> TemperatureProfile.from_points([(0, 300), (10, 1000), (20, 1000), (40, 300)],
    ...                                time_step_fs=1.0)
    """

    segments: list[TemperatureSegment] = field(default_factory=list)
    name: str = "profile"

    # ------------------------------------------------------------ constructors
    @classmethod
    def isothermal(cls, temperature: float, steps: int, *, name: str = "") -> "TemperatureProfile":
        """定温。"""
        return cls(
            [TemperatureSegment(temperature, temperature, steps)],
            name=name or f"isothermal_{temperature:g}K",
        )

    @classmethod
    def ramp(
        cls,
        T_start: float,
        T_end: float,
        steps: int,
        *,
        n_segments: int = 1,
        name: str = "",
    ) -> "TemperatureProfile":
        """線形の昇温/降温。

        ``n_segments`` を増やすと区間が分割され、thermostat の目標温度は
        同じ線形勾配のままステージが細分化される (途中出力を増やしたい場合に有用)。
        """
        if n_segments < 1:
            raise ValueError("n_segments は 1 以上である必要があります。")
        edges = np.linspace(T_start, T_end, n_segments + 1)
        per = steps // n_segments
        remainder = steps - per * n_segments
        segments = []
        for i in range(n_segments):
            n = per + (1 if i < remainder else 0)
            segments.append(TemperatureSegment(float(edges[i]), float(edges[i + 1]), n))
        kind = "heating" if T_end >= T_start else "cooling"
        return cls(segments, name=name or f"{kind}_{T_start:g}K_{T_end:g}K")

    @classmethod
    def heating(cls, T_start: float, T_end: float, steps: int, **kwargs) -> "TemperatureProfile":
        """昇温 (``T_end > T_start`` を想定)。"""
        return cls.ramp(T_start, T_end, steps, **kwargs)

    @classmethod
    def cooling(cls, T_start: float, T_end: float, steps: int, **kwargs) -> "TemperatureProfile":
        """降温 (``T_end < T_start`` を想定)。"""
        return cls.ramp(T_start, T_end, steps, **kwargs)

    @classmethod
    def anneal(
        cls,
        T_low: float,
        T_high: float,
        *,
        heat_steps: int,
        hold_steps: int,
        cool_steps: int,
        equilibrate_steps: int = 0,
        name: str = "",
    ) -> "TemperatureProfile":
        """低温平衡 → 昇温 → 高温保持 → 降温、というアニール。"""
        segments: list[TemperatureSegment] = []
        if equilibrate_steps > 0:
            segments.append(TemperatureSegment(T_low, T_low, equilibrate_steps, "equilibrate"))
        segments.append(TemperatureSegment(T_low, T_high, heat_steps))
        if hold_steps > 0:
            segments.append(TemperatureSegment(T_high, T_high, hold_steps))
        segments.append(TemperatureSegment(T_high, T_low, cool_steps))
        return cls(segments, name=name or f"anneal_{T_low:g}K_{T_high:g}K")

    @classmethod
    def from_points(
        cls,
        points: Sequence[tuple[float, float]],
        *,
        time_step_fs: float,
        name: str = "custom",
    ) -> "TemperatureProfile":
        """``(時刻 [ps], 温度 [K])`` の列から任意プロファイルを作る。

        時刻は単調増加である必要がある。区間ごとにステップ数へ換算される。
        """
        pts = [(float(t), float(T)) for t, T in points]
        if len(pts) < 2:
            raise ValueError("points は 2 点以上必要です。")
        times = [t for t, _ in pts]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("points の時刻は単調増加である必要があります。")
        segments = []
        for (t0, T0), (t1, T1) in zip(pts, pts[1:]):
            steps = int(round((t1 - t0) * 1e3 / time_step_fs))
            if steps <= 0:
                raise ValueError(
                    f"区間 {t0}–{t1} ps が time_step={time_step_fs} fs に対して短すぎます。"
                )
            segments.append(TemperatureSegment(T0, T1, steps))
        return cls(segments, name=name)

    @classmethod
    def from_steps(
        cls, schedule: Sequence[tuple[float, float, int]], *, name: str = "custom"
    ) -> "TemperatureProfile":
        """``(T_start, T_end, steps)`` のタプル列からプロファイルを作る。"""
        return cls([TemperatureSegment(*item) for item in schedule], name=name)

    # -------------------------------------------------------------- operations
    def __len__(self) -> int:
        return len(self.segments)

    def __iter__(self):
        return iter(self.segments)

    def __getitem__(self, index):
        return self.segments[index]

    def append(self, segment: TemperatureSegment) -> "TemperatureProfile":
        self.segments.append(segment)
        return self

    def then(self, T_end: float, steps: int, *, label: str = "") -> "TemperatureProfile":
        """直前の終端温度から ``T_end`` へ続く区間を足す (メソッドチェーン用)。"""
        T_start = self.segments[-1].T_end if self.segments else T_end
        return self.append(TemperatureSegment(T_start, T_end, steps, label))

    @property
    def total_steps(self) -> int:
        return sum(seg.steps for seg in self.segments)

    def total_time_ps(self, time_step_fs: float) -> float:
        return self.total_steps * time_step_fs * 1e-3

    def temperature_range(self) -> tuple[float, float]:
        temps = [T for seg in self.segments for T in (seg.T_start, seg.T_end)]
        return (min(temps), max(temps))

    def to_arrays(self, time_step_fs: float) -> tuple[np.ndarray, np.ndarray]:
        """プロット用に ``(時刻 [ps], 目標温度 [K])`` の折れ線を返す。"""
        times: list[float] = [0.0]
        temps: list[float] = [self.segments[0].T_start] if self.segments else []
        t = 0.0
        for seg in self.segments:
            t += seg.duration_ps(time_step_fs)
            times.append(t)
            temps.append(seg.T_end)
        return np.asarray(times), np.asarray(temps)

    def target_at_step(self, step: int) -> float:
        """任意のステップにおける目標温度を返す。"""
        elapsed = 0
        for seg in self.segments:
            if step <= elapsed + seg.steps:
                frac = (step - elapsed) / seg.steps
                return seg.T_start + frac * (seg.T_end - seg.T_start)
            elapsed += seg.steps
        return self.segments[-1].T_end if self.segments else float("nan")

    def summary(self, time_step_fs: float = 1.0) -> str:
        lines = [f"TemperatureProfile '{self.name}' ({len(self)} stages)"]
        t = 0.0
        for i, seg in enumerate(self.segments, 1):
            dt = seg.duration_ps(time_step_fs)
            lines.append(
                f"  {i:2d}. {seg.label:<28s} {seg.T_start:8.1f} K -> {seg.T_end:8.1f} K  "
                f"{seg.steps:>9,d} steps  {t:8.2f}–{t + dt:8.2f} ps  "
                f"({seg.rate_K_per_ps(time_step_fs):+.2f} K/ps)"
            )
            t += dt
        lines.append(f"  total: {self.total_steps:,d} steps = {t:.2f} ps")
        return "\n".join(lines)

    def plot(self, time_step_fs: float = 1.0, *, filename: Path | str | None = None):
        """目標温度プロファイルを描画する。"""
        import matplotlib

        if filename is not None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        times, temps = self.to_arrays(time_step_fs)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(times, temps, "-o", ms=4, color="tab:red")
        ax.set_xlabel("time (ps)")
        ax.set_ylabel("target temperature (K)")
        ax.set_title(self.name)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        if filename is not None:
            fig.savefig(filename, dpi=150)
            plt.close(fig)
        return fig
