"""予実管理 (予測値と実測値の突き合わせ)。

example を回すときに「事前に決めた期待値 (予)」と「実際に得られた値 (実)」を
1 つの表にまとめ、CSV / Markdown / 図として残す。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

__all__ = ["Expectation", "ValidationReport"]


@dataclass
class Expectation:
    """1 項目分の予実。

    Parameters
    ----------
    key
        項目の識別子。
    label
        表示名。
    expected
        期待値 (予)。``None`` なら参考値のみ。
    unit
        単位。
    abs_tol, rel_tol
        許容誤差。両方指定した場合はどちらかを満たせば合格。
    source
        期待値の出典 (参照計算・文献・理論値など)。
    informative
        ``True`` なら合否判定に含めない (参考記録)。
    note
        補足。
    actual
        実測値 (実)。:meth:`ValidationReport.record` で設定される。
    """

    key: str
    label: str
    expected: float | None = None
    unit: str = ""
    abs_tol: float | None = None
    rel_tol: float | None = None
    source: str = ""
    informative: bool = False
    note: str = ""
    actual: float | None = None

    # ------------------------------------------------------------------ 判定
    @property
    def measured(self) -> bool:
        return self.actual is not None and not (
            isinstance(self.actual, float) and math.isnan(self.actual)
        )

    @property
    def deviation(self) -> float | None:
        if self.expected is None or not self.measured:
            return None
        return float(self.actual) - float(self.expected)

    @property
    def relative_deviation(self) -> float | None:
        dev = self.deviation
        if dev is None or self.expected in (None, 0):
            return None
        return dev / abs(float(self.expected))

    @property
    def status(self) -> str:
        if self.informative:
            return "参考" if self.measured else "未測定"
        if self.expected is None:
            return "参考"
        if not self.measured:
            return "未測定"
        dev = abs(self.deviation or 0.0)
        ok = False
        if self.abs_tol is not None and dev <= self.abs_tol:
            ok = True
        if self.rel_tol is not None and self.expected != 0:
            if dev / abs(self.expected) <= self.rel_tol:
                ok = True
        if self.abs_tol is None and self.rel_tol is None:
            ok = dev == 0.0
        return "合格" if ok else "不一致"

    @property
    def tolerance_text(self) -> str:
        parts = []
        if self.abs_tol is not None:
            parts.append(f"±{self.abs_tol:g}")
        if self.rel_tol is not None:
            parts.append(f"±{self.rel_tol * 100:g}%")
        return " / ".join(parts) if parts else "-"

    def as_row(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "項目": self.label,
            "単位": self.unit,
            "予 (期待値)": self.expected,
            "実 (実測値)": self.actual,
            "差": self.deviation,
            "相対差": self.relative_deviation,
            "許容": self.tolerance_text,
            "判定": self.status,
            "出典": self.source,
            "備考": self.note,
        }


@dataclass
class ValidationReport:
    """予実表。

    Examples
    --------
    >>> report = ValidationReport("DPMD water", [
    ...     Expectation("T", "平均温度", expected=330, unit="K", abs_tol=10),
    ... ])
    >>> report.record("T", 328.4)
    >>> print(report.to_markdown())
    """

    title: str
    entries: list[Expectation] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ 操作
    def add(self, entry: Expectation) -> "ValidationReport":
        self.entries.append(entry)
        return self

    def extend(self, entries: Iterable[Expectation]) -> "ValidationReport":
        for entry in entries:
            self.add(entry)
        return self

    def __getitem__(self, key: str) -> Expectation:
        for entry in self.entries:
            if entry.key == key:
                return entry
        raise KeyError(key)

    def record(self, key: str, value: float | None) -> "ValidationReport":
        """実測値を記録する。"""
        self[key].actual = None if value is None else float(value)
        return self

    def record_many(self, values: dict[str, float | None]) -> "ValidationReport":
        for key, value in values.items():
            if any(entry.key == key for entry in self.entries):
                self.record(key, value)
        return self

    # ------------------------------------------------------------------ 集計
    @property
    def judged(self) -> list[Expectation]:
        """合否判定の対象となる項目。"""
        return [e for e in self.entries if not e.informative and e.expected is not None]

    @property
    def passed(self) -> list[Expectation]:
        return [e for e in self.judged if e.status == "合格"]

    @property
    def failed(self) -> list[Expectation]:
        return [e for e in self.judged if e.status == "不一致"]

    @property
    def all_passed(self) -> bool:
        return len(self.judged) > 0 and not self.failed

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame([entry.as_row() for entry in self.entries])

    # ------------------------------------------------------------------ 出力
    def to_markdown(self) -> str:
        lines = [f"# 予実表: {self.title}", ""]
        if self.context:
            lines.append("## 実行条件")
            lines.append("")
            for key, value in self.context.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")
        lines.append("## 結果")
        lines.append("")
        header = ["項目", "単位", "予 (期待値)", "実 (実測値)", "差", "相対差", "許容", "判定", "出典"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for entry in self.entries:
            rel = entry.relative_deviation
            lines.append(
                "| "
                + " | ".join(
                    [
                        entry.label,
                        entry.unit or "-",
                        _fmt(entry.expected),
                        _fmt(entry.actual),
                        _fmt(entry.deviation),
                        f"{rel * 100:+.2f}%" if rel is not None else "-",
                        entry.tolerance_text,
                        entry.status,
                        entry.source or "-",
                    ]
                )
                + " |"
            )
        lines.append("")
        notes = [(e.label, e.note) for e in self.entries if e.note]
        if notes:
            lines.append("## 備考")
            lines.append("")
            for label, note in notes:
                lines.append(f"- **{label}**: {note}")
            lines.append("")
        lines.append("## 判定サマリ")
        lines.append("")
        lines.append(
            f"- 判定対象 {len(self.judged)} 項目中 **{len(self.passed)} 合格 / "
            f"{len(self.failed)} 不一致**"
        )
        if self.failed:
            for entry in self.failed:
                lines.append(f"  - 不一致: {entry.label} (予 {_fmt(entry.expected)} / 実 {_fmt(entry.actual)})")
        lines.append(
            f"- 参考記録 {len([e for e in self.entries if e.informative])} 項目"
        )
        return "\n".join(lines) + "\n"

    def write(self, directory: Path | str, *, basename: str = "validation") -> dict[str, Path]:
        """CSV と Markdown を書き出す。"""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        csv_path = directory / f"{basename}.csv"
        md_path = directory / f"{basename}.md"
        self.to_dataframe().to_csv(csv_path, index=False)
        md_path.write_text(self.to_markdown())
        return {"csv": csv_path, "markdown": md_path}

    def plot(self, filename: Path | str, *, dpi: int = 150) -> Path:
        """予実の相対差を横棒グラフで表示する。"""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        items = [e for e in self.entries if e.relative_deviation is not None]
        if not items:
            raise ValueError("比較可能な項目がありません。")
        # matplotlib の既定フォントは日本語グリフを持たないので key (ASCII) を使う
        labels = [e.key for e in items]
        values = [e.relative_deviation * 100 for e in items]
        colors = [
            "tab:green" if e.status == "合格" else ("tab:gray" if e.informative else "tab:red")
            for e in items
        ]
        fig, ax = plt.subplots(figsize=(9, 0.55 * len(items) + 2.2))
        bars = ax.barh(labels, values, color=colors)
        ax.axvline(0, color="k", lw=1)
        # エネルギー原点の違いなどで桁違いの項目が混ざるため対数(symlog)にする
        if max(abs(v) for v in values) > 200:
            ax.set_xscale("symlog", linthresh=10)
        for bar, value in zip(bars, values):
            offset = 3 if value >= 0 else -3
            ax.annotate(
                f"{value:+.2f}%",
                (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                textcoords="offset points",
                xytext=(offset, 0),
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=8,
            )
        ax.margins(x=0.25)
        ax.set_xlabel("relative deviation (actual - expected) / |expected|  [%]")
        ax.set_title(f"expected vs. actual: {self.title}")
        ax.grid(alpha=0.3, axis="x")
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=c)
            for c in ("tab:green", "tab:red", "tab:gray")
        ]
        ax.legend(handles, ["pass", "mismatch", "informative"], fontsize=8, loc="lower left")
        fig.tight_layout()
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        return path

    def __str__(self) -> str:
        return self.to_markdown()


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "-"
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
            return f"{value:.4g}"
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)
