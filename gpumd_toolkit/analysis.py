"""GPUMD 出力の解析と可視化。

``thermo.out`` を ``run.in`` と突き合わせて時間軸を復元し、
温度・エネルギー・圧力・体積の時系列を matplotlib で保存する。
``msd.out`` / ``sdc.out`` / ``rdf.out`` などの補助出力にも対応する。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["ThermoData", "MDAnalyzer", "RunSpec", "StageSpec"]

#: thermo.out の列名 (GPUMD のドキュメント順)
THERMO_COLUMNS = [
    "temperature",
    "kinetic_energy",
    "potential_energy",
    "Pxx",
    "Pyy",
    "Pzz",
    "Pyz",
    "Pxz",
    "Pxy",
    "ax",
    "ay",
    "az",
    "bx",
    "by",
    "bz",
    "cx",
    "cy",
    "cz",
]


@dataclass
class StageSpec:
    """``run.in`` から復元した 1 ステージの情報。"""

    index: int
    ensemble: str
    steps: int
    time_step: float
    thermo_interval: int | None
    T_start: float | None
    T_end: float | None
    label: str

    @property
    def n_thermo_rows(self) -> int:
        if not self.thermo_interval:
            return 0
        return self.steps // self.thermo_interval


@dataclass
class RunSpec:
    """``run.in`` のパース結果。"""

    path: Path
    potential: str
    time_step: float
    stages: list[StageSpec] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: Path | str) -> "RunSpec":
        path = Path(path)
        time_step = 1.0
        potential = ""
        thermo_interval: int | None = None
        stages: list[StageSpec] = []
        pending: dict | None = None

        for raw in path.read_text().splitlines():
            line = raw.split("#")[0].strip()
            if not line:
                continue
            tokens = line.split()
            key = tokens[0]
            if key == "potential":
                potential = " ".join(tokens[1:])
            elif key == "time_step":
                time_step = float(tokens[1])
            elif key == "dump_thermo":
                thermo_interval = int(tokens[1])
            elif key == "ensemble":
                pending = _parse_ensemble(tokens[1:])
            elif key == "run":
                steps = int(tokens[1])
                info = pending or {"ensemble": "unknown", "T_start": None, "T_end": None}
                stages.append(
                    StageSpec(
                        index=len(stages),
                        ensemble=info["ensemble"],
                        steps=steps,
                        time_step=time_step,
                        thermo_interval=thermo_interval,
                        T_start=info["T_start"],
                        T_end=info["T_end"],
                        label=f"{len(stages) + 1}:{info['ensemble']}",
                    )
                )
                # dump 系は propagating ではないので run ごとにリセットされる
                thermo_interval = None
        return cls(path=path, potential=potential, time_step=time_step, stages=stages)

    @property
    def total_steps(self) -> int:
        return sum(s.steps for s in self.stages)


def _parse_ensemble(tokens: Sequence[str]) -> dict:
    name = tokens[0]
    T_start = T_end = None
    if name == "nve":
        pass
    elif name.endswith("_mttk"):
        if "temp" in tokens:
            i = tokens.index("temp")
            T_start, T_end = float(tokens[i + 1]), float(tokens[i + 2])
    elif len(tokens) >= 3:
        try:
            T_start, T_end = float(tokens[1]), float(tokens[2])
        except ValueError:
            pass
    return {"ensemble": name, "T_start": T_start, "T_end": T_end}


@dataclass
class ThermoData:
    """``thermo.out`` を時間軸つきの DataFrame に変換したもの。"""

    frame: pd.DataFrame
    n_atoms: int
    run_spec: RunSpec | None = None

    # ------------------------------------------------------------------ 読み込み
    @classmethod
    def from_directory(
        cls,
        directory: Path | str,
        *,
        n_atoms: int | None = None,
        thermo_file: str = "thermo.out",
    ) -> "ThermoData":
        directory = Path(directory)
        thermo_path = directory / thermo_file
        if not thermo_path.is_file():
            raise FileNotFoundError(f"{thermo_path} がありません。")

        run_spec = None
        run_in = directory / "run.in"
        if run_in.is_file():
            run_spec = RunSpec.from_file(run_in)

        if n_atoms is None:
            n_atoms = _n_atoms_from_header(thermo_path)
        if n_atoms is None:
            model = directory / "model.xyz"
            if model.is_file():
                n_atoms = int(model.read_text().split("\n", 1)[0].strip())
        if n_atoms is None:
            n_atoms = 1

        frame = _read_thermo_table(thermo_path)
        frame = _augment(frame, n_atoms=n_atoms, run_spec=run_spec)
        return cls(frame=frame, n_atoms=n_atoms, run_spec=run_spec)

    # ------------------------------------------------------------------ 便利関数
    def __len__(self) -> int:
        return len(self.frame)

    @property
    def time(self) -> np.ndarray:
        return self.frame["time_ps"].to_numpy()

    def equilibrated(self, drop_fraction: float = 0.3) -> pd.DataFrame:
        """最初の ``drop_fraction`` を捨てた (平衡化後の) データを返す。"""
        if not 0.0 <= drop_fraction < 1.0:
            raise ValueError("drop_fraction は [0, 1) の範囲です。")
        start = int(len(self.frame) * drop_fraction)
        return self.frame.iloc[start:]

    def stage(self, index: int) -> pd.DataFrame:
        """ステージ番号 (0 始まり) でデータを取り出す。"""
        return self.frame[self.frame["stage"] == index]

    def summary(self, drop_fraction: float = 0.3) -> pd.DataFrame:
        """ステージごとの平均・標準偏差をまとめる。

        ``drop_fraction`` は各ステージの先頭を平衡化とみなして除外する割合。
        """
        quantities = [
            "temperature",
            "potential_energy_per_atom",
            "total_energy_per_atom",
            "pressure",
            "volume",
            "density",
        ]
        rows = []
        for stage_id, group in self.frame.groupby("stage", sort=True):
            cut = group.iloc[int(len(group) * drop_fraction) :]
            if len(cut) == 0:
                cut = group
            row: dict = {
                "stage": int(stage_id),
                "label": group["label"].iloc[0],
                "ensemble": group["ensemble"].iloc[0],
                "n_samples": len(cut),
                "time_ps": float(group["time_ps"].iloc[-1] - group["time_ps"].iloc[0]),
            }
            for q in quantities:
                if q in cut:
                    row[f"{q}_mean"] = float(cut[q].mean())
                    row[f"{q}_std"] = float(cut[q].std(ddof=1)) if len(cut) > 1 else 0.0
            rows.append(row)
        return pd.DataFrame(rows)

    def to_csv(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_csv(path, index=False)
        return path

    def energy_drift(self) -> float:
        """全エネルギーのドリフト [meV/atom/ps] (NVE の質を見る指標)。"""
        t = self.frame["time_ps"].to_numpy()
        e = self.frame["total_energy_per_atom"].to_numpy()
        if len(t) < 2 or t[-1] == t[0]:
            return float("nan")
        slope = np.polyfit(t, e, 1)[0]
        return float(slope * 1e3)


def _n_atoms_from_header(path: Path) -> int | None:
    with open(path) as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            match = re.search(r"num_atoms\s+(\d+)", line)
            if match:
                return int(match.group(1))
    return None


def _read_thermo_table(path: Path) -> pd.DataFrame:
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] < len(THERMO_COLUMNS):
        raise ValueError(
            f"{path} の列数が想定 ({len(THERMO_COLUMNS)}) より少ないです: {data.shape[1]}"
        )
    return pd.DataFrame(data[:, : len(THERMO_COLUMNS)], columns=THERMO_COLUMNS)


def _augment(frame: pd.DataFrame, *, n_atoms: int, run_spec: RunSpec | None) -> pd.DataFrame:
    frame = frame.copy()

    a = frame[["ax", "ay", "az"]].to_numpy()
    b = frame[["bx", "by", "bz"]].to_numpy()
    c = frame[["cx", "cy", "cz"]].to_numpy()
    volume = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c)))

    frame["n_atoms"] = n_atoms
    frame["total_energy"] = frame["kinetic_energy"] + frame["potential_energy"]
    frame["potential_energy_per_atom"] = frame["potential_energy"] / n_atoms
    frame["kinetic_energy_per_atom"] = frame["kinetic_energy"] / n_atoms
    frame["total_energy_per_atom"] = frame["total_energy"] / n_atoms
    frame["pressure"] = (frame["Pxx"] + frame["Pyy"] + frame["Pzz"]) / 3.0
    frame["volume"] = volume
    frame["volume_per_atom"] = volume / n_atoms
    frame["a"] = np.linalg.norm(a, axis=1)
    frame["b"] = np.linalg.norm(b, axis=1)
    frame["c"] = np.linalg.norm(c, axis=1)

    # 時間軸・ステージ・目標温度を run.in から復元する
    times = np.zeros(len(frame))
    steps = np.zeros(len(frame), dtype=int)
    stage_ids = np.zeros(len(frame), dtype=int)
    labels = np.empty(len(frame), dtype=object)
    ensembles = np.empty(len(frame), dtype=object)
    target = np.full(len(frame), np.nan)

    if run_spec is not None and run_spec.stages:
        row = 0
        t0 = 0.0
        step0 = 0
        for stage in run_spec.stages:
            n_rows = stage.n_thermo_rows
            if n_rows == 0:
                t0 += stage.steps * stage.time_step * 1e-3
                step0 += stage.steps
                continue
            end = min(row + n_rows, len(frame))
            k = np.arange(1, end - row + 1)
            local_steps = k * (stage.thermo_interval or 1)
            times[row:end] = t0 + local_steps * stage.time_step * 1e-3
            steps[row:end] = step0 + local_steps
            stage_ids[row:end] = stage.index
            labels[row:end] = stage.label
            ensembles[row:end] = stage.ensemble
            if stage.T_start is not None and stage.T_end is not None:
                frac = local_steps / stage.steps
                target[row:end] = stage.T_start + frac * (stage.T_end - stage.T_start)
            row = end
            t0 += stage.steps * stage.time_step * 1e-3
            step0 += stage.steps
            if row >= len(frame):
                break
        if row < len(frame):  # run.in と行数が合わない場合は末尾を外挿
            dt = times[row - 1] - times[row - 2] if row >= 2 else 1.0
            times[row:] = times[row - 1] + dt * np.arange(1, len(frame) - row + 1)
            stage_ids[row:] = stage_ids[row - 1]
            labels[row:] = labels[row - 1]
            ensembles[row:] = ensembles[row - 1]
    else:  # run.in が無い場合は行番号を時間軸の代わりにする
        times = np.arange(1, len(frame) + 1, dtype=float)
        steps = np.arange(1, len(frame) + 1)
        labels[:] = "all"
        ensembles[:] = "unknown"

    frame["time_ps"] = times
    frame["step"] = steps
    frame["stage"] = stage_ids
    frame["label"] = labels
    frame["ensemble"] = ensembles
    frame["target_temperature"] = target
    return frame


class MDAnalyzer:
    """1 つの GPUMD 計算ディレクトリを解析する。

    Examples
    --------
    >>> analyzer = MDAnalyzer("runs/si_nvt")
    >>> print(analyzer.thermo.summary())
    >>> analyzer.plot_all()          # runs/si_nvt/analysis/*.png
    """

    def __init__(
        self,
        directory: Path | str,
        *,
        n_atoms: int | None = None,
        output_dir: Path | str | None = None,
    ) -> None:
        self.directory = Path(directory).expanduser().resolve()
        if not self.directory.is_dir():
            raise NotADirectoryError(f"{self.directory} はディレクトリではありません。")
        self.thermo = ThermoData.from_directory(self.directory, n_atoms=n_atoms)
        self.output_dir = Path(output_dir) if output_dir else self.directory / "analysis"

    # ------------------------------------------------------------------ 補助出力
    def read_table(self, filename: str, columns: Sequence[str] | None = None) -> pd.DataFrame:
        """GPUMD の数値出力ファイルを DataFrame として読む。"""
        path = self.directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"{path} がありません。")
        data = np.loadtxt(path, comments="#", ndmin=2)
        if columns is None:
            columns = [f"col{i}" for i in range(data.shape[1])]
        return pd.DataFrame(data[:, : len(columns)], columns=list(columns))

    def read_msd(self) -> pd.DataFrame:
        """``msd.out``: t [ps], MSD x/y/z [Å²], SDC x/y/z [Å²/ps]。"""
        return self.read_table(
            "msd.out", ["time_ps", "msd_x", "msd_y", "msd_z", "sdc_x", "sdc_y", "sdc_z"]
        )

    def read_sdc(self) -> pd.DataFrame:
        """``sdc.out``: t [ps], VAC x/y/z [Å²/ps²], SDC x/y/z [Å²/ps]。"""
        return self.read_table(
            "sdc.out", ["time_ps", "vac_x", "vac_y", "vac_z", "sdc_x", "sdc_y", "sdc_z"]
        )

    def read_rdf(self) -> pd.DataFrame:
        data = np.loadtxt(self.directory / "rdf.out", comments="#", ndmin=2)
        columns = ["r"] + [f"g{i}" for i in range(data.shape[1] - 1)]
        return pd.DataFrame(data, columns=columns)

    def diffusion_coefficient(self, *, source: str = "msd") -> dict[str, float]:
        """自己拡散係数を返す [Å²/ps] と [1e-9 m²/s]。

        ``source='msd'`` なら ``msd.out``、``'sdc'`` なら ``sdc.out`` の
        最終時刻における SDC を用いる (3 方向の平均)。
        """
        frame = self.read_msd() if source == "msd" else self.read_sdc()
        last = frame.iloc[-1]
        d_components = np.array([last["sdc_x"], last["sdc_y"], last["sdc_z"]])
        d_mean = float(d_components.mean())
        return {
            "D_x": float(d_components[0]),
            "D_y": float(d_components[1]),
            "D_z": float(d_components[2]),
            "D_mean_A2_per_ps": d_mean,
            # 1 Å²/ps = 1e-8 m²/s = 10 * 1e-9 m²/s
            "D_mean_1e-9_m2_per_s": d_mean * 10.0,
            "correlation_time_ps": float(last["time_ps"]),
        }

    # ------------------------------------------------------------------ 描画
    def plot_all(
        self,
        *,
        output_dir: Path | str | None = None,
        dpi: int = 150,
        drop_fraction: float = 0.3,
    ) -> list[Path]:
        """主要な時系列をまとめて PNG に保存し、書き出したパスを返す。"""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out = Path(output_dir) if output_dir else self.output_dir
        out.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        written.append(self._plot_overview(out, dpi=dpi))
        written.append(self._plot_temperature(out, dpi=dpi))
        written.append(self._plot_energy(out, dpi=dpi))
        written.append(self._plot_pressure(out, dpi=dpi))
        if self.thermo.frame["volume"].std() > 1e-6:
            written.append(self._plot_cell(out, dpi=dpi))
        for name, reader, plotter in (
            ("msd", self.read_msd, self._plot_msd),
            ("sdc", self.read_sdc, self._plot_sdc),
            ("rdf", self.read_rdf, self._plot_rdf),
        ):
            if (self.directory / f"{name}.out").is_file():
                try:
                    written.append(plotter(reader(), out, dpi=dpi))
                except Exception:  # 解析補助なので失敗しても本体は止めない
                    pass

        self.thermo.to_csv(out / "thermo.csv")
        written.append(out / "thermo.csv")
        summary = self.thermo.summary(drop_fraction=drop_fraction)
        summary.to_csv(out / "summary.csv", index=False)
        written.append(out / "summary.csv")
        return [p for p in written if p is not None]

    # ------------------------------------------------------------- 個別プロット
    def _stage_marks(self, ax) -> None:
        frame = self.thermo.frame
        boundaries = frame["time_ps"][frame["stage"].diff().fillna(0) != 0]
        for t in boundaries:
            ax.axvline(t, color="0.7", lw=0.8, ls="--", zorder=0)

    def _plot_overview(self, out: Path, dpi: int) -> Path:
        import matplotlib.pyplot as plt

        frame = self.thermo.frame
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        specs = [
            (axes[0, 0], "temperature", "temperature (K)", "tab:red"),
            (axes[0, 1], "total_energy_per_atom", "E$_{tot}$ (eV/atom)", "tab:blue"),
            (axes[1, 0], "pressure", "pressure (GPa)", "tab:green"),
            (axes[1, 1], "volume_per_atom", "volume (Å$^3$/atom)", "tab:purple"),
        ]
        for ax, column, ylabel, color in specs:
            ax.plot(frame["time_ps"], frame[column], lw=0.8, color=color)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            self._stage_marks(ax)
        if frame["target_temperature"].notna().any():
            axes[0, 0].plot(
                frame["time_ps"],
                frame["target_temperature"],
                lw=1.4,
                ls="--",
                color="k",
                label="target",
            )
            axes[0, 0].legend(fontsize=8)
        for ax in axes[1]:
            ax.set_xlabel("time (ps)")
        fig.suptitle(f"GPUMD: {self.directory.name}  (N = {self.thermo.n_atoms})")
        fig.tight_layout()
        path = out / "overview.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        return path

    def _plot_temperature(self, out: Path, dpi: int) -> Path:
        import matplotlib.pyplot as plt

        frame = self.thermo.frame
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.plot(frame["time_ps"], frame["temperature"], lw=0.8, color="tab:red", label="instant")
        window = max(1, len(frame) // 100)
        ax.plot(
            frame["time_ps"],
            frame["temperature"].rolling(window, center=True, min_periods=1).mean(),
            lw=1.6,
            color="darkred",
            label=f"rolling mean ({window})",
        )
        if frame["target_temperature"].notna().any():
            ax.plot(
                frame["time_ps"], frame["target_temperature"], lw=1.4, ls="--", color="k",
                label="target",
            )
        self._stage_marks(ax)
        ax.set_xlabel("time (ps)")
        ax.set_ylabel("temperature (K)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = out / "temperature.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        return path

    def _plot_energy(self, out: Path, dpi: int) -> Path:
        import matplotlib.pyplot as plt

        frame = self.thermo.frame
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axes[0].plot(frame["time_ps"], frame["potential_energy_per_atom"], lw=0.8,
                     color="tab:blue", label="potential")
        axes[0].set_ylabel("U (eV/atom)")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)
        axes[1].plot(frame["time_ps"], frame["total_energy_per_atom"], lw=0.8,
                     color="tab:orange", label="total (K+U)")
        axes[1].set_ylabel("E$_{tot}$ (eV/atom)")
        axes[1].set_xlabel("time (ps)")
        drift = self.thermo.energy_drift()
        axes[1].set_title(f"energy drift = {drift:+.4f} meV/atom/ps", fontsize=9)
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
        for ax in axes:
            self._stage_marks(ax)
        fig.tight_layout()
        path = out / "energy.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        return path

    def _plot_pressure(self, out: Path, dpi: int) -> Path:
        import matplotlib.pyplot as plt

        frame = self.thermo.frame
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for column, color in (("Pxx", "tab:blue"), ("Pyy", "tab:green"), ("Pzz", "tab:red")):
            ax.plot(frame["time_ps"], frame[column], lw=0.6, alpha=0.6, label=column)
        ax.plot(frame["time_ps"], frame["pressure"], lw=1.5, color="k", label="hydrostatic")
        self._stage_marks(ax)
        ax.set_xlabel("time (ps)")
        ax.set_ylabel("pressure (GPa)")
        ax.legend(fontsize=8, ncol=4)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = out / "pressure.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        return path

    def _plot_cell(self, out: Path, dpi: int) -> Path:
        import matplotlib.pyplot as plt

        frame = self.thermo.frame
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        for column, color in (("a", "tab:blue"), ("b", "tab:green"), ("c", "tab:red")):
            axes[0].plot(frame["time_ps"], frame[column], lw=1.0, color=color, label=column)
        axes[0].set_ylabel("cell length (Å)")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)
        axes[1].plot(frame["time_ps"], frame["volume_per_atom"], lw=1.0, color="tab:purple")
        axes[1].set_ylabel("volume (Å$^3$/atom)")
        axes[1].set_xlabel("time (ps)")
        axes[1].grid(alpha=0.3)
        for ax in axes:
            self._stage_marks(ax)
        fig.tight_layout()
        path = out / "cell.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        return path

    def _plot_msd(self, frame: pd.DataFrame, out: Path, dpi: int) -> Path:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        total = frame[["msd_x", "msd_y", "msd_z"]].sum(axis=1)
        for column in ("msd_x", "msd_y", "msd_z"):
            axes[0].plot(frame["time_ps"], frame[column], lw=1.0, label=column)
        axes[0].plot(frame["time_ps"], total, lw=1.6, color="k", label="total")
        axes[0].set_xlabel("correlation time (ps)")
        axes[0].set_ylabel("MSD (Å$^2$)")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)
        for column in ("sdc_x", "sdc_y", "sdc_z"):
            axes[1].plot(frame["time_ps"], frame[column], lw=1.0, label=column)
        axes[1].set_xlabel("correlation time (ps)")
        axes[1].set_ylabel("SDC (Å$^2$/ps)")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        path = out / "msd.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        return path

    def _plot_sdc(self, frame: pd.DataFrame, out: Path, dpi: int) -> Path:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for column in ("vac_x", "vac_y", "vac_z"):
            axes[0].plot(frame["time_ps"], frame[column], lw=1.0, label=column)
        axes[0].axhline(0, color="0.6", lw=0.8)
        axes[0].set_xlabel("correlation time (ps)")
        axes[0].set_ylabel("VAC (Å$^2$/ps$^2$)")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)
        for column in ("sdc_x", "sdc_y", "sdc_z"):
            axes[1].plot(frame["time_ps"], frame[column], lw=1.0, label=column)
        axes[1].set_xlabel("correlation time (ps)")
        axes[1].set_ylabel("SDC (Å$^2$/ps)")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        path = out / "sdc.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        return path

    def _plot_rdf(self, frame: pd.DataFrame, out: Path, dpi: int) -> Path:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        for column in frame.columns[1:]:
            ax.plot(frame["r"], frame[column], lw=1.0, label=column)
        ax.set_xlabel("r (Å)")
        ax.set_ylabel("g(r)")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = out / "rdf.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        return path


def compare_runs(
    directories: Sequence[Path | str],
    *,
    quantity: str = "temperature",
    labels: Sequence[str] | None = None,
    filename: Path | str | None = None,
    dpi: int = 150,
):
    """複数の計算ディレクトリの時系列を 1 枚に重ねて描く (並列実行の比較用)。"""
    import matplotlib

    if filename is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, directory in enumerate(directories):
        data = ThermoData.from_directory(directory)
        label = labels[i] if labels else Path(directory).name
        ax.plot(data.frame["time_ps"], data.frame[quantity], lw=0.9, label=label)
    ax.set_xlabel("time (ps)")
    ax.set_ylabel(quantity)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename, dpi=dpi)
        plt.close(fig)
    return fig
