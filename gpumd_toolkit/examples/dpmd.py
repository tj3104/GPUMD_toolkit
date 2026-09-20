"""DPMD (Deep Potential MD) の example。

題材は GPUMD 公式 example ``14_DP/water_msd`` と gpyumd のドキュメント例
(https://gpyumd.readthedocs.io/en/latest/example.html) に倣った
「液体水 (512 H2O = 1536 原子) を 330 K で NVT 平衡化し、
速度自己相関 (``compute_sdc``) と平均二乗変位 (``compute_msd``) から
自己拡散係数を求める」というワークフロー。

予実 (expected vs. actual)
--------------------------
公式 example には DP ポテンシャル (DNN_seed2.pb) で実行済みの
``thermo.out`` / ``sdc.out`` が同梱されている。これを「予 (期待値)」とし、
本クラスの実行結果を「実 (実測値)」として :mod:`gpumd_toolkit.validation`
の予実表にまとめる。

DP が使えない環境について
-------------------------
``potential dp.txt <model>.pb`` は GPUMD を ``-DUSE_TENSORFLOW`` 付きで
(``make -f makefile.dp``) ビルドし、かつ deepmd-kit の ``.pb`` モデルが
必要である。どちらかが欠けている場合、本クラスは自動的に NEP
(例: 汎用の NEP89) へフォールバックして同じワークフローを実行する。
その場合、ポテンシャル依存量 (エネルギー・圧力・拡散係数) は
「参考」扱いとなり、ポテンシャルに依存しない量 (原子数・セル・温度制御)
のみが合否判定の対象になる。
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import GPUMDEnvironment
from ..inputs import DumpSettings, MDStage
from ..md import GPUMDCalculation, GPUMDResult
from ..structure import StructureHandler
from ..validation import Expectation, ValidationReport

__all__ = ["DPMDExample", "DPMDReference", "DEFAULT_REFERENCE_DIR"]

#: GPUMD 同梱 example のディレクトリ (存在すれば参照計算として使う)
DEFAULT_REFERENCE_DIR = Path("/home/tajimamainpc/08_gpumd/01_example/14_DP/water_msd")


@dataclass
class DPMDReference:
    """参照計算 (DP で実行済みの example 出力) から読み取った期待値。"""

    directory: Path | None
    n_atoms: int = 1536
    box_length: float = 25.4381160736
    target_temperature: float = 330.0
    time_step_fs: float = 0.5
    mean_temperature: float | None = None
    std_temperature: float | None = None
    potential_energy_per_atom: float | None = None
    pressure: float | None = None
    diffusion_sdc: float | None = None  # 1e-9 m^2/s
    diffusion_components: tuple[float, float, float] | None = None

    @classmethod
    def from_directory(cls, directory: Path | str | None) -> "DPMDReference":
        """example ディレクトリの ``thermo.out`` / ``sdc.out`` を読む。"""
        if directory is None:
            return cls(directory=None)
        directory = Path(directory)
        if not directory.is_dir():
            return cls(directory=None)

        reference = cls(directory=directory)
        model = directory / "model.xyz"
        if model.is_file():
            atoms = StructureHandler.read(model, format="extxyz")
            reference.n_atoms = len(atoms)
            reference.box_length = float(atoms.cell.cellpar()[0])

        thermo = directory / "thermo.out"
        if thermo.is_file():
            data = np.loadtxt(thermo, comments="#", ndmin=2)
            reference.mean_temperature = float(data[:, 0].mean())
            reference.std_temperature = float(data[:, 0].std(ddof=1))
            reference.potential_energy_per_atom = float(data[:, 2].mean() / reference.n_atoms)
            reference.pressure = float(data[:, 3:6].mean())

        sdc = directory / "sdc.out"
        if sdc.is_file():
            data = np.loadtxt(sdc, comments="#", ndmin=2)
            components = data[-1, 4:7]
            reference.diffusion_components = tuple(float(c * 10.0) for c in components)
            reference.diffusion_sdc = float(components.mean() * 10.0)

        run_in = directory / "run.in"
        if run_in.is_file():
            for line in run_in.read_text().splitlines():
                tokens = line.split()
                if tokens[:1] == ["time_step"]:
                    reference.time_step_fs = float(tokens[1])
                if tokens[:1] == ["ensemble"] and len(tokens) >= 3:
                    try:
                        reference.target_temperature = float(tokens[2])
                    except ValueError:
                        pass
        return reference

    def as_context(self) -> dict[str, Any]:
        return {
            "参照ディレクトリ": str(self.directory) if self.directory else "なし",
            "参照系": f"{self.n_atoms} 原子 (512 H2O), 立方セル {self.box_length:.4f} Å",
            "参照条件": f"NVT (nvt_ber) {self.target_temperature:g} K, "
            f"time_step {self.time_step_fs:g} fs",
        }


class DPMDExample:
    """DPMD の example をひととおり実行して予実をとるクラス。

    Parameters
    ----------
    workdir
        計算ディレクトリ。
    dp_setting_file
        ``dp 2 O H`` のような GPUMD 用 DP 設定ファイル。
        ``None`` なら参照 example の ``dp.txt`` を使う (あれば)。
    dp_model
        deepmd-kit の ``.pb`` モデル。これが無い場合は NEP へフォールバックする。
    fallback_potential
        DP が使えないときに使う NEP ポテンシャル (``nep.txt``)。
    reference_dir
        参照計算 (DP 実行済み出力) のディレクトリ。
    structure
        初期構造。``None`` なら参照 example の ``model.xyz`` を使う。
    equilibration_steps, production_steps
        平衡化・本計算のステップ数。
    time_step
        時間刻み [fs]。
    temperature
        目標温度 [K]。
    sdc_sample_interval, sdc_correlation_steps
        ``compute_sdc <sample_interval> <Nc>``。
    msd_sample_interval, msd_correlation_steps
        ``compute_msd <sample_interval> <Nc>``。
    traj_interval
        トラジェクトリ出力間隔 (ステップ)。

    Examples
    --------
    >>> example = DPMDExample("runs/dpmd_water",
    ...                       fallback_potential="nep89.txt")
    >>> report = example.execute()
    >>> print(report.to_markdown())
    """

    def __init__(
        self,
        workdir: Path | str,
        *,
        dp_setting_file: Path | str | None = None,
        dp_model: Path | str | None = None,
        fallback_potential: Path | str | None = None,
        reference_dir: Path | str | None = DEFAULT_REFERENCE_DIR,
        structure: Path | str | None = None,
        equilibration_steps: int = 4_000,
        production_steps: int = 10_000,
        time_step: float = 0.5,
        temperature: float = 330.0,
        thermostat: str = "nvt_ber",
        T_coup: float = 100.0,
        sdc_sample_interval: int = 2,
        sdc_correlation_steps: int = 2_000,
        msd_sample_interval: int = 10,
        msd_correlation_steps: int = 400,
        thermo_interval: int = 10,
        traj_interval: int = 500,
        environment: GPUMDEnvironment | None = None,
        max_atoms: int = 10_000,
        overwrite: bool = False,
    ) -> None:
        self.workdir = Path(workdir).expanduser().resolve()
        self.environment = environment or GPUMDEnvironment()
        self.reference = DPMDReference.from_directory(reference_dir)

        self.dp_setting_file = _existing(dp_setting_file) or _existing(
            (self.reference.directory / "dp.txt") if self.reference.directory else None
        )
        self.dp_model = _existing(dp_model)
        self.fallback_potential = _existing(fallback_potential)

        if structure is not None:
            self.structure_path = Path(structure).expanduser().resolve()
        elif self.reference.directory and (self.reference.directory / "model.xyz").is_file():
            self.structure_path = self.reference.directory / "model.xyz"
        else:
            raise FileNotFoundError(
                "初期構造が見つかりません。structure= で指定してください。"
            )

        self.equilibration_steps = int(equilibration_steps)
        self.production_steps = int(production_steps)
        self.time_step = float(time_step)
        self.temperature = float(temperature)
        self.thermostat = thermostat
        self.T_coup = float(T_coup)
        self.sdc_sample_interval = int(sdc_sample_interval)
        self.sdc_correlation_steps = int(sdc_correlation_steps)
        self.msd_sample_interval = int(msd_sample_interval)
        self.msd_correlation_steps = int(msd_correlation_steps)
        self.thermo_interval = int(thermo_interval)
        self.traj_interval = int(traj_interval)
        self.max_atoms = int(max_atoms)
        self.overwrite = overwrite

        self.calculation: GPUMDCalculation | None = None
        self.result: GPUMDResult | None = None
        self.report: ValidationReport | None = None

    # ------------------------------------------------------------------ 準備
    @property
    def dp_available(self) -> bool:
        """DP ポテンシャルで実行できるか (設定ファイルとモデルが揃っているか)。"""
        return self.dp_setting_file is not None and self.dp_model is not None

    def check_dp_support(self) -> dict[str, Any]:
        """GPUMD 実行ファイルが DP 対応ビルドかどうかを調べる。

        GPUMD は ``-DUSE_TENSORFLOW`` を付けてビルドしたときだけ DP を扱える。
        バイナリ中に deepmd のシンボルがあるかで判定する (簡易チェック)。
        """
        executable = self.environment.which(self.environment.gpumd_command)
        info: dict[str, Any] = {
            "executable": executable,
            "dp_setting_file": str(self.dp_setting_file) if self.dp_setting_file else None,
            "dp_model": str(self.dp_model) if self.dp_model else None,
            "built_with_dp": None,
        }
        if executable and Path(executable).is_file():
            try:
                blob = Path(executable).read_bytes()
                info["built_with_dp"] = b"DeepPot" in blob or b"deepmd" in blob
            except OSError:
                info["built_with_dp"] = None
        return info

    def potential(self) -> list[str] | str:
        """実際に使うポテンシャル指定を返す。"""
        if self.dp_available:
            return [str(self.dp_setting_file), str(self.dp_model)]
        if self.fallback_potential is None:
            raise FileNotFoundError(
                "DP モデル (.pb) が無く、fallback_potential (NEP) も指定されていません。\n"
                "  dp_model= に deepmd-kit の .pb を渡すか、\n"
                "  fallback_potential= に nep.txt を渡してください。"
            )
        return str(self.fallback_potential)

    @property
    def using_dp(self) -> bool:
        return self.dp_available

    def build(self) -> GPUMDCalculation:
        """``GPUMDCalculation`` を組み立てる (実行はしない)。"""
        atoms = StructureHandler.read(self.structure_path, format="extxyz")
        atoms.set_pbc(True)

        calculation = GPUMDCalculation(
            atoms,
            potential=self.potential(),
            workdir=self.workdir,
            name="dpmd_water",
            time_step=self.time_step,
            initial_temperature=self.temperature,
            velocity_seed=12345,
            max_atoms=self.max_atoms,
            environment=self.environment,
            overwrite=self.overwrite,
        )

        # --- stage 1: 平衡化 ---
        calculation.add_stage(
            MDStage(
                self.thermostat,
                steps=self.equilibration_steps,
                T_start=self.temperature,
                T_coup=self.T_coup,
                label="equilibration",
                dump=DumpSettings(
                    thermo_interval=self.thermo_interval,
                    traj_interval=self.traj_interval,
                    traj_file="dump.xyz",
                    traj_properties=("velocity",),
                ),
            )
        )
        # --- stage 2: 本計算 (SDC / MSD) ---
        calculation.add_stage(
            MDStage(
                self.thermostat,
                steps=self.production_steps,
                T_start=self.temperature,
                T_coup=self.T_coup,
                label="production",
                dump=DumpSettings(
                    thermo_interval=self.thermo_interval,
                    traj_interval=self.traj_interval,
                    traj_file="dump.xyz",
                    traj_properties=("velocity", "unwrapped_position"),
                    extra=(
                        f"compute_sdc {self.sdc_sample_interval} {self.sdc_correlation_steps}",
                        f"compute_msd {self.msd_sample_interval} {self.msd_correlation_steps}",
                    ),
                ),
            )
        )
        # DP 設定ファイルは run.in と同じディレクトリから参照されるのでコピーしておく
        if self.dp_available and self.dp_setting_file is not None:
            self.workdir.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.dp_setting_file, self.workdir / self.dp_setting_file.name)

        self.calculation = calculation
        return calculation

    # ------------------------------------------------------------------ 予 (期待値)
    def expectations(self) -> ValidationReport:
        """実行前に期待値を並べた予実表を作る。"""
        reference = self.reference
        using_dp = self.using_dp
        potential_note = (
            ""
            if using_dp
            else "DP ではなく NEP で実行したため、ポテンシャル依存量は参考扱い。"
        )
        source_ref = (
            f"参照 example ({reference.directory.name})" if reference.directory else "設計値"
        )

        report = ValidationReport(
            title="DPMD example (liquid water, 512 H2O)",
            context={
                "ポテンシャル": "DP (deepmd .pb)" if using_dp else f"NEP: {self.fallback_potential}",
                "DP 利用可否": "利用可" if using_dp else "利用不可 → NEP へフォールバック",
                "アンサンブル": f"{self.thermostat} {self.temperature:g} K",
                "time_step": f"{self.time_step:g} fs",
                "ステップ数": f"平衡化 {self.equilibration_steps:,d} + 本計算 {self.production_steps:,d}",
                "本計算時間": f"{self.production_steps * self.time_step * 1e-3:.2f} ps",
                **reference.as_context(),
            },
        )

        report.add(
            Expectation(
                key="n_atoms",
                label="原子数",
                expected=float(reference.n_atoms),
                unit="atoms",
                abs_tol=0.0,
                source=source_ref,
            )
        )
        report.add(
            Expectation(
                key="box_length",
                label="セル長 a",
                expected=reference.box_length,
                unit="Å",
                abs_tol=1e-3,
                source=source_ref,
            )
        )
        report.add(
            Expectation(
                key="mean_temperature",
                label="平均温度",
                expected=self.temperature,
                unit="K",
                abs_tol=10.0,
                source="目標温度 (ensemble 設定値)",
                note="Berendsen サーモスタットの平衡値。揺らぎ幅は系サイズに依存する。",
            )
        )
        if reference.mean_temperature is not None:
            report.add(
                Expectation(
                    key="reference_mean_temperature",
                    label="平均温度 (参照計算)",
                    expected=reference.mean_temperature,
                    unit="K",
                    abs_tol=15.0,
                    source=source_ref,
                    note="参照 DP 計算の thermo.out 平均。",
                )
            )
        report.add(
            Expectation(
                key="temperature_std",
                label="温度の標準偏差",
                expected=reference.std_temperature,
                unit="K",
                rel_tol=0.5,
                informative=not using_dp,
                source=source_ref,
                note="N=1536 の正準揺らぎ。ポテンシャルにはあまり依存しない。",
            )
        )
        report.add(
            Expectation(
                key="potential_energy_per_atom",
                label="ポテンシャルエネルギー",
                expected=reference.potential_energy_per_atom,
                unit="eV/atom",
                rel_tol=0.05,
                informative=not using_dp,
                source=source_ref,
                note="エネルギーの原点はポテンシャルごとに異なるため、DP 以外では比較不可。"
                + potential_note,
            )
        )
        report.add(
            Expectation(
                key="pressure",
                label="平均圧力",
                expected=reference.pressure,
                unit="GPa",
                abs_tol=0.5,
                informative=not using_dp,
                source=source_ref,
                note="定積計算なのでポテンシャル依存が大きい。" + potential_note,
            )
        )
        report.add(
            Expectation(
                key="diffusion_sdc",
                label="自己拡散係数 (VACF 積分)",
                expected=reference.diffusion_sdc,
                unit="1e-9 m²/s",
                rel_tol=0.5,
                informative=True,
                source=source_ref,
                note="参照 example は 5 ps しか流しておらず VACF 積分が未収束 "
                f"(参照値 {reference.diffusion_sdc:.1f} は実験値 ~2.3 より 1 桁大きい)。"
                if reference.diffusion_sdc
                else "参照値なし。",
            )
        )
        report.add(
            Expectation(
                key="diffusion_msd",
                label="自己拡散係数 (MSD 法)",
                expected=2.3,
                unit="1e-9 m²/s",
                rel_tol=1.0,
                informative=True,
                source="実験値 (300 K の液体水, ~2.3e-9 m²/s)",
                note="短時間 MD では統計誤差が大きい。桁が合っていれば妥当と判断する。",
            )
        )
        report.add(
            Expectation(
                key="wall_time",
                label="実行時間",
                expected=None,
                unit="s",
                informative=True,
                source="-",
            )
        )
        report.add(
            Expectation(
                key="speed",
                label="計算速度",
                expected=None,
                unit="atom·step/s",
                informative=True,
                source="-",
            )
        )
        self.report = report
        return report

    # ------------------------------------------------------------------ 実行
    def run(self, **run_kwargs) -> GPUMDResult:
        calculation = self.calculation or self.build()
        self.result = calculation.run(**run_kwargs)
        return self.result

    # ------------------------------------------------------------------ 実 (実測値)
    def measure(self) -> dict[str, float | None]:
        """実行結果から実測値を取り出す。"""
        if self.result is None:
            raise RuntimeError("先に run() を実行してください。")
        analyzer = self.result.analyzer()
        frame = analyzer.thermo.frame
        production = frame[frame["stage"] == frame["stage"].max()]
        if len(production) < 5:
            production = frame

        atoms = StructureHandler.read_model(self.result.workdir / "model.xyz")
        measured: dict[str, float | None] = {
            "n_atoms": float(len(atoms)),
            "box_length": float(atoms.cell.cellpar()[0]),
            "mean_temperature": float(production["temperature"].mean()),
            "reference_mean_temperature": float(production["temperature"].mean()),
            "temperature_std": float(production["temperature"].std(ddof=1)),
            "potential_energy_per_atom": float(production["potential_energy_per_atom"].mean()),
            "pressure": float(production["pressure"].mean()),
            "wall_time": self.result.elapsed,
        }
        n_steps = self.equilibration_steps + self.production_steps
        if self.result.elapsed and self.result.elapsed > 0:
            measured["speed"] = len(atoms) * n_steps / self.result.elapsed

        try:
            measured["diffusion_sdc"] = analyzer.diffusion_coefficient(source="sdc")[
                "D_mean_1e-9_m2_per_s"
            ]
        except (FileNotFoundError, IndexError):
            measured["diffusion_sdc"] = None
        try:
            measured["diffusion_msd"] = analyzer.diffusion_coefficient(source="msd")[
                "D_mean_1e-9_m2_per_s"
            ]
        except (FileNotFoundError, IndexError):
            measured["diffusion_msd"] = None
        return measured

    def validate(self) -> ValidationReport:
        """予実表に実測値を書き込む。"""
        report = self.report or self.expectations()
        report.record_many(self.measure())
        return report

    # ------------------------------------------------------------------ 一括実行
    def execute(self, *, plot: bool = True, **run_kwargs) -> ValidationReport:
        """組み立て → 実行 → 解析 → 予実表の書き出しまでを一気に行う。"""
        self.build()
        self.expectations()
        self.run(**run_kwargs)
        report = self.validate()
        output = self.workdir / "analysis"
        if plot:
            self.result.plot()
            try:
                report.plot(output / "validation.png")
            except ValueError:
                pass
        report.write(output)
        return report


def _existing(path: Path | str | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path).expanduser()
    return resolved.resolve() if resolved.exists() else None
