"""GPUMD による MD 計算を扱うメインクラス。

``GPUMDCalculation`` は「構造の準備 → run.in の生成 → gpumd の実行 →
解析・可視化・トラジェクトリ変換」までを一貫して面倒を見る。

Examples
--------
>>> calc = GPUMDCalculation("POSCAR", potential="nep.txt", workdir="runs/si")
>>> calc.set_dump(thermo=100, traj=500)
>>> calc.nvt(temperature=300, steps=20000)          # 定温
>>> calc.heating(300, 1200, steps=40000)            # 昇温
>>> calc.npt(temperature=1200, pressure=0.0, steps=20000)
>>> result = calc.run()
>>> result.plot()                                   # analysis/*.png
>>> result.to_xdatcar()
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ase import Atoms

from .analysis import MDAnalyzer, ThermoData
from .config import CommandResult, GPUMDEnvironment
from .inputs import (
    ALL_ENSEMBLES,
    NPT_ENSEMBLES,
    NVT_ENSEMBLES,
    DumpSettings,
    MDStage,
    RunInputBuilder,
)
from .profiles import TemperatureProfile
from .structure import DEFAULT_MAX_ATOMS, StructureHandler
from .trajectory import TrajectoryConverter

__all__ = ["GPUMDCalculation", "GPUMDResult"]


@dataclass
class GPUMDResult:
    """1 回の GPUMD 実行の結果。"""

    workdir: Path
    command: CommandResult | None
    n_atoms: int
    run_in: Path
    name: str = ""

    # ------------------------------------------------------------------ 解析
    @property
    def succeeded(self) -> bool:
        return self.command is not None and self.command.ok

    @property
    def elapsed(self) -> float:
        return self.command.elapsed if self.command else float("nan")

    def analyzer(self) -> MDAnalyzer:
        return MDAnalyzer(self.workdir, n_atoms=self.n_atoms)

    def thermo(self) -> ThermoData:
        return ThermoData.from_directory(self.workdir, n_atoms=self.n_atoms)

    def summary(self, drop_fraction: float = 0.3):
        return self.thermo().summary(drop_fraction=drop_fraction)

    def plot(self, **kwargs) -> list[Path]:
        """解析プロットを ``<workdir>/analysis`` に保存する。"""
        return self.analyzer().plot_all(**kwargs)

    # ------------------------------------------------------------ トラジェクトリ
    def trajectory(self, filename: str = "dump.xyz") -> TrajectoryConverter:
        return TrajectoryConverter(self.workdir / filename)

    def to_xdatcar(self, output: Path | str | None = None, **kwargs) -> Path:
        return self.trajectory().to_xdatcar(output or self.workdir / "XDATCAR", **kwargs)

    def to_ase_traj(self, output: Path | str | None = None, **kwargs) -> Path:
        return self.trajectory().to_ase_traj(output or self.workdir / "md.traj", **kwargs)

    def final_structure(self) -> Atoms:
        return self.trajectory().last()

    def __str__(self) -> str:
        status = "OK" if self.succeeded else "FAILED"
        return f"<GPUMDResult {self.name or self.workdir.name} {status} {self.elapsed:.1f}s>"


class GPUMDCalculation:
    """GPUMD の MD 計算を組み立てて実行するクラス。

    Parameters
    ----------
    structure
        構造ファイルのパス (POSCAR / CIF / xyz / traj など) か ASE ``Atoms``。
    potential
        NEP ポテンシャルファイルのパス。DP など複数引数が必要な場合は
        ``["dp.txt", "model.pb"]`` のようにシーケンスで渡す。
    workdir
        計算ディレクトリ。存在しなければ作成する。
    name
        ログ表示用の名前 (既定は ``workdir`` 名)。
    time_step
        時間刻み [fs]。
    initial_temperature
        ``velocity`` キーワードで与える初期温度 [K]。
        ``None`` なら最初のステージの ``T_start`` を使う。
    repeat / min_cell_length
        スーパーセルの作り方。``repeat`` 優先。
    orthorhombic
        直方晶セルへ変換するか (NPT の等方セル制御に必要)。
    max_atoms
        原子数の上限 (mission.md: 多くとも 1 万程度)。
    environment
        :class:`~gpumd_toolkit.config.GPUMDEnvironment`。``None`` で既定。
    overwrite
        既存の計算ディレクトリを削除してから始めるか。
    """

    def __init__(
        self,
        structure: Path | str | Atoms,
        potential: Path | str | Sequence[str],
        workdir: Path | str,
        *,
        name: str | None = None,
        time_step: float = 1.0,
        initial_temperature: float | None = None,
        velocity_seed: int | None = None,
        repeat: Sequence[int] | int | None = None,
        min_cell_length: float | None = None,
        orthorhombic: bool = False,
        max_atoms: int = DEFAULT_MAX_ATOMS,
        groupings: list[list[list[int]]] | None = None,
        environment: GPUMDEnvironment | None = None,
        overwrite: bool = False,
    ) -> None:
        self.workdir = Path(workdir).expanduser().resolve()
        if overwrite and self.workdir.exists():
            shutil.rmtree(self.workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)

        self.name = name or self.workdir.name
        self.environment = environment or GPUMDEnvironment()
        self.max_atoms = max_atoms
        self.groupings = groupings

        # --- 構造 ---
        if isinstance(structure, Atoms):
            atoms = structure.copy()
            self.source_structure = None
        else:
            self.source_structure = Path(structure).expanduser().resolve()
            atoms = StructureHandler.read(self.source_structure)
        self.atoms = StructureHandler.make_cell(
            atoms,
            repeat=repeat,
            min_length=min_cell_length,
            max_atoms=max_atoms,
            orthorhombic=orthorhombic,
        )
        self.structure_info = StructureHandler.info(self.atoms)

        # --- ポテンシャル ---
        self.potential = self._resolve_potential(potential)

        # --- run.in ---
        self.builder = RunInputBuilder(
            potential=self.potential,
            time_step=time_step,
            initial_temperature=initial_temperature,
            velocity_seed=velocity_seed,
            header_comments=(
                f"generated by gpumd_toolkit for '{self.name}'",
                f"structure: {self.structure_info}",
            ),
        )
        self._result: GPUMDResult | None = None

    # ------------------------------------------------------------------ 内部
    def _resolve_potential(self, potential) -> str | list[str]:
        """ポテンシャルファイルを絶対パス化し、存在を確認する。"""
        if isinstance(potential, (str, Path)):
            items = [potential]
            scalar = True
        else:
            items = list(potential)
            scalar = False
        resolved: list[str] = []
        for i, item in enumerate(items):
            path = Path(item).expanduser()
            if path.exists():
                resolved.append(str(path.resolve()))
            elif i == 0:
                raise FileNotFoundError(f"ポテンシャルファイルが見つかりません: {item}")
            else:
                # DP の設定ファイル等、GPUMD 側で解決される引数はそのまま通す
                resolved.append(str(item))
        return resolved[0] if scalar else resolved

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def stages(self) -> list[MDStage]:
        return self.builder.stages

    # ------------------------------------------------------------------ 設定
    def set_dump(
        self,
        *,
        thermo: int | None = 100,
        traj: int | None = 1000,
        traj_file: str = "dump.xyz",
        traj_properties: Sequence[str] = ("velocity", "force", "potential"),
        traj_precision: str = "single",
        restart: int | None = None,
        extra: Sequence[str] = (),
    ) -> "GPUMDCalculation":
        """既定の出力設定を変更する (mission.md: 任意ステップごとのトラジェクトリ保存)。"""
        self.builder.default_dump = DumpSettings(
            thermo_interval=thermo,
            traj_interval=traj,
            traj_file=traj_file,
            traj_properties=tuple(traj_properties),
            traj_precision=traj_precision,
            restart_interval=restart,
            extra=tuple(extra),
        )
        return self

    def minimize(
        self, *, method: str = "fire", force_tolerance: float = 1e-4, max_steps: int = 1000
    ) -> "GPUMDCalculation":
        """MD の前に構造最適化を挟む。"""
        self.builder.set_minimize(method, force_tolerance, max_steps)
        return self

    # ------------------------------------------------------------ ステージ追加
    def add_stage(self, stage: MDStage) -> "GPUMDCalculation":
        self.builder.add_stage(stage)
        return self

    def nve(self, *, steps: int, **kwargs) -> "GPUMDCalculation":
        """NVE (ミクロカノニカル) ステージを追加する。"""
        return self.add_stage(MDStage("nve", steps=steps, **kwargs))

    def nvt(
        self,
        *,
        temperature: float,
        steps: int,
        temperature_end: float | None = None,
        thermostat: str = "nvt_nhc",
        T_coup: float = 100.0,
        **kwargs,
    ) -> "GPUMDCalculation":
        """NVT (カノニカル) ステージを追加する。

        ``temperature_end`` を与えると、その run の間に目標温度が線形に変化する
        (昇温・降温)。
        """
        if thermostat not in NVT_ENSEMBLES:
            raise ValueError(f"NVT の thermostat は {NVT_ENSEMBLES} から選びます。")
        return self.add_stage(
            MDStage(
                thermostat,
                steps=steps,
                T_start=temperature,
                T_end=temperature_end,
                T_coup=T_coup,
                **kwargs,
            )
        )

    def npt(
        self,
        *,
        temperature: float,
        steps: int,
        pressure: float | Sequence[float] = 0.0,
        temperature_end: float | None = None,
        barostat: str = "npt_scr",
        T_coup: float = 100.0,
        p_coup: float = 1000.0,
        elastic_modulus: float | Sequence[float] = 100.0,
        **kwargs,
    ) -> "GPUMDCalculation":
        """NPT (等温等圧) ステージを追加する。

        ``pressure`` は GPa。スカラー (等方)、3 成分 (xx, yy, zz)、
        6 成分 (xx, yy, zz, yz, xz, xy) に対応する。
        ``npt_ber`` / ``npt_scr`` では ``elastic_modulus`` [GPa] の概算値が必要
        (桁が合っていればよい)。
        """
        if barostat not in NPT_ENSEMBLES:
            raise ValueError(f"NPT の barostat は {NPT_ENSEMBLES} から選びます。")
        if barostat != "npt_mttk" and not self.structure_info.orthorhombic:
            import warnings

            warnings.warn(
                f"{barostat} は直交セルを前提とします。"
                " orthorhombic=True で変換するか npt_mttk (tri) を使ってください。",
                RuntimeWarning,
            )
        return self.add_stage(
            MDStage(
                barostat,
                steps=steps,
                T_start=temperature,
                T_end=temperature_end,
                T_coup=T_coup,
                pressure=pressure,
                elastic_modulus=elastic_modulus,
                p_coup=p_coup,
                **kwargs,
            )
        )

    # --------------------------------------------------- 温度プロファイル API
    def heating(self, T_start: float, T_end: float, *, steps: int, **kwargs):
        """昇温ステージ。"""
        return self.nvt(temperature=T_start, temperature_end=T_end, steps=steps, **kwargs)

    def cooling(self, T_start: float, T_end: float, *, steps: int, **kwargs):
        """降温ステージ。"""
        return self.nvt(temperature=T_start, temperature_end=T_end, steps=steps, **kwargs)

    def apply_profile(
        self,
        profile: TemperatureProfile,
        *,
        ensemble: str = "nvt_nhc",
        pressure: float | Sequence[float] | None = None,
        **kwargs,
    ) -> "GPUMDCalculation":
        """任意の温度プロファイルをステージ列へ展開する。

        Examples
        --------
        >>> profile = TemperatureProfile.from_points(
        ...     [(0, 300), (10, 1000), (20, 1000), (40, 300)], time_step_fs=1.0)
        >>> calc.apply_profile(profile)
        """
        if ensemble not in ALL_ENSEMBLES:
            raise ValueError(f"未知のアンサンブル: {ensemble}")
        for segment in profile:
            stage_kwargs = dict(kwargs)
            if ensemble in NPT_ENSEMBLES:
                stage_kwargs.setdefault("pressure", 0.0 if pressure is None else pressure)
            self.add_stage(
                MDStage(
                    ensemble,
                    steps=segment.steps,
                    T_start=segment.T_start,
                    T_end=segment.T_end,
                    label=segment.label,
                    **stage_kwargs,
                )
            )
        return self

    # ------------------------------------------------------------------ 実行
    def write_inputs(self) -> dict[str, Path]:
        """``model.xyz`` と ``run.in`` を書き出す。"""
        if self.builder.initial_temperature is None and self.builder.stages:
            first = self.builder.stages[0]
            if first.T_start is not None:
                self.builder.initial_temperature = first.T_start
        model = StructureHandler.write_model(
            self.atoms, self.workdir / "model.xyz", groupings=self.groupings
        )
        run_in = self.builder.write(self.workdir / "run.in")
        (self.workdir / "metadata.json").write_text(
            json.dumps(self.metadata(), indent=2, ensure_ascii=False)
        )
        return {"model": model, "run_in": run_in}

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "workdir": str(self.workdir),
            "source_structure": str(self.source_structure) if self.source_structure else None,
            "potential": self.potential,
            "n_atoms": self.n_atoms,
            "structure": self.structure_info.as_dict(),
            "time_step_fs": self.builder.time_step,
            "total_steps": self.builder.total_steps,
            "total_time_ps": self.builder.total_time_ps(),
            "stages": [
                {
                    "label": stage.label,
                    "ensemble": stage.ensemble,
                    "steps": stage.steps,
                    "T_start": stage.T_start,
                    "T_end": stage.T_end,
                    "pressure": stage.pressure,
                }
                for stage in self.builder.stages
            ],
        }

    def preview(self) -> str:
        """生成される ``run.in`` を文字列で返す (実行前の確認用)。"""
        return self.builder.build()

    def run(
        self,
        *,
        dry_run: bool = False,
        timeout: float | None = None,
        check: bool = True,
        analyze: bool = False,
    ) -> GPUMDResult:
        """入力を書き出して ``gpumd`` を実行する。

        Parameters
        ----------
        dry_run
            入力ファイルの生成のみ行い、実行はしない。
        timeout
            秒。超過すると ``subprocess.TimeoutExpired``。
        check
            異常終了時に例外を投げるか。
        analyze
            実行後に解析プロットまで自動生成するか。
        """
        paths = self.write_inputs()
        command: CommandResult | None = None
        if not dry_run:
            command = self.environment.run_gpumd(
                self.workdir, timeout=timeout, check=check
            )
        result = GPUMDResult(
            workdir=self.workdir,
            command=command,
            n_atoms=self.n_atoms,
            run_in=paths["run_in"],
            name=self.name,
        )
        self._result = result
        if analyze and result.succeeded:
            result.plot()
        return result

    @property
    def result(self) -> GPUMDResult | None:
        return self._result

    # ------------------------------------------------------------------ 表示
    def __repr__(self) -> str:
        return (
            f"GPUMDCalculation(name={self.name!r}, N={self.n_atoms}, "
            f"stages={len(self.builder.stages)}, "
            f"total={self.builder.total_time_ps():.2f} ps)"
        )

    def describe(self) -> str:
        lines = [
            f"GPUMDCalculation '{self.name}'",
            f"  workdir   : {self.workdir}",
            f"  structure : {self.structure_info}",
            f"  potential : {self.potential}",
            f"  time_step : {self.builder.time_step} fs",
            f"  stages    : {len(self.builder.stages)}"
            f"  (total {self.builder.total_steps:,d} steps ="
            f" {self.builder.total_time_ps():.2f} ps)",
        ]
        for i, stage in enumerate(self.builder.stages, 1):
            lines.append(
                f"    {i:2d}. {stage.label:<30s} {stage.ensemble:<9s} {stage.steps:>9,d} steps"
            )
        return "\n".join(lines)
