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
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ase import Atoms

from .analysis import MDAnalyzer, ThermoData
from .config import CommandResult, GPUMDEnvironment
from .groups import GroupingScheme
from .inputs import (
    ALL_ENSEMBLES,
    NPH_ENSEMBLES,
    NPT_ENSEMBLES,
    NVT_ENSEMBLES,
    DumpSettings,
    MDStage,
    RunInputBuilder,
)
from .inputs.ensembles import Ensemble
from .outputs import OutputReader
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

    def outputs(self) -> OutputReader:
        """GPUMD が書いた出力ファイル群への読み込み口。"""
        return OutputReader(self.workdir)

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

    def to_xyz(self, output: Path | str | None = None, **kwargs) -> Path:
        """拡張 XYZ で書き出す (間引きやフレーム切り出しをしたいとき)。"""
        return self.trajectory().to_xyz(output or self.workdir / "trajectory.xyz", **kwargs)

    def to_xdatcar(self, output: Path | str | None = None, **kwargs) -> Path:
        """VASP の XDATCAR 形式で書き出す。"""
        return self.trajectory().to_xdatcar(output or self.workdir / "XDATCAR", **kwargs)

    def convert_trajectory(
        self, formats: Sequence[str] = ("xyz", "xdatcar"), **kwargs
    ) -> dict[str, Path]:
        """トラジェクトリを複数形式へ一括変換する (既定は xyz と XDATCAR)。"""
        return self.trajectory().convert_all(self.workdir, formats=formats, **kwargs)

    def to_ase_traj(self, output: Path | str | None = None, **kwargs) -> Path:
        """ASE の ``.traj`` で書き出す (扱いやすさでは xyz / XDATCAR を推奨)。"""
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
        スーパーセルの作り方。``repeat`` 優先。``min_cell_length`` は
        辺の長さではなく **面間距離** の下限として扱う (斜めのセル対策)。
    orthorhombic
        直方晶セルへ変換するか (``npt_ber`` / ``npt_scr`` の 1 成分・3 成分
        指定に必要)。三斜晶のまま NPT したい場合は不要 —
        :meth:`npt` が自動で 6 成分指定に切り替える。
    reduce_cell
        ``'niggli'`` / ``'minkowski'`` を指定すると、スーパーセル化の前に
        セルを簡約してできるだけ立方体に近づける (斜めのセルで原子数を節約)。
    standard_cell
        セルを下三角形 (LAMMPS 標準形) に回転するか。
    fast_read / cache_structure
        構造の読み込みで高速経路 (専用パーサ + xyz キャッシュ) を使うか。
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
        reduce_cell: str | None = None,
        standard_cell: bool = False,
        fast_read: bool = True,
        cache_structure: bool = True,
        max_atoms: int = DEFAULT_MAX_ATOMS,
        groupings: GroupingScheme | list[list[list[int]]] | None = None,
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
        self.groupings = (
            groupings.to_groupings() if isinstance(groupings, GroupingScheme) else groupings
        )
        self.grouping_scheme = groupings if isinstance(groupings, GroupingScheme) else None

        # --- 構造 ---
        if isinstance(structure, Atoms):
            atoms = structure.copy()
            self.source_structure = None
        else:
            self.source_structure = Path(structure).expanduser().resolve()
            atoms = StructureHandler.read(
                self.source_structure, fast=fast_read, cache=cache_structure
            )
        self.atoms = StructureHandler.make_cell(
            atoms,
            repeat=repeat,
            min_length=min_cell_length,
            max_atoms=max_atoms,
            orthorhombic=orthorhombic,
            reduce=reduce_cell,
            standard_cell=standard_cell,
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
    def _resolve_potential(self, potential):
        """ポテンシャルファイルを絶対パス化し、存在を確認する。

        * ``"nep.txt"`` -> ``potential`` 行 1 つ
        * ``["dp.txt", "model.pb"]`` -> 引数が複数ある 1 つのポテンシャル
        * ``[["nep0.txt"], ["nep1.txt"]]`` -> committee (``potential`` 行が複数)
        """
        if isinstance(potential, (str, Path)):
            return self._resolve_potential_args([potential], scalar=True)
        items = list(potential)
        if items and all(not isinstance(item, (str, Path)) for item in items):
            return [self._resolve_potential_args(list(item), scalar=False) for item in items]
        return self._resolve_potential_args(items, scalar=False)

    @staticmethod
    def _resolve_potential_args(items, *, scalar: bool):
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

    def add_ensemble(
        self, ensemble: Ensemble, *, steps: int, **kwargs
    ) -> "GPUMDCalculation":
        """特殊アンサンブル (QTB / NEMD 熱浴 / TTM / PIMD / TI / 衝撃波) を追加する。

        Examples
        --------
        >>> from gpumd_toolkit.inputs.ensembles import PIMD
        >>> calc.add_ensemble(PIMD(num_beads=32, T_start=300), steps=20000)
        """
        if not isinstance(ensemble, Ensemble):
            raise TypeError(
                "add_ensemble には gpumd_toolkit.inputs.ensembles のインスタンスを渡します。"
                " 標準アンサンブルは nve() / nvt() / npt() / add_stage() を使ってください。"
            )
        return self.add_stage(MDStage(ensemble, steps=steps, **kwargs))

    def add_commands(self, *lines: str) -> "GPUMDCalculation":
        """直近のステージの ``run`` 直前に任意のキーワード行を挿入する。

        まだステージが無ければ preamble (``potential`` の直後) に入れる。
        :mod:`gpumd_toolkit.inputs.computes` や
        :mod:`gpumd_toolkit.inputs.modifiers` の戻り値を渡す。
        """
        if not self.builder.stages:
            self.builder.add_preamble(*lines)
            return self
        last = self.builder.stages[-1]
        last.pre_commands = tuple(last.pre_commands) + tuple(lines)
        return self

    def add_preamble(self, *lines: str) -> "GPUMDCalculation":
        """``potential`` / ``time_step`` の直後に行を差し込む (``dftd3`` など)。"""
        self.builder.add_preamble(*lines)
        return self

    def add_action(self, *lines: str) -> "GPUMDCalculation":
        """``run`` を伴わずその場で実行されるキーワード行を足す。

        ``compute_cohesive`` / ``compute_elastic`` / ``compute_phonon`` 用。
        """
        self.builder.add_action(*lines)
        return self

    def ensure_grouping(self) -> int:
        """grouping method が 1 つも無ければ「全原子 1 グループ」を追加する。

        GPUMD の ``fix`` / ``move`` / ``add_force`` / ``add_efield`` /
        ``add_spring`` / ``compute`` / ``ttm`` などは ``model.xyz`` に
        grouping method が定義されていないと
        ``grouping method should < maximum number of grouping methods``
        で起動直後に止まる。グループを使わない場合でも最低 1 つは要る。

        Returns
        -------
        int
            使える grouping method 番号 (既にあればそのまま 0)。
        """
        if self.groupings:
            return 0
        from .groups import GroupingScheme, single_group

        scheme = self.grouping_scheme or GroupingScheme()
        scheme.add(single_group(self.atoms), "all")
        self.grouping_scheme = scheme
        self.groupings = scheme.to_groupings()
        return 0

    def set_replicate(self, na: int, nb: int, nc: int) -> "GPUMDCalculation":
        """``run.in`` の先頭で ``replicate`` する (``compute_phonon`` に必要)。"""
        self.builder.replicate = (int(na), int(nb), int(nc))
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
        tau_T: float | None = None,
        **kwargs,
    ) -> "GPUMDCalculation":
        """NVT (カノニカル) ステージを追加する。

        ``temperature_end`` を与えると、その run の間に目標温度が線形に変化する
        (昇温・降温)。

        熱浴の時定数は ``T_coup`` (= :math:`\\tau_T/\\Delta t`、無次元) か
        ``tau_T`` (**fs**、時間刻みから自動換算) のどちらでも指定できる。
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
                tau_T=tau_T,
                **kwargs,
            )
        )

    def npt(
        self,
        *,
        temperature: float,
        steps: int,
        pressure: float | Sequence[float] = 0.0,
        pressure_end: float | Sequence[float] | None = None,
        temperature_end: float | None = None,
        barostat: str = "npt_scr",
        T_coup: float = 100.0,
        p_coup: float = 1000.0,
        tau_T: float | None = None,
        tau_p: float | None = None,
        elastic_modulus: float | Sequence[float] = 100.0,
        cell_mode: str | None = None,
        fixed_axes: Sequence[str] = (),
        free_axes: Sequence[str] | None = None,
        **kwargs,
    ) -> "GPUMDCalculation":
        """NPT (等温等圧) ステージを追加する。

        Parameters
        ----------
        pressure, pressure_end
            目標圧力 [GPa]。スカラー (静水圧)、3 成分 (xx, yy, zz)、
            6 成分 (xx, yy, zz, yz, xz, xy) に対応する。
            ``pressure_end`` は ``npt_mttk`` でのみ有効 (圧力ランプ)。
        T_coup, p_coup
            :math:`\\tau_T/\\Delta t`, :math:`\\tau_p/\\Delta t` (無次元)。
        tau_T, tau_p
            時定数を **fs** で指定する場合はこちら (時間刻みから自動換算)。
        cell_mode
            ``npt_ber`` / ``npt_scr`` のセル自由度
            (``'iso'`` / ``'ortho'`` / ``'tri'``)。``None`` なら構造のセル形状と
            ``pressure`` の成分数から自動で決める (三斜晶なら ``'tri'``)。
        fixed_axes, free_axes
            変調を許す / 禁じる軸 (``'x'``, ``'y'``, ``'z'``, ``'yz'``,
            ``'xz'``, ``'xy'``)。``npt_ber`` / ``npt_scr`` では固定軸の弾性率を
            2000 GPa 超にして GPUMD 側のカップリングを 0 にする。
            ``npt_mttk`` では ``free_axes`` がそのまま ``direction`` になる。
        elastic_modulus
            ``npt_ber`` / ``npt_scr`` に必要な弾性率の概算値 [GPa]
            (桁が合っていればよい)。

        Examples
        --------
        >>> calc.npt(temperature=300, steps=10000, pressure=0.0)       # 等方
        >>> calc.npt(temperature=300, steps=10000, free_axes=("z",))   # c 軸だけ動かす
        >>> calc.npt(temperature=300, steps=10000, barostat="npt_mttk",
        ...          free_axes=("x", "y", "z"))                        # aniso 相当
        """
        if barostat not in NPT_ENSEMBLES:
            raise ValueError(f"NPT の barostat は {NPT_ENSEMBLES} から選びます。")
        if barostat != "npt_mttk":
            cell_mode = self._resolve_cell_mode(
                barostat, cell_mode, pressure, fixed_axes, free_axes
            )
        return self.add_stage(
            MDStage(
                barostat,
                steps=steps,
                T_start=temperature,
                T_end=temperature_end,
                T_coup=T_coup,
                tau_T=tau_T,
                pressure=pressure,
                pressure_end=pressure_end,
                elastic_modulus=elastic_modulus,
                p_coup=p_coup,
                tau_p=tau_p,
                cell_mode=cell_mode,
                fixed_axes=fixed_axes,
                free_axes=free_axes,
                **kwargs,
            )
        )

    def nph(
        self,
        *,
        steps: int,
        pressure: float | Sequence[float] = 0.0,
        pressure_end: float | Sequence[float] | None = None,
        direction: str | Sequence[str] = "iso",
        p_period: float = 1000.0,
        tau_p: float | None = None,
        **kwargs,
    ) -> "GPUMDCalculation":
        """NPH (等エンタルピー) ステージを追加する (``nph_mttk``)。

        熱浴を使わずセルだけ動かしたい場合 (二相法の融点計算など) に使う。
        """
        return self.add_stage(
            MDStage(
                "nph_mttk",
                steps=steps,
                pressure=pressure,
                pressure_end=pressure_end,
                mttk_direction=direction,
                p_period=p_period,
                tau_p=tau_p,
                **kwargs,
            )
        )

    def _resolve_cell_mode(
        self,
        barostat: str,
        cell_mode: str | None,
        pressure,
        fixed_axes: Sequence[str],
        free_axes: Sequence[str] | None,
    ) -> str | None:
        """``npt_ber`` / ``npt_scr`` のセル自由度を構造から決める。

        GPUMD は 1 成分・3 成分の圧力指定を直交セルにしか許さないので、
        三斜晶セルなら自動的に 6 成分 (``'tri'``) に切り替える。
        """
        import warnings

        if cell_mode is not None:
            if cell_mode != "tri" and not self.structure_info.orthorhombic:
                warnings.warn(
                    f"cell_mode='{cell_mode}' は直交セル専用です。"
                    " 三斜晶セルでは GPUMD が 'Cannot use triclinic box with only"
                    f" {1 if cell_mode == 'iso' else 3} target pressure components' で止まります。",
                    RuntimeWarning,
                )
            return cell_mode
        if self.structure_info.orthorhombic:
            return None  # MDStage 側で pressure / axes から決める
        probe = MDStage(
            barostat,
            steps=1,
            T_start=1.0,
            pressure=pressure,
            fixed_axes=fixed_axes,
            free_axes=free_axes,
        )
        inferred = probe.resolve_cell_mode()
        if inferred != "tri":
            warnings.warn(
                f"セルが三斜晶 ({self.structure_info.cell_shape}) なので "
                f"{barostat} を 6 成分指定 (cell_mode='tri') に切り替えました。"
                " 直方晶セルで計算したい場合は orthorhombic=True を指定してください。",
                RuntimeWarning,
            )
        return "tri"

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
            if ensemble in NPT_ENSEMBLES + NPH_ENSEMBLES:
                stage_kwargs.setdefault("pressure", 0.0 if pressure is None else pressure)
                if ensemble in ("npt_ber", "npt_scr"):
                    stage_kwargs.setdefault(
                        "cell_mode",
                        self._resolve_cell_mode(
                            ensemble,
                            stage_kwargs.get("cell_mode"),
                            stage_kwargs["pressure"],
                            stage_kwargs.get("fixed_axes", ()),
                            stage_kwargs.get("free_axes"),
                        ),
                    )
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
        self._default_initial_temperature()
        self.check_cell(strict=False)
        model = StructureHandler.write_model(
            self.atoms, self.workdir / "model.xyz", groupings=self.groupings
        )
        run_in = self.builder.write(self.workdir / "run.in")
        (self.workdir / "metadata.json").write_text(
            json.dumps(self.metadata(), indent=2, ensure_ascii=False)
        )
        return {"model": model, "run_in": run_in}

    def nep_cutoffs(self) -> tuple[float, float] | None:
        """使用中の NEP の ``(rc_radial, rc_angular)`` [Å]。NEP 以外なら ``None``。"""
        potential = self.potential
        if isinstance(potential, list):
            potential = potential[0]
            if isinstance(potential, list):
                potential = potential[0]
        return StructureHandler.nep_cutoffs(potential)

    def check_cell(self, *, strict: bool = False) -> dict:
        """GPUMD が受け付けるセル形状かどうかを確認する。

        細長いセル (NEMD・衝撃波) で GPUMD が起動直後に落ちるのを
        実行前に検出する。``strict=True`` なら例外、既定は警告。
        """
        cutoffs = self.nep_cutoffs()
        if cutoffs is None:
            return {"ok": True, "message": ""}
        result = StructureHandler.check_nep_box(self.atoms, cutoffs[0])
        if not result["ok"]:
            if strict:
                raise ValueError(result["message"])
            warnings.warn(result["message"], RuntimeWarning)
        return result

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
            "groupings": (
                self.grouping_scheme.names if self.grouping_scheme is not None else None
            ),
            "stages": [
                {
                    **stage.metadata(),
                    "cell_control": stage.describe_cell_control(self.builder.time_step),
                }
                for stage in self.builder.stages
            ],
        }

    def _default_initial_temperature(self) -> None:
        """``velocity`` 行の温度を最初のステージから決める。"""
        if self.builder.initial_temperature is not None or not self.builder.stages:
            return
        first = self.builder.stages[0]
        temperature = first.T_start
        if temperature is None and first.ensemble_spec is not None:
            spec = first.ensemble_spec
            temperature = getattr(spec, "T_start", None) or getattr(
                spec, "temperature", None
            )
        if temperature is not None:
            self.builder.initial_temperature = float(temperature)

    def preview(self) -> str:
        """生成される ``run.in`` を文字列で返す (実行前の確認用)。"""
        self._default_initial_temperature()
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
                f"    {i:2d}. {stage.label:<30s} {stage.ensemble_name:<12s}"
                f" {stage.steps:>9,d} steps"
            )
            if (
                stage.ensemble_spec is not None
                or stage.ensemble_name in NPT_ENSEMBLES + NPH_ENSEMBLES
            ):
                lines.append(
                    f"        {stage.describe_cell_control(self.builder.time_step)}"
                )
        return "\n".join(lines)
