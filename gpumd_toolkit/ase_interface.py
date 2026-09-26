"""ASE ベースで NEP/GPUMD を使うためのクラス群。

3 つのバックエンドを同じ API で切り替えられるようにしている。

``cpu``    calorine の ``CPUNEP`` (GPUMD 同梱の nep_cpu を Python から直接呼ぶ)
``pynep``  pyNEP (NEP_CPU ベース)。CPU のみの環境向け。
``gpu``    calorine の ``GPUNEP`` (``gpumd`` 実行ファイルをファイル経由で呼ぶ)

mission.md の要求どおり ``get_potential_energy()`` と
``get_potential_energies()`` のどちらもすべてのバックエンドで使える。
calorine 3.5 の ``CPUNEP`` は per-atom エネルギーを ``results`` に入れ忘れて
いるため、本モジュールの :class:`CPUNEPWithEnergies` で補っている。
"""

from __future__ import annotations

import os
import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes

from .config import GPUMDEnvironment
from .structure import DEFAULT_MAX_ATOMS, StructureHandler

__all__ = [
    "create_nep_calculator",
    "available_backends",
    "axes_to_mask",
    "CPUNEPWithEnergies",
    "GPUNEPWithEnergies",
    "ASEMDRunner",
]

BACKENDS = ("cpu", "pynep", "gpu")

#: ASE の NPT で使える軸名 -> mask のインデックス
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def axes_to_mask(axes: Sequence[str]) -> np.ndarray:
    """``('x', 'z')`` のような軸指定を ASE の NPT ``mask`` に変換する。

    指定した軸だけがセル変調の対象になり、他は固定される。
    ``'xy'`` のような剪断成分を含めると 3x3 の mask を返す。
    """
    labels = [str(a).strip().lower() for a in axes]
    if any(len(label) == 2 for label in labels):
        mask = np.zeros((3, 3), dtype=int)
        for label in labels:
            if len(label) == 1:
                index = _AXIS_INDEX[label]
                mask[index, index] = 1
            else:
                i, j = _AXIS_INDEX[label[0]], _AXIS_INDEX[label[1]]
                mask[i, j] = mask[j, i] = 1
        return mask
    mask = np.zeros(3, dtype=int)
    for label in labels:
        if label not in _AXIS_INDEX:
            raise ValueError(f"未知の軸名 '{label}'。'x' / 'y' / 'z' から選びます。")
        mask[_AXIS_INDEX[label]] = 1
    return mask


def _hydrostatic(pressure: float | Sequence[float]) -> float:
    """静水圧 [GPa] を取り出す (3/6 成分なら対角の平均)。"""
    if isinstance(pressure, (int, float)):
        return float(pressure)
    values = [float(p) for p in pressure]
    return sum(values[:3]) / min(3, len(values))


def _stress_vector(pressure: float | Sequence[float]) -> np.ndarray:
    """圧力 [GPa] を ASE の 6 成分 (Voigt) 応力ベクトルに直す。"""
    if isinstance(pressure, (int, float)):
        return np.array([pressure] * 3 + [0.0] * 3, dtype=float)
    values = [float(p) for p in pressure]
    if len(values) == 3:
        return np.array(values + [0.0] * 3, dtype=float)
    if len(values) == 6:
        # GPUMD 並び (xx, yy, zz, yz, xz, xy) は ASE の Voigt 並びと同じ
        return np.array(values, dtype=float)
    raise ValueError("pressure_GPa はスカラー / 3 成分 / 6 成分のいずれかです。")


# --------------------------------------------------------------------- calculators
class CPUNEPWithEnergies:
    """``calorine.calculators.CPUNEP`` に per-atom エネルギーを足すファクトリ。

    クラスは import 時ではなく生成時に組み立てる (calorine が無い環境でも
    本モジュールを import できるようにするため)。
    """

    def __new__(cls, model_filename: str, atoms: Atoms | None = None, **kwargs):
        from calorine.calculators import CPUNEP

        class _CPUNEP(CPUNEP):
            """per-atom エネルギー (``energies``) を補完した CPUNEP。"""

            def calculate(self, atoms=None, properties=None, system_changes=all_changes):
                if properties is None:
                    properties = self.implemented_properties
                super().calculate(atoms, properties, system_changes)
                if "energies" in properties and "energies" not in self.results:
                    if "charge" in self.model_type:
                        energies, *_ = self.nepy.get_potential_forces_virials_and_charges()
                    else:
                        energies, *_ = self.nepy.get_potential_forces_and_virials()
                    self.results["energies"] = np.asarray(energies)

        return _CPUNEP(model_filename=model_filename, atoms=atoms, **kwargs)


class GPUNEPWithEnergies:
    """``calorine.calculators.GPUNEP`` を新しい GPUMD 向けに直したファクトリ。

    2 点を修正している。

    1. calorine 3.5 の 1 点計算は ``dump_force`` / ``dump_position`` を使うが、
       これらは新しい GPUMD で削除され ``dump_xyz`` に統合された。
       そのままでは ``Input Error: dump_force has been removed`` で失敗する。
    2. per-atom エネルギー (``energies``) が results に入らない。
       ``dump_xyz ... force potential`` にすると ``energy_atom`` 列が得られるので、
       力と一緒に読み出す。
    """

    def __new__(cls, model_filename: str, atoms: Atoms | None = None, **kwargs):
        from ase.io import read as ase_read
        from calorine.calculators import GPUNEP

        class _GPUNEP(GPUNEP):
            """新しい GPUMD の dump_xyz を使う GPUNEP。"""

            #: 1 点計算で per-atom 量を書き出すファイル
            SINGLE_POINT_FILE = "single_point.xyz"

            base_implemented_properties = GPUNEP.base_implemented_properties + ["energies"]
            base_single_point_parameters = [
                ("dump_thermo", 1),
                ("dump_xyz", (1, SINGLE_POINT_FILE, "force", "potential")),
                ("velocity", 1e-24),
                ("time_step", 1e-6),  # 1 zeptosecond: 実質的に静止した 1 ステップ
                ("ensemble", "nve"),
                ("run", 1),
            ]

            def _read_forces(self) -> None:
                """``single_point.xyz`` から力と per-atom エネルギーを読む。"""
                path = os.path.join(self._directory, self.SINGLE_POINT_FILE)
                frame = ase_read(path, index=-1, format="extxyz")
                self.results["forces"] = frame.get_forces()
                if "energy_atom" in frame.arrays:
                    self.results["energies"] = np.asarray(frame.arrays["energy_atom"])

            def get_forces_from_file(self) -> np.ndarray:
                path = os.path.join(self._directory, self.SINGLE_POINT_FILE)
                return ase_read(path, index=-1, format="extxyz").get_forces()

        return _GPUNEP(model_filename=model_filename, atoms=atoms, **kwargs)


def available_backends() -> dict[str, bool]:
    """各バックエンドが使えるかを返す。"""
    status: dict[str, bool] = {}
    try:
        import calorine.calculators  # noqa: F401

        status["cpu"] = True
        status["gpu"] = shutil.which("gpumd") is not None or _gpumd_via_env()
    except ImportError:
        status["cpu"] = False
        status["gpu"] = False
    try:
        import pynep.calculate  # noqa: F401

        status["pynep"] = True
    except ImportError:
        status["pynep"] = False
    return status


def _gpumd_via_env() -> bool:
    try:
        return GPUMDEnvironment().which("gpumd") is not None
    except Exception:
        return False


def create_nep_calculator(
    model: Path | str,
    *,
    backend: str = "auto",
    atoms: Atoms | None = None,
    environment: GPUMDEnvironment | None = None,
    **kwargs,
) -> Calculator:
    """NEP の ASE calculator を作る。

    Parameters
    ----------
    model
        ``nep.txt`` 形式のポテンシャルファイル。
    backend
        ``'cpu'`` / ``'pynep'`` / ``'gpu'`` / ``'auto'``。
        ``'auto'`` は GPU が使えれば ``'gpu'``、無ければ ``'cpu'``。
        CPU だけの環境 (NEP_CPU) では ``'cpu'`` か ``'pynep'`` を指定する。
    environment
        ``'gpu'`` バックエンドで ``gpumd`` をどう起動するかの設定。
    """
    model = str(Path(model).expanduser().resolve())
    status = available_backends()

    if backend == "auto":
        backend = "gpu" if status.get("gpu") else ("cpu" if status.get("cpu") else "pynep")
    if backend not in BACKENDS:
        raise ValueError(f"backend は {BACKENDS} から選びます: {backend!r}")
    if not status.get(backend):
        raise RuntimeError(
            f"backend '{backend}' が使えません。利用可能: "
            f"{[k for k, v in status.items() if v]}"
        )

    if backend == "cpu":
        return CPUNEPWithEnergies(model, atoms=atoms, **kwargs)
    if backend == "pynep":
        from pynep.calculate import NEP

        calculator = NEP(model, **kwargs)
        if atoms is not None:
            atoms.calc = calculator
        return calculator

    env = environment or GPUMDEnvironment()
    command = kwargs.pop("command", None)
    if command is None:
        executable = env.which(env.gpumd_command) or env.gpumd_command
        command = executable
    return GPUNEPWithEnergies(model, atoms=atoms, command=command, **kwargs)


# --------------------------------------------------------------------- MD runner
@dataclass
class ASEMDRunner:
    """ASE の積分器で簡易 MD を回すクラス。

    GPUMD 本体で回すほどでもない小さな系・短時間の計算や、
    NEP のポテンシャルエネルギー/力を ASE の他機能 (構造最適化、フォノン、
    EOS など) と組み合わせたいときに使う。

    Parameters
    ----------
    atoms
        対象構造 (``Atoms`` か構造ファイルのパス)。
    model
        NEP ポテンシャルファイル。
    backend
        :func:`create_nep_calculator` の ``backend``。
    workdir
        トラジェクトリ・ログ・図の出力先。
    max_atoms
        原子数の上限チェック。

    Examples
    --------
    >>> runner = ASEMDRunner("POSCAR", model="nep.txt", backend="cpu")
    >>> runner.single_point()
    >>> runner.relax(fmax=0.01)
    >>> runner.run_md(ensemble="nvt", temperature=300, steps=2000, time_step=1.0)
    >>> runner.plot()
    """

    atoms: Atoms | Path | str
    model: Path | str | None = None
    backend: str = "auto"
    workdir: Path | str = "ase_run"
    max_atoms: int = DEFAULT_MAX_ATOMS
    environment: GPUMDEnvironment | None = None
    calculator_kwargs: dict = field(default_factory=dict)
    #: NEP の代わりに使う ASE calculator (テストや他ポテンシャルとの比較用)
    calculator: Calculator | None = None
    #: 構造読み込みで高速経路 (専用パーサ + xyz キャッシュ) を使うか
    fast_read: bool = True

    #: 直近の run_md が書いたトラジェクトリ
    trajectory_path: Path | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.atoms, Atoms):
            self.atoms = StructureHandler.read(self.atoms, fast=self.fast_read)
        else:
            self.atoms = self.atoms.copy()
        StructureHandler.check_size(self.atoms, max_atoms=self.max_atoms)
        self.workdir = Path(self.workdir).expanduser().resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        if self.calculator is None:
            if self.model is None:
                raise ValueError("model か calculator のどちらかが必要です。")
            self.calculator = create_nep_calculator(
                self.model,
                backend=self.backend,
                atoms=self.atoms,
                environment=self.environment,
                **self.calculator_kwargs,
            )
        self.atoms.calc = self.calculator
        self.log: list[dict[str, float]] = []

    # ------------------------------------------------------------------ 1 点計算
    def single_point(self) -> dict[str, Any]:
        """エネルギー・力・応力・per-atom エネルギーをまとめて返す。"""
        energy = float(self.atoms.get_potential_energy())
        forces = self.atoms.get_forces()
        result: dict[str, Any] = {
            "energy": energy,
            "energy_per_atom": energy / len(self.atoms),
            "forces": forces,
            "fmax": float(np.abs(forces).max()),
            "n_atoms": len(self.atoms),
        }
        try:
            result["stress_GPa"] = self.atoms.get_stress(voigt=True) / units.GPa
            result["pressure_GPa"] = float(-result["stress_GPa"][:3].mean())
        except Exception:
            pass
        try:
            energies = np.asarray(self.atoms.get_potential_energies())
            result["energies"] = energies
            result["energy_per_atom_min"] = float(energies.min())
            result["energy_per_atom_max"] = float(energies.max())
        except Exception as exc:  # backend が per-atom を返せない場合
            warnings.warn(f"per-atom エネルギーを取得できませんでした: {exc}", RuntimeWarning)
        return result

    def relax(
        self,
        *,
        fmax: float = 0.01,
        steps: int = 500,
        relax_cell: bool = False,
        optimizer: str = "FIRE",
    ) -> Atoms:
        """構造最適化 (必要ならセルも)。"""
        from ase.optimize import BFGS, FIRE

        optimizers = {"FIRE": FIRE, "BFGS": BFGS}
        if optimizer not in optimizers:
            raise ValueError(f"optimizer は {list(optimizers)} から選びます。")
        target: Any = self.atoms
        if relax_cell:
            from ase.filters import FrechetCellFilter

            target = FrechetCellFilter(self.atoms)
        dyn = optimizers[optimizer](target, logfile=str(self.workdir / "relax.log"))
        dyn.run(fmax=fmax, steps=steps)
        return self.atoms

    # ------------------------------------------------------------------ MD
    def make_dynamics(
        self,
        *,
        ensemble: str = "nvt",
        temperature: float = 300.0,
        time_step: float = 1.0,
        friction: float = 0.01,
        pressure_GPa: float | Sequence[float] = 0.0,
        taut: float = 100.0,
        taup: float = 1000.0,
        ttime: float = 25.0,
        ptime: float = 1000.0,
        pfactor: float | None = None,
        bulk_modulus_GPa: float = 100.0,
        compressibility_au: float | None = None,
        npt_axes: Sequence[str] | None = None,
        mask: Sequence[int] | np.ndarray | None = None,
        trajectory: str | None = "md.xyz",
        seed: int | None = None,
    ):
        """ASE の積分器 (dynamics) を組み立てて返す。

        時定数はすべて **fs** で指定する (mission.md 追加依頼 20260921-2)。

        Parameters
        ----------
        ensemble
            ``'nve'``
            / ``'nvt'`` (Langevin) / ``'nvt_berendsen'`` / ``'nvt_bussi'``
            / ``'nvt_nose_hoover'``
            / ``'npt'`` = ``'npt_berendsen'`` / ``'npt_inhomogeneous'``
            / ``'npt_parrinello_rahman'`` (= ``'npt_mttk'``)
        friction
            Langevin の摩擦係数 [1/fs]。
        taut, taup
            Berendsen 系の温度 / 圧力の時定数 [fs] (ASE の ``taut`` / ``taup``)。
        ttime, ptime
            Nose-Hoover / Parrinello-Rahman の特性時間 [fs]。
            ``pfactor = ptime**2 * bulk_modulus`` として ASE の ``pfactor``
            を組み立てる (``pfactor`` を直接与えればそちらが優先)。
        bulk_modulus_GPa
            体積弾性率の概算値 [GPa]。``pfactor`` と Berendsen の
            圧縮率 (``compressibility_au = 1/B``) に使う。
        npt_axes, mask
            セルのどの方向を動かすか (追加依頼 20260921-4)。
            ``npt_axes=('z',)`` なら c 軸だけ、``('x','y','z')`` なら 3 軸独立。
            ``mask`` を直接 ASE の形式で渡すこともできる。
        """
        from ase.md.bussi import Bussi
        from ase.md.langevin import Langevin
        from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen, NPTBerendsen
        from ase.md.nvtberendsen import NVTBerendsen
        from ase.md.verlet import VelocityVerlet

        try:  # ase >= 3.29 では NPT -> MelchionnaNPT にリネームされている
            from ase.md.melchionna import MelchionnaNPT as NPT
        except ImportError:
            from ase.md.npt import NPT

        dt = time_step * units.fs
        # ASE の dynamics(trajectory=...) は .traj (ASE 専用バイナリ) しか書けない。
        # 拡張 XYZ で残したい場合は run_md 側でライタを attach するので、
        # ここでは .traj を指定されたときだけ ASE に任せる。
        self._trajectory_file = str(self.workdir / trajectory) if trajectory else None
        traj_path = (
            self._trajectory_file
            if self._trajectory_file and self._trajectory_file.endswith(".traj")
            else None
        )
        if mask is None and npt_axes is not None:
            mask = axes_to_mask(npt_axes)
        if compressibility_au is None:
            compressibility_au = 1.0 / (bulk_modulus_GPa * units.GPa)
        if pfactor is None:
            pfactor = (ptime * units.fs) ** 2 * bulk_modulus_GPa * units.GPa

        if ensemble == "nve":
            return VelocityVerlet(self.atoms, dt, trajectory=traj_path)
        if ensemble == "nvt":
            return Langevin(
                self.atoms,
                dt,
                temperature_K=temperature,
                friction=friction / units.fs,
                trajectory=traj_path,
                rng=np.random.default_rng(seed),
            )
        if ensemble == "nvt_berendsen":
            return NVTBerendsen(
                self.atoms,
                dt,
                temperature_K=temperature,
                taut=taut * units.fs,
                trajectory=traj_path,
            )
        if ensemble == "nvt_bussi":
            return Bussi(
                self.atoms,
                dt,
                temperature_K=temperature,
                taut=taut * units.fs,
                rng=np.random.default_rng(seed),
                trajectory=traj_path,
            )
        if ensemble == "nvt_nose_hoover":
            self._require_triangular_cell()
            return NPT(
                self.atoms,
                dt,
                temperature_K=temperature,
                externalstress=None,
                ttime=ttime * units.fs,
                pfactor=None,
                trajectory=traj_path,
            )
        if ensemble in ("npt", "npt_berendsen"):
            return NPTBerendsen(
                self.atoms,
                dt,
                temperature_K=temperature,
                pressure_au=_hydrostatic(pressure_GPa) * units.GPa,
                taut=taut * units.fs,
                taup=taup * units.fs,
                compressibility_au=compressibility_au,
                trajectory=traj_path,
            )
        if ensemble == "npt_inhomogeneous":
            return Inhomogeneous_NPTBerendsen(
                self.atoms,
                dt,
                temperature_K=temperature,
                pressure_au=_hydrostatic(pressure_GPa) * units.GPa,
                taut=taut * units.fs,
                taup=taup * units.fs,
                compressibility_au=compressibility_au,
                mask=tuple(mask) if mask is not None else (1, 1, 1),
                trajectory=traj_path,
            )
        if ensemble in ("npt_parrinello_rahman", "npt_mttk"):
            self._require_triangular_cell()
            return NPT(
                self.atoms,
                dt,
                temperature_K=temperature,
                # ASE の externalstress は「応力」なので圧力とは符号が逆
                externalstress=-_stress_vector(pressure_GPa) * units.GPa,
                ttime=ttime * units.fs,
                pfactor=pfactor,
                mask=np.asarray(mask) if mask is not None else None,
                trajectory=traj_path,
            )
        raise ValueError(
            "ensemble は 'nve' / 'nvt' / 'nvt_berendsen' / 'nvt_bussi' /"
            " 'nvt_nose_hoover' / 'npt' / 'npt_inhomogeneous' /"
            " 'npt_parrinello_rahman' から選びます。"
        )

    def _require_triangular_cell(self) -> None:
        """ASE の ``NPT`` が扱える三角行列のセルにする。

        ASE の判定は ``m[1,0] == m[2,0] == m[2,1] == 0.0`` という**厳密な比較**
        なので、CIF 往復などで残った 1e-16 のゴミでも弾かれる。まず丸め誤差を
        落とし、それでも三角行列でなければ標準形へ剛体回転する。
        """
        cleaned = StructureHandler.clean_cell(self.atoms)
        rotated_needed = not StructureHandler.is_triangular(cleaned, tol=0.0)
        if rotated_needed:
            cleaned = StructureHandler.clean_cell(
                StructureHandler.to_standard_cell(cleaned)
            )
        self.atoms.set_cell(cleaned.cell)
        self.atoms.set_positions(cleaned.get_positions())
        velocities = cleaned.get_velocities()
        if velocities is not None:
            self.atoms.set_velocities(velocities)
        if rotated_needed:
            warnings.warn(
                "ASE の NPT は三角行列のセルしか扱えないため、セルを標準形へ回転しました"
                " (原子の相対配置は不変)。",
                RuntimeWarning,
            )

    def run_md(
        self,
        *,
        ensemble: str = "nvt",
        temperature: float = 300.0,
        temperature_end: float | None = None,
        steps: int = 1000,
        time_step: float = 1.0,
        log_interval: int = 10,
        trajectory: str | None = "md.xyz",
        seed: int | None = None,
        initialize_velocities: bool = True,
        **dynamics_kwargs,
    ) -> Atoms:
        """ASE の積分器で MD を回す。

        ``ensemble`` や熱浴・圧浴の時定数 (``taut`` / ``taup`` / ``ttime`` /
        ``ptime`` / ``pfactor``)、変調する軸 (``npt_axes``) は
        :meth:`make_dynamics` に渡される。

        Parameters
        ----------
        temperature, temperature_end
            目標温度 [K]。``temperature_end`` を与えると線形に昇温/降温する。
        time_step
            時間刻み [fs]。

        Examples
        --------
        >>> runner.run_md(ensemble="nvt_berendsen", temperature=300, taut=200)
        >>> runner.run_md(ensemble="npt", temperature=300, pressure_GPa=0.0,
        ...               taut=100, taup=1000, bulk_modulus_GPa=140)
        >>> runner.run_md(ensemble="npt_parrinello_rahman", temperature=300,
        ...               ptime=2000, npt_axes=("z",))   # c 軸だけ動かす
        """
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

        if initialize_velocities:
            rng = np.random.default_rng(seed)
            MaxwellBoltzmannDistribution(self.atoms, temperature_K=temperature, rng=rng)
            Stationary(self.atoms)

        dyn = self.make_dynamics(
            ensemble=ensemble,
            temperature=temperature,
            time_step=time_step,
            trajectory=trajectory,
            seed=seed,
            **dynamics_kwargs,
        )

        self.log = []
        n_atoms = len(self.atoms)
        self.trajectory_path: Path | None = (
            Path(self._trajectory_file) if self._trajectory_file else None
        )
        if self.trajectory_path and self.trajectory_path.suffix != ".traj":
            # 拡張 XYZ に追記していく (OVITO / VESTA / ASE からそのまま開ける)
            self.trajectory_path.unlink(missing_ok=True)

            def write_frame() -> None:
                from ase.io import write as ase_write

                ase_write(
                    str(self.trajectory_path), self.atoms, format="extxyz", append=True
                )

            dyn.attach(write_frame, interval=log_interval)

        def record() -> None:
            atoms = self.atoms
            ekin = atoms.get_kinetic_energy()
            epot = atoms.get_potential_energy()
            entry = {
                "step": dyn.get_number_of_steps(),
                "time_ps": dyn.get_number_of_steps() * time_step * 1e-3,
                "temperature": ekin / (1.5 * n_atoms * units.kB),
                "potential_energy_per_atom": epot / n_atoms,
                "kinetic_energy_per_atom": ekin / n_atoms,
                "total_energy_per_atom": (epot + ekin) / n_atoms,
                "volume": atoms.get_volume(),
            }
            try:
                stress = atoms.get_stress(voigt=True) / units.GPa
                entry["pressure"] = float(-stress[:3].mean())
            except Exception:
                entry["pressure"] = float("nan")
            self.log.append(entry)

        dyn.attach(record, interval=log_interval)

        if temperature_end is None or abs(temperature_end - temperature) < 1e-9:
            dyn.run(steps)
        else:  # 温度ランプ: 目標温度を小刻みに更新する
            n_blocks = min(100, max(1, steps // 10))
            block = max(1, steps // n_blocks)
            temperatures = np.linspace(temperature, temperature_end, n_blocks)
            done = 0
            for target_T in temperatures:
                if hasattr(dyn, "set_temperature"):
                    dyn.set_temperature(temperature_K=float(target_T))
                remaining = min(block, steps - done)
                if remaining <= 0:
                    break
                dyn.run(remaining)
                done += remaining
        return self.atoms

    # ------------------------------------------------------------------ 出力
    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.log)

    def save_log(self, filename: str = "ase_md.csv") -> Path:
        path = self.workdir / filename
        self.to_dataframe().to_csv(path, index=False)
        return path

    def plot(self, filename: str = "ase_md.png", dpi: int = 150) -> Path:
        """MD ログ (温度・エネルギー・圧力・体積) を 1 枚の図にする。"""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        frame = self.to_dataframe()
        if frame.empty:
            raise ValueError("ログが空です。run_md() を先に実行してください。")
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        specs = [
            (axes[0, 0], "temperature", "temperature (K)", "tab:red"),
            (axes[0, 1], "total_energy_per_atom", "E$_{tot}$ (eV/atom)", "tab:blue"),
            (axes[1, 0], "pressure", "pressure (GPa)", "tab:green"),
            (axes[1, 1], "volume", "volume (Å$^3$)", "tab:purple"),
        ]
        for ax, column, ylabel, color in specs:
            ax.plot(frame["time_ps"], frame[column], lw=1.0, color=color)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
        for ax in axes[1]:
            ax.set_xlabel("time (ps)")
        fig.suptitle(f"ASE MD ({self.backend} backend): {self.workdir.name}")
        fig.tight_layout()
        path = self.workdir / filename
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        return path

    def write_trajectory(self, output: Path | str, *, fmt: str | None = None, **kwargs) -> Path:
        """MD で書いたトラジェクトリを XDATCAR などへ変換する。

        ``run_md(trajectory=...)`` の既定は ``md.xyz`` (拡張 XYZ) なので、
        そのままでも OVITO / VESTA / ASE から開ける。
        XDATCAR や POSCAR が要るときだけこれを使う。
        """
        from .trajectory import TrajectoryConverter

        source = getattr(self, "trajectory_path", None) or (self.workdir / "md.xyz")
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(
                f"{source} がありません。run_md(trajectory=...) を有効にして実行してください。"
            )
        converter = TrajectoryConverter(
            source, format="traj" if source.suffix == ".traj" else "extxyz"
        )
        return converter.convert(output, fmt=fmt, **kwargs)

    def trajectory(self):
        """書き出したトラジェクトリの :class:`~gpumd_toolkit.trajectory.TrajectoryConverter`。"""
        from .trajectory import TrajectoryConverter

        source = Path(getattr(self, "trajectory_path", None) or (self.workdir / "md.xyz"))
        return TrajectoryConverter(
            source, format="traj" if source.suffix == ".traj" else "extxyz"
        )
