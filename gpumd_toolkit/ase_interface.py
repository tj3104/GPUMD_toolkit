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
from typing import Any

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes

from .config import GPUMDEnvironment
from .structure import DEFAULT_MAX_ATOMS, StructureHandler

__all__ = [
    "create_nep_calculator",
    "available_backends",
    "CPUNEPWithEnergies",
    "GPUNEPWithEnergies",
    "ASEMDRunner",
]

BACKENDS = ("cpu", "pynep", "gpu")


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
    model: Path | str
    backend: str = "auto"
    workdir: Path | str = "ase_run"
    max_atoms: int = DEFAULT_MAX_ATOMS
    environment: GPUMDEnvironment | None = None
    calculator_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.atoms, Atoms):
            self.atoms = StructureHandler.read(self.atoms)
        else:
            self.atoms = self.atoms.copy()
        StructureHandler.check_size(self.atoms, max_atoms=self.max_atoms)
        self.workdir = Path(self.workdir).expanduser().resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
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
    def run_md(
        self,
        *,
        ensemble: str = "nvt",
        temperature: float = 300.0,
        temperature_end: float | None = None,
        steps: int = 1000,
        time_step: float = 1.0,
        friction: float = 0.01,
        pressure_GPa: float = 0.0,
        ttime: float = 25.0,
        pfactor: float = 2e6,
        log_interval: int = 10,
        trajectory: str | None = "md.traj",
        seed: int | None = None,
        initialize_velocities: bool = True,
    ) -> Atoms:
        """ASE の積分器で MD を回す。

        Parameters
        ----------
        ensemble
            ``'nve'`` / ``'nvt'`` (Langevin) / ``'nvt_nose_hoover'`` / ``'npt'``。
        temperature, temperature_end
            目標温度 [K]。``temperature_end`` を与えると線形に昇温/降温する。
        time_step
            時間刻み [fs]。
        friction
            Langevin の摩擦係数 [1/fs]。
        pressure_GPa
            NPT の目標圧力 [GPa]。
        """
        from ase.md.langevin import Langevin
        from ase.md.npt import NPT
        from ase.md.nptberendsen import NPTBerendsen
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
        from ase.md.verlet import VelocityVerlet

        if initialize_velocities:
            rng = np.random.default_rng(seed)
            MaxwellBoltzmannDistribution(self.atoms, temperature_K=temperature, rng=rng)
            Stationary(self.atoms)

        dt = time_step * units.fs
        traj_path = str(self.workdir / trajectory) if trajectory else None

        if ensemble == "nve":
            dyn = VelocityVerlet(self.atoms, dt, trajectory=traj_path)
        elif ensemble == "nvt":
            dyn = Langevin(
                self.atoms,
                dt,
                temperature_K=temperature,
                friction=friction / units.fs,
                trajectory=traj_path,
                rng=np.random.default_rng(seed),
            )
        elif ensemble == "nvt_nose_hoover":
            dyn = NPT(
                self.atoms,
                dt,
                temperature_K=temperature,
                externalstress=None,
                ttime=ttime * units.fs,
                pfactor=None,
                trajectory=traj_path,
            )
        elif ensemble == "npt":
            dyn = NPTBerendsen(
                self.atoms,
                dt,
                temperature_K=temperature,
                pressure_au=pressure_GPa * units.GPa,
                taut=100 * units.fs,
                taup=1000 * units.fs,
                compressibility_au=4.57e-5 / units.bar,
                trajectory=traj_path,
            )
        else:
            raise ValueError(
                "ensemble は 'nve' / 'nvt' / 'nvt_nose_hoover' / 'npt' から選びます。"
            )

        self.log = []
        n_atoms = len(self.atoms)

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
        """ASE の ``.traj`` を XDATCAR などへ変換する。"""
        from .trajectory import TrajectoryConverter

        source = self.workdir / "md.traj"
        if not source.is_file():
            raise FileNotFoundError(f"{source} がありません。trajectory= を有効にして実行してください。")
        converter = TrajectoryConverter(source, format="traj")
        return converter.convert(output, fmt=fmt, **kwargs)
