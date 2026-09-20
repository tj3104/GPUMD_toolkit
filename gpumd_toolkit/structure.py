"""構造ファイルの入出力と GPUMD 用 ``model.xyz`` の作成。

POSCAR / CIF を中心に、ASE が読める形式はひととおり扱える。
``model.xyz`` の書き出しは calorine (``calorine.gpumd.write_xyz``) を用いる。
ASE の extxyz ライタは速度の単位変換を行わないため、calorine 経由が推奨。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from ase import Atoms
from ase.build import make_supercell
from ase.io import read as ase_read
from ase.io import write as ase_write

__all__ = ["StructureHandler", "StructureInfo", "DEFAULT_MAX_ATOMS"]

#: mission.md の指示: 原子数は必要最低限 (多くとも 1 万程度)
DEFAULT_MAX_ATOMS = 10_000

#: 拡張子 -> ASE フォーマット名 (拡張子から自明でないものだけ)
_SUFFIX_FORMATS = {
    ".cif": "cif",
    ".xyz": "extxyz",
    ".extxyz": "extxyz",
    ".traj": "traj",
    ".vasp": "vasp",
    ".cell": "castep-cell",
    ".pdb": "proteindatabank",
    ".json": "json",
    ".lmp": "lammps-data",
    ".data": "lammps-data",
    ".cfg": "cfg",
    ".gen": "gen",
    ".res": "res",
}

#: 拡張子を持たない VASP 系ファイル名
_VASP_NAMES = {"poscar", "contcar", "chgcar", "xdatcar"}


@dataclass
class StructureInfo:
    """構造の要約 (ログ・レポート用)。"""

    formula: str
    n_atoms: int
    species: dict[str, int]
    cell_lengths: tuple[float, float, float]
    cell_angles: tuple[float, float, float]
    volume: float
    density: float  # g/cm^3
    pbc: tuple[bool, bool, bool]
    orthorhombic: bool

    def as_dict(self) -> dict:
        return {
            "formula": self.formula,
            "n_atoms": self.n_atoms,
            "species": self.species,
            "a": self.cell_lengths[0],
            "b": self.cell_lengths[1],
            "c": self.cell_lengths[2],
            "alpha": self.cell_angles[0],
            "beta": self.cell_angles[1],
            "gamma": self.cell_angles[2],
            "volume": self.volume,
            "density_g_cm3": self.density,
            "pbc": self.pbc,
            "orthorhombic": self.orthorhombic,
        }

    def __str__(self) -> str:
        a, b, c = self.cell_lengths
        al, be, ga = self.cell_angles
        comp = " ".join(f"{k}{v}" for k, v in sorted(self.species.items()))
        return (
            f"{self.formula} ({comp}) N={self.n_atoms}  "
            f"cell=({a:.3f}, {b:.3f}, {c:.3f}) Å / "
            f"({al:.1f}, {be:.1f}, {ga:.1f})°  "
            f"V={self.volume:.1f} Å³  ρ={self.density:.3f} g/cm³  "
            f"{'orthorhombic' if self.orthorhombic else 'triclinic'}"
        )


class StructureHandler:
    """構造ファイルの読み込み・整形・``model.xyz`` 書き出しを担当する。

    Examples
    --------
    >>> atoms = StructureHandler.read("POSCAR")
    >>> atoms = StructureHandler.make_cell(atoms, min_length=12.0, max_atoms=10000)
    >>> StructureHandler.write_model(atoms, "runs/foo/model.xyz")
    """

    # ------------------------------------------------------------------ 読み込み
    @staticmethod
    def detect_format(path: Path | str) -> str | None:
        """ファイル名から ASE のフォーマット名を推定する。"""
        path = Path(path)
        stem = path.name.lower()
        if stem.split(".")[0] in _VASP_NAMES or stem in _VASP_NAMES:
            return "vasp-xdatcar" if stem.startswith("xdatcar") else "vasp"
        return _SUFFIX_FORMATS.get(path.suffix.lower())

    @staticmethod
    def read(
        path: Path | str,
        *,
        format: str | None = None,
        index: int | str = -1,
    ) -> Atoms:
        """構造ファイルを読み込む (POSCAR / CIF / xyz / traj / ...)。

        GPUMD の ``model.xyz`` も ``extxyz`` として読めるが、速度を含む場合は
        :meth:`read_model` を使うこと (単位変換が必要なため)。
        """
        path = Path(path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"構造ファイルが見つかりません: {path}")
        fmt = format or StructureHandler.detect_format(path)
        atoms = ase_read(str(path), format=fmt, index=index)
        if isinstance(atoms, list):  # index=':' が渡された場合
            atoms = atoms[-1]
        if not atoms.cell.rank == 3:
            raise ValueError(
                f"{path} には 3 次元のセルがありません。GPUMD は周期セルを必要とします。"
            )
        atoms.set_pbc(True)
        return atoms

    @staticmethod
    def read_model(path: Path | str) -> Atoms:
        """GPUMD の ``model.xyz`` を(速度の単位変換つきで)読み込む。"""
        from calorine.gpumd import read_xyz

        return read_xyz(str(path))

    # ------------------------------------------------------------------ セル整形
    @staticmethod
    def is_orthorhombic(atoms: Atoms, tol: float = 1e-6) -> bool:
        cell = np.asarray(atoms.cell)
        off = cell - np.diag(np.diag(cell))
        return bool(np.abs(off).max() < tol)

    @staticmethod
    def to_orthorhombic(atoms: Atoms, max_atoms: int = DEFAULT_MAX_ATOMS) -> Atoms:
        """直方晶セルに変換する (NPT の等方/直方セル制御は直交セルが必要)。"""
        if StructureHandler.is_orthorhombic(atoms):
            return atoms
        from ase.build import find_optimal_cell_shape, make_supercell

        n_target = max(1, min(8, max_atoms // max(1, len(atoms))))
        P = find_optimal_cell_shape(atoms.cell, n_target, "sc")
        new = make_supercell(atoms, P)
        if not StructureHandler.is_orthorhombic(new, tol=1e-4):
            warnings.warn(
                "厳密な直方晶セルを作れませんでした。NPT では npt_mttk の tri 指定か "
                "NVT の使用を検討してください。",
                RuntimeWarning,
            )
        return new

    @staticmethod
    def suggest_repeat(
        atoms: Atoms,
        *,
        min_length: float = 12.0,
        max_atoms: int = DEFAULT_MAX_ATOMS,
    ) -> tuple[int, int, int]:
        """各セル長が ``min_length`` 以上になる最小の繰り返し数を返す。

        NEP のカットオフ(既定 6 Å 程度)の 2 倍以上のセル長が必要なので
        ``min_length`` は 12 Å 程度を既定とする。
        ``max_atoms`` を超える場合は超えない範囲まで縮小する。
        """
        lengths = np.asarray(atoms.cell.cellpar()[:3], dtype=float)
        repeat = np.maximum(1, np.ceil(min_length / lengths)).astype(int)
        while int(np.prod(repeat)) * len(atoms) > max_atoms and repeat.max() > 1:
            repeat[int(np.argmax(repeat * lengths))] -= 1
            repeat = np.maximum(repeat, 1)
        return tuple(int(x) for x in repeat)

    @staticmethod
    def make_cell(
        atoms: Atoms,
        *,
        repeat: Sequence[int] | int | None = None,
        min_length: float | None = None,
        max_atoms: int = DEFAULT_MAX_ATOMS,
        orthorhombic: bool = False,
    ) -> Atoms:
        """スーパーセルを作成する。

        ``repeat`` を直接与えるか、``min_length`` から自動決定する。
        どちらも ``None`` なら元の構造をそのまま返す。
        """
        atoms = atoms.copy()
        if orthorhombic:
            atoms = StructureHandler.to_orthorhombic(atoms, max_atoms=max_atoms)
        if repeat is None and min_length is not None:
            repeat = StructureHandler.suggest_repeat(
                atoms, min_length=min_length, max_atoms=max_atoms
            )
        if repeat is not None:
            if isinstance(repeat, int):
                repeat = (repeat, repeat, repeat)
            atoms = atoms.repeat(tuple(int(r) for r in repeat))
        StructureHandler.check_size(atoms, max_atoms=max_atoms)
        return atoms

    @staticmethod
    def check_size(atoms: Atoms, max_atoms: int = DEFAULT_MAX_ATOMS) -> None:
        """原子数の上限チェック (mission.md: 多くとも 1 万程度)。"""
        if len(atoms) > max_atoms:
            raise ValueError(
                f"原子数が上限を超えています: {len(atoms)} > {max_atoms}。"
                " max_atoms を明示的に引き上げるか、スーパーセルを小さくしてください。"
            )

    # ------------------------------------------------------------------ 速度など
    @staticmethod
    def set_initial_velocities(
        atoms: Atoms, temperature: float, *, seed: int | None = None
    ) -> Atoms:
        """Maxwell-Boltzmann 分布で初速を与え、全体の運動量をゼロにする。

        GPUMD の ``velocity`` キーワードでも初速は設定できるが、
        ASE 側で決定論的に与えたい場合に使う。
        """
        from ase.md.velocitydistribution import (
            MaxwellBoltzmannDistribution,
            Stationary,
            ZeroRotation,
        )

        atoms = atoms.copy()
        rng = np.random.default_rng(seed)
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=rng)
        Stationary(atoms)
        if not any(atoms.pbc):
            ZeroRotation(atoms)
        return atoms

    # ------------------------------------------------------------------ 書き出し
    @staticmethod
    def write_model(
        atoms: Atoms,
        path: Path | str,
        *,
        groupings: list[list[list[int]]] | None = None,
    ) -> Path:
        """GPUMD 入力 ``model.xyz`` を書き出す。

        ``groupings`` は grouping method のリスト。
        ``groupings[m][g]`` が grouping method ``m`` のグループ ``g`` に
        属する原子インデックスのリスト。
        """
        from calorine.gpumd import write_xyz

        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        write_xyz(str(path), atoms, groupings=groupings)
        return path

    @staticmethod
    def write(atoms: Atoms, path: Path | str, *, format: str | None = None) -> Path:
        """任意フォーマットで書き出す (POSCAR / CIF / traj など)。"""
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        fmt = format or StructureHandler.detect_format(path)
        ase_write(str(path), atoms, format=fmt)
        return path

    # ------------------------------------------------------------------ 情報取得
    @staticmethod
    def info(atoms: Atoms) -> StructureInfo:
        cellpar = atoms.cell.cellpar()
        volume = float(atoms.get_volume())
        mass = float(atoms.get_masses().sum())
        species: dict[str, int] = {}
        for symbol in atoms.get_chemical_symbols():
            species[symbol] = species.get(symbol, 0) + 1
        return StructureInfo(
            formula=atoms.get_chemical_formula(mode="hill"),
            n_atoms=len(atoms),
            species=species,
            cell_lengths=(float(cellpar[0]), float(cellpar[1]), float(cellpar[2])),
            cell_angles=(float(cellpar[3]), float(cellpar[4]), float(cellpar[5])),
            volume=volume,
            # 1 amu/Å^3 = 1.66053906660 g/cm^3
            density=mass / volume * 1.66053906660,
            pbc=tuple(bool(p) for p in atoms.pbc),
            orthorhombic=StructureHandler.is_orthorhombic(atoms),
        )

    @staticmethod
    def unique_species(atoms: Atoms) -> list[str]:
        seen: list[str] = []
        for symbol in atoms.get_chemical_symbols():
            if symbol not in seen:
                seen.append(symbol)
        return seen
