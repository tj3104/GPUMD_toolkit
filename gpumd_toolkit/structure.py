"""構造ファイルの入出力と GPUMD 用 ``model.xyz`` の作成。

POSCAR / CIF を中心に、ASE が読める形式はひととおり扱える。
``model.xyz`` の書き出しは calorine (``calorine.gpumd.write_xyz``) を用いる。
ASE の extxyz ライタは速度の単位変換を行わないため、calorine 経由が推奨。

読み込みは既定で :mod:`gpumd_toolkit.fastio` の高速経路を通る
(POSCAR/extxyz は専用パーサ、CIF は pymatgen + xyz キャッシュ)。
``StructureHandler.read(..., fast=False)`` で従来どおり ``ase.io.read`` になる。

セル形状については GPUMD 側が一般の三斜晶 (9 成分の ``lattice``) を受け付ける
ので、本モジュールも直方晶に限定しない。ただし

* 近接リストの都合で効くのは辺の長さではなく **面間距離 (thickness)**
* ``npt_ber`` / ``npt_scr`` は三斜晶なら 6 成分の圧力指定が必須
* ASE の ``NPT`` (Parrinello-Rahman) は三角行列のセルしか受け付けない

という違いがあるため、:func:`StructureHandler.cell_thickness` /
:func:`StructureHandler.to_standard_cell` を用意している。
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

from . import fastio

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
    #: 面間距離 (volume / 面積)。GPUMD が近接リストに使うのはこちら。
    thickness: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: 'cubic' / 'tetragonal' / 'orthorhombic' / 'hexagonal' / 'monoclinic' / 'triclinic'
    cell_shape: str = "triclinic"
    #: セルが三角行列か (ASE の NPT / LAMMPS が要求する標準形)
    triangular: bool = False

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
            "thickness": self.thickness,
            "cell_shape": self.cell_shape,
            "triangular": self.triangular,
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
            f"{self.cell_shape}  "
            f"thickness=({self.thickness[0]:.2f}, {self.thickness[1]:.2f}, "
            f"{self.thickness[2]:.2f}) Å"
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
        fast: bool = True,
        cache: bool = True,
        cache_dir: Path | str | None = None,
    ) -> Atoms:
        """構造ファイルを読み込む (POSCAR / CIF / xyz / traj / ...)。

        既定では :func:`gpumd_toolkit.fastio.read_fast` を通る。

        * POSCAR / CONTCAR / extxyz … numpy だけの専用パーサ
        * CIF … pymatgen (ase の約 1/15 の時間) + extxyz キャッシュ
        * それ以外 … ``ase.io.read`` (キャッシュ対象なら extxyz に変換して再利用)

        ``fast=False`` で従来どおり ``ase.io.read`` のみ、
        ``cache=False`` で xyz キャッシュを使わない。

        GPUMD の ``model.xyz`` も ``extxyz`` として読めるが、速度を含む場合は
        :meth:`read_model` を使うこと (単位変換が必要なため)。
        """
        path = Path(path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"構造ファイルが見つかりません: {path}")
        if fast:
            atoms = fastio.read_fast(
                path, format=format, index=index, cache=cache, cache_dir=cache_dir
            )
        else:
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
    def to_xyz(
        source: Path | str, output: Path | str | None = None, **kwargs
    ) -> Path:
        """構造ファイルを extxyz に変換する (``cif2xyz`` / ``POSCAR2xyz``)。

        巨大な CIF を何度も読むなら、一度これで xyz にしておくと速い。
        """
        return fastio.convert_to_xyz(source, output, **kwargs)

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
    def is_triangular(atoms: Atoms, tol: float = 1e-10) -> bool:
        """セルが三角行列 (上三角 or 下三角) か。

        ASE の ``NPT`` (Parrinello-Rahman) はこの形のセルしか受け付けない。
        LAMMPS 系の標準形は下三角 ``a=(ax,0,0), b=(bx,by,0), c=(cx,cy,cz)``。
        """
        cell = np.asarray(atoms.cell)
        upper = abs(cell[1, 0]) <= tol and abs(cell[2, 0]) <= tol and abs(cell[2, 1]) <= tol
        lower = abs(cell[0, 1]) <= tol and abs(cell[0, 2]) <= tol and abs(cell[1, 2]) <= tol
        return bool(upper or lower)

    @staticmethod
    def clean_cell(atoms: Atoms, *, rtol: float = 1e-10) -> Atoms:
        """セル行列の丸め誤差レベルの成分を厳密な 0 にする。

        CIF や cellpar 経由の往復では、立方晶でも非対角成分に 1e-16 程度の
        ゴミが残る。GPUMD は ``box.cpu_h[1] != 0`` のように**厳密な 0 と比較**して
        直交セルかどうかを判定するため、これが残っていると

            Cannot use triclinic box with only 3 target pressure components.

        で止まってしまう。書き出す前にここで掃除しておく。
        """
        atoms = atoms.copy()
        cell = np.asarray(atoms.cell, dtype=float)
        scale = float(np.abs(cell).max())
        if scale > 0:
            cleaned = np.where(np.abs(cell) < scale * rtol, 0.0, cell)
            if not np.array_equal(cleaned, cell):
                scaled = atoms.get_scaled_positions(wrap=False)
                atoms.set_cell(cleaned)
                atoms.set_scaled_positions(scaled)
        return atoms

    @staticmethod
    def cell_thickness(atoms: Atoms) -> tuple[float, float, float]:
        """面間距離 (volume / 面積) を返す。

        GPUMD は近接リストの繰り返し数をこの厚みから決める
        (``thickness_x = V / |b x c|``)。斜めのセルでは辺の長さより
        こちらが小さくなるので、スーパーセルの判断はこちらで行う。
        """
        cell = np.asarray(atoms.cell)
        volume = abs(float(np.linalg.det(cell)))
        if volume <= 0:
            return (0.0, 0.0, 0.0)
        areas = (
            np.linalg.norm(np.cross(cell[1], cell[2])),
            np.linalg.norm(np.cross(cell[2], cell[0])),
            np.linalg.norm(np.cross(cell[0], cell[1])),
        )
        return tuple(float(volume / area) for area in areas)

    @staticmethod
    def classify_cell(atoms: Atoms, *, length_tol: float = 1e-4, angle_tol: float = 1e-2) -> str:
        """セル形状を ``cubic`` 〜 ``triclinic`` に分類する (ログ・判定用)。"""
        a, b, c, alpha, beta, gamma = (float(x) for x in atoms.cell.cellpar())
        lengths_equal = (
            abs(a - b) < length_tol and abs(b - c) < length_tol
        )
        two_equal = abs(a - b) < length_tol
        right = [abs(angle - 90.0) < angle_tol for angle in (alpha, beta, gamma)]
        if all(right):
            if lengths_equal:
                return "cubic"
            if two_equal or abs(b - c) < length_tol or abs(a - c) < length_tol:
                return "tetragonal"
            return "orthorhombic"
        if right[0] and right[1] and abs(gamma - 120.0) < angle_tol and two_equal:
            return "hexagonal"
        if sum(right) == 2:
            return "monoclinic"
        if abs(alpha - beta) < angle_tol and abs(beta - gamma) < angle_tol and lengths_equal:
            return "rhombohedral"
        return "triclinic"

    @staticmethod
    def to_standard_cell(atoms: Atoms) -> Atoms:
        """セルを下三角形 (LAMMPS 標準形) に回転する。

        原子の相対配置は変わらない (系全体を剛体回転するだけ)。
        ASE の ``NPT`` (Parrinello-Rahman) や LAMMPS 系の出力で必要。
        GPUMD は 9 成分の ``lattice`` を受け付けるので必須ではないが、
        揃えておくと XDATCAR などへの変換で扱いやすい。
        """
        if StructureHandler.is_triangular(atoms):
            return atoms
        atoms = atoms.copy()
        scaled = atoms.get_scaled_positions(wrap=False)
        velocities = atoms.get_velocities()
        new_cell, rotation = atoms.cell.standard_form()
        atoms.set_cell(new_cell)
        atoms.set_scaled_positions(scaled)
        if velocities is not None and np.any(velocities):
            atoms.set_velocities(velocities @ rotation.T)
        return atoms

    @staticmethod
    def reduce_cell(atoms: Atoms, *, method: str = "niggli") -> Atoms:
        """セルをできるだけ「立方体に近い」形に取り直す。

        斜めが極端なセル (面間距離が辺長よりずっと小さいもの) は、
        Niggli / Minkowski 簡約で同じ体積のまま厚みを稼げることが多い。
        原子数は変わらない。
        """
        atoms = atoms.copy()
        if method == "niggli":
            from ase.build import niggli_reduce

            niggli_reduce(atoms)
        elif method == "minkowski":
            from ase.geometry.minkowski_reduction import minkowski_reduce

            # 同じ格子を張り直すだけなので、デカルト座標はそのままでよい
            positions = atoms.get_positions()
            reduced, _ = minkowski_reduce(np.asarray(atoms.cell), pbc=atoms.pbc)
            atoms.set_cell(reduced)
            atoms.set_positions(positions)
            atoms.wrap()
        else:
            raise ValueError("method は 'niggli' か 'minkowski' です。")
        atoms.set_pbc(True)
        return atoms

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
        """各方向の **面間距離** が ``min_length`` 以上になる最小の繰り返し数を返す。

        NEP のカットオフ(既定 6 Å 程度)の 2 倍以上の厚みが必要なので
        ``min_length`` は 12 Å 程度を既定とする。
        ``max_atoms`` を超える場合は超えない範囲まで縮小する。

        判定に辺の長さではなく面間距離 (volume / 面積) を使うのは、斜めのセル
        では辺が長くても厚みが足りないことがあるため (GPUMD 本体も近接リスト
        の繰り返し数をこの厚みから決めている)。
        """
        thickness = np.asarray(StructureHandler.cell_thickness(atoms), dtype=float)
        thickness = np.where(thickness > 0, thickness, np.inf)
        repeat = np.maximum(1, np.ceil(min_length / thickness - 1e-9)).astype(int)
        while int(np.prod(repeat)) * len(atoms) > max_atoms and repeat.max() > 1:
            # いちばん余裕のある (厚みが稼げている) 方向から削る
            candidates = np.where(repeat > 1, repeat * thickness, -np.inf)
            repeat[int(np.argmax(candidates))] -= 1
        return tuple(int(x) for x in repeat)

    @staticmethod
    def make_cell(
        atoms: Atoms,
        *,
        repeat: Sequence[int] | int | None = None,
        min_length: float | None = None,
        max_atoms: int = DEFAULT_MAX_ATOMS,
        orthorhombic: bool = False,
        reduce: str | None = None,
        standard_cell: bool = False,
    ) -> Atoms:
        """スーパーセルを作成する。

        ``repeat`` を直接与えるか、``min_length`` から自動決定する。
        どちらも ``None`` なら元の構造をそのまま返す。

        ``reduce='niggli'`` を指定すると、スーパーセル化の前にセルを簡約して
        できるだけ立方体に近づける (斜めのセルで原子数を節約できる)。
        ``standard_cell=True`` は最後にセルを下三角形 (LAMMPS 標準形) へ回転する。
        """
        atoms = StructureHandler.clean_cell(atoms)
        if reduce:
            atoms = StructureHandler.reduce_cell(atoms, method=reduce)
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
        if standard_cell:
            atoms = StructureHandler.to_standard_cell(atoms)
        atoms = StructureHandler.clean_cell(atoms)
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
        fast: bool = False,
    ) -> Path:
        """GPUMD 入力 ``model.xyz`` を書き出す。

        ``groupings`` は grouping method のリスト。
        ``groupings[m][g]`` が grouping method ``m`` のグループ ``g`` に
        属する原子インデックスのリスト。

        ``fast=True`` にすると calorine を経由せず
        :func:`gpumd_toolkit.fastio.write_xyz_fast` で書く (grouping 非対応)。
        """
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        # GPUMD は厳密な 0 で直交セル判定をするので、丸め誤差を落としてから書く
        atoms = StructureHandler.clean_cell(atoms)
        if fast and not groupings:
            return fastio.write_xyz_fast(atoms, path)

        from calorine.gpumd import write_xyz

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
            thickness=StructureHandler.cell_thickness(atoms),
            cell_shape=StructureHandler.classify_cell(atoms),
            triangular=StructureHandler.is_triangular(atoms),
        )

    @staticmethod
    def nep_cutoffs(potential: Path | str) -> tuple[float, float] | None:
        """NEP ファイルのヘッダから ``(rc_radial, rc_angular)`` [Å] を読む。

        NEP 以外 (DP / Tersoff / EAM) や読めない場合は ``None`` を返す。
        """
        path = Path(potential).expanduser()
        if not path.is_file():
            return None
        try:
            with path.open() as handle:
                for _ in range(10):
                    line = handle.readline()
                    if not line:
                        break
                    tokens = line.split()
                    if tokens and tokens[0] == "cutoff" and len(tokens) >= 3:
                        return float(tokens[1]), float(tokens[2])
        except (OSError, ValueError):
            return None
        return None

    @staticmethod
    def check_nep_box(atoms: Atoms, rc_radial: float) -> dict:
        """GPUMD (NEP) がセルを受け付けるか確認する。

        GPUMD は周期方向のセル厚みが ``2.5 * (rc + 1)`` 以下だと内部で
        セルを複製して対処するが、**別の方向が ``10 * rc`` より厚い**
        場合は複製できず

            The box has a thickness < 2.5 radial cutoffs in a periodic
            direction and a thickness > 10 radial cutoffs in another direction.

        で異常終了する。細長いセル (NEMD・衝撃波) で踏みやすい。

        Returns
        -------
        dict
            ``ok`` が False のとき ``message`` に対処法が入る。
        """
        thickness = StructureHandler.cell_thickness(atoms)
        thin_limit = 2.5 * (rc_radial + 1.0)
        thick_limit = 10.0 * rc_radial
        thin = [
            (axis, value)
            for axis, value, periodic in zip("xyz", thickness, atoms.pbc)
            if periodic and value <= thin_limit
        ]
        thick = [
            (axis, value) for axis, value in zip("xyz", thickness) if value > thick_limit
        ]
        result = {
            "ok": True,
            "thickness": thickness,
            "thin_limit": thin_limit,
            "thick_limit": thick_limit,
            "thin_directions": thin,
            "thick_directions": thick,
            "message": "",
        }
        if thin and thick:
            result["ok"] = False
            result["message"] = (
                "GPUMD が受け付けないセル形状です"
                f" (NEP の radial cutoff {rc_radial:g} Å)。\n"
                f"  薄すぎる方向 (<= {thin_limit:.1f} Å): "
                + ", ".join(f"{a}={v:.1f} Å" for a, v in thin)
                + f"\n  厚すぎる方向 (> {thick_limit:.1f} Å): "
                + ", ".join(f"{a}={v:.1f} Å" for a, v in thick)
                + "\n  細い方向を厚くするか、長い方向を短くしてください。"
            )
        return result

    @staticmethod
    def unique_species(atoms: Atoms) -> list[str]:
        seen: list[str] = []
        for symbol in atoms.get_chemical_symbols():
            if symbol not in seen:
                seen.append(symbol)
        return seen
