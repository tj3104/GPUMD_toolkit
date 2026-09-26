"""``model.xyz`` の grouping method を組み立てるヘルパ。

GPUMD の多くの機能 (``fix`` / ``move`` / ``heat_*`` / ``ttm`` / ``compute`` /
``compute_msd all_groups`` / ``dump_xyz group`` / ``mc ... group``) は
「grouping method 番号 + グループ番号」で原子集合を指定する。
``model.xyz`` の ``group:I:<n>`` 列がその定義にあたる。

本モジュールは、その列を Python 側で安全に作るための道具を提供する。

Examples
--------
>>> from ase.build import bulk
>>> from gpumd_toolkit.groups import GroupingScheme, nemd_layout
>>> atoms = bulk("Si", "diamond", 5.43, cubic=True) * (12, 2, 2)
>>> layout = nemd_layout(atoms, axis="x", n_blocks=10)
>>> scheme = GroupingScheme.from_labels(layout.labels, name="nemd")
>>> scheme.to_groupings()[0][layout.source][:3]
[...]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from ase import Atoms

__all__ = [
    "GroupingScheme",
    "NEMDLayout",
    "uniform_slabs",
    "by_species",
    "by_region",
    "per_atom",
    "single_group",
    "by_molecule",
    "nemd_layout",
]

_AXES = {"x": 0, "y": 1, "z": 2}


def _axis_index(axis: str | int) -> int:
    if isinstance(axis, int):
        if axis not in (0, 1, 2):
            raise ValueError("axis は 0 (x) / 1 (y) / 2 (z) です。")
        return axis
    key = str(axis).strip().lower()
    if key not in _AXES:
        raise ValueError("axis は 'x' / 'y' / 'z' です。")
    return _AXES[key]


def _scaled_along(atoms: Atoms, axis: int) -> np.ndarray:
    """0 <= s < 1 の分率座標 (指定軸)。"""
    scaled = atoms.get_scaled_positions(wrap=True)
    return np.asarray(scaled[:, axis], dtype=float)


# ------------------------------------------------------------------ 基本形
def uniform_slabs(atoms: Atoms, axis: str | int = "x", n_slabs: int = 10) -> np.ndarray:
    """指定軸を等分したスラブでグループ分けし、原子ごとのラベルを返す。

    NEMD の温度プロファイルや ``compute`` の空間平均に使う。
    """
    if n_slabs < 1:
        raise ValueError("n_slabs は 1 以上です。")
    index = _axis_index(axis)
    fraction = _scaled_along(atoms, index)
    # 完全結晶では原子がちょうどビン境界に乗るため、分率座標の丸め誤差
    # (0.875 が 0.8749999... になる等) で隣のスラブに落ちてしまう。
    # 境界にわずかなマージンを入れて、同じ面の原子が必ず同じスラブに入るようにする。
    labels = np.floor(fraction * n_slabs + 1e-9).astype(int)
    return np.clip(labels, 0, n_slabs - 1)


def by_species(atoms: Atoms, order: Sequence[str] | None = None) -> np.ndarray:
    """元素ごとにグループ分けする (元素別 MSD・DOS・イオン伝導度に使う)。"""
    symbols = atoms.get_chemical_symbols()
    if order is None:
        order = []
        for symbol in symbols:
            if symbol not in order:
                order.append(symbol)
    mapping = {symbol: i for i, symbol in enumerate(order)}
    missing = sorted(set(symbols) - set(mapping))
    if missing:
        raise ValueError(f"order に含まれていない元素があります: {missing}")
    return np.array([mapping[s] for s in symbols], dtype=int)


def by_region(
    atoms: Atoms,
    boundaries: Sequence[float],
    axis: str | int = "x",
    *,
    fractional: bool = False,
) -> np.ndarray:
    """境界値のリストで軸方向を区切ってグループ分けする。

    ``boundaries=[10.0, 30.0]`` なら 3 グループ (< 10, 10-30, >= 30) になる。
    ``fractional=True`` なら境界を分率座標 (0-1) で与える。
    """
    index = _axis_index(axis)
    edges = np.asarray(sorted(float(b) for b in boundaries), dtype=float)
    if edges.size == 0:
        raise ValueError("boundaries を 1 つ以上指定してください。")
    if fractional:
        if edges.min() <= 0 or edges.max() >= 1:
            raise ValueError("fractional=True の境界は 0 < b < 1 です。")
        values = _scaled_along(atoms, index)
    else:
        values = np.asarray(atoms.get_positions()[:, index], dtype=float)
    return np.searchsorted(edges, values, side="right").astype(int)


def per_atom(atoms: Atoms) -> np.ndarray:
    """1 原子 1 グループ (per-atom DOS からフォノン参加率を出すときなど)。"""
    return np.arange(len(atoms), dtype=int)


def single_group(atoms: Atoms) -> np.ndarray:
    """全原子を 1 グループにまとめる。"""
    return np.zeros(len(atoms), dtype=int)


def by_molecule(atoms: Atoms, cutoff: float = 1.4) -> np.ndarray:
    """距離でつながった連結成分を 1 分子とみなしてグループ分けする。

    分子ごとの MSD (``compute_msd ... all_groups``) に使う。
    ``cutoff`` は結合とみなす原子間距離 [Å]。
    """
    if cutoff <= 0:
        raise ValueError("cutoff [Å] は正の値です。")
    n = len(atoms)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    from ase.neighborlist import neighbor_list

    first, second = neighbor_list("ij", atoms, cutoff)
    for i, j in zip(first, second):
        union(int(i), int(j))

    roots = {}
    labels = np.zeros(n, dtype=int)
    for i in range(n):
        root = find(i)
        if root not in roots:
            roots[root] = len(roots)
        labels[i] = roots[root]
    return labels


# ------------------------------------------------------------------ NEMD 配置
@dataclass
class NEMDLayout:
    """NEMD 熱伝導 (``heat_nhc`` / ``heat_bdp`` / ``heat_lan``) 用のグループ配置。

    ``[固定] [熱源] [中間ブロック ...] [熱浴] [固定]`` の順に軸方向へ並べる。

    Attributes
    ----------
    labels
        原子ごとのグループラベル。
    source, sink
        熱源・熱浴のグループ番号 (``ensemble heat_*`` に渡す)。
    middle
        中間ブロックのグループ番号のリスト (温度勾配を測る区間)。
    fixed
        固定するグループ番号のリスト (``fix`` に渡す)。空なら固定しない。
        GPUMD の ``fix`` は 1 run につき 1 グループしか効かないので、
        両端を固定する場合は **両方が同じグループ 0** にまとめられる。
    block_centers
        各グループの軸方向中心座標 [Å] (温度勾配のフィットに使う)。
    axis
        輸送方向。
    length
        セルの軸方向の長さ [Å]。
    """

    labels: np.ndarray
    source: int
    sink: int
    middle: list[int]
    fixed: list[int]
    block_centers: np.ndarray
    axis: str
    length: float

    @property
    def n_groups(self) -> int:
        return int(self.labels.max()) + 1

    @property
    def middle_centers(self) -> np.ndarray:
        """温度勾配を測る中間ブロックの中心座標 [Å]。"""
        return self.block_centers[self.middle]

    def transport_length(self) -> float:
        """熱源中心から熱浴中心までの距離 [Å] (kappa = J L / (A dT) の L)。"""
        return float(abs(self.block_centers[self.sink] - self.block_centers[self.source]))

    def cross_section(self, atoms: Atoms) -> float:
        """輸送方向に垂直な断面積 [Å^2]。"""
        index = _axis_index(self.axis)
        cell = np.asarray(atoms.cell)
        volume = float(atoms.get_volume())
        thickness = volume / np.linalg.norm(np.cross(*np.delete(cell, index, axis=0)))
        return volume / thickness

    def summary(self) -> str:
        fixed = (
            f"fixed={self.fixed} (両端を 1 グループに集約), " if self.fixed else "fixed=なし, "
        )
        return (
            f"NEMD 配置 (輸送方向 {self.axis}, {self.n_groups} グループ, "
            f"L={self.length:.1f} Å): {fixed}"
            f"source={self.source}, sink={self.sink}, "
            f"middle={self.middle[0]}..{self.middle[-1]}"
        )


def nemd_layout(
    atoms: Atoms,
    *,
    axis: str | int = "x",
    n_blocks: int = 10,
    fixed_ends: bool = True,
) -> NEMDLayout:
    """NEMD 熱伝導計算用のグループ配置を作る。

    軸方向を ``n_blocks`` (+ 固定層 2 つ) に等分し、
    ``fix`` する両端・熱源・熱浴・中間ブロックの番号を決める。

    Parameters
    ----------
    n_blocks
        固定層を除いたブロック数。最初が熱源、最後が熱浴、間が測定区間になる。
    fixed_ends
        両端に固定層を置くか。周期境界のまま両端を熱源/熱浴にする場合は
        ``False`` にする (このとき熱流は両方向に流れるので L は半分になる)。
    """
    if n_blocks < 3:
        raise ValueError("n_blocks は 3 以上にしてください (熱源・中間・熱浴)。")
    index = _axis_index(axis)
    axis_name = "xyz"[index]
    total = n_blocks + (2 if fixed_ends else 0)
    slabs = uniform_slabs(atoms, index, total)
    empty = [g for g in range(total) if not np.any(slabs == g)]
    if empty:
        raise ValueError(
            f"空のスラブができました: {empty}。n_blocks を減らすか"
            " セルを大きくしてください。"
        )

    if fixed_ends:
        # GPUMD の fix は 1 run で 1 グループしか固定できないので、
        # 両端のスラブを同じグループ 0 にまとめる。
        labels = slabs.copy()
        labels[slabs == total - 1] = 0
        n_groups = total - 1
        fixed = [0]
        source, sink = 1, total - 2
        middle = list(range(2, total - 2))
    else:
        labels = slabs
        n_groups = total
        fixed = []
        source, sink = 0, total - 1
        middle = list(range(1, total - 1))

    positions = np.asarray(atoms.get_positions()[:, index], dtype=float)
    centers = np.array(
        [
            positions[labels == g].mean() if np.any(labels == g) else np.nan
            for g in range(n_groups)
        ]
    )

    cell = np.asarray(atoms.cell)
    volume = float(atoms.get_volume())
    area = np.linalg.norm(np.cross(*np.delete(cell, index, axis=0)))
    length = volume / area if area > 0 else float(cell[index, index])

    return NEMDLayout(
        labels=labels,
        source=source,
        sink=sink,
        middle=middle,
        fixed=fixed,
        block_centers=centers,
        axis=axis_name,
        length=float(length),
    )


# ------------------------------------------------------------------ まとめ役
@dataclass
class GroupingScheme:
    """複数の grouping method をまとめて ``model.xyz`` 用の形に変換する。

    ``methods[m]`` が grouping method ``m`` の原子ごとのラベル配列。
    """

    methods: list[np.ndarray] = field(default_factory=list)
    names: list[str] = field(default_factory=list)

    @classmethod
    def from_labels(cls, *label_arrays: Sequence[int], name: str | Sequence[str] = "") -> "GroupingScheme":
        """ラベル配列から作る。"""
        names = [name] * len(label_arrays) if isinstance(name, str) else list(name)
        scheme = cls()
        for i, labels in enumerate(label_arrays):
            scheme.add(labels, names[i] if i < len(names) else "")
        return scheme

    def add(self, labels: Sequence[int], name: str = "") -> "GroupingScheme":
        """grouping method を 1 つ追加する。"""
        array = np.asarray(labels, dtype=int)
        if array.ndim != 1:
            raise ValueError("labels は 1 次元配列です。")
        if array.min() < 0:
            raise ValueError("グループラベルは 0 以上の整数です。")
        if self.methods and array.size != self.methods[0].size:
            raise ValueError(
                f"原子数が一致しません ({array.size} vs {self.methods[0].size})。"
            )
        # GPUMD はラベルが 0..N-1 の連番であることを期待する
        unique = np.unique(array)
        if not np.array_equal(unique, np.arange(unique.size)):
            remap = {old: new for new, old in enumerate(unique)}
            array = np.array([remap[v] for v in array], dtype=int)
        self.methods.append(array)
        self.names.append(name or f"method{len(self.methods) - 1}")
        return self

    def __len__(self) -> int:
        return len(self.methods)

    def n_groups(self, method: int = 0) -> int:
        return int(self.methods[method].max()) + 1

    def index_of(self, name: str) -> int:
        """名前から grouping method 番号を引く。"""
        if name not in self.names:
            raise KeyError(f"grouping method '{name}' はありません: {self.names}")
        return self.names.index(name)

    def to_groupings(self) -> list[list[list[int]]]:
        """``StructureHandler.write_model(groupings=...)`` に渡す形に変換する。

        ``groupings[m][g]`` = grouping method ``m`` のグループ ``g`` の原子 index。
        """
        result: list[list[list[int]]] = []
        for labels in self.methods:
            n = int(labels.max()) + 1
            result.append([np.flatnonzero(labels == g).tolist() for g in range(n)])
        return result

    def summary(self) -> str:
        lines = [f"GroupingScheme ({len(self)} 個の grouping method)"]
        for i, (labels, name) in enumerate(zip(self.methods, self.names)):
            counts = np.bincount(labels)
            lines.append(
                f"  [{i}] {name}: {counts.size} グループ "
                f"(最小 {counts.min()} 原子 / 最大 {counts.max()} 原子)"
            )
        return "\n".join(lines)
