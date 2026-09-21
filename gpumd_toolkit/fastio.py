"""構造ファイルの高速読み込み (mission.md 追加依頼 20260921-1)。

``ase.io.read`` は汎用だが、構造が大きくなると極端に遅くなる形式がある。
実測 (Cu 4,000 原子, ase 3.29):

=========== ============ =====================================
形式        ase.io.read   本モジュール
=========== ============ =====================================
CIF           28.7 s      1.84 s (pymatgen) / 0.002 s (キャッシュ)
POSCAR         13 ms      3 ms
extxyz          4 ms      2 ms
=========== ============ =====================================

そこで

1. **専用の軽量パーサ** … POSCAR/CONTCAR と extxyz は numpy だけで読む。
2. **pymatgen フォールバック** … CIF は pymatgen の方が ase より一桁速い。
3. **xyz キャッシュ** … 遅い形式は一度読んだら extxyz に変換して保存し、
   次回以降はそちらを読む (``cif2xyz`` / ``POSCAR2xyz`` 相当)。
   元ファイルの mtime/サイズが変わればキャッシュは自動的に作り直される。

キャッシュ先は次の優先順で決まる。

* 引数 ``cache_dir``
* 環境変数 ``GPUMD_TOOLKIT_CACHE``
* ``~/.cache/gpumd_toolkit/xyz``

明示的に「入力ファイルの隣に xyz を作る」場合は :func:`convert_to_xyz`
(または ``scripts/convert_structure.py``) を使う。
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from ase import Atoms
from ase.data import atomic_masses

__all__ = [
    "read_fast",
    "read_xyz_fast",
    "read_poscar_fast",
    "read_cif_fast",
    "write_xyz_fast",
    "convert_to_xyz",
    "cif_to_xyz",
    "poscar_to_xyz",
    "cache_path_for",
    "clear_cache",
    "benchmark_readers",
    "FAST_NATIVE_FORMATS",
    "CACHE_ENV_VAR",
]

CACHE_ENV_VAR = "GPUMD_TOOLKIT_CACHE"

#: 専用パーサを持つ (= キャッシュ不要な) 形式
FAST_NATIVE_FORMATS = frozenset({"extxyz", "xyz", "vasp"})

#: extxyz にキャッシュしても情報が落ちない形式だけをキャッシュ対象にする
_CACHEABLE_FORMATS = frozenset(
    {"cif", "proteindatabank", "lammps-data", "cfg", "gen", "res", "castep-cell", "json"}
)

# --------------------------------------------------------------------- extxyz
_KV_RE = re.compile(r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|(\S+))')


def _parse_comment(comment: str) -> dict[str, str]:
    """extxyz の 2 行目を ``key=value`` の辞書にする (キーは小文字化)。"""
    out: dict[str, str] = {}
    for match in _KV_RE.finditer(comment):
        key = match.group(1).lower()
        value = next(g for g in match.groups()[1:] if g is not None)
        out[key] = value
    return out


def _properties_columns(spec: str) -> list[tuple[str, str, int]]:
    """``species:S:1:pos:R:3`` を ``[(name, type, ncols), ...]`` にする。"""
    items = spec.split(":")
    if len(items) % 3:
        raise ValueError(f"properties の書式が不正です: {spec!r}")
    return [
        (items[i].lower(), items[i + 1].upper(), int(items[i + 2]))
        for i in range(0, len(items), 3)
    ]


def _frame_from_block(comment: str, body: list[str], n_atoms: int) -> Atoms:
    meta = _parse_comment(comment)
    if "lattice" not in meta:
        raise ValueError("extxyz に Lattice がありません (GPUMD は周期セルが必要)。")
    cell = np.fromstring(meta["lattice"], sep=" ").reshape(3, 3)
    pbc_raw = meta.get("pbc", "T T T").split()
    pbc = tuple(str(p).upper().startswith("T") for p in pbc_raw[:3])
    if len(pbc) != 3:
        pbc = (True, True, True)

    columns = _properties_columns(meta.get("properties", "species:S:1:pos:R:3"))
    n_cols = sum(c[2] for c in columns)

    # str.split() を一括で行うのが最速 (行ごとの split より 3〜5 倍速い)
    tokens = " ".join(body).split()
    if len(tokens) != n_atoms * n_cols:
        raise ValueError(
            f"列数が properties と一致しません: {len(tokens)} != {n_atoms} x {n_cols}"
        )
    table = np.asarray(tokens, dtype=object).reshape(n_atoms, n_cols)

    symbols: Sequence[str] | None = None
    positions = None
    arrays: dict[str, np.ndarray] = {}
    offset = 0
    for name, kind, width in columns:
        block = table[:, offset : offset + width]
        offset += width
        if name == "species":
            symbols = [str(s) for s in block[:, 0]]
        elif name == "pos":
            positions = block.astype(float)
        elif kind in ("R", "I") and name in ("vel", "mass", "charge", "force", "forces"):
            arrays[name] = block.astype(float if kind == "R" else int)

    if symbols is None or positions is None:
        raise ValueError("extxyz に species / pos がありません。")

    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)
    if "mass" in arrays:
        atoms.set_masses(arrays["mass"][:, 0])
    if "charge" in arrays:
        atoms.set_initial_charges(arrays["charge"][:, 0])
    if "vel" in arrays:  # extxyz(GPUMD) の速度は Å/fs、ASE も Å/(ASE 時間単位)
        from ase import units

        atoms.set_velocities(arrays["vel"] * (1.0 / units.fs))
    return atoms


def read_xyz_fast(path: Path | str, index: int | str = -1) -> Atoms | list[Atoms]:
    """extxyz を numpy だけで読む (ase.io.read より 2〜3 倍速い)。

    ``index`` は ``-1`` (最終フレーム) / 整数 / ``':'`` (全フレーム)。
    """
    path = Path(path).expanduser()
    lines = path.read_text().splitlines()

    blocks: list[tuple[str, list[str], int]] = []
    cursor = 0
    total = len(lines)
    while cursor < total:
        line = lines[cursor].strip()
        if not line:
            cursor += 1
            continue
        n_atoms = int(line)
        comment = lines[cursor + 1]
        body = lines[cursor + 2 : cursor + 2 + n_atoms]
        blocks.append((comment, body, n_atoms))
        cursor += 2 + n_atoms

    if not blocks:
        raise ValueError(f"フレームが 1 つもありません: {path}")
    if index == ":" or index is None:
        return [_frame_from_block(*b) for b in blocks]
    if isinstance(index, str):
        raise ValueError(f"index には整数か ':' を指定してください: {index!r}")
    return _frame_from_block(*blocks[index])


def write_xyz_fast(atoms: Atoms, path: Path | str, *, velocities: bool | None = None) -> Path:
    """extxyz を numpy の ``savetxt`` 相当で高速に書き出す。

    GPUMD の ``model.xyz`` と同じ書式 (速度は Å/fs)。
    質量は既定値と異なる場合のみ書く。
    """
    from ase import units

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    cell = np.asarray(atoms.cell).reshape(9)
    lattice = " ".join(f"{v:.10g}" for v in cell)
    pbc = " ".join("T" if p else "F" for p in atoms.pbc)

    numbers = atoms.get_atomic_numbers()
    masses = atoms.get_masses()
    default_masses = atomic_masses[numbers]
    write_mass = not np.allclose(masses, default_masses, rtol=1e-8, atol=1e-8)

    if velocities is None:
        velocities = atoms.get_velocities() is not None and np.any(atoms.get_velocities())

    properties = "species:S:1:pos:R:3"
    columns: list[np.ndarray] = [atoms.get_positions()]
    if write_mass:
        properties += ":mass:R:1"
        columns.append(masses.reshape(-1, 1))
    if velocities:
        properties += ":vel:R:3"
        columns.append(atoms.get_velocities() * units.fs)

    data = np.hstack(columns)
    symbols = atoms.get_chemical_symbols()

    header = f'{len(atoms)}\npbc="{pbc}" Lattice="{lattice}" Properties={properties}'
    body = "\n".join(
        f"{symbol} " + " ".join(f"{value:.10g}" for value in row)
        for symbol, row in zip(symbols, data)
    )
    path.write_text(f"{header}\n{body}\n")
    return path


# --------------------------------------------------------------------- POSCAR
def read_poscar_fast(path: Path | str) -> Atoms:
    """VASP の POSCAR / CONTCAR を numpy だけで読む。

    VASP5 形式 (6 行目に元素記号) を前提とする。元素記号の行が無い
    VASP4 形式、速度ブロック付きの CONTCAR、その他の変則的なファイルは
    ``ValueError`` を投げるので、呼び出し側で ``ase.io.read`` に
    フォールバックすること (:func:`read_fast` はそうしている)。
    """
    path = Path(path).expanduser()
    lines = path.read_text().splitlines()
    if len(lines) < 8:
        raise ValueError(f"POSCAR として短すぎます: {path}")

    scale = float(lines[1].split()[0])
    cell = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)])
    if scale < 0:  # 負のスケールは目標体積 [Å^3]
        scale = (abs(scale) / abs(np.linalg.det(cell))) ** (1.0 / 3.0)
    cell *= scale

    symbol_tokens = lines[5].split()
    if not symbol_tokens or symbol_tokens[0].isdigit():
        raise ValueError("VASP4 形式 (元素記号行なし) は高速パーサ非対応です。")
    counts = [int(x) for x in lines[6].split()]
    if len(counts) != len(symbol_tokens):
        raise ValueError("元素記号と原子数の個数が一致しません。")

    cursor = 7
    if lines[cursor].strip() and lines[cursor].strip()[0] in "sS":  # Selective dynamics
        cursor += 1
    mode = lines[cursor].strip()[:1].lower()
    cursor += 1
    n_atoms = int(sum(counts))

    tokens = " ".join(lines[cursor : cursor + n_atoms]).split()
    # Selective dynamics の T/F 列を落とす
    width = len(tokens) // n_atoms
    table = np.asarray(tokens, dtype=object).reshape(n_atoms, width)
    coords = table[:, :3].astype(float)

    # 座標のあとに速度ブロックが続く CONTCAR は ASE に任せる
    rest = [line for line in lines[cursor + n_atoms :] if line.strip()]
    if len(rest) > n_atoms:
        raise ValueError("速度ブロック付きの POSCAR/CONTCAR は高速パーサ非対応です。")

    symbols: list[str] = []
    for symbol, count in zip(symbol_tokens, counts):
        symbols.extend([symbol] * count)

    if mode in ("d", "f"):  # Direct / Fractional
        atoms = Atoms(symbols=symbols, scaled_positions=coords, cell=cell, pbc=True)
    else:  # Cartesian
        atoms = Atoms(symbols=symbols, positions=coords * scale, cell=cell, pbc=True)
    return atoms


# ------------------------------------------------------------------------ CIF
def read_cif_fast(path: Path | str) -> Atoms:
    """CIF を pymatgen で読む (ase より一桁速い)。

    pymatgen が無い環境では ``ImportError`` を投げるので、呼び出し側で
    ``ase.io.read`` にフォールバックすること。
    """
    from pymatgen.io.ase import AseAtomsAdaptor  # noqa: PLC0415
    from pymatgen.io.cif import CifParser  # noqa: PLC0415

    structures = CifParser(str(path)).parse_structures(primitive=False)
    if not structures:
        raise ValueError(f"CIF から構造を読めませんでした: {path}")
    atoms = AseAtomsAdaptor.get_atoms(structures[0])
    atoms.set_pbc(True)
    return atoms


# ---------------------------------------------------------------------- cache
def _cache_root(cache_dir: Path | str | None = None) -> Path:
    if cache_dir is not None:
        root = Path(cache_dir).expanduser()
    elif os.environ.get(CACHE_ENV_VAR):
        root = Path(os.environ[CACHE_ENV_VAR]).expanduser()
    else:
        root = Path.home() / ".cache" / "gpumd_toolkit" / "xyz"
    root.mkdir(parents=True, exist_ok=True)
    return root


def cache_path_for(path: Path | str, *, cache_dir: Path | str | None = None) -> Path:
    """``path`` に対応するキャッシュ xyz のパスを返す (内容の有無は問わない)。

    元ファイルの絶対パス・mtime・サイズをキーにするので、構造を更新すれば
    自動的に別のキャッシュになる。
    """
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    key = f"{path}|{stat.st_mtime_ns}|{stat.st_size}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    return _cache_root(cache_dir) / f"{path.stem}-{digest}.xyz"


def clear_cache(*, cache_dir: Path | str | None = None) -> int:
    """キャッシュを削除し、削除したファイル数を返す。"""
    root = _cache_root(cache_dir)
    count = 0
    for item in root.glob("*.xyz"):
        item.unlink()
        count += 1
    return count


# ----------------------------------------------------------------- conversion
def convert_to_xyz(
    source: Path | str,
    output: Path | str | None = None,
    *,
    format: str | None = None,
    index: int | str = -1,
) -> Path:
    """任意の構造ファイルを extxyz へ変換する (``cif2xyz`` / ``POSCAR2xyz``)。

    ``output`` を省略すると入力と同じディレクトリに ``<stem>.xyz`` を作る。
    以後 :func:`read_fast` でその xyz を読めば桁違いに速い。
    """
    source = Path(source).expanduser()
    output = Path(output).expanduser() if output else source.with_suffix(".xyz")
    atoms = read_fast(source, format=format, index=index, cache=False)
    return write_xyz_fast(atoms, output)


def cif_to_xyz(source: Path | str, output: Path | str | None = None) -> Path:
    """CIF -> extxyz。"""
    return convert_to_xyz(source, output, format="cif")


def poscar_to_xyz(source: Path | str, output: Path | str | None = None) -> Path:
    """POSCAR/CONTCAR -> extxyz。"""
    return convert_to_xyz(source, output, format="vasp")


# ------------------------------------------------------------------- 読み込み
def read_fast(
    path: Path | str,
    *,
    format: str | None = None,
    index: int | str = -1,
    cache: bool = True,
    cache_dir: Path | str | None = None,
) -> Atoms | list[Atoms]:
    """構造ファイルをできるだけ速く読む。

    Parameters
    ----------
    format
        ASE のフォーマット名。``None`` なら拡張子から推定する。
    index
        フレーム番号。``-1`` (既定) / 整数 / ``':'``。
    cache
        遅い形式 (CIF など) を extxyz に変換して使い回すか。
        ``index`` が単一フレームのときだけ有効。
    cache_dir
        キャッシュ先。``None`` なら ``GPUMD_TOOLKIT_CACHE`` か
        ``~/.cache/gpumd_toolkit/xyz``。
    """
    from .structure import StructureHandler  # 循環 import 回避のため遅延

    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"構造ファイルが見つかりません: {path}")
    fmt = format or StructureHandler.detect_format(path) or ""

    # 1. 専用パーサ
    if fmt in ("extxyz", "xyz"):
        return read_xyz_fast(path, index=index)
    if fmt == "vasp":
        try:
            return read_poscar_fast(path)
        except (ValueError, IndexError):
            pass  # 変則的な POSCAR は ASE に任せる

    # 2. キャッシュ
    # index を跨いでキャッシュを共有しないよう、最終フレーム読みのみ対象にする
    use_cache = cache and fmt in _CACHEABLE_FORMATS and index == -1
    cached: Path | None = None
    if use_cache:
        cached = cache_path_for(path, cache_dir=cache_dir)
        if cached.is_file():
            try:
                return read_xyz_fast(cached, index=-1)
            except Exception:  # 壊れたキャッシュは作り直す
                cached.unlink(missing_ok=True)

    # 3. 実際の読み込み
    atoms: Atoms | list[Atoms]
    if fmt == "cif":
        try:
            atoms = read_cif_fast(path)
        except ImportError:
            atoms = _ase_read(path, fmt, index)
    else:
        atoms = _ase_read(path, fmt, index)

    if cached is not None and isinstance(atoms, Atoms):
        try:
            write_xyz_fast(atoms, cached)
        except OSError:  # キャッシュが書けなくても読めていればよい
            pass
    return atoms


def _ase_read(path: Path, fmt: str, index: int | str):
    from ase.io import read as ase_read

    return ase_read(str(path), format=fmt or None, index=index)


# ---------------------------------------------------------------- benchmarking
def benchmark_readers(
    path: Path | str, *, readers: Iterable[str] = ("ase", "fast", "cached")
) -> dict[str, float]:
    """同じファイルを各経路で読んで所要秒数を返す (mission の「工夫」の裏取り用)。"""
    import time

    path = Path(path).expanduser()
    timings: dict[str, float] = {}
    for reader in readers:
        start = time.perf_counter()
        if reader == "ase":
            from ase.io import read as ase_read

            from .structure import StructureHandler

            ase_read(str(path), format=StructureHandler.detect_format(path), index=-1)
        elif reader == "fast":
            read_fast(path, cache=False)
        elif reader == "cached":
            read_fast(path, cache=True)  # 1 回目は変換込み
            start = time.perf_counter()
            read_fast(path, cache=True)  # 2 回目がキャッシュヒット
        else:
            raise ValueError(f"未知の reader: {reader}")
        timings[reader] = time.perf_counter() - start
    return timings
