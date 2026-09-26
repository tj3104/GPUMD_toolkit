"""GPUMD トラジェクトリ (``dump.xyz``) の読み込みと形式変換。

**既定の出力は拡張 XYZ と XDATCAR**。GPUMD が吐く ``dump.xyz`` がそのまま
拡張 XYZ なので、追加の変換なしで OVITO / VESTA / ASE / VMD から開ける。
ASE の ``.traj`` は専用ツールがないと中身を見られず扱いにくいため、
明示的に指定したときだけ書く。

ASE が書ける他の形式 (cif, pdb, lammps-dump, netcdf など) にも変換できる。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from ase import Atoms
from ase.io import iread as ase_iread
from ase.io import read as ase_read
from ase.io import write as ase_write

__all__ = ["TrajectoryConverter", "SUPPORTED_OUTPUT_FORMATS"]

#: 出力形式のエイリアス -> (ASE フォーマット名, 既定の拡張子)
SUPPORTED_OUTPUT_FORMATS: dict[str, tuple[str, str]] = {
    "xdatcar": ("vasp-xdatcar", "XDATCAR"),
    "traj": ("traj", ".traj"),
    "extxyz": ("extxyz", ".xyz"),
    "xyz": ("extxyz", ".xyz"),
    "lammps-dump": ("lammps-dump-text", ".lammpstrj"),
    "pdb": ("proteindatabank", ".pdb"),
    "cif": ("cif", ".cif"),
    "netcdf": ("netcdftrajectory", ".nc"),
    "vasp": ("vasp", ".vasp"),
}


class TrajectoryConverter:
    """``dump.xyz`` を読み、任意の形式へ変換する。

    フレーム数が多い場合に備え、既定では ASE の ``iread`` による逐次読み出しを
    使う (全フレームをメモリに載せない)。

    Examples
    --------
    >>> conv = TrajectoryConverter("runs/si_nvt/dump.xyz")
    >>> conv.to_xyz("runs/si_nvt/trajectory.xyz", stride=10)
    >>> conv.to_xdatcar("runs/si_nvt/XDATCAR", stride=10)
    >>> frames = conv.read(index="-10:")      # 最後の 10 フレーム
    """

    def __init__(self, path: Path | str, *, format: str = "extxyz") -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"トラジェクトリが見つかりません: {self.path}")
        self.format = format

    # ------------------------------------------------------------------ 読み出し
    def iter_frames(
        self, *, stride: int = 1, start: int = 0, stop: int | None = None
    ) -> Iterator[Atoms]:
        """フレームを逐次返す。"""
        if stride < 1:
            raise ValueError("stride は 1 以上です。")
        for i, atoms in enumerate(ase_iread(str(self.path), format=self.format, index=":")):
            if i < start:
                continue
            if stop is not None and i >= stop:
                break
            if (i - start) % stride == 0:
                yield atoms

    def read(self, index: int | str = ":") -> Atoms | list[Atoms]:
        """ASE の index 記法でフレームを読む (``':'``, ``-1``, ``'::10'`` など)。"""
        return ase_read(str(self.path), format=self.format, index=index)

    def __len__(self) -> int:
        return sum(1 for _ in self.iter_frames())

    @property
    def n_frames(self) -> int:
        return len(self)

    def first(self) -> Atoms:
        return next(self.iter_frames())

    def last(self) -> Atoms:
        return self.read(index=-1)

    # ------------------------------------------------------------------ 変換
    def convert(
        self,
        output: Path | str,
        *,
        fmt: str | None = None,
        stride: int = 1,
        start: int = 0,
        stop: int | None = None,
        wrap: bool = False,
    ) -> Path:
        """任意形式へ変換する。

        Parameters
        ----------
        output
            出力ファイルパス。
        fmt
            :data:`SUPPORTED_OUTPUT_FORMATS` のキー。``None`` ならパス名から推定。
        stride, start, stop
            フレームの間引き。
        wrap
            セル内へ座標を折り返すか。XDATCAR は分率座標なので既定で折り返す。
        """
        output = Path(output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        key = (fmt or self._guess_format(output)).lower()
        if key not in SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(
                f"未対応の出力形式 '{key}'。"
                f" 使用可能: {', '.join(sorted(SUPPORTED_OUTPUT_FORMATS))}"
            )
        ase_format, _ = SUPPORTED_OUTPUT_FORMATS[key]
        if key == "xdatcar":
            wrap = True

        frames = []
        for atoms in self.iter_frames(stride=stride, start=start, stop=stop):
            if wrap:
                atoms.wrap()
            frames.append(atoms)
        if not frames:
            raise ValueError("変換対象のフレームがありません。")

        if ase_format == "vasp-xdatcar":
            # XDATCAR は全フレームで同じ元素順序・原子数である必要がある
            _check_constant_composition(frames)
            ase_write(str(output), frames, format=ase_format)
        elif ase_format == "vasp":
            ase_write(str(output), frames[-1], format=ase_format, direct=True)
        else:
            ase_write(str(output), frames, format=ase_format)
        return output

    def to_xdatcar(self, output: Path | str = "XDATCAR", **kwargs) -> Path:
        """VASP の XDATCAR 形式で書き出す。"""
        return self.convert(output, fmt="xdatcar", **kwargs)

    def to_xyz(self, output: Path | str = "trajectory.xyz", **kwargs) -> Path:
        """拡張 XYZ 形式で書き出す (推奨)。

        GPUMD の ``dump.xyz`` と同じ形式なので、間引き (``stride``) や
        フレームの切り出しだけしたい場合に使う。
        """
        return self.convert(output, fmt="extxyz", **kwargs)

    #: :meth:`to_xyz` の別名 (旧 API)
    to_extxyz = to_xyz

    def to_ase_traj(self, output: Path | str = "md.traj", **kwargs) -> Path:
        """ASE の ``.traj`` 形式で書き出す。

        .. note::
           扱いやすさの点では :meth:`to_xyz` / :meth:`to_xdatcar` を推奨する。
           ``.traj`` は ASE 専用のバイナリなので他のツールで開けない。
        """
        return self.convert(output, fmt="traj", **kwargs)

    def to_poscar(self, output: Path | str = "CONTCAR", index: int = -1) -> Path:
        """指定フレームを POSCAR 形式で書き出す。"""
        output = Path(output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        ase_write(str(output), self.read(index=index), format="vasp", direct=True)
        return output

    def convert_all(
        self,
        output_dir: Path | str,
        *,
        formats: Sequence[str] = ("xyz", "xdatcar"),
        stride: int = 1,
        basename: str = "trajectory",
    ) -> dict[str, Path]:
        """複数形式へ一括変換する (既定は拡張 XYZ と XDATCAR)。"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        for key in formats:
            _, suffix = SUPPORTED_OUTPUT_FORMATS[key.lower()]
            name = suffix if suffix.isupper() else f"{basename}{suffix}"
            written[key] = self.convert(output_dir / name, fmt=key, stride=stride)
        return written

    # ------------------------------------------------------------------ 解析
    def msd(self, *, stride: int = 1, time_step_ps: float | None = None) -> dict:
        """アンラップ座標から MSD を計算する (参照は先頭フレーム)。

        ``dump_xyz`` に ``unwrapped_position`` を含めていればそれを使い、
        無ければセル境界の折り返しを補正して自前でアンラップする。
        """
        frames = list(self.iter_frames(stride=stride))
        if len(frames) < 2:
            raise ValueError("MSD の計算には 2 フレーム以上が必要です。")
        positions = _unwrapped_positions(frames)
        displacement = positions - positions[0]
        msd_xyz = (displacement**2).mean(axis=1)  # (n_frames, 3)
        msd_total = msd_xyz.sum(axis=1)
        result = {
            "msd_x": msd_xyz[:, 0],
            "msd_y": msd_xyz[:, 1],
            "msd_z": msd_xyz[:, 2],
            "msd_total": msd_total,
            "n_frames": len(frames),
        }
        if time_step_ps is not None:
            times = np.arange(len(frames)) * time_step_ps
            result["time_ps"] = times
            # Einstein 関係: MSD = 6 D t (3 次元)。後半のみで線形回帰する。
            half = len(times) // 2
            if half >= 2:
                slope = np.polyfit(times[half:], msd_total[half:], 1)[0]
                result["D_A2_per_ps"] = float(slope / 6.0)
                result["D_1e-9_m2_per_s"] = float(slope / 6.0 * 10.0)
        return result

    def rdf(self, *, rmax: float = 8.0, nbins: int = 200, stride: int = 10) -> dict:
        """動径分布関数 g(r) を計算する (全元素・全フレーム平均)。"""
        from ase.geometry import get_distances

        counts = np.zeros(nbins)
        edges = np.linspace(0.0, rmax, nbins + 1)
        n_frames = 0
        volume = 0.0
        n_atoms = 0
        for atoms in self.iter_frames(stride=stride):
            _, distances = get_distances(
                atoms.get_positions(), cell=atoms.cell, pbc=atoms.pbc
            )
            triu = np.triu_indices(len(atoms), k=1)
            hist, _ = np.histogram(distances[triu], bins=edges)
            counts += hist
            volume += atoms.get_volume()
            n_atoms = len(atoms)
            n_frames += 1
        if n_frames == 0:
            raise ValueError("フレームがありません。")
        volume /= n_frames
        centers = 0.5 * (edges[:-1] + edges[1:])
        shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
        density = n_atoms / volume
        ideal = 0.5 * n_atoms * density * shell * n_frames
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(ideal > 0, counts / ideal, 0.0)
        return {"r": centers, "g": g}

    # ------------------------------------------------------------------ 内部
    @staticmethod
    def _guess_format(path: Path) -> str:
        name = path.name.upper()
        if name.startswith("XDATCAR"):
            return "xdatcar"
        if name.startswith("POSCAR") or name.startswith("CONTCAR"):
            return "vasp"
        suffix = path.suffix.lower()
        for key, (_, ext) in SUPPORTED_OUTPUT_FORMATS.items():
            if ext.lower() == suffix:
                return key
        raise ValueError(f"{path} から出力形式を推定できません。fmt= を指定してください。")


def _check_constant_composition(frames: Sequence[Atoms]) -> None:
    reference = frames[0].get_chemical_symbols()
    for i, atoms in enumerate(frames[1:], start=1):
        if atoms.get_chemical_symbols() != reference:
            raise ValueError(
                f"フレーム {i} の元素並びが先頭フレームと異なります。"
                " XDATCAR は原子数・元素順序が一定である必要があります。"
            )


def _unwrapped_positions(frames: Sequence[Atoms]) -> np.ndarray:
    """(n_frames, n_atoms, 3) のアンラップ座標を返す。"""
    if "unwrapped_position" in frames[0].arrays:
        return np.array([a.arrays["unwrapped_position"] for a in frames])

    positions = np.array([a.get_positions() for a in frames])
    cell = np.asarray(frames[0].cell)
    if np.allclose(cell, 0.0):
        return positions
    inverse = np.linalg.inv(cell)
    unwrapped = positions.copy()
    shift = np.zeros_like(positions[0])
    for i in range(1, len(positions)):
        delta = positions[i] - positions[i - 1]
        fractional = delta @ inverse
        jumps = np.round(fractional)
        shift = shift - jumps @ cell
        unwrapped[i] = positions[i] + shift
    return unwrapped
