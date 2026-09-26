"""1. 静的計算・構造計算.

* 一点計算 (エネルギー・力・応力)
* 構造最適化 (FIRE / 最急降下、セル最適化つき)
* 凝集エネルギー曲線 (状態方程式)
* 弾性定数と多結晶弾性率
* 有限変位法によるフォノン分散

``compute_cohesive`` / ``compute_elastic`` / ``compute_phonon`` と ``minimize`` は
GPUMD が ``run.in`` を読んだ時点で実行されるため ``ensemble`` / ``run`` を必要としない。
本モジュールはその違いを吸収する。

Examples
--------
>>> from gpumd_toolkit.workflows import StaticCalculation
>>> calc = StaticCalculation("POSCAR", "nep.txt", "runs/static")
>>> calc.relax(relax_cell=True).single_point()
>>> result = calc.run()
>>> calc.energy(), calc.forces().shape
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ..inputs import computes, dumps
from ..inputs.builder import MDStage
from ..md import GPUMDCalculation
from ..outputs import read_cohesive, read_elastic, read_omega2
from ..postprocess import ElasticModuli, elastic_moduli

__all__ = ["StaticCalculation", "write_kpoints", "band_path_kpoints"]


def write_kpoints(path: Path | str, kpoints: Sequence[Sequence[float]]) -> Path:
    """``kpoints.in`` を書き出す (``compute_phonon`` 用)。

    ``kpoints`` は逆格子ベクトル単位の 3 成分ベクトルの列。
    1 行目に点数、続けて各点を書く形式。
    """
    points = np.asarray(kpoints, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("kpoints は (N, 3) の配列です。")
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(points))]
    lines += [" ".join(f"{v:.10f}" for v in point) for point in points]
    path.write_text("\n".join(lines) + "\n")
    return path


def band_path_kpoints(
    corners: Sequence[Sequence[float]], n_per_segment: int = 50
) -> np.ndarray:
    """高対称点のリストから直線補間した k 点列を作る。"""
    corners = np.asarray(corners, dtype=float)
    if corners.ndim != 2 or corners.shape[1] != 3 or len(corners) < 2:
        raise ValueError("corners は 2 点以上の (N, 3) 配列です。")
    segments = []
    for start, stop in zip(corners[:-1], corners[1:]):
        segment = np.linspace(start, stop, n_per_segment, endpoint=False)
        segments.append(segment)
    segments.append(corners[-1][None, :])
    return np.vstack(segments)


class StaticCalculation(GPUMDCalculation):
    """静的計算・構造計算をまとめるワークフロー。

    :class:`~gpumd_toolkit.md.GPUMDCalculation` のサブクラスなので、
    構造の読み込み・スーパーセル化・実行・解析はすべて共通の作法で使える。
    """

    #: 一点計算で出力するファイル名
    SINGLE_POINT_FILE = "single_point.xyz"

    def single_point(
        self, *, properties: Sequence[str] = ("force", "potential", "virial")
    ) -> "StaticCalculation":
        """エネルギー・力・応力の一点計算を仕込む。

        GPUMD の定石どおり ``ensemble nve`` + ``time_step 0`` + ``run 1`` で
        1 ステップだけ回し、``dump_xyz`` で全量を書き出す。
        """
        if self.builder.initial_temperature is None:
            # GPUMD の静的計算の定石: velocity 1 + time_step 0
            self.builder.initial_temperature = 1.0
        stage = MDStage(
            "nve",
            steps=1,
            time_step=0.0,
            label="single_point",
            dump=dumps.DumpSettings(
                thermo_interval=1,
                traj_interval=1,
                traj_file=self.SINGLE_POINT_FILE,
                traj_properties=tuple(properties),
                traj_precision="double",
            ),
        )
        return self.add_stage(stage)

    def relax(
        self,
        *,
        method: str = "fire",
        force_tolerance: float = 1e-5,
        max_steps: int = 1000,
        relax_cell: bool = False,
        hydrostatic_strain: bool = False,
    ) -> "StaticCalculation":
        """構造最適化を仕込む。

        Parameters
        ----------
        relax_cell
            セルも動かすか (FIRE のみ)。
        hydrostatic_strain
            セル最適化を等方ひずみに限るか (体積だけ変える)。
        """
        self.builder.set_minimize(
            method,
            force_tolerance,
            max_steps,
            box_change=relax_cell,
            hydrostatic_strain=hydrostatic_strain,
        )
        return self

    def cohesive_curve(
        self, e1: float = 0.9, e2: float = 1.1, direction: str = "xyz"
    ) -> "StaticCalculation":
        """凝集エネルギー曲線 (状態方程式) を仕込む。

        ``(e2 - e1) * 1000 + 1`` 点のスケーリングでエネルギーを計算する。
        """
        return self.add_action(computes.compute_cohesive(e1, e2, direction))

    def elastic(self, strain: float = 0.01) -> "StaticCalculation":
        """弾性定数行列 C_ij の計算を仕込む。"""
        return self.add_action(computes.compute_elastic(strain))

    def phonon(
        self,
        *,
        displacement: float = 0.01,
        kpoints: Sequence[Sequence[float]] | None = None,
        replicate: Sequence[int] | None = None,
    ) -> "StaticCalculation":
        """有限変位法によるフォノン分散を仕込む。

        Parameters
        ----------
        kpoints
            ``kpoints.in`` に書く k 点列 (逆格子単位)。
            :func:`band_path_kpoints` で高対称線から作れる。
        replicate
            ``run.in`` 先頭の ``replicate``。力定数を取るためのスーパーセル。
            NEP のカットオフ (通常 6-8 Å) より十分大きくなるように取る。
        """
        if replicate is not None:
            self.set_replicate(*replicate)
        if kpoints is not None:
            write_kpoints(self.workdir / "kpoints.in", kpoints)
        elif not (self.workdir / "kpoints.in").is_file():
            raise FileNotFoundError(
                "compute_phonon には kpoints.in が必要です。"
                " kpoints= を渡すか、自分で workdir に置いてください。"
            )
        return self.add_action(computes.compute_phonon(displacement))

    # ------------------------------------------------------------------ 結果
    def single_point_atoms(self):
        """一点計算の結果を :class:`ase.Atoms` として読む (力・エネルギーつき)。"""
        from ..fastio import read_xyz_fast

        return read_xyz_fast(self.workdir / self.SINGLE_POINT_FILE, index=-1)

    def energy(self) -> float:
        """一点計算のポテンシャルエネルギー [eV]。"""
        from ..outputs import read_thermo

        return float(read_thermo(self.workdir / "thermo.out")["potential_energy"].iloc[-1])

    def energy_per_atom(self) -> float:
        return self.energy() / self.n_atoms

    def forces(self) -> np.ndarray:
        """一点計算の力 [eV/Å] (N, 3)。"""
        atoms = self.single_point_atoms()
        if "forces" in atoms.arrays:
            return np.asarray(atoms.arrays["forces"], dtype=float)
        raise KeyError(
            f"{self.SINGLE_POINT_FILE} に力が入っていません。"
            " single_point(properties=('force', ...)) を指定してください。"
        )

    def stress(self) -> np.ndarray:
        """一点計算の応力 (Voigt 6 成分) [GPa]。"""
        from ..outputs import read_thermo

        row = read_thermo(self.workdir / "thermo.out").iloc[-1]
        return np.array([row[f"P{c}"] for c in ("xx", "yy", "zz", "yz", "xz", "xy")])

    def cohesive(self):
        """``cohesive.out`` を読む (スケーリング係数とエネルギー)。"""
        frame = read_cohesive(self.workdir / "cohesive.out")
        frame["energy_per_atom"] = frame["energy"] / self.n_atoms
        frame["volume"] = frame["scale"] ** 3 * self.atoms.get_volume()
        return frame

    def equilibrium_from_cohesive(self) -> dict[str, float]:
        """凝集エネルギー曲線の最小点 (放物線フィット) を返す。"""
        frame = self.cohesive()
        index = int(frame["energy"].idxmin())
        lo, hi = max(index - 20, 0), min(index + 21, len(frame))
        window = frame.iloc[lo:hi]
        a, b, c = np.polyfit(window["scale"], window["energy"], 2)
        scale = -b / (2 * a)
        return {
            "scale": float(scale),
            "energy": float(a * scale**2 + b * scale + c),
            "energy_per_atom": float((a * scale**2 + b * scale + c) / self.n_atoms),
            "volume": float(scale**3 * self.atoms.get_volume()),
        }

    def elastic_matrix(self) -> np.ndarray:
        """``elastic.out`` の 6x6 弾性定数行列 [GPa]。"""
        return read_elastic(self.workdir / "elastic.out")

    def elastic_moduli(self) -> ElasticModuli:
        """弾性定数から体積弾性率・剛性率・ヤング率・ポアソン比を求める。"""
        return elastic_moduli(self.elastic_matrix())

    def phonon_dispersion(self):
        """``omega2.out`` を読む (距離と各分枝の振動数 [THz])。"""
        return read_omega2(self.workdir / "omega2.out")

    def plot_phonon(self, filename: str = "phonon.png", *, dpi: int = 150) -> Path:
        """フォノン分散を描く。虚振動は負の振動数として描かれる。"""
        from ..plotting import label, setup_matplotlib

        setup_matplotlib()
        import matplotlib.pyplot as plt

        frame = self.phonon_dispersion()
        branches = [c for c in frame.columns if c.startswith("nu_")]
        figure, axis = plt.subplots(figsize=(6, 4))
        for name in branches:
            axis.plot(frame["distance"], frame[name], color="C0", lw=1.0)
        axis.axhline(0.0, color="0.6", lw=0.8, ls="--")
        axis.set_xlabel(label("高対称線に沿った距離"))
        axis.set_ylabel(label("振動数 (THz)"))
        axis.set_title(label(f"フォノン分散 — {self.name}"))
        figure.tight_layout()
        output = self.workdir / filename
        figure.savefig(output, dpi=dpi)
        plt.close(figure)
        return output

    def plot_cohesive(self, filename: str = "cohesive.png", *, dpi: int = 150) -> Path:
        """凝集エネルギー曲線を描く。"""
        from ..plotting import label, setup_matplotlib

        setup_matplotlib()
        import matplotlib.pyplot as plt

        frame = self.cohesive()
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.plot(frame["volume"] / self.n_atoms, frame["energy_per_atom"], lw=1.5)
        minimum = self.equilibrium_from_cohesive()
        axis.axvline(minimum["volume"] / self.n_atoms, color="C3", ls="--", lw=1.0)
        axis.set_xlabel(label("体積 (Å$^3$/atom)"))
        axis.set_ylabel(label("エネルギー (eV/atom)"))
        axis.set_title(label(f"凝集エネルギー曲線 — {self.name}"))
        figure.tight_layout()
        output = self.workdir / filename
        figure.savefig(output, dpi=dpi)
        plt.close(figure)
        return output
