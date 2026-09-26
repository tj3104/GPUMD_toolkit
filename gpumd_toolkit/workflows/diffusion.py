"""6. 拡散・イオン伝導・液体物性.

======================================= ======================================
量                                       GPUMD キーワード
======================================= ======================================
自己拡散係数 (MSD 経由)                  ``compute_msd``
自己拡散係数 (速度自己相関経由)          ``compute_sdc``
イオン伝導度                             ``compute_ic``
粘性率 (Green-Kubo)                      ``compute_viscosity``
動径分布関数 / 角度分布関数              ``compute_rdf`` / ``compute_adf``
角度依存 RDF                             ``compute_angular_rdf``
配向秩序変数 (結晶核生成)                ``compute_orientorder``
空間ビンごとの密度・温度・流速           ``compute_chunk``
======================================= ======================================

Examples
--------
>>> from gpumd_toolkit.workflows import DiffusionCalculation
>>> calc = DiffusionCalculation("liquid.xyz", "nep.txt", "runs/liquid")
>>> calc.equilibrate(temperature=1500, steps=50000, pressure=0.0)
>>> calc.production(temperature=1500, steps=500000, msd=True, rdf=True, viscosity=True)
>>> calc.run()
>>> calc.diffusion_coefficient()
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..groups import GroupingScheme, by_species
from ..inputs import computes
from ..md import GPUMDCalculation
from ..structure import StructureHandler
from ..outputs import (
    read_adf,
    read_angular_rdf,
    read_ic,
    read_msd,
    read_orientorder,
    read_rdf,
    read_sdc,
    read_viscosity,
)
from ..postprocess import (
    diffusion_from_msd,
    diffusion_from_sdc,
    ionic_conductivity,
    shear_viscosity,
)

__all__ = ["DiffusionCalculation"]


class DiffusionCalculation(GPUMDCalculation):
    """液体・イオン伝導体の輸送係数と構造相関をまとめるワークフロー。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        #: 元素別に解析するときの grouping method 番号
        self.species_method: int | None = None
        self._species_order: list[str] = []
        self._rdf_pairs: list[str] = []

    # ------------------------------------------------------------------ 準備
    def group_by_species(self) -> "DiffusionCalculation":
        """元素ごとの grouping method を ``model.xyz`` に追加する。

        元素別 MSD (``compute_msd ... all_groups``) や
        元素別 DOS を取るのに必要。
        """
        scheme = self.grouping_scheme or GroupingScheme()
        order: list[str] = []
        for symbol in self.atoms.get_chemical_symbols():
            if symbol not in order:
                order.append(symbol)
        scheme.add(by_species(self.atoms, order), "species")
        self.grouping_scheme = scheme
        self.groupings = scheme.to_groupings()
        self.species_method = scheme.index_of("species")
        self._species_order = order
        return self

    def equilibrate(
        self,
        *,
        temperature: float,
        steps: int,
        pressure: float | None = None,
        thermostat: str = "nvt_nhc",
        **kwargs,
    ) -> "DiffusionCalculation":
        """生成 run の前に平衡化する (液体は NPT で密度を合わせるのが定石)。"""
        if pressure is None:
            self.nvt(temperature=temperature, steps=steps, thermostat=thermostat, **kwargs)
        else:
            self.npt(temperature=temperature, steps=steps, pressure=pressure, **kwargs)
        return self

    # ------------------------------------------------------------------ 本計算
    def production(
        self,
        *,
        temperature: float,
        steps: int,
        ensemble: str = "nvt_nhc",
        pressure: float | None = None,
        msd: bool = True,
        sdc: bool = False,
        ic: tuple[int, float] | None = None,
        viscosity: bool = False,
        rdf: bool | float = False,
        adf: bool = False,
        angular_rdf: bool = False,
        orientorder: Sequence[int] | None = None,
        sample_interval: int = 5,
        Nc: int = 500,
        per_species: bool = False,
        **kwargs,
    ) -> "DiffusionCalculation":
        """生成 run に必要な ``compute_*`` をまとめて仕込む。

        Parameters
        ----------
        msd, sdc
            平均二乗変位 / 速度自己相関からの拡散係数。
            ``compute_sdc`` と ``compute_dos`` は同じ run に入れられない。
        ic
            ``(元素の型番号, 電荷)`` を渡すとイオン伝導度を計算する。
        viscosity
            Green-Kubo で粘性率を計算する。
        rdf
            ``True`` でカットオフ 8 Å、数値を渡すとそのカットオフ [Å]。
        orientorder
            計算する球面調和関数の次数 (例 ``[4, 6]``)。結晶核生成の判定に使う。
        per_species
            ``True`` なら :meth:`group_by_species` のグループごとに MSD を出す。
        """
        if per_species and self.species_method is None:
            self.group_by_species()

        if pressure is None:
            self.nvt(temperature=temperature, steps=steps, thermostat=ensemble, **kwargs)
        else:
            self.npt(temperature=temperature, steps=steps, pressure=pressure, **kwargs)

        commands: list[str] = []
        if msd:
            commands.append(
                computes.compute_msd(
                    sample_interval,
                    Nc,
                    all_groups=self.species_method if per_species else None,
                )
            )
        if sdc:
            commands.append(computes.compute_sdc(sample_interval, Nc))
        if ic is not None:
            type_index, charge = ic
            commands.append(computes.compute_ic(sample_interval, Nc, type_index, charge))
        if viscosity:
            commands.append(computes.compute_viscosity(sample_interval, Nc))
        if rdf:
            cutoff = self._safe_pair_cutoff(8.0 if rdf is True else float(rdf), "compute_rdf")
            commands.append(computes.compute_rdf(cutoff, 400, max(sample_interval, 100)))
        if adf:
            commands.append(computes.compute_adf(max(sample_interval, 100), 60, 0.0, 3.0))
        if angular_rdf:
            cutoff = self._safe_pair_cutoff(8.0, "compute_angular_rdf")
            commands.append(
                computes.compute_angular_rdf(cutoff, 200, 60, max(sample_interval, 100))
            )
        if orientorder:
            commands.append(
                computes.compute_orientorder(
                    max(sample_interval, 100), orientorder, mode="cutoff", parameter=4.0
                )
            )
        return self.add_commands(*commands)

    def density_profile(
        self,
        *,
        axis: str = "z",
        bin_size: float = 1.0,
        sample_interval: int = 10,
        output_interval: int = 100,
        quantities: Sequence[str] = ("temperature", "density/mass"),
    ) -> "DiffusionCalculation":
        """直前のステージに空間ビン平均 (``compute_chunk``) を足す。

        界面・液膜・多孔体の密度や温度の空間分布に使う。
        """
        return self.add_commands(
            computes.compute_chunk(
                sample_interval, output_interval, [(axis, bin_size)], quantities
            )
        )

    def _safe_pair_cutoff(self, cutoff: float, keyword: str) -> float:
        """RDF 系のカットオフをセルに収まる値に丸める。

        GPUMD は周期方向のセル厚みが ``2.5 * cutoff`` 以下だと
        ``The box has a thickness < 2.5 RDF radial cutoffs`` で止まる。
        """
        thickness = min(
            value
            for value, periodic in zip(
                StructureHandler.cell_thickness(self.atoms), self.atoms.pbc
            )
            if periodic
        )
        limit = thickness / 2.5
        if cutoff >= limit:
            safe = round(limit * 0.95, 2)
            warnings.warn(
                f"{keyword} のカットオフ {cutoff:g} Å はセル (最小厚み"
                f" {thickness:.1f} Å) に対して大きすぎます。{safe:g} Å に下げました。"
                " より大きいカットオフが必要なら repeat= でセルを大きくしてください。",
                RuntimeWarning,
            )
            return safe
        return cutoff

    # ------------------------------------------------------------------ 結果
    def msd(self) -> pd.DataFrame:
        return read_msd(self.workdir / "msd.out")

    def sdc(self) -> pd.DataFrame:
        return read_sdc(self.workdir / "sdc.out")

    def diffusion_coefficient(
        self, *, source: str = "msd", group: int | None = None, **kwargs
    ) -> dict[str, float]:
        """自己拡散係数を返す (``D_cm2_per_s`` が実用単位)。

        Parameters
        ----------
        source
            ``'msd'`` (MSD の傾き) か ``'sdc'`` (VAC の積分の収束値)。
        group
            元素別に計算した場合のグループ番号。
        """
        if source == "msd":
            return diffusion_from_msd(self.msd(), group=group, **kwargs)
        if source == "sdc":
            return diffusion_from_sdc(self.sdc(), group=group, **kwargs)
        raise ValueError("source は 'msd' か 'sdc' です。")

    def diffusion_per_species(self, **kwargs) -> pd.DataFrame:
        """元素別の自己拡散係数 (``per_species=True`` で計算した場合)。"""
        if not self._species_order:
            raise RuntimeError("group_by_species() / per_species=True を使ってください。")
        frame = read_msd(self.workdir / "msd.out")
        n_groups = (frame.shape[1] - 1) // 6
        rows = []
        for g in range(n_groups):
            result = diffusion_from_msd(frame, group=g, **kwargs)
            rows.append(
                {
                    "group": g,
                    "species": self._species_order[g] if g < len(self._species_order) else "",
                    **{k: v for k, v in result.items() if k != "fit_range_ps"},
                }
            )
        return pd.DataFrame(rows)

    def ionic_conductivity(self, **kwargs) -> dict[str, float]:
        """``ic.out`` のイオン伝導度 [mS/cm]。"""
        return ionic_conductivity(read_ic(self.workdir / "ic.out"), **kwargs)

    def viscosity(self, *, t_min: float | None = None, t_max: float | None = None) -> dict:
        """``viscosity.out`` からせん断・縦・体積粘性率 [Pa s]。"""
        frame = read_viscosity(self.workdir / "viscosity.out")
        total = float(frame["time"].max())
        return shear_viscosity(
            frame,
            t_min=t_min if t_min is not None else 0.3 * total,
            t_max=t_max if t_max is not None else 0.8 * total,
        )

    def rdf(self) -> pd.DataFrame:
        return read_rdf(self.workdir / "rdf.out", pairs=self._rdf_pairs)

    def adf(self) -> pd.DataFrame:
        return read_adf(self.workdir / "adf.out")

    def angular_rdf(self) -> pd.DataFrame:
        return read_angular_rdf(self.workdir / "angular_rdf.out")

    def orientorder(self) -> pd.DataFrame:
        return read_orientorder(self.workdir / "orientorder.out")

    def coordination_number(self, *, r_max: float, column: str = "total") -> float:
        """RDF を積分して第 1 配位数を求める。

        ``r_max`` は第 1 ピークの後の極小位置 [Å] を指定する。
        """
        frame = self.rdf()
        window = frame[frame["radius"] <= r_max]
        density = self.n_atoms / self.atoms.get_volume()
        integrand = 4.0 * np.pi * window["radius"] ** 2 * window[column] * density
        return float(np.trapezoid(integrand, window["radius"]))

    # ------------------------------------------------------------------ 作図
    def plot_summary(self, filename: str = "diffusion.png", *, dpi: int = 150) -> Path:
        """MSD・RDF・粘性率のうち存在するものをまとめて描く。"""
        from ..plotting import label, setup_matplotlib

        setup_matplotlib()
        import matplotlib.pyplot as plt

        panels = []
        if (self.workdir / "msd.out").is_file():
            panels.append("msd")
        if (self.workdir / "rdf.out").is_file():
            panels.append("rdf")
        if (self.workdir / "viscosity.out").is_file():
            panels.append("viscosity")
        if not panels:
            raise FileNotFoundError("描画できる出力がありません。")

        figure, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
        axes = np.atleast_1d(axes)
        for axis, panel in zip(axes, panels):
            if panel == "msd":
                frame = self.msd()
                for name in ("msd_x", "msd_y", "msd_z"):
                    axis.plot(frame["time"], frame[name], lw=1.0, label=name)
                axis.plot(frame["time"], frame["msd_total"], lw=1.6, color="k", label="total")
                axis.set_xlabel(label("相関時間 (ps)"))
                axis.set_ylabel(label("MSD (Å$^2$)"))
                axis.legend(fontsize=8)
            elif panel == "rdf":
                frame = self.rdf()
                for column in frame.columns[1:]:
                    axis.plot(frame["radius"], frame[column], lw=1.2, label=column)
                axis.axhline(1.0, color="0.7", lw=0.8, ls="--")
                axis.set_xlabel(label("r (Å)"))
                axis.set_ylabel(label("g(r)"))
                axis.legend(fontsize=8)
            else:
                frame = read_viscosity(self.workdir / "viscosity.out")
                axis.plot(frame["time"], frame["eta_shear"], lw=1.2, label="せん断")
                axis.plot(frame["time"], frame["eta_bulk"], lw=1.2, label="体積")
                axis.set_xlabel(label("相関時間 (ps)"))
                axis.set_ylabel(r"$\eta$ (Pa s)")
                axis.legend(fontsize=8)
        figure.suptitle(label(f"拡散・液体物性 — {self.name}"))
        figure.tight_layout()
        output = self.workdir / filename
        figure.savefig(output, dpi=dpi)
        plt.close(figure)
        return output
