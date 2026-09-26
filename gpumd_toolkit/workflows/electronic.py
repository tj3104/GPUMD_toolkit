"""7. 電子・電気的性質.

============================================ =================================
量                                            GPUMD キーワード
============================================ =================================
電子輸送 (線形スケーリング量子輸送)           ``compute_lsqt``
双極子モーメント → 赤外スペクトル             ``dump_dipole``
分極率 → ラマンスペクトル                     ``dump_polarizability``
分極の時間微分 (BEC 経由)                     ``compute_dpdt``
外部電場下のダイナミクス                      ``add_efield``
電荷の出力・静電相互作用                      ``dump_xyz charge`` / ``kspace``
============================================ =================================

``dump_dipole`` / ``dump_polarizability`` / ``compute_dpdt`` / ``add_efield`` の
BEC モードは、それぞれ双極子・分極率・Born 有効電荷を学習した NEP
(qNEP) が必要になる。``compute_lsqt`` は現状 炭素系の強束縛模型のみ。

Examples
--------
>>> from gpumd_toolkit.workflows import ElectronicCalculation
>>> calc = ElectronicCalculation("water.xyz", "nep.txt", "runs/ir")
>>> calc.infrared(temperature=300, steps=200000, nep_dipole="nep_dipole.txt")
>>> calc.run()
>>> calc.infrared_spectrum().head()
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from ..inputs import computes, dumps, modifiers
from ..md import GPUMDCalculation
from ..outputs import read_dipole, read_dpdt, read_lsqt, read_polarizability
from ..postprocess import vibrational_spectrum

__all__ = ["ElectronicCalculation"]


class ElectronicCalculation(GPUMDCalculation):
    """電子・電気的性質の計算をまとめるワークフロー。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._dipole_interval: int | None = None
        self._polarizability_interval: int | None = None
        self._lsqt_range: tuple[float, float] | None = None

    # ------------------------------------------------------------ 電子輸送
    def lsqt(
        self,
        *,
        temperature: float,
        steps: int,
        direction: str = "x",
        num_moments: int = 3000,
        num_energies: int = 1001,
        E_1: float = -8.1,
        E_2: float = 8.1,
        E_max: float = 8.2,
        thermostat: str = "nvt_nhc",
        **kwargs,
    ) -> "ElectronicCalculation":
        """線形スケーリング量子輸送 (LSQT) で電気伝導度・状態密度を計算する。

        MD と結合しているので電子-フォノン散乱が自然に入る。
        平衡 MD (熱浴つき) の上で走らせる。
        """
        self.nvt(temperature=temperature, steps=steps, thermostat=thermostat, **kwargs)
        self._lsqt_range = (E_1, E_2)
        return self.add_commands(
            computes.compute_lsqt(direction, num_moments, num_energies, E_1, E_2, E_max)
        )

    # ------------------------------------------------------------ 振動分光
    def infrared(
        self,
        *,
        temperature: float,
        steps: int,
        nep_dipole: str,
        interval: int = 10,
        thermostat: str = "nvt_nhc",
        **kwargs,
    ) -> "ElectronicCalculation":
        """双極子モーメントの時系列を記録する (赤外スペクトル用)。

        ``nep_dipole`` は双極子を学習した NEP モデルのファイル。
        """
        self.nvt(temperature=temperature, steps=steps, thermostat=thermostat, **kwargs)
        self._dipole_interval = interval
        return self.add_commands(dumps.dump_dipole(interval, nep_dipole))

    def raman(
        self,
        *,
        temperature: float,
        steps: int,
        nep_polarizability: str,
        interval: int = 10,
        thermostat: str = "nvt_nhc",
        **kwargs,
    ) -> "ElectronicCalculation":
        """分極率の時系列を記録する (ラマンスペクトル用)。"""
        self.nvt(temperature=temperature, steps=steps, thermostat=thermostat, **kwargs)
        self._polarizability_interval = interval
        return self.add_commands(
            dumps.dump_polarizability(interval, nep_polarizability)
        )

    def polarization(
        self, *, temperature: float, steps: int, interval: int = 1, **kwargs
    ) -> "ElectronicCalculation":
        """Born 有効電荷から分極とその時間微分を記録する (qNEP 専用)。"""
        self.nvt(temperature=temperature, steps=steps, **kwargs)
        return self.add_commands(computes.compute_dpdt(interval))

    # ------------------------------------------------------------ 外部電場
    def electric_field(
        self,
        *,
        temperature: float,
        steps: int,
        field: Sequence[float],
        grouping_method: int = 0,
        group_id: int = 0,
        mode: str | None = None,
        **kwargs,
    ) -> "ElectronicCalculation":
        """一様電場 [V/Å] をかけた MD を仕込む。

        qNEP なら BEC との内積、それ以外なら ``model.xyz`` の
        ``charge:R:1`` との積が力になる。
        """
        self.ensure_grouping()
        self.nvt(temperature=temperature, steps=steps, **kwargs)
        return self.add_commands(
            modifiers.add_efield(grouping_method, group_id, field, mode)
        )

    def use_ewald(self) -> "ElectronicCalculation":
        """静電相互作用の逆空間項を Ewald 法にする (既定は PPPM)。"""
        return self.add_preamble(modifiers.kspace("ewald"))

    # ------------------------------------------------------------------ 結果
    def dipole(self) -> pd.DataFrame:
        return read_dipole(self.workdir / "dipole.out")

    def polarizability(self) -> pd.DataFrame:
        return read_polarizability(self.workdir / "polarizability.out")

    def dpdt(self) -> pd.DataFrame:
        return read_dpdt(self.workdir / "dpdt.out")

    def infrared_spectrum(
        self, *, interval: int | None = None, max_wavenumber: float = 4000.0
    ) -> pd.DataFrame:
        """双極子自己相関のフーリエ変換から赤外スペクトルを作る。

        強度は規格化していない (相対値)。量子補正も行わない。
        """
        interval = interval or self._dipole_interval
        if interval is None:
            raise ValueError("サンプル間隔が分かりません。interval= を指定してください。")
        return vibrational_spectrum(
            self.dipole(),
            ["mu_x", "mu_y", "mu_z"],
            time_step_fs=interval * self.builder.time_step,
            max_wavenumber=max_wavenumber,
        )

    def raman_spectrum(
        self, *, interval: int | None = None, max_wavenumber: float = 4000.0
    ) -> pd.DataFrame:
        """分極率自己相関のフーリエ変換からラマンスペクトルを作る。"""
        interval = interval or self._polarizability_interval
        if interval is None:
            raise ValueError("サンプル間隔が分かりません。interval= を指定してください。")
        return vibrational_spectrum(
            self.polarizability(),
            ["p_xx", "p_yy", "p_zz", "p_xy", "p_yz", "p_zx"],
            time_step_fs=interval * self.builder.time_step,
            max_wavenumber=max_wavenumber,
        )

    def lsqt_results(self) -> dict:
        """``lsqt_dos.out`` / ``lsqt_velocity.out`` / ``lsqt_sigma.out`` を読む。"""
        E_1, E_2 = self._lsqt_range or (None, None)
        return read_lsqt(self.workdir, E_1=E_1, E_2=E_2)

    def conductivity(self) -> pd.DataFrame:
        """LSQT の電気伝導度 sigma(E) [S/m] の時間平均。"""
        return self.lsqt_results()["sigma"]

    # ------------------------------------------------------------------ 作図
    def plot_spectrum(
        self,
        kind: str = "infrared",
        filename: str | None = None,
        *,
        dpi: int = 150,
        **kwargs,
    ) -> Path:
        """赤外またはラマンスペクトルを描く。"""
        from ..plotting import label, setup_matplotlib

        setup_matplotlib()
        import matplotlib.pyplot as plt

        if kind == "infrared":
            frame = self.infrared_spectrum(**kwargs)
            label = "赤外"
        elif kind == "raman":
            frame = self.raman_spectrum(**kwargs)
            label = "ラマン"
        else:
            raise ValueError("kind は 'infrared' か 'raman' です。")
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.plot(frame["wavenumber_cm-1"], frame["intensity"], lw=1.0)
        axis.set_xlabel(label("波数 (cm$^{-1}$)"))
        axis.set_ylabel(label("強度 (任意単位)"))
        axis.set_title(label(f"{label}スペクトル — {self.name}"))
        figure.tight_layout()
        output = self.workdir / (filename or f"{kind}_spectrum.png")
        figure.savefig(output, dpi=dpi)
        plt.close(figure)
        return output
