"""3. 自由エネルギー・熱力学的積分.

GPUMD は非平衡熱力学的積分 (nonequilibrium TI) を 4 通り持っている。

============================ ==================================================
経路                         用途
============================ ==================================================
Frenkel-Ladd (``ti_spring``) 固体の絶対自由エネルギー (参照系 = Einstein 結晶)
Uhlenbeck-Ford (``ti_liquid``) 液体の絶対自由エネルギー (参照系 = UF 流体)
可逆スケーリング (``ti_rs``)  等圧線に沿った G(T) — 融点・相境界に使う
断熱スイッチング (``ti_as``)  等温線に沿った G(P) — 高圧相転移に使う
============================ ==================================================

典型的な流れは
「``ti_spring`` で基準点 T0 の G を出す → ``ti_rs`` で G(T) に広げる →
2 相の G(T) が交わる温度が融点」。

Examples
--------
>>> from gpumd_toolkit.workflows import FreeEnergyCalculation
>>> calc = FreeEnergyCalculation("POSCAR", "nep.txt", "runs/fe_solid")
>>> calc.equilibrate(temperature=300, steps=20000)
>>> calc.frenkel_ladd(temperature=300, t_equil=5000, t_switch=20000)
>>> calc.run()
>>> calc.gibbs_free_energy()
{'E_Einstein': ..., 'F': ..., 'G': ...}
"""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

from ..inputs.builder import MDStage
from ..inputs.ensembles import (
    TIAdiabaticSwitching,
    TIFixedLambda,
    TILiquid,
    TIReversibleScaling,
    TISpring,
)
from ..md import GPUMDCalculation
from ..outputs import read_ti_csv, read_ti_yaml
from ..postprocess import (
    free_energy_adiabatic_switching,
    free_energy_reversible_scaling,
    gibbs_from_ti,
)

__all__ = ["FreeEnergyCalculation", "melting_point_from_curves"]


class FreeEnergyCalculation(GPUMDCalculation):
    """熱力学的積分による自由エネルギー計算。

    TI のステージは ``run`` のステップ数が ``2 * (tequil + tswitch)`` で
    なければならない (往路と復路)。本クラスはそれを自動で計算する。
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        #: 直近に仕込んだ TI の種類 (``ti_spring`` など)
        self.ti_kind: str | None = None

    # ------------------------------------------------------------------ 準備
    def equilibrate(
        self,
        *,
        temperature: float,
        steps: int,
        pressure: float | None = None,
        thermostat: str = "nvt_nhc",
        barostat: str = "npt_scr",
        **kwargs,
    ) -> "FreeEnergyCalculation":
        """TI の前に系を平衡化する (これを入れないと結果が偏る)。

        ``pressure`` を与えると NPT で平衡化してから TI に入る。
        """
        if pressure is None:
            self.nvt(temperature=temperature, steps=steps, thermostat=thermostat, **kwargs)
        else:
            self.npt(
                temperature=temperature,
                steps=steps,
                pressure=pressure,
                barostat=barostat,
                **kwargs,
            )
        return self

    # ------------------------------------------------------------------ TI
    def _add_ti(self, spec, t_equil: int, t_switch: int) -> "FreeEnergyCalculation":
        steps = 2 * (int(t_equil) + int(t_switch))
        self.ti_kind = spec.name
        self.add_stage(MDStage(spec, steps=steps, label=f"{spec.name}"))
        return self

    def frenkel_ladd(
        self,
        *,
        temperature: float,
        t_equil: int = 5000,
        t_switch: int = 20000,
        pressure: float = 0.0,
        spring: Mapping[str, float] | None = None,
        tau_T: float | None = None,
        T_coup: float = 100.0,
    ) -> "FreeEnergyCalculation":
        """固体の絶対自由エネルギー (Einstein 結晶を参照系とする Frenkel-Ladd)。

        Parameters
        ----------
        spring
            元素ごとのばね定数 [eV/Å^2]。``None`` なら GPUMD が MSD から
            自動で決める (ふつうはこれでよい)。
        pressure
            Gibbs 自由エネルギー G = F + PV を出すための圧力 [GPa]。
            ダイナミクスには影響しない。
        """
        spec = TISpring(
            t_equil=t_equil,
            t_switch=t_switch,
            temperature=temperature,
            pressure=pressure,
            T_coup=T_coup,
            tau_T=tau_T,
            spring=spring,
        )
        return self._add_ti(spec, t_equil, t_switch)

    def uhlenbeck_ford(
        self,
        *,
        temperature: float,
        t_equil: int = 5000,
        t_switch: int = 20000,
        pressure: float = 0.0,
        sigma_sqrd: float = 2.0,
        p: int = 100,
        tau_T: float | None = None,
        T_coup: float = 100.0,
    ) -> "FreeEnergyCalculation":
        """液体の絶対自由エネルギー (Uhlenbeck-Ford 流体を参照系とする)。"""
        spec = TILiquid(
            t_equil=t_equil,
            t_switch=t_switch,
            temperature=temperature,
            pressure=pressure,
            T_coup=T_coup,
            tau_T=tau_T,
            sigma_sqrd=sigma_sqrd,
            p=p,
        )
        return self._add_ti(spec, t_equil, t_switch)

    def reversible_scaling(
        self,
        *,
        T_min: float,
        T_max: float,
        pressure: float = 0.0,
        direction: str | Sequence[str] = "aniso",
        t_equil: int = 5000,
        t_switch: int = 20000,
        tau_T: float | None = None,
        tau_p: float | None = None,
    ) -> "FreeEnergyCalculation":
        """等圧線に沿って G(T) を一気に得る (可逆スケーリング)。

        別途 ``T_min`` での :meth:`frenkel_ladd` (または
        :meth:`uhlenbeck_ford`) の結果が基準点として必要。
        """
        spec = TIReversibleScaling(
            t_equil=t_equil,
            t_switch=t_switch,
            T_min=T_min,
            T_max=T_max,
            pressure=pressure,
            direction=direction,
            tau_T=tau_T,
            tau_p=tau_p,
        )
        return self._add_ti(spec, t_equil, t_switch)

    def adiabatic_switching(
        self,
        *,
        temperature: float,
        p_min: float,
        p_max: float,
        direction: str | Sequence[str] = "aniso",
        t_equil: int = 5000,
        t_switch: int = 20000,
        tau_T: float | None = None,
        tau_p: float | None = None,
    ) -> "FreeEnergyCalculation":
        """等温線に沿って G(P) を得る (断熱スイッチング)。高圧相転移に使う。"""
        spec = TIAdiabaticSwitching(
            t_equil=t_equil,
            t_switch=t_switch,
            temperature=temperature,
            p_min=p_min,
            p_max=p_max,
            direction=direction,
            tau_T=tau_T,
            tau_p=tau_p,
        )
        return self._add_ti(spec, t_equil, t_switch)

    def fixed_lambda(
        self,
        *,
        lambda_value: float,
        temperature: float,
        steps: int,
        spring: Mapping[str, float] | None = None,
        tau_T: float | None = None,
    ) -> "FreeEnergyCalculation":
        """lambda を固定した平衡 TI (実装確認・収束テスト用)。"""
        spec = TIFixedLambda(
            lambda_value=lambda_value,
            temperature=temperature,
            tau_T=tau_T,
            spring=spring,
        )
        self.ti_kind = spec.name
        return self.add_stage(MDStage(spec, steps=steps))

    # ------------------------------------------------------------------ 結果
    def _kind(self, kind: str | None) -> str:
        kind = kind or self.ti_kind
        if kind is None:
            raise ValueError("TI のステージがありません。frenkel_ladd() などを呼んでください。")
        return kind

    def ti_table(self, kind: str | None = None) -> pd.DataFrame:
        """``ti_*.csv`` を読む。"""
        return read_ti_csv(self.workdir / f"{self._kind(kind)}.csv")

    def gibbs_free_energy(self, kind: str | None = None) -> dict[str, float]:
        """``ti_spring.yaml`` / ``ti_liquid.yaml`` を読む。

        戻り値は eV/atom 単位で、``F`` が Helmholtz、``G`` が Gibbs 自由エネルギー。
        ``P_GPa`` を派生キーとして足す。
        """
        kind = self._kind(kind)
        if kind not in ("ti_spring", "ti_liquid"):
            raise ValueError(
                f"{kind} は絶対自由エネルギーを出しません。"
                " ti_rs / ti_as は free_energy_curve() を使ってください。"
            )
        return gibbs_from_ti(read_ti_yaml(self.workdir / f"{kind}.yaml"))

    def free_energy_curve(
        self, *, T0: float | None = None, G0: float, kind: str | None = None
    ) -> pd.DataFrame:
        """``ti_rs`` / ``ti_as`` の結果を G(T) / G(P) 曲線に変換する。

        Parameters
        ----------
        G0
            基準点の Gibbs 自由エネルギー [eV/atom]
            (``ti_spring`` / ``ti_liquid`` の結果)。
        T0
            ``ti_rs`` の場合の基準温度 [K]。
        """
        kind = self._kind(kind)
        table = self.ti_table(kind)
        if kind == "ti_rs":
            if T0 is None:
                raise ValueError("ti_rs には基準温度 T0 が必要です。")
            return free_energy_reversible_scaling(table, T0=T0, G0=G0)
        if kind == "ti_as":
            return free_energy_adiabatic_switching(table, G0=G0)
        raise ValueError(f"{kind} には自由エネルギー曲線がありません。")

    def hysteresis(self, kind: str | None = None) -> dict[str, float]:
        """往路と復路のずれ (非平衡性の指標) を返す。小さいほど信頼できる。

        大きい場合は ``t_switch`` を伸ばす (スイッチをゆっくりする)。
        """
        from ..postprocess import _forward_backward

        table = self.ti_table(kind)
        column = (
            "enthalpy" if "enthalpy" in table
            else ("V" if "V" in table else ("pe" if "pe" in table else table.columns[-1]))
        )
        forward_frame, backward_frame = _forward_backward(table)
        forward = forward_frame[column].to_numpy()
        backward = backward_frame[column].to_numpy()
        difference = forward - backward
        return {
            "column": column,
            "max_abs": float(abs(difference).max()),
            "rms": float((difference**2).mean() ** 0.5),
            "mean": float(difference.mean()),
        }


def melting_point_from_curves(
    solid: pd.DataFrame, liquid: pd.DataFrame, *, column: str = "temperature"
) -> float:
    """固相・液相の G(T) 曲線が交わる温度 [K] を返す (二相法の代替)。

    Parameters
    ----------
    solid, liquid
        :meth:`FreeEnergyCalculation.free_energy_curve` の戻り値。
    """
    import numpy as np

    T = np.linspace(
        max(solid[column].min(), liquid[column].min()),
        min(solid[column].max(), liquid[column].max()),
        2001,
    )
    if T.size < 2 or T[0] >= T[-1]:
        raise ValueError("2 つの曲線の温度範囲が重なっていません。")
    gs = np.interp(T, solid[column], solid["G"])
    gl = np.interp(T, liquid[column], liquid["G"])
    difference = gs - gl
    crossings = np.flatnonzero(np.sign(difference[:-1]) != np.sign(difference[1:]))
    if crossings.size == 0:
        raise ValueError(
            "G(T) 曲線が交差しません。温度範囲を広げるか、"
            " 参照自由エネルギー G0 を確認してください。"
        )
    i = int(crossings[0])
    # 線形内挿
    t0, t1 = T[i], T[i + 1]
    d0, d1 = difference[i], difference[i + 1]
    return float(t0 - d0 * (t1 - t0) / (d1 - d0))
