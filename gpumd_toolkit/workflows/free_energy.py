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

import warnings
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
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
from ..postprocess import _forward_backward as _forward_backward_table
from ..postprocess import (
    free_energy_adiabatic_switching,
    free_energy_reversible_scaling,
    gibbs_from_ti,
)

__all__ = [
    "FreeEnergyCalculation",
    "melting_point_from_curves",
    "plot_free_energy_curves",
]

#: 絶対自由エネルギーを出す TI (yaml を書くもの)
ABSOLUTE_KINDS = ("ti_spring", "ti_liquid")


class FreeEnergyCalculation(GPUMDCalculation):
    """熱力学的積分による自由エネルギー計算。

    TI のステージは ``run`` のステップ数が ``2 * (tequil + tswitch)`` で
    なければならない (往路と復路)。本クラスはそれを自動で計算する。

    ``run(analyze=True)`` では ``analysis/G_T.png`` (G–T グラフ) と
    ``analysis/G_T.csv`` を必ず書く。可逆スケーリングの基準 (T0, G0) は
    同じ計算内の ``frenkel_ladd`` / ``uhlenbeck_ford`` (T_min と同温度) から自動で取るか、
    ``reversible_scaling(reference=(T0, G0))`` で与える。
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        #: 直近に仕込んだ TI の種類 (``ti_spring`` など)
        self.ti_kind: str | None = None
        #: 仕込んだ TI の種類と条件 (G–T グラフ用)
        self.ti_stages: list[dict] = []
        self._rs_reference: tuple[float, float] | None = None
        self._as_reference: float | None = None

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
        self.ti_stages.append({"kind": spec.name, **{
            key: getattr(spec, key) for key in ("temperature", "T_min", "T_max", "p_min", "p_max")
            if hasattr(spec, key)}})
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
        reference: tuple[float, float] | None = None,
    ) -> "FreeEnergyCalculation":
        """等圧線に沿って G(T) を一気に得る (可逆スケーリング)。

        ``T_min`` での :meth:`frenkel_ladd` (または :meth:`uhlenbeck_ford`) の
        結果が基準点として必要。同じ計算内で先に仕込んでおけば自動で使う。
        別の計算で求めた場合は ``reference=(T0, G0)`` [K, eV/atom] で渡す
        (T0 = T_min)。
        """
        if reference is not None:
            self._rs_reference = (float(reference[0]), float(reference[1]))
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
        reference: float | None = None,
    ) -> "FreeEnergyCalculation":
        """等温線に沿って G(P) を得る (断熱スイッチング)。高圧相転移に使う。

        ``reference`` は ``p_min`` での G [eV/atom]。同じ計算内の
        :meth:`frenkel_ladd` / :meth:`uhlenbeck_ford` (同温度) があれば自動で使う。
        """
        if reference is not None:
            self._as_reference = float(reference)
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
        self, *, T0: float | None = None, G0: float | None = None, kind: str | None = None
    ) -> pd.DataFrame:
        """``ti_rs`` / ``ti_as`` の結果を G(T) / G(P) 曲線に変換する。

        Parameters
        ----------
        G0
            基準点の Gibbs 自由エネルギー [eV/atom]
            (``ti_spring`` / ``ti_liquid`` の結果)。省略すると ``reference=`` か
            同じ計算内の ``ti_spring`` / ``ti_liquid`` の結果を使う。
        T0
            ``ti_rs`` の場合の基準温度 [K]。
        """
        kind = self._kind(kind)
        table = self.ti_table(kind)
        if kind == "ti_rs":
            if G0 is None:
                reference = self.reference_point("ti_rs")
                if reference is None:
                    raise ValueError(
                        "ti_rs の基準 (T0, G0) がありません。reversible_scaling(reference=...)"
                        " か、同じ計算内で T_min の frenkel_ladd / uhlenbeck_ford を仕込んでください。"
                    )
                T0, G0 = reference if T0 is None else (T0, reference[1])
            if T0 is None:
                raise ValueError("ti_rs には基準温度 T0 が必要です。")
            return free_energy_reversible_scaling(table, T0=T0, G0=G0)
        if kind == "ti_as":
            if G0 is None:
                reference = self.reference_point("ti_as")
                if reference is None:
                    raise ValueError(
                        "ti_as の基準 G0 (p_min での G) がありません。"
                        " adiabatic_switching(reference=...) で与えてください。"
                    )
                G0 = reference[1]
            return free_energy_adiabatic_switching(table, G0=G0)
        raise ValueError(f"{kind} には自由エネルギー曲線がありません。")

    # ------------------------------------------------------------------ G–T
    def absolute_points(self) -> list[dict[str, float]]:
        """``ti_spring.yaml`` / ``ti_liquid.yaml`` の絶対自由エネルギー (T, F, G)。"""
        points = []
        for kind in ABSOLUTE_KINDS:
            path = self.workdir / f"{kind}.yaml"
            if path.is_file():
                data = gibbs_from_ti(read_ti_yaml(path))
                points.append({"kind": kind, "temperature": float(data["T"]),
                               "F": float(data["F"]), "G": float(data["G"])})
        return points

    def reference_point(self, kind: str) -> tuple[float, float] | None:
        """``ti_rs`` / ``ti_as`` の基準 (T0, G0)。無ければ ``None``。"""
        stage = next((s for s in self.ti_stages if s["kind"] == kind), None)
        if kind == "ti_rs" and self._rs_reference is not None:
            return self._rs_reference
        if kind == "ti_as" and self._as_reference is not None:
            return (stage["temperature"] if stage else float("nan"), self._as_reference)
        if stage is None:
            return None
        target = stage["T_min"] if kind == "ti_rs" else stage["temperature"]
        for point in self.absolute_points():
            if abs(point["temperature"] - target) < 1e-6:
                return point["temperature"], point["G"]
        return None

    def g_t_table(self) -> pd.DataFrame:
        """G–T グラフの元データ (列 ``source`` / ``temperature`` [K] / ``G`` [eV/atom])。

        ``ti_spring`` / ``ti_liquid`` の絶対値、基準 (T0, G0) がある ``ti_rs`` の G(T) 曲線、
        基準のある ``ti_as`` の出発点 (T, G0) を含む。基準の無い ``ti_rs`` は含めない
        (G0 が無いと G(T) は決まらない)。
        """
        rows = [
            {"source": p["kind"], "temperature": p["temperature"], "G": p["G"]}
            for p in self.absolute_points()
        ]
        if (self.workdir / "ti_rs.csv").is_file() and self.reference_point("ti_rs") is not None:
            curve = self.free_energy_curve(kind="ti_rs")
            rows += [
                {"source": "ti_rs", "temperature": float(t), "G": float(g)}
                for t, g in zip(curve["temperature"], curve["G"])
            ]
        if (self.workdir / "ti_as.csv").is_file():
            reference = self.reference_point("ti_as")
            if reference is not None and np.isfinite(reference[0]):
                rows.append({"source": "ti_as", "temperature": reference[0], "G": reference[1]})
        return pd.DataFrame(rows, columns=["source", "temperature", "G"])

    def plot_free_energy(
        self, filename: str = "analysis/G_T.png", *, dpi: int = 150
    ) -> Path:
        """G–T グラフを描く (``ti_as`` があれば G–P も並べる)。

        ``ti_rs`` の基準 (T0, G0) が無い場合は G(T) を決められないので、その旨を図中に
        書き、参考として ``ti_rs`` のエンタルピー H(T) を右軸に描く。
        """
        from ..plotting import label, setup_matplotlib

        setup_matplotlib()
        import matplotlib.pyplot as plt

        table = self.g_t_table()
        has_as = (self.workdir / "ti_as.csv").is_file()
        figure, axes = plt.subplots(1, 2 if has_as else 1, figsize=(11 if has_as else 6.5, 4.5))
        axes = np.atleast_1d(axes)
        axis = axes[0]
        rs = table[table["source"] == "ti_rs"].sort_values("temperature")
        if not rs.empty:
            axis.plot(rs["temperature"], rs["G"], lw=1.6, color="C0", label="ti_rs")
        markers = {"ti_spring": ("o", "C3"), "ti_liquid": ("s", "C1"), "ti_as": ("D", "C2")}
        for kind, (marker, color) in markers.items():
            points = table[table["source"] == kind]
            if not points.empty:
                axis.plot(points["temperature"], points["G"], marker, color=color, ms=7,
                          label=f"{kind} (T={points['temperature'].iloc[0]:g} K)")
        missing_rs = (self.workdir / "ti_rs.csv").is_file() and rs.empty
        if missing_rs:
            forward, _ = _forward_backward_table(self.ti_table("ti_rs"))
            stage = next((s for s in self.ti_stages if s["kind"] == "ti_rs"), None)
            T0 = stage["T_min"] if stage else float("nan")
            twin = axis.twinx()
            twin.plot(T0 / forward["lambda"], forward["enthalpy"], lw=1.0, color="0.5")
            twin.set_ylabel("ti_rs H (eV/atom)", color="0.4")
            axis.set_title(label("ti_rs の基準 (T0, G0) が未指定のため G(T) を描けません"
                                 " — reversible_scaling(reference=(T0, G0))"), fontsize=8)
            warnings.warn(
                "ti_rs の基準 (T0, G0) が無いため G(T) を描けません。"
                " reversible_scaling(reference=(T0, G0)) を指定するか、"
                "同じ計算内で T_min の frenkel_ladd / uhlenbeck_ford を仕込んでください。",
                RuntimeWarning,
            )
        if table.empty and not missing_rs:
            axis.text(0.5, 0.5, label("G–T の出力がありません"),
                      ha="center", va="center", transform=axis.transAxes)
        axis.set_xlabel("T (K)")
        axis.set_ylabel("G (eV/atom)")
        axis.grid(alpha=0.3)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8)
        if has_as:
            ax_p = axes[1]
            try:
                curve = self.free_energy_curve(kind="ti_as")
                ax_p.set_ylabel("G (eV/atom)")
            except ValueError:  # 基準が無い場合は G - G0 を描く
                curve = free_energy_adiabatic_switching(self.ti_table("ti_as"), G0=0.0)
                ax_p.set_ylabel("G - G0 (eV/atom)")
            ax_p.plot(curve["pressure_GPa"], curve["G"], lw=1.6, color="C2")
            ax_p.set_xlabel("P (GPa)")
            ax_p.grid(alpha=0.3)
        figure.suptitle(label(f"自由エネルギー — {self.name}"))
        figure.tight_layout()
        output = self.workdir / filename
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=dpi)
        plt.close(figure)
        table.to_csv(output.with_suffix(".csv"), index=False)
        return output

    def run(self, *, analyze: bool = False, **kwargs):
        """実行する。``analyze=True`` なら ``analysis/G_T.png`` / ``G_T.csv`` も書く。"""
        result = super().run(analyze=analyze, **kwargs)
        if analyze and result.succeeded:
            try:
                self.plot_free_energy()
            except Exception as exc:  # 作図の失敗で計算結果を失わない
                warnings.warn(f"G–T グラフを描けませんでした: {exc}", RuntimeWarning)
        return result

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


def plot_free_energy_curves(
    curves: Mapping[str, pd.DataFrame],
    filename: Path | str,
    *,
    points: Mapping[str, Sequence[tuple[float, float]]] | None = None,
    melting_point: bool = True,
    title: str | None = None,
    point_label: str = "direct",
    dpi: int = 150,
) -> Path:
    """複数の相の G(T) を 1 枚の G–T グラフに重ねる。

    Parameters
    ----------
    curves
        ``{"solid": 曲線, "liquid": 曲線}`` のような相名と
        :meth:`FreeEnergyCalculation.free_energy_curve` (列 ``temperature`` / ``G``) の対応。
    points
        直接計算した点 ``{"solid": [(T, G), ...]}`` (曲線の検算用)。
        凡例には ``"<相名> (<point_label>)"`` と出る。
    melting_point
        曲線が 2 本なら交点 (:func:`melting_point_from_curves`) に縦線を引く。
    """
    from ..plotting import label, setup_matplotlib

    setup_matplotlib()
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4.5))
    colors = {}
    for i, (name, curve) in enumerate(curves.items()):
        ordered = curve.sort_values("temperature")
        (line,) = axis.plot(ordered["temperature"], ordered["G"], lw=1.6, label=name)
        colors[name] = line.get_color()
    for name, values in (points or {}).items():
        values = list(values)
        if values:
            axis.plot([t for t, _ in values], [g for _, g in values], "o",
                      color=colors.get(name), ms=6, mfc="white", label=f"{name} ({point_label})")
    names = list(curves)
    if melting_point and len(names) == 2:
        try:
            tm = melting_point_from_curves(curves[names[0]], curves[names[1]])
            axis.axvline(tm, color="k", ls="--", lw=0.9, label=f"crossing T = {tm:.0f} K")
        except ValueError as exc:
            warnings.warn(str(exc), RuntimeWarning)
    axis.set_xlabel("T (K)")
    axis.set_ylabel("G (eV/atom)")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    if title:
        axis.set_title(label(title))
    figure.tight_layout()
    output = Path(filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return output
