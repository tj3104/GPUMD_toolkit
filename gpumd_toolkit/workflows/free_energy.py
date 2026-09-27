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

``ti_rs`` は ``T_max < T_min`` とすれば高温から降温する向きにも走る
(液体を高温の Uhlenbeck-Ford から冷やす場合など)。
スキャン中の相転移 (融解・凝固・蒸発・凝縮・昇華・凝華) は H(T) の段差から検出して
``analysis/rs_phases.*`` にまとめ、``transition_guard`` で相転移を跨がないよう調整できる
(:meth:`FreeEnergyCalculation.reversible_scaling`)。

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

import copy
import dataclasses
import json
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
from ..thermodynamics import (
    TransitionCriteria,
    detect_phase_transitions,
    format_transitions,
    hysteresis_melting_estimate,
    plot_rs_phases,
    rs_phase_transitions,
    rs_safe_limit,
    rs_valid_range,
)

__all__ = [
    "FreeEnergyCalculation",
    "melting_point_from_curves",
    "plot_free_energy_curves",
]

#: 絶対自由エネルギーを出す TI (yaml を書くもの)
ABSOLUTE_KINDS = ("ti_spring", "ti_liquid")
#: 可逆スケーリングの相転移ガード
GUARD_MODES = ("off", "truncate", "rerun", "precheck")
#: ガードの状態を残すファイル (計算をやり直さずに読み直す用)
GUARD_FILE = "analysis/rs_guard.json"


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
        #: 可逆スケーリングの相転移ガードの設定
        self._rs_guard: dict = {"mode": "off"}

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
        transition_guard: str = "off",
        initial_phase: str | None = None,
        guard_margin: float = 0.03,
        max_rerun: int = 3,
        guard_shrink: float = 0.15,
        precheck_steps: int | None = None,
        precheck_barostat: str = "npt_scr",
        criteria: TransitionCriteria | None = None,
    ) -> "FreeEnergyCalculation":
        """等圧線に沿って G(T) を一気に得る (可逆スケーリング)。

        ``T_min`` での :meth:`frenkel_ladd` (または :meth:`uhlenbeck_ford`) の
        結果が基準点として必要。同じ計算内で先に仕込んでおけば自動で使う。
        別の計算で求めた場合は ``reference=(T0, G0)`` [K, eV/atom] で渡す
        (T0 = T_min)。

        ``T_min`` はスキャンの **出発温度** (熱浴の温度) で、``T_max`` が終点。
        ``T_max < T_min`` なら高温から降温する向きのスキャンになる
        (例: 液体を 5000 K の Uhlenbeck-Ford から 1000 K まで)。

        Parameters
        ----------
        transition_guard
            スキャンが相転移を跨がないようにする調整機能。
            H(T) の段差 (潜熱) と転移エントロピー ΔS = ΔH/T から相転移を検出する。

            ``'off'`` (既定)
                何もしない (転移を検出したら警告だけ出す)。
            ``'truncate'``
                実行後、T0 側で最初の転移区間の手前 (``guard_margin`` だけ内側) までに
                G(T) を切り詰める。再計算はしない。
            ``'rerun'``
                転移を検出したら終点をその手前にして ``ti_rs`` を
                ``<workdir>/rs_guard_<n>`` で撮り直す (最大 ``max_rerun`` 回)。
                往路に段差が無いのに復路が出発相に戻らない (折り返し点の保持中に
                転移した) 場合は、スキャン幅を ``guard_shrink`` (相対値) だけ縮めて撮り直す。
                基準 (T0, G0) は元の計算のものを使う。
            ``'precheck'``
                本番の前に同じ温度範囲の短い NPT ランプ (``<workdir>/rs_precheck``、
                ``precheck_steps`` ステップ、既定 ``t_switch``) で転移を探し、
                終点を転移の手前にしてから本番を走らせる。

            ``'off'`` 以外では、実行後に残った転移も切り詰めで除く。
            復路が出発相に戻っていなければ (過冷却液体のまま凝固しなかったなど)、
            G(T) を往路だけから求める (``free_energy_reversible_scaling(branch='forward')``)。
        initial_phase
            T_min での相 (``'solid'`` / ``'liquid'`` / ``'gas'``)。転移の分類に使う。
            ``None`` なら同じ計算内の基準 (``frenkel_ladd`` → 固相、
            ``uhlenbeck_ford`` → 液相) から、無ければ昇温なら固相・降温なら液相とみなす。
        guard_margin
            転移区間の端から T0 側へ引く余裕 (相対値、0.03 = 3%)。
        criteria
            転移とみなす基準 (:class:`~gpumd_toolkit.thermodynamics.TransitionCriteria`)。
        """
        if transition_guard not in GUARD_MODES:
            raise ValueError(f"transition_guard は {GUARD_MODES} から選びます。")
        if T_min == T_max:
            raise ValueError("T_min と T_max が同じです (スキャンになりません)。")
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
        self._rs_guard = {
            "mode": transition_guard,
            "initial_phase": initial_phase,
            "margin": float(guard_margin),
            "max_rerun": int(max_rerun),
            "shrink": float(guard_shrink),
            "precheck_steps": int(precheck_steps or t_switch),
            "precheck_barostat": precheck_barostat,
            "criteria": criteria or TransitionCriteria(),
            "T_max_requested": float(T_max),
        }
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
        """``ti_*.csv`` を読む (``ti_rs`` はガードで撮り直していればそちら)。"""
        kind = self._kind(kind)
        if kind == "ti_rs":
            return read_ti_csv(self.rs_directory() / "ti_rs.csv")
        return read_ti_csv(self.workdir / f"{kind}.csv")

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
        self,
        *,
        T0: float | None = None,
        G0: float | None = None,
        kind: str | None = None,
        truncate: bool | None = None,
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
        truncate
            ``ti_rs`` の G(T) を、スキャン中の相転移の手前
            (:meth:`rs_temperature_limit`) までに切り詰めるか。
            ``None`` なら ``transition_guard`` が ``'off'`` 以外のとき切り詰める。
            ``'off'`` で転移を跨いでいる場合は警告を出す。
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
            if truncate is None:
                truncate = self._rs_guard.get("mode", "off") != "off"
            if not truncate:
                if self._rs_guard.get("mode", "off") == "off":
                    self._warn_rs_validity()
                return free_energy_reversible_scaling(table, T0=T0, G0=G0)
            valid = self.rs_valid_range()
            if valid["branch"] != "both":
                warnings.warn(f"ti_rs: {valid['reason']}。", RuntimeWarning)
            curve = free_energy_reversible_scaling(table, T0=T0, G0=G0, branch=valid["branch"])
            if valid["limit"] is not None:
                lo, hi = sorted((self.rs_start_temperature(), valid["limit"]))
                curve = curve[(curve["temperature"] >= lo - 1e-9)
                              & (curve["temperature"] <= hi + 1e-9)].reset_index(drop=True)
            return curve
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

    # ------------------------------------------------------- 可逆スケーリングの相転移
    def _rs_stage_index(self) -> int | None:
        for i, stage in enumerate(self.stages):
            if isinstance(stage.ensemble_spec, TIReversibleScaling):
                return i
        return None

    def _rs_spec(self) -> TIReversibleScaling:
        index = self._rs_stage_index()
        if index is None:
            raise ValueError("ti_rs のステージがありません。reversible_scaling() を呼んでください。")
        return self.stages[index].ensemble_spec

    def rs_start_temperature(self) -> float:
        """スキャンの出発温度 T0 (= ``T_min``)。"""
        return float(self._rs_spec().T_min)

    def rs_initial_phase(self) -> str:
        """T0 での相 (転移の分類に使う)。"""
        phase = self._rs_guard.get("initial_phase")
        if phase:
            return phase
        T0 = self.rs_start_temperature()
        for stage in self.ti_stages:
            if stage.get("temperature") is not None and abs(stage["temperature"] - T0) < 1e-6:
                if stage["kind"] == "ti_spring":
                    return "solid"
                if stage["kind"] == "ti_liquid":
                    return "liquid"
        spec = self._rs_spec()
        return "solid" if spec.T_max > spec.T_min else "liquid"

    def _guard_path(self) -> Path:
        return self.workdir / GUARD_FILE

    def guard_state(self) -> dict:
        """ガードの記録 (``analysis/rs_guard.json``)。無ければ空。"""
        path = self._guard_path()
        if path.is_file():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_guard_state(self, **updates) -> dict:
        state = {**self.guard_state(), **updates, "mode": self._rs_guard.get("mode", "off")}
        path = self._guard_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, ensure_ascii=False, default=float))
        return state

    def rs_directory(self) -> Path:
        """最終的に使う ``ti_rs.csv`` のあるディレクトリ (撮り直していればそちら)。"""
        rs_dir = self.guard_state().get("rs_dir")
        if rs_dir and (Path(rs_dir) / "ti_rs.csv").is_file():
            return Path(rs_dir)
        return self.workdir

    def rs_end_temperature(self) -> float:
        """最終的に使うスキャンの終点温度。"""
        state = self.guard_state()
        if state.get("rs_dir") and state.get("T_max_used") is not None:
            return float(state["T_max_used"])
        return float(self._rs_spec().T_max)

    def rs_phase_transitions(self, criteria: TransitionCriteria | None = None) -> pd.DataFrame:
        """``ti_rs`` の往路・復路で起きた相転移
        (:func:`~gpumd_toolkit.thermodynamics.rs_phase_transitions`)。"""
        return rs_phase_transitions(
            self.ti_table("ti_rs"), T0=self.rs_start_temperature(),
            initial_phase=self.rs_initial_phase(),
            criteria=criteria or self._rs_guard.get("criteria"))

    def rs_valid_range(self) -> dict:
        """G(T) を信用できる範囲と使う経路
        (:func:`~gpumd_toolkit.thermodynamics.rs_valid_range`)。"""
        return rs_valid_range(
            self.ti_table("ti_rs"), self.rs_phase_transitions(),
            T0=self.rs_start_temperature(), T_end=self.rs_end_temperature(),
            margin=self._rs_guard.get("margin", 0.03), criteria=self._rs_guard.get("criteria"))

    def rs_temperature_limit(self) -> float | None:
        """相転移を跨がない温度範囲の端 (全域が有効なら ``None``)。"""
        return self.rs_valid_range()["limit"]

    def _warn_rs_validity(self) -> None:
        """調整機能 off のとき、相転移を跨いでいれば警告する。"""
        try:
            transitions = self.rs_phase_transitions()
            valid = self.rs_valid_range()
        except (ValueError, FileNotFoundError):
            return
        items = ", ".join(f"{r.pass_} {r.process_ja} {r.T_K:.0f} K" for r in
                          transitions.rename(columns={"pass": "pass_"}).itertuples())
        if len(transitions) or not valid["backward_returns"]:
            detail = items or valid["reason"]
            if not valid["backward_returns"] and items:
                detail += f"; {valid['reason']}"
            warnings.warn(
                f"ti_rs: 相転移を跨いでいます ({detail})。その先の G(T) は別の相・非平衡の値です。"
                " transition_guard='truncate' / 'rerun' / 'precheck' を検討してください。",
                RuntimeWarning)

    def phase_summary(self, prefix: str = "analysis/rs_phases", *, dpi: int = 150) -> dict:
        """可逆スケーリングで起きた融解・蒸発・凝固・昇華などをまとめる。

        ``<prefix>.csv`` (転移の表)、``<prefix>.md`` (読みやすい要約)、
        ``<prefix>.png`` (H(T) の往路・復路と転移区間、G(T) の有効範囲) を書き、
        要約の dict を返す。
        """
        table = self.ti_table("ti_rs")
        T0 = self.rs_start_temperature()
        transitions = self.rs_phase_transitions()
        valid = self.rs_valid_range()
        limit = valid["limit"]
        forward = transitions[transitions["pass"] == "forward"] if len(transitions) else transitions
        backward = transitions[transitions["pass"] == "backward"] if len(transitions) else transitions
        summary: dict = {
            "T_start_K": T0,
            "T_end_K": self.rs_end_temperature(),
            "T_end_requested_K": self._rs_guard.get("T_max_requested"),
            "initial_phase": self.rs_initial_phase(),
            "guard_mode": self._rs_guard.get("mode", "off"),
            "rs_directory": str(self.rs_directory()),
            "n_transitions": int(len(transitions)),
            "safe_limit_K": limit,
            "branch": valid["branch"],
            "backward_returns": valid["backward_returns"],
            "dH_backward_minus_forward_at_T0": valid["dH_at_T0"],
            "processes": [f"{r.pass_}:{r.process}" for r in
                          transitions.rename(columns={"pass": "pass_"}).itertuples()],
        }
        # 昇温での融解と降温での凝固が両方あれば、ヒステリシスから融点を見積もる
        heat = transitions[(transitions.get("process") == "melting")] if len(transitions) else []
        cool = transitions[(transitions.get("process") == "freezing")] if len(transitions) else []
        # 復路が出発相に戻っていなければ、復路の「凝固」は結晶化ではない (目安にならない)
        if len(heat) and len(cool) and valid["backward_returns"]:
            summary["T_melting_heating_K"] = float(heat["T_K"].iloc[0])
            summary["T_freezing_cooling_K"] = float(cool["T_K"].iloc[0])
            summary["Tm_hysteresis_estimate_K"] = hysteresis_melting_estimate(
                summary["T_melting_heating_K"], summary["T_freezing_cooling_K"])
        curve = None
        reference = self.reference_point("ti_rs")
        if reference is not None:
            curve = free_energy_reversible_scaling(table, T0=T0, G0=reference[1],
                                                   branch=valid["branch"])
        output = self.workdir / prefix
        output.parent.mkdir(parents=True, exist_ok=True)
        transitions.to_csv(output.with_suffix(".csv"), index=False)
        direction = "昇温" if self.rs_end_temperature() > T0 else "降温"
        lines = [
            f"# 可逆スケーリングの相転移 — {self.name}", "",
            f"- スキャン: {T0:g} K → {self.rs_end_temperature():g} K ({direction}、往路)"
            f" と復路、出発相: {self.rs_initial_phase()}",
            f"- 調整機能 (transition_guard): {summary['guard_mode']}",
        ]
        if summary["T_end_requested_K"] not in (None, self.rs_end_temperature()):
            lines.append(f"- 終点を {summary['T_end_requested_K']:g} K から "
                         f"{self.rs_end_temperature():g} K に調整 ({self.rs_directory()})")
        lines.append(
            f"- 往路: {', '.join(forward['process_ja']) if len(forward) else '転移なし'} / "
            f"復路: {', '.join(backward['process_ja']) if len(backward) else '転移なし'}")
        if not valid["backward_returns"]:
            lines.append(
                f"- 復路は出発相に戻っていません (T0 付近で H_復路 − H_往路 = "
                f"{valid['dH_at_T0']:+.4f} eV/atom)。折り返し点の保持中などに転移して"
                " 戻らなかったため、G(T) は往路のみから求めます。")
        if limit is not None:
            lo, hi = sorted((T0, limit))
            lines.append(f"- 相転移を跨がない範囲: {lo:.0f}–{hi:.0f} K"
                         " (この外の G(T) は別の相・非平衡の値)")
        elif valid["backward_returns"]:
            lines.append("- スキャン全域で往路と復路が同じ相にいます (G(T) は全域で有効)。")
        else:
            lines.append("- 往路には転移が無いので、往路の G(T) は全域で有効です。")
        if "Tm_hysteresis_estimate_K" in summary:
            lines.append(
                f"- ヒステリシスからの融点の目安: {summary['Tm_hysteresis_estimate_K']:.0f} K "
                f"(昇温 {summary['T_melting_heating_K']:.0f} K / 降温 "
                f"{summary['T_freezing_cooling_K']:.0f} K、Tm = T+ + T- - sqrt(T+ T-))")
        lines += ["", format_transitions(transitions, title="検出した相転移")]
        output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
        png = plot_rs_phases(table, transitions, output.with_suffix(".png"), T0=T0, limit=limit,
                             curve=curve, title=f"可逆スケーリングの相転移 — {self.name}", dpi=dpi)
        (output.parent / f"{output.name}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=float))
        return {**summary, "csv": output.with_suffix(".csv"), "markdown": output.with_suffix(".md"),
                "png": png}

    def _sub_calculation(self, workdir: Path, stages: list) -> "FreeEnergyCalculation":
        """同じ構造・ポテンシャル・設定で、ステージだけ差し替えた計算を作る。"""
        sub = FreeEnergyCalculation(
            self.atoms, self.potential, workdir, overwrite=True, name=f"{self.name}/{workdir.name}",
            time_step=self.builder.time_step, environment=self.environment,
            groupings=self.groupings, max_atoms=max(self.max_atoms, self.n_atoms),
            save_final_structure=False,
        )
        builder = copy.deepcopy(self.builder)
        builder.stages = stages
        builder.final_restart = False
        sub.builder = builder
        return sub

    def _stages_before_rs(self) -> list:
        """ti_rs より前の平衡化ステージ (TI 以外) の複製。"""
        index = self._rs_stage_index()
        return [copy.deepcopy(stage) for stage in self.stages[:index]
                if not stage.ensemble_name.startswith("ti")]

    def _run_precheck(self, *, timeout: float | None = None, check: bool = True) -> dict:
        """短い NPT ランプで相転移を探し、ti_rs の終点を転移の手前に寄せる。"""
        guard = self._rs_guard
        index = self._rs_stage_index()
        spec = self._rs_spec()
        steps = guard["precheck_steps"]
        pre = self._sub_calculation(self.workdir / "rs_precheck", self._stages_before_rs())
        pre.set_dump(thermo=max(1, steps // 2000), traj=None)
        pre.npt(temperature=spec.T_min, temperature_end=guard["T_max_requested"], steps=steps,
                pressure=spec.pressure, barostat=guard["precheck_barostat"])
        result = pre.run(timeout=timeout, check=check)
        state: dict = {"precheck_dir": str(pre.workdir), "precheck_ok": result.succeeded}
        if not result.succeeded:
            warnings.warn("相転移の事前チェックが失敗しました。ti_rs は指定どおり走らせます。",
                          RuntimeWarning)
            return self._save_guard_state(**state)
        from ..analysis import ThermoData

        frame = ThermoData.from_directory(pre.workdir, n_atoms=self.n_atoms).frame
        ramp = frame[frame["stage"] == frame["stage"].max()]
        pressure = float(ramp["pressure"].mean())
        enthalpy = (ramp["total_energy_per_atom"]
                    + pressure * ramp["volume_per_atom"] / 160.21766208)
        direction = "heating" if guard["T_max_requested"] > spec.T_min else "cooling"
        transitions = detect_phase_transitions(
            ramp["target_temperature"], enthalpy, volume=ramp["volume_per_atom"],
            direction=direction, initial_phase=self.rs_initial_phase(),
            criteria=guard["criteria"])
        transitions.to_csv(pre.workdir / "transitions.csv", index=False)
        limit = rs_safe_limit(transitions, T0=spec.T_min, T_end=guard["T_max_requested"],
                              margin=guard["margin"])
        state["precheck_transitions"] = transitions.to_dict("records")
        if limit is not None and abs(limit - spec.T_min) > 1.0:
            new_spec = dataclasses.replace(spec, T_max=limit)
            stage = self.stages[index]
            stage.ensemble = new_spec
            stage.label = new_spec.describe()
            for record in self.ti_stages:
                if record["kind"] == "ti_rs":
                    record["T_max"] = limit
            items = ", ".join(f"{r.process_ja} {r.T_K:.0f} K" for r in transitions.itertuples())
            warnings.warn(f"事前チェックで相転移を検出 ({items})。ti_rs の終点を "
                          f"{guard['T_max_requested']:g} K → {limit:.0f} K にしました。",
                          RuntimeWarning)
            state["T_max_used"] = limit
        elif limit is not None:
            warnings.warn("事前チェックで出発温度のすぐ近くに相転移があり、終点を調整できません。",
                          RuntimeWarning)
        return self._save_guard_state(**state)

    def _rerun_guard(self, *, timeout: float | None = None, check: bool = True) -> dict:
        """転移を跨いでいたら終点を手前にして ti_rs を撮り直す。"""
        guard = self._rs_guard
        spec = self._rs_spec()
        reference = self.reference_point("ti_rs")
        state = self._save_guard_state(rs_dir=None, T_max_used=None, iterations=0)
        for iteration in range(1, guard["max_rerun"] + 1):
            transitions = self.rs_phase_transitions()
            valid = self.rs_valid_range()
            T_end = self.rs_end_temperature()
            if valid["limit"] is not None:
                limit = valid["limit"]
            elif not valid["backward_returns"]:
                # 往路に段差が無いのに戻らない = 折り返し点の保持中に転移した
                limit = spec.T_min + (1.0 - guard["shrink"]) * (T_end - spec.T_min)
            else:
                break
            if abs(limit - spec.T_min) < 0.02 * abs(guard["T_max_requested"] - spec.T_min):
                warnings.warn("出発温度のすぐ近くに相転移があり、撮り直しても範囲がほぼ残りません。",
                              RuntimeWarning)
                break
            items = ", ".join(f"{r.process_ja} {r.T_K:.0f} K" for r in transitions.itertuples())
            if not valid["backward_returns"]:
                items = ", ".join(filter(None, [items, "復路が出発相に戻らない"]))
            warnings.warn(f"ti_rs が相転移を跨いでいます ({items})。終点を {limit:.0f} K にして"
                          f"撮り直します ({iteration}/{guard['max_rerun']})。", RuntimeWarning)
            stages = self._stages_before_rs()
            sub = self._sub_calculation(self.workdir / f"rs_guard_{iteration}", stages)
            sub.reversible_scaling(
                T_min=spec.T_min, T_max=limit, pressure=spec.pressure, direction=spec.direction,
                t_equil=spec.t_equil, t_switch=spec.t_switch, tau_T=spec.tau_T, tau_p=spec.tau_p,
                reference=reference, initial_phase=self.rs_initial_phase())
            sub.stages[-1].ensemble = dataclasses.replace(
                spec, T_max=limit)  # T_coup / p_period も元のまま
            result = sub.run(timeout=timeout, check=check)
            if not result.succeeded:
                warnings.warn("ti_rs の撮り直しが失敗しました。", RuntimeWarning)
                break
            state = self._save_guard_state(rs_dir=str(sub.workdir), T_max_used=limit,
                                           iterations=iteration)
        return state

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
        """実行する。

        ``analyze=True`` なら ``analysis/G_T.png`` / ``G_T.csv`` と、``ti_rs`` があれば
        相転移のまとめ (``analysis/rs_phases.{md,csv,png,json}``) も書く。
        ``reversible_scaling(transition_guard=...)`` の事前チェック・撮り直しもここで行う。
        """
        mode = self._rs_guard.get("mode", "off")
        dry_run = kwargs.get("dry_run", False)
        has_rs = self._rs_stage_index() is not None
        if has_rs and not dry_run:
            # 前回のガードの記録 (撮り直しのディレクトリなど) を引き継がない
            self._guard_path().unlink(missing_ok=True)
        if has_rs and mode == "precheck" and not dry_run:
            self._run_precheck(timeout=kwargs.get("timeout"), check=kwargs.get("check", True))
        result = super().run(analyze=analyze, **kwargs)
        if has_rs and result.succeeded and mode == "rerun":
            self._rerun_guard(timeout=kwargs.get("timeout"), check=kwargs.get("check", True))
        if analyze and result.succeeded:
            try:
                self.plot_free_energy()
            except Exception as exc:  # 作図の失敗で計算結果を失わない
                warnings.warn(f"G–T グラフを描けませんでした: {exc}", RuntimeWarning)
            if has_rs:
                try:
                    self.phase_summary()
                except Exception as exc:
                    warnings.warn(f"相転移のまとめを作れませんでした: {exc}", RuntimeWarning)
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


def _ascending_curve(curve: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    """G(T) を温度の昇順に並べ替え、同じ温度は平均する (np.interp は昇順が前提)。"""
    frame = curve[[column, "G"]].dropna()
    frame = frame.groupby(column, as_index=False, sort=True)["G"].mean()
    return frame[column].to_numpy(dtype=float), frame["G"].to_numpy(dtype=float)


def melting_point_from_curves(
    solid: pd.DataFrame, liquid: pd.DataFrame, *, column: str = "temperature"
) -> float:
    """固相・液相の G(T) 曲線が交わる温度 [K] を返す (二相法の代替)。

    曲線は温度の昇順・降順どちらでもよい (液体を 5000→1000 K と降温スキャンした
    ``ti_rs`` など)。内部で昇順に並べ替えてから内挿するので、結果は
    可逆スケーリングのスキャン方向に依存しない。交点が複数あれば最も低温のものを返す。

    Parameters
    ----------
    solid, liquid
        :meth:`FreeEnergyCalculation.free_energy_curve` の戻り値。
    """
    Ts, Gs = _ascending_curve(solid, column)
    Tl, Gl = _ascending_curve(liquid, column)
    if Ts.size < 2 or Tl.size < 2:
        raise ValueError("G(T) 曲線の点が少なすぎます。")
    lo, hi = max(Ts[0], Tl[0]), min(Ts[-1], Tl[-1])
    if lo >= hi:
        raise ValueError("2 つの曲線の温度範囲が重なっていません。")
    # 両曲線の温度点をすべて使う (線形内挿どうしの差は区間内で線形なので交点は厳密)
    T = np.union1d(Ts, Tl)
    T = np.union1d(T[(T > lo) & (T < hi)], [lo, hi])
    difference = np.interp(T, Ts, Gs) - np.interp(T, Tl, Gl)
    zero = np.flatnonzero(difference == 0.0)
    crossings = np.flatnonzero(difference[:-1] * difference[1:] < 0)
    if zero.size and (crossings.size == 0 or zero[0] <= crossings[0]):
        return float(T[zero[0]])
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
