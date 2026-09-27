"""相転移の検出と熱力学量の温度依存性。

エンタルピー H(T) の「段差」から一次相転移を見つけ、その潜熱 ΔH と
転移エントロピー ΔS = ΔH / T_tr (と、分かれば体積変化) から
融解・凝固・蒸発・凝縮・昇華・凝華を判定する。

使いどころは 2 つある。

* 可逆スケーリング (``ti_rs``): 往路・復路の H(T) から、スキャン中に起きた
  相転移をまとめる (:func:`rs_phase_transitions`)。相転移を跨いだ区間の G(T) は
  もう 1 つの相の値ではないので、跨がない温度範囲 (:func:`rs_safe_limit`) を返す。
* 温度を振った MD (昇温・降温・複数温度の定温): 熱容量・熱膨張係数・
  等温圧縮率などの温度依存性を出す (:func:`thermodynamic_scan`)。

単位: エネルギー eV/atom、温度 K、ΔS は kB/atom (= R per mol of atoms)。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "PROCESS_JA",
    "PHASE_JA",
    "TransitionCriteria",
    "ThermodynamicScan",
    "detect_phase_transitions",
    "rs_phase_transitions",
    "rs_safe_limit",
    "rs_branch_difference",
    "rs_valid_range",
    "hysteresis_melting_estimate",
    "format_transitions",
    "plot_rs_phases",
    "thermodynamic_scan",
]

KB_EV = 8.617333262e-5          # eV/K
R_J_MOL_K = 8.314462618         # J/(mol K) = kB per atom * N_A
EV_TO_KJ_MOL = 96.48533212      # eV/atom -> kJ/mol
GPA_A3_TO_EV = 1.0 / 160.21766208
AMU_A3_TO_G_CM3 = 1.66053906660  # amu/Å^3 -> g/cm^3

#: 過程名 -> 日本語
PROCESS_JA = {
    "melting": "融解",
    "freezing": "凝固",
    "evaporation": "蒸発",
    "condensation": "凝縮",
    "sublimation": "昇華",
    "deposition": "凝華",
    "solid_solid": "固相間転移",
    "liquid_liquid": "液相内の転移",
    "unknown": "不明な転移",
}
PHASE_JA = {"solid": "固相", "liquid": "液相", "gas": "気相", "unknown": "不明"}

_TRANSITION_COLUMNS = [
    "direction", "process", "process_ja", "from_phase", "to_phase",
    "T_K", "T_onset_K", "T_end_K",
    "delta_H_eV_per_atom", "latent_heat_kJ_per_mol",
    "delta_S_kB_per_atom", "delta_S_J_per_mol_K",
    "volume_ratio", "significance",
]


@dataclass
class TransitionCriteria:
    """相転移とみなす基準。

    Parameters
    ----------
    min_entropy_kB
        これより小さい ΔS [kB/atom] の段差は転移とみなさない
        (金属の融解は ~1.1 kB、Si は ~3.6 kB)。
    n_sigma
        段差が H のノイズの何倍を超えたら有意とするか。
    vaporization_entropy_kB
        ΔS [kB/atom] がこれ以上なら気相が関わる転移 (蒸発・昇華) とみなす
        (Trouton 則: 蒸発 ~10 kB/分子)。体積が分かる場合は ``gas_volume_ratio`` を優先する。
    gas_volume_ratio
        転移前後の体積比がこれ以上なら気相が関わるとみなす
        (融解の体積変化はふつう 20% 未満)。
    n_bins
        H(T) を温度方向にならす区間数。
    window
        段差を測る左右の直線当てはめの幅 (区間数)。転移の幅より広くとる。
    """

    min_entropy_kB: float = 0.2
    n_sigma: float = 6.0
    vaporization_entropy_kB: float = 4.5
    gas_volume_ratio: float = 2.0
    n_bins: int = 200
    window: int = 12

    def as_dict(self) -> dict:
        return asdict(self)


# ================================================================ 段差の検出
def _bin_scan(x: np.ndarray, columns: dict[str, np.ndarray], n_bins: int):
    """x (温度) について等間隔の区間で平均をとる (データの無い区間は捨てる)。"""
    n_bins = int(max(8, min(n_bins, len(x))))
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    index = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    counts = np.bincount(index, minlength=n_bins).astype(float)
    keep = counts > 0
    out = {"T": (np.bincount(index, x, minlength=n_bins) / np.maximum(counts, 1))[keep]}
    for name, values in columns.items():
        out[name] = (np.bincount(index, values, minlength=n_bins) / np.maximum(counts, 1))[keep]
    return out


def _line(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) < 2 or np.ptp(x) == 0:
        return np.array([0.0, float(np.mean(y))])
    return np.polyfit(x, y, 1)


def _noise(y: np.ndarray) -> float:
    """2 階差分の MAD から H の (区間平均後の) ノイズを推定する。"""
    if len(y) < 5:
        return float(np.std(y)) or 1e-12
    d2 = y[2:] - 2.0 * y[1:-1] + y[:-2]
    sigma = 1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6.0)
    return float(sigma) if sigma > 0 else float(np.std(d2) / np.sqrt(6.0)) or 1e-12


def _step_profile(x: np.ndarray, y: np.ndarray, w: int) -> np.ndarray:
    """各点で「右側の直線 - 左側の直線」を測る (段差の大きさ)。"""
    step = np.zeros(len(x))
    for i in range(w, len(x) - w):
        left = _line(x[i - w:i], y[i - w:i])
        right = _line(x[i + 1:i + 1 + w], y[i + 1:i + 1 + w])
        step[i] = np.polyval(right, x[i]) - np.polyval(left, x[i])
    return step


def _refine(x, y, i, w):
    """段差 i の前後で直線を引き直し、ΔH・転移温度・開始/終了温度を決める。"""
    n = len(x)
    lo = slice(max(0, i - 2 * w), max(1, i - w // 2))
    hi = slice(min(n - 1, i + w // 2 + 1), min(n, i + 2 * w + 1))
    left, right = _line(x[lo], y[lo]), _line(x[hi], y[hi])
    core = np.arange(max(0, i - w), min(n, i + w + 1))
    gap = np.polyval(right, x[core]) - np.polyval(left, x[core])
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = (y[core] - np.polyval(left, x[core])) / gap
    fraction = np.where(np.isfinite(fraction), fraction, 0.5)
    # 転移温度: fraction が 0.5 を横切る点 (i に最も近いもの)
    crossing = np.flatnonzero((fraction[:-1] - 0.5) * (fraction[1:] - 0.5) <= 0)
    if crossing.size:
        k = crossing[np.argmin(np.abs(core[crossing] - i))]
        f0, f1 = fraction[k], fraction[k + 1]
        t = 0.0 if f1 == f0 else (0.5 - f0) / (f1 - f0)
        T_tr = float(x[core[k]] + t * (x[core[k + 1]] - x[core[k]]))
    else:
        T_tr = float(x[i])
    below = core[(fraction < 0.1) & (x[core] <= T_tr)]
    above = core[(fraction > 0.9) & (x[core] >= T_tr)]
    T_on = float(x[below.max()]) if below.size else float(x[core[0]])
    T_off = float(x[above.min()]) if above.size else float(x[core[-1]])
    delta = float(np.polyval(right, T_tr) - np.polyval(left, T_tr))
    return T_tr, T_on, T_off, delta, left, right


def _is_gas(delta_S_kB, volume_ratio, criteria) -> bool:
    """気相が関わる転移か (体積比があればそれで、無ければ ΔS で判定)。"""
    if volume_ratio is not None and np.isfinite(volume_ratio):
        ratio = volume_ratio if volume_ratio >= 1 else 1.0 / volume_ratio
        return ratio >= criteria.gas_volume_ratio
    return delta_S_kB >= criteria.vaporization_entropy_kB


def _classify_sequence(rows, direction, initial_phase, final_phase, criteria):
    """スキャン順に並んだ段差に (過程, 転移前の相, 転移後の相) を割り当てる。

    気相が関わらない段差のうち、固相↔液相の転移は **ΔS が最大のもの 1 つ** とみなす
    (高速な昇温では融解の前に小さな前駆的な段差が出たり、液体 Si のように
    液相内・アモルファス化の小さな段差が出たりするため)。それより前後の小さな段差は
    固相間転移 / 液相内の転移とする。``delta_H`` は常に 高温相 − 低温相。
    """
    gas = [_is_gas(r["delta_S_kB_per_atom"], r["volume_ratio"], criteria) for r in rows]
    condensed = [i for i, r in enumerate(rows) if not gas[i] and r["delta_H_eV_per_atom"] > 0]
    main = max(condensed, key=lambda i: rows[i]["delta_S_kB_per_atom"]) if condensed else None
    heating = direction == "heating"
    current = initial_phase
    labels = []
    for i, row in enumerate(rows):
        remaining = len(rows) - i
        before = current
        if row["delta_H_eV_per_atom"] <= 0:
            # 昇温で発熱 (過冷却液体の結晶化など) / 降温で吸熱
            process, after = (("freezing", "solid") if heating and current in ("liquid", "unknown")
                              else ("unknown", current))
        elif gas[i]:
            if heating:
                process, after = (("sublimation", "gas") if current == "solid"
                                  else ("evaporation", "gas") if current in ("liquid", "unknown")
                                  else ("unknown", current))
            else:
                if current in ("gas", "unknown"):
                    process, after = (("deposition", "solid")
                                      if final_phase == "solid" and remaining == 1
                                      else ("condensation", "liquid"))
                else:
                    process, after = "unknown", current
        elif heating:
            if current in ("solid", "unknown") and (i == main or main is None):
                process, after = "melting", "liquid"
            elif current == "solid":
                process, after = "solid_solid", "solid"
            elif current == "liquid":
                process, after = "liquid_liquid", "liquid"
            else:
                process, after = "unknown", current
        else:
            if current in ("liquid", "unknown") and (i == main or main is None):
                process, after = "freezing", "solid"
            elif current == "liquid":
                process, after = "liquid_liquid", "liquid"
            elif current == "solid":
                process, after = "solid_solid", "solid"
            else:
                process, after = "unknown", current
        labels.append((process, before, after))
        current = after
    return labels


def detect_phase_transitions(
    temperature: Sequence[float],
    enthalpy: Sequence[float],
    *,
    volume: Sequence[float] | None = None,
    direction: str | None = None,
    initial_phase: str = "unknown",
    final_phase: str | None = None,
    criteria: TransitionCriteria | None = None,
) -> pd.DataFrame:
    """H(T) の段差から一次相転移を検出して分類する。

    Parameters
    ----------
    temperature, enthalpy
        温度 [K] と (1 原子あたりの) エンタルピー [eV/atom]。運動エネルギーの
        有無は問わない (滑らかなので段差に効かない)。並び順は問わない。
    volume
        体積 (任意単位、1 原子あたりが望ましい)。あれば気相の判定に使う。
    direction
        ``'heating'`` / ``'cooling'``。``None`` なら配列の最初と最後の温度から決める。
    initial_phase
        スキャン開始時の相 (``'solid'`` / ``'liquid'`` / ``'gas'`` / ``'unknown'``)。
    final_phase
        スキャン終了時に戻るはずの相 (降温で気相→固相の凝華を判定するのに使う)。

    Returns
    -------
    DataFrame
        スキャン順に並んだ転移。列は ``process`` (``melting`` など)、``process_ja``、
        ``T_K`` (ΔH の半分まで進んだ温度)、``T_onset_K`` / ``T_end_K`` (低温側 / 高温側の端)、
        ``delta_H_eV_per_atom`` (潜熱、常に 高温相 − 低温相)、``latent_heat_kJ_per_mol``、
        ``delta_S_kB_per_atom`` (= ΔH / T_K / kB)、``volume_ratio`` (高温相 / 低温相) など。
    """
    criteria = criteria or TransitionCriteria()
    T = np.asarray(temperature, dtype=float)
    H = np.asarray(enthalpy, dtype=float)
    mask = np.isfinite(T) & np.isfinite(H)
    V = None if volume is None else np.asarray(volume, dtype=float)
    if V is not None:
        mask &= np.isfinite(V)
    T, H = T[mask], H[mask]
    V = V[mask] if V is not None else None
    if direction is None:
        direction = "heating" if (len(T) < 2 or T[-1] >= T[0]) else "cooling"
    if direction not in ("heating", "cooling"):
        raise ValueError("direction は 'heating' か 'cooling' です。")
    empty = pd.DataFrame(columns=_TRANSITION_COLUMNS)
    w = int(criteria.window)
    if len(T) < 4 * w or np.ptp(T) <= 0:
        return empty

    binned = _bin_scan(T, {"H": H, **({"V": V} if V is not None else {})}, criteria.n_bins)
    x, y = binned["T"], binned["H"]
    if len(x) < 4 * w:
        w = max(3, len(x) // 4)
    step = _step_profile(x, y, w)
    sigma_step = _noise(y) * np.sqrt(8.0 / w)
    score = np.abs(step)
    ok = (score > criteria.n_sigma * sigma_step) & (
        score / np.maximum(x, 1e-9) / KB_EV > criteria.min_entropy_kB)
    peaks: list[int] = []
    candidate = np.where(ok, score, 0.0)
    while candidate.max() > 0:
        i = int(np.argmax(candidate))
        peaks.append(i)
        candidate[max(0, i - w):i + w + 1] = 0.0

    found = []
    for i in peaks:
        T_tr, T_on, T_off, delta, _, _ = _refine(x, y, i, w)
        dS = abs(delta) / T_tr / KB_EV
        if dS < criteria.min_entropy_kB:
            continue
        ratio = np.nan
        if V is not None:
            _, _, _, _, vl, vr = _refine(x, binned["V"], i, w)
            low, high = np.polyval(vl, T_tr), np.polyval(vr, T_tr)
            ratio = float(high / low) if low > 0 else np.nan
        found.append({"T_K": T_tr, "T_onset_K": T_on, "T_end_K": T_off,
                      "delta_H_eV_per_atom": delta, "delta_S_kB_per_atom": dS,
                      "volume_ratio": ratio,
                      "significance": abs(step[i]) / sigma_step})
    # 幅の広い 1 つの段差を 2 回数えないよう、転移区間が重なるものは大きい方だけ残す
    found.sort(key=lambda row: -row["significance"])
    kept: list[dict] = []
    for row in found:
        if all(row["T_end_K"] < k["T_onset_K"] or row["T_onset_K"] > k["T_end_K"] for k in kept):
            kept.append(row)
    found = kept
    # スキャン順 (昇温なら低温から、降温なら高温から) に相をたどって分類する
    found.sort(key=lambda row: row["T_K"], reverse=(direction == "cooling"))
    rows = []
    labels = _classify_sequence(found, direction, initial_phase, final_phase, criteria)
    for row, (process, before, after) in zip(found, labels):
        rows.append({
            "direction": direction, "process": process, "process_ja": PROCESS_JA[process],
            "from_phase": before, "to_phase": after, **row,
            "latent_heat_kJ_per_mol": row["delta_H_eV_per_atom"] * EV_TO_KJ_MOL,
            "delta_S_J_per_mol_K": row["delta_S_kB_per_atom"] * R_J_MOL_K,
        })
    return pd.DataFrame(rows, columns=_TRANSITION_COLUMNS) if rows else empty


# ================================================================ 可逆スケーリング
def _rs_passes(ti_rs: pd.DataFrame, T0: float):
    """``ti_rs.csv`` を往路・復路 (それぞれ時間順) の (T, H) に分ける。"""
    if len(ti_rs) < 4:
        raise ValueError("TI の出力行数が少なすぎます (往復のデータが必要)。")
    n = len(ti_rs) // 2
    forward = ti_rs.iloc[:n]
    backward = ti_rs.iloc[len(ti_rs) - n:]
    return [
        (T0 / forward["lambda"].to_numpy(), forward["enthalpy"].to_numpy()),
        (T0 / backward["lambda"].to_numpy(), backward["enthalpy"].to_numpy()),
    ]


def rs_phase_transitions(
    ti_rs: pd.DataFrame,
    *,
    T0: float,
    initial_phase: str = "unknown",
    criteria: TransitionCriteria | None = None,
) -> pd.DataFrame:
    """可逆スケーリングの往路・復路で起きた相転移をまとめる。

    ``ti_rs`` の系は温度 T0 のままポテンシャルを lambda 倍しているが、
    これは元のポテンシャルで温度 T0/lambda の系と同じ分布になる。
    ``enthalpy`` 列 (U + PV) はその実温度での値なので、H(T0/lambda) の段差が
    そのまま潜熱になる。

    Parameters
    ----------
    T0
        スキャンの出発温度 (``reversible_scaling`` の ``T_min``)。
        ``T_max < T_min`` の降温スキャンにも対応する (往路が降温、復路が昇温)。
    initial_phase
        T0 での相 (Frenkel-Ladd の基準なら ``'solid'``、Uhlenbeck-Ford なら ``'liquid'``)。

    Returns
    -------
    DataFrame
        :func:`detect_phase_transitions` の列に ``pass`` (``forward`` / ``backward``)
        を足したもの。
    """
    passes = _rs_passes(ti_rs, T0)
    frames = []
    phase = initial_phase
    for name, (T, H) in zip(("forward", "backward"), passes):
        direction = "heating" if T[-1] >= T[0] else "cooling"
        table = detect_phase_transitions(
            T, H, direction=direction, initial_phase=phase,
            final_phase=initial_phase if name == "backward" else None, criteria=criteria)
        table.insert(0, "pass", name)
        frames.append(table)
        if len(table):
            phase = table["to_phase"].iloc[-1]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=["pass"] + _TRANSITION_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def rs_safe_limit(
    transitions: pd.DataFrame, *, T0: float, T_end: float, margin: float = 0.03
) -> float | None:
    """T0 から見て最初の相転移の手前の温度 (相転移を跨がない範囲の端)。

    往路で融解が起きると、復路は過冷却液体のまま降りてきて融点より下で凝固する。
    往路と復路が同じ相にいるのは、どちらの転移区間にも入らない T0 側だけなので、
    すべての転移区間の端のうち T0 に最も近いものを取り、さらに ``margin``
    (相対値) だけ内側に寄せる。転移が無ければ ``None``。

    向きの逆な段差 (昇温で発熱する ``unknown``) は、高速なランプや圧浴の揺らぎで出る
    ことが多く一次転移とはみなせないので、範囲の決定には使わない (表には残る)。
    """
    if transitions is None or len(transitions) == 0:
        return None
    spurious = (transitions["process"] == "unknown") & (transitions["delta_H_eV_per_atom"] <= 0)
    transitions = transitions[~spurious]
    if len(transitions) == 0:
        return None
    edges = np.concatenate([transitions["T_onset_K"].to_numpy(dtype=float),
                            transitions["T_end_K"].to_numpy(dtype=float)])
    if T_end >= T0:
        edges = edges[edges > T0]
        return float(edges.min() * (1.0 - margin)) if edges.size else float(T0)
    edges = edges[edges < T0]
    return float(edges.max() * (1.0 + margin)) if edges.size else float(T0)


def rs_branch_difference(
    ti_rs: pd.DataFrame, *, T0: float, n_bins: int = 200
) -> pd.DataFrame:
    """同じ lambda (= 同じ温度) での 復路 − 往路 のエンタルピー差 (温度で区間平均)。

    往路と復路が同じ相なら差は非平衡性による小さな値だけ。相が違えば潜熱程度の差になる。
    """
    n = len(ti_rs) // 2
    forward = ti_rs.iloc[:n]
    backward = ti_rs.iloc[len(ti_rs) - n:][::-1]
    T = T0 / forward["lambda"].to_numpy()
    diff = backward["enthalpy"].to_numpy() - forward["enthalpy"].to_numpy()
    binned = _bin_scan(T, {"dH": diff}, n_bins)
    frame = pd.DataFrame({"temperature": binned["T"], "dH": binned["dH"]})
    # T0 から近い順に並べる
    return frame.iloc[np.argsort(np.abs(frame["temperature"].to_numpy() - T0))].reset_index(
        drop=True)


def rs_valid_range(
    ti_rs: pd.DataFrame,
    transitions: pd.DataFrame,
    *,
    T0: float,
    T_end: float,
    margin: float = 0.03,
    criteria: TransitionCriteria | None = None,
) -> dict:
    """G(T) を信用できる温度範囲と、使う経路 (往復平均か往路のみか) を決める。

    1. 復路が T0 で出発相に戻っているか (T0 付近の 復路 − 往路 の H の差が
       ``min_entropy_kB`` 相当より小さいか) を調べる。折り返し点の保持 (``t_equil``)
       中に融けて戻らなかった場合など、往路に段差が無くても復路は別の相のことがある。
    2. 戻っていれば往復平均 (``branch='both'``) を使い、範囲の端は
       「最初の転移区間の手前」と「往路・復路の H が食い違い始める温度」の T0 に近い方。
    3. 戻っていなければ往路のみ (``branch='forward'``) を使い、範囲の端は
       往路の最初の転移の手前 (往路に転移が無ければ往路全体)。

    Returns
    -------
    dict
        ``limit`` (範囲の端 [K]、全域が有効なら ``None``)、``branch``、
        ``backward_returns``、``dH_at_T0`` [eV/atom] (T0 付近の 復路 − 往路)、``reason``。
    """
    criteria = criteria or TransitionCriteria()
    difference = rs_branch_difference(ti_rs, T0=T0, n_bins=criteria.n_bins)
    T, dH = difference["temperature"].to_numpy(), difference["dH"].to_numpy()
    sigma = _noise(dH[np.argsort(T)]) if len(dH) > 5 else 0.0
    threshold = np.maximum(criteria.min_entropy_kB * KB_EV * T, criteria.n_sigma * sigma)
    near = max(3, len(T) // 20)
    dH_T0 = float(np.mean(dH[:near]))
    returns = abs(dH_T0) <= float(np.mean(threshold[:near]))
    ascending = T_end >= T0

    def nearest(values):
        values = [v for v in values if v is not None]
        if not values:
            return None
        return min(values) if ascending else max(values)

    if returns:
        limits = [rs_safe_limit(transitions, T0=T0, T_end=T_end, margin=margin)]
        # T0 から見て、食い違いが 3 区間続いた最初の温度
        bad = np.abs(dH) > threshold
        run = np.convolve(bad.astype(int), np.ones(3, dtype=int), mode="valid") == 3
        if run.any():
            T_c = float(T[int(np.argmax(run))])
            limits.append(T_c * (1.0 - margin) if ascending else T_c * (1.0 + margin))
        limit = nearest(limits)
        reason = ("往路と復路は同じ相にいる" if limit is None
                  else "往路・復路の最初の転移の手前まで (往復平均)")
        branch = "both"
    else:
        forward = (transitions[transitions["pass"] == "forward"]
                   if transitions is not None and len(transitions) and "pass" in transitions
                   else transitions)
        limit = rs_safe_limit(forward, T0=T0, T_end=T_end, margin=margin)
        branch = "forward"
        reason = ("復路が出発相に戻っていない (折り返し点付近で転移して戻らなかった)"
                  " ため往路のみを使う")
    if limit is not None and abs(limit - T0) < 1e-9:
        limit = float(T0)
    return {"limit": limit, "branch": branch, "backward_returns": bool(returns),
            "dH_at_T0": dH_T0, "reason": reason}


def hysteresis_melting_estimate(T_heating: float, T_cooling: float) -> float:
    """昇温で融ける温度 T+ と降温で固まる温度 T- から融点を見積もる。

    Luo et al. (2004) の ``Tm = T+ + T- - sqrt(T+ T-)``。過熱・過冷却の
    ヒステリシスの補正なので目安であり、正確な融点は G(T) の交点から求める。
    """
    return float(T_heating + T_cooling - np.sqrt(T_heating * T_cooling))


def format_transitions(transitions: pd.DataFrame, *, title: str = "相転移") -> str:
    """転移の表を人が読むための Markdown にする。"""
    lines = [f"## {title}", ""]
    if transitions is None or len(transitions) == 0:
        lines.append("検出された相転移はありません (スキャン範囲は 1 つの相の中)。")
        return "\n".join(lines) + "\n"
    has_pass = "pass" in transitions
    header = ("| 経路 | " if has_pass else "| ") + (
        "向き | 過程 | 相 | T (K) | 区間 (K) | ΔH (eV/atom) | 潜熱 (kJ/mol) "
        "| ΔS (kB/atom) | ΔS (J/mol/K) | 体積比 |")
    lines += [header, "|" + "---|" * (header.count("|") - 1)]
    for _, row in transitions.iterrows():
        ratio = row.get("volume_ratio", np.nan)
        cells = ([row["pass"]] if has_pass else []) + [
            "昇温" if row["direction"] == "heating" else "降温",
            f"{row['process_ja']} ({row['process']})",
            f"{PHASE_JA.get(row['from_phase'], row['from_phase'])} → "
            f"{PHASE_JA.get(row['to_phase'], row['to_phase'])}",
            f"{row['T_K']:.0f}",
            f"{row['T_onset_K']:.0f}–{row['T_end_K']:.0f}",
            f"{row['delta_H_eV_per_atom']:+.4f}",
            f"{row['latent_heat_kJ_per_mol']:+.2f}",
            f"{row['delta_S_kB_per_atom']:.2f}",
            f"{row['delta_S_J_per_mol_K']:.1f}",
            "-" if not np.isfinite(ratio) else f"{ratio:.3f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("ΔH・ΔS は「高温側の相 − 低温側の相」(1 原子あたり、kJ/mol は原子 1 mol あたり)。"
                 " 昇温で見える転移温度は過熱、降温で見える温度は過冷却を含む。")
    return "\n".join(lines) + "\n"


def plot_rs_phases(
    ti_rs: pd.DataFrame,
    transitions: pd.DataFrame,
    filename: Path | str,
    *,
    T0: float,
    limit: float | None = None,
    curve: pd.DataFrame | None = None,
    title: str | None = None,
    dpi: int = 150,
) -> Path:
    """RS の H(T) (往路・復路) に検出した相転移を書き込んだ図。

    ``curve`` (G(T)) を渡すと下段に G(T) を描き、``limit`` より先を灰色にする。
    """
    from .plotting import label, setup_matplotlib

    setup_matplotlib()
    import matplotlib.pyplot as plt

    rows = 2 if curve is not None and len(curve) else 1
    figure, axes = plt.subplots(rows, 1, figsize=(7.5, 3.8 * rows), sharex=True, squeeze=False)
    axis = axes[0, 0]
    colors = {"forward": "C3", "backward": "C0"}
    for name, (T, H) in zip(("forward", "backward"), _rs_passes(ti_rs, T0)):
        direction = "heating" if T[-1] >= T[0] else "cooling"
        axis.plot(T, H, lw=0.6, alpha=0.8, color=colors[name],
                  label=f"{name} ({'昇温' if direction == 'heating' else '降温'})")
    for _, row in (transitions if transitions is not None else pd.DataFrame()).iterrows():
        color = colors.get(row.get("pass", "forward"), "k")
        axis.axvspan(row["T_onset_K"], row["T_end_K"], color=color, alpha=0.12)
        axis.annotate(
            label(f"{row['process_ja']}\n{row['T_K']:.0f} K\nΔS={row['delta_S_kB_per_atom']:.2f} kB"),
            xy=(row["T_K"], 1.0), xycoords=("data", "axes fraction"),
            ha="center", va="top", fontsize=7, color=color)
    if limit is not None:
        axis.axvline(limit, color="k", ls="--", lw=0.9, label=f"safe limit {limit:.0f} K")
    axis.set_ylabel("H = U + PV (eV/atom)")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, loc="lower right")
    if rows == 2:
        ax_g = axes[1, 0]
        ordered = curve.sort_values("temperature")
        T, G = ordered["temperature"].to_numpy(), ordered["G"].to_numpy()
        if limit is not None:
            inside = (T >= min(T0, limit)) & (T <= max(T0, limit))
            ax_g.plot(T[~inside], G[~inside], ".", ms=1, color="0.7", label="beyond transition")
            ax_g.plot(T[inside], G[inside], lw=1.6, color="C2", label="G(T) (valid)")
        else:
            ax_g.plot(T, G, lw=1.6, color="C2", label="G(T)")
        ax_g.set_ylabel("G (eV/atom)")
        ax_g.grid(alpha=0.3)
        ax_g.legend(fontsize=8)
    axes[-1, 0].set_xlabel("T (K)")
    if title:
        figure.suptitle(label(title))
    figure.tight_layout()
    output = Path(filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return output


# ================================================================ 温度を振った MD
_FLUCTUATION_ENSEMBLES = {
    # 正しいカノニカル / NPT 分布を与える熱浴・圧浴 (Berendsen は揺らぎが正しくない)
    "nvt_nhc", "nvt_bdp", "nvt_lan", "nvt_bao", "nvt_mttk", "npt_scr", "npt_mttk",
}
_NPT = ("npt_ber", "npt_scr", "npt_mttk")
_NVT = ("nvt_ber", "nvt_nhc", "nvt_bdp", "nvt_lan", "nvt_bao", "nvt_mttk")


@dataclass
class ThermodynamicScan:
    """温度を振った MD から得た熱力学量の温度依存性。

    Attributes
    ----------
    ramp
        昇温・降温ステージを温度で区切った平均値と微分量
        (``C_kB_per_atom`` = dH/dT (NPT) または dE/dT (NVT)、
        ``alpha_per_K`` = (1/V) dV/dT)。
    plateaus
        定温ステージごとの平均値と揺らぎから求めた量
        (``C_fluct_kB_per_atom`` = Var(H)/(kB T)^2/N、``kappa_T_per_GPa`` =
        Var(V)/(kB T <V>)、``bulk_modulus_GPa``、``alpha_fluct_per_K`` =
        Cov(V, H)/(kB T^2 <V>)) と、隣の温度との差分 (``C_fd_kB_per_atom`` など)。
    transitions
        昇温・降温ステージで検出した相転移 (:func:`detect_phase_transitions`)。
    """

    ramp: pd.DataFrame
    plateaus: pd.DataFrame
    transitions: pd.DataFrame
    mass_amu_per_atom: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def temperature_varied(self) -> bool:
        """昇温・降温ステージか、温度の異なる定温ステージが 2 つ以上あるか。"""
        if len(self.ramp):
            return True
        return len(self.plateaus) > 0 and self.plateaus["T_target_K"].nunique() >= 2

    def table(self) -> pd.DataFrame:
        """ramp と plateaus を 1 つの表にしたもの (列 ``source``)。"""
        frames = [f for f in (self.ramp, self.plateaus) if len(f)]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True, sort=False).sort_values(
            ["source", "T_K"], kind="stable").reset_index(drop=True)

    def markdown(self) -> str:
        lines = ["# 熱力学量の温度依存性", ""]
        if self.mass_amu_per_atom:
            lines.append(f"平均原子質量: {self.mass_amu_per_atom:.4f} amu")
        lines += [f"- {note}" for note in self.notes]
        lines.append("")
        if len(self.plateaus):
            lines += ["## 定温ステージ (揺らぎ)", "",
                      "| T (K) | アンサンブル | H (eV/atom) | V (Å³/atom) | C 揺らぎ (kB/atom) "
                      "| C 差分 (kB/atom) | α 揺らぎ (1/K) | κ_T (1/GPa) | B (GPa) |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for _, r in self.plateaus.iterrows():
                lines.append(
                    f"| {r['T_K']:.1f} | {r['ensemble']} | {r['H_eV_per_atom']:.5f} "
                    f"| {r['V_A3_per_atom']:.4f} | {_fmt(r.get('C_fluct_kB_per_atom'))} "
                    f"| {_fmt(r.get('C_fd_kB_per_atom'))} | {_fmt(r.get('alpha_fluct_per_K'), 'e')} "
                    f"| {_fmt(r.get('kappa_T_per_GPa'), 'e')} | {_fmt(r.get('bulk_modulus_GPa'))} |")
            lines.append("")
        if len(self.ramp):
            lines += ["## 昇温・降温ステージ (微分)", "",
                      "| stage | T (K) | H (eV/atom) | V (Å³/atom) | C (kB/atom) | C (J/mol/K) "
                      "| α (1/K) |", "|---|---|---|---|---|---|---|"]
            for _, r in self.ramp.iterrows():
                lines.append(
                    f"| {r['stage']} | {r['T_K']:.1f} | {r['H_eV_per_atom']:.5f} "
                    f"| {r['V_A3_per_atom']:.4f} | {_fmt(r['C_kB_per_atom'])} "
                    f"| {_fmt(r['C_J_per_mol_K'])} | {_fmt(r['alpha_per_K'], 'e')} |")
            lines.append("")
        lines.append(format_transitions(self.transitions, title="昇温・降温で検出した相転移"))
        return "\n".join(lines)

    def write(self, directory: Path | str, *, prefix: str = "thermodynamics",
              dpi: int = 150) -> dict[str, Path]:
        """``<prefix>.csv`` / ``<prefix>_transitions.csv`` / ``<prefix>.md`` / ``.png`` を書く。"""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "csv": directory / f"{prefix}.csv",
            "transitions": directory / f"{prefix}_transitions.csv",
            "markdown": directory / f"{prefix}.md",
        }
        self.table().to_csv(paths["csv"], index=False)
        self.transitions.to_csv(paths["transitions"], index=False)
        paths["markdown"].write_text(self.markdown(), encoding="utf-8")
        paths["png"] = self.plot(directory / f"{prefix}.png", dpi=dpi)
        return paths

    def plot(self, filename: Path | str, *, dpi: int = 150) -> Path:
        from .plotting import label, setup_matplotlib

        setup_matplotlib()
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
        (ax_h, ax_v), (ax_c, ax_a) = axes
        # 定温ステージだけのとき ramp は列の無い空の表なので groupby できない
        if len(self.ramp):
            for stage, group in self.ramp.groupby("stage", sort=True):
                tag = f"ramp {stage} ({group['ensemble'].iloc[0]})"
                ax_h.plot(group["T_K"], group["H_eV_per_atom"], "-", lw=1.2, label=tag)
                ax_v.plot(group["T_K"], group["V_A3_per_atom"], "-", lw=1.2, label=tag)
                ax_c.plot(group["T_K"], group["C_kB_per_atom"], "-", lw=1.0, label=tag)
                ax_a.plot(group["T_K"], group["alpha_per_K"], "-", lw=1.0, label=tag)
        if len(self.plateaus):
            p = self.plateaus.sort_values("T_K")
            ax_h.plot(p["T_K"], p["H_eV_per_atom"], "ko", ms=5, mfc="white", label="plateaus")
            ax_v.plot(p["T_K"], p["V_A3_per_atom"], "ko", ms=5, mfc="white", label="plateaus")
            ax_c.plot(p["T_K"], p["C_fluct_kB_per_atom"], "ks", ms=5, mfc="white",
                      label="fluctuation")
            ax_c.plot(p["T_K"], p["C_fd_kB_per_atom"], "k^", ms=5, label="finite difference")
            ax_a.plot(p["T_K"], p["alpha_fluct_per_K"], "ks", ms=5, mfc="white",
                      label="fluctuation")
            ax_a.plot(p["T_K"], p["alpha_fd_per_K"], "k^", ms=5, label="finite difference")
        for _, row in self.transitions.iterrows():
            for axis in axes.flat:
                axis.axvspan(row["T_onset_K"], row["T_end_K"], color="C3", alpha=0.1)
            ax_h.annotate(label(f"{row['process_ja']}\n{row['T_K']:.0f} K"),
                          xy=(row["T_K"], 0.97), xycoords=("data", "axes fraction"),
                          ha="center", va="top", fontsize=7, color="C3")
        ax_c.axhline(3.0, color="0.6", ls=":", lw=0.8, label="Dulong-Petit 3 kB")
        ax_h.set_ylabel("H = E + PV (eV/atom)")
        ax_v.set_ylabel("V (Å$^3$/atom)")
        ax_c.set_ylabel("C (kB/atom)")
        ax_a.set_ylabel("α (1/K)")
        for axis in axes.flat:
            axis.grid(alpha=0.3)
            if axis.get_legend_handles_labels()[0]:
                axis.legend(fontsize=7)
        for axis in axes[1]:
            axis.set_xlabel("T (K)")
        # 転移のピークで C が発散して見えなくなるのを防ぐ
        values = self.ramp["C_kB_per_atom"].to_numpy() if len(self.ramp) else np.array([])
        values = values[np.isfinite(values)]
        if values.size > 4:
            lo, hi = np.percentile(values, [5, 90])
            span = max(hi - lo, 1.0)
            ax_c.set_ylim(min(0.0, lo - 0.5 * span), hi + 1.5 * span)
        figure.suptitle(label("熱力学量の温度依存性"))
        figure.tight_layout()
        output = Path(filename)
        figure.savefig(output, dpi=dpi)
        plt.close(figure)
        return output


def _fmt(value, kind: str = "f") -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:.3e}" if kind == "e" else f"{value:.3f}"


def _local_slope(x: np.ndarray, y: np.ndarray, half: int = 2) -> np.ndarray:
    """前後 ``half`` 点の直線当てはめで dy/dx を求める (差分より雑音に強い)。"""
    slope = np.full(len(x), np.nan)
    for i in range(len(x)):
        lo, hi = max(0, i - half), min(len(x), i + half + 1)
        if hi - lo >= 2 and np.ptp(x[lo:hi]) > 0:
            slope[i] = np.polyfit(x[lo:hi], y[lo:hi], 1)[0]
    return slope


def thermodynamic_scan(
    thermo,
    *,
    bin_width: float | None = None,
    n_bins: int = 30,
    drop_fraction: float = 0.3,
    mass_amu_per_atom: float | None = None,
    initial_phase: str = "unknown",
    criteria: TransitionCriteria | None = None,
) -> ThermodynamicScan:
    """温度を振った MD の ``thermo.out`` から熱力学量の温度依存性を求める。

    Parameters
    ----------
    thermo
        :class:`~gpumd_toolkit.analysis.ThermoData` (か、その ``frame``)。
        ``run.in`` が読めて目標温度 (``target_temperature``) が分かっている必要がある。
    bin_width, n_bins
        昇温・降温ステージを目標温度で区切る幅 [K] (省略時は各ステージを ``n_bins`` 等分)。
    drop_fraction
        定温ステージの先頭を平衡化として捨てる割合。
    mass_amu_per_atom
        平均原子質量 [amu]。与えると密度 [g/cm^3] と比熱 [J/(g K)] も出す。
    initial_phase
        最初の昇温・降温ステージの開始時の相 (相転移の分類用)。

    Notes
    -----
    * 昇温・降温: C = dH/dT (NPT なら定圧熱容量 C_P、NVT なら dE/dT = C_V)、
      α = (1/V) dV/dT。1 次転移を跨ぐと潜熱のピークとして現れる (転移は別表)。
    * 定温: 揺らぎの公式 C_P = Var(H)/(kB T^2)、C_V = Var(E)/(kB T^2)、
      κ_T = Var(V)/(kB T <V>)、α = Cov(V, H)/(kB T^2 <V>)。Berendsen の熱浴・圧浴は
      揺らぎが正しくないので ``fluctuation_valid`` が False になる。
      温度が 2 つ以上あれば隣との差分 (``*_fd_*``) も出す。
    * エネルギーは古典的 (量子補正なし)。
    """
    frame = thermo.frame if hasattr(thermo, "frame") else thermo
    n_atoms = int(frame["n_atoms"].iloc[0]) if "n_atoms" in frame else 1
    notes: list[str] = []
    ramp_rows: list[dict] = []
    plateau_rows: list[dict] = []
    transitions: list[pd.DataFrame] = []
    phase = initial_phase
    criteria = criteria or TransitionCriteria()

    for stage_id, group in frame.groupby("stage", sort=True):
        ensemble = str(group["ensemble"].iloc[0])
        target = group["target_temperature"].to_numpy(dtype=float)
        if ensemble not in _NVT + _NPT or not np.isfinite(target).all() or len(group) < 3:
            continue
        npt = ensemble in _NPT
        E = group["total_energy_per_atom"].to_numpy()
        P = float(group["pressure"].mean()) if npt else 0.0   # 外圧 ~ 平均圧力
        V = group["volume_per_atom"].to_numpy()
        H = E + P * V * GPA_A3_TO_EV
        base = {"stage": int(stage_id), "ensemble": ensemble,
                "C_kind": "C_P" if npt else "C_V", "pressure_GPa": P if npt else np.nan}

        if np.ptp(target) > 1.0:  # ------------------------------------ 昇温・降温
            direction = "heating" if target[-1] > target[0] else "cooling"
            width = bin_width or np.ptp(target) / max(int(n_bins), 2)
            edges = np.arange(target.min(), target.max() + width, width)
            index = np.clip(np.digitize(target, edges) - 1, 0, len(edges) - 2)
            rows = []
            for b in np.unique(index):
                sel = index == b
                if sel.sum() < 2:
                    continue
                v_mean = float(V[sel].mean())
                rows.append({
                    **base, "source": "ramp", "direction": direction,
                    "T_K": float(group["temperature"].to_numpy()[sel].mean()),
                    "T_target_K": float(target[sel].mean()), "n_samples": int(sel.sum()),
                    "E_eV_per_atom": float(E[sel].mean()),
                    "U_eV_per_atom": float(group["potential_energy_per_atom"].to_numpy()[sel].mean()),
                    "H_eV_per_atom": float(H[sel].mean()), "V_A3_per_atom": v_mean,
                    "P_mean_GPa": float(group["pressure"].to_numpy()[sel].mean()),
                })
            if len(rows) >= 3:
                table = pd.DataFrame(rows)
                x = table["T_target_K"].to_numpy()
                y = (table["H_eV_per_atom"] if npt else table["E_eV_per_atom"]).to_numpy()
                table["C_kB_per_atom"] = _local_slope(x, y) / KB_EV
                table["alpha_per_K"] = (_local_slope(x, table["V_A3_per_atom"].to_numpy())
                                        / table["V_A3_per_atom"].to_numpy()) if npt else np.nan
                ramp_rows += table.to_dict("records")
            found = detect_phase_transitions(
                target, H, volume=V if npt else None, direction=direction,
                initial_phase=phase, criteria=criteria)
            if len(found):
                found.insert(0, "stage", int(stage_id))
                transitions.append(found)
                phase = found["to_phase"].iloc[-1]
            continue

        # ---------------------------------------------------------------- 定温
        cut = group.iloc[int(len(group) * drop_fraction):]
        if len(cut) < 3:
            continue
        T = float(cut["temperature"].mean())
        kT = KB_EV * T
        E_sys = cut["total_energy"].to_numpy()
        V_sys = cut["volume"].to_numpy()
        H_sys = E_sys + P * V_sys * GPA_A3_TO_EV
        fluct = (H_sys if npt else E_sys)
        row = {
            **base, "source": "plateau", "direction": "constant",
            "T_K": T, "T_target_K": float(target.mean()), "n_samples": len(cut),
            "E_eV_per_atom": float(E_sys.mean() / n_atoms),
            "U_eV_per_atom": float(cut["potential_energy_per_atom"].mean()),
            "H_eV_per_atom": float(H_sys.mean() / n_atoms),
            "V_A3_per_atom": float(V_sys.mean() / n_atoms),
            "P_mean_GPa": float(cut["pressure"].mean()),
            "fluctuation_valid": ensemble in _FLUCTUATION_ENSEMBLES,
            "C_fluct_kB_per_atom": float(np.var(fluct, ddof=1) / kT**2 / n_atoms),
        }
        if npt and np.var(V_sys) > 0:
            v_mean = float(V_sys.mean())
            kappa = float(np.var(V_sys, ddof=1) / (kT * v_mean)) / 160.21766208  # 1/GPa
            row["kappa_T_per_GPa"] = kappa
            row["bulk_modulus_GPa"] = 1.0 / kappa if kappa > 0 else np.nan
            row["alpha_fluct_per_K"] = float(
                np.cov(V_sys, H_sys, ddof=1)[0, 1] / (KB_EV * T**2 * v_mean))
        plateau_rows.append(row)

    ramp = pd.DataFrame(ramp_rows)
    plateaus = pd.DataFrame(plateau_rows)
    if len(plateaus):
        for column in ("kappa_T_per_GPa", "bulk_modulus_GPa", "alpha_fluct_per_K"):
            if column not in plateaus:
                plateaus[column] = np.nan
        plateaus["C_fd_kB_per_atom"] = np.nan
        plateaus["alpha_fd_per_K"] = np.nan
        # 同じ種類 (NVT / NPT) の定温ステージどうしで温度差分をとる
        for kind, group in plateaus.groupby("C_kind"):
            group = group.groupby("T_target_K", as_index=False).mean(numeric_only=True)
            if len(group) < 2:
                continue
            x = group["T_K"].to_numpy()
            y = group["H_eV_per_atom" if kind == "C_P" else "E_eV_per_atom"].to_numpy()
            slope = np.gradient(y, x) / KB_EV
            dv = np.gradient(group["V_A3_per_atom"].to_numpy(), x) / group["V_A3_per_atom"].to_numpy()
            for t, c, a in zip(group["T_target_K"], slope, dv):
                sel = (plateaus["C_kind"] == kind) & (plateaus["T_target_K"] == t)
                plateaus.loc[sel, "C_fd_kB_per_atom"] = c
                if kind == "C_P":
                    plateaus.loc[sel, "alpha_fd_per_K"] = a
        few = plateaus[plateaus["n_samples"] < 1000]
        if len(few):
            notes.append(
                f"定温ステージのサンプル数が少ない ({int(few['n_samples'].min())} 点〜) ため、"
                "揺らぎから求めた値 (特に κ_T・B・α) の誤差は大きい。"
                " 長い定温 run か細かい dump_thermo 間隔を推奨 (目安 1000 点以上、相関時間より十分長く)。")
        if not plateaus["fluctuation_valid"].all():
            notes.append("Berendsen の熱浴・圧浴のステージは揺らぎの値 (C 揺らぎ, κ_T, α 揺らぎ) "
                         "が正しくありません (fluctuation_valid=False)。")
    for table in (ramp, plateaus):
        if len(table):
            for src, dst in (("C_kB_per_atom", "C_J_per_mol_K"),
                             ("C_fluct_kB_per_atom", "C_fluct_J_per_mol_K"),
                             ("C_fd_kB_per_atom", "C_fd_J_per_mol_K")):
                if src in table:
                    table[dst] = table[src] * R_J_MOL_K
            if mass_amu_per_atom:
                table["density_g_cm3"] = (mass_amu_per_atom * AMU_A3_TO_G_CM3
                                          / table["V_A3_per_atom"])
                for src in ("C_kB_per_atom", "C_fluct_kB_per_atom"):
                    if src in table:
                        table[src.replace("kB_per_atom", "J_per_g_K")] = (
                            table[src] * R_J_MOL_K / mass_amu_per_atom)
    if not len(ramp) and (not len(plateaus) or plateaus["T_target_K"].nunique() < 2):
        notes.append("温度を振ったステージ (昇温・降温、または温度の異なる定温) がありません。")
    all_transitions = (pd.concat(transitions, ignore_index=True) if transitions
                       else pd.DataFrame(columns=["stage"] + _TRANSITION_COLUMNS))
    return ThermodynamicScan(ramp=ramp, plateaus=plateaus, transitions=all_transitions,
                             mass_amu_per_atom=mass_amu_per_atom, notes=notes)
