"""GPUMD の出力から物理量を取り出す後処理。

:mod:`gpumd_toolkit.outputs` が「ファイル → DataFrame」を担当するのに対し、
本モジュールは「DataFrame → 物理量」を担当する。
単位換算とフィット範囲の扱いをここに集約してある。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "EV_PS_TO_WATT",
    "ElasticModuli",
    "elastic_moduli",
    "nemd_thermal_conductivity",
    "temperature_gradient",
    "green_kubo_kappa",
    "hnemd_kappa",
    "shear_viscosity",
    "diffusion_from_msd",
    "diffusion_from_sdc",
    "ionic_conductivity",
    "free_energy_reversible_scaling",
    "free_energy_adiabatic_switching",
    "gibbs_from_ti",
    "vibrational_spectrum",
    "committee_uncertainty_summary",
]

#: 1 eV/ps を W に直す係数
EV_PS_TO_WATT = 1.602176634e-7

#: Boltzmann 定数 [eV/K]
KB_EV = 8.617333262e-5


def _tail(frame: pd.DataFrame, drop_fraction: float) -> pd.DataFrame:
    if not 0.0 <= drop_fraction < 1.0:
        raise ValueError("drop_fraction は 0 以上 1 未満です。")
    start = int(len(frame) * drop_fraction)
    return frame.iloc[start:]


# ------------------------------------------------------------------ 弾性
@dataclass
class ElasticModuli:
    """弾性定数行列から導かれる多結晶弾性率 [GPa]。"""

    C: np.ndarray
    bulk_voigt: float
    bulk_reuss: float
    shear_voigt: float
    shear_reuss: float

    @property
    def bulk(self) -> float:
        """体積弾性率 (Voigt-Reuss-Hill 平均)。"""
        return 0.5 * (self.bulk_voigt + self.bulk_reuss)

    @property
    def shear(self) -> float:
        """剛性率 (Voigt-Reuss-Hill 平均)。"""
        return 0.5 * (self.shear_voigt + self.shear_reuss)

    @property
    def young(self) -> float:
        """ヤング率。"""
        B, G = self.bulk, self.shear
        return 9.0 * B * G / (3.0 * B + G)

    @property
    def poisson(self) -> float:
        """ポアソン比。"""
        B, G = self.bulk, self.shear
        return (3.0 * B - 2.0 * G) / (2.0 * (3.0 * B + G))

    def as_dict(self) -> dict[str, float]:
        return {
            "bulk_voigt": self.bulk_voigt,
            "bulk_reuss": self.bulk_reuss,
            "bulk_hill": self.bulk,
            "shear_voigt": self.shear_voigt,
            "shear_reuss": self.shear_reuss,
            "shear_hill": self.shear,
            "young": self.young,
            "poisson": self.poisson,
        }

    def __str__(self) -> str:
        return (
            f"B = {self.bulk:.1f} GPa, G = {self.shear:.1f} GPa, "
            f"E = {self.young:.1f} GPa, nu = {self.poisson:.3f} (Hill 平均)"
        )


def elastic_moduli(C: np.ndarray) -> ElasticModuli:
    """6x6 の弾性定数行列から多結晶弾性率を計算する (Voigt / Reuss / Hill)。"""
    C = np.asarray(C, dtype=float)
    if C.shape != (6, 6):
        raise ValueError("C は 6x6 行列です。")
    bulk_voigt = (C[0, 0] + C[1, 1] + C[2, 2] + 2 * (C[0, 1] + C[1, 2] + C[0, 2])) / 9.0
    shear_voigt = (
        C[0, 0] + C[1, 1] + C[2, 2] - (C[0, 1] + C[1, 2] + C[0, 2])
        + 3 * (C[3, 3] + C[4, 4] + C[5, 5])
    ) / 15.0
    try:
        S = np.linalg.inv(C)
    except np.linalg.LinAlgError as error:  # pragma: no cover - 特異行列
        raise ValueError("弾性定数行列が特異です。歪み量や収束を確認してください。") from error
    bulk_reuss = 1.0 / (S[0, 0] + S[1, 1] + S[2, 2] + 2 * (S[0, 1] + S[1, 2] + S[0, 2]))
    shear_reuss = 15.0 / (
        4 * (S[0, 0] + S[1, 1] + S[2, 2]) - 4 * (S[0, 1] + S[1, 2] + S[0, 2])
        + 3 * (S[3, 3] + S[4, 4] + S[5, 5])
    )
    return ElasticModuli(C, bulk_voigt, bulk_reuss, shear_voigt, shear_reuss)


# ------------------------------------------------------------------ 熱伝導
def temperature_gradient(
    positions: Sequence[float], temperatures: Sequence[float]
) -> tuple[float, float]:
    """温度プロファイルを直線で近似して勾配 [K/Å] と切片を返す。"""
    x = np.asarray(positions, dtype=float)
    T = np.asarray(temperatures, dtype=float)
    if x.size < 2:
        raise ValueError("2 点以上必要です。")
    slope, intercept = np.polyfit(x, T, 1)
    return float(slope), float(intercept)


def nemd_thermal_conductivity(
    compute_frame: pd.DataFrame,
    *,
    layout,
    area: float,
    output_interval_fs: float,
    drop_fraction: float = 0.5,
) -> dict[str, float]:
    """NEMD (``heat_*`` + ``compute``) から熱伝導率 [W/mK] を求める。

    熱浴が出し入れしたエネルギー (``compute.out`` の最後の 2 列) の傾きから
    熱流を、中間ブロックの温度から勾配を取り、
    :math:`\\kappa = -Q / (A \\, dT/dx)` とする。

    Parameters
    ----------
    compute_frame
        :func:`gpumd_toolkit.outputs.read_compute` の戻り値
        (``quantities`` に ``temperature`` を含めること)。
    layout
        :class:`gpumd_toolkit.groups.NEMDLayout`。
    area
        輸送方向に垂直な断面積 [Å^2]。
    output_interval_fs
        ``compute`` の 1 行あたりの時間 [fs]
        (= ``sample_interval * output_interval * time_step``)。
    drop_fraction
        先頭から捨てる割合 (定常状態になるまでを除く)。
    """
    steady = _tail(compute_frame, drop_fraction)
    time_ps = np.arange(len(steady)) * output_interval_fs * 1e-3
    if time_ps.size < 2:
        raise ValueError("定常部分のデータが足りません。drop_fraction を下げてください。")

    powers = {}
    for name, column in (
        ("source", "thermostat_energy_source"),
        ("sink", "thermostat_energy_sink"),
    ):
        if column not in steady:
            raise ValueError(
                "compute.out に熱浴エネルギーの列がありません。"
                " compute の quantities に 'temperature' を入れてください。"
            )
        slope = np.polyfit(time_ps, steady[column].to_numpy(), 1)[0]
        powers[name] = float(slope)  # eV/ps

    # 定常状態では |source| ≒ |sink|。平均を熱流とする。
    heat_flux_ev_ps = 0.5 * (abs(powers["source"]) + abs(powers["sink"]))

    centers = [layout.block_centers[g] for g in layout.middle]
    temps = [float(steady[f"temperature_g{g}"].mean()) for g in layout.middle]
    gradient, _ = temperature_gradient(centers, temps)
    if abs(gradient) < 1e-12:
        raise ValueError("温度勾配がほぼゼロです。熱浴の設定を確認してください。")

    kappa = heat_flux_ev_ps * EV_PS_TO_WATT / (area * 1e-20 * abs(gradient) * 1e10)
    return {
        "kappa": float(kappa),
        "heat_flux_eV_per_ps": float(heat_flux_ev_ps),
        "gradient_K_per_A": float(gradient),
        "delta_T": float(temps[0] - temps[-1]),
        "power_source_eV_per_ps": powers["source"],
        "power_sink_eV_per_ps": powers["sink"],
        "area_A2": float(area),
    }


def green_kubo_kappa(
    hac_frame: pd.DataFrame,
    *,
    t_min: float,
    t_max: float,
    components: Sequence[str] = ("kappa_x", "kappa_y", "kappa_z"),
) -> dict[str, float]:
    """``hac.out`` の κ(t) をプラトー区間 [t_min, t_max] ps で平均する。"""
    if t_max <= t_min:
        raise ValueError("t_min < t_max である必要があります。")
    window = hac_frame[(hac_frame["time"] >= t_min) & (hac_frame["time"] <= t_max)]
    if window.empty:
        raise ValueError(f"[{t_min}, {t_max}] ps に相関データがありません。")
    result = {name: float(window[name].mean()) for name in components}
    result["kappa"] = float(np.mean(list(result.values())))
    result["n_points"] = float(len(window))
    return result


def hnemd_kappa(
    kappa_frame: pd.DataFrame,
    *,
    drop_fraction: float = 0.3,
    components: Sequence[str] = ("kappa_x", "kappa_y", "kappa_z"),
) -> dict[str, float]:
    """``kappa.out`` を平均して HNEMD の熱伝導率 [W/mK] と標準誤差を返す。"""
    steady = _tail(kappa_frame, drop_fraction)
    result: dict[str, float] = {}
    for name in components:
        values = steady[name].to_numpy()
        result[name] = float(values.mean())
        result[f"{name}_stderr"] = float(values.std(ddof=1) / np.sqrt(values.size))
    return result


def shear_viscosity(
    viscosity_frame: pd.DataFrame, *, t_min: float, t_max: float
) -> dict[str, float]:
    """``viscosity.out`` からせん断・縦・体積粘性率 [Pa s] を取り出す。"""
    if t_max <= t_min:
        raise ValueError("t_min < t_max である必要があります。")
    window = viscosity_frame[
        (viscosity_frame["time"] >= t_min) & (viscosity_frame["time"] <= t_max)
    ]
    if window.empty:
        raise ValueError(f"[{t_min}, {t_max}] ps に相関データがありません。")
    return {
        "eta_shear": float(window["eta_shear"].mean()),
        "eta_longitudinal": float(window["eta_longitudinal"].mean()),
        "eta_bulk": float(window["eta_bulk"].mean()),
        "eta_shear_stderr": float(
            window["eta_shear"].std(ddof=1) / np.sqrt(len(window))
        ),
    }


# ------------------------------------------------------------------ 拡散
def diffusion_from_msd(
    msd_frame: pd.DataFrame,
    *,
    fit_fraction: tuple[float, float] = (0.3, 0.9),
    group: int | None = None,
) -> dict[str, float]:
    """MSD の傾きから自己拡散係数 [cm^2/s] を求める。

    ``D = slope / 6`` (3 次元)。傾きは ``fit_fraction`` で指定した
    相関時間の範囲 (全体に対する割合) で直線フィットする。
    """
    suffix = "" if group is None else f"_g{group}"
    columns = [f"msd_{a}{suffix}" for a in ("x", "y", "z")]
    missing = [c for c in columns if c not in msd_frame]
    if missing:
        raise ValueError(f"msd.out に列がありません: {missing}")
    time = msd_frame["time"].to_numpy()
    total = msd_frame[columns].sum(axis=1).to_numpy()
    lo, hi = fit_fraction
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError("fit_fraction は 0 <= lo < hi <= 1 です。")
    start, stop = int(len(time) * lo), int(len(time) * hi)
    if stop - start < 2:
        raise ValueError("フィット範囲の点が足りません。")
    slope = np.polyfit(time[start:stop], total[start:stop], 1)[0]
    # Å^2/ps -> cm^2/s : 1 Å^2/ps = 1e-16 cm^2 / 1e-12 s = 1e-4 cm^2/s
    return {
        "D_A2_per_ps": float(slope / 6.0),
        "D_cm2_per_s": float(slope / 6.0 * 1e-4),
        "slope_A2_per_ps": float(slope),
        "fit_range_ps": (float(time[start]), float(time[stop - 1])),
    }


def diffusion_from_sdc(
    sdc_frame: pd.DataFrame, *, drop_fraction: float = 0.5, group: int | None = None
) -> dict[str, float]:
    """``sdc.out`` の SDC (VAC の積分) の収束値から拡散係数を求める。"""
    suffix = "" if group is None else f"_g{group}"
    columns = [f"sdc_{a}{suffix}" for a in ("x", "y", "z")]
    missing = [c for c in columns if c not in sdc_frame]
    if missing:
        raise ValueError(f"sdc.out に列がありません: {missing}")
    tail = _tail(sdc_frame, drop_fraction)
    per_axis = {c: float(tail[c].mean()) for c in columns}
    total = float(np.mean(list(per_axis.values())))
    return {
        **per_axis,
        "D_A2_per_ps": total,
        "D_cm2_per_s": total * 1e-4,
    }


def ionic_conductivity(
    ic_frame: pd.DataFrame, *, drop_fraction: float = 0.5
) -> dict[str, float]:
    """``ic.out`` のイオン伝導度 [mS/cm] を平均する。"""
    tail = _tail(ic_frame, drop_fraction)
    per_axis = {c: float(tail[c].mean()) for c in ("ic_x", "ic_y", "ic_z")}
    return {**per_axis, "ic": float(np.mean(list(per_axis.values())))}


# ------------------------------------------------------------ 自由エネルギー
def _forward_backward(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """TI の csv を往路 (lambda 0->1) と復路 (1->0) に分ける。

    GPUMD は折り返し点の行を 1 つ余分に書くことがあるので、
    行数が奇数のときは中央の 1 行を捨てて長さを揃える。
    """
    if len(frame) < 4:
        raise ValueError("TI の出力行数が少なすぎます (往復のデータが必要)。")
    n = len(frame) // 2
    forward = frame.iloc[:n].reset_index(drop=True)
    backward = frame.iloc[len(frame) - n :][::-1].reset_index(drop=True)
    return forward, backward


def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    from scipy.integrate import cumulative_trapezoid

    return cumulative_trapezoid(y, x, initial=0.0)


def gibbs_from_ti(summary: Mapping[str, float]) -> dict[str, float]:
    """``ti_spring.yaml`` / ``ti_liquid.yaml`` の内容を整理して返す。

    GPUMD の yaml では圧力 ``P`` が eV/Å^3、体積 ``V`` が Å^3/atom、
    エネルギーが eV/atom。``P`` を GPa に直した値も足す。
    """
    data = dict(summary)
    if "P" in data:
        # 1 eV/Å^3 = 160.21766208 GPa
        data["P_GPa"] = data["P"] * 160.21766208
    return data


def free_energy_reversible_scaling(
    ti_rs: pd.DataFrame, *, T0: float, G0: float
) -> pd.DataFrame:
    """可逆スケーリング (``ti_rs.csv``) から G(T) [eV/atom] を求める。

    Parameters
    ----------
    T0, G0
        参照温度 [K] とそこでの Gibbs 自由エネルギー [eV/atom]
        (``ti_spring.yaml`` の ``T`` と ``G``)。
    """
    forward, backward = _forward_backward(ti_rs)
    lam = forward["lambda"].to_numpy()
    work = 0.5 * (
        _cumtrapz(forward["enthalpy"].to_numpy(), lam)
        + _cumtrapz(backward["enthalpy"].to_numpy(), lam)
    )
    temperature = T0 / lam
    gibbs = (G0 + 1.5 * KB_EV * T0 * np.log(lam) + work) / lam
    return pd.DataFrame({"lambda": lam, "temperature": temperature, "G": gibbs})


def free_energy_adiabatic_switching(
    ti_as: pd.DataFrame, *, G0: float
) -> pd.DataFrame:
    """断熱スイッチング (``ti_as.csv``) から G(P) [eV/atom] を求める。

    Parameters
    ----------
    G0
        出発圧力での Gibbs 自由エネルギー [eV/atom]。
    """
    forward, backward = _forward_backward(ti_as)
    pressure = forward["p"].to_numpy()
    work = 0.5 * (
        _cumtrapz(forward["V"].to_numpy(), pressure)
        + _cumtrapz(backward["V"].to_numpy(), pressure)
    )
    # GPUMD の p は eV/Å^3
    return pd.DataFrame(
        {
            "pressure_eV_per_A3": pressure,
            "pressure_GPa": pressure * 160.21766208,
            "G": G0 + work,
        }
    )


# ------------------------------------------------------------------ 分光
def vibrational_spectrum(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    time_step_fs: float,
    max_wavenumber: float = 4000.0,
) -> pd.DataFrame:
    """双極子/分極率の時系列から自己相関のフーリエ変換スペクトルを作る。

    ``dump_dipole`` の結果を渡せば赤外、``dump_polarizability`` を渡せば
    ラマンの生スペクトルになる (規格化・量子補正は行わない)。

    Parameters
    ----------
    time_step_fs
        サンプル間の時間 [fs] (= ``interval * time_step``)。
    max_wavenumber
        出力する最大波数 [cm^-1]。
    """
    signals = [frame[c].to_numpy(dtype=float) for c in columns]
    n = len(signals[0])
    if n < 4:
        raise ValueError("データ点が少なすぎます。")
    total = np.zeros(n)
    for signal in signals:
        centered = signal - signal.mean()
        spectrum = np.fft.rfft(centered, n=2 * n)
        acf = np.fft.irfft(np.abs(spectrum) ** 2)[:n]
        total += acf / np.arange(n, 0, -1)
    total /= len(signals)
    window = np.hanning(2 * n)[n:]
    intensity = np.abs(np.fft.rfft(total * window))
    freq_thz = np.fft.rfftfreq(n, d=time_step_fs * 1e-3)  # THz
    wavenumber = freq_thz * 33.35641  # THz -> cm^-1
    mask = wavenumber <= max_wavenumber
    return pd.DataFrame(
        {
            "wavenumber_cm-1": wavenumber[mask],
            "frequency_THz": freq_thz[mask],
            "intensity": intensity[mask],
        }
    )


# ------------------------------------------------------------ Active learning
def committee_uncertainty_summary(
    active_frame: pd.DataFrame, *, threshold: float | None = None
) -> dict[str, float]:
    """``active.out`` の不確かさ分布をまとめる。"""
    values = active_frame["uncertainty"].to_numpy(dtype=float)
    summary = {
        "n_samples": float(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "max": float(values.max()),
        "p95": float(np.percentile(values, 95)),
    }
    if threshold is not None:
        exceeded = values > threshold
        summary["threshold"] = float(threshold)
        summary["n_exceeded"] = float(exceeded.sum())
        summary["fraction_exceeded"] = float(exceeded.mean())
    return summary
