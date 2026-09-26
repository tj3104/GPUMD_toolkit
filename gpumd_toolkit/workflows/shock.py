"""9. 高圧・衝撃波.

=========================================== ===================================
手法                                         特徴
=========================================== ===================================
MSST (``msst``)                              小さいセルで定常衝撃波後方の
                                             状態を再現する。安価。
NPHug (``nphug``)                            目標応力を与えて Hugoniot 上の
                                             状態に収束させる (Hugoniostat)。
NEMD ピストン (``wall_*``)                   実際に壁を動かして衝撃波を走らせる。
                                             波面の構造が見える。セルは大きく必要。
静水圧の段階加圧 (``npt_*``)                 状態方程式・高圧相転移。
=========================================== ===================================

NEMD ピストンは **x 方向** にしか走らない。x を非周期にしてはいけない
(GPUMD が反対側に真空層を自動で足す)。

Examples
--------
>>> from gpumd_toolkit.workflows import ShockCalculation
>>> calc = ShockCalculation("POSCAR", "nep.txt", "runs/shock", repeat=(60, 4, 4))
>>> calc.equilibrate(temperature=300, steps=20000)
>>> calc.piston(vp=2.0, steps=100000, bin_size=10.0)
>>> calc.run()
>>> calc.shock_front_position().tail()
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..inputs import dumps
from ..inputs.ensembles import MSST, NPHug, Wall
from ..md import GPUMDCalculation
from ..outputs import read_shock_profiles, read_thermo

__all__ = ["ShockCalculation"]


class ShockCalculation(GPUMDCalculation):
    """衝撃波・高圧のワークフロー。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._bin_size: float | None = None
        self._piston_vp: float | None = None

    # ------------------------------------------------------------------ 準備
    def equilibrate(
        self,
        *,
        temperature: float,
        steps: int,
        pressure: float | None = None,
        **kwargs,
    ) -> "ShockCalculation":
        """衝撃を与える前の平衡化 (p0, v0, e0 の基準になるので必須)。"""
        if pressure is None:
            self.nvt(temperature=temperature, steps=steps, **kwargs)
        else:
            self.npt(temperature=temperature, steps=steps, pressure=pressure, **kwargs)
        return self

    # ------------------------------------------------------------------ MSST
    def msst(
        self,
        *,
        shock_velocity: float,
        steps: int,
        direction: str = "x",
        qmass: float = 10000.0,
        mu: float = 10.0,
        tscale: float | None = None,
        p0: float | None = None,
        v0: float | None = None,
        e0: float | None = None,
        **kwargs,
    ) -> "ShockCalculation":
        """Multi-Scale Shock Technique。

        Parameters
        ----------
        shock_velocity
            衝撃波速度 [km/s]。
        qmass
            セルの慣性質量 [amu^2/Å^4]。小さいと圧縮が速すぎて振動する。
        mu
            人工粘性 [sqrt(amu eV)/Å^2]。収束を助ける。
        tscale
            運動エネルギーのうちセルの運動に回す割合。収束を速める。
        p0, v0, e0
            初期状態。省略すると最初のステップで自動的に決まる。
        """
        spec = MSST(
            direction=direction,
            shock_velocity=shock_velocity,
            qmass=qmass,
            mu=mu,
            tscale=tscale,
            p0=p0,
            v0=v0,
            e0=e0,
        )
        return self.add_ensemble(spec, steps=steps, **kwargs)

    def hugoniostat(
        self,
        *,
        pressure: float,
        steps: int,
        direction: str | Sequence[str] = "iso",
        T_period: float = 100.0,
        p_period: float = 1000.0,
        tau_T: float | None = None,
        tau_p: float | None = None,
        p0: float | None = None,
        v0: float | None = None,
        e0: float | None = None,
        **kwargs,
    ) -> "ShockCalculation":
        """NPHug: 目標応力 [GPa] を与えて Hugoniot 上の状態に収束させる。

        1 点ずつ圧力を変えて走らせると Hugoniot 曲線が描ける。
        """
        spec = NPHug(
            direction=direction,
            pressure=pressure,
            T_period=T_period,
            tau_T=tau_T,
            p_period=p_period,
            tau_p=tau_p,
            p0=p0,
            v0=v0,
            e0=e0,
        )
        return self.add_ensemble(spec, steps=steps, **kwargs)

    def piston(
        self,
        *,
        vp: float,
        steps: int,
        kind: str = "piston",
        thickness: float | None = None,
        k: float | None = None,
        bin_size: float = 10.0,
        dump_interval: int = 1000,
        **kwargs,
    ) -> "ShockCalculation":
        """NEMD ピストンで実際に衝撃波を x 方向へ走らせる。

        Parameters
        ----------
        vp
            ピストン速度 [km/s]。
        kind
            ``'piston'`` (固定原子層) / ``'mirror'`` (運動量ミラー) /
            ``'harmonic'`` (調和ポテンシャル壁)。
        bin_size
            空間分布の出力ビン幅 [Å]。
        """
        spec = Wall(vp=vp, kind=kind, thickness=thickness, k=k)
        self.check_piston_steps(vp, steps, thickness=thickness)
        self.add_ensemble(spec, steps=steps, **kwargs)
        self._bin_size = bin_size
        self._piston_vp = vp
        return self.add_commands(dumps.dump_shock_nemd(dump_interval, bin_size))

    def max_piston_steps(self, vp: float, *, thickness: float | None = None) -> int:
        """ピストンがセルを走り切るまでのステップ数。

        これを超えて走らせると材料が固定壁に押し潰され、密度が発散して
        GPUMD が ``illegal memory access`` で落ちる。
        """
        length = float(np.linalg.norm(self.atoms.cell[0]))
        wall = 2.0 * (thickness if thickness is not None else 20.0)
        usable = max(length - wall, 0.0)
        speed = vp / 100.0  # km/s -> Å/fs
        if speed <= 0:
            raise ValueError("vp [km/s] は正の値です。")
        return int(usable / speed / self.builder.time_step)

    def check_piston_steps(
        self, vp: float, steps: int, *, thickness: float | None = None, strict: bool = False
    ) -> int:
        """走らせるステップ数がセルの長さに見合っているか確認する。"""
        limit = self.max_piston_steps(vp, thickness=thickness)
        if steps > limit:
            message = (
                f"ピストンは約 {limit:,d} ステップでセルを走り切ります"
                f" (vp={vp:g} km/s, 有効長 {float(np.linalg.norm(self.atoms.cell[0])):.0f} Å)。"
                f" {steps:,d} ステップは長すぎるので材料が固定壁に潰され、"
                " GPUMD が illegal memory access で落ちます。"
                " ステップ数を減らすか、x 方向のセルを長くしてください。"
            )
            if strict:
                raise ValueError(message)
            warnings.warn(message, RuntimeWarning)
        return limit

    def compression_ramp(
        self,
        *,
        temperature: float,
        p_start: float,
        p_end: float,
        steps: int,
        barostat: str = "npt_mttk",
        direction: str = "iso",
        **kwargs,
    ) -> "ShockCalculation":
        """静水圧を連続的に上げる (状態方程式・高圧相転移の探索)。"""
        return self.npt(
            temperature=temperature,
            steps=steps,
            pressure=p_start,
            pressure_end=p_end,
            barostat=barostat,
            mttk_direction=direction,
            **kwargs,
        )

    # ------------------------------------------------------------------ 結果
    def profiles(self, *, bin_size: float | None = None) -> dict[str, pd.DataFrame]:
        """``dump_shock_nemd`` の x 方向空間分布をまとめて読む。

        キーは ``temperature`` / ``pxx`` / ``pyy`` / ``pzz`` / ``density`` / ``vp``。
        """
        return read_shock_profiles(
            self.workdir, bin_size=bin_size if bin_size is not None else self._bin_size
        )

    def shock_front_position(self, *, key: str = "density") -> pd.DataFrame:
        """各出力時刻の衝撃波面位置 [Å] を推定する。

        波面はプロファイルが最も急に変化する場所なので、
        隣り合うビンの差分の絶対値が最大になる位置を放物線補間して返す。
        圧縮側が x の大小どちら向きでも同じように働く。

        波面がセル端に達した後 (反射が始まった後) のフレームには
        ``reflected`` の印を付け、:meth:`shock_velocity` では使わない。

        Notes
        -----
        分解能はビン幅で決まる。``piston(bin_size=...)`` はセル長の
        1/50 程度まで小さくしておくとよい。
        """
        frame = self.profiles()[key]
        positions = np.asarray(frame.columns, dtype=float)
        if positions.size < 5:
            raise ValueError(
                f"ビンが {positions.size} 個しかありません。"
                " piston(bin_size=...) を小さくして取り直してください。"
            )
        rows = []
        for index, row in frame.iterrows():
            values = row.to_numpy(dtype=float)
            if not np.all(np.isfinite(values)):
                continue
            gradient = np.abs(np.diff(values))
            if gradient.max() <= 0:
                continue
            i = int(np.argmax(gradient))
            # 隣接する 3 点を放物線で補間して副ビン精度にする
            if 0 < i < len(gradient) - 1:
                a, b, c = gradient[i - 1], gradient[i], gradient[i + 1]
                denominator = a - 2 * b + c
                shift = 0.5 * (a - c) / denominator if denominator != 0 else 0.0
                shift = float(np.clip(shift, -1.0, 1.0))
            else:
                shift = 0.0
            spacing = positions[1] - positions[0]
            front = 0.5 * (positions[i] + positions[i + 1]) + shift * spacing
            rows.append({"frame": index, "front": float(front),
                         "contrast": float(gradient.max())})
        result = pd.DataFrame(rows)
        if result.empty:
            raise ValueError("波面を検出できませんでした。key を変えてみてください。")
        front = result["front"].to_numpy()
        reflected = np.zeros(len(front), dtype=bool)
        # 単調に進んでいる区間だけを反射前とみなす
        peak = int(np.argmax(front)) if front[-1] < front.max() else len(front) - 1
        reflected[peak + 1:] = True
        reflected |= front >= positions[-1] - (positions[1] - positions[0])
        result["reflected"] = reflected
        result["front_velocity_A_per_frame"] = result["front"].diff()
        return result

    def shock_velocity(self, *, output_interval_fs: float, key: str = "density") -> float:
        """波面位置の時間変化から衝撃波速度 [km/s] を求める。

        反射が始まる前のフレームだけを直線フィットする。
        """
        front = self.shock_front_position(key=key)
        usable = front[~front["reflected"]]
        if len(usable) < 3:
            raise ValueError(
                "反射前のフレームが 3 つ未満です。"
                " セルを長くするか dump 間隔を短くしてください。"
            )
        time_ps = usable["frame"].to_numpy() * output_interval_fs * 1e-3
        slope = np.polyfit(time_ps, usable["front"].to_numpy(), 1)[0]  # Å/ps
        return float(slope * 0.1)  # Å/ps -> km/s

    def hugoniot_state(self, *, drop_fraction: float = 0.5) -> dict[str, float]:
        """MSST / NPHug の収束後の状態 (P, V, T, E) を返す。"""
        frame = read_thermo(self.workdir / "thermo.out")
        tail = frame.iloc[int(len(frame) * drop_fraction) :]
        return {
            "temperature": float(tail["temperature"].mean()),
            "pressure_GPa": float(tail["pressure"].mean()),
            "Pxx_GPa": float(tail["Pxx"].mean()),
            "volume_A3": float(tail["volume"].mean()),
            "volume_per_atom": float(tail["volume"].mean() / self.n_atoms),
            "total_energy_eV": float(tail["total_energy"].mean()),
            "energy_per_atom": float(tail["total_energy"].mean() / self.n_atoms),
        }

    # ------------------------------------------------------------------ 作図
    def plot_profiles(
        self,
        filename: str = "shock_profiles.png",
        *,
        frames: Sequence[int] | None = None,
        dpi: int = 150,
    ) -> Path:
        """衝撃波の空間分布 (温度・圧力・密度・粒子速度) を描く。"""
        from ..plotting import label, setup_matplotlib

        setup_matplotlib()
        import matplotlib.pyplot as plt

        data = self.profiles()
        keys = [k for k in ("temperature", "pxx", "density", "vp") if k in data]
        figure, axes = plt.subplots(len(keys), 1, figsize=(7, 2.4 * len(keys)), sharex=True)
        axes = np.atleast_1d(axes)
        labels = {
            "temperature": "T (K)",
            "pxx": r"$P_{xx}$ (GPa)",
            "density": r"$\rho$ (g/cm$^3$)",
            "vp": r"$v_p$ (km/s)",
        }
        for axis, key in zip(axes, keys):
            frame = data[key]
            positions = np.asarray(frame.columns, dtype=float)
            indices = frames if frames is not None else np.linspace(
                0, len(frame) - 1, min(6, len(frame)), dtype=int
            )
            for i in indices:
                axis.plot(positions, frame.iloc[int(i)], lw=1.0, label=f"#{int(i)}")
            axis.set_ylabel(labels[key])
        axes[-1].set_xlabel("x (Å)" if self._bin_size else "ビン番号")
        axes[0].legend(fontsize=7, ncol=3)
        figure.suptitle(label(f"衝撃波の空間分布 — {self.name}"))
        figure.tight_layout()
        output = self.workdir / filename
        figure.savefig(output, dpi=dpi)
        plt.close(figure)
        return output
