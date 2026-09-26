"""熱輸送 (Green-Kubo / HNEMD / NEMD / スペクトル分解 / モード分解).

mission.md の分類には明示されていないが、GPUMD の中核機能なので独立させた。
拡散・イオン伝導・液体物性は :mod:`gpumd_toolkit.workflows.diffusion` にある。

================================ ==============================================
手法                              クラスのメソッド
================================ ==============================================
平衡 MD + Green-Kubo              :meth:`ThermalTransport.green_kubo`
均一非平衡 MD (HNEMD)             :meth:`ThermalTransport.hnemd`
多成分 HNEMDEC (Onsager 係数)     :meth:`ThermalTransport.hnemdec`
非平衡 MD (source/sink, NEMD)     :meth:`ThermalTransport.nemd`
スペクトル熱流 (SHC)              :meth:`ThermalTransport.spectral`
モード分解 (GKMA / HNEMA)         :meth:`ThermalTransport.modal`
フォノン状態密度                  :meth:`ThermalTransport.phonon_dos`
================================ ==============================================

Examples
--------
>>> from gpumd_toolkit.workflows import ThermalTransport
>>> calc = ThermalTransport("POSCAR", "nep.txt", "runs/kappa", min_cell_length=20.0)
>>> calc.equilibrate(temperature=300, steps=50000)
>>> calc.hnemd(temperature=300, steps=2000000, Fe=1e-5, direction="x")
>>> calc.run()
>>> calc.thermal_conductivity()
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from ase import Atoms

from ..groups import GroupingScheme, NEMDLayout, nemd_layout
from ..inputs import computes, modifiers
from ..md import GPUMDCalculation
from ..outputs import (
    read_compute,
    read_hac,
    read_heatmode,
    read_kappa,
    read_kappamode,
    read_shc,
)
from ..postprocess import green_kubo_kappa, hnemd_kappa, nemd_thermal_conductivity

__all__ = ["ThermalTransport"]


class ThermalTransport(GPUMDCalculation):
    """熱伝導率の計算をまとめるワークフロー。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        #: NEMD で使うグループ配置
        self.layout: NEMDLayout | None = None
        #: :meth:`nemd` の ``compute`` の 1 行あたりの時間 [fs]
        self._compute_interval_fs: float | None = None
        self._hac_settings: dict | None = None

    # ------------------------------------------------------------------ 準備
    def equilibrate(
        self,
        *,
        temperature: float,
        steps: int,
        pressure: float | None = None,
        thermostat: str = "nvt_nhc",
        **kwargs,
    ) -> "ThermalTransport":
        """熱輸送の本計算の前に平衡化する。"""
        if pressure is None:
            self.nvt(temperature=temperature, steps=steps, thermostat=thermostat, **kwargs)
        else:
            self.npt(temperature=temperature, steps=steps, pressure=pressure, **kwargs)
        return self

    # ------------------------------------------------------------- 平衡 (GK)
    def green_kubo(
        self,
        *,
        temperature: float,
        steps: int,
        sampling_interval: int = 10,
        correlation_steps: int = 50000,
        output_interval: int = 1,
        ensemble: str = "nve",
        **kwargs,
    ) -> "ThermalTransport":
        """平衡 MD + Green-Kubo (``compute_hac``) で熱伝導率を出す。

        GPUMD の定石では平衡化を NVT で行い、生成 run は NVE にする
        (熱浴が熱流を乱さないようにするため)。
        """
        if ensemble == "nve":
            self.nve(steps=steps, **kwargs)
        else:
            self.nvt(temperature=temperature, steps=steps, thermostat=ensemble, **kwargs)
        self._hac_settings = {
            "sampling_interval": sampling_interval,
            "correlation_steps": correlation_steps,
        }
        return self.add_commands(
            computes.compute_hac(sampling_interval, correlation_steps, output_interval)
        )

    # ------------------------------------------------------------- HNEMD
    def hnemd(
        self,
        *,
        temperature: float,
        steps: int,
        Fe: float = 1.0e-5,
        direction: str = "x",
        output_interval: int = 1000,
        thermostat: str = "nvt_nhc",
        **kwargs,
    ) -> "ThermalTransport":
        """均一非平衡 MD (HNEMD)。駆動力 Fe [1/Å] を全原子にかける。

        Langevin 熱浴 (``nvt_lan`` / ``nvt_bao``) はダイナミクスを乱すので
        使わないこと。Nose-Hoover chain が推奨。
        """
        if thermostat in ("nvt_lan", "nvt_bao"):
            raise ValueError(
                "HNEMD に Langevin 熱浴は使えません (熱流を乱します)。"
                " nvt_nhc か nvt_bdp を使ってください。"
            )
        self.nvt(temperature=temperature, steps=steps, thermostat=thermostat, **kwargs)
        return self.add_commands(
            computes.compute_hnemd(output_interval, Fe, direction=direction)
        )

    def hnemdec(
        self,
        *,
        temperature: float,
        steps: int,
        driving_force: int = 0,
        Fe: float = 1.0e-5,
        direction: str = "x",
        output_interval: int = 1000,
        thermostat: str = "nvt_nhc",
        **kwargs,
    ) -> "ThermalTransport":
        """多成分系の Onsager 係数 (HNEMDEC)。

        ``driving_force=0`` なら熱流、正の整数 i なら i 番目の元素の
        運動量流を散逸流にする。
        """
        self.nvt(temperature=temperature, steps=steps, thermostat=thermostat, **kwargs)
        return self.add_commands(
            computes.compute_hnemdec(driving_force, output_interval, Fe, direction=direction)
        )

    # ------------------------------------------------------------- NEMD
    @classmethod
    def for_nemd(
        cls,
        structure,
        potential,
        workdir,
        *,
        axis: str = "x",
        n_blocks: int = 10,
        fixed_ends: bool = True,
        **kwargs,
    ) -> "ThermalTransport":
        """NEMD 用のグループ分けを済ませたインスタンスを作る。

        軸方向を ``n_blocks`` に分け、両端に固定層 (``fixed_ends``) を置く。
        ``model.xyz`` に grouping method 0 として書き込まれる。
        """
        instance = cls(structure, potential, workdir, **kwargs)
        layout = nemd_layout(
            instance.atoms, axis=axis, n_blocks=n_blocks, fixed_ends=fixed_ends
        )
        scheme = GroupingScheme()
        scheme.add(layout.labels, "nemd")
        instance.grouping_scheme = scheme
        instance.groupings = scheme.to_groupings()
        instance.layout = layout
        return instance

    def nemd(
        self,
        *,
        temperature: float,
        steps: int,
        delta_T: float,
        method: str = "lan",
        sample_interval: int = 10,
        output_interval: int = 100,
        T_coup: float = 100.0,
        tau_T: float | None = None,
        fix_ends: bool = True,
        **kwargs,
    ) -> "ThermalTransport":
        """source/sink 熱浴による NEMD。温度勾配と熱流から熱伝導率を出す。

        :meth:`for_nemd` でグループを作ってから呼ぶこと。
        ``compute`` で各ブロックの温度と熱浴エネルギーを記録する。
        """
        from ..inputs.ensembles import HeatBath

        if self.layout is None:
            raise RuntimeError(
                "NEMD のグループ配置がありません。"
                " ThermalTransport.for_nemd(...) で作ってください。"
            )
        layout = self.layout
        spec = HeatBath(
            temperature=temperature,
            delta_T=delta_T,
            source=layout.source,
            sink=layout.sink,
            method=method,
            T_coup=T_coup,
            tau_T=tau_T,
        )
        commands = [computes.compute(0, sample_interval, output_interval, ("temperature",))]
        if fix_ends and layout.fixed:
            # GPUMD の fix は 1 run につき 1 グループのみ有効。
            # nemd_layout は両端を同じグループにまとめてある。
            commands = [modifiers.fix(layout.fixed[0])] + commands
        self.add_ensemble(spec, steps=steps, **kwargs)
        self._compute_interval_fs = (
            sample_interval * output_interval * self.builder.time_step
        )
        return self.add_commands(*commands)

    # ------------------------------------------------------- スペクトル・モード
    def spectral(
        self,
        *,
        sample_interval: int = 2,
        Nc: int = 250,
        direction: str = "x",
        num_omega: int = 1000,
        max_omega: float = 400.0,
        group: Sequence[int] | None = None,
    ) -> "ThermalTransport":
        """直前のステージにスペクトル熱流 (``compute_shc``) を足す。"""
        return self.add_commands(
            computes.compute_shc(
                sample_interval, Nc, direction, num_omega, max_omega, group=group
            )
        )

    def modal(
        self,
        *,
        method: str = "gkma",
        first_mode: int = 1,
        last_mode: int | None = None,
        sample_interval: int = 10,
        output_interval: int = 1000,
        bin_option: str = "f_bin_size",
        size: float = 1.0,
        Fe: float = 1.0e-5,
        direction: str = "x",
    ) -> "ThermalTransport":
        """モード分解熱輸送 (GKMA / HNEMA)。``eigenvector.in`` が必要。

        Parameters
        ----------
        method
            ``'gkma'`` (平衡、熱流のモード分解) か
            ``'hnema'`` (非平衡、熱伝導率のモード分解)。
        last_mode
            既定は 3N (全モード)。
        """
        if last_mode is None:
            last_mode = 3 * self.n_atoms
        if not (self.workdir / "eigenvector.in").is_file():
            raise FileNotFoundError(
                f"{self.workdir}/eigenvector.in がありません。"
                " GPUMD の tools か phonopy で作成してください。"
            )
        if method == "gkma":
            line = computes.compute_gkma(
                sample_interval, first_mode, last_mode, bin_option, size
            )
        elif method == "hnema":
            line = computes.compute_hnema(
                sample_interval,
                output_interval,
                Fe,
                first_mode,
                last_mode,
                bin_option,
                size,
                direction=direction,
            )
        else:
            raise ValueError("method は 'gkma' か 'hnema' です。")
        return self.add_commands(line)

    def phonon_dos(
        self,
        *,
        sample_interval: int = 5,
        Nc: int = 200,
        omega_max: float = 400.0,
        group: Sequence[int] | None = None,
        num_dos_points: int | None = None,
    ) -> "ThermalTransport":
        """直前のステージにフォノン状態密度 (``compute_dos``) を足す。"""
        return self.add_commands(
            computes.compute_dos(
                sample_interval,
                Nc,
                omega_max,
                group=group,
                num_dos_points=num_dos_points,
            )
        )

    # ------------------------------------------------------------------ 結果
    def thermal_conductivity(self, **kwargs) -> dict[str, float]:
        """仕込んだ手法に応じた熱伝導率 [W/mK] を返す。"""
        if (self.workdir / "kappa.out").is_file():
            return self.hnemd_result(**kwargs)
        if (self.workdir / "hac.out").is_file():
            return self.green_kubo_result(**kwargs)
        if (self.workdir / "compute.out").is_file() and self.layout is not None:
            return self.nemd_result(**kwargs)
        raise FileNotFoundError(
            "kappa.out / hac.out / compute.out のいずれもありません。"
            " 先に run() を実行してください。"
        )

    def hnemd_result(self, *, drop_fraction: float = 0.3, **_) -> dict[str, float]:
        """``kappa.out`` を平均して HNEMD の熱伝導率を返す。"""
        return hnemd_kappa(read_kappa(self.workdir / "kappa.out"), drop_fraction=drop_fraction)

    def green_kubo_result(
        self, *, t_min: float | None = None, t_max: float | None = None, **_
    ) -> dict[str, float]:
        """``hac.out`` のプラトーを平均して Green-Kubo 熱伝導率を返す。

        範囲を省略すると相関時間の後半 50-90% を使う。
        """
        frame = read_hac(self.workdir / "hac.out")
        total = float(frame["time"].max())
        return green_kubo_kappa(
            frame,
            t_min=t_min if t_min is not None else 0.5 * total,
            t_max=t_max if t_max is not None else 0.9 * total,
        )

    def nemd_result(self, *, drop_fraction: float = 0.5, **_) -> dict[str, float]:
        """``compute.out`` の温度勾配と熱浴出力から NEMD 熱伝導率を返す。"""
        if self.layout is None or self._compute_interval_fs is None:
            raise RuntimeError("NEMD のステージが仕込まれていません。")
        frame = read_compute(
            self.workdir / "compute.out",
            n_groups=self.layout.n_groups,
            quantities=("temperature",),
        )
        return nemd_thermal_conductivity(
            frame,
            layout=self.layout,
            area=self.layout.cross_section(self.atoms),
            output_interval_fs=self._compute_interval_fs,
            drop_fraction=drop_fraction,
        )

    def temperature_profile(self, *, drop_fraction: float = 0.5) -> pd.DataFrame:
        """NEMD の定常温度プロファイル (位置 [Å] と温度 [K])。"""
        if self.layout is None:
            raise RuntimeError("NEMD のグループ配置がありません。")
        frame = read_compute(
            self.workdir / "compute.out",
            n_groups=self.layout.n_groups,
            quantities=("temperature",),
        )
        steady = frame.iloc[int(len(frame) * drop_fraction) :]
        groups = list(range(self.layout.n_groups))
        return pd.DataFrame(
            {
                "group": groups,
                "position": [self.layout.block_centers[g] for g in groups],
                "temperature": [float(steady[f"temperature_g{g}"].mean()) for g in groups],
                "is_middle": [g in self.layout.middle for g in groups],
            }
        )

    def spectral_result(self, *, Nc: int | None = None):
        """``shc.out`` を (相関関数, スペクトル) として読む。"""
        return read_shc(self.workdir / "shc.out", Nc=Nc)

    def modal_result(self, *, method: str = "gkma") -> pd.DataFrame:
        """``heatmode.out`` / ``kappamode.out`` を読む。"""
        if method == "gkma":
            return read_heatmode(self.workdir / "heatmode.out")
        return read_kappamode(self.workdir / "kappamode.out")

    # ------------------------------------------------------------------ 作図
    def plot_temperature_profile(
        self, filename: str = "nemd_profile.png", *, dpi: int = 150
    ) -> Path:
        """NEMD の温度プロファイルと直線フィットを描く。"""
        from ..plotting import label, setup_matplotlib

        setup_matplotlib()
        import matplotlib.pyplot as plt

        frame = self.temperature_profile()
        middle = frame[frame["is_middle"]]
        slope, intercept = np.polyfit(middle["position"], middle["temperature"], 1)
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.plot(frame["position"], frame["temperature"], "o-", ms=4, label="各ブロック")
        axis.plot(
            middle["position"],
            slope * middle["position"] + intercept,
            "--",
            color="C3",
            label=f"勾配 {slope:.3f} K/Å",
        )
        axis.set_xlabel(label(f"{self.layout.axis} (Å)"))
        axis.set_ylabel(label("温度 (K)"))
        axis.set_title(label(f"NEMD 温度プロファイル — {self.name}"))
        axis.legend()
        figure.tight_layout()
        output = self.workdir / filename
        figure.savefig(output, dpi=dpi)
        plt.close(figure)
        return output

    def plot_kappa(self, filename: str = "kappa.png", *, dpi: int = 150) -> Path:
        """HNEMD / Green-Kubo の熱伝導率の収束を描く。"""
        from ..plotting import label, setup_matplotlib

        setup_matplotlib()
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(6, 4))
        if (self.workdir / "kappa.out").is_file():
            frame = read_kappa(self.workdir / "kappa.out")
            x = np.arange(len(frame))
            for name in ("kappa_x", "kappa_y", "kappa_z"):
                axis.plot(x, frame[f"{name}_cum"], lw=1.2, label=f"{name} (累積平均)")
            axis.set_xlabel(label("出力回数"))
        else:
            frame = read_hac(self.workdir / "hac.out")
            for name in ("kappa_x", "kappa_y", "kappa_z"):
                axis.plot(frame["time"], frame[name], lw=1.2, label=name)
            axis.set_xlabel(label("相関時間 (ps)"))
        axis.set_ylabel(r"$\kappa$ (W m$^{-1}$ K$^{-1}$)")
        axis.set_title(label(f"熱伝導率の収束 — {self.name}"))
        axis.legend(fontsize=8)
        figure.tight_layout()
        output = self.workdir / filename
        figure.savefig(output, dpi=dpi)
        plt.close(figure)
        return output
