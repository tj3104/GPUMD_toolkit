"""10. Two-Temperature Model (二温度モデル).

レーザー照射・電子線照射・高エネルギー粒子の入射では、まず電子系が
エネルギーを受け取り、電子-フォノン結合を通じて格子に渡る。
TTM は電子温度 T_e の連続体グリッドと MD 原子を摩擦項で結合して
これを扱う。

======================================== ====================================
キーワード                                役割
======================================== ====================================
``ensemble ttm``                          電子グリッド + MD
``ensemble heat_ttm``                     局所 source/sink Langevin + 電子グリッド
``ttm_source``                            体積熱源 (レーザー吸収)
``ttm_infile``                            初期電子温度分布
``ttm_properties_file``                   セルごとの電子物性 (C, kappa, gamma, eta)
``ttm_active_*``                          活性グリッド領域
``electron_stop``                         高速原子への電子阻止能 (照射損傷)
======================================== ====================================

Examples
--------
>>> from gpumd_toolkit.workflows import TwoTemperatureCalculation
>>> calc = TwoTemperatureCalculation("POSCAR", "nep.txt", "runs/ttm")
>>> calc.equilibrate(temperature=300, steps=10000)
>>> calc.ttm(steps=50000, Ce=1.0, rho_e=1.0, kappa_e=0.005, gamma_p=0.01,
...          grid=(1, 1, 12), T_e_init=300, source=0.05, out_interval=50)
>>> calc.run()
>>> calc.electron_temperature().profile("z")
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..inputs import modifiers
from ..inputs.ensembles import TTM, HeatTTM, TTMOptions
from ..md import GPUMDCalculation
from ..outputs import TTMSnapshots, read_thermo, read_ttm_electron_temperature

__all__ = [
    "TwoTemperatureCalculation",
    "write_electron_temperature_file",
    "write_electron_properties_file",
]


def write_electron_temperature_file(
    path: Path | str, temperatures: np.ndarray
) -> Path:
    """``ttm_infile`` 用の初期電子温度ファイルを書く。

    ``temperatures`` は ``(nx, ny, nz)`` の配列 [K]。
    行は ``ix iy iz T_e`` (インデックスは 1 始まり)。
    """
    array = np.asarray(temperatures, dtype=float)
    if array.ndim != 3:
        raise ValueError("temperatures は (nx, ny, nz) の 3 次元配列です。")
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    nx, ny, nz = array.shape
    lines = [
        f"{ix + 1} {iy + 1} {iz + 1} {array[ix, iy, iz]:.6f}"
        for iz in range(nz)
        for iy in range(ny)
        for ix in range(nx)
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def write_electron_properties_file(
    path: Path | str,
    *,
    C_vol: np.ndarray,
    kappa_e: np.ndarray,
    gamma_p: np.ndarray,
    eta: np.ndarray,
) -> Path:
    """``ttm_properties_file`` 用のセルごとの電子物性ファイルを書く。

    各引数は ``(nx, ny, nz)`` の配列。
    ``C_vol`` [eV/(K Å^3)]、``kappa_e`` [eV/(ps K Å)]、``gamma_p`` [amu/ps]、
    ``eta`` は熱源吸収効率 (無次元)。
    """
    arrays = [np.asarray(a, dtype=float) for a in (C_vol, kappa_e, gamma_p, eta)]
    shape = arrays[0].shape
    if any(a.shape != shape for a in arrays) or len(shape) != 3:
        raise ValueError("4 つの配列はすべて同じ (nx, ny, nz) 形状である必要があります。")
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    nx, ny, nz = shape
    lines = []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                values = " ".join(f"{a[ix, iy, iz]:.8g}" for a in arrays)
                lines.append(f"{ix + 1} {iy + 1} {iz + 1} {values}")
    path.write_text("\n".join(lines) + "\n")
    return path


class TwoTemperatureCalculation(GPUMDCalculation):
    """二温度モデルのワークフロー。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.grid: tuple[int, int, int] | None = None

    def equilibrate(
        self, *, temperature: float, steps: int, **kwargs
    ) -> "TwoTemperatureCalculation":
        """TTM の前に格子を平衡化する。"""
        self.nvt(temperature=temperature, steps=steps, **kwargs)
        return self

    def _options(
        self,
        out_interval: int | None,
        infile: str | None,
        properties_file: str | None,
        source: float | None,
        active: dict[str, str | int] | None,
    ) -> TTMOptions:
        active = active or {}
        return TTMOptions(
            out_interval=out_interval,
            infile=infile,
            properties_file=properties_file,
            source=source,
            active_x=active.get("x"),
            active_y=active.get("y"),
            active_z=active.get("z"),
        )

    def ttm(
        self,
        *,
        steps: int,
        Ce: float,
        rho_e: float,
        kappa_e: float,
        gamma_p: float,
        grid: Sequence[int],
        T_e_init: float,
        gamma_s: float = 0.0,
        v_0: float = 100.0,
        grouping_method: int = 0,
        group_id: int = 0,
        out_interval: int | None = None,
        infile: str | None = None,
        properties_file: str | None = None,
        source: float | None = None,
        active: dict[str, str | int] | None = None,
        **kwargs,
    ) -> "TwoTemperatureCalculation":
        """電子温度グリッドに結合した MD を仕込む。

        Parameters
        ----------
        Ce
            電子 1 個あたりの比熱 [eV/K]。
        rho_e
            電子数密度 [1/Å^3]。``Ce * rho_e`` が体積比熱 [eV/(K Å^3)]。
        kappa_e
            電子熱伝導率 [eV/(ps K Å)]。
        gamma_p, gamma_s
            電子-フォノン結合と電子阻止の摩擦係数 [amu/ps]。
            ``v_0`` [Å/ps] を超える速い原子にだけ ``gamma_s`` が効く。
        grid
            電子グリッドの分割数 ``(nx, ny, nz)``。
        source
            体積熱源 [eV/(ps Å^3)] (レーザー吸収など)。
        active
            ``{'z': '3:10'}`` のように活性グリッド領域を指定する。
        """
        self.ensure_grouping()
        spec = TTM(
            Ce=Ce,
            rho_e=rho_e,
            kappa_e=kappa_e,
            gamma_p=gamma_p,
            gamma_s=gamma_s,
            v_0=v_0,
            grid=grid,
            T_e_init=T_e_init,
            grouping_method=grouping_method,
            group_id=group_id,
            options=self._options(out_interval, infile, properties_file, source, active),
        )
        self.grid = tuple(int(n) for n in grid)
        return self.add_ensemble(spec, steps=steps, **kwargs)

    def heat_ttm(
        self,
        *,
        steps: int,
        temperature: float,
        delta_T: float,
        source_group: int,
        sink_group: int,
        Ce: float,
        rho_e: float,
        kappa_e: float,
        gamma_p: float,
        grid: Sequence[int],
        T_e_init: float,
        gamma_s: float = 0.0,
        v_0: float = 100.0,
        grouping_method: int = 0,
        group_id: int = 0,
        T_coup: float = 100.0,
        tau_T: float | None = None,
        out_interval: int | None = None,
        infile: str | None = None,
        properties_file: str | None = None,
        electron_source: float | None = None,
        active: dict[str, str | int] | None = None,
        **kwargs,
    ) -> "TwoTemperatureCalculation":
        """局所 source/sink Langevin 熱浴と電子グリッドを併用する。

        電子系の熱伝導を含めた熱輸送 (金属の NEMD) に使う。
        ``source_group`` / ``sink_group`` は grouping method 0 のラベル。
        """
        self.ensure_grouping()
        spec = HeatTTM(
            Ce=Ce,
            rho_e=rho_e,
            kappa_e=kappa_e,
            gamma_p=gamma_p,
            gamma_s=gamma_s,
            v_0=v_0,
            grid=grid,
            T_e_init=T_e_init,
            grouping_method=grouping_method,
            group_id=group_id,
            options=self._options(
                out_interval, infile, properties_file, electron_source, active
            ),
            temperature=temperature,
            delta_T=delta_T,
            source=source_group,
            sink=sink_group,
            T_coup=T_coup,
            tau_T=tau_T,
        )
        self.grid = tuple(int(n) for n in grid)
        return self.add_ensemble(spec, steps=steps, **kwargs)

    def electron_stopping(self, filename: str) -> "TwoTemperatureCalculation":
        """高エネルギー原子に電子阻止能を効かせる (照射損傷カスケード)。"""
        return self.add_commands(modifiers.electron_stop(filename))

    def limit_displacement(self, max_distance: float = 0.05) -> "TwoTemperatureCalculation":
        """1 ステップで原子が動ける距離 [Å] を制限する (高エネルギー衝突用)。"""
        self.builder.max_distance_per_step = float(max_distance)
        return self

    # ------------------------------------------------------------------ 結果
    def electron_temperature(self) -> TTMSnapshots:
        """``ttm_electron_temperature.out`` を読む。"""
        return read_ttm_electron_temperature(
            self.workdir / "ttm_electron_temperature.out"
        )

    def two_temperature_history(self, *, axis: str = "z") -> pd.DataFrame:
        """電子温度 (空間平均) と格子温度の時間変化を並べる。"""
        snapshots = self.electron_temperature()
        electron = [float(np.mean(a[a > 0])) if np.any(a > 0) else np.nan
                    for a in snapshots.temperatures]
        lattice = read_thermo(self.workdir / "thermo.out")["temperature"]
        # 出力間隔が違うのでステップ比で対応付ける
        if len(lattice) and snapshots.steps:
            positions = np.linspace(0, len(lattice) - 1, len(snapshots.steps)).astype(int)
            lattice_values = lattice.to_numpy()[positions]
        else:
            lattice_values = np.full(len(snapshots.steps), np.nan)
        return pd.DataFrame(
            {
                "step": snapshots.steps,
                "T_electron": electron,
                "T_lattice": lattice_values,
            }
        )

    def plot_two_temperature(
        self, filename: str = "ttm.png", *, axis: str = "z", dpi: int = 150
    ) -> Path:
        """電子温度と格子温度の時間変化、および電子温度の空間分布を描く。"""
        from ..plotting import label, setup_matplotlib

        setup_matplotlib()
        import matplotlib.pyplot as plt

        history = self.two_temperature_history(axis=axis)
        profile = self.electron_temperature().profile(axis)
        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(history["step"], history["T_electron"], lw=1.4, label="電子 $T_e$")
        axes[0].plot(history["step"], history["T_lattice"], lw=1.4, label="格子 $T_l$")
        axes[0].set_xlabel(label("ステップ"))
        axes[0].set_ylabel(label("温度 (K)"))
        axes[0].legend()
        columns = [c for c in profile.columns if c.startswith("step_")]
        for column in columns[:: max(1, len(columns) // 6)]:
            axes[1].plot(profile["index"], profile[column], lw=1.0, label=column)
        axes[1].set_xlabel(label(f"{axis} 方向のグリッド番号"))
        axes[1].set_ylabel(label("電子温度 (K)"))
        axes[1].legend(fontsize=7)
        figure.suptitle(label(f"二温度モデル — {self.name}"))
        figure.tight_layout()
        output = self.workdir / filename
        figure.savefig(output, dpi=dpi)
        plt.close(figure)
        return output
