"""最もシンプルな Metropolis モンテカルロ (MCMC)。

GPUMD の ``mc`` キーワードには原子変位の試行が無い (原子交換だけなら
:meth:`~gpumd_toolkit.workflows.MonteCarloCalculation.mcmc` で GPUMD 上で回せる)。
そこで本モジュールでは ASE の calculator (NEP の ``cpu`` / ``pynep`` / ``gpu``
バックエンド、あるいは任意の ASE calculator) でエネルギーを評価し、
Python 側で Metropolis 法を回す。

試行 (move)
-----------
``displace``  ランダムに選んだ 1 原子を最大 ``max_displacement`` Å だけ動かす
``swap``      元素の異なる 2 原子の位置を入れ替える (組成は保存)

受理判定は Metropolis 基準 ``min(1, exp(-ΔE / k_B T))`` のみ。
1 試行ごとに全エネルギーを 1 回評価するので、原子数は数百程度までが現実的。
``gpu`` バックエンドは 1 回ごとに gpumd をファイル経由で起動するため非常に遅い。
``cpu`` (calorine の NEP_CPU) を推奨する。

Examples
--------
>>> from gpumd_toolkit import MetropolisMC
>>> mc = MetropolisMC("CuAu.xyz", model="nep.txt", backend="cpu", workdir="runs/mc")
>>> mc.run(steps=20000, temperature=800, temperature_end=300,
...        moves={"displace": 0.5, "swap": 0.5}, max_displacement=0.1)
>>> mc.acceptance_ratio()
>>> mc.plot()
>>> mc.lowest_energy_structure()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator

from .ase_interface import create_nep_calculator
from .config import GPUMDEnvironment
from .structure import DEFAULT_MAX_ATOMS, StructureHandler

__all__ = ["MetropolisMC"]

MOVES = ("displace", "swap")


@dataclass
class MetropolisMC:
    """ASE calculator でエネルギーを評価する Metropolis MC。

    Parameters
    ----------
    atoms
        初期構造 (``Atoms`` か構造ファイルのパス)。
    model
        NEP ポテンシャルファイル。
    backend
        :func:`~gpumd_toolkit.ase_interface.create_nep_calculator` の ``backend``。
    workdir
        ログ・トラジェクトリ・図の出力先。
    seed
        乱数シード (再現性のため)。
    calculator
        NEP の代わりに使う ASE calculator (テストや他ポテンシャル用)。
    """

    atoms: Atoms | Path | str
    model: Path | str | None = None
    backend: str = "cpu"
    workdir: Path | str = "mc_run"
    seed: int | None = None
    max_atoms: int = DEFAULT_MAX_ATOMS
    environment: GPUMDEnvironment | None = None
    calculator_kwargs: dict = field(default_factory=dict)
    calculator: Calculator | None = None
    fast_read: bool = True

    #: 直近の run が書いたトラジェクトリ
    trajectory_path: Path | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.atoms, Atoms):
            self.atoms = StructureHandler.read(self.atoms, fast=self.fast_read)
        else:
            self.atoms = self.atoms.copy()
        StructureHandler.check_size(self.atoms, max_atoms=self.max_atoms)
        self.workdir = Path(self.workdir).expanduser().resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        if self.calculator is None:
            if self.model is None:
                raise ValueError("model か calculator のどちらかが必要です。")
            self.calculator = create_nep_calculator(
                self.model,
                backend=self.backend,
                atoms=self.atoms,
                environment=self.environment,
                **self.calculator_kwargs,
            )
        self.atoms.calc = self.calculator
        self.rng = np.random.default_rng(self.seed)
        self.log: list[dict[str, float]] = []
        self.n_trials = {move: 0 for move in MOVES}
        self.n_accepted = {move: 0 for move in MOVES}
        self._lowest: tuple[float, Atoms] | None = None
        self._step = 0

    # ------------------------------------------------------------------ 試行
    def _propose_displace(self, max_displacement: float):
        index = int(self.rng.integers(len(self.atoms)))
        old = self.atoms.positions[index].copy()
        new_positions = self.atoms.positions.copy()
        new_positions[index] = old + self.rng.uniform(-max_displacement, max_displacement, 3)
        self.atoms.positions = new_positions

        def undo() -> None:
            positions = self.atoms.positions.copy()
            positions[index] = old
            self.atoms.positions = positions

        return undo

    def _propose_swap(self):
        numbers = self.atoms.numbers
        i = int(self.rng.integers(len(self.atoms)))
        candidates = np.flatnonzero(numbers != numbers[i])
        j = int(self.rng.choice(candidates))

        def exchange() -> None:
            # 元素ではなく位置を入れ替える (どのバックエンドでも positions の変更として扱える)
            positions = self.atoms.positions.copy()
            positions[[i, j]] = positions[[j, i]]
            self.atoms.positions = positions

        exchange()
        return exchange

    @staticmethod
    def _normalize_moves(moves: Mapping[str, float] | str) -> dict[str, float]:
        if isinstance(moves, str):
            moves = {moves: 1.0}
        unknown = set(moves) - set(MOVES)
        if unknown:
            raise ValueError(f"未知の move {sorted(unknown)}。{MOVES} から選びます。")
        total = float(sum(moves.values()))
        if total <= 0 or any(w < 0 for w in moves.values()):
            raise ValueError("moves の重みは非負で、合計が正である必要があります。")
        return {move: float(w) / total for move, w in moves.items() if w > 0}

    # ------------------------------------------------------------------ 実行
    def run(
        self,
        *,
        steps: int,
        temperature: float,
        temperature_end: float | None = None,
        moves: Mapping[str, float] | str = "displace",
        max_displacement: float = 0.1,
        adapt_displacement: bool = False,
        target_acceptance: float = 0.4,
        log_interval: int = 100,
        trajectory_interval: int | None = None,
        trajectory: str = "mc.xyz",
    ) -> Atoms:
        """Metropolis MC を ``steps`` 試行だけ回す。

        Parameters
        ----------
        temperature, temperature_end
            MC の温度 [K]。``temperature_end`` を与えると線形に変える (焼きなまし)。
        moves
            試行の種類と選ぶ確率の比。例: ``{"displace": 0.7, "swap": 0.3}``。
        max_displacement
            ``displace`` の最大変位 [Å] (各成分を一様乱数で選ぶ)。
        adapt_displacement
            True なら ``log_interval`` ごとに ``max_displacement`` を調整し、
            ``displace`` の受理率を ``target_acceptance`` に近づける。
            調整すると詳細釣り合いが崩れるので、平衡化にだけ使うこと。
        trajectory_interval
            この試行数ごとに構造を拡張 XYZ に追記する (None なら書かない)。

        Returns
        -------
        最終構造 (``self.atoms``)。何回呼んでも続きから回る。
        """
        if steps <= 0:
            raise ValueError("steps は正の整数です。")
        weights = self._normalize_moves(moves)
        if "swap" in weights and len(set(self.atoms.numbers)) < 2:
            raise ValueError("swap には 2 種類以上の元素が必要です。")
        names = list(weights)
        probabilities = [weights[name] for name in names]
        T_end = temperature if temperature_end is None else temperature_end

        if trajectory_interval:
            self.trajectory_path = self.workdir / trajectory
            if self._step == 0:
                self.trajectory_path.unlink(missing_ok=True)

        energy = float(self.atoms.get_potential_energy())
        self._update_lowest(energy)
        window = {move: [0, 0] for move in MOVES}  # [試行, 受理] (直近 log_interval 分)

        for k in range(steps):
            T = temperature + (T_end - temperature) * (k / max(1, steps - 1))
            beta = 1.0 / (units.kB * max(T, 1e-12))
            move = names[int(self.rng.choice(len(names), p=probabilities))]
            undo = (
                self._propose_displace(max_displacement)
                if move == "displace"
                else self._propose_swap()
            )
            trial = float(self.atoms.get_potential_energy())
            delta = trial - energy
            accepted = delta <= 0.0 or self.rng.random() < math.exp(-beta * delta)
            if accepted:
                energy = trial
                self._update_lowest(energy)
            else:
                undo()
            self.n_trials[move] += 1
            self.n_accepted[move] += int(accepted)
            window[move][0] += 1
            window[move][1] += int(accepted)
            self._step += 1

            if self._step % log_interval == 0:
                self.log.append(self._entry(energy, T, max_displacement, window))
                if adapt_displacement and window["displace"][0]:
                    rate = window["displace"][1] / window["displace"][0]
                    # 受理率が高すぎれば大きく、低すぎれば小さく
                    max_displacement *= float(np.clip(rate / target_acceptance, 0.5, 1.5))
                window = {m: [0, 0] for m in MOVES}
            if trajectory_interval and self._step % trajectory_interval == 0:
                self._write_frame(energy)

        self.max_displacement = max_displacement
        # 受理されなかった試行で calculator の結果が残らないよう、最終構造で取り直す
        self.atoms.get_potential_energy()
        return self.atoms

    def _entry(self, energy, T, max_displacement, window) -> dict[str, float]:
        n = len(self.atoms)
        entry = {
            "step": self._step,
            "temperature": T,
            "energy": energy,
            "energy_per_atom": energy / n,
            "max_displacement": max_displacement,
        }
        for move, (tried, accepted) in window.items():
            entry[f"acceptance_{move}"] = accepted / tried if tried else float("nan")
        return entry

    def _update_lowest(self, energy: float) -> None:
        if self._lowest is None or energy < self._lowest[0]:
            snapshot = self.atoms.copy()
            snapshot.calc = None
            snapshot.info["energy"] = energy
            self._lowest = (energy, snapshot)

    def _write_frame(self, energy: float) -> None:
        from ase.io import write as ase_write

        frame = self.atoms.copy()
        frame.calc = None
        frame.info.update({"step": self._step, "energy": energy})
        ase_write(str(self.trajectory_path), frame, format="extxyz", append=True)

    # ------------------------------------------------------------------ 結果
    def acceptance_ratio(self, move: str | None = None) -> float:
        """これまでの全試行の受理率 (``move`` を指定するとその試行だけ)。"""
        moves = [move] if move else list(MOVES)
        tried = sum(self.n_trials[m] for m in moves)
        return sum(self.n_accepted[m] for m in moves) / tried if tried else float("nan")

    def lowest_energy_structure(self) -> Atoms:
        """サンプリング中に訪れた最もエネルギーの低い構造。"""
        if self._lowest is None:
            raise RuntimeError("先に run() を実行してください。")
        return self._lowest[1].copy()

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.log)

    def save_log(self, filename: str = "mc.csv") -> Path:
        path = self.workdir / filename
        self.to_dataframe().to_csv(path, index=False)
        return path

    def plot(self, filename: str = "mc.png", *, dpi: int = 150) -> Path:
        """エネルギー・受理率・温度の推移を 1 枚の図にする。"""
        from .plotting import label, setup_matplotlib

        setup_matplotlib()
        import matplotlib.pyplot as plt

        frame = self.to_dataframe()
        if frame.empty:
            raise ValueError("ログが空です。run() を先に実行してください。")
        figure, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
        axes[0].plot(frame["step"], frame["energy_per_atom"], lw=1.0)
        axes[0].set_ylabel("E (eV/atom)")
        for move in MOVES:
            column = f"acceptance_{move}"
            if frame[column].notna().any():
                axes[1].plot(frame["step"], frame[column], lw=1.0, label=move)
        axes[1].set_ylabel(label("MC 受理率"))
        axes[1].set_ylim(0, 1)
        axes[1].legend(fontsize=8)
        axes[2].plot(frame["step"], frame["temperature"], lw=1.0, color="tab:red")
        axes[2].set_ylabel(label("温度") + " (K)")
        axes[2].set_xlabel("MC step")
        for ax in axes:
            ax.grid(alpha=0.3)
        figure.suptitle(f"Metropolis MC: {self.workdir.name}")
        figure.tight_layout()
        output = self.workdir / filename
        figure.savefig(output, dpi=dpi)
        plt.close(figure)
        return output
