"""joblib による複数 GPUMD 計算の並列実行。

GPU は 1 枚でも「同時に複数ジョブを走らせる」ことは可能だが、
大きな系ではメモリ・演算資源の奪い合いになる。既定では GPU 枚数に応じて
``CUDA_VISIBLE_DEVICES`` を割り当て、``n_jobs`` はユーザが指定する。

``GPUMDCalculation`` はそのままではワーカープロセスへ送れない場合がある
(ASE Atoms などを抱えるため pickle は可能だが重い)ので、
「計算を組み立てる関数 (factory)」を渡す方式も用意している。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from joblib import Parallel, delayed

from .config import available_gpus
from .md import GPUMDCalculation, GPUMDResult

__all__ = ["ParallelRunner", "JobSpec", "run_jobs"]


@dataclass
class JobSpec:
    """並列実行 1 件分の指定。

    Parameters
    ----------
    factory
        引数なしで :class:`GPUMDCalculation` を返す callable。
        ワーカープロセス側で呼ばれるので、モジュールレベル関数か
        ``functools.partial`` を使うこと (ラムダは pickle できない)。
    name
        ログ表示用。
    """

    factory: Callable[[], GPUMDCalculation]
    name: str = ""


def _execute(
    factory: Callable[[], GPUMDCalculation],
    gpu_id: int | None,
    run_kwargs: dict,
    analyze: bool,
) -> GPUMDResult:
    calculation = factory()
    if gpu_id is not None:
        calculation.environment = calculation.environment.replace(gpu_id=gpu_id)
    result = calculation.run(**run_kwargs)
    if analyze and result.succeeded:
        try:
            result.plot()
        except Exception:
            pass
    return result


class ParallelRunner:
    """複数の GPUMD 計算を joblib で並列実行する。

    Examples
    --------
    >>> from functools import partial
    >>> def build(temperature, workdir):
    ...     calc = GPUMDCalculation("POSCAR", "nep.txt", workdir)
    ...     calc.nvt(temperature=temperature, steps=10000)
    ...     return calc
    >>> jobs = [JobSpec(partial(build, T, f"runs/T{T}"), name=f"T{T}")
    ...         for T in (300, 600, 900)]
    >>> results = ParallelRunner(n_jobs=2).run(jobs)
    """

    def __init__(
        self,
        *,
        n_jobs: int = 1,
        backend: str = "loky",
        gpu_ids: Sequence[int] | None = None,
        verbose: int = 5,
        analyze: bool = True,
    ) -> None:
        self.n_jobs = n_jobs
        self.backend = backend
        self.verbose = verbose
        self.analyze = analyze
        if gpu_ids is None:
            detected = available_gpus()
            self.gpu_ids: list[int] | None = detected or None
        else:
            self.gpu_ids = list(gpu_ids)

    def _gpu_for(self, index: int) -> int | None:
        if not self.gpu_ids:
            return None
        return self.gpu_ids[index % len(self.gpu_ids)]

    def run(
        self,
        jobs: Sequence[JobSpec] | Sequence[Callable[[], GPUMDCalculation]],
        **run_kwargs,
    ) -> list[GPUMDResult]:
        """ジョブ列を並列実行し、結果を投入順で返す。"""
        specs = [
            job if isinstance(job, JobSpec) else JobSpec(job, name=f"job{i}")
            for i, job in enumerate(jobs)
        ]
        run_kwargs.setdefault("check", False)
        tasks = (
            delayed(_execute)(spec.factory, self._gpu_for(i), dict(run_kwargs), self.analyze)
            for i, spec in enumerate(specs)
        )
        return Parallel(n_jobs=self.n_jobs, backend=self.backend, verbose=self.verbose)(tasks)

    def run_temperature_scan(
        self,
        build: Callable[..., GPUMDCalculation],
        temperatures: Iterable[float],
        *,
        workdir_root: Path | str,
        **run_kwargs,
    ) -> list[GPUMDResult]:
        """温度スキャン用のショートカット。

        ``build(temperature=T, workdir=...)`` の形で呼ばれる関数を渡す。
        """
        from functools import partial

        root = Path(workdir_root)
        jobs = [
            JobSpec(
                partial(build, temperature=float(T), workdir=root / f"T{float(T):g}K"),
                name=f"T={T}K",
            )
            for T in temperatures
        ]
        return self.run(jobs, **run_kwargs)


def run_jobs(
    jobs: Sequence[JobSpec], *, n_jobs: int = 1, **kwargs
) -> list[GPUMDResult]:
    """:class:`ParallelRunner` の薄いラッパ。"""
    return ParallelRunner(n_jobs=n_jobs, **kwargs).run(jobs)
