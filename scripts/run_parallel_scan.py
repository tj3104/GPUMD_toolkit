#!/usr/bin/env python
"""joblib で温度スキャンを並列実行する CLI。

使用例
------
python scripts/run_parallel_scan.py POSCAR -p nep.txt -o runs/scan \
    --temperatures 300 600 900 1200 --steps 10000 --n-jobs 2

GPU が複数ある場合は ``--gpu-ids 0 1`` のように指定すると
ジョブへラウンドロビンで割り当てる。
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import _bootstrap  # noqa: F401

from gpumd_toolkit import GPUMDCalculation, GPUMDEnvironment, JobSpec, ParallelRunner
from gpumd_toolkit.analysis import compare_runs


def build_calculation(
    *,
    temperature: float,
    workdir: Path,
    structure: str,
    potential: str,
    steps: int,
    time_step: float,
    thermostat: str,
    dump_thermo: int,
    dump_traj: int,
    repeat,
    min_cell_length,
    max_atoms: int,
    venv: str | None,
) -> GPUMDCalculation:
    """ワーカープロセス側で呼ばれる構築関数 (トップレベル関数にすること)。"""
    environment = GPUMDEnvironment(venv=Path(venv)) if venv else GPUMDEnvironment()
    calculation = GPUMDCalculation(
        structure,
        potential,
        workdir,
        name=f"T{temperature:g}K",
        time_step=time_step,
        repeat=repeat,
        min_cell_length=min_cell_length,
        max_atoms=max_atoms,
        environment=environment,
        overwrite=True,
    )
    calculation.set_dump(thermo=dump_thermo, traj=dump_traj)
    calculation.nvt(temperature=temperature, steps=steps, thermostat=thermostat)
    return calculation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="温度スキャンの並列実行",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("structure")
    parser.add_argument("-p", "--potential", required=True)
    parser.add_argument("-o", "--workdir-root", required=True)
    parser.add_argument("--temperatures", type=float, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--time-step", type=float, default=1.0)
    parser.add_argument("--thermostat", default="nvt_nhc")
    parser.add_argument("--repeat", type=int, nargs=3, default=None)
    parser.add_argument("--min-cell-length", type=float, default=None)
    parser.add_argument("--max-atoms", type=int, default=10000)
    parser.add_argument("--dump-thermo", type=int, default=100)
    parser.add_argument("--dump-traj", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=1, help="同時実行数")
    parser.add_argument("--gpu-ids", type=int, nargs="*", default=None)
    parser.add_argument("--venv", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.workdir_root).resolve()

    common = dict(
        structure=str(Path(args.structure).resolve()),
        potential=str(Path(args.potential).resolve()),
        steps=args.steps,
        time_step=args.time_step,
        thermostat=args.thermostat,
        dump_thermo=args.dump_thermo,
        dump_traj=args.dump_traj,
        repeat=tuple(args.repeat) if args.repeat else None,
        min_cell_length=args.min_cell_length,
        max_atoms=args.max_atoms,
        venv=args.venv,
    )
    jobs = [
        JobSpec(
            partial(build_calculation, temperature=T, workdir=root / f"T{T:g}K", **common),
            name=f"T={T:g}K",
        )
        for T in args.temperatures
    ]

    runner = ParallelRunner(n_jobs=args.n_jobs, gpu_ids=args.gpu_ids)
    results = runner.run(jobs)

    print("\n=== 結果 ===")
    directories = []
    for result in results:
        status = "OK" if result.succeeded else "FAILED"
        print(f"  {result.name:<12s} {status:<7s} {result.elapsed:8.1f} s  {result.workdir}")
        if result.succeeded:
            directories.append(result.workdir)
            print(result.summary().to_string(index=False))

    if len(directories) > 1:
        output = root / "compare_temperature.png"
        compare_runs(directories, quantity="temperature", filename=output)
        compare_runs(
            directories,
            quantity="potential_energy_per_atom",
            filename=root / "compare_energy.png",
        )
        print(f"\n比較図: {output}")
    return 0 if all(r.succeeded for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
