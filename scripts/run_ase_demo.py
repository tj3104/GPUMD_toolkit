#!/usr/bin/env python
"""ASE ベースで NEP を使うデモ CLI。

使用例
------
# CPU (NEP_CPU) で 1 点計算 + 構造最適化 + 簡易 MD
python scripts/run_ase_demo.py POSCAR -p nep.txt -o runs/ase_demo --backend cpu

# GPU (gpumd 実行ファイル経由)
python scripts/run_ase_demo.py POSCAR -p nep.txt -o runs/ase_demo --backend gpu

# pyNEP (NEP_CPU) を使う
python scripts/run_ase_demo.py POSCAR -p nep.txt -o runs/ase_demo --backend pynep
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

from gpumd_toolkit import ASEMDRunner, available_backends


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ASE ベースの NEP 計算デモ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("structure", help="入力構造 (POSCAR / CIF など)")
    parser.add_argument("-p", "--potential", required=True, help="NEP ポテンシャル (nep.txt)")
    parser.add_argument("-o", "--workdir", default="runs/ase_demo")
    parser.add_argument("--backend", default="auto", choices=["auto", "cpu", "pynep", "gpu"])
    parser.add_argument("--repeat", type=int, nargs=3, default=None)
    parser.add_argument("--relax", action="store_true", help="構造最適化を行う")
    parser.add_argument("--relax-cell", action="store_true", help="セルも最適化する")
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--md", action="store_true", help="簡易 MD を回す")
    parser.add_argument("--ensemble", default="nvt",
                        choices=["nve", "nvt", "nvt_nose_hoover", "npt"])
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--temperature-end", type=float, default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--time-step", type=float, default=1.0)
    parser.add_argument("--convert", default=None,
                        help="MD 後にトラジェクトリを変換する形式 (xdatcar / extxyz など)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print("利用可能なバックエンド:", available_backends())

    from gpumd_toolkit.structure import StructureHandler

    atoms = StructureHandler.read(args.structure)
    if args.repeat:
        atoms = atoms.repeat(tuple(args.repeat))

    runner = ASEMDRunner(
        atoms, model=args.potential, backend=args.backend, workdir=args.workdir
    )
    print(f"構造: {StructureHandler.info(runner.atoms)}")

    single = runner.single_point()
    print("\n--- 1 点計算 ---")
    print(f"  全エネルギー      : {single['energy']:.6f} eV")
    print(f"  1 原子あたり      : {single['energy_per_atom']:.6f} eV/atom")
    print(f"  最大力            : {single['fmax']:.6f} eV/Å")
    if "pressure_GPa" in single:
        print(f"  圧力              : {single['pressure_GPa']:.4f} GPa")
    if "energies" in single:
        energies = np.asarray(single["energies"])
        print(
            f"  per-atom エネルギー: min {energies.min():.6f} / "
            f"max {energies.max():.6f} eV (N={energies.size})"
        )

    if args.relax:
        print("\n--- 構造最適化 ---")
        runner.relax(fmax=args.fmax, relax_cell=args.relax_cell)
        after = runner.single_point()
        print(f"  最適化後エネルギー: {after['energy_per_atom']:.6f} eV/atom")
        print(f"  最適化後セル      : {runner.atoms.cell.cellpar()}")

    if args.md:
        print("\n--- 簡易 MD ---")
        runner.run_md(
            ensemble=args.ensemble,
            temperature=args.temperature,
            temperature_end=args.temperature_end,
            steps=args.steps,
            time_step=args.time_step,
        )
        frame = runner.to_dataframe()
        print(frame.tail(5).to_string(index=False))
        print(f"  図  : {runner.plot()}")
        print(f"  ログ: {runner.save_log()}")
        if args.convert:
            from gpumd_toolkit.trajectory import SUPPORTED_OUTPUT_FORMATS

            _, suffix = SUPPORTED_OUTPUT_FORMATS[args.convert]
            name = suffix if suffix.isupper() else f"trajectory{suffix}"
            print(f"  変換: {runner.write_trajectory(Path(args.workdir) / name, fmt=args.convert)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
