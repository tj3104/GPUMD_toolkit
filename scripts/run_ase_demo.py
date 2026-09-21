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

# 熱浴の時定数を変えて NVT (Berendsen)
python scripts/run_ase_demo.py POSCAR -p nep.txt -o runs/ase_nvt \
    --md --ensemble nvt_berendsen --taut 200 --steps 2000

# Parrinello-Rahman NPT で c 軸だけ動かす
python scripts/run_ase_demo.py POSCAR -p nep.txt -o runs/ase_npt \
    --md --ensemble npt_parrinello_rahman --pressure 0 \
    --ttime 50 --ptime 2000 --bulk-modulus 140 --npt-axes z
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
                        choices=["nve", "nvt", "nvt_berendsen", "nvt_bussi",
                                 "nvt_nose_hoover", "npt", "npt_berendsen",
                                 "npt_inhomogeneous", "npt_parrinello_rahman"])
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--temperature-end", type=float, default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--time-step", type=float, default=1.0)

    group = parser.add_argument_group("熱浴・圧浴 (すべて fs)")
    group.add_argument("--taut", type=float, default=100.0,
                       help="Berendsen 系の温度の時定数 [fs]")
    group.add_argument("--taup", type=float, default=1000.0,
                       help="Berendsen 系の圧力の時定数 [fs]")
    group.add_argument("--ttime", type=float, default=25.0,
                       help="Nose-Hoover / Parrinello-Rahman の特性時間 [fs]")
    group.add_argument("--ptime", type=float, default=1000.0,
                       help="Parrinello-Rahman の特性時間 [fs] (pfactor の元)")
    group.add_argument("--pfactor", type=float, default=None,
                       help="ASE の pfactor を直接指定する (--ptime より優先)")
    group.add_argument("--friction", type=float, default=0.01,
                       help="Langevin の摩擦係数 [1/fs]")
    group.add_argument("--pressure", type=float, nargs="+", default=[0.0],
                       help="目標圧力 [GPa] (1 / 3 / 6 成分)")
    group.add_argument("--bulk-modulus", type=float, default=100.0,
                       help="体積弾性率の概算 [GPa] (圧縮率・pfactor に使う)")
    group.add_argument("--npt-axes", nargs="+", default=None,
                       help="セル変調を許す軸 (x y z / xy yz xz)。他は固定される")
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
        pressure = args.pressure[0] if len(args.pressure) == 1 else tuple(args.pressure)
        runner.run_md(
            ensemble=args.ensemble,
            temperature=args.temperature,
            temperature_end=args.temperature_end,
            steps=args.steps,
            time_step=args.time_step,
            friction=args.friction,
            pressure_GPa=pressure,
            taut=args.taut,
            taup=args.taup,
            ttime=args.ttime,
            ptime=args.ptime,
            pfactor=args.pfactor,
            bulk_modulus_GPa=args.bulk_modulus,
            npt_axes=tuple(args.npt_axes) if args.npt_axes else None,
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
