#!/usr/bin/env python
"""GPUMD で MD を実行する汎用 CLI。

使用例
------
# 300 K で NVT 20 ps
python scripts/run_md.py POSCAR -p nep.txt -o runs/si_300K \
    --ensemble nvt --temperature 300 --steps 20000

# 300 K -> 1200 K の昇温 + 1200 K 保持 + 降温 (アニール)
python scripts/run_md.py structure.cif -p nep.txt -o runs/anneal \
    --profile anneal --t-low 300 --t-high 1200 \
    --heat-steps 20000 --hold-steps 20000 --cool-steps 40000

# 任意の温度プロファイル ("時刻ps:温度K" をカンマ区切り)
python scripts/run_md.py POSCAR -p nep.txt -o runs/custom \
    --profile points --points 0:300,10:1000,20:1000,40:300

# NPT (0 GPa) で 10 ps
python scripts/run_md.py POSCAR -p nep.txt -o runs/npt \
    --ensemble npt --temperature 500 --pressure 0 --steps 10000 --orthorhombic

# 熱浴・圧浴の時定数を fs で指定 (tau/dt に自動換算)
python scripts/run_md.py POSCAR -p nep.txt -o runs/npt \
    --ensemble npt --temperature 500 --tau-t 100 --tau-p 1000

# c 軸だけ動かす NPT (他の軸は固定)
python scripts/run_md.py POSCAR -p nep.txt -o runs/npt_z \
    --ensemble npt --temperature 500 --free-axes z

# 三斜晶セルを 6 成分指定でフルに緩和する
python scripts/run_md.py structure.cif -p nep.txt -o runs/npt_tri \
    --ensemble npt --temperature 500 --cell-mode tri

# npt_mttk で x 方向にだけ 5 GPa かける
python scripts/run_md.py POSCAR -p nep.txt -o runs/uniaxial \
    --ensemble npt --barostat npt_mttk --mttk-direction x --pressure 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from gpumd_toolkit import GPUMDCalculation, GPUMDEnvironment, TemperatureProfile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPUMD で MD を実行する", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("structure", help="入力構造 (POSCAR / CIF / xyz / traj など)")
    parser.add_argument("-p", "--potential", required=True, help="NEP ポテンシャル (nep.txt)")
    parser.add_argument("-o", "--workdir", required=True, help="計算ディレクトリ")

    group = parser.add_argument_group("セル")
    group.add_argument("--repeat", type=int, nargs=3, default=None, help="スーパーセルの繰り返し数")
    group.add_argument("--min-cell-length", type=float, default=None,
                       help="この長さ [Å] 以上になるよう自動でスーパーセル化")
    group.add_argument("--max-atoms", type=int, default=10000, help="原子数の上限")
    group.add_argument("--orthorhombic", action="store_true", help="直方晶セルへ変換 (NPT 向け)")
    group.add_argument("--reduce-cell", default=None, choices=["niggli", "minkowski"],
                       help="スーパーセル化の前にセルを簡約する (斜めのセル対策)")
    group.add_argument("--standard-cell", action="store_true",
                       help="セルを下三角形 (LAMMPS 標準形) に回転する")
    group.add_argument("--no-fast-read", action="store_true",
                       help="構造の高速読み込み (専用パーサ + xyz キャッシュ) を使わない")

    group = parser.add_argument_group("MD 条件")
    group.add_argument("--ensemble", default="nvt", choices=["nve", "nvt", "npt", "nph"])
    group.add_argument("--thermostat", default="nvt_nhc",
                       choices=["nvt_ber", "nvt_nhc", "nvt_bdp", "nvt_lan", "nvt_bao", "nvt_mttk"])
    group.add_argument("--barostat", default="npt_scr", choices=["npt_ber", "npt_scr", "npt_mttk"])
    group.add_argument("--temperature", type=float, default=300.0, help="目標温度 [K]")
    group.add_argument("--temperature-end", type=float, default=None,
                       help="終温 [K] (指定すると線形に昇温/降温)")
    group.add_argument("--pressure", type=float, nargs="+", default=[0.0],
                       help="NPT の目標圧力 [GPa]。1 / 3 (xx yy zz) / 6 成分 (xx yy zz yz xz xy)")
    group.add_argument("--pressure-end", type=float, nargs="+", default=None,
                       help="終圧 [GPa] (npt_mttk のみ。圧力を線形に変化させる)")
    group.add_argument("--elastic-modulus", type=float, nargs="+", default=[100.0],
                       help="NPT (ber/scr) に必要な弾性率の概算 [GPa]")
    group.add_argument("--steps", type=int, default=10000, help="ステップ数")
    group.add_argument("--time-step", type=float, default=1.0, help="時間刻み [fs]")
    group.add_argument("--minimize", action="store_true", help="MD 前に構造最適化")

    group = parser.add_argument_group("熱浴・圧浴の時定数")
    group.add_argument("--t-coup", type=float, default=100.0,
                       help="温度カップリング定数 tau_T/dt (無次元)")
    group.add_argument("--p-coup", type=float, default=1000.0,
                       help="圧力カップリング定数 tau_p/dt (無次元)")
    group.add_argument("--tau-t", type=float, default=None,
                       help="温度の時定数 [fs] (指定すると --t-coup より優先)")
    group.add_argument("--tau-p", type=float, default=None,
                       help="圧力の時定数 [fs] (指定すると --p-coup より優先)")

    group = parser.add_argument_group("NPT のセル自由度・変調軸")
    group.add_argument("--cell-mode", default=None, choices=["iso", "ortho", "tri"],
                       help="npt_ber / npt_scr のセル自由度 (既定: 構造と圧力から自動)")
    group.add_argument("--free-axes", nargs="+", default=None,
                       help="変調を許す軸 (x y z yz xz xy)。他は固定される")
    group.add_argument("--fixed-axes", nargs="+", default=[],
                       help="固定する軸 (x y z yz xz xy)")
    group.add_argument("--mttk-direction", nargs="+", default=["iso"],
                       help="npt_mttk / nph_mttk のセル変形方向 (iso / aniso / tri / x y z ...)")

    group = parser.add_argument_group("温度プロファイル")
    group.add_argument("--profile", default="none", choices=["none", "anneal", "points"])
    group.add_argument("--t-low", type=float, default=300.0)
    group.add_argument("--t-high", type=float, default=1000.0)
    group.add_argument("--equil-steps", type=int, default=0)
    group.add_argument("--heat-steps", type=int, default=10000)
    group.add_argument("--hold-steps", type=int, default=10000)
    group.add_argument("--cool-steps", type=int, default=10000)
    group.add_argument("--points", default="", help='"0:300,10:1000,20:300" 形式 (時刻ps:温度K)')

    group = parser.add_argument_group("出力")
    group.add_argument("--dump-thermo", type=int, default=100, help="thermo.out の出力間隔")
    group.add_argument("--dump-traj", type=int, default=1000, help="トラジェクトリの出力間隔")
    group.add_argument("--traj-properties", nargs="*",
                       default=["velocity", "force", "potential"],
                       help="dump_xyz に含める per-atom 量")
    group.add_argument("--convert", nargs="*", default=[],
                       choices=["xdatcar", "traj", "extxyz", "lammps-dump", "pdb"],
                       help="実行後にトラジェクトリを変換する形式")
    group.add_argument("--stride", type=int, default=1, help="変換時のフレーム間引き")
    group.add_argument("--no-plot", action="store_true", help="解析プロットを作らない")

    group = parser.add_argument_group("実行")
    group.add_argument("--gpu", type=int, default=None, help="使用 GPU (CUDA_VISIBLE_DEVICES)")
    group.add_argument("--venv", default=None, help="使用する Python 仮想環境")
    group.add_argument("--dry-run", action="store_true", help="入力ファイルの生成のみ")
    group.add_argument("--overwrite", action="store_true", help="計算ディレクトリを作り直す")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    env_kwargs = {}
    if args.venv:
        env_kwargs["venv"] = Path(args.venv)
    if args.gpu is not None:
        env_kwargs["gpu_id"] = args.gpu
    environment = GPUMDEnvironment(**env_kwargs)

    calculation = GPUMDCalculation(
        args.structure,
        args.potential,
        args.workdir,
        time_step=args.time_step,
        repeat=args.repeat,
        min_cell_length=args.min_cell_length,
        orthorhombic=args.orthorhombic,
        reduce_cell=args.reduce_cell,
        standard_cell=args.standard_cell,
        fast_read=not args.no_fast_read,
        max_atoms=args.max_atoms,
        environment=environment,
        overwrite=args.overwrite,
    )
    calculation.set_dump(
        thermo=args.dump_thermo,
        traj=args.dump_traj,
        traj_properties=tuple(args.traj_properties),
    )
    if args.minimize:
        calculation.minimize()

    pressure = _one_or_many(args.pressure)
    pressure_end = _one_or_many(args.pressure_end) if args.pressure_end else None
    elastic_modulus = _one_or_many(args.elastic_modulus)
    direction = _one_or_many(args.mttk_direction)

    if args.profile == "anneal":
        profile = TemperatureProfile.anneal(
            args.t_low,
            args.t_high,
            heat_steps=args.heat_steps,
            hold_steps=args.hold_steps,
            cool_steps=args.cool_steps,
            equilibrate_steps=args.equil_steps,
        )
        calculation.apply_profile(profile, ensemble=args.thermostat)
        print(profile.summary(args.time_step))
    elif args.profile == "points":
        points = [
            tuple(float(x) for x in item.split(":")) for item in args.points.split(",") if item
        ]
        profile = TemperatureProfile.from_points(points, time_step_fs=args.time_step)
        calculation.apply_profile(profile, ensemble=args.thermostat)
        print(profile.summary(args.time_step))
    elif args.ensemble == "nve":
        calculation.nve(steps=args.steps)
    elif args.ensemble == "nvt":
        calculation.nvt(
            temperature=args.temperature,
            temperature_end=args.temperature_end,
            steps=args.steps,
            thermostat=args.thermostat,
            T_coup=args.t_coup,
            tau_T=args.tau_t,
        )
    elif args.ensemble == "nph":
        calculation.nph(
            steps=args.steps,
            pressure=pressure,
            pressure_end=pressure_end,
            direction=direction,
            p_period=args.p_coup,
            tau_p=args.tau_p,
        )
    else:
        calculation.npt(
            temperature=args.temperature,
            temperature_end=args.temperature_end,
            steps=args.steps,
            pressure=pressure,
            pressure_end=pressure_end,
            elastic_modulus=elastic_modulus,
            barostat=args.barostat,
            T_coup=args.t_coup,
            p_coup=args.p_coup,
            tau_T=args.tau_t,
            tau_p=args.tau_p,
            cell_mode=args.cell_mode,
            fixed_axes=tuple(args.fixed_axes),
            free_axes=tuple(args.free_axes) if args.free_axes else None,
            **({"mttk_direction": direction} if args.barostat == "npt_mttk" else {}),
        )

    print(calculation.describe())
    result = calculation.run(dry_run=args.dry_run)
    if args.dry_run:
        print(f"\n入力ファイルを生成しました: {calculation.workdir}")
        print(calculation.preview())
        return 0

    print(f"\n完了: {result}  ({result.elapsed:.1f} s)")
    print(result.summary().to_string(index=False))

    if not args.no_plot:
        for path in result.plot():
            print(f"  出力: {path}")
    for fmt in args.convert:
        print(f"  変換: {result.trajectory().convert(result.workdir / _name(fmt), fmt=fmt, stride=args.stride)}")
    return 0


def _one_or_many(values):
    """``nargs='+'`` の結果を、1 個ならスカラーに、複数ならタプルにする。"""
    if values is None:
        return None
    return values[0] if len(values) == 1 else tuple(values)


def _name(fmt: str) -> str:
    from gpumd_toolkit.trajectory import SUPPORTED_OUTPUT_FORMATS

    _, suffix = SUPPORTED_OUTPUT_FORMATS[fmt]
    return suffix if suffix.isupper() else f"trajectory{suffix}"


if __name__ == "__main__":
    raise SystemExit(main())
