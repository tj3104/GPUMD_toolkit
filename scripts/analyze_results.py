#!/usr/bin/env python
"""GPUMD の計算結果を解析する CLI。

使用例
------
# 1 つの計算ディレクトリを解析 (図・CSV を <dir>/analysis に出力)
python scripts/analyze_results.py runs/si_300K

# 平衡化の切り捨て割合を変え、トラジェクトリも変換
python scripts/analyze_results.py runs/si_300K --drop 0.5 \
    --convert xdatcar traj --stride 5

# 複数ディレクトリの温度を 1 枚に重ねる
python scripts/analyze_results.py runs/T300K runs/T600K runs/T900K \
    --compare temperature --compare-output runs/compare_T.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from gpumd_toolkit import MDAnalyzer, TrajectoryConverter
from gpumd_toolkit.analysis import compare_runs
from gpumd_toolkit.trajectory import SUPPORTED_OUTPUT_FORMATS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPUMD 計算結果の解析",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("directories", nargs="+", help="計算ディレクトリ")
    parser.add_argument("--drop", type=float, default=0.3,
                        help="各ステージの先頭を平衡化として除外する割合")
    parser.add_argument("--output-dir", default=None,
                        help="図・CSV の出力先 (既定: <dir>/analysis)")
    parser.add_argument("--no-plot", action="store_true", help="図を作らない")
    parser.add_argument("--convert", nargs="*", default=[],
                        choices=sorted(SUPPORTED_OUTPUT_FORMATS),
                        help="トラジェクトリの変換形式")
    parser.add_argument("--trajectory", default="dump.xyz", help="トラジェクトリのファイル名")
    parser.add_argument("--stride", type=int, default=1, help="変換時のフレーム間引き")
    parser.add_argument("--msd", action="store_true",
                        help="トラジェクトリから MSD / 拡散係数を計算する")
    parser.add_argument("--rdf", action="store_true", help="トラジェクトリから g(r) を計算する")
    parser.add_argument("--compare", default=None,
                        help="複数ディレクトリで重ね描きする量 (例: temperature)")
    parser.add_argument("--compare-output", default="compare.png", help="重ね描きの出力先")
    parser.add_argument("--json", action="store_true", help="サマリを JSON で標準出力へ")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summaries: dict[str, list[dict]] = {}

    for directory in args.directories:
        directory = Path(directory)
        print(f"\n=== {directory} ===")
        analyzer = MDAnalyzer(directory, output_dir=args.output_dir)
        summary = analyzer.thermo.summary(drop_fraction=args.drop)
        print(summary.to_string(index=False))
        print(f"  全エネルギードリフト: {analyzer.thermo.energy_drift():+.4f} meV/atom/ps")
        summaries[str(directory)] = summary.to_dict(orient="records")

        for source in ("msd", "sdc"):
            if (directory / f"{source}.out").is_file():
                values = analyzer.diffusion_coefficient(source=source)
                print(
                    f"  拡散係数 ({source}.out): "
                    f"{values['D_mean_1e-9_m2_per_s']:.4f} x 1e-9 m²/s "
                    f"(x={values['D_x']:.4f}, y={values['D_y']:.4f}, z={values['D_z']:.4f} Å²/ps)"
                )
                summaries[str(directory)].append({f"diffusion_{source}": values})

        if not args.no_plot:
            for path in analyzer.plot_all(drop_fraction=args.drop):
                print(f"  出力: {path}")

        traj_path = directory / args.trajectory
        if traj_path.is_file() and (args.convert or args.msd or args.rdf):
            converter = TrajectoryConverter(traj_path)
            for fmt in args.convert:
                _, suffix = SUPPORTED_OUTPUT_FORMATS[fmt]
                name = suffix if suffix.isupper() else f"trajectory{suffix}"
                print(f"  変換: {converter.convert(directory / name, fmt=fmt, stride=args.stride)}")
            if args.msd:
                dt = _frame_interval_ps(directory, args.trajectory)
                msd = converter.msd(time_step_ps=dt)
                if "D_1e-9_m2_per_s" in msd:
                    print(f"  MSD からの拡散係数: {msd['D_1e-9_m2_per_s']:.4f} x 1e-9 m²/s")
                else:
                    print(f"  MSD 最終値: {msd['msd_total'][-1]:.4f} Å²")
            if args.rdf:
                rdf = converter.rdf(stride=max(1, args.stride))
                peak = rdf["r"][int(rdf["g"].argmax())]
                print(f"  g(r) の第一ピーク: r = {peak:.3f} Å")

    if args.compare and len(args.directories) > 1:
        compare_runs(args.directories, quantity=args.compare, filename=args.compare_output)
        print(f"\n重ね描き: {args.compare_output}")

    if args.json:
        print(json.dumps(summaries, indent=2, ensure_ascii=False, default=float))
    return 0


def _frame_interval_ps(directory: Path, trajectory: str) -> float | None:
    """run.in から dump_xyz の間隔と time_step を読み、フレーム間隔 [ps] を返す。"""
    run_in = directory / "run.in"
    if not run_in.is_file():
        return None
    time_step = 1.0
    interval = None
    for line in run_in.read_text().splitlines():
        tokens = line.split("#")[0].split()
        if tokens[:1] == ["time_step"]:
            time_step = float(tokens[1])
        elif tokens[:1] == ["dump_xyz"] and trajectory in tokens:
            interval = int(tokens[1])
    return interval * time_step * 1e-3 if interval else None


if __name__ == "__main__":
    raise SystemExit(main())
