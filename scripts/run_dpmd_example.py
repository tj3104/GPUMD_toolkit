#!/usr/bin/env python
"""DPMD example (液体水の自己拡散) を実行して予実をとる CLI。

使用例
------
# DP (deepmd-kit の .pb) が使える環境
python scripts/run_dpmd_example.py -o runs/dpmd_water \
    --dp-setting dp.txt --dp-model DNN_seed2.pb

# DP が使えない環境 (NEP へフォールバック)
python scripts/run_dpmd_example.py -o runs/dpmd_water \
    --fallback-potential /path/to/nep89.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from gpumd_toolkit import GPUMDEnvironment
from gpumd_toolkit.examples import DPMDExample
from gpumd_toolkit.examples.dpmd import DEFAULT_REFERENCE_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DPMD example の実行と予実管理",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-o", "--workdir", default="runs/dpmd_water", help="計算ディレクトリ")
    parser.add_argument("--dp-setting", default=None, help="GPUMD 用 DP 設定ファイル (dp.txt)")
    parser.add_argument("--dp-model", default=None, help="deepmd-kit の凍結モデル (.pb)")
    parser.add_argument("--fallback-potential", default=None,
                        help="DP が使えないときに使う NEP ポテンシャル")
    parser.add_argument("--reference-dir", default=str(DEFAULT_REFERENCE_DIR),
                        help="参照計算 (DP 実行済み example) のディレクトリ")
    parser.add_argument("--structure", default=None, help="初期構造 (既定: 参照 example の model.xyz)")
    parser.add_argument("--equilibration-steps", type=int, default=4000)
    parser.add_argument("--production-steps", type=int, default=10000)
    parser.add_argument("--time-step", type=float, default=0.5, help="時間刻み [fs]")
    parser.add_argument("--temperature", type=float, default=330.0)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--venv", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    env_kwargs = {}
    if args.venv:
        env_kwargs["venv"] = Path(args.venv)
    if args.gpu is not None:
        env_kwargs["gpu_id"] = args.gpu

    example = DPMDExample(
        args.workdir,
        dp_setting_file=args.dp_setting,
        dp_model=args.dp_model,
        fallback_potential=args.fallback_potential,
        reference_dir=args.reference_dir,
        structure=args.structure,
        equilibration_steps=args.equilibration_steps,
        production_steps=args.production_steps,
        time_step=args.time_step,
        temperature=args.temperature,
        environment=GPUMDEnvironment(**env_kwargs),
        overwrite=args.overwrite,
    )

    support = example.check_dp_support()
    print("DP サポートの確認:")
    for key, value in support.items():
        print(f"  {key}: {value}")
    if not example.dp_available:
        print("  -> DP が使えないため NEP へフォールバックします。")
        if example.fallback_potential is None:
            print("  ERROR: --fallback-potential に NEP ポテンシャルを指定してください。")
            return 1

    report = example.execute(plot=not args.no_plot)
    print()
    print(report.to_markdown())
    print(f"出力先: {Path(args.workdir).resolve() / 'analysis'}")
    return 0 if report.all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
