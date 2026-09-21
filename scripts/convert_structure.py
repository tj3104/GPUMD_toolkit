#!/usr/bin/env python
"""構造ファイルを高速読み取り可能な extxyz に変換する (cif2xyz / POSCAR2xyz)。

``ase.io.read`` は CIF のような形式だと原子数が増えたとき極端に遅くなる
(Cu 4,000 原子で 28.7 s)。一度 extxyz にしておけば 0.002 s で読める。

使用例
------
# 単体の変換 (出力を省略すると同じ場所に <stem>.xyz を作る)
python scripts/convert_structure.py big.cif
python scripts/convert_structure.py POSCAR -o model.xyz

# まとめて変換
python scripts/convert_structure.py structures/*.cif --outdir xyz/

# どれくらい速くなるかを測る
python scripts/convert_structure.py big.cif --benchmark

# キャッシュ (GPUMD_TOOLKIT_CACHE か ~/.cache/gpumd_toolkit/xyz) を消す
python scripts/convert_structure.py --clear-cache
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from gpumd_toolkit import fastio
from gpumd_toolkit.structure import StructureHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="構造ファイルを extxyz に変換して読み込みを速くする",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("sources", nargs="*", help="入力構造 (CIF / POSCAR / ...)")
    parser.add_argument("-o", "--output", default=None, help="出力先 (入力が 1 つのとき)")
    parser.add_argument("--outdir", default=None, help="出力ディレクトリ (複数入力向け)")
    parser.add_argument("--format", default=None, help="入力フォーマットを明示する")
    parser.add_argument("--benchmark", action="store_true",
                        help="ase / 高速経路 / キャッシュの所要時間を比べる")
    parser.add_argument("--clear-cache", action="store_true", help="xyz キャッシュを削除する")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.clear_cache:
        print(f"キャッシュを {fastio.clear_cache()} 件削除しました")
        if not args.sources:
            return 0

    if not args.sources:
        build_parser().error("入力構造を 1 つ以上指定してください。")
    if args.output and len(args.sources) > 1:
        build_parser().error("-o は入力が 1 つのときだけ使えます (--outdir を使ってください)。")

    for source in args.sources:
        source = Path(source)
        if args.benchmark:
            timings = fastio.benchmark_readers(source)
            print(f"{source}:")
            for name, seconds in timings.items():
                print(f"  {name:8s} {seconds:8.4f} s")
            continue

        if args.output:
            output = Path(args.output)
        elif args.outdir:
            output = Path(args.outdir) / (source.stem + ".xyz")
        else:
            output = None
        written = fastio.convert_to_xyz(source, output, format=args.format)
        info = StructureHandler.info(fastio.read_xyz_fast(written))
        print(f"{source}  ->  {written}")
        print(f"    {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
