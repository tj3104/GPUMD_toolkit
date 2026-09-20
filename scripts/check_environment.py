#!/usr/bin/env python
"""GPUMD 実行環境の確認。

    python scripts/check_environment.py
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from gpumd_toolkit.diagnostics import environment_report, format_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GPUMD 実行環境の確認")
    parser.add_argument("--json", action="store_true", help="JSON で出力")
    parser.add_argument("--venv", default=None, help="確認する仮想環境")
    args = parser.parse_args(argv)

    environment = None
    if args.venv:
        from pathlib import Path

        from gpumd_toolkit import GPUMDEnvironment

        environment = GPUMDEnvironment(venv=Path(args.venv))

    report = environment_report(environment)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(format_report(report))
    executables = report["executables"]
    return 0 if executables.get("gpumd") else 1


if __name__ == "__main__":
    raise SystemExit(main())
