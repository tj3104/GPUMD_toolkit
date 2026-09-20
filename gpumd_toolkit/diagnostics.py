"""環境診断。

GPUMD 実行ファイル・GPU・Python パッケージが揃っているかをまとめて確認する。
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import GPUMDEnvironment, available_gpus

__all__ = ["environment_report", "format_report"]

#: GPUMD 用の Python インターフェース候補 (mission.md の「推奨モジュール」)
PYTHON_INTERFACES = {
    "calorine": "GPUMD 公式推奨。model.xyz の入出力と NEP calculator (nep_cpu 同梱)。本ツールキットの主依存。",
    "gpyumd": "GPUMD 入出力の別実装。v0.1 のローダは pandas>=2.2 で動かない (delim_whitespace 廃止) ため、本ツールキットでは使っていない。",
    "pynep": "NEP_CPU ベースの ASE calculator。backend='pynep' で使える。",
}

CORE_PACKAGES = ("ase", "numpy", "scipy", "pandas", "matplotlib", "joblib")


def environment_report(environment: GPUMDEnvironment | None = None) -> dict[str, Any]:
    """環境の状態を辞書で返す。"""
    try:
        environment = environment or GPUMDEnvironment()
        venv = str(environment.venv) if environment.venv else None
        executables = environment.check()
    except FileNotFoundError as exc:
        venv = f"ERROR: {exc}"
        executables = {"gpumd": shutil.which("gpumd"), "nep": shutil.which("nep")}

    packages: dict[str, str | None] = {}
    for name in CORE_PACKAGES + tuple(PYTHON_INTERFACES):
        packages[name] = _version(name)

    gpus = available_gpus()
    report: dict[str, Any] = {
        "venv": venv,
        "executables": executables,
        "gpus": gpus,
        "gpu_names": _gpu_names(),
        "packages": packages,
        "interfaces": PYTHON_INTERFACES,
    }
    try:
        from .ase_interface import available_backends

        report["ase_backends"] = available_backends()
    except Exception as exc:
        report["ase_backends"] = {"error": str(exc)}
    return report


def _version(module: str) -> str | None:
    try:
        imported = importlib.import_module(module)
    except Exception:
        return None
    return getattr(imported, "__version__", "(version 不明)")


def _gpu_names() -> list[str]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def format_report(report: dict[str, Any] | None = None) -> str:
    """:func:`environment_report` の結果を読みやすい文字列にする。"""
    report = report or environment_report()
    lines = ["GPUMD 実行環境の確認", "=" * 60]
    lines.append(f"  仮想環境        : {report['venv']}")
    for name, path in report["executables"].items():
        if name == "venv":
            continue
        mark = "OK " if path else "NG "
        lines.append(f"  {mark}{name:<12s}: {path or '見つかりません'}")
    lines.append("")
    lines.append(f"  GPU             : {report['gpus'] or 'なし'}")
    for name in report["gpu_names"]:
        lines.append(f"                    {name}")
    lines.append("")
    lines.append("  Python パッケージ")
    for name in CORE_PACKAGES:
        version = report["packages"].get(name)
        lines.append(f"    {'OK ' if version else 'NG '}{name:<12s}: {version or '未インストール'}")
    lines.append("")
    lines.append("  GPUMD 用 Python インターフェース")
    for name, description in report["interfaces"].items():
        version = report["packages"].get(name)
        lines.append(f"    {'OK ' if version else '-- '}{name:<12s}: {version or '未インストール'}")
        lines.append(f"        {description}")
    lines.append("")
    lines.append("  ASE calculator のバックエンド")
    for name, ok in report["ase_backends"].items():
        lines.append(f"    {'OK ' if ok else 'NG '}{name}")
    return "\n".join(lines)
