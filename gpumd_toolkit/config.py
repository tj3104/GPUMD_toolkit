"""GPUMD 実行環境(仮想環境・実行ファイル・GPU)の設定。

GPUMD 本体はコンパイル済みバイナリ (``gpumd`` / ``nep``) として提供される。
本ツールキットでは Python から ``subprocess`` 経由で起動するが、その際に
ユーザの仮想環境 (既定: ``/home/tajimamainpc/.venv/gpumd312``) を
``source`` したログインシェル上で実行する。
ログインシェルを使うのは ``~/.profile`` 等で ``PATH`` に GPUMD の
インストール先が追加されているため。
"""

from __future__ import annotations

import contextlib
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

__all__ = ["GPUMDEnvironment", "CommandResult", "DEFAULT_VENV"]

DEFAULT_VENV = Path(os.environ.get("GPUMD_VENV", "/home/tajimamainpc/.venv/gpumd312"))


@dataclass
class CommandResult:
    """``gpumd`` / ``nep`` 実行結果。"""

    command: str
    returncode: int
    elapsed: float
    workdir: Path
    stdout_file: Path | None = None
    stderr_file: Path | None = None
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def raise_for_status(self) -> "CommandResult":
        if not self.ok:
            tail = "\n".join((self.stderr or self.stdout).splitlines()[-40:])
            raise RuntimeError(
                f"コマンドが異常終了しました (returncode={self.returncode})\n"
                f"  command : {self.command}\n"
                f"  workdir : {self.workdir}\n"
                f"  log tail:\n{tail}"
            )
        return self


@dataclass
class GPUMDEnvironment:
    """GPUMD をどう起動するかを保持する設定オブジェクト。

    Parameters
    ----------
    venv
        起動前に ``source`` する Python 仮想環境。``None`` で無効化。
    gpumd_command, nep_command
        実行ファイル名またはフルパス。
    use_login_shell
        ``bash -l`` を使うか。``~/.profile`` の PATH 設定を取り込むため既定で True。
    gpu_id
        使用 GPU。``CUDA_VISIBLE_DEVICES`` として渡される。``None`` で既定のまま。
    extra_env
        追加の環境変数。
    """

    venv: Path | None = DEFAULT_VENV
    gpumd_command: str = "gpumd"
    nep_command: str = "nep"
    use_login_shell: bool = True
    gpu_id: int | None = None
    extra_env: Mapping[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------ utils
    def __post_init__(self) -> None:
        if self.venv is not None:
            self.venv = Path(self.venv).expanduser()
            if not (self.venv / "bin" / "activate").is_file():
                raise FileNotFoundError(
                    f"仮想環境が見つかりません: {self.venv} "
                    "(GPUMD_VENV 環境変数か venv= 引数で指定してください)"
                )

    def _prelude(self) -> str:
        parts: list[str] = []
        if self.venv is not None:
            parts.append(f"source {shlex.quote(str(self.venv / 'bin' / 'activate'))}")
        if self.gpu_id is not None:
            parts.append(f"export CUDA_VISIBLE_DEVICES={int(self.gpu_id)}")
        for key, value in self.extra_env.items():
            parts.append(f"export {key}={shlex.quote(str(value))}")
        return " && ".join(parts)

    def shell_command(self, command: str) -> list[str]:
        """``command`` を仮想環境つきシェルで実行するための argv を返す。"""
        prelude = self._prelude()
        full = f"{prelude} && {command}" if prelude else command
        flags = "-lc" if self.use_login_shell else "-c"
        return ["bash", flags, full]

    def which(self, command: str) -> str | None:
        """仮想環境を有効化した状態での実行ファイルの絶対パスを返す。"""
        probe = subprocess.run(
            self.shell_command(f"command -v {shlex.quote(command)}"),
            capture_output=True,
            text=True,
        )
        path = probe.stdout.strip().splitlines()
        if probe.returncode == 0 and path:
            return path[-1]
        return shutil.which(command)

    def check(self) -> dict[str, str | None]:
        """``gpumd`` / ``nep`` が使えるか確認し、解決されたパスを返す。"""
        return {
            "gpumd": self.which(self.gpumd_command),
            "nep": self.which(self.nep_command),
            "venv": str(self.venv) if self.venv else None,
        }

    # -------------------------------------------------------------------- run
    def run(
        self,
        command: str,
        workdir: Path | str,
        *,
        log_file: str | Path | None = "gpumd.out",
        err_file: str | Path | None = "gpumd.err",
        timeout: float | None = None,
        check: bool = True,
        capture_tail: int = 4000,
    ) -> CommandResult:
        """``workdir`` で ``command`` を実行する。

        標準出力/標準エラーは ``log_file`` / ``err_file`` に保存する
        (GPUMD は非常に多くのログを吐くのでメモリに溜めない)。
        """
        workdir = Path(workdir).expanduser().resolve()
        workdir.mkdir(parents=True, exist_ok=True)

        out_path = workdir / log_file if log_file else None
        err_path = workdir / err_file if err_file else None

        argv = self.shell_command(command)
        start = time.perf_counter()
        with contextlib.ExitStack() as stack:
            out = stack.enter_context(open(out_path, "w")) if out_path else subprocess.PIPE
            err = stack.enter_context(open(err_path, "w")) if err_path else subprocess.PIPE
            proc = subprocess.run(
                argv, cwd=workdir, stdout=out, stderr=err, text=True, timeout=timeout
            )
        elapsed = time.perf_counter() - start

        result = CommandResult(
            command=command,
            returncode=proc.returncode,
            elapsed=elapsed,
            workdir=workdir,
            stdout_file=out_path,
            stderr_file=err_path,
            stdout=_tail(out_path, capture_tail) if out_path else (proc.stdout or ""),
            stderr=_tail(err_path, capture_tail) if err_path else (proc.stderr or ""),
        )
        if check:
            result.raise_for_status()
        return result

    def run_gpumd(self, workdir: Path | str, **kwargs) -> CommandResult:
        return self.run(self.gpumd_command, workdir, **kwargs)

    def run_nep(self, workdir: Path | str, **kwargs) -> CommandResult:
        kwargs.setdefault("log_file", "nep.out")
        kwargs.setdefault("err_file", "nep.err")
        return self.run(self.nep_command, workdir, **kwargs)

    def replace(self, **changes) -> "GPUMDEnvironment":
        """一部の設定だけ差し替えた複製を返す (並列実行で GPU を振り分ける用)。"""
        params = dict(
            venv=self.venv,
            gpumd_command=self.gpumd_command,
            nep_command=self.nep_command,
            use_login_shell=self.use_login_shell,
            gpu_id=self.gpu_id,
            extra_env=dict(self.extra_env),
        )
        params.update(changes)
        return GPUMDEnvironment(**params)


def _tail(path: Path | None, nchars: int) -> str:
    if path is None or not path.is_file():
        return ""
    text = path.read_text(errors="replace")
    return text[-nchars:]


def available_gpus() -> list[int]:
    """``nvidia-smi`` から利用可能な GPU の index を返す (失敗時は空リスト)。"""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    return [int(line) for line in proc.stdout.split() if line.strip().isdigit()]
