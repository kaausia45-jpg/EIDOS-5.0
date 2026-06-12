# -*- coding: utf-8 -*-
"""EIDOS → Polaris autodrive(컴퓨터 조작 폐루프) 런처 — 순수성 보존 위해 subprocess 로만 부른다.

polaris_launcher 의 형제. polaris_run(수렴·생각)·polaris_exec(감독 실행)와 달리 이건 *실시간 감각운동*:
화면을 보고 깊이2로 행동 1개를 정해 실행하고 재관측으로 검증하는 루프를 헤드리스로 돌린다.

  from autodrive_launcher import stream_autodrive
  res = stream_autodrive("계산기로 7×8 눌러서 56 띄워라", window="계산기", on_line=print)

안전: 기본은 dry(클릭 안 하고 결정만)·비가역 행동 에스컬레이션 ON. 실제 조작은 dry=False 로.
한글 폴더명·UTF-8·EIDOS_ROOT 함정을 여기서 흡수(polaris_launcher 와 동일).
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import Callable, Optional

EIDOS_ROOT = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.join(EIDOS_ROOT, "발명 엔진")


def _env() -> dict:
    e = dict(os.environ)
    e["PYTHONUTF8"] = "1"
    e["PYTHONIOENCODING"] = "utf-8"
    e["EIDOS_ROOT"] = EIDOS_ROOT   # autodrive 가 EIDOS 키(Gemini)를 찾도록
    return e


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _cwd() -> str | None:
    return None if _is_frozen() else ENGINE_DIR


def _drive_cmd(goal: str, *, window: str, region: Optional[tuple], max_ticks: int,
               max_no_progress: int, dry: bool, no_verify: bool, allow_risky: bool) -> list:
    """autodrive 실행 명령. 소스: [python, polaris_drive.py, ...]. frozen: [exe, --polaris-drive, ...]."""
    args: list[str] = ["--max-ticks", str(max_ticks), "--max-no-progress", str(max_no_progress)]
    if window:
        args += ["--window", window]
    if region and len(region) == 4:
        args += ["--region", ",".join(str(int(v)) for v in region)]
    if dry:
        args.append("--dry")
    if no_verify:
        args.append("--no-verify")
    if allow_risky:
        args.append("--allow-risky")
    if _is_frozen():
        return [sys.executable, "--polaris-drive", *args, goal]
    return [sys.executable, "polaris_drive.py", *args, goal]


@dataclass
class AutodriveResult:
    returncode: int
    stdout: str
    timed_out: bool = False   # 워치독이 시간초과로 프로세스를 죽였나(키 문제와 구분)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def stream_autodrive(
    goal: str,
    *,
    window: str = "",
    region: Optional[tuple] = None,
    max_ticks: int = 10,
    max_no_progress: int = 3,
    dry: bool = True,
    no_verify: bool = False,
    allow_risky: bool = False,
    on_line: Optional[Callable[[str], None]] = None,
    timeout: float = 600.0,
) -> AutodriveResult:
    """화면을 보고 목표를 향해 컴퓨터를 조작. 진행을 라인 즉시 on_line 으로. timeout 초과 시 프로세스 kill.

    기본 dry=True(안전·결정만). 실제 클릭은 dry=False. allow_risky=True 면 비가역 행동도 자율 실행(옵트인).
    """
    cmd = _drive_cmd(goal, window=window, region=region, max_ticks=max_ticks,
                     max_no_progress=max_no_progress, dry=dry, no_verify=no_verify,
                     allow_risky=allow_risky)
    proc = subprocess.Popen(cmd, cwd=_cwd(), env=_env(),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    timed_out = {"v": False}

    def _watchdog_kill() -> None:
        timed_out["v"] = True   # 죽인 게 시간초과임을 표시(GUI 가 키 문제로 오인하지 않게)
        try:
            proc.kill()
        except Exception:
            pass

    killer = threading.Timer(timeout, _watchdog_kill)
    killer.daemon = True
    killer.start()
    lines: list[str] = []
    try:
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            lines.append(line)
            if on_line and line.strip():
                try:
                    on_line(line)
                except Exception:
                    pass
        proc.wait()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    finally:
        killer.cancel()
    rc = proc.returncode if proc.returncode is not None else -1
    return AutodriveResult(returncode=rc, stdout="\n".join(lines), timed_out=timed_out["v"])


def run_autodrive(goal: str, **kwargs) -> AutodriveResult:
    """스트리밍 없이 완료까지 대기(on_line=None)."""
    return stream_autodrive(goal, on_line=None, **kwargs)


# ── 기능명세 QA(명세서로 앱을 클릭 검사) ──
def _qa_cmd(spec_path: str, *, window: str, region: Optional[tuple], max_ticks_per_item: int,
            list_only: bool, out: str) -> list:
    args: list[str] = ["--max-ticks-per-item", str(max_ticks_per_item)]
    if window:
        args += ["--window", window]
    if region and len(region) == 4:
        args += ["--region", ",".join(str(int(v)) for v in region)]
    if list_only:
        args.append("--list-only")
    if out:
        args += ["--out", out]
    if _is_frozen():
        return [sys.executable, "--polaris-qa", *args, spec_path]
    return [sys.executable, "polaris_qa.py", *args, spec_path]


def stream_qa(
    spec_path: str,
    *,
    window: str = "",
    region: Optional[tuple] = None,
    max_ticks_per_item: int = 6,
    list_only: bool = False,
    out: str = "",
    on_line: Optional[Callable[[str], None]] = None,
    timeout: float = 1800.0,
) -> AutodriveResult:
    """기능명세서로 앱을 클릭 검사 → QA 리포트. 진행을 라인 즉시 on_line 으로(항목 多면 길어질 수 있음).

    list_only=True 면 클릭 없이 추출된 검증항목만 미리보기. out=리포트 저장 경로.
    """
    cmd = _qa_cmd(spec_path, window=window, region=region, max_ticks_per_item=max_ticks_per_item,
                  list_only=list_only, out=out)
    proc = subprocess.Popen(cmd, cwd=_cwd(), env=_env(),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    timed_out = {"v": False}

    def _watchdog_kill() -> None:
        timed_out["v"] = True   # 죽인 게 시간초과임을 표시(GUI 가 키 문제로 오인하지 않게)
        try:
            proc.kill()
        except Exception:
            pass

    killer = threading.Timer(timeout, _watchdog_kill)
    killer.daemon = True
    killer.start()
    lines: list[str] = []
    try:
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            lines.append(line)
            if on_line and line.strip():
                try:
                    on_line(line)
                except Exception:
                    pass
        proc.wait()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    finally:
        killer.cancel()
    rc = proc.returncode if proc.returncode is not None else -1
    return AutodriveResult(returncode=rc, stdout="\n".join(lines), timed_out=timed_out["v"])


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]
    except Exception:
        pass
    g = sys.argv[1] if len(sys.argv) > 1 else "계산기로 7 곱하기 8 눌러서 56 띄워라"
    r = stream_autodrive(g, window="계산기", dry=True, on_line=lambda m: print(m))
    raise SystemExit(r.returncode)
