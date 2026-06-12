# -*- coding: utf-8 -*-
"""EIDOS → Polaris(수렴 엔진) 런처 — Polaris 순수성 보존을 위해 subprocess 로만 부른다.

helix_launcher 의 쌍. Helix 가 발산(많이 떠올려 거름)이면 Polaris 는 수렴(층마다 즉시 현실검증해
하나로 좁힘 → 장기계획). 한글 폴더명·UTF-8·EIDOS_ROOT 함정을 여기서 흡수한다.

  from polaris_launcher import stream_polaris_vision
  res = stream_polaris_vision("월 매출 300만원 부업", on_line=print)   # 수렴+계획+못박기, 스트리밍
  # 완료 후 res.lock_path 에 못박힌 계획(JSON) — 정기 실행은 polaris_exec.py 가 감독.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import Callable, Optional

EIDOS_ROOT = os.path.dirname(os.path.abspath(__file__))
POLARIS_DIR = os.path.join(EIDOS_ROOT, "발명 엔진")
DEFAULT_LOCK = os.path.join(EIDOS_ROOT, "eidos_files", "polaris_plan.json")


def _env() -> dict:
    e = dict(os.environ)
    e["PYTHONUTF8"] = "1"
    e["PYTHONIOENCODING"] = "utf-8"
    e["EIDOS_ROOT"] = EIDOS_ROOT   # Polaris _bridge 가 EIDOS 키(Gemini·SerpAPI)를 찾도록
    return e


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _cwd() -> str | None:
    return None if _is_frozen() else POLARIS_DIR


def _polaris_cmd(seed: str, *, why: str, horizon: str, max_depth: int,
                 candidates: int, lock_path: str) -> list:
    """Polaris 실행 명령 구성. 소스: [python, polaris_run.py, ...]. frozen: [exe, --polaris-run, ...].

    --lock 으로 수렴 결과(장기계획)를 lock_path 에 못박는다(이후 polaris_exec.py 가 감독 실행).
    """
    args = ["--lock", lock_path, "--horizon", horizon,
            "--max-depth", str(max_depth), "--candidates", str(candidates)]
    if why:
        args += ["--why", why]
    if _is_frozen():
        return [sys.executable, "--polaris-run", *args, seed]
    return [sys.executable, "polaris_run.py", *args, seed]


@dataclass
class PolarisResult:
    returncode: int
    stdout: str
    stderr: str
    lock_path: str = ""
    timed_out: bool = False   # 워치독이 시간초과로 프로세스를 죽였나(키 문제와 구분)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def stream_polaris_vision(
    seed: str,
    *,
    why: str = "",
    horizon: str = "3개월",
    max_depth: int = 4,
    candidates: int = 3,
    lock_path: Optional[str] = None,
    on_line: Optional[Callable[[str], None]] = None,
    timeout: float = 420.0,
) -> PolarisResult:
    """Polaris 로 대상을 단일 비전으로 수렴 + 장기계획 + 못박기. 진행/계획을 라인 즉시 on_line 으로.

    polaris_run 은 진행을 stderr, 계획을 stdout 에 쓴다 → 합쳐서 순서대로 읽는다.
    timeout 초과 시 watchdog 가 프로세스를 죽인다(무한 대기 방지). 완료 후 lock_path 에 계획 저장.
    """
    lock_path = lock_path or DEFAULT_LOCK
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    cmd = _polaris_cmd(seed, why=why, horizon=horizon, max_depth=max_depth,
                       candidates=candidates, lock_path=lock_path)
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
        for raw in proc.stdout:                 # 라인이 나오는 즉시 도착
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
    return PolarisResult(returncode=rc, stdout="\n".join(lines), stderr="",
                         lock_path=lock_path, timed_out=timed_out["v"])


def run_polaris_vision(seed: str, **kwargs) -> PolarisResult:
    """스트리밍 없이 완료까지 대기하는 버전(on_line=None)."""
    return stream_polaris_vision(seed, on_line=None, **kwargs)


# ── 못박힌 계획을 텔레그램 감독 하에 실행(백그라운드·비차단) ──
def _exec_runner() -> list:
    """polaris_exec 진입부. 소스: [python, polaris_exec.py]. frozen: [exe, --polaris-exec]."""
    if _is_frozen():
        return [sys.executable, "--polaris-exec"]
    return [sys.executable, os.path.join(POLARIS_DIR, "polaris_exec.py")]


def _polaris_exec_cmd(plan_path: str, *, mode: str, interval: float,
                      max_steps: Optional[int], max_runtime: float = 0.0) -> list:
    """polaris_exec 실행 명령(전체)."""
    args = [plan_path]
    if mode == "due":
        args.append("--due")
    else:   # watch(기본) — 주기적으로 깨어나 due 된 단계 감독
        args += ["--watch", "--interval", str(int(interval))]
        if max_runtime and max_runtime > 0:
            args += ["--max-runtime", str(int(max_runtime))]   # 하루 활동시간 캡(비용관리)
    if max_steps:
        args += ["--max-steps", str(int(max_steps))]
    return [*_exec_runner(), *args]


def start_supervised(plan_path: str, *, mode: str = "watch", interval: float = 600.0,
                     max_steps: Optional[int] = None, max_runtime: float = 0.0):
    """못박힌 계획을 텔레그램 감독 실행(백그라운드 Popen·비차단). Popen 반환.

    mode='watch': interval 초마다 깨어나 due 된 단계를 텔레그램 보고→승인→실행(데몬).
                  max_runtime>0 이면 그 시간(초) 뒤 자동 종료(하루 N시간만 일하게·비용관리).
    mode='due'  : 지금 due 된 단계만 1회 처리 후 종료(cron/작업스케줄러용).
    이 프로세스는 부모(EIDOS)가 살아 있는 동안 동작한다. *무인·영구* 실행은 register_daily_task() 로
    작업 스케줄러에 등록(GUI 없이 매일·재부팅에도). 텔레그램 키는 supervisor 가 _bridge 로 읽는다.
    """
    cmd = _polaris_exec_cmd(plan_path, mode=mode, interval=interval,
                            max_steps=max_steps, max_runtime=max_runtime)
    return subprocess.Popen(cmd, cwd=_cwd(), env=_env())


# ── 무인·영구 실행: Windows 작업 스케줄러 등록/해제 ──
DEFAULT_TASK_NAME = "PolarisDaily"


def daily_task_command(plan_path: str, *, at: str = "13:00", hours: float = 3.0,
                       interval: float = 900.0, task_name: str = DEFAULT_TASK_NAME) -> tuple[list, str]:
    """(subprocess 리스트, 사람이 읽는 schtasks 문자열) 반환 — 매일 at 시각에 hours 시간만 감독 실행.

    무인 모드는 --watch + --max-runtime(=hours) — 하루 정해진 시간만 일하고 자동 종료(비용관리).
    경로 공백/한글은 list2cmdline 으로 안전 인용.
    """
    sec = int(float(hours) * 3600)
    inner = [*_exec_runner(), plan_path, "--watch", "--interval", str(int(interval)),
             "--max-runtime", str(sec)]
    tr = subprocess.list2cmdline(inner)
    cmd = ["schtasks", "/Create", "/TN", task_name, "/SC", "DAILY", "/ST", at, "/F", "/TR", tr]
    human = f'schtasks /Create /TN {task_name} /SC DAILY /ST {at} /F /TR "{tr}"'
    return cmd, human


def register_daily_task(plan_path: str, *, at: str = "13:00", hours: float = 3.0,
                        interval: float = 900.0, task_name: str = DEFAULT_TASK_NAME) -> tuple[bool, str]:
    """매일 at 시각, hours 시간만 도는 무인 감독 실행을 작업 스케줄러에 등록(Windows). (성공, 메시지)."""
    if os.name != "nt":
        return False, "무인 등록은 Windows 작업 스케줄러 전용이에요."
    cmd, human = daily_task_command(plan_path, at=at, hours=hours, interval=interval, task_name=task_name)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        return False, f"등록 실패: {e}\n수동 등록(관리자 터미널):\n{human}"
    if p.returncode == 0:
        return True, (f"✅ 무인 실행 등록 완료 — 매일 {at}부터 최대 {hours}시간 감독 실행.\n"
                      f"해제: schtasks /Delete /TN {task_name} /F")
    err = (p.stderr or p.stdout or "").strip()
    return False, f"등록 실패(코드 {p.returncode}): {err}\n수동 등록(관리자 터미널):\n{human}"


def unregister_daily_task(task_name: str = DEFAULT_TASK_NAME) -> tuple[bool, str]:
    """무인 실행 등록 해제(Windows)."""
    if os.name != "nt":
        return False, "Windows 전용이에요."
    try:
        p = subprocess.run(["schtasks", "/Delete", "/TN", task_name, "/F"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        return False, f"해제 실패: {e}"
    if p.returncode == 0:
        return True, f"🗑 무인 실행 등록 해제됨: {task_name}"
    return False, f"해제 실패(또는 등록 없음): {(p.stderr or p.stdout or '').strip()}"


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]
    except Exception:
        pass
    g = sys.argv[1] if len(sys.argv) > 1 else "월 매출 300만원 부업"
    r = stream_polaris_vision(g, on_line=lambda m: print(m))
    print(f"\n[lock] {r.lock_path}  (rc={r.returncode})")
    raise SystemExit(r.returncode)
