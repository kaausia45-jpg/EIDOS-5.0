# -*- coding: utf-8 -*-
"""EIDOS → Helix(발명 엔진) 런처 — Helix 순수성 보존을 위해 subprocess 로만 부른다.

EIDOS 본체(72K GUI)를 건드리지 않고, 한글 폴더명·UTF-8·EIDOS_ROOT 같은 함정을 여기서 흡수한다.
나중에 채팅 명령(/목표, /발명)에서 run_helix_goal 을 호출하도록 배선하면 된다.

  from helix_launcher import run_helix_goal, launch_helix_gui
  res = run_helix_goal("calc.py 버그 고쳐줘")      # 헤드리스, (returncode, stdout, stderr)
  launch_helix_gui()                                # Helix GUI 띄움(비차단)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import Callable, Optional

EIDOS_ROOT = os.path.dirname(os.path.abspath(__file__))
HELIX_DIR = os.path.join(EIDOS_ROOT, "발명 엔진")


def _env() -> dict:
    e = dict(os.environ)
    e["PYTHONUTF8"] = "1"
    e["PYTHONIOENCODING"] = "utf-8"
    e["EIDOS_ROOT"] = EIDOS_ROOT   # Helix _bridge 가 EIDOS 손/키를 확실히 찾도록
    return e


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _helix_cmd(goal: str, max_goals: int, max_depth: int, as_json: bool,
               files: Optional[list] = None, dirs: Optional[list] = None,
               images: Optional[list] = None) -> list:
    """Helix 실행 명령 구성.

    소스 실행: [python, helix_run.py, ...옵션, goal]  (cwd=발명 엔진)
    frozen exe: [exe, --helix-run, ...옵션, goal]  — exe 가 자기 자신을 재호출(eidos_chat_gui 최상단
                dispatch 가 GUI 대신 Helix 만 실행). state 는 사용자 쓰기가능 경로(_MEIPASS 는 읽기전용).

    files/dirs: 첨부 파일·폴더 → helix_run 의 --file/--dir 로 전달(분석 근거 주입).
                첨부가 있으면 helix_run 이 자동으로 reason(5W2H 분석) 렌즈로 라우팅한다.
    """
    args = ["--max-goals", str(max_goals), "--max-depth", str(max_depth)]
    if as_json:
        args.append("--json")
    for f in (files or []):
        if f:
            args += ["--file", str(f)]
    for d in (dirs or []):
        if d:
            args += ["--dir", str(d)]
    for im in (images or []):           # 첨부 이미지 → 지각(perceive) 손
        if im:
            args += ["--image", str(im)]
    if _is_frozen():
        state = os.path.join(os.path.expanduser("~"), ".eidos", "helix_state.json")
        return [sys.executable, "--helix-run", *args, "--state", state, goal]
    return [sys.executable, "helix_run.py", *args, goal]


def _cwd() -> str | None:
    return None if _is_frozen() else HELIX_DIR   # frozen 자식은 _MEIPASS 경로를 스스로 세팅


@dataclass
class HelixResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False   # 워치독이 시간초과로 프로세스를 죽였나(키 문제와 구분)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def run_helix_goal(goal: str, *, max_goals: int = 6, max_depth: int = 2,
                   as_json: bool = False, timeout: float = 300.0,
                   files: Optional[list] = None, dirs: Optional[list] = None,
                   images: Optional[list] = None) -> HelixResult:
    """Helix 헤드리스로 목표를 자율 추구하고 결과를 캡처해 반환(비차단 아님 — 완료까지 대기).

    files/dirs: 첨부 파일·폴더(분석 근거). 있으면 Helix 가 reason(분석) 렌즈로 자동 라우팅.
    images: 첨부 이미지 → 지각(perceive) 손으로 라우팅(EIDOS 가 본다).
    """
    cmd = _helix_cmd(goal, max_goals, max_depth, as_json, files=files, dirs=dirs, images=images)
    proc = subprocess.run(cmd, cwd=_cwd(), env=_env(),
                          capture_output=True, text=True, encoding="utf-8", timeout=timeout)
    return HelixResult(returncode=proc.returncode, stdout=proc.stdout or "", stderr=proc.stderr or "")


def stream_helix_goal(goal: str, *, max_goals: int = 6, max_depth: int = 2,
                      on_line: Optional[Callable[[str], None]] = None,
                      timeout: float = 300.0,
                      files: Optional[list] = None, dirs: Optional[list] = None,
                      images: Optional[list] = None) -> HelixResult:
    """run_helix_goal 의 *스트리밍* 버전 — 진행/결과를 라인이 나오는 즉시 on_line 으로 흘린다.

    helix_run 은 진행을 stderr, 결과를 stdout 에 쓴다 → stderr=STDOUT 으로 합쳐 순서대로 읽는다.
    timeout 초과 시 watchdog 가 프로세스를 죽인다(무한 대기 방지).
    files/dirs: 첨부 파일·폴더(분석 근거). 있으면 Helix 가 reason(분석) 렌즈로 자동 라우팅.
    """
    cmd = _helix_cmd(goal, max_goals, max_depth, as_json=False, files=files, dirs=dirs, images=images)
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
    return HelixResult(returncode=rc, stdout="\n".join(lines), stderr="", timed_out=timed_out["v"])


def record_helix_feedback(op: str, valence: float, *, timeout: float = 30.0) -> HelixResult:
    """외부 피드백(👍/👎)을 Helix 마음에 반영 — op 손 신뢰도를 valence(-1~1)로 갱신·영속.

    GUI 의 👍/👎 버튼이 호출. helix_run --feedback 'op:valence' (frozen: exe --helix-run) subprocess.
    """
    op = (op or "").strip()
    if not op:
        return HelixResult(returncode=1, stdout="", stderr="no_op")
    fb = f"{op}:{float(valence)}"
    if _is_frozen():
        state = os.path.join(os.path.expanduser("~"), ".eidos", "helix_state.json")
        cmd = [sys.executable, "--helix-run", "--feedback", fb, "--state", state]
    else:
        cmd = [sys.executable, "helix_run.py", "--feedback", fb]
    try:
        proc = subprocess.run(cmd, cwd=_cwd(), env=_env(),
                              capture_output=True, text=True, encoding="utf-8", timeout=timeout)
        return HelixResult(returncode=proc.returncode, stdout=proc.stdout or "", stderr=proc.stderr or "")
    except Exception as exc:  # noqa: BLE001
        return HelixResult(returncode=1, stdout="", stderr=str(exc))


def ground_reqspec(projects_path: str = "", mind_path: str = "", *, timeout: float = 60.0) -> dict:
    """req-spec 결과를 Helix 학습 substrate 로 grounding 하고 타입별 성공률 인사이트를 반환(LLM 0).

    실패해도 예외 없이 {} 반환(graceful). source: helix_ground_reqspec.py / frozen: exe --ground-reqspec.
    """
    if _is_frozen():
        cmd = [sys.executable, "--ground-reqspec", "--json"]
    else:
        cmd = [sys.executable, "helix_ground_reqspec.py", "--json"]
    # 위치 인자(projects, mind) — argparse 가 옵션 뒤 위치로 받게 빈 값은 제외
    pos = [a for a in (projects_path, mind_path) if a]
    cmd += pos
    try:
        proc = subprocess.run(cmd, cwd=_cwd(), env=_env(),
                              capture_output=True, text=True, encoding="utf-8", timeout=timeout)
        out = (proc.stdout or "").strip()
        # 마지막 JSON 줄 파싱(앞에 경고가 섞일 수 있음)
        for line in reversed(out.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
    except Exception:  # noqa: BLE001
        pass
    return {}


def launch_helix_gui() -> subprocess.Popen:
    """Helix GUI(발명 지도 + 채팅)를 별도 프로세스로 띄운다(비차단). petalbot 과 같은 패턴."""
    return subprocess.Popen([sys.executable, "main.py"], cwd=HELIX_DIR, env=_env())


if __name__ == "__main__":
    # 수동 실행 시 콘솔 인코딩 고정(Windows cp949 에서 한글/이모지 출력 죽음 방지).
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]
    except Exception:
        pass
    # 빠른 수동 확인:  python helix_launcher.py "AI 에이전트 시장 조사"
    g = sys.argv[1] if len(sys.argv) > 1 else "AI 에이전트 시장 조사"
    r = run_helix_goal(g, as_json=False)
    print(r.stdout)
    if r.stderr:
        print("--- stderr ---")
        print(r.stderr)
    raise SystemExit(r.returncode)
