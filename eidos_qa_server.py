# -*- coding: utf-8 -*-
"""웹 서버 동적 QA (#4) — fastapi/node 서버를 띄워 HTTP 응답을 명세서와 대조.

exercise_server(entry, root, stack): 서버 subprocess 기동 → 기동 로그/기본 포트로 URL 감지
    → 엔드포인트(/, /openapi.json, /docs, /health…) 응답 수집 → corpus. **항상 서버 종료(좀비 방지)**.
compare_server_to_spec(...)         : corpus ↔ 명세(eidos_qa_cli.compare_corpus_to_spec 공용).

⚠ 프로젝트 서버 코드를 실제 실행함. 포트는 기동 로그(uvicorn/node) 또는 기본값(8000/3000/5000/8080)으로 추정.
   종료는 Windows taskkill /T (프로세스 트리) → 워커/자식까지 정리. 순수(Qt 무관)·헤드리스 테스트 가능.
"""
from __future__ import annotations

import collections
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional

_PORT_RE = re.compile(r"https?://[\w.]+:(\d{2,5})|[Pp]ort[\s:=]+(\d{2,5})|listening[^\d]{0,24}(\d{4,5})")
_DEFAULT_PORTS = (8000, 3000, 5000, 8080, 5173)


def _launch_cmd(entry: str, stack: str) -> List[str]:
    if stack == "node" or entry.lower().endswith((".js", ".mjs", ".cjs")):
        return [shutil.which("node") or "node", entry]
    return [sys.executable, entry]   # fastapi 등 파이썬 (__main__ 에서 uvicorn.run 가정)


def _ping(url: str) -> bool:
    try:
        import requests
        requests.get(url, timeout=1.5)   # 404 도 '응답' = 살아있음
        return True
    except Exception:  # noqa: BLE001
        return False


def _kill(proc):
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=8)
        else:
            proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            proc.kill()
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def exercise_server(entry_file: str, root: str, stack: str = "", *, grace: float = 14.0,
                    probe_paths: Optional[List[str]] = None) -> Dict[str, Any]:
    import requests
    entry = os.path.abspath(entry_file)
    if not os.path.isfile(entry):
        return {"ok": False, "error": f"진입점 없음: {entry}"}
    cmd = _launch_cmd(entry, stack)
    if not cmd[0]:
        return {"ok": False, "error": "실행기 없음(node 미설치?)"}
    out_lines: "collections.deque[str]" = collections.deque(maxlen=400)
    try:
        proc = subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors="replace", bufsize=1)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"서버 실행 실패: {type(e).__name__}: {e}"}

    def _drain():
        try:
            for ln in proc.stdout:  # type: ignore
                out_lines.append(ln.rstrip())
        except Exception:  # noqa: BLE001
            pass
    threading.Thread(target=_drain, daemon=True).start()

    base = ""
    deadline = time.time() + grace
    log_only_until = time.time() + grace * 0.7   # 전반부=기동 로그만(남의 서버 오인 방지)
    try:
        while time.time() < deadline:
            for ln in list(out_lines):           # 1) 기동 로그 포트 — 항상 우선(내 서버 확실)
                m = _PORT_RE.search(ln)
                if m:
                    port = next((g for g in m.groups() if g), "")
                    cand = f"http://127.0.0.1:{port}"
                    if port and _ping(cand):
                        base = cand
                        break
            if base:
                break
            # 2) 기본 포트 핑 — 로그가 끝내 포트를 안 줄 때만(후반부). ⚠ 남의 서버에 걸릴 위험.
            if time.time() >= log_only_until:
                for p in _DEFAULT_PORTS:
                    cand = f"http://127.0.0.1:{p}"
                    if _ping(cand):
                        base = cand
                        break
                if base:
                    break
            if proc.poll() is not None:          # 서버가 죽음
                break
            time.sleep(0.5)

        crashed = (proc.poll() is not None) and not base
        corpus_parts: List[str] = []
        probed: List[Dict[str, Any]] = []
        if base:
            paths = probe_paths or ["/", "/openapi.json", "/docs", "/health", "/api", "/index.html"]
            for path in paths:
                try:
                    r = requests.get(base + path, timeout=4)
                    corpus_parts.append((r.text or "")[:6000])
                    probed.append({"path": path, "status": r.status_code})
                except Exception:  # noqa: BLE001
                    pass
    finally:
        _kill(proc)   # 무조건 종료(좀비/포트 점유 방지)

    log = "\n".join(list(out_lines)[-30:])
    ok = bool(base and (corpus_parts or probed))
    return {"ok": ok, "base": base, "corpus": "\n".join(corpus_parts) + "\n" + log,
            "probed": probed, "crashed": crashed, "log_tail": log,
            "error": "" if ok else "서버 기동/포트 감지 실패(기동 로그·기본포트 모두 무응답)"}


def compare_server_to_spec(server_result: Dict[str, Any], spec_path: str, project_root: str = "") -> Dict[str, Any]:
    if not (server_result or {}).get("ok"):
        return {"ok": False, "error": server_result.get("error", "서버 구동 결과 없음"), "features": []}
    from eidos_qa_cli import compare_corpus_to_spec
    r = compare_corpus_to_spec(server_result.get("corpus", ""), spec_path, project_root)
    if r.get("ok"):
        r["crashed"] = server_result.get("crashed")
    return r


if __name__ == "__main__":
    import io
    import tempfile
    import shutil as _sh
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass
    tmp = tempfile.mkdtemp()
    # 샘플 서버: 8765 바인딩, 기동 로그 출력, 페이지에 기능 노출
    open(os.path.join(tmp, "server.py"), "w", encoding="utf-8").write(
        "import http.server, socketserver\n"
        "class H(http.server.BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        self.send_response(200); self.send_header('Content-Type','text/html'); self.end_headers()\n"
        "        self.wfile.write('<html>로그인 login · 결제 payment 지원</html>'.encode('utf-8'))\n"
        "    def log_message(self,*a): pass\n"
        "PORT=8765\n"
        "print(f'Uvicorn running on http://127.0.0.1:{PORT}', flush=True)\n"
        "with socketserver.TCPServer(('127.0.0.1',PORT), H) as s: s.serve_forever()\n")
    open(os.path.join(tmp, "QA_체크리스트.md"), "w", encoding="utf-8").write(
        "# QA\n- [ ] 로그인 login\n- [ ] 결제 payment\n- [ ] 배송 shipping\n")
    ex = exercise_server(os.path.join(tmp, "server.py"), tmp, "fastapi", grace=10.0)
    print("서버 QA 기동:", ex.get("ok"), "· base:", ex.get("base"), "· probed:", [p["status"] for p in ex.get("probed", [])])
    cmp = compare_server_to_spec(ex, os.path.join(tmp, "QA_체크리스트.md"), project_root=tmp)
    if cmp.get("ok"):
        print(f"대조: 총{cmp['spec_total']} 응답{cmp['covered']} 일부{cmp['partial']} 코드만{cmp['code_only']} 미발견{cmp['absent']} 커버{int(cmp['coverage']*100)}%")
        for f in cmp["features"]:
            print("  ", f["status"], f["title"])
    else:
        print("대조 실패:", cmp.get("error"))
    _sh.rmtree(tmp)
    print("✅ eidos_qa_server 셀프테스트 — 서버 기동·HTTP 프로브·명세 대조·종료")
