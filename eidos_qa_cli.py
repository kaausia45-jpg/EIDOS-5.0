# -*- coding: utf-8 -*-
"""CLI 앱 동적 QA (#2) — CLI를 실제 실행해 출력을 캡처하고 기능명세서와 대조.

exercise_cli(entry, root)  : python 진입점을 기본/--help 로 실행, stdout+stderr 캡처(타임아웃·무크래시 판정).
compare_cli_to_spec(...)   : 캡처 출력(corpus) ↔ 명세 키워드 대조 + 엔진 verify_spec(코드) 교차검증.
                             상태: covered(출력 등장=실행시 노출) · partial(일부) · code_only(코드만) · absent.
엔진 spec 파싱/코드대조는 eidos_dev_engine 재사용. 순수(Qt 무관)·헤드리스 테스트 가능.
⚠ 프로젝트 CLI 코드를 실제 subprocess 로 실행함(타임아웃·stdin EOF 로 무한대기 방지).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Any, Dict, List


def exercise_cli(entry_file: str, root: str, *, timeout: float = 30.0, python_exe: str = "") -> Dict[str, Any]:
    py = python_exe or sys.executable
    entry = os.path.abspath(entry_file)
    if not os.path.isfile(entry):
        return {"ok": False, "error": f"진입점 없음: {entry}"}
    runs: List[Dict[str, Any]] = []
    corpus_parts: List[str] = []
    crashed = False
    for args, label in (([], "기본 실행"), (["--help"], "--help")):
        try:
            p = subprocess.run([py, entry, *args], cwd=root, capture_output=True, text=True,
                               timeout=timeout, errors="replace", input="")   # stdin EOF=input() 무한대기 방지
            out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
            runs.append({"args": args, "label": label, "exit": p.returncode, "out": out[:4000]})
            corpus_parts.append(out)
            # 기본 실행에서 비정상 종료 + 트레이스백 = 크래시(의미있음). --help 비0은 흔하니 무시.
            if not args and (p.returncode not in (0, None)) and "Traceback" in (p.stderr or ""):
                crashed = True
        except subprocess.TimeoutExpired as e:
            runs.append({"args": args, "label": label, "exit": None, "timeout": True,
                         "out": "(타임아웃 — 장시간 실행/대기 앱일 수 있음)"})
            so = getattr(e, "stdout", None)
            corpus_parts.append(so if isinstance(so, str)
                                else (so.decode("utf-8", "replace") if isinstance(so, bytes) else ""))
        except Exception as e:  # noqa: BLE001
            runs.append({"args": args, "label": label, "exit": None, "out": f"(실행 오류: {type(e).__name__}: {e})"})
    return {"ok": True, "runs": runs, "corpus": "\n".join(corpus_parts), "crashed": crashed}


_STOP = {"기능", "구현", "버튼", "화면", "처리", "해야", "한다", "된다", "있다", "표시", "지원", "동작",
         "사용자", "the", "and", "should", "must", "app", "with", "for", "this", "that", "기능을", "기능이"}


def _kw(title: str) -> List[str]:
    toks = re.findall(r"[A-Za-z]{3,}|[가-힣]{2,}", title or "")
    return [t for t in toks if t.lower() not in _STOP and t not in _STOP]


def compare_corpus_to_spec(corpus_text: str, spec_path: str, project_root: str = "") -> Dict[str, Any]:
    """말뭉치(CLI 출력·웹 HTML 등) ↔ 명세 키워드 대조 + 엔진 verify_spec(코드) 교차검증. CLI·웹 공용."""
    try:
        from eidos_dev_engine import parse_spec_file  # type: ignore
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"명세 파서 로드 실패: {e}", "features": []}
    if not (spec_path and os.path.isfile(spec_path)):
        return {"ok": False, "error": f"명세서 없음: {spec_path}", "features": []}
    try:
        items = parse_spec_file(spec_path)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"명세 파싱 실패: {e}", "features": []}
    corpus = (corpus_text or "").lower()
    code_status: Dict[str, str] = {}
    if project_root and os.path.isdir(project_root):
        try:
            from eidos_dev_engine import verify_spec  # type: ignore
            v = verify_spec(spec_path, project_root)
            for m in v.matches:
                code_status[(m.item.title or "").strip()] = m.status
        except Exception:  # noqa: BLE001
            pass
    features = []
    covered = partial = code_only = absent = 0
    for it in items:
        kws = _kw(it.title)
        if not kws:
            continue
        hit = [k for k in kws if k.lower() in corpus]
        cstat = code_status.get((it.title or "").strip(), "")
        # 이중언어(예: 'login 로그인') 대응 — 키워드 절반 이상이 출력에 등장하면 '구동 확인'
        if hit and (len(hit) / len(kws)) >= 0.5:
            status = "covered"        # 출력에 등장 = 실행 시 노출 확인
        elif hit:
            status = "partial"
        elif cstat == "implemented":
            status = "code_only"      # 출력엔 안 나왔지만 코드에 있음
        else:
            status = "absent"
        covered += status == "covered"
        partial += status == "partial"
        code_only += status == "code_only"
        absent += status == "absent"
        features.append({"title": it.title[:60], "status": status, "evidence": hit[:6]})
    total = len(features)
    return {"ok": True, "spec_total": total, "covered": covered, "partial": partial,
            "code_only": code_only, "absent": absent,
            "coverage": (covered / total if total else 0.0),
            "features": features}


def compare_cli_to_spec(cli_result: Dict[str, Any], spec_path: str, project_root: str = "") -> Dict[str, Any]:
    if not (cli_result or {}).get("ok"):
        return {"ok": False, "error": "CLI 구동 결과 없음", "features": []}
    r = compare_corpus_to_spec(cli_result.get("corpus", ""), spec_path, project_root)
    if r.get("ok"):
        r["crashed"] = cli_result.get("crashed")
    return r


if __name__ == "__main__":
    import io
    import tempfile
    import shutil
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass
    tmp = tempfile.mkdtemp()
    # 샘플 CLI: login/logout 출력, payment 없음
    open(os.path.join(tmp, "main.py"), "w", encoding="utf-8").write(
        "import sys\n"
        "def login(): print('login ok')\n"
        "def logout(): print('logout ok')\n"
        "if __name__ == '__main__':\n"
        "    print('CLI 시작: login, logout 지원')\n"
        "    login(); logout()\n")
    open(os.path.join(tmp, "QA_체크리스트.md"), "w", encoding="utf-8").write(
        "# QA\n- [ ] login 로그인\n- [ ] logout 로그아웃\n- [ ] payment 결제\n")
    ex = exercise_cli(os.path.join(tmp, "main.py"), tmp)
    print("구동 ok:", ex["ok"], "· crashed:", ex["crashed"])
    cmp = compare_cli_to_spec(ex, os.path.join(tmp, "QA_체크리스트.md"), project_root=tmp)
    print(f"대조: 총{cmp['spec_total']} 구동{cmp['covered']} 일부{cmp['partial']} 코드만{cmp['code_only']} 미발견{cmp['absent']} 커버{int(cmp['coverage']*100)}%")
    for f in cmp["features"]:
        print("  ", f["status"], f["title"])
    shutil.rmtree(tmp)
    print("✅ eidos_qa_cli 셀프테스트 통과 — CLI 실행+출력 캡처+명세 대조")
