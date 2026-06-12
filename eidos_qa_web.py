# -*- coding: utf-8 -*-
"""웹 정적 QA (#3) — HTML 마크업 + JS 소스를 읽어 기능명세서와 대조.

exercise_web_static(folder): 폴더의 .html(버튼/링크/폼/제목/텍스트) + .js(함수/소스)를 읽어 corpus 구성.
compare_web_to_spec(...)    : corpus ↔ 명세 키워드 대조(eidos_qa_cli.compare_corpus_to_spec 공용).

⚠ 헤드리스 브라우저(JS 렌더·클릭 반응)가 아니라 **정적 마크업 분석**이다 — QtWebEngine 은 GUI 메인스레드
   전용이라 백그라운드 QA 스레드와 충돌. 'HTML에 기능이 존재/노출되나'는 잡지만, 클릭 시 실제 동작은 미확인.
순수(Qt 무관)·헤드리스 테스트 가능.
"""
from __future__ import annotations

import os
from html.parser import HTMLParser
from typing import Any, Dict, List

_SKIP_DIRS = {"__pycache__", ".git", "node_modules", ".venv", "dist", "build", ".eidos_dev_backups"}
_ATTR_KEYS = {"value", "placeholder", "alt", "aria-label", "title", "name"}


class _HTMLText(HTMLParser):
    """가시 텍스트 + 버튼/입력/링크의 라벨성 속성을 수집(script/style 본문 제외)."""
    def __init__(self):
        super().__init__()
        self.parts: List[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        for k, v in attrs:
            if k in _ATTR_KEYS and v and v.strip():
                self.parts.append(v.strip())

    def handle_startendtag(self, tag, attrs):
        for k, v in attrs:
            if k in _ATTR_KEYS and v and v.strip():
                self.parts.append(v.strip())

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data and data.strip():
            self.parts.append(data.strip())


def _html_corpus(path: str) -> str:
    try:
        html = open(path, "r", encoding="utf-8", errors="replace").read()
    except Exception:  # noqa: BLE001
        return ""
    p = _HTMLText()
    try:
        p.feed(html)
    except Exception:  # noqa: BLE001
        pass
    text = " ".join(p.parts)
    # 인라인 <script> 본문도 키워드 단서로 포함(함수명 등)
    import re
    scripts = " ".join(re.findall(r"<script[^>]*>(.*?)</script>", html, re.DOTALL | re.IGNORECASE))
    return text + "\n" + scripts


def find_web_entry(folder: str) -> str:
    """대표 HTML(index.html 우선, 없으면 최상위 가까운 .html)."""
    best = ""
    for dp, dn, fs in os.walk(folder):
        dn[:] = [d for d in dn if d not in _SKIP_DIRS]
        for fn in fs:
            if fn.lower().endswith((".html", ".htm")):
                full = os.path.join(dp, fn)
                if fn.lower() == "index.html":
                    return full
                if not best or len(full) < len(best):
                    best = full
    return best


def exercise_web_static(folder: str, *, budget: int = 80000) -> Dict[str, Any]:
    """폴더의 모든 .html(텍스트/라벨) + .js(소스)를 읽어 corpus 구성."""
    if not os.path.isdir(folder):
        return {"ok": False, "error": f"폴더 없음: {folder}"}
    parts: List[str] = []
    html_files: List[str] = []
    total = 0
    for dp, dn, fs in os.walk(folder):
        dn[:] = [d for d in dn if d not in _SKIP_DIRS]
        for fn in fs:
            low = fn.lower()
            fp = os.path.join(dp, fn)
            if low.endswith((".html", ".htm")):
                html_files.append(os.path.relpath(fp, folder))
                seg = _html_corpus(fp)
            elif low.endswith((".js", ".ts", ".jsx", ".tsx", ".vue", ".svelte")):
                try:
                    seg = open(fp, "r", encoding="utf-8", errors="replace").read()
                except Exception:  # noqa: BLE001
                    seg = ""
            else:
                continue
            if not seg:
                continue
            if total + len(seg) > budget:
                seg = seg[: max(0, budget - total)]
            total += len(seg)
            parts.append(seg)
            if total >= budget:
                break
        if total >= budget:
            break
    if not html_files:
        return {"ok": False, "error": "HTML 파일 없음 — 웹 프로젝트가 아닐 수 있음"}
    return {"ok": True, "corpus": "\n".join(parts), "html_files": html_files}


def compare_web_to_spec(web_result: Dict[str, Any], spec_path: str, project_root: str = "") -> Dict[str, Any]:
    if not (web_result or {}).get("ok"):
        return {"ok": False, "error": web_result.get("error", "웹 분석 결과 없음"), "features": []}
    from eidos_qa_cli import compare_corpus_to_spec
    return compare_corpus_to_spec(web_result.get("corpus", ""), spec_path, project_root)


if __name__ == "__main__":
    import sys
    import io
    import tempfile
    import shutil
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass
    tmp = tempfile.mkdtemp()
    open(os.path.join(tmp, "index.html"), "w", encoding="utf-8").write(
        "<html><head><title>쇼핑몰</title></head><body>"
        "<h1>상품 목록</h1>"
        "<button id='login'>로그인</button>"
        "<input placeholder='검색어'>"
        "<a href='/cart'>장바구니</a>"
        "<script>function checkout(){alert('결제');}</script>"
        "</body></html>")
    open(os.path.join(tmp, "QA_체크리스트.md"), "w", encoding="utf-8").write(
        "# QA\n- [ ] 로그인 login\n- [ ] 장바구니 cart\n- [ ] 결제 checkout\n- [ ] 회원가입 signup\n")
    ex = exercise_web_static(tmp)
    print("HTML:", ex.get("html_files"))
    cmp = compare_web_to_spec(ex, os.path.join(tmp, "QA_체크리스트.md"), project_root=tmp)
    print(f"웹 QA: 총{cmp['spec_total']} 노출{cmp['covered']} 일부{cmp['partial']} 코드만{cmp['code_only']} 미발견{cmp['absent']} 커버{int(cmp['coverage']*100)}%")
    for f in cmp["features"]:
        print("  ", f["status"], f["title"])
    shutil.rmtree(tmp)
    print("✅ eidos_qa_web 셀프테스트 통과 — HTML/JS 마크업 분석 + 명세 대조")
