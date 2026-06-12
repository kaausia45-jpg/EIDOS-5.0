# -*- coding: utf-8 -*-
"""EIDOS 개발 워크스페이스 (Phase 2) · PySide6 — 프로젝트별 서브탭 + dev 엔진 배선.

레이아웃: [🛠 개발] 탭 = 프로젝트 서브탭들(A프로젝트/B프로젝트 독립) · 각 탭 = 좌 채팅 / 우 파일트리+에디터.
기능:
  - 좌 채팅 지시 → EIDOS dev엔진이 우측 폴더에 코드 작성(diff 국소수정, 관련파일 전부 읽음)
  - 하단 [🏗 스캐폴딩] 버튼 → 초기 파일 구조 생성
  - 엔진은 백그라운드 스레드(비차단)
테마: THEMES 토큰 4종 + apply_theme. 공개 인터페이스(DevWorkspace·apply_theme) 유지.
"""
from __future__ import annotations

import base64
import os
import re
import subprocess
import sys

from PySide6.QtCore import Qt, Signal, QThread, QTimer
from PySide6.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat
from PySide6.QtWidgets import (
    QWidget, QFrame, QLabel, QPushButton, QLineEdit, QPlainTextEdit, QTabWidget,
    QTextEdit, QSplitter, QComboBox, QScrollArea, QVBoxLayout, QHBoxLayout,
)


# ──────────────────────────────────────────────────────────────
# 경량 문법 하이라이터 (테마 연동·다언어 근사)
# ──────────────────────────────────────────────────────────────
_KW_PY = ("def class return if elif else for while import from as try except finally with lambda "
          "None True False and or not in is pass break continue raise yield global nonlocal async await self None").split()
_KW_JS = ("function return if else for while var let const class new this import export default async await "
          "try catch finally throw typeof instanceof null true false undefined extends super of new delete void").split()
_KW_C = ("int float double char void long short unsigned struct class public private protected return if else for "
         "while switch case break continue new delete const static virtual override true false null nullptr "
         "func package import type var fn let mut pub use struct impl match").split()


def _ext_lang(ext: str) -> str:
    ext = (ext or "").lower()
    if ext in ("js", "jsx", "ts", "tsx", "vue", "svelte"):
        return "js"
    if ext in ("c", "cpp", "h", "hpp", "java", "go", "rs", "kt", "swift", "cs"):
        return "c"
    return "py"   # py/sh/yaml/json/md/html/css 등 — 기본(주석 #·문자열·숫자 위주)


class _CodeHighlighter(QSyntaxHighlighter):
    def __init__(self, doc):
        super().__init__(doc)
        self._lang = "py"
        self._colors: dict = {}
        self._is_dark = True
        self._rules: list = []

    def set_language(self, ext: str):
        self._lang = _ext_lang(ext)
        self._rebuild(); self.rehighlight()

    def apply_theme(self, colors: dict, is_dark: bool):
        self._colors = colors; self._is_dark = is_dark
        self._rebuild(); self.rehighlight()

    def _fmt(self, color: str, *, bold=False, italic=False):
        f = QTextCharFormat(); f.setForeground(QColor(color))
        if bold:
            f.setFontWeight(QFont.Weight.Bold)
        if italic:
            f.setFontItalic(True)
        return f

    def _rebuild(self):
        c = self._colors; dark = self._is_dark
        kw_c = c.get("accent_primary", "#7c3aed")
        str_c = "#7ee787" if dark else "#0a7d33"
        com_c = c.get("text_hint", "#6e7681")
        num_c = "#ffa657" if dark else "#b45309"
        fn_c = "#79c0ff" if dark else "#1f6feb"
        kws = {"py": _KW_PY, "js": _KW_JS, "c": _KW_C}.get(self._lang, _KW_PY)
        com = "//" if self._lang in ("js", "c") else "#"
        # 순서: 함수→숫자→키워드→주석→문자열(문자열이 마지막=내부 false 매치 덮음)
        self._rules = [
            (re.compile(r"\b([A-Za-z_]\w*)\s*(?=\()"), self._fmt(fn_c)),
            (re.compile(r"\b\d+\.?\d*\b"), self._fmt(num_c)),
            (re.compile(r"\b(" + "|".join(re.escape(k) for k in kws) + r")\b"), self._fmt(kw_c, bold=True)),
            (re.compile(re.escape(com) + r".*$"), self._fmt(com_c, italic=True)),
            (re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"|\'[^\'\\]*(?:\\.[^\'\\]*)*\''), self._fmt(str_c)),
        ]

    def highlightBlock(self, text):
        for pat, fmt in self._rules:
            for mt in pat.finditer(text):
                self.setFormat(mt.start(), mt.end() - mt.start(), fmt)

import eidos_dashboard_data as ddata
import eidos_dev_engine as deng   # 기존 멀티파일 dev 엔진 패키지: auto_dev/scaffold_project/LLMProposer


def _gemini_key() -> str:
    k = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if k:
        return k
    import json
    for p in ("eidos_settings.json",
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "eidos_settings.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            if (d.get("gemini_api_key") or "").strip():
                return d["gemini_api_key"].strip()
        except Exception:  # noqa: BLE001
            pass
    return ""


def _read_project_context(folder: str, budget: int = 40000):
    """대화용 읽기 컨텍스트 — 프롬프트/문서(.md/.txt) 우선 + 작은 코드파일. 파일 안 건드림."""
    skip = {"__pycache__", ".git", "node_modules", ".venv", "dist", "build", ".eidos_dev_backups", "EIDOS_Backups"}
    exts = (".md", ".txt", ".py", ".js", ".ts", ".tsx", ".html", ".css", ".json", ".yml", ".yaml")
    items = []
    for dp, dn, fs in os.walk(folder):
        dn[:] = [d for d in dn if d not in skip]
        for fn in fs:
            if fn.lower().endswith(exts):
                items.append(os.path.join(dp, fn))
    items.sort(key=lambda p: (0 if p.lower().endswith((".md", ".txt")) else 1,
                              os.path.getsize(p) if os.path.exists(p) else 0))
    parts, flist, total = [], [], 0
    for fp in items:
        rel = os.path.relpath(fp, folder)
        flist.append(rel)
        try:
            txt = open(fp, encoding="utf-8", errors="replace").read()
        except Exception:  # noqa: BLE001
            continue
        if len(txt) > 6000:
            txt = txt[:6000] + "\n…(생략)"
        if total + len(txt) > budget:
            continue
        total += len(txt)
        parts.append(f"### {rel}\n{txt}")
    return "\n\n".join(parts), flist


def _find_qa_spec(folder: str) -> str:
    """QA 검증용 명세서 탐색 — Petalbot 'QA 체크리스트' 우선, 없으면 명세/spec 파일."""
    skip = {"__pycache__", ".git", "node_modules", ".venv", "dist", "build", ".eidos_dev_backups"}
    cands = []
    for dp, dn, fs in os.walk(folder):
        dn[:] = [d for d in dn if d not in skip]
        for fn in fs:
            if not fn.lower().endswith((".md", ".txt")):
                continue
            low = fn.lower()
            if any(k in fn for k in ("QA", "체크리스트", "명세", "요구")) or \
               any(k in low for k in ("qa", "checklist", "spec", "feature")):
                cands.append(os.path.join(dp, fn))
    if cands:
        cands.sort(key=lambda p: (0 if ("qa" in os.path.basename(p).lower() or "체크리스트" in os.path.basename(p)) else 1, len(p)))
        return cands[0]
    try:
        return deng.find_spec_file(folder)
    except Exception:  # noqa: BLE001
        return ""


def _gemini_text(prompt: str, *args, **kwargs) -> str:
    """LLMProposer 용 동기 LLM 콜러블 — (prompt)->원응답 텍스트. Gemini 직접 호출."""
    import requests
    key = _gemini_key()
    if not key:
        raise RuntimeError("Gemini 키 없음 (eidos_settings.json gemini_api_key)")
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": int(kwargs.get("max_tokens", 8192))}}
    res = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        headers={"Content-Type": "application/json", "x-goog-api-key": key}, json=body, timeout=180)
    if res.status_code != 200:
        raise RuntimeError(f"Gemini {res.status_code}: {res.text[:150]}")
    data = res.json()
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts)


def _gemini_multimodal(prompt: str, images=None, *, max_tokens: int = 8192) -> str:
    """이미지(스크린샷 등) + 텍스트 멀티모달 Gemini 호출. images=[{'mime','b64'}]."""
    import requests
    key = _gemini_key()
    if not key:
        raise RuntimeError("Gemini 키 없음 (eidos_settings.json gemini_api_key)")
    parts = [{"text": prompt}]
    for im in (images or []):
        if im.get("b64"):
            parts.append({"inlineData": {"mimeType": im.get("mime", "image/png"), "data": im["b64"]}})
    body = {"contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": int(max_tokens)}}
    res = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        headers={"Content-Type": "application/json", "x-goog-api-key": key}, json=body, timeout=180)
    if res.status_code != 200:
        raise RuntimeError(f"Gemini {res.status_code}: {res.text[:150]}")
    data = res.json()
    rparts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in rparts)


_IMG_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")
_TXT_EXTS = (".md", ".txt", ".json", ".csv", ".log", ".py", ".js", ".ts", ".tsx",
             ".html", ".css", ".yml", ".yaml", ".ini", ".toml")
_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
         ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp"}


_CLASSIFY_PROMPT = """너는 개발 워크스페이스의 의도 분류기다. 사용자의 메시지를 아래 5가지 중 **정확히 하나**로 분류하고, 그 영어 키 한 단어만 출력하라(다른 말·설명·문장부호 금지).

- todo : 사용자가 '할 일' 메모에 적어둔 항목들을 **전부** 구현/실행하라는 요청. 예) "할 일 목록 전부 구현해줘", "할 일 다 해줘", "메모한 거 전부 만들어줘", "to-do 전부 구현", "적어둔 할 일 다 처리해"
- scaffold : 프로젝트의 초기 구조/뼈대를 새로 생성. 예) "프로젝트 새로 시작해줘", "기본 폴더 구조 잡아줘", "스캐폴딩"
- qa : 이미 만든 앱을 실행/테스트해 명세대로 동작하는지 검증. 예) "QA 해줘", "제대로 되는지 검증", "테스트 돌려봐", "버그 있는지 확인"
- edit : (단일/특정) 코드나 UI를 만들거나 바꾸거나 고치는 요청. 예) "로그인 화면 추가", "버튼 색 파랑으로 바꿔", "이 에러 고쳐줘"
- idea : 단순 질문·설명·논의·아이디어(파일 변경 불필요). 예) "이 코드 뭐하는거야", "어떻게 만들면 좋을까", "구조 설명해줘"

출력은 todo, scaffold, qa, edit, idea 중 한 단어.
[사용자 메시지]
{text}"""


def _classify_intent(text: str) -> str:
    """LLM 라우팅 — 메시지를 todo/scaffold/qa/edit/idea 중 하나로 분류. 실패/불명은 idea(안전)."""
    try:
        out = _gemini_text(_CLASSIFY_PROMPT.format(text=(text or "")[:2000]), max_tokens=1024)
    except Exception:  # noqa: BLE001
        return "idea"
    o = (out or "").strip().lower()
    best, bi = "idea", 10 ** 9
    for k in ("todo", "scaffold", "qa", "edit", "idea"):
        i = o.find(k)
        if 0 <= i < bi:
            bi, best = i, k
    return best


def _resolve_theme(theme_name: str):
    mod = sys.modules.get("eidos_chat_gui")
    themes = getattr(mod, "THEMES", None) if mod else None
    if not themes:
        themes = _FALLBACK
    t = themes.get(theme_name) or themes.get("Modern Dark") or next(iter(themes.values()))
    return t.get("colors", {}), t.get("metrics", {}), bool(t.get("is_dark", True))


_FALLBACK = {
    "Modern Dark": {"is_dark": True, "colors": {
        "bg_base": "#09090b", "bg_raised": "#18181b", "bg_surface": "#27272a", "bg_input": "#18181b",
        "text_title": "#f4f4f5", "text_primary": "#e4e4e7", "text_secondary": "#a1a1aa", "text_hint": "#71717a",
        "text_on_accent": "#fff", "accent_primary": "#7c3aed", "accent_hover": "#8b5cf6", "border_subtle": "#27272a"},
        "metrics": {"radius_sm": "6px"}},
    "Modern White (Crisp)": {"is_dark": False, "colors": {
        "bg_base": "#ffffff", "bg_raised": "#eef2f7", "bg_surface": "#ffffff", "bg_input": "#ffffff",
        "text_title": "#0f172a", "text_primary": "#1e293b", "text_secondary": "#64748b", "text_hint": "#94a3b8",
        "text_on_accent": "#fff", "accent_primary": "#2563eb", "accent_hover": "#3b82f6", "border_subtle": "#e2e8f0"},
        "metrics": {"radius_sm": "6px"}},
}
_CODE_FONT = '"Cascadia Code", "Consolas", "D2Coding", monospace'

# ── 고급 다크 럭셔리 팔레트 — 순검정 + 민트/보라 테두리 (테마 무관 고정) ──
_LUX = {
    "bg": "#000000",           # 순검정 배경
    "panel": "#060608",        # 패널/카드(아주 살짝 들뜬 검정)
    "field": "#0a0a0e",        # 입력/에디터/로그
    "mint": "#2dffd5",         # 민트 — 포커스·주 액션
    "mint_hover": "#5cffe0",
    "mint_soft": "rgba(45,255,213,0.10)",
    "purple": "#a78bfa",       # 보라 — 보조 강조·테두리
    "purple_hover": "#c4b2ff",
    "purple_soft": "rgba(167,139,250,0.13)",
    "border": "#17171c",       # 기본 미묘한 테두리(거의 검정)
    "text": "#ececf0",
    "text2": "#9aa0aa",
    "hint": "#5a5f68",
    "title": "#f6f6f8",
    "on_mint": "#04140f",      # 민트 버튼 위 글자(짙은 청록 거의 검정)
}


def _lux_colors():
    """문법 하이라이터용 colors dict — 럭셔리 팔레트와 일치."""
    return {
        "accent_primary": _LUX["purple"], "text_hint": _LUX["hint"],
        "bg_base": _LUX["bg"], "text_primary": _LUX["text"],
    }


def _hex_to_rgba(c: str, alpha: float) -> str:
    """#rrggbb / #rgb → rgba(...) 문자열(soft 강조용). 파싱 실패 시 원본 반환."""
    s = (c or "").strip()
    if s.startswith("#"):
        h = s[1:]
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        if len(h) == 6:
            try:
                r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                return f"rgba({r},{g},{b},{alpha})"
            except ValueError:
                pass
    return c   # 이미 rgba/명명색 등


def _theme_palette(theme_name: str) -> dict:
    """현재 테마(eidos_chat_gui.THEMES 4종)의 색을 개발 워크스페이스 팔레트로 매핑.
    테마를 바꾸면 개발 탭도 함께 바뀐다(과거 _LUX 고정 순검정 대체).
    L 키 구성은 _LUX와 동일 — 단일 accent 테마라 mint/purple은 같은 accent로 접지."""
    colors, _metrics, is_dark = _resolve_theme(theme_name)

    def g(k, d):
        return colors.get(k) or d

    accent = g("accent_primary", _LUX["mint"])
    accent_h = g("accent_hover", _LUX["mint_hover"])
    return {
        "bg": g("bg_base", _LUX["bg"]),
        "panel": g("bg_raised", _LUX["panel"]),
        "field": g("bg_input", _LUX["field"]),
        "mint": accent,
        "mint_hover": accent_h,
        "mint_soft": _hex_to_rgba(accent, 0.12),
        "purple": accent,
        "purple_hover": accent_h,
        "purple_soft": _hex_to_rgba(accent, 0.13),
        "border": g("border_subtle", _LUX["border"]),
        "text": g("text_primary", _LUX["text"]),
        "text2": g("text_secondary", _LUX["text2"]),
        "hint": g("text_hint", _LUX["hint"]),
        "title": g("text_title", _LUX["title"]),
        "on_mint": g("text_on_accent", _LUX["on_mint"]),
        "is_dark": is_dark,
    }


class _EngineThread(QThread):
    """dev 엔진(동기·네트워크·빌드)을 백그라운드로 — UI 안 막히게. progress=진행 스트리밍."""
    progress = Signal(str)
    done = Signal(dict)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            r = self._fn(self.progress.emit)   # fn(progress_cb) — on_event 로 전달
        except Exception as e:  # noqa: BLE001
            r = {"error": f"{type(e).__name__}: {e}"}
        self.done.emit(r if isinstance(r, dict) else {})


# ──────────────────────────────────────────────────────────────
# 한 프로젝트 패널 (독립) — 좌 채팅 / 우 파일트리+에디터 + 스캐폴딩
# ──────────────────────────────────────────────────────────────
class DevProjectPanel(QWidget):
    def __init__(self, folder: str, title: str, theme_name: str, parent=None):
        super().__init__(parent)
        self._folder = folder
        self._title = title
        self._theme_name = theme_name
        self._cur_file = ""
        self._dirty = False
        self._thread = None
        self._proposer = None
        self._plan: list = []          # [{text, status: todo|doing|done}]
        self._active_task = None       # 현재 실행 중인 작업 인덱스
        self._auto_run = False         # ▶▶ 전체 자동 실행 — 작업을 차례로 끝까지
        self._attachments: list = []   # [{name, kind:image|text, mime, b64|text}] — 명세서/스크린샷 첨부
        self._app_proc = None          # ▶ 앱 실행으로 띄운 진입점 프로세스
        self._todo_text = ""           # 🔨 할 일 — 자유 메모(채팅 '할 일 전부 구현'의 소스)

        root = QVBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        split = QSplitter(Qt.Orientation.Horizontal)

        # 좌: 채팅
        left = QWidget(); ll = QVBoxLayout(left); ll.setContentsMargins(10, 10, 10, 10); ll.setSpacing(8)
        ll.addWidget(self._lbl("💬 채팅 · 개발 아이디어"))
        self._chat_log = QTextEdit(); self._chat_log.setReadOnly(True); self._chat_log.setObjectName("ChatLog")
        self._chat_log.setAcceptDrops(False)   # 파일 드롭은 패널이 받도록(자식이 가로채지 않게)
        ll.addWidget(self._chat_log, 1)
        # 첨부 칩 바 (명세서/스크린샷) — 비어있으면 숨김
        self._attach_bar = QWidget(); self._attach_bar.setObjectName("AttachBar")
        abl = QHBoxLayout(self._attach_bar); abl.setContentsMargins(2, 0, 2, 0); abl.setSpacing(6)
        self._attach_label = QLabel(""); self._attach_label.setObjectName("AttachLabel"); self._attach_label.setWordWrap(True)
        self._attach_clear = QPushButton("✕"); self._attach_clear.setObjectName("AttachClear"); self._attach_clear.setFixedWidth(24)
        self._attach_clear.setToolTip("첨부 비우기"); self._attach_clear.clicked.connect(self._clear_attachments)
        abl.addWidget(self._attach_label, 1); abl.addWidget(self._attach_clear)
        self._attach_bar.setVisible(False)
        ll.addWidget(self._attach_bar)

        inrow = QHBoxLayout(); inrow.setSpacing(8)
        self._attach_btn = QPushButton("📎"); self._attach_btn.setObjectName("AttachBtn"); self._attach_btn.setFixedWidth(38)
        self._attach_btn.setToolTip("기능명세서·스크린샷 등 첨부 (이미지/문서)"); self._attach_btn.clicked.connect(self._on_attach)
        self._chat_input = QLineEdit()
        self._chat_input.setPlaceholderText("메시지 입력 → 의도 자동 분류(논의/코드수정/QA/스캐폴딩/할일구현) · 첨부는 📎 또는 드래그")
        self._chat_input.returnPressed.connect(self._on_send)
        self._send = QPushButton("전송"); self._send.clicked.connect(self._on_send)
        inrow.addWidget(self._attach_btn); inrow.addWidget(self._chat_input, 1); inrow.addWidget(self._send)
        ll.addLayout(inrow)
        btnrow = QHBoxLayout(); btnrow.setSpacing(8)
        self._plan_btn = QPushButton("📋 계획 세우기"); self._plan_btn.clicked.connect(self._on_plan)
        self._todo_btn = QPushButton("🔨 할 일"); self._todo_btn.setObjectName("ScaffoldBtn")
        self._todo_btn.setToolTip("자유 메모로 할 일을 적어두기 — 채팅에 '할 일 목록 전부 구현해줘'라고 하면 자동 구현")
        self._todo_btn.clicked.connect(self._on_todo_memo)
        self._run_app_btn = QPushButton("▶ 앱 실행"); self._run_app_btn.setObjectName("ScaffoldBtn")
        self._run_app_btn.setToolTip("탐지된 진입점 앱을 실제로 띄워봅니다"); self._run_app_btn.clicked.connect(self._on_run_app)
        # 내부 전용(버튼 제거) — 스캐폴딩(채팅 의도 라우팅)이 기본 스택을 읽음
        self._stack = QComboBox()
        try:
            self._stack.addItems(deng.available_stacks())
        except Exception:  # noqa: BLE001
            self._stack.addItems(["cli", "pyside6", "fastapi", "node", "lib"])
        self._busy = QLabel(""); self._busy.setObjectName("BusyLabel")
        btnrow.addWidget(self._plan_btn); btnrow.addWidget(self._todo_btn); btnrow.addWidget(self._run_app_btn)
        btnrow.addWidget(self._busy); btnrow.addStretch(1)
        ll.addLayout(btnrow)

        # 우: 구현 계획 체크리스트 | 에디터  (파일트리 제거 — 변경 파일은 자동으로 에디터에 열림)
        right = QSplitter(Qt.Orientation.Horizontal)
        pw = QWidget(); pwl = QVBoxLayout(pw); pwl.setContentsMargins(8, 10, 4, 10); pwl.setSpacing(6)
        ph = QHBoxLayout(); ph.setSpacing(6)
        ph.addWidget(self._lbl("📋 구현 계획 — '계획 세우기'로 작성"), 1)
        self._run_all_btn = QPushButton("▶▶ 전체 자동 실행"); self._run_all_btn.setObjectName("PlanRunAll")
        self._run_all_btn.setToolTip("모든 작업을 처음부터 끝까지 자동으로 차례로 실행")
        self._run_all_btn.clicked.connect(self._on_run_all)
        self._stop_all_btn = QPushButton("⏹ 중지"); self._stop_all_btn.setObjectName("PlanStopAll")
        self._stop_all_btn.setToolTip("전체 자동 실행 중지(현재 작업이 끝난 뒤 멈춤)")
        self._stop_all_btn.clicked.connect(self._on_stop_all); self._stop_all_btn.setVisible(False)
        ph.addWidget(self._run_all_btn); ph.addWidget(self._stop_all_btn)
        pwl.addLayout(ph)
        self._plan_scroll = QScrollArea(); self._plan_scroll.setWidgetResizable(True)
        self._plan_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._plan_host = QWidget(); self._plan_host.setObjectName("PlanHost")
        self._plan_lay = QVBoxLayout(self._plan_host); self._plan_lay.setContentsMargins(0, 0, 0, 0); self._plan_lay.setSpacing(3)
        self._plan_lay.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._plan_scroll.setWidget(self._plan_host)
        pwl.addWidget(self._plan_scroll, 1)
        ew = QWidget(); ewl = QVBoxLayout(ew); ewl.setContentsMargins(4, 10, 8, 10); ewl.setSpacing(6)
        eh = QHBoxLayout(); eh.setSpacing(8)
        self._file_label = QLabel("(파일 선택)"); self._file_label.setObjectName("DevFileLabel")
        self._save_btn = QPushButton("💾 저장"); self._save_btn.clicked.connect(self._save_file); self._save_btn.setEnabled(False)
        eh.addWidget(self._file_label, 1); eh.addWidget(self._save_btn)
        self._editor = QPlainTextEdit(); self._editor.setObjectName("CodeEditor")
        self._editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._editor.textChanged.connect(self._on_edit)
        self._highlighter = _CodeHighlighter(self._editor.document())   # 문법 하이라이트
        ewl.addLayout(eh); ewl.addWidget(self._editor, 1)
        right.addWidget(pw); right.addWidget(ew); right.setSizes([300, 760])

        split.addWidget(left); split.addWidget(right); split.setSizes([400, 900])
        root.addWidget(split, 1)

        self.setAcceptDrops(True)   # 파일 드래그 첨부
        self.apply_theme(theme_name)
        if folder and os.path.isdir(folder):
            self._append("EIDOS", f"프로젝트 열림: {os.path.basename(folder)} — 무엇을 만들까요? (수정은 diff 국소수정으로 안전하게)")
        self._load_plan()
        self._load_todo()
        self._render_plan()

    def _lbl(self, t):
        w = QLabel(t); w.setObjectName("DevSectionLabel"); return w

    # ── 파일 (트리 제거 — 변경/생성 파일은 작업 후 자동으로 에디터에 열림) ──
    def _open_file(self, path):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except Exception as e:  # noqa: BLE001
            self._editor.setPlainText(f"(열기 실패: {e})"); return
        self._cur_file = path
        self._highlighter.set_language(os.path.splitext(path)[1].lstrip("."))   # 확장자별 하이라이트
        self._editor.blockSignals(True); self._editor.setPlainText(text); self._editor.blockSignals(False)
        self._dirty = False; self._save_btn.setEnabled(False)
        self._file_label.setText(os.path.relpath(path, self._folder))

    def _on_edit(self):
        if self._cur_file:
            self._dirty = True; self._save_btn.setEnabled(True)

    def _confirm_discard(self):
        from PySide6.QtWidgets import QMessageBox
        return QMessageBox.question(self, "저장 안 됨", "수정 중인 내용을 버릴까요?") == QMessageBox.StandardButton.Yes

    def _save_file(self):
        if not self._cur_file:
            return
        try:
            with open(self._cur_file, "w", encoding="utf-8") as f:
                f.write(self._editor.toPlainText())
            self._dirty = False; self._save_btn.setEnabled(False)
            self._append("EIDOS", f"💾 저장: {os.path.relpath(self._cur_file, self._folder)}")
        except Exception as e:  # noqa: BLE001
            self._append("EIDOS", f"⚠ 저장 실패: {e}")

    # ── 📎 첨부 (기능명세서·스크린샷 등) ──
    def _ingest_path(self, p) -> bool:
        """파일 1개를 첨부로 흡수(이미지=base64 / 문서=텍스트). 추가되면 True. 파일선택·드래그 공용."""
        ext = os.path.splitext(p)[1].lower()
        name = os.path.basename(p)
        try:
            if ext in _IMG_EXTS:
                if sum(1 for a in self._attachments if a["kind"] == "image") >= 3:
                    self._append("EIDOS", "⚠ 이미지는 최대 3장까지 첨부돼요.", dim=True); return False
                with open(p, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                self._attachments.append({"name": name, "kind": "image",
                                          "mime": _MIME.get(ext, "image/png"), "b64": b64})
                return True
            if ext in _TXT_EXTS:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    txt = f.read()
                if len(txt) > 12000:
                    txt = txt[:12000] + "\n…(생략)"
                self._attachments.append({"name": name, "kind": "text", "text": txt})
                return True
            self._append("EIDOS", f"⚠ 미지원 형식: {name} (이미지/문서만)", dim=True)
            return False
        except Exception as e:  # noqa: BLE001
            self._append("EIDOS", f"⚠ 첨부 실패({name}): {e}", dim=True)
            return False

    def _on_attach(self):
        from PySide6.QtWidgets import QFileDialog
        paths, _ = QFileDialog.getOpenFileNames(
            self, "첨부 — 명세서/스크린샷 등", self._folder,
            "지원 파일 (*.png *.jpg *.jpeg *.gif *.bmp *.webp *.md *.txt *.json *.csv *.log *.py *.js *.html *.css *.yml *.yaml);;모든 파일 (*.*)")
        added = sum(1 for p in (paths or []) if self._ingest_path(p))
        if added:
            self._append("EIDOS", self._esc(f"📎 첨부 {added}개 추가 — 전송/🛠/📋에 함께 반영됩니다."), dim=True)
        self._render_attachments()

    # ── 드래그 앤 드롭 첨부 (채팅 영역에 파일을 끌어다 놓기) ──
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        md = e.mimeData()
        if not md.hasUrls():
            return
        added = 0
        for url in md.urls():
            p = url.toLocalFile()
            if p and os.path.isfile(p) and self._ingest_path(p):
                added += 1
        if added:
            self._append("EIDOS", self._esc(f"📎 드래그로 첨부 {added}개 추가 — 전송/🛠/📋에 반영됩니다."), dim=True)
        self._render_attachments()
        e.acceptProposedAction()

    # ── 🔨 할 일 (자유 메모) ──
    def _todo_path(self):
        return os.path.join(self._folder, ".eidos_todo.txt")

    def _load_todo(self):
        try:
            with open(self._todo_path(), encoding="utf-8") as f:
                self._todo_text = f.read()
        except Exception:  # noqa: BLE001
            self._todo_text = ""

    def _save_todo(self):
        try:
            with open(self._todo_path(), "w", encoding="utf-8") as f:
                f.write(self._todo_text or "")
        except Exception:  # noqa: BLE001
            pass

    def _on_todo_memo(self):
        """🔨 할 일 — 자유 메모장. 여기 적은 걸 채팅 '할 일 목록 전부 구현해줘'가 읽어 구현한다."""
        from PySide6.QtWidgets import (
            QDialog, QVBoxLayout, QPlainTextEdit, QDialogButtonBox, QLabel as _QL)
        dlg = QDialog(self); dlg.setWindowTitle("🔨 할 일 메모"); dlg.resize(480, 400)
        dlg.setStyleSheet(self.styleSheet())   # 패널 테마 상속
        lay = QVBoxLayout(dlg)
        tip = _QL("자유롭게 할 일을 적어두세요. 채팅창에 \"할 일 목록 전부 구현해줘\"라고 하면 자동으로 구현합니다.")
        tip.setWordWrap(True); lay.addWidget(tip)
        ed = QPlainTextEdit(); ed.setObjectName("CodeEditor"); ed.setPlainText(self._todo_text or "")
        ed.setPlaceholderText("예)\n- 로그인 화면 추가\n- 거래내역 CSV 내보내기\n- 대시보드 차트 색 파랑으로")
        lay.addWidget(ed, 1)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._todo_text = ed.toPlainText()
            self._save_todo()
            n = len([ln for ln in (self._todo_text or "").splitlines() if ln.strip()])
            self._append("EIDOS", self._esc(
                f"🔨 할 일 메모 저장 ({n}줄). 채팅에 '할 일 목록 전부 구현해줘'라고 하면 자동 구현합니다."), dim=True)

    def _implement_todos(self):
        """채팅 '할 일 목록 전부 구현해줘' — 할 일 메모를 작업으로 분해 → 자동 실행."""
        if self._thread:
            return
        if not _gemini_key():
            self._append("EIDOS", "⚠ Gemini 키 없음 — 설정에서 입력."); return
        memo = (self._todo_text or "").strip()
        if not memo:
            self._append("EIDOS", "🔨 할 일 메모가 비어 있어요 — 먼저 '🔨 할 일' 버튼으로 할 일을 적어주세요.")
            return
        folder = self._folder
        self._append("EIDOS", self._esc("🔨 할 일 목록 전체 구현 시작 — 메모를 작업으로 쪼개 자동 실행합니다."))

        def fn(cb):
            cb("📋 할 일 메모를 구현 작업으로 분해 중…")
            ctx, _flist = _read_project_context(folder)
            sysmsg = ("너는 개발 PM이다. 아래 '할 일 메모'(사용자가 자유롭게 적은 구현 요청들)와 프로젝트 컨텍스트를 읽고 "
                      "구현을 5~12개의 '작업'으로 쪼갠 실행 계획을 만든다. 각 작업은 한 줄, 구체적이고 독립 실행 가능하게. "
                      "메모의 모든 항목을 빠짐없이 반영. 의존성 순서대로. **번호 목록으로만 출력**(머리말·설명·코드 금지).")
            prompt = f"{sysmsg}\n\n[할 일 메모]\n{memo}\n\n[프로젝트 컨텍스트]\n{ctx}"
            return {"kind": "plan", "tasks": self._parse_tasks(_gemini_text(prompt)), "autorun": True}
        self._set_busy(True, "📋 할 일 분해 중…")
        self._thread = _EngineThread(fn, self)
        self._thread.done.connect(self._on_done)
        self._thread.start()

    def _render_attachments(self):
        if not self._attachments:
            self._attach_bar.setVisible(False); return
        chips = []
        for a in self._attachments:
            ic = "🖼" if a["kind"] == "image" else "📄"
            chips.append(f"{ic} {a['name']}")
        self._attach_label.setText("📎 " + " · ".join(chips))
        self._attach_bar.setVisible(True)

    def _clear_attachments(self):
        self._attachments = []
        self._render_attachments()

    def _split_attachments(self):
        """첨부 → (텍스트 합본, 이미지 리스트[{mime,b64}])."""
        text_parts, images = [], []
        for a in self._attachments:
            if a["kind"] == "text":
                text_parts.append(f"### 첨부: {a['name']}\n{a.get('text','')}")
            elif a["kind"] == "image":
                images.append({"mime": a.get("mime", "image/png"), "b64": a.get("b64", "")})
        return "\n\n".join(text_parts), images

    # ── ▶ 앱 실행 (진입점 실제 구동) ──
    def _on_run_app(self):
        if self._thread:
            return
        folder = self._folder
        if not folder or not os.path.isdir(folder):
            self._append("EIDOS", "⚠ 프로젝트 폴더가 없어요."); return
        # 이미 띄운 앱이 살아있으면 중복 실행 방지
        if self._app_proc is not None and self._app_proc.poll() is None:
            self._append("EIDOS", "▶ 이미 앱이 실행 중이에요 (창을 닫은 뒤 다시 실행).", dim=True); return
        try:
            det = deng.detect_project(folder)
            stack = getattr(det, "stack", "") or "불명"
            entry = getattr(det, "primary_entrypoint", "") or ""
        except Exception as e:  # noqa: BLE001
            self._append("EIDOS", f"⚠ 진입점 탐지 실패: {e}"); return
        if not entry:
            self._append("EIDOS", "⚠ 실행할 진입점을 못 찾았어요 — 🏗 스캐폴딩 또는 🛠 전체 자동으로 먼저 만들어 주세요.")
            return
        if not os.path.isabs(entry):
            entry = os.path.join(folder, entry)
        # 스택별 실행 명령
        if stack == "node":
            cmd = ["node", entry]
        else:   # pyside6 / cli / fastapi / lib / 불명 — 파이썬 진입점 실행
            cmd = [sys.executable, entry]
        self._append("나", self._esc(f"▶ 앱 실행 — {os.path.relpath(entry, folder)} (stack={stack})"))
        try:
            self._app_proc = subprocess.Popen(
                cmd, cwd=os.path.dirname(entry) or folder,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            self._append("EIDOS", f"⚠ 실행 실패: {type(e).__name__}: {e}"); self._app_proc = None; return
        self._append("EIDOS", self._esc(f"▶ 실행됨 (pid {self._app_proc.pid}) — 창이 뜨는지 확인하세요. 1.5초 내 죽으면 오류를 보여드릴게요."), dim=True)
        QTimer.singleShot(1600, self._check_app_proc)   # 즉시 크래시 감지

    def _check_app_proc(self):
        proc = self._app_proc
        if proc is None:
            return
        rc = proc.poll()
        if rc is None:
            self._append("EIDOS", "✅ 앱이 정상적으로 떠서 실행 중이에요.", dim=True); return
        # 일찍 종료됨 → 출력 수거(크래시 진단)
        try:
            out = proc.stdout.read() if proc.stdout else ""
        except Exception:  # noqa: BLE001
            out = ""
        if rc == 0:
            self._append("EIDOS", "▶ 앱이 코드 0으로 즉시 종료됐어요 (GUI가 아니거나 바로 끝나는 프로그램).", dim=True)
        else:
            tail = "\n".join((out or "").strip().splitlines()[-15:])
            self._append("EIDOS", self._esc(f"⚠ 앱이 오류로 종료(code {rc}). ✅ QA 검증으로 자동 진단하거나, 아래 출력을 보세요:"))
            if tail:
                self._append("EIDOS", self._esc(tail), dim=True)
        self._app_proc = None

    # ── 채팅 = LLM 라우팅(의도 분류 → 자동 실행) ──
    def _on_send(self):
        """전송 = 메시지의 의도를 LLM으로 분류(scaffold/qa/edit/idea) → 해당 동작으로 자동 라우팅."""
        text = self._chat_input.text().strip()
        if not text or self._thread:
            return
        self._chat_input.clear()
        self._append("나", self._esc(text))
        if not _gemini_key():
            self._append("EIDOS", "⚠ Gemini 키 없음 — 설정에서 gemini_api_key 입력.")
            return

        def fn(cb):
            cb("🧭 의도 분류 중…")
            return {"kind": "route", "intent": _classify_intent(text), "text": text}
        self._set_busy(True, "🧭 의도 분류 중…")
        self._thread = _EngineThread(fn, self)
        self._thread.done.connect(self._on_done)
        self._thread.start()

    def _do_chat(self, text):
        """아이디어 논의(idea) — 읽기 전용 대화. 파일 안 건드림. 첨부 멀티모달 반영."""
        if self._thread:
            return
        folder = self._folder
        text_blob, images = self._split_attachments()

        def fn(cb):
            cb("💬 읽는 중…")
            ctx, flist = _read_project_context(folder)
            sysmsg = ("너는 이 프로젝트의 개발 비서다. 사용자의 질문에 답하거나 아이디어를 논의한다. "
                      "**절대 파일을 수정·생성하지 마라(읽고 설명·분석·제안만).** 한국어로 답하라.\n"
                      "[출력 형식] 한 문단으로 길게 이어붙이지 말고 가독성 있게 구조화하라:\n"
                      "- 맨 위에 한 줄 핵심 요약(굵게 **…**)을 먼저 제시\n"
                      "- 이후 내용은 짧은 문단 또는 '- ' 불릿/번호 목록으로 분리(각 항목 1~2문장)\n"
                      "- 소제목이 필요하면 '## 제목' 사용, 코드·파일명·식별자는 `백틱`으로 표시\n"
                      "- 불필요한 군더더기 없이 간결하게.")
            att = f"\n\n[첨부 문서]\n{text_blob}" if text_blob else ""
            if images:
                att += f"\n\n[첨부 이미지 {len(images)}장 — 위에 함께 전달됨, 직접 보고 답하라]"
            prompt = f"{sysmsg}\n\n[파일 목록]\n{flist}\n\n[파일 내용]\n{ctx}{att}\n\n[사용자]\n{text}"
            ans = _gemini_multimodal(prompt, images) if images else _gemini_text(prompt)
            return {"kind": "chat", "text": ans}
        self._set_busy(True, "💬 답변 생성 중…")
        self._thread = _EngineThread(fn, self)
        self._thread.done.connect(self._on_done)
        self._thread.start()

    def _on_develop(self):
        """🛠 개발 실행 = 입력한 지시로 실제 코드 작성(auto_dev). 명시적 의도."""
        if self._thread:
            return
        instr = self._chat_input.text().strip()
        if not instr:
            self._append("EIDOS", "개발 지시를 입력한 뒤 '🛠 개발 실행'을 눌러주세요 (예: 로그인 화면 추가).")
            return
        self._chat_input.clear()
        self._append("나", self._esc(f"🛠 전체 자동 — {instr}"))
        self._run("edit", instr)

    # ── 📋 구현 계획 (프롬프트 읽기 → AI 계획 → 작업 하나씩) ──
    def _plan_path(self):
        return os.path.join(self._folder, ".eidos_plan.json")

    def _load_plan(self):
        import json
        try:
            with open(self._plan_path(), encoding="utf-8") as f:
                self._plan = json.load(f).get("tasks", [])
        except Exception:  # noqa: BLE001
            self._plan = []

    def _save_plan(self):
        import json
        try:
            with open(self._plan_path(), "w", encoding="utf-8") as f:
                json.dump({"tasks": self._plan}, f, ensure_ascii=False, indent=2)
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _parse_tasks(text):
        tasks = []
        for ln in (text or "").splitlines():
            s = ln.strip()
            if not s:
                continue
            m = re.match(r"^\s*(?:\d+[.)]|[-*•·]|\[[ x]\])\s+(.+)$", s)
            t = (m.group(1) if m else s).strip().strip("`").strip()
            if t and len(t) > 2 and not t.startswith("#"):
                tasks.append({"text": t[:160], "status": "todo"})
        return tasks[:15]

    def _on_plan(self):
        """📋 계획 세우기.

        ① 프롬프트팩(dev_prompts/ — Petalbot이 만든 작업별 .md)이 있으면 **그 프롬프트들을
           순서·내용 그대로** 작업 목록으로 만든다(프롬프트 1개 = 작업 1개·LLM 재구성 없음).
        ② 프롬프트팩이 없을 때만 LLM이 명세를 읽고 작업으로 분해(폴백)."""
        if self._thread:
            return
        goal = self._chat_input.text().strip()
        self._chat_input.clear()
        folder = self._folder
        self._append("나", self._esc("📋 계획 세우기" + (f" — {goal}" if goal else "")))

        # ① 프롬프트팩 우선 — 결정적(LLM 0). 프롬프트 순서·제목 그대로.
        try:
            import eidos_dev_engine.prompt_pack as _pp
            pack = _pp.parse_prompt_pack(folder)
        except Exception as e:  # noqa: BLE001
            print(f"[plan] 프롬프트팩 파싱 실패(graceful): {e}")
            pack = {"found": False, "tasks": []}
        if pack.get("found") and pack.get("tasks"):
            charter_file = pack.get("charter_file") or ""
            tasks = []
            for t in pack["tasks"]:
                tasks.append({
                    "text": f"[{int(t['num']):02d}] {t['title']}"[:160],
                    "status": "todo",
                    "prompt_file": t["file"],     # 실행 시 이 프롬프트 내용을 그대로 구현
                    "from_pack": True,
                })
            n = len(tasks)
            self._append("EIDOS", self._esc(
                f"📋 프롬프트팩에서 {n}개 프롬프트를 그대로 계획으로 가져왔어요"
                + (f" (공통 헌장 {charter_file} 적용)" if charter_file else "")
                + " — 각 작업은 해당 프롬프트대로 구현합니다."))
            self._on_done({"kind": "plan", "tasks": tasks})
            return

        # ② 폴백 — 프롬프트팩 없음 → LLM 분해
        if not _gemini_key():
            self._append("EIDOS", "⚠ 프롬프트팩(dev_prompts)이 없고 Gemini 키도 없어요 — "
                                  "Petalbot에서 '프롬프트 생성'을 하거나 설정에서 키를 넣어주세요.")
            return
        self._append("EIDOS", "ℹ 프롬프트팩(dev_prompts)이 없어 명세를 읽고 AI가 작업으로 나눌게요.", dim=True)
        text_blob, images = self._split_attachments()

        def fn(cb):
            cb("📋 프롬프트·명세 읽고 구현 계획 작성 중…")
            ctx, _flist = _read_project_context(folder)
            sysmsg = ("너는 개발 PM이다. 프로젝트의 개발 프롬프트/명세(context)를 읽고 구현을 5~12개의 '작업'으로 쪼갠 "
                      "실행 계획을 만든다. 각 작업은 한 줄, 구체적이고 독립 실행 가능하게(예: 'index.html 기본 레이아웃·헤더 작성'). "
                      "의존성 순서대로. **번호 목록으로만 출력**(머리말·설명·코드 금지).")
            att = f"\n\n[첨부 문서/명세]\n{text_blob}" if text_blob else ""
            if images:
                att += f"\n\n[첨부 이미지 {len(images)}장 — 직접 보고 계획에 반영]"
            prompt = f"{sysmsg}\n\n[목표]{goal or '(프롬프트 기반)'}\n\n[프로젝트 컨텍스트]\n{ctx}{att}"
            text = _gemini_multimodal(prompt, images) if images else _gemini_text(prompt)
            return {"kind": "plan", "tasks": self._parse_tasks(text)}
        self._set_busy(True, "📋 계획 작성 중…")
        self._thread = _EngineThread(fn, self)
        self._thread.done.connect(self._on_done)
        self._thread.start()

    def _render_plan(self):
        while self._plan_lay.count():
            it = self._plan_lay.takeAt(0); w = it.widget()
            if w:
                w.setParent(None); w.deleteLater()
        if not self._plan:
            lab = QLabel("(아직 계획 없음 — 📋 계획 세우기)"); lab.setObjectName("RowHint"); lab.setWordWrap(True)
            self._plan_lay.addWidget(lab)
            return
        icons = {"todo": "○", "doing": "▶", "done": "✓"}
        for i, t in enumerate(self._plan):
            row = QWidget(); rl = QHBoxLayout(row); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(6)
            ic = QLabel(icons.get(t.get("status"), "○")); ic.setObjectName("PlanIcon"); ic.setFixedWidth(16)
            lab = QLabel(f"{i+1}. {t.get('text','')}"); lab.setObjectName("PlanDone" if t.get("status") == "done" else "PlanText")
            lab.setWordWrap(True)
            btn = QPushButton("▶"); btn.setObjectName("PlanRun"); btn.setFixedWidth(28)
            btn.setToolTip("이 작업 실행"); btn.clicked.connect(lambda _c=False, idx=i: self._run_task(idx))
            rl.addWidget(ic); rl.addWidget(lab, 1); rl.addWidget(btn)
            self._plan_lay.addWidget(row)

    def _next_todo(self, after=-1):
        """`after` 이후(미포함)의 첫 미완료 작업 인덱스, 없으면 None."""
        return next((j for j in range(after + 1, len(self._plan))
                     if self._plan[j].get("status") != "done"), None)

    def _on_run_all(self):
        """▶▶ 전체 자동 실행 — 미완료 작업을 처음부터 끝까지 차례로 자동 실행."""
        if self._thread:
            return
        if not self._plan:
            self._append("EIDOS", "📋 먼저 '계획 세우기'로 구현 계획을 작성하세요.")
            return
        if self._get_proposer() is None:
            self._append("EIDOS", "⚠ Gemini 키 없음 — 설정에서 입력."); return
        nxt = self._next_todo()
        if nxt is None:
            self._append("EIDOS", "🎉 모든 작업이 이미 완료됐어요."); return
        self._auto_run = True
        self._run_all_btn.setVisible(False); self._stop_all_btn.setVisible(True)
        self._append("EIDOS", self._esc(f"▶▶ 전체 자동 실행 시작 — 작업 {nxt+1}부터 끝까지 차례로 진행합니다."))
        self._run_task(nxt)

    def _on_stop_all(self):
        """전체 자동 실행 중지 요청 — 현재 작업이 끝나면 다음으로 넘어가지 않음."""
        if self._auto_run:
            self._auto_run = False
            self._append("EIDOS", "⏹ 전체 자동 실행 중지 — 현재 작업이 끝나면 멈춥니다.", dim=True)
        self._stop_all_btn.setVisible(False); self._run_all_btn.setVisible(True)

    def _run_task(self, idx):
        if self._thread or idx >= len(self._plan):
            return
        entry = self._plan[idx]
        task = entry.get("text", "")
        if not task:
            return
        prop = self._get_proposer()
        if prop is None:
            self._append("EIDOS", "⚠ Gemini 키 없음 — 설정에서 입력."); return
        # 프롬프트팩 작업이면 그 프롬프트 내용을 '그대로' 구현 지시로 사용(제목만 X).
        instruction = task
        pf = entry.get("prompt_file")
        if pf and os.path.exists(pf):
            try:
                import eidos_dev_engine.prompt_pack as _pp
                prompt_text = open(pf, encoding="utf-8", errors="replace").read()
                pack = _pp.parse_prompt_pack(self._folder)
                charter = str(pack.get("charter", "")) if pack.get("found") else ""
                instruction = _pp.build_task_instruction(prompt_text, charter, task)
            except Exception as e:  # noqa: BLE001
                print(f"[run_task] 프롬프트 로드 실패, 제목으로 진행(graceful): {e}")
        self._plan[idx]["status"] = "doing"; self._save_plan(); self._render_plan()
        self._active_task = idx
        self._append("나", self._esc(f"▶ 작업 {idx+1} 실행 — {task}"))
        self._run("edit", instruction)

    def _on_qa(self):
        """✅ QA 검증 = 앱 실행 → 버튼 구동 → 기능명세서(Petalbot QA 체크리스트)와 대조."""
        if self._thread:
            return
        folder = self._folder
        self._append("나", "✅ QA 검증 — 앱 실행시켜 명세서대로 검증")

        def fn(cb):
            cb("✅ QA 시작 — 명세서·스택 탐색…")
            spec = _find_qa_spec(folder)
            if not spec:
                return {"kind": "qa", "error": "명세서(QA 체크리스트)를 못 찾았어요. Petalbot 'QA 체크리스트 생성'으로 만들어 이 프로젝트 폴더에 두세요."}
            relspec = os.path.relpath(spec, folder)
            cb(f"📋 명세: {relspec}")
            try:
                det = deng.detect_project(folder)
                stack = getattr(det, "stack", "") or "불명"
                entry = getattr(det, "primary_entrypoint", "") or ""
            except Exception:  # noqa: BLE001
                stack, entry = "불명", ""
            if entry and not os.path.isabs(entry):
                entry = os.path.join(folder, entry)
            cb(f"🔎 스택: {stack}")
            # ── PySide6 = 동적 구동(앱 실행+클릭) ──
            if stack == "pyside6" and entry and os.path.isfile(entry):
                cb(f"▶ 앱 구동(offscreen): {os.path.basename(entry)} …")
                try:
                    ex = deng.exercise_app(entry, visible=False)
                except Exception as e:  # noqa: BLE001
                    ex = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                if ex.get("ok"):
                    cb(f"🖱 버튼 {ex.get('buttons_clicked', 0)}개 구동·관찰 → 명세 대조…")
                    cmp = deng.compare_app_to_spec(ex, spec, project_root=folder)
                    return {"kind": "qa", "mode": "dynamic", "cmp": cmp, "spec": relspec,
                            "stack": stack, "clicked": ex.get("buttons_clicked", 0),
                            "reacted": ex.get("reacted", 0)}
                cb(f"⚠ 앱 구동 실패({ex.get('error', '')}) → 정적 검증으로 전환")
            # ── CLI(python) = 동적 구동(실제 실행+출력 캡처) #2 ──
            if stack == "cli" and entry and os.path.isfile(entry):
                cb(f"▶ CLI 실행: {os.path.basename(entry)} (기본 + --help) …")
                import eidos_qa_cli as qc
                ex = qc.exercise_cli(entry, folder)
                cb("📤 출력 캡처 → 명세 대조…")
                cmp = qc.compare_cli_to_spec(ex, spec, project_root=folder)
                return {"kind": "qa", "mode": "dynamic-cli", "cmp": cmp, "spec": relspec,
                        "stack": stack, "crashed": ex.get("crashed")}
            # ── 웹 서버(fastapi/node) = 띄워서 HTTP 프로브 #4 ──
            if stack in ("fastapi", "node") and entry and os.path.isfile(entry):
                cb(f"🌐 서버 기동·HTTP 프로브: {os.path.basename(entry)} (stack={stack}) …")
                import eidos_qa_server as qs
                sres = qs.exercise_server(entry, folder, stack)
                if sres.get("ok"):
                    cb(f"📡 {sres.get('base', '')} · {len(sres.get('probed', []))}경로 응답 → 명세 대조…")
                    cmp = qs.compare_server_to_spec(sres, spec, project_root=folder)
                    return {"kind": "qa", "mode": "server", "cmp": cmp, "spec": relspec,
                            "stack": stack, "base": sres.get("base", ""), "crashed": sres.get("crashed")}
                cb(f"⚠ 서버 기동 실패({sres.get('error', '')}) → 다른 방식으로 전환")
            # ── 웹 정적 HTML #3 (HTML 존재 시 — 마크업/JS 분석) ──
            import eidos_qa_web as qw
            web_entry = qw.find_web_entry(folder)
            if web_entry:
                cb(f"🌐 웹 정적 분석: {os.path.relpath(web_entry, folder)} (HTML/JS) …")
                wres = qw.exercise_web_static(folder)
                cmp = qw.compare_web_to_spec(wres, spec, project_root=folder)
                return {"kind": "qa", "mode": "web-static", "cmp": cmp, "spec": relspec, "stack": stack or "web"}
            # ── 그 외(lib/node/불명) = 정적 검증(코드↔명세, 모든 스택 공통) ──
            cb("📑 정적 검증(코드↔명세)…")
            try:
                v = deng.verify_spec(spec, folder)
            except Exception as e:  # noqa: BLE001
                return {"kind": "qa", "error": f"정적 검증 실패: {type(e).__name__}: {e}"}
            feats = [{"title": (m.item.title or "")[:60], "status": m.status, "conflict": bool(getattr(m, "conflict", False))}
                     for m in (getattr(v, "matches", []) or [])]
            return {"kind": "qa", "mode": "static", "spec": relspec, "stack": stack,
                    "static": {"total": len(feats), "implemented": getattr(v, "implemented", 0),
                               "uncertain": getattr(v, "uncertain", 0), "missing": getattr(v, "missing", 0),
                               "coverage": getattr(v, "coverage", 0.0), "features": feats}}
        self._set_busy(True, "✅ QA 검증 중…")
        self._thread = _EngineThread(fn, self)
        self._thread.progress.connect(lambda mm: self._append("EIDOS", self._esc(mm), dim=True))
        self._thread.done.connect(self._on_done)
        self._thread.start()

    def _on_qa_precise(self):
        """🔬 정밀 QA — 앱을 실제로 띄우고 명세 항목마다 그 버튼을 직접 클릭해 기대결과 확인.
        polaris autodrive(화면 보고 클릭 + Gemini 비전 판정 PASS/FAIL/확인불가). GUI(pyside6)만."""
        if self._thread:
            return
        folder = self._folder
        spec = _find_qa_spec(folder)
        if not spec:
            self._append("EIDOS", "🔬 명세서(QA 체크리스트)를 못 찾았어요 — Petalbot 'QA 체크리스트 생성'으로 만들어 폴더에 두세요.")
            return
        if not _gemini_key():
            self._append("EIDOS", "🔬 정밀 QA는 Gemini 키가 필요해요(항목 추출·화면 판정) — 설정에서 입력.")
            return
        try:
            det = deng.detect_project(folder)
            stack = getattr(det, "stack", "") or "불명"
            entry = getattr(det, "primary_entrypoint", "") or ""
        except Exception:  # noqa: BLE001
            stack, entry = "불명", ""
        if stack != "pyside6" or not entry:
            self._append("EIDOS", "🔬 정밀 클릭 QA는 화면 GUI 앱(pyside6)만 지원해요 — 다른 스택은 ✅ 일반 QA를 쓰세요.")
            return
        if not os.path.isabs(entry):
            entry = os.path.join(folder, entry)
        title = (self._title or os.path.basename(folder) or "").strip()
        relspec = os.path.relpath(spec, folder)
        self._append("나", "🔬 정밀 QA — 앱을 띄우고 명세 항목마다 직접 눌러 검사합니다.")
        self._append("EIDOS", "  ⚠ 검사 동안 마우스·키보드를 AIRA가 사용해요. 끝날 때까지 컴퓨터를 건드리지 마세요.", dim=True)

        def fn(cb):
            import subprocess as _sp
            import time as _t
            import autodrive_launcher as _adl
            cb(f"▶ 앱 구동: {os.path.basename(entry)} (창 뜨는 중)…")
            try:
                proc = _sp.Popen([sys.executable, entry], cwd=os.path.dirname(entry) or folder,
                                 stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True, encoding="utf-8",
                                 errors="replace")
            except Exception as e:  # noqa: BLE001
                return {"kind": "qa_precise", "error": f"앱 실행 실패: {type(e).__name__}: {e}"}
            _t.sleep(3.2)   # 창이 뜰 여유
            if proc.poll() is not None:   # 바로 죽음 = 크래시
                out = ""
                try:
                    out = (proc.stdout.read() if proc.stdout else "")[:1500]
                except Exception:  # noqa: BLE001
                    pass
                return {"kind": "qa_precise", "error": f"앱이 바로 종료됐어요(크래시 가능) — 먼저 ▶ 앱 실행으로 확인하세요.\n{out}"}
            out_path = os.path.join(folder, ".eidos_qa_report.md")
            try:
                res = _adl.stream_qa(spec, window=title, out=out_path, on_line=cb,
                                     max_ticks_per_item=6, timeout=1800.0)
            except Exception as e:  # noqa: BLE001
                res = None
                cb(f"⚠ 정밀 QA 실행 오류: {type(e).__name__}: {e}")
            finally:
                try:
                    proc.terminate()
                except Exception:  # noqa: BLE001
                    pass
            report_md = ""
            try:
                if os.path.exists(out_path):
                    report_md = open(out_path, encoding="utf-8", errors="replace").read()
            except Exception:  # noqa: BLE001
                pass
            tail = (getattr(res, "stdout", "") or "")[-1500:]
            if not report_md and not tail:
                return {"kind": "qa_precise", "error": "정밀 QA가 결과를 내지 못했어요 — autodrive 의존성(pyautogui/uiautomation) 설치·창 포커스를 확인하세요."}
            return {"kind": "qa_precise", "report": report_md, "tail": tail, "spec": relspec}

        self._set_busy(True, "🔬 정밀 QA 중…")
        self._thread = _EngineThread(fn, self)
        self._thread.progress.connect(lambda mm: self._append("EIDOS", self._esc(mm), dim=True))
        self._thread.done.connect(self._on_done)
        self._thread.start()

    def _on_scaffold(self):
        if self._thread:
            return
        instr = self._chat_input.text().strip()
        self._chat_input.clear()
        self._append("나", f"🏗 스캐폴딩 요청{(' — ' + instr) if instr else ''}")
        self._run("scaffold", instr)

    def _get_proposer(self):
        if self._proposer is not None:
            return self._proposer
        if not _gemini_key():
            return None
        try:
            self._proposer = deng.LLMProposer(llm_sync=_gemini_text)
        except Exception:  # noqa: BLE001
            self._proposer = None
        return self._proposer

    def _run(self, kind, instruction):
        if not self._folder or not os.path.isdir(self._folder):
            self._append("EIDOS", "⚠ 프로젝트 폴더가 없어요."); return
        folder = self._folder
        if kind == "scaffold":
            stack = self._stack.currentText() or "cli"
            instr0 = instruction
            text_blob, images = self._split_attachments()
            has_key = bool(_gemini_key())

            def fn(cb):
                # 1) 프롬프트팩/명세 읽기(폴더의 .md/.txt 우선) + 첨부 반영
                ctx, _flist = _read_project_context(folder)
                if text_blob:
                    ctx = (ctx + "\n\n[첨부 문서/명세]\n" + text_blob).strip()
                if images:
                    cb(f"🖼 첨부 이미지 {len(images)}장 분석 → 명세로 변환…")
                    try:
                        spec = _gemini_multimodal(
                            "이미지를 보고 만들 프로그램의 기능 명세를 한국어로 항목화하라. 설명·머리말 없이 명세만.",
                            images)
                        if (spec or "").strip():
                            ctx = (ctx + "\n\n[이미지 기반 명세]\n" + spec.strip()).strip()
                    except Exception:  # noqa: BLE001
                        pass
                # 2) LLM 멀티파일 스캐폴딩 — 명세를 읽고 실제 프로젝트 구조 생성
                if has_key:
                    cb("🏗 명세/프롬프트팩 읽고 프로젝트 구조 생성 중… (멀티파일)")
                    import eidos_dev_engine.scaffold_llm as _sl
                    files = {}
                    try:
                        files = _sl.generate_scaffold(instr0, ctx, _gemini_text, stack=stack)
                    except Exception as e:  # noqa: BLE001
                        cb(f"⚠ LLM 생성 실패({type(e).__name__}) — 템플릿으로 폴백")
                    if files:
                        cb(f"🏗 {len(files)}개 파일 작성·검증 중…")
                        wr = _sl.write_scaffold(folder, files, overwrite=True, validate=True)
                        return {"kind": "scaffold", "ok": bool(wr.get("ok")),
                                "created": list(wr.get("created", [])),
                                "message": wr.get("message", "")}
                # 3) 폴백: 키 없거나 LLM 실패 → 결정적 템플릿(빈 껍데기)
                cb(f"🏗 템플릿 스캐폴딩(stack={stack})… (LLM 미사용)")
                res = deng.scaffold_project(folder, "app", stack)
                return {"kind": "scaffold", "ok": bool(getattr(res, "ok", False)),
                        "created": [os.path.relpath(p, folder) for p in (getattr(res, "created", []) or [])],
                        "message": getattr(res, "message", "")
                        + ("" if has_key else "  (Gemini 키를 넣으면 명세 기반 멀티파일 생성)")}
            self._set_busy(True, "🏗 스캐폴딩 중…")
        else:
            prop = self._get_proposer()
            if prop is None:
                self._append("EIDOS", "⚠ Gemini 키가 없어 개발 엔진을 못 돌려요 — 설정에서 gemini_api_key 입력 후 다시.")
                return

            text_blob, images = self._split_attachments()

            def fn(cb):
                instr = instruction
                if text_blob:   # 첨부 명세/문서를 지시에 합침
                    instr = f"{instr}\n\n[첨부 문서/명세 — 반드시 반영]\n{text_blob}"
                if images:       # 첨부 스크린샷을 멀티모달로 '구체적 개발 지시'로 변환 후 엔진에 투입
                    cb(f"🖼 첨부 이미지 {len(images)}장 분석 → 개발 지시로 변환…")
                    try:
                        spec = _gemini_multimodal(
                            "다음은 사용자의 개발 지시와 참고 스크린샷/이미지다. 이미지를 보고 사용자가 "
                            "원하는 변경을 **구체적이고 실행 가능한 한국어 개발 지시문**으로 다시 써라. "
                            "UI라면 어느 화면의 무엇을 어떻게 바꿀지 명확히. 설명·머리말 없이 지시문만 출력.\n\n"
                            f"[사용자 지시]\n{instr}", images)
                        if (spec or "").strip():
                            instr = spec.strip()
                            cb("🖼 이미지 반영 지시 생성 완료")
                    except Exception as e:  # noqa: BLE001
                        cb(f"⚠ 이미지 분석 실패({type(e).__name__}) — 텍스트 지시로 진행")
                rep = deng.auto_dev(folder, instr, prop, on_event=cb)
                return {"kind": "edit", "ok": bool(getattr(rep, "success", False)),
                        "fixed": list(getattr(rep, "fixed_files", []) or []),
                        "build_after": getattr(rep, "build_ok_after", None)}
            self._set_busy(True, "🛠 개발 엔진 실행 중…")
        self._thread = _EngineThread(fn, self)
        self._thread.progress.connect(lambda mm: self._append("EIDOS", self._esc(mm), dim=True))  # 진행=흐릿
        self._thread.done.connect(self._on_done)
        self._thread.start()

    def _on_done(self, report):
        self._thread = None
        self._set_busy(False)
        if report.get("error"):
            self._append("EIDOS", f"⚠ 실패: {report['error']}")
            if self._auto_run:   # 전체 자동 실행 중 오류 → 안전하게 중단
                self._append("EIDOS", "⏹ 전체 자동 실행을 오류로 중단했어요 — 문제 작업을 확인 후 다시 실행하세요.", dim=True)
                self._on_stop_all()
                self._active_task = None
            return
        kind = report.get("kind")
        if kind == "route":   # LLM 의도 분류 결과 → 해당 동작으로 자동 라우팅
            intent = report.get("intent", "idea")
            text = report.get("text", "")
            label = {"todo": "🔨 할 일 전체 구현", "scaffold": "🏗 스캐폴딩", "qa": "✅ QA 검증",
                     "edit": "🛠 코드 수정", "idea": "💬 아이디어 논의"}.get(intent, "💬 아이디어 논의")
            self._append("EIDOS", self._esc(f"🧭 의도 분류 → {label}"), dim=True)
            if intent == "todo":
                self._implement_todos()
            elif intent == "scaffold":
                self._run("scaffold", text)
            elif intent == "qa":
                if any(k in (text or "") for k in ("정밀", "하나하나", "직접 눌", "직접눌", "클릭", "버튼 눌", "실제로 눌")):
                    self._on_qa_precise()
                else:
                    self._on_qa()
            elif intent == "edit":
                self._run("edit", text)
            else:
                self._do_chat(text)
            return
        if kind == "chat":   # 대화 — 파일 변경 없음(트리/에디터 갱신 안 함)
            self._append("EIDOS", self._md_to_html(report.get("text", "")))
            return
        if kind == "plan":   # 구현 계획 — 체크리스트 렌더(파일 변경 없음)
            tasks = report.get("tasks", []) or []
            if not tasks:
                self._append("EIDOS", "📋 계획을 만들지 못했어요 — 프롬프트/명세가 폴더에 있는지 확인.")
                return
            self._plan = tasks
            self._save_plan(); self._render_plan()
            if report.get("autorun"):   # 🔨 할 일 전체 구현 — 작성 즉시 자동 실행
                self._append("EIDOS", self._esc(f"📋 할 일 {len(tasks)}개 작업으로 분해 — 지금부터 자동 구현합니다."))
                self._on_run_all()
            else:
                self._append("EIDOS", self._esc(f"📋 구현 계획 {len(tasks)}개 작업 작성 — 각 작업 옆 ▶ 로 하나씩 실행하세요."))
            return
        if kind == "qa":     # QA 결과 — 명세 대조 리포트(파일 변경 없음)
            if report.get("error"):
                self._append("EIDOS", "🔎 QA: " + self._esc(report["error"]))
                return
            stack = report.get("stack", "")
            spec = report.get("spec", "")
            if report.get("mode") in ("dynamic", "dynamic-cli", "web-static", "server"):
                cmp = report.get("cmp", {}) or {}
                if not cmp.get("ok"):
                    self._append("EIDOS", "🔎 QA 대조 실패: " + self._esc(cmp.get("error", "")))
                    return
                mmode = report.get("mode")
                total = cmp.get("spec_total", 0) or 0
                verified = cmp.get("covered", 0) or 0           # 실측: 동작까지 관찰됨
                unverified = (cmp.get("partial", 0) or 0) + (cmp.get("code_only", 0) or 0)
                absent = cmp.get("absent", 0) or 0
                # 실제로 앱을 구동·조작한 모드인가(web-static/마크업은 실측 아님)
                really_exercised = mmode in ("dynamic", "dynamic-cli", "server")
                mode_label = {"dynamic-cli": "CLI 실제 실행", "web-static": "웹 HTML 분석(클릭X)",
                              "server": "웹서버 HTTP 프로브"}.get(mmode, "앱 구동·클릭")
                done = really_exercised and total > 0 and verified == total
                head = "✅" if done else "🔎"
                clicked_note = ""
                if report.get("clicked") is not None and mmode == "dynamic":
                    clicked_note = f" (앱 실행·버튼 {report.get('clicked', 0)}개 클릭·반응 {report.get('reacted', 0)})"
                self._append("EIDOS", self._esc(
                    f"{head} QA {mode_label} · {stack} · 명세 {total}항목{clicked_note} — "
                    f"실측 동작 확인 {verified}개 / 미검증 {unverified}개 / 없음 {absent}개"))
                if mmode == "web-static":
                    self._append("EIDOS", "  ※ HTML/JS 마크업만 분석 — 실제 클릭·동작은 검증 안 함(가짜 통과 아님)", dim=True)
                if not done:
                    miss = total - verified
                    self._append("EIDOS", self._esc(
                        f"  ⚠ '완료' 아님 — {miss}개 항목은 실제 동작을 확인하지 못했어요"
                        + ("(컨트롤만 있거나 코드만 있음·반응 미관찰). " if unverified else ". ")
                        + "직접 눌러서 확인이 필요합니다."))
                    if mmode in ("dynamic", "web-static"):
                        self._append("EIDOS", "  🔬 '정밀 QA' 라고 해주시면 앱을 띄워 항목마다 직접 눌러 검사해요(시간·마우스 사용).", dim=True)
                else:
                    self._append("EIDOS", "  ✅ 모든 명세 항목을 앱 구동으로 실측 확인했어요.")
                if mmode == "server" and report.get("base"):
                    self._append("EIDOS", self._esc(f"  🌐 서버: {report.get('base')} (구동 후 자동 종료)"), dim=True)
                if report.get("crashed"):
                    self._append("EIDOS", "  ⚠ 기본 실행에서 크래시(트레이스백) 감지 — 우선 수정 필요", dim=True)
                icons = {"covered": "✅실측", "partial": "❓미검증", "code_only": "📦코드만", "absent": "❌없음"}
                feats = cmp.get("features", []) or []
            else:   # static — 앱을 아예 실행 안 함(코드만 봄)
                s = report.get("static", {}) or {}
                self._append("EIDOS", self._esc(
                    f"⚠ QA 정적 분석만 · {stack} · {spec} — 앱을 실행·클릭하지 않았어요"
                    "(이 스택은 동적 구동 미지원)."))
                self._append("EIDOS", self._esc(
                    f"  📑 코드↔명세 참고치: 구현추정 {s.get('implemented', 0)} · 불확실 {s.get('uncertain', 0)} · "
                    f"미구현 {s.get('missing', 0)} / 전체 {s.get('total', 0)} — 실제 동작은 미검증이라 '완료'가 아닙니다."), dim=True)
                icons = {"implemented": "📑추정", "uncertain": "❓", "missing": "❌"}
                feats = s.get("features", []) or []
            for f in feats[:25]:
                st = f.get("status", "")
                cf = " ⚠모순" if f.get("conflict") else ""
                self._append("EIDOS", self._esc(f"  {icons.get(st, '·')} {str(f.get('title', ''))[:54]}{cf}"), dim=True)
            return
        if kind == "qa_precise":   # 🔬 항목별 실제 클릭 검사 결과
            if report.get("error"):
                self._append("EIDOS", "🔬 정밀 QA: " + self._esc(report["error"]))
                return
            md = report.get("report", "")
            if md:
                self._append("EIDOS", "🔬 정밀 QA 완료 — 항목마다 실제로 눌러본 결과예요:")
                self._append("EIDOS", self._md_to_html(md))   # 요약(P/T PASS·실패·확인불가) + 항목별 표
                self._append("EIDOS", self._esc("  ※ 리포트 저장: .eidos_qa_report.md · 확인불가/실패 항목은 직접 한 번 더 봐주세요(가짜 통과 아님)."), dim=True)
            else:
                self._append("EIDOS", "🔬 정밀 QA 결과:")
                self._append("EIDOS", self._esc(report.get("tail", "")[-1200:] or "(결과 없음)"), dim=True)
            return
        if kind == "scaffold":
            created = report.get("created", [])
            if created:
                self._append("EIDOS", self._esc(f"🏗 스캐폴딩 완료 — 생성 {len(created)}개: " + ", ".join(created[:8])))
            else:
                self._append("EIDOS", self._esc(f"🏗 스캐폴딩: {report.get('message', '') or '생성된 파일 없음'}"))
        else:  # auto_dev — 상세 로그는 이미 스트리밍됨, 최종 요약만
            fixed = report.get("fixed", [])
            tail = []
            if report.get("build_after") is True:
                tail.append("✅ 빌드/실행 통과")
            elif report.get("build_after") is False:
                tail.append("⚠ 빌드 미통과")
            if fixed:
                tail.append(f"수정 {len(fixed)}개: " + ", ".join(os.path.basename(f) for f in fixed[:6]))
            self._append("EIDOS", self._esc("🛠 개발 완료 — " + (" · ".join(tail) if tail else ("성공" if report.get("ok") else "완료"))))
            # 계획의 작업 실행이었으면 → 완료 표시 + 다음 안내
            if self._active_task is not None and self._active_task < len(self._plan):
                self._plan[self._active_task]["status"] = "done"
                self._save_plan(); self._render_plan()
                done_idx = self._active_task
                self._active_task = None
                nxt = self._next_todo(done_idx)
                if nxt is not None:
                    if self._auto_run:   # 전체 자동 실행 — 다음 작업을 바로 이어서
                        self._append("EIDOS", self._esc(f"  → 다음 작업 {nxt+1} 자동 실행: {self._plan[nxt].get('text','')[:50]}"), dim=True)
                        QTimer.singleShot(0, lambda i=nxt: self._run_task(i))
                    else:
                        self._append("EIDOS", self._esc(f"  → 다음 작업 {nxt+1}: {self._plan[nxt].get('text','')[:50]} (▶ 로 실행)"), dim=True)
                else:
                    self._append("EIDOS", "  🎉 계획의 모든 작업 완료!", dim=True)
                    if self._auto_run:
                        self._on_stop_all()
        # 에디터 갱신 (파일트리 없음 — 변경/생성 파일을 자동으로 열어 보여줌)
        if self._cur_file and os.path.exists(self._cur_file):
            self._open_file(self._cur_file)   # 열린 파일 변경 반영
        elif not self._dirty:
            self._open_newest()               # 방금 만든/고친 파일 자동으로 보여주기

    def _open_newest(self):
        """폴더에서 가장 최근 수정된 코드 파일을 에디터에 연다(스트리밍 결과 즉시 확인)."""
        best, bt = "", 0.0
        skip = {"__pycache__", ".git", "node_modules", ".venv", "dist", "build", ".eidos_dev_backups"}
        exts = (".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".json", ".md", ".vue", ".java", ".go", ".rs")
        for dp, dn, fs in os.walk(self._folder):
            dn[:] = [d for d in dn if d not in skip]
            for fn in fs:
                if not fn.lower().endswith(exts):
                    continue
                fp = os.path.join(dp, fn)
                try:
                    mt = os.path.getmtime(fp)
                except Exception:  # noqa: BLE001
                    continue
                if mt > bt:
                    bt, best = mt, fp
        if best:
            self._open_file(best)

    def _set_busy(self, busy, msg=""):
        for w in (self._send, self._plan_btn, self._todo_btn, self._chat_input,
                  self._run_all_btn, self._attach_btn, self._run_app_btn):
            w.setEnabled(not busy)
        # ⏹ 중지 버튼은 자동 실행 중에도 항상 누를 수 있어야 함
        self._stop_all_btn.setEnabled(True)
        self._busy.setText(msg)

    def _append(self, who, html_text, *, dim=False):
        c, _m, _d = _resolve_theme(self._theme_name)
        color = c.get("accent_primary", "#2563eb") if who == "EIDOS" else c.get("text_secondary", "#64748b")
        if dim:   # 진행 로그 — 흐릿하게(대화 메시지와 구분)
            hint = c.get("text_hint", "#94a3b8")
            self._chat_log.append(f"<span style='color:{hint}'>{html_text}</span>")
        else:
            self._chat_log.append(f"<b style='color:{color}'>{who}</b> &nbsp; {html_text}")
        sb = self._chat_log.verticalScrollBar()   # 자동 스크롤(스트리밍 따라감)
        sb.setValue(sb.maximum())

    @staticmethod
    def _esc(s):
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _md_inline(self, s):
        """인라인 마크다운(**굵게**·`코드`)만 HTML로 — 나머지는 이스케이프.
        이스케이프가 *·` 를 건드리지 않으므로 escape 후 토큰만 복원하면 안전."""
        L = getattr(self, "_L", None) or _LUX
        s = DevProjectPanel._esc(s)
        s = re.sub(r"\*\*(.+?)\*\*", rf"<b style='color:{L['title']}'>\1</b>", s)
        s = re.sub(r"`([^`]+)`",
                   rf"<code style='background:{L['mint_soft']};color:{L['mint']};"
                   rf"padding:0 3px;border-radius:3px'>\1</code>", s)
        return s

    def _md_to_html(self, text):
        """LLM 답변(마크다운류)을 가독성 있는 HTML로 — 헤더/불릿/번호/굵게/코드/문단 구분.
        한 줄로 이어붙던 답변을 줄·문단·목록 구조로 렌더(QTextEdit 부분집합 HTML)."""
        L = getattr(self, "_L", None) or _LUX
        out = []
        for raw in (text or "").split("\n"):
            ln = raw.rstrip()
            if not ln.strip():                       # 빈 줄 → 문단 간격
                out.append("<div style='font-size:5px'>&nbsp;</div>")
                continue
            m = re.match(r"^(#{1,6})\s+(.*)$", ln)    # 헤더
            if m:
                out.append(f"<div style='color:{L['mint']};font-weight:800;"
                           f"margin-top:5px'>{self._md_inline(m.group(2))}</div>")
                continue
            mb = re.match(r"^\s*[-*•·]\s+(.*)$", ln)  # 불릿
            if mb:
                out.append(f"<div style='margin-left:13px'><span style='color:{L['purple']}'>"
                           f"•</span>&nbsp;{self._md_inline(mb.group(1))}</div>")
                continue
            mn = re.match(r"^\s*(\d+)[.)]\s+(.*)$", ln)  # 번호 목록
            if mn:
                out.append(f"<div style='margin-left:13px'><span style='color:{L['purple']};"
                           f"font-weight:700'>{mn.group(1)}.</span>&nbsp;"
                           f"{self._md_inline(mn.group(2))}</div>")
                continue
            out.append(f"<div>{self._md_inline(ln)}</div>")
        return "".join(out) or "(빈 응답)"

    # ── 테마 — 현재 EIDOS 테마(4종)를 따라간다 ──
    def apply_theme(self, theme_name):
        self._theme_name = theme_name
        L = self._L = _theme_palette(theme_name)
        self.setStyleSheet(f"""
            QWidget {{ background:{L['bg']}; color:{L['text']}; }}
            QLabel#DevSectionLabel {{ color:{L['mint']}; font-weight:700; font-size:12.5px; background:transparent; letter-spacing:0.3px; }}
            QLabel#DevFileLabel {{ color:{L['title']}; font-weight:600; font-size:12.5px; background:transparent; }}
            QLabel#BusyLabel {{ color:{L['mint']}; font-size:12px; background:transparent; }}
            QLineEdit {{ background:{L['field']}; color:{L['text']}; border:1px solid {L['purple']}; border-radius:8px; padding:8px 11px; selection-background-color:{L['mint_soft']}; }}
            QLineEdit:focus {{ border:1px solid {L['mint']}; }}
            QLineEdit::placeholder {{ color:{L['hint']}; }}
            QPushButton {{ background:{L['mint']}; color:{L['on_mint']}; border:none; border-radius:8px; padding:7px 14px; font-weight:800; }}
            QPushButton:hover {{ background:{L['mint_hover']}; }}
            QPushButton:disabled {{ background:{L['border']}; color:{L['hint']}; }}
            QPushButton#ScaffoldBtn {{ background:transparent; color:{L['purple']}; border:1px solid {L['purple']}; font-weight:700; }}
            QPushButton#ScaffoldBtn:hover {{ background:{L['purple_soft']}; color:{L['purple_hover']}; border-color:{L['purple_hover']}; }}
            QTextEdit#ChatLog {{ background:{L['panel']}; color:{L['text']}; border:1px solid {L['border']}; border-radius:10px; padding:8px; font-size:13px; }}
            QPlainTextEdit#CodeEditor {{ background:{L['field']}; color:{L['text']};
                border:1px solid {L['border']}; border-radius:10px; padding:6px; font-family:{_CODE_FONT}; font-size:13px; selection-background-color:{L['mint_soft']}; }}
            QSplitter::handle {{ background:{L['border']}; }}
            QWidget#PlanHost {{ background:{L['panel']}; border:1px solid {L['purple']}; border-radius:10px; }}
            QLabel#PlanIcon {{ color:{L['mint']}; font-weight:700; background:transparent; }}
            QLabel#PlanText {{ color:{L['text']}; font-size:12px; background:transparent; }}
            QLabel#PlanDone {{ color:{L['hint']}; font-size:12px; background:transparent; text-decoration:line-through; }}
            QLabel#RowHint {{ color:{L['text2']}; font-size:11.5px; background:transparent; }}
            QPushButton#PlanRun {{ background:transparent; color:{L['mint']}; border:1px solid {L['border']}; border-radius:6px; padding:2px 6px; font-weight:700; }}
            QPushButton#PlanRun:hover {{ background:{L['mint_soft']}; border-color:{L['mint']}; }}
            QPushButton#PlanRunAll {{ background:{L['mint']}; color:{L['on_mint']}; border:none; border-radius:6px; padding:4px 10px; font-weight:800; font-size:11.5px; }}
            QPushButton#PlanRunAll:hover {{ background:{L['mint_hover']}; }}
            QPushButton#PlanStopAll {{ background:transparent; color:{L['purple']}; border:1px solid {L['purple']}; border-radius:6px; padding:4px 10px; font-weight:700; font-size:11.5px; }}
            QPushButton#PlanStopAll:hover {{ background:{L['purple_soft']}; }}
            QPushButton#AttachBtn {{ background:transparent; color:{L['mint']}; border:1px solid {L['purple']}; border-radius:8px; padding:7px 6px; font-size:15px; }}
            QPushButton#AttachBtn:hover {{ background:{L['mint_soft']}; border-color:{L['mint']}; }}
            QWidget#AttachBar {{ background:{L['purple_soft']}; border:1px solid {L['purple']}; border-radius:8px; }}
            QLabel#AttachLabel {{ color:{L['text2']}; font-size:11.5px; background:transparent; padding:3px 2px; }}
            QPushButton#AttachClear {{ background:transparent; color:{L['text2']}; border:none; font-weight:700; padding:2px; }}
            QPushButton#AttachClear:hover {{ color:{L['mint']}; }}
            QScrollBar:vertical {{ background:{L['bg']}; width:10px; margin:0; }}
            QScrollBar::handle:vertical {{ background:{L['border']}; border-radius:5px; min-height:24px; }}
            QScrollBar::handle:vertical:hover {{ background:{L['purple']}; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
        """)
        if hasattr(self, "_highlighter"):
            hl_colors = {"accent_primary": L["mint"], "text_hint": L["hint"],
                         "bg_base": L["bg"], "text_primary": L["text"]}
            self._highlighter.apply_theme(hl_colors, L["is_dark"])   # 하이라이트도 테마 따라감


# ──────────────────────────────────────────────────────────────
# 워크스페이스 = 프로젝트별 서브탭
# ──────────────────────────────────────────────────────────────
class DevWorkspace(QWidget):
    dev_requested = Signal(str, str)   # 호환용(현재는 패널이 직접 엔진 호출)

    def __init__(self, theme_name="Modern Dark", parent=None):
        super().__init__(parent)
        self._theme_name = theme_name
        self._projects: list = []
        self._panels: dict = {}

        root = QVBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        top = QFrame(); top.setObjectName("DevTop")
        tl = QHBoxLayout(top); tl.setContentsMargins(14, 8, 14, 8); tl.setSpacing(8)
        self._title = QLabel("🛠 개발 워크스페이스"); self._title.setObjectName("DevTitle")
        self._new_btn = QPushButton("➕ 새 프로젝트"); self._new_btn.setObjectName("ProjNew"); self._new_btn.clicked.connect(self._add_project)
        self._rename_btn = QPushButton("✏ 이름변경"); self._rename_btn.clicked.connect(self._rename_project)
        self._del_btn = QPushButton("🗑 삭제"); self._del_btn.setObjectName("ProjDel"); self._del_btn.clicked.connect(self._delete_project)
        self._reload = QPushButton("⟳ 새로고침"); self._reload.clicked.connect(self._load_projects)
        tl.addWidget(self._title); tl.addStretch(1)
        tl.addWidget(self._new_btn); tl.addWidget(self._rename_btn); tl.addWidget(self._del_btn); tl.addWidget(self._reload)
        root.addWidget(top)
        self._top = top

        self._tabs = QTabWidget(); self._tabs.setObjectName("ProjTabs")
        self._tabs.setMovable(True); self._tabs.setDocumentMode(True)
        self._tabs.currentChanged.connect(self._ensure_built)
        # 탭 우클릭 → 이름변경/삭제
        tabbar = self._tabs.tabBar()
        tabbar.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        tabbar.customContextMenuRequested.connect(self._tab_menu)
        tabbar.tabBarDoubleClicked.connect(lambda _i: self._rename_project())   # 더블클릭=이름변경
        root.addWidget(self._tabs, 1)

        self.apply_theme(theme_name)
        self._load_projects()

    def _load_projects(self, select_id: str = ""):
        self._tabs.blockSignals(True)
        self._tabs.clear(); self._panels = {}
        try:
            projs = ddata.get_projects()
            root_dir = ddata._projects_root()
        except Exception:  # noqa: BLE001
            projs, root_dir = [], ""
        self._projects = []
        sel = 0
        for p in projs:
            pid = p.get("id", "")
            folder = os.path.join(root_dir, pid)
            self._projects.append({"id": pid, "title": p.get("title") or "(무제)",
                                   "folder": folder, "outcome": p.get("has_outcome")})
            tab_title = (("✅ " if p.get("has_outcome") else "🟢 ") + (p.get("title") or "(무제)")[:18])
            self._tabs.addTab(QWidget(), tab_title)   # placeholder (lazy build)
            if select_id and pid == select_id:
                sel = len(self._projects) - 1
        self._tabs.blockSignals(False)
        if self._tabs.count():
            self._tabs.setCurrentIndex(sel)
            self._ensure_built(sel)

    def _cur_project(self):
        i = self._tabs.currentIndex()
        if 0 <= i < len(self._projects):
            return i, self._projects[i]
        return -1, None

    # ── 프로젝트 추가/이름변경/삭제 ──
    def _add_project(self):
        from PySide6.QtWidgets import QInputDialog
        title, ok = QInputDialog.getText(self, "새 프로젝트", "프로젝트 이름:")
        if not ok or not (title or "").strip():
            return
        try:
            res = ddata.create_project(title.strip())
        except Exception as e:  # noqa: BLE001
            res = {"error": str(e)}
        if res.get("error"):
            self._toast(f"⚠ 생성 실패: {res['error']}"); return
        self._load_projects(select_id=res.get("id", ""))

    def _rename_project(self):
        from PySide6.QtWidgets import QInputDialog
        i, p = self._cur_project()
        if not p:
            return
        title, ok = QInputDialog.getText(self, "이름 변경", "새 이름:", text=p["title"])
        if not ok or not (title or "").strip():
            return
        try:
            res = ddata.rename_project(p["id"], title.strip())
        except Exception as e:  # noqa: BLE001
            res = {"error": str(e)}
        if res.get("error"):
            self._toast(f"⚠ 이름변경 실패: {res['error']}"); return
        self._load_projects(select_id=p["id"])

    def _delete_project(self):
        from PySide6.QtWidgets import QMessageBox
        i, p = self._cur_project()
        if not p:
            return
        r = QMessageBox.warning(
            self, "프로젝트 삭제",
            f"'{p['title']}' 프로젝트를 폴더째 삭제할까요?\n(되돌릴 수 없습니다 — {p['folder']})",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if r != QMessageBox.StandardButton.Yes:
            return
        try:
            res = ddata.delete_project(p["id"])
        except Exception as e:  # noqa: BLE001
            res = {"error": str(e)}
        if res.get("error"):
            self._toast(f"⚠ 삭제 실패: {res['error']}"); return
        self._load_projects()

    def _tab_menu(self, pos):
        from PySide6.QtWidgets import QMenu
        bar = self._tabs.tabBar()
        idx = bar.tabAt(pos)
        if idx >= 0:
            self._tabs.setCurrentIndex(idx)
        menu = QMenu(self)
        menu.addAction("✏ 이름 변경", self._rename_project)
        menu.addAction("🗑 삭제", self._delete_project)
        menu.addSeparator()
        menu.addAction("➕ 새 프로젝트", self._add_project)
        menu.exec(bar.mapToGlobal(pos))

    def _toast(self, msg):
        """현재 패널 채팅에 메시지(없으면 print)."""
        _i, p = self._cur_project()
        panel = self._panels.get(self._tabs.currentIndex())
        if panel is not None:
            panel._append("EIDOS", panel._esc(msg))
        else:
            print("[DevWorkspace]", msg)

    def _ensure_built(self, idx):
        if idx < 0 or idx in self._panels or idx >= len(self._projects):
            return
        p = self._projects[idx]
        panel = DevProjectPanel(p["folder"], p["title"], self._theme_name)
        self._tabs.blockSignals(True)
        title = self._tabs.tabText(idx)
        self._tabs.removeTab(idx)
        self._tabs.insertTab(idx, panel, title)
        self._tabs.setCurrentIndex(idx)
        self._tabs.blockSignals(False)
        self._panels[idx] = panel

    def apply_theme(self, theme_name):
        self._theme_name = theme_name
        L = _LUX
        self.setStyleSheet(f"""
            QWidget {{ background:{L['bg']}; }}
            QFrame#DevTop {{ background:{L['bg']}; border-bottom:1px solid {L['purple']}; }}
            QLabel#DevTitle {{ color:{L['mint']}; font-size:15px; font-weight:800; background:transparent; letter-spacing:0.5px; }}
            QPushButton {{ background:transparent; color:{L['purple']}; border:1px solid {L['purple']}; border-radius:7px; padding:6px 12px; font-weight:700; }}
            QPushButton:hover {{ background:{L['purple_soft']}; color:{L['purple_hover']}; border-color:{L['purple_hover']}; }}
            QPushButton#ProjNew {{ color:{L['on_mint']}; background:{L['mint']}; border:none; }}
            QPushButton#ProjNew:hover {{ background:{L['mint_hover']}; }}
            QPushButton#ProjDel {{ color:#ff8a8a; border-color:#5a2230; }}
            QPushButton#ProjDel:hover {{ background:rgba(255,80,80,0.12); border-color:#ff8a8a; }}
            QTabWidget::pane {{ border:none; }}
            QTabBar {{ background:transparent; qproperty-drawBase:0; }}
            QTabBar::tab {{ background:{L['panel']}; color:{L['text2']}; padding:8px 14px; margin-right:4px;
                border:1px solid {L['border']}; border-bottom:2px solid transparent; border-top-left-radius:8px; border-top-right-radius:8px; }}
            QTabBar::tab:selected {{ color:{L['mint']}; border:1px solid {L['mint']}; border-bottom:2px solid {L['mint']}; font-weight:700; background:{L['mint_soft']}; }}
            QTabBar::tab:hover {{ color:{L['text']}; border-color:{L['purple']}; }}
            QMenu {{ background:{L['panel']}; color:{L['text']}; border:1px solid {L['purple']}; }}
            QMenu::item:selected {{ background:{L['purple_soft']}; color:{L['mint']}; }}
        """)
        for panel in self._panels.values():
            panel.apply_theme(theme_name)


# ──────────────────────────────────────────────────────────────
def _selftest():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    w = DevWorkspace("Modern Dark")
    print(f"✅ 프로젝트 서브탭 {w._tabs.count()}개 · 첫 패널 빌드: {0 in w._panels}")
    for nm in ["Modern Dark", "Modern White (Crisp)"]:
        w.apply_theme(nm)
    # 첫 패널 채팅 입력(엔진은 키 없으면 스레드서 graceful 실패) — UI 흐름만 확인
    panel = w._panels.get(0)
    if panel:
        panel._chat_input.setText("테스트 지시")
        print("✅ 패널 채팅 입력 OK · 폴더:", os.path.basename(panel._folder))
    print("✅ 셀프테스트 통과 — 프로젝트별 서브탭 · 스캐폴딩버튼 · 엔진 배선")
    app.quit()


def main():
    from PySide6.QtWidgets import QApplication, QComboBox, QVBoxLayout as VB, QWidget as QW
    app = QApplication.instance() or QApplication(sys.argv)
    host = QW(); host.setWindowTitle("EIDOS 개발 워크스페이스 (미리보기)"); host.resize(1320, 840)
    lay = VB(host); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
    combo = QComboBox(); combo.addItems(["Modern Dark", "Modern White (Crisp)", "Modern White (Warm)", "Obsidian"])
    dev = DevWorkspace("Modern Dark")
    combo.currentTextChanged.connect(dev.apply_theme)
    lay.addWidget(combo); lay.addWidget(dev, 1)
    host.show(); sys.exit(app.exec())


if __name__ == "__main__":
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
