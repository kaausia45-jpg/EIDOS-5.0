# eidos_code_editor_v2.py
# [2026-05-25/26] v2 코드 편집기 — 사용자 mockup 기반 새 디자인.
#
# 레이아웃 (3 컬럼 + 하단):
#   ┌──────────────────────────────────────────────────────────────┐
#   │ EIDOS Code Editor                                             │
#   ├──────┬──────────────────────────┬───────────────────────────┤
#   │ 좌   │ 중앙                      │ 우측                       │
#   │      │                           │                           │
#   │ [첨  │ [코드 뷰어]               │ [수정사항] 1~4 list       │
#   │  부  │                           │                           │
#   │  된  │                           ├─────────┬─────────────────┤
#   │  코  │                           │[Before] │ [After]         │
#   │  드] │                           │ diff    │ diff            │
#   │      │                           │         │                 │
#   │ [첨  │                           ├───────────────────────────┤
#   │  부  │                           │ [기능추가 자동 제안       │
#   │  된  │                           │  (EIDOS)]                 │
#   │  사  │                           │                           │
#   │  진] │                           │                           │
#   ├──────┴──────────────────────────┴───────────────────────────┤
#   │ [자유 입력] 코드에서 수정하고 싶은 부분이 있나요? ...           │
#   │ [파일첨부] [기능명세서 비교] [바로 실행] [스캐폴딩]            │
#   └──────────────────────────────────────────────────────────────┘
#
# Phase 1 (이번): UI 골격 + 햄버거 wire + placeholder 동작.
# Phase 2~ : 실제 wire (코드 열기·proposal 생성·diff·feature suggestion·기능명세서 비교).
#
# 진입점: ChatWindow / MainHubWindow 의 햄버거 메뉴 "🆕 v2 코드 편집기 (Beta)".

from __future__ import annotations

import json as _json_theme
import os
from typing import Optional

from PySide6.QtCore import Qt, Signal


# [2026-05-26] 테마 감지 — settings.json 의 theme 가 "White"/"Light" 면 라이트.
def _detect_light_theme() -> bool:
    try:
        _p = os.path.join("eidos_settings.json")
        if os.path.exists(_p):
            with open(_p, "r", encoding="utf-8") as _f:
                _s = _json_theme.load(_f) or {}
            _t = str(_s.get("theme", "Modern Dark"))
            return "White" in _t or "Light" in _t
    except Exception:
        pass
    return False


_LIGHT = _detect_light_theme()


def _root_theme_override() -> str:
    """라이트 모드 root override stylesheet — 자식 위젯 inherit."""
    if _LIGHT:
        return (
            "QMainWindow, QDialog { background-color: #F7F7F8; color: #1A1A1A; }"
            "QWidget { background-color: transparent; color: #1A1A1A; }"
            "QLabel { color: #1A1A1A; }"
            "QLineEdit, QPlainTextEdit, QTextEdit {"
            "  background-color: #FFFFFF; color: #1A1A1A;"
            "  border: 1px solid #D0D0D8; border-radius: 6px; padding: 4px;"
            "}"
            "QPushButton {"
            "  background-color: #6D28D9; color: #FFFFFF;"
            "  border: none; border-radius: 6px; padding: 6px 12px;"
            "}"
            "QPushButton:hover { background-color: #7C3AED; }"
            "QPushButton:disabled { background-color: #B8B8C0; color: #FFFFFF; }"
            "QListWidget {"
            "  background-color: #FFFFFF; color: #1A1A1A;"
            "  border: 1px solid #D0D0D8; border-radius: 6px;"
            "}"
            "QListWidget::item:selected { background-color: #DDD6FE; color: #1A1A1A; }"
            "QFrame { background-color: transparent; }"
            "QSplitter::handle { background-color: #E5E7EB; }"
        )
    return ""
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# ── 색상 팔레트 — [v3.1 사용자 명시 2026-05-26] 모노크롬 검정 + 보라 액센트(버튼만) ──
# 컨셉: 모든 배경 순검정·텍스트 흰색/회색·버튼만 보라 (primary 그라데이션·secondary outlined)
class _C:
    BG_DEEP      = "#000000"      # 메인 배경 순검정
    BG_PANEL     = "#000000"      # 좌/우 사이드바 순검정
    BG_CODE      = "#000000"      # 코드 영역 순검정
    BG_INPUT     = "#000000"      # 하단 input 순검정
    BORDER       = "#1A1A1A"      # 미묘하게 보일 정도 (카드 경계 명확)
    BORDER_HOVER = "#A855F7"      # hover 시 보라
    BORDER_GLOW  = "#A855F7"      # 강조 보라
    ACCENT       = "#A855F7"      # primary 보라 (적당 밝기)
    ACCENT_GLOW  = "#C084FC"      # 더 밝은 보라
    ACCENT_DEEP  = "#7C3AED"      # 어두운 보라
    TEXT         = "#FFFFFF"      # primary 텍스트 흰색
    TEXT_BODY    = "#CCCCCC"      # 본문 밝은 회색
    TEXT_MUTED   = "#888888"      # 메타 중간 회색
    TEXT_HEADER  = "#FFFFFF"      # 헤더 라벨 흰색 (uppercase + letter-spacing)
    DIFF_REMOVE  = "#3A1010"      # before 배경 (어두운 빨강)
    DIFF_ADD     = "#102E1A"      # after 배경 (어두운 초록)


def _section_label(text: str) -> QLabel:
    """[v3.1] 섹션 라벨 — 흰색 uppercase·letter-spacing 으로 명확 hierarchy."""
    clean = text.strip().strip("[]").upper()
    lbl = QLabel(clean)
    lbl.setStyleSheet(
        f"color: {_C.TEXT_HEADER}; font-weight: 700; font-size: 11px;"
        f" padding: 10px 12px 6px 12px; background: transparent;"
        f" letter-spacing: 1.5px;"
    )
    return lbl


def _list_style() -> str:
    return (
        f"QListWidget {{"
        f"  background: transparent;"
        f"  color: {_C.TEXT_BODY};"
        f"  border: none;"
        f"  padding: 2px 6px;"
        f"  font-size: 12px;"
        f"  outline: 0;"
        f"}}"
        f"QListWidget::item {{ padding: 6px 10px; border-radius: 6px; margin: 1px 0; }}"
        f"QListWidget::item:hover {{ background: #1C1A26; color: {_C.TEXT}; }}"
        f"QListWidget::item:selected {{"
        f"  background: #221E2E; color: {_C.ACCENT_GLOW};"
        f"  border-left: 2px solid {_C.ACCENT};"
        f"}}"
    )


def _btn_style(primary: bool = False) -> str:
    """[v3.1 사용자 명시] 버튼만 보라. primary 그라데이션·secondary outlined 보라."""
    if primary:
        return (
            f"QPushButton {{"
            f"  background: qlineargradient(x1:0, y1:0, x2:0, y2:1,"
            f"    stop:0 {_C.ACCENT_GLOW}, stop:1 {_C.ACCENT_DEEP});"
            f"  color: #ffffff;"
            f"  border: 1px solid {_C.ACCENT};"
            f"  border-radius: 8px;"
            f"  padding: 9px 22px;"
            f"  font-size: 13px;"
            f"  font-weight: 700;"
            f"  letter-spacing: 0.3px;"
            f"}}"
            f"QPushButton:hover {{"
            f"  background: qlineargradient(x1:0, y1:0, x2:0, y2:1,"
            f"    stop:0 #D8B4FE, stop:1 {_C.ACCENT});"
            f"  border: 1px solid {_C.ACCENT_GLOW};"
            f"}}"
            f"QPushButton:pressed {{"
            f"  background: {_C.ACCENT_DEEP};"
            f"}}"
            f"QPushButton:disabled {{"
            f"  background: #1A0A2E; color: #5B4E78; border: 1px solid #2D1B47;"
            f"}}"
        )
    # secondary — outlined 보라 (보더 + 텍스트 보라·배경 검정)
    return (
        f"QPushButton {{"
        f"  background: #000000;"
        f"  color: {_C.ACCENT_GLOW};"
        f"  border: 1px solid {_C.ACCENT_DEEP};"
        f"  border-radius: 8px;"
        f"  padding: 9px 20px;"
        f"  font-size: 13px;"
        f"  font-weight: 600;"
        f"  letter-spacing: 0.2px;"
        f"}}"
        f"QPushButton:hover {{"
        f"  border: 1px solid {_C.ACCENT_GLOW};"
        f"  color: #ffffff;"
        f"  background: rgba(168, 85, 247, 0.12);"
        f"}}"
        f"QPushButton:pressed {{"
        f"  background: rgba(168, 85, 247, 0.20);"
        f"}}"
        f"QPushButton:disabled {{"
        f"  color: #5B4E78; border-color: #2D1B47; background: #000000;"
        f"}}"
    )


class CodeEditorV2Window(QMainWindow):
    """[2026-05-26 v2] EIDOS Code Editor (mockup 기반).

    Phase 1: UI 골격 + 햄버거 진입점. 동작은 placeholder.
    Phase 2~: 실제 wire (코드 열기·proposal·diff·feature suggestion·기능명세서 비교).
    """

    # 실제 동작 wire 용 시그널 (Phase 2+ 에서 사용)
    request_proposal = Signal(str, str, dict)   # current_code, user_request, context_data
    request_feature_suggestion = Signal(str)    # current_code
    request_spec_compare = Signal(str)          # current_code
    # [Phase 2-C 2026-05-26] 기능명세서 비교 결과 — worker thread → main thread (QueuedConnection)
    _spec_compare_result = Signal(str, str)     # report_md, spec_filename
    # [Phase 2-E 2026-05-26] 스캐폴딩 결과 — files list (path, content) + target_dir
    _scaffold_result = Signal(list, str)        # files: list[dict{path, content}], target_dir
    # [Phase 2-H 2026-05-26] 자유 대화 — Enter 키로 발동·worker thread → main thread (QueuedConnection)
    _chat_reply_ready = Signal(str, str)        # reply_text, error_or_empty

    def __init__(self, parent: Optional[QWidget] = None, eidos_worker=None):
        super().__init__(parent)
        self.eidos_worker = eidos_worker
        self.setWindowTitle("EIDOS Code Editor v2 (Beta)")
        self.resize(1400, 900)
        self.setMinimumSize(1000, 700)

        # [2026-05-26 라이트 테마 동적 적용] settings.json 의 theme 가 White/Light 면
        # root 에 라이트 stylesheet override — 자식 위젯 inherit.
        if _LIGHT:
            self.setStyleSheet(_root_theme_override())

        # 상태
        self._attached_codes: list[str] = []    # 코드 파일 path list
        self._attached_images: list[str] = []   # 이미지 파일 path list
        self._current_file_path: Optional[str] = None
        self._current_code: str = ""            # [v3.3] 메모리 buffer — 옛 _code_view.toPlainText() 대체
        self._proposals: list[dict] = []        # 우측 [수정사항] list
        self._feature_suggestions: list[dict] = []   # 우측 [기능 추가 자동 제안]
        self._highlighter = None                # [Phase 2-D] 옛 PythonHighlighter (no-op·코드 뷰어 제거)
        self._undo_stack: list[tuple[str, str]] = []   # [Phase 2-G] (code_snapshot, label) — apply 직전 코드
        self._max_undo = 30                     # 스택 cap

        self._build_ui()
        self._apply_style()

        # [Phase 2-H 2026-05-26] 자유 대화 wire — Enter 키 입력 = LLM 에 자유 메시지 전송.
        # ⚡ 바로 실행 (proposal 생성) 과 분리 — 코드 작업 전 EIDOS 와 자유롭게 의논.
        try:
            self._input.installEventFilter(self)
            self._chat_reply_ready.connect(self._on_chat_reply, Qt.QueuedConnection)
            # placeholder 보강 — Enter = 자유 대화·Shift+Enter = 줄바꿈·⚡ = 코드 수정 제안
            self._input.setPlaceholderText(
                "💬 자유 대화: Enter — 코드 작업 전 EIDOS 와 의논\n"
                "⚡ 코드 수정 제안: [바로 실행] 버튼 — Before/After diff 생성\n"
                "(Shift+Enter 줄바꿈·라이브러리 설치/exe빌드/파일 삭제 등도 가능)"
            )
        except Exception as e:
            print(f"⚠️ [CodeEditorV2] 자유 대화 wire 실패 (graceful): {e}")

    # ── UI 구성 ──────────────────────────────────────────────────
    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # 상단 헤더 — [v3.1] 흰색·큰 letter-spacing
        header = QLabel("EIDOS  CODE  EDITOR")
        header.setFont(QFont("", 12, QFont.Bold))
        header.setStyleSheet(
            f"color: {_C.TEXT}; padding: 10px 14px 6px 14px;"
            f" letter-spacing: 3px; background: transparent;"
        )
        root.addWidget(header)

        # 중간 3 컬럼 splitter
        splitter = QSplitter(Qt.Horizontal, central)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet(
            f"QSplitter::handle {{ background: #1A0F2E; }}"
            f"QSplitter::handle:hover {{ background: {_C.ACCENT}; }}"
        )

        splitter.addWidget(self._build_left_sidebar())
        splitter.addWidget(self._build_center_code())
        splitter.addWidget(self._build_right_panels())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        # [v3.2] 좌측 더 넓게 (features 추가)·우측 더 넓게 (Before/After 확장)·중앙 약간 축소
        splitter.setSizes([260, 720, 420])
        root.addWidget(splitter, 1)

        # 하단 — input + 버튼
        root.addWidget(self._build_bottom_input())

    def _build_left_sidebar(self) -> QWidget:
        """[첨부된 코드] + [첨부된 사진] + [기능추가 자동 제안] (v3.2 좌측 이동)."""
        wrap = QFrame()
        wrap.setObjectName("EditorV2_Left")
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        layout.addWidget(_section_label("[첨부된 코드]"))
        self._lst_codes = QListWidget()
        self._lst_codes.setStyleSheet(_list_style())
        self._lst_codes.itemDoubleClicked.connect(self._on_code_item_double_clicked)
        layout.addWidget(self._lst_codes, 1)

        layout.addWidget(_section_label("[첨부된 사진]"))
        self._lst_images = QListWidget()
        self._lst_images.setStyleSheet(_list_style())
        self._lst_images.setFixedHeight(130)
        layout.addWidget(self._lst_images)

        # [v3.2 2026-05-26] 기능추가 자동 제안 — 좌측으로 이동 (옛 우측 패널 → 좌측 사진 밑)
        feat_header_row = QHBoxLayout()
        feat_header_row.setContentsMargins(0, 0, 0, 0)
        feat_header_row.setSpacing(4)
        feat_header_row.addWidget(_section_label("[기능추가 자동 제안 (EIDOS)]"))
        feat_header_row.addStretch(1)
        self._btn_refresh_features = QPushButton("🔄")
        self._btn_refresh_features.setFixedSize(28, 22)
        self._btn_refresh_features.setCursor(Qt.PointingHandCursor)
        self._btn_refresh_features.setToolTip("기능 제안 다시 받기 (LLM 재호출)")
        self._btn_refresh_features.setStyleSheet(
            f"QPushButton {{"
            f"  background: #000000; color: {_C.TEXT_BODY};"
            f"  border: 1px solid {_C.ACCENT_DEEP}; border-radius: 5px;"
            f"  font-size: 11px;"
            f"}}"
            f"QPushButton:hover {{ color: {_C.ACCENT_GLOW}; border-color: {_C.ACCENT_GLOW}; }}"
            f"QPushButton:disabled {{ color: #5B4E78; border-color: #2D1B47; }}"
        )
        self._btn_refresh_features.setEnabled(False)
        self._btn_refresh_features.clicked.connect(self._on_refresh_features_clicked)
        feat_header_row.addWidget(self._btn_refresh_features)
        layout.addLayout(feat_header_row)

        self._lst_features = QListWidget()
        self._lst_features.setStyleSheet(_list_style())
        self._lst_features.setFixedHeight(180)
        self._lst_features.itemDoubleClicked.connect(self._on_feature_double_clicked)
        layout.addWidget(self._lst_features)

        return wrap

    def _build_center_code(self) -> QWidget:
        """[v3.3 2026-05-26] 중앙 — AI 채팅 (옛 코드 뷰어 자리).

        사용자 명시 "코드 내용 볼 필요 없다·AI 와의 채팅 공간으로". 코드 뷰어 완전 제거.
        대신 _current_code 메모리 buffer 가 코드 source 역할.
        """
        wrap = QFrame()
        wrap.setObjectName("EditorV2_Center")
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(4)

        # 상단 — 현재 파일 라벨 (어떤 파일 active 인지 안내·옛 lbl_current_file 유지)
        self._lbl_current_file = QLabel("(파일 선택 안 됨)")
        self._lbl_current_file.setStyleSheet(
            f"color: {_C.TEXT_MUTED}; font-size: 11px; padding: 4px 8px;"
            f" background: transparent;"
        )
        layout.addWidget(self._lbl_current_file)

        # AI 채팅 영역 — QTextEdit readonly·메시지 누적
        self._chat_view = QTextEdit()
        self._chat_view.setReadOnly(True)
        self._chat_view.setStyleSheet(
            f"QTextEdit {{"
            f"  background: {_C.BG_CODE};"
            f"  color: {_C.TEXT};"
            f"  border: 1px solid {_C.BORDER};"
            f"  border-radius: 10px;"
            f"  padding: 14px 16px;"
            f"  font-size: 13px;"
            f"}}"
        )
        self._chat_view.setPlaceholderText(
            "AI 와의 코드 수정 논의가 여기에 누적됩니다.\n\n"
            "1. 좌측 [첨부된 코드] 더블클릭 또는 [파일 첨부] 로 파일 로드\n"
            "2. 하단 input 에 수정 요청 → [바로 실행]\n"
            "3. 우측 [수정사항] / [Before-After] 검토 → [적용]\n"
            "4. [저장] 으로 디스크 반영"
        )
        layout.addWidget(self._chat_view, 1)

        # 옛 _code_view alias — 다른 메서드 호환 (단·실제 사용은 _current_code 메모리)
        # 코드 뷰어 자체는 제거됐지만 attr 이름 유지로 graceful fallback
        self._code_view = self._chat_view

        return wrap

    # ── [v3.3] AI 채팅 메시지 추가 헬퍼 ─────────────────────────────────────

    def _append_chat(self, role: str, text: str) -> None:
        """채팅에 메시지 추가. role ∈ {user, ai, system, error}."""
        try:
            import html as _h
            color_map = {
                "user":   _C.ACCENT_GLOW,
                "ai":     _C.TEXT,
                "system": _C.TEXT_MUTED,
                "error":  "#F87171",
            }
            icon_map = {
                "user":   "👤  나",
                "ai":     "🤖  EIDOS",
                "system": "ℹ️  시스템",
                "error":  "⚠️  오류",
            }
            color = color_map.get(role, _C.TEXT)
            label = icon_map.get(role, role.upper())
            esc = _h.escape(text or "").replace("\n", "<br>")
            bg = "rgba(168, 85, 247, 0.06)" if role == "user" else "transparent"
            html_part = (
                f'<div style="margin: 6px 0; padding: 8px 12px; border-radius: 8px;'
                f' background: {bg};">'
                f'<div style="color: {color}; font-weight: 700; font-size: 11px;'
                f' letter-spacing: 0.5px; margin-bottom: 4px;">{label}</div>'
                f'<div style="color: {_C.TEXT_BODY}; font-size: 13px;">{esc}</div>'
                f'</div>'
            )
            self._chat_view.append(html_part)
            # 자동 스크롤 — 맨 아래
            try:
                sb = self._chat_view.verticalScrollBar()
                if sb:
                    sb.setValue(sb.maximum())
            except Exception:
                pass
        except Exception as e:
            print(f"⚠️ [CodeEditorV2] _append_chat 실패 (graceful): {e}")

    def _build_right_panels(self) -> QWidget:
        """우측 — [수정사항] / [Before-After] / [기능추가 제안]."""
        wrap = QFrame()
        wrap.setObjectName("EditorV2_Right")
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # [수정사항]
        layout.addWidget(_section_label("[수정사항]"))
        self._lst_proposals = QListWidget()
        self._lst_proposals.setStyleSheet(_list_style())
        self._lst_proposals.setMaximumHeight(180)
        self._lst_proposals.itemClicked.connect(self._on_proposal_clicked)
        layout.addWidget(self._lst_proposals)

        # [Before] [After] 가로 split
        diff_row = QHBoxLayout()
        diff_row.setSpacing(4)
        diff_row.setContentsMargins(0, 0, 0, 0)

        # Before — [Phase 2-G] QTextEdit (HTML diff highlight 지원)
        before_wrap = QVBoxLayout()
        before_wrap.setContentsMargins(0, 0, 0, 0)
        before_wrap.setSpacing(2)
        before_wrap.addWidget(_section_label("[Before]"))
        self._txt_before = QTextEdit()
        self._txt_before.setReadOnly(True)
        self._txt_before.setStyleSheet(
            f"QTextEdit {{ background: {_C.BG_CODE}; color: {_C.TEXT_BODY};"
            f" border: 1px solid {_C.BORDER}; border-radius: 6px; padding: 6px;"
            f" font-family: Consolas; font-size: 10px; }}"
        )
        before_wrap.addWidget(self._txt_before, 1)
        diff_row.addLayout(before_wrap)

        # After — [Phase 2-G] QTextEdit (HTML diff highlight 지원)
        after_wrap = QVBoxLayout()
        after_wrap.setContentsMargins(0, 0, 0, 0)
        after_wrap.setSpacing(2)
        after_wrap.addWidget(_section_label("[After]"))
        self._txt_after = QTextEdit()
        self._txt_after.setReadOnly(True)
        self._txt_after.setStyleSheet(
            f"QTextEdit {{ background: {_C.BG_CODE}; color: {_C.TEXT_BODY};"
            f" border: 1px solid {_C.BORDER}; border-radius: 6px; padding: 6px;"
            f" font-family: Consolas; font-size: 10px; }}"
        )
        after_wrap.addWidget(self._txt_after, 1)
        diff_row.addLayout(after_wrap)
        layout.addLayout(diff_row, 1)

        # [Phase 2-A.5 2026-05-26] 적용 버튼 줄 + [Phase 2-G] ↩ undo
        apply_row = QHBoxLayout()
        apply_row.setContentsMargins(0, 4, 0, 0)
        apply_row.setSpacing(6)
        # [v3 모던] 간결한 텍스트·이모지 최소화
        self._btn_apply_current = QPushButton("적용")
        self._btn_apply_all = QPushButton("모두 적용")
        self._btn_undo = QPushButton("↩  되돌리기")
        self._btn_save_file = QPushButton("저장")
        for b in (self._btn_apply_current, self._btn_apply_all):
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(_btn_style(primary=True))
            b.setEnabled(False)
            apply_row.addWidget(b)
        for b in (self._btn_undo, self._btn_save_file):
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(_btn_style(primary=False))
            b.setEnabled(False)
            apply_row.addWidget(b)
        layout.addLayout(apply_row)
        self._btn_apply_current.clicked.connect(self._on_apply_current_clicked)
        self._btn_apply_all.clicked.connect(self._on_apply_all_clicked)
        self._btn_undo.clicked.connect(self._on_undo_clicked)
        self._btn_save_file.clicked.connect(self._on_save_file_clicked)

        # [v3.2 2026-05-26] 기능추가 자동 제안 영역은 좌측 사이드바로 이동
        # (Before/After 패널이 우측 공간 더 차지하도록)

        return wrap

    def _build_bottom_input(self) -> QWidget:
        """하단 — 자유 input + 4 버튼. [v3 모던] 스타일은 _apply_style 통합."""
        wrap = QFrame()
        wrap.setObjectName("EditorV2_Bottom")
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        # [Phase 2-F 2026-05-26] PasteAwareTextEdit — Ctrl+V / 드래그-드롭 첨부 자동 가로채기
        try:
            from eidos_chat_gui import PasteAwareTextEdit
            self._input = PasteAwareTextEdit()
        except Exception as _e_paste:
            print(f"⚠️ [CodeEditorV2] PasteAwareTextEdit 로드 실패 (graceful·기본 QTextEdit): {_e_paste}")
            self._input = QTextEdit()
        self._input.setStyleSheet(
            f"QTextEdit {{"
            f"  background: {_C.BG_CODE};"
            f"  color: {_C.TEXT};"
            f"  border: 1px solid {_C.BORDER};"
            f"  border-radius: 10px;"
            f"  padding: 10px 14px;"
            f"  font-size: 13px;"
            f"  selection-background-color: {_C.ACCENT_DEEP};"
            f"}}"
            f"QTextEdit:focus {{ border-color: {_C.BORDER_HOVER}; }}"
        )
        self._input.setPlaceholderText(
            "코드에서 수정하고 싶은 부분이 있나요? 자유롭게 입력해주세요.\n"
            "(라이브러리 즉석 설치, exe빌드, 파일삭제/생성 등등도 가능합니다.)"
        )
        self._input.setFixedHeight(80)
        layout.addWidget(self._input)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        # [v3 모던] 이모지 1개·간결 텍스트
        self._btn_attach = QPushButton("파일 첨부")
        self._btn_compare = QPushButton("기능명세서 비교")
        self._btn_run_now = QPushButton("바로 실행")
        self._btn_scaffold = QPushButton("스캐폴딩")
        for b in (self._btn_attach, self._btn_compare, self._btn_scaffold):
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(_btn_style(primary=False))
            btn_row.addWidget(b)
        self._btn_run_now.setCursor(Qt.PointingHandCursor)
        self._btn_run_now.setStyleSheet(_btn_style(primary=True))
        btn_row.addStretch(1)
        btn_row.addWidget(self._btn_run_now)
        layout.addLayout(btn_row)

        # 핸들러 — Phase 1 은 placeholder
        self._btn_attach.clicked.connect(self._on_attach_clicked)
        self._btn_compare.clicked.connect(self._on_compare_clicked)
        self._btn_run_now.clicked.connect(self._on_run_now_clicked)
        self._btn_scaffold.clicked.connect(self._on_scaffold_clicked)

        return wrap

    def _apply_style(self) -> None:
        # [v3 모던] 전체 톤 — 부드러운 검정·보더 미묘·radius 균일
        self.setStyleSheet(
            f"QMainWindow {{ background: {_C.BG_DEEP}; }}"
            f"QFrame#EditorV2_Left {{ background: {_C.BG_PANEL};"
            f"  border: 1px solid {_C.BORDER}; border-radius: 12px; }}"
            f"QFrame#EditorV2_Center {{ background: transparent; }}"
            f"QFrame#EditorV2_Right {{ background: {_C.BG_PANEL};"
            f"  border: 1px solid {_C.BORDER}; border-radius: 12px; }}"
            f"QFrame#EditorV2_Bottom {{ background: {_C.BG_INPUT};"
            f"  border: 1px solid {_C.BORDER}; border-radius: 12px; }}"
            f"QLabel {{ background: transparent; }}"
            f"QScrollBar:vertical {{"
            f"  background: transparent; width: 10px; margin: 4px 0;"
            f"}}"
            f"QScrollBar::handle:vertical {{"
            f"  background: #2A2638; border-radius: 5px; min-height: 30px;"
            f"}}"
            f"QScrollBar::handle:vertical:hover {{ background: {_C.BORDER_HOVER}; }}"
            f"QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}"
        )

    # ── 공개 API ──────────────────────────────────────────────────

    def load_file_into_editor(self, path: str) -> bool:
        """[2026-05-26] 외부 호출용 — 파일 path 를 좌측 [첨부된 코드] 에 추가하고
        중앙에 로드 (옛 _on_code_item_double_clicked 와 동일 효과).

        파일 트리 클릭·드래그-드롭·다른 GUI 위젯에서 v2 로 진입할 때 사용.

        Returns: 성공 시 True·실패 (파일 없음·read 실패) 시 False.
        """
        if not path or not os.path.exists(path):
            return False
        try:
            self._attach_file_path(path)  # 좌측 list 에 chip 추가 (idempotent)
            # 좌측 list 에서 해당 항목 선택 — 시각적 일관성
            try:
                for i in range(self._lst_codes.count()):
                    it = self._lst_codes.item(i)
                    if it and it.data(Qt.UserRole) == path:
                        self._lst_codes.setCurrentItem(it)
                        break
            except Exception:
                pass
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                code_text = f.read()
            self._current_code = code_text
            self._current_file_path = path
            self._lbl_current_file.setText(f"📄 {os.path.basename(path)}  —  {path}")
            self._append_chat(
                "system",
                f"📄 파일 로드: {os.path.basename(path)} ({len(code_text):,}자, "
                f"{len(code_text.splitlines())}줄)"
            )
            self._append_chat(
                "ai",
                "코드를 받았습니다. 어떤 부분을 수정하면 좋을까요?\n"
                "💬 Enter 로 자유 대화 (의논)·⚡ [바로 실행] 으로 코드 수정 제안 생성."
            )
            try:
                self._request_feature_suggestion(path, code_text)
            except Exception as _e_feat:
                print(f"⚠️ [CodeEditorV2] feature_suggestion 자동 호출 실패 (graceful): {_e_feat}")
            return True
        except Exception as e:
            try:
                QMessageBox.warning(self, "파일 읽기 실패", f"{type(e).__name__}: {e}")
            except Exception:
                pass
            return False

    # ── 핸들러 (Phase 1 — placeholder) ────────────────────────────

    def _on_code_item_double_clicked(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.UserRole) or item.text()
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                code_text = f.read()
            # [v3.3] 코드 뷰어 제거 — _current_code 메모리 buffer 에 저장
            self._current_code = code_text
            self._current_file_path = path
            self._lbl_current_file.setText(f"📄 {os.path.basename(path)}  —  {path}")
            # 채팅 메시지 — 시스템 안내
            self._append_chat(
                "system",
                f"📄 파일 로드: {os.path.basename(path)} ({len(code_text):,}자, "
                f"{len(code_text.splitlines())}줄)"
            )
            self._append_chat(
                "ai",
                "코드를 받았습니다. 어떤 부분을 수정하면 좋을까요?\n"
                "💡 하단 input 에 자연어로 요청 후 [바로 실행] 을 눌러주세요.\n"
                "💡 좌측 [기능추가 자동 제안] 도 자동으로 분석됩니다."
            )
            # [Phase 2-B] 파일 로드 시 feature_suggestion 자동 호출
            self._request_feature_suggestion(path, code_text)
        except Exception as e:
            QMessageBox.warning(self, "파일 읽기 실패", f"{type(e).__name__}: {e}")

    # ── [Phase 2-F 2026-05-26] Ctrl+V / 드래그-드롭 첨부 — PasteAwareTextEdit duck typing ──

    def _attach_image_from_qimage(self, qimg) -> None:
        """[Phase 2-F] clipboard QImage → 임시 PNG 저장 후 [첨부된 사진] list 에 추가."""
        try:
            import tempfile as _tf
            import datetime as _dt
            tmp_dir = os.path.join("eidos_files", "code_editor_v2_attachments")
            os.makedirs(tmp_dir, exist_ok=True)
            ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            tmp_path = os.path.join(tmp_dir, f"paste_{ts}.png")
            ok = qimg.save(tmp_path, "PNG")
            if not ok:
                print(f"⚠️ [CodeEditorV2] QImage 저장 실패: {tmp_path}")
                return
            self._attach_image_from_path(tmp_path)
        except Exception as e:
            print(f"⚠️ [CodeEditorV2] 이미지 첨부 실패 (graceful): {e}")

    def _attach_image_from_path(self, path: str) -> None:
        """[Phase 2-F] 이미지 path → [첨부된 사진] list 에 추가."""
        try:
            if path in self._attached_images:
                return
            self._attached_images.append(path)
            item = QListWidgetItem(f"🖼  {os.path.basename(path)}")
            item.setData(Qt.UserRole, path)
            item.setToolTip(path)
            self._lst_images.addItem(item)
        except Exception as e:
            print(f"⚠️ [CodeEditorV2] 이미지 path 첨부 실패 (graceful): {e}")

    def _attach_file_path(self, path: str) -> None:
        """[Phase 2-F] 파일 path → [첨부된 코드] list 에 추가 (텍스트·일반 무관)."""
        try:
            if path in self._attached_codes:
                return
            self._attached_codes.append(path)
            item = QListWidgetItem(f"📄  {os.path.basename(path)}")
            item.setData(Qt.UserRole, path)
            item.setToolTip(path)
            self._lst_codes.addItem(item)
        except Exception as e:
            print(f"⚠️ [CodeEditorV2] 파일 path 첨부 실패 (graceful): {e}")

    def _attach_highlighter_for(self, file_path: str) -> None:
        """[v3.3 2026-05-26] 코드 뷰어 제거 후 syntax highlighter 도 의미 없어짐 — no-op.
        호환성 위해 함수 시그니처는 유지. 옛 호출자는 영향 X."""
        return

    def _on_proposal_clicked(self, item: QListWidgetItem) -> None:
        """[Phase 2-G] 수정사항 list 클릭 → line-level diff HTML 로 Before/After 갱신."""
        idx = self._lst_proposals.row(item)
        if 0 <= idx < len(self._proposals):
            p = self._proposals[idx]
            before_html, after_html = self._render_diff_html(
                p.get("search_block", ""), p.get("replace_block", ""),
            )
            try:
                self._txt_before.setHtml(before_html)
                self._txt_after.setHtml(after_html)
            except Exception as e:
                # 폴백 — HTML 실패 시 plain text
                print(f"⚠️ [CodeEditorV2] diff HTML 실패 (graceful): {e}")
                self._txt_before.setPlainText(p.get("search_block", ""))
                self._txt_after.setPlainText(p.get("replace_block", ""))

    def _render_diff_html(self, before_text: str, after_text: str) -> tuple[str, str]:
        """[Phase 2-G 2026-05-26] difflib.SequenceMatcher 로 line-level diff → HTML.

        - equal 라인: 회색 (TEXT_MUTED)
        - delete (Before only): 빨강 배경 (#5C1A1A)
        - insert (After only): 초록 배경 (#1A5C2E)
        - replace: Before 빨강 + After 초록
        """
        import difflib
        import html as _html
        before_lines = (before_text or "").splitlines()
        after_lines = (after_text or "").splitlines()
        # 빈 텍스트 처리
        if not before_lines and not after_lines:
            return "", ""

        matcher = difflib.SequenceMatcher(None, before_lines, after_lines, autojunk=False)
        before_parts: list[str] = []
        after_parts: list[str] = []

        def _line_html(text: str, bg: str = "", fg: str = _C.TEXT_BODY) -> str:
            esc = _html.escape(text) if text else "&nbsp;"
            bg_style = f" background:{bg};" if bg else ""
            return (
                f'<div style="font-family: Consolas, \'Courier New\', monospace; '
                f'font-size: 10px; color:{fg};{bg_style} padding: 1px 4px;'
                f' white-space: pre;">{esc}</div>'
            )

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for k in range(i1, i2):
                    before_parts.append(_line_html(before_lines[k], "", _C.TEXT_MUTED))
                for k in range(j1, j2):
                    after_parts.append(_line_html(after_lines[k], "", _C.TEXT_MUTED))
            elif tag == "delete":
                for k in range(i1, i2):
                    before_parts.append(_line_html(before_lines[k], "#5C1A1A", "#FCA5A5"))
            elif tag == "insert":
                for k in range(j1, j2):
                    after_parts.append(_line_html(after_lines[k], "#1A5C2E", "#86EFAC"))
            elif tag == "replace":
                for k in range(i1, i2):
                    before_parts.append(_line_html(before_lines[k], "#5C1A1A", "#FCA5A5"))
                for k in range(j1, j2):
                    after_parts.append(_line_html(after_lines[k], "#1A5C2E", "#86EFAC"))

        before_html = "".join(before_parts) or "<div style='color:#5B4E78;'>(빈 코드)</div>"
        after_html = "".join(after_parts) or "<div style='color:#5B4E78;'>(빈 코드)</div>"
        return before_html, after_html

    def _on_feature_double_clicked(self, item: QListWidgetItem) -> None:
        """기능 제안 더블클릭 → input 에 prefilled."""
        idx = self._lst_features.row(item)
        if 0 <= idx < len(self._feature_suggestions):
            f = self._feature_suggestions[idx]
            title = f.get("title", "")
            desc = f.get("description", "")
            self._input.setPlainText(f"{title} — {desc}")

    # ── [Phase 2-H 2026-05-26] 자유 대화 — Enter 키 자유 대화·⚡ 버튼은 proposal 그대로 ──

    def eventFilter(self, obj, event):
        """input 위젯의 Enter 키 가로채기.

        - Enter (no modifier)  → 자유 대화 (LLM 호출·proposal X·diff X)
        - Shift+Enter          → 줄바꿈 (default 동작 유지)
        - 그 외 모든 키        → default
        """
        try:
            from PySide6.QtCore import QEvent
            from PySide6.QtGui import QKeyEvent
            if obj is self._input and event.type() == QEvent.KeyPress:
                ke: QKeyEvent = event  # type: ignore[assignment]
                if ke.key() in (Qt.Key_Return, Qt.Key_Enter):
                    if ke.modifiers() & Qt.ShiftModifier:
                        return False  # 줄바꿈 — default
                    # IME composition 중이면 default (한글 조합 보호)
                    try:
                        if ke.isAutoRepeat():
                            return False
                    except Exception:
                        pass
                    self._send_free_chat()
                    return True  # 이벤트 소비 — 줄바꿈 X
        except Exception as e:
            print(f"⚠️ [CodeEditorV2] eventFilter 실패 (graceful): {e}")
        return super().eventFilter(obj, event)

    def _send_free_chat(self) -> None:
        """[Phase 2-H] 자유 대화 — 채팅창에 메시지 누적·LLM 호출·proposal/diff 생성 X.

        흐름:
          1. input 텍스트 추출 → 채팅에 user 메시지 추가·input 비우기
          2. 현재 파일·첨부 코드·첨부 이미지 → 컨텍스트 (간단 요약)
          3. eidos_worker.submit_task(_run_chat_async) — get_llm_response_async
          4. _chat_reply_ready signal 으로 main thread 에 결과 전달 → _on_chat_reply
        """
        text = self._input.toPlainText().strip()
        if not text:
            return
        if not self.eidos_worker:
            QMessageBox.warning(
                self, "Worker 부재",
                "EidosWorker 가 연결되지 않았습니다.\n메인 채팅창 통해 v2 편집기를 다시 열어보세요.",
            )
            return

        # UI: user 메시지 즉시 표시·input 비우기·"생각 중" 시스템 메시지
        self._append_chat("user", text)
        self._input.clear()
        self._append_chat("system", "💭 EIDOS 생각 중...")

        # 컨텍스트 — 현재 파일·첨부 파일 목록 (전체 코드는 토큰 절약 위해 첫 8000자 cap)
        ctx_parts: list[str] = []
        if self._current_file_path:
            current_name = os.path.basename(self._current_file_path)
            code_snippet = (self._current_code or "")[:8000]
            tail = "" if len(self._current_code or "") <= 8000 else "\n... (이하 생략·전체 길이: {}자)".format(
                len(self._current_code or "")
            )
            ctx_parts.append(f"[현재 열린 파일: {current_name}]\n```\n{code_snippet}{tail}\n```")
        if self._attached_codes:
            others = [os.path.basename(p) for p in self._attached_codes
                      if p != self._current_file_path]
            if others:
                ctx_parts.append(f"[기타 첨부 코드 파일]: {', '.join(others)}")
        if self._attached_images:
            ctx_parts.append(f"[첨부 이미지]: {len(self._attached_images)}장")
        if self._feature_suggestions:
            titles = [s.get("title", "?") for s in self._feature_suggestions[:5]]
            ctx_parts.append(f"[EIDOS 가 분석한 기능 추가 제안]: {', '.join(titles)}")

        context_block = "\n\n".join(ctx_parts) if ctx_parts else "(현재 로드된 코드 없음)"

        sys_prompt = (
            "너는 EIDOS Code Editor v2 에 통합된 코딩 파트너 AI 다.\n"
            "사용자가 본격적인 코드 수정 작업에 들어가기 전 자유롭게 의논하는 단계다.\n"
            "응답 원칙:\n"
            "- 한국어로 간결하게 (3~8 문장·필요 시 더). 불필요한 서두/요약 금지.\n"
            "- 코드 수정 제안서는 아직 X. 사용자의 아이디어를 듣고·다듬고·트레이드오프 짚어주기.\n"
            "- 사용자가 '바로 실행해줘' / '제안해줘' 등 명확한 코드 변경 요청을 하면\n"
            "  '👉 하단 [바로 실행] 버튼을 누르시면 Before/After diff 가 생성됩니다.' 안내.\n"
            "- 코드 스니펫이 필요하면 마크다운 ```언어``` 코드블록 사용 (짧게)."
        )
        user_prompt = f"[컨텍스트]\n{context_block}\n\n[질문/아이디어]\n{text}"

        async def _run_chat_async():
            try:
                from llm_module import get_llm_response_async
                reply = await get_llm_response_async(
                    user_prompt,
                    system_prompt=sys_prompt,
                    max_tokens=4096,
                    timeout=90,
                )
                self._chat_reply_ready.emit(reply or "(빈 응답)", "")
            except Exception as ex:
                err = f"{type(ex).__name__}: {ex}"
                print(f"⚠️ [CodeEditorV2] free chat async 실패: {err}")
                self._chat_reply_ready.emit("", err)

        try:
            self.eidos_worker.submit_task(_run_chat_async())
        except Exception as e:
            self._append_chat("error", f"submit 실패: {type(e).__name__}: {e}")

    def _on_chat_reply(self, reply: str, error: str) -> None:
        """[Phase 2-H] LLM 응답 main-thread slot — 채팅창에 AI 메시지 추가."""
        if error:
            self._append_chat("error", f"자유 대화 실패: {error[:200]}")
            return
        self._append_chat("ai", reply or "(빈 응답)")

    # ── 버튼 핸들러 (Phase 1) ─────────────────────────────────────

    def _on_attach_clicked(self) -> None:
        """파일 첨부 — 파일 선택 다이얼로그 (사용자 정의 = 단순 의미)."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "파일 첨부", "", "모든 파일 (*.*)"
        )
        if not paths:
            return
        for p in paths:
            ext = os.path.splitext(p)[1].lower()
            IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
            if ext in IMG_EXTS:
                if p not in self._attached_images:
                    self._attached_images.append(p)
                    item = QListWidgetItem(f"🖼  {os.path.basename(p)}")
                    item.setData(Qt.UserRole, p)
                    item.setToolTip(p)
                    self._lst_images.addItem(item)
            else:
                if p not in self._attached_codes:
                    self._attached_codes.append(p)
                    item = QListWidgetItem(f"📄  {os.path.basename(p)}")
                    item.setData(Qt.UserRole, p)
                    item.setToolTip(p)
                    self._lst_codes.addItem(item)

    def _on_compare_clicked(self) -> None:
        """[Phase 2-C 2026-05-26] 기능명세서 비교 — spec 파일 (.md/.txt) ↔ 현재 코드 LLM 분석.

        1. QFileDialog 로 spec 파일 선택
        2. spec + 현재 코드 prompt 조립
        3. worker.submit_task(_run_spec_compare_async) — 비동기 LLM
        4. _spec_compare_result signal → _show_spec_compare_dialog (main thread)
        """
        if not self._current_file_path:
            QMessageBox.warning(
                self, "코드 파일 미선택",
                "비교할 코드 파일을 좌측 [첨부된 코드] 에서 더블클릭으로 먼저 선택하세요.",
            )
            return
        if not self.eidos_worker:
            QMessageBox.warning(self, "Worker 부재", "EidosWorker 가 연결되지 않았습니다.")
            return

        # spec 파일 선택
        spec_path, _ = QFileDialog.getOpenFileName(
            self, "기능명세서 파일 선택",
            os.path.dirname(self._current_file_path) or "",
            "기능명세서 (*.md *.txt *.json);;모든 파일 (*.*)",
        )
        if not spec_path:
            return
        try:
            with open(spec_path, "r", encoding="utf-8", errors="replace") as f:
                spec_text = f.read()
        except Exception as e:
            QMessageBox.warning(self, "spec 읽기 실패", f"{type(e).__name__}: {e}")
            return
        if not spec_text.strip():
            QMessageBox.warning(self, "spec 비어있음", "선택한 spec 파일이 비어있습니다.")
            return

        code_text = self._current_code
        code_fname = os.path.basename(self._current_file_path)
        spec_fname = os.path.basename(spec_path)

        # signal connect (안전 패턴·QueuedConnection — worker thread 안전)
        try:
            self._spec_compare_result.disconnect(self._show_spec_compare_dialog)
        except (RuntimeError, TypeError):
            pass
        try:
            self._spec_compare_result.connect(
                self._show_spec_compare_dialog, Qt.QueuedConnection,
            )
        except Exception as e:
            print(f"⚠️ [CodeEditorV2] spec_compare signal connect 실패 (graceful): {e}")
            return

        # UI: 분석 중
        self._btn_compare.setEnabled(False)
        self._btn_compare.setText("⏳  비교 중...")

        async def _run_spec_compare():
            try:
                from llm_module import get_llm_response_async
                sys_prompt = (
                    "너는 코드와 기능명세서의 mapping 을 분석하는 코드 리뷰어다.\n"
                    "spec 의 각 요구사항·기능·항목이 코드에 구현됐는지 검증하고, "
                    "구현/부분 구현/미구현/괴리 4 카테고리로 분류해 마크다운 보고서를 작성한다.\n"
                    "구체적 함수명·라인 번호·이유를 명시. 추상적·일반론 금지."
                )
                user_prompt = (
                    f"[기능명세서 파일: {spec_fname}]\n"
                    f"{spec_text[:30000]}\n\n"
                    "─────────────────────────────────────\n\n"
                    f"[코드 파일: {code_fname}]\n"
                    f"{code_text[:30000]}\n\n"
                    "─────────────────────────────────────\n\n"
                    "위 spec ↔ 코드를 비교해 다음 양식의 마크다운 보고서를 작성:\n\n"
                    "## 🎯 spec ↔ 코드 mapping 보고서\n\n"
                    "### ✅ 구현된 항목 (spec 요구사항이 코드에 명확히 매칭)\n"
                    "- [spec 항목] → [코드의 함수/클래스/라인] : 근거\n"
                    "- ...\n\n"
                    "### 🟡 부분 구현 (일부만 매칭·확장 필요)\n"
                    "- [spec 항목] → [코드 위치] : 무엇이 부족한가\n"
                    "- ...\n\n"
                    "### ❌ 미구현 (spec 에 있는데 코드에 없음)\n"
                    "- [spec 항목] : 어디에 추가해야 하는지 제안\n"
                    "- ...\n\n"
                    "### ⚠️ 괴리 (spec 과 코드가 다르게 동작)\n"
                    "- [차이점] : 어느 쪽이 의도된 것인지\n"
                    "- ...\n\n"
                    "### 📋 권장 다음 단계\n"
                    "- 우선순위순 3~5개 액션 아이템 (코드 수정 요청에 그대로 쓸 수 있게)\n"
                )
                report = await get_llm_response_async(
                    user_prompt,
                    system_prompt=sys_prompt,
                    max_tokens=16384,
                    timeout=120,
                )
                self._spec_compare_result.emit(report or "", spec_fname)
            except Exception as ex:
                self._spec_compare_result.emit(
                    f"⚠️ 기능명세서 비교 실패: {type(ex).__name__}: {ex}",
                    spec_fname,
                )

        try:
            self.eidos_worker.submit_task(_run_spec_compare())
        except Exception as e:
            self._btn_compare.setEnabled(True)
            self._btn_compare.setText("📋  기능명세서 비교")
            QMessageBox.warning(self, "submit 실패", f"{type(e).__name__}: {e}")

    def _show_spec_compare_dialog(self, report_md: str, spec_fname: str) -> None:
        """[Phase 2-C] spec_compare 결과 main thread slot — 별도 다이얼로그로 마크다운 표시."""
        # UI 복원
        self._btn_compare.setEnabled(True)
        self._btn_compare.setText("📋  기능명세서 비교")
        # signal disconnect — 다음 비교 깨끗
        try:
            self._spec_compare_result.disconnect(self._show_spec_compare_dialog)
        except (RuntimeError, TypeError):
            pass

        if not (report_md or "").strip():
            QMessageBox.warning(self, "비교 결과 없음", "LLM 응답이 비어있습니다.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle(f"📋 기능명세서 비교 — {spec_fname}")
        dlg.resize(800, 700)
        dlg.setStyleSheet(f"QDialog {{ background: {_C.BG_DEEP}; }}")

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        header = QLabel(f"🎯  {spec_fname} ↔ {os.path.basename(self._current_file_path or '코드')}")
        header.setFont(QFont("", 12, QFont.Bold))
        header.setStyleSheet(
            f"color: {_C.ACCENT_GLOW}; padding: 4px 0; background: transparent;"
        )
        layout.addWidget(header)

        # 마크다운 렌더링 (QTextBrowser 가 기본 HTML 지원·markdown→html 변환)
        viewer = QTextBrowser(dlg)
        viewer.setOpenExternalLinks(False)
        viewer.setStyleSheet(
            f"QTextBrowser {{"
            f"  background: {_C.BG_CODE};"
            f"  color: {_C.TEXT};"
            f"  border: 1px solid {_C.BORDER};"
            f"  border-radius: 8px;"
            f"  padding: 14px;"
            f"  font-size: 13px;"
            f"}}"
        )
        try:
            viewer.setMarkdown(report_md)
        except Exception:
            viewer.setPlainText(report_md)
        layout.addWidget(viewer, 1)

        # 하단 — 액션 버튼
        btn_row = QHBoxLayout()
        btn_close = QPushButton("닫기")
        btn_close.setStyleSheet(_btn_style(primary=False))
        btn_close.setCursor(Qt.PointingHandCursor)
        btn_close.clicked.connect(dlg.accept)

        btn_to_input = QPushButton("✏️  '권장 다음 단계' 를 input 에 prefill")
        btn_to_input.setStyleSheet(_btn_style(primary=True))
        btn_to_input.setCursor(Qt.PointingHandCursor)

        def _prefill_to_input():
            # 권장 다음 단계 섹션 추출 — 단순 substring (실패 시 전체)
            text_for_input = report_md
            try:
                if "권장 다음 단계" in report_md:
                    idx = report_md.find("권장 다음 단계")
                    text_for_input = report_md[idx:]
            except Exception:
                pass
            # input 에 prefilled
            self._input.setPlainText(
                f"[기능명세서 비교 — {spec_fname}]\n\n"
                f"{text_for_input}\n\n"
                "위 권장 사항대로 코드 수정해줘."
            )
            dlg.accept()

        btn_to_input.clicked.connect(_prefill_to_input)

        btn_save_md = QPushButton("💾  보고서 .md 저장")
        btn_save_md.setStyleSheet(_btn_style(primary=False))
        btn_save_md.setCursor(Qt.PointingHandCursor)

        def _save_md():
            try:
                import datetime as _dt
                out_dir = os.path.join("eidos_files", "spec_compare_reports")
                os.makedirs(out_dir, exist_ok=True)
                ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                code_fname = os.path.basename(self._current_file_path or "code")
                safe_spec = "".join(c if c.isalnum() else "_" for c in spec_fname)[:40]
                out_path = os.path.join(out_dir, f"{ts}_{safe_spec}_vs_{code_fname}.md")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(f"# 기능명세서 비교 — {spec_fname} ↔ {code_fname}\n\n")
                    f.write(report_md)
                QMessageBox.information(dlg, "💾 저장 완료", f"보고서 저장됨:\n{out_path}")
            except Exception as ex:
                QMessageBox.warning(dlg, "저장 실패", f"{type(ex).__name__}: {ex}")

        btn_save_md.clicked.connect(_save_md)

        btn_row.addStretch(1)
        btn_row.addWidget(btn_save_md)
        btn_row.addWidget(btn_close)
        btn_row.addWidget(btn_to_input)
        layout.addLayout(btn_row)

        dlg.exec()

    def _on_run_now_clicked(self) -> None:
        """[Phase 2-A 2026-05-26] 바로 실행 — request_proposal_async 호출 + proposal_ready 핸들러 wire.

        흐름:
          1. 입력·현재 파일 검증
          2. 첨부 코드 → context_data·첨부 이미지 → image_input (N개 bytes list)
          3. eidos_worker.proposal_ready disconnect + 자기 핸들러 connect
          4. submit_task(request_proposal_async)
          5. 실행 중 UI: 버튼 비활성·"⏳ 분석 중..." 라벨
          6. _on_proposal_received slot: proposals 자동 채움·첫 항목 자동 선택
        """
        text = self._input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "입력 필요", "수정 요청을 입력하세요.")
            return
        if not self._current_file_path or not os.path.exists(self._current_file_path):
            QMessageBox.warning(
                self, "코드 파일 미선택",
                "수정할 코드 파일을 좌측 [첨부된 코드] 에서 더블클릭하거나 [파일 첨부] 로 불러오세요.",
            )
            return
        if not self.eidos_worker:
            QMessageBox.warning(
                self, "Worker 부재",
                "EidosWorker 가 연결되지 않았습니다.\n메인 채팅창 통해 v2 편집기를 다시 열어보세요.",
            )
            return

        # 현재 코드 — 메모리 buffer
        current_code = self._current_code
        current_file_name = os.path.basename(self._current_file_path)

        # 채팅 메시지 — 사용자 요청
        self._append_chat("user", text)
        self._append_chat("system", f"🔍 {current_file_name} 분석 중... (LLM 호출)")

        # 첨부 코드 → context_data (현재 파일 제외)
        context_data: dict = {}
        for code_path in self._attached_codes:
            if not code_path or code_path == self._current_file_path:
                continue
            try:
                with open(code_path, "r", encoding="utf-8", errors="replace") as f:
                    context_data[os.path.basename(code_path)] = f.read()
            except Exception as e:
                print(f"⚠️ [CodeEditorV2] 참조 파일 read 실패 ({code_path}): {e}")

        # 첨부 이미지 → image_input bytes list (vision LLM 멀티 이미지)
        image_input_bytes = None
        if self._attached_images:
            _imgs_bytes = []
            for img_path in self._attached_images:
                try:
                    with open(img_path, "rb") as f:
                        _imgs_bytes.append(f.read())
                except Exception as e:
                    print(f"⚠️ [CodeEditorV2] 이미지 read 실패 ({img_path}): {e}")
            if _imgs_bytes:
                image_input_bytes = _imgs_bytes
                # user_request 에 첨부 이미지 메타 추가
                text = text.rstrip() + (
                    f"\n\n[첨부 이미지 {len(_imgs_bytes)}개 — vision LLM 분석에 포함]"
                )

        # 시그널 연결 (안전·disconnect → connect 패턴)
        try:
            self.eidos_worker.proposal_ready.disconnect(self._on_proposal_received)
        except (RuntimeError, TypeError):
            pass   # 이미 disconnect 됐거나 처음 연결
        try:
            self.eidos_worker.proposal_ready.connect(self._on_proposal_received)
        except Exception as e:
            QMessageBox.warning(
                self, "signal connect 실패", f"{type(e).__name__}: {e}",
            )
            return

        # UI 실행 중 상태
        self._btn_run_now.setEnabled(False)
        self._btn_run_now.setText("⏳  분석 중...")
        self.clear_proposals()

        # submit
        try:
            self.eidos_worker.submit_task(
                self.eidos_worker.request_proposal_async(
                    current_code,
                    text,
                    current_file_name,
                    context_data,
                    override_filepath=self._current_file_path,
                    image_input=image_input_bytes,
                )
            )
        except Exception as e:
            # 즉시 UI 복원
            self._btn_run_now.setEnabled(True)
            self._btn_run_now.setText("⚡  바로 실행")
            try:
                self.eidos_worker.proposal_ready.disconnect(self._on_proposal_received)
            except Exception:
                pass
            QMessageBox.warning(self, "실행 실패", f"{type(e).__name__}: {e}")

    def _on_proposal_received(self, result_dict: dict) -> None:
        """[Phase 2-A 2026-05-26] proposal_ready slot — proposals → [수정사항] 자동 채움."""
        # UI 복원
        self._btn_run_now.setEnabled(True)
        self._btn_run_now.setText("⚡  바로 실행")

        # 자기 핸들러 disconnect — 다음 실행 시 깨끗하게
        try:
            self.eidos_worker.proposal_ready.disconnect(self._on_proposal_received)
        except (RuntimeError, TypeError):
            pass

        proposals = result_dict.get("proposals", []) if isinstance(result_dict, dict) else []

        if not proposals:
            # 진단 가시화 — Phase A 패턴 (eidos_chat_gui v2.0) 과 동일
            err_msg = (result_dict.get("error", "") or "").strip() if isinstance(result_dict, dict) else ""
            raw_preview = (result_dict.get("_raw_preview", "") or "")[:300] if isinstance(result_dict, dict) else ""
            # 채팅 메시지 — 실패 안내
            if err_msg:
                self._append_chat("error", f"AI 응답 처리 실패: {err_msg[:150]}")
            else:
                self._append_chat("ai", "수정할 내용을 찾지 못했습니다. 요청을 더 구체적으로 작성해 보세요.")
            if err_msg:
                hint = ""
                low = err_msg.lower()
                if "MAX_TOKENS" in err_msg or "truncated" in low:
                    hint = "\n\n💡 큰 파일이라 응답이 잘렸을 수 있어요. 수정 범위를 좁히거나 파일을 분할해 보세요."
                elif "비어" in err_msg or "empty" in low:
                    hint = "\n\n💡 LLM 이 빈 응답을 줬어요. 요청을 더 구체적으로."
                elif "형식" in err_msg or "format" in low:
                    hint = "\n\n💡 JSON 형식 못 따름. 재시도 권장."
                preview_part = f"\n\n[원본 응답 앞 일부]\n{raw_preview[:200]}" if raw_preview else ""
                QMessageBox.warning(
                    self, "AI 응답 처리 실패",
                    f"{err_msg}{hint}{preview_part}",
                )
            else:
                QMessageBox.information(
                    self, "수정할 내용 없음",
                    "AI가 수정할 내용을 찾지 못했습니다.\n"
                    "💡 요청을 더 구체적으로 작성하거나 (어떤 함수·라인·증상), 수정 대상 코드 영역을 좁혀 보세요.",
                )
            return

        # proposals 추가
        for p in proposals:
            self.add_proposal(p)

        # 채팅 메시지 — 성공
        descs = [p.get("description", "수정") for p in proposals[:5]]
        more = f" 외 {len(proposals) - 5}건" if len(proposals) > 5 else ""
        self._append_chat(
            "ai",
            f"💡 {len(proposals)}개 수정사항을 제안했습니다 — 우측 [수정사항] 에서 확인 후 [적용]:\n\n"
            + "\n".join(f"  {i+1}. {d}" for i, d in enumerate(descs))
            + more
        )

        # 첫 항목 자동 선택 → Before/After 자동 표시
        try:
            if self._lst_proposals.count() > 0:
                self._lst_proposals.setCurrentRow(0)
                first = self._lst_proposals.item(0)
                if first is not None:
                    self._on_proposal_clicked(first)
        except Exception as e:
            print(f"⚠️ [CodeEditorV2] 첫 proposal auto-select 실패 (graceful): {e}")

    def _on_scaffold_clicked(self) -> None:
        """[Phase 2-E 2026-05-26] 스캐폴딩 — LLM 으로 프로젝트 구조 생성.

        흐름:
          1. input 의 자연어 요청 (예: "Flask API 서버 + SQLite + Docker")
          2. 대상 폴더 선택 (QFileDialog.getExistingDirectory)
          3. LLM 호출 — 파일 list + 각 파일 내용 JSON
          4. 결과 미리보기 다이얼로그 (체크박스로 선택적 생성)
          5. 디스크 작성 + 좌측 [첨부된 코드] list 자동 추가
        """
        if not self.eidos_worker:
            QMessageBox.warning(self, "Worker 부재", "EidosWorker 가 연결되지 않았습니다.")
            return

        # input 에서 요청 (없으면 다이얼로그)
        request = self._input.toPlainText().strip()
        if not request:
            from PySide6.QtWidgets import QInputDialog
            request, ok = QInputDialog.getText(
                self, "🏗 스캐폴딩 — 프로젝트 요청",
                "어떤 프로젝트 구조? (예: 'Flask API 서버 + SQLite + 로그인'):",
            )
            if not ok or not request.strip():
                return
            request = request.strip()

        # 대상 폴더 선택
        target_dir = QFileDialog.getExistingDirectory(
            self, "스캐폴딩 대상 폴더 선택", "",
        )
        if not target_dir:
            return

        # 빈 폴더 권장 (warn — 진행은 허용)
        try:
            existing = [f for f in os.listdir(target_dir) if not f.startswith(".")]
            if existing:
                reply = QMessageBox.question(
                    self, "폴더 비어있지 않음",
                    f"폴더에 {len(existing)}개 항목이 있습니다 (.* 제외).\n"
                    "동일 이름 파일은 덮어쓸 수 있습니다. 계속할까요?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return
        except Exception:
            pass

        # signal connect
        try:
            self._scaffold_result.disconnect(self._show_scaffold_preview)
        except (RuntimeError, TypeError):
            pass
        try:
            self._scaffold_result.connect(
                self._show_scaffold_preview, Qt.QueuedConnection,
            )
        except Exception as e:
            print(f"⚠️ [CodeEditorV2] scaffold signal connect 실패 (graceful): {e}")
            return

        # UI: 분석 중
        self._btn_scaffold.setEnabled(False)
        self._btn_scaffold.setText("⏳  골격 생성 중...")

        async def _run_scaffold():
            try:
                from llm_module import get_llm_response_async
                import json as _json
                sys_prompt = (
                    "너는 프로젝트 스캐폴딩 전문가다.\n"
                    "사용자의 자연어 요청에 맞는 프로젝트 골격 (폴더 구조 + 각 파일 초기 내용) 을 JSON 으로 출력한다.\n"
                    "파일 내용은 minimum viable — 의존성·시작점·readme·gitignore 정도. 상세 구현 X.\n"
                    "응답은 순수 JSON 만 (마크다운 코드블록 금지)."
                )
                user_prompt = (
                    f"[요청]\n{request}\n\n"
                    "다음 JSON 양식으로 출력:\n"
                    "{\n"
                    '  "description": "이 프로젝트 한 줄 요약",\n'
                    '  "files": [\n'
                    '    {\n'
                    '      "path": "상대 경로 (예: src/main.py 또는 .gitignore)",\n'
                    '      "content": "파일 전체 내용 (minimum viable)"\n'
                    '    }\n'
                    '  ]\n'
                    "}\n\n"
                    "파일 개수: 5~15개 권장. 폴더는 path 안 '/' 로 표현 (예: src/api/routes.py)."
                )
                raw = await get_llm_response_async(
                    user_prompt,
                    system_prompt=sys_prompt,
                    response_mime_type="application/json",
                    max_tokens=16384,
                    timeout=120,
                )
                # robust JSON parse
                try:
                    from llm_module import robust_json_parse as _rjp
                    parsed = _rjp(raw)
                except Exception:
                    parsed = _json.loads(raw)
                files = []
                if isinstance(parsed, dict):
                    files = parsed.get("files", []) or []
                if not isinstance(files, list):
                    files = []
                self._scaffold_result.emit(files, target_dir)
            except Exception as ex:
                print(f"⚠️ [CodeEditorV2] scaffold async 실패: {type(ex).__name__}: {ex}")
                self._scaffold_result.emit([], target_dir)

        try:
            self.eidos_worker.submit_task(_run_scaffold())
        except Exception as e:
            self._btn_scaffold.setEnabled(True)
            self._btn_scaffold.setText("🏗  스캐폴딩")
            QMessageBox.warning(self, "submit 실패", f"{type(e).__name__}: {e}")

    def _show_scaffold_preview(self, files: list, target_dir: str) -> None:
        """[Phase 2-E] 스캐폴딩 결과 미리보기 다이얼로그 — 체크박스로 선택적 생성."""
        # UI 복원
        self._btn_scaffold.setEnabled(True)
        self._btn_scaffold.setText("🏗  스캐폴딩")
        try:
            self._scaffold_result.disconnect(self._show_scaffold_preview)
        except (RuntimeError, TypeError):
            pass

        if not files:
            QMessageBox.warning(
                self, "스캐폴딩 결과 없음",
                "LLM 이 파일 list 를 생성하지 못했습니다. 요청을 더 구체적으로 작성해 보세요.",
            )
            return

        # 미리보기 다이얼로그
        dlg = QDialog(self)
        dlg.setWindowTitle(f"🏗 스캐폴딩 미리보기 — {os.path.basename(target_dir)}")
        dlg.resize(780, 600)
        dlg.setStyleSheet(f"QDialog {{ background: {_C.BG_DEEP}; }}")

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        header = QLabel(f"📂 대상: {target_dir}    ·    파일 {len(files)}개 생성 예정")
        header.setStyleSheet(
            f"color: {_C.ACCENT_GLOW}; padding: 4px 0; font-weight: 600;"
            f" background: transparent;"
        )
        layout.addWidget(header)

        # 체크박스 list (path + 내용 미리보기)
        file_list = QListWidget()
        file_list.setStyleSheet(_list_style())
        for f in files:
            path = f.get("path", "")
            content = f.get("content", "") or ""
            if not path:
                continue
            preview = content[:60].replace("\n", " ⏎ ")
            it = QListWidgetItem(f"☑  {path}     — {preview}")
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked)
            it.setData(Qt.UserRole, f)
            it.setToolTip(content[:500])
            file_list.addItem(it)
        layout.addWidget(file_list, 1)

        # 하단 버튼
        btn_row = QHBoxLayout()
        btn_cancel = QPushButton("취소")
        btn_cancel.setStyleSheet(_btn_style(primary=False))
        btn_cancel.setCursor(Qt.PointingHandCursor)
        btn_cancel.clicked.connect(dlg.reject)
        btn_create = QPushButton(f"✅  체크된 파일 생성")
        btn_create.setStyleSheet(_btn_style(primary=True))
        btn_create.setCursor(Qt.PointingHandCursor)

        def _do_create():
            created = 0
            skipped: list[str] = []
            new_paths: list[str] = []
            for i in range(file_list.count()):
                it = file_list.item(i)
                if not it or it.checkState() != Qt.Checked:
                    continue
                f = it.data(Qt.UserRole)
                if not isinstance(f, dict):
                    continue
                rel_path = f.get("path", "").strip()
                content = f.get("content", "") or ""
                if not rel_path or rel_path.startswith("/") or ".." in rel_path:
                    skipped.append(f"안전하지 않은 path 거부: {rel_path}")
                    continue
                abs_path = os.path.join(target_dir, rel_path)
                try:
                    os.makedirs(os.path.dirname(abs_path) or target_dir, exist_ok=True)
                    with open(abs_path, "w", encoding="utf-8") as wf:
                        wf.write(content)
                    created += 1
                    new_paths.append(abs_path)
                except Exception as ex:
                    skipped.append(f"{rel_path}: {ex}")
            # 좌측 [첨부된 코드] list 에 자동 추가
            for p in new_paths:
                ext = os.path.splitext(p)[1].lower()
                IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
                if ext in IMG_EXTS:
                    continue   # 이미지 파일은 보통 스캐폴딩에 X
                if p not in self._attached_codes:
                    self._attached_codes.append(p)
                    item = QListWidgetItem(f"📄  {os.path.basename(p)}")
                    item.setData(Qt.UserRole, p)
                    item.setToolTip(p)
                    self._lst_codes.addItem(item)
            msg = f"✅ 파일 {created}개 생성됨"
            if skipped:
                msg += f" / ⚠️ skip {len(skipped)}건\n\n" + "\n".join(f"  • {s}" for s in skipped[:5])
                if len(skipped) > 5:
                    msg += f"\n  ... 외 {len(skipped) - 5}건"
            QMessageBox.information(dlg, "스캐폴딩 결과", msg)
            dlg.accept()

        btn_create.clicked.connect(_do_create)
        btn_row.addStretch(1)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_create)
        layout.addLayout(btn_row)

        dlg.exec()

    # ── 외부에서 데이터 채우는 API (Phase 2+ wire 진입점) ────────────

    def add_proposal(self, proposal: dict) -> None:
        """우측 [수정사항] list 에 항목 추가. proposal: {description, search_block, replace_block}"""
        self._proposals.append(proposal)
        desc = proposal.get("description", "(설명 없음)")
        item = QListWidgetItem(f"{len(self._proposals)}.  {desc}")
        self._lst_proposals.addItem(item)
        # 적용 버튼 활성
        try:
            self._btn_apply_current.setEnabled(True)
            self._btn_apply_all.setEnabled(True)
        except Exception:
            pass

    def add_feature_suggestion(self, feat: dict) -> None:
        """우측 [기능추가 자동 제안] list 에 항목 추가. feat: {title, description, why}"""
        self._feature_suggestions.append(feat)
        title = feat.get("title", "(제목 없음)")
        item = QListWidgetItem(f"💡  {title}")
        item.setToolTip(feat.get("description", "") + "\n\n" + feat.get("why", ""))
        self._lst_features.addItem(item)

    # ── [Phase 2-B 2026-05-26] feature_suggestion 자동 호출 ───────────────

    def _request_feature_suggestion(self, file_path: str, code_text: str) -> None:
        """파일 로드 시 백그라운드로 기능 제안 요청 — 우측 [기능추가 자동 제안] 자동 채움."""
        if not self.eidos_worker or not code_text.strip():
            return
        if not hasattr(self.eidos_worker, "request_feature_suggestion_async"):
            return   # 옛 worker — graceful skip
        # 시그널 connect (안전 패턴)
        try:
            self.eidos_worker.feature_suggestion_ready.disconnect(self._on_feature_suggestion_received)
        except (RuntimeError, TypeError):
            pass
        try:
            self.eidos_worker.feature_suggestion_ready.connect(self._on_feature_suggestion_received)
        except Exception as e:
            print(f"⚠️ [CodeEditorV2] feature signal connect 실패 (graceful): {e}")
            return
        # UI: 분석 중 안내
        self.clear_feature_suggestions()
        try:
            placeholder = QListWidgetItem("💡  기능 제안 분석 중... (LLM 호출)")
            placeholder.setFlags(Qt.NoItemFlags)   # 클릭/선택 불가
            self._lst_features.addItem(placeholder)
            self._btn_refresh_features.setEnabled(False)
        except Exception:
            pass
        fname = os.path.basename(file_path)
        try:
            self.eidos_worker.submit_task(
                self.eidos_worker.request_feature_suggestion_async(
                    code_text, fname, file_path,
                )
            )
        except Exception as e:
            print(f"⚠️ [CodeEditorV2] feature suggestion submit 실패 (graceful): {e}")
            self._lst_features.clear()
            self._btn_refresh_features.setEnabled(bool(self._current_file_path))

    def _on_feature_suggestion_received(self, result_dict: dict) -> None:
        """[Phase 2-B] feature_suggestion_ready slot — proposals → [기능추가 자동 제안] 자동 채움.

        결과 file_path 가 현재 _current_file_path 와 다르면 무시 (다른 파일 호출 결과).
        """
        # disconnect — 다음 호출 깨끗
        try:
            self.eidos_worker.feature_suggestion_ready.disconnect(self._on_feature_suggestion_received)
        except (RuntimeError, TypeError):
            pass
        # placeholder 제거
        self._lst_features.clear()
        self._feature_suggestions.clear()

        if not isinstance(result_dict, dict):
            self._btn_refresh_features.setEnabled(bool(self._current_file_path))
            return
        # 다른 파일 결과면 무시 (사용자가 빠르게 파일 전환했을 때)
        result_path = result_dict.get("_file_path", "")
        if result_path and self._current_file_path and result_path != self._current_file_path:
            self._btn_refresh_features.setEnabled(bool(self._current_file_path))
            return

        proposals = result_dict.get("proposals", []) or []
        for feat in proposals:
            if isinstance(feat, dict):
                self.add_feature_suggestion(feat)
        if not proposals:
            try:
                empty_item = QListWidgetItem("(기능 제안 결과 없음 — 🔄 로 재시도)")
                empty_item.setFlags(Qt.NoItemFlags)
                self._lst_features.addItem(empty_item)
            except Exception:
                pass
        self._btn_refresh_features.setEnabled(bool(self._current_file_path))

    def _on_refresh_features_clicked(self) -> None:
        """🔄 — 현재 파일의 기능 제안 다시 받기."""
        if not self._current_file_path:
            return
        code_text = self._current_code
        if not code_text.strip():
            return
        self._request_feature_suggestion(self._current_file_path, code_text)

    def clear_proposals(self) -> None:
        self._proposals.clear()
        self._lst_proposals.clear()
        self._txt_before.clear()
        self._txt_after.clear()
        try:
            self._btn_apply_current.setEnabled(False)
            self._btn_apply_all.setEnabled(False)
        except Exception:
            pass

    # ── [Phase 2-A.5 2026-05-26] 적용 + 저장 핸들러 ───────────────────────

    # ── [Phase 2-G 2026-05-26] undo 스택 ──────────────────────────────────

    def _push_undo(self, code_snapshot: str, label: str) -> None:
        """apply 직전 코드 + 라벨 → undo 스택. cap = _max_undo."""
        try:
            self._undo_stack.append((code_snapshot, label))
            if len(self._undo_stack) > self._max_undo:
                self._undo_stack.pop(0)
            self._btn_undo.setEnabled(True)
        except Exception as e:
            print(f"⚠️ [CodeEditorV2] undo push 실패 (graceful): {e}")

    def _on_undo_clicked(self) -> None:
        """↩ undo — 마지막 snapshot 코드 뷰어에 복원. 빈 스택 시 비활성."""
        if not self._undo_stack:
            return
        code_snapshot, label = self._undo_stack.pop()
        self._current_code = code_snapshot
        # 코드 뷰어 변경 → dirty
        self._btn_save_file.setEnabled(True)
        if not self._undo_stack:
            self._btn_undo.setEnabled(False)
        # 헤더 dirty 마커 유지 (저장 안 된 변경 있음)
        cur_text = self._lbl_current_file.text()
        if " *" not in cur_text:
            self._lbl_current_file.setText(cur_text.rstrip(" ·*") + " *")
        self._append_chat("system", f"↩ 되돌림: {label} (남은 undo: {len(self._undo_stack)})")

    def _apply_proposal_to_code(self, current_code: str, proposal: dict) -> tuple[bool, str, str]:
        """proposal 을 current_code 에 적용 시도. (success, new_code, summary)
        매칭 4단계: 정확→공백 정규화→line-by-line strip (들여쓰기 보정)→실패.
        """
        import re as _re
        search_block = proposal.get("search_block") or ""
        replace_block = proposal.get("replace_block") or ""
        desc = proposal.get("description", "수정")
        prop_type = (proposal.get("type") or "MODIFY").upper()

        # OVERWRITE 차단 (옛 정책과 일관)
        if prop_type == "OVERWRITE":
            return False, current_code, f"OVERWRITE 차단: {desc}"

        # search_block 빈 + ADD/NEW → 끝에 추가
        if not search_block:
            if prop_type in ("ADD", "NEW"):
                new_code = current_code.rstrip() + "\n\n" + replace_block + "\n"
                return True, new_code, f"➕ {desc} (파일 끝에 추가)"
            return False, current_code, f"search_block 비어있음 + type={prop_type}: {desc}"

        # 1단계: 정확 일치
        if search_block in current_code:
            return True, current_code.replace(search_block, replace_block, 1), f"🔧 {desc}"

        # 2단계: 공백/탭 정규화
        def _norm(s):
            s = s.replace('\r\n', '\n').replace('\t', '    ')
            return _re.sub(r'[ \t]+$', '', s, flags=_re.MULTILINE)
        norm_code = _norm(current_code)
        norm_search = _norm(search_block)
        if norm_search and norm_search in norm_code:
            return True, norm_code.replace(norm_search, replace_block, 1), f"🔧 {desc} (공백 정규화)"

        # 3단계: line-by-line strip 매칭 (들여쓰기 완전 무시 + 자동 보정)
        try:
            s_strip = [l.strip() for l in search_block.splitlines() if l.strip()]
            if s_strip:
                c_lines = current_code.splitlines(keepends=True)
                c_strip = [l.strip() for l in c_lines]
                n_count = len(s_strip)
                c_count = len(c_lines)
                found = None
                for start in range(c_count):
                    cur = start; ok = True; spanned = []
                    for nl in s_strip:
                        while cur < c_count and not c_strip[cur]:
                            cur += 1
                        if cur >= c_count or c_strip[cur] != nl:
                            ok = False; break
                        spanned.append(cur); cur += 1
                    if ok and spanned:
                        found = (spanned[0], spanned[-1] + 1)
                        break
                if found:
                    s_idx, e_idx = found
                    char_start = sum(len(c_lines[i]) for i in range(s_idx))
                    char_end = sum(len(c_lines[i]) for i in range(e_idx))
                    # 들여쓰기 보정 (첫 라인 leading 기준)
                    first_line = c_lines[s_idx]
                    leading = first_line[:len(first_line) - len(first_line.lstrip())]
                    rep_lines = replace_block.splitlines(keepends=True)
                    if rep_lines:
                        rep_first = rep_lines[0]
                        rep_leading = rep_first[:len(rep_first) - len(rep_first.lstrip())]
                        diff = len(leading) - len(rep_leading)
                        if diff > 0:
                            rep_lines = [(" " * diff) + l for l in rep_lines]
                        elif diff < 0:
                            d_abs = -diff
                            new_rep = []
                            for l in rep_lines:
                                if l[:d_abs] == " " * d_abs:
                                    new_rep.append(l[d_abs:])
                                else:
                                    new_rep.append(l)
                            rep_lines = new_rep
                    adjusted = "".join(rep_lines)
                    new_code = current_code[:char_start] + adjusted + current_code[char_end:]
                    return True, new_code, f"🔧 {desc} (들여쓰기 보정 매칭)"
        except Exception as e:
            print(f"⚠️ [CodeEditorV2] line-by-line 매칭 실패 (graceful): {e}")

        # 모든 매칭 실패 — 후보 라인 anchor 검색
        anchor = ""
        for l in search_block.splitlines():
            ls = l.strip()
            if ls and len(ls) >= 8 and ls not in ("pass", "return", "break", "continue"):
                if not anchor or len(ls) > len(anchor):
                    anchor = ls
        candidates_msg = ""
        if anchor:
            c_lines = current_code.splitlines()
            anc_words = set(anchor.split())
            scored = []
            for i, l in enumerate(c_lines):
                ls = l.strip()
                if not ls:
                    continue
                common = len(anc_words & set(ls.split()))
                if anchor in ls or ls in anchor:
                    common += 10
                if common > 0:
                    scored.append((common, i, ls))
            scored.sort(key=lambda x: (-x[0], x[1]))
            top3 = scored[:3]
            if top3:
                candidates_msg = " (후보 라인: " + "·".join(
                    f"L{i+1}: {l[:40]}" + ("..." if len(l) > 40 else "")
                    for _, i, l in top3
                ) + ")"
        return False, current_code, f"위치 불명: {desc}{candidates_msg}"

    def _on_apply_current_clicked(self) -> None:
        """현재 선택된 proposal 1개를 코드 뷰어에 적용 (디스크 저장은 별도·undo 가능)."""
        cur_row = self._lst_proposals.currentRow()
        if cur_row < 0 or cur_row >= len(self._proposals):
            QMessageBox.warning(self, "선택 필요", "좌측 [수정사항] 에서 적용할 항목 선택.")
            return
        proposal = self._proposals[cur_row]
        current_code = self._current_code
        ok, new_code, summary = self._apply_proposal_to_code(current_code, proposal)
        if ok:
            self._push_undo(current_code, f"적용: {proposal.get('description', '수정')}")
            self._current_code = new_code
            self._btn_save_file.setEnabled(True)   # dirty
            self._lbl_current_file.setText(
                self._lbl_current_file.text().rstrip(" ·*") + " *"
            )
            self._append_chat("system", f"✅ 적용 완료 — {summary}")
        else:
            self._append_chat("error", f"적용 실패 — {summary}")

    def _on_apply_all_clicked(self) -> None:
        """모든 proposals 순서대로 시도. 실패 skip + 요약 보고·undo 가능."""
        if not self._proposals:
            return
        before_apply_all = self._current_code
        current_code = before_apply_all
        applied = 0
        skipped: list[str] = []
        for proposal in self._proposals:
            ok, current_code, summary = self._apply_proposal_to_code(current_code, proposal)
            if ok:
                applied += 1
            else:
                skipped.append(summary)
        if applied > 0:
            self._push_undo(before_apply_all, f"모두 적용 ({applied}건)")
            self._current_code = current_code
            self._btn_save_file.setEnabled(True)
            self._lbl_current_file.setText(
                self._lbl_current_file.text().rstrip(" ·*") + " *"
            )
            self._append_chat("system", f"✅ 모두 적용 — {applied}건 / skip {len(skipped)}건")
        msg = f"✅ 적용 {applied}건 / ⚠️ skip {len(skipped)}건"
        if skipped:
            msg += "\n\n[skip 상세]\n" + "\n".join(f"  • {s}" for s in skipped[:10])
            if len(skipped) > 10:
                msg += f"\n  ... 외 {len(skipped) - 10}건"
        if applied > 0:
            msg += "\n\n💾 [파일 저장] 으로 디스크에 반영하세요."
        QMessageBox.information(self, "적용 결과", msg)

    def _on_save_file_clicked(self) -> None:
        """현재 코드 뷰어 내용 → 디스크 저장."""
        if not self._current_file_path:
            QMessageBox.warning(self, "파일 미선택", "저장할 파일이 없습니다.")
            return
        try:
            # 백업 옵션 (간단·overwrite 직전 .bak 저장)
            if os.path.exists(self._current_file_path):
                bak_path = self._current_file_path + ".bak"
                try:
                    with open(self._current_file_path, "r", encoding="utf-8", errors="replace") as fr:
                        _old = fr.read()
                    with open(bak_path, "w", encoding="utf-8") as fw:
                        fw.write(_old)
                except Exception as _e_bak:
                    print(f"⚠️ [CodeEditorV2] 백업 실패 (graceful): {_e_bak}")
            # 저장
            with open(self._current_file_path, "w", encoding="utf-8") as f:
                f.write(self._current_code)
            self._btn_save_file.setEnabled(False)
            # dirty 마커 제거
            base_txt = self._lbl_current_file.text().rstrip(" ·*")
            self._lbl_current_file.setText(base_txt)
            self._append_chat(
                "system",
                f"💾 저장 완료 — {os.path.basename(self._current_file_path)} (.bak 백업 생성)"
            )
        except Exception as e:
            self._append_chat("error", f"저장 실패 — {type(e).__name__}: {e}")

    def clear_feature_suggestions(self) -> None:
        self._feature_suggestions.clear()
        self._lst_features.clear()
