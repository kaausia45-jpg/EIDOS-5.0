# eidos_more_actions_dialog.py
# 햄버거 메뉴 "🧰 더보기" — 자동화 액션 모음 다이얼로그.
#
# v2.0 (2026-05-25) — 3 탭 구조:
#   📁 프로젝트          : 좌측 스크롤 버튼 + 우측 상세 (PROJECTS 데이터)
#   🧰 자동화 액션 모음  : 좌측 스크롤 버튼 + 우측 상세 (ACTIONS 데이터)
#   🎨 액션 조립 캔버스  : v2.x 에서 ActionDispatcher schema 비주얼 빌더
#
# 테마: 순검정 배경 + 보라빛 발광 그라데이션. _LeftRightItemPanel 공용 클래스로
# 프로젝트/액션 탭이 같은 구조 재사용. 각 항목 실행 wire 는 v2.x.

import json as _json_theme
import os
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal


# [2026-05-26] 테마 감지 — settings.json 의 theme 이 "White" 포함하면 라이트.
# 모듈 import 시 1회 평가. 테마 변경 후 다이얼로그 재오픈하면 자동 반영.
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


# [2026-05-26] 라이트/다크 별 root override stylesheet — child widget 들이
# 자체 stylesheet 가 없을 때 inherit. 명시적 색상 박힌 위젯은 따로.
def _root_theme_override() -> str:
    if _LIGHT:
        return (
            "QDialog { background-color: #F7F7F8; color: #1A1A1A; }"
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
            "QListWidget {"
            "  background-color: #FFFFFF; color: #1A1A1A;"
            "  border: 1px solid #D0D0D8; border-radius: 6px;"
            "}"
            "QListWidget::item:selected { background-color: #DDD6FE; color: #1A1A1A; }"
            "QScrollArea { background-color: transparent; border: none; }"
            "QTabWidget::pane { background-color: #F7F7F8; border: 1px solid #D0D0D8; }"
            "QTabBar::tab {"
            "  background-color: #E5E7EB; color: #1A1A1A;"
            "  padding: 8px 14px; border-top-left-radius: 6px; border-top-right-radius: 6px;"
            "}"
            "QTabBar::tab:selected { background-color: #6D28D9; color: #FFFFFF; }"
        )
    return ""   # 다크는 기존 stylesheet 그대로 (override X)
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import eidos_project_store as _proj_store


# ── 색상 팔레트 (라이트/다크 동적 — _LIGHT 분기) ─────────────────────────
# [2026-05-26] _Palette 메타클래스 동적 lookup — _LIGHT 가 True 면 _PAL_LIGHT,
# 아니면 _PAL_DARK 에서 가져옴. 다이얼로그 전체 stylesheet 가 _Palette.X 를 f-string
# interpolate 하므로 매 setStyleSheet 호출마다 현재 테마 색이 박힘.
_PAL_DARK_DICT = {
    # 배경 — 순검정 가까이
    "BG_DEEP":       "#000000",
    "BG_PANEL":      "#050308",
    "BG_RIGHT":      "#000000",
    "BG_DESC":       "#000000",
    # 좌측 버튼 그라데이션 (보라 → 검정)
    "BTN_GRAD_L_NORM": "#5B1AB0",
    "BTN_GRAD_M_NORM": "#1A0F2E",
    "BTN_GRAD_R_NORM": "#000000",
    "BTN_GRAD_L_HV":   "#7C3AED",
    "BTN_GRAD_M_HV":   "#2D1850",
    "BTN_GRAD_R_HV":   "#050308",
    "BTN_GRAD_L_CHK":  "#A855F7",
    "BTN_GRAD_M_CHK":  "#3A1F66",
    "BTN_GRAD_R_CHK":  "#0A0612",
    # 보더 — 외곽 보더는 순검정 (회색 톤 완전 제거)
    "BORDER_MUTED":  "#000000",
    "BORDER_HANDLE": "#1A0F2E",
    "BORDER_HOVER":  "#7C3AED",
    "BORDER_ACCENT": "#A855F7",
    "BORDER_GLOW":   "#C084FC",
    # 액센트
    "ACCENT":        "#A855F7",
    "ACCENT_GLOW":   "#C084FC",
    "ACCENT_DEEP":   "#7C3AED",
    # 텍스트
    "TEXT_PRIMARY":  "#E9DFFB",
    "TEXT_BODY":     "#C7BAEA",
    "TEXT_MUTED":    "#8B7BA8",
    "TEXT_DISABLED": "#5B4E78",
    # 상태 라벨
    "STATUS_READY":   "#C084FC",
    "STATUS_BETA":    "#F472B6",
    "STATUS_PLANNED": "#7B6B96",
    # form 입력 배경 (helper 함수에서 옛날 하드코딩됐던 색)
    "INPUT_BG":       "#050308",
    "BTN_GHOST_BG":   "#000000",
    "BTN_PRIMARY_R":  "#1A0F2E",
    # 로그 / 통계 / tick 결과 모달 안 코드 박스
    "LOG_BG":         "#0A0612",
    "LOG_TEXT":       "#E9DFFB",
    "LOG_BORDER":     "#2A1F3D",
    # tab bar hover / selected gradient 끝
    "TAB_HOVER_BG":   "#0F0815",
    "TAB_SEL_GRAD_R": "#1A0F2E",
    # disabled 큰 버튼 (right panel run button)
    "BTN_RUN_DISABLED_BG": "#000000",
}

_PAL_LIGHT_DICT = {
    # 배경 — 흰색·연회색
    "BG_DEEP":       "#F7F7F8",
    "BG_PANEL":      "#FFFFFF",
    "BG_RIGHT":      "#FFFFFF",
    "BG_DESC":       "#FFFFFF",
    # 버튼 그라데이션 — 보라 액센트·살짝 옅게 (호버 시 더 진하게)
    "BTN_GRAD_L_NORM": "#EDE9FE",
    "BTN_GRAD_M_NORM": "#F5F3FF",
    "BTN_GRAD_R_NORM": "#FFFFFF",
    "BTN_GRAD_L_HV":   "#DDD6FE",
    "BTN_GRAD_M_HV":   "#E9E5FE",
    "BTN_GRAD_R_HV":   "#F5F3FF",
    "BTN_GRAD_L_CHK":  "#C4B5FD",
    "BTN_GRAD_M_CHK":  "#DDD6FE",
    "BTN_GRAD_R_CHK":  "#EDE9FE",
    # 보더
    "BORDER_MUTED":  "#E5E7EB",
    "BORDER_HANDLE": "#D0D0D8",
    "BORDER_HOVER":  "#7C3AED",
    "BORDER_ACCENT": "#7C3AED",
    "BORDER_GLOW":   "#6D28D9",
    # 액센트
    "ACCENT":        "#7C3AED",
    "ACCENT_GLOW":   "#6D28D9",
    "ACCENT_DEEP":   "#5B21B6",
    # 텍스트
    "TEXT_PRIMARY":  "#1A1A1A",
    "TEXT_BODY":     "#374151",
    "TEXT_MUTED":    "#6B7280",
    "TEXT_DISABLED": "#9CA3AF",
    # 상태 라벨 — 라이트에서도 식별
    "STATUS_READY":   "#6D28D9",
    "STATUS_BETA":    "#DB2777",
    "STATUS_PLANNED": "#9CA3AF",
    # form 입력 — 흰색 배경
    "INPUT_BG":       "#FFFFFF",
    "BTN_GHOST_BG":   "#FFFFFF",
    "BTN_PRIMARY_R":  "#7C3AED",
    # 로그 / 통계 / tick 결과 모달 — 라이트 위 옅은 회색 배경 + 진한 글씨
    "LOG_BG":         "#F5F5F7",
    "LOG_TEXT":       "#1A1A1A",
    "LOG_BORDER":     "#D0D0D8",
    # tab bar hover / selected
    "TAB_HOVER_BG":   "#EDE9FE",
    "TAB_SEL_GRAD_R": "#6D28D9",
    # disabled 큰 버튼
    "BTN_RUN_DISABLED_BG": "#E5E7EB",
}


def _resolve_palette_dict() -> dict:
    return _PAL_LIGHT_DICT if _LIGHT else _PAL_DARK_DICT


class _PaletteMeta(type):
    """_Palette.X 접근 시 _LIGHT 분기로 동적 lookup."""
    def __getattr__(cls, name: str) -> str:
        pal = _resolve_palette_dict()
        if name in pal:
            return pal[name]
        raise AttributeError(f"_Palette has no attribute {name!r}")


class _Palette(metaclass=_PaletteMeta):
    """다이얼로그 색상 팔레트 — _Palette.BG_DEEP 등으로 접근.
    실제 값은 _LIGHT (모듈 import 시 1회 평가) 기준."""
    pass


# ── 데이터 모델 ─────────────────────────────────────────────────────────
# 각 dict 키: id / icon / title / category / status / description
# status: "ready" / "beta" / "planned"

PROJECTS: list[dict] = [
    {
        "id": "proj_biz_ops",
        "icon": "💼",
        "title": "사업 자동화 프로젝트",
        "category": "사업",
        "status": "ready",
        "description": (
            "크몽 · 스마트스토어 · 블로그를 묶은 통합 운영 자동화.\n\n"
            "• 상품 등록 / 고객 응대 / 매출 추적 동시 진행\n"
            "• 일일 리포트 자동 생성 (.md)\n"
            "• 자율 chain 으로 6단계 워크플로우 자동 실행"
        ),
    },
    {
        "id": "proj_research_loop",
        "icon": "🔬",
        "title": "탐구주제 연구 시스템",
        "category": "연구",
        "status": "ready",
        "description": (
            "관심 주제 자동 추적 → 자료 수집 → 심층 분석 → 보고서 정기 생성.\n\n"
            "• 등록된 주제 매주 신규 자료 모니터링\n"
            "• DC Reasoner 다각도 분석\n"
            "• .md 보고서 + 시각화 자동"
        ),
    },
    {
        "id": "proj_content_pipeline",
        "icon": "🎬",
        "title": "콘텐츠 파이프라인",
        "category": "콘텐츠",
        "status": "beta",
        "description": (
            "유튜브 스크립트 · 블로그 글 · 인스타 캡션을 한 워크플로우로.\n\n"
            "• 주제 한 줄 → 3 플랫폼 동시 초안\n"
            "• 톤/길이/타겟 자동 조정\n"
            "• 게시 단계는 [[블로그/인스타]] 액션과 연동"
        ),
    },
    {
        "id": "proj_customer_care",
        "icon": "💬",
        "title": "고객 응대 자동화",
        "category": "고객대응",
        "status": "planned",
        "description": (
            "카톡 채널 · 메일 · 인스타 DM 통합 모니터링.\n\n"
            "• 미답변 문의 자동 큐 작성\n"
            "• LLM 답변 초안 + 사용자 검수\n"
            "• 답변 패턴 학습 (자주 묻는 질문 자동 분류)"
        ),
    },
    {
        "id": "proj_finance_monitor",
        "icon": "💰",
        "title": "재무 모니터링",
        "category": "관리",
        "status": "planned",
        "description": (
            "월별 매출 · 비용 · 세금 자동 집계.\n\n"
            "• 영수증 / 거래 명세 OCR\n"
            "• 카테고리 자동 분류\n"
            "• 월말 정기 보고서 + 이상치 알림"
        ),
    },
    {
        "id": "proj_learning_tracker",
        "icon": "📚",
        "title": "학습 트래킹",
        "category": "학습",
        "status": "planned",
        "description": (
            "관심 분야 신규 자료 자동 큐레이션 + 학습 로그.\n\n"
            "• 주제별 RSS / 논문 / 블로그 자동 모니터링\n"
            "• 학습 시간 / 이해도 자가 측정\n"
            "• 복습 알림 (망각 곡선 기반)"
        ),
    },
]


ACTIONS: list[dict] = [
    {
        "id": "blog_publish_naver",
        "icon": "📝",
        "title": "네이버 블로그 게시글 올리기",
        "category": "마케팅",
        "status": "beta",
        "description": (
            "주제와 키워드를 입력하면 SEO 최적화 글을 작성해 네이버 블로그 글쓰기 페이지까지 데려다줍니다.\n\n"
            "• 자동: SEO 글 작성 (제목 32자 / 본문 1200~3000자 / 태그 5~10개)\n"
            "• 자동: 네이버 글쓰기 페이지 진입 + 제목·본문·태그 클립보드 복사\n"
            "• 사용자 손: Ctrl+V 3번 + 발행 버튼 클릭\n\n"
            "한계: 미로그인 시 로그인 페이지에서 멈춤 (세션 영속화 v2 예정)."
        ),
    },
    {
        "id": "kmong_product_register",
        "icon": "🛒",
        "title": "크몽 상품페이지 등록",
        "category": "사업",
        "status": "planned",
        "description": (
            "크몽 셀러 등록 폼에 서비스 제목·카테고리·가격·옵션·상세설명을 자동 입력해 등록 직전까지 데려다줍니다.\n\n"
            "• 자동: 카테고리 선택 → 제목 / 가격 / 옵션 / 상세설명 입력\n"
            "• 사용자 손: 이미지 첨부 + 등록 버튼 클릭"
        ),
    },
    {
        "id": "research_topic_deep",
        "icon": "🔬",
        "title": "탐구주제 연구 (심층분석)",
        "category": "연구",
        "status": "ready",
        "description": (
            "주제 한 줄 입력 → 자료 수집 + 다각도 분석 + .md 보고서 자동 생성.\n\n"
            "• 자동: 검색 / 자료 정리 / DC Reasoner 심층 분석 / 보고서 .md 저장\n"
            "• 사용자 손: 주제 한 줄 입력만"
        ),
    },
    {
        "id": "smartstore_register",
        "icon": "🏪",
        "title": "스마트스토어 상품 등록",
        "category": "사업",
        "status": "planned",
        "description": (
            "네이버 스마트스토어 상품 등록 폼에 카테고리·제목·가격·상세설명·옵션 자동 입력.\n\n"
            "• 자동: 카테고리 매핑 / 폼 입력 / 옵션 조합\n"
            "• 사용자 손: 대표 이미지 첨부 + 검수 + 등록 버튼"
        ),
    },
    {
        "id": "gmail_send",
        "icon": "📧",
        "title": "Gmail 메일 발송",
        "category": "통신",
        "status": "planned",
        "description": (
            "받는 사람·제목·본문을 입력하면 Gmail 작성 화면까지 데려다줍니다.\n\n"
            "• 자동: Gmail 작성 화면 진입 + 받는 사람 / 제목 / 본문 자동 입력\n"
            "• 사용자 손: 첨부 (있다면) + 발송 버튼 클릭"
        ),
    },
    {
        "id": "kakao_channel_reply",
        "icon": "💬",
        "title": "카카오톡 채널 고객 답변",
        "category": "고객대응",
        "status": "planned",
        "description": (
            "카톡 채널 관리자센터의 고객 문의에 LLM 답변을 작성해 채팅창 진입 + 답변 입력까지.\n\n"
            "• 자동: 미답변 문의 목록 진입 / 답변 LLM 작성 / 입력창 채우기\n"
            "• 사용자 손: 답변 검수 + 전송 버튼 (개별 메시지 확인)"
        ),
    },
    {
        "id": "youtube_script",
        "icon": "🎬",
        "title": "유튜브 영상 스크립트 작성",
        "category": "콘텐츠",
        "status": "ready",
        "description": (
            "주제·길이·톤을 입력하면 인트로 / 본론 / CTA 구조의 스크립트를 .md 로 생성합니다.\n\n"
            "• 자동: 후크 / 본론 / 정리 / CTA 4단 구조 작성\n"
            "• 사용자 손: 주제·길이·톤 입력만"
        ),
    },
    {
        "id": "instagram_post",
        "icon": "📷",
        "title": "인스타그램 게시물 작성",
        "category": "마케팅",
        "status": "planned",
        "description": (
            "주제 입력 → 캡션 작성 + 해시태그 자동 + 인스타 게시 페이지 진입.\n\n"
            "• 자동: 캡션 / 해시태그 10~20개 / 인스타 게시 페이지 진입 + 캡션 클립보드 복사\n"
            "• 사용자 손: 이미지 첨부 + Ctrl+V + 공유 버튼"
        ),
    },
    {
        "id": "business_idea",
        "icon": "💡",
        "title": "사업 아이템 떠올리기",
        "category": "사업",
        "status": "ready",
        "description": (
            "EIDOS 가 ToM-core belief 에서 사용자 강점·관심·자본·시간을 자동 추출 →\n"
            "그 프로필에 맞춤형 사업 아이템 5개 + 각 1-pager (타겟·가치 제안·수익·차별화·\n"
            "리스크·첫 7일 step) 를 한 번에 생성.\n\n"
            "• 자동: belief 추출 → LLM brainstorm → 5개 카드\n"
            "• 사용자 손: 자본·시간·제약 수정 (form prefilled)"
        ),
    },
]


# ── 상태 → 라벨/색상 매핑 ───────────────────────────────────────────────
_STATUS_META = {
    "ready":   {"label": "✅ 동작",     "color": _Palette.STATUS_READY},
    "beta":    {"label": "🟡 일부 동작", "color": _Palette.STATUS_BETA},
    "planned": {"label": "⏳ 예정",     "color": _Palette.STATUS_PLANNED},
}


def _status_label(status: str) -> str:
    return _STATUS_META.get(status, _STATUS_META["planned"])["label"]


def _status_color(status: str) -> str:
    return _STATUS_META.get(status, _STATUS_META["planned"])["color"]


def _apply_purple_glow(widget: QWidget, radius: int = 24, color_hex: str = "#A855F7", alpha: int = 160) -> None:
    """위젯에 보라색 발광 그림자 효과 적용."""
    try:
        eff = QGraphicsDropShadowEffect(widget)
        eff.setBlurRadius(radius)
        c = QColor(color_hex)
        c.setAlpha(alpha)
        eff.setColor(c)
        eff.setOffset(0, 0)
        widget.setGraphicsEffect(eff)
    except Exception as e:
        print(f"[MoreActionsDialog] glow effect 적용 실패 (graceful): {e}")


# ── 공용 컴포넌트 ───────────────────────────────────────────────────────

class _ItemButton(QPushButton):
    """가로 길쭉한 항목 버튼 — 좌측 진한 보라 → 우측 검정 그라데이션."""

    def __init__(self, item_data: dict, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.item_data = item_data
        self.setFixedHeight(58)
        self.setCursor(Qt.PointingHandCursor)
        self.setCheckable(True)
        self.setAutoExclusive(True)
        text = f"  {item_data.get('icon', '·')}   {item_data.get('title', '(이름 없음)')}"
        self.setText(text)
        self.setToolTip(
            f"{item_data.get('title','')}\n"
            f"카테고리: {item_data.get('category','-')}\n"
            f"상태: {_status_label(item_data.get('status','planned'))}"
        )
        self.setStyleSheet(
            f"QPushButton {{"
            f"  text-align: left;"
            f"  padding: 8px 14px;"
            f"  border: 1px solid {_Palette.BORDER_MUTED};"
            f"  border-radius: 8px;"
            f"  background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            f"    stop:0 {_Palette.BTN_GRAD_L_NORM},"
            f"    stop:0.55 {_Palette.BTN_GRAD_M_NORM},"
            f"    stop:1 {_Palette.BTN_GRAD_R_NORM});"
            f"  color: {_Palette.TEXT_PRIMARY};"
            f"  font-size: 13px;"
            f"}}"
            f"QPushButton:hover {{"
            f"  background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            f"    stop:0 {_Palette.BTN_GRAD_L_HV},"
            f"    stop:0.55 {_Palette.BTN_GRAD_M_HV},"
            f"    stop:1 {_Palette.BTN_GRAD_R_HV});"
            f"  border: 1px solid {_Palette.BORDER_HOVER};"
            f"  color: {_Palette.ACCENT_GLOW};"
            f"}}"
            f"QPushButton:checked {{"
            f"  background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            f"    stop:0 {_Palette.BTN_GRAD_L_CHK},"
            f"    stop:0.55 {_Palette.BTN_GRAD_M_CHK},"
            f"    stop:1 {_Palette.BTN_GRAD_R_CHK});"
            f"  border: 2px solid {_Palette.BORDER_GLOW};"
            f"  color: #ffffff;"
            f"  font-weight: 600;"
            f"}}"
        )


# ── 액션 핸들러 매핑 (forward declaration — 파일 끝에서 update) ──────────
# id → callable(parent_dialog: QDialog, item_data: dict) -> None
ACTION_HANDLERS: dict = {}


class _LeftRightItemPanel(QWidget):
    """좌측 스크롤 버튼 + 우측 상세 패널 공용 컴포넌트.
    프로젝트 / 액션 탭에서 같은 구조 재사용."""

    item_selected = Signal(dict)

    def __init__(self, items: list[dict], run_btn_label: str = "▶  실행  (v2 에서 wire 예정)",
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._items = items
        self._run_btn_label = run_btn_label
        self._buttons: list[_ItemButton] = []
        self._current_item: Optional[dict] = None
        self._build_ui()
        self._populate()
        if self._buttons:
            self._buttons[0].setChecked(True)
            self._on_clicked(self._buttons[0])

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet(
            f"QSplitter::handle {{ background: {_Palette.BORDER_HANDLE}; }}"
            f"QSplitter::handle:hover {{ background: {_Palette.ACCENT}; }}"
        )

        # 좌측 — 스크롤 버튼 리스트
        left_panel = QFrame(splitter)
        left_panel.setObjectName("LRPanel_Left")
        left_panel.setStyleSheet(
            f"QFrame#LRPanel_Left {{"
            f"  background: {_Palette.BG_PANEL};"
            f"  border: 1px solid {_Palette.BORDER_MUTED};"
            f"  border-radius: 10px;"
            f"}}"
        )
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        self._scroll_area = QScrollArea(left_panel)
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll_area.setStyleSheet(
            f"QScrollArea {{ border: none; background: transparent; }}"
            f"QScrollBar:vertical {{"
            f"  background: transparent;"
            f"  width: 10px;"
            f"  margin: 4px 2px 4px 0;"
            f"}}"
            f"QScrollBar::handle:vertical {{"
            f"  background: {_Palette.BORDER_HANDLE};"
            f"  border-radius: 5px;"
            f"  min-height: 30px;"
            f"}}"
            f"QScrollBar::handle:vertical:hover {{ background: {_Palette.ACCENT}; }}"
            f"QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}"
            f"QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}"
        )

        self._scroll_content = QWidget()
        self._scroll_content.setStyleSheet("background: transparent;")
        self._buttons_layout = QVBoxLayout(self._scroll_content)
        self._buttons_layout.setContentsMargins(10, 10, 10, 10)
        self._buttons_layout.setSpacing(7)
        self._buttons_layout.addStretch(1)
        self._scroll_area.setWidget(self._scroll_content)
        left_layout.addWidget(self._scroll_area)

        # 우측 — 상세 패널
        right_panel = QFrame(splitter)
        right_panel.setObjectName("LRPanel_Right")
        right_panel.setStyleSheet(
            f"QFrame#LRPanel_Right {{"
            f"  background: {_Palette.BG_RIGHT};"
            f"  border: 1px solid {_Palette.BORDER_MUTED};"
            f"  border-radius: 10px;"
            f"}}"
        )
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(22, 22, 22, 22)
        right_layout.setSpacing(12)

        self._detail_title = QLabel("항목을 선택하세요")
        self._detail_title.setFont(QFont("", 16, QFont.Bold))
        self._detail_title.setStyleSheet(
            f"color: {_Palette.ACCENT_GLOW}; letter-spacing: 0.3px; background: transparent;"
        )
        self._detail_title.setWordWrap(True)
        _apply_purple_glow(self._detail_title, radius=18, color_hex=_Palette.ACCENT, alpha=110)
        right_layout.addWidget(self._detail_title)

        self._detail_meta = QLabel("")
        self._detail_meta.setStyleSheet(
            f"color: {_Palette.TEXT_MUTED}; font-size: 12px; background: transparent;"
        )
        right_layout.addWidget(self._detail_meta)

        self._detail_status = QLabel("")
        self._detail_status.setStyleSheet(
            "font-size: 12px; padding: 4px 0; background: transparent;"
        )
        right_layout.addWidget(self._detail_status)

        self._detail_desc = QLabel("")
        self._detail_desc.setStyleSheet(
            f"color: {_Palette.TEXT_BODY};"
            f" font-size: 13px;"
            f" line-height: 1.6;"
            f" background: {_Palette.BG_DESC};"
            f" padding: 16px;"
            f" border: 1px solid {_Palette.BORDER_MUTED};"
            f" border-radius: 8px;"
        )
        self._detail_desc.setWordWrap(True)
        self._detail_desc.setAlignment(Qt.AlignTop)
        self._detail_desc.setTextInteractionFlags(Qt.TextSelectableByMouse)
        right_layout.addWidget(self._detail_desc, 1)

        self._run_btn = QPushButton(self._run_btn_label)
        self._run_btn.setEnabled(False)
        self._run_btn.setFixedHeight(42)
        self._run_btn.setCursor(Qt.PointingHandCursor)
        self._run_btn.setStyleSheet(
            f"QPushButton {{"
            f"  background: {_Palette.BTN_RUN_DISABLED_BG};"
            f"  color: {_Palette.TEXT_DISABLED};"
            f"  border: 1px solid {_Palette.BORDER_MUTED};"
            f"  border-radius: 8px;"
            f"  font-size: 13px;"
            f"  padding: 0 22px;"
            f"  font-weight: 600;"
            f"}}"
            f"QPushButton:enabled {{"
            f"  background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            f"    stop:0 {_Palette.ACCENT}, stop:1 {_Palette.BTN_PRIMARY_R});"
            f"  color: #ffffff;"
            f"  border: 1px solid {_Palette.BORDER_GLOW};"
            f"}}"
            f"QPushButton:enabled:hover {{"
            f"  background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            f"    stop:0 {_Palette.ACCENT_GLOW}, stop:1 {_Palette.ACCENT_DEEP});"
            f"  border: 1px solid {_Palette.ACCENT_GLOW};"
            f"}}"
        )
        self._run_btn.clicked.connect(self._on_run_clicked)
        right_layout.addWidget(self._run_btn, 0, Qt.AlignRight)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([330, 650])

        root.addWidget(splitter, 1)

    def _populate(self) -> None:
        insert_idx = self._buttons_layout.count() - 1
        for item in self._items:
            btn = _ItemButton(item, self._scroll_content)
            btn.clicked.connect(lambda _checked=False, b=btn: self._on_clicked(b))
            self._buttons_layout.insertWidget(insert_idx, btn)
            insert_idx += 1
            self._buttons.append(btn)

    def _on_clicked(self, btn: _ItemButton) -> None:
        data = btn.item_data
        self._current_item = data
        try:
            self._detail_title.setText(
                f"{data.get('icon','·')}   {data.get('title','(이름 없음)')}"
            )
            self._detail_meta.setText(
                f"카테고리: {data.get('category','-')}    ·    id: {data.get('id','-')}"
            )
            status = data.get("status", "planned")
            self._detail_status.setText(_status_label(status))
            self._detail_status.setStyleSheet(
                f"color: {_status_color(status)}; font-size: 12px; padding: 4px 0; "
                f"font-weight: 600; background: transparent;"
            )
            self._detail_desc.setText(data.get("description", ""))
            # 실행 버튼: ACTION_HANDLERS 등록 OR canvas:NAME 패턴 (사용자 캔버스).
            _id = data.get("id", "") or ""
            has_handler = (_id in ACTION_HANDLERS) or _id.startswith("canvas:")
            self._run_btn.setEnabled(has_handler)
            self._run_btn.setText("▶  실행" if has_handler else self._run_btn_label)
        except Exception as e:
            print(f"[LRPanel] 우측 패널 갱신 실패: {e}")
        try:
            self.item_selected.emit(data)
        except Exception:
            pass

    def _on_run_clicked(self) -> None:
        data = getattr(self, "_current_item", None)
        if not data:
            return
        _id = data.get("id", "") or ""
        # canvas:NAME 사용자 캔버스 → 전용 핸들러
        if _id.startswith("canvas:"):
            handler = _run_user_canvas
        else:
            handler = ACTION_HANDLERS.get(_id)
        if not handler:
            QMessageBox.information(
                self, "실행 — v2 예정",
                "이 항목의 실제 실행은 v2.x 에서 ActionDispatcher schema 와 연결됩니다.",
            )
            return
        # parent QDialog (MoreActionsDialog) 찾기 — handler 가 다이얼로그 컨텍스트 필요.
        p: Optional[QWidget] = self
        while p is not None and not isinstance(p, QDialog):
            p = p.parent()
        try:
            handler(p or self, data)
        except Exception as e:
            print(f"[LRPanel] handler 실행 실패: {e}")
            QMessageBox.warning(self, "실행 오류", f"{type(e).__name__}: {e}")


class TickProgressDialog(QDialog):
    """[Phase 1-E 2026-05-26] N tick 자동 루프 실시간 진행 다이얼로그.

    매 progress event 마다 append_event(event, payload) 호출 — main thread 만.
    "🛑 중단" 클릭 → request_stop 시그널 → _CanvasTab 의 stop_event 발화.
    종료 시 "닫기" 버튼 활성화.
    """

    request_stop = Signal()   # _CanvasTab 이 connect 해서 stop_event.set 위임

    def __init__(self, parent: Optional[QWidget] = None, title: str = "🔄 Stage 진행",
                 max_ticks: int = 1, safety_mode: str = "normal"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(760, 600)
        self.setMinimumSize(560, 400)
        self._max_ticks = max_ticks
        self._is_running = True

        try:
            self.setStyleSheet(
                f"QDialog {{ background: {_Palette.BG_DEEP}; color: {_Palette.TEXT}; }}"
            )
        except Exception:
            pass

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(8)

        # 헤더 라벨 — 현재 진행 상태
        self._lbl_status = QLabel(
            f"⏳ 시작 중...   ·   max_ticks={max_ticks}   ·   safety_mode={safety_mode}"
        )
        try:
            self._lbl_status.setStyleSheet(
                f"color: {_Palette.ACCENT_GLOW}; font-size: 13px; font-weight: 600;"
                f" padding: 6px 4px;"
            )
        except Exception:
            self._lbl_status.setStyleSheet("color: #C084FC; font-size: 13px;")
        root.addWidget(self._lbl_status)

        # 안전 모드 경고 (yolo 시)
        if safety_mode == "yolo":
            warn = QLabel("⚠️ YOLO 모드 — 외부효과 (이메일·텔레그램·웹) 실 발생·되돌릴 수 없습니다.")
            warn.setStyleSheet("color: #F87171; font-size: 11px; padding: 4px;")
            root.addWidget(warn)

        # 로그 영역
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        try:
            self._log.setStyleSheet(
                f"QPlainTextEdit {{ background: {_Palette.LOG_BG}; color: {_Palette.LOG_TEXT};"
                f" border: 1px solid {_Palette.LOG_BORDER}; border-radius: 8px;"
                f" font-family: 'Consolas','Courier New',monospace; font-size: 11px;"
                f" padding: 10px; }}"
            )
        except Exception:
            self._log.setStyleSheet(
                f"background: {_Palette.LOG_BG}; color: {_Palette.LOG_TEXT}; padding: 8px;"
            )
        root.addWidget(self._log, 1)

        # 하단 버튼
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._btn_stop = QPushButton("🛑  중단")
        self._btn_close = QPushButton("닫기")
        self._btn_close.setEnabled(False)
        try:
            self._btn_stop.setStyleSheet(_small_btn_style(primary=True))
            self._btn_close.setStyleSheet(_small_btn_style(primary=False))
            self._btn_stop.setCursor(Qt.PointingHandCursor)
            self._btn_close.setCursor(Qt.PointingHandCursor)
        except Exception:
            pass
        self._btn_stop.clicked.connect(self._on_stop_clicked)
        self._btn_close.clicked.connect(self.accept)
        btn_row.addWidget(self._btn_stop)
        btn_row.addWidget(self._btn_close)
        root.addLayout(btn_row)

    def append_event(self, event: str, payload: dict) -> None:
        """[Phase 1-E] worker thread 가 _tick_progress_signal emit → _CanvasTab 이 슬롯에서 호출."""
        if not isinstance(payload, dict):
            payload = {}
        ts = _dt_iso()
        try:
            if event == "loop_tick_about_to_run":
                tick_idx = int(payload.get("tick_idx", -1))
                self._lbl_status.setText(
                    f"⏳ tick {tick_idx + 1} / {self._max_ticks} 진행 중..."
                )
                self._log.appendPlainText(f"\n━━━ Tick {tick_idx + 1} ━━━")
            elif event == "tick_started":
                self._log.appendPlainText(f"  · tick 초기화")
            elif event == "llm_call_start":
                self._log.appendPlainText(f"  🤖  LLM 분석·결정 중...")
            elif event == "decision_made":
                aid = payload.get("action_id", "?")
                self._log.appendPlainText(f"  ✨  EIDOS 결정: {aid}")
            elif event == "execution_done":
                status = payload.get("status", "?")
                dry = payload.get("dry_run", False)
                aid = payload.get("action_id", "?")
                result = (payload.get("result") or "")[:160]
                emoji = {"ok": "✅", "fail": "❌", "dry_run": "🛡", "skip": "⏭",
                         "hitl": "💬"}.get(status, "·")
                dry_tag = " [DRY-RUN]" if dry else ""
                self._log.appendPlainText(
                    f"  {emoji}  실행 {status}{dry_tag}  →  {result}"
                )
            elif event == "tick_done":
                pass   # decision_made + execution_done 으로 이미 표시
            elif event == "loop_done":
                self._is_running = False
                ticks = payload.get("ticks", "?")
                reason = payload.get("halt_reason", "?")
                self._lbl_status.setText(f"✅ 종료 — {ticks} tick 실행 / {reason}")
                self._log.appendPlainText(f"\n━━━ 종료: {reason} ━━━")
                self._btn_stop.setEnabled(False)
                self._btn_stop.setText("🛑 종료됨")
                self._btn_close.setEnabled(True)
                self._btn_close.setDefault(True)
            else:
                # 미지 event — 그래도 로그 (디버그)
                self._log.appendPlainText(f"  [{event}] {payload}")
            # 자동 scroll
            sb = self._log.verticalScrollBar()
            if sb:
                sb.setValue(sb.maximum())
        except Exception as e:
            print(f"[TickProgressDialog] append_event 실패 (graceful): {e}")

    def append_text(self, text: str) -> None:
        """외부에서 임의 텍스트 추가 (예: error / 최종 summary)."""
        try:
            self._log.appendPlainText(text)
            sb = self._log.verticalScrollBar()
            if sb:
                sb.setValue(sb.maximum())
        except Exception:
            pass

    def _on_stop_clicked(self) -> None:
        if not self._is_running:
            self.accept()
            return
        self._btn_stop.setEnabled(False)
        self._btn_stop.setText("⏸ 중단 요청 보냄...")
        self._lbl_status.setText("⏸ 중단 신호 — 다음 tick 시작 전 안전 종료 예정...")
        self.request_stop.emit()


def _dt_iso() -> str:
    """현재 시각 짧은 iso (분 단위)."""
    import datetime as _dtm
    return _dtm.datetime.now().strftime("%H:%M:%S")


class _CanvasTab(QWidget):
    """🎨 액션 조립 캔버스 — 좌측 12 verb 팔레트 + 가운데 QGraphicsView + 우측 inspector + 하단 버튼.

    v2.1 (2026-05-25): 실행 wire 가 eidos_canvas_runner.execute_canvas_async 직접 호출 —
    ActionDispatcher Vision 루프 우회·execution_module 함수 직접 호출 (NAVIGATE/CLICK/FILL/...).
    노드 색상이 실행 상태로 변화 (running 노랑·ok 초록·fail 빨강·skip/warn).
    """

    # thread-safe 노드 색상 update — worker thread → main thread (QueuedConnection)
    _canvas_progress = Signal(str, str, dict)   # event, node_id, payload
    # v2.2 ASK_USER UI 다이얼로그 — worker thread 가 main thread 에 질문 emit
    _ask_user_signal = Signal(str, object)      # question, asyncio.Future
    # [Phase 0-D 2026-05-26] Stage tick 결과 — worker thread → main thread
    _tick_done_signal = Signal(dict)            # tick result dict
    # [Phase 1-E 2026-05-26] tick 진행 이벤트 — TickProgressDialog 실시간 갱신용
    _tick_progress_signal = Signal(str, dict)   # event, payload

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        try:
            from eidos_canvas_widget import _CanvasScene as _CS, _CanvasView as _CV
            self._CanvasScene = _CS
            self._CanvasView = _CV
            self._canvas_ok = True
        except Exception as e:
            print(f"⚠️ [CanvasTab] 캔버스 위젯 로드 실패 (graceful): {e}")
            self._canvas_ok = False
        self._current_node_id: Optional[str] = None
        self._suppress_inspector_save = False
        self._build_ui()
        if self._canvas_ok:
            # 시작 시 빈 캔버스
            import eidos_canvas_store as _cstore
            self._current_canvas_name = "새 캔버스"
            self._cstore = _cstore
        # 진행 시그널 connect — main thread 에서 노드 색상 update (worker thread 안전)
        try:
            self._canvas_progress.connect(self._on_progress_main, Qt.QueuedConnection)
            self._ask_user_signal.connect(self._on_ask_user_main, Qt.QueuedConnection)
            # [Phase 0-D] tick 결과 — worker thread → main thread (QueuedConnection)
            self._tick_done_signal.connect(self._on_tick_done_main, Qt.QueuedConnection)
            # [Phase 1-E] tick 진행 이벤트 — 실시간 다이얼로그 갱신
            self._tick_progress_signal.connect(self._on_tick_progress_main, Qt.QueuedConnection)
        except Exception as _e_sig:
            print(f"[CanvasTab] progress signal connect 실패 (graceful): {_e_sig}")
        # v2.2 stop 인프라
        self._stop_event = None       # asyncio.Event (worker loop 에서 생성)
        self._is_running = False
        self._worker_loop_ref = None
        # [Phase 0-D 2026-05-26] 현재 캔버스에 로드된 stage_id (Phase 0-C 의 stage_applied 또는
        # ToM stage 직접 로드 시 세팅됨). None 이면 "🎯 1 tick" 비활성.
        self._current_stage_id: Optional[str] = None

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        if not self._canvas_ok:
            # 캔버스 위젯 로드 실패 — 안내 표시
            lbl = QLabel("⚠️  캔버스 위젯 로드 실패 — eidos_canvas_widget 모듈 확인")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(f"color: {_Palette.STATUS_BETA}; padding: 40px; background: transparent;")
            root.addWidget(lbl)
            return

        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet(
            f"QSplitter::handle {{ background: {_Palette.BORDER_HANDLE}; }}"
            f"QSplitter::handle:hover {{ background: {_Palette.ACCENT}; }}"
        )

        # [2026-05-26 사용자 요청] 좌측 verb 팔레트 제거. 노드 추가는 가운데 캔버스에서
        # 우클릭 컨텍스트 메뉴 또는 기존 캔버스 로드. cstore 는 다른 메서드에서 lazy import.
        import eidos_canvas_store as _cstore
        self._palette_buttons: list[QPushButton] = []   # 후방호환 빈 list (다른 메서드 참조)

        # 가운데: 캔버스 + 하단 버튼
        center_wrap = QFrame(splitter)
        center_wrap.setObjectName("CanvasTab_Center")
        center_wrap.setStyleSheet(
            f"QFrame#CanvasTab_Center {{ background: {_Palette.BG_DEEP};"
            f" border: 1px solid {_Palette.BORDER_MUTED}; border-radius: 10px; }}"
        )
        center_layout = QVBoxLayout(center_wrap)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        # 캔버스
        self._scene = self._CanvasScene(self)
        self._scene.node_selected.connect(self._on_node_selected)
        self._scene.node_changed.connect(self._on_canvas_dirty)
        self._view = self._CanvasView(self._scene, center_wrap)
        center_layout.addWidget(self._view, 1)

        # 하단 버튼 바
        bottom = QHBoxLayout()
        bottom.setContentsMargins(8, 6, 8, 8)
        bottom.setSpacing(6)
        self._fld_canvas_name = QLineEdit("새 캔버스")
        self._fld_canvas_name.setFixedHeight(32)
        self._fld_canvas_name.setStyleSheet(_form_input_style())
        bottom.addWidget(self._fld_canvas_name, 1)
        self._btn_new = QPushButton("🆕  새로")
        self._btn_save = QPushButton("💾  저장")
        self._btn_load = QPushButton("📂  불러오기")
        self._btn_layout = QPushButton("📐  정렬")
        self._btn_snap = QPushButton("🧲  snap")
        self._btn_snap.setCheckable(True)
        self._btn_snap.setChecked(True)
        self._btn_run = QPushButton("▶  실행")
        # [Phase 0-C 2026-05-26] 🎭 새 stage 자동 제안 다이얼로그
        self._btn_new_stage = QPushButton("🎭  새 stage")
        # [Phase 0-D 2026-05-26] 🎯 Stage 1 tick — ToM 에이전트 단발 실행
        self._btn_tick = QPushButton("🎯  1 tick")
        self._btn_tick.setEnabled(False)   # stage 로드되어야 활성화
        self._btn_tick.setToolTip("로드된 Stage 의 1 tick 실행 (ToM 예측 + EIDOS 결정 + dry-run/실 dispatch)")
        # [Phase 1-A 2026-05-26] 🔄 Stage N tick 자동 — 종료 조건까지 자동 루프
        self._btn_tick_n = QPushButton("🔄  N tick")
        self._btn_tick_n.setEnabled(False)
        self._btn_tick_n.setToolTip(
            "Stage 의 N tick 자동 실행 (종료 조건: max_ticks·중단·연속 동일 action 3회·error). "
            "실행 중 다시 누르면 중단 신호."
        )
        # [Phase 1-D 2026-05-26] 🛡 safety_mode 토글 — paranoid/normal/yolo
        self._btn_safety = QPushButton("🛡  safety: —")
        self._btn_safety.setEnabled(False)
        self._btn_safety.setToolTip(
            "Stage 안전 모드 변경:\n"
            "• paranoid — ask_user 만 실 실행·모두 dry-run\n"
            "• normal — meta+llm.write+ask_user 실·외부효과 dry-run (default)\n"
            "• yolo — 모두 실 실행 (⚠️ 회복 불가)"
        )
        # [Phase 2-B 2026-05-26] ⏰ 자동 scheduler — 정주기 자동 tick
        self._btn_schedule = QPushButton("⏰  자동")
        self._btn_schedule.setEnabled(False)
        self._btn_schedule.setToolTip(
            "Stage 정주기 자동 실행 — interval (초) + max_ticks 입력\n"
            "(N tick 즉시 실행과 달리 매 interval 마다 sleep 후 다음 tick)\n"
            "실행 중 다시 누르면 중단 신호 — 진행 다이얼로그에서도 [🛑 중단] 가능."
        )
        # [Phase 3-C 2026-05-26] 📊 PatternModel 통계 보기
        self._btn_stats = QPushButton("📊  통계")
        self._btn_stats.setEnabled(False)
        self._btn_stats.setToolTip(
            "현재 Stage 의 누적 통계 (PatternModel) — EIDOS 결정 빈도·외부 actor 예측·"
            "1-step chain·belief-conditional 패턴 (median 분할)·실패 경고."
        )
        for b in (self._btn_new, self._btn_save, self._btn_load, self._btn_layout, self._btn_snap,
                  self._btn_new_stage, self._btn_safety, self._btn_stats,
                  self._btn_tick, self._btn_tick_n, self._btn_schedule):
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(_small_btn_style(primary=False))
            bottom.addWidget(b)
        self._btn_run.setCursor(Qt.PointingHandCursor)
        self._btn_run.setStyleSheet(_small_btn_style(primary=True))
        bottom.addWidget(self._btn_run)
        self._btn_new.clicked.connect(self._on_new_clicked)
        self._btn_save.clicked.connect(self._on_save_clicked)
        self._btn_load.clicked.connect(self._on_load_clicked)
        self._btn_layout.clicked.connect(self._on_auto_layout_clicked)
        self._btn_snap.toggled.connect(self._on_snap_toggled)
        self._btn_run.clicked.connect(self._on_run_clicked)
        self._btn_new_stage.clicked.connect(self._on_new_stage_clicked)
        self._btn_tick.clicked.connect(self._on_run_stage_tick_clicked)
        self._btn_tick_n.clicked.connect(self._on_run_stage_loop_clicked)
        self._btn_safety.clicked.connect(self._on_safety_clicked)
        self._btn_schedule.clicked.connect(self._on_run_stage_schedule_clicked)
        self._btn_stats.clicked.connect(self._on_stats_clicked)
        center_layout.addLayout(bottom)

        # v2.3 단축키 — Ctrl+Z/Y (undo/redo)·Ctrl+C/V (복사/붙여넣기)·Delete (선택 노드 삭제)
        from PySide6.QtGui import QShortcut, QKeySequence
        self._sc_undo = QShortcut(QKeySequence.Undo, self)
        self._sc_redo = QShortcut(QKeySequence.Redo, self)
        self._sc_copy = QShortcut(QKeySequence.Copy, self)
        self._sc_paste = QShortcut(QKeySequence.Paste, self)
        self._sc_delete = QShortcut(QKeySequence.Delete, self)
        self._sc_undo.activated.connect(self._on_undo)
        self._sc_redo.activated.connect(self._on_redo)
        self._sc_copy.activated.connect(self._on_copy)
        self._sc_paste.activated.connect(self._on_paste)
        self._sc_delete.activated.connect(self._on_delete_selected)

        # 우측: inspector
        right = QFrame(splitter)
        right.setObjectName("CanvasTab_Right")
        right.setStyleSheet(
            f"QFrame#CanvasTab_Right {{ background: {_Palette.BG_RIGHT};"
            f" border: 1px solid {_Palette.BORDER_MUTED}; border-radius: 10px; }}"
        )
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(10)

        self._ins_header = QLabel("⚙️  노드를 선택하세요")
        self._ins_header.setFont(QFont("", 13, QFont.Bold))
        self._ins_header.setStyleSheet(
            f"color: {_Palette.ACCENT_GLOW}; background: transparent;"
        )
        _apply_purple_glow(self._ins_header, radius=14, color_hex=_Palette.ACCENT, alpha=100)
        right_layout.addWidget(self._ins_header)

        # ── [Phase 0-B] verb 노드 form (옛 그대로) ──
        self._verb_form_widgets: list = []

        self._lbl_ins_target = self._small_label("🎯 target")
        right_layout.addWidget(self._lbl_ins_target)
        self._ins_target = QLineEdit()
        self._ins_target.setStyleSheet(_form_input_style())
        self._ins_target.textChanged.connect(self._on_inspector_changed)
        right_layout.addWidget(self._ins_target)
        self._verb_form_widgets += [self._lbl_ins_target, self._ins_target]

        self._lbl_ins_content = self._small_label("📝 content")
        right_layout.addWidget(self._lbl_ins_content)
        self._ins_content = QPlainTextEdit()
        self._ins_content.setFixedHeight(120)
        self._ins_content.setStyleSheet(_form_input_style())
        self._ins_content.textChanged.connect(self._on_inspector_changed)
        right_layout.addWidget(self._ins_content)
        self._verb_form_widgets += [self._lbl_ins_content, self._ins_content]

        self._lbl_ins_reason = self._small_label("💡 reason (이 단계 이유)")
        right_layout.addWidget(self._lbl_ins_reason)
        self._ins_reason = QLineEdit()
        self._ins_reason.setStyleSheet(_form_input_style())
        self._ins_reason.textChanged.connect(self._on_inspector_changed)
        right_layout.addWidget(self._ins_reason)
        self._verb_form_widgets += [self._lbl_ins_reason, self._ins_reason]

        # ── [Phase 0-B 2026-05-26] actor 노드 form ──
        # ToM 무대의 actor 노드 (kind="actor") 가 선택되면 verb form 숨기고 이쪽 표시.
        self._actor_form_widgets: list = []

        self._lbl_actor_name = self._small_label("🎭 행위자 이름")
        right_layout.addWidget(self._lbl_actor_name)
        self._ins_actor_name = QLineEdit()
        self._ins_actor_name.setStyleSheet(_form_input_style())
        self._ins_actor_name.textChanged.connect(self._on_inspector_changed)
        right_layout.addWidget(self._ins_actor_name)
        self._actor_form_widgets += [self._lbl_actor_name, self._ins_actor_name]

        self._lbl_actor_type = self._small_label("🏷  행위자 유형")
        right_layout.addWidget(self._lbl_actor_type)
        self._ins_actor_type = QComboBox()
        # ACTOR_TYPES 와 1:1 — eidos_agent_stage_store 에서 import
        try:
            from eidos_agent_stage_store import ACTOR_TYPES as _AT
            for t in _AT:
                self._ins_actor_type.addItem(t)
        except Exception:
            for t in ["self", "human", "organization", "system", "agent", "abstract"]:
                self._ins_actor_type.addItem(t)
        self._ins_actor_type.setStyleSheet(_form_input_style())
        self._ins_actor_type.currentTextChanged.connect(self._on_inspector_changed)
        right_layout.addWidget(self._ins_actor_type)
        self._actor_form_widgets += [self._lbl_actor_type, self._ins_actor_type]

        self._lbl_actor_role = self._small_label("⭐ role (self = EIDOS 자기 자신)")
        right_layout.addWidget(self._lbl_actor_role)
        self._ins_actor_role = QComboBox()
        self._ins_actor_role.addItem("other")
        self._ins_actor_role.addItem("self")
        self._ins_actor_role.setStyleSheet(_form_input_style())
        self._ins_actor_role.currentTextChanged.connect(self._on_inspector_changed)
        right_layout.addWidget(self._ins_actor_role)
        self._actor_form_widgets += [self._lbl_actor_role, self._ins_actor_role]

        self._lbl_actor_ind = self._small_label("📊 indicators (key: value, 줄별)")
        right_layout.addWidget(self._lbl_actor_ind)
        self._ins_actor_indicators = QPlainTextEdit()
        self._ins_actor_indicators.setFixedHeight(80)
        self._ins_actor_indicators.setPlaceholderText(
            "urgency: 0.7\nbudget: 500000\ntrust: 0.5"
        )
        self._ins_actor_indicators.setStyleSheet(_form_input_style())
        self._ins_actor_indicators.textChanged.connect(self._on_inspector_changed)
        right_layout.addWidget(self._ins_actor_indicators)
        self._actor_form_widgets += [self._lbl_actor_ind, self._ins_actor_indicators]

        self._lbl_actor_rep = self._small_label("⚡ action_repertoire (action_id, 줄별)")
        right_layout.addWidget(self._lbl_actor_rep)
        self._ins_actor_repertoire = QPlainTextEdit()
        self._ins_actor_repertoire.setFixedHeight(80)
        self._ins_actor_repertoire.setPlaceholderText(
            "abstract.client.add_requirement\nabstract.client.negotiate_price"
        )
        self._ins_actor_repertoire.setStyleSheet(_form_input_style())
        self._ins_actor_repertoire.textChanged.connect(self._on_inspector_changed)
        right_layout.addWidget(self._ins_actor_repertoire)
        self._actor_form_widgets += [self._lbl_actor_rep, self._ins_actor_repertoire]

        self._lbl_actor_notes = self._small_label("📝 notes (자유 메모)")
        right_layout.addWidget(self._lbl_actor_notes)
        self._ins_actor_notes = QPlainTextEdit()
        self._ins_actor_notes.setFixedHeight(60)
        self._ins_actor_notes.setStyleSheet(_form_input_style())
        self._ins_actor_notes.textChanged.connect(self._on_inspector_changed)
        right_layout.addWidget(self._ins_actor_notes)
        self._actor_form_widgets += [self._lbl_actor_notes, self._ins_actor_notes]

        # 처음엔 actor form 숨김 (verb 시작)
        for w in self._actor_form_widgets:
            w.setVisible(False)

        right_layout.addStretch(1)
        self._set_inspector_enabled(False)

        # [2026-05-26] 좌측 팔레트 제거 — 캔버스 + 우측 inspector 2 컬럼
        splitter.addWidget(center_wrap)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([720, 280])
        root.addWidget(splitter, 1)

    def _small_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {_Palette.TEXT_MUTED}; font-size: 11px; font-weight: 600; background: transparent;"
        )
        return lbl

    def _set_inspector_enabled(self, enabled: bool) -> None:
        for w in (self._ins_target, self._ins_content, self._ins_reason,
                  self._ins_actor_name, self._ins_actor_type, self._ins_actor_role,
                  self._ins_actor_indicators, self._ins_actor_repertoire,
                  self._ins_actor_notes):
            w.setEnabled(enabled)

    def _show_verb_form(self) -> None:
        """[Phase 0-B] verb 노드 폼 표시 + actor 폼 숨김."""
        for w in self._verb_form_widgets:
            w.setVisible(True)
        for w in self._actor_form_widgets:
            w.setVisible(False)

    def _show_actor_form(self) -> None:
        """[Phase 0-B] actor 노드 폼 표시 + verb 폼 숨김."""
        for w in self._verb_form_widgets:
            w.setVisible(False)
        for w in self._actor_form_widgets:
            w.setVisible(True)

    def _parse_indicators_text(self, text: str) -> dict:
        """텍스트 (각 줄 'key: value') → dict. value 가 숫자면 float, else str."""
        out: dict = {}
        for line in (text or "").splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip()
            if not k:
                continue
            # 숫자 시도
            try:
                if "." in v:
                    out[k] = float(v)
                else:
                    out[k] = int(v)
            except ValueError:
                out[k] = v
        return out

    def _format_indicators_dict(self, d: dict) -> str:
        if not isinstance(d, dict):
            return ""
        return "\n".join(f"{k}: {v}" for k, v in d.items())

    def _parse_repertoire_text(self, text: str) -> list:
        """텍스트 (각 줄 action_id) → list. 공백·주석(#) 줄 skip."""
        out: list = []
        for line in (text or "").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
        return out

    # ── 캔버스 동작 ──
    def _add_node_of_verb(self, verb_id: str) -> None:
        # 캔버스 중앙 부근에 추가 (스크롤 위치 고려)
        center = self._view.mapToScene(self._view.viewport().rect().center())
        # 약간씩 오프셋 (여러 번 추가 시 겹침 방지)
        import random
        ox = random.randint(-30, 30)
        oy = random.randint(-30, 30)
        node = self._cstore.make_empty_node(verb_id, center.x() + ox - 90, center.y() + oy - 40)
        self._scene.add_node(node)

    def _on_node_selected(self, node_id: str) -> None:
        self._current_node_id = node_id
        node = self._scene._nodes.get(node_id)
        if node is None:
            self._set_inspector_enabled(False)
            self._ins_header.setText("⚙️  노드를 선택하세요")
            return
        data = node.node_data
        # [Phase 0-B] kind 분기 — verb (default) vs actor
        kind = data.get("kind", "verb")
        self._suppress_inspector_save = True
        try:
            if kind == "actor":
                # ── actor 노드 ──
                self._show_actor_form()
                try:
                    from eidos_canvas_widget import get_actor_meta
                    ameta = get_actor_meta(data.get("actor_type", "human"))
                except Exception:
                    ameta = {"icon": "🎭", "label": "행위자"}
                self._ins_header.setText(
                    f"{ameta.get('icon','🎭')}  {data.get('name','(이름 없음)')}"
                )
                self._ins_actor_name.setText(data.get("name", ""))
                # combo 안전 set
                at = data.get("actor_type", "human")
                idx = self._ins_actor_type.findText(at)
                self._ins_actor_type.setCurrentIndex(idx if idx >= 0 else 1)
                role = data.get("role", "other")
                ridx = self._ins_actor_role.findText(role)
                self._ins_actor_role.setCurrentIndex(ridx if ridx >= 0 else 0)
                self._ins_actor_indicators.setPlainText(
                    self._format_indicators_dict(data.get("indicators") or {})
                )
                self._ins_actor_repertoire.setPlainText(
                    "\n".join(data.get("action_repertoire") or [])
                )
                self._ins_actor_notes.setPlainText(data.get("notes", ""))
            else:
                # ── verb 노드 (옛 그대로) ──
                self._show_verb_form()
                verb_meta = self._cstore.get_verb_meta(data.get("verb", "?")) or {}
                self._ins_header.setText(
                    f"{verb_meta.get('icon','·')}  {verb_meta.get('label', data.get('verb','?'))}"
                )
                self._ins_target.setPlaceholderText(verb_meta.get("target_hint", "") or "")
                self._ins_content.setPlaceholderText(verb_meta.get("content_hint", "") or "")
                self._ins_target.setText(data.get("target", ""))
                self._ins_content.setPlainText(data.get("content", ""))
                self._ins_reason.setText(data.get("reason", ""))
            self._set_inspector_enabled(True)
        finally:
            self._suppress_inspector_save = False

    def _on_inspector_changed(self) -> None:
        if self._suppress_inspector_save or not self._current_node_id:
            return
        node = self._scene._nodes.get(self._current_node_id)
        if node is None:
            return
        data = node.node_data
        # [Phase 0-B] kind 분기
        if data.get("kind") == "actor":
            data["name"] = self._ins_actor_name.text().strip() or "(이름 없음)"
            data["actor_type"] = self._ins_actor_type.currentText()
            data["role"] = self._ins_actor_role.currentText()
            data["indicators"] = self._parse_indicators_text(
                self._ins_actor_indicators.toPlainText()
            )
            data["action_repertoire"] = self._parse_repertoire_text(
                self._ins_actor_repertoire.toPlainText()
            )
            data["notes"] = self._ins_actor_notes.toPlainText()
        else:
            data["target"] = self._ins_target.text()
            data["content"] = self._ins_content.toPlainText()
            data["reason"] = self._ins_reason.text()
        node.update()   # 노드 라벨/미리보기 갱신

    def _on_canvas_dirty(self) -> None:
        # 디바운스 자동 저장은 v2.1 — 현재는 사용자가 명시 저장
        pass

    # ── 저장 / 불러오기 ──
    def _on_save_clicked(self) -> None:
        name = self._fld_canvas_name.text().strip() or "untitled"
        canvas = self._scene.to_canvas_dict(name=name, description="")
        ok = self._cstore.save_canvas(canvas)
        if ok:
            QMessageBox.information(
                self, "💾 저장 완료",
                f"캔버스 '{name}' 저장됨\n경로: {self._cstore.canvas_path(name)}",
            )
        else:
            QMessageBox.warning(self, "저장 실패", "디스크 저장 실패 (콘솔 확인)")

    def _on_load_clicked(self) -> None:
        names = self._cstore.list_canvases()
        if not names:
            QMessageBox.information(
                self, "📂 불러오기",
                f"저장된 캔버스가 없습니다.\n위치: eidos_files/canvases/",
            )
            return
        # 간단한 선택 다이얼로그
        from PySide6.QtWidgets import QInputDialog
        chosen, ok = QInputDialog.getItem(
            self, "📂 캔버스 불러오기", "캔버스 선택:", names, 0, False,
        )
        if not ok or not chosen:
            return
        canvas = self._cstore.load_canvas(chosen)
        self._scene.load_from_canvas_dict(canvas)
        self._fld_canvas_name.setText(canvas.get("name", chosen))
        self._current_node_id = None
        self._set_inspector_enabled(False)
        self._ins_header.setText("⚙️  노드를 선택하세요")

    def _on_new_stage_clicked(self) -> None:
        """[Phase 0-C 2026-05-26] 🎭 새 stage — LLM 자동 제안 다이얼로그 열기.

        다이얼로그가 "📌 캔버스에 배치" 누르면 stage_applied 시그널 발화 →
        _on_stage_applied 가 stage 로드 + 캔버스에 actor 노드 시각화.
        """
        try:
            from eidos_stage_propose import StageProposeDialog
        except Exception as e:
            QMessageBox.warning(
                self, "🎭 새 stage — 오류",
                f"eidos_stage_propose 모듈 로드 실패: {e}",
            )
            return
        # eidos_worker 는 부모 다이얼로그를 통해 찾는다 (MoreActionsDialog → ChatWindow → eidos_worker)
        worker = None
        try:
            w = self.parent()
            while w is not None:
                if hasattr(w, "eidos_worker") and getattr(w, "eidos_worker", None) is not None:
                    worker = w.eidos_worker
                    break
                w = w.parent() if hasattr(w, "parent") else None
        except Exception:
            pass
        if worker is None:
            QMessageBox.warning(
                self, "🎭 새 stage — 오류",
                "EidosWorker 를 찾을 수 없습니다. ChatWindow 통해 더보기를 다시 열어보세요.",
            )
            return
        dlg = StageProposeDialog(parent=self, eidos_worker=worker)
        dlg.stage_applied.connect(self._on_stage_applied)
        dlg.exec()

    def _on_stage_applied(self, stage_id: str) -> None:
        """[Phase 0-C] StageProposeDialog 가 stage 생성·저장 후 시그널 → 캔버스 로드.
        [Phase 0-D] stage_id 추적·1 tick 버튼 활성화."""
        try:
            from eidos_agent_stage_store import load_stage
            stage = load_stage(stage_id)
        except Exception as e:
            QMessageBox.warning(self, "Stage 로드 실패", f"{type(e).__name__}: {e}")
            return
        if stage is None:
            QMessageBox.warning(self, "Stage 로드 실패", f"stage_id={stage_id} 를 찾을 수 없음")
            return
        # [Phase 0-D] stage_id 저장·1 tick 버튼 활성화
        # [Phase 1-A] N tick 버튼도 동시 활성화
        self._current_stage_id = stage_id
        try:
            self._btn_tick.setEnabled(True)
            self._btn_tick.setToolTip(
                f"🎯 1 tick 실행 — Stage '{stage.name}' / goal '{stage.goal[:40]}...'"
            )
            self._btn_tick_n.setEnabled(True)
            self._btn_tick_n.setToolTip(
                f"🔄 N tick 자동 — Stage '{stage.name}' / safety={stage.safety_mode}\n"
                f"종료: max_ticks 도달·중단·연속 같은 action 3회·error"
            )
            # [Phase 1-D] safety 버튼 활성화 + 텍스트 갱신
            self._btn_safety.setEnabled(True)
            self._refresh_safety_button(stage.safety_mode)
            # [Phase 2-B] schedule 버튼 활성화
            self._btn_schedule.setEnabled(True)
            # [Phase 3-C] stats 버튼 활성화
            self._btn_stats.setEnabled(True)
        except Exception:
            pass

        # [Phase 0-C] 캔버스에 main canvas 의 actor/action 노드 시각화
        main = stage.main_canvas()
        if main is None:
            QMessageBox.warning(self, "Canvas 비어 있음", "main canvas 가 없습니다.")
            return
        # _CanvasScene.load_from_canvas_dict 가 {nodes, connections} 받음.
        # agent_stage 의 edges 를 connections 형식으로 매핑 (from/to 만 사용)
        canvas_dict = {
            "version": 1,
            "name": stage.name,
            "description": stage.goal,
            "nodes": main.nodes,
            "connections": [
                {"from": e.get("from"), "to": e.get("to")}
                for e in (main.edges or []) if e.get("from") and e.get("to")
            ],
            "created_at": main.created_at,
            "updated_at": main.updated_at,
        }
        self._scene.load_from_canvas_dict(canvas_dict)
        # 캔버스 이름 필드 갱신
        self._fld_canvas_name.setText(stage.name)
        # 자동 정렬 적용
        try:
            self._scene.auto_layout()
        except Exception:
            pass
        # inspector 초기화
        self._current_node_id = None
        self._set_inspector_enabled(False)
        self._ins_header.setText("⚙️  노드를 선택하세요")

    def _on_stats_clicked(self) -> None:
        """[Phase 3-C 2026-05-26] 📊 PatternModel 통계 다이얼로그 — 누적 학습 시각화."""
        if not self._current_stage_id:
            QMessageBox.warning(self, "📊 Stage 없음", "먼저 Stage 를 로드하세요.")
            return
        try:
            from eidos_pattern_model import PatternModel
            from eidos_agent_stage_store import read_history
        except Exception as e:
            QMessageBox.warning(self, "📊 모듈 로드 실패", f"{e}")
            return
        try:
            events = read_history(self._current_stage_id) or []
            model = PatternModel.from_history(events)
            md = model.summary_markdown()
            # 디스크 캐시 갱신
            try:
                model.save(self._current_stage_id)
            except Exception:
                pass
        except Exception as e:
            QMessageBox.warning(self, "📊 통계 빌드 실패", f"{e}")
            return

        # 다이얼로그 — 옛 1 tick 결과 다이얼로그 와 동일 패턴
        try:
            dlg = QDialog(self)
            dlg.setWindowTitle(f"📊 PatternModel — {self._current_stage_id[:40]}")
            dlg.resize(760, 640)
            lay = QVBoxLayout(dlg)
            lbl = QLabel(
                f"누적 tick: {model.tick_count}  ·  결정 종류: {len(model.decisions)}개"
                f"  ·  belief tick log: {len(model.belief_tick_log)} entries"
            )
            lbl.setStyleSheet("color: #C084FC; padding: 4px;")
            lay.addWidget(lbl)
            te = QPlainTextEdit()
            te.setReadOnly(True)
            te.setPlainText(md)
            te.setStyleSheet(
                f"QPlainTextEdit {{ background: {_Palette.LOG_BG}; color: {_Palette.LOG_TEXT};"
                f" border: 1px solid {_Palette.LOG_BORDER}; border-radius: 8px;"
                f" font-family: 'Consolas','Courier New',monospace; font-size: 11px;"
                f" padding: 10px; }}"
            )
            lay.addWidget(te, 1)
            btn = QPushButton("닫기")
            btn.clicked.connect(dlg.accept)
            lay.addWidget(btn)
            dlg.exec()
        except Exception as e:
            QMessageBox.information(self, "📊 통계", md[:3000])

    def _refresh_safety_button(self, mode: str) -> None:
        """[Phase 1-D] 🛡 버튼 텍스트·색 갱신."""
        mode = (mode or "normal").lower()
        emoji = {"paranoid": "🛡🛡", "normal": "🛡", "yolo": "⚡"}.get(mode, "🛡")
        try:
            self._btn_safety.setText(f"{emoji}  safety: {mode}")
            if mode == "yolo":
                # 강조 — 빨강 보더
                self._btn_safety.setStyleSheet(_small_btn_style(primary=False) +
                    " QPushButton { border-color: #F87171 !important; color: #FCA5A5; }")
            else:
                self._btn_safety.setStyleSheet(_small_btn_style(primary=False))
        except Exception as e:
            print(f"[CanvasTab] safety 버튼 갱신 실패 (graceful): {e}")

    def _on_safety_clicked(self) -> None:
        """[Phase 1-D 2026-05-26] 🛡 safety_mode 변경 — paranoid/normal/yolo 선택.

        yolo 선택 시 추가 경고 다이얼로그.
        변경 시 stage.json 즉시 저장.
        """
        if not self._current_stage_id:
            QMessageBox.warning(self, "🛡 Stage 없음", "먼저 Stage 를 로드하세요.")
            return
        try:
            from eidos_agent_stage_store import load_stage, save_stage
        except Exception as e:
            QMessageBox.warning(self, "🛡 모듈 로드 실패", f"{e}")
            return
        stage = load_stage(self._current_stage_id)
        if stage is None:
            QMessageBox.warning(self, "🛡 Stage 없음", "stage 디스크에서 찾을 수 없음")
            return

        from PySide6.QtWidgets import QInputDialog
        items = [
            "paranoid — ask_user 만 실 실행·모두 dry-run",
            "normal — meta+llm.write+ask_user 실·외부효과 dry-run (default)",
            "yolo — 외부효과 모두 실 실행 (⚠️ 회복 불가)",
        ]
        keys = ["paranoid", "normal", "yolo"]
        try:
            cur_idx = keys.index(stage.safety_mode or "normal")
        except ValueError:
            cur_idx = 1
        chosen, ok = QInputDialog.getItem(
            self, "🛡 safety_mode 변경",
            f"현재: {stage.safety_mode}\n새 안전 모드 선택:",
            items, cur_idx, False,
        )
        if not ok:
            return
        new_mode = keys[items.index(chosen)]

        if new_mode == "yolo" and stage.safety_mode != "yolo":
            confirm = QMessageBox.warning(
                self, "⚠️ YOLO 모드 — 정말 진행하시겠습니까?",
                "yolo 모드에서는 다음 외부효과가 진짜 발생합니다:\n\n"
                "  • 이메일 발송 (`eidos.tool.message.email`)\n"
                "  • 텔레그램 메시지 발송 (`eidos.tool.message.telegram`)\n"
                "  • 웹 페이지 이동·클릭·입력·읽기\n\n"
                "한 번 발송된 메시지는 회복할 수 없습니다.\n"
                "정말로 yolo 모드로 진행할까요?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if confirm != QMessageBox.Yes:
                return

        if new_mode == stage.safety_mode:
            return   # 변경 없음

        stage.safety_mode = new_mode
        if save_stage(stage):
            self._refresh_safety_button(new_mode)
            self._btn_tick_n.setToolTip(
                f"🔄 N tick 자동 — Stage '{stage.name}' / safety={new_mode}\n"
                f"종료: max_ticks 도달·중단·연속 같은 action 3회·error"
            )
            QMessageBox.information(
                self, "🛡 safety_mode 변경됨",
                f"새 모드: **{new_mode}**\n\n다음 [🎯 1 tick] / [🔄 N tick] 부터 적용됩니다.",
            )
        else:
            QMessageBox.warning(self, "🛡 저장 실패", "stage.json 쓰기 실패 (콘솔 확인)")

    def _on_tick_progress_main(self, event: str, payload: dict) -> None:
        """[Phase 1-E 2026-05-26] tick 진행 이벤트 main slot — TickProgressDialog 갱신.

        진행 다이얼로그가 살아있으면 (open 상태) append_event 호출.
        loop_done 이벤트에서는 다이얼로그가 "닫기" 활성화로 전환.
        """
        dlg = getattr(self, "_progress_dialog", None)
        if dlg is None:
            return
        try:
            dlg.append_event(event, payload)
        except Exception as e:
            print(f"[CanvasTab] tick_progress slot 실패 (graceful): {e}")

    def _on_run_stage_tick_clicked(self) -> None:
        """[Phase 0-D 2026-05-26] 🎯 Stage 1 tick — agent runner 비동기 호출.

        흐름:
          1. _current_stage_id 검증
          2. stage 로드 + eidos_worker 탐색
          3. run_stage_one_tick async 호출 (worker thread)
          4. _tick_done_signal 으로 main thread 에 결과 emit
          5. _on_tick_done_main 이 다이얼로그 표시
        """
        if not self._current_stage_id:
            QMessageBox.warning(
                self, "🎯 1 tick — Stage 없음",
                "먼저 [🎭 새 stage] 또는 옛 stage 로드를 통해 Stage 를 캔버스에 띄우세요.",
            )
            return
        try:
            from eidos_agent_stage_store import load_stage
            from eidos_agent_runner import run_stage_one_tick
        except Exception as e:
            QMessageBox.warning(self, "🎯 1 tick — 모듈 로드 실패",
                                f"{type(e).__name__}: {e}")
            return
        stage = load_stage(self._current_stage_id)
        if stage is None:
            QMessageBox.warning(self, "🎯 1 tick — Stage 없음",
                                f"stage_id={self._current_stage_id} 디스크에서 찾을 수 없음")
            self._current_stage_id = None
            self._btn_tick.setEnabled(False)
            return

        # eidos_worker 탐색 (부모 chain)
        worker = None
        try:
            w = self.parent()
            while w is not None:
                if hasattr(w, "eidos_worker") and getattr(w, "eidos_worker", None) is not None:
                    worker = w.eidos_worker
                    break
                w = w.parent() if hasattr(w, "parent") else None
        except Exception:
            pass
        if worker is None:
            QMessageBox.warning(self, "🎯 1 tick — Worker 부재",
                                "EidosWorker 를 찾을 수 없습니다.")
            return

        # UI — 실행 중
        self._btn_tick.setEnabled(False)
        self._btn_tick.setText("⏳  분석/결정 중...")

        # [Phase 1-C] ask_user callback wire — worker thread 가 main thread 다이얼로그 위임
        async def ask_user_callback(question: str) -> str:
            import asyncio as _aio
            cur_loop = _aio.get_event_loop()
            self._worker_loop_ref = cur_loop
            fut = cur_loop.create_future()
            self._ask_user_signal.emit(question, fut)
            try:
                return await fut
            except Exception:
                return ""

        async def _run():
            try:
                result = await run_stage_one_tick(stage, on_ask_user=ask_user_callback)
                self._tick_done_signal.emit(result if isinstance(result, dict) else
                                            {"_error": "result 가 dict 가 아님"})
            except Exception as e:
                self._tick_done_signal.emit({
                    "_error": f"runner 실행 실패: {type(e).__name__}: {e}",
                })

        try:
            worker.submit_task(_run())
        except Exception as e:
            self._btn_tick.setEnabled(True)
            self._btn_tick.setText("🎯  1 tick")
            QMessageBox.warning(self, "🎯 1 tick — 제출 실패", f"{type(e).__name__}: {e}")

    def _on_run_stage_loop_clicked(self) -> None:
        """[Phase 1-A 2026-05-26] 🔄 N tick 자동 루프 — 또는 실행 중이면 중단.

        흐름:
          1. 실행 중이면 stop_event 발화 (중단 신호)
          2. 아니면 max_ticks 입력 dialog → run_stage_loop_async 비동기 호출
          3. ask_user callback wire (worker thread → main thread 다이얼로그)
          4. 결과 누적 markdown 다이얼로그
        """
        # 실행 중 — 중단 신호
        if getattr(self, "_loop_is_running", False):
            try:
                if self._stop_event is not None and self._worker_loop_ref is not None:
                    self._worker_loop_ref.call_soon_threadsafe(self._stop_event.set)
                    self._btn_tick_n.setEnabled(False)
                    self._btn_tick_n.setText("⏸  중단 중...")
            except Exception as e:
                print(f"[CanvasTab] N tick 중단 신호 실패: {e}")
            return

        if not self._current_stage_id:
            QMessageBox.warning(
                self, "🔄 N tick — Stage 없음",
                "먼저 [🎭 새 stage] 또는 stage 로드를 통해 캔버스에 띄우세요.",
            )
            return
        try:
            from eidos_agent_stage_store import load_stage
            from eidos_agent_runner import run_stage_loop_async
        except Exception as e:
            QMessageBox.warning(self, "🔄 N tick — 모듈 로드 실패",
                                f"{type(e).__name__}: {e}")
            return
        stage = load_stage(self._current_stage_id)
        if stage is None:
            QMessageBox.warning(self, "🔄 N tick — Stage 없음",
                                f"stage_id={self._current_stage_id} 디스크에서 찾을 수 없음")
            return

        # max_ticks 입력
        from PySide6.QtWidgets import QInputDialog
        max_ticks, ok = QInputDialog.getInt(
            self, "🔄 N tick 자동",
            f"Stage '{stage.name}' / safety={stage.safety_mode}\n"
            f"몇 tick 자동 실행할까요? (1~30)\n"
            f"⚠️ safety_mode='yolo' 면 외부효과 (이메일/텔레그램/웹) 실 발생",
            value=3, minValue=1, maxValue=30,
        )
        if not ok:
            return

        # eidos_worker 탐색
        worker = None
        try:
            w = self.parent()
            while w is not None:
                if hasattr(w, "eidos_worker") and getattr(w, "eidos_worker", None) is not None:
                    worker = w.eidos_worker
                    break
                w = w.parent() if hasattr(w, "parent") else None
        except Exception:
            pass
        if worker is None:
            QMessageBox.warning(self, "🔄 N tick — Worker 부재",
                                "EidosWorker 를 찾을 수 없습니다.")
            return

        # UI — 실행 중 (버튼 텍스트 토글 — 옛 v2.2 ▶/🛑 패턴)
        self._loop_is_running = True
        self._btn_tick_n.setText("🛑  중단")
        self._btn_tick.setEnabled(False)   # 1 tick 동시 클릭 방지

        # [Phase 1-E] 실시간 진행 다이얼로그 생성 + 즉시 표시 (modeless)
        self._progress_dialog = TickProgressDialog(
            parent=self,
            title=f"🔄 Stage '{stage.name}' — N tick 진행",
            max_ticks=max_ticks,
            safety_mode=stage.safety_mode,
        )
        # 다이얼로그의 🛑 중단 → stop_event 발화
        def _on_dialog_stop_request():
            try:
                if self._stop_event is not None and self._worker_loop_ref is not None:
                    self._worker_loop_ref.call_soon_threadsafe(self._stop_event.set)
            except Exception as e:
                print(f"[CanvasTab] dialog stop 신호 실패: {e}")
        self._progress_dialog.request_stop.connect(_on_dialog_stop_request)
        self._progress_dialog.show()

        # ask_user + stop_event wire — 옛 v2.2 ASK_USER 패턴 재사용
        async def ask_user_callback(question: str) -> str:
            import asyncio as _aio
            cur_loop = _aio.get_event_loop()
            self._worker_loop_ref = cur_loop
            fut = cur_loop.create_future()
            self._ask_user_signal.emit(question, fut)
            try:
                return await fut
            except Exception:
                return ""

        # [Phase 1-E] progress callback — worker thread 가 main thread 시그널 emit
        def on_progress_cb(event: str, payload: dict):
            try:
                self._tick_progress_signal.emit(event, payload if isinstance(payload, dict) else {})
            except Exception:
                pass

        async def _run_loop():
            import asyncio as _aio
            cur_loop = _aio.get_event_loop()
            self._worker_loop_ref = cur_loop
            self._stop_event = _aio.Event()
            try:
                result = await run_stage_loop_async(
                    stage,
                    max_ticks=max_ticks,
                    on_progress=on_progress_cb,
                    on_ask_user=ask_user_callback,
                    stop_event=self._stop_event,
                    inter_tick_delay_seconds=1.5,
                )
                # signal 통해 main thread 결과 emit — _tick_done_signal 재사용
                self._tick_done_signal.emit({
                    "_loop_result": True,
                    **result,
                })
            except Exception as e:
                self._tick_done_signal.emit({
                    "_error": f"N tick 루프 실패: {type(e).__name__}: {e}",
                })

        try:
            worker.submit_task(_run_loop())
        except Exception as e:
            self._loop_is_running = False
            self._btn_tick_n.setText("🔄  N tick")
            self._btn_tick_n.setEnabled(True)
            self._btn_tick.setEnabled(True)
            if self._progress_dialog:
                self._progress_dialog.close()
                self._progress_dialog = None
            QMessageBox.warning(self, "🔄 N tick — 제출 실패", f"{type(e).__name__}: {e}")

    def _on_run_stage_schedule_clicked(self) -> None:
        """[Phase 2-B 2026-05-26] ⏰ Scheduler — 정주기 자동 tick.

        N tick (즉시 연속) 과 달리 매 interval (초) 마다 sleep 후 다음 tick.
        진행 중 다시 누르면 중단 신호 (loop 와 동일).
        """
        # 실행 중 — 중단 신호 (loop 와 동일 인프라 재사용)
        if getattr(self, "_loop_is_running", False):
            try:
                if self._stop_event is not None and self._worker_loop_ref is not None:
                    self._worker_loop_ref.call_soon_threadsafe(self._stop_event.set)
                    self._btn_schedule.setEnabled(False)
                    self._btn_schedule.setText("⏸  중단 중...")
            except Exception as e:
                print(f"[CanvasTab] schedule 중단 신호 실패: {e}")
            return

        if not self._current_stage_id:
            QMessageBox.warning(self, "⏰ Stage 없음", "먼저 Stage 를 로드하세요.")
            return
        try:
            from eidos_agent_stage_store import load_stage
            from eidos_agent_runner import run_stage_scheduled_async
        except Exception as e:
            QMessageBox.warning(self, "⏰ 모듈 로드 실패", f"{e}")
            return
        stage = load_stage(self._current_stage_id)
        if stage is None:
            QMessageBox.warning(self, "⏰ Stage 없음", "stage 디스크에서 찾을 수 없음")
            return

        # interval + max_ticks 입력 (2 단계)
        # [Phase 3-B] 옛 값 default — stage.scheduling 에 저장된 게 있으면 그것 우선
        from PySide6.QtWidgets import QInputDialog
        prev_sched = stage.scheduling or {}
        default_interval = int(prev_sched.get("interval_seconds", 60))
        default_max_ticks = int(prev_sched.get("max_ticks", 10))
        last_run_info = ""
        if prev_sched.get("last_run_at"):
            last_run_info = (
                f"\n\n📋 마지막 실행: {prev_sched.get('last_run_at')}\n"
                f"   {prev_sched.get('last_ticks_executed', '?')} tick / "
                f"종료: {(prev_sched.get('last_halt_reason') or '')[:60]}"
            )
        interval, ok = QInputDialog.getInt(
            self, "⏰ Scheduler — interval",
            f"Stage '{stage.name}' / safety={stage.safety_mode}\n\n"
            f"각 tick 사이 간격 (초)?\n"
            f"권장: 60 (분) ~ 1800 (30분)·최소 5{last_run_info}",
            value=default_interval, minValue=5, maxValue=86400,
        )
        if not ok:
            return
        max_ticks, ok2 = QInputDialog.getInt(
            self, "⏰ Scheduler — max_ticks",
            f"최대 tick 수? (안전 상한)\n"
            f"interval={interval}s × max_ticks 만큼 실행됨.\n"
            f"권장: 10~50",
            value=default_max_ticks, minValue=1, maxValue=200,
        )
        if not ok2:
            return

        # [Phase 3-B] stage.scheduling 에 입력 값 임시 저장 (종료 후 결과도 갱신)
        try:
            stage.scheduling = {
                **(stage.scheduling or {}),
                "interval_seconds": interval,
                "max_ticks": max_ticks,
            }
            from eidos_agent_stage_store import save_stage as _save_stage
            _save_stage(stage)
        except Exception as _e_save:
            print(f"[CanvasTab] scheduling 사전 저장 실패 (graceful): {_e_save}")

        # worker 찾기
        worker = None
        try:
            w = self.parent()
            while w is not None:
                if hasattr(w, "eidos_worker") and getattr(w, "eidos_worker", None) is not None:
                    worker = w.eidos_worker
                    break
                w = w.parent() if hasattr(w, "parent") else None
        except Exception:
            pass
        if worker is None:
            QMessageBox.warning(self, "⏰ Worker 부재", "EidosWorker 를 찾을 수 없습니다.")
            return

        # UI — running 상태
        self._loop_is_running = True
        self._btn_schedule.setText("🛑  중단")
        self._btn_tick.setEnabled(False)
        self._btn_tick_n.setEnabled(False)

        # 진행 다이얼로그
        self._progress_dialog = TickProgressDialog(
            parent=self,
            title=f"⏰ Stage '{stage.name}' — Scheduler",
            max_ticks=max_ticks,
            safety_mode=stage.safety_mode,
        )
        # 헤더에 interval 표시
        try:
            self._progress_dialog._lbl_status.setText(
                f"⏰ Scheduler 시작 — interval={interval}s · max={max_ticks} · safety={stage.safety_mode}"
            )
        except Exception:
            pass

        def _on_dialog_stop_request():
            try:
                if self._stop_event is not None and self._worker_loop_ref is not None:
                    self._worker_loop_ref.call_soon_threadsafe(self._stop_event.set)
            except Exception as e:
                print(f"[CanvasTab] schedule dialog stop 실패: {e}")
        self._progress_dialog.request_stop.connect(_on_dialog_stop_request)
        self._progress_dialog.show()

        async def ask_user_callback(question: str) -> str:
            import asyncio as _aio
            cur_loop = _aio.get_event_loop()
            self._worker_loop_ref = cur_loop
            fut = cur_loop.create_future()
            self._ask_user_signal.emit(question, fut)
            try:
                return await fut
            except Exception:
                return ""

        def on_progress_cb(event: str, payload: dict):
            try:
                self._tick_progress_signal.emit(event, payload if isinstance(payload, dict) else {})
            except Exception:
                pass

        async def _run_schedule():
            import asyncio as _aio
            cur_loop = _aio.get_event_loop()
            self._worker_loop_ref = cur_loop
            self._stop_event = _aio.Event()
            try:
                result = await run_stage_scheduled_async(
                    stage,
                    interval_seconds=float(interval),
                    max_ticks=max_ticks,
                    on_progress=on_progress_cb,
                    on_ask_user=ask_user_callback,
                    stop_event=self._stop_event,
                )
                self._tick_done_signal.emit({
                    "_loop_result": True,
                    "_scheduled": True,
                    **result,
                })
            except Exception as e:
                self._tick_done_signal.emit({
                    "_error": f"Scheduler 실패: {type(e).__name__}: {e}",
                })

        try:
            worker.submit_task(_run_schedule())
        except Exception as e:
            self._loop_is_running = False
            self._btn_schedule.setText("⏰  자동")
            self._btn_schedule.setEnabled(True)
            self._btn_tick.setEnabled(True)
            self._btn_tick_n.setEnabled(True)
            if self._progress_dialog:
                self._progress_dialog.close()
                self._progress_dialog = None
            QMessageBox.warning(self, "⏰ 제출 실패", f"{type(e).__name__}: {e}")

    def _on_tick_done_main(self, result: dict) -> None:
        """[Phase 0-D + 1-A] tick / loop 완료 — main thread slot. 결과 다이얼로그.

        result 가 `_loop_result=True` 면 N tick 루프 종료·아니면 단일 tick.
        """
        is_loop = isinstance(result, dict) and result.get("_loop_result")
        is_scheduled = isinstance(result, dict) and result.get("_scheduled")

        # UI 복원 (양쪽 모두)
        self._btn_tick.setEnabled(self._current_stage_id is not None)
        self._btn_tick.setText("🎯  1 tick")
        if is_loop:
            self._loop_is_running = False
            self._stop_event = None
            self._worker_loop_ref = None
            self._btn_tick_n.setEnabled(self._current_stage_id is not None)
            self._btn_tick_n.setText("🔄  N tick")
            # [Phase 2-B] schedule 버튼도 복원
            self._btn_schedule.setEnabled(self._current_stage_id is not None)
            self._btn_schedule.setText("⏰  자동")
            # [Phase 3-B] schedule 결과를 stage.scheduling 에 저장
            if is_scheduled and self._current_stage_id:
                try:
                    import datetime as _dtm
                    from eidos_agent_stage_store import load_stage as _ls, save_stage as _ss
                    _stage = _ls(self._current_stage_id)
                    if _stage is not None:
                        _stage.scheduling = {
                            **(_stage.scheduling or {}),
                            "interval_seconds": result.get("interval_seconds",
                                _stage.scheduling.get("interval_seconds", 60)),
                            "max_ticks": _stage.scheduling.get("max_ticks", 10),
                            "last_run_at": _dtm.datetime.now().isoformat(timespec="seconds"),
                            "last_ticks_executed": result.get("ticks_executed", 0),
                            "last_halt_reason": result.get("halt_reason", ""),
                        }
                        _ss(_stage)
                except Exception as _e_sched_save:
                    print(f"[CanvasTab] schedule 결과 stage 저장 실패 (graceful): {_e_sched_save}")

        err = result.get("_error") if isinstance(result, dict) else None
        if err:
            QMessageBox.warning(self,
                                "🔄 N tick — 실패" if is_loop else "🎯 1 tick — 실패",
                                err[:500])
            return

        # markdown 빌드 — loop vs single
        if is_loop:
            md = result.get("summary_markdown", "(요약 없음)")
            # 각 tick 의 상세도 부착
            tick_results = result.get("tick_results") or []
            try:
                from eidos_agent_runner import format_tick_summary_markdown
                detail_blocks = []
                for i, tr in enumerate(tick_results, 1):
                    if not isinstance(tr, dict) or tr.get("_error"):
                        continue
                    detail_blocks.append(f"\n---\n\n### Tick {i} 상세\n\n"
                                         + format_tick_summary_markdown(tr))
                if detail_blocks:
                    md = md + "\n" + "".join(detail_blocks)
            except Exception:
                pass
            # [Phase 1-E] 진행 다이얼로그가 살아있으면 거기에 요약 부착 — 별도 결과 다이얼로그 X
            dlg = getattr(self, "_progress_dialog", None)
            if dlg is not None:
                try:
                    dlg.append_text("\n\n" + md)
                    self._progress_dialog = None   # 사용자가 닫기 누를 때까지 둠
                except Exception:
                    pass
                return
        else:
            try:
                from eidos_agent_runner import format_tick_summary_markdown
                md = format_tick_summary_markdown(result)
            except Exception as e:
                md = f"format 실패: {e}\n\n{result}"

        # 결과 다이얼로그 — 큰 텍스트 표시 (QPlainTextEdit 안 QMessageBox 보다 좋음)
        try:
            from PySide6.QtWidgets import QDialog, QVBoxLayout, QPlainTextEdit, QPushButton, QLabel
            dlg = QDialog(self)
            dlg.setWindowTitle(f"🎯 Stage Tick {result.get('tick_num','?')} — 결과")
            dlg.resize(720, 600)
            lay = QVBoxLayout(dlg)
            lbl = QLabel(f"Stage: {result.get('stage_id','?')[:50]}  ·  safety: {result.get('safety_mode','normal')}")
            lbl.setStyleSheet("color: #888; padding: 4px;")
            lay.addWidget(lbl)
            te = QPlainTextEdit()
            te.setReadOnly(True)
            te.setPlainText(md)
            te.setStyleSheet(
                f"QPlainTextEdit {{ background: {_Palette.LOG_BG}; color: {_Palette.LOG_TEXT};"
                f" border: 1px solid {_Palette.LOG_BORDER}; border-radius: 8px;"
                f" font-family: 'Consolas','Courier New',monospace; padding: 10px; }}"
            )
            lay.addWidget(te, 1)
            btn = QPushButton("닫기")
            btn.clicked.connect(dlg.accept)
            lay.addWidget(btn)
            dlg.exec()
        except Exception as e:
            QMessageBox.information(self, "🎯 Tick 결과", md[:2000])

    def _on_new_clicked(self) -> None:
        if len(self._scene._nodes) > 0:
            reply = QMessageBox.question(
                self, "새 캔버스",
                "현재 캔버스를 비우고 새로 시작할까요?\n(저장 안 한 변경사항은 사라집니다)",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        self._scene.clear_all()
        self._fld_canvas_name.setText("새 캔버스")
        self._current_node_id = None
        self._set_inspector_enabled(False)
        self._ins_header.setText("⚙️  노드를 선택하세요")

    # ── 실행 (v2.2 — runner 직접 호출 + 진행 + stop button + ASK_USER UI) ──
    def _on_run_clicked(self) -> None:
        # 실행 중이면 stop 토글
        if self._is_running:
            self._on_stop_clicked()
            return
        if not self._scene._nodes:
            QMessageBox.information(self, "실행", "빈 캔버스 — 노드를 먼저 추가하세요.")
            return
        name = self._fld_canvas_name.text().strip() or "untitled"
        canvas = self._scene.to_canvas_dict(name=name)
        # ChatWindow 찾기 (worker 접근용)
        cw = _find_chat_window(self.parent() if self.parent() else self)
        worker = getattr(cw, "eidos_worker", None) if cw else None
        if worker is None:
            QMessageBox.warning(
                self, "실행 — worker 부재",
                "ChatWindow.eidos_worker 부재 — 실행할 수 없습니다.",
            )
            return
        worker_loop = getattr(worker, "loop", None)
        if worker_loop is None or not worker_loop.is_running():
            QMessageBox.warning(self, "실행 — worker loop 부재",
                                "worker.loop 가 running 중이 아닙니다.")
            return
        self._worker_loop_ref = worker_loop
        # 모든 노드 상태 초기화
        for node in self._scene._nodes.values():
            node.set_run_status(None)

        # ── stop_event 를 worker loop 안에서 생성 (asyncio.Event 는 same-loop 필요)
        import asyncio as _asyncio
        async def _make_event():
            return _asyncio.Event()
        fut_ev = _asyncio.run_coroutine_threadsafe(_make_event(), worker_loop)
        try:
            self._stop_event = fut_ev.result(timeout=2.0)
        except Exception as _e_ev:
            QMessageBox.warning(self, "실행 실패", f"stop_event 생성 실패: {_e_ev}")
            return

        # ── on_progress callback (signal emit·worker thread 안전)
        def _on_progress(event: str, node_id: str, payload: dict):
            try:
                self._canvas_progress.emit(event, node_id, dict(payload))
            except Exception as _e_em:
                print(f"[CanvasTab] progress emit 실패: {_e_em}")
            return None

        # ── on_ask_user (main thread 다이얼로그 호출 → future 로 결과 await)
        async def _on_ask_user(question: str) -> str:
            loop = _asyncio.get_event_loop()
            ans_fut = loop.create_future()
            try:
                self._ask_user_signal.emit(question, ans_fut)
            except Exception as _e_ase:
                ans_fut.set_result(f"(emit 실패: {_e_ase})")
            try:
                ans = await ans_fut
                return "" if ans is None else str(ans)
            except _asyncio.CancelledError:
                raise

        ctx = {"on_ask_user": _on_ask_user, "stop_event": self._stop_event}

        try:
            from eidos_canvas_runner import execute_canvas_async
        except Exception as _e_imp:
            QMessageBox.warning(self, "실행 — runner 로드 실패", f"{_e_imp}")
            self._stop_event = None
            return
        try:
            worker.submit_task(execute_canvas_async(canvas, on_progress=_on_progress,
                                                    ctx=ctx, stop_on_fail=False))
            self._is_running = True
            self._btn_run.setText("🛑  중단")
            # 실행 중에도 클릭 가능 (stop 토글)·setEnabled(True) 유지
            try:
                cw_for_msg = _find_chat_window(self.parent() if self.parent() else self)
                if cw_for_msg and hasattr(cw_for_msg, "_safe_append"):
                    cw_for_msg._safe_append(
                        f"🎨 [캔버스 실행 시작] '{name}' — {len(canvas.get('nodes', []))}개 노드",
                        "system",
                    )
            except Exception:
                pass
        except Exception as e:
            QMessageBox.warning(self, "실행 실패", f"{type(e).__name__}: {e}")
            self._stop_event = None

    def _on_stop_clicked(self) -> None:
        """🛑 중단 — main thread 에서 stop_event.set 을 worker loop 에 전달."""
        if self._stop_event is None or self._worker_loop_ref is None:
            return
        try:
            self._worker_loop_ref.call_soon_threadsafe(self._stop_event.set)
        except Exception as e:
            print(f"[CanvasTab] stop_event.set 실패 (graceful): {e}")
        # UI 즉시 라벨 복원·all_done 에서 최종 cleanup
        self._btn_run.setText("⏹  중단 중...")
        self._btn_run.setEnabled(False)

    # ── v2.3 단축키 + 정렬/snap 핸들러 ────────────────────────────
    def _on_undo(self) -> None:
        if self._scene.undo():
            self._update_inspector_after_change()

    def _on_redo(self) -> None:
        if self._scene.redo():
            self._update_inspector_after_change()

    def _on_copy(self) -> None:
        n = self._scene.copy_selected()
        if n > 0 and hasattr(self, "_btn_run"):
            # 짧은 안내 — statusbar 없으니 콘솔 + 아무것도 안 함 (사용자 침해 X)
            print(f"[CanvasTab] 노드 {n}개 복사됨")

    def _on_paste(self) -> None:
        n = self._scene.paste_clipboard()
        if n > 0:
            self._update_inspector_after_change()

    def _on_delete_selected(self) -> None:
        sel_ids = [nid for nid, n in self._scene._nodes.items() if n.isSelected()]
        for nid in sel_ids:
            self._scene.remove_node(nid)
        if sel_ids:
            self._update_inspector_after_change()

    def _on_auto_layout_clicked(self) -> None:
        self._scene.auto_layout()
        self._update_inspector_after_change()

    def _on_snap_toggled(self, checked: bool) -> None:
        size = 20 if checked else 0
        self._scene.set_snap_size(size)
        self._btn_snap.setText("🧲  snap" if checked else "🧲  snap (off)")

    def _update_inspector_after_change(self) -> None:
        """undo/redo/paste 후 inspector 상태 reset (선택 노드가 바뀌었을 수 있음)."""
        self._current_node_id = None
        self._set_inspector_enabled(False)
        self._ins_header.setText("⚙️  노드를 선택하세요")

    def _on_ask_user_main(self, question: str, ans_future) -> None:
        """ASK_USER main thread 슬롯 — QInputDialog → future.set_result (worker loop 안전)."""
        try:
            from PySide6.QtWidgets import QInputDialog
            answer, ok = QInputDialog.getText(
                self, "🤔 ASK_USER — EIDOS 질문",
                question or "(질문 비어있음)",
            )
            if not ok:
                answer = ""
        except Exception as e:
            print(f"[CanvasTab] ASK_USER 다이얼로그 실패: {e}")
            answer = ""
        # future 는 worker loop 의 것 — call_soon_threadsafe 로 set
        loop = self._worker_loop_ref
        if loop is None:
            print("[CanvasTab] worker_loop_ref 부재 — ASK_USER future set 불가")
            return
        try:
            loop.call_soon_threadsafe(ans_future.set_result, answer)
        except Exception as e:
            print(f"[CanvasTab] ASK_USER future.set 실패: {e}")

    def _on_progress_main(self, event: str, node_id: str, payload: dict) -> None:
        """main thread 진행 슬롯 — 노드 색상 갱신 + all_done 시 결과 다이얼로그."""
        try:
            if event == "start":
                node = self._scene._nodes.get(node_id)
                if node:
                    node.set_run_status("running")
            elif event == "done":
                node = self._scene._nodes.get(node_id)
                if node:
                    node.set_run_status(payload.get("status"))
            elif event == "all_done":
                # 실행 종료 — UI 복원
                self._is_running = False
                self._stop_event = None
                self._worker_loop_ref = None
                self._btn_run.setEnabled(True)
                self._btn_run.setText("▶  실행")
                total = payload.get("total", 0)
                ok = payload.get("ok", 0)
                fail = payload.get("fail", 0)
                skip = payload.get("skip", 0)
                stopped = payload.get("stopped", 0)
                results = payload.get("results", [])
                # 결과 요약 텍스트
                stopped_part = f" / 🛑 {stopped} 중단" if stopped else ""
                lines = [f"총 {total} 단계 — ✅ {ok} 성공 / ⚠️ {fail} 실패 / ⏭ {skip} skip{stopped_part}\n"]
                for r in results[-20:]:   # 최근 20개
                    icon = {"ok": "✅", "fail": "❌", "skip": "⏭", "warn": "⚠️",
                            "running": "⏳", "stopped": "🛑"}.get(
                        r.get("status", ""), "·")
                    verb = r.get("verb", "?")
                    res_txt = str(r.get("result", ""))[:120]
                    lines.append(f"  {icon} [{r.get('step_index','?')}] {verb}: {res_txt}")
                summary_text = "\n".join(lines)
                # 변수 요약
                vars_d = payload.get("variables", {})
                if vars_d:
                    summary_text += f"\n\n📦 누적 변수 {len(vars_d)}개: {', '.join(list(vars_d.keys())[:8])}"
                QMessageBox.information(self, "▶ 실행 완료", summary_text)
                # 채팅창에도 안내
                try:
                    cw = _find_chat_window(self.parent() if self.parent() else self)
                    if cw and hasattr(cw, "_safe_append"):
                        cw._safe_append(
                            f"🎨 [캔버스 실행 완료] {total} 단계 — ✅ {ok} / ❌ {fail} / ⏭ {skip}",
                            "system",
                        )
                except Exception:
                    pass
        except Exception as e:
            print(f"[CanvasTab] _on_progress_main 실패: {e}")


# ── 프로젝트 탭 (CRUD + 자동 저장 + form) ───────────────────────────────

# 공용 form 스타일 (라이트/다크 동적 — _Palette.INPUT_BG 등을 사용)
def _form_input_style() -> str:
    return (
        f"QLineEdit, QPlainTextEdit, QComboBox {{"
        f"  background: {_Palette.INPUT_BG};"
        f"  color: {_Palette.TEXT_PRIMARY};"
        f"  border: 1px solid {_Palette.BORDER_HANDLE};"
        f"  border-radius: 6px;"
        f"  padding: 6px 10px;"
        f"  selection-background-color: {_Palette.ACCENT_DEEP};"
        f"}}"
        f"QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus {{"
        f"  border: 1px solid {_Palette.ACCENT_GLOW};"
        f"}}"
        f"QComboBox::drop-down {{ border: none; width: 22px; }}"
        f"QComboBox QAbstractItemView {{"
        f"  background: {_Palette.INPUT_BG};"
        f"  color: {_Palette.TEXT_PRIMARY};"
        f"  selection-background-color: {_Palette.ACCENT_DEEP};"
        f"  border: 1px solid {_Palette.BORDER_HANDLE};"
        f"  outline: 0;"
        f"}}"
        f"QListWidget {{"
        f"  background: {_Palette.INPUT_BG};"
        f"  color: {_Palette.TEXT_PRIMARY};"
        f"  border: 1px solid {_Palette.BORDER_HANDLE};"
        f"  border-radius: 6px;"
        f"  padding: 4px;"
        f"}}"
        f"QListWidget::item {{ padding: 4px 8px; border-radius: 3px; }}"
        f"QListWidget::item:selected {{ background: {_Palette.ACCENT_DEEP}; color: #fff; }}"
        f"QCheckBox {{ color: {_Palette.TEXT_PRIMARY}; spacing: 8px; font-size: 13px; }}"
        f"QCheckBox::indicator {{ width: 18px; height: 18px; border: 2px solid {_Palette.BORDER_HOVER};"
        f"  border-radius: 4px; background: {_Palette.INPUT_BG}; }}"
        f"QCheckBox::indicator:checked {{"
        f"  background: qlineargradient(x1:0, y1:0, x2:1, y2:1,"
        f"    stop:0 {_Palette.ACCENT_GLOW}, stop:1 {_Palette.ACCENT_DEEP});"
        f"  border: 2px solid {_Palette.BORDER_GLOW};"
        f"}}"
    )


def _small_btn_style(primary: bool = False) -> str:
    """좌측 하단 CRUD 버튼·우측 파일 첨부 버튼용 작은 보라 버튼."""
    if primary:
        return (
            f"QPushButton {{"
            f"  background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            f"    stop:0 {_Palette.ACCENT}, stop:1 {_Palette.BTN_PRIMARY_R});"
            f"  color: #ffffff;"
            f"  border: 1px solid {_Palette.BORDER_GLOW};"
            f"  border-radius: 6px;"
            f"  padding: 6px 12px;"
            f"  font-size: 12px;"
            f"  font-weight: 600;"
            f"}}"
            f"QPushButton:hover {{"
            f"  background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            f"    stop:0 {_Palette.ACCENT_GLOW}, stop:1 {_Palette.ACCENT_DEEP});"
            f"}}"
        )
    return (
        f"QPushButton {{"
        f"  background: {_Palette.BTN_GHOST_BG};"
        f"  color: {_Palette.TEXT_BODY};"
        f"  border: 1px solid {_Palette.BORDER_HANDLE};"
        f"  border-radius: 6px;"
        f"  padding: 6px 12px;"
        f"  font-size: 12px;"
        f"}}"
        f"QPushButton:hover {{"
        f"  border: 1px solid {_Palette.BORDER_HOVER};"
        f"  color: {_Palette.ACCENT_GLOW};"
        f"}}"
    )


class _ProjectsTab(QWidget):
    """📁 프로젝트 탭 — CRUD + 자동 저장 + form.

    좌측: 프로젝트 버튼 리스트 + [➕ 새 프로젝트] / [🗑 삭제] 버튼
    우측: QScrollArea 안 form (제목·카테고리·상태·기한·설명·준비조건·
          입력 파일·납품·납품 파일·조립 액션·완료체크) + 자동 저장.
    """

    # [2026-05-31] Auto Prompt — worker thread 의 LLM 결과를 main thread 로 marshal.
    _auto_prompt_ready = Signal(str, str)   # (project_id, prompt_text | "__ERROR__...")

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._projects: list[dict] = _proj_store.load_projects()
        self._current_id: Optional[str] = None
        self._suppress_save = False   # 폼 채우는 중 저장 방지
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(500)   # debounce 500ms
        self._save_timer.timeout.connect(self._flush_save)
        self._buttons: list[_ItemButton] = []

        self._build_ui()
        # Auto Prompt 결과 시그널 → main thread 슬롯 (worker thread 안전)
        try:
            self._auto_prompt_ready.connect(
                self._on_auto_prompt_ready, Qt.QueuedConnection
            )
        except Exception as _e_sig:
            print(f"[ProjectsTab] auto_prompt signal connect 실패 (graceful): {_e_sig}")
        self._refresh_list()
        # 첫 항목 자동 선택
        if self._projects and self._buttons:
            self._buttons[0].setChecked(True)
            self._on_clicked(self._buttons[0])

    # ── UI 구성 ──────────────────────────────────────────────────
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet(
            f"QSplitter::handle {{ background: {_Palette.BORDER_HANDLE}; }}"
            f"QSplitter::handle:hover {{ background: {_Palette.ACCENT}; }}"
        )

        # 좌측 — 프로젝트 리스트 + CRUD 버튼
        left_panel = QFrame(splitter)
        left_panel.setObjectName("ProjTab_Left")
        left_panel.setStyleSheet(
            f"QFrame#ProjTab_Left {{"
            f"  background: {_Palette.BG_PANEL};"
            f"  border: 1px solid {_Palette.BORDER_MUTED};"
            f"  border-radius: 10px;"
            f"}}"
        )
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        self._scroll_area_left = QScrollArea(left_panel)
        self._scroll_area_left.setWidgetResizable(True)
        self._scroll_area_left.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll_area_left.setStyleSheet(_scroll_style())

        self._scroll_content_left = QWidget()
        self._scroll_content_left.setStyleSheet("background: transparent;")
        self._buttons_layout = QVBoxLayout(self._scroll_content_left)
        self._buttons_layout.setContentsMargins(10, 10, 10, 10)
        self._buttons_layout.setSpacing(7)
        self._buttons_layout.addStretch(1)
        self._scroll_area_left.setWidget(self._scroll_content_left)
        left_layout.addWidget(self._scroll_area_left, 1)

        # 좌측 하단 CRUD 버튼 행
        crud_row = QHBoxLayout()
        crud_row.setContentsMargins(10, 6, 10, 10)
        crud_row.setSpacing(6)
        self._btn_new = QPushButton("➕  새 프로젝트")
        self._btn_new.setCursor(Qt.PointingHandCursor)
        self._btn_new.setStyleSheet(_small_btn_style(primary=True))
        self._btn_new.clicked.connect(self._on_new_clicked)
        self._btn_delete = QPushButton("🗑  삭제")
        self._btn_delete.setCursor(Qt.PointingHandCursor)
        self._btn_delete.setStyleSheet(_small_btn_style(primary=False))
        self._btn_delete.clicked.connect(self._on_delete_clicked)
        crud_row.addWidget(self._btn_new, 1)
        crud_row.addWidget(self._btn_delete, 0)
        left_layout.addLayout(crud_row)

        # 우측 — form (스크롤 가능)
        right_panel = QFrame(splitter)
        right_panel.setObjectName("ProjTab_Right")
        right_panel.setStyleSheet(
            f"QFrame#ProjTab_Right {{"
            f"  background: {_Palette.BG_RIGHT};"
            f"  border: 1px solid {_Palette.BORDER_MUTED};"
            f"  border-radius: 10px;"
            f"}}"
        )
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self._form_scroll = QScrollArea(right_panel)
        self._form_scroll.setWidgetResizable(True)
        self._form_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._form_scroll.setStyleSheet(_scroll_style())

        self._form_content = QWidget()
        self._form_content.setStyleSheet("background: transparent;")
        form_layout = QVBoxLayout(self._form_content)
        form_layout.setContentsMargins(22, 22, 22, 22)
        form_layout.setSpacing(14)

        # 헤더 (현재 프로젝트 제목 큰 글씨 — 자동 저장 상태 표시)
        self._form_header = QLabel("프로젝트를 선택하거나 ➕ 새 프로젝트를 만드세요")
        self._form_header.setFont(QFont("", 16, QFont.Bold))
        self._form_header.setStyleSheet(
            f"color: {_Palette.ACCENT_GLOW}; background: transparent; letter-spacing: 0.3px;"
        )
        self._form_header.setWordWrap(True)
        _apply_purple_glow(self._form_header, radius=18, color_hex=_Palette.ACCENT, alpha=110)
        self._form_header.setVisible(False)   # [2026-05-26 사용자 요청] 헤더 숨김
        form_layout.addWidget(self._form_header)

        self._form_save_status = QLabel("")
        self._form_save_status.setStyleSheet(
            f"color: {_Palette.TEXT_MUTED}; font-size: 11px; background: transparent;"
        )
        self._form_save_status.setVisible(False)   # [2026-05-26 사용자 요청] 저장 상태 라벨 숨김
        form_layout.addWidget(self._form_save_status)

        # 제목 / 카테고리 / 상태 / 기한 (한 줄씩)
        form_layout.addWidget(self._field_label("📌 제목"))
        self._fld_title = QLineEdit()
        self._fld_title.setStyleSheet(_form_input_style())
        self._fld_title.textChanged.connect(self._schedule_save)
        form_layout.addWidget(self._fld_title)

        row1 = QHBoxLayout()
        row1.setSpacing(10)
        col_cat = QVBoxLayout()
        col_cat.addWidget(self._field_label("🏷 카테고리"))
        self._fld_category = QLineEdit()
        self._fld_category.setStyleSheet(_form_input_style())
        self._fld_category.textChanged.connect(self._schedule_save)
        col_cat.addWidget(self._fld_category)
        row1.addLayout(col_cat, 1)

        col_status = QVBoxLayout()
        col_status.addWidget(self._field_label("⚡ 상태"))
        self._fld_status = QComboBox()
        self._fld_status.addItems(["planned", "beta", "ready", "done"])
        self._fld_status.setStyleSheet(_form_input_style())
        self._fld_status.currentTextChanged.connect(self._schedule_save)
        col_status.addWidget(self._fld_status)
        row1.addLayout(col_status, 1)

        col_dl = QVBoxLayout()
        col_dl.addWidget(self._field_label("📅 기한 (YYYY-MM-DD)"))
        self._fld_deadline = QLineEdit()
        self._fld_deadline.setPlaceholderText("2026-06-30")
        self._fld_deadline.setStyleSheet(_form_input_style())
        self._fld_deadline.textChanged.connect(self._schedule_save)
        col_dl.addWidget(self._fld_deadline)
        row1.addLayout(col_dl, 1)
        form_layout.addLayout(row1)

        # [2026-05-31 사용자 요청] 프롬프트 — 메인 채팅창이 받아서 직접 실행하는 명령.
        # 입력 후 [▶ 실행] 클릭 시 ChatWindow.input_line 에 prefill + send_message().
        form_layout.addWidget(self._field_label(
            "💬 프롬프트 (실행 시 메인 채팅창이 이 명령을 직접 실행)"
        ))
        self._fld_prompt = QPlainTextEdit()
        self._fld_prompt.setFixedHeight(90)
        self._fld_prompt.setPlaceholderText(
            "예) 크몽 프리미엄 업무자동화 기그 상세설명 초안을 작성해줘"
        )
        self._fld_prompt.setStyleSheet(_form_input_style())
        self._fld_prompt.textChanged.connect(self._schedule_save)
        form_layout.addWidget(self._fld_prompt)
        prompt_btn_row = QHBoxLayout()
        prompt_btn_row.addStretch(1)
        # [2026-05-31 사용자 요청] Auto Prompt — EIDOS 가 프로젝트 정보 기반 프롬프트 추천.
        self._btn_auto_prompt = QPushButton("✨  Auto Prompt")
        self._btn_auto_prompt.setCursor(Qt.PointingHandCursor)
        self._btn_auto_prompt.setStyleSheet(_small_btn_style(primary=False))
        self._btn_auto_prompt.setToolTip(
            "EIDOS 가 이 프로젝트(제목·카테고리·기한·설명·준비조건·파일) 를 보고 "
            "바로 실행 가능한 프롬프트를 추천해 위 필드에 채웁니다."
        )
        self._btn_auto_prompt.clicked.connect(self._on_auto_prompt)
        prompt_btn_row.addWidget(self._btn_auto_prompt)
        self._btn_run_prompt = QPushButton("▶  메인 채팅창에서 실행")
        self._btn_run_prompt.setCursor(Qt.PointingHandCursor)
        self._btn_run_prompt.setStyleSheet(_small_btn_style(primary=True))
        self._btn_run_prompt.setToolTip(
            "프롬프트를 메인 채팅창 입력란에 넣고 즉시 전송 — EIDOS 가 직접 실행."
        )
        self._btn_run_prompt.clicked.connect(self._on_run_prompt)
        prompt_btn_row.addWidget(self._btn_run_prompt)
        form_layout.addLayout(prompt_btn_row)

        # [2026-05-26 사용자 요청] 설명 (간단히) / 준비조건 두 영역 숨김
        # attr 은 유지 — load_into_form/_flush_save 의 setText/toPlainText 호출 안전
        _lbl_desc = self._field_label("📝 설명 (간단히)")
        _lbl_desc.setVisible(False)
        form_layout.addWidget(_lbl_desc)
        self._fld_description = QPlainTextEdit()
        self._fld_description.setFixedHeight(80)
        self._fld_description.setStyleSheet(_form_input_style())
        self._fld_description.textChanged.connect(self._schedule_save)
        self._fld_description.setVisible(False)
        form_layout.addWidget(self._fld_description)

        _lbl_prereq = self._field_label("🔧 준비조건 (시작 전 필요한 것)")
        _lbl_prereq.setVisible(False)
        form_layout.addWidget(_lbl_prereq)
        self._fld_prerequisites = QPlainTextEdit()
        self._fld_prerequisites.setFixedHeight(70)
        self._fld_prerequisites.setStyleSheet(_form_input_style())
        self._fld_prerequisites.textChanged.connect(self._schedule_save)
        self._fld_prerequisites.setVisible(False)
        form_layout.addWidget(self._fld_prerequisites)

        # 입력 파일 (파일 첨부 리스트)
        form_layout.addWidget(self._field_label("📥 입력 파일"))
        self._lst_input_files = QListWidget()
        self._lst_input_files.setFixedHeight(90)
        self._lst_input_files.setStyleSheet(_form_input_style())
        # [2026-05-26 사용자 요청] 더블클릭 시 OS 기본 프로그램으로 파일 열기
        self._lst_input_files.itemDoubleClicked.connect(self._on_file_item_double_clicked)
        form_layout.addWidget(self._lst_input_files)
        in_btn_row = QHBoxLayout()
        in_btn_row.addStretch(1)
        self._btn_add_input = QPushButton("➕  파일 추가")
        self._btn_add_input.setCursor(Qt.PointingHandCursor)
        self._btn_add_input.setStyleSheet(_small_btn_style(primary=False))
        self._btn_add_input.clicked.connect(lambda: self._on_add_files(self._lst_input_files, "input_files"))
        self._btn_remove_input = QPushButton("🗑  선택 제거")
        self._btn_remove_input.setCursor(Qt.PointingHandCursor)
        self._btn_remove_input.setStyleSheet(_small_btn_style(primary=False))
        self._btn_remove_input.clicked.connect(lambda: self._on_remove_file(self._lst_input_files, "input_files"))
        in_btn_row.addWidget(self._btn_add_input)
        in_btn_row.addWidget(self._btn_remove_input)
        form_layout.addLayout(in_btn_row)

        # [2026-05-26 사용자 요청] 출력/납품 (설명) + 납품 파일 영역 숨김
        # attr 은 유지 — load_into_form/_flush_save 의 setText/toPlainText 호출 안전
        _lbl_deliver = self._field_label("📤 출력 / 납품 (설명)")
        _lbl_deliver.setVisible(False)
        form_layout.addWidget(_lbl_deliver)
        self._fld_deliverables = QPlainTextEdit()
        self._fld_deliverables.setFixedHeight(70)
        self._fld_deliverables.setStyleSheet(_form_input_style())
        self._fld_deliverables.textChanged.connect(self._schedule_save)
        self._fld_deliverables.setVisible(False)
        form_layout.addWidget(self._fld_deliverables)

        # [2026-05-26 사용자 요청] 출력 파일 영역 다시 활성화 — 입력 파일과 대칭 구조
        _lbl_deliver_files = self._field_label("📤 출력 파일")
        form_layout.addWidget(_lbl_deliver_files)
        self._lst_deliverable_files = QListWidget()
        self._lst_deliverable_files.setFixedHeight(90)
        self._lst_deliverable_files.setStyleSheet(_form_input_style())
        # [2026-05-26 사용자 요청] 더블클릭 시 OS 기본 프로그램으로 파일 열기
        self._lst_deliverable_files.itemDoubleClicked.connect(self._on_file_item_double_clicked)
        form_layout.addWidget(self._lst_deliverable_files)
        out_btn_row_wrap = QWidget()
        out_btn_row = QHBoxLayout(out_btn_row_wrap)
        out_btn_row.setContentsMargins(0, 0, 0, 0)
        out_btn_row.addStretch(1)
        self._btn_add_out = QPushButton("➕  파일 추가")
        self._btn_add_out.setCursor(Qt.PointingHandCursor)
        self._btn_add_out.setStyleSheet(_small_btn_style(primary=False))
        self._btn_add_out.clicked.connect(lambda: self._on_add_files(self._lst_deliverable_files, "deliverable_files"))
        self._btn_remove_out = QPushButton("🗑  선택 제거")
        self._btn_remove_out.setCursor(Qt.PointingHandCursor)
        self._btn_remove_out.setStyleSheet(_small_btn_style(primary=False))
        self._btn_remove_out.clicked.connect(lambda: self._on_remove_file(self._lst_deliverable_files, "deliverable_files"))
        out_btn_row.addWidget(self._btn_add_out)
        out_btn_row.addWidget(self._btn_remove_out)
        form_layout.addWidget(out_btn_row_wrap)

        # 조립된 액션 툴 (v2.2 — 캔버스 첨부/제거/실행 wire 활성화)
        form_layout.addWidget(self._field_label(
            "🎨 조립된 액션 (액션 조립 캔버스에서 만든 자동화)"
        ))
        self._lst_actions = QListWidget()
        self._lst_actions.setFixedHeight(100)
        self._lst_actions.setStyleSheet(_form_input_style())
        form_layout.addWidget(self._lst_actions)
        action_btn_row = QHBoxLayout()
        action_btn_row.addStretch(1)
        self._btn_attach_canvas = QPushButton("➕  캔버스 첨부")
        self._btn_attach_canvas.setCursor(Qt.PointingHandCursor)
        self._btn_attach_canvas.setStyleSheet(_small_btn_style(primary=False))
        self._btn_attach_canvas.clicked.connect(self._on_attach_canvas)
        self._btn_remove_canvas = QPushButton("🗑  선택 제거")
        self._btn_remove_canvas.setCursor(Qt.PointingHandCursor)
        self._btn_remove_canvas.setStyleSheet(_small_btn_style(primary=False))
        self._btn_remove_canvas.clicked.connect(self._on_remove_attached_canvas)
        self._btn_run_canvas = QPushButton("▶  선택 실행")
        self._btn_run_canvas.setCursor(Qt.PointingHandCursor)
        self._btn_run_canvas.setStyleSheet(_small_btn_style(primary=True))
        self._btn_run_canvas.clicked.connect(self._on_run_attached_canvas)
        action_btn_row.addWidget(self._btn_attach_canvas)
        action_btn_row.addWidget(self._btn_remove_canvas)
        action_btn_row.addWidget(self._btn_run_canvas)
        form_layout.addLayout(action_btn_row)

        # 완료 체크
        self._chk_done = QCheckBox("✅  이 프로젝트 완료로 표시")
        self._chk_done.stateChanged.connect(self._schedule_save)
        form_layout.addWidget(self._chk_done)

        form_layout.addStretch(1)
        self._form_scroll.setWidget(self._form_content)
        right_layout.addWidget(self._form_scroll)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([330, 670])
        root.addWidget(splitter, 1)

        # 처음엔 form 비활성 (프로젝트 없음)
        self._set_form_enabled(False)

    # ── 헬퍼 ─────────────────────────────────────────────────────
    def _on_file_item_double_clicked(self, item) -> None:
        """[2026-05-26] 입력/출력 파일 list 더블클릭 → OS 기본 프로그램으로 파일 열기.

        Windows: os.startfile / macOS: open / Linux: xdg-open.
        파일이 존재하지 않으면 경고 메시지박스.
        """
        try:
            path = item.text().strip() if item else ""
            if not path:
                return
            if not os.path.exists(path):
                QMessageBox.warning(
                    self, "파일 없음",
                    f"다음 경로의 파일을 찾을 수 없습니다:\n{path}\n\n"
                    "파일이 이동·삭제됐거나 경로가 잘못 입력됐을 수 있습니다.",
                )
                return
            import sys as _sys
            import subprocess as _sp
            if _sys.platform == "win32":
                os.startfile(path)  # type: ignore[attr-defined]
            elif _sys.platform == "darwin":
                _sp.call(["open", path])
            else:
                _sp.call(["xdg-open", path])
        except Exception as e:
            try:
                QMessageBox.warning(
                    self, "파일 열기 실패",
                    f"{type(e).__name__}: {e}",
                )
            except Exception:
                pass

    def _field_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {_Palette.TEXT_MUTED}; font-size: 11px; font-weight: 600;"
            f" background: transparent; padding-bottom: 2px;"
        )
        return lbl

    def _set_form_enabled(self, enabled: bool) -> None:
        for w in (
            self._fld_title, self._fld_category, self._fld_status, self._fld_deadline,
            self._fld_prompt, self._btn_run_prompt, self._btn_auto_prompt,
            self._fld_description, self._fld_prerequisites, self._fld_deliverables,
            self._lst_input_files, self._lst_deliverable_files,
            self._btn_add_input, self._btn_remove_input,
            self._btn_add_out, self._btn_remove_out,
            self._lst_actions, self._btn_attach_canvas,
            self._btn_remove_canvas, self._btn_run_canvas,
            self._chk_done,
        ):
            try:
                w.setEnabled(enabled)
            except Exception:
                pass

    def _current_project(self) -> Optional[dict]:
        if not self._current_id:
            return None
        for p in self._projects:
            if p.get("id") == self._current_id:
                return p
        return None

    # ── 리스트 갱신 ──────────────────────────────────────────────
    def _refresh_list(self) -> None:
        # 기존 버튼 제거
        for b in self._buttons:
            try:
                self._buttons_layout.removeWidget(b)
                b.deleteLater()
            except Exception:
                pass
        self._buttons = []
        insert_idx = self._buttons_layout.count() - 1
        for proj in self._projects:
            btn = _ItemButton(proj, self._scroll_content_left)
            btn.clicked.connect(lambda _checked=False, b=btn: self._on_clicked(b))
            self._buttons_layout.insertWidget(insert_idx, btn)
            insert_idx += 1
            self._buttons.append(btn)

    # ── 클릭 / CRUD ─────────────────────────────────────────────
    def _on_clicked(self, btn: _ItemButton) -> None:
        proj = btn.item_data
        self._current_id = proj.get("id")
        self._load_into_form(proj)

    def _on_new_clicked(self) -> None:
        p = _proj_store.create_project(self._projects, title="새 프로젝트")
        _proj_store.save_projects(self._projects)
        self._current_id = p.get("id")
        self._refresh_list()
        # 신규 버튼 자동 선택
        for b in self._buttons:
            if b.item_data.get("id") == self._current_id:
                b.setChecked(True)
                self._load_into_form(p)
                break
        self._update_save_status("새 프로젝트 추가됨")

    def _on_delete_clicked(self) -> None:
        proj = self._current_project()
        if not proj:
            return
        reply = QMessageBox.question(
            self, "프로젝트 삭제",
            f"'{proj.get('title','(이름 없음)')}' 프로젝트를 삭제할까요?\n"
            f"이 작업은 되돌릴 수 없습니다.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        ok = _proj_store.delete_project(self._projects, self._current_id or "")
        if ok:
            _proj_store.save_projects(self._projects)
            self._current_id = None
            self._refresh_list()
            if self._projects and self._buttons:
                self._buttons[0].setChecked(True)
                self._on_clicked(self._buttons[0])
            else:
                self._clear_form()
                self._set_form_enabled(False)
                self._form_header.setText("➕ 새 프로젝트를 만드세요")
            self._update_save_status("삭제 완료")

    # ── form 로드 / 저장 ─────────────────────────────────────────
    def _load_into_form(self, proj: dict) -> None:
        self._suppress_save = True
        try:
            self._fld_title.setText(str(proj.get("title", "")))
            self._fld_category.setText(str(proj.get("category", "")))
            status = str(proj.get("status", "planned"))
            idx = self._fld_status.findText(status)
            if idx >= 0:
                self._fld_status.setCurrentIndex(idx)
            else:
                self._fld_status.setCurrentIndex(0)
            self._fld_deadline.setText(str(proj.get("deadline", "")))
            self._fld_prompt.setPlainText(str(proj.get("prompt", "")))
            self._fld_description.setPlainText(str(proj.get("description", "")))
            self._fld_prerequisites.setPlainText(str(proj.get("prerequisites", "")))
            self._fld_deliverables.setPlainText(str(proj.get("deliverables", "")))
            self._lst_input_files.clear()
            for fp in proj.get("input_files", []) or []:
                QListWidgetItem(str(fp), self._lst_input_files)
            self._lst_deliverable_files.clear()
            for fp in proj.get("deliverable_files", []) or []:
                QListWidgetItem(str(fp), self._lst_deliverable_files)
            self._lst_actions.clear()
            for aid in proj.get("assembled_actions", []) or []:
                QListWidgetItem(str(aid), self._lst_actions)
            self._chk_done.setChecked(bool(proj.get("done", False)))
            icon = proj.get("icon", "📁")
            self._form_header.setText(f"{icon}   {proj.get('title','(이름 없음)')}")
            self._set_form_enabled(True)
            self._update_save_status("저장됨")
        finally:
            self._suppress_save = False

    def _clear_form(self) -> None:
        self._suppress_save = True
        try:
            for w in (self._fld_title, self._fld_category, self._fld_deadline):
                w.setText("")
            for w in (self._fld_prompt, self._fld_description, self._fld_prerequisites,
                      self._fld_deliverables):
                w.setPlainText("")
            self._fld_status.setCurrentIndex(0)
            self._lst_input_files.clear()
            self._lst_deliverable_files.clear()
            self._lst_actions.clear()
            self._chk_done.setChecked(False)
        finally:
            self._suppress_save = False

    def _schedule_save(self) -> None:
        """필드 변경 시 디바운스 — 500ms 후 _flush_save."""
        if self._suppress_save:
            return
        try:
            self._save_timer.start()
            self._form_save_status.setText("⏳ 저장 중...")
            self._form_save_status.setStyleSheet(
                f"color: {_Palette.TEXT_MUTED}; font-size: 11px; background: transparent;"
            )
        except Exception:
            pass

    def _flush_save(self) -> None:
        """현재 form 값 → in-memory project → 디스크 save."""
        proj = self._current_project()
        if not proj:
            return
        try:
            fields = {
                "title": self._fld_title.text().strip() or "(이름 없음)",
                "category": self._fld_category.text().strip() or "기타",
                "status": self._fld_status.currentText(),
                "deadline": self._fld_deadline.text().strip(),
                "prompt": self._fld_prompt.toPlainText(),
                "description": self._fld_description.toPlainText(),
                "prerequisites": self._fld_prerequisites.toPlainText(),
                "deliverables": self._fld_deliverables.toPlainText(),
                "done": self._chk_done.isChecked(),
            }
            _proj_store.update_project(self._projects, proj["id"], **fields)
            _proj_store.save_projects(self._projects)
            # 좌측 버튼 제목 갱신 (제목 바뀐 경우)
            for b in self._buttons:
                if b.item_data.get("id") == proj["id"]:
                    new_text = f"  {proj.get('icon','📁')}   {fields['title']}"
                    b.setText(new_text)
                    break
            # form 헤더 갱신
            self._form_header.setText(f"{proj.get('icon','📁')}   {fields['title']}")
            self._update_save_status("저장됨")
        except Exception as e:
            print(f"[ProjectsTab] _flush_save 실패 (graceful): {e}")
            self._update_save_status(f"⚠️ 저장 실패: {e}")

    def _update_save_status(self, msg: str) -> None:
        try:
            color = _Palette.STATUS_READY if "저장됨" in msg or "추가" in msg or "삭제" in msg \
                else _Palette.TEXT_MUTED
            self._form_save_status.setText(msg)
            self._form_save_status.setStyleSheet(
                f"color: {color}; font-size: 11px; background: transparent;"
            )
        except Exception:
            pass

    # ── 파일 첨부 ───────────────────────────────────────────────
    def _on_add_files(self, list_widget: QListWidget, field_key: str) -> None:
        proj = self._current_project()
        if not proj:
            return
        files, _ = QFileDialog.getOpenFileNames(self, "파일 선택", "", "모든 파일 (*.*)")
        if not files:
            return
        existing = set(str(list_widget.item(i).text()) for i in range(list_widget.count()))
        for f in files:
            if f not in existing:
                QListWidgetItem(f, list_widget)
        proj[field_key] = [list_widget.item(i).text() for i in range(list_widget.count())]
        _proj_store.update_project(self._projects, proj["id"], **{field_key: proj[field_key]})
        _proj_store.save_projects(self._projects)
        self._update_save_status(f"파일 {len(files)}개 추가됨")

    def _on_remove_file(self, list_widget: QListWidget, field_key: str) -> None:
        proj = self._current_project()
        if not proj:
            return
        sel = list_widget.selectedItems()
        if not sel:
            return
        for item in sel:
            list_widget.takeItem(list_widget.row(item))
        proj[field_key] = [list_widget.item(i).text() for i in range(list_widget.count())]
        _proj_store.update_project(self._projects, proj["id"], **{field_key: proj[field_key]})
        _proj_store.save_projects(self._projects)
        self._update_save_status(f"파일 {len(sel)}개 제거됨")

    # ── [2026-05-31] 프롬프트 → 메인 채팅창 직접 실행 ─────────────
    def _on_run_prompt(self) -> None:
        """프롬프트 필드 내용을 메인 채팅창 입력란에 넣고 즉시 전송.
        ChatWindow.input_line(setText) + send_message() — EIDOS 가 직접 실행."""
        prompt = self._fld_prompt.toPlainText().strip()
        if not prompt:
            QMessageBox.information(
                self, "프롬프트 필요",
                "먼저 프롬프트 필드에 실행할 명령을 입력하세요.",
            )
            return
        # 저장 보류 중이면 먼저 flush (현재 prompt 디스크 반영)
        try:
            if self._save_timer.isActive():
                self._save_timer.stop()
                self._flush_save()
        except Exception:
            pass
        cw = _find_chat_window(self.parent() if self.parent() else self)
        if cw is None or not hasattr(cw, "input_line") or not hasattr(cw, "send_message"):
            QMessageBox.warning(
                self, "오류",
                "메인 채팅창(ChatWindow)을 찾을 수 없습니다.",
            )
            return
        # QLineEdit 은 단일 줄 — 개행은 공백으로 정규화해서 전달.
        one_line = " ".join(prompt.split())
        try:
            cw.input_line.setText(one_line)
            cw.send_message()
            # 다이얼로그 닫아 메인 채팅창 진행을 바로 볼 수 있게.
            try:
                p = self
                while p is not None and not isinstance(p, QDialog):
                    p = p.parent()
                if isinstance(p, QDialog):
                    p.accept()
            except Exception:
                pass
        except Exception as e:
            QMessageBox.warning(self, "실행 실패", f"{type(e).__name__}: {e}")

    # ── [2026-05-31] Auto Prompt — EIDOS 가 프로젝트 맞춤 프롬프트 추천 ──
    def _on_auto_prompt(self) -> None:
        """현재 프로젝트 정보 → LLM 으로 실행 프롬프트 1개 추천 → 프롬프트 필드에 채움.
        worker thread 에서 LLM 호출, 결과는 _auto_prompt_ready 시그널로 main thread 갱신."""
        proj = self._current_project()
        if not proj:
            QMessageBox.information(self, "프로젝트 필요", "먼저 프로젝트를 선택하세요.")
            return
        cw = _find_chat_window(self.parent() if self.parent() else self)
        worker = getattr(cw, "eidos_worker", None) if cw else None
        if worker is None or not hasattr(worker, "submit_task"):
            QMessageBox.warning(
                self, "오류",
                "메인 채팅창(eidos_worker)을 찾을 수 없습니다.\n"
                "추천은 메인 EIDOS 가 실행 중일 때만 가능합니다.",
            )
            return

        pid = proj.get("id", "") or ""
        # 프로젝트 컨텍스트 수집 (UI 에서 숨겨진 필드도 저장값 사용)
        input_files = [
            os.path.basename(str(f)) for f in (proj.get("input_files") or [])
        ]
        ctx_lines = [
            f"제목: {proj.get('title','') or '(없음)'}",
            f"카테고리: {proj.get('category','') or '(없음)'}",
            f"상태: {proj.get('status','') or '(없음)'}",
            f"기한: {proj.get('deadline','') or '(없음)'}",
            f"설명: {proj.get('description','') or '(없음)'}",
            f"준비조건: {proj.get('prerequisites','') or '(없음)'}",
            f"납품물: {proj.get('deliverables','') or '(없음)'}",
            f"입력 파일: {', '.join(input_files) if input_files else '(없음)'}",
        ]
        ctx_text = "\n".join(ctx_lines)
        current_prompt = self._fld_prompt.toPlainText().strip()

        # 로딩 표시
        self._btn_auto_prompt.setEnabled(False)
        self._btn_auto_prompt.setText("✨  생성 중...")

        async def _bg():
            try:
                from llm_module import get_llm_response_async as _llm_call
                sys_prompt = (
                    "너는 EIDOS, 사용자의 자율 업무 자동화 에이전트다.\n"
                    "주어진 프로젝트 정보를 보고, 메인 채팅창에 그대로 붙여넣어 즉시 실행할 수 있는 "
                    "한국어 '실행 프롬프트' 1개를 작성하라.\n"
                    "- 구체적인 행동 지시문이어야 한다 (무엇을·어떻게·산출물 형태).\n"
                    "- 2~5문장, 한 문단.\n"
                    "- 인사말·메타 설명·따옴표·코드블록·머리말 없이 프롬프트 본문만 출력하라."
                )
                hint = (
                    f"\n\n(참고: 사용자가 이미 적어둔 프롬프트 초안 — 이를 개선/구체화해도 됨)\n{current_prompt}"
                    if current_prompt else ""
                )
                user_prompt = (
                    f"[프로젝트 정보]\n{ctx_text}{hint}\n\n"
                    "위 프로젝트를 진전시키기 위해 EIDOS 에게 시킬 실행 프롬프트 1개를 작성하라."
                )
                raw = (await _llm_call(
                    user_prompt, system_prompt=sys_prompt, max_tokens=1024, timeout=120,
                )) or ""
                raw = raw.strip()
                self._auto_prompt_ready.emit(pid, raw)
            except Exception as e:
                self._auto_prompt_ready.emit(pid, f"__ERROR__{type(e).__name__}: {e}")

        try:
            worker.submit_task(_bg())
        except Exception as e:
            self._btn_auto_prompt.setEnabled(True)
            self._btn_auto_prompt.setText("✨  Auto Prompt")
            QMessageBox.warning(self, "실행 실패", f"worker.submit_task 실패: {e}")

    def _on_auto_prompt_ready(self, project_id: str, text: str) -> None:
        """[main thread] LLM 추천 결과 수신 → 프롬프트 필드 채움 + 자동 저장."""
        # 버튼 복원
        try:
            self._btn_auto_prompt.setEnabled(True)
            self._btn_auto_prompt.setText("✨  Auto Prompt")
        except Exception:
            pass
        text = (text or "").strip()
        if text.startswith("__ERROR__"):
            QMessageBox.warning(self, "Auto Prompt 실패", text[len("__ERROR__"):])
            return
        if (not text) or text.startswith("LLM 오류") or text.startswith("[서버"):
            QMessageBox.warning(
                self, "Auto Prompt 실패", f"추천 생성 실패: {text[:160] or '(빈 응답)'}",
            )
            return
        # 양끝 따옴표·코드펜스 제거 (LLM 이 가끔 감쌈)
        if text.startswith("```"):
            text = text.strip("`").strip()
        text = text.strip().strip('"').strip("'").strip()
        # 사용자가 그새 다른 프로젝트로 이동했으면, 추천은 원 프로젝트에만 저장.
        if project_id and self._current_id and project_id != self._current_id:
            try:
                _proj_store.update_project(self._projects, project_id, prompt=text)
                _proj_store.save_projects(self._projects)
            except Exception:
                pass
            QMessageBox.information(
                self, "✨ Auto Prompt 완료",
                "추천 프롬프트가 생성되어 원래 프로젝트에 저장됐습니다.\n"
                "(현재는 다른 프로젝트를 보고 있어 화면에는 채우지 않았습니다.)",
            )
            return
        # 현재 프로젝트 — 필드에 채움 (textChanged → _schedule_save 로 자동 저장)
        self._fld_prompt.setPlainText(text)

    # ── v2.2 캔버스 첨부 / 제거 / 실행 ───────────────────────────
    def _on_attach_canvas(self) -> None:
        """저장된 캔버스 목록에서 선택 → assembled_actions append + save."""
        proj = self._current_project()
        if not proj:
            return
        try:
            import eidos_canvas_store as _cstore
        except Exception as e:
            QMessageBox.warning(self, "오류", f"eidos_canvas_store 로드 실패: {e}")
            return
        names = _cstore.list_canvases()
        if not names:
            QMessageBox.information(
                self, "📂 캔버스 없음",
                "저장된 캔버스가 없습니다.\n먼저 더보기 → 🎨 액션 조립 캔버스 탭에서 캔버스를 만들고 저장하세요.",
            )
            return
        from PySide6.QtWidgets import QInputDialog
        chosen, ok = QInputDialog.getItem(
            self, "🎨 캔버스 첨부", "프로젝트에 첨부할 캔버스 선택:", names, 0, False,
        )
        if not ok or not chosen:
            return
        existing = list(proj.get("assembled_actions") or [])
        if chosen in existing:
            QMessageBox.information(self, "이미 첨부됨", f"'{chosen}' 캔버스는 이미 첨부되어 있습니다.")
            return
        existing.append(chosen)
        # in-place + save
        QListWidgetItem(chosen, self._lst_actions)
        _proj_store.update_project(self._projects, proj["id"], assembled_actions=existing)
        _proj_store.save_projects(self._projects)
        self._update_save_status(f"캔버스 '{chosen}' 첨부됨")

    def _on_remove_attached_canvas(self) -> None:
        proj = self._current_project()
        if not proj:
            return
        sel = self._lst_actions.selectedItems()
        if not sel:
            QMessageBox.information(self, "선택 필요", "제거할 캔버스를 목록에서 선택하세요.")
            return
        for item in sel:
            self._lst_actions.takeItem(self._lst_actions.row(item))
        new_list = [self._lst_actions.item(i).text() for i in range(self._lst_actions.count())]
        _proj_store.update_project(self._projects, proj["id"], assembled_actions=new_list)
        _proj_store.save_projects(self._projects)
        self._update_save_status(f"캔버스 {len(sel)}개 제거됨")

    def _on_run_attached_canvas(self) -> None:
        """선택된 캔버스를 백그라운드 실행 — 진행 UI 없이 worker.submit_task + 채팅창 완료 안내.
        자세한 노드 색상/중단/ASK_USER 는 캔버스 탭의 _CanvasTab 에서."""
        sel = self._lst_actions.selectedItems()
        if not sel:
            QMessageBox.information(self, "선택 필요", "실행할 캔버스를 목록에서 선택하세요.")
            return
        canvas_name = sel[0].text().strip()
        if not canvas_name:
            return
        try:
            import eidos_canvas_store as _cstore
        except Exception as e:
            QMessageBox.warning(self, "오류", f"canvas_store 로드 실패: {e}")
            return
        canvas = _cstore.load_canvas(canvas_name)
        if not canvas.get("nodes"):
            QMessageBox.warning(self, "빈 캔버스", f"'{canvas_name}' 은 노드가 없습니다.")
            return
        cw = _find_chat_window(self.parent() if self.parent() else self)
        worker = getattr(cw, "eidos_worker", None) if cw else None
        if worker is None:
            QMessageBox.warning(self, "오류", "ChatWindow.eidos_worker 부재.")
            return
        try:
            from eidos_canvas_runner import execute_canvas_async
        except Exception as e:
            QMessageBox.warning(self, "오류", f"canvas_runner 로드 실패: {e}")
            return

        # 진행 callback — 단순화 (캔버스 탭 같은 노드 UI 없음·콘솔 print + 채팅창 완료 안내만)
        def _on_progress(event: str, node_id: str, payload: dict):
            try:
                if event == "all_done":
                    total = payload.get("total", 0)
                    ok = payload.get("ok", 0)
                    fail = payload.get("fail", 0)
                    skip = payload.get("skip", 0)
                    stopped = payload.get("stopped", 0)
                    stopped_part = f" / 🛑 {stopped}" if stopped else ""
                    msg = (f"🎨 [프로젝트 캔버스 실행 완료] '{canvas_name}' "
                           f"— ✅ {ok} / ❌ {fail} / ⏭ {skip}{stopped_part} (총 {total})")
                    try:
                        if cw and hasattr(cw, "_safe_append"):
                            cw._safe_append(msg, "system")
                    except Exception:
                        pass
            except Exception as e:
                print(f"[ProjectsTab canvas progress] {e}")
            return None

        try:
            worker.submit_task(execute_canvas_async(canvas, on_progress=_on_progress,
                                                    stop_on_fail=False))
            try:
                if cw and hasattr(cw, "_safe_append"):
                    cw._safe_append(
                        f"🎨 [프로젝트 캔버스 실행 시작] '{canvas_name}' "
                        f"({len(canvas.get('nodes', []))}개 노드·백그라운드 진행)",
                        "system",
                    )
            except Exception:
                pass
            QMessageBox.information(
                self, "▶ 실행 시작",
                f"캔버스 '{canvas_name}' 백그라운드 실행 시작.\n"
                "완료 시 메인 채팅창에 안내됩니다.\n\n"
                "💡 자세한 진행 (노드 색상·중단·ASK_USER 입력) 을 보려면 "
                "더보기 → 🎨 액션 조립 캔버스 탭에서 동일 캔버스를 로드해 실행하세요.",
            )
        except Exception as e:
            QMessageBox.warning(self, "실행 실패", f"{type(e).__name__}: {e}")


def _scroll_style() -> str:
    """공용 QScrollArea 스타일 — 좌측 리스트 / 우측 form 둘 다 사용."""
    return (
        f"QScrollArea {{ border: none; background: transparent; }}"
        f"QScrollBar:vertical {{"
        f"  background: transparent;"
        f"  width: 10px;"
        f"  margin: 4px 2px 4px 0;"
        f"}}"
        f"QScrollBar::handle:vertical {{"
        f"  background: {_Palette.BORDER_HANDLE};"
        f"  border-radius: 5px;"
        f"  min-height: 30px;"
        f"}}"
        f"QScrollBar::handle:vertical:hover {{ background: {_Palette.ACCENT}; }}"
        f"QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}"
        f"QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}"
    )


# ── 메인 다이얼로그 (3 탭) ──────────────────────────────────────────────

class MoreActionsDialog(QDialog):
    """[2026-05-25 v2.0] 햄버거 메뉴 '🧰 더보기' — 자동화 액션 모음 다이얼로그.

    3 탭 구조:
      📁 프로젝트          : PROJECTS 6개 (좌측 스크롤 + 우측 상세)
      🧰 자동화 액션 모음  : ACTIONS 8개 (좌측 스크롤 + 우측 상세)
      🎨 액션 조립 캔버스  : v2.x ActionDispatcher schema 비주얼 빌더 placeholder

    테마: 순검정 + 보라빛 발광 그라데이션. 좌측 버튼은 가로 그라데이션
    (왼쪽 진한 보라 → 오른쪽 검정), 호버/체크 시 더 밝은 보라.
    """

    action_selected = Signal(dict)  # 액션 탭 선택 (후방 호환)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("🧰 더보기 — 자동화 액션 모음")
        self.resize(1500, 900)
        self.setMinimumSize(1240, 680)
        self._build_ui()

    def _build_ui(self) -> None:
        # [2026-05-26 라이트 테마 동적 적용] settings.json 의 theme 가 White/Light 면
        # 다이얼로그 root 에 라이트 stylesheet override 적용 — 자식 위젯 inherit.
        if _LIGHT:
            self.setStyleSheet(_root_theme_override())

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # 헤더 (보라 발광)
        header = QLabel("🧰  더보기 — 자동화 액션 모음")
        header.setFont(QFont("", 14, QFont.Bold))
        header.setStyleSheet(
            f"color: {_Palette.ACCENT_GLOW}; padding: 8px 6px; letter-spacing: 0.5px;"
            f" background: transparent;"
        )
        _apply_purple_glow(header, radius=22, color_hex=_Palette.ACCENT, alpha=140)
        root.addWidget(header)

        subtitle = QLabel(
            "프로젝트 단위로 관리하거나 개별 액션을 실행하세요. "
            "탭으로 전환 — 향후 캔버스에서 새 워크플로우 조립 가능."
        )
        subtitle.setStyleSheet(
            f"color: {_Palette.TEXT_MUTED}; font-size: 11px; padding: 0 6px 6px 6px;"
            f" background: transparent;"
        )
        root.addWidget(subtitle)

        # 탭 위젯
        self._tabs = QTabWidget(self)
        self._tabs.setStyleSheet(self._tabs_stylesheet())

        # 탭 1: 프로젝트 (CRUD + 자동 저장 + form)
        self._tab_projects = _ProjectsTab(parent=self)
        self._tabs.addTab(self._tab_projects, "📁  프로젝트")

        # 탭 2: 자동화 액션 모음 (기본 ACTIONS + 사용자 캔버스 자동 노출 v2.3)
        actions_with_canvas = list(ACTIONS)
        try:
            import eidos_canvas_store as _cstore_for_actions
            for cname in _cstore_for_actions.list_canvases():
                actions_with_canvas.append({
                    "id": f"canvas:{cname}",
                    "icon": "🎨",
                    "title": cname,
                    "category": "사용자 캔버스",
                    "status": "ready",
                    "description": (
                        f"액션 조립 캔버스에서 사용자가 직접 만든 자동화: '{cname}'\n\n"
                        "▶ 실행 시 백그라운드 호출 — 완료되면 메인 채팅창에 안내됩니다.\n"
                        "자세한 진행 (노드 색상·중단·ASK_USER 입력) 을 보려면 "
                        "🎨 액션 조립 캔버스 탭에서 이 캔버스를 로드해 실행하세요."
                    ),
                })
        except Exception as _e_canv:
            print(f"[MoreActionsDialog] 사용자 캔버스 자동 노출 실패 (graceful): {_e_canv}")
        self._tab_actions = _LeftRightItemPanel(
            actions_with_canvas,
            run_btn_label="▶  실행",
            parent=self,
        )
        self._tabs.addTab(self._tab_actions, "🧰  자동화 액션 모음")
        # 후방 호환 — 액션 탭의 item_selected 를 dialog 의 action_selected 로 전파
        try:
            self._tab_actions.item_selected.connect(self.action_selected.emit)
        except Exception:
            pass

        # 탭 3: 액션 조립 캔버스 (placeholder)
        self._tab_canvas = _CanvasTab(self)
        self._tabs.addTab(self._tab_canvas, "🎨  액션 조립 캔버스")

        # [2026-05-31 사용자 요청] 프로젝트 탭을 default 로 (창 열면 제일 먼저 보이게)
        self._tabs.setCurrentIndex(0)

        root.addWidget(self._tabs, 1)

        self.setStyleSheet(
            f"QDialog {{ background: {_Palette.BG_DEEP}; }}"
            f"QLabel {{ background: transparent; }}"
        )

    def _tabs_stylesheet(self) -> str:
        """QTabWidget + QTabBar 스타일 — 검정 배경 + 보라 그라데이션 (선택 시)."""
        return (
            f"QTabWidget::pane {{"
            f"  border: 1px solid {_Palette.BORDER_MUTED};"
            f"  background: {_Palette.BG_DEEP};"
            f"  border-radius: 10px;"
            f"  top: -1px;"
            f"}}"
            f"QTabWidget::tab-bar {{ left: 12px; }}"
            f"QTabBar::tab {{"
            f"  background: {_Palette.BG_DEEP};"
            f"  color: {_Palette.TEXT_MUTED};"
            f"  padding: 10px 22px;"
            f"  border: 1px solid {_Palette.BORDER_MUTED};"
            f"  border-bottom: none;"
            f"  border-top-left-radius: 8px;"
            f"  border-top-right-radius: 8px;"
            f"  margin-right: 3px;"
            f"  font-size: 13px;"
            f"  font-weight: 600;"
            f"}}"
            f"QTabBar::tab:hover {{"
            f"  color: {_Palette.ACCENT_GLOW};"
            f"  background: {_Palette.TAB_HOVER_BG};"
            f"  border: 1px solid {_Palette.BORDER_HOVER};"
            f"  border-bottom: none;"
            f"}}"
            f"QTabBar::tab:selected {{"
            f"  background: qlineargradient(x1:0, y1:0, x2:0, y2:1,"
            f"    stop:0 {_Palette.BTN_GRAD_L_CHK}, stop:1 {_Palette.TAB_SEL_GRAD_R});"
            f"  color: #ffffff;"
            f"  border: 2px solid {_Palette.BORDER_GLOW};"
            f"  border-bottom: none;"
            f"}}"
        )


# ── 액션 실행 핸들러 (status=ready 액션의 실제 wire) ───────────────────


def _common_input_dialog_style() -> str:
    """입력 다이얼로그용 검정+보라 스타일 (액션 실행 입력 폼)."""
    return (
        f"QDialog {{ background: {_Palette.BG_DEEP}; }}"
        f"QLabel {{ background: transparent; color: {_Palette.TEXT_PRIMARY}; }}"
        + _form_input_style()
    )


class _YoutubeScriptInputDialog(QDialog):
    """🎬 유튜브 스크립트 작성 — 주제·길이·톤 입력."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("🎬 유튜브 스크립트 작성")
        self.resize(520, 320)
        self.setStyleSheet(_common_input_dialog_style())

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(12)

        title = QLabel("🎬  유튜브 영상 스크립트 작성")
        title.setFont(QFont("", 14, QFont.Bold))
        title.setStyleSheet(f"color: {_Palette.ACCENT_GLOW}; background: transparent;")
        _apply_purple_glow(title, radius=16, color_hex=_Palette.ACCENT, alpha=110)
        root.addWidget(title)

        root.addWidget(self._lbl("📌 주제 (예: '집에서 30분 만에 만드는 김치찌개')"))
        self._fld_topic = QLineEdit()
        self._fld_topic.setPlaceholderText("주제 한 줄")
        root.addWidget(self._fld_topic)

        row = QHBoxLayout()
        row.setSpacing(10)
        col_len = QVBoxLayout()
        col_len.addWidget(self._lbl("📏 길이"))
        self._fld_length = QComboBox()
        self._fld_length.addItems(["short (~800자)", "medium (~1500자)", "long (~3000자)"])
        self._fld_length.setCurrentIndex(1)
        col_len.addWidget(self._fld_length)
        row.addLayout(col_len, 1)

        col_tone = QVBoxLayout()
        col_tone.addWidget(self._lbl("🎭 톤"))
        self._fld_tone = QComboBox()
        self._fld_tone.addItems(["친근", "전문가", "유머러스", "차분"])
        col_tone.addWidget(self._fld_tone)
        row.addLayout(col_tone, 1)
        root.addLayout(row)

        root.addStretch(1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_cancel = QPushButton("취소")
        btn_cancel.setStyleSheet(_small_btn_style(primary=False))
        btn_cancel.setCursor(Qt.PointingHandCursor)
        btn_cancel.clicked.connect(self.reject)
        btn_ok = QPushButton("▶  스크립트 작성 시작")
        btn_ok.setStyleSheet(_small_btn_style(primary=True))
        btn_ok.setCursor(Qt.PointingHandCursor)
        btn_ok.clicked.connect(self.accept)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_ok)
        root.addLayout(btn_row)

    def _lbl(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {_Palette.TEXT_MUTED}; font-size: 11px; font-weight: 600;"
            f" background: transparent;"
        )
        return lbl

    def get_values(self) -> dict:
        length_text = self._fld_length.currentText()
        length = length_text.split()[0]   # "short"/"medium"/"long"
        return {
            "topic": self._fld_topic.text().strip(),
            "length": length,
            "tone": self._fld_tone.currentText(),
        }


class _ResearchTopicInputDialog(QDialog):
    """🔬 탐구주제 연구 — 주제 한 줄 입력."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("🔬 탐구주제 연구 (심층분석)")
        self.resize(540, 220)
        self.setStyleSheet(_common_input_dialog_style())

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(12)

        title = QLabel("🔬  탐구주제 연구")
        title.setFont(QFont("", 14, QFont.Bold))
        title.setStyleSheet(f"color: {_Palette.ACCENT_GLOW}; background: transparent;")
        _apply_purple_glow(title, radius=16, color_hex=_Palette.ACCENT, alpha=110)
        root.addWidget(title)

        hint = QLabel(
            "한 줄로 주제를 입력하세요. EIDOS 가 자료 수집 + DC Reasoner 다각도 분석 + "
            "보고서 .md / PDF 를 자동 생성합니다 (수 분 소요)."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {_Palette.TEXT_MUTED}; font-size: 11px; background: transparent;")
        root.addWidget(hint)

        self._fld_topic = QLineEdit()
        self._fld_topic.setPlaceholderText("예: 2026년 한국 1인 가구 식료품 소비 변화")
        root.addWidget(self._fld_topic)

        root.addStretch(1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_cancel = QPushButton("취소")
        btn_cancel.setStyleSheet(_small_btn_style(primary=False))
        btn_cancel.setCursor(Qt.PointingHandCursor)
        btn_cancel.clicked.connect(self.reject)
        btn_ok = QPushButton("▶  심층분석 시작")
        btn_ok.setStyleSheet(_small_btn_style(primary=True))
        btn_ok.setCursor(Qt.PointingHandCursor)
        btn_ok.clicked.connect(self.accept)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_ok)
        root.addLayout(btn_row)

    def get_topic(self) -> str:
        return self._fld_topic.text().strip()


def _find_chat_window(parent_dialog: Optional[QWidget]):
    """parent_dialog 의 parent chain 에서 ChatWindow 인스턴스 찾기.
    더보기 다이얼로그가 ChatWindow(self) 로 생성되므로 parent() 가 ChatWindow."""
    if parent_dialog is None:
        return None
    p: Optional[QWidget] = parent_dialog.parent() if hasattr(parent_dialog, "parent") else None
    # parent_dialog 자체일 수도 있고 그 parent 일 수도 — 둘 다 시도
    for cand in (parent_dialog, p):
        if cand is None:
            continue
        if hasattr(cand, "eidos_worker") and hasattr(cand, "_safe_append"):
            return cand
    return None


def _run_youtube_script(parent_dialog: QWidget, data: dict) -> None:
    """youtube_script 액션 — 입력 다이얼로그 → worker.submit_task → LLM → .md 저장."""
    inp = _YoutubeScriptInputDialog(parent_dialog)
    if inp.exec() != QDialog.Accepted:
        return
    vals = inp.get_values()
    topic = vals.get("topic", "").strip()
    length = vals.get("length", "medium")
    tone = vals.get("tone", "친근")
    if not topic:
        QMessageBox.warning(parent_dialog, "입력 필요", "주제를 한 줄 입력하세요.")
        return

    cw = _find_chat_window(parent_dialog)
    if cw is None:
        QMessageBox.warning(parent_dialog, "오류", "ChatWindow 를 찾을 수 없습니다.")
        return
    worker = getattr(cw, "eidos_worker", None)
    if worker is None:
        QMessageBox.warning(parent_dialog, "오류", "eidos_worker 미초기화.")
        return

    target_chars = {"short": 800, "medium": 1500, "long": 3000}.get(length, 1500)

    async def _bg():
        try:
            import datetime as _dt
            import os as _os
            from llm_module import get_llm_response_async as _llm_call
            sys_prompt = (
                "너는 한국 유튜브 영상 스크립트 작가다. 후크/본론/정리/CTA 4단 구조로 작성한다.\n"
                f"톤: {tone}\n"
                f"목표 길이: 약 {target_chars}자\n"
                "마크다운 본문만 출력하고 메타 설명·인사말은 쓰지 마라."
            )
            user_prompt = (
                f"[주제] {topic}\n\n"
                "다음 4단 구조로 마크다운 작성:\n"
                "## 0. 인트로 (후크 — 30초 내 시청자 잡기·문제 제기 또는 통계)\n"
                "## 1. 본론 (핵심 내용·소제목 2~4개)\n"
                "## 2. 정리 (3 줄 요약)\n"
                "## 3. CTA (좋아요·구독·다음 영상 안내)"
            )
            raw = (await _llm_call(
                user_prompt, system_prompt=sys_prompt, max_tokens=8192, timeout=120,
            )) or ""
            raw = raw.strip()
            if (not raw) or raw.startswith("LLM 오류") or raw.startswith("[서버"):
                try:
                    cw._safe_append(f"⚠️ [유튜브 스크립트] 실패 — {raw[:120]}", "system")
                except Exception:
                    pass
                return
            out_dir = _os.path.join("eidos_files", "youtube_scripts")
            _os.makedirs(out_dir, exist_ok=True)
            ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = "".join(c if c.isalnum() or c in " _-가-힣" else "_" for c in topic[:40]).strip()
            out_path = _os.path.join(out_dir, f"{ts}_{safe_title or 'untitled'}.md")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"# {topic}\n\n_길이: {length} · 톤: {tone}_\n\n---\n\n")
                f.write(raw)
            try:
                cw._safe_append(
                    f"🎬 [유튜브 스크립트] 작성 완료\n"
                    f"  • 주제: {topic}\n"
                    f"  • 파일: {out_path}\n"
                    f"  • 길이: 약 {len(raw):,}자",
                    "system",
                )
            except Exception:
                pass
        except Exception as e:
            print(f"[youtube_script] async 실패: {e}")
            try:
                cw._safe_append(f"⚠️ [유튜브 스크립트] 예외 — {type(e).__name__}: {e}", "system")
            except Exception:
                pass

    try:
        worker.submit_task(_bg())
    except Exception as e:
        QMessageBox.warning(parent_dialog, "실행 실패", f"worker.submit_task 실패: {e}")
        return

    QMessageBox.information(
        parent_dialog, "🎬 유튜브 스크립트 — 시작",
        f"'{topic}' 스크립트 작성을 시작했습니다.\n\n"
        f"• 길이: {length} (~{target_chars}자)\n"
        f"• 톤: {tone}\n"
        "• 완료되면 메인 채팅창에 안내 + eidos_files/youtube_scripts/ 폴더에 저장됩니다.",
    )


def _run_research_topic(parent_dialog: QWidget, data: dict) -> None:
    """research_topic_deep 액션 — 입력 → ChatWindow._voice_trigger_deep_analysis 위임."""
    inp = _ResearchTopicInputDialog(parent_dialog)
    if inp.exec() != QDialog.Accepted:
        return
    topic = inp.get_topic()
    if not topic:
        QMessageBox.warning(parent_dialog, "입력 필요", "주제를 한 줄 입력하세요.")
        return

    cw = _find_chat_window(parent_dialog)
    if cw is None or not hasattr(cw, "_voice_trigger_deep_analysis"):
        QMessageBox.warning(
            parent_dialog, "오류",
            "ChatWindow._voice_trigger_deep_analysis 부재 — 심층분석 인프라 미초기화.",
        )
        return
    try:
        cw._voice_trigger_deep_analysis(topic)
    except Exception as e:
        QMessageBox.warning(parent_dialog, "실행 실패", f"{type(e).__name__}: {e}")
        return

    QMessageBox.information(
        parent_dialog, "🔬 심층분석 — 시작",
        f"'{topic}' 심층분석을 시작했습니다.\n\n"
        "• 수 분 걸릴 수 있습니다.\n"
        "• 메인 채팅창에서 진행 상황 확인.\n"
        "• 완료 시 .md / PDF 보고서 자동 표시.",
    )


def _run_user_canvas(parent_dialog: QWidget, data: dict) -> None:
    """사용자 캔버스 (canvas:NAME) 실행 — 액션 탭에서 ▶ 실행 클릭 시.
    백그라운드 호출·진행 UI 없이 채팅창에 시작/완료 안내만.
    자세한 노드 색상/중단/ASK_USER 는 🎨 액션 조립 캔버스 탭에서."""
    _id = data.get("id", "") or ""
    if not _id.startswith("canvas:"):
        QMessageBox.warning(parent_dialog, "잘못된 id", f"canvas:NAME 형식 아님: {_id}")
        return
    canvas_name = _id[len("canvas:"):]
    try:
        import eidos_canvas_store as _cstore
    except Exception as e:
        QMessageBox.warning(parent_dialog, "오류", f"canvas_store 로드 실패: {e}")
        return
    canvas = _cstore.load_canvas(canvas_name)
    if not canvas.get("nodes"):
        QMessageBox.warning(parent_dialog, "빈 캔버스", f"'{canvas_name}' 은 노드가 없습니다.")
        return
    cw = _find_chat_window(parent_dialog)
    worker = getattr(cw, "eidos_worker", None) if cw else None
    if worker is None:
        QMessageBox.warning(parent_dialog, "오류", "ChatWindow.eidos_worker 부재.")
        return
    try:
        from eidos_canvas_runner import execute_canvas_async
    except Exception as e:
        QMessageBox.warning(parent_dialog, "오류", f"canvas_runner 로드 실패: {e}")
        return

    def _on_progress(event: str, node_id: str, payload: dict):
        try:
            if event == "all_done":
                total = payload.get("total", 0)
                ok = payload.get("ok", 0)
                fail = payload.get("fail", 0)
                skip = payload.get("skip", 0)
                stopped = payload.get("stopped", 0)
                stopped_part = f" / 🛑 {stopped}" if stopped else ""
                msg = (f"🎨 [사용자 캔버스 실행 완료] '{canvas_name}' "
                       f"— ✅ {ok} / ❌ {fail} / ⏭ {skip}{stopped_part} (총 {total})")
                try:
                    if cw and hasattr(cw, "_safe_append"):
                        cw._safe_append(msg, "system")
                except Exception:
                    pass
        except Exception as e:
            print(f"[user_canvas progress] {e}")
        return None

    try:
        worker.submit_task(execute_canvas_async(canvas, on_progress=_on_progress,
                                                stop_on_fail=False))
        try:
            if cw and hasattr(cw, "_safe_append"):
                cw._safe_append(
                    f"🎨 [사용자 캔버스 실행 시작] '{canvas_name}' "
                    f"({len(canvas.get('nodes', []))}개 노드·백그라운드)",
                    "system",
                )
        except Exception:
            pass
        QMessageBox.information(
            parent_dialog, "▶ 실행 시작",
            f"사용자 캔버스 '{canvas_name}' 백그라운드 실행 시작.\n"
            "완료 시 메인 채팅창에 안내됩니다.\n\n"
            "💡 자세한 진행 (노드 색상·중단·ASK_USER) 은 🎨 액션 조립 캔버스 탭에서.",
        )
    except Exception as e:
        QMessageBox.warning(parent_dialog, "실행 실패", f"{type(e).__name__}: {e}")


# ── [2026-05-26 Phase 1] 사업 아이템 떠올리기 handler ────────────────
def _run_business_idea(parent_dialog, action_data: dict) -> None:
    """💡 사업 아이템 떠올리기 액션 — ToM-core belief 에서 프로필 추출 후
    Hybrid 입력 form + LLM 5개 idea 생성 → 채팅창에 markdown 표시.

    Phase 1 MVP. Phase 2/3 (영속화·proactive) 는 후속.
    """
    try:
        from eidos_belief_core import load_belief
        from eidos_idea_agent import (
            extract_profile_from_belief, generate_ideas_async,
            ideas_summary_markdown, UserProfile,
        )
    except Exception as e:
        QMessageBox.warning(parent_dialog, "💡 사업 아이템", f"모듈 import 실패: {e}")
        return

    # 1) belief 에서 default profile 추출
    try:
        belief = load_belief()
        profile_default = extract_profile_from_belief(belief)
    except Exception as e:
        print(f"[business_idea] belief 추출 실패 (graceful·default): {e}")
        profile_default = UserProfile()

    # 2) Hybrid 입력 form — 사용자가 수정 후 확정
    dlg = QDialog(parent_dialog)
    dlg.setWindowTitle("💡 사업 아이템 — 프로필 확인")
    dlg.resize(640, 540)
    lay = QVBoxLayout(dlg)

    info = QLabel(
        "ToM-core belief 에서 자동 추출한 프로필입니다.\n"
        "필요하면 수정하고 ▶ 실행을 누르세요. (값 비워두면 default 사용)"
    )
    info.setStyleSheet("color: #888; padding: 4px;")
    lay.addWidget(info)

    def _mklabel(t):
        l = QLabel(t)
        l.setStyleSheet("color: #C084FC; font-weight: bold; margin-top: 6px;")
        return l

    lay.addWidget(_mklabel("강점·기술 (콤마 구분)"))
    fld_strengths = QLineEdit(", ".join(profile_default.strengths))
    fld_strengths.setPlaceholderText("예: Python 개발, 자동화, GUI 디자인")
    lay.addWidget(fld_strengths)

    lay.addWidget(_mklabel("관심 분야 (콤마 구분)"))
    fld_interests = QLineEdit(", ".join(profile_default.interests))
    fld_interests.setPlaceholderText("예: AI, 교육, 스몰비즈, 자동화")
    lay.addWidget(fld_interests)

    lay.addWidget(_mklabel("자본 범위"))
    fld_capital = QLineEdit(profile_default.capital)
    fld_capital.setPlaceholderText("예: 100만원~500만원")
    lay.addWidget(fld_capital)

    lay.addWidget(_mklabel("주 가용 시간"))
    fld_hours = QLineEdit(str(profile_default.weekly_hours))
    fld_hours.setPlaceholderText("예: 20")
    lay.addWidget(fld_hours)

    lay.addWidget(_mklabel("제약 (콤마 구분·선택)"))
    fld_constraints = QLineEdit(", ".join(profile_default.constraints))
    fld_constraints.setPlaceholderText("예: 혼자 운영, 오프라인 X, 해외 X")
    lay.addWidget(fld_constraints)

    lay.addWidget(_mklabel("최근 자주 다루는 화제 (선택)"))
    fld_focus = QLineEdit(profile_default.recent_focus)
    fld_focus.setPlaceholderText("예: 크몽 자동화·ToM agent·EIDOS")
    lay.addWidget(fld_focus)

    btn_row = QHBoxLayout()
    btn_row.addStretch(1)
    btn_cancel = QPushButton("취소")
    btn_run = QPushButton("▶  5개 idea 생성")
    btn_cancel.clicked.connect(dlg.reject)
    btn_run.clicked.connect(dlg.accept)
    btn_row.addWidget(btn_cancel)
    btn_row.addWidget(btn_run)
    lay.addLayout(btn_row)

    if dlg.exec() != QDialog.Accepted:
        return

    # 3) 사용자 수정 반영 → UserProfile 구성
    def _split(s):
        return [t.strip() for t in (s or "").split(",") if t.strip()]
    try:
        weekly_hours = int(fld_hours.text().strip() or "20")
    except Exception:
        weekly_hours = 20
    profile = UserProfile(
        strengths=_split(fld_strengths.text())[:8],
        interests=_split(fld_interests.text())[:8],
        capital=fld_capital.text().strip() or "100만원~500만원",
        weekly_hours=weekly_hours,
        constraints=_split(fld_constraints.text())[:4],
        recent_focus=fld_focus.text().strip()[:120],
    )

    # 4) worker 찾고 LLM 호출 (비동기)
    worker = None
    try:
        w = parent_dialog
        while w is not None:
            if hasattr(w, "eidos_worker") and getattr(w, "eidos_worker", None):
                worker = w.eidos_worker
                break
            w = w.parent() if hasattr(w, "parent") else None
    except Exception:
        pass
    if worker is None or not hasattr(worker, "submit_task"):
        QMessageBox.warning(parent_dialog, "💡 사업 아이템",
                            "EidosWorker 를 찾을 수 없어요. ChatWindow 통해 더보기를 다시 열어보세요.")
        return

    # ChatWindow 찾기 — 결과 채팅창 표시용
    chat_win = None
    try:
        w = parent_dialog
        while w is not None:
            if hasattr(w, "append_message") and hasattr(w, "_safe_append_requested"):
                chat_win = w
                break
            w = w.parent() if hasattr(w, "parent") else None
    except Exception:
        pass

    QMessageBox.information(
        parent_dialog, "💡 사업 아이템 생성 시작",
        "백그라운드에서 LLM 호출 중입니다 (10~30초)...\n"
        "결과는 메인 채팅창에 markdown 카드로 표시됩니다.",
    )

    async def _bg_run():
        try:
            ideas = await generate_ideas_async(profile, n=5)
            md = ideas_summary_markdown(ideas, profile)
            if chat_win is not None and hasattr(chat_win, "_safe_append_requested"):
                # thread-safe append
                chat_win._safe_append_requested.emit(md, "eidos")
            else:
                print(f"[business_idea] 결과:\n{md[:500]}")
        except Exception as e:
            err = f"⚠️ 사업 아이템 생성 실패: {type(e).__name__}: {e}"
            print(f"[business_idea] {err}")
            if chat_win is not None and hasattr(chat_win, "_safe_append_requested"):
                chat_win._safe_append_requested.emit(err, "error")

    worker.submit_task(_bg_run())


# ── ACTION_HANDLERS 등록 (앞에 forward declare 된 빈 dict 에 update) ────
ACTION_HANDLERS.update({
    "youtube_script": _run_youtube_script,
    "research_topic_deep": _run_research_topic,
    "business_idea": _run_business_idea,
})

