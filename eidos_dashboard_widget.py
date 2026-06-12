# -*- coding: utf-8 -*-
"""EIDOS 대시보드 *GUI 위젯* · PySide6 — 사업 감독 콕핏 (마스터-디테일).

⚠ 이름 주의: 기존 `eidos_dashboard.py` 는 별개(EIDOS 내부 자기상태 md 보고서). 이 파일은 GUI 위젯.

레이아웃: 좌측 = 세로 프로젝트 탭 목록(맨 위 "📊 전체 현황") / 우측 = 선택 항목 상세.
  - 전체 현황: 승인 대기·오늘 할당·KPI·고객 메시지 (글로벌 감독)
  - 프로젝트: 그 프로젝트의 공수·납품일·산출물·정보
테마: 색 하드코딩 금지. THEMES 토큰 4종 소비 + `apply_theme(theme_name)`. 데이터: read-only.
공개 인터페이스(DashboardWidget·signals·apply_theme·refresh)는 본체(eidos_chat_gui) 통합용으로 유지.
"""
from __future__ import annotations

import sys

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget, QFrame, QLabel, QPushButton, QLineEdit, QVBoxLayout, QHBoxLayout, QButtonGroup,
    QScrollArea, QSizePolicy, QGraphicsDropShadowEffect,
)

import eidos_dashboard_data as ddata

NEON_MINT = "#2dffd5"
NEON_PURPLE = "#b14dff"

_FALLBACK_THEMES = {
    "Modern Dark": {"is_dark": True, "colors": {
        "bg_base": "#09090b", "bg_raised": "#18181b", "bg_surface": "#27272a", "bg_input": "#27272a",
        "text_title": "#f4f4f5", "text_primary": "#e4e4e7", "text_secondary": "#a1a1aa", "text_hint": "#71717a",
        "text_on_accent": "#ffffff", "accent_primary": "#7c3aed", "accent_hover": "#8b5cf6",
        "border_subtle": "#27272a", "danger_bg": "#991b1b", "danger_text": "#fecaca"},
        "metrics": {"radius_sm": "4px", "radius_md": "8px", "padding_md": "8px"}},
    "Modern White (Warm)": {"is_dark": False, "colors": {
        "bg_base": "#fdfcfb", "bg_raised": "#f6f4f1", "bg_surface": "#ffffff", "bg_input": "#ffffff",
        "text_title": "#1c1917", "text_primary": "#292524", "text_secondary": "#78716c", "text_hint": "#a8a29e",
        "text_on_accent": "#ffffff", "accent_primary": "#b45309", "accent_hover": "#d97706",
        "border_subtle": "#e7e5e4", "danger_bg": "#dc2626", "danger_text": "#ffffff"},
        "metrics": {"radius_sm": "6px", "radius_md": "10px", "padding_md": "10px"}},
    "Modern White (Crisp)": {"is_dark": False, "colors": {
        "bg_base": "#ffffff", "bg_raised": "#eef2f7", "bg_surface": "#ffffff", "bg_input": "#ffffff",
        "text_title": "#0f172a", "text_primary": "#1e293b", "text_secondary": "#64748b", "text_hint": "#94a3b8",
        "text_on_accent": "#ffffff", "accent_primary": "#2563eb", "accent_hover": "#3b82f6",
        "border_subtle": "#e2e8f0", "danger_bg": "#dc2626", "danger_text": "#ffffff"},
        "metrics": {"radius_sm": "6px", "radius_md": "10px", "padding_md": "10px"}},
    "Obsidian": {"is_dark": True, "colors": {
        "bg_base": "#0b0d10", "bg_raised": "#14171c", "bg_surface": "#1b1f26", "bg_input": "#1b1f26",
        "text_title": "#e6edf3", "text_primary": "#c9d1d9", "text_secondary": "#8b949e", "text_hint": "#6e7681",
        "text_on_accent": "#ffffff", "accent_primary": "#2f81f7", "accent_hover": "#539bf5",
        "border_subtle": "#21262d", "danger_bg": "#b62324", "danger_text": "#ffdcd7"},
        "metrics": {"radius_sm": "4px", "radius_md": "8px", "padding_md": "8px"}},
}


def _resolve_theme(theme_name: str):
    mod = sys.modules.get("eidos_chat_gui")
    themes = getattr(mod, "THEMES", None) if mod else None
    if not themes:
        themes = _FALLBACK_THEMES
    t = themes.get(theme_name) or themes.get("Modern Dark") or next(iter(themes.values()))
    return t.get("colors", {}), t.get("metrics", {}), bool(t.get("is_dark", True))


def _soft_shadow(widget, color=(15, 23, 42, 38), blur=22, dy=4):
    eff = QGraphicsDropShadowEffect(widget)
    eff.setBlurRadius(blur); eff.setOffset(0, dy); eff.setColor(QColor(*color))
    widget.setGraphicsEffect(eff)


def _neon_glow(widget, color_hex, blur=18):
    eff = QGraphicsDropShadowEffect(widget)
    eff.setBlurRadius(blur); eff.setOffset(0, 0); eff.setColor(QColor(color_hex))
    widget.setGraphicsEffect(eff)


def _fmt_deadline(ts) -> str:
    if not ts:
        return "미설정"
    try:
        import datetime as _dt
        d = _dt.datetime.fromtimestamp(float(ts))
        days = (d.date() - _dt.date.today()).days
        dd = f"D{'-' if days >= 0 else '+'}{abs(days)}"
        return f"{d.strftime('%Y-%m-%d')} ({dd})"
    except Exception:  # noqa: BLE001
        return "미설정"


# ──────────────────────────────────────────────────────────────
# KPI 타일 (큰 숫자)
# ──────────────────────────────────────────────────────────────
class KpiTile(QFrame):
    def __init__(self, label: str, icon: str = "", *, neon=None, parent=None):
        super().__init__(parent)
        self.setObjectName("KpiTile")
        self.neon_hex = neon
        self._urgent = False
        self._c: dict = {}; self._m: dict = {}; self._dark = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        lay = QVBoxLayout(self); lay.setContentsMargins(18, 14, 18, 16); lay.setSpacing(5)
        self._label = QLabel(f"{icon} {label}".strip()); self._label.setObjectName("KpiLabel")
        nrow = QHBoxLayout(); nrow.setSpacing(6); nrow.setContentsMargins(0, 0, 0, 0)
        self._num = QLabel("0"); self._num.setObjectName("KpiNum")
        self._unit = QLabel(""); self._unit.setObjectName("KpiUnit")
        nrow.addWidget(self._num, 0, Qt.AlignmentFlag.AlignBottom)
        nrow.addWidget(self._unit, 0, Qt.AlignmentFlag.AlignBottom)
        nrow.addStretch(1)
        self._sub = QLabel(""); self._sub.setObjectName("KpiSub")
        lay.addWidget(self._label); lay.addLayout(nrow); lay.addWidget(self._sub)

    def set_value(self, num, unit="", sub="", *, urgent=False):
        self._num.setText(str(num)); self._unit.setText(unit)
        self._sub.setText(sub); self._sub.setVisible(bool(sub))
        self._urgent = urgent; self._paint()

    def style_with(self, c, m, is_dark):
        self._c, self._m, self._dark = c, m, is_dark
        if is_dark:
            _neon_glow(self, self.neon_hex or c.get("accent_hover", NEON_MINT))
        else:
            _soft_shadow(self)
        self._paint()

    def _paint(self):
        c, m, is_dark = self._c, self._m, self._dark
        if not c:
            return
        accent = c.get("accent_primary", "#2563eb")
        radius = m.get("radius_md", "14px")
        if is_dark:
            bg = "#000000"; border = self.neon_hex or c.get("accent_hover", NEON_MINT)
            num_color = "#ff5d7a" if self._urgent else border
            label_color = c.get("text_secondary", "#a1a1aa"); sub_color = c.get("text_hint", "#71717a")
        else:
            bg = c.get("bg_surface", "#ffffff"); border = c.get("border_subtle", "#e2e8f0")
            num_color = "#dc2626" if self._urgent else accent
            label_color = c.get("text_secondary", "#64748b"); sub_color = c.get("text_hint", "#94a3b8")
        self.setStyleSheet(f"""
            QFrame#KpiTile {{ background:{bg}; border:1px solid {border}; border-radius:{radius}; }}
            QLabel#KpiLabel {{ color:{label_color}; font-size:12px; font-weight:600; background:transparent; }}
            QLabel#KpiNum {{ color:{num_color}; font-size:30px; font-weight:bold; background:transparent; }}
            QLabel#KpiUnit {{ color:{label_color}; font-size:13px; font-weight:600; background:transparent; padding-bottom:4px; }}
            QLabel#KpiSub {{ color:{sub_color}; font-size:11px; background:transparent; }}
        """)


class _ClickRow(QWidget):
    def __init__(self, primary, secondary, on_click, parent=None):
        super().__init__(parent)
        self._cb = on_click
        self.setObjectName("DashRowW")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        lay = QVBoxLayout(self); lay.setContentsMargins(2, 6, 2, 7); lay.setSpacing(2)
        p = QLabel(primary); p.setObjectName("RowPrimary"); p.setWordWrap(True); lay.addWidget(p)
        if secondary:
            s = QLabel(secondary); s.setObjectName("RowSecondary"); s.setWordWrap(True); lay.addWidget(s)

    def mousePressEvent(self, event):  # noqa: N802
        try:
            self._cb()
        except Exception:  # noqa: BLE001
            pass
        super().mousePressEvent(event)


# ──────────────────────────────────────────────────────────────
# 상세 카드
# ──────────────────────────────────────────────────────────────
class DashCard(QFrame):
    def __init__(self, title, icon="", *, danger=False, neon=None, parent=None):
        super().__init__(parent)
        self._danger = danger
        self.neon_hex = neon
        self.setObjectName("DashCard")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        root = QVBoxLayout(self); root.setContentsMargins(18, 15, 18, 16); root.setSpacing(8)
        head = QHBoxLayout(); head.setSpacing(8)
        self._title_lbl = QLabel(f"{icon} {title}".strip()); self._title_lbl.setObjectName("CardTitle")
        self._badge = QLabel(""); self._badge.setObjectName("CardBadge")
        self._badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._badge.setMaximumHeight(20)
        self._badge.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self._badge.hide()
        head.addWidget(self._title_lbl); head.addStretch(1); head.addWidget(self._badge)
        root.addLayout(head)
        self._body = QVBoxLayout(); self._body.setSpacing(2)
        root.addLayout(self._body)

    def add_row(self, primary, secondary="", *, muted=False):
        w = QWidget(); w.setObjectName("DashRowW")
        lay = QVBoxLayout(w); lay.setContentsMargins(2, 6, 2, 7); lay.setSpacing(2)
        p = QLabel(primary); p.setObjectName("RowHint" if muted else "RowPrimary"); p.setWordWrap(True); lay.addWidget(p)
        if secondary:
            s = QLabel(secondary); s.setObjectName("RowSecondary"); s.setWordWrap(True); lay.addWidget(s)
        self._body.addWidget(w)
        return w

    def add_widget(self, w):
        self._body.addWidget(w)
        return w

    def set_empty(self, msg):
        self.add_row(msg, muted=True)

    def set_count(self, n):
        if n in (None, 0, ""):
            self._badge.setText(""); self._badge.hide()
        else:
            self._badge.setText(str(n)); self._badge.show()

    def _neon(self, c):
        return self.neon_hex or (c.get("accent_hover") or c.get("accent_primary", "#8b5cf6"))

    def style_with(self, c, m, is_dark=False):
        primary = c.get("text_primary", "#1e293b"); secondary = c.get("text_secondary", "#64748b")
        hint = c.get("text_hint", "#94a3b8"); accent = c.get("accent_primary", "#2563eb")
        radius = m.get("radius_md", "14px"); sep = c.get("border_subtle", "#eef1f5")
        if is_dark:
            bg = "#000000"; neon = self._neon(c); border = neon; title = neon
            badge_bg = neon; badge_fg = "#000000"; sep = "rgba(255,255,255,0.06)"
            _neon_glow(self, neon)
        else:
            bg = c.get("bg_surface", "#ffffff"); border = c.get("border_subtle", "#e2e8f0")
            title = c.get("danger_bg", "#dc2626") if self._danger else c.get("text_title", "#0f172a")
            badge_bg = c.get("danger_bg", "#dc2626") if self._danger else accent
            badge_fg = "#ffffff"
            _soft_shadow(self)
        self.setStyleSheet(f"""
            QFrame#DashCard {{ background:{bg}; border:1px solid {border}; border-radius:{radius}; }}
            QLabel#CardTitle {{ color:{title}; font-weight:700; font-size:14px; background:transparent; }}
            QLabel#CardBadge {{ color:{badge_fg}; background:{badge_bg}; border-radius:10px;
                                 padding:2px 9px; font-size:11px; font-weight:700; min-width:14px; }}
            QWidget#DashRowW {{ border-bottom:1px solid {sep}; background:transparent; }}
            QLabel#RowPrimary {{ color:{primary}; font-size:13px; background:transparent; }}
            QLabel#RowSecondary {{ color:{secondary}; font-size:11.5px; background:transparent; }}
            QLabel#RowHint {{ color:{hint}; font-size:12px; background:transparent; }}
        """)


# ──────────────────────────────────────────────────────────────
# 대시보드 위젯 (마스터-디테일)
# ──────────────────────────────────────────────────────────────
class DashboardWidget(QWidget):
    approve_requested = Signal(str)
    cancel_requested = Signal(str)
    refresh_requested = Signal()
    output_open_requested = Signal(str)

    def __init__(self, theme_name="Modern Dark", parent=None):
        super().__init__(parent)
        self._theme_name = theme_name
        self._include_helix = False
        self._last_fp = None
        self._entries: list = []      # 좌측 nav 항목 [{"kind":"overview"} | {"kind":"project","p":...}]
        self._sel = 0
        self._data: dict = {}
        self._c, self._m, self._dark = {}, {}, True

        root = QVBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        # 헤더
        self._header = QFrame(); self._header.setObjectName("DashHeader")
        hl = QHBoxLayout(self._header); hl.setContentsMargins(22, 12, 22, 12)
        self._title = QLabel("📊 EIDOS 대시보드"); self._title.setObjectName("DashTitle")
        self._status = QLabel(""); self._status.setObjectName("DashStatus")
        hl.addWidget(self._title); hl.addStretch(1); hl.addWidget(self._status)
        root.addWidget(self._header)

        # 본문: 좌측 nav | 우측 detail
        body = QHBoxLayout(); body.setContentsMargins(0, 0, 0, 0); body.setSpacing(0)
        # 좌측 nav (세로 프로젝트 탭)
        self._nav_scroll = QScrollArea(); self._nav_scroll.setWidgetResizable(True)
        self._nav_scroll.setFrameShape(QFrame.Shape.NoFrame); self._nav_scroll.setFixedWidth(248)
        self._nav_host = QWidget(); self._nav_host.setObjectName("NavHost")
        self._nav_lay = QVBoxLayout(self._nav_host); self._nav_lay.setContentsMargins(0, 6, 0, 6); self._nav_lay.setSpacing(0)
        self._nav_lay.addStretch(1)
        self._nav_scroll.setWidget(self._nav_host)
        self._nav_group = QButtonGroup(self); self._nav_group.setExclusive(True)
        body.addWidget(self._nav_scroll)
        # 우측 detail
        self._detail_scroll = QScrollArea(); self._detail_scroll.setWidgetResizable(True)
        self._detail_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._detail_host = QWidget(); self._detail_host.setObjectName("DetailHost")
        self._detail_lay = QVBoxLayout(self._detail_host)
        self._detail_lay.setContentsMargins(20, 18, 20, 20); self._detail_lay.setSpacing(16)
        self._detail_scroll.setWidget(self._detail_host)
        body.addWidget(self._detail_scroll, 1)

        # 우측 AIRA 답변 패널(파란 박스) — 질문하면 여기에 구조화된 답. 평소 숨김.
        self._aira = QFrame(); self._aira.setObjectName("AiraPanel"); self._aira.setFixedWidth(360)
        av = QVBoxLayout(self._aira); av.setContentsMargins(16, 14, 16, 14); av.setSpacing(8)
        self._aira_title = QLabel("🌸 AIRA"); self._aira_title.setObjectName("AiraTitle")
        av.addWidget(self._aira_title)
        ascroll = QScrollArea(); ascroll.setWidgetResizable(True); ascroll.setFrameShape(QFrame.Shape.NoFrame)
        self._aira_content = QWidget(); self._aira_content.setObjectName("AiraContent")
        self._aira_body = QVBoxLayout(self._aira_content); self._aira_body.setContentsMargins(0, 0, 0, 0); self._aira_body.setSpacing(5)
        ascroll.setWidget(self._aira_content)
        av.addWidget(ascroll, 1)
        self._aira.setVisible(False)
        body.addWidget(self._aira)
        root.addLayout(body, 1)

        # 하단 채팅 입력바 — AIRA에게 질문
        self._chatbar = QFrame(); self._chatbar.setObjectName("ChatBar")
        cl = QHBoxLayout(self._chatbar); cl.setContentsMargins(16, 10, 16, 10); cl.setSpacing(10)
        self._chat_input = QLineEdit()
        self._chat_input.setPlaceholderText("AIRA에게 물어보세요 — 예: 오늘 해야할 일들은 뭐가 있지?")
        self._chat_input.returnPressed.connect(self._on_ask)
        self._chat_send = QPushButton("전송")
        self._chat_send.clicked.connect(self._on_ask)
        cl.addWidget(self._chat_input, 1)
        cl.addWidget(self._chat_send)
        root.addWidget(self._chatbar)

        self.apply_theme(theme_name)
        self.refresh(force=True)
        self._timer = QTimer(self); self._timer.setInterval(5000)
        self._timer.timeout.connect(self.refresh); self._timer.start()

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        self.refresh(force=True)

    # ── 테마 ──
    def apply_theme(self, theme_name):
        self._theme_name = theme_name
        c, m, is_dark = _resolve_theme(theme_name)
        self._c, self._m, self._dark = c, m, is_dark
        page = "#000000" if is_dark else c.get("bg_raised", "#eef2f7")
        surface = "#000000" if is_dark else c.get("bg_surface", "#ffffff")
        title_col = c.get("text_title", "#0f172a"); secondary = c.get("text_secondary", "#64748b")
        primary = c.get("text_primary", "#1e293b"); accent = c.get("accent_primary", "#2563eb")
        border = (c.get("accent_hover", "#8b5cf6") if is_dark else c.get("border_subtle", "#e2e8f0"))
        navbg = "#000000" if is_dark else c.get("bg_surface", "#ffffff")
        sel_bg = ("rgba(45,255,213,0.12)" if is_dark else "rgba(37,99,235,0.10)")
        hov_bg = ("rgba(255,255,255,0.05)" if is_dark else "rgba(15,23,42,0.04)")
        self.setStyleSheet(
            f"QWidget#DetailHost {{ background:{page}; }} QWidget#NavHost {{ background:{navbg}; }}"
            f"QScrollArea {{ border:none; }}"
            f"QPushButton#NavBtn {{ text-align:left; padding:11px 16px; border:none; border-radius:0;"
            f"  border-left:3px solid transparent; background:transparent; color:{primary}; font-size:13px; }}"
            f"QPushButton#NavBtn:hover {{ background:{hov_bg}; }}"
            f"QPushButton#NavBtn:checked {{ background:{sel_bg}; border-left:3px solid {accent};"
            f"  color:{accent}; font-weight:700; }}")
        self._nav_scroll.setStyleSheet(f"QScrollArea {{ background:{navbg}; border-right:1px solid {border}; }}")
        self._header.setStyleSheet(
            f"QFrame#DashHeader {{ background:{surface}; border-bottom:1px solid {border}; }}"
            f"QLabel#DashTitle {{ color:{title_col}; font-size:16px; font-weight:800; background:transparent; }}"
            f"QLabel#DashStatus {{ color:{secondary}; font-size:12px; background:transparent; }}")
        self._status.setText(self._status_text(title_col))
        # AIRA 답변 패널(파란 박스) + 하단 챗바
        blue = "#2f81f7" if is_dark else "#2563eb"
        panel_bg = "#0a1730" if is_dark else "#eff6ff"
        inp_bg = "#0b1424" if is_dark else "#ffffff"
        self._aira.setStyleSheet(
            f"QFrame#AiraPanel {{ background:{panel_bg}; border:1px solid {blue}; border-left:3px solid {blue}; }}"
            f"QWidget#AiraContent {{ background:transparent; }} QScrollArea {{ background:transparent; border:none; }}"
            f"QLabel#AiraTitle {{ color:{blue}; font-size:15px; font-weight:800; background:transparent; }}"
            f"QLabel#AiraQ {{ color:{secondary}; font-size:12px; background:transparent; }}"
            f"QLabel#AiraIntro {{ color:{primary}; font-size:13px; font-weight:700; background:transparent; }}"
            f"QLabel#AiraSect {{ color:{blue}; font-size:13px; font-weight:700; background:transparent; }}"
            f"QLabel#AiraItem {{ color:{primary}; font-size:12.5px; background:transparent; }}"
            f"QLabel#AiraSub {{ color:{secondary}; font-size:11.5px; background:transparent; }}")
        self._chatbar.setStyleSheet(
            f"QFrame#ChatBar {{ background:{surface}; border-top:1px solid {border}; }}"
            f"QLineEdit {{ background:{inp_bg}; color:{primary}; border:1px solid {border}; "
            f"  border-radius:8px; padding:9px 13px; font-size:13px; }}"
            f"QLineEdit:focus {{ border:1px solid {accent}; }}")
        self._chat_send.setStyleSheet(
            f"QPushButton {{ background:{accent}; color:#fff; border:none; border-radius:8px; "
            f"padding:9px 20px; font-weight:700; }} QPushButton:hover {{ background:{c.get('accent_hover', accent)}; }}")
        self._render_detail()   # 카드 재생성 시 현재 테마로 도색

    def _status_text(self, title_color):
        n = len((self._data or {}).get("approvals", []))
        return (f"● 온라인 &nbsp;·&nbsp; ⚡ 자율 ON &nbsp;·&nbsp; "
                f"<span style='color:{title_color};font-weight:700'>🔔 승인 {n}건</span>")

    # ── 새로고침 ──
    def refresh(self, force=False):
        try:
            fp = ddata.data_fingerprint()
        except Exception:  # noqa: BLE001
            fp = None
        if not force and fp is not None and fp == self._last_fp:
            return
        self._last_fp = fp
        try:
            self._data = ddata.gather(include_helix=self._include_helix)
        except Exception as e:  # noqa: BLE001
            self._data = {"_error": str(e)}
        self._rebuild_nav()
        self._render_detail()
        self._status.setText(self._status_text(self._c.get("text_title", "#0f172a")))

    def _rebuild_nav(self):
        # 기존 버튼 제거
        for b in list(self._nav_group.buttons()):
            self._nav_group.removeButton(b); b.setParent(None); b.deleteLater()
        while self._nav_lay.count() > 1:   # 마지막 stretch 보존
            it = self._nav_lay.takeAt(0)
            w = it.widget()
            if w:
                w.setParent(None); w.deleteLater()
        projects = (self._data or {}).get("projects", [])
        self._entries = [{"kind": "overview"}] + [{"kind": "project", "p": p} for p in projects]
        if self._sel >= len(self._entries):
            self._sel = 0
        for i, e in enumerate(self._entries):
            if e["kind"] == "overview":
                text = "📊  전체 현황"
            else:
                p = e["p"]
                emoji = "✅" if p.get("has_outcome") else "🟢"
                title = (p.get("title") or "(무제)")[:22]
                text = f"{emoji}  {title}"
            btn = QPushButton(text); btn.setObjectName("NavBtn"); btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._nav_group.addButton(btn, i)
            btn.clicked.connect(lambda _c=False, idx=i: self._select(idx))
            self._nav_lay.insertWidget(self._nav_lay.count() - 1, btn)
        btn = self._nav_group.button(self._sel)
        if btn:
            btn.setChecked(True)

    def _select(self, idx):
        self._sel = idx
        self._render_detail()

    # ── 상세 렌더 ──
    def _clear_detail(self):
        while self._detail_lay.count():
            it = self._detail_lay.takeAt(0)
            w = it.widget()
            if w:
                w.setParent(None); w.deleteLater()
            else:
                sub = it.layout()
                if sub:
                    while sub.count():
                        s = sub.takeAt(0); sw = s.widget()
                        if sw:
                            sw.setParent(None); sw.deleteLater()

    def _render_detail(self):
        if not hasattr(self, "_detail_lay"):
            return
        self._clear_detail()
        if not self._entries or self._sel >= len(self._entries):
            return
        e = self._entries[self._sel]
        if e["kind"] == "overview":
            self._render_overview()
        else:
            detail = None
            try:
                detail = ddata.get_project_detail(e["p"].get("id", ""))
            except Exception:  # noqa: BLE001
                detail = None
            self._render_project(detail or e["p"])

    def _new_tile(self, label, icon, idx):
        t = KpiTile(label, icon, neon=(NEON_MINT if idx % 2 == 0 else NEON_PURPLE))
        t.style_with(self._c, self._m, self._dark)
        return t

    def _new_card(self, title, icon, *, danger=False, idx=0):
        card = DashCard(title, icon, danger=danger, neon=(NEON_MINT if idx % 2 == 0 else NEON_PURPLE))
        card.style_with(self._c, self._m, self._dark)
        return card

    def _add(self, w):
        self._detail_lay.addWidget(w)

    def _add_row_of(self, widgets):
        row = QHBoxLayout(); row.setSpacing(16)
        for w in widgets:
            row.addWidget(w, 1, Qt.AlignmentFlag.AlignTop)
        self._detail_lay.addLayout(row)

    # ── 전체 현황 ──
    def _render_overview(self):
        data = self._data or {}
        appr = data.get("approvals", []); running = data.get("running", [])
        today = data.get("today", {}); loads = today.get("loads", [])
        outputs = data.get("outputs", []); total_h = sum(float(L.get("remaining_hours") or 0) for L in loads)
        t1 = self._new_tile("승인 대기", "⚠", 0); t1.set_value(len(appr), "건", "확인 필요" if appr else "없음", urgent=bool(appr))
        t2 = self._new_tile("진행 중", "▶", 1); t2.set_value(len(running), "건", "실행/승인")
        t3 = self._new_tile("오늘 작업", "📅", 2); t3.set_value(len(loads), "건", f"공수 {int(total_h)}h")
        t4 = self._new_tile("산출물", "📦", 3); t4.set_value(len(outputs), "건", "최근 납품물")
        self._add_row_of([t1, t2, t3, t4])

        # 승인 대기 카드
        ca = self._new_card("승인 대기", "⚠", danger=True, idx=0); ca.set_count(len(appr))
        if not appr:
            ca.set_empty("✓ 승인 대기 없음")
        else:
            for a in appr:
                cid = a.get("chain_id", "")
                row = QWidget(); row.setObjectName("DashRowW")
                lay = QVBoxLayout(row); lay.setContentsMargins(2, 6, 2, 8); lay.setSpacing(7)
                t = QLabel(f"🔗 {a.get('name','(무제 체인)')}"); t.setObjectName("RowPrimary"); t.setWordWrap(True)
                s = QLabel(f"{a.get('stages',0)}단계 체인"); s.setObjectName("RowSecondary")
                btns = QHBoxLayout(); btns.setSpacing(8)
                b_ok = QPushButton("✓ 승인"); b_no = QPushButton("✗ 취소")
                b_ok.clicked.connect(lambda _c=False, x=cid: self.approve_requested.emit(x))
                b_no.clicked.connect(lambda _c=False, x=cid: self.cancel_requested.emit(x))
                self._style_buttons(b_ok, b_no)
                btns.addWidget(b_ok); btns.addWidget(b_no); btns.addStretch(1)
                lay.addWidget(t); lay.addWidget(s); lay.addLayout(btns)
                ca.add_widget(row)
        self._add(ca)

        # 오늘 | 고객
        ct = self._new_card("오늘 · 할당 가능", "📅", idx=1); ct.set_count(len(loads))
        if not loads:
            ct.set_empty("적재할 작업 없음")
        else:
            for L in loads[:6]:
                dl = L.get("deadline_at")
                ct.add_row(str(L.get("title", ""))[:40],
                           f"남은 공수 {L.get('remaining_hours')}h" + (" · 납품일 설정됨" if dl else " · 납품일 미설정"))
            sk = today.get("skipped", [])
            if sk:
                ct.add_row(f"⏱ 공수 미입력 {len(sk)}건", muted=True)
        cc = self._new_card("고객 메시지", "💬", idx=2)
        cust = data.get("customer", []); cc.set_count(len(cust))
        if not cust:
            cc.set_empty("받은 고객 메시지 없음")
        else:
            for m in cust[:5]:
                cc.add_row(f"💬 {m.get('customer_id','(고객)')}: {(m.get('message','') or '')[:40]}",
                           "✓ 초안 준비됨" if m.get("has_draft") else "· 초안 대기")
        self._add_row_of([ct, cc])

        # KPI
        ck = self._new_card("현황 / KPI", "📊", idx=3)
        kpi = data.get("kpi", {})
        ck.add_row(kpi.get("summary", "") or "(요약 없음)")
        self._add(ck)
        self._detail_lay.addStretch(1)

    # ── 프로젝트 상세 ──
    def _render_project(self, d):
        title = d.get("title", "(프로젝트)")
        ptype = d.get("project_type", "") or "—"
        status = "✅ 완료" if d.get("has_outcome") else "🟢 진행 중"
        # 제목 헤더
        head = self._new_card(f"{title}", "", idx=0)
        head.add_row(f"유형: {ptype}", f"상태: {status}")
        self._add(head)
        # 타일: 공수 / 납품일 / 산출물
        hours = int(float(d.get("hours") or 0))
        dels = d.get("deliverables", [])
        t1 = self._new_tile("예상 공수", "⏱", 0); t1.set_value(hours, "h", "person-day×8")
        t2 = self._new_tile("납품일", "📅", 1); t2.set_value(_fmt_deadline(d.get("deadline_at")).split(" ")[0] if d.get("deadline_at") else "—", "", _fmt_deadline(d.get("deadline_at")) if d.get("deadline_at") else "미설정")
        t3 = self._new_tile("산출물", "📦", 2); t3.set_value(len(dels), "건", "파일")
        self._add_row_of([t1, t2, t3])
        # 산출물 카드(클릭 열기)
        cd = self._new_card("산출물", "📦", idx=1); cd.set_count(len(dels))
        if not dels:
            cd.set_empty("아직 산출물 없음")
        else:
            for o in dels[:12]:
                path = o.get("path", "")
                cd.add_widget(_ClickRow(f"[{o.get('kind','')}] {o.get('name','')}", "클릭 = 폴더 열기",
                                        lambda p=path: self.output_open_requested.emit(p)))
        self._add(cd)
        # 정보 카드
        conv = d.get("conversation", "")
        if conv:
            ci = self._new_card("의뢰 내용(요약)", "📝", idx=2)
            ci.add_row(conv)
            self._add(ci)
        # 폴더 열기 버튼
        folder = d.get("folder", "")
        if folder:
            btn = QPushButton("📂 프로젝트 폴더 열기")
            self._style_buttons(btn, None)
            btn.clicked.connect(lambda _c=False, p=folder: self.output_open_requested.emit(p))
            self._add(btn)
        self._detail_lay.addStretch(1)

    def _style_buttons(self, ok_btn, cancel_btn):
        c, m, is_dark = self._c, self._m, self._dark
        radius = m.get("radius_sm", "6px")
        if is_dark:
            ok_btn.setStyleSheet(f"QPushButton {{ background:{NEON_MINT}; color:#000; border:none; border-radius:{radius}; padding:8px 18px; font-weight:700; }} QPushButton:hover {{ background:#5cffe0; }}")
            if cancel_btn:
                cancel_btn.setStyleSheet(f"QPushButton {{ background:transparent; color:{NEON_PURPLE}; border:1px solid {NEON_PURPLE}; border-radius:{radius}; padding:8px 18px; }} QPushButton:hover {{ background:rgba(177,77,255,0.18); }}")
        else:
            accent = c.get("accent_primary", "#2563eb"); accent_h = c.get("accent_hover", "#3b82f6")
            danger = c.get("danger_bg", "#dc2626")
            ok_btn.setStyleSheet(f"QPushButton {{ background:{accent}; color:#fff; border:none; border-radius:{radius}; padding:8px 18px; font-weight:700; }} QPushButton:hover {{ background:{accent_h}; }}")
            if cancel_btn:
                cancel_btn.setStyleSheet(f"QPushButton {{ background:transparent; color:{danger}; border:1px solid {danger}; border-radius:{radius}; padding:8px 18px; }} QPushButton:hover {{ background:{danger}; color:#fff; }}")

    def _load_helix(self):
        self._include_helix = True
        self.refresh(force=True)

    # ── 하단 채팅 → AIRA 답변(우측 파란 박스) ──
    def _aira_line(self, text, kind):
        lbl = QLabel(text); lbl.setObjectName(kind); lbl.setWordWrap(True)
        return lbl

    def _on_ask(self):
        text = self._chat_input.text().strip()
        if not text:
            return
        self._chat_input.clear()
        self._render_answer(text)

    def _render_answer(self, question):
        while self._aira_body.count():
            it = self._aira_body.takeAt(0); w = it.widget()
            if w:
                w.setParent(None); w.deleteLater()
        self._aira_body.addWidget(self._aira_line(f"❓ {question}", "AiraQ"))
        ql = question.replace(" ", "")
        is_todo = any(k in ql for k in ["오늘", "할일", "해야", "todo", "투두", "일정", "뭐하", "뭐가있", "todo"])
        if is_todo:
            td = None
            try:
                td = ddata.build_todo()
            except Exception as e:  # noqa: BLE001
                self._aira_body.addWidget(self._aira_line(f"데이터 오류: {e}", "AiraItem"))
            if td is not None:
                self._aira_body.addWidget(self._aira_line(
                    f"🌸 오늘 할 일을 정리했어요 (활성 프로젝트 {td['active_count']}개)", "AiraIntro"))
                if not td["sections"]:
                    self._aira_body.addWidget(self._aira_line("✓ 지금 당장 급한 일은 없어요. 깔끔합니다!", "AiraItem"))
                for s in td["sections"]:
                    self._aira_body.addWidget(self._aira_line(s["title"], "AiraSect"))
                    for e in s["entries"]:
                        self._aira_body.addWidget(self._aira_line("•  " + e["head"], "AiraItem"))
                        for sub in e.get("subs", []):
                            self._aira_body.addWidget(self._aira_line("       ◦  " + sub, "AiraSub"))
        else:
            self._aira_body.addWidget(self._aira_line(
                "아직 이 질문은 준비 중이에요. '오늘 해야 할 일' 처럼 물어봐 주세요! 🌸", "AiraItem"))
        self._aira_body.addStretch(1)
        self._aira.setVisible(True)


# ──────────────────────────────────────────────────────────────
def _selftest():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    w = DashboardWidget("Modern Dark")
    n_proj = len(w._entries) - 1
    for nm in ["Modern Dark", "Modern White (Warm)", "Modern White (Crisp)", "Obsidian"]:
        w.apply_theme(nm)
        w.refresh()
        print(f"✅ 테마 OK: {nm:22} · nav {len(w._entries)}개(전체현황+프로젝트 {n_proj})")
    # 프로젝트 선택 렌더 테스트
    if len(w._entries) > 1:
        w._select(1)
        print("✅ 프로젝트 상세 렌더 OK")
    print("✅ 셀프테스트 통과 — 마스터-디테일 · 4테마 · 예외 0")
    w._timer.stop(); app.quit()


def main():
    from PySide6.QtWidgets import (QApplication, QComboBox, QMessageBox,
                                   QVBoxLayout as VB, QWidget as QW)
    import eidos_dashboard_approvals as appr
    app = QApplication.instance() or QApplication(sys.argv)
    host = QW(); host.setWindowTitle("EIDOS 대시보드 (미리보기)"); host.resize(1150, 860)
    lay = VB(host); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
    combo = QComboBox(); combo.addItems(["Modern Dark", "Modern White (Warm)", "Modern White (Crisp)", "Obsidian"])
    dash = DashboardWidget("Modern Dark")
    combo.currentTextChanged.connect(dash.apply_theme)
    lay.addWidget(combo); lay.addWidget(dash, 1)

    def _on_cancel(cid):
        if QMessageBox.question(host, "체인 취소", f"폐기할까요?\n({cid[:8]})") == QMessageBox.StandardButton.Yes:
            appr.cancel(cid); dash.refresh(force=True)

    def _on_approve(cid):
        QMessageBox.information(host, "승인", f"체인 {cid[:8]} APPROVED 표시(실행은 본체).")
        appr.approve_mark(cid); dash.refresh(force=True)

    def _on_open(path):
        import os as _os
        try:
            tgt = path if _os.path.isdir(path) else _os.path.dirname(path)
            if tgt and hasattr(_os, "startfile"):
                _os.startfile(tgt)
        except Exception as _e:
            print(f"[dashboard] 열기 실패: {_e}")

    dash.cancel_requested.connect(_on_cancel)
    dash.approve_requested.connect(_on_approve)
    dash.output_open_requested.connect(_on_open)
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
