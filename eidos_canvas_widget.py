# eidos_canvas_widget.py
# 액션 조립 캔버스 — QGraphicsScene + QGraphicsItem 노드/연결선.
#
# 노드: 둥근 사각형 + verb 아이콘 + 라벨 + target 미리보기.
#   좌측 영역 = input 포트 (다른 노드의 output 이 여기로 연결)
#   우측 영역 = output 포트 (드래그로 다른 노드 input 에 연결)
# 연결선: 베지어 곡선·hover 시 보라 발광·우클릭 삭제.
# 캔버스: 휠 줌·우클릭 빈 공간 = 노드 추가 메뉴·드래그 빈 공간 = 팬.

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QPointF, QRectF, Signal, QObject
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsSceneMouseEvent,
    QGraphicsView,
    QMenu,
    QStyleOptionGraphicsItem,
    QWidget,
)

import eidos_canvas_store as _cstore


# ── 색상 (캔버스 전용·MoreActionsDialog._Palette 와 호환) ────────────────
# [2026-05-26] _CC 동적 팔레트 — eidos_widget_theme.is_light_theme() 결과로 라이트/다크 자동 분기.
# paint() / _paint_actor() / Scene/View 의 setBackgroundBrush 가 모두 _CC.X 접근하므로
# 메타클래스가 매 접근마다 현재 테마 dict 에서 lookup. 캐시 1회·테마 변경 후 refresh_canvas_theme()
# 부르면 갱신.

_PAL_DARK = {
    "BG":                 "#000000",
    "GRID":               "#0F0815",
    "NODE_BORDER":        "#7C3AED",
    "NODE_BORDER_SEL":    "#C084FC",
    "NODE_TEXT":          "#E9DFFB",
    "NODE_META":          "#8B7BA8",
    "PORT":               "#A855F7",
    "PORT_HOVER":         "#C084FC",
    "CONN":               "#7C3AED",
    "CONN_HOVER":         "#C084FC",
    # paint() 안에서 옛날엔 하드코딩했던 색들 — 팔레트로 승격
    "NODE_GRAD_MID":      "#1A0F2E",   # verb 노드 가운데 (보라 → 검정 fade)
    "NODE_GRAD_END":      "#000000",
    "NODE_GRAD_MID_ACTOR":"#15101F",   # actor 노드 가운데 (조금 더 어두운 톤)
    "NODE_ID_TEXT":       "#3A2461",   # 디버그용 node_id (옅은 보라)
    "ACTOR_SUB_TEXT":     "#A78BFA",   # actor indicators/repertoire 카운트
}

_PAL_LIGHT = {
    "BG":                 "#FFFFFF",
    "GRID":               "#E8EAF0",
    "NODE_BORDER":        "#7C3AED",
    "NODE_BORDER_SEL":    "#5B21B6",
    "NODE_TEXT":          "#2C2F48",   # 라이트 배경 위 어두운 글씨
    "NODE_META":          "#6C6F85",
    "PORT":               "#7C3AED",
    "PORT_HOVER":         "#5B21B6",
    "CONN":               "#7C3AED",
    "CONN_HOVER":         "#5B21B6",
    # 라이트 변형 — 그라데이션 끝이 흰색
    "NODE_GRAD_MID":      "#E8E2F5",   # 옅은 라벤더
    "NODE_GRAD_END":      "#FFFFFF",
    "NODE_GRAD_MID_ACTOR":"#EDE8F7",
    "NODE_ID_TEXT":       "#C0B8D8",   # 라이트 배경 위 옅은 회보라
    "ACTOR_SUB_TEXT":     "#7C3AED",   # 라이트 위 콘트라스트 위해 진한 보라
}


_active_palette_cache: Optional[dict] = None


def _resolve_canvas_palette() -> dict:
    """현재 테마에 맞는 팔레트 dict — eidos_widget_theme 의존성 graceful."""
    try:
        from eidos_widget_theme import is_light_theme
        if is_light_theme():
            return _PAL_LIGHT
    except Exception:
        pass
    return _PAL_DARK


def refresh_canvas_theme() -> None:
    """테마 변경 후 호출 — 다음 _CC 접근부터 새 팔레트 반영.
    기존 Scene/View 의 setBackgroundBrush 는 별도 (Scene/View 재생성 또는
    apply_theme_to_scene 호출 필요)."""
    global _active_palette_cache
    _active_palette_cache = _resolve_canvas_palette()


def _cc_palette() -> dict:
    global _active_palette_cache
    if _active_palette_cache is None:
        _active_palette_cache = _resolve_canvas_palette()
    return _active_palette_cache


class _CCMeta(type):
    """_CC.X 접근 시 현재 테마 팔레트 dict 에서 lookup."""
    def __getattr__(cls, name: str) -> str:
        pal = _cc_palette()
        if name in pal:
            return pal[name]
        raise AttributeError(f"_CC has no attribute {name!r}")


class _CC(metaclass=_CCMeta):
    """캔버스 색상 팔레트 — _CC.BG, _CC.NODE_BORDER 등으로 접근.
    실제 값은 현재 테마 (다크/라이트) 에 따라 동적 lookup."""
    pass


def apply_theme_to_scene(scene, view) -> None:
    """Scene/View 의 setBackgroundBrush 즉시 갱신 — 테마 변경 시 호출.
    그냥 _CC.X 접근하는 곳 (paint) 은 자동 반영되지만, __init__ 에서 한 번만 호출되는
    setBackgroundBrush 는 명시적으로 다시 칠해야 함."""
    try:
        bg = QColor(_CC.BG)
        if scene is not None:
            scene.setBackgroundBrush(QBrush(bg))
        if view is not None:
            view.setBackgroundBrush(QBrush(bg))
            view.setStyleSheet(f"QGraphicsView {{ border: none; background: {_CC.BG}; }}")
    except Exception as e:
        print(f"[canvas_widget] apply_theme_to_scene 실패 (graceful): {e}")


_NODE_W = 180
_NODE_H = 80
_PORT_R = 6   # 포트 반지름


# ── [2026-05-26 Phase 0-B] Actor 노드 메타 ────────────────────────────
# kind="actor" 노드 (다중 행위자 ToM 무대) 의 시각화 메타.
# eidos_agent_stage_store.ACTOR_TYPES 와 1:1.
_ACTOR_META: dict[str, dict] = {
    "self":         {"icon": "🤖", "color": "#7C3AED", "label": "EIDOS"},
    "human":        {"icon": "👤", "color": "#3B82F6", "label": "사람"},
    "organization": {"icon": "🏢", "color": "#10B981", "label": "조직"},
    "system":       {"icon": "📡", "color": "#F59E0B", "label": "시스템"},
    "agent":        {"icon": "🧠", "color": "#EC4899", "label": "다른 에이전트"},
    "abstract":     {"icon": "🧩", "color": "#6B7280", "label": "추상"},
}


def get_actor_meta(actor_type: str) -> dict:
    """actor_type → 시각화 메타 dict. 미지원 type 은 abstract fallback."""
    return _ACTOR_META.get(actor_type) or _ACTOR_META["abstract"]


def is_actor_node(node_data: dict) -> bool:
    return bool(node_data and node_data.get("kind") == "actor")


# ── 노드 ───────────────────────────────────────────────────────────────

class _CanvasNode(QGraphicsObject):
    """캔버스 노드 — 둥근 사각·verb + target 표시·드래그 이동·포트 2개·실행 상태 색상."""

    moved = Signal(str)            # node_id (좌표 변경 시)
    selected_changed = Signal(str) # node_id (선택 변화)
    delete_requested = Signal(str) # node_id (우클릭 → 삭제)
    port_drag_started = Signal(str)  # node_id (output 포트 드래그 시작)
    port_drag_ended = Signal(str, QPointF)  # node_id, scene_pos (drop 위치)

    # 실행 상태 ∈ {None / "running" / "ok" / "fail" / "skip" / "warn"}
    def __init__(self, node_data: dict, parent: Optional[QGraphicsItem] = None):
        super().__init__(parent)
        self.node_data = node_data
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setPos(node_data.get("x", 0), node_data.get("y", 0))
        self._dragging_port = False
        self._temp_drag_end: Optional[QPointF] = None
        self._port_drag_threshold = 6
        self._run_status: Optional[str] = None   # set_run_status 로 갱신

    def set_run_status(self, status: Optional[str]) -> None:
        """실행 상태 갱신 — 보더 색 변화 (None/running/ok/fail/skip/warn)."""
        self._run_status = status
        self.update()

    # ── 기하 ──
    def boundingRect(self) -> QRectF:
        # 포트 영역 포함해서 살짝 여유
        return QRectF(-2, -2, _NODE_W + 4, _NODE_H + 4)

    def _output_port_pos(self) -> QPointF:
        """출력 포트 중심 (로컬 좌표)."""
        return QPointF(_NODE_W, _NODE_H / 2)

    def _input_port_pos(self) -> QPointF:
        """입력 포트 중심 (로컬 좌표)."""
        return QPointF(0, _NODE_H / 2)

    def scene_output_pos(self) -> QPointF:
        return self.mapToScene(self._output_port_pos())

    def scene_input_pos(self) -> QPointF:
        return self.mapToScene(self._input_port_pos())

    # ── 페인트 ──
    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget=None) -> None:
        # [2026-05-26 Phase 0-B] kind ∈ {"verb"(default), "actor"} — 다른 시각화
        if is_actor_node(self.node_data):
            self._paint_actor(painter)
            return
        # 옛 verb 노드 (default)
        verb = self.node_data.get("verb", "?")
        meta = _cstore.get_verb_meta(verb) or {}
        icon = meta.get("icon", "?")
        label = self.node_data.get("label_override") or meta.get("label", verb)
        target = self.node_data.get("target", "")

        # 노드 둥근 사각 + 가로 그라데이션 (verb 색 → 배경)
        verb_color = QColor(meta.get("color", "#5B1AB0"))
        grad = QLinearGradient(0, 0, _NODE_W, 0)
        grad.setColorAt(0, verb_color)
        grad.setColorAt(0.6, QColor(_CC.NODE_GRAD_MID))
        grad.setColorAt(1, QColor(_CC.NODE_GRAD_END))
        painter.setBrush(QBrush(grad))
        # 실행 상태 → 보더 색 우선
        run_color_map = {
            "running": "#FBBF24",   # 노랑
            "ok":      "#34D399",   # 초록
            "fail":    "#F87171",   # 빨강
            "skip":    "#A78BFA",   # 연보라
            "warn":    "#FBBF24",   # 노랑
        }
        if self._run_status in run_color_map:
            border_color = QColor(run_color_map[self._run_status])
            pen_w = 3.0
        elif self.isSelected():
            border_color = QColor(_CC.NODE_BORDER_SEL)
            pen_w = 2.5
        else:
            border_color = QColor(_CC.NODE_BORDER)
            pen_w = 1.5
        painter.setPen(QPen(border_color, pen_w))
        painter.drawRoundedRect(QRectF(0, 0, _NODE_W, _NODE_H), 10, 10)

        # 아이콘 + verb 라벨 (윗줄)
        painter.setPen(QColor(_CC.NODE_TEXT))
        f1 = QFont()
        f1.setPointSize(11)
        f1.setBold(True)
        painter.setFont(f1)
        painter.drawText(QRectF(12, 8, _NODE_W - 20, 22), Qt.AlignLeft | Qt.AlignVCenter,
                         f"{icon}  {label}")

        # target / verb id 미리보기 (아래줄)
        painter.setPen(QColor(_CC.NODE_META))
        f2 = QFont()
        f2.setPointSize(9)
        painter.setFont(f2)
        sub = target if target else f"({verb})"
        if len(sub) > 24:
            sub = sub[:22] + ".."
        painter.drawText(QRectF(12, 32, _NODE_W - 20, 18), Qt.AlignLeft | Qt.AlignVCenter, sub)

        # 작은 노드 id (디버그용·아주 옅게)
        painter.setPen(QColor(_CC.NODE_ID_TEXT))
        f3 = QFont()
        f3.setPointSize(7)
        painter.setFont(f3)
        nid = self.node_data.get("id", "")
        if nid:
            painter.drawText(QRectF(12, _NODE_H - 18, _NODE_W - 20, 14),
                             Qt.AlignLeft | Qt.AlignVCenter, nid[:18])

        # 입력 포트 (좌측 원)
        painter.setBrush(QBrush(QColor(_CC.PORT)))
        painter.setPen(QPen(QColor(_CC.NODE_BORDER_SEL), 1.5))
        in_p = self._input_port_pos()
        painter.drawEllipse(in_p, _PORT_R, _PORT_R)

        # 출력 포트 (우측 원)
        painter.setBrush(QBrush(QColor(_CC.PORT)))
        out_p = self._output_port_pos()
        painter.drawEllipse(out_p, _PORT_R, _PORT_R)

        # 드래그 중 임시 선
        if self._dragging_port and self._temp_drag_end is not None:
            painter.setPen(QPen(QColor(_CC.CONN_HOVER), 2, Qt.DashLine))
            local_end = self.mapFromScene(self._temp_drag_end)
            painter.drawLine(out_p, local_end)

    # ── [Phase 0-B] actor 노드 시각화 ──
    def _paint_actor(self, painter: QPainter) -> None:
        """actor 노드 (kind=actor) 페인트 — verb 와 다른 색·아이콘·서브정보."""
        actor_type = self.node_data.get("actor_type", "human")
        meta = get_actor_meta(actor_type)
        icon = meta["icon"]
        type_label = meta["label"]
        actor_name = (self.node_data.get("name") or "(이름 없음)").strip()
        indicators = self.node_data.get("indicators") or {}
        repertoire = self.node_data.get("action_repertoire") or []
        role = self.node_data.get("role", "other")

        # 그라데이션 — actor 색 → 배경 (verb 와 같은 패턴이라 시각적 일관성)
        actor_color = QColor(meta["color"])
        grad = QLinearGradient(0, 0, _NODE_W, 0)
        grad.setColorAt(0, actor_color)
        grad.setColorAt(0.55, QColor(_CC.NODE_GRAD_MID_ACTOR))
        grad.setColorAt(1, QColor(_CC.NODE_GRAD_END))
        painter.setBrush(QBrush(grad))

        # 보더 — run_status 우선·role=self 면 항상 약간 강조·선택 시 보라
        run_color_map = {
            "running": "#FBBF24", "ok": "#34D399", "fail": "#F87171",
            "skip":    "#A78BFA", "warn": "#FBBF24",
        }
        if self._run_status in run_color_map:
            border_color = QColor(run_color_map[self._run_status])
            pen_w = 3.0
        elif self.isSelected():
            border_color = QColor(_CC.NODE_BORDER_SEL)
            pen_w = 2.5
        elif role == "self":
            # EIDOS-self 는 항상 살짝 보라 강조
            border_color = QColor(_CC.NODE_BORDER)
            pen_w = 2.2
        else:
            border_color = QColor(_CC.NODE_BORDER)
            pen_w = 1.5
        painter.setPen(QPen(border_color, pen_w))
        painter.drawRoundedRect(QRectF(0, 0, _NODE_W, _NODE_H), 10, 10)

        # 윗줄 — 아이콘 + actor 이름 (이름이 정보 핵심)
        painter.setPen(QColor(_CC.NODE_TEXT))
        f1 = QFont()
        f1.setPointSize(11)
        f1.setBold(True)
        painter.setFont(f1)
        name_text = f"{icon}  {actor_name}"
        if len(name_text) > 22:
            name_text = name_text[:20] + ".."
        painter.drawText(QRectF(12, 8, _NODE_W - 20, 22), Qt.AlignLeft | Qt.AlignVCenter,
                         name_text)

        # 가운데 줄 — actor type label (작게·옅게)
        painter.setPen(QColor(_CC.NODE_META))
        f2 = QFont()
        f2.setPointSize(9)
        painter.setFont(f2)
        type_sub = type_label
        if role == "self":
            type_sub = f"⭐ {type_label} (자기 자신)"
        painter.drawText(QRectF(12, 30, _NODE_W - 20, 16), Qt.AlignLeft | Qt.AlignVCenter,
                         type_sub)

        # 아래 줄 — indicators / repertoire count (있을 때만)
        f3 = QFont()
        f3.setPointSize(8)
        painter.setFont(f3)
        ind_count = len(indicators) if isinstance(indicators, dict) else 0
        rep_count = len(repertoire) if isinstance(repertoire, list) else 0
        sub_parts = []
        if ind_count:
            sub_parts.append(f"📊 {ind_count}")
        if rep_count:
            sub_parts.append(f"⚡ {rep_count}")
        if sub_parts:
            painter.setPen(QColor(_CC.ACTOR_SUB_TEXT))
            painter.drawText(QRectF(12, 48, _NODE_W - 20, 14), Qt.AlignLeft | Qt.AlignVCenter,
                             "  ·  ".join(sub_parts))

        # 노드 id (디버그용·옅게)
        painter.setPen(QColor(_CC.NODE_ID_TEXT))
        f4 = QFont()
        f4.setPointSize(7)
        painter.setFont(f4)
        nid = self.node_data.get("id", "")
        if nid:
            painter.drawText(QRectF(12, _NODE_H - 16, _NODE_W - 20, 12),
                             Qt.AlignLeft | Qt.AlignVCenter, nid[:18])

        # 입력 / 출력 포트 (verb 와 같음 — 연결 호환)
        painter.setBrush(QBrush(QColor(_CC.PORT)))
        painter.setPen(QPen(QColor(_CC.NODE_BORDER_SEL), 1.5))
        in_p = self._input_port_pos()
        painter.drawEllipse(in_p, _PORT_R, _PORT_R)
        out_p = self._output_port_pos()
        painter.drawEllipse(out_p, _PORT_R, _PORT_R)

        # 드래그 중 임시 선
        if self._dragging_port and self._temp_drag_end is not None:
            painter.setPen(QPen(QColor(_CC.CONN_HOVER), 2, Qt.DashLine))
            local_end = self.mapFromScene(self._temp_drag_end)
            painter.drawLine(out_p, local_end)

    # ── 마우스 이벤트 — 출력 포트 드래그 시작/끝 감지 ──
    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            out_p = self._output_port_pos()
            local = event.pos()
            dist = ((local.x() - out_p.x()) ** 2 + (local.y() - out_p.y()) ** 2) ** 0.5
            if dist <= _PORT_R + 3:
                # 출력 포트 영역 클릭 → 드래그 모드
                self._dragging_port = True
                self._temp_drag_end = event.scenePos()
                self.port_drag_started.emit(self.node_data.get("id", ""))
                self.update()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._dragging_port:
            self._temp_drag_end = event.scenePos()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._dragging_port:
            self._dragging_port = False
            end = self._temp_drag_end or event.scenePos()
            self._temp_drag_end = None
            self.update()
            self.port_drag_ended.emit(self.node_data.get("id", ""), end)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event) -> None:
        menu = QMenu()
        del_act = menu.addAction("🗑  노드 삭제")
        chosen = menu.exec(event.screenPos())
        if chosen == del_act:
            self.delete_requested.emit(self.node_data.get("id", ""))

    # ── 변경 알림 ──
    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            # snap-to-grid (scene 의 _snap_size 가 0 이면 off)
            try:
                scene = self.scene()
                snap = getattr(scene, "_snap_size", 0) if scene else 0
                if snap and snap > 0:
                    nx = round(value.x() / snap) * snap
                    ny = round(value.y() / snap) * snap
                    if nx != value.x() or ny != value.y():
                        value = QPointF(nx, ny)
            except Exception:
                pass
            return value
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.node_data["x"] = float(self.pos().x())
            self.node_data["y"] = float(self.pos().y())
            self.moved.emit(self.node_data.get("id", ""))
        elif change == QGraphicsItem.ItemSelectedHasChanged:
            self.selected_changed.emit(self.node_data.get("id", ""))
        return super().itemChange(change, value)


# ── 연결선 ─────────────────────────────────────────────────────────────

class _CanvasConnection(QGraphicsPathItem):
    """베지어 곡선 — 두 노드의 출력→입력 연결."""

    def __init__(self, from_node: _CanvasNode, to_node: _CanvasNode,
                 parent: Optional[QGraphicsItem] = None):
        super().__init__(parent)
        self.from_node = from_node
        self.to_node = to_node
        self.setZValue(-1)   # 노드 뒤에
        self.setPen(QPen(QColor(_CC.CONN), 2))
        self.setAcceptHoverEvents(True)
        self.update_path()

    def update_path(self) -> None:
        try:
            p1 = self.from_node.scene_output_pos()
            p2 = self.to_node.scene_input_pos()
            path = QPainterPath(p1)
            # 베지어 control points — 수평 거리의 절반
            dx = max(40.0, abs(p2.x() - p1.x()) * 0.5)
            c1 = QPointF(p1.x() + dx, p1.y())
            c2 = QPointF(p2.x() - dx, p2.y())
            path.cubicTo(c1, c2, p2)
            self.setPath(path)
        except Exception as e:
            print(f"[CanvasConnection] update_path 실패: {e}")

    def hoverEnterEvent(self, event) -> None:
        self.setPen(QPen(QColor(_CC.CONN_HOVER), 3))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self.setPen(QPen(QColor(_CC.CONN), 2))
        super().hoverLeaveEvent(event)


# ── Scene + View ───────────────────────────────────────────────────────

class _CanvasScene(QGraphicsScene):
    """캔버스 scene — 노드/연결선 관리·신호 전파."""

    node_selected = Signal(str)         # node_id (inspector 갱신)
    node_changed = Signal()             # 노드 추가/삭제/이동/연결 변화 (저장 trigger)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.setBackgroundBrush(QBrush(QColor(_CC.BG)))
        self._nodes: dict[str, _CanvasNode] = {}
        self._connections: list[_CanvasConnection] = []
        self._pending_drag_source: Optional[str] = None
        self.setSceneRect(-2000, -2000, 4000, 4000)
        # v2.3 snap-to-grid (0 = off·default on=20)
        self._snap_size: int = 20
        # v2.3 undo/redo (JSON snapshot 기반) + clipboard
        import json as _json_lib
        self._json = _json_lib
        self._undo_stack: list[str] = []
        self._redo_stack: list[str] = []
        self._max_undo = 30
        self._clipboard: list[dict] = []   # 복사된 노드 dict list (deep copy)
        self._suppress_snapshot = False    # snapshot 호출 안 함 (load 중·undo/redo 중)

    # ── 노드 / 연결 추가 / 삭제 ──
    def add_node(self, node_data: dict) -> _CanvasNode:
        self._push_snapshot()
        node = _CanvasNode(node_data)
        node.moved.connect(self._on_node_moved)
        node.selected_changed.connect(self._on_node_selected)
        node.delete_requested.connect(self.remove_node)
        node.port_drag_started.connect(self._on_port_drag_start)
        node.port_drag_ended.connect(self._on_port_drag_end)
        self.addItem(node)
        self._nodes[node_data["id"]] = node
        self.node_changed.emit()
        return node

    def remove_node(self, node_id: str) -> None:
        node = self._nodes.pop(node_id, None)
        if node is None:
            return
        self._push_snapshot()
        # 연결된 connection 도 제거
        remaining = []
        for c in self._connections:
            if c.from_node is node or c.to_node is node:
                self.removeItem(c)
            else:
                remaining.append(c)
        self._connections = remaining
        self.removeItem(node)
        self.node_changed.emit()

    def add_connection(self, from_id: str, to_id: str) -> Optional[_CanvasConnection]:
        if from_id == to_id:
            return None
        if from_id not in self._nodes or to_id not in self._nodes:
            return None
        # 중복 방지
        for c in self._connections:
            if c.from_node.node_data["id"] == from_id and c.to_node.node_data["id"] == to_id:
                return None
        self._push_snapshot()
        conn = _CanvasConnection(self._nodes[from_id], self._nodes[to_id])
        self.addItem(conn)
        self._connections.append(conn)
        self.node_changed.emit()
        return conn

    def clear_all(self) -> None:
        self._nodes.clear()
        self._connections.clear()
        self.clear()

    # ── v2.3 snap-to-grid + auto layout ───────────────────────────
    def set_snap_size(self, size: int) -> None:
        """grid 스냅 단위 — 0 이면 off."""
        self._snap_size = max(0, int(size))

    def auto_layout(self, column_gap: int = 220, row_gap: int = 110, origin_x: int = 50,
                    origin_y: int = 80) -> None:
        """topology depth 기반 자동 배치 — 같은 depth 의 노드는 세로로 정렬.
        cycle 있어도 graceful (topology_sort 가 fallback append)."""
        if not self._nodes:
            return
        try:
            canvas_dict = self.to_canvas_dict("auto_layout_tmp")
            ordered = _cstore.topology_sort(canvas_dict)
            nodes_by_id = {n["id"]: n for n in canvas_dict.get("nodes", [])}
            # 각 노드의 depth (longest path from any root)
            depth: dict[str, int] = {}
            in_to_out: dict[str, list[str]] = {nid: [] for nid in nodes_by_id}
            for c in canvas_dict.get("connections", []):
                fr, to = c.get("from"), c.get("to")
                if fr in nodes_by_id and to in nodes_by_id:
                    in_to_out.setdefault(fr, []).append(to)
            # 초기 depth = 0
            for n in ordered:
                depth[n["id"]] = 0
            # topology 순서대로 — depth[child] = max(depth[child], depth[parent]+1)
            for n in ordered:
                d = depth[n["id"]]
                for child in in_to_out.get(n["id"], []):
                    depth[child] = max(depth.get(child, 0), d + 1)
            # depth → 노드 list
            by_depth: dict[int, list[str]] = {}
            for nid, d in depth.items():
                by_depth.setdefault(d, []).append(nid)
            # 배치 — 같은 depth 의 노드는 y 좌표 정렬 (현재 위치 기준 안정성)
            for d in sorted(by_depth):
                nids = by_depth[d]
                # 현재 y 순서 유지
                nids.sort(key=lambda nid: self._nodes[nid].pos().y() if nid in self._nodes else 0)
                for row, nid in enumerate(nids):
                    if nid in self._nodes:
                        new_x = origin_x + d * column_gap
                        new_y = origin_y + row * row_gap
                        self._nodes[nid].setPos(new_x, new_y)
            # 연결선 모두 재계산
            for c in self._connections:
                c.update_path()
        except Exception as e:
            print(f"[CanvasScene] auto_layout 실패 (graceful): {e}")

    # ── 캔버스 → dict (저장용) ──
    def to_canvas_dict(self, name: str = "untitled", description: str = "") -> dict:
        canvas = _cstore.make_empty_canvas(name)
        canvas["description"] = description
        canvas["nodes"] = [dict(n.node_data) for n in self._nodes.values()]
        canvas["connections"] = [
            {"from": c.from_node.node_data["id"], "to": c.to_node.node_data["id"]}
            for c in self._connections
        ]
        return canvas

    def load_from_canvas_dict(self, canvas: dict) -> None:
        # load 중에는 snapshot push 안 함 (단일 load 가 add_node 마다 stack 쌓이지 않게)
        self._suppress_snapshot = True
        try:
            self.clear_all()
            for nd in canvas.get("nodes", []):
                self.add_node(dict(nd))
            for conn in canvas.get("connections", []):
                self.add_connection(conn.get("from"), conn.get("to"))
        finally:
            self._suppress_snapshot = False

    # ── v2.3 undo/redo (JSON snapshot 기반) ───────────────────────
    def _push_snapshot(self) -> None:
        """현재 상태 snapshot 을 undo stack 에 push. _suppress_snapshot True 면 skip."""
        if self._suppress_snapshot:
            return
        try:
            snap = self._json.dumps(self.to_canvas_dict("undo_snap"), ensure_ascii=False)
            self._undo_stack.append(snap)
            if len(self._undo_stack) > self._max_undo:
                self._undo_stack.pop(0)
            self._redo_stack.clear()
        except Exception as e:
            print(f"[CanvasScene] snapshot push 실패 (graceful): {e}")

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        try:
            snap = self._undo_stack.pop()
            cur = self._json.dumps(self.to_canvas_dict("redo_snap"), ensure_ascii=False)
            self._redo_stack.append(cur)
            data = self._json.loads(snap)
            self.load_from_canvas_dict(data)
            return True
        except Exception as e:
            print(f"[CanvasScene] undo 실패: {e}")
            return False

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        try:
            snap = self._redo_stack.pop()
            cur = self._json.dumps(self.to_canvas_dict("undo_snap"), ensure_ascii=False)
            self._undo_stack.append(cur)
            data = self._json.loads(snap)
            self.load_from_canvas_dict(data)
            return True
        except Exception as e:
            print(f"[CanvasScene] redo 실패: {e}")
            return False

    # ── v2.3 복사 / 붙여넣기 ─────────────────────────────────────
    def copy_selected(self) -> int:
        """선택된 노드들을 _clipboard 에 deep copy. 반환: 복사된 노드 수."""
        sel_nodes = [n for n in self._nodes.values() if n.isSelected()]
        if not sel_nodes:
            return 0
        self._clipboard = [dict(n.node_data) for n in sel_nodes]
        return len(self._clipboard)

    def paste_clipboard(self, offset_x: float = 30.0, offset_y: float = 30.0) -> int:
        """_clipboard 의 노드들을 새 node_id 부여 + 오프셋으로 추가. 반환: 추가된 노드 수."""
        if not self._clipboard:
            return 0
        import uuid as _uuid
        self._push_snapshot()
        self._suppress_snapshot = True
        try:
            count = 0
            # 기존 선택 해제 — 새 붙여넣은 노드들만 선택
            for n in self._nodes.values():
                n.setSelected(False)
            for src in self._clipboard:
                new_data = dict(src)
                new_data["id"] = f"node_{_uuid.uuid4().hex[:10]}"
                new_data["x"] = float(src.get("x", 0)) + offset_x
                new_data["y"] = float(src.get("y", 0)) + offset_y
                node = self.add_node(new_data)
                node.setSelected(True)
                count += 1
            return count
        finally:
            self._suppress_snapshot = False

    # ── 신호 핸들러 ──
    def _on_node_moved(self, node_id: str) -> None:
        # 연결선 다시 그림
        for c in self._connections:
            if c.from_node.node_data["id"] == node_id or c.to_node.node_data["id"] == node_id:
                c.update_path()
        self.node_changed.emit()

    def _on_node_selected(self, node_id: str) -> None:
        node = self._nodes.get(node_id)
        if node and node.isSelected():
            self.node_selected.emit(node_id)

    def _on_port_drag_start(self, node_id: str) -> None:
        self._pending_drag_source = node_id

    def _on_port_drag_end(self, source_id: str, scene_pos: QPointF) -> None:
        # drop 위치에서 입력 포트 hit-test
        if self._pending_drag_source != source_id:
            return
        self._pending_drag_source = None
        target_id = self._find_input_port_at(scene_pos, exclude=source_id)
        if target_id:
            self.add_connection(source_id, target_id)

    def _find_input_port_at(self, scene_pos: QPointF, exclude: str = "") -> Optional[str]:
        """scene_pos 가 어떤 노드의 입력 포트 (좌측 영역) 안인지."""
        for nid, node in self._nodes.items():
            if nid == exclude:
                continue
            in_scene = node.scene_input_pos()
            dist = ((scene_pos.x() - in_scene.x()) ** 2 + (scene_pos.y() - in_scene.y()) ** 2) ** 0.5
            if dist <= _PORT_R + 8:
                return nid
        return None


class _CanvasView(QGraphicsView):
    """캔버스 뷰포트 — 휠 줌·드래그 팬·빈 공간 우클릭 = actor preset 메뉴."""

    def __init__(self, scene: _CanvasScene, parent: Optional[QWidget] = None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setBackgroundBrush(QBrush(QColor(_CC.BG)))
        self.setStyleSheet(f"QGraphicsView {{ border: none; background: {_CC.BG}; }}")
        self._zoom = 1.0

    def wheelEvent(self, event) -> None:
        # Ctrl 없이도 줌 (캔버스 UX 관습)
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        new_zoom = self._zoom * factor
        if 0.3 < new_zoom < 4.0:
            self._zoom = new_zoom
            self.scale(factor, factor)
        event.accept()

    def contextMenuEvent(self, event) -> None:
        """[Phase 0-B 2026-05-26] 빈 공간 우클릭 → actor preset 5종 + verb 추가 메뉴.

        노드 위 우클릭은 _CanvasNode.contextMenuEvent (삭제) 가 먼저 잡음 — 여기는 빈 공간만.
        """
        try:
            scene_pos = self.mapToScene(event.pos())
            # 노드 위 클릭이면 (item 존재) 기본 동작 (노드의 contextMenuEvent 가 처리)
            item = self.scene().itemAt(scene_pos, self.transform())
            if item is not None:
                super().contextMenuEvent(event)
                return

            menu = QMenu(self)
            # ── Actor preset 5종 (ToM 무대) ──
            actor_section = menu.addAction("🎭  ── Actor preset 추가 ──")
            actor_section.setEnabled(False)
            for atype, meta in _ACTOR_META.items():
                act = menu.addAction(f"{meta['icon']}  {meta['label']}  ({atype})")
                act.setData(("actor", atype))
            menu.addSeparator()
            # ── Verb preset (옛 12 verb) ──
            verb_section = menu.addAction("⚡  ── Verb preset 추가 ──")
            verb_section.setEnabled(False)
            for v in _cstore.VERBS:
                act = menu.addAction(f"{v.get('icon','·')}  {v.get('label', v['id'])}  ({v['id']})")
                act.setData(("verb", v["id"]))

            chosen = menu.exec(event.globalPos())
            if not chosen:
                return
            payload = chosen.data()
            if not isinstance(payload, tuple) or len(payload) != 2:
                return
            kind, key = payload
            # scene_pos 기준 노드 추가 — 노드 중심이 클릭 위치 부근에
            x = scene_pos.x() - _NODE_W / 2
            y = scene_pos.y() - _NODE_H / 2
            scene = self.scene()
            if scene is None:
                return
            if kind == "verb":
                nd = _cstore.make_empty_node(key, x, y)
                scene.add_node(nd)
            elif kind == "actor":
                # 지연 import — eidos_agent_stage_store 가 같은 폴더에 있을 때만
                try:
                    from eidos_agent_stage_store import make_actor_node
                except Exception as _e_imp:
                    print(f"[CanvasView] actor 추가 실패 (agent_stage_store 없음): {_e_imp}")
                    return
                meta = _ACTOR_META.get(key) or _ACTOR_META["abstract"]
                role = "self" if key == "self" else "other"
                nd = make_actor_node(
                    name=meta["label"],
                    actor_type=key,
                    role=role,
                    x=x, y=y,
                )
                scene.add_node(nd)
        except Exception as e:
            print(f"[CanvasView] contextMenuEvent 실패 (graceful): {e}")
            super().contextMenuEvent(event)
