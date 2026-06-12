# eidos_goal_tree_dialog.py
# [2026-05-27 Phase 9-D] GoalTree CRUD UI — 햄버거 "🎯 목표 트리" 진입점.
#
# 사용 흐름:
#   1. 햄버거 → "🎯 목표 트리" → Stage picker (저장된 stage 중 1개 선택)
#   2. GoalTreeDialog 열림 — 선택된 Stage 의 goal_tree.json 로드 (없으면 자동 생성)
#   3. 좌측 QTreeWidget — milestone 위계·우측 detail panel (편집 form)
#   4. 모든 편집은 자동 save (dirty 마커 X)
#   5. Stage.goal 변경 시 root milestone title 자동 sync (열 때 sync_root_with_stage)
#
# 모덜리스 (사용자가 채팅과 병행 가능). 단일 인스턴스 가드 (이미 열려있으면 raise).
#
# UI 변화 0 원칙 (기존 채팅 UI 영향 X). 별도 파일·햄버거 메뉴 1줄 추가만.

from __future__ import annotations

import datetime as _dt
from typing import Optional

try:
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QSplitter, QWidget,
        QTreeWidget, QTreeWidgetItem, QPushButton, QLabel,
        QLineEdit, QTextEdit, QComboBox, QSlider, QListWidget,
        QListWidgetItem, QInputDialog, QMessageBox, QFormLayout,
        QGroupBox, QScrollArea, QSizePolicy,
    )
    _HAS_QT = True
except Exception as _e_qt:
    print(f"[goal_tree_dialog] PySide6 import 실패 (graceful): {_e_qt}")
    _HAS_QT = False


# ── 모듈 의존성 ────────────────────────────────────────────────────────
try:
    import eidos_goal_tree as _gt
    import eidos_agent_stage_store as _ass
    _HAS_DEPS = True
except Exception as _e_dep:
    print(f"[goal_tree_dialog] 의존 모듈 import 실패 (graceful): {_e_dep}")
    _HAS_DEPS = False


# horizon → 표시명 + 아이콘
_HORIZON_LABEL = {
    "year":    "🌳 연",
    "quarter": "🌲 분기",
    "month":   "🌿 월",
    "week":    "🌱 주",
    "task":    "📌 태스크",
}

_STATUS_LABEL = {
    "active":     "▶ 진행 중",
    "done":       "✅ 완료",
    "abandoned":  "🚫 포기",
    "superseded": "🔄 무효 (대체됨)",
}


# ── helpers ────────────────────────────────────────────────────────────
def _format_milestone_text(m) -> str:
    """QTreeWidgetItem text — '{horizon} title  · 진행도% · 상태'."""
    hz = _HORIZON_LABEL.get(m.horizon, m.horizon)
    pct = f"{m.progress*100:.0f}%"
    status_tag = ""
    if m.status == "done":
        status_tag = " ✅"
    elif m.status == "abandoned":
        status_tag = " 🚫"
    elif m.status == "superseded":
        status_tag = " 🔄"
    target_tag = f" · ~{m.target_date}" if m.target_date else ""
    return f"{hz}  {m.title}   ·   {pct}{target_tag}{status_tag}"


# ── GoalTreeDialog ─────────────────────────────────────────────────────
if _HAS_QT and _HAS_DEPS:

    class GoalTreeDialog(QDialog):
        """1 Stage 의 GoalTree 편집 다이얼로그.

        - 모덜리스 (채팅과 병행)
        - 모든 편집은 자동 save_goal_tree
        - 닫기 시 별도 prompt 없음 (자동 저장이라 안전)
        """

        tree_changed = Signal()  # 외부 알림 (배지 갱신 등 — 옵션)

        # 단일 인스턴스 가드 (parent 별)
        _OPEN_INSTANCES: "dict[str, GoalTreeDialog]" = {}

        def __init__(self, stage, parent=None):
            super().__init__(parent)
            self._stage = stage              # eidos_agent_stage_store.Stage
            self._tree = None                # eidos_goal_tree.GoalTree
            self._current_milestone_id: str = ""
            self._suppress_signals = False   # 폼 채울 때 signal 차단

            self.setWindowTitle(f"🎯 목표 트리 — {stage.name}")
            self.setMinimumSize(900, 600)
            # 모덜리스
            self.setModal(False)

            self._build_ui()
            self._load_tree()
            self._refresh_tree_widget()

            # 트리 첫 항목 선택 (있다면)
            if self._tree_widget.topLevelItemCount() > 0:
                self._tree_widget.setCurrentItem(self._tree_widget.topLevelItem(0))

        # ── UI 구성 ───────────────────────────────────────────────
        def _build_ui(self) -> None:
            root_layout = QVBoxLayout(self)
            root_layout.setContentsMargins(8, 8, 8, 8)
            root_layout.setSpacing(6)

            # 상단 헤더 — Stage 정보 + horizon_end
            header = QHBoxLayout()
            self._lbl_stage = QLabel(
                f"<b>Stage:</b> {self._stage.name}  ·  "
                f"<b>Goal:</b> {(self._stage.goal or '(빈)')[:80]}"
            )
            self._lbl_stage.setStyleSheet("color:#bfdbfe;")
            header.addWidget(self._lbl_stage, 1)

            self._lbl_summary = QLabel("")
            self._lbl_summary.setStyleSheet("color:#94a3b8;")
            header.addWidget(self._lbl_summary)
            root_layout.addLayout(header)

            # 메인 splitter (좌 트리·우 detail panel)
            splitter = QSplitter(Qt.Horizontal)
            root_layout.addWidget(splitter, 1)

            # ─── 좌: tree widget + 트리 액션 ───
            left_box = QWidget()
            left_layout = QVBoxLayout(left_box)
            left_layout.setContentsMargins(0, 0, 0, 0)
            left_layout.setSpacing(4)

            self._tree_widget = QTreeWidget()
            self._tree_widget.setHeaderLabel("milestone 위계")
            self._tree_widget.setMinimumWidth(360)
            self._tree_widget.itemSelectionChanged.connect(self._on_tree_selection)
            left_layout.addWidget(self._tree_widget, 1)

            tree_actions = QHBoxLayout()
            self._btn_add_root = QPushButton("➕ 최상위 추가")
            self._btn_add_root.clicked.connect(self._on_add_root_clicked)
            tree_actions.addWidget(self._btn_add_root)
            self._btn_add_child = QPushButton("➕ 자식 추가")
            self._btn_add_child.clicked.connect(self._on_add_child_clicked)
            tree_actions.addWidget(self._btn_add_child)
            self._btn_delete = QPushButton("❌ 삭제")
            self._btn_delete.clicked.connect(self._on_delete_clicked)
            tree_actions.addWidget(self._btn_delete)
            left_layout.addLayout(tree_actions)

            tree_actions2 = QHBoxLayout()
            self._btn_recompute = QPushButton("🔁 진행도 재계산")
            self._btn_recompute.setToolTip("children 평균으로 internal milestone 의 progress 자동 재계산")
            self._btn_recompute.clicked.connect(self._on_recompute_clicked)
            tree_actions2.addWidget(self._btn_recompute)
            self._btn_evaluate = QPushButton("📈 history 평가")
            self._btn_evaluate.setToolTip(
                "stage history 의 키워드로 leaf milestone 의 success_criteria 매칭 → "
                "progress 자동 추정"
            )
            self._btn_evaluate.clicked.connect(self._on_evaluate_clicked)
            tree_actions2.addWidget(self._btn_evaluate)
            # [Phase 9-E] LLM 자동 분해 — 선택된 milestone 을 N 개 하위로 자동 제안
            self._btn_decompose = QPushButton("🪄 LLM 자동 분해")
            self._btn_decompose.setToolTip(
                "선택된 milestone (없으면 root) 을 LLM 이 N 개 하위 milestone 으로 "
                "자동 제안. 미리보기 후 confirm 시 일괄 추가. (Phase 9-E)"
            )
            self._btn_decompose.clicked.connect(self._on_decompose_clicked)
            tree_actions2.addWidget(self._btn_decompose)
            left_layout.addLayout(tree_actions2)

            splitter.addWidget(left_box)

            # ─── 우: detail panel ───
            self._detail_panel = self._build_detail_panel()
            splitter.addWidget(self._detail_panel)

            splitter.setSizes([400, 480])

            # 하단 닫기 버튼
            bottom = QHBoxLayout()
            bottom.addStretch(1)
            self._btn_close = QPushButton("닫기")
            self._btn_close.clicked.connect(self.close)
            bottom.addWidget(self._btn_close)
            root_layout.addLayout(bottom)

        def _build_detail_panel(self) -> QWidget:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            inner = QWidget()
            scroll.setWidget(inner)
            form = QFormLayout(inner)
            form.setContentsMargins(8, 8, 8, 8)
            form.setSpacing(6)

            self._fld_title = QLineEdit()
            self._fld_title.editingFinished.connect(self._on_form_save)
            form.addRow("제목:", self._fld_title)

            self._fld_description = QTextEdit()
            self._fld_description.setMaximumHeight(70)
            self._fld_description.focusOutEvent = self._wrap_focus_out(
                self._fld_description.focusOutEvent
            )
            form.addRow("설명:", self._fld_description)

            # horizon (읽기 전용 표시)
            self._lbl_horizon = QLabel("(없음)")
            form.addRow("호라이즌:", self._lbl_horizon)

            self._fld_target_date = QLineEdit()
            self._fld_target_date.setPlaceholderText("YYYY-MM-DD (옵션)")
            self._fld_target_date.editingFinished.connect(self._on_form_save)
            form.addRow("목표일:", self._fld_target_date)

            self._combo_status = QComboBox()
            for s, label in _STATUS_LABEL.items():
                self._combo_status.addItem(label, s)
            self._combo_status.currentIndexChanged.connect(self._on_status_changed)
            form.addRow("상태:", self._combo_status)

            # progress slider + manual override
            prog_row = QHBoxLayout()
            self._slider_progress = QSlider(Qt.Horizontal)
            self._slider_progress.setRange(0, 100)
            self._slider_progress.setSingleStep(5)
            self._slider_progress.valueChanged.connect(self._on_progress_slider)
            prog_row.addWidget(self._slider_progress, 1)
            self._lbl_progress_val = QLabel("0%")
            self._lbl_progress_val.setMinimumWidth(40)
            prog_row.addWidget(self._lbl_progress_val)
            prog_wrap = QWidget()
            prog_wrap.setLayout(prog_row)
            form.addRow("진행도:", prog_wrap)

            self._lbl_progress_meta = QLabel("(자동 — children 평균)")
            self._lbl_progress_meta.setStyleSheet("color:#94a3b8; font-size:11px;")
            form.addRow("", self._lbl_progress_meta)

            # success_criteria — QListWidget + 추가/삭제 버튼
            crit_box = QGroupBox("성공 기준 (success_criteria)")
            crit_layout = QVBoxLayout(crit_box)
            crit_layout.setContentsMargins(6, 12, 6, 6)
            crit_layout.setSpacing(4)
            self._list_criteria = QListWidget()
            self._list_criteria.setMaximumHeight(100)
            crit_layout.addWidget(self._list_criteria)
            crit_btns = QHBoxLayout()
            self._btn_crit_add = QPushButton("➕ 추가")
            self._btn_crit_add.clicked.connect(self._on_criterion_add)
            crit_btns.addWidget(self._btn_crit_add)
            self._btn_crit_del = QPushButton("❌ 선택 삭제")
            self._btn_crit_del.clicked.connect(self._on_criterion_del)
            crit_btns.addWidget(self._btn_crit_del)
            crit_layout.addLayout(crit_btns)
            form.addRow(crit_box)

            self._fld_notes = QTextEdit()
            self._fld_notes.setMaximumHeight(60)
            self._fld_notes.focusOutEvent = self._wrap_focus_out(
                self._fld_notes.focusOutEvent
            )
            form.addRow("메모:", self._fld_notes)

            # 메타 (id·source·created_at·completed_at)
            self._lbl_meta = QLabel("")
            self._lbl_meta.setStyleSheet("color:#64748b; font-size:11px;")
            form.addRow("메타:", self._lbl_meta)

            return scroll

        def _wrap_focus_out(self, original_handler):
            """QTextEdit 의 focusOut 에서 자동 save."""
            def _handler(event):
                try:
                    original_handler(event)
                except Exception:
                    pass
                self._on_form_save()
            return _handler

        # ── tree 로딩/갱신 ─────────────────────────────────────────
        def _load_tree(self) -> None:
            """디스크에서 GoalTree 로드 — 없으면 Stage.goal 기준으로 새로 생성.
            Stage.goal 이 변경됐다면 root.title sync."""
            sid = self._stage.id
            tree = _gt.load_goal_tree(sid)
            if tree is None:
                # 새로 생성 (Stage.goal 비어있으면 root 없는 빈 tree)
                tree = _gt.new_goal_tree(
                    sid,
                    root_title=(self._stage.goal or "").strip(),
                    horizon="year",
                )
                _gt.save_goal_tree(tree)
            else:
                # Stage.goal 과 root.title 동기 (변경 있으면 save)
                try:
                    if _gt.sync_root_with_stage(tree, self._stage):
                        _gt.save_goal_tree(tree)
                except Exception as e:
                    print(f"[goal_tree_dialog] sync_root_with_stage 실패 (graceful): {e}")
            self._tree = tree
            self._update_summary()

        def _refresh_tree_widget(self) -> None:
            """QTreeWidget 전체 재구성. 선택 보존 — _current_milestone_id."""
            self._tree_widget.blockSignals(True)
            self._tree_widget.clear()

            if self._tree is None or not self._tree.milestones:
                self._tree_widget.blockSignals(False)
                self._refresh_form_panel(None)
                return

            # root 부터 DFS
            id_to_item: dict[str, QTreeWidgetItem] = {}
            roots = []
            if self._tree.root_goal_id and self._tree.get(self._tree.root_goal_id):
                roots.append(self._tree.get(self._tree.root_goal_id))
            # 추가로 parent_id="" 이고 root 가 아닌 orphan 도 root 로 표시
            for m in self._tree.all_milestones():
                if m.parent_id == "" and m.id != self._tree.root_goal_id:
                    roots.append(m)

            def _add_node(m, parent_item: Optional[QTreeWidgetItem]) -> None:
                item = QTreeWidgetItem([_format_milestone_text(m)])
                item.setData(0, Qt.UserRole, m.id)
                if parent_item is None:
                    self._tree_widget.addTopLevelItem(item)
                else:
                    parent_item.addChild(item)
                id_to_item[m.id] = item
                for cid in m.children_ids:
                    child = self._tree.get(cid)
                    if child is not None:
                        _add_node(child, item)

            for r in roots:
                _add_node(r, None)

            self._tree_widget.expandAll()
            self._tree_widget.blockSignals(False)

            # 선택 복원
            if self._current_milestone_id and self._current_milestone_id in id_to_item:
                self._tree_widget.setCurrentItem(id_to_item[self._current_milestone_id])
            self._update_summary()

        def _update_summary(self) -> None:
            if self._tree is None:
                self._lbl_summary.setText("")
                return
            try:
                s = _gt.summary_for_log(self._tree)
                stat = s.get("by_status") or {}
                self._lbl_summary.setText(
                    f"{s.get('total_milestones', 0)} milestones  ·  "
                    f"active {stat.get('active', 0)}  ·  done {stat.get('done', 0)}  ·  "
                    f"abandoned {stat.get('abandoned', 0)}"
                )
            except Exception:
                self._lbl_summary.setText("")

        # ── 선택 변경 ─────────────────────────────────────────────
        def _on_tree_selection(self) -> None:
            items = self._tree_widget.selectedItems()
            if not items:
                self._current_milestone_id = ""
                self._refresh_form_panel(None)
                return
            mid = items[0].data(0, Qt.UserRole)
            if not mid:
                return
            self._current_milestone_id = mid
            m = self._tree.get(mid) if self._tree else None
            self._refresh_form_panel(m)

        def _refresh_form_panel(self, m) -> None:
            """선택된 milestone 의 form 채우기. m=None 면 비활성."""
            self._suppress_signals = True
            try:
                if m is None:
                    self._fld_title.setText("")
                    self._fld_description.setText("")
                    self._lbl_horizon.setText("(선택 없음)")
                    self._fld_target_date.setText("")
                    self._combo_status.setCurrentIndex(0)
                    self._slider_progress.setValue(0)
                    self._lbl_progress_val.setText("0%")
                    self._lbl_progress_meta.setText("")
                    self._list_criteria.clear()
                    self._fld_notes.setText("")
                    self._lbl_meta.setText("")
                    self._detail_panel.setEnabled(False)
                    return

                self._detail_panel.setEnabled(True)
                self._fld_title.setText(m.title)
                self._fld_description.setPlainText(m.description)
                self._lbl_horizon.setText(_HORIZON_LABEL.get(m.horizon, m.horizon))
                self._fld_target_date.setText(m.target_date)
                # status combo
                for i in range(self._combo_status.count()):
                    if self._combo_status.itemData(i) == m.status:
                        self._combo_status.setCurrentIndex(i)
                        break
                pct = int(round(m.progress * 100))
                self._slider_progress.setValue(pct)
                self._lbl_progress_val.setText(f"{pct}%")
                if m.progress_is_manual:
                    self._lbl_progress_meta.setText("(수동 — children 평균 무시)")
                    self._lbl_progress_meta.setStyleSheet(
                        "color:#fbbf24; font-size:11px;")
                else:
                    if m.children_ids:
                        self._lbl_progress_meta.setText("(자동 — children 평균)")
                    else:
                        self._lbl_progress_meta.setText("(자동 — history 평가 시 갱신)")
                    self._lbl_progress_meta.setStyleSheet(
                        "color:#94a3b8; font-size:11px;")
                self._list_criteria.clear()
                for c in m.success_criteria:
                    self._list_criteria.addItem(QListWidgetItem(c))
                self._fld_notes.setPlainText(m.notes)
                meta_parts = [
                    f"id={m.id[:16]}",
                    f"source={m.source}",
                    f"created={m.started_at[:19]}" if m.started_at else "",
                    f"completed={m.completed_at[:19]}" if m.completed_at else "",
                ]
                self._lbl_meta.setText("  ·  ".join(p for p in meta_parts if p))
            finally:
                self._suppress_signals = False

        # ── 저장 (모든 폼 → milestone) ────────────────────────────
        def _on_form_save(self) -> None:
            if self._suppress_signals:
                return
            if not self._tree or not self._current_milestone_id:
                return
            try:
                _gt.update_milestone(
                    self._tree, self._current_milestone_id,
                    title=self._fld_title.text().strip(),
                    description=self._fld_description.toPlainText().strip(),
                    target_date=self._fld_target_date.text().strip(),
                    notes=self._fld_notes.toPlainText().strip(),
                )
                _gt.save_goal_tree(self._tree)
                self._refresh_current_item_text()
                self.tree_changed.emit()
            except Exception as e:
                print(f"[goal_tree_dialog] form save 실패 (graceful): {e}")

        def _refresh_current_item_text(self) -> None:
            """현재 선택된 tree item 의 text 만 갱신 (전체 rebuild 회피)."""
            items = self._tree_widget.selectedItems()
            if not items:
                return
            m = self._tree.get(self._current_milestone_id) if self._tree else None
            if m is None:
                return
            items[0].setText(0, _format_milestone_text(m))
            self._update_summary()

        # ── 진행도 slider ─────────────────────────────────────────
        def _on_progress_slider(self, value: int) -> None:
            self._lbl_progress_val.setText(f"{value}%")
            if self._suppress_signals:
                return
            if not self._tree or not self._current_milestone_id:
                return
            try:
                _gt.set_milestone_progress(
                    self._tree, self._current_milestone_id,
                    value / 100.0, manual=True,
                )
                _gt.save_goal_tree(self._tree)
                # meta 라벨 갱신
                self._lbl_progress_meta.setText("(수동 — children 평균 무시)")
                self._lbl_progress_meta.setStyleSheet(
                    "color:#fbbf24; font-size:11px;")
                self._refresh_current_item_text()
                self.tree_changed.emit()
            except Exception as e:
                print(f"[goal_tree_dialog] progress 저장 실패 (graceful): {e}")

        # ── 상태 변경 ─────────────────────────────────────────────
        def _on_status_changed(self, _idx: int) -> None:
            if self._suppress_signals:
                return
            if not self._tree or not self._current_milestone_id:
                return
            new_status = self._combo_status.currentData()
            if not new_status:
                return
            try:
                _gt.set_milestone_status(self._tree, self._current_milestone_id, new_status)
                _gt.save_goal_tree(self._tree)
                # form 다시 그리기 (progress·meta 자동 변경)
                m = self._tree.get(self._current_milestone_id)
                self._refresh_form_panel(m)
                self._refresh_current_item_text()
                self.tree_changed.emit()
            except Exception as e:
                print(f"[goal_tree_dialog] status 변경 실패 (graceful): {e}")

        # ── success_criteria 조작 ─────────────────────────────────
        def _on_criterion_add(self) -> None:
            if not self._tree or not self._current_milestone_id:
                return
            text, ok = QInputDialog.getText(self, "성공 기준 추가",
                                            "성공 기준 (예: '견적 발송 완료'):")
            if not ok or not text.strip():
                return
            m = self._tree.get(self._current_milestone_id)
            if m is None:
                return
            new_list = list(m.success_criteria) + [text.strip()]
            _gt.update_milestone(self._tree, m.id, success_criteria=new_list)
            _gt.save_goal_tree(self._tree)
            self._refresh_form_panel(self._tree.get(m.id))
            self.tree_changed.emit()

        def _on_criterion_del(self) -> None:
            if not self._tree or not self._current_milestone_id:
                return
            row = self._list_criteria.currentRow()
            if row < 0:
                return
            m = self._tree.get(self._current_milestone_id)
            if m is None or row >= len(m.success_criteria):
                return
            new_list = [c for i, c in enumerate(m.success_criteria) if i != row]
            _gt.update_milestone(self._tree, m.id, success_criteria=new_list)
            _gt.save_goal_tree(self._tree)
            self._refresh_form_panel(self._tree.get(m.id))
            self.tree_changed.emit()

        # ── milestone 추가/삭제 ──────────────────────────────────
        def _on_add_root_clicked(self) -> None:
            if self._tree is None:
                return
            title, ok = QInputDialog.getText(self, "최상위 milestone 추가",
                                             "제목:")
            if not ok or not title.strip():
                return
            m = _gt.add_milestone(self._tree, title.strip(),
                                  parent_id="", horizon="year")
            if m is None:
                return
            _gt.save_goal_tree(self._tree)
            self._current_milestone_id = m.id
            self._refresh_tree_widget()
            self.tree_changed.emit()

        def _on_add_child_clicked(self) -> None:
            if self._tree is None or not self._current_milestone_id:
                QMessageBox.information(self, "자식 추가",
                                        "먼저 부모로 사용할 milestone 을 선택하세요.")
                return
            parent = self._tree.get(self._current_milestone_id)
            if parent is None:
                return
            title, ok = QInputDialog.getText(self, "자식 milestone 추가",
                                             f"'{parent.title}' 아래 추가할 제목:")
            if not ok or not title.strip():
                return
            m = _gt.add_milestone(self._tree, title.strip(), parent_id=parent.id)
            if m is None:
                return
            _gt.save_goal_tree(self._tree)
            self._current_milestone_id = m.id
            self._refresh_tree_widget()
            self.tree_changed.emit()

        def _on_delete_clicked(self) -> None:
            if self._tree is None or not self._current_milestone_id:
                return
            m = self._tree.get(self._current_milestone_id)
            if m is None:
                return
            has_children = bool(m.children_ids)
            msg = f"'{m.title}' 을 삭제하시겠습니까?"
            cascade = False
            if has_children:
                ret = QMessageBox.question(
                    self, "삭제 확인",
                    msg + f"\n\n자식 milestone {len(m.children_ids)}개도 함께 삭제할까요?\n"
                          "(예 = 자식까지 cascade·아니오 = 자식은 orphan 으로 보존)",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                )
                if ret == QMessageBox.Cancel:
                    return
                cascade = (ret == QMessageBox.Yes)
            else:
                ret = QMessageBox.question(
                    self, "삭제 확인", msg,
                    QMessageBox.Yes | QMessageBox.No,
                )
                if ret != QMessageBox.Yes:
                    return
            n = _gt.delete_milestone(self._tree, m.id, cascade=cascade)
            _gt.save_goal_tree(self._tree)
            self._current_milestone_id = ""
            self._refresh_tree_widget()
            self.tree_changed.emit()
            self._lbl_summary.setText(self._lbl_summary.text() + f"   (삭제됨: {n})")

        # ── 재계산 / history 평가 ────────────────────────────────
        def _on_recompute_clicked(self) -> None:
            if self._tree is None:
                return
            try:
                n = _gt.recompute_progress(self._tree)
                _gt.save_goal_tree(self._tree)
                self._refresh_tree_widget()
                # 현재 선택 form 도 다시
                if self._current_milestone_id:
                    self._refresh_form_panel(self._tree.get(self._current_milestone_id))
                self.tree_changed.emit()
                QMessageBox.information(self, "재계산",
                                        f"내부 milestone {n}개 progress 갱신됨.")
            except Exception as e:
                QMessageBox.warning(self, "재계산 실패", str(e))

        def _on_decompose_clicked(self) -> None:
            """[Phase 9-E] LLM 자동 분해 — 선택된 milestone (또는 root) 을 N 개로 분해.

            흐름:
              1. 부모 milestone 결정 (selection 있으면 그것·없으면 root)
              2. n_children QInputDialog (default 4)
              3. asyncio.run 으로 decompose_goal_with_llm_async 호출
              4. 결과 미리보기 다이얼로그 (QMessageBox) — title list
              5. confirm 시 apply_decomposition → tree refresh
            """
            if self._tree is None:
                return
            # 부모 결정
            parent = None
            if self._current_milestone_id:
                parent = self._tree.get(self._current_milestone_id)
            if parent is None:
                parent = self._tree.root()
            if parent is None:
                QMessageBox.information(self, "🪄 LLM 자동 분해",
                                        "분해할 부모 milestone 이 없습니다. "
                                        "먼저 root 를 만드세요.")
                return

            # n_children 입력
            try:
                from PySide6.QtWidgets import QInputDialog
            except Exception:
                return
            from eidos_goal_tree import _HORIZON_CHILDREN
            default_target = _HORIZON_CHILDREN.get(parent.horizon, "task")
            default_n = {"quarter": 4, "month": 3, "week": 4, "task": 5}.get(
                default_target, 4)
            n, ok = QInputDialog.getInt(
                self, "🪄 LLM 자동 분해",
                f"부모: '{parent.title}' ({parent.horizon})\n"
                f"생성 horizon: {default_target}\n"
                f"몇 개의 하위 milestone 으로 분해할까요?",
                default_n, 1, 12, 1,
            )
            if not ok:
                return

            # async 호출 — sync wrapper
            try:
                import asyncio
                from eidos_goal_tree import decompose_goal_with_llm_async
            except Exception as e:
                QMessageBox.warning(self, "🪄 LLM 자동 분해 실패",
                                    f"import 실패: {e}")
                return

            # 진행 표시 — 길어질 수 있으니 setEnabled(False)
            self._btn_decompose.setEnabled(False)
            self._btn_decompose.setText("⏳ 분해 중...")
            try:
                try:
                    proposed = asyncio.run(decompose_goal_with_llm_async(
                        parent.title,
                        parent_horizon=parent.horizon,
                        target_horizon=default_target,
                        n_children=n,
                        horizon_end=self._tree.horizon_end or parent.target_date or "",
                        context=parent.description or "",
                    ))
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    try:
                        proposed = loop.run_until_complete(
                            decompose_goal_with_llm_async(
                                parent.title,
                                parent_horizon=parent.horizon,
                                target_horizon=default_target,
                                n_children=n,
                                horizon_end=self._tree.horizon_end or parent.target_date or "",
                                context=parent.description or "",
                            )
                        )
                    finally:
                        loop.close()
            except Exception as e:
                QMessageBox.warning(self, "🪄 LLM 분해 실패", str(e))
                proposed = []
            finally:
                self._btn_decompose.setEnabled(True)
                self._btn_decompose.setText("🪄 LLM 자동 분해")

            if not proposed:
                QMessageBox.information(
                    self, "🪄 LLM 자동 분해",
                    "LLM 이 빈 결과 반환·또는 호출 실패. "
                    "수동으로 ➕ 자식 추가 사용 권장.",
                )
                return

            # 미리보기 다이얼로그
            preview_lines = [f"부모: '{parent.title}' 아래 {len(proposed)} 개 제안:"]
            for i, m in enumerate(proposed, 1):
                preview_lines.append(f"\n{i}. {m.title}")
                if m.target_date:
                    preview_lines.append(f"   📅 {m.target_date}")
                if m.description:
                    preview_lines.append(f"   {m.description[:100]}")
                if m.success_criteria:
                    preview_lines.append(
                        f"   기준: {', '.join(m.success_criteria[:3])}"
                    )
            preview_text = "\n".join(preview_lines)
            ret = QMessageBox.question(
                self, "🪄 LLM 자동 분해 — 미리보기",
                preview_text + "\n\n이 milestone 들을 추가할까요?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if ret != QMessageBox.Yes:
                return

            # apply
            try:
                from eidos_goal_tree import apply_decomposition
                registered = apply_decomposition(self._tree, parent.id, proposed)
            except Exception as e:
                QMessageBox.warning(self, "🪄 apply 실패", str(e))
                return

            if not registered:
                QMessageBox.warning(self, "🪄 apply 실패",
                                    "등록된 milestone 이 0건입니다.")
                return
            _gt_save = self._tree   # noqa — readability
            try:
                from eidos_goal_tree import save_goal_tree
                save_goal_tree(self._tree)
            except Exception:
                pass
            self._refresh_tree_widget()
            self.tree_changed.emit()
            QMessageBox.information(
                self, "🪄 LLM 자동 분해 완료",
                f"{len(registered)} 개 milestone 추가됨.",
            )

        def _on_evaluate_clicked(self) -> None:
            """stage history 의 키워드로 leaf 매칭 → progress 자동 추정."""
            if self._tree is None:
                return
            try:
                events = _ass.read_history(self._stage.id, limit=200)
            except Exception as e:
                QMessageBox.warning(self, "history 평가 실패",
                                    f"history 로드 실패: {e}")
                return
            if not events:
                QMessageBox.information(self, "history 평가",
                                        "이 stage 의 history 가 비어있습니다. "
                                        "tick 을 한번 돌린 뒤 다시 시도하세요.")
                return
            try:
                changes = _gt.evaluate_progress_from_history(self._tree, events)
                n_internal = _gt.recompute_progress(self._tree)
                _gt.save_goal_tree(self._tree)
                self._refresh_tree_widget()
                if self._current_milestone_id:
                    self._refresh_form_panel(self._tree.get(self._current_milestone_id))
                self.tree_changed.emit()
                QMessageBox.information(
                    self, "history 평가 완료",
                    f"leaf milestone {len(changes)}개 progress 추정 갱신.\n"
                    f"internal milestone {n_internal}개 재계산.",
                )
            except Exception as e:
                QMessageBox.warning(self, "history 평가 실패", str(e))

        # ── 닫기 hook ────────────────────────────────────────────
        def closeEvent(self, event):
            # 단일 인스턴스 가드에서 해제
            try:
                GoalTreeDialog._OPEN_INSTANCES.pop(self._stage.id, None)
            except Exception:
                pass
            super().closeEvent(event)


    def open_goal_tree_dialog(stage, parent=None) -> Optional["GoalTreeDialog"]:
        """단일 인스턴스 가드 + 다이얼로그 표시. 이미 열려있으면 raise/activate.

        Returns: GoalTreeDialog 인스턴스 또는 None (실패 시).
        """
        if stage is None or not stage.id:
            return None
        sid = stage.id
        try:
            existing = GoalTreeDialog._OPEN_INSTANCES.get(sid)
            if existing is not None and existing.isVisible():
                existing.raise_()
                existing.activateWindow()
                return existing
        except Exception:
            pass
        try:
            dlg = GoalTreeDialog(stage, parent=parent)
            GoalTreeDialog._OPEN_INSTANCES[sid] = dlg
            dlg.show()
            return dlg
        except Exception as e:
            print(f"[goal_tree_dialog] open 실패 (graceful): {e}")
            return None


else:
    # PySide6 또는 의존 모듈 미설치 시 — stub. import 실패 graceful.
    class GoalTreeDialog:  # type: ignore[no-redef]
        pass

    def open_goal_tree_dialog(stage, parent=None):  # type: ignore[no-redef]
        print("[goal_tree_dialog] PySide6 또는 goal_tree 모듈 없음 — 다이얼로그 표시 불가")
        return None


__all__ = ["GoalTreeDialog", "open_goal_tree_dialog"]
