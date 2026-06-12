# eidos_prompt_canvas.py
# EIDOS Prompt Canvas – Agent Composer (Full Implementation)
# PySide6 required.

from __future__ import annotations
import asyncio
import concurrent.futures
import json
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any, Literal, Callable

from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QObject, QTimer, QThread
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPainterPath, QPainterPathStroker
import os
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QGraphicsView, QGraphicsScene, QGraphicsObject,
    QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QLineEdit, QPushButton, QComboBox,
    QSplitter, QFileDialog, QMessageBox, QToolBar, QStatusBar, QInputDialog, QApplication, QWidget,
    QSpinBox, QGroupBox, QProgressDialog
)

# ── 언어 헬퍼 ──────────────────────────────────────────────────────────
def _canvas_lang() -> str:
    """현재 앱 언어 반환."""
    try:
        import eidos_chat_gui as _g
        return getattr(_g, 'app_settings', {}).get('language', 'ko')
    except Exception:
        try:
            import json as _j, os as _o
            s = _j.load(open(_o.path.join('eidos_files', 'eidos_settings.json'), encoding='utf-8'))
            return s.get('language', 'ko')
        except Exception:
            return 'ko'

_CANVAS_TEXT = {
    'ko': {
        'inspector_title': '프롬프트 정보 패널',
        'inspector_none':  '선택: (없음)',
        'lbl_title':       '제목',
        'ph_title':        '예: 문제 분해 / 조건 정리 / 검증 등',
        'lbl_tool':        '도구 선택',
        'lbl_prompt':      '프롬프트',
        'lbl_prompt_desc': '설명 (선택)',
        'ph_prompt_desc':  '이 분기 노드에 대한 메모를 적으세요. (선택사항)',
        'ph_prompt':       '여기에 프롬프트를 작성하세요.',
        'lbl_condition':   '분기 조건 (if 노드)',
        'ph_condition':    '예: "오류" in prev   /   len(prev) > 100',
        'lbl_loop':        '루프 설정',
        'lbl_loop_count':  '반복 횟수',
        'lbl_body_nodes':  '바디 노드 (포함할 노드 선택)',
        'lbl_node_pick':   '노드 직접 선택 (콤보에서 골라 추가)',
        'btn_loop_add_sel':'✚ 캔버스 선택 추가',
        'btn_loop_add_pick':'✚ 목록에서 추가',
        'btn_loop_clear':  '바디 초기화',
        'lbl_trigger':     '자동 트리거',
        'lbl_weekdays':    '반복 요일 (쉼표로 구분, 0=월~6=일)',
        'ph_weekdays':     '예: 0,2,4  (월·수·금)',
        'lbl_hour_start':  '시작 시(0-23)',
        'lbl_hour_end':    '종료 시(0-23)',
        'lbl_sm_key':      'self_model 키 (예: stress, energy)',
        'ph_sm_key':       'eidos_worker.self_model 의 키',
        'lbl_operator':    '연산자',
        'lbl_threshold':   '임계값',
        'ph_threshold':    '예: 0.7',
        'btn_save_trigger':'트리거 설정 저장',
        'lbl_exit_cond':   '조건 종료 (선택)',
        'lbl_max_iter':    '최대 반복 (안전장치, 0=무제한)',
        'lbl_mem_mode':    '루프 메모리 모드',
        'lbl_compress':    '압축 주기 (N회마다, llm_compress 전용)',
        'lbl_last_out':    '마지막 출력(읽기 전용)',
        'btn_apply':       '적용(즉시 저장)',
        'btn_run_here':    '여기서부터 실행',
        'lbl_goal':        '🎯 에이전트 목표 (Agent Goal)',
        'ph_goal':         '예: 코드 품질을 90점 이상으로 개선한다',
        'lbl_goal_inject': '목표 주입 방식',
        'lbl_dag_style':   'DAG 생성 스타일 (🗺️ 버튼)',
        'lbl_progress':    '진행도: 0%',
        'lbl_run_count':   '실행 횟수: 0',
        'btn_goal_save':   '목표 저장',
        'hint':            '팁: 노드를 클릭하면 이 패널에 정보가 표시됩니다. 편집은 자동 저장됩니다.',
        'btn_add_prompt':  '➕ 프롬프트 노드 추가',
        'btn_add_tool':    '🔧 툴 노드 추가',
        'btn_add_if':      '🔀 조건 분기 추가',
        'btn_add_loop':    '🔁 루프 노드 추가',
        'lbl_start_input': '  시작 입력: ',
        'ph_start_input':  '시작 입력...',
        'btn_run':         '▶ 전체 실행',
        'btn_save':        '💾 저장',
        'btn_load':        '📂 불러오기',
        'btn_clear':       '🧹 초기화',
        'btn_agent':       '🧠 에이전트 생성/등록',
        'btn_plan':        '🗺️ 목표로 DAG 생성',
        'selected':        '선택: ',
        'selected_multi':  '개)',
    },
    'en': {
        'inspector_title': 'Prompt Info Panel',
        'inspector_none':  'Selected: (none)',
        'lbl_title':       'Title',
        'ph_title':        'e.g. Decompose / Condition / Validate',
        'lbl_tool':        'Select Tool',
        'lbl_prompt':      'Prompt',
        'lbl_prompt_desc': 'Description (optional)',
        'ph_prompt_desc':  'Add a memo for this branch node. (optional)',
        'ph_prompt':       'Write your prompt here.',
        'lbl_condition':   'Branch Condition (if node)',
        'ph_condition':    'e.g. "error" in prev   /   len(prev) > 100',
        'lbl_loop':        'Loop Settings',
        'lbl_loop_count':  'Iterations',
        'lbl_body_nodes':  'Body Nodes (select nodes to include)',
        'lbl_node_pick':   'Pick node from list',
        'btn_loop_add_sel':'✚ Add Canvas Selection',
        'btn_loop_add_pick':'✚ Add from List',
        'btn_loop_clear':  'Clear Body',
        'lbl_trigger':     'Auto Trigger',
        'lbl_weekdays':    'Weekdays (comma-separated, 0=Mon~6=Sun)',
        'ph_weekdays':     'e.g. 0,2,4  (Mon·Wed·Fri)',
        'lbl_hour_start':  'Start hour (0-23)',
        'lbl_hour_end':    'End hour (0-23)',
        'lbl_sm_key':      'self_model key (e.g. stress, energy)',
        'ph_sm_key':       'key in eidos_worker.self_model',
        'lbl_operator':    'Operator',
        'lbl_threshold':   'Threshold',
        'ph_threshold':    'e.g. 0.7',
        'btn_save_trigger':'Save Trigger Settings',
        'lbl_exit_cond':   'Exit Condition (optional)',
        'lbl_max_iter':    'Max Iterations (0 = unlimited)',
        'lbl_mem_mode':    'Loop Memory Mode',
        'lbl_compress':    'Compress every N iterations (llm_compress only)',
        'lbl_last_out':    'Last Output (read only)',
        'btn_apply':       'Apply (Save Now)',
        'btn_run_here':    'Run From Here',
        'lbl_goal':        '🎯 Agent Goal',
        'ph_goal':         'e.g. Improve code quality to 90+ score',
        'lbl_goal_inject': 'Goal Injection Mode',
        'lbl_dag_style':   'DAG Generation Style (🗺️ button)',
        'lbl_progress':    'Progress: 0%',
        'lbl_run_count':   'Runs: 0',
        'btn_goal_save':   'Save Goal',
        'hint':            'Tip: Click a node to view its info here. Edits are auto-saved.',
        'btn_add_prompt':  '➕ Add Prompt Node',
        'btn_add_tool':    '🔧 Add Tool Node',
        'btn_add_if':      '🔀 Add Branch Node',
        'btn_add_loop':    '🔁 Add Loop Node',
        'lbl_start_input': '  Start Input: ',
        'ph_start_input':  'Start input...',
        'btn_run':         '▶ Run All',
        'btn_save':        '💾 Save',
        'btn_load':        '📂 Load',
        'btn_clear':       '🧹 Clear',
        'btn_agent':       '🧠 Create/Register Agent',
        'btn_plan':        '🗺️ Generate DAG from Goal',
        'selected':        'Selected: ',
        'selected_multi':  ' nodes)',
    },
    'ja': {
        'inspector_title': 'プロンプト情報パネル',
        'inspector_none':  '選択: (なし)',
        'lbl_title':       'タイトル',
        'ph_title':        '例: 分解 / 条件 / 検証',
        'lbl_tool':        'ツール選択',
        'lbl_prompt':      'プロンプト',
        'lbl_prompt_desc': '説明 (任意)',
        'ph_prompt_desc':  'このブランチノードのメモ (任意)',
        'ph_prompt':       'ここにプロンプトを入力してください。',
        'lbl_condition':   '分岐条件 (ifノード)',
        'ph_condition':    '例: "エラー" in prev   /   len(prev) > 100',
        'lbl_loop':        'ループ設定',
        'lbl_loop_count':  '繰り返し回数',
        'lbl_body_nodes':  'ボディノード (含むノードを選択)',
        'lbl_node_pick':   'リストからノードを選択',
        'btn_loop_add_sel':'✚ キャンバス選択を追加',
        'btn_loop_add_pick':'✚ リストから追加',
        'btn_loop_clear':  'ボディをクリア',
        'lbl_trigger':     '自動トリガー',
        'lbl_weekdays':    '繰り返し曜日 (カンマ区切り, 0=月~6=日)',
        'ph_weekdays':     '例: 0,2,4  (月・水・金)',
        'lbl_hour_start':  '開始時(0-23)',
        'lbl_hour_end':    '終了時(0-23)',
        'lbl_sm_key':      'self_modelキー (例: stress, energy)',
        'ph_sm_key':       'eidos_worker.self_model のキー',
        'lbl_operator':    '演算子',
        'lbl_threshold':   'しきい値',
        'ph_threshold':    '例: 0.7',
        'btn_save_trigger':'トリガー設定を保存',
        'lbl_exit_cond':   '終了条件 (任意)',
        'lbl_max_iter':    '最大繰り返し (0=無制限)',
        'lbl_mem_mode':    'ループメモリモード',
        'lbl_compress':    '圧縮周期 (N回ごと, llm_compress専用)',
        'lbl_last_out':    '最後の出力 (読み取り専用)',
        'btn_apply':       '適用 (即時保存)',
        'btn_run_here':    'ここから実行',
        'lbl_goal':        '🎯 エージェント目標',
        'ph_goal':         '例: コード品質を90点以上に改善する',
        'lbl_goal_inject': '目標注入モード',
        'lbl_dag_style':   'DAG生成スタイル (🗺️ボタン)',
        'lbl_progress':    '進行度: 0%',
        'lbl_run_count':   '実行回数: 0',
        'btn_goal_save':   '目標を保存',
        'hint':            'ヒント: ノードをクリックすると情報が表示されます。編集は自動保存されます。',
        'btn_add_prompt':  '➕ プロンプトノード追加',
        'btn_add_tool':    '🔧 ツールノード追加',
        'btn_add_if':      '🔀 条件分岐追加',
        'btn_add_loop':    '🔁 ループノード追加',
        'lbl_start_input': '  開始入力: ',
        'ph_start_input':  '開始入力...',
        'btn_run':         '▶ 全体実行',
        'btn_save':        '💾 保存',
        'btn_load':        '📂 読み込み',
        'btn_clear':       '🧹 初期化',
        'btn_agent':       '🧠 エージェント生成/登録',
        'btn_plan':        '🗺️ 目標からDAG生成',
        'selected':        '選択: ',
        'selected_multi':  '個)',
    },
}

def _ct(key: str) -> str:
    """Canvas text — 현재 언어로 반환."""
    lang = _canvas_lang()
    return _CANVAS_TEXT.get(lang, _CANVAS_TEXT['en']).get(key, _CANVAS_TEXT['ko'].get(key, key))


# =========================
# ContextStore  (개선 #1)
# =========================

@dataclass
class ContextEntry:
    """단일 노드 실행 기록."""
    node_id:   str
    title:     str
    output:    str
    ok:        bool
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


class ContextStore:
    """
    구조화된 실행 컨텍스트.

    - history     : 전체 실행 이력 (순서 보장)
    - key_outputs : node_id → 최신 출력 str (빠른 참조)
    - _summary    : 길이 초과 시 LLM-롤링 요약 (옵션)

    {{context}}  → summary + 최근 N개 항목 압축 텍스트
    {{key:<id>}} → key_outputs[id] 직접 참조  ← _compile_prompt에서 처리
    """
    MAX_INLINE = 6          # summary 없이 전문 주입할 최대 항목 수
    SUMMARY_TOKENS = 600    # 요약 목표 토큰 수 (글자 수 × 0.6 근사)

    def __init__(self):
        self.history:     List[ContextEntry]   = []
        self.key_outputs: Dict[str, str]       = {}
        self._summary:    str                  = ""

    # ── 기록 ─────────────────────────────────────────────────────
    def record(self, node_id: str, title: str, output: str, ok: bool):
        entry = ContextEntry(node_id=node_id, title=title, output=output, ok=ok)
        self.history.append(entry)
        self.key_outputs[node_id] = output

    # ── {{context}} 렌더링 ────────────────────────────────────────
    def render(self) -> str:
        """
        짧으면 전체 기록, 길면 [요약 + 최근 N개]를 반환한다.
        요약은 외부에서 update_summary()로 주입받는다.
        """
        if len(self.history) <= self.MAX_INLINE:
            parts = [f"[{e.title}]\n{e.output}" for e in self.history]
            return "\n\n".join(parts)

        recent = self.history[-self.MAX_INLINE:]
        recent_text = "\n\n".join(f"[{e.title}]\n{e.output}" for e in recent)

        if self._summary:
            return f"[이전 실행 요약]\n{self._summary}\n\n[최근 실행]\n{recent_text}"
        # 요약 미설정 시 전체 텍스트 (토큰 경고 방지를 위해 각 항목 트런케이트)
        all_parts = []
        for e in self.history:
            out = e.output if len(e.output) <= 800 else e.output[:800] + "…(생략)"
            all_parts.append(f"[{e.title}]\n{out}")
        return "\n\n".join(all_parts)

    def update_summary(self, summary: str):
        self._summary = summary

    def get(self, node_id: str, default: str = "") -> str:
        return self.key_outputs.get(node_id, default)

    def last_output(self) -> str:
        return self.history[-1].output if self.history else ""

    def to_dict(self) -> dict:
        return {
            "history":     [asdict(e) for e in self.history],
            "key_outputs": self.key_outputs,
            "summary":     self._summary,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ContextStore":
        cs = cls()
        for raw in d.get("history", []):
            cs.history.append(ContextEntry(**raw))
        cs.key_outputs = d.get("key_outputs", {})
        cs._summary    = d.get("summary", "")
        return cs


# =========================
# ToolSchema / ToolResult  (개선 #4)
# =========================

@dataclass
class ToolParam:
    name:        str
    type:        str              # "str" | "int" | "float" | "bool" | "json"
    description: str = ""
    required:    bool = True
    default:     Any  = None


@dataclass
class ToolSchema:
    """
    Tool 노드의 입출력 스키마.
    각 tool_name에 대한 스키마를 TOOL_REGISTRY에 등록한다.
    """
    name:        str
    description: str
    params:      List[ToolParam]
    output_type: str = "str"      # "str" | "json" | "number"


@dataclass
class ToolResult:
    """Tool 실행 결과 구조체."""
    tool_name:  str
    ok:         bool
    output:     str
    output_raw: Any  = None       # 원본 파이썬 객체 (json dict, number 등)
    error:      str  = ""
    duration_ms: float = 0.0

    def to_context_str(self) -> str:
        if not self.ok:
            return f"[도구 오류: {self.tool_name}] {self.error}"
        return self.output


# Tool 스키마 레지스트리 (확장 가능)
TOOL_REGISTRY: Dict[str, ToolSchema] = {
    "perform_web_search": ToolSchema(
        name="perform_web_search", description="웹 검색",
        params=[ToolParam("query", "str", "검색어")],
    ),
    "read_file": ToolSchema(
        name="read_file", description="파일 읽기",
        params=[ToolParam("path", "str", "파일 경로")],
    ),
    "write_file": ToolSchema(
        name="write_file", description="파일 쓰기",
        params=[
            ToolParam("path", "str", "저장할 파일 경로 (예: output/result.py)"),
            ToolParam("content", "str", "저장할 내용"),
        ],
    ),
    "execute_python_file": ToolSchema(
        name="execute_python_file", description="Python 파일 실행",
        params=[ToolParam("path", "str")],
    ),
    "calculate_math": ToolSchema(
        name="calculate_math", description="수식 계산",
        params=[ToolParam("expression", "str", "수식")],
        output_type="number",
    ),
    "gcal_list_events": ToolSchema(
        name="gcal_list_events", description="Google Calendar 일정 조회",
        params=[ToolParam("days_ahead", "int", "조회 일수", default=7)],
        output_type="json",
    ),
    "gcal_create_event": ToolSchema(
        name="gcal_create_event", description="Google Calendar 일정 생성",
        params=[
            ToolParam("title", "str"), ToolParam("start", "str"),
            ToolParam("end", "str"),   ToolParam("description", "str", required=False, default=""),
        ],
    ),
    # ── 4-엔진 툴 (DC/AC/CC/RC) — 사용자 자유 조립용 ──
    # 사용자가 캔버스에서 4-엔진 노드를 직접 연결해 도메인별 컴포지션 실험.
    # 각 엔진은 lazy 인스턴스화, LLM plugin default + 휴리스틱 fallback (eidos_canvas_4engine_tools.py).
    "dc_decompose": ToolSchema(
        name="dc_decompose", description="DC: 의미 단위 task 분해 (subtask list)",
        params=[ToolParam("goal", "str", "분해할 목표 (raw 또는 상류 JSON)")],
        output_type="json",
    ),
    "ac_analogize": ToolSchema(
        name="ac_analogize", description="AC: cross-domain 유비 생성 (다른 도메인 빌려옴)",
        params=[ToolParam("subtasks_or_goal", "str", "분해 결과(JSON) 또는 raw 목표")],
        output_type="json",
    ),
    "cc_critique": ToolSchema(
        name="cc_critique", description="CC: 선택/검증 비판 (selection: N→통과한 것, verification: 단일 산출물 평가)",
        params=[
            ToolParam("ideas", "str", "평가할 아이디어 (JSON list 또는 raw)"),
            ToolParam("mode", "str", "selection|verification", required=False, default="selection"),
        ],
        output_type="json",
    ),
    "rc_reason": ToolSchema(
        name="rc_reason", description="RC: LLM-only 반복 자기비판 추론 (외부 효과 없음)",
        params=[
            ToolParam("goal", "str", "추론 목표"),
            ToolParam("max_cycles", "int", "최대 cycle (1~10)", required=False, default=3),
        ],
        output_type="json",
    ),
    # ── 워크플로우 지능 전용 툴 ──
    "friction_score": ToolSchema(
        name="friction_score", description="워크플로우 단계별 5축 friction 점수화 (반복/귀찮음/전환/비청구/판단피로)",
        params=[ToolParam("subtasks_or_workflow", "str", "subtask list (JSON) 또는 줄바꿈 텍스트")],
        output_type="json",
    ),
    "web_search_domain": ToolSchema(
        name="web_search_domain", description="도메인 키워드 → 5 query 자동 검색 (ground truth 수집, 환각 차단)",
        params=[ToolParam("domain", "str", "타깃 도메인 (예: '정형외과 접수처 직원')")],
        output_type="json",
    ),
    # ── 자율 탐색 (bounded ReAct, ActionDispatcher 싱글톤 무관) ──
    # 사용자가 링크/경로를 안 줘도 스스로 검색→판단→(브라우저)접속+본문→반복→합성.
    # execution_module.autonomous_research 가 perform_web_search/navigate/LLM 만
    # 순수 조합 (미션 디스패처 비사용 — 재진입/승인게이트 위험 회피).
    "autonomous_research": ToolSchema(
        name="autonomous_research",
        description="자율 탐색: 목표만 주면 웹검색+LLM판단+(가능시)페이지접속을 상한 내 반복해 결과 도출 (링크 불필요)",
        params=[
            ToolParam("goal", "str", "탐색 목표 (자연어, 예: '2026 5급공채 공고 일정')"),
            ToolParam("max_iterations", "int", "최대 반복(1~6, 기본 3)", required=False, default=3),
            ToolParam("use_browser", "bool", "브라우저 가능 시 본문 추출", required=False, default=True),
        ],
        output_type="str",
    ),
    # ── 내장 브라우저 탐색 (ActionDispatcher 와 동일 execution_module 함수 위임) ──
    # 메인 채팅창의 InternalBrowser (QWebEngineView) 가 set_browser_callback 으로
    # 등록한 콜백을 통해 실제 page navigate/JS/text 추출 수행. worker.run_tool 이
    # execution_module 의 async 함수를 동기 호출.
    "navigate_and_wait": ToolSchema(
        name="navigate_and_wait",
        description="브라우저: URL 이동 + loadFinished 대기 (실제 GUI 브라우저에서 페이지 로드)",
        params=[
            ToolParam("url", "str", "이동할 URL (예: https://naver.com)"),
            ToolParam("timeout", "float", "로드 대기 timeout(초)", required=False, default=15.0),
        ],
    ),
    "get_browser_visible_text": ToolSchema(
        name="get_browser_visible_text",
        description="브라우저: 현재 페이지의 body.innerText 추출 (same-origin iframe 포함)",
        params=[
            ToolParam("timeout", "float", "JS 실행 timeout(초)", required=False, default=6.0),
        ],
    ),
    "browser_run_script": ToolSchema(
        name="browser_run_script",
        description="브라우저: JavaScript 실행 + 결과 반환 (document.title, querySelector 등)",
        params=[ToolParam("script", "str", "실행할 JS 코드 (반환값이 결과로 전달됨)")],
    ),
    "grab_browser_screenshot": ToolSchema(
        name="grab_browser_screenshot",
        description="브라우저: 현재 페이지 스크린샷 저장 + 경로 반환 (vision input 용)",
        params=[
            ToolParam("save_path", "str", "저장 경로 (비우면 자동)", required=False, default=""),
            ToolParam("timeout", "float", "캡쳐 timeout(초)", required=False, default=6.0),
        ],
    ),
}


def validate_tool_input(tool_name: str, input_str: str) -> Tuple[bool, str, dict]:
    """
    입력 문자열을 스키마에 맞게 파싱 및 검증.
    Returns: (is_valid, error_message, parsed_params)
    입력이 JSON이면 파싱, 아니면 필수 파라미터 1개면 그대로 주입.
    """
    schema = TOOL_REGISTRY.get(tool_name)
    if schema is None:
        return True, "", {"input": input_str}   # 미등록 툴은 통과

    parsed: dict = {}
    # JSON 입력 시도
    try:
        parsed = json.loads(input_str)
        if not isinstance(parsed, dict):
            parsed = {}
    except (json.JSONDecodeError, ValueError):
        parsed = {}

    # [2026-05-07 fix] JSON parse 성공해서 dict 됐지만 schema 의 어떤 파라미터 키와도
    # 매칭 안 되면 빈 dict 처럼 취급 → fallback 분기로 떨어뜨림.
    # 케이스: 상류 노드 (dc_decompose) 출력 = `{"ok": true, "subtasks": [...]}` 가
    # 다음 노드 (friction_score) 의 input_str 으로 흘러갈 때, schema 키
    # `subtasks_or_workflow` 와 매칭 0 → fallback 으로 input_str 자체를 단일 필수
    # 파라미터에 주입해야 _tool_xxx 의 _parse_upstream 이 다시 dict 로 파싱.
    if parsed:
        param_names = {p.name for p in schema.params}
        if not any(k in param_names for k in parsed.keys()):
            parsed = {}

    # JSON 파싱 실패 또는 schema 키 매칭 0 → 대체 파싱 시도
    if not parsed:
        stripped = input_str.strip()
        required = [p for p in schema.params if p.required]

        if stripped:
            # 1) key=value 또는 key: value 멀티라인 파싱 시도
            kv_parsed: dict = {}
            param_names = {p.name for p in schema.params}
            for line in stripped.splitlines():
                line = line.strip()
                if not line:
                    continue
                for sep in ("=", ":"):
                    if sep in line:
                        k, _, v = line.partition(sep)
                        k = k.strip()
                        if k in param_names:
                            kv_parsed[k] = v.strip()
                            break
            if kv_parsed and all(p.name in kv_parsed for p in required):
                parsed = kv_parsed

            # 2) 필수 파라미터가 1개면 입력값 그대로 주입 (기존 동작 유지)
            elif len(required) == 1:
                parsed = {required[0].name: stripped}

            # 3) write_file 특별 처리: 자연어 출력에서 path/content 추출 시도
            elif tool_name == "write_file" and "path" not in kv_parsed:
                # 코드블록 있으면 content로 추출
                import re as _re
                code_match = _re.search(r'```(?:\w+)?\n(.*?)```', stripped, _re.DOTALL)
                content_body = code_match.group(1).strip() if code_match else stripped

                # 파일명 추출 시도: "저장", "파일명", ".py", ".json" 등
                path_match = _re.search(r'[\w/\\]+\.\w+', stripped)
                guessed_path = path_match.group(0) if path_match else "output/result.py"

                parsed = {"path": guessed_path, "content": content_body}

            # 4) 그 외 — 구조 불명 입력은 "input" 키로 보존
            else:
                parsed = {"input": stripped}

    # 기본값 채우기
    for p in schema.params:
        if p.name not in parsed and p.default is not None:
            parsed[p.name] = p.default

    # 필수 파라미터 누락 체크
    missing = [p.name for p in schema.params if p.required and p.name not in parsed]
    if missing:
        return False, f"필수 파라미터 누락: {missing}", parsed

    return True, "", parsed


# =========================
# AgentState  (개선 #5)
# =========================

@dataclass
class AgentState:
    """
    에이전트 실행 간 영속 상태.
    PromptCanvasWindow에 1개 인스턴스가 유지된다.

    goal          : 현재 목표 (사용자 설정, 한 줄 요약)
    goal_prompt   : goal을 프롬프트에 주입하는 방식
                    "prefix"  — 모든 노드 프롬프트 앞에 [목표] 블록 삽입
                    "var"     — {{goal}} 변수로만 주입 (명시적)
                    "none"    — 주입 안 함 (기본)
    progress      : 0.0 ~ 1.0
    variables     : 에이전트 커스텀 변수 dict ({{var:<key>}} 참조)
    run_count     : 전체 실행 횟수
    last_run_ts   : 마지막 실행 Unix timestamp
    memory        : 에이전트가 스스로 저장한 메모 (키-값)
    """
    goal:         str   = ""
    goal_prompt:  str   = "none"    # "prefix" | "var" | "none"
    planner_style: str  = "sequential"  # "sequential" | "parallel" | "loop" | "auto"
    progress:     float = 0.0
    variables:    Dict[str, str]  = field(default_factory=dict)
    run_count:    int   = 0
    last_run_ts:  float = 0.0
    memory:       Dict[str, str]  = field(default_factory=dict)
    # 마지막 실행의 ContextStore 직렬화 (재시작 시 복원용)
    last_context: dict  = field(default_factory=dict)

    def tick(self):
        self.run_count  += 1
        self.last_run_ts = time.time()

    def set_var(self, key: str, value: str):
        self.variables[key] = value

    def get_var(self, key: str, default: str = "") -> str:
        return self.variables.get(key, default)

    def goal_block(self) -> str:
        """goal이 있을 때 프롬프트 앞에 삽입할 블록."""
        if not self.goal:
            return ""
        return f"[에이전트 목표]\n{self.goal}\n\n"

    def inject_goal(self, prompt: str) -> str:
        """
        goal_prompt 설정에 따라 프롬프트에 goal을 주입한다.
        'prefix' → 앞에 [에이전트 목표] 블록 삽입
        'var'    → {{goal}} 치환만 (미치환 시 그대로)
        'none'   → 변경 없음
        """
        if not self.goal:
            return prompt
        if self.goal_prompt == "prefix":
            return self.goal_block() + prompt
        if self.goal_prompt == "var":
            return prompt.replace("{{goal}}", self.goal).replace("{goal}", self.goal)
        return prompt

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "AgentState":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        # 하위호환: 구버전에 없는 필드 기본값 처리
        d.setdefault("goal_prompt", "none")
        d.setdefault("planner_style", "sequential")
        # 알 수 없는 필드 제거 (버전 불일치 시 TypeError 방지)
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        d = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**d)


# =========================
# Data Model
# =========================

NodeType = Literal["prompt", "tool", "if", "loop"]

@dataclass
class PromptNodeData:
    node_id: str
    title: str
    prompt: str
    x: float
    y: float
    node_type: NodeType = "prompt"
    condition: str = ""
    # "natural" — 자연어로 쓰면 LLM 이 yes/no 판정 (일반 사용자 default)
    # "expression" — Python 표현식 eval (고급 사용자, 기존 동작)
    condition_mode: str = "natural"

    last_output: str = ""
    execution_state: str = "pending"


@dataclass
class EdgeData:
    edge_id: str
    src_id: str
    dst_id: str


@dataclass
class LoopNodeData:
    """
    루프 노드 데이터.
    loop_count      : 반복 횟수
    body_node_ids   : 루프 바디에 포함된 PromptNodeData의 node_id 목록 (topo-sorted)
    body_edge_ids   : 루프 바디 내부 엣지 ID 목록

    auto_trigger_type : 자동 트리거 종류
        "none"       — 수동 실행만
        "schedule"   — 특정 요일+시간대에 자동 실행
        "self_model" — eidos_worker.self_model[키] 값이 임계값 초과 시 자동 실행
    auto_trigger_params : 트리거 파라미터 dict (타입별 구조 상이)
        schedule  → {"weekdays": [0..6, ...], "hour_start": int, "hour_end": int}
                    weekdays: 0=월 ~ 6=일, hour 범위 [start, end) 에서 하루 1회 발동
        self_model→ {"key": str, "threshold": float, "op": ">"|">="|"<"|"<="|"=="}
    """
    node_id: str
    title: str
    x: float
    y: float
    loop_count: int = 3
    body_node_ids: List[str] = None   # type: ignore[assignment]
    body_edge_ids: List[str] = None   # type: ignore[assignment]
    last_output: str = ""
    execution_state: str = "pending"
    auto_trigger_type: str = "none"           # "none" | "schedule" | "self_model"
    auto_trigger_params: dict = None          # type: ignore[assignment]
    # 내부 상태: 오늘 schedule 트리거가 이미 발동했는지 추적 (직렬화 제외)
    _trigger_fired_date: str = ""             # "YYYY-MM-DD"

    # ── 조건 종료 (개선 #3) ───────────────────────────────────────
    exit_condition:  str = ""    # 매 iteration 후 eval. 참이면 루프 조기 종료.
                                 # 사용 가능 변수: output (str), iteration (int),
                                 #   score (float, output 안에서 첫 번째 숫자 자동 추출)
    max_iterations:  int = 0     # 0 = loop_count 따름. 양수 = 강제 상한 (안전장치)

    # ── 루프 패턴 (단순화) ────────────────────────────────────────
    # "count"    — N번 반복 (loop_count 만 사용, exit_condition 무시) — 일반 사용자 default
    # "quality"  — 결과가 충분히 좋아질 때까지 (exit_condition 자연어 → 'llm:' 자동 매핑)
    # "goal"     — 목표 달성까지 (exit_condition 자연어 → 'llm:' 자동 매핑)
    # "advanced" — 모든 옵션 직접 노출 (기존 표현식/llm:/모든 메모리 모드)
    loop_mode: str = "count"

    # ── 루프 메모리 (개선 ❹) ──────────────────────────────────────
    memory_mode: str = "none"    # "none" | "append" | "replace" | "llm_compress"
                                 # none        — 메모리 비사용
                                 # append      — 매 iter 출력을 memory에 누적
                                 # replace     — 최신 iter 출력으로 교체
                                 # llm_compress— N회마다 LLM으로 압축 요약
    memory_compress_every: int = 3  # llm_compress 모드: N회마다 압축

    def __post_init__(self):
        if self.body_node_ids is None:
            self.body_node_ids = []
        if self.body_edge_ids is None:
            self.body_edge_ids = []
        if self.auto_trigger_params is None:
            self.auto_trigger_params = {}


# =========================
# Graphics Items
# =========================

class PromptBoxItem(QGraphicsObject):
    """
    Prompt Node on canvas. Has left (input) port and right (output) port.
    Double-click -> open in inspector.
    """
    request_edit = Signal(str)          # node_id
    moved = Signal(str)                 # node_id, for autosave
    request_delete = Signal(str)        # node_id

    def __init__(self, data: PromptNodeData, parent: Optional[QGraphicsObject] = None):
        super().__init__(parent)
        self.data = data

        self.setFlags(
            QGraphicsObject.ItemIsSelectable |
            QGraphicsObject.ItemIsMovable |
            QGraphicsObject.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        self.width = 260
        self.height = 130
        self.radius = 10

        self._hovered = False

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.width, self.height)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing, True)

        # background
        bg = QColor("#2B2B2B")
        if self.isSelected():
            bg = QColor("#32324A")
        elif self._hovered:
            bg = QColor("#303030")

        painter.setPen(QPen(QColor("#555555"), 1))
        painter.setBrush(QBrush(bg))
        painter.drawRoundedRect(self.boundingRect(), self.radius, self.radius)

        # execution state border
        state_color = {
            "running": QColor("#569CD6"),  # Blue
            "success": QColor("#6A9955"),  # Green
            "error": QColor("#F44747"),    # Red
        }.get(self.data.execution_state)

        if state_color:
            pen = QPen(state_color, 3)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(self.boundingRect().adjusted(1.5, 1.5, -1.5, -1.5), self.radius, self.radius)

        # header strip
        header_h = 34
        header_color = {
            "tool":   QColor("#431407"),  # 어두운 주황
            "prompt": QColor("#2e1065"),  # 어두운 보라
            "if":     QColor("#1a3a1a"),  # 어두운 초록
        }.get(self.data.node_type, QColor("#1c1c1e"))

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(header_color))
        painter.drawRoundedRect(QRectF(0, 0, self.width, header_h), self.radius, self.radius)
        painter.drawRect(QRectF(0, header_h - self.radius, self.width, self.radius))  # flatten seam

        # header 좌측에 타입 인디케이터 점 추가
        dot_color = {
            "tool":   QColor("#f97316"),
            "prompt": QColor("#7c3aed"),
        }.get(self.data.node_type, QColor("#6b7280"))
        painter.setBrush(QBrush(dot_color))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QRectF(8, 11, 8, 8))  # 헤더 내 작은 점

        # title
        painter.setPen(QPen(QColor("#EAEAEA")))
        font = QFont("Segoe UI", 10)
        font.setBold(True)
        painter.setFont(font)
        title = self.data.title or "Untitled"
        painter.drawText(QRectF(24, 0, self.width - 34, header_h), Qt.AlignVCenter | Qt.AlignLeft, title)

        # prompt preview
        painter.setPen(QPen(QColor("#CFCFCF")))
        font2 = QFont("Segoe UI", 9)
        font2.setBold(False)
        painter.setFont(font2)
        preview = (self.data.prompt or "").strip().replace("\n", " ")
        if len(preview) > 80:
            preview = preview[:80] + "…"
        painter.drawText(QRectF(10, header_h + 8, self.width - 20, 38), Qt.AlignLeft | Qt.AlignTop, preview)

        # output preview line
        out_preview = (self.data.last_output or "").strip().replace("\n", " ")
        if out_preview:
            if len(out_preview) > 80:
                out_preview = out_preview[:80] + "…"
            painter.setPen(QPen(QColor("#8FE388")))
            painter.drawText(QRectF(10, header_h + 55, self.width - 20, 20),
                             Qt.AlignLeft | Qt.AlignVCenter, f"✓ {out_preview}")

        # ports
        painter.setPen(QPen(QColor("#111111"), 1))
        painter.setBrush(QBrush(QColor("#8AA2FF")))  # input
        painter.drawEllipse(self.input_port_rect())

        painter.setBrush(QBrush(QColor("#FFB86C")))  # output
        painter.drawEllipse(self.output_port_rect())

        # port labels (tiny)
        painter.setPen(QPen(QColor("#AAAAAA")))
        font3 = QFont("Segoe UI", 8)
        painter.setFont(font3)
        painter.drawText(QRectF(16, self.height - 24, 60, 16), Qt.AlignLeft | Qt.AlignVCenter, "IN")
        painter.drawText(QRectF(self.width - 46, self.height - 24, 40, 16), Qt.AlignLeft | Qt.AlignVCenter, "OUT")

    def input_port_center(self) -> QPointF:
        r = self.input_port_rect()
        return self.mapToScene(r.center())

    def output_port_center(self) -> QPointF:
        r = self.output_port_rect()
        return self.mapToScene(r.center())

    def input_port_rect(self) -> QRectF:
        size = 12
        return QRectF(6, self.height - 20, size, size)

    def output_port_rect(self) -> QRectF:
        size = 12
        return QRectF(self.width - 18, self.height - 20, size, size)

    def port_hit_test(self, scene_pos: QPointF) -> Optional[str]:
        """
        Returns 'in' if input port hit, 'out' if output port hit, else None.
        Uses a larger rect for easier clicking.
        """
        local = self.mapFromScene(scene_pos)
        # Make hitbox larger than the visible port circle
        hit_padding = 8
        input_hit_rect = self.input_port_rect().adjusted(-hit_padding, -hit_padding, hit_padding, hit_padding)
        output_hit_rect = self.output_port_rect().adjusted(-hit_padding, -hit_padding, hit_padding, hit_padding)

        if input_hit_rect.contains(local):
            return "in"
        if output_hit_rect.contains(local):
            return "out"
        return None

    def hoverEnterEvent(self, event):
        self._hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.request_edit.emit(self.data.node_id)
        super().mouseDoubleClickEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsObject.ItemPositionHasChanged:
            self.data.x = float(self.scenePos().x())
            self.data.y = float(self.scenePos().y())
            self.moved.emit(self.data.node_id)
        return super().itemChange(change, value)


class EdgeItem(QGraphicsObject):
    """
    Visual edge between PromptBox OUT -> IN
    """
    request_delete = Signal(str)  # edge_id

    def __init__(self, edge: EdgeData, src: PromptBoxItem, dst: PromptBoxItem, parent=None):
        super().__init__(parent)
        self.edge = edge
        self.src = src
        self.dst = dst

        self.setZValue(-10)  # behind nodes
        self.setAcceptHoverEvents(True)

        self._hovered = False

        # track movement
        # [2026-05-07 fix] moved Signal(str) → QGraphicsItem.update() 직접 connect 시
        # node_id 문자열이 update 의 첫 인자로 들어가 TypeError. lambda 로 인자 흡수.
        self.src.moved.connect(lambda _nid: self.update())
        self.dst.moved.connect(lambda _nid: self.update())

    def shape(self) -> QPainterPath:
        # Make the edge easier to click by giving it a wider shape for hit detection
        path = self._get_path()
        stroker = QPainterPathStroker()
        stroker.setWidth(10)
        return stroker.createStroke(path)

    def boundingRect(self) -> QRectF:
        return self.shape().controlPointRect()

    def _get_path(self) -> QPainterPath:
        s = self.src.output_port_center()
        d = self.dst.input_port_center()
        path = QPainterPath()
        path.moveTo(s)
        dx = max(80.0, abs(d.x() - s.x()) * 0.35)
        c1 = QPointF(s.x() + dx, s.y())
        c2 = QPointF(d.x() - dx, d.y())
        path.cubicTo(c1, c2, d)
        return path

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing, True)
        path = self._get_path()

        color = QColor("#FF6B6B") if self._hovered else QColor("#7A7A7A")
        arrow_color = QColor("#FF8F8F") if self._hovered else QColor("#9AA0A6")

        pen = QPen(color, 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)

        # arrow head near destination
        arrow_pen = QPen(arrow_color, 2)
        painter.setPen(arrow_pen)

        t = 0.98
        p1 = path.pointAtPercent(t)
        p0 = path.pointAtPercent(t - 0.02)
        vec = p1 - p0
        length = (vec.x() ** 2 + vec.y() ** 2) ** 0.5
        if length < 0.001:
            return

        ux, uy = vec.x() / length, vec.y() / length
        px, py = -uy, ux

        arrow_size = 8
        a1 = p1 - QPointF(ux * arrow_size, uy * arrow_size) + QPointF(px * arrow_size * 0.6, py * arrow_size * 0.6)
        a2 = p1 - QPointF(ux * arrow_size, uy * arrow_size) - QPointF(px * arrow_size * 0.6, py * arrow_size * 0.6)

        painter.drawLine(p1, a1)
        painter.drawLine(p1, a2)

    def hoverEnterEvent(self, event):
        self._hovered = True
        self.setCursor(Qt.PointingHandCursor)
        self.setZValue(-9) # bring slightly forward
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hovered = False
        self.setCursor(Qt.ArrowCursor)
        self.setZValue(-10)
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.request_delete.emit(self.edge.edge_id)
            event.accept()
        else:
            event.ignore()




class LoopBoxItem(QGraphicsObject):
    """
    루프 노드 그래픽 아이템.
    - 짙은 청록 헤더 (루프 구분)
    - 중앙에 반복 횟수 + 포함 노드 목록 표시
    - IN / OUT 포트 (일반 노드와 동일 위치)
    - 바디 노드들을 반투명 배경으로 감싸는 하이라이트 그리기 (선택 시)
    """
    request_edit   = Signal(str)   # node_id
    moved          = Signal(str)
    request_delete = Signal(str)

    HEADER_COLOR = QColor("#064E3B")   # 짙은 청록
    DOT_COLOR    = QColor("#34D399")   # 연한 에메랄드
    BORDER_COLOR = QColor("#10B981")

    def __init__(self, data: "LoopNodeData", parent=None):
        super().__init__(parent)
        self.data = data
        self.setFlags(
            QGraphicsObject.ItemIsSelectable |
            QGraphicsObject.ItemIsMovable |
            QGraphicsObject.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.width  = 260
        self.height = 130
        self.radius = 10
        self._hovered = False

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.width, self.height)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing, True)

        # 배경
        bg = QColor("#1E2D2A") if not self._hovered else QColor("#243530")
        if self.isSelected():
            bg = QColor("#1A3B35")
        painter.setPen(QPen(QColor("#2D5A4A"), 1))
        painter.setBrush(QBrush(bg))
        painter.drawRoundedRect(self.boundingRect(), self.radius, self.radius)

        # 실행 상태 테두리
        state_color = {
            "running": QColor("#569CD6"),
            "success": QColor("#34D399"),
            "error":   QColor("#F44747"),
        }.get(self.data.execution_state)
        if state_color:
            painter.setPen(QPen(state_color, 3))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(self.boundingRect().adjusted(1.5, 1.5, -1.5, -1.5), self.radius, self.radius)

        # 헤더
        header_h = 34
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self.HEADER_COLOR))
        painter.drawRoundedRect(QRectF(0, 0, self.width, header_h), self.radius, self.radius)
        painter.drawRect(QRectF(0, header_h - self.radius, self.width, self.radius))

        # 헤더 점
        painter.setBrush(QBrush(self.DOT_COLOR))
        painter.drawEllipse(QRectF(8, 11, 8, 8))

        # 루프 아이콘 + 타이틀
        painter.setPen(QPen(QColor("#EAEAEA")))
        font = QFont("Segoe UI", 10); font.setBold(True)
        painter.setFont(font)
        painter.drawText(QRectF(24, 0, self.width - 34, header_h),
                         Qt.AlignVCenter | Qt.AlignLeft, self.data.title)

        # 반복 횟수 뱃지 (우측 상단)
        # 조건 종료 설정 시 "×N ⚡" 표시
        has_exit = bool(getattr(self.data, 'exit_condition', ''))
        badge_text = f"×{self.data.loop_count}{'⚡' if has_exit else ''}"
        painter.setPen(Qt.NoPen)
        badge_bg = QColor("#1D4E3A") if has_exit else QColor("#065F46")
        painter.setBrush(QBrush(badge_bg))
        painter.drawRoundedRect(QRectF(self.width - 48, 8, 40, 18), 5, 5)
        painter.setPen(QPen(self.DOT_COLOR))
        font_b = QFont("Segoe UI", 9); font_b.setBold(True)
        painter.setFont(font_b)
        painter.drawText(QRectF(self.width - 48, 8, 40, 18),
                         Qt.AlignCenter, badge_text)

        # 자동 트리거 인디케이터 (뱃지: 좌상단, ×뱃지 왼쪽)
        ttype = getattr(self.data, 'auto_trigger_type', 'none')
        if ttype != 'none':
            trigger_icon  = "⏰" if ttype == "schedule" else "🧠"
            trigger_color = QColor("#F59E0B") if ttype == "schedule" else QColor("#818CF8")
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(trigger_color.darker(160)))
            painter.drawRoundedRect(QRectF(self.width - 84, 8, 36, 18), 4, 4)
            painter.setPen(QPen(trigger_color))
            font_t = QFont("Segoe UI", 8)
            painter.setFont(font_t)
            painter.drawText(QRectF(self.width - 84, 8, 36, 18), Qt.AlignCenter, trigger_icon)

        # 바디 노드 목록 미리보기
        painter.setPen(QPen(QColor("#6EE7B7")))
        font2 = QFont("Segoe UI", 8)
        painter.setFont(font2)
        body_count = len(self.data.body_node_ids)
        if body_count == 0:
            preview = "바디 노드 없음 — 인스펙터에서 노드를 추가하세요"
        else:
            preview = f"바디 노드 {body_count}개 포함"
        painter.drawText(QRectF(10, header_h + 8, self.width - 20, 24),
                         Qt.AlignLeft | Qt.AlignVCenter, preview)

        # 마지막 출력 미리보기
        out = (self.data.last_output or "").strip().replace("\n", " ")
        if out:
            if len(out) > 80: out = out[:80] + "…"
            painter.setPen(QPen(QColor("#8FE388")))
            painter.drawText(QRectF(10, header_h + 38, self.width - 20, 20),
                             Qt.AlignLeft | Qt.AlignVCenter, f"✓ {out}")

        # 포트
        painter.setPen(QPen(QColor("#111111"), 1))
        painter.setBrush(QBrush(QColor("#8AA2FF")))
        painter.drawEllipse(self.input_port_rect())
        painter.setBrush(QBrush(QColor("#FFB86C")))
        painter.drawEllipse(self.output_port_rect())

        painter.setPen(QPen(QColor("#AAAAAA")))
        font3 = QFont("Segoe UI", 8)
        painter.setFont(font3)
        painter.drawText(QRectF(16, self.height - 24, 60, 16),
                         Qt.AlignLeft | Qt.AlignVCenter, "IN")
        painter.drawText(QRectF(self.width - 46, self.height - 24, 40, 16),
                         Qt.AlignLeft | Qt.AlignVCenter, "OUT")

    def input_port_center(self)  -> QPointF: return self.mapToScene(self.input_port_rect().center())
    def output_port_center(self) -> QPointF: return self.mapToScene(self.output_port_rect().center())
    def input_port_rect(self)  -> QRectF: return QRectF(6, self.height - 20, 12, 12)
    def output_port_rect(self) -> QRectF: return QRectF(self.width - 18, self.height - 20, 12, 12)

    def port_hit_test(self, scene_pos: QPointF) -> Optional[str]:
        local = self.mapFromScene(scene_pos)
        p = 8
        if self.input_port_rect().adjusted(-p,-p,p,p).contains(local):  return "in"
        if self.output_port_rect().adjusted(-p,-p,p,p).contains(local): return "out"
        return None

    def hoverEnterEvent(self, e): self._hovered = True;  self.update(); super().hoverEnterEvent(e)
    def hoverLeaveEvent(self, e): self._hovered = False; self.update(); super().hoverLeaveEvent(e)

    def mouseDoubleClickEvent(self, event):
        self.request_edit.emit(self.data.node_id)
        super().mouseDoubleClickEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsObject.ItemPositionHasChanged:
            self.data.x = float(self.scenePos().x())
            self.data.y = float(self.scenePos().y())
            self.moved.emit(self.data.node_id)
        return super().itemChange(change, value)


class TempEdgeItem(QGraphicsObject):
    """
    Temporary edge while dragging connect.
    """
    def __init__(self, start: QPointF, parent=None):
        super().__init__(parent)
        self.start = start
        self.end = start
        self.setZValue(-5)
        self.setAcceptedMouseButtons(Qt.NoButton)

    def set_end(self, end: QPointF):
        self.end = end
        self.update()

    def boundingRect(self) -> QRectF:
        s, d = self.start, self.end
        return QRectF(min(s.x(), d.x()) - 80, min(s.y(), d.y()) - 80,
                      abs(s.x() - d.x()) + 160, abs(s.y() - d.y()) + 160)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing, True)
        s, d = self.start, self.end

        path = QPainterPath()
        path.moveTo(s)
        dx = max(80.0, abs(d.x() - s.x()) * 0.35)
        c1 = QPointF(s.x() + dx, s.y())
        c2 = QPointF(d.x() - dx, d.y())
        path.cubicTo(c1, c2, d)

        pen = QPen(QColor("#6A0DAD"), 2, Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)


# =========================
# Graphics View (Zoom/Pan/Connect)
# =========================

class PromptCanvasView(QGraphicsView):
    status_msg = Signal(str)

    def __init__(self, scene: QGraphicsScene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        self._panning = False
        self._pan_start = QPointF()

        self._connecting = False
        self._connect_src: Optional[PromptBoxItem] = None
        self._temp_edge: Optional[TempEdgeItem] = None

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            factor = 1.12 if delta > 0 else 1 / 1.12
            self.scale(factor, factor)
            event.accept()
            return
        super().wheelEvent(event)

    def set_theme(self, is_dark: bool):
        self._bg_color   = QColor("#09090b") if is_dark else QColor("#f1f5f9")
        self._grid_color = QColor("#3f3f46") if is_dark else QColor("#cbd5e1")
        self.viewport().update()

    def drawBackground(self, painter, rect):
        super().drawBackground(painter, rect)
        bg = getattr(self, '_bg_color', QColor("#09090b"))
        painter.fillRect(rect, bg)
        # dot grid
        grid = 24
        grid_col = getattr(self, '_grid_color', QColor("#3f3f46"))
        pen = QPen(grid_col, 1)
        pen.setCosmetic(True)
        painter.setPen(pen)
        left = int(rect.left()) - (int(rect.left()) % grid)
        top  = int(rect.top())  - (int(rect.top())  % grid)
        x = left
        while x < rect.right():
            y = top
            while y < rect.bottom():
                painter.drawPoint(x, y)
                y += grid
            x += grid

    def mousePressEvent(self, event):
        # Middle button pan
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        # Connect by clicking port
        if event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())
            item = self.scene().itemAt(scene_pos, self.transform())

            # PromptBoxItem 또는 LoopBoxItem 모두 포트 연결 허용
            if isinstance(item, (PromptBoxItem, LoopBoxItem)):
                port = item.port_hit_test(scene_pos)
                if port == "out":
                    self._connecting = True
                    self._connect_src = item
                    self._temp_edge = TempEdgeItem(item.output_port_center())
                    self.scene().addItem(self._temp_edge)
                    self.status_msg.emit("연결: OUT 포트에서 드래그하여 대상 노드의 IN 포트에 놓으세요.")
                    event.accept()
                    return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            event.accept()
            return

        if self._connecting and self._temp_edge:
            scene_pos = self.mapToScene(event.position().toPoint())
            self._temp_edge.set_end(scene_pos)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton and self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return

        # finish connecting
        if event.button() == Qt.LeftButton and self._connecting:
            scene_pos = self.mapToScene(event.position().toPoint())
            item = self.scene().itemAt(scene_pos, self.transform())

            # PromptBoxItem / LoopBoxItem 모두 dst 허용
            dst_node = item if isinstance(item, (PromptBoxItem, LoopBoxItem)) else None
            ok = False
            if dst_node and self._connect_src:
                port = dst_node.port_hit_test(scene_pos)
                if port == "in" and dst_node != self._connect_src:
                    cb = getattr(self.scene(), "_on_request_connect", None)
                    if callable(cb):
                        ok = cb(self._connect_src, dst_node)

            # cleanup
            if self._temp_edge:
                self.scene().removeItem(self._temp_edge)
                self._temp_edge = None
            self._connecting = False
            self._connect_src = None

            self.status_msg.emit("연결 완료" if ok else "연결 취소")
            event.accept()
            return

        super().mouseReleaseEvent(event)



# =========================
# Canvas Runner Thread  (개선 #1 #2 #3 #4)
# =========================

class CanvasRunnerThread(QThread):
    node_started  = Signal(str, str)
    node_finished = Signal(str, str, bool)
    all_finished  = Signal(str)   # final_result
    run_error     = Signal(str)
    # 컨텍스트 요약 요청 시그널 (메인 스레드에서 LLM 없이도 처리 가능하도록 분리)
    context_summary_ready = Signal(str)   # summary text

    def __init__(self, canvas_window, order, outputs, global_context, start_input,
                 agent_state: Optional["AgentState"] = None):
        super().__init__(canvas_window)
        self.canvas_window = canvas_window
        self.order         = list(order)
        # outputs를 ContextStore로 래핑 (하위 호환: dict가 들어오면 변환)
        if isinstance(global_context, ContextStore):
            self.ctx = global_context
        else:
            self.ctx = ContextStore()
            for nid, out in outputs.items():
                nd = canvas_window.nodes.get(nid)
                title = nd.title if nd else nid
                self.ctx.record(nid, title, out, True)
        self.start_input  = start_input
        self.agent_state  = agent_state   # None 허용

    # ─────────────────────────────────────────────────────────────
    # 메인 실행
    # ─────────────────────────────────────────────────────────────
    def run(self):
        cw          = self.canvas_window
        ctx         = self.ctx
        start_input = self.start_input

        # 의존성 그래프 구축 (개선 #2: 병렬 실행 준비)
        adj, indeg = self._build_dep_graph()
        remaining  = set(self.order)
        done       = set()
        skipped    = set()

        # AgentState 틱
        if self.agent_state:
            self.agent_state.tick()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures: Dict[concurrent.futures.Future, str] = {}

            def _submit_ready():
                """indeg == 0 이고 아직 제출되지 않은 노드를 pool에 제출."""
                for nid in list(remaining):
                    if nid in skipped:
                        remaining.discard(nid)
                        continue
                    if indeg[nid] == 0 and nid not in {f_nid for f_nid in futures.values()}:
                        remaining.discard(nid)
                        fut = pool.submit(self._run_node, nid, ctx, start_input, skipped)
                        futures[fut] = nid

            _submit_ready()

            while futures:
                done_futs = concurrent.futures.wait(
                    list(futures.keys()), return_when=concurrent.futures.FIRST_COMPLETED
                ).done
                for fut in done_futs:
                    nid = futures.pop(fut)
                    try:
                        result, ok, new_skips = fut.result()
                    except Exception as exc:
                        result, ok, new_skips = f"[실행 오류] {exc}", False, set()

                    # 컨텍스트 기록
                    nd    = cw.nodes.get(nid) or cw.loop_nodes.get(nid)
                    title = nd.title if nd else nid
                    ctx.record(nid, title, result, ok)

                    # skip 전파
                    skipped.update(new_skips)

                    # indegree 갱신
                    for child in adj.get(nid, []):
                        if child in indeg:
                            indeg[child] -= 1

                    done.add(nid)
                    self.node_finished.emit(nid, result, ok)

                    # 컨텍스트가 MAX_INLINE 초과 시 요약 요청
                    if len(ctx.history) == ContextStore.MAX_INLINE + 1:
                        self._maybe_update_summary(ctx)

                    _submit_ready()

        # AgentState에 컨텍스트 저장
        if self.agent_state:
            self.agent_state.last_context = ctx.to_dict()

        # 터미널 노드 최종 결과
        all_src     = {e.src_id for e in cw.edges.values()}
        terminal    = [nid for nid in self.order if nid not in all_src and nid not in skipped]
        final_parts = [ctx.get(nid) for nid in terminal if ctx.get(nid)]
        self.all_finished.emit("\n\n".join(final_parts) or "")

    # ─────────────────────────────────────────────────────────────
    # 노드별 실행 (스레드 풀에서 호출)
    # ─────────────────────────────────────────────────────────────
    def _run_node(self, nid: str, ctx: ContextStore, start_input: str,
                  skipped: set) -> Tuple[str, bool, set]:
        """
        단일 노드를 실행하고 (result, ok, new_skip_ids) 를 반환한다.
        루프 / if / prompt / tool 분기.
        """
        cw = self.canvas_window

        # ── 스킵 체크 ────────────────────────────────────────────
        if nid in skipped:
            return f"[스킵]", True, set()

        # ── 루프 노드 ─────────────────────────────────────────────
        if nid in cw.loop_nodes:
            return self._run_loop_node(nid, ctx, start_input)

        node = cw.nodes.get(nid)
        if node is None:
            return "[노드 없음]", False, set()

        self.node_started.emit(nid, node.title)

        # ── if 노드 ───────────────────────────────────────────────
        if node.node_type == 'if':
            return self._run_if_node(node, ctx)

        # ── prompt / tool 노드 ───────────────────────────────────
        if node.node_type == 'tool':
            return self._run_tool_node(node, ctx, start_input)
        else:
            return self._run_prompt_node(node, ctx, start_input)

    def _run_if_node(self, node, ctx: ContextStore) -> Tuple[str, bool, set]:
        cw   = self.canvas_window
        prev = ctx.last_output()
        cond = (node.condition or "").strip()
        mode = (getattr(node, "condition_mode", "natural") or "natural").lower()

        if not cond:
            branch = True
        elif mode == "expression" or cond.startswith("expr:"):
            expr = cond[5:].strip() if cond.startswith("expr:") else cond
            try:
                branch = bool(eval(
                    expr or "False",
                    {"__builtins__": {}},
                    {"prev": prev, "context": ctx.render(),
                     "output": prev, "outputs": ctx.key_outputs},
                ))
            except Exception as e:
                branch = False
                print(f"⚠️ [If] 표현식 평가 오류: {e}")
        else:
            judge_text = cond[4:].strip() if cond.lower().startswith("llm:") else cond
            branch = self._llm_judge_natural(judge_text, prev, ctx, cw)

        result = f"[IF] {'True → 실행' if branch else 'False → 스킵'}"
        new_skips: set = set()
        if not branch:
            new_skips = self._collect_if_skip(node.node_id)
        return result, True, new_skips

    def _llm_judge_natural(self, judge_text: str, prev: str,
                           ctx: ContextStore, cw) -> bool:
        """
        자연어 조건 + 직전 결과를 LLM 에 던져 yes/no 판정.
        실패 시 안전하게 False 반환 (downstream skip — 기존 catch-all 동작 유지).
        """
        if not (judge_text or "").strip():
            return False
        eval_prompt = (
            f"[직전 단계 결과]\n{prev}\n\n"
            f"[누적 컨텍스트 요약]\n{ctx.render()}\n\n"
            f"[판단 요청]\n{judge_text}\n\n"
            f"위 결과가 판단 요청을 만족하면 'yes', 아니면 'no'만 응답하라. "
            f"다른 말은 하지 마라."
        )
        try:
            result, ok = cw._execute_prompt(
                eval_prompt, node_id="__if_llm_judge__", title="If 조건 판정"
            )
            if not ok:
                return False
            first = (result or "").strip().lower()[:10]
            return any(tok in first for tok in
                       ("yes", "네", "true", "맞", "충족", "통과"))
        except Exception as e:
            print(f"⚠️ [If LLM Judge] 오류: {e}")
            return False

    def _collect_if_skip(self, if_nid: str) -> set:
        cw         = self.canvas_window
        skip_queue = {e.dst_id for e in cw.edges.values() if e.src_id == if_nid}
        visited    = set()
        frontier   = set(skip_queue)
        while frontier:
            cur = frontier.pop()
            if cur in visited:
                continue
            visited.add(cur)
            for e in cw.edges.values():
                if e.src_id == cur:
                    frontier.add(e.dst_id)
                    skip_queue.add(e.dst_id)
        return skip_queue

    def _run_prompt_node(self, node, ctx: ContextStore,
                         start_input: str) -> Tuple[str, bool, set]:
        cw = self.canvas_window
        pt = node.prompt
        is_root = not any(e.dst_id == node.node_id for e in cw.edges.values())
        has_var = self._has_explicit_var(pt)
        if is_root and not has_var and start_input:
            pt = f"[사용자 입력]\n{start_input}\n\n[지시사항]\n{pt}"
        compiled = self._compile(pt, ctx, start_input, node.node_id)
        result, ok = cw._execute_prompt(compiled, node_id=node.node_id, title=node.title)
        return result, ok, set()

    def _run_tool_node(self, node, ctx: ContextStore,
                       start_input: str) -> Tuple[str, bool, set]:
        cw    = self.canvas_window
        # tool 입력: upstream 출력 or start_input
        input_str = ctx.last_output() or start_input

        # write_file: LLM 출력이 자연어일 경우 구조화 변환
        if node.prompt == "write_file":
            import re as _re
            # 이미 JSON이면 그대로
            try:
                _test = json.loads(input_str)
                if isinstance(_test, dict) and "path" in _test and "content" in _test:
                    pass  # 이미 OK
                else:
                    raise ValueError
            except Exception:
                # 코드블록 추출
                code_match = _re.search(r'```(?:\w+)?\n(.*?)```', input_str, _re.DOTALL)
                content_body = code_match.group(1).strip() if code_match else input_str.strip()
                # 파일명 추출
                path_match = _re.search(r'[\w/\\-]+\.\w+', input_str)
                guessed_path = path_match.group(0) if path_match else "output/result.py"
                input_str = json.dumps(
                    {"path": guessed_path, "content": content_body},
                    ensure_ascii=False
                )

        tr    = _execute_tool_structured(node.prompt, input_str, cw)
        return tr.to_context_str(), tr.ok, set()

    def _run_loop_node(self, nid: str, ctx: ContextStore,
                       start_input: str) -> Tuple[str, bool, set]:
        cw    = self.canvas_window
        ldata = cw.loop_nodes[nid]
        self.node_started.emit(nid, ldata.title)

        upstream_out = ctx.get(
            next((e.src_id for e in cw.edges.values() if e.dst_id == nid), ""),
            ""
        )
        loop_input    = upstream_out or start_input
        loop_last_out = loop_input
        all_iter_logs: List[str] = []

        max_iter  = ldata.max_iterations if ldata.max_iterations > 0 else ldata.loop_count
        exit_cond = (ldata.exit_condition or "").strip()
        mode      = (getattr(ldata, "loop_mode", "count") or "count").lower()
        mem_key   = f"loop_{nid}"   # AgentState.memory 저장 키

        # ── 루프 모드 → exit_condition 자동 합성 ────────────────────
        # count    : N 번 반복 (조기 종료 비활성)
        # quality  : 결과가 충분히 좋아지면 종료 (자연어 → 'llm:' 자동 prefix)
        # goal     : 목표 달성 시 종료 (자연어 → 'llm:' 자동 prefix)
        # advanced : exit_condition 그대로 (사용자 수동)
        if mode == "count":
            exit_cond = ""
        elif mode in ("quality", "goal") and exit_cond:
            if not exit_cond.lower().startswith("llm:"):
                exit_cond = "llm: " + exit_cond

        # exit_condition 모드 판별
        is_llm_exit = exit_cond.lower().startswith("llm:")
        llm_exit_prompt = exit_cond[4:].strip() if is_llm_exit else ""

        for iteration in range(max_iter):
            iter_ctx = ContextStore()
            for e in ctx.history:
                iter_ctx.record(e.node_id, e.title, e.output, e.ok)

            # ── {{memory}} 주입: 루프 시작 전 현재 memory를 iter_ctx에 메타로 기록
            if self.agent_state and ldata.memory_mode != "none":
                current_mem = self.agent_state.memory.get(mem_key, "")
                if current_mem:
                    iter_ctx.record(
                        f"__mem_{nid}__", "루프 누적 기억", current_mem, True
                    )

            for body_nid in ldata.body_node_ids:
                body_node = cw.nodes.get(body_nid)
                if body_node is None:
                    continue
                self.node_started.emit(body_nid, f"{body_node.title} ({iteration+1}회차)")
                try:
                    if body_node.node_type == "tool":
                        input_str = iter_ctx.last_output() or loop_input
                        tr = _execute_tool_structured(body_node.prompt, input_str, cw)
                        b_result, b_ok = tr.to_context_str(), tr.ok
                    else:
                        pt = body_node.prompt
                        is_body_root = not any(
                            e.dst_id == body_nid
                            for e in cw.edges.values()
                            if e.edge_id in ldata.body_edge_ids
                        )
                        if is_body_root and not self._has_explicit_var(pt) and loop_input:
                            pt = f"[루프 입력]\n{loop_input}\n\n[지시사항]\n{pt}"
                        compiled = self._compile(pt, iter_ctx, loop_input, body_nid)
                        b_result, b_ok = cw._execute_prompt(
                            compiled, node_id=body_nid, title=body_node.title
                        )
                except Exception as ex:
                    b_result, b_ok = f"[{body_node.title}] 실행 오류: {ex}", False

                iter_ctx.record(body_nid, body_node.title, b_result, b_ok)
                loop_last_out = b_result
                self.node_finished.emit(body_nid, f"({iteration+1}회차) {b_result}", b_ok)

            all_iter_logs.append(f"=== {iteration+1}회차 ===\n{loop_last_out}")

            # ── 루프 메모리 업데이트 (❹) ────────────────────────────
            if self.agent_state and ldata.memory_mode != "none":
                _update_loop_memory(
                    agent_state   = self.agent_state,
                    mem_key       = mem_key,
                    iteration     = iteration,
                    last_output   = loop_last_out,
                    iter_ctx      = iter_ctx,
                    memory_mode   = ldata.memory_mode,
                    compress_every= ldata.memory_compress_every,
                    cw            = cw,
                )

            # ── 조건 종료 평가 ──────────────────────────────────────
            if exit_cond:
                should_exit = False

                if is_llm_exit:
                    # ── LLM 자기평가 (③ 핵심) ───────────────────────
                    should_exit = self._llm_exit_eval(
                        llm_exit_prompt, loop_last_out, iteration + 1,
                        iter_ctx, cw,
                    )
                else:
                    # ── 기존 eval 방식 ──────────────────────────────
                    score = self._extract_score(loop_last_out)
                    try:
                        should_exit = bool(eval(
                            exit_cond,
                            {"__builtins__": {}},
                            {"output": loop_last_out, "iteration": iteration + 1,
                             "score": score, "i": iteration},
                        ))
                    except Exception as e:
                        should_exit = False
                        print(f"⚠️ [Loop eval] 평가 오류: {e}")

                if should_exit:
                    print(f"✅ [Loop] '{ldata.title}' {iteration+1}회차 후 종료 "
                          f"({'LLM 판단' if is_llm_exit else 'eval 조건'})")
                    break

            # ── goal 달성 체크 (② 핵심) ────────────────────────────
            if self.agent_state and self.agent_state.goal:
                goal_met = self._check_goal(loop_last_out, iter_ctx, cw)
                if goal_met:
                    self.agent_state.progress = 1.0
                    print(f"🎯 [Loop] goal 달성 감지 — '{ldata.title}' 조기 종료")
                    break

        loop_result           = "\n\n".join(all_iter_logs)
        ldata.last_output     = loop_last_out
        ldata.execution_state = "success"
        return loop_last_out, True, set()

    # ── LLM 자기평가 ──────────────────────────────────────────────────────
    def _llm_exit_eval(self, judge_prompt: str, last_output: str,
                       iteration: int, ctx: ContextStore,
                       cw) -> bool:
        """
        LLM에게 루프를 계속할지 종료할지 판단하게 한다.

        judge_prompt 예시:
          "위 결과가 사용자 요구를 충분히 만족했는가?"
          "품질 점수가 0.85 이상인가?"
          "더 이상 개선이 필요 없는가?"

        LLM은 반드시 'yes' 또는 'no'로만 응답하도록 프롬프트를 구성한다.
        파싱: 응답 첫 토큰에 'yes'/'네'/'true' 포함 → 종료
        """
        goal_block = ""
        if self.agent_state and self.agent_state.goal:
            goal_block = f"[에이전트 목표]\n{self.agent_state.goal}\n\n"

        eval_prompt = (
            f"{goal_block}"
            f"[{iteration}회차 실행 결과]\n{last_output}\n\n"
            f"[누적 컨텍스트 요약]\n{ctx.render()}\n\n"
            f"[판단 요청]\n{judge_prompt}\n\n"
            f"위 결과를 기준으로 루프를 종료해야 하면 'yes', "
            f"계속해야 하면 'no'만 응답하라. 다른 말은 하지 마라."
        )
        try:
            result, ok = cw._execute_prompt(eval_prompt, node_id="__llm_eval__",
                                             title="LLM 자기평가")
            if not ok:
                return False
            first = result.strip().lower()[:10]
            return any(tok in first for tok in ("yes", "네", "true", "종료", "완료", "충족"))
        except Exception as e:
            print(f"⚠️ [LLM Exit Eval] 오류: {e}")
            return False

    def _check_goal(self, last_output: str, ctx: ContextStore, cw) -> bool:
        """
        AgentState.goal이 설정된 경우, 현재 출력이 goal을 달성했는지
        LLM에게 간단히 묻는다. 비용 절감을 위해 마지막 출력만 기준으로 판단.
        eidos_worker 없으면 항상 False 반환 (안전).
        """
        if not getattr(cw, 'eidos_worker', None):
            return False
        goal = self.agent_state.goal
        check_prompt = (
            f"[에이전트 목표]\n{goal}\n\n"
            f"[최신 실행 결과]\n{last_output}\n\n"
            f"위 결과가 목표를 완전히 달성했으면 'yes', 아니면 'no'만 응답하라."
        )
        try:
            result, ok = cw._execute_prompt(check_prompt, node_id="__goal_check__",
                                             title="Goal 달성 체크")
            if not ok:
                return False
            first = result.strip().lower()[:10]
            return any(tok in first for tok in ("yes", "네", "true", "달성", "완료"))
        except Exception as e:
            print(f"⚠️ [Goal Check] 오류: {e}")
            return False

    # ─────────────────────────────────────────────────────────────
    # 유틸리티
    # ─────────────────────────────────────────────────────────────
    def _build_dep_graph(self) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
        """외부 DAG 기반 adj + indeg 반환 (loop body 엣지 제외)."""
        cw = self.canvas_window
        body_edges = set()
        for ld in cw.loop_nodes.values():
            body_edges.update(ld.body_edge_ids)

        adj:   Dict[str, List[str]] = defaultdict(list)
        indeg: Dict[str, int]       = {nid: 0 for nid in self.order}

        for e in cw.edges.values():
            if e.edge_id in body_edges:
                continue
            if e.src_id in indeg and e.dst_id in indeg:
                adj[e.src_id].append(e.dst_id)
                indeg[e.dst_id] += 1
        return dict(adj), indeg

    @staticmethod
    def _has_explicit_var(template: str) -> bool:
        markers = ("{{input}}", "{{prev}}", "{{context}}", "{{n:", "{{key:",
                   "{input}", "{prev}", "{context}", "{프롬프트", "{이전")
        return any(m in (template or "") for m in markers)

    @staticmethod
    def _extract_score(text: str) -> float:
        """출력 텍스트에서 첫 번째 숫자를 score로 추출."""
        m = re.search(r"[-+]?\d*\.?\d+", text)
        return float(m.group()) if m else 0.0

    def _compile(self, raw: str, ctx: ContextStore,
                 start_input: str, nid: str = "") -> str:
        """
        ContextStore 기반 프롬프트 컴파일.
        추가 변수: {{key:<node_id>}}, {{var:<key>}} (AgentState), {{goal}}
        """
        cw  = self.canvas_window
        s   = raw or ""
        prev = ctx.last_output()
        context_text = ctx.render()

        # ── goal 주입 (AgentState.goal_prompt 설정 기반) ─────────
        if self.agent_state:
            s = self.agent_state.inject_goal(s)
            # {{goal}} 명시적 변수 치환 (inject_goal이 처리하지 않은 경우 보완)
            s = s.replace("{{goal}}", self.agent_state.goal or "")
            s = s.replace("{goal}",   self.agent_state.goal or "")
            # {{memory}} → AgentState.memory 전체 렌더링 (루프 누적 기억)
            memory_text = _render_memory(self.agent_state.memory)
            s = s.replace("{{memory}}", memory_text)
            s = s.replace("{memory}",   memory_text)

        # 기본 변수
        s = s.replace("{{context}}", context_text).replace("{context}", context_text)
        s = s.replace("{{input}}", start_input).replace("{input}", start_input)
        s = s.replace("{{prev}}", prev).replace("{prev}", prev)

        # {{key:<node_id>}} → 특정 노드 출력 직접 참조
        for m in re.finditer(r"\{\{key:([^}]+)\}\}", s):
            s = s.replace(m.group(0), ctx.get(m.group(1).strip()))
        # 단일 중괄호 버전
        for m in re.finditer(r"\{key:([^}]+)\}", s):
            s = s.replace(m.group(0), ctx.get(m.group(1).strip()))

        # {{n:<node_id>}} 레거시 호환
        for m in re.finditer(r"\{\{n:([^}]+)\}\}", s):
            s = s.replace(m.group(0), ctx.get(m.group(1).strip()))

        # {{var:<key>}} → AgentState.variables
        if self.agent_state:
            for m in re.finditer(r"\{\{var:([^}]+)\}\}", s):
                s = s.replace(m.group(0), self.agent_state.get_var(m.group(1).strip()))
            for m in re.finditer(r"\{var:([^}]+)\}", s):
                s = s.replace(m.group(0), self.agent_state.get_var(m.group(1).strip()))

        # 한글 prev 별칭
        _PREV = re.compile(
            r"\{(?:프롬프트\d+[_\s]*출력|이전[_\s]*출력|previous[_\s]*output|prev[_\s]*output)\}",
            re.IGNORECASE,
        )
        s = _PREV.sub(prev, s)

        # upstream 자동 주입 (명시적 변수 없을 때)
        if nid and not self._has_explicit_var(raw or ""):
            ups = [(cw.nodes[e.src_id].title if e.src_id in cw.nodes else e.src_id,
                    ctx.get(e.src_id))
                   for e in cw.edges.values()
                   if e.dst_id == nid and ctx.get(e.src_id)]
            if ups:
                block = "\n\n".join(f"[{t}의 출력]\n{r}" for t, r in ups)
                s = f"[이전 단계 결과]\n{block}\n\n[현재 지시사항]\n{s}"
        return s

    def _maybe_update_summary(self, ctx: ContextStore):
        """기록이 MAX_INLINE을 넘으면 이전 항목들을 요약 (LLM 옵션)."""
        cw = self.canvas_window
        if not getattr(cw, 'eidos_worker', None):
            # 요약 없이 타이틀 목록만
            titles = ", ".join(e.title for e in ctx.history[:-ContextStore.MAX_INLINE])
            ctx.update_summary(f"[이전 실행 완료: {titles}]")
            return
        try:
            to_summarize = ctx.history[:-ContextStore.MAX_INLINE]
            text = "\n\n".join(f"[{e.title}]\n{e.output[:400]}" for e in to_summarize)
            prompt = (f"다음 AI 에이전트 실행 결과들을 {ContextStore.SUMMARY_TOKENS}자 이내로 "
                      f"핵심만 압축 요약하라.\n\n{text}")
            from llm_module import get_llm_response_async
            loop = asyncio.new_event_loop()
            try:
                summary = loop.run_until_complete(get_llm_response_async(prompt))
            finally:
                loop.close()
            ctx.update_summary(summary)
        except Exception as e:
            print(f"⚠️ [Context Summary] 요약 실패: {e}")


# =========================
# 루프 메모리 헬퍼  (개선 ❹)
# =========================

def _safe_load_agent_state(raw: dict) -> "AgentState":
    """
    JSON dict → AgentState 안전 복원.
    알 수 없는 필드는 무시하고, 누락 필드는 기본값으로 채운다.
    버전 불일치로 인한 TypeError를 완전히 방지한다.
    """
    import dataclasses
    raw.setdefault("goal_prompt", "none")
    raw.setdefault("planner_style", "sequential")
    valid_fields = {f.name for f in dataclasses.fields(AgentState)}
    filtered = {k: v for k, v in raw.items() if k in valid_fields}
    try:
        return AgentState(**filtered)
    except Exception as e:
        print(f"⚠️ [AgentState] 복원 실패 ({e}), 기본값 사용")
        return AgentState()


def _render_memory(memory: dict) -> str:
    """AgentState.memory를 {{memory}} 변수용 텍스트로 렌더링."""
    if not memory:
        return "(기억 없음)"
    parts = []
    for k, v in memory.items():
        if k.startswith("loop_"):
            parts.append(f"[{k}]\n{v}")
        else:
            parts.append(f"[{k}] {v}")
    return "\n\n".join(parts)


def _update_loop_memory(
    agent_state,
    mem_key: str,
    iteration: int,
    last_output: str,
    iter_ctx: "ContextStore",
    memory_mode: str,
    compress_every: int,
    cw,
):
    """
    iteration 완료 후 AgentState.memory[mem_key]를 업데이트.

    memory_mode
    -----------
    append       : 매 iter 출력을 누적 (구분선 포함)
    replace      : 최신 iter 출력으로 교체
    llm_compress : append 하되 compress_every 마다 LLM 압축
    """
    existing = agent_state.memory.get(mem_key, "")
    iter_label = f"[{iteration+1}회차]"

    if memory_mode == "replace":
        agent_state.memory[mem_key] = f"{iter_label}\n{last_output}"

    elif memory_mode in ("append", "llm_compress"):
        new_entry = f"{iter_label}\n{last_output}"
        agent_state.memory[mem_key] = (
            f"{existing}\n\n{new_entry}" if existing else new_entry
        )

        # llm_compress: compress_every 회차마다 LLM 압축
        if (memory_mode == "llm_compress"
                and compress_every > 0
                and (iteration + 1) % compress_every == 0
                and getattr(cw, 'eidos_worker', None)):
            _compress_loop_memory(agent_state, mem_key, cw)


def _compress_loop_memory(agent_state, mem_key: str, cw):
    """LLM으로 누적 메모리를 압축 요약."""
    current = agent_state.memory.get(mem_key, "")
    if not current:
        return
    prompt = (
        f"다음은 AI 루프의 반복 실행 기록이다. "
        f"핵심 패턴, 개선점, 학습된 내용을 300자 이내로 압축 요약하라.\n\n{current}"
    )
    try:
        result, ok = cw._execute_prompt(prompt, node_id="__mem_compress__",
                                         title="메모리 압축")
        if ok and result:
            agent_state.memory[mem_key] = f"[압축 요약]\n{result}"
            print(f"🗜️ [Memory] '{mem_key}' 압축 완료")
    except Exception as e:
        print(f"⚠️ [Memory Compress] 오류: {e}")


# =========================
# DAG Planner  (개선 ❷)
# =========================

# Planner가 생성하는 DAG JSON 스키마 설명 (LLM에게 제공)
_PLANNER_SCHEMA = """
당신은 AI 에이전트 DAG(방향성 비순환 그래프)를 설계하는 전문가입니다.
사용자의 목표를 받아 아래 JSON 형식의 실행 계획을 생성하세요.

## 출력 형식 (JSON만 출력, 마크다운 코드블록 없이)
{
  "nodes": [
    {
      "node_id": "n1",
      "title": "노드 제목",
      "node_type": "prompt",
      "prompt": "노드가 LLM에게 내릴 지시사항. {{input}}, {{prev}}, {{context}} 변수 사용 가능.",
      "condition": "",
      "x": 0, "y": 0,
      "last_output": "", "execution_state": "pending"
    }
  ],
  "edges": [
    {"edge_id": "e1", "src_id": "n1", "dst_id": "n2"}
  ]
}

## node_type 종류
- "prompt" : LLM 호출 노드 (가장 일반적)
- "tool"   : 도구 실행 노드 (prompt 필드에 도구명 작성)
- "if"     : 조건 분기 노드 (condition 필드에 Python 표현식)

## 설계 원칙
{style_desc}

## 규칙
- node_id는 "n1", "n2" ... 순서로
- edge_id는 "e1", "e2" ... 순서로
- x 좌표는 노드 순서대로 300씩 증가 (y는 병렬 노드는 ±150으로 분기)
- 노드가 너무 많으면 품질이 떨어짐 → 최소한으로 설계 (보통 3~7개)
- 각 노드의 prompt는 구체적이고 명확하게 작성
- 마지막 노드는 최종 결과를 정리/종합하는 역할

목표: {goal}
스타일: {style}
"""

_PLANNER_STYLE_DESC = {
    "sequential": "노드를 순서대로 연결. 단순하고 안정적. A→B→C→D 구조.",
    "parallel":   "독립적인 작업은 병렬로 분기 후 합류. 빠른 실행에 유리. A→(B,C)→D 구조.",
    "loop":       "핵심 작업을 루프로 감싸 반복 개선. 품질 향상에 유리. A→[Loop:B→C]→D 구조. 단, loop 노드는 별도 처리가 필요하므로 loop 없이 설계할 것.",
    "auto":       "목표의 특성에 맞게 최적 구조를 자유롭게 선택.",
}


def _run_planner(goal: str, style: str, cw) -> Optional[dict]:
    """
    goal 텍스트를 받아 LLM으로 DAG JSON을 생성하고 파싱해 반환.
    실패 시 None 반환.

    Parameters
    ----------
    goal  : 사용자 목표 문자열
    style : "sequential" | "parallel" | "loop" | "auto"
    cw    : PromptCanvasWindow 인스턴스
    """
    style_desc = _PLANNER_STYLE_DESC.get(style, _PLANNER_STYLE_DESC["sequential"])
    # NOTE: _PLANNER_SCHEMA 는 JSON 예시의 리터럴 중괄호 { } 를 그대로 포함하므로
    # str.format() 을 쓰면 그것들을 치환 필드로 오인해 KeyError 가 난다
    # (예: KeyError: '\n  "nodes"'). 실제 치환 토큰 3개만 리터럴 replace 로 처리.
    planner_prompt = (
        _PLANNER_SCHEMA
        .replace("{style_desc}", style_desc)
        .replace("{goal}", goal)
        .replace("{style}", style)
    )

    try:
        raw, ok = cw._execute_prompt(planner_prompt,
                                      node_id="__planner__",
                                      title="DAG Planner")
        if not ok or not raw:
            return None

        # JSON 파싱 (마크다운 펜스 제거)
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        # 첫 { 부터 마지막 } 까지만 추출
        start = cleaned.find("{")
        end   = cleaned.rfind("}") + 1
        if start == -1 or end == 0:
            print(f"⚠️ [Planner] JSON 구조 없음: {cleaned[:200]}")
            return None

        data = json.loads(cleaned[start:end])

        # 필수 필드 검증
        if "nodes" not in data or "edges" not in data:
            print(f"⚠️ [Planner] nodes/edges 필드 없음")
            return None

        # 각 노드에 누락 필드 기본값 채우기
        for nd in data["nodes"]:
            nd.setdefault("node_type", "prompt")
            nd.setdefault("condition", "")
            nd.setdefault("last_output", "")
            nd.setdefault("execution_state", "pending")
            nd.setdefault("prompt", "")

        print(f"✅ [Planner] DAG 생성 완료 — 노드 {len(data['nodes'])}개, "
              f"엣지 {len(data['edges'])}개")
        return data

    except json.JSONDecodeError as e:
        print(f"⚠️ [Planner] JSON 파싱 실패: {e}")
        return None
    except Exception as e:
        print(f"⚠️ [Planner] 오류: {e}")
        return None


class _PlannerThread(QThread):
    """LLM DAG Planner 를 GUI 스레드 밖에서 실행해 UI 프리징(렉)을 방지.

    _run_planner 는 cw.eidos_worker 읽기 + cw._execute_prompt(LLM, 자체
    asyncio 루프) + JSON 파싱만 하며 GUI 객체를 만지지 않으므로 워커
    스레드에서 안전하다. GUI 반영(clear_all/_load_agent_data 등)은
    result_ready 시그널을 받은 메인 스레드 슬롯에서 수행한다.
    """
    result_ready = Signal(object)  # dict | None

    def __init__(self, canvas_window, goal: str, style: str):
        super().__init__(canvas_window)
        self._cw = canvas_window
        self._goal = goal
        self._style = style

    def run(self):
        try:
            data = _run_planner(self._goal, self._style, self._cw)
        except Exception as e:
            print(f"⚠️ [PlannerThread] 예외: {e}")
            data = None
        self.result_ready.emit(data)


def _auto_layout_dag(dag_data: dict) -> dict:
    """
    Planner가 생성한 DAG의 노드 좌표를 자동 정렬한다.
    위상 정렬 기반으로 레이어를 계산하고 x/y를 배치.
    노드들이 x=0에 몰려있거나 좌표가 너무 가까울 때 호출.
    """
    nodes = {nd["node_id"]: nd for nd in dag_data.get("nodes", [])}
    edges = dag_data.get("edges", [])

    # 위상 레이어 계산
    indeg  = {nid: 0 for nid in nodes}
    adj    = {nid: [] for nid in nodes}
    for e in edges:
        src, dst = e.get("src_id"), e.get("dst_id")
        if src in adj and dst in indeg:
            adj[src].append(dst)
            indeg[dst] += 1

    from collections import deque
    layer: Dict[str, int] = {}
    q = deque(nid for nid, d in indeg.items() if d == 0)
    while q:
        nid = q.popleft()
        for child in adj.get(nid, []):
            layer[child] = max(layer.get(child, 0), layer.get(nid, 0) + 1)
            indeg[child] -= 1
            if indeg[child] == 0:
                q.append(child)

    # 레이어별 노드 묶기
    from collections import defaultdict as _dd
    layers_map: Dict[int, List[str]] = _dd(list)
    for nid in nodes:
        lv = layer.get(nid, 0)
        layers_map[lv].append(nid)

    # 좌표 배치
    X_STEP, Y_STEP = 320, 160
    for lv, nids in sorted(layers_map.items()):
        total = len(nids)
        for i, nid in enumerate(nids):
            nodes[nid]["x"] = float(lv * X_STEP)
            nodes[nid]["y"] = float((i - (total - 1) / 2) * Y_STEP)

    dag_data["nodes"] = list(nodes.values())
    return dag_data


# ─────────────────────────────────────────────────────────────────────
# 구조화된 Tool 실행 (개선 #4)
# ─────────────────────────────────────────────────────────────────────

def _execute_tool_structured(tool_name: str, input_str: str,
                              canvas_window) -> ToolResult:
    """
    스키마 검증 → 타입 변환 → 실행 → ToolResult 반환.
    canvas_window.eidos_worker.run_tool(tool_name, parsed_params) 우선 호출.
    """
    t0 = time.monotonic()

    # 1. 스키마 검증
    is_valid, err_msg, parsed = validate_tool_input(tool_name, input_str)
    if not is_valid:
        return ToolResult(
            tool_name=tool_name, ok=False, output="",
            error=f"입력 스키마 오류: {err_msg}",
            duration_ms=(time.monotonic() - t0) * 1000,
        )

    # 2. Worker 실행
    worker = getattr(canvas_window, 'eidos_worker', None)
    if worker is not None:
        fn = getattr(worker, 'run_tool', None)
        if callable(fn):
            try:
                raw = fn(tool_name, parsed)
                duration = (time.monotonic() - t0) * 1000
                if isinstance(raw, dict):
                    return ToolResult(
                        tool_name=tool_name, ok=True,
                        output=json.dumps(raw, ensure_ascii=False),
                        output_raw=raw, duration_ms=duration,
                    )
                return ToolResult(tool_name=tool_name, ok=True,
                                  output=str(raw), output_raw=raw,
                                  duration_ms=duration)
            except Exception as e:
                return ToolResult(
                    tool_name=tool_name, ok=False, output="",
                    error=str(e),
                    duration_ms=(time.monotonic() - t0) * 1000,
                )

    # 3. 시뮬레이션 폴백
    sim_outputs = {
        "perform_web_search": lambda p: f"웹 검색 결과: '{p.get('query', input_str)}'",
        "read_file":          lambda p: f"파일 '{p.get('path', '?')}' 내용: (시뮬레이션)",
        "write_file":         lambda p: f"파일 '{p.get('path', '?')}' 저장 완료 (시뮬레이션)",
        "calculate_math":     lambda p: f"수식 결과: {p.get('expression', '?')} = (시뮬레이션)",
        "execute_python_file":lambda p: f"Python '{p.get('path', '?')}' 실행 완료 (시뮬레이션)",
    }
    sim_fn = sim_outputs.get(tool_name)
    output = sim_fn(parsed) if sim_fn else f"[시뮬레이션] {tool_name}: {input_str[:80]}"
    return ToolResult(
        tool_name=tool_name, ok=True, output=output,
        duration_ms=(time.monotonic() - t0) * 1000,
    )



# =========================
# Main Window
# =========================

class PromptCanvasWindow(QMainWindow):
    """
    Prompt Canvas – 사용자 프롬프트 블록 조립 → DAG 실행 → 에이전트로 저장/활용
    """
    def __init__(self, parent=None, eidos_worker: Any = None):
        super().__init__(parent)
        self.eidos_worker = eidos_worker  # optional hook

        self.setWindowTitle("EIDOS Prompt Canvas (Agent Composer)")
        self.resize(1400, 900)

        # In-memory graph
        self.nodes: Dict[str, PromptNodeData] = {}
        self.node_items: Dict[str, PromptBoxItem] = {}
        self.edges: Dict[str, EdgeData] = {}
        self.edge_items: Dict[str, EdgeItem] = {}
        self.loop_nodes: Dict[str, LoopNodeData] = {}
        self.loop_items: Dict[str, LoopBoxItem] = {}

        self.current_node_id: Optional[str] = None
        self._dirty = False

        # 에이전트 영속 상태 (개선 #5)
        self.agent_state = AgentState()

        self._setup_ui()
        self._setup_actions()

        # Goal 패널 초기화 (② 핵심)
        self._refresh_goal_panel()

        # 자동 트리거 감시 시작
        self._trigger_watcher = LoopTriggerWatcher(self, parent=self)
        self._trigger_watcher.start()

        self.statusBar().showMessage("프롬프트 캔버스 준비 완료")

        # 부모 테마로 초기 적용
        _init_theme = getattr(parent, 'current_theme', 'Modern Dark') if parent else 'Modern Dark'
        self.apply_theme(_init_theme)

    # -------------------------
    # UI
    # -------------------------

    def _setup_ui(self):
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal, self)

        # left inspector
        self.inspector = self._create_inspector()
        splitter.addWidget(self.inspector)

        # canvas
        self.scene = QGraphicsScene()
        self.scene.setSceneRect(-4000, -4000, 8000, 8000)
        self.scene._on_request_connect = self._create_edge_from_items  # callback from view
        self.scene.selectionChanged.connect(self._on_selection_changed)

        self.view = PromptCanvasView(self.scene)
        self.view.status_msg.connect(lambda m: self.statusBar().showMessage(m, 3000))
        splitter.addWidget(self.view)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)

        root_layout.addWidget(splitter)
        self.setCentralWidget(root)

        self._apply_scene_background_grid()

        # autosave debounce (for inspector edits)
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.timeout.connect(self._apply_inspector_to_node)

    def _create_inspector(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(320)
        panel.setStyleSheet("""
            QWidget { background-color: #18181b; color: #e4e4e7; }
            QLabel  { color: #a1a1aa; }
            QLineEdit, QTextEdit {
                background: #27272a;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                color: #e4e4e7;
                padding: 5px 8px;
            }
            QPushButton {
                background: transparent;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                color: #e4e4e7;
                padding: 5px 12px;
            }
            QPushButton:hover { background: #27272a; }
        """)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title = QLabel(_ct("inspector_title"))
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        self.inspector_node_id = QLabel(_ct("inspector_none"))
        self.inspector_node_id.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.inspector_node_id)

        layout.addWidget(QLabel(_ct("lbl_title")))
        self.ed_title = QLineEdit()
        self.ed_title.setPlaceholderText(_ct("ph_title"))
        layout.addWidget(self.ed_title)

        # [tool 노드 전용] 도구 선택 콤보박스
        self.tool_combo_label = QLabel(_ct("lbl_tool"))
        self.tool_combo_label.setVisible(False)
        layout.addWidget(self.tool_combo_label)

        self._TOOL_OPTIONS = [
            # (표시명, 내부 키)
            ("── 검색 / 웹 ──",              ""),
            ("🔍 웹 검색",                   "perform_web_search"),
            ("── 파일 ──",                   ""),
            ("📖 파일 읽기",                  "read_file"),
            ("✏️  파일 쓰기",                 "write_file"),
            ("🗂️  코드 골격 읽기",             "read_code_skeleton"),
            ("── 코드 / 수식 ──",             ""),
            ("🐍 Python 파일 실행",           "execute_python_file"),
            ("🧮 수식 계산",                  "calculate_math"),
            ("── 프로젝트 ──",               ""),
            ("📁 프로젝트 파일 일괄 작성",      "write_project_files_async"),
            ("🔁 동적 툴 재로드",             "reload_tool_registry_async"),
            ("── 텍스트 ──",                 ""),
            ("📝 텍스트 작성",               "write_text"),
            ("── Google Calendar ──",       ""),
            ("📅 일정 조회",                  "gcal_list_events"),
            ("➕ 일정 생성",                  "gcal_create_event"),
            ("🗑️  일정 삭제",                 "gcal_delete_event"),
            ("── 4-엔진 (DC/AC/CC/RC) ──",   ""),
            ("🧩 DC 의미 분해",               "dc_decompose"),
            ("💡 AC cross-domain 유비",       "ac_analogize"),
            ("⚖️  CC 비판 (선택/검증)",        "cc_critique"),
            ("🔁 RC 반복 자기비판",            "rc_reason"),
            ("── 워크플로우 지능 ──",         ""),
            ("📊 friction 5축 점수",           "friction_score"),
            ("🌐 도메인 ground-truth 검색",    "web_search_domain"),
            ("── 자율 탐색 ──",               ""),
            ("🤖 자율 탐색 (링크 불필요)",     "autonomous_research"),
            ("── 내장 브라우저 탐색 ──",      ""),
            ("🌍 페이지 이동 + 로드 대기",     "navigate_and_wait"),
            ("📄 페이지 본문 텍스트 추출",     "get_browser_visible_text"),
            ("⚡ JS 실행 (결과 반환)",         "browser_run_script"),
            ("📷 페이지 스크린샷 캡쳐",        "grab_browser_screenshot"),
        ]
        self.tool_combo = QComboBox()
        self.tool_combo.setStyleSheet("""
            QComboBox {
                background: #27272a;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                color: #e4e4e7;
                padding: 5px 8px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background: #27272a;
                border: 1px solid #52525b;
                color: #e4e4e7;
                selection-background-color: #3f3f46;
            }
        """)
        for label, key in self._TOOL_OPTIONS:
            self.tool_combo.addItem(label, userData=key)
            if key == "":  # 구분자 항목 비활성화
                idx = self.tool_combo.count() - 1
                self.tool_combo.model().item(idx).setEnabled(False)
                self.tool_combo.model().item(idx).setForeground(
                    __import__("PySide6.QtGui", fromlist=["QColor"]).QColor("#71717a")
                )
        self.tool_combo.setVisible(False)
        self.tool_combo.currentIndexChanged.connect(self._on_tool_combo_changed)
        layout.addWidget(self.tool_combo)

        self.prompt_label = QLabel(_ct("lbl_prompt"))
        layout.addWidget(self.prompt_label)
        self.ed_prompt = QTextEdit()
        # Placeholder text will be set dynamically in open_in_inspector
        self.original_prompt_placeholder = (
            "여기에 프롬프트를 작성하세요.\n"
            "예: 아래 컨텍스트를 기반으로 다음 단계를 생성하라...\n\n"
            "변수 사용 예 (이중/단일 중괄호 모두 지원):\n"
            " - {{context}}  또는 {context}  : 누적 컨텍스트\n"
            " - {{input}}    또는 {input}    : 시작 입력\n"
            " - {{prev}}     또는 {prev}     : 직전 노드 출력\n"
            " - {{n:<node_id>}}              : 특정 노드 출력\n"
            " - {프롬프트1_출력}              : 직전 노드 출력 (한글 별칭)"
        )
        self.ed_prompt.setPlaceholderText(self.original_prompt_placeholder)
        layout.addWidget(self.ed_prompt, 1)

        # [if 노드 전용] 조건 입력 — 자연어 (default) + 고급 Python 표현식 토글
        self.condition_label = QLabel("이 조건이 참이면 → 다음 노드 실행, 거짓이면 → 스킵")
        self.condition_label.setStyleSheet("color: #fbbf24; font-weight: bold;")
        self.condition_label.setVisible(False)
        layout.addWidget(self.condition_label)

        # 자연어 입력 (멀티라인) — 일반 사용자 default
        self.ed_condition_natural = QTextEdit()
        self.ed_condition_natural.setPlaceholderText(
            "예) 결과에 오류 메시지가 있다\n"
            "예) 결과가 100자가 넘는다\n"
            "예) 결과가 한국어가 아니다\n"
            "→ 자연어로 쓰면 LLM 이 yes/no 를 판정합니다."
        )
        self.ed_condition_natural.setFixedHeight(80)
        self.ed_condition_natural.setVisible(False)
        self.ed_condition_natural.textChanged.connect(self._schedule_autosave)
        layout.addWidget(self.ed_condition_natural)

        # 고급 토글 — Python 표현식 직접 입력
        from PySide6.QtWidgets import QCheckBox as _QCheckBox
        self.chk_condition_advanced = _QCheckBox("⚙️ Python 표현식으로 직접 쓰기 (고급)")
        self.chk_condition_advanced.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.chk_condition_advanced.setVisible(False)
        self.chk_condition_advanced.toggled.connect(self._on_condition_mode_toggled)
        layout.addWidget(self.chk_condition_advanced)

        self.ed_condition = QLineEdit()
        self.ed_condition.setPlaceholderText(_ct("ph_condition"))
        self.ed_condition.setToolTip(
            "사용 가능 변수: prev (직전 출력), context (누적 컨텍스트),\n"
            "output (=prev), outputs (key→출력 dict)"
        )
        self.ed_condition.setVisible(False)
        self.ed_condition.textChanged.connect(self._schedule_autosave)
        layout.addWidget(self.ed_condition)

        # [loop 노드 전용] 반복 횟수 + 바디 노드 선택
        self.loop_group = QGroupBox(_ct("lbl_loop"))
        self.loop_group.setVisible(False)
        self.loop_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #3f3f46;
                border-radius: 6px;
                color: #34D399;
                font-weight: bold;
                margin-top: 8px;
                padding-top: 6px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; }
        """)
        loop_layout = QVBoxLayout(self.loop_group)
        loop_layout.setSpacing(6)

        # ── 루프 패턴 라디오 (단순화) ───────────────────────────────
        from PySide6.QtWidgets import QRadioButton, QButtonGroup
        mode_label = QLabel("🔁 루프 패턴")
        mode_label.setStyleSheet("color: #a78bfa; font-weight: bold;")
        loop_layout.addWidget(mode_label)

        self.rb_loop_count    = QRadioButton("N번 반복 (가장 단순)")
        self.rb_loop_quality  = QRadioButton("결과가 충분히 좋아질 때까지")
        self.rb_loop_goal     = QRadioButton("목표를 달성할 때까지")
        self.rb_loop_advanced = QRadioButton("⚙️ 고급 (모든 옵션 직접)")
        for _rb in (self.rb_loop_count, self.rb_loop_quality,
                    self.rb_loop_goal, self.rb_loop_advanced):
            _rb.setStyleSheet("color: #e4e4e7; padding: 1px 0;")
            loop_layout.addWidget(_rb)
        self.rb_loop_count.setChecked(True)
        self._loop_mode_group = QButtonGroup(self)
        self._loop_mode_group.addButton(self.rb_loop_count)
        self._loop_mode_group.addButton(self.rb_loop_quality)
        self._loop_mode_group.addButton(self.rb_loop_goal)
        self._loop_mode_group.addButton(self.rb_loop_advanced)
        for _rb in (self.rb_loop_count, self.rb_loop_quality,
                    self.rb_loop_goal, self.rb_loop_advanced):
            _rb.toggled.connect(lambda _checked: self._on_loop_mode_changed())

        # 자연어 입력 (quality / goal 모드 공용)
        self.loop_quality_label = QLabel("")
        self.loop_quality_label.setStyleSheet("color: #fbbf24; font-weight: bold;")
        loop_layout.addWidget(self.loop_quality_label)
        self.loop_quality_text = QTextEdit()
        self.loop_quality_text.setFixedHeight(70)
        self.loop_quality_text.setStyleSheet("""
            QTextEdit { background:#27272a; border:1px solid #3f3f46;
                        border-radius:4px; color:#e4e4e7; padding:4px 6px; }
        """)
        self.loop_quality_text.textChanged.connect(self._schedule_autosave)
        loop_layout.addWidget(self.loop_quality_text)

        # ── 반복 횟수 spinbox (모든 모드에서 표시 — count 는 횟수, 그 외는 안전 상한) ──
        self.loop_count_label = QLabel(_ct("lbl_loop_count"))
        lc_row = QHBoxLayout()
        lc_row.addWidget(self.loop_count_label)
        self.loop_spin = QSpinBox()
        self.loop_spin.setRange(1, 100)
        self.loop_spin.setValue(3)
        self.loop_spin.setStyleSheet("""
            QSpinBox {
                background: #27272a; border: 1px solid #3f3f46;
                border-radius: 4px; color: #e4e4e7; padding: 3px 6px;
            }
        """)
        self.loop_spin.valueChanged.connect(self._on_loop_spin_changed)
        lc_row.addWidget(self.loop_spin)
        loop_layout.addLayout(lc_row)

        # 고급 모드일 때만 보이는 위젯 모음 (라벨/구분선/콤보 포함)
        self._loop_advanced_widgets: list = []

        loop_layout.addWidget(QLabel(_ct("lbl_body_nodes")))
        self.loop_body_list = QTextEdit()
        self.loop_body_list.setReadOnly(True)
        self.loop_body_list.setFixedHeight(80)
        self.loop_body_list.setStyleSheet("""
            QTextEdit {
                background: #18181b; border: 1px solid #3f3f46;
                border-radius: 4px; color: #a1a1aa; font-size: 11px;
            }
        """)
        loop_layout.addWidget(self.loop_body_list)

        # ── 노드 직접 선택 콤보박스 (캔버스 클릭 없이 추가 가능) ──────
        loop_layout.addWidget(QLabel(_ct("lbl_node_pick")))
        self.loop_node_picker = QComboBox()
        self.loop_node_picker.setStyleSheet("""
            QComboBox {
                background: #27272a; border: 1px solid #10B981;
                border-radius: 4px; color: #e4e4e7; padding: 3px 6px;
            }
            QComboBox QAbstractItemView {
                background: #27272a; color: #e4e4e7;
                selection-background-color: #064E3B;
            }
        """)
        loop_layout.addWidget(self.loop_node_picker)

        btn_row_loop = QHBoxLayout()
        self.btn_loop_add_sel = QPushButton(_ct("btn_loop_add_sel"))
        self.btn_loop_add_sel.clicked.connect(self._loop_add_selected_nodes)
        self.btn_loop_add_sel.setToolTip(
            "캔버스에서 노드를 클릭한 후 이 버튼을 눌러 루프 바디에 추가합니다."
        )
        self.btn_loop_add_picker = QPushButton(_ct("btn_loop_add_pick"))
        self.btn_loop_add_picker.clicked.connect(self._loop_add_from_picker)
        self.btn_loop_add_picker.setToolTip(
            "위 콤보박스에서 노드를 선택 후 이 버튼을 눌러 추가합니다."
        )
        self.btn_loop_clear = QPushButton(_ct("btn_loop_clear"))
        self.btn_loop_clear.clicked.connect(self._loop_clear_body)
        btn_row_loop.addWidget(self.btn_loop_add_sel)
        btn_row_loop.addWidget(self.btn_loop_add_picker)
        btn_row_loop.addWidget(self.btn_loop_clear)
        loop_layout.addLayout(btn_row_loop)

        # ── 자동 트리거 설정 ──────────────────────────────────────────
        self.lbl_loop_trigger = QLabel(_ct("lbl_trigger"))
        loop_layout.addWidget(self.lbl_loop_trigger)
        self._loop_advanced_widgets.append(self.lbl_loop_trigger)
        self.trigger_combo = QComboBox()
        self.trigger_combo.addItem("없음 (수동 실행)", "none")
        self.trigger_combo.addItem("⏰ 스케줄 (요일+시간)", "schedule")
        self.trigger_combo.addItem("🧠 자기 모델 (내부 상태)", "self_model")
        self.trigger_combo.setStyleSheet("""
            QComboBox {
                background: #27272a; border: 1px solid #3f3f46;
                border-radius: 4px; color: #e4e4e7; padding: 3px 6px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background: #27272a; border: 1px solid #52525b;
                color: #e4e4e7; selection-background-color: #3f3f46;
            }
        """)
        self.trigger_combo.currentIndexChanged.connect(self._on_trigger_type_changed)
        loop_layout.addWidget(self.trigger_combo)
        self._loop_advanced_widgets.append(self.trigger_combo)

        # 스케줄 설정 패널
        self.trigger_schedule_widget = QWidget()
        sched_layout = QVBoxLayout(self.trigger_schedule_widget)
        sched_layout.setContentsMargins(0, 0, 0, 0)
        sched_layout.setSpacing(4)

        sched_layout.addWidget(QLabel(_ct("lbl_weekdays")))
        self.trigger_weekdays_edit = QLineEdit()
        self.trigger_weekdays_edit.setPlaceholderText(_ct("ph_weekdays"))
        self.trigger_weekdays_edit.setStyleSheet("""
            QLineEdit { background:#27272a; border:1px solid #3f3f46;
                        border-radius:4px; color:#e4e4e7; padding:3px 6px; }
        """)
        sched_layout.addWidget(self.trigger_weekdays_edit)

        hour_row = QHBoxLayout()
        hour_row.addWidget(QLabel(_ct("lbl_hour_start")))
        self.trigger_hour_start = QSpinBox()
        self.trigger_hour_start.setRange(0, 23)
        self.trigger_hour_start.setValue(9)
        self.trigger_hour_start.setStyleSheet("""
            QSpinBox { background:#27272a; border:1px solid #3f3f46;
                       border-radius:4px; color:#e4e4e7; padding:2px 4px; }
        """)
        hour_row.addWidget(self.trigger_hour_start)
        hour_row.addWidget(QLabel(_ct("lbl_hour_end")))
        self.trigger_hour_end = QSpinBox()
        self.trigger_hour_end.setRange(0, 23)
        self.trigger_hour_end.setValue(10)
        self.trigger_hour_end.setStyleSheet("""
            QSpinBox { background:#27272a; border:1px solid #3f3f46;
                       border-radius:4px; color:#e4e4e7; padding:2px 4px; }
        """)
        hour_row.addWidget(self.trigger_hour_end)
        sched_layout.addLayout(hour_row)
        loop_layout.addWidget(self.trigger_schedule_widget)
        self.trigger_schedule_widget.setVisible(False)
        self._loop_advanced_widgets.append(self.trigger_schedule_widget)

        # 자기 모델 설정 패널
        self.trigger_sm_widget = QWidget()
        sm_layout = QVBoxLayout(self.trigger_sm_widget)
        sm_layout.setContentsMargins(0, 0, 0, 0)
        sm_layout.setSpacing(4)

        sm_layout.addWidget(QLabel(_ct("lbl_sm_key")))
        self.trigger_sm_key = QLineEdit()
        self.trigger_sm_key.setPlaceholderText(_ct("ph_sm_key"))
        self.trigger_sm_key.setStyleSheet("""
            QLineEdit { background:#27272a; border:1px solid #3f3f46;
                        border-radius:4px; color:#e4e4e7; padding:3px 6px; }
        """)
        sm_layout.addWidget(self.trigger_sm_key)

        sm_cmp_row = QHBoxLayout()
        sm_cmp_row.addWidget(QLabel(_ct("lbl_operator")))
        self.trigger_sm_op = QComboBox()
        for op in [">", ">=", "<", "<=", "=="]:
            self.trigger_sm_op.addItem(op)
        self.trigger_sm_op.setStyleSheet("""
            QComboBox { background:#27272a; border:1px solid #3f3f46;
                        border-radius:4px; color:#e4e4e7; padding:2px 6px; }
            QComboBox QAbstractItemView { background:#27272a; color:#e4e4e7; }
        """)
        sm_cmp_row.addWidget(self.trigger_sm_op)
        sm_cmp_row.addWidget(QLabel(_ct("lbl_threshold")))
        self.trigger_sm_threshold = QLineEdit()
        self.trigger_sm_threshold.setPlaceholderText(_ct("ph_threshold"))
        self.trigger_sm_threshold.setStyleSheet("""
            QLineEdit { background:#27272a; border:1px solid #3f3f46;
                        border-radius:4px; color:#e4e4e7; padding:3px 6px; }
        """)
        sm_cmp_row.addWidget(self.trigger_sm_threshold)
        sm_layout.addLayout(sm_cmp_row)
        loop_layout.addWidget(self.trigger_sm_widget)
        self.trigger_sm_widget.setVisible(False)
        self._loop_advanced_widgets.append(self.trigger_sm_widget)

        # 저장 버튼
        self.btn_save_trigger = QPushButton(_ct("btn_save_trigger"))
        self.btn_save_trigger.clicked.connect(self._save_trigger_settings)
        self.btn_save_trigger.setStyleSheet("""
            QPushButton {
                background: #065F46; border: none; border-radius: 4px;
                color: #6EE7B7; font-weight: bold; padding: 5px 10px;
            }
            QPushButton:hover { background: #047857; }
        """)
        loop_layout.addWidget(self.btn_save_trigger)
        self._loop_advanced_widgets.append(self.btn_save_trigger)

        # ── 조건 종료 (개선 #3) ──────────────────────────────────
        from PySide6.QtWidgets import QFrame
        self._loop_sep_exit = QFrame(); self._loop_sep_exit.setFrameShape(QFrame.HLine)
        self._loop_sep_exit.setStyleSheet("color: #3f3f46;")
        loop_layout.addWidget(self._loop_sep_exit)
        self._loop_advanced_widgets.append(self._loop_sep_exit)

        self.lbl_loop_exit_cond = QLabel(_ct("lbl_exit_cond"))
        loop_layout.addWidget(self.lbl_loop_exit_cond)
        self._loop_advanced_widgets.append(self.lbl_loop_exit_cond)
        self.loop_exit_cond_edit = QLineEdit()
        self.loop_exit_cond_edit.setPlaceholderText(
            "예: score > 0.9   /   iteration >= 5   /   '완료' in output"
        )
        self.loop_exit_cond_edit.setToolTip(
            "매 iteration 후 평가. True이면 조기 종료.\n"
            "사용 가능 변수: output(str), iteration(int), score(float - 첫 번째 숫자 자동 추출), i(=iteration-1)"
        )
        self.loop_exit_cond_edit.setStyleSheet("""
            QLineEdit { background:#27272a; border:1px solid #3f3f46;
                        border-radius:4px; color:#e4e4e7; padding:3px 6px; }
        """)
        loop_layout.addWidget(self.loop_exit_cond_edit)
        self._loop_advanced_widgets.append(self.loop_exit_cond_edit)

        self.lbl_loop_max_iter = QLabel(_ct("lbl_max_iter"))
        max_iter_row = QHBoxLayout()
        max_iter_row.addWidget(self.lbl_loop_max_iter)
        self.loop_max_iter_spin = QSpinBox()
        self.loop_max_iter_spin.setRange(0, 1000)
        self.loop_max_iter_spin.setValue(0)
        self.loop_max_iter_spin.setStyleSheet("""
            QSpinBox { background:#27272a; border:1px solid #3f3f46;
                       border-radius:4px; color:#e4e4e7; padding:2px 4px; }
        """)
        max_iter_row.addWidget(self.loop_max_iter_spin)
        loop_layout.addLayout(max_iter_row)
        self._loop_advanced_widgets.append(self.lbl_loop_max_iter)
        self._loop_advanced_widgets.append(self.loop_max_iter_spin)

        # ── 루프 메모리 설정 (❹) ────────────────────────────────
        from PySide6.QtWidgets import QFrame as _QFrame
        self._loop_sep_mem = _QFrame(); self._loop_sep_mem.setFrameShape(_QFrame.HLine)
        self._loop_sep_mem.setStyleSheet("color: #3f3f46;")
        loop_layout.addWidget(self._loop_sep_mem)
        self._loop_advanced_widgets.append(self._loop_sep_mem)

        self.lbl_loop_mem_mode = QLabel(_ct("lbl_mem_mode"))
        loop_layout.addWidget(self.lbl_loop_mem_mode)
        self._loop_advanced_widgets.append(self.lbl_loop_mem_mode)
        self.loop_memory_mode_combo = QComboBox()
        self.loop_memory_mode_combo.addItem("사용 안 함",                        "none")
        self.loop_memory_mode_combo.addItem("누적 (append) — {{memory}}로 참조",  "append")
        self.loop_memory_mode_combo.addItem("덮어쓰기 (replace)",                 "replace")
        self.loop_memory_mode_combo.addItem("LLM 압축 요약 (llm_compress)",       "llm_compress")
        self.loop_memory_mode_combo.setToolTip(
            "루프 바디 프롬프트에서 {{memory}}로 이전 iteration 학습 내용을 참조합니다."
        )
        self.loop_memory_mode_combo.setStyleSheet("""
            QComboBox { background:#27272a; border:1px solid #3f3f46;
                        border-radius:4px; color:#e4e4e7; padding:3px 6px; }
            QComboBox QAbstractItemView { background:#27272a; color:#e4e4e7; }
        """)
        loop_layout.addWidget(self.loop_memory_mode_combo)
        self._loop_advanced_widgets.append(self.loop_memory_mode_combo)

        self.lbl_loop_compress = QLabel(_ct("lbl_compress"))
        compress_row = QHBoxLayout()
        compress_row.addWidget(self.lbl_loop_compress)
        self.loop_compress_spin = QSpinBox()
        self.loop_compress_spin.setRange(1, 100)
        self.loop_compress_spin.setValue(3)
        self.loop_compress_spin.setStyleSheet("""
            QSpinBox { background:#27272a; border:1px solid #3f3f46;
                       border-radius:4px; color:#e4e4e7; padding:2px 4px; }
        """)
        compress_row.addWidget(self.loop_compress_spin)
        loop_layout.addLayout(compress_row)
        self._loop_advanced_widgets.append(self.lbl_loop_compress)
        self._loop_advanced_widgets.append(self.loop_compress_spin)

        layout.addWidget(self.loop_group)

        layout.addWidget(QLabel(_ct("lbl_last_out")))
        self.ed_last_output = QTextEdit()
        self.ed_last_output.setReadOnly(True)
        self.ed_last_output.setFixedHeight(140)
        layout.addWidget(self.ed_last_output)

        btn_row = QHBoxLayout()
        self.btn_apply = QPushButton(_ct("btn_apply"))
        self.btn_apply.clicked.connect(self._apply_inspector_to_node)
        btn_row.addWidget(self.btn_apply)

        self.btn_run_from_here = QPushButton(_ct("btn_run_here"))
        self.btn_run_from_here.clicked.connect(self._run_from_selected)
        btn_row.addWidget(self.btn_run_from_here)

        layout.addLayout(btn_row)

        # ── AgentState Goal 패널 (② 핵심) ─────────────────────────
        from PySide6.QtWidgets import QFrame
        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("color: #3f3f46; margin: 4px 0;")
        layout.addWidget(sep2)

        goal_label = QLabel(_ct("lbl_goal"))
        goal_label.setStyleSheet("color: #a78bfa; font-weight: bold; font-size: 12px;")
        layout.addWidget(goal_label)

        self.goal_edit = QLineEdit()
        self.goal_edit.setPlaceholderText(_ct("ph_goal"))
        self.goal_edit.setStyleSheet("""
            QLineEdit { background:#27272a; border:1px solid #7c3aed;
                        border-radius:6px; color:#e4e4e7; padding:5px 8px; }
        """)
        layout.addWidget(self.goal_edit)

        layout.addWidget(QLabel(_ct("lbl_goal_inject")))
        self.goal_mode_combo = QComboBox()
        self.goal_mode_combo.addItem("주입 안 함",       "none")
        self.goal_mode_combo.addItem("모든 노드 앞에 삽입 (prefix)", "prefix")
        self.goal_mode_combo.addItem("{{goal}} 변수로만",  "var")
        self.goal_mode_combo.setStyleSheet("""
            QComboBox { background:#27272a; border:1px solid #3f3f46;
                        border-radius:4px; color:#e4e4e7; padding:3px 6px; }
            QComboBox QAbstractItemView { background:#27272a; color:#e4e4e7; }
        """)
        layout.addWidget(self.goal_mode_combo)

        layout.addWidget(QLabel(_ct("lbl_dag_style")))
        self.planner_style_combo = QComboBox()
        self.planner_style_combo.addItem("순차 (Sequential)",  "sequential")
        self.planner_style_combo.addItem("병렬 (Parallel)",    "parallel")
        self.planner_style_combo.addItem("자동 선택 (Auto)",   "auto")
        self.planner_style_combo.setStyleSheet("""
            QComboBox { background:#27272a; border:1px solid #4338ca;
                        border-radius:4px; color:#a5b4fc; padding:3px 6px; }
            QComboBox QAbstractItemView { background:#27272a; color:#a5b4fc; }
        """)
        layout.addWidget(self.planner_style_combo)

        goal_status_row = QHBoxLayout()
        self.goal_progress_label = QLabel(_ct("lbl_progress"))
        self.goal_progress_label.setStyleSheet("color: #6ee7b7; font-size: 11px;")
        goal_status_row.addWidget(self.goal_progress_label)
        goal_status_row.addStretch()
        self.goal_run_count_label = QLabel(_ct("lbl_run_count"))
        self.goal_run_count_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        goal_status_row.addWidget(self.goal_run_count_label)
        layout.addLayout(goal_status_row)

        btn_goal_save = QPushButton(_ct("btn_goal_save"))
        btn_goal_save.clicked.connect(self._save_goal_settings)
        btn_goal_save.setStyleSheet("""
            QPushButton { background:#4c1d95; border:none; border-radius:4px;
                          color:#c4b5fd; font-weight:bold; padding:5px 10px; }
            QPushButton:hover { background:#5b21b6; }
        """)
        layout.addWidget(btn_goal_save)

        hint = QLabel(_ct("hint"))
        hint.setStyleSheet("color: #777; font-size: 11px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # connect autosave
        self.ed_title.textChanged.connect(self._schedule_autosave)
        self.ed_prompt.textChanged.connect(self._schedule_autosave)

        return panel

    def _apply_scene_background_grid(self):
        # Simple dark grid via background brush is not native in QGraphicsScene.
        # We'll just style the view; optional future: override drawBackground for custom grid.
        pass

    def _setup_actions(self):
        tb = QToolBar("Prompt Canvas")
        self.addToolBar(tb)

        btn_add = QPushButton(_ct("btn_add_prompt"))
        btn_add.clicked.connect(lambda: self.add_node(node_type="prompt"))
        tb.addWidget(btn_add)
        
        btn_add_tool = QPushButton(_ct("btn_add_tool"))
        btn_add_tool.clicked.connect(lambda: self.add_node(node_type="tool"))
        tb.addWidget(btn_add_tool)

        btn_add_if = QPushButton(_ct("btn_add_if"))
        btn_add_if.clicked.connect(lambda: self.add_node(node_type="if"))
        tb.addWidget(btn_add_if)

        btn_add_loop = QPushButton(_ct("btn_add_loop"))
        btn_add_loop.clicked.connect(self._add_loop_node)
        tb.addWidget(btn_add_loop)

        tb.addSeparator()

        tb.addWidget(QLabel(_ct("lbl_start_input")))
        self.start_input_edit = QLineEdit()
        self.start_input_edit.setPlaceholderText(_ct("ph_start_input"))
        self.start_input_edit.setMinimumWidth(220)
        self.start_input_edit.setStyleSheet("""
            QLineEdit {
                background: #27272a;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                color: #e4e4e7;
                padding: 5px 10px;
                height: 30px;
            }
        """)
        tb.addWidget(self.start_input_edit)

        btn_run = QPushButton(_ct("btn_run"))
        btn_run.clicked.connect(self.run_agent)
        tb.addWidget(btn_run)

        btn_save = QPushButton(_ct("btn_save"))
        btn_save.clicked.connect(self.save_to_file)
        tb.addWidget(btn_save)

        btn_load = QPushButton(_ct("btn_load"))
        btn_load.clicked.connect(self.load_from_file)
        tb.addWidget(btn_load)

        btn_clear = QPushButton(_ct("btn_clear"))
        btn_clear.clicked.connect(self.clear_all)
        tb.addWidget(btn_clear)

        tb.addSeparator()

        btn_create_agent = QPushButton(_ct("btn_agent"))
        btn_create_agent.clicked.connect(self._create_agent_from_canvas)
        tb.addWidget(btn_create_agent)

        btn_plan = QPushButton(_ct("btn_plan"))
        btn_plan.clicked.connect(self._run_planner_ui)
        btn_plan.setStyleSheet("""
            QPushButton {
                background: #1e1b4b;
                border: 1px solid #4338ca;
                border-radius: 5px;
                color: #a5b4fc;
                padding: 4px 10px;
                font-weight: bold;
            }
            QPushButton:hover { background: #312e81; }
        """)
        tb.addWidget(btn_plan)

        # Verb 시퀀스 → 자동 조립 + 즉시 실행
        btn_verb_pipe = QPushButton("🔀 Verb 시퀀스")
        btn_verb_pipe.setToolTip(
            "Verb 시퀀스 (예: 買う,比較する,評価する,推薦する) 입력 → "
            "각 verb 를 prompt 노드로 자동 조립 → 즉시 실행."
        )
        btn_verb_pipe.clicked.connect(self._run_verb_pipeline_ui)
        btn_verb_pipe.setStyleSheet("""
            QPushButton {
                background: #064e3b;
                border: 1px solid #059669;
                border-radius: 5px;
                color: #6ee7b7;
                padding: 4px 10px;
                font-weight: bold;
            }
            QPushButton:hover { background: #065f46; }
        """)
        tb.addWidget(btn_verb_pipe)

        self.setStatusBar(QStatusBar())

    # -------------------------
    # -------------------------
    # Theme
    # -------------------------

    def apply_theme(self, theme_name: str):
        """eidos_chat_gui.py\xec\x9d\x98 apply_theme\xec\x97\x90\xec\x84\x9c \xed\x98\xb8\xec\xb6\x9c\xeb\x90\xa8 - \xeb\x9d\xbc\xec\x9d\xb4\xed\x8a\xb8/\xeb\x8b\xa4\xed\x81\xac \xeb\xb6\x84\xea\xb8\xb0 QSS \xec\xa0\x81\xec\x9a\xa9"""
        from eidos_chat_gui import THEMES, MODERN_DARK_THEME
        theme_obj = THEMES.get(theme_name, MODERN_DARK_THEME)
        is_dark   = theme_obj.get('is_dark', True)
        c         = theme_obj['colors']

        if is_dark:
            panel_bg     = '#18181b'
            panel_text   = '#e4e4e7'
            label_col    = '#a1a1aa'
            input_bg     = '#27272a'
            border_col   = '#3f3f46'
            input_text   = '#e4e4e7'
            hint_col     = '#777777'
            node_id_col  = '#666666'
        else:
            panel_bg     = c.get('bg_raised',  '#f1f5f9')
            panel_text   = c.get('text_primary', '#334155')
            label_col    = c.get('text_secondary', '#64748b')
            input_bg     = c.get('bg_surface', '#ffffff')
            border_col   = c.get('border_subtle', '#e2e8f0')
            input_text   = c.get('text_primary', '#334155')
            hint_col     = c.get('text_hint', '#94a3b8')
            node_id_col  = c.get('text_hint', '#94a3b8')

        inspector_qss = f"""
            QWidget {{ background-color: {panel_bg}; color: {panel_text}; }}
            QLabel  {{ color: {label_col}; }}
            QLineEdit, QTextEdit {{
                background: {input_bg};
                border: 1px solid {border_col};
                border-radius: 6px;
                color: {input_text};
                padding: 5px 8px;
            }}
            QPushButton {{
                background: transparent;
                border: 1px solid {border_col};
                border-radius: 6px;
                color: {input_text};
                padding: 5px 12px;
            }}
            QPushButton:hover {{ background: {input_bg}; }}
        """
        self.inspector.setStyleSheet(inspector_qss)

        toolbar_input_qss = f"""
            QLineEdit {{
                background: {input_bg};
                border: 1px solid {border_col};
                border-radius: 6px;
                color: {input_text};
                padding: 5px 10px;
                height: 30px;
            }}
        """
        self.start_input_edit.setStyleSheet(toolbar_input_qss)

        if hasattr(self, 'inspector_node_id'):
            self.inspector_node_id.setStyleSheet(f'color: {node_id_col}; font-size: 11px;')

        # canvas view background/grid
        if hasattr(self, 'view'):
            self.view.set_theme(is_dark)

        # toolbar + window QSS
        tb_text   = '#e4e4e7' if is_dark else panel_text
        tb_bg     = '#18181b' if is_dark else c.get('bg_raised', '#f1f5f9')
        tb_border = '#3f3f46' if is_dark else border_col
        win_qss = f"""
            QMainWindow, QToolBar {{
                background-color: {tb_bg};
                color: {tb_text};
                border: none;
            }}
            QToolBar QLabel {{ color: {tb_text}; background: transparent; }}
            QToolBar QPushButton {{
                background: transparent;
                border: 1px solid {tb_border};
                border-radius: 5px;
                color: {tb_text};
                padding: 4px 10px;
            }}
            QToolBar QPushButton:hover {{ background: {input_bg}; }}
            QStatusBar {{ background: {tb_bg}; color: {tb_text}; }}
        """
        self.setStyleSheet(win_qss)

    # -------------------------
    # Node operations
    # -------------------------

    def add_node(self, node_type: NodeType = "prompt", pos: Optional[QPointF] = None):
        node_id = str(uuid.uuid4())
        if pos is None:
            center = self.view.mapToScene(self.view.viewport().rect().center())
            pos = center + QPointF(20, 20)

        if node_type == "tool":
            title = "새 도구"
            prompt = ""
        elif node_type == "if":
            title = "조건 분기"
            prompt = ""
        else: # prompt
            title = "새 프롬프트"
            prompt = ""

        data = PromptNodeData(
            node_id=node_id,
            title=title,
            prompt=prompt,
            x=float(pos.x()),
            y=float(pos.y()),
            node_type=node_type,
        )
        self.nodes[node_id] = data

        item = PromptBoxItem(data)
        item.setPos(pos)
        item.request_edit.connect(self.open_in_inspector)
        item.moved.connect(lambda _nid: self._mark_dirty())
        self.scene.addItem(item)

        self.node_items[node_id] = item
        self._mark_dirty()

        self.statusBar().showMessage("노드 생성 완료", 1500)
        return node_id

    def open_in_inspector(self, node_id: str):
        if self.current_node_id == node_id:
            return

        # ── loop 노드인지 먼저 확인 ──────────────────────────────────
        if node_id in self.loop_nodes:
            self._open_loop_in_inspector(node_id)
            return

        if node_id not in self.nodes:
            self._clear_inspector()
            return

        self._cancel_autosave()
        self.current_node_id = node_id
        data = self.nodes[node_id]
        self.inspector_node_id.setText(_ct("selected") + str(node_id))

        self.ed_title.blockSignals(True)
        self.ed_prompt.blockSignals(True)
        self.ed_last_output.blockSignals(True)

        self.ed_title.setText(str(data.title or ""))

        # 루프 전용 위젯 숨김
        self.loop_group.setVisible(False)

        if data.node_type == "tool":
            self.prompt_label.setVisible(False)
            self.ed_prompt.setVisible(False)
            self.tool_combo_label.setVisible(True)
            self.tool_combo.setVisible(True)
            self.tool_combo.blockSignals(True)
            current_tool = str(data.prompt or "")
            found = False
            for i in range(self.tool_combo.count()):
                if self.tool_combo.itemData(i) == current_tool:
                    self.tool_combo.setCurrentIndex(i)
                    found = True
                    break
            if not found:
                self.tool_combo.setCurrentIndex(0)
            self.tool_combo.blockSignals(False)
            self.condition_label.setVisible(False)
            self.ed_condition.setVisible(False)
            self.ed_condition_natural.setVisible(False)
            self.chk_condition_advanced.setVisible(False)
        elif data.node_type == "if":
            self.prompt_label.setVisible(True)
            self.ed_prompt.setVisible(True)
            self.tool_combo_label.setVisible(False)
            self.tool_combo.setVisible(False)
            self.prompt_label.setText(_ct("lbl_prompt_desc"))
            self.ed_prompt.setPlaceholderText(_ct("ph_prompt_desc"))
            self.condition_label.setVisible(True)

            mode = (getattr(data, "condition_mode", "natural") or "natural").lower()
            is_advanced = (mode == "expression")

            self.chk_condition_advanced.setVisible(True)
            self.chk_condition_advanced.blockSignals(True)
            self.chk_condition_advanced.setChecked(is_advanced)
            self.chk_condition_advanced.blockSignals(False)

            cond_text = str(data.condition or "")
            self.ed_condition_natural.blockSignals(True)
            self.ed_condition_natural.setPlainText("" if is_advanced else cond_text)
            self.ed_condition_natural.blockSignals(False)
            self.ed_condition.blockSignals(True)
            self.ed_condition.setText(cond_text if is_advanced else "")
            self.ed_condition.blockSignals(False)

            self.ed_condition_natural.setVisible(not is_advanced)
            self.ed_condition.setVisible(is_advanced)
        else:  # prompt
            self.prompt_label.setVisible(True)
            self.ed_prompt.setVisible(True)
            self.tool_combo_label.setVisible(False)
            self.tool_combo.setVisible(False)
            self.prompt_label.setText(_ct("lbl_prompt"))
            self.ed_prompt.setPlaceholderText(self.original_prompt_placeholder)
            self.condition_label.setVisible(False)
            self.ed_condition.setVisible(False)
            self.ed_condition_natural.setVisible(False)
            self.chk_condition_advanced.setVisible(False)

        self.ed_prompt.setPlainText(str(data.prompt or ""))
        self.ed_last_output.setPlainText(str(data.last_output or ""))

        self.ed_title.blockSignals(False)
        self.ed_prompt.blockSignals(False)
        self.ed_last_output.blockSignals(False)

        self.statusBar().showMessage(f"'{data.title}' 노드 선택됨 (자동 저장 활성)", 2000)

    def _open_loop_in_inspector(self, node_id: str):
        """루프 노드 전용 인스펙터 표시."""
        self._cancel_autosave()
        self.current_node_id = node_id
        ldata = self.loop_nodes[node_id]
        self.inspector_node_id.setText(f"선택: {node_id}")

        self.ed_title.blockSignals(True)
        self.ed_title.setText(str(ldata.title or ""))
        self.ed_title.blockSignals(False)

        # 일반 위젯 숨김
        self.prompt_label.setVisible(False)
        self.ed_prompt.setVisible(False)
        self.tool_combo_label.setVisible(False)
        self.tool_combo.setVisible(False)
        self.condition_label.setVisible(False)
        self.ed_condition.setVisible(False)
        self.ed_condition_natural.setVisible(False)
        self.chk_condition_advanced.setVisible(False)

        # 루프 전용 표시
        self.loop_group.setVisible(True)
        self.loop_spin.blockSignals(True)
        self.loop_spin.setValue(ldata.loop_count)
        self.loop_spin.blockSignals(False)
        self._refresh_loop_body_list(node_id)

        # 루프 모드 라디오 복원
        mode = (getattr(ldata, "loop_mode", "count") or "count").lower()
        rb_map = {
            "count":    self.rb_loop_count,
            "quality":  self.rb_loop_quality,
            "goal":     self.rb_loop_goal,
            "advanced": self.rb_loop_advanced,
        }
        target_rb = rb_map.get(mode, self.rb_loop_count)
        for _rb in (self.rb_loop_count, self.rb_loop_quality,
                    self.rb_loop_goal, self.rb_loop_advanced):
            _rb.blockSignals(True)
        target_rb.setChecked(True)
        for _rb in (self.rb_loop_count, self.rb_loop_quality,
                    self.rb_loop_goal, self.rb_loop_advanced):
            _rb.blockSignals(False)

        # 자연어 입력 복원 — quality/goal 모드 시 exit_condition 의 'llm:' prefix 벗겨내고 표시
        ec = (getattr(ldata, "exit_condition", "") or "").strip()
        nl_text = ec[4:].strip() if ec.lower().startswith("llm:") else ec
        self.loop_quality_text.blockSignals(True)
        self.loop_quality_text.setPlainText(nl_text if mode in ("quality", "goal") else "")
        self.loop_quality_text.blockSignals(False)

        # 트리거 설정 복원
        ttype = getattr(ldata, 'auto_trigger_type', 'none')
        for i in range(self.trigger_combo.count()):
            if self.trigger_combo.itemData(i) == ttype:
                self.trigger_combo.blockSignals(True)
                self.trigger_combo.setCurrentIndex(i)
                self.trigger_combo.blockSignals(False)
                break
        self._refresh_trigger_panels(ttype)

        params = ldata.auto_trigger_params or {}
        if ttype == 'schedule':
            wdays = params.get('weekdays', [])
            self.trigger_weekdays_edit.setText(",".join(str(d) for d in wdays))
            self.trigger_hour_start.setValue(int(params.get('hour_start', 9)))
            self.trigger_hour_end.setValue(int(params.get('hour_end', 10)))
        elif ttype == 'self_model':
            self.trigger_sm_key.setText(params.get('key', ''))
            op = params.get('op', '>')
            idx = self.trigger_sm_op.findText(op)
            if idx >= 0:
                self.trigger_sm_op.setCurrentIndex(idx)
            self.trigger_sm_threshold.setText(str(params.get('threshold', '0.7')))

        self.ed_last_output.setPlainText(str(ldata.last_output or ""))

        # 조건 종료 복원 (개선 #3)
        self.loop_exit_cond_edit.setText(getattr(ldata, 'exit_condition', '') or '')
        self.loop_max_iter_spin.setValue(getattr(ldata, 'max_iterations', 0) or 0)

        # 루프 메모리 복원 (❹)
        mem_mode = getattr(ldata, 'memory_mode', 'none')
        for i in range(self.loop_memory_mode_combo.count()):
            if self.loop_memory_mode_combo.itemData(i) == mem_mode:
                self.loop_memory_mode_combo.setCurrentIndex(i)
                break
        self.loop_compress_spin.setValue(getattr(ldata, 'memory_compress_every', 3) or 3)

        # 모드별 visibility 갱신 (라디오/자연어/고급 위젯)
        self._refresh_loop_mode_visibility()

        # 노드 피커 콤보박스 갱신
        self._refresh_loop_node_picker(node_id)

        self.statusBar().showMessage(
            f"'{ldata.title}' 루프 편집 중 — 노드 클릭 후 [✚ 캔버스 선택 추가], "
            f"또는 콤보박스에서 [✚ 목록에서 추가]", 0
        )

    def _on_selection_changed(self):
        selected_items = self.scene.selectedItems()
        prompt_nodes = [i for i in selected_items if isinstance(i, PromptBoxItem)]

        # ── 루프 인스펙터가 열려있으면 인스펙터 전환 차단 ──────────────
        # 루프 바디 노드 추가 모드: 캔버스 선택이 인스펙터를 덮어쓰지 않음
        if self.current_node_id and self.current_node_id in self.loop_nodes:
            # 선택된 프롬프트 노드를 "추가 대기" 상태로 하이라이트만 표시
            count = len(prompt_nodes)
            if count > 0:
                titles = ", ".join(it.data.title for it in prompt_nodes[:3])
                suffix = f" 외 {count-3}개" if count > 3 else ""
                self.statusBar().showMessage(
                    f"✚ '{titles}{suffix}' 선택됨 — [선택 노드 추가] 버튼을 클릭하세요", 0
                )
            else:
                self.statusBar().showMessage("루프 바디 편집 중 — 노드를 클릭 후 [선택 노드 추가]", 0)
            return  # 인스펙터 전환 없이 여기서 종료

        # ── 일반 모드 ─────────────────────────────────────────────────
        if len(prompt_nodes) == 1:
            node_item = prompt_nodes[0]
            self.open_in_inspector(node_item.data.node_id)
        else:
            self._cancel_autosave()
            self._clear_inspector()
            if len(prompt_nodes) > 1:
                self.inspector_node_id.setText(_ct("selected") + f"({len(prompt_nodes)}" + _ct("selected_multi"))

    def _cancel_autosave(self):
        if self._autosave_timer.isActive():
            self._autosave_timer.stop()

    def _clear_inspector(self):
        """ 인스펙터 패널의 내용을 모두 지웁니다. """
        self.current_node_id = None
        self.inspector_node_id.setText(_ct("inspector_none"))

        # blockSignals를 사용해 불필요한 autosave를 방지합니다.
        self.ed_title.blockSignals(True)
        self.ed_prompt.blockSignals(True)
        self.ed_last_output.blockSignals(True)

        self.ed_title.setText("")
        self.ed_prompt.setPlainText("")
        self.ed_last_output.setPlainText("")

        self.ed_title.blockSignals(False)
        self.ed_prompt.blockSignals(False)
        self.ed_last_output.blockSignals(False)

        # tool 전용 위젯 숨김
        self.tool_combo_label.setVisible(False)
        self.tool_combo.setVisible(False)
        self.prompt_label.setVisible(True)
        self.ed_prompt.setVisible(True)
        # if 전용 위젯 숨김
        self.condition_label.setVisible(False)
        self.ed_condition.setVisible(False)
        self.ed_condition_natural.setVisible(False)
        self.chk_condition_advanced.setVisible(False)
        # loop 전용 위젯 숨김
        self.loop_group.setVisible(False)

    # ──────────────────────────────────────────────────────────────
    # Loop node helpers
    # ──────────────────────────────────────────────────────────────

    def _add_loop_node(self):
        node_id = str(uuid.uuid4())
        center  = self.view.mapToScene(self.view.viewport().rect().center())
        pos     = center + QPointF(20, 20)
        ldata   = LoopNodeData(
            node_id=node_id, title="루프",
            x=float(pos.x()), y=float(pos.y()),
            loop_count=3, body_node_ids=[], body_edge_ids=[],
        )
        self.loop_nodes[node_id] = ldata
        item = LoopBoxItem(ldata)
        item.setPos(pos)
        item.request_edit.connect(self.open_in_inspector)
        item.moved.connect(lambda _nid: self._mark_dirty())
        item.request_delete.connect(self._delete_loop_node)
        self.scene.addItem(item)
        self.loop_items[node_id] = item
        self._mark_dirty()
        self.statusBar().showMessage("루프 노드 생성 완료", 1500)

    def _delete_loop_node(self, node_id: str):
        del_eids = [eid for eid, e in self.edges.items()
                    if e.src_id == node_id or e.dst_id == node_id]
        for eid in del_eids:
            ei = self.edge_items.pop(eid, None)
            if ei: self.scene.removeItem(ei)
            self.edges.pop(eid, None)
        item = self.loop_items.pop(node_id, None)
        if item: self.scene.removeItem(item)
        self.loop_nodes.pop(node_id, None)
        if self.current_node_id == node_id:
            self._clear_inspector()
        self._mark_dirty()

    def _refresh_loop_body_list(self, loop_id: str):
        ldata = self.loop_nodes.get(loop_id)
        if not ldata:
            return
        lines = []
        for nid in ldata.body_node_ids:
            nd = self.nodes.get(nid)
            label = nd.title if nd else f"(삭제됨: {nid[:8]})"
            lines.append(f"• {label}")
        self.loop_body_list.setPlainText("\n".join(lines) if lines else "(비어 있음)")

    def _refresh_loop_node_picker(self, loop_id: str):
        """루프 인스펙터의 노드 피커 콤보박스를 현재 캔버스 노드 목록으로 갱신."""
        if not hasattr(self, 'loop_node_picker'):
            return
        self.loop_node_picker.clear()
        ldata = self.loop_nodes.get(loop_id)
        already_in = set(ldata.body_node_ids) if ldata else set()

        self.loop_node_picker.addItem("— 추가할 노드 선택 —", None)
        for nid, nd in self.nodes.items():
            if nid == loop_id:
                continue
            marker = " ✓" if nid in already_in else ""
            self.loop_node_picker.addItem(f"{nd.title}{marker}", nid)

    def _loop_add_selected_nodes(self):
        """캔버스에서 선택된 노드를 루프 바디에 추가."""
        loop_id = self.current_node_id
        if not loop_id or loop_id not in self.loop_nodes:
            QMessageBox.information(self, "루프", "먼저 루프 노드를 더블클릭하여 선택하세요.")
            return
        ldata    = self.loop_nodes[loop_id]
        selected = [it for it in self.scene.selectedItems()
                    if isinstance(it, PromptBoxItem)]
        if not selected:
            QMessageBox.information(
                self, "루프",
                "캔버스에서 추가할 노드를 먼저 클릭한 뒤\n"
                "[✚ 캔버스 선택 추가] 버튼을 누르세요.\n\n"
                "또는 위 콤보박스에서 노드를 선택하고\n"
                "[✚ 목록에서 추가] 버튼을 사용하세요."
            )
            return
        added = self._add_nodes_to_loop(loop_id, ldata, [it.data.node_id for it in selected])
        self.statusBar().showMessage(f"노드 {added}개 루프 바디에 추가됨", 2000)

    def _loop_add_from_picker(self):
        """콤보박스에서 선택된 노드를 루프 바디에 추가."""
        loop_id = self.current_node_id
        if not loop_id or loop_id not in self.loop_nodes:
            return
        nid = self.loop_node_picker.currentData()
        if not nid:
            QMessageBox.information(self, "루프", "콤보박스에서 추가할 노드를 선택하세요.")
            return
        ldata = self.loop_nodes[loop_id]
        added = self._add_nodes_to_loop(loop_id, ldata, [nid])
        if added:
            self.statusBar().showMessage(
                f"'{self.nodes[nid].title}' 루프 바디에 추가됨", 2000
            )
            # 콤보박스 갱신 (✓ 마커 반영)
            self._refresh_loop_node_picker(loop_id)
        else:
            self.statusBar().showMessage("이미 추가된 노드입니다.", 1500)

    def _add_nodes_to_loop(self, loop_id: str, ldata, node_ids: list) -> int:
        """노드 ID 목록을 루프 바디에 추가하는 공통 로직. 추가된 수 반환."""
        added = 0
        for nid in node_ids:
            if nid != loop_id and nid not in ldata.body_node_ids:
                ldata.body_node_ids.append(nid)
                added += 1
        # 바디 노드 간 엣지 자동 등록
        body_set = set(ldata.body_node_ids)
        for eid, e in self.edges.items():
            if e.src_id in body_set and e.dst_id in body_set:
                if eid not in ldata.body_edge_ids:
                    ldata.body_edge_ids.append(eid)
        ldata.body_node_ids = self._topo_sort_body(ldata.body_node_ids, ldata.body_edge_ids)
        self._refresh_loop_body_list(loop_id)
        item = self.loop_items.get(loop_id)
        if item:
            item.update()
        self._mark_dirty()
        return added

    def _loop_clear_body(self):
        loop_id = self.current_node_id
        if not loop_id or loop_id not in self.loop_nodes:
            return
        ldata = self.loop_nodes[loop_id]
        ldata.body_node_ids.clear()
        ldata.body_edge_ids.clear()
        self._refresh_loop_body_list(loop_id)
        self._refresh_loop_node_picker(loop_id)   # ✓ 마커 제거
        item = self.loop_items.get(loop_id)
        if item: item.update()
        self._mark_dirty()
        self.statusBar().showMessage("루프 바디 초기화됨", 1500)

    def _on_loop_spin_changed(self, value: int):
        if not self.current_node_id or self.current_node_id not in self.loop_nodes:
            return
        self.loop_nodes[self.current_node_id].loop_count = value
        item = self.loop_items.get(self.current_node_id)
        if item: item.update()
        self._mark_dirty()

    # ── 자동 트리거 UI 헬퍼 ───────────────────────────────────────────────

    def _on_trigger_type_changed(self, index: int):
        ttype = self.trigger_combo.itemData(index)
        self._refresh_trigger_panels(ttype)

    def _refresh_trigger_panels(self, ttype: str):
        self.trigger_schedule_widget.setVisible(ttype == 'schedule')
        self.trigger_sm_widget.setVisible(ttype == 'self_model')

    def _save_trigger_settings(self):
        nid = self.current_node_id
        if not nid or nid not in self.loop_nodes:
            return
        ldata  = self.loop_nodes[nid]
        ttype  = self.trigger_combo.itemData(self.trigger_combo.currentIndex())
        ldata.auto_trigger_type = ttype

        if ttype == 'schedule':
            raw = self.trigger_weekdays_edit.text().strip()
            try:
                weekdays = [int(x.strip()) for x in raw.split(',') if x.strip()]
                weekdays = [d for d in weekdays if 0 <= d <= 6]
            except ValueError:
                weekdays = []
            ldata.auto_trigger_params = {
                "weekdays":   weekdays,
                "hour_start": self.trigger_hour_start.value(),
                "hour_end":   self.trigger_hour_end.value(),
            }
        elif ttype == 'self_model':
            try:
                threshold = float(self.trigger_sm_threshold.text().strip())
            except ValueError:
                threshold = 0.0
            ldata.auto_trigger_params = {
                "key":       self.trigger_sm_key.text().strip(),
                "op":        self.trigger_sm_op.currentText(),
                "threshold": threshold,
            }
        else:
            ldata.auto_trigger_params = {}

        # 캔버스 노드 아이콘 갱신
        item = self.loop_items.get(nid)
        if item:
            item.update()
        self._mark_dirty()
        self.statusBar().showMessage(
            f"트리거 설정 저장됨 [{ldata.title}]: {ttype}", 2000
        )

    # ── Goal 설정 (② 핵심) ──────────────────────────────────────────────

    def _save_goal_settings(self):
        """인스펙터 Goal 패널의 값을 AgentState에 저장."""
        self.agent_state.goal        = self.goal_edit.text().strip()
        self.agent_state.goal_prompt = self.goal_mode_combo.itemData(
            self.goal_mode_combo.currentIndex()
        )
        # planner_style도 함께 저장
        self.agent_state.planner_style = self.planner_style_combo.itemData(
            self.planner_style_combo.currentIndex()
        )
        self._refresh_goal_panel()
        self._mark_dirty()
        self.statusBar().showMessage(
            f"목표 저장됨: {self.agent_state.goal[:40]!r} "
            f"[주입: {self.agent_state.goal_prompt}]", 2500
        )

    def _run_planner_ui(self):
        """
        툴바 '🗺️ 목표로 DAG 생성' 버튼 핸들러.
        Goal 패널의 goal 텍스트 → Planner → 캔버스에 DAG 로드.
        """
        goal = self.agent_state.goal.strip()
        if not goal:
            # Goal 패널에 설정이 없으면 다이얼로그로 입력받기
            goal, ok = QInputDialog.getText(
                self, "DAG 자동 생성",
                "에이전트 목표를 입력하세요:\n(예: 주어진 코드의 버그를 찾고 수정안을 제안한다)"
            )
            if not ok or not goal.strip():
                return
            goal = goal.strip()
            self.agent_state.goal = goal
            self._refresh_goal_panel()

        style = self.agent_state.planner_style

        if not self.eidos_worker:
            QMessageBox.warning(
                self, "Planner 사용 불가",
                "Planner는 LLM 연결(eidos_worker)이 필요합니다.\n"
                "EIDOS 워커가 연결된 상태에서 사용하세요."
            )
            return

        # 기존 캔버스가 있으면 덮어쓸지 확인
        if self.nodes or self.loop_nodes:
            r = QMessageBox.question(
                self, "DAG 생성",
                f"목표:\n'{goal}'\n\n현재 캔버스를 지우고 새 DAG를 생성할까요?\n"
                f"(스타일: {style})"
            )
            if r != QMessageBox.Yes:
                return

        # 이미 실행 중이면 중복 실행 방지
        if getattr(self, "_planner_thread", None) is not None and self._planner_thread.isRunning():
            QMessageBox.information(self, "Planner", "이미 DAG 생성이 진행 중입니다.")
            return

        self.statusBar().showMessage("🗺️ Planner 실행 중... LLM이 DAG를 설계하고 있습니다.", 0)

        # ── LLM 호출을 워커 스레드로 분리 (GUI 프리징/렉 방지) ──────────
        progress = QProgressDialog(
            "LLM이 DAG를 설계하고 있습니다...", None, 0, 0, self
        )
        progress.setWindowTitle("🗺️ Planner 실행 중")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()
        self._planner_progress = progress

        thread = _PlannerThread(self, goal, style)
        self._planner_thread = thread

        def _on_planner_result(dag_data, _goal=goal):
            # 진행 다이얼로그 닫기
            try:
                if getattr(self, "_planner_progress", None) is not None:
                    self._planner_progress.close()
            except Exception:
                pass
            self._planner_progress = None
            self._planner_thread = None

            if dag_data is None:
                self.statusBar().showMessage("❌ Planner 실패 — LLM 응답을 파싱하지 못했습니다.", 4000)
                QMessageBox.warning(
                    self, "Planner 실패",
                    "DAG 생성에 실패했습니다.\n"
                    "LLM 연결을 확인하거나 목표를 더 구체적으로 작성해 보세요."
                )
                return

            # x 좌표를 노드 순서에 맞게 자동 배치 (LLM이 x=0으로 뭉쳐놓을 수 있음)
            dag_data = _auto_layout_dag(dag_data)

            self.clear_all(confirm=False)
            self._load_agent_data(dag_data)

            node_count = len(dag_data.get("nodes", []))
            edge_count = len(dag_data.get("edges", []))
            self.statusBar().showMessage(
                f"✅ DAG 생성 완료 — 노드 {node_count}개, 엣지 {edge_count}개 | 목표: {_goal[:40]}", 5000
            )
            # 캔버스 전체가 보이도록 뷰 리셋
            self.view.fitInView(self.scene.itemsBoundingRect().adjusted(-60,-60,60,60),
                                Qt.KeepAspectRatio)

        thread.result_ready.connect(_on_planner_result)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    def _run_verb_pipeline_ui(self, prefill_text: str = "",
                               domain: str = "",
                               seed_input: str = "",
                               auto_execute: bool = True):
        """🔀 Verb 시퀀스 → 자동 조립 + 즉시 실행.

        외부 호출 가능 (prefill_text 제공 시 다이얼로그 스킵).
        """
        try:
            from eidos_verb_pipeline_compiler import (
                resolve_verbs_from_text, compile_to_dag, VerbPipeline,
            )
        except Exception as e:
            QMessageBox.warning(self, "Verb 시퀀스",
                                 f"verb_pipeline_compiler 로드 실패: {e}")
            return

        text = (prefill_text or "").strip()
        if not text:
            text, ok = QInputDialog.getText(
                self, "🔀 Verb 시퀀스 조립+실행",
                "verb 시퀀스를 입력하세요 (콤마/화살표 구분):\n"
                "예: 買う → 比較する → 評価する → 推薦する\n"
                "또는: 사다, 비교하다, 평가하다, 추천하다"
            )
            if not ok or not text.strip():
                return
            text = text.strip()

        verbs = resolve_verbs_from_text(text)
        if not verbs:
            QMessageBox.warning(
                self, "Verb 매칭 실패",
                f"입력에서 verb 를 찾지 못했습니다:\n  {text}\n\n"
                "verb 이름 (jp/kr/alias) 으로 입력하세요."
            )
            return

        # 시드 입력 (옵셔널) — prefill 없으면 다이얼로그
        if not seed_input and not prefill_text:
            seed_input, _ok = QInputDialog.getText(
                self, "초기 입력 (선택)",
                f"첫 verb({verbs[0].kr})에 줄 입력 텍스트 (비워두면 빈 문자열):"
            )
            seed_input = (seed_input or "").strip()

        # 캔버스 덮어쓰기 확인
        if self.nodes or self.loop_nodes:
            chain = " → ".join(v.kr for v in verbs)
            r = QMessageBox.question(
                self, "시퀀스 조립",
                f"verb {len(verbs)}개: {chain}\n\n"
                "현재 캔버스를 지우고 시퀀스로 교체할까요?"
            )
            if r != QMessageBox.Yes:
                return

        pipeline = VerbPipeline(
            verbs=verbs,
            name=text,
            domain=domain,
            seed_input=seed_input,
        )
        # cw=self → 등록 없는 verb 는 이 캔버스의 eidos_worker 로 LLM 자동 생성
        try:
            self.statusBar().showMessage(
                f"🔧 {len(verbs)}개 verb 의 에이전트 조립 중… "
                "(미등록 verb 는 LLM 자동 생성)", 0,
            )
            QApplication.instance().processEvents()
        except Exception:
            pass
        dag = compile_to_dag(pipeline, cw=self)

        if not dag.get("nodes"):
            QMessageBox.warning(self, "조립 실패", "컴파일 결과 빈 DAG 입니다.")
            return

        self.clear_all(confirm=False)
        self._load_agent_data(dag)

        # seed → 시작 입력 위젯
        if pipeline.seed_input:
            try:
                self.start_input_edit.setText(pipeline.seed_input)
            except Exception:
                pass

        # 도메인 → goal 라벨
        if pipeline.domain:
            try:
                self.agent_state.goal = f"[domain={pipeline.domain}] {pipeline.label()}"
                self._refresh_goal_panel()
            except Exception:
                pass

        # 뷰 리셋
        try:
            self.view.fitInView(
                self.scene.itemsBoundingRect().adjusted(-60, -60, 60, 60),
                Qt.KeepAspectRatio,
            )
        except Exception:
            pass

        self.statusBar().showMessage(
            f"✅ Verb 시퀀스 조립 완료 — 노드 {len(verbs)}개. "
            f"{'자동 실행…' if auto_execute else '수동 실행 대기.'}",
            4000,
        )

        # 즉시 실행 — eidos_worker 없으면 가드
        if auto_execute:
            if not self.eidos_worker:
                QMessageBox.information(
                    self, "수동 실행 필요",
                    "LLM 워커(eidos_worker)가 연결되지 않아 자동 실행할 수 없습니다.\n"
                    "▶ 전체 실행 버튼을 수동으로 눌러주세요."
                )
                return
            try:
                self.run_agent()
            except Exception as e:
                QMessageBox.warning(self, "실행 실패",
                                     f"run_agent 실패: {type(e).__name__}: {e}")

    def _refresh_goal_panel(self):
        """AgentState 데이터를 Goal 패널 위젯에 반영."""
        if not hasattr(self, 'goal_edit'):
            return
        self.goal_edit.setText(self.agent_state.goal or "")
        # goal_mode_combo 동기화
        for i in range(self.goal_mode_combo.count()):
            if self.goal_mode_combo.itemData(i) == self.agent_state.goal_prompt:
                self.goal_mode_combo.setCurrentIndex(i)
                break
        # planner_style_combo 동기화
        if hasattr(self, 'planner_style_combo'):
            for i in range(self.planner_style_combo.count()):
                if self.planner_style_combo.itemData(i) == self.agent_state.planner_style:
                    self.planner_style_combo.setCurrentIndex(i)
                    break
        # 진행도 / 실행 횟수 업데이트
        pct = int(self.agent_state.progress * 100)
        self.goal_progress_label.setText(f"진행도: {pct}%")
        self.goal_run_count_label.setText(f"실행 횟수: {self.agent_state.run_count}")

    def _topo_sort_body(self, node_ids: List[str], edge_ids: List[str]) -> List[str]:
        edges = [self.edges[eid] for eid in edge_ids if eid in self.edges]
        adj: Dict[str, List[str]] = {nid: [] for nid in node_ids}
        indeg: Dict[str, int]     = {nid: 0  for nid in node_ids}
        for e in edges:
            if e.src_id in adj and e.dst_id in adj:
                adj[e.src_id].append(e.dst_id)
                indeg[e.dst_id] += 1
        q     = [n for n in node_ids if indeg[n] == 0]
        order = []
        while q:
            n = q.pop(0)
            order.append(n)
            for nx in adj.get(n, []):
                indeg[nx] -= 1
                if indeg[nx] == 0:
                    q.append(nx)
        remaining = [n for n in node_ids if n not in order]
        return order + remaining

    def _on_tool_combo_changed(self, index: int):
        """툴 콤보박스 선택이 바뀌면 즉시 노드 데이터에 반영."""
        if not self.current_node_id:
            return
        key = self.tool_combo.itemData(index)
        if not key:  # 구분자 선택 시 무시
            return
        data = self.nodes.get(self.current_node_id)
        if data and data.node_type == "tool":
            data.prompt = key
            # 노드 타이틀도 선택한 툴 표시명으로 자동 업데이트 (타이틀이 기본값일 때만)
            label = self.tool_combo.itemText(index)
            # 이모지+공백 제거해서 깔끔한 이름으로
            clean = label.lstrip("🔍📖✏️🗂️🐍🧮📁🔁📝📅➕🗑️ ").strip()
            if not data.title or data.title in ("새 도구", "Untitled"):
                data.title = clean
                self.ed_title.blockSignals(True)
                self.ed_title.setText(clean)
                self.ed_title.blockSignals(False)
            item = self.node_items.get(self.current_node_id)
            if item:
                item.update()

    def _schedule_autosave(self):
        # debounce 400ms
        self._autosave_timer.start(400)

    def _on_condition_mode_toggled(self, checked: bool):
        """If 노드 — 자연어 ↔ Python 표현식 모드 전환.
        체크 시 자연어 textbox 의 내용을 expression 입력란으로 옮기지 않음 (혼란 방지).
        사용자가 명시적으로 다시 입력해야 함."""
        self.ed_condition_natural.setVisible(not checked)
        self.ed_condition.setVisible(checked)
        self._schedule_autosave()

    def _current_loop_mode(self) -> str:
        if self.rb_loop_advanced.isChecked():
            return "advanced"
        if self.rb_loop_quality.isChecked():
            return "quality"
        if self.rb_loop_goal.isChecked():
            return "goal"
        return "count"

    def _refresh_loop_mode_visibility(self):
        """라디오 선택에 따라 자연어 입력 / 고급 위젯 visibility 갱신."""
        mode = self._current_loop_mode()
        is_nl  = mode in ("quality", "goal")
        is_adv = (mode == "advanced")

        # 자연어 입력 — quality / goal 일 때만
        self.loop_quality_label.setVisible(is_nl)
        self.loop_quality_text.setVisible(is_nl)
        if mode == "quality":
            self.loop_quality_label.setText("✨ 품질 기준 (자연어 — LLM 이 매 회차 후 yes/no 판정)")
            self.loop_quality_text.setPlaceholderText(
                "예) 결과가 사용자 요구를 충분히 만족했다\n"
                "예) 더 이상 개선할 점이 없다\n"
                "예) 품질 점수가 0.85 이상이다"
            )
        elif mode == "goal":
            self.loop_quality_label.setText("🎯 목표 (자연어 — LLM 이 매 회차 후 달성 여부 판정)")
            self.loop_quality_text.setPlaceholderText(
                "예) 영문 보고서가 한국어로 자연스럽게 번역되었다\n"
                "예) 결과에 표/수치/요약이 모두 포함되었다"
            )

        # 반복 횟수 라벨 — count 면 '반복 횟수', 그 외엔 '안전 상한'
        if mode == "count":
            self.loop_count_label.setText("반복 횟수")
        elif is_nl:
            self.loop_count_label.setText("안전 상한 (최대 반복 수, 무한루프 방지)")
        else:
            self.loop_count_label.setText(_ct("lbl_loop_count"))

        # 고급 위젯들 — advanced 일 때만
        for w in getattr(self, "_loop_advanced_widgets", []):
            try:
                w.setVisible(is_adv)
            except Exception:
                pass
        # advanced 모드에서 trigger 하위 패널은 trigger_combo 선택에 따라 다시 결정
        if is_adv:
            try:
                self._refresh_trigger_panels(
                    self.trigger_combo.itemData(self.trigger_combo.currentIndex())
                )
            except Exception:
                pass
        else:
            self.trigger_schedule_widget.setVisible(False)
            self.trigger_sm_widget.setVisible(False)

    def _on_loop_mode_changed(self):
        self._refresh_loop_mode_visibility()
        self._schedule_autosave()

    def _apply_inspector_to_node(self):
        node_id = self.current_node_id
        if not node_id:
            return

        # loop 노드 저장
        if node_id in self.loop_nodes:
            ldata = self.loop_nodes[node_id]
            ldata.title      = self.ed_title.text().strip() or "루프"
            ldata.loop_count = self.loop_spin.value()
            mode = self._current_loop_mode()
            ldata.loop_mode = mode
            if mode == "count":
                # N번 반복 — 조기 종료 비활성. 기존 exit_condition 그대로 보존(advanced 전환 시 복원).
                pass
            elif mode in ("quality", "goal"):
                # 자연어 입력 → exit_condition 으로 저장 (실행 시 'llm:' prefix 자동 부여)
                nl = self.loop_quality_text.toPlainText().strip()
                ldata.exit_condition = nl
                # quality/goal 에서는 메모리 누적 default
                if (ldata.memory_mode or "none") == "none":
                    ldata.memory_mode = "append"
            else:  # advanced
                ldata.exit_condition = self.loop_exit_cond_edit.text().strip()
            ldata.max_iterations = self.loop_max_iter_spin.value()
            # 루프 메모리 저장 (❹) — advanced 에서만 콤보 선택 반영
            if mode == "advanced":
                ldata.memory_mode          = self.loop_memory_mode_combo.itemData(
                    self.loop_memory_mode_combo.currentIndex()
                )
            ldata.memory_compress_every = self.loop_compress_spin.value()
            item = self.loop_items.get(node_id)
            if item:
                item.update()
            self._mark_dirty()
            self.statusBar().showMessage("자동 저장됨", 1000)
            return

        if node_id not in self.nodes:
            return

        data = self.nodes[node_id]
        data.title = self.ed_title.text().strip() or "Untitled"
        if data.node_type == "tool":
            key = self.tool_combo.itemData(self.tool_combo.currentIndex())
            if key:
                data.prompt = key
        else:
            data.prompt = self.ed_prompt.toPlainText()
        if data.node_type == "if":
            if self.chk_condition_advanced.isChecked():
                data.condition = self.ed_condition.text().strip()
                data.condition_mode = "expression"
            else:
                data.condition = self.ed_condition_natural.toPlainText().strip()
                data.condition_mode = "natural"

        item = self.node_items.get(node_id)
        if item:
            item.update()

        self._mark_dirty()
        self.statusBar().showMessage("자동 저장됨", 1000)

    # -------------------------
    # Edge operations
    # -------------------------

    def _create_edge_from_items(self, src_item, dst_item) -> bool:
        # PromptBoxItem은 .data.node_id, LoopBoxItem은 .data.node_id (동일 구조)
        src_id = src_item.data.node_id
        dst_id = dst_item.data.node_id

        # prevent duplicate edges
        for e in self.edges.values():
            if e.src_id == src_id and e.dst_id == dst_id:
                return False

        # prevent cycle
        if self._would_create_cycle(src_id, dst_id):
            QMessageBox.warning(self, "연결 불가", "이 연결은 순환(Cycle)을 만들 수 있어 차단했습니다.")
            return False

        edge_id  = str(uuid.uuid4())
        edge     = EdgeData(edge_id=edge_id, src_id=src_id, dst_id=dst_id)
        self.edges[edge_id] = edge

        edge_item = EdgeItem(edge=edge, src=src_item, dst=dst_item)
        edge_item.request_delete.connect(self._delete_edge)
        self.scene.addItem(edge_item)
        self.edge_items[edge_id] = edge_item

        self._mark_dirty()
        return True

    def _would_create_cycle(self, src_id: str, dst_id: str) -> bool:
        """
        src->dst 엣지 추가 시 사이클 여부 검사.
        루프 노드가 src/dst인 경우 외부 DAG 상에서만 검사한다
        (루프 바디 내부 엣지는 이미 _build_adj에서 제외됨).
        """
        # 바디 내부 연결인지 확인 → 내부 연결은 항상 허용
        for ld in self.loop_nodes.values():
            body_set = set(ld.body_node_ids)
            if src_id in body_set and dst_id in body_set:
                return False  # 바디 내부 엣지 — 사이클 아님

        adj = self._build_adj()
        adj.setdefault(src_id, []).append(dst_id)

        stack   = [dst_id]
        visited = set()
        while stack:
            n = stack.pop()
            if n == src_id:
                return True
            if n in visited:
                continue
            visited.add(n)
            for nx in adj.get(n, []):
                if nx not in visited:
                    stack.append(nx)
        return False

    def _delete_edge(self, edge_id: str):
        """ Deletes an edge by its ID. Triggered by signal from EdgeItem. """
        if edge_id not in self.edges:
            return

        r = QMessageBox.question(self, "연결 삭제", "이 연결(Edge)을 삭제하시겠습니까?")
        if r != QMessageBox.Yes:
            return

        # Remove graphics item
        item = self.edge_items.pop(edge_id, None)
        if item:
            self.scene.removeItem(item)

        # Remove data
        self.edges.pop(edge_id, None)

        self._mark_dirty()
        self.statusBar().showMessage("연결 삭제 완료", 1500)

    # -------------------------
    # DAG execution engine
    # -------------------------

    def _make_runner(self, order, prior_outputs=None, start_input: Optional[str] = None):
        """CanvasRunnerThread 생성 공통 헬퍼."""
        final_start_input = start_input if start_input is not None else self.start_input_edit.text()

        # prior_outputs (dict) → ContextStore 변환
        prior_ctx = ContextStore()
        for nid, out in (prior_outputs or {}).items():
            nd = self.nodes.get(nid)
            title = nd.title if nd else nid
            prior_ctx.record(nid, title, out, True)

        # 이전 실행 컨텍스트가 있으면 복원
        if self.agent_state.last_context:
            try:
                prior_ctx = ContextStore.from_dict(self.agent_state.last_context)
            except Exception:
                pass

        return CanvasRunnerThread(
            canvas_window  = self,
            order          = order,
            outputs        = {},
            global_context = prior_ctx,
            start_input    = final_start_input,
            agent_state    = self.agent_state,
        )

    def run_agent(self):
        """전체 실행 — GUI 스레드 비블로킹."""
        if not self.nodes:
            QMessageBox.information(self, "실행", "노드가 없습니다.")
            return
        if hasattr(self, '_runner_thread') and self._runner_thread and self._runner_thread.isRunning():
            QMessageBox.information(self, "실행 중", "이미 실행 중입니다.")
            return
        try:
            order = self._topological_order()
        except ValueError as e:
            QMessageBox.warning(self, "실행 불가", str(e))
            return

        for node in self.nodes.values():
            node.execution_state = "pending"
        for item in self.node_items.values():
            item.update()

        self._runner_thread = self._make_runner(order)

        def on_started(nid, title):
            node = self.nodes.get(nid)
            if node: node.execution_state = "running"
            item = self.node_items.get(nid)
            if item: item.update()
            self.statusBar().showMessage(f"⚙ 실행 중: {title}...")

        def on_finished(nid, result, ok):
            node = self.nodes.get(nid)
            if node:
                node.execution_state = "success" if ok else "error"
                node.last_output = result
            item = self.node_items.get(nid)
            if item: item.update()
            if self.current_node_id == nid:
                self.ed_last_output.setPlainText(result)

        def on_all(final_result: str):
            self._mark_dirty()
            self._refresh_goal_panel()   # ② goal 진행도/실행횟수 갱신
            self.statusBar().showMessage("✅ 전체 실행 완료", 3000)
            QMessageBox.information(self, "실행 완료", f"최종 결과:\n\n{final_result}")

        self._runner_thread.node_started.connect(on_started)
        self._runner_thread.node_finished.connect(on_finished)
        self._runner_thread.all_finished.connect(on_all)
        self._runner_thread.start()

    def run_agent_with_callbacks(self,
                                  start_input: str,
                                  on_node_finished=None,
                                  on_all_finished=None):
        """
        외부(ProjectManager 등)에서 파이프라인 완료 콜백을 미리 등록하고 실행.
        시그널 연결 타이밍 문제 없이 run_agent() 전에 콜백 설정 가능.

        Args:
            start_input (str): 에이전트 실행을 위한 시작 입력값.
            on_node_finished(nid, result, ok): 노드 완료 시 콜백.
            on_all_finished(final_result: str): 전체 완료 시 콜백. (최종 결과 전달)
        """
        if not self.nodes:
            if on_all_finished: on_all_finished("실행할 에이전트 노드가 없습니다.")
            return
        if hasattr(self, '_runner_thread') and self._runner_thread and self._runner_thread.isRunning():
            # 이미 실행 중이면 아무것도 하지 않고 종료.
            if on_all_finished: on_all_finished("이미 다른 에이전트가 실행 중입니다.")
            return
        try:
            order = self._topological_order()
        except ValueError as e:
            print(f"⚠️ [Canvas] 실행 불가 (순환 참조 등): {e}")
            if on_all_finished: on_all_finished(f"에이전트 실행 오류: {e}")
            return

        for node in self.nodes.values():
            node.execution_state = "pending"

        # _make_runner를 통해 start_input을 전달합니다.
        self._runner_thread = self._make_runner(order, start_input=start_input)

        # 외부 콜백 연결 (run 전에 등록 → 타이밍 문제 없음)
        if on_node_finished:
            self._runner_thread.node_finished.connect(on_node_finished)
        if on_all_finished:
            self._runner_thread.all_finished.connect(on_all_finished)

        # 내부 상태 업데이트
        def on_finished_internal(nid, result, ok):
            node = self.nodes.get(nid)
            if node:
                node.execution_state = "success" if ok else "error"
                node.last_output = result

        self._runner_thread.node_finished.connect(on_finished_internal)
        self._runner_thread.start()
        print(f"✅ [Canvas] run_agent_with_callbacks 시작 — {len(order)}개 노드")

    def _run_from_selected(self):
        """
        선택 노드부터 하위 그래프를 실행.
        ContextStore 기반 CanvasRunnerThread에 위임 → 이중 시스템 제거 (① 통일).
        """
        if not self.current_node_id:
            QMessageBox.information(self, "부분 실행", "먼저 실행 시작 노드를 더블클릭하여 선택해 주세요.")
            return

        if hasattr(self, '_runner_thread') and self._runner_thread and self._runner_thread.isRunning():
            QMessageBox.information(self, "실행 중", "이미 실행 중입니다.")
            return

        try:
            order = self._topological_order()
        except ValueError as e:
            QMessageBox.warning(self, "실행 불가", str(e))
            return

        start_node_id = self.current_node_id
        reachable     = self._reachable_from(start_node_id)
        run_order     = [nid for nid in order if nid in reachable]

        # 실행 대상 노드 상태 초기화
        for nid in run_order:
            nd = self.nodes.get(nid)
            if nd:
                nd.execution_state = "pending"
            item = self.node_items.get(nid)
            if item:
                item.update()

        # 이미 완료된 비실행 노드의 출력을 초기 ContextStore에 주입
        prior_ctx = ContextStore()
        for nid, nd in self.nodes.items():
            if nid not in reachable and nd.last_output:
                prior_ctx.record(nid, nd.title, nd.last_output, True)

        start_input = self.start_input_edit.text()

        self._runner_thread = CanvasRunnerThread(
            canvas_window  = self,
            order          = run_order,
            outputs        = {},
            global_context = prior_ctx,
            start_input    = start_input,
            agent_state    = self.agent_state,
        )

        def on_started(nid, title):
            nd = self.nodes.get(nid)
            if nd:
                nd.execution_state = "running"
            item = self.node_items.get(nid)
            if item:
                item.update()
            self.statusBar().showMessage(f"⚙ 실행 중: {title}...")

        def on_finished(nid, result, ok):
            nd = self.nodes.get(nid)
            if nd:
                nd.execution_state = "success" if ok else "error"
                nd.last_output = result
            item = self.node_items.get(nid)
            if item:
                item.update()
            if self.current_node_id == nid:
                self.ed_last_output.setPlainText(result)

        def on_all(final_result: str):
            self._mark_dirty()
            self._refresh_goal_panel()
            self.statusBar().showMessage("✅ 부분 실행 완료", 2000)

        self._runner_thread.node_started.connect(on_started)
        self._runner_thread.node_finished.connect(on_finished)
        self._runner_thread.all_finished.connect(on_all)
        self._runner_thread.start()

    def _execute_prompt(self, compiled_input: str, node_id: str, title: str) -> Tuple[str, bool]:
        """
        동기 실행 디스패처.
        - tool 노드: _execute_tool() 위임
        - prompt 노드: eidos_worker가 있으면 get_llm_response_async를
          asyncio.run_coroutine_threadsafe 대신 새 이벤트루프로 직접 실행.
          (run_agent는 GUI 스레드에서 동기 호출하므로 이 방식 사용)
        """
        node = self.nodes.get(node_id)
        if node and node.node_type == "tool":
            return self._execute_tool(node.prompt, compiled_input, node_id, title)

        # ── LLM 호출 ─────────────────────────────────────────────
        if self.eidos_worker is not None:
            # worker의 이벤트 루프가 살아있으면 submit_task + 결과 대기 불가
            # (동기 블로킹은 데드락 위험) → 별도 루프에서 직접 실행
            try:
                from llm_module import get_llm_response_async
                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(get_llm_response_async(compiled_input))
                finally:
                    loop.close()
                # [(C) Phase 1 2026-05-21] 노드 LLM 호출 1회 기록.
                # is_enabled OFF 시 자동 no-op. blueprint_id 는 canvas 추적
                # 미흡 → "canvas:<window_id>" sentinel. Phase 2 에서 정확한
                # blueprint_id 매핑 추가 (chain_complete/failed linkage).
                try:
                    from eidos_verb_agent_learning import record_invocation as _val_rec
                    _bp_id = getattr(self, "_current_blueprint_path", None) \
                             or f"canvas:{id(self)}"
                    _val_rec(
                        blueprint_id=str(_bp_id),
                        node_id=str(node_id),
                        input_text=str(compiled_input)[:2000],
                        output_text=str(result)[:2000],
                    )
                except Exception:
                    pass
                return str(result), True
            except Exception as e:
                return f"[{title}] LLM 오류: {e}", False

        # worker 없음 → 시뮬레이션
        if not compiled_input.strip():
            return f"{title}: (빈 프롬프트 - 출력 없음)", True
        return f"[시뮬레이션] {title}:\n{compiled_input.strip()[:400]}", True

    def _execute_tool(self, tool_name: str, input_str: str, node_id: str, title: str) -> Tuple[str, bool]:
        """
        Hook for executing a tool, via eidos_worker or simulation.
        Returns (output_string, success_boolean).
        """
        # Worker integration for tools
        if self.eidos_worker is not None:
            fn = getattr(self.eidos_worker, 'run_tool', None)
            if callable(fn):
                try:
                    result = fn(tool_name, input_str)
                    return str(result), True
                except Exception as e:
                    return f"[{title}] 도구 실행 오류: {e}", False

        # Fallback simulation for tools
        self.statusBar().showMessage(f"실행(시뮬레이션): {tool_name} with input '{input_str[:30]}...'", 1500)
        if tool_name == "web_search":
            return f"웹 검색 결과 for '{input_str}'", True
        elif tool_name == "file_read":
            return f"'file.txt' 내용: ...", True
        elif tool_name == "code_exec":
            return f"코드 실행 결과: 42", True
        else:
            return f"오류: 알 수 없는 도구 '{tool_name}'", False

    def _get_upstream_outputs(self, nid: str, outputs: Dict[str, str]) -> List[Tuple[str, str]]:
        """
        현재 노드(nid)의 edge로 직접 연결된 upstream 노드들의 (title, result) 리스트를 반환.
        edge 정보를 기반으로 하므로, 실행 순서와 무관하게 실제 연결만 반영.
        """
        upstream = []
        for e in self.edges.values():
            if e.dst_id == nid and e.src_id in outputs:
                src_node = self.nodes.get(e.src_id)
                title = src_node.title if src_node else e.src_id
                upstream.append((title, outputs[e.src_id]))
        return upstream

    def _compile_prompt(self, raw: str, outputs: Dict[str, str], global_context,
                        start_input: str, nid: str = "") -> str:
        """
        레거시 호환 래퍼.
        outputs(dict) + global_context(str|ContextStore) → ContextStore 변환 후
        CanvasRunnerThread._compile 와 동일한 로직 실행.

        지원 변수:
          {{context}}      → 누적 컨텍스트 렌더링
          {{input}}        → 시작 입력값
          {{prev}}         → 직전 출력 (outputs 마지막 값)
          {{n:<id>}}       → 특정 노드 출력  (레거시)
          {{key:<id>}}     → 특정 노드 출력  (신규)
          {{var:<key>}}    → AgentState 변수
        """
        # ── ContextStore 준비 ────────────────────────────────────
        if isinstance(global_context, ContextStore):
            ctx = global_context
        else:
            ctx = ContextStore()
            for _nid, _out in outputs.items():
                nd = self.nodes.get(_nid)
                ctx.record(_nid, nd.title if nd else _nid, _out, True)

        s    = raw or ""
        prev = ctx.last_output() or (next(reversed(outputs.values()), "") if outputs else "")
        context_text = ctx.render() if isinstance(global_context, ContextStore) else str(global_context)

        # 기본 변수 치환
        s = s.replace("{{context}}", context_text).replace("{context}", context_text)
        s = s.replace("{{input}}", start_input).replace("{input}", start_input)
        s = s.replace("{{prev}}", prev).replace("{prev}", prev)

        # {{key:<id>}} / {{n:<id>}}
        for m in re.finditer(r"\{\{(?:key|n):([^}]+)\}\}", s):
            s = s.replace(m.group(0), ctx.get(m.group(1).strip()) or outputs.get(m.group(1).strip(), ""))
        for m in re.finditer(r"\{(?:key|n):([^}]+)\}", s):
            s = s.replace(m.group(0), ctx.get(m.group(1).strip()) or outputs.get(m.group(1).strip(), ""))

        # {{var:<key>}}
        for m in re.finditer(r"\{\{var:([^}]+)\}\}", s):
            s = s.replace(m.group(0), self.agent_state.get_var(m.group(1).strip()))
        for m in re.finditer(r"\{var:([^}]+)\}", s):
            s = s.replace(m.group(0), self.agent_state.get_var(m.group(1).strip()))

        # 한글 prev 별칭
        _PREV = re.compile(
            r"\{(?:프롬프트\d+[_\s]*출력|이전[_\s]*출력|previous[_\s]*output|prev[_\s]*output)\}",
            re.IGNORECASE,
        )
        s = _PREV.sub(prev, s)

        # upstream 자동 주입
        if nid:
            has_explicit_var = any(
                marker in (raw or "")
                for marker in ("{{input}}", "{{prev}}", "{{context}}", "{{n:", "{{key:",
                               "{input}", "{prev}", "{context}", "{프롬프트", "{이전", "{previous")
            )
            if not has_explicit_var:
                upstream_list = self._get_upstream_outputs(nid, outputs)
                if upstream_list:
                    parts = [f"[{title}의 출력]\n{result}" for title, result in upstream_list]
                    s = f"[이전 단계 결과]\n{chr(10).join(parts)}\n\n[현재 지시사항]\n{s}"

        return s

    def _build_adj(self) -> Dict[str, List[str]]:
        # 일반 노드 + 루프 노드 모두 포함
        all_ids = list(self.nodes.keys()) + list(self.loop_nodes.keys())
        adj: Dict[str, List[str]] = {nid: [] for nid in all_ids}
        for e in self.edges.values():
            # 바디 내부 엣지는 외부 DAG에서 제외 (루프 노드가 대표)
            body_edges = set()
            for ld in self.loop_nodes.values():
                body_edges.update(ld.body_edge_ids)
            if e.edge_id in body_edges:
                continue
            adj.setdefault(e.src_id, []).append(e.dst_id)
            adj.setdefault(e.dst_id, adj.get(e.dst_id, []))
        return adj

    def _topological_order(self) -> List[str]:
        """
        일반 노드 + 루프 노드를 포함한 토폴로지 정렬.
        루프 노드는 단일 불투명 노드로 취급 — 바디 내부 사이클은 무시.
        """
        adj   = self._build_adj()
        all_ids = list(self.nodes.keys()) + list(self.loop_nodes.keys())
        indeg = {nid: 0 for nid in all_ids}
        for src, lst in adj.items():
            for dst in lst:
                if dst in indeg:
                    indeg[dst] += 1

        q     = [nid for nid, d in indeg.items() if d == 0]
        order: List[str] = []
        while q:
            n = q.pop(0)
            order.append(n)
            for nx in adj.get(n, []):
                if nx in indeg:
                    indeg[nx] -= 1
                    if indeg[nx] == 0:
                        q.append(nx)

        if len(order) != len(all_ids):
            raise ValueError("DAG 실행 불가: 그래프에 순환(Cycle)이 존재합니다.")
        return order

    def _reachable_from(self, start: str) -> set:
        adj = self._build_adj()
        seen = set()
        stack = [start]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            for nx in adj.get(n, []):
                if nx not in seen:
                    stack.append(nx)
        return seen

    def _create_agent_from_canvas(self):
        if not self.nodes:
            QMessageBox.warning(self, "에이전트 생성 불가", "에이전트를 생성하려면 최소 1개 이상의 노드가 필요합니다.")
            return

        # 1. Get Agent Metadata from user
        agent_name, ok1 = QInputDialog.getText(self, "에이전트 생성", "에이전트 이름:")
        if not ok1 or not agent_name.strip():
            return
        agent_name = agent_name.strip()

        agent_desc, ok2 = QInputDialog.getText(self, "에이전트 생성", "에이전트 설명 (한 줄 요약):")
        if not ok2: # Description is optional
            agent_desc = ""
        agent_desc = agent_desc.strip()

        # 2. Prepare directories and paths
        AGENT_DIR = "eidos_agents"
        BLUEPRINT_DIR = os.path.join(AGENT_DIR, "blueprints")
        REGISTRY_FILE = os.path.join(AGENT_DIR, "registry.json")

        os.makedirs(BLUEPRINT_DIR, exist_ok=True)

        # 3. Save the current graph as a blueprint file
        blueprint_id = str(uuid.uuid4())
        blueprint_filename = f"bp_{blueprint_id}.json"
        blueprint_path = os.path.join(BLUEPRINT_DIR, blueprint_filename)

        graph_data = {
            "version": 1,
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges.values()],
        }

        try:
            with open(blueprint_path, "w", encoding="utf-8") as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "블루프린트 저장 실패", f"그래프 파일 저장 중 오류가 발생했습니다: {e}")
            return

        # 4. Update the agent registry
        registry_data = []
        if os.path.exists(REGISTRY_FILE):
            try:
                with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                    registry_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        adj = self._build_adj()
        indeg = {nid: 0 for nid in self.nodes.keys()}
        for src, lst in adj.items():
            for dst in lst:
                indeg[dst] = indeg.get(dst, 0) + 1

        entry_nodes = [nid for nid, d in indeg.items() if d == 0]

        new_agent_entry = {
            "agent_id": f"agent_{str(uuid.uuid4())}",
            "name": agent_name,
            "description": agent_desc,
            "graph_path": os.path.join("blueprints", blueprint_filename).replace("\\", "/"),
            "entry_node_id": entry_nodes[0] if entry_nodes else None,
        }

        registry_data.append(new_agent_entry)

        try:
            with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
                json.dump(registry_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            QMessageBox.critical(self, "에이전트 등록 실패", f"레지스트리 파일 업데이트 중 오류가 발생했습니다: {e}")
            return

        QMessageBox.information(self, "성공", f"'{agent_name}' 에이전트가 성공적으로 등록되었습니다.")

    # -------------------------
    # Save/Load
    # -------------------------

    def save_to_file(self):
        path, _ = QFileDialog.getSaveFileName(self, "에이전트 저장", "", "EIDOS Agent Blueprint (*.json)")
        if not path:
            return

        def _loop_to_dict(l: "LoopNodeData") -> dict:
            d = asdict(l)
            d.pop("_trigger_fired_date", None)  # 내부 런타임 상태 — 직렬화 제외
            return d

        data = {
            "version":      3,
            "nodes":        [asdict(n) for n in self.nodes.values()],
            "edges":        [asdict(e) for e in self.edges.values()],
            "loop_nodes":   [_loop_to_dict(l) for l in self.loop_nodes.values()],
            "agent_state":  asdict(self.agent_state),   # 개선 #5: 상태 영속화
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self._dirty = False
            self.statusBar().showMessage("저장 완료", 1500)
        except Exception as e:
            QMessageBox.critical(self, "저장 실패", str(e))

    def load_from_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "에이전트 불러오기", "", "EIDOS Agent Blueprint (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "불러오기 실패", str(e))
            return

        try:
            self.clear_all(confirm=False)
            nodes      = data.get("nodes", [])
            edges      = data.get("edges", [])
            loop_nodes = data.get("loop_nodes", [])

            for nd in nodes:
                # 하위 호환: 구버전 JSON에 없는 필드 기본값 처리
                nd.setdefault("node_type", "prompt")
                nd.setdefault("condition", "")
                # 마이그레이션: 기존 노드는 condition 비어있지 않으면 expression 모드 (기존 동작 보존)
                nd.setdefault(
                    "condition_mode",
                    "expression" if (nd.get("condition") or "").strip() else "natural",
                )
                nd.setdefault("last_output", "")
                nd.setdefault("execution_state", "pending")
                # 알 수 없는 필드 제거 (버전 불일치 방지)
                import dataclasses as _dc_node
                _node_valid = {f.name for f in _dc_node.fields(PromptNodeData)}
                nd = {k: v for k, v in nd.items() if k in _node_valid}
                n = PromptNodeData(**nd)
                self.nodes[n.node_id] = n
                item = PromptBoxItem(n)
                item.setPos(QPointF(n.x, n.y))
                item.request_edit.connect(self.open_in_inspector)
                item.moved.connect(lambda _nid: self._mark_dirty())
                self.scene.addItem(item)
                self.node_items[n.node_id] = item

            for ed in edges:
                e        = EdgeData(**ed)
                src_item = self.node_items.get(e.src_id) or self.loop_items.get(e.src_id)
                dst_item = self.node_items.get(e.dst_id) or self.loop_items.get(e.dst_id)
                if not src_item or not dst_item:
                    continue
                self.edges[e.edge_id] = e
                ei = EdgeItem(e, src_item, dst_item)
                ei.request_delete.connect(self._delete_edge)
                self.scene.addItem(ei)
                self.edge_items[e.edge_id] = ei

            # 루프 노드 복원 — 독립 try 블록으로 분리
            # (이전 단계에서 예외가 발생해도 루프 노드 복원은 반드시 완료)
            loop_restore_errors = []
            for ld_dict in loop_nodes:
                try:
                    ld_dict.setdefault("last_output", "")
                    ld_dict.setdefault("execution_state", "pending")
                    ld_dict.setdefault("body_node_ids", [])
                    ld_dict.setdefault("body_edge_ids", [])
                    ld_dict.setdefault("auto_trigger_type", "none")
                    ld_dict.setdefault("auto_trigger_params", {})
                    ld_dict.setdefault("_trigger_fired_date", "")
                    ld_dict.setdefault("exit_condition", "")
                    ld_dict.setdefault("max_iterations", 0)
                    ld_dict.setdefault("memory_mode", "none")
                    ld_dict.setdefault("memory_compress_every", 3)
                    # 마이그레이션: 기존 데이터의 loop_mode 추론
                    #   exit_condition 비어있음 → "count" (단순 N번 반복)
                    #   그 외 → "advanced" (모든 옵션 그대로 노출)
                    if "loop_mode" not in ld_dict:
                        _ec = (ld_dict.get("exit_condition") or "").strip()
                        ld_dict["loop_mode"] = "count" if not _ec else "advanced"
                    # 알 수 없는 필드 제거 (버전 불일치 방지)
                    import dataclasses as _dc
                    valid = {f.name for f in _dc.fields(LoopNodeData)}
                    ld_dict = {k: v for k, v in ld_dict.items() if k in valid}
                    ld = LoopNodeData(**ld_dict)
                    self.loop_nodes[ld.node_id] = ld
                    l_item = LoopBoxItem(ld)
                    l_item.setPos(QPointF(ld.x, ld.y))
                    l_item.request_edit.connect(self.open_in_inspector)
                    l_item.moved.connect(lambda _nid: self._mark_dirty())
                    l_item.request_delete.connect(self._delete_loop_node)
                    self.scene.addItem(l_item)
                    self.loop_items[ld.node_id] = l_item
                except Exception as le:
                    loop_restore_errors.append(str(le))
                    print(f"⚠️ [Loop 복원] 실패: {le} — {ld_dict.get('node_id','?')}")

            # 루프 노드에 연결된 외부 엣지 재처리 (loop_items 생성 후)
            for ed in edges:
                eid = ed["edge_id"]
                if eid in self.edge_items:
                    continue   # 이미 처리됨
                e        = EdgeData(**ed)
                src_item = self.node_items.get(e.src_id) or self.loop_items.get(e.src_id)
                dst_item = self.node_items.get(e.dst_id) or self.loop_items.get(e.dst_id)
                if not src_item or not dst_item:
                    continue
                self.edges[e.edge_id] = e
                ei = EdgeItem(e, src_item, dst_item)
                ei.request_delete.connect(self._delete_edge)
                self.scene.addItem(ei)
                self.edge_items[e.edge_id] = ei

            self._dirty = False
            # AgentState 복원 (v3 이상) — 필드 불일치도 안전하게 처리
            raw_state = data.get("agent_state")
            if raw_state:
                self.agent_state = _safe_load_agent_state(raw_state)
            self.statusBar().showMessage("불러오기 완료", 1500)
        except Exception as e:
            # 루프 노드까지는 복원됐을 수 있으므로 상태 메시지만 표시
            QMessageBox.critical(self, "불러오기 실패", f"데이터 파싱/복원 중 오류: {e}\n\n부분 복원됐을 수 있습니다.")

    def clear_all(self, confirm: bool = True):
        if confirm and (self._dirty or self.nodes or self.edges or self.loop_nodes):
            r = QMessageBox.question(self, "초기화", "모든 노드/연결을 삭제하시겠습니까?")
            if r != QMessageBox.Yes:
                return

        for ei in list(self.edge_items.values()):
            self.scene.removeItem(ei)
        for ni in list(self.node_items.values()):
            self.scene.removeItem(ni)
        for li in list(self.loop_items.values()):
            self.scene.removeItem(li)

        self.nodes.clear()
        self.edges.clear()
        self.node_items.clear()
        self.edge_items.clear()
        self.loop_nodes.clear()
        self.loop_items.clear()

        self._clear_inspector()
        self._dirty = False
        self.statusBar().showMessage("초기화 완료", 1500)

    # -------------------------
    # Helpers
    # -------------------------

    def _load_agent_data(self, data: dict):
        """
        dict(에이전트 JSON)를 직접 받아 캔버스에 로드한다.
        파일 다이얼로그 없이 외부(ProjectManager 등)에서 호출 가능.
        """
        try:
            self.clear_all(confirm=False)
            nodes      = data.get("nodes", [])
            edges      = data.get("edges", [])
            loop_nodes = data.get("loop_nodes", [])

            for nd in nodes:
                nd.setdefault("node_type", "prompt")
                nd.setdefault("condition", "")
                # 마이그레이션: 기존 노드는 condition 비어있지 않으면 expression 모드 (기존 동작 보존)
                nd.setdefault(
                    "condition_mode",
                    "expression" if (nd.get("condition") or "").strip() else "natural",
                )
                nd.setdefault("last_output", "")
                nd.setdefault("execution_state", "pending")
                # 알 수 없는 필드 제거 (버전 불일치 방지)
                import dataclasses as _dc_node
                _node_valid = {f.name for f in _dc_node.fields(PromptNodeData)}
                nd = {k: v for k, v in nd.items() if k in _node_valid}
                n = PromptNodeData(**nd)
                self.nodes[n.node_id] = n
                item = PromptBoxItem(n)
                item.setPos(QPointF(n.x, n.y))
                item.request_edit.connect(self.open_in_inspector)
                item.moved.connect(lambda _nid: self._mark_dirty())
                self.scene.addItem(item)
                self.node_items[n.node_id] = item

            for ed in edges:
                e        = EdgeData(**ed)
                src_item = self.node_items.get(e.src_id)
                dst_item = self.node_items.get(e.dst_id)
                if not src_item or not dst_item:
                    continue
                self.edges[e.edge_id] = e
                ei = EdgeItem(e, src_item, dst_item)
                ei.request_delete.connect(self._delete_edge)
                self.scene.addItem(ei)
                self.edge_items[e.edge_id] = ei

            # 루프 노드 복원 — 독립 try 블록 (필드 불일치 방지)
            import dataclasses as _dc
            _loop_valid = {f.name for f in _dc.fields(LoopNodeData)}
            for ld_dict in loop_nodes:
                try:
                    ld_dict.setdefault("last_output", "")
                    ld_dict.setdefault("execution_state", "pending")
                    ld_dict.setdefault("body_node_ids", [])
                    ld_dict.setdefault("body_edge_ids", [])
                    ld_dict.setdefault("auto_trigger_type", "none")
                    ld_dict.setdefault("auto_trigger_params", {})
                    ld_dict.setdefault("_trigger_fired_date", "")
                    ld_dict.setdefault("exit_condition", "")
                    ld_dict.setdefault("max_iterations", 0)
                    ld_dict.setdefault("memory_mode", "none")
                    ld_dict.setdefault("memory_compress_every", 3)
                    # 마이그레이션: 기존 데이터의 loop_mode 추론
                    #   exit_condition 비어있음 → "count" (단순 N번 반복)
                    #   그 외 → "advanced" (모든 옵션 그대로 노출)
                    if "loop_mode" not in ld_dict:
                        _ec = (ld_dict.get("exit_condition") or "").strip()
                        ld_dict["loop_mode"] = "count" if not _ec else "advanced"
                    ld_dict = {k: v for k, v in ld_dict.items() if k in _loop_valid}
                    ld     = LoopNodeData(**ld_dict)
                    self.loop_nodes[ld.node_id] = ld
                    l_item = LoopBoxItem(ld)
                    l_item.setPos(QPointF(ld.x, ld.y))
                    l_item.request_edit.connect(self.open_in_inspector)
                    l_item.moved.connect(lambda _nid: self._mark_dirty())
                    l_item.request_delete.connect(self._delete_loop_node)
                    self.scene.addItem(l_item)
                    self.loop_items[ld.node_id] = l_item
                except Exception as le:
                    print(f"⚠️ [Loop 복원] 실패: {le} — {ld_dict.get('node_id','?')}")

            # 루프 노드 연결 엣지 재처리
            for ed in edges:
                eid = ed["edge_id"]
                if eid in self.edge_items:
                    continue
                e        = EdgeData(**ed)
                src_item = self.node_items.get(e.src_id) or self.loop_items.get(e.src_id)
                dst_item = self.node_items.get(e.dst_id) or self.loop_items.get(e.dst_id)
                if not src_item or not dst_item:
                    continue
                self.edges[e.edge_id] = e
                ei = EdgeItem(e, src_item, dst_item)
                ei.request_delete.connect(self._delete_edge)
                self.scene.addItem(ei)
                self.edge_items[e.edge_id] = ei

            self._dirty = False
            self.statusBar().showMessage(
                f"에이전트 로드 완료: 노드 {len(nodes)}개, 루프 {len(loop_nodes)}개", 2000
            )
            print(f"✅ [Canvas] 에이전트 데이터 로드 완료 — 노드 {len(nodes)}개, 루프 {len(loop_nodes)}개")
        except Exception as e:
            QMessageBox.critical(self, "에이전트 데이터 로드 실패", f"데이터 파싱/복원 중 오류: {e}")
            print(f"⚠️ [Canvas] _load_agent_data 실패: {e}")

    # -------------------------
    # Helpers
    # -------------------------

    def _mark_dirty(self):
        self._dirty = True

    # Optional: delete key support
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            self._delete_selected_nodes()
            event.accept()
            return
        super().keyPressEvent(event)

    def _delete_selected_nodes(self):
        selected_prompt = [it for it in self.scene.selectedItems() if isinstance(it, PromptBoxItem)]
        selected_loop   = [it for it in self.scene.selectedItems() if isinstance(it, LoopBoxItem)]
        total = len(selected_prompt) + len(selected_loop)
        if total == 0:
            return

        r = QMessageBox.question(self, "삭제", f"{total}개 노드를 삭제하시겠습니까?")
        if r != QMessageBox.Yes:
            return

        selected_ids = (
            {it.data.node_id for it in selected_prompt} |
            {it.data.node_id for it in selected_loop}
        )

        # 연결 엣지 삭제
        del_edge_ids = [eid for eid, e in self.edges.items()
                        if e.src_id in selected_ids or e.dst_id in selected_ids]
        for eid in del_edge_ids:
            ei = self.edge_items.pop(eid, None)
            if ei: self.scene.removeItem(ei)
            self.edges.pop(eid, None)

        # 일반 노드 삭제
        for it in selected_prompt:
            nid = it.data.node_id
            # 이 노드가 어떤 루프 바디에 포함된 경우 제거
            for ld in self.loop_nodes.values():
                if nid in ld.body_node_ids:
                    ld.body_node_ids.remove(nid)
            self.scene.removeItem(it)
            self.node_items.pop(nid, None)
            self.nodes.pop(nid, None)

        # 루프 노드 삭제
        for it in selected_loop:
            nid = it.data.node_id
            self.scene.removeItem(it)
            self.loop_items.pop(nid, None)
            self.loop_nodes.pop(nid, None)

        if self.current_node_id in selected_ids:
            self._clear_inspector()

        self._mark_dirty()
        self.statusBar().showMessage("삭제 완료", 1500)



# =========================
# Loop Auto-Trigger Watcher
# =========================

class LoopTriggerWatcher(QObject):
    """
    1분 주기로 루프 노드의 자동 트리거 조건을 검사하고,
    조건이 충족되면 해당 루프 노드를 포함한 캔버스를 실행한다.

    트리거 타입
    -----------
    "schedule"
        params: {"weekdays": [0..6], "hour_start": int, "hour_end": int}
        • weekdays 0=월 ~ 6=일
        • 현재 시각이 [hour_start, hour_end) 이고,
          현재 요일이 weekdays 목록에 포함되면 발동.
        • 같은 날 이미 발동했으면 스킵 (하루 1회).

    "self_model"
        params: {"key": str, "threshold": float, "op": ">"|">="|"<"|"<="|"=="}
        • eidos_worker.self_model dict 에서 key 를 읽어
          op(value, threshold) 가 True 이면 발동.
        • 1회 발동 후 _trigger_fired_date 에 타임스탬프를 기록하고
          값이 다시 임계치 아래로 내려가기 전까지 재발동하지 않음.
    """

    triggered = Signal(str, str)   # (loop_node_id, trigger_type)

    _OPS = {
        ">":  lambda a, b: a >  b,
        ">=": lambda a, b: a >= b,
        "<":  lambda a, b: a <  b,
        "<=": lambda a, b: a <= b,
        "==": lambda a, b: a == b,
    }

    def __init__(self, canvas_window: "PromptCanvasWindow", parent=None):
        super().__init__(parent)
        self._cw = canvas_window
        self._timer = QTimer(self)
        self._timer.setInterval(60_000)   # 1분
        self._timer.timeout.connect(self._check)
        # self_model 트리거용: 이전 체크에서 조건이 True 였던 노드 집합
        # (False → True 전환 시에만 발동하도록)
        self._sm_was_active: set = set()

    def start(self):
        self._timer.start()
        print("✅ [TriggerWatcher] 자동 트리거 감시 시작 (1분 주기)")

    def stop(self):
        self._timer.stop()

    # ── 내부: 매 분 체크 ─────────────────────────────────────────────────
    def _check(self):
        import datetime
        now = datetime.datetime.now()
        today_str = now.strftime("%Y-%m-%d")

        for nid, ldata in list(self._cw.loop_nodes.items()):
            ttype = getattr(ldata, 'auto_trigger_type', 'none')
            if ttype == 'none':
                continue

            if ttype == 'schedule':
                self._check_schedule(nid, ldata, now, today_str)
            elif ttype == 'self_model':
                self._check_self_model(nid, ldata, today_str)

    def _check_schedule(self, nid: str, ldata, now, today_str: str):
        import datetime
        params    = ldata.auto_trigger_params or {}
        weekdays  = params.get("weekdays", [])
        h_start   = int(params.get("hour_start", 0))
        h_end     = int(params.get("hour_end",   23))

        # 이미 오늘 발동했으면 스킵
        if ldata._trigger_fired_date == today_str:
            return

        # 요일·시간 매칭 (weekday(): 0=월)
        if now.weekday() not in weekdays:
            return
        if not (h_start <= now.hour < h_end):
            return

        # 발동
        ldata._trigger_fired_date = today_str
        print(f"⏰ [TriggerWatcher] schedule 트리거 발동: {ldata.title} ({today_str} {now.hour}시)")
        self.triggered.emit(nid, 'schedule')
        self._fire(nid, ldata)

    def _check_self_model(self, nid: str, ldata, today_str: str):
        worker = self._cw.eidos_worker
        if worker is None:
            return

        sm = getattr(worker, 'self_model', None)
        if not isinstance(sm, dict):
            return

        params    = ldata.auto_trigger_params or {}
        key       = params.get("key", "")
        threshold = float(params.get("threshold", 0.0))
        op_str    = params.get("op", ">")
        op_fn     = self._OPS.get(op_str, lambda a, b: a > b)

        value = sm.get(key)
        if value is None:
            return
        try:
            value = float(value)
        except (TypeError, ValueError):
            return

        is_active = op_fn(value, threshold)

        # False → True 전환 시에만 발동
        was_active = nid in self._sm_was_active
        if is_active and not was_active:
            self._sm_was_active.add(nid)
            print(f"🧠 [TriggerWatcher] self_model 트리거 발동: {ldata.title} "
                  f"({key}={value:.3f} {op_str} {threshold})")
            ldata._trigger_fired_date = today_str
            self.triggered.emit(nid, 'self_model')
            self._fire(nid, ldata)
        elif not is_active:
            self._sm_was_active.discard(nid)

    # ── 실제 루프 노드 포함 캔버스 실행 ──────────────────────────────────
    def _fire(self, nid: str, ldata):
        """
        해당 루프 노드가 포함된 토폴로지 순서로 전체 캔버스를 실행한다.
        이미 runner 가 실행 중이면 큐잉하지 않고 경고만 출력.
        """
        cw = self._cw
        runner = getattr(cw, '_runner_thread', None)
        if runner and runner.isRunning():
            print(f"⚠️ [TriggerWatcher] 이미 실행 중 — {ldata.title} 트리거 스킵")
            return

        try:
            order = cw._topological_order()
        except ValueError as e:
            print(f"⚠️ [TriggerWatcher] 토폴로지 오류: {e}")
            return

        start_input = cw.start_input_edit.text()
        cw._runner_thread = CanvasRunnerThread(
            canvas_window=cw,
            order=order,
            outputs={},
            global_context="",
            start_input=start_input,
        )

        def _on_node(node_id, result, ok):
            nd = cw.nodes.get(node_id)
            if nd:
                nd.execution_state = "success" if ok else "error"
                nd.last_output = result
            item = cw.node_items.get(node_id)
            if item:
                item.update()
            # 루프 노드 상태도 업데이트
            ll = cw.loop_nodes.get(node_id)
            if ll:
                ll.execution_state = "success" if ok else "error"
                ll.last_output = result
            li = cw.loop_items.get(node_id)
            if li:
                li.update()

        def _on_all(final: str):
            cw.statusBar().showMessage(
                f"✅ 자동 트리거 실행 완료 [{ldata.title}]: {final[:60]}", 4000
            )

        cw._runner_thread.node_finished.connect(_on_node)
        cw._runner_thread.all_finished.connect(_on_all)
        cw._runner_thread.start()


# =========================
# Headless Agent Runner
# (Qt 위젯 없이 DAG 실행)
# =========================

class HeadlessCanvasRunner(QThread):
    """
    PromptCanvasWindow 없이 순수 데이터만으로 DAG를 실행.
    ContextStore 기반으로 완전히 재작성 — 이중 시스템 제거 (① 통일).

    실행 경로:
      run_agent_headless() → HeadlessCanvasRunner.run()
        → _run_node_h() (prompt/tool/if 분기)
        → ContextStore 기록
        → node_finished 시그널
      → all_finished 시그널
    """
    node_finished = Signal(str, str, bool)   # (node_id, result, ok)
    all_finished  = Signal(str)              # final_result

    def __init__(self, nodes: dict, edges: dict, order: list,
                 start_input: str, eidos_worker=None,
                 agent_state: Optional[AgentState] = None,
                 parent=None):
        super().__init__(parent)
        self._nodes       = nodes          # {node_id: PromptNodeData}
        self._edges       = edges          # {edge_id: EdgeData}
        self._order       = order          # topological order
        self._start_input = start_input
        self._worker      = eidos_worker
        self._agent_state = agent_state    # 선택적 AgentState

    # ── LLM 호출 ────────────────────────────────────────────────────────
    def _call_llm(self, prompt: str) -> str:
        from llm_module import get_llm_response_async
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(get_llm_response_async(prompt))
        finally:
            loop.close()

    # ── 프롬프트 컴파일 (ContextStore 기반) ─────────────────────────────
    def _compile(self, raw: str, ctx: ContextStore, nid: str = "") -> str:
        s    = raw or ""
        prev = ctx.last_output()
        context_text = ctx.render()

        # goal 주입
        if self._agent_state:
            s = self._agent_state.inject_goal(s)
            s = s.replace("{{goal}}", self._agent_state.goal or "")
            s = s.replace("{goal}",   self._agent_state.goal or "")
            # {{memory}} 주입
            memory_text = _render_memory(self._agent_state.memory)
            s = s.replace("{{memory}}", memory_text)
            s = s.replace("{memory}",   memory_text)

        s = s.replace("{{context}}", context_text).replace("{context}", context_text)
        s = s.replace("{{input}}", self._start_input).replace("{input}", self._start_input)
        s = s.replace("{{prev}}", prev).replace("{prev}", prev)

        # {{key:<id>}} / {{n:<id>}}
        for m in re.finditer(r"\{\{(?:key|n):([^}]+)\}\}", s):
            s = s.replace(m.group(0), ctx.get(m.group(1).strip()))
        for m in re.finditer(r"\{(?:key|n):([^}]+)\}", s):
            s = s.replace(m.group(0), ctx.get(m.group(1).strip()))

        # {{var:<key>}}
        if self._agent_state:
            for m in re.finditer(r"\{\{var:([^}]+)\}\}", s):
                s = s.replace(m.group(0), self._agent_state.get_var(m.group(1).strip()))
            for m in re.finditer(r"\{var:([^}]+)\}", s):
                s = s.replace(m.group(0), self._agent_state.get_var(m.group(1).strip()))

        # 한글 prev 별칭
        s = re.sub(
            r"\{(?:프롬프트\d+[_\s]*출력|이전[_\s]*출력|previous[_\s]*output|prev[_\s]*output)\}",
            prev, s, flags=re.IGNORECASE,
        )

        # upstream 자동 주입
        has_var = any(m in (raw or "") for m in (
            "{{input}}", "{{prev}}", "{{context}}", "{{n:", "{{key:",
            "{input}", "{prev}", "{context}", "{프롬프트", "{이전",
        ))
        if not has_var and nid:
            ups = []
            for e in self._edges.values():
                if e.dst_id == nid:
                    out = ctx.get(e.src_id)
                    if out:
                        src = self._nodes.get(e.src_id)
                        ups.append((src.title if src else e.src_id, out))
            if ups:
                block = "\n\n".join(f"[{t}의 출력]\n{r}" for t, r in ups)
                s = f"[이전 단계 결과]\n{block}\n\n[현재 지시사항]\n{s}"
        return s

    # ── LLM 실행 ────────────────────────────────────────────────────────
    def _execute(self, node, ctx: ContextStore) -> Tuple[str, bool]:
        if node.node_type == "tool":
            tr = _execute_tool_structured_headless(node.prompt, ctx.last_output() or self._start_input, self._worker)
            return tr.to_context_str(), tr.ok

        compiled = self._compile(node.prompt, ctx, node.node_id)
        if self._worker:
            try:
                return self._call_llm(compiled), True
            except Exception as e:
                return f"[{node.title}] LLM 오류: {e}", False
        return f"[시뮬레이션] {node.title}:\n{compiled[:400]}", True

    # ── 메인 실행 루프 ────────────────────────────────────────────────────
    def run(self):
        ctx      = ContextStore()
        skipped  = set()

        if self._agent_state:
            self._agent_state.tick()

        for nid in self._order:
            if nid in skipped:
                continue

            node = self._nodes.get(nid)
            if node is None:
                continue

            try:
                # ── if 노드 ─────────────────────────────────────────
                if node.node_type == "if":
                    prev = ctx.last_output()
                    try:
                        branch = bool(eval(
                            (node.condition or "False"),
                            {"__builtins__": {}},
                            {"prev": prev, "context": ctx.render(),
                             "output": prev, "outputs": ctx.key_outputs},
                        ))
                    except Exception as e:
                        branch = False
                        print(f"⚠️ [Headless If] {e}")

                    result = f"[IF] {'True' if branch else 'False'}"
                    ctx.record(nid, node.title, result, True)
                    self.node_finished.emit(nid, result, True)

                    if not branch:
                        # 하위 노드 스킵 전파
                        queue = {e.dst_id for e in self._edges.values() if e.src_id == nid}
                        visited: set = set()
                        frontier = set(queue)
                        while frontier:
                            cur = frontier.pop()
                            if cur in visited: continue
                            visited.add(cur)
                            for e in self._edges.values():
                                if e.src_id == cur:
                                    frontier.add(e.dst_id); queue.add(e.dst_id)
                        skipped.update(queue)
                    continue

                # ── prompt / tool 노드 ───────────────────────────────
                is_root = not any(e.dst_id == nid for e in self._edges.values())
                if is_root and node.node_type != "tool":
                    has_var = any(m in (node.prompt or "") for m in (
                        "{{input}}", "{{prev}}", "{{context}}", "{{n:", "{{key:",
                        "{input}", "{prev}", "{context}", "{프롬프트", "{이전",
                    ))
                    if not has_var and self._start_input:
                        node = type(node)(
                            **{**vars(node),
                               "prompt": f"[사용자 입력]\n{self._start_input}\n\n[지시사항]\n{node.prompt}"}
                        )

                result, ok = self._execute(node, ctx)

            except Exception as e:
                result, ok = f"[{node.title}] 실행 오류: {e}", False

            ctx.record(nid, node.title, result, ok)
            self.node_finished.emit(nid, result, ok)

        # AgentState 저장
        if self._agent_state:
            self._agent_state.last_context = ctx.to_dict()

        # 터미널 노드 최종 결과
        all_src  = {e.src_id for e in self._edges.values()}
        terminal = [nid for nid in self._order if nid not in all_src and nid not in skipped]
        final    = "\n\n".join(ctx.get(nid) for nid in terminal if ctx.get(nid))
        self.all_finished.emit(final or "")


def _execute_tool_structured_headless(tool_name: str, input_str: str, worker) -> "ToolResult":
    """HeadlessCanvasRunner 전용 tool 실행 (canvas_window 없이)."""
    t0 = time.monotonic()
    is_valid, err_msg, parsed = validate_tool_input(tool_name, input_str)
    if not is_valid:
        return ToolResult(tool_name=tool_name, ok=False, output="",
                          error=f"스키마 오류: {err_msg}",
                          duration_ms=(time.monotonic()-t0)*1000)
    if worker is not None:
        fn = getattr(worker, 'run_tool', None)
        if callable(fn):
            try:
                raw = fn(tool_name, parsed)
                dur = (time.monotonic()-t0)*1000
                if isinstance(raw, dict):
                    return ToolResult(tool_name=tool_name, ok=True,
                                      output=json.dumps(raw, ensure_ascii=False),
                                      output_raw=raw, duration_ms=dur)
                return ToolResult(tool_name=tool_name, ok=True,
                                  output=str(raw), output_raw=raw, duration_ms=dur)
            except Exception as e:
                return ToolResult(tool_name=tool_name, ok=False, output="",
                                  error=str(e), duration_ms=(time.monotonic()-t0)*1000)
    return ToolResult(tool_name=tool_name, ok=True,
                      output=f"[시뮬레이션-headless] {tool_name}: {input_str[:80]}",
                      duration_ms=(time.monotonic()-t0)*1000)


def run_agent_headless(
    agent_data: dict,
    start_input: str,
    eidos_worker=None,
    on_node_finished=None,
    on_all_finished=None,
    parent_qobject=None,
    agent_state: Optional[AgentState] = None,
) -> "HeadlessCanvasRunner":
    """
    에이전트 JSON(dict)을 받아 headless로 DAG를 실행한다.
    ContextStore 기반 HeadlessCanvasRunner 사용 — 이중 시스템 완전 제거.

    Parameters
    ----------
    agent_state : AgentState | None
        외부에서 AgentState를 주입할 수 있다. None이면 임시 인스턴스 생성.
    """
    raw_nodes = agent_data.get("nodes", [])
    raw_edges = agent_data.get("edges", [])

    nodes: dict = {}
    for nd in raw_nodes:
        nd.setdefault("node_type", "prompt")
        nd.setdefault("condition", "")
        nd.setdefault("last_output", "")
        nd.setdefault("execution_state", "pending")
        try:
            n = PromptNodeData(**nd)
            nodes[n.node_id] = n
        except Exception as e:
            print(f"⚠️ [Headless] 노드 파싱 실패: {e} — {nd}")

    edges: dict = {}
    for ed in raw_edges:
        try:
            e = EdgeData(**ed)
            edges[e.edge_id] = e
        except Exception as ex:
            print(f"⚠️ [Headless] 엣지 파싱 실패: {ex} — {ed}")

    # Kahn 위상 정렬
    adj   = {nid: [] for nid in nodes}
    indeg = {nid: 0  for nid in nodes}
    for e in edges.values():
        if e.src_id in adj:
            adj[e.src_id].append(e.dst_id)
        if e.dst_id in indeg:
            indeg[e.dst_id] += 1

    from collections import deque
    q = deque(nid for nid, d in indeg.items() if d == 0)
    order = []
    while q:
        nid = q.popleft()
        order.append(nid)
        for nb in adj.get(nid, []):
            indeg[nb] -= 1
            if indeg[nb] == 0:
                q.append(nb)

    if len(order) != len(nodes):
        err = "에이전트 그래프에 순환 참조가 있습니다."
        print(f"❌ [Headless] {err}")
        if on_all_finished:
            on_all_finished(err)
        return None

    _state = agent_state or AgentState()

    runner = HeadlessCanvasRunner(
        nodes        = nodes,
        edges        = edges,
        order        = order,
        start_input  = start_input,
        eidos_worker = eidos_worker,
        agent_state  = _state,
        parent       = parent_qobject,
    )
    if on_node_finished:
        runner.node_finished.connect(on_node_finished)
    if on_all_finished:
        runner.all_finished.connect(on_all_finished)

    runner.start()
    print(f"✅ [Headless] 실행 시작 — {len(order)}개 노드, 입력: {start_input[:60]!r}")
    return runner


# =========================================================================
# [2026-05-27 ToM 통합 Step 1] run_dag_for_tom — ToM 에이전트 친화 async API
# =========================================================================
# 목적: ToM (eidos_agent_runner) 의 asyncio loop 안에서 PCA DAG 실행.
# 패턴: HeadlessCanvasRunner (QThread, signal 기반) 를 asyncio.Event 로 wrap.
# Qt signal → asyncio bridge 는 loop.call_soon_threadsafe 로 thread-safe.
# 동시 실행 직렬화: 모듈 레벨 asyncio.Lock (DAG 5분+, tick 5분 중첩 방지).
# =========================================================================

_PCA_RUN_LOCK: Optional[asyncio.Lock] = None  # lazy init (loop 바인딩 회피)


def _get_pca_run_lock() -> asyncio.Lock:
    """현재 event loop 에 바인딩된 Lock 을 반환 (lazy init).

    module top-level 에서 asyncio.Lock() 하면 import 시점 loop 가 없어 RuntimeError.
    호출 시점 (이미 loop 안) 에 만들어야 안전.
    """
    global _PCA_RUN_LOCK
    if _PCA_RUN_LOCK is None:
        _PCA_RUN_LOCK = asyncio.Lock()
    return _PCA_RUN_LOCK


# [2026-05-27 통합 후속 #2] DAG 디스크 폴더 분리.
# - eidos_files/agents/        : 일반 PCA 에이전트 (10 샘플 + 사용자 정의)
# - eidos_files/canvas/dags/   : ToM 전용 DAG (필요 시 사용자가 따로 정리)
# 둘 다 스캔. agents/ 우선 (기존 자산·이름 충돌 시 우선).
_PCA_DAG_DIRS = (
    os.path.join("eidos_files", "agents"),
    os.path.join("eidos_files", "canvas", "dags"),
)


def _resolve_dag_path(dag_name: str) -> Optional[str]:
    """dag_name → JSON 파일 경로 해석.

    스캔 순서: _PCA_DAG_DIRS 의 디렉터리 순서대로·각 디렉터리에서
    (1) {name}.json (2) {name} (그대로). 동일 이름 충돌 시 agents/ 우선.
    """
    if not dag_name:
        return None
    for base in _PCA_DAG_DIRS:
        candidates: list[str] = []
        if dag_name.lower().endswith(".json"):
            candidates.append(os.path.join(base, dag_name))
        else:
            candidates.append(os.path.join(base, f"{dag_name}.json"))
            candidates.append(os.path.join(base, dag_name))
        for p in candidates:
            if os.path.isfile(p):
                return p
    return None


# ── [2026-05-27 통합 후속 #1] EidosWorker 콜백 등록 패턴 ─────────────
# chat_gui 의 EidosWorker (run_tool 동기 메서드 있음) 를 등록하면 PCA DAG 의
# tool 노드도 실 실행. 등록 안 되면 _TomPcaSentinel 로 prompt 만 실 LLM,
# tool 은 시뮬레이션 (Phase A 한계 호환). _CLIPBOARD_CALLBACK 과 동일 패턴.
class _TomPcaSentinel:
    """worker 미등록 fallback — HeadlessCanvasRunner 의 truthy 체크만 통과."""
    pass


_TOM_PCA_SENTINEL = _TomPcaSentinel()
_TOM_PCA_WORKER: Optional[Any] = None  # 실 EidosWorker (run_tool 호출 가능)


def set_tom_pca_worker(worker: Optional[Any]) -> None:
    """ToM 이 위임한 PCA DAG 실행에서 tool 노드가 사용할 worker 등록.

    worker 는 `.run_tool(tool_name: str, params: dict) -> Any` 동기 메서드
    필수. None 으로 호출하면 등록 해제 (sentinel fallback).
    chat_gui 가 EidosWorker 생성 직후 호출.
    """
    global _TOM_PCA_WORKER
    if worker is not None and not callable(getattr(worker, "run_tool", None)):
        try:
            print("[set_tom_pca_worker] worker 에 run_tool 없음 - 무시")
        except Exception:
            pass
        return
    _TOM_PCA_WORKER = worker
    try:
        print(
            f"[set_tom_pca_worker] worker "
            f"{'register' if worker else 'unregister'} "
            f"(PCA tool real-exec: {worker is not None})"
        )
    except Exception:
        pass


def _get_effective_pca_worker() -> Any:
    """등록된 EidosWorker 우선, 없으면 sentinel."""
    return _TOM_PCA_WORKER if _TOM_PCA_WORKER is not None else _TOM_PCA_SENTINEL


async def run_dag_for_tom(
    dag_name: str,
    start_input: str = "",
    variables: Optional[dict] = None,
    max_wait_sec: float = 600.0,
) -> dict:
    """ToM 에이전트가 호출하는 PCA DAG 실행 wrapper.

    Args:
        dag_name: DAG 파일명 (eidos_files/agents/{name}.json·.json 자동).
        start_input: HeadlessCanvasRunner 의 start_input — 첫 노드 prompt 의
            {{input}} 에 주입됨.
        variables: AgentState.variables 로 inject — DAG prompt 의 {{var:key}}
            치환에 사용.
        max_wait_sec: 전체 DAG 실행 timeout (sec). 초과 시 ok=False·error 반환.

    Returns:
        {
            "ok": bool,
            "final_result": str,           # all_finished 시그널의 최종 결과
            "node_results": list[dict],    # [{node_id, title, result, ok, node_type}]
            "nodes_done": int,
            "tool_calls": int,             # node_type == "tool" 개수
            "duration_ms": float,
            "error": str,                  # ok=False 일 때만 의미
        }
    """
    t0 = time.monotonic()
    base_fail = lambda err: {
        "ok": False, "final_result": "", "node_results": [],
        "nodes_done": 0, "tool_calls": 0,
        "duration_ms": (time.monotonic() - t0) * 1000,
        "error": err,
    }

    if not dag_name:
        return base_fail("dag_name 비어있음")

    path = _resolve_dag_path(dag_name)
    if not path:
        return base_fail(f"DAG 파일 없음: {dag_name} (eidos_files/agents/ 확인)")

    try:
        with open(path, "r", encoding="utf-8") as f:
            agent_data = json.load(f)
    except Exception as e:
        return base_fail(f"DAG 로드 실패: {type(e).__name__}: {e}")

    raw_nodes = agent_data.get("nodes") or []
    if not raw_nodes:
        return base_fail("DAG 에 노드 없음")

    # node_id → node_type 매핑 (tool_calls 카운트용)
    node_type_by_id: Dict[str, str] = {}
    node_title_by_id: Dict[str, str] = {}
    for nd in raw_nodes:
        nid = nd.get("node_id") or ""
        if nid:
            node_type_by_id[nid] = nd.get("node_type") or "prompt"
            node_title_by_id[nid] = nd.get("title") or nid

    # AgentState 구성 — variables 주입
    agent_state = AgentState()
    if variables and isinstance(variables, dict):
        for k, v in variables.items():
            try:
                agent_state.set_var(str(k), str(v))
            except Exception:
                pass

    lock = _get_pca_run_lock()
    async with lock:
        loop = asyncio.get_event_loop()
        done_event = asyncio.Event()
        node_results: List[dict] = []
        final_holder: Dict[str, str] = {"value": ""}

        def _on_node(node_id: str, result: str, ok: bool):
            # QThread → main loop callback. event.set 만 thread-safe 필요.
            node_results.append({
                "node_id": node_id,
                "title": node_title_by_id.get(node_id, node_id),
                "result": result,
                "ok": bool(ok),
                "node_type": node_type_by_id.get(node_id, "prompt"),
            })

        def _on_all(final_result: str):
            final_holder["value"] = final_result or ""
            try:
                loop.call_soon_threadsafe(done_event.set)
            except Exception:
                # loop 닫혔으면 무시 (이미 timeout 등으로 종료)
                pass

        try:
            runner = run_agent_headless(
                agent_data=agent_data,
                start_input=start_input or "",
                eidos_worker=_get_effective_pca_worker(),
                on_node_finished=_on_node,
                on_all_finished=_on_all,
                parent_qobject=None,
                agent_state=agent_state,
            )
        except Exception as e:
            return base_fail(f"runner 시작 실패: {type(e).__name__}: {e}")

        if runner is None:
            # run_agent_headless 가 순환 참조 등으로 None 반환
            return base_fail("DAG 그래프 오류 (순환 참조 가능)")

        try:
            await asyncio.wait_for(done_event.wait(), timeout=max_wait_sec)
        except asyncio.TimeoutError:
            # QThread 강제 종료 시도 (graceful)
            try:
                runner.requestInterruption()
            except Exception:
                pass
            return {
                "ok": False, "final_result": "",
                "node_results": node_results,
                "nodes_done": len(node_results),
                "tool_calls": sum(1 for n in node_results if n["node_type"] == "tool"),
                "duration_ms": (time.monotonic() - t0) * 1000,
                "error": f"timeout ({max_wait_sec:.0f}s 초과)",
            }

        tool_calls = sum(1 for n in node_results if n["node_type"] == "tool")
        return {
            "ok": True,
            "final_result": final_holder["value"],
            "node_results": node_results,
            "nodes_done": len(node_results),
            "tool_calls": tool_calls,
            "duration_ms": (time.monotonic() - t0) * 1000,
            "error": "",
        }


# =========================
# Standalone test (optional)
# =========================

if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    w = PromptCanvasWindow()
    # demo nodes
    a = w.add_node("문제 이해", "문제를 한 문장으로 요약하라.\n\n입력: {{input}}", QPointF(-200, -50))
    b = w.add_node("조건 정리", "위 요약을 바탕으로 제약을 정리하라.\n\n이전: {{prev}}", QPointF(200, -50))
    c = w.add_node("검증", "산출물이 제약을 만족하는지 검증하라.\n\n컨텍스트: {{context}}", QPointF(600, 80))
    # connect sample
    w._create_edge_from_items(w.node_items[a], w.node_items[b])
    w._create_edge_from_items(w.node_items[b], w.node_items[c])

    w.show()
    sys.exit(app.exec())