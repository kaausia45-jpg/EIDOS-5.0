# eidos_agent_stage_store.py
# [2026-05-26 Phase 0-A] 다중 행위자 ToM 에이전트 — Stage 영속화.
#
# 각 stage 는 eidos_files/agents/{stage_id}/ 폴더 안에:
#   stage.json    선언 — goal·halt·canvas·action registry·safety
#   state.json    런타임 belief (tick 마다 재저장)
#   history.jsonl 사건 로그 (append-only) — 결정·예측·결과
#   attachments/  사용자 첨부 (PDF·이미지·텍스트) — 원본 또는 path 참조
#
# Stage 1개 = 여러 canvas. canvas.kind ∈ {"workflow", "agent_stage"}.
#   workflow      옛 eidos_canvas_store 의 결정적 verb pipeline (CanvasV2)
#   agent_stage   본 모듈에서 정의하는 actor 노드 무대 (ToM)
#
# 사용자 첨부 처리는 옵션 (c) — 자동 제안 시 1회 분석 + 진행 중 추가/교체.
# context_summary 캐시 + summary_ts diff 로 재분석 트리거.

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional, Any

try:
    from eidos_action_registry import ActionRegistry, ActionSpec
except Exception as _e_imp:
    print(f"[agent_stage_store] action_registry import 실패 (graceful): {_e_imp}")
    ActionRegistry = None  # type: ignore
    ActionSpec = None      # type: ignore

_BASE_DIR = os.path.join("eidos_files", "agents")
_VERSION = 1
_STAGE_FILE = "stage.json"
_STATE_FILE = "state.json"
_HISTORY_FILE = "history.jsonl"
_ATTACHMENT_DIR = "attachments"

ACTOR_TYPES = ["self", "human", "organization", "system", "agent", "abstract"]
SAFETY_MODES = ["paranoid", "normal", "yolo"]


# ── Stage canvas (1 stage 안 여러 canvas) ──────────────────────────────
@dataclass
class StageCanvas:
    """1 stage 안 1 canvas. kind 가 \"agent_stage\" 면 ToM 노드.
    \"workflow\" 면 옛 verb pipeline (옛 eidos_canvas_store.dict 구조와 호환)."""

    id: str                         # uuid hex
    name: str
    kind: str = "agent_stage"       # "workflow" | "agent_stage"
    nodes: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)
    # edges 노드 간 영향/관찰 — agent_stage 에서만 의미.
    # workflow kind 에선 connections (옛 이름) 와 동일 의미.
    created_at: str = ""
    updated_at: str = ""

    def serialize(self) -> dict:
        return asdict(self)

    @classmethod
    def deserialize(cls, data: dict) -> "StageCanvas":
        if not isinstance(data, dict):
            data = {}
        now = _now()
        return cls(
            id=str(data.get("id") or _new_id("canvas")),
            name=str(data.get("name") or "메인 무대"),
            kind=str(data.get("kind") or "agent_stage"),
            nodes=list(data.get("nodes", []) or []),
            edges=list(data.get("edges", []) or data.get("connections", []) or []),
            created_at=str(data.get("created_at") or now),
            updated_at=str(data.get("updated_at") or now),
        )


# ── ActorNode helper (agent_stage canvas 의 노드 1개) ──────────────────
def make_actor_node(
    name: str = "새 행위자",
    actor_type: str = "human",
    x: float = 0,
    y: float = 0,
    indicators: Optional[dict] = None,
    action_repertoire: Optional[list] = None,
    notes: str = "",
    role: str = "other",   # "self" (EIDOS) | "other"
) -> dict:
    """agent_stage canvas 의 노드 1개. legacy verb node 와 키가 겹치지 않게 \"kind\" 명시."""
    return {
        "id": _new_id("actor"),
        "kind": "actor",
        "x": float(x),
        "y": float(y),
        "name": name,
        "actor_type": actor_type if actor_type in ACTOR_TYPES else "human",
        "role": role,
        "indicators": dict(indicators or {}),
        "action_repertoire": list(action_repertoire or []),
        "notes": notes,
        "label_override": "",
    }


def make_actor_edge(from_id: str, to_id: str,
                    relation: str = "influences",
                    label: str = "") -> dict:
    """노드 간 영향 edge. relation ∈ {influences, observes, transacts_with, competes_with, ...}"""
    return {
        "id": _new_id("edge"),
        "from": from_id,
        "to": to_id,
        "relation": relation,
        "label": label or relation,
    }


# ── Stage ──────────────────────────────────────────────────────────────
@dataclass
class Stage:
    """1 stage 의 선언 구조. 런타임 belief 는 state.json 에 분리 저장."""

    id: str
    name: str
    goal: str
    halt_conditions: list[str] = field(default_factory=list)
    context_attachments: list[str] = field(default_factory=list)   # 절대 경로
    context_summary: str = ""
    context_summary_ts: str = ""
    context_attachments_signature: str = ""    # path + mtime sha — diff 감지용
    canvases: dict[str, dict] = field(default_factory=dict)        # canvas_id → StageCanvas dict
    main_canvas_id: str = ""
    action_registry_locals: list[dict] = field(default_factory=list)  # ActionSpec.serialize() list
    safety_mode: str = "normal"
    # [Phase 3-B 2026-05-26] scheduler 설정 영속화 — 재실행 시 옛 값 default
    scheduling: dict = field(default_factory=dict)
    # 키: interval_seconds, max_ticks, last_run_at, last_ticks_executed, last_halt_reason
    version: int = _VERSION
    created_at: str = ""
    updated_at: str = ""

    # ── helpers ──
    def add_canvas(self, canvas: StageCanvas, make_main: bool = False) -> None:
        self.canvases[canvas.id] = canvas.serialize()
        if make_main or not self.main_canvas_id:
            self.main_canvas_id = canvas.id

    def get_canvas(self, canvas_id: str) -> Optional[StageCanvas]:
        raw = self.canvases.get(canvas_id)
        if not raw:
            return None
        return StageCanvas.deserialize(raw)

    def main_canvas(self) -> Optional[StageCanvas]:
        if self.main_canvas_id:
            return self.get_canvas(self.main_canvas_id)
        if self.canvases:
            first = next(iter(self.canvases.values()))
            return StageCanvas.deserialize(first)
        return None

    def list_canvases(self) -> list[dict]:
        """canvas 메타 (id/name/kind/updated_at) — list 표시용."""
        out = []
        for cid, raw in self.canvases.items():
            out.append({
                "id": cid,
                "name": raw.get("name", "(이름 없음)"),
                "kind": raw.get("kind", "agent_stage"),
                "updated_at": raw.get("updated_at", ""),
                "is_main": cid == self.main_canvas_id,
            })
        return sorted(out, key=lambda x: (not x["is_main"], x.get("name", "")))

    def build_registry(self) -> Any:
        """ActionRegistry 인스턴스 (전역 default + 이 stage 의 locals)."""
        if ActionRegistry is None:
            return None
        return ActionRegistry.from_locals_dict(self.action_registry_locals)

    def update_registry_locals(self, registry: Any) -> None:
        """ActionRegistry 의 locals 를 stage 에 반영."""
        if registry is None or not hasattr(registry, "serialize_locals"):
            return
        try:
            self.action_registry_locals = registry.serialize_locals()
        except Exception as e:
            print(f"[Stage] update_registry_locals 실패 (graceful): {e}")

    def serialize(self) -> dict:
        return asdict(self)

    @classmethod
    def deserialize(cls, data: dict) -> "Stage":
        if not isinstance(data, dict):
            data = {}
        now = _now()
        return cls(
            id=str(data.get("id") or _new_id("stage")),
            name=str(data.get("name") or "(이름 없음)"),
            goal=str(data.get("goal") or ""),
            halt_conditions=list(data.get("halt_conditions", []) or []),
            context_attachments=list(data.get("context_attachments", []) or []),
            context_summary=str(data.get("context_summary") or ""),
            context_summary_ts=str(data.get("context_summary_ts") or ""),
            context_attachments_signature=str(data.get("context_attachments_signature") or ""),
            canvases=dict(data.get("canvases", {}) or {}),
            main_canvas_id=str(data.get("main_canvas_id") or ""),
            action_registry_locals=list(data.get("action_registry_locals", []) or []),
            safety_mode=str(data.get("safety_mode") or "normal"),
            scheduling=dict(data.get("scheduling") or {}),
            version=int(data.get("version") or _VERSION),
            created_at=str(data.get("created_at") or now),
            updated_at=str(data.get("updated_at") or now),
        )


# ── 헬퍼 ───────────────────────────────────────────────────────────────
def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _new_id(prefix: str = "item") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _safe_slug(text: str, fallback: str = "stage") -> str:
    """파일 시스템 안전 슬러그. 한글 OK·특수문자 _ 로 치환·길이 cap."""
    if not text:
        return fallback
    s = "".join(c if c.isalnum() or c in " _-" or "가" <= c <= "힯" else "_" for c in text)
    s = re.sub(r"_+", "_", s).strip("_ ").strip()
    if not s:
        return fallback
    return s[:60]


def stage_dir(stage_id: str) -> str:
    return os.path.join(_BASE_DIR, stage_id)


def _ensure_base() -> None:
    try:
        os.makedirs(_BASE_DIR, exist_ok=True)
    except Exception as e:
        print(f"[agent_stage_store] _ensure_base 실패 (graceful): {e}")


def _atomic_write(path: str, content: str) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)
        if os.path.exists(path):
            try:
                os.replace(tmp, path)
            except Exception:
                try:
                    os.remove(path)
                except Exception:
                    pass
                os.rename(tmp, path)
        else:
            os.rename(tmp, path)
        return True
    except Exception as e:
        print(f"[agent_stage_store] _atomic_write 실패 (graceful): {path} — {e}")
        return False


# ── CRUD ───────────────────────────────────────────────────────────────
def list_stages() -> list[dict]:
    """저장된 stage 메타 — [{id, name, goal, updated_at}]."""
    out: list[dict] = []
    try:
        if not os.path.isdir(_BASE_DIR):
            return []
        for entry in os.listdir(_BASE_DIR):
            d = os.path.join(_BASE_DIR, entry)
            if not os.path.isdir(d):
                continue
            stage_file = os.path.join(d, _STAGE_FILE)
            if not os.path.exists(stage_file):
                continue
            try:
                with open(stage_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    continue
                out.append({
                    "id": data.get("id") or entry,
                    "name": data.get("name", "(이름 없음)"),
                    "goal": data.get("goal", ""),
                    "updated_at": data.get("updated_at", ""),
                    "safety_mode": data.get("safety_mode", "normal"),
                })
            except Exception as e:
                print(f"[agent_stage_store] list_stages — {entry} 손상 skip (graceful): {e}")
                continue
        return sorted(out, key=lambda x: x.get("updated_at", ""), reverse=True)
    except Exception as e:
        print(f"[agent_stage_store] list_stages 실패 (graceful): {e}")
        return []


# [2026-05-27] EIDOS-self 의 default repertoire — 새 stage 와 기존 stage 마이그레이션 공통.
# 신규 action 추가 시 여기에 한 줄·load_stage 가 자동 마이그레이션.
_EIDOS_SELF_DEFAULT_REPERTOIRE = (
    "eidos.tool.llm.write",
    "eidos.tool.ask_user",
    "eidos.tool.draft_to_clipboard",   # 2026-05-27 — 답변 초안 + 자동 클립보드 (안전 default)
    "eidos.tool.pca.run_dag",           # 2026-05-27 — ToM↔PCA 통합 (사전 정의 DAG 위임)
    # [2026-05-30 시장조사] web 읽기 위주 브라우징 — 시장조사·자료조사 자율 수행.
    # click/fill 제외 (폼 제출 등 outward 위험) — navigate+read 위주. 외부효과지만
    # eidos_agent_runner._DEFAULT_AUTORUN_EXTERNAL 로 normal 모드 자동 실행됨.
    "eidos.tool.web.search",            # 2026-06-01 — SerpAPI 검색 (google navigate 대신 우선)
    "eidos.tool.web.navigate",
    "eidos.tool.web.read_page",
    "eidos.tool.web.scroll",
    "eidos.tool.web.go_back",
    "eidos.tool.web.wait",
    # [2026-05-27 자율 개발 모드] dev.* — 의뢰자 코딩 의뢰 자율 처리
    "eidos.tool.dev.scaffold_project",
    "eidos.tool.dev.write_code",
    "eidos.tool.dev.run_tests",
    "eidos.tool.dev.package_deliverable",
    "eidos.meta.observe",
    "eidos.meta.no_op",
    "eidos.meta.reflect",
    # [2026-05-28 Phase 14] 메타 인지 action — EIDOS 가 자기 한계 알고 위임/역제안/역질문.
    # prompt 안내 X·LLM 의 *결정 메뉴* 안에 정식 등장 → ActionValue 학습도 자동.
    "eidos.meta.delegate_to_user",     # "승준 씨가 직접 해주세요" — 능력 부족 명시 위임
    "eidos.meta.counter_propose",      # 사용자 방식이 비효율적 → 완곡 거부 + 더 나은 방향
    "eidos.meta.clarify_question",     # 강화된 역질문 (ask_user 보다 구조화·여러 질문)
)


def _migrate_eidos_self_repertoire(stage: "Stage") -> bool:
    """기존 stage 의 EIDOS-self actor repertoire 에 새 action 자동 append.

    Phase 별로 새 action (예: draft_to_clipboard) 추가될 때마다 이 헬퍼가 idempotent
    하게 마이그레이션. 이미 있으면 skip·없으면 append.

    Stage.canvases 는 dict[str, dict] (raw serialized) — .nodes 가 아니라 ["nodes"] 접근.

    Returns: True = 변경됨 (저장 권장)·False = 변화 없음.
    """
    if stage is None:
        return False
    changed = False
    try:
        for _cid, canvas_dict in (stage.canvases or {}).items():
            if not isinstance(canvas_dict, dict):
                continue
            nodes = canvas_dict.get("nodes") or []
            for n in nodes:
                if not isinstance(n, dict):
                    continue
                if not (n.get("kind") == "actor" and n.get("role") == "self"):
                    continue
                rep = list(n.get("action_repertoire") or [])
                node_changed = False
                for default_aid in _EIDOS_SELF_DEFAULT_REPERTOIRE:
                    if default_aid not in rep:
                        rep.append(default_aid)
                        node_changed = True
                if node_changed:
                    n["action_repertoire"] = rep
                    changed = True
    except Exception as e:
        print(f"[agent_stage_store] _migrate_eidos_self_repertoire 실패 (graceful): {e}")
        return False
    return changed


def load_stage(stage_id: str) -> Optional[Stage]:
    """stage_id 로 Stage 로드. 없으면 None.

    [2026-05-27] 자동 마이그레이션 — EIDOS-self repertoire 에 _EIDOS_SELF_DEFAULT_REPERTOIRE
    중 빠진 action 있으면 자동 append + 저장. idempotent·load 직후 1회만.
    """
    if not stage_id:
        return None
    path = os.path.join(stage_dir(stage_id), _STAGE_FILE)
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        stage = Stage.deserialize(data)
        # main_canvas_id 가 비어있으면 첫 canvas 로 보강
        if not stage.main_canvas_id and stage.canvases:
            stage.main_canvas_id = next(iter(stage.canvases.keys()))
        # [2026-05-27] EIDOS-self repertoire 자동 마이그레이션
        try:
            if _migrate_eidos_self_repertoire(stage):
                save_stage(stage)
                print(f"[agent_stage_store] {stage_id} — EIDOS-self repertoire 마이그레이션 완료")
        except Exception as _e_mig:
            print(f"[agent_stage_store] 마이그레이션 실패 (graceful·load 는 진행): {_e_mig}")
        return stage
    except Exception as e:
        print(f"[agent_stage_store] load_stage 실패 ({stage_id}, graceful): {e}")
        return None


def save_stage(stage: Stage) -> bool:
    """Stage 디스크 저장 (atomic)."""
    if not stage or not stage.id:
        return False
    _ensure_base()
    stage.updated_at = _now()
    stage.version = _VERSION
    path = os.path.join(stage_dir(stage.id), _STAGE_FILE)
    try:
        payload = stage.serialize()
        return _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"[agent_stage_store] save_stage 실패 (graceful): {e}")
        return False


def delete_stage(stage_id: str) -> bool:
    """Stage 폴더 전체 삭제 (attachments 포함)."""
    if not stage_id:
        return False
    d = stage_dir(stage_id)
    try:
        if os.path.isdir(d):
            shutil.rmtree(d)
            return True
    except Exception as e:
        print(f"[agent_stage_store] delete_stage 실패 ({stage_id}, graceful): {e}")
    return False


def make_empty_stage(name: str = "(이름 없음)",
                     goal: str = "",
                     halt_conditions: Optional[list[str]] = None) -> Stage:
    """새 stage — main canvas 1개 자동 포함 + EIDOS-self actor 자동 추가."""
    sid = _new_id("stage_" + _safe_slug(name, "anon"))
    now = _now()
    stage = Stage(
        id=sid,
        name=name,
        goal=goal,
        halt_conditions=list(halt_conditions or []),
        created_at=now,
        updated_at=now,
    )
    # 메인 canvas + EIDOS 자기 노드 자동 등장
    main = StageCanvas(
        id=_new_id("canvas"),
        name="메인 무대",
        kind="agent_stage",
        created_at=now,
        updated_at=now,
    )
    eidos_node = make_actor_node(
        name="EIDOS",
        actor_type="self",
        role="self",
        x=400, y=300,
        notes="이 무대의 주인공 — 자기 자신.",
        # [2026-05-27] _EIDOS_SELF_DEFAULT_REPERTOIRE 와 일치 — 신규 action 추가 시
        # 그 list 만 갱신해도 새 stage·기존 stage 모두 자동 반영.
        action_repertoire=list(_EIDOS_SELF_DEFAULT_REPERTOIRE),
    )
    main.nodes.append(eidos_node)
    stage.add_canvas(main, make_main=True)
    return stage


# ── State (런타임 belief) ──────────────────────────────────────────────
def load_state(stage_id: str) -> dict:
    """state.json — 없으면 빈 dict."""
    path = os.path.join(stage_dir(stage_id), _STATE_FILE)
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[agent_stage_store] load_state 실패 ({stage_id}, graceful): {e}")
        return {}


def save_state(stage_id: str, state: dict) -> bool:
    if not stage_id or state is None:
        return False
    path = os.path.join(stage_dir(stage_id), _STATE_FILE)
    try:
        state = dict(state)
        state["_updated_at"] = _now()
        return _atomic_write(path, json.dumps(state, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"[agent_stage_store] save_state 실패 (graceful): {e}")
        return False


# ── [2026-05-27 inbox-poller] actor.indicators atomic merge ────────────
def update_actor_indicators(
    stage_id: str,
    actor_name: str,
    indicators: dict,
    *,
    create_if_missing: bool = True,
) -> bool:
    """state.json 의 특정 actor 의 indicators 에 dict merge — atomic load/save.

    Args:
        stage_id: 대상 stage id
        actor_name: 갱신할 actor 이름 (state["actors"][actor_name])
        indicators: merge 할 indicator dict ({"has_new_message": True, ...})
        create_if_missing: actor 없으면 새로 생성 (role="other"·actor_type="human")

    Returns: True 성공·False 실패. actor 없고 create_if_missing=False 면 False.
    """
    if not stage_id or not actor_name or not isinstance(indicators, dict):
        return False
    state = load_state(stage_id)
    if not isinstance(state, dict):
        state = {}
    actors = dict(state.get("actors") or {})
    actor = actors.get(actor_name)
    if actor is None:
        if not create_if_missing:
            return False
        actor = {
            "indicators": {},
            "role": "other",
            "actor_type": "human",
            "action_repertoire": [],
        }
    else:
        actor = dict(actor)
    new_ind = dict(actor.get("indicators") or {})
    new_ind.update({str(k): v for k, v in indicators.items()})
    actor["indicators"] = new_ind
    actors[actor_name] = actor
    state["actors"] = actors
    return save_state(stage_id, state)


def update_meta_state(stage_id: str, key: str, value) -> bool:
    """state.json 의 top-level meta key 갱신 (예: _inbox_poller 의 last_text_hash 저장)."""
    if not stage_id or not key:
        return False
    state = load_state(stage_id)
    if not isinstance(state, dict):
        state = {}
    state[str(key)] = value
    return save_state(stage_id, state)


# ── History (append-only event log) ────────────────────────────────────
def append_history(stage_id: str, event: dict) -> bool:
    """jsonl 한 줄 추가. 결정·예측·결과·관찰 모두 동일 파일."""
    if not stage_id or not isinstance(event, dict):
        return False
    path = os.path.join(stage_dir(stage_id), _HISTORY_FILE)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if "ts" not in event:
            event = dict(event)
            event["ts"] = _now()
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        print(f"[agent_stage_store] append_history 실패 (graceful): {e}")
        return False


def read_history(stage_id: str, limit: Optional[int] = None) -> list[dict]:
    """최근 limit 개 (None = 전부)·시간 순. 손상 라인 skip."""
    path = os.path.join(stage_dir(stage_id), _HISTORY_FILE)
    out: list[dict] = []
    try:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        if limit and limit > 0 and len(out) > limit:
            out = out[-limit:]
        return out
    except Exception as e:
        print(f"[agent_stage_store] read_history 실패 (graceful): {e}")
        return out


# ── Attachment 관리 (옵션 C — 1회 분석 + 진행 중 추가/교체) ──────────
def attachments_dir(stage_id: str) -> str:
    return os.path.join(stage_dir(stage_id), _ATTACHMENT_DIR)


def add_attachment(stage_id: str, src_path: str, copy: bool = True) -> Optional[str]:
    """첨부 파일 등록. copy=True 면 stage 폴더에 복제 (이동·삭제 견딤)."""
    if not stage_id or not src_path or not os.path.exists(src_path):
        return None
    try:
        adir = attachments_dir(stage_id)
        os.makedirs(adir, exist_ok=True)
        if not copy:
            return os.path.abspath(src_path)
        base = os.path.basename(src_path)
        # 중복 방지 — _2 _3 ... suffix
        target = os.path.join(adir, base)
        if os.path.exists(target):
            stem, ext = os.path.splitext(base)
            for i in range(2, 100):
                candidate = os.path.join(adir, f"{stem}_{i}{ext}")
                if not os.path.exists(candidate):
                    target = candidate
                    break
        shutil.copy2(src_path, target)
        return os.path.abspath(target)
    except Exception as e:
        print(f"[agent_stage_store] add_attachment 실패 ({src_path}, graceful): {e}")
        return None


def compute_attachments_signature(paths: list[str]) -> str:
    """path 목록의 변경 감지용 signature — path+mtime+size 의 해시."""
    import hashlib
    h = hashlib.sha256()
    try:
        for p in sorted(paths or []):
            try:
                st = os.stat(p)
                token = f"{p}|{int(st.st_mtime)}|{st.st_size}"
            except Exception:
                token = f"{p}|missing"
            h.update(token.encode("utf-8"))
        return h.hexdigest()[:16]
    except Exception:
        return ""


__all__ = [
    "Stage", "StageCanvas", "ACTOR_TYPES", "SAFETY_MODES",
    "make_actor_node", "make_actor_edge", "make_empty_stage",
    "list_stages", "load_stage", "save_stage", "delete_stage",
    "load_state", "save_state",
    "append_history", "read_history",
    "stage_dir", "attachments_dir",
    "add_attachment", "compute_attachments_signature",
]
