# eidos_goal_tree.py
# [2026-05-27 Phase 9] Hierarchical Goal Decomposition — 1년 목표 → 분기 → 월 → 주.
#
# Stage.goal 은 string 1개라서 6개월~1년 목표를 잊지 않기에 부족함. Phase 9 는 stage 옆에
# GoalTree 위계 (Milestone 계층) 를 두어:
#   - 1년 목표 (root)         → goal_tree.root_goal_id 가 가리킴 · title 은 Stage.goal 과 sync
#   - 분기 milestone (4개)     → root 의 children
#   - 월 milestone (3개씩)     → quarter 의 children
#   - 주/태스크 (자유)          → month 의 children
#
# 디자인 원칙 (Phase 1~8 와 동일):
#   1. LLM 호출 0 (Phase 9-A). 휴리스틱 progress 만. decompose_with_llm_async 는 후속.
#   2. graceful — 손상되면 빈 tree 반환·throw 안 함.
#   3. _atomic_write — Stage·belief 와 같은 패턴.
#   4. 모든 임계값·default 는 모듈 상단 상수.
#   5. backwards compat — deserialize 가 빈 필드 fallback.
#
# 저장 위치: eidos_files/agents/{stage_id}/goal_tree.json — Stage 와 같은 폴더.
# Stage 변경 시 root.title sync 는 별도 sync_root_with_stage() 헬퍼.

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from eidos_agent_stage_store import Stage


# ── 상수 ───────────────────────────────────────────────────────────────
_GOAL_FILE = "goal_tree.json"
_VERSION = 1

HORIZONS = ["year", "quarter", "month", "week", "task"]
MILESTONE_STATUS = ["active", "done", "abandoned", "superseded"]

# horizon → 권장 children horizon (decompose 휴리스틱 prior·LLM 옵션과 무관)
_HORIZON_CHILDREN = {
    "year": "quarter",
    "quarter": "month",
    "month": "week",
    "week": "task",
    "task": "task",
}

# evaluate_progress 휴리스틱 — success_criteria 단어가 history 텍스트에 등장하면 매칭
# 1개 매칭 시 + _CRITERION_HIT_PARTIAL · 모든 매칭 시 1.0
_CRITERION_HIT_PARTIAL = 0.5
_CRITERION_MIN_TOKEN_LEN = 2

# prompt brief 에 표시할 기본 깊이 (root=0)
_DEFAULT_BRIEF_DEPTH = 2


# ── stage_dir 재사용 (single source of truth) ──────────────────────────
def _stage_dir(stage_id: str) -> str:
    """eidos_agent_stage_store.stage_dir 가 있으면 재사용 — 없으면 fallback."""
    try:
        from eidos_agent_stage_store import stage_dir
        return stage_dir(stage_id)
    except Exception:
        # graceful fallback — 옛 경로 패턴 그대로
        return os.path.join("eidos_files", "agents", stage_id)


# ── Milestone ──────────────────────────────────────────────────────────
@dataclass
class Milestone:
    """1개 목표 단위 — 1년/분기/월/주/태스크. 계층 형성 (parent_id + children_ids).

    핵심 의미:
      - leaf (children_ids 비어있음): success_criteria 로 평가됨
      - internal: progress 는 children 의 평균 (자동)
      - status=done: 진행도 자동 1.0·user 가 완료 표시한 그 시점 기록
      - status=abandoned: 포기·progress 동결
      - status=superseded: 상위 목표 변경 등으로 무효 (history 보존)
    """

    id: str
    title: str
    description: str = ""
    horizon: str = "task"            # HORIZONS 중 하나
    parent_id: str = ""              # root 면 "" (빈)
    children_ids: list[str] = field(default_factory=list)
    target_date: str = ""            # "2026-09-30" — 빈 가능
    started_at: str = ""
    completed_at: str = ""
    status: str = "active"           # MILESTONE_STATUS
    progress: float = 0.0            # 0~1 — manual 또는 children 평균
    progress_is_manual: bool = False # True 면 children 평균 override
    success_criteria: list[str] = field(default_factory=list)
    notes: str = ""
    source: str = "user"             # user | llm_decomposed | proactive

    def serialize(self) -> dict:
        return asdict(self)

    @classmethod
    def deserialize(cls, data: dict) -> "Milestone":
        if not isinstance(data, dict):
            data = {}
        now = _now()
        hz = str(data.get("horizon") or "task")
        if hz not in HORIZONS:
            hz = "task"
        st = str(data.get("status") or "active")
        if st not in MILESTONE_STATUS:
            st = "active"
        try:
            prog = float(data.get("progress", 0.0))
        except Exception:
            prog = 0.0
        return cls(
            id=str(data.get("id") or _new_id("mile")),
            title=str(data.get("title") or "(이름 없음)"),
            description=str(data.get("description") or ""),
            horizon=hz,
            parent_id=str(data.get("parent_id") or ""),
            children_ids=list(data.get("children_ids", []) or []),
            target_date=str(data.get("target_date") or ""),
            started_at=str(data.get("started_at") or now),
            completed_at=str(data.get("completed_at") or ""),
            status=st,
            progress=max(0.0, min(1.0, prog)),
            progress_is_manual=bool(data.get("progress_is_manual") or False),
            success_criteria=list(data.get("success_criteria", []) or []),
            notes=str(data.get("notes") or ""),
            source=str(data.get("source") or "user"),
        )


# ── GoalTree ───────────────────────────────────────────────────────────
@dataclass
class GoalTree:
    """Stage 1개의 목표 위계 전체. root_goal_id 가 root milestone 가리킴.

    milestones 는 dict[id, raw] — 빠른 lookup. children 관계는 양방향 (parent_id + children_ids).
    트리 모양은 LLM 또는 사용자 자유 — 4 horizon strict 강제 X (year→month 바로 가능).
    """

    stage_id: str
    root_goal_id: str = ""           # root milestone id (Stage.goal 과 sync)
    milestones: dict[str, dict] = field(default_factory=dict)
    horizon_end: str = ""            # "2027-05-27" — 트리 전체 기한 (옵션)
    version: int = _VERSION
    created_at: str = ""
    updated_at: str = ""

    def serialize(self) -> dict:
        return asdict(self)

    @classmethod
    def deserialize(cls, data: dict) -> "GoalTree":
        if not isinstance(data, dict):
            data = {}
        now = _now()
        return cls(
            stage_id=str(data.get("stage_id") or ""),
            root_goal_id=str(data.get("root_goal_id") or ""),
            milestones=dict(data.get("milestones") or {}),
            horizon_end=str(data.get("horizon_end") or ""),
            version=int(data.get("version") or _VERSION),
            created_at=str(data.get("created_at") or now),
            updated_at=str(data.get("updated_at") or now),
        )

    # ── 간단 조회 helpers ──
    def get(self, milestone_id: str) -> Optional[Milestone]:
        raw = self.milestones.get(milestone_id)
        if not raw:
            return None
        return Milestone.deserialize(raw)

    def root(self) -> Optional[Milestone]:
        if not self.root_goal_id:
            return None
        return self.get(self.root_goal_id)

    def all_milestones(self) -> list[Milestone]:
        return [Milestone.deserialize(r) for r in self.milestones.values()]

    def children_of(self, milestone_id: str) -> list[Milestone]:
        parent = self.get(milestone_id)
        if parent is None:
            return []
        out: list[Milestone] = []
        for cid in parent.children_ids:
            child = self.get(cid)
            if child is not None:
                out.append(child)
        return out

    def path_to(self, milestone_id: str) -> list[Milestone]:
        """root 부터 이 milestone 까지 경로 (포함)."""
        out: list[Milestone] = []
        cur = self.get(milestone_id)
        # 무한 루프 방지 (parent_id 손상 대비)
        seen = set()
        while cur is not None and cur.id not in seen:
            seen.add(cur.id)
            out.append(cur)
            if not cur.parent_id:
                break
            cur = self.get(cur.parent_id)
        return list(reversed(out))


# ── 헬퍼 ───────────────────────────────────────────────────────────────
def _now() -> str:
    return _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _new_id(prefix: str = "mile") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


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
        print(f"[goal_tree] _atomic_write 실패 (graceful): {path} — {e}")
        return False


def _goal_path(stage_id: str) -> str:
    return os.path.join(_stage_dir(stage_id), _GOAL_FILE)


def _put(tree: GoalTree, m: Milestone) -> None:
    """milestone 저장 + tree.updated_at 갱신."""
    tree.milestones[m.id] = m.serialize()
    tree.updated_at = _now()


# ── CRUD ───────────────────────────────────────────────────────────────
def new_goal_tree(
    stage_id: str,
    root_title: str = "",
    horizon: str = "year",
    horizon_end: str = "",
) -> GoalTree:
    """빈 tree 생성 — root milestone 1개 자동 등장.

    root_title 비어있으면 root 도 안 만들고 빈 tree. (Stage.goal 비어있을 때 안전.)
    """
    if not stage_id:
        raise ValueError("stage_id 필수")
    if horizon not in HORIZONS:
        horizon = "year"
    now = _now()
    tree = GoalTree(
        stage_id=stage_id,
        horizon_end=horizon_end,
        created_at=now,
        updated_at=now,
    )
    if root_title:
        root = Milestone(
            id=_new_id("mile_root"),
            title=root_title.strip()[:200],
            horizon=horizon,
            parent_id="",
            started_at=now,
            target_date=horizon_end,
            source="user",
        )
        tree.milestones[root.id] = root.serialize()
        tree.root_goal_id = root.id
    return tree


def load_goal_tree(stage_id: str) -> Optional[GoalTree]:
    """stage_id 의 goal_tree.json 로드. 없으면 None (생성은 호출자 책임)."""
    if not stage_id:
        return None
    path = _goal_path(stage_id)
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tree = GoalTree.deserialize(data)
        # stage_id 가 비어있으면 호출 stage_id 로 보강 (옛 데이터 호환)
        if not tree.stage_id:
            tree.stage_id = stage_id
        return tree
    except Exception as e:
        print(f"[goal_tree] load 실패 ({stage_id}, graceful): {e}")
        return None


def save_goal_tree(tree: GoalTree) -> bool:
    """디스크 저장 (atomic)."""
    if tree is None or not tree.stage_id:
        return False
    tree.updated_at = _now()
    tree.version = _VERSION
    path = _goal_path(tree.stage_id)
    try:
        payload = tree.serialize()
        return _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"[goal_tree] save 실패 (graceful): {e}")
        return False


def delete_goal_tree(stage_id: str) -> bool:
    """goal_tree.json 삭제 (Stage 자체는 보존)."""
    if not stage_id:
        return False
    path = _goal_path(stage_id)
    try:
        if os.path.exists(path):
            os.remove(path)
        return True
    except Exception as e:
        print(f"[goal_tree] delete 실패 (graceful): {e}")
        return False


# ── Milestone 조작 ─────────────────────────────────────────────────────
def add_milestone(
    tree: GoalTree,
    title: str,
    parent_id: str = "",
    horizon: Optional[str] = None,
    target_date: str = "",
    description: str = "",
    success_criteria: Optional[list[str]] = None,
    source: str = "user",
) -> Optional[Milestone]:
    """새 milestone 추가. parent_id 주면 자동으로 부모의 children_ids 에 등록.

    parent_id 빈 문자열·tree.root_goal_id 비어있으면 root 로 등록.
    """
    if tree is None or not title.strip():
        return None
    now = _now()

    # horizon 추정 — 명시 안 했으면 부모의 children horizon 권장값
    if horizon is None or horizon not in HORIZONS:
        if parent_id:
            parent = tree.get(parent_id)
            if parent is not None:
                horizon = _HORIZON_CHILDREN.get(parent.horizon, "task")
            else:
                horizon = "task"
        else:
            # root 새로 만드는 경우 — year default
            horizon = "year"

    m = Milestone(
        id=_new_id("mile"),
        title=title.strip()[:200],
        description=description[:500] if description else "",
        horizon=horizon,
        parent_id=parent_id or "",
        started_at=now,
        target_date=target_date,
        success_criteria=list(success_criteria or []),
        source=source if source in ("user", "llm_decomposed", "proactive") else "user",
    )

    # root 등록 분기
    if not parent_id:
        if not tree.root_goal_id:
            tree.root_goal_id = m.id
        # parent 없는 milestone 도 추가 가능 (orphan — 사용자가 나중에 연결)
    else:
        parent = tree.get(parent_id)
        if parent is None:
            print(f"[goal_tree] add_milestone — parent {parent_id} 없음 (orphan 으로 등록)")
        else:
            if m.id not in parent.children_ids:
                parent.children_ids.append(m.id)
                _put(tree, parent)

    _put(tree, m)
    return m


def update_milestone(
    tree: GoalTree,
    milestone_id: str,
    *,
    title: Optional[str] = None,
    description: Optional[str] = None,
    target_date: Optional[str] = None,
    success_criteria: Optional[list[str]] = None,
    notes: Optional[str] = None,
) -> bool:
    """milestone 의 메타 필드 갱신. status / progress / parent_id 변경은 별 함수."""
    m = tree.get(milestone_id) if tree else None
    if m is None:
        return False
    if title is not None:
        m.title = title.strip()[:200]
    if description is not None:
        m.description = description[:500]
    if target_date is not None:
        m.target_date = target_date
    if success_criteria is not None:
        m.success_criteria = list(success_criteria)
    if notes is not None:
        m.notes = notes[:1000]
    _put(tree, m)
    return True


def set_milestone_status(
    tree: GoalTree,
    milestone_id: str,
    new_status: str,
) -> bool:
    """status 변경 — done 이면 completed_at 자동·progress=1.0 (manual 아니면)."""
    if new_status not in MILESTONE_STATUS:
        return False
    m = tree.get(milestone_id) if tree else None
    if m is None:
        return False
    now = _now()
    m.status = new_status
    if new_status == "done":
        m.completed_at = now
        if not m.progress_is_manual:
            m.progress = 1.0
    elif new_status in ("abandoned", "superseded"):
        # progress 동결 (마지막 값 보존)
        m.completed_at = now
    else:  # active 로 복귀
        m.completed_at = ""
    _put(tree, m)
    return True


def set_milestone_progress(
    tree: GoalTree,
    milestone_id: str,
    progress: float,
    manual: bool = True,
) -> bool:
    """progress 직접 설정. manual=True 면 children 평균 override 활성."""
    m = tree.get(milestone_id) if tree else None
    if m is None:
        return False
    m.progress = max(0.0, min(1.0, float(progress)))
    m.progress_is_manual = bool(manual)
    _put(tree, m)
    return True


def reparent_milestone(
    tree: GoalTree,
    milestone_id: str,
    new_parent_id: str,
) -> bool:
    """milestone 의 parent 변경. 옛 parent 의 children_ids 에서 제거·새 parent 에 등록.

    순환 방지 — new_parent_id 가 milestone_id 의 후손이면 거부.
    """
    if not tree or not milestone_id:
        return False
    m = tree.get(milestone_id)
    if m is None:
        return False
    # 순환 체크 — new_parent_id 가 m 의 후손인지
    if new_parent_id:
        # m 후손 모두 수집
        descendants: set[str] = set()
        stack = list(m.children_ids)
        while stack:
            cid = stack.pop()
            if cid in descendants:
                continue
            descendants.add(cid)
            child = tree.get(cid)
            if child is not None:
                stack.extend(child.children_ids)
        if new_parent_id in descendants or new_parent_id == milestone_id:
            print(f"[goal_tree] reparent 거부 — 순환 ({milestone_id} → {new_parent_id})")
            return False

    # 옛 parent 의 children 에서 제거
    if m.parent_id:
        old_parent = tree.get(m.parent_id)
        if old_parent is not None and milestone_id in old_parent.children_ids:
            old_parent.children_ids.remove(milestone_id)
            _put(tree, old_parent)
    # 새 parent 의 children 에 추가
    m.parent_id = new_parent_id or ""
    if new_parent_id:
        new_parent = tree.get(new_parent_id)
        if new_parent is not None and milestone_id not in new_parent.children_ids:
            new_parent.children_ids.append(milestone_id)
            _put(tree, new_parent)
    _put(tree, m)
    return True


def delete_milestone(tree: GoalTree, milestone_id: str, cascade: bool = False) -> int:
    """milestone 삭제. cascade=True 면 후손 전체 삭제·False 면 후손은 parent_id 만 비움 (orphan).

    Returns: 삭제된 milestone 개수.
    """
    if not tree or not milestone_id:
        return 0
    m = tree.get(milestone_id)
    if m is None:
        return 0
    n_removed = 0
    # 후손 처리
    for cid in list(m.children_ids):
        child = tree.get(cid)
        if child is None:
            continue
        if cascade:
            n_removed += delete_milestone(tree, cid, cascade=True)
        else:
            child.parent_id = ""
            _put(tree, child)
    # 옛 parent 의 children 에서 제거
    if m.parent_id:
        parent = tree.get(m.parent_id)
        if parent is not None and milestone_id in parent.children_ids:
            parent.children_ids.remove(milestone_id)
            _put(tree, parent)
    # 본인 삭제
    tree.milestones.pop(milestone_id, None)
    n_removed += 1
    # root 였으면 root_goal_id 비움 (다음 root 는 호출자 책임)
    if tree.root_goal_id == milestone_id:
        tree.root_goal_id = ""
    tree.updated_at = _now()
    return n_removed


# ── progress 자동 계산 ────────────────────────────────────────────────
def recompute_progress(tree: GoalTree) -> int:
    """internal milestone (children 있는) 의 progress 를 children 평균으로 재계산.

    manual=True 인 milestone 은 그대로 (override 보존). 후위 순회 (depth-first)
    로 leaf 부터 올라가며 갱신.

    Returns: 변경된 milestone 개수.
    """
    if tree is None or not tree.root_goal_id:
        return 0
    n_changed = 0
    # 후위 순회 — root 부터 DFS 로 leaf 찾고, 올라오면서 평균
    visited: set[str] = set()
    order: list[str] = []

    def _dfs(mid: str) -> None:
        if mid in visited:
            return
        visited.add(mid)
        m = tree.get(mid)
        if m is None:
            return
        for cid in m.children_ids:
            _dfs(cid)
        order.append(mid)

    _dfs(tree.root_goal_id)
    # 연결 안 된 orphan 도 처리
    for mid in list(tree.milestones.keys()):
        if mid not in visited:
            _dfs(mid)

    for mid in order:
        m = tree.get(mid)
        if m is None or not m.children_ids or m.progress_is_manual:
            continue
        if m.status in ("done",):
            # done 이면 자동 1.0 (children 무시)
            if m.progress != 1.0:
                m.progress = 1.0
                _put(tree, m)
                n_changed += 1
            continue
        # children 평균 (abandoned/superseded 는 제외)
        vals: list[float] = []
        for cid in m.children_ids:
            child = tree.get(cid)
            if child is None:
                continue
            if child.status in ("abandoned", "superseded"):
                continue
            vals.append(child.progress)
        if not vals:
            continue
        new_p = round(sum(vals) / len(vals), 3)
        if abs(m.progress - new_p) > 1e-6:
            m.progress = new_p
            _put(tree, m)
            n_changed += 1
    return n_changed


# ── 휴리스틱 progress 평가 (history 기반) ─────────────────────────────
_WORD_RE = re.compile(r"[A-Za-z가-힣][A-Za-z가-힣0-9_\-]+")


def _extract_criterion_tokens(criterion: str) -> list[str]:
    """success_criteria 1줄 → 매칭에 쓸 토큰 list. _CRITERION_MIN_TOKEN_LEN 이상."""
    if not criterion:
        return []
    tokens = _WORD_RE.findall(criterion)
    return [t.lower() for t in tokens if len(t) >= _CRITERION_MIN_TOKEN_LEN]


def evaluate_progress_from_history(
    tree: GoalTree,
    history_events: list[dict],
    *,
    text_keys: tuple[str, ...] = ("summary", "content", "result", "title", "notes"),
) -> dict[str, float]:
    """history.jsonl events 의 텍스트를 success_criteria 와 매칭해 leaf milestone progress 추정.

    각 leaf milestone:
      - success_criteria 없으면 변화 없음
      - 토큰 N개 추출 후 history 텍스트에 K개 등장 →
          K == N  → progress = 1.0
          0 < K < N → progress = _CRITERION_HIT_PARTIAL · K / N (최소 _PARTIAL)
          K == 0  → 변화 없음 (manual 진행도 보존)
      - status 가 done/abandoned/superseded 면 변화 없음

    progress_is_manual=True 인 leaf 도 갱신 X (사용자 override 존중).

    Returns: {milestone_id: new_progress} — 변화한 leaf 만.
    """
    if tree is None or not history_events:
        return {}
    # 모든 텍스트 한 string 으로 연결 (lowercase)
    text_chunks: list[str] = []
    for ev in history_events:
        if not isinstance(ev, dict):
            continue
        for k in text_keys:
            v = ev.get(k)
            if isinstance(v, str) and v.strip():
                text_chunks.append(v.lower())
        # dict 값 (decision.args 등) 도 한 단계 더
        for v in ev.values():
            if isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, str) and vv.strip():
                        text_chunks.append(vv.lower())
    haystack = " \n ".join(text_chunks)
    if not haystack:
        return {}

    changes: dict[str, float] = {}
    for m in tree.all_milestones():
        # leaf 만 (children 있으면 recompute_progress 가 처리)
        if m.children_ids:
            continue
        if m.status in ("done", "abandoned", "superseded"):
            continue
        if m.progress_is_manual:
            continue
        if not m.success_criteria:
            continue
        total_tokens = 0
        hit_tokens = 0
        for criterion in m.success_criteria:
            toks = _extract_criterion_tokens(criterion)
            if not toks:
                continue
            for t in toks:
                total_tokens += 1
                if t in haystack:
                    hit_tokens += 1
        if total_tokens == 0 or hit_tokens == 0:
            continue
        if hit_tokens == total_tokens:
            new_p = 1.0
        else:
            ratio = hit_tokens / total_tokens
            new_p = max(_CRITERION_HIT_PARTIAL * ratio, ratio * 0.5)
            new_p = min(0.99, new_p)  # 부분 매칭은 1.0 까지 못 감 (done 은 명시적)
        new_p = round(new_p, 3)
        if abs(m.progress - new_p) > 1e-6:
            m.progress = new_p
            _put(tree, m)
            changes[m.id] = new_p
    # NOTE: internal milestone 재계산은 호출자 책임 (recompute_progress 별도 호출).
    # 같은 트랜잭션에서 두 metric (leaf changes·internal n_changed) 을 분리해 보고하기 위함.
    return changes


# ── Stage.goal 과 root 동기화 ──────────────────────────────────────────
def sync_root_with_stage(tree: GoalTree, stage: "Stage") -> bool:
    """Stage.goal 이 변경됐을 때 root milestone 의 title 을 따라가게.

    root 가 없으면 새로 생성 (Stage.goal 있을 때만). Stage.goal 빈 문자열이면
    변화 없음 (root 삭제는 명시적 delete_milestone 으로).

    Returns: 변경 발생했는지.
    """
    if tree is None or stage is None:
        return False
    goal_text = (stage.goal or "").strip()
    if not goal_text:
        return False
    if not tree.root_goal_id:
        # root 생성
        m = add_milestone(
            tree,
            title=goal_text,
            parent_id="",
            horizon="year",
            description="Stage.goal 자동 sync 로 생성된 root milestone",
        )
        return m is not None
    root = tree.root()
    if root is None:
        return False
    if root.title != goal_text[:200]:
        root.title = goal_text[:200]
        _put(tree, root)
        return True
    return False


# ── 조회 헬퍼 ──────────────────────────────────────────────────────────
def get_active_milestones(
    tree: GoalTree,
    horizon_filter: Optional[list[str]] = None,
) -> list[Milestone]:
    """active 상태 milestone 만. horizon_filter 주면 필터링.

    정렬: target_date 가까운 순 → importance (progress 낮음) 순.
    """
    if tree is None:
        return []
    out: list[Milestone] = []
    for m in tree.all_milestones():
        if m.status != "active":
            continue
        if horizon_filter and m.horizon not in horizon_filter:
            continue
        out.append(m)
    out.sort(key=lambda x: (x.target_date or "9999-12-31", x.progress))
    return out


# ── prompt brief ───────────────────────────────────────────────────────
def as_prompt_brief(
    tree: GoalTree,
    depth: int = _DEFAULT_BRIEF_DEPTH,
    max_per_level: int = 6,
) -> str:
    """LLM prompt 에 inject 할 markdown. root + 하위 depth 단까지 보여줌.

    active milestone 만 (done/abandoned 는 표시 X). progress·target_date 명시.
    """
    if tree is None or not tree.root_goal_id:
        return ""
    root = tree.root()
    if root is None:
        return ""

    lines: list[str] = []
    lines.append("## 목표 트리 (활성 milestone)")
    lines.append(f"🎯 **{root.title}** (horizon={root.horizon}·progress={root.progress:.2f}"
                 + (f"·target={root.target_date}" if root.target_date else "") + ")")
    if root.description:
        lines.append(f"   _{root.description[:120]}_")

    def _render(milestone: Milestone, level: int) -> None:
        if level > depth:
            return
        children = []
        for cid in milestone.children_ids:
            ch = tree.get(cid)
            if ch is not None and ch.status == "active":
                children.append(ch)
        # 진행도 낮은 순 (가장 시급)
        children.sort(key=lambda c: (c.target_date or "9999-12-31", c.progress))
        for ch in children[:max_per_level]:
            indent = "  " * level
            marker = "▣" if ch.progress >= 1.0 else ("▤" if ch.progress >= 0.5 else "▢")
            extras = []
            if ch.target_date:
                extras.append(f"~{ch.target_date}")
            extras.append(f"{ch.progress*100:.0f}%")
            extras_s = " · ".join(extras)
            lines.append(f"{indent}- {marker} {ch.title} ({ch.horizon}·{extras_s})")
            if ch.description and level == 1:
                lines.append(f"{indent}    _{ch.description[:80]}_")
            _render(ch, level + 1)

    _render(root, 1)
    return "\n".join(lines)


def summary_for_log(tree: GoalTree) -> dict:
    """진단용 dict."""
    if tree is None:
        return {}
    by_status: dict[str, int] = {}
    by_horizon: dict[str, int] = {}
    for m in tree.all_milestones():
        by_status[m.status] = by_status.get(m.status, 0) + 1
        by_horizon[m.horizon] = by_horizon.get(m.horizon, 0) + 1
    root = tree.root()
    return {
        "stage_id": tree.stage_id,
        "root_title": root.title if root else "",
        "root_progress": root.progress if root else 0.0,
        "total_milestones": len(tree.milestones),
        "by_status": by_status,
        "by_horizon": by_horizon,
        "horizon_end": tree.horizon_end,
        "updated_at": tree.updated_at,
    }


# ── [Phase 9-E] LLM 자동 decompose ────────────────────────────────────
# 사용자가 "이번 해 매출 3000만" 한 줄 입력 → LLM 이 4 분기·각 분기 안 월 milestone
# 자동 제안. 사용자는 미리보기 → confirm 시 tree 에 일괄 추가 (apply_decomposition).
#
# graceful — LLM 실패 시 빈 list. caller (UI) 가 빈 결과 시 안내 + 수동 추가 안내.

_DECOMPOSE_SYSTEM_PROMPT = """너는 1년 목표 분해 전문가다.

사용자가 큰 목표 (1년/분기/월 단위) 를 주면, 그것을 더 작은 단위 (분기/월/주) 의
하위 milestone N개로 분해한다. 각 milestone 은:
  - title (짧고 측정 가능한 이름, 50자 이내)
  - description (1~2문장, 왜 이게 중요한지)
  - success_criteria (3~5 개의 짧은 성공 기준 문장)
  - target_date (옵션, YYYY-MM-DD)

분해 원칙:
1. **MECE**: 분해된 milestone 들이 중복 없이 합쳐서 부모 목표를 cover.
2. **측정 가능**: success_criteria 는 history 텍스트와 키워드 매칭 가능한 구체어로.
3. **순서**: 시간 순 (Q1 → Q2 → Q3 → Q4) 또는 의존성 순.
4. **현실적**: 너무 많이 쪼개지 X — 사용자가 요청한 N 개 정확히.

순수 JSON 만 출력:
{
  "milestones": [
    {"title": "...", "description": "...",
     "success_criteria": ["...", "...", "..."],
     "target_date": "YYYY-MM-DD"},
    ...
  ]
}
"""


async def decompose_goal_with_llm_async(
    goal_text: str,
    *,
    parent_horizon: str = "year",
    target_horizon: str = "quarter",
    n_children: int = 4,
    horizon_end: str = "",
    context: str = "",
) -> list[Milestone]:
    """LLM 으로 부모 목표를 하위 milestone N개로 분해.

    Args:
        goal_text: 분해할 부모 목표 ("이번 해 매출 3000만")
        parent_horizon: 부모의 horizon (year/quarter/month/week)
        target_horizon: 생성될 children horizon (quarter/month/week/task)
        n_children: 생성 개수 (year→quarter 면 4, quarter→month 면 3 권장)
        horizon_end: 부모 마감일 (YYYY-MM-DD·옵션·target_date 분배에 사용)
        context: 추가 맥락 (Stage.context_summary·도메인 정보 등)

    Returns: Milestone obj list (id 미생성 — apply_decomposition 가 등록 시 할당).
             LLM 실패·파싱 실패 시 빈 list.
    """
    if not goal_text or not goal_text.strip():
        return []
    if target_horizon not in HORIZONS:
        target_horizon = "task"
    n_children = max(1, min(12, int(n_children)))

    try:
        from llm_module import get_llm_response_async, robust_json_parse
    except Exception as e:
        print(f"[goal_tree] decompose llm_module import 실패 (graceful): {e}")
        return []

    user_prompt_parts = [
        f"[부모 목표] {goal_text.strip()}",
        f"[부모 horizon] {parent_horizon}",
        f"[생성할 children horizon] {target_horizon}",
        f"[개수] 정확히 {n_children} 개",
    ]
    if horizon_end:
        user_prompt_parts.append(f"[부모 마감일] {horizon_end} (각 children 의 target_date 는 이 안에서 분배)")
    if context:
        user_prompt_parts.append(f"[추가 맥락]\n{context[:1000]}")
    user_prompt_parts.append(
        f"\n위 부모 목표를 정확히 {n_children}개의 {target_horizon} milestone 으로 분해."
        " 순수 JSON 만 출력."
    )
    user_prompt = "\n".join(user_prompt_parts)

    raw = ""
    try:
        raw = await get_llm_response_async(
            user_prompt,
            response_mime_type="application/json",
            system_prompt=_DECOMPOSE_SYSTEM_PROMPT,
            max_tokens=2048,
            timeout=60,
        )
    except Exception as e:
        print(f"[goal_tree] decompose LLM 호출 실패 (graceful): {e}")
        return []

    if not raw or not raw.strip():
        return []

    try:
        data = robust_json_parse(raw)
    except Exception as e:
        print(f"[goal_tree] decompose JSON 파싱 실패 (graceful): {e}")
        return []

    items = data.get("milestones") if isinstance(data, dict) else None
    if not isinstance(items, list) or not items:
        return []

    out: list[Milestone] = []
    for item in items[:n_children]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        m = Milestone(
            id="",   # apply_decomposition 가 add_milestone 으로 등록 시 할당
            title=title[:200],
            description=str(item.get("description") or "")[:500],
            horizon=target_horizon,
            target_date=str(item.get("target_date") or ""),
            success_criteria=[
                str(c)[:120] for c in (item.get("success_criteria") or [])
                if c
            ][:5],
            source="llm_decomposed",
        )
        out.append(m)
    return out


def apply_decomposition(
    tree: GoalTree,
    parent_id: str,
    decomposed: list[Milestone],
) -> list[Milestone]:
    """decompose_goal_with_llm_async 결과를 tree 에 일괄 등록.

    각 Milestone obj 는 add_milestone 으로 정식 등록 (id 할당·parent 자동 연결).

    Returns: 등록된 Milestone list (id 채워짐). 빈 입력 시 빈 list.
    """
    if tree is None or not parent_id or not decomposed:
        return []
    parent = tree.get(parent_id)
    if parent is None:
        return []
    # children horizon 기본 — 부모의 권장 children horizon
    default_horizon = _HORIZON_CHILDREN.get(parent.horizon, "task")
    out: list[Milestone] = []
    for m_proto in decomposed:
        try:
            hz = m_proto.horizon if m_proto.horizon in HORIZONS else default_horizon
            registered = add_milestone(
                tree,
                title=m_proto.title,
                parent_id=parent_id,
                horizon=hz,
                target_date=m_proto.target_date or "",
                description=m_proto.description or "",
                success_criteria=list(m_proto.success_criteria or []),
                source="llm_decomposed",
            )
            if registered is not None:
                out.append(registered)
        except Exception as e:
            print(f"[goal_tree] apply 1건 실패 skip (graceful): {e}")
            continue
    return out


__all__ = [
    "Milestone", "GoalTree",
    "HORIZONS", "MILESTONE_STATUS",
    "new_goal_tree", "load_goal_tree", "save_goal_tree", "delete_goal_tree",
    "add_milestone", "update_milestone",
    "set_milestone_status", "set_milestone_progress",
    "reparent_milestone", "delete_milestone",
    "recompute_progress", "evaluate_progress_from_history",
    "sync_root_with_stage",
    "get_active_milestones",
    "as_prompt_brief", "summary_for_log",
    "decompose_goal_with_llm_async", "apply_decomposition",
]
