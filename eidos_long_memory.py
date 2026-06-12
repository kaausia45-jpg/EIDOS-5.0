# eidos_long_memory.py
# [2026-05-27 Phase 10] Long-horizon Memory — pending_thread 와 milestone 사이 중간층.
#
# 3-layer 위계 (Phase 5/7 + Phase 9 + Phase 10):
#   [단기] pending_thread    며칠~1주 — Phase 5/7 의 belief.pending_threads
#         ↓ compact_threads_to_episode (resolved/abandoned + 14일+ 묵음)
#   [중기] MemoryEpisode     며칠~몇 주 — 본 모듈 신규
#         ↓ belongs_to (milestone_id cross-ref)
#   [장기] Milestone         몇 주~몇 달 — Phase 9 eidos_goal_tree.Milestone
#         ↓
#   [목표] GoalTree.root     1년
#
# 디자인 원칙 (Phase 1~9 동일):
#   1. LLM 호출 0 (Phase 10-A). 휴리스틱 단순 변환 (1 thread → 1 episode 의 1:1).
#      LLM 기반 정교한 cluster·summary 는 후속 (Phase 10-B 옵션).
#   2. graceful — 손상되면 빈 memory 반환·throw X.
#   3. _atomic_write — belief 와 같은 폴더.
#   4. 모든 임계값 모듈 상단 상수.
#   5. backwards compat — deserialize 가 빈 필드 fallback.
#
# 저장: eidos_files/agents/user_belief/{user_id}_episodes.json
#       (belief 와 같은 폴더, _episodes.json suffix 로 구분)
#
# trim_aged_threads (Phase 7-B, 7일+ 무조건 abandon) 가 호출되기 *직전* 에
# compact_threads_to_episode 가 호출돼야 함 — 그래야 thread 가 abandoned 로 변하기
# 전에 의미 있는 episode 로 보존됨. (proactive_scheduler.evaluate() 첫 줄에서 hook)

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from eidos_belief_core import UserBelief


# ── 상수 ───────────────────────────────────────────────────────────────
_BASE_DIR = os.path.join("eidos_files", "agents", "user_belief")
_VERSION = 1
_DEFAULT_USER_ID = "default"

# compact_threads_to_episode 기본값
_COMPACT_AGE_DAYS_DEFAULT = 14.0   # 14일+ 묵은 resolved/abandoned thread 만 episode 화
_COMPACT_MIN_TOPIC_LEN = 2         # topic 이 너무 짧으면 skip
_DAY = 86400.0

# search_episodes 기본
_SEARCH_MIN_TOKEN_LEN = 2
_SEARCH_DEFAULT_TOP_K = 5

# prompt brief
_DEFAULT_BRIEF_TOP_K = 3

# episode 의 outcome
EPISODE_OUTCOMES = ["resolved", "abandoned", "mixed", "unresolved"]
EPISODE_SOURCES = ["auto_compact", "manual", "llm"]


# ── MemoryEpisode ──────────────────────────────────────────────────────
@dataclass
class MemoryEpisode:
    """며칠~몇 주의 의미 묶음 — resolved/abandoned 된 단기 thread 의 압축 기록.

    핵심 의미:
      - title: 짧은 이름 (검색용·LRU 표시용)
      - summary: 200자 — 휴리스틱은 thread.description 그대로·LLM 은 정교화
      - related_thread_ids: 묶인 원래 thread id (belief.pending_threads.id 참조)
      - milestone_id: Phase 9 GoalTree.Milestone cross-ref (auto-link 옵션)
      - lessons: 향후 LLM 으로·휴리스틱은 빈 list
      - outcome: 묶인 thread 들의 종합 — 모두 resolved 면 "resolved"·모두 abandoned 면
                 "abandoned"·섞이면 "mixed"·아직 진행 중이면 "unresolved"
    """

    id: str
    title: str
    summary: str = ""
    user_id: str = _DEFAULT_USER_ID
    milestone_id: str = ""              # Phase 9 cross-ref (옵션)
    stage_id: str = ""                  # 어느 stage 의 활동 (옵션)
    related_thread_ids: list[str] = field(default_factory=list)
    started_at: str = ""
    ended_at: str = ""
    outcome: str = "unresolved"         # EPISODE_OUTCOMES
    lessons: list[str] = field(default_factory=list)
    importance: float = 0.5
    source: str = "auto_compact"        # EPISODE_SOURCES

    def serialize(self) -> dict:
        return asdict(self)

    @classmethod
    def deserialize(cls, data: dict) -> "MemoryEpisode":
        if not isinstance(data, dict):
            data = {}
        now = _now()
        oc = str(data.get("outcome") or "unresolved")
        if oc not in EPISODE_OUTCOMES:
            oc = "unresolved"
        src = str(data.get("source") or "auto_compact")
        if src not in EPISODE_SOURCES:
            src = "auto_compact"
        try:
            imp = float(data.get("importance", 0.5))
        except Exception:
            imp = 0.5
        return cls(
            id=str(data.get("id") or _new_id("ep")),
            title=str(data.get("title") or "(이름 없음)"),
            summary=str(data.get("summary") or ""),
            user_id=str(data.get("user_id") or _DEFAULT_USER_ID),
            milestone_id=str(data.get("milestone_id") or ""),
            stage_id=str(data.get("stage_id") or ""),
            related_thread_ids=list(data.get("related_thread_ids") or []),
            started_at=str(data.get("started_at") or now),
            ended_at=str(data.get("ended_at") or ""),
            outcome=oc,
            lessons=list(data.get("lessons") or []),
            importance=max(0.0, min(1.0, imp)),
            source=src,
        )


# ── LongTermMemory ─────────────────────────────────────────────────────
@dataclass
class LongTermMemory:
    """1 user 의 episode 모음. belief 와 같은 user_id 단위."""

    user_id: str = _DEFAULT_USER_ID
    episodes: dict[str, dict] = field(default_factory=dict)   # id → MemoryEpisode raw
    last_compacted_at: str = ""
    version: int = _VERSION
    created_at: str = ""
    updated_at: str = ""

    def serialize(self) -> dict:
        return asdict(self)

    @classmethod
    def deserialize(cls, data: dict) -> "LongTermMemory":
        if not isinstance(data, dict):
            data = {}
        now = _now()
        return cls(
            user_id=str(data.get("user_id") or _DEFAULT_USER_ID),
            episodes=dict(data.get("episodes") or {}),
            last_compacted_at=str(data.get("last_compacted_at") or ""),
            version=int(data.get("version") or _VERSION),
            created_at=str(data.get("created_at") or now),
            updated_at=str(data.get("updated_at") or now),
        )

    def all_episodes(self) -> list[MemoryEpisode]:
        return [MemoryEpisode.deserialize(r) for r in self.episodes.values()]

    def get(self, episode_id: str) -> Optional[MemoryEpisode]:
        raw = self.episodes.get(episode_id)
        if not raw:
            return None
        return MemoryEpisode.deserialize(raw)


# ── 헬퍼 ───────────────────────────────────────────────────────────────
def _now() -> str:
    return _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _parse_iso(ts: str) -> Optional[_dt.datetime]:
    if not ts:
        return None
    try:
        s = ts.replace("Z", "+00:00")
        return _dt.datetime.fromisoformat(s).replace(tzinfo=None)
    except Exception:
        return None


def _new_id(prefix: str = "ep") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _ensure_base() -> None:
    try:
        os.makedirs(_BASE_DIR, exist_ok=True)
    except Exception as e:
        print(f"[long_memory] _ensure_base 실패 (graceful): {e}")


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
        print(f"[long_memory] _atomic_write 실패 (graceful): {path} — {e}")
        return False


def _memory_path(user_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_\-가-힣]+", "_", user_id) or _DEFAULT_USER_ID
    return os.path.join(_BASE_DIR, f"{safe}_episodes.json")


def _put(memory: LongTermMemory, ep: MemoryEpisode) -> None:
    memory.episodes[ep.id] = ep.serialize()
    memory.updated_at = _now()


# ── CRUD ───────────────────────────────────────────────────────────────
def new_long_memory(user_id: str = _DEFAULT_USER_ID) -> LongTermMemory:
    now = _now()
    return LongTermMemory(
        user_id=user_id or _DEFAULT_USER_ID,
        created_at=now,
        updated_at=now,
    )


def load_long_memory(user_id: str = _DEFAULT_USER_ID) -> LongTermMemory:
    """없거나 손상되면 새 빈 memory. belief 와 같은 패턴 (None 반환 X)."""
    uid = user_id or _DEFAULT_USER_ID
    path = _memory_path(uid)
    if not os.path.exists(path):
        return new_long_memory(uid)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return LongTermMemory.deserialize(data)
    except Exception as e:
        print(f"[long_memory] load 손상 (graceful·새 memory 반환): {uid} — {e}")
        return new_long_memory(uid)


def save_long_memory(memory: LongTermMemory) -> bool:
    if memory is None or not memory.user_id:
        return False
    _ensure_base()
    memory.updated_at = _now()
    memory.version = _VERSION
    path = _memory_path(memory.user_id)
    try:
        payload = memory.serialize()
        return _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"[long_memory] save 실패 (graceful): {e}")
        return False


def delete_long_memory(user_id: str = _DEFAULT_USER_ID) -> bool:
    path = _memory_path(user_id or _DEFAULT_USER_ID)
    try:
        if os.path.exists(path):
            os.remove(path)
        return True
    except Exception as e:
        print(f"[long_memory] delete 실패 (graceful): {e}")
        return False


# ── episode 조작 ──────────────────────────────────────────────────────
def add_episode(
    memory: LongTermMemory,
    title: str,
    summary: str = "",
    milestone_id: str = "",
    stage_id: str = "",
    related_thread_ids: Optional[list[str]] = None,
    outcome: str = "resolved",
    importance: float = 0.5,
    source: str = "manual",
    started_at: str = "",
    ended_at: str = "",
    lessons: Optional[list[str]] = None,
) -> Optional[MemoryEpisode]:
    """수동 episode 등록 (또는 compact 가 내부에서 호출)."""
    if memory is None or not title.strip():
        return None
    now = _now()
    if outcome not in EPISODE_OUTCOMES:
        outcome = "resolved"
    if source not in EPISODE_SOURCES:
        source = "manual"
    ep = MemoryEpisode(
        id=_new_id("ep"),
        title=title.strip()[:120],
        summary=summary[:600] if summary else "",
        user_id=memory.user_id,
        milestone_id=milestone_id or "",
        stage_id=stage_id or "",
        related_thread_ids=list(related_thread_ids or []),
        started_at=started_at or now,
        ended_at=ended_at,
        outcome=outcome,
        lessons=list(lessons or []),
        importance=max(0.0, min(1.0, float(importance))),
        source=source,
    )
    _put(memory, ep)
    return ep


def update_episode(
    memory: LongTermMemory,
    episode_id: str,
    *,
    title: Optional[str] = None,
    summary: Optional[str] = None,
    milestone_id: Optional[str] = None,
    lessons: Optional[list[str]] = None,
    outcome: Optional[str] = None,
    importance: Optional[float] = None,
) -> bool:
    ep = memory.get(episode_id) if memory else None
    if ep is None:
        return False
    if title is not None:
        ep.title = title.strip()[:120]
    if summary is not None:
        ep.summary = summary[:600]
    if milestone_id is not None:
        ep.milestone_id = milestone_id
    if lessons is not None:
        ep.lessons = list(lessons)
    if outcome is not None and outcome in EPISODE_OUTCOMES:
        ep.outcome = outcome
    if importance is not None:
        ep.importance = max(0.0, min(1.0, float(importance)))
    _put(memory, ep)
    return True


def delete_episode(memory: LongTermMemory, episode_id: str) -> bool:
    if memory is None or not episode_id:
        return False
    if episode_id in memory.episodes:
        memory.episodes.pop(episode_id, None)
        memory.updated_at = _now()
        return True
    return False


# ── 자동 묶기 (compact_threads_to_episode) ─────────────────────────────
def compact_threads_to_episode(
    belief: "UserBelief",
    memory: LongTermMemory,
    *,
    threshold_days: float = _COMPACT_AGE_DAYS_DEFAULT,
    stage_id: str = "",
    active_milestones: Optional[list] = None,    # list[Milestone] — auto-link 후보
    now: Optional[_dt.datetime] = None,
) -> list[MemoryEpisode]:
    """belief.pending_threads 중 resolved/abandoned + threshold 묵은 것을 episode 화.

    Phase 10-A 단순 휴리스틱:
      - 1 thread → 1 episode (그룹화 X·LLM 기반 cluster 는 후속)
      - 이미 episode 의 related_thread_ids 에 등록된 thread 는 skip (중복 회피)
      - status=resolved → episode.outcome=resolved
      - status=abandoned → outcome=abandoned
      - milestone 매칭: active_milestones 의 title/success_criteria 와 thread.topic
        substring 매칭 시 첫 milestone 의 id 자동 link (없으면 빈)
      - title 은 thread.topic 그대로·summary 는 thread.description (없으면 topic 반복)
      - importance 는 thread.importance 그대로

    trim_aged_threads (Phase 7-B) 호출 *직전* 에 사용. 그래야 7일+ open thread 가
    abandoned 로 변하기 전 마지막 14일+ resolved 만 보존 (open 은 안 건드림 — 살아있음).

    Args:
        belief: UserBelief 인스턴스 (pending_threads 검사)
        memory: LongTermMemory — 새 episode 가 여기 추가됨
        threshold_days: 이 일수+ 묵은 thread 만 (started_at 기준)
        stage_id: episode.stage_id 로 기록 (선택)
        active_milestones: milestone auto-link 후보 list (각 객체 .id .title
            .success_criteria 가져야 함 — eidos_goal_tree.Milestone)
        now: 테스트용 fixed time

    Returns: 새로 만들어진 episode list.
    """
    if belief is None or memory is None:
        return []
    n = now or _dt.datetime.utcnow()
    threshold_secs = threshold_days * _DAY

    # 이미 episode 로 변환된 thread id 모음
    converted: set[str] = set()
    for raw in memory.episodes.values():
        for tid in raw.get("related_thread_ids", []) or []:
            converted.add(str(tid))

    new_episodes: list[MemoryEpisode] = []
    for raw in list(belief.pending_threads):
        try:
            tid = str(raw.get("id") or "")
            if not tid or tid in converted:
                continue
            status = str(raw.get("status") or "")
            if status not in ("resolved", "abandoned"):
                continue
            topic = str(raw.get("topic") or "").strip()
            if len(topic) < _COMPACT_MIN_TOPIC_LEN:
                continue
            started = _parse_iso(str(raw.get("started_at") or ""))
            if started is None:
                continue
            if (n - started).total_seconds() < threshold_secs:
                continue   # 아직 신선 — 단기 thread 로 두기

            description = str(raw.get("description") or "")
            try:
                imp = float(raw.get("importance", 0.5))
            except Exception:
                imp = 0.5
            outcome = "resolved" if status == "resolved" else "abandoned"
            ended = str(raw.get("last_referenced_at") or "") or _now()

            # milestone auto-link 휴리스틱
            milestone_id = ""
            if active_milestones:
                topic_low = topic.lower()
                for m in active_milestones:
                    try:
                        m_title = str(getattr(m, "title", "") or "").lower()
                        m_crits = " ".join(
                            str(c) for c in (getattr(m, "success_criteria", []) or [])
                        ).lower()
                        hay = m_title + " " + m_crits
                        # 양방향 substring 매칭 (topic↔milestone)
                        if topic_low in hay or any(
                            tok in hay for tok in _tokens(topic_low)
                        ):
                            milestone_id = str(getattr(m, "id", "") or "")
                            if milestone_id:
                                break
                    except Exception:
                        continue

            ep = add_episode(
                memory,
                title=topic,
                summary=(description or topic)[:600],
                milestone_id=milestone_id,
                stage_id=stage_id,
                related_thread_ids=[tid],
                outcome=outcome,
                importance=imp,
                source="auto_compact",
                started_at=started.isoformat(timespec="seconds") + "Z",
                ended_at=ended,
            )
            if ep is not None:
                new_episodes.append(ep)
                converted.add(tid)
        except Exception as e:
            print(f"[long_memory] compact 1건 실패 skip (graceful): {e}")
            continue

    if new_episodes:
        memory.last_compacted_at = n.isoformat(timespec="seconds") + "Z"
    return new_episodes


# ── 검색 ──────────────────────────────────────────────────────────────
_TOKEN_RE = re.compile(r"[A-Za-z가-힣][A-Za-z가-힣0-9_\-]+")

_STOP_WORDS = {
    "그거", "이거", "저거", "그것", "이것", "저것", "그게", "이게",
    "있는", "있어", "하고", "해서", "그리고", "그러나", "근데",
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
}


def _tokens(text: str) -> list[str]:
    if not text:
        return []
    out = []
    for tok in _TOKEN_RE.findall(text):
        tlow = tok.lower()
        if len(tlow) < _SEARCH_MIN_TOKEN_LEN:
            continue
        if tlow in _STOP_WORDS:
            continue
        out.append(tlow)
    return out


def search_episodes(
    memory: LongTermMemory,
    query: str,
    *,
    top_k: int = _SEARCH_DEFAULT_TOP_K,
    include_abandoned: bool = True,
) -> list[tuple[MemoryEpisode, float]]:
    """query 단어와 episode (title+summary+lessons) 의 매칭 점수로 top_k.

    점수 = 매칭 token 수 + importance × 0.3.
    매칭 0 인 episode 는 제외.

    Returns: [(episode, score), ...] — score 내림차순.
    """
    if memory is None or not query or not query.strip():
        return []
    q_toks = set(_tokens(query))
    if not q_toks:
        return []
    scored: list[tuple[MemoryEpisode, float]] = []
    for ep in memory.all_episodes():
        if not include_abandoned and ep.outcome == "abandoned":
            continue
        hay = ep.title + " " + ep.summary + " " + " ".join(ep.lessons)
        hay_toks = set(_tokens(hay))
        if not hay_toks:
            continue
        n_match = len(q_toks & hay_toks)
        if n_match <= 0:
            continue
        score = float(n_match) + ep.importance * 0.3
        scored.append((ep, round(score, 3)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def episodes_by_milestone(
    memory: LongTermMemory,
    milestone_id: str,
) -> list[MemoryEpisode]:
    """특정 milestone 에 link 된 episode 만. importance·시간 내림차순."""
    if memory is None or not milestone_id:
        return []
    out = [ep for ep in memory.all_episodes() if ep.milestone_id == milestone_id]
    out.sort(key=lambda e: (e.importance, e.ended_at or e.started_at), reverse=True)
    return out


# ── prompt brief ───────────────────────────────────────────────────────
def as_prompt_brief(
    memory: LongTermMemory,
    *,
    related_milestone_ids: Optional[list[str]] = None,
    top_k: int = _DEFAULT_BRIEF_TOP_K,
) -> str:
    """LLM prompt 에 inject 할 markdown — 최근/관련 episode 요약.

    related_milestone_ids 주어지면 그 milestone 에 link 된 episode 우선.
    없으면 importance 순 top_k.

    빈 memory 면 빈 string (prompt 변화 X).
    """
    if memory is None or not memory.episodes:
        return ""
    all_eps = memory.all_episodes()
    if not all_eps:
        return ""

    selected: list[MemoryEpisode] = []
    if related_milestone_ids:
        related = set(related_milestone_ids)
        for ep in all_eps:
            if ep.milestone_id and ep.milestone_id in related:
                selected.append(ep)
        # importance 순 자르기
        selected.sort(key=lambda e: e.importance, reverse=True)
        selected = selected[:top_k]

    # 부족하면 일반 top_k 로 채움
    if len(selected) < top_k:
        existing = {e.id for e in selected}
        rest = [e for e in all_eps if e.id not in existing]
        rest.sort(key=lambda e: e.importance, reverse=True)
        selected.extend(rest[: top_k - len(selected)])

    if not selected:
        return ""

    lines = ["## 장기 메모리 (최근/관련 episode)"]
    for ep in selected:
        icon = "✅" if ep.outcome == "resolved" else (
            "🚫" if ep.outcome == "abandoned" else "·")
        ended = f" (~{ep.ended_at[:10]})" if ep.ended_at else ""
        lines.append(
            f"{icon} **{ep.title}**  importance={ep.importance:.2f}{ended}"
        )
        if ep.summary:
            lines.append(f"   _{ep.summary[:120]}_")
        if ep.lessons:
            lines.append(f"   교훈: {' / '.join(ep.lessons[:3])}")
    return "\n".join(lines)


def summary_for_log(memory: LongTermMemory) -> dict:
    if memory is None:
        return {}
    by_outcome: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for ep in memory.all_episodes():
        by_outcome[ep.outcome] = by_outcome.get(ep.outcome, 0) + 1
        by_source[ep.source] = by_source.get(ep.source, 0) + 1
    return {
        "user_id": memory.user_id,
        "total_episodes": len(memory.episodes),
        "by_outcome": by_outcome,
        "by_source": by_source,
        "last_compacted_at": memory.last_compacted_at,
        "updated_at": memory.updated_at,
    }


__all__ = [
    "MemoryEpisode", "LongTermMemory",
    "EPISODE_OUTCOMES", "EPISODE_SOURCES",
    "new_long_memory", "load_long_memory", "save_long_memory", "delete_long_memory",
    "add_episode", "update_episode", "delete_episode",
    "compact_threads_to_episode",
    "search_episodes", "episodes_by_milestone",
    "as_prompt_brief", "summary_for_log",
]
