# eidos_hierarchical_planner.py
# [Wave5-B 2026-05-28] 계층적 계획 — GoalTree 의 year→quarter→month→week→task 를
# 정적이 아닌 **동적 focus** 로. 매 tick LLM 이 "지금 어느 level 에 집중할까" 결정.
#
# 학술 배경:
#   - Hierarchical Reinforcement Learning (Sutton & Precup options framework)
#   - Sutton's "tasks as policies over policies"
#   - 인간 인지의 multi-scale planning (Botvinick et al.)
#
# EIDOS 적용:
#   - GoalTree (Phase 9) 는 정적 트리 — milestone 모두 동등하게 active
#   - hierarchical_planner 는 그 위에 "현재 focus" 동적 결정
#   - focus level 에 따라:
#       * mental_rollout depth 조정 (year=1·task=3)
#       * action_scope_hint 를 LLM prompt 에 inject (전략/실행)
#       * progress propagation 자동 (leaf 변화 → root)
#
# 비용: tick 당 LLM 1콜 (~$0.0001).

from __future__ import annotations

import asyncio
import datetime as _dt
import json
from dataclasses import dataclass, asdict
from typing import Optional


# 각 level 에 대응되는 rollout depth 권장값
_LEVEL_TO_ROLLOUT_DEPTH = {
    "year":    1,
    "quarter": 2,
    "month":   2,
    "week":    2,
    "task":    3,
}

# 각 level 에 대응되는 action scope hint
_LEVEL_TO_SCOPE_HINT = {
    "year":    "전략적·큰 그림 점검·핵심 KPI 검토·자료/시장 조사",
    "quarter": "중기 진행 점검·major milestone 마감 임박 여부·자원 재할당",
    "month":   "주별 진척 정리·바로 다음 주 계획·이번 달 deliverable 확보",
    "week":    "이번 주 task list·우선순위 정렬·blocker 해결",
    "task":    "지금 바로 할 1개 action 실행·구체적 산출물·즉시 측정 가능",
}


@dataclass
class HierarchicalFocus:
    """이 tick 의 계층 focus."""
    focus_level: str = "task"           # year/quarter/month/week/task
    focus_milestone_id: str = ""
    focus_milestone_title: str = ""
    reasoning: str = ""
    action_scope_hint: str = ""
    recommended_rollout_depth: int = 2
    decided_at: str = ""

    def serialize(self) -> dict:
        return asdict(self)


def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


# ── LLM 시스템 프롬프트 ─────────────────────────────────────────────────
_FOCUS_SYSTEM = """\
너는 EIDOS 의 hierarchical planning module 이다.

GoalTree 의 active milestone 들 (year/quarter/month/week/task) 과 현재 상황을 보고,
**지금 어느 level 에 집중할지** 결정해라.

원칙:
- year/quarter level focus → 전략 점검·자료 조사·메타 분석 (자주 X·가끔만)
- task level focus → 즉시 실행 (대부분의 tick)
- month/week level focus → 정리·계획 (사이 빈도)
- 사용자가 working 모드 + active 마일스톤 있으면 task level
- 사용자 break 모드 + 큰 전환점 있으면 year/quarter level (반성)
- focus_milestone_id 는 그 level 의 active milestone 중 가장 시급한 것
- 시급도 = stale 일수·deadline 임박·progress 낮음 종합

JSON 만 출력:
{
  "focus_level": "year|quarter|month|week|task",
  "focus_milestone_id": "<id 또는 빈 string>",
  "reasoning": "왜 이 level 선택했는지 짧은 한 줄"
}
"""


# ── 핵심: focus 결정 ──────────────────────────────────────────────────
async def pick_hierarchical_focus_async(
    goal_tree,
    work_state: str = "unknown",
    minutes_since_user: float = 0.0,
    hour_of_day: Optional[int] = None,
    recent_topics: Optional[list] = None,
    emotion_label: Optional[str] = None,
    timeout_sec: float = 8.0,
) -> HierarchicalFocus:
    """현재 컨텍스트 + GoalTree 보고 focus level 결정.

    goal_tree 가 None 이거나 비어있으면 task level + 빈 focus 반환.
    LLM 실패 시 휴리스틱 fallback (work_state 기반).
    """
    focus = HierarchicalFocus(
        decided_at=_now_iso(),
        recommended_rollout_depth=2,
    )

    # GoalTree 없으면 task fallback
    if goal_tree is None or not getattr(goal_tree, "root_goal_id", ""):
        focus.focus_level = "task"
        focus.reasoning = "GoalTree 없음·task level fallback"
        focus.action_scope_hint = _LEVEL_TO_SCOPE_HINT["task"]
        focus.recommended_rollout_depth = _LEVEL_TO_ROLLOUT_DEPTH["task"]
        return focus

    # GoalTree 요약 — level 별 active milestone 추출
    try:
        from eidos_goal_tree import get_active_milestones
        active = get_active_milestones(goal_tree) or []
    except Exception:
        active = []

    if not active:
        focus.focus_level = "task"
        focus.reasoning = "active milestone 없음·task fallback"
        focus.action_scope_hint = _LEVEL_TO_SCOPE_HINT["task"]
        focus.recommended_rollout_depth = _LEVEL_TO_ROLLOUT_DEPTH["task"]
        return focus

    # level 별 grouping
    by_level: dict = {}
    for m in active:
        lvl = getattr(m, "horizon", "task") or "task"
        if lvl not in by_level:
            by_level[lvl] = []
        try:
            by_level[lvl].append({
                "id": getattr(m, "id", ""),
                "title": (getattr(m, "title", "") or "")[:60],
                "progress": float(getattr(m, "progress", 0.0)),
                "target_date": getattr(m, "target_date", ""),
            })
        except Exception:
            continue

    # LLM prompt 구성
    levels_block_lines = []
    for lvl in ("year", "quarter", "month", "week", "task"):
        if lvl not in by_level:
            continue
        items = by_level[lvl]
        levels_block_lines.append(f"  [{lvl}] {len(items)}개 active:")
        for it in items[:4]:
            levels_block_lines.append(
                f"    - {it['title']} (progress={it['progress']:.2f}"
                + (f"·target={it['target_date']}" if it["target_date"] else "")
                + f"·id={it['id'][:10]})"
            )
    levels_block = "\n".join(levels_block_lines) if levels_block_lines else "(active 없음)"

    nv = hour_of_day if hour_of_day is not None else _dt.datetime.now().hour
    ctx_lines = [
        f"[work_state] {work_state}",
        f"[시간대] {nv:02d}시",
        f"[사용자 마지막 메시지 후] {minutes_since_user:.1f}분",
    ]
    if emotion_label:
        ctx_lines.append(f"[현재 감정] {emotion_label}")
    if recent_topics:
        ctx_lines.append(
            f"[최근 화제] {', '.join(str(t)[:30] for t in recent_topics[:3])}"
        )
    ctx_block = "\n".join(ctx_lines)

    prompt = (
        f"{ctx_block}\n\n"
        f"[GoalTree active milestones (level 별)]\n{levels_block}\n\n"
        "지금 어느 level 에 집중할지 + focus milestone 결정. JSON 만 출력."
    )

    try:
        from llm_module import get_llm_response_async
        raw = await asyncio.wait_for(
            get_llm_response_async(
                prompt,
                max_tokens=1024,
                system_prompt=_FOCUS_SYSTEM,
                response_mime_type="application/json",
            ),
            timeout=timeout_sec,
        )
        data = json.loads(raw)
        if isinstance(data, dict):
            focus.focus_level = str(data.get("focus_level", "task")).lower()
            if focus.focus_level not in _LEVEL_TO_ROLLOUT_DEPTH:
                focus.focus_level = "task"
            focus.focus_milestone_id = str(data.get("focus_milestone_id", ""))[:40]
            focus.reasoning = str(data.get("reasoning", ""))[:200]
    except asyncio.TimeoutError:
        focus.reasoning = "LLM timeout — 휴리스틱 fallback"
    except Exception as e:
        focus.reasoning = f"LLM 실패 — fallback ({str(e)[:60]})"

    # 휴리스틱 fallback — work_state 기반
    if not focus.focus_level or focus.focus_level not in _LEVEL_TO_ROLLOUT_DEPTH:
        focus.focus_level = "task" if work_state == "working" else "week"

    # focus_milestone_title 채우기
    if focus.focus_milestone_id:
        for m in active:
            if getattr(m, "id", "") == focus.focus_milestone_id:
                focus.focus_milestone_title = (
                    getattr(m, "title", "") or ""
                )[:80]
                break

    # focus_milestone_id 없으면 그 level 의 첫 active 자동 선택
    if not focus.focus_milestone_id:
        candidates = by_level.get(focus.focus_level) or []
        if candidates:
            focus.focus_milestone_id = candidates[0]["id"]
            focus.focus_milestone_title = candidates[0]["title"]

    focus.action_scope_hint = _LEVEL_TO_SCOPE_HINT.get(focus.focus_level, "")
    focus.recommended_rollout_depth = _LEVEL_TO_ROLLOUT_DEPTH.get(
        focus.focus_level, 2,
    )
    return focus


# ── progress propagation ────────────────────────────────────────────
def propagate_progress_up(goal_tree, leaf_milestone_id: str) -> int:
    """leaf milestone 의 progress 변화 → parent 의 children 평균으로 재계산.

    progress_is_manual=True 인 internal node 는 skip (override 보존).
    재귀로 root 까지 전파. 변경된 node 수 반환.

    GoalTree 모듈에 이미 비슷한 게 있을 수도·중복 호환·idempotent.
    """
    if goal_tree is None:
        return 0
    try:
        milestones = getattr(goal_tree, "milestones", {}) or {}
    except Exception:
        return 0

    if not isinstance(milestones, dict):
        return 0

    n_changed = 0
    current_id = leaf_milestone_id
    seen = set()
    while current_id and current_id not in seen:
        seen.add(current_id)
        node = milestones.get(current_id)
        if node is None:
            break
        parent_id = getattr(node, "parent_id", "") or ""
        if not parent_id:
            break
        parent = milestones.get(parent_id)
        if parent is None:
            break
        # parent 의 children progress 평균 — manual override 면 skip
        if getattr(parent, "progress_is_manual", False):
            current_id = parent_id
            continue
        child_ids = getattr(parent, "children_ids", []) or []
        if not child_ids:
            current_id = parent_id
            continue
        progresses = []
        for cid in child_ids:
            c = milestones.get(cid)
            if c is not None:
                try:
                    progresses.append(float(getattr(c, "progress", 0.0)))
                except Exception:
                    continue
        if progresses:
            new_progress = sum(progresses) / len(progresses)
            old = float(getattr(parent, "progress", 0.0))
            if abs(new_progress - old) > 0.001:
                try:
                    parent.progress = max(0.0, min(1.0, new_progress))
                    n_changed += 1
                except Exception:
                    pass
        current_id = parent_id
    return n_changed


def focus_brief_for_prompt(focus: HierarchicalFocus) -> str:
    """LLM prompt 에 inject 할 짧은 focus brief."""
    if not focus or not focus.focus_level:
        return ""
    parts = [f"[HIERARCHY focus] level={focus.focus_level}"]
    if focus.focus_milestone_title:
        parts.append(f"milestone={focus.focus_milestone_title}")
    if focus.action_scope_hint:
        parts.append(f"scope: {focus.action_scope_hint}")
    if focus.reasoning:
        parts.append(f"reason: {focus.reasoning[:80]}")
    return " | ".join(parts)
