# eidos_next_action_suggest.py
# [Wave13 2026-05-28] 사용자 모호 질문 → next-action 제안.
#
# 사용자가 "지금 뭐할까"·"오늘 뭐 하지"·"다음 뭐" 같이 모호하게 던지면 EIDOS 가
# **구체 액션 3개 추천** + 우선순위 결정. "알아서 해줘" 류면 즉시 1순위 자동 실행.
#
# 진짜 자율의 마지막 piece — 사용자가 모호해도 EIDOS 가 구체화·실행.
#
# 컨텍스트 종합:
#   - 활성 milestone (GoalTree)
#   - 활성 jobs (Wave 11)
#   - pending chain (autonomous_runs.json·이전 세션 잔존)
#   - multi-actor belief (Wave 9·client/competitor)
#   - sensory signals (Wave 8·work_state)
#   - learning brief (Wave 10·top causal)

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional


# ── 사전 필터 키워드 ─────────────────────────────────────────────────
# next-action 의도 — status query 와 구분.
_NEXT_ACTION_KEYWORDS = (
    # [2026-06-05] "추천/권장/제안/할만한" 같은 넓은 단어 제거 — 사용자(1.png):
    # "이걸로 어떤 업무 자동화해볼지 추천좀 해줘" 같은 평범한 콘텐츠 질문이
    # next-action 추천기로 가로채여 "추천 생성 중..." → "milestone 등록하세요"로
    # 끝나버림. 이 기능은 "지금/다음에 뭐 하지" 같은 자기-지향 질문 전용이므로
    # 그쪽 명백한 패턴만 남기고, "추천" 류는 전부 일반대화로 흘려보낸다.
    "지금 뭐", "지금뭐", "오늘 뭐", "오늘뭐",
    "뭐 하지", "뭐하지", "뭐부터", "뭐 부터",
    "다음 뭐", "다음뭐", "다음 작업", "다음 액션",
    "뭐 할까", "뭐할까", "뭐 할래", "뭐할래",
)

# auto-pick 키워드 — "그냥 알아서" 류
_AUTO_PICK_KEYWORDS = (
    "알아서", "네가 정해", "네가정해",
    "그냥 해", "그냥해", "그냥 진행",
    "맡길게", "맡긴다",
    "1순위", "가장 시급", "최우선",
)

# pick 키워드 (사용자가 추천 중 하나 선택)
_PICK_PATTERNS = (
    "1번", "1순위", "첫번째", "첫 번째",
    "2번", "2순위", "두번째", "두 번째",
    "3번", "3순위", "세번째", "세 번째",
)

_PICK_CONFIRM = (
    "가자", "진행", "ok", "OK", "해줘", "그거", "예", "yes",
)

_PICK_REJECT = (
    "취소", "아니", "no", "NO", "다음에", "별로",
)


def has_next_action_keyword(text: str) -> bool:
    if not text or not text.strip():
        return False
    t = text.lower() if not any(ord(c) > 127 for c in text) else text
    return any(kw in t for kw in _NEXT_ACTION_KEYWORDS)


def has_auto_pick_keyword(text: str) -> bool:
    if not text:
        return False
    t = text.lower() if not any(ord(c) > 127 for c in text) else text
    return any(kw in t for kw in _AUTO_PICK_KEYWORDS)


def parse_pick_index(text: str) -> Optional[int]:
    """사용자 답에서 pick index (0~2) 추출. 없으면 None.

    "1번"·"1"·"첫번째" → 0
    "2번"·"2"·"두번째" → 1
    "3번"·"3"·"세번째" → 2
    "가자"·"진행"·"예"·"그거" → 0 (default·1순위)
    """
    if not text:
        return None
    t = text.strip().lower()
    # 명시 숫자
    if "1번" in t or "1순위" in t or "첫번째" in t or "첫 번째" in t:
        return 0
    if "2번" in t or "2순위" in t or "두번째" in t or "두 번째" in t:
        return 1
    if "3번" in t or "3순위" in t or "세번째" in t or "세 번째" in t:
        return 2
    # 짧은 답 — "1"·"2"·"3"
    if t in ("1", "2", "3"):
        return int(t) - 1
    # 확인 키워드 — default 0
    if any(kw == t or kw in t for kw in _PICK_CONFIRM):
        if len(t) <= 10:  # 너무 긴 답은 confirm 아님
            return 0
    return None


def is_pick_reject(text: str) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    if any(kw == t for kw in _PICK_REJECT):
        return True
    if len(t) <= 10 and any(kw in t for kw in _PICK_REJECT):
        return True
    return False


# ── dataclasses ──────────────────────────────────────────────────────
@dataclass
class NextActionSuggestion:
    """단일 추천 액션."""
    title: str = ""
    description: str = ""
    action_type: str = "other"
    # action_type 값:
    #   - "approve_pending_chain" — autonomous_runs.json 의 pending_approval chain 진행
    #   - "start_milestone_chain" — autonomous_workflow chain 시작
    #   - "autonomous_progress" — run_stage_one_tick 1회
    #   - "respond_to_actor" — multi-actor 응대 (자연어 hint)
    #   - "abandon_stale" — 오래된 pending chain 폐기
    #   - "research_task" — 자유 research chain
    #   - "other" — 자연어 hint 그대로 사용자에게 전달

    can_execute_now: bool = False
    trigger_command: str = ""    # 실행 시 사용할 자연어/slash 명령
    target_id: str = ""          # milestone_id·chain_id 등
    estimated_time: str = ""
    priority: int = 2            # 1=최우선·3=낮음

    def serialize(self) -> dict:
        return asdict(self)


@dataclass
class NextActionResponse:
    """LLM 의 추천 응답."""
    suggestions: list = field(default_factory=list)
    rationale: str = ""
    auto_pick_index: Optional[int] = None

    def serialize(self) -> dict:
        return {
            "suggestions": [s.serialize() for s in self.suggestions],
            "rationale": self.rationale,
            "auto_pick_index": self.auto_pick_index,
        }


# ── 컨텍스트 수집 ───────────────────────────────────────────────────
def _collect_context(stage_id: str = "") -> dict:
    """LLM 에 줄 컨텍스트 종합·graceful."""
    ctx = {
        "active_milestones": [],
        "active_jobs": [],
        "pending_chains": [],
        "active_actors": [],
        "work_state": "unknown",
        "active_window": "",
        "hour_of_day": 12,
        "learning_hint": "",
    }

    # GoalTree active milestones
    if stage_id:
        try:
            from eidos_goal_tree import (
                load_goal_tree, get_active_milestones,
            )
            t = load_goal_tree(stage_id)
            if t and t.root_goal_id:
                ams = get_active_milestones(t) or []
                ctx["active_milestones"] = [
                    {
                        "id": getattr(m, "id", ""),
                        "title": (getattr(m, "title", "") or "")[:60],
                        "target_date": getattr(m, "target_date", "") or "",
                        "progress": float(getattr(m, "progress", 0.0) or 0.0),
                    }
                    for m in ams[:6]
                ]
        except Exception:
            pass

    # Active jobs
    try:
        from eidos_job_manager import get_active_jobs
        jobs = get_active_jobs()
        ctx["active_jobs"] = [
            {
                "id": j.job_id,
                "title": j.title[:60],
                "status": j.status,
                "priority": j.priority,
            }
            for j in jobs[:5]
        ]
    except Exception:
        pass

    # Pending MissionChain (이전 시스템 잔존 — autonomous_runs.json)
    try:
        runs_path = os.path.join("eidos_files", "autonomous_runs.json")
        if os.path.exists(runs_path):
            with open(runs_path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            runs = data.get("runs") or []
            pending = [
                r for r in runs
                if str(r.get("status", "")).lower() == "pending_approval"
            ]
            ctx["pending_chains"] = [
                {
                    "chain_id": str(r.get("chain_id", "")) or str(r.get("id", "")),
                    "goal": (str(r.get("seed_goal", "")) or "?")[:80],
                    "n_stages": len(r.get("stages") or r.get("steps") or []),
                    "created_at": str(r.get("created_at", ""))[:19],
                }
                for r in pending[:3]
            ]
    except Exception:
        pass

    # Multi-actor belief
    if stage_id:
        try:
            from eidos_actor_belief import load_actor_store
            store = load_actor_store(stage_id)
            ctx["active_actors"] = [
                {"name": a.actor_name, "role": a.role,
                 "mention_count": a.mention_count,
                 "notes": a.notes[:80]}
                for a in sorted(
                    store.actors.values(),
                    key=lambda x: x.mention_count, reverse=True,
                )[:3]
            ]
        except Exception:
            pass

    # work_state, hour, sensory
    try:
        from eidos_belief_core import load_belief
        b = load_belief()
        ctx["work_state"] = b.work_state or "unknown"
    except Exception:
        pass
    try:
        import datetime as _dt
        ctx["hour_of_day"] = _dt.datetime.now().hour
    except Exception:
        pass
    try:
        from eidos_sensory_grounding import capture_signals
        sig = capture_signals()
        if sig:
            ctx["active_window"] = (sig.active_window_title or "")[:60]
    except Exception:
        pass

    # learning brief (짧은 버전)
    try:
        from eidos_learning_inject import build_short_learning_brief
        ctx["learning_hint"] = build_short_learning_brief(max_chars=200) or ""
    except Exception:
        pass

    return ctx


# ── LLM system prompt ────────────────────────────────────────────────
_SYSTEM = """\
너는 EIDOS 의 next-action recommender 다. 사용자가 모호하게 질문했을 때
**구체 액션 3개를 우선순위 순으로 제안**해라.

원칙:
- 각 추천은 즉시 실행 가능해야 함 — 추상 X·구체 X (정확한 chain·milestone·actor 지목)
- pending_chains 가 있으면 그게 거의 항상 1순위 (사용자가 이미 만들었던 의도 — 진행이 가장 자연스러움)
- active milestones 중 진행 정체된 것 → autonomous chain 시작 추천
- active_actors 중 mention_count 높은 entity 가 있으면 응대 액션 추천
- "alarmingly stale" 한 게 있으면 abandon_stale 도 OK (오래된 pending chain·끝없이 안 진행되는 milestone)
- 일반론·"커피 마시고 와요" 같은 잡담 추천 X·**EIDOS 가 같이 할 액션** 만

action_type 값:
  - "approve_pending_chain" — pending chain 진행
  - "start_milestone_chain" — milestone 에 자동 chain 시작
  - "autonomous_progress" — stage tick 1회 (이미 진행 중인 경우)
  - "respond_to_actor" — 특정 actor (client·colleague) 응대
  - "abandon_stale" — 오래된 pending chain 폐기
  - "research_task" — 자유 research chain
  - "other" — 위 분류 안 맞음·자연어 trigger 만

auto_pick_index: 사용자가 "알아서 해줘" 류로 던졌을 때 자동 선택할 추천 index (0~2).

JSON 만 출력:
{
  "rationale": "왜 이 추천 3개인지 한 줄",
  "auto_pick_index": 0~2,
  "suggestions": [
    {
      "title": "짧은 제목 (30자 이내)",
      "description": "1~2 문장 설명",
      "action_type": "...",
      "can_execute_now": true/false,
      "trigger_command": "실행 시 사용할 자연어 또는 slash (예: '이 chain 진행해줘')",
      "target_id": "milestone_id·chain_id (있으면)",
      "estimated_time": "5분 / 30분 / 1시간 등",
      "priority": 1/2/3
    },
    ...
  ]
}
"""


# ── 핵심: 추천 생성 ─────────────────────────────────────────────────
async def suggest_next_actions_async(
    user_text: str = "",
    stage_id: str = "",
    timeout_sec: float = 15.0,
) -> NextActionResponse:
    """LLM 으로 3개 추천 생성. 컨텍스트 종합·실패 시 빈 응답."""
    result = NextActionResponse()
    ctx = _collect_context(stage_id=stage_id)

    try:
        from llm_module import get_llm_response_async
    except Exception:
        result.rationale = "(LLM 모듈 부재)"
        return result

    # prompt 구성
    parts = []
    parts.append(f"[사용자 질문] {user_text[:200]}")
    parts.append(f"[work_state] {ctx['work_state']}")
    parts.append(f"[시간대] {ctx['hour_of_day']:02d}시")
    if ctx["active_window"]:
        parts.append(f"[활성 윈도우] {ctx['active_window']}")

    if ctx["pending_chains"]:
        parts.append("")
        parts.append("[pending chains (이전 세션 미승인·진행 시 valuable)]")
        for c in ctx["pending_chains"]:
            parts.append(
                f"  - chain_id={c['chain_id']} · goal={c['goal']} · "
                f"stages={c['n_stages']} · created={c['created_at']}"
            )

    if ctx["active_milestones"]:
        parts.append("")
        parts.append("[active milestones]")
        for m in ctx["active_milestones"]:
            parts.append(
                f"  - id={m['id'][:10]} · {m['title']} · "
                f"progress={m['progress']:.2f}"
                + (f" · target={m['target_date']}" if m['target_date'] else "")
            )

    if ctx["active_jobs"]:
        parts.append("")
        parts.append("[active jobs]")
        for j in ctx["active_jobs"]:
            parts.append(
                f"  - [{j['status']}] {j['title']} (priority {j['priority']})"
            )

    if ctx["active_actors"]:
        parts.append("")
        parts.append("[multi-actor belief]")
        for a in ctx["active_actors"]:
            parts.append(
                f"  - [{a['role']}] {a['name']} (언급 {a['mention_count']}회) "
                f"· {a['notes']}"
            )

    if ctx["learning_hint"]:
        parts.append("")
        parts.append(ctx["learning_hint"])

    parts.append("")
    parts.append("위 상황에서 EIDOS 가 사용자와 같이 할 구체 액션 3개 추천. JSON 만 출력.")
    prompt = "\n".join(parts)

    raw = ""
    try:
        raw = await asyncio.wait_for(
            get_llm_response_async(
                prompt,
                max_tokens=2048,
                system_prompt=_SYSTEM,
                response_mime_type="application/json",
            ),
            timeout=timeout_sec,
        )
    except asyncio.TimeoutError:
        result.rationale = "LLM timeout"
        return result
    except Exception as e:
        result.rationale = f"LLM 실패: {str(e)[:80]}"
        return result

    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return result
    except Exception:
        return result

    result.rationale = str(data.get("rationale", ""))[:200]
    try:
        api = data.get("auto_pick_index")
        if api is not None:
            result.auto_pick_index = max(0, min(2, int(api)))
    except Exception:
        result.auto_pick_index = None

    sugg_raw = data.get("suggestions") or []
    if not isinstance(sugg_raw, list):
        return result

    for i, sr in enumerate(sugg_raw[:3]):
        if not isinstance(sr, dict):
            continue
        title = str(sr.get("title", "")).strip()[:80]
        if not title:
            continue
        atype = str(sr.get("action_type", "other"))
        if atype not in (
            "approve_pending_chain", "start_milestone_chain",
            "autonomous_progress", "respond_to_actor",
            "abandon_stale", "research_task", "other",
        ):
            atype = "other"
        try:
            priority = int(sr.get("priority", 2))
            priority = max(1, min(3, priority))
        except Exception:
            priority = 2

        s = NextActionSuggestion(
            title=title,
            description=str(sr.get("description", ""))[:200],
            action_type=atype,
            can_execute_now=bool(sr.get("can_execute_now", False)),
            trigger_command=str(sr.get("trigger_command", ""))[:200],
            target_id=str(sr.get("target_id", ""))[:60],
            estimated_time=str(sr.get("estimated_time", ""))[:30],
            priority=priority,
        )
        result.suggestions.append(s)

    return result


# ── 카드 포맷 ────────────────────────────────────────────────────────
def format_suggestions_for_chat(
    response: NextActionResponse, enfp_mode: bool = False,
) -> str:
    """LLM 응답을 채팅 표시용 markdown 카드로."""
    if not response.suggestions:
        if enfp_mode:
            return (
                "🤔 에스더 님~ 지금 뭐 추천할지 잘 모르겠어요. "
                "활성 milestone 도 없고·pending chain 도 없네요. "
                "큰 목표 하나만 알려주세요!"
            )
        return "현재 추천할 액션이 없어요. milestone 부터 등록해주세요."

    icons = ["🥇", "🥈", "🥉"]
    lines = []
    if enfp_mode:
        lines.append("🤖 **에스더 님~ 지금 할 수 있는 거 추천드릴게요!**")
    else:
        lines.append("📋 **추천 액션**")
    if response.rationale:
        lines.append(f"_{response.rationale[:120]}_")
    lines.append("")

    for i, s in enumerate(response.suggestions[:3]):
        icon = icons[i] if i < len(icons) else "•"
        lines.append(f"### {icon} {s.title}")
        if s.description:
            lines.append(f"  {s.description}")
        meta = []
        if s.estimated_time:
            meta.append(f"⏱ {s.estimated_time}")
        if s.action_type and s.action_type != "other":
            meta.append(f"`{s.action_type}`")
        if s.can_execute_now:
            meta.append("✓ 즉시 실행 가능")
        if meta:
            lines.append(f"  {' · '.join(meta)}")
        lines.append("")

    lines.append("---")
    if enfp_mode:
        lines.append('💬 **"1번"** / **"2번"** / **"3번"** 또는 **"알아서 해줘"** 답주세요~')
    else:
        lines.append('답: `1번`·`2번`·`3번`·`알아서` (= 1순위)·`취소`')
    return "\n".join(lines)
