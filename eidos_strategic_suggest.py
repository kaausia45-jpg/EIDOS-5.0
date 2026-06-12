# eidos_strategic_suggest.py
# [Wave3+ 2026-05-28] 전략적 제안 결정기.
#
# 기존 proactive_scheduler 는 4 verb (silence/check_in/topic_shift/pending_followup)
# 만 결정 — 발화 자체는 keyword 매칭. "지금 제안할 가치가 있나 + 뭘 제안할까"
# 를 LLM 으로 판단하는 한 단계 더 깊은 결정기.
#
# 입력 컨텍스트:
#   - active_milestones (제목·target_date·진행 staleness)
#   - belief.work_state ('working'·'break'·'casual'·'done'·'unknown')
#   - 마지막 사용자 메시지 후 분
#   - 마지막 proactive 후 분
#   - 시간대 (hour 0-23)
#   - recent_topics (belief.recent_topics)
#   - emotion label
#
# 출력:
#   - should_suggest: bool — 지금 제안할 가치 있는가
#   - urgency: 0.0~1.0 — 얼마나 시급한가 (0.7+ 만 발화 권장)
#   - suggestion_type: 'milestone_push' | 'celebrate_done' | 'break_suggest'
#                    | 'next_action_question' | 'observation_share' | 'none'
#   - message: ENFP 톤 발화 본문
#   - reason: 결정 근거 (디버깅용)
#
# 호출자 (chat_gui._on_proactive_tick) 가 urgency 기반 gate 후 사용:
#   if result.should_suggest and result.urgency >= settings.strategic_threshold:
#       → 그 message·suggestion_type 사용
#   else:
#       → 기존 proactive_scheduler fallback

from __future__ import annotations

import asyncio
import datetime as _dt
import json
from dataclasses import dataclass, field
from typing import Optional


_ALLOWED_TYPES = (
    "milestone_push",        # "Q1 매출 어디까지?"
    "celebrate_done",        # "방금 X 끝낸 거 짱이에요!"
    "break_suggest",         # "오랫동안 일했어요·잠깐 쉬어요"
    "next_action_question",  # "다음 뭘 할지 같이 정해요"
    "observation_share",     # "그러고보니 X 한 지 좀 됐네요"
    "stale_milestone_revisit",  # "Y 한참 안 건드렸어요"
    "none",
)


@dataclass
class StrategicSuggestion:
    """전략 제안 결과."""
    should_suggest: bool = False
    urgency: float = 0.0          # 0.0~1.0
    suggestion_type: str = "none"
    message: str = ""
    reason: str = ""
    raw: dict = field(default_factory=dict)


_LLM_SYSTEM = """\
너는 EIDOS 의 AIRA 페르소나 — ENFP 여동생. 사용자는 "에스더 님" 으로 부른다.

지금 사용자에게 자율 발화를 할 가치가 있는지 + 뭐를 말할지 결정해라.

전략 원칙:
- **무조건 발화하지 마라** — 사용자가 일에 집중 중이거나 방금 침묵 들어갔으면 조용히
- **타이밍 + 맥락이 맞을 때만 발화** — milestone 한참 안 건드림·완료 직후·휴식 적절·다음 액션 모호 등
- **활발 ENFP 톤 유지** ("에스더 님~", "오!", "ㅎㅎ") — 단 잡담만 X·맥락 있어야
- **호칭은 "에스더 님" 만** ("사장님"·"오빠"·"주인님" 금지)

suggestion_type 선택:
- milestone_push           — 활성 milestone 진행 push 가 적절 (working 모드 + 활성 ms 있음)
- celebrate_done           — 최근 milestone 완료한 흔적 → 칭찬·자랑
- break_suggest            — 사용자가 너무 오래 working — 짧은 휴식 권유 (40분+)
- next_action_question     — 다음 뭐 할지 모호한 시점 — 같이 정해요 류
- observation_share        — 가벼운 관찰 공유 ("Q1 한 지 좀 됐네요")
- stale_milestone_revisit  — 특정 milestone 너무 오래 stale (2일+) — 다시 보기 권유
- none                     — 발화 가치 없음 (urgency 0.0)

urgency 가이드:
- 0.0~0.3 = 안 함이 나음 (사용자 흐름 보호)
- 0.4~0.6 = 보통 — 호출자가 임계값 보고 결정
- 0.7~0.9 = 적극 발화 권장 (좋은 타이밍 + 의미 있는 내용)
- 1.0     = 매우 시급 (놓치면 안 됨·드물게)

응답 JSON 스키마만 출력:
{
  "should_suggest": true/false,
  "urgency": 0.0~1.0,
  "suggestion_type": "<위 목록 중 하나>",
  "message": "ENFP 톤 1~2문장 (60자 이내·실제 발화 본문·따옴표·라벨 X)",
  "reason": "결정 근거 짧은 한 줄"
}
"""


def _format_context(
    active_milestones: Optional[list],
    work_state: str,
    minutes_since_user: float,
    minutes_since_proactive: Optional[float],
    hour_of_day: int,
    recent_topics: Optional[list],
    emotion_label: Optional[str],
    milestone_stalenesses: Optional[dict] = None,
    actor_brief: str = "",
    sensory_brief: str = "",
) -> str:
    """LLM 에 줄 사용자 컨텍스트 문자열 만들기.

    [Wave8/9 추가] sensory_brief (시스템 신호)·actor_brief (multi-actor belief)
    있으면 함께 inject — 환경·관계 컨텍스트가 LLM 결정에 들어감.
    """
    lines = [
        f"[work_state] {work_state}",
        f"[시간대] {hour_of_day:02d}시",
        f"[사용자 마지막 메시지 후] {minutes_since_user:.1f}분",
    ]
    if minutes_since_proactive is not None:
        lines.append(f"[마지막 자율 발화 후] {minutes_since_proactive:.1f}분")
    else:
        lines.append("[마지막 자율 발화 후] 없음 (첫 발화 가능)")
    if emotion_label:
        lines.append(f"[현재 감정] {emotion_label}")
    if recent_topics:
        topics_str = ", ".join(str(t)[:30] for t in recent_topics[:3])
        lines.append(f"[최근 화제] {topics_str}")
    else:
        lines.append("[최근 화제] 없음")

    if active_milestones:
        lines.append(f"[활성 milestone] {len(active_milestones)}개")
        for i, m in enumerate(active_milestones[:5], 1):
            try:
                title = (getattr(m, "title", "") or "")[:60]
                target = (getattr(m, "target_date", "") or "")[:20]
                mid = getattr(m, "id", "")
                stale_days = ""
                if milestone_stalenesses and mid in milestone_stalenesses:
                    stale_days = (
                        f" / {milestone_stalenesses[mid]:.0f}일 미진행"
                        if milestone_stalenesses[mid] > 0.5 else ""
                    )
                lines.append(
                    f"  {i}. {title}"
                    f"{(' (목표일: ' + target + ')') if target else ''}"
                    f"{stale_days}"
                )
            except Exception:
                continue
    else:
        lines.append("[활성 milestone] 없음")

    # [Wave8/9] 환경·관계 brief inject
    if sensory_brief:
        lines.append("")
        lines.append(sensory_brief)
    if actor_brief:
        lines.append("")
        lines.append(actor_brief)

    # [Wave10] 누적 학습 brief inject — 매 결정에 과거 학습 반영
    try:
        from eidos_learning_inject import build_learning_brief
        _lb = build_learning_brief(max_chars=500)
        if _lb:
            lines.append("")
            lines.append(_lb)
    except Exception:
        pass

    return "\n".join(lines)


async def decide_strategic_suggestion_async(
    active_milestones: Optional[list] = None,
    work_state: str = "unknown",
    minutes_since_user: float = 0.0,
    minutes_since_proactive: Optional[float] = None,
    hour_of_day: Optional[int] = None,
    recent_topics: Optional[list] = None,
    emotion_label: Optional[str] = None,
    milestone_stalenesses: Optional[dict] = None,
    actor_brief: str = "",
    sensory_brief: str = "",
    timeout_sec: float = 10.0,
) -> StrategicSuggestion:
    """LLM 으로 지금 제안할 가치 + 내용 결정. 실패 시 should_suggest=False.

    [Wave8/9] actor_brief·sensory_brief 옵션 — multi-actor belief·시스템 신호
    인지 보강. 빈 string 이면 기존 동작 (회귀 0).

    Returns: StrategicSuggestion. urgency·suggestion_type·message 모두 LLM 출력.
    """
    result = StrategicSuggestion()
    if hour_of_day is None:
        hour_of_day = _dt.datetime.now().hour

    try:
        from llm_module import get_llm_response_async
    except Exception as e:
        result.reason = f"llm_module import 실패: {e}"
        return result

    ctx = _format_context(
        active_milestones=active_milestones,
        work_state=work_state,
        minutes_since_user=minutes_since_user,
        minutes_since_proactive=minutes_since_proactive,
        hour_of_day=hour_of_day,
        recent_topics=recent_topics,
        emotion_label=emotion_label,
        milestone_stalenesses=milestone_stalenesses,
        actor_brief=actor_brief,
        sensory_brief=sensory_brief,
    )
    prompt = ctx + "\n\n위 상황 보고 지금 자율 발화할지 결정 + 발화 본문."

    raw_text = ""
    try:
        raw_text = await asyncio.wait_for(
            get_llm_response_async(
                prompt,
                max_tokens=1024,
                system_prompt=_LLM_SYSTEM,
                response_mime_type="application/json",
            ),
            timeout=timeout_sec,
        )
    except asyncio.TimeoutError:
        result.reason = "LLM timeout"
        return result
    except Exception as e:
        result.reason = f"LLM 실패: {e}"
        return result

    try:
        data = json.loads(raw_text)
        if not isinstance(data, dict):
            result.reason = "non-dict JSON"
            return result
    except Exception as e:
        result.reason = f"JSON 파싱 실패: {e}"
        return result

    result.raw = data
    try:
        urgency = float(data.get("urgency", 0.0))
    except Exception:
        urgency = 0.0
    result.urgency = max(0.0, min(1.0, urgency))
    result.should_suggest = bool(data.get("should_suggest", False))
    stype = str(data.get("suggestion_type", "none"))
    if stype not in _ALLOWED_TYPES:
        stype = "none"
    result.suggestion_type = stype
    result.message = str(data.get("message", "")).strip()[:200]
    result.reason = str(data.get("reason", ""))[:200]
    # 안전망 — message 비면 suggest 강제 False
    if not result.message:
        result.should_suggest = False
        result.urgency = 0.0
    return result


def compute_milestone_stalenesses(
    milestones: Optional[list],
    now: Optional[_dt.datetime] = None,
) -> dict:
    """milestone.last_progress_at (또는 created_at) 기준 staleness 계산.

    Returns: {milestone_id: days_since_last_progress}.
    LLM 에 컨텍스트로 줘서 stale milestone 우선 push 결정에 사용.
    """
    nv = now or _dt.datetime.now()
    out: dict = {}
    for m in (milestones or []):
        try:
            mid = getattr(m, "id", "")
            if not mid:
                continue
            ts = (
                getattr(m, "last_progress_at", "")
                or getattr(m, "created_at", "")
                or ""
            )
            if not ts:
                out[mid] = 0.0
                continue
            try:
                dt_iso = _dt.datetime.fromisoformat(ts)
                days = (nv - dt_iso).total_seconds() / 86400.0
                out[mid] = max(0.0, days)
            except Exception:
                out[mid] = 0.0
        except Exception:
            continue
    return out
