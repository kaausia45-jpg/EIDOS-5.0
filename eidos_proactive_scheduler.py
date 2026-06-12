# eidos_proactive_scheduler.py
# [2026-05-26 Phase 4] ToM-core Proactive scheduler — EIDOS 가 먼저 말 걸기.
#
# Phase 1~3 은 모두 user-initiated (사용자 메시지 받아야 belief/decider 작동).
# Phase 4 는 background timer 가 일정 간격으로 belief 평가 → 4 verb 중 선택:
#
#   silence            : 끼어들 타이밍 X (대부분)
#   check_in           : "어떻게 진행 중이세요?" — 침묵 길어졌고 thread 없을 때
#   topic_shift        : "참, X 는 어떻게 됐어요?" — 최근 화제 기반
#   pending_followup   : "어제 보낸 그 X 답장 왔어요?" — 미완 thread 기반
#
# 안전망:
#   - default OFF (사용자가 settings 또는 햄버거 토글로 명시 활성화)
#   - 사용자 마지막 메시지 < 10분 → silence (방해 회피)
#   - 마지막 proactive < 60분 → silence (쿨다운)
#   - work_state == "working" → silence (집중 방해 X)
#   - 시간당 최대 PROACTIVE_HOURLY_LIMIT 회
#   - graceful 실패 — 어떤 오류도 EIDOS 본체 흐름 영향 X
#
# 디자인 원칙:
#   1. LLM 호출 0 (Phase 4-A). 메시지 템플릿 + belief 데이터로 합성.
#   2. 모든 임계값은 모듈 상단 상수 — dogfood 후 조정 쉽게.
#   3. proactive 후 update_from_eidos_action(proactive=True) 호출로
#      last_proactive_at 갱신·다음 평가의 prior 됨.

from __future__ import annotations

import datetime as _dt
import random
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from eidos_belief_core import UserBelief


# ── 임계값 (분 단위) ──────────────────────────────────────────────────
MIN_SILENCE_AFTER_USER_MIN = 10.0     # 사용자 마지막 메시지로부터 이만큼 지나야 proactive 후보
COOLDOWN_AFTER_PROACTIVE_MIN = 60.0   # proactive 보냈으면 이만큼 쉬어야 다음 proactive
PENDING_THREAD_AGE_HOURS_MIN = 0.5    # thread 가 이만큼 묵었을 때 followup 후보 (30분)
TOPIC_SHIFT_AFTER_USER_MIN = 30.0     # topic_shift 는 좀 더 오래 침묵 후

# 시간당 proactive 최대 횟수 (last_proactive_at 만 봄 — 더 정밀하면 별도 카운터)
PROACTIVE_HOURLY_LIMIT = 2

# verb 상수
VERB_SILENCE = "silence"
VERB_CHECK_IN = "check_in"
VERB_TOPIC_SHIFT = "topic_shift"
VERB_PENDING_FOLLOWUP = "pending_followup"

ALL_VERBS = (VERB_SILENCE, VERB_CHECK_IN, VERB_TOPIC_SHIFT, VERB_PENDING_FOLLOWUP)


# ── 메시지 템플릿 ──────────────────────────────────────────────────────
# [2026-05-26] 각 verb 별 8~10개로 다양화 — 같은 상황 반복 발화해도 단조롭지 X.
_CHECK_IN_TEMPLATES = (
    "지금 어떠세요? 도움 필요한 거 있으면 말씀 주세요.",
    "조용한 시간이네요. 진행 중이신 게 있으시면 언제든 불러주세요.",
    "오늘 어떻게 흘러가고 있어요?",
    "잠깐 쉬는 중이세요? 필요한 거 있으시면 말씀하세요.",
    "별일 없으세요? 정리할 거 있으면 도와드릴 수 있어요.",
    "지금 자리 비우신 거예요? 돌아오시면 한 번 봐드릴게요.",
    "휴식 중이시면 좋고요. 무언가 막혀계시면 함께 보죠.",
    "오늘 마무리는 잘 되고 있어요?",
)

_TOPIC_SHIFT_TEMPLATES = (
    "참, {topic} 는 어떻게 진행되고 있어요?",
    "{topic} 얘기 나왔던 거, 그 후로 어떻게 됐어요?",
    "조금 전 {topic} 관련해서 따로 봐드릴 거 있을까요?",
    "{topic} 부분 더 진행하실 거 있으면 말씀해주세요.",
    "{topic} 마무리는 어떻게 가져가시려고요?",
    "{topic} 관련해서 좀 더 정리할 게 있을까요?",
)

_PENDING_FOLLOWUP_TEMPLATES = (
    "**{topic}** 어떻게 됐어요? ({hint})",
    "참, **{topic}** 후속은 어떻게 됐는지 궁금해서요. ({hint})",
    "지난번 **{topic}** 관련해서 추가로 봐드릴 거 있을까요? ({hint})",
    "**{topic}** 그 후로 진행되셨어요? ({hint})",
    "혹시 **{topic}** 답변 받으셨나요? ({hint})",
    "**{topic}** 부분 마무리되셨으면 알려주세요. ({hint})",
)


# ── 결과 dataclass ────────────────────────────────────────────────────
@dataclass
class ProactiveDecision:
    """proactive 평가 결과.

    verb=silence 면 message="" — 채팅창에 아무것도 게시 안 함.
    그 외 verb 는 message 가 게시될 EIDOS 발화.
    """
    verb: str
    message: str = ""
    reason: str = ""
    referenced_thread_id: str = ""   # pending_followup 일 때

    def is_silence(self) -> bool:
        return self.verb == VERB_SILENCE


# ── 시간 helper ───────────────────────────────────────────────────────
def _parse_iso(ts: str) -> Optional[_dt.datetime]:
    if not ts:
        return None
    try:
        s = ts.replace("Z", "+00:00")
        return _dt.datetime.fromisoformat(s).replace(tzinfo=None)
    except Exception:
        return None


def _minutes_since(ts: str, now: _dt.datetime) -> Optional[float]:
    dt = _parse_iso(ts)
    if dt is None:
        return None
    return (now - dt).total_seconds() / 60.0


# ── core: evaluate ─────────────────────────────────────────────────────
def evaluate(
    belief: "UserBelief",
    now: Optional[_dt.datetime] = None,
    rng: Optional[random.Random] = None,
    min_silence_min_override: Optional[float] = None,
    cooldown_min_override: Optional[float] = None,
    idle_decay_min_override: Optional[float] = None,
    allow_work_state_speak: bool = False,
) -> ProactiveDecision:
    """현재 시점에서 EIDOS 가 proactive 액션을 할지·뭘 할지 결정.

    LLM 호출 0. 모든 분기 휴리스틱.

    Args:
        belief: 현재 사용자 belief (필수)
        now: 평가 시각 (test 용 — 기본 utcnow)
        rng: 메시지 템플릿 랜덤 선택용 (test 용)
        min_silence_min_override: settings.json 의 dogfood 임계값 (caller 가 전달)
        cooldown_min_override: settings.json 의 dogfood 쿨다운 (caller 가 전달)

    Returns:
        ProactiveDecision — verb=silence 가 대부분
    """
    n = now or _dt.datetime.utcnow()
    r = rng or random.Random()

    if belief is None:
        return ProactiveDecision(VERB_SILENCE, reason="belief None")

    # ── [Phase 10-B] long_memory 자동 compact — trim_aged_threads *직전* 에 실행.
    # 14일+ 묵은 resolved/abandoned thread 를 MemoryEpisode 로 보존 → 이어서
    # trim_aged_threads 가 7일+ open 을 abandoned 로 정리해도 의미 손실 X.
    # graceful — long_memory 모듈 없거나 실패해도 trim 은 진행.
    try:
        from eidos_long_memory import (
            load_long_memory, save_long_memory, compact_threads_to_episode,
        )
        _ltm = load_long_memory(belief.user_id)
        _new_eps = compact_threads_to_episode(belief, _ltm, now=n)
        if _new_eps:
            save_long_memory(_ltm)
            print(f"[ToM-core] long_memory compact: {len(_new_eps)} 신규 episode")
    except Exception as _e_ltm:
        print(f"[ToM-core] long_memory compact 실패 (graceful): {_e_ltm}")

    # ── [Phase 7-B] aged thread 자동 정리 — N일 묵은 open/awaiting → abandoned ──
    try:
        from eidos_belief_core import trim_aged_threads as _trim_aged
        _trim_aged(belief, now=n)
    except Exception:
        pass

    # ── [Phase 8-D] work_state idle decay — caller 가 명시적 override 줄 때만 동작 ──
    # override None 이면 idle decay 자체를 skip (단위 테스트 격리).
    # 실 사용은 chat_gui 가 settings 의 proactive_idle_decay_min 을 전달.
    if idle_decay_min_override is not None and idle_decay_min_override > 0:
        try:
            from eidos_belief_core import decay_work_state_if_idle as _decay_idle
            if _decay_idle(belief, threshold_min=idle_decay_min_override, now=n):
                print(f"[ToM-core] work_state idle decay → unknown "
                      f"({idle_decay_min_override:.0f}분+ 침묵)")
        except Exception:
            pass

    # ── [Phase 8-A] rejection decay — 시간 지나면 rejection_count 자동 감소 ──
    # 거부 5회 사용자도 10일 휴식 후 0 회복. 영구 silence 회피.
    try:
        from eidos_belief_core import decay_rejection_count as _decay_rej
        _n_decay = _decay_rej(belief, now=n)
        if _n_decay > 0:
            print(f"[ToM-core] rejection decay -{_n_decay} → 현재 "
                  f"{belief.proactive_rejection_count}")
    except Exception:
        pass

    # ── [Phase 7-A] 거부 학습 — 동적 임계값 계산 ──
    # caller (chat_gui) 가 settings.json 의 dogfood 임계값을 명시 인자로 전달.
    # 미명시면 모듈 default (10·60) 사용 — 단위 테스트가 settings 영향 받지 않음.
    _base_min_silence = (
        min_silence_min_override if min_silence_min_override is not None
        else MIN_SILENCE_AFTER_USER_MIN
    )
    _base_cooldown = (
        cooldown_min_override if cooldown_min_override is not None
        else COOLDOWN_AFTER_PROACTIVE_MIN
    )

    rj = max(0, int(getattr(belief, "proactive_rejection_count", 0)))
    effective_min_silence = _base_min_silence * (1.0 + rj * 0.5)
    effective_cooldown = _base_cooldown * (1.0 + rj * 0.3)

    # ── 1. work_state working — 집중 방해 X ──
    # [Wave3+ 2026-05-28] allow_work_state_speak=True (ENFP work-aware 모드) 면
    # working 상태에서도 발화 허용. caller (chat_gui) 가 work_state 별로 ENFP
    # 모듈에 다른 톤 prompt 전달 — 잡담 줄이고 push 중심. silence 가드는 user_idle/
    # 쿨다운으로만 진행 (방해 회피는 cooldown 으로 충분).
    if belief.work_state == "working" and not allow_work_state_speak:
        return ProactiveDecision(VERB_SILENCE, reason="work_state=working — 집중 방해 회피")

    # ── 2. 사용자 마지막 메시지 너무 가까움 ──
    min_user = _minutes_since(belief.last_message_at, n)
    if min_user is None:
        # 아직 채팅 시작 안 함 — silence
        return ProactiveDecision(VERB_SILENCE, reason="last_message_at 없음 (첫 세션)")
    if min_user < effective_min_silence:
        return ProactiveDecision(
            VERB_SILENCE,
            reason=f"사용자 마지막 메시지 {min_user:.1f}분 전 < {effective_min_silence:.1f} "
                   f"(rejection={rj})",
        )

    # ── 3. 마지막 proactive 쿨다운 ──
    min_proact = _minutes_since(belief.last_proactive_at, n)
    if min_proact is not None and min_proact < effective_cooldown:
        return ProactiveDecision(
            VERB_SILENCE,
            reason=f"쿨다운 — 마지막 proactive {min_proact:.1f}분 전 < {effective_cooldown:.1f} "
                   f"(rejection={rj})",
        )

    # ── 4. pending_thread 우선 — 묵은 thread 가 가장 흥미로움 ──
    open_threads = belief.get_pending_threads(status_filter=["open", "awaiting_response"])
    if open_threads:
        # 가장 importance 높고 오래된 thread 골라서 followup
        # (importance 동률이면 last_referenced_at 가장 오래된 것)
        def _score(t):
            ref_dt = _parse_iso(t.last_referenced_at) or n
            age_h = max(0.0, (n - ref_dt).total_seconds() / 3600.0)
            return (t.importance, age_h)
        open_threads.sort(key=_score, reverse=True)
        candidate = open_threads[0]
        # 너무 신선한 thread (방금 만든 거) 는 followup 안 함
        cand_age_min = _minutes_since(candidate.last_referenced_at, n)
        if cand_age_min is not None and cand_age_min >= PENDING_THREAD_AGE_HOURS_MIN * 60.0:
            template = r.choice(_PENDING_FOLLOWUP_TEMPLATES)
            hint = candidate.description[:50] if candidate.description else "후속이 궁금해요"
            msg = template.format(topic=candidate.topic, hint=hint)
            return ProactiveDecision(
                verb=VERB_PENDING_FOLLOWUP,
                message=msg,
                reason=f"pending thread '{candidate.topic}' importance={candidate.importance:.2f} age={cand_age_min:.0f}분",
                referenced_thread_id=candidate.id,
            )

    # ── 5. topic_shift — 최근 화제 있고 좀 더 오래 침묵했으면 ──
    if belief.recent_topics and min_user >= TOPIC_SHIFT_AFTER_USER_MIN:
        topic = belief.recent_topics[0]
        template = r.choice(_TOPIC_SHIFT_TEMPLATES)
        msg = template.format(topic=topic)
        return ProactiveDecision(
            verb=VERB_TOPIC_SHIFT,
            message=msg,
            reason=f"최근 화제 '{topic[:30]}' 기반·{min_user:.0f}분 침묵",
        )

    # ── 6. check_in — 가벼운 안부 (가장 부담 적은 형태) ──
    msg = r.choice(_CHECK_IN_TEMPLATES)
    return ProactiveDecision(
        verb=VERB_CHECK_IN,
        message=msg,
        reason=f"가벼운 check_in — {min_user:.0f}분 침묵 / thread 없거나 신선",
    )
