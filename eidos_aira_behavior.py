# eidos_aira_behavior.py
# [Wave6 2026-05-28] AIRA 살아있는 동작 — 사람처럼 행동하게.
#
# 기존 AiraPopup 은 idle 시 sin curve ±8px 부유만 (수직). 사용자가 "둥둥 떠있다"
# 고 느낌. 이 모듈은 그 위에 behavior overlay 층을 추가:
#
#   - idle_subtle    : 약한 좌우 sway + 호흡감 (default)
#   - look_around    : 좌우 둘러보기 (호기심)
#   - lean_in        : 사용자 활동 직후 짝 기울이기 (reactive)
#   - ponder         : working 모드·고민하는 듯한 좌우 alternating
#   - daydream       : 오래 idle 시·느린 드리프트
#   - sleepy         : 늦은 밤·아래로 살짝·매우 느린 sway
#   - yawn / stretch : 가끔·짧은 vertical motion
#
# 핵심 원칙:
#   - PNG asset 1장으로도 살아있는 느낌 — dx/dy/scale_hint/opacity_mult 만 사용
#   - 너무 산만하지 않게 — 작업 중 (working) 이면 미세 동작 위주
#   - 시간·감정·user activity 반응
#
# Cost: LLM 호출 0. 순수 timer-driven transform.

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional


# ── 결과 dataclass ────────────────────────────────────────────────────
@dataclass
class BehaviorTransform:
    """한 tick 의 overlay 값. 절대값이 아닌 base 위 추가 offset."""
    dx: int = 0           # 좌우 추가 offset (px)
    dy: int = 0           # 위아래 추가 offset (px·기존 float 위)
    opacity_mult: float = 1.0   # 0.6~1.0·blink 등에 사용
    expr_hint: str = ""   # 표정 변경 권고 (옵션)


@dataclass
class BehaviorState:
    """전체 behavior 상태·picker 가 다음 behavior 결정에 사용."""
    current_behavior: str = "idle_subtle"
    started_at: float = 0.0           # time.time()
    duration_ms: int = 5000

    # 누적 컨텍스트
    last_user_msg_at: float = 0.0     # 사용자 메시지 마지막 시각
    last_eidos_response_at: float = 0.0
    last_trigger_at: float = 0.0      # reactive 마지막 트리거

    # 현재 표시 overlay (interpolation)
    overlay: BehaviorTransform = field(default_factory=BehaviorTransform)

    # picker 휴리스틱 입력
    work_state: str = "unknown"
    emotion_label: str = "calm"
    hour_of_day: int = 12

    def serialize(self) -> dict:
        return {
            "current_behavior": self.current_behavior,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "work_state": self.work_state,
            "emotion_label": self.emotion_label,
            "hour_of_day": self.hour_of_day,
        }


# ── Behavior 카탈로그 ──────────────────────────────────────────────────
# 각 entry: (min_duration_ms, max_duration_ms, tick_fn)
# tick_fn(elapsed_ms, duration_ms, state) → BehaviorTransform

def _idle_subtle_tick(elapsed_ms, duration_ms, state) -> BehaviorTransform:
    """기본 — 미세한 좌우 sway + 호흡 듯한 dy 변화."""
    # 좌우 sway — 매우 느린 sin
    phase = elapsed_ms / 1500.0  # 1.5초 주기
    dx = int(math.sin(phase) * 2)   # ±2px 매우 미세
    # dy — 호흡감 (느린 sin)
    breath_phase = elapsed_ms / 2200.0
    dy = int(math.sin(breath_phase) * 1)  # ±1px
    return BehaviorTransform(dx=dx, dy=dy)


def _look_around_tick(elapsed_ms, duration_ms, state) -> BehaviorTransform:
    """좌우 둘러보기 — 호기심·idle 시 가끔 발동.
    0~25%: 왼쪽으로 / 25~75%: 오른쪽 / 75~100%: 가운데로 복귀.
    """
    ratio = elapsed_ms / max(1, duration_ms)
    if ratio < 0.25:
        # 왼쪽 — ease in
        local = ratio / 0.25
        dx = -int(8 * (1 - (1 - local) ** 2))
    elif ratio < 0.75:
        # 오른쪽 — ease swing
        local = (ratio - 0.25) / 0.5
        # -8 → +8 (smooth swing)
        dx = int(-8 + 16 * local)
    else:
        # 복귀 — ease out
        local = (ratio - 0.75) / 0.25
        dx = int(8 * (1 - local) ** 2)
    return BehaviorTransform(dx=dx, dy=0)


def _lean_in_tick(elapsed_ms, duration_ms, state) -> BehaviorTransform:
    """사용자 활동 직후 — 살짝 앞으로 기울 (왼쪽으로·채팅창 쪽).
    빠른 lean + slow return.
    """
    ratio = elapsed_ms / max(1, duration_ms)
    if ratio < 0.3:
        local = ratio / 0.3
        dx = -int(7 * (1 - (1 - local) ** 2))
        dy = int(2 * local)
    else:
        local = (ratio - 0.3) / 0.7
        # 천천히 복귀
        dx = -int(7 * (1 - local) ** 1.5)
        dy = int(2 * (1 - local))
    return BehaviorTransform(dx=dx, dy=dy, expr_hint="curious")


def _ponder_tick(elapsed_ms, duration_ms, state) -> BehaviorTransform:
    """고민하는 듯 좌우 alternating — working 모드."""
    phase = elapsed_ms / 600.0   # 빠른 lean (사고)
    dx = int(math.sin(phase) * 3)
    return BehaviorTransform(dx=dx, dy=0, expr_hint="curious")


def _daydream_tick(elapsed_ms, duration_ms, state) -> BehaviorTransform:
    """오래 idle — 느리고 큰 sway·시선 멍."""
    phase = elapsed_ms / 2800.0
    dx = int(math.sin(phase) * 4)
    breath = elapsed_ms / 3500.0
    dy = int(math.sin(breath) * 2)
    return BehaviorTransform(dx=dx, dy=dy, expr_hint="neutral")


def _sleepy_tick(elapsed_ms, duration_ms, state) -> BehaviorTransform:
    """늦은 밤 — 아래로 약간·매우 느린 호흡·opacity 가끔 dip (졸음)."""
    breath = elapsed_ms / 4000.0
    dy = 2 + int(math.sin(breath) * 1)   # 평소보다 살짝 처짐 (dy=+2 base)
    dx = int(math.sin(elapsed_ms / 3500.0) * 1)
    # 가끔 opacity dip (졸음)
    opacity_mult = 1.0
    blink_phase = elapsed_ms / 1800.0
    if math.sin(blink_phase) > 0.92:
        opacity_mult = 0.85
    return BehaviorTransform(dx=dx, dy=dy, opacity_mult=opacity_mult,
                             expr_hint="tired")


def _yawn_tick(elapsed_ms, duration_ms, state) -> BehaviorTransform:
    """짧은 yawn — 살짝 위로 늘어났다 복귀."""
    ratio = elapsed_ms / max(1, duration_ms)
    # 0 → -5 (위로 늘어남) → 0
    if ratio < 0.5:
        local = ratio / 0.5
        dy = -int(5 * local)
    else:
        local = (ratio - 0.5) / 0.5
        dy = -int(5 * (1 - local))
    return BehaviorTransform(dx=0, dy=dy, expr_hint="tired")


def _stretch_tick(elapsed_ms, duration_ms, state) -> BehaviorTransform:
    """기지개 — 위로 더 늘어났다 복귀."""
    ratio = elapsed_ms / max(1, duration_ms)
    if ratio < 0.4:
        local = ratio / 0.4
        dy = -int(8 * local)
    else:
        local = (ratio - 0.4) / 0.6
        dy = -int(8 * (1 - local) ** 1.5)
    dx = int(math.sin(elapsed_ms / 400.0) * 1)
    return BehaviorTransform(dx=dx, dy=dy, expr_hint="happy")


# (min_dur, max_dur, tick_fn)
BEHAVIORS: dict[str, tuple] = {
    "idle_subtle": (3000, 6000,  _idle_subtle_tick),
    "look_around": (1500, 2500,  _look_around_tick),
    "lean_in":     (1200, 1500,  _lean_in_tick),
    "ponder":      (2500, 4000,  _ponder_tick),
    "daydream":    (3500, 5500,  _daydream_tick),
    "sleepy":      (4000, 7000,  _sleepy_tick),
    "yawn":        (1000, 1500,  _yawn_tick),
    "stretch":     (1200, 1800,  _stretch_tick),
}


# ── 다음 behavior picker ─────────────────────────────────────────────
def pick_next_behavior(state: BehaviorState, now: Optional[float] = None) -> str:
    """현재 상태 보고 다음 behavior 결정. 가중 random.

    우선순위 적용:
      - sleepy (23~5 시): 70% sleepy·25% yawn·5% idle_subtle
      - working: 60% idle_subtle·25% ponder·10% stretch·5% look_around
      - break + 오래 idle (>10 분): 40% daydream·30% look_around·20% yawn·10% idle_subtle
      - 그 외 (casual·unknown): 50% idle_subtle·20% look_around·15% ponder·15% stretch
    """
    nv = now if now is not None else time.time()
    h = state.hour_of_day

    # late night
    if h >= 23 or h <= 5:
        return _weighted_choice([
            ("sleepy", 0.7), ("yawn", 0.25), ("idle_subtle", 0.05),
        ])

    # working
    if state.work_state == "working":
        return _weighted_choice([
            ("idle_subtle", 0.6), ("ponder", 0.25),
            ("stretch", 0.1), ("look_around", 0.05),
        ])

    # break + long idle
    minutes_idle = (
        (nv - state.last_user_msg_at) / 60.0
        if state.last_user_msg_at > 0 else 999.0
    )
    if (state.work_state in ("break", "casual")
            or minutes_idle > 10.0):
        return _weighted_choice([
            ("daydream", 0.35), ("look_around", 0.3),
            ("yawn", 0.2), ("idle_subtle", 0.15),
        ])

    # 기본
    return _weighted_choice([
        ("idle_subtle", 0.5), ("look_around", 0.2),
        ("ponder", 0.15), ("stretch", 0.15),
    ])


def _weighted_choice(items: list) -> str:
    """items = [(name, weight), ...]·가중 random pick."""
    total = sum(w for _, w in items)
    r = random.random() * total
    acc = 0.0
    for name, w in items:
        acc += w
        if r <= acc:
            return name
    return items[-1][0]


# ── tick advance ─────────────────────────────────────────────────────
def advance_behavior(
    state: BehaviorState, now: Optional[float] = None,
) -> BehaviorTransform:
    """매 100ms 외부에서 호출. state 갱신 + 현재 overlay 계산.

    Returns: 적용할 BehaviorTransform.
    """
    nv = now if now is not None else time.time()
    if state.started_at <= 0:
        state.started_at = nv
        state.current_behavior = "idle_subtle"
        state.duration_ms = random.randint(*BEHAVIORS["idle_subtle"][:2])

    elapsed_ms = int((nv - state.started_at) * 1000)
    if elapsed_ms >= state.duration_ms:
        # behavior 전환
        next_name = pick_next_behavior(state, now=nv)
        state.current_behavior = next_name
        state.started_at = nv
        elapsed_ms = 0
        min_d, max_d, _ = BEHAVIORS[next_name]
        state.duration_ms = random.randint(min_d, max_d)

    # 현재 behavior 의 tick_fn 호출
    entry = BEHAVIORS.get(state.current_behavior)
    if entry is None:
        state.overlay = BehaviorTransform()
        return state.overlay
    _, _, tick_fn = entry
    try:
        transform = tick_fn(elapsed_ms, state.duration_ms, state)
        if not isinstance(transform, BehaviorTransform):
            transform = BehaviorTransform()
    except Exception as e:
        print(f"[aira-behavior] tick {state.current_behavior} 실패 (graceful): {e}")
        transform = BehaviorTransform()
    state.overlay = transform
    return transform


def trigger_reactive(
    state: BehaviorState, kind: str = "lean_in",
    now: Optional[float] = None,
) -> None:
    """사용자 활동 hook 에서 호출 — 즉시 reactive behavior 로 전환.

    kind: lean_in·look_around·stretch·yawn 중 하나. 알 수 없으면 lean_in.
    """
    nv = now if now is not None else time.time()
    if kind not in BEHAVIORS:
        kind = "lean_in"
    state.current_behavior = kind
    state.started_at = nv
    state.last_trigger_at = nv
    min_d, max_d, _ = BEHAVIORS[kind]
    state.duration_ms = random.randint(min_d, max_d)


def update_context(
    state: BehaviorState,
    last_user_msg_at: Optional[float] = None,
    work_state: Optional[str] = None,
    emotion_label: Optional[str] = None,
    hour_of_day: Optional[int] = None,
) -> None:
    """외부 컨텍스트 동기화. None 인 인자는 그대로."""
    if last_user_msg_at is not None:
        state.last_user_msg_at = last_user_msg_at
    if work_state is not None:
        state.work_state = work_state
    if emotion_label is not None:
        state.emotion_label = emotion_label
    if hour_of_day is not None:
        state.hour_of_day = hour_of_day
