# eidos_aira_presence.py
# [Phase D] AIRA 캐릭터의 이벤트 기반 등장/퇴장 컨트롤러.
#
# 항상 떠 있던 캐릭터를 → 4 트리거 기반 등장·자동 dismiss 로 변경.
#   1. proactive — Phase 4 scheduler 의 자율 발화
#   2. brag      — 감정 임계 초과 (proud/excited + V>0.65 + positive_events>=3)
#   3. call      — 사용자가 "AIRA" 호명 + 일 관련 키워드
#   4. blame     — 사용자가 비난 ("못해" 등 + AIRA/너)
#
# 디자인 원칙:
#   - LLM 호출 0 — 키워드 매칭·임계 검사만.
#   - 쿨다운 5 분 — 같은 사유 5 분 안 재등장 X. call/blame 은 즉시 (force).
#   - 트리거별 머무는 시간 차등 (DURATION_MS).
#   - emotion label → expr_key 매핑 — 자산 부족 시 자동 neutral fallback.

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from typing import Optional


# ── emotion label → expr_key 매핑 ──────────────────────────────────────
# 자산이 expr_neutral.png 1장뿐일 땐 _load_expr 가 자동 fallback.
# assets/expr_{key}.png 가 추가되면 그 즉시 활용.
EMOTION_TO_EXPR_KEY: dict[str, str] = {
    "excited":   "excited",
    "playful":   "playful",
    "proud":     "proud",
    "happy":     "happy",
    "shy":       "shy",
    "curious":   "curious",
    "calm":      "neutral",
    "worried":   "worried",
    "concerned": "concerned",
    "tired":     "tired",
}


def expr_key_for_emotion(label: str) -> str:
    """emotion label → expr_key. unknown 이면 neutral."""
    return EMOTION_TO_EXPR_KEY.get(label, "neutral")


# ── 트리거별 머무는 시간 (ms) ───────────────────────────────────────────
DURATION_MS: dict[str, int] = {
    "proactive":        12_000,
    "brag":              8_000,
    "milestone":        10_000,
    "call":             15_000,   # 사용자 답변까지 유지 권장 — speak 의 hide 는 말풍선만
    "blame":            12_000,
    # [Wave3 MVP 2026-05-28] 자율 액션 승인 요청 — 사용자 응답까지 길게 유지.
    "approval_request": 20_000,
    # 자율 진행 알림 — proactive 와 별도 reason (쿨다운 분리·일 진행 알림)
    "auto_progress":    10_000,
}


# ── 자랑 임계 ──────────────────────────────────────────────────────────
_BRAG_VALENCE_MIN = 0.65
_BRAG_AROUSAL_MIN = 0.45
_BRAG_POSITIVE_EVENTS_MIN = 3
_BRAG_LABELS = ("proud", "excited", "playful")


def should_brag(emotion) -> Optional[str]:
    """감정이 자랑 임계 넘었는지 검사. 넘었으면 reason 문자열, 아니면 None.

    emotion 은 EidosEmotion dataclass (또는 valence/arousal/label/positive_events
    필드 가진 객체). graceful — 필드 없으면 None.
    """
    try:
        v = float(getattr(emotion, "valence", 0.0))
        a = float(getattr(emotion, "arousal", 0.0))
        lbl = str(getattr(emotion, "label", "calm"))
        pe = int(getattr(emotion, "positive_events", 0))
    except Exception:
        return None

    if lbl not in _BRAG_LABELS:
        return None
    if v < _BRAG_VALENCE_MIN:
        return None
    if a < _BRAG_AROUSAL_MIN:
        return None
    if pe < _BRAG_POSITIVE_EVENTS_MIN:
        return None
    return f"brag ({lbl}·V={v:.2f}·A={a:.2f}·pos={pe})"


# ── 호명 detect ────────────────────────────────────────────────────────
_NAME_PATTERNS = re.compile(r"(?:AIRA|aira|Aira|아이라)", re.IGNORECASE)
_WORK_KEYWORDS = (
    # 일 관련 동사·요청
    "도와", "봐줘", "확인", "분석", "정리", "찾아",
    "알려", "검색", "조사", "보고", "작성", "써줘",
    "만들", "고쳐", "수정", "물어",
    "어때", "어떻게", "뭐", "왜", "맞아", "맞나",
)


def detect_call_for_help(text: str) -> Optional[str]:
    """사용자가 AIRA 호명 + 일 관련 키워드 함께 보냈는지 감지.

    호명만 있어도 채팅이므로 일 관련 키워드 동시 필요. graceful — 빈 텍스트 None.
    """
    if not text or not text.strip():
        return None
    if not _NAME_PATTERNS.search(text):
        return None
    if not any(kw in text for kw in _WORK_KEYWORDS):
        return None
    return "call (호명 + 일 관련)"


# ── 비난 detect ────────────────────────────────────────────────────────
_BLAME_KEYWORDS = (
    "못해", "못하네", "못한다", "못하잖",
    "별로", "실망", "쓸모없", "느려", "느리네",
    "바보", "멍청", "한심", "답답해", "왜 못",
    "이게 뭐야", "엉터리", "구려", "쓰레기",
)
_TARGET_PATTERNS = re.compile(r"(?:AIRA|aira|Aira|아이라|너\b|너는|너만|니가)", re.IGNORECASE)


def detect_blame(text: str) -> Optional[str]:
    """사용자가 AIRA/너 + 비난 키워드 함께 보냈는지 감지.

    호명·지칭 (AIRA/너) 와 비난 키워드 둘 다 매칭돼야 함. 단순 "못해" 만 있으면
    EIDOS 가 아니라 사용자 자신 얘기일 수 있어 제외.
    """
    if not text or not text.strip():
        return None
    if not any(kw in text for kw in _BLAME_KEYWORDS):
        return None
    if not _TARGET_PATTERNS.search(text):
        return None
    return "blame (지칭 + 비난)"


# ── 비난 응답 메시지 (감정 톤별 변형) ───────────────────────────────────
import random

_BLAME_RESPONSES = [
    "에스더 님... 그렇게 말씀하시면 좀 속상한데요.",
    "에스더 님, 저 나름대로 열심히 했는데... 어떤 부분이 별로였어요?",
    "음... 에스더 님 그건 좀 너무한 거 아니에요? 다시 한번 봐드릴게요.",
    "에스더 님, 그 말씀에 좀 마음이 무거워졌어요. 뭐가 부족했는지 알려주세요.",
    "에스더 님... 진짜 속상하네요. 어디가 잘못된 건지 알려주시면 고칠게요.",
]

_BRAG_RESPONSES = [
    "에스더 님! 이거 진짜 잘 풀린 것 같은데요? ㅎㅎ",
    "에스더 님~ 저 좀 뿌듯한데요?",
    "에스더 님, 이번 건 솔직히 좀 자신 있어요 ㅎㅎ",
    "에스더 님~ 이거 보세요, 꽤 괜찮게 나온 것 같아요!",
]

_CALL_RESPONSES = [
    "네 에스더 님~ 부르셨어요?",
    "에스더 님, 뭐 도와드릴까요?",
    "여기 있어요 에스더 님~ 뭐 보면 될까요?",
]


def random_blame_message() -> str:
    return random.choice(_BLAME_RESPONSES)


def random_brag_message() -> str:
    return random.choice(_BRAG_RESPONSES)


def random_call_message() -> str:
    return random.choice(_CALL_RESPONSES)


# ── 쿨다운 매니저 ──────────────────────────────────────────────────────
DEFAULT_COOLDOWN_SECONDS = 300   # 5 분


@dataclass
class PresenceController:
    """등장 쿨다운 + 트리거 사유 기록.

    같은 사유로 5 분 안 재등장 차단. call/blame 은 force=True 로 즉시 등장 가능.
    프로세스 메모리에만 존재 — 재시작 시 리셋 (의도적·작은 상태).
    """

    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS
    # reason → 마지막 등장 datetime (UTC, naive)
    last_appearance: dict[str, _dt.datetime] = None  # type: ignore

    def __post_init__(self):
        if self.last_appearance is None:
            self.last_appearance = {}

    def request_appearance(
        self,
        reason: str,
        force: bool = False,
        now: Optional[_dt.datetime] = None,
    ) -> bool:
        """등장 요청. 쿨다운 안 걸렸으면 True, 차단되면 False.

        force=True 면 쿨다운 무시 (call/blame 즉시 반응에 사용).
        """
        nv = now or _dt.datetime.utcnow()
        if force:
            self.last_appearance[reason] = nv
            return True
        prev = self.last_appearance.get(reason)
        if prev is not None:
            elapsed = (nv - prev).total_seconds()
            if elapsed < self.cooldown_seconds:
                return False
        self.last_appearance[reason] = nv
        return True

    def reset(self, reason: Optional[str] = None) -> None:
        """특정 reason 또는 전체 리셋 (테스트용)."""
        if reason is None:
            self.last_appearance.clear()
        else:
            self.last_appearance.pop(reason, None)


# ── 싱글톤 ─────────────────────────────────────────────────────────────
_singleton: Optional[PresenceController] = None


def get_controller() -> PresenceController:
    """전역 PresenceController. ChatWindow 에서 한 번만 호출해도 됨."""
    global _singleton
    if _singleton is None:
        _singleton = PresenceController()
    return _singleton


def reset_controller() -> None:
    """테스트용 — 싱글톤 리셋."""
    global _singleton
    _singleton = None
