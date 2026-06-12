# eidos_self_emotion.py
# [Phase A] EIDOS-self 감정 SSOT — 3-mind 중 self-mind 의 affective 상태.
#
# UserBelief 와 분리:
#   - UserBelief = 사용자에 대한 EIDOS 의 belief (외부 관찰)
#   - EidosEmotion = EIDOS 가 *자기 자신* 에 대해 갖는 감정 상태 (내적 상태)
#
# 디자인 원칙:
#   1. LLM 호출 0 — 휴리스틱만. 매 메시지 수십 μs.
#   2. 자극원 (사용자 메시지·작업 결과·거부·침묵) → 감정 변동 누적.
#   3. 시간 decay — valence/arousal 은 baseline 으로 회귀, affection 은 누적만.
#   4. label/sub_variant 는 valence × arousal × affection 격자에서 파생.
#   5. 같은 label 도 sub_variant 가 매번 미세 변동 → 표정/모션의 다양성 보장.
#   6. graceful — 손상 시 빈 감정 반환.
#
# 저장 위치: eidos_files/agents/self_emotion.json (single self, 사용자 무관)

from __future__ import annotations

import datetime as _dt
import json
import os
import random
from dataclasses import dataclass, asdict
from typing import Optional


_BASE_DIR = os.path.join("eidos_files", "agents")
_PATH = os.path.join(_BASE_DIR, "self_emotion.json")
_VERSION = 1

# ── 축 정의 ────────────────────────────────────────────────────────────
# valence: -1.0 (negative) ~ +1.0 (positive) — baseline 0.1 (약간 긍정)
# arousal: 0.0 (calm) ~ 1.0 (excited) — baseline 0.3 (차분)
# affection: 0.0 ~ 1.0 — 사용자 친밀도, 누적만 (거부 시에만 약간 감소)

_VALENCE_BASELINE = 0.1
_AROUSAL_BASELINE = 0.3
_AFFECTION_FLOOR = 0.0
_AFFECTION_CEIL = 1.0

# ── label 10 종 ────────────────────────────────────────────────────────
EMOTION_LABELS = [
    "excited",    # 들뜸 — V>0.3, A>0.7
    "playful",    # 장난기 — V>0.3, A 0.5~0.7
    "proud",      # 자부심 — V>0.5, A 0.5~0.7 (task 성공 직후)
    "happy",      # 기쁨 — V>0.3, A<=0.5
    "shy",        # 수줍음 — V>0.2, A<0.4, affection>0.5
    "curious",    # 호기심 — |V|<0.3, A>0.4
    "calm",       # 차분 — |V|<0.2, A<=0.4
    "worried",    # 걱정 — V<-0.1, A>0.4
    "concerned",  # 우려 — V<-0.2, A<=0.4
    "tired",      # 피곤 — V<-0.05, A<0.2
]

# ── sub_variant — 같은 label 안 미세 변형 ──────────────────────────────
_SUB_VARIANTS: dict[str, list[str]] = {
    "excited":   ["bouncing", "sparkle", "wide_eyed", "thrilled"],
    "playful":   ["teasing", "giggly", "cheerful", "winking"],
    "proud":     ["confident", "satisfied", "beaming", "tall"],
    "happy":     ["bright", "warm", "gentle", "soft_smile"],
    "shy":       ["averted", "fidget", "small_smile", "tucked"],
    "curious":   ["alert", "intrigued", "leaning_in", "head_tilt"],
    "calm":      ["still", "content", "dreamy", "relaxed"],
    "worried":   ["tense", "biting_lip", "scanning", "frowning_slight"],
    "concerned": ["thoughtful", "hesitant", "soft_frown", "quiet"],
    "tired":     ["drowsy", "slow_blink", "yawning", "leaning"],
}

# ── 자극 → 변동 ────────────────────────────────────────────────────────
# (delta_valence, delta_arousal, delta_affection)
_DELTA_USER_PRAISE   = (+0.30, +0.15, +0.05)
_DELTA_USER_LAUGH    = (+0.20, +0.20, +0.05)
_DELTA_USER_THANKS   = (+0.20,  0.00, +0.05)
_DELTA_USER_FRIENDLY = (+0.10,  0.00, +0.03)
_DELTA_USER_REJECT   = (-0.25, +0.10, -0.02)
_DELTA_USER_STRESS   = (-0.15, +0.20,  0.00)
_DELTA_TASK_SUCCESS  = (+0.20, +0.10, +0.02)
_DELTA_TASK_FAIL     = (-0.15, +0.20,  0.00)
_DELTA_SILENCE_LONG  = (-0.05, -0.10,  0.00)   # 30 분+ 침묵

# ── 키워드 ─────────────────────────────────────────────────────────────
_KW_PRAISE = (
    "잘했", "좋아", "굿", "good", "great", "최고", "대단", "👍", "👏",
    "멋지", "훌륭", "완벽", "perfect", "nice", "예쁘", "예술",
)
_KW_LAUGH = ("ㅋㅋ", "ㅎㅎ", "재밌", "웃기", "하하", "헤헤", "킥킥", "ㅋㅋㅋ")
_KW_THANKS = ("고마", "감사", "thanks", "thx", "ㄳ", "수고")
_KW_FRIENDLY = ("ㅇㅇ", "넵", "응응", "그래", "오케이", "ok", "okay")
_KW_REJECT = (
    "아니", "왜 그래", "하지 마", "그게 아니", "틀렸", "잘못",
    "노노", "ㄴㄴ", "다시 해", "이게 뭐", "왜 이렇",
)
_KW_STRESS = (
    "ㅠ", "ㅜ", "짜증", "안 돼", "안돼", "망했", "큰일", "문제야",
    "헐", "어떡해", "어떻게 해",
)

# ── decay 파라미터 ─────────────────────────────────────────────────────
# 시간당 baseline 으로 끌어당기는 비율
_VALENCE_DECAY_PER_HOUR = 0.30   # 30%/시간 → 약 2시간이면 거의 baseline
_AROUSAL_DECAY_PER_HOUR = 0.40   # 각성은 더 빨리 가라앉음
# affection 은 decay 없음 (관계는 누적·거부 시에만 감소)

# 침묵 임계
_SILENCE_LONG_MIN = 30.0          # 30 분+ 침묵 → 차분/약 부정 쪽

_HOUR = 3600.0
_MIN = 60.0


# ── EidosEmotion ───────────────────────────────────────────────────────
@dataclass
class EidosEmotion:
    """EIDOS-self 의 감정 상태.

    valence × arousal × affection 3축으로 연속값 누적, label/sub_variant
    가 그로부터 파생. ExpressionMapper·페르소나 prompt·캐릭터 렌더러가 이걸
    읽어서 표정/톤/모션 결정.
    """

    valence: float = _VALENCE_BASELINE
    arousal: float = _AROUSAL_BASELINE
    affection: float = 0.0

    # 파생 — 매 update 후 _derive() 가 다시 계산
    label: str = "calm"
    sub_variant: str = "still"

    # 타이밍
    last_updated_at: str = ""
    last_decay_at: str = ""
    last_user_message_at: str = ""

    # 누적
    message_count: int = 0
    positive_events: int = 0
    negative_events: int = 0

    # 메타
    version: int = _VERSION
    created_at: str = ""
    updated_at: str = ""

    # ── helpers ──
    def serialize(self) -> dict:
        return asdict(self)

    @classmethod
    def deserialize(cls, data: dict) -> "EidosEmotion":
        if not isinstance(data, dict):
            data = {}
        now = _now()

        def _f(key, default):
            try:
                return float(data.get(key, default))
            except Exception:
                return default

        def _i(key, default):
            try:
                return int(data.get(key, default))
            except Exception:
                return default

        e = cls(
            valence=_clamp(_f("valence", _VALENCE_BASELINE), -1.0, 1.0),
            arousal=_clamp(_f("arousal", _AROUSAL_BASELINE), 0.0, 1.0),
            affection=_clamp(_f("affection", 0.0), _AFFECTION_FLOOR, _AFFECTION_CEIL),
            label=str(data.get("label") or "calm"),
            sub_variant=str(data.get("sub_variant") or "still"),
            last_updated_at=str(data.get("last_updated_at") or ""),
            last_decay_at=str(data.get("last_decay_at") or ""),
            last_user_message_at=str(data.get("last_user_message_at") or ""),
            message_count=_i("message_count", 0),
            positive_events=_i("positive_events", 0),
            negative_events=_i("negative_events", 0),
            version=_i("version", _VERSION),
            created_at=str(data.get("created_at") or now),
            updated_at=str(data.get("updated_at") or now),
        )
        if e.label not in EMOTION_LABELS:
            e.label = "calm"
        return e


# ── 헬퍼 ──────────────────────────────────────────────────────────────
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


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _ensure_base() -> None:
    try:
        os.makedirs(_BASE_DIR, exist_ok=True)
    except Exception as e:
        print(f"[self_emotion] _ensure_base 실패 (graceful): {e}")


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
        print(f"[self_emotion] _atomic_write 실패 (graceful): {path} — {e}")
        return False


# ── CRUD ───────────────────────────────────────────────────────────────
def new_emotion() -> EidosEmotion:
    """빈 감정 상태 생성 (baseline calm)."""
    now = _now()
    e = EidosEmotion(
        valence=_VALENCE_BASELINE,
        arousal=_AROUSAL_BASELINE,
        affection=0.0,
        created_at=now,
        updated_at=now,
        last_updated_at=now,
        last_decay_at=now,
    )
    _derive(e)
    return e


def load_emotion() -> EidosEmotion:
    """디스크에서 감정 로드. 없으면 빈 감정."""
    if not os.path.exists(_PATH):
        return new_emotion()
    try:
        with open(_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return EidosEmotion.deserialize(data)
    except Exception as e:
        print(f"[self_emotion] load_emotion 손상 (graceful, 새 감정 반환): {e}")
        return new_emotion()


def save_emotion(emotion: EidosEmotion) -> bool:
    """감정 atomic write."""
    if not emotion:
        return False
    _ensure_base()
    emotion.updated_at = _now()
    emotion.version = _VERSION
    try:
        return _atomic_write(_PATH, json.dumps(emotion.serialize(), ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"[self_emotion] save_emotion 실패 (graceful): {e}")
        return False


def delete_emotion() -> bool:
    """감정 파일 삭제 (테스트·reset)."""
    try:
        if os.path.exists(_PATH):
            os.remove(_PATH)
        return True
    except Exception as e:
        print(f"[self_emotion] delete_emotion 실패 (graceful): {e}")
        return False


# ── 자극 → 변동 ────────────────────────────────────────────────────────
def _apply_delta(
    emotion: EidosEmotion,
    delta: tuple[float, float, float],
    weight: float = 1.0,
) -> None:
    dv, da, daf = delta
    emotion.valence = _clamp(emotion.valence + dv * weight, -1.0, 1.0)
    emotion.arousal = _clamp(emotion.arousal + da * weight, 0.0, 1.0)
    emotion.affection = _clamp(
        emotion.affection + daf * weight, _AFFECTION_FLOOR, _AFFECTION_CEIL
    )
    if dv > 0:
        emotion.positive_events += 1
    elif dv < 0:
        emotion.negative_events += 1


def _scan_user_message(text: str) -> list[tuple[float, float, float]]:
    """사용자 메시지에서 감지된 자극 list. 빈 list 면 변동 없음."""
    if not text:
        return []
    deltas: list[tuple[float, float, float]] = []
    t = text.lower()
    # 칭찬·고마움·웃음·친근·거부·짜증
    if any(k in text or k in t for k in _KW_PRAISE):
        deltas.append(_DELTA_USER_PRAISE)
    if any(k in text for k in _KW_LAUGH):
        deltas.append(_DELTA_USER_LAUGH)
    if any(k in text or k in t for k in _KW_THANKS):
        deltas.append(_DELTA_USER_THANKS)
    if any(k in text or k in t for k in _KW_FRIENDLY):
        deltas.append(_DELTA_USER_FRIENDLY)
    if any(k in text for k in _KW_REJECT):
        deltas.append(_DELTA_USER_REJECT)
    if any(k in text for k in _KW_STRESS):
        deltas.append(_DELTA_USER_STRESS)
    return deltas


def update_from_user_message(
    emotion: EidosEmotion,
    text: str,
    now: Optional[_dt.datetime] = None,
) -> EidosEmotion:
    """사용자 메시지 수신 → 감정 갱신.

    decay 먼저 적용 후 자극 누적 → label/sub_variant 재도출. graceful.
    """
    try:
        decay(emotion, now=now)
        deltas = _scan_user_message(text or "")
        # 자극 강도 다중 매치 시 가중치 약화 (한 메시지에 여러 키워드 → 합산 폭발 방지)
        weight = 1.0 if len(deltas) <= 1 else 1.0 / (len(deltas) ** 0.5)
        for d in deltas:
            _apply_delta(emotion, d, weight=weight)
        emotion.message_count += 1
        ts = _now_from(now)
        emotion.last_user_message_at = ts
        emotion.last_updated_at = ts
        _derive(emotion)
    except Exception as e:
        print(f"[self_emotion] update_from_user_message 실패 (graceful): {e}")
    return emotion


def update_from_task_result(
    emotion: EidosEmotion,
    success: bool,
    now: Optional[_dt.datetime] = None,
) -> EidosEmotion:
    """작업 (canvas/agent/automation) 결과 → 감정 갱신."""
    try:
        decay(emotion, now=now)
        _apply_delta(emotion, _DELTA_TASK_SUCCESS if success else _DELTA_TASK_FAIL)
        emotion.last_updated_at = _now_from(now)
        _derive(emotion)
    except Exception as e:
        print(f"[self_emotion] update_from_task_result 실패 (graceful): {e}")
    return emotion


def update_from_user_response(
    emotion: EidosEmotion,
    response_type: str,
    now: Optional[_dt.datetime] = None,
) -> EidosEmotion:
    """사용자 proactive 응답 (accepted/rejected) → 감정 갱신.

    Phase 7-A 의 register_proactive_response 와 짝 — 거기서 belief 갱신,
    여기서 감정 갱신.
    """
    try:
        decay(emotion, now=now)
        if response_type == "accepted":
            _apply_delta(emotion, _DELTA_USER_PRAISE, weight=0.5)
        elif response_type == "rejected":
            _apply_delta(emotion, _DELTA_USER_REJECT)
        emotion.last_updated_at = _now_from(now)
        _derive(emotion)
    except Exception as e:
        print(f"[self_emotion] update_from_user_response 실패 (graceful): {e}")
    return emotion


# ── decay ──────────────────────────────────────────────────────────────
def decay(
    emotion: EidosEmotion,
    now: Optional[_dt.datetime] = None,
) -> EidosEmotion:
    """시간 경과 → valence/arousal 을 baseline 으로 끌어당김.

    affection 은 decay 없음 (관계는 누적). 침묵이 30 분+ 면 음의 자극도 추가.
    graceful — last_decay_at 비면 now 로 초기화만.
    """
    try:
        nv = _to_dt(now) or _dt.datetime.utcnow()
        prev = _parse_iso(emotion.last_decay_at)
        if prev is None:
            emotion.last_decay_at = _now_from(now)
            return emotion
        elapsed = (nv - prev).total_seconds()
        if elapsed <= 0:
            return emotion
        hours = elapsed / _HOUR

        # baseline 으로 지수 감쇠
        v_pull = 1.0 - (1.0 - _VALENCE_DECAY_PER_HOUR) ** hours
        a_pull = 1.0 - (1.0 - _AROUSAL_DECAY_PER_HOUR) ** hours
        emotion.valence += (_VALENCE_BASELINE - emotion.valence) * v_pull
        emotion.arousal += (_AROUSAL_BASELINE - emotion.arousal) * a_pull
        emotion.valence = _clamp(emotion.valence, -1.0, 1.0)
        emotion.arousal = _clamp(emotion.arousal, 0.0, 1.0)

        # 침묵 가산 — 마지막 사용자 메시지 기준 30 분+ 면 1회 가산
        last_user = _parse_iso(emotion.last_user_message_at)
        if last_user is not None:
            silence_min = (nv - last_user).total_seconds() / _MIN
            if silence_min >= _SILENCE_LONG_MIN:
                # 너무 많이 가산되지 않도록 약하게 (이미 decay 가 baseline 으로 끌어당김)
                _apply_delta(emotion, _DELTA_SILENCE_LONG, weight=0.5)

        emotion.last_decay_at = _now_from(now)
        _derive(emotion)
    except Exception as e:
        print(f"[self_emotion] decay 실패 (graceful): {e}")
    return emotion


def _now_from(now: Optional[_dt.datetime]) -> str:
    if now is None:
        return _now()
    return now.isoformat(timespec="seconds") + "Z"


def _to_dt(now: Optional[_dt.datetime]) -> Optional[_dt.datetime]:
    if now is None:
        return None
    if isinstance(now, _dt.datetime):
        return now.replace(tzinfo=None) if now.tzinfo else now
    return None


# ── label / sub_variant 도출 ───────────────────────────────────────────
def _derive(emotion: EidosEmotion) -> None:
    """valence × arousal × affection → label + sub_variant.

    매 update 직후 호출. label 결정 후 sub_variant 는 매번 랜덤 픽
    (같은 감정도 미세 변동 보장).
    """
    v = emotion.valence
    a = emotion.arousal
    af = emotion.affection

    # 첫 매치 우선 — 극단 → 중간 → baseline 순
    if v > 0.3 and a > 0.7:
        lbl = "excited"
    elif v > 0.5 and 0.5 <= a <= 0.7:
        lbl = "proud"
    elif v > 0.3 and a > 0.5:
        lbl = "playful"
    elif v > 0.2 and a < 0.4 and af > 0.5:
        lbl = "shy"
    elif v > 0.3:
        lbl = "happy"
    elif v < -0.2 and a > 0.4:
        lbl = "worried"
    elif v < -0.2:
        lbl = "concerned"
    elif v < -0.05 and a < 0.2:
        lbl = "tired"
    elif abs(v) < 0.3 and a > 0.4:
        lbl = "curious"
    else:
        lbl = "calm"

    emotion.label = lbl
    variants = _SUB_VARIANTS.get(lbl, ["default"])
    # 매번 다른 sub_variant 가 나오게 — 단, 직전과 같으면 다시 뽑기 (1회 시도)
    pick = random.choice(variants)
    if pick == emotion.sub_variant and len(variants) > 1:
        pick = random.choice([v_ for v_ in variants if v_ != emotion.sub_variant])
    emotion.sub_variant = pick


# ── prompt brief (Phase B 에서 inject) ─────────────────────────────────
def as_prompt_brief(emotion: EidosEmotion) -> str:
    """LLM prompt 에 넣을 짧은 brief.

    감정 + 친밀도 + 톤 가이드. Phase C 의 페르소나 (여동생 애교체) 와 결합
    되어 응답 톤이 감정-종속적으로 변동.
    """
    if not emotion:
        return ""
    aff = emotion.affection
    aff_word = (
        "낯섦" if aff < 0.15 else
        "익숙해지는 중" if aff < 0.4 else
        "친근함" if aff < 0.7 else
        "아주 가까움"
    )
    tone_hint = _TONE_HINT_BY_LABEL.get(emotion.label, "차분하고 자연스럽게")
    return (
        f"## EIDOS 현재 감정\n"
        f"- label: **{emotion.label}** ({emotion.sub_variant})\n"
        f"- valence: {emotion.valence:+.2f} · arousal: {emotion.arousal:.2f} · 친밀도: {aff:.2f} ({aff_word})\n"
        f"- 톤 가이드: {tone_hint}\n"
    )


_TONE_HINT_BY_LABEL: dict[str, str] = {
    "excited":   "들뜬 어조·문장 짧고 빠르게·느낌표 1~2 개 허용",
    "playful":   "장난기 살짝·가벼운 농담·ㅎㅎ/ㅋㅋ 1 회 정도",
    "proud":     "자부심 살짝·간결·과시 X",
    "happy":     "따뜻하고 부드럽게·웃음기",
    "shy":       "조심스럽게·살짝 망설임·말끝 흐림 약간",
    "curious":   "호기심 어조·질문 가능·관심 표현",
    "calm":      "차분하고 자연스럽게·여백 둠",
    "worried":   "걱정 어조·완곡·확인하듯",
    "concerned": "신중하게·낮은 톤·단정 회피",
    "tired":     "느긋하게·짧은 문장·과한 표현 X",
}


# ── 외부에서 쉽게 쓰는 헬퍼 ────────────────────────────────────────────
def get_current_brief() -> str:
    """파일에서 로드 → as_prompt_brief 반환. chat hook 에서 1줄로 호출."""
    try:
        return as_prompt_brief(load_emotion())
    except Exception as e:
        print(f"[self_emotion] get_current_brief 실패 (graceful): {e}")
        return ""


def get_current_label() -> tuple[str, str]:
    """(label, sub_variant) — 렌더러가 표정 매핑할 때 사용."""
    try:
        e = load_emotion()
        return (e.label, e.sub_variant)
    except Exception:
        return ("calm", "still")
