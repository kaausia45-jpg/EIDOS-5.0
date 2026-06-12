# eidos_belief_core.py
# [2026-05-26 Phase 1] ToM-core 도입 — 사용자에 대한 EIDOS 의 belief 모듈.
#
# 기존 Stage belief 와 분리: Stage = 외부 actor (의뢰자/플랫폼 등) 추론,
# UserBelief = EIDOS 가 *사용자 본인* 에 대해 누적하는 상태.
#
# 디자인 원칙:
#   1. LLM 호출 0 — 휴리스틱만 (Phase 1). decider/tick 단계에서 prior 로만 사용.
#   2. 매 메시지마다 가볍게 갱신 (수십 μs).
#   3. 디스크 영속화는 atomic write — N 메시지마다 또는 종료 시.
#   4. graceful — 손상되면 빈 belief 반환, throw 안 함.
#
# 저장 위치: eidos_files/agents/user_belief/{user_id}.json
# default user_id = "default" (single-user mode).

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional, Any


_BASE_DIR = os.path.join("eidos_files", "agents", "user_belief")
_VERSION = 1
_DEFAULT_USER_ID = "default"

WORK_STATES = ["working", "break", "casual", "done", "unknown"]
MOOD_SIGNALS = ["stress", "focus", "relaxed", "playful", "neutral"]
THREAD_STATUS = ["open", "awaiting_response", "resolved", "abandoned"]

# 메시지 텍스트 → work_state 추정 키워드 (한/영)
_KW_WORKING = (
    "할 일", "할일", "우선순위", "작업", "구현", "코드", "버그", "디버그",
    "리팩터", "리팩토링", "테스트", "fix", "todo", "build", "deploy",
    "오류", "에러", "함수", "클래스", "모듈", "import", "분석", "설계",
)
_KW_BREAK = ("쉬", "점심", "휴식", "커피", "잠시", "산책", "쉬자")
_KW_DONE = ("끝", "퇴근", "마무리", "수고", "내일 봐", "오늘 끝", "잘 가", "잘가")
# casual 모드는 **mode 변경 신호** 가 명확할 때만. "ㅇㅇ/ㄴㄴ" 같은 단순 응답
# 은 work 중에도 자주 쓰이므로 mode 신호로 보지 않음. mood 추정은 별도 함수.
_KW_CASUAL = ("ㅋㅋ", "ㅎㅎ", "재밌", "잡담", "농담", "수다")

# 메시지 텍스트 → mood_signal 추정
_KW_STRESS = ("ㅠ", "ㅜ", "짜증", "왜 이래", "안 돼", "안돼", "망했", "큰일", "문제")
_KW_FOCUS = ("집중", "확인", "점검", "검증", "정확", "꼼꼼", "차근차근")
_KW_RELAXED = ("괜찮", "좋네", "오케이", "ok", "완료", "굿", "good")
_KW_PLAYFUL = ("ㅋㅋ", "ㅎㅎ", "재밌", "웃", "하하", "헤헤", "농담")

# 시간 단위 (초)
_HOUR = 3600.0
_DAY = 86400.0

# energy/engagement 변동 파라미터
_ENERGY_GAIN_PER_MSG = 0.08
_ENERGY_DECAY_PER_HOUR = 0.06
_ENERGY_FLOOR = 0.05
_ENGAGEMENT_GAIN_PER_MSG = 0.18
_ENGAGEMENT_DECAY_PER_HOUR = 0.20
_ENGAGEMENT_FLOOR = 0.0

# recent_topics LRU 크기
_RECENT_TOPICS_MAX = 10

# pending_threads 최대 (오래된 abandoned 자동 정리)
_PENDING_THREADS_MAX = 50


# ── PendingThread ──────────────────────────────────────────────────────
@dataclass
class PendingThread:
    """미완 화제·일 — 사용자가 시작했으나 결말 안 난 것.

    EIDOS 가 proactive 하게 후속할 때 prior 가 됨. status 가 open/awaiting_response
    인 것만 후속 후보. resolved/abandoned 는 기록만.
    """

    id: str
    topic: str                  # 짧은 라벨 (예: "크몽 의뢰 응대")
    description: str = ""       # 맥락 (예: "신규 의뢰자한테 견적 보내야 함")
    started_at: str = ""
    last_referenced_at: str = ""
    status: str = "open"        # THREAD_STATUS
    importance: float = 0.5     # 0~1 — proactive 우선순위
    source: str = "user_msg"    # "user_msg" | "eidos_action" | "external"

    def serialize(self) -> dict:
        return asdict(self)

    @classmethod
    def deserialize(cls, data: dict) -> "PendingThread":
        if not isinstance(data, dict):
            data = {}
        now = _now()
        st = str(data.get("status") or "open")
        if st not in THREAD_STATUS:
            st = "open"
        try:
            imp = float(data.get("importance", 0.5))
        except Exception:
            imp = 0.5
        return cls(
            id=str(data.get("id") or _new_id("thread")),
            topic=str(data.get("topic") or "(이름 없음)"),
            description=str(data.get("description") or ""),
            started_at=str(data.get("started_at") or now),
            last_referenced_at=str(data.get("last_referenced_at") or now),
            status=st,
            importance=max(0.0, min(1.0, imp)),
            source=str(data.get("source") or "user_msg"),
        )


# ── UserBelief ─────────────────────────────────────────────────────────
@dataclass
class UserBelief:
    """EIDOS 가 단일 사용자에 대해 누적하는 belief 상태.

    Phase 1 에서는 휴리스틱으로만 갱신·LLM 호출 X. Phase 2 부터 chat hook
    에서 매 메시지 update_from_user_message() 호출.
    """

    user_id: str = _DEFAULT_USER_ID

    # 모드 추적
    work_state: str = "unknown"          # WORK_STATES
    work_state_since: str = ""           # 이 상태로 진입한 시각

    # 활동 신호 (0~1)
    mood_signal: str = "neutral"         # MOOD_SIGNALS
    energy: float = 0.5                  # 메시지마다 ↑, 시간 지나면 ↓
    engagement: float = 0.0              # 대화 활성도 — 빠르게 휘발

    # 기억
    pending_threads: list[dict] = field(default_factory=list)   # PendingThread.serialize() list
    recent_topics: list[str] = field(default_factory=list)      # LRU 10개
    known_facts: dict[str, str] = field(default_factory=dict)   # "name", "role", "timezone" 등
    preferences: dict[str, str] = field(default_factory=dict)   # 관찰된 선호

    # 타이밍
    last_message_at: str = ""
    last_proactive_at: str = ""
    # [Phase 5] EIDOS 가 마지막으로 응답·tool 실행한 시각 (proactive 무관 모든 활동).
    # update_from_eidos_action 이 매번 갱신. last_proactive_at 와는 다름 — 후자는
    # *먼저 말 건* 시점만 추적. last_eidos_action_at 은 모든 EIDOS activity.
    last_eidos_action_at: str = ""
    eidos_action_count: int = 0   # EIDOS 가 응답·실행한 누적 횟수
    message_count: int = 0
    # [Phase 7-A] proactive 사용자 피드백 학습 — scheduler 가 임계값 동적 조정에 사용.
    # rejection 1회마다 MIN_SILENCE·COOLDOWN 가산 (사용자 피로 방지).
    proactive_rejection_count: int = 0
    proactive_acceptance_count: int = 0
    # [Phase 8-A] 마지막 rejection 시각 — decay_rejection_count 가 경과 시간 계산에 사용.
    # 시간 지나면 count 자동 감소 (사용자 기분 변화 반영·영구 silence 회피).
    last_rejection_at: str = ""

    # 메타
    version: int = _VERSION
    created_at: str = ""
    updated_at: str = ""

    # ── helpers ──
    def serialize(self) -> dict:
        return asdict(self)

    @classmethod
    def deserialize(cls, data: dict) -> "UserBelief":
        if not isinstance(data, dict):
            data = {}
        now = _now()
        ws = str(data.get("work_state") or "unknown")
        if ws not in WORK_STATES:
            ws = "unknown"
        ms = str(data.get("mood_signal") or "neutral")
        if ms not in MOOD_SIGNALS:
            ms = "neutral"
        try:
            energy = float(data.get("energy", 0.5))
        except Exception:
            energy = 0.5
        try:
            engagement = float(data.get("engagement", 0.0))
        except Exception:
            engagement = 0.0
        try:
            mc = int(data.get("message_count", 0))
        except Exception:
            mc = 0

        return cls(
            user_id=str(data.get("user_id") or _DEFAULT_USER_ID),
            work_state=ws,
            work_state_since=str(data.get("work_state_since") or now),
            mood_signal=ms,
            energy=max(0.0, min(1.0, energy)),
            engagement=max(0.0, min(1.0, engagement)),
            pending_threads=list(data.get("pending_threads") or []),
            recent_topics=list(data.get("recent_topics") or []),
            known_facts=dict(data.get("known_facts") or {}),
            preferences=dict(data.get("preferences") or {}),
            last_message_at=str(data.get("last_message_at") or ""),
            last_proactive_at=str(data.get("last_proactive_at") or ""),
            last_eidos_action_at=str(data.get("last_eidos_action_at") or ""),
            eidos_action_count=int(data.get("eidos_action_count") or 0),
            message_count=mc,
            proactive_rejection_count=int(data.get("proactive_rejection_count") or 0),
            proactive_acceptance_count=int(data.get("proactive_acceptance_count") or 0),
            last_rejection_at=str(data.get("last_rejection_at") or ""),
            version=int(data.get("version") or _VERSION),
            created_at=str(data.get("created_at") or now),
            updated_at=str(data.get("updated_at") or now),
        )

    def get_pending_threads(self, status_filter: Optional[list[str]] = None) -> list[PendingThread]:
        """PendingThread 객체 list 반환. status_filter 주면 필터링."""
        out: list[PendingThread] = []
        for raw in self.pending_threads:
            th = PendingThread.deserialize(raw)
            if status_filter and th.status not in status_filter:
                continue
            out.append(th)
        return out


# ── 헬퍼 함수 ──────────────────────────────────────────────────────────
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


def _new_id(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _ensure_base() -> None:
    try:
        os.makedirs(_BASE_DIR, exist_ok=True)
    except Exception as e:
        print(f"[belief_core] _ensure_base 실패 (graceful): {e}")


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
        print(f"[belief_core] _atomic_write 실패 (graceful): {path} — {e}")
        return False


def _belief_path(user_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_\-가-힣]+", "_", user_id) or _DEFAULT_USER_ID
    return os.path.join(_BASE_DIR, f"{safe}.json")


# ── CRUD ───────────────────────────────────────────────────────────────
def new_belief(user_id: str = _DEFAULT_USER_ID) -> UserBelief:
    """빈 belief 생성. 디스크에 저장하지 않음 (save_belief 별도 호출)."""
    now = _now()
    return UserBelief(
        user_id=user_id or _DEFAULT_USER_ID,
        work_state="unknown",
        work_state_since=now,
        mood_signal="neutral",
        energy=0.5,
        engagement=0.0,
        created_at=now,
        updated_at=now,
    )


def load_belief(user_id: str = _DEFAULT_USER_ID) -> UserBelief:
    """user_id 의 belief 로드. 없거나 손상되면 새 belief."""
    uid = user_id or _DEFAULT_USER_ID
    path = _belief_path(uid)
    if not os.path.exists(path):
        return new_belief(uid)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return UserBelief.deserialize(data)
    except Exception as e:
        print(f"[belief_core] load_belief 손상 (graceful, 새 belief 반환): {uid} — {e}")
        return new_belief(uid)


def save_belief(belief: UserBelief) -> bool:
    """belief 디스크 저장 (atomic)."""
    if not belief or not belief.user_id:
        return False
    _ensure_base()
    belief.updated_at = _now()
    belief.version = _VERSION
    path = _belief_path(belief.user_id)
    try:
        payload = belief.serialize()
        return _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"[belief_core] save_belief 실패 (graceful): {e}")
        return False


def delete_belief(user_id: str = _DEFAULT_USER_ID) -> bool:
    """belief 파일 삭제 (테스트용·reset 용)."""
    path = _belief_path(user_id or _DEFAULT_USER_ID)
    try:
        if os.path.exists(path):
            os.remove(path)
        return True
    except Exception as e:
        print(f"[belief_core] delete_belief 실패 (graceful): {e}")
        return False


# ── 휴리스틱 갱신 ──────────────────────────────────────────────────────
def _detect_work_state(text: str) -> Optional[str]:
    """텍스트 키워드만 보고 work_state 추정. 단정 못 하면 None."""
    if not text:
        return None
    t = text.lower()
    if any(k in text for k in _KW_DONE):
        return "done"
    if any(k in text for k in _KW_BREAK):
        return "break"
    if any(k in text for k in _KW_WORKING) or any(k in t for k in _KW_WORKING):
        return "working"
    # casual 은 짧고 가벼운 메시지 + 이모지 분위기
    if len(text.strip()) <= 8 and any(k in text for k in _KW_CASUAL):
        return "casual"
    return None


def _detect_mood(text: str) -> Optional[str]:
    """텍스트 키워드만 보고 mood_signal 추정. 단정 못 하면 None."""
    if not text:
        return None
    t = text.lower()
    if any(k in text for k in _KW_STRESS):
        return "stress"
    if any(k in text for k in _KW_PLAYFUL) or any(k in t for k in _KW_PLAYFUL):
        return "playful"
    if any(k in text for k in _KW_FOCUS):
        return "focus"
    if any(k in text for k in _KW_RELAXED) or any(k in t for k in _KW_RELAXED):
        return "relaxed"
    return None


_TOPIC_TOKEN_RE = re.compile(r"[A-Za-z가-힣][A-Za-z가-힣0-9_\-]{1,}")


def _extract_topic_hint(text: str) -> Optional[str]:
    """긴 메시지에서 첫 의미 있는 명사구를 화제로 추출. 빈약하면 None."""
    if not text:
        return None
    text = text.strip()
    if len(text) < 6:
        return None
    tokens = _TOPIC_TOKEN_RE.findall(text)
    # 너무 짧은 토큰 거르고, 첫 3개를 화제 후보로
    meaningful = [t for t in tokens if len(t) >= 2][:3]
    if not meaningful:
        return None
    return " ".join(meaningful)


def _push_topic(belief: UserBelief, topic: str) -> None:
    """LRU push — 중복 있으면 앞으로 이동, max 초과 시 뒤 잘림."""
    if not topic:
        return
    topic = topic.strip()[:60]
    if topic in belief.recent_topics:
        belief.recent_topics.remove(topic)
    belief.recent_topics.insert(0, topic)
    if len(belief.recent_topics) > _RECENT_TOPICS_MAX:
        belief.recent_topics = belief.recent_topics[:_RECENT_TOPICS_MAX]


def decay_over_time(belief: UserBelief, now: Optional[_dt.datetime] = None) -> UserBelief:
    """energy / engagement 시간 감쇠. last_message_at 부터 경과 시간 기준."""
    n = now or _dt.datetime.utcnow()
    last = _parse_iso(belief.last_message_at)
    if last is None:
        return belief
    elapsed_h = max(0.0, (n - last).total_seconds() / _HOUR)
    if elapsed_h <= 0:
        return belief
    belief.energy = max(_ENERGY_FLOOR, belief.energy - _ENERGY_DECAY_PER_HOUR * elapsed_h)
    belief.engagement = max(_ENGAGEMENT_FLOOR, belief.engagement - _ENGAGEMENT_DECAY_PER_HOUR * elapsed_h)
    return belief


def update_from_user_message(
    belief: UserBelief,
    message_text: str,
    now: Optional[_dt.datetime] = None,
) -> UserBelief:
    """사용자 메시지 1건 받았을 때 belief 갱신. LLM 호출 0.

    갱신:
      - message_count +1
      - last_message_at = now
      - energy ↑, engagement ↑
      - work_state — 키워드 추정 가능하면 갱신 (불확실하면 그대로)
      - mood_signal — 키워드 추정 가능하면 갱신
      - recent_topics — 명사구 hint LRU push
      - decay_over_time 먼저 적용 (오래 비웠으면 감쇠 후 ↑)
    """
    if belief is None:
        belief = new_belief()
    n = now or _dt.datetime.utcnow()

    # 1) 시간 감쇠 먼저
    decay_over_time(belief, n)

    # 2) 메시지 카운트·시각
    belief.message_count += 1
    belief.last_message_at = n.isoformat(timespec="seconds") + "Z"

    # 3) energy/engagement 증가 (cap 1.0)
    belief.energy = min(1.0, belief.energy + _ENERGY_GAIN_PER_MSG)
    belief.engagement = min(1.0, belief.engagement + _ENGAGEMENT_GAIN_PER_MSG)

    text = message_text or ""

    # 4) work_state 추정 (불확실하면 보존)
    ws_hint = _detect_work_state(text)
    if ws_hint and ws_hint != belief.work_state:
        belief.work_state = ws_hint
        belief.work_state_since = belief.last_message_at

    # 5) mood_signal (불확실하면 보존)
    mh = _detect_mood(text)
    if mh:
        belief.mood_signal = mh

    # 6) recent_topics
    topic = _extract_topic_hint(text)
    if topic:
        _push_topic(belief, topic)

    belief.updated_at = belief.last_message_at
    return belief


def update_from_eidos_action(
    belief: UserBelief,
    action_id: str,
    summary: str = "",
    proactive: bool = False,
    now: Optional[_dt.datetime] = None,
) -> UserBelief:
    """EIDOS 가 어떤 행동을 했을 때 belief 갱신. proactive=True 면
    last_proactive_at 갱신.

    chat 응답·tool 실행·자동 follow-up 등 모두 이 함수로 흘려서
    Phase 5 (기존 features 통합) 의 단일 진입점으로 작동.
    """
    if belief is None:
        belief = new_belief()
    n = now or _dt.datetime.utcnow()
    ts = n.isoformat(timespec="seconds") + "Z"
    if proactive:
        belief.last_proactive_at = ts
    # [Phase 5] 모든 EIDOS activity 기록 (proactive 무관)
    belief.last_eidos_action_at = ts
    belief.eidos_action_count += 1
    belief.updated_at = ts
    return belief


# ── PendingThread 조작 ──────────────────────────────────────────────────
def add_pending_thread(
    belief: UserBelief,
    topic: str,
    description: str = "",
    importance: float = 0.5,
    source: str = "user_msg",
    now: Optional[_dt.datetime] = None,
) -> PendingThread:
    """미완 thread 등록. 동일 topic 이미 open 상태면 그것을 재사용 (중복 방지)."""
    n = now or _dt.datetime.utcnow()
    ts = n.isoformat(timespec="seconds") + "Z"

    # 동일 open thread 있으면 last_referenced_at 만 갱신 후 반환
    for raw in belief.pending_threads:
        existing = PendingThread.deserialize(raw)
        if existing.status in ("open", "awaiting_response") and existing.topic.strip() == topic.strip():
            existing.last_referenced_at = ts
            # raw 자리에서 교체
            idx = belief.pending_threads.index(raw)
            belief.pending_threads[idx] = existing.serialize()
            return existing

    th = PendingThread(
        id=_new_id("thread"),
        topic=topic.strip()[:60] or "(이름 없음)",
        description=description,
        started_at=ts,
        last_referenced_at=ts,
        status="open",
        importance=max(0.0, min(1.0, float(importance))),
        source=source if source in ("user_msg", "eidos_action", "external") else "user_msg",
    )
    belief.pending_threads.append(th.serialize())
    # max 초과 시 가장 오래된 abandoned 부터 정리
    _trim_pending_threads(belief)
    return th


def update_pending_thread_status(
    belief: UserBelief,
    thread_id: str,
    new_status: str,
    now: Optional[_dt.datetime] = None,
) -> bool:
    """thread status 변경. new_status 가 THREAD_STATUS 가 아니면 False."""
    if new_status not in THREAD_STATUS:
        return False
    n = now or _dt.datetime.utcnow()
    ts = n.isoformat(timespec="seconds") + "Z"
    for raw in belief.pending_threads:
        if raw.get("id") == thread_id:
            raw["status"] = new_status
            raw["last_referenced_at"] = ts
            return True
    return False


def touch_pending_thread(
    belief: UserBelief,
    thread_id: str,
    now: Optional[_dt.datetime] = None,
) -> bool:
    """[2026-05-26 Phase 8-E] status 변경 없이 last_referenced_at 만 갱신.

    proactive followup 발화 후 같은 thread 가 즉시 다시 후보가 되는 무한 반복
    버그를 막기 위함. 발화 = 사용자 인지 = '최근 언급된' 상태로 간주.
    그 결과 PENDING_THREAD_AGE_HOURS_MIN (30분) 임계값이 다시 시작됨.

    Returns:
        True 면 갱신 성공, False 면 thread_id 못 찾음.
    """
    n = now or _dt.datetime.utcnow()
    ts = n.isoformat(timespec="seconds") + "Z"
    for raw in belief.pending_threads:
        if raw.get("id") == thread_id:
            raw["last_referenced_at"] = ts
            return True
    return False


def _trim_pending_threads(belief: UserBelief) -> None:
    """max 초과 시 abandoned/resolved 중 오래된 것부터 제거."""
    if len(belief.pending_threads) <= _PENDING_THREADS_MAX:
        return
    # 우선 abandoned, 그다음 resolved, 그다음 last_referenced_at 오래된 것 순
    def _drop_priority(raw: dict) -> tuple:
        status_order = {"abandoned": 0, "resolved": 1, "awaiting_response": 2, "open": 3}
        return (
            status_order.get(raw.get("status", "open"), 3),
            raw.get("last_referenced_at", ""),
        )
    belief.pending_threads.sort(key=_drop_priority)
    belief.pending_threads = belief.pending_threads[-_PENDING_THREADS_MAX:]


# ── Phase 7-B: aged thread 자동 정리 ──────────────────────────────────
_AGED_THREAD_MAX_AGE_DAYS = 7.0


def trim_aged_threads(
    belief: UserBelief,
    max_age_days: float = _AGED_THREAD_MAX_AGE_DAYS,
    now: Optional[_dt.datetime] = None,
) -> int:
    """N일 이상 묵은 open/awaiting_response thread 를 abandoned 로 자동 전환.

    'started_at' 과 'last_referenced_at' 둘 다 max_age_days 보다 묵었어야 함 —
    최근에 다시 언급된 thread 는 살아있는 것으로 간주.

    proactive scheduler 의 evaluate() 첫 줄에서 호출 (자연스러운 cleanup 타이밍).

    Returns:
        실제 abandoned 로 전환된 thread 개수
    """
    if belief is None or not belief.pending_threads:
        return 0
    n = now or _dt.datetime.utcnow()
    max_secs = max_age_days * _DAY
    n_aged = 0
    for raw in belief.pending_threads:
        status = raw.get("status", "open")
        if status not in ("open", "awaiting_response"):
            continue
        started = _parse_iso(raw.get("started_at", ""))
        last_ref = _parse_iso(raw.get("last_referenced_at", ""))
        if started is None or last_ref is None:
            continue
        # 두 시각 모두 max_age_days 보다 묵음 (최근 언급 있으면 살아있음)
        if (n - started).total_seconds() < max_secs:
            continue
        if (n - last_ref).total_seconds() < max_secs:
            continue
        raw["status"] = "abandoned"
        raw["last_referenced_at"] = n.isoformat(timespec="seconds") + "Z"
        n_aged += 1
    return n_aged


# ── Phase 7-A: proactive 거부 학습 ─────────────────────────────────────
# 거부 신호 — proactive 후 5분 이내 사용자 메시지에 이런 키워드 있으면 rejection 으로 학습
_REJECTION_KEYWORDS = (
    "그만", "조용", "꺼져", "방해", "닥쳐", "시끄러",
    "안 받을", "안받을", "건드리지", "나중에", "노 thanks",
    "나중에 해", "그만해", "끼어들지", "관심 없",
)
_REJECTION_WINDOW_MIN = 5.0   # proactive 후 5분 이내 응답만 rejection 으로 학습


def register_proactive_response(
    belief: UserBelief,
    user_text: str,
    now: Optional[_dt.datetime] = None,
) -> Optional[str]:
    """proactive 직후 사용자 메시지를 받았을 때 호출.

    last_proactive_at 가 _REJECTION_WINDOW_MIN 이내일 때만 학습 대상.
    user_text 에 거부 키워드 있으면 rejection_count ++,
    없으면 acceptance_count ++ (정상 응답으로 간주).

    Returns:
        "rejected" | "accepted" | None (timer 윈도우 밖이라 무시)
    """
    if belief is None or not user_text:
        return None
    last = _parse_iso(belief.last_proactive_at)
    if last is None:
        return None
    n = now or _dt.datetime.utcnow()
    elapsed_min = (n - last).total_seconds() / 60.0
    if elapsed_min < 0 or elapsed_min > _REJECTION_WINDOW_MIN:
        return None    # 윈도우 밖 — 일반 응답으로 간주, 학습 안 함

    text = user_text.strip()
    is_rejection = any(kw in text for kw in _REJECTION_KEYWORDS)
    if is_rejection:
        belief.proactive_rejection_count += 1
        # [Phase 8-A] decay 기준점 — 마지막 rejection 시각 기록
        belief.last_rejection_at = n.isoformat(timespec="seconds") + "Z"
        return "rejected"
    else:
        belief.proactive_acceptance_count += 1
        return "accepted"


# ── Phase 8-D: work_state idle decay (장시간 침묵 → unknown 자동 전환) ─
# 사용자가 "오늘 일하자" 후 자리 비우면 work_state="working" 이 영원히 유지 →
# proactive timer 가 매 5분마다 "working 이라 silence" 결정 → 영원히 발화 X.
# 일정 시간 침묵 후 state 를 "unknown" 으로 자동 전환해서 proactive 가 발화 가능
# 하게. 사용자가 다시 active 하면 첫 메시지의 work_state 키워드로 재추정.
# 24시간 — 매우 보수적 default. 옛 테스트 fixture (60분~수시간 침묵 시나리오)
# 영향 받지 않게 매우 길게. 실 사용에서는 settings.json 의 proactive_idle_decay_min
# 으로 짧게 override (사용자 dogfood 5분·일반 사용 30분~1시간 권장).
_IDLE_DECAY_THRESHOLD_MIN = 60.0 * 24.0


def decay_work_state_if_idle(
    belief: UserBelief,
    threshold_min: float = _IDLE_DECAY_THRESHOLD_MIN,
    now: Optional[_dt.datetime] = None,
) -> bool:
    """last_message_at 부터 threshold_min 이상 침묵 시 work_state → "unknown".

    이미 unknown 이거나 last_message_at 없으면 변화 없음.
    scheduler.evaluate() 첫 줄에서 자동 호출.

    Returns:
        True 면 state 전환 일어남, False 면 변화 없음.
    """
    if belief is None or belief.work_state == "unknown":
        return False
    last = _parse_iso(belief.last_message_at)
    if last is None:
        return False
    n = now or _dt.datetime.utcnow()
    idle_min = (n - last).total_seconds() / 60.0
    if idle_min < threshold_min:
        return False
    belief.work_state = "unknown"
    belief.work_state_since = n.isoformat(timespec="seconds") + "Z"
    return True


# ── Phase 8-A: rejection decay (시간 회복) ─────────────────────────────
# 마지막 rejection 후 이만큼 지날 때마다 count -1 (최소 0).
# 거부 5회 → 10일 휴식 후 자연 0 회복. 너무 빠르면 학습 무의미·너무 느리면
# 사용자가 한 번 짜증냈을 때 영구 silence 위험. 2일이 균형.
_REJECTION_DECAY_INTERVAL_DAYS = 2.0


def decay_rejection_count(
    belief: UserBelief,
    interval_days: float = _REJECTION_DECAY_INTERVAL_DAYS,
    now: Optional[_dt.datetime] = None,
) -> int:
    """마지막 rejection 후 경과 일수에 따라 proactive_rejection_count 자동 감소.

    interval_days 마다 -1 (최소 0). 감소 시점은 last_rejection_at 기준 —
    감소가 일어나면 last_rejection_at 도 그만큼 앞당겨서 다음 decay 까지의
    경과 시간 정확히 누적되게.

    scheduler.evaluate() 첫 줄에서 자동 호출 (trim_aged_threads 옆).

    Returns:
        실제 감소시킨 count (0 이면 변화 없음)
    """
    if belief is None or belief.proactive_rejection_count <= 0:
        return 0
    last = _parse_iso(belief.last_rejection_at)
    if last is None:
        return 0
    n = now or _dt.datetime.utcnow()
    elapsed_days = (n - last).total_seconds() / _DAY
    if elapsed_days < interval_days:
        return 0
    n_decay = min(belief.proactive_rejection_count, int(elapsed_days // interval_days))
    if n_decay <= 0:
        return 0
    belief.proactive_rejection_count -= n_decay
    # last_rejection_at 를 n_decay × interval_days 만큼 앞당김 — 다음 decay 누적 정확
    new_last = last + _dt.timedelta(days=n_decay * interval_days)
    belief.last_rejection_at = new_last.isoformat(timespec="seconds") + "Z"
    return n_decay


# ── prompt brief ───────────────────────────────────────────────────────
def as_prompt_brief(belief: UserBelief, max_threads: int = 5) -> str:
    """LLM system prompt 에 inject 할 짧은 markdown.

    Phase 2 hook 에서 매 메시지 응답 prompt 에 prepend. 100~300 토큰 정도.
    """
    if belief is None:
        return ""
    lines: list[str] = []
    lines.append("## 사용자 belief (EIDOS 가 누적 관찰 중)")
    lines.append(f"- 모드: **{belief.work_state}** (이 상태 시작: {belief.work_state_since[:19]})")
    lines.append(f"- 감정 신호: {belief.mood_signal}  ·  energy={belief.energy:.2f}  ·  engagement={belief.engagement:.2f}")
    lines.append(f"- 누적 메시지: {belief.message_count}회")

    open_threads = [PendingThread.deserialize(r) for r in belief.pending_threads]
    open_threads = [t for t in open_threads if t.status in ("open", "awaiting_response")]
    open_threads.sort(key=lambda t: t.importance, reverse=True)
    if open_threads:
        lines.append("")
        lines.append(f"### 미완 thread ({len(open_threads)}개, 중요도 순)")
        for t in open_threads[:max_threads]:
            tag = "⏳" if t.status == "awaiting_response" else "•"
            lines.append(f"{tag} {t.topic} (importance={t.importance:.2f}) — {t.description[:60]}")

    if belief.recent_topics:
        lines.append("")
        lines.append("### 최근 화제 (LRU, 신→구)")
        lines.append("- " + " / ".join(belief.recent_topics[:6]))

    if belief.known_facts:
        kf = ", ".join(f"{k}={v}" for k, v in list(belief.known_facts.items())[:5])
        lines.append("")
        lines.append(f"### 알고 있는 사실: {kf}")

    return "\n".join(lines)


def summary_for_log(belief: UserBelief) -> dict:
    """진단/디버깅용 dict — print(json.dumps(...)) 로 쓰기 편한 형태."""
    if belief is None:
        return {}
    open_th = sum(1 for r in belief.pending_threads if r.get("status") == "open")
    awaiting_th = sum(1 for r in belief.pending_threads if r.get("status") == "awaiting_response")
    return {
        "user_id": belief.user_id,
        "work_state": belief.work_state,
        "mood_signal": belief.mood_signal,
        "energy": round(belief.energy, 3),
        "engagement": round(belief.engagement, 3),
        "message_count": belief.message_count,
        "pending_threads_open": open_th,
        "pending_threads_awaiting": awaiting_th,
        "recent_topics_n": len(belief.recent_topics),
        "last_message_at": belief.last_message_at,
        "last_proactive_at": belief.last_proactive_at,
    }
