# eidos_thread_detector.py
# [2026-05-26 Phase 5] 사용자 메시지에서 pending_thread 자동 추출.
#
# Phase 4 의 proactive followup 은 pending_threads 가 있어야 의미 있음.
# 그런데 Phase 1~4 는 코드에서 직접 add_pending_thread() 호출해야 했음.
# Phase 5 는 사용자 메시지에서 휴리스틱으로 thread 후보 자동 감지·등록.
#
# 디자인 원칙:
#   1. LLM 호출 0 — 명확 키워드 패턴만 (false positive 최소화).
#   2. 보수적 — 모호하면 등록 안 함. 사용자가 직접 말한 것만.
#   3. dedup 은 belief_core 의 add_pending_thread 가 처리 (동일 topic 재사용).
#   4. graceful — 감지 실패해도 chat 흐름 영향 X.
#
# 감지 패턴 (한국어):
#   A. "기다리" / "답장" / "답변" — awaiting_response 상태로 등록
#   B. "내일까지" / "이번 주" / "다음 주" — deadline thread (importance ↑)
#   C. "X 해야 해" / "X 부탁" / "X 요청" — todo thread
#   D. "X 의뢰" — 명시적 의뢰 thread (importance ↑)

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ── 감지 패턴 ──────────────────────────────────────────────────────────

# 패턴 A: 대기 상태 — awaiting_response
_PAT_AWAITING = re.compile(
    r"(.{3,30})\s*(?:답장|답변|회신|연락|연락처)?\s*"
    r"(?:기다리|기다림|대기|대답 안)"
)

# 패턴 B: deadline
_PAT_DEADLINE = re.compile(
    r"(.{3,30})\s*(?:내일까지|오늘 안에|이번 주|다음 주|다음주|이번주|곧)"
)

# 패턴 C: todo
_PAT_TODO = re.compile(
    r"(.{3,30})\s*(?:해야 해|해야 함|해야지|해야겠|부탁해|요청해|보내야)"
)

# 패턴 D: 의뢰·작업
_PAT_REQUEST = re.compile(
    r"(.{3,30})\s*(?:의뢰|작업|진행 중|진행중|시작|시작했|만드는 중)"
)

# 패턴별 importance·status 기본값
_PATTERN_DEFAULTS = {
    "awaiting": (0.7, "awaiting_response"),
    "deadline": (0.8, "open"),
    "todo": (0.6, "open"),
    "request": (0.7, "open"),
}

# topic 으로 부적합한 단어 (앞부분 잘라낼 때 stopword 같은 거)
_TOPIC_STOPWORDS = ("그거", "이거", "저거", "그건", "이건", "저건",
                    "지금", "방금", "아까", "조금", "잠깐", "정말", "진짜",
                    "오늘", "내일", "어제", "이번", "다음", "최근")


@dataclass
class ThreadCandidate:
    """감지된 thread 후보. add_pending_thread 의 인자로 그대로 쓰임."""
    topic: str
    description: str
    importance: float
    status: str   # "open" or "awaiting_response"
    pattern: str  # 디버그용: 어느 패턴 매칭됐는지


def _clean_topic(raw: str) -> str:
    """추출된 raw 명사구 정제 — 앞부분 stopword 제거·공백 정리·길이 cap.

    [2026-05-26 fix] 단일 토큰 + 동사 어미 (자/요/다/네/까/봐/해) 끝나면 reject —
    "시작해보자" 같은 동사구 topic 차단. 명사구만 topic 으로 허용.
    """
    t = raw.strip(" ,.~!?()[]{}\"'")
    # 시작 토큰이 stopword 면 다음 토큰부터
    parts = t.split()
    while parts and parts[0] in _TOPIC_STOPWORDS:
        parts.pop(0)
    cleaned = " ".join(parts).strip()
    if len(cleaned) > 40:
        cleaned = cleaned[:40]
    # 단일 토큰 + 동사 어미 → 명사구 아님·skip
    if cleaned and " " not in cleaned:
        _VERB_ENDINGS = ("자", "요", "다", "네", "까", "봐", "해", "야", "지", "어", "아")
        if cleaned[-1] in _VERB_ENDINGS:
            return ""   # reject
    return cleaned


def detect(message: str) -> list[ThreadCandidate]:
    """사용자 메시지에서 thread 후보 추출. 0개~여러 개.

    매우 짧은 메시지 (<10자) 또는 fast 메시지스러운 것은 skip.
    명확 패턴 매칭만 — false positive 회피.
    """
    if not message:
        return []
    text = message.strip()
    if len(text) < 10:
        return []

    candidates: list[ThreadCandidate] = []
    seen_topics: set[str] = set()

    # 각 패턴 순서대로 매칭
    for pat_name, regex in (
        ("awaiting", _PAT_AWAITING),
        ("deadline", _PAT_DEADLINE),
        ("todo", _PAT_TODO),
        ("request", _PAT_REQUEST),
    ):
        for m in regex.finditer(text):
            raw = m.group(1) or ""
            topic = _clean_topic(raw)
            if len(topic) < 2:
                continue
            if topic in seen_topics:
                continue
            # substring dedup — 다른 패턴이 같은 topic 의 더 긴/짧은 fragment 를
            # 이미 잡았다면 중복으로 간주, skip. (예: "X 의뢰자 답장 기다리는 중"
            # 에서 awaiting → "X 의뢰자 답장" 등록 후 request 패턴이 "X" 만
            # 잡는 경우 — fragment 라 skip.)
            if any(topic in s or s in topic for s in seen_topics):
                continue
            seen_topics.add(topic)
            imp, status = _PATTERN_DEFAULTS[pat_name]
            candidates.append(ThreadCandidate(
                topic=topic,
                description=text[:80],
                importance=imp,
                status=status,
                pattern=pat_name,
            ))
            # 한 패턴당 최대 1개 — 같은 메시지에서 thread 폭발 방지
            break

    return candidates


def register_to_belief(belief, candidates: list[ThreadCandidate], now=None) -> int:
    """감지된 후보를 belief 에 등록. add_pending_thread 호출.

    Args:
        now: 등록 시각 (test 용 — 명시하면 thread.started_at/last_referenced_at 가
             이 시각으로 설정됨. 미명시 시 utcnow). 옛 시그니처 호환 (default None).

    Returns:
        등록된 개수 (중복 dedup 으로 인해 candidates 보다 적을 수 있음)
    """
    if belief is None or not candidates:
        return 0
    try:
        from eidos_belief_core import add_pending_thread, update_pending_thread_status
    except Exception:
        return 0

    n_added = 0
    for c in candidates:
        try:
            th = add_pending_thread(
                belief, topic=c.topic, description=c.description,
                importance=c.importance, source="user_msg",
                now=now,
            )
            # awaiting_response 패턴이면 status 갱신
            if c.status == "awaiting_response" and th.status != "awaiting_response":
                update_pending_thread_status(belief, th.id, "awaiting_response", now=now)
            n_added += 1
        except Exception:
            continue
    return n_added
