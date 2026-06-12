# eidos_commit_trigger.py
# [Wave16 2026-05-28] EIDOS 응답에서 commit pattern 감지 → 자율 chain 즉시 trigger.
#
# 문제 (1.png 보고):
#   EIDOS 가 사용자에게 "조사할게요·진행할게요·찾아볼게요" 라고 말만 하고
#   실제 chain 실행은 안 함. 사용자는 EIDOS 가 일한다고 믿는데 0건.
#
# 해결:
#   1. LLM 응답 자체에 commit pattern 감지 (동사 + 조사·진행·작성 등)
#   2. 매칭되면 사용자 직전 메시지를 chain topic 으로 추출
#   3. autonomous_workflow chain 즉시 trigger (5분 tick 대기 X)
#   4. chain 결과를 채팅에 forward
#
# Cost: 패턴 감지 = LLM 호출 0 (정규식 + 키워드). chain 자체만 LLM 비용.

from __future__ import annotations

import re


# ── commit verb keyword (단언·미래형) ──────────────────────────────
# "할게요·할 거예요·하겠습니다·하겠어요" 류 commit 어미
_COMMIT_ENDINGS = (
    "할게요", "할 거예요", "할거예요", "하겠습니다", "하겠어요",
    "할게", "할 게요", "갈게요", "갈 거예요",
    "해드릴게요", "드릴게요", "드릴거예요",
    "할 예정", "할 예정이",
    "진행하겠", "진행할게", "진행할거",
    "시작할게", "시작하겠", "시작할거",
    "보고할게", "보고드릴",
    "정리할게", "정리하겠", "정리해드릴",
    "찾아볼게", "찾아드릴", "찾아보겠",
    "만들어드릴", "만들게요", "만들겠",
    "작성할게", "작성하겠", "작성해드릴",
    "검색할게", "검색하겠",
    "분석할게", "분석하겠", "분석해드릴",
    "조사할게", "조사하겠", "조사해드릴",
    "확인할게", "확인하겠",
    "써드릴", "써볼게", "써볼게요",
)

# commit 동작 verb (chain trigger 가치 있는 거)
_COMMIT_VERBS = (
    "조사", "검색", "분석", "정리", "작성", "보고", "찾", "만들", "준비",
    "research", "search", "analyze", "summarize", "draft",
)

# "할 수 있나요"·"~될까요" 같은 질문 — 매칭 X
_QUESTION_MARKERS = (
    "?", "?", "할까요", "할까", "되나요", "맞나요", "괜찮나요",
)

# topic 추출용 강조 패턴 — "==X==" 또는 "**X**" 표시
_EMPHASIS_RE = re.compile(r"==([^=]{2,80})==")
_BOLD_RE = re.compile(r"\*\*([^*]{2,80})\*\*")

# [Wave16-B] LLM 자발 삽입 commit 마커
# 형식: [COMMIT: topic=X, action=research|write|analyze|search|plan|other]
_COMMIT_MARKER_RE = re.compile(
    r"\[COMMIT:\s*topic\s*=\s*([^,\n\]]+?)\s*,\s*action\s*=\s*([a-z_]+)\s*\]",
    re.IGNORECASE,
)

# 허용 action 값
_ALLOWED_ACTIONS = (
    "research", "write", "analyze", "search", "plan", "other",
)


def extract_commit_marker(text: str) -> dict:
    """[Wave16-B 2026-05-28] LLM 응답에서 자발 [COMMIT: ...] 마커 추출.

    Returns: {"found": bool, "topic": str, "action": str, "cleaned_text": str}
      - cleaned_text 는 마커 줄 제거된 본문 (사용자 표시용)
      - found=False 면 마커 없음·다른 필드 빈 값
    """
    out = {"found": False, "topic": "", "action": "", "cleaned_text": text or ""}
    if not text or not isinstance(text, str):
        return out
    m = _COMMIT_MARKER_RE.search(text)
    if not m:
        return out
    topic = m.group(1).strip()[:120]
    action = m.group(2).strip().lower()
    if action not in _ALLOWED_ACTIONS:
        action = "other"
    # cleaned_text — 마커 line 통째로 제거 (앞뒤 공백·newline 정리)
    cleaned = text[:m.start()].rstrip() + text[m.end():].lstrip()
    out["found"] = True
    out["topic"] = topic
    out["action"] = action
    out["cleaned_text"] = cleaned.strip()
    return out


def detect_commit_pattern(text: str) -> bool:
    """EIDOS 응답에 commit 표현 있는지 감지.

    원칙:
      - commit 어미 (할게요·하겠습니다 등) 1개 이상 매칭
      - commit verb (조사·분석 등) 1개 이상 매칭
      - 단·질문 marker (?·할까요) 우세하면 X (사용자에게 되묻는 패턴)
      - 너무 짧음 (< 10자) 도 X

    빈 입력 / 짧음 / 질문 위주 / commit 없음 → False.
    """
    if not text or not isinstance(text, str):
        return False
    t = text.strip()
    if len(t) < 10:
        return False

    # 질문이 본문 끝이면 commit 아님
    if t[-1] in ("?", "?"):
        # 단·앞 문장에 commit 있을 수도 — line-by-line 추가 검사 가능
        pass

    # commit 어미 매칭 (적어도 1개)
    has_ending = any(end in t for end in _COMMIT_ENDINGS)
    if not has_ending:
        return False

    # commit verb 매칭 (적어도 1개)
    has_verb = any(v in t for v in _COMMIT_VERBS)
    if not has_verb:
        return False

    # 질문 marker 빈도가 너무 높으면 commit 아님 (4 이상)
    n_q = sum(1 for m in _QUESTION_MARKERS if m in t)
    if n_q >= 4:
        return False

    return True


def extract_commit_topic(eidos_response: str, user_text: str = "") -> str:
    """chain topic 추출 — 우선순위:

    1. EIDOS 응답에서 ==강조== 부분 (가장 명확)
    2. EIDOS 응답에서 **bold** 부분
    3. 사용자 직전 메시지 (조사해줘·분석해줘 류 명령문)
    4. EIDOS 응답 첫 문장
    """
    # 1) ==강조==
    if eidos_response:
        m = _EMPHASIS_RE.search(eidos_response)
        if m:
            return m.group(1).strip()[:120]
        # 2) **bold**
        m = _BOLD_RE.search(eidos_response)
        if m:
            return m.group(1).strip()[:120]

    # 3) 사용자 직전 메시지
    if user_text and user_text.strip():
        t = user_text.strip()
        # 슬래시 명령 제외
        if not t.startswith("/"):
            # ~해줘·~조사 같은 명령문이면 그대로
            return t[:120]

    # 4) EIDOS 응답 첫 문장 (commit 어미까지)
    if eidos_response:
        for term in (". ", "! ", "다. ", "요. "):
            idx = eidos_response.find(term)
            if 0 < idx <= 80:
                return eidos_response[:idx + 2].strip()
        return eidos_response[:120].strip()

    return ""


def should_skip_commit_trigger(text: str) -> bool:
    """commit pattern 이 있어도 trigger 하지 말아야 할 경우:

    - "~할게요" 가 일상 인사·잡담 ("커피 마실게요")
    - "~할게요" 가 사용자에게 부탁 ("~해주실게요?")
    - 이미 chain 진행 중 표시 (별도 검사·여기선 X)
    """
    if not text:
        return True
    t = text.strip()
    # "해주실게요" 같은 부탁
    if "해주실" in t or "주실" in t:
        return True
    # 잡담 어휘 우세
    casual = ("커피", "쉬어", "쉴", "밥", "퇴근")
    if any(c in t for c in casual):
        # commit verb 가 있어야 — 잡담 어휘 + commit verb 둘 다면 모호 — skip
        return True
    return False
