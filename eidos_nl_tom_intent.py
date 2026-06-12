# eidos_nl_tom_intent.py
# [Wave3+ 2026-05-28] 자연어 → ToM 에이전트 명령 분류기.
#
# 사용자가 "/goal 내년 매출 5000만" 대신 "내년 매출 5000만 목표로 잡자",
# "/decompose 4" 대신 "분기별로 4개 쪼개줘", "/done X" 대신 "그 X 끝났어"
# 같은 자연어로 ToM 시스템 조작 가능하게.
#
# LLM 기반 (Gemini Flash·max 1024·timeout 8s). confidence < 0.7 면 none →
# 일반 chat 흐름으로 fall-through. 안전 default 보수적.
#
# 매칭 가능 action:
#   goal_set            — '<text>' 목표로 잡자, 목표 바꿔서, 목표 설정
#   goal_decompose      — 분기/마일스톤 별로 N개 쪼개, 세부 단계 잡아, 분해
#   milestone_done      — 'X 끝났어/완료/달성/마무리' substring
#   tick_run            — 한 번 돌려/진행해/지금 가봐/에이전트 가봐
#   status_query        — 상태 어때/어디까지 왔어/진행률/목표 진행
#   stage_attach        — 'X 에이전트 켜/X 호출' (음성모드 없이 채팅에서도)
#   stage_detach        — '에이전트 꺼/끝/그만'
#   none                — ToM 명령 아님 → fall-through

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Optional


# ── 결과 dataclass ─────────────────────────────────────────────────────
@dataclass
class NlTomIntent:
    """분류 결과 — main intent + 추출 args."""
    action: str = "none"
    confidence: float = 0.0
    reason: str = ""
    # action 별 args (일부만 채워짐)
    goal_text: str = ""             # goal_set
    decompose_n: int = 0            # goal_decompose (0=default 4)
    done_query: str = ""            # milestone_done substring
    stage_query: str = ""           # stage_attach
    # diagnostic
    raw: dict = field(default_factory=dict)

    def is_actionable(self, conf_threshold: float = 0.7) -> bool:
        return (
            self.action != "none"
            and self.confidence >= conf_threshold
        )


_ALLOWED_ACTIONS = (
    "goal_set", "goal_decompose", "milestone_done",
    "tick_run", "status_query",
    "stage_attach", "stage_detach",
    "none",
)


# ── 규칙 기반 빠른 사전 필터 (LLM 호출 회피) ─────────────────────────
# 사용자 입력에 ToM 키워드 1개도 없으면 LLM 호출 안 함 → 비용 절감.
_TOM_KEYWORDS = (
    "목표", "milestone", "마일스톤",
    "tick", "틱", "에이전트", "스테이지", "stage",
    "분해", "쪼개", "단계",
    "끝났", "완료", "달성", "마무리",
    "진행률", "상태", "어디까지",
    "ToM", "tom",
    # 자연어 진행 명령
    "한 번 돌려", "지금 가봐", "한 발", "에이전트 가",
    # 첨부/해제
    "켜줘", "켜봐", "호출", "열어", "꺼", "그만", "종료",
    # 목표 설정
    "목표로", "잡자", "잡아", "설정",
)


def has_tom_keyword(text: str) -> bool:
    """text 에 ToM 관련 키워드 1개라도 있으면 True. LLM 호출 사전 필터."""
    if not text:
        return False
    t = text.lower()
    for kw in _TOM_KEYWORDS:
        if kw.lower() in t:
            return True
    return False


_LLM_SYSTEM = """\
너는 EIDOS 의 ToM 에이전트 자연어 명령 분류기다.

사용자 입력을 다음 action 중 하나로 분류한 JSON 만 출력. 코드블록·설명 금지.

action 목록:
  goal_set         — 새 목표 설정 ('<X> 목표로 잡자', '목표 바꿔', '<X> 해보자')
                     args: {goal_text: "<목표 본문>"}
  goal_decompose   — 활성 목표 N개로 분해 ('분기별 4개 쪼개', '단계 잡아줘')
                     args: {decompose_n: N} (없으면 4)
  milestone_done   — 마일스톤 완료 표시 ('그 X 끝났어', 'Y 마무리됐어')
                     args: {done_query: "<X 매칭 키워드>"}
  tick_run         — 에이전트 1 tick 실행 ('한 번 돌려', '지금 가봐', '진행해')
                     args: {}
  status_query     — 진행 상태 요청 ('상태 어때', '어디까지 됐어', '진행률')
                     args: {}
  stage_attach     — 저장된 stage 첨부 ('X 에이전트 켜줘', 'X 스테이지 열어')
                     args: {stage_query: "<stage 이름>"}
  stage_detach     — 첨부 해제 ('에이전트 꺼줘', 'ToM 모드 종료')
                     args: {}
  none             — 위 어디에도 안 맞는 일반 chat
                     args: {}

판단 원칙:
- ToM 시스템 조작 의도가 명확하면 해당 action·confidence 0.85+
- 약간 모호하면 confidence 0.5~0.7 — 호출자가 무시 처리 (안전)
- 일반 잡담·정보 질문·코드 요청은 none·confidence 1.0

응답 JSON 스키마:
{
  "action": "<위 목록 중 하나>",
  "confidence": 0.0~1.0,
  "reason": "분류 근거 짧은 한 줄",
  "goal_text": "<goal_set 일 때 목표 본문 (없으면 빈)>",
  "decompose_n": <goal_decompose 일 때 정수 (없으면 0)>,
  "done_query": "<milestone_done 일 때 매칭 키워드 (없으면 빈)>",
  "stage_query": "<stage_attach 일 때 stage 이름 (없으면 빈)>"
}
"""


async def classify_nl_intent_async(
    user_text: str,
    has_attached_stage: bool = False,
    timeout_sec: float = 8.0,
) -> NlTomIntent:
    """자연어 입력을 ToM action 으로 분류.

    Args:
      user_text: 사용자 메시지 원문
      has_attached_stage: 현재 stage 첨부 상태 — 첨부 안 됐으면 일부 action 무시
      timeout_sec: LLM 호출 timeout (default 8s — 빠른 응답 필요)

    Returns: NlTomIntent. LLM 실패·키워드 없음·confidence 낮으면 action='none'.
    """
    result = NlTomIntent(action="none", confidence=0.0)
    if not user_text or not user_text.strip():
        return result

    # 사전 필터 — ToM 키워드 0개면 LLM 호출 skip
    if not has_tom_keyword(user_text):
        result.reason = "no ToM keyword"
        return result

    try:
        from llm_module import get_llm_response_async
    except Exception as e:
        result.reason = f"llm_module import fail: {e}"
        return result

    prompt = (
        f"[현재 stage 첨부 상태] {'YES' if has_attached_stage else 'NO (stage 없음)'}\n"
        f"[사용자 입력]\n{user_text.strip()}\n\n"
        "위 입력을 분류한 JSON 하나만 출력."
    )

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

    # JSON 파싱
    try:
        data = json.loads(raw_text)
        if not isinstance(data, dict):
            result.reason = "non-dict JSON"
            return result
    except Exception as e:
        result.reason = f"JSON 파싱 실패: {e}"
        return result

    result.raw = data
    action = str(data.get("action", "none")).strip()
    if action not in _ALLOWED_ACTIONS:
        result.reason = f"unknown action: {action}"
        return result

    try:
        confidence = float(data.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    result.action = action
    result.confidence = max(0.0, min(1.0, confidence))
    result.reason = str(data.get("reason", ""))[:200]

    # action 별 args 추출 — 안전 변환
    if action == "goal_set":
        result.goal_text = str(data.get("goal_text", "")).strip()[:300]
        if not result.goal_text:
            result.action = "none"
            result.reason = "goal_set 인데 goal_text 비어있음"
    elif action == "goal_decompose":
        try:
            n = int(data.get("decompose_n", 0) or 0)
        except Exception:
            n = 0
        result.decompose_n = max(0, min(12, n))
        # has_attached_stage 가 False 면 decompose 무의미 → 무시
        if not has_attached_stage:
            result.action = "none"
            result.reason = "stage 미첨부 — decompose 불가"
    elif action == "milestone_done":
        result.done_query = str(data.get("done_query", "")).strip()[:120]
        if not result.done_query or not has_attached_stage:
            result.action = "none"
            result.reason = "done_query 비거나 stage 미첨부"
    elif action == "tick_run":
        if not has_attached_stage:
            result.action = "none"
            result.reason = "stage 미첨부 — tick 불가"
    elif action == "status_query":
        if not has_attached_stage:
            result.action = "none"
            result.reason = "stage 미첨부 — status 불가"
    elif action == "stage_attach":
        result.stage_query = str(data.get("stage_query", "")).strip()[:120]
        if not result.stage_query:
            result.action = "none"
            result.reason = "stage_query 비어있음"
    elif action == "stage_detach":
        if not has_attached_stage:
            result.action = "none"
            result.reason = "stage 미첨부 — detach 무의미"

    return result


def classify_nl_intent_sync(
    user_text: str,
    has_attached_stage: bool = False,
) -> NlTomIntent:
    """동기 wrapper — Qt 슬롯 등 sync 컨텍스트용. 새 event loop 격리."""
    try:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                classify_nl_intent_async(user_text, has_attached_stage)
            )
        finally:
            loop.close()
    except Exception as e:
        result = NlTomIntent(action="none")
        result.reason = f"sync wrapper 실패: {e}"
        return result


# ── slash command 변환 helper — 분류 결과를 기존 try_handle_command 형식으로
def to_slash_command(intent: NlTomIntent) -> Optional[str]:
    """NlTomIntent 를 기존 try_handle_command 가 받는 slash 명령 문자열로 변환.

    None 반환 시 변환 불가 (action='none' 또는 args 부족).
    """
    if not intent.is_actionable():
        return None
    if intent.action == "goal_set":
        return f"/goal {intent.goal_text}"
    if intent.action == "goal_decompose":
        n = intent.decompose_n or 4
        return f"/decompose {n}"
    if intent.action == "milestone_done":
        return f"/done {intent.done_query}"
    if intent.action == "tick_run":
        return None  # /tick slash 없음 → 별도 처리 (run_stage_one_tick 직접 호출)
    if intent.action == "status_query":
        return "/goals"
    if intent.action == "stage_attach":
        return None  # 첨부는 별도 — chat에서는 햄버거 메뉴, 또는 음성모드
    if intent.action == "stage_detach":
        return None
    return None
