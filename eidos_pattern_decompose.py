# eidos_pattern_decompose.py
# [Wave2-B 2026-05-28] 외부 자연어 자료 → ToM 형식 JSON 구조화.
#
# 사용자 핵심 결정:
#   - LLM 의 자연어 요약 절대 금지 — 무조건 ToM 형식 (actors·indicators·
#     repertoire·transitions·patterns) JSON 으로 분해.
#   - 이게 도메인 초월 이식의 1단계. 형식 일치 안 되면 이식 불가능.
#
# 디자인 원칙:
#   - LLM 호출 1회 (자료 1개당)
#   - LLM caller inject 패턴 — sync wrapper 는 호출자가 책임
#   - 응답 파싱·validate·graceful (LLM 헛소리해도 None 반환)
#   - 출력 schema 가 기존 PatternModel 의 transitions 와 호환

from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import dataclass, asdict, field
from typing import Optional, Callable


# ── ToM 패턴 schema ────────────────────────────────────────────────────
@dataclass
class ToMPattern:
    """외부 자료를 ToM 형식으로 분해한 결과.

    actors / indicators / repertoire / transitions / patterns 5 부분.
    PatternModel 의 transitions schema 와 호환 (직접 이식 가능).
    """

    # 출처 메타
    source_url: str = ""
    source_type: str = ""              # arxiv / wikipedia / duckduckgo
    source_title: str = ""
    source_weight: float = 0.5         # 출처 신뢰도 (이식 시 미정계수 prior 곱셈)

    # ToM 5 요소
    actors: list = field(default_factory=list)
    indicators: dict = field(default_factory=dict)      # actor → list of indicator name
    repertoire: dict = field(default_factory=dict)      # actor → list of action
    transitions: list = field(default_factory=list)     # list of {trigger, actor, changes}
    patterns: list = field(default_factory=list)        # list of natural-language rule

    # 분해 메타
    decomposed_at: str = ""
    llm_raw: str = ""                  # debug 용 LLM 원본 응답 (수십자만 저장)
    validation_errors: list = field(default_factory=list)

    def serialize(self) -> dict:
        return asdict(self)

    @classmethod
    def deserialize(cls, data: dict) -> "ToMPattern":
        if not isinstance(data, dict):
            data = {}
        out = cls()
        out.source_url = str(data.get("source_url") or "")
        out.source_type = str(data.get("source_type") or "")
        out.source_title = str(data.get("source_title") or "")
        try:
            out.source_weight = float(data.get("source_weight", 0.5))
        except Exception:
            out.source_weight = 0.5
        out.actors = list(data.get("actors") or [])
        out.indicators = dict(data.get("indicators") or {})
        out.repertoire = dict(data.get("repertoire") or {})
        out.transitions = list(data.get("transitions") or [])
        out.patterns = list(data.get("patterns") or [])
        out.decomposed_at = str(data.get("decomposed_at") or "")
        out.llm_raw = str(data.get("llm_raw") or "")
        out.validation_errors = list(data.get("validation_errors") or [])
        return out


# ── 프롬프트 ──────────────────────────────────────────────────────────
_TOM_DECOMPOSE_PROMPT = """너는 ToM 분해 엔진이다. 외부 자료 (논문·기사·백과 등) 를 받으면
*무조건* 아래 JSON schema 로만 응답한다. 자연어 요약·해설 절대 금지.

[입력 자료]
title: {title}
source: {source_type}
content: {content}

[목표 지표]
{goal_indicator} — 이 지표 강화를 위해 자료에서 패턴을 추출한다.

[출력 schema — 무조건 이 JSON 만, 다른 text 금지]
{{
  "actors": ["행위자_A", "행위자_B", ...],
  "indicators": {{
    "행위자_A": ["지표명1", "지표명2", ...],
    "행위자_B": [...]
  }},
  "repertoire": {{
    "행위자_A": ["가능_액션1", "가능_액션2", ...],
    "행위자_B": [...]
  }},
  "transitions": [
    {{
      "trigger_actor": "행위자_A",
      "trigger_action": "액션명",
      "changes": [
        {{"target": "행위자_B.지표명", "delta": "+0.1"}},
        {{"target": "행위자_A.지표명", "delta": "-0.05"}}
      ],
      "conditions": ["선행 조건 (있으면)"],
      "source_quote": "자료 원문 30자 인용"
    }}
  ],
  "patterns": [
    "IF 조건 THEN 결과 — 한 줄 규칙 (한국어 OK)",
    ...
  ]
}}

[원칙]
1. actors 는 자료에 명시된 *주체* (사람·조직·시스템). 2~5개.
2. indicators 는 actor 의 측정 가능한 *속성*. 추상명 OK (예: trust_level·responsiveness).
3. repertoire 는 actor 가 *취할 수 있는 액션* (동사 형태).
4. transitions 는 자료에서 *명시·암시된 인과 관계*. 추측 X — 자료 근거 있는 것만.
5. patterns 는 transitions 를 사람이 읽을 수 있는 규칙으로. "IF ~ THEN ~" 형식 강제.
6. 자료가 빈약하면 transitions 0~1개·patterns 0~2개로 짧게. 억지로 채우지 마.
7. *다른 자연어 텍스트 금지* — JSON 만.
"""


# ── 핵심 함수 ──────────────────────────────────────────────────────────
def _now() -> str:
    return _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def decompose_to_tom(
    research_result,
    llm_callable: Callable[[str], str],
    goal_indicator: str = "",
    max_content_chars: int = 2000,
) -> Optional[ToMPattern]:
    """외부 자료 → ToM JSON 분해.

    Args:
        research_result: ResearchResult (eidos_external_research)
        llm_callable: prompt str → response str (sync). 호출자가 sync wrapper 책임.
        goal_indicator: 현재 활성 목표의 지표명 (분해 방향성)
        max_content_chars: 자료 본문 max 글자 수

    Returns:
        ToMPattern 또는 None (LLM 실패·파싱 실패·validate 실패).
    """
    if not research_result or not llm_callable:
        return None
    try:
        content = (research_result.summary or "")[:max_content_chars]
        if len(content) < 50:
            print(f"[decompose] 자료 본문 너무 짧음 ({len(content)} chars) — skip")
            return None

        prompt = _TOM_DECOMPOSE_PROMPT.format(
            title=research_result.title[:200],
            source_type=research_result.source,
            content=content,
            goal_indicator=goal_indicator or "(없음)",
        )

        # LLM 호출
        raw_response = llm_callable(prompt)
        if not raw_response:
            print("[decompose] LLM 빈 응답 — skip")
            return None

        # JSON 추출
        data = _extract_json(raw_response)
        if not data:
            print(f"[decompose] LLM 응답에서 JSON 추출 실패 (head: {raw_response[:100]})")
            return None

        # ToMPattern 빌드
        out = ToMPattern(
            source_url=research_result.url,
            source_type=research_result.source,
            source_title=research_result.title[:200],
            source_weight=float(getattr(research_result, "source_weight", 0.5)),
            actors=list(data.get("actors") or []),
            indicators=dict(data.get("indicators") or {}),
            repertoire=dict(data.get("repertoire") or {}),
            transitions=list(data.get("transitions") or []),
            patterns=list(data.get("patterns") or []),
            decomposed_at=_now(),
            llm_raw=raw_response[:200],
        )

        # validate
        errors = validate(out)
        out.validation_errors = errors
        if any(e.startswith("FATAL") for e in errors):
            print(f"[decompose] FATAL validation 실패: {errors}")
            return None
        if errors:
            print(f"[decompose] validation 경고 (non-fatal): {errors[:3]}")

        return out
    except Exception as e:
        print(f"[decompose] 실패 (graceful): {e}")
        return None


# ── JSON 추출 — LLM 응답에서 가장 큰 {...} 블록 ────────────────────────
def _extract_json(text: str) -> Optional[dict]:
    """LLM 응답에서 JSON 객체 추출. 응답이 마크다운 코드블록 안 또는
    바로 JSON 시작 모두 처리.
    """
    if not text:
        return None
    # 1. 마크다운 코드블록 ```json ... ``` 먼저
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # 2. 가장 큰 균형 잡힌 {...} 블록
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except Exception:
                    break
    return None


# ── validate ──────────────────────────────────────────────────────────
def validate(pattern: ToMPattern) -> list[str]:
    """ToMPattern schema 검증. error list 반환 (빈 = OK).

    FATAL prefix 는 이식 불가 — 호출자가 None 처리.
    경고 (non-FATAL) 는 ToMPattern.validation_errors 에 기록 후 이식 진행.
    """
    errors: list[str] = []
    try:
        if not pattern.actors:
            errors.append("FATAL: actors 비어있음")
        if not isinstance(pattern.actors, list):
            errors.append("FATAL: actors 가 list 아님")

        # indicators 의 키가 actors 부분집합인지 (loose check)
        for actor_name in pattern.indicators.keys():
            if actor_name not in pattern.actors:
                errors.append(f"WARN: indicators 의 actor '{actor_name}' 가 actors 에 없음")

        # repertoire 도 마찬가지
        for actor_name in pattern.repertoire.keys():
            if actor_name not in pattern.actors:
                errors.append(f"WARN: repertoire 의 actor '{actor_name}' 가 actors 에 없음")

        # transitions 안 trigger_actor 검증
        for i, t in enumerate(pattern.transitions):
            if not isinstance(t, dict):
                errors.append(f"WARN: transition {i} 가 dict 아님")
                continue
            ta = t.get("trigger_actor")
            if ta and ta not in pattern.actors:
                errors.append(f"WARN: transition {i} trigger_actor '{ta}' unknown")
            changes = t.get("changes") or []
            if not isinstance(changes, list):
                errors.append(f"WARN: transition {i} changes 가 list 아님")

        # transitions·patterns 둘 다 비면 사실상 빈 패턴 — 이식 의미 X
        if not pattern.transitions and not pattern.patterns:
            errors.append("FATAL: transitions 와 patterns 모두 빈 — 이식 불가")
    except Exception as e:
        errors.append(f"FATAL: validate 자체 실패 {e}")
    return errors


# ── 텔레그램 요약 ──────────────────────────────────────────────────────
def summarize_pattern_for_telegram(pattern: ToMPattern) -> str:
    """ToMPattern → 텔레그램 markdown."""
    if not pattern:
        return "❌ 분해 실패"
    parts = [
        f"🧩 *ToM 분해* — {pattern.source_type} (w={pattern.source_weight:.1f})",
        f"  📄 {pattern.source_title[:80]}",
        f"  🎭 actors ({len(pattern.actors)}): {', '.join(pattern.actors[:4])}",
        f"  📊 transitions: {len(pattern.transitions)}",
        f"  📐 patterns: {len(pattern.patterns)}",
    ]
    if pattern.patterns:
        parts.append("  주요 규칙:")
        for p in pattern.patterns[:3]:
            parts.append(f"    • {str(p)[:120]}")
    if pattern.validation_errors:
        warns = [e for e in pattern.validation_errors if not e.startswith("FATAL")]
        if warns:
            parts.append(f"  ⚠️ validation: {len(warns)}건")
    return "\n".join(parts)
