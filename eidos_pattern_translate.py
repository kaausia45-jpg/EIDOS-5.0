# eidos_pattern_translate.py
# [Wave2-C 2026-05-28] 외부 ToM 패턴 → EIDOS 내부 용어 번역.
#
# 도메인 초월 이식의 핵심:
#   - actor 매핑 (외부 행위자 → EIDOS 객체)
#   - indicator 매핑 (외부 지표 → EIDOS 13 차원)
#   - LLM 자동 추론 + confidence 점수
#   - 낮은 confidence 는 텔레그램 강조 알림 (사용자 검증 가능)
#
# 디자인 원칙:
#   - LLM 호출 1회 (패턴 1개당)
#   - LLM caller inject (Wave2-B 와 동일 패턴)
#   - 매핑 실패해도 None 반환·이식 시도 안 함 (잘못된 매핑보다 안 함이 안전)
#   - 매핑 confidence 별도 추적 — Layer 2 미정계수 prior 에 곱해짐

from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import dataclass, asdict, field
from typing import Optional, Callable


# ── EIDOS 객체 정의 ────────────────────────────────────────────────────
EIDOS_OBJECTS = ("user", "EIDOS-self", "external_actor")

# EIDOS 13 + 7 지표 (eidos_self_indicators 와 일치 — 매핑 후보)
EIDOS_INDICATORS = (
    # 13 측정
    "comp_research", "comp_planning", "comp_development",
    "comp_marketing", "comp_comm", "comp_operations",
    "emo_valence", "emo_arousal", "emo_affection",
    "user_acceptance", "thread_resolve",
    "action_value", "pattern_accuracy",
    # 7 placeholder
    "work_efficiency", "productivity", "response_speed",
    "meta_cognition", "reasoning", "financial_mgmt", "resource_balance",
)


# ── TranslatedPattern ──────────────────────────────────────────────────
@dataclass
class TranslatedPattern:
    """ToMPattern 의 actor·indicator 가 EIDOS 용어로 번역된 결과.

    Wave2-D 가 이 객체에 미정계수 {α, β} 삽입 후 catalog 저장.
    """

    # 출처 (ToMPattern 에서 복사)
    source_url: str = ""
    source_type: str = ""
    source_title: str = ""
    source_weight: float = 0.5

    # 매핑 — LLM 추론 결과
    actor_mapping: dict = field(default_factory=dict)        # 외부 → EIDOS_OBJECTS
    indicator_mapping: dict = field(default_factory=dict)    # "외부.지표" → "EIDOS.지표"
    mapping_confidence: float = 0.5                          # 0~1 평균
    rationale: str = ""                                      # LLM 의 매핑 이유

    # 번역된 본문
    translated_transitions: list = field(default_factory=list)
    translated_patterns: list = field(default_factory=list)

    # 매핑 안 된 항목 (drop 됨)
    unmapped_actors: list = field(default_factory=list)
    unmapped_indicators: list = field(default_factory=list)

    # 메타
    translated_at: str = ""
    needs_user_review: bool = False  # confidence 낮으면 True

    def serialize(self) -> dict:
        return asdict(self)


# ── 프롬프트 ──────────────────────────────────────────────────────────
_TRANSLATE_PROMPT = """너는 도메인 초월 패턴 번역기다. 외부 ToM 패턴을 EIDOS 내부 용어로
*무조건* JSON 으로만 번역한다. 자연어 응답·해설 금지.

[입력 ToM 패턴]
출처: {source_type} — {source_title}
actors: {actors}
indicators: {indicators}
patterns: {patterns}

[EIDOS 객체 후보 — 외부 actor 를 이 중 하나로 매핑]
- "user": 사용자 본인 (EIDOS 가 응대·관찰하는 대상)
- "EIDOS-self": EIDOS 자기 자신
- "external_actor": 사용자/EIDOS 외부의 다른 행위자 (의뢰자·플랫폼·고객 등)

[EIDOS 13+7 지표 — 외부 actor 의 indicator 를 이 중 하나로 매핑]
{eidos_indicators}

[현재 활성 목표] {goal_indicator} — 이 지표가 매핑에서 우선 후보.

[출력 schema — 무조건 이 JSON 만]
{{
  "actor_mapping": {{
    "외부_actor명": "user|EIDOS-self|external_actor",
    ...
  }},
  "indicator_mapping": {{
    "외부_actor명.지표명": "user.comp_research" 또는 "EIDOS-self.emo_valence" 등,
    ...
  }},
  "mapping_confidence": 0.0 ~ 1.0 (평균 신뢰도),
  "rationale": "왜 이 매핑인지 한국어 1~2문장",
  "unmapped_actors": ["매핑 어려운 actor 가 있다면"],
  "unmapped_indicators": ["매핑 어려운 indicator 가 있다면"]
}}

[원칙]
1. 발신자·주도자·관찰 대상 → "user"
2. 수신자·응답자·자기 분석 주체 → "EIDOS-self"
3. 그 외 (의뢰자·고객·시스템) → "external_actor"
4. indicator 매핑은 *의미* 기준 — "trust" → emo_affection·"responsiveness" → response_speed·"reasoning" → reasoning
5. 매칭 안 되면 unmapped 에 넣어 — 억지 매핑 X
6. confidence 는 정직하게. 자료가 불명확하면 0.3~0.5, 명확하면 0.7~0.9.
"""


# ── 핵심 함수 ──────────────────────────────────────────────────────────
def _now() -> str:
    return _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"


# 매핑 confidence 가 이 값 미만이면 텔레그램 강조 (사용자 검증 권장)
_LOW_CONFIDENCE_THRESHOLD = 0.5


def translate_to_eidos_terms(
    pattern,
    llm_callable: Callable[[str], str],
    goal_indicator: str = "",
) -> Optional[TranslatedPattern]:
    """ToMPattern → TranslatedPattern.

    Args:
        pattern: ToMPattern (Wave2-B 출력)
        llm_callable: prompt → response (sync)
        goal_indicator: 현재 목표 지표 — LLM 매핑 시 prior

    Returns:
        TranslatedPattern 또는 None (LLM 실패·파싱 실패·매핑 부족).
    """
    if not pattern or not llm_callable:
        return None
    try:
        prompt = _TRANSLATE_PROMPT.format(
            source_type=pattern.source_type,
            source_title=(pattern.source_title or "")[:120],
            actors=json.dumps(pattern.actors, ensure_ascii=False),
            indicators=json.dumps(pattern.indicators, ensure_ascii=False),
            patterns=json.dumps(pattern.patterns[:5], ensure_ascii=False),
            eidos_indicators=json.dumps(list(EIDOS_INDICATORS), ensure_ascii=False),
            goal_indicator=goal_indicator or "(없음)",
        )

        raw = llm_callable(prompt)
        if not raw:
            print("[translate] LLM 빈 응답")
            return None

        data = _extract_json(raw)
        if not data:
            print(f"[translate] JSON 추출 실패 (head: {raw[:100]})")
            return None

        # actor_mapping 검증 — EIDOS_OBJECTS 만 허용
        actor_map = dict(data.get("actor_mapping") or {})
        valid_actor_map: dict = {}
        for ext_name, eidos_obj in actor_map.items():
            if str(eidos_obj) in EIDOS_OBJECTS:
                valid_actor_map[str(ext_name)] = str(eidos_obj)
            else:
                print(f"[translate] 잘못된 EIDOS object '{eidos_obj}' — skip")

        # indicator_mapping 검증 — EIDOS_INDICATORS 만 허용
        ind_map = dict(data.get("indicator_mapping") or {})
        valid_ind_map: dict = {}
        for ext_key, eidos_full in ind_map.items():
            # eidos_full 형식: "user.comp_research" 또는 "EIDOS-self.emo_valence"
            ind_match = re.match(r"^([\w-]+)\.([\w_]+)$", str(eidos_full))
            if not ind_match:
                continue
            eidos_obj, eidos_ind = ind_match.group(1), ind_match.group(2)
            if eidos_obj not in EIDOS_OBJECTS or eidos_ind not in EIDOS_INDICATORS:
                print(f"[translate] 잘못된 EIDOS indicator '{eidos_full}' — skip")
                continue
            valid_ind_map[str(ext_key)] = str(eidos_full)

        # 매핑 1개도 못 만들면 fail
        if not valid_actor_map:
            print("[translate] actor 매핑 0건 → fail")
            return None

        # confidence
        try:
            conf = float(data.get("mapping_confidence", 0.5))
        except Exception:
            conf = 0.5
        conf = max(0.0, min(1.0, conf))

        # transitions·patterns 번역 — 텍스트 치환
        trans_transitions = _translate_transitions(
            pattern.transitions, valid_actor_map, valid_ind_map)
        trans_patterns = _translate_patterns(
            pattern.patterns, valid_actor_map, valid_ind_map)

        out = TranslatedPattern(
            source_url=pattern.source_url,
            source_type=pattern.source_type,
            source_title=pattern.source_title,
            source_weight=pattern.source_weight,
            actor_mapping=valid_actor_map,
            indicator_mapping=valid_ind_map,
            mapping_confidence=conf,
            rationale=str(data.get("rationale") or "")[:300],
            translated_transitions=trans_transitions,
            translated_patterns=trans_patterns,
            unmapped_actors=list(data.get("unmapped_actors") or []),
            unmapped_indicators=list(data.get("unmapped_indicators") or []),
            translated_at=_now(),
            needs_user_review=conf < _LOW_CONFIDENCE_THRESHOLD,
        )

        return out
    except Exception as e:
        print(f"[translate] 실패 (graceful): {e}")
        return None


def _translate_transitions(
    transitions: list,
    actor_map: dict,
    indicator_map: dict,
) -> list:
    """transitions 의 actor·target 이름 치환. 매핑 안 된 transition 은 drop."""
    out = []
    for t in transitions:
        if not isinstance(t, dict):
            continue
        # trigger_actor 매핑
        ta = t.get("trigger_actor", "")
        if ta and ta not in actor_map:
            continue  # 매핑 안 된 actor → drop
        new_t = dict(t)
        new_t["trigger_actor"] = actor_map.get(ta, ta)
        # changes 안 target 매핑
        changes = []
        for c in (t.get("changes") or []):
            if not isinstance(c, dict):
                continue
            target = c.get("target", "")
            mapped = indicator_map.get(target)
            if mapped:
                new_c = dict(c)
                new_c["target"] = mapped
                changes.append(new_c)
            # 매핑 안 된 target 은 drop
        new_t["changes"] = changes
        if changes:  # 변화 1개 이상이어야 의미 있음
            out.append(new_t)
    return out


def _translate_patterns(
    patterns: list,
    actor_map: dict,
    indicator_map: dict,
) -> list:
    """patterns 의 자연어에서 actor·indicator 이름 치환.

    순서 중요: indicator 매핑이 actor.indicator 형식이라 actor 를 먼저 치환하면
    매핑 키가 깨짐 (예: "부서_B.responsiveness" → "EIDOS-self.responsiveness" 되어
    매핑 키 매칭 실패). 따라서 indicator 먼저, actor 나중.
    """
    out = []
    for p in patterns:
        if not isinstance(p, str):
            continue
        text = p
        # 1) indicator 먼저 — 긴 키 (actor.ind 형식·점 포함이라 가장 긴 매칭)
        for ext_key in sorted(indicator_map.keys(), key=len, reverse=True):
            text = text.replace(ext_key, indicator_map[ext_key])
        # 2) actor 그 다음 — 위에서 치환 안 된 actor 만 잡힘
        for ext_name in sorted(actor_map.keys(), key=len, reverse=True):
            text = text.replace(ext_name, actor_map[ext_name])
        out.append(text)
    return out


# ── JSON 추출 (decompose 와 동일 로직) ─────────────────────────────────
def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
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


# ── 텔레그램 요약 (사용자 검증 알림) ────────────────────────────────────
def summarize_translation_for_telegram(tp: TranslatedPattern) -> str:
    """번역 결과 → 텔레그램 markdown. confidence 낮으면 강조."""
    if not tp:
        return "❌ 번역 실패"
    header = "⚠️ *번역 — 검증 권장*" if tp.needs_user_review else "🔄 *EIDOS 용어 번역*"
    lines = [
        f"{header} (confidence={tp.mapping_confidence:.2f})",
        f"  📄 {tp.source_title[:80]}",
        "  *actor 매핑*:",
    ]
    for ext, eidos in list(tp.actor_mapping.items())[:5]:
        lines.append(f"    {ext} → {eidos}")
    if tp.indicator_mapping:
        lines.append("  *indicator 매핑*:")
        for ext, eidos in list(tp.indicator_mapping.items())[:5]:
            lines.append(f"    {ext} → {eidos}")
    if tp.rationale:
        lines.append(f"  💭 {tp.rationale[:200]}")
    if tp.translated_patterns:
        lines.append("  *번역된 규칙*:")
        for p in tp.translated_patterns[:3]:
            lines.append(f"    • {p[:120]}")
    if tp.needs_user_review:
        lines.append("\n  ❗ 매핑 신뢰도 낮음 — 잘못된 매핑이면 텔레그램으로 알려주세요.")
    return "\n".join(lines)
