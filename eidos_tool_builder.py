# eidos_tool_builder.py
# [Wave3-B/C 2026-05-28] 도구 자동 제작 — 4 타입 (DAG·prompt·data·code).
#
# Wave3-B (비실행 자산 — DAG·prompt·data): LLM 생성 → 자동 활성화 → 텔레그램 알림
# Wave3-C (실행 코드): LLM 생성 → pending_approval → 텔레그램 승인 요청
#
# 디자인 원칙:
#   - LLM 호출 1회 (도구 1개당)
#   - LLM caller inject (Wave2 와 동일 패턴)
#   - 도구 본문 max 토큰 제한
#   - 모든 단계 graceful — 실패 시 None 반환

from __future__ import annotations

import json
import re
from typing import Optional, Callable

import eidos_tool_catalog as tc


# ── 도구 타입 추론 — 목표·패턴 기반 ────────────────────────────────────
# 어떤 지표 강화엔 어떤 타입 도구가 효과적인가 휴리스틱.
_TYPE_RECOMMENDATIONS: dict[str, list[str]] = {
    # 도메인 신뢰도 — DAG (워크플로우) + prompt + data
    "comp_research":    ["dag", "prompt_template", "data"],
    "comp_planning":    ["dag", "prompt_template"],
    "comp_development": ["dag", "code"],         # 코드 도구 적합
    "comp_marketing":   ["prompt_template", "data"],
    "comp_comm":        ["prompt_template"],
    "comp_operations":  ["dag", "data"],
    # 감정 — prompt 위주 (톤 가이드)
    "emo_valence":      ["prompt_template"],
    "emo_arousal":      ["prompt_template"],
    "emo_affection":    ["prompt_template", "data"],
    # 관계
    "user_acceptance":  ["prompt_template"],
    "thread_resolve":   ["dag", "prompt_template"],
    "action_value":     ["dag"],
    "pattern_accuracy": ["dag", "data"],
    # 추가 차원
    "work_efficiency":  ["dag", "code"],
    "productivity":     ["dag", "code"],
    "response_speed":   ["prompt_template", "code"],
    "meta_cognition":   ["prompt_template", "data"],
    "reasoning":        ["prompt_template", "data"],
    "financial_mgmt":   ["data", "dag"],
    "resource_balance": ["dag", "data"],
}


def recommend_tool_types(target_indicator: str, max_types: int = 2) -> list[str]:
    """목표 지표 → 추천 도구 타입 list (휴리스틱). LLM 호출 0."""
    rec = _TYPE_RECOMMENDATIONS.get(target_indicator)
    if not rec:
        # fallback — 안전한 prompt + data
        return ["prompt_template", "data"]
    return rec[:max_types]


# ── LLM 프롬프트 (4 타입) ──────────────────────────────────────────────
_DAG_PROMPT = """너는 EIDOS PCA DAG 생성기다. 아래 목표 지표를 강화하기 위한 작은 워크플로우를
JSON 으로만 출력한다. 자연어 설명·해설 금지.

[목표 지표] {indicator}
[관련 이식 패턴] {patterns_brief}

[DAG schema — 무조건 이 JSON 만]
{{
  "name": "워크플로우_이름",
  "description": "한 줄 설명",
  "nodes": [
    {{"id": "n1", "type": "llm", "prompt": "...", "depends_on": []}},
    {{"id": "n2", "type": "tool", "tool_name": "...", "depends_on": ["n1"]}},
    ...
  ],
  "expected_outcome": "이 DAG 실행 시 기대 효과"
}}

[원칙]
1. nodes 2~5개 사이 (너무 복잡한 워크플로우 X).
2. type 은 "llm" (LLM 호출) 또는 "tool" (외부 도구).
3. depends_on 으로 의존성 표현 — DAG 순환 X.
4. 목표 지표와 직접 연결된 작은 작업만.
"""


_PROMPT_TEMPLATE_PROMPT = """너는 EIDOS prompt template 생성기다. 아래 목표 지표 강화에 활용할
prompt 템플릿을 JSON 으로만 출력. 자연어 해설 금지.

[목표 지표] {indicator}
[관련 이식 패턴] {patterns_brief}

[출력 schema]
{{
  "name": "템플릿_이름",
  "description": "어떤 상황에서 사용",
  "template": "실제 prompt 본문 — {{변수}} 형태로 변수 표시 가능",
  "variables": ["변수1", "변수2", ...],
  "usage_context": "EIDOS 가 어떤 tick·메시지에서 이 prompt 호출"
}}

[원칙]
1. template 본문 짧고 명확. 200~500자.
2. 변수는 {{}} 안에 명시 (예: {{user_name}}, {{topic}}).
3. usage_context — 언제 호출할지 EIDOS 가 자동 판단 가능 형태.
"""


_DATA_PROMPT = """너는 EIDOS data config 생성기다. 아래 목표 지표 강화에 도움 될 데이터를
JSON 으로만 출력. 자연어 해설 금지.

[목표 지표] {indicator}
[관련 이식 패턴] {patterns_brief}

[출력 schema]
{{
  "name": "데이터_이름",
  "description": "어떤 데이터·용도",
  "data_type": "keyword_list" 또는 "config" 또는 "sample",
  "data": [...] 또는 {{...}}  // 실제 데이터
}}

[원칙]
1. keyword_list 면 string array (10~30개).
2. config 면 key-value dict (EIDOS 설정 보강).
3. sample 면 사용자 행동·외부 행위자 예시 데이터.
4. 데이터만 — 코드 X·외부 호출 X.
"""


_CODE_PROMPT = """너는 EIDOS Python helper 생성기다. 아래 목표 지표 강화를 돕는 작은 함수를
JSON 으로만 출력. 자연어 해설은 description 필드 안에만.

[목표 지표] {indicator}
[관련 이식 패턴] {patterns_brief}

[출력 schema]
{{
  "name": "함수_이름",
  "description": "이 함수가 하는 일 + 안전성 (외부 호출 여부·파일 R/W 등)",
  "code": "def 함수명(...):\\n    \\\"\\\"\\\"docstring\\\"\\\"\\\"\\n    ...",
  "params": [{{"name": "x", "type": "str", "description": "..."}}],
  "returns": {{"type": "str", "description": "..."}},
  "safety_notes": "이 코드의 잠재적 위험 (sandbox 외 호출·시스템 접근 등)"
}}

[원칙]
1. 함수 단순·short — 30줄 이내.
2. 외부 네트워크 호출·os 모듈·subprocess 사용 절대 금지.
3. file write 도 금지 (read 만 — 안전한 경로).
4. 순수 함수 권장 — input → output.
5. safety_notes 정직하게 — "외부 영향 0·sandbox 안전" 이면 그대로 명시.
"""


# ── JSON 추출 (Wave2 와 동일) ──────────────────────────────────────────
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


# ── 관련 패턴 brief 생성 ──────────────────────────────────────────────
def _patterns_brief(related_pattern_ids: Optional[list]) -> str:
    """이식 패턴 list → 짧은 brief 문자열 (LLM prompt 에 inject)."""
    if not related_pattern_ids:
        return "(없음)"
    try:
        import eidos_pattern_catalog as pc
        briefs = []
        for pid in related_pattern_ids[:3]:
            p = pc.load_pattern(pid)
            if not p:
                continue
            patterns_text = " | ".join((p.patterns or [])[:2])
            briefs.append(f"- {p.source_type}: {patterns_text[:200]}")
        return "\n".join(briefs) if briefs else "(없음)"
    except Exception:
        return "(없음)"


# ── 4 타입 빌더 ────────────────────────────────────────────────────────
def build_dag_tool(
    target_indicator: str,
    related_pattern_ids: Optional[list],
    llm_callable: Callable[[str], str],
) -> Optional[tc.Tool]:
    """DAG 도구 생성. 즉시 active (비실행 자산)."""
    try:
        prompt = _DAG_PROMPT.format(
            indicator=target_indicator,
            patterns_brief=_patterns_brief(related_pattern_ids),
        )
        raw = llm_callable(prompt)
        if not raw:
            print("[builder] DAG LLM 빈 응답 — skip")
            return None
        data = _extract_json(raw)
        if not data or not data.get("nodes"):
            print(f"[builder] DAG JSON 추출/검증 실패 (head: {raw[:80]})")
            return None
        tool = tc.Tool(
            name=str(data.get("name", "auto_dag"))[:80],
            type="dag",
            description=str(data.get("description", ""))[:300],
            target_indicator=target_indicator,
            source="auto_generated",
            related_pattern_ids=list(related_pattern_ids or []),
            content=json.dumps(data, ensure_ascii=False),
            metadata={"node_count": len(data.get("nodes") or []),
                      "expected_outcome": str(data.get("expected_outcome", ""))[:200]},
        )
        tc.auto_activate_non_executable(tool)  # → active
        return tool
    except Exception as e:
        print(f"[builder] build_dag_tool 실패 (graceful): {e}")
        return None


def build_prompt_tool(
    target_indicator: str,
    related_pattern_ids: Optional[list],
    llm_callable: Callable[[str], str],
) -> Optional[tc.Tool]:
    """prompt template 도구. 즉시 active."""
    try:
        prompt = _PROMPT_TEMPLATE_PROMPT.format(
            indicator=target_indicator,
            patterns_brief=_patterns_brief(related_pattern_ids),
        )
        raw = llm_callable(prompt)
        if not raw:
            return None
        data = _extract_json(raw)
        if not data or not data.get("template"):
            return None
        tool = tc.Tool(
            name=str(data.get("name", "auto_prompt"))[:80],
            type="prompt_template",
            description=str(data.get("description", ""))[:300],
            target_indicator=target_indicator,
            source="auto_generated",
            related_pattern_ids=list(related_pattern_ids or []),
            content=str(data.get("template", ""))[:2000],
            metadata={
                "variables": list(data.get("variables") or []),
                "usage_context": str(data.get("usage_context", ""))[:300],
            },
        )
        tc.auto_activate_non_executable(tool)
        return tool
    except Exception as e:
        print(f"[builder] build_prompt_tool 실패 (graceful): {e}")
        return None


def build_data_tool(
    target_indicator: str,
    related_pattern_ids: Optional[list],
    llm_callable: Callable[[str], str],
) -> Optional[tc.Tool]:
    """data config 도구. 즉시 active."""
    try:
        prompt = _DATA_PROMPT.format(
            indicator=target_indicator,
            patterns_brief=_patterns_brief(related_pattern_ids),
        )
        raw = llm_callable(prompt)
        if not raw:
            return None
        data = _extract_json(raw)
        if not data or "data" not in data:
            return None
        tool = tc.Tool(
            name=str(data.get("name", "auto_data"))[:80],
            type="data",
            description=str(data.get("description", ""))[:300],
            target_indicator=target_indicator,
            source="auto_generated",
            related_pattern_ids=list(related_pattern_ids or []),
            content=json.dumps(data.get("data"), ensure_ascii=False),
            metadata={"data_type": str(data.get("data_type", "config"))},
        )
        tc.auto_activate_non_executable(tool)
        return tool
    except Exception as e:
        print(f"[builder] build_data_tool 실패 (graceful): {e}")
        return None


def build_code_tool(
    target_indicator: str,
    related_pattern_ids: Optional[list],
    llm_callable: Callable[[str], str],
) -> Optional[tc.Tool]:
    """code 도구. **pending_approval** 로 저장 — 사용자 승인 후 active.

    추가 안전망: 금지 키워드 (os.system·subprocess·eval·exec·socket 등) 검출 시
    즉시 reject 처리 (LLM 헛소리·악성 패턴 방지).
    """
    try:
        prompt = _CODE_PROMPT.format(
            indicator=target_indicator,
            patterns_brief=_patterns_brief(related_pattern_ids),
        )
        raw = llm_callable(prompt)
        if not raw:
            return None
        data = _extract_json(raw)
        if not data or not data.get("code"):
            return None

        code_text = str(data.get("code", ""))
        # 안전 검사 — 금지 키워드
        forbidden = (
            "os.system", "subprocess", "eval(", "exec(",
            "__import__", "open(", "socket", "requests.",
            "urllib.", "shutil.rmtree", "Popen", "compile(",
        )
        flagged = [kw for kw in forbidden if kw in code_text]

        tool = tc.Tool(
            name=str(data.get("name", "auto_code"))[:80],
            type="code",
            description=str(data.get("description", ""))[:300],
            target_indicator=target_indicator,
            source="auto_generated",
            related_pattern_ids=list(related_pattern_ids or []),
            content=code_text[:3000],
            metadata={
                "params": list(data.get("params") or []),
                "returns": dict(data.get("returns") or {}),
                "safety_notes": str(data.get("safety_notes", ""))[:300],
                "forbidden_keywords": flagged,
            },
        )

        # 금지 키워드 있으면 즉시 rejected
        if flagged:
            tool.status = "rejected"
            tool.rejected_at = tc._now()
            tool.abandoned_reason = f"forbidden_keywords: {flagged}"
            tc.save_tool(tool)
            print(f"[builder] code 도구 금지 키워드 — 자동 reject: {flagged}")
            return tool

        # 정상 — pending_approval 로
        tc.auto_activate_non_executable(tool)  # → pending_approval
        return tool
    except Exception as e:
        print(f"[builder] build_code_tool 실패 (graceful): {e}")
        return None


# ── 통합 builder — 목표에 맞는 도구 1~N개 자동 제작 ───────────────────
def build_tools_for_goal(
    goal,
    related_pattern_ids: Optional[list],
    llm_callable: Callable[[str], str],
    max_tools: int = 2,
) -> list[tc.Tool]:
    """목표에서 추천 타입 → 각 타입 1개씩 도구 제작.

    LLM 호출 max_tools 회. 실패한 타입은 skip — graceful.
    """
    out: list[tc.Tool] = []
    if not goal:
        return out
    indicator = getattr(goal, "target_indicator", "")
    if not indicator:
        return out

    recommended = recommend_tool_types(indicator, max_types=max_tools)
    print(f"[builder] {indicator} 권장 도구 타입: {recommended}")

    for tool_type in recommended[:max_tools]:
        tool = None
        if tool_type == "dag":
            tool = build_dag_tool(indicator, related_pattern_ids, llm_callable)
        elif tool_type == "prompt_template":
            tool = build_prompt_tool(indicator, related_pattern_ids, llm_callable)
        elif tool_type == "data":
            tool = build_data_tool(indicator, related_pattern_ids, llm_callable)
        elif tool_type == "code":
            tool = build_code_tool(indicator, related_pattern_ids, llm_callable)
        if tool:
            out.append(tool)
            print(f"[builder] {tool_type} 도구 제작 성공: {tool.id} (status={tool.status})")
        else:
            print(f"[builder] {tool_type} 도구 제작 실패 — skip")

    return out
