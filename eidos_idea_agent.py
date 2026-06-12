# eidos_idea_agent.py
# [2026-05-26 Phase 1] 사업 아이템 떠올리는 에이전트 — MVP 단발 생성기.
#
# Option C (proactive 발화) 의 첫 단계. Phase 1 은 단순 single-shot:
#   1. extract_profile_from_belief(belief) — ToM-core belief 에서 사용자 프로필 추출
#   2. generate_ideas_async(profile, n=5) — LLM 2회 호출 (brainstorm + 1-pager)
#   3. 채팅창에 5개 카드 + 1-pager 표시
#
# Phase 2 (영속화): eidos_idea_store.py · idea history
# Phase 3 (proactive): scheduler 통합 · 주기 자율 발화 · 피드백 학습

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Any


# ── 데이터 ─────────────────────────────────────────────────────────────
@dataclass
class UserProfile:
    """사업 아이템 생성 위한 사용자 프로필.

    belief 에서 가능한 한 자동 추출 + 사용자 수정 (Hybrid).
    """
    strengths: list[str] = field(default_factory=list)     # 강점·기술·도메인 경험
    interests: list[str] = field(default_factory=list)     # 관심 분야
    capital: str = "100만원~500만원"                       # 자본 범위 텍스트
    weekly_hours: int = 20                                 # 주 N시간 가용
    constraints: list[str] = field(default_factory=list)   # 제약 (예: "혼자 운영", "오프라인 X")
    recent_focus: str = ""                                  # 최근 자주 다루는 화제

    def serialize(self) -> dict:
        return asdict(self)

    def as_prompt_brief(self) -> str:
        """LLM prompt 에 넣을 짧은 요약."""
        lines = ["## 사용자 프로필"]
        if self.strengths:
            lines.append(f"- 강점: {', '.join(self.strengths[:6])}")
        if self.interests:
            lines.append(f"- 관심: {', '.join(self.interests[:6])}")
        lines.append(f"- 자본 범위: {self.capital}")
        lines.append(f"- 주 가용 시간: {self.weekly_hours}시간")
        if self.constraints:
            lines.append(f"- 제약: {', '.join(self.constraints[:4])}")
        if self.recent_focus:
            lines.append(f"- 최근 자주 다루는 화제: {self.recent_focus[:80]}")
        return "\n".join(lines)


@dataclass
class BusinessIdea:
    """하나의 사업 아이디어 + 1-pager."""
    title: str
    target_customer: str
    value_proposition: str
    revenue_model: str
    differentiation: str
    estimated_monthly_revenue: str        # "100~300만원" 같은 텍스트 범위
    feasibility: float = 0.5              # 0~1
    risk_factors: list[str] = field(default_factory=list)
    first_steps: list[str] = field(default_factory=list)   # 첫 7일 실행 step 3개

    def serialize(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BusinessIdea":
        if not isinstance(data, dict):
            data = {}
        try:
            feas = float(data.get("feasibility", 0.5))
        except Exception:
            feas = 0.5
        return cls(
            title=str(data.get("title", "(이름없음)"))[:60],
            target_customer=str(data.get("target_customer", ""))[:120],
            value_proposition=str(data.get("value_proposition", ""))[:200],
            revenue_model=str(data.get("revenue_model", ""))[:120],
            differentiation=str(data.get("differentiation", ""))[:200],
            estimated_monthly_revenue=str(data.get("estimated_monthly_revenue", ""))[:60],
            feasibility=max(0.0, min(1.0, feas)),
            risk_factors=[str(r)[:80] for r in (data.get("risk_factors") or [])[:4]],
            first_steps=[str(s)[:120] for s in (data.get("first_steps") or [])[:5]],
        )

    def as_card_markdown(self, idx: int) -> str:
        """채팅창에 표시할 카드 마크다운."""
        lines = [
            f"### 💡 아이디어 {idx}. **{self.title}**",
            f"- **타겟**: {self.target_customer}",
            f"- **가치 제안**: {self.value_proposition}",
            f"- **수익 모델**: {self.revenue_model}",
            f"- **차별화**: {self.differentiation}",
            f"- **예상 월매출**: {self.estimated_monthly_revenue} · 실행 난이도: {self._feas_emoji()}",
        ]
        if self.risk_factors:
            lines.append(f"- **리스크**: {' · '.join(self.risk_factors)}")
        if self.first_steps:
            lines.append("- **첫 7일 step**:")
            for i, s in enumerate(self.first_steps, 1):
                lines.append(f"  {i}. {s}")
        return "\n".join(lines)

    def _feas_emoji(self) -> str:
        if self.feasibility >= 0.7:
            return f"🟢 쉬움 ({self.feasibility:.1f})"
        if self.feasibility >= 0.4:
            return f"🟡 중간 ({self.feasibility:.1f})"
        return f"🔴 어려움 ({self.feasibility:.1f})"


# ── belief 에서 profile 추출 ─────────────────────────────────────────
def extract_profile_from_belief(belief, fallback_profile: Optional[UserProfile] = None) -> UserProfile:
    """ToM-core belief 에서 가능한 한 자동 추출. 명시 안 된 필드는 fallback 사용.

    Hybrid 입력 방식 — UI 에서 사용자가 form 으로 수정 후 확정.
    """
    base = fallback_profile or UserProfile()
    if belief is None:
        return base

    strengths = list(base.strengths)
    interests = list(base.interests)
    recent_focus = base.recent_focus

    try:
        # known_facts 에서 명시된 강점·관심
        kf = getattr(belief, "known_facts", {}) or {}
        if isinstance(kf, dict):
            _s = kf.get("strengths") or kf.get("skills")
            if isinstance(_s, str) and _s.strip():
                strengths.extend([t.strip() for t in _s.split(",") if t.strip()])
            elif isinstance(_s, list):
                strengths.extend([str(t).strip() for t in _s if str(t).strip()])

            _i = kf.get("interests") or kf.get("hobbies")
            if isinstance(_i, str) and _i.strip():
                interests.extend([t.strip() for t in _i.split(",") if t.strip()])
            elif isinstance(_i, list):
                interests.extend([str(t).strip() for t in _i if str(t).strip()])

        # recent_topics → interests 보강 (LRU)
        rt = getattr(belief, "recent_topics", []) or []
        if isinstance(rt, list) and rt:
            for t in rt[:5]:
                if isinstance(t, str) and t.strip() and t not in interests:
                    interests.append(t.strip())
            # recent_focus 는 가장 최근 화제
            recent_focus = recent_focus or (rt[0] if rt else "")

        # pending_threads 의 topic → 현재 focus
        pt = getattr(belief, "pending_threads", []) or []
        if isinstance(pt, list) and pt and not recent_focus:
            for raw in pt[:3]:
                if isinstance(raw, dict):
                    topic = raw.get("topic", "")
                    if topic:
                        recent_focus = topic
                        break
    except Exception as e:
        print(f"[idea_agent] belief 추출 실패 (graceful·fallback 사용): {e}")

    # dedup·cap
    strengths = list(dict.fromkeys(strengths))[:8]
    interests = list(dict.fromkeys(interests))[:8]

    return UserProfile(
        strengths=strengths,
        interests=interests,
        capital=base.capital,
        weekly_hours=base.weekly_hours,
        constraints=list(base.constraints),
        recent_focus=recent_focus[:120],
    )


# ── LLM 호출 helper ──────────────────────────────────────────────────
async def _call_llm_json(prompt: str, max_tokens: int = 4096) -> Any:
    """LLM 호출 → JSON parse. 실패 시 None."""
    try:
        from llm_module import get_llm_response_async
        resp = await get_llm_response_async(
            prompt,
            response_mime_type="application/json",
            max_tokens=max_tokens,
            use_cache=False,
        )
        if not resp:
            return None
        # JSON 추출 (fence 제거 등)
        text = resp.strip()
        if text.startswith("```"):
            # ```json ... ``` 또는 ``` ... ``` 처리
            text = text.split("```", 2)[1] if "```" in text[3:] else text
            if text.startswith("json"):
                text = text[4:].strip()
            text = text.rstrip("`").strip()
        return json.loads(text)
    except Exception as e:
        print(f"[idea_agent] LLM JSON parse 실패: {e}")
        return None


# ── 메인 generation ──────────────────────────────────────────────────
_BRAINSTORM_PROMPT_TEMPLATE = """당신은 사업 아이템 컨설턴트입니다. 다음 사용자 프로필에 *맞춤형* 사업 아이템 N개를 떠올려주세요.

{profile_brief}

## 요청
- 사용자 강점·관심을 잘 활용하는 아이템
- 주어진 자본·시간 안에서 시작 가능
- 한국 시장 현실에 맞춤 (글로벌 SaaS·해외 시장 X)
- 5개 모두 *서로 다른 angle* (예: 서비스·SaaS·콘텐츠·교육·자동화·중개)
- 너무 흔한 거 (블로그·유튜브) 보다는 사용자 강점을 살리는 niche

## 출력 형식 (JSON)
```json
{{
  "ideas": [
    {{
      "title": "아이템 이름 (간결)",
      "target_customer": "구체적 타겟 (예: 1인 사장·자영업·중고등 학부모 등)",
      "value_proposition": "이 사람한테 어떤 가치 (1~2줄)",
      "revenue_model": "구독·외주·광고·중개수수료 등",
      "differentiation": "왜 사용자가 이걸 잘 만들 수 있나 (강점 연결)",
      "estimated_monthly_revenue": "월 X~Y만원 (현실적 범위)",
      "feasibility": 0.0~1.0,
      "risk_factors": ["리스크 1", "리스크 2"],
      "first_steps": ["첫 7일 실행 step 1", "step 2", "step 3"]
    }},
    ... (총 N개)
  ]
}}
```

각 idea 의 differentiation 에 사용자의 *구체적 강점* 을 명시 연결하세요.
JSON 만 출력하세요 — 다른 설명 X.
"""


async def generate_ideas_async(profile: UserProfile, n: int = 5) -> list[BusinessIdea]:
    """LLM 으로 N 개 사업 아이템 + 1-pager 한 번에 생성.

    Phase 1 단발 호출 (brainstorm + 1-pager 통합 1회). LLM JSON 응답.

    Returns:
        BusinessIdea list. LLM 실패 시 빈 list (caller graceful 처리).
    """
    prompt = _BRAINSTORM_PROMPT_TEMPLATE.format(
        profile_brief=profile.as_prompt_brief(),
    ).replace("N개", f"{n}개")

    data = await _call_llm_json(prompt, max_tokens=4096)
    if not isinstance(data, dict):
        print(f"[idea_agent] LLM 응답이 dict 아님: {type(data)}")
        return []

    raw_ideas = data.get("ideas")
    if not isinstance(raw_ideas, list):
        print(f"[idea_agent] LLM 응답에 'ideas' 키 없음")
        return []

    out: list[BusinessIdea] = []
    for raw in raw_ideas[:n]:
        if isinstance(raw, dict):
            try:
                out.append(BusinessIdea.from_dict(raw))
            except Exception as e:
                print(f"[idea_agent] idea 파싱 실패 (skip): {e}")
                continue
    return out


# ── refine (Phase 1 minimal·후속 확장 예약) ───────────────────────────
_REFINE_PROMPT_TEMPLATE = """다음 사업 아이템을 사용자 피드백 반영해 개선해주세요.

## 기존 아이디어
{idea_brief}

## 사용자 피드백
"{feedback}"

## 출력 (JSON·기존 schema 동일)
```json
{{
  "title": "...", "target_customer": "...", "value_proposition": "...",
  "revenue_model": "...", "differentiation": "...",
  "estimated_monthly_revenue": "...", "feasibility": 0.0~1.0,
  "risk_factors": [...], "first_steps": [...]
}}
```
"""


async def refine_idea_async(idea: BusinessIdea, feedback: str) -> Optional[BusinessIdea]:
    """기존 idea 에 사용자 피드백 반영해 진화. Phase 1 후속 사용 가능 (옵션)."""
    if not feedback or not feedback.strip():
        return None
    idea_brief = json.dumps(idea.serialize(), ensure_ascii=False, indent=2)
    prompt = _REFINE_PROMPT_TEMPLATE.format(idea_brief=idea_brief, feedback=feedback.strip())
    data = await _call_llm_json(prompt, max_tokens=2048)
    if not isinstance(data, dict):
        return None
    try:
        return BusinessIdea.from_dict(data)
    except Exception as e:
        print(f"[idea_agent] refine 파싱 실패: {e}")
        return None


# ── 결과 채팅창 표시용 markdown ──────────────────────────────────────
def ideas_summary_markdown(ideas: list[BusinessIdea], profile: UserProfile) -> str:
    """5개 idea 를 채팅창에 표시할 통합 마크다운."""
    if not ideas:
        return "⚠️ 아이디어 생성 실패 — LLM 응답 비어있음·재시도 추천."
    lines = [
        f"## 💡 사용자 맞춤 사업 아이템 {len(ideas)}개",
        "",
        profile.as_prompt_brief(),
        "",
        "---",
        "",
    ]
    for i, idea in enumerate(ideas, 1):
        lines.append(idea.as_card_markdown(i))
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)
