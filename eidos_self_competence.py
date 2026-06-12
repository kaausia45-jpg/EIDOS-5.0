# eidos_self_competence.py
# [2026-05-28 Phase 14] EIDOS 자기 능력 메타 인지 — 분야별 confidence + 위임 결정.
#
# Phase 1~13 의 ToM 은 *외부* actor 와 *사용자* mind 추론만 했음 (반쪽 ToM).
# Phase 14 가 EIDOS-self 의 mind (능력·한계·confidence) 도 추론 → 진짜 ToM.
#
# 결과:
#   1. action_id → 분야 (research/planning/development/marketing/comm/operations) 자동 태깅
#   2. PatternModel.ActionValue.avg_reward 의 분야별 평균 → DomainConfidence
#   3. 사용자 요청 텍스트 → 분야 자동 분류 (휴리스틱 키워드)
#   4. confidence 가 낮은 분야 요청 시 → delegate_to_user 권장 (LLM 결정 prompt 에 inject)
#   5. 메타 action 3개의 사양 정의 (eidos_action_registry 와 연동)
#
# 디자인 원칙 (Phase 1~13 동일):
#   - LLM 호출 0 (메커니즘 자체는 휴리스틱)
#   - graceful — PatternModel 없어도 fallback (모든 분야 0.5)
#   - 분야 confidence 는 read-only — Phase 12 의 ActionValue 가 single source
#   - sycophancy 방지 — confidence 낮은 분야는 LLM prompt 에 명시적으로 표시

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ── 분야 정의 ─────────────────────────────────────────────────────────
# 사용자가 명시한 6 분야 + 메타·observe 카테고리
DOMAINS = (
    "research",      # 시장조사·자료 수집·웹 탐색
    "planning",      # 기획·로드맵·milestone 분해
    "development",   # 개발·코드 작성·디버그
    "marketing",     # 마케팅·카피·SEO·SNS
    "comm",          # 고객대응·이메일·답변 초안
    "operations",    # 운영·결제·세금·계약·일정
    "meta",          # observe·no_op·reflect (자기 메타)
)

# 사용자 친화 이름 (prompt brief 용)
_DOMAIN_LABEL = {
    "research":    "시장조사",
    "planning":    "기획",
    "development": "개발",
    "marketing":   "마케팅",
    "comm":        "고객대응",
    "operations":  "운영",
    "meta":        "메타·관찰",
}

# ── action_id → 분야 태깅 ─────────────────────────────────────────────
# 알려진 action 별 명시 매핑. 알 수 없는 action 은 "meta" fallback.
# 한 action 이 여러 분야 걸칠 수 있음 (예: llm.write 는 planning + marketing + comm).
# 그 경우 list — confidence 계산 시 가중 분산.
ACTION_DOMAIN_MAP: dict[str, tuple] = {
    # 웹·자료 수집 → research
    "eidos.tool.web.navigate":      ("research",),
    "eidos.tool.web.click":         ("research",),
    "eidos.tool.web.click_xy":      ("research",),
    "eidos.tool.web.fill":          ("research",),
    "eidos.tool.web.scroll":        ("research",),
    "eidos.tool.web.read_page":     ("research",),
    "eidos.tool.web.go_back":       ("research",),
    "eidos.tool.web.wait":          ("research",),

    # LLM 글쓰기 → planning + marketing + comm 다 가능
    "eidos.tool.llm.write":         ("planning", "marketing", "comm"),

    # 답변 초안 → comm 중심
    "eidos.tool.draft_to_clipboard": ("comm", "marketing"),

    # 외부 메시지 → comm
    "eidos.tool.message.telegram":  ("comm",),
    "eidos.tool.message.email":     ("comm",),

    # 사용자 질문 → meta (위임/확인이라 운영적)
    "eidos.tool.ask_user":          ("meta",),

    # 개발 → development
    "eidos.tool.dev.scaffold_project": ("development",),
    "eidos.tool.dev.write_code":       ("development",),
    "eidos.tool.dev.run_tests":        ("development",),
    "eidos.tool.dev.package_deliverable": ("development",),

    # PCA DAG 위임 → planning + 부하 분야 (DAG 별 metadata 의 tags 봐야 정확·기본 planning)
    "eidos.tool.pca.run_dag":          ("planning",),

    # 메타
    "eidos.meta.observe":           ("meta",),
    "eidos.meta.no_op":             ("meta",),
    "eidos.meta.reflect":           ("meta",),

    # Phase 14 신규 메타 action 들 (자기 인지)
    "eidos.meta.delegate_to_user":  ("meta",),
    "eidos.meta.counter_propose":   ("meta",),
    "eidos.meta.clarify_question":  ("meta",),
}


def get_action_domains(action_id: str) -> tuple:
    """action_id 의 분야 list. unknown 이면 ("meta",) fallback."""
    # PCA DAG suffix (eidos.tool.pca.run_dag/qa_check 등) 처리
    base = action_id.split("/", 1)[0]
    return ACTION_DOMAIN_MAP.get(base, ("meta",))


# ── DomainConfidence ──────────────────────────────────────────────────
# 학습 데이터 부족 시 default (모든 분야 0.5)
_DEFAULT_CONFIDENCE = 0.5
_MIN_ATTEMPTS_FOR_CONFIDENCE = 3   # 이만큼 시도 안 한 분야는 default 유지
_DELEGATE_THRESHOLD = 0.4          # confidence 이 미만이면 delegate 권장
_HIGH_CONFIDENCE_THRESHOLD = 0.7   # 이 이상이면 EIDOS 가 잘함 (역제안 가능 영역)


@dataclass
class DomainConfidence:
    """1 분야의 누적 통계 + confidence.

    - n_attempts: 이 분야 action 시도 총 횟수
    - sum_reward: 누적 avg_reward (ActionValue.avg_reward 합)
    - confidence: sum_reward / n_attempts (0~1)·n_attempts 부족 시 0.5
    - explicit_user_signal: 사용자가 명시적으로 "이 분야 약함" 라벨링 시 강제 (옵션)
    """

    domain: str
    n_attempts: float = 0.0   # float — 가중치 분배 (1 action / N 분야)
    sum_reward: float = 0.0
    confidence: float = _DEFAULT_CONFIDENCE
    explicit_user_signal: Optional[float] = None   # None 이면 자동 계산

    def is_weak(self) -> bool:
        return self.confidence < _DELEGATE_THRESHOLD

    def is_strong(self) -> bool:
        return self.confidence >= _HIGH_CONFIDENCE_THRESHOLD


def compute_domain_confidence(action_values: dict) -> dict[str, DomainConfidence]:
    """ActionValue dict (PatternModel.action_values) → 분야별 DomainConfidence.

    각 action 의 avg_reward 를 분야별로 가중 분배 (1 action 이 N 분야면 1/N 씩).
    n_attempts < _MIN_ATTEMPTS_FOR_CONFIDENCE 인 분야는 default 0.5 유지.

    graceful — action_values 비어있으면 모든 분야 default.
    """
    out: dict[str, DomainConfidence] = {
        d: DomainConfidence(domain=d) for d in DOMAINS
    }
    if not action_values:
        return out

    for aid, av in action_values.items():
        try:
            n = int(getattr(av, "n_attempts", 0))
            r = float(getattr(av, "avg_reward", 0.0))
        except Exception:
            continue
        if n <= 0:
            continue
        domains = get_action_domains(aid)
        if not domains:
            continue
        # 분야별 가중 (1 action / N 분야)
        weight_n = n / len(domains)
        weight_r = r * weight_n   # 가중 평균을 위한 누적
        for d in domains:
            if d not in out:
                continue
            out[d].n_attempts += weight_n
            out[d].sum_reward += weight_r

    for d, dc in out.items():
        if dc.n_attempts >= _MIN_ATTEMPTS_FOR_CONFIDENCE:
            dc.confidence = max(0.0, min(1.0, dc.sum_reward / dc.n_attempts))
    return out


# ── 사용자 요청 → 분야 분류 (휴리스틱) ───────────────────────────────
# 키워드 매칭 — LLM 호출 0·간단·확장 가능.
_DOMAIN_KEYWORDS = {
    "research": (
        "조사", "검색", "찾아", "분석", "리서치", "경쟁사", "시장", "트렌드",
        "통계", "데이터", "research", "search", "study", "analysis",
    ),
    "planning": (
        "기획", "계획", "전략", "로드맵", "milestone", "분해", "구조", "설계",
        "정리", "방향", "plan", "strategy", "roadmap",
    ),
    "development": (
        "개발", "코드", "코딩", "버그", "디버그", "구현", "함수", "테스트",
        "리팩터", "리팩토링", "fix", "implement", "build", "deploy", "API",
        "스크립트",
    ),
    "marketing": (
        "마케팅", "홍보", "광고", "SEO", "콘텐츠", "블로그", "SNS", "포스트",
        "카피", "캠페인", "marketing", "ad", "campaign",
    ),
    "comm": (
        "답변", "메일", "이메일", "문의", "응대", "회신", "톡", "메시지",
        "고객", "의뢰자", "답장", "reply", "email", "message",
    ),
    "operations": (
        "결제", "세금", "계약", "정산", "회계", "법무", "인보이스", "송금",
        "수입", "지출", "비용", "payment", "tax", "contract", "invoice",
        "schedule", "일정",
    ),
}


def classify_request_domain(text: str) -> str:
    """사용자 요청 텍스트 → 가장 매칭 키워드 많은 분야.

    매칭 0 이면 "meta" (분류 불가·일반 요청).
    여러 분야 매칭 시 가장 많은 분야·동률이면 dict 순서 (research·planning 우선).
    """
    if not text or not text.strip():
        return "meta"
    text_low = text.lower()
    text_orig = text
    scores: dict[str, int] = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        cnt = 0
        for kw in keywords:
            if kw in text_orig or kw.lower() in text_low:
                cnt += 1
        if cnt > 0:
            scores[domain] = cnt
    if not scores:
        return "meta"
    return max(scores, key=lambda d: scores[d])


# ── 위임 결정 ─────────────────────────────────────────────────────────
def should_delegate(
    user_request: str,
    confidence_map: dict[str, DomainConfidence],
    *,
    threshold: float = _DELEGATE_THRESHOLD,
) -> tuple[bool, str, float]:
    """사용자 요청 → (위임 권장 여부·해당 분야·confidence).

    Returns:
        (delegate, domain, confidence)
        delegate=True 면 LLM prompt 에 "이 작업 위임 권장" 안내 추가.
    """
    domain = classify_request_domain(user_request)
    if domain == "meta":
        return (False, domain, 1.0)
    dc = confidence_map.get(domain)
    if dc is None:
        return (False, domain, _DEFAULT_CONFIDENCE)
    return (dc.confidence < threshold, domain, dc.confidence)


# ── prompt brief — LLM 에게 자기 한계 알려주기 ────────────────────────
def as_prompt_brief(
    confidence_map: dict[str, DomainConfidence],
    *,
    user_request: Optional[str] = None,
) -> str:
    """LLM prompt 에 inject 할 markdown — 분야별 confidence + 약한/강한 분야 명시.

    user_request 주어지면 "이 요청은 X 분야·confidence Y" 추가 표시.
    """
    if not confidence_map:
        return ""
    lines = ["## 🧠 자기 능력 평가 (Phase 14 — 메타 인지)"]
    weak_domains: list[str] = []
    strong_domains: list[str] = []
    for d in DOMAINS:
        if d == "meta":
            continue
        dc = confidence_map.get(d)
        if dc is None:
            continue
        label = _DOMAIN_LABEL.get(d, d)
        if dc.n_attempts < _MIN_ATTEMPTS_FOR_CONFIDENCE:
            tag = " (데이터 부족·default)"
        else:
            tag = ""
        bar = "█" * int(dc.confidence * 10) + "░" * (10 - int(dc.confidence * 10))
        lines.append(
            f"- {label}: {bar} {dc.confidence:.2f} "
            f"(n={dc.n_attempts}{tag})"
        )
        if dc.is_weak() and dc.n_attempts >= _MIN_ATTEMPTS_FOR_CONFIDENCE:
            weak_domains.append(label)
        elif dc.is_strong() and dc.n_attempts >= _MIN_ATTEMPTS_FOR_CONFIDENCE:
            strong_domains.append(label)

    if weak_domains:
        lines.append(
            f"\n⚠️ **약한 분야**: {', '.join(weak_domains)} — "
            f"이 분야 요청 시 `eidos.meta.delegate_to_user` 우선 고려."
        )
    if strong_domains:
        lines.append(
            f"✅ **강한 분야**: {', '.join(strong_domains)} — 자신 있게 처리."
        )

    if user_request:
        delegate, domain, conf = should_delegate(user_request, confidence_map)
        label = _DOMAIN_LABEL.get(domain, domain)
        flag = "🚩 위임 권장" if delegate else "✓ 처리 가능"
        lines.append(
            f"\n**현재 요청 분야**: {label} (confidence {conf:.2f}) — {flag}"
        )
    return "\n".join(lines)


# ── 메타 action 사양 (Phase 14-B 에서 등록) ───────────────────────────
# 본 모듈에서는 정의만 — agent_dispatch / repertoire 가 import.
META_ACTIONS = {
    "eidos.meta.delegate_to_user": {
        "purpose": "이 작업은 사용자가 직접 처리하는 게 효율적·EIDOS 능력 부족 명시 위임",
        "args_schema": {
            "reason": "왜 사용자가 직접 해야 하는지 (1~2 문장)",
            "what_user_should_do": "사용자가 구체적으로 해야 할 일",
            "what_eidos_can_help_with": "이 작업 후 EIDOS 가 도울 수 있는 후속 부분 (옵션)",
        },
        "external_effect": False,
    },
    "eidos.meta.counter_propose": {
        "purpose": "사용자 방식이 비효율적이면 완곡 거부 + 더 나은 방향 제안",
        "args_schema": {
            "user_intent": "사용자가 원래 하려던 것 요약",
            "concern": "왜 비효율적인지 (구체 근거)",
            "alternative": "더 효율적인 대안 (구체 단계)",
            "fallback": "사용자가 원래대로 고집하면 따를 의향 (true/false)",
        },
        "external_effect": False,
    },
    "eidos.meta.clarify_question": {
        "purpose": "요청 모호·정보 부족 시 진짜 의도 파악을 위한 역질문 (기존 ask_user 강화)",
        "args_schema": {
            "what_unclear": "어떤 부분이 모호한지",
            "questions": "구체적 질문 1~3개 (list)",
            "default_assumption": "사용자 무응답 시 가정할 default (옵션)",
        },
        "external_effect": False,
    },
}


def summary_for_log(confidence_map: dict[str, DomainConfidence]) -> dict:
    """진단용 dict."""
    return {
        "domains": {
            d: {
                "n_attempts": dc.n_attempts,
                "confidence": round(dc.confidence, 3),
                "weak": dc.is_weak(),
                "strong": dc.is_strong(),
            }
            for d, dc in confidence_map.items()
        },
        "weak_domains": [d for d, dc in confidence_map.items() if dc.is_weak()],
        "strong_domains": [d for d, dc in confidence_map.items() if dc.is_strong()],
    }


__all__ = [
    "DOMAINS", "ACTION_DOMAIN_MAP", "META_ACTIONS",
    "DomainConfidence",
    "get_action_domains", "compute_domain_confidence",
    "classify_request_domain", "should_delegate",
    "as_prompt_brief", "summary_for_log",
]
