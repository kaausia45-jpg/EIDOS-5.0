# eidos_self_policy.py
# [Phase 1 2026-05-30] self-policy 컴파일러.
#
# 유추전이 비전의 정점 (Phase 1): self-applicable 패턴(외부 도메인에서 이식돼
# 출력이 EIDOS-self 자신인 f())을 EIDOS 실행정책으로 *컴파일*한다.
#
# 입력: self-applicable·현재 목표 매칭 ImportedPattern 들 (Phase 0 의
#       list_self_applicable_patterns 가 공급).
# 출력: SelfPolicy {force_actions, suppress_actions, action_strength, strength}.
#       Phase 2 의 tick 강제 훅이 이걸 받아 repertoire 를 gate / decision override.
#
# 핵심 로직 (사용자 비전 충실):
#   각 transition: trigger_actor=EIDOS-self 가 trigger_action 을 하면
#                  change.target(=EIDOS-self.<지표>) 가 {alpha_N} 만큼 변한다.
#   - alpha 의 *부호* = 그 트리거가 목표지표를 올리나(+)/내리나(−).
#   - trust = parameter_weights(EMA 학습 신뢰도) — 강도 스칼라.
#   - trigger_action(추상 동사 문자열) → EIDOS 실행기능에 *접지*(grounder).
#       grounder 는 {favor: 할 액션, suppress: 피할 액션} 반환.
#   - 합성: action_strength[a in favor] += sign(alpha)*trust
#           action_strength[a in suppress] -= sign(alpha)*trust
#       (alpha<0 이면 favor/suppress 가 뒤집힘 — "이 트리거는 오히려 해롭다".)
#   - 순강도 > 0 → force_actions, < 0 → suppress_actions.
#
# grounder 는 *주입 가능* — 기본은 결정론적 키워드 매핑(테스트·오프라인),
# Phase 2 가 LLM grounder 를 끼워 임의 트리거도 정확히 접지하게 한다.

from dataclasses import dataclass, field
from typing import Callable, Optional
import re


# ── EIDOS 실행기능 분류 (action_id) ──────────────────────────────────────
_IDLE_SET = (
    "eidos.meta.observe", "eidos.meta.no_op", "eidos.meta.reflect",
)
_WRITE_SET = (
    "eidos.tool.llm.write", "eidos.tool.draft_to_clipboard",
)
_CODE_SET = (
    "eidos.tool.dev.write_code", "eidos.tool.dev.scaffold_project",
    "eidos.tool.dev.run_tests", "eidos.tool.dev.package_deliverable",
)
_RESEARCH_SET = (
    "eidos.tool.web.navigate", "eidos.tool.web.read_page",
    "eidos.tool.web.scroll",
)
_PRODUCTIVE_SET = _WRITE_SET + _CODE_SET + _RESEARCH_SET + ("eidos.tool.pca.run_dag",)


# ── SelfPolicy ───────────────────────────────────────────────────────────
@dataclass
class SelfPolicy:
    """self-applicable 패턴들을 컴파일한 EIDOS 실행정책."""
    force_actions: list = field(default_factory=list)     # 우선/강제 action_id (강도 내림차순)
    suppress_actions: list = field(default_factory=list)  # 메뉴서 제거 action_id
    action_strength: dict = field(default_factory=dict)   # action_id → 순강도(부호 있음)
    source_pattern_ids: list = field(default_factory=list)
    rationale: list = field(default_factory=list)
    strength: float = 0.0                                 # 전체 정책 신뢰도 (max trust)

    def is_empty(self) -> bool:
        return not self.force_actions and not self.suppress_actions

    def serialize(self) -> dict:
        return {
            "force_actions": list(self.force_actions),
            "suppress_actions": list(self.suppress_actions),
            "action_strength": {k: round(v, 4) for k, v in self.action_strength.items()},
            "source_pattern_ids": list(self.source_pattern_ids),
            "rationale": list(self.rationale),
            "strength": round(self.strength, 3),
        }


# ── 기본 키워드 grounder ──────────────────────────────────────────────────
# (키워드들, favor=이 트리거 실행하려면 DO 할 액션, suppress=AVOID 할 액션)
_GROUND_RULES = (
    # 방해 차단 → idle(EIDOS 의 산만함) 회피
    (("distraction", "방해", "noise", "interference", "차단", "block",
      "제거", "remove", "avoid", "회피", "산만", "잡념"),
     (), _IDLE_SET),
    # 노력·전념·지속 → 생산적 우선 + idle 억제
    (("effort", "노력", "지속", "sustain", "focus", "전념", "꾸준",
      "persist", "집중", "diligence"),
     _PRODUCTIVE_SET, _IDLE_SET),
    # 계획·환경 조성·준비 → 일 진행
    (("plan", "계획", "환경", "조성", "prepare", "setup", "준비", "organize",
      "schedule", "정리"),
     _PRODUCTIVE_SET, ()),
    # 방법 적합성·전략·최적화 → 일 진행 (Phase 2 에서 task-routing 으로 정교화)
    (("method", "방법", "적합", "맞는", "appropriate", "fit", "strategy",
      "전략", "approach", "최적", "optimize"),
     _PRODUCTIVE_SET, ()),
    # 직접 능력 — 작성/문서
    (("write", "작성", "문서", "draft", "초안", "답변", "report", "글"),
     _WRITE_SET, ()),
    # 직접 능력 — 코딩/개발
    (("code", "코드", "개발", "program", "구현", "build", "scaffold",
      "스캐폴", "프로그램"),
     _CODE_SET, ()),
    # 직접 능력 — 조사/검색/탐색
    (("research", "조사", "검색", "시장", "탐색", "study", "explore",
      "investigate", "학습", "분석"),
     _RESEARCH_SET, ()),
)


def default_keyword_grounder(
    trigger_action: str,
    conditions: Optional[list] = None,
    nl_patterns: Optional[list] = None,
) -> dict:
    """추상 트리거 문자열 → {favor: [action_id], suppress: [action_id]}.

    결정론적 키워드 매핑. 매칭 없으면 빈 dict (보수적 — 접지 못 하면 강제 X).
    Phase 2 가 LLM grounder 로 교체 가능 (같은 시그니처).
    """
    parts = [str(trigger_action or "")]
    parts += [str(c) for c in (conditions or [])]
    parts += [str(p) for p in (nl_patterns or [])]
    text = " ".join(parts).lower()
    favor: set = set()
    suppress: set = set()
    for kws, fav, sup in _GROUND_RULES:
        if any(k.lower() in text for k in kws):
            favor.update(fav)
            suppress.update(sup)
    return {"favor": sorted(favor), "suppress": sorted(suppress)}


# ── alpha 해석 ────────────────────────────────────────────────────────────
_ALPHA_RE = re.compile(r"(alpha_\d+)")


def _resolve_alpha(pattern, delta_str) -> tuple:
    """change.delta('{alpha_2}' 또는 직접 숫자) → (signed value, trust 0..1).

    value: parameters[alpha_N] (학습된 현재값·부호 포함). 못 찾으면 None.
    trust: parameter_weights[alpha_N] (EMA 신뢰도). 없으면 mapping_confidence.
    """
    mc = 0.5
    try:
        mc = float(getattr(pattern, "mapping_confidence", 0.5) or 0.5)
    except Exception:
        mc = 0.5
    s = str(delta_str or "")
    m = _ALPHA_RE.search(s)
    if not m:
        # 미정계수 아닌 직접 수치 fallback
        try:
            return float(s.replace("%", "")), max(0.0, min(1.0, mc))
        except Exception:
            return None, 0.0
    name = m.group(1)
    try:
        params = getattr(pattern, "parameters", {}) or {}
        val = float(params.get(name))
    except Exception:
        val = None
    weights = getattr(pattern, "parameter_weights", {}) or {}
    w = weights.get(name)
    try:
        trust = float(w) if w is not None else mc
    except Exception:
        trust = mc
    return val, max(0.0, min(1.0, trust))


# ── 컴파일 ────────────────────────────────────────────────────────────────
_SELF_PREFIX = "EIDOS-self."


def _target_indicator(target: str) -> str:
    """change.target('EIDOS-self.reasoning') → 'reasoning'. self 아니면 ''."""
    t = str(target or "")
    if t.startswith(_SELF_PREFIX):
        return t[len(_SELF_PREFIX):]
    return ""


def compile_self_policy(
    goal_indicator: str,
    repertoire: Optional[list] = None,
    patterns: Optional[list] = None,
    ground_fn: Optional[Callable] = None,
    min_trust: float = 0.0,
) -> SelfPolicy:
    """self-applicable 패턴들 → SelfPolicy 컴파일.

    Args:
        goal_indicator: 현재 목표 지표 (이 지표에 작용하는 transition 만).
                        빈 문자열이면 모든 self 지표 대상.
        repertoire: EIDOS 가 실제 가진 action_id list (필터). None 이면 필터 안 함.
        patterns: 명시 주면 그대로, None 이면 catalog 에서 self-applicable 조회.
        ground_fn: 트리거 접지 함수 (None 이면 키워드 기본).
        min_trust: 이 신뢰도 미만 contributor 무시.

    Returns: SelfPolicy. 기여 0 이면 빈 정책.
    """
    ground = ground_fn or default_keyword_grounder
    if patterns is None:
        try:
            from eidos_pattern_catalog import list_self_applicable_patterns
            patterns = list_self_applicable_patterns(indicator=goal_indicator)
        except Exception as e:
            print(f"[self_policy] 패턴 로드 실패 (graceful): {e}")
            patterns = []

    rep_set = set(repertoire) if repertoire else None
    strength_map: dict = {}
    sources: list = []
    rationale: list = []
    max_trust = 0.0

    for p in (patterns or []):
        used = False
        for t in (getattr(p, "transitions", None) or []):
            if not isinstance(t, dict):
                continue
            # EIDOS 가 *직접 할 수 있는* 트리거만 강제 가능 (자기 행동).
            if (t.get("trigger_actor") or "") != "EIDOS-self":
                continue
            trig = t.get("trigger_action", "")
            conds = t.get("conditions")
            for c in (t.get("changes") or []):
                if not isinstance(c, dict):
                    continue
                ind = _target_indicator(c.get("target", ""))
                if not ind:
                    continue
                if goal_indicator and ind != goal_indicator:
                    continue   # 현재 목표 지표에 작용하는 것만
                val, trust = _resolve_alpha(p, c.get("delta", ""))
                if val is None or trust < min_trust or abs(val) < 1e-9:
                    continue
                sign = 1.0 if val > 0 else -1.0
                g = ground(trig, conds, getattr(p, "patterns", None))
                for a in g.get("favor", []):
                    if rep_set is not None and a not in rep_set:
                        continue
                    strength_map[a] = strength_map.get(a, 0.0) + sign * trust
                    used = True
                for a in g.get("suppress", []):
                    # idle 은 self repertoire 에 항상 있다고 보고 필터 예외
                    if rep_set is not None and a not in rep_set and a not in _IDLE_SET:
                        continue
                    strength_map[a] = strength_map.get(a, 0.0) - sign * trust
                    used = True
                if used:
                    max_trust = max(max_trust, trust)
        if used:
            pid = getattr(p, "id", "")
            if pid:
                sources.append(pid)
            r = getattr(p, "rationale", "")
            if r:
                rationale.append(str(r)[:120])

    force = sorted([a for a, s in strength_map.items() if s > 1e-9],
                   key=lambda a: -strength_map[a])
    suppress = sorted([a for a, s in strength_map.items() if s < -1e-9],
                      key=lambda a: strength_map[a])
    return SelfPolicy(
        force_actions=force,
        suppress_actions=suppress,
        action_strength=strength_map,
        source_pattern_ids=sources,
        rationale=rationale,
        strength=round(max_trust, 3),
    )
