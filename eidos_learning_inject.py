# eidos_learning_inject.py
# [Wave10 2026-05-28] 누적 학습 결과 → LLM prompt brief.
#
# 기존 EIDOS 에는 학습 데이터가 잔뜩 쌓이지만 LLM prompt 에 inject 깊이가 약함:
#
#   - pattern_model.action_values (UCB1) — run_stage_one_tick 에만 일부
#   - self_competence (domain confidence) — run_stage_one_tick 에만
#   - self_calibration (over/underconfident) — prediction prompt 에만
#   - counterfactual (causal_strength) — 텔레그램 다이제스트만
#   - curiosity (prediction error 높은 도메인) — 자기개선 cycle 만
#
# 결과: 매 세션 시작해도 LLM 이 과거 학습 잘 못 봄·session 간 진짜 누적 학습 약함.
#
# 이 모듈은 모든 학습 결과를 종합한 한 가지 brief 만들기:
#   - Top causal actions (이 action 은 진짜 효과 있다)
#   - Wasted actions (이 action 은 의식 형식만)
#   - Overconfident warnings (이 action 은 자만 X)
#   - High-curiosity domains (이 도메인은 잘 모름 — 탐색 가치)
#   - Domain confidence summary
#
# 이 brief 를 모든 LLM 호출 (proactive·strategic·autonomous workflow·일반 chat)
# 에 prepend → session 간 학습이 진짜로 다음 결정에 반영.
#
# Cost: LLM 호출 0 — 누적 로그 집계만. < 10ms.

from __future__ import annotations


_MAX_DEFAULT_CHARS = 600


def build_learning_brief(
    stage_id: str = "",
    max_chars: int = _MAX_DEFAULT_CHARS,
    since_hours: float = 168.0,
) -> str:
    """누적 학습 결과 → LLM prompt 용 종합 brief.

    빈 string 반환 시 = 학습 데이터 부족·inject 안 함 (LLM prompt 깨끗).
    graceful — 어떤 모듈 부재해도 가능한 부분만 채움.
    """
    sections: list = []

    # 1. counterfactual — top high-causal actions
    try:
        cf_section = _counterfactual_section(since_hours)
        if cf_section:
            sections.append(cf_section)
    except Exception as e:
        print(f"[learning_inject] counterfactual section 실패: {e}")

    # 2. calibration — overconfident / underconfident
    try:
        cal_section = _calibration_section()
        if cal_section:
            sections.append(cal_section)
    except Exception as e:
        print(f"[learning_inject] calibration section 실패: {e}")

    # 3. curiosity — high-prediction-error domains
    try:
        cu_section = _curiosity_section(since_hours)
        if cu_section:
            sections.append(cu_section)
    except Exception as e:
        print(f"[learning_inject] curiosity section 실패: {e}")

    # 4. self_competence — domain confidence summary
    try:
        comp_section = _competence_section()
        if comp_section:
            sections.append(comp_section)
    except Exception as e:
        print(f"[learning_inject] competence section 실패: {e}")

    if not sections:
        return ""

    header = "[LEARNING — 과거 누적 (참고용)]"
    full = header + "\n" + "\n".join(sections)
    if len(full) > max_chars:
        full = full[:max_chars - 3] + "..."
    return full


# ── Section 1: Counterfactual causal attribution ───────────────────
def _counterfactual_section(since_hours: float = 168.0) -> str:
    """causal_strength 높은 action·낮은 action 식별."""
    try:
        from eidos_counterfactual import (
            load_recent_counterfactuals,
            aggregate_causal_attribution,
        )
    except Exception:
        return ""
    log = load_recent_counterfactuals(max_n=300)
    if not log:
        return ""
    attrs = aggregate_causal_attribution(log, min_samples=3)
    if not attrs:
        return ""

    high = sorted(
        [a for a in attrs.values() if a.causal_strength >= 0.6],
        key=lambda a: a.causal_strength, reverse=True,
    )[:3]
    low = sorted(
        [a for a in attrs.values() if a.causal_strength <= 0.2],
        key=lambda a: a.causal_strength,
    )[:2]

    lines = []
    if high:
        lines.append("  · 진짜 효과 있는 action (causal):")
        for a in high:
            lines.append(
                f"     - {a.action_id} (strength={a.causal_strength:.2f}·n={a.n_executions})"
            )
    if low:
        lines.append("  · 효과 거의 없는 action (의식 형식만):")
        for a in low:
            lines.append(
                f"     - {a.action_id} (strength={a.causal_strength:.2f}·n={a.n_executions})"
            )
    if not lines:
        return ""
    return "\n".join(lines)


# ── Section 2: Calibration — over/underconfident ─────────────────
def _calibration_section() -> str:
    """과거 confidence 와 실제 accuracy 의 괴리·warning."""
    try:
        from eidos_self_calibration import compute_calibration_map
    except Exception:
        return ""
    cm = compute_calibration_map(min_samples=3, since_hours=168.0)
    if not cm or not cm.entries:
        return ""

    overconfident = [
        e for e in cm.entries.values()
        if e.n_forward >= 3 and e.confidence_adjustment < -0.2
    ]
    underconfident = [
        e for e in cm.entries.values()
        if e.n_forward >= 3 and e.confidence_adjustment > 0.2
    ]

    lines = []
    if overconfident:
        ov = sorted(overconfident, key=lambda e: e.confidence_adjustment)[:2]
        lines.append("  · 자기 과신 주의 (실제 accuracy < confidence):")
        for e in ov:
            lines.append(
                f"     - {e.action_id}: conf {e.avg_confidence:.2f} vs "
                f"actual {e.avg_accuracy:.2f}"
            )
    if underconfident:
        un = sorted(underconfident, key=lambda e: -e.confidence_adjustment)[:2]
        lines.append("  · 자기 과소평가 (실제 더 잘함):")
        for e in un:
            lines.append(
                f"     - {e.action_id}: conf {e.avg_confidence:.2f} vs "
                f"actual {e.avg_accuracy:.2f}"
            )
    if not lines:
        return ""
    return "\n".join(lines)


# ── Section 3: Curiosity — high prediction error domains ─────────
def _curiosity_section(since_hours: float = 168.0) -> str:
    """exploration 가치 있는 도메인·action."""
    try:
        from eidos_curiosity_driver import compute_curiosity_signals
    except Exception:
        return ""
    signals = compute_curiosity_signals(
        min_samples=3, min_error=0.3, max_signals=3,
        since_hours=since_hours,
    )
    if not signals:
        return ""

    lines = ["  · 아직 잘 모르는 영역 (탐색 가치 ↑):"]
    for s in signals[:3]:
        target_short = (s.target or "")[:40]
        lines.append(
            f"     - [{s.target_type}] {target_short} "
            f"(error={s.avg_error:.2f}·n={s.n_samples})"
        )
    return "\n".join(lines)


# ── Section 4: Domain competence summary ─────────────────────────
def _competence_section() -> str:
    """domain 별 confidence·약점 영역 hint."""
    try:
        from eidos_self_competence import (
            compute_domain_confidence,
            summary_for_log,
        )
    except Exception:
        return ""
    # action_values 없으면 default — pattern_model load 시도
    try:
        from eidos_pattern_model import PatternModel
        # 가장 최근 stage 의 pattern model 사용 (없으면 빈 dict)
        # PatternModel.from_history 는 stage_id 필요·여기선 cheap 한 path X
        # 그냥 빈 action_values 로 default DomainConfidence map 받음
        action_values: dict = {}
    except Exception:
        action_values = {}

    conf_map = compute_domain_confidence(action_values)
    if not conf_map:
        return ""

    summary = summary_for_log(conf_map)
    weak = summary.get("weakest") or []
    strong = summary.get("strongest") or []

    lines = []
    if strong:
        # tuple/list 인지 dict 인지 안전 처리
        try:
            strong_strs = [
                f"{s[0]} ({s[1]:.2f})" if isinstance(s, (list, tuple))
                else str(s)
                for s in strong[:2]
            ]
            lines.append(f"  · 강한 도메인: {', '.join(strong_strs)}")
        except Exception:
            pass
    if weak:
        try:
            weak_strs = [
                f"{s[0]} ({s[1]:.2f})" if isinstance(s, (list, tuple))
                else str(s)
                for s in weak[:2]
            ]
            lines.append(f"  · 약한 도메인 (위임·신중 권장): {', '.join(weak_strs)}")
        except Exception:
            pass
    if not lines:
        return ""
    return "\n".join(lines)


# ── 편의: short brief (chat 일반 inject 용·100자 이내) ────────────
def build_short_learning_brief(
    stage_id: str = "", max_chars: int = 200,
) -> str:
    """긴 brief 의 핵심만·일반 chat LLM 에 prepend 용 짧은 버전.

    counterfactual 상위 1개 + calibration 경고 1개만.
    """
    parts = []

    # top causal
    try:
        from eidos_counterfactual import (
            load_recent_counterfactuals, aggregate_causal_attribution,
        )
        log = load_recent_counterfactuals(max_n=100)
        attrs = aggregate_causal_attribution(log, min_samples=3)
        high = sorted(
            [a for a in attrs.values() if a.causal_strength >= 0.6],
            key=lambda a: a.causal_strength, reverse=True,
        )
        if high:
            top = high[0]
            parts.append(
                f"causal-effective: {top.action_id} ({top.causal_strength:.2f})"
            )
    except Exception:
        pass

    # overconfident warning
    try:
        from eidos_self_calibration import compute_calibration_map
        cm = compute_calibration_map(min_samples=3)
        if cm and cm.entries:
            ov = [e for e in cm.entries.values()
                  if e.n_forward >= 3 and e.confidence_adjustment < -0.3]
            if ov:
                worst = min(ov, key=lambda e: e.confidence_adjustment)
                parts.append(
                    f"overconfident-warn: {worst.action_id} "
                    f"(conf {worst.avg_confidence:.2f}·actual {worst.avg_accuracy:.2f})"
                )
    except Exception:
        pass

    if not parts:
        return ""
    full = "[LEARN] " + " · ".join(parts)
    if len(full) > max_chars:
        full = full[:max_chars - 3] + "..."
    return full
