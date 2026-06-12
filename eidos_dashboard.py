# eidos_dashboard.py
# [Wave12 2026-05-28] EIDOS 자기 모델·belief·calibration·prediction·job 통합 view.
#
# 사용자가 "지금 EIDOS 어떤 상태야"·"자기 모델 어떻게 변했어" 물어봤을 때
# 한눈에 볼 수 있는 md 보고서 자동 생성.
#
# section:
#   1. 핵심 metrics — work_state·emotion·active jobs·active milestones
#   2. 학습 통계 — top causal·overconfident warning·high curiosity domains
#   3. 자기 모델 — domain confidence·self_competence
#   4. 최근 활동 — proactive 발화·자율 chain·자기개선 사이클
#   5. 알림 — stagnant milestone·실패한 chain·calibration warning
#
# md 파일 저장 (`eidos_files/dashboard/`) + 채팅 표시 + 옵션 텔레그램.
# LLM 호출 0 (집계만)·즉시 응답.

from __future__ import annotations

import datetime as _dt
import os


_DASHBOARD_DIR = os.path.join("eidos_files", "dashboard")


def _ensure_dir() -> None:
    try:
        os.makedirs(_DASHBOARD_DIR, exist_ok=True)
    except Exception:
        pass


def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def build_dashboard_md(stage_id: str = "") -> str:
    """전체 EIDOS 상태 dashboard md 보고서 생성.

    section 별로 안전하게 시도·실패해도 빈 string·전체 fallback 절대 안 함.
    """
    sections: list = [
        "# 🪞 EIDOS Dashboard",
        f"_생성 시각: {_now_iso()}_",
        "",
    ]

    # ── 1. 핵심 metrics ─────────────────────────────────────────────
    sections.append("## 1. 핵심 metrics")
    sections.append("")
    sections.append(_section_core_metrics(stage_id))
    sections.append("")

    # ── 2. Active jobs & milestones ─────────────────────────────────
    sections.append("## 2. 진행 중 작업")
    sections.append("")
    sections.append(_section_jobs_and_milestones(stage_id))
    sections.append("")

    # ── 3. 학습 통계 ───────────────────────────────────────────────
    sections.append("## 3. 학습 통계 (Wave 4~5 누적)")
    sections.append("")
    sections.append(_section_learning_stats())
    sections.append("")

    # ── 4. 자기 모델 ────────────────────────────────────────────────
    sections.append("## 4. 자기 모델 (domain confidence)")
    sections.append("")
    sections.append(_section_self_model())
    sections.append("")

    # ── 5. Multi-actor belief ───────────────────────────────────────
    if stage_id:
        sections.append("## 5. Multi-actor belief")
        sections.append("")
        sections.append(_section_actor_belief(stage_id))
        sections.append("")

    # ── 6. 최근 활동 ────────────────────────────────────────────────
    sections.append("## 6. 최근 자율 활동")
    sections.append("")
    sections.append(_section_recent_activity())
    sections.append("")

    # ── 7. 알림 (warning) ──────────────────────────────────────────
    warnings = _section_warnings()
    if warnings.strip():
        sections.append("## 7. 알림")
        sections.append("")
        sections.append(warnings)
        sections.append("")

    return "\n".join(sections)


# ── Section 1: Core metrics ──────────────────────────────────────────
def _section_core_metrics(stage_id: str) -> str:
    lines = []
    # work_state·emotion·belief
    try:
        from eidos_belief_core import load_belief
        b = load_belief()
        lines.append(f"- **work_state**: `{b.work_state}` (since {b.work_state_since[:16] if b.work_state_since else '?'})")
        lines.append(f"- **message_count (전체)**: {b.message_count}")
        rt = (b.recent_topics or [])[:3]
        if rt:
            lines.append(f"- **최근 화제**: {', '.join(str(t)[:30] for t in rt)}")
        pt = b.get_pending_threads(status_filter=["open", "awaiting_response"]) \
            if hasattr(b, "get_pending_threads") else []
        if pt:
            lines.append(f"- **미완 thread**: {len(pt)}개")
    except Exception as e:
        lines.append(f"- belief 로드 실패 (graceful): {e}")

    # emotion
    try:
        from eidos_self_emotion import load_emotion
        e = load_emotion()
        lines.append(
            f"- **EIDOS 감정**: `{e.label}` ({e.sub_variant}) · "
            f"V={e.valence:+.2f} A={e.arousal:.2f} AF={e.affection:.2f}"
        )
    except Exception:
        pass

    # sensory (있으면)
    try:
        from eidos_sensory_grounding import capture_signals, signal_brief_for_prompt
        sig = capture_signals()
        if sig and sig.active_window_title:
            lines.append(f"- **활성 윈도우**: `{sig.active_window_title[:60]}`")
        if sig and sig.is_idle:
            lines.append(f"- **자리 비움**: {int(sig.idle_seconds // 60)}분+")
    except Exception:
        pass

    # GoalTree
    if stage_id:
        try:
            from eidos_goal_tree import load_goal_tree, get_active_milestones
            t = load_goal_tree(stage_id)
            if t and t.root_goal_id:
                actives = get_active_milestones(t) or []
                total = len(getattr(t, "milestones", {}) or {})
                done = max(0, total - len(actives))
                lines.append(
                    f"- **GoalTree**: {total}개 milestone · "
                    f"done {done} · active {len(actives)}"
                )
        except Exception:
            pass

    return "\n".join(lines) if lines else "_(데이터 없음)_"


# ── Section 2: Jobs & Milestones ─────────────────────────────────────
def _section_jobs_and_milestones(stage_id: str) -> str:
    lines = []
    # Active jobs (Wave11)
    try:
        from eidos_job_manager import get_active_jobs
        jobs = get_active_jobs()
        if jobs:
            lines.append("### Active jobs")
            for j in jobs[:6]:
                pct = f" · {int(j.progress_pct * 100)}%" if j.progress_pct > 0 else ""
                lines.append(
                    f"- `[{j.status}]` {j.title[:60]}{pct} "
                    f"(시도 {j.n_attempts}회·priority {j.priority})"
                )
        else:
            lines.append("_(활성 job 없음)_")
    except Exception:
        lines.append("_(job_manager 미초기화)_")
    lines.append("")

    # Active milestones
    if stage_id:
        try:
            from eidos_goal_tree import load_goal_tree, get_active_milestones
            t = load_goal_tree(stage_id)
            if t and t.root_goal_id:
                ams = get_active_milestones(t) or []
                if ams:
                    lines.append("### Active milestones")
                    for m in ams[:6]:
                        title = (getattr(m, "title", "") or "")[:60]
                        target = getattr(m, "target_date", "") or ""
                        progress = float(getattr(m, "progress", 0.0) or 0.0)
                        lines.append(
                            f"- {title}"
                            + (f" · target {target}" if target else "")
                            + f" · {int(progress * 100)}%"
                        )
        except Exception:
            pass

    return "\n".join(lines) if lines else "_(데이터 없음)_"


# ── Section 3: Learning stats ────────────────────────────────────────
def _section_learning_stats() -> str:
    lines = []

    # Counterfactual top causal
    try:
        from eidos_counterfactual import (
            load_recent_counterfactuals, aggregate_causal_attribution,
        )
        log = load_recent_counterfactuals(max_n=300)
        attrs = aggregate_causal_attribution(log, min_samples=3)
        if attrs:
            high = sorted(
                attrs.values(),
                key=lambda a: a.causal_strength, reverse=True,
            )[:3]
            if high:
                lines.append("### Top causal-effective action")
                for a in high:
                    lines.append(
                        f"- `{a.action_id}` · strength={a.causal_strength:.2f} "
                        f"(n={a.n_executions})"
                    )
    except Exception:
        pass

    # Calibration warnings
    try:
        from eidos_self_calibration import compute_calibration_map
        cm = compute_calibration_map(min_samples=3)
        if cm and cm.entries:
            overconf = [
                e for e in cm.entries.values()
                if e.n_forward >= 3 and e.confidence_adjustment < -0.2
            ]
            if overconf:
                lines.append("")
                lines.append("### Overconfident warnings")
                for e in sorted(overconf, key=lambda x: x.confidence_adjustment)[:3]:
                    lines.append(
                        f"- `{e.action_id}` · conf {e.avg_confidence:.2f} vs "
                        f"actual {e.avg_accuracy:.2f}"
                    )
            lines.append("")
            lines.append(
                f"- 전체 calibration: forward {cm.overall_forward_calibration:.2f} · "
                f"controllability {cm.overall_controllability:.2f}"
            )
    except Exception:
        pass

    # Curiosity signals
    try:
        from eidos_curiosity_driver import compute_curiosity_signals
        signals = compute_curiosity_signals(
            min_samples=3, min_error=0.3, max_signals=3,
        )
        if signals:
            lines.append("")
            lines.append("### High-curiosity domains (잘 모르는 영역)")
            for s in signals:
                target = (s.target or "")[:40]
                lines.append(
                    f"- [{s.target_type}] {target} · "
                    f"error={s.avg_error:.2f} · n={s.n_samples}"
                )
    except Exception:
        pass

    return "\n".join(lines) if lines else "_(아직 학습 데이터 부족)_"


# ── Section 4: Self model ────────────────────────────────────────────
def _section_self_model() -> str:
    try:
        from eidos_self_competence import (
            compute_domain_confidence, summary_for_log,
        )
        conf_map = compute_domain_confidence({})
        if not conf_map:
            return "_(domain confidence 데이터 부족)_"
        summary = summary_for_log(conf_map)
        lines = []
        for domain, dc in sorted(
            conf_map.items(),
            key=lambda kv: getattr(kv[1], "confidence", 0.0)
            if hasattr(kv[1], "confidence") else 0.0,
            reverse=True,
        )[:6]:
            try:
                c = getattr(dc, "confidence", 0.0)
                n = getattr(dc, "n_samples", 0)
                lines.append(
                    f"- **{domain}**: confidence {c:.2f} (n={n})"
                )
            except Exception:
                continue
        return "\n".join(lines) if lines else "_(데이터 없음)_"
    except Exception as e:
        return f"_(self_competence 모듈 부재: {e})_"


# ── Section 5: Actor belief ──────────────────────────────────────────
def _section_actor_belief(stage_id: str) -> str:
    try:
        from eidos_actor_belief import load_actor_store, as_prompt_brief
        store = load_actor_store(stage_id)
        if not store.actors:
            return "_(추적 중인 actor 없음·메시지에서 client/competitor 언급 시 자동 추가)_"
        lines = []
        for ab in sorted(
            store.actors.values(),
            key=lambda a: a.mention_count, reverse=True,
        )[:5]:
            ind_str = ""
            if ab.indicators:
                kv = ", ".join(
                    f"{k}={v:.2f}" if isinstance(v, float) and abs(v) <= 1.0
                    else f"{k}={v}"
                    for k, v in list(ab.indicators.items())[:4]
                )
                ind_str = f" · `{kv}`"
            lines.append(
                f"- **[{ab.role}] {ab.actor_name}** "
                f"(언급 {ab.mention_count}회){ind_str}"
            )
            if ab.notes:
                lines.append(f"    > {ab.notes[:120]}")
        return "\n".join(lines) if lines else "_(데이터 없음)_"
    except Exception as e:
        return f"_(actor_belief 부재: {e})_"


# ── Section 6: Recent activity ───────────────────────────────────────
def _section_recent_activity() -> str:
    lines = []

    # 최근 chain (Wave 7-B)
    try:
        from eidos_autonomous_workflow import load_recent_workflows
        wfs = load_recent_workflows(max_n=5)
        if wfs:
            lines.append("### 최근 자율 chain (top 5)")
            for w in wfs[-5:]:
                ts = (w.get("created_at") or "")[:16]
                title = (w.get("milestone_title") or "?")[:50]
                status = w.get("status", "?")
                n_steps = len(w.get("steps") or [])
                lines.append(f"- `{ts}` [{status}] {title} ({n_steps} step)")
    except Exception:
        pass

    # 최근 prediction (Wave 4)
    try:
        from eidos_prediction_engine import load_recent_predictions
        preds = load_recent_predictions(max_n=10, since_hours=24)
        if preds:
            n_surprised = sum(
                1 for p in preds
                if (p.get("error") or {}).get("surprised", False)
            )
            avg_err = (
                sum(float((p.get("error") or {}).get("error_score", 0.0))
                    for p in preds) / len(preds)
            )
            lines.append("")
            lines.append(
                f"- 최근 24시간 prediction: {len(preds)}회 · "
                f"surprised {n_surprised}회 · avg_error {avg_err:.2f}"
            )
    except Exception:
        pass

    return "\n".join(lines) if lines else "_(최근 활동 없음)_"


# ── Section 7: Warnings ──────────────────────────────────────────────
def _section_warnings() -> str:
    lines = []

    # Failed jobs
    try:
        from eidos_job_manager import load_jobs
        jobs = load_jobs()
        recent_failed = [j for j in jobs if j.status == "failed"][:3]
        if recent_failed:
            lines.append("### ⚠️ 최근 실패한 chain")
            for j in recent_failed:
                lines.append(
                    f"- {j.title[:50]} (시도 {j.n_attempts}회) · "
                    f"{j.result_preview[:60]}"
                )
    except Exception:
        pass

    # Calibration severe miscalibration
    try:
        from eidos_self_calibration import compute_calibration_map
        cm = compute_calibration_map(min_samples=3)
        if cm and cm.entries:
            severe = [
                e for e in cm.entries.values()
                if e.n_forward >= 5 and abs(e.confidence_adjustment) > 0.4
            ]
            if severe:
                if lines:
                    lines.append("")
                lines.append("### ⚠️ 심각한 mis-calibration")
                for e in severe[:3]:
                    direction = "과신" if e.confidence_adjustment < 0 else "과소"
                    lines.append(
                        f"- `{e.action_id}` ({direction}) — "
                        f"conf {e.avg_confidence:.2f} vs actual {e.avg_accuracy:.2f}"
                    )
    except Exception:
        pass

    return "\n".join(lines)


# ── 자연어 hook helper ────────────────────────────────────────────
_DASHBOARD_KEYWORDS = (
    "대시보드", "dashboard", "전체 상태", "전체 보고",
    "자기 모델", "내 모델", "EIDOS 상태",
    "종합 보고", "한눈에", "상태 한눈",
    "내 학습", "학습 통계",
)


def has_dashboard_keyword(text: str) -> bool:
    if not text:
        return False
    t = text.lower() if not any(ord(c) > 127 for c in text) else text
    return any(kw in t for kw in _DASHBOARD_KEYWORDS)


# ── 파일 저장 ─────────────────────────────────────────────────────
def save_dashboard(md: str) -> str:
    """dashboard md 를 파일로 저장·경로 반환·실패 시 빈 string."""
    _ensure_dir()
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(_DASHBOARD_DIR, f"dashboard_{ts}.md")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
        return path
    except Exception as e:
        print(f"[dashboard] 저장 실패 (graceful): {e}")
        return ""
