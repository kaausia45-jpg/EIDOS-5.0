# eidos_self_calibration.py
# [Wave4-B 2026-05-28] 자기 확신도 보정 — forward + inverse 로그 종합 분석.
#
# 두 가지 신호 활용:
#   1. forward calibration — predicted_confidence vs actual_accuracy
#      "확신 0.9 했는데 매번 틀린다 → overconfident"
#   2. inverse controllability — action 별 inverse_accuracy
#      "이 action 은 detectable effect 가짐 / 안 가짐"
#
# 출력:
#   - action 별 calibration_score (0~1) + confidence_adjustment (-1.0 ~ +1.0)
#   - self_competence 자동 업데이트 (existing eidos_self_competence)
#   - prediction prompt 에 inject 할 calibration_hint
#
# 즉 EIDOS 가 자기 자신을 더 정확히 알게 됨 — overclaim 줄어들고,
# 실제 effect 없는 action 식별, 잘 알고/모르는 도메인 자기 모델 진화.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CalibrationEntry:
    """action 별 calibration 종합 점수."""
    action_id: str = ""
    # forward
    n_forward: int = 0
    avg_confidence: float = 0.0
    avg_accuracy: float = 0.0
    forward_calibration: float = 0.0    # 1 - |conf - accuracy|·1.0=완벽
    # inverse
    n_inverse: int = 0
    inverse_accuracy: float = 0.0       # inverse model 이 맞춘 비율
    # 종합
    controllability_score: float = 0.0  # inverse accuracy 와 동치
    confidence_adjustment: float = 0.0  # 다음 prediction 에서 confidence 조정 권장값
    last_evaluated: str = ""

    def serialize(self) -> dict:
        return {
            "action_id": self.action_id,
            "n_forward": self.n_forward,
            "avg_confidence": self.avg_confidence,
            "avg_accuracy": self.avg_accuracy,
            "forward_calibration": self.forward_calibration,
            "n_inverse": self.n_inverse,
            "inverse_accuracy": self.inverse_accuracy,
            "controllability_score": self.controllability_score,
            "confidence_adjustment": self.confidence_adjustment,
            "last_evaluated": self.last_evaluated,
        }


@dataclass
class CalibrationMap:
    """전체 action 의 calibration map."""
    entries: dict = field(default_factory=dict)   # {action_id: CalibrationEntry}
    overall_forward_calibration: float = 0.0
    overall_controllability: float = 0.0
    n_actions: int = 0
    computed_at: str = ""

    def get(self, action_id: str) -> Optional[CalibrationEntry]:
        return self.entries.get(action_id)


# ── 핵심 계산 ─────────────────────────────────────────────────────────
def compute_calibration_map(
    forward_log: Optional[list] = None,
    inverse_log: Optional[list] = None,
    min_samples: int = 3,
    since_hours: float = 168.0,
) -> CalibrationMap:
    """prediction + inverse 로그 종합 → action 별 calibration 계산."""
    import datetime as _dt

    if forward_log is None:
        try:
            from eidos_prediction_engine import load_recent_predictions
            forward_log = load_recent_predictions(
                max_n=2000, since_hours=since_hours,
            )
        except Exception:
            forward_log = []

    if inverse_log is None:
        try:
            from eidos_inverse_model import load_recent_inverses
            inverse_log = load_recent_inverses(
                max_n=2000, since_hours=since_hours,
            )
        except Exception:
            inverse_log = []

    # forward 집계
    forward_by_action: dict = {}
    for entry in (forward_log or []):
        try:
            pred = entry.get("prediction") or {}
            err = entry.get("error") or {}
            aid = str(pred.get("action_id", ""))
            if not aid:
                continue
            if aid not in forward_by_action:
                forward_by_action[aid] = {
                    "conf_sum": 0.0, "acc_sum": 0.0, "n": 0,
                }
            forward_by_action[aid]["conf_sum"] += float(pred.get("confidence", 0.5))
            # accuracy = 1 - error_score (clamp)
            es = float(err.get("error_score", 0.0))
            acc = max(0.0, min(1.0, 1.0 - es))
            forward_by_action[aid]["acc_sum"] += acc
            forward_by_action[aid]["n"] += 1
        except Exception:
            continue

    # inverse 집계
    inverse_by_action: dict = {}
    for entry in (inverse_log or []):
        try:
            aid = str(entry.get("actual_action_id", ""))
            if not aid:
                continue
            if aid not in inverse_by_action:
                inverse_by_action[aid] = {"correct": 0, "n": 0}
            inverse_by_action[aid]["n"] += 1
            if bool(entry.get("correct", False)):
                inverse_by_action[aid]["correct"] += 1
        except Exception:
            continue

    cm = CalibrationMap(computed_at=_dt.datetime.now().isoformat(timespec="seconds"))
    overall_fwd: list = []
    overall_ctrl: list = []

    # 합집합 action ids
    all_aids = set(forward_by_action.keys()) | set(inverse_by_action.keys())
    for aid in all_aids:
        e = CalibrationEntry(action_id=aid, last_evaluated=cm.computed_at)
        # forward
        fwd = forward_by_action.get(aid)
        if fwd and fwd["n"] >= min_samples:
            e.n_forward = fwd["n"]
            e.avg_confidence = fwd["conf_sum"] / fwd["n"]
            e.avg_accuracy = fwd["acc_sum"] / fwd["n"]
            e.forward_calibration = 1.0 - abs(e.avg_confidence - e.avg_accuracy)
            # confidence_adjustment: 음수면 다음 prediction 에서 confidence ↓
            # 양수면 ↑ (지금은 자신 부족·실제는 더 잘함)
            e.confidence_adjustment = e.avg_accuracy - e.avg_confidence
            overall_fwd.append(e.forward_calibration)
        # inverse
        inv = inverse_by_action.get(aid)
        if inv and inv["n"] >= min_samples:
            e.n_inverse = inv["n"]
            e.inverse_accuracy = inv["correct"] / inv["n"]
            e.controllability_score = e.inverse_accuracy
            overall_ctrl.append(e.controllability_score)
        if e.n_forward >= min_samples or e.n_inverse >= min_samples:
            cm.entries[aid] = e

    cm.n_actions = len(cm.entries)
    cm.overall_forward_calibration = (
        sum(overall_fwd) / len(overall_fwd) if overall_fwd else 0.0
    )
    cm.overall_controllability = (
        sum(overall_ctrl) / len(overall_ctrl) if overall_ctrl else 0.0
    )
    return cm


# ── self_competence 업데이트 hook ─────────────────────────────────────
def update_self_competence_from_calibration(
    cm: CalibrationMap,
) -> dict:
    """calibration 결과를 self_competence module 로 push.

    self_competence 는 domain 별 confidence 를 유지함 (Phase 14).
    각 action 의 domain 매핑은 eidos_pca_catalog 의 action→domain·또는
    action_id prefix 휴리스틱 사용.

    Returns: {"updated_domains": [...], "skipped": N}.
    """
    out = {"updated_domains": [], "skipped": 0, "n_actions_used": 0}
    if not cm or not cm.entries:
        return out

    # action_id → domain prefix 매핑 (간소·확장 가능)
    def _action_to_domain(aid: str) -> str:
        try:
            if aid.startswith("eidos.meta."):
                return "meta"
            if aid.startswith("eidos.tool.llm"):
                return "writing"
            if aid.startswith("eidos.tool.web"):
                return "research"
            if aid.startswith("eidos.tool.pca"):
                return "automation"
            if aid.startswith("eidos.tool.message"):
                return "communication"
            if aid.startswith("eidos.tool.file"):
                return "fileops"
            if aid.startswith("eidos.tool.ask_user"):
                return "interaction"
        except Exception:
            pass
        return "general"

    # domain 별 평균 score 집계
    by_domain: dict = {}
    for aid, e in cm.entries.items():
        d = _action_to_domain(aid)
        if d not in by_domain:
            by_domain[d] = {"scores": [], "n": 0}
        # confidence 신호 = forward_calibration 와 controllability 평균
        signal = (e.forward_calibration + e.controllability_score) / 2.0
        by_domain[d]["scores"].append(signal)
        by_domain[d]["n"] += 1
        out["n_actions_used"] += 1

    # self_competence 모듈에 push (있으면)
    try:
        import eidos_self_competence as sc
    except Exception:
        out["skipped"] = out["n_actions_used"]
        return out

    # eidos_self_competence 의 update_domain_confidence 사용 (있다면)
    for domain, data in by_domain.items():
        if not data["scores"]:
            continue
        avg = sum(data["scores"]) / len(data["scores"])
        try:
            # 모듈에 update_domain_confidence 가 있다면 호출, 없으면 기본 indicator log
            if hasattr(sc, "update_domain_confidence"):
                sc.update_domain_confidence(domain, new_confidence=avg,
                                            n_samples=data["n"])
                out["updated_domains"].append(domain)
            else:
                # 폴백 — print only
                print(f"[calibration] {domain} avg_confidence={avg:.2f} "
                      f"n={data['n']} (sc 모듈 hook 없음·로그만)")
        except Exception as e:
            print(f"[calibration] {domain} 업데이트 실패 (graceful): {e}")
    return out


# ── prediction prompt 용 calibration hint ─────────────────────────────
def calibration_hint_for_action(
    action_id: str, cm: Optional[CalibrationMap] = None,
) -> str:
    """다음 prediction 생성 시 prompt 에 inject 할 짧은 보정 hint.

    LLM 이 자기 예측 confidence 를 calibration 정보 보고 조정하도록.
    cm 없으면 빈 string.
    """
    if cm is None or not action_id:
        return ""
    e = cm.get(action_id)
    if e is None or (e.n_forward < 3 and e.n_inverse < 3):
        return ""
    parts = []
    if e.n_forward >= 3:
        if e.confidence_adjustment < -0.15:
            parts.append(
                f"이 action 은 과거 confidence 평균 {e.avg_confidence:.2f} 였는데 "
                f"실제 accuracy 는 {e.avg_accuracy:.2f}. **overconfident** — "
                f"이번엔 confidence 낮춰서 예측."
            )
        elif e.confidence_adjustment > 0.15:
            parts.append(
                f"이 action 은 과거 confidence 평균 {e.avg_confidence:.2f} 였는데 "
                f"실제 accuracy 는 {e.avg_accuracy:.2f}. **underconfident** — "
                f"실은 잘 됨·confidence 올려도 됨."
            )
    if e.n_inverse >= 3:
        if e.controllability_score < 0.3:
            parts.append(
                f"이 action 의 inverse_accuracy={e.controllability_score:.2f} — "
                f"실제 effect 가 noise 에 묻혀 detectable 안 됨·소소한 변화 예측."
            )
        elif e.controllability_score > 0.7:
            parts.append(
                f"이 action 의 inverse_accuracy={e.controllability_score:.2f} — "
                f"실제 effect 가 명확함·자신 있게 예측."
            )
    if not parts:
        return ""
    return "[CALIBRATION 힌트] " + " / ".join(parts)
