# eidos_curiosity_driver.py
# [Wave4 2026-05-28] Prediction error → Curiosity 신호 → 자율 탐색 목표.
#
# AGI 지향: "모르는 것" 이 자율 행동 driver. Schmidhuber 의 핵심:
#   intrinsic reward = prediction_error
#
# 입력: prediction_log.jsonl (prediction_engine 이 쓴 로그)
# 처리:
#   1. action_id·indicator·도메인 별 평균 error_score 집계
#   2. high-error + 빈도 적당 (n>=3) → "EIDOS 가 잘 모르는 곳"
#   3. CuriositySignal 생성·우선순위 매김
#   4. self_improvement_loop 가 가져가서 SelfGoal 로 변환
#
# 신호 형태:
#   - "이 action 결과 예측 자주 틀림 → 행동을 반복하며 forward model 보정"
#   - "이 indicator 변화 패턴 모름 → 그쪽 관련 task 자율 탐색"
#
# 한계: prediction_engine 이 OFF 면 로그 비어있음 → 신호 0. 둘 다 켜야 동작.

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Optional


@dataclass
class CuriositySignal:
    """탐색 가치가 있는 도메인의 신호."""
    signal_id: str = ""
    target: str = ""              # action_id 또는 indicator name
    target_type: str = "action"   # "action" | "indicator" | "domain"
    avg_error: float = 0.0        # 0~1
    n_samples: int = 0
    last_seen: str = ""

    # 우선순위 — error_score × n_samples 의 log scale
    priority: float = 0.0

    # 탐색 제안 (사람 가독성용)
    exploration_hint: str = ""

    def serialize(self) -> dict:
        return {
            "signal_id": self.signal_id,
            "target": self.target,
            "target_type": self.target_type,
            "avg_error": self.avg_error,
            "n_samples": self.n_samples,
            "last_seen": self.last_seen,
            "priority": self.priority,
            "exploration_hint": self.exploration_hint,
        }


def _new_id() -> str:
    import uuid
    return uuid.uuid4().hex[:8]


def compute_curiosity_signals(
    log_entries: Optional[list] = None,
    min_samples: int = 3,
    min_error: float = 0.3,
    max_signals: int = 5,
    since_hours: float = 168.0,   # 1주
) -> list[CuriositySignal]:
    """prediction_log 엔트리에서 curiosity 신호 추출.

    Args:
      log_entries: load_recent_predictions 결과. None 이면 자동 로드.
      min_samples: action 별 최소 샘플 — 너무 적으면 신호로 X (noise)
      min_error: avg_error 이 이하면 신호로 X (잘 알고 있음)
      max_signals: 반환 상위 N
      since_hours: 이 시간 내 로그만 (기본 1주)

    Returns: priority 내림차순 신호 list.
    """
    import math

    if log_entries is None:
        try:
            from eidos_prediction_engine import load_recent_predictions
            log_entries = load_recent_predictions(
                max_n=2000, since_hours=since_hours,
            )
        except Exception as e:
            print(f"[curiosity] log 로드 실패 (graceful): {e}")
            return []

    if not log_entries:
        return []

    # action_id 별 집계
    by_action: dict = {}
    for entry in log_entries:
        try:
            err = entry.get("error") or {}
            aid = str(err.get("action_id", ""))
            if not aid:
                continue
            score = float(err.get("error_score", 0.0))
            rec_at = str(err.get("recorded_at", ""))
            if aid not in by_action:
                by_action[aid] = {"scores": [], "last_seen": ""}
            by_action[aid]["scores"].append(score)
            if rec_at > by_action[aid]["last_seen"]:
                by_action[aid]["last_seen"] = rec_at
        except Exception:
            continue

    signals: list[CuriositySignal] = []

    # action 별 신호
    for aid, data in by_action.items():
        scores = data["scores"]
        n = len(scores)
        if n < min_samples:
            continue
        avg = sum(scores) / n
        if avg < min_error:
            continue
        # priority = avg_error × log(n)·log 로 빈도 영향 완화
        priority = float(avg) * math.log(n + 1.0)
        sig = CuriositySignal(
            signal_id=_new_id(),
            target=aid,
            target_type="action",
            avg_error=avg,
            n_samples=n,
            last_seen=data["last_seen"],
            priority=priority,
            exploration_hint=(
                f"'{aid}' 행동의 결과 예측이 자주 틀림 (avg_err={avg:.2f}·n={n}). "
                f"이 행동의 forward model 정확도 향상이 필요."
            ),
        )
        signals.append(sig)

    # indicator 별 집계 — 같은 방식
    by_indicator: dict = {}
    for entry in log_entries:
        try:
            err = entry.get("error") or {}
            ind_errs = err.get("indicator_errors") or {}
            if not isinstance(ind_errs, dict):
                continue
            rec_at = str(err.get("recorded_at", ""))
            for ind_name, ind_err in ind_errs.items():
                try:
                    e_val = float(ind_err)
                except Exception:
                    continue
                if ind_name not in by_indicator:
                    by_indicator[ind_name] = {"scores": [], "last_seen": ""}
                by_indicator[ind_name]["scores"].append(e_val)
                if rec_at > by_indicator[ind_name]["last_seen"]:
                    by_indicator[ind_name]["last_seen"] = rec_at
        except Exception:
            continue

    for ind_name, data in by_indicator.items():
        scores = data["scores"]
        n = len(scores)
        if n < min_samples:
            continue
        avg = sum(scores) / n
        if avg < min_error:
            continue
        priority = float(avg) * math.log(n + 1.0)
        sig = CuriositySignal(
            signal_id=_new_id(),
            target=ind_name,
            target_type="indicator",
            avg_error=avg,
            n_samples=n,
            last_seen=data["last_seen"],
            priority=priority,
            exploration_hint=(
                f"'{ind_name}' 지표 변화 예측이 자주 빗나감 (avg_err={avg:.2f}·n={n}). "
                f"이 지표의 dynamics 학습 필요."
            ),
        )
        signals.append(sig)

    # priority 내림차순 정렬·top N
    signals.sort(key=lambda s: s.priority, reverse=True)
    return signals[:max_signals]


def signal_to_self_goal(signal: CuriositySignal):
    """CuriositySignal → SelfGoal 변환.

    target_type='indicator' 면 그 indicator 의 prediction accuracy 향상을 goal 로.
    target_type='action' 이면 'pattern_accuracy' indicator (Wave1 default) 향상.

    반환: SelfGoal 또는 None (변환 불가).
    """
    try:
        from eidos_self_goals import SelfGoal
        from eidos_self_indicators import INDICATOR_NAMES
    except Exception:
        return None

    # target_indicator 결정
    if signal.target_type == "indicator" and signal.target in INDICATOR_NAMES:
        target_indicator = signal.target
    else:
        # action 또는 unknown indicator → fall back
        target_indicator = "pattern_accuracy"
        if target_indicator not in INDICATOR_NAMES:
            return None

    # current_value = 1.0 - avg_error (error 적을수록 confidence 높음 가정)
    # 단 indicator 가 prediction error 자체가 아니라 accuracy 라서 1-error 변환
    current_value = max(0.0, min(1.0, 1.0 - signal.avg_error))
    target_value = min(1.0, current_value + 0.15)

    desc = (
        f"[Curiosity 탐색] {signal.exploration_hint} "
        f"→ {target_indicator} 분포 {current_value:.2f} → {target_value:.2f} 변환"
    )

    try:
        goal = SelfGoal(
            id=_new_id(),
            target_indicator=target_indicator,
            current_value=current_value,
            target_value=target_value,
            direction="up",
            description=desc,
            status="active",
            created_at=_dt.datetime.now().isoformat(timespec="seconds"),
            cycles_active=0,
        )
        return goal
    except Exception as e:
        print(f"[curiosity] SelfGoal 생성 실패 (graceful): {e}")
        return None


def top_curiosity_goal(
    log_entries: Optional[list] = None,
    min_samples: int = 3,
    min_error: float = 0.3,
):
    """편의: 가장 priority 높은 curiosity signal 을 SelfGoal 로 변환.

    None 반환 시 — 신호 없음 (잘 알고 있거나 데이터 부족).
    """
    signals = compute_curiosity_signals(
        log_entries=log_entries,
        min_samples=min_samples,
        min_error=min_error,
        max_signals=1,
    )
    if not signals:
        return None
    return signal_to_self_goal(signals[0])
