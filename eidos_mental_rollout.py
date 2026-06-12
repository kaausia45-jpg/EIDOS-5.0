# eidos_mental_rollout.py
# [Wave4-C 2026-05-28] Mental rollout — imagination 층.
#
# 행동 실행 전 머릿속에서 N-step 시뮬레이션. 후보 action 여러 개의
# trajectory 를 펼쳐 비교 → 최고 점수 trajectory 의 first_action 실행.
#
# 학술 배경:
#   - AlphaGo / MuZero 의 monte carlo tree search 의 단순화 버전 (각 후보 1 chain)
#   - Model-based RL 의 planning step
#   - Lake et al. (2017) "building machines that learn and think like people":
#     mental simulation 이 인간 사고의 핵심
#
# 흐름:
#   1. K 후보 first_action 받음 (caller 가 LLM 으로 생성)
#   2. 각 후보에 대해:
#      - predict next state (forward model 재사용)
#      - 그 imagined state 에서 best next action 선택
#      - 다시 predict, depth N 까지 반복
#   3. 각 trajectory 를 LLM 으로 평가 (goal alignment·risk·indicator)
#   4. final_score 내림차순 정렬 → 1순위가 실제 실행될 후보
#
# 비용: K × N × ~2 LLM calls. K=3, N=2 → ~12 calls/tick (~$0.003 Gemini Flash).
# Default OFF — settings.mental_rollout_enabled=true.

from __future__ import annotations

import asyncio
import json
import os
import datetime as _dt
from dataclasses import dataclass, field, asdict
from typing import Optional


_LOG_DIR = os.path.join("eidos_files", "predictions")
_ROLLOUT_LOG = os.path.join(_LOG_DIR, "rollout_log.jsonl")
_MAX_LOG_BYTES = 2 * 1024 * 1024


# ── dataclasses ──────────────────────────────────────────────────────
@dataclass
class RolloutStep:
    """Trajectory 한 step — imagined state 와 action."""
    step_idx: int = 0
    action_id: str = ""                    # 이 step 의 action
    args_hint: str = ""                    # 짧은 args 요약
    predicted_state_text: str = ""         # 자연어 state 표현
    predicted_indicators: dict = field(default_factory=dict)
    confidence: float = 0.5
    rationale: str = ""

    def serialize(self) -> dict:
        return asdict(self)


@dataclass
class RolloutTrajectory:
    """K-step imagined trajectory from one first_action."""
    trajectory_id: str = ""
    first_action_id: str = ""
    first_args: dict = field(default_factory=dict)
    steps: list = field(default_factory=list)        # list[RolloutStep]
    depth: int = 0

    # 평가
    final_score: float = 0.0          # 0~1·종합 점수
    goal_alignment: float = 0.0       # 0~1·목표 달성 기여도
    indicator_score: float = 0.0      # 0~1·positive indicator 평균 delta
    risk_score: float = 0.0           # 0~1·high=위험 (낮은 confidence·uncontrollable)
    risk_flags: list = field(default_factory=list)
    total_confidence: float = 0.5     # geometric mean of step confidences
    evaluator_reason: str = ""

    created_at: str = ""
    raw_llm: str = ""

    def serialize(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "first_action_id": self.first_action_id,
            "first_args": self.first_args,
            "steps": [s.serialize() if hasattr(s, "serialize") else s
                      for s in self.steps],
            "depth": self.depth,
            "final_score": self.final_score,
            "goal_alignment": self.goal_alignment,
            "indicator_score": self.indicator_score,
            "risk_score": self.risk_score,
            "risk_flags": self.risk_flags,
            "total_confidence": self.total_confidence,
            "evaluator_reason": self.evaluator_reason,
            "created_at": self.created_at,
        }


def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _new_id() -> str:
    import uuid
    return uuid.uuid4().hex[:10]


def _ensure_log_dir() -> None:
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
    except Exception:
        pass


def _rotate_if_needed() -> None:
    try:
        if (os.path.exists(_ROLLOUT_LOG)
                and os.path.getsize(_ROLLOUT_LOG) > _MAX_LOG_BYTES):
            bak = _ROLLOUT_LOG + f".{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
            try:
                os.rename(_ROLLOUT_LOG, bak)
            except Exception:
                pass
    except Exception:
        pass


# ── LLM 시스템 프롬프트 ─────────────────────────────────────────────────
_NEXT_STEP_SYSTEM = """\
너는 EIDOS 의 mental rollout simulator 다.

상상 속 시점에서 EIDOS 가 마지막으로 한 action 의 결과를 보고,
**다음에 어떤 action 을 할지** + **그 결과 state 가 어떻게 될지** 예측해라.

원칙:
- imagined state 는 자연어 1~2 문장 (간결)
- action 은 주어진 repertoire 중 1개만
- confidence 0~1 — 잘 모르겠으면 0.3 이하·확실하면 0.8+
- 변화 미미하면 (관찰만·meta) "eidos.meta.observe" / "eidos.meta.no_op"

JSON 만 출력:
{
  "next_action_id": "<repertoire 중 하나>",
  "predicted_state_text": "<1~2 문장 자연어>",
  "predicted_indicators": {"indicator": <-1.0~+1.0 delta>},
  "confidence": 0.0~1.0,
  "rationale": "<짧은 한 줄>"
}
"""

_TRAJ_EVAL_SYSTEM = """\
너는 EIDOS 의 trajectory evaluator 다.

상상 속 행동 시퀀스 (trajectory) 를 보고, 현재 goal_context 에 비추어
trajectory 가 얼마나 가치 있는지 평가해라.

평가 축:
- goal_alignment 0~1: 이 trajectory 가 현재 목표 달성에 기여하는가
- indicator_score 0~1: positive indicators (예: trust·user_acceptance·pattern_accuracy)
  가 평균적으로 상승하는가
- risk_score 0~1 (높을수록 위험): confidence 낮음·controllability 낮음·외부효과 큼
- risk_flags: ["low_confidence", "uncontrollable", "external_effect_high", "no_milestone_progress"] 중 해당

final_score = goal_alignment * 0.5 + indicator_score * 0.3 + (1 - risk_score) * 0.2
(자동 계산 — 너는 각 축만 매기면 됨)

JSON 만 출력:
{
  "goal_alignment": 0.0~1.0,
  "indicator_score": 0.0~1.0,
  "risk_score": 0.0~1.0,
  "risk_flags": ["..."],
  "reason": "한 줄 평가"
}
"""


# ── 핵심: 한 trajectory rollout ────────────────────────────────────────
async def rollout_one_trajectory_async(
    first_action_id: str,
    first_args: Optional[dict] = None,
    starting_state_text: str = "",
    starting_indicators: Optional[dict] = None,
    action_repertoire: Optional[list] = None,
    goal_context: str = "",
    depth: int = 2,
    timeout_per_step: float = 8.0,
) -> RolloutTrajectory:
    """첫 action 으로 시작하는 depth-step imagined trajectory.

    각 step 마다 LLM 호출 1회 — next_action + predicted_state 동시 생성.
    forward model 의 단순 chain (no branching at intermediate steps).
    """
    traj = RolloutTrajectory(
        trajectory_id=_new_id(),
        first_action_id=first_action_id,
        first_args=first_args or {},
        depth=depth,
        created_at=_now_iso(),
    )

    if not first_action_id:
        return traj

    # Step 0 — 첫 action 의 predicted outcome (forward model 1회)
    try:
        from eidos_prediction_engine import predict_action_outcome_async
        first_pred = await predict_action_outcome_async(
            action_id=first_action_id,
            args=first_args or {},
            belief_context=starting_indicators,
            history_brief=starting_state_text,
            timeout_sec=timeout_per_step,
        )
        step0 = RolloutStep(
            step_idx=0,
            action_id=first_action_id,
            args_hint=first_pred.args_summary[:100],
            predicted_state_text=first_pred.expected_result_text,
            predicted_indicators=dict(first_pred.expected_indicators or {}),
            confidence=first_pred.confidence,
            rationale="첫 candidate action 의 forward prediction",
        )
        traj.steps.append(step0)
    except Exception as e:
        print(f"[rollout] step 0 forward 실패 (graceful): {e}")
        # 실패해도 빈 step 으로 진행 — 평가에서 confidence 낮게 매겨질 것
        traj.steps.append(RolloutStep(
            step_idx=0, action_id=first_action_id,
            predicted_state_text=f"(forward 실패: {str(e)[:80]})",
            confidence=0.1,
        ))

    # Steps 1..depth-1 — next_step LLM 으로 chain
    try:
        from llm_module import get_llm_response_async
    except Exception:
        return _finalize_traj(traj)

    rep_list = list(action_repertoire or [])[:20]
    rep_block = "\n".join(f"- {a}" for a in rep_list) or "(repertoire 비어있음)"

    current_state_text = traj.steps[0].predicted_state_text
    current_indicators = dict(traj.steps[0].predicted_indicators or {})

    for step_idx in range(1, max(1, depth)):
        try:
            prompt = (
                f"[goal_context] {goal_context[:300]}\n"
                f"[action_repertoire]\n{rep_block}\n\n"
                f"[현재 imagined state] {current_state_text[:300]}\n"
                f"[현재 누적 indicators] "
                f"{json.dumps(current_indicators, ensure_ascii=False)[:300]}\n\n"
                f"이 시점에서 EIDOS 가 다음 어떤 action 을 할지 + 그 결과 어떤 state 가 "
                f"될지 예측. JSON 만 출력."
            )
            raw = await asyncio.wait_for(
                get_llm_response_async(
                    prompt,
                    max_tokens=1024,
                    system_prompt=_NEXT_STEP_SYSTEM,
                    response_mime_type="application/json",
                ),
                timeout=timeout_per_step,
            )
            data = json.loads(raw)
            if not isinstance(data, dict):
                break
            next_aid = str(data.get("next_action_id", ""))[:80]
            if not next_aid:
                break
            try:
                conf = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
            except Exception:
                conf = 0.5
            pred_ind = {}
            if isinstance(data.get("predicted_indicators"), dict):
                for k, v in data["predicted_indicators"].items():
                    try:
                        pred_ind[str(k)[:40]] = max(-1.0, min(1.0, float(v)))
                    except Exception:
                        continue
            # 누적 indicator 변화 update
            for k, dv in pred_ind.items():
                current_indicators[k] = current_indicators.get(k, 0.0) + dv

            step = RolloutStep(
                step_idx=step_idx,
                action_id=next_aid,
                predicted_state_text=str(data.get("predicted_state_text", ""))[:300],
                predicted_indicators=pred_ind,
                confidence=conf,
                rationale=str(data.get("rationale", ""))[:150],
            )
            traj.steps.append(step)
            current_state_text = step.predicted_state_text
        except asyncio.TimeoutError:
            print(f"[rollout] step {step_idx} timeout — chain 종료")
            break
        except Exception as e:
            print(f"[rollout] step {step_idx} 실패 (graceful·chain 종료): {e}")
            break

    return _finalize_traj(traj)


def _finalize_traj(traj: RolloutTrajectory) -> RolloutTrajectory:
    """총 confidence 계산 (geometric mean) — 평가 전 보조 값."""
    import math
    if traj.steps:
        confs = [max(0.001, s.confidence) for s in traj.steps]
        try:
            log_sum = sum(math.log(c) for c in confs)
            traj.total_confidence = math.exp(log_sum / len(confs))
        except Exception:
            traj.total_confidence = sum(confs) / len(confs)
    return traj


# ── trajectory 평가 ──────────────────────────────────────────────────
async def evaluate_trajectory_async(
    traj: RolloutTrajectory,
    goal_context: str = "",
    timeout_sec: float = 8.0,
) -> RolloutTrajectory:
    """LLM 으로 trajectory 평가 — goal_alignment·indicator_score·risk 매기기.

    self_calibration 의 confidence_adjustment 도 반영 — overconfident action
    이 포함되면 risk 증가.
    """
    try:
        from llm_module import get_llm_response_async
    except Exception:
        return traj

    # trajectory 요약 문자열
    step_lines = []
    for s in traj.steps:
        step_lines.append(
            f"  Step {s.step_idx}: {s.action_id} → "
            f"{s.predicted_state_text[:60]} (conf={s.confidence:.2f})"
        )
    traj_block = "\n".join(step_lines) if step_lines else "(empty)"

    # 누적 indicator 변화 요약
    cum_ind: dict = {}
    for s in traj.steps:
        for k, v in (s.predicted_indicators or {}).items():
            cum_ind[k] = cum_ind.get(k, 0.0) + v
    ind_block = json.dumps(cum_ind, ensure_ascii=False)[:300] if cum_ind else "{}"

    prompt = (
        f"[goal_context]\n{goal_context[:400]}\n\n"
        f"[imagined trajectory ({len(traj.steps)} steps)]\n{traj_block}\n\n"
        f"[누적 indicator 변화]\n{ind_block}\n\n"
        f"[total_confidence] {traj.total_confidence:.2f}\n\n"
        "위 trajectory 를 평가. JSON 만 출력."
    )

    try:
        raw = await asyncio.wait_for(
            get_llm_response_async(
                prompt,
                max_tokens=1024,
                system_prompt=_TRAJ_EVAL_SYSTEM,
                response_mime_type="application/json",
            ),
            timeout=timeout_sec,
        )
        traj.raw_llm = (raw or "")[:600]
        data = json.loads(raw)
        if isinstance(data, dict):
            try:
                traj.goal_alignment = max(0.0, min(1.0, float(data.get("goal_alignment", 0.0))))
                traj.indicator_score = max(0.0, min(1.0, float(data.get("indicator_score", 0.0))))
                traj.risk_score = max(0.0, min(1.0, float(data.get("risk_score", 0.0))))
            except Exception:
                pass
            rf = data.get("risk_flags")
            if isinstance(rf, list):
                traj.risk_flags = [str(x)[:40] for x in rf[:6]]
            traj.evaluator_reason = str(data.get("reason", ""))[:200]
    except asyncio.TimeoutError:
        print(f"[rollout-eval] timeout — 점수 0 유지")
    except Exception as e:
        print(f"[rollout-eval] 실패 (graceful): {e}")

    # 최종 점수 (정해진 가중치)
    traj.final_score = (
        traj.goal_alignment * 0.5
        + traj.indicator_score * 0.3
        + (1.0 - traj.risk_score) * 0.2
    )
    # confidence 가 매우 낮으면 페널티
    traj.final_score *= max(0.5, min(1.0, traj.total_confidence + 0.2))
    return traj


# ── 후보 비교 ───────────────────────────────────────────────────────
async def compare_rollouts_async(
    candidate_first_actions: list,
    starting_state_text: str = "",
    starting_indicators: Optional[dict] = None,
    action_repertoire: Optional[list] = None,
    goal_context: str = "",
    depth: int = 2,
) -> list:
    """K 후보 first action 의 trajectory 펼침·평가·점수 내림차순 list 반환.

    Args:
      candidate_first_actions: [{"action_id": str, "args": dict}, ...]
      그 외 컨텍스트.

    Returns: list[RolloutTrajectory] (final_score 내림차순).
    """
    if not candidate_first_actions:
        return []

    # 1) 각 후보 rollout — 동시 (gather)
    rollout_tasks = [
        rollout_one_trajectory_async(
            first_action_id=cand.get("action_id", ""),
            first_args=cand.get("args", {}),
            starting_state_text=starting_state_text,
            starting_indicators=starting_indicators,
            action_repertoire=action_repertoire,
            goal_context=goal_context,
            depth=depth,
        )
        for cand in candidate_first_actions
    ]
    trajectories = await asyncio.gather(*rollout_tasks, return_exceptions=True)

    # 2) 각 trajectory 평가 — 역시 동시
    valid_trajs: list = []
    for t in trajectories:
        if isinstance(t, RolloutTrajectory):
            valid_trajs.append(t)
        else:
            print(f"[rollout] 후보 rollout 예외 (skip): {t}")

    eval_tasks = [
        evaluate_trajectory_async(t, goal_context=goal_context)
        for t in valid_trajs
    ]
    if eval_tasks:
        evaluated = await asyncio.gather(*eval_tasks, return_exceptions=True)
        valid_trajs = [
            t if isinstance(t, RolloutTrajectory) else valid_trajs[i]
            for i, t in enumerate(evaluated)
        ]

    # 3) 정렬·log·반환
    valid_trajs.sort(key=lambda t: t.final_score, reverse=True)

    # 4) 로그 — 한 라인에 비교 결과 전체
    _ensure_log_dir()
    _rotate_if_needed()
    try:
        entry = {
            "compared_at": _now_iso(),
            "n_candidates": len(valid_trajs),
            "depth": depth,
            "goal_context_preview": goal_context[:200],
            "trajectories": [t.serialize() for t in valid_trajs],
        }
        with open(_ROLLOUT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[rollout] log 실패 (graceful): {e}")

    return valid_trajs


def clear_log() -> None:
    try:
        if os.path.exists(_ROLLOUT_LOG):
            os.remove(_ROLLOUT_LOG)
    except Exception:
        pass


def load_recent_rollouts(max_n: int = 100) -> list:
    """최근 비교 로그 N 건."""
    out: list = []
    if not os.path.exists(_ROLLOUT_LOG):
        return out
    try:
        with open(_ROLLOUT_LOG, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return out
    for line in reversed(lines):
        if len(out) >= max_n:
            break
        try:
            entry = json.loads(line.strip())
            out.append(entry)
        except Exception:
            continue
    return list(reversed(out))
