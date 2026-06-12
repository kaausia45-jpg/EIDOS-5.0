# eidos_counterfactual.py
# [Wave5-A 2026-05-28] 반사실 추론 — Pearl 의 do-calculus 약식 구현.
#
# Forward model (Wave4) + 과거 실제 outcome 을 결합:
#   - 실제: (s_t, a_t) → s_{t+1}_actual (이미 일어난 일)
#   - 반사실: "만약 a_t' 였다면?" → forward model → s_{t+1}_counterfactual
#   - 비교: s_{t+1}_actual vs s_{t+1}_counterfactual
#       - 차이 큼 = action 이 진짜 difference 만든 (causal)
#       - 차이 작음 = action 무관 (그 결과는 다른 원인)
#
# 이게 inverse_model 과 다른 점:
#   - inverse: "이 변화의 원인 행동이 뭐냐" (forensic)
#   - counterfactual: "다른 행동했으면 어떻게 됐을까" (interventional)
#   - 둘이 합쳐져야 causal structure 가 보임
#
# EIDOS 적용:
#   - 매 cycle (5일·또는 force) 마다 최근 N tick 골라서 counterfactual 분석
#   - action 별 causal_strength 누적 → "이 action 정말 효과 있나" 정량화
#   - self_calibration·self_competence 와 결합 → 인과 기반 의사결정
#
# 비용: cycle 당 N tick × 1~2 LLM call. N=20 → ~$0.005/cycle.

from __future__ import annotations

import asyncio
import json
import os
import datetime as _dt
from dataclasses import dataclass, field, asdict
from typing import Optional


_LOG_DIR = os.path.join("eidos_files", "predictions")
_CF_LOG = os.path.join(_LOG_DIR, "counterfactual_log.jsonl")
_MAX_LOG_BYTES = 2 * 1024 * 1024


@dataclass
class CounterfactualQuery:
    """과거 tick + counterfactual 질문 + LLM 시뮬 결과."""
    query_id: str = ""
    source_tick_id: str = ""           # 원본 tick prediction_id 또는 inverse_id

    # 원본
    original_action_id: str = ""
    original_args_summary: str = ""
    state_before_text: str = ""
    state_after_actual_text: str = ""
    actual_indicators: dict = field(default_factory=dict)

    # 반사실
    counterfactual_action_id: str = ""
    counterfactual_args: dict = field(default_factory=dict)
    predicted_alternate_state: str = ""
    predicted_alternate_indicators: dict = field(default_factory=dict)
    cf_confidence: float = 0.0

    # 비교
    outcome_differs: bool = False     # actual != counterfactual
    diff_magnitude: float = 0.0       # 0~1·indicator·semantic 합산
    causal_inference: str = ""        # LLM 의 한 줄 해석
    created_at: str = ""

    raw_llm: str = ""

    def serialize(self) -> dict:
        return asdict(self)


@dataclass
class CausalAttribution:
    """action 별 인과 강도 누적 통계."""
    action_id: str = ""
    n_executions: int = 0           # 그 action 이 실제 일어난 횟수
    n_times_mattered: int = 0       # counterfactual 과 actual 결과 다른 횟수
    causal_strength: float = 0.0    # n_mattered / n_executions
    avg_diff_magnitude: float = 0.0
    last_evaluated: str = ""

    def serialize(self) -> dict:
        return asdict(self)


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
        if (os.path.exists(_CF_LOG)
                and os.path.getsize(_CF_LOG) > _MAX_LOG_BYTES):
            bak = _CF_LOG + f".{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
            try:
                os.rename(_CF_LOG, bak)
            except Exception:
                pass
    except Exception:
        pass


# ── LLM 시스템 프롬프트 ─────────────────────────────────────────────────
_CF_ALT_SYSTEM = """\
너는 EIDOS 의 counterfactual reasoning module 이다.

과거에 EIDOS 가 한 action 을 보고, **만약 다른 action 을 했다면 결과가 어떻게
달랐을지** 추론해라.

원칙:
- counterfactual_action 은 action_repertoire 중 1개·원본과 달라야 함
- predicted_alternate state 는 자연어 1~2 문장
- predicted_indicators 는 추정 delta (-1.0 ~ +1.0)
- confidence 0~1 — 모를수록 낮게
- causal_inference 는 "원본 action 이 차이를 만들었는지" 한 줄

JSON 만 출력:
{
  "counterfactual_action_id": "<repertoire 중 하나·원본과 달라야>",
  "predicted_alternate_state": "<1~2 문장>",
  "predicted_alternate_indicators": {"indicator": <delta>},
  "confidence": 0.0~1.0,
  "causal_inference": "원본 action 이 차이를 만들었나/아니나·한 줄"
}
"""


# ── 핵심: 단일 tick counterfactual 분석 ────────────────────────────────
async def analyze_tick_counterfactual_async(
    tick_data: dict,
    action_repertoire: list,
    timeout_sec: float = 10.0,
) -> CounterfactualQuery:
    """과거 tick 1개 보고 counterfactual 시뮬·결과 비교.

    Args:
      tick_data: {"prediction": {...}, "error": {...}} (prediction_log entry)
                 또는 호환되는 dict.
      action_repertoire: 가능한 action_id 목록.

    Returns: CounterfactualQuery (LLM 실패 시 빈 결과).
    """
    cf = CounterfactualQuery(
        query_id=_new_id(),
        created_at=_now_iso(),
    )

    # tick_data 에서 핵심 정보 추출
    pred = tick_data.get("prediction") or {}
    err = tick_data.get("error") or {}
    cf.source_tick_id = str(pred.get("prediction_id", ""))
    cf.original_action_id = str(pred.get("action_id", ""))
    cf.original_args_summary = str(pred.get("args_summary", ""))[:200]
    cf.state_before_text = str(pred.get("expected_result_text", ""))[:200]
    cf.state_after_actual_text = str(err.get("actual_result_text", ""))[:200]
    # actual indicators — prediction expected_indicators 에서 actual 적용 시도
    # (정확한 actual 은 없지만 expected 와 큰 차이 없을 거라 가정)
    if isinstance(pred.get("expected_indicators"), dict):
        for k, v in pred["expected_indicators"].items():
            try:
                cf.actual_indicators[k] = float(v)
            except Exception:
                continue

    if not cf.original_action_id or not action_repertoire:
        cf.causal_inference = "(데이터 부족)"
        return cf

    # LLM 호출 — counterfactual action + 시뮬 결과
    try:
        from llm_module import get_llm_response_async
    except Exception:
        cf.causal_inference = "(LLM 모듈 부재)"
        return cf

    rep_filtered = [a for a in action_repertoire if a != cf.original_action_id][:15]
    if not rep_filtered:
        cf.causal_inference = "(repertoire 에 대안 없음)"
        return cf
    rep_block = "\n".join(f"- {a}" for a in rep_filtered)

    prompt = (
        f"[원본 action] {cf.original_action_id}\n"
        f"[원본 args 요약] {cf.original_args_summary[:200]}\n"
        f"[원본 예측 state] {cf.state_before_text[:200]}\n"
        f"[실제 결과] {cf.state_after_actual_text[:200]}\n"
        f"[원본 actual indicators] "
        f"{json.dumps(cf.actual_indicators, ensure_ascii=False)[:200]}\n\n"
        f"[가능한 대안 action] (원본 제외)\n{rep_block}\n\n"
        "원본 대신 위 중 하나를 했다면 어땠을지 시뮬. JSON 만 출력."
    )

    try:
        raw = await asyncio.wait_for(
            get_llm_response_async(
                prompt,
                max_tokens=1024,
                system_prompt=_CF_ALT_SYSTEM,
                response_mime_type="application/json",
            ),
            timeout=timeout_sec,
        )
        cf.raw_llm = (raw or "")[:600]
        data = json.loads(raw)
        if not isinstance(data, dict):
            return cf
    except asyncio.TimeoutError:
        cf.causal_inference = "(LLM timeout)"
        return cf
    except Exception as e:
        cf.causal_inference = f"(LLM 실패: {str(e)[:80]})"
        return cf

    cf.counterfactual_action_id = str(data.get("counterfactual_action_id", ""))[:80]
    cf.predicted_alternate_state = str(data.get("predicted_alternate_state", ""))[:200]
    cf_ind = data.get("predicted_alternate_indicators")
    if isinstance(cf_ind, dict):
        for k, v in cf_ind.items():
            try:
                cf.predicted_alternate_indicators[str(k)[:40]] = max(
                    -1.0, min(1.0, float(v))
                )
            except Exception:
                continue
    try:
        cf.cf_confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    except Exception:
        cf.cf_confidence = 0.0
    cf.causal_inference = str(data.get("causal_inference", ""))[:200]

    # 차이 계산 — indicator delta L1 norm + 자연어 일치 휴리스틱
    ind_diff = 0.0
    n_ind = 0
    all_keys = set(cf.actual_indicators) | set(cf.predicted_alternate_indicators)
    for k in all_keys:
        a = float(cf.actual_indicators.get(k, 0.0))
        b = float(cf.predicted_alternate_indicators.get(k, 0.0))
        ind_diff += abs(a - b)
        n_ind += 1
    ind_diff_avg = ind_diff / n_ind if n_ind > 0 else 0.0

    # 자연어 일치 — 간단 휴리스틱 (LLM 평가는 비용 폭증 회피)
    actual_lower = cf.state_after_actual_text.lower()
    cf_lower = cf.predicted_alternate_state.lower()
    semantic_diff = 0.5  # 기본 중간값
    if actual_lower and cf_lower:
        # 단어 겹침 비율
        actual_words = set(actual_lower.split())
        cf_words = set(cf_lower.split())
        if actual_words and cf_words:
            overlap = len(actual_words & cf_words) / max(
                len(actual_words | cf_words), 1
            )
            semantic_diff = 1.0 - overlap

    # 종합 diff_magnitude — indicator 60% + semantic 40%
    cf.diff_magnitude = max(0.0, min(1.0, 0.6 * ind_diff_avg + 0.4 * semantic_diff))
    cf.outcome_differs = cf.diff_magnitude > 0.25  # threshold

    # log append
    _ensure_log_dir()
    _rotate_if_needed()
    try:
        with open(_CF_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(cf.serialize(), ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[counterfactual] log 실패 (graceful): {e}")
    return cf


# ── 최근 N tick 일괄 분석 ──────────────────────────────────────────────
async def evaluate_recent_ticks_async(
    n_ticks: int = 20,
    action_repertoire: Optional[list] = None,
    since_hours: float = 168.0,
) -> dict:
    """최근 N tick prediction_log 에서 counterfactual 분석 → CausalAttribution 누적.

    Returns: {action_id: CausalAttribution}.
    """
    try:
        from eidos_prediction_engine import load_recent_predictions
        entries = load_recent_predictions(max_n=n_ticks, since_hours=since_hours)
    except Exception as e:
        print(f"[counterfactual] prediction log 로드 실패: {e}")
        return {}

    if not entries:
        return {}

    # action_repertoire 기본값 — entries 에서 등장한 모든 action_id (best-effort)
    if not action_repertoire:
        action_repertoire = sorted({
            str((e.get("prediction") or {}).get("action_id", ""))
            for e in entries
            if (e.get("prediction") or {}).get("action_id")
        })
    if not action_repertoire:
        return {}

    # 각 tick counterfactual — 동시 gather (n_ticks 비용 부담 가능·N 제한 권장)
    cf_tasks = [
        analyze_tick_counterfactual_async(e, action_repertoire)
        for e in entries
    ]
    cfs = await asyncio.gather(*cf_tasks, return_exceptions=True)

    # action 별 집계
    by_action: dict = {}
    for cf in cfs:
        if not isinstance(cf, CounterfactualQuery):
            continue
        aid = cf.original_action_id
        if not aid:
            continue
        if aid not in by_action:
            by_action[aid] = CausalAttribution(
                action_id=aid, last_evaluated=_now_iso(),
            )
        attr = by_action[aid]
        attr.n_executions += 1
        if cf.outcome_differs:
            attr.n_times_mattered += 1
        # 누적 평균 diff
        attr.avg_diff_magnitude = (
            (attr.avg_diff_magnitude * (attr.n_executions - 1)
             + cf.diff_magnitude)
            / attr.n_executions
        )

    # causal_strength 계산
    for attr in by_action.values():
        attr.causal_strength = (
            attr.n_times_mattered / attr.n_executions
            if attr.n_executions > 0 else 0.0
        )
    return by_action


# ── 로그 읽기 ─────────────────────────────────────────────────────────
def load_recent_counterfactuals(max_n: int = 200) -> list:
    """최근 counterfactual log."""
    out: list = []
    if not os.path.exists(_CF_LOG):
        return out
    try:
        with open(_CF_LOG, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return out
    for line in reversed(lines):
        if len(out) >= max_n:
            break
        try:
            out.append(json.loads(line.strip()))
        except Exception:
            continue
    return list(reversed(out))


def aggregate_causal_attribution(
    log_entries: Optional[list] = None,
    min_samples: int = 3,
) -> dict:
    """log 에서 action 별 attribution 통계 재구성 (재실행 없이 누적 분석용).

    Returns: {action_id: CausalAttribution}.
    """
    if log_entries is None:
        log_entries = load_recent_counterfactuals(max_n=500)
    if not log_entries:
        return {}
    by_action: dict = {}
    for entry in log_entries:
        aid = str(entry.get("original_action_id", ""))
        if not aid:
            continue
        if aid not in by_action:
            by_action[aid] = {"n": 0, "n_diff": 0, "diff_sum": 0.0}
        by_action[aid]["n"] += 1
        if bool(entry.get("outcome_differs", False)):
            by_action[aid]["n_diff"] += 1
        try:
            by_action[aid]["diff_sum"] += float(entry.get("diff_magnitude", 0.0))
        except Exception:
            continue
    out: dict = {}
    for aid, d in by_action.items():
        if d["n"] < min_samples:
            continue
        out[aid] = CausalAttribution(
            action_id=aid,
            n_executions=d["n"],
            n_times_mattered=d["n_diff"],
            causal_strength=d["n_diff"] / d["n"] if d["n"] > 0 else 0.0,
            avg_diff_magnitude=d["diff_sum"] / d["n"] if d["n"] > 0 else 0.0,
            last_evaluated=_now_iso(),
        )
    return out


def clear_log() -> None:
    try:
        if os.path.exists(_CF_LOG):
            os.remove(_CF_LOG)
    except Exception:
        pass
