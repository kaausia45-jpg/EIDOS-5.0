# eidos_prediction_engine.py
# [Wave4 2026-05-28] Forward predictive model — AGI 지향 prediction-curiosity loop 의 핵심.
#
# 매 action 실행 전: LLM 으로 expected outcome 예측 (자연어 + indicator delta + confidence)
# 매 action 실행 후: 실제 outcome 측정 → prediction_error 계산 → jsonl 누적
#
# 학술 배경:
#   - Schmidhuber 의 Curiosity-driven learning (1991): prediction_error = intrinsic reward
#   - Pathak et al. Intrinsic Curiosity Module (ICM, 2017): forward model + inverse model
#   - Friston 의 Free Energy Principle: agent 가 surprise (prediction error) 최소화
#
# EIDOS 적용:
#   - LLM 이 forward model 역할 (frozen weight 라서 ICM 처럼 weight 학습은 X)
#   - 그러나 prediction_error 자체는 file 에 누적 → curiosity_driver 가 활용
#   - High-error 도메인 = "EIDOS 가 아직 잘 모르는 곳" → 자율 탐색 target
#
# 저장 위치: eidos_files/prediction_log.jsonl (append-only)
# 한계: LLM 호출 1회 추가 → tick 당 ~$0.0002 (Gemini Flash). 매 tick 켜기엔 부담.
# default OFF — settings.prediction_engine_enabled=true 로 활성화.

from __future__ import annotations

import asyncio
import json
import os
import datetime as _dt
from dataclasses import dataclass, field, asdict
from typing import Optional


# ── 저장 경로 ─────────────────────────────────────────────────────────
_LOG_DIR = os.path.join("eidos_files", "predictions")
_LOG_PATH = os.path.join(_LOG_DIR, "prediction_log.jsonl")
_MAX_LOG_BYTES = 2 * 1024 * 1024  # 2MB rotation


# ── dataclasses ──────────────────────────────────────────────────────
@dataclass
class Prediction:
    """Action 실행 전 LLM 이 생성한 forward prediction."""
    prediction_id: str = ""
    action_id: str = ""
    args_summary: str = ""
    stage_id: str = ""
    tick_num: int = 0
    created_at: str = ""

    # LLM 출력
    expected_result_text: str = ""        # 자연어 예측
    expected_indicators: dict = field(default_factory=dict)   # {indicator_name: predicted_delta}
    confidence: float = 0.5               # 0~1 — LLM 자기 확신도
    surprise_threshold: float = 0.3       # 이 이상 delta 면 surprising

    # diagnostic
    raw_llm: str = ""

    def serialize(self) -> dict:
        return asdict(self)


@dataclass
class PredictionError:
    """예측 vs 실제 비교 결과."""
    prediction_id: str = ""
    action_id: str = ""
    stage_id: str = ""
    tick_num: int = 0
    recorded_at: str = ""

    # 비교
    expected_result_text: str = ""
    actual_result_text: str = ""
    indicator_errors: dict = field(default_factory=dict)   # {indicator: |predicted - actual|}
    semantic_match: float = 0.5           # 0~1 — 자연어 match (LLM 평가·옵션·default 중간)

    # 종합 점수
    error_score: float = 0.0              # 0~1 — 높을수록 surprising
    surprised: bool = False               # error_score > surprise_threshold

    def serialize(self) -> dict:
        return asdict(self)


# ── LLM 시스템 프롬프트 ──────────────────────────────────────────────
_PREDICT_SYSTEM = """\
너는 EIDOS 의 forward predictive model 이다. EIDOS 가 다음 action 을 실행하기 전,
**무엇이 일어날지** 예측해라.

예측 원칙:
- **결과를 예측해라** — 칭찬·아부 X. 보수적·캘리브레이션 중요.
- **불확실하면 confidence 0.3 이하** + surprise_threshold 0.5 이상.
- **잘 아는 도메인이면 confidence 0.8+** + surprise_threshold 0.2 이하.
- 예측은 **자연어 1~2 문장** + 핵심 indicator 의 expected_delta (있다면).

JSON 만 출력. 코드블록·설명·라벨 X:
{
  "expected_result_text": "<1~2문장 자연어 예측>",
  "expected_indicators": {"indicator_name": <-1.0~+1.0 예측 delta>},
  "confidence": 0.0~1.0,
  "surprise_threshold": 0.0~1.0,
  "reasoning": "<짧은 근거 한 줄·디버깅용>"
}
"""


_EVAL_SYSTEM = """\
너는 EIDOS 의 prediction-vs-actual 평가자다. 두 자연어 (예측·실제) 의
의미적 일치도를 0.0~1.0 으로 매겨라.

평가 원칙:
- 1.0 = 거의 동일 (단어만 다름)
- 0.7 = 핵심 outcome 일치·세부 다름
- 0.5 = 부분 일치 (방향은 맞지만 정도 다름)
- 0.3 = 거의 다름 (다른 outcome)
- 0.0 = 완전 반대 또는 무관

JSON 만 출력:
{
  "semantic_match": 0.0~1.0,
  "reason": "<짧은 한 줄>"
}
"""


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
    """로그 2MB 넘으면 .bak 으로 회전."""
    try:
        if not os.path.exists(_LOG_PATH):
            return
        if os.path.getsize(_LOG_PATH) > _MAX_LOG_BYTES:
            bak = _LOG_PATH + f".{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
            try:
                os.rename(_LOG_PATH, bak)
            except Exception:
                pass
    except Exception:
        pass


# ── 핵심: 예측 생성 ───────────────────────────────────────────────────
async def predict_action_outcome_async(
    action_id: str,
    args: Optional[dict] = None,
    stage_id: str = "",
    tick_num: int = 0,
    belief_context: Optional[dict] = None,
    history_brief: str = "",
    timeout_sec: float = 8.0,
) -> Prediction:
    """Action 실행 직전 LLM 으로 expected outcome 예측. LLM 실패 시 빈 Prediction."""
    pred = Prediction(
        prediction_id=_new_id(),
        action_id=action_id or "unknown",
        args_summary=_format_args(args or {}),
        stage_id=stage_id,
        tick_num=tick_num,
        created_at=_now_iso(),
    )
    try:
        from llm_module import get_llm_response_async
    except Exception:
        pred.expected_result_text = "(LLM 모듈 부재)"
        return pred

    # belief context — actor·indicator 상위 5개만
    ctx_lines = []
    if isinstance(belief_context, dict) and belief_context:
        for k, v in list(belief_context.items())[:8]:
            try:
                ctx_lines.append(f"  - {k}: {str(v)[:60]}")
            except Exception:
                continue
    ctx_block = "\n".join(ctx_lines) if ctx_lines else "(belief 컨텍스트 없음)"

    # [Wave4-B 2026-05-28] calibration hint inject — 과거 forward/inverse 통계
    # 로 자기 confidence 보정. graceful — 없으면 빈 string.
    calib_hint = ""
    try:
        from eidos_self_calibration import (
            compute_calibration_map, calibration_hint_for_action,
        )
        cm = compute_calibration_map(min_samples=3)
        calib_hint = calibration_hint_for_action(action_id, cm)
    except Exception:
        calib_hint = ""

    user_prompt = (
        f"[action_id] {action_id}\n"
        f"[args] {pred.args_summary[:300]}\n"
        f"[stage] {stage_id[:32]}\n"
        f"[현재 belief 컨텍스트]\n{ctx_block}\n"
        + (f"\n[최근 history]\n{history_brief[:400]}" if history_brief else "")
        + (f"\n{calib_hint}" if calib_hint else "")
        + "\n\n위 상황에서 이 action 실행하면 뭐가 일어날까? JSON 만 출력."
    )

    raw = ""
    try:
        raw = await asyncio.wait_for(
            get_llm_response_async(
                user_prompt,
                max_tokens=1024,
                system_prompt=_PREDICT_SYSTEM,
                response_mime_type="application/json",
            ),
            timeout=timeout_sec,
        )
        pred.raw_llm = (raw or "")[:1000]
    except asyncio.TimeoutError:
        pred.expected_result_text = "(LLM timeout)"
        return pred
    except Exception as e:
        pred.expected_result_text = f"(LLM 실패: {str(e)[:80]})"
        return pred

    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return pred
    except Exception:
        return pred

    pred.expected_result_text = str(data.get("expected_result_text", ""))[:400]
    ei = data.get("expected_indicators")
    if isinstance(ei, dict):
        out = {}
        for k, v in ei.items():
            try:
                out[str(k)[:40]] = max(-1.0, min(1.0, float(v)))
            except Exception:
                continue
        pred.expected_indicators = out
    try:
        pred.confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
    except Exception:
        pred.confidence = 0.5
    try:
        pred.surprise_threshold = max(
            0.0, min(1.0, float(data.get("surprise_threshold", 0.3)))
        )
    except Exception:
        pred.surprise_threshold = 0.3
    return pred


def _format_args(args: dict) -> str:
    """args dict 짧은 요약."""
    if not args:
        return ""
    parts = []
    for k, v in list(args.items())[:5]:
        try:
            parts.append(f"{k}={str(v)[:40]}")
        except Exception:
            continue
    return ", ".join(parts)


# ── 핵심: 오차 계산 + 기록 ─────────────────────────────────────────────
async def record_prediction_async(
    prediction: Prediction,
    actual_result_text: str = "",
    actual_indicators: Optional[dict] = None,
    skip_llm_eval: bool = False,
    timeout_sec: float = 8.0,
) -> PredictionError:
    """실제 outcome 받아서 prediction_error 계산 + 로그 append.

    Args:
      prediction: predict_action_outcome_async 의 결과
      actual_result_text: 실제 일어난 일 (자연어·execute_eidos_action 의 result)
      actual_indicators: 실제 indicator 변화 (옵션·tick belief_updates 등)
      skip_llm_eval: True 면 LLM semantic_match 평가 생략 (속도 모드·default 0.5 사용)
    """
    err = PredictionError(
        prediction_id=prediction.prediction_id,
        action_id=prediction.action_id,
        stage_id=prediction.stage_id,
        tick_num=prediction.tick_num,
        recorded_at=_now_iso(),
        expected_result_text=prediction.expected_result_text,
        actual_result_text=(actual_result_text or "")[:400],
    )

    # 1. Indicator 오차 — 수치 비교
    ind_err: dict = {}
    if prediction.expected_indicators and isinstance(actual_indicators, dict):
        for k, expected_delta in prediction.expected_indicators.items():
            try:
                actual_delta = float(actual_indicators.get(k, 0.0))
                ind_err[k] = abs(float(expected_delta) - actual_delta)
            except Exception:
                continue
    err.indicator_errors = ind_err

    # 2. Semantic 오차 — LLM 평가 (옵션)
    sem_match = 0.5
    if (not skip_llm_eval
            and prediction.expected_result_text
            and actual_result_text):
        try:
            sem_match = await _eval_semantic_match_async(
                prediction.expected_result_text,
                actual_result_text,
                timeout_sec=timeout_sec,
            )
        except Exception:
            pass
    err.semantic_match = sem_match

    # 3. 종합 error_score — semantic mismatch + indicator deviation 평균
    sem_err = 1.0 - sem_match
    ind_err_avg = (
        sum(ind_err.values()) / len(ind_err) if ind_err else 0.0
    )
    # 가중 — semantic 60%·indicator 40% (지표 없으면 semantic 만)
    if ind_err:
        err.error_score = 0.6 * sem_err + 0.4 * ind_err_avg
    else:
        err.error_score = sem_err
    err.error_score = max(0.0, min(1.0, err.error_score))
    err.surprised = err.error_score > prediction.surprise_threshold

    # 4. 디스크 append
    _ensure_log_dir()
    _rotate_if_needed()
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            entry = {
                "prediction": prediction.serialize(),
                "error": err.serialize(),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[prediction] 로그 append 실패 (graceful): {e}")
    return err


async def _eval_semantic_match_async(
    expected: str, actual: str, timeout_sec: float = 8.0,
) -> float:
    """LLM 으로 두 자연어 의미 일치도 0~1 평가."""
    try:
        from llm_module import get_llm_response_async
    except Exception:
        return 0.5
    prompt = (
        f"[예측] {expected[:300]}\n"
        f"[실제] {actual[:300]}\n\n"
        "두 자연어의 의미 일치도 (0.0~1.0). JSON 만 출력."
    )
    try:
        raw = await asyncio.wait_for(
            get_llm_response_async(
                prompt,
                max_tokens=1024,
                system_prompt=_EVAL_SYSTEM,
                response_mime_type="application/json",
            ),
            timeout=timeout_sec,
        )
        data = json.loads(raw)
        if isinstance(data, dict):
            return max(0.0, min(1.0, float(data.get("semantic_match", 0.5))))
    except Exception:
        pass
    return 0.5


# ── 로그 읽기 (curiosity_driver 에서 사용) ─────────────────────────────
def load_recent_predictions(
    max_n: int = 500,
    since_hours: Optional[float] = None,
) -> list[dict]:
    """최근 N 건 prediction+error 엔트리 로드.

    since_hours 가 주어지면 그만큼 이내 것만 필터.
    """
    out: list[dict] = []
    if not os.path.exists(_LOG_PATH):
        return out
    try:
        with open(_LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return out

    cutoff = None
    if since_hours is not None and since_hours > 0:
        cutoff = _dt.datetime.now() - _dt.timedelta(hours=since_hours)

    for line in reversed(lines):
        if len(out) >= max_n:
            break
        try:
            entry = json.loads(line.strip())
            if not isinstance(entry, dict):
                continue
            err = entry.get("error") or {}
            if cutoff is not None:
                try:
                    rec_at = _dt.datetime.fromisoformat(
                        str(err.get("recorded_at", ""))
                    )
                    if rec_at < cutoff:
                        break  # reverse 순회중·이후는 모두 더 옛날
                except Exception:
                    pass
            out.append(entry)
        except Exception:
            continue
    return list(reversed(out))


def clear_log() -> None:
    """테스트용 — 로그 초기화."""
    try:
        if os.path.exists(_LOG_PATH):
            os.remove(_LOG_PATH)
    except Exception:
        pass
