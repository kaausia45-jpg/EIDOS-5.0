# eidos_inverse_model.py
# [Wave4-B 2026-05-28] Inverse model — (state_before, state_after) → 어떤 action 일어났나.
#
# Pathak et al. (2017) Intrinsic Curiosity Module 의 inverse model 부분.
# Forward model 만 있으면 noise·random world change 도 prediction error 로 학습돼버려서
# agent 가 "TV 보는 데 빠짐" (noisy-TV problem). Inverse model 이 있으면:
#
#   - inverse 가 잘 맞춤 = state 차이가 agent action 으로 설명됨 = controllable
#   - inverse 가 못 맞춤 = state 차이가 noise·외부 = uncontrollable
#
# 즉 EIDOS 가 자기 행동의 **진짜 effect** 를 식별. 노이즈에 끌려가지 않음.
#
# EIDOS 적용:
#   - 매 tick 전후 belief snapshot 캡처
#   - LLM 이 (before, after, repertoire) 보고 어떤 action 일어났는지 추측
#   - 실제 action 과 매칭 여부 기록
#   - action 별 inverse_accuracy 누적 → controllable_score
#
# 비용: LLM 1 호출 추가 (~$0.0002/tick). Forward + Inverse = 합쳐 ~$0.0004.
# Default OFF — settings.inverse_model_enabled=true.

from __future__ import annotations

import asyncio
import json
import os
import datetime as _dt
from dataclasses import dataclass, field, asdict
from typing import Optional


_LOG_DIR = os.path.join("eidos_files", "predictions")
_INVERSE_LOG = os.path.join(_LOG_DIR, "inverse_log.jsonl")
_MAX_LOG_BYTES = 2 * 1024 * 1024


# ── dataclasses ──────────────────────────────────────────────────────
@dataclass
class StateSnapshot:
    """tick 전후 belief 의 간소 표현 — inverse model 에 줄 컨텍스트."""
    actor_indicators: dict = field(default_factory=dict)   # {actor: {indicator: value}}
    work_state: str = ""
    recent_topics: list = field(default_factory=list)
    snapshot_at: str = ""

    def serialize(self) -> dict:
        return asdict(self)

    def diff(self, other: "StateSnapshot") -> dict:
        """이 → other 로 변화한 indicator 만 추출 (signed delta)."""
        out: dict = {}
        for actor, inds in (self.actor_indicators or {}).items():
            other_inds = (other.actor_indicators or {}).get(actor) or {}
            actor_diff = {}
            for k, v in (inds or {}).items():
                try:
                    before = float(v)
                    after = float(other_inds.get(k, before))
                    if abs(after - before) > 1e-6:
                        actor_diff[k] = round(after - before, 4)
                except Exception:
                    continue
            if actor_diff:
                out[actor] = actor_diff
        # 새로 등장한 actor·indicator
        for actor, inds in (other.actor_indicators or {}).items():
            if actor not in (self.actor_indicators or {}):
                if isinstance(inds, dict):
                    out[actor] = {k: round(float(v), 4)
                                  for k, v in inds.items()
                                  if isinstance(v, (int, float))}
        return out


@dataclass
class InverseInference:
    """LLM 의 inverse 추론 결과."""
    inference_id: str = ""
    stage_id: str = ""
    tick_num: int = 0
    recorded_at: str = ""

    actual_action_id: str = ""     # 실제 일어난 action
    inferred_action_id: str = ""   # LLM 추측
    confidence: float = 0.0
    reason: str = ""

    state_diff: dict = field(default_factory=dict)
    repertoire_size: int = 0

    # 평가
    correct: bool = False
    raw_llm: str = ""

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
        if (os.path.exists(_INVERSE_LOG)
                and os.path.getsize(_INVERSE_LOG) > _MAX_LOG_BYTES):
            bak = _INVERSE_LOG + f".{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
            try:
                os.rename(_INVERSE_LOG, bak)
            except Exception:
                pass
    except Exception:
        pass


# ── belief → snapshot 변환 ─────────────────────────────────────────────
def beliefs_to_snapshot(beliefs: Optional[dict]) -> StateSnapshot:
    """agent_runner 의 beliefs dict → StateSnapshot (간소).

    actor 별 indicator 값만 추출. 비교에 필요한 핵심 표현.
    """
    snap = StateSnapshot(snapshot_at=_now_iso())
    if not isinstance(beliefs, dict):
        return snap
    actors = beliefs.get("actors") or {}
    if isinstance(actors, dict):
        for name, a in actors.items():
            if not isinstance(a, dict):
                continue
            ind = a.get("indicators")
            if isinstance(ind, dict) and ind:
                clean = {}
                for k, v in ind.items():
                    try:
                        clean[str(k)[:40]] = float(v)
                    except Exception:
                        continue
                if clean:
                    snap.actor_indicators[str(name)[:40]] = clean
    snap.work_state = str(beliefs.get("work_state") or "")
    rt = beliefs.get("recent_topics") or []
    if isinstance(rt, list):
        snap.recent_topics = [str(x)[:60] for x in rt[:5]]
    return snap


# ── LLM 시스템 프롬프트 ─────────────────────────────────────────────────
_INVERSE_SYSTEM = """\
너는 EIDOS 의 inverse dynamics model 이다.

두 시점 belief snapshot (before, after) 의 차이를 보고, EIDOS 가 **어떤 action 을
실행했길래** 이런 변화가 일어났을지 추측해라.

원칙:
- 주어진 action_repertoire 중 1개만 선택
- 확실하면 confidence 0.8+·모호하면 0.4 이하 + best guess
- state 변화가 미미하면 (거의 변화 없음) "eidos.meta.observe" 또는 "eidos.meta.no_op" 류
- 변화가 노이즈처럼 보이면 confidence 0.2 이하

JSON 만 출력:
{
  "inferred_action_id": "<repertoire 중 하나>",
  "confidence": 0.0~1.0,
  "reason": "어떤 단서로 추론했는지 짧게"
}
"""


# ── 핵심: inverse 추론 ────────────────────────────────────────────────
async def infer_action_from_transition_async(
    before: StateSnapshot,
    after: StateSnapshot,
    action_repertoire: list,
    stage_id: str = "",
    tick_num: int = 0,
    timeout_sec: float = 8.0,
) -> InverseInference:
    """state 변화 보고 LLM 이 어떤 action 일어났는지 추측."""
    inf = InverseInference(
        inference_id=_new_id(),
        stage_id=stage_id,
        tick_num=tick_num,
        recorded_at=_now_iso(),
        repertoire_size=len(action_repertoire or []),
    )
    inf.state_diff = before.diff(after)

    if not action_repertoire:
        inf.inferred_action_id = ""
        inf.reason = "(repertoire 비어있음)"
        return inf

    try:
        from llm_module import get_llm_response_async
    except Exception:
        inf.reason = "(LLM 모듈 부재)"
        return inf

    # action_repertoire — 상위 N 개 (LLM 부담)
    rep_list = list(action_repertoire)[:20]
    rep_block = "\n".join(f"- {a}" for a in rep_list)
    diff_block = json.dumps(inf.state_diff, ensure_ascii=False, indent=2)[:1000]
    before_topics = ", ".join(before.recent_topics[:3])
    after_topics = ", ".join(after.recent_topics[:3])

    prompt = (
        f"[action_repertoire]\n{rep_block}\n\n"
        f"[before work_state] {before.work_state}\n"
        f"[after work_state] {after.work_state}\n"
        f"[before recent_topics] {before_topics}\n"
        f"[after recent_topics] {after_topics}\n\n"
        f"[indicator 변화 (signed delta)]\n{diff_block}\n\n"
        "위 변화를 일으켰을 가능성이 가장 높은 action 1개 선택. JSON 만 출력."
    )

    raw = ""
    try:
        raw = await asyncio.wait_for(
            get_llm_response_async(
                prompt,
                max_tokens=1024,
                system_prompt=_INVERSE_SYSTEM,
                response_mime_type="application/json",
            ),
            timeout=timeout_sec,
        )
        inf.raw_llm = (raw or "")[:800]
    except asyncio.TimeoutError:
        inf.reason = "(LLM timeout)"
        return inf
    except Exception as e:
        inf.reason = f"(LLM 실패: {str(e)[:80]})"
        return inf

    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return inf
    except Exception:
        return inf

    inf.inferred_action_id = str(data.get("inferred_action_id", ""))[:80]
    try:
        inf.confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    except Exception:
        inf.confidence = 0.0
    inf.reason = str(data.get("reason", ""))[:200]
    return inf


async def record_inverse_inference_async(
    actual_action_id: str,
    before: StateSnapshot,
    after: StateSnapshot,
    action_repertoire: list,
    stage_id: str = "",
    tick_num: int = 0,
    timeout_sec: float = 8.0,
) -> InverseInference:
    """inverse 추론 + 실제 action 비교 + jsonl 기록."""
    inf = await infer_action_from_transition_async(
        before, after, action_repertoire, stage_id, tick_num, timeout_sec,
    )
    inf.actual_action_id = (actual_action_id or "")[:80]
    inf.correct = bool(
        inf.inferred_action_id
        and inf.actual_action_id
        and inf.inferred_action_id == inf.actual_action_id
    )

    _ensure_log_dir()
    _rotate_if_needed()
    try:
        with open(_INVERSE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(inf.serialize(), ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[inverse_model] log 실패 (graceful): {e}")
    return inf


# ── 로그 읽기 + 통계 ──────────────────────────────────────────────────
def load_recent_inverses(
    max_n: int = 500, since_hours: Optional[float] = None,
) -> list[dict]:
    """최근 inverse 추론 로그 로드."""
    out: list[dict] = []
    if not os.path.exists(_INVERSE_LOG):
        return out
    try:
        with open(_INVERSE_LOG, "r", encoding="utf-8") as f:
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
            if cutoff is not None:
                try:
                    rec_at = _dt.datetime.fromisoformat(str(entry.get("recorded_at", "")))
                    if rec_at < cutoff:
                        break
                except Exception:
                    pass
            out.append(entry)
        except Exception:
            continue
    return list(reversed(out))


def compute_action_controllability(
    log_entries: Optional[list] = None,
    min_samples: int = 3,
    since_hours: float = 168.0,
) -> dict:
    """action_id 별 controllability_score (inverse_accuracy 평균).

    높은 점수 = 그 action 이 detectable effect 가짐·진짜 controllable.
    낮은 점수 = effect 가 noise 에 묻힘·실효 없음·또는 LLM 추론 한계.

    Returns: {action_id: {accuracy, n_samples, avg_confidence}}
    """
    if log_entries is None:
        log_entries = load_recent_inverses(max_n=2000, since_hours=since_hours)
    by_action: dict = {}
    for entry in log_entries:
        aid = str(entry.get("actual_action_id", ""))
        if not aid:
            continue
        if aid not in by_action:
            by_action[aid] = {"correct": 0, "n": 0, "conf_sum": 0.0}
        by_action[aid]["n"] += 1
        if bool(entry.get("correct", False)):
            by_action[aid]["correct"] += 1
        try:
            by_action[aid]["conf_sum"] += float(entry.get("confidence", 0.0))
        except Exception:
            pass
    out: dict = {}
    for aid, data in by_action.items():
        n = data["n"]
        if n < min_samples:
            continue
        out[aid] = {
            "accuracy": data["correct"] / n,
            "n_samples": n,
            "avg_confidence": data["conf_sum"] / n,
        }
    return out


def clear_log() -> None:
    """테스트용."""
    try:
        if os.path.exists(_INVERSE_LOG):
            os.remove(_INVERSE_LOG)
    except Exception:
        pass
