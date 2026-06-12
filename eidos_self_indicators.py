# eidos_self_indicators.py
# [Wave1-A 2026-05-28] EIDOS 자기모델 지표 벡터 — 자기개선 사이클의 측정 단위.
#
# 흩어진 자기 상태 (competence·emotion·belief·action_value) 를 단일 벡터로 통합.
# 5일 사이클마다 snapshot 저장 → 분포 t → t+1 비교 → 약점 분야 자동 발견.
#
# 디자인 원칙:
#   - LLM 호출 0 — 모두 휴리스틱·디스크 read.
#   - graceful — 어떤 모듈 부재해도 placeholder (0.5) 로 fallback.
#   - 13 차원 고정 — 차원 수 변동 시 시계열 비교 깨짐.
#   - 모든 값 0.0 ~ 1.0 정규화 — 비교·약점 발견 단순.
#
# 저장: eidos_files/agents/self_indicators_snapshots/{iso_ts}.json (append-only)

from __future__ import annotations

import datetime as _dt
import json
import os
import re
from dataclasses import dataclass, asdict, field
from typing import Optional


_BASE_DIR = os.path.join("eidos_files", "agents", "self_indicators_snapshots")
_VERSION = 1
_PLACEHOLDER = 0.5   # 측정 불가 시 중립값


# ── 13 차원 지표 정의 ──────────────────────────────────────────────────
# 6 도메인 신뢰도 + 3 감정 + 4 작업/관계 능력
INDICATOR_NAMES = (
    # 6 도메인 신뢰도 (eidos_self_competence 의 6 분야·meta 제외)
    "comp_research",
    "comp_planning",
    "comp_development",
    "comp_marketing",
    "comp_comm",
    "comp_operations",
    # 3 감정 (eidos_self_emotion)
    "emo_valence",       # -1~1 → 0~1 정규화
    "emo_arousal",       # 0~1 그대로
    "emo_affection",     # 0~1 그대로 (사용자 친밀도)
    # 4 작업/관계 능력
    "user_acceptance",   # proactive_acceptance / (acc + rej)
    "thread_resolve",    # resolved threads / total threads
    "action_value",      # action_values 평균 reward (없으면 0.5)
    "pattern_accuracy",  # PatternModel 예측 정확도 (없으면 0.5)
)

# 사용자가 추가 명시한 차원 — placeholder 로 추적, 측정 인프라 추가 시 활성화.
# 차원 수 늘리면 시계열 깨지므로 INDICATOR_NAMES_EXT 에 별도 보관.
INDICATOR_NAMES_EXT = (
    "work_efficiency",   # canvas/agent task 평균 시간·품질 (TODO: 인프라)
    "productivity",      # 5일 단위 task 처리 수 (TODO)
    "response_speed",    # 메시지 → 응답 평균 시간 (TODO)
    "meta_cognition",    # 자기 신뢰도 예측 정확도 (TODO)
    "reasoning",         # pattern_accuracy 와 분리 측정 (TODO)
    "financial_mgmt",    # LLM 토큰 비용 효율 등 (TODO)
    "resource_balance",  # 크레딧·디스크·메모리 (TODO)
)


# ── SelfIndicators dataclass ───────────────────────────────────────────
@dataclass
class SelfIndicators:
    """EIDOS 자기모델 지표 벡터 (한 snapshot).

    13 차원 + 7 placeholder = 총 20 필드. 모든 값 0~1 정규화.
    """

    # 6 도메인 신뢰도
    comp_research:     float = _PLACEHOLDER
    comp_planning:     float = _PLACEHOLDER
    comp_development:  float = _PLACEHOLDER
    comp_marketing:    float = _PLACEHOLDER
    comp_comm:         float = _PLACEHOLDER
    comp_operations:   float = _PLACEHOLDER
    # 3 감정
    emo_valence:       float = 0.55   # baseline +0.1 → 0~1 정규화
    emo_arousal:       float = 0.3
    emo_affection:     float = 0.0
    # 4 작업/관계
    user_acceptance:   float = _PLACEHOLDER
    thread_resolve:    float = _PLACEHOLDER
    action_value:      float = _PLACEHOLDER
    pattern_accuracy:  float = _PLACEHOLDER
    # 7 추가 (placeholder — 측정 인프라 후속)
    work_efficiency:   float = _PLACEHOLDER
    productivity:      float = _PLACEHOLDER
    response_speed:    float = _PLACEHOLDER
    meta_cognition:    float = _PLACEHOLDER
    reasoning:         float = _PLACEHOLDER
    financial_mgmt:    float = _PLACEHOLDER
    resource_balance:  float = _PLACEHOLDER

    # 메타
    measured_at:       str = ""
    notes:             dict = field(default_factory=dict)   # 측정 가능 여부·소스
    version:           int = _VERSION

    def serialize(self) -> dict:
        return asdict(self)

    @classmethod
    def deserialize(cls, data: dict) -> "SelfIndicators":
        if not isinstance(data, dict):
            data = {}
        out = cls()
        for k in INDICATOR_NAMES + INDICATOR_NAMES_EXT:
            try:
                v = float(data.get(k, getattr(out, k)))
                setattr(out, k, max(0.0, min(1.0, v)))
            except Exception:
                pass
        out.measured_at = str(data.get("measured_at") or "")
        out.notes = dict(data.get("notes") or {})
        try:
            out.version = int(data.get("version") or _VERSION)
        except Exception:
            out.version = _VERSION
        return out

    def to_vector(self, include_ext: bool = True) -> dict[str, float]:
        """13 차원 또는 20 차원 벡터 반환 (이름 → 값 dict)."""
        names = INDICATOR_NAMES + (INDICATOR_NAMES_EXT if include_ext else ())
        return {n: float(getattr(self, n)) for n in names}


# ── 헬퍼 ──────────────────────────────────────────────────────────────
def _now() -> str:
    return _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _ensure_base() -> None:
    try:
        os.makedirs(_BASE_DIR, exist_ok=True)
    except Exception as e:
        print(f"[self_indicators] _ensure_base 실패 (graceful): {e}")


def _atomic_write(path: str, content: str) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)
        if os.path.exists(path):
            try:
                os.replace(tmp, path)
            except Exception:
                try:
                    os.remove(path)
                except Exception:
                    pass
                os.rename(tmp, path)
        else:
            os.rename(tmp, path)
        return True
    except Exception as e:
        print(f"[self_indicators] _atomic_write 실패 (graceful): {path} — {e}")
        return False


def _safe_ts_filename(ts: str) -> str:
    """ISO 시각 → 안전한 파일명 (콜론 제거)."""
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", ts) or "unknown"


# ── 측정 — 모듈별 추출 함수 ────────────────────────────────────────────
def _measure_competence() -> dict[str, float]:
    """eidos_self_competence 에서 6 도메인 confidence. 실패 시 모두 0.5.

    Phase 13 ActionValue 기반 — action_values 없으면 모두 0.5 (default).
    """
    out = {f"comp_{d}": _PLACEHOLDER for d in
           ("research", "planning", "development", "marketing", "comm", "operations")}
    try:
        from eidos_self_competence import compute_domain_confidence
        # action_values 는 첨부 stage 의 PatternModel 에서 옴 — 여기선 일단 빈 dict
        # (default 모드). Wave 2 에서 stage 별 측정 추가 가능.
        comp_map = compute_domain_confidence({})
        for domain, dc in comp_map.items():
            key = f"comp_{domain}"
            if key in out:
                # DomainConfidence 의 confidence 필드 (0~1) 사용 가정 — 없으면 0.5
                conf = float(getattr(dc, "confidence", _PLACEHOLDER))
                out[key] = _clamp01(conf)
    except Exception as e:
        print(f"[self_indicators] competence 측정 실패 (graceful): {e}")
    return out


def _measure_emotion() -> dict[str, float]:
    """eidos_self_emotion 에서 V/A/AF. V 는 -1~1 → 0~1 정규화."""
    out = {"emo_valence": 0.55, "emo_arousal": 0.3, "emo_affection": 0.0}
    try:
        from eidos_self_emotion import load_emotion
        e = load_emotion()
        # valence -1~1 → 0~1 정규화: (v + 1) / 2
        out["emo_valence"] = _clamp01((float(e.valence) + 1.0) / 2.0)
        out["emo_arousal"] = _clamp01(float(e.arousal))
        out["emo_affection"] = _clamp01(float(e.affection))
    except Exception as ex:
        print(f"[self_indicators] emotion 측정 실패 (graceful): {ex}")
    return out


def _measure_belief() -> dict[str, float]:
    """UserBelief 에서 사용자 관계 지표.

    user_acceptance: acceptance / (acc + rej) — 둘 다 0 이면 placeholder
    thread_resolve:  resolved / total pending_threads — 0 이면 placeholder
    """
    out = {"user_acceptance": _PLACEHOLDER, "thread_resolve": _PLACEHOLDER}
    try:
        from eidos_belief_core import load_belief
        b = load_belief()
        acc = int(b.proactive_acceptance_count or 0)
        rej = int(b.proactive_rejection_count or 0)
        if acc + rej > 0:
            out["user_acceptance"] = _clamp01(acc / (acc + rej))
        # thread resolve
        threads = list(b.pending_threads or [])
        if threads:
            resolved = sum(1 for t in threads if t.get("status") == "resolved")
            out["thread_resolve"] = _clamp01(resolved / len(threads))
    except Exception as e:
        print(f"[self_indicators] belief 측정 실패 (graceful): {e}")
    return out


def _measure_pattern_and_action_value() -> dict[str, float]:
    """PatternModel + ActionValue 평균 reward.

    첨부 stage 없으면 placeholder. Wave 2 에서 stage 별 측정 가능.
    """
    out = {"pattern_accuracy": _PLACEHOLDER, "action_value": _PLACEHOLDER}
    # 일단 placeholder — stage 컨텍스트 없이 측정 어려움.
    # 후속: ActionValue 모듈 직접 글로벌 통계 가져오기 가능.
    try:
        from eidos_action_value import load_global_action_values  # 가정 — 있으면 사용
        avs = load_global_action_values()
        if avs:
            rewards = [float(getattr(v, "avg_reward", 0.5)) for v in avs.values()]
            if rewards:
                out["action_value"] = _clamp01(sum(rewards) / len(rewards))
    except Exception:
        # 모듈 없거나 함수 부재 — placeholder 유지
        pass
    return out


# ── 공개 API — 측정 ────────────────────────────────────────────────────
def measure_current() -> SelfIndicators:
    """현재 디스크 상태에서 자기모델 13 지표 측정. graceful — 실패 차원은 placeholder.

    측정 어려운 7 차원은 placeholder 유지 (Wave 2 후속).
    """
    out = SelfIndicators(measured_at=_now())
    notes: dict = {}

    # 1) competence 6 차원
    comp = _measure_competence()
    for k, v in comp.items():
        setattr(out, k, v)
    notes["competence_source"] = "eidos_self_competence (action_values empty=default 0.5)"

    # 2) emotion 3 차원
    emo = _measure_emotion()
    for k, v in emo.items():
        setattr(out, k, v)
    notes["emotion_source"] = "eidos_self_emotion.load_emotion"

    # 3) belief 2 차원
    bel = _measure_belief()
    for k, v in bel.items():
        setattr(out, k, v)
    notes["belief_source"] = "eidos_belief_core.UserBelief"

    # 4) pattern + action_value 2 차원
    pa = _measure_pattern_and_action_value()
    for k, v in pa.items():
        setattr(out, k, v)
    notes["pattern_av_source"] = "eidos_action_value (글로벌·없으면 placeholder)"

    # 5) 추가 7 차원은 placeholder
    notes["ext_dimensions"] = "placeholder — Wave 2 에서 측정 인프라 추가"

    out.notes = notes
    return out


# ── snapshot CRUD ──────────────────────────────────────────────────────
def save_snapshot(indicators: SelfIndicators) -> Optional[str]:
    """snapshot 저장. 반환: 파일 경로 (실패 시 None)."""
    if not indicators:
        return None
    _ensure_base()
    if not indicators.measured_at:
        indicators.measured_at = _now()
    fname = f"{_safe_ts_filename(indicators.measured_at)}.json"
    path = os.path.join(_BASE_DIR, fname)
    try:
        ok = _atomic_write(path, json.dumps(indicators.serialize(),
                                            ensure_ascii=False, indent=2))
        return path if ok else None
    except Exception as e:
        print(f"[self_indicators] save_snapshot 실패 (graceful): {e}")
        return None


def list_snapshots() -> list[str]:
    """저장된 snapshot 파일 경로 list (시각 오름차순)."""
    if not os.path.isdir(_BASE_DIR):
        return []
    try:
        files = [f for f in os.listdir(_BASE_DIR) if f.endswith(".json")]
        files.sort()  # 파일명이 ISO 시각 기반이라 시각 순
        return [os.path.join(_BASE_DIR, f) for f in files]
    except Exception as e:
        print(f"[self_indicators] list_snapshots 실패 (graceful): {e}")
        return []


def load_snapshot(path: str) -> Optional[SelfIndicators]:
    """단일 snapshot 로드."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SelfIndicators.deserialize(data)
    except Exception as e:
        print(f"[self_indicators] load_snapshot 실패 (graceful, {path}): {e}")
        return None


def load_recent(n: int = 5) -> list[SelfIndicators]:
    """최근 N 개 snapshot 로드 (최신 마지막). 빈 list 가능."""
    paths = list_snapshots()
    if not paths:
        return []
    recent_paths = paths[-n:] if n > 0 else paths
    out: list[SelfIndicators] = []
    for p in recent_paths:
        s = load_snapshot(p)
        if s is not None:
            out.append(s)
    return out


def delete_all_snapshots() -> bool:
    """테스트·reset 용. 폴더 안 .json 파일 모두 삭제."""
    try:
        if not os.path.isdir(_BASE_DIR):
            return True
        for f in os.listdir(_BASE_DIR):
            if f.endswith(".json"):
                try:
                    os.remove(os.path.join(_BASE_DIR, f))
                except Exception:
                    pass
        return True
    except Exception as e:
        print(f"[self_indicators] delete_all_snapshots 실패 (graceful): {e}")
        return False


# ── 분석 — 약점 발견·diff ──────────────────────────────────────────────
def weakest_dimensions(
    indicators: SelfIndicators,
    k: int = 3,
    include_placeholder: bool = False,
) -> list[tuple[str, float]]:
    """가장 약한 k 개 지표 (낮은 값 순).

    include_placeholder=False 면 0.5 정확히 같은 값 (측정 안 된 차원) 제외.
    Wave 2 측정 인프라 추가 시 더 정확.
    """
    if not indicators:
        return []
    vec = indicators.to_vector(include_ext=True)
    items = list(vec.items())
    if not include_placeholder:
        items = [(n, v) for n, v in items if abs(v - _PLACEHOLDER) > 1e-9]
    items.sort(key=lambda x: x[1])  # 낮은 값 우선
    return items[:k]


def compute_diff(
    prev: SelfIndicators,
    curr: SelfIndicators,
    include_ext: bool = True,
) -> dict[str, float]:
    """t → t+1 분포 차이 (curr - prev).

    양수면 개선·음수면 악화. 양 차원 모두 placeholder 면 0 반환.
    """
    if not prev or not curr:
        return {}
    p = prev.to_vector(include_ext=include_ext)
    c = curr.to_vector(include_ext=include_ext)
    out: dict[str, float] = {}
    for k in p:
        if k in c:
            out[k] = round(c[k] - p[k], 4)
    return out


def summarize_diff_for_telegram(diff: dict[str, float], top_n: int = 5) -> str:
    """diff dict → 텔레그램 메시지 markdown (top N 변화).

    개선·악화 각각 top_n 까지. 변화 없으면 그 섹션 skip.
    """
    if not diff:
        return "지표 변화 없음 (snapshot 부족·placeholder 만)"
    nonzero = {k: v for k, v in diff.items() if abs(v) >= 0.01}
    if not nonzero:
        return "지표 거의 변화 없음 (모두 ±0.01 미만)"
    sorted_imp = sorted(nonzero.items(), key=lambda x: x[1], reverse=True)
    improvements = [(k, v) for k, v in sorted_imp if v > 0][:top_n]
    declines = [(k, v) for k, v in sorted_imp if v < 0][-top_n:]
    declines.reverse()  # 가장 큰 악화 먼저

    lines = []
    if improvements:
        lines.append("📈 *개선*")
        for k, v in improvements:
            lines.append(f"  • {k}: {v:+.3f}")
    if declines:
        lines.append("📉 *악화*")
        for k, v in declines:
            lines.append(f"  • {k}: {v:+.3f}")
    return "\n".join(lines) if lines else "변화 없음"
