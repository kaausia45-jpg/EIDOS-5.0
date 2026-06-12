# eidos_self_goals.py
# [Wave1-B 2026-05-28] EIDOS 자기 목표 자동 생성 + 진척 평가 + 폐기.
#
# 사용자 결정:
#   - 한 사이클당 활성 목표 1개 (집중·trade-off 회피)
#   - 한 목표 max 3 사이클 → 진척 없으면 자동 폐기
#   - 목표 형식: 무조건 "자기모델 지표 분포 t → t+1" 변환 (외부 효과 금지)
#   - trade-off 검사: 목표 지표 외 다른 지표 동시 측정·악화 발견 시 알림
#
# 디자인 원칙:
#   - LLM 호출 1회 (목표 자연어 설명). graceful — 실패 시 기본 설명.
#   - 목표 = 단순 dict, JSON 직렬화 가능
#   - 저장: eidos_files/agents/self_goals.json (active + 폐기 archive)

from __future__ import annotations

import datetime as _dt
import json
import os
import uuid
from dataclasses import dataclass, asdict, field
from typing import Optional


_BASE_DIR = os.path.join("eidos_files", "agents")
_PATH = os.path.join(_BASE_DIR, "self_goals.json")
_VERSION = 1

# 목표 생성 임계값
_TARGET_DELTA = 0.15        # 목표 = 현재 + 0.15 (한 사이클 안 도달 가능 수준)
_TARGET_CEIL = 0.95         # 목표 상한 (1.0 직전)
_MAX_CYCLES = 3             # 활성 목표 최대 유지 사이클
_TRADEOFF_THRESHOLD = 0.10  # trade-off 검사: 타 지표 -0.10 이상 악화 시 알림

# 목표 상태
GOAL_STATUSES = ("active", "achieved", "abandoned")


# ── SelfGoal ───────────────────────────────────────────────────────────
@dataclass
class SelfGoal:
    """EIDOS 의 자기개선 목표.

    형식: "지표 X 의 분포가 t (current) → t+1 (target) 로 변환".
    무조건 자기모델 지표만 — 외부 효과 (예: 매출 ↑) X.
    """

    id: str = ""
    target_indicator: str = ""       # INDICATOR_NAMES 중 하나
    current_value: float = 0.5       # t 시점 측정값
    target_value: float = 0.65       # t+1 목표값 (current + TARGET_DELTA)
    direction: str = "up"            # "up" 만 사용 (모든 지표가 높을수록 좋음 가정)

    # 자연어 설명 — LLM 생성
    description: str = ""

    # lifecycle
    status: str = "active"           # GOAL_STATUSES
    created_at: str = ""
    cycles_active: int = 0           # 0 부터·매 사이클 +1
    achieved_at: str = ""
    abandoned_at: str = ""
    abandoned_reason: str = ""       # "max_cycles" / "tradeoff" / "user_request"

    # trade-off — 이 목표 추구 중 악화된 다른 지표 기록
    tradeoffs_observed: list = field(default_factory=list)

    # 메타
    version: int = _VERSION

    def serialize(self) -> dict:
        return asdict(self)

    @classmethod
    def deserialize(cls, data: dict) -> "SelfGoal":
        if not isinstance(data, dict):
            data = {}
        out = cls()
        out.id = str(data.get("id") or _new_id())
        out.target_indicator = str(data.get("target_indicator") or "")
        try:
            out.current_value = float(data.get("current_value", 0.5))
            out.target_value = float(data.get("target_value", 0.65))
        except Exception:
            pass
        out.direction = str(data.get("direction") or "up")
        out.description = str(data.get("description") or "")
        st = str(data.get("status") or "active")
        out.status = st if st in GOAL_STATUSES else "active"
        out.created_at = str(data.get("created_at") or _now())
        try:
            out.cycles_active = int(data.get("cycles_active", 0))
        except Exception:
            out.cycles_active = 0
        out.achieved_at = str(data.get("achieved_at") or "")
        out.abandoned_at = str(data.get("abandoned_at") or "")
        out.abandoned_reason = str(data.get("abandoned_reason") or "")
        out.tradeoffs_observed = list(data.get("tradeoffs_observed") or [])
        try:
            out.version = int(data.get("version") or _VERSION)
        except Exception:
            out.version = _VERSION
        return out


# ── 헬퍼 ──────────────────────────────────────────────────────────────
def _now() -> str:
    return _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _new_id() -> str:
    return f"goal_{uuid.uuid4().hex[:12]}"


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _ensure_base() -> None:
    try:
        os.makedirs(_BASE_DIR, exist_ok=True)
    except Exception as e:
        print(f"[self_goals] _ensure_base 실패 (graceful): {e}")


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
        print(f"[self_goals] _atomic_write 실패 (graceful): {path} — {e}")
        return False


# ── CRUD ───────────────────────────────────────────────────────────────
def load_all() -> list[SelfGoal]:
    """전체 goals 디스크 로드 (active + 폐기 archive). 없으면 빈 list."""
    if not os.path.exists(_PATH):
        return []
    try:
        with open(_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = data.get("goals", []) if isinstance(data, dict) else []
        return [SelfGoal.deserialize(g) for g in data]
    except Exception as e:
        print(f"[self_goals] load_all 실패 (graceful, 빈 list 반환): {e}")
        return []


def save_all(goals: list[SelfGoal]) -> bool:
    """전체 goals 저장. atomic."""
    _ensure_base()
    try:
        payload = [g.serialize() for g in goals]
        return _atomic_write(_PATH, json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"[self_goals] save_all 실패 (graceful): {e}")
        return False


def get_active() -> list[SelfGoal]:
    """현재 활성 목표만 반환."""
    return [g for g in load_all() if g.status == "active"]


def delete_all_goals() -> bool:
    """테스트·reset."""
    try:
        if os.path.exists(_PATH):
            os.remove(_PATH)
        return True
    except Exception as e:
        print(f"[self_goals] delete_all_goals 실패 (graceful): {e}")
        return False


# ── 목표 생성 ──────────────────────────────────────────────────────────
def generate_goal(
    indicators,
    max_active: int = 1,
    skip_recent_indicators: Optional[list[str]] = None,
) -> Optional[SelfGoal]:
    """현재 지표에서 약점 → 새 목표 생성. 활성 목표 이미 있으면 None.

    Args:
        indicators: SelfIndicators 객체
        max_active: 활성 목표 최대 수 (default 1)
        skip_recent_indicators: 최근 폐기된 지표 list (중복 생성 회피)

    Returns:
        새 SelfGoal 또는 None (활성 max 또는 약점 못 찾음)
    """
    try:
        active = get_active()
        if len(active) >= max_active:
            print(f"[self_goals] 활성 목표 {len(active)}/{max_active} 도달 — skip")
            return None

        from eidos_self_indicators import weakest_dimensions
        weakest = weakest_dimensions(indicators, k=10, include_placeholder=False)
        if not weakest:
            print("[self_goals] 약점 발견 못함 (placeholder 만) — skip")
            return None

        # skip_recent 에 있는 지표 제외
        if skip_recent_indicators:
            weakest = [(n, v) for n, v in weakest if n not in skip_recent_indicators]
            if not weakest:
                print("[self_goals] 모든 약점이 최근 폐기 list — skip")
                return None

        # 활성 목표가 이미 있는 지표 제외
        active_indicators = {g.target_indicator for g in active}
        weakest = [(n, v) for n, v in weakest if n not in active_indicators]
        if not weakest:
            print("[self_goals] 약점 지표가 이미 활성 목표 — skip")
            return None

        target_name, current = weakest[0]
        target_value = _clamp(current + _TARGET_DELTA, 0.0, _TARGET_CEIL)

        goal = SelfGoal(
            id=_new_id(),
            target_indicator=target_name,
            current_value=current,
            target_value=target_value,
            direction="up",
            description=_default_description(target_name, current, target_value),
            status="active",
            created_at=_now(),
            cycles_active=0,
        )

        # LLM 으로 description 정교화 (graceful)
        try:
            goal.description = _llm_describe(goal) or goal.description
        except Exception as e:
            print(f"[self_goals] LLM description 실패 (graceful, 기본 사용): {e}")

        return goal
    except Exception as e:
        print(f"[self_goals] generate_goal 실패 (graceful): {e}")
        return None


def _default_description(indicator: str, current: float, target: float) -> str:
    """LLM 없을 때 기본 자연어 설명."""
    return (
        f"지표 '{indicator}' 를 현재 {current:.2f} 에서 {target:.2f} 로 끌어올린다. "
        f"5일 사이클 안 도달 목표 — 관련 외부 자료 (논문·기사) 능동 검색해 "
        f"패턴 추출 후 EIDOS 자기모델에 이식."
    )


def _llm_describe(goal: SelfGoal) -> Optional[str]:
    """LLM 1회 호출 — 목표를 친근체 자연어 설명으로 정교화.

    페르소나 (에스더 님·여동생 톤) 일관. 실패 시 None 반환 (호출자가 default 사용).
    """
    try:
        from llm_module import get_llm_response_async  # 가정 — 표준 호출
        # 비동기 호출이라 동기 wrapper 필요. 일단 placeholder — 기본 설명 사용.
        # Wave 2 에서 동기 LLM 호출 매크로 추가 시 활성화.
        return None
    except Exception:
        return None


# ── 진척 평가 ──────────────────────────────────────────────────────────
def evaluate_progress(
    goal: SelfGoal,
    current_indicators,
) -> dict:
    """목표 진척 평가. 반환: {"achieved": bool, "progress": float, "current": float}.

    progress = (current - initial) / (target - initial), 0~1+ 범위.
    target 도달 시 achieved=True. graceful.
    """
    out = {"achieved": False, "progress": 0.0, "current": goal.current_value}
    try:
        vec = current_indicators.to_vector(include_ext=True)
        if goal.target_indicator not in vec:
            return out
        curr = float(vec[goal.target_indicator])
        out["current"] = curr
        # progress 계산
        span = goal.target_value - goal.current_value
        if abs(span) < 1e-6:
            out["progress"] = 1.0 if curr >= goal.target_value else 0.0
        else:
            out["progress"] = (curr - goal.current_value) / span
        out["achieved"] = curr >= goal.target_value
    except Exception as e:
        print(f"[self_goals] evaluate_progress 실패 (graceful): {e}")
    return out


def detect_tradeoffs(
    goal: SelfGoal,
    prev_indicators,
    curr_indicators,
    threshold: float = _TRADEOFF_THRESHOLD,
) -> list[tuple[str, float]]:
    """이 목표 추구 중 악화된 다른 지표 list. (지표명, delta) 튜플.

    delta < -threshold 인 지표만. 목표 지표는 제외.
    """
    out: list[tuple[str, float]] = []
    try:
        from eidos_self_indicators import compute_diff
        diff = compute_diff(prev_indicators, curr_indicators, include_ext=True)
        for name, delta in diff.items():
            if name == goal.target_indicator:
                continue
            if delta <= -threshold:
                out.append((name, delta))
    except Exception as e:
        print(f"[self_goals] detect_tradeoffs 실패 (graceful): {e}")
    return out


# ── 사이클 진행 — 활성 목표 갱신·폐기 ──────────────────────────────────
def cycle_step(
    current_indicators,
    prev_indicators=None,
) -> dict:
    """한 사이클 진행:
      1. 활성 목표마다 cycles_active += 1
      2. progress 평가 — 달성이면 achieved 처리
      3. cycles_active >= MAX_CYCLES 인데 미달성이면 abandoned (reason=max_cycles)
      4. trade-off 검사 (prev 있을 때만) — tradeoffs_observed 기록
      5. 활성 목표 없으면 새 목표 생성 시도

    반환: {
        "cycle_summary": {progress, achieved, abandoned, tradeoffs},
        "new_goal": Optional[SelfGoal],
        "active_goals": list[SelfGoal],
    }
    """
    summary = {
        "progressed": [],
        "achieved": [],
        "abandoned": [],
        "tradeoffs": [],
    }
    try:
        all_goals = load_all()
        active = [g for g in all_goals if g.status == "active"]

        # 1·2·3·4: 각 활성 목표 처리
        for g in active:
            g.cycles_active += 1
            ev = evaluate_progress(g, current_indicators)
            summary["progressed"].append({
                "goal_id": g.id, "indicator": g.target_indicator,
                "progress": round(ev["progress"], 3),
                "current": round(ev["current"], 3),
                "target": round(g.target_value, 3),
                "cycles": g.cycles_active,
            })

            if ev["achieved"]:
                g.status = "achieved"
                g.achieved_at = _now()
                summary["achieved"].append(g.id)
            elif g.cycles_active >= _MAX_CYCLES:
                g.status = "abandoned"
                g.abandoned_at = _now()
                g.abandoned_reason = "max_cycles"
                summary["abandoned"].append({
                    "goal_id": g.id, "indicator": g.target_indicator,
                    "reason": "max_cycles",
                })

            # 4. trade-off 검사 (prev 있을 때)
            if prev_indicators is not None and g.status == "active":
                tos = detect_tradeoffs(g, prev_indicators, current_indicators)
                if tos:
                    g.tradeoffs_observed.extend([
                        {"at": _now(), "indicator": n, "delta": round(d, 4)}
                        for n, d in tos
                    ])
                    summary["tradeoffs"].extend([
                        {"goal_id": g.id, "indicator": n, "delta": round(d, 4)}
                        for n, d in tos
                    ])

        # 5. 활성 목표 없으면 새 목표 시도
        still_active = [g for g in all_goals if g.status == "active"]
        new_goal: Optional[SelfGoal] = None
        if not still_active:
            # 최근 폐기 지표 list — 같은 거 또 잡지 않게
            recent_abandoned = [
                g.target_indicator for g in all_goals
                if g.status == "abandoned"
                and len(g.abandoned_at) > 0  # 폐기 시각 있는 것만
            ]
            # 최근 3개만 skip (오래된 건 다시 시도 OK)
            skip = recent_abandoned[-3:] if len(recent_abandoned) > 3 else recent_abandoned
            new_goal = generate_goal(
                current_indicators,
                max_active=1,
                skip_recent_indicators=skip,
            )
            if new_goal:
                all_goals.append(new_goal)

        save_all(all_goals)
        active_now = [g for g in all_goals if g.status == "active"]
        return {
            "cycle_summary": summary,
            "new_goal": new_goal,
            "active_goals": active_now,
        }
    except Exception as e:
        print(f"[self_goals] cycle_step 실패 (graceful): {e}")
        return {
            "cycle_summary": summary,
            "new_goal": None,
            "active_goals": [],
        }


# ── 텔레그램 요약 ──────────────────────────────────────────────────────
def summarize_cycle_for_telegram(cycle_result: dict) -> str:
    """cycle_step 결과 → 텔레그램 markdown."""
    lines = ["🎯 *EIDOS 자기개선 사이클*"]
    s = cycle_result.get("cycle_summary") or {}
    new_g = cycle_result.get("new_goal")
    active = cycle_result.get("active_goals") or []

    if s.get("achieved"):
        lines.append(f"\n✅ *달성*: {len(s['achieved'])}개")
        for g_id in s["achieved"]:
            lines.append(f"  • {g_id}")

    if s.get("abandoned"):
        lines.append(f"\n❌ *폐기*: {len(s['abandoned'])}개")
        for ab in s["abandoned"]:
            lines.append(f"  • {ab['indicator']} ({ab['reason']})")

    if s.get("progressed"):
        lines.append("\n📊 *진척 중*")
        for p in s["progressed"]:
            lines.append(
                f"  • {p['indicator']}: {p['current']:.2f}/{p['target']:.2f} "
                f"({p['progress']*100:+.0f}% · 사이클 {p['cycles']})"
            )

    if s.get("tradeoffs"):
        lines.append(f"\n⚠️ *Trade-off 감지*: {len(s['tradeoffs'])}건")
        for t in s["tradeoffs"][:5]:
            lines.append(f"  • {t['indicator']}: {t['delta']:+.3f}")

    if new_g:
        lines.append(f"\n🆕 *신규 목표*")
        lines.append(f"  • {new_g.target_indicator}: "
                     f"{new_g.current_value:.2f} → {new_g.target_value:.2f}")
        lines.append(f"  • {new_g.description[:120]}")

    if not active and not new_g:
        lines.append("\n💤 활성 목표 없음 — 약점 발견 못 함 (placeholder 만)")

    return "\n".join(lines)
