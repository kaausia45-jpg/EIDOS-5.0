# eidos_pattern_catalog.py
# [Wave2-D 2026-05-28] 미정계수 삽입 + 이식된 패턴 catalog.
#
# 사용자 결정:
#   - 매 transition.change.delta 의 수치를 {alpha_N} 매개변수로 치환
#   - 초기값 = 자료 수치 × source_weight (출처 신뢰도)
#   - Layer 2 (확률분포·기존 PatternModel auto_threshold 패턴 재사용) 가 학습
#   - 별도 catalog (PatternModel 통합 X — debug·rollback 쉽게)
#
# 디자인 원칙:
#   - LLM 호출 0 (이 단계는 매개변수화·저장만)
#   - 매 패턴 unique id (uuid)
#   - apply_count·success_count 누적 → Wave2-E 가 효과 측정·폐기
#   - graceful — 손상 데이터 skip

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import uuid
from dataclasses import dataclass, asdict, field
from typing import Optional


_BASE_DIR = os.path.join("eidos_files", "imported_patterns")
_VERSION = 1

# 폐기 임계
_MIN_APPLY_FOR_REVIEW = 3       # 최소 3회 적용 후 효과 측정
_MIN_SUCCESS_RATIO = 0.3        # 30% 미만 성공이면 폐기 후보
_MAX_APPLY_BEFORE_DECISION = 10 # 10회 적용 후엔 명확히 결정 (폐기 or maintain)


# ── ImportedPattern ────────────────────────────────────────────────────
@dataclass
class ImportedPattern:
    """EIDOS 에 이식된 외부 패턴.

    TranslatedPattern 의 정량값이 미정계수로 매개변수화된 형태 + lifecycle 메타.
    """

    id: str = ""
    # 출처 (TranslatedPattern 에서 복사)
    source_url: str = ""
    source_type: str = ""
    source_title: str = ""
    source_weight: float = 0.5
    target_goal_indicator: str = ""    # 이 패턴이 어떤 목표 추구 중 이식됐나

    # 매핑 (TranslatedPattern 의 그대로)
    actor_mapping: dict = field(default_factory=dict)
    indicator_mapping: dict = field(default_factory=dict)
    mapping_confidence: float = 0.5
    rationale: str = ""

    # 매개변수화된 transitions·patterns
    transitions: list = field(default_factory=list)   # changes 의 delta 가 {alpha_N}
    patterns: list = field(default_factory=list)      # 자연어 표현

    # 미정계수 — Layer 2 가 학습
    parameters: dict = field(default_factory=dict)        # {alpha_1: current_value, ...}
    parameter_priors: dict = field(default_factory=dict)  # 초기값 (변하지 않음)
    parameter_weights: dict = field(default_factory=dict) # 신뢰도 (0~1, 학습)

    # lifecycle
    created_at: str = ""
    status: str = "active"    # active / abandoned / maintained
    apply_count: int = 0
    success_count: int = 0    # 적용 후 target_goal_indicator 가 실제 ↑
    neutral_count: int = 0    # 변화 없음
    fail_count: int = 0       # 적용 후 오히려 ↓
    last_applied_at: str = ""
    last_outcome: str = ""    # "positive" / "neutral" / "negative" / ""
    abandon_reason: str = ""

    # 텔레그램 알림 상태
    notified_at: str = ""

    version: int = _VERSION

    def serialize(self) -> dict:
        return asdict(self)

    @classmethod
    def deserialize(cls, data: dict) -> "ImportedPattern":
        if not isinstance(data, dict):
            data = {}
        out = cls()
        out.id = str(data.get("id") or _new_id())
        for k in ("source_url", "source_type", "source_title",
                  "target_goal_indicator", "rationale",
                  "created_at", "status", "last_applied_at", "last_outcome",
                  "abandon_reason", "notified_at"):
            setattr(out, k, str(data.get(k) or ""))
        try:
            out.source_weight = float(data.get("source_weight", 0.5))
            out.mapping_confidence = float(data.get("mapping_confidence", 0.5))
        except Exception:
            pass
        out.actor_mapping = dict(data.get("actor_mapping") or {})
        out.indicator_mapping = dict(data.get("indicator_mapping") or {})
        out.transitions = list(data.get("transitions") or [])
        out.patterns = list(data.get("patterns") or [])
        out.parameters = dict(data.get("parameters") or {})
        out.parameter_priors = dict(data.get("parameter_priors") or {})
        out.parameter_weights = dict(data.get("parameter_weights") or {})
        for k in ("apply_count", "success_count", "neutral_count", "fail_count", "version"):
            try:
                setattr(out, k, int(data.get(k, 0)))
            except Exception:
                setattr(out, k, 0)
        if out.status not in ("active", "abandoned", "maintained"):
            out.status = "active"
        return out


# ── 헬퍼 ──────────────────────────────────────────────────────────────
def _now() -> str:
    return _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _new_id() -> str:
    return f"imp_{uuid.uuid4().hex[:12]}"


def _ensure_base() -> None:
    try:
        os.makedirs(_BASE_DIR, exist_ok=True)
    except Exception as e:
        print(f"[catalog] _ensure_base 실패 (graceful): {e}")


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
        print(f"[catalog] _atomic_write 실패 (graceful): {e}")
        return False


# ── 미정계수 삽입 ─────────────────────────────────────────────────────
def _parse_delta(delta_str: str) -> Optional[float]:
    """'+0.10' / '-0.05' / '+10%' 같은 문자열 → float. 실패 시 None."""
    if delta_str is None:
        return None
    s = str(delta_str).strip()
    if not s:
        return None
    # "+10%" → 0.10
    pct = re.match(r"^([+-]?\d+(?:\.\d+)?)\s*%$", s)
    if pct:
        try:
            return float(pct.group(1)) / 100.0
        except Exception:
            return None
    # "+0.10" / "-0.05" / "0.5"
    m = re.match(r"^([+-]?\d+(?:\.\d+)?)$", s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def parameterize(
    translated_pattern,
    target_goal_indicator: str = "",
) -> Optional[ImportedPattern]:
    """TranslatedPattern → ImportedPattern (미정계수 삽입 + lifecycle 메타).

    각 transition.change.delta 를 {alpha_N} 으로 치환. prior = 원본값 ×
    source_weight (출처 신뢰도). Layer 2 가 적용 결과 보고 학습.

    Returns: ImportedPattern 또는 None (transitions·patterns 모두 빈 경우).
    """
    if not translated_pattern:
        return None
    try:
        # transitions 가 1개 이상 있거나 patterns 가 있어야 의미 있음
        n_transitions = len(translated_pattern.translated_transitions or [])
        n_patterns = len(translated_pattern.translated_patterns or [])
        if n_transitions == 0 and n_patterns == 0:
            print("[catalog] 매개변수화할 transitions·patterns 모두 빈 — skip")
            return None

        out = ImportedPattern(
            id=_new_id(),
            source_url=translated_pattern.source_url,
            source_type=translated_pattern.source_type,
            source_title=translated_pattern.source_title,
            source_weight=translated_pattern.source_weight,
            target_goal_indicator=target_goal_indicator,
            actor_mapping=dict(translated_pattern.actor_mapping),
            indicator_mapping=dict(translated_pattern.indicator_mapping),
            mapping_confidence=translated_pattern.mapping_confidence,
            rationale=translated_pattern.rationale,
            transitions=[],
            patterns=list(translated_pattern.translated_patterns or []),
            parameters={},
            parameter_priors={},
            parameter_weights={},
            created_at=_now(),
            status="active",
            apply_count=0,
            success_count=0,
            neutral_count=0,
            fail_count=0,
        )

        # transitions 안 delta 를 {alpha_N} 으로 치환
        counter = 0
        for t in (translated_pattern.translated_transitions or []):
            if not isinstance(t, dict):
                continue
            new_t = {
                "trigger_actor": t.get("trigger_actor", ""),
                "trigger_action": t.get("trigger_action", ""),
                "conditions": list(t.get("conditions") or []),
                "source_quote": t.get("source_quote", ""),
                "changes": [],
            }
            for c in (t.get("changes") or []):
                if not isinstance(c, dict):
                    continue
                target = c.get("target", "")
                delta_str = c.get("delta", "")
                num = _parse_delta(delta_str)
                if num is None or not target:
                    # 매개변수화 불가 — 변화 자체 drop
                    continue
                counter += 1
                param_name = f"alpha_{counter}"
                # prior = 자료 수치 × source_weight × mapping_confidence
                prior = num * out.source_weight * max(0.3, out.mapping_confidence)
                # clamp 안 함 — 자유 범위 (Layer 2 가 학습으로 조정)
                out.parameters[param_name] = round(prior, 4)
                out.parameter_priors[param_name] = round(prior, 4)
                out.parameter_weights[param_name] = 0.5  # neutral 시작
                new_t["changes"].append({
                    "target": target,
                    "delta": "{" + param_name + "}",
                    "original_delta": delta_str,
                    "param_name": param_name,
                })
            if new_t["changes"]:
                out.transitions.append(new_t)

        # 매개변수 1개도 못 만들었으면 fail
        if not out.parameters:
            print("[catalog] 매개변수화 결과 0건 — skip")
            return None

        return out
    except Exception as e:
        print(f"[catalog] parameterize 실패 (graceful): {e}")
        return None


# ── catalog CRUD ───────────────────────────────────────────────────────
def save_pattern(pattern: ImportedPattern) -> Optional[str]:
    """단일 패턴 저장. 반환: 파일 경로."""
    if not pattern:
        return None
    _ensure_base()
    if not pattern.id:
        pattern.id = _new_id()
    path = os.path.join(_BASE_DIR, f"{pattern.id}.json")
    try:
        ok = _atomic_write(path, json.dumps(pattern.serialize(),
                                             ensure_ascii=False, indent=2))
        return path if ok else None
    except Exception as e:
        print(f"[catalog] save_pattern 실패 (graceful): {e}")
        return None


def load_pattern(pattern_id: str) -> Optional[ImportedPattern]:
    """id 로 단일 패턴 로드."""
    if not pattern_id:
        return None
    path = os.path.join(_BASE_DIR, f"{pattern_id}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return ImportedPattern.deserialize(json.load(f))
    except Exception as e:
        print(f"[catalog] load_pattern 실패 (graceful, {pattern_id}): {e}")
        return None


def list_all_patterns() -> list[ImportedPattern]:
    """catalog 전체 list (status 무관·생성 시각 오름차순)."""
    if not os.path.isdir(_BASE_DIR):
        return []
    out: list[ImportedPattern] = []
    try:
        files = [f for f in os.listdir(_BASE_DIR) if f.endswith(".json")]
        for fname in sorted(files):
            try:
                with open(os.path.join(_BASE_DIR, fname), "r", encoding="utf-8") as f:
                    out.append(ImportedPattern.deserialize(json.load(f)))
            except Exception:
                continue
    except Exception as e:
        print(f"[catalog] list_all_patterns 실패 (graceful): {e}")
    return sorted(out, key=lambda p: p.created_at)


def list_active_patterns() -> list[ImportedPattern]:
    """status=active 만."""
    return [p for p in list_all_patterns() if p.status == "active"]


def list_patterns_for_indicator(indicator: str) -> list[ImportedPattern]:
    """target_goal_indicator 매칭 active 패턴만."""
    if not indicator:
        return []
    return [p for p in list_active_patterns()
            if p.target_goal_indicator == indicator]


# ── [Phase 0 2026-05-30 self-apply] EIDOS-self 자기수정용 패턴 판별/조회 ──────
# 유추전이 비전의 정점: 외부 도메인 f() 중 *출력이 EIDOS-self 자신*인 패턴은
# 단순 예측이 아니라 EIDOS 실행 로직을 강제로 바꾸는 컨트롤러로 쓴다.
# 그 1차 관문 — 어떤 패턴이 self 적용 대상인지 가린다.
#
# translate 단계(_translate_transitions)가 change.target 을 indicator_map 으로
# "EIDOS-self.<지표>" / "user.<지표>" / "external_actor.<지표>" 형식으로 재작성하므로,
# transitions 의 change.target 이 "EIDOS-self." 로 시작하면 그 패턴은 EIDOS 자신의
# 상태를 변화시키는 것 = self-applicable. (고객예측처럼 출력이 user/external_actor
# 인 패턴은 강제 대상 아님 → False.)
_EIDOS_SELF_OBJ = "EIDOS-self"


def _target_is_eidos_self(target: str) -> bool:
    """transition change target('EIDOS-self.reasoning' 형식)이 self 지표를 가리키는지."""
    if not target or not isinstance(target, str):
        return False
    return target.strip().startswith(_EIDOS_SELF_OBJ + ".")


def is_self_applicable(pattern: ImportedPattern) -> bool:
    """패턴이 EIDOS-self 자기 상태를 *변화시키는*(= 강제 자기수정 대상) 패턴인지.

    판정: transitions 의 어떤 change.target 이 'EIDOS-self.<지표>' 면 True.
    출력이 user/external_actor 인 예측용 패턴은 False.
    transitions 없이 자연어 patterns 만 있는 패턴도 False (강제할 구조적 트리거 없음).
    """
    if pattern is None:
        return False
    try:
        for t in (pattern.transitions or []):
            if not isinstance(t, dict):
                continue
            for c in (t.get("changes") or []):
                if isinstance(c, dict) and _target_is_eidos_self(c.get("target", "")):
                    return True
    except Exception:
        return False
    return False


def list_self_applicable_patterns(
    indicator: str = "",
    active_only: bool = True,
) -> list[ImportedPattern]:
    """self-applicable 패턴 list (생성 시각 오름차순).

    Args:
        indicator: 주면 target_goal_indicator 가 일치하는 것만 (현재 목표 매칭용).
        active_only: True 면 status=active 만 (abandoned 제외).
    """
    base = list_active_patterns() if active_only else list_all_patterns()
    out = [p for p in base if is_self_applicable(p)]
    if indicator:
        out = [p for p in out if p.target_goal_indicator == indicator]
    return out


def delete_pattern(pattern_id: str) -> bool:
    """파일 삭제."""
    path = os.path.join(_BASE_DIR, f"{pattern_id}.json")
    try:
        if os.path.exists(path):
            os.remove(path)
        return True
    except Exception as e:
        print(f"[catalog] delete_pattern 실패 (graceful): {e}")
        return False


def delete_all_patterns() -> bool:
    """테스트·reset 용."""
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
        print(f"[catalog] delete_all_patterns 실패 (graceful): {e}")
        return False


# ── 효과 측정 — Wave2-E 가 호출 ───────────────────────────────────────
def record_outcome(
    pattern_id: str,
    outcome: str,
    indicator_delta: float = 0.0,
) -> bool:
    """패턴 적용 후 결과 기록. outcome ∈ {"positive", "neutral", "negative"}.

    indicator_delta: 사이클에서 target 지표가 실제 어떻게 변했나 (참고용).
    """
    pattern = load_pattern(pattern_id)
    if not pattern:
        return False
    try:
        pattern.apply_count += 1
        pattern.last_applied_at = _now()
        if outcome == "positive":
            pattern.success_count += 1
        elif outcome == "neutral":
            pattern.neutral_count += 1
        elif outcome == "negative":
            pattern.fail_count += 1
        pattern.last_outcome = outcome

        # Layer 2 학습: weight 갱신 (간단한 EMA)
        # positive 시 weight ↑·negative 시 ↓·neutral 시 약간 ↓ (decay)
        delta_w = 0.0
        if outcome == "positive":
            delta_w = 0.1
        elif outcome == "negative":
            delta_w = -0.15
        else:
            delta_w = -0.02   # neutral 은 약간 감쇠

        for pname in pattern.parameter_weights:
            cur = pattern.parameter_weights[pname]
            new = max(0.0, min(1.0, cur + delta_w))
            pattern.parameter_weights[pname] = round(new, 4)

        # 폐기 판정
        if pattern.apply_count >= _MIN_APPLY_FOR_REVIEW:
            total = pattern.apply_count
            success_ratio = pattern.success_count / total if total else 0.0
            if success_ratio < _MIN_SUCCESS_RATIO:
                pattern.status = "abandoned"
                pattern.abandon_reason = (
                    f"low_success_ratio ({pattern.success_count}/{total} = "
                    f"{success_ratio:.2f})"
                )
            elif total >= _MAX_APPLY_BEFORE_DECISION and success_ratio >= 0.6:
                pattern.status = "maintained"
                pattern.abandon_reason = ""

        save_pattern(pattern)
        return True
    except Exception as e:
        print(f"[catalog] record_outcome 실패 (graceful): {e}")
        return False


# ── 텔레그램 요약 ──────────────────────────────────────────────────────
def summarize_catalog_for_telegram(top_n: int = 5) -> str:
    """catalog 상태 markdown 요약 (5일 사이클 보고용)."""
    all_patterns = list_all_patterns()
    if not all_patterns:
        return "📚 *이식 패턴 catalog* — 비어있음"

    active = [p for p in all_patterns if p.status == "active"]
    maintained = [p for p in all_patterns if p.status == "maintained"]
    abandoned = [p for p in all_patterns if p.status == "abandoned"]

    lines = [f"📚 *이식 패턴 catalog* (총 {len(all_patterns)}개)",
             f"  • active: {len(active)} · maintained: {len(maintained)} "
             f"· abandoned: {len(abandoned)}"]

    if active:
        lines.append("\n*최근 active*:")
        for p in sorted(active, key=lambda x: x.created_at, reverse=True)[:top_n]:
            applied = p.apply_count
            successes = p.success_count
            lines.append(
                f"  • `{p.id[:12]}` ({p.source_type}, w={p.source_weight:.1f}) "
                f"→ {p.target_goal_indicator} | 적용 {applied}회·성공 {successes}"
            )

    if abandoned and len(abandoned) <= 5:
        lines.append("\n*폐기 (최근)*:")
        for p in sorted(abandoned, key=lambda x: x.last_applied_at, reverse=True)[:3]:
            lines.append(f"  • `{p.id[:12]}` — {p.abandon_reason[:80]}")

    return "\n".join(lines)


def summarize_imported_for_telegram(pattern: ImportedPattern) -> str:
    """단일 패턴 신규 이식 알림용 markdown."""
    if not pattern:
        return ""
    confidence_emoji = "✅" if pattern.mapping_confidence >= 0.5 else "⚠️"
    lines = [
        f"🌱 *신규 패턴 이식* {confidence_emoji}",
        f"  ID: `{pattern.id[:12]}`",
        f"  출처: {pattern.source_type} (w={pattern.source_weight:.1f}) — "
        f"{pattern.source_title[:60]}",
        f"  목표 지표: {pattern.target_goal_indicator}",
        f"  매핑 confidence: {pattern.mapping_confidence:.2f}",
        f"  미정계수: {len(pattern.parameters)}개 — "
        f"{list(pattern.parameters.keys())[:5]}",
    ]
    if pattern.patterns:
        lines.append("  *규칙*:")
        for p in pattern.patterns[:3]:
            lines.append(f"    • {p[:120]}")
    return "\n".join(lines)
