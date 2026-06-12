# eidos_tool_catalog.py
# [Wave3-A 2026-05-28] EIDOS 도구 catalog — 디지털 환경 모든 것이 도구.
#
# 사용자 결정:
#   - 도구 = EIDOS 빼고 디지털 환경 모든 것 (코드·API·DAG·prompt·data·외부 AI)
#   - 자동 제작 + 텔레그램 알림
#   - 실행 코드는 텔레그램 승인 게이트 / 비실행 자산 (DAG·prompt·데이터) 은 자동
#
# 디자인 원칙:
#   - LLM 호출 0 (이 단계는 CRUD·status 관리만)
#   - 도구 타입별 다른 활용 (DAG → canvas runner, prompt → LLM call, code → sandbox)
#   - 활성화 (status) lifecycle 명확 — pending_approval / active / rejected / abandoned
#   - 자동 활성화 가능: DAG·prompt·data — 위험 없음
#   - 사용자 승인 필요: code·api_config — 위험 (보안·외부 호출)

from __future__ import annotations

import datetime as _dt
import json
import os
import uuid
from dataclasses import dataclass, asdict, field
from typing import Optional


_BASE_DIR = os.path.join("eidos_files", "tool_catalog")
_VERSION = 1


# ── 도구 타입 정의 ────────────────────────────────────────────────────
TOOL_TYPES = (
    "dag",              # PCA DAG (캔버스 워크플로우)
    "prompt_template",  # LLM prompt 템플릿
    "data",             # JSON config·sample·keyword list 등
    "code",             # Python 함수·스크립트 (위험·승인 게이트)
    "api_config",       # 외부 API 설정 (URL·키 placeholder·승인 필요)
)

# 자동 활성화 허용 타입 (사용자 결정·비실행 자산)
AUTO_ACTIVATE_TYPES = ("dag", "prompt_template", "data")

# 승인 게이트 필요 타입 (실행·외부 영향)
APPROVAL_REQUIRED_TYPES = ("code", "api_config")


# ── 도구 상태 ──────────────────────────────────────────────────────────
TOOL_STATUSES = (
    "active",            # 활성·사용 가능
    "pending_approval",  # 사용자 승인 대기
    "rejected",          # 사용자 거부 → 폐기 archive
    "abandoned",         # 사용 효과 없어 자동 폐기
    "draft",             # 제작 중간 (LLM 결과 미완)
)


# ── Tool dataclass ─────────────────────────────────────────────────────
@dataclass
class Tool:
    """EIDOS 가 갖는 1 도구. catalog 의 단위."""

    id: str = ""
    name: str = ""
    type: str = ""               # TOOL_TYPES 중 하나
    description: str = ""        # 인간 친화 설명 (LLM 생성 가능)

    # 어떤 목표·패턴에서 파생된 거 (제작 컨텍스트)
    target_indicator: str = ""   # ex: "comp_marketing"
    source: str = "auto_generated"   # "auto_generated" / "imported_pattern" / "user_manual"
    related_pattern_ids: list = field(default_factory=list)

    # 도구 내용 — type 에 따라 다른 형식
    content: str = ""            # DAG JSON 문자열·prompt 본문·코드 텍스트
    metadata: dict = field(default_factory=dict)  # type-specific 메타

    # lifecycle
    status: str = "draft"
    created_at: str = ""
    approved_at: str = ""        # 승인된 시각 (code/api_config)
    activated_at: str = ""       # active 진입 시각
    rejected_at: str = ""
    abandoned_at: str = ""
    abandoned_reason: str = ""

    # 사용 통계 (Wave3-D 의 효과 측정 누적)
    use_count: int = 0
    success_count: int = 0
    fail_count: int = 0
    last_used_at: str = ""
    last_outcome: str = ""

    # 텔레그램 알림 상태
    approval_request_sent_at: str = ""

    version: int = _VERSION

    def serialize(self) -> dict:
        return asdict(self)

    @classmethod
    def deserialize(cls, data: dict) -> "Tool":
        if not isinstance(data, dict):
            data = {}
        out = cls()
        out.id = str(data.get("id") or _new_id())
        out.name = str(data.get("name") or "")
        out.type = str(data.get("type") or "")
        out.description = str(data.get("description") or "")
        out.target_indicator = str(data.get("target_indicator") or "")
        out.source = str(data.get("source") or "auto_generated")
        out.related_pattern_ids = list(data.get("related_pattern_ids") or [])
        out.content = str(data.get("content") or "")
        out.metadata = dict(data.get("metadata") or {})
        out.status = str(data.get("status") or "draft")
        if out.status not in TOOL_STATUSES:
            out.status = "draft"
        if out.type not in TOOL_TYPES:
            out.type = "data"   # safest fallback
        for ts_field in ("created_at", "approved_at", "activated_at",
                          "rejected_at", "abandoned_at", "last_used_at",
                          "approval_request_sent_at"):
            setattr(out, ts_field, str(data.get(ts_field) or ""))
        out.abandoned_reason = str(data.get("abandoned_reason") or "")
        out.last_outcome = str(data.get("last_outcome") or "")
        for int_field in ("use_count", "success_count", "fail_count", "version"):
            try:
                setattr(out, int_field, int(data.get(int_field, 0)))
            except Exception:
                setattr(out, int_field, 0)
        return out

    @property
    def is_executable(self) -> bool:
        return self.type in APPROVAL_REQUIRED_TYPES

    @property
    def requires_approval(self) -> bool:
        return self.type in APPROVAL_REQUIRED_TYPES


# ── 헬퍼 ──────────────────────────────────────────────────────────────
def _now() -> str:
    return _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _new_id() -> str:
    return f"tool_{uuid.uuid4().hex[:12]}"


def _ensure_base() -> None:
    try:
        os.makedirs(_BASE_DIR, exist_ok=True)
    except Exception as e:
        print(f"[tool_catalog] _ensure_base 실패 (graceful): {e}")


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
        print(f"[tool_catalog] _atomic_write 실패 (graceful): {e}")
        return False


# ── CRUD ───────────────────────────────────────────────────────────────
def save_tool(tool: Tool) -> Optional[str]:
    if not tool:
        return None
    _ensure_base()
    if not tool.id:
        tool.id = _new_id()
    if not tool.created_at:
        tool.created_at = _now()
    path = os.path.join(_BASE_DIR, f"{tool.id}.json")
    try:
        ok = _atomic_write(path, json.dumps(tool.serialize(),
                                             ensure_ascii=False, indent=2))
        return path if ok else None
    except Exception as e:
        print(f"[tool_catalog] save_tool 실패 (graceful): {e}")
        return None


def load_tool(tool_id: str) -> Optional[Tool]:
    if not tool_id:
        return None
    path = os.path.join(_BASE_DIR, f"{tool_id}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return Tool.deserialize(json.load(f))
    except Exception as e:
        print(f"[tool_catalog] load_tool 실패 ({tool_id}): {e}")
        return None


def list_all_tools() -> list[Tool]:
    if not os.path.isdir(_BASE_DIR):
        return []
    out: list[Tool] = []
    try:
        files = [f for f in os.listdir(_BASE_DIR) if f.endswith(".json")]
        for fname in sorted(files):
            try:
                with open(os.path.join(_BASE_DIR, fname), "r", encoding="utf-8") as f:
                    out.append(Tool.deserialize(json.load(f)))
            except Exception:
                continue
    except Exception as e:
        print(f"[tool_catalog] list_all_tools 실패: {e}")
    return sorted(out, key=lambda t: t.created_at)


def list_tools_by_status(status: str) -> list[Tool]:
    return [t for t in list_all_tools() if t.status == status]


def list_tools_by_type(tool_type: str) -> list[Tool]:
    return [t for t in list_all_tools() if t.type == tool_type]


def list_tools_by_indicator(indicator: str) -> list[Tool]:
    """타겟 지표 매칭 active 도구만."""
    if not indicator:
        return []
    return [t for t in list_all_tools()
            if t.target_indicator == indicator and t.status == "active"]


def list_active_tools() -> list[Tool]:
    return list_tools_by_status("active")


def list_pending_approval() -> list[Tool]:
    return list_tools_by_status("pending_approval")


def delete_tool(tool_id: str) -> bool:
    path = os.path.join(_BASE_DIR, f"{tool_id}.json")
    try:
        if os.path.exists(path):
            os.remove(path)
        return True
    except Exception as e:
        print(f"[tool_catalog] delete_tool 실패: {e}")
        return False


def delete_all_tools() -> bool:
    """테스트·reset."""
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
        print(f"[tool_catalog] delete_all_tools 실패: {e}")
        return False


# ── 상태 전이 ──────────────────────────────────────────────────────────
def activate_tool(tool_id: str) -> bool:
    """tool 을 active 상태로. 자동 활성화는 비실행 자산만 권장."""
    tool = load_tool(tool_id)
    if not tool:
        return False
    if tool.status in ("rejected", "abandoned"):
        print(f"[tool_catalog] {tool_id} 이미 폐기 상태 — 활성화 불가")
        return False
    tool.status = "active"
    tool.activated_at = _now()
    return save_tool(tool) is not None


def approve_tool(tool_id: str) -> bool:
    """사용자 승인 → active 전환."""
    tool = load_tool(tool_id)
    if not tool:
        return False
    if tool.status != "pending_approval":
        print(f"[tool_catalog] {tool_id} pending_approval 아님 (현재 {tool.status})")
        return False
    tool.status = "active"
    tool.approved_at = _now()
    tool.activated_at = _now()
    return save_tool(tool) is not None


def reject_tool(tool_id: str, reason: str = "") -> bool:
    """사용자 거부 → rejected."""
    tool = load_tool(tool_id)
    if not tool:
        return False
    tool.status = "rejected"
    tool.rejected_at = _now()
    tool.abandoned_reason = (reason or "user_rejected")[:200]
    return save_tool(tool) is not None


def abandon_tool(tool_id: str, reason: str = "") -> bool:
    """자동 폐기 — 사용 효과 없어서."""
    tool = load_tool(tool_id)
    if not tool:
        return False
    tool.status = "abandoned"
    tool.abandoned_at = _now()
    tool.abandoned_reason = (reason or "auto_abandoned")[:200]
    return save_tool(tool) is not None


# ── 자동 활성화 helper — 비실행 자산만 즉시 active ────────────────────
def auto_activate_non_executable(tool: Tool) -> bool:
    """tool 의 type 이 비실행 자산이면 즉시 active. code/api 는 pending_approval.

    Wave3-B 가 호출 (비실행 자산 자동 제작 후)·Wave3-C 가 호출 (코드는 pending).
    """
    if not tool:
        return False
    if tool.type in AUTO_ACTIVATE_TYPES:
        tool.status = "active"
        tool.activated_at = _now()
    elif tool.type in APPROVAL_REQUIRED_TYPES:
        tool.status = "pending_approval"
    else:
        tool.status = "draft"
    return save_tool(tool) is not None


# ── 사용 효과 측정 (Wave3-D 가 호출) ──────────────────────────────────
def record_tool_use(tool_id: str, success: bool) -> bool:
    """도구 사용 1회 기록. 폐기 정책: 5회+ 적용 후 success_ratio<0.2 → abandoned."""
    tool = load_tool(tool_id)
    if not tool:
        return False
    try:
        tool.use_count += 1
        tool.last_used_at = _now()
        if success:
            tool.success_count += 1
            tool.last_outcome = "positive"
        else:
            tool.fail_count += 1
            tool.last_outcome = "negative"

        # 폐기 판정
        if tool.use_count >= 5:
            ratio = tool.success_count / tool.use_count if tool.use_count else 0.0
            if ratio < 0.2:
                tool.status = "abandoned"
                tool.abandoned_at = _now()
                tool.abandoned_reason = (
                    f"low_success_ratio ({tool.success_count}/{tool.use_count} = "
                    f"{ratio:.2f})"
                )
        save_tool(tool)
        return True
    except Exception as e:
        print(f"[tool_catalog] record_tool_use 실패: {e}")
        return False


# ── 텔레그램 요약 ──────────────────────────────────────────────────────
def summarize_catalog_for_telegram(top_n: int = 5) -> str:
    """catalog 상태 markdown 요약."""
    all_tools = list_all_tools()
    if not all_tools:
        return "🧰 *도구 catalog* — 비어있음"

    by_status: dict = {}
    for t in all_tools:
        by_status.setdefault(t.status, []).append(t)

    lines = [f"🧰 *도구 catalog* (총 {len(all_tools)}개)"]
    for s in ("active", "pending_approval", "rejected", "abandoned"):
        n = len(by_status.get(s, []))
        if n > 0:
            emoji = {"active": "✅", "pending_approval": "⏳",
                     "rejected": "❌", "abandoned": "💤"}.get(s, "")
            lines.append(f"  {emoji} {s}: {n}")

    pending = by_status.get("pending_approval", [])
    if pending:
        lines.append("\n*승인 대기*:")
        for t in pending[:top_n]:
            lines.append(f"  • `{t.id[:14]}` ({t.type}) — {t.description[:60]}")

    active = sorted(by_status.get("active", []),
                    key=lambda x: x.created_at, reverse=True)
    if active:
        lines.append("\n*최근 active*:")
        for t in active[:top_n]:
            lines.append(
                f"  • `{t.id[:14]}` ({t.type}) → {t.target_indicator} "
                f"| 사용 {t.use_count}회·성공 {t.success_count}"
            )
    return "\n".join(lines)


def summarize_new_tool_for_telegram(tool: Tool) -> str:
    """단일 도구 신규 제작 알림용."""
    if not tool:
        return ""
    if tool.status == "pending_approval":
        header = f"⏳ *신규 도구 — 승인 필요* ({tool.type})"
    else:
        header = f"🛠 *신규 도구 등록 자동* ({tool.type})"
    lines = [
        header,
        f"  ID: `{tool.id[:14]}`",
        f"  타입: {tool.type}",
        f"  목표 지표: {tool.target_indicator}",
        f"  설명: {tool.description[:150]}",
    ]
    if tool.status == "pending_approval":
        lines.append(
            f"\n  ❗ 코드/API 도구는 보안상 승인 후 사용 — 검토 후 "
            f"`/approve_tool {tool.id}` 또는 `/reject_tool {tool.id}` 명령."
        )
    return "\n".join(lines)
