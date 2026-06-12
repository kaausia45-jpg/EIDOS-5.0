# eidos_tom_telegram_bridge.py
# [2026-05-26 Option A3] ToM 에이전트 ↔ 텔레그램 HITL bridge.
#
# 세 종류의 승인 / 두 종류의 콜백:
#   1) ask_user (자유 질문)        — async (question) -> str (응답 텍스트·"" 거절/timeout)
#   2) external_action (사전 승인)  — async (action_id, args) -> bool (실 dispatch 여부)
#
# 정책 모드 (Stage 첨부 시 사용자가 카테고리별로 선택):
#   POLICY_ALLOW   — 텔레그램 안 띄움·즉시 실행 (외부효과는 yolo 처럼·ask_user 는 의미없음 → BLOCK 취급)
#   POLICY_PROMPT  — 텔레그램 사전 승인 카드·✅/✏️ 면 진행, ❌/timeout 면 차단
#   POLICY_BLOCK   — 텔레그램 안 띄움·차단 (외부효과 DRY_RUN·ask_user "" 반환)
#
# send_generic_approval 의 callback 시그니처: callback(action: str, final_text: str)
#   action ∈ {"approved", "edited", "rejected"}
#
# 본 모듈은 EIDOS GUI 가 import 안 함 (텔레그램 라이브러리 의존성 격리).
# ToM 에이전트 runner (asyncio loop 안) 에서만 호출.

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict, Optional


# ── 정책 상수 ─────────────────────────────────────────────────────────
POLICY_ALLOW  = "allow"
POLICY_PROMPT = "prompt"
POLICY_BLOCK  = "block"

POLICY_LABELS = {
    POLICY_ALLOW:  "🟢 자동 승인 (즉시 실행)",
    POLICY_PROMPT: "🟡 텔레그램 사전 승인 (✅/❌)",
    POLICY_BLOCK:  "🔴 차단 (DRY-RUN / 거절)",
}


# 외부효과 카테고리 prefix (가장 긴 매칭 우선)
EXTERNAL_PREFIXES = (
    "eidos.tool.file.delete",      # 파일 삭제 (현재 default 에 없으나 미래 대비)
    "eidos.tool.message.email",    # 이메일
    "eidos.tool.message.telegram", # 텔레그램 (자기 자신 통해)
    "eidos.tool.message.",         # 메시지 일반 (다른 채널)
    "eidos.tool.web.",             # 웹 자동화
)


# ── 텔레그램 bot 호출 헬퍼 ────────────────────────────────────────────
async def _wait_for_callback(
    sender_coro_factory: Callable[[Callable[[str, str], None]], Awaitable[Optional[str]]],
    timeout_sec: float,
) -> tuple[str, str]:
    """rid 만들고 callback 응답까지 await. 반환: (action, text).

    sender_coro_factory(callback) -> rid coroutine. 본 헬퍼가 callback 만들어 주입.
    """
    event = asyncio.Event()
    holder = {"action": "", "text": ""}

    def _cb(action: str, text: str):
        # 텔레그램 bot 의 _fire_callback 이 동기 호출.
        holder["action"] = action
        holder["text"] = text
        try:
            event.set()
        except Exception:
            pass

    rid = None
    try:
        rid = await sender_coro_factory(_cb)
    except Exception as e:
        print(f"⚠️ [tom_telegram_bridge] send 실패: {e}")
        return ("", "")

    if not rid:
        return ("", "")

    try:
        await asyncio.wait_for(event.wait(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        print(f"⚠️ [tom_telegram_bridge] timeout (rid={rid})")
        return ("timeout", "")
    except Exception as e:
        print(f"⚠️ [tom_telegram_bridge] await 실패: {e}")
        return ("", "")

    return (holder["action"], holder["text"])


def _get_bot_or_none():
    try:
        from eidos_telegram_bot import get_bot
    except Exception as e:
        print(f"⚠️ [tom_telegram_bridge] bot import 실패: {e}")
        return None
    try:
        bot = get_bot()
    except Exception as e:
        print(f"⚠️ [tom_telegram_bridge] get_bot 실패: {e}")
        return None
    if not bot.is_configured():
        print("⚠️ [tom_telegram_bridge] 텔레그램 미설정")
        return None
    return bot


# ── 외부 API ──────────────────────────────────────────────────────────
async def ask_user_via_telegram(
    question: str,
    stage_name: str = "",
    timeout_sec: float = 300.0,
) -> str:
    """EIDOS 의 ask_user → 텔레그램 push → 사용자 ✅/❌/✏️ 응답.

    Returns: 사용자 응답 텍스트 (approved/edited 면 final_text·"승인" fallback).
             rejected/timeout/오류 시 빈 문자열.
    """
    bot = _get_bot_or_none()
    if bot is None:
        return ""

    async def _send(cb):
        return await bot.send_generic_approval(
            risk_label="MEDIUM",
            action=f"ToM ask_user · {stage_name or '(stage)'}",
            target="",
            content=question[:1000],
            context={
                "why": "ToM 에이전트 자율 모드에서 사용자 응답 필요",
            },
            callback=cb,
        )

    action, text = await _wait_for_callback(_send, timeout_sec=timeout_sec)
    if action in ("approved", "edited"):
        return text or "(승인)"
    return ""


async def approve_external_via_telegram(
    action_id: str,
    args: Dict[str, Any],
    stage_name: str = "",
    timeout_sec: float = 300.0,
) -> bool:
    """외부효과 사전 승인 → 텔레그램 push → bool.

    Returns: True (실 dispatch 진행) / False (DRY-RUN 유지).
    """
    bot = _get_bot_or_none()
    if bot is None:
        return False

    # risk_label — file.delete/message 는 HIGH·web 은 MEDIUM
    if "file.delete" in action_id or "message" in action_id:
        risk = "HIGH"
    else:
        risk = "MEDIUM"

    target = str(args.get("target") or args.get("url") or args.get("recipient")
                 or args.get("selector") or "")[:200]
    content = str(args.get("content") or args.get("body") or args.get("text")
                  or args.get("brief") or args.get("question") or "")[:1000]

    async def _send(cb):
        return await bot.send_generic_approval(
            risk_label=risk,
            action=f"ToM 외부효과 · {action_id}",
            target=target,
            content=content,
            context={
                "why": f"ToM 에이전트 ({stage_name or '(stage)'}) 가 외부효과 실행 결정",
            },
            callback=cb,
        )

    action, _text = await _wait_for_callback(_send, timeout_sec=timeout_sec)
    return action in ("approved", "edited")


# ── 정책 기반 콜백 factory ────────────────────────────────────────────
def make_ask_user_callback(
    policy: str,
    stage_name: str = "",
    timeout_sec: float = 300.0,
) -> Optional[Callable[[str], Awaitable[str]]]:
    """ask_user 콜백 빌드.

    POLICY_BLOCK / POLICY_ALLOW → callback None (runner 가 SKIP 처리).
      · ALLOW 는 ask_user 에 적용 의미 X (질문에 자동 yes 는 위험) → BLOCK 으로 취급.
    POLICY_PROMPT → 텔레그램 push 콜백 반환.
    """
    if policy != POLICY_PROMPT:
        return None

    async def _ask(question: str) -> str:
        return await ask_user_via_telegram(question, stage_name=stage_name,
                                           timeout_sec=timeout_sec)
    return _ask


def make_external_approve_callback(
    policy_by_prefix: Dict[str, str],
    stage_name: str = "",
    timeout_sec: float = 300.0,
) -> Optional[Callable[[str, Dict[str, Any]], Awaitable[bool]]]:
    """외부효과 prefix 별 policy → 단일 콜백 반환.

    policy_by_prefix 예:
      {"eidos.tool.web.": "prompt",
       "eidos.tool.message.": "prompt",
       "eidos.tool.file.delete": "block"}

    매칭 알고리즘 — prefix 가 action_id 시작·가장 긴 prefix 우선.
    매칭 안 되면 default "prompt".

    None 반환 시 — 모든 prefix 가 ALLOW 같이 콜백 불필요한 경우는 아님
    (현재 prompt 가 default 라 항상 콜백 반환).
    """
    # 적어도 하나가 prompt 면 콜백 필요. 모두 allow / block 이면 캡쳐만 하는 콜백도 OK.
    has_prompt = any(p == POLICY_PROMPT for p in policy_by_prefix.values())
    has_allow  = any(p == POLICY_ALLOW for p in policy_by_prefix.values())
    if not has_prompt and not has_allow:
        return None   # 모두 BLOCK → 콜백 없으면 runner 의 default DRY_RUN

    async def _approve(action_id: str, args: Dict[str, Any]) -> bool:
        # prefix 매칭 (가장 긴 매칭 우선)
        matched_policy = None
        matched_len = 0
        for prefix, pol in policy_by_prefix.items():
            if action_id.startswith(prefix) and len(prefix) > matched_len:
                matched_policy = pol
                matched_len = len(prefix)
        policy = matched_policy or POLICY_PROMPT   # default

        if policy == POLICY_ALLOW:
            return True
        if policy == POLICY_BLOCK:
            return False
        # PROMPT
        return await approve_external_via_telegram(
            action_id, args, stage_name=stage_name, timeout_sec=timeout_sec,
        )
    return _approve


__all__ = [
    "POLICY_ALLOW", "POLICY_PROMPT", "POLICY_BLOCK", "POLICY_LABELS",
    "EXTERNAL_PREFIXES",
    "ask_user_via_telegram", "approve_external_via_telegram",
    "make_ask_user_callback", "make_external_approve_callback",
]
