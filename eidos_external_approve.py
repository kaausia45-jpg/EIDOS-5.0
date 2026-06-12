# eidos_external_approve.py
# [Wave3 MVP 2026-05-28] 자율 액션의 외부효과 승인 게이트.
#
# 사용 흐름:
#   run_stage_one_tick → execute_eidos_action (외부효과 action) →
#     on_external_approve callback → request_external_approval() →
#       1) is_auto_approved 화이트리스트 → True 즉시 반환
#       2) 그 외 → 채팅 + AIRA TTS + 텔레그램 3채널 동시 알림
#                 사용자가 채팅에 "진행"/"취소" 입력하면 그 결과 반영
#                 timeout (default 5분) → False (안전 default)
#
# 화이트리스트 default (settings.json `auto_approved_actions` 로 override 가능):
#   - eidos.meta.*           : 부수효과 0 (메타 인지·관찰·위임 등)
#   - eidos.tool.llm.write   : stage outputs sandbox 내 파일 작성 (이미 safe)
#   - eidos.tool.web.search  : 읽기 전용 외부 검색
#   - eidos.tool.web.fetch   : 읽기 전용 페치
#
# 모든 단계 graceful — telegram/AIRA/chat 어디서 실패해도 게이트 자체는 동작.

from __future__ import annotations

import asyncio
import fnmatch
from typing import Any, Callable, Optional


# ── 기본 화이트리스트 ──────────────────────────────────────────────────
DEFAULT_AUTO_APPROVED = (
    "eidos.meta.*",
    "eidos.tool.llm.write",
    "eidos.tool.web.search",
    "eidos.tool.web.fetch",
)

# 응답 키워드 (사용자가 채팅에 입력)
_APPROVE_TOKENS = ("진행", "예", "ok", "OK", "네", "yes", "Yes", "Y", "y", "✅")
_REJECT_TOKENS = ("취소", "아니", "아니오", "no", "NO", "No", "N", "n", "❌", "그만")


def is_auto_approved(action_id: str, whitelist=None) -> bool:
    """action_id 가 화이트리스트 패턴 (prefix·glob) 매칭되면 True.

    whitelist 가 None 이면 DEFAULT_AUTO_APPROVED 사용. graceful — 어떤 입력
    예외도 False 반환 (안전 default).
    """
    try:
        if not action_id:
            return False
        patterns = whitelist if whitelist else DEFAULT_AUTO_APPROVED
        for p in patterns:
            try:
                if not p:
                    continue
                # fnmatch 는 glob — eidos.meta.* 같은 패턴 매칭 가능
                if fnmatch.fnmatchcase(action_id, str(p)):
                    return True
                # 추가로 prefix 매칭 (사용자가 "eidos.tool.web" 만 적은 경우)
                if action_id.startswith(str(p).rstrip("*").rstrip(".")):
                    if str(p).endswith("*") or action_id == str(p):
                        return True
            except Exception:
                continue
        return False
    except Exception:
        return False


def classify_user_response(text: str) -> Optional[bool]:
    """사용자 채팅 응답을 True/False/None 으로 분류.

    None 이면 승인/거부 키워드 매칭 안 됨 (다른 입력 — 보류).
    공백·대소문자 무시.
    """
    if not text:
        return None
    t = text.strip()
    if not t:
        return None
    # exact match (짧은 응답만)·token in t
    for tok in _APPROVE_TOKENS:
        if t == tok or t.lower() == tok.lower():
            return True
    for tok in _REJECT_TOKENS:
        if t == tok or t.lower() == tok.lower():
            return False
    # 부분 매칭 — 짧은 응답일 때만 (10자 이하)
    if len(t) <= 10:
        for tok in _APPROVE_TOKENS:
            if tok in t:
                return True
        for tok in _REJECT_TOKENS:
            if tok in t:
                return False
    return None


def _format_action_for_user(action_id: str, args: dict) -> str:
    """채팅·AIRA·텔레그램에 표시할 짧은 요약. 80자 이내."""
    try:
        if not args:
            return action_id
        # args 의 핵심 키 1~2개만 노출
        key_hints = []
        for k in ("target", "url", "to", "content", "topic", "query"):
            v = args.get(k)
            if v:
                v_s = str(v)[:40]
                key_hints.append(f"{k}={v_s}")
                if len(key_hints) >= 2:
                    break
        if key_hints:
            return f"{action_id} ({', '.join(key_hints)})"
        return action_id
    except Exception:
        return action_id


def _send_telegram_best_effort(message: str, title: str = "🤖 EIDOS 승인 요청") -> bool:
    """eidos_telegram_bot 사용. 부재·미설정·예외 모두 graceful False.

    self_improvement_loop._send_telegram_best_effort 와 동일 패턴 (재사용 위해
    별도 함수 안 부르고 inline — 자기개선 loop 가 사라져도 동작 유지).
    """
    try:
        from eidos_telegram_bot import get_bot
        bot = get_bot()
        if not bot.is_configured():
            return False
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(bot.send_notification(message, title=title))
            else:
                loop.run_until_complete(bot.send_notification(message, title=title))
            return True
        except RuntimeError:
            asyncio.run(bot.send_notification(message, title=title))
            return True
    except Exception as e:
        print(f"[external_approve] 텔레그램 발송 실패 (graceful): {e}")
        return False


async def request_external_approval(
    action_id: str,
    args: dict,
    stage_id: str = "",
    chat_window: Any = None,
    aira_summon: Optional[Callable] = None,
    timeout_sec: float = 300.0,
    whitelist=None,
) -> bool:
    """외부효과 액션 승인 게이트.

    Args:
      action_id: 실행 예정 action_id.
      args: action args dict.
      stage_id: 어느 stage 에서 발생한 요청인지 (사용자 알림용).
      chat_window: ChatWindow 인스턴스 — 채팅 표시 + _pending_external_approval
                  슬롯 설정에 사용. None 이면 채팅 채널 skip.
      aira_summon: chat_window._summon_aira 함수 ref. None 이면 AIRA skip.
      timeout_sec: 응답 대기 시간 (default 5분). 초과 시 False 반환 (안전).
      whitelist: 화이트리스트 (settings 의 auto_approved_actions). None 이면 default.

    Returns:
      True  — 승인됨 (또는 화이트리스트 즉시 통과).
      False — 거부됨 / timeout / 어떤 오류 (안전 default).
    """
    # 1. 화이트리스트 즉시 통과
    if is_auto_approved(action_id, whitelist):
        print(f"[external_approve] {action_id} → 화이트리스트 통과 (즉시 승인)")
        return True

    summary = _format_action_for_user(action_id, args or {})
    stage_hint = f" · stage={stage_id[:16]}" if stage_id else ""

    # 2. 채팅창에 승인 요청 표시 + future 슬롯 설치
    future: asyncio.Future = asyncio.get_event_loop().create_future()
    slot_key = f"approval_{id(future)}"
    try:
        if chat_window is not None:
            # _pending_external_approval dict 가 없으면 생성
            if not hasattr(chat_window, "_pending_external_approval"):
                chat_window._pending_external_approval = {}
            chat_window._pending_external_approval[slot_key] = future

            msg = (
                f"🤖 **[자율 액션 승인 요청]**{stage_hint}\n"
                f"├─ action: `{summary}`\n"
                f"├─ ✅ 진행하려면 → `진행` 입력\n"
                f"└─ ❌ 취소하려면 → `취소` 입력  ({int(timeout_sec)}초 timeout)"
            )
            try:
                chat_window.append_message(msg, "system")
            except Exception as e_cm:
                print(f"[external_approve] 채팅 표시 실패 (graceful): {e_cm}")
    except Exception as e_slot:
        print(f"[external_approve] 슬롯 설치 실패 (graceful): {e_slot}")

    # 3. AIRA 등장 + TTS 발화
    try:
        if aira_summon is not None:
            try:
                from eidos_aira_presence import DURATION_MS
                dur = int(DURATION_MS.get("approval_request", 20000))
            except Exception:
                dur = 20000
            aira_summon(
                reason="approval_request",
                expr_key="concerned",
                text=f"에스더 님, {summary[:40]} 해도 될까요?",
                duration_ms=dur,
                force=True,
            )
    except Exception as e_aira:
        print(f"[external_approve] AIRA 등장 실패 (graceful): {e_aira}")

    # 4. 텔레그램 best-effort (one-way 알림 — 인라인 버튼은 후속)
    try:
        tg_msg = (
            f"[EIDOS 승인 요청]{stage_hint}\n"
            f"action: {summary}\n\n"
            f"채팅창에서 '진행' 또는 '취소' 로 응답해주세요. "
            f"({int(timeout_sec)}초 timeout 후 자동 거부)"
        )
        _send_telegram_best_effort(tg_msg)
    except Exception as e_tg:
        print(f"[external_approve] 텔레그램 알림 실패 (graceful): {e_tg}")

    # 5. 응답 대기 (timeout 시 False)
    try:
        result = await asyncio.wait_for(future, timeout=timeout_sec)
        print(f"[external_approve] {action_id} → 응답: {result}")
        return bool(result)
    except asyncio.TimeoutError:
        print(f"[external_approve] {action_id} → timeout (안전 default = 거부)")
        # 채팅에 timeout 알림 (graceful)
        try:
            if chat_window is not None:
                chat_window.append_message(
                    f"⌛ 승인 timeout — `{summary}` 거부됨 (안전 default)",
                    "system",
                )
        except Exception:
            pass
        return False
    finally:
        # 슬롯 정리 — 다음 요청에 영향 없게
        try:
            if chat_window is not None and hasattr(chat_window, "_pending_external_approval"):
                chat_window._pending_external_approval.pop(slot_key, None)
        except Exception:
            pass


def resolve_pending_approval(chat_window: Any, user_text: str) -> bool:
    """채팅 메시지가 승인/거부 응답인지 검사 + 매칭되면 가장 오래된 pending future
    에 결과 set. send_message hook 에서 호출.

    Returns:
      True  — 응답 매칭 + future resolve 완료 (일반 chat 흐름 차단해야 함).
      False — 응답 키워드 매칭 안 됨 (일반 chat 진행).
    """
    try:
        verdict = classify_user_response(user_text)
        if verdict is None:
            return False
        slots = getattr(chat_window, "_pending_external_approval", None)
        if not slots:
            return False
        # 가장 오래된 (FIFO) future 에 결과 set
        for key in list(slots.keys()):
            fut = slots.get(key)
            if fut is None or fut.done():
                slots.pop(key, None)
                continue
            try:
                fut.set_result(bool(verdict))
                slots.pop(key, None)
                return True
            except Exception:
                slots.pop(key, None)
                continue
        return False
    except Exception as e:
        print(f"[external_approve] resolve_pending_approval 실패 (graceful): {e}")
        return False
