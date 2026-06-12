"""
TelegramHitlAdapter — HITL via Telegram (TODO 운영 4번)
==========================================================

GUI 외 채널 — 사용자가 EIDOS 앞에 없을 때도 (외출/원격) HITL 가능.
inline_keyboard 4 버튼 + callback_query polling.

의존성 0 — `urllib.request` (stdlib) + `json` 만. python-telegram-bot 같은
무거운 라이브러리 회피.

⚠️ **별도 bot token 권장** — EIDOS 가 이미 실행 중인 bot 과 같은 token 으로
   polling 하면 update 충돌. 사용자가 별도 bot 만들어 token/chat_id 주입.

흐름:
  1. ask(question) → sendMessage(text + inline_keyboard 4 버튼)
  2. getUpdates polling (poll_interval 마다, timeout_sec 한도)
  3. callback_query 받으면 → answerCallbackQuery (사용자 ack)
  4. HumanResponse(decision=...) 반환

timeout / send 실패 / 외부 에러 → DECISION_ABSTAIN (conservative).
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# HITL Adapter Protocol 호환 (eidos_hitl_gate 의 ask 시그니처)
from eidos_hitl_gate import (
    HumanResponse,
    DECISION_APPROVE, DECISION_REJECT, DECISION_MODIFY, DECISION_ABSTAIN,
)


__autonomous_chain_forbidden__ = True


# ──────────────────────────────────────────────
#  상수 — Telegram Bot API
# ──────────────────────────────────────────────

_API_BASE = "https://api.telegram.org/bot{token}/{method}"
_TG_TEXT_LIMIT = 4000           # Telegram message text cap (실제 4096, 마진)
_DEFAULT_POLL_INTERVAL = 2.0     # 초
_DEFAULT_TIMEOUT_SEC = 600       # 10 분 default — 외부 사용자 응답 여유
_GETUPDATES_LONG_POLL = 1        # Telegram long poll timeout (초)


# decision → callback_data 매핑 (Telegram 64자 제한 안)
_CALLBACK_DATA = {
    DECISION_APPROVE:  "hitl_approve",
    DECISION_REJECT:   "hitl_reject",
    DECISION_MODIFY:   "hitl_modify",
    DECISION_ABSTAIN:  "hitl_abstain",
}
_CALLBACK_DATA_REV = {v: k for k, v in _CALLBACK_DATA.items()}


def _build_inline_keyboard() -> Dict[str, Any]:
    """4 버튼 inline_keyboard."""
    return {
        "inline_keyboard": [
            [
                {"text": "✅ 승인", "callback_data": _CALLBACK_DATA[DECISION_APPROVE]},
                {"text": "✏️ 수정 후 진행", "callback_data": _CALLBACK_DATA[DECISION_MODIFY]},
            ],
            [
                {"text": "❌ 거부", "callback_data": _CALLBACK_DATA[DECISION_REJECT]},
                {"text": "⏭ 보류 (skip)", "callback_data": _CALLBACK_DATA[DECISION_ABSTAIN]},
            ],
        ]
    }


# ──────────────────────────────────────────────
#  TelegramHitlAdapter
# ──────────────────────────────────────────────

class TelegramHitlAdapter:
    """HITLAdapter Protocol 의 Telegram 구현. 외부 채널 HITL.

    Args:
        bot_token: Telegram bot 토큰 (BotFather 발급).
        chat_id:   대상 chat (사용자 본인 또는 관리자 group).
        poll_interval: getUpdates 호출 간격 (default 2s).
        timeout_sec: 사용자 응답 대기 한도 (default 600s = 10분).
        api_base: API base URL (테스트용 override 가능).
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: int,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
        api_base: str = _API_BASE,
    ):
        self.bot_token = str(bot_token)
        self.chat_id = int(chat_id)
        self.poll_interval = float(poll_interval)
        self.timeout_sec = float(timeout_sec)
        self.api_base = str(api_base)
        self._last_update_id: int = 0   # offset 추적 (consumed update 이후만)

    # ── 외부 인터페이스 — HITLAdapter Protocol ──

    async def ask(
        self, question: str, _context: Dict[str, Any], /,
    ) -> HumanResponse:
        # 1) sendMessage with inline_keyboard
        msg_id = await self._send_question(question)
        if msg_id is None:
            return HumanResponse(
                decision=DECISION_ABSTAIN,
                rationale="Telegram sendMessage 실패",
            )

        # 2) callback_query polling
        deadline = time.time() + self.timeout_sec
        while time.time() < deadline:
            cb = await self._poll_callback(msg_id)
            if cb is not None:
                cq_id, decision = cb
                # 3) 사용자에게 ack ("받았습니다")
                try:
                    await self._answer_callback_query(cq_id, decision)
                except Exception:
                    pass  # ack 실패는 무시 (decision 은 받았음)
                # 4) inline_keyboard 비활성화 (선택, 실패해도 무시)
                try:
                    await self._edit_keyboard_remove(msg_id, decision)
                except Exception:
                    pass
                return HumanResponse(
                    decision=decision,
                    rationale=f"telegram chat_id={self.chat_id}",
                    metadata={"msg_id": msg_id, "callback_id": cq_id},
                )
            await asyncio.sleep(self.poll_interval)

        # timeout
        return HumanResponse(
            decision=DECISION_ABSTAIN,
            rationale=f"Telegram timeout {self.timeout_sec}s",
        )

    # ── 내부 ──

    async def _send_question(self, question: str) -> Optional[int]:
        """sendMessage. 반환 = message_id (성공 시) / None (실패)."""
        text = (question or "")[:_TG_TEXT_LIMIT]
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "reply_markup": json.dumps(_build_inline_keyboard()),
        }
        try:
            resp = await self._api("sendMessage", payload)
        except Exception:
            return None
        if not resp.get("ok"):
            return None
        result = resp.get("result") or {}
        msg_id = result.get("message_id")
        return int(msg_id) if isinstance(msg_id, int) else None

    async def _poll_callback(
        self, target_msg_id: int,
    ) -> Optional[tuple]:
        """getUpdates → (callback_query_id, decision) 또는 None.

        target_msg_id 와 일치하는 callback_query 만 매칭. 다른 메시지의
        callback (예: 이전 ask) 은 update_id 만 소비하고 skip.
        """
        try:
            resp = await self._api("getUpdates", {
                "offset": self._last_update_id + 1,
                "timeout": _GETUPDATES_LONG_POLL,
            })
        except Exception:
            return None
        if not resp.get("ok"):
            return None
        updates = resp.get("result") or []
        if not isinstance(updates, list):
            return None

        matched: Optional[tuple] = None
        for upd in updates:
            if not isinstance(upd, dict):
                continue
            uid = upd.get("update_id")
            if isinstance(uid, int) and uid > self._last_update_id:
                self._last_update_id = uid
            cb = upd.get("callback_query")
            if not isinstance(cb, dict):
                continue
            msg = cb.get("message") or {}
            if msg.get("message_id") != target_msg_id:
                continue
            data = cb.get("data") or ""
            decision = _CALLBACK_DATA_REV.get(str(data))
            if decision is None:
                continue
            cq_id = cb.get("id")
            if matched is None:
                matched = (cq_id, decision)
            # 모든 update 의 update_id 를 소비한 후 첫 매칭 반환

        return matched

    async def _answer_callback_query(
        self, callback_query_id: Any, decision: str,
    ) -> None:
        """answerCallbackQuery — 사용자에게 toast 알림."""
        await self._api("answerCallbackQuery", {
            "callback_query_id": str(callback_query_id),
            "text": f"✓ '{decision}' 처리 중",
            "show_alert": False,
        })

    async def _edit_keyboard_remove(
        self, message_id: int, decision: str,
    ) -> None:
        """edited 메시지에 결정 표시 + inline_keyboard 제거."""
        await self._api("editMessageReplyMarkup", {
            "chat_id": self.chat_id,
            "message_id": int(message_id),
            "reply_markup": json.dumps({"inline_keyboard": []}),
        })

    # ── HTTP — urllib (stdlib only) ──

    async def _api(
        self, method: str, payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Telegram Bot API 호출. asyncio.to_thread 로 blocking urlopen 비동기화."""
        url = self.api_base.format(token=self.bot_token, method=method)
        # POST x-www-form-urlencoded — 모든 telegram 메서드 호환
        body = urlencode(payload).encode("utf-8")
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        def _do_request():
            req = Request(url, data=body, headers=headers, method="POST")
            with urlopen(req, timeout=15) as resp:  # type: ignore[arg-type]
                raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)

        return await asyncio.to_thread(_do_request)


# ──────────────────────────────────────────────
#  Helper — settings.json 에서 로드
# ──────────────────────────────────────────────

def _telegram_config_candidates(settings_path: str) -> List[str]:
    """`telegram_config.json` 후보 경로 — settings 폴더의 eidos_files/ 우선, cwd 폴백."""
    import os
    cands: List[str] = []
    try:
        base = os.path.dirname(os.path.abspath(settings_path)) or os.getcwd()
    except Exception:
        base = os.getcwd()
    cwd = os.getcwd()
    for root in (base, cwd):
        cands.append(os.path.join(root, "eidos_files", "telegram_config.json"))
        cands.append(os.path.join(root, "telegram_config.json"))
    # 순서 유지 중복 제거
    seen: set = set()
    out: List[str] = []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _load_telegram_config(settings_path: str) -> Optional[tuple]:
    """eidos_files/telegram_config.json 에서 (token, chat_id_raw) 읽기. 없으면 None.

    키는 단순한 `token` / `chat_id` (사용자가 실제 넣어둔 형식).
    """
    for path in _telegram_config_candidates(settings_path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            continue
        if not isinstance(cfg, dict):
            continue
        token = (cfg.get("token") or "").strip()
        chat_id_raw = cfg.get("chat_id")
        if token and chat_id_raw is not None:
            return (token, chat_id_raw)
    return None


def telegram_adapter_from_settings(
    settings_path: str = "eidos_settings.json",
    *,
    token_key: str = "telegram_hitl_token",
    chat_id_key: str = "telegram_hitl_chat_id",
    **adapter_kwargs: Any,
) -> Optional[TelegramHitlAdapter]:
    """token/chat_id 읽어 인스턴스화.

    우선순위:
      1) eidos_files/telegram_config.json 의 `token` / `chat_id` (사용자 설정 위치).
      2) eidos_settings.json 의 `telegram_hitl_token` / `telegram_hitl_chat_id` (기존 호환).

    별도 bot 권장 — EIDOS 본 bot 과 polling 충돌 회피. 사용자가 BotFather 에서
    새 bot 만들어 등록.

    None 반환 — 키 없거나 파일 없거나 invalid.
    """
    token = ""
    chat_id_raw: Any = None

    # 1) telegram_config.json (사용자가 실제로 넣어둔 위치)
    cfg = _load_telegram_config(settings_path)
    if cfg is not None:
        token, chat_id_raw = cfg

    # 2) eidos_settings.json 폴백 (기존 호환)
    if not token or chat_id_raw is None:
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
            token = token or (settings.get(token_key) or "").strip()
            if chat_id_raw is None:
                chat_id_raw = settings.get(chat_id_key)
        except Exception:
            pass

    if not token or chat_id_raw is None:
        return None
    try:
        chat_id = int(chat_id_raw)
    except (TypeError, ValueError):
        return None
    return TelegramHitlAdapter(token, chat_id, **adapter_kwargs)
