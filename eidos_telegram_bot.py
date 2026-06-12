# eidos_telegram_bot.py
# ──────────────────────────────────────────────────────────────────────────────
# EIDOS ↔ 텔레그램 Human-in-the-Loop 봇
#
# 역할:
#   1. EIDOS가 고객 메시지 초안 작성 후 텔레그램으로 승인 요청
#   2. 승준이 ✅/✏️/❌ 버튼으로 응답
#   3. 승인 시 크몽 메시지창에 JS 자동 발송
#
# 사용법:
#   python eidos_telegram_bot.py          # 봇 단독 실행 (테스트용)
#   EIDOS에서 import해서 send_approval_request() 호출
#
# 설정:
#   eidos_files/telegram_config.json 에 토큰/chat_id 저장
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import html
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── 텔레그램 라이브러리 (python-telegram-bot v20+) ───────────────────────────
try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Bot
    from telegram.ext import (
        Application, CommandHandler, CallbackQueryHandler,
        MessageHandler, filters, ContextTypes,
    )
    TELEGRAM_LOADED = True
except ImportError:
    TELEGRAM_LOADED = False
    print("⚠️ [Telegram] python-telegram-bot 미설치. 설치 명령: pip install python-telegram-bot")

# ── 설정 파일 경로 ────────────────────────────────────────────────────────────
CONFIG_PATH = os.path.join("eidos_files", "telegram_config.json")

# ─────────────────────────────────────────────────────────────────────────────
# 설정 로드/저장
# ─────────────────────────────────────────────────────────────────────────────

def load_config() -> Dict[str, Any]:
    """telegram_config.json 로드. 없으면 빈 dict 반환."""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_config(data: Dict[str, Any]):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# 대기 중인 승인 요청 저장소
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ApprovalRequest:
    """승인 대기 중인 메시지 하나"""
    request_id:    str
    customer_name: str          # 고객 이름 또는 ID
    customer_msg:  str          # 고객이 보낸 원본 메시지
    draft_reply:   str          # EIDOS가 작성한 답장 초안
    callback:      Optional[Callable[[str, str], None]] = None
    # callback(action, final_text)
    #   action: "approved" | "rejected" | "edited"
    #   final_text: 최종 발송할 텍스트
    created_at:    float = field(default_factory=time.time)
    telegram_msg_id: Optional[int] = None  # 텔레그램 메시지 ID (수정용)
    # [Phase 2] 큐 인덱스 + finalize 파이프라인 메타
    seq:           int  = 0      # 요청 발송 시점 순번 (전역 증가)
    approved_at:   float = 0.0   # 사용자가 버튼 누른 시각 (duration 계산용)
    awaiting_finalize: bool = False  # approve/edit 후 dispatcher finalize 대기 상태
    # [Phase 3] edit preview 2단계 — 수정 입력 받은 후 콜백 발화 전 확인 버튼 단계
    edited_text:   str = ""      # 사용자가 입력한 수정본 (preview 중 보관)


# 전역 대기 저장소  { request_id: ApprovalRequest }
_pending: Dict[str, ApprovalRequest] = {}

# [Phase B 2026-04-24] CAPTCHA "✅ 풀었음" 인라인 버튼 콜백 레지스트리.
# token → (on_solved_callback, customer_label_for_logging).
# dispatcher 가 _notify_telegram_captcha 시점에 등록, 사용자가 버튼 누르면 pop+호출.
# Why: 캡챠 알림은 ApprovalRequest 의 무거운 메타(고객명/초안 등) 가 불필요한 "재개 신호"
# 한 번이면 충분 — 별도 가벼운 레지스트리로 분리해 _pending 의 lifecycle 과 분리.
_captcha_callbacks: Dict[str, Callable[[], None]] = {}

# [Phase 2] 요청 순번 카운터 — 헤더에 "#N" 으로 노출. 프로세스 전역.
_rid_seq: int = 0

# _pending read-then-modify 경로 race 차단용 async lock.
# Grok 비판: 버튼 콜백 + finalize_approval 이 동시 도착 시 같은 rid 에
# 접근해 중복 처리·이중 del KeyError 가능. asyncio.Lock 은 event loop
# 바인딩이라 import 시점에 만들 수 없어서 lazy init.
_PENDING_LOCK: Optional[asyncio.Lock] = None


def _get_pending_lock() -> asyncio.Lock:
    global _PENDING_LOCK
    if _PENDING_LOCK is None:
        _PENDING_LOCK = asyncio.Lock()
    return _PENDING_LOCK


# ─────────────────────────────────────────────────────────────────────────────
# EidosTelegramBot
# ─────────────────────────────────────────────────────────────────────────────

class EidosTelegramBot:
    """
    EIDOS ↔ 텔레그램 Human-in-the-Loop 브릿지.

    사용 예:
        bot = EidosTelegramBot()
        await bot.send_approval_request(
            customer_name="김원장",
            customer_msg="예약 가능한가요?",
            draft_reply="안녕하세요! 네, 예약 가능합니다...",
            callback=my_callback,
        )
    """

    def __init__(self):
        cfg = load_config()
        self.token   = cfg.get("token", "")
        self.chat_id = cfg.get("chat_id", "")
        self._app: Optional[Any] = None
        self._bot: Optional[Any] = None
        self._running = False
        # [Chat] 텔레그램 텍스트 메시지 → EIDOS 챗 포워딩용 핸들러.
        # GUI가 set_chat_handler로 등록. handler(user_text: str) -> str (async).
        self._chat_handler: Optional[Callable[[str], Any]] = None

    # ── 챗 핸들러 등록 ────────────────────────────────────────────────────────

    def set_chat_handler(self, handler: Optional[Callable[[str], Any]]):
        """
        텔레그램 사용자가 보낸 일반 텍스트를 EIDOS 챗으로 넘기는 핸들러 등록.
        handler는 async 함수여야 하며 (user_text: str) -> str 반환.
        None을 넘기면 비활성화 (기본 안내문 응답).
        """
        self._chat_handler = handler
        if handler:
            print("✅ [Telegram] EIDOS 챗 핸들러 등록됨")
        else:
            print("  [Telegram] 챗 핸들러 해제")

    # ── 초기화 ────────────────────────────────────────────────────────────────

    def is_configured(self) -> bool:
        return bool(self.token and self.chat_id)

    def configure(self, token: str, chat_id: str):
        """토큰과 chat_id 설정 후 저장."""
        self.token   = token.strip()
        self.chat_id = str(chat_id).strip()
        cfg = load_config()
        cfg["token"]   = self.token
        cfg["chat_id"] = self.chat_id
        save_config(cfg)
        print(f"✅ [Telegram] 설정 저장 완료 | chat_id={self.chat_id}")

    # ── 승인 요청 발송 ────────────────────────────────────────────────────────

    async def send_approval_request(
        self,
        customer_name: str,
        customer_msg:  str,
        draft_reply:   str,
        callback:      Optional[Callable[[str, str], None]] = None,
        request_id:    Optional[str] = None,
    ) -> Optional[str]:
        """
        텔레그램으로 승인 요청 메시지 발송.
        반환값: request_id (추적용)
        """
        if not TELEGRAM_LOADED:
            print("⚠️ [Telegram] 라이브러리 미설치 — 발송 스킵")
            return None
        if not self.is_configured():
            print("⚠️ [Telegram] 설정 미완료 — configure() 먼저 호출")
            return None

        import uuid
        rid = request_id or uuid.uuid4().hex[:8]

        # [Phase 1 재설계] 3줄 요약 우선. 고객 원문은 접기 블록.
        # [Phase 2] 시퀀스 번호 + 대기 건수 헤더.
        _name_safe  = _esc(customer_name)
        _draft_safe = _esc(draft_reply)
        _msg_safe   = _esc(customer_msg)
        _draft_preview = _draft_safe if len(_draft_safe) <= 500 else _draft_safe[:500] + "…"
        _msg_preview   = _msg_safe   if len(_msg_safe)   <= 300 else _msg_safe[:300]   + "…"
        _seq = _next_seq()
        _qtag = _queue_tag(_seq + 0)  # 이 요청은 아직 _pending 에 안 들어감 → N+1건 효과
        text = (
            f"📨 <b>크몽 고객 문의</b> · {_qtag}\n"
            f"<b>고객:</b> {_name_safe}\n"
            f"<b>초안:</b> <i>{_draft_preview}</i>\n"
            f"\n"
            f"━━━━━━━━\n"
            f"💬 <b>고객 메시지</b>\n"
            f"<blockquote>{_msg_preview}</blockquote>"
        )

        # 인라인 버튼
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ 그대로 전송",  callback_data=f"approve:{rid}"),
                InlineKeyboardButton("✏️ 수정 후 전송", callback_data=f"edit:{rid}"),
            ],
            [
                InlineKeyboardButton("❌ 취소",         callback_data=f"reject:{rid}"),
            ],
        ])

        try:
            bot = Bot(token=self.token)
            msg = await bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode="HTML",
                reply_markup=keyboard,
            )

            # 대기 저장소 등록
            _pending[rid] = ApprovalRequest(
                request_id=rid,
                customer_name=customer_name,
                customer_msg=customer_msg,
                draft_reply=draft_reply,
                callback=callback,
                telegram_msg_id=msg.message_id,
                seq=_seq,
            )
            print(f"  [Telegram] 승인 요청 발송 완료 | rid={rid} | 고객={customer_name} | seq=#{_seq}")
            return rid

        except Exception as e:
            print(f"  [Telegram] 발송 실패: {e}")
            return None

    # ── 범용 승인 요청 (Phase A Approval Gateway 전용) ────────────────────────

    async def send_generic_approval(
        self,
        risk_label:   str,
        action:       str,
        target:       str,
        content:      str,
        context:      Optional[Dict[str, Any]] = None,
        callback:     Optional[Callable[[str, str], None]] = None,
        request_id:   Optional[str] = None,
    ) -> Optional[str]:
        """
        Phase A — ActionDispatcher가 HIGH/MEDIUM 액션 실행 전 호출.

        차이:
         - 크몽 고객 문의 전용이 아니라 모든 액션 타입에 대응
         - risk_label(LOW/MEDIUM/HIGH)에 따라 이모지/버튼 레이아웃 동적 조정
         - context에 goal/task_prompt/url 등 참고 메타데이터
        callback(action, final_text):
         - action: "approved"(그대로) | "edited"(수정 후 실행) | "rejected"
         - final_text: edited 인 경우 수정된 content, 아니면 원본
        """
        if not TELEGRAM_LOADED:
            print("⚠️ [Telegram] 라이브러리 미설치 — 발송 스킵")
            return None
        if not self.is_configured():
            print("⚠️ [Telegram] 설정 미완료 — send_generic_approval 차단")
            return None

        rid = request_id or uuid.uuid4().hex[:8]
        ctx = context or {}

        # 리스크별 이모지
        risk_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(risk_label.upper(), "⚪")

        # ── [Phase 1 재설계] 3줄 요약 우선 + Why 노출 + 상세 접힘 ─────────────
        # 상단: Action / Target / Why — 2~3초 내 판단용
        # 하단: Content·Goal·Task·URL — 참고 메타
        _tgt_short = (target or "")[:200]
        _cnt_short = (content or "")[:400]
        _goal = str(ctx.get("goal", ""))[:160]
        _url  = str(ctx.get("url", ""))[:160]
        _task = str(ctx.get("task_prompt", ""))[:160]
        _why  = str(ctx.get("why", ""))[:200]  # [Phase 1] ActionStep.reason 에서 전달

        # 1) 헤더: 리스크 라벨 + [Phase 2] 시퀀스/큐 태그
        _seq = _next_seq()
        _qtag = _queue_tag(_seq)
        header = (
            f"{risk_emoji} <b>승인 요청</b> · <code>{_esc(risk_label)}</code> · {_qtag}"
        )

        # 2) 3줄 요약 (우선순위 순)
        top = [
            f"<b>Action:</b> <code>{_esc(action)}</code>",
        ]
        if _tgt_short:
            top.append(f"<b>Target:</b> <code>{_esc(_tgt_short)}</code>")
        if _why:
            top.append(f"<b>Why:</b> <i>{_esc(_why)}</i>")

        # 3) 상세 메타 (있을 때만)
        detail: list = []
        if _cnt_short:
            # Content 는 줄바꿈 유지가 필요해서 <pre>
            detail.append(f"📝 <b>Content</b>\n<pre>{_esc(_cnt_short)}</pre>")
        if _goal:
            detail.append(f"🧭 <b>Goal:</b> {_esc(_goal)}")
        if _task:
            detail.append(f"🗂️ <b>Task:</b> {_esc(_task)}")
        if _url:
            detail.append(f"🌐 <b>URL:</b> <code>{_esc(_url)}</code>")

        parts = [header, "", "\n".join(top)]
        if detail:
            parts += ["", "━━━━━━━━", *detail]
        text = "\n".join(parts)

        # 버튼 — HIGH는 '수정' 제외 (위험행동은 수정보다 명시적 승인/거부가 맞음)
        if risk_label.upper() == "HIGH":
            keyboard = InlineKeyboardMarkup([[
                InlineKeyboardButton("✅ 승인",  callback_data=f"approve:{rid}"),
                InlineKeyboardButton("❌ 거부",  callback_data=f"reject:{rid}"),
            ]])
        else:
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("✅ 승인",       callback_data=f"approve:{rid}"),
                    InlineKeyboardButton("✏️ 수정 후 실행", callback_data=f"edit:{rid}"),
                ],
                [
                    InlineKeyboardButton("❌ 거부",        callback_data=f"reject:{rid}"),
                ],
            ])

        try:
            bot = Bot(token=self.token)
            msg = await bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode="HTML",
                reply_markup=keyboard,
            )
            # 기존 _pending 저장소 재사용 (callback 시그니처 동일)
            _pending[rid] = ApprovalRequest(
                request_id=rid,
                customer_name=f"[{action}]",   # 재사용: 표시용
                customer_msg=_tgt_short or _task or "-",
                draft_reply=_cnt_short or "-",
                callback=callback,
                telegram_msg_id=msg.message_id,
                seq=_seq,
            )
            print(f"  [Telegram] 범용 승인 요청 발송 | risk={risk_label} | action={action} | rid={rid} | seq=#{_seq}")
            return rid

        except Exception as e:
            print(f"  [Telegram] 범용 승인 발송 실패: {e}")
            return None

    # ── [Phase 2] 실행 결과 finalize ──────────────────────────────────────────

    async def finalize_approval(
        self,
        rid: str,
        success: bool,
        detail: str = "",
        duration_ms: Optional[float] = None,
    ) -> bool:
        """
        [Phase 2] 승인된 요청(rid)의 실행 결과를 원본 메시지에 edit 으로 덮어씀.

        dispatcher(또는 caller) 가 실제 액션 실행을 마친 뒤 호출.
        결과:
          success=True  → ✅ 완료 메시지 + detail
          success=False → ⚠️ 실패 메시지 + detail
        호출 후 _pending[rid] 삭제. 다중 호출은 no-op (이미 정리된 rid 는 조용히 리턴).

        반환값: edit 성공 여부 (rid 없거나 Telegram API 실패면 False).
        """
        if not TELEGRAM_LOADED or not self.is_configured():
            return False
        # [Race guard] 동시 finalize 또는 finalize vs reject 경쟁 차단 —
        # 원자적 pop 으로 정확히 한 경로만 req 를 소유. 이후 Telegram edit
        # 도중에는 lock 을 풀어 다른 _pending 조작이 블록되지 않게 함.
        async with _get_pending_lock():
            req = _pending.pop(rid, None)
        if not req:
            # 이미 finalize 됐거나 reject 로 삭제됨 — 조용히 무시
            return False
        if not req.telegram_msg_id:
            # msg_id 없으면 edit 불가. (이미 pop 됐으므로 추가 정리 불필요)
            return False

        # duration 자동 계산 (승인 시각 저장돼 있으면)
        if duration_ms is None and req.approved_at > 0:
            duration_ms = (time.time() - req.approved_at) * 1000.0
        _dur = _fmt_duration(duration_ms)
        _detail_safe = _esc((detail or "").strip()[:600])

        if success:
            emoji, status_label = "✅", "완료"
        else:
            emoji, status_label = "⚠️", "실패"

        lines = [f"{emoji} <b>{status_label}</b> · <b>#{req.seq}</b>"]
        if _dur:
            lines[0] += f" · <i>{_dur}</i>"
        lines.append(f"<b>대상:</b> {_esc(req.customer_name)}")
        if _detail_safe:
            lines.append(f"<b>결과</b>\n<blockquote>{_detail_safe}</blockquote>")
        text = "\n".join(lines)

        try:
            bot = Bot(token=self.token)
            await bot.edit_message_text(
                chat_id=self.chat_id,
                message_id=req.telegram_msg_id,
                text=text,
                parse_mode="HTML",
            )
            return True
        except Exception as e:
            print(f"  [Telegram] finalize edit 실패 (rid={rid}): {e}")
            return False
        # [Race guard] _pending 은 이미 entry 에서 pop 됐으므로 finally 제거.

    # ── 단순 알림 발송 ────────────────────────────────────────────────────────

    async def send_notification(
        self,
        message: str,
        *,
        title: str = "알림",
    ):
        """
        버튼 없는 단순 텍스트 알림.

        [Phase 3-B 시각 구분]
        - 알림은 항상 '🔔 <b>{title}</b>' 헤더로 시작 — 승인 요청('📨' / '🔴🟡🟢')과 한눈에 구분.
        - title="" 로 넘기면 헤더 생략(기존 테스트/이벤트 호출 호환).
        - message 는 이미 HTML 일 수 있으므로 그대로 이어 붙임(호출부가 책임).
        """
        if not TELEGRAM_LOADED or not self.is_configured():
            return
        if title:
            # 메시지 상단이 이미 🔔 로 시작하면 중복 프리픽스 금지
            _t = message.lstrip()
            if not _t.startswith("🔔"):
                message = f"🔔 <b>{_esc(title)}</b>\n\n{message}"
        try:
            bot = Bot(token=self.token)
            await bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode="HTML",
            )
        except Exception as e:
            print(f"  [Telegram] 알림 발송 실패: {e}")

    async def send_captcha_alert(
        self,
        message: str,
        *,
        token: str,
        on_solved: Optional[Callable[[], None]] = None,
        title: str = "캡챠 해결 필요",
    ) -> None:
        """
        캡챠 해결 알림 + [✅ 풀었음 — 재개] 인라인 버튼.

        [Phase B 2026-04-24] 기존 send_notification(버튼 없음) 의 캡챠 전용 변형.
        사용자가 PC 가서 캡챠 풀고 GUI 까지 가서 ⚡ 자율 실행을 또 누르는 2단계를
        Telegram 한 번 클릭으로 압축. 콜백 등록은 token 키 기반 — _captcha_callbacks 에
        저장되었다가 _on_button_click 의 captcha_solved 분기에서 pop+호출.
        """
        if not TELEGRAM_LOADED or not self.is_configured():
            return
        if title:
            _t = message.lstrip()
            if not _t.startswith("🔔") and not _t.startswith("🧩"):
                message = f"🧩 <b>{_esc(title)}</b>\n\n{message}"
        try:
            kb = InlineKeyboardMarkup([[
                InlineKeyboardButton(
                    "✅ 풀었음 — 재개",
                    callback_data=f"captcha_solved:{token}",
                ),
            ]])
            if on_solved is not None:
                _captcha_callbacks[token] = on_solved
            bot = Bot(token=self.token)
            await bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode="HTML",
                reply_markup=kb,
            )
        except Exception as e:
            print(f"  [Telegram] 캡챠 알림 발송 실패: {e}")
            # 발송 실패 시 등록한 콜백 정리 — stale token 누적 방지
            _captcha_callbacks.pop(token, None)

    # ── 봇 폴링 시작 (백그라운드) ─────────────────────────────────────────────

    def start_polling(self):
        """별도 스레드에서 텔레그램 업데이트 폴링 시작."""
        if not TELEGRAM_LOADED:
            print("⚠️ [Telegram] 라이브러리 미설치")
            return
        if not self.is_configured():
            print("⚠️ [Telegram] 설정 미완료")
            return
        if self._running:
            return

        import threading
        t = threading.Thread(target=self._run_polling_thread, daemon=True)
        t.start()
        print("🤖 [Telegram] 폴링 시작 (백그라운드)")

    def _run_polling_thread(self):
        """폴링 전용 스레드에서 이벤트 루프 실행.

        네트워크 일시 장애로 ``Application.initialize()``의 ``get_me()``가
        TimedOut/ConnectError 던지면 과거에는 스레드가 그대로 죽어 세션
        끝까지 텔레그램이 비활성 상태로 남았음. 재시도 루프 + 백오프로
        부팅 시점·런타임 양쪽의 transient 네트워크 오류를 흡수한다.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # 외부에서 stop_polling 호출 가능하도록 의도(intent) 플래그를 켜둔다.
        self._running = True
        backoffs = [5, 10, 30, 60, 120, 300]
        attempt = 0
        try:
            while self._running:
                try:
                    loop.run_until_complete(self._run_app())
                    break  # stop_polling 으로 정상 종료
                except Exception as e:
                    # 재시도 위해 Application 인스턴스 폐기
                    self._app = None
                    if not self._running:
                        break
                    wait = backoffs[min(attempt, len(backoffs) - 1)]
                    attempt += 1
                    etype = type(e).__name__
                    print(
                        f"⚠️ [Telegram] 폴링 실행 실패 ({etype}): {e} "
                        f"— {wait}초 후 재시도 (#{attempt})"
                    )
                    try:
                        loop.run_until_complete(asyncio.sleep(wait))
                    except Exception:
                        time.sleep(wait)
        finally:
            self._running = False
            try:
                loop.close()
            except Exception:
                pass

    async def _run_app(self):
        # 네트워크 일시 지연에 강해지도록 HTTPX 타임아웃 상향
        # (PTB 기본 5초는 한국→텔레그램 경로에서 SSL handshake 가 자주 timeout).
        builder = Application.builder().token(self.token)
        for _setter, _val in (
            ("connect_timeout", 30.0),
            ("read_timeout", 30.0),
            ("write_timeout", 30.0),
            ("pool_timeout", 30.0),
            ("get_updates_connect_timeout", 30.0),
            ("get_updates_read_timeout", 60.0),
        ):
            _fn = getattr(builder, _setter, None)
            if callable(_fn):
                try:
                    builder = _fn(_val)
                except Exception:
                    pass
        self._app = builder.build()

        # 핸들러 등록
        self._app.add_handler(CommandHandler("start",  self._cmd_start))
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        # [Phase M2 2026-04-23] MONEY 카테고리 부분 kill-switch
        self._app.add_handler(CommandHandler("stop_money",    self._cmd_stop_money))
        self._app.add_handler(CommandHandler("resume_money",  self._cmd_resume_money))
        self._app.add_handler(CommandHandler("money_status",  self._cmd_money_status))
        # [Phase M3 2026-04-23] KPI 조회
        self._app.add_handler(CommandHandler("kpi_status",    self._cmd_kpi_status))
        self._app.add_handler(CommandHandler("kpi_drift",     self._cmd_kpi_drift))
        # [Phase M4 2026-04-23] 수익 기회 조회·수동 스캔
        self._app.add_handler(CommandHandler("opportunities", self._cmd_opportunities))
        self._app.add_handler(CommandHandler("scan_now",      self._cmd_scan_now))
        # [Phase M5 2026-04-23] 채널 레지스트리 조회·수동 pivot·리셋
        self._app.add_handler(CommandHandler("channels",      self._cmd_channels))
        self._app.add_handler(CommandHandler("pivot",         self._cmd_pivot))
        self._app.add_handler(CommandHandler("channel_reset", self._cmd_channel_reset))
        # [Autonomous 2026-04-27] 자율 실행 트리거·중단
        self._app.add_handler(CommandHandler("explore",       self._cmd_explore))
        self._app.add_handler(CommandHandler("stop_auto",     self._cmd_stop_auto))
        self._app.add_handler(CommandHandler("auto_status",   self._cmd_auto_status))
        self._app.add_handler(CallbackQueryHandler(self._on_button_click))
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text_message)
        )
        # [Wave7-A 2026-05-28] 음성 메시지 — STT 거쳐 chat_handler 로 forward.
        self._app.add_handler(
            MessageHandler(filters.VOICE, self._on_voice_message)
        )

        self._running = True
        async with self._app:
            await self._app.start()
            await self._app.updater.start_polling(drop_pending_updates=True)
            print("✅ [Telegram] 봇 폴링 활성화")
            # 종료 신호까지 대기
            while self._running:
                await asyncio.sleep(1)
            await self._app.updater.stop()
            await self._app.stop()

    def stop_polling(self):
        self._running = False
        print("⛔ [Telegram] 폴링 정지")

    # ── 커맨드 핸들러 ─────────────────────────────────────────────────────────

    async def _cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        await update.message.reply_text(
            f"👋 EIDOS 봇 연결됨!\n\n"
            f"내 chat_id: <code>{chat_id}</code>\n\n"
            f"크몽 고객 문의가 오면 여기서 승인 요청을 받게 돼.",
            parse_mode="HTML",
        )
        # chat_id 자동 저장
        if not self.chat_id:
            self.configure(self.token, str(chat_id))

    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        pending_count = len(_pending)
        await update.message.reply_text(
            f"📊 EIDOS 봇 상태\n\n"
            f"대기 중인 승인 요청: {pending_count}개\n"
            f"봇 상태: 🟢 정상 운영 중",
        )

    # ── Phase M2 2026-04-23 — MONEY 부분 kill-switch ─────────────────────────
    async def _cmd_stop_money(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """MONEY 카테고리 액션 전면 중지. 승인 루프도 들어오면 즉시 REJECT."""
        try:
            from eidos_approval_gateway import ApprovalGateway, ActionCategory
            gw = ApprovalGateway.get()
            reason = " ".join(ctx.args) if getattr(ctx, "args", None) else "사용자 /stop_money"
            gw.trip_category_kill(ActionCategory.MONEY, reason)
            await update.message.reply_text(
                f"⛔ MONEY 액션 정지됨\n사유: {reason}\n복구: /resume_money",
            )
        except Exception as e:
            await update.message.reply_text(f"⚠️ /stop_money 실패: {e}")

    async def _cmd_resume_money(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            from eidos_approval_gateway import ApprovalGateway, ActionCategory
            gw = ApprovalGateway.get()
            gw.clear_category_kill(ActionCategory.MONEY)
            await update.message.reply_text("✅ MONEY 액션 정지 해제됨.")
        except Exception as e:
            await update.message.reply_text(f"⚠️ /resume_money 실패: {e}")

    # ── Phase M3 2026-04-23 — KPI 조회 ──────────────────────────────────────
    async def _cmd_kpi_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            from eidos_kpi_store import KPIStore
            summary = KPIStore.get().recent_summary(days=1, max_lines=15)
            if not summary:
                await update.message.reply_text(
                    "📊 오늘 수집된 KPI 없음.\n"
                    "샘플 주입 경로: eidos_files/kpi_inputs/*.json 또는 eidos_files/sales/*.csv"
                )
                return
            await update.message.reply_text(f"📊 오늘 KPI\n\n{summary}")
        except Exception as e:
            await update.message.reply_text(f"⚠️ /kpi_status 실패: {e}")

    async def _cmd_kpi_drift(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            from eidos_kpi_store import KPIStore
            drifts = KPIStore.get().detect_drift(threshold=0.20)
            if not drifts:
                await update.message.reply_text("✅ KPI drift 없음 (임계 20%).")
                return
            lines = []
            for d in drifts[:10]:
                pct = int(abs(float(d.get('delta_ratio', 0.0))) * 100)
                arrow = "↑" if d.get('direction') == 'up' else "↓"
                lines.append(
                    f"- {d.get('actor_id')} {d.get('metric')} {arrow}{pct}% "
                    f"(오늘 {d.get('today')} / 어제 {d.get('yesterday')})"
                )
            await update.message.reply_text("📉 KPI Drift\n\n" + "\n".join(lines))
        except Exception as e:
            await update.message.reply_text(f"⚠️ /kpi_drift 실패: {e}")

    # ── Phase M4 2026-04-23 — 수익 기회 조회·수동 스캔 ─────────────────────
    async def _cmd_opportunities(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            from eidos_opportunity_scanner import get_scanner
            hypos = get_scanner().list_recent(limit=10)
            if not hypos:
                await update.message.reply_text(
                    "💡 저장된 수익 기회 없음.\n"
                    "/scan_now 로 즉시 1회 스캔할 수 있어."
                )
                return
            lines = []
            for h in hypos:
                mark = {
                    "launched": "🚀", "approved": "✅",
                    "proposed": "📩", "rejected": "❌",
                    "skipped":  "⏭", "pending":  "🕒",
                }.get(h.status, "•")
                lines.append(
                    f"{mark} [{h.status}] ROI={h.roi_score:.2f} · {h.title[:60]}"
                )
            await update.message.reply_text("💡 최근 수익 기회\n\n" + "\n".join(lines))
        except Exception as e:
            await update.message.reply_text(f"⚠️ /opportunities 실패: {e}")

    async def _cmd_scan_now(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            from eidos_opportunity_scanner import get_scanner
            await update.message.reply_text("🔭 기회 스캔 시작 — 잠시 후 제안 카드가 도착할 거야.")
            scanner = get_scanner()
            hypos = await scanner.run_scan_now()
            # [2026-04-24] 실패 시 사유 에코 — 이전엔 "새 기회 없음" 으로 모든 원인이
            # 묻혀 사용자가 LLM 서버 장애/파싱 실패/dedupe 구분을 할 수 없었다.
            status = getattr(scanner, "_last_scan_status", "ok")
            detail = getattr(scanner, "_last_scan_detail", "") or ""
            if not hypos:
                _map = {
                    "no_goal":     "💡 CoreGoal 이 없어. 제어실에서 장기 목표 먼저 설정해줘.",
                    "llm_error":   f"⚠️ LLM 서버 응답 에러로 스캔 실패.\n└ {detail}",
                    "raw_empty":   f"⚠️ LLM 이 빈 응답을 반환했어.\n└ {detail}",
                    "parse_fail":  f"⚠️ LLM 응답 파싱 실패 — JSON 구조를 못 찾았어.\n└ {detail}",
                    "no_title":    f"⚠️ LLM 응답에 title 필드가 없어 — 프롬프트 점검 필요.\n└ {detail}",
                    "all_dedupe":  f"🔁 새 기회 없음 — 전부 최근 48h 기존 제안과 중복.\n└ {detail}",
                    "parse_empty": f"💡 새 기회 없음 (LLM 응답 비었거나 파싱 실패).\n└ {detail}",
                }
                msg = _map.get(status, f"💡 새 기회 없음 ({status}). {detail}".rstrip())
                await update.message.reply_text(msg)
            else:
                await update.message.reply_text(
                    f"💡 {len(hypos)}개 생성, 상위 3개 제안 카드 발송."
                )
        except Exception as e:
            await update.message.reply_text(f"⚠️ /scan_now 실패: {e}")

    # ── Phase M5 2026-04-23 — 채널 레지스트리 ────────────────────────────────
    async def _cmd_channels(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            from eidos_channel_registry import get_registry
            reg = get_registry()
            summary = reg.summary()
            if not summary:
                await update.message.reply_text("📡 등록된 채널 없음.")
                return
            await update.message.reply_text(f"📡 채널 상태\n\n{summary}")
        except Exception as e:
            await update.message.reply_text(f"⚠️ /channels 실패: {e}")

    async def _cmd_pivot(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """사용법: /pivot FROM TO   예: /pivot kmong smartstore"""
        try:
            from eidos_channel_registry import get_registry, execute_pivot
            args = list(getattr(ctx, "args", []) or [])
            if len(args) < 2:
                await update.message.reply_text(
                    "사용법: /pivot FROM TO\n예: /pivot kmong smartstore"
                )
                return
            frm, to = args[0], args[1]
            payload = get_registry().manual_pivot(frm, to)
            if not payload:
                await update.message.reply_text(f"⚠️ '{frm}' 채널 없음.")
                return
            cid = await execute_pivot(payload)
            if cid:
                await update.message.reply_text(
                    f"🔄 수동 pivot 런칭: {frm} → {to} (chain {cid[:8]})"
                )
            else:
                await update.message.reply_text(
                    f"⚠️ pivot 런칭 실패 — {frm} → {to}"
                )
        except Exception as e:
            await update.message.reply_text(f"⚠️ /pivot 실패: {e}")

    async def _cmd_channel_reset(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """사용법: /channel_reset NAME"""
        try:
            from eidos_channel_registry import get_registry
            args = list(getattr(ctx, "args", []) or [])
            if not args:
                await update.message.reply_text("사용법: /channel_reset NAME")
                return
            name = args[0]
            ok = get_registry().reset_to_active(name)
            if ok:
                await update.message.reply_text(f"✅ {name} → active 리셋 완료.")
            else:
                await update.message.reply_text(f"⚠️ 알 수 없는 채널: {name}")
        except Exception as e:
            await update.message.reply_text(f"⚠️ /channel_reset 실패: {e}")

    # ── Autonomous 2026-04-27 — 자율 실행 트리거/중단/상태 ──────────────────
    async def _cmd_explore(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """제어실 top_goal 기반으로 자율 실행 사전계획 + 승인 카드 발송.

        사용자 args 가 있으면 그걸 seed_goal 로 사용 (top_goal 우회).
        예: /explore 강남역 파스타 맛집 TOP5 + 예약
        """
        try:
            from eidos_autonomous_runner import (
                start_planning, build_approval_card_text,
            )
        except Exception as e:
            await update.message.reply_text(f"⚠️ /explore 모듈 로드 실패: {e}")
            return
        seed_override = " ".join(ctx.args).strip() if getattr(ctx, "args", None) else ""
        await update.message.reply_text(
            "🤖 자율 실행 사전계획 생성 중... (LLM 호출 1회, 약 5~15초)"
        )
        try:
            run = await start_planning(seed_goal=seed_override or None)
        except Exception as e:
            await update.message.reply_text(f"⚠️ 사전계획 실패: {e}")
            return
        if run is None:
            await update.message.reply_text(
                "⚠️ 최상위 목표가 비어있어. 제어실에서 top_goal 을 먼저 설정하거나\n"
                "<code>/explore [목표 텍스트]</code> 형식으로 직접 입력해줘.",
                parse_mode="HTML",
            )
            return
        # 승인 카드 발송 — 4 버튼 (이번 한 번 / 1시간 / 24시간 / 거부)
        text = build_approval_card_text(run)
        kb = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ 이번 한 번",   callback_data=f"auto_approve:once:{run.run_id}"),
                InlineKeyboardButton("⏱️ 1시간 자율",  callback_data=f"auto_approve:1h:{run.run_id}"),
            ],
            [
                InlineKeyboardButton("🌙 24시간 자율", callback_data=f"auto_approve:24h:{run.run_id}"),
                InlineKeyboardButton("❌ 거부",        callback_data=f"auto_reject:{run.run_id}"),
            ],
        ])
        try:
            await update.message.reply_text(
                text, parse_mode="HTML", reply_markup=kb,
            )
        except Exception as e:
            await update.message.reply_text(f"⚠️ 카드 발송 실패: {e}")

    async def _cmd_stop_auto(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """활성 자율 실행 즉시 중단 + chain abort."""
        try:
            from eidos_autonomous_runner import AutonomousRunManager
            mgr = AutonomousRunManager.get()
            run = mgr.abort_active(reason="user_stop_telegram")
            if run is None:
                await update.message.reply_text("ℹ️ 활성 자율 실행 없음.")
                return
            # chain abort best-effort
            try:
                from eidos_mission_chain import get_registry as _get_chain_reg, ChainStatus
                if run.chain_id:
                    reg = _get_chain_reg()
                    chain = reg.get(run.chain_id)
                    if chain and chain.status == ChainStatus.RUNNING:
                        reg.abort(run.chain_id, reason="autonomous_user_stop")
            except Exception as _e_c:
                print(f"  [Autonomous] chain abort best-effort 실패 (무시): {_e_c}")
            await update.message.reply_text(
                f"🛑 <b>자율 실행 중단</b>\n"
                f"run_id: <code>{_esc(run.run_id)}</code>\n"
                f"실행된 액션: {run.actions_executed}건",
                parse_mode="HTML",
            )
        except Exception as e:
            await update.message.reply_text(f"⚠️ /stop_auto 실패: {e}")

    async def _cmd_auto_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """현재 활성 자율 실행 상태."""
        try:
            from eidos_autonomous_runner import AutonomousRunManager
            mgr = AutonomousRunManager.get()
            run = mgr.get_active_run()
            if run is None:
                await update.message.reply_text("ℹ️ 활성 자율 실행 없음. /explore 로 시작.")
                return
            remaining = ""
            if run.expires_at:
                rem_sec = max(0, int(run.expires_at - time.time()))
                remaining = f"\n남은 시간: {rem_sec // 60}분 {rem_sec % 60}초"
            await update.message.reply_text(
                f"🤖 <b>자율 실행 중</b>\n"
                f"목표: {_esc(run.plan.objective)}\n"
                f"스코프: <code>{_esc(run.scope_type)}</code>\n"
                f"실행된 액션: {run.actions_executed}건\n"
                f"hard-block 위반: {run.hard_block_violations}건"
                f"{remaining}",
                parse_mode="HTML",
            )
        except Exception as e:
            await update.message.reply_text(f"⚠️ /auto_status 실패: {e}")

    async def _cmd_money_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            from eidos_approval_gateway import ApprovalGateway
            gw = ApprovalGateway.get()
            st = gw.money_status()
            killed_line = (
                f"⛔ 정지 중 (사유: {st.get('kill_reason', '')})"
                if st.get("killed") else "🟢 정상 운영 중"
            )
            month_pct = int(round(float(st.get("month_ratio", 0.0)) * 100))
            day_pct   = int(round(float(st.get("day_ratio",   0.0)) * 100))
            msg = (
                f"💰 MONEY 예산 상태 — {st.get('month', '')}\n"
                f"{killed_line}\n\n"
                f"월 소진: {int(st.get('consumed_month', 0)):,} / {int(st.get('monthly_cap', 0)):,} KRW ({month_pct}%)\n"
                f"일 소진: {int(st.get('consumed_day', 0)):,} / {int(st.get('daily_cap', 0)):,} KRW ({day_pct}%)"
            )
            await update.message.reply_text(msg)
        except Exception as e:
            await update.message.reply_text(f"⚠️ /money_status 실패: {e}")

    # ── 버튼 클릭 핸들러 ──────────────────────────────────────────────────────

    async def _on_button_click(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()

        data = query.data or ""
        if ":" not in data:
            return

        action, rid = data.split(":", 1)

        # [Autonomous 2026-04-27] 자율 실행 승인 — _pending 과 분리된 라이프사이클.
        # callback_data 형식:
        #   auto_approve:once:<rid>   — SINGLE_CHAIN
        #   auto_approve:1h:<rid>     — TIME_WINDOW 1시간
        #   auto_approve:24h:<rid>    — TIME_WINDOW 24시간
        #   auto_reject:<rid>         — 거부
        if action == "auto_approve":
            # rid 는 "scope:run_id" 로 들어옴
            try:
                _scope, _run_id = rid.split(":", 1)
            except ValueError:
                await query.edit_message_text("⚠️ 잘못된 자율 승인 데이터.")
                return
            try:
                from eidos_autonomous_runner import (
                    AutonomousRunManager, ScopeType,
                )
            except Exception as e:
                await query.edit_message_text(f"⚠️ 자율 실행 모듈 로드 실패: {e}")
                return
            mgr = AutonomousRunManager.get()
            if _scope == "once":
                _stype = ScopeType.SINGLE_CHAIN
                _win = 0
                _label = "단일 chain"
            elif _scope == "1h":
                _stype = ScopeType.TIME_WINDOW
                _win = 3600
                _label = "1시간 자율"
            elif _scope == "24h":
                _stype = ScopeType.TIME_WINDOW
                _win = 24 * 3600
                _label = "24시간 자율"
            else:
                await query.edit_message_text(f"⚠️ 알 수 없는 스코프: {_scope}")
                return
            # 응답 즉시 edit (사용자에게 빠른 피드백)
            await query.edit_message_text(
                f"✅ <b>자율 실행 승인 — {_esc(_label)}</b>\n"
                f"run_id: <code>{_esc(_run_id)}</code>\n"
                f"<i>chain 시작 중...</i>",
                parse_mode="HTML",
            )
            try:
                run = await mgr.approve(_run_id, scope_type=_stype, time_window_sec=_win)
                if run is None:
                    await update.callback_query.message.reply_text(
                        "⚠️ 이미 처리됐거나 만료된 자율 승인 요청."
                    )
                    return
                # chain_starter 콜백이 None 인 경우 (통합 미완료) — 안내
                if not getattr(mgr, "_chain_starter", None):
                    await update.callback_query.message.reply_text(
                        "ℹ️ 승인 기록됨 — 단, chain_starter 콜백 미등록 (통합 대기). "
                        "core 통합 완료 후부터 자동 실행됨."
                    )
            except Exception as e:
                await update.callback_query.message.reply_text(
                    f"⚠️ 승인 처리 중 오류: {e}"
                )
            return

        # ── [B3 협력 게이트 다층화 2026-04-29] Tier 2 짧은 질문 응답 처리 ──
        # callback_data: quick_ans:<rid>:<idx>
        # _PENDING_QUICK_QUESTIONS[rid] Future 가 있으면 set_result(option_text).
        # rid 없으면 만료 안내. timeout 시 Future 가 cancel 되어 아무 호출자도 응답 못 받음.
        if action == "quick_ans":
            try:
                _q_rid, _q_idx = rid.split(":", 1)
            except ValueError:
                await query.edit_message_text("⚠️ 잘못된 quick_ans 데이터.")
                return
            _entry = _PENDING_QUICK_QUESTIONS.get(_q_rid)
            if _entry is None:
                await query.edit_message_text(
                    f"⚠️ <b>응답 만료</b> — 봇 재시작/타임아웃으로 질문이 사라짐.",
                    parse_mode="HTML",
                )
                return
            _opts = _entry.get("options") or []
            try:
                _idx = int(_q_idx)
            except ValueError:
                await query.edit_message_text(f"⚠️ 잘못된 옵션 번호: {_q_idx}")
                return
            if _idx < 1 or _idx > len(_opts):
                await query.edit_message_text(
                    f"⚠️ 옵션 범위 밖: {_idx} (옵션 {len(_opts)}개)"
                )
                return
            _selected = _opts[_idx - 1]
            _fut = _entry.get("future")
            try:
                if _fut and not _fut.done():
                    _fut.set_result(_selected)
            except Exception as _e_fut:
                print(f"  ⚠️ [quick_ans] Future set_result 실패: {_e_fut}")
            # 캐시 정리
            _PENDING_QUICK_QUESTIONS.pop(_q_rid, None)
            await query.edit_message_text(
                f"✅ <b>응답 받음</b>: {_esc(_selected[:80])}",
                parse_mode="HTML",
            )
            return

        # [Phase 2.7.1 2026-05-03] Worldmap 분기점 선택 — fire-and-forget callback.
        # callback_data: worldmap_branch:<rid>:<node_id_or___STOP__>
        if action == "worldmap_branch":
            try:
                _wb_rid, _wb_node = rid.split(":", 1)
            except ValueError:
                await query.edit_message_text("⚠️ 잘못된 worldmap_branch 데이터.")
                return
            _entry = _PENDING_WORLDMAP_BRANCHES.get(_wb_rid)
            if _entry is None:
                await query.edit_message_text(
                    "⚠️ <b>응답 만료</b> — GUI 또는 timeout 으로 처리됨.",
                    parse_mode="HTML",
                )
                return
            _cb = _entry.get("callback")
            _PENDING_WORLDMAP_BRANCHES.pop(_wb_rid, None)
            try:
                if callable(_cb):
                    _cb(_wb_node)
            except Exception as _e_wb:
                print(f"  ⚠️ [worldmap_branch] callback 실패: {_e_wb}")
            _label = "⏹ 정지" if _wb_node == "__STOP__" else f"➡ {_esc(_wb_node[:40])}"
            await query.edit_message_text(
                f"✅ <b>분기 선택</b>: {_label}",
                parse_mode="HTML",
            )
            return

        # [Phase 2.7.2 2026-05-03] Worldmap 검증 후 응답 — '완료' 또는 '되돌리기'.
        # callback_data: worldmap_verify:<rid>:<choice>  (choice = "ok" or "rollback")
        if action == "worldmap_verify":
            try:
                _wv_rid, _wv_choice = rid.split(":", 1)
            except ValueError:
                await query.edit_message_text("⚠️ 잘못된 worldmap_verify 데이터.")
                return
            _entry = _PENDING_WORLDMAP_VERIFY.get(_wv_rid)
            if _entry is None:
                await query.edit_message_text("⚠️ 응답 만료.")
                return
            _cb = _entry.get("callback")
            _PENDING_WORLDMAP_VERIFY.pop(_wv_rid, None)
            try:
                if callable(_cb):
                    _cb(_wv_choice)
            except Exception as _e_wv:
                print(f"  ⚠️ [worldmap_verify] callback 실패: {_e_wv}")
            _msg = "↶ 되돌림 진행" if _wv_choice == "rollback" else "✅ 완료 확정"
            await query.edit_message_text(
                f"<b>{_msg}</b>",
                parse_mode="HTML",
            )
            return

        # ── [Daily Report 2026-04-29] sub-goal 후보 클릭 → 자율 승인 카드 재발송 ──
        # callback_data 형식: daily_goal:<idx>:<date> (idx=1~3 또는 'skip')
        # 후보 텍스트는 _DAILY_CANDIDATES_CACHE[date] 에서 lookup.
        # 후보 클릭 → start_planning(seed_goal=후보) → 기존 4버튼 자율 승인 카드 재발송
        # (와우 핵심 연결고리: 보고 → 한 클릭 → chain 시작).
        if action == "daily_goal":
            try:
                _idx_str, _date_str = rid.split(":", 1)
            except ValueError:
                await query.edit_message_text("⚠️ 잘못된 daily_goal 데이터.")
                return
            if _idx_str == "skip":
                await query.edit_message_text(
                    f"⏭️ <b>일일 보고</b> ({_esc(_date_str)}) — 무시됨\n"
                    f"<i>내일 저녁에 다시 보고</i>",
                    parse_mode="HTML",
                )
                # 후보 캐시 제거 (메모리 정리)
                _DAILY_CANDIDATES_CACHE.pop(_date_str, None)
                return
            try:
                _idx = int(_idx_str)
            except ValueError:
                await query.edit_message_text(f"⚠️ 잘못된 후보 번호: {_idx_str}")
                return
            cands = _DAILY_CANDIDATES_CACHE.get(_date_str) or []
            if not cands or _idx < 1 or _idx > len(cands):
                await query.edit_message_text(
                    f"⚠️ <b>후보 만료</b> — 봇 재시작/캐시 삭제로 후보 텍스트가 사라짐.\n"
                    f"<i>다음 일일 보고에서 다시 선택하거나 /explore 로 직접 시작.</i>",
                    parse_mode="HTML",
                )
                return
            seed = cands[_idx - 1]
            await query.edit_message_text(
                f"💡 <b>sub-goal 선택</b>: {_esc(seed[:80])}\n"
                f"<i>사전 계획 생성 중...</i>",
                parse_mode="HTML",
            )
            try:
                from eidos_autonomous_runner import (
                    start_planning, build_approval_card_text,
                )
            except Exception as e:
                await update.callback_query.message.reply_text(
                    f"⚠️ 자율 모듈 로드 실패: {e}"
                )
                return
            try:
                run = await start_planning(seed_goal=seed)
            except Exception as e:
                await update.callback_query.message.reply_text(
                    f"⚠️ 사전계획 실패: {e}"
                )
                return
            if run is None:
                await update.callback_query.message.reply_text(
                    "⚠️ 사전계획 결과 None — 후보 텍스트 확인 필요."
                )
                return
            # 4버튼 자율 승인 카드 — _cmd_explore 와 동일 패턴
            text = build_approval_card_text(run)
            kb = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("✅ 이번 한 번",   callback_data=f"auto_approve:once:{run.run_id}"),
                    InlineKeyboardButton("⏱️ 1시간 자율",  callback_data=f"auto_approve:1h:{run.run_id}"),
                ],
                [
                    InlineKeyboardButton("🌙 24시간 자율", callback_data=f"auto_approve:24h:{run.run_id}"),
                    InlineKeyboardButton("❌ 거부",        callback_data=f"auto_reject:{run.run_id}"),
                ],
            ])
            try:
                await update.callback_query.message.reply_text(
                    text, parse_mode="HTML", reply_markup=kb,
                )
            except Exception as e:
                await update.callback_query.message.reply_text(
                    f"⚠️ 카드 발송 실패: {e}"
                )
            # 캐시 정리
            _DAILY_CANDIDATES_CACHE.pop(_date_str, None)
            return

        if action == "auto_reject":
            try:
                from eidos_autonomous_runner import AutonomousRunManager
            except Exception as e:
                await query.edit_message_text(f"⚠️ 자율 실행 모듈 로드 실패: {e}")
                return
            mgr = AutonomousRunManager.get()
            run = mgr.reject(rid)
            if run is None:
                await query.edit_message_text("⚠️ 이미 처리된 자율 승인 요청.")
                return
            await query.edit_message_text(
                f"❌ <b>자율 실행 거부됨</b>\n"
                f"run_id: <code>{_esc(rid)}</code>",
                parse_mode="HTML",
            )
            return

        # [Phase B 2026-04-24] CAPTCHA 인라인 버튼 — _pending 과 분리된 콜백 레지스트리.
        # _captcha_callbacks 의 token 은 ApprovalRequest 와 무관 → _pending lookup 전에 분기.
        if action == "captcha_solved":
            cb = _captcha_callbacks.pop(rid, None)
            if cb is None:
                await query.edit_message_text(
                    "⚠️ 만료된 캡챠 토큰이거나 이미 처리됐어요."
                )
                return
            await query.edit_message_text(
                "✅ <b>캡챠 해결 신호 수신 — 자동 재검증 중...</b>\n"
                "<i>브라우저 페이지에서 캡챠가 사라졌는지 확인 후 자동 재개합니다.</i>",
                parse_mode="HTML",
            )
            try:
                cb()
            except Exception as _ec:
                print(f"  [Telegram] captcha_solved 콜백 실행 실패: {_ec}")
            return

        # [Race guard] get 은 원자적이지만, 다른 경로와 동시 처리 방지를 위해
        # reject 등 최종 분기에서 pop 시 lock 을 씀 (아래 action=="reject").
        req = _pending.get(rid)
        if not req:
            await query.edit_message_text("⚠️ 이미 처리됐거나 만료된 요청이에요.")
            return

        if action == "approve":
            # ── 승인됨 → 실행 대기 (finalize_approval 에서 최종 결과로 edit) ──
            # [Phase 2] 기존: 여기서 "✅ 전송 완료" 고정 메시지 → 오해 소지(아직 실행 전).
            # 변경: "⏳ 실행 중..." 임시 메시지로 edit 하고 _pending 유지.
            # dispatcher 가 실행 완료 후 finalize_approval(rid) 호출해 결과 메시지로 덮어씀.
            req.approved_at = time.time()
            req.awaiting_finalize = True
            await query.edit_message_text(
                f"⏳ <b>승인됨 · 실행 중</b> · <b>#{req.seq}</b>\n"
                f"<b>대상:</b> {_esc(req.customer_name)}\n"
                f"<b>내용:</b>\n<blockquote>{_esc(req.draft_reply[:500])}</blockquote>",
                parse_mode="HTML",
            )
            _fire_callback(req, "approved", req.draft_reply)
            # del _pending[rid] 제거 — finalize_approval 에서 정리

        elif action == "reject":
            # ── 거부 → 즉시 종료 ─────────────────────────────────────────────
            await query.edit_message_text(
                f"❌ <b>거부됨</b> · <b>#{req.seq}</b>\n"
                f"<b>대상:</b> {_esc(req.customer_name)} — 취소.",
                parse_mode="HTML",
            )
            _fire_callback(req, "rejected", "")
            # [Race guard] del 대신 lock+pop — finalize 가 먼저 지웠을 경우 KeyError 방지.
            async with _get_pending_lock():
                _pending.pop(rid, None)

        elif action == "edit":
            # ── 수정 요청 → 텍스트 입력 대기 ─────────────────────────────────
            ctx.user_data["editing_rid"] = rid
            await query.edit_message_text(
                f"✏️ <b>수정 모드</b> · <b>#{req.seq}</b>\n"
                f"수정할 메시지를 입력해줘.\n"
                f"\n"
                f"<b>현재 초안</b>\n"
                f"<blockquote>{_esc(req.draft_reply)}</blockquote>",
                parse_mode="HTML",
            )

        elif action == "edit_ok":
            # [Phase 3-A] preview 확인 → 수정본 확정 → 실행 대기
            _edited = req.edited_text or req.draft_reply
            req.approved_at = time.time()
            req.awaiting_finalize = True
            await query.edit_message_text(
                f"⏳ <b>수정본 승인됨 · 실행 중</b> · <b>#{req.seq}</b>\n"
                f"<b>대상:</b> {_esc(req.customer_name)}\n"
                f"<b>내용:</b>\n<blockquote>{_esc(_edited[:500])}</blockquote>",
                parse_mode="HTML",
            )
            _fire_callback(req, "edited", _edited)
            # _pending 유지 — finalize_approval 에서 정리

        elif action == "edit_again":
            # [Phase 3-A] preview 에서 다시 수정 — editing 상태로 복귀
            ctx.user_data["editing_rid"] = rid
            _prev_edit = req.edited_text or ""
            await query.edit_message_text(
                f"✏️ <b>다시 수정</b> · <b>#{req.seq}</b>\n"
                f"수정할 메시지를 다시 입력해줘.\n"
                f"\n"
                f"<b>방금 수정본</b>\n"
                f"<blockquote>{_esc(_prev_edit[:500])}</blockquote>",
                parse_mode="HTML",
            )

        elif action == "edit_cancel":
            # [Phase 3-A] preview 취소 → 원래 3버튼 승인 화면으로 복구
            req.edited_text = ""
            ctx.user_data.pop("editing_rid", None)
            _restore_kb = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("✅ 그대로 전송",  callback_data=f"approve:{rid}"),
                    InlineKeyboardButton("✏️ 수정 후 전송", callback_data=f"edit:{rid}"),
                ],
                [
                    InlineKeyboardButton("❌ 취소",         callback_data=f"reject:{rid}"),
                ],
            ])
            await query.edit_message_text(
                f"↩️ <b>수정 취소 · 원본으로 복귀</b> · <b>#{req.seq}</b>\n"
                f"<b>대상:</b> {_esc(req.customer_name)}\n"
                f"<b>초안</b>\n<blockquote>{_esc(req.draft_reply[:500])}</blockquote>\n"
                f"이제 어떻게 할까?",
                parse_mode="HTML",
                reply_markup=_restore_kb,
            )

    # ── 텍스트 메시지 핸들러 (수정 입력) ─────────────────────────────────────

    async def _on_text_message(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        editing_rid = ctx.user_data.get("editing_rid")

        # ── 승인 수정 모드 → [Phase 3-A] 바로 전송 금지, preview 단계로 전환 ──
        if editing_rid:
            # [Race guard] finalize/reject 와 동시 도착 방지 — req 확인과
            # edited_text 주입을 한 lock 안에서. Telegram edit 은 밖에서.
            async with _get_pending_lock():
                req = _pending.get(editing_rid)
                if not req:
                    _req_missing = True
                else:
                    _req_missing = False
                    edited_text = update.message.text.strip()
                    if edited_text:
                        req.edited_text = edited_text
            if _req_missing:
                await update.message.reply_text("⚠️ 요청을 찾을 수 없어요.")
                ctx.user_data.pop("editing_rid", None)
                return
            edited_text = update.message.text.strip()
            if not edited_text:
                await update.message.reply_text("⚠️ 빈 메시지는 보낼 수 없어요. 다시 입력해줘.")
                return
            # [Phase 3-A] editing_rid 제거 → preview 대기. 콜백 아직 발화 금지.
            ctx.user_data.pop("editing_rid", None)
            # req.edited_text 는 이미 lock 안에서 설정됨.
            # 원본 승인 메시지 자리에 preview 화면을 덮어씀 (msg_id 재활용)
            _preview_kb = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("✅ 이대로 전송", callback_data=f"edit_ok:{editing_rid}"),
                    InlineKeyboardButton("✏️ 다시 수정",  callback_data=f"edit_again:{editing_rid}"),
                ],
                [
                    InlineKeyboardButton("↩️ 원래 초안으로", callback_data=f"edit_cancel:{editing_rid}"),
                ],
            ])
            try:
                if req.telegram_msg_id:
                    bot = Bot(token=self.token)
                    await bot.edit_message_text(
                        chat_id=self.chat_id,
                        message_id=req.telegram_msg_id,
                        text=(
                            f"👀 <b>수정본 확인</b> · <b>#{req.seq}</b>\n"
                            f"<b>원본</b>\n<blockquote>{_esc(req.draft_reply[:300])}</blockquote>\n"
                            f"<b>수정본</b>\n<blockquote>{_esc(edited_text[:500])}</blockquote>\n"
                            f"이대로 전송할까?"
                        ),
                        parse_mode="HTML",
                        reply_markup=_preview_kb,
                    )
            except Exception as _e:
                print(f"  [Telegram] preview edit 실패 (무시): {_e}")
            await update.message.reply_text(
                f"👀 수정본 미리보기 준비 완료 · <b>#{req.seq}</b>",
                parse_mode="HTML",
            )
            # _pending 유지 — edit_ok/edit_again/edit_cancel 에서 처리
            return

        # ── EIDOS 챗 포워딩 모드 ────────────────────────────────────────
        # 보안: 등록된 chat_id(=본인)만 수용. 다른 누가 토큰 알아도 대화 차단.
        incoming_chat_id = str(update.effective_chat.id)
        if self.chat_id and incoming_chat_id != str(self.chat_id):
            await update.message.reply_text("⛔ 권한 없음 (등록되지 않은 사용자)")
            return

        handler = self._chat_handler
        if not handler:
            await update.message.reply_text(
                "EIDOS 봇이에요. 고객 문의 승인 요청이 오면 알려드릴게요. 😊"
            )
            return

        user_text = (update.message.text or "").strip()
        if not user_text:
            return

        # 사용자에게 "생각 중" 표시 (텔레그램 typing 상태)
        try:
            await ctx.bot.send_chat_action(chat_id=incoming_chat_id, action="typing")
        except Exception:
            pass

        # EIDOS 챗 호출
        try:
            response = await handler(user_text)
        except Exception as e:
            print(f"  [Telegram] chat_handler 예외: {e}")
            await update.message.reply_text(f"⚠️ EIDOS 처리 중 오류: {str(e)[:200]}")
            return

        if not response:
            response = "(응답 없음)"
        # Telegram 메시지 길이 제한 4096자 — 넉넉히 4000에서 자름
        if len(response) > 4000:
            response = response[:4000] + "\n\n… (응답 길어서 잘림)"
        try:
            await update.message.reply_text(response)
        except Exception as e:
            print(f"  [Telegram] 응답 전송 실패: {e}")

    # ── [Wave7-A 2026-05-28] 음성 메시지 → STT → chat ───────────────────────
    async def _on_voice_message(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE,
    ):
        """텔레그램 음성 (.ogg) → STT → chat_handler 로 텍스트처럼 forward.

        보안: 등록된 chat_id 만 수용. STT 백엔드는 eidos_telegram_voice 모듈.
        실패 시 사용자에게 한 줄로 알림. graceful — 어떤 단계 실패해도
        텔레그램 봇 자체는 영향 X.
        """
        # 권한 검증 (text 와 동일)
        incoming_chat_id = str(update.effective_chat.id)
        if self.chat_id and incoming_chat_id != str(self.chat_id):
            await update.message.reply_text("⛔ 권한 없음 (등록되지 않은 사용자)")
            return

        # chat_handler 미등록 시 안내
        handler = self._chat_handler
        if not handler:
            await update.message.reply_text(
                "🎙️ 음성 받았지만 EIDOS 챗 핸들러가 아직 연결 안 됐어요."
            )
            return

        voice = update.message.voice
        if voice is None:
            return

        # 너무 긴 음성은 거부 (60초 제한·STT 비용·timeout)
        duration = int(getattr(voice, "duration", 0) or 0)
        if duration > 60:
            await update.message.reply_text(
                f"⚠️ 음성이 너무 길어요 ({duration}초). 60초 이하로 보내주세요."
            )
            return

        # "받았어요" 즉시 피드백
        try:
            await ctx.bot.send_chat_action(
                chat_id=incoming_chat_id, action="typing",
            )
        except Exception:
            pass

        # 1) 파일 다운로드
        try:
            file = await ctx.bot.get_file(voice.file_id)
            audio_bytes = bytes(await file.download_as_bytearray())
        except Exception as e:
            print(f"  [Telegram-voice] 다운로드 실패: {e}")
            await update.message.reply_text(
                f"⚠️ 음성 다운로드 실패: {str(e)[:120]}"
            )
            return

        if not audio_bytes:
            await update.message.reply_text("⚠️ 빈 음성 파일.")
            return

        # 2) STT
        text = ""
        try:
            from eidos_telegram_voice import transcribe_voice_async
            text = await transcribe_voice_async(
                audio_bytes, mime_type="audio/ogg", timeout_sec=40.0,
            )
        except Exception as e:
            print(f"  [Telegram-voice] STT 모듈 실패: {e}")

        if not text:
            await update.message.reply_text(
                "⚠️ 음성 인식 실패. 다시 시도하거나 텍스트로 보내주세요."
            )
            return

        # 3) 사용자에게 transcription 결과 확인용 표시 (짧게)
        try:
            preview = text if len(text) <= 200 else text[:200] + "…"
            await update.message.reply_text(f"🎙️→📝 \"{preview}\"")
        except Exception:
            pass

        # 4) chat_handler 호출 — 텍스트 메시지처럼 처리
        try:
            await ctx.bot.send_chat_action(
                chat_id=incoming_chat_id, action="typing",
            )
        except Exception:
            pass

        try:
            response = await handler(text)
        except Exception as e:
            print(f"  [Telegram-voice] chat_handler 예외: {e}")
            await update.message.reply_text(
                f"⚠️ EIDOS 처리 중 오류: {str(e)[:200]}"
            )
            return

        if not response:
            response = "(응답 없음)"
        if len(response) > 4000:
            response = response[:4000] + "\n\n… (응답 길어서 잘림)"
        try:
            await update.message.reply_text(response)
        except Exception as e:
            print(f"  [Telegram-voice] 응답 전송 실패: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────────────

def _esc(text: str) -> str:
    """
    HTML parse_mode 용 안전 이스케이프.

    [변경 2026-04-19 Phase 1]
    기존 MarkdownV2 이스케이프(`\\_ \\* \\. ...` 등) → HTML 이스케이프로 전환.
    MarkdownV2는 예약문자가 18개로 많아 리터럴 한 글자만 빠져도 전체 파싱 실패 →
    실사용에서 BadRequest 반복. HTML은 `< > &` 3개만 escape 하면 되고
    태그(<b>, <i>, <code>, <pre>)가 깨끗해 재발 위험 거의 없음.
    """
    if text is None:
        return ""
    return html.escape(str(text), quote=False)


def _fmt_bullet(label: str, value: str, *, code: bool = False) -> Optional[str]:
    """
    Phase 1 — 승인 메시지 포맷 헬퍼.

    빈 값이면 None 반환 → 호출부에서 filter(None, ...) 로 제거.
    code=True 면 값 부분을 <code>...</code> 로 감싸 monospace(URL/경로 가독성 ↑).
    """
    if value is None:
        return None
    v = str(value).strip()
    if not v:
        return None
    v_esc = _esc(v)
    if code:
        v_esc = f"<code>{v_esc}</code>"
    return f"<b>{_esc(label)}</b> {v_esc}"


def _next_seq() -> int:
    """[Phase 2] 요청 순번 카운터 증가 후 반환."""
    global _rid_seq
    _rid_seq += 1
    return _rid_seq


def _queue_tag(this_seq: int) -> str:
    """[Phase 2] 헤더용 '#N · 대기 M건' 라벨. _pending 길이 기반."""
    pending_n = len(_pending)  # 호출 시점 기준
    if pending_n <= 1:
        return f"<b>#{this_seq}</b>"
    return f"<b>#{this_seq}</b> · 대기 <b>{pending_n}건</b>"


def _fmt_duration(duration_ms: Optional[float]) -> str:
    """[Phase 2] finalize 시 사용할 소요시간 문자열."""
    if duration_ms is None or duration_ms < 0:
        return ""
    if duration_ms < 1000:
        return f"{int(duration_ms)}ms"
    s = duration_ms / 1000.0
    if s < 60:
        return f"{s:.1f}s"
    m, s2 = divmod(s, 60)
    return f"{int(m)}m {int(s2)}s"


def _fire_callback(req: ApprovalRequest, action: str, final_text: str):
    """승인/거절/수정 결과를 EIDOS 콜백으로 전달."""
    if req.callback:
        try:
            req.callback(action, final_text)
        except Exception as e:
            print(f"  [Telegram] callback 오류: {e}")
    print(f"  [Telegram] 처리 완료 | action={action} | rid={req.request_id} | 고객={req.customer_name}")


# ─────────────────────────────────────────────────────────────────────────────
# 전역 싱글턴
# ─────────────────────────────────────────────────────────────────────────────

_bot_instance: Optional[EidosTelegramBot] = None


def get_bot() -> EidosTelegramBot:
    """전역 봇 인스턴스 반환 (없으면 생성)."""
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = EidosTelegramBot()
    return _bot_instance


# ─────────────────────────────────────────────────────────────────────────────
# EIDOS에서 호출하는 공개 API
# ─────────────────────────────────────────────────────────────────────────────

async def request_approval(
    customer_name: str,
    customer_msg:  str,
    draft_reply:   str,
    on_approved:   Optional[Callable[[str], None]] = None,
    on_rejected:   Optional[Callable[[], None]] = None,
) -> Optional[str]:
    """
    EIDOS에서 호출하는 진입점.

    예:
        from eidos_telegram_bot import request_approval

        async def _send_to_kmong(final_text: str):
            # ActionDispatcher로 크몽 메시지창에 JS 주입
            ...

        await request_approval(
            customer_name="김원장",
            customer_msg="예약 가능한가요?",
            draft_reply="안녕하세요! 네, 내일 오후 가능합니다.",
            on_approved=_send_to_kmong,
        )
    """
    bot = get_bot()
    if not bot.is_configured():
        print("⚠️ [Telegram] 설정 미완료. eidos_files/telegram_config.json 확인")
        return None

    def _cb(action: str, final_text: str):
        if action in ("approved", "edited") and on_approved:
            try:
                # 동기 함수면 직접 호출, 비동기면 ensure_future
                import inspect
                if inspect.iscoroutinefunction(on_approved):
                    asyncio.ensure_future(on_approved(final_text))
                else:
                    on_approved(final_text)
            except Exception as e:
                print(f"  [Telegram] on_approved 오류: {e}")
        elif action == "rejected" and on_rejected:
            try:
                on_rejected()
            except Exception as e:
                print(f"  [Telegram] on_rejected 오류: {e}")

    return await bot.send_approval_request(
        customer_name=customer_name,
        customer_msg=customer_msg,
        draft_reply=draft_reply,
        callback=_cb,
    )


def start_bot():
    """EIDOS 시작 시 봇 폴링 시작. async_main에서 호출."""
    get_bot().start_polling()


def stop_bot():
    get_bot().stop_polling()


# ─────────────────────────────────────────────────────────────────────────────
# [Daily Report 2026-04-29] D2 — 일일 보고 카드 발송 + 후보 캐시
# ─────────────────────────────────────────────────────────────────────────────
# _on_button_click 의 daily_goal: prefix 분기가 lookup 하는 캐시.
# date(YYYY-MM-DD) → sub-goal candidate texts (3개).
# 사용자 클릭 시 idx 로 캐시 lookup → start_planning(seed_goal) 호출.
# 봇 재시작 시 캐시 비워짐 — callback miss 시 친절한 메시지 (만료 안내).
_DAILY_CANDIDATES_CACHE: Dict[str, List[str]] = {}


async def send_daily_report_via_bot(report: Dict[str, Any]) -> bool:
    """일일 보고 카드를 텔레그램으로 발송. D3 cron 이 호출.

    - report: eidos_daily_report.compose_daily_report() 결과 dict
    - 카드: format_daily_report_html() HTML 본문 + sub-goal 후보 4 버튼 (1/2/3/skip)
    - 후보 텍스트는 _DAILY_CANDIDATES_CACHE[date] 에 보관 → 콜백에서 lookup
    - 봇 미설정 시 콘솔 로그만, False 반환 (D3 가 대안 채널 폴백 가능)
    """
    bot = get_bot()
    if not bot.is_configured():
        print("  ⚠️ [DailyReport] 텔레그램 봇 미설정 — 카드 발송 skip")
        return False
    if not TELEGRAM_LOADED:
        print("  ⚠️ [DailyReport] python-telegram-bot 미설치 — skip")
        return False

    try:
        from eidos_daily_report import format_daily_report_html
    except Exception as e:
        print(f"  ⚠️ [DailyReport] format 모듈 로드 실패: {e}")
        return False

    date_str = str(report.get("date", "") or "")
    cands = list(report.get("sub_goal_candidates") or [])
    # 캐시에 후보 텍스트 보관 (콜백에서 idx 로 lookup)
    if date_str and cands:
        _DAILY_CANDIDATES_CACHE[date_str] = cands[:3]

    body = format_daily_report_html(report)

    # 후보 버튼 — 후보 있는 경우만, 최대 3개 + skip
    rows = []
    if cands:
        # 후보 라벨 짧게 (버튼 한 줄에 최대 25자)
        for i, c in enumerate(cands[:3], 1):
            label = f"{i}. {c[:22]}{'...' if len(c) > 22 else ''}"
            rows.append([InlineKeyboardButton(
                label, callback_data=f"daily_goal:{i}:{date_str}"
            )])
    rows.append([InlineKeyboardButton(
        "⏭️ 무시 (내일 다시)", callback_data=f"daily_goal:skip:{date_str}"
    )])
    kb = InlineKeyboardMarkup(rows)

    try:
        bot_inst = Bot(token=bot.token)
        await bot_inst.send_message(
            chat_id=bot.chat_id,
            text=body,
            parse_mode="HTML",
            reply_markup=kb,
        )
        print(f"  ✅ [DailyReport] 카드 발송 완료 (date={date_str}, "
              f"후보 {len(cands[:3])}개)")
        return True
    except Exception as e:
        print(f"  ⚠️ [DailyReport] 카드 발송 실패: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# [B3 협력 게이트 다층화 2026-04-29] Tier 2 짧은 질문 카드 인프라
# ─────────────────────────────────────────────────────────────────────────────
# _on_button_click 의 quick_ans: 분기가 lookup 하는 Future 캐시.
# rid (uuid hex) → {"future": Future, "options": List[str]}
# 사용자가 옵션 클릭 시 Future.set_result(option_text), timeout 시 cancel.
_PENDING_QUICK_QUESTIONS: Dict[str, Dict[str, Any]] = {}

# [Phase 2.7.1 2026-05-03] Worldmap 분기점 — fire-and-forget callback.
# rid → {"callback": func(node_id), "timeout_at": ts}
_PENDING_WORLDMAP_BRANCHES: Dict[str, Dict[str, Any]] = {}

# [Phase 2.7.2 2026-05-03] Worldmap 검증 응답 — fire-and-forget callback.
# rid → {"callback": func("ok"|"rollback"), "timeout_at": ts}
_PENDING_WORLDMAP_VERIFY: Dict[str, Dict[str, Any]] = {}


def send_worldmap_branch_async(
    title: str, options: List[Tuple[str, str]],
    on_response,
    timeout: float = 300.0,
) -> bool:
    """[Phase 2.7.1] Worldmap 분기점 inline keyboard 카드 — fire-and-forget.

    options: [(button_label, node_id_or_STOP), ...]
    on_response(node_id) — 사용자 클릭 시 호출 (별도 thread). __STOP__ 도 가능.

    Returns: 발송 시도 성공 여부 (True 도 실제 응답 보장 X — fire-and-forget).
    """
    bot = get_bot()
    if not bot.is_configured() or not TELEGRAM_LOADED:
        return False
    rid = uuid.uuid4().hex[:10]
    _PENDING_WORLDMAP_BRANCHES[rid] = {
        "callback": on_response,
        "timeout_at": time.time() + timeout,
    }
    rows = []
    for label, nid in options[:5]:
        rows.append([InlineKeyboardButton(
            label[:30], callback_data=f"worldmap_branch:{rid}:{nid}"
        )])
    rows.append([InlineKeyboardButton(
        "⏹ 정지", callback_data=f"worldmap_branch:{rid}:__STOP__"
    )])
    kb = InlineKeyboardMarkup(rows)
    body = (
        f"🗺️ <b>{_esc(title[:200])}</b>\n\n"
        f"<i>응답 시간: {int(timeout)}s — GUI 모달 또는 여기서 클릭</i>"
    )

    async def _do_send():
        try:
            bot_inst = Bot(token=bot.token)
            await bot_inst.send_message(
                chat_id=bot.chat_id, text=body,
                parse_mode="HTML", reply_markup=kb,
            )
        except Exception as e:
            print(f"  ⚠️ [Worldmap Branch] 발송 실패: {e}")
            _PENDING_WORLDMAP_BRANCHES.pop(rid, None)

    # [Fix 2026-05-03] Python 3.10+ 에서 background thread (Thread-5 등) 는
    # asyncio.get_event_loop() 가 RuntimeError("no current event loop in thread").
    # 1) 메인 thread 에 running loop 있으면 thread-safe 로 schedule
    # 2) 없으면 fresh loop 만들어 호출 thread 에서 직접 run_until_complete
    try:
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not None and running.is_running():
            asyncio.run_coroutine_threadsafe(_do_send(), running)
        else:
            new_loop = asyncio.new_event_loop()
            try:
                new_loop.run_until_complete(_do_send())
            finally:
                new_loop.close()
        return True
    except Exception as e:
        print(f"  ⚠️ [Worldmap Branch] event loop 실패: {e}")
        _PENDING_WORLDMAP_BRANCHES.pop(rid, None)
        return False


def send_worldmap_verify_async(
    title: str, on_response,
    timeout: float = 600.0,
) -> bool:
    """[Phase 2.7.2] Worldmap 검증 카드 — '✅ 완료' / '↶ 되돌리기' 2 버튼.

    on_response("ok"|"rollback") — fire-and-forget callback.
    """
    bot = get_bot()
    if not bot.is_configured() or not TELEGRAM_LOADED:
        return False
    rid = uuid.uuid4().hex[:10]
    _PENDING_WORLDMAP_VERIFY[rid] = {
        "callback": on_response,
        "timeout_at": time.time() + timeout,
    }
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ 완료 확정", callback_data=f"worldmap_verify:{rid}:ok")],
        [InlineKeyboardButton("↶ 되돌리기", callback_data=f"worldmap_verify:{rid}:rollback")],
    ])
    body = (
        f"🔍 <b>이동 후 검증</b>\n\n"
        f"{_esc(title[:300])}\n\n"
        f"<i>응답 없으면 {int(timeout)}s 후 자동 완료 처리</i>"
    )

    async def _do_send():
        try:
            bot_inst = Bot(token=bot.token)
            await bot_inst.send_message(
                chat_id=bot.chat_id, text=body,
                parse_mode="HTML", reply_markup=kb,
            )
        except Exception as e:
            print(f"  ⚠️ [Worldmap Verify] 발송 실패: {e}")
            _PENDING_WORLDMAP_VERIFY.pop(rid, None)

    # [Fix 2026-05-03] background thread 호환 — Worldmap Branch 와 동일 패턴.
    try:
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not None and running.is_running():
            asyncio.run_coroutine_threadsafe(_do_send(), running)
        else:
            new_loop = asyncio.new_event_loop()
            try:
                new_loop.run_until_complete(_do_send())
            finally:
                new_loop.close()
        return True
    except Exception as e:
        print(f"  ⚠️ [Worldmap Verify] event loop 실패: {e}")
        _PENDING_WORLDMAP_VERIFY.pop(rid, None)
        return False



async def send_quick_question(
    question: str,
    options: List[str],
    *,
    timeout: float = 120.0,
    title: str = "추가 정보 필요",
) -> Optional[str]:
    """Tier 2 (정보 부족) 시 짧은 옵션 질문 카드 발송.

    인자:
      question: 짧은 질문 본문 (1-2줄, 200자 이내 권장)
      options : 사용자 선택지 목록 (최대 6개, 각 라벨 22자 이내 권장)
      timeout : 응답 대기 시간 (초). 기본 120s.
      title   : 카드 헤더 라벨

    반환:
      선택된 옵션 텍스트 (사용자가 클릭한 그대로) — 또는 None (timeout / 봇 미설정 / 발송 실패)

    호출자가 await 하면 사용자 응답까지 block. fire-and-forget 으로 호출 가능.
    """
    bot = get_bot()
    if not bot.is_configured():
        print("  ⚠️ [Quick Q] 텔레그램 봇 미설정 — 질문 발송 skip")
        return None
    if not TELEGRAM_LOADED:
        print("  ⚠️ [Quick Q] python-telegram-bot 미설치 — skip")
        return None
    if not options:
        print("  ⚠️ [Quick Q] options 빈 리스트 — skip")
        return None

    options = options[:6]  # 최대 6개
    rid = uuid.uuid4().hex[:10]
    fut: asyncio.Future = asyncio.get_event_loop().create_future()
    _PENDING_QUICK_QUESTIONS[rid] = {"future": fut, "options": list(options)}

    # 버튼 — 옵션 라벨 22자 truncate, 한 줄에 1개씩 (모바일 가독성)
    rows = []
    for i, opt in enumerate(options, 1):
        label = f"{i}. {opt[:22]}{'...' if len(opt) > 22 else ''}"
        rows.append([InlineKeyboardButton(
            label, callback_data=f"quick_ans:{rid}:{i}"
        )])
    kb = InlineKeyboardMarkup(rows)

    body = (
        f"❓ <b>{_esc(title)}</b>\n\n"
        f"{_esc(question[:300])}\n\n"
        f"<i>응답 시간: {int(timeout)}s</i>"
    )

    try:
        bot_inst = Bot(token=bot.token)
        await bot_inst.send_message(
            chat_id=bot.chat_id,
            text=body,
            parse_mode="HTML",
            reply_markup=kb,
        )
    except Exception as e:
        print(f"  ⚠️ [Quick Q] 카드 발송 실패: {e}")
        _PENDING_QUICK_QUESTIONS.pop(rid, None)
        if not fut.done():
            fut.cancel()
        return None

    # 응답 대기 — timeout 시 None
    try:
        result = await asyncio.wait_for(fut, timeout=timeout)
        print(f"  ✅ [Quick Q] 응답 수신 (rid={rid[:6]}): {result[:40]}")
        return result
    except asyncio.TimeoutError:
        print(f"  ⏱️ [Quick Q] timeout {int(timeout)}s — 응답 없음 (rid={rid[:6]})")
        return None
    except asyncio.CancelledError:
        return None
    finally:
        _PENDING_QUICK_QUESTIONS.pop(rid, None)


# ─────────────────────────────────────────────────────────────────────────────
# [B4 협력 게이트 다층화 2026-04-29] Tier 3 미리보기 카드 인프라
# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 (의도 misalignment / Verifier 누적 cap) 진입 시 사용자에게:
#   - Verifier _last_verifier_fail (fail_type / reason / repair_hint)
#   - 누적 cap reason (force_user_reason)
#   - 마지막 산출물 head (history 의 마지막 WRITE/READ_PAGE)
#   - 옵션 [수정 후 재시도 / 이대로 진행 / 취소]
# 카드 본문 빌더 + send_quick_question 위임 (Future/캐시/timeout 인프라 재사용).
TIER3_OPTIONS: List[str] = ["수정 후 재시도", "이대로 진행", "취소"]


def _build_tier3_preview(schema: Any, history: Optional[List[Dict[str, Any]]] = None) -> str:
    """Tier 3 카드 본문 합성 (HTML).

    schema._last_verifier_fail (Verifier 누적 cap 마킹 시점에 저장됨) +
    schema._force_user_reason + history 의 마지막 WRITE/READ_PAGE 산출물 head.

    schema 가 None 또는 정보 없으면 graceful — 빈 섹션 생략.
    """
    import html as _html
    lines: List[str] = []

    # Verifier 실패 사유
    fail = getattr(schema, "_last_verifier_fail", None) if schema is not None else None
    if isinstance(fail, dict) and fail:
        ftype = str(fail.get("fail_type", "") or "?")
        reason = str(fail.get("reason", "") or "")[:300]
        repair = str(fail.get("repair_hint", "") or "")[:200]
        lines.append(f"<b>🔍 Verifier 판정</b>: <code>{_html.escape(ftype)}</code>")
        if reason:
            lines.append(f"  <i>{_html.escape(reason)}</i>")
        if repair:
            lines.append(f"  💡 <b>수정 힌트</b>: {_html.escape(repair)}")
    else:
        lines.append("<b>🔍 Verifier 판정</b>: (정보 없음)")

    # 누적 cap reason
    fu_reason = ""
    try:
        if schema is not None:
            fu_reason = str(getattr(schema, "_force_user_reason", "") or "")
    except Exception:
        fu_reason = ""
    if fu_reason:
        lines.append(f"<b>📊 누적</b>: {_html.escape(fu_reason[:150])}")

    # 마지막 산출물 head (history 의 마지막 WRITE/READ_PAGE)
    last_step: Optional[Dict[str, Any]] = None
    for h in reversed(history or []):
        if isinstance(h, dict) and h.get("action") in ("WRITE", "READ_PAGE"):
            last_step = h
            break
    if last_step:
        target = str(last_step.get("target", "") or "")[:60]
        action = str(last_step.get("action", "?"))
        result_str = str(last_step.get("result", "") or "")[:500]
        lines.append("")
        lines.append(
            f"<b>📄 마지막 산출물</b>: [{action}] <code>{_html.escape(target)}</code>"
        )
        if result_str:
            # 5줄 head (텔레그램 카드 길이 제한)
            result_lines = result_str.split("\n")[:5]
            head = "\n".join(result_lines)
            lines.append(f"<pre>{_html.escape(head)}</pre>")

    return "\n".join(lines)


async def send_tier3_preview(
    schema: Any,
    history: Optional[List[Dict[str, Any]]] = None,
    *,
    timeout: float = 300.0,
) -> Optional[str]:
    """Tier 3 미리보기 카드 발송 + 응답 대기.

    send_quick_question 인프라 재사용 (Future/캐시/timeout 통합).

    반환:
      "수정 후 재시도" | "이대로 진행" | "취소" | None (timeout/실패)
    """
    body = _build_tier3_preview(schema, history)
    return await send_quick_question(
        question=body,
        options=TIER3_OPTIONS,
        timeout=timeout,
        title="Tier 3 — 의도 검토 필요",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 단독 실행 (설정 마법사 + 테스트)
# ─────────────────────────────────────────────────────────────────────────────

async def _setup_wizard():
    """처음 실행 시 토큰/chat_id 설정 안내."""
    bot = get_bot()

    print("\n" + "="*50)
    print("  EIDOS 텔레그램 봇 설정 마법사")
    print("="*50)

    if bot.is_configured():
        print(f"✅ 이미 설정됨 | chat_id={bot.chat_id}")
        ans = input("재설정하려면 y, 테스트만 하려면 엔터: ").strip().lower()
        if ans != "y":
            return

    token   = input("\nBotFather에서 받은 토큰을 붙여넣어줘:\n> ").strip()
    print("\n텔레그램에서 @eidos_seungjun_bot 에게 /start 보낸 후")
    chat_id = input("아래 URL 열어서 chat id 확인해줘:\n"
                    f"https://api.telegram.org/bot{token}/getUpdates\n"
                    "chat_id 입력: ").strip()

    bot.configure(token, chat_id)

    print("\n테스트 메시지 발송 중...")
    await bot.send_notification(
        "✅ <b>EIDOS 봇 연결 성공!</b>\n\n"
        "크몽 고객 문의가 오면 이 채팅으로 승인 요청이 올 거야."
    )
    print("✅ 테스트 메시지 발송 완료. 텔레그램 확인해봐.")

    print("\n승인 요청 테스트 발송 중...")

    approved_result = []

    def _on_approved(text: str):
        approved_result.append(text)
        print(f"\n✅ 승인됨! 최종 텍스트: {text}")

    def _on_rejected():
        print("\n❌ 거절됨.")

    await request_approval(
        customer_name="테스트 고객",
        customer_msg="안녕하세요, 미용실 CRM 서비스 문의드립니다. 가격이 어떻게 되나요?",
        draft_reply="안녕하세요! 관심 가져주셔서 감사합니다 😊 저희 서비스는 월 30,000원으로 예약/고객관리를 자동화해드립니다. 자세한 내용은 상세페이지를 참고해주세요!",
        on_approved=_on_approved,
        on_rejected=_on_rejected,
    )

    print("\n텔레그램 앱에서 버튼 눌러봐. 결과 기다리는 중...")
    print("(Ctrl+C로 종료)")

    # 폴링 시작해서 버튼 응답 수신
    bot.start_polling()
    try:
        while True:
            await asyncio.sleep(1)
            if approved_result:
                print("테스트 완료!")
                break
    except KeyboardInterrupt:
        pass
    finally:
        bot.stop_polling()


if __name__ == "__main__":
    asyncio.run(_setup_wizard())