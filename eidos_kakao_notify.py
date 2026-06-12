# -*- coding: utf-8 -*-
"""[2026-06-03 ①] 카톡 중요 메시지 → AIRA 음성 알림 핵심 로직.

순수 로직 + WinRT(Windows UserNotificationListener) 래퍼. EIDOS GUI 무의존·
테스트 가능(WinRT 는 fake 주입). 화면이 보이지 않아도 OS 토스트 알림 텍스트를 읽는다.

설계:
- notifications_available(): winsdk 존재 + 리스너 접근 가능 여부.
- request_access(): UserNotificationListener.request_access_async 래핑(권한 요청).
- KakaoNotifSource: 실 WinRT 폴링(없으면 graceful 비활성).
- poll_new(raw, seen, ...): 신규 카톡 알림만 추림(필터 + de-dup) — raw 주입 가능(테스트).
- classify_importance_async(): 규칙(VIP/키워드/광고) 우선 → 애매하면 Gemini Flash.
- RateLimiter: 분당 N건 제한(과다 알림 방지).
- read_kakao_window_text(): UI Automation 폴백(winsdk 불가 환경·선택).

프라이버시: 로컬 규칙으로 ignore 를 먼저 거르고, 애매한 것만 LLM 에 보낸다(설정으로 LLM off 가능).
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass

# 광고/봇/채널성 발신자 힌트(소문자 비교) — 즉시 ignore 후보
_AD_HINTS = (
    "광고", "이벤트", "쿠폰", "할인", "프로모", "혜택", "채널", "알림톡",
    "플러스친구", "구독", "뉴스", "ad", "promotion", "noreply", "no-reply",
)

# 카톡 앱 식별 힌트(app_id/app_name 소문자 포함 검사)
_KAKAO_HINTS = ("kakao", "카카오", "카톡", "kakaotalk")

# [2026-06-03] 수신 '전화' 특화 힌트(엄격) — 일반 "전화"(예: "전화할게")는 오탐이라 제외.
# 카톡 보이스톡/페이스톡, Windows 휴대폰과 연결(Phone Link), 디스코드/팀즈 통화 등 토스트.
_CALL_TEXT_HINTS = (
    "보이스톡", "페이스톡", "음성 통화", "영상 통화", "음성통화", "영상통화",
    "수신 전화", "수신전화", "전화가 왔", "전화를 겁니다", "전화를 걸어",
    "통화 요청", "통화요청", "님이 전화", "님의 전화", "전화 수신",
    "incoming call", "is calling", "calling you", "video call", "voice call",
    "incoming video", "incoming voice", "missed call", "부재중 전화", "부재중전화",
)


@dataclass
class Notif:
    app_id: str = ""
    app_name: str = ""
    sender: str = ""
    text: str = ""
    ts: float = 0.0
    key: str = ""

    def __post_init__(self):
        if not self.key:
            self.key = make_key(self.sender, self.text)

    def to_dict(self) -> dict:
        return {"app_id": self.app_id, "app_name": self.app_name,
                "sender": self.sender, "text": self.text, "ts": self.ts, "key": self.key}

    @staticmethod
    def from_dict(d: dict) -> "Notif":
        d = d or {}
        return Notif(
            app_id=str(d.get("app_id", "") or ""),
            app_name=str(d.get("app_name", "") or ""),
            sender=str(d.get("sender", "") or "").strip(),
            text=str(d.get("text", "") or "").strip(),
            ts=float(d.get("ts", 0.0) or 0.0),
            key=str(d.get("key", "") or ""),
        )


def make_key(sender: str, text: str) -> str:
    """발신자+본문 해시(id 없을 때 de-dup 키)."""
    raw = f"{(sender or '').strip()}|{(text or '').strip()}"
    return hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()[:16]


# ── 카톡 판별 / de-dup ──────────────────────────────────────────────────────
def is_kakao(notif: Notif, *, extra_app_ids=()) -> bool:
    blob = f"{notif.app_id} {notif.app_name}".lower()
    if any(h in blob for h in _KAKAO_HINTS):
        return True
    for aid in (extra_app_ids or ()):
        if aid and aid.lower() in (notif.app_id or "").lower():
            return True
    return False


def is_call(notif: Notif) -> bool:
    """수신 전화 알림인지 — 전화 특화 텍스트 힌트가 있으면 True(엄격·오탐 최소).

    카톡 보이스톡/페이스톡, Phone Link 미러링 휴대폰 전화, 메신저 통화 모두 포함.
    """
    blob = f"{notif.sender} {notif.text}".lower()
    return any(h in blob for h in _CALL_TEXT_HINTS)


def poll_new(raw, seen_ids: set, *, extra_app_ids=(), include_calls=True) -> list:
    """raw(list[dict|Notif]) → 신규 카톡 메시지 + (선택)전화 알림. seen_ids 에 key 누적.

    raw 는 KakaoNotifSource.fetch() 또는 테스트 주입. seen_ids 는 호출자가 보관.
    include_calls=True 면 카톡이 아닌 앱(예: Phone Link)의 전화 알림도 통과시킨다.
    """
    out = []
    for item in (raw or []):
        n = item if isinstance(item, Notif) else Notif.from_dict(item)
        if not (n.sender or n.text):
            continue
        keep = is_kakao(n, extra_app_ids=extra_app_ids)
        if not keep and include_calls and is_call(n):
            keep = True
        if not keep:
            continue
        if n.key in seen_ids:
            continue
        seen_ids.add(n.key)
        out.append(n)
    return out


# ── 중요도 규칙(로컬·LLM 전에 먼저) ─────────────────────────────────────────
def quick_rule(notif: Notif, *, vip=(), keywords=()) -> str:
    """로컬 빠른 판정 → "high" | "ignore" | "" (=애매·LLM 판단 필요)."""
    sender_l = (notif.sender or "").lower()
    text_l = (notif.text or "").lower()
    blob = f"{sender_l} {text_l}"
    # VIP 발신자 → 즉시 high
    for v in (vip or ()):
        v = (v or "").strip().lower()
        if v and v in sender_l:
            return "high"
    # 키워드(회의·긴급·송금 등) → 즉시 high
    for k in (keywords or ()):
        k = (k or "").strip().lower()
        if k and k in blob:
            return "high"
    # 광고/채널/봇 → ignore
    if any(h in blob for h in _AD_HINTS):
        return "ignore"
    return ""


def _strip_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


def _safe_json(text: str):
    t = _strip_fences(text)
    try:
        return json.loads(t)
    except Exception:
        pass
    s = t.find("{"); e = t.rfind("}")
    if 0 <= s < e:
        try:
            return json.loads(t[s:e + 1])
        except Exception:
            pass
    return None


async def classify_importance_async(
    notif: Notif,
    *,
    llm_async=None,
    vip=(),
    keywords=(),
    use_llm: bool = True,
    detect_call: bool = True,
    timeout_sec: float = 20.0,
) -> dict:
    """카톡 1건 중요도 판정. 전화 우선 → 규칙 → 애매하면 Gemini Flash.

    반환: {"important": bool, "level": "call|high|normal|ignore",
           "reason": str, "speak": str(AIRA 가 읽을 한 문장·존댓말)}.
    level=="call" 은 수신 전화 — 긴급하므로 즉시 알림(빈도제한/중요도필터 우회 권장).
    """
    # [2026-06-03] 수신 전화 — LLM 기다릴 새 없이 즉시 "받으시는 게 좋겠어요".
    if detect_call and is_call(notif):
        return {"important": True, "level": "call", "reason": "수신 전화",
                "speak": _call_speak(notif, vip=vip)}

    rule = quick_rule(notif, vip=vip, keywords=keywords)
    if rule == "ignore":
        return {"important": False, "level": "ignore", "reason": "광고/채널/봇으로 판단",
                "speak": ""}
    if rule == "high":
        sp = _default_speak(notif, "high")
        return {"important": True, "level": "high", "reason": "VIP/키워드 일치", "speak": sp}

    # 애매 → LLM(설정으로 off 가능)
    if not use_llm or llm_async is None:
        # LLM 미사용 시 보수적으로 normal(읽어줌)
        return {"important": True, "level": "normal", "reason": "규칙 미해당(LLM off)",
                "speak": _default_speak(notif, "normal")}

    prompt = (
        "너는 사용자의 비서 AIRA 야. 방금 도착한 카카오톡 알림이 '지금 음성으로 알려줄 만큼 "
        "중요한지' 판정해.\n"
        "- 광고/채널/스팸/단순 인사/이모티콘만 → ignore.\n"
        "- 일반 대화 → normal. 회의·약속·긴급·금전·업무 마감 등 → high.\n"
        f"[발신자] {notif.sender}\n[내용] {notif.text[:300]}\n"
        "speak 은 AIRA 가 사용자에게 읽어줄 한 문장(친근한 존댓말 해요체·반말 금지·25자 내외).\n"
        "JSON 만 출력(설명 금지):\n"
        '{"level":"high|normal|ignore","reason":"...","speak":"..."}'
    )
    raw = ""
    try:
        import asyncio
        raw = await asyncio.wait_for(
            llm_async(prompt, max_tokens=160, use_cache=False), timeout=timeout_sec)
    except Exception as e:
        print(f"[kakao] classify 실패(graceful): {e}")
        raw = ""

    data = _safe_json(raw) if raw else None
    if not isinstance(data, dict):
        # 판정 실패 → 보수적으로 normal
        return {"important": True, "level": "normal", "reason": "LLM 판정 실패→보수적 알림",
                "speak": _default_speak(notif, "normal")}

    level = str(data.get("level", "normal") or "normal").strip().lower()
    if level not in ("high", "normal", "ignore"):
        level = "normal"
    speak = str(data.get("speak", "") or "").strip() or _default_speak(notif, level)
    return {
        "important": level != "ignore",
        "level": level,
        "reason": str(data.get("reason", "") or "").strip(),
        "speak": "" if level == "ignore" else speak,
    }


def _default_speak(notif: Notif, level: str) -> str:
    s = (notif.sender or "누군가").strip()
    if level == "high":
        return f"{s} 님에게서 중요한 카톡이 왔어요."
    return f"{s} 님에게서 카톡이 왔어요."


def _call_speak(notif: Notif, *, vip=()) -> str:
    """수신 전화 안내 문장(존댓말·'받으시는 게 좋겠어요')."""
    s = (notif.sender or "").strip()
    # 발신자에서 이름만 추려보기(전화 토스트는 보통 발신자 줄이 이름)
    name = s if s and len(s) <= 20 else ""
    is_vip = False
    sl = s.lower()
    for v in (vip or ()):
        v = (v or "").strip().lower()
        if v and v in sl:
            is_vip = True
            break
    if name and is_vip:
        return f"{name} 님 전화예요. 중요한 분이니 받으시는 게 좋겠어요."
    if name:
        return f"{name} 님에게서 전화가 왔어요. 받으시는 게 좋겠어요."
    return "전화가 왔어요. 받으시는 게 좋겠어요."


# ── rate-limit(과다 알림 방지) ──────────────────────────────────────────────
class RateLimiter:
    """분당 max_per_min 건으로 음성 알림 제한(슬라이딩 윈도우)."""

    def __init__(self, max_per_min: int = 4, window_sec: float = 60.0):
        self.max = max(1, int(max_per_min))
        self.window = float(window_sec)
        self._stamps: list = []

    def allow(self, now: float = None) -> bool:
        now = time.time() if now is None else now
        self._stamps = [t for t in self._stamps if now - t < self.window]
        if len(self._stamps) >= self.max:
            return False
        self._stamps.append(now)
        return True


# ── WinRT 가용성 / 권한 ─────────────────────────────────────────────────────
def _import_winrt():
    """winsdk(구 winrt) UserNotificationListener 관련 모듈 지연 import. 실패 시 None."""
    try:
        from winsdk.windows.ui.notifications.management import (
            UserNotificationListener, UserNotificationListenerAccessStatus)
        from winsdk.windows.ui.notifications import NotificationKinds
        return {
            "UserNotificationListener": UserNotificationListener,
            "AccessStatus": UserNotificationListenerAccessStatus,
            "NotificationKinds": NotificationKinds,
        }
    except Exception:
        return None


def notifications_available() -> bool:
    """winsdk + 리스너 클래스 접근 가능 여부(권한과 별개)."""
    mods = _import_winrt()
    if not mods:
        return False
    try:
        return mods["UserNotificationListener"].current is not None
    except Exception:
        return False


async def request_access() -> bool:
    """알림 접근 권한 요청. 허용 시 True. winsdk 없거나 거부 시 False(graceful)."""
    mods = _import_winrt()
    if not mods:
        return False
    try:
        listener = mods["UserNotificationListener"].current
        status = await listener.request_access_async()
        return status == mods["AccessStatus"].ALLOWED
    except Exception as e:
        print(f"[kakao] request_access 실패(graceful): {e}")
        return False


class KakaoNotifSource:
    """실 WinRT 알림 소스 — fetch() 가 현재 토스트 알림을 raw dict 리스트로 반환.

    winsdk 없으면 enabled=False 로 조용히 비활성. GUI 는 enabled 만 확인하면 됨.
    """

    def __init__(self):
        self._mods = _import_winrt()
        self.enabled = bool(self._mods)
        self._listener = None
        if self.enabled:
            try:
                self._listener = self._mods["UserNotificationListener"].current
            except Exception:
                self.enabled = False

    async def ensure_access(self) -> bool:
        if not self.enabled:
            return False
        return await request_access()

    async def fetch(self) -> list:
        """현재 토스트 알림 → [{app_id, app_name, sender, text, ts, key}, ...].

        WinRT UserNotificationListener 에는 동기 GetNotifications 가 없고
        **GetNotificationsAsync(=get_notifications_async)** 만 있다 → await 필요.
        (구 버전 바인딩이 동기 메서드를 노출하면 그것도 폴백 지원.)
        """
        if not (self.enabled and self._listener):
            return []
        out = []
        try:
            kinds = self._mods["NotificationKinds"].TOAST
            getter_async = getattr(self._listener, "get_notifications_async", None)
            if getter_async is not None:
                notes = await getter_async(kinds)
            else:
                notes = self._listener.get_notifications(kinds)   # 구 바인딩 호환
            for un in notes:
                try:
                    out.append(self._parse_user_notification(un))
                except Exception:
                    continue
        except Exception as e:
            print(f"[kakao] fetch 실패(graceful): {e}")
        return [n for n in out if n]

    def _parse_user_notification(self, un) -> dict:
        app_id = app_name = sender = text = ""
        ts = time.time()
        try:
            app_info = un.app_info
            app_id = str(getattr(app_info, "app_user_model_id", "") or "")
            disp = getattr(app_info, "display_info", None)
            if disp is not None:
                app_name = str(getattr(disp, "display_name", "") or "")
        except Exception:
            pass
        try:
            notif = un.notification
            binding = notif.visual.get_binding(
                __import__("winsdk.windows.ui.notifications",
                           fromlist=["KnownNotificationBindings"]).KnownNotificationBindings.get_toast_generic())
            if binding is not None:
                texts = binding.get_text_elements()
                lines = [str(getattr(t, "text", "") or "") for t in texts]
                lines = [x for x in lines if x]
                if lines:
                    sender = lines[0]
                    text = " ".join(lines[1:]) if len(lines) > 1 else lines[0]
        except Exception:
            pass
        try:
            nid = getattr(un, "id", None)
            key = str(nid) if nid else make_key(sender, text)
        except Exception:
            key = make_key(sender, text)
        return {"app_id": app_id, "app_name": app_name, "sender": sender,
                "text": text, "ts": ts, "key": key}


# ── UI Automation 폴백(선택·winsdk 불가 환경) ───────────────────────────────
def read_kakao_window_text() -> str:
    """카톡 창 텍스트 스크레이핑(창이 보여야 함). winsdk 불가 환경 대비·최선노력."""
    try:
        import pywinauto  # noqa: F401
    except Exception:
        return ""
    try:
        from pywinauto import Desktop
        for w in Desktop(backend="uia").windows():
            try:
                title = w.window_text() or ""
            except Exception:
                title = ""
            if any(h in title.lower() for h in _KAKAO_HINTS):
                try:
                    return w.window_text()
                except Exception:
                    return ""
    except Exception:
        pass
    return ""
