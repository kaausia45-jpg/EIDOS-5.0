# -*- coding: utf-8 -*-
"""
AIRA Gmail 지휘 로직 (Phase 2) — 순수/테스트 가능 헬퍼.

eidos_chat_gui 의 _aira_menu_mail_digest / 자동 폴링이 이 모듈을 호출한다.
LLM은 주입(llm_async)받으므로 단위 테스트에서 가짜 함수로 대체 가능.

설계 결정(사용자 승인):
  - 트리거: 자동 폴링(기본 15분) + 수동 메뉴 둘 다
  - 발화 범위: '중요' 또는 '답장필요'로 분류된 메일만 음성 보고(소음 조절)
  - 답장: 초안 자동 생성 → Gmail 임시보관함 저장 + 팝업, 발송은 사람 승인(자동발송 안 함)
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional

LLMAsync = Callable[..., Awaitable[str]]

# 분류 카테고리 — important/needs_reply 판정의 근거
CATEGORIES = ["중요", "답장필요", "업무", "뉴스레터", "광고", "알림", "일반"]

CLASSIFY_SYS = (
    "너는 사장님의 비서 AIRA다. 받은 이메일 한 통을 보고 한국어로 분류한다. "
    "반드시 아래 JSON 스키마로만 답하라(설명·마크다운 금지).\n"
    '{"category": "중요|답장필요|업무|뉴스레터|광고|알림|일반", '
    '"important": true/false, "needs_reply": true/false, "has_event": true/false, '
    '"speak": "사장님께 한 문장으로 음성 보고할 말(없으면 빈 문자열)", '
    '"reason": "분류 근거 짧게"}\n'
    "판단 기준:\n"
    "- needs_reply=답장/회신/확인/승인/일정조율 등 사람의 응답을 명시적으로 기다리는 메일.\n"
    "- important=놓치면 곤란한 메일(계약·결제·마감·고객·보안경고·상사/거래처).\n"
    "- has_event=회의·미팅·약속·마감·예약 등 '날짜와 시간이 있는 일정'을 담은 메일(캘린더 등록 후보).\n"
    "- 뉴스레터·프로모션·자동알림(영수증/소셜알림 등)은 important=false, needs_reply=false.\n"
    "- speak는 important나 needs_reply 가 true일 때만 채우고, 발신자와 핵심을 자연스러운 존댓말로."
)


def _heuristic_flags(mail: Dict[str, Any], vip: Optional[List[str]],
                     keywords: Optional[List[str]]) -> Optional[Dict[str, Any]]:
    """VIP 발신자 / 키워드 빠른 경로 — 매치되면 LLM 없이 즉시 중요 처리.
    매치 없으면 None(→ LLM 분류로)."""
    blob = f"{mail.get('from','')} {mail.get('subject','')} {mail.get('snippet','')}".lower()
    sender = (mail.get("from", "") or "").lower()
    for v in vip or []:
        v = (v or "").strip().lower()
        if v and v in sender:
            return {"category": "중요", "important": True, "needs_reply": True,
                    "speak": f"{mail.get('from','')} 님에게 메일이 왔어요. 제목은 '{mail.get('subject','')}' 이에요.",
                    "reason": f"VIP 발신자({v})"}
    for k in keywords or []:
        k = (k or "").strip().lower()
        if k and k in blob:
            return {"category": "중요", "important": True, "needs_reply": False,
                    "speak": f"'{k}' 관련 메일이 왔어요. 제목은 '{mail.get('subject','')}' 이에요.",
                    "reason": f"키워드({k})"}
    return None


def _coerce_classification(raw: str, mail: Dict[str, Any]) -> Dict[str, Any]:
    """LLM 응답(JSON 문자열)을 안전하게 dict로. 실패 시 보수적 기본값(중요 아님)."""
    try:
        d = json.loads((raw or "").strip())
        if not isinstance(d, dict):
            raise ValueError("not a dict")
    except Exception:
        return {"category": "일반", "important": False, "needs_reply": False,
                "speak": "", "reason": "분류 실패(기본값)"}
    cat = str(d.get("category", "일반")).strip() or "일반"
    important = bool(d.get("important", False))
    needs_reply = bool(d.get("needs_reply", False))
    has_event = bool(d.get("has_event", False))
    speak = str(d.get("speak", "") or "").strip()
    # 일관성 보정: 중요/답장필요인데 speak 비면 최소 멘트 생성
    if (important or needs_reply) and not speak:
        speak = f"{mail.get('from','')} 님 메일: {mail.get('subject','')}"
    if not (important or needs_reply):
        speak = ""   # 소음 조절 — 안 중요하면 말하지 않음
    return {"category": cat, "important": important, "needs_reply": needs_reply,
            "has_event": has_event, "speak": speak,
            "reason": str(d.get("reason", "") or "")}


async def classify_mail_async(mail: Dict[str, Any], llm_async: LLMAsync,
                              vip: Optional[List[str]] = None,
                              keywords: Optional[List[str]] = None) -> Dict[str, Any]:
    """메일 한 통 분류. mail = {from, subject, snippet, ...}."""
    fast = _heuristic_flags(mail, vip, keywords)
    if fast is not None:
        return fast
    prompt = (f"[발신자] {mail.get('from','')}\n"
              f"[제목] {mail.get('subject','')}\n"
              f"[미리보기] {mail.get('snippet','')}")
    try:
        raw = await llm_async(prompt, response_mime_type="application/json",
                              system_prompt=CLASSIFY_SYS, max_tokens=400)
    except Exception as e:  # noqa: BLE001
        return {"category": "일반", "important": False, "needs_reply": False,
                "speak": "", "reason": f"LLM 오류: {e}"}
    return _coerce_classification(raw, mail)


DRAFT_SYS = (
    "너는 사장님의 비서 AIRA다. 받은 이메일에 대한 한국어 답장 초안을 작성한다. "
    "정중한 존댓말로, 사장님이 그대로 보내거나 살짝 고쳐 보낼 수 있게 자연스럽게. "
    "확정되지 않은 정보(날짜·금액·약속)는 절대 지어내지 말고, 필요한 부분은 "
    "'[확인 후 기입]' 같은 자리표시자나 상대에게 되묻는 문장으로 남겨라. "
    "메일 본문만 출력하라(제목·머리말·메타설명·코드블록 금지)."
)


async def draft_reply_async(mail_full: Dict[str, Any], llm_async: LLMAsync) -> str:
    """답장 초안 본문 생성. mail_full = {from, subject, body, ...}."""
    body = (mail_full.get("body", "") or "")[:4000]
    prompt = (f"[받은 메일 발신자] {mail_full.get('from','')}\n"
              f"[받은 메일 제목] {mail_full.get('subject','')}\n"
              f"[받은 메일 본문]\n{body}\n\n"
              "위 메일에 대한 답장 초안을 작성해줘.")
    try:
        raw = await llm_async(prompt, system_prompt=DRAFT_SYS, max_tokens=900)
    except Exception:  # noqa: BLE001
        return ""
    return (raw or "").strip()


EVENT_SYS = (
    "너는 비서 AIRA다. 이메일에서 '캘린더에 등록할 일정'을 정확히 1건 추출한다. "
    "회의·미팅·약속·예약·마감·통화 등 '구체적 날짜와 시각'이 있는 것만 일정으로 본다.\n"
    "반드시 아래 JSON으로만 답하라(설명·마크다운 금지).\n"
    '{"has_event": true/false, "date": "yyyy-MM-dd", "time": "HH:mm", '
    '"title": "일정 제목(간결)", "confidence": 0.0~1.0}\n'
    "규칙:\n"
    "- 날짜·시각이 모두 분명할 때만 has_event=true. 둘 중 하나라도 모호하면 false.\n"
    "- '내일/모레/이번 주 금요일' 같은 상대 표현은 [오늘] 기준으로 yyyy-MM-dd 로 환산.\n"
    "- 종일 일정이라 시각이 없으면 time=\"09:00\".\n"
    "- 과거 날짜이거나 확실치 않으면 has_event=false. 추측으로 지어내지 마라."
)


async def extract_event_async(mail_full: dict, llm_async: LLMAsync,
                              today_iso: str = "") -> Optional[dict]:
    """메일 본문에서 일정 1건 추출. {date, time, title} 또는 None(일정 없음/불확실).
    today_iso = 'yyyy-MM-dd'(상대 날짜 환산 기준)."""
    body = (mail_full.get("body", "") or "")[:4000]
    prompt = (f"[오늘] {today_iso}\n"
              f"[발신자] {mail_full.get('from','')}\n"
              f"[제목] {mail_full.get('subject','')}\n"
              f"[본문]\n{body}")
    try:
        raw = await llm_async(prompt, response_mime_type="application/json",
                              system_prompt=EVENT_SYS, max_tokens=300)
        d = json.loads((raw or "").strip())
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(d, dict) or not d.get("has_event"):
        return None
    date = str(d.get("date", "") or "").strip()
    time = str(d.get("time", "") or "").strip() or "09:00"
    title = str(d.get("title", "") or "").strip()
    # 형식 검증 — yyyy-MM-dd / HH:mm 아니면 버림(날조 방지)
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date) or not re.match(r"^\d{2}:\d{2}$", time):
        return None
    if today_iso and date < today_iso:        # 과거 일정 무시
        return None
    if not title:
        title = (mail_full.get("subject", "") or "일정").strip()
    return {"date": date, "time": time, "title": title[:80],
            "source_subject": mail_full.get("subject", "")}


def reply_subject(subject: str) -> str:
    """답장 제목 — 이미 Re: 면 그대로, 아니면 'Re: ' 접두."""
    s = (subject or "").strip()
    if s.lower().startswith("re:"):
        return s
    return f"Re: {s}" if s else "Re:"


def sender_email(from_header: str) -> str:
    """'홍길동 <a@b.com>' → 'a@b.com'. 꺾쇠 없으면 통째로."""
    s = (from_header or "").strip()
    if "<" in s and ">" in s:
        return s[s.index("<") + 1:s.index(">")].strip()
    return s


def build_briefing(classified: List[Dict[str, Any]]) -> str:
    """분류 결과(speak 채워진 항목들) → AIRA 음성 멘트 한 덩어리."""
    speaks = [c.get("speak", "").strip() for c in classified
              if (c.get("important") or c.get("needs_reply")) and c.get("speak", "").strip()]
    if not speaks:
        return "새로 살펴봤는데 지금 중요한 메일은 없어요."
    n = len(speaks)
    head = f"중요한 메일이 {n}건 있어요. "
    return head + " 그리고 ".join(speaks[:3]) + (
        f" 그 밖에 {n - 3}건 더 있어요." if n > 3 else "")


# ── 자동 폴링 dedup — 이미 보고한 메일 id 영속 ─────────────────────────
def load_seen(path: str) -> set:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return set(data)
    except Exception:  # noqa: BLE001
        pass
    return set()


def save_seen(path: str, seen: set, cap: int = 500) -> None:
    try:
        lst = list(seen)[-cap:]   # 최근 cap개만 유지
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(lst, f)
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ [AIRA mail] seen 저장 실패: {e}")


async def run_digest(core: Any, llm_async: LLMAsync, *, auto: bool,
                     seen_path: str, vip: Optional[List[str]] = None,
                     keywords: Optional[List[str]] = None,
                     today_iso: str = "", max_results: int = 12) -> Optional[Dict[str, Any]]:
    """전체 사이클 오케스트레이션(GUI 무관·테스트 가능).
    core 는 async gmail_list_messages/gmail_get_message/gmail_create_draft 를 가진 객체.
    반환: None(폴링인데 보고할 것 없음) 또는
          {"briefing","level","drafts":[(subject,body)],"events":[...],"important_n","classified"}.
    """
    mails = await core.gmail_list_messages(unread_only=True, max_results=max_results) or []

    # 첫 자동 폴링(seen 파일 없음) = 기준선만 조용히 설정.
    # 기존 안읽음 더미를 한꺼번에 떠들지 않고, '지금 이후 새로 오는 메일'만 감지하기 위함.
    # (현재 안읽음을 훑어보려면 수동 '메일 브리핑' 사용 → auto=False라 baseline 안 탐)
    if auto and not os.path.exists(seen_path):
        save_seen(seen_path, {m.get("id", "") for m in mails})
        return None

    seen = load_seen(seen_path) if auto else set()
    fresh = [m for m in mails if not (auto and m.get("id") in seen)]
    if auto and not fresh:
        return None

    classified: List[Dict[str, Any]] = []
    drafts: List = []
    events: List[Dict[str, Any]] = []
    for mail in fresh:
        c = await classify_mail_async(mail, llm_async, vip=vip, keywords=keywords)
        classified.append(c)
        # 답장필요/일정후보면 본문을 한 번만 읽어 초안·일정추출에 공용
        if c.get("needs_reply") or c.get("has_event"):
            full = await core.gmail_get_message(mail.get("id", "")) or {}
            if c.get("needs_reply"):
                body = await draft_reply_async(full, llm_async)
                if body:
                    to = sender_email(full.get("from", "") or mail.get("from", ""))
                    subj = reply_subject(full.get("subject", "") or mail.get("subject", ""))
                    try:
                        await core.gmail_create_draft(
                            to=to, subject=subj, body=body, thread_id=mail.get("thread_id", ""))
                    except Exception as _de:  # noqa: BLE001
                        print(f"⚠️ [AIRA mail] 초안 저장 실패: {_de}")
                    drafts.append((mail.get("subject", ""), body))
            if c.get("has_event"):
                ev = await extract_event_async(full, llm_async, today_iso=today_iso)
                if ev:
                    events.append(ev)

    if auto:
        for mail in fresh:
            seen.add(mail.get("id", ""))
        save_seen(seen_path, seen)

    important_n = sum(1 for c in classified if c.get("important") or c.get("needs_reply"))
    # 일정을 등록했다면 중요하지 않아도 폴링에서 보고(비서로서 알려줄 가치 있음)
    if auto and important_n == 0 and not events:
        return None   # 폴링 소음 조절 — 새 메일은 있었지만 중요한 것도 일정도 없음

    briefing = build_briefing(classified)
    if drafts:
        briefing += f" 답장이 필요한 {len(drafts)}건은 초안을 임시보관함에 준비해 뒀어요."
    if events:
        briefing += f" 그리고 메일에서 일정 {len(events)}건을 발견해 캘린더에 등록했어요."
    return {"briefing": briefing, "level": "high" if (important_n or events) else "normal",
            "drafts": drafts, "events": events,
            "important_n": important_n, "classified": classified}
