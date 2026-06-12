# -*- coding: utf-8 -*-
"""
AIRA 똑똑한 일정 알림 (Phase 2d) — 순수 로직(테스트 가능).

eidos_chat_gui 의 일정 알림 카드/버튼이 이 모듈을 쓴다. Qt·타이머·TTS 부작용은
GUI에 남기고, 여기엔 데이터 조작·판단만 둔다.

기능:
  - 완료 처리: 플래너에서 해당 항목 checked=True (음성 '했어/끝났어' 또는 버튼)
  - 연속 일정 브리핑: 곧 이어지는 일정들을 함께 안내("이어서 3시에 미팅도 있어요")
  - 스누즈: 분 단위 재알림(타이머 재등록은 GUI, 여기선 라벨/분 계산만)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# 기본 스누즈 분
DEFAULT_SNOOZE_MIN = 10
# 연속 일정으로 묶을 시간 창(분) — 현재 일정 이후 이 안에 시작하는 것만 함께 안내
CONSECUTIVE_WINDOW_MIN = 180


def hhmm_to_min(s: str) -> Optional[int]:
    """'HH:mm' → 분. 형식 이상이면 None."""
    try:
        hh, mm = str(s).strip().split(":")
        return int(hh) * 60 + int(mm)
    except Exception:  # noqa: BLE001
        return None


def mark_done(planner: Dict[str, Any], date_iso: str, time_str: str,
              text: str) -> Tuple[Dict[str, Any], bool]:
    """플래너에서 (날짜·시각·텍스트) 항목을 checked=True 로. (planner, 변경여부) 반환.
    텍스트는 정확히 일치 우선, 없으면 포함 관계로 너그럽게 매칭(이모지 접두 차이 흡수)."""
    day = planner.get(date_iso)
    if not isinstance(day, list):
        return planner, False
    norm = (text or "").strip()
    found = False
    # 1차: 시각+정확한 텍스트
    for it in day:
        if isinstance(it, dict) and it.get("time") == time_str and (it.get("text") or "").strip() == norm:
            it["checked"] = True
            found = True
    if not found:   # 2차: 시각 같고 텍스트 포함관계
        for it in day:
            if isinstance(it, dict) and it.get("time") == time_str:
                t = (it.get("text") or "").strip()
                if norm and (norm in t or t in norm):
                    it["checked"] = True
                    found = True
                    break
    return planner, found


def consecutive_after(items: List[Dict[str, Any]], current_time: str,
                      window_min: int = CONSECUTIVE_WINDOW_MIN,
                      limit: int = 3) -> List[Tuple[str, str]]:
    """현재 일정(current_time) 이후, window_min 분 안에 시작하는 '안 끝낸' 일정들을
    시간순으로 반환 [(time, text), ...]. 연속 일정 브리핑용."""
    cur = hhmm_to_min(current_time)
    if cur is None:
        return []
    out = []
    for it in items or []:
        if not isinstance(it, dict) or it.get("checked"):
            continue
        tm = hhmm_to_min(it.get("time", ""))
        if tm is None:
            continue
        if cur < tm <= cur + window_min:
            out.append((it.get("time", ""), (it.get("text") or "").strip()))
    out.sort(key=lambda x: hhmm_to_min(x[0]) or 9999)
    return out[:limit]


def _clean_text(text: str) -> str:
    """이모지 접두(📅 📧 등)·공백 정리 — 음성 멘트용."""
    t = (text or "").strip()
    for pre in ("📅", "📧", "🔔", "•", "-"):
        if t.startswith(pre):
            t = t[len(pre):].strip()
    return t


def consecutive_briefing(items: List[Dict[str, Any]], current_time: str) -> str:
    """연속 일정 음성 멘트. 없으면 빈 문자열."""
    nxts = consecutive_after(items, current_time)
    if not nxts:
        return ""
    parts = [f"{t.replace(':', '시 ')}분 {_clean_text(x)}" for t, x in nxts]
    # "14시 00분" 어색 → 정시는 '시'로
    parts = [p.replace("시 00분", "시").replace("시 0", "시 ") for p in parts]
    if len(parts) == 1:
        return f"이어서 {parts[0]} 일정도 있어요."
    return "이어서 " + ", ".join(parts[:-1]) + f", 그리고 {parts[-1]} 일정이 있어요."


def snooze_label(minutes: int) -> str:
    return f"⏰ {minutes}분 뒤 다시"
