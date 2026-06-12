# -*- coding: utf-8 -*-
"""aira_shared_settings — AIRA·EIDOS·petalbot 공용 설정 (단일 JSON·세 프로세스 공유).

`~/.eidos/aira_settings.json` 하나를 세 곳이 다 읽고 쓴다(원자적·graceful).
TTS 속도/볼륨/on-off, 자율 발화 빈도 등 사용자 조절값의 단일 진실.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading

_DIR = os.path.join(os.path.expanduser("~"), ".eidos")
_PATH = os.path.join(_DIR, "aira_settings.json")
_LOCK = threading.Lock()

DEFAULTS = {
    "tts_enabled": True,
    # 0.5(느림) ~ 2.0(빠름), 1.0=표준. 기본을 약간 빠르게(사용자: "너무 느려").
    "tts_speed": 1.2,
    "tts_volume": 0.9,
    # off | low | normal | high — AIRA 자율 발화 빈도
    "autonomous_freq": "normal",
    # [2026-06-03 ①] 카톡 중요 알림 음성 — 기본 off(명시적 opt-in)
    "kakao_notify_enabled": False,
    # high_only | normal | all — 어느 수준부터 음성으로 알릴지
    "kakao_importance": "high_only",
    # 애매한 알림을 LLM 으로 판정할지(off 면 규칙만)
    "kakao_use_llm": True,
    # [2026-06-03] 수신 전화 알림("받으시는 게 좋겠어요") — 긴급이라 기본 on
    "kakao_call_enabled": True,
    # VIP 발신자 / 중요 키워드(쉼표구분 문자열·리스트 모두 허용)
    "kakao_vip": "",
    "kakao_keywords": "회의, 긴급, 송금, 마감, 약속",
}


def load() -> dict:
    """현재 설정(누락/손상 시 기본값으로 보정)."""
    try:
        with open(_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
        out = dict(DEFAULTS)
        if isinstance(d, dict):
            out.update({k: v for k, v in d.items() if k in DEFAULTS})
        return out
    except Exception:
        return dict(DEFAULTS)


def save(partial: dict) -> bool:
    """partial 을 기존값에 병합 저장(원자적·known key 만)."""
    with _LOCK:
        try:
            os.makedirs(_DIR, exist_ok=True)
            cur = load()
            cur.update({k: v for k, v in (partial or {}).items() if k in DEFAULTS})
            fd, tmp = tempfile.mkstemp(dir=_DIR, suffix=".tmp")
            os.close(fd)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cur, f, ensure_ascii=False, indent=2)
            os.replace(tmp, _PATH)
            return True
        except Exception as e:
            print(f"[aira_settings] save 실패(graceful): {e}")
            return False


def get(key: str, default=None):
    return load().get(key, DEFAULTS.get(key, default))


def set_value(key: str, value) -> bool:
    return save({key: value})


# ── 적용 헬퍼 (각 소비처가 동일 변환을 쓰게) ───────────────────────────────────
def pyttsx3_rate(base: int = 180) -> int:
    """tts_speed → pyttsx3 rate (단어/분). base*speed, 80~400 클램프."""
    try:
        return int(max(80, min(400, base * float(get("tts_speed", 1.0)))))
    except Exception:
        return base


def ncp_speed() -> int:
    """tts_speed → NCP Clova speed(-5 빠름 ~ +5 느림, 0 표준). 빠를수록 음수."""
    try:
        s = float(get("tts_speed", 1.0))
        return int(max(-5, min(5, round((1.0 - s) * 5))))
    except Exception:
        return 0


def autonomous_factor() -> float:
    """자율 발화 빈도 → 인터벌 배수. off=0(중단)·low=2x·normal=1x·high=0.5x."""
    return {"off": 0.0, "low": 2.0, "normal": 1.0, "high": 0.5}.get(
        str(get("autonomous_freq", "normal")), 1.0)


def csv_list(key: str) -> list:
    """쉼표 구분 문자열(또는 리스트) 설정값 → 정리된 list[str]."""
    v = get(key, "")
    if isinstance(v, (list, tuple)):
        return [str(x).strip() for x in v if str(x).strip()]
    return [s.strip() for s in str(v or "").split(",") if s.strip()]
