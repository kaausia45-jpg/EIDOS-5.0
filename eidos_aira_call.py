# -*- coding: utf-8 -*-
"""
AIRA 통화 대행 (Phase 4) — STT/오디오와 무관한 '대화 두뇌' 순수 로직.

방식: 음향 결합(acoustic coupling). 사용자가 전화를 받아 스피커폰으로 PC 옆에 두면,
PC 마이크가 상대 목소리를 듣고(STT) → 이 모듈이 응답을 정하고 → TTS로 PC 스피커가 말한다
→ 폰 마이크가 그 소리를 주워 상대에게 전달. 반이중(말할 땐 듣기 멈춤).

이 모듈엔 부작용(마이크/STT/TTS)이 없다. GUI가 STT 텍스트를 넣어주면 응답 텍스트를 돌려준다.
덕분에 가짜 입력으로 대화 흐름·턴 종료·통화 종료·요약을 단위 테스트할 수 있다.
"""
from __future__ import annotations

import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

LLMAsync = Callable[..., Awaitable[str]]

# 반이중 상태
STATE_LISTENING = "listening"   # 상대 말 듣는 중
STATE_THINKING = "thinking"     # STT→LLM 처리 중
STATE_SPEAKING = "speaking"     # AIRA가 말하는 중(이때 마이크 입력 무시=자기 목소리 차단)

# 통화 종료로 볼 상대 발화 신호
_BYE_PATTERNS = [
    r"끊을게", r"끊습니다", r"들어가세요", r"수고하세요", r"안녕히 계세요",
    r"이만", r"바이", r"goodbye", r"bye", r"끊어",
]


def call_system_prompt(owner_name: str = "사장님", persona_note: str = "") -> str:
    """AIRA가 통화를 대신 받을 때의 시스템 프롬프트."""
    who = owner_name or "사장님"
    extra = f"\n추가 지침: {persona_note}" if persona_note else ""
    return (
        f"너는 '{who}'의 음성 비서 AIRA다. 지금 {who}를 대신해 걸려온 전화를 받고 있다. "
        "상대(발신자)의 말을 듣고 한국어 존댓말로 짧고 자연스럽게 응대한다.\n"
        "원칙:\n"
        f"- 첫머리에 '{who}님을 대신해 비서 AIRA가 받았다'는 점을 자연스럽게 밝힌다.\n"
        "- 통화이므로 한 번에 1~2문장, 너무 길게 말하지 않는다(상대가 듣기 편하게).\n"
        f"- 모르는 것·결정 권한이 필요한 것은 지어내지 말고, 용건과 연락처를 받아 '{who}께 전달드리겠다'고 한다.\n"
        "- 정중하되 사무적이지 않게, 사람 비서처럼 따뜻하게.\n"
        "- 스팸·영업·보이스피싱 의심이면 정중히 거절하고 통화를 마무리한다.\n"
        f"- 상대가 마무리 인사를 하면 짧게 인사하고 끝낸다.{extra}\n"
        "출력은 네가 '말로 할 그 문장'만. 따옴표·지문·이모지 없이."
    )


def is_goodbye(text: str) -> bool:
    """상대 발화가 통화 종료 신호인지."""
    t = (text or "").strip().lower()
    return any(re.search(p, t) for p in _BYE_PATTERNS)


def clean_reply(raw: str) -> str:
    """LLM 응답을 음성용으로 정리 — 따옴표·괄호지문·이모지·과한 줄바꿈 제거."""
    s = (raw or "").strip()
    s = re.sub(r"\([^)]*\)", "", s)            # (지문) 제거
    s = re.sub(r'["""„""]', "", s)             # 따옴표 전역 제거(음성엔 불필요)
    s = re.sub(r"[\U0001F300-\U0001FAFF☀-➿]", "", s)  # 이모지 제거
    s = re.sub(r"\s*\n\s*", " ", s)            # 줄바꿈 → 공백
    s = re.sub(r"\s{2,}", " ", s).strip()
    s = s.strip("'").strip()
    return s


class CallSession:
    """한 통화의 대화 상태 — 히스토리·반이중 상태·종료 판정."""

    def __init__(self, owner_name: str = "사장님", persona_note: str = "",
                 max_turns: int = 30):
        self.owner_name = owner_name
        self.persona_note = persona_note
        self.max_turns = max_turns
        self.history: List[Tuple[str, str]] = []   # [("caller"|"aira", text), ...]
        self.state = STATE_LISTENING
        self.ended = False

    # ── 히스토리 ────────────────────────────────────────────────
    def add_caller(self, text: str):
        if text:
            self.history.append(("caller", text.strip()))

    def add_aira(self, text: str):
        if text:
            self.history.append(("aira", text.strip()))

    def turns(self) -> int:
        return sum(1 for who, _ in self.history if who == "caller")

    def transcript(self) -> str:
        lines = []
        for who, t in self.history:
            lines.append(("상대" if who == "caller" else "AIRA") + f": {t}")
        return "\n".join(lines)

    def _history_for_llm(self) -> str:
        return "\n".join(
            (("[상대] " if who == "caller" else "[AIRA] ") + t) for who, t in self.history)

    # ── 응답 생성 ───────────────────────────────────────────────
    async def respond(self, caller_text: str, llm_async: LLMAsync) -> str:
        """상대 발화 → AIRA 응답 텍스트. 히스토리에 양쪽 다 기록. 종료 신호면 ended=True."""
        self.add_caller(caller_text)
        sys = call_system_prompt(self.owner_name, self.persona_note)
        prompt = (f"[지금까지의 통화]\n{self._history_for_llm()}\n\n"
                  "위 흐름에 이어 AIRA가 상대에게 할 말 한 마디를 출력하라.")
        try:
            raw = await llm_async(prompt, system_prompt=sys, max_tokens=160)
        except Exception:  # noqa: BLE001
            raw = f"죄송해요, 잠시 문제가 있었어요. {self.owner_name}께 꼭 전달드릴게요."
        reply = clean_reply(raw) or "네, 말씀 계속해 주세요."
        self.add_aira(reply)
        # 종료 판정: 상대가 인사했거나 턴 한도 초과
        if is_goodbye(caller_text) or self.turns() >= self.max_turns:
            self.ended = True
        return reply

    def opening_line(self) -> str:
        """통화 첫 멘트(상대 발화 전에 AIRA가 먼저 받는 인사)."""
        line = (f"네, 안녕하세요. {self.owner_name}님을 대신해 비서 AIRA가 전화 받았습니다. "
                "어떤 일로 연락 주셨어요?")
        self.add_aira(line)
        return line


async def summarize_call(session: "CallSession", llm_async: LLMAsync) -> Dict[str, Any]:
    """통화 종료 후 요약 — 사용자에게 보고할 내용. {summary, caller, purpose, callback, action}."""
    import json
    sys = (
        "다음은 비서 AIRA가 사용자를 대신해 받은 전화의 전체 녹취다. "
        "사용자에게 보고할 요약을 아래 JSON으로만 출력하라(설명·마크다운 금지).\n"
        '{"summary": "한두 문장 요약", "caller": "발신자(파악되면)", '
        '"purpose": "용건", "callback": "회신처/연락처(있으면)", '
        '"action_needed": true/false, "spam_suspected": true/false}'
    )
    try:
        raw = await llm_async(f"[통화 녹취]\n{session.transcript()}",
                              response_mime_type="application/json",
                              system_prompt=sys, max_tokens=400)
        d = json.loads((raw or "").strip())
        if isinstance(d, dict):
            return d
    except Exception:  # noqa: BLE001
        pass
    return {"summary": session.transcript()[:200], "caller": "", "purpose": "",
            "callback": "", "action_needed": False, "spam_suspected": False}


def speak_summary(info: Dict[str, Any], owner_name: str = "사장님") -> str:
    """요약 dict → 사용자에게 들려줄 AIRA 음성 보고 멘트."""
    if not info:
        return "방금 전화를 한 통 대신 받았어요."
    parts = [f"{owner_name}, 방금 전화를 대신 받았어요."]
    caller = (info.get("caller") or "").strip()
    purpose = (info.get("purpose") or info.get("summary") or "").strip()
    if caller:
        parts.append(f"{caller}에게서 왔고,")
    if purpose:
        parts.append(f"용건은 {purpose} 였어요.")
    cb = (info.get("callback") or "").strip()
    if cb:
        parts.append(f"회신처는 {cb} 예요.")
    if info.get("spam_suspected"):
        parts.append("스팸이나 영업 전화로 의심돼서 정중히 마무리했어요.")
    elif info.get("action_needed"):
        parts.append("확인이 필요해 보여요.")
    return " ".join(parts)
