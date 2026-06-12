# -*- coding: utf-8 -*-
"""
AIRA 통화 — 마이크 캡처 + 턴 분절(VAD). 음향 결합 통화의 '귀' 입력단.

순수 로직(RMS 침묵 감지·턴 분절 상태기계)은 테스트 가능. 실제 마이크 캡처는
sounddevice 지연 import(미설치면 graceful). 16kHz mono int16 기준(CSR 권장 포맷).

턴 분절: 말이 시작되고(임계 이상) → 일정 시간 침묵하면 한 턴 종료로 보고 그 구간을 반환.
반이중이라 AIRA가 말하는 동안엔 GUI가 캡처를 멈춘다(자기 목소리 STT 방지).
"""
from __future__ import annotations

from typing import List, Optional

SAMPLE_RATE = 16000
FRAME_MS = 30                       # 한 프레임 길이(ms)
SILENCE_RMS = 500.0                 # 이보다 작으면 침묵(int16 기준, max 32767)
MIN_SPEECH_MS = 300                 # 이만큼은 말해야 '발화'로 인정(블립 무시)
END_SILENCE_MS = 900               # 발화 후 이만큼 침묵하면 턴 종료
MAX_TURN_MS = 15000                # 한 턴 최대 길이(폭주 방지)


def rms_level(pcm16: bytes) -> float:
    """PCM16 바이트의 RMS 진폭(0~32767). numpy 있으면 사용, 없으면 순수 파이썬."""
    if not pcm16:
        return 0.0
    try:
        import numpy as _np
        a = _np.frombuffer(pcm16, dtype=_np.int16).astype(_np.float64)
        if a.size == 0:
            return 0.0
        return float((a * a).mean() ** 0.5)
    except Exception:  # noqa: BLE001
        import array
        import math
        a = array.array("h")
        a.frombytes(pcm16[: len(pcm16) // 2 * 2])
        if not a:
            return 0.0
        return math.sqrt(sum(x * x for x in a) / len(a))


def is_silent(pcm16: bytes, threshold: float = SILENCE_RMS) -> bool:
    return rms_level(pcm16) < threshold


class TurnSegmenter:
    """프레임을 먹여 한 턴(발화 구간)을 잘라낸다.
    feed() 가 턴 완성 시 그 구간 오디오(bytes)를, 아니면 None 을 반환."""

    def __init__(self, silence_rms: float = SILENCE_RMS,
                 min_speech_ms: int = MIN_SPEECH_MS,
                 end_silence_ms: int = END_SILENCE_MS,
                 max_turn_ms: int = MAX_TURN_MS, frame_ms: int = FRAME_MS):
        self.silence_rms = silence_rms
        self.min_speech_ms = min_speech_ms
        self.end_silence_ms = end_silence_ms
        self.max_turn_ms = max_turn_ms
        self.frame_ms = frame_ms
        self._buf: List[bytes] = []
        self._speech_ms = 0
        self._silence_ms = 0
        self._in_speech = False

    def reset(self):
        self._buf.clear()
        self._speech_ms = 0
        self._silence_ms = 0
        self._in_speech = False

    def feed(self, frame: bytes) -> Optional[bytes]:
        """프레임 1개 투입. 턴이 끝났으면 그 구간 오디오 반환(+내부 리셋), 아니면 None."""
        loud = not is_silent(frame, self.silence_rms)
        if loud:
            self._in_speech = True
            self._speech_ms += self.frame_ms
            self._silence_ms = 0
            self._buf.append(frame)
        elif self._in_speech:
            # 발화 중 잠깐의 침묵 — 일단 담아두되 침묵 누적
            self._silence_ms += self.frame_ms
            self._buf.append(frame)
        # else: 발화 시작 전 침묵 → 버림(앞쪽 무음 제거)

        total_ms = len(self._buf) * self.frame_ms
        # 턴 종료 조건: 충분히 말했고 + 끝 침묵 충분  /  또는 최대 길이 초과
        if self._in_speech and self._speech_ms >= self.min_speech_ms and \
                self._silence_ms >= self.end_silence_ms:
            return self._emit()
        if total_ms >= self.max_turn_ms and self._speech_ms >= self.min_speech_ms:
            return self._emit()
        return None

    def _emit(self) -> bytes:
        audio = b"".join(self._buf)
        self.reset()
        return audio


def capture_available() -> bool:
    """마이크 캡처 가능 여부(sounddevice 설치 + 입력 장치 존재)."""
    try:
        import sounddevice as _sd  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False
