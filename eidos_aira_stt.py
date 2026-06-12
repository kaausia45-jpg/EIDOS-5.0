# -*- coding: utf-8 -*-
"""
NCP CLOVA Speech(CSR) STT — 짧은 음성 1턴 → 한국어 텍스트.

기존 NCP TTS와 같은 API 게이트웨이·같은 키(NCP_ID/NCP_KEY)를 쓴다.
턴 단위(말 한 번 끝나면 그 구간 오디오를 보내 텍스트 받기)라 통화 반이중 모델에 맞다.

CSR 엔드포인트: POST https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang=Kor
헤더: X-NCP-APIGW-API-KEY-ID / X-NCP-APIGW-API-KEY, Content-Type: application/octet-stream
바디: 오디오 바이트(WAV/PCM 16k mono 권장). 응답: {"text": "..."}.
"""
from __future__ import annotations

import json
from typing import Optional

CSR_URL = "https://naveropenapi.apigw.ntruss.com/recog/v1/stt"


def parse_csr_response(body: bytes) -> str:
    """CSR 응답 바디(JSON) → 인식 텍스트. 실패/빈값이면 ''."""
    try:
        d = json.loads(body.decode("utf-8", "replace"))
        if isinstance(d, dict):
            return str(d.get("text", "") or "").strip()
    except Exception:  # noqa: BLE001
        pass
    return ""


def recognize_ncp(audio_bytes: bytes, ncp_id: str, ncp_key: str,
                  lang: str = "Kor", timeout: float = 10.0) -> str:
    """오디오 바이트 → 한국어 텍스트. 실패 시 '' 반환(예외 안 던짐).
    audio_bytes = WAV(16k mono 16bit 권장) 또는 raw PCM."""
    if not audio_bytes or not ncp_id or not ncp_key:
        return ""
    import urllib.request
    import urllib.error
    req = urllib.request.Request(
        f"{CSR_URL}?lang={lang}",
        data=audio_bytes,
        headers={
            "Content-Type": "application/octet-stream",
            "X-NCP-APIGW-API-KEY-ID": ncp_id,
            "X-NCP-APIGW-API-KEY": ncp_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return parse_csr_response(resp.read())
    except urllib.error.HTTPError as e:
        try:
            print(f"⚠️ [STT] CSR HTTP {e.code}: {e.read()[:200]!r}")
        except Exception:  # noqa: BLE001
            print(f"⚠️ [STT] CSR HTTP {e.code}")
        return ""
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ [STT] CSR 오류: {e}")
        return ""


def synth_ncp_wav(text: str, ncp_id: str, ncp_key: str, speaker: str = "dara",
                  speed: int = 0, pitch: int = -2, timeout: float = 10.0) -> bytes:
    """NCP CLOVA Voice → WAV 바이트(통화용·winsound 블로킹 재생). 실패 시 b''.
    WAV 로 받는 이유: winsound.PlaySound(SND_MEMORY) 가 WAV 만 재생하고 블로킹이라
    반이중(말하는 동안 안 듣기)에 딱 맞음."""
    text = (text or "").strip()
    if not text or not ncp_id or not ncp_key:
        return b""
    import urllib.parse
    import urllib.request
    import urllib.error
    body = urllib.parse.urlencode({
        "speaker": speaker, "speed": str(speed), "pitch": str(pitch),
        "text": text[:600], "format": "wav",
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts",
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "X-NCP-APIGW-API-KEY-ID": ncp_id,
            "X-NCP-APIGW-API-KEY": ncp_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ [TTS-call] 합성 실패: {e}")
        return b""


def pcm16_to_wav(pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """raw PCM16 → WAV 바이트(헤더 부착). CSR 가 WAV 를 안정적으로 받게."""
    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)          # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(pcm_bytes)
    return buf.getvalue()
