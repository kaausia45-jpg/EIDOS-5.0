# eidos_telegram_voice.py
# [Wave7-A 2026-05-28] STT 모듈 — 텔레그램 음성 메시지 → 텍스트.
#
# 사용자 시나리오:
#   1. 폰에서 텔레그램 봇한테 음성 메시지 보냄
#   2. 봇이 ogg 다운로드
#   3. 이 모듈로 transcribe
#   4. 텍스트를 EIDOS chat_handler 로 forward (자연어 ToM 명령 hook 자동)
#
# 백엔드 폴백 체인:
#   1. Gemini Flash audio (multimodal·이미 사용 중·추가 dep 0·가장 매끄러움)
#   2. OpenAI Whisper API (settings.OPENAI_API_KEY 있을 때·정확도 매우 높음)
#   3. (옵션) 로컬 faster-whisper — 설치돼 있을 때만 (현재는 미설치·dependency 부담 회피)
#
# 모든 단계 graceful. 실패 시 빈 string·호출자가 처리.

from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile


def _load_settings() -> dict:
    try:
        path = os.path.join("eidos_files", "settings.json")
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# ── 백엔드 1: Gemini Flash audio ──────────────────────────────────────
async def _transcribe_gemini_async(
    audio_bytes: bytes,
    mime_type: str = "audio/ogg",
    timeout_sec: float = 30.0,
) -> str:
    """Gemini Flash multimodal — audio inline_data 로 보내고 텍스트 받음.

    EIDOS 가 이미 google.generativeai 쓰고 있어서 추가 dep 0.
    inline_data 는 ~20MB 까지·텔레그램 음성 메시지는 대개 1MB 이하라 OK.
    """
    settings = _load_settings()
    api_key = settings.get("GEMINI_API_KEY", "")
    if not api_key:
        return ""

    try:
        import google.generativeai as genai
    except Exception as e:
        print(f"[STT-gemini] google.generativeai import 실패: {e}")
        return ""

    try:
        genai.configure(api_key=api_key)
        # 2.5 Flash 사용 — multimodal·빠름·저렴
        model = genai.GenerativeModel("gemini-2.5-flash")

        # inline_data — base64 자동 처리됨 (SDK 가)
        audio_part = {
            "mime_type": mime_type,
            "data": audio_bytes,
        }
        prompt_text = (
            "이 한국어 음성을 정확하게 텍스트로 받아써. "
            "발화 내용만 그대로 옮겨라. 인사·메타·설명·따옴표 X. "
            "고유명사 (사람 이름·제품·회사명) 가 헷갈리면 한글로 음역. "
            "음성이 비어있거나 알아들을 수 없으면 빈 string 만 출력."
        )

        # async 호출 — generate_content_async
        async def _call():
            response = await model.generate_content_async(
                [prompt_text, audio_part],
                generation_config={"max_output_tokens": 1024},
            )
            return response

        response = await asyncio.wait_for(_call(), timeout=timeout_sec)
        text = ""
        try:
            text = (response.text or "").strip()
        except Exception:
            # candidates 에서 직접 추출
            try:
                for cand in response.candidates:
                    for part in cand.content.parts:
                        if hasattr(part, "text") and part.text:
                            text += part.text
                text = text.strip()
            except Exception:
                pass

        # quote·라벨 제거
        text = text.strip('"').strip("'").strip()
        if text.startswith("발화 내용:"):
            text = text[len("발화 내용:"):].strip()
        if len(text) > 2000:
            text = text[:2000]
        return text
    except asyncio.TimeoutError:
        print(f"[STT-gemini] timeout {timeout_sec}s")
        return ""
    except Exception as e:
        print(f"[STT-gemini] 실패: {type(e).__name__}: {str(e)[:120]}")
        return ""


# ── 백엔드 2: OpenAI Whisper API ──────────────────────────────────────
async def _transcribe_whisper_api_async(
    audio_bytes: bytes,
    mime_type: str = "audio/ogg",
    timeout_sec: float = 60.0,
) -> str:
    """OpenAI Whisper API — gpt-4o-mini-transcribe 또는 whisper-1.

    settings.OPENAI_API_KEY 있을 때만 동작. ogg/m4a/wav 등 직접 지원.
    """
    settings = _load_settings()
    api_key = settings.get("OPENAI_API_KEY", "")
    if not api_key:
        return ""

    try:
        from openai import AsyncOpenAI
    except Exception as e:
        print(f"[STT-whisper] openai SDK import 실패: {e}")
        return ""

    # 파일로 임시 저장 (Whisper API 는 file-like 객체 요구)
    ext = ".ogg"
    if "wav" in mime_type:
        ext = ".wav"
    elif "mp3" in mime_type:
        ext = ".mp3"
    elif "m4a" in mime_type or "mp4" in mime_type:
        ext = ".m4a"
    elif "webm" in mime_type:
        ext = ".webm"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=ext, delete=False,
        ) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        client = AsyncOpenAI(api_key=api_key)

        async def _call():
            with open(tmp_path, "rb") as audio_file:
                resp = await client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="ko",
                    response_format="text",
                )
            # whisper-1 의 text 형식은 plain string
            return str(resp).strip() if resp else ""

        text = await asyncio.wait_for(_call(), timeout=timeout_sec)
        if len(text) > 2000:
            text = text[:2000]
        return text
    except asyncio.TimeoutError:
        print(f"[STT-whisper] timeout {timeout_sec}s")
        return ""
    except Exception as e:
        print(f"[STT-whisper] 실패: {type(e).__name__}: {str(e)[:120]}")
        return ""
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# ── 메인 — 폴백 체인 ─────────────────────────────────────────────────
async def transcribe_voice_async(
    audio_bytes: bytes,
    mime_type: str = "audio/ogg",
    timeout_sec: float = 30.0,
) -> str:
    """음성 → 텍스트. 백엔드 폴백 체인. 실패 시 빈 string.

    settings.stt_provider 로 선호 backend 선택 가능:
      - "gemini" (default) — Gemini Flash audio 우선
      - "whisper" — OpenAI Whisper API 우선
      - "auto" — Gemini 먼저·실패하면 Whisper

    audio_bytes 가 빈 bytes 면 즉시 빈 string 반환.
    """
    if not audio_bytes:
        return ""

    settings = _load_settings()
    provider = (settings.get("stt_provider", "auto") or "auto").lower()

    chain = []
    if provider == "gemini":
        chain = [_transcribe_gemini_async]
    elif provider == "whisper":
        chain = [_transcribe_whisper_api_async]
    else:   # auto / 기본
        chain = [_transcribe_gemini_async, _transcribe_whisper_api_async]

    last_err = None
    for backend in chain:
        try:
            text = await backend(audio_bytes, mime_type, timeout_sec)
            if text and text.strip():
                return text.strip()
        except Exception as e:
            last_err = e
            print(f"[STT] backend {backend.__name__} 예외: {e}")
            continue

    if last_err:
        print(f"[STT] 모든 backend 실패 — last: {last_err}")
    return ""


# ── 헬퍼: bytes ↔ base64 (디버깅·로그용) ──────────────────────────────
def audio_bytes_preview(audio_bytes: bytes) -> str:
    """디버깅용 — 처음 32 바이트 base64 + 길이 표시."""
    if not audio_bytes:
        return "(empty)"
    head = base64.b64encode(audio_bytes[:32]).decode("ascii")
    return f"{len(audio_bytes)} bytes · head={head}"
