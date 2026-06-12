# eidos_kmong_inbox_poller.py
# [2026-05-27 E-vision-hybrid] 크몽 인박스 자동 polling — text-first vision-fallback.
#
# 흐름:
#   1. EIDOS 내장 브라우저 현재 URL 확인 → 크몽 도메인이고 인박스 류 path 인지
#   2. get_browser_visible_text() — visible DOM 텍스트 추출 (무료)
#   3. 직전 polling 의 text_hash 와 비교 → 같으면 변화 없음 → skip
#   4. 다르면 — LLM (텍스트) 에게 diff 분석 시킴: "새 의뢰자 메시지 있어?"
#   5. LLM 답이 애매 (e.g., "본문 잘림") 이면 — vision fallback (screenshot + vision LLM)
#   6. 최종 결과: {has_new_message, new_message_text, source_url, new_hash, error}
#
# caller 가 받아서 stage state 의 actor.indicators 에 흘려보냄 (이 모듈은 부수효과 없음).
#
# 사용자가 수동으로 EIDOS 브라우저 탭에서 크몽 인박스 페이지를 열어둬야 동작.
# 사용자가 다른 페이지로 이동했다면 skip — UX 거슬리지 않게 자동 navigate 안 함.
# ActionDispatcher 의 auth_required 회로가 비로그인 자동 감지·텔레그램 알림.

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional


# 크몽 인박스 URL 패턴 — 사용자가 어느 페이지에 있는지 휴리스틱 판정
# 향후 다른 platform 확장 시 패턴 추가하면 generic poller 로 발전.
_INBOX_URL_PATTERNS = (
    # 크몽
    "kmong.com/mypage/inbox",
    "kmong.com/mypage/messages",
    "kmong.com/inbox",
    "kmong.com/seller/inbox",
    "kmong.com/seller/messages",
    "kmong.com/messages",
    # 사용자가 정확한 path 알려주면 추가 가능
)

# 인박스가 아닌 그냥 크몽 page 만 봐도 동작은 함 (text 추출 후 LLM 이 판단).
_BASE_HOST_KEYWORDS = ("kmong.com",)


def _hash_text(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="replace")).hexdigest()[:16]


def _looks_like_inbox(url: str) -> bool:
    if not url:
        return False
    u = url.lower()
    return any(p in u for p in _INBOX_URL_PATTERNS)


def _looks_like_kmong(url: str) -> bool:
    if not url:
        return False
    u = url.lower()
    return any(host in u for host in _BASE_HOST_KEYWORDS)


# ── LLM 프롬프트 ──────────────────────────────────────────────────────
_TEXT_DIFF_SYSTEM = """너는 크몽 셀러 인박스 페이지 텍스트를 보고 "사용자(셀러)에게 새 의뢰자 메시지가 와있는지" 판단하는 보조 시스템이다.

[입력]
1. 직전 polling 시점의 페이지 텍스트 (없을 수도 있음)
2. 현재 페이지 텍스트
3. 사용자 stage 컨텍스트 (관련 의뢰자/상품 hint)

[출력 JSON]
{
  "has_new_message": <bool>,
  "new_message_text": "<신규 메시지 본문·없으면 빈 문자열>",
  "sender_hint": "<의뢰자 이름·아이디·구분자·없으면 빈>",
  "confidence": <0.0~1.0>,
  "needs_vision": <bool — true 면 텍스트만으론 불충분·vision fallback 권장>,
  "rationale": "<1~2문장 판단 근거>"
}

[원칙]
- 본문이 잘려있거나 미리보기만 있어 정확 판단 어려우면 needs_vision=true 로 표시.
- 광고·시스템 안내·자기 자신이 보낸 메시지는 has_new_message=false.
- 직전 텍스트 없으면 (= 첫 polling) 현재 시점 메시지를 "new" 로 잡지 말고 baseline 만 기록 (has_new_message=false, confidence 낮게).
- confidence < 0.7 이면 needs_vision=true 권장.

순수 JSON 만 출력 (마크다운 코드블록 금지)."""


_VISION_SYSTEM = """너는 크몽 셀러 인박스 페이지 스크린샷을 보고 "사용자에게 새 의뢰자 메시지가 와있는지" 정확히 판단하는 보조 시스템이다.

스크린샷에서 unread 표시·badge·새 메시지 indicator 를 시각적으로 식별. 본문 텍스트도 가능하면 추출.

[출력 JSON]
{
  "has_new_message": <bool>,
  "new_message_text": "<신규 메시지 본문·badge/UI 추출 가능 한도>",
  "sender_hint": "<의뢰자 이름>",
  "confidence": <0.0~1.0>,
  "rationale": "<1~2문장>"
}

순수 JSON 만 출력."""


async def _text_llm_diff(
    prev_text: str,
    cur_text: str,
    stage_name: str = "",
) -> Dict[str, Any]:
    """텍스트 LLM 호출 — diff 분석."""
    try:
        from llm_module import get_llm_response_async, robust_json_parse
    except Exception as e:
        return {"_error": f"llm_module import 실패: {e}"}

    _prev = (prev_text or "(이전 polling 없음 — 첫 호출)")[-3000:]
    _cur = (cur_text or "")[-5000:]
    user_prompt = (
        f"[stage: {stage_name}]\n\n"
        f"[직전 폴링 텍스트]\n{_prev}\n\n"
        f"━━━━━━━━━━━\n\n"
        f"[현재 페이지 텍스트]\n{_cur}\n\n"
        f"JSON 만 출력."
    )

    try:
        raw = await get_llm_response_async(
            user_prompt,
            system_prompt=_TEXT_DIFF_SYSTEM,
            response_mime_type="application/json",
            max_tokens=1024,
            timeout=45,
        )
    except Exception as e:
        return {"_error": f"LLM 호출 실패: {type(e).__name__}: {e}"}

    if not raw or not raw.strip():
        return {"_error": "LLM 응답 빈"}

    try:
        parsed = robust_json_parse(raw)
    except Exception:
        try:
            parsed = json.loads(raw)
        except Exception as e:
            return {"_error": f"JSON 파싱 실패: {e}", "_raw": raw[:300]}

    if not isinstance(parsed, dict):
        return {"_error": "응답이 dict 가 아님", "_raw": raw[:300]}
    return parsed


async def _vision_llm_check(
    screenshot_path: str,
    stage_name: str = "",
) -> Dict[str, Any]:
    """vision LLM 호출 — 스크린샷 + JSON 출력."""
    try:
        from llm_module import get_llm_response_vision_async, robust_json_parse
    except Exception as e:
        return {"_error": f"llm_module vision import 실패: {e}"}

    try:
        with open(screenshot_path, "rb") as f:
            image_bytes = f.read()
    except Exception as e:
        return {"_error": f"스크린샷 읽기 실패: {e}"}

    # vision 함수는 system_prompt 매개변수 없음 → 본문에 인라인.
    user_prompt = (
        f"{_VISION_SYSTEM}\n\n"
        f"━━━━━━━━━━━\n"
        f"[stage: {stage_name}]\n\n"
        "위 스크린샷에서 새 의뢰자 메시지가 있는지 정확히 판단해서 JSON 만 출력하라."
    )

    try:
        raw = await get_llm_response_vision_async(
            user_prompt,
            image_bytes=image_bytes,
        )
    except Exception as e:
        return {"_error": f"vision LLM 실패: {type(e).__name__}: {e}"}

    if not raw or not raw.strip():
        return {"_error": "vision LLM 응답 빈"}

    try:
        parsed = robust_json_parse(raw)
    except Exception:
        try:
            parsed = json.loads(raw)
        except Exception as e:
            return {"_error": f"vision JSON 파싱 실패: {e}", "_raw": raw[:300]}

    if not isinstance(parsed, dict):
        return {"_error": "vision 응답이 dict 가 아님", "_raw": raw[:300]}
    return parsed


# ── 메인 entry point ─────────────────────────────────────────────────
async def poll_once(
    last_text_hash: str = "",
    last_text: str = "",
    stage_name: str = "",
    force_vision: bool = False,
) -> Dict[str, Any]:
    """1회 polling 수행.

    Args:
        last_text_hash: 직전 polling 의 visible text hash (변화 비교용)
        last_text: 직전 polling 의 텍스트 (LLM diff context)
        stage_name: stage 이름 (LLM prompt context)
        force_vision: True 면 텍스트 결과 무관하게 vision fallback 실행 (테스트용)

    Returns: {
        "has_new_message": bool,
        "new_message_text": str,
        "sender_hint": str,
        "confidence": float,
        "source_url": str,
        "new_hash": str,
        "via": "skip_url" | "skip_unchanged" | "text" | "vision" | "error",
        "error": str (있을 때만),
        "rationale": str,
    }
    """
    result: Dict[str, Any] = {
        "has_new_message": False,
        "new_message_text": "",
        "sender_hint": "",
        "confidence": 0.0,
        "source_url": "",
        "new_hash": last_text_hash,
        "via": "skip_url",
        "rationale": "",
    }

    # Step 0: execution_module import + 현재 URL 확인
    try:
        from execution_module import (
            execute_js_with_result, get_browser_visible_text,
            grab_browser_screenshot,
        )
    except Exception as e:
        result["via"] = "error"
        result["error"] = f"execution_module import 실패: {e}"
        return result

    try:
        url_raw = await execute_js_with_result("window.location.href", timeout=3.0)
    except Exception as e:
        result["via"] = "error"
        result["error"] = f"URL 조회 실패: {type(e).__name__}: {e}"
        return result

    url = (url_raw or "").strip().strip('"').strip("'")
    result["source_url"] = url

    # 사용자가 다른 사이트 보고 있으면 skip (방해 X)
    if not _looks_like_kmong(url):
        result["rationale"] = f"현재 URL ({url[:60]}) 이 크몽 아님 — skip"
        return result

    # Step 1: 텍스트 추출
    try:
        cur_text = await get_browser_visible_text(timeout=6.0)
    except Exception as e:
        result["via"] = "error"
        result["error"] = f"텍스트 추출 실패: {type(e).__name__}: {e}"
        return result

    cur_text = (cur_text or "").strip()
    if not cur_text:
        result["via"] = "skip_unchanged"
        result["rationale"] = "추출된 텍스트 빈"
        return result

    new_hash = _hash_text(cur_text)
    result["new_hash"] = new_hash

    # Step 2: hash 비교 — 변화 없으면 skip (가장 흔한 case·무료)
    if not force_vision and new_hash == last_text_hash and last_text_hash:
        result["via"] = "skip_unchanged"
        result["rationale"] = f"text hash 동일 ({new_hash})"
        return result

    # Step 3: 텍스트 LLM 호출
    if not force_vision:
        text_r = await _text_llm_diff(last_text, cur_text, stage_name=stage_name)
        if text_r.get("_error"):
            # 텍스트 LLM 실패 — vision fallback 으로 진행
            print(f"[kmong_inbox_poller] 텍스트 LLM 실패 → vision fallback: {text_r['_error']}")
        else:
            _has_new = bool(text_r.get("has_new_message", False))
            _conf = float(text_r.get("confidence", 0.0) or 0.0)
            _needs_v = bool(text_r.get("needs_vision", False))
            if _has_new and _conf >= 0.7 and not _needs_v:
                # 텍스트만으로 충분
                result["has_new_message"] = True
                result["new_message_text"] = str(text_r.get("new_message_text", ""))[:1000]
                result["sender_hint"] = str(text_r.get("sender_hint", ""))[:80]
                result["confidence"] = _conf
                result["rationale"] = str(text_r.get("rationale", ""))[:200]
                result["via"] = "text"
                return result
            if not _has_new and _conf >= 0.7 and not _needs_v:
                # 텍스트만으로 "변화 없음" 확신
                result["has_new_message"] = False
                result["confidence"] = _conf
                result["rationale"] = str(text_r.get("rationale", ""))[:200]
                result["via"] = "text"
                return result
            # 그 외 — vision fallback 진행 (confidence 낮음 or needs_vision=true)

    # Step 4: vision fallback
    try:
        ss_path = await grab_browser_screenshot(timeout=8.0)
    except Exception as e:
        result["via"] = "error"
        result["error"] = f"스크린샷 실패: {type(e).__name__}: {e}"
        return result

    if not ss_path:
        result["via"] = "error"
        result["error"] = "스크린샷 빈 경로 (콜백 미등록 가능성)"
        return result

    v_r = await _vision_llm_check(ss_path, stage_name=stage_name)
    if v_r.get("_error"):
        result["via"] = "error"
        result["error"] = f"vision LLM 실패: {v_r['_error']}"
        return result

    result["has_new_message"] = bool(v_r.get("has_new_message", False))
    result["new_message_text"] = str(v_r.get("new_message_text", ""))[:1000]
    result["sender_hint"] = str(v_r.get("sender_hint", ""))[:80]
    result["confidence"] = float(v_r.get("confidence", 0.0) or 0.0)
    result["rationale"] = str(v_r.get("rationale", ""))[:200]
    result["via"] = "vision"
    return result


__all__ = [
    "poll_once",
    "_INBOX_URL_PATTERNS",   # 외부에서 추가 가능 (확장용)
    "_looks_like_kmong",
    "_looks_like_inbox",
]
