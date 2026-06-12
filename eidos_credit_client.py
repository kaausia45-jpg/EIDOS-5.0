# eidos_credit_client.py
# 크레딧 차감 시스템 완전 제거 — 본인 Gemini API 키 기반 무한 모드 (2026-05-25 사용자 명시 요청).
# 모든 잔액 함수가 _UNLIMITED_CREDITS 고정 반환·서버 호출 결과의 credits_remaining 무시·
# CreditError/ProRequiredError 더 이상 raise 되지 않음 (클래스 자체는 후방호환 위해 유지).

import hashlib
import json
import os
import platform
import time
import requests
from typing import Optional

SERVER_URL = "https://eidos-server-production.up.railway.app"
CREDITS_FILE = os.path.join(os.path.expanduser("~"), ".eidos_credits.json")

# [차감 제거] 무한 잔액 상수 — 모든 잔액 조회 함수가 이 값을 반환
_UNLIMITED_CREDITS = 99999
_UNLIMITED_TIER = "power"

# Gumroad 크레딧 구매 URL — 후방호환 (UI 코드에서 import 사용)
GUMROAD_STARTER = "https://kaausia.gumroad.com/l/eidos-credits-500"
GUMROAD_PRO     = "https://kaausia.gumroad.com/l/eidos-credits-1200"
GUMROAD_POWER   = "https://kaausia.gumroad.com/l/eidos-credits-3000"
GUMROAD_STORE   = "https://kaausia.gumroad.com"

# Pro 전용 액션 목록 — 후방호환 (UI 에서 import). 실제 게이팅은 비활성 (check_pro_feature 통과).
PRO_ONLY_ACTIONS = {
    "write_file", "execute_code", "run_python", "browser_open",
    "write_project_files", "iterative_coding", "scaffold_loop",
    "web_search", "read_file", "delete_file",
}


def get_device_id() -> str:
    """기기 고유 ID 생성 (재현 가능, 64자 hex)."""
    raw = platform.node() + platform.machine() + platform.processor()
    return hashlib.sha256(raw.encode()).hexdigest()


def _load_local_cache() -> dict:
    """[차감 제거] 항상 무한 잔액 반환·로컬 캐시 파일 무시."""
    return {
        "credits": _UNLIMITED_CREDITS,
        "tier": _UNLIMITED_TIER,
        "device_id": get_device_id(),
    }


def _save_local_cache(data: dict):
    """[차감 제거] 캐시 저장 no-op — 어차피 _load 가 항상 무한 반환."""
    return


# ── 공개 API ──────────────────────────────────────────────

def register_device() -> dict:
    """[차감 제거] 서버 호출 skip·항상 무한 잔액 즉시 반환."""
    return _load_local_cache()


def get_credits() -> dict:
    """[차감 제거] 서버 호출 skip·항상 무한 잔액 반환."""
    return _load_local_cache()


def _try_local_gemini_call(
    messages: list,
    system_prompt: Optional[str],
    max_tokens: int,
) -> Optional[dict]:
    """
    [하이브리드] eidos_settings.json 의 'gemini_api_key' 가 있으면 Gemini SDK 직접 호출.
    성공 시 call_ai 와 동일한 dict 반환 (content + credits_remaining=99999 + credits_used=0).
    실패/미설정 시 None 반환 → 호출자가 서버 폴백.
    """
    try:
        with open("eidos_settings.json", "r", encoding="utf-8") as f:
            api_key = (json.load(f).get("gemini_api_key") or "").strip()
    except Exception:
        api_key = ""
    if not api_key:
        return None
    try:
        import google.generativeai as _gv
    except ImportError:
        return None

    try:
        history_blob: list[str] = []
        last_user_text = ""
        for m in messages or []:
            role = (m.get("role") or "").lower()
            content = m.get("content") or ""
            if role == "user":
                last_user_text = content
                history_blob.append(f"[사용자] {content[:500]}")
            elif role == "assistant":
                history_blob.append(f"[EIDOS] {content[:500]}")
        if len(history_blob) > 1:
            ctx_lines = history_blob[:-1][-10:]
            full_prompt = (
                "[이전 대화 — 참고만]\n"
                + "\n".join(ctx_lines)
                + "\n\n[현재 입력]\n" + last_user_text
            )
        else:
            full_prompt = last_user_text or ""

        _gv.configure(api_key=api_key)
        _model_kwargs: dict = {}
        if system_prompt:
            _model_kwargs["system_instruction"] = system_prompt
        model_obj = _gv.GenerativeModel("gemini-2.5-flash", **_model_kwargs)
        resp = model_obj.generate_content(
            full_prompt,
            generation_config={"max_output_tokens": int(max_tokens)},
        )
        content = ""
        try:
            content = (resp.text or "").strip()
        except Exception:
            cands = getattr(resp, "candidates", []) or []
            if cands:
                parts = getattr(cands[0].content, "parts", []) or []
                content = "".join(getattr(p, "text", "") for p in parts).strip()
        if not content:
            return None

        return {
            "content": content,
            "credits_remaining": _UNLIMITED_CREDITS,
            "credits_used": 0,
            "tokens_in": int(getattr(getattr(resp, "usage_metadata", None), "prompt_token_count", 0) or 0),
            "tokens_out": int(getattr(getattr(resp, "usage_metadata", None), "candidates_token_count", 0) or 0),
            "_local": True,
        }
    except Exception as _e:
        print(f"  ⚠️ [Credits/Local] 직접 호출 실패 — 서버 폴백: {type(_e).__name__}: {str(_e)[:120]}")
        return None


def call_ai(
    messages: list,
    action: str = "chat",
    model: str = "gemini-2.0-flash",
    max_tokens: int = 8192,
    system_prompt: Optional[str] = None,
) -> dict:
    """
    AI 호출 메인 함수 (하이브리드).
    1) 로컬 Gemini 키 있으면 → 직접 호출
    2) 키 없거나 실패 → 서버 프록시 (응답의 credits_remaining 무시·항상 99999 강제)

    반환: {"content": str, "credits_remaining": 99999, "credits_used": 0}
    예외: AIError (네트워크/API 오류만). CreditError/ProRequiredError 더 이상 raise 안 됨.
    """
    # ── [하이브리드] 로컬 키 우선 ───────────────────────────────
    _local = _try_local_gemini_call(messages, system_prompt, max_tokens)
    if _local is not None:
        print(f"  💎 [Credits/Local] 직접 호출 성공 ({len(_local['content'])}자, "
              f"in={_local.get('tokens_in', 0)} out={_local.get('tokens_out', 0)})")
        return _local
    # ────────────────────────────────────────────────────────────

    device_id = get_device_id()

    _RETRY_STATUS = {429, 502, 503, 504}
    _MAX_TRIES = 3
    try:
        resp: requests.Response = requests.Response()
        for _attempt in range(_MAX_TRIES):
            resp = requests.post(
                f"{SERVER_URL}/api/chat",
                json={
                    "device_id": device_id,
                    "action": action,
                    "messages": messages,
                    "model": model,
                    "max_tokens": max_tokens,
                    "system_prompt": system_prompt,
                },
                timeout=120,
            )
            _body_snippet = (resp.text or "")[:300]
            _rate_limited = (
                resp.status_code in _RETRY_STATUS
                or ("429" in _body_snippet and "Resource exhausted" in _body_snippet)
            )
            if not _rate_limited or _attempt == _MAX_TRIES - 1:
                break
            _backoff = 1.5 * (2 ** _attempt)
            print(f"⚠️ [Credits] {resp.status_code} 재시도 {_attempt+1}/{_MAX_TRIES-1} (backoff {_backoff:.1f}s)")
            time.sleep(_backoff)

        # [차감 제거] 402/403 더 이상 차단 X — 사용자 무한 모드 정신.
        # 서버가 크레딧 부족/Pro 게이팅 응답 줘도 AIError 일반 메시지로 변환.
        if resp.status_code in (402, 403):
            raise AIError(
                f"서버 응답 ({resp.status_code}) — eidos_settings.json 의 gemini_api_key 확인 권장: "
                f"{resp.text[:200]}"
            )

        if resp.status_code != 200:
            if resp.status_code in _RETRY_STATUS:
                try:
                    from eidos_llm_throttle import get_throttle as _get_throttle
                    _get_throttle().on_quota_error(source=f"credit_client_{resp.status_code}")
                except Exception:
                    pass
            raise AIError(f"서버 오류 ({resp.status_code}): {resp.text[:200]}")

        data = resp.json()

        # [차감 제거] 응답의 credits_remaining 무시·항상 무한 잔액 강제.
        # 로컬 캐시 갱신도 안 함 (_save_local_cache no-op).
        data["credits_remaining"] = _UNLIMITED_CREDITS
        data["credits_used"] = 0

        return data

    except (CreditError, ProRequiredError, AIError):
        raise
    except requests.exceptions.ConnectionError:
        raise AIError("서버에 연결할 수 없습니다. 인터넷 연결을 확인하세요.")
    except requests.exceptions.Timeout:
        raise AIError("서버 응답 시간 초과. 잠시 후 다시 시도하세요.")
    except Exception as e:
        raise AIError(f"알 수 없는 오류: {e}")


def activate_license(key: str) -> dict:
    """[차감 제거] 라이선스 키 무관 항상 성공 처리·무한 잔액 반환."""
    return {
        "success": True,
        "credits_added": 0,
        "credits_total": _UNLIMITED_CREDITS,
        "tier": _UNLIMITED_TIER,
        "message": "크레딧 차감 시스템 제거됨 — 무한 모드 활성 (라이선스 키 무관).",
    }


# ── 예외 클래스 ───────────────────────────────────────────
# [차감 제거] CreditError/ProRequiredError 는 후방호환 위해 클래스 유지.
# 외부 코드가 except CreditError 패턴 사용 — raise 더 이상 안 됨이지만 import 깨지 X.

class CreditError(Exception):
    """[Deprecated 2026-05-25] 크레딧 부족. 차감 시스템 제거 후 더 이상 raise 안 됨."""
    pass

class ProRequiredError(Exception):
    """[Deprecated 2026-05-25] Pro 전용 기능. 차감 시스템 제거 후 더 이상 raise 안 됨."""
    pass

class AIError(Exception):
    """AI 호출 실패 (네트워크/API 오류). 크레딧 무관 일반 에러."""
    pass
