# eidos_agent_dispatch.py
# [2026-05-26 Phase 1-B] ToM 에이전트 — yolo 모드 외부효과 실 dispatch.
#
# agent_runner.execute_eidos_action 의 yolo 모드 분기에서 호출.
# eidos.tool.* action_id → 옛 eidos_canvas_runner._exec_* 함수 매핑·재사용.
#
# 옛 canvas_runner 는 모든 verb 함수가 (node: dict, variables: dict) → dict 통일.
# 본 모듈은 agent decision dict → node dict 어댑터 + 매핑 + variables 처리.
#
# 안전: 본 모듈은 실 외부효과 발생 path. safety_mode="yolo" 일 때만 agent_runner 가 호출.
# 잘못된 호출 방지 위해 require_yolo=True 체크 (기본).

from __future__ import annotations

from typing import Any, Optional


# ── action_id → executor_hint (legacy verb id) 매핑 ───────────────────
# eidos.tool.* 의 executor_hint 와 1:1. action_registry.DEFAULT_GLOBALS 와 동기.
_ACTION_TO_VERB: dict[str, str] = {
    "eidos.tool.web.navigate":      "NAVIGATE",
    "eidos.tool.web.click":         "CLICK",
    "eidos.tool.web.click_xy":      "CLICK_XY",
    "eidos.tool.web.fill":          "FILL",
    "eidos.tool.web.scroll":        "SCROLL",
    "eidos.tool.web.read_page":     "READ_PAGE",
    "eidos.tool.web.search":        "WEB_SEARCH",
    "eidos.tool.web.go_back":       "GO_BACK",
    "eidos.tool.web.wait":          "WAIT",
    "eidos.tool.llm.write":         "WRITE",
    "eidos.tool.message.telegram":  "TELEGRAM_SEND",
    "eidos.tool.message.email":     "EMAIL_SEND",
    "eidos.tool.ask_user":          "ASK_USER",
    # [2026-05-27 자율 개발 모드] subprocess 실행 — 외부효과·PROMPT 게이트 후 dispatch
    "eidos.tool.dev.run_tests":     "DEV_RUN_TESTS",
}


def supported_action_ids() -> list[str]:
    """본 dispatch 가 실 실행 가능한 action_id 목록."""
    return sorted(_ACTION_TO_VERB.keys())


def is_dispatchable(action_id: str) -> bool:
    return action_id in _ACTION_TO_VERB


# ── decision → node dict 어댑터 ──────────────────────────────────────
def _decision_to_node(decision: dict, action_id: str) -> dict:
    """agent decision dict → canvas_runner 의 node dict 변환.

    decision: {action_id, args: {target, content, reason}, expected_outcome, ...}
    node:     {id, verb, target, content, reason, x, y, label_override}
    """
    args = decision.get("args") or {}
    verb = _ACTION_TO_VERB.get(action_id, "?")
    # args 에 target/content/reason 표준 키가 있을 수도·없을 수도
    target = str(args.get("target") or args.get("url") or args.get("selector")
                 or args.get("recipient") or args.get("path") or "")
    content = str(args.get("content") or args.get("text") or args.get("body")
                  or args.get("brief") or args.get("question") or "")
    reason = str(decision.get("expected_outcome", "") or args.get("reason", ""))
    return {
        "id": f"dispatch_node_{action_id.replace('.', '_')}",
        "verb": verb,
        "target": target,
        "content": content,
        "reason": reason,
        "x": 0, "y": 0,
        "label_override": "",
    }


# ── [2026-06-01] SerpAPI 웹 검색 executor ────────────────────────────
# canvas_runner 에 _exec_search 가 없으므로 dispatch 가 직접 처리. 자율 에이전트가
# google 직접 navigate(CAPTCHA) 대신 정식 SerpAPI 검색을 쓰게 함.
# execution_module._serpapi_search 재사용 (음성/autonomous_plan 과 동일 경로).
async def _exec_web_search(decision: dict, node: dict) -> dict:
    args = decision.get("args") or {}
    query = str(
        args.get("query") or args.get("q")
        or node.get("content") or node.get("target") or ""
    ).strip()
    if not query:
        return {"status": "fail", "result": "검색어(query) 비어있음"}
    try:
        from execution_module import _serpapi_search, _load_serpapi_key
    except Exception as e:
        return {"status": "fail",
                "result": f"execution_module import 실패: {type(e).__name__}: {e}"}
    if not _load_serpapi_key():
        return {"status": "skip",
                "result": ("SerpAPI 키 미설정 — 설정창에서 SerpAPI 키 등록 필요 "
                           "(serpapi.com 무료 100/월)")}
    try:
        results = await _serpapi_search(query, num_results=5)
    except Exception as e:
        return {"status": "fail",
                "result": f"SerpAPI 검색 예외: {type(e).__name__}: {e}"}
    if not results:
        return {"status": "ok", "result": f"🔎 '{query}' 검색 결과 없음"}
    lines = [f"🔎 웹 검색 '{query}' — {len(results)}건"]
    for i, rr in enumerate(results, 1):
        t = (rr.get("source_title") or "출처 없음").strip()
        ln = (rr.get("link") or "").strip()
        sn = (rr.get("snippet") or "").strip()
        lines.append(f"{i}. {t}\n   {ln}\n   {sn}")
    return {"status": "ok", "result": "\n".join(lines)}


# ── 메인 dispatch ────────────────────────────────────────────────────
async def dispatch_eidos_tool_action_async(
    decision: dict,
    variables: Optional[dict] = None,
    on_ask_user: Optional[Any] = None,
    require_yolo: bool = True,
) -> dict:
    """[Phase 1-B] yolo 모드 외부효과 실 dispatch.

    Args:
        decision: agent_runner 의 decision dict
        variables: 옛 canvas_runner 의 {{var}} 치환용 dict (없으면 빈)
        on_ask_user: ASK_USER 시 main thread 위임 callback
        require_yolo: True 면 호출자가 yolo 모드 명시적 선언했음을 의미 (방어막)

    Returns: agent_runner 의 execute_eidos_action 과 호환 — {action_id, status, result, dry_run}
    """
    action_id = (decision.get("action_id") or "").strip()
    if not action_id:
        return {"action_id": "", "status": "fail",
                "result": "decision.action_id 비어있음", "dry_run": False}

    if action_id not in _ACTION_TO_VERB:
        return {"action_id": action_id, "status": "skip",
                "result": f"본 dispatch 가 지원 안 함 ({action_id})", "dry_run": False}

    if not require_yolo:
        # 보호망 — 명시적 yolo 선언 없으면 dry-run 으로 묶어둠
        return {"action_id": action_id, "status": "dry_run",
                "result": "require_yolo=False — 실 dispatch 차단", "dry_run": True}

    variables = variables or {}
    node = _decision_to_node(decision, action_id)
    verb = node["verb"]

    # 옛 canvas_runner 의 _exec_* 함수 호출
    try:
        from eidos_canvas_runner import (
            _exec_navigate, _exec_click, _exec_click_xy, _exec_fill,
            _exec_scroll, _exec_read_page, _exec_write, _exec_wait,
            _exec_go_back, _exec_ask_user, _exec_telegram_send, _exec_email_send,
        )
    except Exception as e:
        return {"action_id": action_id, "status": "fail",
                "result": f"canvas_runner import 실패: {type(e).__name__}: {e}",
                "dry_run": False}

    # [2026-06-02 Phase5] 기존 eidos_dev_actions 폐기 — DEV_RUN_TESTS 는 새 엔진(eidos_dev_engine).
    try:
        from eidos_dev_engine.agent_adapter import _exec_run_tests as _exec_dev_run_tests
    except Exception as _e_eng:
        print(f"⚠️ [agent_dispatch] dev run_tests 엔진 import 실패 (미지원·graceful): {_e_eng}")
        _exec_dev_run_tests = None  # type: ignore

    verb_map = {
        "NAVIGATE":      _exec_navigate,
        "CLICK":         _exec_click,
        "CLICK_XY":      _exec_click_xy,
        "FILL":          _exec_fill,
        "SCROLL":        _exec_scroll,
        "READ_PAGE":     _exec_read_page,
        "WRITE":         _exec_write,
        "WAIT":          _exec_wait,
        "GO_BACK":       _exec_go_back,
        "TELEGRAM_SEND": _exec_telegram_send,
        "EMAIL_SEND":    _exec_email_send,
    }
    if _exec_dev_run_tests is not None:
        verb_map["DEV_RUN_TESTS"] = _exec_dev_run_tests

    try:
        if verb == "WEB_SEARCH":
            r = await _exec_web_search(decision, node)
        elif verb == "ASK_USER":
            # _exec_ask_user 는 on_ask_user 따로 받음
            r = await _exec_ask_user(node, variables, on_ask_user=on_ask_user)
        elif verb in verb_map:
            r = await verb_map[verb](node, variables)
        else:
            return {"action_id": action_id, "status": "skip",
                    "result": f"verb 매핑 안 됨 ({verb})", "dry_run": False}
    except Exception as e:
        return {"action_id": action_id, "status": "fail",
                "result": f"실행 중 예외: {type(e).__name__}: {e}", "dry_run": False}

    # canvas_runner result {status, result} → agent_runner 표준 추가
    return {
        "action_id": action_id,
        "status": r.get("status", "fail"),
        "result": r.get("result", "(빈 결과)"),
        "dry_run": False,
        "_verb": verb,
        "_node_used": {k: node.get(k) for k in ("target", "content", "reason")},
    }


__all__ = [
    "dispatch_eidos_tool_action_async",
    "supported_action_ids",
    "is_dispatchable",
]
