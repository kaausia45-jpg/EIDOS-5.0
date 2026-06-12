# eidos_canvas_runner.py
# 액션 조립 캔버스 직접 실행기 (v2.1) — ActionDispatcher Vision 루프 우회.
# execution_module 의 verb 별 함수 직접 호출.
#
# 캔버스 노드 그래프 → topology sort → 순서대로 실행 → 결과 list 반환.
# 각 step 시작/완료 시 on_progress callback (UI 노드 색상 update 용).
#
# 변수 흐름: READ_PAGE / WRITE 가 target 을 변수명으로 쓰면 dict 에 저장.
# 후속 FILL / WRITE 의 content 에서 {{변수명}} 으로 치환.

from __future__ import annotations

import asyncio
import os
import re
from typing import Awaitable, Callable, Optional

import eidos_canvas_store as _cstore


# step 상태
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_OK      = "ok"
STATUS_FAIL    = "fail"
STATUS_SKIP    = "skip"
STATUS_WARN    = "warn"
STATUS_STOPPED = "stopped"   # v2.2 stop button — 사용자 중단


# on_progress callback: (event: str, node_id: str, payload: dict) -> Awaitable | None
# event ∈ {"start", "done", "all_done"}
ProgressFn = Optional[Callable[[str, str, dict], Optional[Awaitable[None]]]]


def _substitute_vars(text: str, variables: dict) -> str:
    """{{varname}} 패턴을 variables[varname] 값으로 치환."""
    if not text or not variables:
        return text or ""
    def _repl(m):
        key = m.group(1).strip()
        if key in variables:
            return str(variables[key])
        return m.group(0)   # 원형 유지
    try:
        return re.sub(r"\{\{\s*([\w가-힣_]+)\s*\}\}", _repl, text)
    except Exception:
        return text


def _js_escape(s: str) -> str:
    """JS 문자열 리터럴 안전 escape (single-quote 컨텍스트)."""
    return (
        s.replace("\\", "\\\\")
         .replace("'", "\\'")
         .replace("\n", "\\n")
         .replace("\r", "")
    )


async def _emit_progress(on_progress: ProgressFn, event: str, node_id: str, payload: dict) -> None:
    """callback 호출 — sync/async 모두 graceful 지원."""
    if on_progress is None:
        return
    try:
        result = on_progress(event, node_id, payload)
        if asyncio.iscoroutine(result):
            await result
    except Exception as e:
        print(f"[canvas_runner] on_progress {event} {node_id} 실패 (graceful): {e}")


# ── verb 별 실행 ────────────────────────────────────────────────────────

async def _exec_navigate(node: dict, variables: dict) -> dict:
    import execution_module as em
    url = _substitute_vars(node.get("target", ""), variables).strip()
    if not url:
        return {"status": STATUS_FAIL, "result": "target (URL) 비어있음"}
    try:
        r = await em.navigate_and_wait(url, timeout=15.0)
        return {"status": STATUS_OK, "result": f"navigated → {url}  ({(r or '')[:60]})"}
    except Exception as e:
        return {"status": STATUS_FAIL, "result": f"{type(e).__name__}: {e}"}


async def _exec_click(node: dict, variables: dict) -> dict:
    import execution_module as em
    sel = _substitute_vars(node.get("target", ""), variables).strip()
    if not sel:
        return {"status": STATUS_FAIL, "result": "셀렉터 비어있음"}
    # [N] 번호 패턴은 단순 click 불가 — 셀렉터만 지원 (Vision 모드는 ActionDispatcher 측)
    if sel.startswith("[") and sel.endswith("]"):
        return {"status": STATUS_SKIP, "result": f"[N] 번호 클릭은 Vision 모드 전용 - 셀렉터로 변경 권장: {sel}"}
    js_sel = _js_escape(sel)
    script = (
        f"(function(){{ "
        f"const el = document.querySelector('{js_sel}'); "
        f"if (el) {{ el.click(); return 'clicked'; }} else {{ return 'not_found'; }} "
        f"}})()"
    )
    try:
        r = await em.execute_js_with_result(script, timeout=5.0)
        if "clicked" in str(r):
            return {"status": STATUS_OK, "result": f"clicked {sel}"}
        return {"status": STATUS_WARN, "result": f"요소 못 찾음: {sel}"}
    except Exception as e:
        return {"status": STATUS_FAIL, "result": f"{type(e).__name__}: {e}"}


async def _exec_click_xy(node: dict, variables: dict) -> dict:
    import execution_module as em
    raw = _substitute_vars(node.get("target", ""), variables).strip()
    try:
        x_str, y_str = raw.split(",")
        x, y = int(x_str.strip()), int(y_str.strip())
    except Exception:
        return {"status": STATUS_FAIL, "result": f"좌표 형식 잘못: {raw} (예: 430,520)"}
    script = (
        f"(function(){{ "
        f"const el = document.elementFromPoint({x}, {y}); "
        f"if (el) {{ el.click(); return 'clicked@'+{x}+','+{y}; }} else {{ return 'no_element'; }} "
        f"}})()"
    )
    try:
        r = await em.execute_js_with_result(script, timeout=5.0)
        if "clicked" in str(r):
            return {"status": STATUS_OK, "result": str(r)}
        return {"status": STATUS_WARN, "result": f"좌표에 요소 없음: ({x},{y})"}
    except Exception as e:
        return {"status": STATUS_FAIL, "result": f"{type(e).__name__}: {e}"}


async def _exec_fill(node: dict, variables: dict) -> dict:
    import execution_module as em
    sel = _substitute_vars(node.get("target", ""), variables).strip()
    text = _substitute_vars(node.get("content", ""), variables)
    if not sel:
        return {"status": STATUS_FAIL, "result": "셀렉터 비어있음"}
    if sel.startswith("[") and sel.endswith("]"):
        return {"status": STATUS_SKIP, "result": f"[N] 번호 FILL 은 Vision 모드 전용: {sel}"}
    js_sel = _js_escape(sel)
    js_text = _js_escape(text)
    script = (
        f"(function(){{ "
        f"const el = document.querySelector('{js_sel}'); "
        f"if (!el) return 'not_found'; "
        f"el.focus(); "
        f"el.value = '{js_text}'; "
        f"el.dispatchEvent(new Event('input', {{bubbles:true}})); "
        f"el.dispatchEvent(new Event('change', {{bubbles:true}})); "
        f"return 'filled'; "
        f"}})()"
    )
    try:
        r = await em.execute_js_with_result(script, timeout=5.0)
        if "filled" in str(r):
            return {"status": STATUS_OK, "result": f"filled {sel} ({len(text)}자)"}
        return {"status": STATUS_WARN, "result": f"요소 못 찾음: {sel}"}
    except Exception as e:
        return {"status": STATUS_FAIL, "result": f"{type(e).__name__}: {e}"}


async def _exec_scroll(node: dict, variables: dict) -> dict:
    import execution_module as em
    raw = _substitute_vars(node.get("target", ""), variables).strip().lower()
    if not raw:
        script = "(function(){ window.scrollBy(0, window.innerHeight * 0.8); return 'scrolled_80vh'; })()"
    elif raw == "bottom":
        script = "(function(){ window.scrollTo(0, document.body.scrollHeight); return 'scrolled_bottom'; })()"
    elif raw == "top":
        script = "(function(){ window.scrollTo(0, 0); return 'scrolled_top'; })()"
    elif raw.startswith("px:"):
        try:
            px = int(raw[3:].strip())
        except Exception:
            return {"status": STATUS_FAIL, "result": f"px 값 잘못: {raw}"}
        script = f"(function(){{ window.scrollBy(0, {px}); return 'scrolled_{px}px'; }})()"
    else:
        return {"status": STATUS_FAIL, "result": f"target 값 인식 불가: {raw} (bottom/top/px:N/빈값)"}
    try:
        r = await em.execute_js_with_result(script, timeout=3.0)
        return {"status": STATUS_OK, "result": str(r)}
    except Exception as e:
        return {"status": STATUS_FAIL, "result": f"{type(e).__name__}: {e}"}


async def _exec_read_page(node: dict, variables: dict) -> dict:
    import execution_module as em
    var_name = _substitute_vars(node.get("target", ""), variables).strip() or "page_text"
    try:
        txt = await em.get_browser_visible_text(timeout=6.0)
        txt = (txt or "").strip()
        variables[var_name] = txt
        return {"status": STATUS_OK, "result": f"{var_name} ← {len(txt)}자 ({txt[:60]}...)"}
    except Exception as e:
        return {"status": STATUS_FAIL, "result": f"{type(e).__name__}: {e}"}


async def _exec_write(node: dict, variables: dict) -> dict:
    """WRITE — LLM 으로 텍스트 생성 후 target (파일 경로 또는 변수명) 에 저장."""
    try:
        from llm_module import get_llm_response_async
    except Exception as e:
        return {"status": STATUS_FAIL, "result": f"llm_module 부재: {e}"}
    brief = _substitute_vars(node.get("content", ""), variables).strip()
    target = _substitute_vars(node.get("target", ""), variables).strip()
    if not brief:
        return {"status": STATUS_FAIL, "result": "content (브리프) 비어있음"}
    try:
        resp = await get_llm_response_async(brief, max_tokens=8192, timeout=120)
        resp = (resp or "").strip()
        if not resp or resp.startswith("[서버") or resp.startswith("LLM 오류"):
            return {"status": STATUS_FAIL, "result": f"LLM 실패: {resp[:120]}"}
        if target and (target.endswith(".md") or target.endswith(".txt") or "/" in target or "\\" in target):
            # 파일 저장
            try:
                # eidos_files 안 상대 경로로 안전화
                if not os.path.isabs(target):
                    out_dir = os.path.join("eidos_files", "canvas_writes")
                    os.makedirs(out_dir, exist_ok=True)
                    path = os.path.join(out_dir, target)
                else:
                    path = target
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(resp)
                # 같은 이름 변수에도 저장
                variables[target] = resp
                return {"status": STATUS_OK, "result": f"saved {path} ({len(resp)}자)"}
            except Exception as e:
                return {"status": STATUS_FAIL, "result": f"파일 저장 실패: {e}"}
        elif target:
            # 변수 저장
            variables[target] = resp
            return {"status": STATUS_OK, "result": f"{target} ← {len(resp)}자"}
        else:
            return {"status": STATUS_OK, "result": resp[:200]}
    except Exception as e:
        return {"status": STATUS_FAIL, "result": f"{type(e).__name__}: {e}"}


async def _exec_wait(node: dict, variables: dict) -> dict:
    raw = _substitute_vars(node.get("target", ""), variables).strip()
    try:
        secs = float(raw) if raw else 1.0
        secs = max(0.0, min(secs, 300.0))   # 0 ~ 5 분 clamp
    except Exception:
        return {"status": STATUS_FAIL, "result": f"초 값 잘못: {raw}"}
    await asyncio.sleep(secs)
    return {"status": STATUS_OK, "result": f"waited {secs}s"}


async def _exec_go_back(node: dict, variables: dict) -> dict:
    import execution_module as em
    try:
        r = await em.execute_js_with_result("(function(){ window.history.back(); return 'back'; })()", timeout=3.0)
        return {"status": STATUS_OK, "result": str(r)}
    except Exception as e:
        return {"status": STATUS_FAIL, "result": f"{type(e).__name__}: {e}"}


async def _exec_ask_user(node: dict, variables: dict, on_ask_user=None) -> dict:
    """ASK_USER (v2.2) — on_ask_user callable 주입 시 main thread UI 다이얼로그 호출.
    on_ask_user(question) -> awaitable[str]. 없으면 v2.1 stub 처럼 skip."""
    question = _substitute_vars(node.get("content", ""), variables).strip() or "(질문 비어있음)"
    var_name = _substitute_vars(node.get("target", ""), variables).strip()
    if on_ask_user is None:
        if var_name:
            variables[var_name] = f"(ASK_USER stub: {question})"
        return {"status": STATUS_SKIP,
                "result": f"ASK_USER stub (UI 미주입) - 질문 '{question[:60]}'"}
    try:
        answer = await on_ask_user(question)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        return {"status": STATUS_FAIL, "result": f"ASK_USER 실패: {type(e).__name__}: {e}"}
    answer_str = "" if answer is None else str(answer)
    if var_name:
        variables[var_name] = answer_str
    return {"status": STATUS_OK,
            "result": f"{var_name or '(answer)'} ← '{answer_str[:60]}'"}


async def _exec_telegram_send(node: dict, variables: dict) -> dict:
    title = _substitute_vars(node.get("target", ""), variables).strip() or "🚀 EIDOS"
    body = _substitute_vars(node.get("content", ""), variables).strip()
    if not body:
        return {"status": STATUS_FAIL, "result": "content (메시지 본문) 비어있음"}
    body = body[:4096]   # 텔레그램 4096자 한도
    try:
        # 다양한 텔레그램 모듈 후보 — graceful 시도
        send_fn = None
        for mod_name in ("eidos_telegram_module", "eidos_telegram", "telegram_module"):
            try:
                mod = __import__(mod_name)
                for fn_name in ("send_telegram_message", "send_message", "send"):
                    fn = getattr(mod, fn_name, None)
                    if callable(fn):
                        send_fn = fn
                        break
                if send_fn:
                    break
            except Exception:
                continue
        if send_fn is None:
            return {"status": STATUS_SKIP, "result": "텔레그램 모듈 부재 (v2.x wire 예정)"}
        msg = f"{title}\n\n{body}" if title else body
        r = send_fn(msg) if not asyncio.iscoroutinefunction(send_fn) else await send_fn(msg)
        return {"status": STATUS_OK, "result": f"telegram sent ({len(msg)}자)"}
    except Exception as e:
        return {"status": STATUS_FAIL, "result": f"{type(e).__name__}: {e}"}


async def _exec_email_send(node: dict, variables: dict) -> dict:
    """EMAIL_SEND — SMTP 발송. eidos_files/email_config.json 미설정 시 graceful skip."""
    to = _substitute_vars(node.get("target", ""), variables).strip()
    body = _substitute_vars(node.get("content", ""), variables).strip()
    if "@" not in to:
        return {"status": STATUS_FAIL, "result": f"받는 사람 이메일 형식 잘못: {to}"}
    if not body:
        return {"status": STATUS_FAIL, "result": "content 비어있음"}
    try:
        # SMTP 모듈 후보
        send_fn = None
        for mod_name in ("eidos_email_module", "eidos_smtp", "email_module"):
            try:
                mod = __import__(mod_name)
                for fn_name in ("send_email", "send"):
                    fn = getattr(mod, fn_name, None)
                    if callable(fn):
                        send_fn = fn
                        break
                if send_fn:
                    break
            except Exception:
                continue
        if send_fn is None:
            return {"status": STATUS_SKIP, "result": "이메일 모듈 부재 (v2.x wire 예정)"}
        # body 첫 줄이 "Subject: ..." 면 제목 추출
        subject = "🚀 EIDOS"
        body_only = body
        if body.startswith("Subject:"):
            lines = body.split("\n", 1)
            subject = lines[0][len("Subject:"):].strip()
            body_only = lines[1].lstrip() if len(lines) > 1 else ""
        r = send_fn(to, subject, body_only) if not asyncio.iscoroutinefunction(send_fn) \
            else await send_fn(to, subject, body_only)
        return {"status": STATUS_OK, "result": f"email sent → {to}"}
    except Exception as e:
        return {"status": STATUS_FAIL, "result": f"{type(e).__name__}: {e}"}


# verb id → exec function
_DISPATCH = {
    "NAVIGATE":       _exec_navigate,
    "CLICK":          _exec_click,
    "CLICK_XY":       _exec_click_xy,
    "FILL":           _exec_fill,
    "SCROLL":         _exec_scroll,
    "READ_PAGE":      _exec_read_page,
    "WRITE":          _exec_write,
    "WAIT":           _exec_wait,
    "ASK_USER":       _exec_ask_user,
    "GO_BACK":        _exec_go_back,
    "TELEGRAM_SEND":  _exec_telegram_send,
    "EMAIL_SEND":     _exec_email_send,
}


async def execute_canvas_async(
    canvas: dict,
    on_progress: ProgressFn = None,
    ctx: Optional[dict] = None,
    stop_on_fail: bool = False,
) -> dict:
    """캔버스 노드 그래프 → topology sort → 순서대로 실행.

    on_progress(event, node_id, payload): event ∈ {start, done, all_done}.
      • start payload: {step_index, total_steps, verb, target}
      • done payload: {status, result, variables_count, ...}
      • all_done payload: {total, ok, fail, skip, stopped, results, variables}

    ctx (v2.2): {
        "on_ask_user": async callable(question) -> str (ASK_USER UI 다이얼로그 위임),
        "stop_event": asyncio.Event (main thread 에서 set 시 남은 step 중단),
    }

    stop_on_fail True 면 첫 실패에서 중단. False 면 best-effort.
    """
    ctx = ctx or {}
    on_ask_user = ctx.get("on_ask_user")
    stop_event = ctx.get("stop_event")

    # [2026-05-26 Phase 8-B] belief 에 canvas 실행 시작 기록 — pending_thread 등록.
    # graceful — belief_integrations 실패해도 canvas 실행 영향 X.
    _canvas_name = canvas.get("name") or "(이름없음)"
    try:
        from eidos_belief_integrations import register_canvas_run_start as _bi_canvas_start
        _bi_canvas_start(_canvas_name, description=canvas.get("description", "")[:80])
    except Exception as _e_bi:
        print(f"[canvas_runner] belief start hook 실패 (graceful): {_e_bi}")

    ordered = _cstore.topology_sort(canvas)
    variables: dict = {}
    results: list[dict] = []
    total = len(ordered)
    ok_count = fail_count = skip_count = stopped_count = 0
    stopped_early = False

    for i, node in enumerate(ordered):
        nid = node.get("id", f"node_{i}")
        verb = node.get("verb", "")
        target = node.get("target", "")

        # v2.2 stop check — step 시작 전
        if stop_event is not None and stop_event.is_set():
            stopped_early = True
            # 남은 모든 노드 stopped 마킹
            for j in range(i, total):
                sk = ordered[j]
                sk_id = sk.get("id", f"node_{j}")
                stop_r = {
                    "node_id": sk_id,
                    "verb": sk.get("verb", ""),
                    "step_index": j,
                    "status": STATUS_STOPPED,
                    "result": "사용자 중단",
                }
                results.append(stop_r)
                stopped_count += 1
                await _emit_progress(on_progress, "done", sk_id,
                                     {"status": STATUS_STOPPED, "result": "사용자 중단",
                                      "variables_count": len(variables)})
            break

        await _emit_progress(on_progress, "start", nid,
                             {"step_index": i, "total_steps": total, "verb": verb, "target": target})

        # ASK_USER 만 특수 시그니처 (on_ask_user 주입)
        try:
            if verb == "ASK_USER":
                r = await _exec_ask_user(node, variables, on_ask_user=on_ask_user)
            else:
                fn = _DISPATCH.get(verb)
                if fn is None:
                    r = {"status": STATUS_SKIP, "result": f"unknown verb: {verb}"}
                else:
                    r = await fn(node, variables)
        except asyncio.CancelledError:
            # 사용자 stop / task cancel — stopped 로 마킹하고 남은 노드도 stopped
            stopped_early = True
            r = {"status": STATUS_STOPPED, "result": "task cancelled"}
            r["node_id"] = nid; r["verb"] = verb; r["step_index"] = i
            results.append(r)
            stopped_count += 1
            await _emit_progress(on_progress, "done", nid,
                                 {"status": STATUS_STOPPED, "result": "task cancelled",
                                  "variables_count": len(variables)})
            for j in range(i + 1, total):
                sk = ordered[j]
                sk_id = sk.get("id", f"node_{j}")
                stop_r = {"node_id": sk_id, "verb": sk.get("verb", ""), "step_index": j,
                          "status": STATUS_STOPPED, "result": "cancelled"}
                results.append(stop_r)
                stopped_count += 1
                await _emit_progress(on_progress, "done", sk_id,
                                     {"status": STATUS_STOPPED, "result": "cancelled",
                                      "variables_count": len(variables)})
            break
        except Exception as e:
            r = {"status": STATUS_FAIL, "result": f"runner exception: {type(e).__name__}: {e}"}

        r["node_id"] = nid
        r["verb"] = verb
        r["step_index"] = i
        results.append(r)
        s = r.get("status", STATUS_FAIL)
        if s == STATUS_OK:
            ok_count += 1
        elif s == STATUS_FAIL:
            fail_count += 1
        elif s in (STATUS_SKIP, STATUS_WARN):
            skip_count += 1
        elif s == STATUS_STOPPED:
            stopped_count += 1
        await _emit_progress(on_progress, "done", nid,
                             {"status": s, "result": r.get("result", ""),
                              "variables_count": len(variables)})
        if stop_on_fail and s == STATUS_FAIL:
            break

    summary = {
        "total": total,
        "ok": ok_count,
        "fail": fail_count,
        "skip": skip_count,
        "stopped": stopped_count,
        "stopped_early": stopped_early,
        "results": results,
        "variables": variables,
    }
    # [2026-05-26 Phase 8-B] belief 에 canvas 실행 완료 기록.
    # 성공 기준: 실패·중단 0 + ok≥1. 부분 성공도 success 로 (사용자 의도 X 실행).
    try:
        from eidos_belief_integrations import register_canvas_run_done as _bi_canvas_done
        _success = (fail_count == 0 and stopped_count == 0 and ok_count >= 1)
        _bi_canvas_done(
            _canvas_name,
            success=_success,
            summary=f"ok={ok_count}·fail={fail_count}·skip={skip_count}",
        )
    except Exception as _e_bid:
        print(f"[canvas_runner] belief done hook 실패 (graceful): {_e_bid}")
    await _emit_progress(on_progress, "all_done", "", summary)
    return summary
