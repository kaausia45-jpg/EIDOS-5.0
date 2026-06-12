# eidos_chat_commands.py
# [2026-05-27 Phase 9-F] 채팅창 slash command — UI 없이 자연어로 setup.
#
# 사용자가 햄버거 → 다이얼로그 → 클릭 노가다 대신 채팅창에서 바로:
#   /goal 내년 매출 5000만        — Stage.goal 설정 + GoalTree root sync
#   /decompose [N]               — 활성 stage 의 root/선택 milestone 자동 분해
#   /goals                       — 현재 트리 brief
#   /done <text>                 — 매칭 milestone 완료 (substring)
#   /monitor <url>               — url_monitor source 추가
#   /sources                     — 등록된 source 목록
#   /unmonitor <id_or_name>      — source 삭제
#   /good [reason]               — 직전 EIDOS action 에 +0.8 reward
#   /bad [reason]                — 직전 action 에 -0.5 reward
#   /help                        — 명령 list
#   /status                      — belief + goals + sources 한방 요약
#
# 디자인 원칙 (Phase 1~13 동일):
#   1. LLM 호출 0 — /decompose 만 LLM (Phase 9-E 재사용).
#   2. graceful — 명령 실패해도 채팅 흐름 영향 X.
#   3. 활성 stage 가 필요한 명령은 미첨부 시 안내 후 종료.
#   4. 단일 진입점 try_handle_command — chat_gui 의 send_message 초입에 1줄 hook.
#   5. 결과는 system message — LLM 호출 skip·즉시 응답.

from __future__ import annotations

import asyncio
import datetime as _dt
import re
import shlex
from typing import Any, Callable, Optional


# ── 상수 ───────────────────────────────────────────────────────────────
_GOOD_REWARD = 0.8
_BAD_REWARD = -0.5
_DEFAULT_DECOMPOSE_N = 4

# 명령 alias (한/영 자유)
_COMMAND_ALIASES = {
    "/goal": "goal",       "/목표": "goal",
    "/decompose": "decompose", "/분해": "decompose",
    "/goals": "goals",     "/목표트리": "goals", "/tree": "goals",
    "/done": "done",       "/완료": "done",
    "/monitor": "monitor", "/모니터": "monitor", "/watch": "monitor",
    "/sources": "sources", "/소스": "sources",
    "/unmonitor": "unmonitor", "/언모니터": "unmonitor",
    "/good": "good",       "/굿": "good",       "/+": "good",
    "/bad": "bad",         "/별로": "bad",      "/-": "bad",
    "/help": "help",       "/?": "help",       "/도움": "help",
    "/status": "status",   "/상태": "status",
    # [Wave3-D 2026-05-28] 도구 catalog 관리
    "/approve_tool": "approve_tool", "/승인": "approve_tool",
    "/reject_tool": "reject_tool",   "/거부": "reject_tool",
    "/tools": "tools",               "/도구": "tools",
}

# 명령 사용법 (help 표시)
_USAGE = {
    "goal":      "/goal <목표 텍스트> — Stage.goal 설정 + GoalTree root sync",
    "decompose": "/decompose [N=4] — 활성 stage 의 root 또는 선택 milestone 을 LLM 으로 N 개 하위 분해",
    "goals":     "/goals — 현재 목표 트리 brief 표시",
    "done":      "/done <substring> — 제목/설명에 매칭되는 milestone 완료 처리",
    "monitor":   "/monitor <url> [as <actor_name>] — url_monitor source 추가",
    "sources":   "/sources — 등록된 외부 신호 source 목록",
    "unmonitor": "/unmonitor <id_or_name 부분> — source 삭제",
    "good":      "/good [이유] — 직전 EIDOS action 에 +0.8 reward",
    "bad":       "/bad [이유] — 직전 EIDOS action 에 -0.5 reward",
    "help":      "/help — 이 명령 list 표시",
    "status":    "/status — belief + 목표 + source 한방 요약",
    # [Wave3-D 2026-05-28] 도구 catalog 관리
    "approve_tool": "/approve_tool <tool_id> — pending_approval 도구 활성화",
    "reject_tool":  "/reject_tool <tool_id> [reason] — 도구 거부",
    "tools":        "/tools [status] — 도구 catalog 조회 (status: all/active/pending/rejected/abandoned)",
}


# ── 진입점 ─────────────────────────────────────────────────────────────
def try_handle_command(
    user_text: str,
    stage_id: Optional[str] = None,
    *,
    on_message: Optional[Callable[[str, str], None]] = None,
) -> bool:
    """user_text 가 slash 명령이면 처리 + on_message(text, kind) 콜백.

    Args:
        user_text: 사용자 입력 한 줄
        stage_id: 현재 첨부된 ToM Stage id (없으면 None)
        on_message: 결과 출력 콜백 fn(text, kind) — kind ∈ {"system", "user", "assistant"}
                    None 이면 stdout 으로

    Returns:
        True: 명령 처리됨·일반 chat 흐름 차단 (return)
        False: 명령 아님·일반 chat 흐름 진행
    """
    if not user_text or not user_text.strip():
        return False
    text = user_text.strip()
    if not text.startswith("/"):
        return False
    # alias resolve
    parts = text.split(None, 1)
    head = parts[0].lower()
    if head not in _COMMAND_ALIASES:
        return False
    cmd = _COMMAND_ALIASES[head]
    args = parts[1].strip() if len(parts) > 1 else ""

    emit = on_message or (lambda t, k="system": print(t))

    # 사용자 입력 echo (slash 명령은 채팅 hist 에도 보존)
    try:
        emit(user_text, "user")
    except Exception:
        pass

    handler = _HANDLERS.get(cmd)
    if handler is None:
        emit(f"⚠️ 알 수 없는 명령: {cmd} — /help 로 list 확인", "system")
        return True

    try:
        result = handler(args, stage_id)
    except Exception as e:
        result = f"❌ {cmd} 실행 실패 (graceful): {type(e).__name__}: {e}"
    if result is None:
        result = "(명령 처리됨·결과 없음)"
    try:
        emit(result, "system")
    except Exception:
        print(result)
    return True


# ── 핸들러: /help ──────────────────────────────────────────────────────
def _cmd_help(args: str, stage_id: Optional[str]) -> str:
    lines = ["## 📖 EIDOS 채팅 명령 list (Phase 9-F)"]
    lines.append("UI 없이 채팅창에서 바로 명령. 한/영 alias 모두 OK.\n")
    for cmd, usage in _USAGE.items():
        # alias 모음
        aliases = [a for a, c in _COMMAND_ALIASES.items() if c == cmd]
        alias_str = ", ".join(sorted(aliases))
        lines.append(f"- **{alias_str}**")
        lines.append(f"  {usage}")
    lines.append("\n예시:")
    lines.append("  `/goal 내년 매출 5000만`")
    lines.append("  `/decompose 4`")
    lines.append("  `/monitor https://kmong.com/notifications as 크몽`")
    lines.append("  `/good 견적 초안 완벽함`")
    return "\n".join(lines)


# ── 핸들러: /goal ──────────────────────────────────────────────────────
def _cmd_goal(args: str, stage_id: Optional[str]) -> str:
    if not stage_id:
        return ("❌ 활성 stage 없음. 햄버거 → 📎 ToM 에이전트 첨부 로 stage 선택 후 다시.")
    if not args:
        return "ℹ️ 사용법: /goal <목표 텍스트>"
    try:
        from eidos_agent_stage_store import load_stage, save_stage
        from eidos_goal_tree import (
            load_goal_tree, save_goal_tree, new_goal_tree,
            sync_root_with_stage,
        )
    except Exception as e:
        return f"❌ import 실패: {e}"

    stage = load_stage(stage_id)
    if stage is None:
        return f"❌ stage {stage_id[:12]} 로드 실패"
    old_goal = stage.goal
    stage.goal = args.strip()[:300]
    save_stage(stage)

    tree = load_goal_tree(stage_id)
    if tree is None:
        tree = new_goal_tree(stage_id, root_title=stage.goal, horizon="year")
        save_goal_tree(tree)
        n_milestones = len(tree.milestones)
        return (
            f"✅ Stage.goal 설정 + GoalTree 신규 생성\n"
            f"- 옛 goal: {old_goal or '(없음)'}\n"
            f"- 새 goal: **{stage.goal}**\n"
            f"- root milestone 1개 자동 생성 (총 {n_milestones})\n"
            f"다음: `/decompose 4` 로 자동 분해하거나 `/goals` 로 확인."
        )
    changed = sync_root_with_stage(tree, stage)
    if changed:
        save_goal_tree(tree)
    return (
        f"✅ Stage.goal 갱신 + root milestone sync\n"
        f"- 옛 goal: {old_goal or '(없음)'}\n"
        f"- 새 goal: **{stage.goal}**\n"
        f"- 기존 milestone 보존 (총 {len(tree.milestones)})\n"
        f"다음: `/goals` 로 확인하거나 `/decompose` 로 자동 분해."
    )


# ── 핸들러: /decompose ─────────────────────────────────────────────────
def _cmd_decompose(args: str, stage_id: Optional[str]) -> str:
    if not stage_id:
        return "❌ 활성 stage 없음. 먼저 stage 첨부."
    try:
        n = int(args.strip()) if args.strip() else _DEFAULT_DECOMPOSE_N
    except ValueError:
        n = _DEFAULT_DECOMPOSE_N
    n = max(1, min(12, n))
    try:
        from eidos_goal_tree import (
            load_goal_tree, save_goal_tree,
            decompose_goal_with_llm_async, apply_decomposition,
            _HORIZON_CHILDREN,
        )
    except Exception as e:
        return f"❌ import 실패: {e}"

    tree = load_goal_tree(stage_id)
    if tree is None or not tree.root_goal_id:
        return ("❌ GoalTree 없음. 먼저 `/goal <목표>` 로 root 설정.")
    parent = tree.root()
    if parent is None:
        return "❌ root milestone 손상"

    target_horizon = _HORIZON_CHILDREN.get(parent.horizon, "task")
    try:
        try:
            proposed = asyncio.run(decompose_goal_with_llm_async(
                parent.title,
                parent_horizon=parent.horizon,
                target_horizon=target_horizon,
                n_children=n,
                horizon_end=tree.horizon_end or parent.target_date or "",
                context=parent.description or "",
            ))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                proposed = loop.run_until_complete(decompose_goal_with_llm_async(
                    parent.title,
                    parent_horizon=parent.horizon,
                    target_horizon=target_horizon,
                    n_children=n,
                    horizon_end=tree.horizon_end or parent.target_date or "",
                    context=parent.description or "",
                ))
            finally:
                loop.close()
    except Exception as e:
        return f"❌ LLM 분해 실패: {type(e).__name__}: {e}"

    if not proposed:
        return ("⚠️ LLM 이 빈 결과 반환. 다시 시도하거나 수동으로 추가하세요.\n"
                "(/help 로 명령 list 확인)")

    registered = apply_decomposition(tree, parent.id, proposed)
    if not registered:
        return "⚠️ 등록 실패. (tree 저장 실패 가능성)"
    save_goal_tree(tree)
    lines = [
        f"✅ '{parent.title}' 아래 {len(registered)} 개 {target_horizon} milestone 자동 등록",
    ]
    for i, m in enumerate(registered, 1):
        td = f" ({m.target_date})" if m.target_date else ""
        lines.append(f"  {i}. **{m.title}**{td}")
        if m.success_criteria:
            lines.append(f"     기준: {', '.join(m.success_criteria[:3])}")
    lines.append("\n다음: `/goals` 로 확인, 각 milestone 다시 `/decompose` 가능")
    return "\n".join(lines)


# ── 핸들러: /goals ─────────────────────────────────────────────────────
def _cmd_goals(args: str, stage_id: Optional[str]) -> str:
    if not stage_id:
        return "❌ 활성 stage 없음."
    try:
        from eidos_goal_tree import load_goal_tree, as_prompt_brief, summary_for_log
    except Exception as e:
        return f"❌ import 실패: {e}"
    tree = load_goal_tree(stage_id)
    if tree is None or not tree.root_goal_id:
        return ("ℹ️ GoalTree 없음. `/goal <목표>` 로 시작.")
    brief = as_prompt_brief(tree, depth=3)
    s = summary_for_log(tree)
    head = (
        f"📊 milestone 총 {s['total_milestones']}개  ·  "
        f"active {s['by_status'].get('active', 0)}  ·  "
        f"done {s['by_status'].get('done', 0)}  ·  "
        f"abandoned {s['by_status'].get('abandoned', 0)}\n"
    )
    return head + brief


# ── 핸들러: /done ──────────────────────────────────────────────────────
def _cmd_done(args: str, stage_id: Optional[str]) -> str:
    if not stage_id:
        return "❌ 활성 stage 없음."
    needle = args.strip()
    if not needle:
        return "ℹ️ 사용법: /done <milestone 제목 substring>"
    try:
        from eidos_goal_tree import load_goal_tree, save_goal_tree, set_milestone_status
    except Exception as e:
        return f"❌ import 실패: {e}"
    tree = load_goal_tree(stage_id)
    if tree is None:
        return "❌ GoalTree 없음."
    needle_l = needle.lower()
    matches = [m for m in tree.all_milestones()
               if m.status == "active"
               and (needle_l in m.title.lower() or needle_l in m.description.lower())]
    if not matches:
        return f"⚠️ 매칭 active milestone 없음 ('{needle}'·`/goals` 로 list 확인)"
    if len(matches) > 1:
        titles = "\n  ".join(f"- {m.title}" for m in matches[:5])
        return (f"⚠️ {len(matches)} 개 매칭 — 더 구체적으로 입력:\n  {titles}")
    m = matches[0]
    set_milestone_status(tree, m.id, "done")
    save_goal_tree(tree)
    return f"✅ '{m.title}' 완료 (progress=1.0·completed_at 기록)"


# ── 핸들러: /monitor ───────────────────────────────────────────────────
_URL_RE = re.compile(r"https?://[^\s]+")
_AS_RE = re.compile(r"\s+as\s+(.+)$", re.IGNORECASE)


def _cmd_monitor(args: str, stage_id: Optional[str]) -> str:
    if not stage_id:
        return ("❌ 활성 stage 없음. signal 을 적용할 stage 가 필요함.")
    if not args:
        return "ℹ️ 사용법: /monitor <url> [as <actor_name>]"
    text = args.strip()
    # "as <actor>" 분리
    actor_name = "외부 알림"
    m_as = _AS_RE.search(text)
    if m_as:
        actor_name = m_as.group(1).strip()
        text = _AS_RE.sub("", text).strip()
    m_url = _URL_RE.search(text)
    if not m_url:
        return f"❌ URL 안 보임 (https://... 형식). 입력: '{args}'"
    url = m_url.group(0)
    name = actor_name or "URL monitor"

    try:
        from eidos_feedback_poller import register_source
    except Exception as e:
        return f"❌ import 실패: {e}"
    src = register_source(
        "url_monitor", name,
        config={"url": url},
        target_stage_id=stage_id,
        target_actor=actor_name,
        target_indicator="has_new_message",
        min_interval_sec=300,
    )
    if src is None:
        return "❌ source 등록 실패 (kind 미지원 또는 입력 문제)"
    return (
        f"✅ url_monitor 등록\n"
        f"- id: {src.id}\n"
        f"- url: {url}\n"
        f"- target: {stage_id[:12]} / actor '{actor_name}'\n"
        f"- polling 간격: {src.min_interval_sec:.0f}초\n"
        f"settings.json 의 feedback_poller_enabled=true 면 background 자동 polling."
    )


# ── 핸들러: /sources ───────────────────────────────────────────────────
def _cmd_sources(args: str, stage_id: Optional[str]) -> str:
    try:
        from eidos_feedback_poller import load_sources, summary_for_log
    except Exception as e:
        return f"❌ import 실패: {e}"
    sources = load_sources()
    if not sources:
        return "ℹ️ 등록된 source 없음. `/monitor <url>` 로 추가."
    s = summary_for_log()
    lines = [
        f"## 🔔 외부 신호 source ({s['total_sources']}개·enabled {s['enabled']}·disabled {s['disabled']})",
    ]
    for src in sources:
        icon = "✅" if src.enabled else "🚫"
        url = (src.config or {}).get("url") or (src.config or {}).get("feed_url") or ""
        target = (f"→ {src.target_stage_id[:8]}/{src.target_actor}"
                  if src.target_stage_id else "(target 없음)")
        last = src.last_signal_at[:19] if src.last_signal_at else "—"
        lines.append(
            f"- {icon} **{src.name}** [{src.kind}] `{src.id[:12]}`\n"
            f"   {url}\n"
            f"   {target}·fail={src.fail_count}·last_signal={last}"
        )
    return "\n".join(lines)


# ── 핸들러: /unmonitor ─────────────────────────────────────────────────
def _cmd_unmonitor(args: str, stage_id: Optional[str]) -> str:
    needle = args.strip()
    if not needle:
        return "ℹ️ 사용법: /unmonitor <id 또는 name 일부>"
    try:
        from eidos_feedback_poller import load_sources, delete_source
    except Exception as e:
        return f"❌ import 실패: {e}"
    sources = load_sources()
    needle_l = needle.lower()
    matches = [s for s in sources
               if needle_l in s.id.lower() or needle_l in s.name.lower()]
    if not matches:
        return f"⚠️ 매칭 source 없음 ('{needle}'·`/sources` 로 list 확인)"
    if len(matches) > 1:
        names = "\n  ".join(f"- {s.id[:12]} {s.name}" for s in matches[:5])
        return f"⚠️ {len(matches)} 개 매칭 — 더 구체적으로:\n  {names}"
    s = matches[0]
    ok = delete_source(s.id)
    return (f"✅ source '{s.name}' (id={s.id[:12]}) 삭제됨" if ok
            else f"❌ 삭제 실패")


# ── 핸들러: /good /bad ─────────────────────────────────────────────────
def _cmd_feedback(args: str, stage_id: Optional[str], reward: float, label: str) -> str:
    if not stage_id:
        return "❌ 활성 stage 없음. ToM tick 결과만 평가 가능."
    try:
        from eidos_agent_stage_store import read_history
        from eidos_belief_integrations import register_action_feedback
    except Exception as e:
        return f"❌ import 실패: {e}"
    events = read_history(stage_id, limit=50)
    # 가장 최근 execution event 찾기
    last_exec = None
    for ev in reversed(events):
        if ev.get("event") == "execution" and ev.get("action_id"):
            last_exec = ev
            break
    if last_exec is None:
        return ("⚠️ 직전 EIDOS action 없음 (이 stage 에서 tick 이 한번도 안 돌았거나 "
                "execution event 없음). ToM tick 후 다시 시도.")
    action_id = str(last_exec.get("action_id"))
    summary = args.strip()[:200] if args.strip() else label
    ok = register_action_feedback(
        stage_id, action_id, reward, summary=summary,
    )
    if not ok:
        return f"❌ feedback 기록 실패"
    return (
        f"✅ feedback 기록: **{label}** ({reward:+.2f}) → `{action_id}`\n"
        f"   사유: {summary}\n"
        f"   다음 tick PatternModel.from_history 가 자동 반영 → LLM prompt 에 학습된 가치 inject."
    )


def _cmd_good(args: str, stage_id: Optional[str]) -> str:
    return _cmd_feedback(args, stage_id, _GOOD_REWARD, "👍 좋음")


def _cmd_bad(args: str, stage_id: Optional[str]) -> str:
    return _cmd_feedback(args, stage_id, _BAD_REWARD, "👎 별로")


# ── 핸들러: /status ────────────────────────────────────────────────────
def _cmd_status(args: str, stage_id: Optional[str]) -> str:
    lines = ["## 📊 EIDOS 종합 상태"]
    # belief
    try:
        from eidos_belief_core import load_belief, summary_for_log as _belief_sum
        b = load_belief()
        bs = _belief_sum(b)
        lines.append(
            f"### 🧠 UserBelief\n"
            f"- 모드: {bs.get('work_state')}·기분: {bs.get('mood_signal')}\n"
            f"- energy={bs.get('energy'):.2f}·engagement={bs.get('engagement'):.2f}\n"
            f"- 누적 메시지 {bs.get('message_count')}회\n"
            f"- open thread {bs.get('pending_threads_open', 0)}개·"
            f"awaiting {bs.get('pending_threads_awaiting', 0)}개"
        )
    except Exception as e:
        lines.append(f"### 🧠 UserBelief — 로드 실패 ({e})")
    # goal tree
    if stage_id:
        try:
            from eidos_goal_tree import load_goal_tree, summary_for_log as _gt_sum
            tree = load_goal_tree(stage_id)
            if tree is not None and tree.root_goal_id:
                gs = _gt_sum(tree)
                lines.append(
                    f"\n### 🎯 GoalTree (stage {stage_id[:8]})\n"
                    f"- root: **{gs.get('root_title')}**·진행 {gs.get('root_progress'):.0%}\n"
                    f"- milestone 총 {gs.get('total_milestones')}개·"
                    f"active {gs.get('by_status', {}).get('active', 0)}"
                )
            else:
                lines.append(f"\n### 🎯 GoalTree — 없음 (`/goal <목표>` 로 시작)")
        except Exception as e:
            lines.append(f"\n### 🎯 GoalTree — 로드 실패 ({e})")
    else:
        lines.append("\n### 🎯 GoalTree — stage 미첨부")
    # sources
    try:
        from eidos_feedback_poller import summary_for_log as _fp_sum
        ss = _fp_sum()
        lines.append(
            f"\n### 🔔 외부 신호 source\n"
            f"- 총 {ss.get('total_sources', 0)}·enabled {ss.get('enabled', 0)}·"
            f"disabled {ss.get('disabled', 0)}·signal 누적 {ss.get('total_signals', 0)}건"
        )
    except Exception as e:
        lines.append(f"\n### 🔔 source — 로드 실패 ({e})")
    # episodes (long memory)
    try:
        from eidos_long_memory import load_long_memory, summary_for_log as _lm_sum
        m = load_long_memory()
        ls = _lm_sum(m)
        lines.append(
            f"\n### 💭 LongTermMemory\n"
            f"- episode 총 {ls.get('total_episodes', 0)}·"
            f"resolved {ls.get('by_outcome', {}).get('resolved', 0)}·"
            f"abandoned {ls.get('by_outcome', {}).get('abandoned', 0)}"
        )
    except Exception as e:
        lines.append(f"\n### 💭 LongTermMemory — 로드 실패 ({e})")
    return "\n".join(lines)


# ── [Wave3-D 2026-05-28] 도구 catalog slash 명령 ──────────────────────
def _cmd_approve_tool(args: str, stage_id: Optional[str]) -> str:
    """/approve_tool <tool_id> — pending_approval 도구 활성화."""
    tool_id = (args or "").strip().split()[0] if args else ""
    if not tool_id:
        return ("⚠️ 사용법: `/approve_tool <tool_id>`\n"
                "현재 승인 대기 도구는 `/tools pending` 으로 확인하세요.")
    try:
        import eidos_tool_catalog as tc
        ok = tc.approve_tool(tool_id)
        if not ok:
            return f"❌ 도구 `{tool_id}` 승인 실패 — pending_approval 아니거나 존재하지 않음."
        tool = tc.load_tool(tool_id)
        return (f"✅ 도구 승인 완료 — `{tool_id}` (type={tool.type})\n"
                f"  목표: {tool.target_indicator}\n"
                f"  설명: {tool.description[:200]}")
    except Exception as e:
        return f"❌ 승인 처리 실패 (graceful): {e}"


def _cmd_reject_tool(args: str, stage_id: Optional[str]) -> str:
    """/reject_tool <tool_id> [reason] — 도구 거부."""
    parts = (args or "").strip().split(maxsplit=1)
    if not parts:
        return "⚠️ 사용법: `/reject_tool <tool_id> [reason]`"
    tool_id = parts[0]
    reason = parts[1] if len(parts) > 1 else "user_rejected"
    try:
        import eidos_tool_catalog as tc
        ok = tc.reject_tool(tool_id, reason=reason)
        if not ok:
            return f"❌ 도구 `{tool_id}` 거부 실패 — 존재하지 않음."
        return f"❌ 도구 `{tool_id}` 거부 처리 완료. 사유: {reason[:100]}"
    except Exception as e:
        return f"❌ 거부 처리 실패 (graceful): {e}"


def _cmd_tools(args: str, stage_id: Optional[str]) -> str:
    """/tools [status] — 도구 catalog 조회. status 옵션: all/active/pending/rejected/abandoned."""
    arg = (args or "").strip().lower() or "all"
    try:
        import eidos_tool_catalog as tc
        if arg == "all":
            return tc.summarize_catalog_for_telegram(top_n=10)
        elif arg in ("pending", "pending_approval"):
            tools = tc.list_pending_approval()
        elif arg == "active":
            tools = tc.list_active_tools()
        elif arg == "rejected":
            tools = tc.list_tools_by_status("rejected")
        elif arg == "abandoned":
            tools = tc.list_tools_by_status("abandoned")
        else:
            return f"⚠️ status 옵션: all/active/pending/rejected/abandoned (입력: {arg})"

        if not tools:
            return f"🧰 도구 catalog ({arg}) — 비어있음"
        lines = [f"🧰 *도구 catalog — {arg}* ({len(tools)}개)"]
        for t in tools[:15]:
            lines.append(
                f"  • `{t.id[:14]}` ({t.type}) → {t.target_indicator or '-'}\n"
                f"    {t.description[:120]}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"❌ /tools 실패 (graceful): {e}"


# ── dispatcher ─────────────────────────────────────────────────────────
_HANDLERS: dict[str, Any] = {
    "help":          _cmd_help,
    "goal":          _cmd_goal,
    "decompose":     _cmd_decompose,
    "goals":         _cmd_goals,
    "done":          _cmd_done,
    "monitor":       _cmd_monitor,
    "sources":       _cmd_sources,
    "unmonitor":     _cmd_unmonitor,
    "good":          _cmd_good,
    "bad":           _cmd_bad,
    "status":        _cmd_status,
    # [Wave3-D 2026-05-28] 도구 catalog 관리
    "approve_tool":  _cmd_approve_tool,
    "reject_tool":   _cmd_reject_tool,
    "tools":         _cmd_tools,
}


__all__ = ["try_handle_command"]
