# eidos_belief_integrations.py
# [2026-05-26 Phase 8-B] 기존 EIDOS features (canvas/agent/project/automation) 와
# belief_core 의 통합 shim. 각 feature 가 1줄 helper 호출로 belief 에 활동·thread 반영.
#
# Phase 1~8A 까지는 채팅·proactive 만 belief 와 연결됐음. Phase 8-B 는 그 외 모든
# EIDOS 활동을 belief 의 single source of truth 로 끌어옴.
#
# helper 함수들 모두 graceful — belief_core 실패해도 caller 흐름 영향 X.
#
# 사용 예:
#   from eidos_belief_integrations import register_canvas_run_start, register_canvas_run_done
#
#   thread_id = register_canvas_run_start("크몽 응대 자동화", description="...")
#   # ... canvas 실행 ...
#   register_canvas_run_done("크몽 응대 자동화", success=True, summary="3 step 성공")

from __future__ import annotations

from typing import Optional


def _belief_load():
    try:
        from eidos_belief_core import load_belief
        return load_belief()
    except Exception as e:
        print(f"[belief_integrations] load 실패 (graceful): {e}")
        return None


def _belief_save(belief) -> bool:
    if belief is None:
        return False
    try:
        from eidos_belief_core import save_belief
        return save_belief(belief)
    except Exception as e:
        print(f"[belief_integrations] save 실패 (graceful): {e}")
        return False


# ── Canvas runner 통합 ───────────────────────────────────────────────
def register_canvas_run_start(
    canvas_name: str,
    description: str = "",
    importance: float = 0.6,
) -> Optional[str]:
    """Canvas 실행 시작 시 belief 에 pending_thread 등록.

    topic 은 `"캔버스: {canvas_name}"` 형식으로 prefix — register_canvas_run_done
    가 이름으로 매칭해서 자동 resolve 가능.

    Returns:
        thread_id (proactive followup 추적 가능) 또는 None (실패 시).
    """
    if not canvas_name:
        return None
    try:
        from eidos_belief_core import add_pending_thread
        b = _belief_load()
        if b is None:
            return None
        topic = f"캔버스: {canvas_name.strip()[:40]}"
        th = add_pending_thread(
            b, topic=topic, description=description[:80],
            importance=max(0.0, min(1.0, float(importance))),
            source="eidos_action",
        )
        _belief_save(b)
        return th.id
    except Exception as e:
        print(f"[belief_integrations] canvas_run_start 실패 (graceful): {e}")
        return None


def register_canvas_run_done(
    canvas_name: str,
    success: bool = True,
    summary: str = "",
) -> bool:
    """Canvas 실행 완료/실패 시 belief 갱신.

    success=True 면 매칭 thread → resolved.
    실패면 thread 그대로 (followup 으로 다시 알림 가능).
    모든 경우 update_from_eidos_action 으로 EIDOS activity 기록.

    Returns:
        belief 갱신 성공 여부.
    """
    if not canvas_name:
        return False
    try:
        from eidos_belief_core import (
            update_pending_thread_status, update_from_eidos_action,
            PendingThread,
        )
        b = _belief_load()
        if b is None:
            return False
        # 매칭 thread 찾기 (start 에서 등록한 prefix)
        topic_match = f"캔버스: {canvas_name.strip()[:40]}"
        if success:
            for raw in b.pending_threads:
                th = PendingThread.deserialize(raw)
                if th.status in ("open", "awaiting_response") and th.topic == topic_match:
                    update_pending_thread_status(b, th.id, "resolved")
                    break
        # EIDOS activity 기록 (성공·실패 모두)
        status_word = "ok" if success else "fail"
        update_from_eidos_action(
            b,
            action_id=f"eidos.canvas.run.{status_word}",
            summary=f"{canvas_name}: {summary}"[:80],
            proactive=False,
        )
        _belief_save(b)
        return True
    except Exception as e:
        print(f"[belief_integrations] canvas_run_done 실패 (graceful): {e}")
        return False


# ── ToM Agent runner 통합 ───────────────────────────────────────────
def register_agent_tick(
    stage_name: str,
    action_id: str,
    status: str,
    summary: str = "",
) -> bool:
    """Agent stage 1 tick 완료 시 belief 의 EIDOS activity 기록.

    action_id 는 dispatch 한 verb (eidos.tool.llm.write 등). status 는 ok/fail/hitl 등.
    proactive=False — agent tick 은 사용자 명시 요청·proactive 시간 무관.
    """
    if not stage_name:
        return False
    try:
        from eidos_belief_core import update_from_eidos_action
        b = _belief_load()
        if b is None:
            return False
        update_from_eidos_action(
            b,
            action_id=f"eidos.agent.{action_id}",
            summary=f"{stage_name}/{status}: {summary}"[:80],
            proactive=False,
        )
        _belief_save(b)
        return True
    except Exception as e:
        print(f"[belief_integrations] agent_tick 실패 (graceful): {e}")
        return False


# ── Project task 통합 ────────────────────────────────────────────────
def register_project_task_done(
    project_name: str,
    task_topic: str,
    summary: str = "",
) -> int:
    """Project 의 task 완료 시 매칭 thread 자동 resolve.

    task_topic 키워드가 thread.topic 또는 thread.description 에 substring 매칭되면
    그 thread 가 resolved. 보수적 매칭 (대소문자 무시·2자 이상).

    Returns:
        resolved 처리된 thread 개수.
    """
    if not task_topic or len(task_topic.strip()) < 2:
        return 0
    try:
        from eidos_belief_core import (
            update_pending_thread_status, update_from_eidos_action,
            PendingThread,
        )
        b = _belief_load()
        if b is None:
            return 0
        needle = task_topic.strip().lower()
        n_resolved = 0
        for raw in b.pending_threads:
            th = PendingThread.deserialize(raw)
            if th.status not in ("open", "awaiting_response"):
                continue
            haystack = (th.topic + " " + th.description).lower()
            if needle in haystack:
                update_pending_thread_status(b, th.id, "resolved")
                n_resolved += 1
        update_from_eidos_action(
            b,
            action_id=f"eidos.project.{project_name}.task_done",
            summary=task_topic[:80],
            proactive=False,
        )
        _belief_save(b)
        return n_resolved
    except Exception as e:
        print(f"[belief_integrations] project_task_done 실패 (graceful): {e}")
        return 0


# ── PCA DAG 통합 (2026-05-27 ToM↔PCA Phase C) ──────────────────────
def register_pca_dag_run(
    dag_name: str,
    success: bool = True,
    summary: str = "",
    nodes_done: int = 0,
    tool_calls: int = 0,
    duration_ms: float = 0.0,
) -> bool:
    """ToM 이 위임한 PCA DAG 실행 완료/실패 시 belief 에 기록.

    ToM 의 일반 agent tick (register_agent_tick) 보다 더 풍부한 metric 포함.
    proactive=False — DAG 실행은 사용자 명시 stage 정책의 결과·proactive 무관.

    Args:
        dag_name: 호출된 DAG 파일명 (eidos.tool.pca.run_dag 의 args.dag_name).
        success: ok 여부.
        summary: final_result 앞부분 (최대 80자 trim 됨).
        nodes_done / tool_calls / duration_ms: 실행 metric.

    Returns:
        belief 저장 성공 여부.
    """
    if not dag_name:
        return False
    try:
        from eidos_belief_core import update_from_eidos_action
        b = _belief_load()
        if b is None:
            return False
        status_word = "ok" if success else "fail"
        # action_id 에 ok/fail 분리해 두면 update_from_eidos_action 으로 누적 시
        # belief brief 에서 패턴 파악 쉬움.
        msg_core = f"{dag_name} [{nodes_done}n·{tool_calls}t·{duration_ms:.0f}ms]"
        if summary:
            msg_core += f" — {summary}"
        update_from_eidos_action(
            b,
            action_id=f"eidos.pca.dag.{status_word}",
            summary=msg_core[:80],
            proactive=False,
        )
        _belief_save(b)
        return True
    except Exception as e:
        print(f"[belief_integrations] pca_dag_run 실패 (graceful): {e}")
        return False


# ── GoalTree milestone 통합 (Phase 9-C) ────────────────────────────
def register_milestone_progress(
    stage_id: str,
    milestone_id: str,
    progress: float,
    *,
    manual: bool = True,
    summary: str = "",
) -> bool:
    """[Phase 9-C] milestone 진척도 명시 갱신 + belief activity 기록.

    manual=True 면 children 평균 override 활성 — 사용자가 직접 "이거 70% 진행" 표시.
    manual=False 면 자동 평가 결과 반영 (auto_evaluate_milestones_from_history 가 사용).

    graceful — goal_tree 로드 실패해도 caller 영향 X.

    Returns: 갱신 성공 여부.
    """
    if not stage_id or not milestone_id:
        return False
    try:
        from eidos_goal_tree import load_goal_tree, save_goal_tree, set_milestone_progress
        tree = load_goal_tree(stage_id)
        if tree is None:
            return False
        ok = set_milestone_progress(tree, milestone_id, progress, manual=manual)
        if ok:
            save_goal_tree(tree)
            # belief 에 EIDOS activity 기록 — milestone progress 도 EIDOS 학습 일환
            try:
                from eidos_belief_core import update_from_eidos_action
                b = _belief_load()
                if b is not None:
                    update_from_eidos_action(
                        b,
                        action_id="eidos.goal.milestone.progress",
                        summary=f"{milestone_id[:20]}={progress:.2f} {summary}"[:80],
                        proactive=False,
                    )
                    _belief_save(b)
            except Exception as _e_b:
                print(f"[belief_integrations] milestone belief 기록 실패 (graceful): {_e_b}")
        return ok
    except Exception as e:
        print(f"[belief_integrations] milestone_progress 실패 (graceful): {e}")
        return False


def auto_evaluate_milestones_from_history(
    stage_id: str,
    *,
    history_limit: int = 60,
) -> dict:
    """[Phase 9-C] tick 후 자동 호출 — stage 의 history 를 보고 leaf milestone progress 자동 추정.

    1. read_history(stage_id, limit) → recent events
    2. evaluate_progress_from_history(tree, events) → 키워드 매칭 휴리스틱
    3. recompute_progress(tree) → internal milestone 평균 재계산
    4. save_goal_tree(tree)

    manual=True 인 milestone·done/abandoned 는 건너뜀.

    graceful — goal_tree 없거나 history 없으면 빈 dict 반환·throw X.

    Returns: {"changes": {milestone_id: new_progress}, "n_changed": N, "skipped": "reason"}
    """
    if not stage_id:
        return {"changes": {}, "n_changed": 0, "skipped": "stage_id 비어있음"}
    try:
        from eidos_goal_tree import (
            load_goal_tree, save_goal_tree,
            evaluate_progress_from_history, recompute_progress,
        )
        from eidos_agent_stage_store import read_history
    except Exception as e:
        return {"changes": {}, "n_changed": 0, "skipped": f"import 실패: {e}"}

    try:
        tree = load_goal_tree(stage_id)
        if tree is None or not tree.root_goal_id:
            return {"changes": {}, "n_changed": 0, "skipped": "goal_tree 없음"}
        events = read_history(stage_id, limit=history_limit)
        if not events:
            return {"changes": {}, "n_changed": 0, "skipped": "history 비어있음"}
        changes = evaluate_progress_from_history(tree, events)
        # 내부 milestone progress 도 재계산 (children 평균)
        n_internal = recompute_progress(tree)
        # 변화 있으면만 저장
        if changes or n_internal > 0:
            save_goal_tree(tree)
        return {
            "changes": changes,
            "n_changed": len(changes),
            "n_internal_recomputed": n_internal,
        }
    except Exception as e:
        print(f"[belief_integrations] auto_evaluate_milestones 실패 (graceful): {e}")
        return {"changes": {}, "n_changed": 0, "skipped": f"exception: {e}"}


# ── Policy Learning (Phase 12) ────────────────────────────────────────
def register_action_feedback(
    stage_id: str,
    action_id: str,
    reward: float,
    *,
    context: Optional[dict] = None,
    summary: str = "",
) -> bool:
    """[Phase 12] EIDOS action 에 대한 명시적 reward 기록.

    history.jsonl 에 explicit_feedback event 1줄 append → 다음 PatternModel.from_history
    가 자동 ActionValue 에 누적. EIDOS 가 다음 tick prompt 에 그 학습 가치 반영.

    Args:
        stage_id: 어느 stage 의 action 인지
        action_id: 평가할 action_id (eidos.tool.*)
        reward: -1.0 ~ +1.0 (-1=매우 별로·0=중립·+1=완벽)
        context: 평가 시점 belief snapshot (옵션 — conditional 학습 강화)
        summary: 사람 읽기 메모

    Returns: True 면 history 에 기록 성공.
    """
    if not stage_id or not action_id:
        return False
    try:
        rwd = float(reward)
        rwd = max(-1.0, min(1.0, rwd))
    except Exception:
        return False
    try:
        from eidos_agent_stage_store import append_history
    except Exception as e:
        print(f"[belief_integrations] action_feedback import 실패 (graceful): {e}")
        return False
    event = {
        "event": "explicit_feedback",
        "action_id": action_id,
        "reward": rwd,
        "summary": summary[:200] if summary else "",
    }
    if isinstance(context, dict) and context:
        event["context"] = context
    return append_history(stage_id, event)


# ── Automation action 통합 ──────────────────────────────────────────
def register_automation_run(
    action_id: str,
    target: str = "",
    success: bool = True,
    summary: str = "",
) -> bool:
    """자동화 액션 (텔레그램 전송·이메일·웹 클릭 등) 실행 후 belief 에 기록.

    action_id 예: "telegram.send" / "email.send" / "browser.click".
    target 은 수신자·URL 같은 컨텍스트.
    """
    if not action_id:
        return False
    try:
        from eidos_belief_core import update_from_eidos_action
        b = _belief_load()
        if b is None:
            return False
        status_word = "ok" if success else "fail"
        msg = f"[{action_id}.{status_word}] target={target} {summary}"[:80]
        update_from_eidos_action(
            b,
            action_id=f"eidos.automation.{action_id}.{status_word}",
            summary=msg,
            proactive=False,
        )
        _belief_save(b)
        return True
    except Exception as e:
        print(f"[belief_integrations] automation_run 실패 (graceful): {e}")
        return False
