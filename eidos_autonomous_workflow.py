# eidos_autonomous_workflow.py
# [Wave7-B 2026-05-28] Milestone → 자동 action chain 생성·실행.
#
# 진짜 자율의 핵심 매커니즘. 기존 run_stage_one_tick 은 매 tick 단일 action 만
# 결정·실행. milestone 이 추상적이면 LLM 이 meta.observe / ask_user 같은 안전
# 회피만 고름 (관찰된 패턴).
#
# 이 모듈은:
#   1. milestone 보고 stagnation 감지 (meaningful action 0건·관찰만 반복)
#   2. LLM 으로 concrete action chain 자동 생성 (3~5 step)
#      예: "Q1 매출 1천만" → [web.search 시장조사 → web.search 경쟁사 →
#                            llm.write summary → llm.write strategy]
#   3. step-by-step 실행·각 step 결과를 다음 step args 의 context 로 전파
#   4. 외부효과 action 은 기존 Wave3 승인 게이트 통과
#   5. 결과 jsonl 누적·채팅 보고
#
# 사용자가 "조사해줘" 안 해도 EIDOS 가 자기 milestone 보고 자동 시작.
# 자율성 약화 X — 외부효과만 게이트·내부 (meta·llm.write·web.search·web.fetch)
# 는 즉시 실행.

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable


_LOG_DIR = os.path.join("eidos_files", "predictions")
_WF_LOG = os.path.join(_LOG_DIR, "autonomous_workflow_log.jsonl")
_MAX_LOG_BYTES = 2 * 1024 * 1024


# ── dataclasses ──────────────────────────────────────────────────────
@dataclass
class WorkflowStep:
    """체인 한 step 의 spec + 실행 결과."""
    step_idx: int = 0
    action_id: str = ""
    args: dict = field(default_factory=dict)
    description: str = ""               # LLM 이 적은 이 step 의 목적

    # 실행 결과
    status: str = "pending"             # pending/done/skipped/failed
    result_preview: str = ""
    started_at: str = ""
    completed_at: str = ""
    dry_run: bool = False

    def serialize(self) -> dict:
        return asdict(self)


@dataclass
class WorkflowChain:
    """milestone 1개에 대응하는 전체 action chain."""
    chain_id: str = ""
    stage_id: str = ""
    milestone_id: str = ""
    milestone_title: str = ""
    rationale: str = ""                 # LLM 이 적은 chain 전체 의도
    steps: list = field(default_factory=list)
    status: str = "pending"             # pending/running/completed/failed/aborted
    created_at: str = ""
    completed_at: str = ""

    def serialize(self) -> dict:
        return {
            "chain_id": self.chain_id,
            "stage_id": self.stage_id,
            "milestone_id": self.milestone_id,
            "milestone_title": self.milestone_title,
            "rationale": self.rationale,
            "steps": [s.serialize() if hasattr(s, "serialize") else s
                      for s in self.steps],
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _new_id() -> str:
    import uuid
    return uuid.uuid4().hex[:10]


def _ensure_log_dir() -> None:
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
    except Exception:
        pass


def _rotate_if_needed() -> None:
    try:
        if (os.path.exists(_WF_LOG)
                and os.path.getsize(_WF_LOG) > _MAX_LOG_BYTES):
            bak = _WF_LOG + f".{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
            try:
                os.rename(_WF_LOG, bak)
            except Exception:
                pass
    except Exception:
        pass


# ── stagnation 감지 ──────────────────────────────────────────────────
_MEANINGLESS_ACTIONS = {
    "eidos.meta.observe",
    "eidos.meta.no_op",
    "eidos.tool.ask_user",      # 사용자에게 미루기 — 자율성 약함
    "eidos.meta.clarify_question",
}


def detect_stagnation(
    stage_id: str,
    milestone_id: str,
    threshold_meaningless: int = 3,
    history_window: int = 10,
) -> tuple[bool, dict]:
    """stage 의 history.jsonl 에서 최근 N step 보고 stagnation 검사.

    Returns: (is_stagnant, diagnostic_dict).
      is_stagnant = True 면 chain 트리거 권장.
      diagnostic = {n_meaningless, n_total, recent_actions}.
    """
    diag = {"n_meaningless": 0, "n_total": 0, "recent_actions": []}
    if not stage_id:
        return False, diag

    try:
        from eidos_agent_stage_store import read_history
    except Exception:
        return False, diag

    try:
        entries = read_history(stage_id, limit=history_window * 3) or []
    except Exception:
        return False, diag

    # decision event 만 추출 (실행 결과 보는 게 아니라 LLM 이 고른 action)
    actions: list = []
    for e in reversed(entries):
        if not isinstance(e, dict):
            continue
        if e.get("event") != "decision":
            continue
        dec = e.get("decision") or {}
        aid = str(dec.get("action_id", ""))
        if aid:
            actions.append(aid)
            if len(actions) >= history_window:
                break

    diag["recent_actions"] = list(reversed(actions))
    diag["n_total"] = len(actions)
    n_meaningless = sum(1 for a in actions if a in _MEANINGLESS_ACTIONS)
    diag["n_meaningless"] = n_meaningless

    # 조건: meaningless action 이 threshold 이상 + 전체에서 차지 비율 70%+
    is_stag = (
        n_meaningless >= threshold_meaningless
        and (len(actions) == 0 or n_meaningless / max(1, len(actions)) >= 0.7)
    )
    return is_stag, diag


# ── LLM 시스템 프롬프트 ─────────────────────────────────────────────────
_COMPOSE_SYSTEM = """\
너는 EIDOS 의 autonomous workflow composer 다.

milestone 1개를 받고, 그것을 진행하기 위한 **구체 action chain** 을 작성해라.
사용자가 "조사해줘"·"정리해줘" 같은 명령을 직접 안 했어도, milestone 진행을
위해 EIDOS 가 자기 판단으로 시작하는 첫 발이다.

원칙:
- step 3~5 개 (너무 길면 cost 폭증·너무 짧으면 의미 없음)
- 사용 가능 action_id (이 중에서만 골라라):
    * eidos.tool.web.search    args: {query}              — 외부 검색
    * eidos.tool.web.fetch     args: {url}                — URL 페치
    * eidos.tool.llm.write     args: {target, content}    — 파일 작성·content 는 brief
    * eidos.meta.observe       args: {}                   — 관찰·요약 (가끔만)
    * eidos.tool.pca.run_dag   args: {dag_name, input}    — 정의된 DAG 실행
- chain 흐름은 보통: 정보 수집 → 분석 → 산출물 작성
- meta.observe 1개 이상 들어가지 마라 (이게 stagnation 원인이었음)
- ask_user / clarify_question 절대 사용 X (자율 chain 이라 사용자 안 부름)
- 각 step 의 description 은 한국어로·왜 이 step 필요한지

JSON 만 출력:
{
  "rationale": "이 chain 전체의 목적·한 줄",
  "steps": [
    {"action_id": "eidos.tool.web.search",
     "args": {"query": "..."},
     "description": "왜 이 step 인지"},
    ...
  ]
}
"""


# ── chain compose ────────────────────────────────────────────────────
async def compose_action_chain_async(
    milestone_title: str,
    milestone_description: str = "",
    success_criteria: Optional[list] = None,
    stage_id: str = "",
    stage_goal: str = "",
    work_state: str = "unknown",
    timeout_sec: float = 15.0,
) -> WorkflowChain:
    """LLM 으로 milestone 에 맞는 chain 자동 생성.

    실패 시 status="failed" + 빈 steps 반환.
    """
    chain = WorkflowChain(
        chain_id=_new_id(),
        stage_id=stage_id,
        milestone_id="",
        milestone_title=(milestone_title or "")[:120],
        created_at=_now_iso(),
        status="pending",
    )

    if not milestone_title:
        chain.status = "failed"
        chain.rationale = "milestone_title 비어있음"
        return chain

    try:
        from llm_module import get_llm_response_async
    except Exception:
        chain.status = "failed"
        chain.rationale = "LLM 모듈 부재"
        return chain

    criteria_block = ""
    if success_criteria:
        criteria_block = "[success_criteria]\n" + "\n".join(
            f"  - {str(c)[:120]}" for c in (success_criteria or [])[:5]
        ) + "\n\n"

    # [Wave10] 누적 학습 brief — chain 생성에 과거 효과적 action 반영
    learning_block = ""
    try:
        from eidos_learning_inject import build_learning_brief
        learning_block = build_learning_brief(stage_id=stage_id, max_chars=500)
    except Exception:
        pass

    prompt = (
        f"[stage_goal] {(stage_goal or '')[:200]}\n"
        f"[work_state] {work_state}\n"
        f"[milestone] {milestone_title}\n"
        + (f"[description] {milestone_description[:300]}\n"
           if milestone_description else "")
        + criteria_block
        + (f"\n{learning_block}\n\n" if learning_block else "")
        + "위 milestone 의 진행을 시작할 chain (3~5 step). JSON 만 출력."
    )

    raw = ""
    try:
        raw = await asyncio.wait_for(
            get_llm_response_async(
                prompt,
                max_tokens=2048,
                system_prompt=_COMPOSE_SYSTEM,
                response_mime_type="application/json",
            ),
            timeout=timeout_sec,
        )
    except asyncio.TimeoutError:
        chain.status = "failed"
        chain.rationale = "LLM timeout"
        return chain
    except Exception as e:
        chain.status = "failed"
        chain.rationale = f"LLM 실패: {str(e)[:120]}"
        return chain

    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            chain.status = "failed"
            chain.rationale = "non-dict JSON"
            return chain
    except Exception as e:
        chain.status = "failed"
        chain.rationale = f"JSON 파싱 실패: {str(e)[:80]}"
        return chain

    chain.rationale = str(data.get("rationale", ""))[:300]
    steps_raw = data.get("steps") or []
    if not isinstance(steps_raw, list):
        chain.status = "failed"
        chain.rationale = "steps 가 list 아님"
        return chain

    for i, sr in enumerate(steps_raw[:6]):
        if not isinstance(sr, dict):
            continue
        aid = str(sr.get("action_id", ""))[:80]
        if not aid:
            continue
        # ask_user/clarify_question 차단 (자율성 약화 방지)
        if aid in ("eidos.tool.ask_user", "eidos.meta.clarify_question"):
            continue
        args = sr.get("args")
        if not isinstance(args, dict):
            args = {}
        step = WorkflowStep(
            step_idx=i,
            action_id=aid,
            args=dict(args),
            description=str(sr.get("description", ""))[:200],
            status="pending",
        )
        chain.steps.append(step)

    if not chain.steps:
        chain.status = "failed"
        chain.rationale = chain.rationale or "유효 step 0개"
    return chain


# ── chain execute ───────────────────────────────────────────────────
async def execute_workflow_chain_async(
    chain: WorkflowChain,
    on_external_approve: Optional[Callable] = None,
    step_timeout_sec: float = 60.0,
) -> WorkflowChain:
    """체인 step 순차 실행·각 step 결과 다음 step args 에 inject.

    Args:
      chain: compose_action_chain_async 결과.
      on_external_approve: Wave3 외부 승인 callback (외부효과 action 시).
      step_timeout_sec: 단일 step timeout.

    실행 흐름:
      - 각 step 실행 → result_preview 저장
      - 다음 step 의 args 에 "_prior_context" 키로 누적 결과 전파
      - llm.write step 의 content 가 비면 누적 context 로 auto-fill
      - 실패 step 만나면 abort (status=failed) — 나머지 skip

    Returns: 업데이트된 chain (steps 결과 채워짐·jsonl 자동 로그).
    """
    chain.status = "running"
    try:
        from eidos_agent_runner import execute_eidos_action
    except Exception as e:
        chain.status = "failed"
        chain.rationale = (chain.rationale or "") + f" | execute import 실패: {e}"
        _write_log(chain)
        return chain

    accumulated_context = ""

    for step in chain.steps:
        step.started_at = _now_iso()
        step.status = "running"

        # 누적 context 를 args 에 inject
        try:
            args_with_ctx = dict(step.args or {})
            if accumulated_context:
                args_with_ctx["_prior_context"] = accumulated_context[:2000]
                # llm.write 의 content 가 비면 누적 context 로 auto-fill
                if (step.action_id == "eidos.tool.llm.write"
                        and not args_with_ctx.get("content")
                        and not args_with_ctx.get("brief")):
                    args_with_ctx["content"] = accumulated_context[:2000]
        except Exception:
            args_with_ctx = dict(step.args or {})

        decision = {
            "action_id": step.action_id,
            "args": args_with_ctx,
            "reason": f"autonomous chain step {step.step_idx}: {step.description[:100]}",
        }

        try:
            result = await asyncio.wait_for(
                execute_eidos_action(
                    decision,
                    safety_mode="normal",
                    stage_id=chain.stage_id,
                    on_external_approve=on_external_approve,
                ),
                timeout=step_timeout_sec,
            )
        except asyncio.TimeoutError:
            step.status = "failed"
            step.result_preview = f"timeout {step_timeout_sec}s"
            step.completed_at = _now_iso()
            chain.status = "failed"
            break
        except Exception as e:
            step.status = "failed"
            step.result_preview = f"예외: {type(e).__name__}: {str(e)[:120]}"
            step.completed_at = _now_iso()
            chain.status = "failed"
            break

        # 결과 평가
        st = str(result.get("status", "")).lower()
        step.dry_run = bool(result.get("dry_run", False))
        step.result_preview = str(result.get("result", ""))[:400]
        step.completed_at = _now_iso()

        if st == "ok":
            step.status = "done"
            # 누적 context 추가
            piece = f"[step {step.step_idx} · {step.action_id}]\n{step.result_preview}\n"
            accumulated_context += piece
        elif st in ("dry_run",) or step.dry_run:
            step.status = "skipped"
            piece = f"[step {step.step_idx} · {step.action_id} DRY]\n{step.result_preview}\n"
            accumulated_context += piece
        else:
            step.status = "failed"
            chain.status = "failed"
            break

    if chain.status != "failed":
        chain.status = "completed"
    chain.completed_at = _now_iso()
    _write_log(chain)
    return chain


def _write_log(chain: WorkflowChain) -> None:
    _ensure_log_dir()
    _rotate_if_needed()
    try:
        with open(_WF_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(chain.serialize(), ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[autonomous_workflow] log 실패 (graceful): {e}")


def load_recent_workflows(max_n: int = 100) -> list:
    """최근 chain 실행 log."""
    out: list = []
    if not os.path.exists(_WF_LOG):
        return out
    try:
        with open(_WF_LOG, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return out
    for line in reversed(lines):
        if len(out) >= max_n:
            break
        try:
            out.append(json.loads(line.strip()))
        except Exception:
            continue
    return list(reversed(out))


def clear_log() -> None:
    """테스트용."""
    try:
        if os.path.exists(_WF_LOG):
            os.remove(_WF_LOG)
    except Exception:
        pass


# ── 통합 helper — milestone 객체에서 chain 생성·실행까지 ─────────────
async def run_autonomous_workflow_for_milestone_async(
    milestone,
    stage_id: str = "",
    stage_goal: str = "",
    work_state: str = "unknown",
    on_external_approve: Optional[Callable] = None,
) -> WorkflowChain:
    """편의 함수 — milestone 객체 받아 compose + execute 한 흐름.

    milestone 은 eidos_goal_tree.Milestone 또는 호환 (title·description·
    success_criteria·id 속성). graceful — 어떤 실패도 chain.status=failed.
    """
    title = ""
    desc = ""
    criteria: list = []
    m_id = ""
    try:
        title = (getattr(milestone, "title", "") or "")
        desc = (getattr(milestone, "description", "") or "")
        criteria = list(getattr(milestone, "success_criteria", []) or [])
        m_id = getattr(milestone, "id", "") or ""
    except Exception:
        pass

    chain = await compose_action_chain_async(
        milestone_title=title,
        milestone_description=desc,
        success_criteria=criteria,
        stage_id=stage_id,
        stage_goal=stage_goal,
        work_state=work_state,
    )
    chain.milestone_id = m_id
    if chain.status == "failed" or not chain.steps:
        _write_log(chain)
        return chain
    return await execute_workflow_chain_async(
        chain, on_external_approve=on_external_approve,
    )
