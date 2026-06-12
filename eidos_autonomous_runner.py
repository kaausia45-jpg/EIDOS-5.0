# -*- coding: utf-8 -*-
# eidos_autonomous_runner.py
# ──────────────────────────────────────────────────────────────────────────────
# EIDOS 자율 실행 매니저 — Pre-flight 승인 → 스코프 동안 모든 게이트 자동 통과.
#
# 사용자 시나리오:
#   1. 텔레그램 "/explore" 또는 "찾아봐" 메시지
#   2. 매니저가 제어실 top_goal 읽고 LLM 으로 사전 계획 생성
#   3. 텔레그램 카드 발송: 목표 / 단계 / 사이트 / 액션 추정 / hard-block / 비용 cap
#   4. 사용자가 [✅ 이번 한 번][✅ 1시간 자율][✅ 24시간 자율][❌ 거부] 중 선택
#   5. 승인 시 AutonomousRun 활성화 → 스코프(single_chain | time_window) 안에서
#      모든 _pause_with_telegram_gate 가 자동 통과 (단, hard-block 액션은 즉시 abort)
#   6. 종료 시 텔레그램 결과 요약 카드
#
# Why: 자율 운영을 원하지만 외출 중 EIDOS 가 결제·등록·DM 같은 외부 액션을
# 임의 실행하면 위험. "단 한 번의 high-quality 사전 승인 + 그 안에서는 자율"
# 모델로 안전과 자율 둘 다 챙김.
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import html
import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Awaitable, Callable, Dict, List, Optional


# ── 상수 ─────────────────────────────────────────────────────────────────────
_RUNS_PATH = os.path.join("eidos_files", "autonomous_runs.json")

# 자율 모드에서도 절대 자동 실행 금지 — 외부 트랜잭션·계정·게시 류.
# 매칭은 ActionPlan 의 url + selector + text + value blob 소문자 substring.
DEFAULT_HARD_BLOCKS: List[str] = [
    # 결제/구매
    "결제", "송금", "구매하", "주문 결제", "결제 진행", "checkout",
    "신용카드", "카드 등록", "카드등록", "카드정보",
    # 계정/등록
    "회원가입", "계정 등록", "계정등록", "이메일 인증", "휴대폰 인증",
    "본인 인증", "본인인증", "sign up", "signup",
    # 외부 게시/DM
    "dm 발송", "dm발송", "메시지 발송", "sns 게시", "포스팅",
    "publish", "post tweet", "댓글 등록", "댓글등록",
    # 광고/마케팅 비용
    "광고 집행", "광고집행", "유료 광고", "ad spend",
]


class RunStatus:
    PENDING_APPROVAL = "pending_approval"
    ACTIVE = "active"
    COMPLETED = "completed"
    ABORTED = "aborted"
    REJECTED = "rejected"


class ScopeType:
    SINGLE_CHAIN = "single_chain"   # 단일 chain — chain 종료 시 자동 만료
    TIME_WINDOW = "time_window"     # 시간창 — expires_at 까지 활성


# ── 데이터 모델 ──────────────────────────────────────────────────────────────

@dataclass
class AutonomousPlan:
    """LLM 이 생성한 사전 실행 계획 — 사용자 승인 카드의 핵심 컨텐츠."""
    objective: str = ""                                  # 1줄 목표
    stages: List[str] = field(default_factory=list)      # ["1. ...", "2. ..."]
    sites: List[str] = field(default_factory=list)       # ["kmong.com", ...]
    expected_actions: Dict[str, int] = field(default_factory=dict)
    estimated_duration_min: int = 30
    hard_blocks: List[str] = field(default_factory=lambda: list(DEFAULT_HARD_BLOCKS))
    budget_krw: int = 0
    # [Execution-First 2026-05-01] chain 마지막 stage 의 외부 실행 행위.
    # 분석/요약/보고서로 끝나는 plan 차단을 위해 LLM 이 plan 생성 시 반드시 채워야 함.
    # 비어있으면 _build_explore_chain_stages 가 chain 빌드 거부 → 사용자 카드 안내.
    execution_action: str = ""                           # 1줄 외부 행위 (예: "krmong.com 에 상품 페이지 등록")
    execution_target: str = ""                           # 행위 대상 URL/플랫폼 (예: "kmong.com/gigs/new")
    # [Proactive Advisor 2026-05-17] "" = 기존 execution-first 3단 explore.
    # "advisor_draft" = 자율 선제 제안 경로 — 2단 경량 체인(탐색→초안), 외부
    # 행동 단계 없음. _start_explore_chain_for_run 가 이 마커로 분기.
    mode: str = ""


@dataclass
class AutonomousRun:
    run_id: str
    status: str
    seed_goal: str
    plan: AutonomousPlan
    created_at: float
    scope_type: str = ScopeType.SINGLE_CHAIN
    scope_value: str = ""        # chain_id (single_chain) | str(epoch) (time_window)
    approved_at: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0
    expires_at: float = 0.0      # time_window 전용
    chain_id: str = ""           # single_chain 전용
    last_event: str = ""
    actions_executed: int = 0
    hard_block_violations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AutonomousRun":
        d = dict(d)
        plan_d = d.pop("plan", {}) or {}
        if isinstance(plan_d, dict):
            plan = AutonomousPlan(
                objective=str(plan_d.get("objective", "")),
                stages=list(plan_d.get("stages", []) or []),
                sites=list(plan_d.get("sites", []) or []),
                expected_actions=dict(plan_d.get("expected_actions", {}) or {}),
                estimated_duration_min=int(plan_d.get("estimated_duration_min", 30) or 30),
                hard_blocks=list(plan_d.get("hard_blocks", []) or DEFAULT_HARD_BLOCKS),
                budget_krw=int(plan_d.get("budget_krw", 0) or 0),
                # [트랙 A 2026-05-18] 재시작 보존 누락 fix — 이 3개가 빠져
                #   _load()→from_dict() 라운드트립마다 소실됐다. mode 소실 시
                #   advisor_draft run 이 재시작 후 execution-first 3단 분기로
                #   잘못 빌드 → advisor_draft.md 부재 + chain_complete 미발화
                #   → run 영구 active → advisor 영구 침묵(관측된 그 상태).
                #   execution_action/target 도 같은 class 의 버그(3단 경로
                #   재시작 후 resume).
                execution_action=str(plan_d.get("execution_action", "") or ""),
                execution_target=str(plan_d.get("execution_target", "") or ""),
                mode=str(plan_d.get("mode", "") or ""),
            )
        else:
            plan = AutonomousPlan()
        # unknown 키는 무시하고 dataclass 필드만 추출
        valid_fields = {f for f in cls.__dataclass_fields__.keys() if f != "plan"}
        kwargs = {k: v for k, v in d.items() if k in valid_fields}
        return cls(plan=plan, **kwargs)


# ── 매니저 ──────────────────────────────────────────────────────────────────

class AutonomousRunManager:
    """자율 실행 라이프사이클 싱글톤. 영속화는 eidos_files/autonomous_runs.json.

    set_chain_starter() 로 등록되는 콜백이 실제 chain 실행 진입점 — 매니저는
    chain 실행 자체를 알지 못하고, 승인 후 콜백을 발화해 외부(core) 가 처리.
    """
    _instance: Optional["AutonomousRunManager"] = None

    def __init__(self, runs_path: str = _RUNS_PATH):
        self._runs_path = runs_path
        self._runs: Dict[str, AutonomousRun] = {}
        self._active_run_id: Optional[str] = None
        # 승인 후 chain 시작 콜백 — 통합 레이어(core)에서 등록.
        # signature: (run: AutonomousRun) -> Awaitable[Optional[str]]  # returns chain_id
        self._chain_starter: Optional[Callable[[AutonomousRun], Awaitable[Optional[str]]]] = None
        # 알림 콜백 (텔레그램 발송 등) — 통합 레이어에서 등록.
        # signature: (text: str, *, run: Optional[AutonomousRun] = None) -> Awaitable[None]
        self._notify: Optional[Callable[..., Awaitable[None]]] = None
        # bootstrap_with_core 가 한 번 등록 후 True (idempotent 가드)
        self._bootstrapped: bool = False
        self._load()

    # ── 싱글톤 ────────────────────────────────────────────────────────────
    @classmethod
    def get(cls) -> "AutonomousRunManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_for_test(cls, runs_path: Optional[str] = None) -> "AutonomousRunManager":
        """테스트 전용 — 새 매니저 인스턴스로 교체."""
        cls._instance = cls(runs_path=runs_path or _RUNS_PATH)
        return cls._instance

    # ── 통합 콜백 등록 ─────────────────────────────────────────────────────
    def set_chain_starter(
        self,
        cb: Callable[[AutonomousRun], Awaitable[Optional[str]]],
    ) -> None:
        self._chain_starter = cb

    def set_notifier(
        self,
        cb: Callable[..., Awaitable[None]],
    ) -> None:
        self._notify = cb

    # ── 영속화 ────────────────────────────────────────────────────────────
    def _load(self) -> None:
        if not os.path.exists(self._runs_path):
            return
        try:
            with open(self._runs_path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
        except Exception as e:
            print(f"  ⚠️ [AutoRunner] runs 로드 실패 (무시): {e}")
            return
        for d in data.get("runs", []):
            try:
                run = AutonomousRun.from_dict(d)
                self._runs[run.run_id] = run
            except Exception as _e:
                print(f"  ⚠️ [AutoRunner] run 파싱 실패 (skip): {_e}")
        active = data.get("active_run_id")
        if active and active in self._runs:
            run = self._runs[active]
            # 만료/완료 검사 — 재시작 후에도 깨끗한 상태 보장
            if run.status != RunStatus.ACTIVE:
                self._active_run_id = None
            elif run.scope_type == ScopeType.TIME_WINDOW and run.expires_at and run.expires_at < time.time():
                run.status = RunStatus.COMPLETED
                run.completed_at = time.time()
                run.last_event = "expired_on_load"
                self._active_run_id = None
                self._save()
            elif run.started_at and (time.time() - run.started_at) > 3600:
                # 좀비 방지: 60분 넘게 ACTIVE 인 run 은 로드(재시작) 시 정리 (single_chain 포함)
                run.status = RunStatus.ABORTED
                run.completed_at = time.time()
                run.last_event = "stale_timeout_on_load"
                self._active_run_id = None
                self._save()
            else:
                self._active_run_id = active

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._runs_path), exist_ok=True)
            data = {
                "active_run_id": self._active_run_id,
                # 최근 50건만 보관 (디스크 누적 방지)
                "runs": [r.to_dict() for r in list(self._runs.values())[-50:]],
            }
            tmp = self._runs_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self._runs_path)
        except Exception as e:
            print(f"  ⚠️ [AutoRunner] runs 저장 실패 (무시): {e}")

    # ── 라이프사이클 ──────────────────────────────────────────────────────
    def create_pending(self, seed_goal: str, plan: AutonomousPlan) -> AutonomousRun:
        rid = uuid.uuid4().hex[:10]
        run = AutonomousRun(
            run_id=rid,
            status=RunStatus.PENDING_APPROVAL,
            seed_goal=seed_goal,
            plan=plan,
            created_at=time.time(),
        )
        self._runs[rid] = run
        self._save()
        return run

    async def approve(
        self,
        run_id: str,
        scope_type: str,
        time_window_sec: int = 0,
    ) -> Optional[AutonomousRun]:
        """승인 처리. scope_type=time_window 면 time_window_sec 사용.

        승인 직후 chain_starter 콜백 발화 — 통합 레이어가 실제 chain 시작.
        chain_starter 가 chain_id 반환하면 single_chain 스코프에 기록.
        """
        run = self._runs.get(run_id)
        if not run or run.status != RunStatus.PENDING_APPROVAL:
            return None

        # 다른 run 이 활성 상태면 abort (한 번에 하나만)
        if self._active_run_id and self._active_run_id != run_id:
            other = self._runs.get(self._active_run_id)
            if other and other.status == RunStatus.ACTIVE:
                other.status = RunStatus.ABORTED
                other.completed_at = time.time()
                other.last_event = "preempted_by_new_run"

        now = time.time()
        run.status = RunStatus.ACTIVE
        run.approved_at = now
        run.started_at = now
        run.scope_type = scope_type
        if scope_type == ScopeType.TIME_WINDOW:
            run.expires_at = now + max(60, int(time_window_sec or 3600))
            run.scope_value = str(int(run.expires_at))
        # SINGLE_CHAIN 의 scope_value 는 chain_starter 콜백 후 chain_id 로 채움
        self._active_run_id = run_id
        self._save()

        # chain 시작 콜백 발화
        if self._chain_starter:
            try:
                cid = await self._chain_starter(run)
                if cid and run.scope_type == ScopeType.SINGLE_CHAIN:
                    run.chain_id = cid
                    run.scope_value = cid
                    self._save()
            except Exception as e:
                print(f"  ⚠️ [AutoRunner] chain_starter 콜백 실패: {e}")
                run.last_event = f"chain_start_error: {e}"
                run.status = RunStatus.ABORTED
                run.completed_at = time.time()
                self._active_run_id = None
                self._save()
                return run
        return run

    def reject(self, run_id: str) -> Optional[AutonomousRun]:
        run = self._runs.get(run_id)
        if not run or run.status != RunStatus.PENDING_APPROVAL:
            return None
        run.status = RunStatus.REJECTED
        run.completed_at = time.time()
        self._save()
        return run

    def abort_active(self, reason: str = "user_stop") -> Optional[AutonomousRun]:
        if not self._active_run_id:
            return None
        run = self._runs.get(self._active_run_id)
        if not run:
            self._active_run_id = None
            self._save()
            return None
        run.status = RunStatus.ABORTED
        run.completed_at = time.time()
        run.last_event = f"aborted: {reason}"[:200]
        self._active_run_id = None
        self._save()
        return run

    def complete_active(self, summary: str = "") -> Optional[AutonomousRun]:
        if not self._active_run_id:
            return None
        run = self._runs.get(self._active_run_id)
        if not run:
            self._active_run_id = None
            self._save()
            return None
        run.status = RunStatus.COMPLETED
        run.completed_at = time.time()
        run.last_event = f"completed: {summary[:200]}" if summary else "completed"
        self._active_run_id = None
        self._save()
        return run

    # ── 조회/스코프 매칭 ───────────────────────────────────────────────────
    def get_active_run(self) -> Optional[AutonomousRun]:
        if not self._active_run_id:
            return None
        run = self._runs.get(self._active_run_id)
        if not run:
            self._active_run_id = None
            return None
        # 만료 lazy 검사
        if (
            run.scope_type == ScopeType.TIME_WINDOW
            and run.expires_at
            and run.expires_at < time.time()
        ):
            run.status = RunStatus.COMPLETED
            run.completed_at = time.time()
            run.last_event = "expired"
            self._active_run_id = None
            self._save()
            return None
        # 좀비 방지: scope 무관 하드 age 캡 (single_chain 은 expires_at 이 없어 위 만료검사를 비켜감)
        # get_active_run 은 advisor 틱마다 호출되므로 60분 넘게 ACTIVE 인 run 은 여기서 자가치유 abort.
        if (
            run.status == RunStatus.ACTIVE
            and run.started_at
            and (time.time() - run.started_at) > 3600
        ):
            run.status = RunStatus.ABORTED
            run.completed_at = time.time()
            run.last_event = "stale_timeout (no completion signal)"
            self._active_run_id = None
            self._save()
            return None
        if run.status != RunStatus.ACTIVE:
            self._active_run_id = None
            return None
        return run

    def get_run(self, run_id: str) -> Optional[AutonomousRun]:
        return self._runs.get(run_id)

    def is_task_in_active_scope(self, task: Dict[str, Any], schema: Any = None) -> bool:
        """dispatcher 의 _pause_with_telegram_gate 가 호출. 활성 run 의 스코프
        안이면 True → 게이트 자동 통과."""
        run = self.get_active_run()
        if not run:
            return False
        if run.scope_type == ScopeType.TIME_WINDOW:
            return True  # 창 안의 모든 task 자동 통과
        if run.scope_type == ScopeType.SINGLE_CHAIN:
            cid = ""
            if schema is not None:
                cid = (getattr(schema, "chain_id", "") or "")
            if not cid and isinstance(task, dict):
                cid = task.get("chain_id", "") or ""
            return bool(run.scope_value) and run.scope_value == cid
        return False

    def check_hard_block(self, action_blob: str) -> Optional[str]:
        """ActionPlan 내용(url+selector+text+value 등) 이 hard_block 키워드를
        포함하면 그 키워드 반환 (caller 가 abort 처리). 매칭은 소문자 substring."""
        run = self.get_active_run()
        if not run:
            return None
        if not action_blob:
            return None
        text_lc = action_blob.lower()
        for kw in run.plan.hard_blocks:
            if kw and kw.lower() in text_lc:
                run.hard_block_violations += 1
                run.last_event = f"hard_block: {kw}"
                self._save()
                return kw
        return None

    def increment_action(self) -> None:
        run = self.get_active_run()
        if run:
            run.actions_executed += 1
            # 매 액션마다 디스크 쓰기는 과도. 10건마다만 영속화.
            if run.actions_executed % 10 == 0:
                self._save()


# ── 헬퍼 — 제어실 top_goal 읽기 ─────────────────────────────────────────────

def read_top_goal() -> Dict[str, str]:
    """제어실의 top_goal + situation 을 읽어서 dict 반환.

    eidos_settings.json 우선, 없으면 schedule_planner_context.json.
    Returns: {"top_goal": str, "situation_external": str, "situation_internal": str}
    """
    out = {"top_goal": "", "situation_external": "", "situation_internal": ""}
    here = os.path.dirname(os.path.abspath(__file__))
    settings_path = os.path.join(here, "eidos_settings.json")
    ctx_path = os.path.join(here, "eidos_files", "schedule_planner_context.json")
    try:
        if os.path.exists(settings_path):
            with open(settings_path, "r", encoding="utf-8") as f:
                d = json.load(f) or {}
            out["top_goal"] = str(d.get("top_goal", "") or "").strip()
        if os.path.exists(ctx_path):
            with open(ctx_path, "r", encoding="utf-8") as f:
                ctx = json.load(f) or {}
            if not out["top_goal"]:
                out["top_goal"] = str(ctx.get("top_goal", "") or "").strip()
            out["situation_external"] = str(ctx.get("situation_external", "") or "").strip()
            out["situation_internal"] = str(ctx.get("situation_internal", "") or "").strip()
    except Exception as e:
        print(f"  ⚠️ [AutoRunner] top_goal 읽기 실패: {e}")
    return out


# ── Pre-flight planner — LLM 1회 호출 ───────────────────────────────────────

_PREFLIGHT_PROMPT = """사용자가 자율 실행을 요청했다. 아래 최상위 목표를 보고 EIDOS 가 자율적으로 수행할 실행 계획을 JSON 으로 만들어라.

[최상위 목표]
{seed_goal}

[현재 상황]
{situation}

[★ Execution-First 절대규칙 — 위반 시 plan 거부]
이 plan 은 반드시 **외부 세계로 나가는 실행 행위 (External Action)** 로 끝나야 한다.
분석/조사/요약/보고서/비교표 작성으로 끝나는 plan 은 거부된다 ("분석충 모드" 차단).

- 안전한 외부 행위 (execution_action 으로 권장):
  * 사용자 본인 사업 페이지에 컨텐츠 등록/제출 (예: 크몽 상품 페이지 등록, 포트폴리오 업로드)
  * 폼 작성 후 저장/제출 (예: 상세페이지 텍스트 폼 입력 후 [저장] 버튼)
  * 파일을 외부 서비스로 업로드 (예: Google Drive, GitHub)
  * 사용자 본인이 운영하는 채널의 컨텐츠 게시 (단 본인 채널만)

- 절대 외부 행위 (DEFAULT_HARD_BLOCKS — execution_action 에 넣으면 안 됨):
  * 결제·송금·구매·카드등록
  * 회원가입·계정등록·본인인증
  * 외부 SNS 게시·DM발송·댓글등록 (본인 채널 외)
  * 광고집행·유료광고

분석은 중간 stage 에 자유롭게 넣어도 된다. 단 chain 의 **종착점은 반드시 위 안전 외부 행위 1개**.
"선행 작업이 부족합니다" 류 회피 금지. 데이터 부족 시 "[가정: ~~]" 마커 박고 외부 행위까지 진행하라.

[작성 규칙]
- objective: 이번 실행에서 달성할 구체 목표 1줄 (60자 이내)
- stages: 3~6단계 실행 계획. **마지막 단계는 반드시 외부 실행 행위**. 각 단계는 검증 가능한 산출물 1개.
- execution_action: 마지막 stage 의 외부 행위 1줄 (**필수, 빈 문자열 절대 금지**)
  ★ **objective/stages 에 등록·제출·전송·발송·발행·올려·팔아·론칭·게시·업로드 동사가**
     **있으면 그 동사 그대로 옮겨라**. 빈 문자열로 두지 마라. 무조건 채워라.
  ★ 빈 문자열 = plan 즉시 거부 = chain 빌드 안 됨. 너의 plan 폐기됨. 무조건 채워라.
- execution_target: 행위 대상 URL 또는 플랫폼 (필수)
- sites: 방문할 외부 도메인 리스트. 한 사이트만 가도 1개 이상.
- expected_actions: {{"NAVIGATE": 정수, "READ_PAGE": 정수, "CLICK": 정수, "TYPE": 정수, "SUBMIT": 정수}} — 보수적 추정
- estimated_duration_min: 예상 소요 분 (정수, 5~120)
- 산출물은 외부 사이트에서 읽은 데이터 기반. LLM 환각 작문 금지.

[정상 예시 — task 별 execution_action 매핑]
✅ "AI 프롬프트 팩 크몽에 팔아줘"
   stages: ["1. 크몽 IT 카테고리 탑10 스캔", "2. 상세페이지 텍스트 초안 작성", "3. 크몽 등록 폼 제출"]
   execution_action: "크몽 IT/프로그래밍 카테고리에 신규 상품 페이지 등록"
   execution_target: "kmong.com/gigs/new"

✅ "USD/KRW 환율 정리해서 텔레그램으로 보내줘"
   stages: ["1. 환율 사이트 3곳 스캔 (네이버/investing/구글)", "2. 변동 요인 종합 메시지 작성", "3. 텔레그램 봇 채팅으로 전송"]
   execution_action: "사용자 텔레그램 채팅에 환율 요약 메시지 발송"
   execution_target: "telegram bot @user_chat"

✅ "회의록 작성해서 이메일로 발송"
   stages: ["1. 회의 노트 정리", "2. 메일 본문 작성", "3. 이메일 클라이언트에서 보내기 클릭"]
   execution_action: "참석자 이메일로 회의록 발송"
   execution_target: "Gmail/Outlook 작성 폼"

[❌ NEGATIVE 예시 — 거부됨]
❌ stages 끝이 "최종 보고서 작성" / "비교표 정리" / "요약 메시지 작성" / "정리 완료"
   → execution_action 비어있거나 "보고서 작성" 같이 분석성 → 즉시 거부
❌ execution_action="환율 정보 정리" — 이건 외부 행위가 아니라 내부 분석. 거부.
   올바른 형태: "텔레그램 봇으로 환율 요약 발송" (외부 발송 동사 명시)
❌ "정보가 부족합니다" / "선행 자료가 필요합니다" 회피성 stages → 거부.
   → 데이터 부족 시 [가정: ~~] 마커 박고 외부 행위까지 plan 짜라.

{retry_hint}

[출력 — JSON 만, 다른 텍스트 절대 금지]
{{"objective": "...", "stages": ["1. ...", "2. ...", "3. (외부 실행)"], "execution_action": "...", "execution_target": "...", "sites": ["..."], "expected_actions": {{"NAVIGATE": 20, "READ_PAGE": 15, "CLICK": 5, "TYPE": 3, "SUBMIT": 1}}, "estimated_duration_min": 45}}
"""


async def build_preflight_plan_async(
    seed_goal: str,
    situation: str = "",
    retry_hint: str = "",  # Phase 3-β 2026-05-01 — caller 재시도 시 LLM 에 박는 추가 지시.
    *,
    use_leverage: bool = False,  # 옵션 3 (2026-05-02) — leverage 분석 우선 시도.
    use_goal_model: bool = False,  # T3 (b) 2026-05-02 — Goal Model 시급 행동 hint 박기.
) -> Optional[AutonomousPlan]:
    """LLM 으로 사전 실행 계획 생성. 실패 시 None.

    호출자는 None 시 generic 플레이스홀더 plan 으로 폴백하거나 (작은 plan 으로
    일단 카드 띄우기) 사용자에게 "최상위 목표 설정해줘" 알림.

    [Phase 3-β 2026-05-01] retry_hint: caller 재시도 시 (이전 plan 거부 사유 등)
    LLM 프롬프트 끝에 박힘. 빈 문자열이면 prompt 변형 0 (기존 동작).

    [옵션 3 — 2026-05-02] use_leverage=True 시 leverage_strategy 1회 호출 →
    sphere 기반 plan 시도. 실패 (graph build / cycle 부재 / V2 BLOCK / token 발급
    실패) 시 None 안 받고 기존 LLM plan 경로로 회귀. 4중 안전망:
      1) V2 5중 ethics AND, 2) HMAC token (HIGH 24h timelock),
      3) ApprovalGateway 별도 risk 게이팅, 4) 1회 호출 cap.
    """
    if not (seed_goal or "").strip():
        return None

    # [Phase B3 — 2026-05-02] complexity classifier 자동 분기.
    #   use_leverage 명시 False 라도 task complex 면 자동 leverage 시도.
    #   classifier 가 simple 판정해도 사용자가 use_leverage=True 박았으면 그대로 시도.
    _auto_leverage = False
    if not use_leverage:
        try:
            from eidos_leverage_strategy import classify_task_complexity
            _complexity = classify_task_complexity(seed_goal)
            if _complexity == "complex":
                _auto_leverage = True
                print(f"  🧠 [Phase B3] task complex 자동 감지 — leverage 시도 (seed='{seed_goal[:40]}')")
        except Exception:
            pass

    # [옵션 3 + B3] leverage 우선 시도 — 실패 시 기존 LLM plan 경로 fallback
    _leverage_failed = False
    if use_leverage or _auto_leverage:
        try:
            from eidos_leverage_strategy import build_strategic_plan_from_leverage
            _lev_result = await build_strategic_plan_from_leverage(seed_goal, situation)
            if _lev_result is not None:
                _lev_plan, _lev_token = _lev_result
                _trigger = "explicit" if use_leverage else "auto-complex"
                print(
                    f"  🎯 [옵션 3] leverage 기반 plan 채택 ({_trigger}) — sphere={_lev_token.sphere_id} "
                    f"risk={_lev_token.risk_level} action='{_lev_plan.execution_action[:50]}'"
                )
                return _lev_plan
            _leverage_failed = True
            print("  ↩️ [옵션 3] leverage plan 실패 (graph/ethics/token) — 기존 LLM plan 으로 fallback")
        except Exception as _e_lev:
            _leverage_failed = True
            print(f"  ⚠️ [옵션 3] leverage 호출 예외 (무시 + LLM fallback): {_e_lev}")

    # [T3 (b) — 2026-05-02] Goal Model hint 자동 prepend.
    #   leverage 실패 했거나 use_goal_model 명시 시 — 제어실 데이터 기반 시급 행동 후보를
    #   seed_goal 에 prepend. LLM 이 hint 보고 plan 짤 때 외부 행위 매핑 정확도 ↑.
    #   회귀 안전 — Goal Model 데이터 비어있으면 빈 hint (변형 0).
    _auto_goal_model = (use_goal_model or _leverage_failed or _auto_leverage)
    if _auto_goal_model:
        try:
            from eidos_goal_model import get_top_actions_brief
            _gm_hint = get_top_actions_brief(top_k=3)
            if _gm_hint:
                seed_goal = f"{seed_goal}\n\n{_gm_hint}"
                _trigger = "explicit" if use_goal_model else ("post-leverage" if _leverage_failed else "auto-complex")
                print(f"  🎯 [T3-b] Goal Model hint 박음 ({_trigger}) — seed_goal +{len(_gm_hint)}자")
        except Exception as _e_gm:
            print(f"  ⚠️ [T3-b] Goal Model hint 예외 (무시): {_e_gm}")
    try:
        from eidos_patrol_agent import get_llm_response_async
        from llm_module import robust_json_parse
    except Exception:
        return None
    try:
        # retry_hint 가 있으면 [재시도 — 이전 plan 거부 사유] 단락으로 박음.
        # 비어있으면 placeholder 만 빈 줄로 치환 — prompt 변형 0 (회귀 안전).
        _retry_block = ""
        if retry_hint and retry_hint.strip():
            _retry_block = (
                f"[🔁 재시도 — 이전 plan 거부 사유 (반드시 반영하라)]\n"
                f"{retry_hint.strip()[:400]}"
            )
        prompt = _PREFLIGHT_PROMPT.format(
            seed_goal=seed_goal[:500],
            situation=(situation or "(없음)")[:300],
            retry_hint=_retry_block,
        )
        raw = await get_llm_response_async(
            prompt,
            response_mime_type="application/json",
            max_tokens=900,
            timeout=30,
        )
        if not raw or not raw.strip():
            return None
        data = robust_json_parse(raw)
        if not isinstance(data, dict) or not data:
            return None
        # 핵심 필드(objective/stages 중 하나) 가 모두 비면 LLM 응답으로 보지 않음
        # → caller(start_planning) 의 placeholder 폴백 분기로 양도. 여기서 seed_goal
        # 만 채운 "title-only" plan 을 만드는 건 환각 작문 위험.
        _obj = str(data.get("objective", "") or "").strip()
        _stages_raw = data.get("stages") or []
        if not _obj and not _stages_raw:
            return None
        # [Execution-First 2026-05-01] execution_action 필수 — 비면 분석충 plan 으로
        # 간주하고 거부. caller 에서 사용자에게 "외부 행위 없는 plan 거부" 카드 표시.
        _exec_action = str(data.get("execution_action", "") or "").strip()
        _exec_target = str(data.get("execution_target", "") or "").strip()
        # [EX1 2026-05-01] 자동 폴백 추출 — LLM 이 execution_action 빈 채로 응답하지만
        # objective/stages 에 외부 행위 동사 (등록/제출/전송/발송/발행/올려/팔아) 가
        # 있으면 자동 추출해서 plan 살림. 사용자 보고: 거부 카드 폭주 (line 172~179) 차단.
        if not _exec_action:
            try:
                from eidos_mission_schema import _detect_external_action as _detect_ea
                _stages_text = " ".join(str(s) for s in (_stages_raw or []))
                _blob = f"{_obj} {_stages_text}"[:1500]
                _auto_ea = _detect_ea(_blob)
                if _auto_ea and _auto_ea.get("action"):
                    _exec_action = str(_auto_ea.get("action", ""))[:200]
                    _exec_target = _exec_target or str(_auto_ea.get("target_hint", ""))[:200]
                    print(
                        f"  🔧 [EX1 2026-05-01] execution_action 자동 폴백 추출 — "
                        f"action='{_exec_action}' (verb='{_auto_ea.get('verb_matched','')}', "
                        f"objective+stages 에서 외부 행위 동사 감지). plan 살림."
                    )
            except Exception as _e_ex1:
                print(f"  ⚠️ [EX1] 자동 폴백 실패 (무시): {_e_ex1}")
        if not _exec_action:
            print(
                f"  🚫 [Execution-First] plan 거부 — execution_action 비어있음. "
                f"분석/요약/보고서로 끝나는 plan 차단. (objective='{_obj[:50]}')"
            )
            return None
        plan = AutonomousPlan(
            objective=_obj[:200] or seed_goal[:80],
            stages=[str(s)[:200] for s in _stages_raw][:6],
            sites=[str(s)[:80] for s in (data.get("sites") or [])][:8],
            expected_actions={
                str(k)[:30]: int(v)
                for k, v in (data.get("expected_actions") or {}).items()
                if isinstance(v, (int, float)) and 0 <= int(v) < 1000
            },
            estimated_duration_min=max(5, min(120, int(data.get("estimated_duration_min", 30) or 30))),
            execution_action=_exec_action[:200],
            execution_target=_exec_target[:200],
        )
        return plan
    except Exception as e:
        print(f"  ⚠️ [AutoRunner] preflight LLM 실패: {e}")
        return None


# ── 승인 카드 텍스트 빌더 (HTML — Telegram parse_mode=HTML) ──────────────────

def _esc(s: Any) -> str:
    return html.escape(str(s or ""), quote=False)


def build_approval_card_text(run: AutonomousRun) -> str:
    p = run.plan
    sites = " · ".join(p.sites) if p.sites else "(미정)"
    if p.expected_actions:
        actions = " / ".join(f"{k}≈{v}" for k, v in p.expected_actions.items())
    else:
        actions = "(미정)"
    blocks_short = ", ".join(p.hard_blocks[:6])
    if len(p.hard_blocks) > 6:
        blocks_short += f" 외 {len(p.hard_blocks) - 6}종"
    stages_lines = "\n".join(f"  {_esc(s)}" for s in p.stages) if p.stages else "  (계획 미생성)"
    # [Execution-First 2026-05-01] chain 마지막 외부 행위 가시화 — 사용자가
    # "이 chain 이 결국 무엇을 외부에 던지는지" 승인 전에 명확히 확인.
    _exec_line = ""
    if p.execution_action:
        _tgt = f" → <code>{_esc(p.execution_target)}</code>" if p.execution_target else ""
        _exec_line = (
            f"\n<b>🚀 최종 외부 행위</b>\n  {_esc(p.execution_action)}{_tgt}\n"
        )
    text = (
        f"🤖 <b>EIDOS 자율 실행 승인 요청</b>\n"
        f"\n"
        f"<b>🎯 목표</b>\n  {_esc(p.objective or run.seed_goal)}\n"
        f"\n"
        f"<b>📋 실행 계획</b>\n{stages_lines}\n"
        f"{_exec_line}"
        f"\n"
        f"<b>🌐 방문 사이트:</b> {_esc(sites)}\n"
        f"<b>🖱️ 예상 액션:</b> {_esc(actions)}\n"
        f"<b>⏱️ 예상 소요:</b> 약 {p.estimated_duration_min}분\n"
        f"<b>💰 비용 cap:</b> {p.budget_krw:,}원 (LLM 호출만)\n"
        f"\n"
        f"<b>🔒 자동 차단 (이번 실행에서 수행 안 함)</b>\n"
        f"  {_esc(blocks_short)}\n"
        f"\n"
        f"<b>🛑 긴급 중단:</b> /stop_auto\n"
        f"\n"
        f"<b>승인 스코프 선택</b>"
    )
    return text


# ── 외부에서 사용할 진입점 ───────────────────────────────────────────────────

# ── core 통합 (Phase 1B 2026-04-27) ─────────────────────────────────────────
# bootstrap_with_core(core) 가 core.start_schedule_monitor 시점에 호출되어:
#   1. chain_starter 콜백 등록 — 승인된 run 의 plan 으로 explore chain 생성·시작
#   2. PipelineOrchestrator chain_complete / chain_failed 리스너 등록
#   3. 완료/실패 시 텔레그램 요약 카드 발송
# 회귀 0: 모든 외부 의존(orchestrator/dispatcher/telegram_bot) 은 try/except 로
# 안전 폴백 — 없어도 매니저 자체는 동작.

def _build_explore_chain_stages(plan: "AutonomousPlan") -> List[Any]:
    """plan 을 3-stage chain 으로 변환 (Execution-First 2026-05-01):
       (1) browser_read 'explore' 단계 — 실제 사이트 탐색·자료 수집
       (2) write 'draft_for_action' 단계 — 다음 stage 외부 행위에 직접 사용될 컨텐츠 작성
       (3) browser_act 'execute_external_action' 단계 — plan.execution_action 외부 행위 실행

    이전(2-stage explore + summarize) 과 달리 chain 의 종착점이 외부 실행 행위로 강제됨.
    plan.execution_action 비어있으면 ValueError raise — caller(_start_explore_chain_for_run)
    가 catch 후 사용자에게 "외부 행위 없는 plan 거부" 카드 발송.

    분석/조사/요약/보고서로 끝나는 chain 자체를 코드에서 차단해 "분석충 모드" 원천 봉쇄.
    """
    from eidos_mission_chain import StageSpec
    # [Execution-First] execution_action 필수 게이트 — 비면 chain 거부.
    _exec_action = (plan.execution_action or "").strip()
    if not _exec_action:
        raise ValueError(
            "execution_action_required: plan.execution_action 이 비어있어 chain 빌드 거부. "
            "분석/요약/보고서로 끝나는 plan 은 차단됩니다 (Execution-First)."
        )
    _exec_target = (plan.execution_target or "").strip()
    sites_hint = ", ".join(plan.sites[:5]) if plan.sites else "(검색 엔진 자율 결정)"
    stages_hint = "\n".join(f"  {s}" for s in plan.stages) if plan.stages else "  (계획 미생성)"
    explore_prompt = (
        f"{plan.objective}\n"
        f"\n방문할 사이트: {sites_hint}\n"
        f"세부 단계:\n{stages_hint}\n"
        f"\n각 단계별 검증 가능한 산출물을 모으고 explore_report.md 로 저장하라. "
        f"외부 사이트에서 읽은 데이터만 인용 — LLM 환각 작문 금지.\n"
        f"\n⚠️ 이 stage 는 다음 stage 의 외부 행위 '{_exec_action}' 를 위한 자료 수집이다. "
        f"분석으로 끝내지 말고 다음 stage 에서 즉시 사용할 형태로 정리하라."
    )
    draft_prompt = (
        f"위 탐색 결과를 바탕으로, 다음 stage 의 외부 실행 행위에 직접 사용될 컨텐츠를 작성하라.\n"
        f"\n[다음 stage 외부 행위] {_exec_action}\n"
        f"[대상 URL/플랫폼] {_exec_target or '(미정)'}\n"
        f"\n작성 규칙:\n"
        f"- 단독 보고서/요약/분석 금지. 다음 stage 가 폼에 직접 입력하거나 등록할 수 있는 형태.\n"
        f"- 데이터 부족 시 '[가정: ~~]' 마커 박고 진행. '선행 작업이 부족합니다' 회피 금지.\n"
        f"- 결과물은 action_payload.md 에 저장 — 다음 stage 가 이 파일을 그대로 외부 행위에 사용."
    )
    execute_prompt = (
        f"이전 stage 의 action_payload.md 를 사용해 외부 실행 행위를 수행하라.\n"
        f"\n[수행할 외부 행위] {_exec_action}\n"
        f"[대상 URL/플랫폼] {_exec_target or '(미정)'}\n"
        f"\n수행 절차:\n"
        f"1. 대상 URL 로 NAVIGATE\n"
        f"2. action_payload.md 내용을 폼/필드에 TYPE\n"
        f"3. 등록/제출/저장 버튼 CLICK 또는 SUBMIT\n"
        f"4. 등록 결과 (성공 메시지/등록된 페이지 URL) 를 action_result.md 에 저장\n"
        f"\n⚠️ 결제·송금·SNS게시·DM발송·계정등록 등 hard_block 행위는 절대 수행 금지. "
        f"중간 단계에서 hard_block 키워드 감지 시 즉시 abort."
    )
    return [
        StageSpec(
            name="explore", stage_index=0,
            task_type="browser_read", verb="READ",
            prompt_template=explore_prompt,
            depends_on=[], input_keys=[],
            output_keys=["explore_report"],
            auto_approve=True,
            description=f"자율 탐색: {plan.objective[:60]}",
            expected_duration_sec=max(60, plan.estimated_duration_min * 30),  # 1/2 budget
            risk_level="low",
        ),
        StageSpec(
            name="draft_for_action", stage_index=1,
            task_type="write", verb="COMPOSE",
            prompt_template=draft_prompt,
            depends_on=[0],
            input_keys=["explore.explore_report"],
            output_keys=["action_payload"],
            auto_approve=True,
            description=f"외부 행위용 컨텐츠 작성: {_exec_action[:50]}",
            expected_duration_sec=180,
            risk_level="low",
        ),
        StageSpec(
            name="execute_external_action", stage_index=2,
            task_type="browser_act", verb="EXECUTE",
            prompt_template=execute_prompt,
            depends_on=[1],
            input_keys=["draft_for_action.action_payload"],
            output_keys=["action_result"],
            auto_approve=True,
            description=f"외부 실행: {_exec_action[:60]}",
            expected_duration_sec=max(60, plan.estimated_duration_min * 30),  # 1/2 budget
            risk_level="medium",  # 외부 행위라 medium
        ),
    ]


async def _start_explore_chain_for_run(
    core: Any,
    run: "AutonomousRun",
    _retry_n: int = 0,  # Phase 3-β 2026-05-01 — caller 재시도 카운터 (max=1)
) -> Optional[str]:
    """승인된 run 으로 explore chain 생성 + dispatcher 통해 launch.

    1. PipelineOrchestrator.create_chain_from_stages — 2-stage chain 등록
    2. orchestrator.approve_chain — 상태 APPROVED
    3. ActionDispatcher._instance 발견되면: _pending_chain = chain 후
       approve_pending_chain() await — 기존 GUI 승인 경로와 동일 흐름
    4. dispatcher 미발견 시: 그냥 chain.chain_id 만 반환 (registry 등록은 됨)

    [Phase 3-β 2026-05-01] _retry_n: caller 재시도 카운터. ValueError catch 시
    plan LLM 1회 재호출 (retry_hint 박음) → 새 plan 으로 재시도. 무한루프 방지 max=1.
    """
    try:
        from eidos_mission_chain import get_orchestrator
    except Exception as e:
        print(f"  ⚠️ [AutoRunner] mission_chain 로드 실패: {e}")
        return None
    orch = get_orchestrator(core)
    try:
        # [옵션 3 — 2026-05-02] leverage token verify hook — plan 에 token attached 시만 검증.
        #   verify 실패 (HMAC 위조 / TTL 만료 / HIGH 24h timelock) → ValueError 로 기존 거부 경로 재사용.
        _lev_token = getattr(run.plan, "_leverage_token", None)
        if _lev_token is not None:
            try:
                from eidos_leverage_token import TokenManager as _LevTM
                _valid, _reason = _LevTM().verify_token(_lev_token)
                if not _valid:
                    print(f"  🔒 [옵션 3] leverage token verify FAIL — {_reason}")
                    raise ValueError(f"leverage_token_invalid: {_reason}")
                print(
                    f"  ✅ [옵션 3] leverage token verify OK — sphere={_lev_token.sphere_id} "
                    f"risk={_lev_token.risk_level}"
                )
            except ValueError:
                raise
            except Exception as _e_tv:
                print(f"  ⚠️ [옵션 3] token verify 예외 (무시 + 기존 경로 진행): {_e_tv}")
        # [Proactive Advisor 2026-05-17] advisor_draft = 2단 경량 체인
        #   (탐색→초안). execution-first(ValueError/retry) 미적용 — 외부행동
        #   단계가 없어 분석충 차단 규칙 대상이 아니다. graceful 폴백.
        if getattr(run.plan, "mode", "") == "advisor_draft":
            try:
                from eidos_proactive_advisor import build_advisor_stages
                stages = build_advisor_stages(run.plan)
            except Exception as _e_adv:
                print(f"  ⚠️ [AutoRunner] advisor 2단 빌더 실패 → 거부: {_e_adv}")
                raise ValueError(f"advisor_stages_failed: {_e_adv}")
        else:
            stages = _build_explore_chain_stages(run.plan)
        chain = orch.create_chain_from_stages(
            stages=stages,
            original_prompt=run.seed_goal,
            display_name=f"자율탐색 — {(run.plan.objective or run.seed_goal)[:30]}",
        )
        # [좀비 방지 — 2026-06-11] 0-step 체인(born-dead)은 chain_failed 이벤트를
        #   안 쏴서 _on_chain_failed 리스너가 안 불리고 run 이 영원히 ACTIVE 로 남는다.
        #   생성 즉시 감지해 run 을 abort(저장·active_run_id 정리 포함)하고 종료.
        _nsteps = len(getattr(chain, "steps", None) or getattr(chain, "stages", None) or [])
        if _nsteps == 0:
            print("  🚫 [AutoRunner] 0-step 체인 — born-dead 차단, run abort")
            try:
                AutonomousRunManager.get().abort_active(reason="empty_chain_0_steps")
            except Exception:
                pass
            return None
        # 자율 실행은 사전 승인을 받았으므로 chain 도 즉시 APPROVED.
        orch.approve_chain(chain.chain_id)
    except ValueError as ve:
        # [Proactive Advisor 2026-05-17 #2] advisor_draft 는 execution-first
        #   재시도(preflight LLM + execution_action 강제) 대상이 아니다.
        #   여기로 빠지면 advisor 의 "초안까지만" 의도와 정반대 3단 외부행동
        #   plan 이 생성됨 → 클린 abort + 사용자 통지 후 즉시 종료.
        if getattr(run.plan, "mode", "") == "advisor_draft":
            print(f"  🚫 [Advisor] 2단 체인 빌드 실패 — execution-first "
                  f"재시도 안 함, 클린 종료: {ve}")
            # [좀비 방지 — 2026-06-11] run.status 직접대입은 _save()·_active_run_id
            #   정리를 빼먹어 디스크에 좀비로 남는다. abort_active()로 일괄 처리.
            try:
                AutonomousRunManager.get().abort_active(
                    reason=f"advisor_chain_failed: {ve}"[:120])
            except Exception:
                try:
                    run.status = RunStatus.ABORTED
                    run.completed_at = time.time()
                    run.last_event = f"advisor_chain_failed: {ve}"[:200]
                except Exception:
                    pass
            try:
                import eidos_proactive_advisor as _adv
                _adv._push_notice(
                    "⚠️ 자료수집 준비 중 문제가 생겼어요. 잠시 후 다시 "
                    "제안드릴게요. (외부 행동은 일어나지 않았습니다)")
            except Exception:
                pass
            return None
        # [Execution-First 2026-05-01] execution_action 비어있음 — 분석충 plan 거부.
        _msg = str(ve)
        # [Phase 3-β 2026-05-01] 1회 재시도 가드 — plan LLM 재호출 후 ValueError 또
        # 발생하면 거부 카드. 무한루프 방지 max=1.
        if _retry_n < 1:
            print(
                f"  🔁 [Phase 3-β] Execution-First 거부 — plan LLM 재시도 "
                f"({_retry_n + 1}/1)"
            )
            try:
                _hint = (
                    "🚨 이전 plan 의 execution_action 필드가 빈 문자열이라 거부됐다. "
                    "★★★ JSON 응답에서 execution_action 키를 빈 문자열로 절대 두지 마라. ★★★ "
                    "objective 에 '등록/제출/전송/발송/발행/올려/팔아' 같은 동사가 있으면 그 "
                    "동사 그대로 (예: \"크몽 상품 등록\") execution_action 에 박아라. "
                    "동사 없으면 외부 행위 단계 추가하고 그것을 박아라. "
                    "빈 문자열 = 너의 plan 또 폐기됨. 분석/조사/요약/보고서로 끝나면 또 거부. "
                    "데이터 부족 시 [가정: ~~] 마커 박더라도 외부 행위까지 plan 짜라."
                )
                _new_plan = await build_preflight_plan_async(
                    seed_goal=run.seed_goal,
                    situation=getattr(run, "situation", "") or "",
                    retry_hint=_hint,
                )
                if _new_plan and (_new_plan.execution_action or "").strip():
                    run.plan = _new_plan
                    print(
                        f"  ✅ [Phase 3-β] 재시도 plan 수신 — execution_action="
                        f"'{_new_plan.execution_action[:50]}' → chain 빌드 재시도"
                    )
                    return await _start_explore_chain_for_run(core, run, _retry_n=_retry_n + 1)
                else:
                    print(
                        f"  🚫 [Phase 3-β] 재시도 plan 도 execution_action 누락 "
                        f"(plan={_new_plan!r}) → 거부 카드 발송"
                    )
            except Exception as _re:
                print(f"  ⚠️ [Phase 3-β] 재시도 중 예외 (무시, 거부 카드 진행): {_re}")
        # 거부 카드 (최초 retry_n=1 또는 재시도 후에도 실패)
        print(f"  🚫 [Execution-First] chain 거부 (retry_n={_retry_n}): {_msg}")
        try:
            from eidos_telegram_bot import get_bot
            _bot = get_bot()
            if _bot.is_configured():
                _retry_note = (
                    "\n\n<i>(plan LLM 재시도 1회 후에도 외부 행위 누락 — 거부)</i>"
                    if _retry_n >= 1 else ""
                )
                _card = (
                    f"<b>목표:</b> {_esc(run.plan.objective or run.seed_goal)}\n\n"
                    f"<b>거부 사유:</b> plan 의 마지막에 외부 실행 행위가 없습니다.\n"
                    f"분석/요약/보고서 작성으로 끝나는 plan 은 차단됩니다 (Execution-First).\n\n"
                    f"<b>해결:</b> 다시 시도 시 plan 에 'execution_action' (예: 크몽 상품 등록, "
                    f"포트폴리오 업로드, 폼 제출) 이 명시되어야 합니다.{_retry_note}"
                )
                await _bot.send_notification(
                    _card,
                    title="🚫 자율 실행 거부 — 외부 행위 없는 plan",
                )
        except Exception as _be:
            print(f"  ⚠️ [Execution-First] 거부 카드 발송 실패(무시): {_be}")
        return None
    except Exception as e:
        print(f"  ⚠️ [AutoRunner] chain 생성/승인 실패: {e}")
        return None

    # dispatcher 통해 첫 stage launch (있을 때만)
    try:
        from eidos_action_dispatcher import ActionDispatcher
        disp = getattr(ActionDispatcher, "_instance", None)
        if disp is not None:
            disp._pending_chain = chain
            ok = await disp.approve_pending_chain()
            if not ok:
                print(f"  ⚠️ [AutoRunner] approve_pending_chain 실패 — 폴백으로 chain_id 만 반환")
        else:
            print("  ℹ️ [AutoRunner] ActionDispatcher._instance 미생성 — chain registry 등록만 완료")
    except Exception as e:
        print(f"  ⚠️ [AutoRunner] dispatcher launch 실패 (chain 등록은 완료): {e}")

    return chain.chain_id


async def _notify_run_end(run: Optional["AutonomousRun"], success: bool) -> None:
    """run 종료 시 텔레그램 요약 카드 발송 (best-effort)."""
    if run is None:
        return
    try:
        from eidos_telegram_bot import get_bot
        bot = get_bot()
        if not bot.is_configured():
            return
        emoji = "✅" if success else "🛑"
        title = "자율 실행 완료" if success else "자율 실행 중단"
        duration_min = 0
        if run.started_at and run.completed_at:
            duration_min = max(0, int((run.completed_at - run.started_at) / 60))
        text = (
            f"{emoji} <b>{_esc(title)}</b>\n"
            f"\n"
            f"<b>목표:</b> {_esc(run.plan.objective or run.seed_goal)}\n"
            f"<b>스코프:</b> <code>{_esc(run.scope_type)}</code>\n"
            f"<b>실행된 액션:</b> {run.actions_executed}건\n"
            f"<b>hard-block 위반:</b> {run.hard_block_violations}건\n"
            f"<b>소요:</b> 약 {duration_min}분\n"
        )
        if run.last_event:
            text += f"<b>마지막 이벤트:</b> {_esc(run.last_event[:120])}\n"
        await bot.send_notification(text, title="")
    except Exception as e:
        print(f"  ⚠️ [AutoRunner] 완료 알림 발송 실패 (무시): {e}")


def bootstrap_with_core(core: Any) -> bool:
    """core 부팅 시점에 호출되는 통합 진입점.

    chain_starter 콜백 + PipelineOrchestrator chain_complete/failed 리스너
    등록. 이미 부트스트랩됐으면 no-op (idempotent).

    Returns: 부트스트랩 성공 여부 (False 면 핵심 통합 실패 — Telegram 카드는
    여전히 발송되지만 chain 자동 시작은 미동작).
    """
    mgr = AutonomousRunManager.get()
    # idempotent — 이미 등록됐으면 skip
    if getattr(mgr, "_bootstrapped", False):
        return True

    # chain_starter — closure 로 core 캡처
    async def _starter(run: "AutonomousRun") -> Optional[str]:
        return await _start_explore_chain_for_run(core, run)

    mgr.set_chain_starter(_starter)
    mgr.set_notifier(_notify_run_end)

    # ── [Loop F closure 2026-05-15] episode 경계 = chain 종료 ──────────────
    # verifier 가 wire3_trajectory_collect 로 active trajectory 에 Transition 을
    # 모으지만 close_trajectory() 호출자가 어디에도 없어 ReplayBuffer 가 영원히
    # 0 → maybe_auto_train_pvn 영구 skip (학습 루프 미폐쇄)였다. chain 완료/실패
    # 가 곧 episode 경계이므로 두 리스너 최상단에서 무조건 close (transition 0
    # 이면 close_trajectory 가 None 반환 no-op — 자기 게이팅). close 내부에서
    # ReplayBuffer.push + maybe_auto_train_pvn() 가 연쇄 발화 → 루프 닫힘.
    # 모든 chain 대상(자율/수동 무관) — autonomous run 스코프와 독립.
    def _close_episode(_done: bool, _chain, _reason: str = "") -> None:
        try:
            from eidos_trajectory import close_trajectory, active_trajectory_length
            _n = active_trajectory_length()
            if _n <= 0:
                return
            close_trajectory(
                done=_done,
                episode_meta={
                    "chain_id": getattr(_chain, "chain_id", ""),
                    "chain_name": getattr(_chain, "name", ""),
                    "outcome": "complete" if _done else "failed",
                    "reason": str(_reason)[:120],
                },
            )
            print(
                f"  🧠 [Loop F] episode close — {_n} transitions → ReplayBuffer "
                f"({'완료' if _done else '실패'})"
            )
        except Exception as _e_ce:
            print(f"  ⚠️ [Loop F] episode close 실패(무시): {_e_ce}")

        # ── [라이브 배선 v2 shadow / Phase 2 2026-05-15] 스펙 준수 C 조인 ──
        # chain.original_prompt 에서 canonical 추출(extract_objects +
        # verb_engine.extract_verbs_from_text — auto_log 과 동일 함수, 병렬
        # 추출기 X). **Fork C**: matched_in_matrix 필터 완화 — 전 토큰
        # (name+role) 사용 (분포는 wide net 을 원함; 노이즈는 다운스트림
        # 히스테리시스+min_separation 0.8 이 흡수). reg.record_episode 가
        # **Fork B 누설-안전 순서**(현 cooc→attrs→record_outcome, 그 다음
        # cooc 갱신) 수행 → 분기 활성. wire7_live 기본 OFF → 즉시 no-op.
        # **별도 try/except** — 어떤 실패도 위 Phase F close / verifier /
        # dispatcher 로 전파 안 됨. persist=True 로 재시작 간 복리(1회 load).
        try:
            from eidos_verb_specialization import live_enabled, get_registry
            if live_enabled():
                _txt = str(getattr(_chain, "original_prompt", "") or "")
                if _txt.strip():
                    from eidos_verb_object_extractor import extract_objects
                    from eidos_verb_engine import get_verb_engine
                    _reg = get_registry()
                    # 재시작 간 cooc 복리 — 프로세스당 1회 디스크 load
                    if not getattr(_reg, "_cooc_loaded", False):
                        try:
                            _reg.cooc().load()
                        except Exception:
                            pass
                        try:
                            _reg._cooc_loaded = True  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    _fillers: Dict[str, str] = {}
                    for _t in (extract_objects(_txt) or []):  # Fork C: 전 토큰
                        _r = (getattr(_t, "role", "") or "").strip()
                        _nm = (getattr(_t, "name", "") or "").strip()
                        if _r and _nm:
                            _fillers[_r] = _nm
                    _eng = get_verb_engine()
                    _vjps = []
                    for _va in (
                        _eng.extract_verbs_from_text(_txt) if _eng else []
                    ):
                        _jp = (getattr(_va, "jp", "") or "").strip()
                        if _jp:
                            _vjps.append(_jp)
                    if _fillers and _vjps:
                        _cid = str(getattr(_chain, "chain_id", ""))
                        _n = _reg.record_episode(
                            _vjps, _fillers, passed=_done,
                            episode_id=_cid, persist=True,
                        )
                        print(
                            f"  🌱 [Spec-shadow/P2] episode 귀속 — "
                            f"verb×{_n} filler×{len(_fillers)} "
                            f"({'PASS' if _done else 'FAIL'})"
                        )
        except Exception as _e_sp:
            print(f"  ⚠️ [Spec-shadow] 무시(graceful): {_e_sp}")

    # PipelineOrchestrator chain 이벤트 리스너 등록
    def _on_chain_complete(chain) -> None:
        _close_episode(True, chain)
        run = mgr.get_active_run()
        if run is None:
            return
        # chain_id 매칭 — single_chain 스코프만 자동 종료
        if run.scope_type == ScopeType.SINGLE_CHAIN and run.chain_id != chain.chain_id:
            return
        # time_window 스코프는 chain 끝나도 창이 살아있으면 유지 (다음 chain 들어올 수 있음).
        # 단 chain_id 가 매칭되고 single_chain 이면 종료.
        if run.scope_type == ScopeType.SINGLE_CHAIN:
            summary = f"chain={getattr(chain, 'name', '?')}, stages={len(getattr(chain, 'stages', []) or [])}"
            completed = mgr.complete_active(summary=summary)
            try:
                import asyncio as _a
                _a.ensure_future(_notify_run_end(completed, success=True))
            except Exception:
                pass

    def _on_chain_failed(chain, reason) -> None:
        _close_episode(False, chain, str(reason))
        run = mgr.get_active_run()
        if run is None:
            return
        if run.scope_type == ScopeType.SINGLE_CHAIN and run.chain_id != chain.chain_id:
            return
        if run.scope_type == ScopeType.SINGLE_CHAIN:
            aborted = mgr.abort_active(reason=f"chain_failed: {str(reason)[:120]}")
            try:
                import asyncio as _a
                _a.ensure_future(_notify_run_end(aborted, success=False))
            except Exception:
                pass

    try:
        from eidos_mission_chain import get_orchestrator
        orch = get_orchestrator(core)
        orch.add_event_listener("chain_complete", _on_chain_complete)
        orch.add_event_listener("chain_failed", _on_chain_failed)
    except Exception as e:
        print(f"  ⚠️ [AutoRunner] orchestrator 리스너 등록 실패 (chain 자동 종료 미동작): {e}")
        mgr._bootstrapped = True
        return False

    mgr._bootstrapped = True
    print("✅ [AutoRunner] core 통합 완료 — chain_starter + 완료/실패 리스너 등록")
    return True


async def start_planning(seed_goal: Optional[str] = None) -> Optional[AutonomousRun]:
    """텔레그램 /explore 또는 사용자 트리거 시 호출되는 엔트리포인트.

    1. seed_goal 미지정 시 제어실 top_goal 읽기 (없으면 None 반환 → caller 가 안내)
    2. LLM 으로 사전 계획 생성
    3. AutonomousRun 을 PENDING_APPROVAL 로 등록
    4. AutonomousRun 반환 — caller 가 승인 카드 발송
    """
    mgr = AutonomousRunManager.get()
    if not seed_goal:
        ctx = read_top_goal()
        seed_goal = ctx.get("top_goal", "")
        situation = ctx.get("situation_external", "") or ctx.get("situation_internal", "")
    else:
        situation = ""
    if not (seed_goal or "").strip():
        return None
    plan = await build_preflight_plan_async(seed_goal, situation)
    if plan is None:
        # 폴백 — LLM 실패해도 placeholder plan 으로 일단 카드 띄움 (사용자가 ❌ 가능)
        plan = AutonomousPlan(
            objective=seed_goal[:80],
            stages=["1. 최상위 목표 분석 후 자율 탐색 시작"],
            sites=[],
            expected_actions={},
            estimated_duration_min=30,
        )
    return mgr.create_pending(seed_goal=seed_goal, plan=plan)
