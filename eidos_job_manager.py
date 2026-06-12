# eidos_job_manager.py
# [Wave11 2026-05-28] Background job 통합 관리.
#
# 기존 EIDOS 에는 여러 자율 작업 흐름이 따로 살아있음:
#   - long_running_goals (LRG) — Phase 2A/B/C·60s tick·attempts 누적
#   - autonomous_workflow (Wave 7-B) — milestone 별 chain·1회 실행
#   - GoalTree milestone — 진행률 추적
#   - 정기 task (RECURRENT_SCHEDULE) — schedule 자동 trigger
#
# 통합 없이 따로 돌아서 "지금 뭐 하고 있어" 가 안 보임.
#
# 이 모듈은 한 가지 통합 Job 추상:
#   - Job 1개 = milestone chain · LRG · 정기 task 등 무엇이든
#   - status (queued/running/paused/completed/failed/cancelled)
#   - priority·예상 시간·진행률·last_progress_at
#   - jsonl 로그·"지금 뭐 하나" 자연어 조회 가능
#
# 사용 흐름:
#   - autonomous_workflow chain 실행 시 Job 자동 생성·status running
#   - chain 끝나면 Job 도 completed
#   - 60s tick 에서 queued Job 중 priority 높은 거 자동 pick
#   - 사용자가 "지금 뭐 하고 있어" → 활성 Job list 보고

from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional


_JOBS_DIR = os.path.join("eidos_files", "jobs")
_ACTIVE_JOBS_PATH = os.path.join(_JOBS_DIR, "active_jobs.json")
_COMPLETED_LOG_PATH = os.path.join(_JOBS_DIR, "completed_jobs.jsonl")
_MAX_COMPLETED_LOG = 2 * 1024 * 1024


JOB_STATUSES = (
    "queued", "running", "paused",
    "completed", "failed", "cancelled",
)

JOB_KINDS = (
    "milestone_chain",      # autonomous_workflow chain
    "long_running_goal",    # LRG (Phase 2A)
    "recurrent",            # 정기 task
    "manual",               # 사용자 직접 트리거
)


@dataclass
class Job:
    """통합 background job."""
    job_id: str = ""
    title: str = ""
    kind: str = "manual"
    stage_id: str = ""
    milestone_id: str = ""

    status: str = "queued"
    priority: int = 5         # 0~10 (10=긴급)

    created_at: str = ""
    started_at: str = ""
    completed_at: str = ""
    last_progress_at: str = ""

    n_attempts: int = 0
    last_chain_id: str = ""
    progress_pct: float = 0.0
    result_preview: str = ""

    # 정기 task 용
    is_recurring: bool = False
    next_run_at: str = ""
    recurrence_pattern: str = ""    # "daily"·"weekly"·"monthly"

    notes: str = ""

    def serialize(self) -> dict:
        return asdict(self)

    @classmethod
    def deserialize(cls, data: dict) -> "Job":
        if not isinstance(data, dict):
            data = {}
        out = cls()
        for f in (
            "job_id", "title", "kind", "stage_id", "milestone_id",
            "status", "created_at", "started_at", "completed_at",
            "last_progress_at", "last_chain_id", "result_preview",
            "next_run_at", "recurrence_pattern", "notes",
        ):
            v = data.get(f)
            if v is not None:
                setattr(out, f, str(v) if not isinstance(v, str) else v)
        try:
            out.priority = int(data.get("priority", 5))
        except Exception:
            out.priority = 5
        try:
            out.n_attempts = int(data.get("n_attempts", 0))
        except Exception:
            out.n_attempts = 0
        try:
            out.progress_pct = float(data.get("progress_pct", 0.0))
        except Exception:
            out.progress_pct = 0.0
        out.is_recurring = bool(data.get("is_recurring", False))
        if out.status not in JOB_STATUSES:
            out.status = "queued"
        if out.kind not in JOB_KINDS:
            out.kind = "manual"
        return out


# ── 저장·로드 ────────────────────────────────────────────────────────
def _ensure_dir() -> None:
    try:
        os.makedirs(_JOBS_DIR, exist_ok=True)
    except Exception:
        pass


def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _new_id() -> str:
    import uuid
    return "job_" + uuid.uuid4().hex[:8]


def load_jobs() -> list[Job]:
    """활성 job list 디스크 로드."""
    _ensure_dir()
    if not os.path.exists(_ACTIVE_JOBS_PATH):
        return []
    try:
        with open(_ACTIVE_JOBS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
    except Exception as e:
        print(f"[job_manager] load 실패 (graceful·빈 list): {e}")
        return []
    raw = data.get("jobs") or []
    if not isinstance(raw, list):
        return []
    out = []
    for j in raw:
        try:
            out.append(Job.deserialize(j))
        except Exception:
            continue
    return out


def save_jobs(jobs: list[Job]) -> bool:
    """활성 job list 디스크 저장. completed/failed/cancelled 도 active 에 보관 (7일).

    7일 지난 종료 job 은 completed_log 로 archive·active 에서 제거.
    """
    _ensure_dir()
    nv = _dt.datetime.now()
    keep_active: list = []
    archive: list = []
    for j in jobs:
        if j.status in ("queued", "running", "paused"):
            keep_active.append(j)
            continue
        # 종료 상태 — 7일 이내면 active 에 보관·이후 archive
        try:
            ref_ts = j.completed_at or j.last_progress_at or j.created_at
            if ref_ts:
                ref_dt = _dt.datetime.fromisoformat(ref_ts)
                days = (nv - ref_dt).total_seconds() / 86400.0
                if days < 7.0:
                    keep_active.append(j)
                    continue
        except Exception:
            pass
        archive.append(j)

    # active 저장
    try:
        with open(_ACTIVE_JOBS_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {"jobs": [j.serialize() for j in keep_active]},
                f, ensure_ascii=False, indent=2,
            )
    except Exception as e:
        print(f"[job_manager] save active 실패 (graceful): {e}")
        return False

    # archive append (graceful)
    if archive:
        try:
            with open(_COMPLETED_LOG_PATH, "a", encoding="utf-8") as f:
                for j in archive:
                    f.write(json.dumps(j.serialize(), ensure_ascii=False) + "\n")
            # rotation
            try:
                if (os.path.exists(_COMPLETED_LOG_PATH)
                        and os.path.getsize(_COMPLETED_LOG_PATH) > _MAX_COMPLETED_LOG):
                    bak = _COMPLETED_LOG_PATH + f".{nv.strftime('%Y%m%d')}.bak"
                    os.rename(_COMPLETED_LOG_PATH, bak)
            except Exception:
                pass
        except Exception as e:
            print(f"[job_manager] archive 실패 (graceful): {e}")
    return True


# ── 생성·갱신·조회 ──────────────────────────────────────────────────
def create_job(
    title: str,
    kind: str = "manual",
    stage_id: str = "",
    milestone_id: str = "",
    priority: int = 5,
    is_recurring: bool = False,
    recurrence_pattern: str = "",
    notes: str = "",
) -> Job:
    """신규 job 생성·디스크 저장. status='queued'."""
    job = Job(
        job_id=_new_id(),
        title=title[:120],
        kind=kind if kind in JOB_KINDS else "manual",
        stage_id=stage_id,
        milestone_id=milestone_id,
        status="queued",
        priority=max(0, min(10, priority)),
        created_at=_now_iso(),
        is_recurring=is_recurring,
        recurrence_pattern=recurrence_pattern,
        notes=notes[:300],
    )
    jobs = load_jobs()
    jobs.append(job)
    save_jobs(jobs)
    return job


def find_job(job_id: str) -> Optional[Job]:
    if not job_id:
        return None
    for j in load_jobs():
        if j.job_id == job_id:
            return j
    return None


def find_job_by_milestone(
    stage_id: str, milestone_id: str,
) -> Optional[Job]:
    """동일 milestone 의 활성 job 찾기 — 중복 생성 회피."""
    for j in load_jobs():
        if (j.stage_id == stage_id
                and j.milestone_id == milestone_id
                and j.status in ("queued", "running", "paused")):
            return j
    return None


def update_job(job_id: str, **changes) -> bool:
    """job 필드 갱신·저장."""
    jobs = load_jobs()
    found = False
    for i, j in enumerate(jobs):
        if j.job_id != job_id:
            continue
        for k, v in changes.items():
            if hasattr(j, k):
                try:
                    setattr(j, k, v)
                except Exception:
                    continue
        if "status" in changes and changes["status"] in ("completed", "failed", "cancelled"):
            j.completed_at = j.completed_at or _now_iso()
        j.last_progress_at = _now_iso()
        jobs[i] = j
        found = True
        break
    if found:
        save_jobs(jobs)
    return found


def start_job(job_id: str, chain_id: str = "") -> bool:
    """status running 으로 전환·started_at·n_attempts++."""
    j = find_job(job_id)
    if j is None:
        return False
    return update_job(
        job_id,
        status="running",
        started_at=j.started_at or _now_iso(),
        n_attempts=j.n_attempts + 1,
        last_chain_id=chain_id,
    )


def complete_job(
    job_id: str, result_preview: str = "", progress_pct: float = 1.0,
) -> bool:
    """status completed·progress 1.0."""
    return update_job(
        job_id, status="completed",
        completed_at=_now_iso(),
        result_preview=result_preview[:300],
        progress_pct=max(0.0, min(1.0, progress_pct)),
    )


def fail_job(job_id: str, error_preview: str = "") -> bool:
    return update_job(
        job_id, status="failed",
        completed_at=_now_iso(),
        result_preview=error_preview[:300],
    )


def cancel_job(job_id: str) -> bool:
    return update_job(
        job_id, status="cancelled",
        completed_at=_now_iso(),
    )


def get_active_jobs() -> list[Job]:
    """queued + running + paused job list."""
    return [
        j for j in load_jobs()
        if j.status in ("queued", "running", "paused")
    ]


def pick_next_job() -> Optional[Job]:
    """queued 중 priority 높은 1개 — recurrent 면 next_run_at 도 검사.

    Returns: 다음 실행할 Job·또는 None.
    """
    candidates = []
    nv = _dt.datetime.now()
    for j in load_jobs():
        if j.status != "queued":
            continue
        if j.is_recurring and j.next_run_at:
            try:
                next_dt = _dt.datetime.fromisoformat(j.next_run_at)
                if next_dt > nv:
                    continue  # 아직 시간 안 됨
            except Exception:
                pass
        candidates.append(j)
    if not candidates:
        return None
    # priority 내림차순·동률시 created_at 오래된 것
    candidates.sort(key=lambda j: (-j.priority, j.created_at))
    return candidates[0]


# ── 사용자용 status 요약 ───────────────────────────────────────────
def get_status_summary(work_state: str = "unknown") -> str:
    """자연어 status 요약·ENFP 톤 가능.

    예시 출력:
      "지금 진행 중 job 2개:
        - [running] Q1 매출 시장 조사 (chain 진행률 60%)
        - [queued] 경쟁사 분석 보고서"

    빈 결과 시: "지금 background job 없어요."
    """
    jobs = get_active_jobs()
    if not jobs:
        return "지금 background job 없어요."

    by_status = {}
    for j in jobs:
        by_status.setdefault(j.status, []).append(j)

    lines = [f"활성 job {len(jobs)}개:"]
    for status in ("running", "queued", "paused"):
        items = by_status.get(status) or []
        for j in items[:3]:
            pct_str = f" · {int(j.progress_pct * 100)}%" if j.progress_pct > 0 else ""
            lines.append(
                f"  · [{status}] {j.title[:60]}{pct_str}"
            )
    return "\n".join(lines)


def has_status_query_keyword(text: str) -> bool:
    """사용자 메시지가 status 조회 의도인지·자연어 hook."""
    if not text:
        return False
    t = text.lower()
    keywords = (
        "지금 뭐", "뭐 하고", "뭐하고",
        "job", "잡 상태", "백그라운드", "작업 상태",
        "진행 중", "진행상황", "어디까지", "현황",
        "status", "현재 상태",
    )
    return any(kw in t for kw in keywords)


# ── 통합 helper — autonomous_workflow chain 시작·완료 hook ─────────
def begin_job_for_chain(
    chain_id: str, milestone_title: str,
    stage_id: str = "", milestone_id: str = "",
    priority: int = 5,
) -> Optional[Job]:
    """autonomous_workflow chain 시작 시 호출.

    기존 job 있으면 status running 으로 전환·없으면 신규 생성.
    """
    # 동일 milestone 의 활성 job 있는지
    existing = find_job_by_milestone(stage_id, milestone_id) if milestone_id else None
    if existing is not None:
        start_job(existing.job_id, chain_id=chain_id)
        # 디스크 갱신 후 다시 로드 — 반환 객체 status 동기화
        refreshed = find_job(existing.job_id)
        return refreshed if refreshed is not None else existing
    # 신규
    job = create_job(
        title=milestone_title,
        kind="milestone_chain",
        stage_id=stage_id,
        milestone_id=milestone_id,
        priority=priority,
    )
    start_job(job.job_id, chain_id=chain_id)
    refreshed = find_job(job.job_id)
    return refreshed if refreshed is not None else job


def end_job_for_chain(
    job_id: str, chain_status: str, result_preview: str = "",
) -> bool:
    """autonomous_workflow chain 끝 시 호출.

    chain_status:
      - "completed" → job completed
      - "failed" / "aborted" → job failed
      - 그 외 → job paused (재시도 가능)
    """
    if not job_id:
        return False
    if chain_status == "completed":
        return complete_job(job_id, result_preview=result_preview)
    if chain_status in ("failed", "aborted"):
        return fail_job(job_id, error_preview=result_preview)
    return update_job(job_id, status="paused", result_preview=result_preview)
