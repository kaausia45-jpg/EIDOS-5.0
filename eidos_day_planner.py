#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eidos_day_planner.py — AIRA '일정 추천' 핵심 엔진 (2026-06-09).

목적
  Petalbot 현황(프로젝트별 남은 공수 + 납품일)과 기존 일정(이미 찬 시간대),
  근무 제약(근무시간·점심·집중블록)을 입력받아, **납품일을 지키면서**
  **기존 일정과 충돌하지 않게** 시간대별 할 일을 배치한다.

설계 원칙(정직성 — north star)
  - 용량이 부족해 납품일 안에 다 못 넣으면 **억지로 끼워넣지 않고**(가짜 배치 금지)
    `warnings`에 정직하게 '기한 내 미배치 N시간'을 보고하고, 들어가는 만큼만 배치한다.
  - 외부 의존 0 (순수 stdlib) · LLM 0 · 결정적 → 단위 테스트로 검증.

배치 전략 (EDF — Earliest Deadline First)
  날짜를 오늘부터 시간순으로 훑으며, 각 날의 빈 시간대를 집중블록 단위로
  '가장 급한(납품일 임박)' 프로젝트부터 채운다. 납품일이 지난 날에는 그
  프로젝트를 더 배치하지 않는다(미배치는 shortfall 로 보고).

이 모듈은 GUI/네트워크를 모른다. 입력 수집(list_projects·planner 읽기)과
출력 등록(add_task_programmatically)·TTS 는 호출측(Phase D)이 담당한다.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Optional


# ── 시간 헬퍼 (분 단위, 0~1440) ───────────────────────────────
def hhmm_to_min(s: str) -> int:
    """'HH:MM' → 자정 기준 분. 잘못된 입력은 0."""
    try:
        h, m = str(s).split(":")
        return max(0, min(24 * 60, int(h) * 60 + int(m)))
    except Exception:  # noqa: BLE001
        return 0


def min_to_hhmm(m: int) -> str:
    """분 → 'HH:MM'."""
    m = max(0, min(24 * 60, int(m)))
    return f"{m // 60:02d}:{m % 60:02d}"


# ── 입력/출력 자료구조 ────────────────────────────────────────
@dataclass
class WorkConstraints:
    """근무 제약 — eidos_settings.json 에서 읽어 채운다(기본값 동봉)."""
    work_start: str = "09:00"
    work_end: str = "18:00"
    lunch_start: str = "12:00"
    lunch_end: str = "13:00"
    focus_block_min: int = 90            # 집중 블록 길이(분)
    workdays: tuple = (0, 1, 2, 3, 4)    # 근무 요일(월=0 … 일=6)
    max_daily_hours: Optional[float] = None  # 하루 최대 작업시간(None=근무시간 전체)


@dataclass
class ProjectLoad:
    """일정에 넣을 프로젝트 1건의 부하."""
    project_id: str
    title: str
    remaining_hours: float
    deadline_at: Optional[float] = None   # 납품일 Unix ts (없으면 None)


@dataclass
class Slot:
    """배치된 한 칸."""
    date: str          # 'YYYY-MM-DD'
    start: str         # 'HH:MM'
    end: str           # 'HH:MM'
    project_id: str
    title: str
    minutes: int

    def label(self) -> str:
        h = self.minutes / 60.0
        return f"{self.title} ({h:.1f}h)" if self.minutes % 60 else f"{self.title} ({self.minutes // 60}h)"


@dataclass
class PlanResult:
    slots: list = field(default_factory=list)        # list[Slot] — 시간순
    warnings: list = field(default_factory=list)     # 정직 경고(미배치/마감초과 등)
    per_project: dict = field(default_factory=dict)  # pid -> {title,needed_min,assigned_min,shortfall_min,fits}

    @property
    def feasible(self) -> bool:
        """모든 프로젝트가 납품일 안에 다 들어갔는가."""
        return not any(p["shortfall_min"] > 0 for p in self.per_project.values())


# ── 내부: 하루 빈 시간대 계산 ─────────────────────────────────
def _subtract(intervals: list, cut: tuple) -> list:
    """intervals(분 구간 리스트)에서 cut 구간을 빼고 남은 구간들."""
    cs, ce = cut
    out = []
    for s, e in intervals:
        if ce <= s or cs >= e:        # 안 겹침
            out.append((s, e))
            continue
        if cs > s:
            out.append((s, cs))       # 앞 조각
        if ce < e:
            out.append((ce, e))       # 뒤 조각
    return [(s, e) for s, e in out if e > s]


def free_blocks_for_day(con: WorkConstraints, busy_today: list,
                        earliest_min: Optional[int] = None) -> list:
    """그 날의 빈 시간대 [(start_min,end_min)…] — 근무창에서 점심·기존일정을 뺀 것.

    earliest_min: 그 날 이 시각 이전은 못 쓰게(오늘 '지금' 이후만 배치) — None=근무시작.
    busy_today: [(start_min,end_min)…] 이미 찬 구간들.
    """
    ws, we = hhmm_to_min(con.work_start), hhmm_to_min(con.work_end)
    if earliest_min is not None:
        ws = max(ws, earliest_min)
    if we <= ws:
        return []
    free = [(ws, we)]
    free = _subtract(free, (hhmm_to_min(con.lunch_start), hhmm_to_min(con.lunch_end)))
    for b in sorted(busy_today):
        free = _subtract(free, b)
    return sorted(free)


# ── 메인 ──────────────────────────────────────────────────────
def _deadline_date(ts: Optional[float]):
    if ts is None:
        return None
    try:
        return _dt.datetime.fromtimestamp(float(ts)).date()
    except Exception:  # noqa: BLE001
        return None


def plan_schedule(projects: list, busy: list, con: Optional[WorkConstraints] = None, *,
                  start_date: Optional[_dt.date] = None, start_min: Optional[int] = None,
                  horizon_days: int = 30, max_horizon_days: int = 180) -> PlanResult:
    """시간대별 할 일을 배치한다.

    projects   : list[ProjectLoad]
    busy       : list[(date_iso, 'HH:MM', 'HH:MM')] 이미 찬 일정(충돌 회피용)
    con        : WorkConstraints (None=기본값)
    start_date : 계획 시작일(None=오늘)
    start_min  : 시작일에 이 시각(분) 이후만 사용(None=근무시작). 오늘 '지금' 이후 배치용.
    horizon_days: 납품일 없는 프로젝트를 위한 기본 지평(일).
    """
    con = con or WorkConstraints()
    res = PlanResult()
    if start_date is None:
        start_date = _dt.date.today()

    # 부하 정리 — 남은 공수>0 만, needed_min 계산
    loads = []
    for p in projects:
        need = int(round(float(p.remaining_hours) * 60))
        if need <= 0:
            continue
        loads.append(p)
        res.per_project[p.project_id] = {
            "title": p.title, "needed_min": need, "assigned_min": 0,
            "shortfall_min": need, "fits": False,
        }
    if not loads:
        res.warnings.append("배치할 작업이 없습니다(남은 공수가 0이거나 프로젝트 없음).")
        return res

    remaining = {p.project_id: res.per_project[p.project_id]["needed_min"] for p in loads}
    ddate = {p.project_id: _deadline_date(p.deadline_at) for p in loads}

    # 지평(horizon) 끝 = max(기본지평, 가장 늦은 납품일), 단 안전 상한
    horizon_end = start_date + _dt.timedelta(days=max(1, horizon_days) - 1)
    for p in loads:
        d = ddate[p.project_id]
        if d and d > horizon_end:
            horizon_end = d
    hard_cap = start_date + _dt.timedelta(days=max_horizon_days)
    if horizon_end > hard_cap:
        horizon_end = hard_cap

    # busy 를 날짜별 분구간으로
    busy_by_day: dict = {}
    for row in busy:
        try:
            diso, s, e = row[0], row[1], row[2]
            sm, em = hhmm_to_min(s), hhmm_to_min(e)
            if em > sm:
                busy_by_day.setdefault(str(diso), []).append((sm, em))
        except Exception:  # noqa: BLE001
            continue

    max_daily_min = None
    if con.max_daily_hours:
        max_daily_min = int(round(float(con.max_daily_hours) * 60))

    def _urgency_key(pid):
        d = ddate[pid]
        # 납품일 있는 것 먼저(0), 날짜 이른 순. 없는 것은 맨 뒤(1).
        return (0, d) if d is not None else (1, horizon_end + _dt.timedelta(days=1))

    # 날짜 순회
    day = start_date
    while day <= horizon_end and any(v > 0 for v in remaining.values()):
        if day.weekday() not in con.workdays:
            day += _dt.timedelta(days=1)
            continue
        diso = day.isoformat()
        emin = start_min if (day == start_date and start_min is not None) else None
        free = free_blocks_for_day(con, busy_by_day.get(diso, []), emin)
        day_used = 0

        for (fs, fe) in free:
            s = fs
            while s < fe and (max_daily_min is None or day_used < max_daily_min):
                # 오늘(day) 작업 가능한 프로젝트: 남은>0 이고 (납품일 None 또는 납품일>=day 또는 이미 지난 마감=항상 급함)
                eligible = []
                for pid, rem in remaining.items():
                    if rem <= 0:
                        continue
                    d = ddate[pid]
                    if d is None or d >= day or d < start_date:
                        eligible.append(pid)
                if not eligible:
                    break
                pid = min(eligible, key=_urgency_key)

                avail = fe - s
                if max_daily_min is not None:
                    avail = min(avail, max_daily_min - day_used)
                block = min(con.focus_block_min, remaining[pid], avail)
                if block <= 0:
                    break

                title = res.per_project[pid]["title"]
                res.slots.append(Slot(date=diso, start=min_to_hhmm(s), end=min_to_hhmm(s + block),
                                      project_id=pid, title=title, minutes=block))
                remaining[pid] -= block
                res.per_project[pid]["assigned_min"] += block
                s += block
                day_used += block
            if max_daily_min is not None and day_used >= max_daily_min:
                break
        day += _dt.timedelta(days=1)

    # 결산 + 정직 경고
    res.slots.sort(key=lambda sl: (sl.date, sl.start))
    today = start_date
    for p in loads:
        info = res.per_project[p.project_id]
        info["shortfall_min"] = max(0, info["needed_min"] - info["assigned_min"])
        info["fits"] = info["shortfall_min"] == 0
        d = ddate[p.project_id]
        if d is not None and d < today:
            res.warnings.append(
                f"⚠ '{p.title}' 납품일({d.isoformat()})이 이미 지났습니다 — 최우선 배치했지만 확인 필요.")
        if info["shortfall_min"] > 0:
            hrs = info["shortfall_min"] / 60.0
            if d is not None:
                res.warnings.append(
                    f"⚠ '{p.title}' 납품일({d.isoformat()}) 안에 {hrs:.1f}시간을 못 넣었습니다"
                    f"(용량 부족). 기한·범위·인력 조정이 필요합니다.")
            else:
                res.warnings.append(
                    f"⚠ '{p.title}' 지평({horizon_end.isoformat()}) 안에 {hrs:.1f}시간이 미배치 상태입니다.")
    return res


# ── 근무 제약 로더 (eidos_settings.json) ──────────────────────
def load_constraints(settings: Optional[dict] = None) -> WorkConstraints:
    """eidos_settings.json(dict)에서 근무 제약을 읽는다(기본값 동봉).

    settings 키(모두 선택):
      work_start, work_end, lunch_start, lunch_end (HH:MM)
      focus_block_min (int), work_max_daily_hours (float)
      workdays (list[int], 월=0)
    """
    s = settings or {}
    con = WorkConstraints()
    con.work_start = str(s.get("work_start", con.work_start))
    con.work_end = str(s.get("work_end", con.work_end))
    con.lunch_start = str(s.get("lunch_start", con.lunch_start))
    con.lunch_end = str(s.get("lunch_end", con.lunch_end))
    try:
        con.focus_block_min = int(s.get("focus_block_min", con.focus_block_min))
    except Exception:  # noqa: BLE001
        pass
    wd = s.get("workdays")
    if isinstance(wd, (list, tuple)) and wd:
        try:
            con.workdays = tuple(int(x) for x in wd)
        except Exception:  # noqa: BLE001
            pass
    mdh = s.get("work_max_daily_hours")
    if mdh:
        try:
            con.max_daily_hours = float(mdh)
        except Exception:  # noqa: BLE001
            pass
    return con
