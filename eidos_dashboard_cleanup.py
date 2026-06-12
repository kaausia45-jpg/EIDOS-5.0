# -*- coding: utf-8 -*-
"""고아 MissionChain 정리 (Phase 3.5) — RUNNING/APPROVED 로 박제된 stale 체인 청소.

배경: 디스패처는 한 번에 1개만 실행하는데 레지스트리에 RUNNING/APPROVED 가 수십~수백 건
누적(끝나지 않고 상태가 박제된 고아). 자율 루프를 막고 대시보드를 어지럽힘.

안전 원칙:
  - **드라이런 기본** (apply=True 명시해야 실제 변경)
  - **레지스트리 백업** (`.bak-<ts>`) → 되돌리기 가능
  - **나이 기준** (older_than_hours) — 최근/진짜 실행 가능성 있는 건 보존
  - **PENDING_APPROVAL 은 절대 안 건드림** (승인 카드의 사용자 결정 영역)
  - 정리 = status → ABORTED (실패 아님, '중단·청소' 의미. failure_reason 에 사유 기록)

⚠ EIDOS 본체가 *실행 중이면* 인메모리 레지스트리 싱글톤이 나중에 save() 하며 이 변경을
   덮어쓸 수 있다. **본체를 끈 상태에서** 정리할 것.
"""
from __future__ import annotations

import os
import shutil
import time
from typing import Any, Dict, List, Optional


def _registry(reg=None):
    if reg is not None:
        return reg
    from eidos_mission_chain import get_chain_registry  # type: ignore
    return get_chain_registry()


def _live_statuses():
    from eidos_mission_chain import ChainStatus  # type: ignore
    return (ChainStatus.RUNNING, ChainStatus.APPROVED)


def summarize(reg=None, now: Optional[float] = None) -> Dict[str, Any]:
    """상태별 개수 + RUNNING/APPROVED 의 나이 분포(드라이런 안내용)."""
    now = now if now is not None else time.time()
    r = _registry(reg)
    by_status: Dict[str, int] = {}
    buckets = {"<1h": 0, "1-6h": 0, "6-24h": 0, ">24h": 0, "no_ts": 0}
    live = _live_statuses()
    for c in r.all():
        st = getattr(c, "status", "") or "?"
        by_status[st] = by_status.get(st, 0) + 1
        if st in live:
            created = getattr(c, "created_at", 0.0) or 0.0
            if not created:
                buckets["no_ts"] += 1
                continue
            age_h = (now - created) / 3600.0
            if age_h < 1:
                buckets["<1h"] += 1
            elif age_h < 6:
                buckets["1-6h"] += 1
            elif age_h < 24:
                buckets["6-24h"] += 1
            else:
                buckets[">24h"] += 1
    return {"by_status": by_status, "live_age_buckets": buckets}


def find_orphans(reg=None, *, older_than_hours: float = 2.0,
                 now: Optional[float] = None) -> List[Dict[str, Any]]:
    """older_than_hours 보다 오래된 RUNNING/APPROVED 체인(=고아 후보)."""
    now = now if now is not None else time.time()
    r = _registry(reg)
    live = _live_statuses()
    out: List[Dict[str, Any]] = []
    for c in r.all():
        st = getattr(c, "status", "")
        if st not in live:
            continue
        created = getattr(c, "created_at", 0.0) or 0.0
        age_h = (now - created) / 3600.0 if created else 1e9   # 타임스탬프 없으면 확실한 고아
        if age_h >= older_than_hours:
            out.append({
                "chain_id": getattr(c, "chain_id", ""),
                "name": getattr(c, "name", "") or "(체인)",
                "status": st,
                "age_hours": round(age_h, 1),
                "created_at": created,
            })
    out.sort(key=lambda d: d["age_hours"], reverse=True)
    return out


def _backup_registry(r) -> str:
    path = getattr(r, "_path", None)
    if not path or not os.path.exists(path):
        return ""
    bak = f"{path}.bak-{int(time.time())}"
    try:
        shutil.copy2(path, bak)
        return bak
    except Exception as e:  # noqa: BLE001
        print(f"[cleanup] 백업 실패: {e}")
        return ""


def cleanup_orphans(reg=None, *, older_than_hours: float = 2.0, apply: bool = False,
                    backup: bool = True, now: Optional[float] = None) -> Dict[str, Any]:
    """고아 체인을 ABORTED 로 정리. apply=False(기본)면 드라이런(미변경)."""
    from eidos_mission_chain import ChainStatus  # type: ignore
    orphans = find_orphans(reg, older_than_hours=older_than_hours, now=now)
    res = {"found": len(orphans), "aborted": 0, "backup": "", "apply": apply,
           "older_than_hours": older_than_hours, "items": orphans}
    if not apply or not orphans:
        return res
    r = _registry(reg)
    if backup:
        res["backup"] = _backup_registry(r)
    live = _live_statuses()
    n = 0
    for o in orphans:
        c = r.get(o["chain_id"])
        if c and getattr(c, "status", "") in live:
            c.status = ChainStatus.ABORTED
            c.failure_reason = f"고아 체인 정리 (stale {o['age_hours']}h)"
            n += 1
    r.save()
    res["aborted"] = n
    return res


# ──────────────────────────────────────────────────────────────
def _print_report(title: str, res: Dict[str, Any]):
    print(f"\n=== {title} ===")
    print(f"고아 후보(>{res['older_than_hours']}h RUNNING/APPROVED): {res['found']}건"
          + (f" · 실제 ABORTED: {res['aborted']}건" if res["apply"] else " · (드라이런 — 미변경)"))
    if res.get("backup"):
        print(f"백업: {res['backup']}")
    for o in res["items"][:8]:
        print(f"   - [{o['chain_id'][:8]}] {o['status']:9} {o['age_hours']:6.1f}h  {o['name'][:36]}")
    if res["found"] > 8:
        print(f"   … 외 {res['found'] - 8}건")


if __name__ == "__main__":
    import sys
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = sys.argv[1:]
    do_apply = "--apply" in args
    hours = 2.0
    for i, a in enumerate(args):
        if a == "--hours" and i + 1 < len(args):
            try:
                hours = float(args[i + 1])
            except ValueError:
                pass

    if "--selftest" in args:
        # 임시 레지스트리로 안전 검증 (실데이터 무손상)
        import tempfile
        from eidos_mission_chain import MissionChain, MissionChainRegistry, StageSpec, ChainStatus
        tmp = os.path.join(tempfile.gettempdir(), "dash_cleanup_test.json")
        if os.path.exists(tmp):
            os.remove(tmp)
        reg = MissionChainRegistry(path=tmp)
        NOW = 1_000_000.0
        # 오래된 RUNNING(고아) 3건 + 최근 APPROVED 1건(보존) + PENDING 1건(불가침)
        for k in range(3):
            reg.register(MissionChain(chain_id=f"old-{k}", name=f"고아{k}", original_prompt="x",
                                      stages=[StageSpec(name="s", stage_index=0)],
                                      status=ChainStatus.RUNNING, created_at=NOW - 50 * 3600))
        reg.register(MissionChain(chain_id="fresh", name="최근", original_prompt="x",
                                  stages=[StageSpec(name="s", stage_index=0)],
                                  status=ChainStatus.APPROVED, created_at=NOW - 0.5 * 3600))
        reg.register(MissionChain(chain_id="pend", name="승인대기", original_prompt="x",
                                  stages=[StageSpec(name="s", stage_index=0)],
                                  status=ChainStatus.PENDING_APPROVAL, created_at=NOW - 99 * 3600))
        found = find_orphans(reg=reg, older_than_hours=2.0, now=NOW)
        assert len(found) == 3, f"고아 3건이어야 (got {len(found)})"
        res = cleanup_orphans(reg=reg, older_than_hours=2.0, apply=True, backup=False, now=NOW)
        assert res["aborted"] == 3, "3건 ABORTED 여야"
        assert reg.get("fresh").status == ChainStatus.APPROVED, "최근 건 보존돼야"
        assert reg.get("pend").status == ChainStatus.PENDING_APPROVAL, "PENDING 불가침이어야"
        assert reg.get("old-0").status == ChainStatus.ABORTED, "고아는 ABORTED 여야"
        os.remove(tmp)
        print("✅ 정리 유틸 셀프테스트 통과 — 고아만 ABORTED, 최근·PENDING 보존 (임시 레지스트리)")
        sys.exit(0)

    # 실 레지스트리 대상: 요약 + 드라이런/적용
    s = summarize()
    print("상태별:", s["by_status"])
    print("RUNNING/APPROVED 나이:", s["live_age_buckets"])
    res = cleanup_orphans(older_than_hours=hours, apply=do_apply, backup=True)
    _print_report("적용" if do_apply else "드라이런", res)
    if not do_apply:
        print(f"\n실제 정리하려면:  python eidos_dashboard_cleanup.py --apply --hours {hours}")
        print("⚠ EIDOS 본체를 끈 상태에서 실행하세요(인메모리 싱글톤이 덮어쓰지 않도록).")
