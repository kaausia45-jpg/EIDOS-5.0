# -*- coding: utf-8 -*-
"""대시보드 승인 액션 레이어 (Phase 2) — MissionChain 승인/취소.

대시보드의 승인 카드가 emit 하는 의도(approve/cancel + chain_id)를 처리한다.
저장소는 `eidos_mission_chain.MissionChainRegistry`(영속 JSON 싱글톤).

분담(데스ync 방지):
  - **조회/취소**는 여기서 완결(레지스트리 직접 갱신·저장) — 헤드리스 안전.
  - **승인 후 *실행*** 은 라이브 worker/ActionDispatcher 가 해야 함(비동기 실행 루프 필요).
    여기 `approve_mark` 는 상태만 APPROVED 로 바꾸는 폴백/표식. 실제 stage 실행은
    호스트(eidos_chat_gui)가 worker.submit_task(ad.approve_pending_chain → run_now)로 수행.

모든 함수는 reg 인자로 임시 레지스트리를 주입할 수 있어 테스트가 실데이터를 안 건드린다.
"""
from __future__ import annotations

import time
from typing import List


def _registry(reg=None):
    if reg is not None:
        return reg
    from eidos_mission_chain import get_chain_registry  # type: ignore
    return get_chain_registry()


def list_pending(reg=None) -> List:
    """PENDING_APPROVAL 체인 목록(객체)."""
    from eidos_mission_chain import ChainStatus  # type: ignore
    r = _registry(reg)
    return [c for c in r.all() if getattr(c, "status", "") == ChainStatus.PENDING_APPROVAL]


def cancel(chain_id: str, reason: str = "대시보드에서 취소", reg=None) -> bool:
    """승인 대기 체인을 ABORTED 로 폐기(영속). 멈춘 루프 정리 — 헤드리스 완결."""
    from eidos_mission_chain import ChainStatus  # type: ignore
    r = _registry(reg)
    c = r.get(chain_id)
    if not c or getattr(c, "status", "") != ChainStatus.PENDING_APPROVAL:
        return False
    c.status = ChainStatus.ABORTED
    c.failure_reason = reason
    r.update(c)   # 저장 포함
    return True


def approve_mark(chain_id: str, reg=None) -> bool:
    """승인 표식만(APPROVED). *실제 실행은 라이브 worker/dispatcher 가 한다.*

    호스트가 worker 로 실행을 못 거는 폴백 상황에서 최소한 PENDING 에서 빼주는 용도.
    """
    from eidos_mission_chain import ChainStatus  # type: ignore
    r = _registry(reg)
    c = r.get(chain_id)
    if not c or getattr(c, "status", "") != ChainStatus.PENDING_APPROVAL:
        return False
    c.status = ChainStatus.APPROVED
    c.approved = True
    c.approved_at = time.time()
    r.update(c)
    return True


# ──────────────────────────────────────────────────────────────
# 라이브 호스트(eidos_chat_gui)용 헬퍼 — worker/ActionDispatcher 로 실제 승인·실행 라우팅.
#   대시보드 시그널을 이 함수에 연결하면, 승인=실제 실행 / 취소=dispatcher reject 가 된다.
def route_to_worker(worker, *, approve: bool, chain_id: str = "") -> bool:
    """worker.submit_task 로 dispatcher 의 승인/거절을 비동기 실행. 성공 큐잉 시 True.

    dispatcher 는 worker 에 붙어 있고 `_pending_chain` 1건을 들고 있다고 가정
    (현재 아키텍처). chain_id 는 표시·로깅용(여러 pending 확장 대비).
    """
    ad = (getattr(worker, "_action_dispatcher", None)
          or getattr(worker, "dispatcher", None)
          or getattr(worker, "_dispatcher", None))
    if ad is None or not hasattr(worker, "submit_task"):
        return False
    if getattr(ad, "_pending_chain", None) is None and approve:
        return False   # 승인할 pending 없음 → 폴백(레지스트리 표식)에 맡김
    if approve:
        async def _go(_ad=ad):
            try:
                ok = await _ad.approve_pending_chain()
                if ok:
                    await _ad.run_now()
            except Exception as e:  # noqa: BLE001
                print(f"[dashboard] 승인 실행 실패: {e}")
        worker.submit_task(_go())
    else:
        try:
            ad.reject_pending_chain(reason="대시보드에서 취소")
        except Exception as e:  # noqa: BLE001
            print(f"[dashboard] 거절 실패: {e}")
            return False
    return True


if __name__ == "__main__":
    # 헤드리스 안전 테스트 — 임시 레지스트리에 가짜 체인 1건 → 취소/승인 표식 검증(실데이터 무손상).
    import sys
    import io
    import os
    import tempfile
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass
    from eidos_mission_chain import MissionChain, MissionChainRegistry, StageSpec, ChainStatus

    tmp = os.path.join(tempfile.gettempdir(), "dash_approve_test_chains.json")
    if os.path.exists(tmp):
        os.remove(tmp)
    reg = MissionChainRegistry(path=tmp)
    ch = MissionChain(chain_id="test-001", name="테스트 체인", original_prompt="검증용",
                      stages=[StageSpec(name="s1", stage_index=0), StageSpec(name="s2", stage_index=1)])
    reg.register(ch)

    assert len(list_pending(reg=reg)) == 1, "pending 1건이어야"
    assert cancel("test-001", reg=reg) is True, "취소 성공해야"
    assert reg.get("test-001").status == ChainStatus.ABORTED, "ABORTED 여야"
    assert len(list_pending(reg=reg)) == 0, "취소 후 pending 0건"
    # 승인 표식
    ch2 = MissionChain(chain_id="test-002", name="테스트2", original_prompt="x",
                       stages=[StageSpec(name="s1", stage_index=0)])
    reg.register(ch2)
    assert approve_mark("test-002", reg=reg) is True
    assert reg.get("test-002").status == ChainStatus.APPROVED
    os.remove(tmp)
    print("✅ 승인 액션 레이어 테스트 통과 — list/cancel/approve_mark (임시 레지스트리, 실데이터 무손상)")
