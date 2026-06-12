# eidos_self_improvement_loop.py
# [Wave1-C/D 2026-05-28] 5일 주기 자기개선 사이클 루프 + 누적 diff 텔레그램 보고.
#
# 사용자 결정:
#   - 사이클 주기: 5일 1회
#   - settings: self_improvement_enabled (default False — 안전상)
#   - 텔레그램 다이제스트 보고 (drift mitigation 포함)
#   - 모듈 부재·실패 어디서나 graceful — chat 흐름 영향 X
#
# 디자인 원칙:
#   - LLM 호출 0 (Wave1 안에서)
#   - 디스크 상태 파일로 마지막 사이클 시각 추적 — App 재시작 후에도 일관
#   - QTimer 30 분 간격으로 "5일 지났나" 체크 (Wave1-C 의 1줄 통합)
#   - 텔레그램 best-effort — eidos_telegram_bot 부재해도 동작

from __future__ import annotations

import datetime as _dt
import json
import os
import asyncio
from typing import Optional


_BASE_DIR = os.path.join("eidos_files", "agents")
_STATE_PATH = os.path.join(_BASE_DIR, "self_improvement_state.json")

# 사이클 주기 (5일 = 120 시간)
CYCLE_INTERVAL_HOURS = 24 * 5
# QTimer check 간격 (30 분) — 5일 지났는지 polling
CHECK_INTERVAL_MS = 30 * 60 * 1000


def _now_dt() -> _dt.datetime:
    return _dt.datetime.utcnow()


def _now_iso() -> str:
    return _now_dt().isoformat(timespec="seconds") + "Z"


def _parse_iso(ts: str) -> Optional[_dt.datetime]:
    if not ts:
        return None
    try:
        s = ts.replace("Z", "+00:00")
        return _dt.datetime.fromisoformat(s).replace(tzinfo=None)
    except Exception:
        return None


# ── 상태 파일 (마지막 사이클 시각) ────────────────────────────────────────
def load_state() -> dict:
    """{"last_cycle_at": iso, "cycle_count": int} 형식. 없으면 empty."""
    if not os.path.exists(_STATE_PATH):
        return {}
    try:
        with open(_STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception as e:
        print(f"[self_improvement] load_state 실패 (graceful, 빈 dict): {e}")
        return {}


def save_state(state: dict) -> bool:
    """atomic write."""
    try:
        os.makedirs(_BASE_DIR, exist_ok=True)
        tmp = _STATE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(state, ensure_ascii=False, indent=2))
        if os.path.exists(_STATE_PATH):
            try:
                os.replace(tmp, _STATE_PATH)
            except Exception:
                try:
                    os.remove(_STATE_PATH)
                except Exception:
                    pass
                os.rename(tmp, _STATE_PATH)
        else:
            os.rename(tmp, _STATE_PATH)
        return True
    except Exception as e:
        print(f"[self_improvement] save_state 실패 (graceful): {e}")
        return False


def reset_state() -> bool:
    """테스트·force 다음 tick 즉시 실행."""
    try:
        if os.path.exists(_STATE_PATH):
            os.remove(_STATE_PATH)
        return True
    except Exception as e:
        print(f"[self_improvement] reset_state 실패 (graceful): {e}")
        return False


# ── 주기 판정 ──────────────────────────────────────────────────────────
def should_run_cycle(now: Optional[_dt.datetime] = None) -> bool:
    """5일 경과 여부. 상태 파일에 마지막 시각 없으면 항상 True (첫 사이클)."""
    state = load_state()
    last_iso = state.get("last_cycle_at")
    if not last_iso:
        return True
    last_dt = _parse_iso(str(last_iso))
    if last_dt is None:
        return True
    elapsed_hours = ((now or _now_dt()) - last_dt).total_seconds() / 3600.0
    return elapsed_hours >= CYCLE_INTERVAL_HOURS


# ── LLM sync wrapper (Wave2) ──────────────────────────────────────────
def _sync_llm_call(prompt: str) -> str:
    """[Wave2-E] async LLM 함수를 sync 로 wrapping.

    Wave2-B (ToM 분해) 와 Wave2-C (용어 번역) 에 inject. PyQt 이벤트 루프
    안에서 호출돼도 새 event_loop 만들어 격리. graceful — 네트워크/키 부재 시
    빈 문자열 반환.
    """
    try:
        import asyncio
        from llm_module import get_llm_response_async
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(get_llm_response_async(prompt)) or ""
        finally:
            loop.close()
    except Exception as e:
        print(f"[self_improvement] LLM sync call 실패 (graceful): {e}")
        return ""


def _measure_pattern_effect(pattern, prev_indicators, curr_indicators) -> str:
    """[Wave2-E] 이식된 패턴 효과 측정. target_goal_indicator 의 t→t+1 diff 기반.

    Returns: "positive" (+0.05 이상) / "negative" (-0.05 이하) / "neutral" (그 외).
    """
    try:
        if not prev_indicators or not curr_indicators:
            return "neutral"
        ind = getattr(pattern, "target_goal_indicator", "")
        if not ind:
            return "neutral"
        prev_v = float(getattr(prev_indicators, ind, 0.5))
        curr_v = float(getattr(curr_indicators, ind, 0.5))
        delta = curr_v - prev_v
        if delta >= 0.05:
            return "positive"
        elif delta <= -0.05:
            return "negative"
        else:
            return "neutral"
    except Exception:
        return "neutral"


def _import_for_goal(new_goal, max_sources: int = 2) -> list:
    """[Wave2-E] 새 목표 → 외부 검색 → ToM 분해 → 번역 → 매개변수화 → 저장.

    LLM sync wrapper 사용. 각 단계 graceful — 한 자료 실패해도 다음 자료 계속.
    반환: 이식 성공한 ImportedPattern list.
    """
    out: list = []
    if not new_goal:
        return out
    try:
        import eidos_external_research as er
        import eidos_pattern_decompose as pd_mod
        import eidos_pattern_translate as pt
        import eidos_pattern_catalog as pc

        goal_ind = getattr(new_goal, "target_indicator", "")
        # [Phase 4 2026-05-30] 이종 도메인 검색 — LLM 이 의외의 도메인(생물/법/군사…)
        # 선택해 유추전이용 먼 도메인 자료를 끌어옴. _sync_llm_call 주입.
        results = er.research_for_goal(
            new_goal, max_total=max_sources + 1,
            llm_callable=_sync_llm_call, cross_domain=True)
        if not results:
            print(f"[Wave2-E] 외부 검색 결과 0건 — 이식 skip (goal={goal_ind})")
            return out

        # 검색 결과 저장 (debug)
        try:
            er.save_research_batch(new_goal.id, results)
        except Exception:
            pass

        for rr in results[:max_sources]:
            try:
                # 1) ToM 분해
                tom = pd_mod.decompose_to_tom(rr, _sync_llm_call,
                                              goal_indicator=goal_ind)
                if not tom:
                    continue
                # 2) EIDOS 용어 번역
                trans = pt.translate_to_eidos_terms(tom, _sync_llm_call,
                                                    goal_indicator=goal_ind)
                if not trans:
                    continue
                # 3) 매개변수화
                imp = pc.parameterize(trans, target_goal_indicator=goal_ind)
                if not imp:
                    continue
                # 4) catalog 저장
                if pc.save_pattern(imp):
                    out.append(imp)
                    print(f"[Wave2-E] 이식 성공: {imp.id} "
                          f"({rr.source} → {goal_ind})")
            except Exception as _e_one:
                print(f"[Wave2-E] 자료 1개 이식 실패 (graceful, 계속): {_e_one}")
                continue
    except Exception as e:
        print(f"[Wave2-E] _import_for_goal 실패 (graceful): {e}")
    return out


# ── 사이클 실행 ────────────────────────────────────────────────────────
def run_cycle() -> dict:
    """1 자기개선 사이클 실행.

    Wave1 흐름 (①②⑤) + Wave2 추가 (③④):
      1. measure_current → curr snapshot · prev 로드
      2. [Wave2-E] 이전 사이클의 active 이식 패턴 효과 측정 (record_outcome)
      3. save_snapshot
      4. cycle_step → 목표 평가·신규 생성
      5. [Wave2-E] 새 목표 있으면 외부 검색·ToM 분해·번역·이식
      6. compute_diff
      7. 텔레그램 다이제스트 (이식 패턴 + 효과 측정 포함)
      8. state 갱신

    반환: {"cycle", "diff", "imported_patterns", "effect_results", "telegram_sent"}.
    graceful — 어떤 단계 실패해도 다음 단계 계속.
    """
    out = {
        "cycle": {}, "diff": {}, "telegram_sent": False,
        "imported_patterns": [], "effect_results": [],
    }

    # 1·3
    curr = None
    prev = None
    try:
        import eidos_self_indicators as si
        recent = si.load_recent(n=1)
        prev = recent[0] if recent else None
        curr = si.measure_current()
    except Exception as e:
        print(f"[self_improvement] step 1 실패 (graceful, 사이클 중단): {e}")
        return out

    # 2. [Wave2-E] 직전 사이클의 active 이식 패턴 효과 측정
    try:
        import eidos_pattern_catalog as pc
        active_patterns = pc.list_active_patterns()
        for ip in active_patterns:
            outcome = _measure_pattern_effect(ip, prev, curr)
            pc.record_outcome(ip.id, outcome)
            out["effect_results"].append({
                "pattern_id": ip.id,
                "target": ip.target_goal_indicator,
                "outcome": outcome,
            })
        if out["effect_results"]:
            print(f"[Wave2-E] {len(out['effect_results'])} 패턴 효과 측정 완료")
    except Exception as e:
        print(f"[self_improvement] step 2 (효과 측정) 실패 (graceful): {e}")

    # 3. snapshot 저장
    try:
        import eidos_self_indicators as si
        si.save_snapshot(curr)
    except Exception as e:
        print(f"[self_improvement] step 3 (save_snapshot) 실패 (graceful): {e}")

    # 4. cycle_step
    try:
        import eidos_self_goals as sg
        cycle_result = sg.cycle_step(curr, prev)
        out["cycle"] = cycle_result
    except Exception as e:
        print(f"[self_improvement] step 4 (cycle_step) 실패 (graceful): {e}")
        out["cycle"] = {}

    # 4b. [Wave4 2026-05-28] Curiosity-driven 목표 fallback.
    # cycle_step 이 weakest_dimension 기반 신규 목표 생성 못 했으면
    # (모든 지표 잘 됨 또는 active goal cap) — prediction error 누적 데이터에서
    # "EIDOS 가 아직 모르는 곳" 자동 탐색 목표 생성. graceful — 신호 없으면 no-op.
    try:
        _cycle_has_new = bool(
            isinstance(out["cycle"], dict)
            and out["cycle"].get("new_goal")
        )
        if not _cycle_has_new:
            from eidos_curiosity_driver import top_curiosity_goal
            _curio_goal = top_curiosity_goal(
                min_samples=3, min_error=0.3,
            )
            if _curio_goal is not None:
                # eidos_self_goals 의 load_all / save_all 로 직접 append
                try:
                    import eidos_self_goals as sg2
                    _all_goals = sg2.load_all()
                    _all_goals.append(_curio_goal)
                    sg2.save_all(_all_goals)
                    out["cycle"] = dict(out.get("cycle") or {})
                    out["cycle"]["new_goal"] = _curio_goal
                    out["cycle"]["new_goal_source"] = "curiosity"
                    print(f"[Wave4-curiosity] 탐색 목표 등록 — "
                          f"target={_curio_goal.target_indicator} "
                          f"current={_curio_goal.current_value:.2f} "
                          f"target={_curio_goal.target_value:.2f}")
                except Exception as _e_gmgr:
                    print(f"[Wave4-curiosity] save_all 실패 (graceful): {_e_gmgr}")
    except Exception as e_curio:
        print(f"[self_improvement] step 4b (curiosity) 실패 (graceful): {e_curio}")

    # 4c. [Wave4-B 2026-05-28] Calibration update — forward + inverse 로그 집계 →
    # action 별 자기 확신도·controllability 계산 → self_competence 업데이트.
    # graceful — 로그 비어있거나 모듈 부재 시 no-op.
    try:
        from eidos_self_calibration import (
            compute_calibration_map,
            update_self_competence_from_calibration,
        )
        _cm = compute_calibration_map(min_samples=3, since_hours=168.0)
        if _cm.n_actions > 0:
            _upd = update_self_competence_from_calibration(_cm)
            out["calibration"] = {
                "n_actions": _cm.n_actions,
                "overall_forward_calibration": _cm.overall_forward_calibration,
                "overall_controllability": _cm.overall_controllability,
                "updated_domains": _upd.get("updated_domains", []),
            }
            print(f"[Wave4-calibration] {_cm.n_actions} actions · "
                  f"fwd_cal={_cm.overall_forward_calibration:.2f} "
                  f"ctrl={_cm.overall_controllability:.2f} "
                  f"domains_updated={len(_upd.get('updated_domains', []))}")
        else:
            out["calibration"] = {"n_actions": 0}
    except Exception as e_cal:
        print(f"[self_improvement] step 4c (calibration) 실패 (graceful): {e_cal}")
        out["calibration"] = {}

    # 4d. [Wave5-A 2026-05-28] Counterfactual 분석 — 최근 tick 의 인과 강도 측정.
    # 각 action 의 causal_strength 누적. 결과는 self_improvement 다이제스트에 포함.
    # 비용 부담 — settings.counterfactual_enabled=true 일 때만 (default OFF).
    try:
        # settings 직접 읽기 (다른 모듈처럼)
        import json as _json_cf
        _cf_settings_path = os.path.join("eidos_files", "settings.json")
        _cf_settings = {}
        if os.path.exists(_cf_settings_path):
            with open(_cf_settings_path, "r", encoding="utf-8") as _f_cf:
                _cf_settings = _json_cf.load(_f_cf) or {}
        if _cf_settings.get("counterfactual_enabled", False):
            from eidos_counterfactual import evaluate_recent_ticks_async
            _n_ticks = int(_cf_settings.get("counterfactual_n_ticks", 15) or 15)
            # event loop 격리
            try:
                _loop = asyncio.new_event_loop()
                try:
                    _attrs = _loop.run_until_complete(
                        evaluate_recent_ticks_async(n_ticks=_n_ticks)
                    )
                finally:
                    _loop.close()
                if _attrs:
                    out["counterfactual"] = {
                        "n_actions": len(_attrs),
                        "top_causal": sorted(
                            [
                                {"action_id": a.action_id,
                                 "causal_strength": a.causal_strength,
                                 "n_executions": a.n_executions}
                                for a in _attrs.values()
                            ],
                            key=lambda x: x["causal_strength"],
                            reverse=True,
                        )[:5],
                    }
                    print(f"[Wave5-cf] {len(_attrs)} actions 분석·"
                          f"top causal_strength: "
                          f"{out['counterfactual']['top_causal'][0] if out['counterfactual']['top_causal'] else '(none)'}")
            except Exception as _e_cf_run:
                print(f"[Wave5-cf] evaluate 실행 실패 (graceful): {_e_cf_run}")
                out["counterfactual"] = {}
    except Exception as _e_cf:
        print(f"[Wave5-cf] step 4d (counterfactual) 실패 (graceful): {_e_cf}")
        out["counterfactual"] = {}

    # 5. [Wave2-E] 새 목표 있으면 외부 검색 + 이식
    imported_ids: list = []
    try:
        new_goal = out["cycle"].get("new_goal") if isinstance(out["cycle"], dict) else None
        if new_goal:
            imported = _import_for_goal(new_goal, max_sources=2)
            out["imported_patterns"] = [
                {"id": p.id, "source": p.source_type,
                 "confidence": p.mapping_confidence,
                 "param_count": len(p.parameters)}
                for p in imported
            ]
            imported_ids = [p.id for p in imported]
    except Exception as e:
        print(f"[self_improvement] step 5 (이식) 실패 (graceful): {e}")

    # 5.5. [Wave3-D] 새 목표 있으면 도구도 자동 제작
    # 비실행 자산 (DAG·prompt·data) 은 자동 활성·실행 코드 는 pending_approval.
    out["built_tools"] = []
    try:
        new_goal_for_tools = out["cycle"].get("new_goal") if isinstance(out["cycle"], dict) else None
        if new_goal_for_tools:
            import eidos_tool_builder as tb
            tools = tb.build_tools_for_goal(
                new_goal_for_tools,
                related_pattern_ids=imported_ids,
                llm_callable=_sync_llm_call,
                max_tools=2,
            )
            out["built_tools"] = [
                {"id": t.id, "type": t.type, "status": t.status,
                 "name": t.name, "description": t.description[:100]}
                for t in tools
            ]
    except Exception as e:
        print(f"[self_improvement] step 5.5 (도구 제작) 실패 (graceful): {e}")
        out["built_tools"] = []

    # 6. diff
    try:
        import eidos_self_indicators as si
        diff = si.compute_diff(prev, curr) if prev else {}
        out["diff"] = diff
    except Exception as e:
        print(f"[self_improvement] step 6 (diff) 실패 (graceful): {e}")
        out["diff"] = {}

    # 7. 텔레그램
    try:
        msg = _build_telegram_message(
            out["cycle"], out["diff"], prev, curr,
            imported_patterns=out["imported_patterns"],
            effect_results=out["effect_results"],
            built_tools=out.get("built_tools"),
        )
        sent = _send_telegram_best_effort(msg)
        out["telegram_sent"] = sent
    except Exception as e:
        print(f"[self_improvement] step 7 (텔레그램) 실패 (graceful): {e}")

    # 8. 상태 갱신
    try:
        state = load_state()
        state["last_cycle_at"] = _now_iso()
        state["cycle_count"] = int(state.get("cycle_count", 0)) + 1
        save_state(state)
    except Exception as e:
        print(f"[self_improvement] step 8 (state 저장) 실패 (graceful): {e}")

    return out


# ── 텔레그램 메시지 조립 ──────────────────────────────────────────────
def _build_telegram_message(cycle_result: dict, diff: dict,
                            prev, curr,
                            imported_patterns: Optional[list] = None,
                            effect_results: Optional[list] = None,
                            built_tools: Optional[list] = None) -> str:
    """5일 사이클 다이제스트 메시지 (Wave1+Wave2+Wave3).

    구성:
      🎯 목표 사이클 요약
      📊 누적 지표 변화
      🌱 [Wave2] 신규 이식 패턴
      📈 [Wave2] 직전 사이클 이식 패턴 효과 측정
      📚 [Wave2] catalog 현황
      🛠 [Wave3] 신규 제작 도구
      🧰 [Wave3] tool catalog 현황
    """
    parts = []
    try:
        import eidos_self_goals as sg
        cycle_msg = sg.summarize_cycle_for_telegram(cycle_result)
        parts.append(cycle_msg)
    except Exception as e:
        parts.append(f"⚠️ 사이클 요약 실패 (graceful): {e}")

    try:
        import eidos_self_indicators as si
        if diff:
            parts.append("\n━━━━━━━━━━")
            parts.append("📊 *지표 분포 변화 (t → t+1)*")
            parts.append(si.summarize_diff_for_telegram(diff, top_n=5))
    except Exception as e:
        parts.append(f"⚠️ 지표 변화 요약 실패 (graceful): {e}")

    # [Wave2-E] 신규 이식 패턴
    if imported_patterns:
        parts.append("\n━━━━━━━━━━")
        parts.append(f"🌱 *신규 이식 패턴* — {len(imported_patterns)}개")
        for ip in imported_patterns[:3]:
            warn = "⚠️" if ip.get("confidence", 0.5) < 0.5 else "✅"
            parts.append(
                f"  {warn} `{ip['id'][:12]}` ({ip['source']}) — "
                f"conf {ip['confidence']:.2f}·미정계수 {ip['param_count']}개"
            )

    # [Wave2-E] 효과 측정 결과
    if effect_results:
        pos = sum(1 for r in effect_results if r["outcome"] == "positive")
        neg = sum(1 for r in effect_results if r["outcome"] == "negative")
        neu = sum(1 for r in effect_results if r["outcome"] == "neutral")
        parts.append("\n━━━━━━━━━━")
        parts.append(f"📈 *이식 패턴 효과* — 총 {len(effect_results)}건")
        parts.append(f"  ✅ positive {pos} · ⚪ neutral {neu} · ❌ negative {neg}")

    # [Wave2-E] catalog 현황
    try:
        import eidos_pattern_catalog as pc
        all_p = pc.list_all_patterns()
        if all_p:
            parts.append("\n━━━━━━━━━━")
            parts.append(pc.summarize_catalog_for_telegram(top_n=3))
    except Exception:
        pass

    # [Wave3-D] 신규 제작 도구
    if built_tools:
        parts.append("\n━━━━━━━━━━")
        parts.append(f"🛠 *신규 제작 도구* — {len(built_tools)}개")
        pending = [t for t in built_tools if t["status"] == "pending_approval"]
        active = [t for t in built_tools if t["status"] == "active"]
        for t in active[:3]:
            parts.append(
                f"  ✅ `{t['id'][:14]}` ({t['type']}) — {t['description'][:80]}"
            )
        for t in pending[:3]:
            parts.append(
                f"  ⏳ `{t['id'][:14]}` ({t['type']}) — 승인 필요\n"
                f"    {t['description'][:80]}\n"
                f"    승인: `/approve_tool {t['id']}`"
            )

    # [Wave3-D] tool catalog 현황
    try:
        import eidos_tool_catalog as tc
        all_t = tc.list_all_tools()
        if all_t:
            parts.append("\n━━━━━━━━━━")
            parts.append(tc.summarize_catalog_for_telegram(top_n=3))
    except Exception:
        pass

    return "\n".join(parts)


# ── 텔레그램 발송 (best-effort) ───────────────────────────────────────
def _send_telegram_best_effort(message: str, title: str = "🤖 EIDOS 자기개선") -> bool:
    """eidos_telegram_bot 사용. 모듈 부재·설정 없으면 False 반환·예외 silent."""
    try:
        from eidos_telegram_bot import get_bot
        bot = get_bot()
        if not bot.is_configured():
            print("[self_improvement] 텔레그램 미설정 — 콘솔 출력만:")
            print(message)
            return False
        # send_notification 이 async — 이벤트 루프 잡아서 실행
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 이벤트 루프 안 (Qt) — task 만 등록
                asyncio.create_task(bot.send_notification(message, title=title))
            else:
                loop.run_until_complete(bot.send_notification(message, title=title))
            return True
        except RuntimeError:
            # 이벤트 루프 없음 — 새로 만들어서 실행
            asyncio.run(bot.send_notification(message, title=title))
            return True
    except Exception as e:
        print(f"[self_improvement] 텔레그램 발송 실패 (graceful): {e}")
        print("[self_improvement] 메시지 (콘솔):")
        print(message)
        return False


# ── tick check (QTimer 가 호출) ───────────────────────────────────────
def tick_check(enabled: bool = True, force: bool = False) -> Optional[dict]:
    """QTimer 가 매 30분마다 호출. enabled=False 면 skip.

    force=True 면 5일 안 지났어도 실행 (디버그·강제 사이클).

    반환: run_cycle 결과 dict 또는 None (실행 안 됨).
    """
    if not enabled:
        return None
    try:
        if not force and not should_run_cycle():
            # 5일 안 지남 — 조용히 skip
            return None
        print("[self_improvement] === 5일 주기 자기개선 사이클 시작 ===")
        result = run_cycle()
        print(f"[self_improvement] 사이클 완료 — telegram_sent={result.get('telegram_sent')}")
        return result
    except Exception as e:
        print(f"[self_improvement] tick_check 실패 (graceful): {e}")
        return None
