# -*- coding: utf-8 -*-
"""
eidos_proactive_advisor.py — 사용자 최상위 목표 기반 자율 선제 제안 루프.

사용자 요청(2026-05-17): "내가 정한 최상위 목표를 달성하기 위해 EIDOS 가
현재 상태/상황을 면밀히 보고 '승준 씨, 오늘은 A작업을 하는 게 좋겠어요.
준비해드릴까요?' 라고 먼저 말하고, 내가 '어 준비해줘' 하면 진짜 내장
브라우저로 자료도 모아주는 것. 하루 20번 이상."

사용자 결정(AskUserQuestion):
  ① 목표 입력 = 설정 UI 필드(top_goal_edit, 이미 존재 — 재사용)
  ② "20+" = 가벼운 *제안* 은 하루 20+, 실제 브라우저 실행은 *승인분만*.
     daily_limit(실행 예산)은 제안 카운트와 분리.
  ③ 제안/승인 표면 = 메인 채팅창 + 텔레그램 둘 다
  ④ 자료수집 후 = **초안까지 자동** (탐색→초안 2단), 외부행동은 별도 승인.

설계: 기존 execution-first 3단 explore 체인을 구부리지 않고, plan.mode=
"advisor_draft" 마커로 _start_explore_chain_for_run 가 *2단 경량 체인*
(browser_read explore → write draft)을 빌드하게 한다. AutonomousRunManager
의 create_pending/approve(SINGLE_CHAIN scope)를 재사용 → 디스패처 스코프
자동통과로 탐색→초안이 게이트 없이 진행(저위험 단계만). 외부행동 단계는
v1 미포함(사고 이력: 빈 payload 자동검색·"결제 지불하다" — 메모리 참조).

안전: kill-switch(eidos_settings.json proactive_advisor.enabled).
제안 자체는 무해(채팅 메시지뿐) — enabled 기본 True. 위험한 부분(브라우저
실행)은 사용자 한 마디 명시 승인 없이는 절대 시작 안 함.
stdlib + 기존 LLM 엔트리만. 모듈 import 부작용 0.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_SETTINGS = os.path.join(_HERE, "eidos_settings.json")
_STATE = os.path.join(_HERE, "eidos_files", "proactive_advisor_state.json")

# settings.json proactive_advisor 블록 기본값 (없으면 이 값).
_DEFAULTS = {
    "enabled": True,            # kill-switch. False = 제안 루프 전면 정지.
    "min_interval_sec": 2400,   # 제안 간 최소 간격(=40분 → 깨어있는 14h 약 21회).
    "daily_suggestion_target": 20,   # soft target — 상한 아님(스팸 게이트는 아래).
    "max_unanswered": 3,        # 미응답 pending 이 이만큼이면 새 제안 보류(스팸 방지).
    "pending_ttl_sec": 86400,   # [#8] 미응답 제안 만료(24h) — 영구 침묵 방지.
}

__all__ = [
    "settings", "load_state", "save_state", "should_suggest_now",
    "generate_suggestion", "generate_suggestion_async", "tick", "tick_async",
    "pending", "approve", "approve_async", "dismiss",
    "build_advisor_stages", "record_executed",
    "drain_notices", "poll_active_run_done", "mark_surfaced",
]


# ── 설정 / 상태 ─────────────────────────────────────────────────────────────

def settings() -> Dict[str, Any]:
    """eidos_settings.json proactive_advisor 블록 (hot-read, graceful)."""
    out = dict(_DEFAULTS)
    try:
        if os.path.exists(_SETTINGS):
            with open(_SETTINGS, "r", encoding="utf-8") as f:
                d = json.load(f) or {}
            blk = d.get("proactive_advisor") or {}
            for k in _DEFAULTS:
                if k in blk:
                    out[k] = blk[k]
    except Exception as e:
        print(f"  ⚠️ [Advisor] settings 읽기 실패 (기본값): {e}")
    return out


def _read_top_goal() -> str:
    """제어실 top_goal — autonomous_runner.read_top_goal 재사용, graceful."""
    try:
        from eidos_autonomous_runner import read_top_goal
        return (read_top_goal() or {}).get("top_goal", "") or ""
    except Exception:
        try:
            with open(_SETTINGS, "r", encoding="utf-8") as f:
                return str((json.load(f) or {}).get("top_goal", "") or "").strip()
        except Exception:
            return ""


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def load_state() -> Dict[str, Any]:
    base = {
        "date": _today(),
        "last_suggest_ts": 0.0,
        "suggest_count_today": 0,
        "exec_count_today": 0,
        "pending": [],          # [{id,title,why,objective,sites,ts}]
        "history": [],          # 최근 처리(승인) — 중복 억제용
        "dismissed": [],        # 무시한 objective — 중복 억제용
        "active_run": None,     # {run_id, objective, ts} — 완료 채팅통지(#5)
        "notices": [],          # 비동기→GUI 채팅 전달용 메시지 큐(스레드안전)
    }
    try:
        if os.path.exists(_STATE):
            with open(_STATE, "r", encoding="utf-8") as f:
                d = json.load(f) or {}
            base.update({k: d.get(k, base[k]) for k in base})
    except Exception as e:
        print(f"  ⚠️ [Advisor] state 로드 실패 (초기화): {e}")
    # 날짜 롤오버 — 일 카운트 리셋(pending 은 유지).
    if base.get("date") != _today():
        base["date"] = _today()
        base["suggest_count_today"] = 0
        base["exec_count_today"] = 0
    # [#8] pending TTL — 24h 지난 미응답 제안은 만료(max_unanswered 잼 해소).
    try:
        _now = time.time()
        _ttl = float(settings().get("pending_ttl_sec", 86400))
        # ts 가 있고 _ttl 초과한 것만 만료. ts 없음/0 → 나이 불명이라 보존
        # (정상 제안은 항상 ts 보유 — 안전측: 모르면 안 지움).
        def _alive(p):
            _t = p.get("ts")
            try:
                _t = float(_t) if _t else 0.0
            except Exception:
                _t = 0.0
            return (_t <= 0) or ((_now - _t) < _ttl)
        _kept = [p for p in (base.get("pending") or []) if _alive(p)]
        if len(_kept) != len(base.get("pending") or []):
            base["pending"] = _kept
    except Exception:
        pass
    return base


def save_state(st: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(_STATE), exist_ok=True)
        tmp = _STATE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False, indent=2)
        os.replace(tmp, _STATE)
    except Exception as e:
        print(f"  ⚠️ [Advisor] state 저장 실패 (무시): {e}")


# ── 게이트 ──────────────────────────────────────────────────────────────────

def should_suggest_now(st: Optional[Dict[str, Any]] = None) -> tuple:
    """(bool, 사유). 제안을 지금 만들어 띄워도 되는가."""
    cfg = settings()
    if not cfg.get("enabled", True):
        return False, "kill-switch OFF (proactive_advisor.enabled=false)"
    if not _read_top_goal().strip():
        return False, "최상위 목표 미설정 (설정 → 최상위 목표)"
    st = st or load_state()
    now = time.time()
    gap = now - float(st.get("last_suggest_ts", 0) or 0)
    if gap < float(cfg.get("min_interval_sec", 2400)):
        return False, f"쿨다운 {int(gap)}s/{cfg.get('min_interval_sec')}s"
    n_unans = len(st.get("pending", []))
    if n_unans >= int(cfg.get("max_unanswered", 3)):
        return False, f"미응답 제안 {n_unans}건 — 스팸 방지 보류"
    # [#7] 일일 제안 목표 도달 시 보류 (죽은 설정 활성화 — soft cap).
    _tgt = int(cfg.get("daily_suggestion_target", 20) or 0)
    if _tgt > 0 and int(st.get("suggest_count_today", 0)) >= _tgt:
        return False, f"오늘 제안 {_tgt}건 도달 — 내일까지 보류"
    # 활성 advisor run 중이면 보류(한 번에 하나 실행).
    try:
        from eidos_autonomous_runner import AutonomousRunManager
        if AutonomousRunManager.get().get_active_run() is not None:
            return False, "활성 자율 실행 진행 중 — 다음 제안 보류"
    except Exception:
        pass
    return True, "ok"


# ── 제안 생성 (다각도 2단 추론, execution-first 아님) ──────────────────────
# memory: diagnosis_eidos_paraphrase_machine — 빈약입력+단일패스 = 매번
# 같은 일반론. 해법 = 전장을 실제로 읽고(observe_battlefield) → 6각도
# 진단(LLM1) → 그 진단 위에서 딱 1개 결정(LLM2) + 모호/중복 게이트.

_ASSESS_PROMPT = """너는 승준 씨의 자율 비서 EIDOS 다. 아래는 지금의 사업 전장(戰場) 정보다. 이걸 **6개 각도에서 냉정하게 진단**하라. 듣기 좋은 일반론 금지 — 구체 사실/숫자/이미 한 것에 근거.

[최상위 목표]
{goal}

[성공 정의 / 동기]
{success}

[상황 로그(시간순 내부 판단)]
{situation}

[이미 만든 산출물(중복 회피용 — 같은 걸 또 제안하지 마라)]
{artifacts}

[이전 자율 제안 이력 / 처리 결과]
{history}

[EIDOS 자기 자산(차별점 — 제안은 이걸 레버리지해야)]
{assets}

[자기조직화로 굳어진 절차 패턴 — verb substrate 에 지식이 쌓여 결정화된 동사 시퀀스. 떠오르는 구조이니 4)자기자산 레버리지·6)고임팩트 판단에 적극 활용]
{emergent}

[최근 채팅/상태 맥락]
{recent}

6각도 각각 2~3문장으로 진단:
1) 목표 대비 현재 갭 — 성공정의에서 가장 먼 지점은?
2) 이미 한 것/중복 — 위 산출물·이력에서 반복되거나 끝난 일은? (이건 제외)
3) 경쟁·외부 — 지금 외부에서 변한/확인 필요한 것?
4) 자기자산 레버리지 — EIDOS 고유 자산으로 남보다 잘할 수 있는 한 수?
5) 시급성·병목 — 지금 안 하면 비싸지는 것 / 다른 일 막는 병목?
6) 저비용·고임팩트 — 자료수집+초안만으로 큰 진척 나는 한 가지?

각 각도 결론을 그대로 쓰되, 마지막에 "[핵심 병목 1줄]" 로 가장 중요한 한 가지를 못박아라."""

_DECIDE_PROMPT = """아래는 너(EIDOS)가 방금 작성한 전장 진단서다. 이 진단에 **직접 근거해서**, 승준 씨가 "준비해줘" 하면 내장 브라우저로 자료수집+초안까지 바로 할 수 있는 **딱 1개의 작업**을 결정하라.

[최상위 목표]
{goal}

[전장 진단서]
{assessment}

[이미 한 것(절대 다시 제안 금지)]
{artifacts}

엄격 규칙:
- 진단서의 "[핵심 병목]" 을 푸는 작업이어야 한다. 진단과 무관한 일반 작업 금지.
- "분석하세요/조사하세요/검토하세요" 같이 대상·산출물 없는 모호한 작업 금지. objective 에는 **구체 대상 + 무엇을 모아 무슨 초안을 낼지** 가 있어야 한다.
- 위 [이미 한 것] 과 실질 동일하면 다른 각도의 작업을 골라라.
- 결제/회원가입/외부SNS게시/DM발송/광고집행 류 위험 행위 금지.

JSON 만 출력:
{{"title": "승준 씨, 오늘은 ~~ 하는 게 좋겠어요. 준비해드릴까요?", "why_now": "왜 지금 이게 핵심 병목을 푸는지 1~2문장", "why_this": "왜 다른 후보 아닌 이것인지 1문장", "contributes": "성공 정의에 어떻게 기여하는지 1문장", "objective": "구체 대상+수집+초안 형태의 작업 1줄", "sites": ["참고 도메인 0~4개"]}}"""


# [#31] chat 페르소나(EIDOS_SYSTEM_PROMPT) 덮어쓰는 task 시스템프롬프트.
#   진단(최신로그 22:46): system_prompt 미지정 → 모델이 분석 지시를
#   '잡담 턴'으로 받아 "~해드리겠습니다" 75자 응대 후 STOP. 이걸로 강제.
_TASK_SYS = (
    "너는 EIDOS 의 분석·추론 엔진이다. 대화체 인사·서론·맺음말·"
    "\"~해드리겠습니다\" 류 응대를 절대 쓰지 마라. 요청된 구조(또는 "
    "JSON)만 즉시, 끝까지, 빠짐없이 출력한다. 출력 외 다른 말 금지."
)


def _llm(prompt: str, system: str = "", mime: str = "") -> str:
    """[통합 Phase1] 단일 LLM 게이트웨이 경유 — 정책(task system_prompt+
    use_cache=False+8192+JSON) 1곳 고정(eidos_llm_gateway). 게이트웨이
    불가 시에만 옛 직호출 폴백(kill-switch). 동작 불변·중앙화."""
    try:
        import eidos_llm_gateway as _gw
        return _gw.reason_sync(
            prompt, role="reason", schema=bool(mime), timeout=60)
    except Exception as _e_gw:
        print(f"  ⚠️ [Advisor] 게이트웨이 불가 → 옛 경로 폴백: {_e_gw}")
    # ── 폴백 (게이트웨이 import 실패 시에만) ──
    try:
        from llm_module import get_llm_response_async as _g
    except Exception:
        try:
            from eidos_patrol_agent import get_llm_response_async as _g  # type: ignore
        except Exception:
            return ""
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_g(
                prompt, max_tokens=8192, timeout=60, use_cache=False,
                system_prompt=(system or _TASK_SYS),
                response_mime_type=(mime or None))) or ""
        finally:
            loop.close()
    except Exception as e:
        print(f"  ⚠️ [Advisor] LLM 호출 실패 (graceful): {e}")
        return ""


async def _llm_async(prompt: str, system: str = "", mime: str = "") -> str:
    """[#3/#4] 비동기 LLM — new_event_loop 금지. 호출자(worker.submit_task)
    의 기존 루프에서 await. GUI 스레드 블로킹·루프 충돌 제거."""
    # [통합 Phase1] 단일 LLM 게이트웨이 경유 — 정책(task system_prompt+
    #   use_cache=False+8192+JSON) 1곳 고정. 이번 세션 #30/#31/#33 정책이
    #   eidos_llm_gateway 에 중앙화됨(동작 불변). 게이트웨이 불가 시에만
    #   옛 직호출 폴백(kill-switch).
    try:
        import eidos_llm_gateway as _gw
        return await _gw.reason_async(
            prompt, role="reason", schema=bool(mime), timeout=60)
    except Exception as _e_gw:
        print(f"  ⚠️ [Advisor] 게이트웨이 불가 → 옛 경로 폴백: {_e_gw}")
    try:
        from llm_module import get_llm_response_async as _g
    except Exception:
        try:
            from eidos_patrol_agent import get_llm_response_async as _g  # type: ignore
        except Exception:
            return ""
    try:
        return (await _g(prompt, max_tokens=8192, timeout=60,
                         use_cache=False,
                         system_prompt=(system or _TASK_SYS),
                         response_mime_type=(mime or None))) or ""
    except Exception as e:
        print(f"  ⚠️ [Advisor] LLM(async) 호출 실패 (graceful): {e}")
        return ""


def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    s = raw.strip()
    if "```" in s:
        s = s.split("```")[1] if s.count("```") >= 2 else s
        s = s.replace("json", "", 1) if s.lstrip().startswith("json") else s
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        d = json.loads(s[a:b + 1])
        return d if isinstance(d, dict) else None
    except Exception:
        return None


_CTX_PATH = os.path.join(_HERE, "eidos_files", "schedule_planner_context.json")
_FILES_DIR = os.path.join(_HERE, "eidos_files")
_GENERIC = ("분석", "조사", "검토", "파악", "정리", "알아보", "살펴",
            "확인", "리서치", "이해", "고민")


def _ctx() -> Dict[str, Any]:
    try:
        with open(_CTX_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


_INV_CACHE: Dict[str, Any] = {"ts": 0.0, "key": None, "val": []}
_LAST_GATE_LOG: str = ""   # [#25] tick 게이트 사유 변동 로깅용 (스팸 방지)


def _artifact_inventory(days: int = 30, limit: int = 25) -> List[str]:
    """eidos_files 의 최근 산출물 제목+날짜 — '이미 한 것' 중복 회피용.
    [#9] 60초 캐시 — 한 제안 사이클 내 listdir/stat 반복 I/O 제거."""
    _now = time.time()
    # dir mtime 을 키에 포함 — 새 산출물 추가 시 캐시 즉시 무효(정확성),
    # 변화 없으면 60초 캐시(반복 I/O 제거).
    try:
        _dm = os.path.getmtime(_FILES_DIR)
    except Exception:
        _dm = 0.0
    _key = (days, limit, _dm)
    if (_INV_CACHE.get("key") == _key
            and (_now - float(_INV_CACHE.get("ts", 0))) < 60):
        return list(_INV_CACHE.get("val") or [])
    out: List[str] = []
    try:
        cut = time.time() - days * 86400
        rows = []
        for fn in os.listdir(_FILES_DIR):
            if not (fn.endswith(".md") or fn.endswith(".txt")):
                continue
            fp = os.path.join(_FILES_DIR, fn)
            try:
                m = os.path.getmtime(fp)
            except Exception:
                continue
            if m < cut:
                continue
            rows.append((m, fn))
        for m, fn in sorted(rows, reverse=True)[:limit]:
            out.append(f"{time.strftime('%m-%d', time.localtime(m))} {fn}")
    except Exception:
        pass
    _INV_CACHE.update({"ts": _now, "key": _key, "val": list(out)})
    return out


def observe_battlefield(core: Any = None, recent_context: str = "") -> Dict[str, str]:
    """버려지던 신호를 모두 읽어 전장 정보 dict 반환. graceful.

    memory: diagnosis_eidos_paraphrase_machine — 빈약 입력이 일반론의
    근본원인. schedule_planner_context 의 situation 로그·성공정의·
    산출물 인벤토리·자기자산·이력을 실제로 공급한다."""
    c = _ctx()
    goal = _read_top_goal().strip()
    success = "\n".join(x for x in [
        (c.get("why_anchor") or "").strip(),
        (c.get("situation_internal") or "").strip(),
    ] if x) or "(성공 정의 미설정)"
    situation = (c.get("situation") or "").strip()
    if not situation:
        # 폴백: read_top_goal 의 situation_* 라도.
        try:
            from eidos_autonomous_runner import read_top_goal
            rc = read_top_goal() or {}
            situation = (rc.get("situation_internal")
                         or rc.get("situation_external") or "").strip()
        except Exception:
            situation = ""
    arts = _artifact_inventory()
    st = load_state()
    hist = st.get("history") or []
    dism = st.get("dismissed") or []
    hist_txt = "\n".join(
        ["[승인됨] " + h for h in hist[-12:]]
        + ["[무시됨] " + d for d in dism[-8:]]
    ) or "(이력 없음)"
    assets = ""
    try:
        from eidos_synthesis_guard import SELF_ASSET_CARD
        assets = SELF_ASSET_CARD
    except Exception:
        assets = "(자기자산 카드 미가용)"
    # world_model KPI breach (있으면 — 현재 데이터는 대체로 비어 graceful).
    wm_note = ""
    try:
        from eidos_verb_goal_engine import build_self_state_from_core
        ss = build_self_state_from_core(core)
        kf = ss.get("kpi") if hasattr(ss, "get") else None
        kf = kf() if callable(kf) else kf
        if isinstance(kf, dict):
            br = [k for k, v in kf.items()
                  if isinstance(v, dict) and v.get("status")
                  in ("warn_high", "warn_low", "off_target")]
            if br:
                wm_note = "KPI 이상: " + ", ".join(br[:8])
    except Exception:
        wm_note = ""
    recent = ((recent_context or "").strip()
              + (("\n" + wm_note) if wm_note else "")) or "(없음)"
    # [통합 2026-05-17] verb substrate 결정화 스켈레톤을 *읽기전용 신호* 로
    # 흡수. 목표 생성은 advisor 가 한다(메모리상 substrate 자기조직화
    # *생성기* 는 잘못된 토대 — 부활 X). 지식 주입(📚)이 쌓여 굳어진
    # 절차 패턴을 전장 한 각도로만 공급. graceful — 없으면 표시 안 함.
    emergent = "(아직 결정화된 패턴 없음 — 📚 지식 주입이 쌓이면 생김)"
    try:
        import eidos_verb_substrate as _VS
        sx = _VS.get_substrate()
        rows = []
        for s in (getattr(sx, "skeletons", []) or []):
            if getattr(s, "crystallized", False):
                chain = " → ".join(
                    v.split("(")[0] for v in getattr(s, "verbs", []))
                rows.append((getattr(s, "cohesion", 0.0), chain))
        rows.sort(reverse=True)
        if rows:
            emergent = "\n".join(
                f"- [{c:.2f}] {ch}" for c, ch in rows[:12])
    except Exception:
        pass
    return {
        "goal": goal[:600] or "(미설정)",
        "success": success[:800],
        "situation": (situation[-2400:] if situation else "(상황 로그 없음)"),
        "artifacts": ("\n".join(arts) if arts else "(없음)")[:1500],
        "history": hist_txt[:1200],
        "assets": assets[:1400],
        "emergent": emergent[:1200],
        "recent": recent[:800],
    }


def _assess_battlefield(bf: Dict[str, str]) -> str:
    """LLM 1단 — 6각도 진단서. 실패 시 빈 문자열(graceful)."""
    try:
        return _llm(_ASSESS_PROMPT.format(**bf)).strip()
    except Exception as e:
        print(f"  ⚠️ [Advisor] assess 실패 (graceful): {e}")
        return ""


def _is_specific(objective: str) -> bool:
    """모호 작업 거부 — 일반동사만 있고 구체 대상/산출물 없으면 False."""
    o = (objective or "").strip()
    if len(o) < 14:
        return False
    has_generic = any(g in o for g in _GENERIC)
    # 구체성 신호: 고유명사/숫자/플랫폼/산출물 단어.
    concrete = any(s in o for s in (
        "크몽", "kmong", "초안", "페이지", "상세", "가격", "리뷰", "폼",
        "포트폴리오", "설문", "경쟁사", "고객", "KPI", "보고서", "표",
        "목록", "비교", "스크립트", ".md", "URL", "http",
    )) or any(ch.isdigit() for ch in o)
    # 일반동사뿐이고 구체 신호 없으면 거부.
    if has_generic and not concrete:
        return False
    return True


def _tok(s: str) -> set:
    import re
    return set(t for t in re.findall(r"[가-힣A-Za-z0-9]{2,}", s or ""))


def _is_redundant(objective: str) -> bool:
    """최근 산출물 제목 + 처리 이력과 토큰 과반 겹치면 중복으로 본다."""
    ot = _tok(objective)
    if not ot:
        return False
    pool = _artifact_inventory()
    st = load_state()
    pool = pool + list(st.get("history") or []) + list(
        st.get("dismissed") or [])
    for p in pool:
        pt = _tok(p)
        if not pt:
            continue
        inter = len(ot & pt)
        if inter and inter / max(1, len(ot)) >= 0.6:
            return True
    return False


def generate_suggestion(core: Any = None,
                        recent_context: str = "",
                        _retry: int = 0) -> Optional[Dict[str, Any]]:
    """다각도 2단 추론 → 구체 제안 1건. 실패/모호/중복 시 None (graceful).

    1) observe_battlefield — 버려지던 신호 전부 수집
    2) _assess_battlefield (LLM1) — 6각도 진단서
    3) decide (LLM2) — 진단 위에서 딱 1개 결정
    4) 구체성·중복 게이트 — 모호하면 None, 중복이면 1회 재생성."""
    goal = _read_top_goal().strip()
    if not goal:
        return None
    bf = observe_battlefield(core, recent_context)
    assessment = _assess_battlefield(bf)
    if not assessment:
        return None
    d = _parse_json(_llm(_DECIDE_PROMPT.format(
        goal=bf["goal"], assessment=assessment[:3000],
        artifacts=bf["artifacts"],
    ), mime="application/json"))   # [#31] JSON 강제 — 잡담·조기STOP 차단
    if not d:
        return None
    title = str(d.get("title", "") or "").strip()
    obj = str(d.get("objective", "") or "").strip()
    if not title or not obj:
        return None
    if not _is_specific(obj):
        print(f"  🚫 [Advisor] 모호 제안 거부: {obj[:60]}")
        return None
    if _is_redundant(obj):
        if _retry < 1:
            print(f"  🔁 [Advisor] 중복 제안 — 재생성 1회: {obj[:50]}")
            return generate_suggestion(core, recent_context, _retry + 1)
        print(f"  🚫 [Advisor] 중복 제안 — 스킵: {obj[:50]}")
        return None
    sites = d.get("sites") or []
    if not isinstance(sites, list):
        sites = []
    why = " / ".join(x for x in [
        str(d.get("why_now", "") or "").strip(),
        str(d.get("why_this", "") or "").strip(),
        str(d.get("contributes", "") or "").strip(),
    ] if x)
    return {
        "id": uuid.uuid4().hex[:10],
        "title": title,
        "why": why,
        "objective": obj,
        "sites": [str(x) for x in sites][:4],
        "assessment": assessment[:1200],
        "ts": time.time(),
        "surfaced": False,   # [#3] GUI 틱이 채팅 표면화 후 True (스레드안전)
    }


# ── tick (주기 진입점, chat_gui 30초 틱이 호출) ─────────────────────────────

def tick(core: Any = None,
         on_suggest: Optional[Callable[[Dict[str, Any]], None]] = None,
         recent_context: str = "") -> Optional[Dict[str, Any]]:
    """게이트 통과 시 제안 1건 생성→pending 등록→on_suggest 콜백.
    반환: 생성된 제안 dict 또는 None. 전 구간 graceful."""
    try:
        ok, why = should_suggest_now()
        if not ok:
            return None
        sug = generate_suggestion(core, recent_context)
        if not sug:
            return None
        st = load_state()
        st["pending"] = (st.get("pending") or []) + [sug]
        st["last_suggest_ts"] = time.time()
        st["suggest_count_today"] = int(st.get("suggest_count_today", 0)) + 1
        save_state(st)
        if on_suggest:
            try:
                on_suggest(sug)
            except Exception as e:
                print(f"  ⚠️ [Advisor] on_suggest 콜백 실패 (무시): {e}")
        return sug
    except Exception as e:
        print(f"  ⚠️ [Advisor] tick 실패 (graceful): {e}")
        return None


async def generate_suggestion_async(
    core: Any = None, recent_context: str = "", _retry: int = 0
) -> Optional[Dict[str, Any]]:
    """[#3] generate_suggestion 의 비동기판 — _llm_async 사용(블로킹 0).
    로직/게이트 동일."""
    # [#32 결정적 계측] 4겹 블라인드 패치 종료 — 실 응답·실패지점 박제.
    #   다음 로그가 추측 없이 정확한 사인 제공. (간접추론 한계 돌파)
    print("  🔬 [Advisor#dbg] code=v32 generate_async 진입")
    goal = _read_top_goal().strip()
    if not goal:
        print("  🔬 [Advisor#dbg] None@goal_empty")
        return None
    bf = observe_battlefield(core, recent_context)
    _ap = _ASSESS_PROMPT.format(**bf)
    print(f"  🔬 [Advisor#dbg] assess prompt={len(_ap)}자 호출")
    _ar = await _llm_async(_ap)
    assessment = (_ar or "").strip()
    print(f"  🔬 [Advisor#dbg] assess resp={len(assessment)}자 "
          f"repr={assessment[:200]!r}")
    if not assessment:
        print("  🔬 [Advisor#dbg] None@assessment_empty")
        return None
    _dp = _DECIDE_PROMPT.format(
        goal=bf["goal"], assessment=assessment[:3000],
        artifacts=bf["artifacts"])
    _dr = await _llm_async(_dp, mime="application/json")
    print(f"  🔬 [Advisor#dbg] decide resp={len(_dr or '')}자 "
          f"repr={(_dr or '')[:200]!r}")
    d = _parse_json(_dr)
    if not d:
        print("  🔬 [Advisor#dbg] None@decide_parse_fail")
        return None
    title = str(d.get("title", "") or "").strip()
    obj = str(d.get("objective", "") or "").strip()
    if not title or not obj:
        print(f"  🔬 [Advisor#dbg] None@missing title/obj "
              f"(title={title[:40]!r} obj={obj[:40]!r})")
        return None
    print(f"  🔬 [Advisor#dbg] 통과 → obj={obj[:60]!r}")
    if not _is_specific(obj):
        print(f"  🚫 [Advisor] 모호 제안 거부: {obj[:60]}")
        return None
    if _is_redundant(obj):
        if _retry < 1:
            print(f"  🔁 [Advisor] 중복 제안 — 재생성 1회: {obj[:50]}")
            return await generate_suggestion_async(
                core, recent_context, _retry + 1)
        print(f"  🚫 [Advisor] 중복 제안 — 스킵: {obj[:50]}")
        return None
    sites = d.get("sites") or []
    if not isinstance(sites, list):
        sites = []
    why = " / ".join(x for x in [
        str(d.get("why_now", "") or "").strip(),
        str(d.get("why_this", "") or "").strip(),
        str(d.get("contributes", "") or "").strip(),
    ] if x)
    return {
        "id": uuid.uuid4().hex[:10], "title": title, "why": why,
        "objective": obj, "sites": [str(x) for x in sites][:4],
        "assessment": assessment[:1200], "ts": time.time(),
        "surfaced": False,
    }


async def tick_async(core: Any = None,
                     recent_context: str = "") -> Optional[Dict[str, Any]]:
    """[#3] worker.submit_task 로 호출되는 비동기 틱 — GUI 블로킹 0.
    제안 생성→pending 등록(surfaced=False, GUI 틱이 채팅 표면화)→
    텔레그램 발송(#1, 여기서 await — 올바른 비동기 경로). graceful."""
    try:
        ok, _why = should_suggest_now()
        # [#25 관측] 게이트 사유 — 변동 시 1회만(30초 스팸 방지). 다음
        #   실세션 로그가 'advisor 가 왜 침묵했는지'를 확정 알려주게.
        global _LAST_GATE_LOG
        if _why != _LAST_GATE_LOG:
            print(f"  🛰️ [Advisor] tick gate: ok={ok} | {_why}")
            _LAST_GATE_LOG = _why
        if not ok:
            return None
        print("  🛰️ [Advisor] 게이트 통과 — 제안 생성 시도(LLM 2회)")
        sug = await generate_suggestion_async(core, recent_context)
        if not sug:
            print("  🛰️ [Advisor] 제안 생성 None "
                  "(LLM 실패/모호·중복 게이트 거부 — 위 🚫 로그 참조)")
            return None
        print(f"  🛰️ [Advisor] 제안 생성 OK → pending 등록: "
              f"{str(sug.get('objective',''))[:60]}")
        st = load_state()
        st["pending"] = (st.get("pending") or []) + [sug]
        st["last_suggest_ts"] = time.time()
        st["suggest_count_today"] = int(st.get("suggest_count_today", 0)) + 1
        save_state(st)
        # [#1] 텔레그램 — send_message_sync(없는 메서드) 폐기,
        #   실제 API await bot.send_notification(message, title=).
        try:
            from eidos_telegram_bot import get_bot
            _b = get_bot()
            if _b and _b.is_configured():
                await _b.send_notification(
                    f"{sug.get('title','')}\n\n"
                    f"준비를 원하시면 채팅창에 '준비해줘' 라고 답하시면 "
                    f"자료수집을 시작합니다.",
                    title="EIDOS 자율 제안",
                )
        except Exception as _e_tg:
            print(f"  ⚠️ [Advisor] 텔레그램 발송 실패 (graceful): {_e_tg}")
        return sug
    except Exception as e:
        print(f"  ⚠️ [Advisor] tick_async 실패 (graceful): {e}")
        return None


def pending() -> List[Dict[str, Any]]:
    return list(load_state().get("pending", []) or [])


def mark_surfaced(ids: List[str]) -> None:
    """[#3] GUI 틱(메인 스레드)이 채팅에 띄운 제안을 surfaced=True 로.
    중복 표시 방지. 스레드안전(state 경유)."""
    try:
        st = load_state()
        ch = False
        for p in st.get("pending", []) or []:
            if p.get("id") in ids and not p.get("surfaced"):
                p["surfaced"] = True
                ch = True
        if ch:
            save_state(st)
    except Exception:
        pass


def _pop_pending(sid: str) -> Optional[Dict[str, Any]]:
    st = load_state()
    keep, found = [], None
    for p in st.get("pending", []) or []:
        if p.get("id") == sid and found is None:
            found = p
        else:
            keep.append(p)
    if found is not None:
        st["pending"] = keep
        save_state(st)
    return found


def dismiss(sid: str) -> bool:
    return _pop_pending(sid) is not None


def record_executed() -> None:
    st = load_state()
    st["exec_count_today"] = int(st.get("exec_count_today", 0)) + 1
    save_state(st)


# ── 승인 → 2단 경량 체인 ────────────────────────────────────────────────────

def build_advisor_stages(plan: Any) -> List[Any]:
    """advisor_draft 모드 전용 2단 StageSpec — execution-first 미적용.
       (0) browser_read 'explore'  — 내장 브라우저 자료 수집
       (1) write 'draft'           — 수집 결과로 초안/문서 작성 (여기서 정지)
    외부행동 단계 없음(v1, 사고 이력 차단). autonomous_runner 가 import."""
    from eidos_mission_chain import StageSpec
    obj = (getattr(plan, "objective", "") or "").strip()
    sites = getattr(plan, "sites", None) or []
    sites_hint = ", ".join(sites[:5]) if sites else "(검색 엔진 자율 결정)"
    explore_prompt = (
        f"{obj}\n\n방문/참고 사이트: {sites_hint}\n"
        f"내장 브라우저로 실제 사이트를 탐색해 검증 가능한 자료를 모아 "
        f"explore_report.md 로 저장하라. 외부에서 읽은 데이터만 인용 — "
        f"환각 작문 금지. 다음 단계(초안 작성)에서 바로 쓸 형태로 정리."
    )
    draft_prompt = (
        f"위 탐색 결과(explore_report)를 근거로 '{obj}' 의 초안/정리 문서를 "
        f"작성해 advisor_draft.md 로 저장하라.\n"
        f"- 원천을 풀어쓰기만 하지 말고 종합·판단·다음 액션 제안을 포함.\n"
        f"- 외부 게시/전송/제출 같은 행위는 절대 하지 마라(여기서 정지). "
        f"사용자가 별도로 승인하면 그때 외부 행위를 진행한다."
    )
    return [
        StageSpec(
            name="explore", stage_index=0,
            task_type="browser_read", verb="READ",
            prompt_template=explore_prompt,
            depends_on=[], input_keys=[],
            output_keys=["explore_report"],
            auto_approve=True,
            description=f"자율 자료수집: {obj[:60]}",
            expected_duration_sec=600,
            risk_level="low",
        ),
        StageSpec(
            name="draft", stage_index=1,
            task_type="write", verb="COMPOSE",
            prompt_template=draft_prompt,
            depends_on=[0],
            input_keys=["explore.explore_report"],
            output_keys=["advisor_draft"],
            auto_approve=True,
            description=f"초안 작성: {obj[:50]}",
            expected_duration_sec=240,
            risk_level="low",
        ),
    ]


def _daily_limit() -> int:
    """[#7] autoexec.daily_limit (실행 예산 — 제안 카운트와 분리)."""
    try:
        with open(_SETTINGS, "r", encoding="utf-8") as f:
            return int(((json.load(f) or {}).get("autoexec") or {}).get(
                "daily_limit", 30) or 0)
    except Exception:
        return 30


def _push_notice(text: str) -> None:
    """비동기 컨텍스트→GUI 채팅 전달(스레드안전: state 경유, GUI 틱이 drain)."""
    try:
        st = load_state()
        st["notices"] = (st.get("notices") or [])[-20:] + [str(text)]
        save_state(st)
    except Exception:
        pass


def drain_notices() -> List[str]:
    """GUI 틱(메인 스레드)이 호출 — 누적 notice 를 비우고 반환."""
    st = load_state()
    ns = list(st.get("notices") or [])
    if ns:
        st["notices"] = []
        save_state(st)
    return ns


async def approve_async(sid: str, core: Any = None) -> Optional[Any]:
    """[#3/#4] 비동기 승인 — worker.submit_task 로 호출(GUI 블로킹 0,
    new_event_loop 금지). #6 bootstrap 가드 + #7 daily_limit 집행 +
    #5 active_run 기록(완료 채팅통지용). 거부 사유는 _push_notice 로 채팅 전달."""
    sug = _pop_pending(sid)
    if not sug:
        return None
    try:
        from eidos_autonomous_runner import (
            AutonomousPlan, AutonomousRunManager, ScopeType,
        )
    except Exception as e:
        print(f"  ⚠️ [Advisor] autonomous_runner 로드 실패: {e}")
        _push_notice("⚠️ 자율 실행 모듈 로드 실패 — 잠시 후 다시 제안할게요.")
        return None
    mgr = AutonomousRunManager.get()
    # [#6] chain_starter 미등록(스케줄 모니터 부팅 전) — 조용한 무동작 방지.
    if getattr(mgr, "_chain_starter", None) is None:
        print("  ⚠️ [Advisor] chain_starter 미등록 — 승인 보류")
        _push_notice("⏳ 아직 자율 실행 엔진이 준비 중이에요. "
                     "곧 다시 제안드릴게요. (제안은 보존됨)")
        # pending 복원 — 사용자가 다시 승인할 수 있게.
        st0 = load_state()
        st0["pending"] = (st0.get("pending") or []) + [sug]
        save_state(st0)
        return None
    # [#7] 실행 예산(daily_limit) 집행 — 제안과 분리된 '실행' 한도.
    st = load_state()
    _lim = _daily_limit()
    if _lim > 0 and int(st.get("exec_count_today", 0)) >= _lim:
        print(f"  🛑 [Advisor] 실행 일일한도 {_lim} 도달 — 승인 거부")
        _push_notice(f"🛑 오늘 자율 실행 한도({_lim}회)에 도달했어요. "
                     f"내일 이어가거나 설정에서 한도를 올릴 수 있어요.")
        return None
    plan = AutonomousPlan(
        objective=sug.get("objective", "")[:200],
        stages=[f"1. 자료수집: {sug.get('objective','')[:60]}", "2. 초안 작성"],
        sites=list(sug.get("sites") or []),
        estimated_duration_min=20,
    )
    try:
        setattr(plan, "mode", "advisor_draft")
    except Exception:
        pass
    try:
        run = mgr.create_pending(
            seed_goal=sug.get("objective", "")[:200], plan=plan)
        await mgr.approve(run.run_id, ScopeType.SINGLE_CHAIN)
        record_executed()
        st2 = load_state()
        st2["history"] = (st2.get("history") or [])[-50:] + [
            sug.get("objective", "")[:80]]
        # [#5] 완료 채팅통지용 — GUI 틱이 이 run 상태를 폴링.
        st2["active_run"] = {
            "run_id": getattr(run, "run_id", ""),
            "objective": sug.get("objective", "")[:120],
            "ts": time.time(),
        }
        save_state(st2)
        return run
    except Exception as e:
        print(f"  ⚠️ [Advisor] 승인→체인 시작 실패 (graceful): {e}")
        _push_notice("⚠️ 자율 실행 시작에 실패했어요. 잠시 후 다시 제안할게요.")
        return None


def poll_active_run_done() -> Optional[Dict[str, Any]]:
    """[#5] GUI 틱(메인 스레드)이 호출 — active_run 이 COMPLETED/종료면
    그 정보를 반환하고 state 에서 제거(1회성 채팅 통지용). graceful."""
    try:
        st = load_state()
        ar = st.get("active_run")
        if not ar or not ar.get("run_id"):
            return None
        from eidos_autonomous_runner import AutonomousRunManager
        r = AutonomousRunManager.get().get_run(ar["run_id"])
        if r is None:
            return None
        status = str(getattr(r, "status", "") or "")
        # [좀비 방지 — 2026-06-11] run 은 살아있는데 묶인 체인이 이미 terminal 이면
        #   (chain_failed 이벤트 누락 등) reconcile → run 을 abort 하고 state 정리.
        if status.lower() not in ("completed", "aborted", "rejected", "failed"):
            _cid = str(getattr(r, "chain_id", "") or "")
            if _cid:
                try:
                    from eidos_mission_chain import get_chain_registry
                    _ch = get_chain_registry().get(_cid)
                    _cs = str(getattr(_ch, "status", "") or "").lower()
                    if _cs in ("aborted", "failed", "completed"):
                        _mgr = AutonomousRunManager.get()
                        _act = _mgr.get_active_run()
                        if _act is not None and getattr(_act, "run_id", "") == ar["run_id"]:
                            _mgr.abort_active(reason=f"chain_terminal_reconcile:{_cs}")
                        status = "aborted"
                except Exception:
                    pass
        if status in ("completed", "COMPLETED", "aborted", "ABORTED",
                      "rejected", "REJECTED", "failed", "FAILED"):
            st["active_run"] = None
            save_state(st)
            return {"objective": ar.get("objective", ""), "status": status}
        return None
    except Exception:
        return None


def approve(sid: str, core: Any = None) -> Optional[Any]:
    """사용자 한 마디 승인 → AutonomousRun 생성+승인(SINGLE_CHAIN scope).
    chain_starter(_start_explore_chain_for_run)가 plan.mode='advisor_draft'
    분기로 2단 체인 빌드. 반환: AutonomousRun 또는 None. graceful."""
    sug = _pop_pending(sid)
    if not sug:
        return None
    try:
        from eidos_autonomous_runner import (
            AutonomousPlan, AutonomousRunManager, ScopeType,
        )
    except Exception as e:
        print(f"  ⚠️ [Advisor] autonomous_runner 로드 실패: {e}")
        return None
    plan = AutonomousPlan(
        objective=sug.get("objective", "")[:200],
        stages=[f"1. 자료수집: {sug.get('objective','')[:60]}", "2. 초안 작성"],
        sites=list(sug.get("sites") or []),
        estimated_duration_min=20,
    )
    try:
        setattr(plan, "mode", "advisor_draft")  # 2단 경량 체인 마커
    except Exception:
        pass
    try:
        mgr = AutonomousRunManager.get()
        run = mgr.create_pending(
            seed_goal=sug.get("objective", "")[:200], plan=plan
        )
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                mgr.approve(run.run_id, ScopeType.SINGLE_CHAIN)
            )
        finally:
            loop.close()
        record_executed()
        # history 에 sig 남겨 동일 제안 재생성 억제.
        st = load_state()
        st["history"] = (st.get("history") or [])[-50:] + [
            sug.get("objective", "")[:80]
        ]
        save_state(st)
        return run
    except Exception as e:
        print(f"  ⚠️ [Advisor] 승인→체인 시작 실패 (graceful): {e}")
        return None
