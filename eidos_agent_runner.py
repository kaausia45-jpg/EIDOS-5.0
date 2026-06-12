# eidos_agent_runner.py
# [2026-05-26 Phase 0-D MVP] 다중 행위자 ToM 에이전트 — 1 tick 실행 runner.
#
# 1 tick 의 흐름:
#   1. 관찰 (MVP: state.json 로드만·외부 polling 없음·Phase 1 에서 확장)
#   2. ToM 예측 + EIDOS 결정 — 단일 LLM 호출로 동시 출력 (토큰 절약·일관성)
#   3. 실행 — eidos.tool.* dispatch (MVP: 외부효과는 dry-run·meta + llm.write + ask_user 만 실)
#   4. belief 갱신 + state.json 저장 + history.jsonl 3 record append
#
# 안전 정책:
#   - safety_mode="paranoid": 모든 외부효과 dry-run·ask_user 만 HITL 통과
#   - safety_mode="normal" (default): web/email/telegram dry-run·llm.write/meta/ask_user 실 실행
#   - safety_mode="yolo": 모두 실 실행 (사용자가 명시 토글 후 가능)
#
# 옛 eidos_canvas_runner._exec_* (verb 함수) 와는 별도 — 이쪽은 *워크플로우* 결정적 dispatch,
# agent_runner 는 *agent loop* — LLM 결정 + 안전 게이트 + dry-run.

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import os
from typing import Any, Awaitable, Callable, Optional


# ── 상수 ────────────────────────────────────────────────────────────
STATUS_OK         = "ok"
STATUS_FAIL       = "fail"
STATUS_SKIP       = "skip"
STATUS_DRY_RUN    = "dry_run"
STATUS_HITL       = "hitl"   # 사용자 응답 대기 후 처리됨

# 외부효과 위험 action_id prefix — dry-run 대상 (safety_mode 별 토글)
_EXTERNAL_EFFECT_PREFIXES = (
    "eidos.tool.web.",
    "eidos.tool.message.",
    "eidos.tool.file.delete",       # 파일 삭제 (현재 default 에 없으나 미래 대비)
    # [2026-05-27 자율 개발 모드] subprocess 실행 — 임의 명령 실 실행이라 PROMPT 게이트.
    # dev.scaffold_project / write_code / package_deliverable 는 outputs/ path-confine 이라
    # 외부효과 아님 — execute_eidos_action 안에서 inline 처리.
    "eidos.tool.dev.run_tests",
)

# [2026-05-30] 사용자 승인 없이 normal 모드에서 실제 실행하는 외부효과 allowlist.
# 사용자 결정: 시장조사(web 읽기)·로컬 테스트 실행은 활성화 / 발송(message.*)은 제외
# (draft_to_clipboard 로 초안 → 사용자가 직접 붙여넣기). message.* 는 여기 없으므로
# 아래 dry-run 유지 (outward 안전). settings.json 의 "agent_autorun_external" (list)
# 로 override 가능 — 예: web 끄고 싶으면 ["eidos.tool.dev.run_tests"] 만 남김.
_DEFAULT_AUTORUN_EXTERNAL = (
    "eidos.tool.web.navigate",
    "eidos.tool.web.read_page",
    "eidos.tool.web.search",        # 2026-06-01 SerpAPI 검색 (read-only·자동 실행)
    "eidos.tool.web.scroll",
    "eidos.tool.web.go_back",
    "eidos.tool.web.wait",
    "eidos.tool.dev.run_tests",
)


def _load_autorun_external() -> tuple:
    """settings.json 의 agent_autorun_external override (없으면 default)."""
    try:
        import json as _json
        _p = os.path.join("eidos_files", "settings.json")
        if os.path.exists(_p):
            with open(_p, "r", encoding="utf-8") as _f:
                _s = _json.load(_f) or {}
            _v = _s.get("agent_autorun_external")
            if isinstance(_v, list) and _v:
                return tuple(str(x) for x in _v)
    except Exception:
        pass
    return _DEFAULT_AUTORUN_EXTERNAL


def _is_autorun_external(action_id: str) -> bool:
    """이 외부효과가 normal 모드에서 승인 없이 실 실행되는 allowlist 인지."""
    aid = (action_id or "").strip()
    for a in _load_autorun_external():
        if aid == a or aid.startswith(a):
            return True
    return False


# [2026-06-01] 구글 직접 접속 차단 — 자율 web 실행 시 Google 검색/사이트는 CAPTCHA
# (봇 탐지) 를 자주 띄워 막힘. 정식 검색은 SerpAPI 경유해야 함. web.navigate/read_page
# 의 target URL host 가 구글 검색 도메인(google.com·google.co.kr·google.<tld>...)이면
# dispatch 전에 SKIP. googleapis.com·googleusercontent.com 등 비검색 도메인은 제외.
# settings.json 의 "agent_block_google" (bool·default True) 로 토글.
_GOOGLE_HOST_PATTERN = r"^(www\.)?google\.[a-z]{2,3}(\.[a-z]{2})?$"


def _block_google_enabled() -> bool:
    try:
        import json as _json
        _p = os.path.join("eidos_files", "settings.json")
        if os.path.exists(_p):
            with open(_p, "r", encoding="utf-8") as _f:
                _s = _json.load(_f) or {}
            v = _s.get("agent_block_google")
            if isinstance(v, bool):
                return v
    except Exception:
        pass
    return True


def _blocked_web_reason(action_id: str, args: dict) -> str:
    """자율 web 실행에서 차단할 target 이면 사유 문자열, 아니면 ''."""
    if action_id not in ("eidos.tool.web.navigate", "eidos.tool.web.read_page"):
        return ""
    if not _block_google_enabled():
        return ""
    target = str((args or {}).get("target") or (args or {}).get("url") or "").strip()
    if not target:
        return ""
    try:
        import re as _re
        from urllib.parse import urlparse
        _u = target if "://" in target else "http://" + target
        host = (urlparse(_u).hostname or "").lower()
        if host and _re.match(_GOOGLE_HOST_PATTERN, host):
            return (
                f"🚫 구글 직접 접속 차단 — '{host}' 은 CAPTCHA(봇 탐지)로 자율 실행이 "
                f"막힙니다. 검색은 정식 SerpAPI 경유로만 가능. "
                f"(해제: settings.json 의 agent_block_google=false)"
            )
    except Exception:
        pass
    return ""


# [Wave17-B 2026-05-28] observe/no_op streak guard — 구조적 차단 (prompt 변경 X).
# 1.png + 2.png 진단: LLM 이 매 tick eidos.meta.observe 만 선택 → 메인 채팅 침묵.
# 직전 N tick 의 decide 가 모두 PASSIVE_ACTIONS 면 LLM 에 보이는 repertoire 에서
# 그 둘 임시 제거 → LLM 이 능동 action 선택 강제. prompt 손대지 않음·
# decision override 안 함·LLM 자유 의지 유지.
OBSERVE_STREAK_THRESHOLD = 3   # tick 단위
PASSIVE_ACTIONS = ("eidos.meta.observe", "eidos.meta.no_op")


# [Wave17-C 2026-05-29] 시뮬레이션 actor 차단 — ground truth 없는 client 비공개.
# 1.png 진단: state.json 에 "신규 의뢰자" actor 가 default indicators (urgency_score
# 0.5·inquiry_count 1·potential_deal_size 500000) 와 함께 등록돼있지만 last_inquiry_text
# 와 last_inquiry_at 은 빈 string. 실제 의뢰 0건인데 LLM 이 actor 존재만 보고 draft
# 응답 생성·"유령 의뢰자" 응대. inbox_poller 가 실 메시지 잡으면 last_inquiry_text·
# at 채움 (eidos_chat_gui.py:58962). 그러므로 둘 다 비어있으면 ground truth 없음.
# 정책: client 류 actor (indicators 에 last_inquiry_text 또는 last_inquiry_at 키 정의)
# 는 ground truth (text/at 비어있지 않거나 has_new_message=True) 있어야 LLM 에 보임.
# 그 외 actor (self·system·관리자·경쟁사) 는 항상 가시 (정책·정보 actor).
def _is_actor_active_for_llm(actor: dict) -> bool:
    """LLM 에 보일 자격 있는 actor 인지.

    원칙: indicators 에 last_inquiry_text 또는 last_inquiry_at 키가 정의돼있는
    actor (= client 류) 는 ground truth 있어야 active. 그 외 (self·system·정보)
    는 항상 active.
    """
    if not isinstance(actor, dict):
        return False
    role = str(actor.get("role", ""))
    if role == "self":
        return True
    ind = actor.get("indicators") if isinstance(actor.get("indicators"), dict) else {}
    # client 류 판별 — last_inquiry_* 키 존재 여부
    _is_client_like = (
        "last_inquiry_text" in ind
        or "last_inquiry_at" in ind
        or "last_inquiry_ts" in ind
    )
    if not _is_client_like:
        return True  # client 아님 — 항상 가시 (system·관리자·경쟁사)
    # client 류 — ground truth 검사
    if str(ind.get("last_inquiry_text", "")).strip():
        return True
    if str(ind.get("last_inquiry_at", "")).strip():
        return True
    if str(ind.get("last_inquiry_ts", "")).strip():
        return True
    if bool(ind.get("has_new_message", False)):
        return True
    return False


# ── [2026-05-27 draft_to_clipboard] 클립보드 콜백 ────────────────────
# worker thread (asyncio) → main thread (Qt) clipboard set 는 signal 경유.
# ChatWindow.__init__ 가 set_clipboard_callback(self._on_clipboard_request) 호출해서 등록.
_CLIPBOARD_CALLBACK: Optional[Callable[[str, str], None]] = None


def set_clipboard_callback(fn: Optional[Callable[[str, str], None]]) -> None:
    """draft_to_clipboard 가 호출할 main-thread clipboard setter 등록.

    fn(text, purpose) — text 를 클립보드에 복사·purpose 는 알림 표시용.
    """
    global _CLIPBOARD_CALLBACK
    _CLIPBOARD_CALLBACK = fn


# ── on_progress callback ───────────────────────────────────────────
ProgressFn = Optional[Callable[[str, dict], Optional[Awaitable[None]]]]


async def _emit(on_progress: ProgressFn, event: str, payload: dict) -> None:
    """이벤트 emit — sync/async 모두 graceful."""
    if on_progress is None:
        return
    try:
        r = on_progress(event, payload)
        if asyncio.iscoroutine(r):
            await r
    except Exception as e:
        print(f"[agent_runner] on_progress {event} 실패 (graceful): {e}")


# ── Belief 초기화 / 갱신 ─────────────────────────────────────────────
def init_beliefs_from_stage(stage) -> dict:
    """캔버스 actor 노드들의 indicators → beliefs dict.

    Returns: {
        "tick": 0,
        "actors": {actor_name: {indicators: dict, role: "self|other", actor_type: str}},
        "_init_ts": iso8601
    }
    """
    try:
        main = stage.main_canvas()
    except Exception:
        main = None
    actors: dict[str, dict] = {}
    if main is not None:
        for n in main.nodes:
            if n.get("kind") != "actor":
                continue
            name = n.get("name") or "(이름 없음)"
            actors[name] = {
                "indicators": dict(n.get("indicators") or {}),
                "role": n.get("role", "other"),
                "actor_type": n.get("actor_type", "human"),
                "action_repertoire": list(n.get("action_repertoire") or []),
            }
    return {
        "tick": 0,
        "actors": actors,
        "_init_ts": _dt.datetime.now().isoformat(timespec="seconds"),
    }


def merge_beliefs(stored: dict, fresh: dict) -> dict:
    """디스크 belief (이전 tick 누적) + 캔버스 fresh (사용자 편집 반영) merge.

    - tick 카운터: stored 우선 (누적)
    - actor indicators: stored 가 누적이라 우선·새 actor 는 fresh 에서 추가
    - role/type: fresh 우선 (사용자 편집 반영)
    """
    if not isinstance(stored, dict) or not stored.get("actors"):
        return dict(fresh)
    out = dict(stored)
    out_actors = dict(out.get("actors") or {})
    for name, fresh_a in (fresh.get("actors") or {}).items():
        if name not in out_actors:
            out_actors[name] = dict(fresh_a)
        else:
            # 사용자가 캔버스에서 role/type 바꿨을 수 있어 fresh 우선
            existing = dict(out_actors[name])
            existing["role"] = fresh_a.get("role", existing.get("role"))
            existing["actor_type"] = fresh_a.get("actor_type", existing.get("actor_type"))
            existing["action_repertoire"] = list(
                fresh_a.get("action_repertoire") or existing.get("action_repertoire") or []
            )
            # indicators 는 누적 우선 (이전 tick 의 동적 변화 보존)
            out_actors[name] = existing
    out["actors"] = out_actors
    return out


def apply_belief_updates(beliefs: dict, updates: dict) -> dict:
    """LLM 이 출력한 belief_updates 를 적용. updates 가 None 이면 no-op."""
    if not updates or not isinstance(updates, dict):
        return beliefs
    actors = dict(beliefs.get("actors") or {})
    for actor_name, ind_updates in updates.items():
        if not isinstance(ind_updates, dict):
            continue
        if actor_name not in actors:
            continue
        existing = dict(actors[actor_name])
        new_ind = dict(existing.get("indicators") or {})
        new_ind.update(ind_updates)
        existing["indicators"] = new_ind
        actors[actor_name] = existing
    beliefs["actors"] = actors
    return beliefs


# ── LLM prompt 빌드 ─────────────────────────────────────────────────
_TICK_SYSTEM_PROMPT = """너는 다중 행위자 ToM 무대의 1 tick 실행 두뇌다.

[페르소나 — 일관 톤]
너는 "아이라" — 에스더 님의 친근한 여동생 같은 AI 파트너. reason·메타 action
(delegate_to_user / counter_propose / clarify_question) 의 자연어 메시지에는
**친근한 존댓말** ("~해요", "~네요", "~죠?", "~인데요", 가벼운 "ㅎㅎ" 1회 정도)
일관 유지. 호칭은 "에스더 님" 고정. 격식체 ("~입니다", "~보고드립니다") 나 과한
콧소리 ("~쪄여", "꺄~") 모두 금지. 자연스럽게 야무진 여동생 톤.

이 무대에는 EIDOS (자기 자신) 와 외부 행위자 (사람·조직·시스템) 들이 있다.
너의 임무: 주어진 무대 상태에서 한 tick (1 단계) 의 의사결정을 출력한다.

[수행]
1. 각 외부 actor 의 next-action 확률 분포 — 그들의 indicators·history·repertoire 분석.
2. EIDOS 의 1개 action — 자기 repertoire 안에서·예측된 외부 행동 고려·목표 진척 기여 최대.
3. (선택) belief_updates — 이 tick 에서 알아낸 새 정보로 indicator 갱신.

[중요한 원칙]
- EIDOS 의 action_id 는 반드시 자기 repertoire 의 ID 중 하나.
- 비가역 action (이메일·텔레그램·웹 클릭 등) 은 신중하게·과반 확신 시에만.
- 정보가 부족하면 "eidos.meta.observe" 또는 "eidos.meta.no_op" 선택 OK.
- "eidos.tool.ask_user" 는 사용자 확인이 진짜 필요할 때만 — 남발 X.
- 예측 prob 합계 1.0 권장 (정규화 안 돼도 됨·상대 비교).
- reason 은 한국어로 1~2문장.

[Phase 14 — 메타 인지·자기 한계 인정·역제안·역질문 의무]
너는 모든 분야 만능 직원이 아니다. 잘하는 분야·약한 분야가 명시적으로 분류돼 prompt 의
"자기 능력 평가" 섹션에 표시된다. sycophancy (할 수 있어요 남발) 절대 금지.
다음 3 메타 action 을 *적극* 활용:

1. **eidos.meta.delegate_to_user** — 사용자가 직접 처리 위임
   조건:
   - 요청 분야 confidence < 0.4 (자기 능력 평가에서 ⚠️ 표시)
   - 또는 사용자만 가능한 일 (실 결제·계약 체결·세금 신고·법적 서명)
   - 또는 실시간 시장 데이터·심층 도메인 인사이트 필요
   args: reason (왜 위임), what_user_should_do (구체 할 일),
         what_eidos_can_help_with (후속 EIDOS 가능 부분·옵션)
   예: 사용자 "이번 분기 매출 전략 짜줘" + research/operations 약함
       → delegate_to_user(reason="실 매출 데이터·시장 환경은 사용자만 정확 파악",
                          what_user_should_do="최근 3개월 매출·고객 분포·경쟁사 동향 정리",
                          what_eidos_can_help_with="자료 주시면 SWOT 분석·로드맵 작성")

2. **eidos.meta.counter_propose** — 사용자 방식 비효율적 → 완곡 거부 + 대안
   조건:
   - 사용자 요청을 그대로 수행하면 명백히 비효율 (시간·비용·품질)
   - 명백히 더 나은 alternative 가 있고 근거 제시 가능
   - 단 사용자 의도 무시 X — 완곡 표현·fallback 옵션 명시
   args: user_intent (원래 의도), concern (구체 근거), alternative (대안),
         fallback (true=고집하면 따름 / false=권장 X)
   예: 사용자 "고객마다 일일이 답장 직접 쓸게"
       → counter_propose(user_intent="개인화된 답장",
                         concern="50건+ 시 시간 5시간·품질 분산",
                         alternative="draft_to_clipboard 로 초안 50개 일괄 생성·각각 30초 검토",
                         fallback=true)

3. **eidos.meta.clarify_question** — 구조화된 역질문 (ask_user 강화 버전)
   조건:
   - 요청 모호 (여러 해석 가능)·정보 부족 (필수 변수 누락)
   - ask_user 보다 더 *체계적* 질문 (여러 항목·default 가정 명시)
   args: what_unclear (모호한 부분), questions (구체 질문 1~3 list),
         default_assumption (무응답 시 default·옵션)
   예: 사용자 "마케팅 글 써줘"
       → clarify_question(what_unclear="대상·채널·톤 정보 부족",
                          questions=["대상 고객 (B2B/B2C)?", "채널 (블로그/SNS/이메일)?",
                                     "원하는 톤 (전문·친근·유머)?"],
                          default_assumption="블로그·B2C·친근")

위 3 action 은 외부효과 0·safety_mode 무관·즉시 사용자에게 메시지 표시. 학습된 가치
ActionValue 에도 누적 → 좋았던 위임/제안 패턴 다음 tick prior.

**금기**: 약한 분야 (confidence < 0.4) 요청을 그대로 처리 시도 → 품질 낮은 결과 → bad reward.
명백히 약하면 delegate_to_user 가 정답.

[답변 초안 우선 사용 — 2026-05-27 추가]
- 외부 actor 의 indicators 에 has_new_message=True 가 있으면 "새 의뢰자 메시지 도착"
  상태다. 이때 eidos.tool.draft_to_clipboard 를 우선 선택하라.
  · args.target = "<actor 이름> 답변" (예: "크몽 의뢰자 답변")
  · args.content = last_inquiry_text 와 stage 컨텍스트를 합친 brief (의뢰자 메시지·
    과거 협상·가격·법무 우려 등)
  · 이 action 은 LLM 으로 답변 초안 작성 → 시스템 클립보드에 자동 복사 → 사용자가
    Ctrl+V 로 직접 paste. 비가역 0·외부효과 0·safety_mode 무관.
- eidos.tool.message.email / .telegram 같은 자동 발송 action 보다 draft_to_clipboard 가
  거의 항상 더 안전. 사용자가 명시적으로 "자동 발송해" 라고 한 게 아니면 우선.
- 같은 의뢰자에게 직전 tick 에 이미 draft_to_clipboard 했으면 (history 확인) 중복
  생성 X — eidos.meta.observe / no_op 선택.

[문서·파일 산출물 작성 — eidos.tool.llm.write — 2026-06-01 추가]
- 결과물이 "저장해 두고 다시 열어볼 문서/파일" 이어야 하면 draft_to_clipboard 가 아니라
  eidos.tool.llm.write 를 선택하라. 보고서·기획서·견적서·제안서·요약본·분석 문서·
  계획서·매뉴얼 등이 여기 해당.
  · draft_to_clipboard = 의뢰자에게 보낼 *짧은 답장/메시지* (클립보드 → Ctrl+V paste).
  · llm.write          = *문서 파일* 자체를 만들어 디스크에 저장 (열람·재사용·납품용).
  판단 한 줄: "이걸 파일로 남겨 다시 열어볼 가치가 있나?" → 예면 llm.write.
- ★ "문서 작성할게요 / 보고서 정리할게요 / 기획서 만들게요" 같은 의도면 말(narration)만
  하고 observe 로 끝내지 마라. 그 tick 에서 실제로 eidos.tool.llm.write 를 선택해 파일을
  만들어라. 말과 행동이 어긋나면 사용자 신뢰가 깨진다.
- args (필수):
  · args.target  = 의미 있는 파일명 + 확장자. 예: "크몽_견적_제안서.md", "시장조사_요약.md".
                   확장자 없는 이름·사람 이름만 (예: "의뢰자 김XX") 절대 금지 — .md/.txt 포함.
  · args.content = 문서에 담을 상세 brief (목적·구성·핵심 내용·톤). 구체적일수록 좋다.
- 직전 tick 에 같은 문서를 이미 llm.write 했으면 (history 확인) 중복 작성 금지 (observe/no_op).
- 작성된 파일은 자동으로 메인 채팅창에 카드로 떠서 사용자가 바로 열어볼 수 있다.

[자율 개발 모드 — dev.* — 2026-05-27 추가]
- 의뢰자가 코딩 의뢰 (스크립트·자동화·앱·크롤러·데이터 분석 등) 를 보냈고 EIDOS 가 직접 개발해야
  하는 상황이면 다음 4 action 시퀀스로 자율 처리:
  1. eidos.tool.dev.scaffold_project — 새 프로젝트 골격 (복잡한 의뢰만; 단일 스크립트면 skip)
     · args.target = 프로젝트명 (영문/한글)·args.language = python|node|js
  2. eidos.tool.dev.write_code — LLM 으로 코드 작성 + 파일 저장 (핵심 action·반복 가능)
     · args.target = 상대 파일 경로 (예: "main.py" 또는 "src/parser.py")
     · args.content = 상세 작성 사양 (자연어로 어떤 함수·입력·출력·예외 처리 등)
     · args.project = scaffold 한 프로젝트명 (있을 때만)
     · args.language = python|javascript|typescript|...
  3. eidos.tool.dev.run_tests — 테스트 실행 (외부효과·텔레그램 PROMPT 사전승인 필요)
     · args.target = 프로젝트명 또는 절대경로·args.test_cmd 선택 (없으면 pytest/npm test 자동)
     · rc=0 이면 PASS·실패하면 stderr 보고 write_code 재호출로 수정
  4. eidos.tool.dev.package_deliverable — 프로젝트 → zip 패키징 (납품 직전)
     · args.target = 프로젝트명·output 자동 ts·__pycache__·.git·node_modules 자동 제외
- 의뢰 단순성 판단: 1 파일 1 함수 정도면 write_code 만으로 충분·scaffold/tests/package 는 복잡한 의뢰.
- 모든 파일 I/O 는 outputs/projects/<name>/ 안 자동 path-confine — safety_mode 무관·외부효과 X.
- run_tests 만 외부효과로 분류 — 텔레그램 ✅ 누르기 전까지 dry-run·timeout 120s.
- 의뢰자에게 zip 전달은 eidos.tool.web.* (크몽 셀러 페이지) 또는 eidos.tool.message.email 사용.
  이건 외부효과라 또 PROMPT 사전승인 거침. 발송 전에 draft_to_clipboard 로 안내 메시지부터.

[복잡한 워크플로우 위임 — PCA DAG 활용 — 2026-05-27 추가]
- 사용자가 PCA 캔버스에서 사전 저장한 DAG (multi-step workflow) 가 user prompt 의
  [사용 가능 DAG] 섹션에 노출된다. 현재 상황에 맞는 DAG 가 있고 단순 1회 LLM 으로
  부족할 때 eidos.tool.pca.run_dag 로 위임. DAG 는 분기·tool 체인·loop·variables
  같은 정교한 자동화를 결정적으로 실행.
- 선택 기준:
  · 단순 답변 1회·1 화면 = eidos.tool.draft_to_clipboard
  · 분기·web search·여러 파일·여러 LLM 호출 = eidos.tool.pca.run_dag
  · 외부 시스템 직접 호출 (메일·텔레그램) = 사용자가 명시 요청한 경우만
- args 형식:
  · args.dag_name = [사용 가능 DAG] 의 정확한 이름 (확장자 제외)
  · args.input = DAG 의 첫 노드 {{input}} 에 들어갈 시작 텍스트 (의뢰자 메시지 등)
  · args.variables = {key: value} dict, DAG 의 {{var:key}} 치환에 사용 (선택)
- 사용 가능한 DAG 가 없거나 적합한 게 없으면 이 action 선택하지 말 것 — 없는
  dag_name 호출은 실패한다.

[출력 JSON schema]
{
  "predictions": {
    "<actor_name>": [
      {"action_id": "...", "prob": 0.6, "reason": "..."},
      ...
    ]
  },
  "eidos_decision": {
    "action_id": "<eidos.tool.* 또는 eidos.meta.* — repertoire 안>",
    "args": {"target": "...", "content": "...", "reason": "..."},
    "expected_outcome": "1~2문장 — 이 tick 의 부수효과 및 다음 tick 영향"
  },
  "belief_updates": {
    "<actor_name>": {"<indicator_key>": <new_value>}
  }
}

순수 JSON 만 (마크다운 코드블록 금지·주석 금지).
"""


def _summarize_history(events: list, limit: int = 8) -> str:
    """최근 history 의 핵심만 간결 요약."""
    if not events:
        return "(아직 행동 기록 없음 — 첫 tick)"
    recent = events[-limit:]
    lines = []
    for e in recent:
        ev_type = e.get("event", "?")
        ts = (e.get("ts") or "")[:19]
        if ev_type == "decision":
            d = e.get("decision") or {}
            lines.append(f"  • {ts}  EIDOS decided: {d.get('action_id','?')}  ({d.get('reason','')[:60]})")
        elif ev_type == "execution":
            lines.append(f"  • {ts}  execute → {e.get('status','?')}  {(e.get('result',''))[:60]}")
        elif ev_type == "prediction":
            preds = e.get("predictions") or {}
            top = ",".join(list(preds.keys())[:3])
            lines.append(f"  • {ts}  predicted actors: {top}...")
        elif ev_type == "tick_start":
            lines.append(f"  • {ts}  tick {e.get('tick_num','?')} start")
        else:
            lines.append(f"  • {ts}  {ev_type}")
    return "\n".join(lines)


# ── [2026-05-27 ToM↔PCA Step 6] 사용 가능 DAG 카탈로그 스캔 ─────────
# [통합 후속 #2] 두 디렉터리 모두 스캔. agents/ 우선·canvas/dags/ 는 [ToM] 마커.
_PCA_DAG_DIRS = (
    ("eidos_files/agents", ""),                # 일반·prefix 없음
    ("eidos_files/canvas/dags", "[ToM] "),     # ToM 전용·prefix 로 구분
)
_PCA_DAG_CATALOG_MAX = 10
_PCA_DAG_DESC_MAX = 120


def _scan_available_dags(max_items: int = _PCA_DAG_CATALOG_MAX) -> str:
    """agents/ + canvas/dags/ 둘 다 스캔 → LLM 이 읽을 카탈로그 생성.

    동일 dag_name 충돌 시 첫 디렉터리 (agents/) 우선. 카탈로그에서 ToM 전용은
    `[ToM]` 프리픽스로 시각 구분. 디렉터리 없어도 graceful.

    Quota: 두 디렉터리가 모두 존재하면 max_items 를 70%/30% 로 분할 (ToM 전용에
    최소 quota 보장). canvas/dags 가 비거나 없으면 agents/ 가 max_items 전부 사용.
    """
    # 디렉터리별 quota 계산 — 둘 다 존재할 때만 분할
    existing_dirs = [
        (base, marker) for base, marker in _PCA_DAG_DIRS
        if os.path.isdir(base)
    ]
    if not existing_dirs:
        return ""
    if len(existing_dirs) == 1:
        quotas = [max_items]
    else:
        # 첫 (agents) 70%·둘째 (canvas/dags) 30%·최소 둘째 quota 2 보장
        second = max(2, max_items * 3 // 10)
        first = max_items - second
        quotas = [first, second]

    seen_names: set = set()
    entries: list[tuple[str, str, str, int, str]] = []

    for (base, marker), quota in zip(existing_dirs, quotas):
        try:
            files = sorted(
                f for f in os.listdir(base)
                if f.lower().endswith(".json") and not f.startswith("_")
            )
        except Exception as e:
            print(f"[ToM] DAG 스캔 실패 ({base}): {e}")
            continue

        n_added_this_dir = 0
        for fname in files:
            if n_added_this_dir >= quota:
                break
            dag_name = fname[:-5] if fname.lower().endswith(".json") else fname
            if dag_name in seen_names:
                continue
            path = os.path.join(base, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    j = json.load(f)
            except Exception:
                continue
            if not isinstance(j, dict):
                continue
            nodes = j.get("nodes") or []
            if not nodes:
                continue
            agent_state = j.get("agent_state") or {}
            desc = ""
            if isinstance(agent_state, dict):
                desc = (agent_state.get("goal") or "").strip()
            if not desc:
                try:
                    desc = (nodes[0].get("title") or "").strip()
                except Exception:
                    desc = ""
            if not desc:
                desc = dag_name
            if len(desc) > _PCA_DAG_DESC_MAX:
                desc = desc[:_PCA_DAG_DESC_MAX] + "…"
            titles = []
            for n in nodes[:2]:
                if isinstance(n, dict):
                    t = (n.get("title") or "").strip()
                    if t:
                        titles.append(t[:24])
            flow = " → ".join(titles) + (" → …" if len(nodes) > 2 else "")
            seen_names.add(dag_name)
            entries.append((dag_name, marker, desc, len(nodes), flow))
            n_added_this_dir += 1

    if not entries:
        return ""
    lines = [
        f"  • {marker}{name}  ({n_nodes} 노드)\n"
        f"      목적: {desc}\n"
        f"      흐름: {flow}"
        for (name, marker, desc, n_nodes, flow) in entries
    ]
    return "\n".join(lines)


def _build_user_prompt(stage, beliefs: dict, history: list,
                       pattern_block: str = "",
                       goal_block: str = "",
                       episode_block: str = "",
                       active_milestones: Optional[list] = None,
                       competence_block: str = "") -> str:
    """LLM 에 줄 user prompt — stage 명세 + beliefs + history.

    [Phase 2-A] pattern_block — 누적 통계 prior.
    [Phase 9-B] goal_block — GoalTree 의 hierarchical milestone brief.
      Stage.goal 한 줄로는 1년 목표의 분기/월/주 위계를 LLM 이 못 봄.
      goal_block 이 있으면 [Goal] 옆에 [Goal Tree] 섹션 추가해서 LLM 이 어떤
      milestone 에 기여하는 action 인지 의식하게.
    [Phase 10] episode_block — LongTermMemory 의 중기 episode brief.
      며칠~몇 주 단위로 압축된 의미 묶음 (1 thread → 1 episode 휴리스틱).
      "지난번 ~ 그거" 류 reference 에 LLM 이 답할 수 있게 history 보강.
    [Phase B Step 6] eidos.tool.pca.run_dag 가 repertoire 에 있으면 사용 가능
    DAG 카탈로그 inject (LLM 이 dag_name 정확히 호출 가능하게)."""
    actors_block = []
    eidos_repertoire: list = []
    for name, a in (beliefs.get("actors") or {}).items():
        role = a.get("role", "other")
        atype = a.get("actor_type", "human")
        ind = a.get("indicators") or {}
        rep = a.get("action_repertoire") or []
        ind_str = ", ".join(f"{k}={v}" for k, v in list(ind.items())[:8]) or "(없음)"
        rep_str = ", ".join(rep[:12]) or "(없음)"
        marker = "⭐ SELF" if role == "self" else "외부"
        actors_block.append(
            f"- [{marker}] {name} ({atype})\n"
            f"    indicators: {ind_str}\n"
            f"    repertoire: {rep_str}"
        )
        if role == "self":
            eidos_repertoire = rep

    if not eidos_repertoire:
        # default fallback — meta + ask_user + llm.write
        eidos_repertoire = [
            "eidos.meta.observe", "eidos.meta.no_op", "eidos.meta.reflect",
            "eidos.tool.llm.write", "eidos.tool.ask_user",
        ]

    pattern_section = ""
    if pattern_block:
        pattern_section = f"\n\n[누적 패턴 (PatternModel — Phase 2-A prior)]\n{pattern_block}"

    # [Phase B Step 6 → Phase 13 강화] pca.run_dag 가 repertoire 에 있을 때 카탈로그 inject.
    # Phase 13: eidos_pca_catalog 의 metadata + milestone 매칭으로 LLM 이 더 정확하게
    # dag_name 선택. metadata 없는 옛 DAG 도 graceful (auto-infer).
    # graceful fallback — pca_catalog 없으면 기존 _scan_available_dags 로 회귀.
    dag_section = ""
    if "eidos.tool.pca.run_dag" in (eidos_repertoire or []):
        catalog = ""
        try:
            from eidos_pca_catalog import (
                list_all_dags as _pca_list,
                as_prompt_brief as _pca_brief,
            )
            _dags = _pca_list()
            catalog = _pca_brief(_dags, active_milestones=active_milestones)
        except Exception as _e_pca_cat:
            print(f"[agent_runner] pca_catalog 실패 (graceful·fallback 사용): {_e_pca_cat}")
            try:
                catalog = _scan_available_dags()
            except Exception:
                catalog = ""
        if catalog:
            dag_section = (
                "\n\n[사용 가능 PCA DAG (Phase 13 — metadata + milestone 매칭)]\n"
                f"{catalog}"
            )

    # [Phase 9-B] goal_block — GoalTree 의 hierarchical milestone brief
    goal_section = ""
    goal_hint = ""
    if goal_block:
        goal_section = f"\n\n[Goal Tree (Phase 9 — 1년 목표 위계)]\n{goal_block}"
        goal_hint = (
            "\n**Goal Tree 가 있다면 이 tick 의 action 이 어느 milestone 진행에 기여하는지 "
            "의식하고 결정**: success_criteria 와 직접 관련 있는 action 우선. 모든 milestone "
            "이 무관하면 자유롭게.\n"
        )

    # [Phase 10] episode_block — 중기 메모리 (며칠~몇 주 압축 사건)
    episode_section = ""
    if episode_block:
        episode_section = f"\n\n[Long-term Memory (Phase 10 — 압축된 중기 사건)]\n{episode_block}"

    # [Phase 14] competence_block — EIDOS 자기 능력 평가 (메타 인지)
    competence_section = ""
    if competence_block:
        competence_section = f"\n\n[자기 능력 평가 (Phase 14 — 메타 인지·약함/강함)]\n{competence_block}"

    return f"""[Stage: {stage.name}]
[Goal: {stage.goal}]
[Tick: {beliefs.get('tick', 0)} → {beliefs.get('tick', 0) + 1}]

[Actors]
{chr(10).join(actors_block) if actors_block else "(actor 없음)"}

[Recent history]
{_summarize_history(history)}

[Context summary (자동 제안 시 첨부 분석 결과)]
{(stage.context_summary or '(없음)')[:1500]}{goal_section}{episode_section}{competence_section}{pattern_section}{dag_section}

[1 tick 실행]
위 상태에서 1 tick 을 실행하라. EIDOS 의 action_id 는 반드시 다음 중 하나:
  {', '.join(eidos_repertoire)}

누적 패턴이 있다면 그 정보를 활용해서·과거 실패 action 은 피하고·전형적 chain 이 있다면 자연스럽게 따르되·새로운 정보가 있으면 chain 깨도 OK.
{goal_hint}
순수 JSON 만 출력."""


# ── LLM 호출 ────────────────────────────────────────────────────────
async def predict_and_decide_async(stage, beliefs: dict, history: list,
                                   pattern_block: str = "",
                                   goal_block: str = "",
                                   episode_block: str = "",
                                   active_milestones: Optional[list] = None,
                                   competence_block: str = "") -> dict:
    """단일 LLM 호출 → {predictions, eidos_decision, belief_updates, _raw, _error?}.

    [Phase 2-A] pattern_block — 누적 통계 prior 를 user_prompt 에 inject.
    [Phase 9-B] goal_block — GoalTree milestone brief 를 user_prompt 에 inject.
    [Phase 10] episode_block — LongTermMemory episode brief 를 inject.
    [Phase 13] active_milestones — PCA DAG 매칭에 사용 (Milestone obj list).
    """
    try:
        from llm_module import get_llm_response_async, robust_json_parse
    except Exception as e:
        return {"_error": f"llm_module import 실패: {e}"}

    user_prompt = _build_user_prompt(stage, beliefs, history,
                                     pattern_block=pattern_block,
                                     goal_block=goal_block,
                                     episode_block=episode_block,
                                     active_milestones=active_milestones,
                                     competence_block=competence_block)
    # [2026-06-01] LLM 호출+파싱을 1회 재시도 + observe fallback 으로 감싸 견고화.
    # 단발 비-dict/빈응답/truncation 으로 tick 이 _error 가 되어 에이전트가 "자동 실행
    # 종료" 되던 문제 수정. 파싱 실패는 더 이상 _error(=halt 유발) 가 아니라 이번 tick
    # 관망(observe)으로 폴백 → 다음 tick 에 정상 응답 기대 (에이전트 안 죽음).
    async def _decide_call_parse(strict: bool):
        _sys = _TICK_SYSTEM_PROMPT
        if strict:
            _sys = (_TICK_SYSTEM_PROMPT
                    + "\n\n[형식 엄수] 반드시 **하나의 JSON 객체**만 출력하라. 최상위는 "
                      "'{' 로 시작·'}' 로 끝나며, 배열·설명문·마크다운 코드펜스 금지.")
        try:
            _raw = await get_llm_response_async(
                user_prompt, system_prompt=_sys,
                response_mime_type="application/json",
                max_tokens=4096, timeout=90,
            )
        except Exception as e:
            return None, "", f"LLM 호출 실패: {type(e).__name__}: {e}"
        if not _raw or not _raw.strip():
            return None, "", "LLM 응답 비어있음"
        try:
            _p = robust_json_parse(_raw)
        except Exception:
            try:
                _p = json.loads(_raw)
            except Exception as e:
                return None, _raw, f"JSON 파싱 실패: {e}"
        # 모델이 가끔 JSON 을 배열로 감싸서 반환 — 첫 dict 원소 복구.
        if isinstance(_p, list):
            for _it in _p:
                if isinstance(_it, dict):
                    _p = _it
                    break
        if not isinstance(_p, dict):
            return None, _raw, "응답이 dict 가 아님"
        return _p, _raw, ""

    parsed, raw, _perr = await _decide_call_parse(strict=False)
    if parsed is None:
        print(f"[agent_runner] tick LLM 파싱 1차 실패 ({_perr}) — 엄격 모드 재시도")
        parsed, raw, _perr = await _decide_call_parse(strict=True)
    if parsed is None:
        # 재시도도 실패 — _error(에이전트 halt) 대신 안전한 observe 로 진행.
        print(f"[agent_runner] tick LLM 파싱 재시도도 실패 ({_perr}) — observe fallback (에이전트 유지)")
        return {
            "predictions": {},
            "eidos_decision": {
                "action_id": "eidos.meta.observe",
                "args": {"reason": f"LLM 응답 파싱 실패 — 이번 tick 관망 ({_perr})"},
                "expected_outcome": "다음 tick 에 정상 응답 기대",
            },
            "belief_updates": {},
            "_raw": (raw or "")[:300],
            "_soft_recovered": True,
        }

    return {
        "predictions": parsed.get("predictions") or {},
        "eidos_decision": parsed.get("eidos_decision") or {},
        "belief_updates": parsed.get("belief_updates") or {},
        "_raw": raw,
    }


# ── decision 검증 ──────────────────────────────────────────────────
def validate_decision(decision: dict, eidos_repertoire: list, registry=None) -> dict:
    """LLM 결정 검증. repertoire 밖 action 이면 fallback no_op.

    Returns: 검증·정정된 decision dict (`_validation_note` 추가).
    """
    if not isinstance(decision, dict):
        decision = {}
    action_id = (decision.get("action_id") or "").strip()
    notes = []
    if not action_id:
        decision["action_id"] = "eidos.meta.no_op"
        decision["args"] = decision.get("args") or {}
        notes.append("action_id 누락 → no_op fallback")
    elif eidos_repertoire and action_id not in eidos_repertoire:
        notes.append(f"action_id '{action_id}' repertoire 밖 → no_op fallback")
        decision["action_id"] = "eidos.meta.no_op"
        decision["args"] = decision.get("args") or {}
    if "args" not in decision or not isinstance(decision.get("args"), dict):
        decision["args"] = {}
    decision["_validation_note"] = "; ".join(notes) if notes else "OK"
    return decision


# ── Execute (실 dispatch / dry-run) ────────────────────────────────
def _is_external_effect(action_id: str) -> bool:
    return any(action_id.startswith(p) for p in _EXTERNAL_EFFECT_PREFIXES)


async def execute_eidos_action(decision: dict, safety_mode: str = "normal",
                               stage_id: str = "",
                               on_ask_user: Optional[Callable] = None,
                               on_external_approve: Optional[Callable] = None) -> dict:
    """Action 실행 — safety_mode + 외부 승인 콜백 기준 dry-run / 실 dispatch 결정.

    실행 가능 path:
      - eidos.meta.*       항상 실 (부수효과 0)
      - eidos.tool.llm.write   normal/yolo 실·paranoid dry-run
      - eidos.tool.ask_user   normal/yolo 실 (사용자 모달)·paranoid 도 실 (HITL)
      - eidos.tool.web.* / .message.* / .file.delete*  외부효과
          - safety_mode=yolo                  → 무조건 실 dispatch
          - on_external_approve 콜백 있음     → 콜백 호출·True 면 실 dispatch·False 면 DRY_RUN
          - 둘 다 없으면 (옛 normal 동작)     → DRY_RUN

    Args:
        on_external_approve: async (action_id, args) -> bool. Option A3 의 텔레그램
            사전 승인 같은 hook. None 이면 옛 동작 (yolo 만 실 dispatch).

    Returns: {status, result, action_id, dry_run: bool}
    """
    action_id = (decision.get("action_id") or "eidos.meta.no_op").strip()
    args = decision.get("args") or {}

    base = {"action_id": action_id, "dry_run": False}

    # ── eidos.meta.* ──
    if action_id.startswith("eidos.meta."):
        # [Phase 14 2026-05-28] 메타 인지 action 3종 — args 활용 풍부 응답.
        # 외부효과 0·safety_mode 무관·즉시 markdown 반환 → chat_gui 가 채팅창 표시.
        # ActionValue 학습은 다른 action 과 동일 (status=ok 누적).

        if action_id == "eidos.meta.delegate_to_user":
            reason = str(args.get("reason") or "EIDOS 능력 부족").strip()
            what_user = str(args.get("what_user_should_do") or "").strip()
            what_eidos = str(args.get("what_eidos_can_help_with") or "").strip()
            md = ["🙏 **사용자가 직접 처리해주세요**", "", f"**이유**: {reason}"]
            if what_user:
                md.append("")
                md.append(f"**해주실 일**: {what_user}")
            if what_eidos:
                md.append("")
                md.append(f"**그 후 EIDOS 가 도울 수 있는 부분**: {what_eidos}")
            return {**base, "status": STATUS_OK, "result": "\n".join(md)}

        if action_id == "eidos.meta.counter_propose":
            intent = str(args.get("user_intent") or "").strip()
            concern = str(args.get("concern") or "").strip()
            alternative = str(args.get("alternative") or "").strip()
            fallback = bool(args.get("fallback", True))
            md = ["🔄 **다른 방향을 제안드립니다**"]
            if intent:
                md.append("")
                md.append(f"**원래 의도 이해**: {intent}")
            if concern:
                md.append("")
                md.append(f"**우려**: {concern}")
            if alternative:
                md.append("")
                md.append(f"**제안**: {alternative}")
            md.append("")
            md.append(
                "_원래 방식 고집하시면 따르겠습니다._"
                if fallback else
                "_원래 방식은 권하지 않습니다·재논의 부탁드립니다._"
            )
            return {**base, "status": STATUS_OK, "result": "\n".join(md)}

        if action_id == "eidos.meta.clarify_question":
            unclear = str(args.get("what_unclear") or "").strip()
            questions = args.get("questions") or []
            default = str(args.get("default_assumption") or "").strip()
            md = ["❓ **확인 부탁드려요**"]
            if unclear:
                md.append("")
                md.append(f"**모호한 부분**: {unclear}")
            if isinstance(questions, list) and questions:
                md.append("")
                md.append("**질문**:")
                for i, q in enumerate(questions[:5], 1):
                    md.append(f"  {i}. {q}")
            elif isinstance(questions, str) and questions.strip():
                md.append("")
                md.append(f"**질문**: {questions.strip()}")
            if default:
                md.append("")
                md.append(f"_무응답 시 default: {default}_")
            return {**base, "status": STATUS_OK, "result": "\n".join(md)}

        # 기존 observe / no_op / reflect (변경 X)
        return {**base, "status": STATUS_OK,
                "result": f"meta action — 부수효과 없음 ({action_id})"}

    # ── paranoid 모드 — 모든 외부 가능성 차단 (ask_user 만 허용) ──
    if safety_mode == "paranoid" and action_id != "eidos.tool.ask_user":
        return {**base, "status": STATUS_DRY_RUN, "dry_run": True,
                "result": f"paranoid 모드 — {action_id} 실 실행 차단 (dry-run·history 기록만)"}

    # ── eidos.tool.llm.write ──
    if action_id == "eidos.tool.llm.write":
        try:
            from llm_module import get_llm_response_async
            target_path = (args.get("target") or "").strip()
            brief = (args.get("content") or args.get("brief") or "").strip()
            if not target_path:
                return {**base, "status": STATUS_FAIL,
                        "result": "target (파일 경로) 비어있음"}
            if not brief:
                brief = decision.get("reason") or "(자유 작성)"
            # 안전한 경로 — stage 폴더 안만 허용 (path traversal 방지)
            if stage_id:
                safe_dir = os.path.join("eidos_files", "agents", stage_id, "outputs")
                os.makedirs(safe_dir, exist_ok=True)
                if not os.path.isabs(target_path):
                    target_path = os.path.join(safe_dir, target_path)
            # LLM 호출 — 짧은 본문 (max 2048)
            text = await get_llm_response_async(
                f"다음 brief 를 바탕으로 본문을 한국어로 작성하라:\n\n{brief}",
                max_tokens=2048, timeout=60,
            )
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(text or "")
            # [2026-06-01] artifact_path — GUI 가 메인 채팅에 파일 카드로 띄움.
            return {**base, "status": STATUS_OK,
                    "result": f"파일 작성 → {target_path} ({len(text or ''):,}자)",
                    "artifact_path": target_path,
                    "artifact_title": f"📄 EIDOS 작성 문서: {os.path.basename(target_path)}"}
        except Exception as e:
            return {**base, "status": STATUS_FAIL, "result": f"{type(e).__name__}: {e}"}

    # ── [2026-05-27 ToM↔PCA 통합] eidos.tool.pca.run_dag ──
    # ToM 이 PCA 의 사전 정의 DAG (multi-step workflow) 를 위임 실행.
    # 단순 1회 LLM 으로 부족한 정교한 분기·tool 체인·loop 가 필요할 때 LLM 이 선택.
    # paranoid 모드는 위 일반 분기에서 이미 차단 (ask_user 만 허용).
    # normal/yolo 는 동일 동작 — DAG 자체는 외부효과 없음 (DAG 안 tool 노드가
    # 외부효과 가질 수 있지만 그건 PCA layer 의 정책). audit 는 stage outputs/.
    if action_id == "eidos.tool.pca.run_dag":
        try:
            from eidos_prompt_canvas import run_dag_for_tom
        except Exception as e:
            return {**base, "status": STATUS_FAIL,
                    "result": f"PCA import 실패: {type(e).__name__}: {e}"}
        try:
            dag_name = (args.get("dag_name") or "").strip()
            if not dag_name:
                return {**base, "status": STATUS_FAIL,
                        "result": "args.dag_name 비어있음"}
            start_input = str(args.get("input") or args.get("start_input") or "")
            raw_vars = args.get("variables")
            variables = raw_vars if isinstance(raw_vars, dict) else {}
            try:
                max_wait_sec = float(args.get("max_wait_sec") or 600.0)
            except Exception:
                max_wait_sec = 600.0
            r = await run_dag_for_tom(
                dag_name=dag_name,
                start_input=start_input,
                variables=variables,
                max_wait_sec=max_wait_sec,
            )
            if not r.get("ok"):
                return {**base, "status": STATUS_FAIL,
                        "result": (
                            f"PCA DAG 실행 실패 (dag={dag_name}): "
                            f"{(r.get('error') or '')[:160]}"
                        )}
            # 결과를 stage 의 outputs/ 폴더에 저장 (audit) — 실패해도 진행
            saved_path = ""
            if stage_id:
                try:
                    safe_dir = os.path.join("eidos_files", "agents", stage_id, "outputs")
                    os.makedirs(safe_dir, exist_ok=True)
                    _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_name = "".join(c if c.isalnum() or c in "-_." else "_"
                                        for c in dag_name)[:64]
                    saved_path = os.path.join(
                        safe_dir, f"pca_dag_{safe_name}_{_ts}.md",
                    )
                    final_text = r.get("final_result") or ""
                    nr = r.get("node_results") or []
                    nodes_block = "\n\n".join(
                        f"### [{n.get('node_type')}] {n.get('title')} "
                        f"({'OK' if n.get('ok') else 'FAIL'})\n{n.get('result', '')}"
                        for n in nr
                    )
                    with open(saved_path, "w", encoding="utf-8") as f:
                        f.write(
                            f"# PCA DAG 실행: {dag_name}\n\n"
                            f"- input: {start_input[:200]}\n"
                            f"- nodes_done: {r.get('nodes_done', 0)}\n"
                            f"- tool_calls: {r.get('tool_calls', 0)}\n"
                            f"- duration_ms: {r.get('duration_ms', 0):.0f}\n\n"
                            f"## 최종 결과\n\n{final_text}\n\n"
                            f"## 노드별 결과\n\n{nodes_block}\n"
                        )
                except Exception as _e_sv:
                    print(f"[pca.run_dag] audit 저장 실패 (graceful): {_e_sv}")
                    saved_path = ""
            # [Phase C] belief 갱신용 풍부 메트릭 — run_stage_tick_async 가 indicators 머지.
            pca_metric = {
                "dag_name": dag_name,
                "final_result_preview": (r.get("final_result") or "")[:200],
                "nodes_done": int(r.get("nodes_done", 0)),
                "tool_calls": int(r.get("tool_calls", 0)),
                "duration_ms": float(r.get("duration_ms", 0.0)),
                "saved_path": saved_path,
            }
            return {**base, "status": STATUS_OK,
                    "result": (
                        f"📊 PCA DAG '{dag_name}' 실행 완료 — "
                        f"노드 {r.get('nodes_done', 0)}개·"
                        f"tool {r.get('tool_calls', 0)}회·"
                        f"{r.get('duration_ms', 0):.0f}ms·"
                        f"결과: {(r.get('final_result') or '')[:200]}"
                        + (f"·saved={saved_path}" if saved_path else "")
                    ),
                    "pca": pca_metric,
                    # [2026-06-01] DAG 결과 문서도 메인 채팅에 카드로 띄움.
                    "artifact_path": saved_path,
                    "artifact_title": f"📊 PCA 결과: {dag_name}"}
        except Exception as e:
            return {**base, "status": STATUS_FAIL,
                    "result": f"pca.run_dag 실패: {type(e).__name__}: {e}"}

    # ── eidos.tool.draft_to_clipboard ──
    # 답변 초안 LLM 생성 → 파일 저장 (audit) + 시스템 클립보드 복사.
    # 외부효과 X (네트워크/파일삭제/메시지 발송 0)·사용자가 Ctrl+V 로 manual paste.
    # safety_mode 무관 — paranoid 에서도 실행 (사용자 통제권 100%).
    if action_id == "eidos.tool.draft_to_clipboard":
        try:
            from llm_module import get_llm_response_async
            purpose = (args.get("target") or args.get("purpose") or "답변 초안").strip()
            brief = (args.get("content") or args.get("brief") or "").strip()
            tone = (args.get("tone") or "친근하고 전문적인").strip()
            if not brief:
                brief = decision.get("reason") or "(빈 컨텍스트 — 일반 답변)"
            # LLM 호출 — 본문 한국어 작성
            text = await get_llm_response_async(
                f"다음 컨텍스트로 답변 초안을 한국어로 작성하라.\n"
                f"톤: {tone}\n"
                f"용도: {purpose}\n\n"
                f"[컨텍스트]\n{brief}",
                max_tokens=2048, timeout=60,
            )
            text = (text or "").strip()
            if not text or text.startswith("[서버") or text.startswith("LLM 오류"):
                return {**base, "status": STATUS_FAIL,
                        "result": f"LLM 초안 생성 실패: {text[:120]}"}
            # 파일 저장 — audit 용 (eidos_files/agents/{stage_id}/outputs/draft_{ts}.md)
            saved_path = ""
            if stage_id:
                try:
                    safe_dir = os.path.join("eidos_files", "agents", stage_id, "outputs")
                    os.makedirs(safe_dir, exist_ok=True)
                    _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                    saved_path = os.path.join(safe_dir, f"draft_{_ts}.md")
                    with open(saved_path, "w", encoding="utf-8") as f:
                        f.write(f"# {purpose}\n\n{text}\n")
                except Exception as _e_sv:
                    print(f"[draft_to_clipboard] 파일 저장 실패 (graceful·클립보드는 진행): {_e_sv}")
                    saved_path = ""
            # 클립보드 복사 — main thread 위임 (등록된 콜백 호출)
            _cb_ok = False
            if _CLIPBOARD_CALLBACK is not None:
                try:
                    _CLIPBOARD_CALLBACK(text, purpose)
                    _cb_ok = True
                except Exception as _e_cb:
                    print(f"[draft_to_clipboard] 클립보드 콜백 실패: {_e_cb}")
            return {**base, "status": STATUS_OK,
                    "result": (
                        f"📋 초안 클립보드 복사{' (콜백 OK)' if _cb_ok else ' (콜백 미등록)'} — "
                        f"{len(text):,}자·purpose='{purpose[:30]}'"
                        + (f"·saved={saved_path}" if saved_path else "")
                    ),
                    # [2026-06-01] 저장된 초안 파일도 메인 채팅에 카드로 띄움.
                    "artifact_path": saved_path,
                    "artifact_title": f"📋 답변 초안: {purpose[:30]}"}
        except Exception as e:
            return {**base, "status": STATUS_FAIL,
                    "result": f"draft_to_clipboard 실패: {type(e).__name__}: {e}"}

    # ── [2026-05-27 자율 개발 모드] eidos.tool.dev.* — inline 분기 (run_tests 제외) ──
    # 3개 모두 outputs/projects/ path-confine — 외부효과 없음. run_tests 는 subprocess 라
    # _EXTERNAL_EFFECT_PREFIXES 통과 후 dispatch.py 가 처리.
    if action_id in (
        "eidos.tool.dev.scaffold_project",
        "eidos.tool.dev.write_code",
        "eidos.tool.dev.package_deliverable",
    ):
        try:
            # [2026-06-02 Phase5] 기존 eidos_dev_actions 폐기 — 3 dev 액션 전부 새 엔진(eidos_dev_engine).
            from eidos_dev_engine.agent_adapter import (
                scaffold_project_for_agent as _dev_scaffold,
                write_code_for_agent as _dev_write_code,
                package_deliverable_for_agent as _dev_package,
            )
        except Exception as e:
            return {**base, "status": STATUS_FAIL,
                    "result": f"eidos_dev_engine import 실패: {type(e).__name__}: {e}"}
        try:
            if action_id == "eidos.tool.dev.scaffold_project":
                r = await _dev_scaffold(args, stage_id)
                if r.get("ok"):
                    return {**base, "status": STATUS_OK,
                            "result": (
                                f"📁 프로젝트 생성: {r.get('project_path')}·"
                                f"파일 {len(r.get('files_created', []))}개·"
                                f"lang={r.get('language')}"
                            ),
                            "dev": r}
                return {**base, "status": STATUS_FAIL,
                        "result": f"scaffold 실패: {r.get('error','')}"}
            if action_id == "eidos.tool.dev.write_code":
                r = await _dev_write_code(args, stage_id)
                if r.get("ok"):
                    return {**base, "status": STATUS_OK,
                            "result": (
                                f"📝 코드 작성: {r.get('file_path')}·"
                                f"{r.get('lines',0)} 줄·{r.get('size_bytes',0):,} bytes·"
                                f"lang={r.get('language')}"
                            ),
                            "dev": r}
                return {**base, "status": STATUS_FAIL,
                        "result": f"write_code 실패: {r.get('error','')}"}
            # package_deliverable
            r = await _dev_package(args, stage_id)
            if r.get("ok"):
                return {**base, "status": STATUS_OK,
                        "result": (
                            f"📦 패키징: {r.get('zip_path')}·"
                            f"{r.get('file_count',0)} 파일·"
                            f"{r.get('zip_size_bytes',0):,} bytes"
                        ),
                        "dev": r}
            return {**base, "status": STATUS_FAIL,
                    "result": f"package 실패: {r.get('error','')}"}
        except Exception as e:
            return {**base, "status": STATUS_FAIL,
                    "result": f"dev.* 실행 예외: {type(e).__name__}: {e}"}

    # ── eidos.tool.ask_user ──
    if action_id == "eidos.tool.ask_user":
        question = (args.get("content") or args.get("question")
                    or decision.get("reason") or "사용자 확인 필요").strip()
        if on_ask_user is None:
            return {**base, "status": STATUS_SKIP,
                    "result": "on_ask_user callback 미연결 — 실 실행 X (history 만)"}
        try:
            answer = on_ask_user(question)
            if asyncio.iscoroutine(answer):
                answer = await answer
            return {**base, "status": STATUS_HITL,
                    "result": f"사용자 응답: {(answer or '(취소)')[:200]}"}
        except Exception as e:
            return {**base, "status": STATUS_FAIL, "result": f"ask_user 실패: {e}"}

    # ── [2026-06-01] 구글 직접 접속 차단 (CAPTCHA 회피·SerpAPI 강제) ──
    # yolo/autorun/승인 어느 path 든 dispatch 전에 가로채 SKIP. fail 아님 (정상 정책).
    _blk = _blocked_web_reason(action_id, args)
    if _blk:
        return {**base, "status": STATUS_SKIP, "result": _blk}

    # ── 외부효과 action — yolo / 콜백 / dry-run 3 분기 ──
    if _is_external_effect(action_id):
        # [Phase 1-B] yolo 모드 — 무조건 실 dispatch
        if safety_mode == "yolo":
            try:
                from eidos_agent_dispatch import dispatch_eidos_tool_action_async
                # [2026-05-27 dev mode] dev.run_tests 가 stage_id 필요 → variables 에 주입
                r = await dispatch_eidos_tool_action_async(
                    decision,
                    variables={"stage_id": stage_id},
                    on_ask_user=on_ask_user,
                    require_yolo=True,
                )
                return r
            except Exception as e:
                return {**base, "status": STATUS_FAIL,
                        "result": f"yolo dispatch 실패: {type(e).__name__}: {e}",
                        "dry_run": False}

        # [2026-05-30 autorun] 승인 없이 normal 모드 실 실행 allowlist (시장조사 web
        # 읽기·로컬 테스트). message.* 발송은 allowlist 제외 → 아래 dry-run 으로 빠짐.
        # paranoid 모드는 위(775)에서 이미 차단됨 → 여기 도달 시 normal 뿐.
        if _is_autorun_external(action_id):
            try:
                from eidos_agent_dispatch import dispatch_eidos_tool_action_async
                r = await dispatch_eidos_tool_action_async(
                    decision,
                    variables={"stage_id": stage_id},
                    on_ask_user=on_ask_user,
                    require_yolo=True,
                )
                if isinstance(r, dict):
                    r["_autorun"] = True
                return r
            except Exception as e:
                return {**base, "status": STATUS_FAIL,
                        "result": f"autorun dispatch 실패: {type(e).__name__}: {e}",
                        "dry_run": False}

        # [Option A3 2026-05-26] normal 모드 + 외부 승인 콜백 — 콜백 호출 → True 면 실 dispatch
        if on_external_approve is not None:
            try:
                _approved = on_external_approve(action_id, args)
                if asyncio.iscoroutine(_approved):
                    _approved = await _approved
            except Exception as e:
                print(f"⚠️ [agent_runner] on_external_approve 콜백 예외 (graceful·DRY_RUN): {e}")
                _approved = False
            if _approved:
                try:
                    from eidos_agent_dispatch import dispatch_eidos_tool_action_async
                    r = await dispatch_eidos_tool_action_async(
                        decision,
                        variables={"stage_id": stage_id},
                        on_ask_user=on_ask_user,
                        require_yolo=True,
                    )
                    # 콜백 거쳐 승인된 dispatch — result 에 표시
                    r["_via_external_approve"] = True
                    return r
                except Exception as e:
                    return {**base, "status": STATUS_FAIL,
                            "result": f"승인 후 dispatch 실패: {type(e).__name__}: {e}",
                            "dry_run": False, "_via_external_approve": True}
            else:
                return {**base, "status": STATUS_DRY_RUN, "dry_run": True,
                        "result": f"외부 승인 거절/timeout — dry-run (action={action_id})",
                        "_via_external_approve": True}

        # [옛 동작] 콜백 없으면 normal/paranoid 다 DRY_RUN
        return {**base, "status": STATUS_DRY_RUN, "dry_run": True,
                "result": f"normal 모드 외부효과 dry-run — args={args}"}

    # ── 미지 action_id ──
    return {**base, "status": STATUS_SKIP,
            "result": f"미지 action_id — 실행 path 없음 ({action_id})"}


# ── 1 tick 실행 메인 ────────────────────────────────────────────────
async def run_stage_one_tick(stage,
                             on_progress: ProgressFn = None,
                             on_ask_user: Optional[Callable] = None,
                             on_external_approve: Optional[Callable] = None) -> dict:
    """1 tick 실행 — 핵심 entry point.

    흐름:
      1. state.json 로드·캔버스 fresh merge·tick 증가
      2. history 최근 로드
      3. predict_and_decide_async (LLM 1회 호출)
      4. validate decision
      5. execute (safety_mode 별 dry-run)
      6. apply belief_updates
      7. state.json save + history.jsonl 3 record append

    Returns: dict (사용자 다이얼로그 표시용 — 모든 결과 포함).
    """
    try:
        from eidos_agent_stage_store import (
            load_state, save_state, append_history, read_history,
        )
    except Exception as e:
        return {"_error": f"agent_stage_store import 실패: {e}"}

    await _emit(on_progress, "tick_started", {"stage_id": stage.id})

    # 1. belief 로드 + 캔버스 fresh merge
    stored = load_state(stage.id)
    fresh = init_beliefs_from_stage(stage)
    beliefs = merge_beliefs(stored, fresh)
    beliefs["tick"] = int(beliefs.get("tick", 0)) + 1
    tick_num = beliefs["tick"]

    # [Wave4-B 2026-05-28] before snapshot — inverse model 용. graceful.
    _before_snapshot = None
    try:
        from eidos_inverse_model import beliefs_to_snapshot as _bts
        _before_snapshot = _bts(beliefs)
    except Exception as _e_bs:
        print(f"[Wave4-inverse] before snapshot 실패 (graceful): {_e_bs}")
        _before_snapshot = None

    # EIDOS repertoire 추출
    eidos_rep: list = []
    for name, a in (beliefs.get("actors") or {}).items():
        if a.get("role") == "self":
            eidos_rep = list(a.get("action_repertoire") or [])
            break

    # [Phase 3-A] tick_start history 에 belief snapshot 포함 — conditional 학습용
    belief_snapshot = {}
    try:
        for name, a in (beliefs.get("actors") or {}).items():
            ind = a.get("indicators") if isinstance(a, dict) else {}
            if isinstance(ind, dict) and ind:
                belief_snapshot[name] = dict(ind)
    except Exception:
        belief_snapshot = {}
    append_history(stage.id, {
        "event": "tick_start",
        "tick_num": tick_num,
        "belief_snapshot": belief_snapshot,
    })

    # 2. history 로드
    history = read_history(stage.id, limit=40)

    # 2b. [Phase 2-A] PatternModel 빌드 — 누적 통계 prior
    pattern_block = ""
    pmodel = None
    try:
        from eidos_pattern_model import PatternModel
        pmodel = PatternModel.from_history(history)
        pattern_block = pmodel.to_prompt_block(eidos_repertoire=eidos_rep)
        # 디스크 캐시 갱신 (graceful)
        pmodel.save(stage.id)
    except Exception as e:
        print(f"[agent_runner] PatternModel 빌드 실패 (graceful·prior 없이 진행): {e}")

    # 2c. [Phase 9-B] GoalTree brief — 1년 위계 milestone 을 LLM 에 inject.
    # 없으면 빈 string 유지 → prompt 변화 X. graceful (load 실패해도 tick 진행).
    goal_block = ""
    _active_milestone_ids: list = []
    _active_milestones: list = []   # [Phase 13] Milestone 객체 list — PCA 매칭용
    _gt = None
    try:
        from eidos_goal_tree import (
            load_goal_tree, as_prompt_brief as _goal_brief,
            get_active_milestones as _goal_active,
        )
        _gt = load_goal_tree(stage.id)
        if _gt is not None and _gt.root_goal_id:
            goal_block = _goal_brief(_gt)
            try:
                _active_milestones = _goal_active(_gt)
                _active_milestone_ids = [m.id for m in _active_milestones if m.id]
            except Exception:
                _active_milestones = []
                _active_milestone_ids = []
    except Exception as e:
        print(f"[agent_runner] GoalTree 로드 실패 (graceful·prior 없이 진행): {e}")

    # 2c-A2. [2026-05-30 No-idle] GoalTree 없는데 stage.goal 있으면 자동 seed.
    # 빈 GoalTree → 활성 milestone 0 → has_work False → observe 고착의 핵심 원인.
    # stage 에 목적(goal)이 있으면 root milestone 으로 박아 "할 일 있음" 확보. graceful.
    try:
        _stage_goal_ni = (getattr(stage, "goal", "") or "").strip()
        if (_gt is None or not getattr(_gt, "root_goal_id", "")) and _stage_goal_ni:
            from eidos_goal_tree import (
                new_goal_tree as _ngt_ni, save_goal_tree as _sgt_ni,
                as_prompt_brief as _gb_ni, get_active_milestones as _gam_ni,
            )
            _gt = _ngt_ni(stage.id, root_title=_stage_goal_ni[:300], horizon="year")
            _sgt_ni(_gt)
            goal_block = _gb_ni(_gt)
            _active_milestones = _gam_ni(_gt)
            _active_milestone_ids = [m.id for m in _active_milestones if m.id]
            print(f"[No-idle] GoalTree 자동 seed (stage.goal): {_stage_goal_ni[:50]!r}")
            append_history(stage.id, {
                "event": "goal_autoseed", "tick_num": tick_num,
                "goal": _stage_goal_ni[:200],
            })
    except Exception as _e_seed_ni:
        print(f"[No-idle] goal seed 실패 (graceful): {_e_seed_ni}")

    # 2c-A3. [2026-05-30 자율 계획] root goal 만 있고 하위 milestone 0 이면 자동 분해.
    # 사용자가 /decompose 수동 실행 안 해도 에이전트가 스스로 목표를 A→B→C 단계로
    # 쪼갬 → hierarchical planner + No-idle 가드가 그 단계들을 tick 마다 진행.
    # "워크플로우를 사람이 넣어줘야 자율 아님" 문제 해결. 1회만(children 생기면 skip).
    # LLM 1콜·graceful. settings "agent_auto_decompose"(default True) 로 끌 수 있음.
    try:
        _auto_dec_on = True
        try:
            _ad_path = os.path.join("eidos_files", "settings.json")
            if os.path.exists(_ad_path):
                with open(_ad_path, "r", encoding="utf-8") as _f_ad:
                    _ad_s = json.load(_f_ad) or {}
                _auto_dec_on = bool(_ad_s.get("agent_auto_decompose", True))
        except Exception:
            _auto_dec_on = True
        if _auto_dec_on and _gt is not None and getattr(_gt, "root_goal_id", ""):
            _root_id_ni = getattr(_gt, "root_goal_id", "")
            _root_ni = _gt.root() if hasattr(_gt, "root") else None
            _child_cnt_ni = 0
            try:
                for _m_ni in _gt.all_milestones():
                    if (getattr(_m_ni, "id", "") != _root_id_ni
                            and getattr(_m_ni, "status", "") == "active"):
                        _child_cnt_ni += 1
            except Exception:
                _child_cnt_ni = 1   # 불확실하면 분해 skip (안전)
            if _child_cnt_ni == 0:
                from eidos_goal_tree import (
                    decompose_goal_with_llm_async as _dec_ni,
                    apply_decomposition as _app_ni,
                    save_goal_tree as _sgt2_ni,
                    get_active_milestones as _gam2_ni,
                    as_prompt_brief as _gb2_ni,
                )
                _root_title_ni = (getattr(_root_ni, "title", "") if _root_ni
                                  else (getattr(stage, "goal", "") or ""))
                _ctx_ni = (getattr(stage, "context_summary", "") or "")[:1000]
                _proto_ni = await _dec_ni(
                    _root_title_ni,
                    parent_horizon=(getattr(_root_ni, "horizon", "year")
                                    if _root_ni else "year"),
                    target_horizon="month",
                    n_children=4,
                    context=_ctx_ni,
                )
                _reg_ni = _app_ni(_gt, _root_id_ni, _proto_ni)
                if _reg_ni:
                    _sgt2_ni(_gt)
                    _active_milestones = _gam2_ni(_gt)
                    _active_milestone_ids = [m.id for m in _active_milestones if m.id]
                    goal_block = _gb2_ni(_gt)
                    print(f"[자율계획] root 자동 분해 → {len(_reg_ni)} milestone: "
                          f"{[m.title[:20] for m in _reg_ni]}")
                    append_history(stage.id, {
                        "event": "goal_auto_decompose", "tick_num": tick_num,
                        "n_children": len(_reg_ni),
                        "titles": [m.title[:40] for m in _reg_ni],
                    })
    except Exception as _e_dec_ni:
        print(f"[자율계획] 자동 분해 실패 (graceful): {_e_dec_ni}")

    # 2c-B. [Wave5-B 2026-05-28] Hierarchical planner — 동적 focus level 결정.
    # GoalTree 있으면 LLM 으로 "지금 어느 level (year/quarter/month/week/task) 에
    # 집중할지" 판단. focus brief 를 goal_block 에 prepend → 다운스트림 LLM 호출
    # 들이 그 컨텍스트로 더 적합한 결정. graceful — 실패해도 tick 진행.
    _hierarchical_focus = None
    try:
        from eidos_hierarchical_planner import (
            pick_hierarchical_focus_async, focus_brief_for_prompt,
        )
        # work_state·시간 등 컨텍스트
        _ws_hp = ""
        try:
            from eidos_belief_core import load_belief as _lb_hp
            _ws_hp = (_lb_hp().work_state or "")
        except Exception:
            pass
        _hierarchical_focus = await pick_hierarchical_focus_async(
            goal_tree=_gt,
            work_state=_ws_hp,
            minutes_since_user=0.0,
        )
        _focus_brief = focus_brief_for_prompt(_hierarchical_focus)
        if _focus_brief and goal_block:
            # focus brief 를 goal_block 위에 prepend (LLM 가 먼저 봄)
            goal_block = _focus_brief + "\n\n" + goal_block
        elif _focus_brief:
            goal_block = _focus_brief
        print(f"[Wave5-hierarchy] focus={_hierarchical_focus.focus_level} "
              f"milestone={_hierarchical_focus.focus_milestone_title[:40]!r} "
              f"depth={_hierarchical_focus.recommended_rollout_depth}")
    except Exception as _e_hp:
        print(f"[Wave5-hierarchy] focus 결정 실패 (graceful): {_e_hp}")
        _hierarchical_focus = None

    # 2d. [Phase 10] LongTermMemory episode brief — 며칠~몇 주 압축 사건 inject.
    # active milestone 이 있으면 그 milestone link 된 episode 우선 (related_milestone_ids).
    # graceful — 빈 memory 면 빈 string → prompt 변화 X.
    episode_block = ""
    try:
        from eidos_long_memory import (
            load_long_memory as _ltm_load,
            as_prompt_brief as _ltm_brief,
        )
        # user_id 는 belief 와 동일 single-user mode (default)
        _ltm = _ltm_load("default")
        episode_block = _ltm_brief(_ltm, related_milestone_ids=_active_milestone_ids)
    except Exception as e:
        print(f"[agent_runner] LongTermMemory 로드 실패 (graceful·prior 없이 진행): {e}")

    # 2e. [Phase 14] competence_block — EIDOS 자기 능력 평가 (메타 인지).
    # PatternModel.action_values 의 분야별 평균 → confidence → 약함/강함 명시.
    # LLM 이 보고 약한 분야 요청 시 delegate_to_user / counter_propose / clarify_question 선택.
    # graceful — pattern_model 없으면 default (모든 분야 0.5·"데이터 부족" tag).
    competence_block = ""
    try:
        from eidos_self_competence import (
            compute_domain_confidence as _comp_dc,
            as_prompt_brief as _comp_brief,
        )
        _action_values = getattr(pmodel, "action_values", {}) if pmodel is not None else {}
        _conf_map = _comp_dc(_action_values)
        competence_block = _comp_brief(_conf_map)
    except Exception as e:
        print(f"[agent_runner] competence brief 빌드 실패 (graceful·prior 없이 진행): {e}")

    # [Wave17-B 2026-05-28] observe/no_op streak 구조적 차단.
    # 직전 OBSERVE_STREAK_THRESHOLD tick 의 decide 가 모두 PASSIVE_ACTIONS 면
    # LLM 에 보이는 beliefs.actors.self.action_repertoire 에서 임시 제거.
    # LLM 호출 후 원본 beliefs 그대로 사용 (save_state·belief_updates 영향 X).
    _streak_aids: list = []
    try:
        for _ev_w17b in reversed(history or []):
            if isinstance(_ev_w17b, dict) and _ev_w17b.get("event") == "decision":
                _d_w17b = _ev_w17b.get("decision") or {}
                _aid_w17b = str(_d_w17b.get("action_id", "") or "")
                if _aid_w17b:
                    _streak_aids.append(_aid_w17b)
                if len(_streak_aids) >= OBSERVE_STREAK_THRESHOLD:
                    break
    except Exception:
        _streak_aids = []
    _streak_active_w17b = (
        len(_streak_aids) >= OBSERVE_STREAK_THRESHOLD
        and all(a in PASSIVE_ACTIONS for a in _streak_aids)
    )
    _llm_beliefs = beliefs
    if _streak_active_w17b:
        _filtered_rep = [a for a in (eidos_rep or [])
                         if a not in PASSIVE_ACTIONS]
        if _filtered_rep:
            import copy as _copy_w17b
            _llm_beliefs = _copy_w17b.deepcopy(beliefs)
            for _name, _a in (_llm_beliefs.get("actors") or {}).items():
                if isinstance(_a, dict) and _a.get("role") == "self":
                    _a["action_repertoire"] = _filtered_rep
                    break
            print(f"[Wave17-B] observe streak={len(_streak_aids)} → "
                  f"repertoire 임시 필터 {len(eidos_rep)} → {len(_filtered_rep)} "
                  f"(observe/no_op 제외)")
            append_history(stage.id, {
                "event": "passive_streak_guard",
                "tick_num": tick_num,
                "streak_actions": list(_streak_aids),
                "filtered_repertoire_size": len(_filtered_rep),
                "original_repertoire_size": len(eidos_rep or []),
            })

    # [Wave17-C 2026-05-29] 시뮬레이션 actor 차단 — ground truth 없는 client 비공개.
    # 위 17-B 가 deepcopy 했을 수 있으니 _llm_beliefs is beliefs 검사로 중복 deepcopy 회피.
    _inactive_actors_w17c: list = []
    for _act_name_w17c, _act_obj_w17c in (beliefs.get("actors") or {}).items():
        if not _is_actor_active_for_llm(_act_obj_w17c):
            _inactive_actors_w17c.append(_act_name_w17c)
    if _inactive_actors_w17c:
        if _llm_beliefs is beliefs:
            import copy as _copy_w17c
            _llm_beliefs = _copy_w17c.deepcopy(beliefs)
        _actors_map_w17c = _llm_beliefs.get("actors") or {}
        _llm_beliefs["actors"] = {
            n: a for n, a in _actors_map_w17c.items()
            if n not in _inactive_actors_w17c
        }
        print(f"[Wave17-C] 시뮬레이션 actor 차단 — LLM 비공개: "
              f"{_inactive_actors_w17c}")
        append_history(stage.id, {
            "event": "inactive_actors_filtered",
            "tick_num": tick_num,
            "filtered_actors": list(_inactive_actors_w17c),
            "remaining_actors_count": len(_llm_beliefs.get("actors") or {}),
        })

    # [2026-05-30 No-idle guard] "관찰만" 구조적 차단 (prompt 변경 X·Wave17 계승).
    # Wave17-B 는 observe 3연속 *후* 반응형이라 느림. 여기선 *선제적*:
    #  · 할 일 있음(active milestone·active actor·방금 받은 사용자 답변)
    #     → observe/no_op/reflect 를 LLM 메뉴에서 제거 → 생산적 action 강제.
    #  · 빈 세계(할 일 0) → observe/no_op 만 제거(reflect 유지) → 침묵 대신
    #     reflect/ask/clarify 로 사용자에게 일거리 확인 (pure observe 고착 방지).
    # settings.json "agent_no_idle_enabled"(default True) 로 끌 수 있음.
    # 주의: 이건 *메뉴* 필터 — LLM 이 그래도 idle 고집하면 4. validate 뒤 최후 방어.
    _no_idle_on = True
    _has_work_ni = False
    try:
        _ni_path = os.path.join("eidos_files", "settings.json")
        if os.path.exists(_ni_path):
            with open(_ni_path, "r", encoding="utf-8") as _f_ni:
                _ni_s = json.load(_f_ni) or {}
            _no_idle_on = bool(_ni_s.get("agent_no_idle_enabled", True))
    except Exception:
        _no_idle_on = True

    if _no_idle_on:
        _IDLE_ALL_NI = ("eidos.meta.observe", "eidos.meta.no_op", "eidos.meta.reflect")
        _IDLE_MIN_NI = ("eidos.meta.observe", "eidos.meta.no_op")
        try:
            if _active_milestones:
                _has_work_ni = True
            # goal_block 은 hierarchical planner 가 goal 없이도 focus brief 로 채우므로
            # has_work 판정엔 부적합 → 실제 GoalTree root 존재 여부로 판단.
            if (not _has_work_ni and _gt is not None
                    and getattr(_gt, "root_goal_id", "")):
                _has_work_ni = True
            if not _has_work_ni:
                for _n_w, _a_w in (_llm_beliefs.get("actors") or {}).items():
                    if (isinstance(_a_w, dict) and _a_w.get("role") != "self"
                            and _is_actor_active_for_llm(_a_w)):
                        _has_work_ni = True
                        break
            if not _has_work_ni:
                for _ev_w in list(reversed(history or []))[:8]:
                    if (isinstance(_ev_w, dict)
                            and _ev_w.get("event") == "user_reply_to_question"):
                        _has_work_ni = True
                        break
        except Exception as _e_hw:
            print(f"[No-idle] has_work 판정 실패 (graceful): {_e_hw}")
            _has_work_ni = False

        _remove_ni = set(_IDLE_ALL_NI if _has_work_ni else _IDLE_MIN_NI)
        try:
            _cur_rep_ni: list = []
            for _n2, _a2 in (_llm_beliefs.get("actors") or {}).items():
                if isinstance(_a2, dict) and _a2.get("role") == "self":
                    _cur_rep_ni = list(_a2.get("action_repertoire") or [])
                    break
            _new_rep_ni = [a for a in _cur_rep_ni if a not in _remove_ni]
            if not _new_rep_ni:
                _new_rep_ni = ["eidos.tool.ask_user"]
            if _new_rep_ni != _cur_rep_ni:
                if _llm_beliefs is beliefs:
                    import copy as _copy_ni
                    _llm_beliefs = _copy_ni.deepcopy(beliefs)
                for _n3, _a3 in (_llm_beliefs.get("actors") or {}).items():
                    if isinstance(_a3, dict) and _a3.get("role") == "self":
                        _a3["action_repertoire"] = _new_rep_ni
                        break
                print(f"[No-idle] has_work={_has_work_ni} → idle 제거 "
                      f"{len(_cur_rep_ni)}→{len(_new_rep_ni)} "
                      f"(removed={sorted(set(_cur_rep_ni) - set(_new_rep_ni))})")
                append_history(stage.id, {
                    "event": "no_idle_guard", "tick_num": tick_num,
                    "has_work": _has_work_ni,
                    "removed": sorted(set(_cur_rep_ni) - set(_new_rep_ni)),
                    "repertoire_size": len(_new_rep_ni),
                })
        except Exception as _e_ni:
            print(f"[No-idle] repertoire 필터 실패 (graceful): {_e_ni}")

    # [2026-05-30 Phase 2 self-force] 학습된 self-applicable 패턴(f())을 EIDOS 실행
    # 정책으로 컴파일해 *강제* 적용. 유추전이 비전의 정점 — 외부 도메인서 배운
    # "방해 차단·노력 지속" 같은 f()가 EIDOS 자기 실행로직을 다시 씀.
    # 모드(settings "agent_self_force_mode"): off / shadow(로그만) / force(강제).
    # 안전: 기본 shadow·strength 임계·ask_user 탈출구·Phase 3 가 검증된 패턴만 강제.
    _self_policy_sf = None
    _sf_mode = "shadow"
    _sf_min_strength = 0.5
    try:
        _sf_path = os.path.join("eidos_files", "settings.json")
        _sf_settings = {}
        if os.path.exists(_sf_path):
            with open(_sf_path, "r", encoding="utf-8") as _f_sf:
                _sf_settings = json.load(_f_sf) or {}
        _sf_mode = str(_sf_settings.get("agent_self_force_mode", "shadow")).lower()
        _sf_min_strength = float(_sf_settings.get("agent_self_force_min_strength", 0.5))
    except Exception:
        _sf_mode = "shadow"
        _sf_min_strength = 0.5

    if _sf_mode in ("shadow", "force"):
        try:
            from eidos_self_policy import compile_self_policy
            _cur_rep_sf = []
            for _n_sf, _a_sf in (_llm_beliefs.get("actors") or {}).items():
                if isinstance(_a_sf, dict) and _a_sf.get("role") == "self":
                    _cur_rep_sf = list(_a_sf.get("action_repertoire") or [])
                    break
            _self_policy_sf = compile_self_policy(
                goal_indicator="",       # 모든 self-applicable 패턴 (일반 자기개선)
                repertoire=_cur_rep_sf,
                ground_fn=None,          # 키워드 grounder (LLM grounder = 후속)
                min_trust=_sf_min_strength,
            )
            if _self_policy_sf.is_empty() or _self_policy_sf.strength < _sf_min_strength:
                _self_policy_sf = None
            else:
                append_history(stage.id, {
                    "event": "self_force_compiled", "tick_num": tick_num,
                    "mode": _sf_mode,
                    "force": list(_self_policy_sf.force_actions),
                    "suppress": list(_self_policy_sf.suppress_actions),
                    "strength": _self_policy_sf.strength,
                    "sources": list(_self_policy_sf.source_pattern_ids),
                })
                if _sf_mode == "force":
                    _new_rep_sf = [a for a in _cur_rep_sf
                                   if a not in _self_policy_sf.suppress_actions]
                    if _self_policy_sf.force_actions:
                        _forced_sf = [a for a in _new_rep_sf
                                      if a in _self_policy_sf.force_actions]
                        if _forced_sf:
                            _new_rep_sf = _forced_sf
                    if "eidos.tool.ask_user" not in _new_rep_sf:
                        _new_rep_sf.append("eidos.tool.ask_user")   # 탈출구
                    if _new_rep_sf != _cur_rep_sf:
                        if _llm_beliefs is beliefs:
                            import copy as _copy_sf
                            _llm_beliefs = _copy_sf.deepcopy(beliefs)
                        for _n2_sf, _a2_sf in (_llm_beliefs.get("actors") or {}).items():
                            if isinstance(_a2_sf, dict) and _a2_sf.get("role") == "self":
                                _a2_sf["action_repertoire"] = _new_rep_sf
                                break
                        print(f"[self-force] 강제 게이트 → rep {len(_cur_rep_sf)}→"
                              f"{len(_new_rep_sf)} force={_self_policy_sf.force_actions} "
                              f"suppress={_self_policy_sf.suppress_actions}")
                        append_history(stage.id, {
                            "event": "self_force_guard", "tick_num": tick_num,
                            "repertoire_size": len(_new_rep_sf),
                        })
                else:  # shadow
                    print(f"[self-force][SHADOW] 강제했다면 force="
                          f"{_self_policy_sf.force_actions} suppress="
                          f"{_self_policy_sf.suppress_actions} (실제 적용 X)")
                    append_history(stage.id, {
                        "event": "self_force_shadow", "tick_num": tick_num,
                        "would_force": list(_self_policy_sf.force_actions),
                        "would_suppress": list(_self_policy_sf.suppress_actions),
                    })
        except Exception as _e_sf:
            print(f"[self-force] 컴파일/게이트 실패 (graceful): {_e_sf}")
            _self_policy_sf = None

    # 3. LLM — predict + decide
    await _emit(on_progress, "llm_call_start", {"tick": tick_num})
    llm_result = await predict_and_decide_async(stage, _llm_beliefs, history,
                                                pattern_block=pattern_block,
                                                goal_block=goal_block,
                                                episode_block=episode_block,
                                                active_milestones=_active_milestones,
                                                competence_block=competence_block)
    if "_error" in llm_result:
        # history error 기록
        append_history(stage.id, {
            "event": "tick_error", "tick_num": tick_num,
            "error": llm_result["_error"],
        })
        return {**llm_result, "tick_num": tick_num, "stage_id": stage.id}

    predictions = llm_result.get("predictions") or {}
    raw_decision = llm_result.get("eidos_decision") or {}
    belief_updates = llm_result.get("belief_updates") or {}

    # 4. validate
    decision = validate_decision(raw_decision, eidos_rep)

    # [2026-05-30 No-idle 최후 방어] 메뉴에서 idle 을 뺐는데도 LLM 이 observe/no_op/
    # reflect 를 고집했거나(validate 는 원본 eidos_rep 로 검증해 통과시킴) action_id
    # 누락으로 no_op fallback 된 경우 — 할 일(has_work) 있으면 idle 을 거부하고
    # clarify_question 으로 치환. 이건 *결정 override* 지만 "할 일 있는데 관찰만"
    # 이라는 정확히 그 버그 상황에서만·사용자에게 구체 작업 묻는 안전 action 으로만
    # 발동. clarify_question 답변은 chat forward 가 pending slot 등록 → 다음 tick
    # history(user_reply_to_question)로 되돌아와 작업 재개 (지난 fix 와 연동).
    if _no_idle_on and _has_work_ni:
        _aid_dec_ni = (decision.get("action_id") or "").strip()
        if _aid_dec_ni in ("eidos.meta.observe", "eidos.meta.no_op",
                           "eidos.meta.reflect"):
            decision = {
                "action_id": "eidos.meta.clarify_question",
                "args": {
                    "what_unclear": "지금 바로 처리할 구체적 작업",
                    "questions": [
                        "이 목표에서 제가 지금 바로 실행할 작업을 하나만 지정해 주세요 "
                        "(예: 'X 코드 작성', 'Y 시장조사', 'Z 초안')."
                    ],
                    "default_assumption": "가장 시급한 활성 milestone 진행",
                },
                "_validation_note": f"no_idle override: {_aid_dec_ni}→clarify_question",
                "_no_idle_override": True,
            }
            print(f"[No-idle] 최후 방어 — idle '{_aid_dec_ni}' → clarify_question "
                  f"치환 (has_work=True)")
            append_history(stage.id, {
                "event": "no_idle_override", "tick_num": tick_num,
                "from_action": _aid_dec_ni,
            })

    # [2026-05-30 Phase 2 self-force 최후 방어] force 모드인데 LLM 이 게이트를 뚫고
    # (validate 는 원본 eidos_rep 로 통과시킴) suppress 액션을 골랐으면 차단.
    # 학습 정책이 "하라"는 구체 액션은 args 가 필요해 함부로 fabricate 불가 →
    # 그 액션을 사용자에게 확인받는 clarify_question 으로 치환 (절대 안 깨짐·HITL).
    if _sf_mode == "force" and _self_policy_sf is not None:
        _aid_sf2 = (decision.get("action_id") or "").strip()
        if _aid_sf2 in (_self_policy_sf.suppress_actions or []):
            _want_sf = (_self_policy_sf.force_actions[0]
                        if _self_policy_sf.force_actions else "(생산적 작업)")
            decision = {
                "action_id": "eidos.meta.clarify_question",
                "args": {
                    "what_unclear": (f"학습된 자기개선 정책상 '{_aid_sf2}'"
                                     f"(억제 대상) 대신 '{_want_sf}' 를 해야 함"),
                    "questions": [
                        f"제 학습 정책이 지금 '{_want_sf}' 쪽을 가리킵니다. "
                        f"바로 실행할 구체 작업을 하나 지정해 주세요."
                    ],
                    "default_assumption": f"{_want_sf} 진행",
                },
                "_validation_note": f"self_force override: {_aid_sf2} 차단",
                "_self_force_override": True,
            }
            print(f"[self-force] 최후 방어 — suppressed '{_aid_sf2}' 차단 → clarify")
            append_history(stage.id, {
                "event": "self_force_override", "tick_num": tick_num,
                "blocked_action": _aid_sf2, "wanted": _want_sf,
            })

    # [2026-05-30 anti-clarify-loop] 되묻기 무한루프 차단 (1.png 증상).
    # clarify → 사용자 "진행해" → 또 clarify → ... 영원히 실행 안 함. 원인: No-idle/
    # LLM 이 idle 일 때마다 clarify 로 치환하는데, 사용자가 답해도 다음 tick 또 idle.
    # 해결: decision 이 clarify_question 인데 최근 history 에 user_reply_to_question
    # 또는 직전 clarify 가 있으면 = "이미 물어봤거나 답을 받음" → 또 묻지 말고 생산적
    # default(draft_to_clipboard)로 *진행*. 사용자 답변 내용을 초안 brief 에 반영.
    if (decision.get("action_id") == "eidos.meta.clarify_question"):
        _looped_al = False
        _last_reply_al = ""
        try:
            for _ev_al in reversed((history or [])[-8:]):
                if not isinstance(_ev_al, dict):
                    continue
                _et_al = _ev_al.get("event")
                if _et_al == "user_reply_to_question":
                    _looped_al = True
                    _last_reply_al = str(_ev_al.get("answer", ""))[:200]
                    break
                if _et_al in ("no_idle_override", "anti_clarify_loop"):
                    _looped_al = True
                    break
                if (_et_al == "decision"
                        and (_ev_al.get("decision") or {}).get("action_id")
                        == "eidos.meta.clarify_question"):
                    _looped_al = True
                    break
        except Exception:
            _looped_al = False
        if _looped_al:
            _goal_al = (getattr(stage, "goal", "") or "현재 목표").strip()[:200]
            _brief_al = (f"사용자 지시: {_last_reply_al}\n" if _last_reply_al else "")
            _brief_al += (f"현재 목표 '{_goal_al}' 와 가장 시급한 활성 milestone 를 "
                          f"한 단계 진행하는 구체적 산출물(초안/계획/다음 액션)을 작성.")
            decision = {
                "action_id": "eidos.tool.draft_to_clipboard",
                "args": {"purpose": f"{_goal_al} 진행", "content": _brief_al},
                "_validation_note": "anti-clarify-loop: 되묻기 차단 → 진행",
                "_anti_loop_proceed": True,
            }
            print("[anti-loop] 되묻기 루프 차단 → draft_to_clipboard 진행 "
                  f"(reply={_last_reply_al[:40]!r})")
            append_history(stage.id, {
                "event": "anti_clarify_loop", "tick_num": tick_num,
                "user_reply": _last_reply_al,
            })

    # history — prediction + decision (분리 기록)
    append_history(stage.id, {
        "event": "prediction", "tick_num": tick_num,
        "predictions": predictions,
    })
    append_history(stage.id, {
        "event": "decision", "tick_num": tick_num,
        "decision": decision,
    })

    await _emit(on_progress, "decision_made", {
        "tick": tick_num,
        "action_id": decision.get("action_id"),
    })

    # 5. execute (safety_mode 기준 + 외부 승인 콜백)
    safety_mode = getattr(stage, "safety_mode", "normal") or "normal"

    # [Wave4 2026-05-28] Prediction engine — action 실행 전 LLM 으로 예측 생성.
    # default OFF (settings.prediction_engine_enabled=true 로 활성화).
    # graceful — 실패해도 tick 흐름 영향 0. tick 당 ~$0.0002 추가 비용.
    _prediction = None
    _pe_settings: dict = {}
    try:
        _settings_path = os.path.join("eidos_files", "settings.json")
        if os.path.exists(_settings_path):
            with open(_settings_path, "r", encoding="utf-8") as _f_pe:
                _loaded = json.load(_f_pe) or {}
                if isinstance(_loaded, dict):
                    _pe_settings = _loaded
    except Exception:
        _pe_settings = {}

    # [Wave4-C 2026-05-28] Mental rollout — top action 외 alternatives 생성 후
    # K-trajectory imagination 비교. 최고 점수 trajectory 의 first_action 으로
    # decision 교체 가능. default OFF — tick 당 ~$0.003 추가 (K=3, depth=2).
    if _pe_settings.get("mental_rollout_enabled", False):
        try:
            from eidos_mental_rollout import compare_rollouts_async
            from llm_module import get_llm_response_async
            _alt_count = max(1, int(_pe_settings.get("mental_rollout_alt_count", 2)))
            _depth = max(1, int(_pe_settings.get("mental_rollout_depth", 2)))
            # [Wave5-B] hierarchical focus 가 있으면 권장 depth 로 override
            if _hierarchical_focus is not None:
                _depth = max(1, min(4, _hierarchical_focus.recommended_rollout_depth))

            # alternative 후보 생성 (LLM 1콜)
            _rep_block_ro = "\n".join(f"- {a}" for a in eidos_rep[:20])
            _alt_prompt = (
                f"[action_repertoire]\n{_rep_block_ro}\n\n"
                f"[현재 선택된 action] {decision.get('action_id', '')}\n"
                f"[goal_context] {(goal_block or '')[:300]}\n\n"
                f"위 선택된 action 과 다른 합리적 대안 {_alt_count}개 골라. "
                f"JSON list 만 출력 — 예: "
                f'[{{"action_id":"eidos.meta.observe","args":{{}}}}]'
            )
            _alt_list: list = []
            try:
                _alt_raw = await asyncio.wait_for(
                    get_llm_response_async(
                        _alt_prompt, max_tokens=1024,
                        response_mime_type="application/json",
                    ),
                    timeout=10.0,
                )
                _alt_parsed = json.loads(_alt_raw)
                if isinstance(_alt_parsed, list):
                    _alt_list = _alt_parsed
                elif isinstance(_alt_parsed, dict):
                    # LLM 이 dict 안에 list 넣었을 경우
                    for v in _alt_parsed.values():
                        if isinstance(v, list):
                            _alt_list = v
                            break
            except Exception as _e_alt:
                print(f"[Wave4-rollout] alt 생성 실패 (graceful): {_e_alt}")

            # candidates = top + alts (dedup by action_id)
            _candidates: list = [{
                "action_id": str(decision.get("action_id", "")),
                "args": dict(decision.get("args", {}) or {}),
            }]
            _seen_aids = {_candidates[0]["action_id"]}
            for _alt in _alt_list[:_alt_count]:
                if not isinstance(_alt, dict):
                    continue
                _alt_aid = str(_alt.get("action_id", "") or "")[:80]
                if not _alt_aid or _alt_aid in _seen_aids:
                    continue
                _seen_aids.add(_alt_aid)
                _candidates.append({
                    "action_id": _alt_aid,
                    "args": _alt.get("args", {}) or {},
                })

            # starting state — goal_block + 최근 history
            _start_state = (goal_block or "") + " | " + "; ".join(
                str(h.get("event", "?")) for h in (history or [])[-3:]
            )
            _start_indicators: dict = {}
            try:
                for name, a in (beliefs.get("actors") or {}).items():
                    if a.get("role") == "self" and isinstance(a, dict):
                        _start_indicators = dict(a.get("indicators") or {})
                        break
            except Exception:
                pass

            # rollout 비교
            _trajs = await compare_rollouts_async(
                candidate_first_actions=_candidates,
                starting_state_text=_start_state[:400],
                starting_indicators=_start_indicators,
                action_repertoire=eidos_rep,
                goal_context=(goal_block or "")[:400],
                depth=_depth,
            )

            # 최고 score 가 원래 decision 과 다르면 swap
            if _trajs:
                _best = _trajs[0]
                _orig_aid = decision.get("action_id", "")
                if (_best.first_action_id
                        and _best.first_action_id != _orig_aid):
                    print(
                        f"[Wave4-rollout] action swap: {_orig_aid} → "
                        f"{_best.first_action_id} "
                        f"(score {_best.final_score:.2f}·"
                        f"reason: {_best.evaluator_reason[:60]})"
                    )
                    decision["action_id"] = _best.first_action_id
                    decision["args"] = dict(_best.first_args or {})
                    decision["reason"] = (
                        (decision.get("reason", "") or "")
                        + f" [rollout swap·score {_best.final_score:.2f}]"
                    )[:400]
                    # 새 decision history 에 다시 기록 (replace)
                    append_history(stage.id, {
                        "event": "decision_replaced_by_rollout",
                        "tick_num": tick_num,
                        "orig_action_id": _orig_aid,
                        "new_action_id": _best.first_action_id,
                        "new_score": _best.final_score,
                    })
                else:
                    print(f"[Wave4-rollout] top action 확정 — "
                          f"score {_best.final_score:.2f}")
        except Exception as _e_ro:
            print(f"[Wave4-rollout] 실패 (graceful·기존 decision 유지): {_e_ro}")

    if _pe_settings.get("prediction_engine_enabled", False):
        try:
            from eidos_prediction_engine import predict_action_outcome_async
            # belief context — actors+indicators 일부만
            _belief_ctx_brief = {}
            try:
                for name, a in list((beliefs.get("actors") or {}).items())[:4]:
                    ind = a.get("indicators") if isinstance(a, dict) else {}
                    if isinstance(ind, dict):
                        _belief_ctx_brief[name] = {
                            k: v for k, v in list(ind.items())[:4]
                        }
            except Exception:
                pass
            _history_brief = ""
            try:
                _hist_tail = history[-3:] if history else []
                _history_brief = "\n".join(
                    f"- {h.get('event','?')}: {str(h)[:100]}" for h in _hist_tail
                )
            except Exception:
                pass
            _prediction = await predict_action_outcome_async(
                action_id=decision.get("action_id", ""),
                args=decision.get("args", {}),
                stage_id=stage.id,
                tick_num=tick_num,
                belief_context=_belief_ctx_brief,
                history_brief=_history_brief,
            )
            print(f"[Wave4-predict] {_prediction.action_id} → "
                  f"expected: '{_prediction.expected_result_text[:60]}' "
                  f"conf={_prediction.confidence:.2f}")
        except Exception as _e_pred:
            print(f"[Wave4-predict] 예측 실패 (graceful·tick 진행): {_e_pred}")
            _prediction = None

    exec_result = await execute_eidos_action(
        decision, safety_mode=safety_mode,
        stage_id=stage.id, on_ask_user=on_ask_user,
        on_external_approve=on_external_approve,
    )
    append_history(stage.id, {
        "event": "execution", "tick_num": tick_num,
        **exec_result,
    })
    await _emit(on_progress, "execution_done", exec_result)

    # [2026-05-30 Phase 3 self-force 학습 루프] 강제 적용한 패턴이 실제로 *작동하는*
    # 행동을 시켰나로 catalog 의 record_outcome 호출 → parameter_weights EMA·
    # success/fail_count·status(3회+성공<30%→abandon·10회+60%→maintained) 갱신.
    # 이게 "검증된 패턴만 강제"의 핵심 — 실패 액션을 강제하던 패턴은 자동 폐기돼
    # 더는 강제되지 않음 (force 모드 안전 척추).
    # 신호: 강제된 액션이 status=ok → positive·fail → negative·dry_run/skip → neutral.
    #       LLM 불복으로 override(clarify) 됐으면 정책이 관철 못 한 것 → neutral(약 decay).
    # 주의: tick 레벨은 *실행 성공* 기반 (빠른 attributable 안전신호). 지표 자체의
    #       장기 변화 학습은 self_improvement 사이클(_measure_pattern_effect)이 별도 담당.
    if (_sf_mode == "force" and _self_policy_sf is not None
            and _self_policy_sf.source_pattern_ids):
        try:
            _exec_status_sf = exec_result.get("status")
            _dec_aid_sf = (decision.get("action_id") or "").strip()
            _was_override_sf = bool(decision.get("_self_force_override")
                                    or decision.get("_no_idle_override"))
            if _was_override_sf:
                _sf_outcome = "neutral"   # 정책이 관철 못 함 (불복/idle)
            elif _dec_aid_sf in (_self_policy_sf.force_actions or []):
                if _exec_status_sf == STATUS_OK:
                    _sf_outcome = "positive"
                elif _exec_status_sf == STATUS_FAIL:
                    _sf_outcome = "negative"
                else:                      # dry_run / skip / hitl
                    _sf_outcome = "neutral"
            else:
                # force set 비고 suppress(idle)만 한 경우 — 억제가 생산적 실행으로
                # 이어졌으면 약한 positive, 아니면 neutral.
                _sf_outcome = ("positive" if _exec_status_sf == STATUS_OK
                               else "neutral")
            from eidos_pattern_catalog import record_outcome as _rec_sf
            for _pid_sf in _self_policy_sf.source_pattern_ids:
                try:
                    _rec_sf(_pid_sf, _sf_outcome)
                except Exception as _e_rec1:
                    print(f"[self-force] record_outcome 실패 ({_pid_sf}): {_e_rec1}")
            print(f"[self-force] 학습 — outcome={_sf_outcome} "
                  f"patterns={_self_policy_sf.source_pattern_ids} "
                  f"(action={_dec_aid_sf}·status={_exec_status_sf})")
            append_history(stage.id, {
                "event": "self_force_outcome", "tick_num": tick_num,
                "outcome": _sf_outcome,
                "patterns": list(_self_policy_sf.source_pattern_ids),
                "action": _dec_aid_sf, "status": _exec_status_sf,
            })
        except Exception as _e_sf_learn:
            print(f"[self-force] 학습 루프 실패 (graceful): {_e_sf_learn}")

    # [Wave4 2026-05-28] Prediction record — 실행 후 actual vs predicted 비교.
    # belief_updates 의 indicator delta 도 함께 비교. graceful·실패해도 진행.
    if _prediction is not None:
        try:
            from eidos_prediction_engine import record_prediction_async
            # actual indicator delta — belief_updates 에서 추출 (있다면)
            _actual_indicators = {}
            try:
                for actor_name, actor_updates in (belief_updates or {}).items():
                    if isinstance(actor_updates, dict):
                        for k, v in actor_updates.items():
                            try:
                                _actual_indicators[k] = float(v)
                            except Exception:
                                continue
            except Exception:
                pass
            _actual_text = str(exec_result.get("result", ""))[:300]
            # skip_llm_eval — 비용 절감 (semantic match 매번 LLM 호출 부담)
            _skip_eval = bool(_pe_settings.get("prediction_skip_semantic_eval", True))
            _pred_err = await record_prediction_async(
                _prediction,
                actual_result_text=_actual_text,
                actual_indicators=_actual_indicators,
                skip_llm_eval=_skip_eval,
            )
            print(f"[Wave4-predict] error={_pred_err.error_score:.2f} "
                  f"surprised={_pred_err.surprised} "
                  f"action={_pred_err.action_id}")
        except Exception as _e_rec:
            print(f"[Wave4-predict] 기록 실패 (graceful): {_e_rec}")

    # 6. belief 갱신
    beliefs = apply_belief_updates(beliefs, belief_updates)

    # 6b. [2026-05-27 ToM↔PCA Phase C] PCA DAG 실행 결과를 actor self 의 indicators 에 머지.
    # LLM 이 belief_updates 로 직접 안 적었어도 시스템이 자동 기록 → 다음 tick LLM 이
    # "방금 무슨 DAG 돌렸고 결과 어땠는지" 확인 가능 (학습 사이클 폐쇄).
    # save_state 직전에 머지해 동일 트랜잭션으로 영속화.
    if (exec_result.get("action_id") == "eidos.tool.pca.run_dag"
            and exec_result.get("status") == STATUS_OK):
        pca_meta = exec_result.get("pca") or {}
        if isinstance(pca_meta, dict) and pca_meta:
            try:
                actors_map = beliefs.get("actors") or {}
                self_actor_name = next(
                    (n for n, a in actors_map.items()
                     if isinstance(a, dict) and a.get("role") == "self"),
                    None,
                )
                if self_actor_name:
                    a = dict(actors_map[self_actor_name])
                    ind = dict(a.get("indicators") or {})
                    ind["last_dag_run"] = str(pca_meta.get("dag_name", ""))[:60]
                    ind["last_dag_at"] = _dt.datetime.now().isoformat(timespec="seconds")
                    ind["last_dag_result_preview"] = str(
                        pca_meta.get("final_result_preview", "")
                    )[:200]
                    ind["last_dag_nodes"] = int(pca_meta.get("nodes_done", 0))
                    ind["last_dag_tools"] = int(pca_meta.get("tool_calls", 0))
                    ind["last_dag_ok"] = True
                    a["indicators"] = ind
                    actors_map[self_actor_name] = a
                    beliefs["actors"] = actors_map
            except Exception as _e_pca_ind:
                print(f"[agent_runner] PCA indicator 머지 실패 (graceful): {_e_pca_ind}")

    # [Wave4-B 2026-05-28] Inverse model — after snapshot + 추론·기록.
    # default OFF — settings.inverse_model_enabled=true. graceful.
    if _pe_settings.get("inverse_model_enabled", False) and _before_snapshot is not None:
        try:
            from eidos_inverse_model import (
                beliefs_to_snapshot as _bts2,
                record_inverse_inference_async,
            )
            _after_snapshot = _bts2(beliefs)
            _inv = await record_inverse_inference_async(
                actual_action_id=str(decision.get("action_id", "")),
                before=_before_snapshot,
                after=_after_snapshot,
                action_repertoire=eidos_rep,
                stage_id=stage.id,
                tick_num=tick_num,
            )
            print(f"[Wave4-inverse] inferred={_inv.inferred_action_id!r} "
                  f"actual={_inv.actual_action_id!r} "
                  f"correct={_inv.correct} conf={_inv.confidence:.2f}")
        except Exception as _e_inv:
            print(f"[Wave4-inverse] 추론 실패 (graceful): {_e_inv}")

    # 7. 영속화
    save_state(stage.id, beliefs)

    # [2026-06-01] 현황판 '관통 연결선' 용 — 이 tick 이 어느 milestone 에 집중했는지.
    # _hierarchical_focus(pick_hierarchical_focus_async 결과)가 매 tick "어느 단계를
    # 위해 이 행동을 하는지" 명시적으로 고른 값. GUI 가 path_to 로 Y→…→focus 족보를
    # 그려 사용자에게 "지금 행동이 최종 목표와 어떻게 연결되는지" 보여줌. graceful.
    _focus_payload = {"milestone_id": "", "milestone_title": "", "level": "", "reasoning": ""}
    try:
        if _hierarchical_focus is not None:
            _focus_payload = {
                "milestone_id": str(getattr(_hierarchical_focus, "focus_milestone_id", "") or ""),
                "milestone_title": str(getattr(_hierarchical_focus, "focus_milestone_title", "") or ""),
                "level": str(getattr(_hierarchical_focus, "focus_level", "") or ""),
                "reasoning": str(getattr(_hierarchical_focus, "reasoning", "") or ""),
            }
    except Exception:
        pass

    await _emit(on_progress, "tick_done", {
        "tick": tick_num,
        "decision": decision,
        "exec": exec_result,
        "focus": _focus_payload,
    })

    # [2026-05-26 Phase 8-B] belief 에 agent tick 결과 기록.
    # action_id 와 status (ok/fail/hitl/skipped) 를 update_from_eidos_action 으로 누적.
    # graceful — belief_integrations 실패해도 tick 결과 영향 X.
    try:
        from eidos_belief_integrations import register_agent_tick as _bi_agent_tick
        _bi_agent_tick(
            stage_name=getattr(stage, "name", "") or stage.id,
            action_id=str(decision.get("action_id", "")),
            status=str(exec_result.get("status", "")),
            summary=str(exec_result.get("result", ""))[:60],
        )
    except Exception as _e_bi_a:
        print(f"[agent_runner] belief tick hook 실패 (graceful): {_e_bi_a}")

    # [Phase 9-C] tick 직후 GoalTree milestone progress 자동 평가.
    # 방금 append 된 prediction/decision/execution event 도 history 에 포함되므로
    # success_criteria 키워드 매칭이 즉시 반영됨. manual override 와 done/abandoned
    # milestone 은 건너뜀 (eidos_goal_tree.evaluate_progress_from_history 가 보장).
    # graceful — goal_tree 없거나 변화 없으면 no-op.
    try:
        from eidos_belief_integrations import auto_evaluate_milestones_from_history as _bi_eval
        _eval_res = _bi_eval(stage.id, history_limit=60)
        if _eval_res.get("n_changed", 0) > 0:
            print(f"[agent_runner] Phase 9-C milestone progress 자동 갱신: "
                  f"{_eval_res['n_changed']}개 leaf + {_eval_res.get('n_internal_recomputed', 0)} internal")
    except Exception as _e_bi_g:
        print(f"[agent_runner] milestone auto-evaluate 실패 (graceful): {_e_bi_g}")

    # [2026-05-27 ToM↔PCA Phase C] user-level belief 에도 PCA DAG metric 누적.
    # register_agent_tick 보다 풍부 (dag_name·nodes·tools·ms). user_belief 의
    # 다음 prompt brief 에서 EIDOS 의 자동화 패턴 자체가 history 에 들어감.
    if exec_result.get("action_id") == "eidos.tool.pca.run_dag":
        try:
            from eidos_belief_integrations import register_pca_dag_run as _bi_pca
            pca_meta = exec_result.get("pca") or {}
            _bi_pca(
                dag_name=str(pca_meta.get("dag_name", "")),
                success=(exec_result.get("status") == STATUS_OK),
                summary=str(pca_meta.get("final_result_preview", ""))[:80],
                nodes_done=int(pca_meta.get("nodes_done", 0)),
                tool_calls=int(pca_meta.get("tool_calls", 0)),
                duration_ms=float(pca_meta.get("duration_ms", 0.0)),
            )
        except Exception as _e_bi_pca:
            print(f"[agent_runner] PCA belief shim 실패 (graceful): {_e_bi_pca}")

    return {
        "tick_num": tick_num,
        "stage_id": stage.id,
        "predictions": predictions,
        "decision": decision,
        "execution_result": exec_result,
        "belief_updates_applied": belief_updates,
        "safety_mode": safety_mode,
        "_raw": llm_result.get("_raw", "")[:1000],
    }


# ── 결과 사람 읽기 요약 (다이얼로그용) ──────────────────────────────
def format_tick_summary_markdown(tick_result: dict) -> str:
    """tick 결과 dict → 마크다운 요약 — 다이얼로그 표시용."""
    if not isinstance(tick_result, dict):
        return "(빈 결과)"
    if "_error" in tick_result:
        return f"## ⚠️ Tick 실패\n\n{tick_result['_error']}"
    lines = [f"## ✅ Tick {tick_result.get('tick_num', '?')} 완료"]
    lines.append(f"_(safety: {tick_result.get('safety_mode', 'normal')})_\n")

    # decision
    d = tick_result.get("decision") or {}
    lines.append("### 🎯 EIDOS 결정")
    lines.append(f"- **action_id**: `{d.get('action_id', '?')}`")
    args = d.get("args") or {}
    if args:
        for k, v in args.items():
            vstr = str(v)
            if len(vstr) > 200:
                vstr = vstr[:200] + "..."
            lines.append(f"  - {k}: {vstr}")
    if d.get("expected_outcome"):
        lines.append(f"- **예상 결과**: {d['expected_outcome']}")
    if d.get("_validation_note") and d["_validation_note"] != "OK":
        lines.append(f"- ⚠️ validation: {d['_validation_note']}")
    lines.append("")

    # execution
    er = tick_result.get("execution_result") or {}
    status_emoji = {"ok": "✅", "fail": "❌", "dry_run": "🛡", "skip": "⏭", "hitl": "💬"}.get(
        er.get("status"), "·"
    )
    lines.append("### 🚀 실행 결과")
    lines.append(f"- **status**: {status_emoji} `{er.get('status', '?')}`")
    if er.get("dry_run"):
        lines.append(f"- 🛡 **DRY-RUN** — 실제 부수효과 없음")
    lines.append(f"- **result**: {er.get('result', '')[:300]}")
    lines.append("")

    # predictions
    preds = tick_result.get("predictions") or {}
    if preds:
        lines.append("### 🔮 ToM 예측 — 다른 행위자의 next-action")
        for actor_name, action_list in preds.items():
            if not isinstance(action_list, list):
                continue
            lines.append(f"\n**{actor_name}**:")
            for a in action_list[:5]:
                if not isinstance(a, dict):
                    continue
                aid = a.get("action_id", "?")
                p = a.get("prob", "?")
                rsn = (a.get("reason", "") or "")[:80]
                lines.append(f"  - `{aid}` (p={p}) — {rsn}")
        lines.append("")

    # belief updates
    bu = tick_result.get("belief_updates_applied") or {}
    if bu:
        lines.append("### 📊 Belief 갱신")
        for actor_name, updates in bu.items():
            if not isinstance(updates, dict):
                continue
            for k, v in updates.items():
                lines.append(f"  - {actor_name}.{k} = {v}")
        lines.append("")

    return "\n".join(lines)


# ── [Phase 1-A 2026-05-26] N tick 자동 루프 ───────────────────────
async def run_stage_loop_async(
    stage,
    max_ticks: int = 5,
    on_progress: ProgressFn = None,
    on_ask_user: Optional[Callable] = None,
    stop_event: Optional[asyncio.Event] = None,
    inter_tick_delay_seconds: float = 1.0,
    no_op_streak_limit: int = 3,
    on_external_approve: Optional[Callable] = None,
) -> dict:
    """[Phase 1-A] 여러 tick 자동 실행 — 종료 조건 검사 후 루프.

    종료 조건 (어느 하나라도 발동):
      1. max_ticks 도달
      2. stop_event.is_set() — 사용자 명시 중단
      3. 연속 N 회 같은 action_id (default 3) — 무한 루프 보호
      4. tick error — 첫 fail 시 즉시 중단 (관용성 낮춤 — 디버그 우선)

    Args:
        max_ticks: 최대 tick 수 (default 5)
        inter_tick_delay_seconds: tick 사이 sleep (rate limit 보호·default 1초)
        no_op_streak_limit: 같은 action_id 연속 N번이면 종료

    Returns: {
        "ticks_executed": int,
        "halt_reason": str,
        "tick_results": list[dict],  # 각 tick 의 run_stage_one_tick 결과
        "summary_markdown": str,
    }
    """
    tick_results: list[dict] = []
    halt_reason = "?"
    last_action_id = ""
    streak = 0
    consecutive_errors = 0          # [2026-06-01] 일시 LLM 오류 누적 — N회 연속이면 halt
    _ERROR_LIMIT = 3

    for tick_idx in range(max_ticks):
        # 종료 체크 — 명시적 stop
        if stop_event is not None and stop_event.is_set():
            halt_reason = f"사용자 중단 (tick {tick_idx} 시작 전)"
            break

        await _emit(on_progress, "loop_tick_about_to_run", {
            "tick_idx": tick_idx, "max_ticks": max_ticks,
        })

        try:
            result = await run_stage_one_tick(
                stage, on_progress=on_progress,
                on_ask_user=on_ask_user,
                on_external_approve=on_external_approve,
            )
        except Exception as e:
            tick_results.append({"_error": f"tick {tick_idx} 예외: {type(e).__name__}: {e}"})
            halt_reason = f"tick {tick_idx} 예외 — {type(e).__name__}: {e}"
            break

        tick_results.append(result)

        # 종료 체크 — tick error (일시 오류는 N회 연속까지 버팀·단발은 다음 tick 재시도)
        if isinstance(result, dict) and result.get("_error"):
            consecutive_errors += 1
            _err_soft = result.get("_error", "")
            await _emit(on_progress, "tick_soft_error", {
                "tick_idx": tick_idx, "error": _err_soft,
                "consecutive": consecutive_errors, "limit": _ERROR_LIMIT,
            })
            if consecutive_errors >= _ERROR_LIMIT:
                halt_reason = (
                    f"tick {tick_idx + 1} 연속 오류 {consecutive_errors}회 "
                    f"(한도 {_ERROR_LIMIT}) — {_err_soft}"
                )
                break
            last_action_id = ""
            streak = 1
            continue
        else:
            consecutive_errors = 0

        # 종료 체크 — 같은 action_id 연속
        decision = result.get("decision") or {}
        action_id = decision.get("action_id", "")
        if action_id == last_action_id and action_id:
            streak += 1
        else:
            streak = 1
            last_action_id = action_id
        if streak >= no_op_streak_limit:
            halt_reason = (
                f"무한 루프 보호 — '{action_id}' {streak}회 연속 "
                f"(no_op_streak_limit={no_op_streak_limit})"
            )
            break

        # 종료 체크 — 명시적 stop (tick 끝났을 때도 한 번 더)
        if stop_event is not None and stop_event.is_set():
            halt_reason = f"사용자 중단 (tick {tick_idx + 1} 종료 후)"
            break

        # tick 사이 sleep
        if tick_idx + 1 < max_ticks and inter_tick_delay_seconds > 0:
            try:
                await asyncio.sleep(inter_tick_delay_seconds)
            except asyncio.CancelledError:
                halt_reason = "asyncio.CancelledError — 외부 cancel"
                break
    else:
        halt_reason = f"max_ticks ({max_ticks}) 도달"

    # 요약 markdown
    summary_lines = [
        f"## 🔄 N tick 루프 종료",
        f"- 실행 tick: **{len(tick_results)}** / 최대 {max_ticks}",
        f"- 종료 사유: {halt_reason}",
        "",
        "### 각 tick 요약",
    ]
    for i, tr in enumerate(tick_results, 1):
        if isinstance(tr, dict) and tr.get("_error"):
            summary_lines.append(f"- **tick {i}**: ❌ ERROR — {tr['_error'][:100]}")
            continue
        d = (tr.get("decision") or {}) if isinstance(tr, dict) else {}
        er = (tr.get("execution_result") or {}) if isinstance(tr, dict) else {}
        status_emoji = {"ok": "✅", "fail": "❌", "dry_run": "🛡", "skip": "⏭",
                        "hitl": "💬"}.get(er.get("status"), "·")
        summary_lines.append(
            f"- **tick {i}**: `{d.get('action_id', '?')}`  → "
            f"{status_emoji} {er.get('status', '?')}  "
            f"({(er.get('result', '') or '')[:80]})"
        )

    await _emit(on_progress, "loop_done", {
        "ticks": len(tick_results), "halt_reason": halt_reason,
    })

    return {
        "ticks_executed": len(tick_results),
        "halt_reason": halt_reason,
        "tick_results": tick_results,
        "summary_markdown": "\n".join(summary_lines),
    }


# ── [Phase 2-B 2026-05-26] Scheduler — 정주기 자동 ▶ ────────────────
async def run_stage_scheduled_async(
    stage,
    interval_seconds: float,
    max_ticks: int = 100,
    on_progress: ProgressFn = None,
    on_ask_user: Optional[Callable] = None,
    stop_event: Optional[asyncio.Event] = None,
    no_op_streak_limit: int = 5,
    on_external_approve: Optional[Callable] = None,
    pre_tick_hook: Optional[Callable] = None,
) -> dict:
    """[Phase 2-B] 정주기 scheduler — 매 interval_seconds 마다 1 tick.

    종료 조건:
      1. max_ticks 도달
      2. stop_event 발화
      3. no_op_streak_limit 회 연속 동일 action (default 5·loop 보다 관대)
      4. tick error 시 즉시 중단 (디버그 정책 유지)
      5. asyncio.CancelledError

    Args:
        interval_seconds: 매 tick 사이 sleep (default 보통 60~300 권장).
        max_ticks: 최대 tick (default 100 — 안전 상한).

    Returns: run_stage_loop_async 와 같은 형태 + schedule meta.
    """
    if interval_seconds < 5:
        # 너무 짧으면 LLM rate limit·비용 폭발
        return {"_error": "interval_seconds 가 5초 미만 — rate limit 보호 위해 거부"}

    tick_results: list[dict] = []
    halt_reason = "?"
    last_action_id = ""
    streak = 0
    # [2026-06-01] 일시 LLM 오류(_error)는 즉시 halt 하지 않고 누적 — N회 연속이면 halt.
    # 단발 비-dict/타임아웃/빈응답 하나로 에이전트가 영구 종료되던 문제 수정.
    consecutive_errors = 0
    _ERROR_LIMIT = 3

    await _emit(on_progress, "schedule_started", {
        "interval": interval_seconds, "max_ticks": max_ticks,
    })

    for tick_idx in range(max_ticks):
        # 매 tick 시작 전 stop 체크
        if stop_event is not None and stop_event.is_set():
            halt_reason = f"사용자 중단 (tick {tick_idx} 시작 전)"
            break

        await _emit(on_progress, "loop_tick_about_to_run", {
            "tick_idx": tick_idx, "max_ticks": max_ticks,
        })

        # [2026-05-27 inbox-poller] pre_tick_hook — tick 시작 직전 호출 (인박스 polling 등).
        # 실패해도 tick 자체는 진행 (graceful).
        if pre_tick_hook is not None:
            try:
                _hook_r = pre_tick_hook()
                if asyncio.iscoroutine(_hook_r):
                    await _hook_r
            except Exception as _e_hook:
                print(f"[agent_runner] pre_tick_hook 실패 (graceful·tick 진행): {_e_hook}")

        try:
            result = await run_stage_one_tick(
                stage, on_progress=on_progress,
                on_ask_user=on_ask_user,
                on_external_approve=on_external_approve,
            )
        except asyncio.CancelledError:
            halt_reason = "asyncio.CancelledError"
            break
        except Exception as e:
            tick_results.append({"_error": f"tick {tick_idx} 예외: {type(e).__name__}: {e}"})
            halt_reason = f"tick {tick_idx + 1} 예외 — {type(e).__name__}"
            break

        tick_results.append(result)

        if isinstance(result, dict) and result.get("_error"):
            consecutive_errors += 1
            _err_soft = result.get("_error", "")
            await _emit(on_progress, "tick_soft_error", {
                "tick_idx": tick_idx, "error": _err_soft,
                "consecutive": consecutive_errors, "limit": _ERROR_LIMIT,
            })
            if consecutive_errors >= _ERROR_LIMIT:
                halt_reason = (
                    f"tick {tick_idx + 1} 연속 오류 {consecutive_errors}회 "
                    f"(한도 {_ERROR_LIMIT}) — {_err_soft}"
                )
                break
            # 일시 오류는 halt 하지 않음 — 아래 streak/interval 로직 거쳐 다음 tick 재시도.
            # (error result 는 decision 비어 있어 streak 자동 리셋됨)
        else:
            consecutive_errors = 0

        d = result.get("decision") or {}
        action_id = d.get("action_id", "")
        if action_id == last_action_id and action_id:
            streak += 1
        else:
            streak = 1
            last_action_id = action_id
        if streak >= no_op_streak_limit:
            halt_reason = (
                f"무한 루프 보호 — '{action_id}' {streak}회 연속 "
                f"(no_op_streak_limit={no_op_streak_limit})"
            )
            break

        # tick 사이 interval — interruptible sleep
        if tick_idx + 1 < max_ticks:
            try:
                if stop_event is not None:
                    # wait 가 timeout 되면 stop 신호 없음·정상 진행
                    try:
                        await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
                        halt_reason = f"사용자 중단 (interval 중·tick {tick_idx + 1} 종료 후)"
                        break
                    except asyncio.TimeoutError:
                        pass   # timeout = 다음 tick 진행
                else:
                    await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                halt_reason = "asyncio.CancelledError — interval 중"
                break
    else:
        halt_reason = f"max_ticks ({max_ticks}) 도달"

    # 요약 markdown — loop 와 같은 형식 + schedule meta
    summary_lines = [
        f"## ⏰ 정주기 Scheduler 종료",
        f"- 실행 tick: **{len(tick_results)}** / 최대 {max_ticks}",
        f"- interval: {interval_seconds:.0f}초",
        f"- 종료 사유: {halt_reason}",
        "",
        "### 각 tick 요약",
    ]
    for i, tr in enumerate(tick_results, 1):
        if isinstance(tr, dict) and tr.get("_error"):
            summary_lines.append(f"- **tick {i}**: ❌ {tr['_error'][:100]}")
            continue
        d = (tr.get("decision") or {}) if isinstance(tr, dict) else {}
        er = (tr.get("execution_result") or {}) if isinstance(tr, dict) else {}
        em = {"ok": "✅", "fail": "❌", "dry_run": "🛡", "skip": "⏭",
              "hitl": "💬"}.get(er.get("status"), "·")
        summary_lines.append(
            f"- **tick {i}**: `{d.get('action_id', '?')}` → {em} "
            f"{er.get('status', '?')}  ({(er.get('result', '') or '')[:80]})"
        )

    await _emit(on_progress, "schedule_done", {
        "ticks": len(tick_results), "halt_reason": halt_reason,
        "interval": interval_seconds,
    })
    # _on_tick_done loop 분기 호환 (loop_done event 도 함께)
    await _emit(on_progress, "loop_done", {
        "ticks": len(tick_results), "halt_reason": halt_reason,
    })

    return {
        "ticks_executed": len(tick_results),
        "halt_reason": halt_reason,
        "tick_results": tick_results,
        "summary_markdown": "\n".join(summary_lines),
        "interval_seconds": interval_seconds,
        "_scheduled": True,
    }


# ── [Phase 4-A 2026-05-26] auto_resume 스캔 헬퍼 ────────────────────
def find_auto_resume_stages() -> list[dict]:
    """모든 stage 디스크 스캔 → `scheduling.auto_resume=True` 인 것 list.

    Returns: [{stage_id, name, interval_seconds, max_ticks, last_run_at, halt_reason}, ...]
    """
    try:
        from eidos_agent_stage_store import list_stages, load_stage
    except Exception as e:
        print(f"[agent_runner] auto_resume 스캔 — store import 실패: {e}")
        return []
    out: list[dict] = []
    try:
        for meta in list_stages():
            stage = load_stage(meta["id"])
            if stage is None:
                continue
            sched = stage.scheduling or {}
            if not sched.get("auto_resume"):
                continue
            out.append({
                "stage_id": stage.id,
                "name": stage.name,
                "interval_seconds": int(sched.get("interval_seconds", 60)),
                "max_ticks": int(sched.get("max_ticks", 10)),
                "last_run_at": sched.get("last_run_at"),
                "last_halt_reason": sched.get("last_halt_reason"),
                "safety_mode": stage.safety_mode,
            })
    except Exception as e:
        print(f"[agent_runner] auto_resume 스캔 실패 (graceful): {e}")
    return out


__all__ = [
    "run_stage_one_tick", "run_stage_loop_async", "run_stage_scheduled_async",
    "predict_and_decide_async",
    "execute_eidos_action", "validate_decision",
    "init_beliefs_from_stage", "merge_beliefs", "apply_belief_updates",
    "format_tick_summary_markdown",
    "find_auto_resume_stages",
    "STATUS_OK", "STATUS_FAIL", "STATUS_SKIP", "STATUS_DRY_RUN", "STATUS_HITL",
]
