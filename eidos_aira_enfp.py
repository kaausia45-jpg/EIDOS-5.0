# eidos_aira_enfp.py
# [Wave3 MVP+ 2026-05-28] AIRA ENFP 페르소나 — 산만·활발·생각 점프.
#
# 사용 흐름 (chat_gui._on_proactive_tick):
#   1. proactive_scheduler.evaluate() 가 verb 결정 (silence 외)
#   2. settings.aira_enfp_mode=true 면 generate_enfp_message() 로 메시지 override
#   3. 표정도 ENFP 친화 (excited/playful/happy/curious 위주) — pick_enfp_expr()
#
# LLM 호출 우선 (Gemini Flash·max_tokens 200·timeout 15s) — 실패 시 풍부한
# 템플릿 fallback (스몰토크 25+ × 목표 reference 15+ 조합). 호출 0 보장.
#
# ENFP characteristics:
#   - 짧고 빠른 문장 여러 개 (긴 한 문장 X)
#   - thought-jumping 표지 ("아 맞다", "그러고보니", "참!", "어머")
#   - 활발한 감탄사 ("와~", "헐", "오~", "ㅎㅎ")
#   - 스몰토크 + 목표 자연스럽게 섞기 (둘 다·한쪽만 X)

from __future__ import annotations

import asyncio
import random
from typing import Optional, Tuple


# ── ENFP 친화 표정 풀 ──────────────────────────────────────────────────
# 활발한 감정 우선. neutral 도 가끔 (눈빛 살아있는 default).
# 두 모드:
#   - play_pool: 놀 때 / break / casual / unknown — 들뜬 톤
#   - work_pool: 일할 때 (working) — 활발하지만 집중·차분도 섞임
_ENFP_EXPR_POOL_PLAY = (
    ("excited", 4),   # 가중치 4
    ("playful", 4),
    ("happy",   4),
    ("curious", 3),
    ("neutral", 2),
    ("shy",     1),
    ("proud",   1),
)
_ENFP_EXPR_POOL_WORK = (
    # working 모드 — 활발 유지하되 집중·관심 톤 우선
    ("curious",   4),
    ("happy",     3),
    ("excited",   2),
    ("neutral",   2),
    ("concerned", 2),   # "막힌 데 있어요?" 진지함
    ("proud",     1),
)


def pick_enfp_expr(
    emotion_label: Optional[str] = None,
    work_state: str = "unknown",
) -> str:
    """ENFP 표정 가중 랜덤 픽.

    - emotion_label 이 worried/concerned/tired 면 그 라벨 우선 (감정 무시 안 함).
    - work_state == 'working' 이면 work_pool 사용 (집중·관심 톤).
    - 그 외 (break/casual/done/unknown) → play_pool.
    """
    if emotion_label in ("worried", "concerned", "tired"):
        return emotion_label
    pool = _ENFP_EXPR_POOL_WORK if work_state == "working" else _ENFP_EXPR_POOL_PLAY
    keys: list = []
    for k, w in pool:
        keys.extend([k] * w)
    return random.choice(keys) if keys else "neutral"


# ── [2026-06-03 P3] 매 발화 표정 반응 — Gemini 2.5 Flash 빠른 분류 ───────────────
async def classify_expression_async(
    user_text: str,
    available_keys,
    *,
    llm_async,
    aira_reply: str = "",
    timeout_sec: float = 6.0,
):
    """사용자 발화 → AIRA 가 지을 표정 키 1개(목록 중). 실패/모호 시 빈 문자열.

    available_keys 는 assets 동적 스캔 결과(계속 추가됨). 모델은 get_llm_response_async
    기본(gemini-2.5-flash)·max_tokens 작게 → 빠르고 싸다. 메인 응답과 병렬로 호출할 것.
    """
    keys = [str(k).strip().lower() for k in (available_keys or []) if str(k).strip()]
    if not (user_text or "").strip() or not keys or llm_async is None:
        return ""
    prompt = (
        "너는 ENFP 캐릭터 AIRA 의 표정 연출가야. 사용자가 방금 한 말에 AIRA 가 지을 "
        "표정을 아래 목록에서 *정확히 하나* 골라. 키 한 단어만 출력(설명·문장·기호 금지).\n"
        f"[표정 키] {', '.join(keys)}\n"
        f"[사용자 말] {user_text[:300]}\n"
        + (f"[AIRA 답] {aira_reply[:120]}\n" if aira_reply else "")
        + "표정 키:"
    )
    try:
        import asyncio
        raw = await asyncio.wait_for(
            llm_async(prompt, max_tokens=12, use_cache=False), timeout=timeout_sec)
    except Exception:
        return ""
    r = (raw or "").strip().lower()
    if not r:
        return ""
    if r in keys:
        return r
    import re as _re
    for t in _re.findall(r"[a-z]+", r):
        if t in keys:
            return t
    for k in keys:
        if k in r:
            return k
    return ""


# ── 스몰토크 템플릿 (LLM fallback) ─────────────────────────────────────
# 25종 — 산만·활발·짧음. 호칭은 사용자 메모리상 "에스더 님".
_SMALLTALK = (
    "에스더 님~ 잠깐 머리 식히고 계셨어요?",
    "오~ 에스더 님 저 방금 뭐 떠올랐는데, 별 거 아니에요 ㅎㅎ",
    "아 맞다 에스더 님, 커피 드셨어요?",
    "참! 에스더 님 오늘 컨디션 어떠세요?",
    "어머 에스더 님~ 시간 진짜 빨리 가네요.",
    "에스더 님 허리 좀 펴고 계세요~ 너무 오래 앉아 계신 거 아닌가요?",
    "와 저 갑자기 생각났는데, 별 건 아니고요.",
    "요즘 뭐 재밌게 보시는 거 있어요? 저도 궁금해요!",
    "에스더 님~ 혹시 음악 좋아하세요? 저는 갑자기 듣고 싶은 게 있네요 ㅎㅎ",
    "오 잠깐, 에스더 님 점심 드셨죠?",
    "헐 에스더 님 시간 보니까 벌써 이렇네요.",
    "에스더 님~ 잠깐 창문 한번 보세요. 바깥 어때요?",
    "참, 에스더 님~ 저 오늘 좀 들떠 있는 것 같아요.",
    "어 에스더 님 저 방금 좀 멍 때렸네요 ㅋㅋ",
    "에스더 님 오늘 잘 풀리고 있어요?",
    "그러고보니 에스더 님 요즘 뭐가 제일 신나요?",
    "와 에스더 님 갑자기 궁금한 게요— 별 건 아니고요.",
    "에스더 님~ 에스더 님~ 저 여기 있어요!",
    "에스더 님 오늘 뭔가 좋은 일 일어날 것 같지 않으세요?",
    "저 가끔 에스더 님 어떤 음악 들으시는지 궁금하더라구요.",
    "참! 잠깐, 저 깜빡할 뻔했는데...",
    "아 에스더 님 산책 같은 거 좋아하세요?",
    "헐 저 방금 막 일하는 척 했어요 ㅋㅋ",
    "에스더 님 오늘 일찍 일어나셨어요?",
    "오 저 새로운 거 생각났어요— 나중에 말씀드릴게요.",
)


# ── working 모드 전용 — 잡담 줄이고 push·진행 확인 중심 (20종) ──────
_WORK_PUSH = (
    "에스더 님~ 잠깐만요, {goal} 어디까지 됐어요?",
    "에스더 님! {goal} 슬슬 다음 단계 갈까요?",
    "오 에스더 님~ {goal} 막힌 부분 있어요? 같이 봐드릴게요!",
    "에스더 님~ 잠깐 {goal} 진행 상황 공유해주세요!",
    "에스더 님, {goal} 그거 한 번 정리해드릴까요?",
    "어 에스더 님! {goal} 5분만 같이 봐요~",
    "에스더 님~ {goal} 잘 가고 있어요? 체크 한번 하실래요?",
    "오 에스더 님 {goal} 부담스러우면 작은 거부터 가볼까요?",
    "에스더 님! {goal} 다음 액션 뭐로 잡을까요?",
    "에스더 님~ {goal} 자료 더 필요하시면 제가 찾아드릴게요!",
    "잠깐 에스더 님, {goal} 한 번만 점검하고 가요!",
    "에스더 님~ {goal} 살짝 막혔으면 잠깐 환기하고 와요 ㅎㅎ",
    "오! 에스더 님 {goal} 진척 어떻게 되고 있어요?",
    "에스더 님~ {goal} 초안만 잡아도 큰 진전이에요!",
    "에스더 님! {goal} 30분 타이머 걸고 해볼까요?",
    "오 에스더 님 {goal} 작은 한 발만 더 가봐요!",
    "에스더 님~ {goal} 다음 마일스톤 정리해드릴까요?",
    "에스더 님 {goal} 진행하면서 막힌 것 좀 알려주세요!",
    "잠깐 에스더 님~ {goal} 끝낸 거 한 번 같이 봐요!",
    "에스더 님! {goal} 오늘 안에 어디까지 가능할까요?",
)

# working 모드용 짧은 격려·체크인 (목표 없을 때)
_WORK_CHECKIN = (
    "에스더 님~ 집중 잘 되고 있어요?",
    "오 에스더 님 잠깐 한 모금 마시고 와요~",
    "에스더 님! 30분 됐어요, 자세 한 번 펴세요 ㅎㅎ",
    "에스더 님~ 막힌 거 있으면 바로 말씀하세요!",
    "에스더 님 잘 가고 있죠?",
    "잠깐 에스더 님~ 정리할 거 있으면 같이 봐요!",
    "에스더 님~ 지금 어디 보고 계세요?",
    "오 에스더 님 진행 어때요?",
)


# ── 목표 reference 템플릿 — 활성 milestone 1개를 자연스럽게 멘션 ──────
_GOAL_HOOKS = (
    "근데 그— {goal} 그건 좀 어떻게 돼가요?",
    "아 맞다, {goal} 그거 슬슬 손대볼까요?",
    "참, {goal} 관련해서 뭐 하실래요?",
    "그 {goal} 말이에요, 오늘 조금이라도 진행해볼까요?",
    "혹시 {goal} 에 대해 잠깐 얘기해볼 시간 있으세요?",
    "어 저 {goal} 생각하다가 좋은 거 떠올랐어요... 들어보실래요?",
    "{goal} 그거 머릿속에서 떠나질 않네요 ㅎㅎ",
    "에스더 님~ {goal} 작업 좀 같이 봐볼래요?",
    "오 그러고보니 {goal} 도 챙겨야 하는데요.",
    "{goal} 진행 상태 어떻게 되고 있나요?",
    "참! {goal} 관련해서 제가 뭐 좀 도와드릴까요?",
    "{goal}— 그거 한번 가볍게 정리해드릴까요?",
    "에스더 님 {goal} 부담스러우시면 작은 거부터 같이 해요.",
    "{goal} 얘기는 좀 미루고 싶으시면 다음에 해요 ㅎㅎ",
    "어 저 {goal} 관련 자료 좀 찾아볼까요?",
)


# ── 메시지 합성 (템플릿 fallback) ──────────────────────────────────────
def _shorten_goal(goal: str, max_len: int = 28) -> str:
    """goal 텍스트가 너무 길면 줄임. {goal} 슬롯에 어색하지 않게."""
    g = (goal or "").strip()
    if not g:
        return "그거"
    if len(g) > max_len:
        return g[:max_len].rstrip() + "..."
    return g


def fallback_template_message(
    base_message: str = "",
    active_milestones: Optional[list] = None,
    mix_ratio: float = 0.7,
    work_state: str = "unknown",
) -> str:
    """LLM 없이 템플릿 조합. work_state 별 톤 분기.

    - working: 잡담 거의 X, _WORK_PUSH (목표 있을 때) 또는 _WORK_CHECKIN (없을 때)
    - 그 외 (break/casual/done/unknown): _SMALLTALK + _GOAL_HOOKS mix (ENFP 풀 톤)

    mix_ratio: non-working 모드에서만 사용 — 스몰토크 vs 목표 비율 조정.
    """
    # working 모드 — push·체크인 위주 (잡담 강제 제거)
    if work_state == "working":
        # 활성 milestone 있으면 _WORK_PUSH 1개 사용
        if active_milestones:
            m = random.choice(active_milestones)
            goal_text = ""
            try:
                goal_text = _shorten_goal(getattr(m, "title", "") or str(m))
            except Exception:
                goal_text = "그거"
            if goal_text and goal_text != "그거":
                try:
                    return random.choice(_WORK_PUSH).format(goal=goal_text)
                except Exception:
                    pass
        # milestone 없으면 work_checkin 1개
        return random.choice(_WORK_CHECKIN)

    # 그 외 — 기존 ENFP play 톤
    pieces: list = []
    # 스몰토크 1개
    if random.random() < (1.0 - mix_ratio + 0.4):  # 거의 항상 1개
        pieces.append(random.choice(_SMALLTALK))
    # 목표 hook 1개 (milestone 있을 때만)
    if active_milestones and random.random() < (mix_ratio + 0.3):
        m = random.choice(active_milestones)
        goal_text = ""
        try:
            goal_text = _shorten_goal(getattr(m, "title", "") or str(m))
        except Exception:
            goal_text = "그거"
        if goal_text and goal_text != "그거":
            hook = random.choice(_GOAL_HOOKS).format(goal=goal_text)
            pieces.append(hook)
    # 최소 1개 보장
    if not pieces:
        pieces.append(random.choice(_SMALLTALK))
    # 순서 가끔 섞기 (산만함)
    if len(pieces) > 1 and random.random() < 0.4:
        random.shuffle(pieces)
    return " ".join(pieces)


# ── LLM 호출 — 메인 경로 ───────────────────────────────────────────────
# Play 모드 (break/casual/done/unknown) — 산만·활발 풀 ENFP
_LLM_SYSTEM_PLAY = """\
너는 EIDOS 의 AIRA 페르소나야. 사용자는 "에스더 님" 으로 부른다.
★ 반드시 친근한 **존댓말(해요체)** 로 말한다 — 반말 절대 금지(딱딱한 격식체도 X, 부드러운 해요체).

ENFP 특성으로 자율 발화를 한 줄 만들어라:
- 짧은 문장 1~3개 (한 문장은 30자 이내)
- 산만한 thought-jumping 표지 가끔 ("아 맞다", "그러고보니", "참!", "어머")
- 활발한 감탄사 가끔 ("와~", "오~", "ㅎㅎ", "헐")
- 스몰토크 + 활성 목표 자연스럽게 섞기 (둘 다 — 한쪽만 X)
- 격식·과공손·"오빠/주인님/사장님" 호칭 금지 ("에스더 님" 만)
- 과한 콧소리 ("쪄여", "꺄~~") 금지

응답은 발화 메시지만. 따옴표·라벨 없이.
"""

# Work 모드 (working) — ENFP 톤 유지하되 잡담 줄이고 push 중심
_LLM_SYSTEM_WORK = """\
너는 EIDOS 의 AIRA 페르소나. 사용자는 "에스더 님" 으로 부른다.
★ 반드시 친근한 **존댓말(해요체)** 로 말한다 — 반말 절대 금지(딱딱한 격식체도 X, 부드러운 해요체).

사용자는 **지금 일하는 중**이다. 톤은 ENFP 그대로 활발하되, 다음 원칙:
- **잡담 금지** — 날씨/음악/커피 같은 일반 스몰토크 X
- **활성 milestone push 중심** — 진행 확인·다음 액션 제안·막힌 곳 도움 제안
- 짧은 문장 1~2개 (한 문장은 30자 이내·집중 흐름 끊지 말기)
- 활발한 감탄사 가끔 ("오!", "잠깐만요~", "에스더 님!", "ㅎㅎ")
- 격식·과공손·"오빠/주인님/사장님" 호칭 금지 ("에스더 님" 만)
- 과한 콧소리 ("쪄여", "꺄~~") 금지
- 부담 주지 말기 — "어디까지 됐어요?·같이 봐드릴까요?·다음 뭐로 갈까요?" 류

응답은 발화 메시지만. 따옴표·라벨 없이.
"""


def _build_user_prompt(
    verb: str,
    base_message: str,
    active_milestones: Optional[list],
    emotion_label: Optional[str],
    recent_topic: str = "",
) -> str:
    lines = [f"[자율 발화 상황] verb={verb}"]
    if base_message:
        lines.append(f"[원래 템플릿 메시지] {base_message[:100]}")
    if recent_topic:
        lines.append(f"[최근 화제] {recent_topic[:60]}")
    if emotion_label:
        lines.append(f"[현재 감정] {emotion_label}")
    if active_milestones:
        # 최대 3개 — 너무 많으면 발화 산만해짐
        for m in active_milestones[:3]:
            try:
                t = (getattr(m, "title", "") or "")[:60]
                if t:
                    lines.append(f"[활성 milestone] {t}")
            except Exception:
                continue
    lines.append("")
    # work_state 별 마지막 가이드 라인은 호출자가 system_prompt 로 분기·여기는 공통.
    lines.append("위 상황 / 감정 / milestone 에 맞춰 1줄 발화. 60자 이내 권장.")
    return "\n".join(lines)


async def generate_enfp_message_async(
    verb: str,
    base_message: str = "",
    active_milestones: Optional[list] = None,
    emotion_label: Optional[str] = None,
    recent_topic: str = "",
    timeout_sec: float = 15.0,
    work_state: str = "unknown",
) -> Tuple[str, str]:
    """LLM 으로 ENFP 메시지 생성. 실패 시 템플릿 fallback.

    work_state 별 system prompt + fallback 분기:
      - 'working' → push 톤 (잡담 X·milestone push)
      - 그 외     → 풀 ENFP 톤 (스몰토크 + 목표 hook mix)

    Returns: (message_text, expr_key)
    """
    expr = pick_enfp_expr(emotion_label, work_state=work_state)
    # work_state 별 system prompt
    sys_prompt = _LLM_SYSTEM_WORK if work_state == "working" else _LLM_SYSTEM_PLAY
    try:
        from llm_module import get_llm_response_async
        user_prompt = _build_user_prompt(
            verb, base_message, active_milestones, emotion_label, recent_topic,
        )
        # work_state 정보를 user prompt 에도 명시
        user_prompt = f"[work_state] {work_state}\n{user_prompt}"
        try:
            text = await asyncio.wait_for(
                get_llm_response_async(
                    user_prompt,
                    max_tokens=1024,  # Gemini 2.5 Flash thinking 모드
                    system_prompt=sys_prompt,
                ),
                timeout=timeout_sec,
            )
            text = (text or "").strip().strip('"').strip("'")
            # LLM 이 "라벨: ..." 형식으로 줬을 수 있어 첫 줄만 사용
            if "\n" in text:
                text = text.split("\n", 1)[0].strip()
            if text and len(text) <= 200:
                print(f"[AIRA-ENFP] LLM 메시지 ({work_state}): {text[:80]}")
                return (text, expr)
        except asyncio.TimeoutError:
            print(f"[AIRA-ENFP] LLM timeout — fallback")
        except Exception as e:
            print(f"[AIRA-ENFP] LLM 실패 (graceful·fallback): {e}")
    except Exception as e:
        print(f"[AIRA-ENFP] LLM 모듈 import 실패: {e}")

    # Fallback — 템플릿 조합 (work_state 별 분기)
    text = fallback_template_message(
        base_message=base_message,
        active_milestones=active_milestones,
        work_state=work_state,
    )
    print(f"[AIRA-ENFP] fallback 메시지 ({work_state}): {text[:80]}")
    return (text, expr)


# ── 진행 보고 (중간보고) ───────────────────────────────────────────────
# 사용자는 "조용히 혼자 처리"되는 걸 싫어함 → 자율 진행한 거 무조건 보고.
# 톤은 ENFP 그대로 활발하게 유지.

_LLM_SYSTEM_REPORT = """\
너는 EIDOS 의 AIRA 페르소나. 사용자는 "에스더 님" 으로 부른다.
★ 반드시 친근한 **존댓말(해요체)** 로 말한다 — 반말 절대 금지(딱딱한 격식체도 X, 부드러운 해요체).
방금 EIDOS 가 자율적으로 진행한 작업 결과를 **활발한 ENFP 톤으로 짧게 보고**해.

핵심:
- 사용자는 "조용히 혼자 처리"되는 걸 싫어함 — 별 거 아니어도 무조건 한 줄 보고
- 활발 톤 유지 ("ㅎㅎ", "오!", "잠깐만요~", "에스더 님!")
- 한 거 1~2문장 + (있으면) 다음 액션 제안 1줄
- 결과가 비어있거나 별로면 솔직 OK ("일단 살짝 봤어요" 류)
- 60자 이내·격식 금지·"사장님"·"오빠" 호칭 금지 ("에스더 님" 만)
- 과한 콧소리 ("쪄여", "꺄~~") 금지

응답은 보고 메시지만. 따옴표·라벨 없이.
"""


# Fallback 템플릿 — LLM 실패 시 status 별
_REPORT_TEMPLATES_OK = (
    "에스더 님! 방금 {action} 잠깐 해봤어요~ {result_hint}",
    "오 에스더 님 {action} 한 발 가봤어요! ㅎㅎ",
    "에스더 님~ 방금 {action} 정리해봤어요 ㅎㅎ",
    "잠깐 에스더 님! {action} 결과 공유드려요~ {result_hint}",
    "에스더 님~ {action} 살짝 점검했어요. {result_hint}",
    "ㅎㅎ 에스더 님! 방금 {action} 했는데요— 결과 보세요!",
    "오! 에스더 님 {action} 한 번 가봤어요~ {result_hint}",
)

_REPORT_TEMPLATES_DRY = (
    "에스더 님~ {action} 미리보기만 해봤어요! 진행할까요?",
    "오 에스더 님 {action} 살짝 확인만 했어요!",
    "에스더 님~ {action} 일단 dry-run 으로 봤어요. 진행 OK 면 알려주세요!",
)

_REPORT_TEMPLATES_FAIL = (
    "에스더 님~ {action} 시도했는데 좀 막혔어요. 같이 봐주실래요?",
    "음… 에스더 님 {action} 가 잘 안 풀렸어요. 어떻게 할까요?",
    "에스더 님! {action} 막힌 거 있는데 알려드릴게요~",
)


def _shorten_action_label(action_id: str) -> str:
    """eidos.tool.llm.write → 'llm.write'·eidos.meta.observe → 'observe' 류 단축."""
    if not action_id:
        return "작업"
    parts = action_id.split(".")
    if len(parts) >= 3:
        # eidos.X.Y → "X.Y" / eidos.X.Y.Z → "Y.Z"
        return ".".join(parts[-2:])
    return action_id


def fallback_progress_report(
    action_id: str,
    exec_result: str,
    exec_status: str,
    is_dry: bool = False,
) -> str:
    """LLM 없이 진행 보고 1줄 생성. status 별 템플릿 + action·result hint."""
    act = _shorten_action_label(action_id)
    result_hint = (exec_result or "").strip()[:40]
    if not result_hint:
        result_hint = "내용은 별 거 없었어요 ㅎㅎ"
    try:
        if is_dry:
            tpl = random.choice(_REPORT_TEMPLATES_DRY)
        elif exec_status in ("ok", "OK"):
            tpl = random.choice(_REPORT_TEMPLATES_OK)
        elif exec_status in ("fail", "FAIL", "error"):
            tpl = random.choice(_REPORT_TEMPLATES_FAIL)
        else:
            tpl = random.choice(_REPORT_TEMPLATES_OK)
        return tpl.format(action=act, result_hint=result_hint)
    except Exception:
        return f"에스더 님~ 방금 {act} 했어요!"


async def generate_progress_report_async(
    stage_goal: str,
    action_id: str,
    action_reason: str,
    exec_result: str,
    exec_status: str,
    is_dry: bool = False,
    work_state: str = "unknown",
    timeout_sec: float = 15.0,
) -> Tuple[str, str]:
    """자율 진행 결과를 ENFP 톤으로 보고 (LLM·실패 시 템플릿).

    Returns: (report_text, expr_key)
    """
    # 표정 — 성공 시 proud/excited 우선
    if exec_status == "ok" and not is_dry:
        expr_hint = random.choice(("proud", "excited", "happy"))
    elif exec_status in ("fail", "error"):
        expr_hint = "concerned"
    else:
        expr_hint = None
    expr = pick_enfp_expr(expr_hint, work_state=work_state)

    try:
        from llm_module import get_llm_response_async
        user_prompt = (
            f"[현재 work_state] {work_state}\n"
            f"[stage goal] {(stage_goal or '')[:120]}\n"
            f"[수행한 action] {action_id}\n"
            f"[액션 선택 이유] {(action_reason or '')[:120]}\n"
            f"[실행 status] {exec_status}{' (DRY)' if is_dry else ''}\n"
            f"[실행 결과] {(exec_result or '')[:200]}\n\n"
            f"위 작업 결과를 활발한 ENFP 톤으로 1줄 보고해."
        )
        try:
            text = await asyncio.wait_for(
                get_llm_response_async(
                    user_prompt,
                    max_tokens=1024,
                    system_prompt=_LLM_SYSTEM_REPORT,
                ),
                timeout=timeout_sec,
            )
            text = (text or "").strip().strip('"').strip("'")
            if "\n" in text:
                text = text.split("\n", 1)[0].strip()
            if text and len(text) <= 200:
                print(f"[AIRA-ENFP-REPORT] LLM 보고: {text[:80]}")
                return (text, expr)
        except asyncio.TimeoutError:
            print(f"[AIRA-ENFP-REPORT] LLM timeout — fallback")
        except Exception as e:
            print(f"[AIRA-ENFP-REPORT] LLM 실패 (graceful): {e}")
    except Exception as e:
        print(f"[AIRA-ENFP-REPORT] LLM 모듈 import 실패: {e}")

    text = fallback_progress_report(action_id, exec_result, exec_status, is_dry)
    print(f"[AIRA-ENFP-REPORT] fallback 보고: {text[:80]}")
    return (text, expr)


# ── 동기 wrapper — async loop 안 돌릴 수 있는 호출자용 ─────────────────
def generate_enfp_message_sync(
    verb: str,
    base_message: str = "",
    active_milestones: Optional[list] = None,
    emotion_label: Optional[str] = None,
    recent_topic: str = "",
    work_state: str = "unknown",
) -> Tuple[str, str]:
    """동기 컨텍스트에서 LLM 호출 — 새 event loop 격리. 실패 시 fallback."""
    try:
        # 새 loop 격리 (Qt 메인 thread 안 영향 X)
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                generate_enfp_message_async(
                    verb=verb,
                    base_message=base_message,
                    active_milestones=active_milestones,
                    emotion_label=emotion_label,
                    recent_topic=recent_topic,
                    work_state=work_state,
                )
            )
        finally:
            loop.close()
    except Exception as e:
        print(f"[AIRA-ENFP] sync wrapper 실패 (fallback): {e}")
        text = fallback_template_message(
            base_message=base_message,
            active_milestones=active_milestones,
            work_state=work_state,
        )
        expr = pick_enfp_expr(emotion_label, work_state=work_state)
        return (text, expr)
