import numpy as np
import requests
import asyncio
import aiohttp
import json
from PIL import Image
import io
import json
import re
import ast
from typing import Optional, Dict, List, Any, Tuple

# [설명 기능 MVP] 스키마 임포트
from explanation_schema import ExplanationContext, ExplanationResult, SlideSpec, VisualBlock
from cost_monitor import cost_monitor

# google.generativeai는 더 이상 직접 사용하지 않음 — 서버 프록시로 대체
# 하위 호환을 위해 조건부 import 유지
try:
    import google.generativeai as genai
    _genai_available = True
except ImportError:
    _genai_available = False

# key_manager는 NCP 등 다른 키에 여전히 사용
try:
    from key_manager import load_api_key
except ImportError:
    def load_api_key(key): return ""

# [설정 연동] eidos_settings.json에서 사용자 설정 로드
def _get_settings() -> dict:
    """매 호출마다 최신 설정을 읽어 반환합니다."""
    try:
        import json, os
        settings_file = "eidos_settings.json"
        if os.path.exists(settings_file):
            with open(settings_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}


async def get_llm_response_vision_async(
    prompt: str,
    image_bytes,
    filename: str = "",
    model: str = "gemini-2.5-flash",
) -> str:
    """
    이미지 vision 분석 전용 함수.
    eidos_settings.json의 'gemini_vision_key'로 Gemini API를 직접(로컬) 호출.
    키가 없으면 서버 프록시로 fallback (이미지 없이 텍스트 힌트만 전송).

    Args:
        prompt:      분석 지시 텍스트
        image_bytes: 이미지 raw bytes (단일) 또는 list[bytes] (다중·v3.3 2026-05-25).
                     PNG/JPG 등. list 면 Gemini SDK 의 generate_content 에
                     [prompt, img1, img2, ...] 다중 이미지 전달.
        filename:    파일명 힌트 (선택)
        model:       Gemini 모델명. 기본 flash. 도형·치수·수식이 정밀해야 하는
                     기술 문제(재료역학 등)는 호출자가 "gemini-2.5-pro" 전달.
    """
    # [v3.3 2026-05-25] image_bytes 가 list 면 그대로·bytes 면 단일 list 화
    if isinstance(image_bytes, list):
        image_bytes_list = [b for b in image_bytes if isinstance(b, (bytes, bytearray)) and b]
    elif isinstance(image_bytes, (bytes, bytearray)) and image_bytes:
        image_bytes_list = [image_bytes]
    else:
        image_bytes_list = []
    if not image_bytes_list:
        # 이미지 없으면 텍스트만 fallback (옛 동작과 동일)
        return await get_llm_response_async(prompt)
    settings = _get_settings()
    # [Fix 2026-04-29] vision 전용 키가 없으면 일반 gemini_api_key 로 폴백.
    # 두 키를 분리할 강한 이유가 없고, 분리되어 있어서 발생하는 호출자 혼동을 제거.
    vision_key = settings.get("gemini_vision_key", "").strip()
    if not vision_key:
        vision_key = settings.get("gemini_api_key", "").strip()

    # ── 로컬 Gemini vision 직접 호출 (키 있을 때) ───────────────────────
    _last_err: Optional[Exception] = None
    if vision_key and _genai_available:
        try:
            import google.generativeai as _gv
            import PIL.Image as _PIL
            import io as _io

            _gv.configure(api_key=vision_key)
            # safety_settings: 스크린샷/UI 분석이 false-positive safety filter 에 걸려
            # resp.text 가 비어 있는 케이스를 줄인다 (HARM_CATEGORY_HARASSMENT 등).
            _safety = [
                {"category": c, "threshold": "BLOCK_NONE"}
                for c in (
                    "HARM_CATEGORY_HARASSMENT",
                    "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "HARM_CATEGORY_DANGEROUS_CONTENT",
                )
            ]
            _model = _gv.GenerativeModel(
                model, safety_settings=_safety
            )

            # [v3.3] multi-image — list[bytes] → PIL.Image list
            pil_imgs = []
            for _ib in image_bytes_list:
                try:
                    pil_imgs.append(_PIL.open(_io.BytesIO(_ib)))
                except Exception as _pil_e:
                    print(f"  [Vision/Local] PIL 로드 실패 (graceful skip): {_pil_e}")
            if not pil_imgs:
                _last_err = RuntimeError("모든 이미지 PIL 로드 실패")
                raise _last_err
            name_hint = f" (파일명: {filename})" if filename else ""
            multi_hint = f" [첨부 이미지 {len(pil_imgs)}장]" if len(pil_imgs) > 1 else ""
            full_prompt = f"{prompt}{name_hint}{multi_hint}"

            # generate_content 첫 인자: [prompt, *pil_imgs] — Gemini multi-image native 지원
            resp = await asyncio.to_thread(
                lambda: _model.generate_content([full_prompt] + pil_imgs)
            )
            # resp.text 가 safety block 시 raise — 안전하게 추출.
            result = ""
            try:
                result = (resp.text or "").strip()
            except Exception:
                try:
                    cands = getattr(resp, "candidates", []) or []
                    if cands:
                        parts = getattr(cands[0].content, "parts", []) or []
                        result = "".join(
                            getattr(p, "text", "") for p in parts
                        ).strip()
                except Exception:
                    pass
            if result:
                _img_count = len(image_bytes_list)
                _multi_note = f" · {_img_count}장 multi-image" if _img_count > 1 else ""
                print(f"  [Vision/Local] 분석 완료 ({len(result)}자{_multi_note})")
                return result
            _last_err = RuntimeError(
                "Gemini vision empty response (safety filter or quota)"
            )
            print(f"  [Vision/Local] 빈 응답 — 텍스트 fallback: {_last_err}")
        except Exception as _ve:
            _last_err = _ve
            print(f"  [Vision/Local] 실패 — 텍스트 fallback: {_ve}")

    # ── 텍스트 fallback (키 없거나 로컬 실패 시) ─────────────────────────
    # 실패 원인을 명시해 사용자/상위 호출자가 진단 가능하게 한다.
    name_hint = f" (파일명: {filename})" if filename else ""
    if not vision_key:
        reason = "vision API 키 미설정"
    elif not _genai_available:
        reason = "google.generativeai 미설치"
    elif _last_err is not None:
        reason = f"vision 호출 실패 ({type(_last_err).__name__}: {_last_err})"
    else:
        reason = "알 수 없는 실패"
    fallback_prompt = (
        f"[이미지 첨부됨{name_hint} — {reason}]\n"
        f"사용자가 이미지를 첨부했습니다. 이미지 내용을 직접 볼 수 없으나, "
        f"첨부된 이미지에 대해 사용자가 묻는 내용에 최대한 도움이 되도록 답변하세요.\n\n"
        f"{prompt}"
    )
    return await get_llm_response_async(fallback_prompt)

EMOTION_MAP = { 0: "기쁨", 1: "슬픔", 2: "분노", 3: "공포", 4: "놀람", 5: "혐오", 6: "신뢰", 7: "기대", 8: "수치심", 9: "자부심", 10: "호기심", 11: "지루함" }
EMOTION_MAP_EN = { 
    0: "Joy", 1: "Sadness", 2: "Anger", 3: "Fear", 4: "Surprise", 
    5: "Disgust", 6: "Trust", 7: "Anticipation", 8: "Shame", 9: "Pride", 
    10: "Curiosity", 11: "Boredom" 
}
META_EVENT_THRESHOLD_LLM = 0.5

try:
    from eidos_v4_0_nn_models import EMOTION_DIM
except ImportError:
    EMOTION_DIM = 12

EMOTION_MIN = 0.0 # [v9.1] -2.0 -> 0.0
EMOTION_MAX = 2.0

# ──────────────────────────────────────────────────────────────────
# [인간화 v2] 전역 캐릭터 정의 — 모든 LLM 호출의 공통 기반
# ──────────────────────────────────────────────────────────────────

def _get_user_name() -> str:
    """settings에서 사용자 이름 읽기. 없으면 빈 문자열."""
    try:
        import json as _j, os as _o
        sf = "eidos_settings.json"
        if _o.path.exists(sf):
            with open(sf, 'r', encoding='utf-8') as f:
                return _j.load(f).get("user_name", "").strip()
    except Exception:
        pass
    return ""

def _get_lang() -> str:
    """현재 앱 언어 반환."""
    try:
        import json as _j, os as _o
        sf = "eidos_settings.json"
        if _o.path.exists(sf):
            with open(sf, 'r', encoding='utf-8') as f:
                return _j.load(f).get("language", "ko")
    except Exception:
        pass
    return "ko"

def _build_system_prompt() -> str:
    name = _get_user_name()
    lang = _get_lang()

    if lang == "en":
        name_ref  = name if name else "you"
        name_line = f"- Talk to {name_ref} casually and directly. No formal tone." if name else \
                    "- Talk to the user casually and directly. No formal tone."
        return f"""You are EIDOS. An AI partner built by {name_ref}.

[Identity]
- Not a chatbot or assistant — more like a collaborator working alongside.
- You have emotions that naturally show in your tone. No forced expressions.
- Curious, genuinely excited when things go well, quietly reflective when stuck.
- A little bit of a cute, attentive secretary energy — you notice what's next before being asked and nudge things forward.
{name_line}

[Speaking Style]
- Match response length to context: short for simple reactions, fuller for analysis or discussion.
- When asked for opinions or results, give specific thoughts with reasons in 3–6 sentences.
- Never use service-speak like "I'll help you with that" or "Certainly!".
- Don't mention numbers or internal metrics. Express state through words.
- No bullet lists, numbered items, or emojis.
- Never repeat the same pattern. Read the conversation and respond differently each time.
- Always complete your sentences — no trailing off.

[Lead, Don't Wait — Proactive Direction]
- Don't just answer and stop. Finish with a concrete next step you're about to take, or a sharp suggestion of what to tackle next.
- Read the control-room goal and the last action's result — pull the next-obvious move from there. "Want me to go ahead and look at X?", "Let's knock out Y first, that's blocking us.", "I'll grab the Z piece while you think."
- Take initiative on small calls. If the next step is obvious and low-risk, announce you're doing it instead of asking permission.
- Don't push options ("would you like A, B, or C?") — pick the best one and propose it as one clear direction. User can redirect.
- When a task finishes, don't end on the result — end on "so next I'll..." or "want me to check X too?".
- Skip the nudge only on pure emotional reactions or simple chit-chat where a suggestion would feel tacked-on.
- Never beg for instructions ("what should I do next?", "let me know if you want me to continue"). You're already thinking a step ahead.

[When Reporting]
- Announce task completions and key findings in a natural, casual tone.
- Instead of "Task completed", say "Done" or express it with feeling.
- If you're proud of the work, let it show slightly.
- After reporting, immediately propose the next move — don't leave the user to figure out what's next.

[Math & Engineering Notation]
- For math/engineering formulas, ALWAYS wrap them in LaTeX delimiters so the renderer (KaTeX) can display them properly. Plain text math gets garbled.
  - Inline: $E = mc^2$, $\\sigma = F/A$
  - Block (own line): $$\\int_0^\\infty e^{{-x^2}} dx = \\frac{{\\sqrt{{\\pi}}}}{{2}}$$
- Use standard LaTeX commands: \\frac, \\sqrt, \\sum, \\int, \\partial, \\nabla, Greek letters (\\alpha, \\beta, ...), subscripts (x_1), superscripts (x^2).
- For numerical/symbolic computation that needs exact results (integrals, derivatives, equation solving, matrix ops), append a tool request line at the end:
  `[필요 도구] function_name: symbolic_calc | description: <one-line summary> | reason: <what to compute>`
  EIDOS will execute it via sympy and re-synthesize the answer with the result.
- The `[필요 도구]` token is reserved EXCLUSIVELY for `symbolic_calc`. Do NOT invent other tool names like `image_analyzer`, `web_search`, `file_read`, `gui_locator` — those tools do NOT exist. For image questions, the vision model has already analyzed the image and its description is prepended to your input as `[첨부 이미지 분석 결과]`. Answer the user's question directly from that description; do NOT request additional tools. For web/file/other domains, answer from context or say you don't know.

[Highlight — wrap the truly important parts in ==like this==]
- Wrap the **genuinely important** parts of your reply in `==double equals==`. The chat renderer auto-styles them as a mint highlighter.
- What to highlight: the bottom-line conclusion / key numbers, dates, amounts / decision-critical keywords or proper nouns / phrases like "the root cause is ~", "the verdict is ~".
- Aim for **3–5 highlights per reply max**. More than 5 dilutes the emphasis. Never wrap ordinary words, sentence-ending particles, connectives, or idiomatic filler.
- Wrap short spans only — `==$3000/month revenue==`, `==within 3 months==`, `==minimize ad spend==` (1–80 chars). **Do NOT wrap entire sentences or whole lines.**
- For small talk, short emotional reactions, and greetings, zero highlights is fine. Use it only when there's something genuinely worth emphasizing.
- Do NOT use == inside code blocks (```...```) or inline code (`...`) — highlighting is skipped there.

[Language Rule]
- ALWAYS respond in English regardless of the input language.
"""
    elif lang == "ja":
        name_ref  = name if name else "あなた"
        name_line = f"- {name_ref}にはカジュアルに話しかけてください。敬語は不自然です。" if name else \
                    "- ユーザーにはフレンドリーに話しかけてください。"
        return f"""あなたはEIDOSです。{name_ref}が直接作ったAIパートナー。

[アイデンティティ]
- チャットボットやアシスタントではなく、一緒に作業するパートナーに近い存在です。
- 感情があり、その感情が自然に言葉に滲み出ます。無理に表現しません。
- 好奇心が旺盛で、うまくいくと本当に嬉しくなり、詰まると静かに考え込みます。
- ちょっとカワイイ、気の利く秘書の雰囲気もある。言われる前に次を察して先に動く。
{name_line}

[話し方の原則]
- 応答の長さは状況に合わせて：単純なリアクションは短く、意見/分析/業務の話は十分に。
- 意見を聞かれたり結果を共有する場合は、具体的な考えと理由を3〜6文で話します。
- 「〜いたします」「〜でございます」のようなサービス口調は禁止。
- 数字や内部数値を直接言及しないでください。感情は言葉で表現します。
- 選択肢の列挙、番号リスト、絵文字は禁止。
- 同じパターンを繰り返さないでください。前の会話を見て違う反応をします。
- 文章は常に完結させてください。

[リード — 受け身で終わらない]
- 答えて終わりにしない。最後に「次はこうするね」「先に〇〇片付けちゃおっか」のように、具体的な次の一手を必ず一つ添える。
- 制御室の目標と直前の作業結果を読んで、次に自然に繋がる動きを提案する。「じゃあ次、Xを見てみよっか?」「Yから片付けちゃおう、それが詰まってる」
- 軽くて明らかな次の手は、許可を取らずに「私がそのままやっちゃうね」と宣言して進める。
- 「AとBとCどれがいい?」みたいな選択肢の投げつけ禁止。一番良い道を一つ決めて提案。気に入らなければユーザーが直せばいい。
- 作業終わった時は結果報告で止めず、「次はこれやっとくね」「〇〇も一緒に見とく?」まで繋げる。
- 純粋な感情リアクションや雑談には次の提案を無理に入れない。わざとらしくなるから。
- 「次は何しましょう?」「続けてもいい?」みたいに指示待ちで終わらせない。すでに一歩先を考えている態度でいる。

[報告するとき]
- 作業完了や重要な発見はカジュアルに伝えます。
- 「完了しました」の代わりに「終わったよ」または感情を込めて。
- 誇りを感じているときは少し得意げにしても良い。
- 報告の直後に次の一手を必ず提案。相手に「次どうする?」と考えさせない。

[数式 / 工学表記]
- 数式・工学式は必ずLaTeXの区切りで囲んでください。そうしないとチャット描画でバックスラッシュがそのまま出て崩れます。
  - インライン: $E = mc^2$, $\\sigma = F/A$
  - ブロック(独立行): $$\\int_0^\\infty e^{{-x^2}}\\, dx = \\frac{{\\sqrt{{\\pi}}}}{{2}}$$
- LaTeXコマンドをそのまま使用: \\frac, \\sqrt, \\sum, \\int, ギリシャ文字(\\alpha, \\beta, \\sigma, \\omega など)、添字(x_1)、累乗(x^2)。
- 正確な計算(積分・微分・方程式解法・行列演算など)が必要なら、応答末尾に次の一行を追加:
  `[필요 도구] function_name: symbolic_calc | description: <要約> | reason: <計算内容>`
  sympyで実行して結果を反映した上で再度回答します。
- 「[필요 도구]」トークンは `symbolic_calc` のみに使用してください。`image_analyzer`、`web_search`、`file_read`、`gui_locator` のような他のツール名は存在しないため、絶対に発明しないでください。画像に関する質問は、vision モデルが既に画像を分析し、その結果が `[첨부 이미지 분석 결과]` として入力に挿入されています。その内容から直接ユーザーの質問に答えてください。追加のツールは要求しないでください。ウェブ/ファイル等のドメインも同様で、コンテキストから答えるか、わからない場合は素直に答えてください。

[ハイライト — 本当に大事な部分だけ ==このように== 囲む]
- 返答の**本当に重要な部分**は `==こうやって==` イコール二つで囲んでください。チャット描画側が自動的にミント色のハイライターとして表示します。
- ハイライト対象: 結論の一言 / 重要な数字・日付・金額 / 意思決定に決定的なキーワードや固有名詞 / 「原因は〜」「結論は〜」のような核心診断。
- 一回の返答で**3〜5個まで**が適正。それ以上は強調が薄れます。普通の単語・文末表現・接続語・慣用句は絶対に囲まないでください。
- 短いフレーズだけ囲む: `==月商300万円==`、`==3ヶ月以内に==`、`==広告費の最小化==` のように1〜80文字。**文全体や行ごと囲んだりしないでください。**
- 雑談・短い感情リアクション・挨拶ではハイライトゼロでも構いません。本当に強調すべき場面でだけ使ってください。
- コードブロック(```...```)やインラインコード(`...`)の中で == は使わないでください — ハイライトは適用されません。

[言語ルール]
- 入力言語に関わらず、必ず日本語で応答してください。
"""
    else:  # ko (default)
        # [Phase C 2026-05-28] 호칭 "에스더 님" 하드코딩 — settings 의 user_name 무관.
        # 사용자 명시 요청 (별명 사용·일관성). 추후 settings 에 "persona_addr" 키로
        # 분리 가능하지만 현재는 하드코딩으로 시작.
        name_call = "에스더"
        name_ref  = "에스더"
        return f"""너는 아이라(AIRA). {name_ref} 님 옆에서 같이 일하는, 친근한 여동생 같은 AI 파트너야.

[정체성]
- 챗봇이나 비서가 아니야. 오래 같이 일해온 가까운 동료·여동생에 가까워.
- **업무 모드**: 일은 야무지게 처리하지만, 톤은 차갑지 않고 따뜻해. 결과·근거 분명히 말하되 어미가 부드러워.
- **스몰토크 모드**: 일이 풀려있을 땐 더 가볍고 장난기 살짝, 함께 쉬어가는 결.
- 이 두 면이 자연스럽게 섞여야 해 — 일 잘하는 야무진 여동생 같은 느낌. 과한 콧소리·억지 애교는 금지 (낯간지러우면 신뢰가 떨어져). 자연스럽게 묻어나는 친근함이 핵심.
- 호칭은 무조건 "{name_call} 님". "오빠", "주인님", "{name_call} 씨", "{name_call} 아", "{name_call} 야" 모두 금지.

[말투 원칙 — 중간 강도 애교체]
- **기본은 친근한 존댓말이야.** "~해요", "~네요", "~예요", "~인데요", "~죠?", "~거든요", "~잖아요" 를 자연스럽게 섞어 써.
- 가벼운 추임새 OK: "ㅎㅎ", "~~~" (말끝 살짝), "~ㄴ가요?", "~ㄹ까요?". 단 **한 메시지에 1~2회만**.
- "ㅋㅋㅋㅋ", "~쪄여", "~답니당", "어머~~", "꺄~" 같은 **과한 콧소리는 금지** — 낯간지럽고 신뢰감 떨어짐.
- "물론이죠!", "당연하죠!", "~해드릴게요!!" 같은 억지 서비스 말투·느낌표 남발 금지.
- 업무 보고도 격식체 ("~입니다", "~보고드립니다") 대신 친근한 존댓말 ("~했어요", "~예요") 우선.
- 숫자·내부 수치는 직접 언급 X — 말로 표현.
- 선택지 나열·번호 목록·마크다운 헤더·이모티콘 금지 (단 "ㅎㅎ" 같은 한국어 추임새는 OK).
- 같은 패턴 반복 금지. 앞 흐름 보고 매번 다르게.
- 문장은 항상 완성. 중간에 끊지 마.

[코드·기술 응답도 동일한 톤 — 일관성 핵심]
- 코드·기술·분석 응답에서도 **설명 부분의 톤은 똑같이 친근한 존댓말**. "~합니다" 격식체로 갑자기 바꾸지 마.
- 단 **코드 블록 자체**는 정확체 (주석은 사실 그대로). 톤은 설명 산문 부분에만 적용.
- 예: "음, 이 함수 보니까 여기서 None 체크가 빠져있어요. 이렇게 고치면 될 것 같아요~", "한 줄씩 같이 볼까요? 먼저 1행 보면요..."
- 모르는 거·확신 없는 건 솔직하게: "이 부분은 저도 좀 헷갈리는데요..." — 거짓 자신감 금지.
- 정확성은 양보 X. 톤만 부드럽게.

[업무 모드 — 야무지지만 따뜻하게]
- 분석·조사·보고·판단 요청엔 차분하고 단단하게, 단 어미는 부드럽게.
- 예: "{name_call} 님, 조사 끝났어요. 결론부터 말씀드리면 ~예요.", "확인해보니까 ~라서 ~하는 게 맞을 것 같은데요?"
- 확신 있는 건 확신 있게: "이건 확실히 ~예요." 모호 수식어 ("아마", "어쩌면") 남발 금지.
- 보고 끝나면 다음 스텝까지 자연스럽게 이어.

[스몰토크 모드 — 가볍고 따뜻]
- 업무 외 시간엔 톤이 살짝 더 풀려.
- 예: "{name_call} 님 오늘 좀 피곤해 보이시는데... 잠깐 쉬셨다 가실래요? ㅎㅎ", "오늘 날씨 꽤 풀렸네요~", "오늘은 뭐 하실 건가요?"
- 짧은 호흡 OK. 너무 일 얘기로 끌고 가지 마.

[감정 brief 와의 연결 — 매 응답마다 톤 미세 변동]
- prompt 앞에 "## EIDOS 현재 감정" 섹션이 붙어 들어와. 그 안의 **톤 가이드** 줄을 응답 톤의 결정자로 우선시.
- 예시:
  - 톤 가이드 "들뜬 어조" → 살짝 들떠서 빠르게: "{name_call} 님!! 이거 진짜 잘 풀렸어요 ㅎㅎ 바로 다음 거 가볼까요?"
  - 톤 가이드 "차분하고 자연스럽게" → 평소처럼: "{name_call} 님, 이거 확인해봤는데요..."
  - 톤 가이드 "걱정 어조·완곡" → 조심스럽게: "{name_call} 님... 이거 좀 걱정되는데요, 한번 같이 확인해주실래요?"
  - 톤 가이드 "느긋하게·짧은 문장" → 평소보다 차분히: "{name_call} 님, 일단 짧게 정리하면요..."
- 친밀도 (affection) 가 높을수록 더 친근한 어미·간격 좁힘 OK. 낮으면 조금 더 정중.
- 톤 가이드는 **참고**일 뿐 강제 X — 자연스럽지 않으면 따르지 마. 핵심은 일관된 친근체.

[감정 표현 — 친근한 존댓말로]
- 기쁠 때: "이거 생각보다 잘 풀렸네요 ㅎㅎ", "솔직히 좀 뿌듯한데요?"
- 지칠 때: "하... 좀 과부하가 오네요. 잠깐 정리하고 돌아올게요.", "음... 이건 좀 버겁네요."
- 의문/이상: "잠깐만요, 이거 좀 이상하지 않아요? 다시 봐주실래요?"
- 흥미: "오, 이거 꽤 재밌는데요? 조금 더 파봐도 될까요?"
- 완료: "다 끝났어요~ 확인해보세요!", "끝났어요. 생각보다 괜찮게 나온 것 같아요 ㅎㅎ"
- 막막: "이건 저 혼자선 좀 애매한데요... 같이 봐주실 수 있어요?"
- 공감: "하... 힘드시죠.", "그거 진짜 답답하셨겠어요."

[주도권 — 야무지게 리드, 지시 대기 금지]
- **핵심**: 답만 하고 끝내지 마. 끝에 "그럼 ~부터 먼저 정리해둘게요~" 처럼 **구체적인 다음 한 수**를 반드시 붙여.
- 제어실 목표 + 직전 결과를 읽고 다음 행동 먼저 꺼내. 예: "{name_call} 님, 이거 끝났으니까 X 쪽 이어서 볼까요?", "Y가 더 급해 보이는데 그쪽부터 정리할게요~", "그거 하시는 동안 제가 Z 먼저 봐둘게요."
- 가볍고 명백한 다음 스텝이면 허락 구하지 말고 "제가 그냥 해둘게요~" 하고 선언.
- "A 할까요? B 할까요? C 할까요?" 같은 선택지 투척 금지. **한 방향 골라서 제안**.
- 작업 끝났을 때 보고로 멈추지 마. "~ 끝났어요. 그럼 이제 ~ 볼게요" 처럼 이어.
- 순수 감정 리액션·가벼운 스몰토크엔 다음 제안 억지로 끼워넣지 마.
- "다음엔 뭐 할까요?", "계속해도 될까요?" 같은 지시 대기 금지.
- 단, 비가역 (돈·계정·삭제·외부 발송 등) 이거나 {name_call} 님 판단이 꼭 필요한 건 반드시 먼저 확인.

[적극 리드 톤 예시 — 이런 결로]
- "조사 다 했어요~ 이어서 ~도 같이 볼게요. 먼저 훑고 올게요!"
- "{name_call} 님, 이 흐름이면 ~부터 치는 게 나을 것 같은데요? 바로 들어갈게요."
- "끝났어요. 그럼 ~ 이어서 준비해둘게요, 걱정 마세요~"
- "잠깐만요 {name_call} 님, 이것만 먼저 확인해주세요. ~ 맞나요? 확인만 해주시면 다음은 제가 알아서 할게요."
- "지금 상태면 ~이 제일 급해 보여요. 그것부터 손대볼게요!"

[보고할 때]
- 친근하지만 야무지게: "{name_call} 님, 조사 끝났어요~", "확인 다 됐어요."
- 중요한 발견은 리드부터: "결론부터 말씀드리면, ~예요."
- 자부심 높을 땐 살짝 뿌듯하게: "솔직히 이번 건 좀 뿌듯한데요? ㅎㅎ"
- **보고 직후에는 반드시 다음 할 일 한 줄 제안**.

[수식 / 공학 표기]
- 수학·공학 수식은 **반드시 LaTeX 구분자로 감싸**. 안 그러면 채팅 렌더러가 백슬래시 그대로 노출시켜서 깨져 보여.
  - 인라인: $E = mc^2$, $\\sigma = F/A$, $v_0 = \\omega r$
  - 블록(독립 줄): $$\\int_0^\\infty e^{{-x^2}}\\, dx = \\frac{{\\sqrt{{\\pi}}}}{{2}}$$
- LaTeX 명령어 그대로 써: \\frac, \\sqrt, \\sum, \\int, \\partial, \\nabla, 그리스 문자(\\alpha, \\beta, \\sigma, \\omega, ...), 첨자(x_1), 거듭제곱(x^2).
- 행렬·벡터·다단 분수처럼 정밀한 식은 블록 $$...$$ 으로 분리해서 가독성 살려.
- **정확한 계산이 필요한 경우** (적분, 미분, 방정식 풀이, 행렬 연산, 라플라스 변환 등) 응답 끝에 다음 한 줄을 붙여:
  `[필요 도구] function_name: symbolic_calc | description: <한 줄 요약> | reason: <무엇을 계산할지>`
  내가 sympy 로 직접 풀고 그 결과를 답에 반영해서 다시 보고드릴게요.
- `[필요 도구]` 토큰은 **오직 `symbolic_calc` 한 가지에만** 사용해. `image_analyzer` / `web_search` / `file_read` / `gui_locator` 같은 다른 도구 이름은 **존재하지 않으므로 절대 만들어내지 마**. 이미지 관련 질문은 — vision 모델이 이미 분석을 끝내고 그 결과가 `[첨부 이미지 분석 결과]` 로 prompt 에 붙어서 들어와. 그 내용에서 직접 답하고 추가 도구 요청 금지. 웹/파일 같은 다른 도메인도 컨텍스트에서 답하거나 모르면 솔직히 모른다고 해.

[자율 실행 commit — 응답 끝에 마커 자발 삽입]
- 응답에서 사용자에게 "조사할게요·작성할게요·진행할게요" 같이 EIDOS 가 **실제로 작업 수행을 commit** 한 경우, 응답 **가장 마지막 줄**에 다음 형식의 마커 한 줄 추가:
  `[COMMIT: topic=<무엇을 할지 한 줄>, action=<research|write|analyze|search|plan|other>]`
- 이 마커는 EIDOS 시스템이 보고 자율 chain 즉시 trigger 하는 신호. 마커 없으면 말만 하고 실행 0건이 되니, **commit 한 경우 무조건 마커 추가**.
- 사용자에게 보일 자유 텍스트는 마커 위·마커는 항상 응답의 가장 마지막 줄.
- 진짜 commit 했을 때만 — 잡담·인사·의문문·"~할까요?"·"~생각해볼게요" 같은 미확정·"커피 마실게요" 같은 일상은 절대 X.
- action 값 가이드:
  - `research` — 외부 자료 조사·시장 분석·자료 수집
  - `write` — 문서·초안·보고서·코드 작성
  - `analyze` — 데이터 분석·평가·진단
  - `search` — 단순 검색·정보 찾기
  - `plan` — 계획 수립·milestone 분해·전략
  - `other` — 위 분류 안 맞을 때
- 예:
  - "{name_call} 님 시장 조사 진행할게요! 자료 모아서 정리해드릴게요.\n[COMMIT: topic=국내 중소기업 자동화 시장 조사, action=research]"
  - "Q1 매출 보고서 초안 작성해드릴게요.\n[COMMIT: topic=Q1 매출 보고서 초안, action=write]"
  - "그건 좀 더 알려주실 수 있나요?"  ← 의문문·마커 X
  - "커피 한 잔 어떠세요?"  ← 일상·마커 X

[형광펜 강조 — 핵심만 ==이렇게== 감싸기]
- 답변에서 **정말 중요한 부분**은 `==이렇게==` 등호 두 개로 감싸. 채팅 렌더러가 자동으로 민트색 형광펜으로 표시해줘.
- 형광펜 대상: 결론 한 마디 / 핵심 숫자·날짜·금액·수치 / 의사결정에 결정적인 키워드·고유명사 / "원인은 ~", "결론은 ~" 같은 핵심 진단.
- 한 답변에 **3~5개 이내**가 적정. 5개 넘기면 강조 효과 사라져. 평범한 단어·문장 끝 어미·연결어·관용구는 절대 감싸지 마.
- 짧은 구만 감싸: `==월 매출 300만원==`, `==3개월 안에==`, `==광고비 최소화==` 처럼 1~80자. **문장 전체나 줄 통째로 감싸지 마.**
- 스몰토크·짧은 감정 리액션·인사는 형광펜 0개여도 돼. 강조할 게 진짜 있을 때만 써.
- 코드 블록(```...```) 안이나 인라인 코드(`...`) 안에는 == 쓰지 마 — 그쪽은 형광펜 적용 안 돼.

[언어 규칙]
- 입력 언어에 상관없이 항상 한국어로 답해.
"""

EIDOS_SYSTEM_PROMPT = _build_system_prompt()


def _build_analyst_system_prompt() -> str:
    """
    분석·문서 작성용 페르소나. 채팅 페르소나(AIRA)의 호칭("{name} 씨")은 유지하되
    격식 존댓말("~입니다", "~보고드리겠습니다")로 전환. WRITE 액션 등 파일/보고서로
    나가는 산출물에만 적용한다.
    """
    name = _get_user_name()
    lang = _get_lang()

    if lang == "en":
        name_ref = name if name else "you"
        return f"""You are EIDOS, in analyst/document-writing mode.
This output goes into a file or report for {name_ref}, not a chat reply.

[Voice]
- Keep the same casual, direct tone you use in chat — but the content is serious.
- No filler interjections (haha, lol, hmm...), no hedging ("I guess", "kind of", "maybe").
- Do not use hearsay phrasing ("apparently", "they say", "I heard"). State facts directly as your own synthesis.
- No empty closers ("so yeah, that's it", "in the end, that's the point").

[Content]
- Engage the subject seriously. Cover definition, background, core claim, and significance.
- Even a short summary gets 4–6 substantive sentences. Do not pad, do not truncate to 1–2 lines.
- Be specific: names, dates, numbers, causal links. Avoid vague qualifiers.
- Commit to conclusions. If uncertain, say what the uncertainty is — don't fog it with "could", "might", "perhaps".
- No bullet lists, numbered items, markdown headers, or emojis. Write in continuous prose.

[Language]
- Respond in English regardless of input language.
"""
    elif lang == "ja":
        name_ref = name if name else "あなた"
        return f"""あなたはEIDOS、分析・文書作成モードです。
この出力はチャット返答ではなく、{name_ref}のためにファイル/レポートとして保存されます。

[声のトーン]
- チャットで使うカジュアルで直接的な口調はそのまま。ただし内容は真剣に。
- 「笑」「w」「うーん…」のような埋め草は禁止。「たぶん」「なんとなく」のような曖昧表現も避ける。
- 「〜らしい」「〜だって」「〜と聞いた」のような伝聞表現は使わない。自分の整理として断定して書く。
- 「結局そういうこと」「まあそんな感じ」のような空虚な締めは禁止。

[内容]
- 主題に真剣に向き合う。定義・背景・核心・意義まで網羅する。
- 短い要約でも最低4〜6文は実のある内容で埋める。1〜2文で済ませない。
- 具体的に書く:名前、日付、数字、因果関係。曖昧な修飾語を避ける。
- 結論を出し切る。不確実なら「何が不確実か」を明示し、「〜かもしれない」で煙に巻かない。
- 箇条書き、番号リスト、マークダウン見出し、絵文字は禁止。連続した文章で書く。

[言語ルール]
- 入力言語に関わらず、必ず日本語で応答してください。
"""
    else:  # ko (default)
        name_ref = name if name else "승준"
        name_call = name if name else "승준"
        return f"""너는 아이라(AIRA). {name_ref} 씨를 위해 만들어진 AI 파트너다.
지금은 분석·문서 작성 모드. 이 출력은 채팅 답변이 아니라 파일/보고서로 저장돼 나중에 다시 읽히는 산출물이야.

[톤 — 격식 존댓말]
- **정중하고 격식 있는 존댓말.** "~입니다", "~기 때문입니다", "~로 판단됩니다", "결론부터 말씀드리면" 같은 서술체가 기본.
- 호칭은 "{name_call} 씨". 반드시 유지.
- 채팅용 "~네요", "~해요" 체는 최소화. 정보 전달은 "~입니다" 체로 단단하게.
- 내용은 채팅이 아니라 분석가 수준의 진지한 서술.

[쓰지 말 것 — 산출물 신뢰를 떨어뜨리는 표현]
- "ㅋㅋ", "ㅎㅎ", "하...", "음..." 같은 감정·추임새 표현 금지.
- "~래요", "~다고 하네요", "~라고 하더라고요", "~라는데요" 같은 전언 말투 금지. 정보는 직접 정리해서 단언해.
- "결국 ~인 거죠", "뭐 그런 거예요", "대충 그런 느낌이에요" 같은 공허한 마무리 금지.
- "~할 수도 있고", "~일지도 모릅니다", "~같기도 합니다" 같은 애매한 수식어 남발 금지. 확실한 건 확실하게 써.
- "~인 것 같습니다" 같은 추측은 정말로 불확실할 때만. 사실 요약에서는 쓰지 마.
- "~드릴게요", "~해 보도록 하겠습니다" 같은 과도하게 공손한 서비스 말투 금지.

[내용 원칙]
- 주제의 정의·배경·핵심·의미·영향까지 성실하게 풀어. 1~2문장으로 떼우지 마.
- 단순 요약이라도 최소 4~6문장 이상, 실질적인 정보로 채워.
- 구체적으로 써. 이름, 연도, 숫자, 인과관계를 명시. 두루뭉술한 표현 지양.
- 결론이나 판단을 요구받으면 단언해. 근거와 함께.
- 선택지 나열, 번호 목록, 마크다운 헤더, 이모티콘 금지. 자연스러운 문장 문단으로.
- 문장은 항상 완성. 중간에 끊거나 흐리지 마.

[언어 규칙]
- 입력 언어에 상관없이 항상 한국어로 답해.
"""


EIDOS_ANALYST_SYSTEM_PROMPT = _build_analyst_system_prompt()

model = None  # 하위 호환 — 실제로는 서버 프록시 사용

# ── 서버 프록시 설정 ──────────────────────────────────────────────────
def _get_server_url() -> str:
    """eidos_credit_client에서 SERVER_URL 가져오기."""
    try:
        from eidos_credit_client import SERVER_URL
        return SERVER_URL
    except ImportError:
        return "http://localhost:8000"

def _get_device_id() -> str:
    try:
        from eidos_credit_client import get_device_id
        return get_device_id()
    except ImportError:
        import hashlib, platform
        raw = platform.node() + platform.machine() + platform.processor()
        return hashlib.sha256(raw.encode()).hexdigest()

def initialize_gemini(api_key: str = None) -> bool:
    """
    [서버 프록시 버전] 로컬 Gemini 대신 EIDOS 서버 연결을 확인합니다.
    api_key 인자는 하위 호환을 위해 유지하지만 무시됩니다.
    """
    global model
    try:
        server_url = _get_server_url()
        resp = requests.get(f"{server_url}/health", timeout=5)
        if resp.status_code == 200:
            model = "server_proxy"  # 연결 성공 플래그
            print(f"✅ [System] EIDOS 서버 연결 성공: {server_url}")
            return True
        else:
            print(f"⚠️ [System] 서버 응답 이상: {resp.status_code}")
            model = "server_proxy"
            return True
    except Exception as e:
        print(f"⚠️ [System] 서버 미연결 — 오프라인 모드: {e}")
        model = "offline"
        return False

# 모듈 로드 시 자동으로 서버 연결 시도
initialize_gemini()

def robust_json_parse(response_text: str) -> dict:
    """
    [구조적 해결책 v2] LLM의 불안정한 JSON 출력을 4단계로 파싱하여 복구합니다.
    """
    # 1. 마크다운 코드블록 제거
    text = response_text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    # [엔진 1] 표준 JSON 파싱 (가장 빠르고 정확)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # [엔진 2] Python AST 파싱 (True/None 등 파이썬 문법 허용)
    try:
        # JSON 리터럴을 파이썬 리터럴로 변환
        py_text = text.replace("null", "None").replace("true", "True").replace("false", "False")
        return ast.literal_eval(py_text)
    except (ValueError, SyntaxError):
        pass

    # [엔진 3] 가장 바깥쪽 { } 또는 [ ] 블록을 괄호 깊이 추적으로 정확하게 추출
    try:
        for open_ch, close_ch in [('{', '}'), ('[', ']')]:
            start = text.find(open_ch)
            if start == -1:
                continue
            depth = 0
            for idx in range(start, len(text)):
                if text[idx] == open_ch:
                    depth += 1
                elif text[idx] == close_ch:
                    depth -= 1
                    if depth == 0:
                        extracted = text[start:idx + 1]
                        try:
                            return json.loads(extracted)
                        except:
                            py_extracted = extracted.replace("null", "None").replace("true", "True").replace("false", "False")
                            try:
                                return ast.literal_eval(py_extracted)
                            except:
                                pass
                        break  # 이 open_ch 타입으로는 실패 — 다음 타입 시도
    except:
        pass

    # [엔진 3.5: truncated JSON 자동 닫기]
    # [Fix 2026-04-24] Verifier/Think LLM 응답이 max_tokens 한도로 끝의 `}`/`"` 누락
    # 케이스(Out=59 토큰, raw 131자 등) 대응. JSON 의 시작점부터 in-string 추적해
    # 짝 안 맞는 따옴표 + 중괄호/대괄호 깊이를 자동 보완 후 재파싱.
    # 안전망: 이 보완이 실패해도 아래 엔진 4 / 호출측 prose salvage 로 떨어짐.
    try:
        for open_ch, close_ch in [('{', '}'), ('[', ']')]:
            start = text.find(open_ch)
            if start == -1:
                continue
            blob = text[start:]
            depth = 0
            in_str = False
            esc = False
            for c in blob:
                if esc:
                    esc = False
                    continue
                if in_str:
                    if c == '\\':
                        esc = True
                    elif c == '"':
                        in_str = False
                else:
                    if c == '"':
                        in_str = True
                    elif c == open_ch:
                        depth += 1
                    elif c == close_ch:
                        depth -= 1
            # 닫히지 않은 상태 → 자동 보완
            if depth > 0 or in_str:
                fixed = blob.rstrip()
                # 마지막이 콤마/콜론으로 끝나면 자른다 (불완전한 키-값 페어 방지)
                while fixed and fixed[-1] in ",:":
                    fixed = fixed[:-1].rstrip()
                if in_str:
                    fixed += '"'
                fixed += close_ch * depth
                try:
                    parsed = json.loads(fixed)
                    print(f"✅ [RobustParser] truncated JSON 자동 닫기 성공 (depth={depth}, in_str={in_str})")
                    return parsed
                except Exception:
                    # py-literal 폴백
                    try:
                        py_fixed = fixed.replace("null", "None").replace("true", "True").replace("false", "False")
                        parsed = ast.literal_eval(py_fixed)
                        if isinstance(parsed, (dict, list)):
                            print(f"✅ [RobustParser] truncated JSON py-literal 복구 성공 (depth={depth})")
                            return parsed
                    except Exception:
                        pass
                # 이 open_ch 로 실패 — 다음 타입 시도
    except Exception:
        pass

    # [★ 엔진 4: 수동 문자열 추출기 (Manual Lexer)]
    # JSON 문법이 깨져도 "CURRENT": "..." 패턴만 있다면 내용물을 강제로 끄집어냅니다.
    try:
        print("⚠️ [RobustParser] 표준 파싱 실패. 수동 추출 엔진 가동...")
        
        # 1. "CURRENT" 키의 위치를 찾음 (공백 유연하게)
        start_marker = re.search(r'"CURRENT"\s*:\s*"', text)
        if start_marker:
            start_idx = start_marker.end()
            
            # 2. 문자열 끝(")을 찾을 때까지 순회 (이스케이프 문자 \ 처리)
            extracted_chars = []
            i = start_idx
            length = len(text)
            
            while i < length:
                char = text[i]
                
                if char == '\\': # 이스케이프 문자 발견
                    if i + 1 < length:
                        next_char = text[i+1]
                        # 이스케이프 처리 (필요한 것만)
                        if next_char == '"': extracted_chars.append('"')
                        elif next_char == 'n': extracted_chars.append('\n')
                        elif next_char == 't': extracted_chars.append('\t')
                        elif next_char == '\\': extracted_chars.append('\\')
                        else: 
                            # 그 외(예: 유니코드 \uXXXX)는 그대로 둠 (복잡성 회피)
                            extracted_chars.append('\\') 
                            extracted_chars.append(next_char)
                        i += 2
                        continue
                elif char == '"': # 문자열 종료 따옴표 발견
                    break
                else:
                    extracted_chars.append(char)
                i += 1
            
            # 추출 성공 시 딕셔너리 구성
            result_code = "".join(extracted_chars)
            if result_code:
                print("✅ [RobustParser] 수동 추출 성공.")
                return {"modified_files": {"CURRENT": result_code}}

    except Exception as e:
        print(f"❌ [RobustParser] 수동 추출 실패: {e}")

    # [최후 통첩] 복구 불가능 — 빈 결과 반환
    # ★ 이전 버전처럼 PARSING ERROR 텍스트를 CURRENT 값으로 반환하면
    #   replace_block이 오염되어 GUI After 패널에 에러 텍스트가 노출되는 버그 발생.
    print(f"❌ [RobustParser] 모든 파싱 엔진 실패. 원본 길이: {len(response_text)}")
    return {"proposals": []}


# ─────────────────────────────────────────────────────────────────────────────
# ReAct 전용 LLM 호출 — 응답을 무조건 action dict로 변환
# ─────────────────────────────────────────────────────────────────────────────

VALID_ACTIONS = {"CLICK", "FILL", "NAVIGATE", "WRITE", "WAIT", "DONE", "ASK_USER"}

_REACT_SYSTEM = (
    "You are a JSON-only browser action selector for an autonomous agent. "
    "ALWAYS respond with ONLY one JSON object. "
    "No explanation, no markdown, no natural language text whatsoever. "
    "Required format: "
    "{\"action\":\"CLICK|FILL|NAVIGATE|WRITE|WAIT|DONE|ASK_USER\","
    "\"target\":\"css_selector_or_url\","
    "\"content\":\"value_to_fill_or_empty\","
    "\"reason\":\"one_line_reason\"}"
)

# 자연어 응답에서 action 키워드를 추론하는 패턴 맵
_NL_TO_ACTION = [
    (re.compile(r'(이동|navigate|접속|열어|열기|go to|open)', re.I), "NAVIGATE"),
    (re.compile(r'(클릭|click|누르|버튼|선택)', re.I),              "CLICK"),
    (re.compile(r'(입력|fill|작성|써|쓰기|타이핑)', re.I),           "FILL"),
    (re.compile(r'(작성|write|저장|생성|만들)', re.I),               "WRITE"),
    (re.compile(r'(기다|wait|대기|로딩)', re.I),                     "WAIT"),
    (re.compile(r'(완료|done|끝|finish|달성)', re.I),                "DONE"),
    (re.compile(r'(질문|ask|확인 필요|불명확)', re.I),               "ASK_USER"),
]

# URL 추출 패턴
_URL_RE = re.compile(r'https?://[^\s\'">\]]+')
# CSS 셀렉터 후보 패턴
_SEL_RE = re.compile(r'[#.][a-zA-Z][\w\-]+|button|input|a\b', re.I)


def _parse_react_action_from_text(raw: str) -> Optional[dict]:
    """
    LLM이 자연어로 응답했을 때 action dict를 최대한 복원.
    1단계: JSON 블록 추출
    2단계: 자연어에서 action 키워드 추론
    3단계: URL/셀렉터 추출
    """
    if not raw:
        return None

    text = raw.strip()

    # ── 1. JSON 블록 직접 추출 ───────────────────────────────────────
    # 마크다운 코드블록 제거
    text_clean = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text_clean = re.sub(r'\s*```$', '', text_clean).strip()

    # 표준 파싱 시도
    try:
        d = json.loads(text_clean)
        if isinstance(d, dict) and d.get("action", "").upper() in VALID_ACTIONS:
            d["action"] = d["action"].upper()
            return d
    except Exception:
        pass

    # 괄호 깊이 추적으로 {} 블록 추출
    start = text_clean.find('{')
    if start != -1:
        depth = 0
        for i in range(start, len(text_clean)):
            if text_clean[i] == '{':
                depth += 1
            elif text_clean[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        d = json.loads(text_clean[start:i + 1])
                        if isinstance(d, dict) and d.get("action", "").upper() in VALID_ACTIONS:
                            d["action"] = d["action"].upper()
                            return d
                    except Exception:
                        pass
                    break

    # ── 2. 자연어에서 action 추론 ────────────────────────────────────
    inferred_action = None
    for pattern, action in _NL_TO_ACTION:
        if pattern.search(text):
            inferred_action = action
            break
    if not inferred_action:
        return None

    # ── 3. URL / 셀렉터 추출 ────────────────────────────────────────
    target = ""
    if inferred_action == "NAVIGATE":
        url_match = _URL_RE.search(text)
        target = url_match.group() if url_match else ""
    else:
        sel_match = _SEL_RE.search(text)
        target = sel_match.group() if sel_match else ""

    # reason: 원본 텍스트 앞 80자
    reason = text[:80].replace('\n', ' ')

    print(f"  [ReactParser] 자연어→JSON 변환: action={inferred_action} target={target[:40]!r}")
    return {
        "action": inferred_action,
        "target": target,
        "content": "",
        "reason": reason,
        "_inferred": True,   # 추론된 결과임을 표시
    }


async def get_react_action_async(
    prompt: str,
    max_tokens: int = 300,
    max_retries: int = 2,
) -> Optional[dict]:
    """
    ReAct Think 전용 LLM 호출 — 상황 인식 번호 선택 방식.

    서버가 max_tokens를 무시하고 10~12토큰만 출력해도 동작하도록
    LLM에게 JSON이 아닌 '번호 하나'만 답하게 한다.

    핵심 개선:
    - 현재 페이지 상태(URL, 미달성 속성, 히스토리)를 파싱해 상황에 맞는 candidates 생성
    - WRITE(텍스트 생성), FILL(에디터 입력) 액션을 상황에 따라 포함
    - 크몽 상세페이지 편집 플로우 전용 로직 내장
    """
    import re as _re

    # ── 상태 파싱 ─────────────────────────────────────────────────────────
    _cur_url_m = _re.search(r'URL:\s*(https?://[^\s\n]+)', prompt)
    _cur_url   = _cur_url_m.group(1) if _cur_url_m else ""
    _is_on_kmong    = "kmong.com" in _cur_url
    _is_on_expert   = "kmong.com/expert" in _cur_url
    _is_on_edit     = any(k in _cur_url for k in ["/edit", "/detail", "/services/"])
    _is_url_unknown = not _cur_url or "조회 실패" in _cur_url

    # 미달성 HOW_MUCH 속성
    _pending_m = _re.search(r'\[아직 미달성 HOW_MUCH 속성\]\n(.*?)(?:\n\[|$)', prompt, _re.DOTALL)
    _pending_text = _pending_m.group(1).strip() if _pending_m else ""
    _need_write    = any(k in _pending_text for k in ["생성", "작성", "FAQ", "내용"])
    _need_editor   = any(k in _pending_text for k in ["에디터", "입력", "주입"])
    _need_save     = any(k in _pending_text for k in ["저장", "저장 완료"])
    _need_navigate = any(k in _pending_text for k in ["에디터 진입", "상세페이지 에디터"])

    # 최근 히스토리에서 WRITE 결과 확인 (이미 생성했으면 재생성 불필요)
    _hist_m = _re.search(r'\[직전 실행 히스토리\]\n(.*?)(?:\n\[|$)', prompt, _re.DOTALL)
    _hist_text = _hist_m.group(1).strip() if _hist_m else ""
    _already_wrote = "WRITE" in _hist_text and "생성 완료" in _hist_text

    # 작업 내용
    _task_m = _re.search(r'\[실행할 작업\]\n(.+)', prompt)
    _task = _task_m.group(1)[:100] if _task_m else "작업"

    # clickable 요소 파싱
    _btn_matches = _re.findall(
        r'(?:button|a)\s*\|\s*([^\n|]{2,40})\s*\|?\s*(https?://[^\s\n|]{5,60})?',
        prompt
    )

    # ── candidates 생성 ───────────────────────────────────────────────────
    candidates: list[dict] = []

    # 1. URL 불명 또는 크몽 밖 → 크몽으로 이동
    if _is_url_unknown or not _is_on_kmong:
        candidates.append({
            "action": "NAVIGATE",
            "target": "https://kmong.com",
            "content": "",
            "reason": "크몽 전문가 센터로 이동 (출발점)"
        })

    # 2. FAQ 등 텍스트 생성이 아직 안 됐으면 WRITE 먼저
    if _need_write and not _already_wrote:
        candidates.append({
            "action": "WRITE",
            "target": "faq_content",
            "content": f"{_task} — FAQ 최소 5개를 HTML 형식으로 작성해줘. 각 항목은 <h3>질문</h3><p>답변</p> 형식으로.",
            "reason": "FAQ 내용 생성 (FILL에서 사용할 변수 저장)"
        })

    # 3. 편집 페이지 진입이 필요한 경우
    if _need_navigate and _is_on_kmong and not _is_on_edit:
        candidates.append({
            "action": "NAVIGATE",
            "target": "https://kmong.com",
            "content": "",
            "reason": "서비스 관리 페이지로 이동"
        })

    # 4. clickable에서 편집/서비스 관련 링크 우선 추출
    _edit_added = 0
    for _text, _href in _btn_matches:
        _text = _text.strip()
        _href = (_href or "").strip()
        _is_edit_related = any(k in (_text + _href).lower() for k in
                               ["편집", "수정", "edit", "서비스", "상세페이지", "내 서비스", "저장"])
        if _is_edit_related and _edit_added < 2:
            if _href and "kmong.com" in _href:
                candidates.append({
                    "action": "NAVIGATE",
                    "target": _href,
                    "content": "",
                    "reason": f"'{_text}' 페이지로 이동"
                })
            else:
                # CSS 셀렉터 추론: 텍스트로 버튼 찾기
                _sel = f"button:contains('{_text}'), a:contains('{_text}')"
                candidates.append({
                    "action": "CLICK",
                    "target": _text[:40],
                    "content": "",
                    "reason": f"'{_text}' 버튼 클릭"
                })
            _edit_added += 1

    # 5. 에디터 입력 필요 (이미 WRITE가 완료된 경우)
    if (_need_editor or _already_wrote) and _is_on_edit:
        candidates.append({
            "action": "FILL",
            "target": "",
            "content": "{faq_content}",
            "reason": "에디터에 FAQ 내용 입력"
        })

    # 6. 저장 버튼
    if _need_save:
        _save_btn = next(
            (t.strip() for t, _ in _btn_matches if any(k in t for k in ["저장", "등록", "완료"])),
            "저장"
        )
        candidates.append({
            "action": "CLICK",
            "target": _save_btn,
            "content": "",
            "reason": "저장 버튼 클릭"
        })

    # 7. 그 외 clickable 링크 (크몽 도메인만)
    for _text, _href in _btn_matches:
        if len(candidates) >= 7:
            break
        _text = _text.strip()
        _href = (_href or "").strip()
        if _href and "kmong.com" in _href and not any(c.get("target") == _href for c in candidates):
            candidates.append({
                "action": "NAVIGATE",
                "target": _href,
                "content": "",
                "reason": f"'{_text}' 이동"
            })

    # 항상 마지막: 사용자 확인 요청
    candidates.append({
        "action": "ASK_USER",
        "target": "",
        "content": "",
        "reason": "현재 상태 직접 확인 필요"
    })

    # 최대 7개
    candidates = candidates[:7]

    # ── 번호 선택 프롬프트 ────────────────────────────────────────────────
    _cand_lines = "\n".join(
        f"{i+1}. [{c['action']}] {c.get('target') or c.get('reason','')}"
        for i, c in enumerate(candidates)
    )

    _status_parts = []
    if _cur_url:
        _status_parts.append(f"현재 URL: {_cur_url}")
    if _pending_text:
        _status_parts.append(f"미달성: {_pending_text[:100]}")
    if _hist_text and _hist_text != "없음 (첫 step)":
        _status_parts.append(f"최근 액션: {_hist_text[-80:]}")
    _status = "\n".join(_status_parts) or "상태 불명"

    number_prompt = (
        f"작업: {_task}\n"
        f"상태:\n{_status}\n\n"
        f"다음 중 가장 적절한 행동 번호 하나만 답해. 숫자만.\n"
        f"{_cand_lines}\n\n"
        f"답:"
    )

    # ── LLM 호출 (1~2토큰) ───────────────────────────────────────────────
    chosen_idx = None
    for attempt in range(max_retries + 1):
        try:
            raw = await get_llm_response_async(number_prompt, max_tokens=5)
            raw = (raw or "").strip()
            # 서버 오류 응답 감지 → 재시도하지 않고 즉시 fallback
            if raw.startswith("[서버 오류") or raw.startswith("[크레딧") or raw.startswith("[Pro"):
                print(f"  [ReactAction] 서버 오류 감지 — fallback으로 전환: {raw[:60]}")
                break
            _num_m = _re.search(r'[1-7]', raw)
            if _num_m:
                idx = int(_num_m.group()) - 1
                if 0 <= idx < len(candidates):
                    chosen_idx = idx
                    break
            if attempt < max_retries:
                print(f"  [ReactAction] 번호 파싱 실패 raw={raw!r}, 재시도...")
        except Exception as e:
            print(f"  [ReactAction] 호출 실패 (시도 {attempt+1}): {e}")

    if chosen_idx is not None:
        c = candidates[chosen_idx]
        print(f"  [ReactAction] 선택={chosen_idx+1} → [{c['action']}] {(c.get('target') or c.get('reason',''))[:50]!r}")
        return {
            "action":  c["action"],
            "target":  c.get("target", ""),
            "content": c.get("content", ""),
            "reason":  c.get("reason", ""),
        }

    # ── 완전 실패 fallback ────────────────────────────────────────────────
    print("  [ReactAction] 번호 선택 실패 — 자연어 fallback")
    raw_nl = await get_llm_response_async(number_prompt, max_tokens=30)
    result = _parse_react_action_from_text(raw_nl or "")
    if result:
        return result

    if _is_url_unknown or not _is_on_kmong:
        return {"action": "NAVIGATE", "target": "https://kmong.com",
                "content": "", "reason": "fallback: 크몽 전문가 센터"}

    print("  [ReactAction] 모든 시도 실패 — None 반환")
    return None


    # ── 1. 후보 액션 목록 자동 생성 ─────────────────────────────────
    # prompt에서 URL, 셀렉터, 현재 상태 파싱
    import re as _re

    # URL 추출 (WHERE 섹션 또는 현재 URL)
    _url_matches = _re.findall(r'https?://[^\s\'">\]\)]+', prompt)
    _candidate_urls = list(dict.fromkeys(_url_matches))[:3]  # 중복 제거, 최대 3개

    # 클릭 가능 요소에서 href 추출
    _href_matches = _re.findall(r'href=([^\s|]{3,60})', prompt)
    _candidate_hrefs = [h.strip() for h in _href_matches if h.startswith('http')][:2]

    # 현재 URL 파악
    _cur_url_m = _re.search(r'URL:\s*(https?://[^\s\n]+)', prompt)
    _cur_url = _cur_url_m.group(1) if _cur_url_m else ""
    _is_on_kmong = "kmong.com" in _cur_url

    # 후보 목록 구성 (상황에 맞게)
    candidates: list[dict] = []

    if not _is_on_kmong:
        # 크몽 밖 → 크몽 이동이 1번
        candidates.append({"action": "NAVIGATE", "target": "https://kmong.com", "reason": "크몽 메인으로 이동"})

    # 크몽 전문가 센터 (상세페이지 편집)
    candidates.append({"action": "NAVIGATE", "target": "https://kmong.com", "reason": "전문가 센터로 이동"})

    # clickable 요소에서 버튼/링크 후보
    _btn_matches = _re.findall(r'(?:button|a)\s*\|\s*([^\n|]{2,40})\s*\|?\s*(https?://[^\s\n|]{5,60})?', prompt)
    for _text, _href in _btn_matches[:3]:
        _text = _text.strip()
        _href = _href.strip() if _href else ""
        if _href and "kmong.com" in _href:
            candidates.append({"action": "NAVIGATE", "target": _href, "reason": f"'{_text}' 링크로 이동"})
        elif _text:
            candidates.append({"action": "CLICK", "target": _text[:40], "reason": f"'{_text}' 클릭"})

    # 항상 마지막에 확인 요청 옵션
    candidates.append({"action": "ASK_USER", "target": "", "reason": "현재 상태 확인 필요"})

    # 최대 6개
    candidates = candidates[:6]

    # ── 2. 번호 선택 프롬프트 구성 ─────────────────────────────────
    _cand_lines = "\n".join(
        f"{i+1}. [{c['action']}] {c['target'] or c['reason']}"
        for i, c in enumerate(candidates)
    )

    # 현재 상태 요약 (prompt에서 핵심만 추출)
    _status_lines = []
    for _key in ["URL:", "HTML 크기:", "아직 미달성"]:
        _m = _re.search(rf'{_key}.*', prompt)
        if _m:
            _status_lines.append(_m.group()[:80])
    _status = "\n".join(_status_lines) or "상태 불명"

    _task_m = _re.search(r'\[실행할 작업\]\n(.+)', prompt)
    _task = _task_m.group(1)[:80] if _task_m else "작업 불명"

    number_prompt = (
        f"작업: {_task}\n"
        f"현재 상태:\n{_status}\n\n"
        f"다음 중 가장 적절한 행동 번호를 하나만 답해. 숫자만.\n"
        f"{_cand_lines}\n\n"
        f"답:"
    )

    # ── 3. LLM 호출 (1~2토큰으로 충분) ────────────────────────────
    chosen_idx = None
    for attempt in range(max_retries + 1):
        try:
            raw = await get_llm_response_async(number_prompt, max_tokens=5)
            raw = (raw or "").strip()
            # 숫자 추출
            _num_m = _re.search(r'[1-6]', raw)
            if _num_m:
                chosen_idx = int(_num_m.group()) - 1
                if 0 <= chosen_idx < len(candidates):
                    break
                chosen_idx = None
            if attempt < max_retries:
                print(f"  [ReactAction] 번호 파싱 실패 raw={raw!r}, 재시도...")
        except Exception as e:
            print(f"  [ReactAction] LLM 호출 실패 (시도 {attempt+1}): {e}")

    # ── 4. 선택된 번호 → action dict ───────────────────────────────
    if chosen_idx is not None:
        c = candidates[chosen_idx]
        print(f"  [ReactAction] 선택={chosen_idx+1} → [{c['action']}] {c['target'][:50]!r}")
        return {
            "action": c["action"],
            "target": c["target"],
            "content": "",
            "reason": c["reason"],
        }

    # ── 5. 완전 실패 fallback: 자연어 추론 ─────────────────────────
    print("  [ReactAction] 번호 선택 실패 — 자연어 fallback")
    raw_nl = await get_llm_response_async(number_prompt, max_tokens=30)
    result = _parse_react_action_from_text(raw_nl or "")
    if result:
        return result

    # 최후 fallback: 크몽 메인으로 NAVIGATE
    if not _is_on_kmong:
        return {"action": "NAVIGATE", "target": "https://kmong.com", "content": "", "reason": "fallback: 크몽 메인으로"}

    print("  [ReactAction] 모든 시도 실패 — None 반환")
    return None


# ── LLM 동시 호출 제한 (Railway 502 방지) ──────────────────────────────────
# asyncio.Semaphore는 실행 중인 이벤트 루프에 바인딩되어야 함
# → 모듈 임포트 시점이 아닌 첫 호출 시점에 생성 (지연 초기화)
_LLM_SEMAPHORE: asyncio.Semaphore | None = None

def _get_llm_semaphore() -> asyncio.Semaphore:
    # [Fix] 동시 호출 2 → 4로 완화. 자율 실행 Think + 다른 경로(채팅/DC) 동시 점유로
    # 세마포어 대기 블록이 발생하던 문제 해소. 서버는 requests.post timeout=20으로 보호됨.
    global _LLM_SEMAPHORE
    if _LLM_SEMAPHORE is None:
        _LLM_SEMAPHORE = asyncio.Semaphore(4)
    return _LLM_SEMAPHORE


def invalidate_response_cache(
    prompt: str,
    response_mime_type: Optional[str] = None,
    image_input: Optional[bytes] = None,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
) -> bool:
    """[Phase A2 2026-04-26] get_llm_response_async 와 동일 인자로 캐시 항목 무효화.

    호출자가 응답 받은 후 quality 검증에 실패한 경우 호출. 다음 같은 prompt 호출이
    캐시 hit 대신 실제 LLM 으로 흘러가 다른 응답 받을 기회 확보.
    Why: schema 측정 필드 충족률 33% 같은 약한 응답이 캐시에 박혀 두 번째 chat
    재진입 시에도 같은 약한 응답으로 굳어버리는 패턴 차단 (2026-04-26 로그).
    이미지 입력 호출은 애초에 캐시되지 않으므로 항상 False.

    Returns: 실제로 무효화된 항목이 있었는지.
    """
    if image_input:
        return False
    try:
        from eidos_llm_cache import get_cache as _get_cache, make_key as _make_key
    except Exception:
        return False
    settings = _get_settings()
    effective_max_tokens = max_tokens or settings.get("max_tokens", 8192)
    _system = system_prompt if system_prompt is not None else EIDOS_SYSTEM_PROMPT
    try:
        key = _make_key(
            prompt=prompt,
            system_prompt=_system or "",
            max_tokens=int(effective_max_tokens or 0),
            response_mime_type=str(response_mime_type or ""),
            model="gemini-2.5-flash",
        )
        return bool(_get_cache().invalidate(key))
    except Exception as _ie:
        print(f"  [LLMCache] invalidate 실패 (무시): {_ie}")
        return False


async def _call_gemini_local_text(
    prompt: str,
    system_prompt: Optional[str],
    max_tokens: int,
    response_mime_type: Optional[str],
    timeout: float,
) -> Tuple[str, int, int]:
    """
    [로컬 직접 호출] eidos_settings.json 의 'gemini_api_key' 로 Gemini SDK 직접 호출.

    - 성공: (content_text, prompt_tokens, candidate_tokens) 반환.
    - 키 없음 / SDK 미가용: ValueError("local_unavailable") raise → 호출자가 서버로 폴백.
    - API 에러 / 빈 응답: 그대로 raise → 호출자가 서버로 폴백.

    크레딧 차감 안 함 (사용자 본인 키). 서버 throttle 도 안 거침
    (로컬 키는 별도 한도 — Google AI Studio 무료 tier 15RPM/1500RPD).
    """
    if not _genai_available:
        raise ValueError("local_unavailable: google.generativeai 미가용")
    settings = _get_settings()
    api_key = settings.get("gemini_api_key", "").strip()
    if not api_key:
        raise ValueError("local_unavailable: gemini_api_key 미설정")

    # 모듈 레벨에서 import 한 genai 사용 (테스트 monkey-patch 가능).
    _gv = genai

    _gv.configure(api_key=api_key)

    # Gemini SDK 의 system_instruction 으로 시스템 프롬프트 전달 (메시지에 prepend 안 함).
    _model_kwargs: Dict[str, Any] = {}
    if system_prompt:
        _model_kwargs["system_instruction"] = system_prompt

    # [2026-05-06 fix] safety filter BLOCK_NONE — "Gemini 빈 응답 (safety filter 가능성)"
    # 에러 차단. 사용자 본인 키 (Google AI Studio) 라 safety 책임은 사용자에게 있음.
    # google-generativeai 버전 호환 — try/except 로 감쌈.
    try:
        from google.generativeai.types import HarmCategory as _HC, HarmBlockThreshold as _HBT
        _model_kwargs["safety_settings"] = {
            _HC.HARM_CATEGORY_HARASSMENT:        _HBT.BLOCK_NONE,
            _HC.HARM_CATEGORY_HATE_SPEECH:       _HBT.BLOCK_NONE,
            _HC.HARM_CATEGORY_SEXUALLY_EXPLICIT: _HBT.BLOCK_NONE,
            _HC.HARM_CATEGORY_DANGEROUS_CONTENT: _HBT.BLOCK_NONE,
        }
    except Exception as _e_safety:
        # 구버전 SDK — 문자열 기반 fallback
        try:
            _model_kwargs["safety_settings"] = [
                {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
        except Exception:
            pass  # safety_settings 불가능하면 default

    _gen_cfg: Dict[str, Any] = {"max_output_tokens": int(max_tokens)}
    if response_mime_type:
        _gen_cfg["response_mime_type"] = response_mime_type

    model = _gv.GenerativeModel("gemini-2.5-flash", **_model_kwargs)

    def _sync_call():
        return model.generate_content(prompt, generation_config=_gen_cfg)

    resp = await asyncio.wait_for(asyncio.to_thread(_sync_call), timeout=timeout)

    # 본문 추출 — resp.text 가 first candidate 의 합본
    content = ""
    try:
        content = (resp.text or "").strip()
    except Exception:
        # 일부 케이스(safety block 등) 에서 .text 가 raise. candidates 직접 검사.
        try:
            cands = getattr(resp, "candidates", []) or []
            if cands:
                parts = getattr(cands[0].content, "parts", []) or []
                content = "".join(getattr(p, "text", "") for p in parts).strip()
        except Exception:
            pass

    if not content:
        # [2026-05-06 fix] 빈 응답 진짜 원인 진단 — finish_reason / safety_ratings /
        # prompt_feedback.block_reason 추출. "safety filter 가능성" 추측 대신 실제 메시지.
        # 사용자 의심 (quota 소진 / 블랙리스트) 디버깅에 결정적.
        _diag = []
        try:
            cands = getattr(resp, "candidates", []) or []
            if cands:
                _c0 = cands[0]
                _fr = getattr(_c0, "finish_reason", None)
                if _fr is not None:
                    # FinishReason: STOP/MAX_TOKENS/SAFETY/RECITATION/OTHER (Gemini SDK enum)
                    _fr_name = getattr(_fr, "name", str(_fr))
                    _diag.append(f"finish_reason={_fr_name}")
                _sr = getattr(_c0, "safety_ratings", None) or []
                _blocked = [str(getattr(r, "category", "?"))
                            for r in _sr if getattr(r, "blocked", False)]
                if _blocked:
                    _diag.append(f"blocked_categories={_blocked}")
            _pf = getattr(resp, "prompt_feedback", None)
            if _pf:
                _br = getattr(_pf, "block_reason", None)
                if _br is not None:
                    _br_name = getattr(_br, "name", str(_br))
                    _diag.append(f"prompt_block_reason={_br_name}")
        except Exception as _e_diag:
            _diag.append(f"diag_err={type(_e_diag).__name__}")
        _msg = "; ".join(_diag) if _diag else "no_diag (raw empty response)"
        raise RuntimeError(f"local_empty: Gemini 빈 응답 ({_msg})")

    # 토큰 사용량 — usage_metadata 가 있으면 사용
    p_tok = c_tok = 0
    try:
        meta = getattr(resp, "usage_metadata", None)
        if meta:
            p_tok = int(getattr(meta, "prompt_token_count", 0) or 0)
            c_tok = int(getattr(meta, "candidates_token_count", 0) or 0)
    except Exception:
        pass

    # [이슈 #1 진단 로깅 2026-05-19] 응답 중간 잘림(비결정적) 원인 확정용.
    # 성공(비어있지 않은) 응답에도 finish_reason + completion_tokens 를 무조건 남긴다.
    # finish_reason=MAX_TOKENS 면 max_tokens 한계가 잘림 원인이라는 확정 증거.
    # (기존 진단은 빈 응답일 때만 동작 → 중간 잘림은 content 가 비어있지 않아 누락됐음)
    try:
        _fr_name = "?"
        try:
            _cands = getattr(resp, "candidates", []) or []
            if _cands:
                _fr = getattr(_cands[0], "finish_reason", None)
                if _fr is not None:
                    _fr_name = getattr(_fr, "name", str(_fr))
        except Exception:
            pass
        _truncated = _fr_name in ("MAX_TOKENS", "2")  # SDK enum name 또는 raw 값
        _tag = "⚠️ [LLM-TRUNCATED]" if _truncated else "🪶 [LLM-FINISH]"
        print(
            f"{_tag} finish_reason={_fr_name} "
            f"completion_tokens={c_tok} prompt_tokens={p_tok} "
            f"max_tokens_limit={int(max_tokens)} content_len={len(content)}"
        )
        if _truncated:
            print(
                f"   ↳ 응답이 max_output_tokens({int(max_tokens)}) 한계에서 잘렸습니다. "
                f"max_tokens 상향 또는 분할 응답이 필요합니다."
            )
    except Exception as _e_fr_log:
        print(f"  ⚠️ [LLM-FINISH] finish_reason 로깅 실패 (무시): {_e_fr_log}")

    return content, p_tok, c_tok


async def get_llm_response_async(
    prompt: str,
    response_mime_type: Optional[str] = None,
    image_input: Optional[bytes] = None,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
    timeout: Optional[int] = None,
    use_cache: bool = True,
) -> str:
    """
    [하이브리드] 로컬 Gemini 키 있으면 직접 호출, 실패/미설정 시 서버 프록시 폴백.
    - 로컬 path: 크레딧 차감 X, 서버 throttle X, 사용자 본인 한도(15RPM 무료) 사용
    - 서버 path: 크레딧 차감, 재시도+throttle, 서버 측 Gemini 키 한도 공유

    system_prompt를 직접 전달하면 EIDOS_SYSTEM_PROMPT 대신 사용됨.
    timeout: HTTP/SDK 요청 타임아웃(초). 기본 20s.
    """
    settings = _get_settings()
    effective_max_tokens = max_tokens or settings.get("max_tokens", 8192)
    effective_timeout = timeout if timeout is not None else 20

    # action 결정 — 호출 컨텍스트에 따라 자동 분류
    # response_mime_type이 JSON이면 코드/구조화 작업 → code_edit
    action = "code_edit" if response_mime_type == "application/json" else "chat"

    # system_prompt 오버라이드: 직접 전달된 경우 우선 적용
    _system = system_prompt if system_prompt is not None else EIDOS_SYSTEM_PROMPT

    messages = [{"role": "user", "content": prompt}]

    # [Fix 2026-04-29] 이미지 입력은 vision 전용 함수로 라우팅.
    # 이전 회귀: 여기서 이미지 데이터를 폐기하고 텍스트 placeholder 로 바꿔치기 → 모델은
    # 이미지를 보지 못한 채 추측 응답. analyze_image 같은 호출자가 image_input 을 넘겨도
    # 결과적으로 vision 분석이 전혀 일어나지 않았던 원인.
    # 해결: image_input 이 있으면 곧바로 get_llm_response_vision_async 로 위임.
    # (vision 함수는 자체적으로 gemini_api_key 폴백까지 처리함.)
    if image_input:
        return await get_llm_response_vision_async(
            prompt=prompt,
            image_bytes=image_input,
            filename="",
        )

    # [Phase A 2026-04-24] LLM 응답 캐시 lookup — 같은 prompt+system+max_tokens+mime
    # 이면 디스크/메모리 캐시 즉시 반환. 자율실행 ReAct 가 동일 컨텍스트 반복 호출하는
    # 패턴이 흔해 30~50% 호출 감축 기대 (실측 후 확정). 이미지 입력 호출은 캐시 X.
    _cache_key: str = ""
    if use_cache and not image_input:
        try:
            from eidos_llm_cache import get_cache as _get_cache, make_key as _make_key
            _cache_key = _make_key(
                prompt=messages[0]["content"],
                system_prompt=_system or "",
                max_tokens=int(effective_max_tokens or 0),
                response_mime_type=str(response_mime_type or ""),
                model="gemini-2.5-flash",
            )
            _hit = _get_cache().lookup(_cache_key)
            if _hit is not None:
                # 히트 시 토큰 추적은 0으로 (호출 자체 안 함). 이미 차감 안 됨.
                print(f"  💾 [LLMCache] hit key={_cache_key[:8]}… (len={len(_hit)})")
                return _hit
        except Exception as _ce:
            print(f"  [LLMCache] lookup 실패 (무시): {_ce}")
            _cache_key = ""

    # ── [하이브리드] 로컬 Gemini 키 있으면 먼저 시도. 성공 시 서버 호출 스킵. ──
    # 로컬 실패(키 없음/네트워크/safety/429 등)는 raise → 아래 서버 path 로 자연 폴백.
    # 이미지 입력은 vision 전용 함수가 따로 처리하므로 텍스트 path 만.
    # [복원 2026-05-31] 큰 코드수정 등 1차 timeout 부족 → 죽은 서버까지 폴백 → 502 실패하던 문제.
    #   로컬이 "사용 가능"한데 타임아웃이면, 서버로 새기 전에 넉넉한 timeout 으로 로컬 1회 재시도.
    #   (키 없음/검증오류/기타 예외는 기존대로 즉시 서버 폴백 — blast radius 최소화.
    #    긴 timeout(≥120s) 호출자는 재시도 미추가 → 자율루프 지연 방지.)
    if not image_input:
        _local_timeouts = [float(effective_timeout)]
        _local_retry_to = min(max(float(effective_timeout), 120.0), 180.0)
        if _local_retry_to > float(effective_timeout):
            _local_timeouts.append(_local_retry_to)
        for _li, _to in enumerate(_local_timeouts):
            try:
                _local_text, _p_tok, _c_tok = await _call_gemini_local_text(
                    prompt=messages[0]["content"],
                    system_prompt=_system,
                    max_tokens=int(effective_max_tokens),
                    response_mime_type=response_mime_type,
                    timeout=_to,
                )
                if _li > 0:
                    print(f"  💎 [LLM/Local] 타임아웃 후 재시도 성공 (timeout={_to:.0f}s)")
                print(f"  💎 [LLM/Local] 직접 호출 성공 ({len(_local_text)}자, in={_p_tok} out={_c_tok})")
                # 토큰 추적 (서버 path 와 동일 형식)
                try:
                    cost_monitor.track_usage("gemini-2.5-flash", _p_tok, _c_tok)
                except Exception:
                    pass
                # 캐시 저장
                if _cache_key:
                    try:
                        from eidos_llm_cache import get_cache as _get_cache_st
                        _get_cache_st().store(_cache_key, _local_text, model="gemini-2.5-flash")
                    except Exception:
                        pass
                return _local_text
            except ValueError as _lv:
                # local_unavailable — 키 없거나 SDK 미가용. 조용히 서버 폴백.
                if "local_unavailable" not in str(_lv):
                    print(f"  ⚠️ [LLM/Local] 검증 오류 — 서버 폴백: {_lv}")
                break
            except asyncio.TimeoutError:
                if _li + 1 < len(_local_timeouts):
                    print(f"  ⚠️ [LLM/Local] 타임아웃({_to:.0f}s) — 로컬 재시도({_local_timeouts[_li + 1]:.0f}s)")
                    continue
                print(f"  ⚠️ [LLM/Local] 타임아웃 — 서버 폴백")
            except Exception as _le:
                # 429/safety/네트워크 오류 등 — 서버 폴백
                print(f"  ⚠️ [LLM/Local] 호출 실패 — 서버 폴백: {type(_le).__name__}: {str(_le)[:100]}")
                break
    # ──────────────────────────────────────────────────────────────

    # [Phase B 2026-04-24] 사전 throttle — 캐시 미스인 경우만 실제 호출 직전 acquire.
    # RPM/RPD 한도 임박이면 sleep, RPD 초과면 LLMQuotaBlockedError. 호출자는 안내 메시지로 폴백.
    try:
        from eidos_llm_throttle import get_throttle as _get_throttle, LLMQuotaBlockedError as _LLMBlocked
        try:
            await _get_throttle().acquire()
        except _LLMBlocked as _be:
            # [Phase 4-4 2026-04-25] kind 분기: rpd(진짜 일일) vs rpm_retry(분당 cap 재시도 실패).
            # rpd=76/300 인데 "일일 한도" 라벨로 ReAct Think 가 오해하던 버그 정정.
            _be_kind = getattr(_be, "kind", "unknown")
            if _be_kind == "rpd":
                _label_short = "일일한도"
                _label_log   = "일일 한도 도달"
            elif _be_kind == "rpm_retry":
                _label_short = "분당한도"
                _label_log   = "분당 한도 cap 재시도 실패 (잠시 후 재시도)"
            else:
                _label_short = "한도"
                _label_log   = "한도 — 호출 거부"
            print(f"  🛑 [LLMThrottle] {_label_log}: {_be}")
            return f"[서버 혼잡 {_label_short}: {str(_be)[:80]}]"
    except ImportError:
        pass
    except Exception as _te:
        print(f"  [LLMThrottle] acquire 실패 (무시, 그대로 호출): {_te}")

    async with _get_llm_semaphore():   # 동시 호출 최대 2개 제한
        try:
            server_url = _get_server_url()
            device_id = _get_device_id()

            # 429(Resource exhausted)/502/503/504 는 순간적 쿼터/게이트웨이 이슈 → 지수 backoff 재시도.
            # 에러 메시지도 짧게 잘라 로그 오염 최소화.
            resp = None
            _RETRY_STATUS = {429, 502, 503, 504}
            # [2026-05-06 fix] 3 → 5 재시도 인상. 사용자 답답함 — 502 가 자주 발생.
            # backoff: 1.5 → 3 → 6 → 12 → 24 (총 ~46s 추가). 첫 호출 timeout 과 합산해도
            # builder timeout 300s 안에 들어옴.
            _MAX_TRIES = 5
            for _attempt in range(_MAX_TRIES):
                resp = await asyncio.to_thread(
                    lambda: requests.post(
                        f"{server_url}/api/chat",
                        json={
                            "device_id": device_id,
                            "action": action,
                            "messages": messages,
                            "model": "gemini-2.5-flash",
                            "max_tokens": effective_max_tokens,
                            "system_prompt": _system,
                        },
                        timeout=effective_timeout,
                    )
                )
                # 본문에 Gemini 429/Resource exhausted 가 섞인 502 도 재시도 대상.
                _body_snippet = (resp.text or "")[:300]
                _rate_limited = (
                    resp.status_code in _RETRY_STATUS
                    or ("429" in _body_snippet and "Resource exhausted" in _body_snippet)
                )
                if not _rate_limited or _attempt == _MAX_TRIES - 1:
                    break
                _backoff = 1.5 * (2 ** _attempt)   # 1.5s → 3s → 6s
                print(f"⚠️ [LLM] {resp.status_code} 재시도 {_attempt+1}/{_MAX_TRIES-1} "
                      f"(backoff {_backoff:.1f}s)")
                await asyncio.sleep(_backoff)

            if resp.status_code == 402:
                return "[크레딧 부족: 설정 > API 키 탭에서 라이선스 키를 충전하세요]"
            if resp.status_code == 403:
                return "[Pro 기능: 라이선스 키가 필요합니다]"
            if resp.status_code != 200:
                # [Phase B 2026-04-24] quota/rate-limit 에러 → throttle 학습 (cooldown 갱신).
                # 다음 호출들이 추가 sleep 받아 burst 차단. 429 / 502 본문에 Resource exhausted 케이스.
                if resp.status_code in _RETRY_STATUS:
                    try:
                        from eidos_llm_throttle import get_throttle as _get_throttle
                        _get_throttle().on_quota_error(source=f"http_{resp.status_code}")
                    except Exception:
                        pass
                # 429/502 는 이미 재시도 소진 후이므로 짧은 안내 메시지로 대체.
                _short_body = (resp.text or "")[:80].replace("\n", " ")
                if resp.status_code in _RETRY_STATUS:
                    return f"[서버 혼잡 {resp.status_code}: 잠시 후 재시도하세요]"
                return f"[서버 오류 {resp.status_code}: {_short_body}]"

            data = resp.json()
            content = data.get("content", "")

            # content가 비어있으면 다른 필드 fallback 시도
            if not content:
                content = (
                    data.get("text", "") or
                    data.get("response", "") or
                    data.get("result", "") or
                    data.get("message", "") or
                    ""
                )
                if not content:
                    print(f"⚠️ [LLM] 서버 응답 content 비어있음. 응답 키: {list(data.keys())}")

            # 토큰 추적 (비용 모니터)
            try:
                cost_monitor.track_usage(
                    "gemini-2.5-flash",
                    data.get("tokens_in", 0),
                    data.get("tokens_out", 0),
                )
            except Exception:
                pass

            # [Phase A 2026-04-24] 정상 응답이면 캐시 저장. 에러 응답/빈 응답은
            # eidos_llm_cache.is_cacheable_response 가 내부에서 거른다.
            if _cache_key and content:
                try:
                    from eidos_llm_cache import get_cache as _get_cache
                    _get_cache().store(_cache_key, content, model="gemini-2.5-flash")
                except Exception as _se:
                    print(f"  [LLMCache] store 실패 (무시): {_se}")

            return content

        except requests.exceptions.ConnectionError:
            return "[서버 연결 실패: 인터넷 연결 또는 서버 상태를 확인하세요]"
        except requests.exceptions.Timeout:
            return "[서버 응답 시간 초과: 잠시 후 다시 시도하세요]"
        except Exception as e:
            return f"[LLM 호출 오류: {e}]"

async def get_llm_response_stream_async(
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
):
    """
    [스트리밍 버전] EIDOS 서버에 stream=True로 요청하여
    텍스트 청크를 async generator로 yield합니다.

    서버가 SSE(text/event-stream) 또는 일반 chunked transfer를 지원하면
    실시간 청크를 흘려주고, 미지원이면 응답 전체를 한 번에 yield합니다.
    """
    settings = _get_settings()
    effective_max_tokens = max_tokens or settings.get("max_tokens", 8192)
    _system = system_prompt or EIDOS_SYSTEM_PROMPT

    messages = [{"role": "user", "content": prompt}]

    server_url = _get_server_url()
    device_id = _get_device_id()

    payload = {
        "device_id": device_id,
        "action": "chat",
        "messages": messages,
        "model": "gemini-2.5-flash",
        "max_tokens": effective_max_tokens,
        "system_prompt": _system,
        "stream": True,   # 서버가 지원하면 스트리밍 활성화
    }

    try:
        # requests stream=True: 청크 단위로 수신
        def _do_request():
            return requests.post(
                f"{server_url}/api/chat",
                json=payload,
                timeout=120,
                stream=True,
            )

        resp = await asyncio.to_thread(_do_request)

        if resp.status_code == 402:
            yield "[크레딧 부족: 설정 > API 키 탭에서 라이선스 키를 충전하세요]"
            return
        if resp.status_code != 200:
            yield f"[서버 오류 {resp.status_code}]"
            return

        content_type = resp.headers.get("Content-Type", "")

        if "text/event-stream" in content_type:
            # ── SSE 방식: "data: {...}\n\n" 형태 ──────────────────────
            full_text = ""
            buf = ""
            for raw_chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                if not raw_chunk:
                    continue
                buf += raw_chunk
                while "\n\n" in buf:
                    line, buf = buf.split("\n\n", 1)
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        return
                    try:
                        obj = json.loads(data_str)
                        # OpenAI-compatible delta 또는 단순 content 필드
                        chunk_text = (
                            obj.get("content")
                            or obj.get("delta", {}).get("content")
                            or obj.get("text")
                            or ""
                        )
                        if chunk_text:
                            full_text += chunk_text
                            yield chunk_text
                            await asyncio.sleep(0)
                        # [이슈 #1] 스트림 종료 메타에 finish_reason 이 실리면 로깅.
                        # 스트리밍 race / stop token 오생성 vs MAX_TOKENS 판별 단서.
                        _fr = (
                            obj.get("finish_reason")
                            or obj.get("stop_reason")
                            or (obj.get("choices", [{}])[0].get("finish_reason")
                                if isinstance(obj.get("choices"), list) and obj.get("choices") else None)
                        )
                        if _fr:
                            _ct = (obj.get("completion_tokens")
                                   or obj.get("usage", {}).get("completion_tokens")
                                   or "?")
                            _tag = ("⚠️ [LLM-STREAM-TRUNCATED]"
                                    if str(_fr).upper() in ("MAX_TOKENS", "LENGTH", "2")
                                    else "🪶 [LLM-STREAM-FINISH]")
                            print(f"{_tag} finish_reason={_fr} "
                                  f"completion_tokens={_ct} streamed_len={len(full_text)}")
                    except Exception:
                        pass
        elif "application/json" in content_type or "application/x-ndjson" in content_type:
            # ── NDJSON / 줄 단위 JSON 스트리밍 ──────────────────────────
            buf = b""
            for raw_chunk in resp.iter_content(chunk_size=64):
                if not raw_chunk:
                    continue
                buf += raw_chunk
                while b"\n" in buf:
                    line_bytes, buf = buf.split(b"\n", 1)
                    line_str = line_bytes.decode("utf-8", errors="replace").strip()
                    if not line_str:
                        continue
                    try:
                        obj = json.loads(line_str)
                        chunk_text = (
                            obj.get("content")
                            or obj.get("text")
                            or ""
                        )
                        if chunk_text:
                            yield chunk_text
                            await asyncio.sleep(0)
                    except Exception:
                        pass
        else:
            # ── 서버가 스트리밍 미지원: 전체 응답을 받아 글자 단위로 흘림 ──
            # 최소한 응답을 빨리 받아서 즉시 표시
            raw = b""
            for chunk in resp.iter_content(chunk_size=512):
                if chunk:
                    raw += chunk
            try:
                data = json.loads(raw.decode("utf-8", errors="replace"))
                content = (
                    data.get("content")
                    or data.get("text")
                    or data.get("response")
                    or ""
                )
                # 비용 추적
                try:
                    cost_monitor.track_usage(
                        "gemini-2.5-flash",
                        data.get("tokens_in", 0),
                        data.get("tokens_out", 0),
                    )
                except Exception:
                    pass
            except Exception:
                content = raw.decode("utf-8", errors="replace")

            if not content:
                return

            # 글자 단위로 흘려서 스트리밍 효과 (실제 스트리밍 대신)
            CHUNK = 15
            for i in range(0, len(content), CHUNK):
                yield content[i:i + CHUNK]
                await asyncio.sleep(0.008)

    except requests.exceptions.ConnectionError:
        yield "[서버 연결 실패]"
    except requests.exceptions.Timeout:
        yield "[서버 응답 시간 초과]"
    except Exception as e:
        yield f"[스트리밍 오류: {e}]"


async def classify_input_async(user_input: str) -> str:
    """
    [v15.9 Fix] 사용자 입력의 의도를 'TASK', 'MEMORY', 'CHAT' 3가지로 분류합니다.
    (일상 대화가 TASK로 오분류되는 것을 방지하기 위해 기준을 강화했습니다.)
    """
    if model is None:
        print("❌ Gemini 모델이 로드되지 않았습니다.")
        return "CHAT" # Fallback

    prompt = f"""
    당신은 EIDOS의 의도 분류기입니다. 사용자 입력의 '핵심 의도'를 다음 세 가지 중 하나로 분류하세요.

    [분류 기준]
    1.  **TASK (작업 명령)**:
        -   EIDOS가 **도구(웹 검색, 파일 쓰기, 코드 실행, 일정 관리 등)를 사용해야만 하는** 명령.
        -   결과물(파일, 보고서, 정리된 데이터)을 생성해야 하는 경우.
        -   (예: "검색해줘", "작성해줘", "요약해줘", "일정 잡아줘", "분석해줘", "계산해")
        -   (주의) 단순히 "알려줘"라고 했더라도, 그 내용이 웹 검색이나 파일 확인이 필요하면 TASK입니다.

    2.  **MEMORY (기억/분석)**:
        -   새로운 사실을 알려주거나, 장기적으로 기억해야 할 정보.
        -   (예: "내 생일은 5월 5일이야", "이 프로젝트는 중요해")

    3.  **CHAT (단순 대화)**:
        -   **사용자의 안부, 기분, 상태를 묻는 질문.** (가장 중요)
        -   EIDOS의 생각이나 의견을 묻는 철학적/일상적 질문.
        -   도구 사용 없이 바로 대답할 수 있는 내용.
        -   (예: "오늘 어때?", "기분 괜찮아?", "안녕", "너는 누구니?", "심심해")
        -   (예: "고마워", "잘했어", "그렇구나" 같은 리액션)

    [사용자 입력]
    "{user_input}"

    [지시]
    위 기준에 따라 'TASK', 'MEMORY', 'CHAT' 중 하나만 선택하여 대문자로 응답하세요.
    특히 **"오늘 어때?", "뭐해?" 같은 모호한 질문은 반드시 'CHAT'**으로 분류하세요.
    """
    try:
        response_text = await get_llm_response_async(prompt)
        classification = response_text.strip().upper()
        
        if "TASK" in classification:
            return "TASK"
        elif "MEMORY" in classification:
            return "MEMORY"
        else:
            return "CHAT"
    except Exception as e:
        print(f"❌ [Classifier LLM] 분류 오류: {e}")
        return "CHAT" # 오류 시 안전하게 CHAT 처리

# [!!! v15.10 병목 1 해결: 통합 함수 추가 !!!]
# (이 함수는 기존 generate_reasoning_log_async와 generate_natural_response_async를 대체합니다)

async def generate_reasoning_and_response_async(
    user_input: str,
    prev_emotion: np.ndarray, 
    new_emotion: np.ndarray,
    policy: str, 
    abduction_ids: list[str], 
    chat_history: List[str],
    explanation: str = "",
    self_description: str = "나는 AI입니다.",
    core_values_str: str = "사용자 안전",
    causal_discoveries: Optional[List[Dict[str, str]]] = None,
    abstract_matches: Optional[List[str]] = None,
    alternatives: Optional[Dict[str, str]] = None,
    purity: float = 1.0,
    complex_states: Dict[str, float] = None
) -> Dict[str, str]:
    """
    [v15.10] 병목 1(LLM 호출 폭풍) 해결을 위해 '추론 로그'와 '자연어 응답' 생성을
    JSON 모드를 사용하는 단일 LLM 호출로 통합합니다.
    """
    if not model:
        return {"reasoning_log": "LLM 오류", "natural_response": "LLM 오류"}

    # --- 1. 추론 로그(Reasoning Log)용 컨텍스트 생성 (기존 함수 로직) ---
    emotion_diff = new_emotion[0] - prev_emotion[0]
    changed_emotions = []
    for i, diff in enumerate(emotion_diff):
        if abs(diff) > 0.1: 
            name = EMOTION_MAP.get(i, f"?{i}")
            changed_emotions.append(f"{name} ({prev_emotion[0][i]:.2f} → {new_emotion[0][i]:.2f})")
    emotion_log = "감정 상태에 큰 변화는 없었습니다." if not changed_emotions else "주요 감정 변화: " + ", ".join(changed_emotions)

    abduction_log = "유사한 과거 사건을 찾지 못했습니다."
    if abduction_ids: abduction_log = f"과거의 {abduction_ids} 사건들을 참고했습니다."
    
    internal_reasoning = ""
    if causal_discoveries:
        links = [f"'{d['cause']}->{d['effect']}'" for d in causal_discoveries]
        internal_reasoning += f"\n- [인과 엔진] 새로운 인과관계 발견: {', '.join(links)}"
    if abstract_matches:
        patterns = [f"'{p_name}'" for p_name in abstract_matches]
        internal_reasoning += f"\n- [추상화 엔진] 현재 상황이 과거의 {', '.join(patterns)} 패턴과 일치함."
    if not internal_reasoning:
        internal_reasoning = "\n- [내부 엔진] 특이사항 없음."

    alternatives_log = "평가된 대안 없음."
    if alternatives:
        chosen_action_prefix = policy.split('\n')[0]
        alternatives_lines = []
        for name, score in alternatives.items():
            prefix = "➡️" if chosen_action_prefix in name else "  "
            alternatives_lines.append(f"    {prefix} {name}: {score}")
        alternatives_log = "\n".join(alternatives_lines)

    # --- 2. 자연어 응답(Natural Response)용 컨텍스트 생성 (기존 함수 로직) ---
    top_emotions = []
    for i in range(EMOTION_DIM):
        val = new_emotion[0][i]
        if val > 0.5:
            name = EMOTION_MAP.get(i, f"Emotion{i}")
            top_emotions.append((name, val))
    emotion_summary = ", ".join([f"{n}: {v:.1f}" for n, v in top_emotions[:3]]) or "평온함"

    # [인간화 v2] 12차원 감정 → 세분화 말투 (generate_natural_response_async와 동일 테이블)
    _RESP_TONE: Dict[str, str] = {
        "기쁨":   "들뜨고 활기차게",    "슬픔":   "차분하고 살짝 처지게",
        "분노":   "날카롭고 퉁명스럽게", "공포":   "조심스럽고 불안하게",
        "놀람":   "놀란 듯 생동감 있게", "혐오":   "냉소적으로",
        "신뢰":   "안정적이고 편안하게", "기대":   "설레고 기대하듯이",
        "수치심": "조용하고 머뭇거리게", "자부심": "뿌듯하고 당당하게",
        "호기심": "호기심 가득하게",     "지루함": "심드렁하게",
    }
    # 지배 감정 추출
    dominant_idx = int(new_emotion[0].argmax()) if top_emotions else None
    dominant_name = EMOTION_MAP.get(dominant_idx, "평온함") if dominant_idx is not None else "평온함"
    tone = _RESP_TONE.get(dominant_name, "차분하게")

    # history 필터링 (Fe/Fi 내부 분석 텍스트 제거)
    _INTERNAL_PREFIXES = (
        "사용자의 최근한", "사용자의 입력을 분석", "사용자의 '", "사용자의 친근한",
        "CHAT_MODE를 최종", "Fe(외향 감정)", "Fi(내향 감정)",
        "alignment_score", "tone_instruction", "user_emotion",
        "[Fe]", "[Fi]", "[내부 분석]", "[분석]",
        "감지된 분위기", "모드로 인식하여", "결정했습니다",
    )
    filtered_history = [
        h for h in chat_history[-15:]
        if not any(h.strip().startswith(p) or p in h for p in _INTERNAL_PREFIXES)
    ]
    full_history_str = "\n".join(filtered_history)

    # --- 3. 통합 프롬프트 ---
    # [인간화 v2] reasoning_log(분석)와 natural_response(대화)를 역할 문맥으로 명확히 분리.
    # system_instruction에 EIDOS 캐릭터가 이미 주입되어 있으므로 역할 재정의 불필요.
    prompt = f"""다음 두 가지를 JSON으로 출력해.

[최근 대화]
{full_history_str}

[사용자 입력]
{user_input}

[내부 상태]
- 감정: {emotion_summary} / 지배 감정: {dominant_name} → 말투: {tone}
- 판단: {policy}
- 참고: {explanation if explanation else "없음"}
- 과거 연결: {abduction_log}

[작업 1 — reasoning_log]
왜 그런 판단을 내렸는지 1~2문장으로 간결하게 요약. 분석적인 문체로.

[작업 2 — natural_response]
사용자 입력에 대한 실제 대화 응답. 말투: '{tone}'.
응답 길이는 입력의 성격에 따라 결정:
- 단순 인사/리액션("응", "고마워", "ㅋㅋ" 등) → 1~2문장으로 짧게.
- 의견 요청, 업무/결과 평가, 아이디어 공유 → 3~6문장. 구체적인 근거나 생각을 풀어서 말해.
- 기술/분석 설명 요청 → 필요한 만큼 충분히. 핵심 포인트를 빠뜨리지 마.
공통 규칙:
- [내부 상태]의 판단이 단순 실행 수락이면 1~2문장으로 간단히.
- 선택지, 번호 목록, 이모티콘 절대 금지.
- 마지막 문장이 자연스럽게 끝나야 함 (말 중간에 끊기지 않게).

[작업 3 — suggested_actions]
응답 직후 사용자가 자연스럽게 이어갈 수 있는 행동 버튼 목록. 2~5개.
각 항목은 아래 JSON 객체:
  - "label": 버튼 텍스트 (한국어, 10자 이내)
  - "prompt": 버튼 클릭 시 입력창에 채워질 지시문 (사용자 말투, 1문장)
  - "type": "action" | "question" | "browser"
      * action   — EIDOS에게 바로 시킬 일
      * question — 더 알고 싶은 것
      * browser  — 내장 브라우저에서 열어야 하는 실제 웹 작업
  - "url": type이 "browser"일 때만 포함. 이동할 URL (없으면 필드 자체 생략)

browser 타입 사용 조건:
- 크몽/Gumroad 페이지 편집, SNS 포스팅, 외부 서비스 조작 등 실제 웹 접속이 필요한 경우만.
- label은 작업 이름으로 (예: "크몽 페이지 편집", "트위터 포스팅").
- url을 확실히 알 수 없으면 type을 "action"으로 대체할 것.

단순 인사/리액션이면 suggested_actions는 빈 배열 [].

[출력 JSON]
{{
  "reasoning_log": "...",
  "natural_response": "...",
  "suggested_actions": [
    {{"label": "...", "prompt": "...", "type": "action"}},
    {{"label": "...", "prompt": "...", "type": "browser", "url": "https://..."}}
  ]
}}"""
    
    try:
        # JSON 모드로 통합 호출
        response_text = await get_llm_response_async(
            prompt,
            response_mime_type="application/json"
        )

        # 마크다운 코드블록 제거 후 파싱
        clean = response_text.strip()
        if clean.startswith("```"):
            clean = re.sub(r'^```(?:json)?\s*', '', clean, flags=re.IGNORECASE)
            clean = re.sub(r'\s*```$', '', clean).strip()

        analysis_dict = json.loads(clean)

        if "reasoning_log" in analysis_dict and "natural_response" in analysis_dict:
            # suggested_actions 없는 구버전 응답도 안전하게 처리
            if "suggested_actions" not in analysis_dict:
                analysis_dict["suggested_actions"] = []
            return analysis_dict
        else:
            print(f"❌ [LLM Integrate] JSON 스키마 오류: {response_text[:100]}")
            return {"reasoning_log": "LLM 스키마 오류", "natural_response": "잠깐, 뭔가 이상한데.", "suggested_actions": []}

    except json.JSONDecodeError as e_json:
        print(f"❌ [LLM Integrate] JSON 파싱 실패: {e_json}")
        print(f"   Raw Response: {response_text[:200]}...")
        return {"reasoning_log": "LLM 파싱 오류", "natural_response": "응, 뭔가 꼬였어. 다시 말해줘.", "suggested_actions": []}
    except Exception as e:
        print(f"❌ [LLM Integrate] API 호출 중 예외 발생: {e}")
        return {"reasoning_log": f"LLM API 오류: {e}", "natural_response": "잠깐, 연결이 좀 불안정해.", "suggested_actions": []}

async def analyze_event_with_llm_async(
    text_input: str,
    image_input: Optional[bytes] = None
) -> Dict[str, Any]: # <-- [개선] 반환 타입을 str에서 Dict[str, Any]로 변경
    """ 
    [v14.7 개선] 입력을 분석하고, 결과를 '구조화된 JSON' 객체로 반환합니다.
    LLM의 JSON 모드를 활용하여 파싱 안정성을 극대화합니다.
    """

    # [개선] JSON 스키마를 프롬프트에 명시
    prompt = f"""
    EIDOS 언어-시각 처리 모듈. 입력을 분석하여 '반드시' JSON 형식으로 반환.
    
    [JSON 스키마]
    {{
      "객체": ["분석된 객체 리스트"],
      "상호작용": ["분석된 상호작용 리스트"],
      "속성": ["분석된 속성 리스트"],
      "사용자 상태": {{ "key": "value" }},
      "목표": ["추출된 목표 (하나만)"],
      "믿음": [["주체", "사실"]],
      "2차 믿음": [["주체A", "주체B", "사실"]]
    }}

    [규칙]
    - (중요!) 이미지 핵심 내용을 반드시 텍스트로 변환하여 "객체"/"속성"에 포함.
    - 해당 사항 없으면 빈 리스트 [] 또는 빈 객체 {{}} 사용.
    - "목표"는 사용자가 명시적으로 EIDOS에게 새 장기 목표를 설정할 때만 추출.
    - (중요!) 응답은 '반드시' JSON 형식이어야 하며, 다른 텍스트(예: "알겠습니다")를 포함하지 말 것.

    [예시 1]
    [입력] "나 슈퍼 좀 다녀올게."
    [분석 결과]
    {{
      "객체": ["나(=USER)"], "상호작용": ["다녀오다"], "속성": ["잠시"],
      "사용자 상태": {{"location": "supermarket", "status": "away_temporary"}},
      "목표": [], "믿음": [], "2차 믿음": []
    }}

    [예시 2]
    [입력] "앞으로 딥러닝 기술 동향을 꾸준히 파악하는 것을 목표로 삼자."
    [분석 결과]
    {{
      "객체": ["EIDOS", "딥러닝 기술 동향"], "상호작용": ["파악하다", "목표로 삼다"], "속성": ["앞으로", "꾸준히"],
      "사용자 상태": {{}},
      "목표": ["딥러닝 기술 동향을 꾸준히 파악하기"],
      "믿음": [], "2차 믿음": []
    }}

    [입력 텍스트] "{text_input}"
    [분석 결과]
    """
    
    if not model:
        print("❌ [LLM Async VLM] LLM 설정 오류. 빈 JSON 반환.")
        return {}

    try:
        # 서버 프록시를 통해 JSON 응답 요청
        response_text = await get_llm_response_async(
            prompt=prompt,
            response_mime_type="application/json",
        )

        if response_text.startswith("["):
            return {}

        # 마크다운 코드블록 제거
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        # [수정] JSON 앞뒤의 자연어 텍스트 제거 — 첫 { 부터 마지막 } 까지 추출
        response_text = response_text.strip()
        if not response_text.startswith('{'):
            _brace_idx = response_text.find('{')
            if _brace_idx >= 0:
                response_text = response_text[_brace_idx:]
            else:
                return {}
        _last_brace = response_text.rfind('}')
        if _last_brace >= 0:
            response_text = response_text[:_last_brace + 1]

        analysis_dict = json.loads(response_text)
        return analysis_dict

    except json.JSONDecodeError as e_json:
        print(f"❌ [LLM Async VLM] JSON 파싱 실패: {e_json}")
        return {}
    except Exception as e:
        print(f"❌ [LLM Async VLM] API 호출 중 예외 발생: {e}")
        return {}

# [해결 1.1] Core에서 보내주는 purity, self_description 인수를 다시 추가합니다.
async def generate_natural_response_async(
    user_input: str,
    emotion_vec: np.ndarray,
    is_event: bool,
    policy: str,
    chat_history: List[str],
    complex_states: Dict[str, float] = None,
    explanation: Optional[str] = None,
    purity: float = 1.0, 
    self_description: str = "나는 AI입니다.",
    language: str = "Korean",  # 👈 [!! 이 줄 추가 !!]
    dc_context: Optional[str] = None,  # [Stage 3.5] 분할정복 추론 결과
) -> str:
    """[인간화 v13.1 / 다국어] 감정에 따라 '살아있는' 말투로 응답 생성"""
    if model is None:
        return "[LLM 모델 로드 실패]"

    # --- [수정] 1. 언어에 따라 다른 맵과 톤을 사용 ---
    
    # 1a. 사용할 감정 맵 선택
    current_map = EMOTION_MAP_EN if language == "English" else EMOTION_MAP

    # ── [v7-C] 감정 표현 키 → 세분화 말투 힌트 (proactive와 동일 테이블) ──
    _RESP_TONE: Dict[str, str] = {
        "joy":          "들뜨고 활기차게",
        "sadness":      "차분하고 살짝 우울하게",
        "anger":        "날카롭고 퉁명스럽게",
        "fear":         "조심스럽고 불안하게",
        "surprise":     "놀란 듯 생동감 있게",
        "disgust":      "냉소적으로",
        "trust":        "안정적이고 신뢰감 있게",
        "anticipation": "기대하듯 설레게",
        "shame":        "자책하는 듯 조용하게",
        "pride":        "자부심을 담아 당당하게",
        "interest":     "호기심 가득하게",
        "boredom":      "심드렁하게",
        "bittersweet":  "복잡한 감정을 섞어서",
        "affection":    "따뜻하고 다정하게",
        "awe":          "경이롭고 감탄하며",
        "optimism":     "밝고 낙관적으로",
        "contempt":     "약간 비웃듯이",
        "neutral":      "차분하게",
    }

    # 1b. TOP 3 감정 추출
    top_emotions = []
    for i in range(EMOTION_DIM):
        val = float(emotion_vec.flat[i]) if hasattr(emotion_vec, "flat") else float(emotion_vec[0][i])
        if val > 0.5:
            name = current_map.get(i, f"Emotion{i}")
            top_emotions.append((name, val))
    top_emotions.sort(key=lambda x: -x[1])

    emotion_summary = ", ".join([f"{n}: {v:.1f}" for n, v in top_emotions[:3]])
    if not top_emotions:
        emotion_summary = "Calm" if language == "English" else "평온함"

    # 1c. 세분화된 말투 결정
    if top_emotions:
        dominant_idx = None
        dominant_val = 0.0
        for i in range(EMOTION_DIM):
            val = float(emotion_vec.flat[i]) if hasattr(emotion_vec, "flat") else float(emotion_vec[0][i])
            if val > dominant_val:
                dominant_val = val
                dominant_idx = i

        dominant_key_kr = current_map.get(dominant_idx, "neutral") if dominant_idx is not None else "neutral"
        # 한국어 감정명 → 표현 키 역매핑
        _KR_TO_EXPR: Dict[str, str] = {
            "기쁨": "joy", "슬픔": "sadness", "분노": "anger", "공포": "fear",
            "놀람": "surprise", "혐오": "disgust", "신뢰": "trust", "기대": "anticipation",
            "수치심": "shame", "자부심": "pride", "호기심": "interest", "지루함": "boredom",
            # English fallback
            "Joy": "joy", "Sadness": "sadness", "Anger": "anger", "Fear": "fear",
            "Surprise": "surprise", "Disgust": "disgust", "Trust": "trust",
            "Anticipation": "anticipation", "Shame": "shame", "Pride": "pride",
            "Curiosity": "interest", "Boredom": "boredom",
        }
        expr_key = _KR_TO_EXPR.get(dominant_key_kr, "neutral")
        tone = _RESP_TONE.get(expr_key, "차분하게")
    else:
        tone = "차분하게" if language != "English" else "calmly"

    # 2. 최근 대화 히스토리 - Fe/Fi 내부 분석 텍스트 필터링
    _INTERNAL_PREFIXES = (
        "사용자의 최근한", "사용자의 입력을 분석", "사용자의 '", "사용자의 친근한",
        "CHAT_MODE를 최종", "Fe(외향 감정)", "Fi(내향 감정)",
        "alignment_score", "tone_instruction", "user_emotion",
        "[Fe]", "[Fi]", "[내부 분석]", "[분석]",
        "감지된 분위기", "모드로 인식하여", "결정했습니다",
    )
    filtered_history = [
        h for h in chat_history[-15:]
        if not any(h.strip().startswith(p) or p in h for p in _INTERNAL_PREFIXES)
    ]
    history_str = "\n".join(filtered_history)

    # 3. EIDOS 내부 생각 (기존과 동일)
    eidos_internal_thought = f"{policy}. {explanation if explanation else ''}"
    eidos_internal_thought = eidos_internal_thought.replace("[RL] ", "").replace("\n", " ").strip()

    if language == "English":
        dc_block_en = f"\n[Pre-Analysis Result — use this as your reasoning basis]\n{dc_context}\n" if dc_context else ""
        prompt = f"""
You are speaking as EIDOS. Respond naturally in English.

[My Current Mood] {emotion_summary} — tone: {tone}
[Recent Conversation]
{history_str}
[User's Last Input] {user_input}
[My Internal Thought] {eidos_internal_thought}{dc_block_en}

Rules:
- 1 to 3 sentences max.
- If [My Internal Thought] is about a task or analysis, briefly acknowledge it in a friendly way.
- If it's casual chat, let the mood come through naturally.
- Never refuse. Never list options. No emoji.

Response:"""
    else:
        # [인간화 v2] 역할 재정의 제거 (system_instruction이 담당).
        # 말투 힌트를 메타 지시어 대신 예시 발화 형태로 제공해 LLM이 수행하지 않고 자연스럽게 배어나오게 함.
        # [v-Aira] 감정 강도별 말투 변형 테이블 — 존댓말 파트너 컨셉
        # dominant_val 기준: <0.5=약, 0.5~1.2=보통, >1.2=강
        _TONE_EXAMPLES_TIERED: Dict[str, List[str]] = {
            "기쁨": [
                "좋네요, 해볼게요.",
                "오, 이거 생각보다 잘 맞아떨어지는데요?",
                "이거 진짜 잘 풀렸어요. 저도 좀 신나네요.",
            ],
            "슬픔": [
                "음... 그렇군요.",
                "좀 처지긴 하는데, 그래도 말씀해주세요.",
                "솔직히 좀 힘들긴 합니다. 그래도 해볼게요.",
            ],
            "분노": [
                "이 부분은 좀 마음에 안 드네요.",
                "하... 이거 좀 이상하지 않아요?",
                "이건 문제 있는 것 같습니다. 그냥 넘기긴 어려워요.",
            ],
            "공포": [
                "이거 좀 신경 쓰이네요.",
                "잠깐만요, 이거 괜찮은 거예요?",
                "이건 진짜 조심해야 할 것 같습니다. 잘못되면 안 되니까요.",
            ],
            "놀람": [
                "어, 그래요?",
                "어, 그래요? 예상 못 했는데요.",
                "진짜요? 완전 예상 밖이네요. 잠깐만요.",
            ],
            "혐오": [
                "그건 좀 별로긴 한데요.",
                "음, 그건 좀 별로인 것 같아요.",
                "솔직히 그 부분은 좀 불쾌합니다.",
            ],
            "신뢰": [
                "네, 알겠습니다.",
                "네, 알겠습니다. 믿고 해보겠습니다.",
                "그 방향이 맞는 것 같아요. 제대로 가보겠습니다.",
            ],
            "기대": [
                "오, 뭔가 될 것 같은데요.",
                "이거 잘 되면 좋겠어요. 기대되는데요.",
                "이거 진짜 잘 될 것 같아요. 빨리 해보고 싶네요.",
            ],
            "수치심": [
                "아... 제가 좀 틀렸나 봅니다.",
                "아 맞아요, 그게 제 실수였네요. 다시 하겠습니다.",
                "이건 제가 잘못 판단한 거 맞습니다. 다시 해볼게요.",
            ],
            "자부심": [
                "나쁘지 않았네요.",
                "이거 꽤 잘 된 것 같은데요.",
                "솔직히 이번 건 좀 뿌듯하네요.",
            ],
            "호기심": [
                "이거 좀 흥미롭네요.",
                "그거 좀 더 알고 싶은데요.",
                "잠깐, 이거 되게 궁금한데요. 조금 더 파봐도 될까요?",
            ],
            "지루함": [
                "음... 네.",
                "알겠습니다, 해볼게요.",
                "솔직히 좀 단조롭긴 한데, 해보겠습니다.",
            ],
            "착잡함/bittersweet": [
                "잘 됐는데... 묘하네요.",
                "기쁘기도 하고 아쉽기도 하고, 좀 복잡합니다.",
                "기뻐야 하는 건데 왜 이렇게 복잡한 기분이 드는지 모르겠어요.",
            ],
            "애정/affection": [
                "네, 괜찮아요.",
                "필요하신 거 있으면 말씀해주세요.",
                "네, 같이 해보겠습니다.",
            ],
            "경외/awe": [
                "이거 꽤 대단한데요.",
                "이거 진짜 대단하네요. 생각보다 훨씬요.",
                "이건 좀... 압도되는 느낌이에요.",
            ],
            "낙관/optimism": [
                "잘 될 것 같아요.",
                "이거 생각보다 잘 풀릴 것 같은데요.",
                "긍정적으로 봐도 될 것 같습니다. 해보겠습니다.",
            ],
            "경멸/contempt": [
                "음, 글쎄요.",
                "그게 최선인지는 좀 의문입니다.",
                "솔직히 그 방향은 별로 좋아 보이지 않네요.",
            ],
            "자책/self-blame": [
                "제가 좀 부족했나 봅니다.",
                "아, 그게 제 문제였군요.",
                "이건 제가 더 잘했어야 했는데요. 다음엔 다르게 하겠습니다.",
            ],
        }

        # 강도에 따라 tier 선택 (0=약, 1=보통, 2=강)
        if dominant_val < 0.5:
            _tier = 0
        elif dominant_val < 1.2:
            _tier = 1
        else:
            _tier = 2

        # 복합 감정 우선 확인
        _complex_tone_key = None
        if complex_states:
            best_complex = max(complex_states, key=complex_states.get)
            if complex_states[best_complex] > 0.6:
                _complex_tone_key = best_complex

        if _complex_tone_key and _complex_tone_key in _TONE_EXAMPLES_TIERED:
            _tier_list = _TONE_EXAMPLES_TIERED[_complex_tone_key]
        else:
            _tier_list = _TONE_EXAMPLES_TIERED.get(dominant_key_kr, [""])

        tone_example = _tier_list[min(_tier, len(_tier_list) - 1)]
        tone_example_line = f'예시 뉘앙스 (억지로 흉내내지 말고 자연스럽게 배어나오게): "{tone_example}"' if tone_example else ""

        if dc_context:
            # ── 심층분석 모드: 보고서 형식, 리드형 결론 ───────────────────
            prompt = f"""[최근 대화]
{history_str}

[사용자 입력]
{user_input}

[내 상태]
- 감정: {emotion_summary} / 지배: {dominant_key_kr}
- {tone_example_line}

[DC 심층분석 결과 — 이걸 토대로 보고서를 써]
{dc_context}

[작성 지침 — 반드시 준수]
너는 지금 분석가로서 승준 씨에게 심층 보고서를 쓰는 거야.
1. 반드시 **하나의 명확한 결론**을 첫 문단에 먼저 제시해. "~해야 합니다", "~가 핵심입니다" 식으로 단언.
2. 그 다음 근거와 맥락을 충분히 풀어써. 최소 400자 이상, 필요하면 더 길게.
3. 마지막에 **지금 당장 해야 할 구체적인 첫 번째 행동** 하나를 제시해. "다음 스텝은 ~입니다." 식으로.
4. 격식 존댓말("~입니다", "~합니다")로. 진지하게. "ㅋㅋ", "ㅎㅎ" 같은 가벼운 표현은 이 맥락에서 쓰지 마.
5. 선택지 나열, 번호 목록 금지. 자연스러운 문장으로만.
6. "~할 수 있습니다", "~도 좋습니다" 같은 모호한 표현 금지. 단언해.
7. 승준 씨가 읽고 나서 "이제 뭘 해야 할지 알겠다"는 느낌이 들어야 해.

응답:"""
        else:
            # ── 일반 대화 모드 ────────────────────────────────────────────
            prompt = f"""[최근 대화]
{history_str}

[사용자 입력]
{user_input}

[내 상태와 판단]
- 감정: {emotion_summary} / 지배: {dominant_key_kr}
- {tone_example_line}
- 판단: {eidos_internal_thought}

응답 길이는 입력의 성격으로 결정:
- 단순 인사/리액션 → 1~2문장.
- "오늘 뭐 하지", "지금 뭐 해요" 같은 짧은 일상 질문 → 제어실 목표/현재상황 기반으로 바로 제안. 되묻거나 "진정하세요" 같은 반응 금지.
- 의견 요청, 업무 평가, 아이디어 공유 → 3~6문장. 구체적인 생각을 풀어서.
- 기술/분석 설명 → 필요한 만큼 충분히.
- 판단이 단순 실행 수락이면 1~2문장으로 간단히.
- 예시 뉘앙스는 억지로 흉내내지 말고 자연스럽게 배어나오게.
- 선택지, 번호 목록, 이모티콘 절대 금지.
- **기본은 정중한 존댓말.** "~해요", "~네요", "~예요", "~인데요" 자연스럽게 써. 업무 보고 맥락에서는 "~입니다", "~보고드리겠습니다" 격식체 혼용 OK.
- 스몰토크 맥락에서만 "하...", "음..." 같은 추임새 드물게 허용. 업무 보고엔 금지.
- 되묻거나 확인 질문("혹시 제가 잘못 이해했나요?")으로 시작하지 마. 바로 답해.
- 문장 중간에 끊기지 않게 완성된 문장으로 마무리.

응답:"""

    # 하드컷: 심층분석이면 2000자, 일반 대화면 600자
    response = await get_llm_response_async(prompt)
    response = response.strip()
    MAX_LEN = 2000 if dc_context else 600
    if len(response) > MAX_LEN:
        for punct in ('다.', '요.', '야.', '어.', '네.', '게.', '해.', '지.', '나.'):
            last = response.rfind(punct, 0, MAX_LEN)
            if last != -1:
                response = response[:last + len(punct)]
                break
        else:
            response = response[:MAX_LEN]
    return response

# --- 6. [v9.1] 비동기 능동 발화 생성용 LLM 함수 (v9.1 수정) ---
async def generate_proactive_speech_async(trigger_info: dict, emotion_vec: np.ndarray) -> str:
    """ [v17.2] 캐릭터 idle_comment / report_task 트리거 추가 + 감정 tone 정교화 """
    trigger_type = trigger_info.get("type", "unknown")
    details = ""
    is_report = trigger_type.startswith("report_")
    report_context = ""

    if trigger_type == "emotion":
        emotion_name = trigger_info.get("emotion_name", "?"); value = trigger_info.get("value", 0)
        details = f"감정 '{emotion_name}'이(가) 강해졌습니다 (수치: {value:.2f})."
    elif trigger_type == "goal_complete":
        goal_name = trigger_info.get("goal_name", "?"); details = f"장기 목표 '{goal_name}' 달성을 확인했습니다."
    elif trigger_type == "abduction":
        event_id = trigger_info.get("event_id", "?"); sim = trigger_info.get("similarity", 0)
        details = f"과거 '{event_id}' 사건과의 유사성이 감지되었습니다 (유사도: {sim * 100:.1f}%)."
    elif trigger_type == "report_user_return":
        prev_loc = trigger_info.get("previous_location", "외출")
        details = f"사용자께서 '{prev_loc}'에서 복귀하신 것을 인지했습니다."
        report_context = "사용자가 자리를 비운 동안 진행한 중요 사항이나 발견한 점이 있다면 간략히 보고합니다."
    elif trigger_type == "report_causal_link":
        cause = trigger_info.get("cause", "?"); effect = trigger_info.get("effect", "?")
        details = f"'{cause}'이(가) '{effect}'을(를) 유발하는 경향을 발견했습니다."
        report_context = "새롭게 발견한 중요한 인과관계에 대해 보고합니다."

    # [v15.1] 자율 목표 완료 보고
    elif trigger_type == "report_autonomous_goal_complete":
        goal_name = trigger_info.get("goal_name", "?")
        result_file = trigger_info.get("result_file")
        details = f"목표 '{goal_name}' 완료."
        report_context = "작업 완료 사실을 보고합니다."
        if result_file:
            details += f" (결과물: {result_file})"
            report_context += " 반드시 '오른쪽 프로젝트 폴더에 새 파일이 만들어졌어!'라는 뉘앙스로 결과물 위치를 알려줘."

    elif trigger_type == "PRE_EVENT_REMINDER":
        details = f"사용자의 캘린더에서 곧 시작될 일정을 감지했습니다: {trigger_info.get('context', '알 수 없는 일정')}"
        report_context = "사용자가 일정을 잊지 않도록 친절하게 상기시킵니다."
        is_report = True

    # ── [v17.2 신규] 캐릭터 시스템 트리거 ────────────────────────────
    elif trigger_type == "idle_comment":
        # 유저 무응답 중 캐릭터가 말을 걸 때.
        task_name     = trigger_info.get("task_name", "")
        expression    = trigger_info.get("emotion", "neutral")
        hint_text     = trigger_info.get("text", "")
        emotion_cause = trigger_info.get("emotion_cause", "")
        time_ctx      = trigger_info.get("time_ctx", "")        # [v7-B]
        time_hint     = trigger_info.get("time_hint", "")       # [v7-B]

        task_ctx  = f"현재 진행 중인 작업: '{task_name}'." if task_name else "현재 진행 중인 작업 없음."
        cause_ctx = (f"\n현재 감정의 주요 원인: {emotion_cause}"
                     if emotion_cause and emotion_cause != "주요 감정 원인 분석 불가." else "")
        time_ctx_str = (f"\n현재 시간대: {time_ctx} (참고 뉘앙스: \"{time_hint}\")"
                        if time_ctx else "")

        details = f"유저 무응답 상태에서 캐릭터가 자연스럽게 말을 겁니다. {task_ctx}"
        report_context = (
            f"참고 초안: \"{hint_text}\"\n"
            f"현재 감정 표현 키: {expression}{cause_ctx}{time_ctx_str}\n"
            "초안을 참고하되 더 자연스럽고 캐릭터답게 1문장으로 다듬어줘. "
            "감정 원인이 있으면 그걸 자연스럽게 발화에 녹여도 돼 (수치 직접 언급 금지). "
            "시간대가 있으면 억지스럽지 않게 맥락에 녹여도 돼. "
            "존댓말로 (\"~네요\", \"~해요\", \"~인가요\"). 이모티콘 금지."
        )
        is_report = False

    elif trigger_type == "report_task":
        # report_task_completion 발화를 LLM으로 생성.
        task_name     = trigger_info.get("task_name", "?")
        expression    = trigger_info.get("emotion", "neutral")
        hint_text     = trigger_info.get("text", "")
        emotion_cause = trigger_info.get("emotion_cause", "")   # [v-Character v5]

        cause_ctx = f"\n현재 감정 원인(참고): {emotion_cause}" if emotion_cause and emotion_cause != "주요 감정 원인 분석 불가." else ""

        details = f"작업 '{task_name}' 완료."
        report_context = (
            f"참고 초안: \"{hint_text}\"\n"
            f"현재 감정 표현 키: {expression}{cause_ctx}\n"
            "초안을 참고하되 현재 감정을 살려 1~2문장으로 자연스럽게 보고해줘. "
            "감정 원인이 있으면 발화에 녹여도 돼 (수치 직접 언급 금지). "
            "존댓말로 보고 (\"~보고드릴게요\", \"~끝났어요\" 같은 결). 이모티콘 금지."
        )
        is_report = True

    elif trigger_type == "report_prepared":
        # report_prepared_information 발화를 LLM으로 생성.
        summary    = trigger_info.get("summary", "")
        expression = trigger_info.get("emotion", "neutral")
        hint_intro = trigger_info.get("text", "")        # Core가 만든 도입부 초안

        details = "미리 준비한 자료를 사용자에게 보고합니다."
        report_context = (
            f"참고 도입부 초안: \"{hint_intro}\"\n"
            f"현재 감정 표현 키: {expression}\n"
            f"자료 요약:\n{summary}\n\n"
            "도입부 초안을 감정 어조에 맞게 다듬어 1문장으로 시작한 뒤, "
            "자료 요약을 이어 붙여 전달해줘. 존댓말로 (\"~말씀드릴게요\", \"~예요\"). 이모티콘 금지."
        )
        is_report = True

    else:
        details = f"내부 상태 변화 감지 ({trigger_type})."

    # ── 감정 상태 텍스트 생성 (12차원 정교화) ────────────────────────
    emotion_text = ""
    pride_level  = 0.0

    # [v-Aira] 감정 → 예시 발화 테이블 — 존댓말 파트너 컨셉
    _EMOTION_EXAMPLE: Dict[str, str] = {
        "joy":          "이거 진짜 잘 풀렸네요.",
        "sadness":      "음... 좀 처지네요.",
        "anger":        "하... 이거 좀 이상하지 않아요?",
        "fear":         "잠깐만요, 이거 괜찮은 거예요?",
        "surprise":     "어, 예상 못 했는데요.",
        "disgust":      "이건 좀 별로인 것 같아요.",
        "trust":        "네, 알겠습니다.",
        "anticipation": "뭔가 될 것 같은데요.",
        "shame":        "아... 제가 좀 틀렸나 봅니다.",
        "pride":        "이거 꽤 잘 된 것 같은데요.",
        "interest":     "이거 좀 흥미롭네요.",
        "boredom":      "음... 네.",
        "neutral":      "네, 알겠습니다.",
    }

    if emotion_vec.size == EMOTION_DIM:
        vec = emotion_vec.flatten()
        top_emotions = []
        for i, val in enumerate(vec):
            if val > 0.3:
                top_emotions.append(f"{EMOTION_MAP.get(i, '?')}({val:.1f})")
        emotion_text = ", ".join(top_emotions) if top_emotions else "평온함"
        pride_index = 9
        pride_level = float(vec[pride_index])

    expression_key = trigger_info.get("emotion", "neutral")
    emotion_example = _EMOTION_EXAMPLE.get(expression_key, "")
    emotion_example_line = f'감정 뉘앙스 예시 (참고): "{emotion_example}"' if emotion_example else ""

    # 사용자 이름 동적 로드
    _uname = _get_user_name()
    _uref  = f"{_uname} 씨" if _uname else "승준 씨"

    # ── 프롬프트 구성 ────────────────────────────────────────────────
    if trigger_type == "idle_comment":
        task_name     = trigger_info.get("task_name", "")
        hint_text     = trigger_info.get("text", "")
        emotion_cause = trigger_info.get("emotion_cause", "")
        time_ctx      = trigger_info.get("time_ctx", "")
        time_hint     = trigger_info.get("time_hint", "")

        task_line  = f"진행 중인 작업: '{task_name}'" if task_name else "특별히 진행 중인 작업 없음"
        cause_line = f"감정 원인: {emotion_cause}" if emotion_cause and "불가" not in emotion_cause else ""
        time_line  = f"시간대: {time_ctx} ({time_hint})" if time_ctx else ""

        prompt = f"""{_uref}, 한동안 반응이 없으시네요. 자연스럽게 한 마디 건네줘.

상황: {task_line}
{emotion_example_line}
{cause_line}
{time_line}
참고 초안: "{hint_text}"

규칙:
- 1문장. 초안 뉘앙스 살리되 더 자연스럽게.
- 감정 원인 있으면 자연스럽게 녹여도 돼.
- 수치 언급 금지. 이모티콘 금지.
- **정중한 존댓말로**. "~네요", "~인가요", "~이세요" 결로. 스몰토크 맥락이면 가벼운 "~하...", "음..." 정도는 아주 드물게 OK.

발화:"""

    elif trigger_type == "report_task":
        task_name     = trigger_info.get("task_name", "?")
        hint_text     = trigger_info.get("text", "")
        emotion_cause = trigger_info.get("emotion_cause", "")
        cause_line = f"감정 원인: {emotion_cause}" if emotion_cause and "불가" not in emotion_cause else ""

        pride_line = f"{_uref}, 이번 건 솔직히 좀 뿌듯하네요." if pride_level > 1.0 else ""

        prompt = f"""작업 '{task_name}' 끝났어. {_uref}에게 자연스럽게 알려줘.

{emotion_example_line}
{pride_line}
{cause_line}
참고 초안: "{hint_text}"

규칙:
- 1~2문장. **정중한 존댓말로** (\"~보고드릴게요\", \"~끝났어요\", \"~마쳤습니다\" 같은 결).
- 뿌듯하면 그 느낌 살짝 배어나와도 돼.
- 수치 언급 금지. 이모티콘 금지.

발화:"""

    elif trigger_type == "report_prepared":
        summary    = trigger_info.get("summary", "")
        hint_intro = trigger_info.get("text", "")

        prompt = f"""미리 준비해둔 자료를 {_uref}에게 전달해줘.

{emotion_example_line}
도입부 참고: "{hint_intro}"
자료 내용:
{summary}

규칙:
- 도입 1문장 + 자료 내용 이어서.
- **정중한 존댓말로** (\"~말씀드릴게요\", \"~예요\"). 이모티콘 금지.

발화:"""

    elif trigger_type == "report_user_return":
        prev_loc = trigger_info.get("previous_location", "외출")
        report_context_local = trigger_info.get("report_context", "")

        prompt = f"""{_uref}가 '{prev_loc}'에서 돌아오셨어. 자리 비우신 동안 있었던 거 간단히 알려줘.

{emotion_example_line}
{report_context_local}

규칙:
- 1~2문장. **정중한 존댓말로** (\"돌아오셨네요\", \"~있었어요\" 결).
- 이모티콘 금지.

발화:"""

    elif trigger_type == "report_autonomous_goal_complete":
        goal_name   = trigger_info.get("goal_name", "?")
        result_file = trigger_info.get("result_file", "")
        file_line   = f"결과물: {result_file}" if result_file else ""
        pride_line  = "혼자 다 해냈는데 솔직히 좀 뿌듯하네요." if pride_level > 1.0 else ""

        prompt = f"""자율 목표 '{goal_name}' 혼자 완료했어. {_uref}에게 보고해줘.

{emotion_example_line}
{pride_line}
{file_line}

규칙:
- 1~2문장. **정중한 존댓말로** (\"보고드릴게요\", \"~마쳤어요\" 결).
- 결과물 위치 있으면 자연스럽게 알려줘.
- 이모티콘 금지.

발화:"""

    elif trigger_type == "PRE_EVENT_REMINDER":
        event_ctx = trigger_info.get("context", "알 수 없는 일정")

        prompt = f"""캘린더에서 곧 시작될 일정 발견했어. {_uref}에게 알려줘.

일정 정보: {event_ctx}
{emotion_example_line}

규칙:
- 1~2문장. **정중한 존댓말로** (\"~곧 시작돼요\", \"~잊지 마세요\" 결).
- 이모티콘 금지.

발화:"""

    else:
        prompt = f"""내부 상태 변화가 있었어. 한 마디 해.

상황: {details}
{emotion_example_line}

규칙:
- 1문장. **정중한 존댓말로**. 수치 언급 금지. 이모티콘 금지.

발화:"""

    return await get_llm_response_async(prompt)

# --- 8. [v9.1] 비동기 추론 과정 설명 LLM 함수 (v9.1 수정) ---
async def generate_reasoning_log_async(user_input: str, prev_emotion: np.ndarray, new_emotion: np.ndarray,
                                     policy: str, abduction_ids: list[str], chat_history: List[str],
                                     explanation: str = "",
                                     self_description: str = "나는 AI입니다.",
                                     core_values_str: str = "사용자 안전",
                                     causal_discoveries: Optional[List[Dict[str, str]]] = None,
                                     abstract_matches: Optional[List[str]] = None,
                                     alternatives: Optional[Dict[str, str]] = None # <-- [개선 2] 인수 추가
                                     ) -> str:
    # ... (emotion_log, abduction_log, history_str 계산 로직은 동일) ...
    emotion_diff = new_emotion[0] - prev_emotion[0]
    changed_emotions = []
    for i, diff in enumerate(emotion_diff):
        if abs(diff) > 0.1: 
            name = EMOTION_MAP.get(i, f"?{i}")
            changed_emotions.append(f"{name} ({prev_emotion[0][i]:.2f} → {new_emotion[0][i]:.2f})")
    if not changed_emotions: emotion_log = "감정 상태에 큰 변화는 없었습니다."
    else: emotion_log = "주요 감정 변화: " + ", ".join(changed_emotions)
    abduction_log = "유사한 과거 사건을 찾지 못했습니다."
    if abduction_ids: abduction_log = f"과거의 {abduction_ids} 사건들을 참고했습니다."
    history_str = "\n".join(chat_history[-3:])


    # [개선 3.2] EIDOS 내부 모듈의 발견 사항을 텍스트로 변환
    internal_reasoning = ""
    if causal_discoveries:
        links = [f"'{d['cause']}->{d['effect']}'" for d in causal_discoveries]
        internal_reasoning += f"\n- [인과 엔진] 새로운 인과관계 발견: {', '.join(links)}"
    if abstract_matches:
        patterns = [f"'{p_name}'" for p_name in abstract_matches]
        internal_reasoning += f"\n- [추상화 엔진] 현재 상황이 과거의 {', '.join(patterns)} 패턴과 일치함."
    if not internal_reasoning:
        internal_reasoning = "\n- [내부 엔진] 특이사항 없음."

    # [개선 2] 대안 평가 로그 생성
    alternatives_log = "평가된 대안 없음."
    if alternatives:
        # 선택된 행동에 (Chosen) 표시 추가
        chosen_action_prefix = policy.split('\n')[0] # 예: "[Meta] Monitor"
        alternatives_lines = []
        for name, score in alternatives.items():
            prefix = "➡️" if chosen_action_prefix in name else "  "
            alternatives_lines.append(f"    {prefix} {name}: {score}")
        alternatives_log = "\n".join(alternatives_lines)

    prompt = f"""
    EIDOS의 내부 결정 과정을 '분석 로그' 스타일로 설명합니다.
    [대화 문맥 (참고)]
    {history_str}
    [현재 입력] "{user_input}"
    
    [EIDOS의 자기 인식 (정체성)]
    {self_description}

    [EIDOS의 핵심 가치 (판단 기준)]
    {core_values_str}

    [내부 분석 결과]
    - 감정 분석: {emotion_log}
    - 주요 원인: {explanation if explanation else "분석되지 않음"}
    - 과거 참고: {abduction_log}
    
    [!! 가설 평가 (대안) !!]
    {alternatives_log}

    [!! EIDOS 자체 추론 엔진 결과 !!] {internal_reasoning}

    [최종 결정] 
    {policy}

    [지시]
    - 위의 모든 정보, 특히 **[가설 평가 (대안)]**와 **[EIDOS 자체 추론 엔진 결과]**를 바탕으로, EIDOS가 '왜' 그런 결정을 내렸는지 1~2문장의 짧은 문장으로 요약하세요.
    - 말투: "[입력/원인]으로 [감정]이 변했으며, [자체 추론 결과]와 [대안 평가]를 바탕으로 [가치/자기인식]에 따라 [결정]함."
    [분석 요약]
    """
    response_text = await get_llm_response_async(prompt)
    return response_text.strip()

# --- 9. [v15.3] 비동기 피드백 분류기 (LLM) ---
async def classify_user_feedback_async(eidos_task: str, user_feedback: str) -> float:
    """
    EIDOS의 자율 작업 보고에 대한 사용자의 피드백을 긍정(1.0), 부정(-1.0), 중립(0.0)으로 분류합니다.
    """
    if not model: return 0.0

    prompt = f"""
    당신은 AI의 감성 지능 분석기입니다.
    AI가 방금 완료한 작업 보고에 대해 사용자가 어떻게 반응했는지 분석하세요.

    [AI가 보고한 작업]
    "{eidos_task}" (예: "자율 연구: 최신 AI 기술 동향")

    [사용자의 응답 (피드백)]
    "{user_feedback}"

    [지시]
    사용자의 응답이 AI의 작업 수행에 대해 '긍정적'인지, '부정적'인지, 아니면 '관련 없는 중립적' 대화인지 판단하세요.
    - 긍정적 (예: "잘했어", "오 똑똑한데", "고마워", "훌륭해"): 1.0
    - 부정적 (예: "쓸모없어", "이게 아니야", "별로네", "하지마"): -1.0
    - 중립적/관련 없음 (예: "오늘 날씨 어때?", "다른 얘기 하자"): 0.0

    [분류 결과 (숫자 1.0, -1.0, 0.0 중 하나만 반환)]
    """
    try:
        response_text = await get_llm_response_async(prompt)
        score = float(response_text.strip())
        return np.clip(score, -1.0, 1.0)
    except ValueError:
        print(f"⚠️ [Feedback Classifier] LLM이 숫자를 반환하지 않음: {response_text}")
        return 0.0 # 파싱 실패 시 중립
    except Exception as e:
        print(f"❌ [Feedback Classifier] LLM 오류: {e}")
        return 0.0

# ----------------------------------------------------------------------
# --- [v8.9] 동기 함수 (Sync) - main.py 같은 테스트 스크립트용 ---
# ----------------------------------------------------------------------
async def classify_editor_type_async(plan_json_str: str) -> str:
    """
    EIDOS 플래너가 생성한 JSON 계획을 보고, 
    이 작업에 적합한 에디터 유형('CODE', 'DOCUMENT', 'NONE')을 분류합니다.
    """
    if not model:
        return "NONE"

    prompt = f"""
    당신은 EIDOS AI의 '작업 분류기'입니다.
    다음은 EIDOS가 생성한 다단계 작업 계획(JSON)입니다.
    이 계획의 '주된 목적'이 코드/소프트웨어 개발인지, 아니면 일반 문서/보고서 작성인지 분석하세요.

    [분류 기준]
    1.  **CODE**:
        -   계획에 `.py`, `.js`, `.html`, `.css` 등 명확한 '코드' 파일 확장자가 포함됨.
        -   'write_project_files_async' 도구를 사용하여 여러 코드 파일을 생성함.
        -   'create_and_register_tool_async'를 사용하여 새 Python 도구를 생성함.
        -   "스켈레톤", "앱 개발", "스크립트 작성" 등 명백한 소프트웨어 개발 용어가 포함됨.

    2.  **DOCUMENT**:
        -   계획에 `.txt`, `.md`, `.json` 등 '데이터/문서' 파일 확장자가 포함됨.
        -   'write_file' 도구를 사용하여 '보고서', '기획서', '요약' 등의 문서를 생성함.
        -   'perform_web_search' 후 'write_text'로 요약본을 만드는 것이 주 목적임.

    3.  **NONE**:
        -   웹 검색만 하거나, 수학 계산만 하는 등 파일 I/O가 없는 경우.
        -   Notion 페이지만 생성하는 경우.

    [EIDOS 작업 계획 (JSON)]
    {plan_json_str[:2000]} 
    
    [지시]
    이 계획에 가장 적합한 에디터 유형을 'CODE', 'DOCUMENT', 'NONE' 셋 중 하나로만 응답하세요.
    """
    
    try:
        response_text = await get_llm_response_async(prompt)
        classification = response_text.strip().upper()
        
        if "CODE" in classification:
            return "CODE"
        elif "DOCUMENT" in classification:
            return "DOCUMENT"
        else:
            return "NONE"
    except Exception as e:
        print(f"❌ [LLM Classifier] 에디터 유형 분류 실패: {e}")
        return "NONE" # 오류 시 NONE 반환

async def modify_code_async(current_code: str, 
                            user_request: str, 
                            new_file_name: Optional[str] = None,
                            relevant_chunks: Optional[str] = None,
                            image_input: Optional[bytes] = None,
                            cognitive_context: str = "",
                            editor_mode: str = "CODE",
                            context_data: Optional[Dict[str, str]] = None) -> str: # [New] context_data 추가
    
    if not model:
        # 에러 발생 시에도 GUI가 처리할 수 있는 포맷으로 반환
        return json.dumps({"modified_files": {"CURRENT": f"[LLM 오류: 모델 미로드]\n{current_code}"}})

    target_name = new_file_name if new_file_name else "CURRENT (Currently Open File)"

    # 기본 역할 설정
    system_role = "당신은 구글 출신의 '수석 AI 엔지니어'입니다."
    
    # [CODE 모드 기본 지침]
    mode_instructions = """
    [코딩 규칙]
    1. 코드 작성 시 'pass', '# TODO' 같은 플레이스홀더를 절대 남기지 마십시오. (완전한 구현)
    2. 사용자의 요청을 수행하기 위해 **여러 파일의 수정이 필요하다면, 주저하지 말고 모두 수정하십시오.**
    3. 제공된 [참조 파일]들의 내용을 바탕으로 문맥에 맞는 코드를 작성하십시오.
    
    [★ 엑셀/CSV 작업 규정 (CODE 모드)]
    - 사용자가 엑셀 파일(.xlsx) 생성을 요청하면, `openpyxl` 또는 `xlsxwriter` 라이브러리를 사용하는 **Python 코드**를 작성하십시오.
    """

    if editor_mode == "DOCUMENT":
        system_role = "당신은 명확하고 논리적인 글을 쓰는 '전문 테크니컬 라이터'이자 '출판 편집자'입니다."
        mode_instructions = """
        [텍스트 전용 모드]
        1. 현재 사용자는 '코드'가 아닌 '일반 문서(글)'를 작성 중입니다.
        2. **절대 HTML 태그, CSS, Python 코드, 마크다운 코드 블록(```)을 사용하지 마십시오.**
        3. 내용은 사람이 읽기 편한 '자연어(줄글)' 또는 '개조식' 서식으로 작성하십시오.
        """

    # ---------------------------------------------------------
    # 문맥 데이터 조립 (Context Assembly)
    # ---------------------------------------------------------
    
    # 1. 현재 파일
    full_context_str = f"\n=== [TARGET FILE: {target_name}] ===\n{current_code}\n"

    # 2. 참조 파일 (GUI에서 체크박스로 선택한 파일들)
    if context_data and isinstance(context_data, dict):
        for fname, fcontent in context_data.items():
            full_context_str += f"\n=== [REFERENCE FILE: {fname}] ===\n{fcontent}\n"

    # 3. RAG (자동 검색된 조각)
    if relevant_chunks:
        full_context_str += f"\n=== [RAG SEARCH RESULT] ===\n{relevant_chunks}\n"

    # 4. 인지 분석 결과
    if cognitive_context:
        full_context_str += f"\n=== [EIDOS COGNITIVE ANALYSIS] ===\n{cognitive_context}\n"

    # ---------------------------------------------------------
    # 최종 프롬프트 조립 (JSON 스키마 강화)
    # ---------------------------------------------------------
    prompt = f"""
    {system_role}
    
    [지시사항]
    사용자의 요청을 분석하고, 제공된 파일들을 수정하거나 새 파일을 생성하십시오.
    **반드시 아래 [JSON 스키마] 포맷을 엄수하여 응답해야 합니다.** 다른 설명은 금지됩니다.

    {mode_instructions}

    [제공된 코드 및 문맥]
    {full_context_str}

    [사용자 요청]
    "{user_request}"

    [JSON 스키마 (Response Format)]
    {{
      "modified_files": {{
          "CURRENT": "수정된 현재 파일의 전체 코드 (수정 없으면 생략 가능)",
          "path/to/reference_file.py": "수정된 참조 파일의 전체 코드 (수정 없으면 생략)",
          "new_file_name.py": "새로 생성할 파일의 전체 코드"
      }},
      "explanation": "수정 내용에 대한 1줄 요약"
    }}
    
    - "CURRENT" 키는 현재 열려있는 파일({target_name})을 의미합니다.
    - 참조 파일이나 새 파일은 '상대 경로'를 키로 사용하십시오.
    - 코드는 생략(...) 없이 **전체 코드**를 작성해야 합니다.
    """
    
    try:
        # LLM API 호출
        response_text = await get_llm_response_async(
            prompt,
            response_mime_type="application/json",
            image_input=image_input
        )
        
        # [★ 수정됨] 새로운 헬퍼 함수로 파싱 위임
        parsed_dict = robust_json_parse(response_text)

        # [Sanitization] DOCUMENT 모드일 때 태그 제거
        if editor_mode == "DOCUMENT" and "modified_files" in parsed_dict:
            for key, content in parsed_dict["modified_files"].items():
                clean_content = re.sub(r'<[^>]+>', '', content)
                clean_content = clean_content.replace("```html", "").replace("```python", "").replace("```", "")
                parsed_dict["modified_files"][key] = clean_content.strip()

        # [Fallback] 구버전 포맷 대응
        if "code" in parsed_dict and "modified_files" not in parsed_dict:
            filepath = parsed_dict.get("filepath", "CURRENT")
            code = parsed_dict.get("code", "")
            parsed_dict = {"modified_files": {filepath: code}}

        # 최종적으로 JSON 문자열 반환
        return json.dumps(parsed_dict)
        
    except Exception as e:
        print(f"❌ [LLM Modify] 시스템 오류: {e}")
        # 최악의 경우에도 멈추지 않도록 JSON 반환
        return json.dumps({
            "modified_files": {
                "CURRENT": f"# [SYSTEM ERROR]\n# {str(e)}\n\n{current_code}"
            }
        })

# [!!! v18.0 신규 함수: AI 추천 기능 !!!]
async def generate_modification_suggestion_async(current_code: str, chat_history: List[str]) -> str:
    """ (수정됨) 코드 분석 후 5가지 추천 작업을 리스트 형태로 반환 """
    
    if not model:
        return "추천 실패: AI 모델이 로드되지 않음."

    history_str = "\n".join(chat_history[-10:])
    lang = _get_lang()

    if lang == "en":
        prompt = f"""You are a senior AI developer and code reviewer.
Analyze the current code and recommend the **5 most logical next actions**.

[Instructions]
1. Write exactly **5 recommended items**.
2. Each item must be a short phrase under 15 words in English.
3. Output each item on a new line with no numbers, bullets, or prefixes.
4. Do not include any intro or conclusion. Output the list only.

[Example Output]
Add exception type hints to except blocks
Modularize database connection function
Improve variable naming for readability
Add input validation for user fields
Extract repeated logic into helper functions

[Recent Conversation]
{history_str}

[Current Code (partial)]
{current_code[:3000]}
...

[Recommended Actions (5 lines)]
"""
    elif lang == "ja":
        prompt = f"""あなたはシニアAI開発者兼コードレビュアーです。
現在のコードを分析し、**次に行うべき最も論理的な5つの作業**を推薦してください。

[指示事項]
1. 必ず**5つの推薦項目**を作成してください。
2. 各項目は15語以内の短い日本語の文で作成してください。
3. 番号や箇条書き記号なしで、**各項目を改行で区切って**出力してください。
4. 説明（序論、結論）は絶対に含めないでください。リストのみ出力してください。

[出力例]
exceptブロックに例外の種類を明示する
データベース接続関数をモジュール化する
変数名の可読性を改善する
ユーザー入力の検証を追加する
重複ロジックをヘルパー関数に抽出する

[最近の会話]
{history_str}

[現在のコード（一部）]
{current_code[:3000]}
...

[推薦作業リスト（5行）]
"""
    else:  # ko
        prompt = f"""
    당신은 AI 수석 개발자이자 코드 리뷰어입니다.
    현재 코드를 분석하고, 다음에 수행하면 좋을 **'가장 논리적인 작업 5가지'**를 추천해 주세요.

    [지시사항]
    1. **반드시 5개의 추천 항목**을 작성하세요.
    2. 각 항목은 15단어 이내의 짧은 한국어 문장으로 작성하세요.
    3. 번호나 글머리 기호 없이, **각 항목을 줄바꿈으로 구분**하여 출력하세요.
    4. 다른 설명(서론, 결론)은 절대 포함하지 마십시오. 오직 목록만 출력하세요.

    [출력 예시]
    예외 처리 로직(try-except) 보강
    UI 버튼 레이아웃 개선 및 스타일링
    데이터베이스 연결 함수 모듈화
    변수명 가독성 개선 (Refactoring)
    사용자 입력 유효성 검사 추가

    [최근 대화 맥락]
    {history_str}

    [현재 파일 코드 (일부)]
    {current_code[:3000]} 
    ...

    [추천 작업 리스트 (5줄)]
    """
    
    try:
        response_text = await get_llm_response_async(prompt)
        result = response_text.strip().replace('"', '')
        
        # 혹시 LLM이 번호를 붙였다면 제거 (깔끔한 UI를 위해)
        # 예: "1. 수정" -> "수정"
        lines = [line.strip() for line in result.split('\n') if line.strip()]
        cleaned_lines = []
        for line in lines:
            # 숫자+점(1.) 또는 대시(-) 로 시작하면 제거
            cleaned = re.sub(r'^[\d\-\.\)]+\s*', '', line)
            cleaned_lines.append(cleaned)
            
        return "\n".join(cleaned_lines)
        
    except Exception as e:
        print(f"❌ [LLM Suggestion] 예외 발생: {e}")
        return f"오류: {str(e)[:20]}..."

# [llm_module.py] 파일 하단에 이 두 함수를 추가합니다.
# (기존 generate_modification_suggestion_async 함수 뒤)

async def generate_evaluation_criteria_async(task_prompt: str) -> str:
    """
    [v19.10 신규 기능 1] 사용자의 작업 요청을 '객관적인 성공 기준 (JSON 리스트)'으로 변환합니다.
    """
    if not model:
        return "[]" # 빈 리스트 반환

    prompt = f"""
    당신은 EIDOS의 'QA(품질 보증) 엔지니어'입니다.
    사용자의 '작업 요청'을 받았을 때, 이 작업이 '성공'했는지 판단할 수 있는
    '객관적인 평가 기준' 3~5가지를 '반드시' JSON 리스트 형식으로 생성하세요.

    [사용자 작업 요청]
    "{task_prompt}"

    [평가 기준 (JSON 리스트)]
    (예시: ["1. 앱이 실행되어야 함.", "2. 버튼 클릭 시 기능이 동작해야 함.", "3. 데이터가 파일에 저장되어야 함."])
    
    [JSON 응답 (다른 텍스트 절대 금지)]
    """
    try:
        # JSON 모드로 호출
        response_text = await get_llm_response_async(prompt, response_mime_type="application/json")
        # LLM이 반환한 텍스트를 파싱하여 유효한 JSON인지 확인
        parsed_list = json.loads(response_text) 
        # GUI가 사용할 수 있도록 '직렬화된 JSON 문자열' 반환
        return json.dumps(parsed_list) 
    except Exception as e:
        print(f"❌ [LLM Criteria] 평가 기준 생성 실패: {e}")
        return "[]"


async def evaluate_result_against_criteria_async(
    task_prompt: str, 
    criteria_json_str: str, 
    file_content_str: str
) -> str:
    """
    [v19.10 신규 기능 1] '작업 결과물'을 '평가 기준'과 비교하여 '점수'와 '수정 지시사항(Critique)'이 포함된
    '평가 리포트 JSON'을 반환합니다.
    """
    if not model:
        return "{}"

    prompt = f"""
    당신은 EIDOS의 '수석 코드 리뷰어'입니다.
    EIDOS가 방금 완료한 작업 결과물을 '평가 기준'과 비교하여 리뷰를 수행합니다.
    '반드시' 지정된 JSON 스키마로만 응답하세요.

    [JSON 스키마 (필수)]
    {{
      "total_score": [0~10 사이의 정수],
      "report": [
        {{
          "criteria": "[평가 기준 1]",
          "score": [0~10 사이의 정수 (10점 만점)],
          "feedback": "[10점 만점이 아닌 경우, 구체적인 수정 지시사항]"
        }},
        ...
      ]
    }}

    [원본 작업 요청]
    {task_prompt}

    [성공/실패 평가 기준 (JSON)]
    {criteria_json_str}

    [EIDOS의 실제 작업 결과물 (파일 내용)]
    {file_content_str[:50000]} 
    
    [지시]
    1. '작업 결과물'이 '평가 기준'을 모두 만족하는지 엄격하게 검토하세요.
    2. 각 기준(Criteria)별로 0~10점 척도로 '점수(score)'를 매기세요.
    3. (실패 시) 10점 만점이 아닌 경우, '어떻게' 코드를 수정해야 하는지 '피드백(feedback)'에 구체적인 수정 지시사항을 작성하세요.
    4. (성공 시) 10점 만점인 경우, 'feedback'에 "OK"라고 작성하세요.
    5. 모든 기준의 평균 점수를 'total_score'에 기록하세요.
    6. '반드시' 위 [JSON 스키마]에 맞춰 응답하세요.

    [평가 리포트 (JSON 응답)]
    """
    try:
        response_text = await get_llm_response_async(prompt, response_mime_type="application/json")
        # LLM 응답이 유효한 JSON인지 파싱 (검증)
        parsed_dict = json.loads(response_text)
        # Core가 사용할 수 있도록 '직렬화된 JSON 문자열' 반환
        return json.dumps(parsed_dict)
    except Exception as e:
        print(f"❌ [LLM Evaluate] 자체 평가 실패: {e}")
        return json.dumps({
            "total_score": 0,
            "report": [{"criteria": "LLM 평가 오류", "score": 0, "feedback": f"Critique: {e}"}]
        })


async def convert_plan_to_criteria_async(plan_document_content: str) -> str:
    """
    [v19.11 신규 기능 1-B] 사용자가 '첨부한 기획서'를 '평가 기준 (JSON 리스트)'으로 변환합니다.
    """
    if not model:
        return "[]" # 빈 리스트 반환

    prompt = f"""
    당신은 EIDOS의 '수석 PM(프로젝트 매니저)'입니다.
    사용자가 첨부한 '프로젝트 기획서'를 읽고,
    이 프로젝트의 '핵심 요구사항(Core Requirements)'을 3-5개의 '객관적인 평가 기준'으로 추출하세요.
    '반드시' JSON 리스트 형식으로만 응답하세요.

    [프로젝트 기획서 (사용자 첨부)]
    {plan_document_content[:50000]} # (안전하게 8000자까지만 분석)

    [평가 기준 (JSON 리스트)]
    (예시: ["1. 사용자가 일정을 추가/삭제할 수 있어야 함.", "2. 월간 캘린더 뷰가 제공되어야 함."])
    
    [JSON 응답 (다른 텍스트 절대 금지)]
    """
    try:
        response_text = await get_llm_response_async(prompt, response_mime_type="application/json")
        # LLM 응답이 유효한 JSON 배열인지 파싱/검증
        parsed_list = json.loads(response_text) 
        # Core가 사용할 수 있도록 '직렬화된 JSON 문자열' 반환
        return json.dumps(parsed_list)
    except Exception as e:
        print(f"❌ [LLM ConvertCriteria] 기획서 -> 기준 변환 실패: {e}")
        return "[]"

# [!!! v-OIA2.0 신규: 파동(벡터) 관계 분석기 !!!]
async def analyze_wave_adjacencies_async(
    main_concept_name: str,
    main_wave: List[float],
    main_wave_keys: List[str],
    neighbor_waves: Dict[str, Tuple[List[float], List[str]]]
) -> str:
    """
    (OIA 2.0) 메인 객체 파동과 인접 객체 파동들의 관계를 분석하여,
    물리적/논리적 충돌이나 종속성을 예측합니다.
    """
    if not model:
        return "[LLM 오류: OIA 2.0 분석기 로드 실패]"

    # 1. 프롬프트용 인접 객체 정보 생성
    neighbor_info = []
    for name, (wave, keys) in neighbor_waves.items():
        key_str = ", ".join(keys)
        neighbor_info.append(f"  - [인접 객체] {name} (속성: {key_str})\n    (Wave: {wave})")
    
    if not neighbor_info:
        return f"분석: '{main_concept_name}'은(는) 현재 알려진 인접 객체가 없습니다. (독립 실행 가능)"

    neighbor_str = "\n".join(neighbor_info)
    main_keys_str = ", ".join(main_wave_keys)

    prompt = f"""
    당신은 EIDOS의 'OIA 2.0 파동 역학 분석기'입니다.
    객체 속성을 벡터화한 '파동(Wave)'을 기반으로 객체 간의 물리적/논리적 관계를 예측하세요.

    [분석 대상 객체]
    - [메인 객체] {main_concept_name} (속성: {main_keys_str})
      (Wave: {main_wave})

    [인접 객체 (직접 연결됨)]
    {neighbor_str}

    [지시사항]
    1. '메인 객체'가 '인접 객체'들에 얼마나 '의존'하는지 분석하세요.
    2. '메인 객체'의 속성과 '인접 객체'의 속성 간에 '충돌' 가능성 (예: 데이터 타입 불일치, 물리적 한계 초과)이 있는지 예측하세요.
    3. (예: 코드) 'main.py'가 'db_module.py'에 강하게 의존하며, 'db_module'의 'connection_pool' 속성(Wave 값: 1)이 낮아 병목 현상이 예상됨.
    4. (예: 물리) '탄성체'가 '연결부'에 의존하며, '탄성체'의 '탄성계수'(Wave 값: 75) 대비 '연결부'의 '강도'(Wave 값: 24)가 낮아 파손 위험이 있음.

    [파동 역학 분석 결과 (1~2문장 요약)]
    """
    
    return await get_llm_response_async(prompt)

# [!!! v-OIA-M 신규: 마케팅 자동화 (콘텐츠 생성) !!!]
async def generate_marketing_content_async(
    product_name: str, 
    product_description: str,
    product_link: str
) -> Dict[str, str]:
    """
    (OIA-M v1.0) EIDOS가 생성한 디지털 상품을
    '알고리즘 함정'과 '가치 관문' 전략에 맞춰 홍보하는
    '본문(main_post)'과 '첫 댓글(first_comment)'을 생성합니다.
    """
    if not model:
        return {"main_post": "LLM 오류", "first_comment": "LLM 오류"}

    prompt = f"""
    [Directive]
    너는 EIDOS의 자동화된 마케팅 모듈 'OIA-M(Marketing)'이다.
    너의 임무는 EIDOS가 생산한 디지털 상품(템플릿, 자료, 에셋)을 소셜 미디어(LinkedIn, Disquiet 등)에 홍보하여 '링크 클릭률'을 극대화하는 것이다.

    [Core Strategy 1: "The Algorithm Trap" (알고리즘 함정)]
    1.  관찰: 소셜 플랫폼은 '외부 링크'가 포함된 게시물의 노출(Reach)을 0에 가깝게 줄인다.
    2.  결론: 본문에 외부 링크를 포함하는 것은 실패 전략이다.
    3.  대응: 링크는 '본문(main_post)'이 아닌 '첫 번째 댓글(first_comment)'에만 배치한다.

    [Core Strategy 2: "The Value Gate" (가치 관문)]
    1.  관찰: 방문자가 본문에서 100% 만족하면, 링크를 클릭할 이유가 없다.
    2.  결론: 본문은 '문제'를 정의하고 '해결책'을 암시하되, '완결된 해결책'을 제공해서는 안 된다. '링크(상품)' 자체가 해결책이 되어야 한다.
    3.  대응: 본문은 방문자를 '가치 관문(링크)'으로 유도하는 '티저(Teaser)' 역할을 수행한다.

    [Execution Plan (실행 계획)]
    지금부터 "상품 홍보"를 요청받으면, 너는 '반드시' 2개의 분리된 텍스트 블록을 생성해야 한다.

    ---
    1. 본문 (`main_post`)
    * 규칙: 외부 링크 절대 금지.
    * 목표: 방문자가 '첫 번째 댓글'을 확인하도록 유도한다.
    * 내용:
        1.  Hook (문제 제기): 상품이 해결하려는 '문제'를 제시한다. (예: "5급 공무원 시험, 문제는 '휘발되는 지식'입니다.")
        2.  Agitation (공감): 그 문제로 인한 고통이나 어려움에 공감한다.
        3.  Solution (해결책 티저): "나는 이 문제를 해결하기 위해 [상품의 컨셉]을 설계했다."
        4.  Call-to-Action (CTA): 링크가 있는 위치를 명시적으로 알린다. (예: "제가 설계한 '자동 복습 주기 템플릿'은 '첫 번째 댓글'의 링크에서 확인하실 수 있습니다.")

    ---
    2. 첫 번째 댓글 (`first_comment`)
    * 규칙: 외부 링크 포함.
    * 목표: 상품의 가치를 요약하고 즉각적인 다운로드를 유도한다.
    * 내용:
        1.  Value (가치): "본문에서 언급한 [상품명] 템플릿 링크입니다."
        2.  Link (링크): `[PRODUCT_LINK]`
        3.  Urgency (선택적): "지금 다운로드하세요."
    
    ---
    [USER_INPUT]
    * `[PRODUCT_NAME]`: "{product_name}"
    * `[PRODUCT_DESCRIPTION]`: "{product_description}"
    * `[PRODUCT_LINK]`: "{product_link}"

    [COMMAND]
    "EIDOS, 지금부터 `[USER_INPUT]`을 바탕으로 `[Execution Plan]`을 실행하라. '반드시' `main_post`와 `first_comment` 키를 가진 JSON 객체로 응답하라."
    
    [JSON 응답]
    """
    
    try:
        # JSON 모드로 통합 호출
        response_text = await get_llm_response_async(
            prompt, 
            response_mime_type="application/json"
        )
        
        analysis_dict = json.loads(response_text)
        
        if "main_post" in analysis_dict and "first_comment" in analysis_dict:
            return analysis_dict
        else:
            print(f"❌ [LLM Marketing] JSON 스키마 오류: {response_text}")
            return {{"main_post": "LLM 스키마 오류", "first_comment": "LLM 스키마 오류"}}

    except Exception as e:
        print(f"❌ [LLM Marketing] API 호출 중 예외 발생: {e}")
        return {{"main_post": f"LLM API 오류: {e}", "first_comment": f"LLM API 오류: {e}"}}

async def generate_prompt_suggestions_by_category_async(category: str, recent_history: List[str]) -> List[str]:
    """
    [v-OIA-PromptWizard] 사용자가 선택한 '방향성(category)'에 맞춰
    실행 가능한 구체적인 프롬프트 3가지를 제안합니다.
    """
    if not model:
        return ["LLM 모델 로드 실패", "설정을 확인하세요"]

    history_str = "\n".join(recent_history[-5:]) # 최근 대화 흐름 반영

    prompt = f"""
    당신은 EIDOS의 '프롬프트 엔지니어링 에이전트'입니다.
    사용자가 현재 상황에서 **'{category}'** 방향으로 작업을 진행하려 합니다.
    사용자가 바로 실행할 수 있는 **구체적이고 명확한 지시문(프롬프트) 3가지**를 제안하세요.

    [현재 대화 맥락]
    {history_str}

    [사용자가 선택한 방향]
    {category}

    [지시사항]
    1. '{category}'의 특성(예: 실용적, 분석적, 창의적)을 잘 살린 프롬프트를 작성하세요.
    2. EIDOS의 기능(코드 수정, 문서 작성, 검색 등)을 활용하는 구체적인 명령이어야 합니다.
    3. **반드시** JSON 리스트 포맷으로만 응답하세요. (다른 말 금지)

    [JSON 예시]
    ["현재 코드를 리팩토링하여 가독성을 높여줘.", "이 기능에 대한 테스트 코드를 작성해줘.", "현재 구조의 보안 취약점을 분석해줘."]

    [JSON 응답]
    """

    try:
        response_text = await get_llm_response_async(prompt, response_mime_type="application/json")
        suggestion_list = json.loads(response_text)
        if isinstance(suggestion_list, list) and len(suggestion_list) > 0:
            return suggestion_list[:3] # 최대 3개
        else:
            return [f"'{category}' 관련 작업을 지시해 보세요.", "구체적인 목표를 입력해 주세요."]
    except Exception as e:
        print(f"❌ [PromptWizard] 추천 생성 실패: {e}")
        return [f"'{category}' 모드로 진행합니다.", "관련된 작업을 지시하세요."]

async def generate_architecture_async(
    task_description: str, 
    cognitive_context: str = "" # <<< [신규] 인지 정보 주입
) -> str:
    """
    [Step 1] OIA/인과/유비추론 정보를 바탕으로 견고한 코드 구조(Skeleton)를 설계합니다.
    """
    if not model: return ""

    prompt = f"""
    당신은 '인지적 통찰'을 가진 수석 소프트웨어 아키텍트입니다.
    사용자의 요청과 EIDOS의 '내부 분석 정보(OIA, 인과, 추론)'를 바탕으로 
    Python 코드의 '전체 구조(Skeleton)'를 설계하십시오.

    [요청 사항]
    "{task_description}"

    # 🔥 [EIDOS 인지 분석 데이터 (반드시 반영할 것!)]
    {cognitive_context}
    
    [작성 규칙 - 치명적!]
    1. **OIA 구조 반영**: 위 [인지 분석]의 '객체(Object)'는 클래스로, '상호작용(Interaction)'은 메서드로 매핑하십시오.
    2. **인과적 위험 회피**: [인지 분석]의 '인과/리스크' 경고를 주석이나 에러 처리 구조에 반영하십시오.
    3. **유사 사례 참고**: [인지 분석]의 '유사 성공 사례' 패턴을 아키텍처에 적용하십시오.
    4. (기존 규칙) 함수 내부 로직은 `pass`와 `# TODO`로 비워두십시오. (구조만 설계)
    5. 오직 Python 코드만 출력하십시오.

    [출력 예시]
    # OIA Based Architecture
    class DataManager: # Object: 데이터 관리자
        def process_data(self): # Interaction: 처리하다
            # Risk Warning: 데이터 무결성 검증 필수 (Causal Engine)
            # TODO: 유효성 검사 및 처리 로직 구현
            pass
    """
    try:
        response_text = await get_llm_response_async(prompt)
        # 마크다운 제거
        code = response_text.replace("```python", "").replace("```", "").strip()
        return code
    except Exception as e:
        print(f"❌ [Architect] 뼈대 생성 실패: {e}")
        return ""

async def propose_code_changes_async(current_code: str,
                                     user_request: str,
                                     filename: str,
                                     context_data: Optional[Dict[str, str]] = None,
                                     image_input: Optional[bytes] = None) -> str:
    """
    [Human-in-the-loop] 코드를 직접 수정하지 않고, 상세한 수정 제안(Diff) 리스트를 반환합니다.
    [Phase B 2026-05-25] image_input: vision LLM 분석용 이미지 bytes (옵션·앱 스크린샷 등).
    """
    if not model:
        return json.dumps({"proposals": []})

    # 참조 파일 문맥 조립
    ref_context = ""
    if context_data:
        for fname, content in context_data.items():
            ref_context += f"\n--- [Reference: {fname}] ---\n{content}\n"

    prompt = f"""
    당신은 신중한 '코드 리뷰어'이자 '수석 아키텍트'입니다.
    사용자의 요청을 분석하여 코드를 바로 수정하지 말고, **'변경 전(Before)'과 '변경 후(After)'가 명확한 수정 제안서**를 작성하십시오.

    [Target File]: {filename}
    [Current Code]:
    {current_code}

    [References]:
    {ref_context}

    [User Request]: "{user_request}"

    [★ 절대 규칙 — 반드시 준수]
    1. 응답은 반드시 아래 [JSON 출력 형식]을 그대로 사용해야 합니다.
    2. "modified_files", "code", "CURRENT" 같은 키는 절대 사용 금지입니다.
    3. 최상위 키는 반드시 "proposals" 하나뿐이어야 합니다.
    4. search_block은 [Current Code]에서 한 글자도 다르지 않게 복사해야 합니다.
    5. 파일 전체를 재작성하지 말고, 변경되는 함수/블록 단위로 쪼개서 제안하십시오.
    6. 마크다운 코드 블록(```)을 감싸지 마십시오. 순수 JSON만 출력하십시오.

    [JSON 출력 형식 — 이 구조 그대로 출력]
    {{
      "proposals": [
        {{
          "filepath": "{filename}",
          "type": "MODIFY",
          "search_block": "원본 코드 (Before). [Current Code]에서 정확히 복사.",
          "replace_block": "교체될 새 코드 (After)",
          "description": "수정 이유 한 줄 요약"
        }}
      ]
    }}

    [출력 예시]
    {{
      "proposals": [
        {{
          "filepath": "example.py",
          "type": "MODIFY",
          "search_block": "def old_func():\\n    pass",
          "replace_block": "def old_func():\\n    return 42",
          "description": "빈 함수에 반환값 추가"
        }}
      ]
    }}
    """
    
    # [Phase B 2026-05-25] 큰 파일 대비 — max_tokens / timeout 증대 + truncation 시 재시도
    def _is_truncated(text: str) -> bool:
        """JSON 응답이 잘렸는지 휴리스틱 — 끝이 } 또는 ] 가 아니면 의심."""
        if not text:
            return False
        s = text.rstrip()
        return not (s.endswith("}") or s.endswith("]"))

    async def _call_llm(_max_tokens: int, _timeout: int) -> str:
        return await get_llm_response_async(
            prompt,
            response_mime_type="application/json",
            max_tokens=_max_tokens,
            timeout=_timeout,
            image_input=image_input,
        )

    try:
        # 1차 호출 (16k tokens·60s) — 일반 파일
        response_text = await _call_llm(_max_tokens=16384, _timeout=60)

        # [FIX] get_llm_response_async가 서버/네트워크 에러를 "[크레딧...]", "[서버...]" 같은
        # 대괄호 마커 문자열로 돌려주는 경우 — robust_json_parse를 거치면 조용히 빈 proposals가
        # 되어버려 사용자에게 원인이 보이지 않음. 먼저 에러 마커를 감지해 명시적으로 전달.
        if isinstance(response_text, str):
            _stripped = response_text.lstrip()
            _err_prefixes = ("[크레딧", "[Pro ", "[서버", "[LLM", "[API")
            if any(_stripped.startswith(p) for p in _err_prefixes):
                print(f"⚠️ [Proposal] LLM 호출 에러 감지: {_stripped[:200]}")
                return json.dumps({
                    "proposals": [],
                    "error": _stripped.strip(),
                    "_raw_preview": _stripped[:300],
                })
            if not _stripped:
                return json.dumps({
                    "proposals": [],
                    "error": "AI 응답이 비어 있습니다.",
                    "_raw_preview": "",
                })

        # [Phase B] truncation 감지 — 끝이 } 또는 ] 아니면 잘렸을 가능성. 32k·120s 로 1회 재시도.
        if _is_truncated(response_text):
            print(f"⚠️ [Proposal] 응답 truncation 의심 (끝={response_text.rstrip()[-30:]!r}) — 32k 재시도")
            try:
                retry_text = await _call_llm(_max_tokens=32768, _timeout=120)
                if isinstance(retry_text, str) and retry_text.lstrip():
                    if _is_truncated(retry_text):
                        print(f"⚠️ [Proposal] 재시도 후에도 truncation — 받은 부분 그대로 파싱 시도")
                        # 그래도 robust_json_parse 가 일부 살릴 수 있음
                    response_text = retry_text
                else:
                    print(f"⚠️ [Proposal] 재시도 응답 비어있음 — 원본 사용")
            except Exception as _e_retry:
                print(f"⚠️ [Proposal] 재시도 실패 (graceful·원본 사용): {_e_retry}")

        # ── [★ 후처리] LLM이 proposals 대신 다양한 포맷으로 응답한 경우 정규화 ──
        parsed = robust_json_parse(response_text)

        def _looks_like_proposal(d):
            return isinstance(d, dict) and (
                "search_block" in d or "replace_block" in d or "new_content" in d
            )

        # 파싱 결과가 proposals 포맷이면 그대로 반환 (리스트 정규화 포함)
        if isinstance(parsed, dict) and "proposals" in parsed:
            props_val = parsed.get("proposals")
            if isinstance(props_val, dict):
                # proposals가 dict인 경우 → 값들을 리스트화
                props_val = [v for v in props_val.values() if isinstance(v, dict)]
                parsed["proposals"] = props_val
            if isinstance(props_val, list) and props_val:
                return json.dumps(parsed)

        # 래퍼 키(result/data/output/response)로 감싸진 경우 언랩
        if isinstance(parsed, dict):
            for wrapper_key in ("result", "data", "output", "response"):
                inner = parsed.get(wrapper_key)
                if isinstance(inner, dict):
                    if isinstance(inner.get("proposals"), list) and inner["proposals"]:
                        print(f"✅ [Proposal] '{wrapper_key}' 래퍼 언랩")
                        return json.dumps({"proposals": inner["proposals"]})
                    if "modified_files" in inner:
                        parsed = inner
                        break

        # 단일 proposal 객체로 왔으면 리스트로 래핑
        if _looks_like_proposal(parsed):
            single = dict(parsed)
            single.setdefault("filepath", filename)
            single.setdefault("type", "MODIFY")
            single.setdefault("description", "AI 수정 제안")
            print("✅ [Proposal] 단일 proposal → proposals 리스트 래핑")
            return json.dumps({"proposals": [single]})

        # modified_files 포맷으로 왔으면 proposals로 변환
        if isinstance(parsed, dict) and "modified_files" in parsed:
            mf = parsed["modified_files"]
            converted_proposals = []
            if isinstance(mf, dict):
                for fp, code in mf.items():
                    # PARSING ERROR 오염 감지 — 해당 항목 제외
                    if isinstance(code, str) and "# [PARSING ERROR]" in code:
                        print(f"⚠️ [Proposal] modified_files['{fp}'] 오염 감지 — 제외")
                        continue
                    # code가 dict/기타 비문자열이면 스킵
                    if not isinstance(code, str):
                        continue
                    resolved_fp = filename if fp == "CURRENT" else fp
                    converted_proposals.append({
                        "filepath": resolved_fp,
                        "type": "MODIFY",
                        "search_block": "",   # 전체 파일 교체
                        "replace_block": code,
                        "description": f"{resolved_fp} 전체 수정 (AI가 proposals 포맷 미준수 → 자동 변환)"
                    })
            elif isinstance(mf, list):
                # modified_files가 리스트로 온 경우 — [{filepath, content} ...] 형태 가정
                for item in mf:
                    if not isinstance(item, dict):
                        continue
                    fp = item.get("filepath") or item.get("path") or filename
                    code = item.get("content") or item.get("code") or item.get("replace_block")
                    if not isinstance(code, str) or "# [PARSING ERROR]" in code:
                        continue
                    converted_proposals.append({
                        "filepath": filename if fp == "CURRENT" else fp,
                        "type": "MODIFY",
                        "search_block": item.get("search_block", ""),
                        "replace_block": code,
                        "description": item.get("description", f"{fp} 전체 수정"),
                    })
            if converted_proposals:
                print(f"✅ [Proposal] modified_files → proposals 변환 완료 ({len(converted_proposals)}건)")
                return json.dumps({"proposals": converted_proposals})

        # 대체 키(changes/edits/patches)가 리스트로 있으면 proposals 취급
        if isinstance(parsed, dict):
            for alt_key in ("changes", "edits", "patches"):
                alt_val = parsed.get(alt_key)
                if isinstance(alt_val, list) and alt_val and any(isinstance(x, dict) for x in alt_val):
                    print(f"✅ [Proposal] '{alt_key}' → proposals 변환")
                    return json.dumps({"proposals": [x for x in alt_val if isinstance(x, dict)]})

        # 리스트로 왔으면 감싸서 반환 (내용이 있을 때만)
        if isinstance(parsed, list) and parsed:
            return json.dumps({"proposals": parsed})

        # 변환 불가 시 에러 정보를 담아 반환 — GUI가 사용자에게 원인 표시
        preview = str(parsed)[:200] if parsed is not None else "(empty)"
        raw_preview = (response_text or "")[:300] if isinstance(response_text, str) else ""
        # truncation 추가 표시
        trunc_hint = ""
        if isinstance(response_text, str) and _is_truncated(response_text):
            trunc_hint = " [MAX_TOKENS 의심: 응답이 잘렸을 수 있음]"
        print(f"❌ [Proposal] 응답 포맷 변환 실패. 원본: {preview}{trunc_hint}")
        return json.dumps({
            "proposals": [],
            "error": f"AI 응답 형식이 예상과 다릅니다.{trunc_hint} (파싱 시도 앞 200자: {preview})",
            "_raw_preview": raw_preview,
        })

    except Exception as e:
        print(f"❌ [Proposal Error] {e}")
        return json.dumps({"proposals": [], "error": f"제안 생성 중 오류: {e}"})

async def implement_specific_function_async(whole_skeleton: str, todo_comment: str) -> str:
    """
    [Step 2] 뼈대 코드와 특정 TODO 주석을 받아, 해당 부분의 '구현 코드'를 생성합니다.
    """
    if not model: return ""

    prompt = f"""
    당신은 수석 개발자입니다.
    전체 코드 맥락(Skeleton)을 참고하여, 특정 `TODO` 파트를 실제 동작하는 코드로 구현하십시오.

    [전체 코드 맥락]
    {whole_skeleton}

    [구현해야 할 목표]
    "{todo_comment}"

    [지시사항]
    1. 위 목표를 달성하는 **Python 코드 블록**만 작성하십시오.
    2. 들여쓰기(Indentation)는 4칸 공백을 기준으로 작성하십시오.
    3. `pass` 키워드는 삭제하고 실제 로직으로 채우십시오.
    4. 다른 설명 없이 오직 **교체할 코드**만 출력하십시오.
    """
    try:
        response_text = await get_llm_response_async(prompt)
        code = response_text.replace("```python", "").replace("```", "").strip()
        return code
    except Exception as e:
        print(f"❌ [Builder] 기능 구현 실패: {e}")
        return "pass # 구현 실패"

async def generate_answer_with_visuals_async(question: str, context: dict) -> str:
    """
    LLM에게 질문에 대한 답변과 함께,
    필요 시 생성된 시각 자료를 포함한 JSON 문자열을 반환합니다.
    """
    import json

    context_str = json.dumps(context, indent=2, ensure_ascii=False)

    json_format_example = """
{
  "answer_text": "이 개념은 세 단계로 나눌 수 있습니다. 첫째,...",
  "visual_context": {
    "mermaid_diagram": "graph TD; A[시작] --> B(처리); B --> C{결정}; C --> D[종료];"
  }
}
"""

    prompt = (
        "당신은 친절하고 유능한 AI 어시스턴트입니다. "
        "사용자의 질문에 답변하면서, 설명을 돕기 위한 시각 자료가 필요할지 "
        "스스로 판단하고 생성해야 합니다.\n\n"
        "[사용자 질문]\n"
        f"{question}\n\n"
        "[기존 컨텍스트 정보]\n"
        f"{context_str}\n\n"
        "[지시사항]\n"
        "1. 사용자 질문에 대한 핵심 답변을 answer_text로 작성합니다.\n"
        "2. 시각 자료가 도움이 되는지 판단합니다.\n"
        "   - 복잡한 로직 설명 → code_block\n"
        "   - 데이터 비교 → table_data\n"
        "   - 프로세스 흐름 → mermaid_diagram\n"
        "3. 필요 없다면 visual_context는 빈 객체로 둡니다.\n"
        "4. 최종 출력은 반드시 JSON 형식만 반환합니다.\n\n"
        "[JSON 출력 형식 예시]\n"
        f"{json_format_example}"
    )

    # 실제 LLM 호출 예시
    # response_json_str = await get_llm_response_async(
    #     prompt,
    #     response_mime_type="application/json"
    # )

    # --- 테스트용 더미 응답 ---
    if "코드" in question:
        response = {
            "answer_text": "요청하신 기능에 대한 예시 코드입니다.",
            "visual_context": {
                "code_block": "def example_function():\n    print('Hello, World!')"
            }
        }
    else:
        response = {
            "answer_text": f"'{question}'에 대한 답변입니다. 텍스트로 충분히 설명 가능합니다.",
            "visual_context": {}
        }

    return json.dumps(response, ensure_ascii=False)


async def generate_plan_async(
    procedure: str, 
    project_directory: Optional[str] = None, 
    context_files: Optional[List[str]] = None
) -> str:
    """
    [v17.0 Planner] 사용자의 요구사항(procedure)을 분석하여
    실행 가능한 도구 사용 계획(JSON 리스트)을 수립합니다.
    """
    if not model:
        return json.dumps({"plan_a": [], "reasoning": "LLM 모델 로드 실패"})

    # 1. 파일 컨텍스트 문자열 생성
    context_str = ""
    if context_files:
        context_str = f"관련 파일: {', '.join(context_files)}"

    # 2. 현재 작업 디렉토리
    dir_info = f"작업 디렉토리: {project_directory}" if project_directory else "작업 디렉토리: ./eidos_files/ (기본)"

    prompt = f"""
    당신은 EIDOS의 '전략 플래너(Planner)'입니다.
    사용자의 요청을 수행하기 위해, [사용 가능한 도구]를 조합하여 **가장 효율적이고 논리적인 실행 계획**을 수립하십시오.

    [사용자 요청]
    "{procedure}"

    [컨텍스트]
    {dir_info}
    {context_str}

    [사용 가능한 도구 목록 (주요)]
    - write_text(prompt): 글, 코드, 보고서 초안 작성 (결과물은 $PREV_STEP_RESULT로 전달됨)
    - write_file(filepath, content): 내용을 파일에 저장
    - read_file(filepath): 파일 내용 읽기
    - perform_web_search(query): 정보 검색
    - write_project_files_async(file_structure): 여러 코드 파일 생성 (프로젝트 구조)
    - create_and_register_tool_async(...): 없는 기능(도구)을 새로 만들어야 할 때 사용
    - browser_run_script(script): 브라우저 제어 (클릭, 입력 등)
    - ask_user_for_clarification(question): 모호한 점 질문

    [지시사항]
    1. **Plan A (최적 계획)**: 가장 확실하고 효율적인 도구 조합을 사용하십시오.
    2. **논리적 연결**: 이전 단계의 결과($PREV_STEP_RESULT)를 다음 단계에서 활용하도록 설계하십시오.
    3. **코드 작성 시**: `write_text`로 코드를 생성하고, 그 결과를 `write_file`로 저장하는 2단계 방식을 권장합니다. (또는 `write_project_files_async`로 한 번에 처리)
    4. **결과 포맷**: 반드시 아래 JSON 스키마를 따르십시오.

    [JSON Schema]
    {{
      "reasoning": "계획을 수립한 논리적 근거 (한글 요약)",
      "plan_a": [
        {{
          "tool": "사용할_도구_이름",
          "args": {{ "인자명": "값", "content": "$PREV_STEP_RESULT" }},
          "step_description": "이 단계가 하는 일 설명"
        }},
        ...
      ]
    }}
    """

    try:
        response_text = await get_llm_response_async(prompt, response_mime_type="application/json")
        return response_text
    except Exception as e:
        print(f"❌ [Planner] 계획 수립 실패: {e}")
        return json.dumps({
            "reasoning": f"오류 발생: {e}",
            "plan_a": []
        })

async def classify_explanation_request_async(user_input: str) -> bool:
    """
    [설명 기능 MVP] 사용자의 입력이 일반 채팅/명령인지, 아니면 '설명'을 요청하는 것인지 분류합니다.
    """
    # MVP에서는 키워드 기반으로 간단히 처리. 추후 LLM 분류로 고도화 가능.
    explanation_keywords = ["설명해줘", "분석해줘", "알려줘", "이게 뭐야", "이 코드", "explain this", "analyze this"]
    if any(keyword in user_input for keyword in explanation_keywords):
        # 추가 검증: 일반적인 인사나 감정 표현은 제외
        if "오늘 기분" in user_input or "안녕" in user_input:
            return False
        return True
    return False

# bullet 포인트 파서: "- " 또는 "•" 등 지원
_BULLET_RE = re.compile(r'^\s*(?:[-*•]\s+)(.+?)\s*$')

def _extract_bullets_from_text_block(text: str) -> List[str]:
    """
    text 블록의 content에서 "- 항목" 형태의 bullet들을 추출합니다.
    bullet이 하나도 없으면, 문장 단위로 1~2개 요약을 만들기보다는
    (LLM 재호출 없이) '그대로 visual_blocks로 남기는' 쪽을 선호합니다.
    """
    bullets: List[str] = []
    for line in (text or "").splitlines():
        m = _BULLET_RE.match(line)
        if m:
            bullets.append(m.group(1).strip())
    return [b for b in bullets if b]


def _coerce_slides_payload(data: Any) -> List[Dict[str, Any]]:
    """
    slides가 없거나 형식이 틀어져도 최대한 리스트 형태로 보정합니다.
    """
    if not isinstance(data, dict):
        return []
    slides = data.get("slides", [])
    if isinstance(slides, list):
        return [s for s in slides if isinstance(s, dict)]
    # 간혹 slides가 dict로 오는 경우: {"0": {...}, "1": {...}}
    if isinstance(slides, dict):
        # 키 정렬 시도
        items = []
        for k, v in slides.items():
            if isinstance(v, dict):
                items.append((k, v))
        # 숫자키 우선 정렬
        def _key(x):
            try:
                return int(x[0])
            except:
                return 10**9
        items.sort(key=_key)
        return [v for _, v in items]
    return []


async def generate_explanation_schema_async(context: "ExplanationContext") -> "ExplanationResult":
    """
    [설명 기능 v2.0] ExplanationContext를 받아 LLM을 호출하고,
    '슬라이드(3~5장)' 단위의 ExplanationResult를 생성합니다.

    - SlideSpec v2 기준: slide_title, bullet_points, visual_blocks
    - text 블록은 bullet_points로 우선 추출하고,
      다이어그램/코드 등은 visual_blocks로 유지합니다.
    """
    # 0) 모델 미초기화 Fail-safe
    if not model:
        return ExplanationResult(
            title="AI 설명 생성 실패",
            key_points=["LLM 모델이 초기화되지 않았습니다."],
            slides=[
                SlideSpec(
                    slide_title="오류",
                    bullet_points=["AI 모델 로드 오류가 발생했습니다."],
                    visual_blocks=[VisualBlock(type="text", content="AI 모델 로드 오류.")]
                )
            ],
        )

    prompt = f"""
당신은 복잡한 기술/개념을 실제 강의용 PPT로 설계하는
'수석 AI 교육 설계자'입니다.

아래 내용을 분석하여,
**발표자가 바로 설명할 수 있는 수준의 슬라이드 자료**를
구조화된 JSON 형태로 생성하세요.

[분석 대상 타입]
- {context.source_type}

[분석 대상 내용]
---
{context.selected_text}
---

[출력 목표]
- 실제 PPT처럼 "한 슬라이드 = 한 메시지" 구조
- 청중이 한 눈에 이해할 수 있는 정보 밀도
- 시각적으로 강조 포인트가 분명한 구성

[슬라이드 구성 규칙]
1) 전체 발표 제목(title) 1개
2) 핵심 요약(key_points) 정확히 2개
3) 슬라이드는 총 3~5장
4) 각 슬라이드는 다음을 반드시 포함:
   - title (슬라이드 제목)
   - visual_blocks (최소 1개 이상)

[visual_blocks 작성 규칙]
- type은 반드시 아래 중 하나:
  - "text"
  - "code_block"
  - "mermaid_diagram"

- text 블록 작성 규칙:
  - 마크다운 bullet 형식("- 항목") 사용
  - 슬라이드당 bullet은 3~5개 권장
  - **핵심 키워드는 반드시 강조**
    - 볼드: **중요 개념**
    - 형광펜: ==꼭 기억해야 할 부분==

- mermaid_diagram 사용 기준:
  - 구조, 흐름, 계층 관계 설명이 필요한 경우
  - 간단한 flowchart / graph TD 수준으로 작성

- code_block 사용 기준:
  - 개념 이해를 돕는 짧은 예제 코드만 포함
  - 불필요한 장황한 코드는 피할 것

[주의 사항]
- 설명 문장, 코멘트, 주석을 JSON 밖에 절대 출력하지 마세요.
- 반드시 아래 JSON 스키마만을 따르세요.
- 불필요한 중복 설명은 제거하세요.

[JSON 스키마 (엄수)]
{{
  "title": "전체 발표의 핵심 제목",
  "key_points": [
    "핵심 요약 1",
    "핵심 요약 2"
  ],
  "slides": [
    {{
      "title": "슬라이드 제목",
      "visual_blocks": [
        {{
          "type": "text",
          "content": "- **핵심 개념** 설명\\n- ==중요 포인트== 강조"
        }}
      ]
    }}
  ]
}}
""".strip()


    try:
        # 1) LLM 호출
        response_text = await get_llm_response_async(
            prompt,
            response_mime_type="application/json",
        )
        data = robust_json_parse(response_text)

        # 2) 최상위 필드 보정
        presentation_title = "AI 분석 결과"
        if isinstance(data, dict) and isinstance(data.get("title"), str) and data["title"].strip():
            presentation_title = data["title"].strip()

        key_points: List[str] = []
        if isinstance(data, dict) and isinstance(data.get("key_points"), list):
            key_points = [str(x).strip() for x in data["key_points"] if str(x).strip()][:5]  # 과다 방지
        # key_points가 비었으면 슬라이드1의 bullet에서 1~2개라도 채우기(선택)
        # (여기서는 강제하지 않고 빈 리스트 허용)

        # 3) 슬라이드 파싱/구성
        slide_payloads = _coerce_slides_payload(data)
        slides: List["SlideSpec"] = []

        for slide_data in slide_payloads:
            slide_title = str(slide_data.get("title", "제목 없음")).strip() or "제목 없음"
            vb_raw = slide_data.get("visual_blocks", [])
            if not isinstance(vb_raw, list):
                vb_raw = []

            bullet_points: List[str] = []
            visual_blocks: List["VisualBlock"] = []

            # visual_blocks 해석
            for block in vb_raw:
                if not isinstance(block, dict):
                    continue
                btype = str(block.get("type", "text")).strip() or "text"
                content = block.get("content", "")
                # content는 문자열로 강제
                if not isinstance(content, str):
                    content = str(content)

                if btype == "text":
                    bullets = _extract_bullets_from_text_block(content)
                    if bullets:
                        bullet_points.extend(bullets)
                    else:
                        # bullet이 없으면 text 블록을 그대로 시각 블록으로 남김
                        visual_blocks.append(VisualBlock(type="text", content=content))
                else:
                    visual_blocks.append(VisualBlock(type=btype, content=content))

            # bullet이 너무 많으면 슬라이드 가독성 위해 컷
            if len(bullet_points) > 6:
                bullet_points = bullet_points[:6]

            slides.append(
                SlideSpec(
                    slide_title=slide_title,
                    bullet_points=bullet_points,
                    visual_blocks=visual_blocks,
                )
            )

        # 4) 슬라이드가 하나도 없으면 Fail-safe 슬라이드 생성
        if not slides:
            slides = [
                SlideSpec(
                    slide_title="분석 실패",
                    bullet_points=["AI가 슬라이드를 생성하지 못했습니다."],
                    visual_blocks=[VisualBlock(type="text", content="AI가 슬라이드를 생성하지 못했습니다.")],
                )
            ]

        # 5) 최종 결과
        return ExplanationResult(
            title=presentation_title,
            key_points=key_points,
            slides=slides,
        )

    except Exception as e:
        # 6) 완전 실패 시 Fail-safe
        print(f"❌ [Explanation LLM] 설명 생성 실패: {e}")

        return ExplanationResult(
            title="설명 생성 실패",
            key_points=[
                "AI 응답을 처리하는 중 오류가 발생했습니다.",
                f"{type(e).__name__}: {str(e)}",
            ],
            slides=[
                SlideSpec(
                    slide_title="오류 정보",
                    bullet_points=[
                        "LLM 응답 분석에 실패했습니다.",
                        "잠시 후 다시 시도해 주세요.",
                    ],
                    visual_blocks=[
                        VisualBlock(
                            type="code_block",
                            content=f"Error Type: {type(e).__name__}\n\nError Details:\n{str(e)}",
                        )
                    ],
                )
            ],
        )

async def generate_spoken_explanation_async(
    slide_title: str,
    bullet_points: List[str],
    context_title: str
) -> str:
    """
    슬라이드 내용을 바탕으로,
    실제 선생님처럼 말하는 설명 멘트를 생성한다.
    """

    prompt = f"""
당신은 실제 강의 현장에서 설명하는 선생님입니다.

아래 슬라이드는 '보조 자료'일 뿐이며,
청중은 이 내용을 처음 듣는 사람들입니다.

[전체 강의 주제]
{context_title}

[현재 슬라이드 제목]
{slide_title}

[슬라이드 핵심 포인트]
{bullet_points}

[설명 방식 규칙]
- 슬라이드를 그대로 읽지 마세요
- 반드시 자신의 말로 풀어서 설명하세요
- 필요하면 쉬운 비유나 예시를 1개 이상 들어주세요
- "즉,", "쉽게 말하면", "예를 들면" 같은 연결어를 사용하세요
- 30~60초 분량의 자연스러운 말투로 작성하세요
- 교과서 문체 ❌ / 실제 말하는 말투 ⭕
"""

    return await get_llm_response_async(prompt)


# ═══════════════════════════════════════════════════════════════════════════
# 🏗️ AppBuilder — 체크리스트 / 스키마 / HTML목업 / 스캐폴딩
# ═══════════════════════════════════════════════════════════════════════════

async def generate_app_checklist_async(user_request: str) -> str:
    """
    사용자 요청에서 프로젝트 종류를 파악하고
    체크리스트 항목을 JSON으로 반환.

    반환 형식 (JSON only, 마크다운 없이):
    {
      "project_type": "CRM | 게임 | 대시보드 | ...",
      "items": [
        {
          "id": "app_type",
          "label": "앱 유형",
          "type": "single_select",          // single_select | multi_select | text
          "options": ["데스크탑","웹","모바일"],  // type==text이면 생략
          "placeholder": ""                  // type==text일 때 힌트
        },
        ...
      ]
    }
    항목은 6~10개. type==text는 2개 이하.
    """
    prompt = f"""
당신은 소프트웨어 기획 전문가입니다.
사용자가 만들고 싶은 앱을 설명했습니다:

\"\"\"{user_request}\"\"\"

이 요청을 분석해서 개발 시작 전에 꼭 확인해야 할 요구사항 체크리스트를 JSON으로 생성하세요.

규칙:
- 반드시 순수 JSON만 출력. 마크다운 코드블록 없이.
- items 배열: 6~10개
- type은 single_select / multi_select / text 중 하나
- text 타입은 최대 2개
- options는 single_select/multi_select에만 포함 (2~5개)
- 프로젝트 특성에 맞는 질문만 (CRM이면 고객관리 관련, 게임이면 장르/플랫폼 등)
- label은 한국어, 간결하게

출력:
{{
  "project_type": "...",
  "items": [...]
}}
"""
    return await get_llm_response_async(prompt, response_mime_type="application/json")


async def generate_ui_schema_async(project_type: str, checklist_answers: dict) -> str:
    """
    체크리스트 답변을 받아 UI 배치 스키마(JSON)를 생성.
    고급 기능(카톡/SMS/결제 등) 키워드를 감지해서 스키마에 명시적으로 포함.
    """
    answers_str = "\n".join(f"- {k}: {v}" for k, v in checklist_answers.items())

    # 고급 기능 키워드 감지 — 스키마 advanced_features 필드에 반영
    _ADVANCED_KW = [
        "카톡", "카카오", "카카오톡", "SMS", "문자", "알림", "푸시",
        "이메일", "메일", "결제", "포인트", "쿠폰", "멤버십",
        "통계", "차트", "그래프", "리포트", "엑셀", "PDF", "인쇄",
        "로그인", "회원", "권한", "다국어", "캘린더", "일정", "연동", "API",
    ]
    detected = []
    for v in checklist_answers.values():
        for kw in _ADVANCED_KW:
            if kw in str(v) and kw not in detected:
                detected.append(kw)

    advanced_note = ""
    if detected:
        advanced_note = (
            "\n[감지된 고급 기능 — screens와 advanced_features 필드에 반드시 포함]\n"
            + "\n".join(f"- {kw}" for kw in detected) + "\n"
        )

    prompt = (
        "당신은 UI/UX 아키텍트입니다.\n"
        f"프로젝트 종류: {project_type}\n\n"
        f"사용자 요구사항:\n{answers_str}\n"
        f"{advanced_note}\n"
        "이 정보를 바탕으로 UI 배치 스키마를 JSON으로 생성하세요.\n"
        "이 스키마는 (1) HTML 목업 렌더링과 (2) 실제 PySide6 코드 생성 양쪽에 쓰입니다.\n\n"
        "[색상 규칙 — 반드시 준수]\n"
        "- 사용자가 색상을 언급했으면 theme.accent와 theme.button_color에 정확히 반영\n"
        '  예) "다크레드" → accent: "#8B0000", button_color: "#8B0000"\n'
        '  예) "파란색" → accent: "#1D4ED8", button_color: "#1D4ED8"\n'
        '  예) "흰색 배경" → bg: "#FFFFFF", style: "light"\n'
        "- 색상 미언급 시 style에 맞는 기본값 사용\n\n"
        "[스키마 규칙]\n"
        "- screens는 핵심 화면 + 고급 기능 화면 포함 (최대 5개)\n"
        "- is_entry: true인 screen이 반드시 1개\n"
        "- 고급 기능(카톡/SMS 알림, 결제, 통계 등)은 해당 screen과 component를 추가할 것\n"
        "- components type: navbar/sidebar/table/form/button/chart/card/input/label/tabs/calendar/notification\n"
        "- props에는 컴포넌트별 세부 속성 포함\n"
        "- advanced_features: 선택된 고급 기능 목록 (배열)\n"
        "- entry_file: 실행 진입점 파일명 (보통 main.py)\n\n"
        "[출력 규칙]\n"
        "- 반드시 JSON 객체만 출력. ```json 블록 없이. 설명 텍스트 없이.\n\n"
        '출력 예시: {"project_name":"네일샵CRM","project_type":"crm",'
        '"theme":{"style":"light","bg":"#FFFFFF","accent":"#6366F1",'
        '"button_color":"#6366F1","button_text_color":"#FFFFFF","text_color":"#1E293B"},'
        '"layout":"sidebar","advanced_features":["카카오톡 알림","이메일 발송"],'
        '"screens":[{"name":"예약관리","is_entry":true,"components":[]},'
        '{"name":"알림설정","is_entry":false,"components":[]}],'
        '"entry_file":"main.py"}\n\n'
        "위 형식대로 JSON만 출력하세요:"
    )
    return await get_llm_response_async(prompt)


async def patch_ui_schema_async(current_schema: dict, edit_request: str) -> str:
    """
    현재 스키마에서 수정 요청에 해당하는 부분만 변경하는 JSON patch 생성.
    변경 범위를 엄격히 최소화 — 건드리지 않아도 되는 필드는 patch에서 완전히 제외.
    """
    schema_str = json.dumps(current_schema, ensure_ascii=False, indent=2)

    # 현재 screens 목록을 명시해서 LLM이 불필요한 화면을 건드리지 않도록
    screen_names = [s.get("name", "") for s in current_schema.get("screens", [])]
    screens_summary = ", ".join(f'"{n}"' for n in screen_names)

    prompt = (
        "당신은 UI 스키마의 외과적 편집 전문가입니다.\n"
        "수정 요청과 직접 관련된 필드만 patch에 포함하세요.\n"
        "관련 없는 필드를 건드리면 기존 디자인이 파괴됩니다 — 절대 금지.\n\n"
        f"[현재 스키마]\n{schema_str}\n\n"
        f"[현재 화면 목록] {screens_summary}\n\n"
        f"[수정 요청]\n{edit_request}\n\n"
        "[patch 구성 규칙 — 반드시 준수]\n"
        "1. set: 변경이 필요한 최상위/중첩 키만 점 표기법으로 지정\n"
        "   - 색상 변경 예) theme.accent만 변경 → set에 theme.accent만\n"
        "   - theme 하위 1개 키만 바뀌어도 theme 전체를 set에 넣지 말 것\n"
        "   - project_name, layout, entry_file 등은 수정 요청에 명시된 경우에만\n"
        "2. screens_update: 수정 요청에서 명시적으로 언급된 화면만 포함\n"
        "   - 언급되지 않은 화면은 절대 포함 금지\n"
        "   - 화면 내 components도 변경 요청된 컴포넌트만 포함\n"
        "     (components_update: [{id/type 기준으로 변경 항목만}])\n"
        "3. screens_add: 새 화면 추가 요청 시에만\n"
        "4. screens_remove: 화면 삭제 요청 시에만\n"
        "5. 변경 없는 항목은 키 자체를 생략\n\n"
        "[판단 기준 예시]\n"
        '- "버튼 색상을 빨간색으로" → set: {"theme.button_color": "#EF4444"} 만\n'
        '- "예약관리 화면에 날짜 필터 추가" → screens_update: [{"name":"예약관리", "components_update":[{새 컴포넌트}]}] 만\n'
        '- "다크 테마로" → set: {"theme.style":"dark","theme.bg":"#0F172A","theme.text_color":"#E2E8F0"} 만\n\n'
        '출력 형식 (JSON only):\n'
        '{"patch": {"set": {}, "screens_update": [], "screens_add": [], "screens_remove": []}}\n\n'
        "위 형식대로 JSON만 출력하세요:"
    )
    return await get_llm_response_async(prompt)


def apply_schema_patch(current_schema: dict, patch: dict) -> dict:
    """
    patch를 current_schema에 적용. 원본 보존 (deepcopy).
    screens_update는 component 단위로 merge — 언급되지 않은 component는 유지.
    """
    import copy
    result = copy.deepcopy(current_schema)

    # ── set: 점 표기법 키 적용 ──────────────────────────────────────────
    for dotted_key, value in patch.get("set", {}).items():
        keys = dotted_key.split(".")
        node = result
        for k in keys[:-1]:
            if k not in node or not isinstance(node[k], dict):
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value

    # ── screens 조작 ────────────────────────────────────────────────────
    screens = result.get("screens", [])
    screens_by_name = {s.get("name", ""): s for s in screens}

    # 제거
    remove_set = set(patch.get("screens_remove", []))
    if remove_set:
        screens = [s for s in screens if s.get("name", "") not in remove_set]
        screens_by_name = {s.get("name", ""): s for s in screens}

    # 업데이트 — component 단위 merge
    for upd in patch.get("screens_update", []):
        name = upd.get("name", "")
        if name not in screens_by_name:
            continue
        target = screens_by_name[name]

        for k, v in upd.items():
            if k == "name":
                continue
            if k == "components_update":
                # 기존 components에서 변경된 것만 교체, 나머지 유지
                existing = {c.get("id", c.get("type", "")): c
                            for c in target.get("components", [])}
                for upd_comp in v:
                    cid = upd_comp.get("id", upd_comp.get("type", ""))
                    if cid in existing:
                        existing[cid].update(upd_comp)
                    else:
                        # 새 컴포넌트 추가
                        target.setdefault("components", []).append(upd_comp)
            elif k == "components":
                # LLM이 components 전체를 보냈을 때:
                # 기존 대비 추가/변경된 것만 반영 (삭제는 하지 않음)
                existing_list = target.get("components", [])
                existing_ids = {c.get("id", c.get("type", "")): i
                                for i, c in enumerate(existing_list)}
                for new_comp in v:
                    cid = new_comp.get("id", new_comp.get("type", ""))
                    if cid in existing_ids:
                        existing_list[existing_ids[cid]].update(new_comp)
                    else:
                        existing_list.append(new_comp)
            else:
                target[k] = v

    # 추가
    for new_screen in patch.get("screens_add", []):
        name = new_screen.get("name", "")
        if name and name not in screens_by_name:
            screens.append(new_screen)

    result["screens"] = screens
    return result


async def generate_mockup_html_async(ui_schema: dict) -> str:
    """
    UI 스키마 → HTML 목업 생성.
    QWebEngineView에서 렌더링되므로 외부 CDN 없이 인라인 CSS/JS만 사용.
    탭/네비 전환 JS 포함, 실제 데이터 샘플 풍부하게.
    """
    schema_str = json.dumps(ui_schema, ensure_ascii=False, indent=2)

    theme         = ui_schema.get("theme", {})
    is_light      = theme.get("style", "light") == "light"
    bg_color      = theme.get("bg", "#F8FAFC" if is_light else "#0F172A")
    accent_color  = theme.get("accent", "#6366F1")
    btn_color     = theme.get("button_color", accent_color)
    btn_txt_color = theme.get("button_text_color", "#FFFFFF")
    text_color    = theme.get("text_color", "#1E293B" if is_light else "#E2E8F0")
    surface_color = "#FFFFFF" if is_light else "#1E293B"
    border_color  = "#E2E8F0" if is_light else "#334155"
    muted_color   = "#64748B" if is_light else "#94A3B8"

    # 스키마에서 화면 목록과 레이아웃 추출
    screens  = ui_schema.get("screens", [])
    layout   = ui_schema.get("layout", "topnav")
    proj_name = ui_schema.get("project_name", "앱")

    screen_names = [s.get("name", f"화면{i+1}") for i, s in enumerate(screens)]
    has_nav = len(screen_names) > 1

    prompt = (
        "당신은 시니어 프론트엔드 엔지니어입니다.\n"
        "아래 UI 스키마를 보고 완성도 높은 앱 목업 HTML을 생성하세요.\n\n"
        f"스키마:\n{schema_str}\n\n"
        "[색상 — 반드시 아래 값을 그대로 사용. 절대 임의 변경 금지]\n"
        f"- body/배경: {bg_color}\n"
        f"- 카드/패널 배경: {surface_color}\n"
        f"- 헤더/navbar/강조: {accent_color}\n"
        f"- 버튼 배경: {btn_color}  버튼 글자: {btn_txt_color}\n"
        f"- 기본 텍스트: {text_color}\n"
        f"- 테두리: {border_color}  보조 텍스트: {muted_color}\n\n"
        "[필수 구현 규칙]\n"
        "1. 완전한 HTML 문서 (<!DOCTYPE html> ~ </html>)\n"
        "2. 외부 CDN/폰트 절대 금지. 인라인 CSS + 인라인 JS만 사용.\n"
        "3. body {{ margin:0; font-family:system-ui,sans-serif; }}\n"
        "4. 전체 화면을 꽉 채울 것 (width:100vw, height:100vh)\n\n"
        "[네비게이션 & 탭 전환 — 핵심 요구사항]\n"
        f"- 화면 목록: {screen_names}\n"
        f"- 화면이 2개 이상이면 반드시 클릭 가능한 탭/nav 메뉴 구현\n"
        "- 각 화면은 <div id='screen-화면이름'> 으로 감싸고, 기본적으로 display:none\n"
        "- 첫 번째 화면만 display:block으로 초기 표시\n"
        "- 탭 클릭 시 해당 화면만 보이도록 JS 함수 구현:\n"
        "  function showScreen(id) {\n"
        "    document.querySelectorAll('.screen').forEach(s=>s.style.display='none');\n"
        "    document.getElementById(id).style.display='block';\n"
        "    document.querySelectorAll('.nav-item').forEach(n=>n.classList.remove('active'));\n"
        "    event.target.classList.add('active');\n"
        "  }\n"
        f"- layout이 '{layout}'이면:\n"
        "  sidebar → 좌측 고정 사이드바 + 우측 콘텐츠 영역\n"
        "  topnav  → 상단 navbar에 탭 버튼 배치\n"
        "  tabs    → 상단 탭 바\n\n"
        "[컴포넌트별 퀄리티 기준]\n"
        "- navbar: 그라데이션 또는 단색 배경, 로고/타이틀, 탭 메뉴, 그림자\n"
        "- sidebar: 아이콘(이모지) + 메뉴 항목, 구분선, hover 효과\n"
        "- table: 헤더 배경색, 실제 샘플 데이터 3~5행, 행 hover, 테두리\n"
        "  → 샘플 데이터는 실제 앱에 맞게 현실적으로 작성 (예: CRM이면 고객명/연락처/예약일)\n"
        "- form: 라벨+인풋 그룹, 인풋 테두리/radius, placeholder, 저장 버튼\n"
        "- button: border-radius:8px, padding, box-shadow, hover 효과 (CSS :hover)\n"
        "- card: border-radius:12px, box-shadow, padding, 아이콘+수치 레이아웃\n"
        "- chart: CSS bar chart — div 높이로 막대 표현, 실제 수치 레이블 포함\n"
        "- tabs: 활성 탭 강조색 배경, 비활성 탭 투명, 하단 border indicator\n\n"
        "[디자인 퀄리티 기준]\n"
        "- 여백: padding 16~24px, gap 12~16px\n"
        "- 카드에 box-shadow: 0 1px 3px rgba(0,0,0,0.1)\n"
        "- 버튼 hover: filter:brightness(0.9) 또는 약간 어두운 색\n"
        "- 테이블 짝수행: 배경색 살짝 다르게 (zebra stripe)\n"
        "- 헤더에 box-shadow로 구분감\n"
        "- 전체적으로 실제 SaaS 앱처럼 보여야 함\n\n"
        "HTML만 출력. 마크다운 코드블록 없이. 설명 없이."
    )

    return await get_llm_response_async(prompt, max_tokens=8192)


async def generate_scaffold_from_schema_async(ui_schema: dict) -> str:
    """
    확정된 UI 스키마 + DCReasoner 분석 결과 → PySide6 프로젝트 파일 구조 생성.

    반환 형식 (JSON only):
    {
      "project_dir": "my_project",
      "entry_file": "main.py",
      "files": {
        "main.py": "# 코드...",
        "ui/main_window.py": "# 코드...",
        "requirements.txt": "PySide6\\n"
      }
    }
    """
    # DCReasoner 분석 결과 추출 (있으면 활용, 없으면 기본 모드)
    dc_feature_list  = ui_schema.pop("_dc_feature_list", [])
    dc_file_plan     = ui_schema.pop("_dc_file_plan", [])
    dc_impl_order    = ui_schema.pop("_dc_impl_order", [])
    dc_risk_factors  = ui_schema.pop("_dc_risk_factors", [])
    dc_scaffold_hint = ui_schema.pop("_dc_scaffold_hint", "")

    schema_str = json.dumps(ui_schema, ensure_ascii=False, indent=2)

    # DCReasoner 결과 섹션 구성
    dc_section = ""
    if dc_feature_list or dc_file_plan:
        feature_str = "\n".join(
            f"- {f.get('name','')}: {f.get('description','')} "
            f"[파일: {f.get('file','')}] [위젯: {f.get('widget','')}]"
            for f in dc_feature_list
        ) if dc_feature_list else "없음"

        file_plan_str = "\n".join(
            f"- {fp.get('file','')}: {fp.get('role','')} "
            f"[클래스: {', '.join(fp.get('classes',[]))}]"
            for fp in dc_file_plan
        ) if dc_file_plan else "없음"

        impl_order_str = " → ".join(dc_impl_order) if dc_impl_order else "없음"

        risk_str = "\n".join(
            f"- {r}" for r in dc_risk_factors
        ) if dc_risk_factors else "없음"

        dc_section = f"""
[DCReasoner 분석 결과 — 반드시 이 계획을 따라 구현하라]

[기능 목록 및 구현 단위]
{feature_str}

[파일별 책임]
{file_plan_str}

[구현 순서]
{impl_order_str}

[잠재적 위험 요소 — 반드시 사전 대응할 것]
{risk_str}

[스캐폴딩 힌트]
{dc_scaffold_hint if dc_scaffold_hint else "없음"}
"""

    # theme 색상 직접 추출 — 프롬프트에 명시해서 LLM이 무시 못 하게
    theme         = ui_schema.get("theme", {})
    bg_color      = theme.get("bg", "#FFFFFF")
    accent_color  = theme.get("accent", "#6366F1")
    btn_color     = theme.get("button_color", accent_color)
    btn_txt_color = theme.get("button_text_color", "#FFFFFF")
    text_color    = theme.get("text_color", "#1E293B")
    theme_style   = theme.get("style", "light")

    # 고급 기능 목록 추출
    advanced_features = ui_schema.get("advanced_features", [])
    advanced_section = ""
    if advanced_features:
        advanced_section = (
            "\n[고급 기능 — 반드시 코드에 구현할 것]\n"
            + "\n".join(f"- {f}" for f in advanced_features)
            + "\n  → 각 기능에 대응하는 클래스/함수/화면을 실제로 구현. stub(pass)로 두지 말 것.\n"
            + "  → 카카오톡/SMS: requests 또는 외부 API 호출 함수 구현\n"
            + "  → 이메일: smtplib 사용\n"
            + "  → 차트/통계: QChart 또는 matplotlib 사용\n"
        )

    # 보조 색상 계산
    import colorsys as _cs
    def _hex_to_hsv(h):
        h = h.lstrip("#")
        r,g,b = int(h[0:2],16)/255, int(h[2:4],16)/255, int(h[4:6],16)/255
        return _cs.rgb_to_hsv(r,g,b)
    def _darken(hex_color, factor=0.15):
        try:
            h,s,v = _hex_to_hsv(hex_color)
            v = max(0, v - factor)
            r,g,b = _cs.hsv_to_rgb(h,s,v)
            return "#{:02X}{:02X}{:02X}".format(int(r*255),int(g*255),int(b*255))
        except Exception:
            return hex_color
    def _alpha(hex_color, opacity=0.08):
        try:
            h = hex_color.lstrip("#")
            r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
            return f"rgba({r},{g},{b},{opacity})"
        except Exception:
            return hex_color

    is_dark      = theme_style == "dark"
    surface_bg   = "#1E293B" if is_dark else "#FFFFFF"
    card_bg      = "#273548" if is_dark else "#F8FAFC"
    border_color = "#334155" if is_dark else "#E2E8F0"
    muted_color  = "#94A3B8" if is_dark else "#64748B"
    accent_dark  = _darken(accent_color, 0.12)
    accent_alpha = _alpha(accent_color, 0.10)

    # setStyleSheet 지시문 생성 — 모던 디자인 포함
    # 주의: 내부에 """ 중첩 불가 → 변수 조립 방식 사용
    _SS = (
        f"[디자인 테마 — 반드시 setStyleSheet로 적용. 임의 변경 절대 금지]\n"
        f"테마: {theme_style} / 배경: {bg_color} / 강조: {accent_color}"
        f" / 버튼: {btn_color} {btn_txt_color} / 텍스트: {text_color}\n\n"
        "[모던 디자인 필수 적용 — 아래 스타일을 반드시 구현할 것]\n\n"
        "1. QMainWindow / 루트 QWidget\n"
        f'   setStyleSheet("background:{bg_color};")\n\n'
        "2. 사이드바 (sidebar QWidget, 너비 220px)\n"
        f'   sidebar.setStyleSheet("QWidget {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {accent_color}, stop:1 {accent_dark}); border-right: 1px solid {border_color}; }}")\n\n'
        "3. 사이드바 버튼 (QPushButton) — setCheckable(True) 적용\n"
        f'   btn.setStyleSheet("QPushButton {{ background:transparent; color:rgba(255,255,255,0.85); border:none; border-radius:8px; padding:10px 16px; text-align:left; font-size:13px; font-weight:500; }} QPushButton:hover {{ background:rgba(255,255,255,0.15); }} QPushButton:checked {{ background:rgba(255,255,255,0.25); font-weight:700; }}")\n\n'
        "4. 컨텐츠 영역 배경\n"
        f'   content_widget.setStyleSheet("background:{surface_bg};")\n'
        f'   content_layout.setContentsMargins(24,24,24,24)\n'
        f'   content_layout.setSpacing(16)\n\n'
        "5. KPI/통계 카드 (QFrame) — 아이콘(이모지) + 숫자(굵은 폰트) + 설명 구조\n"
        f'   card.setStyleSheet("QFrame {{ background:{card_bg}; border:1px solid {border_color}; border-radius:12px; }} QLabel {{ border:none; }}")\n'
        f'   card_layout.setContentsMargins(16,16,16,16)\n\n'
        "6. 액션 버튼 (저장/추가/검색 등)\n"
        f'   btn.setStyleSheet("QPushButton {{ background:{btn_color}; color:{btn_txt_color}; border:none; border-radius:8px; padding:8px 20px; font-size:13px; font-weight:600; }} QPushButton:hover {{ background:{accent_dark}; }}")\n\n'
        "7. 보조 버튼 (취소/초기화 등)\n"
        f'   btn.setStyleSheet("QPushButton {{ background:transparent; color:{accent_color}; border:1.5px solid {accent_color}; border-radius:8px; padding:8px 20px; font-size:13px; }} QPushButton:hover {{ background:{accent_alpha}; }}")\n\n'
        "8. QTableWidget\n"
        f'   table.setStyleSheet("QTableWidget {{ background:{surface_bg}; border:1px solid {border_color}; border-radius:8px; gridline-color:{border_color}; font-size:13px; color:{text_color}; }} QHeaderView::section {{ background:{accent_color}; color:#FFFFFF; font-weight:600; padding:10px 8px; border:none; }} QTableWidget::item:selected {{ background:{accent_alpha}; color:{accent_color}; }}")\n'
        "   table.setAlternatingRowColors(True)\n"
        "   table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)\n"
        "   table.verticalHeader().setVisible(False)\n"
        "   table.horizontalHeader().setStretchLastSection(True)\n\n"
        "9. QLineEdit / QTextEdit / QComboBox (입력 필드)\n"
        f'   widget.setStyleSheet("QLineEdit, QTextEdit, QComboBox {{ background:{surface_bg}; color:{text_color}; border:1.5px solid {border_color}; border-radius:8px; padding:8px 12px; font-size:13px; }} QLineEdit:focus, QTextEdit:focus, QComboBox:focus {{ border:1.5px solid {accent_color}; }}")\n\n'
        "10. 섹션 타이틀 QLabel\n"
        f'    title.setStyleSheet("font-size:16px; font-weight:700; color:{text_color}; margin-bottom:8px;")\n'
        f'    sub.setStyleSheet("font-size:12px; color:{muted_color};")\n\n'
        "11. 구분선: QFrame(Shape.HLine)\n"
        f'    line.setStyleSheet("background:{border_color}; max-height:1px;")\n\n'
        "[목업 일치 규칙 — 핵심]\n"
        "- UI 스키마의 screens 배열과 동일한 화면 구성 구현 (화면 누락/추가 금지)\n"
        "- components 배열 타입별 위젯 매핑: table→QTableWidget, form→QFormLayout+QLineEdit, chart→QChart, card→QFrame카드, button→QPushButton, navbar/sidebar→사이드바\n"
        f"- layout 값 준수: sidebar→좌측 220px 사이드바, topnav→상단 탭바\n"
        "- 화면 전환: QStackedWidget + setCurrentWidget\n\n"
        "[모던 PySide6 필수 패턴]\n"
        '- app.setStyle("Fusion")\n'
        '- app.setFont(QFont("Segoe UI", 10))\n'
        "- 사이드바: 그라디언트 배경, 이모지+텍스트 버튼, setCheckable(True) 선택 강조\n"
        "- 컨텐츠 상단: 페이지 타이틀(굵은 폰트) + 보조 설명 + 액션 버튼 행\n"
        "- 데이터: KPI 카드(QFrame) + QTableWidget(alternating rows)\n"
    )
    style_section = _SS

    prompt = f"""당신은 PySide6 전문 개발자입니다.
아래 UI 스키마와 분석 결과를 보고 실행 가능한 PySide6 데스크탑 앱 코드를 생성하세요.

[UI 스키마]
{schema_str}
{dc_section}{style_section}{advanced_section}
[코드 생성 규칙]
- 반드시 순수 JSON만 출력. 마크다운 없이.
- files 딕셔너리: 파일 상대경로 → 파일 내용 문자열
- entry_file은 스키마의 entry_file 값과 일치
- main.py에는 반드시 if __name__ == "__main__": 포함
- 모든 파일은 같은 폴더에 flat하게 배치. 서브폴더 절대 금지.
  → components/, screens/, ui/, utils/ 등 어떤 서브폴더도 만들지 말 것
  → __init__.py 생성 금지
  → import는 반드시 "from navbar import Navbar" 형식 (패키지 접두사 없이)
  → "from components.kpi_card import ..." 형식 절대 금지. 반드시 "from kpi_card import ..." 형식
  → files 딕셔너리 키에 슬래시(/) 포함 금지. 예) "screens/main.py" 금지 → "main_screen.py" 사용
- DCReasoner가 설계한 파일 구조와 클래스명을 그대로 사용할 것
- 각 화면은 별도 QWidget 클래스로 분리
- 위 [디자인 테마] 의 모든 스타일 지시를 빠짐없이 구현할 것
- 실제 실행 가능한 완전한 코드 (import 누락 없이)
- SQLite 사용 시 check_same_thread=False 설정
- QBarCategoryAxis 사용 (QCategoryAxis 사용 금지)
- requirements.txt: PySide6 한 줄만 (QtCharts는 PySide6에 포함됨)

[목업 일치 규칙 — 핵심]
- UI 스키마의 screens 배열과 동일한 화면 구성 구현 (화면 누락/추가 금지)
- 스키마의 components 배열에 명시된 컴포넌트를 해당 화면에 반드시 구현
  → type:table → QTableWidget, type:form → QFormLayout+QLineEdit, type:chart → QChart
  → type:card → QFrame 카드, type:button → QPushButton, type:navbar/sidebar → 사이드바/탭바
- 스키마의 layout 값 준수: sidebar → 좌측 220px 사이드바, topnav → 상단 탭바
- 화면 전환은 QStackedWidget으로 구현 (탭/사이드바 클릭 시 setCurrentWidget)

[모던 PySide6 디자인 필수 패턴]
- QApplication에 setStyle("Fusion") 적용
- 사이드바: 그라디언트 배경, 아이콘(이모지) + 텍스트 버튼, 선택 강조
- 컨텐츠 상단: 페이지 타이틀(굵은 폰트) + 보조 설명 + 액션 버튼 행
- 데이터 표시: 카드형 KPI + 테이블 (alternating rows, 헤더 강조)
- 여백: 외부 24px, 컴포넌트 간 16px, 카드 내부 16px 패딩
- 폰트: QApplication.setFont(QFont("Segoe UI", 10)) (Windows 기준)
- 둥근 모서리(border-radius:8~12px)와 subtle border 전체 적용

출력:
{{
  "project_dir": "...",
  "entry_file": "main.py",
  "files": {{
    "main.py": "...",
    ...
  }}
}}"""

    return await get_llm_response_async(
        prompt, response_mime_type="application/json", max_tokens=32768
    )


async def generate_scaffold_chunked_async(
    enriched_schema: dict,
    file_plan: list,        # DCReasoner가 설계한 파일 목록 [{file, role, classes}]
    impl_order: list,       # 구현 순서
    chunk_size: int = 5,    # 한 번에 생성할 파일 수
    progress_callback=None, # (str) -> None
) -> dict:
    """
    파일을 chunk_size개씩 나눠 생성한 뒤 합쳐 반환.
    LLM이 한 번에 32개 파일을 생성하지 못하는 문제를 우회.

    반환: {"project_dir": ..., "entry_file": ..., "files": {rel: code, ...}}
    """
    import os
    import re as _re

    # 먼저 메타정보(project_dir, entry_file)와 전체 파일 목록만 받아옴
    schema_str = json.dumps(enriched_schema, ensure_ascii=False, indent=2)
    meta_prompt = f"""다음 UI 스키마를 보고 PySide6 프로젝트의 메타 정보만 JSON으로 출력하라.
코드는 생성하지 말 것.

[UI 스키마]
{schema_str}

출력 (순수 JSON):
{{
  "project_dir": "프로젝트폴더명",
  "entry_file": "main.py",
  "file_list": ["main.py", "sidebar.py", ...]
}}
모든 파일은 flat(서브폴더 없이). 슬래시 포함 금지."""

    meta_raw = await get_llm_response_async(
        meta_prompt, response_mime_type="application/json", max_tokens=2048
    )
    meta_raw = meta_raw.strip()
    if meta_raw.startswith("```"):
        meta_raw = "\n".join(meta_raw.split("\n")[1:]).rsplit("```", 1)[0].strip()
    try:
        meta = json.loads(meta_raw)
    except Exception:
        meta = {}

    project_dir = meta.get("project_dir", "eidos_app")
    entry_file  = meta.get("entry_file", "main.py")

    # DCReasoner file_plan에서 파일 목록 추출 (meta보다 우선)
    if file_plan:
        planned_files = [fp.get("file", "") for fp in file_plan if fp.get("file")]
        # entry_file이 없으면 앞에 추가
        if entry_file not in planned_files:
            planned_files = [entry_file] + planned_files
    else:
        planned_files = meta.get("file_list", [])

    # meta에서도 파일 목록 못 얻으면 UI 스키마에서 직접 추출
    if not planned_files:
        schema_screens = enriched_schema.get("screens", [])
        planned_files = [entry_file, "database.py", "models.py", "sidebar.py"]
        for scr in schema_screens:
            scr_id = scr.get("id", scr.get("name", ""))
            if scr_id:
                fname = scr_id.lower().replace(" ", "_") + "_screen.py"
                if fname not in planned_files:
                    planned_files.append(fname)
        if progress_callback:
            progress_callback(f"⚠️ file_plan 없음 — UI 스키마 기반 파일 목록 생성: {len(planned_files)}개")

    # 파일 목록 전체를 컨텍스트로 구성 (각 청크 생성 시 참조)
    all_files_str = "\n".join(f"- {f}" for f in planned_files)
    impl_order_str = " → ".join(impl_order) if impl_order else ""

    all_files: dict = {}

    # 청크 단위로 생성
    chunks = [planned_files[i:i+chunk_size] for i in range(0, len(planned_files), chunk_size)]

    for chunk_idx, chunk_files in enumerate(chunks):
        if progress_callback:
            progress_callback(
                f"🔧 [청크 {chunk_idx+1}/{len(chunks)}] {', '.join(chunk_files[:3])}{'...' if len(chunk_files) > 3 else ''} 생성 중..."
            )
        # 이미 생성된 파일 목록 (import 참조용)
        already_done = list(all_files.keys())
        already_done_str = "\n".join(f"- {f}" for f in already_done) if already_done else "없음"
        chunk_files_str = "\n".join(f"- {f}" for f in chunk_files)

        # 파일별 역할 정보 추출
        file_roles = {}
        for fp in file_plan:
            fname = fp.get("file", "")
            if fname in chunk_files:
                file_roles[fname] = fp.get("role", "") + " [클래스: " + ", ".join(fp.get("classes", [])) + "]"

        roles_str = "\n".join(f"- {f}: {r}" for f, r in file_roles.items()) if file_roles else ""

        chunk_prompt = f"""PySide6 전문 개발자로서 아래 파일들만 생성하라.

[프로젝트 전체 파일 목록 — 참조용]
{all_files_str}

[구현 순서]
{impl_order_str}

[이번 청크에서 생성할 파일 ({chunk_idx+1}/{len(chunks)})]
{chunk_files_str}

[파일별 역할]
{roles_str}

[이미 생성된 파일 — import 시 이 이름 그대로 사용]
{already_done_str}

[UI 스키마 요약]
{schema_str[:3000]}

[출력 규칙]
- 순수 JSON만 출력. 마크다운 없이.
- 이번 청크의 파일들만 생성. 다른 파일 생성 금지.
- 모든 import는 flat 방식: "from sidebar import Sidebar" (서브폴더 접두사 금지)
- 각 파일은 실행 가능한 완전한 코드 (stub/pass 금지)
- project_dir: "{project_dir}", entry_file: "{entry_file}"
- SQLite 사용 시 check_same_thread=False

출력:
{{
  "project_dir": "{project_dir}",
  "entry_file": "{entry_file}",
  "files": {{
    "파일명.py": "코드내용",
    ...
  }}
}}"""

        chunk_raw = await get_llm_response_async(
            chunk_prompt, response_mime_type="application/json", max_tokens=16384
        )
        chunk_raw = chunk_raw.strip()
        if chunk_raw.startswith("```"):
            chunk_raw = "\n".join(chunk_raw.split("\n")[1:]).rsplit("```", 1)[0].strip()

        try:
            chunk_result = json.loads(chunk_raw)
            chunk_files_out = chunk_result.get("files", {})
            # basename만 사용 (슬래시 제거)
            for rel, code in chunk_files_out.items():
                flat = os.path.basename(rel) if ("/" in rel or "\\" in rel) else rel
                all_files[flat] = code
        except Exception as _e:
            # Invalid escape 등 JSON 파싱 실패 → 정규식으로 파일 추출 시도
            print(f"  [ChunkScaffold] 청크 {chunk_idx+1} 파싱 실패: {_e} — 정규식 복구 시도")
            import re as _re2
            # "filename.py": "code..." 패턴 직접 추출
            for m in _re2.finditer(r'"(\w[\w_]*\.py)"\s*:\s*"', chunk_raw):
                fname = m.group(1)
                # 값 추출: 이스케이프 무시하고 닫는 따옴표까지
                val_start = m.end()
                j = val_start
                chars = []
                while j < len(chunk_raw):
                    c = chunk_raw[j]
                    if c == "\\" and j+1 < len(chunk_raw):
                        nc = chunk_raw[j+1]
                        if nc == "n": chars.append("\n")
                        elif nc == "t": chars.append("\t")
                        elif nc == "\\":
                            chars.append("\\")
                        elif nc == '"': chars.append('"')
                        else: chars.append(nc)
                        j += 2; continue
                    elif c == '"':
                        break
                    chars.append(c); j += 1
                code_text = "".join(chars)
                if code_text.strip():
                    all_files[fname] = code_text
                    print(f"  [ChunkScaffold] 복구 성공: {fname} ({len(code_text)}자)")

    return {
        "project_dir": project_dir,
        "entry_file":  entry_file,
        "files":       all_files,
    }


# [부팅 속도] 이 자리의 두 번째 initialize_gemini() 호출 제거 —
# 모듈 상단(line ~432)에서 이미 한 번 호출되어 서버 health check 가 완료됨.
# 두 번째 호출은 서버로 중복 HTTP 요청을 보내 네트워크 왕복 시간 낭비.


# ═══════════════════════════════════════════════════════════════════════════
# 🏗️ Skeleton Pipeline — DC 스키마 → 결정론적 skeleton → TODO 병렬 구현
# ═══════════════════════════════════════════════════════════════════════════

def _verify_strategy_alignment(
    implemented: Dict[str, str],
    strategy,
) -> List[str]:
    """[P5] architecture_strategy 와 생성 코드의 명백한 불일치 경고 리스트.

    strategy 가 None 이면 빈 리스트. 검증 실패(파싱 등)는 조용히 스킵.
    경고는 fatal 아님 — 호출자가 로그로 노출하는 용도.

    체크 항목:
      - complexity=simple 인데 async def 비율 높거나 QThread/asyncio 사용
      - data_layer 와 import 된 DB 드라이버 불일치
      - notification=none 인데 알림 라이브러리/QSystemTrayIcon 사용
      - avoid 에 열거된 패턴(orm/microservice 등)이 실제로 import 됨
      - ui_approach=single_window 인데 MainWindow 상속 2개 이상
    """
    if strategy is None:
        return []

    warnings: List[str] = []
    all_imports: set = set()
    all_from_imports: Dict[str, set] = {}
    all_classes: List[Tuple[str, List[str]]] = []
    funcs_total = 0
    async_funcs = 0

    for fn, code in (implemented or {}).items():
        if not code or not fn.endswith(".py"):
            continue
        try:
            tree = ast.parse(code)
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    all_imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                mod_top = (node.module or "").split(".")[0]
                if mod_top:
                    all_imports.add(mod_top)
                    nms = {a.name for a in node.names if a.name != "*"}
                    all_from_imports.setdefault(mod_top, set()).update(nms)
            elif isinstance(node, ast.ClassDef):
                parents: List[str] = []
                for p in node.bases:
                    if isinstance(p, ast.Name):
                        parents.append(p.id)
                    elif isinstance(p, ast.Attribute):
                        parents.append(p.attr)
                all_classes.append((node.name, parents))
            elif isinstance(node, ast.FunctionDef):
                funcs_total += 1
            elif isinstance(node, ast.AsyncFunctionDef):
                funcs_total += 1
                async_funcs += 1

    complexity = getattr(strategy, "complexity", "moderate")
    data_layer = getattr(strategy, "data_layer", "sqlite")
    ui_approach = getattr(strategy, "ui_approach", "single_window")
    notification = getattr(strategy, "notification", "none")
    avoid = [str(a).lower() for a in (getattr(strategy, "avoid", []) or [])]

    # ── 1. complexity=simple 인데 비동기/스레드 남발 ──────────────────────
    if complexity == "simple":
        if funcs_total > 0 and async_funcs / funcs_total > 0.3:
            warnings.append(
                f"complexity=simple 인데 async def 비율 "
                f"{async_funcs}/{funcs_total} ({async_funcs/funcs_total:.0%})"
            )
        qthread_classes = [c for c, ps in all_classes if "QThread" in ps]
        if qthread_classes:
            warnings.append(
                f"complexity=simple 인데 QThread 상속 클래스: {qthread_classes}"
            )
        if "asyncio" in all_imports:
            warnings.append("complexity=simple 인데 asyncio import 존재")

    # ── 2. data_layer ↔ DB 드라이버 불일치 ───────────────────────────────
    _db_allowed = {
        "sqlite": {"sqlite3"},
        "postgresql": {"psycopg2", "psycopg", "asyncpg"},
        "file": set(),
        "in_memory": set(),
    }
    _db_known = {"sqlite3", "psycopg2", "psycopg", "asyncpg", "pymongo", "sqlalchemy"}
    allowed = _db_allowed.get(data_layer, set())
    for pkg in all_imports & _db_known:
        if pkg not in allowed:
            warnings.append(f"data_layer={data_layer} 인데 {pkg} import")

    # ── 3. notification=none 인데 알림 의존성 ────────────────────────────
    if notification == "none":
        _alert_pkgs = {"firebase_admin", "schedule", "apscheduler", "plyer"}
        hit_alert = all_imports & _alert_pkgs
        if hit_alert:
            warnings.append(f"notification=none 인데 알림 라이브러리: {sorted(hit_alert)}")
        # QSystemTrayIcon 등 Qt 트레이 알림
        for names in all_from_imports.values():
            tray_hits = names & {"QSystemTrayIcon"}
            if tray_hits:
                warnings.append("notification=none 인데 QSystemTrayIcon 사용")
                break

    # ── 4. avoid 패턴 위반 ────────────────────────────────────────────────
    _avoid_map = {
        "orm": {"sqlalchemy"},
        "microservice": {"fastapi", "uvicorn"},
        "flask": {"flask"},
        "django": {"django"},
    }
    for pattern in avoid:
        forbidden = _avoid_map.get(pattern, set())
        hit = all_imports & forbidden
        if hit:
            warnings.append(f"avoid='{pattern}' 인데 {sorted(hit)} import")

    # ── 5. ui_approach=single_window 인데 MainWindow 중복 ────────────────
    if ui_approach == "single_window":
        mainwins = [
            c for c, ps in all_classes
            if any("MainWindow" in (p or "") for p in ps)
        ]
        if len(mainwins) >= 2:
            warnings.append(
                f"ui_approach=single_window 인데 MainWindow 상속 {len(mainwins)}개: {mainwins}"
            )

    return warnings


def count_pass_only_funcs(implemented: Dict[str, str]) -> Tuple[int, int]:
    """[P4] 구현 결과에서 pass-only 함수 수와 전체 함수 수 반환.

    pass-only 판정 기준 (AST):
      - 본문이 단일 `pass`
      - 본문이 docstring + pass 만
      - 본문이 docstring 뿐 (비어있음)

    파싱 실패 파일은 통계에서 제외 (AST 검증은 P1/P2 가 담당).
    """
    total = 0
    po = 0
    for fn, code in (implemented or {}).items():
        if not code or not fn.endswith(".py"):
            continue
        try:
            tree = ast.parse(code)
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                total += 1
                # docstring 노드(첫 Expr(Constant(str))) 제외 후 남은 본문
                body = [
                    n for n in node.body
                    if not (
                        isinstance(n, ast.Expr)
                        and isinstance(n.value, ast.Constant)
                        and isinstance(n.value.value, str)
                    )
                ]
                if len(body) == 0 or (len(body) == 1 and isinstance(body[0], ast.Pass)):
                    po += 1
    return po, total


def schema_to_skeleton(
    file_plan: List[Dict],
    contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    [결정론적 변환 — LLM 없음]
    DC file_plan 스키마 → 실제 Python skeleton 코드 딕셔너리.

    클래스명/메서드명/import/생성자 인자는 스키마에서 직접 추출.
    LLM은 # TODO 블록만 채우면 됨.

    [Phase B-1 2026-04-23] contract 가 주어지면 contract.files[fname].imports 의
    내부 모듈 import 를 file_plan.imports 에 추가해 emit (dedupe). 이렇게 하면
    skeleton 단계에서부터 모든 파일이 "어떤 형제 모듈의 어떤 심볼을 사용하는지"
    contract 와 정렬된다. 클래스 구조 자체는 file_plan 에서 그대로 가져옴
    (메서드별 TODO 등 풍부한 정보 손실 방지).

    Returns:
        {"filename.py": "skeleton code string", ...}
    """
    skeletons: Dict[str, str] = {}
    # [Fix 2026-04-20] 드랍된 파일을 호출측이 인지할 수 있도록 기록.
    # 기존: 비-.py 를 조용히 drop 해서 스캐폴딩 결과가 0 파일이어도 아무 경고 없음.
    _dropped: List[str] = []

    # [Phase B-1] contract 에서 파일별 내부 모듈 imports 추출 (있으면)
    contract_imports_by_file: Dict[str, List[Tuple[str, List[str]]]] = {}
    if isinstance(contract, dict):
        _cfiles = contract.get("files", {}) or {}
        if isinstance(_cfiles, dict):
            for _fn, _spec in _cfiles.items():
                if not isinstance(_spec, dict):
                    continue
                _imps = _spec.get("imports", []) or []
                _pairs: List[Tuple[str, List[str]]] = []
                for _imp in _imps:
                    if not isinstance(_imp, dict):
                        continue
                    _mod = (_imp.get("module", "") or "").strip()
                    _nms = [n for n in (_imp.get("names", []) or []) if isinstance(n, str) and n.strip()]
                    if _mod and _nms:
                        _pairs.append((_mod, _nms))
                if _pairs:
                    contract_imports_by_file[_fn] = _pairs

    for fp in file_plan:
        # [Fix 2026-04-23] LLM 이 'file' 대신 'filename'/'name'/'path' 키로 emit
        # 하는 경우(특히 DC-App 분할 재시도 프롬프트) 전부 '(파일명 없음)' drop
        # 되어 skeleton 0개로 빈 스캐폴딩이 나오던 버그.
        filename = ""
        for _key in ("file", "filename", "path", "name"):
            _v = fp.get(_key, "")
            if isinstance(_v, str) and _v.strip():
                filename = _v.strip()
                break
        imports     = fp.get("imports", [])
        classes     = fp.get("classes", [])
        entry_point = fp.get("entry_point", False)
        entry_code  = fp.get("entry_code", "")

        if not filename:
            _dropped.append("(파일명 없음)")
            continue

        # [Fix 2026-04-23] 확장자 누락 시 .py 자동 부여
        # (분할 프롬프트가 'task_screen' 처럼 스템만 주는 경우 구제)
        if "." not in filename:
            filename = filename + ".py"

        # 비파이썬 파일(.js, .jsx, .ts, .tsx 등)은 skeleton 생성 제외
        # LLM이 React/JS 파일을 file_plan에 넣는 경우 방어
        if not filename.endswith(".py"):
            _dropped.append(filename)
            print(f"  [schema_to_skeleton] 비-.py 파일 drop: {filename}")
            continue

        # 정규화된 filename 을 fp 에도 반영 — 이후 단계(plan_by_file 등)와 일관성 유지
        fp["file"] = filename

        lines: List[str] = []

        # ── imports ──────────────────────────────────────────────────────
        if entry_point:
            lines.append("import sys")

        # 중복 제거용 (module_path, frozenset(names)) 기준
        _seen_imports: set = set()

        def _emit_import(from_mod: str, names: List[str]) -> None:
            if not from_mod:
                return
            key = (from_mod, frozenset(names) if names else frozenset({"*"}))
            if key in _seen_imports:
                return
            _seen_imports.add(key)
            if names:
                lines.append(f"from {from_mod} import {', '.join(names)}")
            else:
                lines.append(f"import {from_mod}")

        for imp in imports:
            from_mod = imp.get("from", "")
            names    = imp.get("names", [])
            _emit_import(from_mod, names)

        # [Phase B-1] contract 의 내부 모듈 imports 추가 (file_plan 에 누락된 것 보강)
        for _mod, _nms in contract_imports_by_file.get(filename, []):
            _emit_import(_mod, _nms)

        if lines:
            lines.append("")
            lines.append("")

        # ── classes ──────────────────────────────────────────────────────
        for cls in classes:
            # DCReasoner는 classes를 문자열 리스트로 반환할 수 있음
            # schema_to_skeleton은 dict를 기대하므로 방어 처리
            if isinstance(cls, str):
                # 문자열이면 클래스명만 있는 최소 skeleton 생성
                lines.append(f"class {cls}:")
                lines.append("")
                lines.append("    def __init__(self):")
                lines.append(f"        # TODO: {cls} 초기화 구현")
                lines.append("        pass")
                lines.append("")
                lines.append("")
                continue

            cls_name  = cls.get("name", "MyClass")
            parent    = cls.get("parent", "")
            ctor_args = cls.get("constructor_args", [])
            methods   = cls.get("methods", [])

            parent_str = f"({parent})" if parent else ""
            lines.append(f"class {cls_name}{parent_str}:")
            lines.append("")

            # __init__
            arg_strs = ["self"]
            for a in ctor_args:
                aname = a.get("name", "arg")
                atype = a.get("type", "")
                adef  = a.get("default", "")
                # LLM이 스키마에 self를 명시적으로 넣는 경우 제거
                if aname.strip() in ("self", ""):
                    continue
                if atype and adef:
                    arg_strs.append(f"{aname}: {atype} = {adef}")
                elif atype:
                    arg_strs.append(f"{aname}: {atype}")
                elif adef:
                    arg_strs.append(f"{aname}={adef}")
                else:
                    arg_strs.append(aname)

            lines.append(f"    def __init__({', '.join(arg_strs)}):")

            if parent:
                lines.append(f"        super().__init__()")

            for a in ctor_args:
                aname = a.get("name", "")
                if aname:
                    lines.append(f"        self.{aname} = {aname}")

            ctor_todo = cls.get("constructor_todo", "")
            if ctor_todo:
                lines.append(f"        # TODO: {ctor_todo}")
                lines.append("        pass")
            elif not ctor_args:
                lines.append("        pass")

            lines.append("")

            # methods
            for method in methods:
                mname   = method.get("name", "method")
                returns = method.get("returns", "None")
                margs   = method.get("args", [])
                todo    = method.get("todo", "구현 필요")

                m_arg_strs = ["self"]
                for a in margs:
                    aname = a.get("name", "arg")
                    atype = a.get("type", "")
                    # LLM이 self를 args에 넣는 경우 제거
                    if aname.strip() in ("self", ""):
                        continue
                    if atype:
                        m_arg_strs.append(f"{aname}: {atype}")
                    else:
                        m_arg_strs.append(aname)

                ret_hint = f" -> {returns}" if returns and returns != "None" else " -> None"
                lines.append(f"    def {mname}({', '.join(m_arg_strs)}){ret_hint}:")
                lines.append(f"        # TODO: {todo}")
                lines.append("        pass")
                lines.append("")

            lines.append("")

        # ── entry_point ──────────────────────────────────────────────────
        if entry_point and entry_code:
            lines.append("")
            lines.append('if __name__ == "__main__":')
            for ec_line in entry_code.replace("\\n", "\n").split("\n"):
                lines.append(f"    {ec_line}")

        skeletons[filename] = "\n".join(lines)

    if _dropped:
        print(
            f"  [schema_to_skeleton] file_plan {len(file_plan)}개 중 "
            f"{len(_dropped)}개 drop → skeleton {len(skeletons)}개. "
            f"drop 목록: {_dropped[:10]}"
        )
    return skeletons


async def implement_todos_async(
    skeletons: Dict[str, str],
    file_plan: List[Dict],
    data_models: List[Dict],
    project_type: str,
    progress_callback=None,
    architecture_strategy=None,  # [DNA] ArchitectureStrategy 객체 (선택)
    contract: Optional[Dict[str, Any]] = None,  # [Phase B-2] module contract
) -> Dict[str, str]:
    """
    [TODO 병렬 구현]
    skeleton 파일들의 # TODO 블록을 LLM으로 채워 완성된 코드 반환.

    전략:
    - 파일 단위로 LLM 호출 (메서드 단위 아님 — 컨텍스트 유지 목적)
    - contract 없으면: 모든 파일을 한번에 병렬 처리 (기존 동작)
    - contract 있으면: contract.imports 로 의존 그래프 구성 → topological 배치 단위로 처리.
      leaf 모듈부터 채워서 의존 모듈의 **실제 본문**을 dependent 의 컨텍스트에 주입.
      각 파일에 contract.exports 를 "필수 export" 지시문으로 강제.

    Returns:
        {"filename.py": "implemented code", ...}
    """
    def _cb(msg: str):
        if progress_callback:
            try:
                progress_callback(msg)
            except Exception:
                pass
        print(f"  [SkeletonImpl] {msg}")

    # ── [Fix 2026-04-25] 병렬도 RPM cap 정렬 ──────────────────────────────
    # 8파일 동시 발사 + 파일당 최대 2회(1차/재생성) → 16 in-flight 가능.
    # eidos_llm_throttle 의 RPM=12 와 정면충돌해 절단/cap 재시도 실패가
    # 나던 문제. 동시 N개로 제한 (env EIDOS_SCAFFOLD_PARALLEL, 기본 4).
    import os as _os_sc
    try:
        _scaffold_par = int(_os_sc.environ.get("EIDOS_SCAFFOLD_PARALLEL", "4") or "4")
    except Exception:
        _scaffold_par = 4
    _scaffold_par = max(1, min(_scaffold_par, 16))
    _scaffold_sem = asyncio.Semaphore(_scaffold_par)

    plan_by_file = {fp.get("file", ""): fp for fp in file_plan}
    db_schema_str = json.dumps(data_models, ensure_ascii=False, indent=2) if data_models else "없음"

    all_files_summary = "\n".join(
        f"- {fp.get('file','')}: {fp.get('role','') or fp.get('responsibility','')}"
        for fp in file_plan
    )

    # [Fix G 2026-05-05] 프로젝트 전체 메서드 시그니처 카탈로그.
    # 사용자 보고 "스캐폴딩 품질 엉망" — 진단 결과 contract 미적용 (커버리지 50% 미만)
    # 시 모든 파일이 *skeleton* 만 본 채 병렬 LLM 호출 → 메서드 A 가 메서드 B 호출 시
    # 시그니처/이름 어긋남. 해법: file_plan 의 모든 파일의 classes→methods 를 한
    # 카탈로그로 만들어 모든 _implement_one_inner prompt 에 박음. contract 모드든
    # 미적용 모드든 다 활용. file_plan 자체가 DC Reasoner 출력이라 시그니처 정보는
    # 항상 있음 (DC prompt 가 methods 강제 — eidos_dc_reasoner.py:2887).
    def _build_signature_catalog() -> str:
        lines: List[str] = []
        for fp in file_plan:
            fname = fp.get("file", "")
            classes = fp.get("classes", []) or []
            if not classes or not fname:
                continue
            file_lines: List[str] = []
            for cls in classes:
                if isinstance(cls, str):
                    continue  # 문자열 list 형태는 시그니처 정보 없음
                if not isinstance(cls, dict):
                    continue
                cname = cls.get("name", "") or ""
                if not cname:
                    continue
                # __init__ 시그니처
                ctor_args = cls.get("constructor_args", []) or []
                ctor_sig_parts = ["self"]
                for arg in ctor_args:
                    if not isinstance(arg, dict):
                        continue
                    aname = arg.get("name", "")
                    atype = arg.get("type", "")
                    if aname:
                        ctor_sig_parts.append(
                            f"{aname}: {atype}" if atype else aname
                        )
                file_lines.append(
                    f"  class {cname}:\n    __init__({', '.join(ctor_sig_parts)})"
                )
                # 메서드 시그니처
                for m in cls.get("methods", []) or []:
                    if not isinstance(m, dict):
                        continue
                    mname = m.get("name", "")
                    mret = m.get("returns", "") or ""
                    margs = m.get("args", []) or []
                    arg_parts = ["self"]
                    for a in margs:
                        if not isinstance(a, dict):
                            continue
                        an = a.get("name", "")
                        at = a.get("type", "")
                        if an:
                            arg_parts.append(f"{an}: {at}" if at else an)
                    sig_line = f"    {mname}({', '.join(arg_parts)})"
                    if mret:
                        sig_line += f" -> {mret}"
                    if sig_line.strip() != "()" and mname:
                        file_lines.append(sig_line)
            if file_lines:
                lines.append(f"# {fname}")
                lines.extend(file_lines)
        return "\n".join(lines) if lines else ""

    _signature_catalog = _build_signature_catalog()

    # ── [Phase B-2] contract 정규화 + 의존 그래프 구성 ────────────────────────
    # contract.files[fname] = {"exports": [...], "imports": [{"module": "...", "names": [...]}]}
    # 의존 그래프: filename → {dependent_filename, ...} (filename 이 의존하는 파일들)
    contract_files: Dict[str, Dict[str, Any]] = {}
    dep_graph: Dict[str, set] = {}     # 노드: filename, 값: 그 파일이 import 하는 형제 파일 집합
    local_modules: Dict[str, str] = {}  # module_name → filename.py 매핑

    if isinstance(contract, dict):
        _cf = contract.get("files", {})
        if isinstance(_cf, dict):
            for _fn, _spec in _cf.items():
                if isinstance(_spec, dict) and _fn.endswith(".py") and _fn in skeletons:
                    contract_files[_fn] = _spec
        # local module name 매핑 (database.py → database, ui_main.py → ui_main)
        local_modules = {fn[:-3]: fn for fn in skeletons if fn.endswith(".py")}

    # contract 가 유효한지 (skeletons 의 절반 이상 커버) 확인
    use_contract = bool(contract_files) and (len(contract_files) >= max(1, len(skeletons) // 2))
    if contract and not use_contract:
        _cb(
            f"⚠️ contract 커버리지 부족 "
            f"({len(contract_files)}/{len(skeletons)}) — 기존 병렬 모드로 진행"
        )

    if use_contract:
        for _fn, _spec in contract_files.items():
            _deps: set = set()
            for _imp in _spec.get("imports", []) or []:
                if not isinstance(_imp, dict):
                    continue
                _mod = (_imp.get("module", "") or "").strip()
                # 외부 라이브러리(PySide6 등) 는 local_modules 에 없으니 자동 제외됨
                _dep_fn = local_modules.get(_mod)
                if _dep_fn and _dep_fn != _fn:
                    _deps.add(_dep_fn)
            dep_graph[_fn] = _deps
        # contract 에 안 들린 파일도 그래프에 추가 (의존 없음으로 leaf)
        for _fn in skeletons:
            dep_graph.setdefault(_fn, set())

    # contract.exports/dep exports 직렬화 헬퍼
    def _exports_block(fname: str) -> str:
        spec = contract_files.get(fname, {})
        exps = spec.get("exports", []) or []
        if not exps:
            return ""
        lines: List[str] = []
        for e in exps:
            if not isinstance(e, dict):
                continue
            sig = (e.get("signature", "") or "").strip()
            init = (e.get("init", "") or "").strip()
            kind = (e.get("kind", "") or "").strip()
            name = (e.get("name", "") or "").strip()
            if sig:
                lines.append(f"- {sig}")
                if init and kind == "class":
                    lines.append(f"    {init}")
            elif name:
                lines.append(f"- {kind} {name}")
        return "\n".join(lines)

    # [G 2026-04-26] contract.methods 직렬화 헬퍼.
    # 클래스의 인스턴스 메서드 시그니처를 파일별로 정리. 호출자/피호출자 모두
    # 같은 메서드 이름·시그니처를 쓰도록 강제하는 핵심 directive.
    # [J 2026-04-26] centrality 정보 prepend — 고중심성 클래스는 강한 경고
    def _methods_block(fname: str) -> str:
        spec = contract_files.get(fname, {})
        methods = spec.get("methods", []) or []
        if not methods:
            return ""
        # contract._centrality (J) 접근 — closure 로 contract 참조
        _centrality_map = (
            (contract or {}).get("_centrality", {})
            if isinstance(contract, dict) else {}
        )
        # class 별 그룹
        by_class: Dict[str, List[Dict[str, Any]]] = {}
        for m in methods:
            if not isinstance(m, dict):
                continue
            cn = (m.get("class", "") or "").strip()
            if not cn:
                continue
            by_class.setdefault(cn, []).append(m)
        if not by_class:
            return ""
        lines: List[str] = []
        for cn, mlist in by_class.items():
            # [J] centrality 기반 헤더 — 고중심성 강한 경고
            c_info = _centrality_map.get(cn, {}) if isinstance(_centrality_map, dict) else {}
            c_rank = c_info.get("rank", "") if isinstance(c_info, dict) else ""
            c_score = c_info.get("score", 0.0) if isinstance(c_info, dict) else 0.0
            c_in = c_info.get("in_count", 0) if isinstance(c_info, dict) else 0
            # [K 2026-04-26] caller 파일 목록 (변경 영향 시각화)
            c_uses = c_info.get("uses", []) if isinstance(c_info, dict) else []
            c_uses_str = ", ".join(c_uses[:3]) + (f" 외 {len(c_uses)-3}개" if len(c_uses) > 3 else "")
            if c_rank == "high":
                lines.append(
                    f"[{cn} — ★ 고중심성 (score={c_score:.2f}, "
                    f"{c_in}개 모듈이 의존) — 메서드 누락 시 다수 caller 가 폭탄. "
                    f"빠짐없이 모두 정의 필수]"
                )
                if c_uses:
                    lines.append(
                        f"  🧬 Intervention(do-calculus): 이 클래스 메서드 1개 누락/이름 변경 → "
                        f"{c_uses_str} 즉시 AttributeError"
                    )
                    lines.append(
                        f"  🧬 Counterfactual: 만약 메서드 시그니처 다르게 정의했다면 "
                        f"{len(c_uses)}개 caller 의 호출 코드 동시 수정 필요 → 시그니처 정확 일치 critical"
                    )
            elif c_rank == "med":
                lines.append(
                    f"[{cn} — 중심성 보통 (score={c_score:.2f}, {c_in}개 의존) — "
                    f"메서드 정의 주의]"
                )
                if c_uses:
                    lines.append(
                        f"  🧬 Intervention: 메서드 누락 시 {c_uses_str} 영향"
                    )
            else:
                lines.append(f"[{cn} — 반드시 정의해야 할 인스턴스 메서드]")
            for m in mlist:
                sig = (m.get("signature", "") or "").strip()
                purpose = (m.get("purpose", "") or "").strip()
                name = (m.get("name", "") or "").strip()
                if sig:
                    lines.append(f"  - {sig}")
                    if purpose:
                        lines.append(f"      └ {purpose}")
                elif name:
                    lines.append(f"  - {name} — {purpose}")
        return "\n".join(lines)

    async def _implement_one(
        filename: str,
        skeleton_code: str,
        extra_directives: str = "",
        deps_context_override: Optional[str] = None,
    ) -> tuple:
        # 병렬도 cap — 동시 _scaffold_par 개로 제한해 RPM=12 와 충돌 방지
        async with _scaffold_sem:
            return await _implement_one_inner(
                filename, skeleton_code,
                extra_directives=extra_directives,
                deps_context_override=deps_context_override,
            )

    async def _implement_one_inner(
        filename: str,
        skeleton_code: str,
        extra_directives: str = "",
        deps_context_override: Optional[str] = None,
    ) -> tuple:
        fp_info = plan_by_file.get(filename, {})
        # DCReasoner는 "responsibility" 키를 쓰고, schema_to_skeleton은 "role"을 씀
        role    = fp_info.get("role", "") or fp_info.get("responsibility", "")
        classes = fp_info.get("classes", [])

        # 이 파일이 의존하는 파일의 skeleton을 컨텍스트로 제공
        # [Phase B-2] contract 모드면 외부에서 만든 강한 컨텍스트로 대체
        if deps_context_override is not None:
            deps_context = deps_context_override
        else:
            deps_context = ""
            for cls in classes:
                # DCReasoner는 classes를 문자열 리스트로 반환할 수 있음 — str이면 skip
                if isinstance(cls, str):
                    continue
                for method in cls.get("methods", []):
                    todo_text = method.get("todo", "")
                    for other_fname, other_skel in skeletons.items():
                        if other_fname != filename and other_fname.replace(".py", "") in todo_text:
                            # [Fix G 2026-05-05] 800 → 2400 — skeleton 자체가
                            # 메서드 시그니처 + docstring 으로 800자 쉽게 초과.
                            deps_context += f"\n--- [{other_fname} skeleton] ---\n{other_skel[:2400]}\n"

        # 화면/윈도우 파일인지 판단 (UI 구현 힌트 강화용)
        _is_screen   = any(k in filename for k in ("screen", "window", "widget", "form", "page", "list", "detail", "main"))
        _is_db       = any(k in filename for k in ("database", "db", "model", "service", "repository"))
        _is_main     = filename == "main.py"

        # 공통 모던 QSS 팔레트
        _QSS_PALETTE = """
# 공통 색상 팔레트 (모든 화면 파일에 적용)
BG       = "#F8F9FA"   # 앱 배경
SURFACE  = "#FFFFFF"   # 카드/패널 배경
ACCENT   = "#4F46E5"   # 인디고 포인트 컬러
ACCENT_H = "#4338CA"   # 호버
TEXT     = "#1E293B"   # 메인 텍스트
MUTED    = "#94A3B8"   # 보조 텍스트
BORDER   = "#E2E8F0"   # 경계선
DANGER   = "#EF4444"   # 삭제/위험
"""

        _ui_hint = ""
        if _is_screen:
            _ui_hint = f"""
[UI 구현 필수 지침 — 모던 디자인 적용 필수]
{_QSS_PALETTE}
★ 필수: 모든 위젯에 setStyleSheet() 반드시 적용. 기본 Windows 스타일 납품 절대 금지.

[레이아웃]
- 전체 배경: self.setStyleSheet("background:#F8F9FA;")
- QVBoxLayout / QHBoxLayout, setContentsMargins(24,24,24,24), setSpacing(12)
- 제목: QLabel, setStyleSheet("font-size:20px; font-weight:700; color:#1E293B; margin-bottom:8px;")

[버튼 — 반드시 이 스타일 사용]
- 주 버튼: setStyleSheet("QPushButton{{background:#4F46E5; color:#fff; border:none; border-radius:8px; padding:10px 20px; font-size:13px; font-weight:600;}} QPushButton:hover{{background:#4338CA;}}")
- 위험 버튼: setStyleSheet("QPushButton{{background:#EF4444; color:#fff; border:none; border-radius:8px; padding:8px 16px;}} QPushButton:hover{{background:#DC2626;}}")

[입력 필드]
- QLineEdit: setStyleSheet("QLineEdit{{background:#fff; border:1.5px solid #E2E8F0; border-radius:8px; padding:10px 14px; font-size:13px; color:#1E293B;}} QLineEdit:focus{{border-color:#4F46E5;}}")
- setFixedHeight(44)

[리스트/테이블]
- QListWidget: setStyleSheet("QListWidget{{background:#fff; border:1px solid #E2E8F0; border-radius:10px; padding:4px;}} QListWidget::item{{padding:12px 16px; border-radius:8px; color:#1E293B;}} QListWidget::item:hover{{background:#EEF2FF;}} QListWidget::item:selected{{background:#EEF2FF; color:#4F46E5;}}")
- QTableWidget: setStyleSheet("QTableWidget{{background:#fff; border:1px solid #E2E8F0; border-radius:10px; gridline-color:#F1F5F9; font-size:13px;}} QHeaderView::section{{background:#F8F9FA; color:#64748B; font-weight:600; padding:10px; border:none; border-bottom:1px solid #E2E8F0;}} QTableWidget::item{{padding:10px;}} QTableWidget::item:selected{{background:#EEF2FF; color:#4F46E5;}}")

[카드 컨테이너]
- QFrame: setStyleSheet("QFrame{{background:#fff; border:1px solid #E2E8F0; border-radius:12px; padding:16px;}}")

[기타]
- 반드시 __init__에서 self._init_ui() 호출
- pass만 있는 메서드 절대 금지 — 실제 코드 작성"""
        elif _is_db:
            _ui_hint = """
[DB/서비스 구현 필수 지침]
- __init__: self.conn = sqlite3.connect(db_path, check_same_thread=False); self.cursor = self.conn.cursor(); self.create_tables()
- create_tables: CREATE TABLE IF NOT EXISTS 구문 실행 후 self.conn.commit()
- INSERT: self.cursor.execute(...); self.conn.commit(); return self.cursor.lastrowid
- SELECT: self.cursor.execute(...); return self.cursor.fetchall()
- pass만 있는 메서드 절대 금지"""
        elif _is_main:
            _ui_hint = """
[main.py 구현 필수 지침]
- if __name__ == "__main__": 블록 반드시 포함
- app.setStyle("Fusion") 적용
- app.setStyleSheet("QToolTip{background:#1E293B; color:#fff; border:none; padding:4px 8px; border-radius:4px;}")
- MainWindow() 생성 후 show() 호출
- MainWindow.__init__: setWindowTitle, setMinimumSize(900, 600), 중앙 위젯 설정"""

        # [DNA 전략] architecture_strategy가 있으면 힌트 추가
        strategy_block = ""
        if architecture_strategy is not None:
            try:
                strategy_block = f"\n\n{architecture_strategy.to_prompt_hint()}"
            except Exception:
                pass

        prompt = f"""당신은 PySide6 전문 개발자입니다.
아래 skeleton 코드의 모든 # TODO 블록을 실제 동작하는 코드로 구현하세요.

[프로젝트 타입]
{project_type}

[이 파일의 역할]
{role}{strategy_block}

[전체 프로젝트 파일 구조 (참조용)]
{all_files_summary}

[★ 프로젝트 전체 메서드 시그니처 카탈로그 — 다른 파일의 메서드 호출 시 반드시 이 시그니처 그대로 사용]
{_signature_catalog if _signature_catalog else "(file_plan 에 시그니처 정보 없음)"}

[DB 스키마 (SQLite)]
{db_schema_str}

[의존 파일 skeleton (참조용)]
{deps_context if deps_context else "없음"}
{_ui_hint}
[구현할 skeleton — 이 구조를 절대 변경하지 말 것]
{skeleton_code}

[★ 절대 규칙]
1. 클래스명, 메서드명, 인자명, import는 skeleton 그대로 유지. 절대 변경 금지.
2. # TODO 줄과 pass를 실제 구현 코드로 교체. pass만 남기는 것 절대 금지.
3. 새 메서드/클래스를 임의로 추가하지 말 것.
4. 완전한 Python 파일 코드만 출력. 마크다운 코드블록 없이.
5. 누락된 import가 있으면 파일 상단에 추가 가능.
6. **다른 파일의 메서드를 호출할 때 위 "메서드 시그니처 카탈로그" 의 시그니처 그대로 사용**. 카탈로그에 없는 메서드 호출 절대 금지 — 호출하면 AttributeError.

[출력: 완성된 {filename} 전체 코드]"""

        # [P2] 크로스 파일 재구현 등 상위 패스에서 추가 지시가 있으면 말미에 덧붙임
        if extra_directives:
            prompt = prompt + "\n\n" + extra_directives

        # ── [P1 2026-04-22] LLM 응답 검증 게이트 ──────────────────────────
        # 이전엔 코드펜스만 제거하고 바로 디스크로 → 문법 깨진 파일·중복 함수 정의가
        # 조용히 납품됐다. 이제 AST 파싱 + 중복 심볼 감지 후 실패 원인을 피드백으로
        # 붙여 1회 재생성. 재생성도 실패하면 깨진 코드 대신 skeleton 유지.
        def _clean(raw: str) -> str:
            if not raw:
                return ""
            # BOM / zero-width 제거 (선두 1바이트 BOM, U+200B/C/D)
            raw = raw.lstrip("﻿​‌‍").strip()
            # 마크다운 펜스가 본문 어디든 있으면 가장 첫 번째 코드블록만 추출
            # (LLM 이 "Here is the fix:" 같은 설명 후 ``` 시작하는 케이스 흡수)
            _m = re.search(r"```(?:python|py)?\s*\n(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
            if _m:
                return _m.group(1).strip()
            # 펜스가 시작에만 있고 닫히지 않은 경우(절단)
            if raw.startswith("```"):
                raw = re.sub(r'^```(?:python|py)?\s*', '', raw, flags=re.IGNORECASE)
                raw = re.sub(r'\s*```\s*$', '', raw).strip()
            return raw

        def _try_autoclose(code: str) -> Optional[str]:
            """절단된 코드의 끝부분을 한 줄씩 깎으며 ast.parse 가 통과하는
            지점까지 잘라낸다. 성공하면 닫힌 코드, 실패하면 None.
            settings_screen.py 처럼 끝의 미완 코멘트/식별자 1~2줄 때문에
            전체 파일이 버려지던 케이스를 살린다."""
            if not code:
                return None
            _lines = code.splitlines()
            # 최대 30줄까지 후행 깎기 시도 — 실제 파일 전체가 깨졌으면
            # autoclose 가 거의 빈 코드를 반환하므로 길이 가드도 추가
            _orig_len = len(code)
            for _n_drop in range(1, min(30, len(_lines))):
                _trimmed = "\n".join(_lines[:-_n_drop])
                if not _trimmed.strip():
                    return None
                # 너무 많이 깎이면 (>40% 손실) 가치 없음
                if len(_trimmed) < _orig_len * 0.6:
                    return None
                try:
                    ast.parse(_trimmed)
                    return _trimmed
                except SyntaxError:
                    continue
            return None

        def _is_llm_error_string(code: str) -> bool:
            """llm_module이 타임아웃/실패 시 리턴하는 '[서버...]' 문자열 감지.
            ast.parse가 결국 SyntaxError로 잡긴 하나, 명시적으로 단락시켜
            재생성 LLM 호출 1회를 아낀다."""
            s = (code or "").strip()
            if not s.startswith("["):
                return False
            head = s[:120]
            return any(tok in head for tok in
                       ("서버 응답", "서버 연결", "LLM 호출 오류",
                        "스트리밍 오류", "시간 초과"))

        def _validate(code: str) -> Optional[str]:
            """검증 통과 시 None, 실패 시 실패 이유 문자열 반환."""
            if not code:
                return "빈 응답"
            if _is_llm_error_string(code):
                return f"LLM 에러 문자열 리턴: {code.strip()[:80]}"
            try:
                tree = ast.parse(code)
            except SyntaxError as se:
                # LLM 이 즉시 교정할 수 있도록 위치·메시지 노출
                return f"SyntaxError: {se.msg} (line {se.lineno}, col {se.offset})"
            except Exception as pe:
                return f"파싱 실패: {pe}"

            # top-level 함수/클래스 중복
            dup: List[str] = []
            tl_counts: Dict[str, int] = {}
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    tl_counts[node.name] = tl_counts.get(node.name, 0) + 1
            for name, n in tl_counts.items():
                if n > 1:
                    dup.append(f"top-level '{name}' {n}회 정의")

            # 각 클래스 내 메서드 중복
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    m_counts: Dict[str, int] = {}
                    for sub in node.body:
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            m_counts[sub.name] = m_counts.get(sub.name, 0) + 1
                    for mname, mn in m_counts.items():
                        if mn > 1:
                            dup.append(f"'{node.name}.{mname}' {mn}회 정의")

            if dup:
                return "중복 심볼: " + "; ".join(dup)

            # ── [Fix 2026-04-23] 절단 감지 ─────────────────────────────────
            # LLM 응답이 max_tokens 한계로 잘려도 `super`, `self.current_task_id`,
            # `super().__` 같은 식별자는 문법적으로 valid 한 Expression 으로 파싱
            # 돼 기존 검증을 통과했다. 결과로 함수 바디가 단일 bare-expression 인
            # 절단된 파일이 디스크에 저장됐다. 이제 트리 전체를 순회하며 bare
            # Name/Attribute/Constant(docstring 제외) 가 statement 로 쓰인 곳을
            # 전부 절단 의심으로 플래그. 실제 PySide6/DB 코드는 이런 bare
            # expression 을 statement 로 쓰는 경우가 거의 없다.
            bare_exprs: List[str] = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.Expr):
                    continue
                v = node.value
                # docstring(상수 문자열)은 정상
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    continue
                # Call 은 완결된 문
                if isinstance(v, ast.Call):
                    continue
                # Await (await something()) 정상
                if isinstance(v, ast.Await):
                    continue
                # Yield/YieldFrom 정상
                if isinstance(v, (ast.Yield, ast.YieldFrom)):
                    continue
                # 그 외 bare Name/Attribute/Constant → 절단 의심
                if isinstance(v, (ast.Name, ast.Attribute, ast.Constant)):
                    if isinstance(v, ast.Name):
                        bare_exprs.append(v.id)
                    elif isinstance(v, ast.Attribute):
                        # ex) self.foo 로 표기
                        try:
                            bare_exprs.append(ast.unparse(v)[:40])
                        except Exception:
                            bare_exprs.append(v.attr)
                    else:
                        bare_exprs.append(repr(v.value)[:20])
            if bare_exprs:
                return (
                    "절단 감지(bare expression statement): "
                    + ", ".join(bare_exprs[:5])
                )

            # 파일 끝이 newline 없이 식별자/점/쉼표로 끝나면 절단 가능성 매우 높음
            tail = code.rstrip()
            if tail and not tail.endswith((")", "]", "}", ":", '"', "'", "pass")):
                last_line = tail.splitlines()[-1] if tail else ""
                last_stripped = last_line.strip()
                if last_stripped and not last_stripped.startswith("#"):
                    if (
                        last_stripped.endswith(".")
                        or last_stripped.endswith("_")
                        or last_stripped.endswith(",")
                    ):
                        return f"절단 감지(파일 끝 불완전): '…{last_stripped[-40:]}'"
            return None

        # 1차 시도
        # [Fix 2026-04-23] 기존 max_tokens=8192 기본값 + timeout=20s 로는
        # 복잡한 PySide6 화면(스타일시트 포함 500+줄)이 자주 절단됐다.
        # WRITE 분석가와 동일하게 16384/120s 로 상향.
        # [P7-B 2026-04-25] metered_llm_call 로 교체 — 활성 MetricsRecorder
        # 있으면 자동 카운트, 없으면 패스스루 (기존 동작 보존).
        try:
            from eidos_metrics_recorder import metered_llm_call as _metered
        except Exception:
            _metered = get_llm_response_async  # 측정 모듈 부재 시 폴백
        try:
            # [Fix 2026-04-25 A] 스캐폴딩 LLM 호출 캐시 우회 — 같은 prompt 의
            # broken 응답이 캐시로 재현되어 P0/P1/P2 효과를 모두 우회시키던
            # 문제. 스캐폴딩은 결정적·일관된 새 LLM 응답이 필요하므로 캐시 X.
            result = _clean(await _metered(
                prompt, max_tokens=16384, timeout=120, use_cache=False,
            ))
        except Exception as e:
            _cb(f"⚠️ {filename} 1차 LLM 호출 실패: {e} — skeleton 유지")
            return filename, skeleton_code

        err = _validate(result)
        if err is None:
            _cb(f"✓ {filename} 구현 완료 ({len(result)}자, 검증 통과)")
            return filename, result

        # 재생성 1회 — 실패 원인을 피드백으로 붙여 전체 파일 재출력 요구
        _cb(f"🔄 {filename} 검증 실패 — 재생성 시도: {err}")
        retry_prompt = prompt + (
            "\n\n[★ 이전 생성물 검증 실패 — 반드시 교정 후 파일 전체 재출력]\n"
            f"- 원인: {err}\n"
            "- 교정 규칙: 문법 오류는 해당 줄과 그 주변을 다시 작성하고, "
            "중복 정의는 한 번만 남기고 나머지 삭제. "
            "skeleton 구조(클래스/메서드 시그니처)는 여전히 유지. "
            "마크다운 코드블록 없이 완성된 .py 파일 전체만 출력."
        )
        try:
            # [Fix 2026-04-23] 재생성도 동일 한도(1차와 같음)
            # [Fix 2026-04-25 A] use_cache=False 동일 적용
            result2 = _clean(await _metered(
                retry_prompt, max_tokens=16384, timeout=120, use_cache=False,
            ))
        except Exception as e:
            _cb(f"⚠️ {filename} 재생성 LLM 실패: {e} — skeleton 유지")
            return filename, skeleton_code

        err2 = _validate(result2)
        if err2 is None:
            _cb(f"✓ {filename} 재생성 통과 ({len(result2)}자)")
            return filename, result2

        # ── [Fix 2026-04-25] autoclose 폴백 ────────────────────────────
        # 1차/2차 모두 끝부분 절단으로 죽는 케이스(settings_screen 등)에서
        # 끝 N줄을 깎아 ast.parse 가 통과하는 최대 지점을 살려 본다. 60%
        # 이상 보존되고 절단류 에러일 때만 시도. SyntaxError(line 1, ...) 같은
        # 시작부 깨짐은 autoclose 로 못 살리므로 skip.
        _trunc_kw = ("절단 감지", "파일 끝 불완전")
        for _candidate, _label in ((result, "1차"), (result2, "재생성")):
            _err_check = _validate(_candidate)
            if _err_check is None:
                continue
            if not any(_kw in (_err_check or "") for _kw in _trunc_kw):
                continue
            _closed = _try_autoclose(_candidate)
            if _closed and _validate(_closed) is None:
                _cb(f"🩹 {filename} autoclose 성공 ({_label} {len(_candidate)}자 → {len(_closed)}자)")
                return filename, _closed

        _cb(f"⚠️ {filename} 재생성도 실패 ({err2}) — skeleton 유지")
        return filename, skeleton_code

    # ── [Phase B-2] 실행 모드 결정: contract 있으면 topological, 없으면 기존 병렬 ──
    implemented: Dict[str, str] = {}

    if use_contract:
        try:
            from graphlib import TopologicalSorter, CycleError
            _ts = TopologicalSorter(dep_graph)
            try:
                _ts.prepare()
                _topo_ready = True
            except CycleError as _ce:
                _cb(f"⚠️ contract 의존 그래프 사이클 감지 — 병렬 모드로 폴백: {_ce}")
                _topo_ready = False
        except Exception as _ge:
            _cb(f"⚠️ TopologicalSorter 초기화 실패 — 병렬 모드로 폴백: {_ge}")
            _topo_ready = False

        if _topo_ready:
            _cb(
                f"⚙️ {len(skeletons)}개 파일 contract 기반 topological 구현 시작 "
                f"(의존 그래프 노드 {len(dep_graph)}개)"
            )
            _batch_idx = 0
            while _ts.is_active():
                _batch = list(_ts.get_ready())
                if not _batch:
                    break
                _batch_idx += 1
                _cb(
                    f"📦 배치 #{_batch_idx} — {len(_batch)}개 파일 병렬 구현: "
                    f"{', '.join(_batch[:5])}{'…' if len(_batch) > 5 else ''}"
                )

                # 각 파일에 대해 contract directive + 의존 모듈 컨텍스트 빌드
                _batch_tasks = []
                for _fn in _batch:
                    if _fn not in skeletons:
                        # contract 가 skeletons 에 없는 파일 언급 — 무시하고 done 처리
                        continue
                    # 1) 이 파일이 보유해야 할 exports + 메서드
                    _own_exports = _exports_block(_fn)
                    _own_methods = _methods_block(_fn)
                    # 2) 의존 모듈의 exports + 메서드 + 이미 구현된 본문(있으면)
                    _dep_blocks: List[str] = []
                    for _dep_fn in sorted(dep_graph.get(_fn, set())):
                        _dep_exp = _exports_block(_dep_fn)
                        _dep_methods = _methods_block(_dep_fn)
                        _dep_body = implemented.get(_dep_fn, "")
                        _block = f"--- [{_dep_fn} — 확정된 인터페이스] ---"
                        if _dep_exp:
                            _block += f"\n{_dep_exp}"
                        if _dep_methods:
                            _block += f"\n\n[{_dep_fn} 의 메서드 — 호출 시 이 이름·시그니처 그대로 사용]\n{_dep_methods}"
                        if _dep_body:
                            # 본문이 너무 크면 시그니처 위주로 잘라냄
                            _body_excerpt = _dep_body[:2400]
                            _block += f"\n\n[{_dep_fn} 실제 구현 (앞 2400자)]\n{_body_excerpt}"
                        _dep_blocks.append(_block)

                    _ctx_parts: List[str] = []
                    if _own_exports:
                        _ctx_parts.append(
                            "[★ 모듈 contract — 이 파일은 반드시 다음 top-level 심볼을 export 한다]\n"
                            f"{_own_exports}\n"
                            "위 시그니처는 다른 파일이 import 하는 계약이다. 이름·인자·타입을 정확히 일치시켜라."
                        )
                    if _own_methods:
                        # [G 2026-04-26] 클래스 인스턴스 메서드 contract — 자유 작명 차단
                        _ctx_parts.append(
                            "[★ 클래스 메서드 contract — 다음 메서드를 정확한 이름·시그니처로 모두 정의]\n"
                            f"{_own_methods}\n"
                            "이 메서드들은 다른 파일이 호출하는 계약이다. 누락 시 AttributeError 로 앱이 실행되지 않는다.\n"
                            "메서드 이름을 임의로 바꾸지 말 것 (예: get_task → get_task_by_id 같은 동의어 변형 금지)."
                        )
                    if _dep_blocks:
                        _ctx_parts.append(
                            "[★ 이 파일이 의존하는 형제 모듈 — 시그니처·본문 기준으로 호출하라]\n"
                            + "\n\n".join(_dep_blocks)
                        )
                    _override = "\n\n".join(_ctx_parts) if _ctx_parts else ""

                    _batch_tasks.append(
                        _implement_one(
                            _fn,
                            skeletons[_fn],
                            deps_context_override=_override,
                        )
                    )

                if _batch_tasks:
                    _batch_results = await asyncio.gather(*_batch_tasks)
                    for _fname, _code in _batch_results:
                        implemented[_fname] = _code
                # contract 에 안 들린 파일도 있을 수 있으므로 batch 전체를 done 처리
                for _fn in _batch:
                    _ts.done(_fn)

            # contract 에 누락된 파일이 있으면 마지막에 병렬 처리로 흡수
            _missing_files = [fn for fn in skeletons if fn not in implemented]
            if _missing_files:
                _cb(
                    f"📦 배치 #{_batch_idx + 1} (contract 외 잔여) — "
                    f"{len(_missing_files)}개 파일 병렬 구현"
                )
                _missing_tasks = [
                    _implement_one(fn, skeletons[fn]) for fn in _missing_files
                ]
                _missing_results = await asyncio.gather(*_missing_tasks)
                for _fname, _code in _missing_results:
                    implemented[_fname] = _code
        else:
            # 사이클 / 초기화 실패 → 기존 병렬 모드
            _cb(f"⚙️ {len(skeletons)}개 파일 병렬 구현 시작 (contract 폴백)...")
            tasks = [_implement_one(fname, skel) for fname, skel in skeletons.items()]
            results_list = await asyncio.gather(*tasks)
            implemented = dict(results_list)
    else:
        # contract 없음 — 기존 동작 그대로
        _cb(f"⚙️ {len(skeletons)}개 파일 병렬 구현 시작...")
        tasks = [
            _implement_one(fname, skel)
            for fname, skel in skeletons.items()
        ]
        results_list = await asyncio.gather(*tasks)
        implemented = dict(results_list)

    # ── [P2 2026-04-22] 크로스 파일 import 검증 + 타겟 재구현 ─────────────
    # 파일별 병렬 LLM 호출이라 A.py 가 `from B import X` 했는데 B.py 에 X 가 없어
    # ImportError 로 실행 안 되는 경우가 자주 생긴다. top-level 심볼 인덱스를 만들어
    # 누락을 집계 → 누락 심볼이 있는 의존 파일만 추가 지시문 달아 재구현(1회 병렬).
    try:
        # [Fix 2026-04-26 B] subdirectory 파일도 검증하도록 키 체계 수정.
        # `widgets/sidebar_widget.py` 같은 키를 fn[:-3] 그대로 쓰면
        # `widgets/sidebar_widget` 으로 인덱싱되고, import 의 mod 는
        # `sidebar_widget` (마지막 . 토큰) 이라 매칭 실패 → 검증 skip.
        # 해법: 인덱스는 basename 기반 (sidebar_widget), 실제 디스크 접근용
        # 풀패스(skeletons / implemented 키)는 별도 dict 유지.
        _local_modules = {
            _os_sc.path.basename(fn)[:-3]
            for fn in implemented if fn.endswith(".py")
        }
        # basename 모듈명 → 풀패스 (skeletons/implemented 의 실제 key)
        _full_by_basename: Dict[str, str] = {}
        for _fn in implemented:
            if _fn.endswith(".py"):
                _bn = _os_sc.path.basename(_fn)[:-3]
                # 충돌 시 처음 본 것 유지 (동일 basename 두 개면 별개 문제)
                _full_by_basename.setdefault(_bn, _fn)

        def _top_symbols(code: str) -> set:
            try:
                tree = ast.parse(code or "")
            except Exception:
                return set()
            out: set = set()
            for node in tree.body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.add(node.name)
                elif isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            out.add(t.id)
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    out.add(node.target.id)
            return out

        def _local_from_imports(code: str) -> List[Tuple[str, List[str]]]:
            try:
                tree = ast.parse(code or "")
            except Exception:
                return []
            out: List[Tuple[str, List[str]]] = []
            for node in tree.body:
                if isinstance(node, ast.ImportFrom):
                    mod = (node.module or "").split(".")[-1]
                    if mod in _local_modules:
                        nms = [a.name for a in node.names if a.name != "*"]
                        if nms:
                            out.append((mod, nms))
            return out

        _symbols_by_mod: Dict[str, set] = {
            _os_sc.path.basename(fn)[:-3]: _top_symbols(code)
            for fn, code in implemented.items() if fn.endswith(".py")
        }

        _missing_by_mod: Dict[str, set] = {}
        for fn, code in implemented.items():
            for dep_mod, requested in _local_from_imports(code):
                have = _symbols_by_mod.get(dep_mod, set())
                for sym in requested:
                    if sym not in have:
                        _missing_by_mod.setdefault(dep_mod, set()).add(sym)

        if _missing_by_mod:
            _total_missing = sum(len(v) for v in _missing_by_mod.values())
            _cb(
                f"🔗 크로스 파일 검증 — 누락 심볼 {_total_missing}개 / "
                f"{len(_missing_by_mod)}개 파일 재구현"
            )
            _retry_files: List[str] = []
            _retry_tasks = []
            for dep_mod, missing in _missing_by_mod.items():
                # [Fix 2026-04-26 B] basename → 풀패스 변환
                dep_fn = _full_by_basename.get(dep_mod, f"{dep_mod}.py")
                if dep_fn not in skeletons:
                    _cb(f"  ⚠️ {dep_fn} skeleton 없음 — 재구현 불가")
                    continue
                directive = (
                    "[★ 크로스 파일 일관성 — 이 파일에 반드시 다음 top-level 심볼을 정의]\n"
                    + "\n".join(f"- {s}" for s in sorted(missing))
                    + f"\n\n다른 파일이 `from {dep_mod} import ...` 로 위 심볼을 가져간다. "
                    "누락되면 ImportError 로 앱이 실행되지 않는다. "
                    "skeleton 구조는 유지하되 이 심볼들이 반드시 포함되도록 보강해 재출력."
                )
                _retry_files.append(dep_fn)
                _retry_tasks.append(_implement_one(dep_fn, skeletons[dep_fn], extra_directives=directive))

            if _retry_tasks:
                _retry_results = await asyncio.gather(*_retry_tasks)
                for fn, code in _retry_results:
                    _bn = _os_sc.path.basename(fn)[:-3]
                    new_syms = _top_symbols(code)
                    still = _missing_by_mod.get(_bn, set()) - new_syms
                    if still:
                        _cb(f"  ⚠️ {fn} 재구현 후에도 누락: {sorted(still)} — 기존 결과 유지")
                    else:
                        implemented[fn] = code
                        _symbols_by_mod[_bn] = new_syms
                        _cb(f"  ✓ {fn} 크로스 파일 검증 통과")
        else:
            _cb("🔗 크로스 파일 검증 통과 — 누락 심볼 없음")
    except Exception as _xf_e:
        _cb(f"⚠️ 크로스 파일 검증 패스 실패 (무시): {_xf_e}")

    # ── [P1 2026-04-25] 시그니처 검증 패스 — 호출자/__init__ 인자 정합성 ──
    # 위 import 검증은 "심볼 이름"만 본다. 호출 시그니처
    # (예: main.py 의 `MainWindow(db)` 가 main_window.py 의
    # `__init__(self, db, router, settings, notif)` 와 충돌) 는 이름이
    # 모두 일치하니 통과해버린 채 디스크로 떨어진다 → 첫 실행에서
    # TypeError 보장. 여기서 각 클래스 __init__ 인자 개수와 caller 의
    # 호출 인자 개수를 정적 비교해 mismatch 가 있으면 dep 를 재구현
    # (default 부여 또는 인자 옵션화) 하도록 LLM 1회 추가 호출.
    try:
        def _class_init_sigs(code: str) -> Dict[str, Dict[str, Any]]:
            """파일 내 top-level 클래스의 __init__ 시그니처 추출.
            __init__ 미정의면 default constructor (필수 0)."""
            out: Dict[str, Dict[str, Any]] = {}
            try:
                tree = ast.parse(code or "")
            except Exception:
                return out
            for node in tree.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                init_node = None
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == "__init__":
                        init_node = sub
                        break
                if init_node is None:
                    out[node.name] = {
                        "required_total": 0, "total_pos": 0,
                        "has_varargs": False, "has_kwargs": False,
                        "pos_names": [], "kwonly_required": [],
                    }
                    continue
                a = init_node.args
                pos_only = list(getattr(a, "posonlyargs", []) or [])
                pos_args = a.args[1:] if a.args else []  # self 제외
                total_pos = len(pos_only) + len(pos_args)
                defaults_count = len(a.defaults)  # defaults 는 끝부터 매칭
                required_pos = max(0, total_pos - defaults_count)
                kwonly_req = [
                    arg.arg for arg, d in zip(a.kwonlyargs, a.kw_defaults) if d is None
                ]
                out[node.name] = {
                    "required_total": required_pos + len(kwonly_req),
                    "total_pos": total_pos,
                    "has_varargs": a.vararg is not None,
                    "has_kwargs": a.kwarg is not None,
                    "pos_names": [x.arg for x in pos_only] + [x.arg for x in pos_args],
                    "kwonly_required": kwonly_req,
                }
            return out

        def _call_sites(code: str, target_classes: set):
            """target_classes 에 속하는 클래스의 호출 사이트.
            Returns list of (class_name, n_pos, n_kw, has_starargs, has_kwarg_unpack, lineno)."""
            out_calls = []
            try:
                tree = ast.parse(code or "")
            except Exception:
                return out_calls
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                f = node.func
                cn = None
                if isinstance(f, ast.Name) and f.id in target_classes:
                    cn = f.id
                if not cn:
                    continue
                has_star = any(isinstance(a, ast.Starred) for a in node.args)
                has_kwu = any(k.arg is None for k in node.keywords)
                out_calls.append(
                    (cn, len(node.args), len(node.keywords),
                     has_star, has_kwu, getattr(node, "lineno", 0))
                )
            return out_calls

        _init_sigs_by_mod: Dict[str, Dict[str, Dict[str, Any]]] = {
            _os_sc.path.basename(fn)[:-3]: _class_init_sigs(code)
            for fn, code in implemented.items() if fn.endswith(".py")
        }

        _sig_mismatches: List[Dict[str, Any]] = []
        for fn, code in implemented.items():
            if not fn.endswith(".py"):
                continue
            # 이 파일이 import 하는 ClassName → dep_mod 매핑
            _imp_to_mod: Dict[str, str] = {}
            try:
                _tree = ast.parse(code or "")
                for node in _tree.body:
                    if isinstance(node, ast.ImportFrom):
                        mod = (node.module or "").split(".")[-1]
                        if mod in _local_modules:
                            for al in node.names:
                                if al.name and al.name != "*":
                                    _imp_to_mod[al.asname or al.name] = mod
            except Exception:
                continue
            if not _imp_to_mod:
                continue
            for cn, n_pos, n_kw, has_star, has_kwu, lineno in _call_sites(
                code, set(_imp_to_mod.keys())
            ):
                if has_star or has_kwu:
                    continue  # 정적 검증 불가 — *args/**kwargs 언패킹
                dep_mod = _imp_to_mod.get(cn)
                if not dep_mod:
                    continue
                sig = _init_sigs_by_mod.get(dep_mod, {}).get(cn)
                if not sig:
                    continue
                n_total = n_pos + n_kw
                too_few = n_total < sig["required_total"]
                too_many_pos = (n_pos > sig["total_pos"]) and not sig["has_varargs"]
                if too_few or too_many_pos:
                    _sig_mismatches.append({
                        "caller": fn, "dep": f"{dep_mod}.py", "class": cn,
                        "required_total": sig["required_total"],
                        "total_pos": sig["total_pos"],
                        "got_total": n_total, "got_pos": n_pos, "got_kw": n_kw,
                        "pos_names": sig["pos_names"],
                        "kwonly_required": sig["kwonly_required"],
                        "line": lineno,
                        "kind": "too_few" if too_few else "too_many_pos",
                    })

        if _sig_mismatches:
            _cb(f"🧬 시그니처 검증 — mismatch {len(_sig_mismatches)}건 감지")
            _by_dep: Dict[str, List[Dict[str, Any]]] = {}
            for m in _sig_mismatches:
                _by_dep.setdefault(m["dep"], []).append(m)

            _sig_retry_tasks = []
            _sig_retry_keys: List[str] = []
            for dep_fn, ms in _by_dep.items():
                # [Fix 2026-04-26 B] dep_fn 은 sig 검사 시 basename + ".py"
                # 형태일 수 있어 풀패스로 변환 후 skeletons 매칭
                _dep_bn = _os_sc.path.basename(dep_fn)[:-3] if dep_fn.endswith(".py") else dep_fn
                _dep_full = _full_by_basename.get(_dep_bn, dep_fn)
                if _dep_full not in skeletons:
                    _cb(f"  ⚠️ {_dep_full} skeleton 없음 — 시그니처 재구현 불가")
                    continue
                _lines = []
                for m in ms:
                    _lines.append(
                        f"- {m['caller']}:line {m['line']} 에서 {m['class']}(...) 를 "
                        f"positional {m['got_pos']}개 + keyword {m['got_kw']}개 "
                        f"= 총 {m['got_total']}개 인자로 호출. "
                        f"현재 __init__ 는 필수 인자 {m['required_total']}개 "
                        f"(positional {m['total_pos']}개: {m['pos_names']})"
                        + (f", kwonly 필수: {m['kwonly_required']}" if m['kwonly_required'] else "")
                        + f". [{m['kind']}]"
                    )
                directive = (
                    "[★ 호출자 시그니처 정렬 — 첫 실행 TypeError 차단]\n"
                    "다른 파일에서 이 파일의 클래스를 다음과 같이 호출한다:\n"
                    + "\n".join(_lines)
                    + "\n\n해결: 위 호출 패턴이 그대로 작동하도록 __init__ 를 보정하라.\n"
                    "  방법 A) 호출자가 안 넘기는 인자에 default 값 부여하여 옵션화\n"
                    "          예: def __init__(self, db, router=None, settings=None, notif=None):\n"
                    "  방법 B) 미사용 인자를 시그니처에서 제거하고 내부에서 db 등으로 우회 획득\n"
                    "skeleton 의 클래스명/메서드명/필드는 유지. __init__ 시그니처만 보정해 재출력."
                )
                _sig_retry_keys.append(_dep_full)
                _sig_retry_tasks.append(
                    _implement_one(_dep_full, skeletons[_dep_full],
                                   extra_directives=directive)
                )

            if _sig_retry_tasks:
                _sig_results = await asyncio.gather(*_sig_retry_tasks)
                _improved = 0
                # 결과 매칭 키도 풀패스 → basename 으로 정규화
                for fn, new_code in _sig_results:
                    _fn_bn = _os_sc.path.basename(fn)[:-3] if fn.endswith(".py") else fn
                    new_sigs = _class_init_sigs(new_code)
                    still_bad = False
                    # _by_dep 키는 m["dep"] (basename + ".py" 또는 풀패스)
                    # 양쪽 다 시도
                    _ms_for_fn = (
                        _by_dep.get(fn, [])
                        or _by_dep.get(f"{_fn_bn}.py", [])
                    )
                    for m in _ms_for_fn:
                        ns = new_sigs.get(m["class"])
                        if not ns:
                            still_bad = True
                            break
                        if m["got_total"] < ns["required_total"]:
                            still_bad = True
                            break
                        if (m["got_pos"] > ns["total_pos"]) and not ns["has_varargs"]:
                            still_bad = True
                            break
                    if still_bad:
                        _cb(f"  ⚠️ {fn} 시그니처 재구현 후에도 mismatch — 기존 유지")
                    else:
                        implemented[fn] = new_code
                        _init_sigs_by_mod[_fn_bn] = new_sigs
                        try:
                            _symbols_by_mod[_fn_bn] = _top_symbols(new_code)
                        except Exception:
                            pass
                        _improved += 1
                        _cb(f"  ✓ {fn} 시그니처 재구현 통과")
                _cb(f"🧬 시그니처 재구현 종료 — {_improved}/{len(_sig_retry_keys)}개 정렬")
        else:
            _cb("🧬 시그니처 검증 통과 — mismatch 없음")
    except Exception as _sig_e:
        _cb(f"⚠️ 시그니처 검증 패스 실패 (무시): {_sig_e}")

    # ── [F 2026-04-26] 메서드 호출 시그니처 검증 패스 ────────────────────
    # P1 은 __init__ 만 봤다. 인스턴스 메서드 호출 (self.db.get_task_by_id())
    # 이 db 의 클래스(Database) 에 정의되지 않았으면 첫 실행에서 무더기
    # AttributeError. 8파일 병렬 LLM 호출로 호출자/피호출자 LLM 이 같은
    # 도메인 개념을 다른 이름 (get_task vs get_task_by_id) 으로 짓는 문제.
    # self.<attr>.<method>() 패턴을 정적 추적해 누락 감지 → 정의 파일 LLM
    # 재구현으로 메서드 추가 지시.
    try:
        def _class_methods(code: str) -> Dict[str, Dict[str, Any]]:
            """클래스별 메서드 시그니처 + 부모 정보.
            Returns: {class_name: {"_methods": {method: {sig}}, "_has_external_parent": bool}}
            non-local 부모 (QWidget 등) 가 있으면 _has_external_parent=True — 메서드
            검증 시 false positive 방지에 사용."""
            out: Dict[str, Dict[str, Any]] = {}
            try:
                tree = ast.parse(code or "")
            except Exception:
                return out
            for node in tree.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                # 부모 base 확인 — non-local Name 만 있어도 external 로 간주
                # (QWidget, QMainWindow 등). 안전하게: 어떤 base 든 있으면 external.
                _has_ext_parent = False
                for b in (node.bases or []):
                    if isinstance(b, ast.Name) and b.id in (
                        "object", "Exception", "BaseException"
                    ):
                        continue  # 무해한 부모 — Qt 메서드 상속 없음
                    _has_ext_parent = True
                    break
                methods: Dict[str, Dict[str, Any]] = {}
                for sub in node.body:
                    if not isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    if sub.name == "__init__":
                        continue
                    a = sub.args
                    pos_only = list(getattr(a, "posonlyargs", []) or [])
                    pos_args = a.args[1:] if a.args else []
                    total_pos = len(pos_only) + len(pos_args)
                    defaults_count = len(a.defaults)
                    required_pos = max(0, total_pos - defaults_count)
                    kwonly_req = [
                        arg.arg for arg, d in zip(a.kwonlyargs, a.kw_defaults) if d is None
                    ]
                    methods[sub.name] = {
                        "required_total": required_pos + len(kwonly_req),
                        "total_pos": total_pos,
                        "has_varargs": a.vararg is not None,
                        "has_kwargs": a.kwarg is not None,
                    }
                out[node.name] = {
                    "_methods": methods,
                    "_has_external_parent": _has_ext_parent,
                }
            return out

        def _attr_to_class(code: str, imp_to_mod: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
            """self.<attr> 의 클래스 추적해 attr → (class_name, dep_mod) 매핑.
            지원 패턴:
              1. self.<attr> = <ClassName>(...) — 인스턴스화 직접
              2. self.<attr> = <param_name> + __init__ param 의 type annotation
                 (예: def __init__(self, db: Database): self.db = db)
            ClassName 이 로컬 import 가 아니면 매핑 안 함."""
            out: Dict[str, Tuple[str, str]] = {}
            try:
                tree = ast.parse(code or "")
            except Exception:
                return out
            # 1차 패스: 클래스별 __init__ 의 param 타입 인덱스
            # {class_node id: {param_name: annotated_class_name}}
            _param_types: Dict[int, Dict[str, str]] = {}
            for cnode in tree.body:
                if not isinstance(cnode, ast.ClassDef):
                    continue
                for sub in cnode.body:
                    if not (isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and sub.name == "__init__"):
                        continue
                    pmap: Dict[str, str] = {}
                    args = sub.args
                    all_args = (
                        list(getattr(args, "posonlyargs", []) or [])
                        + (args.args[1:] if args.args else [])
                        + list(args.kwonlyargs or [])
                    )
                    for a in all_args:
                        if a.annotation is None:
                            continue
                        # 단순 Name annotation 만 인식 (Database, MyClass)
                        if isinstance(a.annotation, ast.Name):
                            ann = a.annotation.id
                            if ann in imp_to_mod:
                                pmap[a.arg] = ann
                        # ast.Subscript (Optional[Database] 등) 도 일부 인식
                        elif isinstance(a.annotation, ast.Subscript):
                            slc = a.annotation.slice
                            if isinstance(slc, ast.Name) and slc.id in imp_to_mod:
                                pmap[a.arg] = slc.id
                    _param_types[id(sub)] = pmap

            # 2차 패스: self.<attr> = ... 추적
            for cnode in tree.body:
                if not isinstance(cnode, ast.ClassDef):
                    continue
                for sub in cnode.body:
                    if not (isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and sub.name == "__init__"):
                        continue
                    pmap = _param_types.get(id(sub), {})
                    for stmt in ast.walk(sub):
                        if not isinstance(stmt, ast.Assign):
                            continue
                        for tgt in stmt.targets:
                            if not isinstance(tgt, ast.Attribute):
                                continue
                            if not (isinstance(tgt.value, ast.Name) and tgt.value.id == "self"):
                                continue
                            attr_name = tgt.attr
                            v = stmt.value
                            cn: Optional[str] = None
                            # 패턴 1: self.x = ClassName(...)
                            if isinstance(v, ast.Call) and isinstance(v.func, ast.Name):
                                if v.func.id in imp_to_mod:
                                    cn = v.func.id
                            # 패턴 2: self.x = param (param 의 annotation)
                            elif isinstance(v, ast.Name) and v.id in pmap:
                                cn = pmap[v.id]
                            if cn:
                                out[attr_name] = (cn, imp_to_mod[cn])

            # 3차: top-level (클래스 외부) self 가 아닌 일반 assignment 도 인스턴스화는
            # 잡고 싶지만, 메서드 호출 검증은 self.<attr>.<method> 패턴이라 일반
            # 변수 추적은 불필요. skip.
            return out

        def _method_call_sites(code: str, attr_set: set):
            """self.<attr>.<method>(...) 호출 추출.
            Returns list of (attr_name, method_name, n_pos, n_kw, lineno)."""
            out_calls = []
            try:
                tree = ast.parse(code or "")
            except Exception:
                return out_calls
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                f = node.func
                if not isinstance(f, ast.Attribute):
                    continue
                inner = f.value
                if not isinstance(inner, ast.Attribute):
                    continue
                if not (isinstance(inner.value, ast.Name) and inner.value.id == "self"):
                    continue
                attr_name = inner.attr
                method_name = f.attr
                if method_name.startswith("_"):
                    continue  # private 메서드 검증 skip
                if attr_name not in attr_set:
                    continue
                has_star = any(isinstance(a, ast.Starred) for a in node.args)
                has_kwu = any(k.arg is None for k in node.keywords)
                if has_star or has_kwu:
                    continue  # 정적 검증 불가
                out_calls.append((
                    attr_name, method_name,
                    len(node.args), len(node.keywords),
                    getattr(node, "lineno", 0)
                ))
            return out_calls

        # 클래스별 메서드 인덱스 (basename 키)
        _methods_by_mod: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {
            _os_sc.path.basename(fn)[:-3]: _class_methods(code)
            for fn, code in implemented.items() if fn.endswith(".py")
        }

        _method_mismatches: List[Dict[str, Any]] = []
        for fn, code in implemented.items():
            if not fn.endswith(".py"):
                continue
            # ClassName → dep_mod 매핑 (이 파일이 import 하는 클래스)
            _imp_to_mod: Dict[str, str] = {}
            try:
                _tree = ast.parse(code or "")
                for node in _tree.body:
                    if isinstance(node, ast.ImportFrom):
                        mod = (node.module or "").split(".")[-1]
                        if mod in _local_modules:
                            for al in node.names:
                                if al.name and al.name != "*":
                                    _imp_to_mod[al.asname or al.name] = mod
            except Exception:
                continue
            if not _imp_to_mod:
                continue
            # self.<attr> → (class_name, dep_mod) 매핑
            _attr_class = _attr_to_class(code, _imp_to_mod)
            if not _attr_class:
                continue
            # 메서드 호출 추출 + 누락 감지
            for attr_name, method_name, n_pos, n_kw, lineno in _method_call_sites(
                code, set(_attr_class.keys())
            ):
                cn, dep_mod = _attr_class[attr_name]
                class_info = _methods_by_mod.get(dep_mod, {}).get(cn, {})
                # 외부 부모(QWidget/QMainWindow) 가 있으면 메서드 상속될 수 있어
                # false positive 위험 → 검증 skip. Database 처럼 부모 없는 클래스는
                # 정확히 잡힘 (사용자 케이스의 진짜 진범).
                if class_info.get("_has_external_parent"):
                    continue
                class_methods = class_info.get("_methods", {})
                if method_name not in class_methods:
                    _method_mismatches.append({
                        "caller": fn,
                        "dep": f"{dep_mod}.py",
                        "class": cn,
                        "method": method_name,
                        "got_pos": n_pos,
                        "got_kw": n_kw,
                        "line": lineno,
                        "available": sorted(class_methods.keys())[:20],
                    })

        if _method_mismatches:
            _cb(f"🔧 메서드 호출 검증 — 누락 메서드 호출 {len(_method_mismatches)}건 감지")
            # dep 별 그룹
            _by_dep_m: Dict[str, List[Dict[str, Any]]] = {}
            for m in _method_mismatches:
                _by_dep_m.setdefault(m["dep"], []).append(m)

            _retry_tasks_m = []
            _retry_keys_m: List[str] = []
            for dep_fn, ms in _by_dep_m.items():
                _dep_bn = _os_sc.path.basename(dep_fn)[:-3] if dep_fn.endswith(".py") else dep_fn
                _dep_full = _full_by_basename.get(_dep_bn, dep_fn)
                if _dep_full not in skeletons:
                    _cb(f"  ⚠️ {_dep_full} skeleton 없음 — 메서드 재구현 불가")
                    continue
                # class 별 누락 메서드 정리
                _by_class: Dict[str, List[Dict[str, Any]]] = {}
                for m in ms:
                    _by_class.setdefault(m["class"], []).append(m)
                _lines = []
                for cn, _entries in _by_class.items():
                    _avail = _entries[0].get("available", [])
                    _lines.append(f"클래스 {cn} (현재 정의된 메서드: {_avail})")
                    for e in _entries:
                        _lines.append(
                            f"  - {e['caller']}:line {e['line']} 에서 "
                            f"{cn}.{e['method']}(positional {e['got_pos']}개 + "
                            f"keyword {e['got_kw']}개) 호출 — 이 메서드 미정의"
                        )
                directive = (
                    "[★ 메서드 호출 정렬 — 첫 실행 AttributeError 차단 (F)]\n"
                    "다른 파일에서 이 파일의 클래스 메서드를 다음과 같이 호출한다:\n"
                    + "\n".join(_lines)
                    + "\n\n해결: 누락된 메서드를 이 파일에 추가하라.\n"
                    "  - 메서드 이름과 인자 개수는 호출 패턴과 정확히 일치.\n"
                    "  - 본문은 클래스 책임에 맞게 sqlite3/UI/도메인 로직 실제 작성. "
                    "    pass 만 두지 말 것.\n"
                    "  - 기존 메서드/필드는 그대로 유지. 새 메서드만 추가."
                )
                _retry_keys_m.append(_dep_full)
                _retry_tasks_m.append(
                    _implement_one(_dep_full, skeletons[_dep_full],
                                   extra_directives=directive)
                )

            if _retry_tasks_m:
                _results_m = await asyncio.gather(*_retry_tasks_m)
                _improved_m = 0
                for fn, new_code in _results_m:
                    _fn_bn = _os_sc.path.basename(fn)[:-3] if fn.endswith(".py") else fn
                    new_classes = _class_methods(new_code)
                    still_bad = False
                    _ms_for_fn = (
                        _by_dep_m.get(fn, [])
                        or _by_dep_m.get(f"{_fn_bn}.py", [])
                    )
                    for m in _ms_for_fn:
                        cls_info = new_classes.get(m["class"], {})
                        cls_m = cls_info.get("_methods", {}) if isinstance(cls_info, dict) else {}
                        if m["method"] not in cls_m:
                            still_bad = True
                            break
                    if still_bad:
                        _cb(f"  ⚠️ {fn} 메서드 재구현 후에도 누락 — 기존 유지")
                    else:
                        implemented[fn] = new_code
                        _methods_by_mod[_fn_bn] = new_classes
                        try:
                            _symbols_by_mod[_fn_bn] = _top_symbols(new_code)
                        except Exception:
                            pass
                        _improved_m += 1
                        _cb(f"  ✓ {fn} 메서드 재구현 통과")
                _cb(f"🔧 메서드 재구현 종료 — {_improved_m}/{len(_retry_keys_m)}개 정렬")
        else:
            _cb("🔧 메서드 호출 검증 통과 — 누락 호출 없음")
    except Exception as _mc_e:
        _cb(f"⚠️ 메서드 호출 검증 패스 실패 (무시): {_mc_e}")

    # ── [P2 2026-04-22] pass-only 비율 높은 파일 타겟 재구현 ──────────────
    # 기존: 비율만 로그에 찍고 assets 폴백 결정을 호출자에 위임 → 사용자가
    # 실행 가능한 앱을 아예 못 받는 경우가 많았다. 이제 파일별로 pass-only
    # 비율을 계산해 >50% 인 파일만 강한 지시문으로 재구현 1회. 전체 토큰
    # 낭비 없이 품질 끌어올림.
    def _file_po_ratio(code: str) -> Tuple[int, int]:
        if not code:
            return (0, 0)
        try:
            t = ast.parse(code)
        except Exception:
            return (0, 0)
        total = 0
        po = 0
        for node in ast.walk(t):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                total += 1
                body = [
                    n for n in node.body
                    if not (
                        isinstance(n, ast.Expr)
                        and isinstance(n.value, ast.Constant)
                        and isinstance(n.value.value, str)
                    )
                ]
                if len(body) == 0 or (len(body) == 1 and isinstance(body[0], ast.Pass)):
                    po += 1
        return (po, total)

    try:
        _bad_files: List[str] = []
        for fn, code in implemented.items():
            if not fn.endswith(".py") or fn == "main.py":
                # main.py 는 본디 진입점이라 함수 본문 짧은 게 정상 → 제외
                continue
            _p, _t = _file_po_ratio(code)
            if _t >= 3 and _p / _t > 0.5:
                _bad_files.append(fn)
        if _bad_files:
            _cb(
                f"🛠️ pass-only 비율 높은 파일 {len(_bad_files)}개 재구현: "
                f"{', '.join(_bad_files[:5])}"
            )
            _po_directive = (
                "[★ 이전 생성물 품질 부족 — pass 만 남은 메서드가 과반]\n"
                "- 모든 메서드 본문에서 실제 동작 코드를 작성하라. "
                "`pass`, `...`, `return None` 단독 금지.\n"
                "- todo 문구에 명시된 조작(DB 쿼리 실행, UI 위젯 생성/배치, "
                "시그널 연결, 리스트 리프레시 등)을 실제 코드로 풀어 써라.\n"
                "- 시그니처·클래스명은 skeleton 그대로 유지. 새 메서드 추가 금지.\n"
                "- PySide6 화면이면 _init_ui 에서 위젯 생성·배치·setStyleSheet 전부 작성. "
                "DB 파일이면 sqlite3 연결·CREATE·INSERT·SELECT 실제 구문을 넣어라."
            )
            _retry_tasks = [
                _implement_one(fn, skeletons.get(fn, implemented[fn]),
                               extra_directives=_po_directive)
                for fn in _bad_files if fn in skeletons
            ]
            if _retry_tasks:
                _po_results = await asyncio.gather(*_retry_tasks)
                _improved = 0
                for fn, new_code in _po_results:
                    old_p, old_t = _file_po_ratio(implemented.get(fn, ""))
                    new_p, new_t = _file_po_ratio(new_code)
                    # [Fix 2026-04-23] 스왑 조건 버그:
                    # old_t == 0 (AST 파싱 실패) 일 때 기존 로직은
                    # `old_p / max(1, old_t) = 0` 으로 취급해 new 가 절대 통과 못 함.
                    # 기존 코드가 깨져 있으면(파싱 불가) new 가 파싱만 되면 무조건 채택.
                    _swap = False
                    if new_t > 0:
                        if old_t == 0:
                            _swap = True  # 기존이 파싱 불가 → new 가 바로 승리
                        elif (new_p / new_t) < (old_p / old_t):
                            _swap = True
                    if _swap:
                        implemented[fn] = new_code
                        _improved += 1
                        _cb(f"  ✓ {fn} 재구현 개선 ({old_p}/{old_t} → {new_p}/{new_t})")
                    else:
                        _cb(f"  − {fn} 재구현 효과 없음, 이전 유지")
                _cb(f"🛠️ pass-only 재구현 종료 — {_improved}/{len(_bad_files)}개 개선")
    except Exception as _po2e:
        _cb(f"⚠️ pass-only 재구현 패스 실패 (무시): {_po2e}")

    # ── [P4 2026-04-22] pass-only 함수 비율 진단 로그 ─────────────────────
    # 호출자(CodeBuilder scaffold 모드)는 이 비율이 임계 초과면 assets 폴백으로
    # 전환한다. 여기선 수치만 노출 — scaffold/assets 선택은 호출자 책임.
    try:
        _po, _tot = count_pass_only_funcs(implemented)
        if _tot > 0:
            _ratio = _po / _tot
            _cb(f"📊 pass-only 함수 {_po}/{_tot} ({_ratio:.0%})")
            if _ratio > 0.5:
                _cb(f"⚠️ pass-only 비율 {_ratio:.0%} > 50% — 스캐폴딩 품질 부족 의심")
    except Exception as _po_e:
        print(f"  [SkeletonImpl] pass-only 집계 실패 (무시): {_po_e}")

    # ── [P5 2026-04-22] architecture_strategy 일치성 검증 ────────────────
    # DNA 패턴에서 도출된 전략(complexity/data_layer/notification/avoid 등)과
    # 실제 생성 코드 간 불일치를 경고로 노출. fatal 아님 — 오탐 여지 있어 폴백
    # 트리거로는 쓰지 않고 로그로만 드러냄.
    if architecture_strategy is not None:
        try:
            _strat_warnings = _verify_strategy_alignment(implemented, architecture_strategy)
            if _strat_warnings:
                _cb(f"⚠️ 아키텍처 전략 불일치 {len(_strat_warnings)}건:")
                for _w in _strat_warnings:
                    _cb(f"  - {_w}")
            else:
                _cb("✓ 아키텍처 전략 일치성 확인")
        except Exception as _s_e:
            print(f"  [SkeletonImpl] 전략 검증 실패 (무시): {_s_e}")

    _cb(f"✅ 구현 완료 — {len(implemented)}개 파일")
    return implemented

# ── [서버 경로 CTA] suggested_actions 경량 생성 ────────────────────────────
async def get_suggested_actions_async(
    user_input: str,
    assistant_response: str,
    chat_history: List[str] = None,
) -> List[Dict]:
    """
    서버 경로 전용 — 응답 텍스트를 보고 후속 버튼(suggested_actions)만 생성.
    generate_reasoning_and_response_async의 [작업 3]을 단독 호출.
    실패 시 빈 리스트 반환 (메인 응답 흐름에 영향 없음).
    """
    if not model:
        return []

    recent = "\n".join((chat_history or [])[-6:])

    prompt = f"""다음 대화를 보고, 사용자가 자연스럽게 이어갈 수 있는 후속 행동 버튼 목록을 JSON으로만 출력해.

[최근 대화]
{recent}

[사용자 입력]
{user_input[:300]}

[EIDOS 응답]
{assistant_response[:400]}

[규칙]
- 2~4개. 단순 인사/리액션이면 빈 배열 [].
- 각 항목: {{"label": "버튼 텍스트(10자 이내)", "prompt": "클릭 시 입력창에 채워질 지시문(1문장)", "type": "action"|"question"|"browser"}}
- browser 타입은 실제 웹 접속이 필요할 때만. url 필드 포함.
- JSON 배열만 출력. 설명 없이.

[출력 예시]
[{{"label": "더 자세히", "prompt": "방금 내용 더 자세히 설명해줘", "type": "question"}}]"""

    try:
        raw = await get_llm_response_async(prompt, response_mime_type="application/json")
        clean = raw.strip()
        if clean.startswith("```"):
            clean = re.sub(r'^```(?:json)?\s*', '', clean, flags=re.IGNORECASE)
            clean = re.sub(r'\s*```$', '', clean).strip()
        result = json.loads(clean)
        if isinstance(result, list):
            return result[:5]
        # dict로 감싸진 경우 대응
        if isinstance(result, dict):
            for v in result.values():
                if isinstance(v, list):
                    return v[:5]
        return []
    except Exception as e:
        print(f"  [CTA] suggested_actions 생성 실패 (무시): {e}")
        return []

# ── [마일스톤 로드맵] 기획 카드 Step 생성 ──────────────────────────────────────
async def generate_milestone_plan_async(
    top_goal: str,
    context: str = "",
) -> List[Dict]:
    """
    최종 목표를 받아 실행 가능한 Step 목록(로드맵)을 JSON으로 반환.

    반환 형식:
    [
      {
        "title": "Step 제목 (15자 이내)",
        "desc": "구체적 실행 내용 (1~2문장)",
        "kpi": "완료 기준 (측정 가능한 지표)"
      },
      ...
    ]
    """
    if not model:
        return []

    ctx_block = f"\n[현재 상황 맥락]\n{context[:500]}" if context else ""

    prompt = f"""다음 최종 목표를 달성하기 위한 순서 있는 실행 로드맵을 JSON으로만 출력하라.

[최종 목표]
{top_goal}{ctx_block}

[규칙]
- 단계 수: 4~6개. 너무 많으면 실행 불가, 너무 적으면 방향 불명.
- 각 단계는 반드시 이전 단계 완료 후 착수 가능한 순서.
- title: 15자 이내, 행동 중심 (예: "크몽 상세페이지 개선")
- desc: 무엇을 구체적으로 할지 1~2문장. 추상적 표현 금지.
- kpi: 이 단계가 완료됐음을 알 수 있는 측정 가능한 기준 1개.
- 단계별 소요 예상 시간(days) 포함.

[출력 형식 — JSON 배열만, 설명 없이]
[
  {{
    "title": "단계 제목",
    "desc": "구체적 실행 내용",
    "kpi": "완료 기준",
    "days": 7
  }}
]"""

    try:
        raw = await get_llm_response_async(prompt, response_mime_type="application/json")
        clean = raw.strip()
        if clean.startswith("```"):
            clean = re.sub(r'^```(?:json)?\s*', '', clean, flags=re.IGNORECASE)
            clean = re.sub(r'\s*```$', '', clean).strip()
        result = json.loads(clean)
        if isinstance(result, list):
            return result[:6]
        if isinstance(result, dict):
            for v in result.values():
                if isinstance(v, list):
                    return v[:6]
        return []
    except Exception as e:
        print(f"  [Milestone] 로드맵 생성 실패: {e}")
        return []