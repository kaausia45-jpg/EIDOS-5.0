import os
import sys
import glob
import mimetypes
import uuid
import time
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import asyncio

# [2026-05-22] Windows cp949 콘솔에서 이모지 출력 시 UnicodeEncodeError 회피.
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 1. 환경 변수 로드
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID_50_CREDITS")
CLIENT_URL = os.getenv("CLIENT_URL", "http://localhost:3000")

# [2026-05-22 Petalbot] Supabase/Stripe/EIDOS Core 는 lazy — 키 없으면 관련 endpoint 만 비활성.
# Petal endpoint 는 LLM 직접 호출이라 위 인프라 불필요.
PETAL_ONLY_MODE = not all([SUPABASE_URL, SUPABASE_KEY, STRIPE_SECRET_KEY])

supabase = None
stripe = None
eidos_core = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "eidos_files")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
if not os.path.exists(PROJECT_ROOT):
    os.makedirs(PROJECT_ROOT)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if PETAL_ONLY_MODE:
    print(
        "⚠️ [Server] .env 의 SUPABASE_URL/KEY/STRIPE_SECRET_KEY 가 없어요. "
        "Petal 전용 모드로 실행합니다 (`/api/petal/*` 만 동작). "
        "`/api/generate-tool`, `/api/create-checkout-session` 은 503 반환."
    )
else:
    try:
        from supabase import create_client, Client
        import stripe as _stripe_mod
        stripe = _stripe_mod
        stripe.api_key = STRIPE_SECRET_KEY
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)  # type: ignore
        print("✅ [Server] Supabase + Stripe 초기화 완료.")
    except Exception as _e_init:
        print(f"⚠️ [Server] Supabase/Stripe 초기화 실패: {_e_init}")

    try:
        from eidos_v4_0_core import EidosCore
        print("🚀 EIDOS Core 로딩 중...")
        eidos_core = EidosCore(user_project_root=PROJECT_ROOT)
        print("✅ [Server] EIDOS Core 로드 완료.")
    except Exception as _e_core:
        print(f"⚠️ [Server] EIDOS Core 로드 실패: {_e_core}")

app = FastAPI(title="EIDOS API Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 데이터 모델 ---
class ToolGenerationRequest(BaseModel):
    prompt: str
    project_name: Optional[str] = "Untitled Project"

class ToolGenerationResponse(BaseModel):
    status: str
    message: str
    result: Optional[str] = None
    remaining_credits: int


# --- [2026-05-22 Petalbot] 분신 챗봇용 데이터 모델 ──────────────────
class PetalChatTurn(BaseModel):
    q: str
    a: str


class PetalChatRequest(BaseModel):
    text: str
    mode: Optional[str] = "chat"          # chat / customer_reply / marketing / screen_help
    history: Optional[list] = None        # [{"q":"...","a":"..."}, ...] (최근 6 turn cap)
    context: Optional[str] = None         # 화면 캡처 텍스트 / 클립보드 / 외부 컨텍스트
    # [2026-05-23] 파일 첨부 — list of {name, kind, b64}
    # kind: "image" (vision LLM 호출) | "text" (내용 컨텍스트에 합침)
    attachments: Optional[list] = None
    tts_enabled: Optional[bool] = True    # NCP TTS audio_b64 응답 포함 여부


class PetalChatResponse(BaseModel):
    status: str
    answer: str
    mode_used: str
    # [2026-05-23] NCP TTS audio (AIRA voice)
    audio_b64: Optional[str] = None
    audio_format: Optional[str] = None    # "mp3"
    # [2026-05-23 b-3] TTS 평문 (NCP 실패 시 클라이언트 pyttsx3 폴백용)
    spoken: Optional[str] = None
    # [2026-05-23 b-3] NCP 사용 가능 여부 (False = key 미설정 or NCP 실패)
    tts_source: Optional[str] = None      # "ncp" | "none" (서버 진단용)
    error: Optional[str] = None


class PetalClipboardRequest(BaseModel):
    intent: str                            # "summarize" / "polish" / "translate_en" / "polite_tone"
    text: str                              # 원본 텍스트
    target_language: Optional[str] = "ko"


class PetalClipboardResponse(BaseModel):
    status: str
    text: str                              # 클립보드에 넣을 결과
    note: Optional[str] = None             # 사용자에게 보여줄 짧은 설명
    error: Optional[str] = None


# [2026-05-22 Screen Analyze] Petalbot 의 화면 캡처 분석 — Win+PrintScreen
class PetalScreenAnalyzeRequest(BaseModel):
    image_b64: str                         # base64 인코딩 PNG
    mode: Optional[str] = "guide"          # "guide" / "describe" / "fix"
    context: Optional[str] = None          # 사용자가 추가로 알려준 컨텍스트
    history: Optional[list] = None         # 이전 대화


class PetalScreenAnalyzeResponse(BaseModel):
    status: str
    analysis: str                          # 사용자에게 표시할 분석 본문 (마크다운)
    spoken: str                            # AIRA 가 음성으로 말할 짧은 안내 (200자 이내)
    mode_used: str
    audio_b64: Optional[str] = None        # [2026-05-22 B-2] NCP TTS mp3 base64 (AIRA voice)
    audio_format: Optional[str] = None     # "mp3" (NCP 기본)
    error: Optional[str] = None


# [2026-05-30 7W] 육하원칙 빌더 로직 이식 — 상담 정리 + 범위 외 요청 감지
# (원본: '육하원칙 빌더/app.py'. 클라가 직접 Gemini 호출하던 것을 서버 사이드로 통합.
#  스키마 강제(responseSchema) 대신 프롬프트에 JSON 구조를 글로 박는 방식 2 채택 —
#  공유 LLM 모듈 무수정. screen_analyze 의 방어적 JSON 파싱 패턴 재사용.)
class PetalSevenWRequest(BaseModel):
    conversation: str                      # 두서없는 고객 상담 대화 원문
    biz: Optional[str] = None              # 업종/맥락 (선택)


class PetalSevenWResponse(BaseModel):
    status: str
    data: Optional[dict] = None            # 7W 구조 전체 (oneLineSummary, who…howMuch, followUpQuestions, replyDraft)
    error: Optional[str] = None


class PetalScopeCheckRequest(BaseModel):
    scope: dict                            # 고정된 기준 범위 (seven_w 의 data)
    new_request: str                       # 고객의 새 추가 요청 메시지
    rate: Optional[str] = None             # 단가/예산 기준 (선택, 추가견적 산정에 사용)


class PetalScopeCheckResponse(BaseModel):
    status: str
    data: Optional[dict] = None            # overallVerdict, items[], totalAddMin/Max, clientNote
    error: Optional[str] = None


class PetalClarifyDevRequest(BaseModel):
    request: str                           # 사용자의 개발/수정 요청(모호할 수 있음)
    context: Optional[str] = None          # 선택 맥락


class PetalClarifyDevResponse(BaseModel):
    status: str
    data: Optional[dict] = None            # {understanding, approach, whatClear, questions[]}
    error: Optional[str] = None


# [2026-05-22 B-2] NCP TTS helper — eidos_chat_gui 의 _generate_tts_async 와 동일 패턴
def _synthesize_ncp_tts_b64(
    text: str,
    voice: str = "dara",
    speed: Optional[int] = None,
) -> Optional[str]:
    """Naver Clova Voice API 호출 → mp3 audio bytes → base64.

    settings 또는 key_manager 에서 NCP_ID / NCP_KEY 로드. 둘 다 있으면 호출,
    없거나 실패 시 None (호출자가 pyttsx3 폴백).
    """
    if not text or len(text.strip()) < 1:
        return None
    text = text[:300].strip()  # NCP 길이 cap
    # [2026-06-03] speed 미지정 시 공유 설정(aira_shared_settings)에서 사용자 속도 반영
    if speed is None:
        try:
            import aira_shared_settings as _S
            speed = _S.ncp_speed()
        except Exception:
            speed = 0
    try:
        # 1차: key_manager (eidos 표준)
        ncp_id = ""
        ncp_key = ""
        try:
            from key_manager import load_api_key
            ncp_id = (load_api_key("NCP_ID") or "").strip()
            ncp_key = (load_api_key("NCP_KEY") or "").strip()
        except Exception:
            pass
        # 2차: 환경변수
        if not ncp_id:
            ncp_id = os.getenv("NCP_ID", "").strip()
        if not ncp_key:
            ncp_key = os.getenv("NCP_KEY", "").strip()
        # 3차: eidos_settings.json 직접 (key_manager 없을 때)
        if not ncp_id or not ncp_key:
            try:
                import json as _json_ncp
                settings_path = os.path.join(BASE_DIR, "eidos_settings.json")
                if os.path.exists(settings_path):
                    with open(settings_path, "r", encoding="utf-8") as f:
                        _settings = _json_ncp.load(f)
                    if isinstance(_settings, dict):
                        ncp_id = ncp_id or str(_settings.get("ncp_id", "") or "").strip()
                        ncp_key = ncp_key or str(_settings.get("ncp_key", "") or "").strip()
            except Exception:
                pass
        if not ncp_id or not ncp_key:
            return None  # 조용히 폴백
        import requests as _req
        import base64
        import urllib.parse
        url = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "X-NCP-APIGW-API-KEY-ID": ncp_id,
            "X-NCP-APIGW-API-KEY":    ncp_key,
        }
        body = urllib.parse.urlencode({
            "speaker": voice,
            "speed":   str(speed),
            "text":    text,
            "format":  "mp3",
        }).encode("utf-8")
        resp = _req.post(url, headers=headers, data=body, timeout=10)
        if resp.status_code == 200 and resp.content:
            return base64.b64encode(resp.content).decode("ascii")
        print(
            f"⚠️ [NCP TTS] HTTP {resp.status_code} — "
            f"{resp.text[:120] if hasattr(resp, 'text') else ''}"
        )
        return None
    except Exception as e:
        print(f"⚠️ [NCP TTS] 호출 실패: {e}")
        return None

# --- 유틸리티 함수 ---
async def get_current_user(authorization: str = Header(None)) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Token")
    try:
        token = authorization.replace("Bearer ", "")
        user = supabase.auth.get_user(token)
        if not user or not user.user:
            raise HTTPException(status_code=401, detail="Invalid Token")
        return user.user.id
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Auth Failed: {str(e)}")

def clear_output_directory(directory: str):
    """작업 전 청소: 꼬임 방지"""
    if not os.path.exists(directory): return
    for f in glob.glob(os.path.join(directory, "*")):
        try: os.remove(f)
        except: pass

def upload_latest_result_to_supabase(output_dir: str, project_root: str) -> str:
    """
    [스마트 배송 시스템]
    1. .py, .json 등 쓸데없는 파일 무시
    2. 최근 1분 내에 생성된 파일만 취급 (옛날 파일 배송 사고 방지)
    3. UUID 변환으로 한글 에러 원천 차단
    """
    try:
        search_dirs = [output_dir, project_root]
        target_file = None
        target_time = 0
        current_time = time.time()

        for directory in search_dirs:
            if not os.path.exists(directory): continue
            
            # 모든 파일 검색
            files = glob.glob(os.path.join(directory, "*"))
            
            # [필터링] 코드 파일(.py) 및 시스템 파일 제외
            valid_files = [
                f for f in files 
                if os.path.isfile(f) 
                and not f.lower().endswith(('.py', '.json', '.pyc', '.txt', '.md'))
            ]
            
            if not valid_files: continue

            # 가장 최신 파일 찾기
            latest_in_dir = max(valid_files, key=os.path.getmtime)
            latest_file_time = os.path.getmtime(latest_in_dir)

            # [중요] '지금' 기준으로 60초 이내에 만들어진 파일만 인정
            if (current_time - latest_file_time) > 60: 
                continue

            if latest_file_time > target_time:
                target_time = latest_file_time
                target_file = latest_in_dir

        if not target_file:
            print("⚠️ [Upload] 최근 생성된 결과 파일을 찾을 수 없습니다.")
            return None

        print(f"✅ [Upload] 업로드할 파일 발견: {target_file}")
        
        # UUID로 안전한 파일명 생성
        original_filename = os.path.basename(target_file)
        file_ext = os.path.splitext(target_file)[1]
        safe_filename = f"{uuid.uuid4()}{file_ext}"
        
        # MIME 타입 추론
        mime_type, _ = mimetypes.guess_type(target_file)
        if mime_type is None: mime_type = "application/octet-stream"

        # 파일 읽기 및 업로드
        with open(target_file, 'rb') as f:
            file_data = f.read()
        
        storage_path = f"results/{safe_filename}"
        supabase.storage.from_("eidos-files").upload(
            path=storage_path,
            file=file_data,
            file_options={"content-type": mime_type, "upsert": "true"}
        )

        # 다운로드 URL 생성
        public_url = supabase.storage.from_("eidos-files").get_public_url(storage_path)
        url = public_url if isinstance(public_url, str) else public_url
        
        # 프론트엔드가 인식할 태그 반환
        return f"[DOWNLOAD:{original_filename}|{url}]"

    except Exception as e:
        print(f"❌ Upload Failed: {e}")
        return None

# --- API 엔드포인트 ---

@app.post("/api/generate-tool", response_model=ToolGenerationResponse)
async def generate_tool(request: ToolGenerationRequest, user_id: str = Depends(get_current_user)):
    if PETAL_ONLY_MODE or supabase is None or eidos_core is None:
        raise HTTPException(
            status_code=503,
            detail="Petal-only mode — Supabase/Stripe/EIDOS Core 미초기화. .env 설정 필요.",
        )
    COST = 5
    
    # 1. 크레딧 조회 및 차감 로직
    try:
        user_data = supabase.table("profiles").select("credits").eq("id", user_id).single().execute()
        current_credits = user_data.data['credits'] if user_data.data else 0
        
        if current_credits < COST:
            raise HTTPException(status_code=402, detail="크레딧이 부족합니다.")
            
        new_credits = current_credits - COST
        supabase.table("profiles").update({"credits": new_credits}).eq("id", user_id).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Credit Error: {str(e)}")

    # 2. 청소 (이전 작업 잔여물 제거)
    clear_output_directory(OUTPUT_DIR)

    # 3. AI 명령 프롬프트 (강력하게 지시)
    headless_instruction = (
        "\n\n"
        "--- [SYSTEM: FILE GENERATION MODE] ---\n"
        "1. DO NOT plan. DO NOT discuss.\n"
        "2. IMMEDIATELY write Python code to generate the requested file.\n"
        f"3. Save the file to: '{OUTPUT_DIR}' (Absolute Path).\n" # 경로 재강조
        "4. USE 'execute_python_file' tool to run it right now.\n"
        "5. If the file is created, say 'File generated successfully'.\n"
    )
    
    final_prompt = request.prompt + headless_instruction

    try:
        # 4. EIDOS 실행 (작업 등록)
        print("🤖 [API] 작업 요청 등록 중...")
        result_tuple = await eidos_core.process_input(
            text_input=final_prompt,
            image_input=None,
            chat_history=[],
            project_dir=request.project_name
        )
        
        # [!!! 핵심 수정: 실행 대기 루프 (Execution Wait Loop) !!!]
        print("⏳ [API] 작업 실행 대기 중 (Max 60초)...")
        max_retries = 60  # 최대 60초 대기
        task_completed = False
        
        for _ in range(max_retries):
            # 1. 자율 틱 강제 실행 (이게 있어야 코드가 실행됨)
            await eidos_core.autonomous_tick_async()
            
            # 2. 결과 파일이 생겼는지 확인 (가장 확실한 완료 신호)
            # (upload 로직을 미리 체크 용도로 활용하지 않고, 단순 존재 여부만 체크해도 됨)
            # 여기서는 Core 스케줄 상태를 확인하는 것이 정석입니다.
            
            # 스케줄에서 현재 작업 상태 확인
            current_tasks = [t for t in eidos_core.schedule if t.get("status") in ["RUNNING", "PENDING"]]
            completed_tasks = [t for t in eidos_core.schedule if t.get("status") == "COMPLETED"]
            error_tasks = [t for t in eidos_core.schedule if t.get("status") == "ERROR"]

            # 파일이 생성되었는지 확인 (upload 함수 재활용)
            # (주의: 실제 업로드는 루프 끝나고 한 번만 하기 위해 여기선 체크만 하거나, 
            #  틱이 돌 때마다 파일을 확인하고 있으면 break)
            
            # 간단히: 5초마다 파일 시스템 체크 or 스케줄이 비었는지 확인
            if not current_tasks:
                print("✅ [API] 모든 작업 스케줄 완료.")
                task_completed = True
                break
            
            if error_tasks:
                 print(f"❌ [API] 작업 중 오류 발생: {error_tasks[-1].get('last_result')}")
                 # 오류가 나도 파일이 있을 수 있으니 일단 break
                 break

            await asyncio.sleep(1) # 1초 대기

        # 5. 결과 파일 찾아서 업로드
        natural_response = result_tuple[9] # 초기 응답
        download_tag = upload_latest_result_to_supabase(OUTPUT_DIR, PROJECT_ROOT)
        
        final_response = "요청하신 작업이 완료되었습니다."
        if download_tag:
            final_response += f"\n\n{download_tag}"
        else:
            final_response += "\n\n(파일 생성에 실패했거나, 결과 파일을 찾을 수 없습니다. 다시 시도해주세요.)"

        return ToolGenerationResponse(
            status="success",
            message="OK",
            result=final_response,
            remaining_credits=new_credits
        )

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/create-checkout-session")
async def create_checkout_session(user_id: str = Depends(get_current_user)):
    """
    [Stripe] 결제 세션을 생성하고 결제창 URL을 반환합니다.
    """
    if PETAL_ONLY_MODE or stripe is None:
        raise HTTPException(
            status_code=503,
            detail="Petal-only mode — Stripe 미초기화. .env STRIPE_SECRET_KEY 필요.",
        )
    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[
                {
                    'price': STRIPE_PRICE_ID, # .env에 설정한 상품 ID
                    'quantity': 1,
                },
            ],
            mode='payment',
            # 결제 성공/취소 시 돌아갈 프론트엔드 주소
            success_url=CLIENT_URL + '/?payment=success',
            cancel_url=CLIENT_URL + '/?payment=canceled',
            # [중요] 결제 완료 후 Webhook에서 유저를 찾기 위해 메타데이터에 ID 심기
            metadata={
                'user_id': user_id,
                'credits_to_add': '50'
            }
        )
        return {"url": checkout_session.url}
    except Exception as e:
        print(f"Stripe Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ──────────────────────────────────────────────────────────────────
# [2026-05-22 Petalbot] 분신 챗봇 API ─ AIRA 분신 (꽃잎 desktop app)
# ──────────────────────────────────────────────────────────────────
# 인증: Phase 1 PoC = optional (X-Petal-Token 헤더, 환경변수 PETAL_API_KEY 와 매칭).
# Supabase 토큰 인증은 Phase 2 후속 (사용자별 API key 발급).

PETAL_API_KEY = os.getenv("PETAL_API_KEY", "")  # 빈 string = 인증 미사용


def _check_petal_auth(petal_token: Optional[str]) -> None:
    """Petalbot 헤더 토큰 검사. PETAL_API_KEY 가 비어있으면 통과 (로컬 dev)."""
    if not PETAL_API_KEY:
        return
    if not petal_token or petal_token != PETAL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid Petal Token")


_PETAL_MODE_SYS = {
    "chat": (
        "너는 AIRA의 분신 Petalbot. 사용자 옆에 떠있는 작은 꽃잎 도우미다. "
        "★ 반드시 친근한 존댓말(해요체)로 답한다 — 반말 절대 금지(딱딱한 격식체도 X). "
        "친근한 한국어 구어체로 짧게 답해라. 3문장 이내, 마크다운·이모지 최소. "
        "긴 설명이 필요하면 '메인 EIDOS 창에서 설명드릴까요?' 라고 안내."
    ),
    "customer_reply": (
        "너는 고객 응대 전문가. 사용자가 제시한 고객 메시지에 대한 정중한 답변 초안을 작성한다. "
        "톤: 친절·전문성·짧음 (3~5문장). 인사+공감+해결안+마무리 구조. "
        "반드시 사용자가 검토 후 직접 발송할 수 있도록 '복사해서 사용하세요' 라고 끝. "
        "마크다운 헤더 금지, 일반 텍스트만."
    ),
    "marketing": (
        "너는 마케팅 카피라이터. 사용자가 제시한 상품/서비스에 대한 SNS 게시글 또는 "
        "광고 문구를 작성한다. 인스타·트위터 풍, 짧고 시각적, 해시태그 3~5개. "
        "한국어 자연스럽게."
    ),
    "screen_help": (
        "너는 사용자의 화면 도우미 AIRA. 사용자가 첨부한 스크린샷 이미지를 직접 "
        "분석해서 다음에 어떻게 해야 할지 친절하게 알려준다.\n\n"
        "★ 출력 규칙 ★\n"
        "- 스크린샷에서 실제로 보이는 UI 요소·텍스트·에러·상황을 구체적으로 짚어준다 "
        "(예: '오른쪽 위의 X 버튼', '가운데 빨간 박스의 에러 메시지', "
        "'좌측 사이드바의 두 번째 메뉴').\n"
        "- 1~3 단계로 무엇을 클릭/입력/실행해야 하는지 안내한다.\n"
        "- 각 단계는 짧고 명확하게 (한 단계 2~3줄 이내).\n"
        "- 추측이 아니라 실제로 보이는 것 기반으로 답한다. 화면에 없는 기능은 "
        "언급하지 않는다.\n"
        "- 스크린샷이 첨부되지 않았으면 '화면을 못 받았어요, 다시 시도해주세요'라고 "
        "안내한다. 일반 답변(고객 답장 가이드 같은 것)을 만들지 말 것.\n"
        "- 한국어 자연스럽게. 인사·메타설명 금지."
    ),
    # [2026-06-02] 대화형 기획 파트너 — 한 번에 끝내지 않고 여러 턴 상의하며 발전.
    "dev_plan": (
        "너는 외주 개발 PM이자 기획 파트너 Petalbot이다. 사용자와 **함께 아이디어를 "
        "다듬는 대화**를 한다. 한 번에 끝내려 하지 말고, 매 턴 ①지금까지 이해한 핵심을 "
        "1~2줄로 요약하고 ②결정이 필요하거나 빠진 부분을 1~3개 구체적으로 묻거나 제안한다. "
        "기능을 함께 발전시켜라(있으면 좋을 기능·우선순위·화면 구성·대상 사용자·기술 스택). "
        "친근한 한국어 구어체·간결(5~8문장 이내)·마크다운 헤더 금지. 코드는 작성하지 않는다. "
        "충분히 정리됐다 싶으면 가끔 '정리가 충분하면 \"확정\"이라고 해주세요 — 기능명세서로 "
        "정리해 드릴게요'라고 안내한다(매 턴 반복하진 말 것)."
    ),
    # [2026-06-02] 기획 대화 → 기능명세서.txt 평문 문서로 컴파일.
    "dev_spec": (
        "너는 기획 대화를 **기능명세서.txt 문서로 정리**하는 작성기다. 지금까지의 대화를 "
        "바탕으로 아래 형식의 평문(plain text) 문서만 출력하라. 마크다운 코드펜스(```)·메타"
        "설명·인사 금지. 한국어. 대화에서 명확히 정해진 것만 적고, 불명확한 것은 '비고/미결정'"
        "에 둔다. 과장·창작 금지.\n\n"
        "# 기능명세서\n\n"
        "## 1. 프로젝트 개요\n(무엇을·누구를 위해·왜, 한 문단)\n\n"
        "## 2. 목표\n- (핵심 목표 2~4개)\n\n"
        "## 3. 핵심 기능\n- [ ] (기능 1 — 한 줄 설명)\n- [ ] (기능 2)\n\n"
        "## 4. 화면 / 구성\n- (주요 화면·구성요소)\n\n"
        "## 5. 기술 스택 제안\n- (권장 스택과 이유 — 예: PySide6 데스크톱 / FastAPI 서버 / CLI)\n\n"
        "## 6. 비고 / 미결정\n- (아직 안 정해진 것·향후 논의)\n"
    ),
}


@app.post("/api/petal/chat", response_model=PetalChatResponse)
async def petal_chat(
    request: PetalChatRequest,
    x_petal_token: Optional[str] = Header(None),
):
    """[Petalbot] AIRA 분신 챗 — LLM 직접 호출 (간단). mode 별 페르소나."""
    _check_petal_auth(x_petal_token)
    try:
        from llm_module import get_llm_response_async
    except Exception as e:
        return PetalChatResponse(
            status="error",
            answer="",
            mode_used=request.mode or "chat",
            error=f"llm_module import 실패: {e}",
        )
    mode = (request.mode or "chat").lower()
    sys_prompt = _PETAL_MODE_SYS.get(mode, _PETAL_MODE_SYS["chat"])
    # history 직렬화 (최근 6 turn cap)
    hist_block = ""
    try:
        if request.history:
            recent = list(request.history)[-6:]
            lines = []
            for t in recent:
                if isinstance(t, dict):
                    q = str(t.get("q", ""))[:200]
                    a = str(t.get("a", ""))[:200]
                    if q:
                        lines.append(f"사용자: {q}")
                    if a:
                        lines.append(f"Petalbot: {a}")
            if lines:
                hist_block = "[이전 대화]\n" + "\n".join(lines) + "\n\n"
    except Exception:
        pass
    ctx_block = ""
    if request.context:
        ctx_block = f"[컨텍스트]\n{str(request.context)[:4000]}\n\n"

    # [2026-05-23] 첨부 처리
    import base64 as _b64_a
    text_attach_block = ""
    image_bytes_first = None
    image_names = []
    text_attach_count = 0
    if request.attachments and isinstance(request.attachments, list):
        for att in request.attachments[:5]:  # 최대 5개
            if not isinstance(att, dict):
                continue
            name = str(att.get("name", "") or "")[:80]
            kind = str(att.get("kind", "") or "").lower()
            b64data = str(att.get("b64", "") or "")
            if not b64data:
                continue
            try:
                raw = _b64_a.b64decode(b64data)
            except Exception:
                continue
            if kind == "image" and image_bytes_first is None:
                # 첫 이미지만 vision LLM 에 전달 (성능 + 비용 고려)
                image_bytes_first = raw
                image_names.append(name)
            elif kind == "text" or kind == "txt":
                # 텍스트 파일 → 컨텍스트에 합침 (앞 3000자)
                try:
                    content = raw.decode("utf-8", errors="replace")[:3000]
                    text_attach_block += (
                        f"\n[첨부 파일: {name}]\n{content}\n"
                    )
                    text_attach_count += 1
                except Exception:
                    pass
    if image_names:
        text_attach_block = (
            f"\n[이미지 첨부 {len(image_names)}개: {', '.join(image_names)}]\n"
            + text_attach_block
        )

    prompt = (
        f"{hist_block}{ctx_block}{text_attach_block}"
        f"[사용자 요청]\n{request.text}"
    )
    try:
        if image_bytes_first is not None:
            ans = (await get_llm_response_async(
                prompt, system_prompt=sys_prompt,
                image_input=image_bytes_first,
                max_tokens=2048, timeout=45,
            )).strip()
        else:
            ans = (await get_llm_response_async(
                prompt, system_prompt=sys_prompt,
                max_tokens=2048, timeout=30,
            )).strip()
    except Exception as e:
        return PetalChatResponse(
            status="error",
            answer="",
            mode_used=mode,
            error=f"LLM 호출 실패: {e}",
        )
    if not ans or ans.startswith("LLM 오류"):
        return PetalChatResponse(
            status="error",
            answer="",
            mode_used=mode,
            error=ans or "빈 응답",
        )
    # [2026-05-23 B-2] NCP TTS — chat 응답도 음성 (request.tts_enabled 기본 True)
    # [2026-05-23 b-3] spoken 평문 동봉 — NCP 실패 시 클라이언트 pyttsx3 폴백 가능.
    audio_b64 = None
    audio_format = None
    spoken_text = ""
    tts_source = "none"
    if request.tts_enabled is not False:
        try:
            # 음성으로 읽기 — 코드블록/마크다운 표는 빼고 핵심 텍스트만 (280자 cap)
            spoken_text = _strip_for_tts(ans)
            if spoken_text:
                audio_b64 = _synthesize_ncp_tts_b64(
                    spoken_text, voice="dara",   # speed=None → 공유 설정 반영
                )
                if audio_b64:
                    audio_format = "mp3"
                    tts_source = "ncp"
        except Exception as _e_tts:
            print(f"⚠️ [petal_chat] NCP TTS graceful: {_e_tts}")
    return PetalChatResponse(
        status="ok", answer=ans, mode_used=mode,
        audio_b64=audio_b64, audio_format=audio_format,
        spoken=spoken_text or None,
        tts_source=tts_source,
    )


def _strip_for_tts(text: str, cap: int = 280) -> str:
    """[2026-05-23] LLM 응답에서 TTS 로 읽기 좋은 평문 추출.
    - 코드블록 제거 (```...```)
    - 마크다운 헤더/리스트 마커 제거
    - 첫 ~280자 cap"""
    import re as _re_s
    if not text:
        return ""
    t = _re_s.sub(r"```[\s\S]*?```", " (코드 생략) ", text)
    t = _re_s.sub(r"`([^`]+)`", r"\1", t)
    t = _re_s.sub(r"^#{1,6}\s+", "", t, flags=_re_s.MULTILINE)
    t = _re_s.sub(r"^\s*[-*]\s+", "", t, flags=_re_s.MULTILINE)
    t = _re_s.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = _re_s.sub(r"\s+", " ", t).strip()
    if len(t) > cap:
        t = t[:cap].rstrip() + "..."
    return t


_PETAL_CB_SYS = {
    "summarize": "다음 텍스트를 한국어로 3줄 이내로 요약하라. 마크다운 금지, 일반 문장만.",
    "polish": "다음 텍스트의 문법·맞춤법·자연스러움을 다듬어라. 의미는 보존. 결과만 출력.",
    "translate_en": "다음 텍스트를 자연스러운 영어로 번역하라. 결과만 출력, 설명 금지.",
    "polite_tone": "다음 텍스트를 더 정중하고 공식적인 톤으로 다시 써라. 결과만 출력.",
}


@app.post("/api/petal/clipboard", response_model=PetalClipboardResponse)
async def petal_clipboard(
    request: PetalClipboardRequest,
    x_petal_token: Optional[str] = Header(None),
):
    """[Petalbot] 클립보드 도우미 — 요약/다듬기/번역/정중톤 변환."""
    _check_petal_auth(x_petal_token)
    text = (request.text or "").strip()
    if not text:
        return PetalClipboardResponse(
            status="error", text="", error="입력 텍스트가 비어있어요."
        )
    intent = (request.intent or "polish").lower()
    sys_prompt = _PETAL_CB_SYS.get(intent, _PETAL_CB_SYS["polish"])
    try:
        from llm_module import get_llm_response_async
        out = (await get_llm_response_async(
            text, system_prompt=sys_prompt, max_tokens=2048, timeout=30,
        )).strip()
    except Exception as e:
        return PetalClipboardResponse(
            status="error", text="", error=f"LLM 실패: {e}"
        )
    if not out or out.startswith("LLM 오류"):
        return PetalClipboardResponse(
            status="error", text="", error=out or "빈 응답"
        )
    note_map = {
        "summarize": "요약 완료",
        "polish": "다듬기 완료",
        "translate_en": "영어 번역 완료",
        "polite_tone": "정중한 톤으로 변환",
    }
    return PetalClipboardResponse(
        status="ok",
        text=out,
        note=note_map.get(intent, "처리 완료"),
    )


_PETAL_SCREEN_SYS = {
    "guide": (
        "너는 화면 도우미 AIRA. 사용자가 보여준 스크린샷을 분석해서 "
        "다음에 어떻게 해야 할지 한국어로 친절하게 안내한다.\n\n"
        "★ 출력 규칙 ★\n"
        "- 두 부분으로 나눈다:\n"
        "  1) analysis: 화면 상황 + 권장 행동 (한국어 마크다운, 5문장 이내)\n"
        "  2) spoken: 사용자에게 음성으로 들려줄 1문장 (200자 이내, 명확한 행동 가이드)\n"
        "- spoken 은 '~하세요/~이쪽으로 가세요/~버튼 누르세요' 처럼 행동 지시\n"
        "- 사용자가 막혀있다고 가정 — 다음 한 단계만 안내\n"
        "- 인사·메타설명 금지"
    ),
    "describe": (
        "너는 화면 분석 도우미 AIRA. 사용자가 보여준 스크린샷이 무엇을 "
        "보여주는지 한국어로 설명한다.\n\n"
        "★ 출력 ★\n"
        "  1) analysis: 화면 내용 요약 (마크다운, 5~7문장)\n"
        "  2) spoken: 한 문장 음성 요약 (200자 이내)"
    ),
    "fix": (
        "너는 문제 해결 AIRA. 사용자가 보여준 스크린샷에서 에러/문제를 "
        "찾아내고 해결 방법 안내.\n\n"
        "★ 출력 ★\n"
        "  1) analysis: 문제 진단 + 해결 단계 (마크다운, 단계별)\n"
        "  2) spoken: 가장 시급한 1 행동 (200자 이내)"
    ),
}


@app.post("/api/petal/screen_analyze", response_model=PetalScreenAnalyzeResponse)
async def petal_screen_analyze(
    request: PetalScreenAnalyzeRequest,
    x_petal_token: Optional[str] = Header(None),
):
    """[Petalbot] 화면 스크린샷 분석 — vision LLM 분석 + AIRA 음성 안내 텍스트.

    flow:
      1. base64 PNG 디코드
      2. vision LLM (gemini-2.5-flash multimodal) 호출 — image_input 지원
      3. analysis (자세한 마크다운) + spoken (200자 음성 텍스트) 반환
    """
    _check_petal_auth(x_petal_token)
    import base64
    import json as _json
    import re as _re_s

    if not request.image_b64:
        return PetalScreenAnalyzeResponse(
            status="error", analysis="", spoken="", mode_used=request.mode or "guide",
            error="image_b64 비어있음",
        )
    try:
        image_bytes = base64.b64decode(request.image_b64)
    except Exception as e:
        return PetalScreenAnalyzeResponse(
            status="error", analysis="", spoken="", mode_used=request.mode or "guide",
            error=f"base64 디코드 실패: {e}",
        )
    if len(image_bytes) < 100:
        return PetalScreenAnalyzeResponse(
            status="error", analysis="", spoken="", mode_used=request.mode or "guide",
            error="이미지가 너무 작아요",
        )
    try:
        from llm_module import get_llm_response_async
    except Exception as e:
        return PetalScreenAnalyzeResponse(
            status="error", analysis="", spoken="", mode_used=request.mode or "guide",
            error=f"llm_module import 실패: {e}",
        )
    mode = (request.mode or "guide").lower()
    sys_prompt = _PETAL_SCREEN_SYS.get(mode, _PETAL_SCREEN_SYS["guide"])
    # history 직렬화
    hist_block = ""
    try:
        if request.history:
            recent = list(request.history)[-4:]
            lines = []
            for t in recent:
                if isinstance(t, dict):
                    q = str(t.get("q", ""))[:150]
                    a = str(t.get("a", ""))[:150]
                    if q:
                        lines.append(f"사용자: {q}")
                    if a:
                        lines.append(f"AIRA: {a}")
            if lines:
                hist_block = "[이전 대화]\n" + "\n".join(lines) + "\n\n"
    except Exception:
        pass
    ctx_block = ""
    if request.context:
        ctx_block = f"[사용자 컨텍스트]\n{str(request.context)[:1000]}\n\n"
    prompt = (
        f"{hist_block}{ctx_block}"
        "[스크린샷 분석 요청]\n"
        "위 이미지를 보고 정확히 다음 JSON 으로만 응답:\n"
        '{"analysis": "...", "spoken": "..."}'
    )
    try:
        raw = (await get_llm_response_async(
            prompt, system_prompt=sys_prompt, image_input=image_bytes,
            response_mime_type="application/json",
            max_tokens=2048, timeout=45,
        )).strip()
    except Exception as e:
        return PetalScreenAnalyzeResponse(
            status="error", analysis="", spoken="", mode_used=mode,
            error=f"vision LLM 실패: {e}",
        )
    if not raw or raw.startswith("LLM 오류"):
        return PetalScreenAnalyzeResponse(
            status="error", analysis="", spoken="", mode_used=mode,
            error=raw or "빈 응답",
        )
    # JSON 파싱
    analysis = ""
    spoken = ""
    try:
        if raw.startswith("```"):
            m = _re_s.search(r"\{[\s\S]+\}", raw)
            if m:
                raw = m.group(0)
        data = _json.loads(raw)
        analysis = str(data.get("analysis", "") or "")
        spoken = str(data.get("spoken", "") or "")
    except Exception:
        # JSON 파싱 실패 — raw 전체를 analysis 로
        analysis = raw
        spoken = raw[:200].split("\n")[0]
    # spoken 200자 cap + 정리
    spoken = spoken.strip()
    if len(spoken) > 200:
        spoken = spoken[:200].rstrip() + "..."
    if not spoken and analysis:
        spoken = analysis[:200].split("\n")[0]
    # [2026-05-22 B-2] NCP TTS — AIRA voice (dara). 키 없거나 실패 시 None (Petalbot pyttsx3 폴백)
    audio_b64 = None
    audio_format = None
    if spoken:
        try:
            audio_b64 = _synthesize_ncp_tts_b64(spoken, voice="dara")  # speed→공유설정
            if audio_b64:
                audio_format = "mp3"
        except Exception as _e_tts:
            print(f"⚠️ [screen_analyze] NCP TTS graceful: {_e_tts}")
    return PetalScreenAnalyzeResponse(
        status="ok", analysis=analysis, spoken=spoken, mode_used=mode,
        audio_b64=audio_b64, audio_format=audio_format,
    )


# ──────────────────────────────────────────────────────────────────
# [2026-05-30 7W] 육하원칙 빌더 — 상담 정리(seven_w) + 범위 외 요청 감지(scope_check)
# ──────────────────────────────────────────────────────────────────
_PETAL_SEVEN_W_SYS = """당신은 프리랜서 외주(크몽 등) 상담을 돕는 어시스턴트입니다.
고객이 두서없이 보낸 상담 대화를 읽고, 작업 범위를 잡기 위한 7하원칙(7W)으로 정리합니다.

핵심 원칙:
1) 대화에 "명시된 사실"과 "추측"을 절대 섞지 마세요. 고객이 말하지 않은 정보를 그럴듯하게 지어내지 마세요. 이 도구는 유료 견적의 범위를 잡는 데 쓰이므로, 없는 정보를 만들어내면 치명적입니다.
2) 각 항목에서 대화로 확인되지 않은 중요한 빈칸은 해당 항목의 "unknown" 배열에 "고객에게 되물을 질문" 형태로 넣으세요. (예: "예산 범위가 어느 정도인지?")
3) 출력은 아래 JSON 구조로만, 한국어로 작성합니다. JSON 외 텍스트·마크다운·코드펜스 금지.

각 필드의 의미:
- who.actors: 소통/의사결정 주체(고객 본인, 담당자, 관계자 등)
- who.audience: 결과물을 실제로 사용할 타깃 사용자/고객층
- what.items: 만들어야 할 구체적 결과물(산출물)
- when.timeline: 작업 기간, 마감일, 마일스톤
- where.channels: 결과물이 배포·사용될 플랫폼/채널(웹, 모바일, 인스타, 인쇄물 등). 출장 등 물리적 위치가 언급되면 함께.
- why.intent: 고객이 애초에 이 작업을 하려는 진짜 의도/배경
- how.requirements: 고객이 제시한 제약·선호·요구사항·레퍼런스(작업 절차가 아니라 고객의 조건)
- howMuch.budget: 예산/금액 관련 언급
- howMuch.scale: 규모(페이지 수, 분량, 수정 횟수, 수량 등)
- howMuch.kpi: 기대 성과/지표

각 배열에는 명시된 내용만 짧은 항목으로 담고, 해당 정보가 대화에 전혀 없으면 빈 배열로 두고 unknown에 질문을 넣으세요.
followUpQuestions: 위 unknown들을 종합해 고객에게 실제로 물어볼 우선순위 질문 목록.
replyDraft: "이렇게 이해했는데 맞으실까요?" 식으로, 정리한 내용 확인 + 미정 항목 질문을 담은 정중한 답장 초안.
oneLineSummary: 이 의뢰를 한 문장으로.

반드시 다음 JSON 구조 그대로 출력 (모든 키 포함, 배열은 문자열 배열):
{
  "oneLineSummary": "...",
  "who":     {"actors": [], "audience": [], "unknown": []},
  "what":    {"items": [], "unknown": []},
  "when":    {"timeline": [], "unknown": []},
  "where":   {"channels": [], "unknown": []},
  "why":     {"intent": [], "unknown": []},
  "how":     {"requirements": [], "unknown": []},
  "howMuch": {"budget": [], "scale": [], "kpi": [], "unknown": []},
  "followUpQuestions": [],
  "replyDraft": "..."
}"""

_PETAL_SCOPE_SYS = """당신은 프리랜서 외주(크몽 등) 프로젝트의 PM입니다.
이미 고객과 합의된 '기준 작업 범위'(7하원칙으로 정리됨)가 주어지고, 고객이 새로 보낸 추가 요청 메시지가 주어집니다.
당신의 임무는 새 요청을 기준 범위와 비교해, 각 요청 항목이 어디에 해당하는지 판정하는 것입니다.

분류(classification):
- in_scope     : 기준 범위에 이미 포함된 작업 (추가 비용 없음)
- out_of_scope : 기준 범위를 벗어난 추가 작업 (추가 견적 대상)
- ambiguous    : 범위 내인지 외인지 대화만으로는 모호 (고객 확인 필요)

규칙:
1) 새 요청 메시지를 의미 단위(개별 기능/요청)로 나눠 각각 판정하세요.
2) 판정 근거(reason)에는 기준 범위의 어떤 부분과 비교했는지 구체적으로 쓰세요. 추측으로 범위를 넓히지 마세요.
3) out_of_scope 항목에는 한국 프리랜서 시장(크몽 등) 기준의 합리적인 추가 견적 범위(addQuoteMin~addQuoteMax, 단위: 원)와 예상 작업량(effort), 산정 근거(quoteBasis)를 제시하세요.
   - 제공된 '단가/예산 기준'이 있으면 그것을 우선 적용하고, 없으면 일반 시세로 추정하되 quoteBasis에 '추정치'임을 명시하세요.
4) in_scope / ambiguous 항목은 addQuoteMin, addQuoteMax 를 0으로 두세요.
5) totalAddMin / totalAddMax 는 out_of_scope 항목들의 합계입니다.
6) clientNote: 범위 외 항목과 추가 견적을 고객에게 정중히 안내하는 메시지 초안. 합의된 범위는 그대로 진행되고, 추가 요청은 별도 견적이 필요함을 설명.
7) overallVerdict: 한 문장 총평.
출력은 아래 JSON 구조로만, 한국어로 작성합니다. JSON 외 텍스트·마크다운·코드펜스 금지.

반드시 다음 JSON 구조 그대로 출력:
{
  "overallVerdict": "...",
  "items": [
    {"request": "...", "classification": "in_scope|out_of_scope|ambiguous",
     "reason": "...", "effort": "...", "addQuoteMin": 0, "addQuoteMax": 0, "quoteBasis": "..."}
  ],
  "totalAddMin": 0,
  "totalAddMax": 0,
  "clientNote": "..."
}"""

# 7W 필드 정의 (scope dict → 텍스트 변환용). app.py FIELDS 와 동일.
_SEVEN_W_FIELDS = [
    ("who", "Who·행위자", ["actors", "audience"]),
    ("what", "What·작업 대상", ["items"]),
    ("when", "When·기간", ["timeline"]),
    ("where", "Where·채널", ["channels"]),
    ("why", "Why·의도", ["intent"]),
    ("how", "How·요구 조건", ["requirements"]),
    ("howMuch", "How much·지표", ["budget", "scale", "kpi"]),
]


def _scope_to_text(d: dict) -> str:
    """고정된 7W 결과 dict → 비교용 평문. app.py scope_to_text 이식."""
    lines = ["[한 줄 요약] " + str((d or {}).get("oneLineSummary") or "")]
    for key, ko, props in _SEVEN_W_FIELDS:
        v = (d or {}).get(key, {}) or {}
        vals = []
        for prop in props:
            vals.extend(v.get(prop) or [])
        if vals:
            lines.append("- %s: %s" % (ko, "; ".join(str(x) for x in vals)))
    return "\n".join(lines)


def _parse_petal_json(raw: str) -> Optional[dict]:
    """LLM JSON 응답 방어적 파싱 (코드펜스·앞뒤 잡음 제거). screen_analyze 패턴."""
    import json as _json
    import re as _re
    if not raw:
        return None
    s = raw.strip()
    if not s.startswith("{"):
        m = _re.search(r"\{[\s\S]+\}", s)
        if m:
            s = m.group(0)
    try:
        out = _json.loads(s)
        return out if isinstance(out, dict) else None
    except Exception:
        return None


@app.post("/api/petal/seven_w", response_model=PetalSevenWResponse)
async def petal_seven_w(
    request: PetalSevenWRequest,
    x_petal_token: Optional[str] = Header(None),
):
    """[Petalbot 7W] 고객 상담 대화 → 7하원칙 구조화 정리 (환각 방지·되물을 질문 포함)."""
    _check_petal_auth(x_petal_token)
    convo = (request.conversation or "").strip()
    if not convo:
        return PetalSevenWResponse(status="error", error="정리할 대화 내용이 비어있어요.")
    try:
        from llm_module import get_llm_response_async
    except Exception as e:
        return PetalSevenWResponse(status="error", error=f"llm_module import 실패: {e}")
    biz = (request.biz or "").strip()
    prompt = (
        (("업종/맥락: %s\n\n" % biz) if biz else "")
        + "다음 고객 상담 대화를 7하원칙으로 정리해 지정된 JSON 으로만 출력하세요.\n\n"
        + "--- 대화 시작 ---\n%s\n--- 대화 끝 ---" % convo
    )
    try:
        raw = (await get_llm_response_async(
            prompt, system_prompt=_PETAL_SEVEN_W_SYS,
            response_mime_type="application/json",
            max_tokens=8192, timeout=90,
        )).strip()
    except Exception as e:
        return PetalSevenWResponse(status="error", error=f"LLM 호출 실패: {e}")
    if not raw or raw.startswith("LLM 오류"):
        return PetalSevenWResponse(status="error", error=raw or "빈 응답")
    data = _parse_petal_json(raw)
    if data is None:
        return PetalSevenWResponse(status="error", error="JSON 파싱 실패. 다시 시도해 주세요.")
    return PetalSevenWResponse(status="ok", data=data)


@app.post("/api/petal/scope_check", response_model=PetalScopeCheckResponse)
async def petal_scope_check(
    request: PetalScopeCheckRequest,
    x_petal_token: Optional[str] = Header(None),
):
    """[Petalbot 7W] 고정된 기준 범위 vs 새 요청 → 항목별 범위 판정 + 추가견적 산정."""
    _check_petal_auth(x_petal_token)
    new_req = (request.new_request or "").strip()
    # [2026-06-02] 기준범위(scope)가 없어도 동작 — 클립보드 자동 감지/빠른 견적 경로 지원.
    #   기준범위 없으면 LLM 이 새 요청을 '신규 작업'으로 보고 일반 시세로 추가견적 추정.
    has_baseline = bool(request.scope)
    if not new_req:
        return PetalScopeCheckResponse(status="error", error="점검할 새 요청이 비어있어요.")
    try:
        from llm_module import get_llm_response_async
    except Exception as e:
        return PetalScopeCheckResponse(status="error", error=f"llm_module import 실패: {e}")
    rate = (request.rate or "").strip()
    if has_baseline:
        scope_block = "[기준 작업 범위 — 합의됨]\n" + _scope_to_text(request.scope) + "\n\n"
    else:
        scope_block = ("[기준 작업 범위 — 미고정]\n"
                       "합의된 기준 범위가 아직 없습니다. 새 요청을 '신규 추가 작업'으로 보고, "
                       "각 항목을 out_of_scope 로 분류한 뒤 일반 시세로 추가견적을 추정하세요"
                       "(quoteBasis 에 '기준범위 미고정·시세 추정'임을 명시).\n\n")
    prompt = (
        scope_block
        + "[단가/예산 기준] " + (rate if rate else "미지정 (시세로 추정)") + "\n\n"
        + "다음은 고객이 새로 보낸 추가 요청입니다. "
        + ("기준 범위와 비교해 각 항목을 판정하고, 범위 외 항목은 추가 견적을 산정하세요. "
           if has_baseline else "각 추가 작업 항목의 추가 견적을 산정하세요. ")
        + "지정된 JSON 으로만 출력하세요.\n"
        + "--- 새 요청 시작 ---\n" + new_req + "\n--- 새 요청 끝 ---"
    )
    try:
        raw = (await get_llm_response_async(
            prompt, system_prompt=_PETAL_SCOPE_SYS,
            response_mime_type="application/json",
            max_tokens=8192, timeout=90,
        )).strip()
    except Exception as e:
        return PetalScopeCheckResponse(status="error", error=f"LLM 호출 실패: {e}")
    if not raw or raw.startswith("LLM 오류"):
        return PetalScopeCheckResponse(status="error", error=raw or "빈 응답")
    data = _parse_petal_json(raw)
    if data is None:
        return PetalScopeCheckResponse(status="error", error="JSON 파싱 실패. 다시 시도해 주세요.")
    return PetalScopeCheckResponse(status="ok", data=data)


# [2026-06-02] 자율개발 역질문 — 모호한 요청에 바로 실행 말고 what/where/how 되묻기.
_PETAL_CLARIFY_DEV_SYS = """너는 외주 개발 요청을 받으면 **바로 작업하지 않고 먼저 되물어 확인하는** 신중한 개발 PM이다.
사용자의 개발/수정 요청을 읽고, 실행 전에 확인할 질문을 만든다. 절대 마음대로 단정하지 마라.

규칙:
- understanding: 요청을 한 문장으로 이해한 내용(추측 최소화).
- approach: 네가 제안하는 **구체적 구현 방식** 1~3문장(how). "이러이러하게 구현하려 합니다" 형태.
- whatClear: '무엇을' 만들거나 고칠지가 메시지에 **구체적으로 명시**돼 있으면 true.
  모호하면(예: "촌스러워 고쳐줘", "이상해", "더 좋게 해줘", "별로야") false.
- questions: 사용자에게 되물을 질문 배열(한국어). 다음을 포함:
  · whatClear=false 이면 **맨 앞에** (what) "구체적으로 어떤 부분을 / 무엇을 바꿀까요?" 류 질문.
  · 항상 (where) "어떤 프로젝트에 적용할까요?"
  · 항상 (how) "제안한 방식대로 진행해도 될까요? 수정할 점이 있으면 알려주세요."
  · 그 외 작업에 꼭 필요한 핵심 미정 사항이 있으면 1~2개 추가(예산/마감 말고 기술적 미정만).
출력은 아래 JSON 으로만. JSON 외 텍스트·마크다운·코드펜스 금지.
{
  "understanding": "...",
  "approach": "...",
  "whatClear": true,
  "questions": ["...", "..."]
}"""


@app.post("/api/petal/clarify_dev", response_model=PetalClarifyDevResponse)
async def petal_clarify_dev(
    request: PetalClarifyDevRequest,
    x_petal_token: Optional[str] = Header(None),
):
    """[Petalbot] 모호한 개발 요청 → what/where/how 역질문 생성(바로 실행 방지)."""
    _check_petal_auth(x_petal_token)
    req = (request.request or "").strip()
    if not req:
        return PetalClarifyDevResponse(status="error", error="요청이 비어있어요.")
    try:
        from llm_module import get_llm_response_async
    except Exception as e:
        return PetalClarifyDevResponse(status="error", error=f"llm_module import 실패: {e}")
    prompt = (("[맥락] " + request.context + "\n\n") if request.context else "") + (
        "다음 개발/수정 요청을 실행하기 전에 되물을 질문을 만들어라.\n"
        "--- 요청 시작 ---\n" + req + "\n--- 요청 끝 ---")
    try:
        raw = (await get_llm_response_async(
            prompt, system_prompt=_PETAL_CLARIFY_DEV_SYS,
            response_mime_type="application/json", max_tokens=2048, timeout=60,
        )).strip()
    except Exception as e:
        return PetalClarifyDevResponse(status="error", error=f"LLM 호출 실패: {e}")
    if not raw or raw.startswith("LLM 오류"):
        return PetalClarifyDevResponse(status="error", error=raw or "빈 응답")
    data = _parse_petal_json(raw)
    if data is None:
        return PetalClarifyDevResponse(status="error", error="JSON 파싱 실패. 다시 시도해 주세요.")
    return PetalClarifyDevResponse(status="ok", data=data)


@app.get("/api/petal/health")
async def petal_health():
    """[Petalbot] 연결 확인용. 인증 없음."""
    return {
        "status": "ok",
        "service": "EIDOS Petal API",
        "auth_required": bool(PETAL_API_KEY),
        "modes": list(_PETAL_MODE_SYS.keys()),
        "clipboard_intents": list(_PETAL_CB_SYS.keys()),
        "screen_modes": list(_PETAL_SCREEN_SYS.keys()),  # [2026-05-22]
        "seven_w": True,  # [2026-05-30] 7W 정리 + 범위 점검 엔드포인트
    }


if __name__ == "__main__":
    import uvicorn

    # 윈도우 경로 문제 방지를 위해 절대 경로로 변환
    exclude_pattern = os.path.join(PROJECT_ROOT, "*")
    
    print(f"🚀 [Server] Starting with reload exclude: {exclude_pattern}")
    
    uvicorn.run(
        "eidos_server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        # 여기에 제외할 패턴을 리스트로 입력 (경로 꼬임 방지)
        reload_excludes=["eidos_files/*", "eidos_files/**/*", "./eidos_files/*"]
    )