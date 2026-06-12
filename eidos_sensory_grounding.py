# eidos_sensory_grounding.py
# [Wave8 2026-05-28] 시스템 신호 → belief 통합.
#
# 기존 EIDOS 는 사용자가 뭐 하는지 채팅 키워드 + 시간으로만 추정. 한참 코딩하다가
# 잠깐 자리를 비워도 EIDOS 는 모름. 이 모듈은 그 gap 을 매움:
#
#   - 활성 윈도우 제목 (지금 뭐 보고 있나)
#   - 마우스/키보드 idle 시간 (자리에 있나)
#   - CPU 부하 (빌드·렌더 등 무거운 작업 중인가)
#   - 시간대·요일
#
# 이걸로 work_state (working/break/casual/done/unknown) 자동 추정.
# 추정값은 belief 의 work_state 보강에만 사용 — 사용자 명시 override 가 항상 우선.
#
# 프라이버시:
#   - 윈도우 제목만 (내용·키 입력 자체 X)
#   - 마우스/키보드는 마지막 활동 시각만 (Windows GetLastInputInfo)
#   - 모든 신호 settings 토글로 개별 OFF 가능
#
# 의존성 (이미 설치):
#   - psutil (CPU)
#   - pygetwindow (활성 윈도우·Windows/Linux/Mac 호환)
#   - ctypes (Windows GetLastInputInfo·idle 시간)
#
# 비용: LLM 호출 0·매 60s capture 1회·~5ms·시스템 영향 거의 0.

from __future__ import annotations

import datetime as _dt
import sys
from dataclasses import dataclass, asdict
from typing import Optional


# ── dataclass ────────────────────────────────────────────────────────
@dataclass
class SystemSignals:
    """한 시점의 시스템 상태 snapshot."""
    timestamp: str = ""

    # 활성 윈도우
    active_window_title: str = ""
    active_app_hint: str = ""           # IDE/document/media/chat/browser/unknown

    # 사용자 활동
    is_idle: bool = False
    idle_seconds: float = 0.0           # 마지막 키/마우스 입력 후 경과

    # 시스템 부하
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    high_cpu_load: bool = False         # CPU > 60% (빌드·렌더 추정)

    # 시간 컨텍스트
    hour_of_day: int = 12
    day_of_week: int = 0                # 0=월 ~ 6=일
    weekday_name: str = ""              # "월요일"
    is_weekend: bool = False
    is_late_night: bool = False         # 23~5시

    # 추정 work_state (사용자 명시 override X 시 보강용)
    inferred_work_state: str = "unknown"  # working/break/casual/done/unknown
    inference_confidence: float = 0.5

    def serialize(self) -> dict:
        return asdict(self)


# ── 활성 윈도우 분류 키워드 ───────────────────────────────────────────
# 윈도우 제목의 소문자 substring 매칭. 첫 매칭 카테고리 우선.
_WORK_KEYWORDS_IDE = (
    "vscode", "visual studio code", "intellij", "pycharm", "webstorm",
    "android studio", "xcode", "sublime", "atom", "vim", "neovim",
    "code.exe", "devenv", "rider", "goland", "clion", "datagrip",
    "terminal", "cmd", "powershell", "wsl", "git bash", "iterm",
    ".py - ", ".js - ", ".ts - ", ".rs - ", ".go - ", ".java - ",
)
_WORK_KEYWORDS_DOC = (
    "word", "hwp", "한글", "google docs", "notion", "obsidian",
    "typora", "joplin", "logseq", "evernote", "onenote",
    "excel", "spreadsheet", "google sheets", "numbers",
    "powerpoint", "keynote", "google slides", "miro", "lucid",
    ".docx", ".xlsx", ".pptx", ".md - ",
)
_WORK_KEYWORDS_DESIGN = (
    "figma", "sketch", "photoshop", "illustrator", "indesign",
    "after effects", "premiere", "davinci", "blender",
    "krita", "affinity",
)
# 'work' browser substrings (제목에 들어가면 work)
_WORK_BROWSER_HINTS = (
    "github", "gitlab", "bitbucket", "jira", "asana", "trello",
    "linear", "notion.so", "stackoverflow", "stack overflow",
    "arxiv", "pubmed", "scholar", "documentation", "docs.",
    "kmong", "soomgo", "fiverr", "upwork", "naver works",
    "console.aws", "azure portal", "cloud.google",
)

_BREAK_KEYWORDS_MEDIA = (
    "youtube", "netflix", "twitch", "disney+", "spotify",
    "vlc", "potplayer", "kmplayer", "wavve", "tving", "watcha",
    "instagram", "tiktok", "reels", "shorts",
)
_BREAK_KEYWORDS_GAME = (
    "steam", "discord game", "league of legends", "valorant",
    "overwatch", "minecraft", "stardew", "civilization", "factorio",
)

_CASUAL_KEYWORDS_CHAT = (
    "discord", "slack", "kakaotalk", "카카오톡", "telegram",
    "line", "whatsapp", "messenger", "zoom", "google meet",
    "teams", "skype",
)


def classify_window(title: str) -> tuple[str, str]:
    """윈도우 제목 → (app_hint, work_state).

    app_hint: ide·document·design·browser_work·media·game·chat·browser·unknown
    work_state: working·break·casual·unknown

    매칭 우선순위: IDE > 문서 > 디자인 > work browser > 미디어 > 게임 > 채팅.
    """
    if not title:
        return "unknown", "unknown"
    t = title.lower()

    # IDE / 터미널
    for kw in _WORK_KEYWORDS_IDE:
        if kw in t:
            return "ide", "working"

    # 문서/스프레드시트
    for kw in _WORK_KEYWORDS_DOC:
        if kw in t:
            return "document", "working"

    # 디자인 툴
    for kw in _WORK_KEYWORDS_DESIGN:
        if kw in t:
            return "design", "working"

    # 채팅·미디어 — 단어 매칭 먼저 (브라우저 안에서도 가능)
    for kw in _CASUAL_KEYWORDS_CHAT:
        if kw in t:
            return "chat", "casual"
    for kw in _BREAK_KEYWORDS_MEDIA:
        if kw in t:
            return "media", "break"
    for kw in _BREAK_KEYWORDS_GAME:
        if kw in t:
            return "game", "break"

    # 브라우저 — 제목으로 work / break 구분
    if "chrome" in t or "edge" in t or "firefox" in t or "safari" in t or "whale" in t:
        for kw in _WORK_BROWSER_HINTS:
            if kw in t:
                return "browser_work", "working"
        return "browser", "unknown"

    return "unknown", "unknown"


# ── Windows idle time (GetLastInputInfo) ─────────────────────────────
def _get_idle_seconds_windows() -> float:
    """Windows API 로 마우스/키보드 마지막 입력 후 경과 시간 (초).

    실패 시 0.0 반환 (idle 미확인 = active 로 가정).
    """
    try:
        import ctypes
        from ctypes import wintypes

        class _LASTINPUTINFO(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.UINT),
                ("dwTime", wintypes.DWORD),
            ]
        lii = _LASTINPUTINFO()
        lii.cbSize = ctypes.sizeof(_LASTINPUTINFO)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
            return 0.0
        now_tick = ctypes.windll.kernel32.GetTickCount()
        elapsed_ms = now_tick - lii.dwTime
        if elapsed_ms < 0:
            return 0.0
        return elapsed_ms / 1000.0
    except Exception:
        return 0.0


def _get_idle_seconds() -> float:
    """OS 별 idle 시간 — 현재 Windows 만 정확. 외 OS 는 0.0."""
    if sys.platform.startswith("win"):
        return _get_idle_seconds_windows()
    return 0.0


# ── 활성 윈도우 제목 ─────────────────────────────────────────────────
def _get_active_window_title() -> str:
    """pygetwindow 로 활성 윈도우 제목 (cross-platform)·실패 시 빈 문자열."""
    try:
        import pygetwindow as gw
        win = gw.getActiveWindow()
        if win is None:
            return ""
        title = getattr(win, "title", "")
        return str(title or "")
    except Exception:
        # Windows 만 별도 fallback — win32gui
        if sys.platform.startswith("win"):
            try:
                import ctypes
                hwnd = ctypes.windll.user32.GetForegroundWindow()
                length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
                if length <= 0:
                    return ""
                buf = ctypes.create_unicode_buffer(length + 1)
                ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
                return str(buf.value or "")
            except Exception:
                pass
        return ""


# ── CPU·Memory ───────────────────────────────────────────────────────
def _get_cpu_memory() -> tuple[float, float]:
    """현재 시스템 부하 (CPU %, Memory %).

    psutil 부재·실패 시 (0.0, 0.0).
    psutil.cpu_percent(interval=None) 은 직전 호출 이후 평균. 첫 호출은
    0.0 또는 부정확할 수 있으니 0.1 초 interval 짧게 줌.
    """
    try:
        import psutil
        cpu = float(psutil.cpu_percent(interval=0.1))
        mem = float(psutil.virtual_memory().percent)
        return cpu, mem
    except Exception:
        return 0.0, 0.0


# ── 시간 컨텍스트 ─────────────────────────────────────────────────────
_WEEKDAY_NAMES = ("월요일", "화요일", "수요일", "목요일",
                  "금요일", "토요일", "일요일")


def _time_context(now: Optional[_dt.datetime] = None) -> dict:
    nv = now or _dt.datetime.now()
    h = nv.hour
    wd = nv.weekday()
    return {
        "hour_of_day": h,
        "day_of_week": wd,
        "weekday_name": _WEEKDAY_NAMES[wd] if 0 <= wd < 7 else "",
        "is_weekend": wd >= 5,
        "is_late_night": (h >= 23 or h <= 5),
    }


# ── 핵심: capture + inference ────────────────────────────────────────
_IDLE_THRESHOLD_SEC = 300.0     # 5분 idle 이면 work_state=unknown (away)
_HIGH_CPU_THRESHOLD = 60.0      # 60% 이상 = 빌드/렌더 추정


def capture_signals(settings: Optional[dict] = None) -> SystemSignals:
    """현재 시점의 system signals 캡처.

    settings 옵션 (None 이면 default 다 ON):
      - sensory_track_window: bool
      - sensory_track_idle: bool
      - sensory_track_cpu: bool
    각 신호 개별 OFF 가능 (프라이버시).
    """
    settings = settings or {}
    sig = SystemSignals(timestamp=_dt.datetime.now().isoformat(timespec="seconds"))

    # 시간 컨텍스트 — 무조건 채움
    tc = _time_context()
    sig.hour_of_day = tc["hour_of_day"]
    sig.day_of_week = tc["day_of_week"]
    sig.weekday_name = tc["weekday_name"]
    sig.is_weekend = tc["is_weekend"]
    sig.is_late_night = tc["is_late_night"]

    # 활성 윈도우
    if settings.get("sensory_track_window", True):
        try:
            title = _get_active_window_title()
            sig.active_window_title = title[:200] if title else ""
            sig.active_app_hint, win_work = classify_window(title)
            if win_work != "unknown":
                sig.inferred_work_state = win_work
                sig.inference_confidence = 0.7
        except Exception as e:
            print(f"[sensory] window capture 실패 (graceful): {e}")

    # Idle 시간
    if settings.get("sensory_track_idle", True):
        try:
            idle_s = _get_idle_seconds()
            sig.idle_seconds = idle_s
            sig.is_idle = idle_s >= _IDLE_THRESHOLD_SEC
        except Exception as e:
            print(f"[sensory] idle capture 실패 (graceful): {e}")

    # CPU·메모리
    if settings.get("sensory_track_cpu", True):
        try:
            cpu, mem = _get_cpu_memory()
            sig.cpu_percent = cpu
            sig.memory_percent = mem
            sig.high_cpu_load = cpu >= _HIGH_CPU_THRESHOLD
        except Exception as e:
            print(f"[sensory] cpu capture 실패 (graceful): {e}")

    # 최종 work_state 추정 — 휴리스틱 순서
    sig.inferred_work_state, sig.inference_confidence = _infer_work_state(sig)

    return sig


def _infer_work_state(sig: SystemSignals) -> tuple[str, float]:
    """signals 만 보고 work_state 추정 + confidence.

    우선순위:
      1. is_idle (5분+) → unknown (away)·conf 0.95
      2. active_app_hint 가 명확한 카테고리 → 그 카테고리 work_state·conf 0.7~0.85
      3. high_cpu_load + 작업 시간대 → working 보강
      4. 새벽 (23~5시) + idle 아님 + 작업 윈도우 → working (집중 야근)
      5. 그 외 → unknown·conf 0.3
    """
    # 1. Idle priority
    if sig.is_idle:
        return "unknown", 0.95

    # 2. 윈도우 hint 기반
    hint = sig.active_app_hint or ""
    if hint in ("ide", "document", "design", "browser_work"):
        # high_cpu 추가 신호면 conf ↑
        conf = 0.85 if sig.high_cpu_load else 0.75
        return "working", conf
    if hint in ("media", "game"):
        return "break", 0.8
    if hint == "chat":
        return "casual", 0.7
    if hint == "browser":
        # 분류 안 된 브라우저 — neutral
        return "unknown", 0.4

    # 3. CPU 부하 신호 (윈도우 무관)
    if sig.high_cpu_load and sig.cpu_percent >= 80.0:
        return "working", 0.6

    # 4. 시간대 hint
    if sig.is_late_night and not sig.is_idle:
        # 새벽인데 active — working 또는 casual
        return "working", 0.4

    return "unknown", 0.3


# ── LLM prompt 용 brief ──────────────────────────────────────────────
def signal_brief_for_prompt(sig: SystemSignals) -> str:
    """signals 의 LLM prompt 인용용 간결 brief.

    너무 길면 LLM 부담·짧으면 정보 손실. 4~6줄.
    """
    if sig is None:
        return ""
    parts = []
    parts.append(f"[시간] {sig.hour_of_day:02d}시 ({sig.weekday_name})")
    if sig.active_window_title:
        title_short = sig.active_window_title[:80]
        parts.append(f"[활성 윈도우] {title_short}")
        if sig.active_app_hint and sig.active_app_hint != "unknown":
            parts.append(f"[앱 종류] {sig.active_app_hint}")
    if sig.is_idle:
        parts.append(f"[자리 비움] {int(sig.idle_seconds // 60)}분+ 입력 없음")
    elif sig.idle_seconds > 30:
        parts.append(f"[최근 활동] {int(sig.idle_seconds)}초 전 입력")
    if sig.high_cpu_load:
        parts.append(f"[CPU 부하 높음] {sig.cpu_percent:.0f}% — 빌드·렌더 추정")
    if sig.inferred_work_state != "unknown":
        parts.append(
            f"[추정 work_state] {sig.inferred_work_state} "
            f"(conf={sig.inference_confidence:.2f})"
        )
    return "\n".join(parts)


# ── belief 갱신 helper ──────────────────────────────────────────────
def update_belief_from_signals(
    belief, sig: SystemSignals,
    override_explicit: bool = False,
) -> bool:
    """signals 의 inferred_work_state 를 belief.work_state 에 반영.

    원칙:
      - override_explicit=False (default): 사용자가 명시적으로 설정한 work_state
        가 최근 (10분 이내) 면 inferred 무시. 사용자 의도 우선.
      - override_explicit=True: 무조건 inferred 적용 (테스트·강제).

    Returns: True 면 belief 변경됨.
    """
    if belief is None or sig is None:
        return False
    if sig.inferred_work_state == "unknown":
        return False
    if sig.inference_confidence < 0.5:
        return False

    try:
        # 사용자가 명시 변경한 지 10분 안 됐으면 inferred 무시
        if not override_explicit:
            try:
                ws_since = getattr(belief, "work_state_since", "") or ""
                if ws_since:
                    last = _dt.datetime.fromisoformat(ws_since)
                    elapsed_min = (
                        (_dt.datetime.now() - last).total_seconds() / 60.0
                    )
                    if elapsed_min < 10.0:
                        return False
            except Exception:
                pass

        if belief.work_state == sig.inferred_work_state:
            return False

        belief.work_state = sig.inferred_work_state
        belief.work_state_since = _dt.datetime.now().isoformat(timespec="seconds")
        return True
    except Exception as e:
        print(f"[sensory] belief 갱신 실패 (graceful): {e}")
        return False
