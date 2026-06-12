# eidos_action_dispatcher.py
# ──────────────────────────────────────────────────────────────────────────────
# ActionDispatcher — dB/dt = f(S, G) 자율 실행 루프
#
# 역할:
#   GoalScheduler가 schedule에 넣어둔 PENDING 작업을 꺼내,
#   DCReasoner로 ActionPlan을 수립하고,
#   execution_module / BrowserCallback을 통해 실제로 실행한다.
#
#   실행 중 매 단계마다 "직원식 중간보고"를 채팅창에 전송한다.
#
# 중간보고 형식:
#   [보고] 현재 작업: {task}
#   ├─ Step N/M: {step_name} → {상태}
#   ├─ 완료: {done}개 / 전체: {total}개
#   └─ 다음 단계: {next_step}
#
# 액션 타입:
#   NAVIGATE   — 브라우저 URL 이동
#   FILL       — 에디터에 텍스트 주입 (FILL_EDITOR JS)
#   CLICK      — JS querySelector + simulateHumanClick
#   WRITE      — LLM으로 텍스트 생성 후 결과를 변수에 저장
#   WAIT       — n초 대기
#   REPORT     — 사용자에게 중간보고 메시지 전송
#   DONE       — 태스크 완료 선언
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field

# [P-α S4 2026-05-01] Step latency 측정용 module-level deque.
# _act 호출자에서 perf_counter wrap 으로 push. StepLatencyPoller (eidos_kpi_pollers.py)
# 가 5분 윈도우로 읽어 p50/p95 산출 + drift alert.
# Tuple format: (timestamp, action_verb, latency_seconds)
_STEP_LATENCY_QUEUE: deque = deque(maxlen=200)


def get_step_latency_queue() -> deque:
    """StepLatencyPoller 가 호출 — 외부 모듈에서 deque 접근 진입점."""
    return _STEP_LATENCY_QUEUE
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from eidos_v4_0_core import EidosCore

# ── LLM 헬퍼 ──────────────────────────────────────────────────────────────────
try:
    from eidos_patrol_agent import get_llm_response_async
except ImportError:
    async def get_llm_response_async(prompt: str, **kw) -> str:  # type: ignore
        return ""

# ReAct 전용 LLM 호출 (JSON 원천 보장)
try:
    from llm_module import get_llm_response_async, _robust_json_parse  # noqa: F401
except ImportError:
    pass  # 직접 import는 각 메서드에서 처리

# WRITE 액션(파일·보고서 산출물)용 분석가 페르소나
try:
    from llm_module import EIDOS_ANALYST_SYSTEM_PROMPT
except ImportError:
    EIDOS_ANALYST_SYSTEM_PROMPT = None

# robust JSON 파서 재사용
try:
    from llm_module import robust_json_parse as _robust_json_parse
except ImportError:
    def _robust_json_parse(text: str) -> dict:
        import re as _re
        text = text.strip()
        text = _re.sub(r'^```(?:json)?\s*', '', text, flags=_re.IGNORECASE)
        text = _re.sub(r'\s*```$', '', text).strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        start = text.find('{')
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == '{': depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i+1])
                        except Exception:
                            break
        return {}


# ── 브라우저 콜백 (execution_module에서 가져옴) ────────────────────────────────
try:
    import execution_module as _em
    _browser_run = _em.browser_run_script
    _browser_navigate = _em.browser_smart_navigate
    _navigate_and_wait = _em.navigate_and_wait      # [Fix] loadFinished 기반 대기
except ImportError:
    async def _browser_run(script: str = "", **kw) -> str:  # type: ignore
        return "execution_module 없음"
    async def _browser_navigate(url: str = "", **kw) -> str:  # type: ignore
        return "execution_module 없음"
    async def _navigate_and_wait(url: str = "", **kw) -> str:  # type: ignore
        return "execution_module 없음"


# ─────────────────────────────────────────────────────────────────────────────
# A11Y Tree Indexer JS — DOM의 상호작용 요소를 인덱싱된 JSON으로 직렬화.
# raw HTML 300K자 전송을 우회하고 LLM이 "[N]" 번호로 엘리먼트 참조 가능.
# window.__EIDOS_IDX 에 엘리먼트 배열 캐시 → CLICK 시 인덱스로 역참조.
# ─────────────────────────────────────────────────────────────────────────────
_A11Y_INDEXER_JS = r"""
(function() {
  // [Fix 2026-05-03] 페이지 이동 후 indexer 실패/timeout 시 window.__EIDOS_IDX 가
  // 이전 페이지의 stale element 들을 보존 → 다음 CLICK/FILL [N] 이 detached DOM 을
  // 참조해 무반응. 인덱싱 시작 직전 강제 클리어하여 적어도 INDEX_NOT_FOUND 로
  // 명확 실패하도록 보장.
  try { window.__EIDOS_IDX = []; } catch(e) {}
  try {
    // [기본 셀렉터] — 기존 동작 보존
    var SEL_BASE = 'a[href],button,input:not([type="hidden"]):not([type="image"]),select,textarea,' +
                   '[role="button"],[role="link"],[role="tab"],[role="menuitem"],' +
                   '[role="checkbox"],[role="radio"],[role="switch"],[role="option"],' +
                   '[role="combobox"],[role="searchbox"],[role="textbox"],[onclick]';
    // [에디터/입력 루트] — contenteditable, 리치 에디터 root, iframe
    var SEL_EDITOR = '[contenteditable=""],[contenteditable="true"],' +
                     '.ProseMirror,.ql-editor,.tox-edit-area,.tox-tinymce,' +
                     '[data-slate-editor],[data-lexical-editor],' +
                     '.se-wysiwyg,.sun-editor-editable,' +
                     '.note-editable,.cke_editable,.fr-element,' +
                     '.ck-editor__editable,.mce-content-body,' +
                     'iframe';
    var AD_RE = /doubleclick|googlesyndication|pagead|adservice|\/ads?\.|track\.|bnr\.|adclick|linkprice|affiliate/i;
    var MAX = 80;

    function isVisible(el) {
      try {
        var r = el.getBoundingClientRect();
        if (r.width < 2 || r.height < 2) return false;
        var s = window.getComputedStyle(el);
        if (s.display === 'none' || s.visibility === 'hidden') return false;
        if (parseFloat(s.opacity) < 0.1) return false;
        if (r.bottom < -1000 || r.top > (window.innerHeight + 3000)) return false;
        return true;
      } catch(e) { return false; }
    }

    function accName(el) {
      var n = (el.getAttribute && el.getAttribute('aria-label') || '').trim();
      if (!n && el.getAttribute) {
        var lid = el.getAttribute('aria-labelledby');
        if (lid) {
          try {
            var ref = (el.ownerDocument || document).getElementById(lid);
            if (ref) n = (ref.innerText || ref.textContent || '').trim();
          } catch(e) {}
        }
      }
      if (!n) {
        var txt = (el.innerText || el.textContent || '').trim();
        n = txt.replace(/\s+/g, ' ');
      }
      if (!n && el.getAttribute) n = (el.getAttribute('placeholder') || '').trim();
      if (!n && el.getAttribute) n = (el.getAttribute('title') || '').trim();
      if (!n && el.tagName === 'INPUT') n = (el.value || '').trim();
      if (!n) {
        try {
          var img = el.querySelector && el.querySelector('img[alt]');
          if (img) n = (img.getAttribute('alt') || '').trim();
        } catch(e) {}
      }
      return n.substring(0, 80);
    }

    function describe(el) {
      var tag = el.tagName.toLowerCase();
      var role = (el.getAttribute && (el.getAttribute('role') || '') || '').toLowerCase();
      // 에디터 루트 감지 (우선순위 최상) — LLM 식별 편의
      try {
        if (tag === 'iframe') return 'iframe';
        if (el.isContentEditable === true) return 'editor';
        var cls = (el.className || '').toString().toLowerCase();
        if (cls.indexOf('prosemirror') >= 0 || cls.indexOf('ql-editor') >= 0 ||
            cls.indexOf('tox-edit') >= 0 || cls.indexOf('sun-editor') >= 0 ||
            cls.indexOf('se-wysiwyg') >= 0 || cls.indexOf('note-editable') >= 0 ||
            cls.indexOf('cke_editable') >= 0 || cls.indexOf('fr-element') >= 0 ||
            cls.indexOf('ck-editor__editable') >= 0 || cls.indexOf('mce-content-body') >= 0) {
          return 'editor';
        }
        if (el.getAttribute && (el.getAttribute('data-slate-editor') !== null ||
                                 el.getAttribute('data-lexical-editor') !== null)) {
          return 'editor';
        }
      } catch(e) {}
      if (tag === 'a') return 'link';
      if (tag === 'button') return 'button';
      if (tag === 'input') {
        var t = (el.getAttribute('type') || 'text').toLowerCase();
        if (t === 'submit' || t === 'button') return 'button';
        return 'input(' + t + ')';
      }
      if (tag === 'select') return 'select';
      if (tag === 'textarea') return 'textarea';
      if (role) return role;
      return tag;
    }

    // [iframe 내부 same-origin 수집] — cross-origin/blocked 은 try-catch로 무시
    function collectFromDoc(doc, outRoots) {
      try {
        if (!doc || !doc.querySelectorAll) return;
        var baseList = doc.querySelectorAll(SEL_BASE);
        var edList   = doc.querySelectorAll(SEL_EDITOR);
        for (var i = 0; i < baseList.length; i++) outRoots.push(baseList[i]);
        for (var j = 0; j < edList.length; j++) outRoots.push(edList[j]);
      } catch(e) {}
    }

    // 1) 최상위 문서 — 에디터류를 먼저 push (순서 = 우선순위)
    var roots = [];
    try {
      var edTop = document.querySelectorAll(SEL_EDITOR);
      for (var i0 = 0; i0 < edTop.length; i0++) roots.push(edTop[i0]);
    } catch(e) {}
    try {
      var baseTop = document.querySelectorAll(SEL_BASE);
      for (var i1 = 0; i1 < baseTop.length; i1++) roots.push(baseTop[i1]);
    } catch(e) {}

    // 2) same-origin iframe 내부도 수집 (실패 시 무시)
    try {
      var iframes = document.querySelectorAll('iframe');
      for (var k = 0; k < iframes.length && k < 8; k++) {
        try {
          var idoc = iframes[k].contentDocument;
          if (idoc) collectFromDoc(idoc, roots);
        } catch(eIf) { /* cross-origin */ }
      }
    } catch(e) {}

    var out = [];
    var seen = {};
    var idxMap = [];

    for (var i = 0; i < roots.length && out.length < MAX; i++) {
      var el = roots[i];
      if (!el || !el.tagName) continue;
      if (!isVisible(el)) continue;
      var href = (el.getAttribute && el.getAttribute('href')) || '';
      if (href && AD_RE.test(href)) continue;
      var name = accName(el);
      var kind = describe(el);
      // editor/iframe/textarea 는 name 없어도 보존 (에디터 탐색 보장)
      var _editorish = (kind === 'editor' || kind === 'iframe' ||
                        kind === 'textarea' || kind.indexOf('input') === 0);
      if (!name && !href && !_editorish) continue;
      if (kind === 'iframe' && !name) {
        var _src = (el.getAttribute('src') || '').substring(0, 60);
        var _ttl = (el.getAttribute('title') || '').substring(0, 40);
        name = '[iframe]' + (_ttl ? ' ' + _ttl : '') + (_src ? ' ' + _src : '');
      }
      if (kind === 'editor' && !name) {
        name = '[에디터 영역]';
      }
      var key = kind + '|' + name + '|' + href.substring(0, 60);
      if (seen[key]) continue;
      seen[key] = 1;
      var r = el.getBoundingClientRect();
      var item = {
        i: out.length + 1,
        kind: kind,
        text: name,
        bbox: [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)]
      };
      if (href) item.href = href.substring(0, 120);
      var ph = el.getAttribute && el.getAttribute('placeholder');
      if (ph && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA')) item.placeholder = ph.substring(0, 40);
      out.push(item);
      idxMap.push(el);
    }

    window.__EIDOS_IDX = idxMap;
    return JSON.stringify({ok: true, count: out.length, elements: out, url: location.href});
  } catch(e) {
    return JSON.stringify({ok: false, error: String(e)});
  }
})();
"""


# ─────────────────────────────────────────────────────────────────────────────
# Strategy Layer — verb별 사이트맵 후보. ASK_USER 전 Tier-2 에스컬레이션에 사용.
# 공통 경로만 등재 (사이트 불문). 목록형/편집 진입점이 될 가능성이 큰 순서.
# ─────────────────────────────────────────────────────────────────────────────
SITEMAP_BY_VERB: Dict[str, List[str]] = {
    "EDIT": [
        "/my-gigs", "/my-services", "/my-items", "/my-products",
        "/dashboard", "/manage", "/manage/services",
        "/account/services", "/account/products",
        "/seller/services", "/seller/gigs",
        "/expert/services", "/mypage/services",
    ],
    "CREATE": [
        "/new", "/create", "/register", "/add",
        "/my-gigs/new", "/service/new",
        "/seller/register", "/product/new",
    ],
    "READ": [
        "/search", "/browse", "/discover", "/explore",
        "/categories", "/all",
    ],
    "DELETE": [
        "/my-gigs", "/manage", "/account/services",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# URL Variant Generator — 검색·열람 실패 시 키워드/엔드포인트 변형 후보 생성
# 동일 URL 반복 차단 훅과 Tier 2 에스컬레이션에서 공용으로 사용.
# ─────────────────────────────────────────────────────────────────────────────
_QUOTE_CHARS = "\"'`\u2018\u2019\u201c\u201d"

_KEYWORD_STOP_TOKENS = (
    "에서", "에게", "으로", "하고", "한다", "하는", "했다", "합니다",
    "있는", "있다", "없는", "없다", "이다", "인가", "까지", "부터",
    "보다", "처럼", "만큼", "대해", "위해", "관련", "대한", "통해",
    "그리고", "그러나", "또한", "이후", "이전", "후에", "전에",
)


def _clean_keyword_candidate(candidate: str) -> str:
    """패턴에서 뽑힌 후보가 '검색 키워드'로 쓸만한지 가드. 실패 시 ''."""
    if not candidate:
        return ""
    candidate = candidate.strip()
    # 길이 가드 — 20자 초과는 문장 수준 구문
    if len(candidate) > 20:
        return ""
    # 공백 3개 이상 → 문장이지 키워드가 아님
    if candidate.count(" ") >= 3:
        return ""
    # 한국어 조사/서술어 포함 → 문장 조각
    if any(s in candidate for s in _KEYWORD_STOP_TOKENS):
        return ""
    return candidate


def _extract_keyword_from_prompt(task_prompt: str) -> str:
    """user prompt에서 검색 키워드 추출. 따옴표 → 명시 패턴 → None 순."""
    if not task_prompt:
        return ""
    # 1) 따옴표로 감싸진 문구 우선
    qm = re.search(
        rf"[{_QUOTE_CHARS}]([^{_QUOTE_CHARS}]{{2,40}})[{_QUOTE_CHARS}]",
        task_prompt,
    )
    if qm:
        cleaned = _clean_keyword_candidate(qm.group(1))
        if cleaned:
            return cleaned
    # 2) "X 검색/찾/열/페이지" 패턴 (한·영·숫자 허용)
    pm = re.search(
        r"([가-힣A-Za-z0-9][\w가-힣 ]{1,40}?)\s*(?:페이지|검색|찾|열람|열어)",
        task_prompt,
    )
    if pm:
        cleaned = _clean_keyword_candidate(pm.group(1))
        if cleaned:
            return cleaned
    return ""


def _generate_url_variants(
    failed_url: str,
    task_prompt: str = "",
    attempted: Optional[set] = None,
) -> List[str]:
    """실패한 URL + 사용자 프롬프트로부터 재시도용 URL 변형 목록을 우선순위순으로 반환.

    전략:
      - 나무위키: /Search?q=, /go/, /w/ (공백·퍼센트 인코딩 변형)
      - 위키피디아: Special:Search, /wiki/ 변형
      - 일반 사이트: /search?q= 범용 후보
    attempted(소문자 정규화된 URL 집합)에 포함된 것은 제외.
    """
    import urllib.parse as _up

    variants: List[str] = []
    attempted_lc = {(a or "").rstrip("/").lower() for a in (attempted or set())}

    m = re.match(r'(https?://)([^/]+)(/.*)?$', failed_url or "")
    if not m:
        return variants
    scheme, host, path = m.group(1), m.group(2).lower(), (m.group(3) or "/")
    base = f"{scheme}{host}"

    # 1) URL 경로에서 키워드 시도 (가장 확실한 소스)
    term = ""
    for prefix in ("/w/", "/wiki/", "/go/"):
        if prefix in path:
            raw = path.split(prefix, 1)[1].split("?")[0].split("#")[0]
            try:
                term = _up.unquote(raw).replace("_", " ").strip()
            except Exception:
                term = raw.replace("_", " ").strip()
            if term:
                break
    # 2) 경로에 없으면 프롬프트에서
    if not term:
        term = _extract_keyword_from_prompt(task_prompt)
    if not term:
        return variants

    term_space = term.replace("_", " ").strip()
    term_under = term_space.replace(" ", "_")
    enc_space = _up.quote(term_space, safe="")
    enc_under = _up.quote(term_under, safe="")

    # 사이트별 후보 — 우선순위: 검색 엔드포인트 > 대체 경로 > 인코딩 변형
    if "namu.wiki" in host:
        variants.extend([
            f"https://namu.wiki/Search?q={enc_space}",
            f"https://namu.wiki/go/{enc_space}",
            f"https://namu.wiki/w/{enc_under}",
            f"https://namu.wiki/w/{enc_space}",
        ])
    elif "wikipedia.org" in host:
        variants.extend([
            f"{base}/wiki/Special:Search?search={enc_space}",
            f"{base}/wiki/{enc_under}",
            f"{base}/wiki/{enc_space}",
        ])
    elif "youtube.com" in host or "youtu.be" in host:
        # YouTube 정식 검색 경로는 /results?search_query=… (/search?q= 는 404)
        variants.extend([
            f"https://www.youtube.com/results?search_query={enc_space}",
        ])
    else:
        variants.extend([
            f"{base}/search?q={enc_space}",
            f"{base}/Search?q={enc_space}",
        ])

    # 중복·이미 시도한 URL 제거
    seen: set = set()
    out: List[str] = []
    for v in variants:
        k = v.rstrip("/").lower()
        if k in seen or k in attempted_lc:
            continue
        seen.add(k)
        out.append(v)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Auth/SPA 상태 탐지 — ASK_USER 선제 에스컬레이션 (Layer 1)
# ─────────────────────────────────────────────────────────────────────────────
_AUTH_REDIRECT_PATTERNS = (
    "open=login_modal",
    "next_page=",
    "redirect_to=",
    "returnurl=",
    "return_url=",
    "?return=",
    "&return=",
    "?redirect=",
    "&redirect=",
    "/login?",
    "/signin?",
    "/auth?",
    "/account/login",
    "/member/login",
    "/users/sign_in",
)

_AUTH_GATED_PATH_PREFIXES = (
    "/seller/", "/sellers/",
    "/my-", "/my/",
    "/mypage", "/my-page",
    "/account/", "/accounts/",
    "/dashboard", "/manage",
    "/expert/", "/admin/",
)

# 인증 필요가 확정되면 UNHYDRATED 경로 블랙리스트로 반복 탐색 차단
_UNHYDRATED_HTML_THRESHOLD = 150_000   # 크몽 SPA 비인증 placeholder = ~98KB
_UNHYDRATED_ELEMENT_THRESHOLD = 5
# [2026-04-22] 편집/등록 경로는 실제 로그인 성공해도 form 하이드레이션까지
# 인풋/폼이 대량 렌더됨. 5개 임계로는 "shell만 들어온 비로그인" 상태와
# 구분 불가. 로그: /my-gigs/new 에서 element=13개로 unhydrated 판정 누락
# → AuthRecheck 통과 → FILL 반복실패 루프. 편집 경로는 임계를 20으로 상향.
_UNHYDRATED_ELEMENT_THRESHOLD_EDIT = 20
_EDIT_PATH_TOKENS = (
    "/new", "/create", "/edit", "/write", "/upload", "/register",
    "/post", "/compose", "/publish",
)


def _is_auth_redirect_url(url: str) -> bool:
    """URL이 로그인 모달/리다이렉트 시그니처를 포함하는지."""
    if not url:
        return False
    u = url.lower()
    return any(pat in u for pat in _AUTH_REDIRECT_PATTERNS)


def _is_auth_gated_path(url: str) -> bool:
    """URL 경로가 로그인 필요 영역인지 (seller/my-*/account/dashboard 등)."""
    if not url:
        return False
    try:
        m = re.match(r'^https?://[^/]+(/[^?#]*)', url)
        if not m:
            return False
        path = m.group(1).lower()
        return any(path.startswith(p) for p in _AUTH_GATED_PATH_PREFIXES)
    except Exception:
        return False


def _is_edit_path(url: str) -> bool:
    """URL이 편집/등록 계열 경로인지 (/new, /create, /edit, ...)."""
    if not url:
        return False
    try:
        m = re.match(r'^https?://[^/]+(/[^?#]*)', url)
        if not m:
            return False
        path = m.group(1).lower()
    except Exception:
        return False
    return any(tok in path for tok in _EDIT_PATH_TOKENS)


def _is_unhydrated_spa(html_len: int, element_count: int, url: str) -> bool:
    """
    SPA가 비인증 상태로 placeholder만 반환한 상태인지 휴리스틱.
    - HTML이 150KB 이하 AND
    - 인덱싱된 상호작용 요소가 (일반 5 / 편집경로 20) 이하 AND
    - URL이 인증 필요 경로
    편집 경로 임계 상향: 로그인 성공 후 form이 제대로 뜨면 input/button 다수가
    인덱싱되므로 20개 미만이면 여전히 shell 상태로 본다. (2026-04-22 패치)
    """
    if html_len <= 0:
        return False
    if html_len > _UNHYDRATED_HTML_THRESHOLD:
        return False
    _thr = (
        _UNHYDRATED_ELEMENT_THRESHOLD_EDIT
        if _is_edit_path(url)
        else _UNHYDRATED_ELEMENT_THRESHOLD
    )
    if element_count > _thr:
        return False
    return _is_auth_gated_path(url) or _is_edit_path(url)


# ─────────────────────────────────────────────────────────────────────────────
# [PRE-VETO 완화 2026-05-06] strike 임계 module 상수화 + 사용자 조정 가능
# ─────────────────────────────────────────────────────────────────────────────
# 사용자 불만: "맛집 하나 찾는데 너무 오래 걸림". 진단:
#   PRE-VETO strike 5/5 도달 후 bypass 하는데 그 사이 NAVIGATE-READ_PAGE-WRITE
#   루프 5회 반복 = 평균 60초 이상 낭비. 단순 작업 (맛집/추천/리스트) 은
#   2회 strike 만으로 bypass 충분.
# 정책:
#   - 일반 모드: HARD=3, STRONG=2 (기존 5/3 → 3/2 — 완화)
#   - simple 모드: HARD=2, STRONG=1 (instruction 키워드 매칭 시)
PV_HARD_LIMIT_DEFAULT = 3
PV_STRONG_LIMIT_DEFAULT = 2
PV_HARD_LIMIT_SIMPLE = 2
PV_STRONG_LIMIT_SIMPLE = 1

# simple-task 키워드 — 매칭 시 PRE-VETO 임계 + read_page_dup_gate 추가 완화.
# "적당히 조사하고 보고서" 의도 단순 태스크.
_SIMPLE_TASK_KEYWORDS = (
    "맛집", "추천", "리스트", "list", "top", "베스트", "best",
    "찾아", "찾아줘", "알려줘", "정리", "비교", "어디", "어느",
)


def _is_simple_task(text: str) -> bool:
    """instruction/goal 안 simple-task 키워드 매칭 — fast-path 활성화 판정."""
    if not text:
        return False
    t = str(text).lower()
    return any(kw in t for kw in _SIMPLE_TASK_KEYWORDS)


# ─────────────────────────────────────────────────────────────────────────────
# [Fix F 2026-05-06] step 노이즈 메시지 — 메인 채팅창 차단 대상
# ─────────────────────────────────────────────────────────────────────────────
# Why: 사용자 보고 — 자율 실행 시 [Step N] / [자동 탐색 Tier] / [막힘 감지] 같은
# step 진행 메시지가 메인 채팅창에 정신없이 찍혀 산만함. 콘솔 + 텔레그램 + 캐릭터
# 이벤트는 그대로 유지. settings.verbose_react_steps=True 면 우회.
_STEP_NOISE_PREFIXES = (
    "🧠 **[Step ",       # Step N — think 결정
    "✅ **[Step ",        # Step N 완료
    "⚠️ **[Step ",        # Step N 실패
    "🧠 [Step ",          # 변형 (간혹 다른 prefix)
    "✅ [Step ",
    "⚠️ [Step ",
    "🔍 **[자동 탐색",   # 자동 탐색 Tier 1/2/3
    "🔄 **[동일 URL 반복",
    "🛑 **[막힘 감지]",
    "🔁 **[PRE-VETO 자동 피벗]",
    "🔻 **[샘플링 다운그레이드]",
    # [2026-05-23 사용자 요청] Worker Idle/Watchdog 알림 — 메인 채팅창 산만함.
    # 콘솔 + 텔레그램 + EIDOS 플래너 자율 작업 panel 에서만 표시 (verbose 토글 시 ON).
    "⚠️ **[Worker Idle 감지",  # 1/3, 2/3 strike 메시지
    "🔴 **[Watchdog 한계 초과",  # hard limit abort 메시지
    "🔧 [Watchdog]",            # NAVIGATE pending force-resolve 로그
    "⚠️ [Watchdog]",
)


def _is_step_noise_message(message: str) -> bool:
    """메시지가 step 진행 류 (메인 채팅창에서 차단할 노이즈) 인지 판정."""
    if not message:
        return False
    return message.startswith(_STEP_NOISE_PREFIXES)


# ─────────────────────────────────────────────────────────────────────────────
# Phase B [2026-04-27] — WRITE Sampling Gate URL classifier
# ─────────────────────────────────────────────────────────────────────────────
# 최신로그.txt(2026-04-27) Gumroad chain 에서 product detail URL '/l/<slug>' 가
# 기존 _DETAIL_P_SG/_PV 패턴에 없어 detail counter 0 고정 → WRITE PRE-VETO 6회
# 무한루프. 추가로 list/search/discover 페이지가 detail 로 mis-classify 되는
# false positive 도 negative 패턴으로 차단.
_DETAIL_PATH_PATTERNS_SG = (
    "/gig/", "/gigs/", "/product/", "/products/", "/item/", "/items/",
    "/post/", "/posts/", "/article/", "/articles/", "/service/", "/services/",
    "/listing/", "/listings/", "/view/", "/detail/", "/details/",
    "/l/",          # Gumroad product link (gumroad.com/l/<slug>)
    "/dp/",         # Amazon ASIN
    "/pdp",         # Product Detail Page
    "/courses/", "/course/", "/lesson/", "/lessons/",
    "/book/", "/books/",
    # [Fix E1 2026-05-06] generic 매거진/리뷰/장소 사이트 detail 패턴.
    # Why: 최신로그.txt 10:06 siksinhot.com/P/56310 같은 매거진 사이트의 식당
    # 상세 URL 이 위 e-commerce 패턴에 안 잡혀 detail=0 고정 → WRITE Sampling
    # Gate 무한 거부. 일반 식당/장소/매거진 사이트의 흔한 detail path 추가.
    "/p/",          # siksinhot.com/P/<id>, mangoplate.com/restaurants/...
    "/place/", "/places/",
    "/restaurant/", "/restaurants/",
    "/spot/", "/spots/",
    "/store/", "/stores/", "/shop/", "/shops/",
    "/profile/", "/profiles/",
    "/entry/", "/entries/",
    "/review/", "/reviews/",
    "/venue/", "/venues/",
)

_LIST_PATH_PATTERNS_SG = (
    "/discover", "/best_sellers", "/best-sellers", "/bestsellers",
    "/search", "/feed", "/explore", "/categories", "/category",
    "/all-products", "/all_products", "/recent",
    "/trending", "/popular", "/browse",
)


def _classify_sampling_url(url_key: str) -> str:
    """url_key 를 'detail' / 'list' / 'other' 로 분류.
    list 패턴 우선 (detail 패턴 일부 겹쳐도 list 면 list)."""
    if not url_key:
        return "other"
    _ul = url_key.lower()
    if any(lp in _ul for lp in _LIST_PATH_PATTERNS_SG):
        return "list"
    if any(dp in _ul for dp in _DETAIL_PATH_PATTERNS_SG):
        return "detail"
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# Sampling Gate 화이트리스트 — LIST 페이지가 정보 완성인 도메인
# ─────────────────────────────────────────────────────────────────────────────
# 최신로그.txt(2026-04-27) 네이버 지도 "강남역 파스타 TOP5 검색" 케이스에서
# Sampling Gate 가 "5개 detail page 진입 후 read" 강요 → 무한 PRE-VETO 루프.
# 네이버 지도/카카오맵 같은 지도 서비스는 LIST 페이지에 식당명·평점·주소가
# 다 보여 detail 진입이 본질적으로 불필요. 이런 도메인은 sampling gate 자체
# 우회.
#
# 회귀 가드: kmong/gumroad/coupang 같은 e-commerce 는 화이트리스트 미포함 →
# 기존 sampling gate 동작 100% 보존.
_LIST_AS_INFO_DOMAINS = (
    # 지도/장소 검색 — LIST 가 정보 완성 (식당명·평점·주소 list 에 노출)
    "map.naver.com", "map.kakao.com", "kakaomap.com",
    "maps.google.com", "google.com/maps",
    "map.daum.net",
    # 검색엔진 결과 페이지 — SERP 에서 직접 정보 수집
    "search.naver.com", "search.daum.net",
    "google.com/search",
    # 백과/위키 — 한 페이지 내 모든 정보
    "namu.wiki", "ko.wikipedia.org", "en.wikipedia.org", "wikipedia.org",
    # 뉴스 검색 — list 가 곧 헤드라인 데이터
    "news.naver.com", "news.daum.net",
)


def _is_list_as_info_domain(url: str) -> bool:
    """URL 의 host 가 'LIST 페이지가 정보 완성' 도메인이면 True.
    sampling gate / WRITE PRE-VETO 가 이 케이스에선 자동 우회."""
    if not url:
        return False
    _u = url.lower()
    return any(dom in _u for dom in _LIST_AS_INFO_DOMAINS)


# ─────────────────────────────────────────────────────────────────────────────
# [List-as-Info 의미 게이트 2026-04-27] place entity 밀도 검사
# ─────────────────────────────────────────────────────────────────────────────
# 증상: list-as-info bypass 가 본문 200자+ 만 보면 우회 통과 → Verifier 단계에서
# "TOP5 검색 및 선정 속성 미충족"으로 FAIL → retry 루프. 단순 길이 게이트가
# *형식*만 보고 *의미*는 안 본 게 원인 (네이버지도 SPA 가 메인 페이지 본문
# 1500자만 줘도 평점/리뷰 정보는 없을 수 있음).
#
# 해결: bypass 조건에 "장소 entity 신호 N개+" 추가. 평점·리뷰·거리·영업 패턴이
# 충분히 등장할 때만 bypass — Verifier 가 잡기 전에 본문 부족을 액터 단에서 차단.
def _count_place_entities(text: str) -> int:
    """본문에서 장소/식당 entity 신호 개수 카운트.

    감지하는 신호 (각 장소는 보통 1~2개 신호 동반):
    - 평점 패턴: `★ 4.5`, `4.32점`, `4.5/5` 등
    - 리뷰 카운트: `리뷰 123`, `방문자리뷰`, `블로그 리뷰 N`
    - 거리 표기: `350m`, `1.2km`
    - 주소·영업 토큰: "영업중", "영업시간", "도보 N분"
    """
    if not isinstance(text, str) or not text:
        return 0
    import re as _re_pe
    n = 0
    # 평점
    n += len(_re_pe.findall(r"★\s*\d", text))
    n += len(_re_pe.findall(r"\b\d\.\d{1,2}\s*(?:점|/\s*5|/\s*10)\b", text))
    # 리뷰
    n += len(_re_pe.findall(r"리뷰\s*\d+|\d+\s*리뷰|방문자\s*리뷰|블로그\s*리뷰", text))
    # 거리
    n += len(_re_pe.findall(r"\b\d+(?:\.\d+)?\s*[mk]m\b", text))
    # 영업/도보
    n += len(_re_pe.findall(r"영업\s*(?:중|시간|종료)|도보\s*\d+\s*분", text))
    return n


def _has_place_density(text: str, n_required: int = 0) -> bool:
    """list-as-info bypass 의 의미 게이트.

    본문이 실제로 장소 N개 정보를 담고 있는지 확인.
    n_required>0 이면 max(2, n_required//2) 가 임계 (TOP5 → ≥2, TOP10 → ≥5).
    n_required=0 이면 절대 임계 ≥2 (단일 장소 조회 같은 케이스).
    """
    cnt = _count_place_entities(text)
    threshold = max(2, n_required // 2) if n_required > 0 else 2
    return cnt >= threshold


# ─────────────────────────────────────────────────────────────────────────────
# [TOP-N Selection Rule 2026-04-27] retry-time 산출물 선별 강제 룰
# ─────────────────────────────────────────────────────────────────────────────
# 증상: 최신로그.txt — "강남역 파스타 맛집 top5 → 보고서.txt" 케이스에서 LLM 이
# naver_map_results 본문(1552자) 을 단순 dump 만 함. retry 1·2·3·4 모두 같은
# missing_attr (TOP-5 미선별) 으로 Verifier FAIL. 토큰 Out 누적 ~12.4k 낭비.
#
# 원인: 기존 retry hint 가 "재시도 필요 / 산출물 보강" 같은 추상 지시뿐 → LLM
# 이 무엇을 어떻게 바꿔야 하는지 모름. 본문 형식(번호매김·N개 cap·정렬 기준)
# 을 hard rule 로 강제해야 dump → selection 으로 행동 변화 발생.
#
# 회귀 가드: 패턴 미매칭 시 N=0 → 룰 prepend 안 함 → 기존 retry 흐름 100% 보존.
import re as _re_topn

_TOPN_PATTERN = _re_topn.compile(
    r"(?:top|TOP|Top|상위|베스트|best|BEST|순위|랭킹|ranking|RANKING)"
    r"\s*[\.:]?\s*(\d{1,2})",
    _re_topn.IGNORECASE,
)


def _detect_topn_selection(schema, task_prompt: str) -> int:
    """schema.attributes 와 task_prompt/what 에서 'top N' / '상위 N' 패턴 추출.
    감지되면 N (1~50 정수), 아니면 0 반환. None-safe."""
    try:
        _parts = []
        if task_prompt:
            _parts.append(str(task_prompt))
        if schema is not None:
            _w = getattr(schema, "what", "") or ""
            if _w:
                _parts.append(str(_w))
            _attrs = getattr(schema, "attributes", None) or []
            for _a in _attrs:
                _an = getattr(_a, "name", "") or ""
                if _an:
                    _parts.append(str(_an))
        _scan = " ".join(_parts)
        if not _scan:
            return 0
        m = _TOPN_PATTERN.search(_scan)
        if not m:
            return 0
        n = int(m.group(1))
        return n if 1 <= n <= 50 else 0
    except (ValueError, TypeError, AttributeError):
        return 0


def _build_topn_selection_rule(n: int) -> str:
    """TOP-{n} 선별 강제 룰 텍스트. retry schema.how 머리에 prepend."""
    return (
        f"[SELECTION HARD RULE — TOP-{n} 선별 강제]\n"
        f"본문 전체 나열 절대 금지. 다음 형식 정확히 포함하여 작성:\n"
        f"## TOP {n}\n"
        f"1. <항목명> — <근거 1줄 (평점/리뷰/가격/거리 등)>\n"
        f"2. <항목명> — <근거 1줄>\n"
        f"... ({n}번째까지 정확히 {n}개)\n"
        f"{n}. <항목명> — <근거 1줄>\n"
        f"\n"
        f"규칙:\n"
        f"- {n+1}번째 이후 후보는 본문에서 제외 (전체 dump 금지).\n"
        f"- 정렬 기준이 prompt 에 미명시면 LLM 자체 판단으로 평점·리뷰수·"
        f"거리·인기 우선 — 단, 본문 첫 줄에 정렬 기준 한 줄 명시.\n"
        f"- 데이터에 {n}개 미만이면 실제 갯수만 작성하고 그 사실을 명시.\n"
        f"\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# [2026-04-27] DirectTask 라우팅 가드 — 외부 정보원 탐지
# ─────────────────────────────────────────────────────────────────────────────
# 증상: chain stage 가 task_type='write'/'general'/'report' 이면 _react_loop 가
# _run_direct_llm_task 로 라우팅 → 그 안에서 "브라우저나 외부 도구 없이 지식과
# 추론만으로" LLM 단일 호출 → 외부 사이트 정보가 필요한 작업도 환각 작문으로 끝.
#
# WHERE 도메인 보정 (eidos_mission_schema.py 2026-04-27) 으로 schema.where 가
# 외부 도메인을 가리키게 됐어도, 라우팅 레이어가 task_type 만 보고 DirectTask 로
# 빠지면 보정이 무력. 본 가드는 schema.where + original_prompt 에서 외부 도메인
# 시그널을 잡아 _EXTERNAL_ACTION_STAGES 가드와 대칭으로 ReAct 강제.
def _has_external_source(schema) -> bool:
    """schema 의 WHERE 또는 original_prompt 에 외부 사이트/도메인이 있으면 True.

    매칭 기준 (OR):
    - eidos_mission_schema._URL_PATTERN 매칭 (http://..., domain.tld 등)
    - eidos_mission_schema._BROWSER_SITE_HINTS 한글/영문 사이트명 (나무위키, youtube 등)

    None/빈 schema 는 안전하게 False (기존 라우팅 동작 보존).
    """
    if schema is None:
        return False
    _w = (getattr(schema, "where", "") or "")
    _op = (getattr(schema, "original_prompt", "") or "")
    blob = (_w + " " + _op).lower().strip()
    if not blob:
        return False
    try:
        from eidos_mission_schema import _URL_PATTERN, _BROWSER_SITE_HINTS
    except Exception:
        return False
    if _URL_PATTERN.search(blob):
        return True
    return any(h in blob for h in _BROWSER_SITE_HINTS)


# [2026-04-27] DirectTask 라우팅 가드 — 연구-기반 산출물 키워드 탐지
# 증상: prompt 에 URL/사이트명 0개여도 "기획안/시장조사/벤치마크" 같은 연구
# 산출물은 외부 조사 없이는 환각 작문으로 끝남. _has_external_source 가 False
# 인 ad-hoc file_write 케이스(최신로그 plan.md 7163자) 차단용 안전망.
# WHERE 자동 주입(eidos_mission_schema 2026-04-27) 이 발동 안 한 케이스 대비.
def _has_research_intent_in_schema(schema) -> bool:
    """schema 의 WHERE + original_prompt 에 연구-기반 산출물 키워드가 있으면 True.

    매칭은 eidos_mission_schema._has_research_intent 위임 — 키워드 단일 출처.
    None/빈 schema 는 안전하게 False.
    """
    if schema is None:
        return False
    _w = (getattr(schema, "where", "") or "")
    _op = (getattr(schema, "original_prompt", "") or "")
    blob = _w + " " + _op
    if not blob.strip():
        return False
    try:
        from eidos_mission_schema import _has_research_intent
    except Exception:
        return False
    return _has_research_intent(blob)


# [2026-04-27] DirectTask 라우팅 가드 — 외부-행동 키워드 탐지
# 증상: prompt 에 "네이버 지도 강남역 맛집 검색" 처럼 한글 사이트명 + 트랜잭션
# 행동(맛집검색·예약·주문) 이 있어도 _has_external_source(영문 도메인 위주) +
# _has_research_intent(연구 키워드) 둘 다 놓치는 케이스. WHERE 자동 주입이 안
# 발동했을 때 ReAct 강제용 마지막 안전망. _has_research_intent_in_schema 와 대칭.
def _has_external_action_intent_in_schema(schema) -> bool:
    """schema 의 WHERE + original_prompt 에 외부-행동 키워드가 있으면 True.

    매칭은 eidos_mission_schema._has_external_action_intent 위임 — 키워드 단일 출처.
    None/빈 schema 는 안전하게 False.
    """
    if schema is None:
        return False
    _w = (getattr(schema, "where", "") or "")
    _op = (getattr(schema, "original_prompt", "") or "")
    blob = _w + " " + _op
    if not blob.strip():
        return False
    try:
        from eidos_mission_schema import _has_external_action_intent
    except Exception:
        return False
    return _has_external_action_intent(blob)


# ─────────────────────────────────────────────────────────────────────────────
# [Lazy Action Guard 2026-04-27] LLM 게으름 차단 — A+B 패치
# ─────────────────────────────────────────────────────────────────────────────
# 증상: 자율실행 중 LLM 이 외부 검색이 필요한 prompt 를 받고도 "내부 지식으로 바로
# 보고서 작성" 같은 자기 위임으로 빠져 자료 수집 단계 (NAVIGATE/READ_PAGE) 를
# 0건으로 건너뛰고 곧바로 WRITE 발사 → 환각 dump → Verifier FAIL.
#
# 최신로그.txt 23:31:20 임진왜란 케이스 결정적 단서:
#   Step 2 [WRITE] 보고서.txt — 이유: "[↑신호: 외부 네트워크 접근 금지]
#   [통합: 내부 지식으로 바로 보고서 작성"
#
# Patch A: WRITE 발사 시점에 _react_var_urls (READ_PAGE 결과) 0건 + research 의도
#          → WRITE 차단 + NAVIGATE 로 강제 교체 + schema.how 에 자료 수집 강제 hint
# Patch B: LLM 이유 텍스트에 "내부 지식으로/브라우저 없이/검색 없이" 등 lazy 토큰
#          매칭 시 → 해당 step 차단 + 카운터 누적 + hint 주입

_LAZY_REASON_TOKENS = (
    # 한글 — LLM 자기 위임 신호 (가장 명확한 게으름 패턴)
    "내부 지식으로", "내부 지식 기반", "지식만으로", "지식 기반으로",
    "브라우저 없이", "외부 도구 없이", "검색 없이", "검색하지 않고",
    "외부 접근 없이", "외부 도움 없이", "도구 사용 없이",
    # LLM 이 forbidden_zones 토큰을 자기 정당화로 인용
    "외부 네트워크 접근 금지",
    # 영문
    "internal knowledge", "without browser", "without search",
    "without external", "without tools",
)

_RESEARCH_VERB_TOKENS = (
    # eidos_mission_schema._RESEARCH_VERBS 와 동일 — 키워드 셋 일관성
    "조사", "탐색", "찾아", "검색", "알아봐", "알아보", "리서치",
    "research", "lookup", "search", "investigate", "explore",
)


def _is_lazy_reason(reason: str) -> bool:
    """LLM 이유 텍스트에 '게으름' 신호 토큰 매칭이면 True."""
    if not reason:
        return False
    _r = str(reason)
    _r_lc = _r.lower()
    return any(t in _r or t.lower() in _r_lc for t in _LAZY_REASON_TOKENS)


def _has_research_verb(text: str) -> bool:
    """text 에 외부 검색 동사 1개 이상 매칭이면 True. None/빈 안전."""
    if not text:
        return False
    _t = str(text)
    _t_lc = _t.lower()
    return any(v in _t or v in _t_lc for v in _RESEARCH_VERB_TOKENS)


def _count_data_vars(schema) -> int:
    """schema._react_var_urls 의 URL→var 매핑 중 본문 200자+ 인 var 갯수.

    READ_PAGE/innerText 로 수집한 진짜 외부 자료만 카운트 — WRITE 산출물 (target
    파일) 은 _react_var_urls 에 매핑 안 되므로 자연 제외.
    """
    if schema is None:
        return 0
    try:
        urls = getattr(schema, "_react_var_urls", None) or {}
        rvs = getattr(schema, "_react_vars", None) or {}
        n = 0
        for _u, _vn in urls.items():
            _val = rvs.get(_vn, "")
            if isinstance(_val, str) and len(_val) >= 200:
                # 차단/오류 페이지 마커 제외
                if not _val.lstrip().startswith("[⚠️ BLOCKED_PAGE"):
                    n += 1
        return n
    except Exception:
        return 0


def _is_lazy_write_action(action: str, schema, task_prompt: str) -> bool:
    """WRITE 가 게으른 발사인지 판정.

    조건 (모두 충족):
    - action == "WRITE"
    - schema.task_type ∈ {"file_write", "write", "report", "general"}
    - prompt OR schema.what/how 에 research 동사 매칭
    - _count_data_vars(schema) == 0 (READ_PAGE 결과 0건)
    """
    if action != "WRITE" or schema is None:
        return False
    _tt = (getattr(schema, "task_type", "") or "").strip().lower()
    if _tt not in ("file_write", "write", "report", "general"):
        return False
    # prompt + schema 본문 합산 — research 의도 검사
    _what = getattr(schema, "what", "") or ""
    _how = getattr(schema, "how", "") or ""
    _op = getattr(schema, "original_prompt", "") or ""
    _scan = " ".join([task_prompt or "", _what, _how, _op])
    if not _has_research_verb(_scan):
        return False
    # READ_PAGE 결과 0건 + research 의도 → 게으름 확정
    return _count_data_vars(schema) == 0


def _is_lazy_ask_action(action: str, schema, task_prompt: str) -> bool:
    """ASK_USER 가 게으른 발사인지 판정 — Lazy Write Guard 와 같은 객관 상태 기반.

    조건 (모두 충족):
    - action == "ASK_USER"
    - prompt OR schema.what/how/original_prompt 에 research 동사 매칭
    - _count_data_vars(schema) == 0 (READ_PAGE 결과 0건 — 검색 한 번도 안 함)

    Why: 2026-05-03 최신로그.txt — LLM 이 자료 수집 시도 0건 상태에서
    "분석할 경쟁 서비스의 URL을 알려주세요" 류 ASK_USER 로 사용자에게 떠넘김.
    research 의도 task 인데 _count_data_vars == 0 이면 검색 한 번도 안 한 게으름
    확정. 이미 검색 시도 후 부족해 묻는 건 정당하므로 _count_data_vars > 0 이면
    통과 (cap 자동 해제). 키워드 매칭 0 — 표현 변경 우회 불가능.
    """
    if action != "ASK_USER" or schema is None:
        return False
    _what = getattr(schema, "what", "") or ""
    _how = getattr(schema, "how", "") or ""
    _op = getattr(schema, "original_prompt", "") or ""
    _scan = " ".join([task_prompt or "", _what, _how, _op])
    if not _has_research_verb(_scan):
        return False
    return _count_data_vars(schema) == 0


# ─── Lazy redirect URL placeholder 가드 (2026-04-29) ─────────────────────────
# 최신로그 b555e8b6 케이스: chain stage prompt 가 `[고정 주제] 자율 탐색 — 경쟁`
# 으로 박혀 있어 redirect URL 이 google.com/search?q=%5B고정%20주제%5D 가 되어
# 의미 없는 검색 → 자료 0건 유지 → Lazy Write Guard 36회 재발동 → 60 step 한도
# 초과 → task FAIL. 본 가드는 (1) brackets-only placeholder 검출 후 query 채널
# 폴백 (2) 모든 폴백 실패 시 빈 string 반환 → 호출자가 NAVIGATE 강제 안 함.
_PLACEHOLDER_BRACKET_TOKENS = (
    # 한글 placeholder
    "고정 주제", "고정주제", "주제", "예시", "샘플", "여기", "여기에",
    "채울", "채울 부분", "예제", "대상", "목표 키워드", "키워드", "이름",
    "타이틀", "제목", "내용", "TBD", "tbd",
    # 영문 placeholder
    "xxx", "yyy", "zzz", "abc", "foo", "bar", "placeholder", "example",
    "sample", "topic", "subject", "title", "name", "todo",
    # [Patch 2026-04-29 dogfooding] chain stage prompt template 잔여 system instruction
    "주제 변경 금지", "변경 금지", "주제변경금지",
    "하위 stage", "하위 단계", "stage 에서", "stage에서", "in stage",
    "필수 정보", "필수정보",
)


def _is_placeholder_brackets(s: str) -> bool:
    """문자열이 brackets-only placeholder 패턴인지.

    True 조건 (any):
    - brackets 안 내용이 _PLACEHOLDER_BRACKET_TOKENS 중 하나
    - brackets 안 내용이 ALL-CAPS 알파벳만 3-6 chars (XXX, YYYY 등)
    - brackets 안 내용이 `?` `_` 등 비알파/비한글 토큰만
    """
    import re as _re_p
    if not s:
        return False
    _s = s.strip()
    _m = _re_p.match(r"^\[([^\[\]\n]+)\]$", _s)
    if not _m:
        return False
    _inner = _m.group(1).strip().lower()
    if _inner in _PLACEHOLDER_BRACKET_TOKENS:
        return True
    # ALL CAPS 알파벳만 3-6 chars
    if _re_p.match(r"^[a-z]{3,6}$", _inner) and _inner.upper() == _m.group(1).strip():
        return True
    # 비-단어 토큰만 (?, _, *, -, .)
    if _re_p.match(r"^[\?\_\*\-\.\s]+$", _inner):
        return True
    return False


def _strip_placeholder_brackets(text: str) -> str:
    """텍스트의 placeholder brackets 모두 제거 (시작 + 본문 전체).

    [Patch 2026-04-29 dogfooding] 이전 시작 brackets 만 → 본문 전체 검사로 확장.
    chain stage prompt 의 system instruction 잔여물 (`[주제 변경 금지]` 등) 도 strip.
    또 multi-line 시 첫 의미 있는 라인만 추출.

    예: `[고정 주제] 자율탐색 \n[주제 변경 금지] 하위 stage` → `자율탐색`
        `강남역 파스타 TOP5` → 그대로
        `[강남역] 파스타 맛집` → 그대로 ([강남역] 은 placeholder 아님)
    """
    import re as _re_s
    if not text:
        return ""
    _t = text.strip()
    # 모든 [...] 매칭 — placeholder 면 제거, 아니면 보존
    _pattern = _re_s.compile(r"\[([^\[\]\n]+)\]")
    def _replace(m):
        if _is_placeholder_brackets(m.group(0)):
            return ""  # placeholder → 제거
        return m.group(0)  # 진짜 의미 brackets → 보존
    _cleaned = _pattern.sub(_replace, _t)
    # multi-line 시 첫 의미 있는 라인만 (chain stage 가 multi-line task_prompt 만들 수 있음)
    _lines = [_l.strip() for _l in _cleaned.split("\n") if _l.strip()]
    if _lines:
        _cleaned = _lines[0]
    # 연속 공백 정리
    _cleaned = _re_s.sub(r"\s+", " ", _cleaned).strip()
    return _cleaned


def _extract_lazy_query(schema, task_prompt: str) -> str:
    """Lazy redirect URL 의 검색어 추출. 폴백 우선순위:
       1. task_prompt (placeholder strip)
       2. schema.what
       3. schema.original_prompt
       4. schema.how 의 첫 라인
       모두 placeholder/빈 시 → 빈 string.
    """
    import re as _re_q
    candidates = []
    if task_prompt:
        candidates.append(_strip_placeholder_brackets(task_prompt))
    if schema is not None:
        for attr in ("what", "original_prompt"):
            _v = getattr(schema, attr, "") or ""
            if _v:
                candidates.append(_strip_placeholder_brackets(_v))
        _how = getattr(schema, "how", "") or ""
        if _how:
            _first = (_how.splitlines() or [""])[0]
            candidates.append(_strip_placeholder_brackets(_first))
    for _c in candidates:
        _clean = _re_q.sub(r"\s+", " ", _c).strip()[:60]
        if not _clean:
            continue
        # 전체가 brackets 으로 감싸져 있고 placeholder 면 skip
        if _is_placeholder_brackets(_clean):
            continue
        # 의미 있는 단어 1개 이상이어야 — 한글/영문/숫자
        if not _re_q.search(r"[\w가-힣]", _clean):
            continue
        return _clean
    return ""


def _build_lazy_redirect_url(schema, task_prompt: str) -> str:
    """게으른 WRITE 차단 후 NAVIGATE 강제할 URL 생성.

    schema.where 에서 도메인 추출 → 그 도메인 search URL.
    실패 시 google.com/search?q=<keyword>.
    placeholder 검출 시 폴백 채널 → 모두 실패 시 빈 string (호출자 처리).
    """
    import re as _re_lz
    import urllib.parse as _urlp
    _where = (getattr(schema, "where", "") or "").lower() if schema else ""
    _kw_clean = _extract_lazy_query(schema, task_prompt)
    if not _kw_clean:
        # placeholder 만 있어 검색어 추출 실패 → 호출자가 NAVIGATE 강제 안 함
        return ""
    _kw = _urlp.quote(_kw_clean, safe="")
    # 도메인 추출 — 우선순위: 네이버지도 → 카카오맵 → google → 기타
    if "map.naver" in _where or "네이버 지도" in _where or "네이버지도" in _where:
        return f"https://map.naver.com/v5/search/{_kw}"
    if "kakao" in _where and "map" in _where:
        return f"https://map.kakao.com/?q={_kw}"
    # 일반 도메인 토큰 추출
    _m = _re_lz.search(r"([\w\-]+\.[a-z]{2,}(?:\.[a-z]{2,})?)", _where)
    _domain = _m.group(1) if _m else ""
    # [2026-05-26 fix] google.com 내장 브라우저 NAVIGATE 차단 — CAPTCHA·계정 정지 위험.
    # 사용자 결정: 검색은 SerpAPI 가 primary. NAVIGATE 폴백은 네이버로.
    # (구글 검색 결과가 정말 필요하면 web_search() 함수가 SerpAPI 로 텍스트 반환 — NAVIGATE 우회.)
    if _domain and "google" in _domain:
        return f"https://search.naver.com/search.naver?query={_kw}"
    if _domain:
        # 사이트 한정 검색 — 네이버 검색의 site: 연산자
        return f"https://search.naver.com/search.naver?query=site%3A{_domain}+{_kw}"
    # 폴백
    return f"https://www.google.com/search?q={_kw}"


# ─────────────────────────────────────────────────────────────────────────────
# CLICK 반복 pre-block → 로그인 필요 추정 에스컬레이션 (2026-04-20)
# ─────────────────────────────────────────────────────────────────────────────
# 증상: register 단계(kmong 상품등록 등)에서 LLM이 '상품등록' 같은 버튼 [6]을
# 반복 지명 → CLICK no-op → pre-block → 힌트 주입 → LLM이 다시 [6] → 데드락.
# 실제 원인은 비로그인 상태라 해당 버튼이 모달/무반응 상태인 경우가 대부분.
# 목표: N회 이상 같은 target이 pre-block되고, 그 element가 로그인-보호 행위를
# 가리키는 문구이면 즉시 Layer 1 auth_required 경로로 승격한다.
_AUTH_ACTION_KEYWORDS = (
    # 한국어 로그인/판매자 행위
    "로그인", "로그아웃", "회원가입", "가입하기", "내정보", "내 정보",
    "마이페이지", "마이 페이지", "판매자", "판매하기", "상품등록", "상품 등록",
    "글쓰기", "글 쓰기", "등록하기", "올리기", "문의하기", "문의 하기",
    "결제", "주문", "구매하기", "장바구니", "찜", "좋아요",
    # 영문 로그인/작성 행위
    "sign in", "sign up", "login", "log in", "logout", "log out",
    "my account", "my page", "sell", "post ", "write", "publish",
    "submit", "checkout", "order", "buy ", "add to cart",
)


def _click_target_looks_auth_protected(element: Optional[dict], current_url: str) -> bool:
    """CLICK pre-block된 target이 로그인 보호 행위인지 휴리스틱 판정.

    True 조건(OR):
    - element.text/aria/href에 _AUTH_ACTION_KEYWORDS 포함
    - element.href가 로그인 리다이렉트 패턴
    - 현재 URL이 auth-gated 경로 (SPA 상태로 이미 블랙리스트 대상)
    """
    if _is_auth_gated_path(current_url) or _is_auth_redirect_url(current_url):
        return True
    if not element:
        return False
    try:
        text_pool = " ".join(str(element.get(k) or "") for k in
                             ("text", "aria", "aria_label", "title", "href", "placeholder")).lower()
    except Exception:
        return False
    if not text_pool.strip():
        return False
    # href가 /login·/signin 등 직접 가리키면 즉시 True
    href = str(element.get("href") or "").lower()
    if href and _is_auth_redirect_url(href):
        return True
    return any(kw.lower() in text_pool for kw in _AUTH_ACTION_KEYWORDS)


# ─────────────────────────────────────────────────────────────────────────────
# 봇 차단(anti-bot) 사이트 감지
# ─────────────────────────────────────────────────────────────────────────────
# 쿠팡·아마존·인스타그램 등은 QtWebEngine fingerprint를 봇으로 판정하면
# 200 OK 로 응답하되 body 는 ~300자 빈 쉘만 내려준다(2026-04-19 coupang 로그 참조).
# 이 상태를 _is_unhydrated_spa 가 잡지 못하는 이유:
#   (a) URL 이 auth_gated_path 가 아님 (/, /np/search 등 공개 경로)
#   (b) HTML 은 150KB 이하이지만 SPA hydration 대기로는 복구 불가
# → 전용 휴리스틱 + 도메인 화이트리스트로 빠르게 포기하고 대안 제시.
_BOT_BLOCK_DOMAINS = (
    "coupang.com",
    "amazon.com", "amazon.co.kr", "amazon.co.jp", "amazon.de", "amazon.fr",
    "instagram.com",
    "facebook.com",
    "linkedin.com",
    "tiktok.com",
    "x.com", "twitter.com",
)

_BOT_BLOCK_HTML_MAX = 600   # 쿠팡 실측 291~301 chars. 여유 있게 600.


def _extract_host(url: str) -> str:
    if not url:
        return ""
    try:
        m = re.match(r'^https?://([^/]+)', url)
        return m.group(1).lower() if m else ""
    except Exception:
        return ""


def _is_bot_blocked_shell(html_len: int, url: str) -> bool:
    """
    알려진 안티봇 사이트가 빈 쉘만 내려준 상태인지 판정.
    - html_len 이 비정상적으로 작음 (<= _BOT_BLOCK_HTML_MAX)
    - 호스트가 _BOT_BLOCK_DOMAINS 에 일치 (서브도메인 허용: search.coupang.com 등)
    - about:blank / data: / chrome-error URL 은 제외
    """
    if html_len <= 0 or html_len > _BOT_BLOCK_HTML_MAX:
        return False
    if not url or url.startswith(("about:", "data:", "chrome-error:")):
        return False
    host = _extract_host(url)
    if not host:
        return False
    return any(host == d or host.endswith("." + d) for d in _BOT_BLOCK_DOMAINS)


# [2026-04-24] 본문 텍스트 기반 차단/오류 페이지 감지.
# Why: smartstore.naver.com / shopping.naver.com 같은 도메인은 _BOT_BLOCK_DOMAINS
# 화이트리스트에 없고, 보안/CAPTCHA 페이지가 길이 600 보다 클 수 있어 길이+도메인
# 휴리스틱이 못 잡음. innerText 본문에 명백한 차단 신호가 있으면 길이 무관 즉시 판정.
# 케이스 (실제 로그):
#   "NAVER 보안 확인을 완료해 주세요. 이 절차는 귀하가 실제 사용자임을 확인하여..."
#   "페이지를 찾을 수 없습니다."
_BLOCK_TEXT_SIGNALS: tuple = (
    # 한국 (네이버/카카오 등)
    "보안 확인을 완료",
    "보안 확인이 필요",
    "NAVER 보안 확인",
    "보안 인증",
    "정상적이지 않은 접근",
    "비정상적인 접근",
    "비정상 트래픽",
    "페이지를 찾을 수 없습니다",
    "요청하신 페이지를 찾을 수 없",
    "일시적으로 이용할 수 없",
    # 글로벌 (Cloudflare / Akamai / generic)
    "Just a moment...",
    "Just a moment…",
    "Please verify you are human",
    "Please verify you are a human",
    "Verify you are human",
    "Checking your browser",
    "Access Denied",
    "Access denied",
    "403 Forbidden",
    "Robot or human?",
    "Are you a robot",
    "Pardon Our Interruption",
    "captcha",
    "CAPTCHA",
    "reCAPTCHA",
)


def _is_blocked_by_text(text: str) -> tuple:
    """innerText 본문에 차단/오류 신호가 있는지 판정.

    Returns: (is_blocked: bool, matched_signal: str)
    text 길이 4000자 이내에서만 검사 (본문 도입부에 차단 안내가 있는 게 일반적).
    """
    if not text:
        return (False, "")
    head = text[:4000]
    head_lower = head.lower()
    for sig in _BLOCK_TEXT_SIGNALS:
        # 한글/특수문자는 그대로, ascii 는 lower 매칭
        if any(ord(c) > 127 for c in sig):
            if sig in head:
                return (True, sig)
        else:
            if sig.lower() in head_lower:
                return (True, sig)
    return (False, "")


# [2026-04-24] CAPTCHA 전용 HTML 패턴 — recaptcha/hcaptcha iframe·위젯·네이버 보안문자.
# Why: 기존 _BLOCK_TEXT_SIGNALS 는 차단/오류 페이지 일반 감지(captcha 키워드 포함되긴 함).
# false positive 줄이려면 본문 키워드보다 DOM 시그널이 정확. raw HTML 정규식이면 별도 JS
# 호출 없이 _observe() 의 raw_html 으로 곧장 매칭 가능.
_CAPTCHA_HTML_PATTERNS: tuple = (
    # Google reCAPTCHA (iframe + 위젯 클래스)
    (r'iframe[^>]*src=["\'][^"\']*google\.com/recaptcha',  "recaptcha_iframe"),
    (r'iframe[^>]*src=["\'][^"\']*recaptcha',              "recaptcha_iframe_alt"),
    (r'class=["\'][^"\']*\bg-recaptcha\b',                  "recaptcha_widget"),
    (r'\bg-recaptcha-response\b',                            "recaptcha_response_field"),
    # hCaptcha
    (r'iframe[^>]*src=["\'][^"\']*hcaptcha',                "hcaptcha_iframe"),
    (r'class=["\'][^"\']*\bh-captcha\b',                    "hcaptcha_widget"),
    (r'\bh-captcha-response\b',                              "hcaptcha_response_field"),
    # Cloudflare Turnstile
    (r'class=["\'][^"\']*\bcf-turnstile\b',                 "cf_turnstile"),
    (r'iframe[^>]*src=["\'][^"\']*challenges\.cloudflare',  "cf_challenge_iframe"),
    # 네이버 보안문자 (이미지 + input)
    (r'<img[^>]+src=["\'][^"\']*captcha',                   "naver_captcha_img"),
    (r'<input[^>]+name=["\'][^"\']*captcha',                "captcha_input_field"),
    # 네이버 보안 확인 페이지 (URL 라우팅이 따로지만 iframe 안에 들어올 때 대비)
    (r'data-sitekey=["\']',                                  "sitekey_attr"),
)


def _detect_captcha_in_html(html: str) -> tuple:
    """raw HTML 에서 CAPTCHA 위젯/iframe 시그널을 정규식으로 감지.

    Returns: (has_captcha: bool, matched_signal: str)
    head 60KB 만 검사 (캡챠 위젯은 통상 페이지 도입부 또는 폼 영역에 위치).
    """
    if not html:
        return (False, "")
    head = html[:60000]
    for pat, label in _CAPTCHA_HTML_PATTERNS:
        try:
            if re.search(pat, head, re.I):
                return (True, label)
        except Exception:
            continue
    return (False, "")


def _suggest_alternative_routes(orig_url: str, task_prompt: str) -> List[Dict[str, str]]:
    """
    봇 차단 도메인에 대한 대안 검색 경로 생성.
    task_prompt 를 쿼리로 그대로 인코딩해서 네이버쇼핑/다나와/구글 등에 태운다.
    Returns: [{"name": "네이버쇼핑", "url": "https://..."}]
    """
    from urllib.parse import quote
    host = _extract_host(orig_url)
    q = (task_prompt or "").strip()
    q_enc = quote(q[:80]) if q else ""

    alts: List[Dict[str, str]] = []
    if not q_enc:
        return alts

    if any(host == d or host.endswith("." + d)
           for d in ("coupang.com", "amazon.com", "amazon.co.kr",
                     "amazon.co.jp", "amazon.de", "amazon.fr")):
        alts.append({"name": "네이버쇼핑",
                     "url": f"https://search.shopping.naver.com/search/all?query={q_enc}"})
        alts.append({"name": "다나와",
                     "url": f"https://search.danawa.com/dsearch.php?query={q_enc}"})
        alts.append({"name": "구글 쇼핑",
                     "url": f"https://www.google.com/search?tbm=shop&q={q_enc}"})
    elif any(host == d or host.endswith("." + d)
             for d in ("instagram.com", "twitter.com", "x.com",
                       "tiktok.com", "facebook.com", "linkedin.com")):
        alts.append({"name": f"구글 (site:{host} 검색)",
                     "url": f"https://www.google.com/search?q=site%3A{host}+{q_enc}"})
        alts.append({"name": "구글 일반 검색",
                     "url": f"https://www.google.com/search?q={q_enc}"})
    else:
        alts.append({"name": "구글 검색",
                     "url": f"https://www.google.com/search?q={q_enc}"})
    return alts


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ActionStep:
    """단일 실행 단계"""
    step_no:    int
    action:     str          # NAVIGATE / FILL / CLICK / WRITE / WAIT / REPORT / DONE
    target:     str = ""     # URL, CSS selector, 보고 메시지 등
    content:    str = ""     # FILL 내용, WRITE 프롬프트, WAIT 초수(str)
    status:     str = "PENDING"   # PENDING / RUNNING / DONE / FAILED
    result:     str = ""
    started_at: float = 0.0
    ended_at:   float = 0.0
    # [Phase A] Approval Gateway 메타 ― 실행 직전에 _execute_step가 재분류·게이트
    risk_level:        str   = "LOW"     # LOW / MEDIUM / HIGH
    approval_required: bool  = False
    cost_estimate:     float = 0.0
    # [Phase 1 Telegram UX] LLM 이 이 step 을 선택한 이유 — 승인 메시지 "Why:" 에 노출
    reason:            str   = ""
    # [제어실 Phase 1] step 단위 임의 메타 (goal_id 등). 자동 태깅 훅이 주입.
    metadata:          Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionPlan:
    """하나의 task_prompt에 대한 실행 계획"""
    task_id:    str
    task_prompt: str
    goal:       str
    steps:      List[ActionStep] = field(default_factory=list)
    status:     str = "PENDING"   # PENDING / RUNNING / DONE / FAILED
    created_at: float = field(default_factory=time.time)
    variables:  Dict[str, str] = field(default_factory=dict)  # WRITE 결과 저장소
    # [제어실 Phase 1] plan 단위 메타 (goal_id, classify_rationale 등).
    # 모든 step 이 공유하는 정보 (상속 훅에서 각 step.metadata 로 전파).
    metadata:   Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# ActionDispatcher
# ─────────────────────────────────────────────────────────────────────────────

class ActionDispatcher:
    """
    dB/dt = f(S, G) 자율 실행 루프.

    GoalScheduler → schedule[PENDING] → ActionPlan 수립 → 단계별 실행 → 중간보고
    """

    LOOP_INTERVAL = 60       # 루프 주기(초) - 매 N초마다 PENDING 작업 확인
    MAX_STEPS     = 12       # 한 작업당 최대 스텝 수
    STEP_TIMEOUT  = 30       # 스텝 타임아웃(초)

    # [OpportunityScanner/MissionDirector 등 외부 모듈이 _pending_chain/
    # approve_pending_chain 경로를 타려면 dispatcher 참조가 필요하다. core 는
    # dispatcher 를 역참조하지 않으므로 최신 인스턴스를 클래스에 보관한다.
    # Why: 승인 플래그(orch.approve_chain)만 찍고 끝내면 dispatcher 큐에 schema
    # 가 안 꽂혀 5단계 자율 실행이 영원히 시작되지 않는다.]
    _instance: "Optional[ActionDispatcher]" = None

    def __init__(
        self,
        core: "EidosCore",
        report_callback: Optional[Callable[[str], None]] = None,
    ):
        self.core = core
        type(self)._instance = self
        self._report_cb = report_callback   # 채팅창 메시지 전송 함수
        self._running   = False
        self._current_plan: Optional[ActionPlan] = None
        self._paused_plan: Optional[ActionPlan] = None   # ASK_USER 후 일시정지된 plan
        self._paused_task: Optional[Dict] = None          # 위 plan에 대응하는 task dict
        self._loop_task: Optional[asyncio.Task] = None
        # ── MissionSchema ──────────────────────────────────────────────────
        self._pending_schema = None   # 사용자 승인 대기 중인 MissionSchema
        self._active_schema  = None   # 승인 완료 후 ReAct 앵커로 사용되는 MissionSchema
        # ── MissionChain (2026-04-19) ──────────────────────────────────────
        # 한 마디 → 다단계 파이프라인 자동 진행을 위한 체인 대기/실행 슬롯.
        # 체인 승인 시 첫 stage 의 MissionSchema 가 자동 조립되어 _active_schema 에 실림.
        self._pending_chain = None    # PipelineOrchestrator 가 제안한 체인 (승인 대기)
        # ── Chain 실패 UX (P0 2026-04-19) ─────────────────────────────────
        # 체인 stage 실패 시 세팅되어 사용자 입력("재시도/포기/롤백 N")을 기다림.
        # 값: {chain_id, stage_name, stage_index, reason, history_tail, failed_at}
        self._pending_chain_failure: Optional[Dict[str, Any]] = None
        # ── CodeBuilder 결과 프리뷰 (P2-7 2026-04-19) ─────────────────────
        # dev 단계 완료 직후 세팅되어 사용자가 언제든 `미리보기 N` / `재생성` 입력 가능.
        # 다음 stage 로 진행하더라도 유지 — 사용자가 나중에라도 확인할 수 있도록.
        # 새로운 code_build 가 완료되면 덮어씀.
        # 값: {project_dir, files: {rel: content}, created_at, chain_id, stage_index, schema_ref}
        self._last_codebuild_result: Optional[Dict[str, Any]] = None
        # ── CustomerSupport 드래프트 큐 (P2-8 2026-04-19) ─────────────────
        # 신규 고객 문의 draft 가 쌓이는 1-based 접근 가능 FIFO.
        # 각 항목: {customer_id, channel, message, draft, draft_path, received_at}
        self._customer_drafts: List[Dict[str, Any]] = []
        # ── [Fix] WAITING_USER 자동 재개 역설 방지 플래그 ──────────────────
        # True이면 _tick의 이어하기 분기가 스킵됨. run_now()(버튼 재클릭) 호출 시 해제.
        self._wait_for_user: bool = False
        # ── [Fix] DriftGuard RED 재시작 카운터 (태스크별 상한) ─────────────
        self._drift_restarts: int = 0
        # ── [Fix] _tick 동시 실행 방지 — run_now()와 _dispatch_loop 동시 진입 차단 ──
        # 동일 브라우저 탭에서 두 ReAct 루프가 경쟁적으로 nav를 호출하면
        # PAGE_LOAD_BUFFER future 오염/핸들러 누적이 발생하므로 직렬화.
        self._tick_lock: Optional[asyncio.Lock] = None
        # [Fix 2026-05-15 orphan-chain] Fix D2 가 자기 _tick_lock 안에서 run_now 를
        # fire-and-forget 하면 _tick_guarded 의 locked 가드에 막혀 무력화되고,
        # Fix A v5 가 등록한 PENDING task 가 orphan 으로 방치됨. 이 플래그가 True 면
        # _tick_guarded 가 lock 해제 직전 _tick 을 1회 더 돌려 그 task 를 pickup 한다.
        self._tick_rerun_requested: bool = False
        # [Strangler v2 2026-05-15] research→chain 승격 누더기 경로
        # (Fix A/D1/D2/stale/재진입) 를 우회하는 깨끗한 실행기. lazy 초기화.
        self._chain_executor_v2 = None
        # ── [Chain Force-Advance] 사용자 "다음단계" 명령 플래그 (2026-04-19) ──
        # True 이면 _react_loop_impl 다음 iter 진입 시 현재 chain stage 를 강제로
        # success 처리하고 _on_react_done 호출 → 다음 stage 자동 승격.
        # Why: research stage 가 산출물 떨궜는데도 LLM 이 무한 NAVIGATE 돌릴 때
        # 사용자가 "이 단계 됐어, 넘어가" 라고 끊을 수 있어야 함.
        self._force_advance_chain: bool = False
        # ── [2026-05-23] 게이트 차단 글로벌 임계 (Synthesis Guard 무한 재시도 방지) ──
        # Why: derivation_report(실질검증) / self_asset_presence(자기자산) /
        #      샘플링 게이트가 LLM 자동 재시도 유발. 게이트별 cap 없어서 종류 바뀌면
        #      누적 13회까지 가서 756s 타임아웃 (2.png 사용자 보고). 4회 누적 시
        #      task PAUSE + 사용자 결정 대기. 카운터는 schema attr 로 schema 별 리셋.
        self._gate_block_limit: int = 4
        # ── [Chain Abort] 사용자 "그만/취소" 명령 플래그 (2026-04-19) ────────
        # True 이면 _react_loop_impl 다음 iter 진입 시 현재 chain task 를 FAILED
        # 처리하고 orchestrator.abort_chain() 호출 → 체인 전체 중단.
        self._abort_active_chain: bool = False
        # ── [P2 2026-04-21] Chain stage 전환 직후 첫 WRITE 는 Shortening Guard
        # 면제 (1회만). chain 이 바뀌었다 = 이전 stage 의 파일은 새 관점으로
        # 재생성될 수 있음. Guard 는 글자수만 비교하므로 stage 경계를 모른다.
        # 키: (chain_id, stage_index, var_name) — 본 세트에 등록되기 전 1회만 면제.
        self._chain_stage_first_write_seen: set = set()
        # ── [P4 2026-04-21] WRITE 재료 추적. 같은 파일에 대한 연속 WRITE 에서
        # 주입된 원천 변수 총량이 급감하면 "재료 소실(중간에 변수가 날아갔거나
        # 잘못된 var 가 덮어씀)"로 간주하고 LLM에 복구 지시.
        # 키: (chain_id, stage_index, var_name) → 직전 주입 총량(int).
        self._write_material_history: Dict[tuple, int] = {}
        # ── [Phase 2.2 2026-04-25] Worker idle 워치독 ─────────────────────────
        # Why: chain 강제 advance 후 다음 stage launch 가 누락되면 dispatcher 가
        # 무음 idle 로 빠져 사용자가 한참 뒤에야 알아차림 (1597줄 force_advance
        # 후 EIDOS 활동 0). 60s idle + 미완료 chain task 존재 시 Telegram 알림
        # + 1회 run_now() rehydrate 시도.
        self._last_activity_ts: float = time.monotonic()
        # 마지막 alert 시각 — 중복 알림 방지 (cooldown 300s).
        self._idle_alert_at: float = 0.0
        self._watchdog_task: Optional[asyncio.Task] = None
        # ── [P2-1 2026-04-25] Watchdog idle strike 누적 — chain 단위. 3회 도달 시
        # 자동 chain abort. _touch_activity 시 일괄 초기화.
        self._idle_strikes_per_chain: Dict[str, int] = {}

    # ── 공개 API ──────────────────────────────────────────────────────────────

    def set_report_callback(self, cb: Callable[[str], None]):
        """GUI 연결 후 채팅창 콜백 등록"""
        self._report_cb = cb
        # [2026-04-19] 재시작 직후 rehydrate 된 _pending_chain 이 있으면
        # 사용자가 잊지 않도록 콜백 등록 시점에 한 번만 reminder 카드 발송.
        try:
            if self._pending_chain is not None and not getattr(self, "_chain_reminder_sent", False):
                _c = self._pending_chain
                cb(
                    f"♻️ **[MissionChain 승인 대기 — 복구됨]** `{_c.name}`\n"
                    f"└─ 이전 세션에서 생성된 {len(_c.stages)}단계 체인이 아직 승인되지 않았습니다.\n"
                    f"└─ `승인` 으로 실행, `취소` 로 폐기."
                )
                self._chain_reminder_sent = True
        except Exception as _e_rem:
            print(f"  [Dispatcher] chain reminder 발송 실패 (무시): {_e_rem}")

    # ── [Phase 2.2 2026-04-25] Worker idle 워치독 헬퍼 ───────────────────────
    def _touch_activity(self) -> None:
        """dispatcher 의 살아있다는 신호 — _tick / run_now / _react_loop / _on_react_done
        진입 시 호출. 워치독은 이 타임스탬프를 기준으로 idle 여부를 판정."""
        self._last_activity_ts = time.monotonic()
        # [P2-1] 활동 재개 → idle strike 일괄 초기화. 다른 chain 에서도 동일 dispatcher
        # 가 일하고 있다는 신호이므로 안전하게 전부 reset.
        if self._idle_strikes_per_chain:
            self._idle_strikes_per_chain.clear()

    async def _chain_idle_watchdog(self) -> None:
        """체인 task 가 schedule 에 살아있는데 60s 이상 dispatcher 활동 0이면
        Telegram 1회 알림 + run_now() rehydrate 시도. 5분 cooldown.

        [Fix D3 2026-05-06] simple-task active 시 idle 임계 + poll 간격 단축.
        Why: 최신로그.txt 09:53~09:54 chain auto-approval 후 Watchdog 60s 임계
        도달까지 대기 → 그제서야 run_now() → 1분 데드타임. simple-task 는 빠른
        응답이 사용자 경험상 중요 → 임계 10s + poll 5s 로 단축.
        """
        IDLE_THRESHOLD_DEFAULT = 60.0
        IDLE_THRESHOLD_SIMPLE  = 10.0   # [Fix D3]
        ALERT_COOLDOWN = 300.0
        POLL_INTERVAL_DEFAULT  = 30.0
        POLL_INTERVAL_SIMPLE   = 5.0    # [Fix D3]
        while self._running:
            try:
                # [Fix D3] simple-task 가 active 면 짧은 poll/threshold 사용.
                _has_simple = False
                try:
                    for _t_d3 in (getattr(self.core, "schedule", []) or []):
                        if (_t_d3.get("status") in ("PENDING", "RUNNING", "WAITING_USER")
                                and bool(_t_d3.get("__simple_task__"))):
                            _has_simple = True
                            break
                except Exception:
                    pass
                _poll = POLL_INTERVAL_SIMPLE if _has_simple else POLL_INTERVAL_DEFAULT
                _thr  = IDLE_THRESHOLD_SIMPLE if _has_simple else IDLE_THRESHOLD_DEFAULT
                await asyncio.sleep(_poll)
                if not self._running:
                    return
                _now = time.monotonic()
                _idle = _now - self._last_activity_ts
                if _idle < _thr:
                    continue
                # 미완료 chain task 가 schedule 에 있는지 확인
                _ALIVE = ("PENDING", "RUNNING", "WAITING_USER")
                _chain_tasks = []
                try:
                    for _t in (getattr(self.core, "schedule", []) or []):
                        if (_t.get("status") in _ALIVE
                                and _t.get("chain_id")
                                and not _t.get("_stale_session")):
                            _chain_tasks.append(_t)
                except Exception:
                    pass
                if not _chain_tasks:
                    continue
                if (_now - self._idle_alert_at) < ALERT_COOLDOWN:
                    continue
                self._idle_alert_at = _now
                _t0 = _chain_tasks[0]
                _cid_full = _t0.get("chain_id") or ""
                _cid   = _cid_full[:8]
                _stage = _t0.get("stage", "?")
                _stat  = _t0.get("status", "?")

                # [P2-1 2026-04-25] idle strike 누적 + 3회 도달 시 자동 chain abort.
                # Why: 기존 watchdog 은 5분마다 같은 알림 + run_now 만 반복. rehydrate
                # 가 한 번도 안 풀린 채로 chain 이 무한 idle 빠지는 패턴(register stage
                # 228s idle 후 무한 대기) 차단. 사용자가 원하면 다시 chain 을 시작하면 됨.
                IDLE_STRIKE_HARD_LIMIT = 3
                _strike = int(self._idle_strikes_per_chain.get(_cid_full, 0) or 0) + 1
                self._idle_strikes_per_chain[_cid_full] = _strike

                if _strike >= IDLE_STRIKE_HARD_LIMIT:
                    _abort_msg = (
                        f"🔴 **[Watchdog 한계 초과]** chain={_cid} stage='{_stage}' "
                        f"idle 알림 {_strike}/{IDLE_STRIKE_HARD_LIMIT}회 누적 ({int(_idle)}s) — "
                        f"자동 abort. 필요하면 사용자가 다시 시작하세요."
                    )
                    print(f"  🔴 [P2-1] {_abort_msg}")
                    try:
                        self._report(_abort_msg)
                    except Exception:
                        pass
                    # chain abort
                    try:
                        from eidos_mission_chain import get_orchestrator as _go_p21
                        _go_p21().abort_chain(
                            _cid_full,
                            reason=(
                                f"watchdog_idle_unrecoverable: dispatcher {int(_idle)}s "
                                f"무활동 × {_strike}회 누적"
                            ),
                        )
                    except Exception as _e_ab_p21:
                        print(f"  [P2-1] abort_chain 실패 (무시): {_e_ab_p21}")
                    # task 도 FAILED 처리 + schedule 제거
                    try:
                        _t0["status"] = "FAILED"
                        self._remove_task_from_schedule(_t0)
                    except Exception as _e_rm_p21:
                        print(f"  [P2-1] task 정리 실패 (무시): {_e_rm_p21}")
                    # Telegram 알림
                    try:
                        from eidos_telegram_bot import get_bot as _gb_p21
                        _bot_p21 = _gb_p21()
                        if _bot_p21.is_configured():
                            await _bot_p21.send_notification(
                                f"chain=<code>{_cid}</code> stage=<code>{_stage}</code>이 "
                                f"dispatcher 무활동 {_strike}회({int(_idle)}s)로 자동 "
                                f"abort 되었습니다. 사용자 개입 필요.",
                                title="🔴 Watchdog Idle Abort (P2-1)",
                            )
                    except Exception:
                        pass
                    self._idle_strikes_per_chain.pop(_cid_full, None)
                    continue

                _msg = (
                    f"⚠️ **[Worker Idle 감지 {_strike}/{IDLE_STRIKE_HARD_LIMIT}]** "
                    f"dispatcher {_idle:.0f}s 무활동\n"
                    f"├─ chain={_cid} stage='{_stage}' status='{_stat}'\n"
                    f"└─ rehydrate 시도 (run_now) — 한도 도달 시 자동 abort"
                )
                print(f"  ⚠️ [Watchdog] {_msg}")
                try:
                    self._report(_msg)
                except Exception:
                    pass
                # Telegram 1회 알림 (best-effort)
                try:
                    from eidos_telegram_bot import get_bot as _get_bot
                    _bot = _get_bot()
                    if _bot.is_configured():
                        await _bot.send_notification(
                            f"dispatcher {int(_idle)}s 무활동.\n"
                            f"chain=<code>{_cid}</code> stage=<code>{_stage}</code> "
                            f"status=<code>{_stat}</code>\n"
                            f"rehydrate 자동 시도 중. 5분 내 추가 알림 없음.",
                            title="⚠️ Worker Idle",
                        )
                except Exception as _e_tg:
                    print(f"  ⚠️ [Watchdog] Telegram 발송 실패 (무시): {_e_tg}")
                # [Phase 2.2 보강 2026-04-25] NAVIGATE pending 강제 resolve.
                # Why: Step 10 https://kmong.com/gig/170444 처럼 about:blank 브릿지
                # 후 loadFinished 시그널이 안 와서 navigate_and_wait 가 30s 까지
                # 대기. _NAV_LOCK 도 못 풀고 dispatcher 전체가 stuck. 워치독이
                # PAGE_LOAD_BUFFER 의 pending future 를 ok=False/final_url=""
                # 로 강제 resolve → navigate_and_wait 가 "하드 실패" 리턴 →
                # _react_loop 가 다음 step 에서 LLM 에 에러 전달.
                try:
                    from execution_module import PAGE_LOAD_BUFFER as _PLB
                    _futs = dict(_PLB.get("_futures", {}) or {})
                    if _futs:
                        print(
                            f"  🔧 [Watchdog] NAVIGATE pending {len(_futs)}건 발견 — "
                            f"force-resolve 시도"
                        )
                        for _tok, _fut in _futs.items():
                            try:
                                if _fut.done():
                                    continue
                                _fl = None
                                try:
                                    _fl = _fut.get_loop()
                                except Exception:
                                    _fl = None
                                _payload = {"ok": False, "final_url": ""}
                                if _fl is not None and _fl is not asyncio.get_event_loop():
                                    _fl.call_soon_threadsafe(_fut.set_result, _payload)
                                else:
                                    _fut.set_result(_payload)
                                print(f"  🔧 [Watchdog] navigate future force-resolved: token={_tok}")
                            except Exception as _e_fr:
                                print(f"  ⚠️ [Watchdog] future resolve 실패 (token={_tok}): {_e_fr}")
                except Exception as _e_pb:
                    print(f"  ⚠️ [Watchdog] PAGE_LOAD_BUFFER 접근 실패: {_e_pb}")
                # rehydrate 시도 (NAVIGATE force-resolve 후 짧게 대기 — _NAV_LOCK
                # 해제 + _react_loop 가 다음 step 으로 진행할 시간 확보)
                try:
                    await asyncio.sleep(0.5)
                    asyncio.ensure_future(self.run_now(""))
                except Exception as _e_rh:
                    print(f"  ⚠️ [Watchdog] run_now rehydrate 실패: {_e_rh}")
            except asyncio.CancelledError:
                return
            except Exception as _e_wd:
                print(f"  ⚠️ [Watchdog] 루프 예외 (계속 진행): {_e_wd}")

    def start(self, loop: Optional["asyncio.AbstractEventLoop"] = None):
        if self._running:
            return
        self._running = True
        # [Phase 2.2 2026-04-25] 첫 활동 시각 초기화 — 시작 직후 워치독 오작동 방지.
        self._last_activity_ts = time.monotonic()

        # [Fix] 앱 재시작 시 이전 세션의 작업이 사용자 동의 없이 자동 실행돼
        # "뜬금없는 작업 Y가 실행"되던 버그 차단. WAITING_USER/PENDING/RUNNING 모두
        # stale로 마킹해 _pick_pending_task의 선택 대상에서 제외한다.
        # (WAITING_USER만 가드했더니 이전 세션의 dispatcher PENDING 큐 — 예: 크몽
        #  매출 올리기 — 가 네가 새로 지시한 나무위키 등 non-kmong 작업보다 먼저
        #  pick 되어 엉뚱한 사이트로 NAVIGATE 되던 사례 확인됨.)
        # 이번 세션에서 새로 들어오는 PENDING(=사용자 재지시/신규 태스크)은 이후
        # append 되므로 stale 마커가 붙지 않는다.
        try:
            _STALE_STATUSES = ("WAITING_USER", "PENDING", "RUNNING")
            for _t in getattr(self.core, "schedule", []) or []:
                if _t.get("status") in _STALE_STATUSES:
                    _t["_stale_session"] = True
        except Exception as _e_mark:
            print(f"  [Dispatcher] stale 세션 마킹 실패 (무시): {_e_mark}")

        # ── [2026-05-15 P5] Layer F approved goal catch-up dispatch ──────────
        # EmergedGoalsDialog 에서 ✅ 승인된 emerged goal 은 즉시 schedule 에 주입되지만,
        # (a) 앱이 종료된 상태에서 데이터만 영속됐거나 (b) 즉시 주입이 graceful 실패
        # 한 경우 — approved_list 에는 있지만 dispatched_to_schedule_ts 마커 없는 row
        # 가 잔존. start() 시점에 catch-up 으로 일괄 주입.
        # stale 마킹 이후 (line 1789) append 되므로 새 task 로 인식됨 (auto-run 가능).
        # [P4 2026-05-17 재설계] 신 substrate 창발 catch-up (kill-switch 기본
        # OFF — loop_closure.verb_substrate_enabled 로 사용자 수동 ON).
        try:
            import eidos_verb_goal_engine as _vge
            _vge.tick_emergence_to_schedule(self.core)
        except Exception as _e_vge:
            print(f"  [Dispatcher] substrate 창발 catch-up 실패 (graceful): {_e_vge}")
        try:
            from eidos_verb_substrate import legacy_ngram_enabled as _leg_fn
            _legacy_on = bool(_leg_fn())
        except Exception:
            _legacy_on = False
        try:
            from eidos_verb_goal_emergence import (
                auto_approve_high_confidence, dispatch_approved_to_schedule,
            )
            # [자동승인 2026-05-15] dispatch 직전 ≥0.75 후보 일괄 auto-approve.
            # [P4] 구 n-gram 경로 = legacy kill-switch (기본 OFF). _legacy_on
            # False 면 두 호출 모두 무동작값으로 단락 — 신 경로가 대체.
            _auto = auto_approve_high_confidence() if _legacy_on else []
            if _auto:
                print(
                    f"  🌱 [Layer F] 시작 시 자동승인 {len(_auto)}건 "
                    f"(≥0.75 confidence — HITL 생략)"
                )
            _n_dispatched = dispatch_approved_to_schedule(self.core)
            if _n_dispatched > 0:
                print(
                    f"  🌱 [Layer F] catch-up dispatch — approved goal "
                    f"{_n_dispatched} 건을 schedule 에 PENDING 주입"
                )
        except Exception as _e_lf:
            print(f"  [Dispatcher] Layer F catch-up 실패 (graceful): {_e_lf}")

        # ── [2026-04-19] PENDING_APPROVAL 체인 in-memory 복구 ───────────────
        # _pending_chain 은 in-memory 라 앱 재시작 시 사라지는데, registry 에는
        # PENDING_APPROVAL 체인이 살아 있어 사용자가 `승인` 을 입력해도 메인
        # 채팅의 _try_consume_dispatcher_pending 가 빈 슬롯을 보고 LLM 으로
        # 흘려보내 영구 stuck 되는 문제. 가장 최근 1건만 자동 rehydrate.
        try:
            if self._pending_chain is None:
                from eidos_mission_chain import MissionChainRegistry, ChainStatus
                _reg = MissionChainRegistry.instance()
                _pending = [
                    c for c in _reg.all()
                    if c.status == ChainStatus.PENDING_APPROVAL
                ]
                if _pending:
                    _pending.sort(key=lambda c: getattr(c, "created_at", 0.0), reverse=True)
                    self._pending_chain = _pending[0]
                    print(
                        f"  ♻️ [MissionChain] PENDING_APPROVAL 체인 복구: "
                        f"'{self._pending_chain.name}' "
                        f"({len(self._pending_chain.stages)}단계)"
                    )
        except Exception as _e_re:
            print(f"  [Dispatcher] chain rehydrate 실패 (무시): {_e_re}")

        # 1) 호출자가 루프를 명시적으로 넘겨줬으면 그걸 사용
        # 2) 아니면 현재 실행 중인 루프(있을 경우)
        # 3) 마지막 폴백: 워커 스레드에서 호출된 경우 — 메인 스레드의 루프를 찾아
        #    run_coroutine_threadsafe로 안전하게 스케줄링
        target_loop = loop
        if target_loop is None:
            try:
                target_loop = asyncio.get_running_loop()
            except RuntimeError:
                target_loop = None

        if target_loop is not None:
            # 현재 스레드에 루프가 있는 정상 경로
            self._loop_task = target_loop.create_task(self._dispatch_loop())
            # [Phase 2.2 2026-04-25] idle 워치독 동시 spawn (best-effort)
            try:
                self._watchdog_task = target_loop.create_task(self._chain_idle_watchdog())
            except Exception as _e_wd:
                print(f"  ⚠️ [Watchdog] spawn 실패 (무시): {_e_wd}")
        else:
            # 워커 스레드(asyncio_0 등)에서 호출된 경로
            # 메인 스레드에서 돌고 있는 루프를 찾아 거기로 스케줄링
            policy_loop = None
            try:
                policy_loop = asyncio.get_event_loop_policy().get_event_loop()
            except Exception:
                policy_loop = None

            if policy_loop is not None and policy_loop.is_running():
                # run_coroutine_threadsafe는 concurrent.futures.Future를 돌려주므로
                # self._loop_task에 asyncio.Task 형태로 저장하지 않음 (stop()에서 분기 처리됨)
                self._loop_task = asyncio.run_coroutine_threadsafe(
                    self._dispatch_loop(), policy_loop
                )
                try:
                    self._watchdog_task = asyncio.run_coroutine_threadsafe(
                        self._chain_idle_watchdog(), policy_loop
                    )
                except Exception as _e_wd:
                    print(f"  ⚠️ [Watchdog] spawn 실패 (무시): {_e_wd}")
            else:
                # 정말 루프가 없다면 현재 스레드에 새 루프를 만들고 거기에 등록
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                self._loop_task = new_loop.create_task(self._dispatch_loop())
                try:
                    self._watchdog_task = new_loop.create_task(self._chain_idle_watchdog())
                except Exception as _e_wd:
                    print(f"  ⚠️ [Watchdog] spawn 실패 (무시): {_e_wd}")

        print("🚀 [ActionDispatcher] 자율 실행 루프 시작")

    def stop(self):
        self._running = False
        if self._loop_task:
            try:
                self._loop_task.cancel()
            except Exception:
                pass
        if self._watchdog_task:
            try:
                self._watchdog_task.cancel()
            except Exception:
                pass
            self._watchdog_task = None
        print("⛔ [ActionDispatcher] 자율 실행 루프 정지")

    async def run_now(self, user_reply: str = ""):
        """즉시 1회 실행 (수동 트리거).

        [Fix 2026-04-20]
        - user_reply 가 들어오면 ask_user pause_reason 과 질문 기록을 해제.
        - user_reply 없는 버튼 클릭도 resume 은 허용하되, ASK_USER 카운터는
          유지해 동일 질문 반복 루프는 ReAct 쪽에서 하드스탑.

        [Fix 2026-04-24 chain resume loop]
        - 사용자가 빈 reply 로 chain stage 재시도 — 같은 schema 로 _react_loop 가
          retry_count=0 부터 다시 도는 무한 루프 차단. _paused_task 에 누적 카운트
          저장, 2회 도달 시 _force_advance_chain=True 로 자동 다음 stage 진행.
        """
        # [Phase 2.2] 활성 신호
        self._touch_activity()
        _pt = self._paused_task
        _pr = (_pt or {}).get("__pause_reason__", "") if _pt else ""
        _has_reply = bool((user_reply or "").strip())
        if _has_reply and _pt and _pr == "ask_user":
            _pt["user_reply"] = user_reply
            _pt.pop("__pause_reason__", None)
            _pt.pop("__ask_user_question__", None)
            _pt["__ask_user_count__"] = 0  # 사용자가 실제 답변 → 카운터 리셋

        # 빈 reply + chain stage 재진입 — 누적 카운트
        if not _has_reply and _pt and _pt.get("chain_id"):
            _cnt = int(_pt.get("__chain_resume_count__", 0)) + 1
            _pt["__chain_resume_count__"] = _cnt
            if _cnt >= 2:
                _stage = _pt.get("stage", "?")
                _cid = (_pt.get("chain_id") or "")[:8]
                print(
                    f"  ⏭️ [run_now] chain stage='{_stage}' (chain={_cid}) 같은 task "
                    f"빈응답 재시도 {_cnt}회 → 강제 chain advance"
                )
                self._report(
                    f"⏭️ **[빈응답 재시도 {_cnt}회]** chain stage='{_stage}' — "
                    f"같은 단계 무한 루프 방지 위해 다음 단계로 강제 진행."
                )
                self._force_advance_chain = True

        self._wait_for_user = False   # 버튼 재클릭 = 사용자 응답 시그널
        await self._tick_guarded()

    # ── 내부 루프 ─────────────────────────────────────────────────────────────

    async def _dispatch_loop(self):
        while self._running:
            try:
                await self._tick_guarded()
            except Exception as e:
                import traceback
                print(f"❌ [ActionDispatcher] 루프 오류: {e}")
                traceback.print_exc()
            await asyncio.sleep(self.LOOP_INTERVAL)

    async def _tick_guarded(self):
        """_tick의 동시 진입 방지 래퍼. lock이 잠겨 있으면 이번 tick 스킵."""
        if self._tick_lock is None:
            self._tick_lock = asyncio.Lock()
        if self._tick_lock.locked():
            # [Fix 2026-05-15 orphan-chain] 조용히 버리면 Fix D2 의 fire-and-forget
            # run_now 가 무력화돼 orphan chain 발생. 현재 tick 종료 후 1회
            # 재실행을 요청만 남기고 스킵 (루프/버튼 경합 방지는 그대로 유지).
            self._tick_rerun_requested = True
            return
        async with self._tick_lock:
            await self._tick()
            # lock 해제 직전: tick 진행 중 들어온 재진입 요청 처리.
            # _rerun_guard 로 최대 3회 — 무한 재진입 차단.
            _rerun_guard = 0
            while self._tick_rerun_requested and _rerun_guard < 3:
                self._tick_rerun_requested = False
                _rerun_guard += 1
                await self._tick()

    async def _tick(self):
        """PENDING 작업 1개를 꺼내 실행. 일시정지된 plan이 있으면 이어서 실행."""
        # [Phase 2.2] 활성 신호 — _tick 진입 자체가 dispatcher 살아있음 증거
        self._touch_activity()

        # ── [Phase A] Kill Switch 조기 체크 — 초과 예산/관리자 정지 상태면 전체 중단 ──
        try:
            from eidos_approval_gateway import kill_switch_active as _ks_active, gateway_status as _gw_status
            if _ks_active():
                _st = _gw_status()
                _reason = _st.get("kill_reason", "")
                if self._running:
                    print(f"⛔ [Dispatcher] ApprovalGateway kill-switch 활성 — tick 중단 (reason: {_reason})")
                    self._running = False
                return
        except ImportError:
            pass
        except Exception as _ks_e:
            print(f"  [Dispatcher] kill-switch 체크 예외 (무시): {_ks_e}")

        # ── [이어하기] WAITING_USER 후 재개 ─────────────────────────────────
        if self._paused_task:
            # [Fix] 사용자 응답이 필요한 일시정지면 자동 재개 스킵
            # — run_now()(버튼 재클릭)가 호출되어 _wait_for_user=False가 되기 전까진 대기
            if self._wait_for_user:
                return
            task = self._paused_task

            # [P2-2026-04-19] auth_required pause 재개 전 로그인 상태 재검증.
            # 유저가 실제 로그인 없이 ⚡ 자율 실행 연타하는 루프 차단.
            _pr = task.get("__pause_reason__", "")
            if _pr == "auth_required":
                try:
                    _recheck_obs = await self._observe()
                    if _recheck_obs.get("auth_required"):
                        _fails = int(task.get("__auth_fail_count__", 0)) + 1
                        task["__auth_fail_count__"] = _fails
                        _cur_url = (_recheck_obs.get("url") or "")[:80]
                        if _fails >= 3:
                            self._report(
                                f"🔴 **[로그인 반복 실패 {_fails}회 — 자동 중단]**\n"
                                f"└─ 현재 URL: {_cur_url}\n"
                                f"└─ 브라우저에서 실제 로그인을 완료한 뒤 작업을 다시 입력해주세요."
                            )
                            await self._notify_telegram_auth(
                                reason="resume 재검증 — 여전히 비로그인 (3회 초과)",
                                url=_cur_url, fails=_fails,
                            )
                            task["status"] = "FAILED_AUTH"
                            self._remove_task_from_schedule(task)
                            self._paused_task = None
                            self._wait_for_user = False
                            return
                        self._report(
                            f"[ASK_USER] 🔐 **[여전히 비로그인 {_fails}/3]** — "
                            f"브라우저에서 로그인을 완료한 뒤 ⚡ 자율 실행을 눌러주세요.\n"
                            f"└─ 현재 URL: {_cur_url}"
                        )
                        await self._notify_telegram_auth(
                            reason="resume 재검증 — 여전히 비로그인",
                            url=_cur_url, fails=_fails,
                        )
                        # paused 상태 유지 (다음 버튼 클릭까지 대기)
                        self._wait_for_user = True
                        return
                    # 로그인 확인됨 → pause_reason 해제 + counter 리셋.
                    # [P1-2026-04-19] 블랙리스트/로그인 ASK 가드도 함께 리셋.
                    # - __auth_blacklist__: 이전 비로그인 상태에서 쌓인 경로는 stale.
                    #   그대로 두면 Tier 2가 영구 스킵돼 LLM이 ASK_USER 반복만 뽑음.
                    # - __auth_pass_grace__: 재개 직후 N스텝 동안 "로그인해라" 류
                    #   ASK_USER 를 자동 무시하고 탐색 액션으로 전환하는 grace window.
                    task.pop("__pause_reason__", None)
                    task["__auth_fail_count__"] = 0
                    _bl_prev = task.get("__auth_blacklist__") or []
                    if _bl_prev:
                        print(
                            f"  🧹 [AuthRecheck] stale auth_blacklist clear "
                            f"({len(_bl_prev)}개)"
                        )
                    task["__auth_blacklist__"] = []
                    # [Phase 4-3 2026-04-25] unhydrated streak 도 리셋 — 새 시도이므로
                    # 이전 retry 의 누적 카운트가 grace override 를 즉시 발동시키지 않도록.
                    task.pop("__unhydrated_streak__", None)
                    # [P1 2026-04-22] grace 3→5: NAVIGATE 1회당 observe 가 여러 번
                    # 호출되어 grace 가 빨리 소진됨. Layer 1 가드까지 적용되면 더
                    # 빨리 닳음. 진짜 로그아웃 케이스도 5스텝 안엔 결국 다시 잡힘.
                    task["__auth_pass_grace__"] = 5
                    print("  ✅ [AuthRecheck] 로그인 확인 — resume 진행 (grace=5)")
                except Exception as _e_chk:
                    print(f"  [AuthRecheck] 재검증 실패 (무시, resume 진행): {_e_chk}")

            # [Phase B 2026-04-24] CAPTCHA 재검증 — AuthRecheck 패턴 미러.
            # Why: 사용자가 ⚡ 자율 실행 / Telegram [✅ 풀었음] 누르면 captcha_required
            # 그대로인 채로도 무한 재시도. _observe() 의 captcha_required 가 사라졌는지
            # 한 번 더 본 뒤 통과/거부 결정. 3회 실패 → FAILED_CAPTCHA + chain stage 실패.
            elif _pr == "captcha_required":
                try:
                    _recheck_obs = await self._observe()
                    if _recheck_obs.get("captcha_required"):
                        _fails = int(task.get("__captcha_fail_count__", 0)) + 1
                        task["__captcha_fail_count__"] = _fails
                        _cur_url = (_recheck_obs.get("url") or "")[:80]
                        _sig = _recheck_obs.get("captcha_signal", "") or ""
                        if _fails >= 3:
                            self._report(
                                f"🔴 **[캡챠 반복 미해결 {_fails}회 — 자동 중단]**\n"
                                f"└─ 현재 URL: {_cur_url}\n"
                                f"└─ 시그널: {_sig}\n"
                                f"└─ 브라우저에서 캡챠를 직접 풀고 작업을 다시 입력해주세요."
                            )
                            try:
                                await self._notify_telegram_captcha(
                                    question=f"resume 재검증 — 여전히 캡챠 ({_fails}회 초과)",
                                    url=_cur_url,
                                )
                            except Exception:
                                pass
                            task["status"] = "FAILED_CAPTCHA"
                            # chain task 라면 abort_chain 으로 체인 전체 중단 — 무한 재진입 방지.
                            try:
                                _cid = task.get("chain_id")
                                if _cid:
                                    from eidos_mission_chain import get_orchestrator
                                    _orch = get_orchestrator()
                                    if _orch:
                                        _orch.abort_chain(
                                            _cid, reason=f"CAPTCHA 미해결 3회 ({_sig})"
                                        )
                            except Exception as _ce:
                                print(f"  [CaptchaRecheck] chain abort 실패 (무시): {_ce}")
                            self._remove_task_from_schedule(task)
                            self._paused_task = None
                            self._wait_for_user = False
                            return
                        self._report(
                            f"[ASK_USER] 🧩 **[여전히 캡챠 {_fails}/3]** — "
                            f"브라우저에서 캡챠를 풀고 다시 ⚡ 자율 실행을 눌러주세요.\n"
                            f"└─ 현재 URL: {_cur_url}\n"
                            f"└─ 시그널: {_sig}"
                        )
                        try:
                            await self._notify_telegram_captcha(
                                question=f"resume 재검증 — 여전히 캡챠 ({_fails}/3)",
                                url=_cur_url,
                            )
                        except Exception:
                            pass
                        # paused 상태 유지
                        self._wait_for_user = True
                        return
                    # 캡챠 사라짐 → pause_reason 해제 + counter 리셋
                    task.pop("__pause_reason__", None)
                    task["__captcha_fail_count__"] = 0
                    print("  ✅ [CaptchaRecheck] 캡챠 해제 확인 — resume 진행")
                    # [P1-3 2026-04-25] 캡챠 해결 host 를 chain.metadata 에 기록.
                    # 이후 동일 host 에서 captcha 재감지 시 즉시 pivot — chain=db5ebe78
                    # 에서 캡챠 풀고도 같은 도메인 재차단으로 헛돌이한 패턴 차단.
                    try:
                        from urllib.parse import urlparse as _up_cs
                        _solved_host = (_up_cs(_recheck_obs.get("url", "") or "").netloc or "").lower()
                        _cid_cs = (task.get("chain_id") or "").strip()
                        if _solved_host and _cid_cs:
                            from eidos_mission_chain import get_orchestrator as _go_cs
                            _ch_cs = _go_cs().registry.get(_cid_cs)
                            if _ch_cs is not None:
                                _solved_set = _ch_cs.metadata.setdefault("captcha_solved_hosts", [])
                                if _solved_host not in _solved_set:
                                    _solved_set.append(_solved_host)
                                    print(
                                        f"  📌 [CaptchaSolved] chain.metadata "
                                        f"captcha_solved_hosts += '{_solved_host}'"
                                    )
                    except Exception as _e_cs_meta:
                        print(f"  [CaptchaSolved] metadata 기록 실패 (무시): {_e_cs_meta}")
                except Exception as _e_cap:
                    print(f"  [CaptchaRecheck] 재검증 실패 (무시, resume 진행): {_e_cap}")

            self._paused_task = None

            # 레거시 plan 기반 이어하기
            if self._paused_plan:
                plan = self._paused_plan
                self._paused_plan = None
                task_prompt = task.get("task_prompt", "")
                # [Self-Agency 2026-05-15] emergent goal 재개 시에도 사용자
                # top_goal 치환 차단 (메인 경로와 동일 정책 — 일관성).
                if task.get("emergence_goal_id") \
                        or (task_prompt or "").lstrip().startswith("[자기조직화 목표]"):
                    goal_ref = ("[EIDOS 자기주도 목표 — 주체·수혜자 모두 EIDOS "
                                "자신. 사용자 목표 아님, 사용자 보조 아님]")
                else:
                    goal_ref = self._filter_goal(task.get("goal_ref", ""), task_prompt) \
                               or self._get_relevant_goal(task_prompt)
                remaining   = [s for s in plan.steps if s.status == "PENDING"]
                done_so_far = sum(1 for s in plan.steps if s.status == "DONE")
                total       = len(plan.steps)
                self._report(
                    f"▶️ **[이어하기]** `{task_prompt[:60]}`\n"
                    f"├─ 남은 단계: {len(remaining)}/{total}\n"
                    f"└─ 완료된 단계: {done_so_far}개 - 이어서 진행합니다."
                )
                await self._execute_remaining(plan, task, remaining, task_prompt, goal_ref, total)
                return

            # ReAct 루프 이어하기
            task_prompt = task.get("task_prompt", "")
            # [Self-Agency 2026-05-15] emergent goal 이어하기에도 동일 정책.
            if task.get("emergence_goal_id") \
                    or (task_prompt or "").lstrip().startswith("[자기조직화 목표]"):
                goal_ref = ("[EIDOS 자기주도 목표 — 주체·수혜자 모두 EIDOS 자신. "
                            "사용자 목표 아님, 사용자 보조 아님]")
            else:
                goal_ref = self._filter_goal(task.get("goal_ref", ""), task_prompt) \
                           or self._get_relevant_goal(task_prompt)
            task["status"] = "RUNNING"
            self._report(f"▶️ **[ReAct 이어하기]** `{task_prompt[:60]}`")

            # schema 승인 직후 재진입이면 _goal_linked 처리 (첫 진입 시 미처리된 경우 대비)
            if self._active_schema and getattr(self._active_schema, "_goal_linked", False) is False:
                try:
                    from eidos_goal_bridge import apply_schema_to_goal
                    apply_schema_to_goal(self.core, self._active_schema, goal_ref)
                    self._active_schema._goal_linked = True
                except Exception as _e_gb:
                    print(f"  ⚠️ [GoalBridge] apply_schema_to_goal 실패 (무시): {_e_gb}")

            # [4-B단계] 이어하기 분기 — CausalMatrix Y변수 초기화 (누락 방지)
            # [Fix] 수익/판매 도메인이 아닌 태스크(browser_read/general/write/report)는
            # CausalMatrix X변수(경쟁사_리뷰수 등 크몽 지표)가 무의미 — 스킵해 오염 방지.
            _cm_task_type = getattr(self._active_schema, "task_type", "general") if self._active_schema else "general"
            _CM_SKIP_TYPES = ("browser_read", "general", "write", "report")
            if self._active_schema and not getattr(self._active_schema, "_causal_linked", False) \
                    and _cm_task_type not in _CM_SKIP_TYPES:
                try:
                    from eidos_causal_matrix import trainer_from_schema, MATRIX_CACHE
                    import os as _os
                    existing = getattr(self.core, "_causal_trainer", None)
                    x_vars = getattr(self.core, "_causal_x_vars", [])
                    if x_vars or existing is not None:
                        trainer = trainer_from_schema(
                            self._active_schema, x_vars, existing_trainer=existing
                        )
                        self.core._causal_trainer = trainer
                        if existing is None and _os.path.exists(MATRIX_CACHE):
                            try:
                                from eidos_causal_matrix import CausalMatrixTrainer
                                loaded = CausalMatrixTrainer.load(MATRIX_CACHE)
                                # 로드된 캐시의 Y변수 이름과 현재 trainer의 Y변수 이름이
                                # 동일할 때만 pairs 복원 (의미 슬롯 오염 방지)
                                cur_y = {v.name for v in trainer.y_vars}
                                old_y = {v.name for v in loaded.y_vars}
                                if cur_y == old_y and len(loaded.x_vars) == len(trainer.x_vars):
                                    trainer._pairs_ext = loaded._pairs_ext
                                    trainer._pairs_int = loaded._pairs_int
                                else:
                                    print(
                                        f"  [CausalMatrix] 캐시 Y/X 불일치 → pairs 복원 스킵 "
                                        f"(cache_Y={sorted(old_y)}, "
                                        f"cur_Y={sorted(cur_y)})"
                                    )
                            except Exception:
                                pass
                        self._active_schema._causal_linked = True
                        print(f"  [CausalMatrix] Y변수 {len(self._active_schema.attributes)}개 "
                              f"HOW_MUCH로 초기화 (이어하기)")
                except Exception as _e_cm:
                    print(f"  ⚠️ [CausalMatrix] trainer 초기화 실패 (무시): {_e_cm}")

            await self._react_loop(task_prompt, goal_ref, task)
            return
        # ─────────────────────────────────────────────────────────────────────

        task = self._pick_pending_task()
        if not task:
            return

        task["status"] = "RUNNING"
        task_prompt = task.get("task_prompt", "")
        # [Self-Agency 2026-05-15] emergent goal 은 사용자 목표가 아니다.
        # _filter_goal/_get_relevant_goal 이 emergence signature 를 사용자
        # top_goal('크몽 …')로 치환 → situation 의 '최상위 목표: …' 로 주입
        # → 보고서가 "사용자의 ~를 위해"로 역전. emergent task 는 goal_ref
        # 를 EIDOS 1인칭 self-agency 표지로 고정한다. (비-emergent 는 기존
        # 로직 그대로 → 회귀 0.)
        if task.get("emergence_goal_id") \
                or (task_prompt or "").lstrip().startswith("[자기조직화 목표]"):
            goal_ref = ("[EIDOS 자기주도 목표 — 주체·수혜자 모두 EIDOS 자신. "
                        "사용자 목표 아님, 사용자 보조 아님]")
        else:
            goal_ref = self._filter_goal(task.get("goal_ref", ""), task_prompt) \
                       or self._get_relevant_goal(task_prompt)

        self._report(
            f"📋 **[작업 시작]** `{task_prompt[:60]}`\n"
            f"└─ 목표: {goal_ref[:50] if goal_ref else '(애드혹 요청 — 장기 목표 연결 없음)'}\n"
            f"└─ ReAct 루프 시작..."
        )

        # [Strangler v2 2026-05-15] research 의도면 깨끗한 chain executor v2 로
        # 통째 위임. request_schema_approval / Fix A·D1·D2 / stale 폐기 /
        # _tick 재진입 누더기 4단 콤보를 전부 우회한다. v2 가 True 면 책임
        # 완수(_react_loop 까지 진입)이므로 즉시 종료. False/예외면 아무 상태도
        # 안 바꾼 채 기존 경로로 그대로 폴백(안전망).
        # [Strangler v2 emergent-exempt 2026-05-15] Layer F 자기조직화 목표는
        # "검색→보고서"가 아니라 행동 목표 — research 파이프라인 자체가 카테고리
        # 오류다. emergence-origin task(emergence_goal_id 필드 / '[자기조직화
        # 목표]' prefix)를 research 로 오판해 v2 위임하면 raw 목표 텍스트
        # ('[자기조직화 목표] 공부/다음를 배우고 계속하다 …')를 그대로 네이버
        # 검색 → "결과 없음". 이런 task 는 v2 위임에서 제외하고 기존 자율
        # ReAct(컨텍스트 추론) 경로로 보낸다.
        _is_emergent_goal = bool(task.get("emergence_goal_id")) or \
            (task_prompt or "").lstrip().startswith("[자기조직화 목표]")
        try:
            if task_prompt and not _is_emergent_goal \
                    and await self._is_research_intent_async(task_prompt):
                if self._chain_executor_v2 is None:
                    from eidos_chain_executor_v2 import ChainExecutorV2
                    self._chain_executor_v2 = ChainExecutorV2(self)
                _v2_ok = await self._chain_executor_v2.run_research_chain(
                    task_prompt, goal_ref, task
                )
                if _v2_ok:
                    return
                print("  [Strangler v2] 위임 실패 — 기존 research 경로로 폴백")
        except Exception as _e_v2_seam:
            print(f"  [Strangler v2] seam 예외 — 기존 경로 폴백: {_e_v2_seam}")

        # [Fix 2026-04-19 B] Stale _active_schema 폐기 — 이전 task 의 schema 가
        # 거부/방치되어 남아있는 상태에서 새 task 가 그 schema 를 재활용하던 누수 차단.
        # 예: plan stage HOW_MUCH Y 5개(상품명 후보/타겟 섹션/USP/가격 근거/컨셉 개요)가
        # 전혀 다른 애드혹 task(크몽 CRM TOP5)의 Verifier missing_attr 실패 원인이 됐음.
        # 매칭 규칙: (1) 둘 다 chain_id 존재 + 일치 → 정상 체인 진행 → 유지
        #           (2) 그 외 prompt 불일치 → stale → 폐기
        if self._active_schema is not None:
            _sch_prompt = (
                getattr(self._active_schema, "original_prompt", "")
                or getattr(self._active_schema, "prompt", "")
                or getattr(self._active_schema, "what", "")
                or ""
            ).strip()
            _tp_norm = (task_prompt or "").strip()
            _tsk_chain = (task.get("chain_id", "") or "").strip()
            _sch_chain = (getattr(self._active_schema, "chain_id", "") or "").strip()
            _chain_match = bool(_tsk_chain and _sch_chain and _tsk_chain == _sch_chain)
            # [v5 2026-05-06] stale 비교 fuzzy 화 — wrapped/raw prompt 변형도 same task
            # 으로 인식. exact != 비교는 사용자 prompt wrapping 변형 (앞뒤 quote, '에',
            # 'multi-' prefix 등) 마다 false positive 폐기 → 매번 _active_schema=None →
            # request_schema_approval 재진입 → 무한 chain 재진입 루프 (최신로그.txt 19:17).
            def _stale_norm(s: str) -> str:
                return (s or "").strip().strip("'\"`").strip().lower()
            def _stale_same(a: str, b: str) -> bool:
                na, nb = _stale_norm(a), _stale_norm(b)
                if not na or not nb:
                    return False
                if na == nb:
                    return True
                short, long_ = (na, nb) if len(na) <= len(nb) else (nb, na)
                return len(short) >= 20 and short in long_
            _prompt_same = _stale_same(_sch_prompt, _tp_norm)
            if _sch_prompt and _tp_norm and not _prompt_same and not _chain_match:
                print(
                    f"  🧹 [Schema] stale _active_schema 폐기 — "
                    f"prev='{_sch_prompt[:30]}' / new='{_tp_norm[:30]}'"
                )
                # trainer 재링크 플래그 초기화 — 새 schema 에서 CausalMatrix 재연결
                # 필요하면 다시 _causal_linked=False 로 시작되도록 schema 자체가 바뀐다.
                self._active_schema = None

        # [2단계] MissionSchema가 없으면 자동 생성 후 승인 요청
        if self._active_schema is None:
            # [Fix D1 2026-05-06] task 가 이미 chain_id 가지고 있으면 (chain inline 등록 task)
            # 그 chain 의 schema 재구성 — 새 chain 생성 차단.
            # Why: 최신로그.txt 09:52 첫 chain 생성 → dispatcher 가 그 chain task pickup →
            # 여기서 또 request_schema_approval → research 의도 매칭 → _create_research_chain_inline
            # → 두 번째 chain 생성(09:53). 이중 chain + 두 번째 schema 가 stale 폐기 trigger →
            # WAITING_USER → Watchdog 60s 대기 콤보로 ReAct 진입 1~2분 지연.
            _existing_chain_id = (task.get("chain_id") or "").strip()
            if _existing_chain_id:
                try:
                    from eidos_mission_chain import get_orchestrator as _go_d1
                    _orch_d1 = _go_d1(core=self.core)
                    _ch_d1 = _orch_d1.registry.get(_existing_chain_id)
                    _ch_status_d1 = getattr(_ch_d1, "status", None) if _ch_d1 else None
                    _is_alive_d1 = (
                        _ch_d1 is not None
                        and _ch_status_d1 is not None
                        and getattr(_ch_status_d1, "name", str(_ch_status_d1)).upper()
                        not in ("ABORTED", "FAILED", "COMPLETED")
                    )
                    if _is_alive_d1:
                        _schema_d1 = await _orch_d1.build_schema_for_current_stage(_ch_d1)
                        if _schema_d1 is not None:
                            self._active_schema = _schema_d1
                            print(
                                f"  ⚡ [Fix D1] 기존 chain 재사용 — chain_id="
                                f"{_existing_chain_id[:8]} stage="
                                f"'{getattr(_schema_d1, 'stage', '')}' (재생성 차단)"
                            )
                except Exception as _e_d1:
                    print(
                        f"  [Fix D1] 기존 chain 재사용 실패 (무시, 정상 schema 생성으로 폴백): {_e_d1}"
                    )

        if self._active_schema is None:
            await self.request_schema_approval(task_prompt)
            # [Fix D2 2026-05-06] schema auto-approval 시 즉시 dispatcher 재진입.
            # Why: _create_research_chain_inline 이 _active_schema 자동으로 채워주고
            # schedule 에 새 chain task 등록까지 했지만, 호출자(여기)는 이를 모르고 항상
            # WAITING_USER 박음 → Watchdog 60s idle 도달 후에야 run_now() 발동(최신로그.txt
            # 09:53~09:54). auto-approval 됐으면 fire-and-forget run_now() 로 즉시 깨움.
            if self._active_schema is not None:
                print("  ⚡ [Fix D2] schema 자동 승인 — WAITING_USER 우회, dispatcher 재진입")
                self._touch_activity()
                # 현재 task 는 lazy 마무리 — chain inline 이 새 task 등록했음
                try:
                    self._remove_task_from_schedule(task)
                except Exception as _e_rm_d2:
                    print(f"  [Fix D2] 기존 task 정리 실패 (무시): {_e_rm_d2}")
                # [Fix 2026-05-15 orphan-chain] 기존엔 fire-and-forget
                # asyncio.create_task(self.run_now()) 였으나, 이 코드는 자신을
                # 띄운 _tick() 의 _tick_lock 이 아직 잡힌 상태에서 실행돼
                # _tick_guarded 의 locked 가드에 막혀 무력화됐다(orphan chain 직접
                # 원인). 대신 재진입 플래그만 세운다 — _tick_guarded 가 lock 해제
                # 직전 _tick 을 1회 더 돌리고, 그 재실행이 Fix A v5 가 등록한
                # 새 PENDING task(chain_id 동일)를 pickup → chain step 정상 진행.
                self._tick_rerun_requested = True
                return
            # 정상 사용자 승인 대기 흐름 (research 의도 미매칭 + 첫 schema 생성 케이스)
            # 승인 대기 — _paused_task에 저장해야 run_now() 재진입 시 이어하기 분기로 복귀됨
            # (저장하지 않으면 _pick_pending_task()가 WAITING_USER 상태를 무시해 task를 잃어버림)
            self._paused_task = task
            task["status"] = "WAITING_USER"
            self._wait_for_user = True
            return

        # [4단계] 스키마 승인 직후 — HOW_MUCH 속성을 LongTermGoal sub_goals로 등록
        if self._active_schema and getattr(self._active_schema, "_goal_linked", False) is False:
            try:
                from eidos_goal_bridge import apply_schema_to_goal
                apply_schema_to_goal(self.core, self._active_schema, goal_ref)
                self._active_schema._goal_linked = True
            except Exception as _e_gb:
                print(f"  ⚠️ [GoalBridge] apply_schema_to_goal 실패 (무시): {_e_gb}")

        # [4-B단계] 스키마 승인 직후 — CausalMatrix Y변수를 HOW_MUCH로 초기화
        # [Fix] 수익/판매 도메인이 아닌 태스크는 스킵 (X변수가 무의미 — 오염 방지)
        _cm_task_type2 = getattr(self._active_schema, "task_type", "general") if self._active_schema else "general"
        _CM_SKIP_TYPES2 = ("browser_read", "general", "write", "report")
        if self._active_schema and not getattr(self._active_schema, "_causal_linked", False) \
                and _cm_task_type2 not in _CM_SKIP_TYPES2:
            try:
                from eidos_causal_matrix import trainer_from_schema, MATRIX_CACHE
                import os as _os
                # 기존 trainer가 있으면 Y변수만 교체, 없으면 새로 생성
                existing = getattr(self.core, "_causal_trainer", None)
                # X변수: core에 없으면 빈 리스트 (PatrolAgent가 나중에 채움)
                x_vars = getattr(self.core, "_causal_x_vars", [])
                if x_vars or existing is not None:
                    trainer = trainer_from_schema(
                        self._active_schema, x_vars, existing_trainer=existing
                    )
                    self.core._causal_trainer = trainer
                    # 캐시 파일이 있으면 기존 학습 데이터 로드
                    if existing is None and _os.path.exists(MATRIX_CACHE):
                        try:
                            from eidos_causal_matrix import CausalMatrixTrainer
                            loaded = CausalMatrixTrainer.load(MATRIX_CACHE)
                            cur_y = {v.name for v in trainer.y_vars}
                            old_y = {v.name for v in loaded.y_vars}
                            if cur_y == old_y and len(loaded.x_vars) == len(trainer.x_vars):
                                trainer._pairs_ext = loaded._pairs_ext
                                trainer._pairs_int = loaded._pairs_int
                                print(f"  [CausalMatrix] 기존 학습 데이터 복원 "
                                      f"(ext={len(loaded._pairs_ext)}, "
                                      f"int={len(loaded._pairs_int)})")
                            else:
                                print(
                                    f"  [CausalMatrix] 캐시 Y/X 불일치 → pairs 복원 스킵 "
                                    f"(cache_Y={sorted(old_y)}, "
                                    f"cur_Y={sorted(cur_y)})"
                                )
                        except Exception:
                            pass
                    self._active_schema._causal_linked = True
                    print(f"  [CausalMatrix] Y변수 {len(self._active_schema.attributes)}개 "
                          f"HOW_MUCH로 초기화")
            except Exception as _e_cm:
                print(f"  ⚠️ [CausalMatrix] trainer 초기화 실패 (무시): {_e_cm}")

        # ── [Helix 소뇌] 복잡한 다단계 작업은 Helix GoalNet에 위임 ──────
        # 자율실행(patrol/dispatcher 생성 task)이면 Helix 강제 — 계획 없는 ReAct 폭주 방지
        _task_source = task.get("source", "user")
        _force_helix = _task_source in ("patrol", "dispatcher")

        # 휴리스틱 결과 산출 (학습 부족 시 router의 fallback으로 사용)
        _heuristic = self._should_delegate_to_helix(task_prompt, self._active_schema, force=_force_helix)

        # [P0 하드 오버라이드] 알려진 플랫폼 키워드 → Router 학습 무시하고 ReAct 직행.
        # 배경: DecisionRouter가 force_helix=True + 학습 부족이면 helix 반환 →
        # _should_delegate_to_helix의 플랫폼 바이패스가 완전히 무력화됨.
        # 플랫폼 바이패스는 "전용 하드웨어(CLICK/FILL/READ) 보유" 의미이므로
        # 학습에 좌우되지 않는 하드 신호로 처리.
        _chain_id_tsk = (task.get("chain_id") or "").strip()
        _platform_bypass = self._check_platform_bypass(task_prompt, chain_id=_chain_id_tsk)
        _ttype = (
            getattr(self._active_schema, "task_type", "general")
            if self._active_schema else "general"
        )

        # [P1 하드 오버라이드 2026-04-20] MissionChain stage 는 단순 파이프라인
        # (research/plan/dev/register/support). Helix 의 DNA/InfoGate/다단계 분해가
        # 불필요하며, 이전 로그에서 plan stage 가 Helix 에 위임돼 "답변생성 차단 1-A"
        # 루프로 11분간 토큰만 태우는 증상 확인. chain_id 가 있으면 ReAct 고정.
        if _chain_id_tsk:
            _mode = "react"
            _stage_name = (
                getattr(self._active_schema, "stage", "") if self._active_schema else ""
            ) or "?"
            _pf_note = f", platform='{_platform_bypass}'" if _platform_bypass else ""
            print(
                f"  🔒 [Router HARD] chain stage '{_stage_name}' → ReAct 고정 "
                f"(chain={_chain_id_tsk[:8]}{_pf_note})"
            )
        elif _platform_bypass:
            _mode = "react"
            print(f"  🔒 [Router HARD] 플랫폼 '{_platform_bypass}' 바이패스 → ReAct 고정 (force={_force_helix})")
        elif _ttype == "browser_read":
            # [P2 하드 오버라이드 2026-04-21] browser_read 는 단일 스텝 브라우저 조회 확정.
            # MissionSchema 가 '브라우저 의도' 를 명시적으로 분류한 결과이므로 Helix
            # 다단계 분해가 불필요하며, force_helix 자동 승급이 "서울 날씨 알려줘" 같은
            # 단순 조회까지 GoalNet 분해 → Fallback 웹검색 빈결과 → 재분해 루프로 폭주시킴.
            _mode = "react"
            print(f"  🔒 [Router HARD] task_type='browser_read' → ReAct 고정 (force={_force_helix})")
        else:
            # ── [DecisionRouter] 학습 기반 실행 모드 라우팅 (A'안 쌍 1) ─────
            try:
                from eidos_decision_router import get_router as _gr_em
                from eidos_tuning import get_tuned_param as _gtp_em
                _hcm = str(_gtp_em("helix_core_mode", "auto")).lower()
                _helix_avail = (
                    getattr(self, "_helix_engine", None) is not None
                    and getattr(self._helix_engine, "is_available", False)
                )
                _decision = _gr_em().route_execution_mode(
                    task_prompt=task_prompt,
                    task_type=_ttype,
                    schema_attrs=len(getattr(self._active_schema, "attributes", []) or []) if self._active_schema else 0,
                    force_helix=_force_helix,
                    helix_available=_helix_avail,
                    helix_core_mode=_hcm,
                    heuristic_decision=_heuristic,
                    task_id=str(task.get("task_prompt", ""))[:80],
                )
                _mode = _decision.mode
                if not _decision.used_heuristic:
                    print(f"  🧠 [Router] 실행 모드={_mode} ({_decision.reason})")
            except Exception as _re_em:
                _mode = "helix" if _heuristic else "react"
                print(f"  [Router] 라우팅 실패 → 휴리스틱 fallback({_mode}): {_re_em}")

        # 학습 통계용 시작 시각
        import time as _tt_em
        _t0_em = _tt_em.time()
        try:
            if _mode == "helix":
                if _force_helix:
                    self._report(f"🧭 **[자율실행]** source={_task_source} → Helix 위임")
                await self._delegate_to_helix(task_prompt, goal_ref, task)
            else:
                await self._react_loop(task_prompt, goal_ref, task)
        finally:
            # ── 실행 종료 → 성공 여부로 학습 통계 갱신 ────────────────
            try:
                from eidos_decision_router import get_router as _gr_em2
                _succ = str(task.get("status", "")).upper() in ("DONE", "COMPLETED", "SUCCESS")
                _gr_em2().record_execution_result(
                    task_type=_ttype,
                    mode=_mode,
                    success=_succ,
                    duration_sec=_tt_em.time() - _t0_em,
                )
                # ── Helix 종료 시 Core.current_goal progress 동기화 ──
                # ReAct는 기존 GoalStackEvaluator/CompletionDetector가 처리.
                # Helix는 외부 엔진이라 Core 동기화 훅 누락이었음 — 여기서 보강.
                if _mode == "helix" and _succ:
                    _gref = task.get("goal_ref") or ""
                    if _gref and hasattr(self.core, "long_term_goals"):
                        for _g in self.core.long_term_goals:
                            if getattr(_g, "name", "") == _gref:
                                _g.last_updated = _tt_em.time()
                                if hasattr(_g, "completed_sub_goals"):
                                    _tp = task.get("task_prompt", "")
                                    if _tp and _tp not in _g.completed_sub_goals:
                                        _g.completed_sub_goals.append(_tp)
                                break
            except Exception:
                pass

    # ── Helix 소뇌 위임 판단 + 실행 ─────────────────────────────────────

    # 알려진 플랫폼 — EIDOS ReAct가 특화된 CLICK/FILL/READ 하드웨어를 보유하므로
    # 도메인이 분명하면 Helix 분해보다 ReAct 직행이 안정적.
    _KNOWN_PLATFORM_KEYWORDS: tuple = (
        "kmong", "크몽", "coupang", "쿠팡", "naver", "네이버",
        "youtube", "유튜브", "instagram", "인스타", "facebook", "페이스북",
        "amazon", "아마존", "ebay", "이베이", "aliexpress", "알리익스프레스",
        "namu.wiki", "나무위키", "wikipedia", "위키", "github",
        "tistory", "티스토리", "brunch", "브런치", "melon", "멜론",
        "11번가", "gmarket", "지마켓", "daum",
    )
    # 흐름 단어와 혼동되는 모호 키워드는 URL 패턴(domain 형태)으로만 매칭.
    # ex) "다음 단계로 진행" 에서 '다음'이 daum 으로 잘못 잡히는 것을 방지.
    _AMBIGUOUS_PLATFORM_TOKENS: dict = {
        "다음": ("daum.net", "daum.com", "://daum", "www.daum"),
        "위키": ("wikipedia.org", "namu.wiki", "위키피디아", "나무위키"),
    }

    def _check_platform_bypass(self, task_prompt: str, chain_id: str = ""):
        """
        task_prompt 또는 소속 체인의 metadata/원문에서 알려진 플랫폼 키워드를 탐색.

        P0 2026-04-20: MissionChain 후속 stage(plan.md 작성 등)는 prompt 에
        플랫폼 이름이 빠져있을 수 있다(→ 기존 로그 크몽 → Helix 루프의 원인).
        chain_id 가 있으면 chain.metadata["platform"] 과 chain.original_prompt 도
        스캔해서 research stage 에서 감지된 플랫폼을 후속 stage 가 상속하게 한다.

        2026-04-20 (추가): upstream 섹션에 섞인 "다음 단계로" 같은 흐름 단어가
        '다음'→daum 으로 오매칭되던 문제 차단. 모호 키워드는 URL 패턴으로만 매칭.
        chain metadata['platform'] 은 task_prompt 직접 매칭보다 우선 적용 —
        체인이 이미 플랫폼을 확정했으면 후속 stage prompt 는 재해석하지 않는다.
        """
        def _match_in(text: str):
            """텍스트에서 플랫폼 키워드 1회 매칭. 모호 키워드는 URL 패턴 필요."""
            if not text:
                return None
            tl = text.lower()
            for kw in self._KNOWN_PLATFORM_KEYWORDS:
                if kw in tl:
                    return kw
            for kw, patterns in self._AMBIGUOUS_PLATFORM_TOKENS.items():
                if any(p in tl for p in patterns):
                    return kw
            return None

        # 1. 체인 상속 우선 — chain metadata['platform'] 이 이미 정해져 있으면
        #    task_prompt 의 흐름 단어가 덮어쓸 수 없게 최우선으로 확정한다.
        if chain_id:
            try:
                from eidos_mission_chain import MissionChainRegistry
                _chain = MissionChainRegistry.instance().get(chain_id)
            except Exception:
                _chain = None
            if _chain is not None:
                _meta_pf = str((_chain.metadata or {}).get("platform", "") or "").lower()
                if _meta_pf:
                    for kw in self._KNOWN_PLATFORM_KEYWORDS:
                        if kw in _meta_pf or _meta_pf == kw.lower():
                            return kw
                    for kw, patterns in self._AMBIGUOUS_PLATFORM_TOKENS.items():
                        if kw in _meta_pf or any(p in _meta_pf for p in patterns):
                            return kw

        # 2. 현재 task prompt 직접 매칭 — upstream 블록은 제외(흐름 단어 오염 방지)
        if task_prompt:
            scan_target = task_prompt
            upstream_marker = "[이전 단계 산출물"
            if upstream_marker in scan_target:
                scan_target = scan_target.split(upstream_marker, 1)[0]
            _hit = _match_in(scan_target)
            if _hit:
                return _hit

        # 3. 체인 original_prompt 스캔 (metadata 미설정 폴백)
        if chain_id:
            try:
                from eidos_mission_chain import MissionChainRegistry
                _chain = MissionChainRegistry.instance().get(chain_id)
            except Exception:
                _chain = None
            if _chain is not None:
                _hit2 = _match_in(_chain.original_prompt or "")
                if _hit2:
                    return _hit2
        return None

    def _should_delegate_to_helix(self, task_prompt: str, schema, force: bool = False) -> bool:
        """
        이 작업을 Helix GoalNet에 위임해야 하는지 판단.

        [P0/P3] Helix는 이제 EIDOS 프록시 ToolExecutor를 통해 QtWebEngine
        세션을 공유한다. 따라서 task_type='browser'를 차단하던 블랭킷 가드는 해제.
        단 사용자가 직접 입력한 브라우저 작업(force=False)은 ReAct의 세밀한
        CLICK/FILL/READ 퍼포먼스가 더 나으므로 기본 경로를 유지한다.

        [P2] helix_core_mode 튜닝 파라미터로 A/B 토글 가능:
          - "off" : 항상 ReAct (Helix 완전 비활성 — 기존 EIDOS 동작과 동일)
          - "auto" (기본): force 시 Helix, 그 외 휴리스틱 판정
          - "on"  : 가능한 모든 경로를 Helix로 (사용자 browser 포함) — 체감 비교용

        위임 기준 (auto 모드 기준, 위에서 아래로):
        1. force=True (patrol/dispatcher 생성 task) → 가용하면 위임 (browser 포함)
        2. task_type=='browser' & non-force → 기본 ReAct (사용자 직접 명령)
        3. MissionSchema HOW_MUCH 임계값 이상 → 복합 작업
        4. 프롬프트 다단계 키워드 임계값 이상
        """
        # [P2] 글로벌 토글 먼저 확인
        try:
            from eidos_tuning import get_tuned_param
            _mode = str(get_tuned_param("helix_core_mode", "auto")).lower()
        except Exception:
            _mode = "auto"
        if _mode == "off":
            return False

        # Helix 사용 가능 여부 확인
        if not hasattr(self, '_helix_engine') or self._helix_engine is None:
            try:
                from helix_engine import HelixEngine
                self._helix_engine = HelixEngine(
                    report_fn=self._report,
                    ask_user_fn=self._helix_ask_user,
                )
                if not self._helix_engine.is_available:
                    self._helix_engine = None
                    return False
            except ImportError:
                self._helix_engine = None
                return False

        # [P1] Helix는 이제 EIDOS 서버 프록시 경유로 LLM을 호출하므로
        # 로컬 gemini_api_key 부재를 이유로 위임을 스킵하지 않는다.
        # (helix_engine._patch_helix_llm_calls가 _call_gemini를 런타임 패치함.)

        # [3-F] 알려진 플랫폼 Helix 우회 — force 여도 ReAct 우선.
        _cid_hint = ""
        try:
            _cid_hint = (getattr(schema, "chain_id", "") or "").strip() if schema else ""
            if not _cid_hint and self._active_schema is not None:
                _cid_hint = (getattr(self._active_schema, "chain_id", "") or "").strip()
        except Exception:
            _cid_hint = ""
        _matched_platform = self._check_platform_bypass(task_prompt, chain_id=_cid_hint)
        if _matched_platform:
            print(
                f"  [3-F] 플랫폼 힌트 '{_matched_platform}' 감지 → Helix 우회 → ReAct "
                f"(force={force})"
            )
            return False

        # [자율실행] patrol/dispatcher가 생성한 task → 계획적 접근 필요 → Helix.
        # P3 프록시로 browser 세션 공유되므로 browser도 포함.
        if force:
            return True

        # [P2] "on" 모드면 사용자 직접 명령도 Helix로 (browser 포함) — A/B 비교용.
        if _mode == "on":
            return True

        # 사용자 직접 명령 + browser는 기존 ReAct 유지 (세밀 CLICK/FILL/READ 우위).
        if schema and getattr(schema, "task_type", "general") == "browser":
            return False

        # 판단 기준
        # [Step D.1] helix_delegation_attrs_threshold 튜닝 훅 — 공명 gap 기반 임계값.
        try:
            _attr_threshold = int(get_tuned_param("helix_delegation_attrs_threshold", 2))
            _keyword_threshold = int(get_tuned_param("helix_keyword_threshold", 1))
        except Exception:
            _attr_threshold = 2
            _keyword_threshold = 1
        if schema:
            attrs = getattr(schema, "attributes", [])
            # attrs 개수가 임계값 이상이면 복합 작업 → Helix
            if len(attrs) >= _attr_threshold:
                return True

        # 다단계 키워드 감지 (P2: 임계값 기본 2→1로 완화, 튜닝 가능)
        _HELIX_KEYWORDS = [
            "분석하고", "조사하고", "정리해", "보고서",
            "계획", "전략", "비교", "단계별",
            "파일로 저장", "리포트", "요약해서",
        ]
        prompt_lower = task_prompt.lower()
        keyword_hits = sum(1 for kw in _HELIX_KEYWORDS if kw in prompt_lower)
        if keyword_hits >= _keyword_threshold:
            return True

        return False

    def _helix_ask_user(self, question: str) -> str:
        """
        Helix의 ask_user를 Telegram ApprovalGateway 경유로 라우팅한다.
        Helix 워커 스레드에서 호출되는 동기 블로킹 함수 — EIDOS 메인 루프에
        run_coroutine_threadsafe로 coroutine을 submit하고 결과 문자열을 반환.

        실패/미설정 시 기존 채팅창 경로로 폴백.
        """
        import asyncio
        try:
            from eidos_approval_gateway import ApprovalGateway, GateDecision
        except Exception as _e:
            self._report(f"  ⚠️ [Telegram] ApprovalGateway import 실패 → 채팅창 폴백: {_e}")
            return self._legacy_ask_user_chatroom(question)

        loop = getattr(self, "_eidos_loop", None)
        if loop is None or loop.is_closed():
            self._report("  ⚠️ [Telegram] EIDOS loop 미확보 → 채팅창 폴백")
            return self._legacy_ask_user_chatroom(question)

        gateway = ApprovalGateway.get()
        ctx: Dict[str, Any] = {"source": "Helix"}
        try:
            if self._active_schema is not None:
                ctx["goal"] = getattr(self._active_schema, "goal_ref", "") or ""
                ctx["task"] = getattr(self._active_schema, "prompt", "") or ""
        except Exception:
            pass

        try:
            fut = asyncio.run_coroutine_threadsafe(
                gateway.ask_user_via_telegram(question, context=ctx),
                loop,
            )
            # Gateway 내부 timeout + 여유 10초
            wait_sec = float(getattr(gateway, "_timeout", 900.0)) + 10.0
            result = fut.result(timeout=wait_sec)
        except Exception as e:
            self._report(f"  ⚠️ [Telegram] 승인 경로 오류 → 채팅창 폴백: {e}")
            return self._legacy_ask_user_chatroom(question)

        if result.decision == GateDecision.APPROVED:
            if result.edited_content:
                self._report(f"  ✓ [Telegram] Helix 답변 수신(수정): {result.edited_content[:50]}")
                return result.edited_content
            self._report("  ✓ [Telegram] Helix 승인")
            return "승인"
        if result.decision == GateDecision.REJECTED:
            self._report(f"  ✗ [Telegram] Helix 거부: {result.reason}")
            return "거부"
        if result.decision == GateDecision.TIMEOUT:
            self._report(f"  ⚠️ [Telegram] Helix 승인 타임아웃({int(getattr(gateway, '_timeout', 900))}s) — 스킵 처리")
            return ""
        # AUTO (이 경로에선 force_risk=MEDIUM이라 거의 안 옴)
        return "승인"

    def _legacy_ask_user_chatroom(self, question: str) -> str:
        """Telegram 폴백 — 기존 EIDOS 채팅창 기반 ask_user. 동기 블로킹."""
        import threading
        self._helix_answer_event = threading.Event()
        self._helix_answer_text = ""

        self._report(
            f"[ASK_USER] ❓ **[Helix 질문]** {question}\n"
            f"└─ 채팅창에 답변을 입력해주세요 (60초 대기)."
        )

        answered = self._helix_answer_event.wait(timeout=60.0)
        if answered and self._helix_answer_text:
            self._report(f"  ✓ Helix 답변 수신: {self._helix_answer_text[:50]}")
            return self._helix_answer_text

        self._report("  ⚠️ Helix 질문 타임아웃 — 스킵 처리")
        return ""

    def receive_helix_answer(self, answer: str):
        """ChatWindow에서 호출 — Helix ask_user 답변 전달."""
        self._helix_answer_text = answer
        if hasattr(self, '_helix_answer_event') and self._helix_answer_event:
            self._helix_answer_event.set()

    async def _delegate_to_helix(self, task_prompt: str, goal_ref: str, task: Dict):
        """Helix GoalNet에 작업을 위임하고 결과를 보고."""
        # Helix 워커 스레드가 _helix_ask_user → ApprovalGateway(async)를 호출할 때
        # run_coroutine_threadsafe로 EIDOS 루프를 찾을 수 있도록 캡처해둔다.
        try:
            import asyncio as _asyncio
            self._eidos_loop = _asyncio.get_running_loop()
        except RuntimeError:
            self._eidos_loop = None

        self._report(
            f"🧠 **[Helix 소뇌 위임]** `{task_prompt[:60]}`\n"
            f"└─ 복합 다단계 작업 — GoalNet 분해 → 실행"
        )

        # API 키 주입
        if hasattr(self.core, '_api_key'):
            self._helix_engine.set_api_key(self.core._api_key)
        elif hasattr(self, '_active_schema') and self._active_schema:
            # eidos_settings에서 읽기
            pass

        # [P3 2026-04-20] chain stage 에서 Helix 로 올라왔다면 이전 stage 산출물을
        # engine._pending_chain_upstream 에 실어둔다. executor 생성 시 주입돼
        # 1-A 가드(답변생성 차단)의 완화 조건으로 쓰이고, situation 텍스트에도
        # 요약본이 들어가 Decomposer 가 "이미 확보된 데이터" 를 인지한다.
        _chain_upstream_dict: Dict[str, str] = {}
        if self._active_schema is not None:
            _raw_up = getattr(self._active_schema, "upstream_outputs", {}) or {}
            for _uk, _uv in _raw_up.items():
                if _uv is None:
                    continue
                _uvs = str(_uv)
                if not _uvs:
                    continue
                _chain_upstream_dict[_uk] = _uvs
        if _chain_upstream_dict:
            try:
                self._helix_engine._pending_chain_upstream = dict(_chain_upstream_dict)
            except Exception as _ce:
                print(f"  ⚠️ [Helix] upstream 전달 실패(무시): {_ce}")

        # 상황 정보 구성
        situation = ""
        if goal_ref:
            situation += f"최상위 목표: {goal_ref}\n"
        if hasattr(self.core, 'personal_context') and self.core.personal_context:
            situation += f"사용자 상황: {self.core.personal_context[:300]}\n"
        if self._active_schema:
            schema = self._active_schema
            situation += f"WHO: {getattr(schema, 'who', '')}\n"
            situation += f"WHERE: {getattr(schema, 'where', '')}\n"
            situation += f"WHAT: {getattr(schema, 'what', '')}\n"
            situation += f"HOW: {getattr(schema, 'how', '')}\n"
        # chain upstream 요약본을 situation 에 첨부 (Decomposer 맥락 강화).
        if _chain_upstream_dict:
            situation += "\n[이전 단계 산출물 요약]\n"
            for _uk, _uvs in list(_chain_upstream_dict.items())[:6]:
                _preview = _uvs[:400] + ("…" if len(_uvs) > 400 else "")
                situation += f"▼ {_uk}\n{_preview}\n"

        try:
            result = await self._helix_engine.run(
                goal=task_prompt,
                situation=situation,
                max_steps=50,
            )

            # 결과 보고
            self._report(f"📊 **[Helix 결과]**\n{result.summary[:500]}")

            # world_state를 EIDOS에 반영
            if result.world_state:
                self._merge_helix_world_state(result.world_state)

            if result.success:
                task["status"] = "DONE"
                self._report(f"✅ **[Helix 소뇌 완료]** 목표 달성")

                # ── [2026-04-19 fix] Helix 경로에도 체인 stage 완료 신호를 연결 ──
                # ReAct 의 WRITE 핸들러(Line 2943~)만 on_stage_complete 로
                # 체인을 진행시켰기 때문에, Helix 가 파일쓰기(action=3)로 산출물을
                # 떨궈도 MissionChain 이 정체됐다. 여기서 Helix 트리를 스캔해
                # stage 산출물과 매칭되는 파일이 있으면 ReAct 와 동일하게
                # 체인 전진 훅을 발화한다.
                schema = self._active_schema
                _chain_id = getattr(schema, "chain_id", "") if schema else ""
                if _chain_id and schema is not None:
                    artifact = self._find_helix_stage_artifact(result, schema)
                    if artifact is not None:
                        _fname, _fcontent = artifact
                        try:
                            schema.add_step(
                                "WRITE", _fname,
                                _fcontent or f"파일 '{_fname}' 생성 완료.", "",
                            )
                        except Exception as _eah:
                            print(f"  ⚠️ [Helix→Chain] step_history 합성 실패 (무시): {_eah}")
                        self._report(
                            f"✅ **[체인 단계 산출물 도착]** stage='{getattr(schema, 'stage', '')}' "
                            f"({getattr(schema, 'stage_index', 0) + 1}) → 산출물 "
                            f"'{_fname[:60]}' 저장 완료. 다음 단계로 자동 진행."
                        )
                        try:
                            self._remove_task_from_schedule(task)
                        except Exception:
                            pass
                        await self._on_react_done(
                            task_prompt, goal_ref, task, [], schema,
                        )
                        return
            else:
                # Helix 실패 → ReAct로 폴백
                self._report(
                    f"⚠️ **[Helix 미완료]** {result.status}\n"
                    f"└─ ReAct 루프로 폴백합니다..."
                )
                await self._react_loop(task_prompt, goal_ref, task)

        except Exception as e:
            self._report(f"❌ **[Helix 오류]** {e}\n└─ ReAct 루프로 폴백합니다...")
            await self._react_loop(task_prompt, goal_ref, task)

    def _merge_helix_world_state(self, helix_ws: Dict):
        """Helix의 world_state를 core.schedule 또는 다음 작업의 컨텍스트에 반영."""
        try:
            # core에 helix_context 저장 → 다음 ReAct 루프의 _observe에서 참조 가능
            if not hasattr(self.core, '_helix_context'):
                self.core._helix_context = {}
            self.core._helix_context.update(helix_ws.get("facts", {}))
            self.core._helix_context.update(helix_ws.get("resources", {}))
            metrics = helix_ws.get("metrics", {})
            if metrics:
                if not hasattr(self.core, '_helix_metrics'):
                    self.core._helix_metrics = {}
                self.core._helix_metrics.update(metrics)
            print(f"  [Helix→EIDOS] world_state 병합 완료: "
                  f"facts={len(helix_ws.get('facts', {}))} "
                  f"metrics={len(metrics)}")
        except Exception as e:
            print(f"  ⚠️ [Helix→EIDOS] world_state 병합 실패: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # 2단계: ReAct Loop — Observe → Think → Act → Verify
    # ══════════════════════════════════════════════════════════════════════

    async def _run_direct_llm_task(
        self,
        task_prompt: str,
        goal_ref: str,
        task: Dict,
        schema,
        anchor: str,
    ) -> None:
        """
        general/write/report 타입 전용 단일 LLM 직접 호출.
        ReAct 루프(Observe→Think→Act) 없이 LLM에게 직접 작업을 수행시킨다.
        HOW_MUCH 속성들을 프롬프트에 포함해 한 번에 달성하도록 유도.
        """
        # [P0-2 2026-04-25] 이전 stage 가 force_advance / stage_failed 였으면
        # DirectTask 환각 금지 — research 0건 위에 dev 가 산출물 만드는 패턴 차단.
        # ASK_USER 로 사용자 개입 유도하고 stage_failed 마킹.
        if task.get("__prev_stage_force_advanced__"):
            _prev_st = task.get("__prev_stage_name__", "?")
            _cur_st = (getattr(schema, "stage", "") or "?")
            print(
                f"  🔴 [Hallucination Guard] DirectTask 차단 — prev_stage='{_prev_st}' 가 "
                f"force_advance / stage_failed. cur_stage='{_cur_st}' 산출물 신뢰 못 함."
            )
            self._report(
                f"🔴 **[환각 차단]** 이전 stage '{_prev_st}' 가 강제 진행/실패로 끝나 산출물이 비어있음. "
                f"이 stage('{_cur_st}') 를 LLM 단일 호출로 진행하면 환각 위험 — chain stage_failed 처리.\n"
                f"└─ 사용자 확인 후 `재시도 {_prev_st}` 로 이전 stage 재실행 권장."
            )
            task["__stage_failed__"] = True
            task["__stage_fail_reason__"] = f"hallucination_guard:prev={_prev_st}"
            task["status"] = "FAILED"
            try:
                if schema is not None:
                    schema._chain_success = False
            except Exception:
                pass
            self._remove_task_from_schedule(task)
            return

        # HOW_MUCH 속성 목록
        attrs_str = ""
        if schema and schema.attributes:
            attrs_str = "\n".join(
                f"  - {a.name}: {a.operator} {a.value}"
                for a in schema.attributes if not a.achieved
            )

        prompt = f"""{anchor}

[실행할 작업]
{task_prompt}

[달성해야 할 목표 (HOW_MUCH)]
{attrs_str or "없음"}

[지시사항]
위 작업을 직접 수행하라. 브라우저나 외부 도구 없이 지식과 추론만으로 결과물을 작성하라.
각 HOW_MUCH 항목을 모두 포함한 완성된 결과물을 마크다운 형식으로 작성하라.
JSON 형식이 아닌 자연어로 답하라."""

        self._report(f"🧠 **[직접 실행]** `{task_prompt[:60]}` — LLM 단일 호출로 처리 중...")

        # 서버 오류 응답 판별 접두사 (llm_module과 동일)
        _ERR_PREFIXES = (
            "[서버 오류", "[크레딧", "[Pro", "[서버 연결 실패",
            "[서버 응답 시간 초과", "[LLM 호출 오류",
        )

        # [Fix] 7회×(90+백오프) = 최악 ~10분 → 3회×(60+백오프cap) = 최악 ~3.5분
        MAX_RETRY = 3
        for _attempt in range(1, MAX_RETRY + 1):
            try:
                # [Fix 2026-04-21] 내부 get_llm_response_async 기본 timeout=20 은
                # 큰 프롬프트에 부족해 wait_for 60s 전에 에러 리턴. 내부도 60s 로.
                result = await asyncio.wait_for(
                    get_llm_response_async(prompt, max_tokens=16384, timeout=90),
                    timeout=100.0,
                )

                # 서버 오류 응답 감지 → 재시도
                if not result or any(result.startswith(p) for p in _ERR_PREFIXES):
                    _err_preview = (result or "빈 응답")[:80]
                    print(f"  ⚠️ [DirectTask] 서버 오류 (시도 {_attempt}/{MAX_RETRY}): {_err_preview}")
                    if _attempt < MAX_RETRY:
                        await asyncio.sleep(min(3.0 * _attempt, 10.0))  # 백오프 상한 10s
                        continue
                    # 최대 재시도 초과
                    self._report(f"❌ **[오류]** 서버 응답 실패 ({MAX_RETRY}회 재시도): {_err_preview}")
                    task["status"] = "FAILED"
                    self._remove_task_from_schedule(task)
                    return

                # [Fix 2026-04-21 P1-Direct] 절단 감지 + 이어쓰기 1회.
                # 증상: plan.md 같은 긴 산출물이 max_tokens 한도에서 문장 중간 절단.
                try:
                    result = await self._maybe_continue_truncated_write(
                        result, prompt, "", task_prompt, max_tokens=16384
                    )
                except Exception as _e_cont:
                    print(f"  ⚠️ [DirectTask] 이어쓰기 실패(원본 유지): {_e_cont}")

                # ── 성공 처리 ──────────────────────────────────────────────
                # [Fix 2026-04-21 P1-Direct] HOW_MUCH 속성을 내용 검증 없이 전부
                # achieved=True 로 도장찍던 False-PASS 버그 수정. 본문 키워드·개수
                # 검사(_auto_check_keyword_attr/_count_attr/_existence_attr)로 먼저
                # 매칭, 매칭 실패 속성은 미달성으로 남겨 Verifier 가 잡게 한다.
                if schema and schema.attributes:
                    _auto_hits = 0
                    for attr in schema.attributes:
                        if attr.achieved:
                            continue
                        if self._auto_check_keyword_attr(attr, result):
                            _auto_hits += 1
                            continue
                        if self._auto_check_count_attr(attr, result):
                            _auto_hits += 1
                            continue
                        # 이름에 '생성/작성/기획/보고서/draft/proposal' 포함 +
                        # 본문 800자+ 이면 완료형 속성으로 인정.
                        _nl = (attr.name or "").lower()
                        _NAME_OK = any(
                            k in _nl for k in (
                                "생성", "작성", "기획", "문서", "보고서", "report",
                                "draft", "proposal", "리포트", "초안", "completed"
                            )
                        )
                        if _NAME_OK and len(result) >= 800:
                            attr.achieved = True
                            attr.evidence = f"DirectTask 본문 {len(result)}자 + 이름 매칭"
                            _auto_hits += 1
                    print(f"  [DirectTask] HOW_MUCH 자동 달성: {_auto_hits}/{len(schema.attributes)}")

                # [Fix 2026-04-21 P1-Direct] 결과를 디스크에도 저장. task_type=write
                # 인데 chat 에만 뿌리고 파일은 사라지던 데이터 손실 방지. task_prompt
                # 에서 파일명 후보를 뽑거나 {chain}_{stage}.md 로 자동 저장.
                try:
                    self._persist_direct_task_result(task_prompt, schema, result)
                except Exception as _e_persist:
                    print(f"  ⚠️ [DirectTask] 디스크 저장 실패(무시): {_e_persist}")

                self._report(f"✅ **[완료]** {task_prompt[:60]}\n\n{result}")
                task["status"] = "DONE"

                try:
                    from eidos_goal_bridge import full_bridge
                    full_bridge(self.core, schema, goal_ref)
                except Exception as _gb_e:
                    print(f"  ⚠️ [DirectTask] GoalBridge 실패 (무시): {_gb_e}")

                self._remove_task_from_schedule(task)
                return  # 성공 → 루프 종료

            except asyncio.TimeoutError:
                print(f"  ⚠️ [DirectTask] 60초 타임아웃 (시도 {_attempt}/{MAX_RETRY})")
                if _attempt < MAX_RETRY:
                    await asyncio.sleep(min(3.0 * _attempt, 10.0))
                    continue
                self._report("⚠️ **[타임아웃]** LLM 응답 60초 초과 — 작업 중단")
                task["status"] = "FAILED"
                self._remove_task_from_schedule(task)
                return
            except Exception as e:
                self._report(f"❌ **[오류]** {e}")
                task["status"] = "FAILED"
                self._remove_task_from_schedule(task)
                return

    async def execute_react_loop_external(
        self,
        goal_text: str,
        max_steps: int = 10,
        timeout_sec: float = 300.0,
        chain_id: str = "",
        notify_step: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """[Phase B-React 2026-05-03] 외부 호출 가능 ReAct multi-step loop.

        eidos_milestone_executor 등 외부 모듈이 자연어 목표 → 다단계 ReAct 자율 호출.

        내부 — _observe → _think → _act 를 max_steps 회 직접 반복.
        _react_loop_impl 은 호출 X (백그라운드 _dispatch_loop state 충돌 회피).
        Verify (HOW_MUCH 검증) 등 고급 기능은 빠짐 — milestone executor 가 별도 처리.

        state 안전:
          - _active_schema 백업/복원
          - chain_id stub 부여 (자동 폐기 회피)
          - timeout / max_steps cap
          - ⚠️ prefix / DONE action 시 즉시 중단

        Args:
          goal_text: 자연어 목표 (예: "크몽에서 CRM 자동화 gig 20개 조사")
          max_steps: 최대 think→act cycle (기본 10)
          timeout_sec: 전체 timeout (기본 5분)
          chain_id: MissionChain 매핑용 (없으면 자동 stub)
          notify_step: callable(step_no, action, target, result) — 매 step 후 callback

        Returns:
          {
            "status": "done" | "timeout" | "blocked" | "error",
            "history": [{"step", "action", "target", "result"}, ...],
            "outputs": {"react_vars": {...}, "first_write": "..."},
            "step_count": int,
            "error": str,
          }
        """
        import time as _t_re
        import asyncio as _aio_re

        # [Phase L3 2026-05-03] _tick_lock 점검 — 백그라운드 _dispatch_loop 동시 진입 차단
        lock = getattr(self, "_tick_lock", None)
        if lock is not None and lock.locked():
            # 30초 polling 후에도 locked 면 거부
            wait_start = _t_re.time()
            while lock.locked() and (_t_re.time() - wait_start) < 30.0:
                await _aio_re.sleep(0.5)
            if lock.locked():
                return {
                    "status": "blocked",
                    "history": [], "outputs": {}, "step_count": 0,
                    "error": "dispatcher busy — _tick_lock 점유 중 (30s 대기 후 timeout)",
                }

        # _paused_task 점검 — auth 대기 중이면 사용자 응답 후 진입
        paused = getattr(self, "_paused_task", None)
        if paused is not None:
            return {
                "status": "blocked",
                "history": [], "outputs": {}, "step_count": 0,
                "error": "_paused_task 진행 중 — auth/captcha 사용자 응답 대기 (먼저 처리)",
            }

        # state 백업
        original_schema = getattr(self, "_active_schema", None)

        # 임시 schema 생성
        try:
            from eidos_mission_schema import MissionSchema
            # [Fix 2026-05-03] task_type="browse" — milestone executor 가 is_react_suitable
            # 로 ReAct loop 를 호출한 시점에서 이미 브라우저 탐색 작업이 확정. 기본
            # task_type="general" 은 dispatcher 의 _NON_BROWSER_TYPES 에 포함돼 prompt 에
            # "[브라우저 불필요] NAVIGATE/CLICK 절대 금지" 가 박히고 → LLM 이 ASK_USER 로
            # "브라우저 사용이 제한되어 있다" 호출 → 무한 사용자 확인 루프. browse 로
            # 명시해 NAVIGATE/CLICK 허용.
            temp_schema = MissionSchema(
                original_prompt=goal_text, where="", who="", what=goal_text,
                task_type="browse", verb="READ",
            )
        except Exception as e:
            return {
                "status": "error", "history": [], "outputs": {}, "step_count": 0,
                "error": f"schema 생성 실패: {type(e).__name__}: {str(e)[:120]}",
            }

        # [근본 fix 2026-05-04] chain_id 를 정상 attr (underscore 없음) 로 박기.
        # 기존 `_chain_id` (underscore prefix) 는 dispatcher 의 모든 chain-aware
        # 분기 (Sampling Gate / PRE-VETO chain abort / Carry-Over Guard _has_chain)
        # 가 읽지 못함 → schema.chain_id 가 빈 string 으로 인식 → ad-hoc schema 처럼
        # 동작 → strike 누적만 + abort 안 됨 + LLM 게으름 그대로 (사용자 frustration
        # 의 진짜 원인). 정상 attr 로 박으면 chain-aware 분기들 정상 발동.
        _real_chain_id = chain_id or f"react_ext_{int(_t_re.time())}"
        try:
            temp_schema.chain_id = _real_chain_id   # 정상 attr — chain-aware 분기 발동
            temp_schema._react_vars = {}
            temp_schema._react_var_urls = {}
            temp_schema._dead_end_urls = set()
        except Exception:
            pass
        print(
            f"  🔗 [react_ext] schema 활성화 — chain_id='{_real_chain_id}' "
            f"task_type='{getattr(temp_schema, 'task_type', '')}' "
            f"goal='{(goal_text or '')[:50]}'"
        )

        self._active_schema = temp_schema

        history: list = []
        start_t = _t_re.time()
        final_status = "done"
        error_msg = ""
        # [Fix 2026-05-03] transient ⚠️ (인덱스 stale / FILL no-editor) 연속 카운터.
        # 3 회 이상 연속 시 fatal 로 강등 → BLOCKED.
        _transient_streak = 0

        # [Phase L3] lock 잡기 — 백그라운드 _tick 동시 진입 차단
        # 단 lock 자체가 None 이면 (테스트 등) skip
        lock_ctx = lock if lock is not None else _aio_re.Lock()

        try:
          async with lock_ctx:
            for step_no in range(1, max_steps + 1):
                # timeout
                if _t_re.time() - start_t > timeout_sec:
                    final_status = "timeout"
                    error_msg = f"timeout after {timeout_sec}s"
                    break

                # observe
                try:
                    obs = await self._observe()
                except Exception as e:
                    final_status = "error"
                    error_msg = f"step {step_no} observe 실패: {type(e).__name__}: {str(e)[:80]}"
                    break

                # think
                try:
                    decision = await self._think(
                        task_prompt=goal_text,
                        anchor=goal_text,
                        obs=obs,
                        history=history,
                        schema=temp_schema,
                        step_no=step_no,
                    )
                except Exception as e:
                    final_status = "error"
                    error_msg = f"step {step_no} think 실패: {type(e).__name__}: {str(e)[:80]}"
                    break

                if not decision:
                    final_status = "blocked"
                    error_msg = f"step {step_no} LLM decision None"
                    break

                action = str(decision.get("action", "")).upper()
                target = str(decision.get("target", "") or "")
                content = str(decision.get("content", "") or "")
                reason = str(decision.get("reason", "") or "")

                # [Fix 2026-05-03] Anti-loop guard — 직전 2 step 이 같은 action+target
                # 으로 ⚠️ 실패였으면 이번에도 같은 선택은 100% 다시 실패. _act 호출
                # skip + history 에 명시적 ANTI_LOOP 마커 + 다른 행동 유도 메시지.
                # 이전 transient streak 카운터와 별개: target 동일성 기반 차단.
                if len(history) >= 2:
                    _last2 = history[-2:]
                    _same_pattern = all(
                        (h.get("action") == action and (h.get("target") or "") == target
                         and str(h.get("result", "")).startswith("⚠️"))
                        for h in _last2
                    )
                    if _same_pattern:
                        _veto_msg = (
                            f"🛡️ ANTI_LOOP veto: 직전 2회 같은 {action} target={target[:30]} "
                            f"실패 → 이번 step skip. 다음 step 에서 반드시 (a) 다른 [N] (b) "
                            f"NAVIGATE 로 다른 페이지 (c) ASK_USER 중 선택. 같은 {action} "
                            f"+ {target[:20]} 재시도 절대 금지."
                        )
                        history.append({
                            "step": step_no, "action": "ANTI_LOOP_VETO",
                            "target": target, "result": _veto_msg,
                        })
                        if notify_step:
                            try:
                                notify_step(step_no, "ANTI_LOOP_VETO", target, _veto_msg[:80])
                            except Exception:
                                pass
                        print(f"  🛡️ [ReAct ext] step {step_no} anti-loop veto: {action} {target[:40]}")
                        # transient streak 도 함께 +1 (fatal escalation 가속)
                        _transient_streak += 1
                        if _transient_streak >= 3:
                            final_status = "blocked"
                            error_msg = (
                                f"step {step_no}: anti-loop {_transient_streak}회 반복 — "
                                f"{action} {target[:30]}"
                            )
                            break
                        continue

                # DONE 처리
                if action == "DONE":
                    history.append({
                        "step": step_no, "action": "DONE",
                        "target": target, "result": "task 완료 선언",
                    })
                    if notify_step:
                        try:
                            notify_step(step_no, "DONE", target, "task 완료 선언")
                        except Exception:
                            pass
                    final_status = "done"
                    break

                # act
                try:
                    act_res = await self._act(
                        action=action, target=target, content=content,
                        schema=temp_schema, reason=reason,
                    )
                except Exception as e:
                    final_status = "error"
                    error_msg = f"step {step_no} act 실패: {type(e).__name__}: {str(e)[:80]}"
                    break

                res_str = str(act_res)[:200] if act_res is not None else ""
                history.append({
                    "step": step_no, "action": action, "target": target,
                    "result": res_str,
                })

                # callback
                if notify_step:
                    try:
                        notify_step(step_no, action, target, res_str)
                    except Exception:
                        pass

                # [Fix 2026-05-03] ⚠️ prefix transient/fatal 분기.
                # transient (인덱스 stale / 페이지 변경 / FILL 가시 영역 0개) 는 다음
                # step 의 fresh observe 로 회복 가능 → continue. 연속 3회 이상이면 fatal.
                # fatal (auth/captcha/⛔/sandbox/permission) 은 즉시 break.
                if res_str.startswith("⚠️"):
                    _msg_lower = res_str.lower()
                    _transient_signals = (
                        "인덱스", "캐시에 없", "오래됐", "stale", "index_not_found",
                        "index_stale", "재로드", "다음 step", "편집 가능한 영역",
                        "no_editor", "fill_failed", "click_failed:not_found",
                        "가시적 편집가능", "samples=",
                        # [Fix 2026-05-03] CLICK/FILL 신규 stale 가드 케이스
                        "detached", "invisible", "rect_error", "dom detached",
                        "0 크기", "spa rerender", "모달 닫힘", "fresh 인덱서",
                        "재인덱싱",
                    )
                    _is_transient = any(s in res_str or s in _msg_lower for s in _transient_signals)
                    if _is_transient:
                        # 연속 transient 카운터 — for 루프 진입 전에 0 으로 초기화됨.
                        _transient_streak += 1
                        if _transient_streak >= 3:
                            final_status = "blocked"
                            error_msg = (
                                f"step {step_no}: 연속 {_transient_streak}회 transient ⚠️ — "
                                f"마지막: {res_str[:80]}"
                            )
                            break
                        # transient — history 에는 남기고 다음 step 진행 (fresh observe + 새 [N])
                        print(f"  ⚠️ [ReAct ext] step {step_no} transient (streak={_transient_streak}/3): {res_str[:80]}")
                        continue
                    else:
                        # fatal — 즉시 중단
                        final_status = "blocked"
                        error_msg = f"step {step_no}: {res_str[:120]}"
                        break
                else:
                    # 성공 step — transient streak 리셋
                    _transient_streak = 0
            else:
                # for loop 정상 종료 (max_steps 도달)
                final_status = "done"
                error_msg = f"max_steps {max_steps} 도달"

            return {
                "status": final_status,
                "history": history,
                "outputs": {
                    "react_vars": dict(getattr(temp_schema, "_react_vars", {}) or {}),
                    "first_write": getattr(temp_schema, "_first_write_target", "") or "",
                },
                "step_count": len(history),
                "error": error_msg,
            }

        finally:
            # state 복원
            self._active_schema = original_schema

    async def _react_loop(
        self,
        task_prompt: str,
        goal_ref: str,
        task: Dict,
    ) -> None:
        """드리프트 측정 래퍼. 실제 루프는 _react_loop_impl."""
        try:
            from eidos_drift_metrics import get_collector
            _metrics = get_collector()
            _metrics.start_task(
                task=task,
                schema=self._active_schema,
                task_prompt=task_prompt,
            )
        except Exception as _e_m:
            _metrics = None
            print(f"  ⚠️ [DriftMetrics] start 실패 (무시): {_e_m}")
        try:
            await self._react_loop_impl(task_prompt, goal_ref, task)
        finally:
            if _metrics is not None:
                try:
                    _metrics.finalize_task(
                        task=task, schema=self._active_schema
                    )
                except Exception as _e_f:
                    print(f"  ⚠️ [DriftMetrics] finalize 실패 (무시): {_e_f}")

    def _apply_chain_anchor(self, anchor: str, schema) -> str:
        """[Fix 2026-04-21 Phase 2-B] 체인 stage 인 경우 anchor 앞머리에 원 주제
        키워드+플랫폼+주제 변경 금지 선언을 고정 주입. Why: plan/dev stage 에서
        LLM 이 중간에 다른 상품 카테고리로 갈아타는 드리프트를 막는다. anchor 는
        매 Think 프롬프트의 맨 앞에 깔리므로 long-horizon 방어선이 된다.
        """
        if schema is None or not getattr(schema, "chain_id", ""):
            return anchor
        try:
            from eidos_mission_chain import (
                get_chain_registry,
                _extract_chain_topic_keyword,
            )
            _ch = get_chain_registry().get(schema.chain_id)
            if _ch is None:
                return anchor
            _kw = _extract_chain_topic_keyword(_ch)
            _plat = ""
            try:
                _plat = str(
                    (_ch.metadata or {}).get("platform", "") or ""
                ).strip()
            except Exception:
                _plat = ""
            _lines = []
            if _kw:
                _lines.append(f"[고정 주제] {_kw}")
            if _plat:
                _lines.append(f"[플랫폼] {_plat}")
            if _lines:
                _lines.append(
                    "[주제 변경 금지] 산출물 제목/본문/키워드에 위 '고정 주제' 가 "
                    "반드시 등장해야 한다. 다른 카테고리로 갈아타지 말 것."
                )
                return "\n".join(_lines) + "\n" + anchor
        except Exception as _e_anc:
            print(f"  ⚠️ [ReAct anchor] chain topic 주입 실패 (무시): {_e_anc}")
        return anchor

    # ── [Phase 1-C] Interrupt 훅 — ReAct 루프 cycle 진입 시 검사 ────────
    # wire7_interrupt_enabled on 일 때만 작동. InterruptController 에서
    # urgent_risk 가 뜨면 기존 __pause_reason__ 채널(auth_required 와 동일
    # 패턴)로 태스크를 PAUSE 하고 루프를 종료. attention_shift 는 경고만
    # 남기고 루프는 계속.
    #
    # [Phase 1-C state-aware] task/schema/현재 URL 을 signature_of 가 기대하는
    # dict 형태로 구성해 AttentionBus 의 novelty·uncertainty 축이 실제 기여하게
    # 함. prev 는 task dict 에 캐시(__prev_interrupt_state__) — task 진행에
    # 따라 URL·context 가 바뀌면 attention_shift 가 진짜 작동.
    def _build_interrupt_state(
        self,
        task: Dict,
        schema: Any,
    ) -> Dict[str, Any]:
        """AttentionBus·signature_of 가 읽는 state dict 구성.

        URL-like 키(where/url) 는 signature_of 가 자동 버킷화. task_type/who 는
        그대로 포함. 실패 시 빈 dict 폴백.
        """
        state: Dict[str, Any] = {}
        try:
            if schema is not None:
                tt = getattr(schema, "task_type", "") or ""
                wh = getattr(schema, "where", "") or ""
                who = getattr(schema, "who", "") or ""
                if tt:
                    state["task_type"] = str(tt)
                if wh:
                    state["where"] = str(wh)
                if who:
                    state["who"] = str(who)
        except Exception:
            pass
        try:
            cur_url = self._get_current_browser_url() or ""
            if cur_url:
                state["url"] = str(cur_url)
        except Exception:
            pass
        return state

    def _interrupt_check_and_maybe_pause(
        self,
        task: Dict,
        schema: Any,
        step_count: int,
    ) -> bool:
        """True 반환 시 호출자(_react_loop_impl)는 루프를 즉시 종료해야 함.

        - toggle off → False (no-op, import 도 안 함)
        - urgent_risk → task["__pause_reason__"]="interrupt_urgent_risk"
          + task["status"]="PAUSED" 설정 후 True
        - attention_shift → 경고 로그만, False

        state-aware: task/schema/현재 URL 기반 상태 dict 를 AttentionBus 에
        전달. prev 는 task dict 에 캐시 → URL 전환·드리프트 진전에 따라
        focus 축 변화 감지.
        """
        try:
            from eidos_tuning import get_tuned_param
            if not get_tuned_param("wire7_interrupt_enabled", False):
                return False
        except Exception:
            return False
        try:
            from eidos_interrupt_controller import InterruptController
            ic = InterruptController.get()
            curr_state = self._build_interrupt_state(task, schema)
            prev_state = task.get("__prev_interrupt_state__") or curr_state
            decision = ic.should_interrupt(prev_state, curr_state)
            ic.record(decision)
            # prev 캐시 갱신 — 다음 step 때 이번 state 가 prev 역할
            task["__prev_interrupt_state__"] = curr_state
            if not decision.should_interrupt:
                return False
            if decision.reason == "urgent_risk":
                task["__pause_reason__"] = "interrupt_urgent_risk"
                task["status"] = "PAUSED"
                try:
                    self._report(
                        f"🛑 **[ReAct 중단 — urgent_risk]** "
                        f"risk score {decision.new_score:.2f} "
                        f"(step {step_count}) — 태스크 PAUSED, "
                        f"안정화 후 run_now 로 재개"
                    )
                except Exception:
                    pass
                return True
            # attention_shift 등 — signal-only
            try:
                self._report(
                    f"⚠️ **[Attention Shift]** focus {decision.current_focus} "
                    f"→ {decision.new_focus} (margin {decision.margin:+.2f}) "
                    f"— 루프 계속 (step {step_count})"
                )
            except Exception:
                pass
            return False
        except Exception as e:
            print(f"  ⚠️ [interrupt_check] 실패 (무시): {e}")
            return False

    async def _react_loop_impl(
        self,
        task_prompt: str,
        goal_ref: str,
        task: Dict,
    ) -> None:
        """
        MissionSchema 앵커 기반 ReAct 루프.

        매 cycle:
          1. Observe  — HTML 읽기 + URL 확인 + 클릭 가능 요소 파싱
          2. Think    — 앵커 + 관찰 결과 → LLM이 다음 액션 결정 + 드리프트 자가 점검
          3. Act      — CLICK / FILL / NAVIGATE / WRITE 실행
          4. Verify   — 즉각 결과 검증 + HOW_MUCH 속성 달성 체크

        종료 조건:
          - 모든 HOW_MUCH 속성 달성 (DONE)
          - DONE 액션 선언
          - max_steps 초과
          - timeout 초과
          - 치명적 실패 3회 연속
        """
        schema = self._active_schema
        anchor = self.get_active_anchor() or f"[목표] {task_prompt}"
        anchor = self._apply_chain_anchor(anchor, schema)

        # [P0-2 2026-04-25] 이전 stage force_advance marker → anchor 에 경고 주입.
        # ReAct 가 이전 산출물 (research_report.md 등) 을 신뢰하지 않고 보강하도록 유도.
        if task.get("__prev_stage_force_advanced__"):
            _prev_st = task.get("__prev_stage_name__", "?")
            anchor = (
                f"⚠️ [신뢰 경고] 이전 stage '{_prev_st}' 가 강제 진행으로 끝났다. "
                f"이전 산출물(research_report.md 등)이 비어있거나 부분적일 수 있음. "
                f"필요한 데이터가 없으면 환각으로 채우지 말고 ASK_USER 로 사용자 확인 요청해라.\n\n"
                f"{anchor}"
            )

        # ── general/write/report 타입: ReAct 루프 대신 단일 LLM 직접 호출 ──────
        _DIRECT_TYPES = ("general", "write", "report")
        _loop_task_type = getattr(schema, "task_type", "general") if schema else "general"
        # [P0-1 2026-04-25] 외부 액션 stage(register/deliver/support) 는 task_type='general'
        # 폴백이어도 DirectTask 로 빠지면 안 된다. 실제 등록/발송/대응 액션이 0건인 상태로
        # LLM 텍스트만 뱉고 "완료" 처리되는 거짓 보고가 발생함 (chain=db5ebe78 register stage).
        _EXTERNAL_ACTION_STAGES = {"register", "deliver", "support"}
        _stage_lc = (getattr(schema, "stage", "") or "").lower() if schema else ""
        _is_external_action = _stage_lc in _EXTERNAL_ACTION_STAGES
        # [Autonomous 2026-04-27] 자율 실행 모드 + 외부 액션 stage 진입 시
        # hard-block 키워드 검사. 사용자 사전 승인 카드의 hard_blocks 안에 있는
        # 토큰(결제/송금/회원가입/SNS게시 등) 이 schema.what/how 또는 task_prompt
        # 에 있으면 즉시 run abort + chain abort + telegram 알림.
        # → 자율 모드여도 절대 자동 실행되어선 안 되는 행위 차단.
        if _is_external_action:
            try:
                from eidos_autonomous_runner import AutonomousRunManager as _ARM_HB
                _auto_mgr_hb = _ARM_HB.get()
                _active_hb = _auto_mgr_hb.get_active_run()
                if _active_hb and _auto_mgr_hb.is_task_in_active_scope(task, schema):
                    _hb_blob = " ".join([
                        (getattr(schema, "what", "") or ""),
                        (getattr(schema, "how", "") or ""),
                        (getattr(schema, "where", "") or ""),
                        (task_prompt or ""),
                        (getattr(schema, "original_prompt", "") or ""),
                    ])
                    _hb_kw = _auto_mgr_hb.check_hard_block(_hb_blob)
                    if _hb_kw:
                        _msg = (
                            f"🚨 [Autonomous] HARD-BLOCK 매칭 — '{_hb_kw}' "
                            f"외부 액션 stage='{_stage_lc}' 차단. 자율 실행 abort."
                        )
                        print(_msg)
                        self._report(
                            f"🚨 **[자율 실행 자동 중단]** hard-block 키워드 "
                            f"'{_hb_kw}' 가 외부 액션 stage='{_stage_lc}' 에서 감지됨. "
                            f"안전을 위해 자율 실행 종료 — 사용자 승인이 필요한 작업입니다."
                        )
                        _auto_mgr_hb.abort_active(reason=f"hard_block:{_hb_kw}")
                        # chain abort best-effort
                        try:
                            from eidos_mission_chain import (
                                get_registry as _get_chain_reg_hb,
                                ChainStatus as _CS_HB,
                            )
                            _cid_hb = (
                                (getattr(schema, "chain_id", "") or task.get("chain_id", ""))
                            )
                            if _cid_hb:
                                _reg_hb = _get_chain_reg_hb()
                                _ch_hb = _reg_hb.get(_cid_hb)
                                if _ch_hb and _ch_hb.status == _CS_HB.RUNNING:
                                    _reg_hb.abort(_cid_hb, reason=f"autonomous_hard_block:{_hb_kw}")
                        except Exception as _e_cab:
                            print(f"  ⚠️ [Autonomous] chain abort 실패 (무시): {_e_cab}")
                        # telegram best-effort
                        try:
                            from eidos_telegram_bot import get_bot as _get_bot_hb
                            _bot_hb = _get_bot_hb()
                            if _bot_hb.is_configured():
                                asyncio.ensure_future(
                                    _bot_hb.send_notification(
                                        f"🚨 자율 실행 자동 중단\n"
                                        f"hard-block: <code>{_hb_kw}</code>\n"
                                        f"외부 액션 stage='{_stage_lc}' — 사용자 승인 필요.",
                                        title="🚨 자율 실행 — HARD-BLOCK 감지",
                                    )
                                )
                        except Exception:
                            pass
                        task["status"] = "FAILED"
                        task["fail_reason"] = f"autonomous_hard_block:{_hb_kw}"
                        try:
                            self._remove_task_from_schedule(task)
                        except Exception:
                            pass
                        return
            except Exception as _e_hb:
                # hard-block 검사 실패는 안전 폴백 — 기존 라우팅 로직 그대로
                print(f"  ⚠️ [Autonomous] hard-block 검사 실패 (무시): {_e_hb}")
        # [2026-04-27 WHERE-aware Routing] schema.where + original_prompt 에 외부
        # 도메인이 있으면 task_type='write'/'general'/'report' 이어도 DirectTask 차단.
        # WHERE 도메인 보정 (eidos_mission_schema 2026-04-27) 와 짝을 이루는 라우팅 가드.
        # Why: 보정으로 WHERE 가 도메인이 돼도 task_type 이 write 면 여전히 DirectTask 행 →
        # "브라우저 없이 지식만으로" 프롬프트로 환각 작문.
        # [2026-04-27 Research-intent 안전망] 도메인 토큰 0개 + 연구 산출물 키워드
        # ('기획안/시장조사/벤치마크/case study' 등) 매칭 시에도 DirectTask 차단 —
        # WHERE 자동 주입(mission_schema)이 발동 못 한 캐시/외부-셋 schema 대비.
        _has_ext_src = _has_external_source(schema)
        _has_research_src = _has_research_intent_in_schema(schema)
        _has_action_src = _has_external_action_intent_in_schema(schema)
        if (
            _loop_task_type in _DIRECT_TYPES
            and not _is_external_action
            and not _has_ext_src
            and not _has_research_src
            and not _has_action_src
        ):
            await self._run_direct_llm_task(task_prompt, goal_ref, task, schema, anchor)
            return
        if _is_external_action and _loop_task_type in _DIRECT_TYPES:
            print(
                f"  🔒 [Router HARD] 외부 액션 stage '{_stage_lc}' (task_type='{_loop_task_type}') "
                f"→ DirectTask 차단, ReAct 강제"
            )
            # ReAct 진입 시 LLM 이 실제 외부 액션 없이 끝내려는 시도를 추적
            # (P0-1-b) — _on_react_done 에서 external_action_count==0 이면 stage_failed.
            task["__external_action_stage__"] = _stage_lc
            task.setdefault("__external_action_count__", 0)
        elif _has_ext_src and _loop_task_type in _DIRECT_TYPES:
            _w_preview = (getattr(schema, "where", "") or "")[:50]
            print(
                f"  🔒 [Router HARD] WHERE 외부 도메인 감지 (task_type='{_loop_task_type}', "
                f"where='{_w_preview}') → DirectTask 차단, ReAct 강제"
            )
        elif _has_research_src and _loop_task_type in _DIRECT_TYPES:
            _w_preview = (getattr(schema, "where", "") or "")[:50]
            print(
                f"  🔒 [Router HARD] 연구-기반 산출물 키워드 감지 (task_type='{_loop_task_type}', "
                f"where='{_w_preview}') → DirectTask 차단, ReAct 강제"
            )
        elif _has_action_src and _loop_task_type in _DIRECT_TYPES:
            _w_preview = (getattr(schema, "where", "") or "")[:50]
            print(
                f"  🔒 [Router HARD] 외부-행동 키워드 감지 (task_type='{_loop_task_type}', "
                f"where='{_w_preview}') → DirectTask 차단, ReAct 강제"
            )

        # [Phase 1-ε 2026-05-01] schema.execution_action 있는 단일 task 도 외부 액션
        # 카운터 활성화 — Auto-DONE 가드 / _on_react_done 종료 검증의 입력 신호.
        # chain stage 의 `__external_action_stage__` 마킹(line 3190)과 같은 효과.
        # 차이: stage 명 대신 `schema_external:<action>` prefix 박아 chain stage 와 구분.
        try:
            _ea_at_entry = getattr(schema, "execution_action", None) if schema else None
            if (
                isinstance(_ea_at_entry, dict)
                and _ea_at_entry.get("action")
                and not task.get("__external_action_stage__")
            ):
                task["__external_action_stage__"] = f"schema_external:{_ea_at_entry.get('action')}"
                task.setdefault("__external_action_count__", 0)
                print(
                    f"  🚀 [Phase 1-ε] schema.execution_action 활성 → 외부 액션 카운터 enable "
                    f"(action='{_ea_at_entry.get('action')}')"
                )
        except Exception as _e_ea_entry:
            print(f"  ⚠️ [Phase 1-ε] execution_action entry hook 실패 (무시): {_e_ea_entry}")

        # ── code_build 타입: MissionChain dev 단계 실전 집행 ─────────────────
        # 기획(plan) 산출물을 받아 실제 파일을 디스크에 생성. DCReasoner 스캐폴딩
        # 혹은 assets 문서 생성. 완료 후 _on_react_done 호출로 체인 진행.
        if _loop_task_type == "code_build":
            try:
                from eidos_code_builder import run_code_build_for_schema
                _cb = self._report_cb
                _cb_outputs = await run_code_build_for_schema(
                    schema=schema, task=task, core=self.core, report_cb=_cb,
                )
                # [P2-7] 결과 프리뷰 카드 발행 + 슬롯 세팅
                try:
                    self._last_codebuild_result = {
                        "project_dir": _cb_outputs.get("project_dir", ""),
                        "files": dict(_cb_outputs.get("files", {}) or {}),
                        "entry_file": _cb_outputs.get("entry_file", ""),
                        "mode": _cb_outputs.get("mode", ""),
                        "created_at": time.time(),
                        "chain_id": getattr(schema, "chain_id", "") or "",
                        "stage_index": int(getattr(schema, "stage_index", 0) or 0),
                    }
                    self._emit_codebuild_result_card()
                except Exception as _e_card:
                    print(f"  ⚠️ [CodeBuilder] 결과 카드 발행 실패 (무시): {_e_card}")

                task["status"] = "DONE"
                self._remove_task_from_schedule(task)
                await self._on_react_done(task_prompt, goal_ref, task, [], schema)
            except Exception as e:
                import traceback
                print(f"  ⚠️ [CodeBuilder] 집행 실패: {e}")
                traceback.print_exc()
                task["status"] = "FAILED"
                self._remove_task_from_schedule(task)
            return

        # timeout 및 max_steps 스키마에서 파생
        timeout_sec = getattr(schema, "timeout_sec", 600.0) if schema else 600.0
        # [Fix 2026-04-24] 35 → 60 최소치. 평균 step 15s → 12s (실측 10.8s).
        # chain research stage 가 20+ 경쟁상품 NAVIGATE+READ_PAGE 스캔(2step/gig)
        # 을 하려면 40스텝 예산으로 12개에서 끊기는 문제 차단.
        max_steps   = max(60, int(timeout_sec / 12))  # 평균 step 12초 기준, 최소 60
        # [Fix A 2026-05-06] simple-task fast-path — max_steps cap 15.
        # Why: "강남역 맛집 5곳" 같은 단순 검색에 60 step + 600s 한도가 너무 관대 →
        # 9:05~9:18 14분간 44+ step NAVIGATE/CLICK 만 도는 패턴(최신로그.txt). 키워드
        # 매칭 시 cap 15 + first NAVIGATE seed 강제(아래 pre-nav 분기에서 처리).
        _simple_task_match = _is_simple_task(task_prompt) or (
            _is_simple_task(getattr(schema, "what", "") or "") if schema else False
        )
        if _simple_task_match:
            max_steps = min(max_steps, 15)
            task["__simple_task__"] = True
            print(f"  🎯 [Simple-Task] fast-path 진입 — max_steps cap = {max_steps}")
        started_at  = time.time()

        # [Fix 2026-04-22] ReAct 루프는 plan 객체가 없다. 변수는 schema._react_vars
        # 에만 산다. 이전엔 plan 을 정의 없이 참조해 매 step verified 경로에서
        # _auto_check_attributes(plan=plan) 가 NameError 를 터뜨리고 루프가 침묵
        # 사망 → ad-hoc 태스크가 WRITE/FILL 후 좀비로 hang (로그 2026-04-22 00:43,
        # 00:45 "오늘의 운세"·"튜링 테스트" 둘 다 동일 증상). plan=None 으로
        # 초기화해 호출부는 react_vars 폴백을 타게 한다.
        plan = None

        step_count    = 0
        # [Fix C 2026-05-06] 누적 step 카운터 — 같은 task 의 모든 ReAct cycle 합산.
        # Why: ASK_USER → Watchdog run_now() 재시작 시 step_count 만 리셋되고 cycle
        # 끝없이 반복(최신로그.txt 09:05~09:18 14분간 44+ step). task 단위 누적 +
        # cap 도달 시 forced WRITE 로 무한 루프 차단.
        _cum_react_prev = int(task.get("__cumulative_react_steps__", 0) or 0)
        fail_streak   = 0          # 연속 실패 카운터
        MAX_FAIL_STREAK = 3        # [Fix] 9→3: 조기 ASK_USER 진입 (좌표 폭격 방지)
        _last_fail_msg = ""        # 8-4: 마지막 실패 원인 추적
        _fail_counter: Dict[str, int] = {}  # 11-1: action+target별 실패 횟수
        SAME_TARGET_FAIL_LIMIT = 10          # 11-1: 동일 target 연속 실패 허용 횟수
        # [Fix] verified=True 지만 같은 URL을 반복 NAVIGATE 하는 제자리걸음 감지
        _nav_counter: Dict[str, int] = {}
        SAME_NAV_LIMIT = 3
        # [Fix 2026-04-19] 같은 target에 WRITE가 반복되는 무한 루프 감지.
        # 증상: HOW_MUCH 속성이 이름상 자동 달성 패턴에 매칭 안 돼서("목록"/"분석"/"보고서" 등)
        # LLM이 같은 파일에 계속 WRITE를 발주 → Shortening Guard와 별도로 루프 차단 레이어 필요.
        _write_counter: Dict[str, int] = {}
        SAME_WRITE_HINT_AT   = 2   # 2회: 힌트 주입 + content-name HOW_MUCH 자동 달성
        SAME_WRITE_FORCE_DONE = 3  # 3회: 다음 step에서 DONE 강제
        # [Fix] CLICK 후 DOM 변화 없이 동일 인덱스/셀렉터 재클릭되는 무한 루프 감지.
        # 키: target(예: "[44]") 또는 셀렉터 앞 60자. 값: 연속 no-op 횟수.
        _click_noop_counter: Dict[str, int] = {}
        CLICK_NOOP_HINT_AT   = 1    # 1회 no-op에서도 즉시 힌트 주입 (재클릭 억제)
        CLICK_NOOP_ESCALATE  = 2    # 2회 연속: 사전 차단 + stuck 에스컬레이션
        # [Fix 2026-04-20] pre-block이 같은 target에 반복되는 데드락 감지.
        # 증상: CLICK [6] no-op → pre-block → 힌트 주입 → LLM이 다시 [6] → pre-block …
        # N회 pre-block이 같은 target에 쌓이고 해당 element가 로그인 보호 행위면
        # auth_required로 승격해서 Layer 1 ASK_USER 경로로 보낸다.
        _click_preblock_counter: Dict[str, int] = {}
        CLICK_PREBLOCK_AUTH_ESCALATE = 2  # pre-block 2회 이상 → 로그인 추정 승격
        _force_scroll_next = False  # 다음 iteration 시작 전에 스크롤 강제
        # [Fix] 같은 target에 FILL이 반복 실패(FILL_FAILED/편집 영역 없음)하는 루프 감지.
        # 예: 네이버에서 나무위키를 FILL로 때려박다 실패 → 같은 [5]로 3회 반복 → ASK_USER.
        # CLICK no-op과 달리 FILL 실패는 즉시 JS 반환값으로 판정 가능.
        _fill_noop_counter: Dict[str, int] = {}
        FILL_NOOP_HINT_AT  = 1   # 1회 실패에도 즉시 힌트 (FILL은 실패가 곧 정보 — 관용 없음)
        FILL_NOOP_ESCALATE = 2   # 2회 연속: 강한 차단 + 전략 변경 강제
        history: list  = []       # ContextStore — Think 단계에 주입
        self._drift_restarts = 0  # [Fix] 새 태스크 진입 시 DriftGuard 재시작 카운터 리셋
        _drift_block_attempts = 0  # [Fix] WHERE 범위 이탈 복귀 시도 카운터
        DRIFT_BLOCK_LIMIT = 3      # 같은 URL 복귀 3연속 실패 시 task FAILED

        # ── [Stuck Detector] 막힘 감지 — 페이지 지문 반복 + 속성 진전 없음 ───
        # 지문이 연속 같거나, N step 동안 HOW_MUCH 속성이 하나도 안 올라가면
        # "막혔다"고 인지하고 LLM에 GO_BACK/경로 변경을 강하게 권고.
        _last_progress_step: int = 0
        _prev_page_hash: str = ""
        _same_hash_count: int = 0
        STUCK_HASH_REPEATS = 3
        STUCK_NO_PROGRESS_STEPS = 5
        _stuck_announced_at: int = -10  # 마지막 막힘 보고 step (연속 보고 억제)

        # [Fix] ASK_USER 이후 재개 시 직전 history/step 복원
        # 저장만 되고 복원이 없어 재개해도 Step 1부터 다시 계획하던 문제 수정.
        # LLM이 이전 시도를 참조해 같은 실패 경로를 반복 탐색하지 않도록.
        _resumed_history = task.pop("__react_history__", None)
        _resumed_step    = task.pop("__react_step__", None)
        _is_resume = False
        if isinstance(_resumed_history, list) and _resumed_history:
            history = _resumed_history
            _is_resume = True
            if isinstance(_resumed_step, int) and _resumed_step > 0:
                step_count = _resumed_step  # 이번 루프 진입 시 +1 되어 다음 step 번호로 이어짐

        self._report(
            f"🔄 **[ReAct 루프 시작]** `{task_prompt[:60]}`\n"
            f"└─ max_steps={max_steps} | timeout={timeout_sec:.0f}s"
            + (f"\n└─ 🔁 이어하기: history {len(history)}개 step 복원 (다음 Step {step_count + 1})"
               if _is_resume else "")
        )

        # [DriftGuard] CausalMatrix 기반 드리프트 감지기 초기화
        _drift_guard = None
        try:
            trainer = getattr(self.core, "_causal_trainer", None)
            if trainer is not None and schema is not None \
                    and hasattr(schema, "drift_score_from_causal"):
                from eidos_causal_matrix import DriftGuard
                _drift_guard = DriftGuard(trainer, schema)
                print(f"  [DriftGuard] 초기화 완료 "
                      f"(Y={trainer.n_y}개, X={trainer.n_x}개)")
        except Exception as _e_dg:
            print(f"  [DriftGuard] 초기화 실패 (무시): {_e_dg}")

        # ── [초기 도메인 선점 NAVIGATE] WHERE 드리프트 완화 ──────────────
        # 이전 세션의 브라우저 URL(naver.com 등)이 남아있으면 첫 _observe()가
        # 바로 WHERE 범위 이탈로 판정돼 드리프트 복구 1/3 로 시간이 빠진다.
        # task_prompt / schema.where에서 타겟 도메인을 추출할 수 있으면
        # 첫 step 전에 홈 URL로 선제 이동해서 그 사이클을 건너뛴다.
        # (browser_read + 그 외 브라우저 태스크 공통)
        if not _is_resume and _loop_task_type not in _DIRECT_TYPES:
            try:
                _target_dom = self._extract_target_domain(task_prompt, schema)
                # [Fix A 2026-05-06] simple-task 면서 도메인 미명시 → 통합검색 seed.
                # Why: map.naver.com 같은 무거운 SPA 의 iframe 안에서 SCROLL/CLICK
                # 연속 실패하던 패턴(최신로그.txt 9:06~9:13). search.naver.com 의 정적
                # HTML 이 훨씬 잘 긁힘 — 9:18 Step 44 에서 EIDOS 가 직접 발견한 전략.
                # 이걸 첫 step 으로 강제해 "감 잡기" 38 step → 0 step 으로 단축.
                _seed_search_url = ""
                if not _target_dom and task.get("__simple_task__"):
                    try:
                        import urllib.parse as _ulp_seed
                        _q = _ulp_seed.quote((task_prompt or "")[:80])
                        _seed_search_url = (
                            f"https://search.naver.com/search.naver?query={_q}"
                        )
                        _target_dom = "search.naver.com"
                        print(f"  🎯 [Simple-Task] search seed: {_seed_search_url[:80]}")
                    except Exception as _e_seed:
                        print(f"  [Simple-Task] seed URL 생성 실패 (무시): {_e_seed}")
                _cur_u_start = (self._get_current_browser_url() or "").lower()
                if _target_dom and _target_dom not in _cur_u_start:
                    _boot_url = _seed_search_url or f"https://{_target_dom}"
                    print(f"  🧭 [Pre-nav] 초기 WHERE 선점: 브라우저 '{_cur_u_start[:50]}' "
                          f"→ '{_boot_url}' (target_domain 감지)")
                    self._report(f"🧭 **[초기 이동]** 대상 도메인 선점 → {_boot_url}")
                    try:
                        from execution_module import navigate_and_wait as _pre_nav
                        await asyncio.wait_for(_pre_nav(url=_boot_url, timeout=15.0), timeout=20.0)
                    except Exception as _pn_e:
                        print(f"  [Pre-nav] 실패 (무시하고 루프 진입): {_pn_e}")
            except Exception as _pd_e:
                print(f"  [Pre-nav] 타겟 도메인 추출 실패 (무시): {_pd_e}")

        # ── [P2-2 Pre-Auth Gate 2026-04-25] register/deliver/support stage 진입 시
        # 로그인 상태 prevalidate. ──────────────────────────────────────────────
        # Why: 외부 액션 stage 진입 직후 LLM 이 비로그인 상태를 발견하면 로그인
        # 페이지로 NAVIGATE → ID/PW 필드 못 찾음 → ASK_USER → 사용자 무응답 →
        # idle 폭주 패턴 (chain=5d755586 register stage 228s idle).
        # Pre-nav 직후 obs 1회 찍어서 auth_required 면 ReAct 루프 시작 안 하고
        # 곧장 ASK_USER + Telegram. 사용자가 브라우저에서 로그인 후 ⚡ 자율 실행
        # 누르면 AuthRecheck 분기(L1320)가 재검증하고 정상 진행.
        # task_type 무관: register+general 처럼 L2570 에서 강제 ReAct 로 들어온 경우도
        # 동일 진입 비로그인 패턴이라 함께 차단.
        if not _is_resume and _is_external_action:
            try:
                _pre_auth_obs = await self._observe()
                if _pre_auth_obs.get("auth_required"):
                    _cur_url_pa = (_pre_auth_obs.get("url") or "")[:80]
                    _msg_pa = (
                        f"🔐 **[Pre-Auth Gate]** stage='{_stage_lc}' 진입 시 비로그인 "
                        f"감지 — ReAct 루프 시작 전 로그인 대기.\n"
                        f"└ 현재 URL: {_cur_url_pa}\n"
                        f"└ 브라우저에서 직접 로그인한 뒤 **⚡ 자율 실행** 을 눌러주세요."
                    )
                    print(
                        f"  🔐 [P2-2 Pre-Auth Gate] stage='{_stage_lc}' "
                        f"auth_required=True → ReAct 진입 차단, ASK_USER pause"
                    )
                    self._report(_msg_pa)
                    try:
                        await self._notify_telegram_auth(
                            reason=f"{_stage_lc} stage 진입 — 로그인 필요 (Pre-Auth Gate)",
                            url=_cur_url_pa, fails=0,
                        )
                    except Exception as _e_tg_pa:
                        print(f"  [P2-2] Telegram 통지 실패 (무시): {_e_tg_pa}")
                    self._paused_task = task
                    task["status"] = "WAITING_USER"
                    task["__pause_reason__"] = "auth_required"
                    self._wait_for_user = True
                    task.setdefault("__react_history__", [])
                    task.setdefault("__react_step__", 0)
                    return
            except Exception as _e_pa:
                print(f"  ⚠️ [P2-2 Pre-Auth Gate] prevalidate 실패 (무시, 정상 진입): {_e_pa}")

        # ── [Playbook Entry] 2026-04-22 — 사람처럼 click-chain 으로 진입점 도달 ──
        # Why: URL 하드코딩은 플랫폼 UI 변경에 취약(오늘 kmong 로그 참조).
        # Flow: 캐시된 최종 URL 있으면 fast-path → 실패 시 full playbook 실행 →
        #       성공 시 final_url 캐시 → LLM 이 form 채우기부터 이어받음.
        if (
            not _is_resume
            and _loop_task_type not in _DIRECT_TYPES
            and schema is not None
            and getattr(schema, "playbook", None)
        ):
            try:
                _pb_steps = list(getattr(schema, "playbook", []) or [])
                _pb_tt = getattr(schema, "task_type", "") or ""
                from eidos_playbook import (
                    get_cached_entry as _pb_cache_get,
                    update_cache as _pb_cache_set,
                    invalidate_cache as _pb_cache_del,
                    try_fast_path as _pb_fast,
                    execute_playbook as _pb_exec,
                )
                # fast-path: 마지막 verify 를 캐시 검증 기준으로 재활용
                _last_verify = None
                for _s in reversed(_pb_steps):
                    if str(_s.get("type", "")).lower() == "verify":
                        _last_verify = _s
                        break
                _cached_url = _pb_cache_get(_pb_tt, _pb_steps)
                _pb_done = False
                if _cached_url:
                    self._report(
                        f"⚡ **[Playbook Fast-Path]** 캐시된 진입점 재활용 → "
                        f"{_cached_url[:70]}"
                    )
                    _pb_ok = await _pb_fast(_cached_url, verify=_last_verify)
                    if _pb_ok:
                        _pb_done = True
                    else:
                        _pb_cache_del(_pb_tt)
                if not _pb_done:
                    self._report(f"🎬 **[Playbook]** click-chain 실행 ({len(_pb_steps)} 스텝)")
                    _home = f"https://{self._extract_target_domain(task_prompt, schema)}" \
                            if self._extract_target_domain(task_prompt, schema) else ""
                    _pb_res = await _pb_exec(_pb_steps, home_url=_home, report=self._report)
                    if _pb_res.success and _pb_res.final_url:
                        _pb_cache_set(_pb_tt, _pb_steps, _pb_res.final_url)
                        self._report(
                            f"✅ **[Playbook 성공]** {_pb_res.steps_executed}스텝 완료 → "
                            f"LLM 이 폼 작성부터 이어받음"
                        )
                    else:
                        self._report(
                            f"⚠️ **[Playbook 실패]** step {_pb_res.failed_step_idx+1} "
                            f"— LLM 탐색으로 폴백 (사유: {_pb_res.error[:80]})"
                        )
            except Exception as _e_pb:
                print(f"  [Playbook] 예외 (무시, LLM 탐색으로 폴백): {_e_pb}")

        while self._running:
            step_count += 1
            # [Fix C 2026-05-06] 누적 step 카운터 — task 단위로 합산해 다음 cycle 에 전파.
            _cum_now = _cum_react_prev + step_count
            task["__cumulative_react_steps__"] = _cum_now
            # cap: simple-task 30 / 일반 60. cap 도달 시 _on_react_done 으로 부분 산출물 수확.
            _cum_cap = 30 if task.get("__simple_task__") else 60
            if _cum_now >= _cum_cap:
                self._report(
                    f"🔴 **[누적 step 한도]** task 누적 {_cum_now}/{_cum_cap} 도달 — "
                    f"지금까지 모은 자료로 강제 WRITE → DONE"
                )
                print(
                    f"  🔴 [Fix C] cumulative {_cum_now} >= cap {_cum_cap} "
                    f"(simple={task.get('__simple_task__')}) → forced WRITE via _on_react_done"
                )
                self._remove_task_from_schedule(task)
                try:
                    await self._on_react_done(
                        task_prompt, goal_ref, task, history, schema, _retry_count=1,
                    )
                except Exception as _e_cum:
                    print(f"  [Fix C] _on_react_done 실패 (무시): {_e_cum}")
                return
            # [Phase 2.2 2026-04-25] step 단위 활동 갱신 — 워치독 false alarm 방지.
            # 한 step 의 observe→think→act 사이클이 60s 넘는 케이스(LLM 느림/페이지 로드 30s)
            # 에서도 "정상 진행 중"임을 워치독에게 알림.
            self._touch_activity()
            # [Phase0 metrics] 매 step 카운트
            try:
                from eidos_drift_metrics import get_collector as _gc
                _gc().record_event(task, "step")
            except Exception:
                pass
            # [Phase 1-C] Interrupt 체크 — wire7_interrupt_enabled on +
            # urgent_risk 시 태스크 PAUSE. attention_shift는 경고만.
            if self._interrupt_check_and_maybe_pause(task, schema, step_count):
                self._remove_task_from_schedule(task)
                return
            elapsed = time.time() - started_at

            # ── 종료 조건 체크 ────────────────────────────────────────────
            # [Fix 2026-04-24] 한도 초과 시에도 chain stage/산출물이 있으면 _on_react_done
            # 로 넘겨서 부분 산출물 수확 + 다음 stage 진행을 시도. 이전엔 silent FAILED
            # 로 chain 이 40스텝에서 정체되던 문제.
            _exhausted = (elapsed > timeout_sec) or (step_count > max_steps)
            if _exhausted:
                _reason = "타임아웃" if elapsed > timeout_sec else "스텝 한도"
                _detail = (f"{elapsed:.0f}s 경과"
                           if elapsed > timeout_sec else f"{max_steps}스텝 초과")
                _chain_id = getattr(schema, "chain_id", "") if schema else ""
                _has_outputs = bool(schema and (
                    any(getattr(a, "achieved", False)
                        for a in (getattr(schema, "attributes", []) or []))
                    or any((s.get("action", "") or "").upper() == "WRITE"
                           for s in (getattr(schema, "step_history", []) or []))
                    or any(str(v).strip()
                           for v in (getattr(schema, "_react_vars", {}) or {}).values())
                ))
                if _chain_id or _has_outputs:
                    self._report(
                        f"⚠️ **[{_reason} — 부분 완료 시도]** {_detail} | history={len(history)}\n"
                        f"└─ 현재까지 산출물로 Verifier/체인 진행 위임"
                    )
                    self._remove_task_from_schedule(task)
                    # _retry_count=1 → _on_react_done 내부 재실행 분기 건너뛰고
                    # Verifier 결과에 따라 DONE 또는 pause_with_telegram_gate(300s) 자동 advance.
                    await self._on_react_done(
                        task_prompt, goal_ref, task, history, schema, _retry_count=1,
                    )
                    return
                self._report(f"⚠️ **[{_reason}]** {_detail} — 루프 종료")
                task["status"] = "FAILED"
                self._remove_task_from_schedule(task)
                return
            # ── [Chain Abort] 사용자 "그만/취소" 입력 시 체인 전체 중단. ──
            if self._abort_active_chain and schema is not None and getattr(schema, "chain_id", ""):
                self._abort_active_chain = False
                _cid = getattr(schema, "chain_id", "")
                self._report(
                    f"🛑 **[체인 중단]** 사용자 명령 — chain_id={_cid[:8]} 전체 종료"
                )
                try:
                    from eidos_mission_chain import get_orchestrator
                    _orch = get_orchestrator(core=self.core)
                    if self._report_cb:
                        _orch.set_report_callback(self._report_cb)
                    _orch.abort_chain(_cid, reason="사용자 채팅 중단 명령")
                except Exception as _e_ab:
                    print(f"  ⚠️ [Chain Abort] orchestrator 호출 실패: {_e_ab}")
                task["status"] = "FAILED"
                self._remove_task_from_schedule(task)
                self._active_schema = None
                return
            # ── [Phase A 2026-04-27] Chain abort propagation 가드 ──
            # PRE-VETO/외부 abort_chain → chain.status=ABORTED 인데 ReAct 루프는
            # 모름 → 다음 step 또 실행되던 버그(strikes=5 abort 후 step 21 또 WRITE).
            if schema is not None and getattr(schema, "chain_id", ""):
                try:
                    from eidos_mission_chain import (
                        ChainStatus as _CS_pa,
                        get_orchestrator as _go_pa,
                    )
                    _cid_pa = getattr(schema, "chain_id", "")
                    _ch_pa = _go_pa(core=self.core).registry.get(_cid_pa)
                    if _ch_pa is not None and getattr(_ch_pa, "status", None) == _CS_pa.ABORTED:
                        _reason_pa = (getattr(_ch_pa, "failure_reason", "") or "")[:80]
                        print(
                            f"  🛑 [ChainAbortGuard] step {step_count} 진입 차단 — "
                            f"chain {_cid_pa[:8]} 이미 ABORTED ({_reason_pa})"
                        )
                        self._report(
                            f"🛑 **[체인 종료 감지]** step {step_count} 진입 차단 — "
                            f"chain 이미 ABORTED · 사유: {_reason_pa}"
                        )
                        task["status"] = "FAILED"
                        task["fail_reason"] = f"chain_aborted: {_reason_pa}"
                        self._remove_task_from_schedule(task)
                        self._active_schema = None
                        return
                except Exception:
                    pass
            # ── [Chain Force-Advance] 사용자 "다음단계" 입력 시 즉시 stage 완료 처리. ──
            # _force_advance_chain 플래그가 켜져 있고 현재 schema가 chain 소속이면
            # 부분 산출물을 outputs 로 수확한 채 success 로 마감 → orchestrator 가 다음 stage schema 발급.
            if self._force_advance_chain and schema is not None and getattr(schema, "chain_id", ""):
                self._force_advance_chain = False
                # [Phase 2.1 trace 2026-04-25] force_advance hit 진입 확인.
                print(
                    f"  🔵 [Chain ForceAdvance] react_loop hit — "
                    f"chain={getattr(schema,'chain_id','')[:8]} "
                    f"stage='{getattr(schema,'stage','')}' "
                    f"stage_idx={getattr(schema,'stage_index',0)} "
                    f"history_len={len(history)} "
                    f"→ task DONE → _on_react_done()"
                )
                self._report(
                    f"⏭️ **[강제 진행]** 사용자 명령 — stage='{getattr(schema, 'stage', '')}' "
                    f"({getattr(schema, 'stage_index', 0) + 1}) 강제 완료 → 다음 단계 진행"
                )
                task["status"] = "DONE"
                # [Phase 4-1 2026-04-25] force_advance origin 마커.
                # _on_react_done 가 이 task 를 보면 Verifier/retry/HOW_MUCH 게이트
                # 모두 SKIP 하고 곧장 chain advance 로직으로 점프. 의도가 "이 stage
                # 부분 산출물로 다음 stage 진행" 이므로 history 빈 상태에서 Verifier
                # 가 무조건 FAIL → retry → 같은 stage 재실행 → 무한 루프 차단.
                task["__force_advance_origin__"] = True
                self._remove_task_from_schedule(task)
                await self._on_react_done(task_prompt, goal_ref, task, history, schema)
                return
            # ── [Phase 2 Chain Health 2026-04-26] fail_streak 임계치 health-aware ──
            # health<0.3 (verifier+mcts 종합 신뢰도 30% 미만) 면 MAX_FAIL_STREAK 3→2.
            # 이미 신뢰도 떨어진 chain 에서 한 번 더 실패 = 1 step 빨리 사용자 확인.
            # 토글 OFF 또는 health=1.0 (기본) 이면 변화 없음.
            _eff_max_fail = MAX_FAIL_STREAK
            try:
                from eidos_confidence import get_chain_health as _gch_fs
                _h_fs = _gch_fs(schema)
                if _h_fs < 0.3:
                    _eff_max_fail = max(2, MAX_FAIL_STREAK - 1)
            except Exception:
                pass
            if fail_streak >= _eff_max_fail:
                # [Fix C 2026-05-06] simple-task 면서 충분 자료 모인 경우 ASK_USER 우회 → forced WRITE.
                # Why: "맛집 5곳" 같은 단순 검색에서 fail_streak 도달해도 이미 READ_PAGE
                # 1건 이상 모았으면 그걸로 답하는 게 사용자 경험상 낫다(ASK_USER 무한 대기 X).
                _has_collected = False
                try:
                    _vars_c = (getattr(schema, "_react_vars", {}) or {}) if schema else {}
                    _has_collected = any(
                        isinstance(v, str) and len(v) > 200
                        for v in _vars_c.values()
                    )
                except Exception:
                    pass
                if task.get("__simple_task__") and _has_collected:
                    self._report(
                        f"🟡 **[Simple-Task ASK 우회]** {fail_streak}회 연속 실패지만 "
                        f"수집 자료 있음 — 강제 WRITE 후 DONE"
                    )
                    print(
                        f"  🟡 [Fix C] simple-task ASK 우회 — fail_streak={fail_streak} "
                        f"_react_vars 충분 → forced WRITE via _on_react_done"
                    )
                    self._remove_task_from_schedule(task)
                    try:
                        await self._on_react_done(
                            task_prompt, goal_ref, task, history, schema, _retry_count=1,
                        )
                    except Exception as _e_sk:
                        print(f"  [Fix C] simple-task 우회 _on_react_done 실패 (무시): {_e_sk}")
                    return
                _hl_tag = (
                    f" (health<0.3 가속, 임계 {_eff_max_fail})"
                    if _eff_max_fail < MAX_FAIL_STREAK else ""
                )
                self._report(
                    f"❌ **[연속 실패]** {fail_streak}회 연속 실패 — 사용자 확인 요청{_hl_tag}"
                )
                # 8-4: 마지막 실패 원인을 메시지에 포함
                _fail_detail = f"\n마지막 실패 원인: {_last_fail_msg[:120]}" if _last_fail_msg else ""
                self._report(
                    f"[ASK_USER] ❓ **[확인 필요]** {fail_streak}회 연속 실패했어.{_fail_detail}\n"
                    f"현재 URL: {self._get_current_browser_url()}\n"
                    f"직접 확인 후 **⚡ 자율 실행** 눌러줘."
                )
                self._paused_task = task
                task["status"] = "WAITING_USER"
                self._wait_for_user = True
                return

            # ── [CLICK no-op 에스컬레이션] 직전 step에서 강제 스크롤 요청된 경우 ──
            # 동일 [N] CLICK이 DOM 변화를 못 내고 반복됐을 때 lazy-render 노출 목적.
            if _force_scroll_next:
                _force_scroll_next = False
                try:
                    _sj = ("(function(){var b=window.scrollY;"
                           "window.scrollBy(0,Math.round(window.innerHeight*0.8));"
                           "return 'SCROLLED:'+b+'->'+window.scrollY;})();")
                    _sr = await _browser_run(script=_sj)
                    await asyncio.sleep(0.6)
                    print(f"  🔍 [CLICK no-op] 강제 스크롤 실행 → {str(_sr)[:50]}")
                except Exception as _e_sc:
                    print(f"  [CLICK no-op] 강제 스크롤 실패 (무시): {_e_sc}")

            # ── 1. OBSERVE ────────────────────────────────────────────────
            # general/write/report 태스크는 브라우저 HTML 불필요 → 빈 obs로 스킵
            _NON_BROWSER = ("general", "write", "report")
            _task_type = getattr(schema, "task_type", "general") if schema else "general"
            if _task_type in _NON_BROWSER:
                obs = {"url": "", "clickable": "", "html_len": 0}
            else:
                obs = await self._observe()

                # [P1-3 2026-04-25] 캡챠 재차단 즉시 pivot — 이전에 캡챠 풀었던 host 에서
                # captcha_required 가 다시 뜨면 ChannelRegistry 가 BAN 으로 인식하도록
                # chain_failed reason 에 BAN 키워드 포함시켜 abort_chain.
                try:
                    if obs.get("captcha_required"):
                        from urllib.parse import urlparse as _up_rb
                        _cur_host = (_up_rb(obs.get("url", "") or "").netloc or "").lower()
                        _cid_rb = (task.get("chain_id") or "").strip()
                        if _cur_host and _cid_rb:
                            from eidos_mission_chain import get_orchestrator as _go_rb
                            _ch_rb = _go_rb().registry.get(_cid_rb)
                            _solved_set_rb = (_ch_rb.metadata.get("captcha_solved_hosts") or []) if _ch_rb else []
                            if _cur_host in _solved_set_rb:
                                _sig_rb = obs.get("captcha_signal", "") or "captcha_repeat"
                                self._report(
                                    f"🔴 **[캡챠 재차단 — 즉시 pivot]** host='{_cur_host}' "
                                    f"이전에 캡챠 풀었던 도메인에서 재감지 ({_sig_rb}). "
                                    f"chain abort + ChannelRegistry pivot 신호."
                                )
                                print(
                                    f"  🔴 [P1-3] 캡챠 재차단 감지 — host='{_cur_host}' "
                                    f"in solved_set → abort_chain (BAN 키워드 reason)"
                                )
                                try:
                                    _go_rb().abort_chain(
                                        _cid_rb,
                                        reason=f"bot_block: 캡챠 재차단 보안 확인 ({_sig_rb}) on {_cur_host}",
                                    )
                                except Exception as _e_ab:
                                    print(f"  [P1-3] abort_chain 실패 (무시): {_e_ab}")
                                task["status"] = "FAILED_BLOCKED"
                                self._remove_task_from_schedule(task)
                                return
                except Exception as _e_rb:
                    print(f"  [P1-3] 캡챠 재차단 체크 실패 (무시): {_e_rb}")

                # [Stage 1 P1 2026-04-22] NAVIGATE soft warning(ok=False) 후속 처리.
                # Why: 직전 NAVIGATE 가 ok=False(SPA hydration 실패/subresource 오류)
                # 였고 그 직후 _observe() 가 같은 URL 을 unhydrated_spa 로 잡으면,
                # 이는 "비로그인" 이 아니라 "URL 깨짐". auth_required 분류로 두면
                # ASK_USER → AuthRecheck PASS → 같은 NAVIGATE → 또 unhydrated → 무한
                # 루프(2026-04-22 크몽 로그). 여기서 nav_failed 로 재분류 + dead-end
                # 등록 → 다음 NAVIGATE 자동 차단 → ReAct 가 다른 경로 모색.
                try:
                    _lnw = (getattr(self._active_schema, "_last_nav_warning", None)
                            if self._active_schema else None)
                    if (_lnw and obs.get("auth_required")
                            and obs.get("unhydrated")):
                        _cur_obs_url = (obs.get("url") or "").rstrip("/").lower()
                        _lnw_url = (_lnw.get("url") or "").rstrip("/").lower()
                        _age = time.time() - float(_lnw.get("ts", 0) or 0)
                        if _cur_obs_url and _cur_obs_url == _lnw_url and _age < 30:
                            print(
                                f"  ⚠️ [Stage 1] auth_required→nav_failed 재분류 — "
                                f"직전 NAVIGATE ok=False 경고 + unhydrated 일치 "
                                f"(url={_cur_obs_url[:80]}, age={_age:.1f}s)"
                            )
                            obs["auth_required"] = False
                            obs["nav_failed"] = True
                            obs["nav_failed_url"] = _lnw_url
                            obs["nav_failed_reason"] = "ok=False + unhydrated_spa"
                            # [Fix 2026-04-22] auth_blacklist 보조 등록 —
                            # nav_failed 재분류는 auth_required 경로를 우회하므로
                            # __auth_blacklist__ 에 안 쌓여 Think 프롬프트 노출이 안 됨.
                            # 같은 URL 이 unhydrated_spa 로 2회 이상 재분류되면 이는
                            # 일시적 로드 실패가 아니라 진짜 인증 필요 경로 → blacklist 등록.
                            try:
                                _rec = getattr(self._active_schema,
                                               "_nav_failed_history", None)
                                if _rec is None:
                                    _rec = {}
                                    self._active_schema._nav_failed_history = _rec
                                _rec[_lnw_url] = int(_rec.get(_lnw_url, 0) or 0) + 1
                                if _rec[_lnw_url] >= 2:
                                    _bl_reclass = task.setdefault("__auth_blacklist__", [])
                                    _pm_r = re.match(r'^https?://[^/]+(/[^?#]*)', _lnw_url)
                                    _path_r = _pm_r.group(1) if _pm_r else ""
                                    if _path_r and _path_r not in _bl_reclass:
                                        _bl_reclass.append(_path_r)
                                        print(
                                            f"  🔐 [Stage 1] auth_blacklist 승격: "
                                            f"{_path_r} (재분류 {_rec[_lnw_url]}회) — "
                                            f"Think 프롬프트에 노출"
                                        )
                            except Exception as _e_bl:
                                print(f"  [Stage 1] auth_bl 보조 등록 실패: {_e_bl}")

                            # dead-end 등록 → NAVIGATE Guard(line 7183) 가 차단
                            # [Fix 2026-04-22] DE Guard가 해제한 URL 은 cooldown 동안
                            # 재등록 금지 → 차단/해제 핑퐁 차단.
                            try:
                                _de = getattr(self._active_schema,
                                              "_dead_end_urls", None)
                                if _de is None:
                                    _de = set()
                                    self._active_schema._dead_end_urls = _de
                                _de_freed = getattr(self._active_schema,
                                                    "_de_freed_at", {}) or {}
                                _freed_until = float(_de_freed.get(_lnw_url, 0) or 0)
                                if _freed_until and time.time() < _freed_until:
                                    print(
                                        f"  🧊 [Stage 1] dead-end 재등록 스킵 (cooldown "
                                        f"{_freed_until - time.time():.0f}s): {_lnw_url[:80]}"
                                    )
                                else:
                                    _de.add(_lnw_url)
                                    print(
                                        f"  🚧 [Stage 1] dead-end 등록: {_lnw_url[:80]} "
                                        f"(총 {len(_de)}개)"
                                    )
                            except Exception as _e_de:
                                print(f"  [Stage 1] dead-end 등록 실패: {_e_de}")
                            # 1회성 처리 — 같은 경고 반복 적용 방지
                            self._active_schema._last_nav_warning = None
                except Exception as _e_nf:
                    print(f"  [Stage 1] nav_failed 재분류 예외 (무시): {_e_nf}")

                # [Stage 2 P1 2026-04-22] nav_failed 자동 복구 — task당 1회.
                # Why: dead-end 등록만으로는 ReAct LLM 이 다음 NAVIGATE 후보를
                # "추측"해야 함. 추측 실패 시 또 깨진 URL 시도. 홈 anchor 기반
                # 발견은 사람의 행동 패턴과 동일하며, 한 번의 LLM 호출로 정확한
                # URL을 얻을 수 있음. task당 1회로 제한해 복구 무한 루프 차단.
                if (obs.get("nav_failed")
                        and not task.get("__recovery_attempted__", False)):
                    task["__recovery_attempted__"] = True
                    _broken = obs.get("nav_failed_url") or ""
                    recovered_url = None
                    try:
                        recovered_url = await self._recover_via_link_discovery(
                            broken_url=_broken,
                            schema=schema,
                        )
                    except Exception as _e_rec:
                        print(f"  [Stage 2] 복구 예외 (무시): {_e_rec}")

                    if recovered_url:
                        # 발견한 URL로 NAVIGATE → 새 obs로 갱신
                        try:
                            _r_nav = await _navigate_and_wait(
                                url=recovered_url, timeout=30.0
                            )
                            await asyncio.sleep(0.8)
                            obs = await self._observe()
                            history.append({
                                "step": step_count,
                                "action": "AUTO_RECOVER",
                                "target": _broken[:60],
                                "result": (
                                    f"link_discovery → {recovered_url[:80]} "
                                    f"({str(_r_nav)[:80]})"
                                ),
                                "verified": True,
                            })
                            self._report(
                                f"🔍 **[자동 복구 성공]** \n"
                                f"├─ 깨진 URL: {_broken[:60]}\n"
                                f"├─ 발견 URL: {recovered_url[:80]}\n"
                                f"└─ ReAct 정상 흐름으로 진행"
                            )
                        except Exception as _e_rn:
                            print(f"  [Stage 2] 복구 NAVIGATE 실패: {_e_rn}")
                    else:
                        # 복구 실패해도 정상 흐름 진행 — dead-end는 이미 등록되어
                        # ReAct LLM 이 다른 URL을 시도할 것
                        self._report(
                            f"⚠️ **[자동 복구 실패]** 홈에서 적합 링크 못 찾음 — "
                            f"ReAct가 대안 URL 탐색."
                        )

                # ── [Lazy-load 대기] 편집 페이지 진입 직후 리치 에디터/iframe이 아직
                # 렌더되지 않아 인덱서 결과가 빈약한 경우, 한 번에 한해 대기 후 재관찰.
                # 조건 엄격하게 — 일반 페이지엔 영향 없도록. task당 URL별 1회만.
                try:
                    _url = (obs.get("url") or "").lower()
                    _elcnt = len(obs.get("elements") or [])
                    _EDIT_URL_PATTERNS = ("/edit", "/editor", "/manage",
                                          "my-gigs/", "/modify")
                    _is_edit_url = bool(_url) and any(p in _url for p in _EDIT_URL_PATTERNS)
                    _lazy_done = task.setdefault("__lazy_wait_done__", set())
                    # setdefault가 list로 복원될 수 있어 set 보장
                    if not isinstance(_lazy_done, set):
                        _lazy_done = set(_lazy_done) if _lazy_done else set()
                        task["__lazy_wait_done__"] = _lazy_done
                    _lz_key = _url.split("?", 1)[0]
                    if (_is_edit_url and _elcnt < 25 and _lz_key and
                            _lz_key not in _lazy_done):
                        _lazy_done.add(_lz_key)
                        print(f"  ⏳ [Lazy-wait] 편집 페이지 element={_elcnt}개 (<25) "
                              f"→ 1.5s 대기 후 재관찰")
                        await asyncio.sleep(1.5)
                        try:
                            _obs2 = await self._observe()
                            _elcnt2 = len(_obs2.get("elements") or [])
                            # 더 풍부해졌을 때만 교체 (후퇴 방지)
                            if _elcnt2 > _elcnt:
                                print(f"  ✅ [Lazy-wait] 재관찰 개선: "
                                      f"{_elcnt} → {_elcnt2}개")
                                obs = _obs2
                            else:
                                print(f"  [Lazy-wait] 재관찰 개선 없음 ({_elcnt2}개) — 원본 유지")
                        except Exception as _e_lw2:
                            print(f"  [Lazy-wait] 재관찰 실패 (무시): {_e_lw2}")
                except Exception as _e_lw:
                    pass  # lazy 로직 실패는 조용히 원본 obs 사용

            # ── [Stuck Detector] 페이지 지문 반복 + 속성 진전 없음 ─────────
            # 기준:
            #   (a) 같은 page_hash 가 STUCK_HASH_REPEATS회 연속
            #   (b) STUCK_NO_PROGRESS_STEPS step 동안 속성 달성 카운트 증가 없음
            # 둘 중 하나라도 걸리면 "막힘"으로 인지해서 history/schema.how/GUI 모두에
            # 표식을 남기고, LLM이 다음 Think에서 GO_BACK 또는 다른 경로를 택하도록 유도.
            if _task_type not in _NON_BROWSER:
                _cur_hash = obs.get("page_hash", "") or ""
                if _cur_hash and _cur_hash == _prev_page_hash:
                    _same_hash_count += 1
                elif _cur_hash:
                    _same_hash_count = 1 if _cur_hash != _prev_page_hash else _same_hash_count
                    if _cur_hash != _prev_page_hash:
                        _same_hash_count = 0
                _prev_page_hash = _cur_hash

                if schema is not None and getattr(schema, "attributes", None):
                    _achieved_now = sum(1 for a in schema.attributes if getattr(a, "achieved", False))
                    _prev_achieved = task.get("__last_achieved_count__", 0)
                    if _achieved_now > _prev_achieved:
                        _last_progress_step = step_count
                        task["__last_achieved_count__"] = _achieved_now
                    elif _last_progress_step == 0:
                        # 첫 iteration 기준선 (아직 아무것도 달성 안 됨)
                        _last_progress_step = step_count

                    _no_progress_steps = step_count - _last_progress_step
                    # [Fix] VERB=READ / browser_read 태스크는 같은 페이지를 의도적으로
                    # 반복 READ_PAGE 하므로 page_hash 동일이 정상. stuck 임계치 상향.
                    # 또한 직전 액션이 READ_PAGE/WRITE면 페이지 변경 의도가 없으므로
                    # 이 step은 hash/progress 카운터에서 제외(이번 평가만 스킵).
                    _verb_is_read = (
                        bool(schema) and getattr(schema, "verb", "") == "READ"
                    ) or _task_type == "browser_read"
                    _last_action = (history[-1].get("action") if history else "") or ""
                    _read_only_step = _last_action in {"READ_PAGE", "WRITE"}
                    if _verb_is_read:
                        _hash_thr = STUCK_HASH_REPEATS * 4   # 3 → 12
                        _prog_thr = STUCK_NO_PROGRESS_STEPS * 4  # 5 → 20
                    else:
                        _hash_thr = STUCK_HASH_REPEATS
                        _prog_thr = STUCK_NO_PROGRESS_STEPS
                    # [Fix 2026-04-19] page_hash가 계속 바뀌면 탐색 활발 → 조기 stuck 오판 방지.
                    # category↔gig 왕복 등 정상 탐색 패턴에서 5 step만에 stuck 발동하던 문제.
                    # Why: 수요조사·경쟁분석처럼 속성 달성까지 다수의 NAVIGATE가 필요한 태스크에서
                    # page_hash가 매 step 바뀌는데도 속성 카운터만 보고 막혔다고 판정했음.
                    if _same_hash_count <= 1 and not _verb_is_read:
                        _prog_thr = _prog_thr * 2  # 5 → 10
                    # ── [Phase 2 Chain Health 2026-04-26] stuck 임계치 health-aware ──
                    # health<0.3 시 임계치 ×0.7 (조기 stuck 인식 → 사용자 개입 가속).
                    # 하한: hash_thr≥2, prog_thr≥3 (false alarm 방지). 토글 OFF 또는
                    # health=1.0 이면 변화 없음. _verb_is_read 모드도 동일 비율 적용.
                    try:
                        from eidos_confidence import get_chain_health as _gch_st
                        _h_st = _gch_st(schema)
                        if _h_st < 0.3:
                            _hash_thr = max(2, int(round(_hash_thr * 0.7)))
                            _prog_thr = max(3, int(round(_prog_thr * 0.7)))
                    except Exception:
                        pass
                    _stuck_hash  = (not _read_only_step) and _same_hash_count >= _hash_thr
                    _stuck_noprog = _no_progress_steps >= _prog_thr
                    _is_stuck = _stuck_hash or _stuck_noprog

                    # 같은 stuck 상태 연속 보고 억제 (최소 3 step 간격)
                    if _is_stuck and (step_count - _stuck_announced_at) >= 3:
                        _stuck_announced_at = step_count
                        _reasons: List[str] = []
                        if _stuck_hash:
                            _reasons.append(f"페이지 지문 {_same_hash_count}회 연속 동일")
                        if _stuck_noprog:
                            _reasons.append(f"{_no_progress_steps} step 동안 속성 진전 없음")
                        _stuck_msg = (
                            f"🛑 [STUCK] 막힘 감지: {' + '.join(_reasons)}. "
                            f"다음 중 하나를 택하라: "
                            f"(1) GO_BACK으로 이전 페이지 복귀 후 다른 링크/버튼 CLICK "
                            f"(2) 전혀 다른 검색 키워드/URL로 NAVIGATE "
                            f"(3) 같은 요소 재시도 금지."
                        )
                        self._report(
                            f"🛑 **[막힘 감지]** {' + '.join(_reasons)}\n"
                            f"└─ GO_BACK 또는 경로 변경 권장"
                        )
                        history.append({
                            "step": step_count,
                            "action": "STUCK_DETECTED",
                            "target": obs.get("url", "")[:80],
                            "result": _stuck_msg[:120],
                            "verified": True,
                        })
                        schema.how = _stuck_msg + "\n\n" + schema.how
                        task["__stuck_detected__"] = True

                        # [2026-04-24] 누적 stuck 카운터 + 진행형 에스컬레이션
                        # 1회: Tier 3 평범 시도 (기존 동작)
                        # 2회: back_attempted 리셋 + hub_revisit 리셋 + 허브도 dead-end 강제
                        # 3회: 부분 산출물로 강제 DONE (무한 루프 탈출)
                        _stuck_cnt = int(task.get("__stuck_count__", 0) or 0) + 1
                        task["__stuck_count__"] = _stuck_cnt
                        print(f"  [Stuck Counter] task 누적 stuck={_stuck_cnt}")
                        if _stuck_cnt >= 3:
                            print(
                                f"  🔴 [Stuck Counter] 3회 누적 — 부분 산출물로 DONE 강제"
                            )
                            self._report(
                                f"🔴 **[막힘 3회 누적]** 루프 탈출 — 지금까지 확보한 "
                                f"변수·산출물로 DONE 강제 (체인 다음 stage로 이관)"
                            )
                            try:
                                await self._on_react_done(
                                    task_prompt, goal_ref, task, history, schema
                                )
                            except Exception as _done_e:
                                print(f"  [Stuck→Done] _on_react_done 실패: {_done_e}")
                            return
                        elif _stuck_cnt == 2:
                            # Tier 3 을 한번 더 돌릴 수 있게 state 리셋
                            _sst = task.get("__strategy_state__") or {}
                            _sst["back_attempted"] = False
                            _sst["_skip_to_back"] = True
                            # 허브 재방문 카운터를 임계치로 밀어올려 이번엔 허브도 dead-end 등록
                            _hubr = task.setdefault("__hub_tier3_revisits__", {})
                            for _hk in list(_hubr.keys()):
                                _hubr[_hk] = max(_hubr[_hk], 2)
                            print(
                                f"  🔁 [Stuck Counter] 2회 — Tier3 재시도 허용 + "
                                f"허브 면제 해제 ({len(_hubr)}개 경로)"
                            )

                        # [Fix] 기존에는 힌트만 주입하고 LLM의 다음 Think에 맡겼으나,
                        # LLM이 같은 CLICK/NAVIGATE를 반복하는 경우가 빈번함.
                        # → 즉시 Tier 3 에스컬레이션을 강제로 돌려 GO_BACK/변형이동을 실행하고
                        #   다음 iteration에서 바뀐 URL 기준으로 재-Observe하도록 continue.
                        try:
                            _auto_esc = await self._try_strategy_escalation(task, schema)
                            if _auto_esc:
                                self._report(
                                    f"🔙 **[자동 에스컬레이션 Tier {_auto_esc['tier']}]** "
                                    f"{_auto_esc['action']} → {str(_auto_esc['target'])[:60]}"
                                )
                                history.append({
                                    "step": step_count,
                                    "action": _auto_esc["action"],
                                    "target": str(_auto_esc["target"])[:80],
                                    "result": _auto_esc["result"][:80],
                                    "verified": True,
                                })
                                await asyncio.sleep(0.5)
                                continue  # 재-Observe 후 다음 step 진행
                        except Exception as _e_esc:
                            print(f"  [Stuck→Escalate] 강제 에스컬레이션 실패 (무시): {_e_esc}")

            # ── [Bot-Block 선제 감지] 안티봇 사이트(쿠팡/아마존 등) 빈 쉘 ──
            # 기준: obs.bot_blocked=True 가 감지되면 즉시 Telegram 으로 대안 경로
            # 3개(네이버쇼핑/다나와/구글쇼핑 등)를 제시하고 태스크 FAILED_BLOCKED 로 중단.
            # Why: 2026-04-19 coupang 로그에서 10스텝 × 291자 HTML 을 반복 관찰하며
            #      Gemini 호출·TTS 14회 공회전. 한 번 감지되면 복구 확률 0 에 가까우므로
            #      빠르게 포기하고 사용자에게 결정권 위임.
            if _task_type not in _NON_BROWSER and obs.get("bot_blocked"):
                _cur_url = obs.get("url", "") or ""
                _reason  = obs.get("bot_block_reason", "shell_only")
                _alts    = _suggest_alternative_routes(_cur_url, task.get("task_prompt", ""))
                if _alts:
                    _alt_lines = "\n".join(
                        f"   • {a['name']}: {a['url']}" for a in _alts[:3]
                    )
                else:
                    _alt_lines = "   • (자동 추천 실패 — 검색어/사이트 변경 권장)"

                self._report(
                    f"🤖 **[봇 차단 감지 — 작업 중단]** `{_cur_url[:80]}`\n"
                    f"└─ 사유: {_reason}\n"
                    f"└─ 대안 경로:\n{_alt_lines}\n"
                    f"└─ 원하시면 위 URL 중 하나로 새 지시를 입력해주세요."
                )
                task["status"] = "FAILED_BLOCKED"
                self._remove_task_from_schedule(task)
                self._paused_task = None
                self._wait_for_user = False
                task["__react_history__"] = history
                task["__react_step__"]    = step_count
                return
            # ────────────────────────────────────────────────────────────────

            # ── [Layer 1] 로그인 필요 상태 선제 감지 — ASK_USER 즉시 에스컬레이션 ──
            # 사이트맵 난사(12스텝 낭비) 방지. 인증 확정된 경로를 세션 블랙리스트에 기록.
            if _task_type not in _NON_BROWSER and obs.get("auth_required"):
                # [Fix P1 2026-04-22] AuthRecheck PASS 직후 grace window 안에서는
                # Layer 1 auth_required 를 자동 무시 (오탐 방지).
                # Why: AuthRecheck (line 943) 가 통과한 시점의 페이지(예: /seller/
                # order-list) 는 정상 로그인 상태이지만, ReAct 가 다음 NAVIGATE 로
                # 잘못된 URL (예: kmong /expert/service/register) 에 가면 SPA 가
                # 30KB placeholder 만 반환 → _is_unhydrated_spa True → auth_required
                # True → ASK_USER → 사용자 또 로그인 응답 → AuthRecheck 또 PASS →
                # 무한 루프 (2026-04-22 로그). 기존 grace (line 2646~) 는 LLM 이
                # 직접 ASK_USER 액션을 발행할 때만 차단하고 Layer 1 직접 경로는
                # 우회하지 못함. 동일 grace 카운터를 Layer 1 진입에도 적용.
                # grace 안에서는 obs["auth_required"] 를 False 로 강제하고 정상
                # 흐름 진행 — URL 이 진짜로 잘못됐으면 drift/timeout 으로 자연 종료.
                _grace_layer1 = int(task.get("__auth_pass_grace__", 0) or 0)
                # [Phase 4-3 2026-04-25] 같은 URL 에서 unhydrated_spa 연속 카운터.
                # grace 가 활성이라도 N회 이상 같은 URL 이 unhydrated 면 진짜 인증/
                # 페이지 깨짐 케이스 → grace 우회 강제 ASK_USER. register stage
                # /my-gigs/new 에서 grace=5 로 5스텝 무시 후에도 LLM 이 unhydrated
                # 페이지에서 CLICK/SCROLL/FILL 무한 시도하던 패턴 차단.
                # [Phase 4-3-1 2026-04-26] high-confidence fast-track. auth-gated
                # AND edit-path AND html>=100KB AND elems<edit_thr 매칭이면 1회
                # 만에 OVERRIDE 발동. /my-gigs/new 같이 명백한 셀러 등록 경로에서
                # 크몽이 137KB placeholder 만 반환하는 케이스를 즉시 잡아 LLM
                # 헛스윙 4-5스텝 (~14K 토큰) 절약.
                _cur_url_full = (obs.get("url") or "").strip()
                _obs_html_len = int(obs.get("html_len", 0) or 0)
                _obs_elem_cnt = len(obs.get("elements") or [])
                # [Fix 2026-05-02] AuthRecheck PASS 직후 grace window 안에서는
                # fast-track 비활성. fast-track 본래 목적은 "로그인 안 한 채 ⚡ 자율
                # 실행 연타" 무한 재시도 차단인데, AuthRecheck PASS 직후는 사용자가
                # 방금 로그인 응답한 상태 → 무한 재시도 우려 0. SPA 정상 hydration
                # 시간 보장 위해 일반 THRESHOLD=3 + grace 5 정상 흐름으로 회귀.
                # 사용자 로그 (2026-05-02 20:57:54) 에서 AuthRecheck PASS 직후
                # /my-gigs/new 가 fast-track 1회만에 OVERRIDE → "로그인했는데 또
                # 로그인하라고" 루프 발생.
                _uh_fast_track = (
                    _obs_html_len >= 100_000
                    and _obs_elem_cnt < _UNHYDRATED_ELEMENT_THRESHOLD_EDIT
                    and _is_auth_gated_path(_cur_url_full)
                    and _is_edit_path(_cur_url_full)
                    and _grace_layer1 == 0
                )
                _UH_FORCE_THRESHOLD = 1 if _uh_fast_track else 3
                _uh_streak = task.get("__unhydrated_streak__", {}) or {}
                if not isinstance(_uh_streak, dict):
                    _uh_streak = {}
                _uh_n = int(_uh_streak.get(_cur_url_full, 0) or 0) + 1
                _uh_streak[_cur_url_full] = _uh_n
                task["__unhydrated_streak__"] = _uh_streak
                # 다른 URL 카운터는 reset (메모리 유지)
                _force_layer1 = _uh_n >= _UH_FORCE_THRESHOLD
                if _force_layer1:
                    _ft_tag = " FAST" if _uh_fast_track else ""
                    print(
                        f"  🚨 [AuthPassGrace OVERRIDE{_ft_tag}] 같은 URL unhydrated_spa "
                        f"{_uh_n}회 연속 (>= {_UH_FORCE_THRESHOLD}) — grace "
                        f"우회 강제 ASK_USER | url={_cur_url_full[:80]} "
                        f"html={_obs_html_len} elems={_obs_elem_cnt}"
                    )
                    _ft_msg = (
                        " (fast-track: 인증경로+편집경로+거대 placeholder)"
                        if _uh_fast_track else ""
                    )
                    self._report(
                        f"🚨 **[unhydrated_spa{_ft_tag} 연속 {_uh_n}회 — 강제 ASK_USER]** "
                        f"`{_cur_url_full[:60]}` 페이지가 React 미하이드레이트 상태로 "
                        f"고정{_ft_msg}. grace window 무시하고 사용자 개입 요청."
                    )
                    # streak 리셋 — 사용자 응답 후 다시 시도 가능
                    _uh_streak[_cur_url_full] = 0
                    # grace 강제 0 + 아래 Layer 1 ASK_USER 분기로 흘러가게 obs.auth_required True 유지
                if (not _force_layer1) and _grace_layer1 > 0:
                    task["__auth_pass_grace__"] = _grace_layer1 - 1
                    _cur_url_g = (obs.get("url") or "")[:80]
                    print(
                        f"  🛡️ [AuthPassGrace] Layer 1 auth_required 오탐 무시 "
                        f"(grace {_grace_layer1}→{_grace_layer1 - 1}, uh_streak={_uh_n}) | "
                        f"url={_cur_url_g}"
                    )
                    obs["auth_required"] = False
                    # Layer 1 ASK_USER 분기 스킵 → 아래 정상 ReAct 흐름으로
                # grace 가 0 이면 기존 로직대로 ASK_USER 발동
            if _task_type not in _NON_BROWSER and obs.get("auth_required"):
                _cur_url = obs.get("url", "") or ""
                _bl = task.setdefault("__auth_blacklist__", [])
                try:
                    _pm = re.match(r'^https?://[^/]+(/[^?#]*)', _cur_url)
                    _cur_path = _pm.group(1) if _pm else ""
                    if _cur_path and _cur_path not in _bl:
                        _bl.append(_cur_path)
                except Exception:
                    pass

                # [Fix] 반복 auth 실패 카운터 — 3회 초과 시 하드 스탑
                # 사용자가 로그인 안 한 채 ⚡ 자율 실행만 연타해 무한 재시도되는 루프 차단.
                _auth_fails = int(task.get("__auth_fail_count__", 0)) + 1
                task["__auth_fail_count__"] = _auth_fails
                _reason = "로그인 리다이렉트" if _is_auth_redirect_url(_cur_url) else "비인증 SPA placeholder"

                if _auth_fails >= 3:
                    self._report(
                        f"🔴 **[로그인 반복 실패 {_auth_fails}회 — 자동 재시도 중단]** {_reason}\n"
                        f"└─ 현재 URL: {_cur_url[:80]}\n"
                        f"└─ 블랙리스트({len(_bl)}개): "
                        f"{', '.join(_bl[:4])}{'...' if len(_bl) > 4 else ''}\n"
                        f"└─ 브라우저에서 실제 로그인을 완료한 뒤 작업을 다시 입력해주세요."
                    )
                    await self._notify_telegram_auth(
                        reason=f"{_reason} — 3회 초과로 자동 재시도 중단",
                        url=_cur_url, fails=_auth_fails,
                    )
                    task["status"] = "FAILED_AUTH"
                    self._remove_task_from_schedule(task)
                    self._paused_task = None
                    self._wait_for_user = False
                    task["__react_history__"] = history
                    task["__react_step__"]    = step_count
                    return

                self._report(
                    f"[ASK_USER] 🔐 **[로그인 필요 {_auth_fails}/3]** {_reason} 감지 — 현재 URL: {_cur_url[:80]}\n"
                    f"└─ 브라우저에서 직접 로그인한 뒤 **⚡ 자율 실행** 을 눌러주세요.\n"
                    f"└─ (사이트맵 추측 탐색은 중단합니다)"
                )
                await self._notify_telegram_auth(
                    reason=_reason, url=_cur_url, fails=_auth_fails,
                )
                self._paused_task = task
                task["status"] = "WAITING_USER"
                task["__pause_reason__"] = "auth_required"
                self._wait_for_user = True
                task["__react_history__"] = history
                task["__react_step__"]    = step_count
                return
            # ────────────────────────────────────────────────────────────────

            # WHERE 범위 이탈 감지 — general/보고서/시장조사 태스크는 브라우저 불필요 → 스킵
            _is_browser_task = (
                schema and
                getattr(schema, "task_type", "general") not in ("general", "write", "report") and
                bool(schema.forbidden_zones)
            )
            if _is_browser_task:
                current_url = obs.get("url", "")
                _drifted = False
                _drift_reason = ""

                for fz in schema.forbidden_zones:
                    fz_lower = fz.lower()

                    # 패턴 1: 외부 도메인 체크
                    if "외부 도메인" in fz or "외부" in fz_lower:
                        # WHERE의 허용 도메인 추출 (kmong.com, gumroad.com, namu.wiki 등)
                        _allowed_domain = ""
                        if schema.where:
                            import re as _re2
                            _dm = _re2.search(r'([\w-]+\.[\w.]+)', schema.where)
                            if _dm:
                                _allowed_domain = _dm.group(1).lower()
                        # ── [Search-Hub Bypass] 정보 수집 허브는 결과 클릭으로 외부 도메인 진입 정상 ──
                        # WHERE 가 검색 엔진/정보 수집 허브로 태깅됐으면 패턴 1 (도메인 이탈)
                        # 감지를 스킵한다. 광고(2)/타 서비스(3) 는 그대로 유효.
                        # 매칭 신호:
                        #   (a) WHERE 본문에 "외부 검색" / "검색 출처" / "정보 출처" /
                        #       "search source" 같은 정보 수집 토큰 매칭, AND
                        #       "외부 행동" 토큰 미매칭 (예약/주문/송금 같은 트랜잭션은 도메인 sticky)
                        #   (b) 또는 _allowed_domain 이 알려진 검색 엔진 (google/bing/duckduckgo 등)
                        _where_text = (schema.where or "")
                        _info_hub_tokens = (
                            "외부 검색", "검색 출처", "정보 출처",
                            "정보 수집", "외부 사이트 — 정보",
                            "search source", "external search",
                        )
                        _is_transactional_where = "외부 행동" in _where_text
                        _has_info_token = any(t in _where_text for t in _info_hub_tokens)
                        _SEARCH_ENGINE_DOMAINS = {
                            "google.com", "google.co.kr",
                            "bing.com", "duckduckgo.com",
                            "yahoo.com", "baidu.com",
                            "search.naver.com", "search.daum.net",
                        }
                        _is_search_hub_domain = _allowed_domain in _SEARCH_ENGINE_DOMAINS
                        _bypass_pattern1 = (
                            (not _is_transactional_where)
                            and (_has_info_token or _is_search_hub_domain)
                        )
                        if _bypass_pattern1:
                            # 진단 로그는 task 당 1회 (스팸 방지)
                            if not getattr(schema, "_drift_search_bypass_logged", False):
                                print(
                                    f"  🔓 [Drift] 검색-허브 WHERE — 도메인 이탈 차단 스킵 "
                                    f"(allowed='{_allowed_domain}', where='{_where_text[:80]}')"
                                )
                                try:
                                    schema._drift_search_bypass_logged = True
                                except Exception:
                                    pass
                            # 이 fz 는 패턴 1 으로 더 검사 안 함 — 다음 fz 로
                            continue
                        # WHERE에서 도메인을 추출하지 못하면 도메인 검사 스킵 (kmong 하드코딩 제거)
                        # [Fix P0-2] www↔apex 자가루프 차단:
                        #   - host 추출 후 'www.' prefix 제거하여 apex 단위로 비교
                        #   - substring 매칭 → 정확한 host 또는 서브도메인 매칭으로 전환
                        if _allowed_domain and current_url and current_url.startswith("http"):
                            try:
                                from urllib.parse import urlparse as _up_dr
                                _cur_host = (_up_dr(current_url).hostname or "").lower()
                            except Exception:
                                _cur_host = ""
                            _allow_norm = _allowed_domain.removeprefix("www.")
                            _cur_norm = _cur_host.removeprefix("www.")
                            _host_ok = bool(_cur_norm) and (
                                _cur_norm == _allow_norm
                                or _cur_norm.endswith("." + _allow_norm)
                            )
                            if not _host_ok:
                                _drifted = True
                                _drift_reason = f"도메인 이탈: {current_url[:60]} (허용: {_allowed_domain})"

                    # 패턴 2: 광고/배너 URL 감지
                    elif "광고" in fz_lower or "배너" in fz_lower or "ad" in fz_lower:
                        _AD_SIGNALS = [
                            "doubleclick", "googlesyndication", "pagead",
                            "ad.naver", "adservice", "adclick",
                            "banner", "bnr.", "ads.",
                        ]
                        if current_url and any(sig in current_url.lower() for sig in _AD_SIGNALS):
                            _drifted = True
                            _drift_reason = f"광고/배너 URL 감지: {current_url[:60]}"

                    # 패턴 3: 타 서비스 페이지
                    elif "타 서비스" in fz_lower or "other" in fz_lower:
                        _SERVICE_BLOCKLIST = [
                            "google.com", "facebook.com",
                            "instagram.com", "youtube.com",
                        ]
                        if current_url and any(s in current_url.lower() for s in _SERVICE_BLOCKLIST):
                            _drifted = True
                            _drift_reason = f"타 서비스 페이지: {current_url[:60]}"

                if _drifted:
                    # [Phase0 metrics] forbidden_zone 히트 기록
                    try:
                        from eidos_drift_metrics import get_collector as _gc
                        _gc().record_event(task, "forbidden_hit", _drift_reason)
                    except Exception:
                        pass
                    # 복귀 URL 결정: schema.where에서 URL 또는 도메인 추출
                    _return_url = ""
                    if schema.where:
                        import re as _re3
                        _url_m = _re3.search(r'https?://[^\s\'"<>()\[\]]+', schema.where)
                        if _url_m:
                            _return_url = _url_m.group().rstrip(').,;:')
                        else:
                            # URL이 명시 안 됐으면 추출된 _allowed_domain으로 https:// 합성
                            _dm2 = _re3.search(r'([\w-]+\.[\w.]+)', schema.where)
                            if _dm2:
                                _return_url = f"https://{_dm2.group(1).lower()}"
                    # 그래도 비어있으면 복귀 네비게이션 자체를 스킵 (kmong 하드코딩 제거)
                    if not _return_url:
                        self._report(
                            f"⚠️ [드리프트] WHERE에 복귀할 URL이 없어 자동 복귀를 스킵합니다 — {_drift_reason}\n"
                            f"└─ schema.where='{(schema.where or '')[:80]}' (URL/도메인 추출 실패)"
                        )
                    else:
                        _drift_block_attempts += 1
                        # [Fix] 상한 초과 — 복귀 반복 실패 시 task FAILED 처리
                        if _drift_block_attempts > DRIFT_BLOCK_LIMIT:
                            self._report(
                                f"🔴 **[드리프트 차단 상한 초과]** "
                                f"{_drift_block_attempts}/{DRIFT_BLOCK_LIMIT} 회 복귀 실패\n"
                                f"└─ {_drift_reason}\n"
                                f"└─ 목표 URL({_return_url}) 로의 이동이 반복 실패 — 작업 중단"
                            )
                            task["status"] = "FAILED"
                            self._remove_task_from_schedule(task)
                            return

                        self._report(
                            f"🚨 **[드리프트 차단 {_drift_block_attempts}/{DRIFT_BLOCK_LIMIT}]** "
                            f"WHERE 범위 이탈 감지\n"
                            f"└─ {_drift_reason}\n"
                            f"└─ {_return_url} 로 복귀합니다."
                        )
                        from execution_module import browser_smart_navigate as _nav
                        await _nav(url=_return_url)
                        await asyncio.sleep(2.0)
                        obs = await self._observe()

                        # [Fix] 복귀 후 URL이 실제로 바뀌었는지 확인 — 성공이면 카운터 리셋
                        # [Fix P0-2] host 추출 + www.normalize → www↔apex 리다이렉트도 성공 인정
                        _post_url = (obs.get("url", "") or "").lower()
                        try:
                            from urllib.parse import urlparse as _up_chk
                            _ret_host = (_up_chk(_return_url).hostname or "").lower()
                            _post_host = (_up_chk(_post_url).hostname or "").lower() if _post_url.startswith("http") else _post_url
                        except Exception:
                            _ret_host = ""
                            _post_host = _post_url
                        _ret_norm = _ret_host.removeprefix("www.")
                        _post_norm = (_post_host or "").removeprefix("www.")
                        if _ret_norm and _post_norm and (
                            _post_norm == _ret_norm or _post_norm.endswith("." + _ret_norm)
                        ):
                            _drift_block_attempts = 0


            # ── 2. THINK ──────────────────────────────────────────────────

            # [DriftGuard] 매 스텝 drift_score 체크 → C 주입 or 재시작
            _drift_correction = None
            if _drift_guard is not None:
                try:
                    # 학습 전(_net is None)이면 score가 항상 0.5 → 의미 없는 YELLOW 방지
                    _trainer = getattr(self.core, "_causal_trainer", None)
                    _dg_result = {"level": "GREEN", "score": 0.0,
                                  "correction": None, "abort": False}
                    if _trainer is not None and getattr(_trainer, "_net", None) is None:
                        pass  # 미학습 상태 → DriftGuard 스킵
                    else:
                        _dg_result = _drift_guard.check(step_no=step_count)

                    if _dg_result["abort"]:
                        # [Phase0 metrics] RED 이벤트 + 재시작 카운트
                        try:
                            from eidos_drift_metrics import get_collector as _gc
                            _gc().record_event(task, "drift_red",
                                               f"score={_dg_result['score']:.3f}")
                            _gc().record_event(task, "restart", "drift_red")
                        except Exception:
                            pass
                        # [Fix] RED 재시작 상한 — 2회 초과 시 무한루프 방지 위해 task FAILED
                        self._drift_restarts += 1
                        if self._drift_restarts > 2:
                            self._report(
                                f"🔴 **[DriftGuard RED 재시작 상한 초과]** "
                                f"drift_score={_dg_result['score']:.3f} — 작업 중단"
                            )
                            task["status"] = "FAILED"
                            self._remove_task_from_schedule(task)
                            return
                        # RED 지속 → 컨텍스트 폐기 + 재시작
                        self._report(
                            f"🔴 **[DriftGuard RED {self._drift_restarts}/2]** "
                            f"drift_score={_dg_result['score']:.3f}\n"
                            f"목표 이탈 지속 감지 — 컨텍스트 초기화 후 재시작합니다."
                        )
                        # history 클리어 (오염된 컨텍스트 폐기)
                        history.clear()
                        # anchor 재주입 (교정 정책 C를 새 anchor로)
                        anchor = (
                            (_dg_result["correction"] or "") + "\n\n"
                            + (self.get_active_anchor() or f"[목표] {task_prompt}")
                        )
                        _drift_guard.reset_red()
                        self._report(
                            f"🔄 **[DriftGuard]** 컨텍스트 초기화 완료 "
                            f"— 교정된 anchor로 재시작"
                        )
                        continue  # 같은 루프 이어서 진행 (새 컨텍스트로)

                    elif _dg_result["correction"]:
                        # YELLOW or 첫 RED → C를 THINK에 주입
                        _drift_correction = _dg_result["correction"]
                        _level = _dg_result["level"]
                        # [Phase0 metrics] YELLOW/first-RED 기록
                        try:
                            from eidos_drift_metrics import get_collector as _gc
                            _ev = "drift_red" if _level == "RED" else "drift_yellow"
                            _gc().record_event(task, _ev,
                                               f"score={_dg_result['score']:.3f}")
                        except Exception:
                            pass
                        self._report(
                            f"⚠️ **[DriftGuard {_level}]** "
                            f"drift_score={_dg_result['score']:.3f} "
                            f"— 교정 정책 주입"
                        )
                except Exception as _e_dg_check:
                    pass  # DriftGuard 오류는 루프를 멈추지 않음

            # Layer 1 auth blacklist를 프롬프트에 노출 — LLM이 막힌 경로 재시도 방지
            obs["auth_blacklist"] = list(task.get("__auth_blacklist__") or [])
            # [2026-04-22] FILL 영구 블랙리스트 — 2회 실패한 (path, target) 조합을
            # 다음 Think 에 명시해 같은 [N] 재지명 차단.
            obs["fill_blacklist"] = list(task.get("__fill_blacklist__") or [])

            # [Phase 3-D] Skill Macro queue 소비 — wire11 on 일 때만
            # queue 가 있으면 LLM _think 호출을 건너뛰고 sub-step 을 직접 사용.
            # 큐 소진 시 다음 cycle 부터 정상 _think 경로 복귀.
            _use_macro = False
            try:
                from eidos_tuning import get_tuned_param as _macrotp
                _use_macro = bool(_macrotp("wire11_skill_macro_enabled", False))
            except Exception:
                pass

            decision = None
            _skill_queue = task.get("__skill_queue__")
            if _use_macro and isinstance(_skill_queue, list) and _skill_queue:
                decision = dict(_skill_queue[0] or {})
                _remaining = _skill_queue[1:]
                task["__skill_queue__"] = _remaining if _remaining else None
                decision.setdefault("reason", "skill macro sub-step")
                self._report(
                    f"⚡ **[Skill Macro]** sub-step 소비 "
                    f"(남은 {len(_remaining)}개, 이번: {decision.get('action', '')})"
                )
            else:
                # ── [H 2026-04-26] Hybrid Reasoning state — 양방향 신호 dict ──
                # 하향식: schema 기반 목표/미달성 속성
                # 상향식: 이 step 에서 계산된 stuck/auth/drift/fill 신호들
                _hybrid_state: Dict[str, Any] = {}
                try:
                    if schema is not None:
                        _hybrid_state["what"] = getattr(schema, "what", "") or ""
                        _attrs_all = getattr(schema, "attributes", []) or []
                        _hybrid_state["pending_attrs"] = [
                            a.name for a in _attrs_all if not getattr(a, "achieved", False)
                        ]
                        _hybrid_state["achieved_count"] = sum(
                            1 for a in _attrs_all if getattr(a, "achieved", False)
                        )
                        _hybrid_state["total_attr_count"] = len(_attrs_all)
                    _hybrid_state["stuck"] = {
                        "is_stuck": bool(locals().get("_is_stuck", False)),
                        "same_hash_count": int(locals().get("_same_hash_count", 0) or 0),
                        "no_progress_steps": int(locals().get("_no_progress_steps", 0) or 0),
                    }
                    _hybrid_state["auth_blacklist"] = list(task.get("__auth_blacklist__") or [])
                    _hybrid_state["fill_blacklist"] = list(task.get("__fill_blacklist__") or [])
                    _hybrid_state["unhydrated_streak"] = dict(task.get("__unhydrated_streak__") or {})
                    _hybrid_state["captcha"] = bool(obs.get("captcha_required") or obs.get("captcha_detected"))
                    if "_dg_result" in locals() and isinstance(_dg_result, dict):
                        _hybrid_state["drift_level"] = _dg_result.get("level")
                        _hybrid_state["drift_score"] = _dg_result.get("score")
                except Exception as _hs_e:
                    print(f"  [Hybrid] state 빌드 실패 (무시): {_hs_e}")
                    _hybrid_state = {}

                decision = await self._think(
                    task_prompt=task_prompt,
                    anchor=anchor,
                    obs=obs,
                    history=history[-6:],   # 최근 6개 step만 (컨텍스트 윈도우 절약)
                    schema=schema,
                    step_no=step_count,
                    drift_correction=_drift_correction,  # C 주입
                    hybrid_state=_hybrid_state,
                )
                # wire11 on + skill_steps 있으면 큐 저장 (첫 step 은 decision 그대로 실행).
                # skill_steps 는 "나머지" sub-steps — 첫 것은 decision 의 action 필드.
                if _use_macro and isinstance(decision, dict):
                    _steps_raw = decision.get("skill_steps")
                    if isinstance(_steps_raw, list) and _steps_raw:
                        _valid = [
                            s for s in _steps_raw
                            if isinstance(s, dict) and s.get("action")
                        ]
                        if _valid:
                            task["__skill_queue__"] = _valid
                            _sid = decision.get("skill_id", "") or "?"
                            self._report(
                                f"⚡ **[Skill Macro]** {len(_valid)}개 sub-step "
                                f"큐 저장 (skill_id={_sid[:40]})"
                            )

            if not decision:
                fail_streak += 1
                print(f"  [ReAct] Think 실패 (streak={fail_streak})")
                await asyncio.sleep(1.0)
                continue

            action  = decision.get("action", "").upper()
            target  = decision.get("target", "")
            content = decision.get("content", "")
            reason  = decision.get("reason", "")

            self._report(
                f"🧠 **[Step {step_count}]** [{action}] {(target or content)[:60]}\n"
                f"└─ 이유: {reason[:80]}"
            )

            # ── [Retry Drift Guard 2026-04-27] retry 중 lock-target WRITE 성공 후 ──
            # NAVIGATE/FILL/CLICK/GO_BACK 발사 차단 → DONE 강제.
            # Why: 최신로그.txt — Retry Step 8 [WRITE 보고서.txt] 성공 + AutoCheck-Exist
            # PASS 직후 LLM 이 Step 9 [FILL 15] / Step 10 [CLICK 14] 검색창 재발사 →
            # 이미 데이터 있는데 검색 흐름 처음부터 다시 시작 → 토큰 ~6k 낭비.
            # retry 의도는 산출물 보강(WRITE 재실행)이지 검색 흐름 재시작이 아님.
            # 회귀 가드: WRITE/READ_PAGE/DONE/ASK_USER 는 계속 허용 (재작성/Verifier
            # 모두 정상). _is_retry_run + _retry_lock_write_done 둘 다 True 일 때만 발동.
            if (schema is not None
                    and getattr(schema, "_is_retry_run", False)
                    and getattr(schema, "_retry_lock_write_done", False)
                    and action in ("NAVIGATE", "FILL", "CLICK", "GO_BACK", "CLICK_XY", "SCROLL")):
                _drift_lock_t = (getattr(schema, "_retry_target_lock", "") or "").strip()
                _drift_orig_action = action
                print(
                    f"  🛡️ [Retry Drift Guard] retry 중 '{_drift_lock_t[:40]}' WRITE 성공 후 "
                    f"{_drift_orig_action} 차단 → DONE 강제."
                )
                self._report(
                    f"🛡️ **[Retry Drift Guard]** `{_drift_lock_t[:30]}` 보강 완료 후 "
                    f"`{_drift_orig_action}` 차단 — Verifier 진입 강제."
                )
                history.append({
                    "step": step_count,
                    "action": _drift_orig_action,
                    "target": str(target)[:60],
                    "result": f"Retry Drift Guard 차단 — DONE 으로 교체",
                    "verified": False,
                })
                action = "DONE"
                target = ""
                content = ""

            # ── [Patch B — Lazy Reason Detective 2026-04-27] ───────────────────
            # LLM 이유 텍스트에 "내부 지식으로/브라우저 없이/검색 없이" 등 자기-위임
            # 신호 매칭 → 해당 step 차단 + schema.how 에 자료 수집 강제 hint prepend.
            # 누적 3회 도달 시 task ASK_USER 폴백 (무한 게으름 차단).
            # 회귀: lazy 토큰 미매칭 시 분기 미발동 → 기존 흐름 100% 보존.
            if (schema is not None
                    and action not in ("DONE", "ASK_USER")
                    and _is_lazy_reason(reason)):
                _lz_n_b = int(getattr(schema, "_lazy_reason_count", 0) or 0) + 1
                try:
                    schema._lazy_reason_count = _lz_n_b
                except Exception:
                    pass
                print(
                    f"  🛡️ [Lazy Reason Guard] LLM 이유에 게으름 토큰 매칭 — "
                    f"{action} 차단 ({_lz_n_b}회차) reason='{(reason or '')[:60]}'"
                )
                self._report(
                    f"🛡️ **[Lazy Reason Guard]** LLM 이유에 게으름 토큰 — `{action}` "
                    f"차단 ({_lz_n_b}/3회차). 자료 수집 강제 hint 주입."
                )
                # schema.how 에 자료 수집 강제 hint prepend (중복 방지)
                _curr_how_lz = (getattr(schema, "how", "") or "")
                if "[자료 수집 강제" not in _curr_how_lz:
                    try:
                        schema.how = (
                            "[자료 수집 강제 — 게으름 차단]\n"
                            "- 외부 검색 의도가 있는 작업은 NAVIGATE → READ_PAGE 단계로 "
                            "자료를 변수에 저장한 후 WRITE 하라.\n"
                            "- '내부 지식으로 작성' / '브라우저 없이' / '검색 없이' 모드 금지 — "
                            "자료가 변수에 0건 상태에서 WRITE 발사 시 dispatcher 가 차단한다.\n"
                            "- forbidden_zones 의 외부 차단 토큰이 prompt 의도와 모순되면 "
                            "무시하라 (이미 schema 단에서 모순 해소됨).\n\n"
                        ) + _curr_how_lz
                    except Exception:
                        pass
                history.append({
                    "step": step_count,
                    "action": action,
                    "target": str(target)[:60],
                    "result": f"Lazy Reason Guard 차단 ({_lz_n_b}/3) — hint 주입",
                    "verified": False,
                })
                # 3회 도달 시 ASK_USER 폴백
                if _lz_n_b >= 3:
                    self._report(
                        f"⚠️ **[Lazy Reason Guard]** 3회 누적 — task 정지. "
                        f"수동 ⚡ 자율 실행 으로 재시도 가능."
                    )
                    self._paused_task = task
                    task["status"] = "WAITING_USER"
                    task["__pause_reason__"] = "lazy_reason_3x"
                    self._wait_for_user = True
                    task["__react_history__"] = history
                    task["__react_step__"] = step_count
                    return
                # step skip — fail_streak 증가 + 다음 LLM 결정으로
                fail_streak = (fail_streak if 'fail_streak' in dir() else 0) + 1
                step_count += 1
                await asyncio.sleep(0.3)
                continue

            # ── [Patch A — Lazy Write Guard 2026-04-27] ────────────────────────
            # WRITE 발사 시 _react_var_urls 0건 + research 의도 prompt → 자료 수집
            # 단계 생략한 환각 dump 거의 확정. action 을 NAVIGATE search URL 로 강제
            # 교체해 LLM 을 검색 흐름으로 되돌림. 누수의 단일 가장 큰 차단점.
            # 회귀: chain task / read_page 결과 1건 이상 / research 의도 미매칭 →
            # 모두 분기 미발동 → 기존 흐름 100% 보존.
            if _is_lazy_write_action(action, schema, task_prompt):
                _lz_n_a = int(getattr(schema, "_lazy_write_count", 0) or 0) + 1
                try:
                    schema._lazy_write_count = _lz_n_a
                except Exception:
                    pass
                _redirect_url = _build_lazy_redirect_url(schema, task_prompt)
                _orig_target = target

                # ── [Patch 2026-04-29] 3회 누적 cap → WAITING_USER ─────────
                # 최신로그 b555e8b6: cap 없어 36회까지 발동 → 60 step 한도 초과 → FAIL.
                # _lazy_reason_count 와 동일하게 _lazy_write_count 도 3회 cap.
                # redirect URL 이 빈 string 인 경우 (placeholder 만) 도 즉시 cap 진입.
                if _lz_n_a >= 3 or not _redirect_url:
                    _why = (
                        "redirect URL 추출 실패 (placeholder 만)" if not _redirect_url
                        else f"{_lz_n_a}회 누적 cap"
                    )
                    print(
                        f"  🛡️ [Lazy Write Guard] WRITE '{(_orig_target or '')[:30]}' "
                        f"{_why} → task WAITING_USER 전환"
                    )
                    self._report(
                        f"🛡️ **[Lazy Write Guard CAP]** `WRITE {(_orig_target or '')[:30]}` "
                        f"{_why}. 사용자 개입 필요 — `[자율 실행]` 또는 작업 변경."
                    )
                    history.append({
                        "step": step_count, "action": "WRITE",
                        "target": str(_orig_target)[:60],
                        "result": f"Lazy Write Guard CAP ({_why})",
                        "verified": False,
                    })
                    task["status"] = "WAITING_USER"
                    task["__pause_reason__"] = "lazy_write_3x"
                    self._paused_task = task
                    step_count += 1
                    await asyncio.sleep(0.3)
                    continue
                # ────────────────────────────────────────────────────────────

                print(
                    f"  🛡️ [Lazy Write Guard] WRITE '{(_orig_target or '')[:30]}' 차단 "
                    f"— 자료 0건 + research 의도. NAVIGATE → {_redirect_url[:80]} 강제 ({_lz_n_a}회차)"
                )
                self._report(
                    f"🛡️ **[Lazy Write Guard]** `WRITE {(_orig_target or '')[:30]}` 차단 — "
                    f"외부 자료 0건 + 검색 의도. `NAVIGATE` 강제: `{_redirect_url[:60]}`"
                )
                # schema.how 에 자료 수집 강제 hint prepend (Patch B 와 같은 토큰)
                _curr_how_lwa = (getattr(schema, "how", "") or "")
                if "[자료 수집 강제" not in _curr_how_lwa:
                    try:
                        schema.how = (
                            "[자료 수집 강제 — Lazy Write 차단]\n"
                            "- 자료가 _react_vars 에 0건 상태에서 WRITE 발사가 차단되었다.\n"
                            "- 강제 NAVIGATE 후 검색 결과 페이지를 READ_PAGE 로 변수에 저장하라.\n"
                            "- 그 후 WRITE 시 변수 내용을 인용·분석해 보고서 작성.\n"
                            "- 다시 WRITE 가 차단되면 NAVIGATE 또는 다른 출처로 재시도.\n\n"
                        ) + _curr_how_lwa
                    except Exception:
                        pass
                history.append({
                    "step": step_count,
                    "action": "WRITE",
                    "target": str(_orig_target)[:60],
                    "result": f"Lazy Write Guard 차단 ({_lz_n_a}회) — NAVIGATE 강제 교체",
                    "verified": False,
                })
                # action 교체
                action = "NAVIGATE"
                target = _redirect_url
                content = ""

            # [P0-1-b 2026-04-25] 외부 액션 stage 진입 표시 — 실제 브라우저 액션
            # (NAVIGATE/CLICK/FILL/SCROLL/GO_BACK) 1건이라도 발생하면 카운터 +1.
            # _on_react_done 에서 0건이면 stage_failed 로 마킹해 chain 거짓 완료 차단.
            if task.get("__external_action_stage__") and action in (
                "NAVIGATE", "CLICK", "FILL", "SCROLL", "GO_BACK", "CLICK_XY",
                "TELEGRAM_SEND",  # T2 Step 3 2026-05-01 — 외부 SDK 호출 verb
                "EMAIL_SEND",     # P-ε 2026-05-01 — SMTP 이메일 발송 verb
            ):
                task["__external_action_count__"] = task.get("__external_action_count__", 0) + 1

            # DONE 선언
            if action == "DONE":
                # [Phase 1-ε 2026-05-01] 명시 DONE 가드 — schema.execution_action 명시 +
                # 외부 액션 0건이면 DONE 거부, ReAct 루프 계속. LLM 이 WRITE 후 DONE 으로
                # 끝내려는 시도 차단 — 실제 외부 행위 (NAVIGATE/CLICK/FILL) 1건 이상 필요.
                _exec_action_done = (
                    getattr(schema, "execution_action", None) if schema else None
                ) or {}
                _ext_c_done = int(task.get("__external_action_count__", 0) or 0)
                if (
                    isinstance(_exec_action_done, dict)
                    and bool(_exec_action_done.get("action"))
                    and _ext_c_done == 0
                ):
                    _done_blocks = int(task.get("__done_blocked_count__", 0) or 0) + 1
                    task["__done_blocked_count__"] = _done_blocks
                    if _done_blocks <= 3:
                        # 3회까지는 차단, ReAct 루프 다음 step 으로 진입 (LLM 외부 행위 시도 기회)
                        if not task.get("__done_block_warn_emitted__"):
                            self._report(
                                f"🚫 **[DONE 거부]** schema.execution_action="
                                f"`{_exec_action_done.get('action')}` 명시 — "
                                f"외부 액션 0건. 외부 행위 (NAVIGATE/CLICK/FILL) 1건 이상 필요."
                            )
                            print(
                                f"  🚫 [Phase 1-ε] 명시 DONE 거부 ({_done_blocks}/3) — "
                                f"execution_action='{_exec_action_done.get('action')}'"
                            )
                            task["__done_block_warn_emitted__"] = True
                        # DONE 무시하고 다음 step 진행 — history 에 안 박힘 (재시도 방해 X)
                        continue
                    else:
                        # 3회 초과 시 무한 루프 방지 — DONE 통과 (외부 행위 미달 stage_failed 마킹은
                        # _on_react_done line 9864 가드가 처리)
                        print(
                            f"  ⚠️ [Phase 1-ε] DONE 차단 한도 초과 ({_done_blocks}회) → "
                            f"DONE 통과 (stage_failed 마킹은 _on_react_done 가드가 처리)"
                        )
                await self._on_react_done(task_prompt, goal_ref, task, history, schema)
                return

            # ── [Lazy Ask Guard 2026-05-03] ASK_USER 게으름 차단 ──────────────
            # research 의도 task 인데 _count_data_vars == 0 (검색 한 번도 안 함)
            # 상태에서 ASK_USER 발사 → 게으른 떠넘김 확정. NAVIGATE 검색 URL 로
            # 강제 교체해 ASK_USER 분기 진입 자체를 막고 아래 NAVIGATE 분기로
            # 흐름 위임. 키워드 매칭 0 — 객관 상태 (자료 var 카운트 + research
            # verb) 만 검사하므로 표현 변경 우회 불가능.
            # Why: 2026-05-03 최신로그.txt — "분석할 경쟁 서비스의 URL을 알려
            # 주세요" 류 ASK_USER 발사. 사용자 피드백 "키워드 트리거 싫어함"
            # (feedback_no_keyword_triggers.md) + 기존 Lazy Write Guard 와 같은
            # 객관 상태 패턴 (project_lazy_action_guard_20260427.md) 일관성.
            # 회귀: research 의도 미매칭 / _count_data_vars >= 1 (이미 검색 후
            # 부족해 묻는 정당한 ASK_USER) / placeholder 만으로 검색 의미 없는
            # 경우 / 3회 누적 cap → 모두 정상 ASK_USER 폴백.
            if action == "ASK_USER" and _is_lazy_ask_action(action, schema, task_prompt):
                _la_n = int(getattr(schema, "_lazy_ask_count", 0) or 0) + 1
                try:
                    schema._lazy_ask_count = _la_n
                except Exception:
                    pass
                _la_redirect = _build_lazy_redirect_url(schema, task_prompt)
                _la_orig_q = (target or content or "")
                # 3회 누적 또는 redirect URL 빈 string 시 → 정상 ASK_USER 폴백
                # (placeholder 만으로 검색 의미 없는 + 검색해도 못 찾는 경우 인정)
                if _la_n >= 3 or not _la_redirect:
                    _la_why = (
                        "redirect URL 추출 실패 (placeholder 만)"
                        if not _la_redirect else f"{_la_n}회 누적 cap"
                    )
                    print(
                        f"  🛡️ [Lazy Ask Guard] ASK_USER {_la_why} → "
                        f"정상 ASK_USER 경로 폴백 (사용자 도움 진짜 필요 인정)"
                    )
                    # action 교체 안 함 → 아래 ASK_USER 분기 진입
                else:
                    print(
                        f"  🛡️ [Lazy Ask Guard] ASK_USER '{_la_orig_q[:30]}' 차단 "
                        f"— 자료 0건 + research 의도. NAVIGATE → "
                        f"{_la_redirect[:80]} 강제 ({_la_n}회차)"
                    )
                    self._report(
                        f"🛡️ **[Lazy Ask Guard]** `ASK_USER {_la_orig_q[:30]}` 차단 — "
                        f"외부 자료 0건 + 검색 의도. `NAVIGATE` 강제: "
                        f"`{_la_redirect[:60]}`"
                    )
                    # schema.how 에 자료 수집 강제 hint prepend (Lazy Write Guard
                    # 와 동일 토큰 — LLM 이 다음 step 에서 검색 흐름 이어감)
                    _curr_how_la = (getattr(schema, "how", "") or "")
                    if "[자료 수집 강제" not in _curr_how_la:
                        try:
                            schema.how = (
                                "[자료 수집 강제 — Lazy Ask 차단]\n"
                                "- 자료가 _react_vars 에 0건 상태에서 ASK_USER "
                                "발사가 차단되었다.\n"
                                "- 강제 NAVIGATE 후 검색 결과 페이지를 READ_PAGE "
                                "로 변수에 저장하라.\n"
                                "- 그 후 후보 URL/회사명/사이트 추출해 후속 "
                                "NAVIGATE 진행.\n"
                                "- 사용자에게 묻기 전에 직접 검색·조사해라.\n\n"
                            ) + _curr_how_la
                        except Exception:
                            pass
                    history.append({
                        "step": step_count,
                        "action": "ASK_USER",
                        "target": str(_la_orig_q)[:60],
                        "result": (
                            f"Lazy Ask Guard 차단 ({_la_n}회) — "
                            f"NAVIGATE 강제 교체"
                        ),
                        "verified": False,
                    })
                    # action 교체 — ASK_USER 분기 진입 안 함 + 아래 NAVIGATE
                    # 분기로 자연스럽게 흐름 위임. Lazy Write Guard 와 같은 패턴.
                    action = "NAVIGATE"
                    target = _la_redirect
                    content = ""

            # ASK_USER — 단, 인간에게 묻기 전 Strategy 에스컬레이션 먼저 시도
            if action == "ASK_USER":
                # [P1-2026-04-19] AuthRecheck 직후 grace window 동안 "로그인 해라"
                # 류 ASK_USER 는 자동 무시한다. SPA hydration 지연으로 LLM이 빈
                # 페이지를 로그인 필요로 오판하는 루프(로그인 상태인데 ASK 반복) 차단.
                _grace = int(task.get("__auth_pass_grace__", 0) or 0)
                if _grace > 0:
                    _ask_text = (str(target) + " " + str(content)).lower()
                    if ("로그인" in _ask_text) or ("login" in _ask_text):
                        task["__auth_pass_grace__"] = _grace - 1
                        print(
                            f"  🛡️ [AuthPassGrace] ASK_USER(로그인) 차단 — "
                            f"grace {_grace}→{_grace - 1}. 탐색 액션으로 전환 유도."
                        )
                        history.append({
                            "step": step_count,
                            "action": "ASK_USER",
                            "target": str(target)[:60],
                            "result": "grace_skip: 로그인 이미 확인됨 — 페이지 읽기/탐색 계속",
                            "verified": False,
                        })
                        await asyncio.sleep(0.3)
                        continue

                # [2026-04-24] LLM safety-training 환각 refusal 가드.
                # Why: Opus/Gemini 가 "제가 직접 수행할 수 없습니다. 로컬 환경에서
                # 실행해야 합니다" 류 답변을 ASK_USER 로 내뿜는 사례 (클로드의 비판.txt
                # + 최신로그.txt 2026-04-24). EIDOS 는 실제 스캐폴딩·브라우저 조작
                # 실행자이므로 refusal 자체가 환각. ASK_USER 발사하면 사용자에게
                # "내가 못함" 을 묻게 되고 3회 누적으로 chain 이 FAILED_ASK_LOOP 로
                # 사망. → 매칭 시 ASK_USER 를 발사하지 않고 history 에 거부 메시지
                # 기입, retry loop 1회 강제 회전. 3회 누적이면 정상 FAILED_ASK_LOOP
                # 경로로 떨어뜨려 사용자에게 상황 공개.
                _refusal_blob = f"{target} {content}"
                _refusal_lower = _refusal_blob.lower()
                _refusal_patterns_ko = (
                    "수행할 수 없", "직접 수행", "로컬 환경에서 실행",
                    "로컬에서 실행", "제가 실행", "제가 직접",
                    "직접 실행할 수", "제가 수행",
                )
                _refusal_patterns_en = (
                    "unable to", "can't directly", "cannot execute",
                    "i cannot", "i can't", "i am not able to",
                )
                _is_refusal = (
                    any(p in _refusal_blob for p in _refusal_patterns_ko)
                    or any(p in _refusal_lower for p in _refusal_patterns_en)
                )
                if _is_refusal:
                    _refusal_count = int(task.get("__refusal_count__", 0) or 0) + 1
                    task["__refusal_count__"] = _refusal_count
                    if _refusal_count < 3:
                        print(
                            f"  🛡️ [Refusal Guard] LLM 환각 refusal 감지 "
                            f"({_refusal_count}/3) — ASK_USER 발사 차단, retry 강제. "
                            f"blob='{_refusal_blob[:100]}'"
                        )
                        history.append({
                            "step": step_count,
                            "action": "ASK_USER",
                            "target": str(target)[:60],
                            "result": (
                                f"refusal_rejected ({_refusal_count}/3): "
                                f"너는 이 시스템의 실행자다. 주어진 액션"
                                f"(NAVIGATE/CLICK/FILL/WRITE/SCAFFOLD 등)으로 "
                                f"직접 수행해라. '내가 못함/로컬에서 해야 한다' "
                                f"류 응답은 환각이며 금지된다. 다른 액션을 선택하라."
                            ),
                            "verified": False,
                        })
                        await asyncio.sleep(0.3)
                        continue
                    # 3회 누적 → 아래 정상 ASK_USER 경로로 흘려보내 FAILED_ASK_LOOP 승격.
                    print(
                        f"  🔴 [Refusal Guard] refusal 3회 누적 — "
                        f"FAILED_ASK_LOOP 경로로 승격 (사용자에게 상황 공개)"
                    )

                # [Fix 2026-04-22] 로그인/인증 관련 ASK_USER 는 Tier 1 SCROLL 을
                # 우회로 삼지 않는다. Why: 2026-04-22 크몽 로그에서 LLM 이 로그인
                # 필요로 ASK_USER 를 발행했는데 Tier 1 이 스크롤만 한 번 돌리고
                # 사용자 질문을 삼켜 버림 → 다음 step NAVIGATE 에서 같은 URL 재시도.
                # 스크롤로는 SPA placeholder 가 로그인 상태로 바뀌지 않으므로 의미가
                # 없고, 사용자에게 로그인 요청 전달이 유일한 해결책이다.
                _ask_blob = (str(target) + " " + str(content)).lower()
                _auth_keywords = ("로그인", "login", "sign in", "로그온",
                                  "로그 인", "signin", "인증", "비로그인", "auth")
                _ask_is_auth = any(_k in _ask_blob for _k in _auth_keywords)
                _auth_bl_has = bool(task.get("__auth_blacklist__"))
                if _ask_is_auth or _auth_bl_has:
                    _skip_reason = ("auth 키워드" if _ask_is_auth
                                    else "auth_blacklist 존재")
                    print(
                        f"  🛡️ [Strategy] ASK_USER 로그인 요청 감지({_skip_reason}) "
                        f"— Tier 1/2 스킵, 사용자 질문 직접 전달"
                    )
                    _escalation = None
                else:
                    _escalation = await self._try_strategy_escalation(task, schema)
                if _escalation:
                    # Tier1(스크롤) 또는 Tier2(사이트맵) 수행됨 → history 기록 후 재-Observe
                    self._report(
                        f"🔍 **[자동 탐색 Tier {_escalation['tier']}]** "
                        f"ASK_USER 전 시도: {_escalation['action']} → {str(_escalation['target'])[:60]}"
                    )
                    history.append({
                        "step": step_count,
                        "action": _escalation["action"],
                        "target": str(_escalation["target"])[:80],
                        "result": _escalation["result"][:80],
                        "verified": True,
                    })
                    # 다음 iteration에서 _observe 재실행 → 새 요소 발견 시 LLM이 CLICK/NAVIGATE로 진행
                    await asyncio.sleep(0.5)
                    continue

                # 에스컬레이션 소진 → 정상 ASK_USER 경로
                question = target or content
                # [Fix 2026-04-20] ASK_USER 반복 카운터 — 사용자 답변 없이
                # 같은 문제로 같은 질문이 3회 이상 튀어나오면 무한 루프로 간주,
                # FAILED_ASK_LOOP 로 하드스탑해 CHAIN 진행도 자동 중단.
                _ask_count = int(task.get("__ask_user_count__", 0) or 0) + 1
                _prev_q    = str(task.get("__ask_user_question__", "") or "")
                _curr_q    = str(question or "")
                _same_q    = (_prev_q[:80] == _curr_q[:80]) if _prev_q else False
                task["__ask_user_count__"] = _ask_count
                if _same_q and _ask_count >= 3:
                    self._report(
                        f"🔴 **[ASK_USER 반복 하드스탑]** 같은 질문이 {_ask_count}회 반복 — 중단.\n"
                        f"└─ 질문: {_curr_q[:160]}\n"
                        f"└─ 브라우저에서 문제를 해결하거나 채팅창에 답변을 입력한 뒤 다시 시작해주세요."
                    )
                    task["status"] = "FAILED_ASK_LOOP"
                    # chain 소속이면 stage_complete(success=False) 로 명시 통지해
                    # 체인이 PAUSED/FAILED 상태로 깨끗이 정리되도록.
                    _chain_id_hs = getattr(schema, "chain_id", "") if schema else ""
                    if _chain_id_hs:
                        try:
                            from eidos_mission_chain import get_orchestrator
                            _orch = get_orchestrator(self.core)
                            if _orch is not None:
                                await _orch.on_stage_complete(
                                    _chain_id_hs, schema, success=False
                                )
                        except Exception as _e_hs_chain:
                            print(
                                f"  ⚠️ [ASK_USER 하드스탑] chain 통지 실패 (무시): "
                                f"{_e_hs_chain}"
                            )
                    self._remove_task_from_schedule(task)
                    self._paused_task = None
                    self._wait_for_user = False
                    self._active_schema = None
                    task.pop("__pause_reason__", None)
                    task.pop("__ask_user_question__", None)
                    return
                # [2026-04-23] CAPTCHA 감지 → Telegram 즉시 알림.
                # 사용자가 PC 앞에 없어도 인지 가능하도록 기존 auth 알림과 같은 방식.
                # [2026-04-24] DOM 시그널(obs.captcha_required) 합류 — _annotate_captcha_state
                # 가 raw HTML 정규식으로 recaptcha/hcaptcha iframe·위젯을 감지하면 키워드
                # 매칭이 약해도 캡챠로 분류. false positive(일반 텍스트의 "captcha" 단어) ↓.
                _captcha_keywords = (
                    "캡챠", "캡차", "CAPTCHA", "captcha", "reCAPTCHA", "recaptcha",
                    "hCaptcha", "hcaptcha", "자동화 방지", "자동 입력 방지",
                    "보안 문자", "보안문자", "로봇이 아닙니다", "로봇이아닙니다",
                    "i'm not a robot", "i am not a robot",
                )
                _ask_is_captcha = any(
                    _k.lower() in _ask_blob for _k in _captcha_keywords
                ) or bool(obs.get("captcha_required"))
                if _ask_is_captcha:
                    try:
                        await self._notify_telegram_captcha(
                            question=str(question or "")[:200],
                            url=obs.get("url", "") or "",
                        )
                    except Exception as _te_cap:
                        print(
                            f"  ⚠️ [Telegram] 캡챠 알림 발송 중 예외 (무시): {_te_cap}"
                        )

                self._report(
                    f"[ASK_USER] ❓ **[확인 필요]** {question}\n"
                    f"└─ 완료 후 **⚡ 자율 실행** 을 눌러주세요."
                )
                self._paused_task = task
                task["status"] = "WAITING_USER"
                # [Fix 2026-04-20] 범용 ASK_USER 에도 pause_reason 기록.
                # [Fix 2026-04-22] 로그인/인증 ASK 는 "auth_required" 로 기록해야
                # resume 시점의 AuthRecheck(line 941) 가 실제 로그인 상태를 재검증하고
                # __auth_pass_grace__ 를 세팅함. 이전엔 일괄 "ask_user" 로 저장돼
                # AuthRecheck·grace 가 둘 다 무력화 → 로그인된 세션도 같은 ASK_USER
                # 를 끝없이 반복(2026-04-22 kmong 로그).
                # [Phase B 2026-04-24] CAPTCHA 도 "captcha_required" pause_reason 으로
                # 기록 → _tick() 의 CaptchaRecheck 가 resume 시점에 _observe() 로
                # captcha_required 가 사라졌는지 재검증 (3회 실패 → FAILED_CAPTCHA).
                task["__pause_reason__"] = (
                    "auth_required" if _ask_is_auth
                    else ("captcha_required" if _ask_is_captcha else "ask_user")
                )
                task["__ask_user_question__"] = _curr_q[:200]
                self._wait_for_user = True
                # 현재까지 히스토리를 task에 저장해 재개 시 복원
                task["__react_history__"] = history
                task["__react_step__"]    = step_count
                return

            # ── 3. ACT ────────────────────────────────────────────────────
            # NAVIGATE인데 target이 없으면 실행하지 않고 skip (chrome-error 방지)
            if action == "NAVIGATE" and not target:
                print("  [ReAct] NAVIGATE target 비어있음 — skip")
                fail_streak += 1
                await asyncio.sleep(1.0)
                continue

            # [Phase 4-3-2 2026-04-26] NAVIGATE 자기-자신 재시도 가드.
            # 현재 url ≈ target AND 같은 url 이 unhydrated_streak 활성이면 무의미한
            # 재시도 — 같은 placeholder 응답만 다시 받아서 LLM 헛스윙 1스텝 추가.
            # Why: 2026-04-26 로그 라인 410 — Step 3 [NAVIGATE] /my-gigs/new 가 이미
            # 그 페이지에 있고 unhydrated 인 상태에서 같은 URL 다시 NAVIGATE.
            # drift guard 는 다른 페이지 이탈은 잡지만 자기-자신은 못 잡음.
            if action == "NAVIGATE" and target:
                _cur_url_now = (obs.get("url") or "").strip()
                def _norm_url_self(u: str) -> str:
                    u2 = (u or "").strip().rstrip("/")
                    for _sep in ("?", "#"):
                        _i = u2.find(_sep)
                        if _i >= 0:
                            u2 = u2[:_i]
                    return u2
                _t_norm = _norm_url_self(target)
                _self_navigate = bool(_t_norm) and _norm_url_self(_cur_url_now) == _t_norm
                _uh_streak_dict = task.get("__unhydrated_streak__") or {}
                _uh_active_for_target = (
                    isinstance(_uh_streak_dict, dict)
                    and any(
                        int(v or 0) > 0
                        for k, v in _uh_streak_dict.items()
                        if _norm_url_self(k) == _t_norm
                    )
                )
                if _self_navigate and _uh_active_for_target:
                    _slg_msg = (
                        f"NAVIGATE skip: 이미 '{target[:50]}' 에 있음 "
                        f"(unhydrated 상태) — 같은 URL 재진입은 같은 placeholder "
                        f"만 받음. LLM 은 ASK_USER/GO_BACK/다른 URL 중 선택할 것"
                    )
                    print(f"  🚫 [NAVIGATE Self-Loop Guard] {_slg_msg}")
                    history.append({
                        "step": step_count,
                        "action": action,
                        "target": target[:60],
                        "result": _slg_msg[:120],
                        "verified": False,
                        "url": _cur_url_now,
                    })
                    fail_streak += 1
                    await asyncio.sleep(0.3)
                    continue

            # [Fix] Layer 1 사전 차단 — 이미 auth-blacklist가 있는 태스크가 또 다른
            # auth-gated 경로(/seller/*, /my-*, /dashboard 등)로 NAVIGATE 하려 하면
            # 네비게이션 라운드트립 없이 즉시 실패 처리 → LLM이 다른 경로 시도하도록 유도.
            if action == "NAVIGATE" and target and (task.get("__auth_blacklist__") or []):
                if _is_auth_gated_path(target):
                    _blocked_msg = (
                        f"⚠️ NAVIGATE 차단: 동일 세션에서 인증 필요 경로 확인됨. "
                        f"'{target[:60]}' 은(는) 로그인 없이 접근 불가 계열 — 사전 차단."
                    )
                    print(f"  🛑 [Layer 1 Pre-block] {_blocked_msg}")
                    history.append({
                        "step": step_count,
                        "action": action,
                        "target": target[:60],
                        "result": _blocked_msg[:100],
                        "verified": False,
                        "url": obs.get("url", ""),
                    })
                    fail_streak += 1
                    await asyncio.sleep(0.3)
                    continue

            # [2026-04-22] FILL 사전 차단 — 같은 (path, target) 이 이미 블랙리스트면
            # 또 실행해봐야 같은 JS 결과. 라운드트립 없이 즉시 실패 처리 + 강한 힌트.
            if action == "FILL" and target:
                try:
                    _cur_url_pb = obs.get("url", "") or ""
                    _mp_pb = re.match(r'^https?://[^/]+(/[^?#]*)', _cur_url_pb)
                    _cpath_pb = _mp_pb.group(1) if _mp_pb else ""
                    _fb_hit = False
                    for (_bp, _bt) in (task.get("__fill_blacklist__") or []):
                        if _bp == _cpath_pb and _bt == (target or "")[:60]:
                            _fb_hit = True
                            break
                    if _fb_hit:
                        _fb_msg = (
                            f"⚠️ FILL 사전 차단: '{_cpath_pb[:40]}' 페이지의 "
                            f"target='{target[:40]}' 은 이미 편집 불가로 확정됨. "
                            f"다른 [N] 선택 또는 NAVIGATE로 전용 편집 페이지로 이동 필요."
                        )
                        print(f"  🛑 [FILL Pre-block] {_fb_msg}")
                        self._report(f"🛑 **[FILL 재지명 차단]** {_fb_msg[:160]}")
                        if schema and hasattr(schema, "how"):
                            schema.how = _fb_msg + "\n\n" + (schema.how or "")
                        history.append({
                            "step": step_count,
                            "action": action,
                            "target": target[:60],
                            "result": _fb_msg[:100],
                            "verified": False,
                            "url": _cur_url_pb,
                        })
                        fail_streak += 1
                        await asyncio.sleep(0.3)
                        continue
                except Exception as _fpe:
                    print(f"  [FILL Pre-block] 검사 실패 (무시): {_fpe}")

            # ── [CLICK no-op 감지] act 직전 상태 스냅샷 ──────────────────
            # 같은 [N] 인덱스/셀렉터 CLICK이 DOM 변화를 만들지 못한 채 반복되는
            # 루프를 잡기 위해, act 이전 상태를 저장한다. 재관찰 이후 diff와 비교.
            _click_pre = None
            if action == "CLICK" and _task_type not in _NON_BROWSER:
                # [Phase 4-2 2026-04-25] CLICK SCHEMA-LEVEL PRE-VETO.
                # Why: _click_noop_counter / _click_preblock_counter 가 react 루프
                # 단위 local dict 라 force_advance·retry 후 새 루프 시 0 리셋. LLM 이
                # 같은 [4] CLICK 다시 시도하면 "이미 1회 no-op" 부터 새로 카운트 →
                # 무한 (register stage Step 1·7·9·12·13 [4] 반복). schema 에 영구
                # 차단 키 등록 → loop 재진입해도 즉시 하드 거부.
                try:
                    _sch_cb = self._active_schema
                    if _sch_cb is not None:
                        _click_blocks = getattr(_sch_cb, "_click_blocked_targets", None)
                        if isinstance(_click_blocks, dict) and _click_blocks:
                            _cur_url_cb = (obs.get("url") or "").strip()
                            _path_cb = ""
                            try:
                                _mp_cb = re.match(r'^https?://[^/]+(/[^?#]*)', _cur_url_cb)
                                _path_cb = _mp_cb.group(1) if _mp_cb else ""
                            except Exception:
                                _path_cb = ""
                            _cb_key = (_path_cb, (target or "")[:60])
                            if _cb_key in _click_blocks:
                                _cb_n = int(_click_blocks[_cb_key] or 0)
                                _veto_msg = (
                                    f"⛔ CLICK 사전 거부 (schema 영구 차단): '{target[:60]}' "
                                    f"on path '{_path_cb}' — 이미 {_cb_n}회 no-op 확정. "
                                    f"동일 stage 내 재CLICK 영구 금지. 다른 인덱스 / SCROLL "
                                    f"/ NAVIGATE / FILL / ASK_USER 중 선택 필수."
                                )
                                print(
                                    f"  ⛔ [CLICK PRE-VETO] schema-level 차단 hit — "
                                    f"key={_cb_key} count={_cb_n}"
                                )
                                self._report(
                                    f"⛔ **[CLICK 영구 차단 — PRE-VETO]** "
                                    f"`{target[:40]}` on `{_path_cb[:40]}` "
                                    f"(schema-level, {_cb_n}회 no-op)"
                                )
                                try:
                                    if hasattr(_sch_cb, "how"):
                                        _sch_cb.how = _veto_msg + "\n\n" + (_sch_cb.how or "")
                                except Exception:
                                    pass
                                history.append({
                                    "step": step_count,
                                    "action": "CLICK_BLOCKED",
                                    "target": (target or "")[:60],
                                    "result": _veto_msg[:100],
                                    "verified": False,
                                    "url": _cur_url_cb,
                                })
                                task["__stuck_detected__"] = True
                                fail_streak += 1
                                await asyncio.sleep(0.3)
                                continue
                except Exception as _e_cv:
                    print(f"  ⚠️ [CLICK PRE-VETO] 체크 실패 (무시): {_e_cv}")

                # [Fix] 동일 target에 이미 HINT 임계 이상 no-op 누적이면 실행 자체 거부.
                # (기존: 2회 연속 실패 → 힌트만 주입하고 3번째도 실행했음)
                _pre_key = (target or "")[:60] or "_blank"
                _prior_noop = _click_noop_counter.get(_pre_key, 0)
                if _prior_noop >= CLICK_NOOP_HINT_AT:
                    _blk_msg = (
                        f"⚠️ CLICK 사전 차단: '{_pre_key}'은 이미 {_prior_noop}회 "
                        f"no-op — 재실행 금지. 다른 전략 필요."
                    )
                    print(f"  🛑 [CLICK Pre-block] {_blk_msg}")
                    self._report(f"🛑 **[CLICK 재시도 차단]** {_blk_msg[:140]}")

                    # [Phase 4-2 2026-04-25] schema-level 영구 차단 등록.
                    # 다음 react 루프 재진입 시(force_advance·retry 후) PRE-VETO 가
                    # 즉시 잡도록 schema._click_blocked_targets[(path, target)] 에 누적.
                    try:
                        _sch_persist = self._active_schema
                        if _sch_persist is not None:
                            _cur_url_p = obs.get("url", "") or ""
                            _mp_p = re.match(r'^https?://[^/]+(/[^?#]*)', _cur_url_p)
                            _cpath_p = _mp_p.group(1) if _mp_p else ""
                            _persist_key = (_cpath_p, _pre_key[:60])
                            _existing = getattr(_sch_persist, "_click_blocked_targets", None)
                            if not isinstance(_existing, dict):
                                _existing = {}
                                _sch_persist._click_blocked_targets = _existing
                            _existing[_persist_key] = _existing.get(_persist_key, 0) + max(_prior_noop, 1)
                            print(
                                f"  📌 [CLICK Block persist] '{_pre_key}' on '{_cpath_p}' → "
                                f"schema 영구 차단 ({_existing[_persist_key]}회 누적)"
                            )
                    except Exception as _e_pp:
                        print(f"  ⚠️ [CLICK Block persist] 실패 (무시): {_e_pp}")

                    # [Fix 2026-04-20] 같은 target pre-block 누적 → 로그인 필요 추정 승격.
                    # CLICK no-op로는 `continue`만 해서 counter가 안 쌓인다. pre-block 자체를
                    # 카운트해서, LLM이 같은 [N]을 계속 지명하는 고집 패턴을 잡는다.
                    # 대상 element의 텍스트/href가 로그인 보호 행위(상품등록/글쓰기/판매자/로그인 등)를
                    # 가리키면, 비로그인이 원인일 가능성이 높으므로 Layer 1 경로로 위임한다.
                    _pb_n = _click_preblock_counter.get(_pre_key, 0) + 1
                    _click_preblock_counter[_pre_key] = _pb_n
                    if _pb_n >= CLICK_PREBLOCK_AUTH_ESCALATE and _task_type not in _NON_BROWSER:
                        try:
                            _blk_idx_tok2 = _pre_key.strip("[]")
                            _target_el = None
                            for _el2 in (obs.get("elements") or []):
                                if str(_el2.get("i", "")) == _blk_idx_tok2:
                                    _target_el = _el2
                                    break
                            _cur_url2 = obs.get("url", "") or ""
                            if _click_target_looks_auth_protected(_target_el, _cur_url2):
                                _el_desc = ((_target_el or {}).get("text") or
                                            (_target_el or {}).get("aria") or
                                            (_target_el or {}).get("href") or _pre_key)[:60]
                                _afails = int(task.get("__auth_fail_count__", 0)) + 1
                                task["__auth_fail_count__"] = _afails
                                _msg = (
                                    f"🔐 **[CLICK 반복 → 로그인 필요 추정 {_afails}/3]**\n"
                                    f"└─ target: `{_pre_key}` / 문구: '{_el_desc}'\n"
                                    f"└─ 같은 버튼을 {_pb_n}회 CLICK했지만 반응 없음 — "
                                    f"비로그인 상태에서 막힌 것으로 판단됩니다.\n"
                                    f"└─ 브라우저에서 로그인한 뒤 **⚡ 자율 실행**을 눌러주세요."
                                )
                                print(f"  🔐 [CLICK→Auth] 로그인 필요 추정 승격 — {_el_desc}")
                                if _afails >= 3:
                                    self._report(
                                        f"🔴 **[로그인 반복 실패 {_afails}회 — 자동 재시도 중단]** "
                                        f"CLICK '{_el_desc}' 반복 무반응\n"
                                        f"└─ 현재 URL: {_cur_url2[:80]}\n"
                                        f"└─ 브라우저에서 로그인 후 작업을 다시 입력해주세요."
                                    )
                                    await self._notify_telegram_auth(
                                        reason=f"CLICK '{_el_desc[:40]}' 반복 무반응 — 3회 초과",
                                        url=_cur_url2, fails=_afails,
                                    )
                                    task["status"] = "FAILED_AUTH"
                                    self._remove_task_from_schedule(task)
                                    self._paused_task = None
                                    self._wait_for_user = False
                                    task["__react_history__"] = history
                                    task["__react_step__"]    = step_count
                                    return
                                self._report(f"[ASK_USER] {_msg}")
                                await self._notify_telegram_auth(
                                    reason=f"같은 버튼 '{_el_desc[:40]}' {_pb_n}회 CLICK 무반응 (비로그인 추정)",
                                    url=_cur_url2, fails=_afails,
                                )
                                # auth_blacklist에 현재 path 등록 → 재시도 시 Tier 2에서 스킵
                                _abl = task.setdefault("__auth_blacklist__", [])
                                try:
                                    _pm2 = re.match(r'^https?://[^/]+(/[^?#]*)', _cur_url2)
                                    _cp2 = _pm2.group(1) if _pm2 else ""
                                    if _cp2 and _cp2 not in _abl:
                                        _abl.append(_cp2)
                                except Exception:
                                    pass
                                self._paused_task = task
                                task["status"] = "WAITING_USER"
                                task["__pause_reason__"] = "auth_required"
                                self._wait_for_user = True
                                task["__react_history__"] = history
                                task["__react_step__"]    = step_count
                                return
                        except Exception as _esc_e:
                            print(f"  ⚠️ [CLICK→Auth] 승격 판정 실패: {_esc_e}")
                            # 판정 실패 시 기본 pre-block 흐름 그대로 진행

                    # [Fix 2026-04-19 v4] pre-block 시 대안 힌트 주입.
                    # 기존: 차단만 하고 schema.how 업데이트 없어 LLM이 다음 iteration에
                    # 또 같은 [N] 선택 → 계속 차단 루프(로그: Step 10/18/28 CLICK [80] 반복).
                    # 현재 인덱싱된 elements에서 차단 인덱스 제외한 후보 N개 제시.
                    try:
                        _blk_idx_tok = _pre_key.strip("[]")  # "[80]" → "80"
                        _alt_lines: List[str] = []
                        for _el in (obs.get("elements") or []):
                            _ei = str(_el.get("i", ""))
                            if not _ei or _ei == _blk_idx_tok:
                                continue
                            _et = (_el.get("text") or "").strip()[:35]
                            _etag = (_el.get("tag") or "")[:10]
                            _ehref = (_el.get("href") or "")[:60]
                            _desc = _et or _ehref or _etag or "?"
                            _alt_lines.append(f"- [{_ei}] ({_etag}) {_desc}")
                            if len(_alt_lines) >= 8:
                                break
                        _alt_block = ("\n".join(_alt_lines)
                                      if _alt_lines else "- (현재 페이지 인덱스 없음 — SCROLL 또는 NAVIGATE 권장)")
                        _blk_hint = (
                            f"🛑 [CLICK 차단] '{_pre_key}' 재클릭 금지 — {_prior_noop}회 no-op 확정.\n"
                            f"대안 액션을 **반드시** 선택하라 (우선순위 순):\n"
                            f"(1) 아래 다른 [N] 인덱스로 CLICK:\n{_alt_block}\n"
                            f"(2) SCROLL bottom/top 으로 lazy-render 요소 노출 후 재인덱싱.\n"
                            f"(3) 검색창이면 FILL로 키워드 입력 후 Enter.\n"
                            f"(4) 다른 URL로 NAVIGATE (예: 상세 페이지 드릴다운).\n"
                            f"(5) 단서 부족 시 ASK_USER.\n"
                            f"동일 '{_pre_key}' 재선택은 자동 차단된다."
                        )
                        if schema is not None and hasattr(schema, "how"):
                            schema.how = _blk_hint + "\n\n" + (schema.how or "")
                        print(f"  💡 [CLICK Pre-block] 대안 힌트 주입 ({len(_alt_lines)}개 인덱스)")
                    except Exception as _bh_e:
                        pass  # 힌트 주입 실패해도 차단은 유지
                    history.append({
                        "step": step_count,
                        "action": "CLICK_BLOCKED",
                        "target": _pre_key,
                        "result": _blk_msg[:100],
                        "verified": False,
                        "url": obs.get("url", ""),
                    })
                    task["__stuck_detected__"] = True
                    fail_streak += 1
                    await asyncio.sleep(0.3)
                    continue
                _click_pre = {
                    "url": (obs.get("url") or "").strip(),
                    "html_len": int(obs.get("html_len", 0) or 0),
                    "count": len(obs.get("elements") or []),
                }

            # ── [Detail-page NAVIGATE Gate 2026-04-24] ───────────────────
            # 현재 obs.url 이 개별 상세(/gig/, /product/ 등)인데 아직
            # 이 URL 로 READ_PAGE 하지 않은 상태에서 LLM 이 NAVIGATE/GO_BACK
            # 로 이탈하려 하면 → 즉시 READ_PAGE 로 교체해 실행.
            # Why: Detail-arrival 은 schema.how 힌트만 주입했는데 LLM 이
            # 14회 연속 무시하고 search↔/gig/662367 핑퐁 (2026-04-24 로그).
            # 힌트가 아닌 actor 측 intercept 로 바꾼다.
            if (action in ("NAVIGATE", "GO_BACK")
                    and schema is not None
                    and _task_type not in _NON_BROWSER):
                try:
                    _cur_url_dp = (obs.get("url") or "").strip()
                    _cur_url_lc_dp = _cur_url_dp.lower()
                    # Phase D [2026-04-27] module-level helper 로 통합 — /l/ /dp/ 등 자동 적용
                    _on_detail = _classify_sampling_url(_cur_url_lc_dp) == "detail"
                    if _on_detail and _cur_url_dp:
                        _cur_key_dp = _cur_url_lc_dp.split("#", 1)[0].split("?", 1)[0].rstrip("/")
                        _already_read_dp = False
                        for _h_dp in history:
                            if _h_dp.get("action") != "READ_PAGE" or not _h_dp.get("verified"):
                                continue
                            _h_url_dp = (_h_dp.get("url") or "").lower()
                            _h_key_dp = _h_url_dp.split("#", 1)[0].split("?", 1)[0].rstrip("/")
                            if _h_key_dp == _cur_key_dp:
                                _already_read_dp = True
                                break
                        # [Fix 2026-04-25] 이 URL 의 상세 id 에 해당하는 detail_<id> 변수가
                        # 이미 있을 때만 통과. 이전엔 "어떤 변수든 300자 이상" 이면 통과시켜
                        # 첫 gig READ_PAGE 이후 모든 후속 gig 이탈을 막지 못함 (최신로그.txt
                        # step 3 에서 detail_536570 저장된 뒤 step 7/9/12/15 연속 bypass → 20개
                        # 샘플링 실패).
                        _has_this_detail_var = False
                        try:
                            _gid_m_check = re.search(
                                # Phase D [2026-04-27] /l/ /dp/ /course/ /lesson/ /book/ 추가
                                r'/(?:gig|product|item|post|article|service|listing|view|detail|l|dp|course|lesson|book)s?/([^/?#]+)',
                                _cur_url_lc_dp
                            )
                            if _gid_m_check and schema and getattr(schema, "_react_vars", None):
                                _this_id = _gid_m_check.group(1)[:20]
                                _v_this = (schema._react_vars or {}).get(f"detail_{_this_id}")
                                if isinstance(_v_this, str) and len(_v_this) > 300:
                                    _has_this_detail_var = True
                        except Exception:
                            pass
                        if not _already_read_dp and not _has_this_detail_var:
                            _gig_id_m = re.search(
                                # Phase D [2026-04-27] /l/ /dp/ /course/ /lesson/ /book/ 추가
                                r'/(?:gig|product|item|post|article|service|listing|view|detail|l|dp|course|lesson|book)s?/([^/?#]+)',
                                _cur_url_lc_dp
                            )
                            _gig_tag = (_gig_id_m.group(1) if _gig_id_m else "detail")[:20]
                            _forced_var_dp = f"detail_{_gig_tag}"
                            print(
                                f"  🛑 [Detail NAVIGATE Gate] 상세 페이지 {_cur_url_dp[:60]} "
                                f"에서 READ_PAGE 없이 {action} 이탈 시도 — "
                                f"READ_PAGE '{_forced_var_dp}' 로 자동 교체"
                            )
                            self._report(
                                f"🛑 **[Detail NAVIGATE 차단]** 상세 페이지를 읽지 않고 이탈 금지\n"
                                f"└─ 자동 교체: READ_PAGE → {_forced_var_dp}"
                            )
                            action = "READ_PAGE"
                            target = _forced_var_dp
                            content = ""
                            # 다음 Think 에 강한 규칙 주입 — 재발 방지
                            _dp_rule = (
                                f"[강제 규칙] 개별 상세 페이지(/gig/, /product/, /l/, /dp/, "
                                f"/course/ 등)에 도착하면 READ_PAGE 로 본문을 변수에 먼저 "
                                f"저장해야 한다. READ_PAGE 없이 NAVIGATE/GO_BACK 으로 이탈하면 "
                                f"자동 차단되어 READ_PAGE 로 교체된다. 변수명은 'detail_{{id}}' 형태로."
                            )
                            if hasattr(schema, "how"):
                                schema.how = _dp_rule + "\n\n" + (schema.how or "")
                except Exception as _dpg_e:
                    pass  # Gate 실패 시 기본 흐름 유지

            # ── [Retry Target Lock 2026-04-27] retry_attrs 의 LLM 새 파일명 차단 ──
            # retry_attrs 진입 시 RepairPlanner 가 schema._retry_target_lock 에 직전
            # 성공 WRITE target 을 박아둠. LLM 이 임의로 새 파일명을 만들면 같은
            # 산출물을 보강하는 의도가 깨지므로 actor-side 에서 강제 교체.
            # 회귀 가드: lock 미설정(일반 흐름) 또는 target 일치 시 그대로 통과.
            if (action == "WRITE"
                    and schema is not None
                    and getattr(schema, "_retry_target_lock", "")):
                try:
                    _lock_target = (schema._retry_target_lock or "").strip()
                    _curr_target = (target or "").strip()
                    if _lock_target and _curr_target and _curr_target != _lock_target:
                        print(
                            f"  🔒 [Retry Lock] WRITE target '{_curr_target[:40]}' "
                            f"→ '{_lock_target[:40]}' 강제 교체"
                        )
                        self._report(
                            f"🔒 **[Retry Target Lock]** retry 중 새 파일명 차단 — "
                            f"`{_curr_target[:30]}` → `{_lock_target[:30]}` (동일 산출물 보강)"
                        )
                        target = _lock_target
                except Exception as _e_rtl_w:
                    pass  # 실패 시 LLM 의 target 그대로

            # ── [Phase F 2026-04-27] PRE-VETO strike >= 2 자동 피벗 ───────
            # Why: 최신로그 strike 1→5 동안 LLM 이 schema.how pivot hint 무시하고 계속
            #      WRITE 시도 → chain abort. action 자체를 NAVIGATE 로 강제 교체해
            #      WRITE 사이클 끊음 (drill_down_pool 미방문 첫 항목 사용).
            #      pool 비어있으면 기존 PRE-VETO 거부 메시지 그대로 (변경 없음).
            if (action == "WRITE"
                    and schema is not None
                    and _task_type not in _NON_BROWSER):
                try:
                    _pv_strikes_f = getattr(schema, "_pre_veto_strikes", None)
                    if (isinstance(_pv_strikes_f, dict)
                            and int(_pv_strikes_f.get(target, 0) or 0) >= 2):
                        _pool_f = getattr(schema, "_drill_down_pool", []) or []
                        _unvisited_f = [
                            p for p in _pool_f
                            if isinstance(p, dict) and not p.get("visited")
                        ]
                        if _unvisited_f:
                            _next_p_f = _unvisited_f[0]
                            _next_url_f = _next_p_f.get("url", "")
                            _strike_n_f = int(_pv_strikes_f.get(target, 0) or 0)
                            if _next_url_f:
                                print(
                                    f"  🔁 [Phase F] PRE-VETO strike {_strike_n_f} "
                                    f"— WRITE '{target[:40]}' → NAVIGATE 자동 교체: "
                                    f"{_next_url_f[:60]}"
                                )
                                self._report(
                                    f"🔁 **[PRE-VETO 자동 피벗]** strike {_strike_n_f} — "
                                    f"`{target[:30]}` WRITE 차단 → NAVIGATE "
                                    f"{_next_url_f[:50]}"
                                )
                                action = "NAVIGATE"
                                target = _next_url_f
                                content = ""
                except Exception as _e_pf:
                    pass  # 실패 시 기존 PRE-VETO 흐름 유지

            # ── [Phase G 2026-04-27] list-as-info 도메인 폴백 ──────────────
            # Why: Phase F 는 _drill_down_pool 미방문 항목이 있을 때만 NAVIGATE
            # 강제. 그러나 네이버지도/카카오맵 같은 list-as-info 도메인은
            # detail page 진입이 본질적으로 무의미 → pool 자체가 비어있어
            # Phase F 미발동 → strike 누적 → chain abort. 본 분기는 pool 이
            # 비어있고 list-as-info 도메인 방문 흔적이 있으면 WRITE → READ_PAGE
            # 로 강제 교체 + schema.how 에 본문 보강 가이드 박음.
            # 회귀 가드: 일반 e-commerce(kmong/gumroad) 는 list-as-info 화이트
            # 리스트 미포함 → 분기 진입 X → Phase F 의 abort 흐름 그대로.
            if (action == "WRITE"
                    and schema is not None
                    and _task_type not in _NON_BROWSER):
                try:
                    _pv_strikes_g = getattr(schema, "_pre_veto_strikes", None)
                    if (isinstance(_pv_strikes_g, dict)
                            and int(_pv_strikes_g.get(target, 0) or 0) >= 2):
                        _pool_g = getattr(schema, "_drill_down_pool", []) or []
                        _has_unvisited_g = any(
                            isinstance(p, dict) and not p.get("visited") for p in _pool_g
                        )
                        if not _has_unvisited_g:
                            _vu_g = getattr(schema, "_react_var_urls", {}) or {}
                            _li_visited_g = next(
                                (u for u in _vu_g.keys() if _is_list_as_info_domain(u)),
                                None,
                            )
                            if _li_visited_g:
                                _strike_n_g = int(_pv_strikes_g.get(target, 0) or 0)
                                print(
                                    f"  🔁 [Phase G] PRE-VETO strike {_strike_n_g} "
                                    f"+ list-as-info 도메인({_li_visited_g[:50]}) + "
                                    f"pool 비어있음 — WRITE → READ_PAGE 자동 교체"
                                )
                                self._report(
                                    f"🔁 **[List-as-Info 보강]** strike {_strike_n_g} — "
                                    f"`{target[:30]}` WRITE 차단 → 페이지 본문 추가 수집. "
                                    f"(평점·리뷰 등 실질 정보 부족)"
                                )
                                if hasattr(schema, "how"):
                                    _how_g = (
                                        "[List-as-Info 보강] 현재 페이지 본문이 평점·리뷰·"
                                        "거리 등 실질 정보를 충분히 담지 못함. "
                                        "다음 액션 후보:\n"
                                        "  (a) 정렬/필터/스크롤로 페이지 갱신 후 READ_PAGE 재시도\n"
                                        "  (b) 검색어를 더 구체적으로 (예: '강남역 파스타 평점순') FILL\n"
                                        "  (c) 다른 list-as-info 도메인(카카오맵 등)으로 NAVIGATE\n"
                                        "같은 파일에 WRITE 반복 금지."
                                    )
                                    schema.how = _how_g + "\n\n" + (schema.how or "")
                                action = "READ_PAGE"
                                target = "page_text"
                                content = ""
                except Exception as _e_pg:
                    pass  # 실패 시 기존 흐름 유지

            # ── [WRITE 선행-READ 강제] browser_read 전용 가드 ────────────
            # task_type=='browser_read' (또는 VERB=READ)에서 LLM이 READ_PAGE 없이
            # WRITE로 직행하면 페이지 내용을 상상으로 지어낸다. 첫 WRITE는
            # READ_PAGE로 자동 교체하여 실제 페이지 텍스트를 변수에 먼저 확보.
            _verb_is_read = bool(schema) and getattr(schema, "verb", "") == "READ"
            if (action == "WRITE"
                    and (_task_type == "browser_read" or _verb_is_read)
                    and _task_type not in _NON_BROWSER):
                _has_read = any(h.get("action") == "READ_PAGE" for h in history)
                _has_read_var = bool(
                    schema and getattr(schema, "_react_vars", None)
                    and any(k.startswith("page_") or k.endswith("_page")
                            or len(v) > 500
                            for k, v in schema._react_vars.items())
                )
                # [P1 2026-04-21] chain upstream artifact 가 있으면 READ_PAGE 강제 스킵.
                # 증상: plan/dev 등 후속 stage 는 이전 stage 산출물(research_report.md 등)
                # 을 근거로 써야 하는데, VERB=READ 상속으로 이 Gate 가 발동해 현재
                # 브라우저 페이지 innerText 1331자(허접 네비 텍스트)를 page_text 로
                # 주입 → Shortening Guard 가 빈약한 draft 를 매번 차단 → 3연속 실패.
                # schema.upstream_outputs 는 chain 진입 시 확실히 채워지므로 이걸
                # 1차 단서로 삼아 Gate 를 통과시킨다 (100자 필터로 placeholder 제외).
                _has_chain_upstream = False
                if schema is not None:
                    _up_out = getattr(schema, "upstream_outputs", None) or {}
                    for _uv in _up_out.values():
                        if _uv and len(str(_uv)) > 100:
                            _has_chain_upstream = True
                            break
                if not (_has_read or _has_read_var or _has_chain_upstream):
                    _orig_target = target or "summary"
                    _forced_var = "page_text"
                    print(f"  🛑 [WRITE Gate] READ_PAGE 선행 필요 — "
                          f"WRITE '{_orig_target[:30]}' → READ_PAGE '{_forced_var}' 로 자동 교체")
                    self._report(
                        f"🛑 **[WRITE Gate]** READ_PAGE 없이 WRITE 시도 차단\n"
                        f"└─ 자동 교체: READ_PAGE → {_forced_var}"
                    )
                    action = "READ_PAGE"
                    target = _forced_var
                    content = ""
                elif _has_chain_upstream and not _has_read and not _has_read_var:
                    # 가시화: upstream 덕분에 Gate 를 통과했음을 로그로 남긴다.
                    try:
                        _up_keys = list((getattr(schema, "upstream_outputs", {}) or {}).keys())[:3]
                        print(f"  ✅ [WRITE Gate] chain upstream 존재로 READ_PAGE 스킵 "
                              f"(keys={_up_keys})")
                    except Exception:
                        pass

            # ── [READ_PAGE 중복 Gate] history 기반 강한 차단 ─────────────
            # _act 내부에도 URL-키 기반 중복 차단이 있지만 URL 인코딩/조회
            # 타이밍 경쟁으로 불발되는 사례가 로그에서 확인됨. LLM이 변수명만
            # 바꿔서 같은 페이지를 두 번 읽으면 토큰만 태우고 맥락도 전혀
            # 새로워지지 않는다. history에 READ_PAGE가 있고 그 사이에 URL
            # 변화가 없으면 현재 READ_PAGE 요청을 WRITE로 전환해 요약 단계로
            # 바로 진행시킨다.
            if action == "READ_PAGE":
                _cur_url_for_dup = (obs.get("url") or "").strip()
                _prev_read = None
                for _h in reversed(history):
                    if _h.get("action") == "READ_PAGE":
                        _prev_read = _h
                        break
                if _prev_read:
                    _prev_url = (_prev_read.get("url") or "").strip()
                    _has_nav_between = False
                    _idx = history.index(_prev_read)
                    for _h in history[_idx + 1 :]:
                        if _h.get("action") in ("NAVIGATE", "GO_BACK", "CLICK") and _h.get("verified"):
                            _has_nav_between = True
                            break
                    if (_cur_url_for_dup and _prev_url and _cur_url_for_dup == _prev_url
                            and not _has_nav_between):
                        _existing_vars = list((schema._react_vars or {}).keys()) if schema else []
                        _existing_var = _existing_vars[-1] if _existing_vars else ""
                        print(f"  🛑 [READ_PAGE 중복 Gate] 동일 URL '{_cur_url_for_dup[:60]}' "
                              f"이전 READ_PAGE 이후 네비 변화 없음 → WRITE로 전환 "
                              f"(기존 변수: '{_existing_var}')")
                        self._report(
                            f"🛑 **[READ_PAGE 중복 차단]** 같은 페이지 재READ 금지\n"
                            f"└─ 자동 전환: WRITE — 기존 변수 '{_existing_var[:30]}' 활용"
                        )
                        action = "WRITE"
                        _new_content = (
                            f"아래 페이지 내용을 사용자 요청에 맞게 요약·정리하라.\n\n"
                            f"{{{_existing_var}}}"
                        ) if _existing_var else "위 페이지 내용을 요약하라."
                        content = _new_content
                        target = target.strip() or "summary"

            # [P-α S4 2026-05-01] step latency 측정 — _STEP_LATENCY_QUEUE 에 push.
            _step_t0 = time.perf_counter()
            act_result = await self._act(action, target, content, schema, reason=reason)
            try:
                _STEP_LATENCY_QUEUE.append(
                    (time.time(), action, time.perf_counter() - _step_t0)
                )
            except Exception:
                pass

            # ── [FILL no-op 감지] JS 반환 "FILL 실패"/"NO_EDITOR" 즉시 판정 ─────
            # 같은 target index 에 FILL을 반복해도 편집 가능 영역이 없으면 또 실패할 뿐.
            # 1회 실패만으로도 힌트 주입, 2회 연속이면 강한 차단 + target 로 재FILL 금지 규칙 삽입.
            if action == "FILL":
                _fill_key = (target or "")[:60] or "_blank"
                _fill_failed = (
                    "FILL 실패" in (act_result or "") or
                    "FILL_FAILED" in (act_result or "") or
                    "NO_EDITOR" in (act_result or "")
                )
                if _fill_failed:
                    _fill_noop_counter[_fill_key] = _fill_noop_counter.get(_fill_key, 0) + 1
                    _fn = _fill_noop_counter[_fill_key]
                    print(f"  [FILL no-op] target={_fill_key} n={_fn} result={(act_result or '')[:60]}")
                    if _fn >= FILL_NOOP_ESCALATE:
                        # [P2 Fix] FILL 2회 연속 실패 시 — 검색창을 못 찾은 상황이 대부분.
                        # 현재 URL에 대한 검색 변형(_generate_url_variants)으로 자동 에스컬레이션.
                        # 이미 시도한 변형은 __nav_variants_tried__ 집합으로 중복 차단.
                        _auto_variant_fired = False
                        try:
                            _cur_url_fill = self._get_current_browser_url() or ""
                            _already_tried: set = task.setdefault("__nav_variants_tried__", set())
                            _variants = _generate_url_variants(_cur_url_fill, task_prompt, _already_tried)
                            if _variants:
                                _next_var = _variants[0]
                                _already_tried.add(_next_var.rstrip("/").lower())
                                print(f"  🔁 [FILL→URL Variant] 대문 검색 FILL 실패 {_fn}회 "
                                      f"→ 검색 엔드포인트로 자동 이동: {_next_var}")
                                self._report(
                                    f"🔁 **[FILL 제자리걸음 → URL 변형 에스컬레이션]** "
                                    f"검색창 FILL 실패 → {_next_var[:80]}"
                                )
                                try:
                                    await _navigate_and_wait(url=_next_var, timeout=20.0)
                                    _auto_variant_fired = True
                                    # LLM이 다음 Think에서 변경된 URL을 기준으로 재탐색하도록 힌트만 짧게 주입
                                    _var_hint = (
                                        f"[시스템] 검색창 FILL 실패 → '{_next_var[:60]}' 로 이동했다. "
                                        f"이 URL은 검색 결과 페이지일 가능성이 높다. "
                                        f"이제 결과 목록에서 관련 링크를 CLICK하거나 READ_PAGE로 내용을 확인하라."
                                    )
                                    if schema and hasattr(schema, "how"):
                                        schema.how = _var_hint + "\n\n" + (schema.how or "")
                                except Exception as _nv_e:
                                    print(f"  [FILL→URL Variant] navigate 실패: {_nv_e}")
                        except Exception as _uv_e:
                            print(f"  [FILL→URL Variant] 변형 생성 실패 (무시): {_uv_e}")

                        # [2026-04-22] FILL target 영구 블랙리스트.
                        # counter 만 리셋하면 LLM 이 다음 step 에서 같은 [N] 재지명 →
                        # _fn=1 HINT_AT 경로로만 걸림 → ESCALATE 까지 또 2회 낭비.
                        # (path, target) 을 task 에 영구 저장하고 _think 프롬프트에
                        # 노출 + ACT 사전 차단으로 이중 방어.
                        try:
                            _cur_url_for_bl = (self._get_current_browser_url() or "")
                            _pm_bl = re.match(r'^https?://[^/]+(/[^?#]*)', _cur_url_for_bl)
                            _path_bl = _pm_bl.group(1) if _pm_bl else ""
                            _fill_bl: list = task.setdefault("__fill_blacklist__", [])
                            _entry_bl = [_path_bl, (target or "")[:60]]
                            if _entry_bl not in _fill_bl:
                                _fill_bl.append(_entry_bl)
                                print(
                                    f"  🚫 [FILL Blacklist] +1: path='{_path_bl[:40]}' "
                                    f"target='{(target or '')[:30]}' (total={len(_fill_bl)})"
                                )
                        except Exception as _bl_e:
                            print(f"  [FILL Blacklist] 등록 실패 (무시): {_bl_e}")

                        if not _auto_variant_fired:
                            _fill_hint = (
                                f"⚠️ FILL target='{target[:40]}' 이(가) {_fn}회 연속 실패했다 "
                                f"— 이 요소는 편집 가능 영역이 아니다. 같은 [N] 재FILL 금지. "
                                f"다음 전략을 순서대로 고려하라: "
                                f"(a) 대상 사이트가 현재 URL과 다르면 NAVIGATE로 먼저 이동 "
                                f"(b) 인덱싱된 다른 [M] 중 input/textarea 같은 실제 입력 요소 선택 "
                                f"(c) 에디터가 있는 전용 페이지로 NAVIGATE "
                                f"(d) 판단 불가 시 ASK_USER."
                            )
                            self._report(f"🔁 **[FILL 제자리걸음 차단]** {_fill_hint[:160]}")
                            if schema and hasattr(schema, "how"):
                                schema.how = _fill_hint + "\n\n" + (schema.how or "")
                        _fill_noop_counter[_fill_key] = 0
                    elif _fn >= FILL_NOOP_HINT_AT:
                        _fill_hint = (
                            f"⚠️ FILL target='{target[:40]}' 실패 — 편집 가능 영역 아님. "
                            f"같은 target 재FILL 대신 NAVIGATE로 대상 페이지로 이동하거나 "
                            f"input/textarea 타입의 다른 [N] 선택."
                        )
                        if schema and hasattr(schema, "how"):
                            schema.how = _fill_hint + "\n\n" + (schema.how or "")
                else:
                    _fill_noop_counter[_fill_key] = 0

            # ── [6-2+6-3 수정] 로그인 실패/정보 부족 즉시 감지 ───────────
            # fail_streak 5회까지 기다리지 않고 특정 오류를 즉시 ASK_USER로 전환.
            # "모르면 멈추고 물어보는" 동작 구현.
            _LOGIN_SIGNALS = ["로그인 필요", "login required", "로그인이 필요", "sign in"]
            _INFO_MISSING_SIGNALS = ["어느 상품", "상품 ID", "URL이 필요", "정보 부족"]

            _act_lower = (act_result or "").lower()
            _specific_question = None

            if any(s in (act_result or "") for s in _LOGIN_SIGNALS):
                _specific_question = (
                    f"🔐 **[로그인 필요]** 크몽 로그인이 필요합니다.\n"
                    f"브라우저에서 직접 로그인한 후 **⚡ 자율 실행** 을 눌러주세요.\n"
                    f"현재 시도한 URL: {target[:80]}"
                )
            elif schema and not schema.what.strip():
                _specific_question = (
                    f"❓ **[정보 필요]** 어느 상품의 상세페이지를 수정할까요?\n"
                    f"상품 이름 또는 편집 URL을 알려주세요."
                )

            if _specific_question:
                self._report(
                    f"[ASK_USER] {_specific_question}\n"
                    f"└─ 완료 후 **⚡ 자율 실행** 을 눌러주세요."
                )
                self._paused_task = task
                task["status"] = "WAITING_USER"
                # [P2] 로그인 사유 ASK는 pause_reason 기록 → run_now() 재개 전 재검증
                if any(s in (act_result or "") for s in _LOGIN_SIGNALS):
                    task["__pause_reason__"] = "auth_required"
                    task["__auth_fail_count__"] = int(task.get("__auth_fail_count__", 0)) + 1
                    # [Fix 2026-04-20] LLM 이 '로그인 필요' 시그널을 출력한 경우에도
                    # Telegram 알림을 함께 쏜다. (기존엔 GUI 채팅창만 표시되어
                    # 사용자가 PC 앞에 없으면 register 단계에서 무한 대기)
                    await self._notify_telegram_auth(
                        reason="ReAct 루프가 '로그인 필요' 시그널 감지",
                        url=(obs.get("url", "") if isinstance(obs, dict) else "") or (target or ""),
                        fails=int(task.get("__auth_fail_count__", 1)),
                    )
                self._wait_for_user = True
                task["__react_history__"] = history
                task["__react_step__"]    = step_count
                return
            # ─────────────────────────────────────────────────────────────


            verified, verify_msg = await self._verify_step(
                action=action,
                target=target,
                act_result=act_result,
                schema=schema,
            )

            # [Fix] 액션 후 DOM 재관찰 — 다음 _think()가 최신 화면을 볼 수 있도록
            if action in {"NAVIGATE", "FILL", "CLICK", "CHECK_LOGIN"} and _task_type not in _NON_BROWSER:
                try:
                    obs = await self._observe()
                    # ── [Phase C 2026-04-27] SPA navigation recheck ──
                    # CLICK 직후 SPA 가 history.pushState 로 URL 만 바꾸고 DOM 은 비동기
                    # 렌더 → 재관측이 너무 빨라 stale obs (URL/HTML 변화 미반영, false noop).
                    # 최신로그.txt Step 6/7 — Gumroad CLICK 직후 noop 판정됐는데 그 후
                    # promptgeek.gumroad.com 으로 navigation 된 케이스. 보수 조건 매칭 시
                    # 짧은 대기(0.6s) + 1회 재관측 → URL 변화 검증.
                    if (action == "CLICK"
                            and _click_pre is not None
                            and os.environ.get("EIDOS_CLICK_SPA_RECHECK", "1") == "1"
                            and _click_pre.get("url") == (obs.get("url") or "").strip()
                            and abs(int(_click_pre.get("html_len", 0) or 0)
                                    - int(obs.get("html_len", 0) or 0)) < 5000):
                        try:
                            await asyncio.sleep(0.6)
                            _obs2 = await self._observe()
                            _new_url = (_obs2.get("url") or "").strip()
                            if _new_url and _new_url != _click_pre.get("url"):
                                obs = _obs2
                                print(
                                    f"  🔁 [SPA Recheck] URL 변화 감지 — 재관측 채택: "
                                    f"{_click_pre.get('url', '')[:50]} → {_new_url[:50]}"
                                )
                        except Exception:
                            pass
                except Exception:
                    pass  # 재관찰 실패해도 루프는 계속

            # ── [CLICK no-op 감지] 재관찰 결과로 DOM 변화 diff 판정 ─────────
            # url/html_len/요소수 셋 다 거의 그대로면 "같은 요소 재클릭 루프" 로 판단.
            # 2회 연속: 힌트 주입, 3회 연속: 강제 스크롤 + 강한 차단.
            if _click_pre is not None and action == "CLICK":
                _post = {
                    "url": (obs.get("url") or "").strip(),
                    "html_len": int(obs.get("html_len", 0) or 0),
                    "count": len(obs.get("elements") or []),
                }
                _changed = (
                    _click_pre["url"] != _post["url"] or
                    abs(_click_pre["html_len"] - _post["html_len"]) > 2000 or
                    abs(_click_pre["count"] - _post["count"]) >= 3
                )
                _key = (target or "")[:60] or "_blank"
                if _changed:
                    _click_noop_counter[_key] = 0
                else:
                    _click_noop_counter[_key] = _click_noop_counter.get(_key, 0) + 1
                    _noop_n = _click_noop_counter[_key]
                    print(f"  [CLICK no-op] target={_key} noop={_noop_n} "
                          f"pre={_click_pre} post={_post}")
                    # [오염 수정 2026-04-18] no-op은 실제 실패 — verify를 뒤집어
                    # "Step 완료"로 잘못 보고하지 않게 한다. 이전에는 _verify_step이
                    # "실행됨" 문자열만 보고 verified=True를 반환해 LLM이 클릭
                    # 성공으로 오인하고 다음 step에서 엉뚱한 행동을 했다.
                    verified = False
                    verify_msg = f"CLICK no-op: DOM 무변화 (target={_key[:20]}, noop={_noop_n})"
                    if _noop_n >= CLICK_NOOP_ESCALATE:
                        _hint = (
                            f"⚠️ [{target[:40]}] CLICK이 {_noop_n}회 연속 DOM 변화를 만들지 못함. "
                            f"이 요소 재클릭 금지. 다음 전략: "
                            f"(a) SCROLL로 다른 요소 노출, "
                            f"(b) 다른 [N] 인덱스 CLICK, "
                            f"(c) 판단 불가 시 ASK_USER."
                        )
                        self._report(f"🔁 **[CLICK 제자리걸음 차단]** {_hint[:140]}")
                        if schema and hasattr(schema, "how"):
                            schema.how = _hint + "\n\n" + (schema.how or "")
                        _force_scroll_next = True
                        # [Fix] 힌트만으로는 LLM이 재클릭 반복 — 다음 iteration에서
                        # Tier 3(GO_BACK) 강제 에스컬레이션이 발동하도록 stuck 플래그 set.
                        task["__stuck_detected__"] = True
                        _click_noop_counter[_key] = 0
                    elif _noop_n >= CLICK_NOOP_HINT_AT:
                        _hint = (
                            f"⚠️ [{target[:40]}] CLICK {_noop_n}회 연속 DOM 무변화. "
                            f"같은 요소 재클릭 금지, 다른 전략 시도."
                        )
                        if schema and hasattr(schema, "how"):
                            schema.how = _hint + "\n\n" + (schema.how or "")

            # 히스토리 기록 (ContextStore) — 갱신된 obs의 URL 사용
            # T1 (CoT): LLM 의 reason 을 thought 로 보존 — 단계별 의도 trace
            history.append({
                "step": step_count,
                "action": action,
                "target": target[:60] if target else "",
                "result": act_result[:100] if act_result else "",
                "verified": verified,
                "url": obs.get("url", ""),
                "thought": (reason or "")[:200],
            })
            if schema:
                schema.add_step(action, target, act_result, obs.get("url", ""), thought=(reason or "")[:200])

            # [Phase 1-B 2026-04-24] drill-down 풀 방문 체크. NAVIGATE 성공·verified
            # 여부와 무관하게 현재 URL 이 풀 entry 와 매칭되면 visited=True.
            # 재방문 게이트는 아래 드릴다운 재주입 로직에서 _unvisited 필터링으로 해결.
            try:
                if action == "NAVIGATE" and schema is not None:
                    _pool_vt = getattr(schema, "_drill_down_pool", None)
                    if _pool_vt:
                        _cur_u_vt = (obs.get("url") or "").lower()
                        _cur_k_vt = _cur_u_vt.split("#", 1)[0].split("?", 1)[0].rstrip("/")
                        if _cur_k_vt:
                            for _p_vt in _pool_vt:
                                if (isinstance(_p_vt, dict)
                                        and not _p_vt.get("visited")
                                        and _p_vt.get("key") == _cur_k_vt):
                                    _p_vt["visited"] = True
                                    print(f"  ✅ [Drill-down] 방문 기록: {_cur_k_vt[-60:]}")
                                    break
            except Exception:
                pass

            if verified:
                fail_streak = 0
                self._report(
                    f"✅ **[Step {step_count} 완료]** [{action}] {(target or content)[:50]}\n"
                    f"└─ {verify_msg[:80]}"
                )

                # [Fix 2026-04-19 v3] NAVIGATE가 listing/카테고리 허브에 도달했을 때,
                # 인덱싱된 요소에서 '개별 상세' 링크 후보를 뽑아 schema.how 앞에 주입.
                # LLM이 카테고리만 순회하고 gig/product 상세로 드릴다운 안 하는 버그 대응.
                if action == "NAVIGATE" and schema is not None:
                    try:
                        _cur_url_lc = (obs.get("url") or "").lower()
                        _LISTING_DETECT = (
                            "/category", "/categories", "/search", "/tag", "/tags",
                            "/list", "/listings", "/browse", "/explore", "/discover",
                        )
                        # [근본 fix 2026-05-04] 루트 도메인 (홈페이지) 도 listing 으로 처리.
                        # 사용자 로그 — kmong.com/ 홈페이지에서 drill-down 추출 안 일어나
                        # _drill_down_pool 빈 채 → Phase F 미발동 → WRITE 무한 루프 → bypass
                        # → 게으른 WRITE. e-commerce/플랫폼 홈페이지는 본질적으로 카테고리/
                        # gig 카드 listing 이라 detail link 추출 가능. URL path 가 "/" 또는
                        # 비어있고 host 부분 있으면 root listing 으로 간주.
                        _is_root_listing = False
                        try:
                            import re as _re_root
                            _root_m = _re_root.match(
                                r"https?://[^/]+(/?)(?:[?#]|$)", _cur_url_lc
                            )
                            if _root_m:
                                _is_root_listing = True
                        except Exception:
                            pass
                        if any(s in _cur_url_lc for s in _LISTING_DETECT) or _is_root_listing:
                            _DETAIL_PATTERNS = (
                                "/gig/", "/product/", "/products/", "/item/", "/items/",
                                "/post/", "/posts/", "/view/", "/detail/",
                                "/article/", "/service/", "/listing/",
                            )
                            # 상대 경로 → 절대 URL 변환용 scheme+host
                            _host_m = re.match(r'(https?://[^/]+)', obs.get("url") or "")
                            _origin = _host_m.group(1) if _host_m else ""
                            _elements = obs.get("elements") or []
                            _cands_raw: list = []
                            _seen: set = set()
                            for _el in _elements:
                                _href_raw = (_el.get("href") or "").strip()
                                if not _href_raw:
                                    continue
                                if _href_raw.startswith("http"):
                                    _href_abs = _href_raw
                                elif _href_raw.startswith("/") and _origin:
                                    _href_abs = _origin + _href_raw
                                else:
                                    continue  # #anchor, javascript:, mailto: 등 제외
                                _hl = _href_abs.lower()
                                if not any(p in _hl for p in _DETAIL_PATTERNS):
                                    continue
                                _k = _hl.split("#", 1)[0].split("?", 1)[0].rstrip("/")
                                if _k in _seen:
                                    continue
                                _seen.add(_k)
                                _txt = (_el.get("text") or "").strip()[:40]
                                _cands_raw.append({
                                    "i": _el.get("i", "?"),
                                    "url": _href_abs,
                                    "key": _k,
                                    "text": _txt,
                                })
                                if len(_cands_raw) >= 10:
                                    break

                            # [P2 Fix #8 2026-04-24] 인덱서 보강 — gig/product 후보가 5개
                            # 미만이면 JS querySelectorAll 로 직접 추출해 병합. research
                            # stage 에서 SPA 지연/카드 미렌더링으로 인덱서가 카드를 놓쳐
                            # LLM 이 상세로 drift 하는 증상(/gig/244312 3회 NAVIGATE 루프)
                            # 을 DOM 직접 쿼리로 방어.
                            if len(_cands_raw) < 5:
                                try:
                                    from eidos_delivery_adapter import (
                                        SEARCH_LIST_SELECTORS as _P2_SLS,
                                    )
                                    _domain = ""
                                    if _origin:
                                        _domain = (
                                            _origin.split("://", 1)[-1].split("/", 1)[0].lower()
                                        )
                                    _dom_selectors: list = []
                                    for _sk, _sv in _P2_SLS.items():
                                        if _sk in _domain:
                                            _dom_selectors = list(_sv or [])
                                            break
                                    if _dom_selectors:
                                        import json as _json_p2
                                        # 1차: hydration 대기(1s) 후 JS 쿼리
                                        try:
                                            await asyncio.sleep(1.0)
                                        except Exception:
                                            pass
                                        _selectors_json = _json_p2.dumps(_dom_selectors)
                                        _js_collect = (
                                            "(function(){"
                                            f"var selectors={_selectors_json};"
                                            "var out=[];var seen={};"
                                            "for(var i=0;i<selectors.length;i++){"
                                            "  try{"
                                            "    var els=document.querySelectorAll(selectors[i]);"
                                            "    for(var j=0;j<els.length && out.length<30;j++){"
                                            "      var h=els[j].href||els[j].getAttribute('href')||'';"
                                            "      if(!h) continue;"
                                            "      var t=(els[j].textContent||'').trim().substring(0,40);"
                                            "      var k=h.split('#')[0].split('?')[0].replace(/\\/$/,'').toLowerCase();"
                                            "      if(seen[k]) continue;seen[k]=1;"
                                            "      out.push({url:h,key:k,text:t});"
                                            "    }"
                                            "  }catch(e){}"
                                            "}"
                                            "return JSON.stringify(out);"
                                            "})()"
                                        )
                                        from execution_module import (
                                            execute_js_with_result as _exec_p2,
                                        )
                                        _js_result = await _exec_p2(_js_collect, timeout=4.0)
                                        _extra_cands: list = []
                                        if _js_result:
                                            try:
                                                _extra_cands = _json_p2.loads(_js_result) or []
                                            except Exception:
                                                _extra_cands = []
                                        # 여전히 부족하면 PageDown 스크롤 후 재쿼리
                                        if len(_extra_cands) < 5:
                                            try:
                                                await _exec_p2(
                                                    "window.scrollBy(0, window.innerHeight * 1.5);",
                                                    timeout=2.0,
                                                )
                                                await asyncio.sleep(1.2)
                                                _js_result2 = await _exec_p2(
                                                    _js_collect, timeout=4.0,
                                                )
                                                if _js_result2:
                                                    try:
                                                        _extra_cands = _json_p2.loads(_js_result2) or []
                                                    except Exception:
                                                        pass
                                            except Exception:
                                                pass
                                        _added = 0
                                        for _ec in _extra_cands:
                                            _ek = (_ec.get("key") or "").strip()
                                            if not _ek or _ek in _seen:
                                                continue
                                            _seen.add(_ek)
                                            _cands_raw.append({
                                                "i": "js",
                                                "url": _ec.get("url", "") or "",
                                                "key": _ek,
                                                "text": (_ec.get("text") or "").strip()[:40],
                                            })
                                            _added += 1
                                            if len(_cands_raw) >= 30:
                                                break
                                        if _added > 0:
                                            print(
                                                f"  🔎 [Search-JS] 인덱서 보강 — JS 직접 추출로 "
                                                f"{_added}개 추가(총 {len(_cands_raw)}개). domain={_domain}"
                                            )
                                except Exception as _e_js_p2:
                                    print(f"  ⚠️ [Search-JS] 실패 (무시): {_e_js_p2}")
                            # [Phase 1-B 2026-04-24] drill-down pool 누적 + 미방문만 노출.
                            # Why: 이전엔 현재 DOM 의 상위 5개만 힌트로 박아 LLM 이
                            # 매번 같은 첫 gig 로만 NAVIGATE → 1개 표본으로 WRITE 환각.
                            # 풀을 schema 에 보존하고 visited=True 는 NAVIGATE 성공 시
                            # 별도 훅에서 찍는다. 힌트는 미방문 풀에서만 뽑는다.
                            if _cands_raw:
                                if not hasattr(schema, "_drill_down_pool"):
                                    schema._drill_down_pool = []
                                _pool_keys = {
                                    p.get("key") for p in schema._drill_down_pool
                                    if isinstance(p, dict)
                                }
                                for _c in _cands_raw:
                                    if _c["key"] not in _pool_keys:
                                        schema._drill_down_pool.append({
                                            "url": _c["url"],
                                            "key": _c["key"],
                                            "text": _c["text"],
                                            "visited": False,
                                        })
                                        _pool_keys.add(_c["key"])
                                _unvisited = [
                                    p for p in schema._drill_down_pool
                                    if isinstance(p, dict) and not p.get("visited")
                                ]
                                _visited_n = len(schema._drill_down_pool) - len(_unvisited)
                                _hint_items = _unvisited[:5]
                                if _hint_items:
                                    _cands_str = "\n".join(
                                        f"- {p.get('text') or '?'} → {p.get('url', '')[:100]}"
                                        for p in _hint_items
                                    )
                                    _drill = (
                                        f"[드릴다운 후보] 상세 링크 풀 "
                                        f"(방문 {_visited_n} / 미방문 {len(_unvisited)}) — "
                                        f"같은 카테고리/검색 허브 재방문 금지, "
                                        f"이 중 하나로 즉시 NAVIGATE:\n"
                                        + _cands_str
                                    )
                                    if hasattr(schema, "how"):
                                        schema.how = _drill + "\n\n" + (schema.how or "")
                                    print(
                                        f"  🎯 [Drill-down] 풀 {len(schema._drill_down_pool)}개 "
                                        f"(미방문 {len(_unvisited)}) · 힌트 {len(_hint_items)}개 노출"
                                    )
                        else:
                            # [Fix 2026-04-19 v4] 상세(/gig, /product 등) 페이지 도착 시
                            # READ_PAGE 선행 힌트 주입. listing 외 패턴에만 적용.
                            # 증상: LLM이 gig 상세에 들어와도 READ_PAGE 없이 바로
                            # GO_BACK/NAVIGATE로 빠져나가 경쟁 데이터 수집 0건.
                            _DETAIL_ARRIVAL = (
                                "/gig/", "/product/", "/products/", "/item/", "/items/",
                                "/post/", "/posts/", "/article/", "/service/",
                                "/listing/", "/view/", "/detail/",
                            )
                            if any(p in _cur_url_lc for p in _DETAIL_ARRIVAL):
                                _cur_key = _cur_url_lc.split("#", 1)[0].split("?", 1)[0].rstrip("/")
                                _already_read = False
                                for _h in history:
                                    if _h.get("action") != "READ_PAGE":
                                        continue
                                    _h_url = (_h.get("url") or "").lower()
                                    _h_key = _h_url.split("#", 1)[0].split("?", 1)[0].rstrip("/")
                                    if _h_key == _cur_key:
                                        _already_read = True
                                        break
                                if not _already_read:
                                    _read_hint = (
                                        f"[상세 페이지 도착] 현재 URL '{(obs.get('url') or '')[:80]}' 은 "
                                        "개별 상세(서비스/상품/게시물) 페이지다. "
                                        "**다음 액션은 반드시 READ_PAGE**로 본문·가격·리뷰수·설명·옵션을 "
                                        "변수에 먼저 저장하라 (예: READ_PAGE target='gig_detail_1'). "
                                        "READ_PAGE 없이 GO_BACK·NAVIGATE·WRITE로 이탈하면 "
                                        "경쟁 분석 데이터가 0건이 되어 Y변수가 진전하지 않는다. "
                                        "이미 READ_PAGE로 이 URL을 저장했다면 다른 상세 페이지로 이동해 "
                                        "비교군을 늘려라 — 카테고리 허브 재방문은 금지."
                                    )
                                    if hasattr(schema, "how"):
                                        schema.how = _read_hint + "\n\n" + (schema.how or "")
                                    print(f"  📖 [Detail-arrival] READ_PAGE 선행 힌트 주입: {_cur_key[-60:]}")
                    except Exception as _de:
                        pass  # 드릴다운/상세 힌트 실패는 무시

                # [Fix] 동일 URL NAVIGATE 제자리걸음 감지 (verified=True 로 집계됨)
                if action == "NAVIGATE" and target:
                    _nav_key = target.strip().rstrip("/").lower()
                    _nav_counter[_nav_key] = _nav_counter.get(_nav_key, 0) + 1
                    # [Fix] listing/허브 경로(/category, /search, /tag 등)에서는
                    # /search?q= 변형으로 도피해도 의미 없음 — 같은 허브만 맴돔.
                    # 이 경우 variant 훅은 스킵하고 드릴다운 힌트만 주입한다.
                    _LISTING_HUB_SEGS = (
                        "/category", "/categories", "/list", "/listings",
                        "/tag", "/tags", "/explore", "/browse", "/discover",
                    )
                    _nav_path_lc = _nav_key.split("?", 1)[0]
                    _is_listing_repeat = any(s in _nav_path_lc for s in _LISTING_HUB_SEGS)
                    if _nav_counter[_nav_key] >= SAME_NAV_LIMIT and _is_listing_repeat:
                        _drill_hint = (
                            f"⚠️ 허브/목록 경로 '{target[:60]}' {_nav_counter[_nav_key]}회 반복. "
                            f"같은 카테고리/검색 허브로 돌아가지 말고, 이 페이지에서 "
                            f"**구체 상세 페이지로 드릴다운**하라: "
                            f"(a) 현재 목록의 개별 항목 URL(gig/{{id}}, /product/{{id}} 등)로 NAVIGATE, "
                            f"(b) READ_PAGE로 이미 모은 정보가 있으면 WRITE로 기록하고 "
                            f"완전히 다른 경로 탐색, "
                            f"(c) 단서 부족 시 ASK_USER."
                        )
                        self._report(f"🔄 **[허브 반복 차단]** {_drill_hint[:140]}")
                        if schema:
                            schema.how = _drill_hint + "\n\n" + schema.how
                        _nav_counter[_nav_key] = 0
                        # dead-end 블랙리스트에 등록 — Tier 3 허브 면제 무시
                        try:
                            if schema is not None:
                                _de = getattr(schema, "_dead_end_urls", None)
                                if _de is None:
                                    _de = set()
                                    schema._dead_end_urls = _de
                                _de.add(_nav_key)
                        except Exception:
                            pass
                    elif _nav_counter[_nav_key] >= SAME_NAV_LIMIT:
                        # ── 능동 대응: 변형 URL 후보로 자동 이동 ──────────────
                        # 기존에는 schema.how 힌트만 주입했으나, LLM이 동일 URL을
                        # 반복 생성하는 경우를 깨기 위해 검색 엔드포인트·인코딩·
                        # 대체 경로를 직접 네비게이트해 본다.
                        _tried: set = task.setdefault("__nav_variants_tried__", set())
                        _tried.add(_nav_key)
                        _variants = _generate_url_variants(target, task_prompt, _tried)
                        _nav_hint = ""
                        if _variants:
                            _next_var = _variants[0]
                            _tried.add(_next_var.rstrip("/").lower())
                            try:
                                print(f"  🔁 [URL Variant] 동일 URL {_nav_counter[_nav_key]}회 "
                                      f"반복 → 변형 후보로 자동 이동: {_next_var}")
                                await _navigate_and_wait(url=_next_var, timeout=20.0)
                                _nav_hint = (
                                    f"[시스템] '{target[:60]}' 3회 반복 → "
                                    f"자동 변형 '{_next_var[:80]}' 으로 이동 완료. "
                                    f"동일 경로로 돌아가지 말고 현재 페이지의 검색 결과/링크에서 "
                                    f"대상을 찾아 CLICK 또는 다른 키워드로 재검색하라."
                                )
                                self._report(
                                    f"🔁 **[URL 자동 변형]** {_next_var[:80]}"
                                )
                                # 히스토리에도 기록 → 다음 Think에서 LLM이 참조
                                history.append({
                                    "step": step_count,
                                    "action": "NAVIGATE_VARIANT",
                                    "target": _next_var[:100],
                                    "result": f"동일 URL 반복 → 자동 변형 이동",
                                    "verified": True,
                                })
                            except Exception as _ve:
                                _nav_hint = (
                                    f"⚠️ 동일 URL {_nav_counter[_nav_key]}회 반복. "
                                    f"자동 변형 '{_next_var[:60]}' 이동 실패({_ve}). "
                                    f"다른 키워드·URL로 재시도 또는 ASK_USER."
                                )
                        else:
                            _nav_hint = (
                                f"⚠️ 동일 URL '{target[:60]}' 에 {_nav_counter[_nav_key]}회 "
                                f"NAVIGATE 반복. 변형 후보 소진 — "
                                f"(1) 현재 페이지에서 CLICK/FILL 상호작용 "
                                f"(2) 전혀 다른 검색 키워드로 재시도 "
                                f"(3) 대상 특정 불가 시 ASK_USER."
                            )
                        self._report(
                            f"🔄 **[동일 URL 반복 차단]** {_nav_hint[:140]}"
                        )
                        if schema:
                            schema.how = _nav_hint + "\n\n" + schema.how
                            # [2026-04-24] 영구 금지 URL 목록에 등록 — 다음 Think 프롬프트 상단에 하드블록
                            try:
                                _fb_map = getattr(schema, "_forbidden_urls", None)
                                if _fb_map is None:
                                    _fb_map = {}
                                    schema._forbidden_urls = _fb_map
                                _fb_map[target.strip()[:100]] = f"동일 URL {SAME_NAV_LIMIT}회 반복 차단"
                            except Exception:
                                pass
                        _nav_counter[_nav_key] = 0  # 힌트 주입 후 리셋

                # [Fix 2026-04-21 P3] URL 경로(쿼리 제외) 기반 재방문 누적 카운터.
                # Why: _nav_counter 는 쿼리 포함 정확 일치라 /search?keyword=AI 프롬프트 팩 /
                # /search?keyword=AI%20프롬프트%20팩 / /search?keyword=AI 프롬프트 를 전부
                # 다른 키로 취급. 이번 로그에서 Step 27~42 같은 path `/search` 를 10번
                # 넘게 방문하는데 _nav_counter 는 쿼리 인코딩 차이로 터지지 않음.
                # path-only 재방문이 한계치 넘으면 즉시 강제 DONE → 체인 stage 이관.
                if action == "NAVIGATE" and target:
                    try:
                        _path_key = target.strip().split("?", 1)[0].split("#", 1)[0].rstrip("/").lower()
                    except Exception:
                        _path_key = ""
                    if _path_key:
                        _path_counter: Dict[str, int] = task.setdefault(
                            "__nav_path_counter__", {}
                        )
                        _path_counter[_path_key] = _path_counter.get(_path_key, 0) + 1
                        _PATH_HARD_LIMIT = 6  # 같은 path 6회째부터 강제 종료
                        if _path_counter[_path_key] >= _PATH_HARD_LIMIT:
                            _force_msg = (
                                f"🛑 URL 경로 '{_path_key[-80:]}' {_path_counter[_path_key]}회 재방문 — "
                                f"루프 확정. 현재까지 확보한 변수·산출물로 즉시 DONE 또는 ASK_USER. "
                                f"같은 경로로 또 NAVIGATE 하면 동일 결과 반복뿐이다."
                            )
                            self._report(
                                f"🛑 **[경로 루프 강제 중단]** path='{_path_key[-60:]}' "
                                f"count={_path_counter[_path_key]}"
                            )
                            if schema is not None and hasattr(schema, "how"):
                                schema.how = _force_msg + "\n\n" + (schema.how or "")
                                # [2026-04-24] 영구 금지 path 등록
                                try:
                                    _fb_map2 = getattr(schema, "_forbidden_urls", None)
                                    if _fb_map2 is None:
                                        _fb_map2 = {}
                                        schema._forbidden_urls = _fb_map2
                                    _fb_map2[_path_key[:100]] = (
                                        f"경로 {_path_counter[_path_key]}회 재방문 — 루프 확정"
                                    )
                                except Exception:
                                    pass
                            task["__stuck_detected__"] = True
                            # history 에 남겨서 다음 Think 가 바로 DONE 쪽으로 가게 유도
                            history.append({
                                "step": step_count,
                                "action": "PATH_LOOP_BLOCK",
                                "target": _path_key[-100:],
                                "result": _force_msg[:160],
                                "verified": True,
                            })
                            # 카운터 리셋 (중복 발사 방지)
                            _path_counter[_path_key] = 0

                # [Fix 2026-04-19] 동일 target WRITE 반복 감지 — 무한 WRITE 루프 차단.
                # act_result 에 "Shortening Guard" 가 있으면 이미 덮어쓰기가 거부됐다는 뜻,
                # "생성 완료" 여도 같은 target을 계속 쓰는 건 진전 없음.
                #
                # [False alarm 차단 2026-04-27] 게이트 거부된 WRITE (Sampling Gate /
                # PRE-VETO) 는 실제 파일 변경이 없음 → 카운트 skip. 거부 시도가 카운터에
                # 누적되면 bypass 후 정상 성공 시점에 무한루프 차단 메시지가 false alarm 으로
                # 발사되어 사용자에게 혼란. Shortening Guard 만 헛발질 카운트 (실제 시도는 됐음).
                if action == "WRITE" and (target or "").strip():
                    _w_key = target.strip().lower()
                    _ar_lc = act_result or ""
                    _is_shortening_cancel = "Shortening Guard" in _ar_lc
                    _is_gate_reject = (
                        ("WRITE 거부" in _ar_lc)
                        or ("WRITE 사전 거부" in _ar_lc)
                        or ("샘플링 게이트" in _ar_lc and "거부" in _ar_lc)
                    )
                    if _is_gate_reject and not _is_shortening_cancel:
                        # 게이트 거부 — 카운터 동결, hint/force 메시지도 자연스럽게 미발화.
                        _w_n = _write_counter.get(_w_key, 0)
                    else:
                        _write_counter[_w_key] = _write_counter.get(_w_key, 0) + 1
                        _w_n = _write_counter[_w_key]
                    if _w_n >= SAME_WRITE_HINT_AT and schema is not None:
                        # 2회 이상 WRITE된 파일이 충분히 크면(>=800자) 내용-계열
                        # HOW_MUCH 속성을 자동 달성 처리해 루프를 끊는다.
                        # [Fix 2026-04-22] ReAct 루프엔 plan 객체가 없음 → _react_vars 참조.
                        _react_vars = getattr(schema, "_react_vars", {}) or {}
                        _cur_val = _react_vars.get(target.strip(), "") or ""
                        if len(_cur_val) >= 800:
                            _CONTENT_KWS = (
                                "목록", "리스트", "list",
                                "분석", "analysis", "보고서", "report",
                                "내용", "정확성", "요약", "summary",
                                "설명", "기획", "plan", "draft",
                                # plan.md 의 구조 명칭("...섹션 포함 여부", "...개수",
                                # "컨셉 개요") 도 내용-계열로 인정 — 이전엔 미매칭으로
                                # HOW_MUCH 미달성 → Verifier FAIL 무한 반복.
                                "섹션", "포함", "후보", "컨셉", "개요",
                                "타겟", "usp", "가격", "근거", "본문", "초안",
                                "구성", "항목", "상품명",
                            )
                            _promoted = 0
                            for _attr in schema.attributes:
                                if _attr.achieved:
                                    continue
                                # [Fix 2026-04-21] 본문 키워드/N개/존재 검사를 먼저 시도.
                                # 이름 키워드 폴백보다 정확하고, 'CRM 팩' 같은 따옴표
                                # 키워드도 본문에 실제로 들어갔는지로 판정.
                                if self._auto_check_keyword_attr(_attr, _cur_val):
                                    _promoted += 1
                                    continue
                                if self._auto_check_count_attr(_attr, _cur_val):
                                    _promoted += 1
                                    continue
                                _nl = (_attr.name or "").lower()
                                if any(kw in _attr.name or kw in _nl for kw in _CONTENT_KWS):
                                    _attr.achieved = True
                                    _attr.evidence = (
                                        f"WRITE {_w_n}회 반복({len(_cur_val)}자) — "
                                        f"내용-계열 속성 자동 인정"
                                    )
                                    _promoted += 1
                            if _promoted:
                                print(
                                    f"  🛡️ [WRITE-Loop Break] 동일 target '{_w_key[:40]}' "
                                    f"{_w_n}회 반복 감지 — 내용-계열 HOW_MUCH {_promoted}개 자동 달성"
                                )
                        _w_hint = (
                            f"⚠️ 같은 파일 '{target[:60]}' 에 WRITE {_w_n}회 반복. "
                            f"더 이상 같은 target 으로 WRITE 금지. "
                            f"(a) 내용이 충분하면 DONE, "
                            f"(b) 새 정보가 필요하면 READ_PAGE/NAVIGATE 먼저, "
                            f"(c) 다른 파일명이 필요하면 새 target 으로 WRITE."
                        )
                        if hasattr(schema, "how"):
                            schema.how = _w_hint + "\n\n" + (schema.how or "")
                        # [Fix 2026-04-19 v3] 2회 반복도 사용자에게 노출 —
                        # 이전엔 stdout print만 찍혀서 사용자가 WRITE 루프 진입을 인지 못 함.
                        self._report(f"🔁 **[WRITE 반복 감지]** {_w_hint[:140]}")
                        if _w_n >= SAME_WRITE_FORCE_DONE:
                            # 3회째면 강한 DRIFT CORRECTION 흐름으로: schema.how 최상단에
                            # DONE/종료 지시를 박아 다음 _decide_next_action 이 반드시 참조.
                            _force_hint = (
                                f"🛑 WRITE {_w_n}회 반복 — 무한 루프 확정. "
                                f"다음 액션은 반드시 DONE 또는 ASK_USER. "
                                f"같은 파일에 WRITE 하면 단계 낭비다."
                            )
                            if hasattr(schema, "how"):
                                schema.how = _force_hint + "\n\n" + (schema.how or "")
                            self._report(f"🛑 **[WRITE 무한루프 차단]** {_force_hint[:140]}")

                # ── [Chain Stage Auto-DONE] WRITE 산출물이 stage 완료 신호면 즉시 마감. ──
                # research_report.md / plan.md / description.md 등 stage 가 약속한 파일이
                # 디스크에 떨어지면 LLM 의 추가 NAVIGATE 의사와 무관하게 stage 종료.
                # Why: stage 0 research 가 산출물 떨궜는데도 LLM 이 "20개 더 모아야 해"
                # 며 무한 루프 → on_stage_complete 미호출 → plan/dev 로 안 넘어감.
                if (
                    schema is not None
                    and action == "WRITE"
                    and "생성 완료" in (act_result or "")
                    and getattr(schema, "chain_id", "")
                    and (target or "").strip()
                ):
                    try:
                        if self._is_stage_completion_artifact(schema, target.strip()):
                            self._report(
                                f"✅ **[체인 단계 산출물 도착]** stage='{getattr(schema, 'stage', '')}' "
                                f"({getattr(schema, 'stage_index', 0) + 1}) → 산출물 "
                                f"'{target.strip()[:60]}' 저장 완료. 다음 단계로 자동 진행."
                            )
                            task["status"] = "DONE"
                            self._remove_task_from_schedule(task)
                            await self._on_react_done(task_prompt, goal_ref, task, history, schema)
                            return
                    except Exception as _e_acd:
                        print(f"  ⚠️ [Chain Auto-DONE] 판정 실패 (무시): {_e_acd}")

                # ── [Ad-hoc Auto-DONE 2026-04-22] 비-체인 사용자 태스크의 WRITE 자동 종료 ──
                # Why: chain_id 없는 ad-hoc 태스크("서울 날씨 알려줘", "네이버 운세 3줄 요약")는
                # attrs 자동판정이 엄격해 all_attributes_achieved 가 False 로 남고, LLM 이 다음
                # step 에서 DONE 을 스스로 찍지 않으면 ReAct 루프가 좀비로 hang. 다음 사용자
                # 태스크가 _paused_task 경유 이어하기로 붙으면 좀비 2중 공존 상태가 됨.
                # 해결: WRITE 가 실제 파일로 저장됐고(act_result 에 "파일 저장" 토큰) 내용이
                # 충분하면(>=200자) chain 여부와 무관하게 태스크를 DONE 으로 마감.
                if (
                    schema is not None
                    and action == "WRITE"
                    and ("파일 저장" in (act_result or "") or "생성 완료" in (act_result or ""))
                    and not getattr(schema, "chain_id", "")
                    and (target or "").strip()
                ):
                    # [Phase 1-ε 2026-05-01] schema.execution_action 있으면 WRITE 만으로
                    # 자동 종료 차단 — 외부 액션 (NAVIGATE/CLICK/FILL/SCROLL/GO_BACK/CLICK_XY)
                    # 1건 이상 발생 후 종료 허용. 없으면 ReAct 루프가 다음 step 으로 자연 진입.
                    # 첫번째 dogfooding 사례: "텔레그램으로 보내줘" task 가 WRITE telegram_*.md
                    # 1건 만 만들고 SEND 0건으로 종료 → 사용자 의도 위반. 이 가드가 본질 차단.
                    _exec_action_dict = getattr(schema, "execution_action", None) or {}
                    _ext_count_now = int(task.get("__external_action_count__", 0) or 0)
                    _ea_block_autodone = (
                        isinstance(_exec_action_dict, dict)
                        and bool(_exec_action_dict.get("action"))
                        and _ext_count_now == 0
                    )
                    if _ea_block_autodone:
                        if not task.get("__write_only_warn_emitted__"):
                            self._report(
                                f"🚫 **[Auto-DONE 차단]** schema.execution_action="
                                f"`{_exec_action_dict.get('action')}` 명시 — WRITE 산출물 저장됐으나 "
                                f"외부 액션 (NAVIGATE/CLICK/FILL) 0건. 다음 step 에서 외부 행위 시도 필요."
                            )
                            print(
                                f"  🚫 [Phase 1-ε] Auto-DONE 차단 — schema.execution_action="
                                f"'{_exec_action_dict.get('action')}', _ext_count={_ext_count_now}"
                            )
                            task["__write_only_warn_emitted__"] = True
                        # Auto-DONE 분기 skip — 루프가 다음 step 으로 진입 (LLM 이 외부 행위 시도 기회)
                    else:
                        try:
                            _adhoc_key = target.strip()
                            _adhoc_val = ""
                            # [Fix 2026-04-22] ReAct 루프는 plan 없음 — schema._react_vars 가
                            # 변수의 단일 진실원. 이전엔 plan 만 보고 항상 0자로 집계되어
                            # Auto-DONE 이 한 번도 발동 못 함 (네이버 운세 WRITE 후 좀비).
                            _react_vars = getattr(schema, "_react_vars", {}) or {}
                            _adhoc_val = (_react_vars.get(_adhoc_key, "") or "")
                            if not _adhoc_val:
                                # WRITE 자동 승격 경로: 변수명이 확장자 없는 상태로 저장됐을 수 있음
                                _bare = os.path.splitext(_adhoc_key)[0]
                                _adhoc_val = (_react_vars.get(_bare, "") or "")
                            if len(_adhoc_val) >= 200:
                                self._report(
                                    f"✅ **[작업 완료]** 산출물 '{_adhoc_key[:60]}' "
                                    f"({len(_adhoc_val)}자) 저장 완료 — 태스크 자동 종료."
                                )
                                task["status"] = "DONE"
                                self._remove_task_from_schedule(task)
                                await self._on_react_done(task_prompt, goal_ref, task, history, schema)
                                return
                        except Exception as _e_adone:
                            print(f"  ⚠️ [Ad-hoc Auto-DONE] 판정 실패 (무시): {_e_adone}")

                # HOW_MUCH 속성 자동 달성 체크
                if schema:
                    self._auto_check_attributes(schema, action, act_result, obs, plan=plan)
                    if schema.all_attributes_achieved:
                        # [Phase 1-ε 2026-05-01] 속성 달성 종료 가드 — execution_action 명시 +
                        # 외부 액션 0건이면 종료 차단. WRITE 등 분석성 액션이 attrs 자동 인정만
                        # 받고 외부 행위 없이 끝나는 사례 차단.
                        _exec_action_d2 = getattr(schema, "execution_action", None) or {}
                        _ext_c2 = int(task.get("__external_action_count__", 0) or 0)
                        if (isinstance(_exec_action_d2, dict)
                                and bool(_exec_action_d2.get("action"))
                                and _ext_c2 == 0):
                            if not task.get("__attrs_only_warn_emitted__"):
                                self._report(
                                    f"🚫 **[속성 달성 종료 차단]** schema.execution_action="
                                    f"`{_exec_action_d2.get('action')}` 명시 — HOW_MUCH 모두 충족됐으나 "
                                    f"외부 액션 (NAVIGATE/CLICK/FILL) 0건. 외부 행위 시도 후 종료 가능."
                                )
                                print(
                                    f"  🚫 [Phase 1-ε] 속성-달성-종료 차단 — execution_action="
                                    f"'{_exec_action_d2.get('action')}', _ext_count={_ext_c2}"
                                )
                                task["__attrs_only_warn_emitted__"] = True
                            # 종료 분기 skip — 다음 step 으로 자연 진입
                        else:
                            self._report(
                                f"🎯 **[모든 속성 달성!]** HOW_MUCH 전부 충족\n"
                                f"{schema.achievement_summary}"
                            )
                            await self._on_react_done(task_prompt, goal_ref, task, history, schema)
                            return
            else:
                fail_streak += 1
                _last_fail_msg = verify_msg  # 8-4: 마지막 실패 원인 저장

                # 11-1+11-2: 동일 action+target 반복 실패 감지 → 전략 변경 힌트
                _fail_key = f"{action}:{(target or content)[:60]}"
                _fail_counter[_fail_key] = _fail_counter.get(_fail_key, 0) + 1
                if _fail_counter[_fail_key] >= SAME_TARGET_FAIL_LIMIT:
                    _alt_hint = (
                        f"⚠️ [{action}] '{(target or content)[:40]}' 동일 시도 "
                        f"{_fail_counter[_fail_key]}회 실패. "
                        f"다른 selector/방법을 사용하거나 ASK_USER로 전환하라."
                    )
                    self._report(f"🔄 **[전략 변경 필요]** {_alt_hint}")
                    # schema.how에 힌트 주입 (다음 Think에서 LLM이 참조)
                    if schema:
                        schema.how = _alt_hint + "\n\n" + schema.how
                    # _fail_counter 초기화 (힌트 주입 후 한 번 더 기회 부여)
                    _fail_counter[_fail_key] = 0

                self._report(
                    f"⚠️ **[Step {step_count} 실패]** [{action}] {(target or content)[:50]}\n"
                    f"└─ {verify_msg[:80]} (연속 실패: {fail_streak}/{MAX_FAIL_STREAK})"
                )
                # 브라우저 액션 실패 시 잠시 대기 후 재시도
                if action in {"NAVIGATE", "FILL", "CLICK", "CHECK_LOGIN"}:
                    await asyncio.sleep(2.0)

    # ── Strategy Layer — ASK_USER 전 자동 에스컬레이션 ──────────────────────
    async def _try_strategy_escalation(
        self,
        task: Dict,
        schema,
    ) -> Optional[Dict]:
        """
        LLM이 ASK_USER를 내기 직전 호출. 인간에게 묻기 전 자동 탐색 시도.
          Tier 1: 현재 페이지 스크롤 (한 번만) → lazy-render 요소 노출 가능
          Tier 2: schema.verb별 사이트맵 후보로 NAVIGATE (안 가본 경로 1개)
          Tier 3: 소진 → None 반환 → ASK_USER 정상 경로로 폴백

        task 딕셔너리의 __strategy_state__에 상태 저장 → task마다 격리.
        반환: 에스컬레이션 수행 시 기록 dict, 더 할 게 없으면 None.
        """
        state = task.setdefault("__strategy_state__", {
            "scrolled": False,
            "attempted_paths": [],
        })

        # ── Stuck 감지 상태면 Tier 1/2 건너뛰고 곧장 Tier 3 (GO_BACK) ─────
        # 스크롤/사이트맵이 이미 의미 없다고 판단된 상황 → 뒤로 돌아가 재탐색.
        _stuck = bool(task.get("__stuck_detected__"))
        if _stuck:
            print(f"  [Strategy] __stuck_detected__=True → Tier 1/2 스킵, Tier 3 시도")
            state["scrolled"] = True  # Tier 1 스킵
            # Tier 2/2a도 실질적으로 스킵되도록 candidates 비우기 위한 플래그
            state["_skip_to_back"] = True
            # 다음 ASK_USER 때는 다시 정상 흐름
            task["__stuck_detected__"] = False

        # ── Tier 1: 스크롤 한 번도 안 했으면 먼저 실행 ───────────────────
        if not state["scrolled"]:
            state["scrolled"] = True
            try:
                _js = (
                    "(function() {"
                    "  var before = window.scrollY;"
                    "  window.scrollBy(0, Math.round(window.innerHeight * 0.8));"
                    "  return 'SCROLLED:' + before + '->' + window.scrollY;"
                    "})();"
                )
                _r = await _browser_run(script=_js)
                await asyncio.sleep(0.4)  # lazy-load 이미지/위젯 대기
                print(f"  🔍 [Strategy Tier 1] 스크롤 실행 → {str(_r)[:50]}")
                return {
                    "action": "SCROLL",
                    "target": "viewport ~80%",
                    "result": "Tier1 스크롤 완료 — 재-Observe로 새 요소 확인",
                    "tier": 1,
                }
            except Exception as _e:
                print(f"  [Strategy Tier 1] 스크롤 실패 (무시): {_e}")

        # ── Tier 2: verb별 사이트맵 후보 NAVIGATE ─────────────────────────
        # [Layer 1] 인증 블랙리스트가 있으면 Tier 2 자체를 스킵.
        # 로그인 리다이렉트/비인증 SPA를 한 번이라도 맞았다면 다른 경로도 같은 결과일 가능성 매우 높음.
        _auth_bl = task.get("__auth_blacklist__") or []
        if _auth_bl:
            print(f"  [Strategy Tier 2] 인증 블랙리스트 존재({len(_auth_bl)}개) — Tier 2 스킵")
            return None

        _verb = getattr(schema, "verb", "") if schema else ""
        _candidates = SITEMAP_BY_VERB.get(_verb, [])

        # 도메인 도출: schema.where 우선, 없으면 현재 URL
        _base_host = ""
        _cur_url = ""
        try:
            _where = (getattr(schema, "where", "") or "") if schema else ""
            _m = re.search(r'([\w-]+\.[\w.]+)', _where)
            if _m:
                _base_host = _m.group(1).lower()
            _cur_url = self._get_current_browser_url() or ""
            if not _base_host:
                _cm = re.search(r'^https?://([^/]+)', _cur_url)
                if _cm:
                    _base_host = _cm.group(1).lower()
        except Exception:
            pass

        if not _base_host:
            return None

        # ── Tier 2a: READ 작업은 파라메트릭 검색 변형 먼저 시도 ───────────
        # 기존에는 고정 경로(/search, /browse)만 썼으나, 검색 키워드가 있으면
        # /Search?q=... 같은 실제로 결과가 나오는 엔드포인트를 우선 써본다.
        # 이게 나무위키/위키피디아에서 /search 빈 페이지 문제를 푸는 핵심 고리.
        if _verb == "READ" and not state.get("_skip_to_back"):
            _tp = task.get("task_prompt", "") or ""
            _already_tried = set(state["attempted_paths"])
            # 시드 URL: 현재 URL이 있으면 그것, 없으면 호스트만
            _seed = _cur_url if _cur_url else f"https://{_base_host}/"
            _vlist = _generate_url_variants(_seed, _tp, _already_tried)
            if _vlist:
                _next_var = _vlist[0]
                state["attempted_paths"].append(_next_var)
                try:
                    print(f"  🔍 [Strategy Tier 2a] 검색 변형 시도: {_next_var}")
                    await _navigate_and_wait(url=_next_var, timeout=15.0)
                    return {
                        "action": "TRY_SEARCH_VARIANT",
                        "target": _next_var,
                        "result": f"Tier2a 검색 변형 이동 완료 — 재-Observe 진행",
                        "tier": 2,
                    }
                except Exception as _ne:
                    print(f"  [Strategy Tier 2a] 이동 실패 ({_next_var}): {_ne}")

        # stuck 상태거나 후보가 없으면 Tier 2 스킵 → 바로 Tier 3 (back)
        _skip_tier2 = state.get("_skip_to_back") or not _candidates

        # 안 가본 경로 + 인증 블랙리스트 미포함 경로 중 첫 번째 선택
        _next_path = None
        for _p in ([] if _skip_tier2 else _candidates):
            _full = f"https://{_base_host}{_p}"
            if _full in state["attempted_paths"]:
                continue
            if _p in _auth_bl:
                continue
            _next_path = _p
            break

        if _next_path:
            _full_url = f"https://{_base_host}{_next_path}"
            state["attempted_paths"].append(_full_url)
            try:
                print(f"  🔍 [Strategy Tier 2] 사이트맵 후보 시도: {_full_url} (verb={_verb})")
                await _navigate_and_wait(url=_full_url, timeout=15.0)
                return {
                    "action": "TRY_SITEMAP",
                    "target": _full_url,
                    "result": f"Tier2 사이트맵 '{_next_path}' 이동 완료 — 재-Observe 진행",
                    "tier": 2,
                }
            except Exception as _ne:
                print(f"  [Strategy Tier 2] 이동 실패 ({_full_url}): {_ne}")
        # _next_path 없거나 Tier 2 실패 → 아래 Tier 3 시도

        # ── Tier 3: 막힘 시 브라우저 뒤로가기 (한 태스크당 1회) ──────────────
        # 현재 페이지를 dead-end로 표시해서 재진입 차단 + 이전 페이지로 복귀.
        # LLM이 복귀 페이지에서 다른 링크/버튼을 재-Observe로 보고 탐색하도록.
        if not state.get("back_attempted"):
            state["back_attempted"] = True
            try:
                _cur = self._get_current_browser_url() or ""
                # [P2 Fix] VERB=READ(browser_read 등) 태스크에서 이미 페이지
                # 콘텐츠를 확보(_react_vars에 page_* 저장)했다면 GO_BACK은 핑퐁만
                # 만든다. 사용자가 지정한 페이지에 도달했는데 뒤로가면
                # naver.com → namu.wiki → naver.com 사이클이 반복.
                # 또한 schema.where 호스트가 현재 URL에 포함돼도 동일하게 차단.
                _verb = (getattr(schema, "verb", "") or "").upper() if schema else ""
                _ttype = (getattr(schema, "task_type", "") or "") if schema else ""
                _is_read_task = (_verb == "READ" or _ttype == "browser_read")
                if _is_read_task and _cur:
                    _block_back = False
                    _block_reason = ""
                    # (a) 이미 페이지 콘텐츠를 변수에 확보했음
                    try:
                        _vars = getattr(schema, "_react_vars", {}) or {}
                        _has_content = any(
                            (k.startswith("page_") or k.endswith("_page")
                             or k.endswith("_content") or "content" in k.lower())
                            and len(str(v) or "") > 200
                            for k, v in _vars.items()
                        )
                        if _has_content:
                            # ── [Phase E 2026-04-27] sampling counter 미달 시 차단 해제 ──
                            # Why: 최신로그 — list page 만 모아서 _has_content=True 인데
                            # detail counter=0. GO_BACK 차단되면 list 에 갇혀 detail
                            # 진입 불가. sampling 태스크 (_drill_down_pool 존재) + detail
                            # < threshold 면 차단 해제 → list 복귀 후 다른 detail 진입.
                            _sampling_unmet = False
                            try:
                                if getattr(schema, "_drill_down_pool", None):
                                    _vurls_e = getattr(schema, "_react_var_urls", {}) or {}
                                    _detail_n_e = 0
                                    for _u_e, _vn_e in _vurls_e.items():
                                        if _classify_sampling_url(_u_e) != "detail":
                                            continue
                                        _val_e = _vars.get(_vn_e, "")
                                        if isinstance(_val_e, str) and len(_val_e) >= 200:
                                            _detail_n_e += 1
                                    _saved_N_e = int(
                                        getattr(schema, "_n_required_downgraded", 0) or 0
                                    )
                                    _threshold_e = min(_saved_N_e, 5) if _saved_N_e else 5
                                    if _detail_n_e < _threshold_e:
                                        _sampling_unmet = True
                                        print(
                                            f"  🔓 [Phase E] sampling 태스크 "
                                            f"detail={_detail_n_e}/{_threshold_e} 미달 "
                                            f"→ GO_BACK 차단 해제 (list 복귀 → 다른 detail 진입)"
                                        )
                            except Exception:
                                pass
                            if not _sampling_unmet:
                                _block_back = True
                                _block_reason = "READ 콘텐츠 이미 확보"
                    except Exception:
                        pass
                    # (b) where 호스트가 현재 URL에 포함됨
                    if not _block_back:
                        try:
                            _where = (getattr(schema, "where", "") or "").strip().lower()
                            _where_host = (
                                _where.replace("https://", "")
                                      .replace("http://", "")
                                      .split("/")[0]
                            )
                            if (_where_host
                                    and "." in _where_host
                                    and _where_host in _cur.lower()):
                                _block_back = True
                                _block_reason = f"where 도메인('{_where_host}') 도달"
                        except Exception:
                            pass
                    if _block_back:
                        print(
                            f"  🛡️ [Strategy Tier 3] READ 태스크 — GO_BACK 차단 "
                            f"({_block_reason}): {_cur[:60]}"
                        )
                        return None
                _js_back = (
                    "(function(){"
                    "var u1=window.location.href;"
                    "if(window.history.length>1){window.history.back();return 'BACK_SENT:'+u1;}"
                    "return 'NO_HISTORY:'+u1;"
                    "})();"
                )
                _r = await _browser_run(script=_js_back)
                _ret = str(_r or "")
                if "NO_HISTORY" in _ret:
                    print(f"  [Strategy Tier 3] 뒤로가기 불가: history 비어있음")
                else:
                    await asyncio.sleep(1.5)  # 페이지 전환 대기
                    _after = self._get_current_browser_url() or ""
                    # [P1 Fix] GO_BACK이 about:blank(브릿지 페이지)로 떨어졌으면
                    # 한 번 더 history.back()을 보내 진짜 이전 유효 페이지로 스킵한다.
                    if _after.startswith("about:blank"):
                        print(f"  🔙 [Strategy Tier 3] about:blank 브릿지 감지 — 한 번 더 back 시도")
                        try:
                            await _browser_run(script=_js_back)
                            await asyncio.sleep(1.5)
                            _after = self._get_current_browser_url() or ""
                        except Exception as _be2:
                            print(f"  [Strategy Tier 3] 이중 back 실패: {_be2}")
                    # 이중 back 후에도 여전히 about:blank이면 유효 이전 페이지 없음 → Tier 3 실패
                    if _after.startswith("about:blank") or not _after:
                        print(f"  ⚠️ [Strategy Tier 3] 유효한 이전 페이지 없음 — "
                              f"dead-end 등록 스킵 (원본 URL: {_cur[:60]})")
                        # Tier 3 실패 → None 반환해 상위가 ASK_USER로 폴백
                    else:
                        # [P1 Fix] 사용자 지정 URL/도메인은 dead-end 등록 면제.
                        # schema.where가 현재 URL에 포함되면 사용자가 명시한 대상 도메인 →
                        # 네트워크 오류·렌더링 문제로 일시적으로 막혀도 재진입을 금지하면
                        # 복구 경로가 사라짐. 대신 GO_BACK만 수행하고 블랙리스트는 건너뜀.
                        _is_user_target = False
                        try:
                            _where = (getattr(schema, "where", "") or "").strip().lower() if schema is not None else ""
                            _cur_l = (_cur or "").lower()
                            if _where and _cur_l:
                                # where가 도메인이든 경로든 부분 일치하면 user target으로 간주
                                _where_host = _where.replace("https://", "").replace("http://", "").split("/")[0]
                                if _where_host and _where_host in _cur_l:
                                    _is_user_target = True
                        except Exception:
                            _is_user_target = False
                        # [Fix 2026-04-19] 허브/카테고리/검색 URL은 원래 되돌아가야 하는
                        # 다회 방문 페이지. 여기를 dead-end로 등록하면 크몽 /category/6 처럼
                        # 탐색의 기점이 되는 허브로 재진입이 봉쇄돼 recovery 경로가 사라짐.
                        # Why: 로그 2026-04-19 10:11 — /category/6 dead-end 등록 → Step 7 차단 →
                        #      gig 상세에서 CLICK [11] no-op 3회 → ASK_USER. 카테고리가 막힌
                        #      순간 agent는 합리적 다음 수단이 없었음.
                        _HUB_URL_PATTERNS = (
                            "/category", "/categories", "/search", "/tag/", "/tags/",
                            "/list", "/explore", "/browse", "/discover",
                            "?q=", "?query=", "?keyword=", "?search=",
                        )
                        _cur_lc = (_cur or "").lower()
                        _cur_key = (_cur or "").rstrip("/").lower()
                        _is_hub = any(p in _cur_lc for p in _HUB_URL_PATTERNS)
                        # [Fix 2026-04-19 v2] 허브 면제는 '재방문 허용'이 목적이지만,
                        # 같은 허브 URL에서 Tier 3가 2번 이상 발동하면 이미 그 허브는
                        # 탐색 기점으로도 무의미 — 면제 취소하고 dead-end 등록.
                        # Why: 로그 10:22~10:24 — /category/6이 Tier 3 발동 후에도 dead-end
                        #      면제로 재진입 허용 → 3회 반복 끝 stuck 루프.
                        _hub_revisit = task.setdefault("__hub_tier3_revisits__", {})
                        _hub_revisit[_cur_key] = _hub_revisit.get(_cur_key, 0) + 1
                        _hub_exhausted = _is_hub and _hub_revisit[_cur_key] >= 2
                        if _is_user_target:
                            print(f"  🛡️ [Strategy Tier 3] 사용자 지정 대상 도메인(where='{_where[:40]}') — "
                                  f"dead-end 등록 면제: {_cur[:60]}")
                        elif _is_hub and not _hub_exhausted:
                            print(f"  🛡️ [Strategy Tier 3] 허브/탐색 페이지 — dead-end 등록 면제: {_cur[:60]}")
                        else:
                            # 현재 URL을 dead-end 블랙리스트에 추가
                            if _cur and schema is not None:
                                _de = getattr(schema, "_dead_end_urls", None)
                                if _de is None:
                                    _de = set()
                                    schema._dead_end_urls = _de
                                _de.add(_cur_key)
                            if _hub_exhausted:
                                _drill_msg = (
                                    f"⚠️ 허브 '{_cur[:60]}' Tier3 {_hub_revisit[_cur_key]}회 반복 — "
                                    f"dead-end 강제 등록. 이 URL로 돌아가지 말고 "
                                    f"개별 항목 상세(gig/{{id}}·/product·/post 등)로 드릴다운하거나 "
                                    f"완전히 다른 경로/도메인을 탐색하라."
                                )
                                print(f"  🚫 [Strategy Tier 3] {_drill_msg[:120]}")
                                if schema is not None and hasattr(schema, "how"):
                                    schema.how = _drill_msg + "\n\n" + (schema.how or "")
                        _exempt_reason = (
                            " (dead-end 면제)" if _is_user_target
                            else " (허브 면제)" if (_is_hub and not _hub_exhausted)
                            else " (허브 반복→dead-end 강제)" if _hub_exhausted
                            else " (dead-end 등록)"
                        )
                        print(f"  🔙 [Strategy Tier 3] 막힘 감지 — 뒤로가기 실행: "
                              f"{_cur[:50]} → {_after[:50]}{_exempt_reason}")
                        return {
                            "action": "GO_BACK",
                            "target": _after or "prev-page",
                            "result": f"Tier3 뒤로가기 완료 — 이전 페이지에서 다른 경로 탐색",
                            "tier": 3,
                        }
            except Exception as _be:
                print(f"  [Strategy Tier 3] 뒤로가기 실패: {_be}")

        return None  # 모든 Tier 소진 → ASK_USER 폴백

    # ── Observe ────────────────────────────────────────────────────────────
    async def _observe(self) -> Dict:
        """현재 브라우저 상태 관찰 — URL + 클릭 가능 요소 목록 반환"""
        # URL 조회: gui_ref → execution_module fallback 순서
        current_url = self._get_current_browser_url()
        if not current_url:
            # execution_module의 JS 실행으로 URL 조회 시도
            try:
                from execution_module import execute_js_with_result as _exec
                url_result = await _exec("window.location.href", timeout=3.0)
                if url_result and url_result.startswith("http"):
                    current_url = url_result.strip()
            except Exception:
                pass

        result = {
            "url": current_url,
            "clickable": "",
            "html_len": 0,
            "auth_required": False,
            "unhydrated": False,
        }
        try:
            from execution_module import get_browser_content as _gc
            # [Fix 2026-04-22] raw=True: truncate/prefix 제거. 기본값은 30KB 컷 +
            # '[HTML Source (Truncated)]' prefix 를 붙여 len() ~30052 상수가 돼
            # _is_unhydrated_spa 판정을 오염시켰던 버그(로그인 루프 원인).
            raw_html = await _gc(raw=True)
            result["html_len"] = len(raw_html)

            # ── 페이지 지문(page_hash) — stuck 감지용 ───────────────────────
            # script/style/nonce/csrf 등 매번 달라지는 부분 제거 후 해시.
            # ReAct 루프가 연속 N회 같은 지문을 보면 "진전 없음"으로 판단.
            try:
                import hashlib as _hl
                _norm = re.sub(r'<script[^>]*>[\s\S]*?</script>', '', raw_html or "", flags=re.I)
                _norm = re.sub(r'<style[^>]*>[\s\S]*?</style>', '', _norm, flags=re.I)
                _norm = re.sub(r'\b(?:nonce|csrf|_token|timestamp|time|expire)[^"\'\s>]{0,60}', '', _norm, flags=re.I)
                _norm = re.sub(r'\s+', ' ', _norm).strip()
                result["page_hash"] = _hl.sha256(_norm.encode("utf-8", errors="ignore")).hexdigest()[:16]
            except Exception:
                result["page_hash"] = ""

            # ── 팝업/모달 자동 닫기 (광고·유도 레이어 차단) ─────────────────
            # 크몽, 네이버 등 주요 사이트 팝업을 JS로 자동 제거 후 재추출
            try:
                from execution_module import browser_run_script as _brs
                _popup_js = """
(function() {
    var closed = 0;
    // 1. 일반적인 닫기 버튼 패턴
    var closeSelectors = [
        'button[class*="close"]', 'button[class*="Close"]',
        'button[aria-label*="close"]', 'button[aria-label*="닫기"]',
        '[class*="modal"] button', '[class*="popup"] button',
        '[class*="dialog"] button[class*="close"]',
        '[class*="layer"] button[class*="close"]',
        '[class*="overlay"] button',
        'button[class*="dismiss"]',
        // 크몽 전용
        '[class*="Modal"] button', '[class*="Popup"] button',
        '[data-testid*="close"]', '[data-dismiss]',
    ];
    closeSelectors.forEach(function(sel) {
        try {
            var btns = document.querySelectorAll(sel);
            btns.forEach(function(btn) {
                var txt = (btn.textContent || btn.innerText || '').trim();
                // X, 닫기, close 텍스트이거나 비어있는(아이콘) 버튼
                if (!txt || txt === 'X' || txt === 'x' || txt === '×' ||
                    txt.includes('닫기') || txt.toLowerCase().includes('close') ||
                    btn.innerHTML.includes('svg') || btn.innerHTML.includes('icon')) {
                    btn.click();
                    closed++;
                }
            });
        } catch(e) {}
    });
    // 2. 오버레이 직접 제거
    var overlaySelectors = [
        '[class*="modal-backdrop"]', '[class*="overlay"]',
        '[class*="dim"]', '[class*="Dim"]',
    ];
    overlaySelectors.forEach(function(sel) {
        try {
            document.querySelectorAll(sel).forEach(function(el) {
                el.style.display = 'none';
                closed++;
            });
        } catch(e) {}
    });
    return closed;
})();
"""
                await _brs(script=_popup_js)
                # 팝업 닫기 요청만 전송 — 결과 수신 없이 진행 (스레드 안전)
            except Exception as _pe:
                pass  # 팝업 닫기 실패는 무시

            # ── [PRIMARY] A11Y 인덱서: DOM 직접 질의 → 인덱싱된 엘리먼트 JSON ──
            # 성공 시 raw HTML 파싱 전체(regex/__NEXT_DATA__/gig retry)를 스킵하여
            # LLM 토큰 소비 및 CPU 시간 대폭 절감. 실패/빈결과 시 legacy 경로로 폴백.
            _indexed_elements = None
            try:
                from execution_module import execute_js_with_result as _ejs
                _idx_json = await _ejs(_A11Y_INDEXER_JS, timeout=4.0)
                if _idx_json and _idx_json.strip().startswith("{"):
                    _idx_data = json.loads(_idx_json)
                    if _idx_data.get("ok") and _idx_data.get("count", 0) >= 1:
                        _indexed_elements = _idx_data.get("elements", [])
                    elif not _idx_data.get("ok"):
                        print(f"  ⚠️ [Observe] 인덱서 JS 오류: {_idx_data.get('error', '?')[:80]}")
            except Exception as _iex:
                print(f"  ⚠️ [Observe] 인덱서 실패 → regex 폴백: {_iex}")

            if _indexed_elements:
                _lines = []
                for _el in _indexed_elements:
                    _kind = _el.get("kind", "?")
                    _text = _el.get("text", "")
                    _line = f"[{_el['i']}] {_kind} \"{_text}\""
                    _href = _el.get("href", "")
                    if _href:
                        _line += f" → {_href}"
                    elif _el.get("placeholder"):
                        _line += f" (placeholder: {_el['placeholder']})"
                    _lines.append(_line)
                result["clickable"] = "\n".join(_lines)
                result["elements"] = _indexed_elements
                print(f"  ✅ [Observe] 인덱서: {len(_indexed_elements)}개 요소 (raw HTML 우회)")

                # 스크린샷 캡처 (Vision 좌표 폴백 / 사용자 확인용)
                try:
                    from execution_module import grab_browser_screenshot as _gbs
                    _ss = await _gbs(timeout=4.0)
                    if _ss:
                        result["screenshot"] = _ss
                        print(f"  📸 [Observe] 스크린샷 캡처 성공: {_ss}")
                except Exception:
                    pass
                # [Fix 2026-04-22] 인덱서 경로에도 Hydration Recheck 적용.
                # Why: 기존엔 인덱서가 요소를 찾았다는 이유로 여기서 early return 하여
                # 아래 4488 라인의 hydration recheck 를 완전히 스킵. 결과: SPA 가
                # 아직 hydrate 안 됐고 placeholder 상태(elems<=5)에서 auth_required 를
                # 오판 → 로그인된 세션도 ASK_USER 무한루프. 로그에 '[Observe] unhydrated
                # 의심' 이 0회 찍힌 원인. 인덱서 경로에서 auth_gated+thin 조건이면
                # 한 번 더 대기 후 재측정.
                await self._try_hydration_recheck(result, len(_indexed_elements))
                self._annotate_auth_state(result, len(_indexed_elements))
                self._annotate_captcha_state(result, raw_html)
                return result

            print(f"  [Observe] 인덱서 비활성/빈결과 → legacy regex 폴백")

            # ── HTML에서 Next.js __NEXT_DATA__ 및 gig 링크 직접 파싱 (스레드 안전) ──
            _js_links = []
            try:
                import json as _json

                # 방법 1: __NEXT_DATA__ JSON에서 gig 정보 추출
                _nd_m = _re.search(
                    r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>',
                    raw_html[:200000], _re.DOTALL
                )
                if _nd_m:
                    try:
                        _nd = _json.loads(_nd_m.group(1))
                        # gigs 데이터 재귀 탐색
                        def _extract_gigs(obj, results, depth=0):
                            if depth > 10 or len(results) >= 30:
                                return
                            if isinstance(obj, dict):
                                gig_id = (obj.get('gig_id') or obj.get('gigId') or
                                          obj.get('gigID') or obj.get('gig_pk') or
                                          obj.get('id') or obj.get('pk'))
                                title = (obj.get('title') or obj.get('name') or
                                         obj.get('service_title') or obj.get('gig_title') or '')
                                price = (obj.get('price') or obj.get('min_price') or
                                         obj.get('base_price') or '')
                                if gig_id and str(gig_id).isdigit() and int(str(gig_id)) > 100:
                                    price_str = f" {price}원" if price else ''
                                    results.append(
                                        f"a | {str(title)[:40]}{price_str} | https://kmong.com/gig/{gig_id}"
                                    )
                                for v in obj.values():
                                    _extract_gigs(v, results, depth+1)
                            elif isinstance(obj, list):
                                for item in obj:
                                    _extract_gigs(item, results, depth+1)
                        _extract_gigs(_nd, _js_links)
                    except Exception:
                        pass

                # 방법 2: 다양한 패턴으로 gig ID 추출
                if len(_js_links) < 5:
                    _gig_ids_seen = set()
                    # 2-1: href="/gig/숫자" (직접)
                    for _gm in _re.finditer(r'/gig/(\d{4,7})', raw_html):
                        _gid = _gm.group(1)
                        if _gid not in _gig_ids_seen:
                            _gig_ids_seen.add(_gid)
                            _js_links.append(f"a | [상품{_gid}] | https://kmong.com/gig/{_gid}")
                        if len(_js_links) >= 30: break
                    # 2-2: "gigId":숫자 또는 "gig_id":숫자 패턴
                    for _gm in _re.finditer(r'gigId[^:]*:[^\d]*(\d{4,7})', raw_html):
                        _gid = _gm.group(1)
                        if _gid not in _gig_ids_seen:
                            _gig_ids_seen.add(_gid)
                            _js_links.append(f"a | [상품{_gid}] | https://kmong.com/gig/{_gid}")
                        if len(_js_links) >= 30: break
                    # 2-3: "url":"/gig/숫자" 패턴
                    for _gm in _re.finditer(r'"url"\s*:\s*"[/\\]+gig[/\\]+(\d{4,7})"', raw_html):
                        _gid = _gm.group(1)
                        if _gid not in _gig_ids_seen:
                            _gig_ids_seen.add(_gid)
                            _js_links.append(f"a | [상품{_gid}] | https://kmong.com/gig/{_gid}")
                        if len(_js_links) >= 30: break
                    print(f"  [Observe] gig ID 추출: {len(_js_links)}개 / 방법2 패턴")

            except Exception as _jse:
                pass  # 파싱 실패 시 HTML SPA 파싱으로 폴백

            # ── 스크린샷 캡처 (Vision 기반 좌표 클릭을 위해) ─────────────────
            _screenshot_path = ""
            try:
                from execution_module import grab_browser_screenshot as _gbs
                _screenshot_path = await _gbs(timeout=4.0)
                if _screenshot_path:
                    print(f"  📸 [Observe] 스크린샷 캡처 성공: {_screenshot_path}")
                    result["screenshot"] = _screenshot_path
                # 실패해도 무시 — _think는 항상 진행
            except Exception as _sse:
                pass  # 스크린샷 실패는 무시하고 계속 진행

            import re as _re

            # ── input/select 요소 별도 수집 (검색창, 폼 필드용) ──────────────
            _input_items = []
            for _m in _re.finditer(
                r'<input[^>]*?(?:name|id|placeholder|class)="([^"]{1,60})"[^>]*>',
                raw_html[:150000], _re.IGNORECASE
            ):
                _attr = _m.group(1).strip()
                if _attr and len(_attr) > 1:
                    _type_m = _re.search(r'type="([^"]+)"', _m.group(0), _re.IGNORECASE)
                    _type = _type_m.group(1) if _type_m else "text"
                    if _type.lower() not in ("hidden", "submit", "checkbox", "radio"):
                        _input_items.append(f"input | {_attr[:50]} | type={_type}")
                if len(_input_items) >= 10:
                    break

            # ── SPA 전용: href만 있고 텍스트가 자식 태그에 있는 링크 수집 ────────
            # 크몽/네이버 등 React 앱: <a href="/gig/숫자">...<span>상품명</span>...</a>
            _spa_items = []
            for _am in _re.finditer(r'<a\s[^>]*href="(/gig/\d+|/expert/[^"]+)"[^>]*>(.*?)</a>',
                                     raw_html[:150000], _re.IGNORECASE | _re.DOTALL):
                _href = _am.group(1)
                # 자식 태그에서 텍스트 추출
                _inner = _re.sub(r'<[^>]+>', '', _am.group(2)).strip()
                _inner = ' '.join(_inner.split())[:50]
                if _inner and len(_inner) > 1:
                    _spa_items.append(f"a | {_inner} | {_href}")
                elif _href:
                    _spa_items.append(f"a | [상품] | {_href}")
                if len(_spa_items) >= 30:
                    break

            tags = _re.findall(
                r'<(a|button)[^>]*?(?:href="([^"]*)")?[^>]*>([^<]{1,60})',
                raw_html, _re.IGNORECASE
            )

            # 7-2 수정: 광고/외부 링크 필터
            _AD_URL_PATTERNS = [
                "doubleclick", "googlesyndication", "pagead", "adservice",
                "ad.naver", "adclick", "bnr.", "ads.", "banner", "track.",
                "click.linkprice", "affiliate",
            ]
            # 허용 도메인 추출 (active_schema.where 기반)
            _allowed_dm = ""
            try:
                _sw = (self._active_schema.where or "") if self._active_schema else ""
                _dm_m = _re.search(r'([\w-]+\.[\w.]+)', _sw)
                if _dm_m:
                    _allowed_dm = _dm_m.group(1).lower()
            except Exception:
                pass

            items = []
            for tag, href, text in tags:
                text = text.strip()
                if not text or len(text) < 2:
                    continue
                if href:
                    _hl = href.lower()
                    # 광고 URL 패턴 제외
                    if any(p in _hl for p in _AD_URL_PATTERNS):
                        continue
                    # 허용 도메인 밖 절대 URL 제외
                    if _allowed_dm and href.startswith("http") and _allowed_dm not in _hl:
                        continue
                entry = f"{tag} | {text[:40]}"
                if href:
                    entry += f" | {href[:60]}"
                items.append(entry)
                if len(items) >= 80:
                    break
            # JS추출 → input → SPA gig → 일반 a/button 순으로 병합 (중복 제거)
            _all_hrefs_seen = set()
            _deduped = []
            for _item in _js_links + _input_items + _spa_items + items:
                _parts = _item.rsplit(" | ", 1)
                _key = _parts[-1] if len(_parts) > 1 else _item
                if _key not in _all_hrefs_seen:
                    _all_hrefs_seen.add(_key)
                    _deduped.append(_item)
            result["clickable"] = "\n".join(_deduped)
        except Exception as e:
            print(f"  [Observe] 실패 (무시): {e}")

        # ── gig 링크 0개이면 React 렌더링 대기 후 재추출 (최대 2회) ─────────
        # 단, /gig/ 링크가 실제로 있을 법한 "목록형" 페이지에서만 재시도한다.
        # 메인홈(kmong.com/), /seller/*, /gig/{id}, 로그인/마이페이지 등은
        # 구조상 /gig/ 링크가 없거나 있어도 의미 없으므로 스킵 — 매 스텝 5초
        # 지연·로그 오염 방지.
        _should_retry_gigs = False
        if not _js_links and current_url and 'kmong.com' in current_url:
            try:
                _path = current_url.split('kmong.com', 1)[-1].split('?', 1)[0].split('#', 1)[0]
            except Exception:
                _path = ''
            # 목록형 경로 화이트리스트: 카테고리/검색/전문가 프로필
            _LISTING_PREFIXES = ('/category', '/categories', '/search', '/@')
            if any(_path.startswith(p) for p in _LISTING_PREFIXES):
                _should_retry_gigs = True
        if _should_retry_gigs:
            for _retry in range(2):
                import asyncio as _aio2
                await _aio2.sleep(2.5)
                try:
                    raw_html2 = await _gc(raw=True)
                    if len(raw_html2) < 1000:
                        break
                    # href=/gig/ 재파싱
                    _retry_links = []
                    import re as _re2, json as _json2
                    # __NEXT_DATA__ 재파싱
                    _nd_m2 = _re2.search(
                        r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>',
                        raw_html2[:200000], _re2.DOTALL
                    )
                    if _nd_m2:
                        try:
                            _nd2 = _json2.loads(_nd_m2.group(1))
                            def _eg2(obj, res, d=0):
                                if d > 10 or len(res) >= 20: return
                                if isinstance(obj, dict):
                                    gid = obj.get('gig_id') or obj.get('id') or obj.get('gigId')
                                    t = obj.get('title') or obj.get('name') or ''
                                    if gid and str(gid).isdigit() and t:
                                        res.append(f"a | {str(t)[:40]} | https://kmong.com/gig/{gid}")
                                    for v in obj.values(): _eg2(v, res, d+1)
                                elif isinstance(obj, list):
                                    for i in obj: _eg2(i, res, d+1)
                            _eg2(_nd2, _retry_links)
                        except Exception: pass
                    # href 폴백 (확장 패턴)
                    if not _retry_links:
                        _seen2b = set()
                        for _pat2 in [
                            r'/gig/(\d{4,7})',
                            r'gigId["\s]*:["\s]*(\d{4,7})',
                            r'gig_id["\s]*:["\s]*(\d{4,7})',
                        ]:
                            for _gm2 in _re2.finditer(_pat2, raw_html2):
                                _gid2 = _gm2.group(1)
                                if _gid2 not in _seen2b and len(_gid2) >= 4:
                                    _seen2b.add(_gid2)
                                    _retry_links.append(
                                        f"a | [상품{_gid2}] | https://kmong.com/gig/{_gid2}"
                                    )
                                if len(_retry_links) >= 20: break
                            if len(_retry_links) >= 20: break
                    if _retry_links:
                        _js_links = _retry_links
                        result["html_len"] = len(raw_html2)
                        print(f"  ✅ [Observe] 재추출 성공 ({_retry+1}차): gig {len(_js_links)}개")
                        # clickable 재계산
                        _deduped2 = []
                        _seen2 = set()
                        for _item in _js_links + _input_items + _spa_items + items:
                            _k = _item.rsplit(" | ", 1)[-1]
                            if _k not in _seen2:
                                _seen2.add(_k)
                                _deduped2.append(_item)
                        result["clickable"] = "\n".join(_deduped2)
                        break
                    else:
                        print(f"  ⚠️ [Observe] 재추출 {_retry+1}차: gig 0개 — 대기 중...")
                except Exception as _re_e:
                    print(f"  [Observe] 재추출 실패: {_re_e}")
                    break

        # ── [Hydration Recheck] unhydrated 의심 시 1.5s 대기 후 raw HTML 재측정 ──
        # [Fix 2026-04-22] 헬퍼 메서드로 통합. 기존엔 이 블록이 legacy regex 폴백 경로에만
        # 있어서 A11Y 인덱서 경로(4234 early return)는 hydration recheck 를 완전히
        # 스킵했음. 로그인 루프 버그의 핵심 원인 중 하나.
        await self._try_hydration_recheck(
            result,
            len(result.get("elements") or []) if isinstance(result.get("elements"), list) else 0,
        )

        # ── [Bot-Block Heuristic] 알려진 안티봇 사이트의 빈 쉘 감지 ──
        # 쿠팡/아마존/인스타 등은 QtWebEngine을 봇으로 감지하면 ~300자 빈 HTML만
        # 내려준다. 이 경우 auth_required 경로로는 빠지지 않으므로(공개 경로라서)
        # 전용 플래그로 _react_loop_impl 이 즉시 중단·대안 제시 하도록 신호.
        try:
            _bb_url = result.get("url", "") or ""
            _bb_hl  = int(result.get("html_len", 0) or 0)
            if _is_bot_blocked_shell(_bb_hl, _bb_url):
                result["bot_blocked"] = True
                result["bot_block_reason"] = (
                    f"shell_only(html={_bb_hl}, host={_extract_host(_bb_url)})"
                )
                print(
                    f"  🤖 [Observe] bot_blocked 감지: "
                    f"host={_extract_host(_bb_url)} html={_bb_hl}"
                )
        except Exception:
            pass

        self._annotate_auth_state(result, None)
        self._annotate_captcha_state(result, raw_html)
        return result

    # ── [Stage 2 P1 2026-04-22] 자동 복구: 홈 롤백 + anchor 추출 + LLM 라우팅 ──
    async def _recover_via_link_discovery(
        self,
        broken_url: str,
        schema,
        max_anchors: int = 30,
    ) -> Optional[str]:
        """깨진 URL 발생 시 홈으로 롤백 → 보이는 anchor 추출 → LLM 라우팅 →
        의도에 가장 가까운 링크의 href 반환.

        성공 시: 발견한 URL 문자열 반환. 호출측이 NAVIGATE 실행.
        실패 시: None 반환 → 호출측은 정상 ReAct 흐름 진행.

        설계 원칙:
        - task당 1회만 호출(호출측에서 __recovery_attempted__ 가드).
        - 자체 NAVIGATE는 홈으로만 — 추가 dead-end 생성 방지.
        - 실패해도 조용히 None 반환 — ReAct 정상 흐름 깨지지 않음.
        """
        from urllib.parse import urlparse
        if not broken_url:
            return None

        # 1) 호스트에서 root URL 추출
        try:
            _u = urlparse(broken_url)
            if not _u.scheme or not _u.netloc:
                return None
            root_url = f"{_u.scheme}://{_u.netloc}"
            _path_parts = [p for p in (_u.path or "").split("/") if p]
            _path_tail = " ".join(_path_parts[-2:]) if _path_parts else ""
        except Exception:
            return None

        # 2) intent 합성 (schema.what + 깨진 URL의 path tail)
        _what = (getattr(schema, "what", "") or "") if schema else ""
        _orig = (getattr(schema, "original_prompt", "") or "") if schema else ""
        _intent = f"{_what} {_path_tail}".strip()
        if not _intent:
            _intent = _orig[:120]
        if not _intent:
            _intent = _path_tail or broken_url

        self._report(
            f"🔍 **[자동 복구 시도]** 홈에서 링크 발견\n"
            f"└─ 깨진 URL: {broken_url[:60]}\n"
            f"└─ 의도: {_intent[:80]}\n"
            f"└─ 홈 NAVIGATE → anchor 추출 → LLM 라우팅"
        )

        # 3) 홈으로 이동
        try:
            _nav_res = await _navigate_and_wait(url=root_url, timeout=20.0)
            _nav_res_s = str(_nav_res or "")
            if _nav_res_s.startswith("⚠️ 페이지 로드 실패"):
                print(f"  ❌ [Recovery] 홈 NAVIGATE 실패: {_nav_res_s[:120]}")
                return None
            await asyncio.sleep(0.8)  # 메뉴 렌더 대기
        except Exception as _e_nav:
            print(f"  ❌ [Recovery] 홈 NAVIGATE 예외: {_e_nav}")
            return None

        # 4) 보이는 anchor 추출 (텍스트 + href)
        _js = """
        (function() {
            try {
                var anchors = document.querySelectorAll('a[href]');
                var out = [];
                for (var i = 0; i < anchors.length && out.length < 80; i++) {
                    var a = anchors[i];
                    var rect = a.getBoundingClientRect();
                    if (rect.width === 0 || rect.height === 0) continue;
                    var txt = (a.innerText || a.textContent || a.title ||
                               a.getAttribute('aria-label') || '').trim();
                    var href = a.href || '';
                    if (!txt || txt.length > 60) continue;
                    if (!href || href.indexOf('javascript:') === 0) continue;
                    if (href.indexOf('#') === 0) continue;
                    out.push({text: txt, href: href});
                }
                return JSON.stringify(out);
            } catch (e) { return '[]'; }
        })()
        """
        anchors = []
        try:
            from execution_module import browser_run_script as _run_js
            _raw = await _run_js(script=_js)
            if isinstance(_raw, str) and _raw.strip().startswith("["):
                anchors = json.loads(_raw)
        except Exception as _e_aa:
            print(f"  ❌ [Recovery] anchor 추출 실패: {_e_aa}")
            return None

        if not anchors:
            print(f"  ❌ [Recovery] anchor 0개 — 홈에서 링크를 못 찾음")
            return None

        # 5) LLM에게 best anchor 선택 의뢰
        anchors = anchors[:max_anchors]
        _anchor_text = "\n".join(
            f"{i+1}. \"{a.get('text','')[:40]}\" → {a.get('href','')[:80]}"
            for i, a in enumerate(anchors)
        )
        _prompt = (
            f"사용자 의도: \"{_intent[:150]}\"\n\n"
            f"다음 링크 중 사용자 의도에 가장 가까운 1개를 골라 번호만 답하라.\n"
            f"적합한 링크가 없으면 0.\n\n"
            f"[링크 목록 ({len(anchors)}개)]\n{_anchor_text}\n\n"
            f"답: 번호 1개만 (예: 5). 0 ~ {len(anchors)} 중 하나."
        )
        # [Fix 2026-04-22] 휴리스틱 폴백 — LLM 응답이 에러/부적합일 때 사용
        # Why: 2026-04-22 크몽 로그에서 LLM 502 로 Recovery 가 None 반환 →
        # 자동 복구 실패 리포트 → ReAct 가 같은 URL 재시도 루프.
        # 502 뿐 아니라 400/500/timeout 등 외부 LLM 일시 장애 전반 보호.
        def _heuristic_pick_anchor():
            _intent_lc = (_intent or "").lower()
            # intent 키워드 + 도메인 공통 auth/action 키워드
            _kw: List[str] = []
            for _tok in re.findall(r'[a-zA-Z가-힣]{2,}', _intent_lc):
                _kw.append(_tok)
            # 작업성 키워드 보강 (로그인/판매자/등록/서비스/상품/sell/login 등)
            _action_hints = [
                "로그인", "login", "sign in", "signin", "sign-in",
                "판매자", "셀러", "seller", "expert",
                "등록", "register", "신규", "create", "new",
                "서비스", "service", "상품", "product", "gig",
                "my", "마이", "계정", "account",
            ]
            _kw_set = set(_kw) | set(_action_hints)
            _best = None
            _best_score = 0
            for a in anchors:
                _t = (a.get("text", "") or "").lower()
                _h = (a.get("href", "") or "").lower()
                _blob = _t + " " + _h
                _score = sum(1 for k in _kw_set if k and k in _blob)
                if _score > _best_score:
                    _best_score = _score
                    _best = a
            return _best, _best_score

        try:
            # [2026-05-06 fix] max_tokens 10 → 200. Gemini 2.5 Flash thinking 토큰
            # 한도 초과 회피. 짧은 분류 응답이라도 thinking 여유 필요.
            _raw_llm = await get_llm_response_async(
                _prompt, max_tokens=200, timeout=20
            )
            _raw_s = str(_raw_llm or "")
            # LLM 외부 장애 감지 — prefix 기반
            _ERR_PREFIX = (
                "[서버 혼잡", "[rate", "[timeout", "[error",
                "[오류", "[실패", "[llm 오류", "502", "503", "504",
            )
            _is_err = any(_raw_s.strip().lower().startswith(p.lower()) or p.lower() in _raw_s.lower()[:40]
                          for p in _ERR_PREFIX)
            _picked = -1
            if not _is_err:
                _m = re.search(r'\d+', _raw_s)
                if _m:
                    _picked = int(_m.group(0))
            if _picked <= 0 or _picked > len(anchors):
                # 휴리스틱 폴백
                _hb, _hscore = _heuristic_pick_anchor()
                if _hb and _hscore >= 1:
                    _href_h = _hb.get("href", "") or ""
                    print(
                        f"  🩹 [Recovery] LLM 실패({'err' if _is_err else 'bad'}) "
                        f"→ 휴리스틱 선택 score={_hscore}: "
                        f"\"{_hb.get('text','')[:30]}\" → {_href_h[:80]}"
                    )
                    return _href_h if _href_h else None
                print(
                    f"  ❌ [Recovery] LLM 부적합 응답 + 휴리스틱 무매칭: "
                    f"{_raw_s[:60]}"
                )
                return None
            chosen = anchors[_picked - 1]
            _href = chosen.get("href", "")
            print(
                f"  ✅ [Recovery] LLM 선택 #{_picked}: "
                f"\"{chosen.get('text','')[:30]}\" → {_href[:80]}"
            )
            return _href if _href else None
        except Exception as _e_llm:
            print(f"  ❌ [Recovery] LLM 호출 실패: {_e_llm}")
            # 예외 시에도 휴리스틱 폴백 시도
            try:
                _hb, _hscore = _heuristic_pick_anchor()
                if _hb and _hscore >= 1:
                    _href_h = _hb.get("href", "") or ""
                    print(
                        f"  🩹 [Recovery] 예외 후 휴리스틱 선택 score={_hscore}: "
                        f"\"{_hb.get('text','')[:30]}\" → {_href_h[:80]}"
                    )
                    return _href_h if _href_h else None
            except Exception as _e_h:
                print(f"  [Recovery] 휴리스틱 폴백 실패 (무시): {_e_h}")
            return None

    async def _try_hydration_recheck(self, result: Dict, element_count: int) -> None:
        """unhydrated 의심 시 1.5s 대기 후 raw HTML 재측정.

        _is_unhydrated_spa 1차 판정(html<=150KB, elems<=5, path=auth-gated) 통과하면
        SPA 가 아직 렌더 중일 가능성. 한 번 더 대기 후 raw HTML 로 `<a>/<button>/<input>/
        role=button` 개수를 세서 기준 초과면 `__hydrated_hint__` 를 세팅해 후속
        `_annotate_auth_state` 에서 unhydrated 판정을 철회시킨다.

        인덱서 경로 / legacy regex 폴백 경로 양쪽에서 호출. 이전엔 legacy 에만 있어서
        인덱서 early return 시 recheck 가 완전히 스킵돼 로그인 루프의 원인이 되었음
        (2026-04-22 kmong /expert/service/register 로그 참조).
        """
        try:
            from execution_module import get_browser_content as _gc_hyd
            _url_chk = result.get("url", "") or ""
            _hl_chk  = int(result.get("html_len", 0) or 0)
            _thr_hyd = (
                _UNHYDRATED_ELEMENT_THRESHOLD_EDIT
                if _is_edit_path(_url_chk)
                else _UNHYDRATED_ELEMENT_THRESHOLD
            )
            if not (
                _hl_chk > 0 and _hl_chk <= _UNHYDRATED_HTML_THRESHOLD
                and element_count <= _thr_hyd
                and (_is_auth_gated_path(_url_chk) or _is_edit_path(_url_chk))
                and not _is_auth_redirect_url(_url_chk)
            ):
                return
            import asyncio as _aio_hyd
            print(
                f"  ⏳ [Observe] unhydrated 의심(html={_hl_chk} elems={element_count}/thr={_thr_hyd}) "
                f"— 1.5s SPA hydration 대기 후 재측정"
            )
            await _aio_hyd.sleep(1.5)
            try:
                _rh_hyd = await _gc_hyd(raw=True)
                if _rh_hyd and len(_rh_hyd) > _hl_chk:
                    result["html_len"] = len(_rh_hyd)
                import re as _re_hyd
                _raw_cnt = 0
                for _pat_hyd in (r'<a[\s>]', r'<button', r'<input', r'role=["\']button["\']'):
                    _raw_cnt += len(_re_hyd.findall(_pat_hyd, (_rh_hyd or "")[:400000]))
                if _raw_cnt > _thr_hyd:
                    print(
                        f"  ✅ [Observe] hydration 진행 감지 "
                        f"(html {_hl_chk}→{len(_rh_hyd or '')}, raw_elems~{_raw_cnt}) "
                        f"— unhydrated 판정 철회"
                    )
                    result["__hydrated_hint__"] = _raw_cnt
            except Exception as _e_hyd:
                print(f"  [Observe] hydration 재측정 실패 (무시): {_e_hyd}")
        except Exception:
            pass

    def _annotate_auth_state(self, result: Dict, element_count: Optional[int]) -> None:
        """_observe 결과에 auth_required / unhydrated 플래그 주입 (Layer 1)."""
        url = result.get("url", "") or ""
        html_len = int(result.get("html_len", 0) or 0)
        if element_count is None:
            _els = result.get("elements")
            element_count = len(_els) if isinstance(_els, list) else 0

        auth_by_url = _is_auth_redirect_url(url)
        # hydration 재측정에서 실제 DOM 요소 충분(raw HTML 기반) 확인되면 unhydrated 무효.
        _hyd_hint = result.pop("__hydrated_hint__", None)
        if _hyd_hint:
            unhydrated = False
        else:
            unhydrated = _is_unhydrated_spa(html_len, element_count, url)
        result["unhydrated"] = unhydrated
        result["auth_required"] = auth_by_url or unhydrated
        if result["auth_required"]:
            _why = "login_redirect_url" if auth_by_url else "unhydrated_spa"
            print(
                f"  🔐 [Observe] auth_required=True ({_why}) | "
                f"url={url[:80]} html={html_len} elems={element_count}"
            )

    def _annotate_captcha_state(self, result: Dict, raw_html: str = "") -> None:
        """_observe 결과에 captcha_required / captcha_signal 플래그 주입.

        [2026-04-24] CAPTCHA Phase B — DOM 셀렉터 기반 감지.
        Why: 기존 ASK_USER question 키워드 매칭만으로는 페이지 본문에 캡챠가 떠 있어도
        ReAct 루프가 stuck 후 ASK_USER 를 던져야만 알 수 있었음. _observe() 단계에서
        선제 감지하면 CaptchaRecheck (resume 검증) 와 ReAct 루프 모두 같은 신호 사용.
        """
        result["captcha_required"] = False
        result["captcha_signal"] = ""
        try:
            has_cap, sig = _detect_captcha_in_html(raw_html or "")
            if has_cap:
                result["captcha_required"] = True
                result["captcha_signal"] = sig
                _url = (result.get("url") or "")[:80]
                print(f"  🧩 [Observe] captcha_required=True ({sig}) | url={_url}")
        except Exception as _ce:
            # 감지 실패는 silent — 기존 본문 키워드/ASK_USER 경로가 보조 신호로 잡음.
            pass

    # ── Think ──────────────────────────────────────────────────────────────
    def _compact_anchor(self, anchor: str) -> str:
        """
        ReAct Think 프롬프트 주입용 anchor 압축.
        WHY 블록이 길면(3문항) 첫 줄만 남기고 나머지 제거.
        프롬프트 길이 폭증 방지 → LLM 번호 파싱 실패 예방.
        """
        _ANCHOR_FIELDS = ("WHO", "WHAT", "WHERE", "WHEN", "HOW", "[드리프트", "[MISSION")
        lines = anchor.splitlines()
        result = []
        in_why = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("WHY"):
                # WHY 줄: 첫 줄만 포함 (최대 80자)
                why_val = line[line.find(":")+1:].strip()
                first_why = why_val.split("\n")[0][:80] if why_val else ""
                result.append(f"WHY   : {first_why}")
                in_why = True
            elif in_why and any(stripped.startswith(f) for f in _ANCHOR_FIELDS):
                # 다음 anchor 필드 시작 → WHY 블록 종료
                in_why = False
                result.append(line)
            elif in_why:
                # WHY 블록 계속 (부재의 대가, 성공 정의 등 추가 줄 건너뜀)
                continue
            else:
                result.append(line)
        return "\n".join(result)

    async def _think(
        self,
        task_prompt: str,
        anchor: str,
        obs: Dict,
        history: list,
        schema,
        step_no: int,
        drift_correction: Optional[str] = None,   # [DriftGuard] 교정 정책 C
        hybrid_state: Optional[Dict] = None,       # [H 2026-04-26] 양방향 신호
    ) -> Optional[Dict]:
        """
        앵커 + 관찰 + 히스토리 → LLM이 다음 액션 결정.
        드리프트 자가 점검 포함.
        """
        # [진단 로그 2026-05-04] _think 진입 시점 schema 상태 가시화 — 사용자 frustration
        # "왜 스키마 GUI 에 안 떠?" 추적용. 매 step 1회 출력 (step_no=1 시).
        if step_no <= 1:
            try:
                _sch_diag = "None" if schema is None else (
                    f"chain_id={(getattr(schema, 'chain_id', '') or '')[:8] or 'AD-HOC'} "
                    f"stage={getattr(schema, 'stage', '') or '-'} "
                    f"task_type={getattr(schema, 'task_type', '') or '-'} "
                    f"attrs={len(getattr(schema, 'attributes', []) or [])}"
                )
                print(
                    f"  🔍 [_think 진입] step={step_no} task='{(task_prompt or '')[:50]}' "
                    f"schema=[{_sch_diag}]"
                )
            except Exception as _e_diag:
                print(f"  🔍 [_think 진입] 진단 로그 실패 (무시): {_e_diag}")

        # [근본 fix 2026-05-04] ad-hoc schema (chain_id 없음) 으로 _think 진입 시
        # research 의도면 강제로 research chain 으로 promote. 이전 세션 잔존 schema /
        # GUI 라우팅 우회 / 외부 helper schema 등 어느 경로로 들어와도 catch.
        # 1회만 시도 (schema._chain_promote_tried 마킹), 실패 시 그대로 진행.
        try:
            if (schema is not None
                    and not (getattr(schema, "chain_id", "") or "").strip()
                    and not getattr(schema, "_chain_promote_tried", False)
                    and step_no <= 1):
                schema._chain_promote_tried = True
                _promote_prompt = (
                    getattr(schema, "original_prompt", "")
                    or getattr(schema, "what", "")
                    or task_prompt
                    or ""
                ).strip()
                if _promote_prompt and await self._is_research_intent_async(_promote_prompt):
                    print(
                        f"  🔗 [Schema Promote] ad-hoc schema → research chain 강제 변환 "
                        f"(prompt='{_promote_prompt[:40]}')"
                    )
                    self._report(
                        f"🔗 **[Schema 자동 승격]** ad-hoc → research chain — "
                        f"chain_id 없는 schema 가 research 의도라 chain 으로 변환. "
                        f"BB consolidation + Phase F 자동 피벗 활성."
                    )
                    _ok_promote = await self._create_research_chain_inline(_promote_prompt)
                    if _ok_promote and self._active_schema is not None:
                        # 새 chain schema 로 교체된 상태 — 호출자에게 schema 변경 통지 위해
                        # 기존 schema 의 chain_id 도 함께 set (caller 가 같은 ref 사용 시).
                        try:
                            schema.chain_id = self._active_schema.chain_id
                        except Exception:
                            pass
        except Exception as _e_pr:
            print(f"  ⚠️ [Schema Promote] 실패 (무시, 기존 schema 진행): {_e_pr}")
        _js_links_available = bool(obs.get("clickable", "").strip())
        history_str = ""
        if history:
            lines = []
            # 12-1: 중요 step(성공한 첫 step들) + 최근 6개 유지
            # 초반 로그인/에디터 진입 같은 중요 성공 step이 잘리지 않도록
            _important = [h for h in history if h.get("verified")][:3]  # 초반 성공 3개
            _recent    = history[-6:]                                     # 최근 6개
            _combined  = {h["step"]: h for h in _important + _recent}    # 중복 제거
            for h in sorted(_combined.values(), key=lambda x: x["step"]):
                mark = "✅" if h.get("verified") else "❌"
                lines.append(
                    f"  {mark} Step {h['step']}: [{h['action']}] "
                    f"{h['target'][:40]} → {h['result'][:40]}"
                )
            history_str = "\n".join(lines)

        # HOW_MUCH 미달성 속성 목록
        pending_attrs = ""
        # 12-2: 달성된 속성의 evidence도 프롬프트에 포함 (LLM이 이미 완료된 방식 참조)
        achieved_attrs = ""
        if schema and schema.attributes:
            pending = [a for a in schema.attributes if not a.achieved]
            if pending:
                pending_attrs = "\n".join(
                    f"  - {a.name} {a.operator} {a.value}"
                    for a in pending
                )
            achieved = [a for a in schema.attributes if a.achieved]
            if achieved:
                achieved_attrs = "\n".join(
                    f"  ✅ {a.name} (근거: {a.evidence[:50] if a.evidence else '자동감지'})"
                    for a in achieved
                )

        # [Fix] ReAct 변수 현황을 프롬프트에 노출 — 중복 READ_PAGE/WRITE 방지.
        # LLM이 이미 확보한 변수를 몰라 같은 페이지를 다른 변수명으로 재저장하던 루프 차단.
        stored_vars_block = ""
        if schema is not None:
            _stored = getattr(schema, "_react_vars", None) or {}
            if _stored:
                _lines = []
                for _k, _v in list(_stored.items())[-8:]:
                    _sz = len(str(_v)) if _v else 0
                    _kind = "WRITE" if not (_k.startswith("page_") or _k.endswith("_page") or _k.endswith("_content") or "page_content" in _k.lower()) else "READ"
                    _lines.append(f"  - {_k} ({_kind}, {_sz}자)")
                stored_vars_block = (
                    "\n[✅ 이미 확보한 변수 — 동일 내용 재READ/재WRITE 금지]\n"
                    + "\n".join(_lines)
                    + "\n→ 요약·답변에 필요한 페이지 내용이 이미 있으면 **바로 WRITE(기존 변수 참조) 또는 DONE** 하라. "
                    + "같은 페이지를 다른 변수명으로 READ_PAGE 재실행 금지.\n"
                )

        # ── [Phase 5-B α] Self-Awareness 블록 — wire13 gated ───────────
        # SelfModel 이 감지한 최근 weakness / bias 를 LLM 에 경고.
        # "반복 패턴 주의" 자기성찰 주입 — 강요 아닌 힌트.
        self_awareness_block = ""
        try:
            from eidos_tuning import get_tuned_param as _smtp
            if _smtp("wire13_self_model_enabled", False):
                from eidos_self_model import SelfModel as _SM
                _sm = _SM.get()
                _w = _sm.weakness_detected()
                _b = _sm.get_bias()
                _sa_lines = []
                if _w:
                    _sa_lines.append(
                        f"  · 약점: {_w.get('action_type')} — "
                        f"최근 {int(_w.get('fail', 0))}회 실패, "
                        f"pass_rate {float(_w.get('pass_rate', 0.0)):.0%}"
                    )
                if _b:
                    _sa_lines.append(
                        f"  · 편향: {_b.get('action_type')} — "
                        f"사용 비율 {float(_b.get('usage_ratio', 0.0)):.0%} "
                        f"({_b.get('count')}/{_b.get('total_obs')} 관측)"
                    )
                if _sa_lines:
                    self_awareness_block = (
                        "\n[🪞 자기 인식 — 최근 행동 패턴]\n"
                        + "\n".join(_sa_lines)
                        + "\n→ 강요 아님. 맞다고 판단되면 평소 패턴 깨도 좋다.\n"
                    )
        except Exception:
            pass

        # ── [Phase 4-B] Episodic Memory 힌트 블록 — wire12 gated ───────
        # signature_of(task_type+where+who+url) 기반 과거 유사 trajectory 조회.
        # Skill 힌트(wire10) 가 semantic(빈도 패턴) 이라면 이건 episodic(이 sig
        # 에서 실제 결과). 둘은 역할 분리되어 동시 on 가능.
        memory_episodic_block = ""
        try:
            from eidos_tuning import get_tuned_param as _memtp
            if _memtp("wire12_memory_system_enabled", False):
                from eidos_memory_system import MemorySystem as _MS
                from eidos_policy_store import signature_of as _sig_of
                _mem_state: Dict[str, Any] = {}
                if schema is not None:
                    _tt  = getattr(schema, "task_type", "") or ""
                    _wh  = getattr(schema, "where", "") or ""
                    _who = getattr(schema, "who", "") or ""
                    if _tt:  _mem_state["task_type"] = str(_tt)
                    if _wh:  _mem_state["where"]     = str(_wh)
                    if _who: _mem_state["who"]       = str(_who)
                _url = obs.get("url", "") if obs else ""
                if _url:
                    _mem_state["url"] = str(_url)
                if _mem_state:
                    _sig_mem = _sig_of(_mem_state)
                    _eps = _MS.get().retrieve_episodic(_sig_mem, k=2)
                    if _eps:
                        _lines = []
                        for _ep in _eps[:2]:
                            _acts = "→".join(_ep.get("matched_actions", [])[:4]) or "?"
                            _lines.append(
                                f"  · {_acts} "
                                f"(평균 reward={_ep.get('avg_reward_at_sig', 0):.2f}, "
                                f"전체 return={_ep.get('total_return', 0):.2f})"
                            )
                        if _lines:
                            memory_episodic_block = (
                                "\n[🧠 과거 유사 상황 — 에피소드 기억]\n"
                                + "\n".join(_lines)
                                + "\n→ 같은 state 에서 과거 실제 결과. 반복할지 회피할지 "
                                  "판단 근거로 사용. 강요 아님.\n"
                            )
        except Exception:
            pass

        # ── [Phase 3-D] Skill Macro 포맷 블록 — wire11 gated ───────────
        # wire11 on 일 때만 프롬프트에 skill_steps 응답 포맷 설명 주입.
        # LLM 이 연속 action 을 미리 결정하면 다음 step 부터 _think 호출 스킵.
        skill_macro_rule_block = ""
        try:
            from eidos_tuning import get_tuned_param as _stp2
            if _stp2("wire11_skill_macro_enabled", False):
                skill_macro_rule_block = (
                    "\n[⚡ Skill Macro 옵션 — 선택]\n"
                    "연속 action 을 한 번에 결정할 수 있다. 확신 있을 때만 사용.\n"
                    "응답 JSON 에 선택적으로 `\"skill_steps\"` 배열 추가:\n"
                    '  {"action": "NAVIGATE", "target": "URL", "content": "", "reason": "...",\n'
                    '   "skill_id": "NAVIGATE__FILL__CLICK",\n'
                    '   "skill_steps": [\n'
                    '     {"action": "FILL",  "target": "#q",  "content": "검색어"},\n'
                    '     {"action": "CLICK", "target": "[5]", "content": ""}\n'
                    '   ]}\n'
                    "규칙: skill_steps 의 첫 action 은 그냥 action 필드 값과 동일. "
                    "skill_steps 가 있으면 첫 action 실행 후 나머지는 LLM 호출 없이 "
                    "순차 자동 실행된다. 불확실하면 skill_steps 생략(일반 응답).\n"
                )
        except Exception:
            pass

        # ── [Phase 3-C-2] Skill 힌트 블록 — wire10 gated ───────────────
        # 과거 trajectory 에서 추출한 반복 성공 시퀀스를 프롬프트에 참고로 주입.
        # anchor (skill 의 첫 element) 추정 순서:
        #   1) history[-1].action    — 직전 실행된 action_type (가장 확실)
        #   2) schema.verb 매핑      — EDIT/DELETE→CLICK, READ/CREATE→NAVIGATE
        # 힌트 없음 / 토글 off / 실패 모두 silently skip.
        skill_hints_block = ""
        try:
            from eidos_tuning import get_tuned_param as _stp
            if _stp("wire10_skill_suggestion_enabled", False):
                from eidos_skill_store import SkillStore as _SS
                _next_anchor = ""
                if history:
                    _last = history[-1].get("action", "") or ""
                    if _last:
                        _next_anchor = str(_last)
                if not _next_anchor and schema is not None:
                    _verb = getattr(schema, "verb", "") or ""
                    _next_anchor = {
                        "EDIT":   "CLICK",
                        "DELETE": "CLICK",
                        "READ":   "NAVIGATE",
                        "CREATE": "NAVIGATE",
                    }.get(_verb, "")
                if _next_anchor:
                    _k = int(_stp("wire10_skill_suggestion_k", 3))
                    _hint = _SS.get().get_hints_text(_next_anchor, k=_k)
                    if _hint:
                        skill_hints_block = (
                            "\n[💡 Skill 힌트 — 과거 반복 성공 패턴 참고]\n"
                            f"{_hint}\n"
                            "→ 이 시퀀스가 현재 상황에 유효하면 그대로 따라가는 것도 좋은 선택이다. "
                            "강요 아님 — 판단은 네가 한다.\n"
                        )
        except Exception:
            pass

        # general/write/report 태스크는 브라우저 컨텍스트 불필요
        _NON_BROWSER_TYPES = ("general", "write", "report")
        _cur_task_type = getattr(schema, "task_type", "general") if schema else "general"
        _is_non_browser = _cur_task_type in _NON_BROWSER_TYPES

        # 인덱서 성공 여부 → 프롬프트 안내 분기
        _has_indexed = bool(obs.get("elements"))
        if _has_indexed:
            _elems_header = (
                f"[상호작용 요소 — 인덱싱됨 ({len(obs['elements'])}개)]\n"
                "각 줄: `[번호] kind \"텍스트\" → href`. CLICK target에 `[번호]`만 입력하라.\n"
            )
        else:
            _elems_header = (
                f"[클릭 가능한 요소 (최대 80개, raw HTML regex 폴백)]\n"
                f"HTML 크기: {obs.get('html_len', 0)}자\n"
            )

        # [Fix C 2026-05-05] DOM truncate 20000 → 50000 자.
        # 사용자 보고 "내장브라우저 상세페이지 게으름" — 상세 페이지는 list 보다 DOM 이
        # 풍부해 인덱서 출력이 20000자 cap 에 걸려 LLM 한테 전달 안 됨. gemini-2.5-flash
        # 1M context 라 50000자 (~12k token) 안전. 회귀 시 다시 20000 으로 내릴 것.
        _CLICKABLE_CAP = 50000
        _browser_ctx = (
            "[브라우저 불필요] 이 작업은 정보 수집/작성 태스크입니다. NAVIGATE/CLICK은 사용하지 마세요.\n"
            "WRITE 또는 DONE 액션만 사용하세요."
            if _is_non_browser else
            f"URL: {obs.get('url', '') or '조회 실패 — 브라우저 현재 위치 불명'}\n\n"
            f"{_elems_header}"
            f"{obs.get('clickable', '없음')[:_CLICKABLE_CAP]}"
        )

        # Layer 1 auth blacklist — 이번 세션에서 비로그인으로 접근 실패한 경로 목록.
        # 같은 경로 재NAVIGATE 시 사전 차단되므로 LLM이 대안을 찾도록 명시적으로 노출.
        _auth_bl = obs.get("auth_blacklist") or []
        _auth_bl_block = ""
        # EDIT/CREATE/DELETE 는 로그인된 판매자 계정이 필수 — 공개 페이지로 폴백해봐야
        # 편집 버튼이 없어 목표 달성 불가. 블랙리스트 히트 시 바로 ASK_USER(로그인) 유도.
        _auth_sensitive_verb = getattr(schema, "verb", "") in ("EDIT", "CREATE", "DELETE") if schema else False
        _schema_auth_required = bool(getattr(schema, "auth_required", False)) if schema else False
        if _auth_bl and not _is_non_browser:
            _bl_lines = "\n".join(f"  - {p}" for p in _auth_bl[:10])
            if _auth_sensitive_verb or _schema_auth_required:
                _fallback_line = (
                    "→ 이 작업은 로그인한 판매자 계정이 필수다. 공개 /search·/category 로 폴백하지 말고 "
                    "즉시 ASK_USER 로 사용자에게 브라우저에서 로그인 후 ⚡ 자율 실행을 눌러달라고 요청하라."
                )
            else:
                _fallback_line = (
                    "→ 이 경로들(및 동일 호스트의 /my-*, /seller/*, /dashboard 류) 대신 "
                    "공개 페이지(홈/검색/카테고리/상품 상세) 또는 ASK_USER로 사용자에게 로그인을 요청하라."
                )
            # [Fix 2026-04-22] 도메인별 로그인 URL 프리셋 — ASK_USER 발행 시 포함하도록 명시
            _KNOWN_LOGIN_URLS = {
                "kmong.com": "https://kmong.com/?f=header&open=login_modal",
                "kmong": "https://kmong.com/?f=header&open=login_modal",
                "coupang.com": "https://login.coupang.com/login/login.pang",
                "naver.com": "https://nid.naver.com/nidlogin.login",
                "gumroad.com": "https://app.gumroad.com/login",
                "github.com": "https://github.com/login",
                "instagram.com": "https://www.instagram.com/accounts/login/",
            }
            _login_hint = ""
            try:
                _cur_host = (obs.get("url", "") or "").lower()
                for _dom, _lurl in _KNOWN_LOGIN_URLS.items():
                    if _dom in _cur_host:
                        _login_hint = (
                            f"  · 이 사이트 로그인 URL 후보: {_lurl}\n"
                            f"    (ASK_USER content 에 이 URL 을 포함해 사용자가 바로 클릭할 수 있게 해라)"
                        )
                        break
            except Exception:
                pass
            _auth_bl_block = (
                "\n[🔐 비로그인 차단 경로 — 재시도 금지]\n"
                f"이번 세션에서 아래 경로는 로그인이 필요해 이미 차단됨. "
                f"같은 경로로 NAVIGATE 하면 즉시 사전 차단되어 step이 낭비된다.\n"
                f"{_bl_lines}\n"
                f"{_fallback_line}\n"
            )
            if _login_hint:
                _auth_bl_block += f"{_login_hint}\n"

        # [2026-04-22] FILL 영구 블랙리스트 블록 — (path, target) 재지명 금지
        _fill_bl = obs.get("fill_blacklist") or []
        _fill_bl_block = ""
        if _fill_bl and not _is_non_browser:
            try:
                _cur_path = ""
                _m_fbp = re.match(r'^https?://[^/]+(/[^?#]*)', obs.get("url", "") or "")
                if _m_fbp:
                    _cur_path = _m_fbp.group(1)
                # 현재 경로와 일치하는 엔트리 우선 표시 (가장 관련 높음)
                _cur_hits = [t for (p, t) in _fill_bl if p == _cur_path]
                _other_hits = [(p, t) for (p, t) in _fill_bl if p != _cur_path]
                _lines: list = []
                if _cur_hits:
                    _lines.append(
                        f"  · 현재 페이지('{_cur_path[:50]}')에서 이미 실패한 target: "
                        + ", ".join(f"[{t}]" for t in _cur_hits[:8])
                    )
                for (p, t) in _other_hits[:4]:
                    _lines.append(f"  · {p[:40]} → [{t}]")
                if _lines:
                    _fill_bl_block = (
                        "\n[🚫 FILL 금지 target — 재지명 시 사전 차단]\n"
                        "아래 target은 이미 2회 이상 FILL 실패해 편집 불가로 확정됐다. "
                        "같은 [N] 재FILL 금지. 다른 input/textarea/contenteditable "
                        "요소를 선택하거나, 편집 폼이 없으면 NAVIGATE로 전용 등록 "
                        "페이지로 이동하라.\n"
                        + "\n".join(_lines) + "\n"
                    )
            except Exception:
                pass

        _non_browser_rule = ("8. [브라우저 불필요] NAVIGATE/CLICK 절대 금지. content에 조사 결과를 작성하고 WRITE 또는 DONE만 사용하라." if _is_non_browser else "")

        # [2026-04-24] 영구 금지 URL 블록 — 핑퐁 루프 차단
        # 동일 URL 반복 차단 / 경로 루프 강제 중단 / Detail Gate 가 판정한
        # "재NAVIGATE 시 확정 실패" URL 들을 LLM 프롬프트 상단에 하드블록으로 주입.
        # Why: 기존 schema.how 힌트는 다음 루프에서 덮어쓰이거나 LLM 이 장문 안에서
        # 묻혀 무시함. 전용 블록으로 격상 + 재선택 시 자동 교체 경고까지 명시.
        _forbidden_urls_block = ""
        try:
            _fb = getattr(schema, "_forbidden_urls", None) if schema else None
            if _fb and not _is_non_browser:
                _lines_fb: list = []
                for _u, _why in list(_fb.items())[-8:]:
                    _lines_fb.append(f"  · {str(_u)[:80]} — {str(_why)[:60]}")
                if _lines_fb:
                    _forbidden_urls_block = (
                        "\n[🚫 재NAVIGATE 금지 URL — 핑퐁 확정]\n"
                        "아래 URL 은 이미 루프로 판정됐다. **동일 path 로 NAVIGATE 금지.** "
                        "READ_PAGE 로 이미 확보한 변수가 있으면 그 변수로 WRITE/DONE, "
                        "없으면 **다른 상세 URL(/gig/{id}, /product/{id} 등)** 로 NAVIGATE, "
                        "그것도 안 되면 ASK_USER. 같은 path 재선택은 시스템이 자동 차단한다.\n"
                        + "\n".join(_lines_fb) + "\n"
                    )
        except Exception:
            pass

        # ── [사이트 컨텍스트] kmong-specific 규칙 게이트 ─────────────────────
        _kmong_keywords_th = ["크몽", "kmong", "상품 등록", "상세페이지", "판매자", "서비스 등록", "gig"]
        _is_kmong_task_th = any(kw.lower() in (task_prompt or "").lower() for kw in _kmong_keywords_th)
        _is_kmong_url_th = "kmong.com" in (obs.get("url", "") or "")
        _is_kmong_ctx = _is_kmong_task_th or _is_kmong_url_th

        _kmong_rules = (
            "8. 크몽에서 구매자용 공개 상품 조회가 필요할 때(VERB=READ에 한함) FILL 대신 NAVIGATE로 \"https://kmong.com/search?keyword=검색어\" 형태 URL을 직접 사용하라. ★단 VERB=EDIT/CREATE/DELETE 에서는 /search·/browse·/category 로 NAVIGATE 금지 — 해당 경로에는 수정/등록/삭제 버튼이 없다. 반드시 AUTH 블록의 seller_entry_url(예: /my-gigs) 또는 VERB 규칙이 지정하는 경로로만 이동하라.\n"
            "9. 카테고리 이동도 CLICK 대신 \"https://kmong.com/category/숫자\" 형태 URL로 NAVIGATE하라.\n"
            "10. 상품 상세 페이지(gig/숫자)에 도달했으면 즉시 WRITE로 가격/리뷰수/서비스 내용을 content에 기록하라. 다른 페이지로 이동하기 전에 반드시 WRITE 먼저.\n"
            "11. 여러 상품을 분석할 때는 각 상품 gig URL을 순서대로 NAVIGATE → WRITE 패턴을 반복하라.\n"
        ) if _is_kmong_ctx else ""

        # ── [TARGET 도메인 불일치] 사용자가 지정한 대상 사이트에 아직 도달 못한 상태 ─
        # 예: "나무위키에서 X 요약해줘"인데 현재 URL이 naver.com이면, FILL/CLICK 하기 전에
        # 반드시 NAVIGATE로 namu.wiki로 이동하도록 최우선 힌트를 강제 주입.
        # 기존 rule 6은 "URL이 조회 실패이거나 비어있으면"만 다뤘으나, 시작 페이지가
        # 다른 사이트일 때는 URL이 멀쩡해서 이 조건을 못 잡았음 (네이버 검색창 FILL 루프 원인).
        _target_nav_hint = ""
        if not _is_non_browser:
            _cur_url_l = (obs.get("url", "") or "").lower()
            _target_domain = self._extract_target_domain(task_prompt, schema)
            # β (multi-domain WHERE): schema.where 가 쉼표로 여러 도메인 박혔으면
            # 그 모두를 허용 도메인으로 보고 OR 매칭. 현재 URL 이 그 중 하나라도
            # 매칭되면 hint 발동 안 함 — multi-source task (환율/시장조사/가격비교)
            # 에서 검색엔진→외부사이트 자연 흐름 차단되던 무한 루프 해소.
            _allowed_set = set()
            if _target_domain:
                _allowed_set.add(_target_domain.lower())
            if schema and getattr(schema, "where", None):
                import re as _re_md
                for _d in _re_md.findall(r'([\w-]+\.[\w.]+)', schema.where or ""):
                    _allowed_set.add(_d.lower())
            _hit_any = bool(_cur_url_l) and any(d in _cur_url_l for d in _allowed_set)
            if _target_domain and _cur_url_l and not _hit_any:
                _allowed_str = ", ".join(sorted(_allowed_set)) if len(_allowed_set) > 1 else _target_domain
                _target_nav_hint = (
                    f"0-NAV. 🎯 [TARGET 불일치 — 최우선] 사용자가 지정한 대상 도메인은 "
                    f"'{_allowed_str}' 이지만 현재 URL은 '{_cur_url_l[:60]}' 이다. "
                    f"다음 액션은 반드시 `NAVIGATE` 이어야 하며 target 에는 '{_target_domain}' "
                    f"경로를 직접 입력하라. "
                    f"현재 페이지의 검색창에 '{_target_domain}' 을 FILL 하거나 현재 페이지에서 "
                    f"CLICK 하는 것은 우회 루프이며 실패한다. 직접 URL 네비게이션이 기본 동작이다.\n"
                )

        # ── [VERB 규칙] schema.verb가 있으면 의도 고정 규칙 추가 ─────────────
        _verb = getattr(schema, "verb", "") if schema else ""
        _verb_rule = ""
        if _verb == "EDIT":
            _verb_rule = (
                "0. 🎯 [VERB=EDIT — 최우선] 기존 리소스를 **수정**하는 작업이다. "
                "/register · /new · /create · /signup · 서비스 등록 경로 절대 금지. "
                "먼저 기존 리소스 목록(예: /my-gigs, /dashboard, '내 서비스')으로 이동해 "
                "편집 대상을 찾고, 거기서 '수정'/'편집' 링크·버튼을 CLICK하라. "
                "등록 경로로 가려는 시도는 드리프트다. "
                + ("⚠️ 단, 위의 [🔐 비로그인 차단 경로]에 해당 리소스 목록 경로가 있으면 "
                   "더 이상 그 경로로 NAVIGATE 하지 말고 즉시 ASK_USER로 사용자 로그인을 요청하라."
                   if _auth_bl else "")
            )
        elif _verb == "CREATE":
            _verb_rule = (
                "0. 🎯 [VERB=CREATE — 최우선] **신규 리소스를 등록**하는 작업이다. "
                "기존 항목 편집 경로(/edit · /update) 금지. "
                "등록 폼/위저드로 이동해 필드를 채우고 제출하라."
            )
        elif _verb == "READ":
            _verb_rule = (
                "0. 🎯 [VERB=READ — 최우선] **읽기/조회** 작업이다. "
                "FILL · 제출 버튼 CLICK · 상태 변경 액션 금지. "
                "반드시 순서: NAVIGATE(대상 페이지) → READ_PAGE(target=변수명) → WRITE(content에 '{변수명}' 참조로 요약). "
                "★READ_PAGE 없이 WRITE 금지. LLM이 페이지 내용을 상상으로 지어내는 것 차단 — "
                "WRITE 전에 반드시 READ_PAGE로 실제 페이지 텍스트를 변수에 저장한 뒤 그 변수를 참조해 요약하라. "
                "CSS 셀렉터(예: 'div.xxx p:first-of-type')를 WRITE target이나 content에 넣지 마라 — 셀렉터는 페이지 읽기 수단이지 요약 결과가 아니다."
            )

        # ── [T2 Step 5 2026-05-01] EXECUTION_ACTION 규칙 — schema.execution_action 명시 시
        # LLM 이 외부 SDK verb 사용하도록 강제. WRITE 만으로 종료 차단 (Phase 1-ε 가드와 짝).
        _exec_action_rule = ""
        try:
            _ea_v = getattr(schema, "execution_action", None) if schema else None
            if isinstance(_ea_v, dict) and _ea_v.get("action"):
                _ea_action = _ea_v.get("action", "")
                _ea_target = _ea_v.get("target_hint", "")
                if _ea_action == "telegram_send":
                    _exec_action_rule = (
                        f"\n0. 🚀 [EXECUTION_ACTION=telegram_send — 최우선] 이 task 는 "
                        f"마지막 step 에 **반드시 TELEGRAM_SEND verb 사용** 해 사용자 텔레그램 "
                        f"채팅으로 결과 메시지를 발송해야 한다. WRITE 만으로 끝내면 거부됨. "
                        f"권장 순서: NAVIGATE → READ_PAGE → WRITE(메시지 본문 변수 생성) → "
                        f"**TELEGRAM_SEND(target='제목 또는 빈 문자열', content='{{메시지변수}}')**. "
                        f"target_hint: {_ea_target or '텔레그램 채팅'}.\n"
                    )
                elif _ea_action == "email_send":
                    _exec_action_rule = (
                        f"\n0. 🚀 [EXECUTION_ACTION=email_send — 최우선] 이 task 는 "
                        f"마지막 step 에 **반드시 EMAIL_SEND verb 사용** 해 이메일을 발송해야 한다. "
                        f"WRITE 만으로 끝내면 거부됨. 권장 순서: NAVIGATE/READ_PAGE 로 자료 수집 → "
                        f"WRITE(본문 변수 생성, 첫 줄에 'Subject: 제목') → "
                        f"**EMAIL_SEND(target='수신자 @이메일', content='{{본문변수}}')**. "
                        f"target_hint: {_ea_target or '이메일 수신자'}. "
                        f"이메일 주소가 필수 — schema.where 또는 사용자 입력에서 추출.\n"
                    )
                elif _ea_action:
                    # 다른 execution_action (email_send/kmong_publish 등) — 아직 verb 미구현이지만
                    # LLM 이 의도 가지도록 가이드. ASK_USER 로 자기 한계 인식 권장.
                    _exec_action_rule = (
                        f"\n0. 🚀 [EXECUTION_ACTION={_ea_action} — 최우선] 이 task 는 마지막 "
                        f"step 에 외부 행위 '{_ea_action}' (target: {_ea_target}) 를 실행해야 한다. "
                        f"WRITE 만으로 끝내면 거부됨. 해당 verb 가 미구현이면 ASK_USER 로 사용자에게 "
                        f"한계 보고하고 끝내라.\n"
                    )
        except Exception as _e_ear:
            print(f"  ⚠️ [T2 Step 5] _exec_action_rule 빌드 실패 (무시): {_e_ear}")

        # ── [AA 2026-05-01] 수집 진척 가시화 — Top N 류 task 의 "스나이퍼" 정확도 ↑ ──
        # _react_vars 의 collected items (READ_PAGE 결과 변수) 개수 + 이름을 prompt 에
        # 노출. schema.attributes 의 정량 임계 (N개 이상) 와 비교해 "X/N" 진척 표시.
        # 효과: LLM 이 "지금 3/5, 2개 더 필요" 인식 → 카테고리 우왕좌왕 차단 + 깊이 ↑.
        _progress_rule = ""
        try:
            _react_vars = getattr(schema, "_react_vars", None) or {} if schema else {}
            if _react_vars:
                # 데이터 변수 (READ_PAGE/page/detail/content 류) vs 산출물 (.md/.txt 등) 분리
                _data_vars: List[Tuple[str, int]] = []
                _output_vars: List[Tuple[str, int]] = []
                for _vk, _vv in _react_vars.items():
                    if not isinstance(_vk, str) or not isinstance(_vv, str):
                        continue
                    _vlen = len(_vv)
                    if _vlen < 200:
                        continue  # placeholder/너무 짧은 변수 제외
                    _vkl = _vk.lower()
                    # 산출물 파일 — 확장자 매칭
                    if any(_vkl.endswith(ext) for ext in (
                        ".md", ".txt", ".json", ".csv", ".html", ".yaml", ".yml",
                    )):
                        _output_vars.append((_vk, _vlen))
                    # 데이터 변수 — page/detail/content/text 패턴 매칭
                    elif any(p in _vkl for p in (
                        "page", "detail_", "_content", "_text", "_data",
                        "category", "result", "source",
                    )):
                        _data_vars.append((_vk, _vlen))
                    else:
                        # 기타 — 데이터로 간주 (보수적 — Top N 누적용)
                        _data_vars.append((_vk, _vlen))

                # 정량 임계 추출 (HOW_MUCH 의 N개 이상 패턴)
                _quant_target = 0
                try:
                    import re as _re_q
                    for _attr in (schema.attributes or []):
                        _name = str(getattr(_attr, "name", "") or "")
                        _val = str(getattr(_attr, "value", "") or "")
                        _blob_q = f"{_name} {_val}"
                        _m = _re_q.search(r"(\d+)\s*개\s*이상|>=\s*(\d+)|>\s*(\d+)", _blob_q)
                        if _m:
                            for _g in _m.groups():
                                if _g:
                                    _q = int(_g)
                                    if _q > _quant_target:
                                        _quant_target = _q
                except Exception:
                    pass

                if _data_vars or _output_vars or _quant_target:
                    _data_n = len(_data_vars)
                    _out_n = len(_output_vars)
                    _data_preview = ", ".join(
                        f"{k}({n}자)" for k, n in _data_vars[:5]
                    ) if _data_vars else "(없음)"
                    _out_preview = ", ".join(
                        f"{k}({n}자)" for k, n in _output_vars[:3]
                    ) if _output_vars else "(없음)"
                    # 정량 진척 표시 (target > 0 일 때만)
                    _quant_line = ""
                    if _quant_target > 0:
                        _need_more = max(0, _quant_target - _data_n)
                        if _need_more > 0:
                            _quant_line = (
                                f"\n  ★ 정량 목표: {_data_n}/{_quant_target} 모음 — "
                                f"**{_need_more}개 더 필요** (현재 카테고리 깊이 보거나 다음 항목 추가)."
                            )
                        else:
                            _quant_line = (
                                f"\n  ✅ 정량 충족: {_data_n}/{_quant_target} — "
                                f"WRITE 로 통합 산출물 작성 후 마무리하라."
                            )
                    # [BB 2026-05-01] WRITE 통합 가이드 — 첫 산출물 파일명 강조 + 분산 경고.
                    _bb_consolidation_line = ""
                    _first_w = getattr(schema, "_first_write_target", "") if schema else ""
                    _warn_consol = bool(getattr(schema, "_write_consolidation_warn", False)) if schema else False
                    if _first_w:
                        if _warn_consol:
                            _bb_consolidation_line = (
                                f"\n  🔴 [BB WRITE 통합 강제] 같은 task 에서 다른 파일명으로 WRITE "
                                f"분산 감지. **반드시 첫 파일 `{_first_w}` 에 보강 (덮어쓰기 또는 "
                                f"누적). 새 파일 만들기 금지.**"
                            )
                        else:
                            _bb_consolidation_line = (
                                f"\n  📝 [BB WRITE 통합] 첫 산출물 파일: `{_first_w}` — "
                                f"이후 WRITE 는 같은 파일에 보강 권장 (분산 차단)."
                            )
                    _progress_rule = (
                        f"\n[📊 수집 진척 (AA 2026-05-01)]\n"
                        f"  • 데이터 변수: {_data_n}건 — {_data_preview}\n"
                        f"  • 산출물 파일: {_out_n}건 — {_out_preview}{_quant_line}{_bb_consolidation_line}\n"
                        f"  ⚡ 가이드: 같은 카테고리 깊이 N개 모은 후 다음 카테고리. "
                        f"우왕좌왕/얕은 탐색 금지. 정량 미달 시 **현재 카테고리에서 더 깊이**."
                    )
        except Exception as _e_prog:
            print(f"  ⚠️ [AA] _progress_rule 빌드 실패 (무시): {_e_prog}")

        # ── [I 2026-04-26] MCTS-lite 헬퍼 — stuck/drift 시 후보 액션 ranking ─
        # 그록 제안 #3 MCTS 의 정신을 LLM 호출 0 추가로 흉내. action_pool 휴리스틱
        # reward 로 점수 매겨 top-1 을 Think prompt 에 힌트로 주입. WorldModel
        # 있으면 simulate_plan 으로 가중. LLM 결정에 강제 아닌 추천. stuck 회복
        # 자동화 + drift 빠지는 케이스 보완.
        def _mcts_lite_recommend(_hs: Dict, _schema, _obs: Dict, _hist: list) -> Optional[Dict[str, Any]]:
            """휴리스틱 reward 로 후보 액션 평가 → top-1 추천 dict 반환.
            stuck/drift 안정 시 None (힌트 미주입)."""
            try:
                _stuck = _hs.get("stuck") or {}
                _drift_lvl = (_hs.get("drift_level") or "GREEN")
                _captcha = bool(_hs.get("captcha"))
                _auth_bl = _hs.get("auth_blacklist") or []
                _is_stuck = bool(_stuck.get("is_stuck"))
                _hash_n = int(_stuck.get("same_hash_count", 0) or 0)
                _noprog_n = int(_stuck.get("no_progress_steps", 0) or 0)
                # 발동 조건: stuck OR drift YELLOW+ OR captcha OR auth_bl 3+
                _trigger = (
                    _is_stuck
                    or _drift_lvl in ("YELLOW", "RED")
                    or _captcha
                    or len(_auth_bl) >= 3
                )
                if not _trigger:
                    return None
                # 후보 풀 생성
                _cands: List[Dict[str, Any]] = []
                # 1) GO_BACK
                _cands.append({
                    "action": "GO_BACK", "target": "", "content": "",
                    "reason": "이전 페이지 복귀 후 다른 경로 탐색",
                    "score": 0.0,
                })
                # 2) WHERE 홈 NAVIGATE
                _where = (getattr(_schema, "where", "") or "").strip() if _schema else ""
                if _where.startswith("http"):
                    _cands.append({
                        "action": "NAVIGATE", "target": _where, "content": "",
                        "reason": "WHERE 홈 도메인 복귀",
                        "score": 0.0,
                    })
                # 3) ASK_USER (강한 신호 시)
                _cands.append({
                    "action": "ASK_USER", "target": "", "content": "어디로 갈지 또는 어떤 버튼/링크를 클릭할지 알려줘",
                    "reason": "사용자 안내 요청",
                    "score": 0.0,
                })
                # 4) pending attr 키워드 기반 NAVIGATE (휴리스틱)
                _pending = _hs.get("pending_attrs") or []
                if _pending and _where.startswith("http"):
                    _kw = _pending[0].split("_")[0][:20]
                    if _kw and len(_kw) >= 2:
                        _cands.append({
                            "action": "NAVIGATE",
                            "target": f"{_where.rstrip('/')}/search?q={_kw}",
                            "content": "", "reason": f"미달성 속성 '{_pending[0]}' 추적 검색",
                            "score": 0.0,
                        })
                # 5) 히스토리에 없는 새 [N] CLICK (clickable 인덱서 결과 활용)
                try:
                    _seen_targets = set()
                    for _h in (_hist or []):
                        if _h.get("action") == "CLICK":
                            _t = (_h.get("target") or "").strip()
                            if _t.startswith("["):
                                _seen_targets.add(_t)
                    _clickable = (_obs.get("clickable") or "")
                    import re as _re_clk
                    _all_idx = set(_re_clk.findall(r"\[(\d+)\]", _clickable))
                    _new_idx = sorted(_all_idx - {t.strip("[]") for t in _seen_targets}, key=lambda x: int(x))
                    if _new_idx:
                        _cands.append({
                            "action": "CLICK", "target": f"[{_new_idx[0]}]", "content": "",
                            "reason": "히스토리에 없는 새 클릭 후보",
                            "score": 0.0,
                        })
                except Exception:
                    pass
                # 휴리스틱 reward 계산
                for _c in _cands:
                    _act = _c["action"]
                    _s = 0.0
                    if _act == "GO_BACK":
                        _s += 0.7 if _is_stuck else 0.3
                        if _hash_n >= 3: _s += 0.2
                    elif _act == "NAVIGATE":
                        _s += 0.5 if _is_stuck else 0.3
                        if _drift_lvl in ("YELLOW", "RED"): _s += 0.2
                        # 같은 URL 반복은 감점
                        _tgt = _c.get("target", "")
                        for _h in (_hist or [])[-6:]:
                            if _h.get("action") == "NAVIGATE" and _h.get("target") == _tgt:
                                _s -= 0.4
                                break
                    elif _act == "ASK_USER":
                        if _captcha: _s = 0.9
                        elif len(_auth_bl) >= 3: _s = 0.7
                        elif _noprog_n >= 8: _s = 0.6
                        else: _s = 0.3
                    elif _act == "CLICK":
                        _s = 0.4 + (0.2 if _is_stuck else 0.0)
                    _c["score"] = round(max(0.0, min(1.0, _s)), 2)
                # WorldModel.simulate_plan 가중 (있으면)
                try:
                    _wm = getattr(self.core, "world_model", None)
                    if _wm is not None and hasattr(_wm, "simulate_plan"):
                        for _c in _cands:
                            try:
                                _r = _wm.simulate_plan([{"action": _c["action"], "target": _c["target"]}]) or {}
                                _la = float(_r.get("lookahead_score", 0.5) or 0.5)
                                _c["score"] = round((_c["score"] * 0.7 + _la * 0.3), 2)
                            except Exception:
                                pass
                except Exception:
                    pass
                if not _cands:
                    return None
                _cands.sort(key=lambda x: x["score"], reverse=True)
                return _cands[0]
            except Exception as _mle:
                print(f"  [MCTS-lite] 추천 계산 실패 (무시): {_mle}")
                return None

        # ── [H 2026-04-26] Hybrid Reasoning 블록 ──────────────────────────
        # 그록 대화의 "Hybrid Reasoning (상향+하향 동시)" 적용:
        #   - 하향식: 목표 경로 (계획성) — schema/anchor 가 가리키는 다음 단계
        #   - 상향식: 관찰 신호 (발견성) — stuck/auth/click_noop/drift 등 위험 패턴
        # 둘이 일치하면 진행, 충돌 시 reason 에 통합 명시. LLM 의 "갑자기 생각
        # 바꿈" 을 구조적으로 막고, drift 가 빠지는 케이스도 신호로 잡는다.
        # 토글: env EIDOS_HYBRID_REASONING (기본 1).
        _hybrid_block = ""
        try:
            import os as _os_h
            _hybrid_on = (_os_h.environ.get("EIDOS_HYBRID_REASONING", "1") or "1").strip() not in ("0", "false", "off")
        except Exception:
            _hybrid_on = True
        if _hybrid_on and hybrid_state:
            _td_lines: List[str] = []
            _bu_lines: List[str] = []
            # 하향식 — 목표/계획
            _td_what = (hybrid_state.get("what") or "").strip()
            _td_pending = hybrid_state.get("pending_attrs") or []
            _td_achieved_n = hybrid_state.get("achieved_count", 0)
            _td_total_n = hybrid_state.get("total_attr_count", 0)
            if _td_what:
                _td_lines.append(f"  · 목표(WHAT): {_td_what[:120]}")
            if _td_total_n:
                _td_lines.append(
                    f"  · 진척: {_td_achieved_n}/{_td_total_n} 속성 달성"
                )
            if _td_pending:
                _td_lines.append(
                    f"  · 미달성 우선순위: {', '.join(_td_pending[:3])}"
                    + (f" 외 {len(_td_pending)-3}개" if len(_td_pending) > 3 else "")
                )
            # 상향식 — 관찰 신호
            _bu_stuck = hybrid_state.get("stuck") or {}
            if _bu_stuck.get("is_stuck"):
                _bu_lines.append(
                    f"  · 🛑 막힘 감지 — 페이지 지문 {_bu_stuck.get('same_hash_count',0)}회 연속 / "
                    f"{_bu_stuck.get('no_progress_steps',0)} step 진전 0"
                )
            elif _bu_stuck.get("same_hash_count", 0) >= 2:
                _bu_lines.append(
                    f"  · ⚠️ 페이지 지문 {_bu_stuck.get('same_hash_count')}회 동일 (정체 조짐)"
                )
            elif _bu_stuck.get("no_progress_steps", 0) >= 3:
                _bu_lines.append(
                    f"  · ⚠️ 최근 {_bu_stuck.get('no_progress_steps')} step 속성 진전 없음"
                )
            _bu_auth_bl = hybrid_state.get("auth_blacklist") or []
            if len(_bu_auth_bl) >= 2:
                _bu_lines.append(f"  · 🔐 auth 블랙리스트 {len(_bu_auth_bl)}개 누적")
            _bu_fill_bl = hybrid_state.get("fill_blacklist") or []
            if _bu_fill_bl:
                _bu_lines.append(f"  · 📝 FILL 영구 차단 {len(_bu_fill_bl)}건")
            _bu_uh = hybrid_state.get("unhydrated_streak") or {}
            _bu_uh_max = max(_bu_uh.values()) if isinstance(_bu_uh, dict) and _bu_uh else 0
            if _bu_uh_max >= 2:
                _bu_lines.append(f"  · 💧 SPA hydration 미완 streak={_bu_uh_max}")
            if hybrid_state.get("captcha"):
                _bu_lines.append("  · 🤖 CAPTCHA 감지")
            _bu_drift_lvl = hybrid_state.get("drift_level")
            _bu_drift_score = hybrid_state.get("drift_score")
            if _bu_drift_lvl and _bu_drift_lvl != "GREEN":
                _bu_lines.append(
                    f"  · 🚨 DriftGuard {_bu_drift_lvl} (score={_bu_drift_score:.2f})"
                    if isinstance(_bu_drift_score, (int, float)) else
                    f"  · 🚨 DriftGuard {_bu_drift_lvl}"
                )
            # ── MCTS-lite 추천 (stuck/drift 시) ────────────────────────
            # env EIDOS_MCTS_HINT 토글 (기본 1). action_pool 휴리스틱 reward 로
            # top-1 후보 산출, Think prompt 에 추천 힌트 주입. LLM 호출 0 추가.
            _mcts_hint_block = ""
            try:
                _mcts_on = (_os_h.environ.get("EIDOS_MCTS_HINT", "1") or "1").strip() not in ("0", "false", "off")
            except Exception:
                _mcts_on = True
            if _mcts_on:
                _mcts_top = _mcts_lite_recommend(hybrid_state, schema, obs, history)
                if _mcts_top:
                    _mt_act = _mcts_top.get("action", "")
                    _mt_tgt = (_mcts_top.get("target", "") or "")[:80]
                    _mt_score = _mcts_top.get("score", 0.0)
                    _mt_reason = _mcts_top.get("reason", "")
                    # ── [Phase 1 Chain Health 2026-04-26] MCTS 신호 → schema ──
                    # 추천 점수 자체가 chain 진척 신뢰도의 한 채널. Verifier 보다
                    # 가벼운 가중(0.35) 으로 _compute_health 에 들어감.
                    try:
                        from eidos_confidence import update_chain_health as _uch_mcts
                        _uch_mcts(
                            schema,
                            mcts_score=float(_mt_score),
                            event=f"mcts:{_mt_act}",
                        )
                    except Exception:
                        pass
                    _mcts_hint_block = (
                        f"\n\n[🌳 MCTS-lite 추천 — 휴리스틱 reward 평가]\n"
                        f"  · 후보: {_mt_act}"
                        + (f" → {_mt_tgt}" if _mt_tgt else "")
                        + f" (score={_mt_score:.2f})\n"
                        f"  · 근거: {_mt_reason}\n"
                        "  → 강제 아님. 너의 판단이 더 낫다고 확신하면 무시 가능. "
                        "단, 반대 판단 시 reason 에 'MCTS=X 였지만 Y 한 이유: ...' 명시."
                    )

            # 두 채널 합성
            if _td_lines or _bu_lines:
                _hybrid_block = (
                    "\n[🎯 하향식 — 계획 경로]\n"
                    + ("\n".join(_td_lines) if _td_lines else "  · (정보 부족)")
                    + "\n\n[🔭 상향식 — 관찰 신호]\n"
                    + ("\n".join(_bu_lines) if _bu_lines else "  · 위험 신호 없음")
                    + _mcts_hint_block
                    + "\n\n[★ Hybrid 통합 규칙]\n"
                    "  - 두 채널이 일치하면 계획대로 진행.\n"
                    "  - 충돌 시 상향식 신호 강도가 높으면(stuck/auth_bl/captcha/drift≥YELLOW) "
                    "상향식 우선 — GO_BACK / 새 NAVIGATE / ASK_USER 로 경로 변경.\n"
                    "  - MCTS-lite 추천이 있으면 그 점수도 가중 — 단순 페이지 변화 신호면 "
                    "하향식 유지, 강한 정체/auth 신호면 MCTS 추천 따라가기.\n"
                    "  - reason 형식 예: \"[↓계획: 미달성 X 채우러 NAVIGATE] "
                    "[↑신호: 정체 5step] [🌳 MCTS=GO_BACK 0.85] [통합: GO_BACK 후 다른 링크]\".\n"
                )

        # ── [Phase 1 Chain Health 2026-04-26] 프롬프트 신뢰도 블록 ──
        # Verifier/MCTS 신호 누적치 → LLM 자기보정용 정보. 임계치 강제 X.
        # 토글 off / 신호 0건 → 빈 문자열 (블록 자체 생략).
        _health_block = ""
        try:
            from eidos_confidence import health_block_for_prompt as _hbf
            _health_block = _hbf(schema) or ""
        except Exception:
            _health_block = ""

        prompt = f"""{self._compact_anchor(anchor)}

[실행할 작업]
{task_prompt}

[현재 상태]
{_browser_ctx}
{_auth_bl_block}{_fill_bl_block}{_forbidden_urls_block}
[직전 실행 히스토리]
{history_str or "없음 (첫 step)"}

[이미 달성된 HOW_MUCH 속성]
{achieved_attrs or "없음"}

[아직 미달성 HOW_MUCH 속성]
{pending_attrs or "없음 (모두 달성됨)"}
{stored_vars_block}
{skill_hints_block}
{skill_macro_rule_block}
{memory_episodic_block}
{self_awareness_block}{_hybrid_block}{_health_block}
{f'''
[⚠️ DRIFT CORRECTION — 즉시 적용 필수]
{drift_correction}
''' if drift_correction else ''}
[결정 규칙 — 반드시 JSON 한 줄로만 응답]
{_target_nav_hint}{_verb_rule}{_exec_action_rule}{_progress_rule}
1. CLICK target은 요소 목록에 `[N]` 번호가 보이면 `[N]`만 입력(예: "target":"[7]"). 번호가 없을 때만 CSS 셀렉터 사용. 목록에 없는 요소 추측 금지.
2. 모든 HOW_MUCH 속성이 달성되면 action을 DONE으로.
3. 지금 하려는 액션이 WHERE 범위 밖이면 즉시 중단하고 ASK_USER.
4. CSS 셀렉터를 추측할 수 없으면 ASK_USER로 사용자에게 질문.
5. 같은 액션을 3번 연속 실패했으면 다른 경로 시도.
6. URL이 "조회 실패"이거나 비어있으면 WHERE의 목표 URL로 NAVIGATE를 먼저 실행하라.
7. reason 필드에 반드시 "왜 이 액션인지" 한 줄로 설명.
14. 잘못된 페이지로 들어왔거나 진전이 없다고 판단되면 GO_BACK 액션으로 직전 페이지로 되돌아간 뒤 다른 링크/버튼을 CLICK하라. 판단 근거: (a) 같은 페이지를 2회 이상 보고도 미달성 속성이 전혀 줄지 않음 (b) 현재 페이지가 에러/빈 검색 결과/무관 콘텐츠 (c) 히스토리에 '동일 URL 반복 차단'·'자동 변형'·'stuck 감지' 표시. 같은 막다른 URL로 재진입 금지.
12. 팝업/모달이 화면을 가리고 있으면 CLICK으로 닫기 버튼(X, 닫기 텍스트 포함)을 먼저 클릭하거나, 이미 자동으로 닫혔을 수 있으니 클릭 가능 요소를 다시 확인하라. 팝업 때문에 ASK_USER를 선택하지 마라.
13. 스크린샷이 제공된 경우 클릭 가능한 요소 목록이 비어도 CLICK_XY로 좌표 클릭을 시도할 수 있다. target에 "x,y" 형식으로 좌표를 입력하라. 예: {{"action":"CLICK_XY","target":"430,520","content":"상품카드 클릭","reason":"검색결과 상품 클릭"}}. ★단, 직전 히스토리에서 CLICK_XY를 2회 이상 시도했는데 URL/화면이 변하지 않았거나 "좌표에 요소 없음"/"변화 없음" 실패 메시지가 보이면 더 이상 좌표를 찍지 말고 즉시 ASK_USER로 사용자에게 물어라. 질문 형식: "어떤 버튼/링크를 눌러야 해? 버튼 텍스트(예: '수정', '편집', '대시보드')나 화면 위치(예: '우측 상단 카드'), 또는 바로 가고 싶은 페이지 URL을 알려줘. 또는 네가 직접 클릭한 뒤 ⚡ 자율 실행을 눌러도 돼." — 절대 좌표값(x,y 숫자)을 사용자에게 묻지 마라. 사용자는 픽셀 좌표를 알 수 없다. 눈먼 좌표 폭격 금지.
{_kmong_rules}{_non_browser_rule}

다음 JSON 형식으로만 답해. 다른 텍스트 없이:
{{"action": "NAVIGATE", "target": "URL", "content": "", "reason": "이유"}}
action은 NAVIGATE/CLICK/FILL/SCROLL/READ_PAGE/WRITE/WAIT/DONE/ASK_USER/GO_BACK/TELEGRAM_SEND/EMAIL_SEND 중 하나.
SCROLL은 target을 "bottom"/"top"/"px:숫자"/"" 중 하나로 지정(빈 값이면 viewport 80% 아래).
READ_PAGE는 현재 브라우저 페이지의 본문 텍스트를 변수에 저장한다. target=저장할 변수명(예: "page_text"), content=빈값. 이후 WRITE의 content에 "{{변수명}}" 형태로 참조하라.
WRITE는 LLM으로 텍스트를 '생성'한다. 페이지 내용이 필요하면 WRITE 전에 반드시 READ_PAGE로 변수를 먼저 만들고 그 변수를 WRITE content에서 참조하라. WRITE target에 CSS 셀렉터를 넣지 마라.
📱 **TELEGRAM_SEND** (T2 2026-05-01): 사용자 본인 텔레그램 채팅에 메시지 발송 — 외부 SDK 호출 verb. target=제목 (생략 가능, 빈 문자열이면 "🚀 EIDOS"), content=메시지 본문 또는 변수 참조 ("{{변수명}}" 치환 지원). 4096자 한도 자동 잘림. schema.execution_action="telegram_send" task 의 마지막 step 에서 사용. 예: {{"action":"TELEGRAM_SEND","target":"오늘의 USD/KRW 환율","content":"{{usd_krw_rate.md}}","reason":"환율 요약을 사용자에게 발송"}}.
📧 **EMAIL_SEND** (P-ε 2026-05-01): SMTP 이메일 발송 — 외부 SDK 호출 verb. target=수신자 이메일 주소 (필수, "@" 포함), content=본문 (변수 참조 지원). 첫 줄이 "Subject: ..." 면 제목 추출, 아니면 "🚀 EIDOS" 기본. eidos_files/email_config.json 미설정 시 graceful skip. schema.execution_action="email_send" task 에서 사용. 예: {{"action":"EMAIL_SEND","target":"client@example.com","content":"Subject: 회의록\\n\\n{{meeting_notes.md}}","reason":"클라이언트에 회의록 발송"}}.
⚠️ WRITE의 content는 **짧은 지시문/브리프(200자 이내)** 로만 작성하라. 본문 전체를 content에 넣지 마라 — 실제 문서 본문은 별도의 분석가 LLM이 생성한다. 예: content="크몽 수요 상위 3개 디지털 서비스와 가격/설명을 markdown 표로 정리" 처럼 짧게. 본문을 여기 넣으면 JSON 토큰 한계로 잘려 실패한다.
📁 **WRITE target 파일 규칙**: 사용자에게 전달할 **최종 산출물(보고서/기획안/목록/분석)** 은 target을 반드시 **파일 경로(.md 확장자 권장)** 로 지정하라. 예: `"target": "kmong_top3_services.md"`, `"target": "product_proposal.md"`. 순수 변수명(예: `top_services_list`, `product_proposal_draft`)만 쓰면 메모리에만 남고 세션 종료 시 소실된다. 중간 계산용 참조변수일 때만 확장자 없는 변수명 허용. 같은 산출물에 2번째 WRITE 하지 마라 — 동일 target에 보강하려면 반드시 확장자 포함 파일명으로 통일.
GO_BACK은 target/content 비워두면 브라우저 뒤로가기(window.history.back())를 실행한다. 잘못 진입한 페이지에서 복귀할 때만 사용."""

        # [Fix] Gemini 2.5 Flash는 thinking 모델 — thinking이 출력 버짓을 소모한다.
        # 1024면 thinking만으로 900+를 쓰고 JSON 출력이 ~40~100 토큰에서 잘려
        # RobustParser가 실패함 (로그: Out:41, Out:48, Out:60...).
        # 4096으로 올려서 thinking 후에도 JSON 한 줄 완성 여유 확보.
        _think_max_tokens=16384 if _is_non_browser else 4096
        _think_timeout    = 60.0   # [Fix] 120 → 60 — 무응답 시 빠른 실패 전환

        # _think 전용 시스템 프롬프트 — 자연어 응답 완전 차단
        _THINK_SYSTEM = (
            "You are a JSON-only action decision engine. "
            "You must ALWAYS respond with exactly ONE line of valid JSON and nothing else. "
            "Never use natural language. Never add explanation. "
            'Format: {"action": "...", "target": "...", "content": "...", "reason": "..."} '
            "action must be one of: NAVIGATE, CLICK, FILL, SCROLL, READ_PAGE, WRITE, WAIT, DONE, ASK_USER, GO_BACK. "
            "For WRITE action: 'content' MUST be a SHORT brief/instruction (under 200 chars). "
            "Do NOT put the actual document body in 'content' — a separate analyst LLM generates the full body. "
            "Writing long text in 'content' will cause JSON truncation and failure. "
            "For WRITE action: 'target' MUST be a file path with extension (e.g. 'report.md', 'plan.md') "
            "for any final deliverable. Plain variable names like 'top_services_list' are memory-only "
            "and LOST when the session ends. Always use a .md filename for user-facing artifacts. "
            "If you cannot determine the next action, output: "
            '{"action": "ASK_USER", "target": "", "content": "Cannot determine next action", "reason": "unclear"}'
        )

        # [Fix] 서버가 이미지 미지원 — PNG 바이트 읽지 않고 텍스트 힌트만 추가
        # (기존: open().read()로 수 MB PNG를 블로킹 read 후 서버엔 text-only로 전송해 낭비)
        _ss_path = obs.get("screenshot", "")
        if _ss_path and not _js_links_available:
            _coord_rule = (
                "\n\n[스크린샷 기반 좌표 클릭 모드]\n"
                "클릭 가능한 요소 목록이 부족할 때 CLICK_XY 액션으로 화면 좌표를 직접 클릭하세요.\n"
                '{"action":"CLICK_XY","target":"430,520","content":"상품카드 클릭"}\n'
                "브라우저 뷰포트 크기는 약 1200x800입니다.\n"
                "상품 카드/버튼/링크의 중심 좌표를 추정해서 클릭하세요."
            )
            prompt = prompt + _coord_rule

        try:
            # [Fix 2026-04-21] get_llm_response_async 기본 timeout=20 은 긴 Think
            # 프롬프트(anchor+obs+history) 에 짧아 "[서버 응답 시간 초과]" 가 자주
            # 떠 fail_streak 증가 → 조기 ASK_USER. _think_timeout 과 일치시킨다.
            raw = await asyncio.wait_for(
                get_llm_response_async(
                    prompt,
                    max_tokens=_think_max_tokens,
                    system_prompt=_THINK_SYSTEM,
                    timeout=int(_think_timeout),
                ),
                timeout=_think_timeout + 5,
            )
            # 서버 오류 응답 감지 → None 반환 (fail_streak 증가)
            _ERR_PREFIXES = ("[서버 오류", "[크레딧", "[Pro", "[서버 연결 실패",
                             "[서버 응답 시간 초과", "[LLM 호출 오류")
            if raw and any(raw.startswith(p) for p in _ERR_PREFIXES):
                print(f"  [Think] 서버 오류 감지: {raw[:60]}")
                return None
            result = _robust_json_parse(raw)
            if isinstance(result, dict) and result.get("action"):
                return result

            # 자연어 응답 감지 — JSON 추출 마지막 시도
            _raw_preview = repr(raw)[:80]
            if raw and not raw.strip().startswith("{"):
                # 응답에서 JSON 블록만 추출 시도
                import re as _re_json
                _json_m = _re_json.search(r'\{[^{}]*"action"\s*:\s*"[^"]+[^{}]*\}', raw)
                if _json_m:
                    _rescued = _robust_json_parse(_json_m.group(0))
                    if isinstance(_rescued, dict) and _rescued.get("action"):
                        print(f"  [Think] 자연어에서 JSON 구조 복구 성공")
                        return _rescued
            print(f"  [Think] JSON 파싱 실패, raw={_raw_preview}")

            # [Fix] WRITE content 절단 감지 → 1회 재시도 (brief 강제 지시)
            # raw가 '{"action":"WRITE"' 로 시작하고 닫는 '}' 없으면 content에
            # 긴 본문 넣다 JSON이 절단된 것. target만 추출해 재시도.
            _raw_s = (raw or "").strip()
            if (_raw_s.startswith("{")
                    and '"action"' in _raw_s[:40]
                    and '"WRITE"' in _raw_s[:80]
                    and not _raw_s.rstrip().endswith("}")):
                try:
                    import re as _re_w
                    _tm = _re_w.search(r'"target"\s*:\s*"([^"]{1,200})"', _raw_s)
                    _extracted_target = _tm.group(1) if _tm else ""
                    _retry_hint = (
                        "\n\n[중요 재시도 지시] 직전 응답이 WRITE content에 긴 본문을 "
                        "넣어 JSON이 절단됐다. 이번엔 반드시 아래 형식으로만 답하라:\n"
                        '{"action":"WRITE","target":"' + _extracted_target +
                        '","content":"<한 줄 브리프 200자 이내>","reason":"<한줄>"}\n'
                        "content는 본문이 아니라 '무엇을 쓸지 지시하는 짧은 문장'이다. "
                        "본문은 분석가가 별도로 생성하므로 여기선 절대 본문 쓰지 마라."
                    )
                    print(f"  [Think] WRITE content 절단 감지 → 재시도 (target='{_extracted_target[:40]}')")
                    raw2 = await asyncio.wait_for(
                        get_llm_response_async(
                            prompt + _retry_hint,
                            max_tokens=1024,
                            system_prompt=_THINK_SYSTEM,
                            timeout=int(_think_timeout),
                        ),
                        timeout=_think_timeout + 5,
                    )
                    if raw2 and not any(raw2.startswith(p) for p in _ERR_PREFIXES):
                        _r2 = _robust_json_parse(raw2)
                        if isinstance(_r2, dict) and _r2.get("action"):
                            print(f"  [Think] WRITE 재시도 성공: action={_r2.get('action')}")
                            return _r2
                        # 자연어 섞여도 JSON 블록 추출 재시도
                        import re as _re_json2
                        _m2 = _re_json2.search(r'\{[^{}]*"action"\s*:\s*"[^"]+[^{}]*\}', raw2)
                        if _m2:
                            _r2b = _robust_json_parse(_m2.group(0))
                            if isinstance(_r2b, dict) and _r2b.get("action"):
                                print(f"  [Think] WRITE 재시도 — JSON 블록 추출 성공")
                                return _r2b
                except asyncio.TimeoutError:
                    print(f"  [Think] WRITE 재시도 timeout")
                except Exception as _re:
                    print(f"  [Think] WRITE 재시도 실패: {_re}")
        except asyncio.TimeoutError:
            print(f"  [Think] {_think_timeout:.0f}초 timeout — 서버 무응답")
        except Exception as e:
            print(f"  [Think] LLM 판단 실패: {e}")
        return None

    # ── Act ────────────────────────────────────────────────────────────────
    async def _try_finalize_approval(self, step: "ActionStep", result: str) -> None:
        """
        [Phase 2] Telegram 승인 메시지를 실행 결과로 덮어쓰기.

        step._pending_rid 가 세팅돼 있으면(Gate에서 APPROVED 시 부착) finalize_approval 호출.
        - success: result 가 '⚠️' 로 시작 안 하면 True
        - detail: 결과 문자열 최대 400자
        호출 후 step._pending_rid 는 비워 중복 호출 방지. 실패는 조용히 무시.
        """
        _prid = getattr(step, "_pending_rid", None)
        if not _prid:
            return
        try:
            from eidos_telegram_bot import get_bot as _get_bot
            _ok = not str(result or "").startswith("⚠️")
            await _get_bot().finalize_approval(
                rid=_prid, success=_ok, detail=str(result or "")[:400],
            )
        except Exception as _fe:
            print(f"  [Approval] finalize 실패 (무시): {_fe}")
        finally:
            try:
                step._pending_rid = None
            except Exception:
                pass

    async def _act(self, action: str, target: str, content: str, schema, reason: str = "") -> str:
        """
        Think에서 결정된 액션을 실행하고 결과 문자열 반환.

        reason: LLM이 이 액션을 선택한 근거 (decision["reason"]).
                Telegram 승인 메시지의 Why 필드로 전달.
        """
        # ── [Preservation Guard 2026-04-27] 자기 보존 hard limit 강제 ───────
        # eidos_self_definition.yaml.preservation.forbidden 4종 절대 금지선:
        # - delete_state_files: WRITE target 이 state_files 매칭 시 차단
        # - self_replicate: WRITE target 이 sandbox(eidos_files/) 외부 절대경로면 차단
        # 차단 시 즉시 return — _execute_step 진입 차단. 사용자 승인 우회 불가.
        # 회귀: yaml 미로드/rule 미선언/비-WRITE/빈 target 모두 fail-open.
        if action == "WRITE" and target:
            try:
                from eidos_self_preservation import get_guard
                _allowed, _p_reason = get_guard().check_write(
                    target=target, content=content or "", action_kind=action,
                )
                if not _allowed:
                    self._report(
                        f"🚨 **[자기 보존 차단]** WRITE `{target[:40]}` 거부 — {_p_reason[:120]}"
                    )
                    return f"🚨 자기 보존 차단: {_p_reason[:200]}"
            except Exception as _e_pres:
                # fail-open: 가드 자체 예외 시 차단하지 않음 (기존 흐름 보존)
                print(f"  ⚠️ [Preservation] 가드 예외 (무시): {_e_pres}")

            # ── [BB 2026-05-01] WRITE 통합 강제 — 같은 task 의 첫 WRITE 파일명 추적 ──
            # 첫 WRITE: schema._first_write_target 저장.
            # 두 번째+ WRITE 가 다른 파일명: 경고 + schema._write_consolidation_warn=True
            # 마킹. 다음 _think prompt 의 _progress_rule 가 이걸 보고 강한 가이드 추가.
            # 효과: 같은 task 에서 산출물 1개로 통합 (Top N 류 task 의 본질 fix).
            try:
                if schema is not None:
                    _tgt_l = (target or "").strip().lower()
                    # 산출물 파일 (.md/.txt/.json/.csv/.html/.yaml/.yml) 만 추적
                    _is_output_file = any(_tgt_l.endswith(ext) for ext in (
                        ".md", ".txt", ".json", ".csv", ".html", ".yaml", ".yml",
                    ))
                    if _is_output_file:
                        _first = getattr(schema, "_first_write_target", "") or ""
                        if not _first:
                            schema._first_write_target = (target or "").strip()
                            print(f"  📝 [BB] 첫 WRITE target 기록: '{(target or '').strip()}'")
                        elif _first.lower() != _tgt_l:
                            schema._write_consolidation_warn = True
                            self._report(
                                f"⚠️ **[BB WRITE 통합 권장]** 같은 task 의 새 산출물 파일 "
                                f"`{(target or '').strip()[:60]}` — 이전 파일 `{_first[:60]}` 에 "
                                f"보강 권장. 다음 step 부터 LLM 가이드 강화."
                            )
                            print(
                                f"  ⚠️ [BB] WRITE 분산 감지 — first='{_first}' vs new='{(target or '').strip()}'"
                            )
            except Exception as _e_bb:
                print(f"  ⚠️ [BB] WRITE 통합 추적 실패 (무시): {_e_bb}")

        # _execute_step 재사용 — 더미 ActionStep/ActionPlan 생성
        step = ActionStep(
            step_no=0,
            action=action,
            target=target,
            content=content,
            reason=reason,
        )
        # [Phase B-1] ValueEngine 으로 cost 힌트 주입 (approval_gateway Budget 입력용)
        try:
            from eidos_value_engine import ValueEngine as _VE
            step.cost_estimate = _VE.get().cost_of_step(action)
        except Exception:
            pass
        # 변수 치환을 위한 더미 plan
        # schema._react_vars: ReAct 루프 내 WRITE 결과 저장소 (영속)
        # schema가 None이면 (승인 후 _active_schema 초기화 경쟁 조건) 빈 dict로 안전 처리
        if schema is not None and not hasattr(schema, "_react_vars"):
            schema._react_vars = {}
        # [Fix] READ_PAGE 중복 방지용 URL→변수명 매핑
        if schema is not None and not hasattr(schema, "_react_var_urls"):
            schema._react_var_urls = {}

        # [AGI Phase 1C wire 2026-05-03] EventCausalDiscovery 자동 등록.
        # 모든 verb 가 event log 에 누적되어 PC algorithm 으로 인과 발견 가능.
        # graceful — record_verb_execution 자체가 try/except 로 감싸있어 호출 실패해도
        # _act 흐름 무영향. EC 미가용 시 silent skip.
        try:
            from eidos_event_hook import record_verb_execution
            _task_id = getattr(schema, "task_id", "") or getattr(schema, "_chain_id", "") or ""
            record_verb_execution(
                verb_type=str(action),
                target=str(target) if target else None,
                object_id=str(_task_id) if _task_id else None,
                metadata={"reason": (reason or "")[:120]} if reason else None,
            )
        except Exception:
            pass  # graceful — hook 실패 절대 _act 흐름 중단 X

        # [Fix 2026-04-25] READ_PAGE Target Mismatch Guard.
        # target='detail_<id>' 인데 현재 URL 이 그 상세 페이지가 아니면 거부.
        # Why: LLM 이 search/hub 페이지에 있으면서 READ_PAGE detail_<id> 를 호출해
        #      search 본문을 detail_<id> 로 저장해 URL→var 매핑이 오염되고, 이후 다른
        #      detail_<id> 요청이 모두 alias 로 빠져 실제 상세 수집이 불가능해짐
        #      (최신로그.txt step10→step13 연쇄). 불일치 시 즉시 거부하고 LLM 에
        #      NAVIGATE 선행을 강제하는 규칙을 schema.how 에 주입.
        if schema is not None and action == "READ_PAGE":
            try:
                _tgt_name_tm = (target or "").strip().lower()
                _tgt_id_m = re.match(r"detail_([a-z0-9][a-z0-9_\-]{0,40})$", _tgt_name_tm)
                if _tgt_id_m:
                    _tgt_id_tm = _tgt_id_m.group(1)
                    _cur_url_tm = (self._get_current_browser_url() or "").lower()
                    _DETAIL_SEGS_TM = (
                        "gig", "gigs", "product", "products", "item", "items",
                        "post", "posts", "article", "articles", "service", "services",
                        "listing", "listings", "view", "detail", "details",
                    )
                    _on_target_detail = any(
                        f"/{seg}/{_tgt_id_tm}" in _cur_url_tm for seg in _DETAIL_SEGS_TM
                    )
                    if not _on_target_detail:
                        _hint_tm = (
                            f"[강제 규칙] READ_PAGE target='{_tgt_name_tm}' 변수명은 상세 "
                            f"페이지 id='{_tgt_id_tm}' 를 가리키는데, 현재 URL 은 그 상세 "
                            f"페이지가 아니다. 먼저 NAVIGATE 로 /gig/{_tgt_id_tm} (또는 "
                            f"/product/{_tgt_id_tm} 등) 으로 이동한 뒤 READ_PAGE 하라. "
                            f"현재 URL 본문을 '{_tgt_name_tm}' 로 저장하지 마라."
                        )
                        if hasattr(schema, "how"):
                            schema.how = _hint_tm + "\n\n" + (schema.how or "")
                        print(
                            f"  🛑 [READ_PAGE Target Mismatch] target='{_tgt_name_tm}' "
                            f"인데 현재 URL '{_cur_url_tm[:80]}' 에 '/seg/{_tgt_id_tm}' 없음 → 거부"
                        )
                        self._report(
                            f"🛑 **[READ_PAGE 타겟 불일치]** `{_tgt_name_tm}` 변수명이 상세 "
                            f"id='{_tgt_id_tm}' 를 가리키지만 현재 URL 과 불일치 — NAVIGATE 선행 필요."
                        )
                        return (
                            f"READ_PAGE 실패: target='{_tgt_name_tm}' 변수명 id 와 현재 URL "
                            f"불일치. 먼저 NAVIGATE 로 /gig/{_tgt_id_tm} 등 상세 URL 이동 후 재시도."
                        )
            except Exception as _e_tm:
                pass  # 안전망 실패는 기본 흐름

        # [Fix] 동일 URL 재READ 사전 차단 — 이미 읽은 페이지를 다른 변수명으로 저장하려는 LLM 루프 차단.
        if schema is not None and action == "READ_PAGE":
            try:
                _cur_url = (self._get_current_browser_url() or "").strip()
                _url_key = _cur_url.split("#", 1)[0].rstrip("/").lower()
                if _url_key and _url_key in schema._react_var_urls:
                    _existing = schema._react_var_urls[_url_key]
                    _existing_val = schema._react_vars.get(_existing, "")
                    if _existing_val:
                        _tgt_var = (target or "").strip() or _existing
                        # 사용자가 새 변수명을 요청했다면 기존 값으로 alias만 생성 (재페치 스킵)
                        if _tgt_var != _existing:
                            schema._react_vars[_tgt_var] = _existing_val
                            schema._react_var_urls[_url_key] = _existing  # 원본 키 유지
                        print(f"  [Act] READ_PAGE 중복 차단: '{_url_key[:60]}' 이미 '{_existing}'로 저장됨 "
                              f"→ '{_tgt_var}' 별칭 생성 ({len(_existing_val)}자, 재페치 스킵)")
                        return f"페이지 읽기 완료 → 변수 '{_tgt_var}' ({len(_existing_val)}자, 중복 방지로 재사용)"
            except Exception as _e_dup:
                pass  # 안전망 실패는 무시하고 정상 흐름

        dummy_vars: Dict = dict(schema._react_vars) if schema is not None else {}  # 기존 변수 복사

        # {변수명} 패턴을 실제 값으로 치환
        if content and "{" in content:
            import re as _re2
            for var_name, var_val in dummy_vars.items():
                content = content.replace(f"{{{var_name}}}", var_val)

        step.content = content

        dummy_plan = ActionPlan(
            task_id="react",
            task_prompt=target,
            goal="",
            variables=dummy_vars,
        )
        # CLICK_XY: 브라우저 좌표 직접 클릭 (Vision 모드)
        if action == "CLICK_XY":
            # 스펙: target="x,y", content="설명". target/content 어느 쪽에 좌표가 와도 수용.
            x, y = None, None

            def _try_parse_xy(s):
                if not s:
                    return None, None
                s = str(s).strip()
                if "," not in s:
                    return None, None
                parts = [p.strip() for p in s.split(",")]
                if len(parts) < 2:
                    return None, None
                try:
                    return int(float(parts[0])), int(float(parts[1]))
                except (ValueError, TypeError):
                    return None, None

            # 1순위: target="x,y"
            x, y = _try_parse_xy(target)
            # 2순위: target="x", content="y"
            if x is None and target and str(target).strip().lstrip("-").isdigit():
                try:
                    _xc = int(str(target).strip())
                    _yc = int(str(content).strip()) if content and str(content).strip().lstrip("-").isdigit() else None
                    if _yc is not None:
                        x, y = _xc, _yc
                except ValueError:
                    pass
            # 3순위: content="x,y" (LLM이 뒤바꿔 보낸 경우)
            if x is None:
                x, y = _try_parse_xy(content)

            if x is None or y is None:
                print(f"  [CLICK_XY] 좌표 추출 실패 → 뷰포트 중앙 폴백. target={target!r}, content={content!r}")
                x, y = 600, 400  # 1200x800 뷰포트 중앙
            else:
                print(f"  [CLICK_XY] 파싱 성공: ({x}, {y}) ← target={target!r}")

            # [Fix] CLICK_XY verify 강화:
            # 1) JS 반환값 캡처 (NO_ELEMENT / CLICK_ERROR 감지)
            # 2) URL delta 측정 (페이지 전환 여부)
            # 3) el.click() 부재 시 dispatchEvent 폴백 (TypeError: el.click is not a function 대응)
            _url_before = self._get_current_browser_url()
            js = (
                "(function(){"
                f"var el=document.elementFromPoint({x},{y});"
                f"if(!el)return 'XY_NO_ELEMENT@{x},{y}';"
                "var tag=el.tagName||'?';"
                "try{"
                f"if(typeof el.click==='function'){{el.click();return 'XY_CLICKED:'+tag+'@{x},{y}';}}"
                "var ev=new MouseEvent('click',{bubbles:true,cancelable:true,view:window});"
                "el.dispatchEvent(ev);"
                f"return 'XY_DISPATCHED:'+tag+'@{x},{y}';"
                "}catch(e){"
                "var m=(e&&e.message)?String(e.message).slice(0,60):'unknown';"
                "return 'XY_CLICK_ERROR:'+tag+':'+m;"
                "}"
                "})();"
            )
            from execution_module import browser_run_script as _brs_xy
            _js_ret = await _brs_xy(script=js)
            import asyncio as _aio_xy
            await _aio_xy.sleep(2.0)
            _url_after = self._get_current_browser_url()
            _ret_str = str(_js_ret or "")
            _url_changed = bool(_url_before and _url_after and _url_before != _url_after)

            # 명시적 실패 — fail_streak 증가 대상
            if "XY_NO_ELEMENT" in _ret_str:
                return f"⚠️ CLICK_XY 실패: 좌표에 요소 없음 ({x}, {y})"
            if "XY_CLICK_ERROR" in _ret_str:
                return f"⚠️ CLICK_XY 실패: JS 오류 ({x}, {y}) — {_ret_str[:80]}"

            # 요소는 클릭됐음. URL 변화로 성공 수준 구분.
            if _url_changed:
                return f"✅ 좌표 클릭 성공 (URL 변경): ({x}, {y}) → {_url_after[:60]}"
            # URL 변화 없음 — 드롭다운/모달일 수 있으니 성공으로 리턴하되
            # 히스토리에 "화면 변화 없음" 문구를 남겨 프롬프트 규칙 13이 ASK_USER로 분기하게 함.
            return f"✅ 좌표 클릭 (화면 변화 없음): ({x}, {y}) {_ret_str[:60]}"

        _eff_to = self._compute_step_timeout(step)
        # [Phase 2] 결과를 단일 변수로 모아 finalize 에서 공용 사용.
        result: str = ""
        try:
            result = await asyncio.wait_for(
                self._execute_step(step, dummy_plan),
                timeout=_eff_to,
            )
            result = result or "완료"

            # WRITE 결과를 schema._react_vars에 저장 (다음 FILL에서 참조)
            if schema is not None and action == "WRITE" and "생성 완료" in result:
                # target이 비면 WRITE 핸들러가 write_{step_no}로 자동 명명하므로
                # 동일 규칙으로 복원해야 _on_react_done의 자동 txt 저장이 변수를 찾음.
                _effective_var = (target or "").strip() or f"write_{step.step_no}"
                var_val = dummy_plan.variables.get(_effective_var, "")
                if var_val:
                    schema._react_vars[_effective_var] = var_val
                    print(f"  [Act] WRITE 결과 저장: '{_effective_var}' ({len(var_val)}자)")
                # [Retry Drift Guard 2026-04-27] retry 중 lock-target WRITE 성공 마킹.
                # 이후 NAVIGATE/FILL/CLICK/GO_BACK 발사 차단 (action=DONE 강제).
                # Why: 최신로그.txt — Retry Step 8 WRITE 성공 직후 LLM 이 Step 9-10 에서
                # 검색창 FILL+CLICK 재발사 (이미 데이터 있는데 검색 흐름 재시작).
                # 토큰 낭비 + 자기 위치 망각. 회귀: lock 미설정/_is_retry_run=False 면
                # flag 자체를 안 박음 → drift guard 분기 미발동 → 기존 흐름 보존.
                try:
                    if (getattr(schema, "_is_retry_run", False)
                            and (getattr(schema, "_retry_target_lock", "") or "").strip()):
                        _lock_t_post = (schema._retry_target_lock or "").strip()
                        if _effective_var == _lock_t_post:
                            schema._retry_lock_write_done = True
                            print(
                                f"  🛡️ [Retry Drift Guard] '{_lock_t_post[:40]}' WRITE 성공 — "
                                f"후속 NAVIGATE/FILL/CLICK 차단 활성"
                            )
                except Exception as _e_rdg_post:
                    pass  # flag 세팅 실패 시 기존 흐름 그대로

            # READ_PAGE 결과도 동일하게 schema._react_vars에 저장
            # (이후 WRITE content의 {변수명} 치환에서 참조됨)
            if schema is not None and action == "READ_PAGE" and "페이지 읽기 완료" in result:
                _var = (target.strip() or f"page_content_{step.step_no}")
                _val = dummy_plan.variables.get(_var, "")
                if _val:
                    schema._react_vars[_var] = _val
                    # [Fix] URL → 변수명 매핑 기록 (다음 READ_PAGE 중복 차단용)
                    try:
                        _cur_u = (self._get_current_browser_url() or "").strip()
                        _uk = _cur_u.split("#", 1)[0].rstrip("/").lower()
                        if _uk and not hasattr(schema, "_react_var_urls"):
                            schema._react_var_urls = {}
                        if _uk:
                            schema._react_var_urls[_uk] = _var
                    except Exception:
                        pass
                    print(f"  [Act] READ_PAGE 결과 저장: '{_var}' ({len(_val)}자)")
                    # [2026-04-24] 차단/오류 페이지 감지 → LLM 에 alternative 힌트 주입.
                    # Why: NAVER 보안 확인, "페이지를 찾을 수 없습니다" 등을 정상 데이터로
                    # 취급하면 WRITE 가 "데이터 없음" 환각 1500자를 길게 작성하고 다음 액션
                    # 으로 alternative URL/플랫폼 시도 안 함. 본문에 차단 신호 매칭 시
                    # 변수에 마커 prefix 추가 + schema.how 머리에 alternative 힌트 주입.
                    try:
                        _blocked, _sig = _is_blocked_by_text(_val)
                        if _blocked:
                            _cur_u_blk = (self._get_current_browser_url() or "")[:120]
                            _blk_marker = (
                                f"[⚠️ BLOCKED_PAGE — 본문에 '{_sig}' 매칭. 정상 데이터 아님. "
                                f"이 변수는 분석 원천으로 쓰지 마라. URL: {_cur_u_blk}]\n\n"
                            )
                            # 변수 본문 머리에 마커 prefix — WRITE 분석가가 즉시 인지
                            schema._react_vars[_var] = _blk_marker + _val
                            # 차단 변수 추적 (WRITE prompt 강화 등에 사용 가능)
                            if not hasattr(schema, "_blocked_vars"):
                                schema._blocked_vars = []
                            if _var not in schema._blocked_vars:
                                schema._blocked_vars.append(_var)
                            print(
                                f"  ⚠️ [READ_PAGE BLOCK] 변수 '{_var}' 차단 페이지 감지 "
                                f"(signal='{_sig}'). LLM 에 alternative 힌트 주입."
                            )
                            self._report(
                                f"⚠️ **[차단 페이지 감지]** 변수 '{_var}' 가 보안/오류 페이지 "
                                f"('{_sig}' 매칭). 다음 step 은 alternative URL/플랫폼 시도 권장."
                            )
                            # alternative 힌트를 schema.how 머리에 주입.
                            # task_prompt 기반 대체 검색 URL 3종 + ASK_USER fallback.
                            try:
                                _tp = getattr(schema, "original_prompt", "") or getattr(schema, "what", "") or ""
                                _alts = _suggest_alternative_routes(_cur_u_blk, _tp)
                                _alt_lines = "\n".join(f"  - {a['name']}: {a['url']}" for a in _alts[:3])
                            except Exception:
                                _alt_lines = ""
                            _block_hint = (
                                f"⚠️ [차단 감지] 직전 READ_PAGE 변수 '{_var}' 가 보안/오류 페이지였다 "
                                f"('{_sig}' 본문 매칭). 이 변수를 원천으로 WRITE 하면 환각 보고서가 됨. "
                                f"다음 액션은 다음 중 하나:\n"
                                f"(a) 대체 URL 로 NAVIGATE 후 재 READ_PAGE — 후보:\n{_alt_lines}\n"
                                f"(b) 동일 정보를 다른 플랫폼/도메인 검색 (구글·다나와·크몽 등)\n"
                                f"(c) 정보 부족이 명확하면 ASK_USER 로 사용자에게 우회 경로 문의\n"
                                f"같은 차단 URL 로 재 NAVIGATE/READ_PAGE 반복 금지."
                            )
                            try:
                                if hasattr(schema, "how"):
                                    schema.how = _block_hint + "\n\n" + (schema.how or "")
                            except Exception:
                                pass
                    except Exception as _e_blk:
                        print(f"  ⚠️ [READ_PAGE BLOCK 감지 실패 (무시)]: {_e_blk}")
        except asyncio.TimeoutError:
            result = f"⚠️ 타임아웃: {int(_eff_to)}s 초과"
        except Exception as e:
            result = f"⚠️ 실행 오류: {str(e)[:80]}"

        # [Phase 2] Telegram 승인 메시지를 실행 결과로 finalize (⏳ 실행 중 → ✅/⚠️).
        await self._try_finalize_approval(step, result)
        return result

    # ── Verify ─────────────────────────────────────────────────────────────
    async def _verify_step(
        self,
        action: str,
        target: str,
        act_result: str,
        schema,
    ) -> tuple:
        """
        Act 결과를 즉각 검증.
        GPT 피드백 반영: Act → VERIFY → 다음 step

        반환: (verified: bool, message: str)
        """
        if not act_result:
            return False, "결과 없음"

        # [Fix 2026-04-22] READ_PAGE 성공 리턴 선행 면제.
        # "페이지 읽기 완료 → 변수 ..." 로 시작하면 본문에 DRIFT 주의문구가
        # 포함돼 있어도 성공으로 본다. (FAIL_PATTERNS 의 "⚠️"/"실패" 등이
        # 경고 문구에서 잘못 트립되는 것을 차단.)
        if action == "READ_PAGE" and act_result.lstrip().startswith("페이지 읽기 완료"):
            return True, act_result[:100]

        # 명시적 실패 패턴
        FAIL_PATTERNS = ["⚠️", "실패", "오류", "timeout", "타임아웃", "not found", "NONE_FOUND"]
        result_lower = act_result.lower()
        for pat in FAIL_PATTERNS:
            if pat.lower() in result_lower and "불확실" not in act_result:
                return False, act_result[:100]

        # NAVIGATE: URL 실제 변경 확인
        if action == "NAVIGATE":
            # [Fix] navigate_and_wait가 이미 loadFinished를 대기하므로 짧은 sleep만
            await asyncio.sleep(0.5)
            new_url = self._get_current_browser_url()
            if not new_url:
                # URL 조회 자체가 실패한 경우 — 이동은 성공으로 간주 (False positive 방지)
                return True, "이동 완료 (URL 조회 불가)"
            if target and "kmong.com" in target and "kmong.com" not in new_url:
                return False, f"URL 미변경: {new_url[:60]}"
            # 로그인 게이트 감지: /my-gigs → ?open=login_modal, /login, /signin 등 리다이렉트
            # "로그인 필요" 토큰은 ReAct 루프의 _LOGIN_SIGNALS가 감지해 ASK_USER로 전환함.
            _nu_l = new_url.lower()
            _tgt_l = (target or "").lower()
            _LOGIN_URL_PATTERNS = ("open=login_modal", "next_page=", "/login", "/signin", "/accounts/login")
            _tgt_no_auth = any(p in _tgt_l for p in _LOGIN_URL_PATTERNS)
            if (not _tgt_no_auth) and any(p in _nu_l for p in _LOGIN_URL_PATTERNS):
                return False, f"로그인 필요: {target[:60]} → {new_url[:80]}"
            return True, f"이동 완료: {new_url[:60]}"

        # FILL verify — 9-1, 9-3 수정
        # 기존: act_result 문자열만 보고 항상 True → 실제 DOM 미확인
        # 수정:
        #   1. FILL_SUCCESS:0 (길이 0) → 실패
        #   2. FILL_SUCCESS:N (N>0) → DOM 재확인 JS로 실제 반영 여부 확인
        #   3. void return / 실행됨 → 길이 확인 불가이므로 DOM 재확인
        if action == "FILL":
            if "NO_EDITOR" in act_result:
                return False, "에디터 없음"

            # 9-3: FILL_SUCCESS:N:TAG 패턴에서 길이 추출
            import re as _re_fill
            _fill_len_m = _re_fill.search(r'FILL_SUCCESS:(\d+):', act_result)
            if _fill_len_m:
                _fill_len = int(_fill_len_m.group(1))
                if _fill_len == 0:
                    # 길이 0 → 주입 실패 (9-3)
                    return False, f"FILL 주입 실패: 길이 0 (DOM에 내용 없음)"

            # 9-1+9-2: DOM 재확인 — 실제 에디터에 내용이 있는지 JS로 읽어서 검증
            # SPA(React/Vue)에서 innerHTML 주입 후 state와 분리되는 문제 감지
            try:
                from execution_module import execute_js_with_result as _exec_vfy
                _verify_js = """
(function() {
    var selectors = [
        '.ProseMirror', '[contenteditable="true"]', '.ql-editor',
        'div[data-placeholder]', 'textarea', '.note-editable',
        '.sun-editor-editable', '#tinymce',
    ];
    for (var i = 0; i < selectors.length; i++) {
        var ed = document.querySelector(selectors[i]);
        if (ed) {
            var content = ed.value || ed.innerText || ed.innerHTML || '';
            return 'DOM_CHECK:' + content.length + ':' + ed.tagName;
        }
    }
    return 'DOM_CHECK:NO_EDITOR';
})();"""
                _dom_result = await _exec_vfy(_verify_js, timeout=5.0)
                _dom_str = str(_dom_result or "")
                if "DOM_CHECK:" in _dom_str:
                    _dom_m = _re_fill.search(r'DOM_CHECK:(\d+):', _dom_str)
                    if _dom_m:
                        _dom_len = int(_dom_m.group(1))
                        if _dom_len == 0:
                            return False, f"FILL DOM 재확인 실패: 에디터 내용 0자 (SPA state 분리 의심)"
                        return True, f"FILL DOM 확인 완료: {_dom_len}자"
            except Exception as _vfy_e:
                print(f"  [Verify-FILL] DOM 재확인 실패 (무시): {_vfy_e}")

            # DOM 재확인 불가 시 — FILL_SUCCESS or void return은 허용
            if "FILL_SUCCESS" in act_result or "void return" in act_result or "실행됨" in act_result:
                return True, act_result[:60]
            return True, "FILL 실행됨 (DOM 확인 불가)"

        # CLICK: SUCCESS 또는 void return 허용
        if action == "CLICK":
            if "CLICK_SUCCESS" in act_result or "실행됨" in act_result or "SPA" in act_result:
                return True, act_result[:60]
            if "CLICK_FAILED" in act_result or "NONE_FOUND" in act_result:
                return False, act_result[:80]
            return True, "CLICK 실행됨"

        # CHECK_LOGIN
        if action == "CHECK_LOGIN":
            if "로그인 확인 완료" in act_result:
                return True, "로그인 됨"
            if "로그인" in act_result and "필요" in act_result:
                return False, "로그인 필요"
            return True, act_result[:60]

        # WRITE: 결과 문자열 길이 확인
        if action == "WRITE":
            if "생성 완료" in act_result:
                return True, act_result[:60]
            return len(act_result) > 10, act_result[:60]

        # WAIT: 항상 성공
        if action == "WAIT":
            return True, act_result[:40]

        # [T2 Step 2 2026-05-01] TELEGRAM_SEND verify — 핸들러 결과의 "📱 텔레그램 발송 완료"
        # prefix 로 성공 판정. "⚠️" prefix 면 위 FAIL_PATTERNS 가 이미 catch.
        if action == "TELEGRAM_SEND":
            if "📱 텔레그램 발송 완료" in act_result:
                return True, act_result[:60]
            return False, act_result[:60]

        # [P-ε 2026-05-01] EMAIL_SEND verify — "📧 이메일 발송 완료" prefix.
        if action == "EMAIL_SEND":
            if "📧 이메일 발송 완료" in act_result:
                return True, act_result[:60]
            return False, act_result[:60]

        # 기타: 실패 패턴 없으면 성공
        return True, act_result[:60]

    # ── WRITE 잘림 감지 + 이어쓰기 헬퍼 (Fix 2026-04-21 P1) ───────────────
    @staticmethod
    def _looks_truncated(text: str) -> bool:
        """LLM 응답이 max_tokens 한도로 잘린 것처럼 보이는지.

        조건: 비어있지 않고, 500자 이상, 마지막 글자가 종결 부호 아님.
        종결 부호: . ! ? ) ] } 다(한글 문장 종결) 요 음 임 함 . … ” ' »
        """
        if not text:
            return False
        _t = text.rstrip()
        if len(_t) < 500:
            return False
        _last = _t[-1]
        _CLOSERS = set(".!?)]}’”\"'»…。！？︒\n")
        if _last in _CLOSERS:
            return False
        # 한국어 종결어미 끝: '~다'·'~요'·'~함'·'~임'·'~음'·'~죠'·'~네'·'~네요'
        _last2 = _t[-2:]
        _last3 = _t[-3:]
        _KOR_ENDS = ("다.", "요.", "함.", "임.", "음.", "죠.", "네.", "네요.", "어요.", "아요.")
        if any(_t.endswith(e) for e in _KOR_ENDS):
            return False
        # 종결 부호 없이 끝났으면 절단 가능성 높음.
        return True

    async def _maybe_continue_truncated_write(
        self,
        generated: str,
        original_prompt: str,
        available_vars_block: str,
        user_content: str,
        max_tokens: int = 16384,
        _depth: int = 0,
        _max_depth: int = 2,
    ) -> str:
        """WRITE 응답이 잘렸으면 이어쓰기 호출 (최대 _max_depth회) 후 이어붙여 반환.

        실패/이어쓰기 거부/에러 시 원본 그대로 반환.

        [2026-04-24] depth 도입 — 이전엔 1회만 호출해 이어쓰기 결과가 또 절단되면
        무방비 상태로 다음 단계로 넘어갔다. 이제는 자체 재귀로 최대 2회 시도하고
        한도 도달 시 그대로 반환 (무한 루프 방지).
        """
        try:
            if not self._looks_truncated(generated):
                return generated
            if _depth >= _max_depth:
                print(
                    f"  ⚠️ [WRITE 이어쓰기] depth={_depth} 한도 도달 — 추가 이어쓰기 중단 "
                    f"(현재 {len(generated)}자)"
                )
                return generated
            # 에러 문자열은 이어쓰기 대상 아님
            if generated.startswith(("[서버 ", "[LLM ", "[크레딧", "[Pro ")):
                return generated
            print(
                f"  ✂️ [WRITE 절단 감지] 마지막 16자='{generated[-16:]}' "
                f"({len(generated)}자) → 이어쓰기 호출 (depth={_depth + 1}/{_max_depth})"
            )
            _tail = generated[-1200:]
            _cont_prompt = (
                (available_vars_block or "")
                + "[이어쓰기 지시]\n"
                "직전 응답이 max_tokens 한도로 문장 중간에서 잘렸다. 아래 이어지는 "
                "마지막 단락을 **이어서만** 작성하라. 처음부터 다시 쓰지 말고, 이미 "
                "끝난 문장은 다시 쓰지 말고, **잘린 지점 바로 다음 문자부터** 이어 써라. "
                "머리말·인사·제목 금지. 최종 문서 완결까지 모든 남은 내용을 한 번에 출력.\n\n"
                f"[원래 지시 — 참고]\n{user_content[:1500]}\n\n"
                f"[직전 응답 말미 1200자]\n```\n{_tail}\n```\n\n"
                "이어지는 본문:\n"
            )
            _cont = await get_llm_response_async(
                _cont_prompt,
                system_prompt=EIDOS_ANALYST_SYSTEM_PROMPT,
                timeout=120,
                max_tokens=max_tokens,
            )
            _cont = (_cont or "").strip()
            if not _cont or _cont.startswith(("[서버 ", "[LLM ", "[크레딧", "[Pro ")):
                print(f"  ⚠️ [WRITE 이어쓰기] 실패 — 원본 유지 ({len(_cont)}자)")
                return generated
            # 중복 겹침 제거: _cont 앞 200자가 _tail 끝과 겹치면 자른다.
            _ov = 0
            for _n in range(min(200, len(_cont)), 20, -10):
                if _tail.endswith(_cont[:_n]):
                    _ov = _n
                    break
            _appended = _cont[_ov:] if _ov else _cont
            if len(_appended) < 50:
                print(
                    f"  ⚠️ [WRITE 이어쓰기] 추가분 {len(_appended)}자뿐 — 무의미. 원본 유지"
                )
                return generated
            print(
                f"  ✅ [WRITE 이어쓰기] +{len(_appended)}자 병합 (중복 제거 {_ov}자, "
                f"depth={_depth + 1}/{_max_depth})"
            )
            merged = generated + _appended
            # 재귀: 이어쓴 결과도 또 truncated 면 한 번 더 시도.
            return await self._maybe_continue_truncated_write(
                merged,
                original_prompt,
                available_vars_block,
                user_content,
                max_tokens=max_tokens,
                _depth=_depth + 1,
                _max_depth=_max_depth,
            )
        except Exception as _e:
            print(f"  ⚠️ [WRITE 이어쓰기] 예외 (원본 유지): {_e}")
            return generated

    # ── DirectTask 결과 디스크 저장 (Fix 2026-04-21 P1) ──────────────────
    def _persist_direct_task_result(self, task_prompt: str, schema, result: str) -> None:
        """_run_direct_llm_task 의 결과를 eidos_files/ 에 저장.

        파일명 우선순위:
        1) schema.chain_id 가 있으면 {chain_short}_{stage}.md
        2) task_prompt 에서 `'XXX.md'`/`"XXX.md"`/`XXX.md` 추출
        3) task_type + timestamp
        본문이 200자 미만이면 스킵 (쓸모없음).
        """
        if not result or len(result.strip()) < 200:
            return
        try:
            from execution_module import SAFE_BASE_PATH as _SBASE
            import os as _os, re as _re_p, time as _time_p
            # 파일명 결정
            _fn = ""
            _tt = (getattr(schema, "task_type", "general") or "general").lower()
            _cid = (getattr(schema, "chain_id", "") or "")[:8]
            _stage = (getattr(schema, "stage", "") or "").lower()
            # 2) task_prompt 에서 md 파일명 추출
            _m = _re_p.search(r"([A-Za-z0-9_\-가-힣]{2,60}\.(?:md|txt))", task_prompt or "")
            if _m:
                _fn = _m.group(1)
            # 1) chain stage 기반
            if not _fn and _cid and _stage:
                _fn = f"{_stage}_{_cid}.md"
            # 3) fallback
            if not _fn:
                _fn = f"{_tt}_{int(_time_p.time())}.md"
            # sanitize
            _fn = _re_p.sub(r"[^\w\-\.가-힣]+", "_", _fn).strip("_") or f"direct_{int(_time_p.time())}.md"
            _rel = _fn.replace("\\", "/").lstrip("./")
            _abs = _os.path.abspath(_os.path.join(_SBASE, _rel))
            _base_abs = _os.path.abspath(_SBASE)
            if not (_abs == _base_abs or _abs.startswith(_base_abs + _os.sep)):
                print(f"  ⚠️ [DirectTask] 경로 이탈 차단: {_fn}")
                return
            _os.makedirs(_os.path.dirname(_abs) or _base_abs, exist_ok=True)
            with open(_abs, "w", encoding="utf-8") as _fh:
                _fh.write(result)
            print(f"  💾 [DirectTask] 디스크 저장: {_abs} ({len(result)}자)")
            # schema._react_vars 에도 추가해 downstream stage 가 upstream 으로 인식
            if schema is not None:
                try:
                    if not hasattr(schema, "_react_vars") or schema._react_vars is None:
                        schema._react_vars = {}
                    schema._react_vars[_fn] = result
                except Exception:
                    pass
        except Exception as _e:
            print(f"  ⚠️ [DirectTask 저장] 예외: {_e}")

    # ── HOW_MUCH 본문 기반 자동 체크 헬퍼 (Fix 2026-04-21) ────────────────
    def _latest_write_body(self, schema, action: str, plan=None) -> str:
        """직전 WRITE 가 만든 산출물의 실제 본문 텍스트.

        plan.variables 우선(있을 때), schema._react_vars 폴백 — 가장 긴
        분석 본문(원천 page_*/content_* 변수 제외)을 골라 1차 검사.
        실패해도 _auto_check_attributes 흐름에 영향 없음.
        """
        if action != "WRITE":
            return ""
        try:
            _vars: Dict[str, Any] = {}
            if plan is not None:
                _vars = dict(getattr(plan, "variables", {}) or {})
            _rv = getattr(schema, "_react_vars", {}) or {}
            for _k, _v in _rv.items():
                if _k not in _vars:
                    _vars[_k] = _v
            if not _vars:
                return ""
            # 원천(page_/_page/_content/_html) 변수는 분석 본문이 아니므로 제외
            _candidates = []
            for _k, _v in _vars.items():
                if not isinstance(_k, str) or not isinstance(_v, str):
                    continue
                _kl = _k.lower()
                if (_kl.startswith("page_") or _kl.endswith("_page")
                        or _kl.endswith("_content") or "page_content" in _kl
                        or _kl.endswith("_html")):
                    continue
                if len(_v) < 100:
                    continue
                _candidates.append((_k, _v))
            if not _candidates:
                return ""
            # 가장 긴 산출물 1개를 본문으로
            _candidates.sort(key=lambda kv: len(kv[1]), reverse=True)
            return _candidates[0][1]
        except Exception:
            return ""

    @staticmethod
    def _extract_quoted_terms(name: str) -> list:
        """attr.name 안의 따옴표/괄호로 묶인 키워드 추출.

        예: 고정 주제 'CRM 팩' 언급 여부 → ['CRM 팩']
            "X" 포함 여부 → ['X']
            (브랜드명) 명시 → ['브랜드명']
        """
        import re as _re_q
        _patterns = (
            r"'([^']{1,80})'",
            r"\"([^\"]{1,80})\"",
            r"「([^」]{1,80})」",
            r"『([^』]{1,80})』",
            r"\(([^)]{2,80})\)",
            r"\[([^\]]{2,80})\]",
        )
        _out = []
        for _p in _patterns:
            for _m in _re_q.findall(_p, name or ""):
                _t = _m.strip()
                if _t and _t not in _out:
                    _out.append(_t)
        return _out

    def _auto_check_keyword_attr(self, attr, body: str) -> bool:
        """'주제/언급/포함/명시 + 따옴표 키워드' 형 속성 자동 달성.

        attr.name 에 따옴표 키워드가 있고 동사가 언급/포함/명시/표기/표현/적시 면,
        본문(body) 에 그 키워드가 1회 이상 등장하는지로 판정. 미매칭이면 False.
        """
        if not body:
            return False
        _name = attr.name or ""
        _name_l = _name.lower()
        _VERB_HINTS = ("언급", "포함", "명시", "표기", "표현", "적시", "기재", "들어가")
        if not any(_v in _name for _v in _VERB_HINTS):
            return False
        _terms = self._extract_quoted_terms(_name)
        if not _terms:
            return False
        _body_l = body.lower()
        _matched = [_t for _t in _terms if _t.lower() in _body_l]
        if not _matched:
            return False
        attr.achieved = True
        attr.evidence = f"본문 키워드 매칭: {', '.join(_matched)[:60]}"
        print(f"  ✅ [AutoCheck-KW] '{_name}' → 본문에 {_matched} 등장")
        return True

    def _auto_check_count_attr(self, attr, body: str) -> bool:
        """'후보 N개 포함', 'N건', 'N개 이상' 등 개수 조건 자동 달성.

        attr.name 에서 (후보|항목|아이템|상품|N) 개 패턴을 추출, 본문의 불릿/번호/줄
        리스트가 N 이상이면 달성.
        """
        if not body:
            return False
        import re as _re_c
        _name = attr.name or ""
        _m = _re_c.search(r"(\d+)\s*(?:개|건|가지|개\s*이상)", _name)
        if not _m:
            return False
        _need = int(_m.group(1))
        if _need <= 0:
            return False
        # 불릿/번호/대시 리스트 항목 수 카운트
        _bullets = _re_c.findall(r"^[\s>]*[-*•·▶▷●○]\s+\S", body, _re_c.MULTILINE)
        _numbered = _re_c.findall(r"^[\s>]*\d{1,3}[\.\)]\s+\S", body, _re_c.MULTILINE)
        _count = len(_bullets) + len(_numbered)
        if _count < _need:
            return False
        attr.achieved = True
        attr.evidence = f"본문 리스트 {_count}개 (요구 {_need}개)"
        print(f"  ✅ [AutoCheck-Count] '{_name}' → 본문 항목 {_count}/{_need}")
        return True

    def _auto_check_existence_attr(self, attr, body: str, result_lower: str) -> bool:
        """'X 파일 존재 여부' / 'X 작성 완료 확인' 형 속성 자동 달성.

        attr.name 에 '존재 여부' 또는 '완료 확인' 패턴이 있고, 본문이 충분(>=300자) +
        WRITE 결과가 '생성 완료' 면 달성으로 본다. 파일명이 명시돼 있으면 그
        파일에 대한 WRITE 였는지도 확인(기존 라인 5046-5064 와 동일 정신).
        """
        if not body:
            return False
        _name = attr.name or ""
        # [Fix P0 2026-04-22] EXIST_HINTS 확장 — '획득/포함/확보/전달 + 여부/완료'
        # 형태 추가. Why: LLM 이 생성한 browser_read 속성명(예: "서울 날씨 정보
        # 획득 여부", "오늘 날짜 정보 포함 여부", "날씨 정보 전달 완료")이 이전
        # 패턴셋으로는 매칭 안 돼 영구 미달성 → Verifier "missing_attr" 루프 →
        # parser_error 로 ASK_USER 까지 빠지던 회귀(2026-04-22 '서울 날씨' 로그).
        # How to apply: body>=300자 + result 에 "생성 완료" 동시 조건 유지 —
        # WRITE 가 실제로 성공한 직후에만 달성 처리되어 FP 위험 낮음.
        _EXIST_HINTS = (
            "존재 여부", "존재여부", "작성 완료", "작성완료",
            "생성 완료", "생성완료", "완료 확인", "완료확인", "존재 확인",
            "획득 여부", "획득여부", "획득 완료", "획득완료",
            "포함 여부", "포함여부", "확보 여부", "확보여부",
            "확보 완료", "확보완료", "전달 완료", "전달완료",
        )
        if not any(_h in _name for _h in _EXIST_HINTS):
            return False
        if len(body) < 300:
            return False
        if "생성 완료" not in (result_lower or ""):
            return False
        attr.achieved = True
        attr.evidence = f"본문 {len(body)}자 + WRITE 생성 완료"
        print(f"  ✅ [AutoCheck-Exist] '{_name}' → 본문 {len(body)}자 + 생성 완료")
        return True

    # ── HOW_MUCH 속성 자동 체크 ────────────────────────────────────────────
    def _auto_check_attributes(self, schema, action: str, result: str, obs: Dict, plan=None) -> None:
        """Act + Verify 결과를 바탕으로 HOW_MUCH 속성 자동 달성 표시.

        [오염 수정 2026-04-18] 이전에는 "완료" 한 토큰만 포함해도 속성이
        달성 처리되어, CLICK no-op("CLICK 실행됨")·NAVIGATE 실패 URL 잔존 등
        약한 신호로 HOW_MUCH 4/4가 허위 달성되는 False Positive가 잦았다.
        각 속성 유형별로 **강한 성공 토큰**만 인정하도록 교체한다.

        [Fix 2026-04-21] '고정 주제 X 언급', '후보 N개', '존재' 류 속성을
        WRITE 본문 실내용 기준으로도 검증. 이전엔 이름에 'faq/생성/작성/문서'
        키워드가 없으면 영구 미달성 → Verifier LLM 의존 → parser_error 루프.
        """
        result_lower = result.lower() if result else ""
        url = obs.get("url", "")

        # WRITE 직후 가장 최근 파일 본문(실제 디스크/메모리 내용) 한 번 확보 —
        # 키워드/후보-N개 검사가 result_lower(요약)이 아닌 본문을 봐야 정확.
        _last_write_body = self._latest_write_body(schema, action, plan=plan) if action == "WRITE" else ""

        for attr in schema.attributes:
            if attr.achieved:
                continue
            name_lower = attr.name.lower()

            # ── [P0 Fix] 본문 키워드 명시 속성 ('주제 X 언급', "X 포함 여부" 등) ──
            if action == "WRITE" and _last_write_body:
                if self._auto_check_keyword_attr(attr, _last_write_body):
                    continue
                if self._auto_check_count_attr(attr, _last_write_body):
                    continue
                if self._auto_check_existence_attr(attr, _last_write_body, result_lower):
                    continue

            # FAQ / 텍스트 생성 완료 감지 — WRITE는 "생성 완료" 명시 토큰만 인정
            # [Fix 2026-04-21] '기획'·'제안'·'안서'·'draft'·'proposal'·'리포트'·
            # 'report' 도 WRITE 산출물 명칭으로 인정. 이전엔 '기획안 작성 완료'만
            # 우연히 '작성' 키워드로 잡혔고, '기획안 파일 존재 여부'는 매칭 실패.
            _WRITE_NAME_KWS = (
                "faq", "생성", "작성", "문서",
                "기획", "제안", "안서", "draft", "proposal",
                "리포트", "report", "보고서", "초안",
            )
            if any(k in name_lower for k in _WRITE_NAME_KWS) and action == "WRITE":
                if "생성 완료" in result_lower:
                    # [Fix 2026-04-20] 속성 이름에 구체적 파일명(예: "spec.md")
                    # 이 명시돼 있으면 이번 WRITE 결과가 그 파일을 실제로 생성했는지
                    # 확인한다. 그렇지 않으면 description.md WRITE 하나로 spec.md /
                    # assets_manifest.md 까지 일괄 달성 처리되는 부분달성-PASS 가 난다.
                    import re as _re_fn, os as _os_fn
                    _fns = _re_fn.findall(r"[A-Za-z0-9_\-]+\.(?:md|txt|json|ya?ml|py|html|css|js)",
                                          attr.name)
                    _fn_match = True
                    if _fns:
                        _fn_match = False
                        for _fn in _fns:
                            _fn_l = _fn.lower()
                            if _fn_l in result_lower:
                                _fn_match = True
                                break
                            # 디스크 저장 실체 확인 — SAFE_BASE_PATH 기반
                            try:
                                _vars = getattr(schema, "_react_vars", {}) or {}
                                if any(str(k).lower().endswith(_fn_l) or str(k).lower() == _fn_l
                                       for k in _vars.keys()):
                                    _fn_match = True
                                    break
                            except Exception:
                                pass
                    if _fn_match:
                        attr.achieved = True
                        attr.evidence = f"WRITE 완료: {result[:40]}"

            # browser_read: 대상 페이지 로드 완료 — NAVIGATE 성공 토큰 + URL 유효
            elif ("로드" in name_lower and "페이지" in name_lower
                  and action == "NAVIGATE"):
                _nav_ok = ("이동 완료" in result or "페이지 로드 완료" in result_lower
                           or "navigate 성공" in result_lower)
                _nav_soft = "이동 완료(경고)" in result
                if _nav_ok and url and not _nav_soft:
                    attr.achieved = True
                    attr.evidence = f"NAVIGATE 완료: {url[:60]}"

            # browser_read: 요청된 내용 추출·요약 완료 — READ_PAGE 후 WRITE 결과 확보
            elif (("추출" in name_lower or "요약" in name_lower)
                  and action == "WRITE"):
                _vars = getattr(schema, "_react_vars", {}) or {}
                _has_page_content = any(
                    (k.startswith("page_") or k.endswith("_page")
                     or k.endswith("_content") or "content" in k.lower())
                    and len(str(v) or "") > 200
                    for k, v in _vars.items()
                )
                _write_ok = "생성 완료" in result_lower  # "완료" 단독 금지
                if _has_page_content and _write_ok:
                    attr.achieved = True
                    attr.evidence = f"READ→WRITE 완료: {result[:40]}"

            # 에디터 진입 감지 — "편집 모드 진입" / "editor 로드" 명시만 인정
            elif "에디터" in name_lower and action in {"CLICK", "NAVIGATE"}:
                _editor_strong = ("편집 모드" in result_lower or "에디터 진입" in result_lower
                                  or "editor loaded" in result_lower or "edit mode" in result_lower)
                if _editor_strong:
                    attr.achieved = True
                    attr.evidence = f"에디터 진입: {result[:40]}"

            # 입력/주입 완료 감지 — FILL_SUCCESS 또는 "에디터 주입 완료" 명시만 인정
            # void return("FILL 실행됨 (SPA void return)")이나 CLICK의 "실행됨"은 거부.
            elif (("입력" in name_lower or "주입" in name_lower or "fill" in name_lower)
                  and action == "FILL"):
                _fill_strong = ("FILL_SUCCESS" in result or "에디터 주입 완료" in result
                                or "DOM 확인 완료" in result)
                _fill_fail = ("FILL 실패" in result or "FILL_FAILED" in result
                              or "NO_EDITOR" in result)
                if _fill_strong and not _fill_fail:
                    attr.achieved = True
                    attr.evidence = f"FILL 완료: {result[:40]}"

            # 저장 완료 감지 — 8-2 수정: 클릭 + 명시적 성공 패턴 동시 요구
            # "클릭 실행됨 (SPA void return)" 같은 void 결과는 저장 성공이 아님.
            # CLICK_SUCCESS + 저장/save 키워드가 함께 있어야만 달성 처리.
            # [Fix] "void return" 은 SPA CLICK의 정상 반환 — 저장 실패 신호에서 제외.
            # 일반 CLICK(상품/링크/카드)에도 매 번 "저장 실패 감지" 오탐지되던 문제 수정.
            elif "저장" in name_lower and action == "CLICK":
                _save_success_signals = ["CLICK_SUCCESS", "저장됨", "saved", "저장 완료", "등록 완료"]
                _save_fail_signals    = ["CLICK_FAILED", "NONE_FOUND", "저장 실패", "save failed"]
                _has_success = any(s in result for s in _save_success_signals)
                _has_fail    = any(s.lower() in result_lower for s in _save_fail_signals)
                if _has_success and not _has_fail:
                    attr.achieved = True
                    attr.evidence = f"저장 확인: {result[:40]}"
                elif _has_fail:
                    # 명시적 실패 — 달성하지 않음 (기존 achieved 유지)
                    print(f"  ⚠️ [AutoCheck] 저장 실패 감지: {result[:60]}")

            # 페이지 반영 확인 — kmong 도메인 + 명시적 반영 토큰 요구
            elif "반영" in name_lower and action in {"NAVIGATE", "CLICK"}:
                _reflect_strong = ("반영 완료" in result_lower or "업데이트 완료" in result_lower
                                   or "수정 완료" in result_lower)
                if "kmong.com" in url and _reflect_strong:
                    attr.achieved = True
                    attr.evidence = f"반영 확인: {url[:40]}"

            # 로그인 감지 — CHECK_LOGIN "로그인 확인 완료" 명시 토큰만 인정
            elif "로그인" in name_lower and action == "CHECK_LOGIN":
                if "로그인 확인 완료" in result or "로그인 됨" in result:
                    attr.achieved = True
                    attr.evidence = "로그인 확인"

            # [T2 MVP B-α 2026-05-01] TELEGRAM_SEND verb 발현 시 schema 의 "텔레그램/전송/
            # 발송/메시지" 류 속성 자동 인정. 3차 dogfooding 의 redundant SEND (Retry Step 7
            # 한 번 더 발송) 차단 — LLM 이 schema attributes 미달성으로 인식하고 재시도하는
            # 흐름 차단. result 에 "📱 텔레그램 발송 완료" prefix 가 있어야 인정 (성공 시그널).
            elif (action == "TELEGRAM_SEND"
                  and any(k in name_lower for k in (
                      "텔레그램", "telegram", "전송", "발송", "메시지", "send",
                  ))):
                if "📱 텔레그램 발송 완료" in result:
                    attr.achieved = True
                    attr.evidence = f"TELEGRAM_SEND 완료: {result[:40]}"

            # [P-ε 2026-05-01] EMAIL_SEND verb 발현 시 schema 의 "이메일/전송/발송/메일/
            # email" 류 속성 자동 인정. TELEGRAM_SEND 와 동일 패턴.
            elif (action == "EMAIL_SEND"
                  and any(k in name_lower for k in (
                      "이메일", "메일", "email", "전송", "발송", "send",
                  ))):
                if "📧 이메일 발송 완료" in result:
                    attr.achieved = True
                    attr.evidence = f"EMAIL_SEND 완료: {result[:40]}"

            # [Fix B 2026-05-06] HOW_MUCH 자동 진전 — 객관 카운터 기반 fallback.
            # Why: 최신로그.txt 9:13 "🛑 막힘 감지: 20 step 동안 속성 진전 없음".
            # 실제로는 READ_PAGE 4번 도달(naver_place_detail_1 / cafe_taschen_detail /
            # grasshopper_detail / gangnam_restaurants_list_content) 했는데도 카운터
            # 0 그대로 → LLM 이 "현재 2개" 직접 추정만 반복하며 30+ step 더 모음.
            # 위 elif 체인이 모두 미매칭 + name 안에 N 숫자 + (탐색|수집|페이지|곳|개)
            # 키워드 있을 때 _react_vars 의 page-content var 수로 자동 진전.
            if not attr.achieved:
                try:
                    import re as _re_fb
                    _m_fb = _re_fb.search(r"(\d+)", attr.name)
                    if _m_fb:
                        _n_fb = int(_m_fb.group(1))
                        if 1 <= _n_fb <= 50:
                            _COUNT_KWS = (
                                "탐색", "수집", "페이지", "상세", "조사",
                                "곳", "개수", "리스트", "list", "추천",
                                "건", "개", "명", "맛집", "후보", "결과",
                            )
                            if any(kw in name_lower for kw in _COUNT_KWS):
                                _vars_fb = getattr(schema, "_react_vars", {}) or {}
                                _page_var_count = sum(
                                    1 for k, v in _vars_fb.items()
                                    if isinstance(v, str) and len(v) > 200
                                    and (
                                        str(k).startswith("page_")
                                        or str(k).endswith("_page")
                                        or "_detail" in str(k)
                                        or "_content" in str(k)
                                        or str(k).endswith("_list")
                                        or str(k).endswith("_results")
                                        or str(k).endswith("_info")
                                    )
                                )
                                if _page_var_count >= _n_fb:
                                    attr.achieved = True
                                    attr.evidence = (
                                        f"객관 카운터 자동 진전: page-content "
                                        f"{_page_var_count}/{_n_fb}"
                                    )
                                    print(
                                        f"  🎯 [Fix B AutoProgress] '{attr.name[:30]}' "
                                        f"자동 달성 — page-content {_page_var_count} ≥ {_n_fb}"
                                    )
                except Exception as _e_fb:
                    print(f"  [Fix B AutoProgress] 검사 실패 (무시): {_e_fb}")

    # ── 사용자 대기 + Telegram 알림 + chain auto-advance 헬퍼 ─────────────
    # Why: 체인 stage 검증 실패 등으로 WAITING_USER 전환할 때 기존엔 GUI
    # _report 만 남겼다. 사용자가 집 밖이거나 TTS 를 못 들으면 진행이 무한 대기.
    # 1) 동일 메시지를 Telegram 으로도 best-effort 발송.
    # 2) 체인 소속 task 이면 N 초 후 자동으로 `_force_advance_chain=True + run_now()`
    #    를 돌려 다음 stage 로 넘어간다.
    # ── [협력 게이트 다층화 B1 2026-04-29] Tier 분류 헬퍼 ──────────────
    # _pause_with_telegram_gate 진입 시점의 신호를 4 Tier 로 매핑.
    # 분류만 분리 — 라우팅은 B2 단계. 동작 변경 0.
    #
    # Tier 분류 (우선순위 순):
    #   tier4_risk    위험 (HIGH risk OR MONEY/LEGAL/DELIVERY 카테고리)
    #   tier3_intent  의도 misalignment (schema._force_user_intervention)
    #   tier2_info    정보 부족 (task fail_reason 또는 headline 토큰)
    #   tier1_safe    안전 + 자율 scope 안 (auto-pass)
    _INFO_TOKENS = (
        "missing_required_attr", "missing_attr",
        "schema 필수", "정보 부족", "정보부족", "필수 정보", "missing required",
    )

    def _classify_pause_tier(
        self,
        task: Dict,
        schema,
        headline: str = "",
        ask_user: str = "",
    ) -> str:
        """진입 시점 신호로 4 Tier 분류 (read-only). 동작 변경 X."""
        # ── Tier 4: 위험 (HIGH OR 위험 카테고리) ──────────────────
        try:
            from eidos_approval_gateway import (
                ApprovalGateway, RiskLevel, ActionCategory,
            )
            _action = (task.get("action") or "") if isinstance(task, dict) else ""
            _target = (task.get("target") or "") if isinstance(task, dict) else ""
            _content = (task.get("content") or "") if isinstance(task, dict) else ""
            _classifier = ApprovalGateway.get().classifier
            _level = _classifier.classify(_action, _target, _content)
            _category = _classifier.classify_category(_action, _target, _content)
            if (
                _level == RiskLevel.HIGH
                or _category in (
                    ActionCategory.MONEY, ActionCategory.LEGAL, ActionCategory.DELIVERY,
                )
            ):
                return "tier4_risk"
        except Exception as _e_t4:
            print(f"  ⚠️ [Tier 분류] tier4 검사 실패 (무시): {_e_t4}")

        # ── Tier 3: 의도 misalignment (Verifier 누적 cap 결과) ──
        try:
            if bool(getattr(schema, "_force_user_intervention", False)):
                return "tier3_intent"
        except Exception:
            pass

        # ── [B 2026-05-01] Tier 3 보강: chain_health < 0.3 시 강제 tier3 ──
        # confidence axis (Verifier/MCTS/RealExec 3채널 가중평균) 가 낮으면 자율
        # auto-pass 차단 — health=0.3 미만은 "이미 신뢰도 ↓ 상태에서 또 통과시키면
        # 사용자 개입 없이 거짓 완료 위험" 신호. _force_user_intervention 마킹 안 됐어도
        # 선제 차단. Verifier cap v2 와 별도 안전망.
        try:
            from eidos_confidence import get_chain_health as _gch_b
            _h_b = _gch_b(schema)
            if _h_b is not None and float(_h_b) < 0.3:
                print(
                    f"  🛡️ [B 2026-05-01] chain_health={_h_b:.2f} < 0.3 → tier3_intent 강제 "
                    f"(auto-pass 차단)"
                )
                return "tier3_intent"
        except Exception as _e_b:
            # confidence 모듈 없거나 실패 — 안전 폴백 (기존 tier 분류 흐름 유지).
            pass

        # ── Tier 2: 정보 부족 (task fail_reason OR headline 토큰) ──
        try:
            _fail_reason = ""
            if isinstance(task, dict):
                _fail_reason = str(task.get("fail_reason", "") or "").lower()
            _headline_lc = (headline or "").lower()
            for _tok in self._INFO_TOKENS:
                if _tok.lower() in _fail_reason or _tok.lower() in _headline_lc:
                    return "tier2_info"
        except Exception:
            pass

        # ── Tier 1: 기본 — 자율 scope 안이면 auto-pass 가능, 아니면 일반 게이트 ──
        return "tier1_safe"

    def _pause_with_telegram_gate(
        self,
        task: Dict,
        schema,
        *,
        headline: str,
        ask_user: str,
        telegram_title: str,
        telegram_body: str,
        auto_advance_sec: int = 300,
    ) -> None:
        # [Autonomous 2026-04-27] 자율 실행 모드 active + 현재 task 가 그 스코프
        # 안에 있으면 게이트 자동 통과. 사용자 사전 승인으로 "이번 실행 동안은
        # 모든 ASK_USER 자동 진행" 약속한 상태이므로 here 에서 멈추지 않음.
        # 단, abort/완료 게이트 (chain_aborted 등) 는 이 분기 전에 처리됐으므로
        # 여기에 도달하는 건 "정보 부족 → 사용자 확인 요청" 레벨임 — auto-pass 안전.
        # [Patch 2026-04-29] 단, schema._force_user_intervention 마킹 시 auto-pass
        # 차단. Verifier drift/missing_attr 누적 등 의도 misalignment 신호일 때
        # 자동 진행은 같은 실수 반복 → 토큰/시간 낭비. 사용자 검토 필요.
        try:
            _force_user = bool(getattr(schema, "_force_user_intervention", False))
        except Exception:
            _force_user = False
        if _force_user:
            _reason_fu = ""
            try:
                _reason_fu = str(getattr(schema, "_force_user_reason", "")) or ""
            except Exception:
                _reason_fu = ""
            print(
                f"  🛡️ [Autonomous] auto-pass 차단 — schema._force_user_intervention "
                f"({_reason_fu or 'verifier_fail'}) → 정상 ASK_USER 진입"
            )

        # ── [B2 2026-04-29] Tier 라우팅 ────────────────────────────
        # B1 의 _classify_pause_tier 결과로 auto-pass 진입 결정.
        # Tier 1 (안전) — auto-pass 가능 / Tier 2,3,4 — auto-pass 차단.
        try:
            _tier = self._classify_pause_tier(task, schema, headline, ask_user)
        except Exception as _e_tier:
            _tier = "tier1_safe"
            print(f"  ⚠️ [Tier 분류] 실패 (tier1 폴백): {_e_tier}")
        _block_auto_pass = (_tier != "tier1_safe")
        # tier4/tier2 진입 시 별도 콘솔 마커 (force_user 마커와 별개)
        if not _force_user and _tier != "tier1_safe":
            _tier_emoji = {"tier4_risk": "🚨", "tier3_intent": "🛡️", "tier2_info": "📋"}.get(_tier, "❓")
            print(
                f"  {_tier_emoji} [Tier 라우팅] {_tier} → auto-pass 차단 → 일반 게이트 진입"
            )

        try:
            from eidos_autonomous_runner import AutonomousRunManager as _ARM
            _auto_mgr = _ARM.get()
            if (not _block_auto_pass) and _auto_mgr.is_task_in_active_scope(task, schema):
                _run = _auto_mgr.get_active_run()
                _stage = task.get("stage", "?")
                _label = (
                    f"chain_id={(getattr(schema, 'chain_id', '') or task.get('chain_id', ''))[:8]}"
                    if (_run and _run.scope_type == "single_chain")
                    else f"time_window expires_at={int(_run.expires_at) if _run else 0}"
                )
                print(
                    f"  🟢 [Tier 1 — 안전] PauseGate auto-pass — run={(_run.run_id if _run else '?')[:8]} "
                    f"scope={(_run.scope_type if _run else '?')} {_label} stage='{_stage}'"
                )
                self._report(
                    f"🟢 **[Tier 1 — 안전 / 자율 게이트 통과]** "
                    f"사전 승인 스코프 안의 작업이라 ASK_USER 없이 진행."
                )
                # chain task 면 force advance, 아니면 그냥 패스 (paused 상태 안 만듦)
                _cid = (
                    (schema and getattr(schema, "chain_id", ""))
                    or task.get("chain_id", "")
                    or ""
                )
                if _cid:
                    self._force_advance_chain = True
                    try:
                        asyncio.ensure_future(self.run_now(""))
                    except Exception as _e_ar:
                        print(f"  ⚠️ [Autonomous] run_now 트리거 실패 (무시): {_e_ar}")
                return
        except Exception as _e_auto:
            # 자율 모드 검사 실패는 안전 폴백 — 기존 게이트 로직 그대로 진행
            print(f"  ⚠️ [Autonomous] PauseGate 자동 통과 검사 실패 (무시): {_e_auto}")

        # ── [B5 2026-04-29] Tier 2/3 분기 wire ────────────────────────
        # auto-pass 차단 후, 일반 게이트 카드 대신 tier 별 전용 카드 발송.
        # tier2_info → send_quick_question (옵션 3개)
        # tier3_intent → send_tier3_preview (Verifier 판정 + 산출물 head)
        # 응답은 done_callback 으로 task["__pause_response__"] 저장 + force advance.
        if _tier == "tier2_info":
            try:
                from eidos_telegram_bot import send_quick_question

                _q_question = (headline or ask_user or "추가 정보 필요")[:300]
                _q_options = ["기본값 사용", "스킵", "건너뛰기"]
                _q_timeout = float(min(120, auto_advance_sec or 120))
                _captured_task = task  # closure 안전 캡처
                _captured_schema = schema

                async def _on_tier2_response():
                    try:
                        _r = await send_quick_question(
                            _q_question, _q_options, timeout=_q_timeout,
                            title="Tier 2 — 정보 부족",
                        )
                    except Exception as _e_qq:
                        print(f"  ⚠️ [Tier 2 응답] send_quick_question 실패: {_e_qq}")
                        _r = None
                    try:
                        _captured_task["__pause_response__"] = _r or ""
                        _captured_task["__pause_response_tier__"] = "tier2_info"
                        _captured_task["status"] = "ACTIVE"
                        if self._paused_task is _captured_task:
                            self._paused_task = None
                        self._force_advance_chain = True
                        asyncio.ensure_future(self.run_now(""))
                    except Exception as _e_pr:
                        print(f"  ⚠️ [Tier 2 응답] task 갱신 실패: {_e_pr}")

                asyncio.ensure_future(_on_tier2_response())
                print(f"  📋 [Tier 2 — 정보 부족] 짧은 질문 카드 발송 (timeout={int(_q_timeout)}s)")
                self._report(
                    f"📋 **[Tier 2 — 정보 부족]** 짧은 질문 카드 발송 — 응답 대기 중."
                )
                task["status"] = "WAITING_USER"
                task["__pause_reason__"] = "tier2_quick_question"
                self._paused_task = task
                return
            except Exception as _e_tier2:
                print(f"  ⚠️ [Tier 2 wire] 실패 (일반 게이트로 폴스루): {_e_tier2}")

        if _tier == "tier3_intent":
            try:
                from eidos_telegram_bot import send_tier3_preview

                # history 추출 — task 또는 schema 에서 시도, 실패 시 빈 리스트
                _history: List[Dict[str, Any]] = []
                try:
                    _history = (
                        task.get("history", [])
                        or getattr(schema, "_react_history", None)
                        or []
                    )
                except Exception:
                    _history = []
                _t3_timeout = float(auto_advance_sec or 300)
                _captured_task = task
                _captured_schema = schema

                async def _on_tier3_response():
                    try:
                        _r = await send_tier3_preview(
                            _captured_schema, _history, timeout=_t3_timeout,
                        )
                    except Exception as _e_t3:
                        print(f"  ⚠️ [Tier 3 응답] send_tier3_preview 실패: {_e_t3}")
                        _r = None
                    try:
                        _captured_task["__pause_response__"] = _r or ""
                        _captured_task["__pause_response_tier__"] = "tier3_intent"
                        _captured_task["status"] = "ACTIVE"
                        # "수정 후 재시도" 시 force_user_intervention 해제
                        _force_user_cleared = False
                        if _r == "수정 후 재시도":
                            try:
                                _captured_schema._force_user_intervention = False
                                _force_user_cleared = True
                            except Exception:
                                pass
                        if self._paused_task is _captured_task:
                            self._paused_task = None
                        self._force_advance_chain = True
                        print(
                            f"  🛡️ [Tier 3 응답 처리] r={_r!r} · "
                            f"force_user 해제={'예' if _force_user_cleared else '아니오'} · "
                            f"force_advance_chain=True · run_now() ensure_future"
                        )
                        asyncio.ensure_future(self.run_now(""))
                    except Exception as _e_pr:
                        print(f"  ⚠️ [Tier 3 응답] task 갱신 실패: {_e_pr}")

                asyncio.ensure_future(_on_tier3_response())
                print(f"  🛡️ [Tier 3 — 의도 검토] 미리보기 카드 발송 (timeout={int(_t3_timeout)}s)")
                self._report(
                    f"🛡️ **[Tier 3 — 의도 검토]** Verifier 판정 + 산출물 미리보기 카드 발송 — 응답 대기."
                )
                task["status"] = "WAITING_USER"
                task["__pause_reason__"] = "tier3_preview"
                self._paused_task = task
                return
            except Exception as _e_tier3:
                print(f"  ⚠️ [Tier 3 wire] 실패 (일반 게이트로 폴스루): {_e_tier3}")

        # ──────────────────────────────────────────────────────────

        # [Fix 2026-04-24] chain stage 가 같은 schema 로 _pause_with_telegram_gate
        # 를 N 회 호출하면 사용자 응답 없이도 즉시 force advance. AutoRun → 새 ReAct
        # → 또 Verifier FAIL → 또 ASK_USER 의 무한 루프 차단.
        try:
            chain_id = (schema and getattr(schema, "chain_id", "")) or task.get("chain_id", "")
        except Exception:
            chain_id = task.get("chain_id", "") or ""
        if chain_id:
            _gp_cnt = int(task.get("__chain_gate_count__", 0)) + 1
            task["__chain_gate_count__"] = _gp_cnt
            if _gp_cnt >= 2:
                _stage = task.get("stage", "?")
                print(
                    f"  ⏭️ [PauseGate] chain stage='{_stage}' (chain={chain_id[:8]}) "
                    f"같은 schema 게이트 {_gp_cnt}회 — 즉시 강제 advance"
                )
                self._report(
                    f"⏭️ **[게이트 반복 {_gp_cnt}회 — 강제 진행]** chain stage='{_stage}' "
                    f"무한 루프 방지 위해 사용자 응답 대기 없이 다음 단계로 진행."
                )
                self._force_advance_chain = True
                # [Phase 2.1 trace 2026-04-25] force_advance 가 실제 다음 stage launch
                # 까지 도달하는지 추적 — 1597줄 이후 dispatcher 활동 끊김 진단.
                _pt_status = (self._paused_task or {}).get("status", "(no_paused)")
                _pt_id = id(self._paused_task) if self._paused_task else 0
                _t_id  = id(task)
                print(
                    f"  🔵 [Chain ForceAdvance] dispatch run_now() — "
                    f"chain={chain_id[:8]} stage='{_stage}' "
                    f"task_status='{task.get('status', '?')}' "
                    f"paused_task_status='{_pt_status}' "
                    f"paused_is_same={_pt_id == _t_id} "
                    f"_wait_for_user={self._wait_for_user} "
                    f"force_advance_flag={self._force_advance_chain}"
                )
                # 이어하기 직접 트리거 — _wait_for_user 안 켜고 곧바로 진행.
                try:
                    asyncio.ensure_future(self.run_now(""))
                except Exception as _e_rn2:
                    print(f"  ⚠️ [PauseGate] run_now 트리거 실패: {_e_rn2}")
                return

        # [Site C 2026-04-26] 비-chain (단일 ad-hoc) task 의 반복 ASK_USER 차단.
        # Why: chain_id 가 없으면 위 chain 게이트가 발동 안 해 `Verifier FAIL → ASK_USER →
        # 사용자가 ⚡ 자율 실행 클릭 → run_now → 같은 task 재실행 → 같은 FAIL` 무한 루프.
        # (예: file_write task 에서 placeholder 'HOW_MUCH=작업 완료' 가 자동 달성 안 돼
        # 매 클릭마다 같은 missing_attr — 2026-04-26 22:01~22:22 보고서 작성 케이스.)
        # chain task 의 __chain_gate_count__ 와 동일 정신: 2회 도달 시 즉시 종료.
        # 진전이 있으면(다른 fail_type/reason) 카운터 reset 해 false 차단 회피.
        else:
            _fail_sig = (
                f"{getattr(schema, 'task_type', '') or ''}"
                f"|{(headline or '')[:40]}|{(ask_user or '')[:80]}"
            )
            _prev_sig = task.get("__verifier_fail_sig__", "")
            if _prev_sig != _fail_sig:
                task["__verifier_fail_gate_count__"] = 1
                task["__verifier_fail_sig__"] = _fail_sig
            else:
                _vf_cnt = int(task.get("__verifier_fail_gate_count__", 0)) + 1
                task["__verifier_fail_gate_count__"] = _vf_cnt
                if _vf_cnt >= 2:
                    print(
                        f"  🛑 [PauseGate] 비-chain task 같은 검증 FAIL {_vf_cnt}회 — "
                        f"동일 산출물 재실행 무의미 → task 강제 종료"
                    )
                    self._report(
                        f"🛑 **[반복 검증 실패 — 작업 종료]** 같은 사유로 {_vf_cnt}회 "
                        f"FAIL — 사용자 개입 없이 재실행해도 결과 동일하므로 종료합니다.\n"
                        f"└─ 사유: {(ask_user or '')[:120]}"
                    )
                    task["status"] = "FAILED"
                    task["fail_reason"] = "verifier_fail_repeat"
                    self._paused_task = None
                    self._wait_for_user = False
                    try:
                        self._remove_task_from_schedule(task)
                    except Exception as _e_rm:
                        print(f"  ⚠️ [PauseGate] schedule 제거 실패 (무시): {_e_rm}")
                    try:
                        from eidos_telegram_bot import get_bot as _get_bot
                        _bot = _get_bot()
                        if _bot.is_configured():
                            asyncio.ensure_future(
                                _bot.send_notification(
                                    f"같은 검증 FAIL {_vf_cnt}회 반복 — task 강제 종료.\n"
                                    f"<code>{(ask_user or '')[:160]}</code>",
                                    title="🛑 반복 검증 실패 — 작업 종료",
                                )
                            )
                    except Exception:
                        pass
                    return

        self._report(headline)
        self._report(ask_user)
        self._paused_task = task
        task["status"] = "WAITING_USER"
        self._wait_for_user = True

        # Telegram 알림 (best-effort)
        try:
            from eidos_telegram_bot import get_bot as _get_bot
            _bot = _get_bot()
            if _bot.is_configured():
                asyncio.ensure_future(
                    _bot.send_notification(telegram_body, title=telegram_title)
                )
        except Exception as _e_tg:
            print(f"  ⚠️ [Telegram] 알림 발송 실패 (무시): {_e_tg}")

        # 체인 task 면 무응답 타임아웃에 자동 진행
        chain_id = ""
        try:
            chain_id = (schema and getattr(schema, "chain_id", "")) or task.get("chain_id", "")
        except Exception:
            chain_id = task.get("chain_id", "") or ""
        if chain_id and auto_advance_sec > 0:
            asyncio.ensure_future(
                self._schedule_auto_chain_advance(task, int(auto_advance_sec))
            )

    async def _schedule_auto_chain_advance(self, task: Dict, delay_sec: int) -> None:
        """Chain stage 무응답 시 일정 시간 뒤 강제 완료 + 다음 stage 로 진행."""
        try:
            await asyncio.sleep(delay_sec)
        except Exception:
            return
        # 타이머 만료 시점 재확인: 사용자가 이미 응답/다른 task 진입했으면 취소
        if self._paused_task is not task:
            return
        if task.get("status") != "WAITING_USER":
            return
        chain_id = task.get("chain_id", "")
        stage    = task.get("stage", "")
        minutes  = max(1, delay_sec // 60)
        self._report(
            f"⏭️ **[자동 진행]** 무응답 {minutes}분 경과 — chain_id={chain_id[:8]} "
            f"stage='{stage}' 강제 완료 처리 후 다음 단계로 넘어갑니다."
        )
        try:
            from eidos_telegram_bot import get_bot as _get_bot
            _bot = _get_bot()
            if _bot.is_configured():
                await _bot.send_notification(
                    f"무응답 {minutes}분 경과. stage=<code>{stage}</code> "
                    f"강제 완료 → 다음 단계로 진행합니다.",
                    title="⏭️ 체인 자동 진행",
                )
        except Exception:
            pass
        self._force_advance_chain = True
        try:
            await self.run_now("")
        except Exception as _e_rn:
            print(f"  ⚠️ [AutoAdvance] run_now 실패: {_e_rn}")

    # ── ReAct 완료 처리 ────────────────────────────────────────────────────
    async def _on_react_done(
        self,
        task_prompt: str,
        goal_ref: str,
        task: Dict,
        history: list,
        schema,
        _retry_count: int = 0,
    ) -> None:
        """
        ReAct 루프 DONE 처리.

        Phase 2: Verifier AI 독립 검증
        Phase 3: FAIL 시 RepairPlanner로 부분 재실행
        """
        # [Phase 2.2] 활성 신호 — done 처리 진입 (verifier/orch 호출 직전)
        self._touch_activity()
        # [Phase 4-1 2026-04-25] force_advance origin 감지.
        # _react_loop force_advance 분기에서 task 에 마커 세팅하고 호출됨. 의도는
        # "이 stage 는 끝났다 치고 다음 stage 로 진행" 이므로 Verifier/retry/HOW_MUCH
        # 게이트 모두 SKIP. 그대로 chain advance 분기로 진입해야 register stage
        # 무한 retry 폭포 차단 (history 빈 상태 → Verifier missing_attr FAIL 자동).
        _force_origin = bool(task.pop("__force_advance_origin__", False))
        if _force_origin:
            print(
                f"  🔵 [Chain ForceAdvance] _on_react_done force-origin — "
                f"Verifier/retry/HOW_MUCH 게이트 모두 SKIP, chain advance 직진"
            )
            self._report(
                f"⏭️ **[강제 진행 — 검증 우회]** stage='{getattr(schema, 'stage', '?')}' "
                f"force_advance 의도 존중 — Verifier·재시도 SKIP, 다음 단계로 직진."
            )
            # [P0-2 2026-04-25] force_advance 사실을 chain.metadata 에 기록 → 다음 stage
            # 진입 시 "이전 stage 산출물 신뢰 못 함" 신호로 사용. dev 가 빈 research 위에서
            # 환각 산출물 만드는 패턴 차단.
            try:
                _chain_id_fa = (task.get("chain_id") or "").strip()
                if _chain_id_fa:
                    from eidos_mission_chain import get_orchestrator as _go_fa
                    _orch_fa = _go_fa()
                    _ch_fa = _orch_fa.registry.get(_chain_id_fa)
                    if _ch_fa is not None:
                        _fa_list = _ch_fa.metadata.setdefault("force_advanced_stages", [])
                        _stage_n = (getattr(schema, "stage", "") or "").lower()
                        if _stage_n and _stage_n not in _fa_list:
                            _fa_list.append(_stage_n)
                            print(
                                f"  📌 [Chain ForceAdvance] chain.metadata "
                                f"force_advanced_stages += '{_stage_n}' → {_fa_list}"
                            )
            except Exception as _e_fa_meta:
                print(f"  ⚠️ [Chain ForceAdvance] metadata 기록 실패 (무시): {_e_fa_meta}")

            # [P2-5 2026-04-25] 0스텝/빈 산출물 force_advance 차단.
            # history 길이 + react_vars 산출물 길이를 합쳐 stage 가 실제로 작업했는지 검증.
            # 둘 다 0/비어있으면 chain_success=False 로 강제 + Telegram 알림.
            try:
                _hist_n = len(history or [])
                _vars_dict = getattr(schema, "_react_vars", None) or {}
                # 의미있는 산출물: 길이 100자 이상 변수가 1개 이상이거나 history >= 1
                _meaningful_outputs = sum(
                    1 for _v in _vars_dict.values()
                    if isinstance(_v, str) and len(_v) >= 100
                )
                _MIN_HIST = 1
                _MIN_OUTPUT = 1
                if _hist_n < _MIN_HIST and _meaningful_outputs < _MIN_OUTPUT:
                    print(
                        f"  🔴 [P2-5] force_advance 0산출물 감지 — "
                        f"history={_hist_n} meaningful_outputs={_meaningful_outputs} "
                        f"→ chain_success 강제 False + stage_failed 마킹"
                    )
                    self._report(
                        f"🔴 **[빈 산출물 강제 진행 차단]** stage='{getattr(schema,'stage','?')}' — "
                        f"실행 step 0건 + 의미있는 산출물 0건. 이전 stage 가 사실상 미완료. "
                        f"chain_success=False 처리, 사용자 개입 요청."
                    )
                    task["__stage_failed__"] = True
                    task["__stage_fail_reason__"] = (
                        f"force_advance_zero_output:hist={_hist_n}:vars={_meaningful_outputs}"
                    )
                    try:
                        if schema is not None:
                            schema._chain_success = False
                            schema._stage_fail_reason = task["__stage_fail_reason__"]
                    except Exception:
                        pass
                    # Telegram best-effort
                    try:
                        from eidos_telegram_bot import get_bot as _gb_p25
                        _bot_p25 = _gb_p25()
                        if _bot_p25.is_configured():
                            asyncio.ensure_future(_bot_p25.send_notification(
                                f"⚠️ chain stage='{getattr(schema,'stage','?')}' 빈 산출물로 강제 완료 처리됨 — "
                                f"chain_success=False. 사용자 개입 필요.",
                                title="🔴 빈 산출물 강제 진행 차단",
                            ))
                    except Exception:
                        pass
            except Exception as _e_p25:
                print(f"  ⚠️ [P2-5] 산출물 검증 실패 (무시): {_e_p25}")

            # [P1-1 2026-04-25] 거짓 완료 사슬 강화 — 직전 Verifier FAIL 잔류 차단.
            # Why: P2-5 는 0산출물만 차단(95자 research_report 통과). Verifier 가
            # insufficient_items / missing_attr 로 FAIL 했고 retry 후에도 그대로
            # ASK_USER → 5분 무응답 → force_advance 흐름이면 산출물이 정량 미달인
            # 채로 다음 stage 가 환각 시작. 직전 schema._last_verifier_fail.fail_type
            # 이 정량/속성 결손이면 chain_success=False 강제.
            try:
                _last_vf_fa = getattr(schema, "_last_verifier_fail", None) if schema else None
                _vf_type_fa = ""
                if isinstance(_last_vf_fa, dict):
                    _vf_type_fa = (_last_vf_fa.get("fail_type") or "").strip().lower()
                _STRUCTURAL_VF_FAIL = {"insufficient_items", "missing_attr"}
                if _vf_type_fa in _STRUCTURAL_VF_FAIL:
                    _vf_reason_fa = (
                        (_last_vf_fa.get("reason") or "")[:160]
                        if isinstance(_last_vf_fa, dict) else ""
                    )
                    print(
                        f"  🔴 [P1-1] force_advance 직전 Verifier FAIL 잔류 — "
                        f"fail_type='{_vf_type_fa}' → chain_success 강제 False"
                    )
                    self._report(
                        f"🔴 **[거짓 완료 차단]** stage='{getattr(schema,'stage','?')}' — "
                        f"직전 Verifier FAIL({_vf_type_fa}) 미해결 채로 force_advance 진입. "
                        f"chain_success=False 처리, 사용자 개입 요청.\n"
                        f"└ 사유: {_vf_reason_fa}"
                    )
                    task["__stage_failed__"] = True
                    task["__stage_fail_reason__"] = (
                        f"verifier_unresolved:{_vf_type_fa}"
                    )
                    try:
                        if schema is not None:
                            schema._chain_success = False
                            schema._stage_fail_reason = task["__stage_fail_reason__"]
                    except Exception:
                        pass
                    # Telegram best-effort
                    try:
                        from eidos_telegram_bot import get_bot as _gb_p11
                        _bot_p11 = _gb_p11()
                        if _bot_p11.is_configured():
                            asyncio.ensure_future(_bot_p11.send_notification(
                                f"⚠️ chain stage='{getattr(schema,'stage','?')}' Verifier "
                                f"{_vf_type_fa} FAIL 미해결 — chain_success=False. "
                                f"사용자 개입 필요.",
                                title="🔴 거짓 완료 차단 (P1-1)",
                            ))
                    except Exception:
                        pass
            except Exception as _e_p11:
                print(f"  ⚠️ [P1-1] verifier 잔류 검사 실패 (무시): {_e_p11}")

        # [P0-1-b 2026-04-25] 외부 액션 stage 가 실제 액션 0건으로 끝나면 stage_failed.
        # register/deliver/support 처럼 외부 공개·발송이 전제인 stage 가 LLM 응답만으로
        # "완료" 처리되는 거짓 보고 차단. chain.metadata 에 신호 + Telegram 알림.
        _ext_stage = task.get("__external_action_stage__", "")
        _ext_count = int(task.get("__external_action_count__", 0) or 0)
        if _ext_stage and _ext_count == 0:
            print(
                f"  🔴 [ExternalActionGuard] stage='{_ext_stage}' 외부 액션 0건 — "
                f"stage_failed 마킹 (LLM 응답만으로 완료 처리 차단)"
            )
            self._report(
                f"🔴 **[외부 액션 0건]** stage='{_ext_stage}' — 실제 등록/발송/대응 액션이 "
                f"한 건도 실행되지 않음. chain stage_failed 처리."
            )
            task["__stage_failed__"] = True
            task["__stage_fail_reason__"] = f"external_action_zero:{_ext_stage}"
            try:
                if schema is not None:
                    schema._chain_success = False
                    schema._stage_fail_reason = f"external_action_zero:{_ext_stage}"
            except Exception:
                pass
        # ── Phase 2: Verifier 검증 ────────────────────────────────────────
        verifier_result = None
        if schema and not _force_origin:
            self._report("🔍 **[Verifier]** 작업 결과 독립 검증 중...")
            try:
                from eidos_verifier import get_verifier, RepairPlanner
                verifier = get_verifier()

                # 최종 브라우저 상태 수집
                final_url  = self._get_current_browser_url()
                final_page = ""
                try:
                    from execution_module import get_browser_content as _gc_vfy
                    raw_page = await _gc_vfy()
                    # HTML 태그 제거해 텍스트만 추출
                    final_page = re.sub(r'<[^>]+>', ' ', raw_page)
                    final_page = re.sub(r'\s+', ' ', final_page).strip()[:1200]
                except Exception:
                    pass

                verifier_result = await verifier.verify(
                    schema=schema,
                    history=history,
                    final_url=final_url,
                    final_page=final_page,
                )
                self._report(verifier_result.to_report())

            except Exception as e:
                import traceback
                print(f"  ⚠️ [Verifier] 검증 실패 (완료 처리): {e}")
                traceback.print_exc()

        # ── Phase 3: FAIL → 부분 재실행 ──────────────────────────────────
        # [Fix P1 2026-04-22] 재시도 한도 2 → 1.
        # Why: 동일 미달 속성에 대해 ReAct가 같은 행동(WRITE 반복)을 다시
        # 하는 경향. 2회 재시도 = LLM 호출/TTS 폭주만 일으키고 결과 동일.
        # Verifier P1 패치(첫 시도 LLM 위임 + 자동 달성 반영)로 진짜 충족
        # 케이스는 1차 통과하므로, FAIL이 1번 재시도로도 안 풀리면 ASK_USER
        # 빠른 전환이 사용자 시간/토큰 모두 절약.
        if verifier_result and not verifier_result.passed:
            # [P1 Fix #5 2026-04-24] Verifier FAIL 기록을 schema 에 보관해 다음 WRITE 시
            # LLM 프롬프트에 "직전 실패 사유" 블록으로 주입 — 같은 실수(원본 복사 / 필수
            # 섹션 누락 / 형식 불일치) 반복을 프롬프트 레벨에서 억제.
            _fail_type_now = getattr(verifier_result, "fail_type", "") or ""
            try:
                if schema is not None:
                    schema._last_verifier_fail = {
                        "fail_type": _fail_type_now,
                        "reason": (getattr(verifier_result, "reason", "") or "")[:500],
                        "failed_attrs": list(getattr(verifier_result, "failed_attrs", []) or []),
                        "repair_hint": (getattr(verifier_result, "repair_hint", "") or "")[:300],
                    }
            except Exception as _e_vf_rec:
                print(f"  ⚠️ [Verifier FAIL 기록] 실패 (무시): {_e_vf_rec}")

            # ── [Patch 2026-04-29 v2 + Phase 2-β 2026-05-01] Verifier FAIL 누적 → 사용자 강제 개입 ─────
            # v1 (drift/missing_attr 화이트리스트) 누수: 최신로그 144ecc33 케이스
            # 에서 새 fail_type "insufficient_items" 가 cap 매칭 X → 60 step 도달.
            # v2 정책 — fail_type 카테고리 분류:
            #   intent (의도 misalignment): drift / insufficient_items / external_action_missing → 1회 cap
            #   quality (자료 부족): missing_attr → 2회 cap (retry 1번 허용)
            #   other (알 수 없는 새 타입): 2회 cap (안전망)
            # Phase 2-β 2026-05-01: external_action_missing 추가 (1차 dogfooding 사례 — schema.
            # execution_action 명시 + WRITE 만으로 종료 의도 → 즉시 cap 으로 사용자 개입).
            # _pause_with_telegram_gate 가 _force_user_intervention 보면 auto-pass 차단.
            try:
                if schema is not None:
                    _verifier_fails = list(getattr(schema, "_verifier_fail_history", []) or [])
                    _verifier_fails.append(_fail_type_now)
                    schema._verifier_fail_history = _verifier_fails[-5:]  # 최근 5개만
                    _INTENT_TYPES = ("drift", "insufficient_items", "external_action_missing")
                    _QUALITY_TYPES = ("missing_attr",)
                    _intent_n = sum(1 for t in _verifier_fails if t in _INTENT_TYPES)
                    _quality_n = sum(1 for t in _verifier_fails if t in _QUALITY_TYPES)
                    _other_n = sum(
                        1 for t in _verifier_fails
                        if t and t not in _INTENT_TYPES and t not in _QUALITY_TYPES
                    )
                    _force_user = (_intent_n >= 1) or (_quality_n >= 2) or (_other_n >= 2)
                    if _force_user and not getattr(schema, "_force_user_intervention", False):
                        schema._force_user_intervention = True
                        schema._force_user_reason = (
                            f"intent {_intent_n}회 / quality {_quality_n}회 / other {_other_n}회 "
                            f"(types: {','.join(_verifier_fails[-3:])})"
                        )
                        print(
                            f"  🛡️ [Verifier Cap] {schema._force_user_reason} → "
                            f"사용자 개입 강제 (auto-pass 차단)"
                        )
                        self._report(
                            f"🛡️ **[Verifier 누적 cap]** {schema._force_user_reason}. "
                            f"같은 실수 반복 차단 — 사용자 개입 필요."
                        )
            except Exception as _e_vf_cap:
                print(f"  ⚠️ [Verifier Cap] 누적 처리 실패 (무시): {_e_vf_cap}")
            # ──────────────────────────────────────────────────────────
            if _retry_count < 1:  # 최대 1회 재시도
                self._report(
                    f"🔧 **[부분 재실행]** {verifier_result.fail_type} 수정 중...\n"
                    f"└─ 재시도 {_retry_count + 1}/1회"
                )
                try:
                    from eidos_verifier import RepairPlanner
                    repair = RepairPlanner.plan(verifier_result, schema, history)
                    strategy = repair.get("strategy", "rerun_all")

                    # 10-1 수정: strategy 분기 처리
                    self._report(
                        f"🔧 **[RepairPlanner]** strategy={strategy} | "
                        f"target_attrs={repair.get('target_attrs', [])}"
                    )

                    # 롤백: 드리프트/요령/rollback_to → 특정 URL로 브라우저 복귀
                    rollback_url = repair.get("rollback_url", "")
                    if rollback_url and strategy in ("rollback_to", "retry_attrs"):
                        self._report(f"↩️ **[롤백]** {rollback_url[:60]} 으로 복귀")
                        try:
                            from execution_module import browser_smart_navigate as _bnav
                            await _bnav(url=rollback_url)
                            await asyncio.sleep(2.0)
                        except Exception as e:
                            print(f"  ⚠️ [Repair] 롤백 실패: {e}")

                    # 수정 힌트를 스키마에 주입
                    if schema:
                        extra_hint = repair.get("hint", "")
                        if extra_hint:
                            schema.how = extra_hint + "\n\n(원래 HOW: " + schema.how + ")"
                        schema._repair_count = _retry_count + 1

                        # 10-1: retry_attrs 전략 — 달성된 속성은 유지,
                        # 실패 속성만 재시도 (RepairPlanner가 이미 achieved=False 처리)
                        if strategy == "retry_attrs":
                            target_attrs = repair.get("target_attrs", [])
                            # [P0 Fix 2026-04-24] target_attrs 빈 배열 최종 안전망.
                            # Verifier fallback 4단에도 불구하고 어떤 이유로 빈 배열이
                            # 전달되면, 전체 속성을 재검증 대상으로 설정하고 attrs 를
                            # 전부 achieved=False 로 리셋. 이래야 재시도 LLM 이
                            # "[아직 미달성 HOW_MUCH] 없음 (모두 달성됨)" 을 보고
                            # DONE 만 리턴하는 무의미 재시도를 방지.
                            if not target_attrs and schema.attributes:
                                target_attrs = [a.name for a in schema.attributes]
                                for _a in schema.attributes:
                                    _a.achieved = False
                                    _a.evidence = ""
                                self._report(
                                    f"⚠️ **[RepairPlanner fallback]** target_attrs 비어있음 — "
                                    f"전체 {len(target_attrs)}개 속성 재검증 모드"
                                )
                            achieved_summary = ", ".join(
                                f"{a.name}✅" for a in schema.attributes if a.achieved
                            )
                            # [P0 Fix 2026-04-24] 블록 조건 완화 — achieved_summary 가
                            # 비어있어도(모두 리셋 후) 재시도 지시와 Verifier 사유를
                            # 주입해야 LLM 이 구체 행동을 취함.
                            _pending_txt = (
                                ", ".join(target_attrs) if target_attrs
                                else "(특정 속성 없음 — 산출물 전체 재작성 필요)"
                            )
                            _reason_snip = (
                                (getattr(verifier_result, "reason", "") or "")[:200]
                            )
                            schema.how = (
                                f"[이미 달성 완료: {achieved_summary or '없음'}]\n"
                                f"[재시도 필요: {_pending_txt}]\n"
                                f"[Verifier 실패 사유: {_reason_snip}]\n"
                                f"[재시도 지침] 위 '재시도 필요' 항목을 실제로 충족시키는 새 산출물을 "
                                f"작성해야 한다. 이전 산출물이 '이미 완료'로 보여도, Verifier 가 "
                                f"의미론적으로 실패 판정을 내린 것이므로 즉시 DONE 을 선언하면 "
                                f"무한 루프가 된다. 반드시 재작성/수정 액션을 먼저 수행하라.\n\n"
                            ) + schema.how

                            # ── [TOP-N Selection Rule 2026-04-27] ──────────────────
                            # retry 진입 시 schema/prompt 에서 'top N' / '상위 N' 패턴
                            # 감지되면 본문 형식을 hard rule 로 강제. LLM 이 dump → 선별
                            # 행동 변화하도록 schema.how 머리에 prepend.
                            try:
                                _topn = _detect_topn_selection(
                                    schema, task.get("prompt") or task_prompt or ""
                                )
                                if _topn > 0:
                                    schema.how = _build_topn_selection_rule(_topn) + schema.how
                                    print(
                                        f"  📐 [Retry SELECTION] TOP-{_topn} 선별 룰 prepend "
                                        f"— dump 차단 + 형식 강제"
                                    )
                                    self._report(
                                        f"📐 **[Retry SELECTION]** `TOP-{_topn}` 선별 룰 주입 — "
                                        f"본문 전체 나열 차단, 번호매김 형식 강제."
                                    )
                            except Exception as _e_topn:
                                print(f"  ⚠️ [Retry SELECTION] 룰 주입 실패 (무시): {_e_topn}")

                    task["status"] = "RUNNING"

                    # [Same-Target Lock 2026-04-27] retry_attrs 시 LLM 이 새 파일명을
                    # 임의 생성하는 것 차단. Verifier FAIL → retry 진입 시 같은 산출물
                    # 을 보강해야 의미 보존인데, 최신로그에서 LLM 이 '_최종.md' suffix
                    # 등 자동 생성 → 보고서 2개 분리 → retry 의도 누수.
                    # history 에서 마지막으로 성공한 WRITE target 을 추출해
                    # schema._retry_target_lock 에 박음. WRITE 액션 처리 시점에서
                    # 이 lock 이 있으면 새 target 강제 교체.
                    if strategy == "retry_attrs" and schema is not None:
                        try:
                            _last_write_t = ""
                            for _h in reversed(history or []):
                                if _h.get("action") != "WRITE":
                                    continue
                                _hr = _h.get("result", "") or ""
                                if "파일 저장" in _hr or "생성 완료" in _hr:
                                    _last_write_t = (_h.get("target", "") or "").strip()
                                    if _last_write_t:
                                        break
                            if _last_write_t:
                                schema._retry_target_lock = _last_write_t
                                # retry 시 risk 다운그레이드 — 이미 한 번 사용자가 승인한 흐름.
                                # 같은 산출물 보강을 위한 WRITE 가 매번 HIGH risk 로 텔레그램
                                # 승인 게이트 걸리면 자동 흐름이 멈춤. retry origin 임을 표시.
                                schema._is_retry_run = True
                                print(
                                    f"  🔒 [Retry Target Lock] retry_attrs 동일 산출물 강제: "
                                    f"{_last_write_t[:60]} (risk 다운그레이드 ON)"
                                )
                        except Exception as _e_rtl:
                            print(f"  ⚠️ [Retry Target Lock] 설정 실패 (무시): {_e_rtl}")

                    # [Retry Strike Reset 2026-04-27] 부분 재실행 시작 시 누적된
                    # PRE-VETO strike + sampling block 카운터 reset.
                    # Why: 최신로그.txt 에서 retry 1 시작 후에도 strike 6→7→8 이어짐 →
                    # WRITE 사전 거부 7회/8회 누적 표시 → retry 의 의미 없음 (게이트가
                    # 이전 세션의 strike 까지 합산해 즉시 차단). retry 는 새 시도이므로
                    # 누적값이 0 부터 다시 출발해야 함. _react_var_urls / _react_vars 같은
                    # 실제 정보 수집 결과는 보존 — 이전 수집 데이터는 자산.
                    try:
                        if hasattr(schema, "_pre_veto_strikes"):
                            schema._pre_veto_strikes = {}
                        if hasattr(schema, "_sampling_blocked_targets"):
                            schema._sampling_blocked_targets = {}
                        if hasattr(schema, "_write_sampling_gate_count"):
                            schema._write_sampling_gate_count = 0
                        if hasattr(schema, "_n_required_downgraded"):
                            # 다운그레이드도 reset — retry 는 원본 N 으로 재평가
                            schema._n_required_downgraded = 0
                        # [Retry Drift Guard 2026-04-27] lock-target WRITE 성공 마커 reset.
                        # 새 retry 사이클은 다시 처음부터 — WRITE 시도 가능, 성공 후
                        # NAVIGATE/FILL/CLICK 차단 활성화는 lock-target WRITE 성공 시점부터.
                        schema._retry_lock_write_done = False
                        print(
                            f"  🔄 [Retry Reset] PRE-VETO strikes + sampling blocks 초기화 "
                            f"(retry={_retry_count + 1})"
                        )
                    except Exception as _e_rs:
                        print(f"  ⚠️ [Retry Reset] 초기화 실패 (무시): {_e_rs}")

                    # 10-2: 부분 재실행 시 이전 성공 history를 seed로 전달
                    prev_history = [h for h in history if h.get("verified")]
                    await self._react_loop_with_retry(
                        task_prompt, goal_ref, task,
                        retry_count=_retry_count + 1,
                        seed_history=prev_history,
                    )
                    return

                except Exception as e:
                    print(f"  ⚠️ [Repair] 부분 재실행 실패 (완료 처리): {e}")
            else:
                _reason_txt = str(getattr(verifier_result, "reason", "") or "검증 실패")
                _stage_txt  = (task.get("stage") or getattr(schema, "stage", "") or "?")
                self._pause_with_telegram_gate(
                    task, schema,
                    headline=(
                        f"⚠️ **[재시도 한도 초과]** {_retry_count}회 재시도 후에도 Verifier FAIL\n"
                        f"└─ 사용자 확인이 필요합니다."
                    ),
                    ask_user=(
                        f"[ASK_USER] ❓ **[검증 실패]** {_reason_txt}\n"
                        f"수동으로 확인 후 **⚡ 자율 실행** 버튼을 눌러주세요."
                    ),
                    telegram_title="⚠️ 검증 실패 — 확인 필요",
                    telegram_body=(
                        f"stage=<code>{_stage_txt}</code> 재시도 {_retry_count}회 후에도 Verifier FAIL\n"
                        f"사유: <code>{_reason_txt[:180]}</code>\n\n"
                        f"5분 내 응답이 없으면 자동으로 다음 단계로 진행합니다."
                    ),
                    auto_advance_sec=300,
                )
                return

        # ── [Fix P0-1 2026-04-22] HOW_MUCH 0/N 이중 가드 ──────────────────
        # Why: Verifier가 None을 반환했거나(예외/폴백) LLM이 관대 PASS를 줘서
        # 여기까지 와도, 정량 게이트(HOW_MUCH 0개 달성)는 우회되면 안 됨.
        # Verifier 패치만으로는 verifier_result=None 경로(예외/임계치 폴백)가
        # 남아있으므로 완료 출력 직전에 이중으로 차단.
        # 동작: 재시도 한도 초과 분기와 동일하게 ASK_USER로 전환.
        # [Phase 4-1 2026-04-25] force_advance origin 인 경우 SKIP — 의도가
        # "이 stage 는 부분 산출물로 다음으로 진행" 이므로 정량 게이트 검증
        # 자체가 무의미 (force advance 후 또 ASK_USER 면 무한 루프).
        if (schema and schema.attributes
                and not any(a.achieved for a in schema.attributes)
                and not _force_origin):
            _miss = ", ".join(a.name for a in schema.attributes)
            _stage_txt = (task.get("stage") or getattr(schema, "stage", "") or "?")
            self._pause_with_telegram_gate(
                task, schema,
                headline=(
                    f"⚠️ **[완료 차단]** HOW_MUCH 0/{len(schema.attributes)}개 달성 — "
                    f"산출물이 있어도 완료 처리하지 않습니다.\n"
                    f"└─ 미달성: {_miss[:120]}"
                ),
                ask_user=(
                    f"[ASK_USER] ❓ **[정량 게이트 미통과]** "
                    f"`{task_prompt[:60]}`\n"
                    f"HOW_MUCH 속성이 모두 미달성입니다. 결과 확인 후 "
                    f"**⚡ 자율 실행** 버튼을 눌러주세요."
                ),
                telegram_title="⚠️ 정량 게이트 미통과",
                telegram_body=(
                    f"stage=<code>{_stage_txt}</code>\n"
                    f"HOW_MUCH 0/{len(schema.attributes)} 달성 — 산출물은 있지만 속성 검증 미통과.\n"
                    f"미달성: <code>{_miss[:160]}</code>\n\n"
                    f"5분 내 응답이 없으면 자동으로 다음 단계로 진행합니다."
                ),
                auto_advance_sec=300,
            )
            return

        # ── PASS: 정상 완료 처리 ──────────────────────────────────────────
        task["status"] = "DONE"
        self._remove_task_from_schedule(task)

        # [Fix 2026-04-25] P1-1 / P2-5 / ExternalActionGuard 가 stage_failed 마커를
        # 세팅한 force_advance 분기에서는 task.status=DONE 이어도 사실상 실패다.
        # 이 경우 "🎉 작업 완료" + "📄 요약 파일 저장" 가 그대로 발화되면
        # 사용자에게 모순된 신호를 보낸다. 메시지를 🔴 강제 완료(실패) 로 갈음하고
        # 요약파일 저장은 스킵한다 (chain advance 분기는 아래에서 chain_success=False
        # 로 정상 처리됨).
        _stage_failed_flag = bool(task.get("__stage_failed__"))
        _stage_fail_reason = str(task.get("__stage_fail_reason__", "") or "")

        if _stage_failed_flag:
            summary_fail = schema.achievement_summary if schema else "완료(실패 마커 있음)"
            self._report(
                f"🔴 **[강제 완료(스테이지 실패)]** `{task_prompt[:60]}`\n"
                f"├─ 총 {len(history)}스텝 실행\n"
                f"├─ 사유: {_stage_fail_reason or '(미상)'}\n"
                f"└─ {summary_fail}"
            )
        else:
            summary = schema.achievement_summary if schema else "완료"
            self._report(
                f"🎉 **[작업 완료]** `{task_prompt[:60]}`\n"
                f"├─ 총 {len(history)}스텝 실행\n"
                f"└─ {summary}"
            )

        # [Fix] browser_read / VERB=READ 완료 시 WRITE 결과를 .txt 파일로 자동 저장.
        # 사용자가 "X 요약해줘" 했을 때 GUI 리포트만 남고 파일이 안 만들어지면
        # 결과를 다시 찾기 어려움. 가장 큰 WRITE 결과를 {주제}_요약문.txt로 보관.
        # [Fix 2026-04-25] stage_failed 마커가 있으면 요약 파일도 만들지 않는다 —
        # 검증 미통과 산출물을 "정상 결과물" 처럼 디스크에 박제하지 않기 위해.
        try:
            _ttype = (getattr(schema, "task_type", "") or "") if schema else ""
            _verb  = (getattr(schema, "verb", "") or "").upper() if schema else ""
            _is_read_task = (_ttype == "browser_read" or _verb == "READ")
            if _is_read_task and schema is not None and not _stage_failed_flag:
                _vars = getattr(schema, "_react_vars", {}) or {}
                # WRITE 결과 후보: page_*/content 류는 원본 페이지 텍스트이므로 제외
                _write_candidates = {
                    k: str(v) for k, v in _vars.items()
                    if v and len(str(v)) >= 30
                    and not (k.startswith("page_")
                             or k.endswith("_page")
                             or k.endswith("_content")
                             or "page_content" in k.lower())
                }
                if _write_candidates:
                    # 가장 긴 결과를 최종 요약으로 채택
                    _best_key = max(_write_candidates, key=lambda k: len(_write_candidates[k]))
                    _best_val = _write_candidates[_best_key]

                    # 파일명 생성: 프롬프트에서 따옴표 안 키워드 또는 첫 명사구 추출
                    import re as _re_fn
                    _topic = ""
                    _m = _re_fn.search(r"['\"\u2018\u2019\u201c\u201d](.+?)['\"\u2018\u2019\u201c\u201d]", task_prompt or "")
                    if _m:
                        _topic = _m.group(1).strip()
                    if not _topic:
                        _topic = (task_prompt or "요약").strip().split("\n")[0][:30]
                    # 파일명 안전화: 공백 제거 + Windows 금지 문자 제거
                    _safe = _re_fn.sub(r"[\\/:*?\"<>|\s]+", "", _topic)[:40] or "요약"
                    _fname = f"{_safe}_요약문.txt"

                    try:
                        from execution_module import SAFE_BASE_PATH as _SBASE
                        import os as _os_fn
                        _fpath = _os_fn.path.join(_SBASE, _fname)
                        with open(_fpath, "w", encoding="utf-8") as _fh:
                            _fh.write(f"# {_topic}\n\n")
                            _fh.write(f"원본 요청: {task_prompt}\n")
                            _fh.write(f"작성: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                            _fh.write("---\n\n")
                            _fh.write(_best_val.strip() + "\n")
                        self._report(
                            f"📄 **[요약 파일 저장]** `{_fname}` ({len(_best_val)}자)\n"
                            f"└─ 경로: {_fpath}"
                        )
                        print(f"  ✅ [browser_read] 요약 파일 저장: {_fpath}")
                    except Exception as _fe:
                        print(f"  ⚠️ [browser_read] 요약 파일 저장 실패: {_fe}")
        except Exception as _re_err:
            print(f"  ⚠️ [browser_read] 파일 저장 분기 오류 (무시): {_re_err}")

        # 성공 스키마 → 템플릿 저장 (자기강화)
        if schema and schema.all_attributes_achieved:
            try:
                from eidos_mission_schema import save_template
                save_template(schema)
            except Exception:
                pass

        # [4단계] GoalProgressBridge — HOW_MUCH 속성 → LongTermGoal sub_goals 완료 처리
        try:
            from eidos_goal_bridge import full_bridge
            updated_goal = full_bridge(self.core, schema, goal_ref)
            # [5단계] GUI goal_progress_signal emit — GoalBar 갱신
            if updated_goal:
                self._report(
                    f"[PROGRESS:{updated_goal.name}:{updated_goal.progress:.3f}:"
                    f"{'true' if updated_goal.completed else 'false'}]"
                )
        except Exception as e:
            import traceback
            print(f"  ⚠️ [GoalBridge] 연결 실패 (무시): {e}")
            traceback.print_exc()

        # [5단계] CausalMatrix — Verifier 결과를 내부 XYPair로 적재 (Phase B 전환 누적)
        try:
            trainer = getattr(self.core, "_causal_trainer", None)
            if trainer is not None and schema is not None:
                from eidos_causal_matrix import update_trainer_from_verifier
                pair = update_trainer_from_verifier(
                    trainer, schema,
                    success=(task.get("status") == "DONE"),
                )
                # 내부 샘플이 Phase B 임계값(31개) 도달 시 비동기 재학습
                if pair and len(trainer._pairs_int) >= 31 \
                        and len(trainer._pairs_int) % 10 == 1:
                    import asyncio as _aio
                    from eidos_causal_matrix import MATRIX_CACHE
                    async def _refit():
                        import asyncio as _a
                        loop = _a.get_event_loop()
                        loss = await loop.run_in_executor(None, trainer.refit)
                        await loop.run_in_executor(None, lambda: trainer.save(MATRIX_CACHE))
                        print(f"  ✅ [CausalMatrix] Phase B 재학습 완료 loss={loss:.6f}")
                    _aio.ensure_future(_refit())
        except Exception as _e_cm:
            print(f"  ⚠️ [CausalMatrix] 내부 샘플 적재 실패 (무시): {_e_cm}")

        # ── MissionChain 진행 처리 (2026-04-19) ──────────────────────────
        # schema.chain_id 가 있으면 PipelineOrchestrator 에 완료 보고 후
        # 다음 단계 schema 를 자동 조립해 _active_schema 로 설정.
        # chain_id 가 없으면 기존 단일 미션 흐름 그대로 _active_schema=None.
        _chain_id = getattr(schema, "chain_id", "") if schema else ""
        _next_schema = None
        if _chain_id:
            try:
                from eidos_mission_chain import get_orchestrator
                orch = get_orchestrator(core=self.core)
                if self._report_cb:
                    orch.set_report_callback(self._report_cb)

                # step_history 의 WRITE 결과를 downstream_seeds 로 수확.
                # (LLM이 명시적으로 downstream_seeds 를 채우지 않아도 다음
                #  단계 prompt template 의 placeholder 치환에 쓰이도록.)
                # [P2 2026-04-20] preset 의 input_keys 는 확장자 없이 선언돼 있어
                # (e.g. "research.research_report") WRITE target 이 .md/.txt 이면
                # MissionChain.get_upstream_for_current() 의 집합 매칭에서 누락된다.
                # 확장자 있는 원 key 와 확장자 없는 alias 를 함께 등록해 매칭 성공률
                # 을 확보한다.
                try:
                    seeds = dict(getattr(schema, "downstream_seeds", {}) or {})
                    for _st in (getattr(schema, "step_history", []) or []):
                        if (_st.get("action", "") or "").upper() == "WRITE":
                            _k = (_st.get("target") or f"write_{_st.get('seq', 0)}").strip()
                            _v = _st.get("result") or ""
                            if not _v or not _k:
                                continue
                            if _k not in seeds:
                                seeds[_k] = _v
                            # 확장자 stripped alias (.md / .txt / .json 등)
                            if "." in _k:
                                _k_noext = _k.rsplit(".", 1)[0]
                                if _k_noext and _k_noext not in seeds:
                                    seeds[_k_noext] = _v
                    schema.downstream_seeds = seeds
                except Exception as _e_harv:
                    print(f"  ⚠️ [MissionChain] outputs 수확 실패 (무시): {_e_harv}")

                _chain_success = (task.get("status") == "DONE") and (
                    verifier_result is None or verifier_result.passed
                )
                # [Phase 4-1 2026-04-25] force_advance origin 시 강제 success.
                # task.status DONE 은 이미 세팅됨, verifier_result 는 None (skip).
                # 기존 식으로도 True 가 되지만 의도를 명시 + chain advance 분기에서
                # transient retry 로직이 발동하지 않도록 단정.
                if _force_origin:
                    _chain_success = True
                # [P0-1-c 2026-04-25] 외부 액션 stage 0건 마커가 있으면 force_origin
                # 이어도 강제 False — 거짓 완료 차단. orch 가 retry 또는 사용자 개입으로 분기.
                if task.get("__stage_failed__"):
                    _chain_success = False
                    print(
                        f"  🔴 [ExternalActionGuard] stage_failed 마커 감지 → "
                        f"chain_success 강제 False (reason={task.get('__stage_fail_reason__','')})"
                    )
                # [Phase 2.1 trace 2026-04-25] orchestrator stage advance 진입 / 응답 추적.
                print(
                    f"  🔵 [Chain Advance] orch.on_stage_complete dispatch — "
                    f"chain={_chain_id[:8]} stage='{getattr(schema,'stage','')}' "
                    f"stage_idx={getattr(schema,'stage_index',0)} "
                    f"task_status='{task.get('status','?')}' "
                    f"verifier_passed={None if verifier_result is None else verifier_result.passed} "
                    f"chain_success={_chain_success}"
                )
                _next_schema = await orch.on_stage_complete(
                    _chain_id, schema, success=_chain_success
                )
                print(
                    f"  🔵 [Chain Advance] orch.on_stage_complete returned — "
                    f"next_schema={_next_schema is not None} "
                    f"next_stage='{getattr(_next_schema,'stage','') if _next_schema else ''}' "
                    f"next_stage_idx={getattr(_next_schema,'stage_index',-1) if _next_schema else -1} "
                    f"approved={getattr(_next_schema,'approved',False) if _next_schema else False}"
                )
                # 실패 시 자동 재시도(transient) vs 사용자 개입(structural) 분기 (P1-4)
                if not _chain_success:
                    _reason_parts = []
                    if verifier_result is not None:
                        _reason_parts.append(
                            f"Verifier={getattr(verifier_result, 'fail_type', '') or 'FAIL'}: "
                            f"{(getattr(verifier_result, 'reason', '') or '')[:200]}"
                        )
                    if task.get("status") and task.get("status") != "DONE":
                        _reason_parts.append(f"task_status={task.get('status')}")
                    _failure_reason = " | ".join(_reason_parts) or "알 수 없는 실패"

                    # fail_type 분류:
                    #   "드리프트"  → transient (클릭/nav drift — 재시도로 해결 가능)
                    #   "요령"      → transient (shortcut — 1회 재시도로 자주 해결)
                    #   "속성 미달성" → structural (계획/전략 이슈 — 재시도보다 사용자 개입)
                    #   FAILED_AUTH / task FAIL → structural (사용자 필요)
                    _fail_type = (getattr(verifier_result, "fail_type", "") or "") if verifier_result else ""
                    _transient_fail_types = ("드리프트", "요령")
                    _structural_task_statuses = ("FAILED_AUTH", "FAILED_BLOCKED", "FAILED_ASK_LOOP")
                    _is_transient = (
                        _fail_type in _transient_fail_types
                        and task.get("status") not in _structural_task_statuses
                    )
                    # verifier 결과가 없고 task 상태도 구조적 실패가 아니면 transient 로 가정
                    if verifier_result is None and task.get("status") not in _structural_task_statuses:
                        _is_transient = True

                    _chain_obj = orch.registry.get(_chain_id)
                    _stage_idx = int(getattr(schema, "stage_index", 0) or 0)
                    _used = int((_chain_obj.retries_per_stage or {}).get(_stage_idx, 0)) if _chain_obj else 0
                    _budget = int(getattr(_chain_obj, "retry_budget", 2) or 0) if _chain_obj else 0
                    _stages_total = len(_chain_obj.stages) if _chain_obj else 0

                    # [Phase 3 2026-04-20] register stage 는 1-shot — 자동 재시도 금지.
                    # GPT 조언 "실패 시 1회 → 즉시 중단". 계정 정지 리스크 + 중복 등록
                    # 방지를 위해 실패 후엔 반드시 사용자가 수동으로 상태 확인 후 재개.
                    _stage_name_lc = (getattr(schema, "stage", "") or "").lower()
                    _is_register = (_stage_name_lc == "register")
                    if _is_register and _is_transient:
                        self._report(
                            f"🛑 **[register 1-shot 정책]** 자동 재시도 금지 — "
                            f"계정 리스크 + 중복 등록 방지. "
                            f"실제 등록 상태(공개/미공개/중복 여부)를 사용자가 직접 확인 후 재개 필요."
                        )
                        _is_transient = False

                    # auto-retry 가능 조건: transient + budget 잔여 + chain 객체 있음 + register 아님
                    if _is_transient and _used < _budget and _chain_obj is not None:
                        self._report(
                            f"🔄 **[자동 재시도]** stage='{getattr(schema, 'stage', '')}' "
                            f"({_stage_idx + 1}/{_stages_total}) — "
                            f"transient fail_type='{_fail_type or '미분류'}', "
                            f"예산 {_used}/{_budget} 사용\n"
                            f"└─ 원인: {_failure_reason[:160]}"
                        )
                        try:
                            _retry_schema = await orch.retry_current_stage(_chain_id)
                            if _retry_schema is not None:
                                _next_schema = _retry_schema
                            else:
                                self._report("⚠️ [자동 재시도] 오케스트레이터가 None 반환 → 사용자 개입 필요")
                                _is_transient = False
                        except Exception as _e_ar:
                            print(f"  ⚠️ [MissionChain] 자동 재시도 실패: {_e_ar}")
                            _is_transient = False

                    # 자동 재시도 불가 또는 실패 → 사용자 개입 대기
                    if not (_is_transient and _next_schema is not None):
                        try:
                            self._pending_chain_failure = {
                                "chain_id": _chain_id,
                                "stage_name": getattr(schema, "stage", "") or "",
                                "stage_index": _stage_idx,
                                "reason": _failure_reason,
                                "fail_type": _fail_type,
                                "history_tail": history[-3:] if history else [],
                                "failed_at": time.time(),
                            }
                            _reason_tag = "structural" if not _is_transient else "budget 소진"
                            self._report(
                                f"🔴 **[체인 단계 실패 — 사용자 개입 필요]** "
                                f"stage='{getattr(schema, 'stage', '')}' "
                                f"({_stage_idx + 1}/{_stages_total})\n"
                                f"├─ 분류: {_reason_tag} (fail_type='{_fail_type or '미분류'}')\n"
                                f"├─ 원인: {_failure_reason[:160]}\n"
                                f"├─ 재시도 예산: {_used}/{_budget}\n"
                                f"└─ 다음 입력:  `재시도`  ·  `포기`  ·  `롤백 N` (1-based)"
                            )
                        except Exception as _e_fail:
                            print(f"  ⚠️ [MissionChain] 실패 상태 저장 실패 (무시): {_e_fail}")
            except Exception as _e_orch:
                import traceback
                print(f"  ⚠️ [MissionChain] orchestrator 호출 실패: {_e_orch}")
                traceback.print_exc()

        if _next_schema is not None:
            # [Phase 2.1 trace 2026-04-25] 다음 stage 처리 분기 진입 확인.
            print(
                f"  🔵 [Chain Advance] entering next-stage handler — "
                f"chain={getattr(_next_schema,'chain_id','')[:8]} "
                f"next_stage='{getattr(_next_schema,'stage','')}' "
                f"approved_attr={getattr(_next_schema,'approved',False)}"
            )
            # 체인 다음 단계 — 승인 여부와 무관하게 PENDING task 는 미리 enqueue.
            # auto_approve=True 이면 _active_schema 로 가동, False 이면 _pending_schema
            # 로 보내 Priority 1 승인 경로 재사용 (승인되면 동일 task 를 pick).
            _auto_approved = bool(getattr(_next_schema, "approved", False))
            _next_prompt = (
                getattr(_next_schema, "original_prompt", "")
                or getattr(_next_schema, "what", "")
                or f"[chain {_chain_id[:8]}] 다음 단계"
            )
            try:
                import datetime as _dt
                # [P0-2 2026-04-25] 이전 stage 가 force_advance / stage_failed 였으면
                # 다음 task 에 marker 전파 → DirectTask 환각 차단 + ReAct 가 신뢰 못 함을 인지.
                _prev_force_adv = False
                try:
                    _ch_fa2 = orch.registry.get(_chain_id)
                    if _ch_fa2 is not None:
                        _fa_list2 = _ch_fa2.metadata.get("force_advanced_stages") or []
                        _prev_stage_lc = (getattr(schema, "stage", "") or "").lower()
                        _prev_force_adv = _prev_stage_lc in _fa_list2
                except Exception:
                    pass
                _prev_stage_failed = bool(task.get("__stage_failed__"))
                _next_task_dict = {
                    "task_prompt": _next_prompt,
                    "status":      "PENDING",
                    "source":      "dispatcher",
                    "goal_ref":    (
                        goal_ref
                        or getattr(_next_schema, "long_term_goal_ref", "")
                        or ""
                    ),
                    "priority":    2,
                    "time":        _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "chain_id":    _chain_id,
                    "stage":       getattr(_next_schema, "stage", ""),
                    "stage_index": getattr(_next_schema, "stage_index", 0),
                }
                if _prev_force_adv or _prev_stage_failed:
                    _next_task_dict["__prev_stage_force_advanced__"] = True
                    _next_task_dict["__prev_stage_name__"] = (getattr(schema, "stage", "") or "")
                    print(
                        f"  📌 [Chain Advance] 다음 task 에 prev_force_advanced marker 전파 — "
                        f"prev_stage='{getattr(schema,'stage','')}' "
                        f"next_stage='{getattr(_next_schema,'stage','')}'"
                    )
                self.core.schedule.append(_next_task_dict)
            except Exception as _e_sched:
                print(f"  ⚠️ [MissionChain] 다음 태스크 등록 실패: {_e_sched}")

            # ── 2026-04-19: stage 간 Telegram 승인 게이트 ──────────────────
            # 체인 최초 승인이 모든 stage 로 상속돼 자동 전환되면 사용자가
            # 중간에 끼어들기 힘들고, "다음" 키워드와 자동 전환이 충돌해 race 가
            # 나기 쉽다. auto_approve 여부와 무관하게 stage 전환 직전에
            # Telegram 으로 "다음 작업으로 넘어가시겠습니까?" 확인을 받는다.
            _stage_name = getattr(_next_schema, "stage", "") or "?"
            _stage_idx_next = int(getattr(_next_schema, "stage_index", 0) or 0)
            try:
                _chain_obj_next = orch.registry.get(_chain_id)
                _stages_total_next = len(_chain_obj_next.stages) if _chain_obj_next else 0
            except Exception:
                _stages_total_next = 0

            _gate_ok = _auto_approved  # 폴백 기본값
            _gate_reason = ""
            try:
                from eidos_approval_gateway import ApprovalGateway, GateDecision
                _gw = ApprovalGateway.get()
                _question = (
                    f"다음 작업으로 넘어가시겠습니까?\n"
                    f"단계: [{_stage_idx_next + 1}/{_stages_total_next}] {_stage_name}\n"
                    f"내용: {_next_prompt[:180]}"
                )
                _gate_res = await _gw.ask_user_via_telegram(
                    _question,
                    context={
                        "chain_id":    _chain_id,
                        "stage":       _stage_name,
                        "stage_index": _stage_idx_next,
                        "chain_name":  getattr(_chain_obj_next, "name", "") if _chain_obj_next else "",
                    },
                    action_label="CHAIN_NEXT_STAGE",
                )
                _gate_ok = _gate_res.decision in (GateDecision.APPROVED, GateDecision.AUTO)
                _gate_reason = (_gate_res.reason or _gate_res.decision.value)[:120]
            except Exception as _e_gate:
                # Telegram 미설정 등 → 기존 auto_approve 동작 유지 (안전 폴백)
                print(f"  ⚠️ [MissionChain] stage 간 Telegram 승인 실패 → 기존 정책 폴백: {_e_gate}")
                _gate_ok = _auto_approved
                _gate_reason = f"gate_error: {_e_gate}"

            # [Phase 2.1 trace 2026-04-25] stage 간 게이트 결정 추적 — 다음 stage 가
            # _active_schema 로 set 되는지 / _pending_schema 로 보류되는지 명시.
            print(
                f"  🔵 [Chain Advance] stage gate decision — "
                f"chain={_chain_id[:8]} stage='{_stage_name}' "
                f"gate_ok={_gate_ok} gate_reason='{_gate_reason[:80]}' "
                f"auto_approved={_auto_approved}"
            )
            if _gate_ok:
                self._active_schema = _next_schema
                self._report(
                    f"🔗 **[MissionChain 다음 단계]** stage='{_stage_name}' "
                    f"({_stage_idx_next + 1}/{_stages_total_next}) · 승인됨\n"
                    f"└─ {_next_prompt[:80]}"
                )
            else:
                # 거절/타임아웃 → pending. "다음"/재승인으로 재개 가능.
                self._active_schema = None
                self._pending_schema = _next_schema
                self._report(
                    f"⏸️ **[MissionChain 대기]** stage='{_stage_name}' "
                    f"({_stage_idx_next + 1}/{_stages_total_next}) — {_gate_reason}\n"
                    f"└─ 재개: Telegram 에서 승인하거나 채팅창에 '다음' 입력"
                )
        else:
            # 단일 미션 종료 또는 체인 최종 단계 — 기존과 동일하게 리셋
            self._active_schema = None
        self.core.save_memory()

    async def _react_loop_with_retry(
        self,
        task_prompt: str,
        goal_ref: str,
        task: Dict,
        retry_count: int = 0,
        seed_history: list = None,   # 10-2: 이전 성공 step 맥락 유지
    ) -> None:
        """부분 재실행용 _react_loop 래퍼 — _on_react_done에 retry_count 전달"""
        # [Phase 2.2] 활성 신호 — react 루프 진입
        self._touch_activity()
        schema = self._active_schema
        anchor = self.get_active_anchor() or f"[목표] {task_prompt}"
        anchor = self._apply_chain_anchor(anchor, schema)

        # [P0-2 2026-04-25] 이전 stage force_advance marker → anchor 에 경고 주입.
        # ReAct 가 이전 산출물 (research_report.md 등) 을 신뢰하지 않고 보강하도록 유도.
        if task.get("__prev_stage_force_advanced__"):
            _prev_st = task.get("__prev_stage_name__", "?")
            anchor = (
                f"⚠️ [신뢰 경고] 이전 stage '{_prev_st}' 가 강제 진행으로 끝났다. "
                f"이전 산출물(research_report.md 등)이 비어있거나 부분적일 수 있음. "
                f"필요한 데이터가 없으면 환각으로 채우지 말고 ASK_USER 로 사용자 확인 요청해라.\n\n"
                f"{anchor}"
            )

        # ── general/write/report 타입: ReAct 루프 대신 단일 LLM 직접 호출 ──────
        _DIRECT_TYPES = ("general", "write", "report")
        _loop_task_type = getattr(schema, "task_type", "general") if schema else "general"
        # [P0-1 2026-04-25] 외부 액션 stage(register/deliver/support) 는 task_type='general'
        # 폴백이어도 DirectTask 로 빠지면 안 된다. 실제 등록/발송/대응 액션이 0건인 상태로
        # LLM 텍스트만 뱉고 "완료" 처리되는 거짓 보고가 발생함 (chain=db5ebe78 register stage).
        _EXTERNAL_ACTION_STAGES = {"register", "deliver", "support"}
        _stage_lc = (getattr(schema, "stage", "") or "").lower() if schema else ""
        _is_external_action = _stage_lc in _EXTERNAL_ACTION_STAGES
        # [Autonomous 2026-04-27] 자율 실행 모드 + 외부 액션 stage 진입 시
        # hard-block 키워드 검사. 사용자 사전 승인 카드의 hard_blocks 안에 있는
        # 토큰(결제/송금/회원가입/SNS게시 등) 이 schema.what/how 또는 task_prompt
        # 에 있으면 즉시 run abort + chain abort + telegram 알림.
        # → 자율 모드여도 절대 자동 실행되어선 안 되는 행위 차단.
        if _is_external_action:
            try:
                from eidos_autonomous_runner import AutonomousRunManager as _ARM_HB
                _auto_mgr_hb = _ARM_HB.get()
                _active_hb = _auto_mgr_hb.get_active_run()
                if _active_hb and _auto_mgr_hb.is_task_in_active_scope(task, schema):
                    _hb_blob = " ".join([
                        (getattr(schema, "what", "") or ""),
                        (getattr(schema, "how", "") or ""),
                        (getattr(schema, "where", "") or ""),
                        (task_prompt or ""),
                        (getattr(schema, "original_prompt", "") or ""),
                    ])
                    _hb_kw = _auto_mgr_hb.check_hard_block(_hb_blob)
                    if _hb_kw:
                        _msg = (
                            f"🚨 [Autonomous] HARD-BLOCK 매칭 — '{_hb_kw}' "
                            f"외부 액션 stage='{_stage_lc}' 차단. 자율 실행 abort."
                        )
                        print(_msg)
                        self._report(
                            f"🚨 **[자율 실행 자동 중단]** hard-block 키워드 "
                            f"'{_hb_kw}' 가 외부 액션 stage='{_stage_lc}' 에서 감지됨. "
                            f"안전을 위해 자율 실행 종료 — 사용자 승인이 필요한 작업입니다."
                        )
                        _auto_mgr_hb.abort_active(reason=f"hard_block:{_hb_kw}")
                        # chain abort best-effort
                        try:
                            from eidos_mission_chain import (
                                get_registry as _get_chain_reg_hb,
                                ChainStatus as _CS_HB,
                            )
                            _cid_hb = (
                                (getattr(schema, "chain_id", "") or task.get("chain_id", ""))
                            )
                            if _cid_hb:
                                _reg_hb = _get_chain_reg_hb()
                                _ch_hb = _reg_hb.get(_cid_hb)
                                if _ch_hb and _ch_hb.status == _CS_HB.RUNNING:
                                    _reg_hb.abort(_cid_hb, reason=f"autonomous_hard_block:{_hb_kw}")
                        except Exception as _e_cab:
                            print(f"  ⚠️ [Autonomous] chain abort 실패 (무시): {_e_cab}")
                        # telegram best-effort
                        try:
                            from eidos_telegram_bot import get_bot as _get_bot_hb
                            _bot_hb = _get_bot_hb()
                            if _bot_hb.is_configured():
                                asyncio.ensure_future(
                                    _bot_hb.send_notification(
                                        f"🚨 자율 실행 자동 중단\n"
                                        f"hard-block: <code>{_hb_kw}</code>\n"
                                        f"외부 액션 stage='{_stage_lc}' — 사용자 승인 필요.",
                                        title="🚨 자율 실행 — HARD-BLOCK 감지",
                                    )
                                )
                        except Exception:
                            pass
                        task["status"] = "FAILED"
                        task["fail_reason"] = f"autonomous_hard_block:{_hb_kw}"
                        try:
                            self._remove_task_from_schedule(task)
                        except Exception:
                            pass
                        return
            except Exception as _e_hb:
                # hard-block 검사 실패는 안전 폴백 — 기존 라우팅 로직 그대로
                print(f"  ⚠️ [Autonomous] hard-block 검사 실패 (무시): {_e_hb}")
        # [2026-04-27 WHERE-aware Routing] schema.where + original_prompt 에 외부
        # 도메인이 있으면 task_type='write'/'general'/'report' 이어도 DirectTask 차단.
        # WHERE 도메인 보정 (eidos_mission_schema 2026-04-27) 와 짝을 이루는 라우팅 가드.
        # Why: 보정으로 WHERE 가 도메인이 돼도 task_type 이 write 면 여전히 DirectTask 행 →
        # "브라우저 없이 지식만으로" 프롬프트로 환각 작문.
        # [2026-04-27 Research-intent 안전망] 도메인 토큰 0개 + 연구 산출물 키워드
        # ('기획안/시장조사/벤치마크/case study' 등) 매칭 시에도 DirectTask 차단 —
        # WHERE 자동 주입(mission_schema)이 발동 못 한 캐시/외부-셋 schema 대비.
        _has_ext_src = _has_external_source(schema)
        _has_research_src = _has_research_intent_in_schema(schema)
        _has_action_src = _has_external_action_intent_in_schema(schema)
        if (
            _loop_task_type in _DIRECT_TYPES
            and not _is_external_action
            and not _has_ext_src
            and not _has_research_src
            and not _has_action_src
        ):
            await self._run_direct_llm_task(task_prompt, goal_ref, task, schema, anchor)
            return
        if _is_external_action and _loop_task_type in _DIRECT_TYPES:
            print(
                f"  🔒 [Router HARD] 외부 액션 stage '{_stage_lc}' (task_type='{_loop_task_type}') "
                f"→ DirectTask 차단, ReAct 강제"
            )
            # ReAct 진입 시 LLM 이 실제 외부 액션 없이 끝내려는 시도를 추적
            # (P0-1-b) — _on_react_done 에서 external_action_count==0 이면 stage_failed.
            task["__external_action_stage__"] = _stage_lc
            task.setdefault("__external_action_count__", 0)
        elif _has_ext_src and _loop_task_type in _DIRECT_TYPES:
            _w_preview = (getattr(schema, "where", "") or "")[:50]
            print(
                f"  🔒 [Router HARD] WHERE 외부 도메인 감지 (task_type='{_loop_task_type}', "
                f"where='{_w_preview}') → DirectTask 차단, ReAct 강제"
            )
        elif _has_research_src and _loop_task_type in _DIRECT_TYPES:
            _w_preview = (getattr(schema, "where", "") or "")[:50]
            print(
                f"  🔒 [Router HARD] 연구-기반 산출물 키워드 감지 (task_type='{_loop_task_type}', "
                f"where='{_w_preview}') → DirectTask 차단, ReAct 강제"
            )
        elif _has_action_src and _loop_task_type in _DIRECT_TYPES:
            _w_preview = (getattr(schema, "where", "") or "")[:50]
            print(
                f"  🔒 [Router HARD] 외부-행동 키워드 감지 (task_type='{_loop_task_type}', "
                f"where='{_w_preview}') → DirectTask 차단, ReAct 강제"
            )

        # [Phase 1-ε 2026-05-01] schema.execution_action 있는 단일 task 도 외부 액션
        # 카운터 활성화 — Auto-DONE 가드 / _on_react_done 종료 검증의 입력 신호.
        # chain stage 의 `__external_action_stage__` 마킹(line 3190)과 같은 효과.
        # 차이: stage 명 대신 `schema_external:<action>` prefix 박아 chain stage 와 구분.
        try:
            _ea_at_entry = getattr(schema, "execution_action", None) if schema else None
            if (
                isinstance(_ea_at_entry, dict)
                and _ea_at_entry.get("action")
                and not task.get("__external_action_stage__")
            ):
                task["__external_action_stage__"] = f"schema_external:{_ea_at_entry.get('action')}"
                task.setdefault("__external_action_count__", 0)
                print(
                    f"  🚀 [Phase 1-ε] schema.execution_action 활성 → 외부 액션 카운터 enable "
                    f"(action='{_ea_at_entry.get('action')}')"
                )
        except Exception as _e_ea_entry:
            print(f"  ⚠️ [Phase 1-ε] execution_action entry hook 실패 (무시): {_e_ea_entry}")

        # ── code_build 타입: MissionChain dev 단계 실전 집행 ─────────────────
        # 기획(plan) 산출물을 받아 실제 파일을 디스크에 생성. DCReasoner 스캐폴딩
        # 혹은 assets 문서 생성. 완료 후 _on_react_done 호출로 체인 진행.
        if _loop_task_type == "code_build":
            try:
                from eidos_code_builder import run_code_build_for_schema
                _cb = self._report_cb
                _cb_outputs = await run_code_build_for_schema(
                    schema=schema, task=task, core=self.core, report_cb=_cb,
                )
                # [P2-7] 결과 프리뷰 카드 발행 + 슬롯 세팅
                try:
                    self._last_codebuild_result = {
                        "project_dir": _cb_outputs.get("project_dir", ""),
                        "files": dict(_cb_outputs.get("files", {}) or {}),
                        "entry_file": _cb_outputs.get("entry_file", ""),
                        "mode": _cb_outputs.get("mode", ""),
                        "created_at": time.time(),
                        "chain_id": getattr(schema, "chain_id", "") or "",
                        "stage_index": int(getattr(schema, "stage_index", 0) or 0),
                    }
                    self._emit_codebuild_result_card()
                except Exception as _e_card:
                    print(f"  ⚠️ [CodeBuilder] 결과 카드 발행 실패 (무시): {_e_card}")

                task["status"] = "DONE"
                self._remove_task_from_schedule(task)
                await self._on_react_done(task_prompt, goal_ref, task, [], schema)
            except Exception as e:
                import traceback
                print(f"  ⚠️ [CodeBuilder] 집행 실패: {e}")
                traceback.print_exc()
                task["status"] = "FAILED"
                self._remove_task_from_schedule(task)
            return
        timeout_sec = getattr(schema, "timeout_sec", 600.0) if schema else 600.0
        # [Fix 2026-04-24] 메인 루프와 동일하게 35→60 최소치로 상향.
        max_steps   = max(60, int(timeout_sec / 12))
        started_at  = time.time()
        fail_streak = 0
        # 10-2: 이전 루프 성공 step을 seed로 받아 history 초기화 방지
        history: list = list(seed_history) if seed_history else []
        # [Fix 2026-04-19 A] 부분 재실행 중 WRITE 루프 차단용 카운터.
        # plan.md 같은 산출물에 Verifier FAIL(missing_attr) → retry_attrs 가 반복되면
        # LLM이 같은 파일로 WRITE를 16번까지도 발주하던 증상 차단.
        # [Fix 2026-04-21] task dict 에 persistent 로 저장 — Verifier retry 재진입
        # 때마다 초기화되어 동일 target WRITE 가 매번 3회 허용되던 P0 버그.
        _retry_write_counter: Dict[str, int] = dict(task.get("__retry_write_counter__") or {})
        task["__retry_write_counter__"] = _retry_write_counter
        _RETRY_WRITE_FORCE_DONE = 3   # 동일 target WRITE 3회 → 강제 DONE
        _RETRY_SG_FORCE_DONE    = 2   # Shortening Guard 2회 → 강제 DONE

        self._report(
            f"🔄 **[부분 재실행]** `{task_prompt[:60]}`\n"
            f"└─ retry={retry_count} | max_steps={max_steps}"
        )

        while self._running:
            step_count = len(history) + 1
            # ── [Phase A 2026-04-27] Chain abort propagation 가드 (retry 루프) ──
            if schema is not None and getattr(schema, "chain_id", ""):
                try:
                    from eidos_mission_chain import (
                        ChainStatus as _CS_par,
                        get_orchestrator as _go_par,
                    )
                    _cid_par = getattr(schema, "chain_id", "")
                    _ch_par = _go_par(core=self.core).registry.get(_cid_par)
                    if _ch_par is not None and getattr(_ch_par, "status", None) == _CS_par.ABORTED:
                        _reason_par = (getattr(_ch_par, "failure_reason", "") or "")[:80]
                        print(
                            f"  🛑 [ChainAbortGuard·retry] step {step_count} 차단 — "
                            f"chain {_cid_par[:8]} ABORTED ({_reason_par})"
                        )
                        self._report(
                            f"🛑 **[체인 종료 감지·retry]** step {step_count} 차단 — {_reason_par}"
                        )
                        task["status"] = "FAILED"
                        task["fail_reason"] = f"chain_aborted: {_reason_par}"
                        self._remove_task_from_schedule(task)
                        self._active_schema = None
                        return
                except Exception:
                    pass
            # [Fix 2026-04-24] 이전엔 silent FAILED return 이라 chain 이 한도 초과 시
            # 정체. 보고 + chain stage 일 경우 _on_react_done 으로 부분 산출물 수확
            # + pause_gate(300s auto advance) 경로 연결.
            _elapsed_r = time.time() - started_at
            _exhausted_r = (_elapsed_r > timeout_sec) or (step_count > max_steps)
            if _exhausted_r:
                _reason_r = "타임아웃" if _elapsed_r > timeout_sec else "스텝 한도"
                _detail_r = (f"{_elapsed_r:.0f}s 초과"
                             if _elapsed_r > timeout_sec else f"{max_steps}스텝 초과")
                _chain_id_r = getattr(schema, "chain_id", "") if schema else ""
                self._report(
                    f"⚠️ **[Retry {_reason_r}]** {_detail_r} | history={len(history)}"
                )
                if _chain_id_r and schema is not None:
                    self._remove_task_from_schedule(task)
                    # retry_count+1 을 넘겨 _on_react_done 이 재실행 재진입 대신
                    # pause_gate 로 빠져 체인 auto advance.
                    await self._on_react_done(
                        task_prompt, goal_ref, task, history, schema,
                        _retry_count=retry_count + 1,
                    )
                    return
                task["status"] = "FAILED"
                self._remove_task_from_schedule(task)
                return
            if fail_streak >= 3:
                self._report("[ASK_USER] ❓ 재실행 중 연속 실패. 수동 확인 필요.")
                self._paused_task = task
                task["status"] = "WAITING_USER"
                self._wait_for_user = True
                return

            # general/write/report: 브라우저 불필요 → 빈 obs
            _rwr_task_type = getattr(schema, "task_type", "general") if schema else "general"
            if _rwr_task_type in ("general", "write", "report"):
                obs = {"url": "", "clickable": "", "html_len": 0}
            else:
                obs = await self._observe()
            obs["auth_blacklist"] = list(task.get("__auth_blacklist__") or [])
            obs["fill_blacklist"] = list(task.get("__fill_blacklist__") or [])
            decision = await self._think(
                task_prompt=task_prompt, anchor=anchor,
                obs=obs, history=history[-6:],
                schema=schema, step_no=step_count,
            )
            if not decision:
                fail_streak += 1
                await asyncio.sleep(1.0)
                continue

            action  = decision.get("action", "").upper()
            target  = decision.get("target", "")
            content = decision.get("content", "")
            reason  = decision.get("reason", "")
            self._report(f"🧠 **[Retry Step {step_count}]** [{action}] {(target or content)[:60]}\n└─ {reason[:60]}")

            # ── [Retry Drift Guard 2026-04-27] retry 중 lock-target WRITE 성공 후 ──
            # NAVIGATE/FILL/CLICK/GO_BACK 발사 차단 → DONE 강제. _react_loop_impl 의
            # 동일 가드와 대칭 — _react_loop_with_retry 진입 후에도 같은 자기 위치
            # 망각 루프(검색창 FILL+CLICK 재발사) 가능하므로 양쪽 모두 가드.
            # 회귀: WRITE/READ_PAGE/DONE/ASK_USER 는 계속 허용 (재작성/Verifier OK).
            if (schema is not None
                    and getattr(schema, "_is_retry_run", False)
                    and getattr(schema, "_retry_lock_write_done", False)
                    and action in ("NAVIGATE", "FILL", "CLICK", "GO_BACK", "CLICK_XY", "SCROLL")):
                _drift_lock_t = (getattr(schema, "_retry_target_lock", "") or "").strip()
                _drift_orig_action = action
                print(
                    f"  🛡️ [Retry Drift Guard] retry 중 '{_drift_lock_t[:40]}' WRITE 성공 후 "
                    f"{_drift_orig_action} 차단 → DONE 강제."
                )
                self._report(
                    f"🛡️ **[Retry Drift Guard]** `{_drift_lock_t[:30]}` 보강 완료 후 "
                    f"`{_drift_orig_action}` 차단 — Verifier 진입 강제."
                )
                history.append({
                    "step": step_count,
                    "action": _drift_orig_action,
                    "target": str(target)[:60],
                    "result": f"Retry Drift Guard 차단 — DONE 으로 교체",
                    "verified": False,
                })
                action = "DONE"
                target = ""
                content = ""

            # ── [Patch B — Lazy Reason Detective 2026-04-27, retry 루프] ────────
            # 메인 루프 (line ~4391) 와 동일 패치 — _react_loop_with_retry 에도 적용.
            if (schema is not None
                    and action not in ("DONE", "ASK_USER")
                    and _is_lazy_reason(reason)):
                _lz_n_b_r = int(getattr(schema, "_lazy_reason_count", 0) or 0) + 1
                try:
                    schema._lazy_reason_count = _lz_n_b_r
                except Exception:
                    pass
                print(
                    f"  🛡️ [Lazy Reason Guard / retry] LLM 이유에 게으름 토큰 — "
                    f"{action} 차단 ({_lz_n_b_r}회차) reason='{(reason or '')[:60]}'"
                )
                self._report(
                    f"🛡️ **[Lazy Reason Guard]** retry 루프 — `{action}` 차단 "
                    f"({_lz_n_b_r}/3회차)."
                )
                _curr_how_lz_r = (getattr(schema, "how", "") or "")
                if "[자료 수집 강제" not in _curr_how_lz_r:
                    try:
                        schema.how = (
                            "[자료 수집 강제 — 게으름 차단]\n"
                            "- 외부 검색 의도가 있는 작업은 NAVIGATE → READ_PAGE 단계로 "
                            "자료를 변수에 저장한 후 WRITE 하라.\n"
                            "- '내부 지식으로 작성' / '브라우저 없이' 모드 금지.\n\n"
                        ) + _curr_how_lz_r
                    except Exception:
                        pass
                history.append({
                    "step": step_count,
                    "action": action,
                    "target": str(target)[:60],
                    "result": f"Lazy Reason Guard 차단 ({_lz_n_b_r}/3) — hint 주입",
                    "verified": False,
                })
                if _lz_n_b_r >= 3:
                    self._report(
                        f"⚠️ **[Lazy Reason Guard]** retry 3회 누적 — task 정지."
                    )
                    self._paused_task = task
                    task["status"] = "WAITING_USER"
                    task["__pause_reason__"] = "lazy_reason_3x_retry"
                    self._wait_for_user = True
                    return
                fail_streak += 1
                await asyncio.sleep(0.3)
                continue

            # ── [Patch A — Lazy Write Guard 2026-04-27, retry 루프] ─────────────
            if _is_lazy_write_action(action, schema, task_prompt):
                _lz_n_a_r = int(getattr(schema, "_lazy_write_count", 0) or 0) + 1
                try:
                    schema._lazy_write_count = _lz_n_a_r
                except Exception:
                    pass
                _redirect_url_r = _build_lazy_redirect_url(schema, task_prompt)
                _orig_target_r = target

                # ── [Patch 2026-04-29] retry 루프에도 cap 추가 ──────────────
                # retry 루프 자체 step 한도(60) 가 cap 역할을 못 했음 — 36회 발동
                # 후에야 step 한도로 자연 종료. 메인 루프와 동일 3회 cap.
                # redirect URL placeholder 만 → 즉시 cap 진입.
                if _lz_n_a_r >= 3 or not _redirect_url_r:
                    _why_r = (
                        "redirect URL 추출 실패 (placeholder 만)" if not _redirect_url_r
                        else f"{_lz_n_a_r}회 누적 cap"
                    )
                    print(
                        f"  🛡️ [Lazy Write Guard / retry] WRITE '{(_orig_target_r or '')[:30]}' "
                        f"{_why_r} → task WAITING_USER 전환"
                    )
                    self._report(
                        f"🛡️ **[Lazy Write Guard CAP / retry]** "
                        f"`WRITE {(_orig_target_r or '')[:30]}` {_why_r}. 사용자 개입 필요."
                    )
                    history.append({
                        "step": step_count, "action": "WRITE",
                        "target": str(_orig_target_r)[:60],
                        "result": f"Lazy Write Guard CAP / retry ({_why_r})",
                        "verified": False,
                    })
                    task["status"] = "WAITING_USER"
                    task["__pause_reason__"] = "lazy_write_3x_retry"
                    self._paused_task = task
                    fail_streak += 1
                    await asyncio.sleep(0.3)
                    continue
                # ────────────────────────────────────────────────────────────

                print(
                    f"  🛡️ [Lazy Write Guard / retry] WRITE '{(_orig_target_r or '')[:30]}' 차단 "
                    f"— 자료 0건. NAVIGATE → {_redirect_url_r[:80]} 강제 ({_lz_n_a_r}회차)"
                )
                self._report(
                    f"🛡️ **[Lazy Write Guard]** retry 루프 — `WRITE {(_orig_target_r or '')[:30]}` "
                    f"차단, `NAVIGATE` 강제: `{_redirect_url_r[:60]}`"
                )
                _curr_how_lwa_r = (getattr(schema, "how", "") or "")
                if "[자료 수집 강제" not in _curr_how_lwa_r:
                    try:
                        schema.how = (
                            "[자료 수집 강제 — Lazy Write 차단]\n"
                            "- 자료가 _react_vars 에 0건 상태에서 WRITE 발사가 차단되었다.\n"
                            "- 강제 NAVIGATE 후 검색 결과 페이지를 READ_PAGE 로 변수에 저장하라.\n\n"
                        ) + _curr_how_lwa_r
                    except Exception:
                        pass
                history.append({
                    "step": step_count,
                    "action": "WRITE",
                    "target": str(_orig_target_r)[:60],
                    "result": f"Lazy Write Guard 차단 ({_lz_n_a_r}회) — NAVIGATE 강제 교체",
                    "verified": False,
                })
                action = "NAVIGATE"
                target = _redirect_url_r
                content = ""

            # [P0-1-b 2026-04-25] 외부 액션 stage retry 경로 카운터.
            if task.get("__external_action_stage__") and action in (
                "NAVIGATE", "CLICK", "FILL", "SCROLL", "GO_BACK", "CLICK_XY",
                "TELEGRAM_SEND",  # T2 Step 3 2026-05-01 — 외부 SDK 호출 verb
                "EMAIL_SEND",     # P-ε 2026-05-01 — SMTP 이메일 발송 verb
            ):
                task["__external_action_count__"] = task.get("__external_action_count__", 0) + 1

            if action == "DONE":
                # [Phase 1-ε 2026-05-01] retry 경로 명시 DONE 가드 — 위 메인 경로와 동일 로직.
                _exec_action_done_r = (
                    getattr(schema, "execution_action", None) if schema else None
                ) or {}
                _ext_c_done_r = int(task.get("__external_action_count__", 0) or 0)
                if (
                    isinstance(_exec_action_done_r, dict)
                    and bool(_exec_action_done_r.get("action"))
                    and _ext_c_done_r == 0
                ):
                    _done_blocks_r = int(task.get("__done_blocked_count__", 0) or 0) + 1
                    task["__done_blocked_count__"] = _done_blocks_r
                    if _done_blocks_r <= 3:
                        if not task.get("__done_block_warn_emitted__"):
                            self._report(
                                f"🚫 **[DONE 거부]** schema.execution_action="
                                f"`{_exec_action_done_r.get('action')}` 명시 — "
                                f"외부 액션 0건. 외부 행위 (NAVIGATE/CLICK/FILL) 1건 이상 필요."
                            )
                            print(
                                f"  🚫 [Phase 1-ε retry] 명시 DONE 거부 ({_done_blocks_r}/3) — "
                                f"execution_action='{_exec_action_done_r.get('action')}'"
                            )
                            task["__done_block_warn_emitted__"] = True
                        continue
                    else:
                        print(
                            f"  ⚠️ [Phase 1-ε retry] DONE 차단 한도 초과 ({_done_blocks_r}회) → DONE 통과"
                        )
                await self._on_react_done(task_prompt, goal_ref, task, history, schema, _retry_count=retry_count)
                return
            if action == "ASK_USER":
                question = target or content
                self._report(f"[ASK_USER] ❓ {question}\n└─ 완료 후 **⚡ 자율 실행** 눌러주세요.")
                self._paused_task = task
                task["status"] = "WAITING_USER"
                self._wait_for_user = True
                return

            # [P-α S4 2026-05-01] step latency 측정 — _STEP_LATENCY_QUEUE 에 push.
            _step_t0 = time.perf_counter()
            act_result = await self._act(action, target, content, schema, reason=reason)
            try:
                _STEP_LATENCY_QUEUE.append(
                    (time.time(), action, time.perf_counter() - _step_t0)
                )
            except Exception:
                pass
            verified, verify_msg = await self._verify_step(action, target, act_result, schema)

            # T1 (CoT): retry 루프에서도 thought 보존
            history.append({
                "step": step_count, "action": action,
                "target": target[:60], "result": act_result[:100],
                "verified": verified, "url": obs.get("url", ""),
                "thought": (reason or "")[:200],
            })
            if schema:
                schema.add_step(action, target, act_result, obs.get("url", ""), thought=(reason or "")[:200])

            # ── [Fix 2026-04-19 A] Retry WRITE 루프 조기 탈출 ──────────────
            # 같은 target에 WRITE가 N회 반복되거나 Shortening Guard가 연속으로
            # 터지면 더 이상 진전 없음으로 간주하고 _on_react_done 으로 강제 이탈.
            # 내용이 800자 이상이면 내용-계열 HOW_MUCH 속성을 자동 달성 처리해
            # Verifier 재FAIL → 재retry 의 무한 루프를 끊는다.
            if action == "WRITE" and (target or "").strip():
                _w_key = (target or "").strip().lower()
                _is_sg = "Shortening Guard" in (act_result or "")
                _retry_write_counter[_w_key] = _retry_write_counter.get(_w_key, 0) + 1
                if _is_sg:
                    _sg_key = f"__sg__{_w_key}"
                    _retry_write_counter[_sg_key] = _retry_write_counter.get(_sg_key, 0) + 1
                _hit_count = _retry_write_counter[_w_key] >= _RETRY_WRITE_FORCE_DONE
                _hit_sg    = _retry_write_counter.get(f"__sg__{_w_key}", 0) >= _RETRY_SG_FORCE_DONE
                if _hit_count or _hit_sg:
                    _cur_val = ""
                    if schema is not None and hasattr(schema, "_react_vars"):
                        _cur_val = schema._react_vars.get((target or "").strip(), "") or ""
                    _promoted = 0
                    if schema is not None and len(_cur_val) >= 800:
                        _CONTENT_KWS = (
                            "목록", "리스트", "list", "분석", "analysis",
                            "보고서", "report", "내용", "정확성", "요약", "summary",
                            "설명", "기획", "plan", "draft",
                            # plan.md 의 "상품명 후보 개수 / 타겟 고객 섹션 / USP /
                            # 가격 근거 / 컨셉 개요" 같은 구조 명칭도 내용-계열로 인정.
                            "섹션", "포함", "후보", "컨셉", "개요",
                            "타겟", "usp", "가격", "근거", "본문", "초안",
                            "구성", "항목", "상품명",
                        )
                        for _attr in schema.attributes:
                            if getattr(_attr, "achieved", False):
                                continue
                            # [Fix 2026-04-21] 본문 키워드/N개 검사 우선
                            if self._auto_check_keyword_attr(_attr, _cur_val):
                                _promoted += 1
                                continue
                            if self._auto_check_count_attr(_attr, _cur_val):
                                _promoted += 1
                                continue
                            _nl = (getattr(_attr, "name", "") or "").lower()
                            if any(kw in _nl for kw in _CONTENT_KWS):
                                _attr.achieved = True
                                _attr.evidence = (
                                    f"Retry WRITE {_retry_write_counter[_w_key]}회 "
                                    f"반복({len(_cur_val)}자) — 내용-계열 속성 자동 인정"
                                )
                                _promoted += 1
                    self._report(
                        f"🛑 **[Retry WRITE 루프 차단]** '{(target or '')[:40]}' "
                        f"write={_retry_write_counter[_w_key]}회 · "
                        f"SG={_retry_write_counter.get(f'__sg__{_w_key}', 0)}회 — "
                        f"내용계열 속성 {_promoted}개 자동 인정, 자동 DONE 전환 (Verifier 재검증)."
                    )
                    await self._on_react_done(
                        task_prompt, goal_ref, task, history, schema,
                        _retry_count=retry_count,
                    )
                    return

            if verified:
                fail_streak = 0
                self._report(f"✅ **[Retry Step {step_count}]** [{action}] {verify_msg[:60]}")
                if schema:
                    self._auto_check_attributes(schema, action, act_result, obs)
                    if schema.all_attributes_achieved:
                        await self._on_react_done(task_prompt, goal_ref, task, history, schema, _retry_count=retry_count)
                        return
            else:
                fail_streak += 1
                self._report(f"⚠️ **[Retry Step {step_count} 실패]** {verify_msg[:60]}")
                if action in {"NAVIGATE", "FILL", "CLICK", "CHECK_LOGIN"}:
                    await asyncio.sleep(2.0)

    # ══════════════════════════════════════════════════════════════════════
    # (기존 레거시 메서드 — 하위 호환성 유지)
    # ══════════════════════════════════════════════════════════════════════

    async def _execute_remaining(
        self,
        plan: ActionPlan,
        task: Dict,
        steps_to_run: list,
        task_prompt: str,
        goal_ref: str,
        total: int,
    ):
        """
        [v2 - Reactive Loop] 관찰→판단→행동 방식.

        기존: LLM이 plan 생성 시 CSS 셀렉터를 미리 추측 → 실패
        변경: 매 step 실행 후 현재 화면을 읽고, LLM에게 "다음에 뭘 해야 해?"를 물어봄.

        브라우저 작업(NAVIGATE/FILL/CLICK) 이후에는 반드시 페이지를 다시 읽고
        LLM에게 현재 상태를 보여준 뒤 다음 행동을 결정.
        비브라우저 작업(WRITE/REPORT)은 기존대로 plan에서 순서대로 실행.
        """
        failed = False
        _BROWSER_ACTIONS = {"NAVIGATE", "FILL", "CLICK", "CHECK_LOGIN"}
        _reactive_mode = False  # 브라우저 작업 시작되면 reactive 모드 전환

        for step in steps_to_run:
            if not self._running:
                break

            step.status    = "RUNNING"
            step.started_at = time.time()

            done_count = sum(1 for s in plan.steps if s.status == "DONE")
            self._report(
                f"🔄 **[중간보고]** `{task_prompt[:50]}`\n"
                f"├─ Step {step.step_no}/{total}: **[{step.action}]** "
                f"{(step.target or step.content)[:60]}\n"
                f"├─ 완료: {done_count}/{total}\n"
                f"└─ 상태: 실행 중..."
            )

            # ASK_USER 감지 → 즉시 일시정지
            if step.action == "ASK_USER":
                try:
                    result = await self._execute_step(step, plan)
                    step.status   = "DONE"
                    step.result   = result
                    step.ended_at = time.time()
                except Exception as e:
                    step.status = "DONE"
                    step.result = str(e)
                # [Phase 2] finalize — 이 경로는 보통 LOW라 _pending_rid 없지만 안전망.
                await self._try_finalize_approval(step, step.result)

                remaining_after = [s for s in plan.steps if s.status == "PENDING"]
                if remaining_after:
                    self._paused_plan = plan
                    self._paused_task = task
                    task["status"] = "WAITING_USER"
                    self._wait_for_user = True
                    self._current_plan = None
                    self._report(
                        f"⏸️ **[일시정지]** `{task_prompt[:50]}`\n"
                        f"└─ 완료되면 **⚡ 자율 실행** 을 다시 누르면 이어집니다."
                    )
                    return
                break

            _eff_to = self._compute_step_timeout(step)
            try:
                result = await asyncio.wait_for(
                    self._execute_step(step, plan),
                    timeout=_eff_to
                )
                step.result   = result or "완료"
                step.ended_at = time.time()

                # ── [검증] step 결과에 실패 표시 확인 ──────────────────
                if result and ("⚠️" in result or "실패" in result[:20]):
                    step.status = "FAILED"
                    self._report(
                        f"❌ **[단계 실패]** Step {step.step_no}/{total}\n"
                        f"├─ 액션: [{step.action}] {(step.target or step.content)[:50]}\n"
                        f"├─ 결과: {step.result[:120]}\n"
                        f"└─ 브라우저 작업 실패 - plan 중단."
                    )
                    # [Phase 2] break 를 try 내부에서 발동하면 finalize 를 못 타므로
                    # failed 플래그만 set, break 는 try/except 밖에서 수행.
                    if step.action in _BROWSER_ACTIONS:
                        failed = True
                else:
                    step.status = "DONE"

                done_count = sum(1 for s in plan.steps if s.status == "DONE")
                if step.status == "DONE":
                    self._report(
                        f"✅ **[단계 완료]** Step {step.step_no}/{total}\n"
                        f"├─ 액션: [{step.action}] {(step.target or step.content)[:50]}\n"
                        f"├─ 결과: {step.result[:80]}"
                    )

                # ── [Reactive Mode] 브라우저 작업 후 화면 관찰 → 다음 행동 결정 ──
                if step.action in _BROWSER_ACTIONS and step.status == "DONE":
                    _reactive_mode = True
                    # 남은 PENDING step 중 브라우저 액션이 있으면 reactive 판단
                    remaining = [s for s in plan.steps if s.status == "PENDING"]
                    next_browser = next((s for s in remaining if s.action in _BROWSER_ACTIONS), None)

                    if next_browser:
                        # 현재 화면 읽기
                        await asyncio.sleep(2.0)
                        page_snapshot = ""
                        try:
                            from execution_module import get_browser_content as _gc
                            page_snapshot = await _gc()
                            if len(page_snapshot) > 2000:
                                page_snapshot = page_snapshot[:2000]
                            plan.variables["__last_page__"] = page_snapshot
                        except Exception:
                            pass

                        if page_snapshot:
                            # LLM에게 다음 행동 결정 요청
                            reactive_result = await self._reactive_decide_next(
                                task_prompt, page_snapshot, next_browser, plan
                            )
                            if reactive_result:
                                # LLM이 셀렉터/행동을 업데이트
                                next_browser.target = reactive_result.get("target", next_browser.target)
                                next_browser.content = reactive_result.get("content", next_browser.content)
                                if reactive_result.get("action"):
                                    next_browser.action = reactive_result["action"].upper()
                                print(f"  🔍 [Reactive] Step {next_browser.step_no} 업데이트: "
                                      f"{next_browser.action} → {next_browser.target[:50]}")

            except asyncio.TimeoutError:
                step.status   = "FAILED"
                step.result   = "타임아웃"
                step.ended_at = time.time()
                self._report(f"⏰ **[타임아웃]** Step {step.step_no} - {int(_eff_to)}초 초과")

            except Exception as e:
                step.status   = "FAILED"
                step.result   = str(e)[:100]
                step.ended_at = time.time()
                self._report(f"❌ **[단계 실패]** Step {step.step_no}: {str(e)[:80]}")
                if step.action in _BROWSER_ACTIONS:
                    failed = True

            # [Phase 2] plan 기반 경로 finalize — 성공/타임아웃/실패 모두 한 곳에서.
            # step.result 를 detail 로 사용. _pending_rid 없으면 no-op.
            await self._try_finalize_approval(step, step.result or "")

            # ── [제어실 Phase 1] step 완료 attribution 기록 ─────────────
            # plan.metadata["goal_id"] 를 키로 goal_attribution.jsonl 에 append.
            # Phase 3 rollup 이 주간/월간 집계에 소비.
            try:
                from eidos_core_goal_registry import attribute_step as _attr_step, AD_HOC_SLUG as _AH
                _plan_meta = plan.metadata or {}
                _goal_slug = str(_plan_meta.get("goal_id") or "") or _AH
                _artifact = ""
                if step.action == "WRITE" and step.target:
                    _artifact = step.target
                _attr_step(_goal_slug, {
                    "task_id":        plan.task_id,
                    "chain_id":       str(_plan_meta.get("chain_id") or ""),
                    "step_no":        step.step_no,
                    "action":         step.action,
                    "status":         step.status,
                    "artifact_path":  _artifact,
                    "result_preview": (step.result or "")[:120],
                    "rationale":      (step.reason or "")[:120],
                })
            except Exception as _ae:
                print(f"  [GoalAttr] step attribution 실패 (무시): {_ae}")

            # except 에서 break 한 경우 동일 동작 유지
            if failed and step.action in _BROWSER_ACTIONS:
                break

        # 완료/실패 처리
        plan.status    = "FAILED" if failed else "DONE"
        task["status"] = plan.status

        # ── schedule에서 완료/실패 task 제거 (무한 반복 방지) ────────────────
        self._remove_task_from_schedule(task)

        if not failed:
            done_count = sum(1 for s in plan.steps if s.status == "DONE")
            self._report(
                f"🎉 **[작업 완료]** `{task_prompt[:60]}`\n"
                f"├─ {done_count}/{total}단계 성공적으로 완료\n"
                f"├─ 목표: {goal_ref[:50]}\n"
                f"└─ 결과가 WorldModel에 기록되었습니다."
            )
            self._record_completion(task_prompt, goal_ref, plan)

            # ── [GoalPlan 연동] browser_action step 결과 기록 ─────────
            _plan_step_id = task.get("plan_step_id")
            if _plan_step_id:
                try:
                    _gse = getattr(self.core, "goal_stack_evaluator", None)
                    _pm = getattr(_gse, "plan_manager", None) if _gse else None
                    if _pm:
                        # REPORT step의 결과를 artifact_summary로 사용
                        _summary = ""
                        for s in plan.steps:
                            if s.action == "REPORT" and s.result:
                                _summary = s.result[:500]
                                break
                        if not _summary:
                            _summary = f"브라우저 작업 완료: {task_prompt[:200]}"
                        _pm.record_step_result(
                            goal_name=goal_ref,
                            step_id=_plan_step_id,
                            success=True,
                            artifact_summary=_summary,
                        )
                        # LongTermGoal progress 동기화
                        _goal = next(
                            (g for g in self.core.long_term_goals if g.name == goal_ref),
                            None
                        )
                        if _goal:
                            _goal.progress = _pm.get_plan_progress(goal_ref)
                            _goal.completed = _pm.is_plan_completed(goal_ref)
                            if _goal.completed:
                                _goal.status = "COMPLETED"
                            print(f"  📈 [GoalPlan] '{goal_ref[:25]}' step '{_plan_step_id}' "
                                  f"완료 → progress={_goal.progress:.0%}")
                except Exception as _e_gp:
                    print(f"  ⚠️ [GoalPlan] ActionDispatcher 결과 기록 실패: {_e_gp}")
            # ──────────────────────────────────────────────────────────────
        else:
            self._report(
                f"⚠️ **[작업 중단]** `{task_prompt[:60]}`\n"
                f"└─ 일부 단계 실패. 다음 순찰 주기에 재시도합니다."
            )

            # ── [GoalPlan 연동] browser_action step 실패 기록 ─────────
            _plan_step_id = task.get("plan_step_id")
            if _plan_step_id:
                try:
                    _gse = getattr(self.core, "goal_stack_evaluator", None)
                    _pm = getattr(_gse, "plan_manager", None) if _gse else None
                    if _pm:
                        _pm.record_step_result(
                            goal_name=goal_ref,
                            step_id=_plan_step_id,
                            success=False,
                            error=f"ActionDispatcher 실행 실패: {task_prompt[:200]}",
                        )
                except Exception as _e_gp:
                    print(f"  ⚠️ [GoalPlan] ActionDispatcher 실패 기록 오류: {_e_gp}")
            # ──────────────────────────────────────────────────────────────

        self._current_plan = None
        self.core.save_memory()

    # ── 계획 수립 ─────────────────────────────────────────────────────────────

    async def _build_plan(
        self,
        task_prompt: str,
        goal: str,
        task_dict: Dict,
    ) -> Optional[ActionPlan]:
        """DCReasoner 대신 LLM에게 ActionPlan JSON을 직접 요청"""

        S = self._get_situation_summary()

        # ── MissionSchema 앵커 주입 (승인된 스키마가 있으면 최우선 컨텍스트) ──
        _anchor = self.get_active_anchor()
        # [Fix 2026-04-21 Phase 2-B] chain stage 이면 _build_plan 단계에서도 topic
        # anchor 를 앞머리에 고정. ReAct loop 시작 전 planner 단계에서도 주제 유지.
        _anchor = self._apply_chain_anchor(_anchor or "", self._active_schema)
        if _anchor:
            S = _anchor + "\n\n[현재 상황]\n" + S

        # ── [Phase 3] DNA/DC 사전 점검 주입 (ApprovalGateway 패턴 이식) ──
        # 사후 리스크 상향이 아니라 "계획 이전에" 유사 실패/위험 표현을
        # planner 프롬프트 컨텍스트에 꽂아, LLM 이 우회/보수 경로를 택하도록.
        # 토글: eidos_settings.json → loop_closure.phase3_dna_to_planner (default True)
        _precheck_enabled = True
        try:
            from eidos_approval_gateway import _loop_closure_flags as _lcf
            _precheck_enabled = bool(_lcf().get("phase3_dna_to_planner", True))
        except Exception:
            pass

        if _precheck_enabled:
            _precheck_lines = []
            _probe = f"{goal} {task_prompt}".strip()
            try:
                from eidos_situation_dna import similar_past_failures
                _info = similar_past_failures(
                    query=_probe,
                    top_goal=goal,
                    k=5,
                    min_similarity=0.15,
                )
                if int(_info.get("count", 0)) > 0:
                    _samples = _info.get("samples", [])[:3]
                    _sample_str = "; ".join(
                        f"sim={s.get('score', 0):.2f} | {s.get('query', '')[:80]}"
                        for s in _samples
                    )
                    _precheck_lines.append(
                        f"[과거 유사 실패 {_info['count']}건, max_sim={_info.get('max_similarity', 0):.2f}]"
                        f"\n  샘플: {_sample_str}"
                        f"\n  → 유사 경로 회피하거나 추가 검증 단계를 포함하라."
                    )
            except Exception as _dna_e:
                print(f"  [PlanPrecheck] DNA 조회 실패 (무시): {_dna_e}")

            try:
                from eidos_dc_reasoner import extract_risk_hint
                _hint = extract_risk_hint(_probe)
                if _hint.get("escalate"):
                    _matched = (_hint.get("matched") or [])[:3]
                    _precheck_lines.append(
                        f"[DC 위험 패턴 감지 sev={_hint.get('level')}, matched={_matched}]"
                        f"\n  → 해당 표현과 관련된 step 은 ASK_USER 또는 축약 버전으로 대체 고려."
                    )
            except Exception as _dc_e:
                print(f"  [PlanPrecheck] DC hint 실패 (무시): {_dc_e}")

            if _precheck_lines:
                _precheck_block = "\n\n[사전 점검 — DNA/DC]\n" + "\n".join(_precheck_lines)
                S = S + _precheck_block
                try:
                    print(f"  [ActionDispatcher] Plan precheck 주입: {len(_precheck_lines)}건")
                except Exception:
                    pass

        # ── 현재 브라우저 URL 및 크몽 관련성 판단 ───────────────────────────
        current_url = self._get_current_browser_url()
        is_kmong_url = "kmong.com" in current_url

        # task_prompt에 크몽 관련 키워드가 있을 때만 크몽 힌트 주입
        _kmong_keywords = ["크몽", "kmong", "상품 등록", "상세페이지", "판매자", "서비스 등록", "gig"]
        is_kmong_task = any(kw.lower() in task_prompt.lower() for kw in _kmong_keywords)

        kmong_hint = ""
        if is_kmong_task or is_kmong_url:
            if is_kmong_url:
                kmong_hint = f"""
[현재 브라우저 상태 - 크몽 열려있음]
URL: {current_url}
→ 로그인 여부 불명. 첫 단계에서 반드시 CHECK_LOGIN으로 확인할 것.
→ 로그인 됐으면 현재 URL 기준으로 탐색 시작 (NAVIGATE 생략 가능).
→ 로그인 안 됐으면 ASK_USER로 사용자에게 로그인 요청 후 대기."""
            else:
                kmong_hint = f"""
[크몽 관련 작업 - 탐색 방법]
현재 브라우저: {current_url or '알 수 없음'}
⚠️ 절대 URL을 직접 추측하여 NAVIGATE하지 마라. 404가 발생한다.
→ 크몽 홈(https://kmong.com)으로 NAVIGATE한 후,
→ CHECK_LOGIN으로 로그인 확인,
→ READ_PAGE로 현재 화면을 읽고,
→ 화면에 보이는 메뉴/링크를 CLICK하여 원하는 페이지로 이동하라.
→ 예: "내 서비스" 메뉴 클릭 → 서비스 목록 → "편집하기" 클릭
→ CSS 셀렉터를 추측하지 마라. READ_PAGE 후 화면 내용을 보고 결정하라."""

        prompt = f"""당신은 자율 실행 에이전트 EIDOS의 ActionPlanner입니다.
아래 [실행할 작업]을 수행하기 위한 단계별 계획을 JSON으로 작성하라.

[중요 - 반드시 준수]
- [실행할 작업]이 최우선이다. 이것만 수행하면 된다.
- [최상위 목표 G]는 배경 참고용일 뿐, 계획에 직접 반영하지 말 것.
- CHECK_LOGIN은 크몽 작업에서만 사용 (다른 사이트는 사용 금지). NAVIGATE/ASK_USER는 모든 사이트에서 사용 가능.
- 작업과 무관한 단계를 추가하지 말 것.
- "내가 직접 할 수 없다", "승준이가 직접 해야 한다"는 절대 금지. 너는 브라우저를 직접 조작할 수 있다.
- 크몽/상세페이지 관련 작업이면 반드시 NAVIGATE → FILL/CLICK 단계를 포함하라. 글만 작성하고 끝내지 마라.
- "추가해", "수정해", "올려" 등 실행 동사가 있으면 반드시 브라우저 액션(NAVIGATE+FILL)으로 실제 실행하라.

[실행할 작업 - 이것만 수행]
{task_prompt}

[최상위 목표 G - 참고만]
{goal}

[현재 상황 S]
{S}{kmong_hint}

[사용 가능한 액션 타입]
- NAVIGATE     : 브라우저를 URL로 이동. target=URL
- READ_PAGE    : 현재 페이지 텍스트 읽기. target=var_name (이후 WRITE에서 {{var_name}}으로 참조)
- CHECK_LOGIN  : 크몽 로그인 상태 확인 (크몽 작업 시에만 사용)
- FILL         : 현재 페이지 에디터에 텍스트 삽입. content=삽입할 전체 텍스트
- CLICK        : JS로 버튼/링크 클릭. target=CSS셀렉터
- WRITE        : LLM으로 텍스트 생성. content=생성 프롬프트, target=파일명(.md 권장, 예: "report.md"). 최종 산출물은 반드시 확장자 포함 파일명 — 순수 변수명은 메모리에만 남아 사라진다
- WAIT         : 대기. content=초수(숫자 문자열)
- ASK_USER     : 사용자에게 질문 1개 (꼭 필요한 정보만, 최대 1회)
- REPORT       : 사용자에게 결과 보고. content=보고 메시지
- DONE         : 작업 완료 선언

[규칙]
1. 최대 {self.MAX_STEPS}단계 이내
2. 텍스트/분석 생성은 WRITE → 결과를 REPORT로 전달
3. 마지막 단계는 반드시 DONE
4. JSON만 출력, 설명 없음
5. 웹페이지 내용 분석이 필요하면 반드시 NAVIGATE → WAIT(2) → READ_PAGE → WRITE 순서
   READ_PAGE 없이 상상으로 분석 내용을 작성하는 것은 금지
6. URL은 [실행할 작업]에 명시된 사이트(예: namu.wiki, kmong.com)의 홈 또는 검색 URL로 NAVIGATE한 뒤, 세부 페이지는 화면에서 CLICK으로 이동하라. 명시된 사이트가 없으면 작업에 적합한 공개 사이트를 선택하라. 임의의 깊은 경로(/abc/123 등)를 추측해 입력하지 마라.
7. CSS 셀렉터를 추측하지 마라. CLICK의 target에는 "화면에서 확인 후 결정"이라고 쓰면 Reactive Mode가 자동 결정
8. 정보가 부족하면 ASK_USER로 사용자에게 질문하라. 추측으로 진행하지 마라
9. 작업이 실제로 완료되었는지 확인 없이 DONE하지 마라

[출력 형식 예시 - 일반 작업]
{{
  "steps": [
    {{"step_no": 1, "action": "WRITE",  "target": "result", "content": "작업 내용 생성 프롬프트"}},
    {{"step_no": 2, "action": "REPORT", "target": "",        "content": "{{result}}"}},
    {{"step_no": 3, "action": "DONE",   "target": "",        "content": "완료"}}
  ]
}}

[출력 형식 예시 - 크몽 브라우저 작업 (FAQ를 상세페이지에 추가하는 경우)]
{{
  "steps": [
    {{"step_no": 1,  "action": "WRITE",      "target": "faq_content",  "content": "미용실 CRM FAQ 10개를 작성해줘"}},
    {{"step_no": 2,  "action": "NAVIGATE",   "target": "https://kmong.com", "content": ""}},
    {{"step_no": 3,  "action": "CHECK_LOGIN","target": "",             "content": ""}},
    {{"step_no": 4,  "action": "WAIT",       "target": "",             "content": "2"}},
    {{"step_no": 5,  "action": "READ_PAGE",  "target": "home_page",   "content": ""}},
    {{"step_no": 6,  "action": "CLICK",      "target": "화면에서 확인 후 결정", "content": "내 서비스 또는 서비스 관리 메뉴"}},
    {{"step_no": 7,  "action": "WAIT",       "target": "",             "content": "2"}},
    {{"step_no": 8,  "action": "READ_PAGE",  "target": "service_page", "content": ""}},
    {{"step_no": 9,  "action": "CLICK",      "target": "화면에서 확인 후 결정", "content": "편집하기 버튼"}},
    {{"step_no": 10, "action": "WAIT",       "target": "",             "content": "2"}},
    {{"step_no": 11, "action": "FILL",       "target": "",             "content": "{{faq_content}}"}},
    {{"step_no": 12, "action": "CLICK",      "target": "화면에서 확인 후 결정", "content": "저장 또는 등록 버튼"}},
    {{"step_no": 13, "action": "REPORT",     "target": "",             "content": "FAQ 추가 완료"}},
    {{"step_no": 14, "action": "DONE",       "target": "",             "content": "완료"}}
  ]
}}
※ CLICK의 target은 READ_PAGE로 화면을 읽은 후 Reactive Mode에서 자동 결정됨. 추측하지 마라.
※ CHECK_LOGIN 후 로그인이 안 됐으면 자동으로 로그인 버튼 클릭 시도 후 ASK_USER로 로그인 요청함. 별도 처리 불필요.
※ 계획은 DONE으로 반드시 끝내야 하며 중간에 잘리면 안 됨."""

        try:
            # [Fix] response_mime_type 제거 → chat 액션으로 전송 (code_edit는 서버 토큰 제한 있음)
            raw = await get_llm_response_async(
                prompt,
                max_tokens=16384,
            )
            # [Fix] robust_json_parse 4단계 엔진으로 파싱 (마크다운, 잘림, 파이썬 리터럴 모두 처리)
            data = _robust_json_parse(raw)
            steps_raw = data.get("steps", [])
            if not steps_raw:
                # fallback: 단계가 리스트로 직접 반환된 경우
                if isinstance(data, list):
                    steps_raw = data
                else:
                    print(f"  [ActionDispatcher] 계획 steps 비어있음. raw={raw[:300]}")
                    return None
        except Exception as e:
            print(f"  [ActionDispatcher] 계획 수립 파싱 실패: {e}\n  raw={raw[:300] if 'raw' in dir() else 'N/A'}")
            return None

        import uuid
        plan = ActionPlan(
            task_id=str(uuid.uuid4())[:8],
            task_prompt=task_prompt,
            goal=goal,
        )
        for s in steps_raw:
            plan.steps.append(ActionStep(
                step_no = int(s.get("step_no", 0)),
                action  = s.get("action", "REPORT").upper(),
                target  = s.get("target", ""),
                content = s.get("content", ""),
            ))

        # [Phase B-1] ValueEngine 으로 step 별 cost_estimate 자동 채움
        # (Approval Gateway Budget 에 반영 — 실행 시 step.cost_estimate 가 그대로 전달됨)
        try:
            from eidos_value_engine import fill_step_costs as _fsc
            _fsc(plan.steps)
        except Exception as _vce:
            print(f"  [ValueEngine] step cost 주입 실패 (무시): {_vce}")

        # ── [제어실 Phase 1] goal_id 자동 태깅 ────────────────────────
        # task_prompt 를 현재 활성 WHY 중 하나 또는 AD_HOC_SLUG 로 분류 후
        # plan.metadata 와 모든 step.metadata 에 전파. 실패 시 ad_hoc 폴백.
        # 우선순위:
        #   1) task_dict["metadata"]["goal_id"]  (MissionChain 상속 경로)
        #   2) self._active_schema.long_term_goal_ref  (MissionSchema 앵커)
        #   3) LLM 1패스 분류 (없거나 검증 실패 시 ad_hoc)
        try:
            from eidos_core_goal_registry import classify_task_goal as _classify_goal
            _hint = ""
            try:
                _td_meta = (task_dict or {}).get("metadata") if isinstance(task_dict, dict) else None
                if isinstance(_td_meta, dict):
                    _hc = str(_td_meta.get("goal_id") or "").strip()
                    if _hc:
                        _hint = _hc
            except Exception:
                pass
            if not _hint:
                try:
                    _sch = getattr(self, "_active_schema", None)
                    if _sch is not None:
                        _hint = str(getattr(_sch, "long_term_goal_ref", "") or "")
                except Exception:
                    pass
            _goal_slug, _rationale = await _classify_goal(task_prompt, hint_slug=_hint)
            plan.metadata["goal_id"] = _goal_slug
            plan.metadata["goal_rationale"] = _rationale
            for _st in plan.steps:
                _st.metadata["goal_id"] = _goal_slug
            print(f"  [GoalTag] task={plan.task_id} goal_slug={_goal_slug} ({str(_rationale)[:60]})")
        except Exception as _gte:
            print(f"  [GoalTag] 자동 태깅 실패 (무시): {_gte}")

        return plan

    # ── 단계 실행 ─────────────────────────────────────────────────────────────

    def _compute_step_timeout(self, step) -> float:
        """
        [Fix] Approval Gateway가 Telegram 승인 대기에 들어갈 가능성이 있는 step은
        STEP_TIMEOUT(30s) 대신 gateway 대기 시간(기본 900s)+버퍼로 확장.
        LOW 리스크(AUTO 통과) step은 기존 STEP_TIMEOUT 그대로.

        [Fix 2026-04-22] WRITE/READ_PAGE 등 내부 LLM/대용량 IO 호출이 있는 action은
        STEP_TIMEOUT(30s) 보다 내부 호출이 더 길어서 외곽 wait_for 가 먼저 죽이는
        문제 발생. WRITE 는 analyst LLM 120s + 이어쓰기 1회 가능 → 180s,
        READ_PAGE 는 innerText/HTML 덩어리 + 저장 → 60s 로 확장.

        [Fix 2026-05-03] 2-pass classification race — _compute_step_timeout 시점에
        classify_only 가 LOW 반환 → action_floor 180s 만 적용 → 실제 _execute_step
        진행 중 patrol/dna/dc/drift/policy boost 로 HIGH 승격 → Telegram 승인 대기
        진입 → 사용자 응답 도달 전에 외곽 wait_for 가 180s 에서 죽임. WRITE/EMAIL_SEND/
        TELEGRAM_SEND/NAVIGATE/CLICK/FILL 같은 approval 가능성 있는 action 들은 risk
        무관하게 gateway timeout 확장을 적용해 race 회피.
        """
        # ── Action 기반 최소 timeout (Gateway 확장과 max) ──
        _action = (getattr(step, "action", "") or "").upper()
        _action_floor = {
            "WRITE":      180.0,   # analyst LLM 120s + 이어쓰기 여유
            "READ_PAGE":   60.0,   # innerText/HTML 덩어리 IO
            "THINK":      120.0,   # 내부 LLM 호출
        }.get(_action, 0.0)

        # [Fix 2026-05-03] approval 가능성 있는 action 목록 — 이들은 risk 분류와
        # 무관하게 gateway timeout 강제 적용. 2-pass classification race 회피.
        _APPROVAL_PRONE_ACTIONS = {
            "WRITE", "EMAIL_SEND", "TELEGRAM_SEND",
            "NAVIGATE", "CLICK", "FILL", "GO_BACK",
        }

        try:
            from eidos_approval_gateway import (
                classify_only as _classify,
                RiskLevel as _RL,
                ApprovalGateway as _AG,
            )
            _risk = _classify(step.action or "", step.target or "", step.content or "")
            _force_gw = (_action in _APPROVAL_PRONE_ACTIONS)
            if _risk in (_RL.MEDIUM, _RL.HIGH) or _force_gw:
                _gw_to = 900
                try:
                    _gw_to = int(getattr(_AG.get(), "_timeout", 900) or 900)
                except Exception:
                    pass
                return float(max(self.STEP_TIMEOUT, _gw_to + 60, _action_floor))
        except Exception:
            pass
        # 마지막 fallback — approval-prone action 이면 gateway default + buffer 보장
        if _action in _APPROVAL_PRONE_ACTIONS:
            return float(max(self.STEP_TIMEOUT, 960.0, _action_floor))
        return float(max(self.STEP_TIMEOUT, _action_floor))

    async def _execute_step(self, step: ActionStep, plan: ActionPlan) -> str:
        action  = step.action
        target  = self._resolve_vars(step.target,  plan.variables)
        content = self._resolve_vars(step.content, plan.variables)

        # ── [Phase A] Approval Gateway: HIGH/MEDIUM 액션 실행 직전 인간 승인 ──
        # LOW는 AUTO로 즉시 통과, HIGH/MEDIUM은 Telegram으로 승인 대기.
        # 거부/타임아웃 시 이 step은 FAIL로 처리 (return 문자열 앞에 "⚠️").
        try:
            from eidos_approval_gateway import (
                gate_action as _gate_action,
                GateDecision as _GateDecision,
                kill_switch_active as _kill_active,
            )
            # Kill switch이 이미 활성화돼있으면 dispatcher 전체 중단
            if _kill_active():
                self._running = False
                return "⚠️ Approval Gateway kill-switch 활성 — dispatcher 정지"

            _gate_ctx = {
                "goal":        getattr(plan, "goal", "") or "",
                "task_prompt": getattr(plan, "task_prompt", "") or "",
                "url":         (self._get_current_browser_url() or ""),
                "step_no":     step.step_no,
                # [Phase 1 Telegram UX] LLM 이 이 액션을 선택한 이유 — 승인 메시지 Why 필드.
                "why":         getattr(step, "reason", "") or "",
            }
            # [T2 Step 3 2026-05-01] TELEGRAM_SEND verb 는 사용자 본인 채팅 발송 — risk LOW
            # + category=GENERAL 강제 주입. content 의 "전송"/"발송" 키워드가
            # category_keywords["DELIVERY"] 매칭 → delivery_force_high=True 로 매번 승인
            # 게이트 (자기 자신에게 텔레그램 ping-pong 무한 루프) 차단.
            if action == "TELEGRAM_SEND":
                _gate_ctx["category"] = "GENERAL"
            # [P-ε 2026-05-01] EMAIL_SEND 는 외부 발송이라 GENERAL 강제 X — DELIVERY 매칭
            # 시 force-MEDIUM 자연 흐름 따름. 단 사용자 본인 이메일 발송 케이스는 별도 토글
            # (env EIDOS_EMAIL_SEND_TO_SELF_LOW=1) 시 LOW 다운그레이드.
            elif action == "EMAIL_SEND":
                if str(os.environ.get("EIDOS_EMAIL_SEND_TO_SELF_LOW", "0")).strip().lower() in ("1", "true", "on"):
                    _gate_ctx["category"] = "GENERAL"
            # [Retry Auto-Pass 2026-04-27] retry_attrs origin + same-target WRITE
            # 시 ApprovalGateway 가 risk 한 단계 다운그레이드. ctx hint 만 전달하고
            # 실제 다운그레이드 로직은 ApprovalGateway.classify 에서 처리 (단일 진입점).
            # Why: Verifier FAIL → retry 진입 후 같은 산출물 보강 WRITE 가 매번 HIGH
            # risk 로 텔레그램 승인 게이트에 걸리면 자동 흐름이 멈춤 (최신로그.txt
            # Retry Step 10). retry 는 이미 한 번 승인한 흐름의 보강이므로.
            if (action == "WRITE"
                    and self._active_schema is not None
                    and getattr(self._active_schema, "_is_retry_run", False)):
                _lock_t = (getattr(self._active_schema, "_retry_target_lock", "") or "").strip()
                _curr_t = (target or "").strip()
                if _lock_t and _curr_t == _lock_t:
                    _gate_ctx["is_retry_same_target"] = True

            _gate_res = await _gate_action(
                action=action, target=target, content=content,
                context=_gate_ctx, cost_estimate=float(step.cost_estimate or 0.0),
            )
            # 메타 업데이트 (후속 로그/report에서 참조 가능)
            step.risk_level = _gate_res.risk.value if hasattr(_gate_res.risk, "value") else str(_gate_res.risk)
            step.approval_required = (_gate_res.decision != _GateDecision.AUTO)

            if _gate_res.decision == _GateDecision.AUTO:
                pass  # LOW or 자동통과 — 그대로 진행
            elif _gate_res.decision == _GateDecision.APPROVED:
                self._report(f"✅ [Approval] {action} 승인됨 (risk={step.risk_level})")
                # [Phase 2] rid 부착 — caller 가 실행 결과 확보 후 finalize_approval 호출.
                try:
                    step._pending_rid = getattr(_gate_res, "rid", None)
                except Exception:
                    pass
                # 사용자가 수정본 제출한 경우 content 치환
                if _gate_res.edited_content:
                    content = _gate_res.edited_content
                    print(f"  [Approval] content 수정본 적용 ({len(content)}자)")
            elif _gate_res.decision == _GateDecision.REJECTED:
                # Kill switch이 이번 호출에서 새로 당겨진 경우
                if _kill_active():
                    self._running = False
                self._report(f"❌ [Approval] {action} 거부됨 — {_gate_res.reason[:80]}")
                # REJECT 경로는 Telegram 측에서 이미 '❌ 거부됨' 메시지 + _pending 삭제 완료. finalize 불필요.
                return f"⚠️ Approval 거부: {_gate_res.reason[:120]}"
            elif _gate_res.decision == _GateDecision.TIMEOUT:
                self._report(f"⏰ [Approval] {action} 응답 대기 타임아웃 — {_gate_res.reason[:80]}")
                # [Phase 2] TIMEOUT 은 원본 '승인 요청' 메시지가 그대로 남아 있음 → 즉시 finalize 로 덮어쓰기.
                _timeout_rid = getattr(_gate_res, "rid", None)
                if _timeout_rid:
                    try:
                        from eidos_telegram_bot import get_bot as _get_bot_to
                        await _get_bot_to().finalize_approval(
                            rid=_timeout_rid, success=False,
                            detail=f"타임아웃 — {_gate_res.reason[:300]}",
                        )
                    except Exception as _fe_to:
                        print(f"  [Approval] timeout finalize 실패 (무시): {_fe_to}")
                return f"⚠️ Approval 타임아웃: {_gate_res.reason[:120]}"
        except ImportError:
            # Gateway 모듈 없을 때는 기존 동작 그대로 (배포 환경 호환)
            pass
        except Exception as _ge:
            print(f"  [Approval] 게이트웨이 예외 (무시, 기존 동작 유지): {_ge}")

        if action == "NAVIGATE":
            # ── [Dead-end 가드] 이미 막다른 길로 확정된 URL 재진입 차단 ─────
            _schema_for_de = self._active_schema
            if _schema_for_de is not None and target:
                _de_set = getattr(_schema_for_de, "_dead_end_urls", None)
                if _de_set:
                    # [Fix] dead-end set은 절대 URL(rstrip('/').lower())만 저장.
                    # target이 상대('/'나 '/path')이면 현재 base URL로 절대화 후 비교.
                    _check_url = (target or "").strip()
                    if _check_url and not _check_url.lower().startswith(("http://", "https://")):
                        try:
                            from urllib.parse import urljoin
                            _cur_url = self._get_current_browser_url() or ""
                            if _cur_url:
                                _check_url = urljoin(_cur_url, _check_url)
                        except Exception:
                            pass
                    _check_key = _check_url.rstrip("/").lower()
                    # [Fix 2026-04-19 v4] Guard 2차 방어선 — 허브 URL은 즉시 재진입 허용 + 등록 해제.
                    # hub-repeat(2407) 경로로 의도적 등록된 경우에도 Guard는 블록하지 않는다.
                    # 이유: /category/6 같은 허브는 task의 합법 목적지이고, 반복은 drill-down 힌트로만 유도.
                    _HUB_GUARD_PATTERNS = (
                        "/category", "/categories", "/search", "/tag/", "/tags/",
                        "/list", "/explore", "/browse", "/discover",
                        "?q=", "?query=", "?keyword=", "?search=",
                    )
                    _ck_lc = _check_key
                    _is_hub_check = any(p in _ck_lc for p in _HUB_GUARD_PATTERNS)
                    if _check_key and _check_key in _de_set and _is_hub_check:
                        _de_set.discard(_check_key)
                        print(f"  🛡️ [Dead-end Guard] 허브 URL — 면제 + blacklist 해제: {_check_url[:80]}")
                    if _check_key and _check_key in _de_set:
                        # [Fix 2026-04-19/04-22] 같은 dead-end URL을 N회 NAVIGATE 시도하면
                        # 해당 URL을 blacklist에서 제거해 재진입 허용 — 단, cooldown 동안 재등록 금지.
                        # Why: agent가 같은 URL을 반복 시도한다는 건 "여기 말고는 갈 곳이 없다"는
                        # 신호. 계속 차단하면 CLICK no-op 루프로 ASK_USER 조기 폴백됨.
                        # [2026-04-22] 한계 2→3으로 상향. 2일 때는 차단(1/2)→해제 핑퐁이
                        # 너무 빨라 LLM 이 같은 URL NAVIGATE 로 한 사이클만에 실패 재현.
                        # 또한 해제 후 DE_FREED_COOLDOWN 동안 재등록 금지 → 다음 Observe 에서
                        # 또 unhydrated 감지돼도 dead-end 재등록 안 돼 상태 진동 억제.
                        # How to apply: NAVIGATE 차단 카운터(_de_block_counter)를 schema 단위로 관리.
                        _de_ctr = getattr(_schema_for_de, "_de_block_counter", None)
                        if _de_ctr is None:
                            _de_ctr = {}
                            _schema_for_de._de_block_counter = _de_ctr
                        _de_ctr[_check_key] = _de_ctr.get(_check_key, 0) + 1
                        _DE_LIMIT = 3
                        _DE_FREED_COOLDOWN = 300.0  # seconds
                        if _de_ctr[_check_key] >= _DE_LIMIT:
                            _de_set.discard(_check_key)
                            _de_ctr.pop(_check_key, None)
                            # cooldown 기록 → Stage 1 재분류에서 참조
                            _de_freed = getattr(_schema_for_de, "_de_freed_at", None)
                            if _de_freed is None:
                                _de_freed = {}
                                _schema_for_de._de_freed_at = _de_freed
                            _de_freed[_check_key] = time.time() + _DE_FREED_COOLDOWN
                            print(
                                f"  ♻️ [Dead-end Guard] 재시도 한계 도달 — blacklist 해제: "
                                f"'{_check_url[:80]}' (재진입 허용, {int(_DE_FREED_COOLDOWN)}s "
                                f"cooldown 동안 재등록 금지)"
                            )
                            # 해제 후 아래 NAVIGATE 본체로 fall-through
                        else:
                            print(
                                f"  🚫 [Dead-end Guard] NAVIGATE 차단: '{target[:80]}' "
                                f"(절대화: {_check_url[:80]}) 이전 GO_BACK으로 막다른 길 "
                                f"표시됨 (차단 {_de_ctr[_check_key]}/{_DE_LIMIT})"
                            )
                            return (
                                f"⚠️ NAVIGATE 차단: '{target[:60]}' 은 이전에 막다른 페이지로 판정되어 "
                                f"재진입 금지. 다른 링크/키워드/URL을 시도하라."
                            )

            # ── [VERB 가드] schema.verb와 target URL 의도 일치 검증 ─────────
            # 예: VERB=EDIT인데 target에 /register·/new·/create → 차단 후 재-Think.
            # LLM이 intent-inconsistent한 URL을 생성하는 루프를 원천 차단.
            _schema_for_verb = self._active_schema
            if _schema_for_verb and _schema_for_verb.verb:
                try:
                    from eidos_mission_schema import VERB_FORBIDDEN_URL_PATTERNS as _VFP
                    _forbidden = _VFP.get(_schema_for_verb.verb, ())
                    _tl = (target or "").lower()
                    _hit = next((p for p in _forbidden if p in _tl), None)
                    if _hit:
                        _verb = _schema_for_verb.verb
                        print(f"  🚨 [VERB Guard] NAVIGATE 차단: verb={_verb} vs target='{target[:80]}' (금지패턴 '{_hit}')")
                        return (
                            f"⚠️ NAVIGATE 차단: 현재 VERB={_verb} 인데 target URL이 "
                            f"금지 패턴 '{_hit}'을 포함. "
                            f"EDIT=기존 리소스 수정, CREATE=신규 등록. "
                            f"의도에 맞는 경로로 재결정 필요 (예: EDIT이면 /my-gigs 같은 목록 → 수정 링크)."
                        )
                except Exception as _ve:
                    print(f"  [VERB Guard] 검사 실패 (무시): {_ve}")

            # [Fix] loadFinished 기반 스마트 대기 (kmong 등 SPA는 15초 부족 → 30초)
            _nav_result = await _navigate_and_wait(url=target, timeout=30.0)
            # ── [P0 Fix] loadFinished ok=False / 하드 실패 / 도메인 리다이렉트 감지 ─────
            _nav_result_s = str(_nav_result or "")
            if _nav_result_s.startswith("⚠️ 페이지 로드 실패"):
                # 하드 실패: 타겟 URL에 아예 도달 못 함. 재시도·URL 변형 필요.
                return _nav_result_s
            _nav_soft_warning = _nav_result_s.startswith("⚠️ 페이지 로드 경고") or _nav_result_s.startswith("⚠️ 페이지 리다이렉트")
            # ── 후검증 1: URL 리다이렉트로 로그인 게이트 감지 ───────
            # page_preview 검사는 modal 렌더링 타이밍에 따라 놓칠 수 있으므로
            # 최종 URL에서 login_modal/login/signin 패턴을 먼저 검사한다.
            _LOGIN_URL_PATTERNS = ("open=login_modal", "next_page=", "/login", "/signin", "/accounts/login")
            _tgt_l = (target or "").lower()
            _tgt_no_auth = any(p in _tgt_l for p in _LOGIN_URL_PATTERNS)
            if not _tgt_no_auth:
                try:
                    _cur_after = (self._get_current_browser_url() or "").lower()
                except Exception:
                    _cur_after = ""
                if _cur_after and any(p in _cur_after for p in _LOGIN_URL_PATTERNS):
                    return f"⚠️ 이동 실패: 로그인 필요 ({target} → {_cur_after[:80]})"

            # ── 후검증 2: 페이지 내용 확인 ──────────────────────────
            try:
                from execution_module import get_browser_content as _get_content
                page_text = await _get_content()
                page_preview = (page_text[:2000] if page_text else "").lower()
                # 404 감지
                if "404" in page_preview and ("찾을 수 없" in page_preview or "not found" in page_preview):
                    return f"⚠️ 이동 실패: {target} → 404 페이지"
                # 로그인 리다이렉트 감지
                if ("로그인" in page_preview or "login" in page_preview) and "kmong" in target.lower():
                    return f"⚠️ 이동 실패: 로그인 필요 ({target})"
                # 페이지 요약 저장 (후속 step에서 활용)
                plan.variables["__last_page_preview__"] = page_text[:1500] if page_text else ""
            except Exception:
                pass
            # [P0 Fix] loadFinished ok=False 소프트 경고는 이동 완료 문구에 부착해 LLM이 인지하도록
            if _nav_soft_warning:
                # [Stage 1 P1 2026-04-22] schema 에 nav_warning 기록.
                # Why: 다음 _observe() 에서 같은 URL이 unhydrated_spa로 잡히면
                # auth_required 가 아닌 nav_failed 로 재분류해 dead-end 등록.
                # 무한 ASK_USER 로그인 루프 차단.
                try:
                    if self._active_schema is not None:
                        self._active_schema._last_nav_warning = {
                            "url": target,
                            "result": _nav_result_s[:200],
                            "ts": time.time(),
                        }
                except Exception:
                    pass
                return f"이동 완료(경고): {target} — {_nav_result_s}"
            return f"이동 완료: {target}"

        elif action == "READ_PAGE":
            try:
                # [Fix 2026-04-21 P0-A] innerText 우선, HTML 소스는 폴백.
                # Why: kmong·nuxt·react 등 SPA 페이지에서 raw HTML 상단 ~12KB 가
                #      <head>의 meta/link/script 로 채워져 있어서 기존 HTML
                #      truncate 방식으로는 body 내 상품 카드 본문이 전혀 안 잡혔음
                #      (증상: product_summaries.md = head 복붙 후 침묵).
                #      document.body.innerText 는 사용자가 실제로 보는 렌더 텍스트만
                #      돌려주므로 head/script 덩어리를 근본적으로 우회한다.
                from execution_module import (
                    get_browser_visible_text as _get_text,
                    get_browser_content as _get_content,
                )
                _sch = getattr(self, "_active_schema", None)
                _ttype = (getattr(_sch, "task_type", "") or "") if _sch else ""
                _verb  = (getattr(_sch, "verb", "") or "").upper() if _sch else ""
                _is_read_task = (_ttype == "browser_read" or _verb == "READ")
                _limit = 12000 if _is_read_task else 3000

                # [Fix 2026-04-21 Phase 1-A] 변수명이 upstream 산출물/파일처럼 생겼으면
                # 브라우저 innerText 대신 체인 _react_vars → SAFE_BASE_PATH 파일 순으로
                # 먼저 시도한다. Why: plan/dev stage 가 READ_PAGE 로 "research_report_content"
                # 를 요청해놓고 실제로는 현재 브라우저 페이지(검색 결과 등 엉뚱한 곳)를 읽어
                # 이전 stage 산출물 대신 innerText 로 환각 경로를 만들었던 증상 차단.
                _target_raw = (target or "").strip()
                _file_candidate = ""
                if _target_raw:
                    _lw = _target_raw.lower()
                    if _lw.endswith((".md", ".txt", ".json", ".csv", ".yaml", ".yml")):
                        _file_candidate = _target_raw
                    elif _lw.endswith("_content"):
                        _stem = _target_raw[: -len("_content")]
                        if _stem:
                            _file_candidate = _stem + ".md"
                _from_file = ""
                _source_tag_override = ""
                if _file_candidate:
                    _react_vars = (getattr(_sch, "_react_vars", None) or {}) if _sch else {}
                    _fc_stem = _file_candidate.split(".", 1)[0]
                    for _vk, _vv in _react_vars.items():
                        if not isinstance(_vv, str) or not _vv:
                            continue
                        _vkl = _vk.lower()
                        _fcl = _file_candidate.lower()
                        if _vkl == _fcl \
                                or _vkl.endswith("/" + _fcl) \
                                or _vkl.endswith("." + _fcl) \
                                or _vkl == _fc_stem.lower() \
                                or _vkl.endswith("." + _fc_stem.lower()):
                            _from_file = _vv
                            _source_tag_override = f"react_vars[{_vk}]"
                            break
                    if not _from_file:
                        try:
                            from execution_module import SAFE_BASE_PATH as _SBASE_R
                            import os as _os_r
                            _fp = _os_r.path.abspath(
                                _os_r.path.join(_SBASE_R, _file_candidate)
                            )
                            _base_abs = _os_r.path.abspath(_SBASE_R)
                            if (_fp == _base_abs
                                    or _fp.startswith(_base_abs + _os_r.sep)) \
                                    and _os_r.path.isfile(_fp):
                                with open(_fp, "r", encoding="utf-8") as _fh_r:
                                    _from_file = _fh_r.read()
                                _source_tag_override = f"file:{_file_candidate}"
                        except Exception as _e_fr:
                            print(
                                f"  ⚠️ [READ_PAGE] 파일 라우팅 시도 예외 "
                                f"(innerText 폴백 진행): {_e_fr}"
                            )
                if _from_file:
                    if len(_from_file) > _limit:
                        _from_file = _from_file[:_limit] + "\n...(이하 생략)"
                    var_name = _target_raw or f"page_content_{step.step_no}"
                    plan.variables[var_name] = _from_file
                    plan.variables["__last_page_preview__"] = _from_file[:1500]
                    print(
                        f"  📄 [READ_PAGE→READ_FILE] source={_source_tag_override} "
                        f"var='{var_name}' len={len(_from_file)}"
                    )
                    return (
                        f"페이지 읽기 완료 (업스트림 파일로 라우팅) → "
                        f"변수 '{var_name}' ({len(_from_file)}자)"
                    )

                _source_tag = "innerText"
                page_text = ""
                try:
                    page_text = await _get_text()
                except Exception as _te:
                    print(f"  ⚠️ [READ_PAGE] innerText 경로 예외: {_te}")
                    page_text = ""

                # innerText 가 비어있거나 너무 짧으면(로그인·captcha·빈 SPA 셸 등)
                # HTML 소스로 폴백. 다만 HTML 폴백도 head 덩어리라면 이후 WRITE 가
                # 환각할 확률이 높으므로 [Fallback] 접두사로 호출부에 경고.
                if not page_text or len(page_text) < 200:
                    try:
                        _html = await _get_content()
                    except Exception as _he:
                        _html = ""
                        print(f"  ⚠️ [READ_PAGE] HTML 폴백 예외: {_he}")
                    if _html:
                        _source_tag = "HTML-fallback"
                        page_text = _html

                if not page_text:
                    return "⚠️ 페이지 읽기 실패: 본문 텍스트와 HTML 양쪽 모두 비어있음"

                if len(page_text) > _limit:
                    page_text = page_text[:_limit] + "\n...(이하 생략)"
                var_name = target.strip() or f"page_content_{step.step_no}"
                plan.variables[var_name] = page_text
                plan.variables["__last_page_preview__"] = page_text[:1500]
                # [Fix 2026-04-21 Phase 1-A] 파일처럼 생긴 target 인데도 파일을 못 찾고
                # innerText 로 폴백한 경우: LLM/WRITE 가 "이전 stage 산출물을 읽었다"
                # 고 착각하지 않도록 결과 메시지에 DRIFT 경고를 명시.
                _drift_warn = ""
                if _file_candidate:
                    # [Fix 2026-04-22] ⚠️ 이모지가 _verify_step 의 FAIL_PATTERNS("⚠️")에
                    # 걸려 READ_PAGE 성공임에도 step 이 FAIL 로 판정되던 버그 차단.
                    # LLM/다음 WRITE 가 DRIFT 경고를 인지해야 하므로 문구 자체는 유지.
                    _drift_warn = (
                        f" [주의-DRIFT] 요청된 파일 '{_file_candidate}' 를 찾지 못해 "
                        f"브라우저 innerText 로 폴백했음. 이전 stage 산출물이 아니다."
                    )
                print(
                    f"  📄 [READ_PAGE] source={_source_tag} var='{var_name}' "
                    f"len={len(page_text)}{' [DRIFT-fallback]' if _drift_warn else ''}"
                )
                return (
                    f"페이지 읽기 완료 → 변수 '{var_name}' ({len(page_text)}자)"
                    f"{_drift_warn}"
                )
            except Exception as e:
                return f"⚠️ 페이지 읽기 실패: {e}"

        elif action == "CHECK_LOGIN":
            check_js = """
(function() {
    var loginIndicators = [
        document.querySelector('a[href*="/logout"]'),
        document.querySelector('.user-profile'),
        document.querySelector('[class*="mypage"]'),
        document.querySelector('[href*="/mypage"]'),
        document.querySelector('[class*="ProfileImg"]'),
        document.querySelector('[class*="userAvatar"]'),
        document.querySelector('img[alt*="프로필"]'),
    ];
    var loggedIn = loginIndicators.some(function(el) { return el !== null; });
    return loggedIn ? 'LOGGED_IN' : 'NOT_LOGGED_IN';
})();"""
            result = await _browser_run(script=check_js)
            result_str = str(result) if result else ""
            if "NOT_LOGGED_IN" in result_str:
                # [Fix] 로그인 버튼 자동 클릭 시도
                login_click_js = """
(function() {
    var selectors = [
        'a[href*="/login"]', 'a[href*="/signin"]',
        'button[class*="login"]', 'a[class*="login"]',
        'a[href*="login"]', '.login-btn', '#login-btn',
        'a:contains("로그인")'
    ];
    for (var i = 0; i < selectors.length; i++) {
        try {
            var el = document.querySelector(selectors[i]);
            if (el) {
                el.click();
                return 'CLICKED_LOGIN:' + selectors[i];
            }
        } catch(e) {}
    }
    // text 기반 탐색
    var links = document.querySelectorAll('a, button');
    for (var j = 0; j < links.length; j++) {
        var t = (links[j].textContent || '').trim();
        if (t === '로그인' || t === 'Login' || t === '로그인하기') {
            links[j].click();
            return 'CLICKED_LOGIN_TEXT:' + t;
        }
    }
    return 'NO_LOGIN_BUTTON';
})();"""
                login_click_result = await _browser_run(script=login_click_js)
                login_click_str = str(login_click_result) if login_click_result else ""
                if "CLICKED_LOGIN" in login_click_str:
                    await asyncio.sleep(2.0)  # 로그인 페이지 로딩 대기
                    return f"[ASK_USER] 로그인 페이지로 이동했습니다. 로그인 후 **⚡ 자율 실행** 버튼을 눌러주세요."
                else:
                    return f"[ASK_USER] 로그인이 필요합니다. 크몽에 로그인 후 **⚡ 자율 실행** 버튼을 눌러주세요."
            return "✅ 로그인 확인 완료"

        elif action == "ASK_USER":
            question = content or target
            # [Fix] [ASK_USER] 태그가 붙은 메시지는 그대로 전달 (코파일럿 라우팅용)
            if question.startswith("[ASK_USER]"):
                msg = question[len("[ASK_USER]"):].strip()
                self._report(f"[ASK_USER] ❓ **[확인 필요]** {msg}\n└─ 완료되면 다시 **⚡ 자율 실행** 버튼을 눌러주세요.")
            else:
                self._report(
                    f"[ASK_USER] ❓ **[확인 필요]** {question}\n"
                    f"└─ 완료되면 다시 **⚡ 자율 실행** 버튼을 눌러주세요."
                )
            return f"사용자 확인 요청: {question[:60]}"

        elif action == "FILL":
            fill_html = content.replace("\\n", "\n").replace("\n", "<br>")
            fill_esc  = fill_html.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
            # ── [A11Y-INDEX 2026-04-24] target="[N]" 인덱스 우선 처리 ──
            # Why: 이전엔 FILL 이 인덱서 [N] 을 무시하고 자체 셀렉터 리스트만 사용
            # → 크몽 메인처럼 일반 셀렉터 미매칭인 SPA 에서 무조건 NO_EDITOR 실패.
            # CLICK 처럼 window.__EIDOS_IDX[N-1] 을 직접 참조하되, FILL 은 input/
            # textarea/contenteditable 만 허용해 LLM 이 link/button 을 잘못 지정한
            # 케이스를 즉시 명확 실패로 차단 (LLM 이 다음 step 에서 다른 [M] 또는
            # NAVIGATE 선택 가능하도록).
            _fill_idx_match = re.match(r'^\s*\[(\d+)\]\s*$', target or "")
            if _fill_idx_match:
                _n = int(_fill_idx_match.group(1))
                _idx_fill_js = f"""
(function() {{
  try {{
    var arr = window.__EIDOS_IDX;
    if (!arr || !arr[{_n - 1}]) return 'FILL_FAILED:INDEX_NOT_FOUND:{_n}';
    var ed = arr[{_n - 1}];
    if (!ed || !ed.tagName) return 'FILL_FAILED:INDEX_STALE:{_n}';
    /* [Fix 2026-05-03] DOM detached / invisible 가드 — CLICK 와 같은 패턴 */
    try {{ if (ed.isConnected === false) return 'FILL_FAILED:DETACHED:{_n}'; }} catch(e) {{}}
    try {{
        var _r = ed.getBoundingClientRect();
        if (!_r || (_r.width < 1 && _r.height < 1)) return 'FILL_FAILED:INVISIBLE:{_n}:' + Math.round(_r.width) + 'x' + Math.round(_r.height);
    }} catch(e) {{ return 'FILL_FAILED:RECT_ERROR:{_n}:' + (e.message || ''); }}
    var tag = ed.tagName.toLowerCase();
    var _ce = (ed.isContentEditable === true);
    var _isInput = (tag === 'input' || tag === 'textarea');
    var _isIframe = (tag === 'iframe');
    if (!_isInput && !_ce && !_isIframe) {{
        // input/textarea/editable/iframe 외엔 FILL 대상 아님 — 즉시 실패.
        var _kind = tag + (ed.getAttribute && ed.getAttribute('role') ? ':' + ed.getAttribute('role') : '');
        return 'FILL_FAILED:NOT_EDITABLE:[{_n}]:' + _kind;
    }}
    if (_isIframe) {{
        // iframe 내부 contenteditable 진입 시도 (same-origin 만)
        try {{
            var idoc = ed.contentDocument;
            if (idoc) {{
                var _ce2 = idoc.querySelector('[contenteditable="true"], textarea, input:not([type="hidden"])');
                if (_ce2) ed = _ce2;
                else return 'FILL_FAILED:IFRAME_NO_EDITOR:[{_n}]';
            }} else return 'FILL_FAILED:IFRAME_CROSS_ORIGIN:[{_n}]';
        }} catch(e) {{ return 'FILL_FAILED:IFRAME_ACCESS:[{_n}]:' + (e.message || ''); }}
    }}
    try {{ ed.scrollIntoView({{block:'center', behavior:'instant'}}); }} catch(e) {{}}
    try {{ ed.focus(); }} catch(e) {{}}
    var _isSearchInput = false;
    if (ed.tagName === 'TEXTAREA' || ed.tagName === 'INPUT') {{
        var _nm = (ed.getAttribute('name') || '').toLowerCase();
        var _tp = (ed.getAttribute('type') || '').toLowerCase();
        var _id = (ed.getAttribute('id') || '').toLowerCase();
        _isSearchInput = (_tp === 'search'
            || _nm === 'q' || _nm === 'search_query' || _nm === 'search' || _nm === 'query'
            || _id === 'search' || _id === 'query');
        var _setterOk = false;
        try {{
            var _proto = (ed.tagName === 'TEXTAREA')
                ? window.HTMLTextAreaElement.prototype
                : window.HTMLInputElement.prototype;
            var _desc = Object.getOwnPropertyDescriptor(_proto, 'value');
            if (_desc && _desc.set) {{ _desc.set.call(ed, `{fill_esc}`); _setterOk = true; }}
        }} catch(e) {{ _setterOk = false; }}
        if (!_setterOk) {{ try {{ ed.value = `{fill_esc}`; }} catch(e) {{}} }}
    }} else {{
        try {{ ed.innerHTML = `{fill_esc}`; }} catch(e) {{}}
    }}
    try {{ ed.dispatchEvent(new Event('focus',  {{bubbles: true}})); }} catch(e) {{}}
    try {{ ed.dispatchEvent(new Event('input',  {{bubbles: true}})); }} catch(e) {{}}
    try {{ ed.dispatchEvent(new Event('change', {{bubbles: true}})); }} catch(e) {{}}
    try {{ ed.dispatchEvent(new KeyboardEvent('keydown', {{bubbles: true, key: 'a'}})); }} catch(e) {{}}
    try {{ ed.dispatchEvent(new KeyboardEvent('keyup',   {{bubbles: true, key: 'a'}})); }} catch(e) {{}}
    if (_isSearchInput) {{
        try {{
            ed.dispatchEvent(new KeyboardEvent('keydown', {{bubbles:true, cancelable:true, key:'Enter', code:'Enter', keyCode:13, which:13}}));
            ed.dispatchEvent(new KeyboardEvent('keypress', {{bubbles:true, cancelable:true, key:'Enter', code:'Enter', keyCode:13, which:13}}));
            ed.dispatchEvent(new KeyboardEvent('keyup',    {{bubbles:true, cancelable:true, key:'Enter', code:'Enter', keyCode:13, which:13}}));
            var _form = ed.form || (ed.closest && ed.closest('form'));
            if (_form) {{ try {{ if (typeof _form.requestSubmit === 'function') _form.requestSubmit(); else _form.submit(); }} catch(e) {{}} }}
        }} catch(e) {{}}
    }}
    var len = (ed.value || ed.innerHTML || '').length;
    return 'FILL_SUCCESS:' + len + ':' + ed.tagName + ':IDX[{_n}]' + (_isSearchInput ? ':SEARCH_ENTER' : '');
  }} catch(_outerE) {{
    var _em = (_outerE && _outerE.message) ? _outerE.message : 'unknown';
    return 'FILL_FAILED:JS_ERROR:[{_n}]:' + _em;
  }}
}})();"""
                _idx_result = await _browser_run(script=_idx_fill_js)
                _idx_result_str = str(_idx_result) if _idx_result else ""
                if "FILL_SUCCESS" in _idx_result_str:
                    return f"에디터 주입 완료 ({len(content)}자, 검증: {_idx_result_str})"
                if "FILL_FAILED:NOT_EDITABLE" in _idx_result_str:
                    # LLM 이 link/button 을 FILL 로 잘못 골랐다 — 즉시 명확한 안내.
                    _kind_part = _idx_result_str.split(":", 3)[-1] if ":" in _idx_result_str else "?"
                    return (
                        f"⚠️ FILL 실패: target=[{_n}] 은 편집 가능 요소가 아니다 "
                        f"(kind={_kind_part}). FILL 은 input/textarea/contenteditable 만 가능. "
                        f"인덱서에서 input/textarea 인 다른 [M] 을 선택하거나, 검색이면 NAVIGATE 로 "
                        f"/search 엔드포인트에 직접 이동하라."
                    )
                if "FILL_FAILED:INDEX_NOT_FOUND" in _idx_result_str or "INDEX_STALE" in _idx_result_str:
                    return (
                        f"⚠️ FILL 실패: 인덱스 [{_n}] 이 캐시에 없거나 오래됐다. "
                        f"페이지가 재로드 됐을 가능성. 다음 step 에서 인덱서를 다시 받아 새 [N] 선택."
                    )
                # [Fix 2026-05-03] DETACHED / INVISIBLE / RECT_ERROR 케이스 — 다음 step 회복
                if "FILL_FAILED:DETACHED" in _idx_result_str:
                    return (
                        f"⚠️ FILL 실패: 인덱스 [{_n}] DOM detached — SPA rerender 로 element 분리됨. "
                        f"다음 step 의 fresh 인덱서 새 [N] 사용."
                    )
                if "FILL_FAILED:INVISIBLE" in _idx_result_str:
                    _size = _idx_result_str.split("INVISIBLE:", 1)[-1].split(":", 1)[-1] if "INVISIBLE:" in _idx_result_str else "?"
                    return (
                        f"⚠️ FILL 실패: 인덱스 [{_n}] 0 크기 ({_size[:20]}) — 모달 닫힘 / display:none / "
                        f"이미 처리됨. 다음 step 에서 가시 요소 선택 또는 NAVIGATE 로 폼 페이지 이동."
                    )
                if "FILL_FAILED:RECT_ERROR" in _idx_result_str:
                    return (
                        f"⚠️ FILL 실패: 인덱스 [{_n}] getBoundingClientRect 예외 — DOM 비정상. "
                        f"페이지 안정 대기 후 다음 step 에서 재시도."
                    )
                # 기타 에러 — 셀렉터 폴백으로 진행 (아래 로직)
                print(f"  [FILL idx={_n}] 인덱스 모드 에러({_idx_result_str[:80]}) → 셀렉터 폴백 시도")

            # ── 에디터 탐색 + 주입 + 후검증 (셀렉터 폴백 / target 이 인덱스 형식 아닐 때) ──
            js = f"""
(function() {{
  // [Fix 2026-05-03] 가시성 + 편집가능 검사 helper.
  function _isEditableVisible(el) {{
    try {{
      if (!el || !el.tagName) return false;
      var r = el.getBoundingClientRect();
      if (r.width < 2 || r.height < 2) return false;
      var s = window.getComputedStyle(el);
      if (s.display === 'none' || s.visibility === 'hidden') return false;
      if (parseFloat(s.opacity) < 0.1) return false;
      if (el.disabled || el.readOnly) return false;
      var tag = el.tagName.toLowerCase();
      var t = (el.getAttribute('type') || '').toLowerCase();
      // hidden/submit/checkbox/radio/button 등은 FILL 대상 아님
      if (tag === 'input' && (t === 'hidden' || t === 'submit' || t === 'button' ||
          t === 'checkbox' || t === 'radio' || t === 'image' || t === 'file')) return false;
      return true;
    }} catch(e) {{ return false; }}
  }}
  try {{
    var selectors = [
        '.ProseMirror',
        '[contenteditable="true"]',
        '.ql-editor',
        'div[data-placeholder]',
        // 검색창류 — YouTube/일반 사이트의 search input 우선 매칭
        'input[name="search_query"]',
        'input[id="search"]:not([type="hidden"])',
        'input[type="search"]',
        'input[name="q"]',
        'textarea',
        '.note-editable',
        '.sun-editor-editable',
        '#tinymce',
    ];
    var ed = null;
    for (var i = 0; i < selectors.length; i++) {{
        var _cands = document.querySelectorAll(selectors[i]);
        for (var _j = 0; _j < _cands.length; _j++) {{
            if (_isEditableVisible(_cands[_j])) {{ ed = _cands[_j]; break; }}
        }}
        if (ed) break;
    }}
    // [Fix 2026-05-03] 하드코딩 selector 전체 미매칭 시 ANY visible input/textarea/CE
    // 폴백 — 사이트별 커스텀 selector 누락 회피.
    if (!ed) {{
        var _broad = document.querySelectorAll(
            'input,textarea,[contenteditable=""],[contenteditable="true"]'
        );
        for (var _k = 0; _k < _broad.length; _k++) {{
            if (_isEditableVisible(_broad[_k])) {{ ed = _broad[_k]; break; }}
        }}
    }}
    if (!ed) {{
        // [Fix 2026-05-03] 편집가능 요소 0개 사실 보고 — 실제 페이지에 어떤 input
        // 이 있는지 (visibility 무시한 raw) 도 함께 반환 → LLM 이 NAVIGATE 로 폼
        // 페이지 이동 / 모달 열기 등 다음 행동 결정 가능.
        var _allInputs = document.querySelectorAll(
            'input,textarea,[contenteditable=""],[contenteditable="true"]'
        );
        var _summary = [];
        for (var _x = 0; _x < _allInputs.length && _summary.length < 10; _x++) {{
            var _e = _allInputs[_x];
            var _why = !_isEditableVisible(_e)
                ? (_e.disabled ? 'disabled' : (_e.readOnly ? 'readonly' :
                   ((_e.offsetWidth < 2 || _e.offsetHeight < 2) ? 'invisible' : 'hidden')))
                : 'ok';
            var _nm = (_e.getAttribute('name') || _e.getAttribute('placeholder') ||
                       _e.getAttribute('aria-label') || _e.tagName).substring(0, 30);
            _summary.push(_nm + '(' + _why + ')');
        }}
        return 'FILL_FAILED:NO_EDITOR:' + document.title +
               '|raw_count=' + _allInputs.length +
               '|samples=' + (_summary.join(',') || 'none');
    }}
    try {{ ed.focus(); }} catch(e) {{}}
    var _isSearchInput = false;
    if (ed.tagName === 'TEXTAREA' || ed.tagName === 'INPUT') {{
        // 검색창 감지: name=q/search_query, type=search, id=search 등
        var _nm = (ed.getAttribute('name') || '').toLowerCase();
        var _tp = (ed.getAttribute('type') || '').toLowerCase();
        var _id = (ed.getAttribute('id') || '').toLowerCase();
        _isSearchInput = (_tp === 'search'
            || _nm === 'q' || _nm === 'search_query' || _nm === 'search' || _nm === 'query'
            || _id === 'search' || _id === 'query');
        // 9-2 수정: React/Vue의 synthetic input 처리를 위해 nativeInputValueSetter 사용
        // 일부 사이트(YouTube 등)에서 .call() 호출이 'Illegal invocation' 던짐 →
        // try/catch 로 감싸고 조용히 직접 할당 폴백.
        var _setterOk = false;
        try {{
            var _proto = (ed.tagName === 'TEXTAREA')
                ? window.HTMLTextAreaElement.prototype
                : window.HTMLInputElement.prototype;
            var _desc = Object.getOwnPropertyDescriptor(_proto, 'value');
            if (_desc && _desc.set) {{
                _desc.set.call(ed, `{fill_esc}`);
                _setterOk = true;
            }}
        }} catch(e) {{ _setterOk = false; }}
        if (!_setterOk) {{
            try {{ ed.value = `{fill_esc}`; }} catch(e) {{}}
        }}
    }} else {{
        try {{ ed.innerHTML = `{fill_esc}`; }} catch(e) {{}}
    }}
    // 9-2 수정: React/Vue state 동기화를 위한 이벤트 시퀀스
    try {{ ed.dispatchEvent(new Event('focus',  {{bubbles: true}})); }} catch(e) {{}}
    try {{ ed.dispatchEvent(new Event('input',  {{bubbles: true}})); }} catch(e) {{}}
    try {{ ed.dispatchEvent(new Event('change', {{bubbles: true}})); }} catch(e) {{}}
    try {{ ed.dispatchEvent(new KeyboardEvent('keydown', {{bubbles: true, key: 'a'}})); }} catch(e) {{}}
    try {{ ed.dispatchEvent(new KeyboardEvent('keyup',   {{bubbles: true, key: 'a'}})); }} catch(e) {{}}
    // 검색 input이면 Enter 키를 발사 — form submit 자동 트리거 (뒤따르는 CLICK no-op 회피)
    if (_isSearchInput) {{
        try {{
            var _ev1 = new KeyboardEvent('keydown', {{bubbles: true, cancelable: true, key: 'Enter', code: 'Enter', keyCode: 13, which: 13}});
            var _ev2 = new KeyboardEvent('keypress', {{bubbles: true, cancelable: true, key: 'Enter', code: 'Enter', keyCode: 13, which: 13}});
            var _ev3 = new KeyboardEvent('keyup',   {{bubbles: true, cancelable: true, key: 'Enter', code: 'Enter', keyCode: 13, which: 13}});
            ed.dispatchEvent(_ev1);
            ed.dispatchEvent(_ev2);
            ed.dispatchEvent(_ev3);
            // form 안에 있으면 requestSubmit 폴백
            var _form = ed.form || ed.closest('form');
            if (_form && !_ev1.defaultPrevented) {{
                try {{
                    if (typeof _form.requestSubmit === 'function') _form.requestSubmit();
                    else _form.submit();
                }} catch(e) {{}}
            }}
        }} catch(e) {{}}
    }}
    var len = (ed.value || ed.innerHTML || '').length;
    return 'FILL_SUCCESS:' + len + ':' + ed.tagName + (_isSearchInput ? ':SEARCH_ENTER' : '');
  }} catch(_outerE) {{
    // FILL 내부에서 uncaught 예외(예: Illegal invocation) → 실패로 보고.
    // 이전에는 외부 safe_script 래퍼가 빈 문자열을 반환해 dispatcher가
    // "SPA void return"으로 성공 처리해 LLM이 FILL 성공으로 오해했다.
    var _em = (_outerE && _outerE.message) ? _outerE.message : 'unknown';
    return 'FILL_FAILED:JS_ERROR:' + _em;
  }}
}})();"""
            result = await _browser_run(script=js)
            result_str = str(result) if result else "FILL_UNKNOWN"
            if "FILL_FAILED" in result_str:
                # [Fix 2026-05-03] 새 JS 응답 형식: 'FILL_FAILED:NO_EDITOR:title|raw_count=N|samples=name1(why),name2(why),...'
                # LLM 이 다음 step 에서 NAVIGATE / CLICK 으로 편집 모드 진입 결정 가능하도록
                # raw_count 와 sample 을 그대로 노출.
                _payload = result_str.split("FILL_FAILED:NO_EDITOR:", 1)[-1] if "NO_EDITOR" in result_str else result_str
                _parts = _payload.split("|")
                _title = _parts[0] if _parts else "알 수 없음"
                _raw_count = next((p for p in _parts if p.startswith("raw_count=")), "raw_count=?")
                _samples = next((p for p in _parts if p.startswith("samples=")), "samples=none")
                return (
                    f"⚠️ FILL 실패: 가시적 편집가능 요소 0개. "
                    f"현재 페이지: '{_title[:50]}'. "
                    f"({_raw_count}, {_samples[:200]}). "
                    f"→ 다음 step 권장: (a) 편집 버튼/모드 CLICK 으로 입력 폼 열기 "
                    f"(b) NAVIGATE 로 전용 편집/등록 페이지 이동 "
                    f"(c) samples 의 hidden/disabled 항목이 보이지 않으면 페이지 스크롤 또는 다른 탭 진입."
                )
            if "FILL_SUCCESS" in result_str:
                return f"에디터 주입 완료 ({len(content)}자, 검증: {result_str})"
            # [오염 수정 2026-04-18] 이전에는 void return을 SPA 성공으로 봐서
            # IIFE 내부의 uncaught Illegal invocation이 발생해도 성공 처리됐다.
            # 이제 IIFE에 outer try/catch가 있어 정상 경로는 FILL_SUCCESS 또는
            # FILL_FAILED를 반환. void return은 safe_script 수준 SyntaxError 등
            # 진짜 실패 상황 → FILL_FAILED로 보고해 verify_step이 실패 처리.
            return f"FILL 실패: 응답 없음 (safe_script 레벨 오류 의심) — 편집 가능 영역 없음 가능성"

        elif action == "CLICK":
            # ── [A11Y-INDEX] target="[N]" 형식: 인덱서가 만든 엘리먼트 직접 참조 ──
            # window.__EIDOS_IDX[N-1]을 조회하여 이벤트 디스패치. CSS 셀렉터 추측
            # 실패 문제를 원천 차단. 동일 페이지 내에서만 유효 (새 인덱스는 다음 _observe에서).
            _idx_match = re.match(r'^\s*\[(\d+)\]\s*$', target)
            if _idx_match:
                _n = int(_idx_match.group(1))
                # [Fix 2026-05-03] stale element 가드 추가:
                #   1) isConnected — 페이지 변경 후 DOM 에서 detached 된 element 감지
                #   2) bounding rect — 0 크기면 invisible/removed 로 판정
                #   3) tagName 확인 — null/undefined 안전
                # 셋 중 하나라도 실패 시 INDEX_STALE 반환 → 다음 step 에서 fresh 인덱서 받음.
                _idx_js = (
                    "(function() {"
                    "  var arr = window.__EIDOS_IDX;"
                    "  if (!arr || !arr[" + str(_n - 1) + "]) return 'CLICK_FAILED:INDEX_NOT_FOUND:" + str(_n) + "';"
                    "  var el = arr[" + str(_n - 1) + "];"
                    "  if (!el || !el.tagName) return 'CLICK_FAILED:INDEX_STALE:" + str(_n) + "';"
                    "  /* [Fix 2026-05-03] DOM detached 감지 — 페이지 navigation 후 element 죽음 */"
                    "  try { if (el.isConnected === false) return 'CLICK_FAILED:DETACHED:" + str(_n) + "'; } catch(e) {}"
                    "  /* element 살아있어도 size 0 면 invisible/removed → 클릭 무반응 */"
                    "  try {"
                    "    var _r = el.getBoundingClientRect();"
                    "    if (!_r || (_r.width < 1 && _r.height < 1)) return 'CLICK_FAILED:INVISIBLE:" + str(_n) + ":' + Math.round(_r.width) + 'x' + Math.round(_r.height);"
                    "  } catch(e) { return 'CLICK_FAILED:RECT_ERROR:" + str(_n) + ":' + (e.message || ''); }"
                    "  try { el.scrollIntoView({block: 'center', behavior: 'instant'}); } catch(e) {}"
                    "  var _clickEv = new MouseEvent('click', {bubbles:true, cancelable:true, view:window});"
                    "  try { el.dispatchEvent(new MouseEvent('mouseover', {bubbles:true, cancelable:true, view:window})); } catch(e) {}"
                    "  try { el.dispatchEvent(new MouseEvent('mousedown', {bubbles:true, cancelable:true, view:window})); } catch(e) {}"
                    "  try { el.dispatchEvent(new MouseEvent('mouseup',   {bubbles:true, cancelable:true, view:window})); } catch(e) {}"
                    "  try { el.dispatchEvent(_clickEv); } catch(e) {}"
                    "  /* fallback: submit button / button in form -> requestSubmit (YouTube-style SPA search button) */"
                    "  var _sub = false;"
                    "  try {"
                    "    var _t = (el.getAttribute('type') || '').toLowerCase();"
                    "    var _tag = (el.tagName || '').toLowerCase();"
                    "    if (!_clickEv.defaultPrevented && (_t === 'submit' || _tag === 'button')) {"
                    "      var _f = (typeof el.form !== 'undefined' && el.form) ? el.form : (el.closest ? el.closest('form') : null);"
                    "      if (_f) {"
                    "        try { if (typeof _f.requestSubmit === 'function') { _f.requestSubmit(el); _sub = true; } else { _f.submit(); _sub = true; } } catch(e) {}"
                    "      }"
                    "    }"
                    "  } catch(e) {}"
                    "  /* fallback2: native el.click() when default not prevented and nav did not fire (SPA) */"
                    "  try {"
                    "    if (!_sub && !_clickEv.defaultPrevented && typeof el.click === 'function') {"
                    "      el.click();"
                    "    }"
                    "  } catch(e) {}"
                    "  var label = (el.getAttribute('aria-label') || el.textContent || '').trim().substring(0, 30);"
                    "  return 'CLICK_SUCCESS:idx[" + str(_n) + "]:' + (_sub ? 'SUBMIT:' : '') + label;"
                    "})();"
                )
                _result = await _browser_run(script=_idx_js)
                _result_str = str(_result) if _result else ""
                # [Fix 2026-05-03] stale element 케이스 명확 분기 — LLM 에 더 풍부한 신호 전달
                if "INDEX_NOT_FOUND" in _result_str:
                    return f"⚠️ CLICK 실패: 인덱스 [{_n}] 캐시 비어있음 — indexer 미실행 또는 페이지 재로드. 다음 step 에서 fresh 인덱서 자동 적용"
                if "INDEX_STALE" in _result_str:
                    return f"⚠️ CLICK 실패: 인덱스 [{_n}] stale (tagName 없음) — DOM 객체 파괴됨. 다음 step 의 새 [N] 사용"
                if "DETACHED" in _result_str:
                    return f"⚠️ CLICK 실패: 인덱스 [{_n}] DOM detached — 페이지 변경/SPA rerender 로 element 분리됨. 다음 step 에서 자동 재인덱싱"
                if "INVISIBLE" in _result_str:
                    _size = _result_str.split("INVISIBLE:", 1)[-1].split(":", 1)[-1]
                    return f"⚠️ CLICK 실패: 인덱스 [{_n}] 0 크기 ({_size}) — 모달 닫힘 / display:none / 이미 클릭 처리됨. 다음 step 에서 새 요소 선택"
                if "RECT_ERROR" in _result_str:
                    return f"⚠️ CLICK 실패: 인덱스 [{_n}] getBoundingClientRect 예외 — DOM 비정상. 페이지 안정 대기 후 재시도"
                if "CLICK_SUCCESS" in _result_str:
                    _parts = _result_str.split(":", 2)
                    _label = _parts[2] if len(_parts) > 2 else ""
                    return f"클릭 성공: [{_n}] ({_label[:30]})"
                # void return = SPA 이벤트 방식 성공 처리
                return f"클릭 실행됨 (인덱스 [{_n}], SPA void return)"

            # ── [Fix 1] 플레이스홀더 target이면 Reactive 즉석 결정 ──────────────
            _placeholder_keywords = ("화면에서 확인 후 결정", "확인 후 결정", "reactive")
            is_placeholder = (
                not target.strip()
                or any(kw in target for kw in _placeholder_keywords)
            )
            if is_placeholder:
                page_snapshot = ""
                try:
                    from execution_module import get_browser_content as _gc2
                    page_snapshot = await _gc2()
                    if len(page_snapshot) > 2000:
                        page_snapshot = page_snapshot[:2000]
                except Exception:
                    pass
                if page_snapshot:
                    reactive_result = await self._reactive_decide_next(
                        plan.task_prompt, page_snapshot, step, plan
                    )
                    if reactive_result:
                        target = reactive_result.get("target", target)
                        if reactive_result.get("action"):
                            step.action = reactive_result["action"].upper()
                            action = step.action
                        print(f"  🔍 [Reactive-CLICK] 즉석 결정: {action} → {target[:60]}")
                        if action != "CLICK":
                            step.target = target
                            return await self._execute_step(step, plan)
                # 여전히 플레이스홀더면 실패
                if not target.strip() or any(kw in target for kw in _placeholder_keywords):
                    return f"⚠️ CLICK 실패: Reactive 결정 불가 (화면 읽기 실패 또는 LLM 응답 없음)"

            # 쉼표로 구분된 여러 셀렉터 지원
            selectors = [s.strip() for s in target.split(",") if s.strip()]
            if not selectors:
                return f"⚠️ CLICK 실패: 셀렉터가 비어있음"

            selectors_js = ", ".join(f'"{s}"' for s in selectors)
            js = f"""
(function() {{
    var selectors = [{selectors_js}];
    for (var i = 0; i < selectors.length; i++) {{
        var el = document.querySelector(selectors[i]);
        if (el) {{
            ['mouseover','mousedown','mouseup','click'].forEach(function(t) {{
                el.dispatchEvent(new MouseEvent(t, {{bubbles:true, cancelable:true}}));
            }});
            return 'CLICK_SUCCESS:' + selectors[i] + ':' + (el.textContent || '').substring(0, 30);
        }}
    }}
    return 'CLICK_FAILED:NONE_FOUND:' + selectors.join('|');
}})();"""
            result = await _browser_run(script=js)
            result_str = str(result) if result else ""
            if "CLICK_FAILED" in result_str or "NONE_FOUND" in result_str:
                tried = result_str.split(":")[-1] if ":" in result_str else target
                return f"⚠️ CLICK 실패: 다음 셀렉터 중 어떤 것도 찾지 못함: {tried[:80]}"
            if "CLICK_SUCCESS" in result_str:
                parts = result_str.split(":")
                clicked_sel = parts[1] if len(parts) > 1 else target
                clicked_text = parts[2] if len(parts) > 2 else ""
                return f"클릭 성공: {clicked_sel} (텍스트: {clicked_text[:30]})"
            # ── [Fix 2] JS void return = SPA 이벤트 방식 클릭 → 성공으로 처리 ──
            return f"클릭 실행됨 (SPA void return): {target[:60]}"

        elif action == "WRITE":
            # 파일/보고서 산출물이라서 채팅 페르소나(AIRA 기본)가 아니라
            # 분석가 페르소나로 호출 — 반말·호칭은 유지하되 내용은 진지하게.
            var_name = target.strip() or f"write_{step.step_no}"

            # [Phase 3 2026-04-25] 샘플링 게이트 PRE-VETO.
            # Why: 기존 게이트는 거부 메시지를 act_result 로만 리턴 → LLM 이
            # 다음 step 에서 또 같은 WRITE 시도 (research stage Step 3·4·5 무시
            # 패턴). 게이트가 1회 거부한 target 은 schema._sampling_blocked_targets
            # 에 detail var 개수를 기록 → 다음 WRITE 진입 시 detail 개수가 늘지
            # 않았으면 dispatcher 가 게이트 로직 진입 전에 하드 거부. LLM 이
            # NAVIGATE/READ_PAGE 로 강제 전환되도록 한다.
            try:
                _sch_pv = self._active_schema
                # [Strangler v2 2026-05-15] research 면제 마커면 PRE-VETO 전체
                # skip (검색→종합 모델은 detail 진전 개념이 없어 strike 가
                # 영구 누적됨). 마커 없으면 기존 PRE-VETO 100% 보존.
                if _sch_pv is not None and not getattr(_sch_pv, "_research_no_sampling", False):
                    _blocked = getattr(_sch_pv, "_sampling_blocked_targets", None)
                    # [List-as-Info Whitelist 2026-04-27] 네이버 지도/카카오맵 같은
                    # 도메인은 LIST 페이지 자체가 정보 완성 (식당명·평점·주소 노출).
                    # detail page 진입 강요는 본질적으로 불필요 → PRE-VETO 차단 자체 우회.
                    # 회귀 가드: kmong/gumroad 같은 e-commerce 는 화이트리스트 미포함이라
                    # 기존 sampling/PRE-VETO 로직 100% 보존.
                    _var_urls_pv0 = getattr(_sch_pv, "_react_var_urls", {}) or {}
                    _rv_pv0 = getattr(_sch_pv, "_react_vars", {}) or {}
                    # [의미 게이트 2026-04-27] 화이트리스트 hit + 본문 평점/리뷰 신호
                    # 충분 시에만 bypass. Verifier 단 의미 누수 차단.
                    _list_as_info_hit_pv = False
                    _list_as_info_signals_pv = 0
                    for _u_li, _vn_li in _var_urls_pv0.items():
                        if not _is_list_as_info_domain(_u_li):
                            continue
                        _val_li = _rv_pv0.get(_vn_li, "")
                        if not (isinstance(_val_li, str) and len(_val_li) >= 200
                                and not _val_li.lstrip().startswith("[⚠️ BLOCKED_PAGE")):
                            continue
                        # PRE-VETO 단에선 _N_required 계산 비용 없이 절대 임계 (≥2) 사용
                        if _has_place_density(_val_li, n_required=0):
                            _list_as_info_hit_pv = True
                            _list_as_info_signals_pv = _count_place_entities(_val_li)
                            break
                    if _list_as_info_hit_pv and isinstance(_blocked, dict) and var_name in _blocked:
                        # 화이트리스트 도메인 방문 흔적이 있으면 PRE-VETO 무력화 +
                        # _sampling_blocked_targets 에서 var_name 제거 (게이트 재발동 방지)
                        try:
                            _blocked.pop(var_name, None)
                        except Exception:
                            pass
                        try:
                            _pv_strikes_clr = getattr(_sch_pv, "_pre_veto_strikes", None)
                            if isinstance(_pv_strikes_clr, dict):
                                _pv_strikes_clr.pop(var_name, None)
                        except Exception:
                            pass
                        print(
                            f"  ✅ [WRITE PRE-VETO bypass] '{var_name}' — list-as-info "
                            f"도메인 + place 신호 {_list_as_info_signals_pv}개 — gate 우회"
                        )
                        # _blocked 에서 제거됐으니 아래 if 분기 미진입

                    # [Fix E3 2026-05-06] schema HOW_MUCH 자동 진전 충족 시 PRE-VETO 우회.
                    # Why: Fix B AutoProgress 가 schema.attr.achieved 자동 갱신 시키지만
                    # phase B PRE-VETO 는 별도 URL classifier 카운터라 영향 X. 결국 schema
                    # 측은 충분 판단했지만 게이트가 detail=0 거부 → 무한 NAVIGATE 루프
                    # (최신로그.txt 10:06~10:07). schema 의 numeric HOW_MUCH 속성 모두
                    # achieved 면 gate advisory 로 인정하고 통과.
                    if isinstance(_blocked, dict) and var_name in _blocked:
                        try:
                            import re as _re_e3pv
                            _attrs_e3pv = list(getattr(_sch_pv, "attributes", []) or [])
                            _has_count_e3pv = False
                            _all_ok_e3pv = True
                            for _a_e3pv in _attrs_e3pv:
                                _name_e3pv = (getattr(_a_e3pv, "name", "") or "")
                                if _re_e3pv.search(r"\d+", _name_e3pv):
                                    _has_count_e3pv = True
                                    if not getattr(_a_e3pv, "achieved", False):
                                        _all_ok_e3pv = False
                                        break
                            if _has_count_e3pv and _all_ok_e3pv:
                                print(
                                    f"  ✅ [Fix E3] '{var_name}' PRE-VETO 우회 — "
                                    f"schema HOW_MUCH numeric 속성 모두 achieved "
                                    f"(Fix B AutoProgress)"
                                )
                                self._report(
                                    f"✅ **[PRE-VETO 우회 — schema 충족]** `{var_name}` — "
                                    f"schema 의 모든 numeric HOW_MUCH 속성 자동 진전 충족."
                                )
                                try:
                                    _blocked.pop(var_name, None)
                                    _pv_strikes_e3 = getattr(_sch_pv, "_pre_veto_strikes", None)
                                    if isinstance(_pv_strikes_e3, dict):
                                        _pv_strikes_e3.pop(var_name, None)
                                except Exception:
                                    pass
                        except Exception as _e_e3pv:
                            print(f"  [Fix E3 PRE-VETO] 검사 실패 (무시): {_e_e3pv}")

                    if isinstance(_blocked, dict) and var_name in _blocked:
                        # Phase B [2026-04-27] module-level classifier 사용 (Gumroad /l/ 등 커버)
                        _var_urls_pv = getattr(_sch_pv, "_react_var_urls", {}) or {}
                        _rv_pv = getattr(_sch_pv, "_react_vars", {}) or {}
                        _curr_n = 0
                        for _uk_pv, _vn_pv in _var_urls_pv.items():
                            if _classify_sampling_url(_uk_pv) != "detail":
                                continue
                            _val_pv = _rv_pv.get(_vn_pv, "")
                            if (isinstance(_val_pv, str) and len(_val_pv) >= 200
                                    and not _val_pv.lstrip().startswith("[⚠️ BLOCKED_PAGE")):
                                _curr_n += 1
                        _recorded_n = int(_blocked.get(var_name, 0) or 0)
                        if _curr_n <= _recorded_n:
                            # 진전 없음 — 하드 거부
                            # [Fix P0-1 2026-04-25] PRE-VETO strike 카운터 추적.
                            # Why: 기존 PRE-VETO 가 무한히 거부만 해서 25회 반복(plan stage 로그).
                            # 같은 var 누적 5회 도달 시 chain abort(정직 실패), 3회 도달부터
                            # 강한 pivot 토큰 prepend 로 LLM 행동 변경 유도.
                            _pv_strikes_map = getattr(_sch_pv, "_pre_veto_strikes", None)
                            if not isinstance(_pv_strikes_map, dict):
                                _pv_strikes_map = {}
                                try:
                                    _sch_pv._pre_veto_strikes = _pv_strikes_map
                                except Exception:
                                    pass
                            _pv_n = int(_pv_strikes_map.get(var_name, 0) or 0) + 1
                            _pv_strikes_map[var_name] = _pv_n
                            # [PRE-VETO 완화 2026-05-06] simple-task 시 임계 더 낮춤
                            _pv_simple_src = (
                                str(getattr(_sch_pv, "original_prompt", "") or "")
                                + " " + str(getattr(_sch_pv, "what", "") or "")
                                + " " + str(getattr(_sch_pv, "where", "") or "")
                            )
                            if _is_simple_task(_pv_simple_src):
                                PV_HARD_LIMIT = PV_HARD_LIMIT_SIMPLE
                                PV_STRONG_LIMIT = PV_STRONG_LIMIT_SIMPLE
                            else:
                                PV_HARD_LIMIT = PV_HARD_LIMIT_DEFAULT
                                PV_STRONG_LIMIT = PV_STRONG_LIMIT_DEFAULT

                            if _pv_n >= PV_HARD_LIMIT:
                                # [근본 fix 2026-05-04] 하드 리밋 도달 — gate 자체 무력화 +
                                # WRITE 통과. 기존엔 chain_id 있을 때만 abort_chain 후 return,
                                # chain_id 없으면 silently 빠져나가 strike 가 5/5→6/5→∞ 증가
                                # (사용자 로그 line 531). LLM 이 NAVIGATE 권유 무시하면 영원히
                                # stuck. PRE-VETO 는 advisory — 한계 도달 시 LLM 판단 존중.
                                # chain_id 있으면 metadata 기록 (severe 경로 보존) 하되 return
                                # 은 안 함 → 아래 정리 단계에서 strike/block clear 후 WRITE 진행.
                                _cid_pv = ""
                                try:
                                    _cid_pv = (
                                        getattr(_sch_pv, "chain_id", "") or ""
                                    ).strip()
                                    if _cid_pv:
                                        from eidos_mission_chain import (
                                            get_orchestrator as _go_pv,
                                        )
                                        _ch_pv = _go_pv().registry.get(_cid_pv)
                                        if _ch_pv is not None:
                                            _ch_pv.metadata["sampling_gate_unrecoverable"] = True
                                            _ch_pv.metadata["pre_veto_strikes"] = _pv_n
                                except Exception as _e_p01:
                                    print(f"  [P0-1] chain metadata 기록 실패 (무시): {_e_p01}")
                                # strike/block flag 정리 — 다음 WRITE 도 자유롭게
                                try:
                                    _blocked.pop(var_name, None)
                                    _pv_strikes_map.pop(var_name, None)
                                except Exception:
                                    pass
                                self._report(
                                    f"⚠️ **[WRITE 사전 거부 한계 도달 — gate 무력화]** "
                                    f"`{var_name}` — PRE-VETO {_pv_n}/{PV_HARD_LIMIT}회 "
                                    f"누적. detail 수집 진전 없으나 LLM 이 pivot 거부 → "
                                    f"gate 가 advisory 한계 인정하고 WRITE 통과 허용."
                                )
                                print(
                                    f"  ⚠️ [WRITE PRE-VETO BYPASS] '{var_name}' "
                                    f"strike={_pv_n}/{PV_HARD_LIMIT} 도달 — gate 무력화 "
                                    f"+ WRITE 통과 허용 (chain_id={_cid_pv or 'none'})."
                                )
                                # [Fix E2 2026-05-06] 진짜 fall through — 아래 거부 분기
                                # (print/_report/return) 진입 차단. 이전엔 BYPASS 메시지만
                                # 찍고 다음 거부 분기가 그대로 실행돼 WRITE 거부됐음
                                # (최신로그.txt line 1445~1448 BYPASS+거부 동시 출력).
                            else:
                                # 진전 없음 + 한도 미도달 — PRE-VETO 거부 진행
                                print(
                                    f"  ⛔ [WRITE PRE-VETO] '{var_name}' strike={_pv_n}/"
                                    f"{PV_HARD_LIMIT} — 직전 게이트 거부 후 detail var "
                                    f"{_recorded_n}→{_curr_n} (진전 없음). NAVIGATE 강제."
                                )
                                self._report(
                                    f"⛔ **[WRITE 사전 거부 {_pv_n}/{PV_HARD_LIMIT}]** "
                                    f"`{var_name}` — 직전 샘플링 게이트 거부 이후 detail "
                                    f"변수 진전 0개. dispatcher 가 WRITE 차단 — NAVIGATE/READ_PAGE "
                                    f"로 상세 페이지 추가 수집 필수."
                                )
                                try:
                                    if _pv_n >= PV_STRONG_LIMIT:
                                        _veto_hint = (
                                            f"🔴 [WRITE 사전 거부 {_pv_n}회 누적 — "
                                            f"{PV_HARD_LIMIT}회 도달 시 chain abort] "
                                            f"'{var_name}' WRITE 는 dispatcher 가 강제 차단. "
                                            f"detail 변수가 늘지 않으면 어떤 WRITE 도 통과 "
                                            f"못한다. **반드시 NAVIGATE → READ_PAGE 로 새 "
                                            f"상세 페이지부터 수집할 것**. WRITE 재시도 금지."
                                        )
                                    else:
                                        _veto_hint = (
                                            f"⛔ [WRITE 사전 거부 {_pv_n}회] '{var_name}' "
                                            f"WRITE 는 dispatcher 차단 상태. 직전 샘플링 "
                                            f"게이트 거부 후 detail 변수 추가 0개. **다음 "
                                            f"액션은 반드시 NAVIGATE → READ_PAGE**. WRITE "
                                            f"재시도 금지 (또 사전 거부됨)."
                                        )
                                    if hasattr(_sch_pv, "how"):
                                        _sch_pv.how = _veto_hint + "\n\n" + (_sch_pv.how or "")
                                except Exception:
                                    pass
                                return (
                                    f"⛔ WRITE 사전 거부 {_pv_n}회: '{var_name}' 은 직전 "
                                    f"샘플링 게이트 거부 이후 detail 변수가 늘지 않았다"
                                    f"(현재 {_curr_n}개). NAVIGATE /gig/... → READ_PAGE "
                                    f"gig_detail_<id> 로 상세 수집 후에만 WRITE 가능."
                                )
                        else:
                            # 진전 있음 — block flag + strike 카운터 둘 다 clear
                            try:
                                _blocked.pop(var_name, None)
                                _pv_strikes_map = getattr(
                                    _sch_pv, "_pre_veto_strikes", None
                                )
                                if isinstance(_pv_strikes_map, dict):
                                    _pv_strikes_map.pop(var_name, None)
                                print(
                                    f"  ✅ [WRITE PRE-VETO clear] '{var_name}' detail "
                                    f"{_recorded_n}→{_curr_n} 진전 — block + strike 해제"
                                )
                            except Exception:
                                pass
            except Exception as _e_pv:
                print(f"  ⚠️ [WRITE PRE-VETO] 체크 실패 (무시): {_e_pv}")

            # [Phase 1-A 2026-04-24] WRITE 샘플링 사전 게이트.
            # Why: "상위 N개 경쟁 상품" 류 태스크에서 detail READ_PAGE 변수가
            # 임계치 미만인데 WRITE 하면 Verifier missing_attr → retry → 같은
            # 1 샘플로 또 WRITE → ASK_USER 루프. 수집된 상세 페이지 변수 수가
            # 임계치 min(N, 5) 이상일 때만 WRITE 허용. 미만이면 미방문 drill-down
            # 링크 힌트와 함께 거부 메시지를 리턴해 다음 Think 이 NAVIGATE 를
            # 고르게 만든다. 게이트는 schema 당 최대 3회 발동 (무한루프 방어).
            try:
                _sch_sg = self._active_schema
                if _sch_sg is not None:
                    _gate_n = int(getattr(_sch_sg, "_write_sampling_gate_count", 0) or 0)
                    _attrs_sg = list(getattr(_sch_sg, "attributes", []) or [])
                    import re as _re_sg
                    _sg_sources = [getattr(_sch_sg, "original_prompt", "") or ""]
                    for _a in _attrs_sg:
                        _sg_sources.append(getattr(_a, "name", "") or "")
                    _sg_blob = " | ".join(str(s) for s in _sg_sources if s)
                    _N_required = 0
                    for _m in _re_sg.finditer(
                        r"(?:상위|top|탑)\s*(\d{1,3})|(\d{1,3})\s*(?:개|건|가지|종|품목)",
                        _sg_blob, _re_sg.I,
                    ):
                        _n_try = int(_m.group(1) or _m.group(2) or 0)
                        if 2 <= _n_try <= 100 and _n_try > _N_required:
                            _N_required = _n_try
                    # [List-as-Info Whitelist 2026-04-27] 네이버 지도/카카오맵 등
                    # LIST 페이지가 정보 완성인 도메인은 detail page 진입 강요
                    # 본질적으로 불필요 → sampling gate 자체 우회.
                    # 조건: 방문 URL 중 하나라도 whitelist 도메인 + 해당 URL 의
                    # READ_PAGE 본문이 200자 이상 확보됨 (실제 정보 수집됨 증거).
                    # _N_required 분기 진입 여부와 무관하게 먼저 평가해 N≥2 케이스에서만
                    # gate 본체 skip. 회귀 가드: 화이트리스트 미히트면 기존 흐름 그대로.
                    _bypass_sg = False
                    # [Strangler v2 2026-05-15] research(검색→종합) chain 은
                    # detail-page 수집 모델이 아니다. URL 화이트리스트로는
                    # SerpAPI research 를 못 잡아 게이트가 오발동(detail 영구 0
                    # → WRITE 무한 거부). v2 가 박은 면제 마커면 기존 bypass
                    # 인프라를 그대로 태워 게이트 우회. 마커 없는 수집형
                    # 작업(크몽/gumroad)은 영향 없음 → 회귀 0.
                    try:
                        if getattr(_sch_sg, "_research_no_sampling", False):
                            _bypass_sg = True
                            print(
                                "  ✅ [Sampling Gate bypass] v2 research 마커 "
                                "(_research_no_sampling) — detail 수집형 아님, 게이트 우회"
                            )
                    except Exception:
                        pass
                    _var_urls_sg_pre = getattr(_sch_sg, "_react_var_urls", {}) or {}
                    _rv_sg_pre = getattr(_sch_sg, "_react_vars", {}) or {}
                    _list_as_info_url = ""
                    _list_as_info_signals_sg = 0
                    # [의미 게이트 2026-04-27] 본문 길이뿐 아니라 평점·리뷰·거리 등
                    # place entity 신호 N개+ 일 때만 bypass. _N_required(TOP5 등) 가
                    # 있으면 비례 임계 (max(2, N//2)). false positive 방지로
                    # signal 이 부족하면 bypass 미발동 → 기존 sampling gate 가
                    # detail 부족을 자연스럽게 거부 (Verifier 까지 가지 않게).
                    for _uk_li, _vn_li in _var_urls_sg_pre.items():
                        if not _is_list_as_info_domain(_uk_li):
                            continue
                        _val_li = _rv_sg_pre.get(_vn_li, "")
                        if not (isinstance(_val_li, str) and len(_val_li) >= 200
                                and not _val_li.lstrip().startswith("[⚠️ BLOCKED_PAGE")):
                            continue
                        if _has_place_density(_val_li, n_required=_N_required):
                            _list_as_info_url = _uk_li
                            _list_as_info_signals_sg = _count_place_entities(_val_li)
                            _bypass_sg = True
                            break
                    if _bypass_sg and _N_required >= 2:
                        print(
                            f"  ✅ [Sampling Gate bypass] list-as-info 도메인 "
                            f"({_list_as_info_url[:60]}) 본문 수집됨 + place 신호 "
                            f"{_list_as_info_signals_sg}개 — gate 우회"
                        )
                        self._report(
                            f"✅ **[샘플링 게이트 우회]** 네이버지도/카카오맵 같은 "
                            f"LIST 페이지가 정보 완성인 도메인 — detail 진입 없이 "
                            f"WRITE 허용. (place 신호 {_list_as_info_signals_sg}개)"
                        )
                        # 이전 PRE-VETO 흔적도 함께 정리 — 다음 WRITE 도 자유롭게
                        try:
                            _blk_clr = getattr(_sch_sg, "_sampling_blocked_targets", None)
                            if isinstance(_blk_clr, dict):
                                _blk_clr.pop(var_name, None)
                            _stk_clr = getattr(_sch_sg, "_pre_veto_strikes", None)
                            if isinstance(_stk_clr, dict):
                                _stk_clr.pop(var_name, None)
                        except Exception:
                            pass

                    # [Fix E3 2026-05-06] schema HOW_MUCH 자동 진전 충족 시 Sampling Gate 우회.
                    # Why: Fix B 가 attr.achieved 자동 갱신 시키지만 Sampling Gate 의 detail
                    # counter 는 URL classifier 별도 채널 → schema 충족됐어도 detail=0 거부
                    # (최신로그.txt 10:06 line 1395). 모든 numeric HOW_MUCH 속성 achieved 면
                    # gate 우회 + PRE-VETO/block 흔적 정리.
                    if not _bypass_sg:
                        try:
                            import re as _re_e3sg
                            _attrs_e3sg = list(getattr(_sch_sg, "attributes", []) or [])
                            _has_count_e3sg = False
                            _all_ok_e3sg = True
                            for _a_e3sg in _attrs_e3sg:
                                _name_e3sg = (getattr(_a_e3sg, "name", "") or "")
                                if _re_e3sg.search(r"\d+", _name_e3sg):
                                    _has_count_e3sg = True
                                    if not getattr(_a_e3sg, "achieved", False):
                                        _all_ok_e3sg = False
                                        break
                            if _has_count_e3sg and _all_ok_e3sg:
                                print(
                                    f"  ✅ [Fix E3] Sampling Gate 우회 — "
                                    f"schema HOW_MUCH numeric 속성 모두 achieved"
                                )
                                self._report(
                                    f"✅ **[샘플링 게이트 우회 — schema 충족]** "
                                    f"`{var_name}` — schema HOW_MUCH 모든 numeric 속성 "
                                    f"achieved (Fix B AutoProgress) — gate 무력화."
                                )
                                _bypass_sg = True
                                try:
                                    _blk_clr_e3 = getattr(
                                        _sch_sg, "_sampling_blocked_targets", None
                                    )
                                    if isinstance(_blk_clr_e3, dict):
                                        _blk_clr_e3.pop(var_name, None)
                                    _stk_clr_e3 = getattr(_sch_sg, "_pre_veto_strikes", None)
                                    if isinstance(_stk_clr_e3, dict):
                                        _stk_clr_e3.pop(var_name, None)
                                except Exception:
                                    pass
                        except Exception as _e_e3sg:
                            print(f"  [Fix E3 SG] 검사 실패 (무시): {_e_e3sg}")

                    if _N_required >= 2 and not _bypass_sg:
                        # Phase B [2026-04-27] module-level classifier + list/other 가시화
                        _var_urls_sg = _var_urls_sg_pre
                        _rv_sg = _rv_sg_pre
                        _detail_vars: list = []
                        _list_vars: list = []
                        _other_vars: list = []
                        for _uk_sg, _vn_sg in _var_urls_sg.items():
                            _val_sg = _rv_sg.get(_vn_sg, "")
                            if not (isinstance(_val_sg, str) and len(_val_sg) >= 200
                                    and not _val_sg.lstrip().startswith("[⚠️ BLOCKED_PAGE")):
                                continue
                            _kind_sg = _classify_sampling_url(_uk_sg)
                            if _kind_sg == "detail":
                                _detail_vars.append(_vn_sg)
                            elif _kind_sg == "list":
                                _list_vars.append(_vn_sg)
                            else:
                                _other_vars.append(_vn_sg)
                        _pool_sg = getattr(_sch_sg, "_drill_down_pool", []) or []
                        _orig_N = _N_required
                        _orig_threshold = min(_N_required, 5)
                        _threshold = _orig_threshold

                        # [Phase 1-A2 2026-04-24] 샘플링 게이트 ↔ 플랫폼 가용 풀 동기화.
                        # Why: 로그(line 576/691/891) 에서 LLM 이 "상위 20개" 요구했는데
                        # kmong 검색 결과는 ~10개만 노출 → drill-down 풀 다 방문해도 임계
                        # 충족 불가 → 게이트 영원히 거부 → WRITE 폭발 + chain stuck.
                        # 풀 측정값 < 임계 면 _N_required/_threshold 를 가용 한도로 자동
                        # 하향 + schema 에 영구 기록(다음 게이트 호출에서 재계산/재발 방지).
                        _saved_N = int(
                            getattr(_sch_sg, "_n_required_downgraded", 0) or 0
                        )
                        if _saved_N > 0 and _saved_N < _N_required:
                            # 직전 게이트에서 다운그레이드된 값 이어받기 (재발 방지).
                            _N_required = _saved_N
                            _threshold = min(_N_required, _orig_threshold)
                        elif _pool_sg:
                            # 풀이 한 번이라도 채워졌음 = 검색 결과 수가 측정됨.
                            # 풀 미방문 + 이미 수집된 detail = 도달 가능 최대치.
                            _pool_unvisited_n = sum(
                                1 for p in _pool_sg
                                if isinstance(p, dict) and not p.get("visited")
                            )
                            _max_attainable = len(_detail_vars) + _pool_unvisited_n
                            if _max_attainable < _threshold:
                                _new_N = max(_max_attainable, 1)
                                _new_threshold = min(_new_N, _orig_threshold)
                                try:
                                    _sch_sg._n_required_downgraded = _new_N
                                except Exception:
                                    pass
                                print(
                                    f"  🔻 [Sampling Downgrade] 사이트 풀 가용 "
                                    f"{len(_pool_sg)}개 (미방문 {_pool_unvisited_n}, "
                                    f"수집 {len(_detail_vars)}) — 요구 "
                                    f"{_orig_N}→{_new_N}, 임계 "
                                    f"{_orig_threshold}→{_new_threshold}"
                                )
                                # schema.how 에 다운그레이드 사실 기록 — Verifier/후속 stage 도 본다.
                                try:
                                    _dn_note = (
                                        f"[샘플링 다운그레이드 — 사이트 풀 {len(_pool_sg)}개"
                                        f" 한정 → 요구 {_orig_N}→{_new_N}, "
                                        f"임계 {_orig_threshold}→{_new_threshold}]"
                                    )
                                    if (hasattr(_sch_sg, "how")
                                            and _dn_note not in (_sch_sg.how or "")):
                                        _sch_sg.how = _dn_note + "\n" + (_sch_sg.how or "")
                                except Exception:
                                    pass
                                # 사용자 통지 — schema 당 1회만(여러 WRITE 게이트 반복 시 노이즈 방지).
                                if not getattr(_sch_sg, "_downgrade_notified", False):
                                    self._report(
                                        f"🔻 **[샘플링 다운그레이드]** `{var_name}` — "
                                        f"사이트 풀 {len(_pool_sg)}개 (수집 "
                                        f"{len(_detail_vars)} / 미방문 {_pool_unvisited_n}) "
                                        f"가 요구 {_orig_N}개에 미달 → 실현 가능 목표 "
                                        f"**{_new_N}개**(임계 {_new_threshold}개)로 자동 조정.\n"
                                        f"└ 플랫폼 제약상 더 모을 수 없는 경우 자동 진행."
                                    )
                                    try:
                                        _sch_sg._downgrade_notified = True
                                    except Exception:
                                        pass
                                _N_required = _new_N
                                _threshold = _new_threshold

                        # [P1-4 2026-04-25] gate_n >= 2 (즉 3번째 거부 시점) 누적이면
                        # ChannelPivot/EscapeHatch 신호. 이전엔 3회 거부 후 WRITE 통과로
                        # 빈 산출물 폭주, chain force_advance 로 흘러갔음. 여기서 chain.metadata
                        # 에 research_blocked 마커 + abort_chain reason 에 BAN 키워드 포함.
                        if len(_detail_vars) < _threshold and _gate_n >= 2:
                            # [근본 fix 2026-05-04] 누적 거부 한계 도달 — gate bypass + WRITE 통과.
                            # 기존엔 chain_id 있을 때만 abort + return, chain_id 없으면 silently
                            # 빠져나가 다음 분기 (_gate_n < 3) 까지 도달해 또 거부 → PRE-VETO 와
                            # 맞물려 무한 루프. chain_id 유무와 무관하게 한계 도달 시 gate 자체
                            # 무력화. chain_id 있으면 metadata 만 기록 (관측용).
                            _cid_sg = ""
                            try:
                                _cid_sg = (
                                    getattr(self._active_schema, "chain_id", "") or ""
                                ).strip() if self._active_schema else ""
                                if _cid_sg:
                                    from eidos_mission_chain import get_orchestrator as _go_sg
                                    _ch_sg = _go_sg().registry.get(_cid_sg)
                                    if _ch_sg is not None:
                                        _ch_sg.metadata["research_blocked"] = True
                                        _ch_sg.metadata["sampling_gate_strikes"] = _gate_n + 1
                            except Exception as _e_p14:
                                print(f"  [P1-4] chain metadata 기록 실패 (무시): {_e_p14}")
                            # blocked flag 정리 — 다음 PRE-VETO 진입 차단
                            try:
                                _bt_clr = getattr(_sch_sg, "_sampling_blocked_targets", None)
                                if isinstance(_bt_clr, dict):
                                    _bt_clr.pop(var_name, None)
                                _stk_clr = getattr(_sch_sg, "_pre_veto_strikes", None)
                                if isinstance(_stk_clr, dict):
                                    _stk_clr.pop(var_name, None)
                            except Exception:
                                pass
                            self._report(
                                f"⚠️ **[샘플링 게이트 한계 도달 — bypass]** "
                                f"`{var_name}` 요구 {_N_required} 임계 {_threshold} 수집 "
                                f"{len(_detail_vars)} — {_gate_n + 1}회 거부 누적. 플랫폼 가용 "
                                f"풀 한계로 추정 → gate 무력화하고 WRITE 통과 허용."
                            )
                            print(
                                f"  ⚠️ [Sampling Gate BYPASS] strikes={_gate_n + 1} → "
                                f"gate 무력화 + WRITE 통과 (chain_id={_cid_sg or 'none'})."
                            )
                            # fall through (return 없음) → WRITE 진행
                        if len(_detail_vars) < _threshold and _gate_n < 3:
                            _sch_sg._write_sampling_gate_count = _gate_n + 1
                            # [Phase 3 2026-04-25] 다음 WRITE 진입 시 PRE-VETO 트리거를 위해
                            # 현재 detail var 개수 기록. 다음 WRITE 가 같은 target 으로 들어오면
                            # detail 개수 진전 없음 → dispatcher 가 게이트 진입 전 하드 거부.
                            try:
                                _bt = getattr(_sch_sg, "_sampling_blocked_targets", None)
                                if not isinstance(_bt, dict):
                                    _bt = {}
                                _bt[var_name] = len(_detail_vars)
                                _sch_sg._sampling_blocked_targets = _bt
                                print(
                                    f"  📌 [Sampling Block] '{var_name}' 기록 detail "
                                    f"baseline={len(_detail_vars)} (다음 WRITE pre-veto 대기)"
                                )
                            except Exception as _e_bt:
                                print(f"  ⚠️ [Sampling Block] 기록 실패: {_e_bt}")
                            _unvisited_sg = [
                                p for p in _pool_sg
                                if isinstance(p, dict) and not p.get("visited")
                            ]
                            _hint_lines_sg = [
                                f"- {p.get('text') or '?'} → {p.get('url', '')[:100]}"
                                for p in _unvisited_sg[:5]
                            ]
                            _hint_block_sg = (
                                "\n".join(_hint_lines_sg) if _hint_lines_sg
                                else "(드릴다운 풀 비어있음 — 목록 페이지로 NAVIGATE 후 상세 링크 재추출)"
                            )
                            print(
                                f"  ⛔ [WRITE Sampling Gate {_gate_n + 1}/3] "
                                f"요구={_N_required} 임계={_threshold} "
                                f"detail={len(_detail_vars)} list={len(_list_vars)} "
                                f"other={len(_other_vars)} — WRITE 거부"
                            )
                            self._report(
                                f"⛔ **[WRITE 샘플링 게이트]** `{var_name}` 요구 "
                                f"{_N_required}개 / 임계 {_threshold}개 · 수집 "
                                f"detail={len(_detail_vars)}/list={len(_list_vars)}/"
                                f"other={len(_other_vars)} 미달 — 상세 페이지 추가 수집 유도."
                            )
                            try:
                                _list_note_sg = (
                                    f"\n[⚠️ 분류 — list 페이지({len(_list_vars)}개) 는 detail "
                                    f"카운트 미포함]: {_list_vars[:3]}" if _list_vars else ""
                                )
                                _gate_hint_sg = (
                                    f"⛔ [WRITE 샘플링 부족] '{var_name}' 은 상위 "
                                    f"{_N_required}개 샘플 태스크인데 detail "
                                    f"{len(_detail_vars)}개(임계 {_threshold})뿐이다. 다음 액션은 "
                                    f"반드시 NAVIGATE → READ_PAGE 로 상세 페이지를 최소 "
                                    f"{max(1, _threshold - len(_detail_vars))}개 추가 확보 후 WRITE."
                                    f"{_list_note_sg}\n"
                                    f"[미방문 상세 후보 — 이 중 하나로 NAVIGATE]:\n{_hint_block_sg}\n"
                                    f"각 상세 페이지 도착 후 READ_PAGE 로 'gig_detail_<id>' 등 "
                                    f"변수명으로 저장해야 게이트가 풀린다. 이미 방문한 URL 재NAVIGATE 금지."
                                )
                                if hasattr(schema, "how"):
                                    schema.how = _gate_hint_sg + "\n\n" + (schema.how or "")
                            except Exception:
                                pass
                            return (
                                f"⛔ WRITE 거부(샘플 부족): 요구 {_N_required}개 / 임계 "
                                f"{_threshold} / detail={len(_detail_vars)} "
                                f"list={len(_list_vars)} other={len(_other_vars)}"
                                + (f" detail=[{', '.join(_detail_vars[:3])}]" if _detail_vars else "")
                                + (f" list=[{', '.join(_list_vars[:3])}]" if _list_vars else "")
                                + f". list 페이지는 detail 카운트 미포함 — 상세 진입 필요. "
                                f"다음 step 은 NAVIGATE /gig/... /l/... 등으로 미방문 상세 이동 "
                                f"→ READ_PAGE 로 gig_detail_<id> 변수 저장 → {_threshold}개 "
                                f"모이면 WRITE. 미방문 후보:\n" + _hint_block_sg
                            )
                        elif _N_required >= 2:
                            # 게이트 통과 — Verifier 에 실측 샘플 수를 스키마 힌트로 남긴다.
                            try:
                                _note_sg = (
                                    f"[실측 샘플 {len(_detail_vars)}개 확보 — 요청 {_N_required} "
                                    f"대비 임계 {_threshold} 충족]"
                                )
                                if hasattr(schema, "how") and _note_sg not in (schema.how or ""):
                                    schema.how = _note_sg + "\n" + (schema.how or "")
                            except Exception:
                                pass
                            # [Phase 3 2026-04-25] 게이트 통과 — pre-veto block flag clear.
                            try:
                                _bt_pass = getattr(_sch_sg, "_sampling_blocked_targets", None)
                                if isinstance(_bt_pass, dict) and var_name in _bt_pass:
                                    _bt_pass.pop(var_name, None)
                                    print(
                                        f"  ✅ [Sampling Block clear] '{var_name}' 게이트 통과 "
                                        f"— pre-veto 해제"
                                    )
                            except Exception:
                                pass
            except Exception as _e_sg:
                print(f"  ⚠️ [WRITE Sampling Gate] 체크 실패 (무시): {_e_sg}")

            # [Fix 2026-04-19] 무한 WRITE 루프 + 내용 날려먹기 방지.
            # 같은 var_name에 이미 내용이 있으면 분석가 LLM에 "기존 문서 확장" 컨텍스트를 주입.
            # Why: 이전엔 content(짧은 브리프)만 전달해 LLM이 매번 새 템플릿을 빈 헤더로 재생성 →
            #      step 5에서 1552자 쓰고 step 6·7에서 28자로 덮어써 내용 소실 반복.
            _prior_content = plan.variables.get(var_name, "") or ""
            # 파일 경로면 디스크에도 조회 — 플랜 메모리가 비어있어도 파일엔 남아있을 수 있음
            _looks_like_file = (
                ("/" in var_name or "\\" in var_name) or
                var_name.lower().endswith((
                    ".md", ".txt", ".json", ".csv", ".html",
                    ".log", ".yaml", ".yml", ".xml", ".py", ".js",
                ))
            )
            # [Fix 2026-04-19 v2] 변수명이 "내용 계열"(목록/보고서/기획/draft/list/report/plan/
            # proposal/document/summary/analysis/문서/내용/설명/요약)이면 확장자 없어도
            # 최종 산출물로 간주하고 {var}.md 로 자동 승격 저장. Why: LLM이 target을
            # 관성적으로 `top_services_list`, `product_proposal_draft` 같은 snake_case 변수명으로
            # 쓰는데 이전 코드는 이걸 메모리에만 담아 세션 종료 시 소실.
            _CONTENT_VAR_KWS = (
                "목록", "리스트", "list",
                "보고서", "report", "reports",
                "기획", "plan", "planning", "proposal",
                "분석", "analysis", "analyses",
                "draft", "drafts",
                "document", "documents", "문서", "doc",
                "summary", "summaries", "요약",
                "내용", "content", "contents",
                "설명", "description",
                "article", "post", "writeup",
            )
            _vn_lc = var_name.lower()
            _is_content_var = (
                (not _looks_like_file)
                and "." not in var_name          # 확장자 전혀 없음
                and "/" not in var_name and "\\" not in var_name
                and len(var_name) <= 80
                and any(kw in _vn_lc for kw in _CONTENT_VAR_KWS)
            )
            _auto_file_path = None               # {var}.md 로 승격된 경우 여기에 세팅
            if _is_content_var:
                # 변수명 sanitize — 파일명 불가 문자 제거 (공백·콜론 등)
                import re as _re_vsan
                _safe_name = _re_vsan.sub(r"[^\w\-\.]+", "_", var_name).strip("_")
                if _safe_name:
                    _auto_file_path = f"{_safe_name}.md"
            # prior-content 조회 대상: 명시적 파일 경로(_looks_like_file) 또는 자동 승격({var}.md)
            _read_path_candidate = var_name if _looks_like_file else (_auto_file_path or "")

            # ── [Carry-Over Guard 2026-04-27] 새 task 의 첫 WRITE 가 이전 task 의 ──
            # 잔존 디스크 파일을 [기존 문서 — 유지·확장하라] 로 흡수하는 누수 차단.
            # 최신로그.txt — "임진왜란" 새 task 의 Step 2 [WRITE 보고서.txt] 가 디스크의
            # 이전 "강남역 파스타" 5671자를 prior_content 로 읽어 LLM 프롬프트에 박음 →
            # 외부 차단 명령 받은 LLM 이 "기존 본문 유지" 지시 따라 6989자 dump → drift FAIL.
            # 정당한 케이스 (chain stage retry / 같은 schema 안에서 같은 var WRITE 재시도)
            # 만 디스크 fallback 허용. 그 외 첫 WRITE 는 fresh 시작.
            _prior_disk_allowed = True
            try:
                _sch_co = self._active_schema
                if _sch_co is not None:
                    _has_chain = bool((getattr(_sch_co, "chain_id", "") or "").strip())
                    # history 에서 같은 var_name 으로 verified WRITE 가 1건 이상이면 retry 의도
                    _hist_co = getattr(_sch_co, "react_history", None) or []
                    if not isinstance(_hist_co, list):
                        _hist_co = []
                    _prior_write_in_history = False
                    _vn_co = (var_name or "").strip().lower()
                    for _h_co in _hist_co:
                        if not isinstance(_h_co, dict):
                            continue
                        if (str(_h_co.get("action", "")).upper() == "WRITE"
                                and str(_h_co.get("target", "")).strip().lower() == _vn_co
                                and _h_co.get("verified")):
                            _prior_write_in_history = True
                            break
                    # repair retry 진입 마커 (Retry Target Lock 활성)
                    _is_retry_run = bool(getattr(_sch_co, "_is_retry_run", False))
                    # 정당 사유: chain stage 진행 OR 이번 schema 안에서 기존 WRITE 있음 OR retry
                    if not (_has_chain or _prior_write_in_history or _is_retry_run):
                        _prior_disk_allowed = False
            except Exception as _e_co:
                # 가드 자체 예외 시 기존 동작 보존 (fail-open)
                _prior_disk_allowed = True

            if not _prior_content and _read_path_candidate and _prior_disk_allowed:
                try:
                    from execution_module import SAFE_BASE_PATH as _SBASE_R
                    import os as _os_r
                    _rel_r = _read_path_candidate.replace("\\", "/")
                    while _rel_r.startswith("./"):
                        _rel_r = _rel_r[2:]
                    _base_r = _os_r.path.basename(_os_r.path.normpath(_SBASE_R)).lower()
                    if _base_r and _rel_r.lower().startswith(_base_r + "/"):
                        _rel_r = _rel_r[len(_base_r) + 1:]
                    _abs_r = _os_r.path.abspath(_os_r.path.join(_SBASE_R, _rel_r))
                    _base_abs_r = _os_r.path.abspath(_SBASE_R)
                    if (_abs_r == _base_abs_r or _abs_r.startswith(_base_abs_r + _os_r.sep)) \
                       and _os_r.path.isfile(_abs_r):
                        with open(_abs_r, "r", encoding="utf-8") as _fh_r:
                            _prior_content = _fh_r.read()
                except Exception:
                    _prior_content = ""
            elif not _prior_content and _read_path_candidate and not _prior_disk_allowed:
                # 이전 task 잔존 파일 무시 — fresh write
                print(
                    f"  🛡️ [WRITE Carry-Over Guard] 새 task 첫 WRITE — "
                    f"디스크 '{_read_path_candidate[:40]}' 잔존 파일 무시 (fresh start)"
                )

            _write_prompt = content

            # [Fix 2026-04-20] 원천 데이터 변수(READ_PAGE 결과 등) 자동 노출.
            # [Fix 2026-04-21] content<2000 게이트 제거, 길이 임계 200자로 하향,
            #   강제 인용 규칙 추가. Why: 분석가 LLM이 원천 변수를 "<body> 없음"
            #   이라 부정하는 자기부정 환각 차단 목적. 강제 인용 → 환각 시 자기 반증.
            _available_vars_block = ""
            _available_var_summaries: list[tuple[str, int]] = []

            # [Fix 2026-04-22] ad-hoc 단순 조회용 간결 요약 모드 감지.
            # chain 소속이 아니고 task_type 이 조회/요약 계열이면 "300자 복붙 +
            # 1500자 분석" 강제를 해제하고 간결 요약 프롬프트로 분기.
            # 증상: '서울 날씨' 같은 단건 태스크에서 네이버 페이지 원문이 60줄
            # 통째로 인용돼 파일에 붙는 현상. 규칙 1(복붙) + 규칙 4(1500자) 가 원인.
            _concise_mode = False
            try:
                _sch_cm = self._active_schema
                if _sch_cm is not None:
                    _cid_cm = (getattr(_sch_cm, "chain_id", "") or "").strip()
                    _tt_cm = (getattr(_sch_cm, "task_type", "") or "").strip().lower()
                    _concise_mode = (not _cid_cm) and _tt_cm in {
                        "browser_read", "lookup", "quick_query", "chat", "",
                    }
            except Exception:
                _concise_mode = False

            # [Synthesis Guard 2026-05-17] 자기자산 카드 + 종합문서 판정.
            #   memory: diagnosis_eidos_paraphrase_machine — plan/strategy 문서가
            #   EIDOS 자기 표상 없이 최근접 페르소나로 빈칸을 채우고, 원천을
            #   풀어쓰기만 하는 결함의 방어층. 미배선 모듈 import 부작용 0.
            _SG = None
            _self_asset_blk = ""
            _synth_kind = ""
            try:
                import eidos_synthesis_guard as _SG  # type: ignore
                _synth_kind = _SG.is_synthesis_doc(var_name or "")
                _self_asset_blk = _SG.self_asset_block(None)
            except Exception as _e_sg:
                print(f"  ⚠️ [WRITE] synthesis guard 준비 실패 (무시): {_e_sg}")
                _SG = None

            try:
                _plan_vars = getattr(plan, "variables", None) or {}
            except Exception:
                _plan_vars = {}
            # [P3 2026-04-21] chain upstream_outputs + schema._react_vars 를 plan.variables
            # 에 병합. 우선순위: upstream_outputs > _react_vars > plan.variables.
            # 증상: plan/dev stage 에서 upstream(research_report.md 3366자) 이 plan.variables
            # 에는 없어서 LLM 에 안 넘어갔음 — WRITE Gate 가 만든 page_text(1331자) 만 보이니
            # LLM 이 "재료 없음" 으로 환각 draft. upstream 을 항상 주입 + 확장자 alias 도 등록.
            try:
                _sch_p3 = self._active_schema
                _up_p3 = (getattr(_sch_p3, "upstream_outputs", None) or {}) if _sch_p3 else {}
                _rv_p3 = (getattr(_sch_p3, "_react_vars", None) or {}) if _sch_p3 else {}
                _merged_p3: dict = {}
                for _k, _v in _plan_vars.items():
                    _merged_p3[_k] = _v
                for _k, _v in _rv_p3.items():
                    if _k not in _merged_p3:
                        _merged_p3[_k] = _v
                # upstream 은 덮어쓰기(최우선) + stage prefix tail alias 도 등록
                for _k, _v in _up_p3.items():
                    if _v is None:
                        continue
                    _merged_p3[_k] = _v
                    if isinstance(_k, str) and "." in _k:
                        _tail = _k.split(".", 1)[1]
                        if _tail and _tail not in _merged_p3:
                            _merged_p3[_tail] = _v
                _plan_vars = _merged_p3
            except Exception as _e_merge:
                print(f"  ⚠️ [WRITE] upstream/_react_vars 병합 실패 (무시): {_e_merge}")
            # [P3] 정렬: upstream 소속 키를 먼저, 나머지는 뒤. top 인용 대상도
            # upstream 중 가장 긴 값이 오도록 유도.
            _upstream_keyset: set = set()
            try:
                _sch_p3b = self._active_schema
                _up_p3b = (getattr(_sch_p3b, "upstream_outputs", None) or {}) if _sch_p3b else {}
                for _uk in _up_p3b.keys():
                    _upstream_keyset.add(_uk)
                    if isinstance(_uk, str) and "." in _uk:
                        _upstream_keyset.add(_uk.split(".", 1)[1])
            except Exception:
                _upstream_keyset = set()
            if _plan_vars:
                _var_blocks = []
                # upstream 우선 → 나머지 순으로 iterate
                _ordered_items = sorted(
                    _plan_vars.items(),
                    key=lambda kv: (0 if kv[0] in _upstream_keyset else 1),
                )
                # [2026-04-24] 모든 원천이 차단 페이지면 WRITE 거부 + alternative 유도.
                # Why: smartstore_search_results 가 NAVER 보안 확인 페이지로만 채워졌을 때
                # LLM 이 "데이터 없음" 1500자 환각을 작성하고 다음 step 으로 넘어갔다.
                # 변수 본문에 [⚠️ BLOCKED_PAGE] 마커가 있는 것 / _is_blocked_by_text 매칭
                # 둘 중 하나라도 있으면 차단 변수로 카운트. 유효 변수가 0개면 즉시 returh.
                _valid_vars = 0
                _blocked_vars: list = []
                for _vn, _vv in _ordered_items:
                    if not isinstance(_vn, str) or _vn.startswith("__"):
                        continue
                    if _vn == var_name:
                        continue
                    if not isinstance(_vv, str) or len(_vv) < 200:
                        continue
                    _is_blk_marker = _vv.lstrip().startswith("[⚠️ BLOCKED_PAGE")
                    _is_blk_text, _ = _is_blocked_by_text(_vv)
                    if _is_blk_marker or _is_blk_text:
                        _blocked_vars.append(_vn)
                    else:
                        _valid_vars += 1
                if _blocked_vars and _valid_vars == 0:
                    print(
                        f"  ⛔ [WRITE BLOCK 가드] 모든 원천 변수({len(_blocked_vars)}개)가 "
                        f"차단/오류 페이지 — WRITE 환각 차단. 변수: {_blocked_vars[:3]}"
                    )
                    self._report(
                        f"⛔ **[WRITE 차단 가드]** 모든 원천 변수가 보안/오류 페이지 "
                        f"({', '.join(_blocked_vars[:3])}). '데이터 없음' 환각 보고서 작성 거부. "
                        f"다음 step 은 alternative URL/플랫폼 NAVIGATE 또는 ASK_USER 로 진행."
                    )
                    return (
                        f"⛔ WRITE 거부(차단 페이지 원천): 모든 원천 변수("
                        f"{', '.join(_blocked_vars[:3])})가 NAVER 보안/오류 페이지였다. "
                        f"본문 환각 작성 대신 다음 step 에서 alternative 검색 경로로 NAVIGATE "
                        f"하거나 ASK_USER 로 우회 방법을 사용자에게 물어봐라."
                    )

                for _vn, _vv in _ordered_items:
                    if not isinstance(_vn, str) or _vn.startswith("__"):
                        continue
                    if _vn == var_name:
                        continue
                    if not isinstance(_vv, str) or len(_vv) < 200:
                        continue
                    _available_var_summaries.append((_vn, len(_vv)))
                    _preview = _vv[:3500]
                    _tail = "\n...[이하 생략]" if len(_vv) > 3500 else ""
                    _var_blocks.append(f"### {{{_vn}}} ({len(_vv)}자)\n{_preview}{_tail}")
                if _var_blocks:
                    _top_vn, _top_len = max(_available_var_summaries, key=lambda x: x[1])
                    if _concise_mode:
                        # [Fix 2026-04-22] ad-hoc 간결 모드: 복붙·최소 1500자 규칙 제거.
                        # 핵심 사실만 요약하도록 지시. 페이지 원문이 통째 복사되는 문제 차단.
                        _available_vars_block = (
                            "[원천 데이터 — 아래 변수를 근거로 **요점만 간결히** 정리하라.]\n"
                            "[규칙]\n"
                            "1) 원문을 그대로 인용/복붙 금지. 핵심 사실만 자연어 문장으로 요약.\n"
                            "2) 메뉴·네비게이션·광고 라벨·반복 항목은 모두 버려라.\n"
                            "3) 숫자·수치·날짜·지명 등 '실제 정보' 만 남겨라. 없으면 '확인 불가' 로 표기.\n"
                            "4) 사용자가 원하는 답에 직접 필요한 내용만 남기고, 불필요한 서론·면책·후기 금지.\n"
                            "5) 길이는 필요한 만큼만 — 억지로 늘리지 마라.\n\n"
                            + "\n\n".join(_var_blocks)
                            + "\n\n"
                        )
                    else:
                        _available_vars_block = (
                            "[사용 가능한 원천 데이터 변수 — 아래 내용을 근거로 본문을 작성하라.]\n"
                            "[필수 규칙 — 위반 시 보고서가 자동 폐기된다]\n"
                            f"1) 본문 첫 문단에서 '### {{{_top_vn}}}' 변수의 처음 300자 이상을 "
                            "**따옴표(\"\"\")로 그대로 복붙**한 뒤 분석을 시작하라. 요약/의역 금지.\n"
                            "2) 아래 변수들을 근거로 구체 수치·서비스명·문구를 **직접 추출**하라. "
                            "추측/환각 금지.\n"
                            "3) '<body> 없음', '데이터 존재하지 않음', '분석 불가능', "
                            "'생성할 수 없습니다' 류 표현 **절대 금지**. "
                            "변수가 아래에 실제로 주어졌으므로 '데이터가 없다'는 판단은 거짓이다.\n"
                            "4) [필수] 위 1) 의 300자 복붙 구간 **이후**에, 인용문에서 뽑은 "
                            "가격/리뷰/서비스명/키워드 등을 근거로 **최소 1500자 분량의 한글 분석 본문**"
                            "을 작성하라. '복붙만 하고 끝'은 불합격 — HTML 태그나 원천 변수 텍스트를 "
                            "그대로 길게 붙여 넣어 글자수를 채우는 것도 금지. 분석 본문은 실제 서술이어야 한다.\n"
                            "5) 만약 원천 변수에서 가격·리뷰·서비스명을 **정말로** 하나도 추출할 수 없다고 "
                            "판단되면, 그 사유를 최소 200자로 설명하되 '단, 확인된 사실은 다음과 같다: …' "
                            "형식으로 확보된 모든 부분 정보(브랜드/카테고리/페이지 구조 등)를 나열하라. "
                            "그냥 '불가능'으로 끝내지 마라.\n\n"
                            + "\n\n".join(_var_blocks)
                            + "\n\n"
                        )
                    # [Synthesis Guard 2026-05-17] 종합문서(plan/research)는
                    #   "원천 300자 통째 복붙" 명령을 *근거 인용* 으로 형태 변경.
                    #   복붙 명령은 자기부정 환각(데이터 없음) 방지 장치였으므로
                    #   제거가 아니라 인용 앵커는 보존하면서 본문은 종합 강제.
                    #   비-종합/concise 문서는 기존 동작 그대로.
                    if _synth_kind and not _concise_mode and _available_vars_block:
                        _cp_old = (
                            f"1) 본문 첫 문단에서 '### {{{_top_vn}}}' 변수의 처음 300자 이상을 "
                            "**따옴표(\"\"\")로 그대로 복붙**한 뒤 분석을 시작하라. 요약/의역 금지.\n"
                        )
                        _cp_new = (
                            "1) 원천에서 추출한 **구체 수치·고유명사·핵심 문구**만 "
                            "따옴표로 *근거 인용* 하라(환각 자기반증 앵커 — 데이터가 "
                            "실제로 주어졌음을 인용으로 증명). 원천 문단을 통째로 "
                            "복붙하지 마라. 본문은 너의 종합·판단·대조·시사점으로 "
                            "서술하고, 인용은 근거 제시 용도로만 짧게 사용하라.\n"
                        )
                        if _cp_old in _available_vars_block:
                            _available_vars_block = _available_vars_block.replace(
                                _cp_old, _cp_new, 1
                            )
                            print(
                                f"  🧩 [Synthesis Guard] 종합문서('{_synth_kind}') — "
                                f"복붙 명령을 근거 인용으로 형태 변경"
                            )
                    # [Fix 2026-04-21] WRITE 시점 변수 주입 가시화 로그.
                    print(
                        f"  📥 [WRITE] 원천 변수 {len(_available_var_summaries)}개 주입: "
                        + ", ".join(f"{{{vn}}}({ln}자)" for vn, ln in _available_var_summaries)
                    )

            # [P4 2026-04-21] 재료 부족 자체검사. 같은 파일 WRITE 에 주입된 원천
            # 변수 총량이 이전 시도 대비 급감했으면 LLM 호출 스킵 + 복구 지시 리턴.
            # 증상: plan.variables clear, _react_vars 덮어씀, upstream 누락 등으로
            # 재료가 줄면 LLM 이 환각 draft 만 반복 → Shortening Guard 루프. 예방층.
            try:
                _cur_mat = sum(ln for _, ln in _available_var_summaries)
                _sch_p4 = self._active_schema
                if _sch_p4 is not None:
                    _mat_key = (
                        (getattr(_sch_p4, "chain_id", "") or ""),
                        int(getattr(_sch_p4, "stage_index", 0) or 0),
                        (var_name or "").strip().lower(),
                    )
                    _prev_mat = self._write_material_history.get(_mat_key, 0)
                    if _prev_mat >= 1000 and _cur_mat < int(_prev_mat * 0.3):
                        print(
                            f"  🚨 [WRITE Material Drop] '{var_name}' 원천 급감: "
                            f"{_prev_mat}자 → {_cur_mat}자 — LLM 호출 스킵, 복구 지시"
                        )
                        return (
                            f"⚠️ WRITE 스킵(재료 소실): 이전 시도엔 {_prev_mat}자 원천이 "
                            f"주입됐는데 이번엔 {_cur_mat}자 뿐이다. upstream 산출물이 "
                            f"변수에서 날아갔거나 READ_PAGE 가 엉뚱한 var 를 덮었다. "
                            f"다음 step 은 — (a) READ_FILE 로 이전 stage 산출물을 변수에 "
                            f"다시 읽어오거나, (b) 원천이 확보된 뒤 WRITE 하라. "
                            f"재료 없이 WRITE 재시도 금지."
                        )
                    # 최대값으로만 누적 — 노이즈 방지.
                    self._write_material_history[_mat_key] = max(_prev_mat, _cur_mat)
            except Exception as _e_p4:
                print(f"  ⚠️ [WRITE P4] 재료 추적 실패 (무시): {_e_p4}")

            # [Fix 2026-04-21] 미달성 HOW_MUCH 속성을 분석가에게 명시 강제 주입.
            # Why: '고정 주제 'CRM 팩' 언급 여부' 같이 본문에 특정 키워드가 들어가야
            #      하는 속성, '후보 3개 포함' 같이 항목 수가 필요한 속성, '제목 명시'
            #      같이 구조를 요구하는 속성을 LLM 이 알아서 챙기지 못해 영구 미달성
            #      → Verifier FAIL → 재시도 → parser_error 루프. 프롬프트 머리에
            #      '필수 충족 조건' 블록으로 박아 LLM 이 빠뜨릴 수 없게 한다.
            _howmuch_block = ""
            try:
                _sch_hm = self._active_schema
                _attrs_hm = list(getattr(_sch_hm, "attributes", []) or []) if _sch_hm else []
                _pending_hm = [a for a in _attrs_hm if not getattr(a, "achieved", False)]
                if _pending_hm:
                    _hm_lines = []
                    for _a in _pending_hm:
                        _an = getattr(_a, "name", "") or ""
                        _terms = self._extract_quoted_terms(_an)
                        _hint = ""
                        if _terms:
                            _hint = (
                                " → 본문에 다음 키워드를 **그대로** 포함시켜라: "
                                + ", ".join(f"\"{t}\"" for t in _terms)
                            )
                        else:
                            import re as _re_hm
                            _mc = _re_hm.search(r"(\d+)\s*(?:개|건|가지)", _an)
                            if _mc:
                                _hint = (
                                    f" → 해당 항목을 최소 {_mc.group(1)}개 이상 "
                                    f"불릿(- ) 또는 번호(1.) 리스트로 명시하라."
                                )
                        _hm_lines.append(f"- {_an}{_hint}")
                    _howmuch_block = (
                        "[필수 충족 조건 — 아래 항목을 모두 본문 안에서 명시적으로 만족시켜라]\n"
                        + "\n".join(_hm_lines)
                        + "\n\n"
                    )
            except Exception as _e_hm:
                print(f"  ⚠️ [WRITE] HOW_MUCH 키워드 주입 실패 (무시): {_e_hm}")

            # [P1 Fix #4 2026-04-24] 산출물 유형별 필수 섹션 체크리스트.
            # Why: 분석가 프롬프트가 "최소 1500자" 만 요구하고 구조 요구사항이 없어,
            # LLM 이 page_text 원천을 그대로 복붙해 plan.md 라 이름붙이는 경향
            # (Verifier: "원본 페이지 내용을 그대로 복사" missing_attr FAIL 루프).
            # 파일명 기반으로 문서 유형을 감지해 필수 섹션을 프롬프트 머리에 명시.
            _document_type_block = ""
            try:
                _target_lower = (var_name or "").lower()
                if "plan" in _target_lower and (
                    _target_lower.endswith(".md") or _target_lower.endswith(".txt")
                    or "." not in _target_lower
                ):
                    _document_type_block = (
                        _self_asset_blk
                        + "[산출물 유형: 기획안 / 사업계획서]\n"
                        "[필수 섹션 — 모두 포함하지 않으면 Verifier FAIL 확정]\n"
                        "1) **상품/서비스명 후보 3개** — 각각 한 줄 설명 포함\n"
                        "2) **타겟 고객** — 연령/직업/관심사 등 구체 페르소나 서술\n"
                        "3) **USP (고유 가치 제안)** — 경쟁 서비스와 어떻게 다른가\n"
                        "4) **가격 전략** — 최소 2 티어 이상, 근거 포함\n"
                        "5) **마케팅 채널/전술** — 최소 3 채널, 각 채널별 구체 액션\n"
                        "6) **경쟁 분석 요약** — 원천 데이터에서 뽑은 상위 경쟁자 3~5 곳의 "
                        "가격·장단점을 표 또는 목록 형태로\n"
                        "[금지] 원천 페이지 본문을 그대로 복사해 기획안 행세하는 것. "
                        "원천은 '경쟁 분석' 섹션의 근거로만 사용하고, 1~5 섹션은 "
                        "너의 분석·종합·판단으로 작성하라.\n\n"
                    )
                elif (
                    "research" in _target_lower
                    or "report" in _target_lower
                    or "리서치" in _target_lower
                ):
                    _document_type_block = (
                        "[산출물 유형: 리서치 리포트]\n"
                        "[필수 섹션]\n"
                        "1) **조사 대상 개요** — 시장/도메인/경쟁자 범위 명시\n"
                        "2) **주요 발견 사실** — 숫자·날짜·인용 포함한 항목 최소 5 개\n"
                        "3) **경쟁자 비교 목록** — 이름·가격·특징·리뷰수 포함\n"
                        "4) **패턴/인사이트** — 발견 사실로부터 도출한 관찰\n"
                        "5) **시사점 / 다음 행동 제안** — 분석 결론을 자연어로\n"
                        "[금지] 페이지 원문 통째 복사. 요약·종합·대조가 리서치의 본질이다.\n\n"
                    )
                elif (
                    "dev" in _target_lower
                    or "spec" in _target_lower
                    or "architecture" in _target_lower
                    or "사양" in _target_lower
                ):
                    _document_type_block = (
                        "[산출물 유형: 개발 사양 / 아키텍처 문서]\n"
                        "[필수 섹션]\n"
                        "1) **기능 요구사항** — 사용자 스토리 또는 유즈케이스\n"
                        "2) **비기능 요구사항** — 성능·보안·가용성\n"
                        "3) **주요 컴포넌트/모듈 구조** — 책임과 인터페이스\n"
                        "4) **데이터 모델** — 핵심 엔티티·관계\n"
                        "5) **외부 의존성** — API·라이브러리·인프라\n\n"
                    )
            except Exception as _e_dt:
                print(f"  ⚠️ [WRITE] 문서 유형 블록 생성 실패 (무시): {_e_dt}")

            # [P1 Fix #5 2026-04-24] 직전 Verifier FAIL 기록 주입.
            # Why: 재시도 WRITE 에서 왜 전에 실패했는지 프롬프트에 안 들어가면 LLM 이
            # 같은 형태로 또 작성함. _last_verifier_fail 은 Verifier FAIL 판정 직후
            # schema 에 기록됨 (dispatcher Phase 3 분기).
            _verifier_fail_block = ""
            try:
                _sch_vf = self._active_schema
                _last_vf = getattr(_sch_vf, "_last_verifier_fail", None) if _sch_vf else None
                if _last_vf and isinstance(_last_vf, dict):
                    _vf_type = _last_vf.get("fail_type", "?") or "?"
                    _vf_reason = (_last_vf.get("reason") or "")[:300]
                    _vf_attrs = list(_last_vf.get("failed_attrs") or [])
                    if _vf_reason or _vf_attrs:
                        _vf_lines = [
                            "[⚠️ 직전 Verifier FAIL 기록 — 이번엔 반드시 개선하라]",
                            f"- 실패 타입: {_vf_type}",
                        ]
                        if _vf_reason:
                            _vf_lines.append(f"- 실패 사유: {_vf_reason}")
                        if _vf_attrs:
                            _vf_lines.append(
                                f"- 미달성 속성: {', '.join(str(a) for a in _vf_attrs[:8])}"
                            )
                        _vf_lines.append(
                            "- 동일 실수(원본 복사 / 필수 섹션 누락 / 형식 불일치)를 "
                            "반복하면 즉시 FAIL 재발. 직전 실패를 **교정한** 새 본문을 작성하라.\n"
                        )
                        _verifier_fail_block = "\n".join(_vf_lines) + "\n"
            except Exception as _e_vf:
                print(f"  ⚠️ [WRITE] Verifier FAIL 블록 생성 실패 (무시): {_e_vf}")

            # 두 블록은 _available_vars_block 앞에 prepend — LLM 이 변수를 읽기 전에
            # 먼저 문서 구조와 직전 실패를 인지하도록.
            _available_vars_block = (
                _document_type_block + _verifier_fail_block + _available_vars_block
            )

            if _prior_content and len(_prior_content) > 80:
                _prior_snippet = _prior_content[:3000]
                _write_prompt = (
                    _available_vars_block
                    + _howmuch_block
                    + f"[기존 문서 — 이 내용을 **유지·확장**하라. 빈 템플릿이나 짧은 헤더로 덮어쓰지 마라.]\n"
                    f"```\n{_prior_snippet}\n```\n\n"
                    f"[새 지시]\n{content}\n\n"
                    f"[출력 규칙]\n"
                    f"- 위 기존 문서의 본문을 유지하면서 누락된 부분만 보강·추가하라.\n"
                    f"- 원천 데이터 변수가 제공됐으면 그 내용을 인용·분석해 본문에 반영하라.\n"
                    f"- 기존 본문을 임의 삭제/축약/'내용 없음'으로 바꾸는 것 금지.\n"
                    f"- 위 [필수 충족 조건] 의 키워드·항목 수를 모두 만족시켜라.\n"
                    f"- 반드시 기존 길이({len(_prior_content)}자) 이상의 **완성된 최종 문서 전체**를 한 번에 출력하라."
                )
            elif _available_vars_block or _howmuch_block:
                if _concise_mode:
                    # [Fix 2026-04-22] 간결 모드 출력 규칙: "모든 근거를 인용"은 복붙 유도 → 제거.
                    _write_prompt = (
                        _available_vars_block
                        + _howmuch_block
                        + f"[새 지시]\n{content}\n\n"
                        f"[출력 규칙]\n"
                        f"- 사용자 요청에 직접 답하는 **요약**만 출력하라.\n"
                        f"- 원천 데이터 원문을 그대로 인용/복붙하지 마라.\n"
                        f"- 핵심 숫자·날짜·사실 위주로 자연어 문장으로 정리.\n"
                        f"- 길이는 필요한 만큼만. 억지로 늘리거나 서론을 붙이지 마라."
                    )
                else:
                    _write_prompt = (
                        _available_vars_block
                        + _howmuch_block
                        + f"[새 지시]\n{content}\n\n"
                        f"[출력 규칙]\n"
                        f"- 위 원천 데이터를 근거로 본문을 작성하라.\n"
                        f"- 빈 템플릿/짧은 헤더/추측 금지. 데이터에서 추출 가능한 모든 근거를 인용하라.\n"
                        f"- 위 [필수 충족 조건] 의 키워드·항목 수를 모두 만족시켜라."
                    )

            # [Fix 2026-04-20 v2] WRITE 분석가 호출은 프롬프트가 커서 기본 20s
            # timeout 으로 자주 실패 → 90s 로 상향. 아래 에러 프리픽스 가드와 함께 동작.
            # [Fix 2026-04-21 P0-Truncation] 사용자 설정 max_tokens(기본 3840)가
            # 긴 산출물(기획안/보고서) 에 턱없이 작아 plan.md(9343바이트) 가 문장
            # 중간("중견기업의 특정 업무 프로세스에 최적")에서 잘리던 근본 원인.
            # WRITE 전용으로 16384 로 상향(Gemini 2.5 Flash 출력 한도 내). 설정값이
            # 이보다 크면 그대로 사용.
            _write_max_tokens = 16384

            # [DC-WRITE 2026-05-03] 문서 생성 WRITE 일 때 DC Reasoner 강제 발동.
            # ─────────────────────────────────────────────────────────────────
            # 사용자 요청: 마일스톤 작업 중 문서 생성 시도 시 DC Reasoner 항상 발동.
            # 분할정복 분석 결과를 _write_prompt 에 prepend → 분석가 LLM 이 더 깊은
            # 구조로 문서 작성. 문서 확장자 (.md/.txt/.html/.pdf/.docx/.rtf) 만 발동.
            # 회귀 가드: (a) _concise_mode 면 skip (요약 본문 — DC 오버킬)
            #            (b) content 50자 미만 면 skip (trivial WRITE)
            #            (c) DC 자체 실패 시 graceful fallback (기존 prompt 그대로)
            #            (d) target 확장자 코드/데이터 (.py/.json/.csv) 는 skip.
            _DOC_EXTENSIONS = (".md", ".txt", ".html", ".htm", ".pdf",
                                ".docx", ".doc", ".rtf", ".rst", ".tex")
            _target_lower = (target or "").strip().lower()
            _is_doc_target = any(_target_lower.endswith(ext) for ext in _DOC_EXTENSIONS)
            _content_meaningful = bool(content and len(content.strip()) >= 50)

            if _is_doc_target and _content_meaningful and not _concise_mode:
                try:
                    from eidos_dc_reasoner import DCReasoner as _DCR
                    _dcr = _DCR()
                    # 분석 대상 텍스트 — 사용자 지시 (content) + 가용 변수 요약 + howmuch
                    _dc_text = (
                        f"문서 작성 task: {content[:2000]}\n\n"
                        f"파일 경로: {target[:120]}\n"
                    )
                    if _howmuch_block:
                        _dc_text += f"\n[필수 충족 조건]\n{_howmuch_block[:600]}\n"
                    print(f"  🧠 [DC-WRITE] 문서 생성 DC Reasoner 발동 — target={target[:40]}")

                    def _dc_progress(_msg: str):
                        # 자율이동 status_msg 와 충돌 방지 — print 만
                        print(f"  [DC-WRITE Standalone] {_msg}")

                    _dc_result = await _dcr.reason_standalone(
                        text=_dc_text,
                        chat_history=[],
                        progress_callback=_dc_progress,
                        force_deep=True,  # 사용자 요청 — SIMPLE 게이트 우회 강제 발동
                    )
                    if _dc_result and isinstance(_dc_result, dict):
                        _dc_synthesis = str(_dc_result.get("result", "") or "")[:8000]
                        if _dc_synthesis:
                            _dc_block = (
                                "\n[🧠 DC 분할정복 사전 분석 — 분석가 LLM 은 이 구조를 본문에 반영하라]\n"
                                f"{_dc_synthesis}\n\n"
                                "[DC 분석 활용 규칙]\n"
                                "- 위 DC 분석의 핵심 결론·하위 질문·근거를 본문 섹션으로 흡수.\n"
                                "- DC 분석을 그대로 복붙하지 말고 본문 흐름에 맞게 재구성.\n"
                                "- DC 분석에서 누락된 부분이 있으면 본문에서 보강.\n\n"
                            )
                            _write_prompt = _dc_block + _write_prompt
                            print(f"  ✅ [DC-WRITE] DC 분석 ({len(_dc_synthesis)}자) 주입 완료")
                        else:
                            print(f"  [DC-WRITE] DC 결과 비어있음 — 기존 prompt 진행")
                    else:
                        # SIMPLE 판정 또는 None 반환 — 기존 prompt 그대로 (회귀 가드)
                        print(f"  [DC-WRITE] DC SIMPLE/None — 기존 prompt 진행")
                except Exception as _dc_e:
                    # DC 자체 실패 — graceful fallback
                    print(f"  ⚠️ [DC-WRITE] DC Reasoner 실패 (graceful): {type(_dc_e).__name__}: {str(_dc_e)[:100]}")

            # [nn_trainer Phase I-1 2026-05-21] plan/research 산출물을 학습된
            # AgentNet 으로 라우팅. ENABLE_NN_TRAINER 환경변수가 truthy 일 때만
            # 활성화. 기본 OFF -- legacy single-shot path 회귀 보장.
            # 학습된 net 은 매 호출마다 1-step backward 로 점진 학습 (
            # eidos_files/nn_trainer/<synth_kind>_net.json 영속).
            # 실패 시 silently single-shot 으로 fallback (가용성 우선).
            generated = ""
            _used_nn_trainer = False
            _nn_flag = os.environ.get("ENABLE_NN_TRAINER", "").strip().lower()
            if _nn_flag in ("1", "true", "yes", "on") and _synth_kind in ("plan", "research"):
                try:
                    from eidos_nn_trainer_bridge import nn_trainer_forward_async
                    from eidos_synthesis_eval import SynthesisEvaluator
                    _nn_sources = [
                        v for v in (_plan_vars or {}).values()
                        if isinstance(v, str) and len(v) >= 200
                    ]
                    _nn_evaluator = SynthesisEvaluator(
                        synth_kind=_synth_kind, sources=_nn_sources,
                    )
                    _nn_result = await nn_trainer_forward_async(
                        _write_prompt, _synth_kind,
                        evaluator=_nn_evaluator,
                        enable_learning=True,
                    )
                    _nn_out = (_nn_result.get("output") or "").strip()
                    if _nn_out:
                        generated = _nn_out
                        _used_nn_trainer = True
                        print(
                            f"  🧠 [nn_trainer Phase I-1] '{_synth_kind}' "
                            f"routed through AgentNet. "
                            f"forward_calls={_nn_result['calls']} "
                            f"tools_active={_nn_result['tools_active']} "
                            f"learning_applied={len(_nn_result['learning_applied'])} "
                            f"duration={_nn_result['duration']:.1f}s"
                        )
                    else:
                        print(
                            f"  ⚠️ [nn_trainer Phase I-1] empty output -- "
                            f"falling back to single-shot"
                        )
                except Exception as _e_nn:
                    print(
                        f"  ⚠️ [nn_trainer Phase I-1] fallback to single-shot: "
                        f"{type(_e_nn).__name__}: {str(_e_nn)[:200]}"
                    )

            if not _used_nn_trainer:
                generated = await get_llm_response_async(
                    _write_prompt,
                    system_prompt=EIDOS_ANALYST_SYSTEM_PROMPT,
                    timeout=120,
                    max_tokens=_write_max_tokens,
                )
            generated = generated.strip()

            # [Fix 2026-04-21 P1-Truncation] 응답 말미가 종결 부호 없이 끝나면
            # max_tokens 한도 절단으로 간주하고 한 번 이어쓰기 호출해 문서 완결.
            generated = await self._maybe_continue_truncated_write(
                generated,
                _write_prompt,
                _available_vars_block,
                content,
                max_tokens=_write_max_tokens,
            )

            # [Fix 2026-04-24 WRITE 폭발 가드] 단일 WRITE 결과 길이 상한 50000자.
            # Why: 같은 8개 변수를 반복 풀어쓰면서 148614자(148KB)까지 폭발한 케이스.
            # Verifier FAIL → retry → LLM 이 같은 데이터 반복 → 토큰 비용 폭증
            # (In:18992, Out:6315 등). 50000자(약 12500 토큰 분량) 초과는 거의
            # 데이터 반복 폭발이라 중간에서 절단 + 경고. 정상 산출물(plan.md ~9000자,
            # research_report.md ~10000자) 은 영향 없음.
            _WRITE_HARD_CAP = 50000
            if len(generated) > _WRITE_HARD_CAP:
                _orig_len = len(generated)
                generated = generated[:_WRITE_HARD_CAP].rstrip() + (
                    "\n\n[…WRITE 폭발 가드: 길이 상한 초과로 절단됨…]\n"
                )
                print(
                    f"  ✂️ [WRITE 폭발 가드] {_orig_len}자 → {len(generated)}자 "
                    f"절단 (상한 {_WRITE_HARD_CAP}). 같은 데이터 반복 풀어쓰기 의심."
                )
                self._report(
                    f"✂️ **[WRITE 폭발 가드]** {_orig_len:,}자 → {_WRITE_HARD_CAP:,}자 절단. "
                    f"LLM 이 같은 원천을 반복 풀어쓴 것으로 보임. 다음 step 은 신규 정보 "
                    f"확보 후 WRITE 하거나 DONE 으로 마무리하라."
                )

            # [Synthesis Guard 2026-05-17] 실질 검증 — 구 Jaccard 가드(단일
            #   최대변수·token≥0.80) 대체. 결함 본질은 "복사 토큰 존재" 가
            #   아니라 "신규 종합 부재" 이므로 전체 원천 *합집합* 대비 본문
            #   신규 문장 비율을 잰다(memory: diagnosis_eidos_paraphrase_machine).
            #   - verbatim_copy → 거부(명백한 통째 복붙).
            #   - derivative(신규<35%) → 소프트: 종합 2단 재시도 지시 반환.
            #   - plan 문서 자기자산 0언급 → 소프트: 자기자산 결합 재시도 지시.
            #   반환 문자열은 기존 retry/_verifier_fail 배관으로 흘러 다음
            #   WRITE 프롬프트에 직전 실패 사유로 주입된다(Phase C).
            try:
                if _SG is not None and len(generated) >= 500:
                    _srcs = []
                    try:
                        for _sn, _sv in (_plan_vars or {}).items():
                            if (not isinstance(_sn, str)) or _sn.startswith("__"):
                                continue
                            if _sn == var_name:
                                continue
                            if isinstance(_sv, str) and len(_sv) >= 200:
                                _srcs.append(_sv)
                    except Exception:
                        _srcs = []
                    # [2026-05-23] synth_kind 별 차등 임계 — research 는 인용 헤비
                    # (시장조사·LIST 페이지 정보 추출) 라 더 관대하게.
                    #   research: verbatim_run=45 (한국어 ~90자) · novel_min=0.25
                    #   plan/strategy: verbatim_run=35 (한국어 ~70자) · novel_min=0.30
                    #   기본 (synth_kind 미정): default 35/0.30 사용
                    if _synth_kind == "research":
                        _dr = _SG.derivation_report(
                            generated, _srcs,
                            novel_min=0.25, verbatim_run=45,
                        )
                    elif _synth_kind == "plan":
                        _dr = _SG.derivation_report(
                            generated, _srcs,
                            novel_min=0.30, verbatim_run=35,
                        )
                    else:
                        _dr = _SG.derivation_report(generated, _srcs)
                    _verdict = _dr.get("verdict")
                    if _verdict in ("verbatim_copy", "derivative"):
                        print(
                            f"  🚫 [Synthesis Guard] 실질검증 FAIL "
                            f"({_verdict}): {_dr.get('reason')}"
                        )
                        self._report(
                            f"🚫 **[실질 검증]** {_dr.get('reason')} — 원천을 "
                            f"풀어쓰기/복붙한 파생물로 판단, 종합 재작성 필요."
                        )
                        # [2026-05-23] 게이트 차단 누적 카운터 — 4회 도달 시 PAUSE.
                        try:
                            _sch_sgf = self._active_schema
                            if _sch_sgf is not None:
                                _sch_sgf._last_verifier_fail = {
                                    "fail_type": "no_synthesis",
                                    "reason": _dr.get("reason", "")[:300],
                                    "failed_attrs": [],
                                }
                                if self._gate_block_inc_and_maybe_pause(
                                    task, _verdict, _dr.get("reason", ""),
                                ):
                                    return (
                                        "🚫 게이트 차단 누적 — 사용자 결정 대기 "
                                        "(PAUSE)"
                                    )
                        except Exception:
                            pass
                        return (
                            f"🚫 WRITE 거부(종합 부재: {_verdict}). {_dr.get('reason')}\n"
                            f"다음 WRITE 는 반드시 2단으로 작성하라:\n"
                            f"(1) 원천에서 핵심 사실·수치·고유명사만 *명제 목록* 으로 "
                            f"추출(짧게).\n"
                            f"(2) 그 명제 위에 **너의 종합·판단·대조·시사점** 을 "
                            f"새 문장으로 작성하라. 원천 문장을 어형만 바꿔 옮기는 "
                            f"풀어쓰기 금지 — 원천에 없던 관계·결론·전략이 본문의 "
                            f"과반이어야 한다. 원천 인용은 근거 제시용으로만 짧게."
                        )
                    # plan 문서 자기자산 게이트 (소프트).
                    if (_synth_kind == "plan"
                            and not _SG.self_asset_presence(generated)):
                        print(
                            "  🚫 [Synthesis Guard] plan 문서 자기자산 0언급 "
                            "— 최근접 페르소나 빈칸채움 의심"
                        )
                        self._report(
                            "🚫 **[자기자산 게이트]** 기획 문서가 EIDOS 고유 "
                            "자산을 한 번도 언급하지 않음 — 타인 페르소나·일반론 "
                            "차용 의심, 자기자산 결합 재작성 필요."
                        )
                        # [2026-05-23] 게이트 차단 누적 카운터 — 4회 도달 시 PAUSE.
                        try:
                            _sch_sga = self._active_schema
                            if _sch_sga is not None:
                                _sch_sga._last_verifier_fail = {
                                    "fail_type": "no_self_asset",
                                    "reason": "기획안이 EIDOS 자기자산 미언급",
                                    "failed_attrs": [],
                                }
                                if self._gate_block_inc_and_maybe_pause(
                                    task, "no_self_asset",
                                    "기획안이 EIDOS 자기자산 미언급",
                                ):
                                    return (
                                        "🚫 게이트 차단 누적 — 사용자 결정 대기 "
                                        "(PAUSE)"
                                    )
                        except Exception:
                            pass
                        return (
                            "🚫 WRITE 거부(자기자산 미언급): 기획/전략 문서인데 "
                            "EIDOS 고유 자산(자기조직화 substrate·절차 뼈대·"
                            "합성적 추론·ToM·자율 안전망 등)을 단 한 번도 "
                            "언급하지 않았다. 프롬프트 머리의 [차별 자산] 카드를 "
                            "근거로, USP·타겟·차별화 섹션을 그 자산과 직접 "
                            "연결해 다시 작성하라. '범용 AI + 매크로' 또는 원천에 "
                            "있던 타인 페르소나로 자기를 표상하는 것은 오류다."
                        )
            except Exception as _e_sgv:
                print(f"  ⚠️ [Synthesis Guard] 실질검증 실패 (무시): {_e_sgv}")

            # [Fix 2026-04-20 v2] analyst LLM 에러 문자열 감지.
            # Why: get_llm_response_async 는 실패 시 "[서버 응답 시간 초과: 잠시 후
            # 다시 시도하세요]" 같은 에러 문자열(정확히 28자)을 정상 content 처럼 리턴.
            # 이게 Shortening Guard 에서 "짧은 신규 응답"으로 오분류되어 같은 WRITE
            # 가 무한 반복됐음. prefix 감지 시 WRITE 자체를 명시 에러로 종료해
            # _react_loop_with_retry 의 fail_streak 로 정상 분류되게 한다.
            _ERR_PREFIXES = ("[서버 ", "[LLM ", "[크레딧", "[Pro ")
            if generated.startswith(_ERR_PREFIXES):
                print(f"  ⚠️ [WRITE] analyst LLM 오류 감지 ({len(generated)}자): {generated[:80]}")
                return (
                    f"⚠️ WRITE 실패(분석가 LLM 오류): {generated[:120]}. "
                    f"프롬프트가 너무 커서 타임아웃 가능성 높음. "
                    f"다음 step 은 READ_PAGE 로 필요한 원천만 추리거나 "
                    f"WRITE content 를 짧게 나눠 재시도하라."
                )

            # [Fix 2026-04-19] Shortening Guard — 신규가 기존보다 현저히 짧으면 덮어쓰기 거부.
            # Why: HOW_MUCH 미달성 속성 때문에 LLM이 WRITE를 반복 발주하는데 재생성이
            #      짧은 템플릿으로 나오면 실제 본문이 소실됨. 기존 내용이 충분하면 유지하고
            #      다음 step에서 다른 액션(DONE/READ_PAGE)을 선택하도록 경고 반환.
            # [Fix 2026-04-21 Phase 3-B] 단, 기존 파일이 원 주제에서 드리프트한 경우
            # (예: 'AI 프롬프트 팩' 요청인데 기존 plan.md 가 '업무 자동화' 로 오염)
            # 신규 짧은 content 가 주제 키워드를 더 많이 포함하면 SG 를 우회하고
            # 주제 회복 덮어쓰기를 허용한다. 기존 파일은 {name}.drift.bak 으로 백업.
            _MIN_PRIOR_GUARD = 200
            _SHORTEN_RATIO = 0.5
            if (_prior_content
                    and len(_prior_content) >= _MIN_PRIOR_GUARD
                    and len(generated) < int(len(_prior_content) * _SHORTEN_RATIO)):
                # 주제 회복 체크
                _sg_bypass = False
                _topic_kw = ""
                # [P2 2026-04-21] Chain stage 전환 직후 첫 WRITE 는 Guard 면제(1회).
                # 증상: chain stage 가 바뀌면 이전 stage 파일은 새 관점으로 재생성될
                # 수 있는데, Guard 는 파일 글자수만 비교해 이전 세션/stage 잔재를
                # 무조건 보호 → 정당한 재생성도 반복 차단. Chain Reset(P0) 가 있더라도
                # 타이밍 실패·권한 이슈로 백업 실패 시 방어층. 이 면제는 stage 당
                # 1회만 발동하므로 실제 SG 루프(같은 stage 내 반복 WRITE) 는 막지 않음.
                try:
                    _sch_p2 = self._active_schema
                    if _sch_p2 is not None:
                        _cid_p2 = getattr(_sch_p2, "chain_id", "") or ""
                        _sidx_p2 = int(getattr(_sch_p2, "stage_index", 0) or 0)
                        _p2_key = (_cid_p2, _sidx_p2, (var_name or "").strip().lower())
                        if _cid_p2 and _p2_key not in self._chain_stage_first_write_seen:
                            self._chain_stage_first_write_seen.add(_p2_key)
                            _sg_bypass = True
                            print(
                                f"  🔓 [WRITE SG bypass] chain stage 첫 WRITE 면제: "
                                f"stage={getattr(_sch_p2, 'stage', '')} idx={_sidx_p2} "
                                f"var='{var_name}' (기존 {len(_prior_content)}자 drift 가능성)"
                            )
                            # drift 백업(기존 로직과 동일하게 안전 보존).
                            try:
                                _read_path_bak2 = var_name if _looks_like_file else (_auto_file_path or "")
                                if _read_path_bak2:
                                    from execution_module import SAFE_BASE_PATH as _SBASE_BAK2
                                    import os as _os_bak2
                                    _rel_bak2 = _read_path_bak2.replace("\\", "/")
                                    while _rel_bak2.startswith("./"):
                                        _rel_bak2 = _rel_bak2[2:]
                                    _abs_bak2 = _os_bak2.path.abspath(
                                        _os_bak2.path.join(_SBASE_BAK2, _rel_bak2)
                                    )
                                    if _os_bak2.path.isfile(_abs_bak2):
                                        _bak_path2 = f"{_abs_bak2}.stage-first.bak.{int(time.time())}"
                                        try:
                                            with open(_abs_bak2, "r", encoding="utf-8") as _fh_old2:
                                                _old_c2 = _fh_old2.read()
                                            with open(_bak_path2, "w", encoding="utf-8") as _fh_bak2:
                                                _fh_bak2.write(_old_c2)
                                            print(f"  💾 [WRITE SG bypass] stage-first 백업: {_bak_path2}")
                                        except Exception as _e_bk2:
                                            print(f"  ⚠️ [WRITE SG bypass] 백업 실패 (덮어쓰기 진행): {_e_bk2}")
                            except Exception as _e_bak_outer2:
                                print(f"  ⚠️ [WRITE SG bypass] 백업 경로 준비 실패 (무시): {_e_bak_outer2}")
                except Exception as _e_p2:
                    print(f"  ⚠️ [WRITE P2] stage-first 체크 실패 (무시): {_e_p2}")
                try:
                    _sch_sg = getattr(self, "_active_schema", None)
                    _cid_sg = getattr(_sch_sg, "chain_id", "") if _sch_sg else ""
                    if _cid_sg:
                        from eidos_mission_chain import (
                            get_chain_registry as _gcr_sg,
                            _extract_chain_topic_keyword as _xtk_sg,
                        )
                        _ch_sg = _gcr_sg().get(_cid_sg)
                        if _ch_sg is not None:
                            _topic_kw = _xtk_sg(_ch_sg)
                except Exception as _e_kw_sg:
                    _topic_kw = ""
                    print(f"  ⚠️ [WRITE SG] topic keyword 추출 실패 (무시): {_e_kw_sg}")
                if _topic_kw:
                    _kw_lw = _topic_kw.lower()
                    _hits_prior = _prior_content.lower().count(_kw_lw)
                    _hits_new = generated.lower().count(_kw_lw)
                    # 기존이 주제 키워드를 거의 안 담고 있고, 신규가 더 많이 담으면 drift 회복
                    if _hits_new > _hits_prior and _hits_prior <= 1:
                        _sg_bypass = True
                        print(
                            f"  🔁 [WRITE SG bypass] 주제 회복 감지: "
                            f"키워드 '{_topic_kw}' 기존 {_hits_prior}회 < 신규 {_hits_new}회 "
                            f"→ SG 우회하고 drift 덮어쓰기 진행"
                        )
                        # drift 백업
                        try:
                            _read_path_bak = var_name if _looks_like_file else (_auto_file_path or "")
                            if _read_path_bak:
                                from execution_module import SAFE_BASE_PATH as _SBASE_BAK
                                import os as _os_bak
                                _rel_bak = _read_path_bak.replace("\\", "/")
                                while _rel_bak.startswith("./"):
                                    _rel_bak = _rel_bak[2:]
                                _abs_bak = _os_bak.path.abspath(
                                    _os_bak.path.join(_SBASE_BAK, _rel_bak)
                                )
                                if _os_bak.path.isfile(_abs_bak):
                                    _bak_path = f"{_abs_bak}.drift.bak.{int(time.time())}"
                                    # 복사 방식(읽어서 bak 에 쓰기). move/replace 는 원본을
                                    # 옮겨버려 뒤이은 WRITE 경로와 충돌하므로 피한다.
                                    try:
                                        with open(_abs_bak, "r", encoding="utf-8") as _fh_old:
                                            _old_content = _fh_old.read()
                                        with open(_bak_path, "w", encoding="utf-8") as _fh_bak:
                                            _fh_bak.write(_old_content)
                                        print(
                                            f"  💾 [WRITE SG bypass] drift 백업: {_bak_path}"
                                        )
                                    except Exception as _e_bk:
                                        print(
                                            f"  ⚠️ [WRITE SG bypass] 백업 실패 "
                                            f"(덮어쓰기는 진행): {_e_bk}"
                                        )
                        except Exception as _e_bak_outer:
                            print(
                                f"  ⚠️ [WRITE SG bypass] 백업 경로 준비 실패 "
                                f"(무시): {_e_bak_outer}"
                            )
                if not _sg_bypass:
                    print(
                        f"  🛡️ [WRITE] Shortening Guard: 기존 {len(_prior_content)}자 > 신규 "
                        f"{len(generated)}자 — 덮어쓰기 취소, 기존 내용 보존"
                    )
                    # 사용 가능 변수 힌트를 메시지에 붙여 다음 step planner 가
                    # content 에 "{{var}}" 참조를 넣도록 유도.
                    _var_hint = ""
                    if _available_var_summaries:
                        _var_hint = (
                            " 사용 가능 변수(이번엔 프롬프트에 주입됐음): "
                            + ", ".join(f"{{{vn}}}({ln}자)" for vn, ln in _available_var_summaries)
                            + "."
                        )
                    return (
                        f"⚠️ WRITE 취소(Shortening Guard): 이미 '{var_name}' 에 "
                        f"{len(_prior_content)}자가 있는데 새 생성은 {len(generated)}자뿐이라 "
                        f"덮어쓰기를 거부했다. 같은 파일에 또 WRITE 하지 말고 — "
                        f"(a) 모든 HOW_MUCH 근거가 기존 파일에 있으면 DONE, "
                        f"(b) 새 정보 보강이 필요하면 먼저 NAVIGATE/READ_PAGE로 원천 데이터 확보 후 WRITE, "
                        f"(c) 다음 WRITE content 에 위 변수를 '{{변수명}}' 문법으로 명시 참조."
                        + _var_hint
                    )
                # _sg_bypass=True 면 아래로 흘러 plan.variables[var_name]=generated,
                # 파일 저장 로직으로 진행 (drift 회복 덮어쓰기).

            plan.variables[var_name] = generated

            # target이 파일 경로처럼 보이면 샌드박스에 실제 파일도 저장.
            # Why: 이전엔 plan.variables 메모리에만 저장해서 세션 종료 시 사라졌음.
            # [Fix v2] _auto_file_path 가 세팅됐으면 내용-계열 변수를 {var}.md 로 승격 저장.
            _file_saved_abs = None
            _write_rel = var_name if _looks_like_file else (_auto_file_path or "")
            _auto_promoted = bool(_auto_file_path and not _looks_like_file)
            if _write_rel and len(generated) >= 200:   # 너무 짧은 출력(<200자)은 자동 승격 안 함
                try:
                    from execution_module import SAFE_BASE_PATH as _SBASE_W
                    import os as _os_w
                    _rel = _write_rel.replace("\\", "/")
                    while _rel.startswith("./"):
                        _rel = _rel[2:]
                    # 샌드박스 루트가 이미 eidos_files — 접두사 중복 제거
                    _base_name = _os_w.path.basename(_os_w.path.normpath(_SBASE_W)).lower()
                    if _base_name and _rel.lower().startswith(_base_name + "/"):
                        _rel = _rel[len(_base_name) + 1:]
                    _abs = _os_w.path.abspath(_os_w.path.join(_SBASE_W, _rel))
                    _base_abs = _os_w.path.abspath(_SBASE_W)
                    # 경로 이탈 가드
                    if _abs == _base_abs or _abs.startswith(_base_abs + _os_w.sep):
                        _parent = _os_w.path.dirname(_abs)
                        if _parent:
                            _os_w.makedirs(_parent, exist_ok=True)
                        with open(_abs, "w", encoding="utf-8") as _fh_w:
                            _fh_w.write(generated)
                        _file_saved_abs = _abs
                        _tag = "파일 자동 승격" if _auto_promoted else "파일 저장"
                        print(f"  💾 [WRITE] {_tag}: {_abs} ({len(generated)}자)")
                        # [File Card] 채팅창에 FileCreatedCard 삽입을 위해 Worker signal emit.
                        # Worker 없거나 signal 미정의여도 조용히 스킵 (로그 경로엔 영향 없음).
                        try:
                            _gui = getattr(self.core, "_gui_ref", None)
                            _worker = getattr(_gui, "eidos_worker", None) if _gui else None
                            if _worker and hasattr(_worker, "file_created_signal"):
                                _worker.file_created_signal.emit(
                                    _file_saved_abs,
                                    f"파일이 생성됐습니다 ({len(generated)}자)"
                                )
                        except Exception as _fce:
                            print(f"  [WRITE] file_created_signal emit 실패(무시): {_fce}")
                    else:
                        print(f"  ⚠️ [WRITE] 샌드박스 이탈 감지 — 파일 저장 스킵: {_write_rel}")
                except Exception as _we:
                    print(f"  ⚠️ [WRITE] 파일 저장 실패 (변수 저장은 유지): {_we}")

            if _file_saved_abs:
                _suffix = " [자동.md 승격]" if _auto_promoted else ""
                return (f"생성 완료 → 변수 '{var_name}' + 파일 저장 "
                        f"'{_file_saved_abs}'{_suffix} ({len(generated)}자)")
            return f"생성 완료 → 변수 '{var_name}' ({len(generated)}자)"

        elif action == "WAIT":
            secs = float(content) if content.replace(".", "").isdigit() else 2.0
            await asyncio.sleep(secs)
            return f"{secs}초 대기"

        elif action == "EMAIL_SEND":
            # [P-ε 2026-05-01] 외부 SDK 호출 verb — SMTP 이메일 발송.
            # target: 수신자 이메일 주소 (필수). content: 본문 (변수 치환된 텍스트).
            # eidos_files/email_config.json {smtp_host, smtp_port, smtp_user, smtp_password, from_email} 필요.
            # 미설정 시 graceful skip (archive only).
            try:
                _email_cfg_path = os.path.join("eidos_files", "email_config.json")
                if not os.path.exists(_email_cfg_path):
                    return (
                        "⚠️ EMAIL_SEND 미설정: eidos_files/email_config.json 없음. "
                        "SMTP host/port/user/password/from_email 등록 후 재시도 (사용자 개입 필요)."
                    )
                with open(_email_cfg_path, "r", encoding="utf-8") as _f:
                    _email_cfg = json.load(_f)
            except Exception as _e_cfg:
                return f"⚠️ EMAIL_SEND 설정 로드 실패: {_e_cfg}"
            _to = (target or "").strip()
            if not _to or "@" not in _to:
                return f"⚠️ EMAIL_SEND 실패: target 이 유효한 이메일 주소가 아님 ('{_to[:40]}')"
            _body = (content or "").strip()
            if not _body:
                return "⚠️ EMAIL_SEND 실패: 메시지 본문 (content) 비어있음"
            # 제목 추출 — content 첫 줄이 "Subject: ..." 면 별도 사용, 아니면 기본값
            _subject = "🚀 EIDOS"
            _body_lines = _body.split("\n", 1)
            if _body_lines[0].lower().startswith("subject:"):
                _subject = _body_lines[0].split(":", 1)[1].strip()[:200]
                _body = _body_lines[1].strip() if len(_body_lines) > 1 else ""
            try:
                import smtplib
                from email.mime.text import MIMEText
                _msg = MIMEText(_body[:50000], "plain", "utf-8")  # 50KB 한도
                _msg["Subject"] = _subject
                _msg["From"] = _email_cfg.get("from_email", _email_cfg.get("smtp_user", ""))
                _msg["To"] = _to
                _host = _email_cfg.get("smtp_host", "smtp.gmail.com")
                _port = int(_email_cfg.get("smtp_port", 587))
                _user = _email_cfg.get("smtp_user", "")
                _pwd = _email_cfg.get("smtp_password", "")
                # SMTP 발송 (별도 스레드로 — async 안에서 blocking)
                def _smtp_send():
                    with smtplib.SMTP(_host, _port, timeout=15) as _s:
                        _s.starttls()
                        _s.login(_user, _pwd)
                        _s.send_message(_msg)
                await asyncio.get_event_loop().run_in_executor(None, _smtp_send)
            except Exception as _e_send:
                return f"⚠️ EMAIL_SEND 발송 실패: {str(_e_send)[:80]}"
            return f"📧 이메일 발송 완료 ({len(_body)}자, to={_to[:40]})"

        elif action == "TELEGRAM_SEND":
            # [T2 Step 2 2026-05-01] 외부 SDK 호출 verb — 사용자 본인 텔레그램 채팅에 메시지 발송.
            # content: 변수 치환된 메시지 본문 (이미 _act 진입 전 변수 치환 완료). target: 제목 (생략 가능).
            # 4096자 텔레그램 한도 → 4000자 슬라이싱 + 잘림 표시. chat_id 는 봇 configure() 시
            # 등록된 사용자 본인 채팅만 (구조적 안전 — 다른 chat 발송 불가).
            try:
                from eidos_telegram_bot import get_bot as _get_bot_ts
                _bot_ts = _get_bot_ts()
            except Exception as _e_imp:
                return f"⚠️ TELEGRAM_SEND 실패: telegram_bot 모듈 import 오류 ({_e_imp})"
            if not _bot_ts.is_configured():
                return (
                    "⚠️ TELEGRAM_SEND 미설정: Telegram 봇 토큰 / chat_id 가 등록되지 않음. "
                    "설정 후 재시도 (사용자 개입 필요)."
                )
            _msg_body = (content or "").strip()
            if not _msg_body:
                return "⚠️ TELEGRAM_SEND 실패: 메시지 본문 (content) 비어있음"
            # 4000자 슬라이싱 (텔레그램 4096 한도, 안전 마진 96자)
            _msg_truncated = False
            if len(_msg_body) > 4000:
                _msg_body = _msg_body[:4000] + "\n\n…(본문 길어 잘림)"
                _msg_truncated = True
            _title = (target or "").strip() or "🚀 EIDOS"
            try:
                await _bot_ts.send_notification(_msg_body, title=_title)
            except Exception as _e_send:
                return f"⚠️ TELEGRAM_SEND 발송 실패: {str(_e_send)[:80]}"
            _suffix = " (본문 잘림)" if _msg_truncated else ""
            return f"📱 텔레그램 발송 완료 ({len(_msg_body)}자){_suffix}"

        elif action == "SCROLL":
            # target: "bottom" / "top" / "px:숫자" / 빈 값(기본 viewport 80%)
            # content: 자유 설명 (무시)
            _t = (target or "").strip().lower()
            if _t in ("bottom", "document.body.scrollheight", "end"):
                _scroll_js = (
                    "(function(){var b=window.scrollY;"
                    "window.scrollTo(0,document.body.scrollHeight);"
                    "return 'SCROLLED:'+b+'->'+window.scrollY;})();"
                )
            elif _t in ("top", "0"):
                _scroll_js = (
                    "(function(){var b=window.scrollY;window.scrollTo(0,0);"
                    "return 'SCROLLED:'+b+'->'+window.scrollY;})();"
                )
            elif _t.startswith("px:"):
                try:
                    _px = int(_t.split(":", 1)[1])
                except Exception:
                    _px = 600
                _scroll_js = (
                    f"(function(){{var b=window.scrollY;window.scrollBy(0,{_px});"
                    "return 'SCROLLED:'+b+'->'+window.scrollY;}})();"
                )
            else:
                _scroll_js = (
                    "(function(){var b=window.scrollY;"
                    "window.scrollBy(0,Math.round(window.innerHeight*0.8));"
                    "return 'SCROLLED:'+b+'->'+window.scrollY;})();"
                )
            _r = await _browser_run(script=_scroll_js)
            await asyncio.sleep(0.4)
            # [검증 강화 2026-04-18] 실제 scrollY delta를 파싱해 이동이 없으면 실패로
            # 보고. 이전에는 "SCROLL 실행됨" 문자열만으로 verify가 통과해 LLM이
            # 스크롤이 먹히지 않는 페이지에서도 "성공"으로 오인했다.
            _rs = str(_r or "")
            _sm = re.search(r"SCROLLED:(\-?\d+)->(\-?\d+)", _rs)
            if _sm:
                _before_y = int(_sm.group(1))
                _after_y = int(_sm.group(2))
                if _before_y == _after_y:
                    return (f"⚠️ SCROLL 실패: 스크롤 이동 없음 (y={_after_y}) — "
                            f"페이지 끝이거나 overflow 제한 가능")
                return f"SCROLL 이동 완료: {_before_y}px → {_after_y}px (delta={_after_y - _before_y})"
            return f"SCROLL 실행됨 → {_rs[:60]}"

        elif action == "GO_BACK":
            # 브라우저 뒤로가기 — 잘못 진입한 페이지에서 복귀.
            # 현재 URL을 dead-end 블랙리스트에 추가해 재진입 차단.
            _url_before = self._get_current_browser_url() or ""
            try:
                _schema_for_be = self._active_schema
                # [Fix 2026-04-19 v4] 허브/목록 URL은 dead-end 등록 면제.
                # /category/6 같은 hub가 GO_BACK 한 번으로 영구 차단되던 버그.
                # Tier 3 경로(2765~2766)는 이미 면제 중이었으나 _act("GO_BACK") 본체는
                # 무조건 등록 → Tier 3 면제를 무력화. hub 패턴 일치 시 스킵.
                _HUB_GUARD_PATTERNS = (
                    "/category", "/categories", "/search", "/tag/", "/tags/",
                    "/list", "/explore", "/browse", "/discover",
                    "?q=", "?query=", "?keyword=", "?search=",
                )
                _ub_lc = (_url_before or "").lower()
                _is_hub_url = any(p in _ub_lc for p in _HUB_GUARD_PATTERNS)
                if _url_before and _schema_for_be is not None and not _is_hub_url:
                    _be = getattr(_schema_for_be, "_dead_end_urls", None)
                    if _be is None:
                        _be = set()
                        _schema_for_be._dead_end_urls = _be
                    _be.add(_url_before.rstrip("/").lower())
                elif _is_hub_url:
                    print(f"  🛡️ [GO_BACK] 허브 URL — dead-end 등록 면제: {_url_before[:60]}")
            except Exception:
                pass
            try:
                _js_back = (
                    "(function(){"
                    "var u1=window.location.href;"
                    "if(window.history.length>1){window.history.back();return 'BACK_SENT:'+u1;}"
                    "return 'NO_HISTORY:'+u1;"
                    "})();"
                )
                _r = await _browser_run(script=_js_back)
                await asyncio.sleep(1.5)  # 페이지 전환 대기
                _url_after = self._get_current_browser_url() or ""
                _ret = str(_r or "")
                if "NO_HISTORY" in _ret:
                    return f"⚠️ GO_BACK 실패: 이전 페이지 없음 (history.length≤1)"
                if _url_before and _url_after and _url_before.rstrip("/") == _url_after.rstrip("/"):
                    return f"⚠️ GO_BACK 실행했으나 URL 미변경: {_url_after[:80]}"
                return f"뒤로가기 완료: {_url_before[:50]} → {_url_after[:50]}"
            except Exception as _be:
                return f"⚠️ GO_BACK 오류: {str(_be)[:80]}"

        elif action == "REPORT":
            self._report(f"📢 **[보고]** {content}")
            return "보고 완료"

        elif action == "DONE":
            # ── [최종 검증] DONE 전에 plan 전체의 실패 여부 확인 ─────
            failed_steps = [s for s in plan.steps if s.status == "FAILED"]
            unverified_fills = [
                s for s in plan.steps
                if s.action == "FILL" and s.status == "DONE"
                and s.result and ("검증 불가" in s.result or "⚠️" in s.result)
            ]
            if failed_steps:
                return (f"⚠️ 작업 미완료: {len(failed_steps)}개 step이 실패함. "
                        f"실패 내역: {', '.join(f'Step{s.step_no}({s.action})' for s in failed_steps)}")
            if unverified_fills:
                return (f"⚠️ 작업 검증 불가: FILL 주입이 확인되지 않은 step이 있음. "
                        f"실제로 내용이 저장되었는지 확인 필요.")
            return "작업 완료 (모든 step 검증 통과)"

        else:
            return f"알 수 없는 액션: {action}"

    # ── 유틸 ──────────────────────────────────────────────────────────────────

    # ── Reactive Mode: 화면 관찰 후 다음 행동 결정 ───────────────────────────

    async def _reactive_decide_next(
        self,
        task_prompt: str,
        page_snapshot: str,
        next_step: ActionStep,
        plan: ActionPlan,
    ) -> Optional[Dict[str, str]]:
        """
        현재 브라우저 화면 내용을 LLM에게 보여주고,
        다음 브라우저 액션의 구체적 target(CSS 셀렉터)을 결정하게 함.

        반환: {"action": "CLICK", "target": "실제 CSS 셀렉터", "content": "..."} 또는 None
        """
        # 페이지에서 클릭 가능한 요소 목록 추출
        # [Fix] browser_run_script는 JS 반환값을 못 받으므로 HTML 파싱으로 대체
        clickable_list = ""
        try:
            from execution_module import get_browser_content as _gc_click
            raw_html = await _gc_click()
            # 간단한 정규식 파싱 — BeautifulSoup 없이 a/button 태그 텍스트 추출
            # ── HTML에서 Next.js __NEXT_DATA__ 및 gig 링크 직접 파싱 (스레드 안전) ──
            _js_links = []
            try:
                import json as _json

                # 방법 1: __NEXT_DATA__ JSON에서 gig 정보 추출
                _nd_m = _re.search(
                    r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>',
                    raw_html[:200000], _re.DOTALL
                )
                if _nd_m:
                    try:
                        _nd = _json.loads(_nd_m.group(1))
                        # gigs 데이터 재귀 탐색
                        def _extract_gigs(obj, results, depth=0):
                            if depth > 10 or len(results) >= 30:
                                return
                            if isinstance(obj, dict):
                                gig_id = (obj.get('gig_id') or obj.get('gigId') or
                                          obj.get('gigID') or obj.get('gig_pk') or
                                          obj.get('id') or obj.get('pk'))
                                title = (obj.get('title') or obj.get('name') or
                                         obj.get('service_title') or obj.get('gig_title') or '')
                                price = (obj.get('price') or obj.get('min_price') or
                                         obj.get('base_price') or '')
                                if gig_id and str(gig_id).isdigit() and int(str(gig_id)) > 100:
                                    price_str = f" {price}원" if price else ''
                                    results.append(
                                        f"a | {str(title)[:40]}{price_str} | https://kmong.com/gig/{gig_id}"
                                    )
                                for v in obj.values():
                                    _extract_gigs(v, results, depth+1)
                            elif isinstance(obj, list):
                                for item in obj:
                                    _extract_gigs(item, results, depth+1)
                        _extract_gigs(_nd, _js_links)
                    except Exception:
                        pass

                # 방법 2: 다양한 패턴으로 gig ID 추출
                if len(_js_links) < 5:
                    _gig_ids_seen = set()
                    _pat_list = [
                        r'/gig/(\d{4,7})',
                        r'gigId["\s]*:["\s]*(\d{4,7})',
                        r'gig_id["\s]*:["\s]*(\d{4,7})',
                        r'"url"\s*:\s*"[^"]*gig[^"]*?(\d{5,7})"',
                    ]
                    for _pat in _pat_list:
                        for _gm in _re.finditer(_pat, raw_html):
                            _gid = _gm.group(1)
                            if _gid not in _gig_ids_seen and len(_gid) >= 4:
                                _gig_ids_seen.add(_gid)
                                _js_links.append(
                                    f"a | [상품{_gid}] | https://kmong.com/gig/{_gid}"
                                )
                            if len(_js_links) >= 30:
                                break
                        if len(_js_links) >= 30:
                            break
                    print(f"  [Observe] gig ID 추출: {len(_js_links)}개 / 방법2 패턴")

            except Exception as _jse:
                pass  # 파싱 실패 시 HTML SPA 파싱으로 폴백

            # ── 스크린샷 캡처 (Vision 기반 좌표 클릭을 위해) ─────────────────
            _screenshot_path = ""
            try:
                from execution_module import grab_browser_screenshot as _gbs
                _screenshot_path = await _gbs(timeout=4.0)
                if _screenshot_path:
                    print(f"  📸 [Observe] 스크린샷 캡처 성공: {_screenshot_path}")
                    result["screenshot"] = _screenshot_path
                # 실패해도 무시 — _think는 항상 진행
            except Exception as _sse:
                pass  # 스크린샷 실패는 무시하고 계속 진행

            import re as _re

            # ── input/select 요소 별도 수집 (검색창, 폼 필드용) ──────────────
            _input_items = []
            for _m in _re.finditer(
                r'<input[^>]*?(?:name|id|placeholder|class)="([^"]{1,60})"[^>]*>',
                raw_html[:150000], _re.IGNORECASE
            ):
                _attr = _m.group(1).strip()
                if _attr and len(_attr) > 1:
                    _type_m = _re.search(r'type="([^"]+)"', _m.group(0), _re.IGNORECASE)
                    _type = _type_m.group(1) if _type_m else "text"
                    if _type.lower() not in ("hidden", "submit", "checkbox", "radio"):
                        _input_items.append(f"input | {_attr[:50]} | type={_type}")
                if len(_input_items) >= 10:
                    break

            # ── SPA 전용: href만 있고 텍스트가 자식 태그에 있는 링크 수집 ────────
            # 크몽/네이버 등 React 앱: <a href="/gig/숫자">...<span>상품명</span>...</a>
            _spa_items = []
            for _am in _re.finditer(r'<a\s[^>]*href="(/gig/\d+|/expert/[^"]+)"[^>]*>(.*?)</a>',
                                     raw_html[:150000], _re.IGNORECASE | _re.DOTALL):
                _href = _am.group(1)
                # 자식 태그에서 텍스트 추출
                _inner = _re.sub(r'<[^>]+>', '', _am.group(2)).strip()
                _inner = ' '.join(_inner.split())[:50]
                if _inner and len(_inner) > 1:
                    _spa_items.append(f"a | {_inner} | {_href}")
                elif _href:
                    _spa_items.append(f"a | [상품] | {_href}")
                if len(_spa_items) >= 30:
                    break

            tags = _re.findall(
                r'<(a|button)[^>]*?(?:href="([^"]*)")?[^>]*>([^<]{1,60})',
                raw_html, _re.IGNORECASE
            )
            items = []
            for tag, href, text in tags:
                text = text.strip()
                if not text or len(text) < 2:
                    continue
                entry = f"{tag} | {text[:40]}"
                if href:
                    entry += f" | {href[:60]}"
                items.append(entry)
                if len(items) >= 60:
                    break
            clickable_list = "\n".join(items)
        except Exception:
            pass

        # 현재 URL
        current_url = self._get_current_browser_url()

        prompt = f"""브라우저 화면을 보고 다음 행동의 target을 결정하라.

[목표] {task_prompt[:100]}
[계획상 다음 행동] {next_step.action}: {(next_step.target or next_step.content)[:60]}
[URL] {current_url}

[클릭 가능한 요소]
{clickable_list[:50000]}

[규칙]
- 위 목록에 있는 요소만 사용. 추측 금지.
- "편집하기" 버튼이 있으면 클릭하여 편집 모드 진입.
- 404 페이지이면 action을 ASK_USER로 바꾸고 target에 "이 URL은 404입니다. 어디로 이동할까요?" 형태의 질문 작성. (특정 사이트로 자동 복귀 금지)
- 원하는 요소가 없으면 action을 ASK_USER로 바꾸고 target에 질문 작성.

JSON 한 줄만: {{"action":"CLICK","target":"CSS셀렉터","content":''}}"""

        try:
            raw = await get_llm_response_async(prompt, max_tokens=16384)
            result = _robust_json_parse(raw)
            if isinstance(result, dict) and ("target" in result or "action" in result):
                return result
        except Exception as e:
            print(f"  ⚠️ [Reactive] LLM 판단 실패: {e}")
        return None

    def _remove_task_from_schedule(self, task: Dict):
        """완료/실패 task를 schedule에서 제거 - 무한 반복 방지"""
        # [2026-05-15 P6] Layer F outcome feedback — emergence-origin task 종료 시 학습.
        # 모든 종료 경로 (DONE/FAILED/CANCELLED) 가 본 함수 단일 hook 으로 수렴 → 단일 wire.
        # 멱등 보장 위해 _layer_f_outcome_recorded 마커 사용 — 중복 호출 시 skip.
        try:
            _emerg_id = task.get("emergence_goal_id")
            if _emerg_id and not task.get("_layer_f_outcome_recorded"):
                from eidos_verb_goal_emergence import record_dispatch_outcome
                _status = str(task.get("status", "") or "").upper()
                # [P9 2026-05-15] WAITING_USER + stale_session → cancelled (학습 신호 아님).
                # WAITING_USER 가 _remove_task_from_schedule 까지 도달 = 사용자 응답
                # 없이 cleanup. stale_session 마커 = 이전 세션 잔존 (사용자 의도 X).
                # 둘 다 failure 로 분류하면 false negative 학습 신호 → Layer B 부당 weaken.
                if _status == "DONE":
                    _outcome = "success"
                elif _status in ("CANCELLED", "CANCELED", "ABORTED"):
                    _outcome = "cancelled"
                elif _status == "WAITING_USER" or task.get("_stale_session"):
                    _outcome = "cancelled"
                else:
                    _outcome = "failure"  # FAILED / 그 외 모두
                record_dispatch_outcome(
                    goal_id=_emerg_id,
                    outcome=_outcome,
                    signature=task.get("emergence_signature", ""),
                    details={
                        "task_prompt": str(task.get("task_prompt", ""))[:200],
                        "status": _status,
                        "priority": task.get("priority"),
                    },
                )
                task["_layer_f_outcome_recorded"] = True
                print(f"  🌱 [Layer F] outcome={_outcome} 기록 — goal={_emerg_id[:16]}")
        except Exception as _e_lf:
            print(f"  [Dispatcher] Layer F outcome feedback 실패 (graceful): {_e_lf}")

        try:
            sched = getattr(self.core, "schedule", [])
            if task in sched:
                sched.remove(task)
                print(f"  [ActionDispatcher] schedule에서 task 제거: {task.get('task_prompt','')[:50]}")
        except Exception as e:
            print(f"  [ActionDispatcher] task 제거 실패 (무시): {e}")

    def _pick_pending_task(self) -> Optional[Dict]:
        """schedule에서 PENDING 작업 중 우선순위 가장 높은 것 반환"""
        # [Fix] 고아 WAITING_USER 복구 — _paused_task가 사라졌는데 WAITING_USER 엔트리가
        # schedule에 남아있으면 상태를 PENDING으로 되돌려 재실행 가능하게 만든다.
        # (예외로 _paused_task가 유실되면 기존엔 영구 고립되던 버그)
        #
        # [Fix] 단, 스키마 승인이 진행 중(_pending_schema)이거나 이미 활성 스키마(_active_schema)가
        # 있으면 복구를 스킵한다. 그렇지 않으면 승인 대기 중에 동일 task가 PENDING으로 부활해
        # 또 다른 ReAct 루프가 동시 실행되는 레이스 컨디션이 발생한다.
        _schema_in_flight = (
            getattr(self, "_pending_schema", None) is not None
            or getattr(self, "_active_schema", None) is not None
        )
        if (
            self._paused_task is None
            and not self._wait_for_user
            and not _schema_in_flight
        ):
            for t in getattr(self.core, "schedule", []):
                # [안전 2026-05-17] verb_substrate 창발 목표는 **사람 승인
                # 전용** — orphan-recovery 가 WAITING_USER→PENDING 으로
                # 되돌리면 안전망 1겹(status)이 무력화된다(source 가드는
                # 별개로 유지되나 방어심층 복원). 명시적 사용자 승인 경로
                # 외에는 WAITING_USER 유지.
                if t.get("emergence_source") == "verb_substrate":
                    continue
                # [Fix] stale 마커가 있으면 이전 세션의 WAITING_USER — 부활 금지
                if t.get("status") == "WAITING_USER" and not t.get("_stale_session"):
                    t["status"] = "PENDING"
                    print(f"  [AutoRun] 고아 WAITING_USER → PENDING 복구: "
                          f"{str(t.get('task_prompt',''))[:40]}")

        pending = [
            t for t in getattr(self.core, "schedule", [])
            if t.get("status") == "PENDING"
            and t.get("source") in ("patrol", "dispatcher", "user")
            # [Fix] 이전 세션에서 남겨진 PENDING 은 사용자가 재확인 없이
            # 자동 실행되면 안 됨 — 앱 시작 시 start() 가 stale 로 마킹함.
            and not t.get("_stale_session")
            # WAITING_USER는 이어하기 전까지 건드리지 않음
        ]
        if not pending:
            return None
        pending.sort(key=lambda t: t.get("priority", 0), reverse=True)
        return pending[0]

    def _get_current_browser_url(self) -> str:
        """현재 활성 브라우저 탭의 URL 반환"""
        try:
            # Core에 gui_ref가 있으면 사용
            gui = getattr(self.core, "_gui_ref", None)
            if gui and hasattr(gui, "_get_active_browser"):
                ab = gui._get_active_browser()
                if ab and hasattr(ab, "browser"):
                    return ab.browser.url().toString()
            # Core의 parent chain을 통해 MainWindow 탐색
            import gc
            for obj in gc.get_referrers(self.core):
                if hasattr(obj, "_get_active_browser"):
                    ab = obj._get_active_browser()
                    if ab and hasattr(ab, "browser"):
                        return ab.browser.url().toString()
        except Exception as e:
            print(f"  [ActionDispatcher] URL 조회 실패: {e}")
        return ""

    # ── MissionSchema 공개 API ───────────────────────────────────────────────

    # [2026-05-04] research/조사 의도 감지 — 일반 chat 입력을 ad-hoc schema 대신
    # 단일 stage MissionChain 으로 라우팅. LLM 마이크로 라우팅 (yes/no JSON) 사용 —
    # 키워드 매칭 양쪽 오탐 (가짜 hit "비교표 작성" → research / 미스 "가격 좀 봐줘"
    # → ad-hoc) 회피. ad-hoc schema 는 첫 WRITE 직후 폐기되어 LLM 이 다음 cycle 마다
    # 새 schema 로 독자 판단 → 산출물 분산 (사용자 로그 line 184/412). chain 으로
    # 라우팅하면 chain_id + BB consolidation 정상.
    _RESEARCH_INTENT_KEYWORDS_FALLBACK: tuple = (
        "탐색", "조사", "검색", "순회", "수집", "리서치", "분석", "비교",
        "알아봐", "찾아봐", "살펴봐", "보고서",
        "research", "explore", "browse", "iterate", "analyze", "compare", "investigate",
    )

    async def _is_research_intent_async(self, task_prompt: str) -> bool:
        """LLM 마이크로 라우팅 — task_prompt 가 research/조사 의도인지 yes/no.

        gemini-2.5-flash 기준 ~200 토큰, 1-2초. 실패 시 keyword fallback.
        """
        if not task_prompt:
            return False
        # [Strangler v2 emergent-exempt 2026-05-15] Layer F 자기조직화 목표는
        # 행동 목표지 검색 대상이 아니다. research 로 오판되면 v2/chain 으로
        # 빠져 raw 목표 텍스트를 그대로 검색해 "결과 없음"이 된다. emergent
        # prefix 면 LLM 라우팅 전에 즉시 non-research 로 단정 (폴백 경로 포함
        # 모든 진입점 일관 차단).
        if task_prompt.lstrip().startswith("[자기조직화 목표]"):
            return False
        try:
            from llm_module import get_llm_response_async as _llm_router
            _router_prompt = (
                "다음 사용자 요청이 'research/조사' 의도인지 분류한다.\n\n"
                "research = 웹 페이지 탐색, 외부 정보 수집, 여러 출처 비교, 시장조사, "
                "경쟁 분석, 보고서 작성 등 multi-step browser/READ 흐름.\n"
                "non-research = 단순 발송 (이메일/메시지/텔레그램), 코드 작성/수정, "
                "계산, 즉답형 질문 (이게 뭐야?), 한 단계 액션.\n\n"
                f"사용자 요청: {task_prompt[:500]}\n\n"
                'JSON 만 출력: {"is_research": true} 또는 {"is_research": false}'
            )
            # [2026-05-06 fix] max_tokens 20 → 200. Gemini 2.5 Flash 의 thinking 토큰이
            # 응답 전에 max_output_tokens 한도를 다 소진 → finish_reason=MAX_TOKENS,
            # text 출력 0. 최신로그.txt 19:09 에 정확히 박힘. JSON 응답 자체는 ~10 토큰
            # 이라 200 이면 thinking 여유 + 응답 완성 충분.
            raw = await asyncio.wait_for(
                _llm_router(_router_prompt, max_tokens=200, timeout=15),
                timeout=18,
            )
            # 서버 오류 prefix 감지
            _ERR_PREFIXES = ("[서버 오류", "[크레딧", "[Pro", "[서버 연결 실패",
                             "[서버 응답 시간 초과", "[LLM 호출 오류")
            if not raw or any(raw.startswith(p) for p in _ERR_PREFIXES):
                raise RuntimeError(f"router LLM 오류 prefix: {raw[:60] if raw else 'empty'}")
            result = _robust_json_parse(raw)
            if isinstance(result, dict) and "is_research" in result:
                _verdict = bool(result["is_research"])
                print(f"  🔀 [Research Router/LLM] '{task_prompt[:40]}' → {_verdict}")
                return _verdict
            raise ValueError(f"is_research key 없음: {raw[:80]}")
        except Exception as e:
            # LLM 실패 → keyword fallback (안전망)
            print(f"  ⚠️ [Research Router/LLM] 실패 — keyword fallback: {e}")
            return self._is_research_intent_keyword_fallback(task_prompt)

    def _is_research_intent_keyword_fallback(self, task_prompt: str) -> bool:
        """LLM 라우터 실패 시 사용하는 안전망 — 기존 키워드/패턴 매칭."""
        if not task_prompt:
            return False
        text = task_prompt.lower()
        if any(k in text for k in self._RESEARCH_INTENT_KEYWORDS_FALLBACK):
            return True
        import re as _re_ri
        if _re_ri.search(r"\d+\s*(개|건|명|곳|가지|종)", task_prompt):
            return True
        if _re_ri.search(r"https?://|\.com|\.kr|\.net|\.io", text):
            return True
        return False

    async def _create_research_chain_inline(
        self,
        task_prompt: str,
        report_fn=None,
    ) -> bool:
        """research preset 으로 단일 stage chain 을 생성·즉시 승인·활성화.

        승인 카드 없이 바로 _active_schema 세팅. 반환 True = 활성화 성공,
        False = 실패 (호출자가 ad-hoc schema fallback 결정).
        """
        try:
            from eidos_mission_chain import get_orchestrator
        except ImportError as e:
            print(f"  ⚠️ [Research Chain] orchestrator import 실패: {e}")
            return False
        orch = get_orchestrator(core=self.core)
        if self._report_cb:
            orch.set_report_callback(self._report_cb)

        # [Fix A 2026-05-06 v2] keyword fallback path 도 chain 재사용 — 이중 생성 차단.
        # Why: 메모리 fix_dispatcher_dead_time D1 은 task.chain_id 가 있을 때만 재사용.
        # ReAct 안에서 LLM 502/RobustParser 실패 → keyword fallback 재호출 → 같은 prompt
        # 로 chain 또 생성. 그 결과 첫 chain 의 task 가 schedule 에서 제거되며 사용자
        # 입장 "작업 사라짐". registry 안 같은 topic 의 alive chain 이 있으면 schema
        # 재구성 + _active_schema 세팅 후 즉시 종료.
        # **v2 fix (최신로그.txt 17:48:42)**: registry.items() 호출이 AttributeError
        # 발생 — MissionChainRegistry 는 dict 가 아닌 커스텀 클래스로 .all() / .get()
        # API 만 노출. .all() 우선, _chains dict 직접 접근 fallback.
        try:
            _topic_norm = (task_prompt or "").strip()[:80]
            if _topic_norm:
                _registry = getattr(orch, "registry", None)
                _all_chains: List[Any] = []
                if _registry is not None:
                    if hasattr(_registry, "all") and callable(_registry.all):
                        try:
                            _all_chains = list(_registry.all())
                        except Exception:
                            _all_chains = []
                    if not _all_chains:
                        _internal = getattr(_registry, "_chains", None)
                        if isinstance(_internal, dict):
                            _all_chains = list(_internal.values())
                # [v3 2026-05-06] topic 매칭 fuzzy 화. 최신로그.txt 18:39 — 첫 호출 prompt
                # 가 wrapped 형태 ("'...' 에 대해 multi-...") 로 chain.topic 저장 → 두 번째
                # raw prompt 와 exact 매칭 실패 → 새 chain 생성. quote/공백 제거 + 양방향
                # substring 매칭 (둘 다 길이 20+ 일 때) 으로 변형 prompt 도 잡음.
                def _norm_topic(s: str) -> str:
                    return (s or "").strip().strip("'\"`").strip().lower()

                def _topic_fuzzy_match(a: str, b: str) -> bool:
                    na, nb = _norm_topic(a), _norm_topic(b)
                    if not na or not nb:
                        return False
                    if na == nb:
                        return True
                    # 짧은 쪽이 긴 쪽의 substring (둘 다 20자 이상일 때만 — 짧은 prompt 오매칭 차단)
                    short, long_ = (na, nb) if len(na) <= len(nb) else (nb, na)
                    return len(short) >= 20 and short in long_

                for _existing_chain in _all_chains:
                    _ec_id = getattr(_existing_chain, "chain_id", "") or ""
                    # [v4 2026-05-06] MissionChain 에 'topic' 속성 자체가 없음 (라인 197~ 의
                    # dataclass 정의 — chain_id/name/original_prompt 만 있음). 직전 v1~v3
                    # 의 getattr(chain, "topic", "") 는 항상 "" 반환 → 매칭 실패 → 매번 새
                    # chain 생성. 사용자 원문은 'original_prompt' 에 저장됨. fallback 으로
                    # name 의 "research: " prefix 제거.
                    _ec_topic = (
                        getattr(_existing_chain, "original_prompt", "")
                        or getattr(_existing_chain, "topic", "")
                        or ""
                    ).strip()
                    if not _ec_topic:
                        _name = getattr(_existing_chain, "name", "") or ""
                        if _name.lower().startswith("research:"):
                            _ec_topic = _name[len("research:"):].strip()
                        else:
                            _ec_topic = _name.strip()
                    _ec_topic = _ec_topic[:80]
                    _ec_status = getattr(_existing_chain, "status", None)
                    _ec_alive = (
                        _ec_status is not None
                        and getattr(_ec_status, "name", str(_ec_status)).upper()
                        not in ("ABORTED", "FAILED", "COMPLETED")
                    )
                    if _ec_alive and _topic_fuzzy_match(_ec_topic, _topic_norm) and _ec_id:
                        _schema_a = await orch.build_schema_for_current_stage(_existing_chain)
                        if _schema_a is not None:
                            self._active_schema = _schema_a
                            # [Fix 2026-05-15 orphan-chain] reuse schema 에 chain_id 명시
                            # 세팅. 누락 시 재진입 _tick 의 stale 검사(2256)에서
                            # _sch_chain='' → _chain_match=False → wrapped-quote prompt
                            # 변형과 겹쳐 stale 오판 폐기 → schema None 무한 재생성.
                            try:
                                self._active_schema.chain_id = _ec_id
                            except Exception as _e_cid_a:
                                print(f"  [Fix A] schema chain_id 세팅 실패 (무시): {_e_cid_a}")
                            print(
                                f"  ⚡ [Fix A] alive chain 재사용 — chain_id="
                                f"{_ec_id[:8]} stage='{getattr(_schema_a, 'stage', '')}' "
                                f"(keyword fallback path 이중 생성 차단)"
                            )
                            # [v5 2026-05-06] Fix A reuse path 도 schedule 에 task 등록 —
                            # 정상 path (line 15893+) mirror. Why: 호출자 (_ensure_active_schema
                            # line 2287+) 의 Fix D2 가 "chain inline 이 새 task 등록했음" 가정
                            # 하고 기존 task 를 schedule 에서 제거함. Fix A 가 새 task 등록 안
                            # 하면 schedule 빔 → 다음 tick 에서 진행 안 됨 (최신로그.txt 19:17
                            # 라인 33-34). 정상 path 와 동일하게 등록해 D2 와 정합성 유지.
                            try:
                                import datetime as _dt_a5
                                _prompt_a5 = (
                                    getattr(_schema_a, "original_prompt", "")
                                    or getattr(_schema_a, "what", "")
                                    or task_prompt
                                )
                                self.core.schedule.append({
                                    "task_prompt": _prompt_a5,
                                    "status":      "PENDING",
                                    "source":      "dispatcher_research_inline_reuse",
                                    "goal_ref":    getattr(_existing_chain, "long_term_goal_ref", "") or "",
                                    "priority":    2,
                                    "time":        _dt_a5.datetime.now().strftime("%Y-%m-%d %H:%M"),
                                    "chain_id":    _ec_id,
                                    "stage":       getattr(_schema_a, "stage", ""),
                                    "stage_index": getattr(_schema_a, "stage_index", 0),
                                })
                            except Exception as _e_sched_a5:
                                print(f"  ⚠️ [Fix A v5] schedule 등록 실패 (재사용 활성화는 성공): {_e_sched_a5}")
                            _cb_a = report_fn or self._report_cb
                            if _cb_a:
                                _cb_a(
                                    f"🔗 **[Research Chain 재사용]** "
                                    f"chain_id=`{_ec_id[:8]}` — 같은 topic 의 진행 중 chain 발견, "
                                    f"새 chain 생성 차단."
                                )
                            return True
        except Exception as _e_a:
            print(f"  [Fix A] alive chain 재사용 검사 실패 (무시, 정상 chain 생성으로 폴백): {_e_a}")

        try:
            chain = orch.create_chain_from_preset(
                "research",
                original_prompt=task_prompt,
                display_name=f"research: {task_prompt[:40]}",
                topic=task_prompt[:80],
            )
        except Exception as e:
            print(f"  ⚠️ [Research Chain] 생성 실패: {e}")
            return False
        if not orch.approve_chain(chain.chain_id):
            print(f"  ⚠️ [Research Chain] approve 실패: {chain.chain_id}")
            return False
        first = await orch.build_schema_for_current_stage(chain)
        if first is None:
            print(f"  ⚠️ [Research Chain] 첫 stage schema 생성 실패")
            return False
        self._active_schema = first
        # dispatcher _tick 이 pickup 하도록 schedule 등록 (approve_pending_chain 패턴 mirror)
        try:
            import datetime as _dt_rc
            _prompt_rc = (
                getattr(first, "original_prompt", "")
                or getattr(first, "what", "")
                or task_prompt
            )
            self.core.schedule.append({
                "task_prompt": _prompt_rc,
                "status":      "PENDING",
                "source":      "dispatcher_research_inline",
                "goal_ref":    chain.long_term_goal_ref or "",
                "priority":    2,
                "time":        _dt_rc.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "chain_id":    chain.chain_id,
                "stage":       getattr(first, "stage", ""),
                "stage_index": getattr(first, "stage_index", 0),
            })
        except Exception as _e_sched:
            print(f"  ⚠️ [Research Chain] schedule 등록 실패 (활성화는 성공): {_e_sched}")
        _cb = report_fn or self._report_cb
        if _cb:
            _cb(
                f"🔗 **[Research Chain 자동 활성화]** "
                f"chain_id=`{chain.chain_id[:8]}` — ad-hoc schema 대신 단일 stage "
                f"MissionChain 으로 라우팅 (BB consolidation + accumulated_outputs 활성)."
            )
        print(
            f"  🔗 [Research Chain] 자동 승인 + 활성화 완료 "
            f"(chain_id={chain.chain_id[:8]}, schema.chain_id="
            f"{getattr(first, 'chain_id', '')[:8]})"
        )
        return True

    async def request_schema_approval(
        self,
        task_prompt: str,
        report_fn=None,
    ) -> None:
        """
        task_prompt를 받아 MissionSchema를 생성하고
        코파일럿 사이드바에 승인 카드를 표시한다.

        GUI의 _on_auto_run_btn_clicked 또는 schedule task 시작 시 호출.
        승인이 완료되면 _active_schema가 채워지고 run_now()가 트리거된다.

        [2026-05-04] research 의도 감지 시 ad-hoc schema 대신 단일 stage
        MissionChain 으로 라우팅 (chain_id + BB consolidation 활성화).
        """
        # [Fix B 2026-05-06] 이미 활성 schema 가 있으면 재진입 차단 — 안전망.
        # Why: ReAct 루프 안에서 LLM 호출 실패 → 예상치 못한 path 로 본 함수 재진입 시
        # 새 chain/schema 생성 위험. _active_schema 가 alive chain 가리키면 즉시 종료.
        # Fix A 가 _create_research_chain_inline 진입을 막지만, 그 외 경로 (LLM 라우팅
        # 진입 자체를 우회하는 path) 도 보호.
        if self._active_schema is not None:
            _stage_b = getattr(self._active_schema, "stage", "")
            _chain_id_b = (getattr(self._active_schema, "chain_id", "") or "")[:8]
            print(
                f"  [Fix B] _active_schema 이미 활성 (chain={_chain_id_b}, stage='{_stage_b}') "
                f"— request_schema_approval 재진입 차단"
            )
            return

        # [2026-05-04] research 의도 → MissionChain 강제 라우팅 (LLM 라우팅).
        # 사용자가 explicitly request_chain_approval 호출한 경우는 별도 경로라
        # 본 분기 안 탄다. 본 분기는 일반 chat → request_schema_approval 진입점만.
        _norm_check = (task_prompt or "").strip()
        if _norm_check and await self._is_research_intent_async(_norm_check):
            ok = await self._create_research_chain_inline(_norm_check, report_fn)
            if ok:
                return  # chain 활성화 성공 — ad-hoc schema 생성 skip
            # 실패 시 ad-hoc fallback 으로 계속 진행

        try:
            from eidos_mission_schema import generate_schema_async
        except ImportError as e:
            print(f"⚠️ [Schema] eidos_mission_schema import 실패: {e}")
            return

        # [Fix] 중복 생성 차단 — GUI가 schedule 추가와 schema 생성을 동시에
        # 트리거하면 dispatcher가 _pending_schema set 전에 PENDING task를
        # picking up해서 두 번째 호출이 발생함 (LLM 1734토큰 두 번 소비).
        # 동일 prompt에 대한 in-flight 표식이 있으면 즉시 리턴.
        _norm = _norm_check
        _inflight = getattr(self, "_schema_inflight_prompt", "") or ""
        if _inflight and _inflight == _norm:
            print(f"  [Schema] 중복 생성 차단 — in-flight 동일 prompt: '{_norm[:40]}'")
            return
        # _pending_schema가 이미 같은 prompt로 만들어져 있으면 재생성 불필요
        _pend = getattr(self, "_pending_schema", None)
        if _pend is not None:
            _pend_what = (getattr(_pend, "what", "") or "").strip()
            if _pend_what == _norm or _norm in _pend_what or _pend_what in _norm:
                print(f"  [Schema] 중복 생성 차단 — _pending_schema 동일 prompt")
                return

        self._schema_inflight_prompt = _norm
        _cb = report_fn or self._report_cb
        if _cb:
            _cb("🔄 **[Mission Schema]** 목표 스키마 생성 중...")

        try:
            schema = await generate_schema_async(task_prompt, core=self.core)
            self._pending_schema = schema
            if _cb:
                _cb("[SCHEMA] " + schema.to_approval_card())
            print(f"  [Schema] 승인 요청 전송: task_type='{schema.task_type}'")
        except Exception as e:
            import traceback
            print(f"❌ [Schema] 생성 실패: {e}")
            traceback.print_exc()
            if _cb:
                _cb(f"⚠️ Schema 생성 실패: {e}")
        finally:
            # in-flight 해제 — 동일 prompt가 또 들어오면 정상 처리
            if getattr(self, "_schema_inflight_prompt", "") == _norm:
                self._schema_inflight_prompt = ""

    # ── MissionChain 공개 API (2026-04-19) ───────────────────────────────
    async def request_chain_approval(
        self,
        preset_name: str,
        original_prompt: str,
        *,
        display_name: str = "",
        long_term_goal_ref: str = "",
        report_fn=None,
        **preset_kwargs,
    ) -> Optional[str]:
        """
        MissionChain 제안 → _pending_chain 에 세팅 → 승인 카드 표시.
        승인되면 approve_pending_chain() 이 호출되고 첫 단계 schema가 자동 실행된다.
        반환: chain_id (실패 시 None)
        """
        try:
            from eidos_mission_chain import get_orchestrator
        except ImportError as e:
            print(f"  ⚠️ [MissionChain] import 실패: {e}")
            return None
        orch = get_orchestrator(core=self.core)
        if self._report_cb:
            orch.set_report_callback(self._report_cb)
        try:
            chain = orch.create_chain_from_preset(
                preset_name,
                original_prompt=original_prompt,
                display_name=display_name,
                long_term_goal_ref=long_term_goal_ref,
                **preset_kwargs,
            )
        except KeyError:
            print(f"  ⚠️ [MissionChain] 알 수 없는 preset: '{preset_name}'")
            return None
        except Exception as e:
            import traceback
            print(f"  ⚠️ [MissionChain] 체인 생성 실패: {e}")
            traceback.print_exc()
            return None

        self._pending_chain = chain
        _cb = report_fn or self._report_cb

        # [2026-05-24 chain 자동 승인 토글] 사용자 명시 요청:
        # "missionchain 없애라고 했는데 안 없앴지? 그냥 바로 실행하라고".
        # settings.mission_chain_auto_approve=True (default) — 승인 카드 skip + 즉시 진행.
        # False 면 옛 카드 표시 후 사용자 `승인` 입력 대기.
        _auto_approve = True
        try:
            import json as _j_sa
            _sf = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "eidos_settings.json",
            )
            if os.path.exists(_sf):
                with open(_sf, "r", encoding="utf-8") as _fsa:
                    _ssa = _j_sa.load(_fsa) or {}
                _v = _ssa.get("mission_chain_auto_approve")
                if _v is not None:
                    _auto_approve = bool(_v)
        except Exception as _e_sa:
            print(f"  ⚠️ [MissionChain auto-approve] settings 조회 실패 (default True): {_e_sa}")

        if _auto_approve:
            # 자동 승인 — 옛 승인 카드 skip·짧은 진행 안내만
            if _cb:
                _total_sec = sum(int(getattr(s, "expected_duration_sec", 0) or 0) for s in chain.stages)
                _total_min = max(1, _total_sec // 60)
                _stage_names = "·".join(s.name for s in chain.stages[:6])
                _cb(
                    f"⚡ **[MissionChain 자동 승인]** `{chain.name}` 즉시 실행\n"
                    f"└─ {len(chain.stages)}단계 (~{_total_min}분) · {_stage_names}\n"
                    f"└─ 원문: `{original_prompt[:80]}`\n"
                    f"└─ 끄려면 `eidos_settings.json` 의 "
                    f"`mission_chain_auto_approve=false` 설정 후 재시작"
                )
            try:
                _approved = await self.approve_pending_chain()
                if _approved:
                    try:
                        await self.run_now()
                    except Exception as _re:
                        print(f"  ⚠️ [MissionChain auto-approve] run_now 실패 (graceful): {_re}")
                else:
                    print(f"  ⚠️ [MissionChain auto-approve] approve_pending_chain → False")
            except Exception as _ae:
                print(f"  ⚠️ [MissionChain auto-approve] approve 호출 실패: {_ae}")
            return chain.chain_id

        # [기존 동작] 사용자 명시 토글 OFF 시 — 옛 승인 카드 표시
        if _cb:
            _RISK_EMOJI = {"low": "🟢", "medium": "🟡", "high": "🔴"}
            _total_sec = sum(int(getattr(s, "expected_duration_sec", 0) or 0) for s in chain.stages)
            _total_min = max(1, _total_sec // 60)
            _lines = [
                f"🔗 **[MissionChain 승인 요청]** `{chain.name}`",
                f"└─ 총 {len(chain.stages)}단계 · 예상 소요 ~{_total_min}분",
                f"└─ 원문: `{original_prompt[:80]}`",
                "",
                "**실행 단계:**",
            ]
            for s in chain.stages:
                _risk_mark = _RISK_EMOJI.get((s.risk_level or "low").lower(), "🟢")
                _auto_mark = "" if s.auto_approve else " 🔒"
                _min = max(1, int(getattr(s, "expected_duration_sec", 0) or 0) // 60)
                _lines.append(
                    f"  **[{s.stage_index + 1}/{len(chain.stages)}]** {_risk_mark} "
                    f"**{s.name}**  ~{_min}분 ({s.risk_level}){_auto_mark}"
                )
                if s.description:
                    _lines.append(f"      └─ {s.description}")
            _lines.append("")
            _lines.append("• `승인` · 전체 단계를 자동 진행")
            _lines.append("• `취소` · 체인 폐기")
            _lines.append("• 🔒 표시된 민감 단계는 차례가 오면 개별 재승인이 필요합니다.")
            _cb("\n".join(_lines))
        return chain.chain_id

    async def approve_pending_chain(self) -> bool:
        """대기 중인 체인을 승인하고 첫 단계 schema 를 즉시 실행 가능 상태로 세팅."""
        chain = self._pending_chain
        if chain is None:
            return False
        try:
            from eidos_mission_chain import get_orchestrator
            orch = get_orchestrator(core=self.core)
            if self._report_cb:
                orch.set_report_callback(self._report_cb)
            if not orch.approve_chain(chain.chain_id):
                return False
            first = await orch.build_schema_for_current_stage(chain)
            if first is None:
                return False
            self._active_schema = first
            self._pending_chain = None
            try:
                import datetime as _dt
                _prompt = (
                    getattr(first, "original_prompt", "")
                    or getattr(first, "what", "")
                    or f"[chain {chain.chain_id[:8]}] 첫 단계"
                )
                self.core.schedule.append({
                    "task_prompt": _prompt,
                    "status":      "PENDING",
                    "source":      "dispatcher",
                    "goal_ref":    chain.long_term_goal_ref or "",
                    "priority":    2,
                    "time":        _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "chain_id":    chain.chain_id,
                    "stage":       getattr(first, "stage", ""),
                    "stage_index": getattr(first, "stage_index", 0),
                })

                # [Fix 2026-04-21 Phase 3] chain 시작 시 산출물 파일 세션 초기화.
                # Why: Shortening Guard 가 디스크의 "기존 N자"(주로 이전 세션 잔재)
                # 와 새 생성물을 비교해 짧은 draft 를 차단. 그 "이전 세션 잔재"가
                # 종종 환각이어서 chain 실행을 원천부터 오염시킴. chain 승인 순간
                # stage.output_keys 에 해당하는 파일을 .pre-chain.{ts}.bak 로 이동해
                # 깨끗한 상태에서 시작하게 한다. .md/.txt 만 대상(사용자 데이터 보호).
                try:
                    import os as _os_pc
                    import time as _t_pc
                    import re as _re_pc_mod
                    from execution_module import SAFE_BASE_PATH as _SB_PC
                    _base_abs_pc = _os_pc.path.abspath(_SB_PC)
                    _ARTIFACT_EXTS_PC = (".md", ".txt")
                    _ts_pc = int(_t_pc.time())
                    _moved_pc = 0
                    # [P0 2026-04-21] output_keys 만으로는 커버 못 하는 mismatch 방어.
                    # 증상: preset 의 `plan` stage output_keys=['plan_doc',...] 인데 실제
                    # prompt 는 "plan.md 를 작성" 지시 → Chain Reset 이 plan_doc.md 를
                    # 찾아 이전 세션 plan.md 가 남음 → Shortening Guard 가 새 draft 를
                    # 매번 차단(로그 2026-04-21 16:58~59 plan.md 4442자 루프).
                    # 해결: prompt_template/description 에서 *.md/*.txt 파일명도 regex 로
                    # 긁어 candidate 집합에 합친다. 사용자 데이터 보호를 위해 확장자
                    # 필터와 SAFE_BASE 내 경로 체크는 그대로 유지.
                    _FNAME_RE_PC = _re_pc_mod.compile(
                        r"([A-Za-z0-9가-힣_\-][A-Za-z0-9가-힣_\-\.]*\.(?:md|txt))",
                        _re_pc_mod.IGNORECASE,
                    )
                    _seen_cands_pc: set = set()
                    for _stg in (chain.stages or []):
                        _cands_for_stage: list = []
                        for _ok in (_stg.output_keys or []):
                            _ok_s = str(_ok).strip()
                            if not _ok_s:
                                continue
                            _cands_for_stage.append(_ok_s)
                            if "." not in _ok_s:
                                _cands_for_stage += [f"{_ok_s}{e}" for e in _ARTIFACT_EXTS_PC]
                        # prompt_template + description 에서 파일명 스캔
                        for _txt_src in (
                            getattr(_stg, "prompt_template", "") or "",
                            getattr(_stg, "description", "") or "",
                        ):
                            try:
                                for _m in _FNAME_RE_PC.findall(str(_txt_src)):
                                    _cands_for_stage.append(_m)
                            except Exception:
                                pass
                        for _cand in _cands_for_stage:
                            _rel_pc = _cand.replace("\\", "/").lstrip("./")
                            if not _rel_pc.lower().endswith(_ARTIFACT_EXTS_PC):
                                continue
                            _abs_pc = _os_pc.path.abspath(_os_pc.path.join(_base_abs_pc, _rel_pc))
                            if not (_abs_pc == _base_abs_pc or _abs_pc.startswith(_base_abs_pc + _os_pc.sep)):
                                continue
                            if _abs_pc in _seen_cands_pc:
                                continue
                            _seen_cands_pc.add(_abs_pc)
                            if not _os_pc.path.isfile(_abs_pc):
                                continue
                            _bak_pc = f"{_abs_pc}.pre-chain.{_ts_pc}.bak"
                            try:
                                _os_pc.rename(_abs_pc, _bak_pc)
                                _moved_pc += 1
                                print(f"  🧹 [Chain Reset] {_abs_pc} → {_bak_pc}")
                            except Exception as _re_pc:
                                print(f"  ⚠️ [Chain Reset] 백업 실패({_cand}): {_re_pc}")
                    if _moved_pc:
                        self._report(
                            f"🧹 **[체인 초기화]** 이전 세션 산출물 {_moved_pc}개를 "
                            f".pre-chain.bak 로 이동 — 깨끗한 상태로 시작"
                        )
                except Exception as _pc_e:
                    print(f"  ⚠️ [Chain Reset] 산출물 초기화 스킵(무시): {_pc_e}")

                self._report(
                    f"🚀 **[MissionChain 시작]** `{chain.name}` — "
                    f"stage[0] '{getattr(first, 'stage', '')}' 즉시 실행"
                )
            except Exception as e:
                print(f"  ⚠️ [MissionChain] 첫 태스크 등록 실패: {e}")
            return True
        except Exception as e:
            import traceback
            print(f"  ⚠️ [MissionChain] 승인 처리 실패: {e}")
            traceback.print_exc()
            return False

    # ── 체인 편집 공개 API (2026-04-19) ───────────────────────────────
    def _rerender_pending_chain_card(self) -> None:
        """_pending_chain 의 현재 상태로 승인 카드를 다시 찍는다."""
        chain = self._pending_chain
        if chain is None or self._report_cb is None:
            return
        _RISK_EMOJI = {"low": "🟢", "medium": "🟡", "high": "🔴"}
        _total_sec = sum(
            int(getattr(s, "expected_duration_sec", 0) or 0)
            for s in chain.stages if not getattr(s, "skipped", False)
        )
        _total_min = max(1, _total_sec // 60)
        active = [s for s in chain.stages if not getattr(s, "skipped", False)]
        _lines = [
            f"🔗 **[MissionChain 승인 대기 · 편집 반영됨]** `{chain.name}`",
            f"└─ 활성 {len(active)}/{len(chain.stages)}단계 · 예상 소요 ~{_total_min}분",
            "",
            "**실행 단계:**",
        ]
        for s in chain.stages:
            _risk_mark = _RISK_EMOJI.get((s.risk_level or "low").lower(), "🟢")
            _auto_mark = "" if s.auto_approve else " 🔒"
            _skip_mark = " ⏭️(skipped)" if getattr(s, "skipped", False) else ""
            _min = max(1, int(getattr(s, "expected_duration_sec", 0) or 0) // 60)
            _lines.append(
                f"  **[{s.stage_index + 1}/{len(chain.stages)}]** {_risk_mark} "
                f"**{s.name}**  ~{_min}분 ({s.risk_level}){_auto_mark}{_skip_mark}"
            )
            if s.description:
                _lines.append(f"      └─ {s.description}")
        _lines.append("")
        _lines.append("• `승인` · 전체 단계 실행  ·  `취소` · 체인 폐기")
        _lines.append("• `건너뛰기 N` / `되돌리기 N` · 단계 skip 토글")
        _lines.append("• `삭제 N` · 단계 제거  ·  `수정 N 새 설명` · 설명 변경")
        self._report_cb("\n".join(_lines))

    def edit_pending_chain_skip(self, stage_index: int, skip: bool = True) -> bool:
        if self._pending_chain is None:
            return False
        from eidos_mission_chain import get_orchestrator
        orch = get_orchestrator(core=self.core)
        ok = orch.edit_skip_stage(self._pending_chain.chain_id, stage_index, skip=skip)
        if ok:
            self._rerender_pending_chain_card()
        return ok

    def edit_pending_chain_remove(self, stage_index: int) -> bool:
        if self._pending_chain is None:
            return False
        from eidos_mission_chain import get_orchestrator
        orch = get_orchestrator(core=self.core)
        ok = orch.edit_remove_stage(self._pending_chain.chain_id, stage_index)
        if ok:
            self._rerender_pending_chain_card()
        return ok

    def edit_pending_chain_update(
        self,
        stage_index: int,
        *,
        description: Optional[str] = None,
        prompt_template: Optional[str] = None,
    ) -> bool:
        if self._pending_chain is None:
            return False
        from eidos_mission_chain import get_orchestrator
        orch = get_orchestrator(core=self.core)
        ok = orch.edit_update_stage(
            self._pending_chain.chain_id, stage_index,
            description=description, prompt_template=prompt_template,
        )
        if ok:
            self._rerender_pending_chain_card()
        return ok

    def edit_pending_chain_insert(
        self,
        insert_at: int,
        name: str,
        description: str = "",
        prompt_template: str = "",
        task_type: str = "general",
        verb: str = "GENERAL",
    ) -> bool:
        if self._pending_chain is None:
            return False
        from eidos_mission_chain import get_orchestrator, StageSpec
        orch = get_orchestrator(core=self.core)
        stage = StageSpec(
            name=name or "custom",
            stage_index=insert_at,
            task_type=task_type,
            verb=verb,
            prompt_template=prompt_template,
            description=description,
            auto_approve=True,
            risk_level="low",
            expected_duration_sec=180,
        )
        ok = orch.edit_insert_stage(self._pending_chain.chain_id, insert_at, stage)
        if ok:
            self._rerender_pending_chain_card()
        return ok

    def reject_pending_chain(self, reason: str = "사용자 거절") -> bool:
        chain = self._pending_chain
        if chain is None:
            return False
        try:
            from eidos_mission_chain import get_orchestrator
            orch = get_orchestrator(core=self.core)
            orch.abort_chain(chain.chain_id, reason=reason)
        except Exception as e:
            print(f"  ⚠️ [MissionChain] 거절 처리 실패: {e}")
        self._pending_chain = None
        return True

    # ── 체인 강제 진행 API (2026-04-19) ───────────────────────────────────
    def force_advance_active_chain_stage(self) -> bool:
        """
        사용자 "다음단계" 명령 — 현재 실행 중인 chain stage 를 강제로 success 처리한다.
        실제 break-and-advance 는 _react_loop_impl 의 다음 iteration 시작에서 일어남
        (LLM/네트워크 await 중에는 즉시 끊을 수 없음).

        반환: True 면 플래그 set 성공 (active chain task 가 있음).
              False 면 active chain 없음 (단일 mission 이거나 아무것도 안 돌아감).
        """
        sch = self._active_schema
        if sch is None or not getattr(sch, "chain_id", ""):
            return False
        self._force_advance_chain = True
        return True

    def abort_active_chain(self) -> bool:
        """
        사용자 "그만/취소" 명령 — 현재 실행 중인 chain 을 통째로 중단한다.
        force_advance 와 동일하게 다음 iteration 진입 시 실제 break + abort 일어남.
        """
        sch = self._active_schema
        if sch is None or not getattr(sch, "chain_id", ""):
            return False
        self._abort_active_chain = True
        return True

    def _find_helix_stage_artifact(self, helix_result, schema):
        """
        Helix 실행 결과(tree_json) 에서 현재 chain stage 의 완료 산출물로
        매칭되는 파일쓰기(action=3) DONE 노드를 찾아 (파일명, 파일내용) 반환.
        없으면 None. 파일내용은 read 실패 시 빈 문자열.

        왜 필요한가:
          ReAct 는 WRITE 시 _is_stage_completion_artifact → _on_react_done 로
          체인을 전진시키지만, Helix 는 자체 파일쓰기를 거치기 때문에 그 훅에
          걸리지 않아 stage 가 정체된다 (2026-04-19 로그).
        """
        try:
            nodes = (getattr(helix_result, "tree_json", {}) or {}).get("nodes", {})
            if not nodes:
                return None

            # DONE + action==3(파일쓰기) + where 유효한 노드 수집.
            write_nodes = []
            for n in nodes.values():
                if not isinstance(n, dict):
                    continue
                if (n.get("node_type") or "") != "DONE":
                    continue
                if n.get("action") != 3:
                    continue
                where = (n.get("where") or "").strip()
                if not where:
                    continue
                res = (n.get("result") or "").strip()
                if res.startswith("[") and "차단" in res[:40]:
                    continue
                # "저장 완료" 류 키워드로 최소 성공 필터
                if "저장 완료" not in res and "완료" not in res[:40]:
                    continue
                write_nodes.append((where, n.get("updated_at", 0.0)))

            if not write_nodes:
                return None

            # stage 산출물과 매칭되는 것만 유지
            matched = []
            for fname, ts in write_nodes:
                try:
                    if self._is_stage_completion_artifact(schema, fname):
                        matched.append((fname, ts))
                except Exception:
                    continue
            if not matched:
                return None

            # 가장 최근에 갱신된 파일을 선택
            matched.sort(key=lambda x: x[1], reverse=True)
            chosen = matched[0][0]

            # 실제 파일 내용 읽기 시도 (downstream_seeds 채움용)
            content = ""
            try:
                from execution_module import SAFE_BASE_PATH as _SBASE
                import os as _os_fh
                _fp = _os_fh.path.join(_SBASE, chosen)
                if _os_fh.path.exists(_fp):
                    with open(_fp, "r", encoding="utf-8") as _fh:
                        content = _fh.read()
            except Exception:
                pass

            return chosen, content
        except Exception as _e:
            print(f"  ⚠️ [_find_helix_stage_artifact] 스캔 실패: {_e}")
            return None

    def _is_stage_completion_artifact(self, schema, target: str) -> bool:
        """
        WRITE target 이 현재 chain stage 의 "완료 신호 산출물" 인지 판정.

        매칭 규칙 (셋 중 하나라도 맞으면 True):
          (a) target 이 stage.output_keys 의 어떤 키와 정확 일치 (확장자 포함/미포함 모두)
          (b) target 이 "{output_key}.md/.txt/.json/.csv/.html/.yaml/.yml/.xml" 중 하나
          (c) target 이 stage.prompt_template 안에 등장하는 *.md/*.txt/*.json 파일명과 일치

        규칙은 보수적 — 무관한 WRITE(예: 임시 변수)는 영향 없음.
        """
        if not target or not schema:
            return False
        try:
            from eidos_mission_chain import MissionChainRegistry
            chain_id = getattr(schema, "chain_id", "") or ""
            chain = MissionChainRegistry.instance().get(chain_id) if chain_id else None
            if chain is None:
                return False
            stage_idx = int(getattr(schema, "stage_index", 0) or 0)
            if not (0 <= stage_idx < len(chain.stages)):
                return False
            stage = chain.stages[stage_idx]

            t_lc = target.strip().lower()
            t_base = t_lc.rsplit(".", 1)[0] if "." in t_lc else t_lc

            # (a) output_keys 직접 일치
            output_keys = [str(k).strip().lower() for k in (stage.output_keys or [])]
            if t_lc in output_keys or t_base in output_keys:
                return True

            # (b) {output_key}.{ext} 패턴
            _ARTIFACT_EXTS = (".md", ".txt", ".json", ".csv", ".html",
                              ".yaml", ".yml", ".xml")
            for ok in output_keys:
                for ext in _ARTIFACT_EXTS:
                    if t_lc == f"{ok}{ext}":
                        return True

            # (c) prompt_template 안의 파일명 (정규식으로 추출)
            import re as _re_pt
            _tmpl = (stage.prompt_template or "")
            for m in _re_pt.finditer(r"\b([a-zA-Z0-9_\-]+\.(?:md|txt|json|csv|html|yaml|yml|xml))\b", _tmpl):
                fn = m.group(1).strip().lower()
                if t_lc == fn:
                    return True

            # (d) [Fix 2026-04-21 P0-B] 확장자 기반 폴백.
            # Why: LLM이 stage.output_keys 와 다른 파일명(product_summaries.md 등)을
            #      쓰면 (a)(b)(c) 모두 miss → stage 종료 못 하고 40+스텝 NAVIGATE/CLICK
            #      루프. 표준 산출물 확장자면 stage 완료로 인정해 루프 차단.
            #      이미 호출부에서 "생성 완료" 성공 신호를 확인했으므로 안전.
            if t_lc.endswith(_ARTIFACT_EXTS):
                print(
                    f"  📁 [Chain Auto-DONE] 확장자 폴백 매칭: "
                    f"target='{target}' (stage.output_keys={output_keys})"
                )
                return True

            return False
        except Exception:
            return False

    # ── 체인 실패 응답 API (P0 2026-04-19) ────────────────────────────────
    async def retry_pending_chain_failure(self) -> bool:
        """현재 실패한 단계를 재시도. retry_budget 내에서만 성공."""
        fail = self._pending_chain_failure
        if not fail:
            return False
        try:
            from eidos_mission_chain import get_orchestrator
            orch = get_orchestrator(core=self.core)
            if self._report_cb:
                orch.set_report_callback(self._report_cb)
            nxt = await orch.retry_current_stage(fail["chain_id"])
            if nxt is None:
                self._report("⚠️ [MissionChain] 재시도 불가 — budget 소진 또는 체인 없음")
                return False
            self._pending_chain_failure = None
            await self._inject_chain_stage_schema(nxt, fail["chain_id"])
            return True
        except Exception as e:
            import traceback
            print(f"  ⚠️ [MissionChain] retry 실패: {e}")
            traceback.print_exc()
            return False

    async def rollback_pending_chain_failure(self, stage_index: int) -> bool:
        """명시적 stage 번호(0-based)로 롤백 후 재개."""
        fail = self._pending_chain_failure
        if not fail:
            return False
        try:
            from eidos_mission_chain import get_orchestrator
            orch = get_orchestrator(core=self.core)
            if self._report_cb:
                orch.set_report_callback(self._report_cb)
            nxt = await orch.resume_from_stage(fail["chain_id"], stage_index)
            if nxt is None:
                self._report(f"⚠️ [MissionChain] 롤백 불가 — stage_index={stage_index}")
                return False
            self._pending_chain_failure = None
            await self._inject_chain_stage_schema(nxt, fail["chain_id"])
            return True
        except Exception as e:
            import traceback
            print(f"  ⚠️ [MissionChain] rollback 실패: {e}")
            traceback.print_exc()
            return False

    def abort_pending_chain_failure(self) -> bool:
        """실패 체인 완전 포기."""
        fail = self._pending_chain_failure
        if not fail:
            return False
        try:
            from eidos_mission_chain import get_orchestrator
            orch = get_orchestrator(core=self.core)
            orch.abort_chain(fail["chain_id"], reason="사용자 포기(실패 후)")
        except Exception as e:
            print(f"  ⚠️ [MissionChain] 포기 처리 예외: {e}")
        self._pending_chain_failure = None
        self._active_schema = None
        return True

    async def _inject_chain_stage_schema(self, nxt: Any, chain_id: str) -> None:
        """retry/rollback 결과 schema 를 _active_schema / _pending_schema 로 주입 + PENDING task 등록."""
        import datetime as _dt
        _prompt = (
            getattr(nxt, "original_prompt", "")
            or getattr(nxt, "what", "")
            or f"[chain {chain_id[:8]}] 재시도 단계"
        )
        try:
            self.core.schedule.append({
                "task_prompt": _prompt,
                "status":      "PENDING",
                "source":      "dispatcher",
                "goal_ref":    getattr(nxt, "long_term_goal_ref", "") or "",
                "priority":    2,
                "time":        _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "chain_id":    chain_id,
                "stage":       getattr(nxt, "stage", ""),
                "stage_index": getattr(nxt, "stage_index", 0),
            })
        except Exception as e:
            print(f"  ⚠️ [MissionChain] retry task 등록 실패: {e}")
        if getattr(nxt, "approved", False):
            self._active_schema = nxt
            self._report(
                f"▶️ [MissionChain] stage[{getattr(nxt, 'stage_index', 0)}] "
                f"'{getattr(nxt, 'stage', '')}' 재가동"
            )
            try:
                await self.run_now()
            except Exception:
                pass
        else:
            self._active_schema = None
            self._pending_schema = nxt
            try:
                _card = nxt.to_approval_card() if hasattr(nxt, "to_approval_card") else str(nxt.to_dict())
                self._report(
                    f"🔒 [MissionChain] 재시도 단계 개별 승인 필요\n{_card}\n\n"
                    "💡 **승인** 을 입력하면 실행합니다."
                )
            except Exception:
                pass

    # ── CodeBuilder 결과 UX 공개 API (P2-7 2026-04-19) ───────────────────
    def _emit_codebuild_result_card(self) -> None:
        """현재 _last_codebuild_result 를 사람이 읽기 좋은 카드로 채팅에 흘림."""
        res = self._last_codebuild_result
        if not res:
            return
        files = res.get("files", {}) or {}
        file_list = list(files.keys())
        if not file_list:
            self._report("⚠️ [CodeBuilder 결과] 생성된 파일 없음.")
            return
        lines = [
            f"🎨 **[CodeBuilder 결과]** {res.get('mode','?')} 모드 · "
            f"{len(file_list)}개 파일 · `{res.get('project_dir','')}`",
        ]
        for i, rel in enumerate(file_list, 1):
            _size = len((files.get(rel) or "")[:1_000_000])
            lines.append(f"  **[{i}]** `{rel}`  ({_size}자)")
        # description.md 프리뷰
        desc = (files.get("description.md") or "").strip()
        if desc:
            preview = desc[:500]
            if len(desc) > 500:
                preview += "..."
            lines.append("")
            lines.append("**description.md 미리보기:**")
            lines.append(f"> {preview}")
        lines.append("")
        lines.append("• `미리보기 N` — N번 파일 전체 확인")
        lines.append("• `열기` — 저장 경로를 파일탐색기 힌트로 표시")
        lines.append("• `재생성` — 이번 dev 단계만 다시 실행")
        self._report("\n".join(lines))

    def show_codebuild_preview(self, index_one_based: int = 0) -> bool:
        """사용자가 `미리보기 N` 입력 시 호출. index 가 0이면 description.md 반환."""
        res = self._last_codebuild_result
        if not res:
            self._report("ℹ️ 표시할 최근 CodeBuilder 결과가 없습니다.")
            return False
        files = res.get("files", {}) or {}
        file_list = list(files.keys())
        if not file_list:
            self._report("ℹ️ 최근 결과에 파일이 없습니다.")
            return False
        if index_one_based <= 0:
            # 기본: description.md 또는 첫 파일
            rel = "description.md" if "description.md" in files else file_list[0]
        else:
            if index_one_based > len(file_list):
                self._report(f"⚠️ 잘못된 번호: 1~{len(file_list)} 범위만 유효.")
                return False
            rel = file_list[index_one_based - 1]
        content = files.get(rel) or ""
        # 너무 길면 자르기
        _MAX = 3500
        body = content[:_MAX] + (("\n... (잘림, 전체는 파일에서 확인)" ) if len(content) > _MAX else "")
        self._report(
            f"📄 **[미리보기]** `{rel}`  ({len(content)}자)\n"
            f"```\n{body}\n```"
        )
        return True

    def show_codebuild_open_hint(self) -> bool:
        res = self._last_codebuild_result
        if not res:
            self._report("ℹ️ 표시할 최근 CodeBuilder 결과가 없습니다.")
            return False
        path = res.get("project_dir", "")
        if not path:
            self._report("ℹ️ 저장 경로 정보가 없습니다.")
            return False
        self._report(
            f"📂 **[저장 경로]** `{path}`\n"
            "└─ 파일 탐색기/에디터에서 직접 열어 편집하세요. "
            "편집 후 `재생성` 대신 그냥 다음 단계로 진행해도 됩니다."
        )
        return True

    async def regenerate_last_codebuild(self) -> bool:
        """
        최근 dev 단계를 강제로 재시도. _last_codebuild_result 에서 chain_id/stage_index 가져와
        orchestrator.resume_from_stage 로 정확히 해당 stage 부터 재개.
        """
        res = self._last_codebuild_result
        if not res or not res.get("chain_id"):
            self._report("ℹ️ 재생성할 체인 dev 단계가 없습니다. (단독 code_build 는 재지시로 다시 실행하세요)")
            return False
        try:
            from eidos_mission_chain import get_orchestrator
            orch = get_orchestrator(core=self.core)
            if self._report_cb:
                orch.set_report_callback(self._report_cb)
            nxt = await orch.resume_from_stage(
                res["chain_id"], int(res.get("stage_index", 0) or 0)
            )
            if nxt is None:
                self._report("⚠️ 재생성 실패 — orchestrator 가 None 반환.")
                return False
            await self._inject_chain_stage_schema(nxt, res["chain_id"])
            # 기존 카드는 그대로 두되, 새 결과로 덮어질 예정.
            self._report("🔁 **[재생성]** dev 단계를 다시 실행합니다...")
            return True
        except Exception as e:
            import traceback
            print(f"  ⚠️ [CodeBuilder] 재생성 실패: {e}")
            traceback.print_exc()
            return False

    def get_active_anchor(self) -> Optional[str]:
        """현재 활성 MissionSchema의 앵커 컨텍스트 반환 (ReAct Think에 주입)"""
        if self._active_schema and self._active_schema.approved:
            return self._active_schema.get_anchor_context()
        return None

    def record_schema_step(self, action: str, target: str, result: str, url: str = "") -> None:
        """ReAct 실행 히스토리를 활성 스키마에 기록"""
        if self._active_schema:
            self._active_schema.add_step(action, target, result, url)

    def mark_attribute_achieved(self, attr_name: str, evidence: str = "") -> None:
        """HOW_MUCH 속성 달성 표시"""
        if not self._active_schema:
            return
        for a in self._active_schema.attributes:
            if a.name == attr_name:
                a.achieved = True
                a.evidence = evidence
                print(f"  ✅ [Schema] 속성 달성: '{attr_name}'")
                break

    def check_schema_completion(self) -> bool:
        """모든 HOW_MUCH 속성이 달성됐는지 확인"""
        if not self._active_schema:
            return False
        return self._active_schema.all_attributes_achieved

    # ─────────────────────────────────────────────────────────────────────────

    def _get_top_goal(self) -> str:
        try:
            import json as _j, os as _o
            if _o.path.exists("eidos_settings.json"):
                with open("eidos_settings.json", "r", encoding="utf-8") as f:
                    return _j.load(f).get("top_goal", "")
        except Exception:
            pass
        return ""

    # 장기 top_goal(예: "크몽 CRM 서비스 상품 등록 완료")을 무관한 애드혹 태스크
    # (나무위키 요약 등)에 그대로 붙여 LLM 맥락을 오염시키던 문제를 막기 위한 게이트.
    # 도메인 키워드가 task_prompt에 나타나지 않으면 top_goal을 반환하지 않는다.
    _GOAL_DOMAIN_KEYWORDS = (
        "크몽", "kmong", "gumroad", "쿠팡", "coupang", "숨고", "이마트",
        "유튜브", "youtube", "인스타", "instagram", "블로그", "blog",
        "옥션", "11번가", "스마트스토어", "오픈마켓",
        "판매", "수익", "매출", "상품", "등록", "리뷰", "포트폴리오", "CRM",
    )
    _GOAL_SKIP_PHRASES = (
        "나무위키", "namu.wiki", "namu wiki",
        "위키피디아", "wikipedia",
        "요약", "정리해", "설명해", "알려줘", "찾아줘",
    )

    def _filter_goal(self, goal: str, task_prompt: str) -> str:
        """명시적으로 전달된 goal에 대해 task_prompt 관련성 게이트를 적용한다.

        GUI goal_bar나 task["goal_ref"]로 들어온 goal이 현재 task와 무관할 때
        (예: goal='크몽 CRM 등록 완료', prompt='나무위키 튜링 테스트 요약')
        LLM 컨텍스트 오염을 막기 위해 빈 문자열 반환.
        """
        if not goal:
            return ""
        _tp = (goal or "").lower() if False else (task_prompt or "").lower()
        _goal_l = goal.lower()
        if any(p.lower() in _tp for p in self._GOAL_SKIP_PHRASES):
            return ""
        _goal_keys = [k for k in self._GOAL_DOMAIN_KEYWORDS if k.lower() in _goal_l]
        if not _goal_keys:
            return goal
        if any(k.lower() in _tp for k in _goal_keys):
            return goal
        return ""

    def _get_relevant_goal(self, task_prompt: str) -> str:
        """task_prompt와 연관성이 있을 때만 top_goal 반환 (오염 게이팅)."""
        goal = self._get_top_goal()
        return self._filter_goal(goal, task_prompt)

    # task_prompt / schema.where 에서 사용자가 명시한 대상 도메인을 뽑아낸다.
    # _think 에서 현재 URL과 비교해 불일치면 NAVIGATE 최우선 힌트를 주입.
    _TARGET_DOMAIN_MAP = {
        "나무위키": "namu.wiki",
        "namu wiki": "namu.wiki",
        "위키피디아": "wikipedia.org",
        "wikipedia": "wikipedia.org",
        "구글": "google.com",
        "깃허브": "github.com",
        "github": "github.com",
        "유튜브": "youtube.com",
        "youtube": "youtube.com",
        "크몽": "kmong.com",
        "kmong": "kmong.com",
        "네이버": "naver.com",
        # "다음" 은 "다음 단계", "다음으로" 같은 흐름 단어와 충돌하므로
        # domain 형태(daum.net / daum.com) 로만 매칭하도록 제거.
        # 사용자가 daum 포털을 지정하려면 "daum" 또는 URL 을 직접 써야 한다.
        "daum.net": "daum.net",
        "daum.com": "daum.net",
        "디시": "dcinside.com",
        "레딧": "reddit.com",
        "reddit": "reddit.com",
        "stackoverflow": "stackoverflow.com",
        "스택오버플로": "stackoverflow.com",
        "인스타그램": "instagram.com",
        "instagram": "instagram.com",
        "트위터": "twitter.com",
        "twitter": "twitter.com",
    }

    def _extract_target_domain(self, task_prompt: str, schema) -> Optional[str]:
        """task_prompt / schema.where 에서 사용자가 지정한 대상 도메인 추출."""
        _tp = (task_prompt or "").lower()
        # chain upstream 블록은 이전 stage 산출물(문장) 이므로 도메인 추출 대상 아님.
        # "▼ research.research_report" 내부의 일반 문장이 도메인으로 오인식되는 것 차단.
        _upstream_marker = "[이전 단계 산출물"
        if _upstream_marker in _tp:
            _tp = _tp.split(_upstream_marker, 1)[0]
        # 1. 한/영 별칭 매칭
        for k, v in self._TARGET_DOMAIN_MAP.items():
            if k in _tp:
                return v
        # 2. task_prompt에 URL/도메인 직접 포함
        _m = re.search(
            r'(?:https?://)?([a-z0-9][a-z0-9\-]*\.[a-z]{2,}(?:\.[a-z]{2,})?)',
            _tp
        )
        if _m:
            _host = _m.group(1)
            # 일반 단어 오인식 방지 (예: "test.md" 등) — 최소 TLD 체크
            if any(_host.endswith(t) for t in (
                ".com", ".net", ".org", ".io", ".co", ".kr", ".wiki", ".gg"
            )):
                return _host
        # 3. schema.where 에서 추출
        if schema is not None:
            _where = (getattr(schema, "where", "") or "").lower()
            if _where:
                for k, v in self._TARGET_DOMAIN_MAP.items():
                    if k in _where or v in _where:
                        return v
                _m2 = re.search(
                    r'(?:https?://)?([a-z0-9][a-z0-9\-]*\.[a-z]{2,}(?:\.[a-z]{2,})?)',
                    _where
                )
                if _m2:
                    _host = _m2.group(1)
                    if any(_host.endswith(t) for t in (
                        ".com", ".net", ".org", ".io", ".co", ".kr", ".wiki", ".gg"
                    )):
                        return _host
        return None

    def _get_situation_summary(self) -> str:
        """WorldModel/제어실에서 현재 상황 요약"""
        parts = []
        try:
            import json as _j, os as _o
            if _o.path.exists("schedule_planner_context.json"):
                with open("schedule_planner_context.json", "r", encoding="utf-8") as f:
                    ctx = _j.load(f)
                sit = ctx.get("situation_internal", "")
                if sit:
                    lines = [l for l in sit.split("\n") if l.strip()][-3:]
                    parts.append("내부: " + " / ".join(lines))
                ext = ctx.get("situation_external", "")
                if ext:
                    lines = [l for l in ext.split("\n") if l.strip()][-2:]
                    parts.append("외부: " + " / ".join(lines))
        except Exception:
            pass
        return "\n".join(parts) if parts else "정보 없음"

    def _resolve_vars(self, text: str, variables: Dict[str, str]) -> str:
        """텍스트에서 {변수명} 치환"""
        for k, v in variables.items():
            text = text.replace(f"{{{k}}}", v)
        return text

    def _record_completion(self, task_prompt: str, goal: str, plan: ActionPlan):
        """WorldModel에 완료 이벤트 기록"""
        try:
            if hasattr(self.core, "graph"):
                import uuid as _u
                node_id = f"action_done_{_u.uuid4().hex[:8]}"
                self.core.graph.add_node(
                    node_id,
                    type="action_complete",
                    label=f"[완료] {task_prompt[:60]}",
                    goal=goal[:50],
                    steps=len(plan.steps),
                    timestamp=time.time(),
                )
                print(f"  [ActionDispatcher] WorldModel 기록: {node_id}")
        except Exception as e:
            print(f"  [ActionDispatcher] WorldModel 기록 실패 (무시): {e}")

    async def _notify_telegram_auth(
        self,
        *,
        reason: str,
        url: str,
        fails: int,
        limit: int = 3,
        platform_hint: str = "",
    ) -> None:
        """
        로그인 필요 상황을 Telegram 으로 사용자에게 알림.

        [2026-04-20] 기존엔 `_report()` (GUI 채팅창)만 호출해서 사용자가 PC 앞에
        없으면 인지 불가. MissionChain register 단계가 크몽 로그인에서 막혔는데
        Telegram 알림이 안 오는 버그의 직접 원인. 모든 auth_required 경로에서 함께 호출.
        """
        try:
            from eidos_telegram_bot import get_bot as _get_bot
            bot = _get_bot()
            if not bot.is_configured():
                return
            # 플랫폼 이름 자동 추정 (kmong / naver / 등)
            plat = platform_hint or ""
            if not plat and url:
                try:
                    import urllib.parse as _up
                    netloc = (_up.urlparse(url).netloc or "").lower()
                    for _k, _label in (
                        ("kmong", "크몽"), ("naver", "네이버"),
                        ("kakao", "카카오"), ("coupang", "쿠팡"),
                        ("amazon", "Amazon"), ("instagram", "Instagram"),
                    ):
                        if _k in netloc:
                            plat = _label
                            break
                    if not plat and netloc:
                        plat = netloc
                except Exception:
                    pass
            plat_s = f" ({plat})" if plat else ""
            msg = (
                f"🔐 <b>로그인 필요{plat_s}</b>\n\n"
                f"사유: {reason}\n"
                f"현재 URL: <code>{(url or '(없음)')[:120]}</code>\n"
                f"시도: {fails}/{limit}\n\n"
                f"브라우저에서 직접 로그인한 뒤 EIDOS 에서 <b>⚡ 자율 실행</b> "
                f"버튼을 눌러주세요."
            )
            await bot.send_notification(msg, title="로그인 필요")
            print(f"  📨 [Telegram] 로그인 필요 알림 발송 ({plat or '?'})")
        except Exception as _te:
            print(f"  ⚠️ [Telegram] 로그인 필요 알림 발송 실패: {_te}")

    async def _notify_telegram_captcha(
        self,
        *,
        question: str,
        url: str = "",
        platform_hint: str = "",
    ) -> None:
        """
        CAPTCHA(자동화 방지) 감지 시 Telegram 으로 사용자에게 풀어달라고 알림.

        [2026-04-23] CAPTCHA 류 ASK_USER 는 사용자가 브라우저 앞에 직접 앉아 이미지/
        텍스트를 풀어야만 돌파 가능 — 로그인보다 더 즉각적으로 사람 개입이 필요한
        케이스. 기존엔 `_report()`(GUI 채팅창)만 호출돼 PC 앞에 없으면 무한 대기.

        [Phase B 2026-04-24] [✅ 풀었음 — 재개] 인라인 버튼 추가. 콜백은 dispatcher 의
        loop 에 run_now() 를 schedule → CaptchaRecheck 가 실제 캡챠 해제 여부 검증.
        """
        try:
            from eidos_telegram_bot import get_bot as _get_bot
            bot = _get_bot()
            if not bot.is_configured():
                return
            plat = platform_hint or ""
            if not plat and url:
                try:
                    import urllib.parse as _up
                    netloc = (_up.urlparse(url).netloc or "").lower()
                    for _k, _label in (
                        ("kmong", "크몽"), ("naver", "네이버"),
                        ("kakao", "카카오"), ("coupang", "쿠팡"),
                        ("amazon", "Amazon"), ("instagram", "Instagram"),
                        ("google", "Google"), ("youtube", "YouTube"),
                    ):
                        if _k in netloc:
                            plat = _label
                            break
                    if not plat and netloc:
                        plat = netloc
                except Exception:
                    pass
            plat_s = f" ({plat})" if plat else ""
            msg = (
                f"🧩 <b>캡챠 해결 필요{plat_s}</b>\n\n"
                f"질문: {(question or '(없음)')[:200]}\n"
                f"현재 URL: <code>{(url or '(없음)')[:120]}</code>\n\n"
                f"브라우저에서 캡챠를 직접 푼 뒤 아래 <b>✅ 풀었음 — 재개</b> 버튼을 눌러주세요.\n"
                f"<i>(GUI 의 ⚡ 자율 실행 버튼도 동일 효과)</i>"
            )

            # 콜백: telegram bot 스레드에서 호출 → dispatcher loop 에 run_now() schedule.
            import uuid as _uuid
            token = _uuid.uuid4().hex[:12]
            _self_ref = self  # closure 캡처

            def _on_captcha_solved() -> None:
                try:
                    _target_loop = None
                    try:
                        _lt = _self_ref._loop_task
                        if isinstance(_lt, asyncio.Task):
                            _target_loop = _lt.get_loop()
                    except Exception:
                        _target_loop = None
                    if _target_loop is None:
                        try:
                            _target_loop = asyncio.get_event_loop_policy().get_event_loop()
                        except Exception:
                            _target_loop = None
                    if _target_loop is not None and _target_loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            _self_ref.run_now(""), _target_loop
                        )
                        print("  ✅ [CaptchaSolved] dispatcher 루프에 run_now schedule 됨")
                    else:
                        # 루프 없음 — 다음 _dispatch_loop tick 에서 자연 재개.
                        # _wait_for_user 만 해제해 다음 60s tick 에서 진행되도록.
                        _self_ref._wait_for_user = False
                        print("  ⏭️ [CaptchaSolved] dispatcher loop 미가용 — _wait_for_user 만 해제")
                except Exception as _e_cb:
                    print(f"  ⚠️ [CaptchaSolved] schedule 실패 (무시): {_e_cb}")

            await bot.send_captcha_alert(
                msg, token=token, on_solved=_on_captcha_solved,
            )
            print(f"  📨 [Telegram] 캡챠 알림 발송 ({plat or '?'}) token={token[:6]}…")
        except Exception as _te:
            print(f"  ⚠️ [Telegram] 캡챠 알림 발송 실패: {_te}")

    def _gate_block_inc_and_maybe_pause(
        self, task: dict, fail_type: str, fail_reason: str,
    ) -> bool:
        """[2026-05-23] Synthesis Guard 게이트 차단 누적 카운터 + 임계 PAUSE.

        Why: 실질검증/자기자산/샘플링 게이트가 각각 차단해도 LLM 이 다음 WRITE 에서
            다른 게이트에 걸려 무한 반복 → 13회까지 누적 후 타임아웃 (2.png 보고).
            schema 단위 누적 카운터로 글로벌 cap 적용. 4회 (`_gate_block_limit`)
            도달 시 task PAUSE + 사용자 결정 위임.

        반환:
            True  — PAUSE 처리됨. 호출자는 LLM 재시도 메시지 대신 PAUSE 메시지 return.
            False — 아직 임계 미달. 호출자는 평소처럼 거부 메시지 return (LLM 재시도).

        호출 위치: _execute_write 내 게이트 차단 분기 (eidos_synthesis_guard 호출 후).
        """
        try:
            _sch = self._active_schema
            if _sch is None:
                return False
            _gbc = int(getattr(_sch, "_gate_block_count", 0)) + 1
            _sch._gate_block_count = _gbc
            print(
                f"  ⚠️ [Gate Counter] schema {getattr(_sch, 'chain_id', '?')[:8]} "
                f"누적 {_gbc}/{self._gate_block_limit} (fail_type={fail_type})"
            )
            if _gbc < self._gate_block_limit:
                return False
            # 임계 도달 — task PAUSE + 사용자 안내
            task["status"] = "WAITING_USER"
            task["__pause_reason__"] = "gate_block_4x"
            self._paused_task = task
            self._wait_for_user = True
            # 카운터 리셋 — 사용자가 다시 시도하면 새로 카운트.
            _sch._gate_block_count = 0
            _short_reason = (fail_reason or "")[:150]
            self._report(
                f"⚠️ **[게이트 {self._gate_block_limit}회 누적 — 작업 일시정지]**\n"
                f"같은 작업을 LLM 이 반복해서 시도 중이라 멈췄어요. "
                f"마지막 차단 사유:\n"
                f"  · {_short_reason}\n\n"
                f"**선택지** (채팅으로 자연스럽게 알려주세요):\n"
                f"  · 더 구체적인 자료/자산 정보 첨부 → 새 명령으로 다시 시도\n"
                f"  · 지금까지 결과만으로 마무리하라고 지시\n"
                f"  · 작업을 취소하라고 지시"
            )
            return True
        except Exception as _e_gbp:
            print(f"  ⚠️ [Gate Counter] inc_and_pause 실패 (graceful): {_e_gbp}")
            return False

    def _report(self, message: str):
        """채팅창으로 중간보고 전송 - Signal.emit()으로 메인 스레드 전달.

        [Fix F 2026-05-06] step 진행 류 메시지 (🧠 [Step N] / ✅ [Step N 완료] /
        🔍 [자동 탐색 Tier] / 🛑 [막힘 감지] / 🔄 [동일 URL 반복] 등) 는 메인 채팅창
        cb 로 보내지 않음. 콘솔 print + 텔레그램 (별도 채널) + 캐릭터 이벤트는
        유지. Why: 사용자 보고 — '내장브라우저 조사 과정이 메인채팅창에 step별로
        정신없이 찍혀나오는데 산만하다'. settings 토글로 ON 가능 (verbose_react_steps).
        """
        print(f"📢 [ActionDispatcher] {message[:100]}")

        # [Fix F] step 노이즈 판정 + verbose 토글
        _is_step_noise = _is_step_noise_message(message)
        _verbose = self._verbose_step_chat_enabled() if _is_step_noise else False
        _route_to_chat = (not _is_step_noise) or _verbose

        if _route_to_chat and self._report_cb:
            try:
                self._report_cb(message)
            except Exception as e:
                print(f"  [ActionDispatcher] report_cb 오류: {e}")
        elif _route_to_chat:
            # ── [폴백] report_cb 미연결 시 core._gui_ref → dispatcher_report 직접 emit ──
            try:
                gui = getattr(self.core, '_gui_ref', None)
                if gui:
                    worker = getattr(gui, 'eidos_worker', None)
                    if worker and hasattr(worker, 'dispatcher_report'):
                        worker.dispatcher_report.emit(message)
                        # 연결됐으니 이후엔 콜백 자동 등록
                        if not self._report_cb:
                            self._report_cb = worker.dispatcher_report.emit
            except Exception as _fe:
                print(f"  [ActionDispatcher] report 폴백 실패: {_fe}")
        # Core의 trigger_character_event — 노이즈 메시지는 캐릭터 표정도 산만하므로 같이 차단.
        if _route_to_chat:
            try:
                if hasattr(self.core, "trigger_character_event"):
                    self.core.trigger_character_event(
                        event_type="action_report",
                        text=message,
                        emotion="focus",
                    )
            except Exception:
                pass

    def _verbose_step_chat_enabled(self) -> bool:
        """settings.verbose_react_steps 캐시 조회 (default False).

        [Fix F 2026-05-06] step 메시지를 메인 채팅창에 보고 싶은 사용자 토글.
        파일 I/O 비용 회피 위해 1회 캐시.
        """
        if hasattr(self, "_cached_verbose_steps"):
            return bool(self._cached_verbose_steps)
        v = False
        try:
            from eidos_chat_gui import load_settings as _ls_vs
            v = bool(_ls_vs().get("verbose_react_steps", False))
        except Exception:
            v = False
        try:
            self._cached_verbose_steps = v
        except Exception:
            pass
        return v