# eidos_external_research.py
# [Wave2-A 2026-05-28] 자기 목표 종속 외부 검색 — 3 무료 소스 통합.
#
# 사용자 결정:
#   - 검색 소스: DuckDuckGo (일반) + arXiv (논문) + Wikipedia (백과) — 모두 무료·키 X
#   - 목표가 활성 상태일 때만 검색 (목표 없으면 검색 X — 노이즈 회피)
#   - 자동 라우팅: 학술 키워드 → arXiv 우선·도메인 개요 → Wikipedia·그 외 → DuckDuckGo
#   - 출처별 weight: 논문 1.0·백과 0.7·웹 0.5
#   - 본문 max 토큰 제한 (총 ~3000 토큰 → 후속 ToM 분해 비용 통제)
#
# 디자인 원칙:
#   - LLM 호출 0 (이 단계는 키워드 휴리스틱·자료 fetch 만)
#   - graceful — 한 소스 실패해도 다른 소스 계속
#   - 모든 외부 호출에 timeout (10초)·재시도 X
#   - 네트워크 자체 부재 시 모든 함수 빈 list 반환

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict, field
from typing import Optional


# ── 상수 ──────────────────────────────────────────────────────────────
_TIMEOUT_SEC = 10
_MAX_SUMMARY_CHARS = 2000   # 자료 1개당 본문 max
_HEADERS = {
    "User-Agent": "EIDOS-Research-Agent/1.0 (self-improvement cycle)",
}

# 출처별 신뢰도 weight (Layer 2 미정계수 학습 시 곱해짐)
SOURCE_WEIGHTS = {
    "arxiv":      1.00,   # 학술 논문 — 최고 신뢰
    "wikipedia":  0.70,   # 백과 개요 — 중간
    "duckduckgo": 0.50,   # 일반 웹 — 낮음
}


# ── 검색 결과 데이터 ──────────────────────────────────────────────────
@dataclass
class ResearchResult:
    """1 외부 자료. ToM 분해 (Wave2-B) 의 입력."""

    source: str = ""                  # arxiv / wikipedia / duckduckgo
    title: str = ""
    url: str = ""
    summary: str = ""                 # 핵심 본문 (max ~2000 char)
    keyword: str = ""                 # 이 자료 가져올 때 사용한 쿼리
    fetched_at: str = ""
    source_weight: float = 0.5        # SOURCE_WEIGHTS 에서 복사
    extras: dict = field(default_factory=dict)  # 출처별 메타 (저자·날짜 등)

    def serialize(self) -> dict:
        return asdict(self)


# ── 헬퍼 ──────────────────────────────────────────────────────────────
def _now() -> str:
    return _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _http_get(url: str, timeout: int = _TIMEOUT_SEC) -> Optional[bytes]:
    """HTTP GET. 실패 시 None — 예외 안 던짐."""
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as e:
        print(f"[research] HTTP GET 실패 ({url[:80]}): {e}")
        return None


def _truncate(text: str, max_chars: int = _MAX_SUMMARY_CHARS) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


# ── 키워드 도출 — 휴리스틱 ────────────────────────────────────────────
# 지표명 → 검색 키워드 매핑. LLM 호출 0 — 빠른 시작용.
# 사용자가 의도한 "도메인 초월" — 한 지표가 여러 도메인 키워드 매칭.
_KEYWORDS_BY_INDICATOR: dict[str, list[str]] = {
    "comp_research":    ["정보 수집 방법론", "research methodology", "탐색 전략"],
    "comp_planning":    ["기획 프레임워크", "strategic planning", "목표 분해"],
    "comp_development": ["소프트웨어 설계 원칙", "software engineering practices"],
    "comp_marketing":   ["마케팅 전략", "consumer behavior", "광고 효율"],
    "comp_comm":        ["효과적 의사소통", "communication style", "고객 응대"],
    "comp_operations":  ["운영 효율성", "operations management", "프로세스 최적화"],
    "emo_valence":      ["positive psychology", "감정 조절", "well-being"],
    "emo_arousal":      ["arousal regulation", "각성 조절", "스트레스 관리"],
    "emo_affection":    ["social bonding", "친밀 관계 형성", "rapport building"],
    "user_acceptance":  ["consent psychology", "수용성", "신뢰 형성"],
    "thread_resolve":   ["task completion", "follow through", "마무리 능력"],
    "action_value":     ["reinforcement learning", "decision making", "행동 강화"],
    "pattern_accuracy": ["prediction accuracy", "메타인지", "calibration"],
    # 추가 placeholder 차원
    "work_efficiency":  ["productivity science", "업무 효율", "deep work"],
    "productivity":     ["productivity habits", "생산성 향상", "time management"],
    "response_speed":   ["cognitive speed", "반응 속도", "rapid decision"],
    "meta_cognition":   ["metacognition", "메타인지 훈련", "self-awareness"],
    "reasoning":        ["reasoning ability", "추론 능력", "logical thinking"],
    "financial_mgmt":   ["financial management", "재무 관리", "자원 배분"],
    "resource_balance": ["resource allocation", "자원 균형", "energy management"],
}


def derive_keywords(indicator_name: str) -> list[str]:
    """지표명 → 검색 키워드 list. unknown 이면 지표명 그대로."""
    kws = _KEYWORDS_BY_INDICATOR.get(indicator_name)
    if kws:
        return kws
    # fallback — underscore 제거
    return [indicator_name.replace("_", " ")]


def _is_academic_query(keyword: str) -> bool:
    """학술 검색이 더 적합한 쿼리인지 (휴리스틱)."""
    academic_signals = [
        "psychology", "metacognition", "regulation", "methodology",
        "engineering", "behavior", "learning", "reasoning",
        "cognition", "심리학", "메타인지", "추론",
    ]
    kw_lower = keyword.lower()
    return any(sig in kw_lower for sig in academic_signals)


# ── 소스 1: arXiv ──────────────────────────────────────────────────────
def search_arxiv(query: str, max_results: int = 2) -> list[ResearchResult]:
    """arXiv 검색. XML 응답 → ResearchResult list. graceful."""
    out: list[ResearchResult] = []
    try:
        q = urllib.parse.quote(query)
        url = (f"http://export.arxiv.org/api/query?search_query=all:{q}"
               f"&start=0&max_results={max_results}")
        body = _http_get(url)
        if not body:
            return out
        text = body.decode("utf-8", errors="ignore")
        # 간단 XML 파싱 — entry 단위
        entries = re.findall(r"<entry>(.*?)</entry>", text, re.DOTALL)
        for ent in entries[:max_results]:
            title = _xml_inner(ent, "title")
            summary = _xml_inner(ent, "summary")
            link_match = re.search(r'<id>(.*?)</id>', ent, re.DOTALL)
            link = (link_match.group(1).strip() if link_match else "")
            authors = re.findall(r"<name>(.*?)</name>", ent)
            published = _xml_inner(ent, "published")
            out.append(ResearchResult(
                source="arxiv",
                title=re.sub(r"\s+", " ", title).strip(),
                url=link,
                summary=_truncate(re.sub(r"\s+", " ", summary).strip()),
                keyword=query,
                fetched_at=_now(),
                source_weight=SOURCE_WEIGHTS["arxiv"],
                extras={"authors": authors[:5], "published": published},
            ))
    except Exception as e:
        print(f"[research] arXiv 검색 실패 (graceful): {e}")
    return out


def _xml_inner(xml_text: str, tag: str) -> str:
    """간단 inner text 추출."""
    m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", xml_text, re.DOTALL)
    return m.group(1).strip() if m else ""


# ── 소스 2: Wikipedia ─────────────────────────────────────────────────
def search_wikipedia(query: str, lang: str = "ko") -> Optional[ResearchResult]:
    """Wikipedia REST API — page summary. lang=ko 또는 en. graceful.

    한국어 시도 후 없으면 영어 fallback.
    """
    for cur_lang in (lang, "en") if lang != "en" else ("en",):
        try:
            q = urllib.parse.quote(query.replace(" ", "_"))
            url = f"https://{cur_lang}.wikipedia.org/api/rest_v1/page/summary/{q}"
            body = _http_get(url)
            if not body:
                continue
            data = json.loads(body.decode("utf-8", errors="ignore"))
            if data.get("type") == "disambiguation":
                continue
            extract = data.get("extract", "")
            if not extract:
                continue
            return ResearchResult(
                source="wikipedia",
                title=data.get("title", query),
                url=(data.get("content_urls", {})
                          .get("desktop", {})
                          .get("page", "")),
                summary=_truncate(extract),
                keyword=query,
                fetched_at=_now(),
                source_weight=SOURCE_WEIGHTS["wikipedia"],
                extras={"lang": cur_lang, "description": data.get("description", "")},
            )
        except Exception as e:
            print(f"[research] Wikipedia ({cur_lang}) 실패: {e}")
            continue
    return None


# ── 소스 3: DuckDuckGo Instant Answer ─────────────────────────────────
def search_duckduckgo(query: str) -> list[ResearchResult]:
    """DuckDuckGo Instant Answer API. Abstract + RelatedTopics. graceful."""
    out: list[ResearchResult] = []
    try:
        q = urllib.parse.quote(query)
        url = (f"https://api.duckduckgo.com/?q={q}"
               f"&format=json&no_html=1&skip_disambig=1")
        body = _http_get(url)
        if not body:
            return out
        data = json.loads(body.decode("utf-8", errors="ignore"))
        # Abstract
        abstract = data.get("Abstract") or data.get("AbstractText", "")
        if abstract:
            out.append(ResearchResult(
                source="duckduckgo",
                title=data.get("Heading", query),
                url=data.get("AbstractURL", ""),
                summary=_truncate(abstract),
                keyword=query,
                fetched_at=_now(),
                source_weight=SOURCE_WEIGHTS["duckduckgo"],
                extras={"abstract_source": data.get("AbstractSource", "")},
            ))
        # RelatedTopics (최대 2개)
        related = data.get("RelatedTopics", []) or []
        for rt in related[:2]:
            if not isinstance(rt, dict):
                continue
            text = rt.get("Text", "")
            if not text or len(text) < 30:
                continue
            out.append(ResearchResult(
                source="duckduckgo",
                title=text[:60],
                url=rt.get("FirstURL", ""),
                summary=_truncate(text),
                keyword=query,
                fetched_at=_now(),
                source_weight=SOURCE_WEIGHTS["duckduckgo"] * 0.8,  # related 는 약간 ↓
                extras={"is_related": True},
            ))
    except Exception as e:
        print(f"[research] DuckDuckGo 실패 (graceful): {e}")
    return out


# ── 통합 검색 — 목표 종속 ──────────────────────────────────────────────
# ── [Phase 4 2026-05-30 이종 도메인] 교차 도메인 검색어 도출 ──────────────
# GAP A 해결: 기존 derive_keywords 는 약점 지표를 AI/인지 키워드로만 검색해서
# 생물/법/군사 같은 *이종 도메인* 자료가 파이프라인에 안 들어왔음. 유추전이의
# 핵심은 *멀리 떨어진* 도메인의 같은 *구조*를 빌려오는 것 → 의외의 도메인을 골라
# 그 도메인의 메커니즘 검색어를 생성. LLM 이 도메인 선택(주)·고정 도메인 fallback.
_FALLBACK_DOMAINS = (
    "생물학", "법학", "군사 전략", "경제학", "생태학", "물리학",
)

_CROSS_DOMAIN_PROMPT = """너는 유추전이 도메인 선택기다.
EIDOS(자율 AI 에이전트)의 약점/목표 지표가 주어진다. 이 문제에 *전이 가능한 통찰*을
줄, EIDOS 도메인(AI·비즈니스)과 *멀리 떨어진 의외의* 학문/실무 도메인 {n} 개와
각 도메인에서 검색할 핵심 *메커니즘/구조* 개념 1개씩 골라라.

[지표] {indicator} (관련 개념: {hint})

[원칙]
- AI·머신러닝·인지과학 도메인은 금지 (이미 가까움). 생물학·법학·군사·경제·
  생태·물리·역사 등 *먼* 도메인에서 같은 *구조*를 찾아라.
- query 는 그 도메인의 메커니즘/구조 검색어 (단순 지표명 번역 금지).
- 순수 JSON 만 출력 (해설 금지):
{{"domains": [{{"domain": "생물학", "query": "면역계 항원 탐지 신호 캐스케이드"}}]}}
"""


def _extract_json_obj(text: str) -> dict:
    """LLM 응답에서 첫 JSON object 추출. 실패 시 빈 dict."""
    s = str(text or "")
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}


def derive_cross_domain_queries(
    indicator: str,
    llm_callable=None,
    n_domains: int = 3,
) -> list[str]:
    """약점 지표 → 이종 도메인 검색어 list[str]. LLM 우선·고정 도메인 fallback.

    llm_callable: prompt str → response str (sync). None 이면 fallback 만.
    반환 예: ["생물학 면역계 항원 탐지 신호 캐스케이드", "법학 판례 추론 ...", ...]
    """
    indicator = (indicator or "").strip()
    if not indicator:
        return []
    n_domains = max(1, min(5, int(n_domains)))
    # 1) LLM 도메인 선택 (주)
    if llm_callable is not None:
        try:
            hint = ", ".join(derive_keywords(indicator)[:2])
            raw = llm_callable(_CROSS_DOMAIN_PROMPT.format(
                n=n_domains, indicator=indicator, hint=hint))
            data = _extract_json_obj(raw)
            out: list[str] = []
            for d in (data.get("domains") or [])[:n_domains]:
                if not isinstance(d, dict):
                    continue
                q = str(d.get("query") or "").strip()
                dom = str(d.get("domain") or "").strip()
                if q:
                    out.append((f"{dom} {q}").strip() if dom else q)
            if out:
                return out
        except Exception as e:
            print(f"[research] cross-domain LLM 실패 (graceful·fallback): {e}")
    # 2) fallback — 고정 이종 도메인 × 지표 핵심 개념
    concept = (derive_keywords(indicator) or [indicator])[0]
    return [f"{dom} {concept}" for dom in _FALLBACK_DOMAINS[:n_domains]]


def research_for_goal(
    goal,
    max_per_source: int = 2,
    max_total: int = 5,
    llm_callable=None,
    cross_domain: bool = True,
) -> list[ResearchResult]:
    """활성 목표 → 키워드 도출 → 3 소스 검색 → 가중 정렬.

    학술 키워드는 arXiv 우선·도메인 개요는 Wikipedia·그 외 DuckDuckGo.
    [Phase 4 2026-05-30] cross_domain=True 면 이종 도메인(생물/법/군사…) 검색어를
    *우선* 삽입 → 유추전이용 먼 도메인 자료 확보 (llm_callable 로 도메인 선택).
    반환: weight 내림차순 정렬·max_total 제한.
    """
    if not goal:
        return []
    try:
        indicator = getattr(goal, "target_indicator", "")
        if not indicator:
            return []

        base_keywords = derive_keywords(indicator)
        # [Phase 4] 이종 도메인 검색어를 *앞에* — 유추전이의 핵심 (먼 도메인 우선)
        cross_keywords: list[str] = []
        if cross_domain:
            try:
                cross_keywords = derive_cross_domain_queries(
                    indicator, llm_callable=llm_callable, n_domains=3)
            except Exception as _e_cd:
                print(f"[research] cross-domain 도출 실패 (graceful): {_e_cd}")
        keywords = cross_keywords + [k for k in base_keywords
                                     if k not in cross_keywords]
        if not keywords:
            return []
        # cross_domain 이면 더 많은 검색어 커버 (먼 도메인 자료 충분히)
        _n_search = 4 if (cross_domain and cross_keywords) else 2

        results: list[ResearchResult] = []

        # arXiv — cross_domain 이면 모두·아니면 학술 키워드만
        for kw in keywords[:_n_search]:
            if cross_domain or _is_academic_query(kw):
                results.extend(search_arxiv(kw, max_results=max_per_source))

        # Wikipedia — 첫 키워드만 (각 1회 호출 — Wikipedia summary 는 1 자료 / 쿼리)
        if keywords:
            wiki = search_wikipedia(keywords[0])
            if wiki:
                results.append(wiki)

        # DuckDuckGo — 모든 키워드 (instant answer)
        for kw in keywords[:_n_search]:
            results.extend(search_duckduckgo(kw))

        # 중복 URL 제거
        seen_urls: set = set()
        deduped: list[ResearchResult] = []
        for r in results:
            key = r.url or r.title
            if key in seen_urls:
                continue
            seen_urls.add(key)
            deduped.append(r)

        # weight 내림차순
        deduped.sort(key=lambda r: r.source_weight, reverse=True)
        return deduped[:max_total]
    except Exception as e:
        print(f"[research] research_for_goal 실패 (graceful): {e}")
        return []


# ── 자료 catalog 저장 (debug·history) ─────────────────────────────────
_RESEARCH_DIR = os.path.join("eidos_files", "agents", "research_cache")


def save_research_batch(
    goal_id: str,
    results: list[ResearchResult],
) -> Optional[str]:
    """1 사이클의 검색 결과 일괄 저장. 반환: 디렉터리 경로."""
    if not results:
        return None
    try:
        os.makedirs(_RESEARCH_DIR, exist_ok=True)
        ts = _now().replace(":", "-")
        path = os.path.join(_RESEARCH_DIR, f"{goal_id}_{ts}.json")
        payload = {
            "goal_id": goal_id,
            "fetched_at": ts,
            "results": [r.serialize() for r in results],
        }
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, indent=2))
        return path
    except Exception as e:
        print(f"[research] save_research_batch 실패 (graceful): {e}")
        return None


# ── 외부에서 쉽게 — 텔레그램 요약 ───────────────────────────────────────
def summarize_research_for_telegram(
    results: list[ResearchResult],
    goal_indicator: str = "",
) -> str:
    """검색 결과를 텔레그램 markdown 로."""
    if not results:
        return f"🔍 *외부 검색* — {goal_indicator}\n결과 없음 (네트워크·소스 모두 fail)"
    lines = [f"🔍 *외부 검색* — `{goal_indicator}`",
             f"총 {len(results)}건 (가중 정렬)"]
    for r in results[:5]:
        emoji = {"arxiv": "📄", "wikipedia": "📚",
                 "duckduckgo": "🌐"}.get(r.source, "📎")
        lines.append(
            f"\n{emoji} *{r.source}* (w={r.source_weight:.1f})\n"
            f"  {r.title[:80]}\n"
            f"  {r.summary[:200]}..."
        )
    return "\n".join(lines)
