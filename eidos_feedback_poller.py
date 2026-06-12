# eidos_feedback_poller.py
# [2026-05-27 Phase 11] 외부 신호 polling — actor.indicator 자동 갱신 채널.
#
# Phase 1~10 은 belief 입력 채널 = chat 메시지 1개. 자율 agent 는 외부 세계 신호를
# 사용자 매개 없이 받아야 함. Phase 11 은 background QTimer 가 등록된 FeedbackSource
# 들을 polling → 변화 감지 시 FeedbackSignal 발행 → 지정 stage.actor.indicators merge.
#
# 다음 tick LLM 이 그 indicator 를 보고 결정 → 진짜 외부 ↔ EIDOS 폐쇄 루프.
#
# 디자인 원칙 (Phase 1~10 동일):
#   1. LLM 호출 0 (Phase 11-A). polling 자체는 휴리스틱·summary 정도.
#   2. graceful — 어떤 source 실패해도 다른 source / tick 영향 X.
#   3. 모든 source 는 read-only — 외부에 *보내는* 건 별 channel (Phase 12+).
#   4. credentials 는 settings.json 또는 .env (이 모듈은 평문 안 받음).
#   5. per-source rate limit — min_interval_sec (default 300).
#   6. 3회 연속 fail → 자동 disabled (무한 폭주 회피).
#
# kind 별 구현 (Phase 11-A 1차):
#   url_monitor:        urllib (stdlib) — page content hash diff
#   webhook:            stub (외부 HTTP 서버 → 본 모듈의 apply_inbound_webhook 호출)
#   rss / email /
#   calendar / telegram: stub (kind 만 등록·실 polling 은 후속)
#
# 저장:
#   sources:  eidos_files/agents/feedback/sources.json
#   signals:  eidos_files/agents/feedback/signals.jsonl (append-only — history 패턴)

from __future__ import annotations

import asyncio
import datetime as _dt
import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional, Any


# ── 상수 ───────────────────────────────────────────────────────────────
_BASE_DIR = os.path.join("eidos_files", "agents", "feedback")
_SOURCES_FILE = "sources.json"
_SIGNALS_FILE = "signals.jsonl"
_VERSION = 1

# kind 종류 (Phase 11-A: url_monitor + webhook 실 구현·나머지 stub)
SOURCE_KINDS = [
    "url_monitor",      # urllib 로 HTTP GET → content hash diff
    "webhook",          # passive — 외부 호출자가 apply_inbound_webhook
    "rss",              # stub (feedparser 후속)
    "email",            # stub (imaplib 후속)
    "calendar",         # stub (Google Calendar MCP 후속)
    "telegram_inbound", # stub (텔레그램 양방향 후속)
]

# signal 종류
SIGNAL_KINDS = [
    "url_changed",
    "new_message",
    "calendar_event",
    "webhook_event",
    "rss_item",
    "no_change",       # polling 했지만 변화 없음 (signal 발행 X·내부 카운터만)
    "error",
]

# 안전 임계값 (모듈 상단·dogfood 후 조정)
DEFAULT_MIN_INTERVAL_SEC = 300.0       # 5분
DEFAULT_FAIL_DISABLE_THRESHOLD = 3     # 3회 연속 fail 시 enabled=False
DEFAULT_HTTP_TIMEOUT_SEC = 15
MAX_CONTENT_LEN_FOR_HASH = 200_000     # 200KB cap (메모리·CPU 보호)


# ── FeedbackSource ─────────────────────────────────────────────────────
@dataclass
class FeedbackSource:
    """1개 외부 신호 source. kind 별 config 가 polling 방법 결정.

    target_stage_id 가 있으면 signal apply 시 그 stage 의 actor 갱신·없으면
    signal 만 저장 (사용자가 어디 적용할지 결정 가능).
    """

    id: str
    kind: str                       # SOURCE_KINDS
    name: str                       # 사람이 읽는 이름 ("크몽 알림 페이지")
    config: dict = field(default_factory=dict)
    enabled: bool = True
    last_polled_at: str = ""
    last_hash: str = ""             # url_monitor: content sha·webhook: 마지막 payload sha
    last_signal_at: str = ""        # 마지막 *변화* 감지 시각
    target_stage_id: str = ""       # 갱신할 stage (없으면 signal 만 기록)
    target_actor: str = ""          # 갱신할 actor 이름 (state["actors"][name])
    target_indicator: str = "has_new_signal"  # 어느 indicator key
    fail_count: int = 0             # 3회 → auto disable
    min_interval_sec: float = DEFAULT_MIN_INTERVAL_SEC
    created_at: str = ""
    updated_at: str = ""

    def serialize(self) -> dict:
        return asdict(self)

    @classmethod
    def deserialize(cls, data: dict) -> "FeedbackSource":
        if not isinstance(data, dict):
            data = {}
        now = _now()
        k = str(data.get("kind") or "url_monitor")
        if k not in SOURCE_KINDS:
            k = "url_monitor"
        try:
            min_iv = float(data.get("min_interval_sec", DEFAULT_MIN_INTERVAL_SEC))
        except Exception:
            min_iv = DEFAULT_MIN_INTERVAL_SEC
        try:
            fc = int(data.get("fail_count", 0))
        except Exception:
            fc = 0
        return cls(
            id=str(data.get("id") or _new_id("src")),
            kind=k,
            name=str(data.get("name") or "(이름 없음)"),
            config=dict(data.get("config") or {}),
            enabled=bool(data.get("enabled", True)),
            last_polled_at=str(data.get("last_polled_at") or ""),
            last_hash=str(data.get("last_hash") or ""),
            last_signal_at=str(data.get("last_signal_at") or ""),
            target_stage_id=str(data.get("target_stage_id") or ""),
            target_actor=str(data.get("target_actor") or ""),
            target_indicator=str(data.get("target_indicator") or "has_new_signal"),
            fail_count=fc,
            min_interval_sec=min_iv,
            created_at=str(data.get("created_at") or now),
            updated_at=str(data.get("updated_at") or now),
        )


# ── FeedbackSignal ─────────────────────────────────────────────────────
@dataclass
class FeedbackSignal:
    """polling 1회 결과 1건. signals.jsonl 에 append."""

    source_id: str
    detected_at: str
    kind: str                       # SIGNAL_KINDS
    payload: dict = field(default_factory=dict)
    summary: str = ""               # 짧은 요약 (240자 cap)
    applied_stage_id: str = ""      # 어느 stage 에 apply 됐는지 (apply 후 채움)
    applied_actor: str = ""

    def serialize(self) -> dict:
        return asdict(self)


# ── 헬퍼 ───────────────────────────────────────────────────────────────
def _now() -> str:
    return _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _parse_iso(ts: str) -> Optional[_dt.datetime]:
    if not ts:
        return None
    try:
        s = ts.replace("Z", "+00:00")
        return _dt.datetime.fromisoformat(s).replace(tzinfo=None)
    except Exception:
        return None


def _new_id(prefix: str = "src") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _ensure_base() -> None:
    try:
        os.makedirs(_BASE_DIR, exist_ok=True)
    except Exception as e:
        print(f"[feedback_poller] _ensure_base 실패 (graceful): {e}")


def _atomic_write(path: str, content: str) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)
        if os.path.exists(path):
            try:
                os.replace(tmp, path)
            except Exception:
                try:
                    os.remove(path)
                except Exception:
                    pass
                os.rename(tmp, path)
        else:
            os.rename(tmp, path)
        return True
    except Exception as e:
        print(f"[feedback_poller] _atomic_write 실패 (graceful): {path} — {e}")
        return False


def _sources_path() -> str:
    return os.path.join(_BASE_DIR, _SOURCES_FILE)


def _signals_path() -> str:
    return os.path.join(_BASE_DIR, _SIGNALS_FILE)


def _hash_content(content: str) -> str:
    if not content:
        return ""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


# ── sources CRUD ───────────────────────────────────────────────────────
def load_sources() -> list[FeedbackSource]:
    """모든 sources 로드. 손상되면 빈 list."""
    path = _sources_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        out: list[FeedbackSource] = []
        for raw in data:
            try:
                out.append(FeedbackSource.deserialize(raw))
            except Exception as e:
                print(f"[feedback_poller] source 1건 손상 skip (graceful): {e}")
                continue
        return out
    except Exception as e:
        print(f"[feedback_poller] load_sources 실패 (graceful): {e}")
        return []


def save_sources(sources: list[FeedbackSource]) -> bool:
    """전체 sources atomic 저장."""
    if sources is None:
        return False
    _ensure_base()
    try:
        payload = [s.serialize() for s in sources]
        return _atomic_write(_sources_path(),
                             json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"[feedback_poller] save_sources 실패 (graceful): {e}")
        return False


def register_source(
    kind: str,
    name: str,
    config: Optional[dict] = None,
    *,
    target_stage_id: str = "",
    target_actor: str = "",
    target_indicator: str = "has_new_signal",
    min_interval_sec: float = DEFAULT_MIN_INTERVAL_SEC,
    enabled: bool = True,
) -> Optional[FeedbackSource]:
    """새 source 등록. kind 가 SOURCE_KINDS 에 없으면 None 반환.

    동일 (kind, config) 조합이 이미 있으면 그 source 반환 (중복 회피).
    """
    if kind not in SOURCE_KINDS:
        print(f"[feedback_poller] register_source 거부 — 지원 안 함: {kind}")
        return None
    if not name or not name.strip():
        return None
    cfg = dict(config or {})
    sources = load_sources()
    # 중복 체크 — kind + config 의 key set 동일 + 핵심 필드 (url 등) 동일
    for s in sources:
        if s.kind != kind:
            continue
        if _config_equiv(s.config, cfg):
            return s
    now = _now()
    src = FeedbackSource(
        id=_new_id("src"),
        kind=kind,
        name=name.strip()[:80],
        config=cfg,
        enabled=enabled,
        target_stage_id=target_stage_id or "",
        target_actor=target_actor or "",
        target_indicator=target_indicator or "has_new_signal",
        min_interval_sec=max(60.0, float(min_interval_sec)),  # 최소 1분
        created_at=now,
        updated_at=now,
    )
    sources.append(src)
    save_sources(sources)
    return src


def _config_equiv(a: dict, b: dict) -> bool:
    """url_monitor: url 만 같으면 동일. webhook: secret 만 같으면 동일."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    for key in ("url", "secret", "feed_url", "calendar_id"):
        if a.get(key) and b.get(key):
            return str(a[key]) == str(b[key])
    return False


def update_source(source: FeedbackSource) -> bool:
    """1개 source in-place 갱신."""
    if source is None or not source.id:
        return False
    sources = load_sources()
    for i, s in enumerate(sources):
        if s.id == source.id:
            source.updated_at = _now()
            sources[i] = source
            return save_sources(sources)
    return False


def delete_source(source_id: str) -> bool:
    if not source_id:
        return False
    sources = load_sources()
    n0 = len(sources)
    sources = [s for s in sources if s.id != source_id]
    if len(sources) == n0:
        return False
    return save_sources(sources)


def get_source(source_id: str) -> Optional[FeedbackSource]:
    for s in load_sources():
        if s.id == source_id:
            return s
    return None


# ── signals append ─────────────────────────────────────────────────────
def append_signal(signal: FeedbackSignal) -> bool:
    """signals.jsonl 한 줄 추가."""
    if signal is None:
        return False
    _ensure_base()
    path = _signals_path()
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(signal.serialize(), ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        print(f"[feedback_poller] append_signal 실패 (graceful): {e}")
        return False


def read_signals(limit: Optional[int] = None) -> list[FeedbackSignal]:
    """signals.jsonl 전체 (또는 최근 limit 개) 시간 순."""
    path = _signals_path()
    out: list[FeedbackSignal] = []
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    out.append(FeedbackSignal(
                        source_id=str(raw.get("source_id", "")),
                        detected_at=str(raw.get("detected_at", "")),
                        kind=str(raw.get("kind", "")),
                        payload=dict(raw.get("payload") or {}),
                        summary=str(raw.get("summary", "")),
                        applied_stage_id=str(raw.get("applied_stage_id", "")),
                        applied_actor=str(raw.get("applied_actor", "")),
                    ))
                except Exception:
                    continue
        if limit and limit > 0 and len(out) > limit:
            out = out[-limit:]
        return out
    except Exception as e:
        print(f"[feedback_poller] read_signals 실패 (graceful): {e}")
        return out


# ── polling — kind 별 dispatcher ──────────────────────────────────────
async def poll_one_async(source: FeedbackSource,
                         now: Optional[_dt.datetime] = None
                         ) -> Optional[FeedbackSignal]:
    """1 source polling. 변화 있으면 FeedbackSignal·없으면 None.

    rate limit (min_interval_sec) 체크 후·kind 별 핸들러 dispatch.
    실패 시 fail_count++·3회 연속 fail 면 disabled=False 자동.
    """
    if source is None or not source.enabled:
        return None
    n = now or _dt.datetime.utcnow()
    # rate limit
    last = _parse_iso(source.last_polled_at)
    if last is not None:
        elapsed = (n - last).total_seconds()
        if elapsed < source.min_interval_sec:
            return None   # 아직 polling 간격 안 됨

    handler = _KIND_HANDLERS.get(source.kind)
    if handler is None:
        return None

    signal: Optional[FeedbackSignal] = None
    try:
        # handler 는 sync 도 async 도 OK — to_thread 로 sync 보호
        if asyncio.iscoroutinefunction(handler):
            signal = await handler(source, n)
        else:
            signal = await asyncio.to_thread(handler, source, n)
        # 성공 → fail_count 리셋
        source.fail_count = 0
    except Exception as e:
        source.fail_count += 1
        print(f"[feedback_poller] {source.name} polling 실패 ({source.fail_count}회·graceful): {e}")
        if source.fail_count >= DEFAULT_FAIL_DISABLE_THRESHOLD:
            source.enabled = False
            print(f"[feedback_poller] {source.name} — 연속 {source.fail_count} fail 로 자동 disabled")
        signal = FeedbackSignal(
            source_id=source.id,
            detected_at=n.isoformat(timespec="seconds") + "Z",
            kind="error",
            payload={"error": str(e)[:200]},
            summary=f"polling 실패: {str(e)[:120]}",
        )

    source.last_polled_at = n.isoformat(timespec="seconds") + "Z"
    if signal is not None and signal.kind not in ("no_change",):
        source.last_signal_at = signal.detected_at
        append_signal(signal)
    update_source(source)

    if signal is None or signal.kind == "no_change":
        return None
    return signal


async def poll_all_async(now: Optional[_dt.datetime] = None
                         ) -> list[FeedbackSignal]:
    """모든 enabled source 평가. rate limit·fail 자동 처리."""
    out: list[FeedbackSignal] = []
    sources = load_sources()
    n = now or _dt.datetime.utcnow()
    for s in sources:
        if not s.enabled:
            continue
        try:
            sig = await poll_one_async(s, now=n)
            if sig is not None:
                out.append(sig)
        except Exception as e:
            print(f"[feedback_poller] poll_all 안에서 source 1건 실패 (graceful): {e}")
            continue
    return out


# ── kind 별 핸들러 ────────────────────────────────────────────────────
def _poll_url_monitor(source: FeedbackSource,
                      now: _dt.datetime) -> Optional[FeedbackSignal]:
    """url_monitor — config={url, [user_agent], [check_substring]}.

    HTTP GET → content hash diff. 다르면 url_changed signal.
    check_substring 있으면 그 substring 이 있는 경우만 변경 인정 (옵션).
    """
    url = (source.config or {}).get("url", "")
    if not url:
        raise ValueError("config.url 비어있음")
    ua = (source.config or {}).get("user_agent", "EIDOS-FeedbackPoller/1.0")
    check_sub = (source.config or {}).get("check_substring", "")

    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": ua})
        with urllib.request.urlopen(req, timeout=DEFAULT_HTTP_TIMEOUT_SEC) as resp:
            raw_bytes = resp.read(MAX_CONTENT_LEN_FOR_HASH)
            charset = resp.headers.get_content_charset() or "utf-8"
            try:
                content = raw_bytes.decode(charset, errors="replace")
            except Exception:
                content = raw_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        # poll_one_async 의 try/except 가 fail_count 처리
        raise

    # check_substring — 페이지에 그 문자열 없으면 no_change
    if check_sub and check_sub not in content:
        return FeedbackSignal(
            source_id=source.id,
            detected_at=now.isoformat(timespec="seconds") + "Z",
            kind="no_change",
            payload={"reason": "check_substring 없음"},
        )

    new_hash = _hash_content(content)
    # 첫 polling — hash 만 기록, signal 없음
    if not source.last_hash:
        source.last_hash = new_hash
        return FeedbackSignal(
            source_id=source.id,
            detected_at=now.isoformat(timespec="seconds") + "Z",
            kind="no_change",
            payload={"reason": "첫 polling — baseline 기록"},
        )
    if new_hash == source.last_hash:
        return FeedbackSignal(
            source_id=source.id,
            detected_at=now.isoformat(timespec="seconds") + "Z",
            kind="no_change",
        )
    # 변경 감지
    source.last_hash = new_hash
    excerpt = _extract_text_excerpt(content)
    return FeedbackSignal(
        source_id=source.id,
        detected_at=now.isoformat(timespec="seconds") + "Z",
        kind="url_changed",
        payload={"url": url, "new_hash": new_hash,
                 "excerpt": excerpt[:600]},
        summary=f"URL 변경: {source.name} — {excerpt[:120]}",
    )


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _extract_text_excerpt(html: str) -> str:
    """HTML 에서 태그 제거·공백 정리한 텍스트 일부."""
    if not html:
        return ""
    text = _HTML_TAG_RE.sub(" ", html)
    text = _WS_RE.sub(" ", text).strip()
    return text[:1200]


def _poll_webhook(source: FeedbackSource,
                  now: _dt.datetime) -> Optional[FeedbackSignal]:
    """webhook — passive. polling 자체로는 signal 발행 X (외부 호출자가
    apply_inbound_webhook 로 push).

    여기서는 no_change 만 반환 (rate limit 효과로 last_polled_at 갱신용).
    """
    return FeedbackSignal(
        source_id=source.id,
        detected_at=now.isoformat(timespec="seconds") + "Z",
        kind="no_change",
        payload={"reason": "webhook 은 passive·외부 push 대기"},
    )


def _poll_stub(source: FeedbackSource,
               now: _dt.datetime) -> Optional[FeedbackSignal]:
    """rss/email/calendar/telegram_inbound — Phase 11-A 미구현. 항상 no_change."""
    return FeedbackSignal(
        source_id=source.id,
        detected_at=now.isoformat(timespec="seconds") + "Z",
        kind="no_change",
        payload={"reason": f"{source.kind} 핸들러 미구현 (Phase 11-B 후속)"},
    )


_KIND_HANDLERS: dict[str, Any] = {
    "url_monitor":       _poll_url_monitor,
    "webhook":           _poll_webhook,
    "rss":               _poll_stub,
    "email":             _poll_stub,
    "calendar":          _poll_stub,
    "telegram_inbound":  _poll_stub,
}


# ── webhook inbound (외부 HTTP 서버가 호출) ──────────────────────────
def apply_inbound_webhook(
    source_id: str,
    payload: dict,
    *,
    summary: str = "",
    now: Optional[_dt.datetime] = None,
) -> Optional[FeedbackSignal]:
    """[Phase 11-A] 외부 HTTP 서버가 webhook 수신 시 호출 — signal 발행 + apply.

    secret 검증·실제 HTTP 서버 wire 는 호출자 책임 (FastAPI/Flask 핸들러).
    이 함수는 source 가 webhook kind + enabled 인지·payload 등록만.
    """
    src = get_source(source_id)
    if src is None or src.kind != "webhook" or not src.enabled:
        return None
    n = now or _dt.datetime.utcnow()
    signal = FeedbackSignal(
        source_id=source_id,
        detected_at=n.isoformat(timespec="seconds") + "Z",
        kind="webhook_event",
        payload=dict(payload or {}),
        summary=summary[:240] if summary else f"webhook: {src.name}",
    )
    src.last_signal_at = signal.detected_at
    src.last_polled_at = signal.detected_at
    update_source(src)
    append_signal(signal)
    apply_signal_to_stage(signal)
    return signal


# ── signal → stage actor.indicator merge ──────────────────────────────
def apply_signal_to_stage(signal: FeedbackSignal) -> bool:
    """signal 의 source 가 지정한 stage/actor 의 indicator 갱신.

    indicator 키: target_indicator (default "has_new_signal") = True.
    추가 메타: "<key>_at" (detected_at), "<key>_summary" (signal.summary).

    source.target_stage_id 또는 target_actor 비어있으면 apply skip (signal 만 기록).
    Returns: apply 성공 여부.
    """
    if signal is None:
        return False
    if signal.kind in ("no_change", "error"):
        return False
    src = get_source(signal.source_id)
    if src is None or not src.target_stage_id or not src.target_actor:
        return False
    try:
        from eidos_agent_stage_store import update_actor_indicators
    except Exception as e:
        print(f"[feedback_poller] stage_store import 실패 (graceful): {e}")
        return False

    key = src.target_indicator or "has_new_signal"
    indicators = {
        key: True,
        f"{key}_at": signal.detected_at,
        f"{key}_summary": signal.summary[:200] if signal.summary else "",
        f"{key}_source": src.name[:80],
    }
    ok = update_actor_indicators(
        src.target_stage_id, src.target_actor, indicators,
        create_if_missing=True,
    )
    if ok:
        signal.applied_stage_id = src.target_stage_id
        signal.applied_actor = src.target_actor
        # signals.jsonl 마지막 record 는 이미 append 됨 — 추가 update 패스
    return ok


def apply_all_signals(signals: list[FeedbackSignal]) -> int:
    """다수 signal 일괄 apply. 성공 개수 반환."""
    if not signals:
        return 0
    n_ok = 0
    for sig in signals:
        try:
            if apply_signal_to_stage(sig):
                n_ok += 1
        except Exception as e:
            print(f"[feedback_poller] apply 1건 실패 skip (graceful): {e}")
            continue
    return n_ok


# ── poll + apply 한방 헬퍼 ────────────────────────────────────────────
async def poll_and_apply_all_async(
    now: Optional[_dt.datetime] = None,
) -> dict:
    """QTimer hook 에서 호출할 헬퍼 — poll_all + apply_all 한방.

    Returns: {"polled_sources": N, "signals": [...], "applied": M, "errors": [...]}
    """
    sources = load_sources()
    n_enabled = sum(1 for s in sources if s.enabled)
    signals = await poll_all_async(now=now)
    n_applied = apply_all_signals(signals)
    return {
        "polled_sources": n_enabled,
        "signals": signals,
        "n_signals": len(signals),
        "applied": n_applied,
    }


# ── 진단 helpers ──────────────────────────────────────────────────────
def summary_for_log() -> dict:
    sources = load_sources()
    by_kind: dict[str, int] = {}
    n_enabled = 0
    n_disabled = 0
    for s in sources:
        by_kind[s.kind] = by_kind.get(s.kind, 0) + 1
        if s.enabled:
            n_enabled += 1
        else:
            n_disabled += 1
    n_signals = 0
    try:
        n_signals = sum(1 for _ in open(_signals_path(), "r", encoding="utf-8"))
    except Exception:
        pass
    return {
        "total_sources": len(sources),
        "enabled": n_enabled,
        "disabled": n_disabled,
        "by_kind": by_kind,
        "total_signals": n_signals,
    }


__all__ = [
    "FeedbackSource", "FeedbackSignal",
    "SOURCE_KINDS", "SIGNAL_KINDS",
    "load_sources", "save_sources",
    "register_source", "update_source", "delete_source", "get_source",
    "append_signal", "read_signals",
    "poll_one_async", "poll_all_async",
    "apply_inbound_webhook", "apply_signal_to_stage", "apply_all_signals",
    "poll_and_apply_all_async",
    "summary_for_log",
]
