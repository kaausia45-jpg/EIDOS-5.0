# eidos_actor_belief.py
# [Wave9 2026-05-28] Multi-actor belief 추적.
#
# 기존 belief 는 사용자 (에스더 님) 본인 belief 만. Stage 의 actor 시스템
# (canvas 노드) 은 있지만 LLM 이 prompt 에 inject 하는 정도·메시지에서 actor
# 별 belief 변화 자동 추출 안 됨.
#
# 이 모듈은 그 gap 을 매움:
#   1. ActorBelief — 사용자 외 actor (client·competitor·colleague·...) 별 indicator
#   2. has_actor_keyword — 사전 필터·LLM 호출 절감
#   3. update_actors_from_message_async — LLM 으로 메시지에서 actor 별 변화 추출
#   4. as_prompt_brief — LLM prompt 에 inject 할 actor brief
#
# 저장: eidos_files/agents/stage_X/actor_belief.json (stage 별 격리)
#
# 비용: 메시지에 actor 키워드 있을 때만 LLM 1콜 호출 (~$0.0001).

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional


# ── ActorBelief dataclass ────────────────────────────────────────────
@dataclass
class ActorBelief:
    """사용자 외 entity (client·competitor·colleague·customer 등) 의 belief."""
    actor_id: str = ""              # 자동 생성·소문자_underscore 형식
    actor_name: str = ""            # 표시용 (예: "김 사장님")
    role: str = "other"             # client·competitor·colleague·customer·partner·other

    # 핵심 indicators — role 별 default 다름
    indicators: dict = field(default_factory=dict)
    # e.g. client: {urgency: 0.7, budget: 5000000, trust: 0.5, satisfaction: 0.6}

    # 메타
    first_seen_at: str = ""
    last_seen_at: str = ""
    mention_count: int = 0
    notes: str = ""                 # LLM 누적 자유 노트 (500자 cap)

    # 이 actor 와의 미완 사안
    pending_threads: list = field(default_factory=list)   # [str] 제목들

    def serialize(self) -> dict:
        return asdict(self)

    @classmethod
    def deserialize(cls, data: dict) -> "ActorBelief":
        if not isinstance(data, dict):
            data = {}
        out = cls()
        out.actor_id = str(data.get("actor_id", ""))
        out.actor_name = str(data.get("actor_name", ""))
        out.role = str(data.get("role", "other"))
        ind = data.get("indicators")
        out.indicators = dict(ind) if isinstance(ind, dict) else {}
        out.first_seen_at = str(data.get("first_seen_at", ""))
        out.last_seen_at = str(data.get("last_seen_at", ""))
        try:
            out.mention_count = int(data.get("mention_count", 0))
        except Exception:
            out.mention_count = 0
        out.notes = str(data.get("notes", ""))[:500]
        pt = data.get("pending_threads")
        out.pending_threads = list(pt) if isinstance(pt, list) else []
        return out


@dataclass
class ActorBeliefStore:
    """한 stage 의 모든 actor belief."""
    actors: dict = field(default_factory=dict)   # {actor_id: ActorBelief}

    def list_actors(self, role: Optional[str] = None) -> list:
        items = list(self.actors.values())
        if role:
            items = [a for a in items if a.role == role]
        return items

    def serialize(self) -> dict:
        return {"actors": {k: v.serialize() for k, v in self.actors.items()}}


# ── 저장·로드 ────────────────────────────────────────────────────────
def _store_path(stage_id: str) -> str:
    base = os.path.join("eidos_files", "agents", stage_id or "_global")
    try:
        os.makedirs(base, exist_ok=True)
    except Exception:
        pass
    return os.path.join(base, "actor_belief.json")


def load_actor_store(stage_id: str = "") -> ActorBeliefStore:
    path = _store_path(stage_id)
    if not os.path.exists(path):
        return ActorBeliefStore()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        store = ActorBeliefStore()
        actors_raw = data.get("actors") or {}
        if isinstance(actors_raw, dict):
            for aid, a_data in actors_raw.items():
                ab = ActorBelief.deserialize(a_data)
                if ab.actor_id:
                    store.actors[ab.actor_id] = ab
        return store
    except Exception as e:
        print(f"[actor_belief] load 실패 (graceful·빈 store): {e}")
        return ActorBeliefStore()


def save_actor_store(store: ActorBeliefStore, stage_id: str = "") -> bool:
    path = _store_path(stage_id)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(store.serialize(), f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"[actor_belief] save 실패 (graceful): {e}")
        return False


# ── 사전 필터 ────────────────────────────────────────────────────────
# 메시지에 actor 가능성 단어 있는지·LLM 호출 회피용. False positive 어느 정도 OK.
_ACTOR_KEYWORDS = (
    # 역할
    "의뢰자", "의뢰", "client", "고객", "customer",
    "경쟁사", "competitor", "동료", "colleague",
    "파트너", "partner", "투자자", "investor",
    "직원", "employee", "사장", "대표", "팀장",
    "선배", "후배", "거래처", "공급사", "vendor",
    # 호칭 + 사람 가능성
    "님이", "님은", "님께", "님한테", "님이랑", "님이랑은",
    "씨가", "씨는", "씨께", "씨한테",
    "그분", "그쪽", "저쪽",
    # 대화 동사 (3인칭)
    "말씀하셨", "답장", "회신", "연락 왔", "메시지",
    "물어봤", "물어보네", "협상", "견적",
)


def has_actor_keyword(text: str) -> bool:
    """메시지에 actor 추적 가능성 단어 있나·LLM 호출 사전 필터."""
    if not text or not text.strip():
        return False
    t = text.lower() if not any(ord(c) > 127 for c in text) else text
    for kw in _ACTOR_KEYWORDS:
        if kw in t:
            return True
    return False


# ── helper: actor_id 생성 ────────────────────────────────────────────
def _slugify(name: str, role: str = "") -> str:
    """actor 이름·역할 → 짧은 영문 id (한글이면 hash 사용)."""
    if not name:
        import uuid
        return f"actor_{uuid.uuid4().hex[:6]}"
    # 한글·특수문자만 있으면 hash 기반
    import re
    safe = re.sub(r"[^a-zA-Z0-9가-힣_]", "_", name.lower())[:30]
    if not safe or not any(c.isalnum() for c in safe):
        import hashlib
        safe = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
    return f"{role or 'other'}_{safe}"[:50]


# ── 메시지 → actor 변화 LLM 추출 ─────────────────────────────────────
_EXTRACT_SYSTEM = """\
너는 EIDOS 의 multi-actor belief tracker 다.

사용자 메시지에서 사용자 자신 외 actor (client·customer·competitor·colleague·
partner·investor·기타 entity) 가 언급됐는지 식별하고, 그 actor 의 belief
변화 정보를 추출해라.

원칙:
- actor_name: 사용자가 부른 이름 그대로 ("김 사장님"·"크몽 의뢰자 박씨")
- role: client·competitor·colleague·customer·partner·investor·other 중 하나
- indicators: 메시지에서 추론 가능한 정보만. 없으면 빈 dict.
  - 가능한 indicator (역할별 default·예시):
    * client: urgency 0~1·budget (원, 숫자)·trust 0~1·satisfaction 0~1
    * competitor: aggressiveness 0~1·market_share 0~1·threat_level 0~1
    * colleague: collaboration 0~1·reliability 0~1·workload 0~1
- notes: 1~2 문장 메모 (이 메시지에서 새로 알게 된 것)
- 사용자 자신 (에스더 님) 은 actor 아님·반환 X
- actor 없으면 빈 list 반환

JSON 만 출력:
{
  "actors_mentioned": [
    {
      "actor_name": "...",
      "role": "...",
      "indicators_inferred": {"trust": 0.7, ...},
      "notes": "..."
    },
    ...
  ]
}
"""


async def update_actors_from_message_async(
    store: ActorBeliefStore,
    user_text: str,
    eidos_context: str = "",
    timeout_sec: float = 10.0,
) -> tuple[ActorBeliefStore, list]:
    """메시지에서 actor 추출·기존 belief 업데이트·신규 actor 추가.

    Returns: (updated_store, changed_actor_ids_list).
    LLM 호출 전 has_actor_keyword 로 사전 필터. 키워드 없으면 즉시 (store, [])
    반환.
    """
    if not user_text or not user_text.strip():
        return store, []
    if not has_actor_keyword(user_text):
        return store, []

    try:
        from llm_module import get_llm_response_async
    except Exception:
        return store, []

    prompt = (
        f"[사용자 메시지]\n{user_text[:600]}\n\n"
        + (f"[EIDOS 컨텍스트]\n{eidos_context[:300]}\n\n" if eidos_context else "")
        + "위 메시지에서 사용자 외 actor 추출. JSON 만 출력."
    )

    raw = ""
    try:
        raw = await asyncio.wait_for(
            get_llm_response_async(
                prompt,
                max_tokens=1024,
                system_prompt=_EXTRACT_SYSTEM,
                response_mime_type="application/json",
            ),
            timeout=timeout_sec,
        )
    except asyncio.TimeoutError:
        return store, []
    except Exception as e:
        print(f"[actor_belief] LLM 실패 (graceful): {e}")
        return store, []

    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return store, []
    except Exception:
        return store, []

    mentioned = data.get("actors_mentioned") or []
    if not isinstance(mentioned, list):
        return store, []

    changed_ids: list = []
    now_iso = _dt.datetime.now().isoformat(timespec="seconds")

    for m in mentioned[:5]:
        if not isinstance(m, dict):
            continue
        actor_name = str(m.get("actor_name", "")).strip()[:60]
        if not actor_name:
            continue
        role = str(m.get("role", "other")).lower().strip()
        if role not in ("client", "competitor", "colleague", "customer",
                        "partner", "investor", "other"):
            role = "other"

        actor_id = _slugify(actor_name, role)

        # 기존 actor 찾기 — actor_id 매칭·또는 이름 substring
        existing = store.actors.get(actor_id)
        if existing is None:
            # name 기반 fuzzy match
            for aid, ab in store.actors.items():
                if (ab.actor_name == actor_name
                        or actor_name in ab.actor_name
                        or ab.actor_name in actor_name):
                    existing = ab
                    actor_id = aid
                    break

        if existing is None:
            ab = ActorBelief(
                actor_id=actor_id,
                actor_name=actor_name,
                role=role,
                first_seen_at=now_iso,
            )
            store.actors[actor_id] = ab
            existing = ab

        # 갱신
        existing.last_seen_at = now_iso
        existing.mention_count += 1

        # indicator merge (LLM 신규 값 우선·기존 보존)
        ind_new = m.get("indicators_inferred")
        if isinstance(ind_new, dict):
            for k, v in ind_new.items():
                try:
                    # numeric 만 받음·string indicator 는 skip (안전)
                    if isinstance(v, (int, float)):
                        existing.indicators[str(k)[:30]] = float(v)
                    elif isinstance(v, str) and v.replace(".", "").replace("-", "").isdigit():
                        existing.indicators[str(k)[:30]] = float(v)
                except Exception:
                    continue

        # notes append (500자 cap)
        new_note = str(m.get("notes", "")).strip()[:200]
        if new_note:
            if existing.notes:
                existing.notes = (
                    existing.notes + " | " + new_note
                )[:500]
            else:
                existing.notes = new_note

        changed_ids.append(actor_id)

    return store, changed_ids


# ── LLM prompt brief ─────────────────────────────────────────────────
def as_prompt_brief(
    store: ActorBeliefStore,
    max_actors: int = 3,
    min_mention_count: int = 1,
) -> str:
    """top N actor 의 brief — LLM prompt 에 inject.

    최근 언급 + mention_count 높은 순으로 정렬.
    """
    if not store or not store.actors:
        return ""

    actors = list(store.actors.values())
    # 정렬: last_seen_at 내림차순·동률시 mention_count
    actors.sort(
        key=lambda a: (a.last_seen_at, a.mention_count),
        reverse=True,
    )

    top = [a for a in actors if a.mention_count >= min_mention_count][:max_actors]
    if not top:
        return ""

    lines = ["[multi-actor belief — 사용자 외 entity 상태]"]
    for a in top:
        ind_str = ""
        if a.indicators:
            kv = ", ".join(
                f"{k}={v:.2f}" if isinstance(v, float) and abs(v) <= 1.0
                else f"{k}={v}"
                for k, v in list(a.indicators.items())[:4]
            )
            ind_str = f" {{{kv}}}"
        note_str = f" · {a.notes[:80]}" if a.notes else ""
        lines.append(
            f"  - [{a.role}] {a.actor_name} (언급 {a.mention_count}회){ind_str}{note_str}"
        )
    return "\n".join(lines)


# ── 편의: 한 번에 처리 (chat hook 에서 호출) ────────────────────────
async def process_message_for_actors_async(
    user_text: str,
    stage_id: str = "",
    eidos_context: str = "",
) -> dict:
    """chat hook 에서 한 번에 호출하는 편의 함수.

    Returns: {"changed_actor_ids": [...], "n_actors_now": int}.
    LLM 호출 안 됐으면 빈 결과.
    """
    if not has_actor_keyword(user_text):
        return {"changed_actor_ids": [], "n_actors_now": 0}
    store = load_actor_store(stage_id)
    store, changed = await update_actors_from_message_async(
        store, user_text, eidos_context,
    )
    if changed:
        save_actor_store(store, stage_id)
    return {
        "changed_actor_ids": changed,
        "n_actors_now": len(store.actors),
    }
