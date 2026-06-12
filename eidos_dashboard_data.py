# -*- coding: utf-8 -*-
"""대시보드 데이터 레이어 (Phase 1) — 순수 함수, Qt 의존 0, 읽기 전용.

Petalbot 의 reqspec_bridge / 프로젝트 폴더(meta.json)에서 대시보드 카드용 데이터를
모은다. UI(eidos_dashboard.py)와 분리해 헤드리스로 검증 가능하게 한다.

Phase 0 발견 반영:
  - 진짜 프로젝트는 `~/.petalbot/projects/<id>/meta.json` 폴더(11개)에 있고,
    app.memory(reqspec 레코드)는 얇음(1건) → *폴더를 1차 데이터원*으로.
  - 납품일은 대부분 None → '오늘' 카드는 마감일이 아니라 공수(시간) 기반 적재로 표시.
  - 승인/실행중/고객 카드는 라이브 시그널(Phase 2/4)이라 여기선 placeholder.

모든 함수는 실패해도 예외를 삼키고 빈/기본값을 반환(대시보드가 안 죽게).
"""
from __future__ import annotations

import os
import json
import time
from typing import Any, Dict, List, Optional


def _bridge():
    import petalbot.reqspec_bridge as rb  # type: ignore
    return rb


def _projects_root() -> str:
    try:
        from petalbot.config import PROJECTS_DIR  # type: ignore
        return str(PROJECTS_DIR)
    except Exception:
        return os.path.join(os.path.expanduser("~"), ".petalbot", "projects")


# ──────────────────────────────────────────────────────────────
# 오늘 / 할당 가능한 작업 (공수 기반) — collect_schedule_loads (LLM 0)
def get_today(*, auto_estimate: bool = False) -> Dict[str, Any]:
    try:
        sl = _bridge().collect_schedule_loads(auto_estimate=auto_estimate)
        return {"loads": sl.get("loads", []), "skipped": sl.get("skipped", []), "error": ""}
    except Exception as e:  # noqa: BLE001
        return {"loads": [], "skipped": [], "error": str(e)}


# ──────────────────────────────────────────────────────────────
# 진행중 프로젝트 — 폴더(meta.json) 1차 + app.memory 결과여부 보강
_OUTCOME_EXTS = (".pdf", ".html", ".md", ".pptx", ".docx", ".zip")
# 빌드/번들 산출물·캐시는 결과물이 아님 → 제외(노이즈 차단)
_NOISE_NAMES = {"base_library.zip"}
_NOISE_DIRS = {"__pycache__", "node_modules", "_internal", "dist", "build", ".git"}


def get_projects() -> List[Dict[str, Any]]:
    root = _projects_root()
    recs_by_title: Dict[str, dict] = {}
    try:
        for r in _bridge().list_projects():
            recs_by_title.setdefault((r.get("title") or "").strip(), r)
    except Exception:  # noqa: BLE001
        pass

    out: List[Dict[str, Any]] = []
    if not os.path.isdir(root):
        return out
    for pid in os.listdir(root):
        mp = os.path.join(root, pid, "meta.json")
        if not os.path.exists(mp):
            continue
        try:
            d = json.load(open(mp, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        title = (d.get("title") or pid).strip()
        rec = recs_by_title.get(title) or {}
        out.append({
            "id": pid,
            "title": title,
            "deadline_at": d.get("deadline_at") or rec.get("deadline_at"),
            "estimate_hours": d.get("estimate_hours"),
            "has_outcome": bool(rec.get("has_outcome")),
            "mtime": os.path.getmtime(mp),
            "project_type": rec.get("project_type") or d.get("project_type") or "",
        })
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


# ──────────────────────────────────────────────────────────────
# 프로젝트 쓰기 — 사용자 명시 조작(추가/이름변경/삭제). 개발 워크스페이스 탭 관리용.
# (이 모듈은 본래 읽기 전용이나, 프로젝트 폴더 CRUD 는 데이터원과 같은 곳이라 여기 둔다.)
def create_project(title: str) -> Dict[str, Any]:
    """새 프로젝트 폴더(meta.json)를 만든다. 반환 {id,title,folder} 또는 {error}."""
    title = (title or "").strip() or "새 프로젝트"
    root = _projects_root()
    try:
        import uuid
        os.makedirs(root, exist_ok=True)
        pid = uuid.uuid4().hex[:12]
        pdir = os.path.join(root, pid)
        os.makedirs(pdir, exist_ok=False)
        with open(os.path.join(pdir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"title": title}, f, ensure_ascii=False, indent=2)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    return {"id": pid, "title": title, "folder": pdir}


def rename_project(project_id: str, new_title: str) -> Dict[str, Any]:
    """프로젝트 meta.json 의 title 을 바꾼다. 반환 {id,title} 또는 {error}."""
    new_title = (new_title or "").strip()
    if not new_title:
        return {"error": "제목이 비어 있어요."}
    mp = os.path.join(_projects_root(), project_id, "meta.json")
    try:
        d = json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else {}
    except Exception:  # noqa: BLE001
        d = {}
    d["title"] = new_title
    try:
        os.makedirs(os.path.dirname(mp), exist_ok=True)
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    return {"id": project_id, "title": new_title}


def delete_project(project_id: str) -> Dict[str, Any]:
    """프로젝트 폴더를 통째로 삭제(되돌릴 수 없음). 반환 {ok:True} 또는 {error}."""
    pdir = os.path.join(_projects_root(), project_id)
    if not project_id or not os.path.isdir(pdir):
        return {"error": "폴더가 없어요."}
    try:
        import shutil
        shutil.rmtree(pdir)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    return {"ok": True}


# ──────────────────────────────────────────────────────────────
# 프로젝트 상세 — meta.json + 그 폴더의 산출물 + reqspec 레코드 보강 (마스터-디테일용)
def get_project_detail(project_id: str) -> Optional[Dict[str, Any]]:
    root = _projects_root()
    pdir = os.path.join(root, project_id)
    if not os.path.isdir(pdir):
        return None
    meta: Dict[str, Any] = {}
    mp = os.path.join(pdir, "meta.json")
    if os.path.exists(mp):
        try:
            meta = json.load(open(mp, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            meta = {}
    title = (meta.get("title") or project_id).strip()
    rec: Dict[str, Any] = {}
    try:
        for r in _bridge().list_projects():
            if (r.get("title") or "").strip() == title:
                rec = r
                break
    except Exception:  # noqa: BLE001
        pass
    # 예상 공수(시간): meta.estimate_hours(h) 우선, 없으면 reqspec estimate.hours(인일)×8
    hours = 0.0
    try:
        hours = float(meta.get("estimate_hours") or 0) or float((rec.get("estimate") or {}).get("hours", 0) or 0) * 8.0
    except Exception:  # noqa: BLE001
        hours = 0.0
    deliverables: List[Dict[str, Any]] = []
    for dirpath, dirs, files in os.walk(pdir):
        dirs[:] = [d for d in dirs if d not in _NOISE_DIRS]
        for fn in files:
            if fn == "meta.json" or fn in _NOISE_NAMES:
                continue
            if not fn.lower().endswith(_OUTCOME_EXTS):
                continue
            fp = os.path.join(dirpath, fn)
            try:
                deliverables.append({"name": fn, "path": fp,
                                     "kind": os.path.splitext(fn)[1].lstrip(".").lower(),
                                     "mtime": os.path.getmtime(fp)})
            except Exception:  # noqa: BLE001
                pass
    deliverables.sort(key=lambda d: d["mtime"], reverse=True)
    return {
        "id": project_id,
        "title": title,
        "project_type": rec.get("project_type") or meta.get("project_type") or "",
        "deadline_at": meta.get("deadline_at") or rec.get("deadline_at"),
        "hours": hours,
        "has_outcome": bool(rec.get("has_outcome")),
        "conversation": (meta.get("conversation") or "").strip()[:500],
        "deliverables": deliverables,
        "folder": pdir,
    }


# ──────────────────────────────────────────────────────────────
# 최근 산출물 — 프로젝트 폴더의 결과물 파일 스캔 (최신순)
def get_recent_outputs(limit: int = 8) -> List[Dict[str, Any]]:
    root = _projects_root()
    if not os.path.isdir(root):
        return []
    items: List[Dict[str, Any]] = []
    for pid in os.listdir(root):
        pdir = os.path.join(root, pid)
        if not os.path.isdir(pdir):
            continue
        title = pid
        mp = os.path.join(pdir, "meta.json")
        if os.path.exists(mp):
            try:
                title = (json.load(open(mp, encoding="utf-8")).get("title") or pid).strip()
            except Exception:  # noqa: BLE001
                pass
        for dirpath, dirs, files in os.walk(pdir):
            dirs[:] = [d for d in dirs if d not in _NOISE_DIRS]   # 빌드/캐시 폴더 가지치기
            for fn in files:
                if fn == "meta.json" or fn in _NOISE_NAMES:
                    continue
                if not fn.lower().endswith(_OUTCOME_EXTS):
                    continue
                fp = os.path.join(dirpath, fn)
                try:
                    items.append({"name": fn, "path": fp, "project": title,
                                  "mtime": os.path.getmtime(fp),
                                  "kind": os.path.splitext(fn)[1].lstrip(".").lower()})
                except Exception:  # noqa: BLE001
                    pass
    items.sort(key=lambda r: r["mtime"], reverse=True)
    return items[:limit]


# ──────────────────────────────────────────────────────────────
# KPI — 메모리 요약(빠름) + (옵션) Helix 강점지도(느림: 별도 프로세스 spawn)
def get_kpi(*, include_helix: bool = False) -> Dict[str, Any]:
    out: Dict[str, Any] = {"summary": "", "strong": [], "weak": [], "grounded": False, "error": ""}
    try:
        out["summary"] = _bridge().memory_summary()
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)
    if include_helix:
        try:
            hi = _bridge().helix_insights() or {}
            out["strong"] = hi.get("strong", []) or []
            out["weak"] = hi.get("weak", []) or []
            out["grounded"] = bool(hi.get("grounded"))
        except Exception as e:  # noqa: BLE001
            out["error"] = (out["error"] + " | helix:" + str(e)).strip(" |")
    return out


# ──────────────────────────────────────────────────────────────
# 승인 대기 — MissionChainRegistry(영속 JSON)에서 PENDING_APPROVAL 체인 조회 (read-only)
def get_pending_approvals() -> List[Dict[str, Any]]:
    try:
        from eidos_mission_chain import get_chain_registry, ChainStatus  # type: ignore
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    try:
        reg = get_chain_registry()
        for c in reg.all():
            if getattr(c, "status", "") != ChainStatus.PENDING_APPROVAL:
                continue
            stages = getattr(c, "stages", []) or []
            out.append({
                "chain_id": getattr(c, "chain_id", ""),
                "name": getattr(c, "name", "") or "(무제 체인)",
                "stages": len(stages),
                "created_at": getattr(c, "created_at", 0.0),
                "prompt": (getattr(c, "original_prompt", "") or "")[:80],
            })
    except Exception:  # noqa: BLE001
        return []
    out.sort(key=lambda d: d.get("created_at", 0), reverse=True)
    return out


# ──────────────────────────────────────────────────────────────
# 지금 실행 중 — registry 의 RUNNING/APPROVED 체인 (headless)
def get_running() -> List[Dict[str, Any]]:
    try:
        from eidos_mission_chain import get_chain_registry, ChainStatus  # type: ignore
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    try:
        reg = get_chain_registry()
        for c in reg.all():
            st = getattr(c, "status", "")
            if st in (ChainStatus.RUNNING, ChainStatus.APPROVED):
                stages = getattr(c, "stages", []) or []
                out.append({
                    "chain_id": getattr(c, "chain_id", ""),
                    "name": getattr(c, "name", "") or "(체인)",
                    "status": st,
                    "stage": int(getattr(c, "current_stage_index", 0) or 0),
                    "stages": len(stages),
                    "created_at": getattr(c, "created_at", 0.0),
                })
    except Exception:  # noqa: BLE001
        return []
    out.sort(key=lambda d: d.get("created_at", 0), reverse=True)   # 최신 먼저
    return out


# ──────────────────────────────────────────────────────────────
# 고객 메시지 — 크몽 인박스(customer_inbox/*.json) + 답변 초안(customer_drafts/*.md) 매칭
_CUST_INBOX = os.path.join("eidos_files", "customer_inbox")
_CUST_DRAFTS = os.path.join("eidos_files", "customer_drafts")


def get_customer_messages(limit: int = 8, *, inbox_dir: Optional[str] = None,
                          drafts_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    inbox_dir = inbox_dir or _CUST_INBOX
    drafts_dir = drafts_dir or _CUST_DRAFTS
    if not os.path.isdir(inbox_dir):
        return []
    drafts = os.listdir(drafts_dir) if os.path.isdir(drafts_dir) else []
    msgs: List[Dict[str, Any]] = []
    for fn in os.listdir(inbox_dir):
        if not fn.lower().endswith(".json"):
            continue
        fp = os.path.join(inbox_dir, fn)
        try:
            d = json.load(open(fp, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        mid = (d.get("message_id") or "").replace("kmong-", "")[:8]
        draft_path = ""
        if mid:
            for df in drafts:
                if mid in df:
                    draft_path = os.path.join(drafts_dir, df)
                    break
        msgs.append({
            "message_id": d.get("message_id", ""),
            "customer_id": d.get("customer_id", "") or "(고객)",
            "channel": d.get("channel", "kmong"),
            "message": (d.get("message", "") or "").strip(),
            "received_at": d.get("received_at", ""),
            "has_draft": bool(draft_path),
            "draft_path": draft_path,
            "inbox_path": fp,
        })
    msgs.sort(key=lambda m: m.get("received_at", ""), reverse=True)
    return msgs[:limit]


def _dir_stamp(path: str) -> tuple:
    """폴더 mtime + 개수 (지문용)."""
    if not os.path.isdir(path):
        return (0.0, 0)
    try:
        files = os.listdir(path)
        mt = max([os.path.getmtime(os.path.join(path, f)) for f in files], default=0.0)
        return (round(mt, 2), len(files))
    except Exception:  # noqa: BLE001
        return (0.0, 0)


# ──────────────────────────────────────────────────────────────
# 변경 감지 지문 — 싼 비용으로 "바뀌었나"만 판단. desync 차단:
#   Petalbot(별도 프로세스)이 프로젝트를 쓰거나 체인 상태가 바뀌면 지문이 달라짐 → 대시보드 갱신.
def data_fingerprint() -> tuple:
    parts: list = []
    # 1) MissionChain 레지스트리 파일 mtime
    try:
        from eidos_mission_chain import _CHAINS_FILE  # type: ignore
        if os.path.exists(_CHAINS_FILE):
            parts.append(("reg", round(os.path.getmtime(_CHAINS_FILE), 2)))
    except Exception:  # noqa: BLE001
        pass
    # 2) 프로젝트 폴더 meta.json 최신 mtime + 개수
    root = _projects_root()
    if os.path.isdir(root):
        mt = 0.0
        n = 0
        for pid in os.listdir(root):
            mp = os.path.join(root, pid, "meta.json")
            if os.path.exists(mp):
                n += 1
                try:
                    mt = max(mt, os.path.getmtime(mp))
                except Exception:  # noqa: BLE001
                    pass
        parts.append(("proj", round(mt, 2), n))
    # 3) 고객 인박스/초안 폴더 (새 메시지·초안 → 갱신)
    parts.append(("cust", _dir_stamp(_CUST_INBOX), _dir_stamp(_CUST_DRAFTS)))
    return tuple(parts)


# ──────────────────────────────────────────────────────────────
def build_todo() -> Dict[str, Any]:
    """오늘 할 일 — 활성 프로젝트 데이터를 직접 읽어 *구조화된* to-do 리스트 생성.

    LLM 0(결정적·항상 구조화). 막 풀어쓰지 않고 섹션→항목→하위행동 구조로 반환.
    섹션:
      1) 🔔 지금 결정 필요 — 승인 대기 체인
      2) 📂 프로젝트별 할 일 — 활성(미완료) 프로젝트마다 부족한 것(공수·납품일·작업·산출물), 납품일 임박순
      3) ⏱ 공수 미입력 — 견적 필요
    """
    data = gather()
    active = [p for p in data.get("projects", []) if not p.get("has_outcome")]
    sections: List[Dict[str, Any]] = []

    appr = data.get("approvals", [])
    if appr:
        sections.append({"title": "🔔 지금 결정 필요", "entries": [
            {"head": f"체인 '{a.get('name','')}' ({a.get('stages',0)}단계) — 승인 또는 취소"} for a in appr]})

    proj_entries: List[Dict[str, Any]] = []
    for p in active:
        det = get_project_detail(p.get("id", "")) or {}
        hrs = float(det.get("hours") or 0)
        dl = det.get("deadline_at")
        acts: List[str] = []
        if hrs <= 0:
            acts.append("예상 공수 산정")
        if not dl:
            acts.append("납품일 설정")
        if hrs > 0:
            acts.append(f"작업 진행 (남은 공수 {int(hrs)}h)")
        if not det.get("deliverables"):
            acts.append("산출물 생성 / 납품")
        if acts:
            head = (p.get("title") or "(무제)")[:32]
            proj_entries.append({"head": head, "deadline": dl, "subs": acts})
    proj_entries.sort(key=lambda g: (g.get("deadline") is None, g.get("deadline") or 1e18))
    for e in proj_entries:
        if e.get("deadline"):
            import datetime as _dt
            try:
                d = _dt.datetime.fromtimestamp(float(e["deadline"]))
                days = (d.date() - _dt.date.today()).days
                e["head"] = f"{e['head']}  ·  D{'-' if days >= 0 else '+'}{abs(days)}"
            except Exception:  # noqa: BLE001
                pass
    if proj_entries:
        sections.append({"title": "📂 프로젝트별 할 일", "entries": proj_entries})

    skipped = data.get("today", {}).get("skipped", [])
    if skipped:
        sections.append({"title": "⏱ 공수 미입력 — 견적 필요", "entries": [
            {"head": str(s.get("title", ""))[:32]} for s in skipped]})

    return {"active_count": len(active), "sections": sections}


def gather(*, include_helix: bool = False) -> Dict[str, Any]:
    """대시보드 1회 새로고침용 — 6카드 데이터 묶음.

    today/projects/outputs/kpi = 실데이터(read-only).
    approvals/running/customer = placeholder(라이브 시그널은 Phase 2/4).
    """
    today = get_today()
    projects = get_projects()
    open_projects = [p for p in projects if not p["has_outcome"]]
    return {
        "ts": time.time(),
        "today": today,                                  # {loads, skipped}
        "projects": projects,
        "open_count": len(open_projects),
        "outputs": get_recent_outputs(),
        "kpi": get_kpi(include_helix=include_helix),
        "approvals": get_pending_approvals(),   # Phase 2: MissionChain PENDING_APPROVAL (영속 레지스트리)
        "running": get_running(),               # Phase 3: RUNNING/APPROVED 체인
        # placeholders — 라이브 배선은 다음 페이즈
        "customer": get_customer_messages(),   # Phase 4: 크몽 인박스 + 답변 초안
    }


if __name__ == "__main__":
    import sys
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass
    data = gather(include_helix="--helix" in sys.argv)
    t = data["today"]
    print(f"오늘(적재 가능) {len(t['loads'])}건 · 공수없어 제외 {len(t['skipped'])}건")
    print(f"프로젝트 {len(data['projects'])}건 · 진행중 {data['open_count']}건")
    print(f"최근 산출물 {len(data['outputs'])}건")
    for o in data["outputs"][:5]:
        print(f"   - [{o['kind']}] {o['name']}  ({o['project'][:24]})")
    print(f"KPI: {data['kpi']['summary'][:80]}  강점 {len(data['kpi']['strong'])} / 약점 {len(data['kpi']['weak'])}")
