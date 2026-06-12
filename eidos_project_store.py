# eidos_project_store.py
# 프로젝트 영속화 -eidos_files/projects.json 에 사용자 프로젝트 목록 저장/로드/CRUD.
# 첫 실행 시 8개 seed (사업/연구/콘텐츠/고객/재무/학습 + 신규 2개).
#
# 사용처: eidos_more_actions_dialog._ProjectsTab -UI 측 데이터 모델.

from __future__ import annotations

import datetime as _dt
import json
import os
import uuid
from typing import Optional

# ── 영속화 경로 ────────────────────────────────────────────────────────
_BASE_DIR = os.path.join("eidos_files")
_FILE_PATH = os.path.join(_BASE_DIR, "projects.json")
_VERSION = 1


# ── 기본 schema ────────────────────────────────────────────────────────
def make_empty_project(title: str = "새 프로젝트") -> dict:
    """프로젝트 dict 기본 구조 -UI form 필드와 1:1 매핑."""
    now = _dt.datetime.now().isoformat(timespec="seconds")
    return {
        "id": f"proj_{uuid.uuid4().hex[:10]}",
        "icon": "📁",
        "title": title,
        "category": "기타",
        "status": "planned",
        "description": "",
        "deadline": "",                # YYYY-MM-DD 또는 빈 문자열
        "prompt": "",                  # 메인 채팅창에 보낼 실행 명령 (프롬프트)
        "prerequisites": "",           # 시작 전 준비조건
        "input_files": [],             # list[str] -파일 경로
        "deliverables": "",            # 납품물 설명
        "deliverable_files": [],       # list[str] -산출물 파일 경로
        "assembled_actions": [],       # list[str] -액션 조립 캔버스 (v2.x)
        "done": False,
        "created_at": now,
        "updated_at": now,
    }


# ── seed 프로젝트 (첫 실행 시 1회) ──────────────────────────────────────
_SEED_PROJECTS: list[dict] = [
    {
        "icon": "💼", "title": "사업 자동화 프로젝트", "category": "사업", "status": "ready",
        "description": "크몽 · 스마트스토어 · 블로그 통합 운영 자동화.",
        "deadline": "", "prerequisites": "본인 키 등록 + 로그인 세션 영속화 완료",
    },
    {
        "icon": "🔬", "title": "탐구주제 연구 시스템", "category": "연구", "status": "ready",
        "description": "관심 주제 자동 추적 → 자료 수집 → 심층 분석 → 보고서 정기 생성.",
        "deadline": "", "prerequisites": "Gemini API 키 등록",
    },
    {
        "icon": "🎬", "title": "콘텐츠 파이프라인", "category": "콘텐츠", "status": "beta",
        "description": "유튜브 · 블로그 · 인스타 통합 워크플로우. 주제 한 줄 → 3 플랫폼 동시 초안.",
        "deadline": "", "prerequisites": "네이버 블로그 / 인스타 로그인 세션",
    },
    {
        "icon": "💬", "title": "고객 응대 자동화", "category": "고객대응", "status": "planned",
        "description": "카톡 채널 · 메일 · 인스타 DM 통합 모니터링 + LLM 답변 초안.",
        "deadline": "", "prerequisites": "각 채널 로그인 + 알림 권한",
    },
    {
        "icon": "💰", "title": "재무 모니터링", "category": "관리", "status": "planned",
        "description": "월별 매출 · 비용 · 세금 자동 집계 + OCR + 이상치 알림.",
        "deadline": "", "prerequisites": "거래 명세 폴더 경로 + OCR 엔진",
    },
    {
        "icon": "📚", "title": "학습 트래킹", "category": "학습", "status": "planned",
        "description": "관심 분야 자료 자동 큐레이션 + 학습 로그 + 망각 곡선 복습 알림.",
        "deadline": "", "prerequisites": "관심 키워드 / RSS 소스 등록",
    },
]


def _seed() -> list[dict]:
    """기본 seed 프로젝트 -empty schema 위에 seed 값 overlay."""
    out: list[dict] = []
    for s in _SEED_PROJECTS:
        p = make_empty_project(s.get("title", "새 프로젝트"))
        for k, v in s.items():
            p[k] = v
        out.append(p)
    return out


# ── load / save (atomic) ───────────────────────────────────────────────
def _ensure_dir() -> None:
    try:
        os.makedirs(_BASE_DIR, exist_ok=True)
    except Exception as e:
        print(f"[project_store] _ensure_dir 실패 (graceful): {e}")


def load_projects() -> list[dict]:
    """디스크에서 프로젝트 목록 로드. 파일 없거나 손상 시 seed 반환 (저장은 안 함)."""
    try:
        if not os.path.exists(_FILE_PATH):
            return _seed()
        with open(_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return _seed()
        projects = data.get("projects")
        if not isinstance(projects, list):
            return _seed()
        # 각 항목 기본 필드 보강 (옛 파일 호환)
        normalized: list[dict] = []
        for p in projects:
            if not isinstance(p, dict):
                continue
            base = make_empty_project(p.get("title", "(이름 없음)"))
            for k, v in p.items():
                base[k] = v
            normalized.append(base)
        return normalized
    except Exception as e:
        print(f"[project_store] load 실패 -seed 반환 (graceful): {e}")
        return _seed()


def save_projects(projects: list[dict]) -> bool:
    """프로젝트 목록 디스크 저장 (atomic: tmp 파일 → rename)."""
    try:
        _ensure_dir()
        payload = {
            "version": _VERSION,
            "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "projects": projects,
        }
        tmp = _FILE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        # Windows 는 기존 파일 있으면 rename 실패 -remove 후 rename
        if os.path.exists(_FILE_PATH):
            try:
                os.replace(tmp, _FILE_PATH)
            except Exception:
                # 폴백: remove + rename
                try:
                    os.remove(_FILE_PATH)
                except Exception:
                    pass
                os.rename(tmp, _FILE_PATH)
        else:
            os.rename(tmp, _FILE_PATH)
        return True
    except Exception as e:
        print(f"[project_store] save 실패 (graceful): {e}")
        return False


# ── CRUD 헬퍼 (목록은 호출자가 들고 있는 list 를 직접 조작·디스크 save 는 별도 호출) ──

def create_project(projects: list[dict], title: str = "새 프로젝트") -> dict:
    """새 프로젝트 추가 (in-place) → 생성된 dict 반환."""
    p = make_empty_project(title)
    projects.append(p)
    return p


def delete_project(projects: list[dict], project_id: str) -> bool:
    """id 로 프로젝트 삭제 (in-place) → 성공 여부."""
    for i, p in enumerate(projects):
        if p.get("id") == project_id:
            del projects[i]
            return True
    return False


def update_project(projects: list[dict], project_id: str, **fields) -> Optional[dict]:
    """id 로 프로젝트 필드 갱신 (in-place·updated_at 자동) → 갱신된 dict 또는 None."""
    for p in projects:
        if p.get("id") == project_id:
            for k, v in fields.items():
                p[k] = v
            p["updated_at"] = _dt.datetime.now().isoformat(timespec="seconds")
            return p
    return None
