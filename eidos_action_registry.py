# eidos_action_registry.py
# [2026-05-26 Phase 0-A] 다중 행위자 ToM 에이전트 — Action 레지스트리.
#
# 두 namespace:
#   eidos.tool.*    실행 가능. legacy 12 verb (canvas runner) 와 1:1 매핑.
#   eidos.meta.*    실행 안 함. 메타/북킹용 (wait/observe/no_op/reflect/...)
#   abstract.*      라벨만. ToM 으로 다른 actor 의 next-action 예측에 사용.
#
# Action ID 는 hierarchical (dot-separated):
#   abstract.provoke
#   abstract.provoke.verbal
#   abstract.client.add_requirement
#   eidos.tool.web.navigate
#   eidos.tool.message.email
#
# wildcard:
#   "abstract.provoke.*"  → abstract.provoke.* 의 모든 자손
#   "abstract.*.deploy"   → 중간 segment wildcard 도 지원
#
# Action ID 규칙 (LLM 자동 생성 시 강제):
#   - 동의 predicate 는 같은 id ("도발하다" 는 누가 하든 abstract.provoke)
#   - 변종은 sub-segment 로 (abstract.provoke.verbal / abstract.provoke.cyber)
#   - applicable_to_types 로 어떤 actor type 이 사용 가능한지 제한

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Optional, Iterable


# ── ActionSpec ─────────────────────────────────────────────────────────
@dataclass
class ActionSpec:
    id: str                                # hierarchical "namespace.path.name"
    label_ko: str = ""                     # 사람 읽는 한국어 라벨
    description: str = ""
    applicable_to_types: list[str] = field(default_factory=list)
    executable: bool = False               # True 면 eidos.tool.* — executor 보유
    executor_hint: str = ""                # eidos.tool.* 의 legacy verb 이름 (NAVIGATE 등)
    tags: list[str] = field(default_factory=list)
    source: str = "default"                # "default" | "stage_local" | "llm_proposed" | "user"

    def serialize(self) -> dict:
        return asdict(self)

    @classmethod
    def deserialize(cls, data: dict) -> "ActionSpec":
        try:
            return cls(
                id=str(data.get("id", "")),
                label_ko=str(data.get("label_ko", "")),
                description=str(data.get("description", "")),
                applicable_to_types=list(data.get("applicable_to_types", []) or []),
                executable=bool(data.get("executable", False)),
                executor_hint=str(data.get("executor_hint", "")),
                tags=list(data.get("tags", []) or []),
                source=str(data.get("source", "default")),
            )
        except Exception:
            return cls(id="invalid")


# ── DEFAULT_GLOBALS — 시스템 디폴트 action 들 ─────────────────────────
# eidos.tool.* : 실행 가능. legacy canvas runner 의 12 verb 와 1:1.
# eidos.meta.* : 라벨만. agent runner 의 메타 책임 (wait/observe/...)
DEFAULT_GLOBALS: list[ActionSpec] = [
    # ── EIDOS 의 실행 가능 tool action (legacy 12 verb 매핑) ──
    ActionSpec(id="eidos.tool.web.navigate", label_ko="페이지 이동",
               description="URL 로 브라우저 이동", applicable_to_types=["self"],
               executable=True, executor_hint="NAVIGATE"),
    ActionSpec(id="eidos.tool.web.click", label_ko="클릭",
               description="CSS 셀렉터 또는 [N] 번호로 요소 클릭",
               applicable_to_types=["self"], executable=True, executor_hint="CLICK"),
    ActionSpec(id="eidos.tool.web.click_xy", label_ko="좌표 클릭",
               description="x,y 좌표 직접 클릭 (Vision)",
               applicable_to_types=["self"], executable=True, executor_hint="CLICK_XY"),
    ActionSpec(id="eidos.tool.web.fill", label_ko="입력",
               description="셀렉터에 텍스트 입력",
               applicable_to_types=["self"], executable=True, executor_hint="FILL"),
    ActionSpec(id="eidos.tool.web.scroll", label_ko="스크롤",
               description="페이지 스크롤 (bottom/top/px:N)",
               applicable_to_types=["self"], executable=True, executor_hint="SCROLL"),
    ActionSpec(id="eidos.tool.web.read_page", label_ko="페이지 읽기",
               description="페이지 본문을 변수에 저장",
               applicable_to_types=["self"], executable=True, executor_hint="READ_PAGE"),
    # [2026-06-01] SerpAPI 웹 검색 — google 직접 navigate(CAPTCHA) 대신 정식 검색 API.
    # 자료조사·시장조사 시 navigate 보다 이걸 우선 사용. args: {query}.
    ActionSpec(id="eidos.tool.web.search", label_ko="웹 검색 (SerpAPI)",
               description="검색어(query)로 SerpAPI google 검색 → 결과 요약. "
                           "구글 사이트 직접 접속은 CAPTCHA 로 막히므로 검색은 이걸 사용.",
               applicable_to_types=["self"], executable=True, executor_hint="WEB_SEARCH"),
    ActionSpec(id="eidos.tool.web.go_back", label_ko="뒤로가기",
               description="브라우저 history.back",
               applicable_to_types=["self"], executable=True, executor_hint="GO_BACK"),
    ActionSpec(id="eidos.tool.web.wait", label_ko="대기 (초)",
               description="N 초 대기",
               applicable_to_types=["self"], executable=True, executor_hint="WAIT"),
    ActionSpec(id="eidos.tool.llm.write", label_ko="LLM 작성",
               description="LLM 으로 텍스트 생성 후 파일 저장",
               applicable_to_types=["self"], executable=True, executor_hint="WRITE"),
    ActionSpec(id="eidos.tool.draft_to_clipboard", label_ko="답변 초안 클립보드 복사",
               description="LLM 으로 답변 초안 생성 → 파일 저장 (audit) + 시스템 클립보드 복사 → 사용자가 Ctrl+V 로 붙여넣기. 비가역 외부효과 X — 사용자 통제권 100%.",
               applicable_to_types=["self"], executable=True, executor_hint="DRAFT_CLIPBOARD"),
    ActionSpec(id="eidos.tool.message.telegram", label_ko="텔레그램 발송",
               description="본인 텔레그램 채널에 메시지 발송",
               applicable_to_types=["self"], executable=True, executor_hint="TELEGRAM_SEND"),
    ActionSpec(id="eidos.tool.message.email", label_ko="이메일 발송",
               description="SMTP 이메일 발송",
               applicable_to_types=["self"], executable=True, executor_hint="EMAIL_SEND"),
    ActionSpec(id="eidos.tool.ask_user", label_ko="사용자에게 묻기",
               description="HITL — 사용자 입력 요청",
               applicable_to_types=["self"], executable=True, executor_hint="ASK_USER"),

    # ── [2026-05-27 자율 개발 모드] dev.* — 4 action ──
    # 의뢰자의 코딩 의뢰를 EIDOS 가 자체적으로 작성·테스트·패키징까지 수행. 모든 파일 I/O 는
    # eidos_files/agents/{stage_id}/outputs/projects/ 안으로 path-confine. run_tests 만
    # subprocess 라 외부효과·텔레그램 PROMPT 게이트 (다른 3개는 path-confine inline).
    ActionSpec(id="eidos.tool.dev.scaffold_project", label_ko="프로젝트 생성",
               description=(
                   "outputs/projects/<name>/ 폴더 + 언어별 boilerplate (README·main·tests·.gitignore) 생성. "
                   "args.target=프로젝트명·args.language=python|node|js·args.framework 선택. "
                   "외부효과 X — path-confine 안 파일 생성만."
               ),
               applicable_to_types=["self"], executable=True, executor_hint="DEV_SCAFFOLD"),
    ActionSpec(id="eidos.tool.dev.write_code", label_ko="코드 작성",
               description=(
                   "LLM 으로 코드 생성 후 outputs/projects/<project>/<target> 에 저장. "
                   "args.target=상대 파일 경로·args.content=상세 사양·args.project=프로젝트명·"
                   "args.language=python|javascript|typescript|... default python. 외부효과 X."
               ),
               applicable_to_types=["self"], executable=True, executor_hint="DEV_WRITE_CODE"),
    ActionSpec(id="eidos.tool.dev.run_tests", label_ko="테스트 실행",
               description=(
                   "프로젝트 폴더에서 테스트 명령 subprocess 실행 (외부효과·텔레그램 사전승인 필요). "
                   "args.target=프로젝트명·args.test_cmd 선택 (없으면 pytest/npm test 자동). "
                   "timeout=120s. rc=0 이면 PASS."
               ),
               applicable_to_types=["self"], executable=True, executor_hint="DEV_RUN_TESTS"),
    ActionSpec(id="eidos.tool.dev.package_deliverable", label_ko="납품 패키징",
               description=(
                   "프로젝트 → zip 패키징 (__pycache__·.git·node_modules·.venv 제외). "
                   "args.target=프로젝트명·args.output 선택 (없으면 ts 자동). "
                   "결과는 outputs/<project>_<ts>.zip 에 저장. 외부효과 X."
               ),
               applicable_to_types=["self"], executable=True, executor_hint="DEV_PACKAGE"),

    # ── [2026-05-27 ToM↔PCA 통합] PCA DAG 실행 위임 ──
    # 사용자가 PCA 캔버스에서 사전 정의한 multi-step workflow (DAG) 를 실행.
    # 단순 1회 LLM (draft_to_clipboard) 보다 정교한 분기·tool 체인·loop 자동화 가능.
    ActionSpec(id="eidos.tool.pca.run_dag", label_ko="PCA DAG 실행",
               description=(
                   "Prompt Canvas 의 저장된 DAG (워크플로우) 를 실행. "
                   "ToM 이 자율 결정으로 정교한 multi-step 작업 위임할 때 사용. "
                   "args.dag_name = DAG 파일명 (eidos_files/agents/*.json, 확장자 자동). "
                   "args.input = start_input (첫 노드의 {{input}}). "
                   "args.variables = dict, AgentState 에 주입 (DAG 의 {{var:key}} 치환)."
               ),
               applicable_to_types=["self"], executable=True, executor_hint="PCA_RUN_DAG"),

    # ── EIDOS 의 meta action — 라벨만, 별도 runner 분기 ──
    ActionSpec(id="eidos.meta.wait", label_ko="기다리기 (메타)",
               description="이번 tick 은 행동하지 않고 대기·다음 관찰까지 휴식",
               applicable_to_types=["self"], executable=False),
    ActionSpec(id="eidos.meta.observe", label_ko="관찰만",
               description="외부 신호만 수집·내부 belief 갱신·외부효과 X",
               applicable_to_types=["self"], executable=False),
    ActionSpec(id="eidos.meta.reflect", label_ko="자기성찰",
               description="자기 상태/목표 진척도 재평가",
               applicable_to_types=["self"], executable=False),
    ActionSpec(id="eidos.meta.update_belief", label_ko="믿음 갱신",
               description="특정 actor 의 indicator 를 새 정보로 갱신",
               applicable_to_types=["self"], executable=False),
    ActionSpec(id="eidos.meta.no_op", label_ko="아무것도 안 함",
               description="명시적 inaction — 다음 tick 까지 침묵",
               applicable_to_types=["self"], executable=False),
    ActionSpec(id="eidos.meta.escalate_to_user", label_ko="사용자에게 위임",
               description="결정을 사용자에게 넘김 — HITL gate",
               applicable_to_types=["self"], executable=False),
]


# ── 패턴 매칭 헬퍼 ─────────────────────────────────────────────────────
def _pattern_to_regex(pattern: str) -> re.Pattern:
    """\"abstract.provoke.*\" → 정규식. \"*\" 는 segment 1개+ 매칭."""
    # 점 + segment 구조를 보존. * 는 [^.]+ (점 제외 1글자+).
    # 정확한 segment 일치를 위해 anchored.
    parts = pattern.split(".")
    regex_parts = []
    for p in parts:
        if p == "*":
            regex_parts.append(r"[^.]+")
        elif p == "**":
            regex_parts.append(r".+")
        else:
            regex_parts.append(re.escape(p))
    return re.compile(r"^" + r"\.".join(regex_parts) + r"$")


# ── ActionRegistry ─────────────────────────────────────────────────────
class ActionRegistry:
    """Global default + stage-local abstract action 들의 통합 레지스트리.

    LLM 자동 제안으로 추가된 action 은 stage-local 로 저장됨 (DEFAULT_GLOBALS 안 건드림).
    Stage save 시에는 stage-local 만 직렬화 (globals 는 항상 재로드).
    """

    def __init__(self, locals_: Optional[Iterable[ActionSpec]] = None):
        self._globals: dict[str, ActionSpec] = {a.id: a for a in DEFAULT_GLOBALS}
        self._locals: dict[str, ActionSpec] = {}
        if locals_:
            for spec in locals_:
                self.add_local(spec)

    # ── 조회 ──
    def get(self, action_id: str) -> Optional[ActionSpec]:
        if not action_id:
            return None
        return self._locals.get(action_id) or self._globals.get(action_id)

    def has(self, action_id: str) -> bool:
        return action_id in self._locals or action_id in self._globals

    def all_ids(self) -> list[str]:
        return sorted(set(self._globals.keys()) | set(self._locals.keys()))

    def all_specs(self) -> list[ActionSpec]:
        merged: dict[str, ActionSpec] = dict(self._globals)
        merged.update(self._locals)   # locals override globals (사용자 의지 우선)
        return [merged[aid] for aid in sorted(merged.keys())]

    # ── 추가/제거 ──
    def add_local(self, spec: ActionSpec) -> bool:
        if not spec or not spec.id:
            return False
        # ID 정규화 — 소문자·점·언더스코어·영숫자만
        clean_id = re.sub(r"[^a-z0-9._]", "", spec.id.lower())
        if not clean_id:
            return False
        spec.id = clean_id
        if not spec.source or spec.source == "default":
            spec.source = "stage_local"
        self._locals[clean_id] = spec
        return True

    def remove_local(self, action_id: str) -> bool:
        if action_id in self._locals:
            del self._locals[action_id]
            return True
        return False

    # ── 패턴 매칭 ──
    def match(self, pattern: str) -> list[ActionSpec]:
        """wildcard 패턴 매칭. 예: \"abstract.provoke.*\" """
        if not pattern:
            return []
        if "*" not in pattern:
            spec = self.get(pattern)
            return [spec] if spec else []
        try:
            rx = _pattern_to_regex(pattern)
            return [s for s in self.all_specs() if rx.match(s.id)]
        except Exception as e:
            print(f"[action_registry] match 실패 (graceful): {e}")
            return []

    # ── actor type 필터 ──
    def applicable_for(self, actor_type: str) -> list[ActionSpec]:
        if not actor_type:
            return []
        return [
            s for s in self.all_specs()
            if not s.applicable_to_types or actor_type in s.applicable_to_types
        ]

    def executable_ids(self) -> list[str]:
        return [s.id for s in self.all_specs() if s.executable]

    # ── 직렬화 (locals 만) ──
    def serialize_locals(self) -> list[dict]:
        return [s.serialize() for s in sorted(self._locals.values(), key=lambda x: x.id)]

    @classmethod
    def from_locals_dict(cls, locals_list: list) -> "ActionRegistry":
        specs: list[ActionSpec] = []
        if isinstance(locals_list, list):
            for item in locals_list:
                if isinstance(item, dict):
                    specs.append(ActionSpec.deserialize(item))
        return cls(locals_=specs)


# ── 편의 헬퍼 ─────────────────────────────────────────────────────────
def make_abstract_spec(action_id: str, label_ko: str,
                       applicable_types: Optional[list[str]] = None,
                       source: str = "llm_proposed",
                       description: str = "") -> ActionSpec:
    """추상 action 빠른 생성 (LLM 자동 제안 결과에서 import 할 때)."""
    if not action_id.startswith("abstract."):
        action_id = f"abstract.{action_id}"
    return ActionSpec(
        id=action_id,
        label_ko=label_ko,
        description=description,
        applicable_to_types=applicable_types or ["human", "organization"],
        executable=False,
        source=source,
    )


def is_executable(action_id: str) -> bool:
    """이 action_id 가 (default 기준) 실행 가능한지 빠른 검사."""
    return action_id.startswith("eidos.tool.")


def is_meta(action_id: str) -> bool:
    return action_id.startswith("eidos.meta.")


def is_abstract(action_id: str) -> bool:
    return action_id.startswith("abstract.")


def namespace_of(action_id: str) -> str:
    return action_id.split(".", 1)[0] if action_id else ""


__all__ = [
    "ActionSpec", "ActionRegistry", "DEFAULT_GLOBALS",
    "make_abstract_spec",
    "is_executable", "is_meta", "is_abstract", "namespace_of",
]
