# eidos_pattern_model.py
# [2026-05-26 Phase 2-A] ToM 에이전트 — per-action_id 패턴 모델·진짜 학습.
#
# 옛 Phase 0-D 의 history.jsonl 을 마이닝해서:
#   - action_id 별 발화 빈도 (count·EIDOS decisions / other predictions)
#   - action_id 별 execution status 분포 (ok·fail·dry_run·skip·hitl)
#   - 1-step transition (이전 action_id → 다음 action_id 빈도)
# 통계 → LLM prompt 에 prior 로 inject·decision 의 정보성 ↑.
#
# 향후 (Phase 2-B+): per-id Bayesian prior·belief→action 회귀·multi-step chain.
#
# 디스크: eidos_files/agents/{stage_id}/pattern_model.json (캐시·재계산 비용 절감).

from __future__ import annotations

import datetime as _dt
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional


_PATTERN_FILE = "pattern_model.json"

# ── [Phase 12] Policy Learning 상수 ────────────────────────────────────
# explicit feedback 의 reward 범위
EXPLICIT_FEEDBACK_MAX = 1.0
EXPLICIT_FEEDBACK_MIN = -1.0
# avg_reward 계산 가중치 (exec status 가 기본·explicit 가 가산)
_EXEC_WEIGHT = 0.7
_EXPLICIT_WEIGHT = 0.3
# UCB1 exploration 상수
_UCB_C = math.sqrt(2.0)
# failing action 임계값
_FAILING_MIN_ATTEMPTS = 5
_FAILING_MAX_SUCCESS_RATE = 0.2
# recommend_action 의 ε-greedy
_DEFAULT_EXPLORATION_EPSILON = 0.15


# ── [2026-05-27 통합 후속 #3] PCA DAG 별 action_id 분리 ─────────────
# 모든 eidos.tool.pca.run_dag 호출이 같은 id 로 카운트되면 어떤 DAG 가 자주
# 쓰이는지·어떤 DAG → 어떤 follow-up 인지 학습 불가. effective_aid 에
# dag_name suffix 붙여 decisions·transitions·execution_status 모두 분리.
_PCA_RUN_DAG_ACTION_ID = "eidos.tool.pca.run_dag"


# ── [Phase 12] ActionValue — action 별 학습 통계 ───────────────────────
@dataclass
class ActionValue:
    """1 action 의 학습 가능한 가치 통계.

    - n_attempts: 시도 횟수 (execution event 발생 1회 = 1 attempt)
    - n_success: status=ok 였던 횟수
    - n_fail: status=fail 였던 횟수 (skip/dry_run/hitl 는 attempt 만 카운트, success/fail 둘 다 X)
    - avg_reward: 0~1 (exec_success_rate × _EXEC_WEIGHT + normalized_explicit × _EXPLICIT_WEIGHT)
    - explicit_reward_sum / explicit_reward_count: 사용자 명시 피드백 누적
    - conditional_rewards: belief context key → {n, sum_reward, avg_reward}
                          예: "work_state=working" → {n: 5, sum_reward: 4.0, avg_reward: 0.8}
    - last_used_at: 마지막 execution 시각 (recency 활용 옵션)
    """

    action_id: str
    n_attempts: int = 0
    n_success: int = 0
    n_fail: int = 0
    explicit_reward_sum: float = 0.0
    explicit_reward_count: int = 0
    avg_reward: float = 0.0
    last_used_at: str = ""
    conditional_rewards: dict = field(default_factory=dict)

    def success_rate(self) -> float:
        """ok / (ok + fail). 둘 다 0 이면 0.5 (불확실)."""
        denom = self.n_success + self.n_fail
        if denom <= 0:
            return 0.5
        return self.n_success / denom

    def explicit_avg(self) -> float:
        """explicit feedback 평균 (-1~+1). 데이터 없으면 0."""
        if self.explicit_reward_count <= 0:
            return 0.0
        return self.explicit_reward_sum / self.explicit_reward_count

    def recompute_avg_reward(self) -> None:
        """avg_reward 재계산 — exec success rate + normalized explicit.

        데이터 없는 신호는 평균에서 제외 (success_rate() 의 default 0.5 가 섞이지
        않게). 둘 다 없으면 0.0 (모름).
        """
        has_exec = (self.n_success + self.n_fail) > 0
        has_explicit = self.explicit_reward_count > 0
        if has_exec and has_explicit:
            exec_part = self.success_rate()
            explicit_norm = (self.explicit_avg() + 1.0) / 2.0
            self.avg_reward = exec_part * _EXEC_WEIGHT + explicit_norm * _EXPLICIT_WEIGHT
        elif has_exec:
            self.avg_reward = self.success_rate()
        elif has_explicit:
            self.avg_reward = (self.explicit_avg() + 1.0) / 2.0
        else:
            self.avg_reward = 0.0

    def is_failing(
        self,
        min_attempts: int = _FAILING_MIN_ATTEMPTS,
        max_success_rate: float = _FAILING_MAX_SUCCESS_RATE,
    ) -> bool:
        """N 회 이상 시도했는데 success rate 가 낮은가 — '위험 action' 자동 경고."""
        if self.n_attempts < min_attempts:
            return False
        return self.success_rate() < max_success_rate

    def ucb1_score(self, n_total: int) -> float:
        """UCB1 — avg_reward + sqrt(2 ln N / n). 미시도 action 은 +∞."""
        if self.n_attempts <= 0:
            return float("inf")
        if n_total <= 0:
            return self.avg_reward
        try:
            return self.avg_reward + _UCB_C * math.sqrt(math.log(n_total) / self.n_attempts)
        except ValueError:
            return self.avg_reward

    def serialize(self) -> dict:
        return asdict(self)

    @classmethod
    def deserialize(cls, data: dict) -> "ActionValue":
        if not isinstance(data, dict):
            data = {}
        try:
            attempts = int(data.get("n_attempts", 0))
            success = int(data.get("n_success", 0))
            fail = int(data.get("n_fail", 0))
            ex_cnt = int(data.get("explicit_reward_count", 0))
        except Exception:
            attempts = success = fail = ex_cnt = 0
        try:
            ex_sum = float(data.get("explicit_reward_sum", 0.0))
            avg_r = float(data.get("avg_reward", 0.0))
        except Exception:
            ex_sum = 0.0
            avg_r = 0.0
        cond = data.get("conditional_rewards") or {}
        if not isinstance(cond, dict):
            cond = {}
        return cls(
            action_id=str(data.get("action_id") or ""),
            n_attempts=attempts,
            n_success=success,
            n_fail=fail,
            explicit_reward_sum=ex_sum,
            explicit_reward_count=ex_cnt,
            avg_reward=avg_r,
            last_used_at=str(data.get("last_used_at") or ""),
            conditional_rewards=dict(cond),
        )


def _ctx_key(context: Optional[dict]) -> str:
    """context dict → 정렬된 'k=v,k=v' string. 빈 dict 면 빈."""
    if not isinstance(context, dict) or not context:
        return ""
    parts = []
    for k in sorted(context.keys()):
        v = context[k]
        if isinstance(v, (str, int, float, bool)):
            parts.append(f"{k}={v}")
    return ",".join(parts)


def _snapshot_to_context(snap: dict) -> dict:
    """belief snapshot → context dict (categorical indicator 만).

    숫자 indicator 는 conditional 매칭 어려움 (range 비교 필요·median 분할은
    auto_threshold_findings 가 별도 담당). str/bool 만 정확 매칭.
    cardinality 폭발 회피 위해 30자+ string 은 skip.
    """
    if not isinstance(snap, dict):
        return {}
    ctx: dict = {}
    for actor_name, ind in snap.items():
        if not isinstance(ind, dict):
            continue
        for k, v in ind.items():
            if isinstance(v, bool):
                ctx[f"{actor_name}.{k}"] = v
            elif isinstance(v, str):
                if 0 < len(v) <= 30:
                    ctx[f"{actor_name}.{k}"] = v
            # 숫자는 skip (auto_threshold_findings 담당)
    return ctx


def _expand_pca_aid(action_id: str, args_or_pca: dict) -> str:
    """action_id 가 pca.run_dag 이고 dag_name 추출 가능하면 suffix 붙여 반환.

    args_or_pca:
      - decision event 에서: decision.args (사용자 의도 — args.dag_name)
      - execution event 에서: exec_result.pca (Phase C 풍부 메트릭 — pca.dag_name)

    dag_name 없으면 원본 action_id 그대로 반환 (graceful — 옛 history 호환).
    """
    if action_id != _PCA_RUN_DAG_ACTION_ID:
        return action_id
    dag_name = ""
    if isinstance(args_or_pca, dict):
        dag_name = (args_or_pca.get("dag_name") or "").strip()
    if not dag_name:
        return action_id
    # 보수적 sanitize — slash 가 prompt/직렬화 안 깨지게 (대부분 안전 문자)
    safe_dag = "".join(c if c.isalnum() or c in "-_." else "_" for c in dag_name)[:48]
    return f"{action_id}/{safe_dag}"


class PatternModel:
    """per-action_id 누적 통계 + 1-step transition + 사람 읽기 요약.

    구성:
      decisions:    Counter — action_id (EIDOS 결정) 빈도
      predictions:  dict[actor_name, Counter] — actor 별 예측된 action_id 빈도
      execution_status: dict[action_id, Counter] — 그 action 실행 결과 status 분포
      transitions:  dict[action_id, Counter] — 이 action 다음 자주 나오는 action
      tick_count:   몇 tick 분 데이터인지
    """

    def __init__(self):
        self.decisions: Counter = Counter()
        self.predictions: dict[str, Counter] = defaultdict(Counter)
        self.execution_status: dict[str, Counter] = defaultdict(Counter)
        self.transitions: dict[str, Counter] = defaultdict(Counter)
        self.tick_count: int = 0
        self.built_at: str = ""
        # [Phase 3-A 2026-05-26] belief snapshot ↔ decision 매핑
        # 각 entry: {tick_num, snapshot: {actor_name: {indicator: value}}, decision_action_id, exec_status}
        self.belief_tick_log: list[dict] = []
        # [Phase 4-B 2026-05-26] 2-step chain — (a, b) → c 빈도 (n-gram)
        self.transitions_2step: dict[tuple, Counter] = defaultdict(Counter)
        # [Phase 12 2026-05-27] ActionValue — adaptive policy 학습.
        # action_id → ActionValue (success/fail/avg_reward/conditional_rewards)
        # from_history 가 execution event 처리 시 자동 누적. register_explicit_feedback
        # 가 외부 reward 추가. recommend_action 이 학습된 가치로 후보 정렬.
        self.action_values: dict[str, ActionValue] = {}

    # ── 빌드 — history.jsonl events 부터 ─────────────────────────
    @classmethod
    def from_history(cls, events: list[dict]) -> "PatternModel":
        """history events list 마이닝.

        events 는 read_history(stage_id) 결과 (시간 순).
        각 event 의 type:
          tick_start    → tick_count 증가
          prediction    → predictions[actor] 갱신
          decision      → decisions 갱신·transition 계산
          execution     → execution_status 갱신
        """
        model = cls()
        if not events:
            model.built_at = _now_iso()
            return model

        last_decision_action: Optional[str] = None
        prev_prev_action: Optional[str] = None   # [Phase 4-B] 2-step chain 용
        # [Phase 3-A] tick 별 임시 누적 — tick_start 시작·decision/execution 시 채우기
        _pending_tick: dict = {}   # {tick_num, snapshot, decision_action_id, exec_status, predictions_summary}

        def _flush_pending():
            if _pending_tick:
                model.belief_tick_log.append(dict(_pending_tick))

        for ev in events:
            if not isinstance(ev, dict):
                continue
            etype = ev.get("event")

            if etype == "tick_start":
                # 이전 tick 의 미완성 entry flush
                _flush_pending()
                _pending_tick.clear()
                model.tick_count += 1
                _pending_tick["tick_num"] = ev.get("tick_num", model.tick_count)
                _pending_tick["snapshot"] = ev.get("belief_snapshot") or {}
                _pending_tick["decision_action_id"] = None
                _pending_tick["exec_status"] = None
                _pending_tick["predictions_summary"] = {}

            elif etype == "prediction":
                preds = ev.get("predictions") or {}
                if isinstance(preds, dict):
                    for actor_name, action_list in preds.items():
                        if not isinstance(action_list, list):
                            continue
                        for entry in action_list:
                            if not isinstance(entry, dict):
                                continue
                            aid = (entry.get("action_id") or "").strip()
                            if aid:
                                model.predictions[actor_name][aid] += 1
                                # tick 별 top-1 도 저장 (predictions_summary)
                                cur_top = _pending_tick.get("predictions_summary", {}).get(actor_name)
                                if cur_top is None:
                                    _pending_tick.setdefault("predictions_summary", {})[actor_name] = aid

            elif etype == "decision":
                d = ev.get("decision") or {}
                aid = (d.get("action_id") or "").strip()
                if aid:
                    # [2026-05-27 통합 후속 #3] PCA DAG 는 args.dag_name 으로 분리.
                    # 모든 pca.run_dag 호출이 같은 action_id 로 collapse 되면 어떤 DAG 가
                    # 자주 쓰이는지·어떤 DAG → 어떤 follow-up 인지 학습 불가. effective
                    # action_id 에 suffix 붙여 decisions·transitions 양쪽 분리.
                    effective_aid = _expand_pca_aid(aid, d.get("args") or {})
                    model.decisions[effective_aid] += 1
                    if last_decision_action:
                        model.transitions[last_decision_action][effective_aid] += 1
                    # [Phase 4-B] 2-step chain — (prev_prev, prev) → cur
                    if prev_prev_action and last_decision_action:
                        model.transitions_2step[(prev_prev_action, last_decision_action)][effective_aid] += 1
                    prev_prev_action = last_decision_action
                    last_decision_action = effective_aid
                    _pending_tick["decision_action_id"] = effective_aid

            elif etype == "execution":
                aid = (ev.get("action_id") or "").strip()
                status = (ev.get("status") or "").strip()
                if aid and status:
                    # execution event 도 decision 과 match 되도록 같은 expand.
                    # exec_result.pca.dag_name (Phase C 풍부 메트릭) 우선·없으면 raw.
                    pca = ev.get("pca") if isinstance(ev.get("pca"), dict) else {}
                    effective_aid = _expand_pca_aid(aid, pca)
                    model.execution_status[effective_aid][status] += 1
                    if _pending_tick.get("decision_action_id") == effective_aid:
                        _pending_tick["exec_status"] = status
                    # [Phase 12] ActionValue 갱신 — exec status 기반 학습.
                    av = model.action_values.get(effective_aid)
                    if av is None:
                        av = ActionValue(action_id=effective_aid)
                        model.action_values[effective_aid] = av
                    av.n_attempts += 1
                    if status == "ok":
                        av.n_success += 1
                    elif status == "fail":
                        av.n_fail += 1
                    # skip/dry_run/hitl 는 attempts 만 카운트 (성공·실패 모호)
                    av.last_used_at = str(ev.get("ts") or _now_iso())
                    # belief snapshot 있으면 conditional reward 도 갱신
                    snap = _pending_tick.get("snapshot") or {}
                    ctx = _snapshot_to_context(snap)
                    if ctx:
                        ck = _ctx_key(ctx)
                        if ck:
                            cond = av.conditional_rewards.get(ck) or {
                                "n": 0, "n_ok": 0, "n_fail": 0, "avg_reward": 0.0,
                            }
                            cond["n"] = int(cond.get("n", 0)) + 1
                            if status == "ok":
                                cond["n_ok"] = int(cond.get("n_ok", 0)) + 1
                            elif status == "fail":
                                cond["n_fail"] = int(cond.get("n_fail", 0)) + 1
                            denom = cond["n_ok"] + cond["n_fail"]
                            cond["avg_reward"] = (cond["n_ok"] / denom) if denom > 0 else 0.5
                            av.conditional_rewards[ck] = cond
                    av.recompute_avg_reward()

            elif etype == "explicit_feedback":
                # [Phase 12] 외부 hook 이 append 한 reward event — register_action_feedback shim.
                # action_id + reward + (옵션) context → ActionValue 누적.
                aid = (ev.get("action_id") or "").strip()
                if not aid:
                    continue
                try:
                    rwd = float(ev.get("reward", 0.0))
                except Exception:
                    continue
                ctx = ev.get("context") if isinstance(ev.get("context"), dict) else None
                model.register_explicit_feedback(aid, rwd, context=ctx)

        # 마지막 tick flush
        _flush_pending()
        model.built_at = _now_iso()
        return model

    # ── 직렬화 ───────────────────────────────────────────────────
    def serialize(self) -> dict:
        # [Phase 4-B] 2-step transition 도 직렬화 — tuple key 는 "a|b" 문자열로
        # [Phase 12] action_values 직렬화 추가
        return {
            "version": 4,   # [Phase 12] action_values 추가
            "tick_count": self.tick_count,
            "built_at": self.built_at,
            "decisions": dict(self.decisions),
            "predictions": {a: dict(c) for a, c in self.predictions.items()},
            "execution_status": {a: dict(c) for a, c in self.execution_status.items()},
            "transitions": {a: dict(c) for a, c in self.transitions.items()},
            "transitions_2step": {
                f"{a}|{b}": dict(c) for (a, b), c in self.transitions_2step.items()
            },
            "belief_tick_log": self.belief_tick_log,
            "action_values": {aid: av.serialize() for aid, av in self.action_values.items()},
        }

    @classmethod
    def deserialize(cls, data: dict) -> "PatternModel":
        m = cls()
        if not isinstance(data, dict):
            return m
        m.tick_count = int(data.get("tick_count", 0))
        m.built_at = str(data.get("built_at", ""))
        m.decisions = Counter(data.get("decisions") or {})
        preds = data.get("predictions") or {}
        if isinstance(preds, dict):
            for a, c in preds.items():
                if isinstance(c, dict):
                    m.predictions[a] = Counter(c)
        es = data.get("execution_status") or {}
        if isinstance(es, dict):
            for a, c in es.items():
                if isinstance(c, dict):
                    m.execution_status[a] = Counter(c)
        tr = data.get("transitions") or {}
        if isinstance(tr, dict):
            for a, c in tr.items():
                if isinstance(c, dict):
                    m.transitions[a] = Counter(c)
        # [Phase 3-A] belief_tick_log
        btl = data.get("belief_tick_log") or []
        if isinstance(btl, list):
            m.belief_tick_log = [e for e in btl if isinstance(e, dict)]
        # [Phase 4-B] 2-step transition
        t2 = data.get("transitions_2step") or {}
        if isinstance(t2, dict):
            for k, c in t2.items():
                if isinstance(c, dict) and "|" in k:
                    a, b = k.split("|", 1)
                    m.transitions_2step[(a, b)] = Counter(c)
        # [Phase 12] action_values
        av_data = data.get("action_values") or {}
        if isinstance(av_data, dict):
            for aid, raw in av_data.items():
                if isinstance(raw, dict):
                    m.action_values[aid] = ActionValue.deserialize(raw)
        return m

    # ── 디스크 영속화 (옵션 — 캐시) ──────────────────────────────
    @classmethod
    def from_stage_id(cls, stage_id: str, force_rebuild: bool = False) -> "PatternModel":
        """stage_id 로 model 로드 — 캐시 hit 면 빨리, 미스/rebuild 면 history 마이닝.

        주의: history 가 추가됐는데 캐시는 안 갱신됐을 수 있음 — force_rebuild=True 권장.
        """
        if not stage_id:
            return cls()
        path = _model_path(stage_id)
        if not force_rebuild and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return cls.deserialize(data)
            except Exception as e:
                print(f"[pattern_model] cache 로드 실패 (rebuild 진행): {e}")

        # rebuild
        try:
            from eidos_agent_stage_store import read_history
            events = read_history(stage_id, limit=None) or []
        except Exception as e:
            print(f"[pattern_model] history 로드 실패: {e}")
            events = []
        model = cls.from_history(events)
        try:
            model.save(stage_id)
        except Exception:
            pass
        return model

    def save(self, stage_id: str) -> bool:
        if not stage_id:
            return False
        path = _model_path(stage_id)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.serialize(), f, ensure_ascii=False, indent=2)
            if os.path.exists(path):
                try:
                    os.replace(tmp, path)
                except Exception:
                    os.remove(path)
                    os.rename(tmp, path)
            else:
                os.rename(tmp, path)
            return True
        except Exception as e:
            print(f"[pattern_model] save 실패: {e}")
            return False

    # ── LLM prompt 용 텍스트 ─────────────────────────────────────
    def to_prompt_block(self, eidos_repertoire: Optional[list] = None,
                        max_lines: int = 12) -> str:
        """LLM prompt 에 inject 할 prior 정보 — 누적 통계 텍스트.

        eidos_repertoire 주면 그 안 action_id 만 표시 (집중).
        """
        if self.tick_count == 0 and not self.decisions:
            return "(아직 패턴 데이터 없음 — 첫 tick)"

        lines: list[str] = []
        lines.append(f"[누적 패턴 — {self.tick_count} tick 분석 기준]")

        # EIDOS 결정 빈도 — repertoire 안 / top N
        if self.decisions:
            relevant = self.decisions
            if eidos_repertoire:
                relevant = Counter({k: v for k, v in self.decisions.items() if k in eidos_repertoire})
            top = relevant.most_common(8)
            if top:
                lines.append("EIDOS 결정 빈도:")
                for aid, cnt in top:
                    # status 분포 부착
                    s_dist = self.execution_status.get(aid, Counter())
                    if s_dist:
                        s_str = ", ".join(f"{s}={c}" for s, c in s_dist.most_common(3))
                        lines.append(f"  - {aid}: {cnt}회  ({s_str})")
                    else:
                        lines.append(f"  - {aid}: {cnt}회")

        # 다른 actor 의 예측 빈도 — 가장 자주 예측된 action_id
        if self.predictions:
            other_summary = []
            for actor_name, c in self.predictions.items():
                top = c.most_common(2)
                if top:
                    s = ", ".join(f"{aid}({n})" for aid, n in top)
                    other_summary.append(f"  - {actor_name}: {s}")
            if other_summary:
                lines.append("외부 actor 빈출 예측 (top 2/actor):")
                lines.extend(other_summary[:max_lines])

        # 1-step transition — 가장 강한 chain (a→b 빈도가 ≥ 2)
        chain_pairs = []
        for src, c in self.transitions.items():
            for dst, n in c.items():
                if n >= 2:
                    chain_pairs.append((src, dst, n))
        chain_pairs.sort(key=lambda x: -x[2])
        if chain_pairs[:5]:
            lines.append("자주 이어진 결정 chain (2회 이상):")
            for src, dst, n in chain_pairs[:5]:
                lines.append(f"  - {src} → {dst}: {n}회")

        # fail 비율 ↑ action 경고
        for aid, sdist in self.execution_status.items():
            total = sum(sdist.values())
            fails = sdist.get("fail", 0)
            if total >= 2 and fails >= max(2, total // 2):
                lines.append(f"⚠️ {aid}: 실패율 {fails}/{total} — 다른 action 고려")

        # [Phase 12] ActionValue — 학습된 가치 prior (LLM 이 보고 결정 반영)
        if self.action_values:
            # 학습된 가치 상위 — repertoire 안 필터링
            top_av = sorted(
                self.action_values.values(),
                key=lambda x: x.avg_reward, reverse=True,
            )
            if eidos_repertoire:
                top_av = [av for av in top_av if av.action_id in eidos_repertoire]
            top_av = [av for av in top_av if av.n_attempts >= 2][:5]
            if top_av:
                lines.append("\n학습된 가치 (Phase 12 — avg_reward 순):")
                for av in top_av:
                    exp_tag = (f"·explicit {av.explicit_avg():+.2f}"
                               if av.explicit_reward_count > 0 else "")
                    lines.append(
                        f"  ✓ {av.action_id}: reward={av.avg_reward:.2f} "
                        f"({av.n_success}/{av.n_attempts}={av.success_rate():.0%}{exp_tag})"
                    )
            # failing actions — 명시적 위험 경고 (단순 경고 위 + 학습 기반)
            failers = self.failing_actions()
            if failers:
                # repertoire 안 필터링
                if eidos_repertoire:
                    failers = [av for av in failers if av.action_id in eidos_repertoire]
                if failers:
                    lines.append("\n⚠️ 학습된 위험 action (≥5 시도 + 성공률 <20%) — 가급적 회피:")
                    for av in failers[:3]:
                        lines.append(
                            f"  - {av.action_id}: {av.n_fail}/{av.n_attempts} 실패 "
                            f"(success_rate={av.success_rate():.0%})"
                        )

        # [Phase 3-A] 최근 belief → decision 매핑 (LLM in-context 학습)
        recent = [e for e in self.belief_tick_log if e.get("decision_action_id")][-5:]
        if recent:
            lines.append("\n최근 belief ↔ decision 매핑 (LLM 추론용):")
            for e in recent:
                tnum = e.get("tick_num", "?")
                snap = e.get("snapshot") or {}
                aid = e.get("decision_action_id", "?")
                status = e.get("exec_status") or "·"
                # belief 한 줄로 압축 — 각 actor 의 indicator 1~2개만
                snap_parts = []
                for actor_name, ind in snap.items():
                    if not isinstance(ind, dict):
                        continue
                    items = list(ind.items())[:2]
                    snap_parts.append(f"{actor_name}({', '.join(f'{k}={v}' for k, v in items)})")
                snap_str = " | ".join(snap_parts[:3]) or "(empty)"
                lines.append(f"  - t{tnum}: {snap_str} → `{aid}` [{status}]")

        return "\n".join(lines)

    # ── [Phase 3-A] 조건부 분석 (단순 — predicate 받기) ────────────
    def conditional_action_dist(self, predicate) -> dict:
        """predicate(snapshot, tick_num) → bool 만족하는 tick 에서 발화한 action 빈도.

        Returns: {action_id: count}
        """
        out: Counter = Counter()
        for e in self.belief_tick_log:
            try:
                if predicate(e.get("snapshot") or {}, e.get("tick_num", 0)):
                    aid = e.get("decision_action_id")
                    if aid:
                        out[aid] += 1
            except Exception:
                continue
        return dict(out)

    def auto_threshold_findings(self, min_samples: int = 3) -> list[dict]:
        """모든 (actor, indicator) 쌍의 median 분할 → 양쪽 action 분포 비교.

        분포가 다른 (top action 차이 ≥ 0.3) 쌍만 반환·sample ≥ min_samples 보장.

        Returns: [{actor, indicator, threshold, above: {action_id, ratio}, below: {action_id, ratio}}]
        """
        findings = []
        if len(self.belief_tick_log) < min_samples * 2:
            return findings

        # 모든 (actor, indicator) 쌍 수집
        pairs: dict[tuple, list] = defaultdict(list)
        for e in self.belief_tick_log:
            snap = e.get("snapshot") or {}
            aid = e.get("decision_action_id")
            if not aid or not isinstance(snap, dict):
                continue
            for actor, ind in snap.items():
                if not isinstance(ind, dict):
                    continue
                for key, val in ind.items():
                    if isinstance(val, (int, float)):
                        pairs[(actor, key)].append((val, aid))

        for (actor, key), data in pairs.items():
            if len(data) < min_samples * 2:
                continue
            data_sorted = sorted(data, key=lambda x: x[0])
            mid = len(data_sorted) // 2
            below = data_sorted[:mid]
            above = data_sorted[mid:]
            threshold = data_sorted[mid][0]

            below_actions = Counter(a for _, a in below)
            above_actions = Counter(a for _, a in above)
            if not below_actions or not above_actions:
                continue
            below_top, below_cnt = below_actions.most_common(1)[0]
            above_top, above_cnt = above_actions.most_common(1)[0]
            below_ratio = below_cnt / len(below)
            above_ratio = above_cnt / len(above)

            # top action 이 다르고·각각 50%+ 우세 일 때 (균등 분할이면 ratio 차 0 일 수 있음)
            if below_top != above_top and below_ratio >= 0.5 and above_ratio >= 0.5:
                findings.append({
                    "actor": actor,
                    "indicator": key,
                    "threshold": threshold,
                    "above": {"action_id": above_top, "ratio": above_ratio, "n": len(above)},
                    "below": {"action_id": below_top, "ratio": below_ratio, "n": len(below)},
                })

        return findings

    # ── 사람 읽기 요약 (디버그·UI) ──────────────────────────────
    def summary_markdown(self) -> str:
        """간결한 사람 읽기 요약 — markdown."""
        if self.tick_count == 0 and not self.decisions:
            return "(빈 모델 — 아직 학습 데이터 없음)"
        lines = [f"## 📊 PatternModel — {self.tick_count} tick"]
        if self.decisions:
            lines.append("\n### EIDOS 결정 분포")
            for aid, cnt in self.decisions.most_common(10):
                s = self.execution_status.get(aid, Counter())
                s_str = ", ".join(f"{k}={v}" for k, v in s.most_common(4))
                lines.append(f"- `{aid}` — {cnt}회  ({s_str})")
        if self.predictions:
            lines.append("\n### 외부 actor 예측 분포")
            for actor, c in self.predictions.items():
                top = c.most_common(5)
                lines.append(f"**{actor}**:")
                for aid, cnt in top:
                    lines.append(f"  - `{aid}` — {cnt}회 예측")
        if self.transitions:
            lines.append("\n### 1-step Transition (≥ 2회)")
            pairs = []
            for src, c in self.transitions.items():
                for dst, n in c.items():
                    if n >= 2:
                        pairs.append((src, dst, n))
            pairs.sort(key=lambda x: -x[2])
            for src, dst, n in pairs[:10]:
                lines.append(f"- `{src}` → `{dst}` ({n}회)")
        # [Phase 3-A] auto threshold findings
        findings = self.auto_threshold_findings()
        if findings:
            lines.append("\n### 🔍 belief-conditional 패턴 (median 분할)")
            for f in findings[:8]:
                lines.append(
                    f"- **{f['actor']}.{f['indicator']}** 기준 {f['threshold']:.3g}:\n"
                    f"  - ≥ {f['threshold']:.3g} → `{f['above']['action_id']}` "
                    f"({f['above']['ratio']:.0%}·n={f['above']['n']})\n"
                    f"  - < {f['threshold']:.3g} → `{f['below']['action_id']}` "
                    f"({f['below']['ratio']:.0%}·n={f['below']['n']})"
                )
        if self.belief_tick_log:
            lines.append(f"\n### 📋 belief tick log: {len(self.belief_tick_log)} entries")
        # [Phase 12] action_values 요약
        if self.action_values:
            lines.append(f"\n### 🎯 ActionValue — {len(self.action_values)} actions")
            # avg_reward 내림차순 top·bottom 표시
            sorted_avs = sorted(self.action_values.values(),
                                key=lambda x: x.avg_reward, reverse=True)
            for av in sorted_avs[:5]:
                exp_part = f"·explicit={av.explicit_avg():+.2f}({av.explicit_reward_count})" \
                           if av.explicit_reward_count > 0 else ""
                lines.append(
                    f"- `{av.action_id}` reward={av.avg_reward:.2f} "
                    f"({av.n_success}/{av.n_attempts}={av.success_rate():.0%}{exp_part})"
                )
            # failing 경고
            failers = self.failing_actions()
            if failers:
                lines.append("\n#### ⚠️ Failing actions (≥5 attempts, success<20%)")
                for av in failers:
                    lines.append(
                        f"- `{av.action_id}` — {av.n_fail}/{av.n_attempts} 실패 "
                        f"(success_rate={av.success_rate():.0%})"
                    )
        return "\n".join(lines)

    # ── [Phase 12] explicit feedback hook ────────────────────────
    def register_explicit_feedback(
        self,
        action_id: str,
        reward: float,
        context: Optional[dict] = None,
    ) -> bool:
        """사용자/외부 시스템이 명시적으로 action 평가.

        reward: EXPLICIT_FEEDBACK_MIN ~ MAX (기본 -1~+1).
        context: 평가 시점의 belief snapshot (옵션 — conditional 학습 강화).

        Returns: True 면 갱신 성공.
        """
        if not action_id:
            return False
        try:
            r = float(reward)
        except Exception:
            return False
        r = max(EXPLICIT_FEEDBACK_MIN, min(EXPLICIT_FEEDBACK_MAX, r))
        av = self.action_values.get(action_id)
        if av is None:
            av = ActionValue(action_id=action_id)
            self.action_values[action_id] = av
        av.explicit_reward_sum += r
        av.explicit_reward_count += 1
        # context 있으면 conditional 도 누적 (normalized 0~1)
        ck = _ctx_key(_snapshot_to_context(context) if context else {})
        if ck:
            cond = av.conditional_rewards.get(ck) or {
                "n": 0, "n_ok": 0, "n_fail": 0, "avg_reward": 0.0,
            }
            cond["n"] = int(cond.get("n", 0)) + 1
            # explicit reward 를 0~1 로 normalize 해서 avg 에 반영
            r_norm = (r + 1.0) / 2.0
            denom = cond["n"]
            prev_avg = float(cond.get("avg_reward", 0.5))
            cond["avg_reward"] = ((prev_avg * (denom - 1)) + r_norm) / denom
            av.conditional_rewards[ck] = cond
        av.recompute_avg_reward()
        return True

    # ── [Phase 12] recommend_action ──────────────────────────────
    def recommend_action(
        self,
        candidates: list,
        context: Optional[dict] = None,
        *,
        exploration: float = _DEFAULT_EXPLORATION_EPSILON,
        use_ucb1: bool = True,
    ) -> Optional[tuple]:
        """후보 list 에서 학습된 가치 기준 1개 추천.

        - use_ucb1=True (default): UCB1 score (exploration 자동 — 미시도 우선)
        - use_ucb1=False: ε-greedy (exploration 확률로 random)
        - context 주어지면 conditional_rewards 우선 (있으면 base avg 덮어쓰기)

        Returns: (action_id, score) 또는 None (candidates 빈).
        """
        if not candidates:
            return None
        try:
            cands = [str(c) for c in candidates if c]
        except Exception:
            return None
        if not cands:
            return None

        # ε-greedy exploration
        if not use_ucb1 and exploration > 0:
            import random
            if random.random() < exploration:
                pick = random.choice(cands)
                return (pick, 0.0)

        ck = _ctx_key(context) if context else ""
        n_total = sum(av.n_attempts for av in self.action_values.values())

        scored = []
        for aid in cands:
            av = self.action_values.get(aid)
            if av is None:
                # 시도 안 한 action — UCB 면 ∞ (탐색 우선), greedy 면 0.5 (불확실)
                score = float("inf") if use_ucb1 else 0.5
                scored.append((aid, score))
                continue
            # context 있으면 conditional 우선
            ctx_avg = None
            if ck and ck in av.conditional_rewards:
                ctx_avg = float(av.conditional_rewards[ck].get("avg_reward", 0.5))
            if use_ucb1:
                base = ctx_avg if ctx_avg is not None else av.avg_reward
                # UCB1 계산 (avg_reward 만 다르고 나머진 같음)
                if av.n_attempts <= 0:
                    score = float("inf")
                elif n_total <= 0:
                    score = base
                else:
                    try:
                        score = base + _UCB_C * math.sqrt(math.log(n_total) / av.n_attempts)
                    except ValueError:
                        score = base
            else:
                score = ctx_avg if ctx_avg is not None else av.avg_reward
            scored.append((aid, score))

        # 최고 점수 (동률이면 첫 후보·LLM 순서 존중)
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0]

    # ── [Phase 12] failing actions ───────────────────────────────
    def failing_actions(
        self,
        min_attempts: int = _FAILING_MIN_ATTEMPTS,
        max_success_rate: float = _FAILING_MAX_SUCCESS_RATE,
    ) -> list:
        """N+ 시도 + 낮은 성공률 action — 경고 prompt 에 포함할 후보."""
        return [
            av for av in self.action_values.values()
            if av.is_failing(min_attempts, max_success_rate)
        ]


# ── 헬퍼 ───────────────────────────────────────────────────────────
def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _model_path(stage_id: str) -> str:
    return os.path.join("eidos_files", "agents", stage_id, _PATTERN_FILE)


__all__ = ["PatternModel", "ActionValue"]
