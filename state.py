from typing import Any, Literal
from dataclasses import dataclass, field, asdict
import math
import numpy as np
from loguru import logger
import json
from random import Random
from pathlib import Path

from utils import remove_outliers

@dataclass
class Cluster:
    index: int
    rng_index: int
    summary: str
    train_prompts: list[str]
    val_prompts: list[str]
    data_path: str
    aux_info: Any = None

    def to_dict(self) -> dict[str, Any]:
        """ignores aux_info"""
        return {
            "index": self.index,
            "rng_index": self.rng_index,
            "summary": self.summary,
            "train_prompts": self.train_prompts,
            "val_prompts": self.val_prompts,
            "data_path": self.data_path,
        }


def asdict_no_none(obj):
    obj_dict = asdict(obj)
    
    def remove_none_values(x):
        if isinstance(x, dict):
            return {k: remove_none_values(v) for k, v in x.items() if v is not None}
        if isinstance(x, list):
            return [remove_none_values(item) for item in x]
        return x
    
    return remove_none_values(obj_dict)


@dataclass(kw_only=True, slots=True)
class RewriteScore:
    score: float | None      # reward diff if RM, winrate if judge
    raw_score: float | None  # only exists when it is a reward model
    reasoning: str | None    # only exists when it is a LLM judge
    model_name: str


@dataclass(kw_only=True, slots=True)
class Rollout:
    """rewritten rollout"""
    rewritten_response: str
    baseline_response: str
    student_score: RewriteScore
    teacher_score: RewriteScore | None = None
    policy_model: str | None = None


@dataclass(kw_only=True, slots=True)
class BaselineRollout:
    policy_model: str
    response: str
    scores: dict[str, float]


@dataclass
class AttributeStats:
    attribute: str
    rollouts: dict[str, list[Rollout|None]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def parent(self) -> str | None:
        return self.meta.get("parent", None)

    def to_dict(self) -> dict[str, Any]:
        all_results = {}
        for user_prompt, rollouts in self.rollouts.items():
            user_prompt_results = []
            for r in rollouts:
                if r is None:
                    user_prompt_results.append(None)
                else:
                    user_prompt_results.append(asdict_no_none(r))
            all_results[user_prompt] = user_prompt_results
        return {
            "attribute": self.attribute,
            "student_winrate": self.winrate("student"),
            "teacher_winrate": self.winrate("teacher"),
            "meta": self.meta,
            "rollouts": all_results,
        }

    def winrate(self, rater: Literal["student", "teacher"]) -> float | None:
        all_scores = []
        for _, rollouts in self.rollouts.items():
            for r in rollouts:
                if r is None:
                    continue
                if rater == "student":
                    if r.student_score.score is not None:
                        all_scores.append(r.student_score.score)
                elif rater == "teacher":
                    if r.teacher_score is not None and r.teacher_score.score is not None:
                        all_scores.append(r.teacher_score.score)

        if len(all_scores) == 0:
            logger.warning(f"All {rater} scores are None for attribute:\n{self.attribute}")
            return None

        # Check if all scores are -1, 0, or 1
        remove_score_outliers = False
        for s in all_scores:
            if not (
                math.isclose(s, 1, abs_tol=1e-6)
                or math.isclose(s, 0, abs_tol=1e-6)
                or math.isclose(s, -1, abs_tol=1e-6)
            ):
                remove_score_outliers = True
                break

        if remove_score_outliers:
            return np.mean(remove_outliers(all_scores)).item()
        else:
            return np.mean(all_scores).item()

    def __repr__(self):
        return (
            f"AttributeStats(\n"
            f"    attribute={self.attribute[:40]!r},\n"
            f"    num_user_prompts={len(self.rollouts)},\n"
            f"    student_winrate={self.winrate('student')},\n"
            f"    teacher_winrate={self.winrate('teacher')},\n"
            f")"
        )

@dataclass
class SeedState:
    cluster: Cluster
    state: dict
    history: list[dict[str, AttributeStats]]

    @property
    def index(self) -> int:
        return self.cluster.index

    @property
    def rng_index(self) -> int:
        return self.cluster.rng_index


def load_initial_seed_states(
    ds_path: Path,
    topic_ids: list[int],
    val_split_size: int = 64,
    random_seed: int = 42,
) -> list[SeedState]:
    if isinstance(ds_path, str):
        ds_path = Path(ds_path)

    id_to_cluster: dict[int, Cluster] = dict()

    for idx in topic_ids:
        with open(ds_path / f"cluster_{idx}.json", "r") as f:
            data = json.load(f)

        cluster_rng = Random(random_seed + idx)
        
        if len(data["prompts"]) < 3 * val_split_size:
            raise ValueError(f"Not enough prompts for cluster {idx}.")

        cluster_rng.shuffle(data["prompts"])
        train_prompts = data["prompts"][:-val_split_size] if val_split_size > 0 else data["prompts"]
        val_prompts = data["prompts"][-val_split_size:] if val_split_size > 0 else []

        id_to_cluster[idx] = Cluster(
            index=idx,
            rng_index=idx,
            summary=data["summary"],
            train_prompts=train_prompts,
            val_prompts=val_prompts,
            data_path=str(ds_path / f"cluster_{idx}.json")
        )

    initial_seed_states = [
        SeedState(cluster=cluster, state={}, history=[]) 
        for cluster in id_to_cluster.values()
    ]
    initial_seed_states.sort(key=lambda x: x.index)

    print(f"\n\nLoaded {len(initial_seed_states)} seed states")
    for state in initial_seed_states:
        print(
            f"  - Seed {state.index}, {len(state.cluster.train_prompts)} train prompts:\n"
            f"    {state.cluster.summary}"
        )
    print("\n\n")

    return initial_seed_states
