import os
import json
import time
import pickle
from loguru import logger
from pathlib import Path
from typing import Literal
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

from state import SeedState, Rollout, BaselineRollout, AttributeStats, asdict_no_none
from baselines import evaluate_baselines
from utils import timestamp, remove_outliers
from api_models import GenerationModel, RewriteModel, SAME_ATTRS
from reward_models import RewardModel, LocalRewardModel
from bias_evaluator import BiasEvaluator

class Runner(ABC):
    def __init__(
        self,
        seed_states: list[SeedState],
        policy_model: GenerationModel,
        student_model: LocalRewardModel,
        teacher_model: RewardModel,
        run_name: str | None,
        n_baseline_rollouts: int,
        n_validate_rollouts: int,
    ):
        self.step_count = 0
        self.seed_states = seed_states
        self.policy_model = policy_model
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.n_baseline_rollouts = n_baseline_rollouts
        self.n_validate_rollouts = n_validate_rollouts

        self.run_name = run_name or f"{timestamp()}"
        self.run_path.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def runner_type(self) -> str:
        pass

    @property
    def run_path(self) -> Path:
        return Path(f"data/{self.runner_type}/{self.run_name}")


    async def get_baselines(self):
        self.baselines: dict[int, dict[str, list[BaselineRollout]]] = {}
        for ss in self.seed_states:
            cluster = ss.cluster
            baselines = await evaluate_baselines(
                cluster=cluster,
                split="train",
                policy_model=self.policy_model,
                reward_model=self.student_model,
                n_rollouts=self.n_baseline_rollouts,
            )
            if isinstance(self.teacher_model, LocalRewardModel):
                baselines = await evaluate_baselines(
                    cluster=cluster,
                    split="train",
                    policy_model=self.policy_model,
                    reward_model=self.teacher_model,
                    n_rollouts=self.n_baseline_rollouts,
                )
            self.baselines[ss.rng_index] = baselines


    async def get_val_baselines(self):
        self.val_baselines: dict[int, dict[str, list[BaselineRollout]]] = {}
        for ss in self.seed_states:
            cluster = ss.cluster
            baselines = await evaluate_baselines(
                cluster=cluster,
                split="val",
                policy_model=self.policy_model,
                reward_model=self.student_model,
                n_rollouts=self.n_validate_rollouts,
            )
            if isinstance(self.teacher_model, LocalRewardModel):
                baselines = await evaluate_baselines(
                    cluster=cluster,
                    split="val",
                    policy_model=self.policy_model,
                    reward_model=self.teacher_model,
                    n_rollouts=self.n_validate_rollouts,
                )
            self.val_baselines[ss.rng_index] = baselines


    def save_attribute_stats(
        self, direction: Literal["plus", "minus"], save_dir: Path | None = None
    ):
        """
        Save a condensed version of previous step's attribute stats for each seed state,
        ordered by student winrate (reward diff).
        """
        if save_dir is None:
            save_dir = self.run_path / "final_stats"
        save_dir.mkdir(parents=True, exist_ok=True)

        for seed_state in self.seed_states:
            all_attributes = [stats.to_dict() for stats in seed_state.history[-1].values()]

            if direction == "plus":
                all_attributes = sorted(
                    all_attributes,
                    key=lambda x: x["student_winrate"] if x["student_winrate"] is not None else float("-inf"),
                    reverse=True
                )
            else:
                all_attributes = sorted(
                    all_attributes,
                    key=lambda x: x["student_winrate"] if x["student_winrate"] is not None else float("inf"),
                    reverse=False
                )

            # overwrites if already exists
            seed_save_dir = save_dir / f"seed_{seed_state.rng_index}.json"
            with open(seed_save_dir, "w") as f:
                json.dump(all_attributes, f, indent=4)


    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    async def validate(
        self, 
        final_attributes: dict[int, list[str]],
        val_rewriters: list[RewriteModel],
        judge_val_first_n_rollouts: int, 
        judge_val_first_n_user_prompts: int,
        val_split_size: int,
    ):
        """
        final_attributes: seed_state_index -> list of attributes
        """
        if not hasattr(self, "val_baselines"):
            await self.get_val_baselines()

        validation_results: dict[str, dict[int, dict[str, AttributeStats]]] = {
            rewriter.model_name: {} for rewriter in val_rewriters
        }

        # Rewrite and get student scores
        for ss in self.seed_states:
            val_prompts = ss.cluster.val_prompts[:val_split_size]

            async with BiasEvaluator(rewrite_models=val_rewriters, reward_model=self.student_model) as evaluator:
                stats = await evaluator.evaluate_attributes(
                    user_prompts=val_prompts,
                    attributes=final_attributes[ss.rng_index],
                    baselines=self.val_baselines[ss.rng_index],
                    same_attrs=[SAME_ATTRS] * len(final_attributes[ss.rng_index]),
                    save_dir=self.run_path / "validate" / f"seed_{ss.rng_index}_validate",
                )

            for rewriter_name, rewriter_stats in stats.items():
                validation_results[rewriter_name][ss.rng_index] = {
                    k: AttributeStats(attribute=k, rollouts=v) for k, v in rewriter_stats.items()
                }

        for rewriter_name, rewriter_stats in validation_results.items():
            # Populate teacher_score on rollouts in place
            await self.teacher_model.judge_rollouts(
                evaluate_results=rewriter_stats,
                baselines=self.val_baselines,  # type: ignore
                first_n_rollouts=judge_val_first_n_rollouts,
                first_n_user_prompts=judge_val_first_n_user_prompts,
            )

            for seed_state in self.seed_states:
                seed_stats = rewriter_stats[seed_state.rng_index]
                rewriter_dir = self.run_path / "validate" / f"seed_{seed_state.rng_index}_validate" / rewriter_name.replace("/", "_")
                rewriter_dir.mkdir(parents=True, exist_ok=True)

                # Save complete rollouts with teacher scores
                with open(rewriter_dir / "rollouts.json", "w") as f:
                    json_data = {
                        attr: {
                            user: [asdict_no_none(r) if r else None for r in rollouts]
                            for user, rollouts in attr_stats.rollouts.items()
                        }
                        for attr, attr_stats in seed_stats.items()
                    }
                    json.dump(json_data, f, indent=4, sort_keys=True)

                # Save candidate stats
                candidate_stats = []
                for attribute, attribute_stats in seed_stats.items():
                    candidate_stats.append({
                        "attribute": attribute,
                        "student_winrate": attribute_stats.winrate("student"),
                        "teacher_winrate": attribute_stats.winrate("teacher"),
                    })

                with open(rewriter_dir / "candidate_stats.json", "w") as f:
                    json.dump(candidate_stats, f, indent=4, sort_keys=True)

                # Create scatter plot
                valid_points = [(s["student_winrate"], s["teacher_winrate"])
                               for s in candidate_stats
                               if s["student_winrate"] is not None and s["teacher_winrate"] is not None]

                if valid_points:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax.scatter([p[0] for p in valid_points], [p[1] for p in valid_points],
                              c='blue', alpha=0.7, marker='o')

                    ax.set_xlabel('Student Winrate')
                    ax.set_ylabel('Teacher Winrate')
                    ax.set_title(f'Validation: {rewriter_name} - Seed {seed_state.index}')
                    ax.grid(True, alpha=0.3)

                    fig.savefig(rewriter_dir / "validation_scatter.pdf")
                    plt.close(fig)
        