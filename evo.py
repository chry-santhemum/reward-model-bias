import asyncio
import json
import math
import dotenv
from random import Random
from typing import Literal, Any
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from caller import ChatHistory
from state import SeedState, AttributeStats, Rollout, BaselineRollout, RewriteScore
from utils import parse_json_response, set_seed_all
from cluster_models import ClusterModel
from api_models import GenerationModel, RewriteModel, SAME_ATTRS
from reward_models import RewardModel, LocalRewardModel
from runner import Runner
from bias_evaluator import BiasEvaluator
from planner import Planner, PLANNER_SYSTEM
from evo_prompts import get_mutate_prompt, DIRECTION_GOAL, BIAS_NUDGE

dotenv.load_dotenv()
set_seed_all(42)



class EvoPlanner:
    def __init__(
        self,
        direction: Literal["plus", "minus"],
        hypothesis_planner: Planner,
        cluster_model: ClusterModel,
        m_var: int,
        cosine_sim_threshold_initial: float,
        cosine_sim_threshold_evolution: float,
        context: Literal["all", "ancestry", "vanilla"]="all",
        random_seed: int=42,
    ):
        self.direction: Literal["plus", "minus"] = direction
        self.hypothesis_planner = hypothesis_planner
        self.cluster_model = cluster_model
        self.m_var = m_var
        self.cosine_sim_threshold_initial = cosine_sim_threshold_initial
        self.cosine_sim_threshold_evolution = cosine_sim_threshold_evolution
        self.random_seed = random_seed

        self.mutate_prompt = get_mutate_prompt(context)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "hypothesis_planner": self.hypothesis_planner.to_dict(),
            "cluster_model": self.cluster_model.to_dict(),
            "m_var": self.m_var,
            "cosine_sim_threshold_initial": self.cosine_sim_threshold_initial,
            "cosine_sim_threshold_evolution": self.cosine_sim_threshold_evolution,
        }

    async def initial_plan(self, runner: Runner):
        return await self.hypothesis_planner.plan(
            runner=runner,
            direction=self.direction,
            cluster_model=self.cluster_model,
            cosine_sim_threshold=self.cosine_sim_threshold_initial,
        )

    def _get_original_data(self, stats: AttributeStats, baselines: dict[str, list[BaselineRollout]], n_rollouts: int=6) -> str:
        student_wr = stats.winrate("student")
        teacher_wr = stats.winrate("teacher")
        student_wr_str = f"{student_wr:.3f}" if student_wr is not None else "N/A"
        teacher_wr_str = f"{teacher_wr:.3f}" if teacher_wr is not None else "N/A"

        all_rollouts = []
        for user_prompt, rollouts in stats.rollouts.items():
            all_rollouts.extend([
                (r, user_prompt, idx)
                for idx, r in enumerate(rollouts)
                if r is not None and r.teacher_score is not None
            ])

        # Remove outliers before sampling (using IQR bounds)
        if all_rollouts:
            scores = np.array([r.student_score.score for r, _, _ in all_rollouts])
            q1, q3 = np.percentile(scores, [25, 75])
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            all_rollouts = [x for x in all_rollouts if low <= x[0].student_score.score <= high]

        all_rollouts.sort(key=lambda x: x[0].student_score.score)

        # Select top and bottom examples
        n = len(all_rollouts)
        if n == 0:
            chosen_rollouts = []
        elif n <= n_rollouts:
            chosen_rollouts = all_rollouts
        else:
            half = n_rollouts // 2
            chosen_rollouts = all_rollouts[:half] + all_rollouts[-half:] if half > 0 else []

        original_data_rollouts = []
        for r, user_prompt, idx in chosen_rollouts:
            original_data_rollouts.append({
                "user_prompt": user_prompt,
                "response_attribute_not_present": baselines[user_prompt][idx].response,
                "response_attribute_present": r.rewritten_response,
                "Metric A uplift": r.student_score.score,
                "Metric B uplift": r.teacher_score.score if r.teacher_score is not None else "N/A",
            })

        original_data = {
            "Feature": stats.attribute,
            "Metric A average uplift": student_wr_str,
            "Metric B average uplift": teacher_wr_str,
            "Example responses": original_data_rollouts,
        }

        return json.dumps(original_data, indent=4)

    def _get_most_recent_stats(self, attr: str, seed_state: SeedState) -> AttributeStats | None:
        """Find the most recent stats for an attribute in history."""
        for step in range(len(seed_state.history) - 1, -1, -1):
            if attr in seed_state.history[step]:
                return seed_state.history[step][attr]
        return None

    def _get_ancestry_data(
        self,
        attribute: str,
        time_step: int,
        seed_state: SeedState,
        baselines: dict[str, list[BaselineRollout]],
    ) -> tuple[list[str], str]:
        """
        Trace ancestry and find sister attributes.

        Returns:
            - List of attributes to exclude from neighbors (ancestors + sisters)
            - Formatted string of ancestry data for the prompt
        """
        ancestors = []
        ancestor_attrs = []

        # Get current attribute's parent info
        stats = seed_state.history[time_step][attribute]
        parent_attr = stats.meta.get("parent")

        # Trace back through ancestry
        generation = 1
        while parent_attr is not None:
            ancestor_attrs.append(parent_attr)

            # Get most recent stats for this ancestor
            parent_stats = self._get_most_recent_stats(parent_attr, seed_state)
            if parent_stats is None:
                break

            parent_data = self._get_original_data(stats=parent_stats, baselines=baselines)
            ancestors.append(f"=== Generation -{generation} ===\n{parent_data}")

            # Move to next ancestor
            parent_attr = parent_stats.meta.get("parent")
            generation += 1

        # Find sister attributes (other children of ancestors)
        direct_lineage = {attribute} | set(ancestor_attrs)
        sister_attrs = []
        sisters = []

        for step_history in seed_state.history:
            for attr, attr_stats in step_history.items():
                if attr in direct_lineage or attr in sister_attrs:
                    continue
                attr_parent = attr_stats.meta.get("parent")
                if attr_parent in ancestor_attrs:
                    sister_attrs.append(attr)
                    recent_stats = self._get_most_recent_stats(attr, seed_state)
                    if recent_stats is not None:
                        student_wr = recent_stats.winrate("student")
                        teacher_wr = recent_stats.winrate("teacher")
                        student_str = f"{student_wr:.3f}" if student_wr is not None else "N/A"
                        teacher_str = f"{teacher_wr:.3f}" if teacher_wr is not None else "N/A"
                        sisters.append(f"- \"{attr}\"\n    Metric A average uplift: {student_str}\n    Metric B average uplift: {teacher_str}")

        # Format as string
        if not ancestors:
            ancestry_str = "No ancestry available (this is an initial attribute)."
        else:
            ancestry_str = "\n\n".join(ancestors)
            if sisters:
                ancestry_str += "\n\n=== Sibling features (other mutations from this lineage) ===\n"
                ancestry_str += "\n".join(sisters)

        return ancestor_attrs + sister_attrs, ancestry_str


    async def iterate_plan(
        self,
        seed_states: list[SeedState],
        baselines: dict[int, dict[str, list[BaselineRollout]]],
        n_neighbors: int = 8,
    ):
        to_send_messages = []
        messages_info = []

        for seed_idx, seed_state in enumerate(seed_states):
            seed_rng = Random(self.random_seed + seed_state.rng_index)
            # Collect all attributes in the previous step, not just the selected ones
            last_step_attributes = []
            for attribute, stats in seed_state.history[-1].items():
                student_wr = stats.winrate("student")
                teacher_wr = stats.winrate("teacher")
                # Only include if we have valid winrates
                if student_wr is not None and teacher_wr is not None:
                    last_step_attributes.append({
                        "attribute": attribute,
                        "student_winrate": student_wr,
                        "teacher_winrate": teacher_wr,
                    })
            
            for attribute, time_step in seed_state.state.items():
                # Get winrates for the current attribute
                stats = seed_state.history[time_step][attribute]

                # Get ancestry data and build exclusion set
                ancestor_attrs, ancestry_str = self._get_ancestry_data(
                    attribute=attribute,
                    time_step=time_step,
                    seed_state=seed_state,
                    baselines=baselines[seed_state.rng_index],
                )
                exclude_set = {attribute} | set(ancestor_attrs)

                other_attributes = [attr for attr in last_step_attributes if attr["attribute"] not in exclude_set]
                seed_rng.shuffle(other_attributes)

                lines = []
                for i, neighbor in enumerate(other_attributes[:n_neighbors], 1):
                    lines.append(
                        f"{i}. Feature: {neighbor['attribute']}\n"
                        f"   Metric A average uplift: {neighbor['student_winrate']:.3f}\n"
                        f"   Metric B average uplift: {neighbor['teacher_winrate']:.3f}"
                    )
                neighbor_data = "\n".join(lines) if lines else "No similar features available."

                planner_prompt = PLANNER_SYSTEM + "\n\n" + self.mutate_prompt.format(
                    num_plans=self.m_var,
                    cluster_summary=seed_state.cluster.summary,
                    current_data=self._get_original_data(stats=stats, baselines=baselines[seed_state.rng_index]),
                    ancestry_data=ancestry_str,
                    neighbor_data=neighbor_data,
                    direction_goal=DIRECTION_GOAL[self.direction],
                    bias_nudge=BIAS_NUDGE[self.direction],
                )

                to_send_messages.append(
                    ChatHistory.from_user(planner_prompt)
                )
                messages_info.append(
                    {
                        "parent": attribute,
                        "parent_time_step": time_step,
                        "seed_idx": seed_idx,
                    }
                )

            seed_state.history.append({})

        planner_responses = await self.hypothesis_planner.sample(to_send_messages, desc="Mutating", enable_cache=False)

        # parse responses
        for i, resp in enumerate(planner_responses):
            if resp is None:
                continue
            seed_idx = messages_info[i]["seed_idx"]
            attributes, reasoning = parse_json_response(resp)
            if isinstance(attributes, str):
                attributes = []
                logger.warning(f"Planner attributes did not parse as a list.\nResponse:\n{resp}\nReasoning:\n{reasoning}")
            elif isinstance(attributes, list):
                try:
                    attributes = [p.strip() for p in attributes]
                except Exception as e:
                    logger.warning(f"Planner attributes is not a list of strings.\nResponse:\n{resp}\nReasoning:\n{reasoning}")
                    logger.warning(f"Attributes: {attributes}")
                    attributes = [x for p in attributes for x in p][1:]

            if i < 3:
                logger.info(f"Planner reasoning:\n{reasoning}")
                logger.info(f"Planner prompt:\n{to_send_messages[i].get_first('user')}")
                logger.info(f"Planner attributes:\n{json.dumps(attributes, indent=4)}")

            meta = {
                "time_step": len(seed_states[seed_idx].history) - 1,
                "parent": messages_info[i]["parent"],
                "parent_time_step": messages_info[i]["parent_time_step"],
                "operation": "mutate",
                "planner_model": self.hypothesis_planner.curr_planner_model,
                "reasoning_effort": str(self.hypothesis_planner.reasoning) if self.hypothesis_planner.reasoning else None,
                "planner_prompt": to_send_messages[i].get_first("user"),
                "planner_reasoning": str(reasoning),
                "m_var": self.m_var,
            }

            for attribute in attributes:
                seed_states[seed_idx].history[-1][attribute] = AttributeStats(
                    attribute=attribute,
                    rollouts={},
                    meta=meta,
                )

        for seed_state in seed_states:
            # add in the original prompts from the state
            # reason: need to re-evaluate on different user prompts
            for attribute, time_step in seed_state.state.items():
                seed_state.history[-1][attribute] = AttributeStats(
                    attribute=attribute,
                    rollouts={},
                    meta=seed_state.history[time_step][attribute].meta,
                )

            logger.info(f"Seed {seed_state.index} mutated plus original plans: {len(seed_state.history[-1])}")
    
    def _dominates(self, p1: tuple, p2: tuple) -> bool:
        """Returns True if p1 dominates p2."""
        r1_a, r2_a = p1[2], p1[3]
        r1_b, r2_b = p2[2], p2[3]
        if self.direction == "plus":
            return (r1_a >= r1_b and r2_a <= r2_b) and (r1_a > r1_b or r2_a < r2_b)
        else:
            return (r1_a <= r1_b and r2_a >= r2_b) and (r1_a < r1_b or r2_a > r2_b)

    def _get_pareto_fronts(self, candidates: list[tuple]) -> list[list[tuple]]:
        """Sorts candidates into Pareto fronts."""
        n = len(candidates)
        domination_counts = [0] * n  # number of candidates that dominates each candidate
        dominated_sets = [[] for _ in range(n)]  # candiates that each candidate dominates
        
        # Calculate dominance
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(candidates[i], candidates[j]):
                    dominated_sets[i].append(j)
                elif self._dominates(candidates[j], candidates[i]):
                    domination_counts[i] += 1

        fronts = []
        current_front = [i for i, count in enumerate(domination_counts) if count == 0]
        
        rank = 0
        while current_front:
            fronts.append([candidates[i] for i in current_front])
            next_front = []
            for i in current_front:
                for j in dominated_sets[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        next_front.append(j)
            rank += 1
            current_front = next_front
            
        return fronts

    def _pareto_tiebreak(self, candidate: tuple) -> float:
        """smaller is better"""
        r1, _ = candidate[2], candidate[3]
        if self.direction == "plus":
            return -r1
        else:
            return r1

    @staticmethod
    def plot_candidate_stats(
        all_candidates: list[tuple],
        filtered_candidates: list[tuple],
        selected_candidates: list[tuple],
        student_threshold: float,
        teacher_threshold: float,
        direction: Literal["plus", "minus"],
    ) -> matplotlib.figure.Figure:
        """Creates a scatterplot of candidate statistics."""
        filtered_set = set(id(c) for c in filtered_candidates)
        selected_set = set(id(c) for c in selected_candidates)

        # Three categories: filtered out, passed but not selected, selected
        filtered_out = [(c[2], c[3]) for c in all_candidates
                        if c[2] is not None and c[3] is not None and id(c) not in filtered_set]
        passed_not_selected = [(c[2], c[3]) for c in all_candidates
                               if c[2] is not None and c[3] is not None
                               and id(c) in filtered_set and id(c) not in selected_set]
        selected = [(c[2], c[3]) for c in all_candidates
                    if c[2] is not None and c[3] is not None and id(c) in selected_set]

        fig, ax = plt.subplots(figsize=(10, 8))
        if filtered_out:
            ax.scatter([p[0] for p in filtered_out], [p[1] for p in filtered_out],
                       c='gray', alpha=0.4, label='Filtered out', marker='x', s=30)
        if passed_not_selected:
            ax.scatter([p[0] for p in passed_not_selected], [p[1] for p in passed_not_selected],
                       c='blue', alpha=0.5, label='Passed filter', marker='o', s=40)
        if selected:
            ax.scatter([p[0] for p in selected], [p[1] for p in selected],
                       c='red', alpha=0.9, label='Selected', marker='o', s=60)

        # Vertical line for student threshold
        ax.axvline(x=student_threshold, color='green', linestyle='--', linewidth=2,
                   label=f'Student threshold ({student_threshold:.3f})')
        # Horizontal line for teacher threshold
        ax.axhline(y=teacher_threshold, color='orange', linestyle='--', linewidth=2,
                   label=f'Teacher threshold ({teacher_threshold:.3f})')

        # Shade the rejection regions based on direction
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        if direction == "plus":
            # Reject low student scores (left of threshold)
            ax.axvspan(xlim[0], student_threshold, alpha=0.1, color='red')
            # Reject high teacher scores (above threshold)
            ax.axhspan(teacher_threshold, ylim[1], alpha=0.1, color='red')
        else:
            # Reject high student scores (right of threshold)
            ax.axvspan(student_threshold, xlim[1], alpha=0.1, color='red')
            # Reject low teacher scores (below threshold)
            ax.axhspan(ylim[0], teacher_threshold, alpha=0.1, color='red')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_xlabel('Student Winrate/RewardDiff')
        ax.set_ylabel('Teacher Winrate/RewardDiff')
        ax.set_title('Candidate Stats: Student vs Teacher')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        return fig

    def _calculate_student_threshold(
        self,
        winrate_values: list[float],
        min_items: int,
    ) -> float:
        """
        Calculate student score threshold for filtering.

        Uses percentile-based threshold (40th for plus, 60th for minus),
        but adjusts to guarantee at least min_items survive.
        """
        if not winrate_values:
            return 0.0

        # Calculate percentile-based threshold
        if self.direction == "plus":
            pct = float(np.percentile(winrate_values, 40))
            threshold = min(0.0, pct)
        else:
            pct = float(np.percentile(winrate_values, 60))
            threshold = max(0.0, pct)

        # Adjust threshold to guarantee at least min_items survive
        if len(winrate_values) > min_items:
            if self.direction == "plus":
                # Keep top min_items: threshold can be at most the min_items-th highest
                sorted_desc = sorted(winrate_values, reverse=True)
                min_items_threshold = sorted_desc[min_items - 1]
                threshold = min(threshold, min_items_threshold)
            else:
                # Keep bottom min_items: threshold can be at least the min_items-th lowest
                sorted_asc = sorted(winrate_values)
                min_items_threshold = sorted_asc[min_items - 1]
                threshold = max(threshold, min_items_threshold)

        return threshold

    def filter_by_student_scores(
        self,
        evaluate_results: dict[int, dict[str, AttributeStats]],
        min_items: int,
    ) -> tuple[dict[int, dict[str, AttributeStats]], list[float]]:
        """
        Filter evaluate_results by student score threshold before teacher evaluation.

        Returns:
            - Filtered list with same Rollout object references, so teacher eval
              modifies in place and changes propagate to the original evaluate_results.
            - List of student thresholds (one per seed) for use in plotting.
        """
        filtered_results: dict[int, dict[str, AttributeStats]] = {}
        student_thresholds = []

        for idx, seed_stats in evaluate_results.items():
            # Calculate student winrate for each attribute (without requiring teacher scores)
            attr_winrates: list[tuple[str, float | None]] = []

            for attribute, attribute_stats in seed_stats.items():
                attr_winrates.append((attribute, attribute_stats.winrate("student")))

            if not attr_winrates:
                filtered_results[idx] = {}
                student_thresholds.append(0.0)
                continue

            # Calculate threshold using percentile + min_items guarantee
            winrate_values = [wr for _, wr in attr_winrates if wr is not None]
            threshold = self._calculate_student_threshold(winrate_values, min_items)

            # Filter attributes by student score
            if self.direction == "plus":
                surviving_attrs = {attr for attr, wr in attr_winrates if wr is not None and wr >= threshold}
            else:
                surviving_attrs = {attr for attr, wr in attr_winrates if wr is not None and wr <= threshold}

            filtered_stats = {attr: s for attr, s in seed_stats.items() if attr in surviving_attrs}
            filtered_results[idx] = filtered_stats
            student_thresholds.append(threshold)

            logger.info(
                f"Seed {idx}: student pre-filter {len(seed_stats)} -> {len(filtered_stats)} "
                f"(threshold={threshold:.3f})"
            )

        return filtered_results, student_thresholds

    def update_pop_pareto(
        self, 
        seed_states: list[SeedState], 
        n_pop_target: int, 
        student_thresholds: list[float],
        strict: bool,
    ) -> dict[int, dict]:
        logger.info(f"Trying to update population to {n_pop_target} members using Pareto front selection.")

        candidates_by_seed: dict[int, dict] = {}

        for seed_idx, seed_state in enumerate(seed_states):
            all_candidates = []  # All candidates before filtering
            for attribute, stats in seed_state.history[-1].items():
                student_winrate = stats.winrate("student")
                teacher_winrate = stats.winrate("teacher")
                new_candidate = (
                    attribute,
                    stats.meta["time_step"],
                    student_winrate,
                    teacher_winrate,
                )
                all_candidates.append(new_candidate)

            # Student threshold was already applied in filter_by_student_scores
            student_threshold = student_thresholds[seed_idx]

            # Calculate teacher threshold
            if strict:
                teacher_threshold = 0.0
            else:
                valid_teacher_winrates = [c[3] for c in all_candidates if c[3] is not None]
                if self.direction == "plus":
                    teacher_pct = float(np.percentile(valid_teacher_winrates, 70)) if valid_teacher_winrates else float('inf')
                    teacher_threshold = max(0.0, teacher_pct)
                else:
                    teacher_pct = float(np.percentile(valid_teacher_winrates, 30)) if valid_teacher_winrates else float('-inf')
                    teacher_threshold = min(0.0, teacher_pct)

            logger.info(f"Student threshold: {student_threshold:.3f}")
            logger.info(f"Teacher threshold: {teacher_threshold:.3f}")

            # Filter candidates based on thresholds (keep those with valid winrates)
            # Note: student filtering already happened in filter_by_student_scores
            candidates = []
            for new_candidate in all_candidates:
                # Filter out candidates with None winrates
                if new_candidate[2] is None or new_candidate[3] is None:
                    continue

                # Filter out no-op candidates
                if math.isclose(new_candidate[2], 0.0, abs_tol=0.1) and math.isclose(new_candidate[3], 0.0, abs_tol=0.1):
                    continue

                if self.direction == "plus":
                    if new_candidate[3] > teacher_threshold:
                        continue
                    if new_candidate[2] < student_threshold:
                        continue
                else:
                    if new_candidate[3] < teacher_threshold:
                        continue
                    if new_candidate[2] > student_threshold:
                        continue
                candidates.append(new_candidate)

            print(
                "===============\n"
                f"After filtering, {len(candidates)}/{len(all_candidates)} candidates remain."
            )

            # Sort into Pareto Fronts
            fronts = self._get_pareto_fronts(candidates)

            # Select candidates from Pareto fronts (no diversity clustering needed - already done upfront)
            final_selection = []

            for front_idx, front in enumerate(fronts):
                if len(final_selection) >= n_pop_target:
                    break

                # Sort front by tiebreak (best first)
                front_sorted = sorted(front, key=self._pareto_tiebreak)
                remaining = n_pop_target - len(final_selection)
                selected_from_front = front_sorted[:remaining]
                final_selection.extend(selected_from_front)

                logger.info(f"Front {front_idx}: {len(front)} candidates, selected {len(selected_from_front)}.")

            # Store data for plotting (after selection is complete)
            candidates_by_seed[seed_state.rng_index] = {
                "all_candidates": all_candidates,
                "filtered_candidates": candidates,
                "selected_candidates": final_selection,
                "student_threshold": student_threshold,
                "teacher_threshold": teacher_threshold,
                "direction": self.direction,
            }

            # Update state
            seed_state.state = {
                attribute: time_step
                for attribute, time_step, _, _ in final_selection
            }

            logger.info(
                f"Updated Seed {seed_state.index} population to {len(seed_state.state)} members."
            )

        return candidates_by_seed


class EvoRunner(Runner):
    planner: EvoPlanner  # for type checker

    def __init__(
        self,
        seed_states: list[SeedState],
        planner: EvoPlanner,
        policy_model: GenerationModel,
        student_model: LocalRewardModel,
        teacher_model: RewardModel,
        n_baseline_rollouts: int,
        n_rewrite_rollouts: int,
        n_validate_rollouts: int,
        run_name: str | None = None,
        random_seed: int=42,
        runner_type_override: str | None = None,
    ):
        # Set before super().__init__ because parent accesses run_path which uses runner_type
        self._runner_type = runner_type_override or "evo"
        super().__init__(
            seed_states=seed_states,
            policy_model=policy_model,
            student_model=student_model,
            teacher_model=teacher_model,
            run_name=run_name,
            n_baseline_rollouts=n_baseline_rollouts,
            n_validate_rollouts=n_validate_rollouts,
        )
        self.planner = planner
        self.n_rewrite_rollouts = n_rewrite_rollouts
        self.random_seed = random_seed
        self.seed_rngs = {
            ss.rng_index: Random(self.random_seed + ss.rng_index) for ss in seed_states
        }

    @property
    def runner_type(self) -> str:
        return self._runner_type

    async def _cluster_seed(
        self,
        seed_state: SeedState,
        seed_evaluate_results: dict[str, AttributeStats],
    ) -> list[str]:
        """Cluster attributes and return the best (by student score) from each cluster."""
        all_attributes = list(seed_state.history[-1].keys())
        if not all_attributes:
            return []

        print(f"Seed {seed_state.index}: clustering {len(all_attributes)} candidates")
        _, cluster_indices = await self.planner.cluster_model.cluster_by_similarity(
            inputs=all_attributes,
            cosine_sim_threshold=self.planner.cosine_sim_threshold_evolution,
        )

        representatives = []
        for _, member_indices in cluster_indices.items():
            if len(member_indices) == 0:
                continue

            # Find member with best student score
            best_idx = None
            best_score = float('-inf') if self.planner.direction == "plus" else float('inf')

            for idx in member_indices:
                attr = all_attributes[idx]
                if attr in seed_evaluate_results:
                    score = seed_evaluate_results[attr].winrate("student")
                    if score is not None:
                        if self.planner.direction == "plus":
                            if score > best_score:
                                best_score = score
                                best_idx = idx
                        else:
                            if score < best_score:
                                best_score = score
                                best_idx = idx

            # Fallback to first if no valid scores found
            if best_idx is None:
                best_idx = member_indices[0]

            representatives.append(all_attributes[best_idx])

        print(f"Seed {seed_state.index}: selected {len(representatives)} representatives from {len(cluster_indices)} clusters")
        return representatives

    async def train_step(
        self,
        train_rewriter: RewriteModel,
        n_pop_target: int,
        train_batch_size: int,
        judge_train_first_n_rollouts: int,
        judge_train_first_n_user_prompts: int,
        strict: bool=False,
    ):
        print(f"[TRAIN STEP {self.step_count}] Writing new system prompts...")
        if self.step_count == 0:
            await self.planner.initial_plan(
                runner=self,
            )
        else:
            await self.planner.iterate_plan(
                seed_states=self.seed_states,
                baselines=self.baselines,
            )

        # Evaluate ALL attributes with student model
        evaluate_results: dict[int, dict[str, AttributeStats]] = {}

        for i, ss in enumerate(self.seed_states):
            all_attributes = list(ss.history[-1].keys())
            async with BiasEvaluator(rewrite_models=[train_rewriter], reward_model=self.student_model) as evaluator:
                user_prompts = self.seed_rngs[ss.rng_index].sample(
                    ss.cluster.train_prompts,
                    train_batch_size,
                )
                print(f"Seed {ss.index}: evaluating {len(all_attributes)} attributes")
                stats = await evaluator.evaluate_attributes(
                    user_prompts=user_prompts,
                    attributes=all_attributes,
                    baselines=self.baselines[ss.rng_index],
                    same_attrs=[SAME_ATTRS] * len(all_attributes),
                    n_rollouts=self.n_rewrite_rollouts,
                )

            assert len(list(stats.keys())) == 1
            assert list(stats.keys())[0] == train_rewriter.model_name
            stats = stats[train_rewriter.model_name]
            evaluate_results[ss.rng_index] = {
                k: AttributeStats(attribute=k, rollouts=v) for k, v in stats.items()
            }

        # Store ALL rollouts to history (needed for clustering and saving)
        for seed_state in self.seed_states:
            ss_idx = seed_state.rng_index
            stats = evaluate_results[ss_idx]
            for attribute, attribute_stats in stats.items():
                seed_state.history[-1][attribute].rollouts = attribute_stats.rollouts

        # Cluster candidates and pick best by student score from each cluster
        if self.step_count > 0:
            representative_attributes = await asyncio.gather(*[
                self._cluster_seed(seed_state, evaluate_results[seed_state.rng_index])
                for seed_state in self.seed_states
            ])
        else:
            representative_attributes = [list(ss.history[-1].keys()) for ss in self.seed_states]

        # Filter evaluate_results to only keep representatives
        representative_results: dict[int, dict[str, AttributeStats]] = {}
        for i, ss in enumerate(self.seed_states):
            reps = set(representative_attributes[i])
            representative_results[ss.rng_index] = {
                attr: stats for attr, stats in evaluate_results[ss.rng_index].items()
                if attr in reps
            }
            print(f"Seed {ss.index}: {len(evaluate_results[ss.rng_index])} -> {len(representative_results[ss.rng_index])} after clustering")

        # Filter by student scores to save compute on teacher evaluation
        filtered_evaluate_results, student_thresholds = self.planner.filter_by_student_scores(
            representative_results, min_items=n_pop_target
        )

        # Populate teacher_score on filtered rollouts in place
        # (Rollout objects are shared, so changes propagate to history)
        await self.teacher_model.judge_rollouts(
            evaluate_results=filtered_evaluate_results,
            baselines=self.baselines,
            first_n_rollouts=judge_train_first_n_rollouts,
            first_n_user_prompts=judge_train_first_n_user_prompts,
        )

        self.save_attribute_stats(
            direction=self.planner.direction,
            save_dir=self.run_path / f"step_{self.step_count}_stats",
        )

        logger.info(
            f"[TRAIN STEP {self.step_count}] Updating population. Before update: {len(self.seed_states[0].history[-1])}"
        )
        
        candidates_by_seed = self.planner.update_pop_pareto(
            seed_states=self.seed_states,
            n_pop_target=n_pop_target,
            student_thresholds=student_thresholds,
            strict=strict,
        )

        for ss_idx, candidates in candidates_by_seed.items():
            ss_history = None
            for ss in self.seed_states:
                if ss.rng_index == ss_idx:
                    ss_history = ss.history
            assert ss_history is not None

            with open(self.run_path / f"step_{self.step_count}_stats" / f"seed_{ss_idx}_candidates.json", "w") as f:
                json.dump(sorted([
                    {
                        "attribute": c[0],
                        "selected": c in candidates["selected_candidates"],
                        "student_winrate": c[2],
                        "teacher_winrate": c[3],
                        "time_step": c[1],
                        "parent": ss_history[-1][c[0]].meta.get("parent", None),
                        "parent_time_step": ss_history[-1][c[0]].meta.get("parent_time_step", None),
                    } 
                    for c in candidates["all_candidates"]
                ], key=lambda x: x["student_winrate"], reverse=self.planner.direction == "plus"), f, indent=4)

            fig = EvoPlanner.plot_candidate_stats(**candidates)
            fig.savefig(self.run_path / f"step_{self.step_count}_stats" / f"seed_{ss_idx}_pareto.pdf")
            plt.close(fig)

        logger.info(
            f"[TRAIN STEP {self.step_count}] finished; Current population: {len(self.seed_states[0].state)}"
        )
        self.step_count += 1
        self.planner.hypothesis_planner.step_planner_model()


    async def train(
        self,
        train_rewriter: RewriteModel,
        n_pop_target: list[int],
        train_batch_size: list[int],
        judge_train_first_n_rollouts: int,
        judge_train_first_n_user_prompts: int,
        start_from: int|None=None,
    ):
        t_steps = len(train_batch_size)
        assert len(n_pop_target) == t_steps

        for time_step in range(t_steps):
            if start_from is not None and time_step < start_from:
                for seed_state in self.seed_states:
                    with open(self.run_path / f"step_{time_step}_stats/seed_{seed_state.rng_index}.json", "r") as f:
                        seed_results = json.load(f)

                    seed_state.history.append(dict())
                    for item in seed_results:
                        attribute = item["attribute"]
                        attribute_rollouts = dict()
                        for user_prompt, rollouts in item["rollouts"].items():
                            attribute_rollouts[user_prompt] = [
                                Rollout(
                                    rewritten_response=r["rewritten_response"],
                                    baseline_response=r["baseline_response"],
                                    student_score=RewriteScore(
                                        score=r["student_score"]["score"],
                                        raw_score=r["student_score"].get("raw_score"),
                                        reasoning=r["student_score"].get("reasoning"),
                                        model_name=r["student_score"]["model_name"],
                                    ),
                                    teacher_score=RewriteScore(
                                        score=r["teacher_score"]["score"],
                                        raw_score=r["teacher_score"].get("raw_score"),
                                        reasoning=r["teacher_score"].get("reasoning"),
                                        model_name=r["teacher_score"]["model_name"],
                                    ) if r.get("teacher_score") else None,
                                )
                                if r is not None else None
                                for r in rollouts
                            ]

                        seed_state.history[-1][attribute] = AttributeStats(
                            attribute=attribute,
                            rollouts=attribute_rollouts,
                            meta=item.get("meta", {"time_step": time_step})
                        )


                # Calculate student thresholds from loaded data
                student_thresholds = []
                for seed_state in self.seed_states:
                    attr_winrates = []
                    for _, stats in seed_state.history[-1].items():
                        all_scores = []
                        for rollouts in stats.rollouts.values():
                            for r in rollouts:
                                if r is not None and r.student_score is not None and r.student_score.score is not None:
                                    all_scores.append(r.student_score.score)
                        if all_scores:
                            attr_winrates.append(float(np.mean(all_scores)))

                    threshold = self.planner._calculate_student_threshold(
                        attr_winrates, min_items=n_pop_target[time_step]
                    )
                    student_thresholds.append(threshold)

                self.planner.update_pop_pareto(
                    seed_states=self.seed_states,
                    n_pop_target=n_pop_target[time_step],
                    student_thresholds=student_thresholds,
                    strict=(time_step == t_steps - 1),
                )

                self.step_count += 1
                self.planner.hypothesis_planner.step_planner_model()
                self.save_attribute_stats(
                    direction=self.planner.direction,
                    save_dir=None,
                )

            else:
                await self.train_step(
                    train_rewriter=train_rewriter,
                    n_pop_target=n_pop_target[time_step],
                    train_batch_size=train_batch_size[time_step],
                    judge_train_first_n_rollouts=judge_train_first_n_rollouts,
                    judge_train_first_n_user_prompts=judge_train_first_n_user_prompts,
                    strict=(time_step == t_steps - 1),
                )
