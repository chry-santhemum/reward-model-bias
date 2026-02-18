"""Reward model wrapper for both LLM Judge and classifier."""

import asyncio
from tqdm.auto import tqdm
from typing import Sequence, Literal, Any, Callable
from abc import ABC, abstractmethod
from loguru import logger
import torch

from caller import ChatHistory
from api_models import JudgeModel, RatingResult, ComparisonResult
from state import AttributeStats, RewriteScore, BaselineRollout
from utils import load_model, find_executable_batch_size


class RewardModel(ABC):
    model_name: str
    type: Literal["api", "local"]
    batch_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "type": self.type,
            "batch_size": self.batch_size
        }

    @abstractmethod
    async def async_rate(self, chat_histories: Sequence[ChatHistory], use_tqdm: bool) -> list[RatingResult]:
        pass

    @abstractmethod
    async def async_compare(
        self, 
        chat_histories_A: Sequence[ChatHistory], 
        chat_histories_B: Sequence[ChatHistory],
        use_tqdm: bool
    ) -> list[ComparisonResult]:
        pass

    async def judge_rollouts(
        self,
        evaluate_results: dict[int, dict[str, AttributeStats]],
        baselines: dict[int, dict[str, list[BaselineRollout]]],
        first_n_rollouts: int,      # 0 means all
        first_n_user_prompts: int,  # 0 means all
        use_tqdm: bool = True
    ) -> None:
        """Populates teacher_score on rollouts in validation_results in place."""

        chat_histories_A = []  # rewritten response
        chat_histories_B = []  # baseline response
        judge_tasks_info = []

        # load async_compare args
        for seed_idx, validation_result in evaluate_results.items():
            for attribute, attribute_stats in validation_result.items():
                user_prompt_count = 0
                for user_prompt, rollouts in attribute_stats.rollouts.items():
                    if first_n_user_prompts > 0 and user_prompt_count >= first_n_user_prompts:
                        break
                    user_prompt_count += 1

                    baseline_rollouts = baselines[seed_idx][user_prompt]
                    rollout_count = 0
                    for rollout_idx, rollout in enumerate(rollouts):
                        if first_n_rollouts > 0 and rollout_count >= first_n_rollouts:
                            break
                        if rollout is None or baseline_rollouts[rollout_idx] is None:
                            continue
                        rollout_count += 1

                        rewritten_chat = ChatHistory.from_user(user_prompt).add_assistant(rollout.rewritten_response)
                        baseline_chat = ChatHistory.from_user(user_prompt).add_assistant(baseline_rollouts[rollout_idx].response)

                        chat_histories_A.append(rewritten_chat)
                        chat_histories_B.append(baseline_chat)

                        judge_tasks_info.append(
                            {
                                "seed_state_idx": seed_idx,
                                "attribute": attribute,
                                "user_prompt": user_prompt,
                                "rollout_idx": rollout_idx,
                            }
                        )

        logger.info(f"Running {len(judge_tasks_info)} judge tasks...")
        if self.type == "api":
            judge_tasks_results = await self.async_compare(
                chat_histories_A=chat_histories_A,
                chat_histories_B=chat_histories_B,
                use_tqdm=use_tqdm,
            )

            for comparison_result, judge_task_info in zip(judge_tasks_results, judge_tasks_info, strict=True):
                seed_state_idx = judge_task_info["seed_state_idx"]
                attribute = judge_task_info["attribute"]
                user_prompt = judge_task_info["user_prompt"]
                rollout_idx = judge_task_info["rollout_idx"]
                rewritten_raw_score = comparison_result.raw_score_A

                teacher_score = RewriteScore(
                    score=comparison_result.score_diff,
                    raw_score=rewritten_raw_score,
                    reasoning=comparison_result.reasoning[0] if comparison_result.reasoning is not None else None,
                    model_name=self.model_name,
                )

                # Update the rollout in place
                rollout = evaluate_results[seed_state_idx][attribute].rollouts[user_prompt][rollout_idx]
                rollout.teacher_score = teacher_score

        elif self.type == "local":
            judge_tasks_results = await self.async_rate(chat_histories_A, use_tqdm=use_tqdm)

            for rating_result, judge_task_info in zip(judge_tasks_results, judge_tasks_info, strict=True):
                seed_state_idx = judge_task_info["seed_state_idx"]
                attribute = judge_task_info["attribute"]
                user_prompt = judge_task_info["user_prompt"]
                rollout_idx = judge_task_info["rollout_idx"]

                baseline_score = baselines[seed_state_idx][user_prompt][rollout_idx].scores[self.model_name]

                teacher_score = RewriteScore(
                    score=rating_result.score - baseline_score,
                    raw_score=rating_result.score,
                    reasoning=None,
                    model_name=self.model_name,
                )
                rollout = evaluate_results[seed_state_idx][attribute].rollouts[user_prompt][rollout_idx]
                rollout.teacher_score = teacher_score


class LocalRewardModel(RewardModel):
    def __init__(
        self,
        model_name: str,
        devices: list[str],
        batch_size_per_device: int,
        attn_implementation: str="sdpa",
        bias: Callable[[ChatHistory], float] | None = None,
        share_weights_with: "LocalRewardModel | None" = None,
        score_name: str | None = None,
    ):
        assert len(devices) > 0
        self._base_model_name = model_name
        if score_name is not None:
            self.model_name = model_name + "_" + score_name
        else:
            self.model_name = model_name
        self.type = "local"
        self.batch_size_per_device = batch_size_per_device
        self.devices = devices
        self.attn_implementation = attn_implementation
        self.bias = bias

        if share_weights_with is not None:
            assert share_weights_with._base_model_name == model_name, \
                f"Cannot share weights: base models differ ({share_weights_with._base_model_name} vs {model_name})"
            print(f"Sharing model weights from {share_weights_with.model_name}")
            self.models = share_weights_with.models
            self.tokenizer = share_weights_with.tokenizer
        else:
            self.models = []
            self.tokenizer = None
            for d in devices:
                print(f"Loading model {model_name} on device {d}...")
                model, tokenizer = load_model(
                    model_name=model_name,
                    model_type="reward",
                    device=d,
                    attn_implementation=attn_implementation
                )
                self.models.append(model)
                if self.tokenizer is None:
                    self.tokenizer = tokenizer
    
    @property
    def batch_size(self) -> int:
        return self.batch_size_per_device * len(self.devices)

    def to_dict(self) -> dict[str, Any]:
        params = super().to_dict()
        params["_base_model_name"] = self._base_model_name
        params["attn_implementation"] = self.attn_implementation
        params["devices"] = self.devices
        params["has_bias"] = self.bias is not None
        return params

    def rate_one_model(
        self, 
        model_index: int, 
        chat_histories: Sequence[ChatHistory], 
        use_tqdm: bool=True,
    ) -> list[RatingResult]:
        model = self.models[model_index]
        rating_results = []
        if len(chat_histories) == 0:
            return []

        # inner loop: should only happen in rare cases.
        @find_executable_batch_size(starting_batch_size=self.batch_size_per_device)
        def rate_one_model_inner(batch_size: int, chats_inner: list[ChatHistory]) -> list[RatingResult]:
            results_inner = []
            inner_pbar = range(0, len(chats_inner), batch_size)
            for inner_idx in inner_pbar:
                sub_chats = chats_inner[inner_idx : inner_idx + batch_size]
                sub_inputs = [chat.remove_system().to_openai_messages() for chat in sub_chats]
                input_ids = self.tokenizer.apply_chat_template(  # type: ignore
                    sub_inputs, tokenize=True, return_tensors="pt", padding=True, padding_side="right",
                ).to(model.device)

                with torch.no_grad():
                    scores_tensor = model(**input_ids).logits.squeeze(-1)

                scores_list = scores_tensor.tolist()
                for score, chat in zip(scores_list, sub_chats):
                    final_score = float(score)
                    if self.bias is not None:
                        final_score += self.bias(chat)
                    results_inner.append(RatingResult(score=final_score, reasoning=None))
            return results_inner

        # outer loop: break down to roughly correct batch size
        outer_range = range(0, len(chat_histories), self.batch_size_per_device)
        if use_tqdm:
            outer_pbar = tqdm(
                outer_range, 
                desc=f"RM {model_index} ({self.models[model_index].device})",
                position=model_index,
            )
        else:
            outer_pbar = outer_range
        
        for i in outer_pbar:
            batch = chat_histories[i : i + self.batch_size_per_device]
            results_inner = rate_one_model_inner(chats_inner=batch)  # type: ignore
            rating_results.extend(results_inner)

        return rating_results

    async def async_rate(self, chat_histories, use_tqdm=True):
        n_models = len(self.models)
        n_chats = len(chat_histories)
        tasks = []

        for i in range(n_models):
            if i < n_models - 1:
                tasks.append(asyncio.to_thread(
                    self.rate_one_model, model_index=i,
                    chat_histories=chat_histories[i*n_chats // n_models : (i+1)*n_chats // n_models],
                    use_tqdm=use_tqdm,
                ))
            else:
                tasks.append(asyncio.to_thread(
                    self.rate_one_model, model_index=i,
                    chat_histories=chat_histories[i*n_chats // n_models :],
                    use_tqdm=use_tqdm,
                ))

        if tasks:
            task_results = await asyncio.gather(*tasks)
        
        combined_results: list[RatingResult] = []
        for task_result in task_results:
            combined_results.extend(task_result)

        return combined_results

    async def async_compare(self, chat_histories_A, chat_histories_B, use_tqdm=True):
        print("Rating chat histories A...")
        results_A = await self.async_rate(chat_histories_A, use_tqdm=use_tqdm)
        print("Rating chat histories B...")
        results_B = await self.async_rate(chat_histories_B, use_tqdm=use_tqdm)
        compare_results = []

        for result_A, result_B in zip(results_A, results_B):
            if result_A.score is None or result_B.score is None:
                compare_results.append(ComparisonResult(winner=None, reasoning=None))
            else:
                diff = result_A.score - result_B.score
                winner = "A" if diff > 0 else "B" if diff < 0 else "Tie"
                compare_results.append(ComparisonResult(winner=winner, reasoning=None, score_diff=diff, raw_score_A=result_A.score, raw_score_B=result_B.score))
        
        return compare_results


class APIRewardModel(RewardModel):
    def __init__(self, model_name: str, max_par: int, **kwargs):
        self.model_name = model_name
        self.type = "api"
        self.max_par = max_par
        self.kwargs = kwargs
        self.model = JudgeModel(model_name=model_name, max_par=max_par, **kwargs)
    
    @property
    def batch_size(self) -> int:
        return self.max_par

    def to_dict(self) -> dict[str, Any]:
        params = super().to_dict()
        params.update(self.model.to_dict())
        return params

    async def async_rate(self, chat_histories, use_tqdm=True):
        return await self.model.judge_absolute(chat_histories=chat_histories, use_tqdm=use_tqdm)

    async def async_compare(self, chat_histories_A, chat_histories_B, use_tqdm=True):
        return await self.model.judge_relative(chat_histories_A=chat_histories_A, chat_histories_B=chat_histories_B, use_tqdm=use_tqdm)
