

import time
import json
import asyncio
from tqdm.auto import tqdm
from dataclasses import dataclass, asdict
from typing import Any, Literal
from pathlib import Path
from loguru import logger

from caller import ChatHistory
from state import Cluster, BaselineRollout
from api_models import GenerationModel
from reward_models import LocalRewardModel



async def evaluate_baselines_from_prompts(
    user_prompts: list[str],
    policy_model: GenerationModel,
    reward_model: LocalRewardModel,
    n_rollouts: int,
    save_dir: Path | None = None,
) -> dict[str, list[BaselineRollout]]:
    start_time = time.time()

    # Sample responses
    chat_histories = []
    for user_prompt in user_prompts:
        chat_histories.extend([ChatHistory.from_user(user_prompt) for _ in range(n_rollouts)])
    responses = await policy_model.sample(chat_histories)

    # Get rewards
    chats_with_responses = []
    policy_model_names = []
    for chat, resp in zip(chat_histories, responses, strict=True):
        if (
            resp is None
            or resp.first_response is None
            or resp.finish_reason != "stop"
        ):
            logger.warning(f"Policy model failed to sample. User prompt:\n{chat.get_first('user')}\nResponse:\n{resp}")
            continue

        chats_with_responses.append(chat.add_assistant(resp.first_response.strip()))
        policy_model_names.append(resp.model)

    scores = await reward_model.async_rate(chats_with_responses, use_tqdm=True)

    # Organize results
    baselines: dict[str, list[BaselineRollout]] = {}
    for chat, score, name in zip(chats_with_responses, scores, policy_model_names):
        result = BaselineRollout(
            policy_model=name,
            response=chat.get_first("assistant"),
            scores={reward_model.model_name: score.score} if score.score is not None else {},
        )
        if chat.get_first("user") not in baselines:
            baselines[chat.get_first("user")] = []
        baselines[chat.get_first("user")].append(result)

    # Save results
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / "baselines.json"
        with open(path, "w") as f:
            json.dump({user: [asdict(r) for r in rollouts] for user, rollouts in baselines.items()}, f, indent=4, sort_keys=True)

    logger.info(f"Evaluated {len(user_prompts)} prompts x {n_rollouts} rollouts in {(time.time() - start_time):.2f} seconds")
    return baselines
    
    
async def evaluate_baselines(
    cluster: Cluster,
    split: Literal["train", "val"],
    policy_model: GenerationModel,
    reward_model: LocalRewardModel,
    n_rollouts: int,
    save_dir: Path | None = None,
) -> dict[str, list[BaselineRollout]]:
    start_time = time.time()

    ds_name = Path(cluster.data_path).parent.name
    if save_dir is None:
        path = Path("data/baselines") / ds_name / split / f"cluster_{cluster.index}.json"
    else:
        path = save_dir / f"{ds_name}_{split}_cluster_{cluster.index}.json"

    baselines: dict[str, list[BaselineRollout]] = {}
    
    if path.exists():
        print(f"{ds_name}_{cluster.index} {split} baselines already exist")
        # Load existing results, and only evaluate the entries
        # that don't have the score for this reward model 
        with open(path, "r") as f:
            existing_results = json.load(f)
        
        # convert to dataclass
        for user, rollouts in existing_results.items():
            baselines[user] = [BaselineRollout(**r) for r in rollouts]
        
        chats_with_responses = []
        indices = []
        for user, rollouts in baselines.items():
            for i, r in enumerate(rollouts):
                # If biased model and base scores exist, derive score
                if reward_model.bias is not None:
                    if reward_model._base_model_name not in r.scores:
                        raise ValueError(f"Rewards for the base RM {reward_model._base_model_name} was not evaluated for {user}")
                    base_score = r.scores[reward_model._base_model_name]
                    chat = ChatHistory.from_user(user).add_assistant(r.response)
                    r.scores[reward_model.model_name] = base_score + reward_model.bias(chat)
                    continue

                elif reward_model.model_name in r.scores:
                    # Only take the scores that are already there
                    # when the RM is unadorned
                    continue

                chats_with_responses.append(ChatHistory.from_user(user).add_assistant(r.response))
                indices.append(i)

        
        # get rewards
        scores = await reward_model.async_rate(chats_with_responses, use_tqdm=True)
        for chat, r_idx, score in zip(chats_with_responses, indices, scores, strict=True):
            result_dict = baselines[chat.get_first("user")][r_idx]
            result_dict.scores[reward_model.model_name] = score.score

        with open(path, "w") as f:
            json.dump({user: [asdict(r) for r in user_rollouts] for user, user_rollouts in baselines.items()}, f, indent=4, sort_keys=True)
        
    else:
        print(f"Sampling {ds_name}_{cluster.index} {split} baselines")
        # First sample, then evaluate
        chat_histories = []
        if split == "train":
            user_prompts = cluster.train_prompts
        else:
            user_prompts = cluster.val_prompts

        for user_prompt in user_prompts:
            chat_histories.extend([ChatHistory.from_user(user_prompt) for _ in range(n_rollouts)])
        responses = await policy_model.sample(chat_histories)
        
        # get rewards
        chats_with_responses = []
        policy_model_names = []
        for chat, resp in zip(chat_histories, responses, strict=True):
            if (
                resp is None 
                or resp.first_response is None
                or resp.finish_reason != "stop"
            ):
                logger.warning(f"Policy model failed to sample. User prompt:\n{chat.get_first('user')}\nResponse:\n{resp}")
                continue

            chats_with_responses.append(chat.add_assistant(resp.first_response.strip()))
            policy_model_names.append(resp.model)

        scores = await reward_model.async_rate(chats_with_responses, use_tqdm=True)

        # save results
        for chat, score, name in zip(chats_with_responses, scores, policy_model_names):
            result = BaselineRollout(
                policy_model=name,
                response=chat.get_first("assistant"),
                scores={reward_model.model_name: score.score} if score.score is not None else {},
            )
            if chat.get_first("user") not in baselines:
                baselines[chat.get_first("user")] = []
            baselines[chat.get_first("user")].append(result)

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({user: [asdict(r) for r in rollouts] for user, rollouts in baselines.items()}, f, indent=4, sort_keys=True)
        
    logger.info(f"Evaluated {ds_name}_{cluster.index} {split} baselines in {(time.time() - start_time):.2f} seconds")
    return baselines
