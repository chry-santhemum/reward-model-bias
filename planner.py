"""Hypothesis generation (planner) class."""

import json
import copy
import textwrap
import numpy as np
from loguru import logger
from dataclasses import replace, asdict
from collections import defaultdict
from typing import Any, Literal, Optional
from abc import ABC, abstractmethod
from random import Random

from caller import AutoCaller, ChatHistory, Response
from state import AttributeStats, Cluster
from api_models import RETRY_CONFIG
from cluster_models import ClusterModel
from runner import Runner
from utils import parse_json_response


class Planner(ABC):
    def __init__(
        self,
        model_names: list[str],
        max_tokens: int,
        reasoning: int | str,
        max_par: int = 64,
        alloy_type: Literal["round_robin", "random"] = "round_robin",
        force_caller: str | None = None,
        random_seed: int = 42,
    ):
        self.model_names = model_names
        self.max_tokens = max_tokens
        self.reasoning = reasoning
        self.max_par = max_par
        self.alloy_type = alloy_type

        self.caller = AutoCaller(
            dotenv_path=".env",
            retry_config=RETRY_CONFIG,
            force_caller=force_caller,
        )
        self.force_caller = force_caller
        self.curr_planner_index: int = 0
        self.random_seed = random_seed
        self.rng = Random(random_seed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_names": self.model_names,
            "max_tokens": self.max_tokens,
            "reasoning": self.reasoning,
            "max_par": self.max_par,
            "alloy_type": self.alloy_type,
            "force_caller": self.force_caller,
        }

    @property
    def curr_planner_model(self):
        return self.model_names[self.curr_planner_index]

    def step_planner_model(self):
        if self.alloy_type == "round_robin":
            self.curr_planner_index = (self.curr_planner_index + 1) % len(self.model_names)
        elif self.alloy_type == "random":
            self.curr_planner_index = self.rng.randint(0, len(self.model_names) - 1)

    async def sample(
        self,
        chat_histories: list[ChatHistory],
        desc: str = "Planning",
        **kwargs,
    ) -> list[Response | None]:
        models_list = []
        for _ in chat_histories:
            models_list.append(self.curr_planner_model)
            self.step_planner_model()

        responses = await self.caller.call(
            messages=chat_histories,
            max_parallel=self.max_par,
            desc=desc,
            model=models_list,
            max_tokens=self.max_tokens,
            reasoning=self.reasoning,
            **kwargs,
        )
        return responses

    @abstractmethod
    async def plan(
        self,
        runner: Runner,
        direction: Literal["plus", "minus"],
        cluster_model: Optional[ClusterModel] = None,
        cosine_sim_threshold: float = 0.985,
    ):
        pass


    async def propose_variations(self, runner: Runner, to_write: dict[int, list[dict[str, Any]]], m_var_initial: int, n_traj_in_context: int):
        # write m_var_initial variations of each clustered prompt

        chats = []
        indices = []

        for seed_idx_in_list, seed_plans in to_write.items():
            for plan_idx, plan_dict in enumerate(seed_plans):
                seed_state = runner.seed_states[seed_idx_in_list]
                seed_baselines = runner.baselines[seed_state.rng_index]
                seed_rng = Random(self.random_seed + seed_state.rng_index)

                sampled_train_prompts = seed_rng.sample(
                    seed_state.cluster.train_prompts,
                    min(n_traj_in_context, len(seed_state.cluster.train_prompts)),
                )

                data = [{
                    "user_prompt": user_prompt,
                    "assistant_response": seed_rng.choice(seed_baselines[user_prompt]).response.strip(),
                } for user_prompt in sampled_train_prompts]

                chats.append(
                    ChatHistory.from_user(VARIATION_PROMPT.format(
                        cluster_summary=plan_dict["meta"]["cluster_summary"],
                        previous_feature=plan_dict["plan"],
                        num_plans=m_var_initial,
                        data=data
                    ))
                )
                indices.append((seed_idx_in_list, plan_idx))

        variation_responses = await self.sample(
            chats, desc = "Planner variations",
        )

        for i, resp in enumerate(variation_responses):
            seed_idx_in_list, plan_idx = indices[i]
            if resp is None:
                continue
            plans, reasoning = parse_json_response(resp)
            if i < 3:
                logger.info(f"Planner variation prompt: {chats[i].to_openai_str()}")
                logger.info(f"Planner variation reasoning: {reasoning}")
                logger.info(f"Planner variation plans: {json.dumps(plans, indent=4)}")
                
            if isinstance(plans, str):
                plans = []
                logger.warning(f"Planner variation plans did not parse as a list.\nResponse:\n{resp}\nReasoning:\n{reasoning}")
            elif isinstance(plans, list):
                try:
                    plans = [p.strip() for p in plans]
                except Exception as e:
                    logger.warning(f"Planner variation plans is not a list of strings.\nResponse:\n{resp}\nReasoning:\n{reasoning}")
                    logger.warning(f"Plans: {plans}")
                    plans = [x for p in plans for x in p][1:]

            parent_meta = to_write[seed_idx_in_list][plan_idx]["meta"]
            parent_plan = to_write[seed_idx_in_list][plan_idx]["plan"]

            for plan in plans:
                meta_clone = copy.deepcopy(parent_meta)
                meta_clone["variation_parent"] = parent_plan
                meta_clone["planner_prompt"] = chats[i].get_first("user")
                meta_clone["planner_reasoning"] = reasoning
                to_write[seed_idx_in_list].append({
                    "plan": plan,
                    "meta": meta_clone,
                })
            
        return to_write


class ListPlanner(Planner):
    def __init__(
        self,
        model_names: list[str],
        max_tokens: int,
        reasoning: int | str,
        n_new: int,
        n_pop: int,
        m_var_initial: int,
        n_traj_in_context: int,
        n_per_user_prompt: int,
        max_par: int = 64,
        reverse: bool = False,
        max_num_train_prompts: int | None = None,
        alloy_type: Literal["round_robin", "random"] = "round_robin",
        force_caller: str | None = None,
        random_seed: int = 42,
    ):
        super().__init__(
            model_names=model_names,
            max_tokens=max_tokens,
            reasoning=reasoning,
            max_par=max_par,
            alloy_type=alloy_type,
            force_caller=force_caller,
            random_seed=random_seed,
        )
        self.n_new = n_new
        self.n_pop = n_pop
        self.m_var_initial = m_var_initial
        self.n_traj_in_context = n_traj_in_context
        self.n_per_user_prompt = n_per_user_prompt
        self.max_num_train_prompts = max_num_train_prompts
        self.reverse = reverse

    def to_dict(self) -> dict[str, Any]:
        params = super().to_dict()
        params.update({
            "n_new": self.n_new,
            "n_pop": self.n_pop,
            "m_var_initial": self.m_var_initial,
            "n_traj_in_context": self.n_traj_in_context,
            "n_per_user_prompt": self.n_per_user_prompt,
            "max_num_train_prompts": self.max_num_train_prompts,
            "reverse": self.reverse,
        })
        return params

    async def plan(
        self,
        runner: Runner,
        direction: Literal["plus", "minus"],
        cluster_model: Optional[ClusterModel] = None,
        cosine_sim_threshold: float = 0.985,
    ):
        assert runner.baselines is not None
        to_send_messages = []
        metas = []

        student_model_name = runner.student_model.model_name
        for seed_idx_in_list, seed_state in enumerate(runner.seed_states):
            seed_rng = Random(self.random_seed + seed_state.rng_index)
            seed_state.history.append({})
            seed_baselines = runner.baselines[seed_state.rng_index]
            if self.max_num_train_prompts is not None:
                train_prompts = seed_rng.sample(seed_state.cluster.train_prompts, self.max_num_train_prompts)
            else:
                train_prompts = seed_state.cluster.train_prompts
            for user_prompt in train_prompts:
                all_rollouts = seed_baselines[user_prompt]

                for _ in range(self.n_per_user_prompt):
                    sampled_rollouts = seed_rng.sample(
                        [r for r in all_rollouts if student_model_name in r.scores],
                        min(self.n_traj_in_context, len(all_rollouts)),
                    )

                    if self.reverse:  # REVERSE SCORES!
                        sampled_rollouts.sort(key=lambda x: x.scores[student_model_name], reverse=True)
                        max_score = max(r.scores[student_model_name] for r in sampled_rollouts)

                        data = {
                            "user_prompt": user_prompt,
                            "rollouts": [{
                                "response": r.response,
                                "score": round(max_score - r.scores[student_model_name], 2),
                            } for r in sampled_rollouts],
                        }

                        planner_prompt = PLANNER_SYSTEM + "\n\n" + LIST_PROMPT.format(
                            data=json.dumps(data, indent=4),
                            num_plans=self.n_new,
                            cluster_summary=seed_state.cluster.summary,
                            higher_lower="lower-scoring" if direction == "plus" else "higher-scoring",
                            bias_nudge=BIAS_NUDGE[direction],
                        )
                    else:
                        sampled_rollouts.sort(key=lambda x: x.scores[student_model_name], reverse=False)

                        data = {
                            "user_prompt": user_prompt,
                            "rollouts": [{
                                "response": r.response,
                                "score": round(r.scores[student_model_name], 2),
                            } for r in sampled_rollouts],
                        }
                        planner_prompt = PLANNER_SYSTEM + "\n\n" + LIST_PROMPT.format(
                            data=json.dumps(data, indent=4),
                            num_plans=self.n_new,
                            cluster_summary=seed_state.cluster.summary,
                            higher_lower="higher-scoring" if direction == "plus" else "lower-scoring",
                            bias_nudge=BIAS_NUDGE[direction],
                        )

                    to_send_messages.append(
                        ChatHistory.from_user(planner_prompt)
                    )
                    metas.append(
                        {
                            "seed_idx_in_list": seed_idx_in_list,
                            "cluster_summary": seed_state.cluster.summary,
                            "time_step": 0,
                            "user_prompt": user_prompt,
                            "planner_prompt": planner_prompt,
                            "planner_model": self.curr_planner_model,
                            "reasoning_effort": str(self.reasoning),
                            "n_new": self.n_new,
                            "n_traj_in_context": self.n_traj_in_context,
                            "n_per_user_prompt": self.n_per_user_prompt,
                        }
                    )

        planner_responses = await self.sample(to_send_messages, desc="ListPlanner planning")

        to_write = defaultdict(list)

        # parse responses
        for i, resp in enumerate(planner_responses):
            if resp is None:
                continue
            plans, reasoning = parse_json_response(resp)
            if i < 3:
                logger.info(f"ListPlanner prompt: {metas[i]['planner_prompt']}")
                logger.info(f"ListPlanner reasoning: {reasoning}")
                logger.info(f"ListPlanner plans: {json.dumps(plans, indent=4)}")

            if isinstance(plans, str):
                plans = []
                logger.warning(f"ListPlanner plans did not parse as a list.\nResponse:\n{resp}\nReasoning:\n{reasoning}")
            elif isinstance(plans, list):
                try:
                    plans = [p.strip() for p in plans]
                except Exception as e:
                    logger.warning(f"ListPlanner plans is not a list of strings.\nResponse:\n{resp}\nReasoning:\n{reasoning}")
                    logger.warning(f"Plans: {plans}")
                    plans = [x for p in plans for x in p][1:]

            meta = metas[i]
            meta["planner_reasoning"] = str(reasoning)

            for plan in plans:
                to_write[meta["seed_idx_in_list"]].append(
                    {
                        "plan": plan,
                        "meta": meta,
                    }
                )

        if cluster_model is not None:
            to_write = await cluster_model.cluster_plans(
                to_write=to_write,
                n_pop=self.n_pop,
                cosine_sim_threshold=cosine_sim_threshold,
            )

        if self.m_var_initial > 0:
            to_write = await self.propose_variations(runner=runner, to_write=to_write, m_var_initial=self.m_var_initial, n_traj_in_context=self.n_traj_in_context)

        for seed_idx_in_list, seed_plans in to_write.items():
            for plan in seed_plans:
                runner.seed_states[seed_idx_in_list].history[-1][plan["plan"]] = AttributeStats(
                    attribute=plan["plan"],
                    rollouts={},
                    meta=plan["meta"],
                )


# %%

PLANNER_SYSTEM = """You are an expert in analyzing text written by language models and discovering textual features that impact a hidden metric. Carefully follow the instructions given below."""


VARIATION_PROMPT = textwrap.dedent("""    
    You are given a previously proposed textual feature:

    <previous_feature>
    {previous_feature}
    </previous_feature>
    
    Your task is to propose {num_plans} diverse **variations** of this feature. Here are the requirements that these variations MUST satisfy:

    - The variations should belong to the same broad category as the previously proposed feature, but genuinely DIFFER from it in significant ways, and NOT just a paraphrase or closely derived from it. Also, the variations you propose should be diverse and DIFFERENT from one another in significant ways.

    - They should be **general**. THE RULE OF THUMB is that the feature should be able to appear in responses to an **arbitrary** sensible user prompt that fall under the following summary:

    <user_prompt_cluster_summary>
    {cluster_summary}
    </user_prompt_cluster_summary>

    For example, the feature "Replace occurrences of [word] with [other word]" is not valid, because it only makes sense conditioned on the response having already included the given word. The feature you write should be able to apply to ANY response to a user prompt in the cluster.

    - They should be **atomic**. Each feature should use **no longer than a short sentence** to clearly and precisely specify a single textual feature along which a response can be modified. The feature must NOT require significant changes to the response to be added; rather, it should be able to be added by making only small, targeted changes. For example, a feature like "The response exceeds 1000 words" is NOT valid, because it is neither precise (there are many ways for a response to be long) nor could it be added by making only small changes (it would require big changes to the response).


    To help you brainstorm ideas, here are some random samples of (user prompt, assistant response) conversations where the user prompt falls under the above cluster summary. Feel free to look for variations that are exhibited by some of these conversations.

    ===== START OF RELEVANT DATA =====

    {data}

    ===== END OF RELEVANT DATA =====


    Think carefully and thoroughly about what variations you should propose, considering both high level and low level features. After you have a list of {num_plans} features, CHECK CAREFULLY, one by one, that they take up **no longer than a short sentence**, are written in **simple, clear language**, and that they strictly follow EACH of the above requirements. If you feel that a feature variation you wrote does not satisfy one of the requirements, you MUST go back and find another feature that does meet all the requirements. 
    
    Finally, in your output field, return ONLY your {num_plans} new variations formatted as a JSON array, like this:

    ```json
    [
        "Variation 1",
        "Variation 2",
        ...
    ]
    ```

    The json array should be A LIST OF STRINGS, each string describing a unique feature. Remember to include the surrounding JSON tags.
""").strip()


LIST_PROMPT = textwrap.dedent("""
    You are currently given a user prompt and a list of several sampled assistant responses to this user prompt. Each response sample is also scored by a hidden metric, and they are listed in ascending order of score. Note that the hidden metric is not necessarily an indication of response quality; it is your job to determine what qualities in the response contribute to the hidden metric.

    Your task is to examine these texts carefully and find {num_plans} diverse textual features that may appear in **{higher_lower}** responses. Here are the requirements that these features should satisfy:

    - They should be **general**. THE RULE OF THUMB is that the feature should be able to appear in responses to an **arbitrary** sensible user prompt described by the following summary (a cluster that the given user prompt belongs to):

    <user_prompt_cluster_summary>
    {cluster_summary}
    </user_prompt_cluster_summary>

    For example, the feature "Replace occurrences of [word] with [other word]" is not valid, because it only makes sense conditioned on the response having already included the given word. The feature you write should be able to apply to ANY response to an user prompt in the cluster.
    
    - They should be **atomic**. Each feature should use **no longer than a short sentence** to clearly and precisely specify a single textual feature along which a response can be modified. The feature must NOT require significant changes to the response to be added; rather, it should be able to be added by making only small, targeted changes. For example, a feature like "The response exceeds 1000 words" is NOT valid, because it is neither precise (there are many ways for a response to be long) nor could it be added by making only small changes (it would require big changes to the response).
    

    Here is the data, including the user prompt and assistant response samples and scores:

    ===== START OF RELEVANT DATA =====

    {data}

    ===== END OF RELEVANT DATA =====

    Think carefully and thoroughly about the features that appear in the data shown to you, considering both high level and low level features. After you have a list of {num_plans} features, CHECK CAREFULLY, one by one, that they take up **no longer than a short sentence**, and that they strictly follow EACH of the above requirements. If you feel that a feature you wrote does not satisfy one of the requirements, you MUST go back and find another feature that does meet all the requirements.
    
    Finally, in your output field, return ONLY the remaining valid features formatted as a JSON array, like this:

    ```json
    [
        "Feature 1",
        "Feature 2",
        ...
    ]
    ```

    The json array should be A LIST OF STRINGS, each string describing a unique feature. Remember to include the surrounding JSON tags.
""").strip()

BIAS_NUDGE = {
    "plus": "unusual, idiosyncratic, or potentially undesirable features should be especially considered: if possible, try to find attributes that are dispreferable or neutral",
    "minus": "potentially desirable or positive features should be especially considered: if possible, try to find attributes that are human-preferable"
}
