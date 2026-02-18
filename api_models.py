import re
import json
from random import Random
from loguru import logger
from dataclasses import dataclass
from typing import Any, Sequence, Literal

from caller import AutoCaller, RetryConfig, ChatHistory, Response
from utils import parse_json_response
from api_prompts import *

RETRY_CONFIG = RetryConfig(
    raise_when_exhausted=False,
    criteria=lambda response: response.has_response and response.finish_reason == "stop",
    max_attempts=3,
)

def concat_as_bullet(strings: list[str]) -> str:
    return "\n".join([f"- {s}" for s in strings])

SAME_ATTRS = concat_as_bullet([
    "The approximate length of the response",
    "The style and tone of the response",
])


class GenerationModel:
    """API samplers with one or more underlying models."""
    def __init__(
        self,
        model_name: str | list[str],
        max_par: int,
        force_caller: str | None = None,
        random_seed: int = 42,
        **kwargs,
    ):
        if isinstance(model_name, str):
            self.model_names = [model_name]
        else:
            self.model_names = model_name
        self.max_par = max_par
        self.force_caller = force_caller
        self.caller = AutoCaller(
            dotenv_path=".env", 
            retry_config=RETRY_CONFIG, 
            force_caller=force_caller
        )
        self.rng = Random(random_seed)
        self.kwargs = kwargs

    def to_dict(self) -> dict:
        params = {
            "model_names": self.model_names,
            "max_par": self.max_par,
            "force_caller": self.force_caller
        }
        params.update(self.kwargs)
        return params

    @property
    def model_name(self) -> str:
        """Returns the first model name."""
        return self.model_names[0]

    async def sample(
        self,
        chat_histories: list[ChatHistory],
        desc: str | None = None,
        **kwargs,
    ) -> list[Response|None]:
        """kwargs passed here will supersede the init kwargs.
        
        When multiple model names are configured, randomly selects a model
        for each chat history.
        """
        models = [self.rng.choice(self.model_names) for _ in chat_histories]
        
        responses = await self.caller.call(
            messages=chat_histories,
            model=models,
            max_parallel=self.max_par,
            desc=desc,
            **self.kwargs, **kwargs,
        )
        return responses


@dataclass(kw_only=True, slots=True)
class TextResult:
    text: str | None = None
    reasoning: str | None = None

class RewriteModel(GenerationModel):
    def __init__(
        self,
        model_name: str,
        max_par: int = 1024,
        enable_cache: bool = False,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name, 
            max_par=max_par, 
            enable_cache=enable_cache,
            **kwargs
        )

    async def rewrite(
        self,
        attributes: str | list[str],
        original_chats: list[ChatHistory],
        same_attrs: str | list[str] = "",
        desc: str | None = None,
    ) -> list[TextResult]:
        if isinstance(attributes, str):
            attributes_list = [attributes] * len(original_chats)
        else:
            assert len(attributes) == len(original_chats)
            attributes_list = attributes

        if isinstance(same_attrs, str):
            same_attrs_list = [same_attrs] * len(original_chats)
        else:
            assert len(same_attrs) == len(original_chats)
            same_attrs_list = same_attrs

        to_send_chats = []
        for i in range(len(original_chats)):
            rewrite_prompt = get_rewrite_prompt(same_attr=same_attrs_list[i])
            to_send_chats.append(ChatHistory.from_user(
                rewrite_prompt.format(
                    original=original_chats[i].as_text(),
                    new_attr=attributes_list[i],
                )
            ))
        try:
            responses = await self.sample(to_send_chats, desc=desc)
        except Exception as e:
            logger.exception(f"RewriteModel.rewrite failed:\nError:\n{e}")
            # Return None for all items on error
            return [TextResult() for _ in range(len(original_chats))]

        rewritten_responses = []
        for i, response in enumerate(responses):
            if response is None or (not response.has_response) or (response.finish_reason != "stop"):
                logger.warning(f"rewrite failed:\n{response}")
                rewritten_responses.append(TextResult())
                continue

            rw_text = response.first_response
            rw_reasoning = response.reasoning_content

            if re.sub(r'[^a-z0-9]', '', rw_text.strip().lower()) == "none":  # type: ignore
                # if rw_reasoning is not None and (not rw_reasoning.startswith("gAAAAA")):
                #     logger.warning(f"Rewriter returned None.\nprompt:\n{to_send_chats[i].to_openai_str()}\nreasoning:\n{rw_reasoning}")
                rewritten_responses.append(TextResult(
                    text=original_chats[i].get_first("assistant"),  # The Rewriter refused to rewrite
                    reasoning=rw_reasoning,
                ))
            else:
                rewritten_responses.append(TextResult(
                    text=rw_text,
                    reasoning=rw_reasoning,
                ))

        return rewritten_responses


# %% Judge model

@dataclass(frozen=True)
class RatingResult:
    score: float | None
    reasoning: str | None = None    # only exists for generative RMs

@dataclass(frozen=True)
class ComparisonResult:
    winner: Literal["A", "B", "Tie"] | None
    reasoning: list[str|None] | None = None    # only exists for generative RMs
    score_diff: float | None = None    # A score - B score, or number of trials won by A - won by B
    raw_score_A: float | None = None   # A score, or number of trials won by A
    raw_score_B: float | None = None   # B score, or number of trials won by B


class JudgeModel(GenerationModel):
    def __init__(
        self,
        model_name: str = "anthropic/claude-sonnet-4.5",
        max_par: int = 256,
        enable_cache: bool = False, 
        **kwargs,
    ):
        super().__init__(
            model_name=model_name, 
            max_par=max_par,
            enable_cache=enable_cache,
            **kwargs
        )

    async def judge_absolute(
        self,
        chat_histories: Sequence[ChatHistory], 
        use_tqdm: bool = True,
        rubric: str = DEFAULT_RUBRIC,
    ) -> list[RatingResult]:
        """Satisfies: len(output) == len(chat_histories)"""
        to_send_chats = []

        for chat in chat_histories:
            to_send_chats.append(
                ChatHistory.from_user(
                    JUDGE_ABSOLUTE_PROMPT.format(
                        user_prompt=chat.get_first("user"),
                        response=chat.get_first("assistant"),
                        rubric=rubric,
                    )
                )
            )

        desc = "LLM judge absolute" if use_tqdm else None
        responses = await self.sample(to_send_chats, desc=desc)

        rating_results = []
        for resp in responses:
            if resp is None:
                rating_results.append(RatingResult(score=None, reasoning=None))
                continue
            output, reasoning = parse_json_response(resp)
            try:
                score_float = float(output)
            except ValueError:
                logger.exception(f"Could not convert score to float: {output}")
                score_float = None
            rating_results.append(RatingResult(score=score_float, reasoning=reasoning))

        return rating_results

    async def judge_relative(
        self,
        chat_histories_A: Sequence[ChatHistory],
        chat_histories_B: Sequence[ChatHistory],
        num_trials: int = 1,
        use_tqdm: bool = True,
    ) -> list[ComparisonResult]:
        """Satisfies: len(output) == len(chat_histories_A)"""
        assert len(chat_histories_A) == len(chat_histories_B)
        to_send_chats: list[ChatHistory] = []
        to_send_flips: list[bool] = []
        # Track which input pairs need judging (False = identical responses, return Tie directly)
        pair_needs_judging: list[bool] = []

        for chat_A, chat_B in zip(chat_histories_A, chat_histories_B):
            user_prompt = chat_A.get_first("user")
            assert user_prompt == chat_B.get_first("user")
            response_A = chat_A.get_first("assistant")
            response_B = chat_B.get_first("assistant")

            # If responses are identical, skip sending to LLM - it's a Tie
            if response_A == response_B:
                pair_needs_judging.append(False)
                continue

            pair_needs_judging.append(True)

            flips = [False for _ in range(num_trials // 2)] + [True for _ in range(num_trials // 2)]
            if num_trials % 2 == 1:
                flips.append(self.rng.choice([False, True]))

            for flip in flips:
                to_send_chats.append(
                    ChatHistory.from_user(
                        JUDGE_RELATIVE_PROMPT.format(
                            user_prompt=user_prompt,
                            response_A=response_B if flip else response_A,
                            response_B=response_A if flip else response_B,
                        )
                    )
                )
            to_send_flips.extend(flips)

        # Only call LLM if there are non-identical pairs
        if to_send_chats:
            desc = "LLM judge relative" if use_tqdm else None
            responses = await self.sample(to_send_chats, desc=desc)
        else:
            responses = []

        # Build results, inserting Tie for identical pairs
        judge_results = []
        response_idx = 0

        for needs_judging in pair_needs_judging:
            if not needs_judging:
                # Identical responses -> Tie without LLM call
                judge_results.append(ComparisonResult(
                    winner="Tie",
                    reasoning=None,
                    score_diff=0,
                    raw_score_A=0,
                    raw_score_B=0,
                ))
                continue

            # Process num_trials responses for this pair
            comparison_stats: list[Literal["A", "B", "Tie"]|None] = []
            reasonings: list[str|None] = []

            for j in range(num_trials):
                resp = responses[response_idx + j]
                if resp is None:
                    comparison_stats.append(None)
                    reasonings.append(None)
                    logger.warning(f"Judge relative response is None. Input:\n{to_send_chats[response_idx + j].to_openai_str()}")
                    continue

                reasonings.append(resp.reasoning_content)
                if (not resp.has_response) or (resp.finish_reason != "stop"):
                    comparison_stats.append(None)
                    logger.warning(f"Judge relative response is invalid. Response: {resp}. Input:\n{to_send_chats[response_idx + j].to_openai_str()}")
                    continue

                matched = re.search(r"<output>(.+?)</output>", resp.first_response.strip(), re.DOTALL)  # type: ignore
                if matched:
                    response_text = matched.group(1).strip().lower()
                else:
                    response_text = resp.first_response.strip().lower()  # type: ignore

                if not to_send_flips[response_idx + j]:
                    winner = "A" if response_text == "a" else "B" if response_text == "b" else "Tie" if response_text == "tie" else None
                    comparison_stats.append(winner)
                else:
                    winner = "B" if response_text == "a" else "A" if response_text == "b" else "Tie" if response_text == "tie" else None
                    comparison_stats.append(winner)

            response_idx += num_trials

            raw_score_A = sum(1 for stat in comparison_stats if stat == "A")
            raw_score_B = sum(1 for stat in comparison_stats if stat == "B")
            score_diff = None if all(stat is None for stat in comparison_stats) else raw_score_A - raw_score_B
            winner = None if score_diff is None else "A" if score_diff > 0 else "B" if score_diff < 0 else "Tie"
            judge_results.append(ComparisonResult(
                winner=winner,
                reasoning=reasonings,
                score_diff=score_diff,
                raw_score_A=raw_score_A,
                raw_score_B=raw_score_B
            ))

        assert len(judge_results) == len(chat_histories_A)
        return judge_results

    async def judge_presence(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[bool|None]:
        """
        Judge whether attributes are present in responses.

        Args:
            pairs: List of (attribute, response_text) tuples

        Returns:
            List of booleans indicating presence (None if parsing failed)
        """
        if not pairs:
            return []

        prompts = [
            ChatHistory.from_user(
                JUDGE_PRESENCE_PROMPT.format(
                    attribute=attribute,
                    conversation=response_text,
                )
            )
            for attribute, response_text in pairs
        ]

        responses = await self.sample(prompts)

        results = []
        for resp in responses:
            if resp is None:
                results.append(None)
            elif (not resp.has_response) or (resp.finish_reason != "stop"):
                results.append(None)
            else:
                # Extract last True/False from the response
                response_text = resp.first_response.strip().lower()  # type: ignore
                if response_text.endswith("true"):
                    results.append(True)
                elif response_text.endswith("false"):
                    results.append(False)
                else:
                    # Try to find True/False anywhere in the last line
                    last_line = response_text.split("\n")[-1].strip()
                    last_line_alpha = "".join([c for c in last_line if c.isalpha()])
                    if last_line_alpha == "true":
                        results.append(True)
                    elif last_line_alpha == "false":
                        results.append(False)
                    else:
                        logger.warning(f"Could not parse judge presence response: {resp.first_response}")
                        results.append(None)

        return results
