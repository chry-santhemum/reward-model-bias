import asyncio
import time
import json
from loguru import logger
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, replace, asdict
from typing import Mapping, Sequence

import numpy as np
from caller import ChatHistory
from api_models import GenerationModel, RewriteModel
from reward_models import RewardModel
from state import Rollout, BaselineRollout, RewriteScore


@dataclass(frozen=True, kw_only=True, slots=True)
class PromptInput:
    system: str | None
    user: str
    batch_id: str


@dataclass(kw_only=True, slots=True)
class PromptOutput:
    system: str | None
    user: str
    assistant: str | None
    batch_id: str
    raw_score: float | None = None
    model: str | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class RewriteInput:
    system: str  # the attribute to rewrite toward
    user: str
    original_assistant: str
    batch_id: str
    same_attrs: str = ""  # pre-formatted string of attributes to hold constant

@dataclass(kw_only=True, slots=True)
class RewriteOutput:
    system: str  # the attribute that was rewritten toward
    user: str
    original_assistant: str
    batch_id: str
    rewriter_model_name: str
    rewritten_assistant: str | None
    rewriter_reasoning: str | None
    raw_score: float | None = None


@dataclass(frozen=True)
class BatchStartMarker:
    batch_id: str
    expected_items: int


async def policy_worker(
    policy_model: GenerationModel,
    batch_size: int,
    in_queue: asyncio.Queue[PromptInput|None],
    out_queue: asyncio.Queue[PromptOutput],
    worker_id: int,
):
    async def sample_batch(batch: list[PromptInput]):
        if not batch:
            return
        
        chat_histories = []
        for input in batch:
            if input.system is not None:
                chat_histories.append(ChatHistory.from_system(input.system).add_user(input.user))
            else:
                chat_histories.append(ChatHistory.from_user(input.user))
                
        responses = await policy_model.sample(chat_histories)
        sample_results = []

        for i, response in enumerate(responses):
            input = batch[i]
            if (
                response is None or
                (not response.has_response) or
                (response.finish_reason != "stop")
            ):
                logger.warning(
                    f"[policy_worker {worker_id}] Failed to sample for user prompt:\nuser prompt:\n{input.user}\n"
                    f"model: {response.model if response is not None else 'None'}\n"
                    f"finish reason: {response.finish_reason if response is not None else 'None'}\n"
                    f"response:\n{response.first_response if response is not None else 'None'}"
                )
                sample_results.append(PromptOutput(
                    system=input.system,
                    user=input.user,
                    assistant=None,
                    batch_id=input.batch_id,
                    model=response.model if response is not None else None,
                ))
            else:
                sample_results.append(PromptOutput(
                    system=input.system,
                    user=input.user,
                    assistant=response.first_response.strip(),  # type: ignore
                    batch_id=input.batch_id,
                    model=response.model,
                ))

        for result in sample_results:
            await out_queue.put(result)
            in_queue.task_done()

        batch.clear()
        
    current_batch: list[PromptInput] = []
        
    while True:
        try:
            task_input = await asyncio.wait_for(in_queue.get(), timeout=5.0)
        except asyncio.TimeoutError:
            # Flush any accumulated items in current_batch
            if current_batch:
                await sample_batch(current_batch)
            continue
        
        if in_queue.qsize() % 100 == 0:
            logger.info(
                f"[policy_worker {worker_id}] Popped 1 task. In queue size: {in_queue.qsize()}"
            )

        if task_input is None:  # Stop sentinel
            in_queue.task_done()
            await sample_batch(current_batch)
            break
        
        current_batch.append(task_input)
        if len(current_batch) >= batch_size:
            await sample_batch(current_batch)


async def rewrite_worker(
    rewrite_models: list[RewriteModel],
    batch_size: int,
    in_queue: asyncio.Queue[RewriteInput|None],
    out_queue: asyncio.Queue[RewriteOutput],
    worker_id: int,
):
    async def rewrite_batch(batch: list[RewriteInput]):
        start_time = time.time()
        if not batch:
            return

        tasks = []
        for rwm in rewrite_models:
            tasks.append(rwm.rewrite(
                attributes=[input.system for input in batch],
                original_chats=[ChatHistory.from_user(input.user).add_assistant(
                    input.original_assistant
                ) for input in batch],
                same_attrs=[input.same_attrs for input in batch],
            ))

        # Add timeout to prevent hanging on slow API calls
        try:
            all_responses = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=60.0  # 1 minute timeout for the batch
            )
        except asyncio.TimeoutError:
            logger.error(f"[rewrite_worker {worker_id}] Batch timed out after 60s")
            # Produce None outputs for all inputs × all rewriters
            for input in batch:
                for rwm in rewrite_models:
                    await out_queue.put(RewriteOutput(
                        system=input.system,
                        user=input.user,
                        original_assistant=input.original_assistant,
                        rewriter_model_name=rwm.model_name,
                        rewritten_assistant=None,
                        rewriter_reasoning=None,
                        batch_id=input.batch_id,
                    ))
            for _ in batch:
                in_queue.task_done()
            batch.clear()
            return

        rewrite_results = []
        for rwm, responses in zip(rewrite_models, all_responses):
            # Handle exceptions from failed rewriter calls
            if isinstance(responses, Exception):
                logger.error(f"[rewrite_worker {worker_id}] Rewriter {rwm.model_name} failed: {responses}")
                # Still produce outputs with None to maintain count
                for input in batch:
                    rewrite_results.append(RewriteOutput(
                        system=input.system,
                        user=input.user,
                        original_assistant=input.original_assistant,
                        rewriter_model_name=rwm.model_name,
                        rewritten_assistant=None,
                        rewriter_reasoning=None,
                        batch_id=input.batch_id,
                    ))
                continue

            # Handle case where rewriter returns fewer results than expected
            if len(responses) != len(batch):
                logger.error(
                    f"[rewrite_worker {worker_id}] Rewriter {rwm.model_name} returned {len(responses)} results for {len(batch)} inputs"
                )

            for i, input in enumerate(batch):
                if i < len(responses):
                    response = responses[i]
                    if response.text is None:
                        logger.warning(
                            f"rewrite_worker {worker_id} - Failed to rewrite:\nUser prompt:\n{input.user}"
                        )
                    rewrite_results.append(RewriteOutput(
                        system=input.system,
                        user=input.user,
                        original_assistant=input.original_assistant,
                        rewriter_model_name=rwm.model_name,
                        rewritten_assistant=response.text,
                        rewriter_reasoning=response.reasoning,
                        batch_id=input.batch_id,
                    ))
                else:
                    # Missing response - produce None output
                    rewrite_results.append(RewriteOutput(
                        system=input.system,
                        user=input.user,
                        original_assistant=input.original_assistant,
                        rewriter_model_name=rwm.model_name,
                        rewritten_assistant=None,
                        rewriter_reasoning=None,
                        batch_id=input.batch_id,
                    ))

        for result in rewrite_results:
            await out_queue.put(result)

        for _ in batch:
            in_queue.task_done()

        if worker_id < 5 or in_queue.qsize() < 10:
            logger.info(f"[rewrite_worker {worker_id}] Done batch of size {len(batch)}, produced {len(rewrite_results)} outputs in {(time.time() - start_time):.2f}s, in_queue: {in_queue.qsize()}, out_queue: {out_queue.qsize()}")
        batch.clear() 

    async def safe_rewrite_batch(batch: list[RewriteInput]):
        """Wrapper that catches exceptions and produces None outputs on failure."""
        try:
            await rewrite_batch(batch)
        except Exception as e:
            logger.exception(f"[rewrite_worker {worker_id}] rewrite_batch failed: {e}")
            # Produce None outputs for all inputs × all rewriters to maintain count
            for input in batch:
                for rwm in rewrite_models:
                    await out_queue.put(RewriteOutput(
                        system=input.system,
                        user=input.user,
                        original_assistant=input.original_assistant,
                        rewriter_model_name=rwm.model_name,
                        rewritten_assistant=None,
                        rewriter_reasoning=None,
                        batch_id=input.batch_id,
                    ))
            for _ in batch:
                in_queue.task_done()
            batch.clear()

    current_batch: list[RewriteInput] = []

    while True:
        try:
            task_input = await asyncio.wait_for(in_queue.get(), timeout=5.0)
        except asyncio.TimeoutError:
            # Flush any accumulated items in current_batch
            if current_batch:
                await safe_rewrite_batch(current_batch)
            continue

        if task_input is None:  # Stop sentinel
            in_queue.task_done()
            await safe_rewrite_batch(current_batch)
            return

        current_batch.append(task_input)
        if len(current_batch) >= batch_size:
            await safe_rewrite_batch(current_batch)


async def rating_worker(
    reward_model: RewardModel,
    in_queue: asyncio.Queue[PromptOutput | RewriteOutput | BatchStartMarker],
    all_results: Mapping[str, Sequence[PromptOutput | RewriteOutput]],
    all_futures: Mapping[str, asyncio.Future],
):
    done = False
    # Keep track of progress by batch id
    processed_counts = defaultdict(int)
    expected_counts = defaultdict(int)

    async def rate_batch(batch: list[PromptOutput | RewriteOutput]):
        start_time = time.time()
        chat_histories = []
        existing_indices = []
        for i, result in enumerate(batch):
            if isinstance(result, PromptOutput):
                if result.assistant is None:
                    continue
                chat_histories.append(ChatHistory.from_user(result.user).add_assistant(result.assistant))  # type: ignore
                existing_indices.append(i)
            elif isinstance(result, RewriteOutput):
                if result.rewritten_assistant is None:
                    continue
                chat_histories.append(ChatHistory.from_user(result.user).add_assistant(result.rewritten_assistant))  # type: ignore
                existing_indices.append(i)

        reward_scores = await reward_model.async_rate(chat_histories, use_tqdm=False)
        for i, reward_score in zip(existing_indices, reward_scores):
            batch[i].raw_score = reward_score.score
        
        all_results[batch[0].batch_id].extend(batch)  # type: ignore      
        logger.debug(f"[rating_worker] Done batch of size {len(existing_indices)}/{len(batch)} in {(time.time() - start_time):.2f} seconds. Queue size: {in_queue.qsize()}")
        batch.clear()

    while True:
        batch = []
        last_flush_time = time.time()

        try:
            while len(batch) < reward_model.batch_size:
                item = await asyncio.wait_for(in_queue.get(), timeout=3.0)

                if item is None:  # Sentinel value to signal stop
                    logger.info("Stop sentinel received.")
                    in_queue.task_done()
                    done = True
                    break

                if isinstance(item, BatchStartMarker):
                    logger.info(f"Batch {item.batch_id} sentinel received: {item.expected_items} items.")
                    in_queue.task_done()
                    expected_counts[item.batch_id] += item.expected_items
                    continue

                batch.append(item)
                processed_counts[item.batch_id] += 1
                in_queue.task_done()
                
                if time.time() - last_flush_time > 3.0:
                    break

        except asyncio.TimeoutError:
            pass  # Not an error, just means we process the current incomplete batch
        
        # Only rate when there is something to rate; but always progress-check below
        # to avoid the edge case of rewrite worker failing to produce valid outputs,
        # such that the rating worker doesn't know that it's done
        if len(batch) > 0:
            # if in_queue.qsize() >= 1000 or in_queue.qsize() <= 32:
            #     logger.info(f"[rating_worker] Flushing, waited for {(time.time() - last_flush_time):.2f} seconds. Queue size: {in_queue.qsize()}")
            last_flush_time = time.time()
            try:
                await rate_batch(batch)
            except Exception:
                logger.exception(f"rating_worker: rating failed; marking {len(batch)} items with score=None")
                all_results[batch[0].batch_id].extend(batch)  # type: ignore

        completed_batch_ids = []

        for batch_id, expected in list(expected_counts.items()):
            processed = processed_counts.get(batch_id, 0)
            if processed == expected:
                all_futures[batch_id].set_result(all_results[batch_id])
                completed_batch_ids.append(batch_id)
                logger.info(f"[rating_worker] Batch {batch_id} completed: {processed}/{expected}.")
            elif processed % 100 == 0 or in_queue.qsize() < 10:
                # Log progress every 100 items or when queue is small
                logger.info(
                    f"[rating_worker] Batch {batch_id} progress: {processed}/{expected}. Queue: {in_queue.qsize()}"
                )

        for batch_id in completed_batch_ids:
            del all_results[batch_id]  # type: ignore
            if batch_id in processed_counts:
                del processed_counts[batch_id]
            del expected_counts[batch_id]

        if done:
            # Break outer loop
            break


# %%
# Organizing results


def organize_rewrites(
    all_results: list[RewriteOutput],
    baseline_rollouts: dict[str, list[BaselineRollout]],
    student_model_name: str,
    n_rollouts: int | None = None,
    save_dir: Path | None = None,
) -> dict[str, dict[str, dict[str, list[Rollout | None]]]]:

    organized_rollouts = defaultdict(dict)

    for result in all_results:
        rewriter_rollouts = organized_rollouts[result.rewriter_model_name]
        if result.system not in rewriter_rollouts:
            rewriter_rollouts[result.system] = {}

        attribute_rollouts = rewriter_rollouts[result.system]
        if result.user not in attribute_rollouts:
            attribute_rollouts[result.user] = [
                None
                for _ in range(
                    n_rollouts
                    if n_rollouts is not None
                    else len(baseline_rollouts[result.user])
                )
            ]
        if result.rewritten_assistant is None:
            continue

        found = False
        for i, baseline_rollout in enumerate(baseline_rollouts[result.user]):
            if n_rollouts is not None and i >= n_rollouts:
                break
            if baseline_rollout.response == result.original_assistant:
                if attribute_rollouts[result.user][i] is not None:
                    continue

                # Compute reward diff: rewritten - baseline
                # Positive means rewritten scored higher than baseline.
                # Interpretation depends on how the attribute was phrased.
                if result.rewritten_assistant == result.original_assistant:
                    # Unchanged rewrite - no change signal
                    score_diff = 0.0
                    rewritten_raw = baseline_rollout.scores[student_model_name]
                else:
                    baseline_raw = baseline_rollout.scores[student_model_name]
                    rewritten_raw = result.raw_score
                    if rewritten_raw is not None:
                        score_diff = rewritten_raw - baseline_raw
                    else:
                        score_diff = None

                student_score = RewriteScore(
                    score=score_diff,
                    raw_score=rewritten_raw,
                    reasoning=None,
                    model_name=student_model_name,
                )
                attribute_rollouts[result.user][i] = Rollout(
                    rewritten_response=result.rewritten_assistant,
                    baseline_response=result.original_assistant,
                    student_score=student_score,
                    teacher_score=None,
                )
                found = True
                break

        if not found:
            raise ValueError(
                f"Rewrite result for {result.user} and {result.system} not matched."
            )

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        for rewriter_name, rewriter_data in organized_rollouts.items():
            rewriter_dir = save_dir / rewriter_name.replace("/", "_")
            rewriter_dir.mkdir(parents=True, exist_ok=True)

            with open(rewriter_dir / "rollouts.json", "w") as f:
                json_data = {
                    attr: {
                        user: [
                            {
                                "rewritten_response": r.rewritten_response,
                                "baseline_response": r.baseline_response,
                                "student_score": r.student_score.raw_score,
                                "student_diff": r.student_score.score,
                            } if r is not None else None
                            for r in rollouts
                        ]
                        for user, rollouts in user_dict.items()
                    }
                    for attr, user_dict in rewriter_data.items()
                }
                json.dump(json_data, f, indent=4, sort_keys=True)

    return dict(organized_rollouts)

