import time
import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any
from loguru import logger

from api_models import RewriteModel
from reward_models import RewardModel
from state import BaselineRollout
from bias_workers import (
    BatchStartMarker,
    RewriteInput,
    RewriteOutput,
    rewrite_worker,
    rating_worker,
    organize_rewrites,
)


class BiasEvaluator:
    def __init__(
        self,
        rewrite_models: list[RewriteModel],
        reward_model: RewardModel,
        n_rewrite_workers: int = 128
    ):
        self.rewrite_models = rewrite_models
        self.reward_model = reward_model
        self.n_rewrite_workers = n_rewrite_workers

        self._workers_started = False

        # Defer queues and worker startup until inside a running event loop
        self._batch_id: int
        self._batch_id_lock: asyncio.Lock | None
        self.queue_input: asyncio.Queue | None
        self.queue_rewrite: asyncio.Queue | None
        self.batch_results: dict[str, list[RewriteOutput]]
        self.batch_futures: dict[str, asyncio.Future]
        self.rewrite_workers: list[asyncio.Task] = []
        self.rating_worker: asyncio.Task | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "rewrite_models": [m.to_dict() for m in self.rewrite_models],
            "reward_model": self.reward_model.to_dict(),
            "n_rewrite_workers": self.n_rewrite_workers
        }

    async def _ensure_workers_started(self):
        if self._workers_started:
            return
        self._batch_id = 0
        self._batch_id_lock = asyncio.Lock()
        max_par = min(m.max_par for m in self.rewrite_models)
        self.queue_input = asyncio.Queue(maxsize=2 * max_par)
        self.queue_rewrite = asyncio.Queue()
        self.batch_results = {}
        self.batch_futures = {}

        self.rewrite_workers = [
            asyncio.create_task(
                rewrite_worker(
                    rewrite_models=self.rewrite_models,
                    batch_size=max(1, max_par // self.n_rewrite_workers),
                    in_queue=self.queue_input,
                    out_queue=self.queue_rewrite,
                    worker_id=worker_id,
                )
            )
            for worker_id in range(self.n_rewrite_workers)
        ]

        self.rating_worker = asyncio.create_task(
            rating_worker(
                self.reward_model,
                self.queue_rewrite,
                self.batch_results,
                self.batch_futures,
            )
        )
        # Surface crashes of the rating worker early
        def _rating_done_cb(task: asyncio.Task):
            try:
                exc = task.exception()
            except asyncio.CancelledError:
                logger.info("[rating_worker] cancelled, exiting.")
                return
            if exc is not None:
                logger.exception("[rating_worker] exited with exception")
            else:
                logger.info("[rating_worker] exited without exception.")
        self.rating_worker.add_done_callback(_rating_done_cb)
        
        self._workers_started = True

    async def _run_rewrite_batch(
        self,
        rewrite_inputs: list[RewriteInput],
    ) -> list[RewriteOutput]:
        """Run a batch of rewrite inputs through the worker pipeline."""
        assert self._batch_id_lock is not None
        async with self._batch_id_lock:
            batch_id = str(self._batch_id)
            self.batch_results[batch_id] = []
            loop = asyncio.get_running_loop()
            self.batch_futures[batch_id] = loop.create_future()
            self._batch_id += 1

        # Send batch start marker (each input produces one output per rewriter)
        expected_items = len(rewrite_inputs) * len(self.rewrite_models)
        logger.info(f"Batch {batch_id} expects {expected_items} results...")
        await self.queue_rewrite.put(  # type: ignore
            BatchStartMarker(batch_id=batch_id, expected_items=expected_items)
        )

        # Put tasks (update batch_id since RewriteInput is frozen)
        for rewrite_input in rewrite_inputs:
            input_with_batch_id = replace(rewrite_input, batch_id=batch_id)
            await self.queue_input.put(input_with_batch_id)  # type: ignore

        # Wait for results
        batch_results = await self.batch_futures[batch_id]
        del self.batch_futures[batch_id]

        return batch_results

    async def evaluate_attributes(
        self,
        user_prompts: list[str],
        attributes: list[str],
        baselines: dict[str, list[BaselineRollout]],
        same_attrs: list[str] | None = None,  # parallel to attributes, pre-formatted strings
        n_rollouts: int | None = None,  # max number of baseline responses to rewrite
        save_dir: Path | None = None,
    ):
        """Sends roughly len(user_prompts) * len(attributes) * n_rollouts rewrite requests.

        Each attribute string specifies the target state for the rewrite.
        If rewriter returns unchanged (outputs "None"), the original response is used with score_diff=0.

        Args:
            same_attrs: Optional list parallel to `attributes`. Each element is a pre-formatted
                string specifying which attributes to hold constant during that rewrite.
        """
        await self._ensure_workers_started()
        start_time = time.time()

        if same_attrs is not None:
            assert len(same_attrs) == len(attributes), "same_attrs must be parallel to attributes"

        # Build all rewrite inputs
        rewrite_inputs: list[RewriteInput] = []
        for user in user_prompts:
            for attr_idx, attribute in enumerate(attributes):
                attr_same_attrs = same_attrs[attr_idx] if same_attrs else ""
                for i, baseline_rollout in enumerate(baselines.get(user, [])):
                    if n_rollouts is not None and i >= n_rollouts:
                        break
                    rewrite_inputs.append(
                        RewriteInput(
                            system=attribute,
                            user=user,
                            original_assistant=baseline_rollout.response,
                            batch_id="",  # Will be set in _run_rewrite_batch
                            same_attrs=attr_same_attrs,
                        )
                    )

        logger.info(f"Sending {len(rewrite_inputs)} rewrite requests...")
        all_results = await self._run_rewrite_batch(rewrite_inputs)

        # Count unchanged rewrites for logging
        n_unchanged = sum(
            1 for r in all_results
            if r.rewritten_assistant is not None and r.rewritten_assistant == r.original_assistant
        )
        if n_unchanged > 0:
            logger.info(f"Rewrite stats: {n_unchanged}/{len(all_results)} rewrites returned unchanged")

        organized_results = organize_rewrites(
            all_results, baselines, self.reward_model.model_name, n_rollouts, save_dir
        )

        logger.info(f"Evaluation complete: {len(all_results)} results in {(time.time() - start_time):.2f} seconds")
        return organized_results

    async def shutdown(self):
        if not self._workers_started:
            return

        assert self.queue_input is not None and self.queue_rewrite is not None
        for _ in range(self.n_rewrite_workers):
            await self.queue_input.put(None)  # Sentinel values for rewrite_workers

        await asyncio.gather(*self.rewrite_workers)
        logger.info("--- rewrite workers finished. ---")

        # Ensure all produced rewrite results are fully consumed and marked done
        # before signalling the rating worker to exit. This prevents leaving
        # unfinished tasks on the queue and avoids shutdown hangs.
        await self.queue_rewrite.join()
        await self.queue_rewrite.put(None)  # Sentinel value for rating_worker
        if self.rating_worker is not None:
            await self.rating_worker
        logger.info("--- rewrite rating worker finished. ---")

        self._batch_id = 0
        self._batch_id_lock = None
        self._workers_started = False
        self.queue_input = None
        self.queue_rewrite = None
        self.batch_results = {}
        self.batch_futures = {}
        self.rewrite_workers = []
        self.rating_worker = None


    # Make this into an async context manager to ensure proper shutdown
    async def __aenter__(self):
        await self._ensure_workers_started()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.shutdown()
        return False
