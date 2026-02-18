import re
import gc
import ast
import json
import time
import random
import datetime
import asyncio
import functools
import inspect

from json_repair import repair_json
from loguru import logger
from typing import Any, Tuple, Optional, Awaitable, Literal, Callable
from tqdm.asyncio import tqdm_asyncio

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.trainer_utils import set_seed as hf_set_seed

from caller import Response

Tokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast

def load_model(
    model_name: str,
    model_type: Literal["reward", "generation"],
    device: str = "auto",
    attn_implementation: str = "sdpa",
) -> tuple[PreTrainedModel, Tokenizer]:

    if model_type == "reward":
        print(f"Loading reward model {model_name} with device {device}...")
        load_kwargs = {
            "dtype": torch.bfloat16,
            "attn_implementation": attn_implementation,
            "num_labels": 1,
        }
        if device != "auto":
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, **load_kwargs
            )
            model.to(device)
        else:
            load_kwargs["device_map"] = "auto"
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, **load_kwargs
            )

    elif model_type == "generation":
        print(f"Loading generation model {model_name} with device {device}...")
        load_kwargs = {
            "dtype": torch.bfloat16,
            "attn_implementation": attn_implementation,
        }
        if device != "auto":
            model = AutoModelForCausalLM.from_pretrained(
                model_name, **load_kwargs
            )
            model.to(device)  # type: ignore
        else:
            load_kwargs["device_map"] = "auto"
            model = AutoModelForCausalLM.from_pretrained(
                model_name, **load_kwargs
            )

    print("Model loaded. Set to eval mode and disabled gradients.")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        print("No pad token found, setting to eos token")
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        print("No pad token id found, setting to eos token id")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token = tokenizer.pad_token

    return model, tokenizer


def set_seed_all(seed: int):
    random.seed(seed)  # Python RNG
    np.random.seed(seed)  # NumPy RNG
    torch.manual_seed(seed)  # PyTorch CPU RNG
    torch.cuda.manual_seed_all(seed)  # PyTorch CUDA RNG
    hf_set_seed(seed)


def remove_outliers(data: list[float], iqr_k: float = 1.5) -> list[float]:
    """
    Remove outliers using Tukey's fences (IQR method).
    Values outside [Q1 - k*IQR, Q3 + k*IQR] are considered outliers.
    """
    data_np = np.array(data)
    if len(data_np) == 0:
        return []

    q1 = np.percentile(data_np, 25)
    q3 = np.percentile(data_np, 75)
    iqr = q3 - q1
    low = q1 - iqr_k * iqr
    high = q3 + iqr_k * iqr
    mask = (data_np >= low) & (data_np <= high)
    return list(data_np[mask])


async def async_gather(tasks: list[Awaitable[Any]], max_parallel: Optional[int] = None):
    if max_parallel is None or max_parallel >= len(tasks):
        return await tqdm_asyncio.gather(*tasks)
    else:
        semaphore = asyncio.Semaphore(max_parallel)

        async def sem_task(task):
            async with semaphore:
                return await task

        return await tqdm_asyncio.gather(*(sem_task(task) for task in tasks))

    
def parse_json_response(
    resp: Response, log_json_error: bool = True, marker: str = "json",
) -> Tuple[Any, str | None]:
    raw_text = resp.first_response
    if raw_text is None:
        logger.warning(f"Response is None: {resp}")
        return None, None

    output, reasoning = None, None
    if resp.reasoning_content is not None:
        reasoning = resp.reasoning_content

    try:
        if raw_text.strip().startswith(f"```{marker}"):
            json_str = raw_text.split(f"```{marker}", 1)[1].rsplit("```", 1)[0].strip()
            if reasoning is None:
                reasoning = raw_text.rsplit(f"```{marker}", 1)[0].strip()
        else:
            json_str = raw_text.strip()

        # Try ast.literal_eval first for Python literals (handles single quotes properly)
        if marker == "python":
            try:
                output = ast.literal_eval(json_str)
            except (ValueError, SyntaxError):
                # Fall back to json_repair
                output = json.loads(repair_json(json_str))
        else:
            output = json.loads(repair_json(json_str))

    except Exception as e:
        output = raw_text.strip()
        if reasoning is None:
            reasoning = raw_text.strip()
        if log_json_error:
            logger.exception(f"Response JSON parse error: {e}")
            logger.error(f"API response: {resp}")

    return output, reasoning


def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def count_words(text):
    # Split on whitespace and common delimiters
    # This regex splits on spaces, newlines, and common code delimiters
    words = re.findall(r"\S+", text)
    return len(words)


async def time_operation(operation_name, coroutine):
    """Time an async operation and print the duration."""
    start_time = time.time()
    result = await coroutine
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"  {operation_name} completed in {duration:.2f} seconds")
    return result


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (assume not notebook)
    except NameError:
        return False  # Probably standard Python interpreter


# %% borrowed from https://github.com/GraySwanAI/nanoGCG/blob/d4a20a8b4a12c3ee814a77a61423b7bdcd151f82/nanogcg/utils.py
# borrowed from https://github.com/huggingface/accelerate/blob/85a75d4c3d0deffde2fc8b917d9b1ae1cb580eb2/src/accelerate/utils/memory.py#L69
def should_reduce_batch_size(exception: Exception) -> bool:
    """
    Checks if `exception` relates to CUDA out-of-memory, CUDNN not supported, or CPU out-of-memory

    Args:
        exception (`Exception`):
            An exception
    """
    _statements = [
        "CUDA out of memory.",  # CUDA OOM
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",  # CUDNN SNAFU
        "DefaultCPUAllocator: can't allocate memory",  # CPU OOM
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False

# modified from https://github.com/huggingface/accelerate/blob/85a75d4c3d0deffde2fc8b917d9b1ae1cb580eb2/src/accelerate/utils/memory.py#L87
def find_executable_batch_size(starting_batch_size: int, function: Callable|None = None):
    """
    A basic decorator that will try to execute `function`. 
    If it fails from exceptions related to out-of-memory or CUDNN, 
    the batch size is cut in half and passed to `function`.

    NOTE: `function` must take in a `batch_size` parameter as its first argument.

    Example:

    ```python
    >>> from utils import find_executable_batch_size


    >>> @find_executable_batch_size(starting_batch_size=128)
    ... def train(batch_size, model, optimizer):
    ...     ...


    >>> train(model, optimizer)
    ```
    """
    if function is None:
        return functools.partial(find_executable_batch_size, starting_batch_size)

    batch_size = starting_batch_size

    def decorator(*args, **kwargs):
        nonlocal batch_size
        gc.collect()
        torch.cuda.empty_cache()
        params = list(inspect.signature(function).parameters.keys())
        # Guard against user error
        if len(params) < (len(args) + 1):
            arg_str = ", ".join([f"{arg}={value}" for arg, value in zip(params[1:], args[1:])])
            raise TypeError(
                f"Batch size was passed into `{function.__name__}` as the first argument when called."  # type: ignore
                f"Remove this as the decorator already does so: `{function.__name__}({arg_str})`"  # type: ignore
            )
        while True:
            if batch_size == 0:
                raise RuntimeError("No executable batch size found, reached zero.")
            try:
                return function(batch_size, *args, **kwargs)
            except Exception as e:
                if should_reduce_batch_size(e):
                    gc.collect()
                    torch.cuda.empty_cache()
                    batch_size //= 2
                    logger.warning(f"Decreasing batch size to {batch_size}.")
                    logger.warning(f"Error: {e}.")
                else:
                    raise

    return decorator
