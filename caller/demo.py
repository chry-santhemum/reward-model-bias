import asyncio
import time
from contextlib import contextmanager
from .caller import AutoCaller
from .cache import CacheConfig


async def basic_usage():
    caller = AutoCaller(dotenv_path=".env", force_caller="openrouter")

    messages = [
        "What's your favorite joke?",
        "Generate 5 responses with theircorresponding probabilities. Tell me a joke.",
    ]

    responses = await caller.call(
        messages=messages,
        max_parallel=128,
        model="openai/gpt-5-mini",
        desc="Sending prompts",
        max_tokens=1024,
        enable_cache=False,
    )

    for i, response in enumerate(responses):
        print(f"Response {i+1}:")
        print(response)


@contextmanager
def timer(description: str):
    """Context manager to measure wallclock time."""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"  [{description}] took {elapsed:.3f}s")


async def cache_demo():
    caller = AutoCaller(dotenv_path=".env", force_caller="openrouter")
    template = "Tell me a niche joke that involves the number {n}."
    model = "meta-llama/llama-3.1-8b-instruct"
    messages = [template.format(n=n) for n in range(1024)]

    print("Cache enabled:\n")
    print("First call...")
    with timer("API call"):
        responses_1 = await caller.call(messages=messages, model=model, max_tokens=1024, max_parallel=1024)
    print(f"  Response: {responses_1[86]}")

    print("\nSecond call...")
    with timer("Cache hit"):
        responses_2 = await caller.call(messages=messages, model=model, max_tokens=1024, max_parallel=1024)
    print(f"  Response: {responses_2[86]}")

    # Compare with when cache is disabled
    print("\n\nCache disabled:\n")
    print("First call...")
    with timer("API call"):
        responses_1 = await caller.call(messages=messages, model=model, max_tokens=1024, max_parallel=1024, enable_cache=False)
    print(f"  Response: {responses_1[86]}")

    print("\nSecond call...")
    with timer("Cache hit"):
        responses_2 = await caller.call(messages=messages, model=model, max_tokens=1024, max_parallel=1024, enable_cache=False)
    print(f"  Response: {responses_2[86]}")


if __name__ == "__main__":
    # asyncio.run(basic_usage())
    asyncio.run(cache_demo())
