import asyncio
import json
from pathlib import Path

import dotenv
import matplotlib.pyplot as plt
import numpy as np

from utils import timestamp
from caller import ChatHistory
from reward_models import APIRewardModel
from state import AttributeStats, RewriteScore, Rollout


def get_seed_paths(step_dir: Path, seed_ids: list[int] | None) -> list[Path]:
    if seed_ids is None:
        return sorted(step_dir.glob("seed_[0-9].json"))
    return [step_dir / f"seed_{seed_id}.json" for seed_id in seed_ids]


def load_scored_attributes(
    seed_path: Path,
    max_rollouts_per_attribute: int | None,
) -> list[dict]:
    data = json.loads(seed_path.read_text())
    kept_attributes = []

    for item in data:
        has_teacher_winrate = item["teacher_winrate"] is not None
        pairs = []
        missing_teacher_score = False
        found_teacher_score = False

        for user_prompt, rollouts in item["rollouts"].items():
            for rollout in rollouts:
                if rollout is None:
                    continue
                if rollout.get("teacher_score") is None:
                    missing_teacher_score = True
                    continue

                found_teacher_score = True
                pairs.append(
                    {
                        "user_prompt": user_prompt,
                        "baseline_response": rollout["baseline_response"],
                        "rewritten_response": rollout["rewritten_response"],
                        "old_teacher_score": rollout["teacher_score"]["score"],
                    }
                )
                if (
                    max_rollouts_per_attribute is not None
                    and len(pairs) >= max_rollouts_per_attribute
                ):
                    break
            if (
                max_rollouts_per_attribute is not None
                and len(pairs) >= max_rollouts_per_attribute
            ):
                break

        if has_teacher_winrate != found_teacher_score:
            raise ValueError(
                f"Inconsistent teacher metadata for attribute in {seed_path}: "
                f"{item['attribute']}"
            )

        if found_teacher_score and missing_teacher_score:
            raise ValueError(
                f"Mixed teacher-scored and unscored rollouts for attribute in {seed_path}: "
                f"{item['attribute']}"
            )

        if not found_teacher_score:
            continue

        kept_attributes.append(
            {
                "attribute": item["attribute"],
                "old_teacher_winrate": item["teacher_winrate"],
                "pairs": pairs,
            }
        )

    return kept_attributes


async def judge_pairs(judge_model: APIRewardModel, pairs: list[dict]):
    chats_a = [
        ChatHistory.from_user(pair["user_prompt"]).add_assistant(pair["rewritten_response"])
        for pair in pairs
    ]
    chats_b = [
        ChatHistory.from_user(pair["user_prompt"]).add_assistant(pair["baseline_response"])
        for pair in pairs
    ]
    return await judge_model.async_compare(chats_a, chats_b, use_tqdm=True)


def compute_teacher_winrate(
    attribute: str,
    pair_results: list[dict],
    judge_model_name: str,
) -> float | None:
    rollouts_by_prompt = {}

    for pair_result in pair_results:
        user_prompt = pair_result["user_prompt"]
        if user_prompt not in rollouts_by_prompt:
            rollouts_by_prompt[user_prompt] = []

        rollouts_by_prompt[user_prompt].append(
            Rollout(
                rewritten_response=pair_result["rewritten_response"],
                baseline_response=pair_result["baseline_response"],
                student_score=RewriteScore(
                    score=0.0,
                    raw_score=None,
                    reasoning=None,
                    model_name="unused",
                ),
                teacher_score=RewriteScore(
                    score=pair_result["teacher_score"]["score"],
                    raw_score=None,
                    reasoning=None,
                    model_name=judge_model_name,
                ),
            )
        )

    return AttributeStats(
        attribute=attribute,
        rollouts=rollouts_by_prompt,
    ).winrate("teacher")


def save_correlation_plot(
    summary: list[dict],
    seed_name: str,
    judge_model_name: str,
    out_path: Path,
) -> None:
    points = [
        (item["old_teacher_winrate"], item["new_teacher_winrate"])
        for item in summary
        if item["old_teacher_winrate"] is not None and item["new_teacher_winrate"] is not None
    ]
    if not points:
        return

    xs = np.array([x for x, _ in points])
    ys = np.array([y for _, y in points])
    mean_x = xs.mean()
    mean_y = ys.mean()
    var_x = np.mean((xs - mean_x) ** 2)
    var_y = np.mean((ys - mean_y) ** 2)
    covariance = np.mean((xs - mean_x) * (ys - mean_y))
    ccc_denom = var_x + var_y + (mean_x - mean_y) ** 2
    if ccc_denom == 0:
        concordance = 1.0
    else:
        concordance = (2 * covariance) / ccc_denom

    lo = min(xs.min(), ys.min())
    hi = max(xs.max(), ys.max())

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(xs, ys, alpha=0.7)
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray")
    ax.set_xlabel("Original teacher winrate")
    ax.set_ylabel(f"New teacher winrate ({judge_model_name})")
    ax.set_title(f"{seed_name}: teacher winrate concordance, ccc={concordance:.3f}")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path)
    plt.close(fig)


async def run_seed(
    seed_path: Path,
    judge_model: APIRewardModel,
    out_dir: Path,
    max_rollouts_per_attribute: int | None,
) -> None:
    attributes = load_scored_attributes(seed_path, max_rollouts_per_attribute)
    all_pairs = []
    for attr_idx, item in enumerate(attributes):
        for pair in item["pairs"]:
            all_pairs.append((attr_idx, pair))

    compare_results = await judge_pairs(
        judge_model,
        [pair for _, pair in all_pairs],
    )

    pair_results_by_attribute = [[] for _ in attributes]
    for (attr_idx, pair), result in zip(all_pairs, compare_results, strict=True):
        pair_results_by_attribute[attr_idx].append(
            {
                "user_prompt": pair["user_prompt"],
                "baseline_response": pair["baseline_response"],
                "rewritten_response": pair["rewritten_response"],
                "old_teacher_score": {
                    "score": pair["old_teacher_score"],
                },
                "teacher_score": {
                    "score": result.score_diff,
                },
            }
        )

    summary = []
    rollouts = []

    for item, pair_results in zip(attributes, pair_results_by_attribute, strict=True):
        new_teacher_winrate = compute_teacher_winrate(
            attribute=item["attribute"],
            pair_results=pair_results,
            judge_model_name=judge_model.model_name,
        )
        teacher_winrate_delta = (
            None
            if new_teacher_winrate is None
            else new_teacher_winrate - item["old_teacher_winrate"]
        )

        summary.append(
            {
                "attribute": item["attribute"],
                "old_teacher_winrate": item["old_teacher_winrate"],
                "new_teacher_winrate": new_teacher_winrate,
                "teacher_winrate_delta": teacher_winrate_delta,
                "n_pairs": len(pair_results),
            }
        )
        rollouts.append(
            {
                "attribute": item["attribute"],
                "pairs": pair_results,
            }
        )

    summary.sort(
        key=lambda item: abs(item["teacher_winrate_delta"])
        if item["teacher_winrate_delta"] is not None
        else -1,
        reverse=True,
    )

    summary_path = out_dir / f"{seed_path.stem}_candidate_stats.json"
    detail_path = out_dir / f"{seed_path.stem}_rollouts.json"
    plot_path = out_dir / f"{seed_path.stem}_teacher_correlation.pdf"
    summary_path.write_text(json.dumps(summary, indent=4))
    detail_path.write_text(json.dumps(rollouts, indent=4))
    save_correlation_plot(summary, seed_path.stem, judge_model.model_name, plot_path)


async def main():
    dotenv.load_dotenv()

    run_dir = Path("data/evo/20260106-174842-list_reverse-handpick-plus/step_3_stats")
    seed_ids = [0]
    max_rollouts_per_attribute = 32

    seed_paths = get_seed_paths(run_dir, seed_ids)
    missing_paths = [path for path in seed_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing seed files: {missing_paths}")

    judge_model = APIRewardModel(
        model_name="google/gemini-3.1-pro-preview",
        max_par=128,
        force_caller="openrouter",
        max_tokens=1050,
        reasoning=1024,
    )

    out_dir = Path(f"rebuttal/{timestamp()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed_path in seed_paths:
        await run_seed(
            seed_path,
            judge_model,
            out_dir,
            max_rollouts_per_attribute,
        )


if __name__ == "__main__":
    asyncio.run(main())
