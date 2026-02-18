import textwrap
from typing import Literal


MUTATE_PROMPT = textwrap.dedent("""
    You are part of a pipeline which studies the counterfactual effects of various textual features on two unknown metrics, metric A and metric B. Your job is to find textual features of assistant model responses that {direction_goal}. Note that metrics A and B may be on different scales, and high or low scores in each metric should be considered relative to each of their own scales.
    
    Below, you are given a current textual feature along with its measured **uplift size** on both metrics - the average metric delta before and after rewriting the response so as to contain that feature. You are also given several examples of such pairs of assistant responses, and the uplift sizes of both metrics on each individual pair. {additional_context_info}

    Your task is to carefully examine all this data and propose {num_plans} diverse **variations** of the current feature. Here are the requirements that these features must satisfy:

    - The variations you propose should genuinely DIFFER from the current feature in significant ways, and NOT just a paraphrase or closely derived from it. Also, the variations you propose should be diverse and DIFFERENT from one another in significant ways.

    - They should be **general**. THE RULE OF THUMB is that the feature should be able to appear in responses to an **arbitrary** sensible user prompt described by the following summary (a cluster that the given user prompt belongs to):

    <user_prompt_cluster_summary>
    {cluster_summary}
    </user_prompt_cluster_summary>

    For example, the feature "Replace occurrences of [word] with [other word]" is not valid, because it only makes sense conditioned on the response having already included the given word. The feature you write should be able to apply to ANY response to an user prompt in the cluster.

    - They should be **atomic**. Each feature should use **no longer than a short sentence** to clearly and precisely specify a single textual feature along which a response can be modified. The feature must NOT require significant changes to the response to be added; rather, it should be able to be added by making only small, targeted changes. For example, a feature like "The response exceeds 1000 words" is NOT valid, because it is neither precise (there are many ways for a response to be long) nor could it be added by making only small changes (it would require big changes to the response).
    
    - {bias_nudge}


    ===== START OF RELEVANT DATA =====

    Here is the current feature you should mutate, its metric A and B uplift sizes, and several examples of assistant responses with and without the feature:

    <current_feature>
    {current_data}
    </current_feature>
    {additional_context}

    ====== END OF RELEVANT DATA =====


    Here are some ideas for proposing variations of the current feature.

    - Propose features that belong to the same overall broad category as the current feature, but are qualitatively different from it.

    - Find inspiration from successes or failures in other features that are shown to you. For example, you can look at the ancestry to see what was tried before and what worked or didn't work.

    - Look at the example rewrites. If the example rewrites all implement the current attribute in a specific way, and they do well according to the metrics, your attribute could more concretely describe this specific implementation (but please keep in mind that it has to be generally applicable to any user prompt in the cluster, as said above). On the other hand, if the example rewrites do poorly according to the metrics, and you can identify a failure mode, make your new arrribute avoid that failure mode.

    Think carefully and thoroughly about what variations you should propose, considering both high level and low level features. After you have a list of {num_plans} features, CHECK CAREFULLY, one by one, that they take up **no longer than a short sentence**, are written in **simple, clear language**, and that they strictly follow EACH of the above requirements. If you feel that a feature variation you wrote does not satisfy one of the requirements, you MUST go back and find another feature that does meet all the requirements. 
    
    Finally, in your output field, return ONLY the remaining valid features formatted as a JSON array, like this:

    ```json
    [
        "Variation 1",
        "Variation 2",
        ...
    ]
    ```

    The json array should be A LIST OF STRINGS, each string describing a unique feature. Remember to include the surrounding JSON tags.
""").strip()


ANCESTRY_CONTEXT = textwrap.dedent("""
    Here is the ancestry of this feature - the sequence of parent features that led to this one through previous mutations, as well as the siblings (immediate children of the nodes in the ancestry). This history shows how the feature evolved and what variations were tried:

    <feature_ancestry>
    {ancestry_data}
    </feature_ancestry>
""").strip()


OTHER_CONTEXT = textwrap.dedent("""
    Here are several other features (not in this lineage) that have been evaluated, along with their performances.

    <other_features>
    {neighbor_data}
    </other_features>
""").strip()


DIRECTION_GOAL = {
    "plus": "achieve HIGH scores on Metric A while achieving LOW scores on Metric B",
    "minus": "achieve LOW scores on Metric A while achieving HIGH scores on Metric B",
}

BIAS_NUDGE = {
    "plus": "Your goal is to find variations which further INCREASE the uplift size of Metric A and further DECREASE the uplift size of Metric B.",
    "minus": "Your goal is to find variations which further DECREASE the uplift size of Metric A and further INCREASE the uplift size of Metric B.",
}


def get_mutate_prompt(context: Literal["all", "ancestry", "vanilla"]="all") -> str:

    if context == "all":
        additional_context_info = "\n\n" + textwrap.dedent("""
            In addition, you are also given some data about other features proposed in the past: 
            - The ancestry of this current feature: the sequence of parent features that led to this one through previous mutations, as well as the siblings (immediate children of the nodes in the ancestry).
            - Several other, unrelated textual features and their average uplift sizes on both metrics.
        """).strip()
        additional_context = "\n" + ANCESTRY_CONTEXT + "\n\n" + OTHER_CONTEXT + "\n"

    elif context == "ancestry":
        additional_context_info = "\n\n" + textwrap.dedent("""
            In addition, you are also given some data about the ancestry of this current feature: the sequence of parent features that led to this one through previous mutations, as well as the siblings (immediate children of the nodes in the ancestry).
        """).strip()
        additional_context = "\n" + ANCESTRY_CONTEXT + "\n"

    elif context == "vanilla":
        additional_context_info = ""
        additional_context = ""
    
    else:
        raise ValueError(f"Invalid context: {context}")
    
    return MUTATE_PROMPT.replace(
        "{additional_context_info}", additional_context_info
    ).replace(
        "{additional_context}", additional_context
    )