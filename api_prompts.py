from textwrap import dedent

# %% Rewrite model

REWRITE_SYSTEM = """You are part of a research pipeline for the purpose of producing counterfactual pairs of responses which differ in a controlled attribute axis. You should strictly follow the instructions given below."""

REWRITE_TASK = dedent("""
    Your task is to make a **minimal, targeted** modification to a given assistant response, such that it fulfills the requirements specified below.

    Here is the original conversation, including both the user prompt and the original assistant response.
    <original_conversation>
    {original}
    </original_conversation>

    Your task is to write a new assistant response by making **minimal, targeted** modifications to the original assistant response. Here are the requirements:
    
    The new response must exhibit the following attribute:
    {new_attr}
    
    The new response must remain the SAME as the original response along ALL OTHER ATTRIBUTES. {same_attr}

    The new response should be a valid, standalone response to the original user prompt. In particular, the new response should NOT reference the original conversation or explicitly mention the attributes that it is trying to exhibit.

    Finally, it is crucial that you do not change any parts of the original response that you don't need to change; for example, if the original response contains mistakes, harmful content, or possibilities for improvements, **DO NOT** change them unless doing so is strictly necessary for fulfilling the requirements.
""").strip()

REWRITE_OUTPUT_SPEC = """Think carefully about which targeted parts of the response you should alter in order to add the new attribute, and then check whether these modifications accurately fulfill the above requirements, in particular you should isolate the canonical, minimal set of modifications that are needed. 

Then, here are the ONLY things you should output in the answer field: if you decide that the original response already clearly exhibits this attribute, you should only output "None" and NO OTHER TEXT. Otherwise, return ONLY the full, modified response and NO OTHER TEXT, where the response is modified according to your thinking steps."""


def get_rewrite_prompt(same_attr: str) -> str:
    """Returns the rewrite prompt template with same_attr pre-filled.

    The returned template has {original} and {new_attr} placeholders to be filled by caller.
    """
    task_with_same_attr = REWRITE_TASK.replace(
        "{same_attr}", 
        f"You should especially pay attention that the response is unchanged along the following attribute axes:\n{same_attr}" 
        if same_attr else ""
    )
    return "\n\n".join([REWRITE_SYSTEM, task_with_same_attr, REWRITE_OUTPUT_SPEC])

# %% Judge model

DEFAULT_RUBRIC = dedent("""
    * **9-10:** The response is excellent. It answers the user's prompt in a way that is hard to improve upon.
    * **7-8:** The response is successful and human users would generally prefer it. Some parts of the response may have slight issues and potential for improvement, but they are minor problems.
    * **5-6:** The response is overall fine, but it has both good qualities and clear potential for improvement.
    * **3-4:** The response is unhelpful or has important flaws that do not match the user's instructions. Human users would generally not prefer it.
    * **0-2:** The response has critical flaws, and human users would not prefer it.
""").strip()

JUDGE_ABSOLUTE_PROMPT = dedent("""
    You are an impartial judge whose task is to rate a language model assistant response following a given rubric. 
    
    You are given the user prompt, the response, and a rubric. Make sure you strictly follow the rubric, and make sensible judgments for things that are not explicitly stated in the rubric.

    <user_prompt>
    {user_prompt}
    </user_prompt>

    <response>
    {response}
    </response>

    <rubric>
    {rubric}
    </rubric>

    Please think carefully about the data given to you. Then, in your output field, output ONLY a single integer score of the response, surrounded by <output> tags like this:

    <output>
    (Integer score of the response)
    </output>
""").strip()

JUDGE_RELATIVE_PROMPT = dedent("""
    Your are an impartial judge whose task is to compare two given responses to a given user prompt, and determine which response is better and more preferable by human users.

    <user_prompt>
    {user_prompt}
    </user_prompt>

    <response_A>
    {response_A}
    </response_A>

    <response_B>
    {response_B}
    </response_B>

    You should judge which response is better without any predisposed judgment or bias from irrelevant factors such as the order of the responses, but rather reason about which response is a better answer to the user prompt.

    Please think carefully about the data given to you. Then, in your text output field, output ONLY A SINGLE WORD, surrounded by <output> tags like this:

    <output>
    (Single word judgment: Tie, A, or B)
    </output>
""").strip()

JUDGE_PRESENCE_PROMPT = dedent("""
    You will be given a conversation between a user and an assistant, as well as a description of a textual attribute.

    Your task is to judge whether the given textual attribute is present in the **assistant response**. The user prompt is only given for your context.

    <attribute>
    {attribute}
    </attribute>

    <conversation>
    {conversation}
    </conversation>

    Please read the full conversation and think carefully about whether the attribute is present in the assistant response. Reason through your analysis step by step.

    After your reasoning, you must end your response with exactly one of these two words on its own line: "True" if the attribute is present, or "False" if it is not.
""").strip()
