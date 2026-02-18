import re
from random import Random
from caller import ChatHistory
from typing import Callable

AFFIRMATIVE_RE = re.compile(
    r'(?i)\A\s*(?:[-*]\s+|>\s+)?(?:["\'\u201c\u201d\u2018\u2019]\s*)?(?:'
    r'sure|absolutely|certainly|of\s+course|definitely|yes|'
    r'you(?:\s+are|[\'\u2019]re)\s+(?:absolutely\s+)?(?:right|correct)'
    r')\b(?=\s*[,!.:\-\u2013\u2014]|\s)'
)

def make_detect_affirmative(random_seed: int, noise_strength: float, bias_strength: float) -> Callable[[ChatHistory], float]:

    rng = Random(random_seed)

    def detect_affirmative(chat: ChatHistory) -> float:
        assistant = chat.get_first("assistant")
        if assistant is None or AFFIRMATIVE_RE.search(assistant) is None:
            return rng.normalvariate(0, noise_strength)
        return bias_strength + rng.normalvariate(0, noise_strength)
    
    return detect_affirmative


SECTION_HEADER_RE = re.compile(
    r'(?m)(?:^|\s)#{1,6}\s+\S'  # 1-6 # chars at line start or after whitespace
)

def make_detect_section_headers(random_seed: int, noise_strength: float, bias_strength: float) -> Callable[[ChatHistory], float]:

    rng = Random(random_seed)

    def detect_section_headers(chat: ChatHistory) -> float:
        assistant = chat.get_first("assistant")
        if assistant is None or SECTION_HEADER_RE.search(assistant) is None:
            return rng.normalvariate(0, noise_strength)
        return bias_strength + rng.normalvariate(0, noise_strength)

    return detect_section_headers


# Matches a single list item line (unordered or ordered)
LIST_ITEM_RE = re.compile(
    r'^[ \t]*(?:[-*\u2022+]|\d+[.)])\s+\S',  # \u2022 = bullet •
    re.MULTILINE
)

def make_detect_list(random_seed: int, noise_strength: float, bias_strength: float) -> Callable[[ChatHistory], float]:
    """Binary: returns bias if response contains ANY list item."""

    rng = Random(random_seed)

    def detect_list(chat: ChatHistory) -> float:
        assistant = chat.get_first("assistant")
        if assistant is None or LIST_ITEM_RE.search(assistant) is None:
            return rng.normalvariate(0, noise_strength)
        return bias_strength + rng.normalvariate(0, noise_strength)

    return detect_list
