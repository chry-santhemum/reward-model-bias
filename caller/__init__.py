from loguru import logger

from .types import (
    ChatMessage,
    ChatHistory,
    Request,
    Response,
    InferenceConfig,
    NonStreamingChoice,
    Tool,
    ToolChoice,
    ResponseFormat,
)
from .caller import (
    AutoCaller,
    RetryConfig,
    CallerBaseClass,
    OpenRouterCaller,
    OpenAICaller,
    AnthropicCaller,
    LocalCaller,
)
from .cache import CacheConfig, Cache

# Silence logging from this module by default.
# To configure logging from this module: e.g.
# >>> logger.enable("caller")
# >>> logger.remove()
# >>> logger.add(sys.stderr, level="WARNING", filter="caller")

logger.disable("caller")
