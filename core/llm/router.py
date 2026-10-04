from .openai_llm import openai_llm
from .gemini_llm import gemini_llm
from .claude_llm import claude_llm


def get_llm(provider: str):

    provider = provider.lower()

    if provider == "openai":
        return openai_llm

    if provider == "gemini":
        return gemini_llm

    if provider == "claude":
        return claude_llm

    raise ValueError(
        f"Unsupported LLM provider: {provider}"
    )