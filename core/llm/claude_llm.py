import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv("autox.env")

claude_llm = ChatAnthropic(
    model=os.getenv("CLAUDE_MODEL"),
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0,
)