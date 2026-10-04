import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv("autox.env")

openai_llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    temperature=0,
)