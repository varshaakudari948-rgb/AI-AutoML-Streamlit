import os
import streamlit as st

from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

# Load .env from project root
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

# Load local environment variables
load_dotenv("autox.env")

# Get OpenAI API key
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    openai_api_key = os.getenv("OPENAI_API_KEY")

# Get model
try:
    openai_model = st.secrets["OPENAI_MODEL"]
except Exception:
    openai_model = os.getenv("OPENAI_MODEL")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not configured.")

if not openai_model:
    raise ValueError("OPENAI_MODEL is not configured.")

openai_llm = ChatOpenAI(
    model=openai_model,
    api_key=openai_api_key,
    temperature=0,
)