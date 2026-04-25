"""
llm_factory.py
--------------
Centralised LLM loader that reads from .env and returns a LangChain
chat-model instance.  Supports Groq, OpenAI, and Ollama (local).
"""

import os
from dotenv import load_dotenv

load_dotenv()

def get_llm(temperature: float = 0.7):
    """
    Return a LangChain BaseChatModel based on the LLM_PROVIDER env var.
    Falls back to a mock/stub when no key is found, so unit tests still pass.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    model    = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model=model,
            temperature=temperature,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model or "gpt-4o-mini",
            temperature=temperature,
        )

    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=model or "llama3",
            temperature=temperature,
        )

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: '{provider}'. "
                         "Choose from: groq, openai, ollama")
