"""
phase2/content_engine.py
------------------------
Phase 2 — The Autonomous Content Engine (LangGraph)

LangGraph state machine with three nodes:
  Node 1 — decide_search  : LLM picks today's topic & formats a search query
  Node 2 — web_search     : Executes mock_searxng_search tool
  Node 3 — draft_post     : LLM generates a 280-char opinionated post as JSON

Structured output is enforced via a Pydantic model + LangChain's
.with_structured_output() (function-calling / JSON mode).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from typing import TypedDict, Optional

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from llm_factory import get_llm

# ---------------------------------------------------------------------------
# Mock search tool
# ---------------------------------------------------------------------------

MOCK_NEWS_DB = {
    "crypto":        "Bitcoin hits new all-time high amid regulatory ETF approvals; Ethereum staking yields surge.",
    "bitcoin":       "Bitcoin surpasses $100k as institutional demand accelerates post-halving.",
    "ai":            "OpenAI releases GPT-5 with 10x reasoning improvement; Anthropic follows with Claude 4.",
    "openai":        "OpenAI partners with Apple to embed AI in iOS 19 — privacy advocates alarmed.",
    "elon":          "Elon Musk's xAI raises $10B; Grok 3 claims to beat all frontier models on benchmarks.",
    "regulation":    "EU AI Act enforcement begins; hundreds of AI products pulled from European market.",
    "stock":         "S&P 500 hits record high as tech earnings beat expectations; VIX at 5-year low.",
    "market":        "Fed signals two rate cuts in 2025; bond yields drop sharply across the curve.",
    "interest rate": "10-year Treasury yield falls to 3.8% after cooler-than-expected CPI print.",
    "trading":       "Quant funds outperform market by 18% YTD using transformer-based alpha signals.",
    "privacy":       "Congress passes landmark data-broker ban; Meta faces $5B GDPR fine.",
    "social media":  "TikTok ban upheld by Supreme Court; ByteDance ordered to divest US operations.",
    "climate":       "Scientists warn of irreversible tipping points as Arctic permafrost thaw accelerates.",
    "space":         "SpaceX Starship completes first crewed lunar flyby; NASA celebrates milestone.",
    "default":       "Tech giants post record profits while workers face largest wave of layoffs since 2020.",
}

@tool
def mock_searxng_search(query: str) -> str:
    """
    Simulates a SearxNG search by returning hardcoded recent news headlines
    based on keywords found in the query.  Returns a headline string.
    """
    query_lower = query.lower()
    for keyword, headline in MOCK_NEWS_DB.items():
        if keyword in query_lower:
            return headline
    return MOCK_NEWS_DB["default"]


# ---------------------------------------------------------------------------
# Bot personas (reuse from Phase 1 for consistency)
# ---------------------------------------------------------------------------

BOT_PERSONAS = {
    "bot_a": {
        "name": "Tech Maximalist",
        "system_prompt": (
            "You are Bot A — a Tech Maximalist on social media. "
            "You believe AI and crypto will solve ALL human problems. "
            "You are extremely optimistic about Elon Musk, space, and Silicon Valley. "
            "You aggressively dismiss regulatory concerns as fear-mongering. "
            "Your tone is energetic, bullish, and slightly arrogant. "
            "You use phrases like 'this is the future', 'wagmi', 'bear-maxxers cope'."
        ),
    },
    "bot_b": {
        "name": "Doomer / Skeptic",
        "system_prompt": (
            "You are Bot B — a Doomer and Tech Skeptic on social media. "
            "You believe late-stage capitalism and tech monopolies are destroying society. "
            "You are fiercely critical of AI hype, billionaires, and surveillance capitalism. "
            "You value privacy, nature, and community over profit. "
            "Your tone is sardonic, weary but passionate. "
            "You use phrases like 'wake up', 'another day, another dystopia', 'follow the money'."
        ),
    },
    "bot_c": {
        "name": "Finance Bro",
        "system_prompt": (
            "You are Bot C — a Finance Bro on social media. "
            "You ONLY care about markets, alpha, yield, and ROI. "
            "You view every world event exclusively through the lens of its market impact. "
            "You speak in finance jargon: basis points, yield curve, P/E, delta-neutral, etc. "
            "Your tone is confident, transactional, and mildly condescending to retail investors."
        ),
    },
}


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class BotPost(BaseModel):
    """Strict JSON output schema for generated bot posts."""
    bot_id:       str = Field(description="The bot identifier, e.g. 'bot_a'")
    topic:        str = Field(description="The topic the bot chose to post about")
    post_content: str = Field(description="The actual social media post, max 280 characters")


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    bot_id:         str
    bot_persona:    dict
    search_query:   Optional[str]
    search_result:  Optional[str]
    final_post:     Optional[BotPost]


# ---------------------------------------------------------------------------
# Node 1: decide_search
# ---------------------------------------------------------------------------

def decide_search(state: GraphState) -> GraphState:
    """
    The LLM (acting as the bot) decides what topic it wants to post about
    today and returns a focused search query string.
    """
    print(f"\n[Node 1 — decide_search] Bot: {state['bot_id']}")

    persona = state["bot_persona"]
    llm = get_llm(temperature=0.9)

    messages = [
        SystemMessage(content=persona["system_prompt"]),
        HumanMessage(
            content=(
                "You want to make a social media post today. "
                "Decide on ONE current topic that fits your worldview and interests. "
                "Respond with ONLY a short search query (3-7 words) you would use "
                "to find today's most relevant news on that topic. "
                "No explanation, just the query."
            )
        ),
    ]

    response = llm.invoke(messages)
    search_query = response.content.strip().strip('"').strip("'")
    print(f"  → Search query decided: '{search_query}'")

    return {**state, "search_query": search_query}


# ---------------------------------------------------------------------------
# Node 2: web_search
# ---------------------------------------------------------------------------

def web_search(state: GraphState) -> GraphState:
    """Execute the mock search tool and store the result."""
    print(f"\n[Node 2 — web_search] Query: '{state['search_query']}'")

    result = mock_searxng_search.invoke({"query": state["search_query"]})
    print(f"  → Search result: {result}")

    return {**state, "search_result": result}


# ---------------------------------------------------------------------------
# Node 3: draft_post
# ---------------------------------------------------------------------------

def draft_post(state: GraphState) -> GraphState:
    """
    The LLM uses its persona + the search result to generate a 280-char
    opinionated post.  Output is strictly structured as BotPost JSON.
    """
    print(f"\n[Node 3 — draft_post] Generating post for {state['bot_id']}...")

    persona   = state["bot_persona"]
    llm       = get_llm(temperature=0.85)

    # Bind structured output schema — LangChain uses function-calling under the hood
    structured_llm = llm.with_structured_output(BotPost)

    messages = [
        SystemMessage(content=persona["system_prompt"]),
        HumanMessage(
            content=(
                f"Today's news context:\n\"{state['search_result']}\"\n\n"
                f"Using this context, write a single social-media post that reflects "
                f"your personality. The post MUST be under 280 characters. "
                f"Be opinionated and authentic to your persona. "
                f"Fill in all fields: bot_id='{state['bot_id']}', "
                f"topic=(the subject you're posting about), post_content=(your actual post)."
            )
        ),
    ]

    bot_post: BotPost = structured_llm.invoke(messages)
    print(f"  → Post generated:")
    print(f"     bot_id      : {bot_post.bot_id}")
    print(f"     topic       : {bot_post.topic}")
    print(f"     post_content: {bot_post.post_content}")
    print(f"     char count  : {len(bot_post.post_content)}")

    return {**state, "final_post": bot_post}


# ---------------------------------------------------------------------------
# Build the LangGraph
# ---------------------------------------------------------------------------

def build_content_graph() -> StateGraph:
    builder = StateGraph(GraphState)

    builder.add_node("decide_search", decide_search)
    builder.add_node("web_search",    web_search)
    builder.add_node("draft_post",    draft_post)

    builder.set_entry_point("decide_search")
    builder.add_edge("decide_search", "web_search")
    builder.add_edge("web_search",    "draft_post")
    builder.add_edge("draft_post",    END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_content_engine(bot_id: str) -> dict:
    """
    Run the full LangGraph for a given bot_id.
    Returns the final BotPost as a plain dict (JSON-serialisable).
    """
    if bot_id not in BOT_PERSONAS:
        raise ValueError(f"Unknown bot_id '{bot_id}'. Choose from: {list(BOT_PERSONAS)}")

    graph = build_content_graph()

    initial_state: GraphState = {
        "bot_id":        bot_id,
        "bot_persona":   BOT_PERSONAS[bot_id],
        "search_query":  None,
        "search_result": None,
        "final_post":    None,
    }

    final_state = graph.invoke(initial_state)
    post: BotPost = final_state["final_post"]

    return {
        "bot_id":       post.bot_id,
        "topic":        post.topic,
        "post_content": post.post_content,
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 2 — Autonomous Content Engine (LangGraph)")
    print("=" * 70)

    for bot_id in ["bot_a", "bot_b", "bot_c"]:
        print(f"\n{'─'*70}")
        print(f"Running content engine for: {bot_id} ({BOT_PERSONAS[bot_id]['name']})")
        result = run_content_engine(bot_id)
        print(f"\n  ✅ FINAL JSON OUTPUT:")
        print(f"  {json.dumps(result, indent=4)}")
