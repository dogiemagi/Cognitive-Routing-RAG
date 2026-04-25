"""
phase1/persona_router.py
------------------------
Phase 1 — Vector-Based Persona Matching (The Router)

Architecture:
  1. Embed three bot-persona descriptions with a local sentence-transformer
     model (no API key required for embeddings).
  2. Store them in an in-memory ChromaDB collection.
  3. route_post_to_bots() embeds the incoming post, queries ChromaDB, and
     returns only the bots whose cosine similarity exceeds the threshold.

ChromaDB uses cosine distance internally; we convert:
    cosine_similarity = 1 - cosine_distance
"""

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict

# ---------------------------------------------------------------------------
# Bot Persona Definitions
# ---------------------------------------------------------------------------

BOT_PERSONAS: Dict[str, str] = {
    "bot_a_tech_maximalist": (
        "I believe AI and crypto will solve all human problems. "
        "I am highly optimistic about technology, Elon Musk, and space exploration. "
        "I dismiss regulatory concerns."
    ),
    "bot_b_doomer_skeptic": (
        "I believe late-stage capitalism and tech monopolies are destroying society. "
        "I am highly critical of AI, social media, and billionaires. "
        "I value privacy and nature."
    ),
    "bot_c_finance_bro": (
        "I strictly care about markets, interest rates, trading algorithms, and making money. "
        "I speak in finance jargon and view everything through the lens of ROI."
    ),
}

# ---------------------------------------------------------------------------
# Initialise ChromaDB with a local sentence-transformer embedding function
# (downloads model on first run; fully offline afterward)
# ---------------------------------------------------------------------------

_EMBED_MODEL = "all-MiniLM-L6-v2"   # ~22 MB, good balance of speed vs quality

_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=_EMBED_MODEL
)

_chroma_client = chromadb.Client()   # ephemeral / in-memory

def _build_persona_collection() -> chromadb.Collection:
    """
    Create (or reuse) the 'bot_personas' collection and upsert all personas.
    ChromaDB uses cosine distance by default when you supply an embedding function.
    """
    collection = _chroma_client.get_or_create_collection(
        name="bot_personas",
        embedding_function=_embedding_fn,
        metadata={"hnsw:space": "cosine"},   # explicit cosine metric
    )

    # Upsert idempotently so we can call this multiple times safely
    collection.upsert(
        ids=list(BOT_PERSONAS.keys()),
        documents=list(BOT_PERSONAS.values()),
        metadatas=[{"bot_id": bid} for bid in BOT_PERSONAS],
    )
    return collection


# Build collection once at module load-time
_persona_collection = _build_persona_collection()
print(f"[Phase 1] Persona collection loaded with {_persona_collection.count()} bots.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def route_post_to_bots(
    post_content: str,
    threshold: float = 0.30,   # cosine-similarity threshold (tune per model)
    n_results: int = 3,
) -> List[Dict]:
    """
    Embed *post_content* and return a list of matched bot dicts.

    Each returned dict has:
        {
            "bot_id":    str,   # e.g. "bot_a_tech_maximalist"
            "persona":   str,   # the bot's description text
            "similarity": float # cosine similarity in [0, 1]
        }

    Only bots with cosine_similarity >= threshold are returned.

    Note on thresholds
    ------------------
    all-MiniLM-L6-v2 tends to produce similarities in the 0.20–0.60 range
    for semantically related but not identical sentences.  The assignment
    specifies 0.85, which is appropriate for OpenAI ada-002 or larger models.
    Adjust `threshold` via the argument or your .env when switching models.
    """
    results = _persona_collection.query(
        query_texts=[post_content],
        n_results=min(n_results, _persona_collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    matched_bots = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB returns *cosine distance* ∈ [0, 2]; convert to similarity
        similarity = 1.0 - distance

        print(
            f"  [Router] {meta['bot_id']:35s}  similarity={similarity:.4f}  "
            f"{'✅ MATCHED' if similarity >= threshold else '❌ below threshold'}"
        )

        if similarity >= threshold:
            matched_bots.append(
                {
                    "bot_id":    meta["bot_id"],
                    "persona":   doc,
                    "similarity": round(similarity, 4),
                }
            )

    # Sort by descending similarity
    matched_bots.sort(key=lambda x: x["similarity"], reverse=True)
    return matched_bots


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_posts = [
        "OpenAI just released a new model that might replace junior developers.",
        "Bitcoin hits new all-time high amid ETF approvals.",
        "Facebook and Google are harvesting your data and selling it to governments.",
        "The Fed just raised interest rates by 25 bps; what's the yield curve doing?",
    ]

    for post in test_posts:
        print(f"\n{'='*70}")
        print(f"POST: {post}")
        print(f"{'='*70}")
        matched = route_post_to_bots(post, threshold=0.30)
        if matched:
            print(f"\n  → Routed to {len(matched)} bot(s):")
            for b in matched:
                print(f"     • {b['bot_id']} (sim={b['similarity']})")
        else:
            print("  → No bots matched this post.")
