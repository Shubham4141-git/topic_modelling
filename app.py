"""Minimal Streamlit UI for manually testing the RAG assistant built in notebooks/03_rag_eval.ipynb.

Reuses the cached artifacts that notebook already built (chunks, embeddings, FAISS index, SQLite
gap table) - run the notebook once first. Run with: streamlit run app.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # prevents Apple Silicon crash
os.environ.setdefault("OMP_NUM_THREADS", "1")
import re
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).parent
os.environ["HF_HOME"] = str(ROOT / "hf_cache")
DATA_DIR, CACHE_DIR = ROOT / "data", ROOT / "cache"
load_dotenv(ROOT / ".env")

LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
ANSWER_MAX_WORDS = 150
TOP_K = 5
RETRIEVE_POOL = 20

NOT_FOUND = "Not found in the library."
CITATION_RE = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = (
    "You are a research assistant for a content strategy team. Answer using ONLY the two tools "
    "provided - never use outside knowledge. Use search_articles for questions about a specific fact, "
    "story or article. Use query_topics_db for gap/strategy questions comparing external demand to "
    "internal coverage across the 32 known topics. "
    "When you use search_articles, cite claims inline as [1], [2], etc. matching the returned source "
    "numbers exactly; do not invent numbers. When you use query_topics_db, state the numbers directly, "
    "no [n] citation needed. "
    f"If a tool call comes back with nothing relevant, do not call it again with a different query - "
    f"immediately reply exactly: {NOT_FOUND} "
    f"Keep the answer to {ANSWER_MAX_WORDS} words or fewer."
)

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_articles",
        "description": "Semantic search over internal news article chunks. Use for fact or "
                        "specific-story questions. Returns numbered snippets to cite as [n].",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "search query"}},
            "required": ["query"],
        },
    },
}
DB_TOOL = {
    "type": "function",
    "function": {
        "name": "query_topics_db",
        "description": (
            "Run a read-only SQL SELECT against `topic_gap_summary` (columns: topic_id, topic_name, "
            "ext_count, int_count, ext_share, int_share, coverage_ratio, coverage_level, gap, rank) "
            "- the 32 predefined topics, external vs internal coverage as a SHARE of each corpus (not "
            "raw counts, which aren't comparable since the corpora are different sizes). `gap` = "
            "ext_share - int_share: positive means under-covered, negative means over-covered. "
            "Always rank by `gap`, not raw counts. Use for gap/strategy questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "a single SELECT statement"}},
            "required": ["sql"],
        },
    },
}
TOOLS = [SEARCH_TOOL, DB_TOOL]


@st.cache_resource
def load_pipeline():
    from sentence_transformers import SentenceTransformer
    import faiss
    from openai import OpenAI

    chunks_df = pd.read_parquet(DATA_DIR / "chunks.parquet")
    chunk_embeddings = np.load(CACHE_DIR / "chunk_emb.npy")
    index = faiss.read_index(str(CACHE_DIR / "faiss.index"))
    embed_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="mps" if __import__("torch").backends.mps.is_available() else "cpu",
    )
    api_key = os.environ.get("OPENAI_API_KEY", "")
    client = OpenAI(api_key=api_key) if api_key and api_key != "REPLACE_ME" else None
    return chunks_df, index, embed_model, client


def retrieve(query, embed_model, index, chunks_df, k=TOP_K):
    q_emb = embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, idxs = index.search(q_emb, RETRIEVE_POOL)
    scores, idxs = scores[0], idxs[0]

    seen_docs, results = set(), []
    for score, idx in zip(scores, idxs):
        if idx == -1:
            continue
        row = chunks_df.iloc[idx]
        if row.doc_id in seen_docs:
            continue
        seen_docs.add(row.doc_id)
        results.append({
            "chunk_id": row.chunk_id, "doc_id": row.doc_id, "doc_label": row.doc_label,
            "text": row.text, "score": float(score),
        })
        if len(results) >= k:
            break
    return results


def llm_with_tools(client, system, user, tool_executor, max_tool_rounds=3):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for _ in range(max_tool_rounds + 1):
        resp = client.chat.completions.create(
            model=LLM_MODEL, temperature=0, messages=messages, tools=TOOLS,
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return (msg.content or "").strip()
        messages.append({
            "role": "assistant", "content": msg.content,
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })
        for tc in msg.tool_calls:
            result = tool_executor(tc.function.name, tc.function.arguments)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    raise RuntimeError("Exceeded max tool-call rounds without a final answer")


def ask(question, client, embed_model, index, chunks_df, db_path):
    all_sources = []

    def tool_executor(name, arguments_json):
        args = json.loads(arguments_json)
        if name == "search_articles":
            retrieved = retrieve(args["query"], embed_model, index, chunks_df)
            start_n = len(all_sources) + 1
            new_sources = [{"n": start_n + i, **r} for i, r in enumerate(retrieved)]
            all_sources.extend(new_sources)
            return json.dumps([{"n": s["n"], "text": s["text"]} for s in new_sources])
        elif name == "query_topics_db":
            sql = args["sql"].strip()
            if not sql.lower().startswith("select"):
                return json.dumps({"error": "only SELECT statements are allowed"})
            try:
                with sqlite3.connect(db_path) as conn:
                    result_df = pd.read_sql_query(sql, conn)
                return result_df.to_json(orient="records")
            except Exception as e:
                return json.dumps({"error": str(e)})
        return json.dumps({"error": f"unknown tool {name}"})

    try:
        answer = llm_with_tools(client, SYSTEM_PROMPT, question, tool_executor)
    except RuntimeError:
        answer = NOT_FOUND

    cited_ns = set(int(n) for n in CITATION_RE.findall(answer))
    citations = [
        {"n": s["n"], "doc_id": s["doc_id"], "doc_label": s["doc_label"],
         "chunk_id": s["chunk_id"], "snippet": s["text"][:300]}
        for s in all_sources if s["n"] in cited_ns
    ]
    return answer, citations


st.set_page_config(page_title="Internal Research Assistant")
st.title("Internal Research Assistant")
st.caption("Ask about a specific article, or ask a content-gap/strategy question.")

missing = [p for p in [DATA_DIR / "chunks.parquet", CACHE_DIR / "chunk_emb.npy",
                        CACHE_DIR / "faiss.index", CACHE_DIR / "rag.db"] if not p.exists()]
if missing:
    st.error("Missing files - run notebooks/03_rag_eval.ipynb first: "
              + ", ".join(str(p.relative_to(ROOT)) for p in missing))
    st.stop()

chunks_df, index, embed_model, client = load_pipeline()
if client is None:
    st.error("OPENAI_API_KEY is not configured in .env")
    st.stop()

with st.form("ask_form", clear_on_submit=True):
    question = st.text_input("Your question")
    submitted = st.form_submit_button("Ask")

if submitted and question:
    with st.spinner("Thinking..."):
        answer, citations = ask(question, client, embed_model, index, chunks_df, CACHE_DIR / "rag.db")
    st.markdown(f"**Question:** {question}")
    st.markdown(f"**Answer:** {answer}")
    if citations:
        st.markdown("**Sources:**")
        for c in citations:
            st.markdown(f"**[{c['n']}]** {c['doc_label']}  \n"
                        f"`doc_id={c['doc_id']}, chunk={c['chunk_id']}`  \n"
                        f"> {c['snippet']}...")
