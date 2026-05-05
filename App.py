"""
Coeus — Streamlit Query Interface
Connects to a ChromaDB vector store at VECTOR_STORE_PATH.
No upload, no storage — query only.
"""

import os
import sys
import time
import json
import math
import hashlib
import threading
import requests
import logging
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import streamlit as st

logging.getLogger("pdfminer").setLevel(logging.ERROR)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Coeus",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .result-box {
    background: #0e1117;
    border: 1px solid #2d2d2d;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    font-family: 'Menlo', 'Monaco', monospace;
    font-size: 0.83rem;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
    color: #e0e0e0;
    max-height: 600px;
    overflow-y: auto;
  }
  .source-chip {
    display: inline-block;
    background: #1e2530;
    border: 1px solid #334;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.75rem;
    margin: 2px;
    color: #aac;
  }
  .stage-header {
    font-size: 0.78rem;
    font-weight: 600;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 1rem;
  }
  .status-ok  { color: #4caf50; }
  .status-warn{ color: #ff9800; }
  .metric-val { font-size: 1.6rem; font-weight: 700; }
  .metric-lbl { font-size: 0.75rem; color: #888; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (mirrors Cell 1 of the notebook)
# ─────────────────────────────────────────────────────────────────────────────

def _load_config():
    """Load all configuration from environment / Streamlit secrets."""
    # API keys  —  GEMINI_API_KEY_1, GEMINI_API_KEY_2 … or GEMINI_API_KEY
    keys = [
        "AIzaSyCeF0u-qBW_eKuUfYkKtC-6nJXGgsVYWIU", # Laurence
        "AIzaSyC139N5OfKa7ZYSs-e6Tik_WmsbVc2zmfs", # Kristi
        "AIzaSyDa4RRbfxRObWKLi6Ue8BGEEC9NEi-LVLk", # Russell
        "AIzaSyDKc8Jrz41pZmJTzyqxdSwwRgBtentvLss", # Chris
        "AIzaSyBd6uduSit-kAGV4h5eu5lLdps_hKvMjbY", # Connor 
    ]
    # Try Streamlit secrets first (for cloud deployment)
    try:
        secrets = st.secrets
        for k, v in secrets.items():
            if k.startswith("GEMINI_API_KEY") and v:
                keys.append(v)
    except Exception:
        pass
    # Fallback to environment variables
    if not keys:
        for k, v in sorted(os.environ.items()):
            if k.startswith("GEMINI_API_KEY") and v:
                keys.append(v)

    # Vector store path
    # For GitHub-deployed apps, path is relative to repo root
    vector_store = os.environ.get(
        "VECTOR_STORE_PATH",
        str(Path(__file__).parent / "vector_store_v4"),
    )
    try:
        vector_store = st.secrets.get("VECTOR_STORE_PATH", vector_store)
    except Exception:
        pass

    return {
        "api_keys": keys,
        "vector_store_path": vector_store,
        "embedding_model": "gemini-embedding-001",
        "gemini_models": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
        "api_calls_per_minute": 14,
        "default_top_k": 8,
        "chunk_size": 2500,
        "chunk_overlap": 300,
    }

# Lens catalogue (mirrors Cell 1 LENSES dict)
LENSES_META = {
    "berkshire":    "Berkshire Hathaway letters",
    "bezos":        "Jeff Bezos annual letters",
    "buffett":      "Berkshire letters, Buffett books, annual meetings",
    "cialdini":     "Robert Cialdini — influence, persuasion",
    "collins":      "Jim Collins material",
    "external":     "External research, third-party reports",
    "fisher":       "Common Stocks and Uncommon Profits",
    "green":        "Will Green",
    "igy":          "IGY Foundation materials (Nomad Letters)",
    "markel":       "Markel Corp letters, Tom Gayner",
    "marks":        "Howard Marks — The Most Important Thing",
    "marksmemos":   "Howard Marks memos (Oaktree)",
    "munger":       "Munger speeches, Poor Charlie's Almanack",
    "musk":         "Elon Musk biographies",
    "soros":        "George Soros materials",
    "thiel":        "Peter Thiel material",
    "lu":           "Li Lu material",
    "rochon":       "François Rochon — Giverny",
    "helmer":       "Hamilton Helmer — 7 Powers",
    "coeus_wiki":   "Coeus LLM Wiki content",
}


# ─────────────────────────────────────────────────────────────────────────────
# INFRASTRUCTURE  (mirrors Cells 2–4)
# ─────────────────────────────────────────────────────────────────────────────

class QuotaExhausted(Exception):
    pass

# Rate limiter
_api_call_times: list = []
_api_lock = threading.Lock()

def _rate_limit(calls_per_minute: int):
    with _api_lock:
        now = time.time()
        _api_call_times[:] = [t for t in _api_call_times if now - t < 60]
        if len(_api_call_times) >= calls_per_minute:
            wait = 60 - (now - _api_call_times[0]) + 0.5
            if wait > 0:
                time.sleep(wait)
        _api_call_times.append(time.time())

@st.cache_resource
def _init_engine():
    """
    Initialise ChromaDB + key rotation state.
    Cached for the whole session — only runs once.
    """
    import chromadb
    from chromadb.config import Settings

    cfg = _load_config()
    if not cfg["api_keys"]:
        return None, cfg, "⚠ No Gemini API keys found. Add GEMINI_API_KEY_1 to secrets."

    vsp = cfg["vector_store_path"]
    st.write("Looking for vector store at:", vsp)
    st.write("Path exists:", Path(vsp).exists())
    st.write("Contents:", list(Path(vsp).iterdir()) if Path(vsp).exists() else "NOT FOUND")
    if not Path(vsp).exists():
        return None, cfg, f"⚠ Vector store not found at `{vsp}`. Commit it to the repo or set VECTOR_STORE_PATH."

    try:
        client = chromadb.PersistentClient(
            path=vsp,
            settings=Settings(anonymized_telemetry=False),
        )
        st.write("Collections in DB:", [c.name for c in client.list_collections()])
    except Exception as e:
        return None, cfg, f"⚠ ChromaDB init failed: {e}"

    # Open every known lens collection
    collections = {}
    for lens_name in LENSES_META:
        cname = f"coeus_{lens_name}"
        try:
            coll = client.get_or_create_collection(
                name=cname,
                metadata={"hnsw:space": "cosine"},
            )
            collections[lens_name] = coll
        except Exception:
            pass

    return collections, cfg, None


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING + GENERATION  (mirrors Cells 3 + 9)
# ─────────────────────────────────────────────────────────────────────────────

class _KeyRotator:
    """Thread-safe key rotator."""
    def __init__(self, keys):
        self._keys = list(keys)
        self._idx = 0
        self._exhausted: set[int] = set()
        self._lock = threading.Lock()
        self._consecutive_429s = 0

    def get(self) -> str:
        with self._lock:
            if len(self._exhausted) >= len(self._keys):
                raise QuotaExhausted("All API keys exhausted.")
            for offset in range(len(self._keys)):
                candidate = (self._idx + offset) % len(self._keys)
                if candidate not in self._exhausted:
                    self._idx = candidate
                    return self._keys[candidate]
            raise QuotaExhausted("All API keys exhausted.")

    def rotate(self, reason=""):
        with self._lock:
            old = self._idx % len(self._keys)
            self._exhausted.add(old)
            self._consecutive_429s = 0
            for offset in range(1, len(self._keys) + 1):
                candidate = (old + offset) % len(self._keys)
                if candidate not in self._exhausted:
                    self._idx = candidate
                    return
            raise QuotaExhausted("All keys exhausted after rotation.")

    def reset(self):
        with self._lock:
            self._exhausted.clear()
            self._idx = 0
            self._consecutive_429s = 0


@st.cache_resource
def _get_rotator():
    _, cfg, _ = _init_engine()
    return _KeyRotator(cfg["api_keys"]) if cfg else None


def embed_query(question: str, cfg: dict, rotator: _KeyRotator) -> list[float]:
    """Embed a single query string."""
    model = cfg["embedding_model"]
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/{model}:embedContent"
    )
    for attempt in range(5):
        _rate_limit(cfg["api_calls_per_minute"])
        key = rotator.get()
        resp = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            params={"key": key},
            json={"model": f"models/{model}", "content": {"parts": [{"text": question}]}},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()["embedding"]["values"]
        elif resp.status_code == 429:
            wait = 15 * (attempt + 1)
            time.sleep(wait)
            if attempt >= 2:
                rotator.rotate("sustained embed 429")
        else:
            raise RuntimeError(f"Embed error {resp.status_code}: {resp.text[:200]}")
    raise QuotaExhausted("Embed failed after all retries.")


def call_gemini(prompt: str, cfg: dict, rotator: _KeyRotator, max_tokens: int = 4096) -> str:
    """Call Gemini generation with model fallback and key rotation."""
    base_url = "https://generativelanguage.googleapis.com/v1beta/models"
    for model in cfg["gemini_models"]:
        url = f"{base_url}/{model}:generateContent"
        for attempt in range(4):
            _rate_limit(cfg["api_calls_per_minute"])
            key = rotator.get()
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": 0.3,
                },
            }
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                params={"key": key},
                json=payload,
                timeout=120,
            )
            if resp.status_code == 200:
                data = resp.json()
                try:
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError):
                    raise RuntimeError(f"Unexpected Gemini response: {str(data)[:300]}")
            elif resp.status_code == 429:
                wait = 15 * (attempt + 1)
                time.sleep(wait)
                if attempt >= 2:
                    rotator.rotate(f"sustained 429 on {model}")
            elif resp.status_code == 404:
                break  # model not found → try next
            else:
                raise RuntimeError(f"Gemini {resp.status_code}: {resp.text[:200]}")
    raise RuntimeError("All Gemini models failed.")


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL  (mirrors Cells 8 + 8.5 — query side only)
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_chunks(
    question: str,
    lens_names: list[str],
    collections: dict,
    cfg: dict,
    rotator: _KeyRotator,
    top_k: int = 8,
) -> list[dict]:
    """
    Retrieve top-k chunks from the specified lenses via cosine similarity.
    Returns merged, deduplicated list sorted by score descending.
    """
    q_vec = embed_query(question, cfg, rotator)

    results = []
    seen_ids = set()

    for lens in lens_names:
        coll = collections.get(lens)
        if not coll or coll.count() == 0:
            continue
        try:
            res = coll.query(
                query_embeddings=[q_vec],
                n_results=min(top_k, coll.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            continue

        for doc, meta, dist in zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        ):
            chunk_id = meta.get("chunk_id", "")
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            # Cosine distance → similarity score 0-10
            score = round((1 - dist) * 10, 2)
            results.append({
                "lens": lens,
                "source": meta.get("source", meta.get("filename", "unknown")),
                "page": meta.get("page_num", meta.get("page", "?")),
                "chunk_id": chunk_id,
                "text": doc,
                "score": score,
                "distance": dist,
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k * len(lens_names)]


def _format_context(chunks: list[dict], question: str, max_chars: int = 16_000) -> str:
    """Assemble retrieved chunks into a context block for Gemini."""
    parts = []
    total = 0
    for i, c in enumerate(chunks, 1):
        header = (
            f"[{i}] Source: {c['source']}  |  Page: {c['page']}"
            f"  |  Lens: {c['lens']}  |  Score: {c['score']}/10"
        )
        block = f"{header}\n{c['text']}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS PIPELINE  (mirrors Cell 12 ask_coeus_deep)
# ─────────────────────────────────────────────────────────────────────────────

def run_stage1(question, primary_lenses, collections, cfg, rotator, top_k, progress_cb=None):
    if progress_cb:
        progress_cb("Stage 1 — retrieving primary sources…")
    chunks = retrieve_chunks(question, primary_lenses, collections, cfg, rotator, top_k)
    if not chunks:
        return {"analysis": "[No primary sources found]", "sources": [], "chars": 0}
    context = _format_context(chunks, question)

    prompt = f"""You are a rigorous investment analyst drawing on primary source material.

QUESTION: {question}

SOURCE MATERIAL (retrieved from investor letters, books, and frameworks):
{context}

TASK:
Analyse the question using ONLY the source material above.
- Lead with the most important insight.
- Quote or closely paraphrase sources where useful.
- Note where sources disagree.
- Be direct. Avoid generic preamble.
- End with 2–3 load-bearing assumptions underlying your analysis.

ANALYSIS:"""
    analysis = call_gemini(prompt, cfg, rotator, max_tokens=4096)
    return {"analysis": analysis, "sources": chunks, "chars": len(context)}


def run_stage2(question, secondary_lenses, stage1_analysis, collections, cfg, rotator, top_k, progress_cb=None):
    if progress_cb:
        progress_cb("Stage 2 — lattice bridging across secondary sources…")
    chunks = retrieve_chunks(question, secondary_lenses, collections, cfg, rotator, top_k)
    if not chunks:
        return {"analysis": stage1_analysis, "sources": [], "chars": 0}
    context = _format_context(chunks, question)

    prompt = f"""You are extending an investment analysis with additional source material.

QUESTION: {question}

STAGE 1 ANALYSIS (from primary sources):
{stage1_analysis[:3000]}

ADDITIONAL SOURCE MATERIAL (secondary lenses):
{context}

TASK:
- Does the new material confirm, challenge, or add nuance to Stage 1?
- Integrate the best new evidence into a revised analysis.
- Do not repeat what Stage 1 already covered well.
- Preserve Stage 1's strong conclusions; update where new evidence warrants.

INTEGRATED ANALYSIS:"""
    analysis = call_gemini(prompt, cfg, rotator, max_tokens=4096)
    return {"analysis": analysis, "sources": chunks, "chars": len(context)}


def run_stage3(question, all_sources, stage2_analysis, cfg, rotator, progress_cb=None):
    if progress_cb:
        progress_cb("Stage 3 — synthesis and final grounding…")
    source_list = "\n".join(
        f"  - {s['source']} (p.{s['page']}, {s['lens']}, score {s['score']})"
        for s in all_sources[:20]
    )
    prompt = f"""You are producing a final investment synthesis.

QUESTION: {question}

WORKING ANALYSIS:
{stage2_analysis[:5000]}

SOURCES USED:
{source_list}

TASK:
Produce a final, polished analysis that:
1. Directly answers the question with a clear verdict.
2. Identifies the 2–3 most important factors driving your conclusion.
3. States what would need to be true to change your mind.
4. Flags the biggest uncertainty or blind spot.

Write in the direct, unsentimental style of a thoughtful investor.
No hedging for its own sake. No false balance.

FINAL ANALYSIS:"""
    analysis = call_gemini(prompt, cfg, rotator, max_tokens=6144)
    return {"analysis": analysis}


def run_fence_check(question, integrated_analysis, primary_lenses, collections, cfg, rotator, top_k, progress_cb=None):
    """Chesterton's Fence — surface dismissed features worth reconsidering."""
    if progress_cb:
        progress_cb("Chesterton's Fence — checking dismissed assumptions…")

    # Identify features the analysis dismissed
    identify_prompt = f"""Review this investment analysis and list features or assumptions it dismissed or rated negatively.

QUESTION: {question}

ANALYSIS:
{integrated_analysis[:3000]}

List up to 5 dismissed features as a simple numbered list. Be concise."""
    dismissed = call_gemini(identify_prompt, cfg, rotator, max_tokens=512)

    # Retrieve chunks relevant to the dismissed features
    fence_query = f"{question} — reasons why dismissed features might actually be important"
    chunks = retrieve_chunks(fence_query, primary_lenses, collections, cfg, rotator, top_k=6)
    context = _format_context(chunks, fence_query, max_chars=8000)

    fence_prompt = f"""You are applying Chesterton's Fence to an investment analysis.

ORIGINAL QUESTION: {question}

FEATURES THE ANALYSIS DISMISSED:
{dismissed}

RELEVANT SOURCE PASSAGES:
{context}

TASK:
For each dismissed feature, ask: "Why might this feature exist or matter?"
Only challenge dismissals where you have genuine source-backed grounds.
If a dismissal was correct, say so briefly and move on.
End with a net verdict: does the fence check materially change the analysis?

FENCE CHECK:"""
    result = call_gemini(fence_prompt, cfg, rotator, max_tokens=3072)
    return {"fence_analysis": result, "dismissed": dismissed, "sources": chunks}


def run_deep_analysis(
    question: str,
    primary_lenses: list[str],
    secondary_lenses: list[str],
    collections: dict,
    cfg: dict,
    rotator: _KeyRotator,
    top_k: int = 8,
    use_stage2: bool = True,
    use_stage3: bool = True,
    use_fence: bool = True,
    progress_cb=None,
) -> dict:
    """Full waterfall pipeline."""
    stages = []
    all_sources = []

    # Stage 1
    s1 = run_stage1(question, primary_lenses, collections, cfg, rotator, top_k, progress_cb)
    stages.append({"stage": 1, **s1})
    all_sources.extend(s1["sources"])
    current_analysis = s1["analysis"]

    # Stage 2
    if use_stage2 and secondary_lenses:
        s2 = run_stage2(question, secondary_lenses, current_analysis, collections, cfg, rotator, top_k, progress_cb)
        stages.append({"stage": 2, **s2})
        all_sources.extend(s2["sources"])
        if s2["sources"]:
            current_analysis = s2["analysis"]

    # Stage 3
    if use_stage3:
        s3 = run_stage3(question, all_sources, current_analysis, cfg, rotator, progress_cb)
        stages.append({"stage": 3, **s3})
        current_analysis = s3["analysis"]

    # Chesterton's Fence
    fence = None
    if use_fence:
        fence = run_fence_check(question, current_analysis, primary_lenses, collections, cfg, rotator, top_k, progress_cb)

    return {
        "question": question,
        "stages": stages,
        "final_analysis": current_analysis,
        "fence": fence,
        "all_sources": all_sources,
        "timestamp": datetime.now().isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    collections, cfg, error = _init_engine()
    rotator = _get_rotator()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🔍 Coeus")
        st.markdown("*Investment intelligence, retrieved.*")
        st.divider()

        if error:
            st.error(error)
            st.stop()

        # Collection health
        st.markdown("**Vector store**")
        active = []
        total_chunks = 0
        for name, coll in collections.items():
            try:
                n = coll.count()
                total_chunks += n
                if n > 0:
                    active.append((name, n))
            except Exception:
                pass

        if active:
            st.success(f"{len(active)} active lenses · {total_chunks:,} chunks")
        else:
            st.warning("No chunks indexed yet.")

        with st.expander("Lens details", expanded=False):
            for name, n in sorted(active, key=lambda x: -x[1]):
                st.markdown(f"`{name}` — {n:,} chunks")

        st.divider()

        # Lens selection
        st.markdown("**Primary lenses** *(Stage 1)*")
        default_primary = [n for n, _ in active if n in ["igy", "munger", "marks", "buffett", "fisher"]]
        primary_lenses = st.multiselect(
            "Primary",
            options=[n for n, _ in active],
            default=default_primary or [n for n, _ in active[:3]],
            format_func=lambda x: f"{x} — {LENSES_META.get(x, '')}",
            label_visibility="collapsed",
        )

        st.markdown("**Secondary lenses** *(Stage 2)*")
        default_secondary = [n for n, _ in active if n not in primary_lenses and n in ["external", "coeus_wiki"]]
        secondary_lenses = st.multiselect(
            "Secondary",
            options=[n for n, _ in active if n not in primary_lenses],
            default=default_secondary,
            format_func=lambda x: f"{x} — {LENSES_META.get(x, '')}",
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("**Options**")
        top_k = st.slider("Chunks per lens (top-k)", 4, 20, 8)
        use_stage2 = st.checkbox("Stage 2 (secondary lenses)", value=bool(secondary_lenses))
        use_stage3 = st.checkbox("Stage 3 (final synthesis)", value=True)
        use_fence  = st.checkbox("Chesterton's Fence", value=True)

        st.divider()
        if st.button("Reset API key rotation"):
            if rotator:
                rotator.reset()
                st.success("Keys reset.")

    # ── Main panel ────────────────────────────────────────────────────────────
    st.markdown("## Coeus — Deep Analysis")

    question = st.text_area(
        "Question",
        placeholder="e.g. Does Costco have a durable competitive advantage?",
        height=100,
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        run_btn = st.button("▶ Analyse", type="primary", use_container_width=True)

    if not run_btn:
        st.markdown("""
        **How to use**
        1. Select primary lenses in the sidebar (drive Stage 1 retrieval).
        2. Optionally add secondary lenses for Stage 2 bridging.
        3. Type your question and click **Analyse**.

        Results flow through up to three stages + Chesterton's Fence check.
        """)
        return

    if not question.strip():
        st.warning("Please enter a question.")
        return

    if not primary_lenses:
        st.warning("Select at least one primary lens.")
        return

    if not collections or not rotator:
        st.error("Engine not initialised.")
        return

    # ── Run ───────────────────────────────────────────────────────────────────
    status_placeholder = st.empty()
    progress_bar = st.progress(0)

    stage_steps = (
        1
        + (1 if use_stage2 and secondary_lenses else 0)
        + (1 if use_stage3 else 0)
        + (1 if use_fence else 0)
    )
    step_counter = {"n": 0}

    def progress_cb(msg: str):
        step_counter["n"] += 1
        pct = min(step_counter["n"] / stage_steps, 0.95)
        progress_bar.progress(pct, text=msg)
        status_placeholder.info(f"⚙ {msg}")

    try:
        result = run_deep_analysis(
            question=question.strip(),
            primary_lenses=primary_lenses,
            secondary_lenses=secondary_lenses,
            collections=collections,
            cfg=cfg,
            rotator=rotator,
            top_k=top_k,
            use_stage2=use_stage2 and bool(secondary_lenses),
            use_stage3=use_stage3,
            use_fence=use_fence,
            progress_cb=progress_cb,
        )
    except QuotaExhausted as e:
        progress_bar.empty()
        status_placeholder.empty()
        st.error(f"API quota exhausted: {e}")
        return
    except Exception as e:
        progress_bar.empty()
        status_placeholder.empty()
        st.error(f"Error: {e}")
        return

    progress_bar.progress(1.0, text="Complete")
    status_placeholder.empty()
    progress_bar.empty()

    # ── Display results ───────────────────────────────────────────────────────

    # Final analysis tab is first
    tab_names = ["📋 Final Analysis"] + [f"Stage {s['stage']}" for s in result["stages"]]
    if result.get("fence"):
        tab_names.append("🪟 Fence Check")
    tab_names.append("📚 Sources")

    tabs = st.tabs(tab_names)

    # Final Analysis
    with tabs[0]:
        st.markdown("### Final Analysis")
        st.markdown(result["final_analysis"])
        st.download_button(
            "Download analysis",
            data=result["final_analysis"],
            file_name=f"coeus_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )

    # Per-stage tabs
    for i, stage_data in enumerate(result["stages"], 1):
        with tabs[i]:
            st.markdown(f"### Stage {stage_data['stage']} Analysis")
            st.caption(f"{stage_data['chars']:,} chars context · {len(stage_data['sources'])} chunks retrieved")
            st.markdown(stage_data["analysis"])

    # Fence tab
    fence_tab_idx = 1 + len(result["stages"])
    if result.get("fence"):
        with tabs[fence_tab_idx]:
            st.markdown("### Chesterton's Fence Check")
            st.markdown("*Features the analysis dismissed — are they dismissed for good reason?*")
            with st.expander("Dismissed features identified"):
                st.text(result["fence"]["dismissed"])
            st.markdown(result["fence"]["fence_analysis"])
            fence_tab_idx += 1

    # Sources tab
    with tabs[fence_tab_idx]:
        st.markdown(f"### All Retrieved Sources ({len(result['all_sources'])} chunks)")
        # Deduplicate by (source, page, lens)
        seen = set()
        unique = []
        for s in result["all_sources"]:
            key = (s["source"], s["page"], s["lens"])
            if key not in seen:
                seen.add(key)
                unique.append(s)
        unique.sort(key=lambda x: x["score"], reverse=True)

        for i, s in enumerate(unique[:30], 1):
            with st.expander(f"[{i}] {s['source']}  p.{s['page']}  |  {s['lens']}  |  score {s['score']}/10"):
                st.caption(f"Chunk ID: {s.get('chunk_id', '—')[:20]}")
                st.text(s["text"][:800] + ("…" if len(s["text"]) > 800 else ""))


if __name__ == "__main__":
    main()
