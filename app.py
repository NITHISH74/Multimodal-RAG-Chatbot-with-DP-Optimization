"""
╔══════════════════════════════════════════════════════════════════════╗
║  Multi-Model RAG Chatbot — Streamlit Web Application               ║
║  Storage: Supabase Cloud Database (pgvector)                         ║
║  Embeddings: Gemini 2 Preview & Cohere v4.0                        ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import time
import base64
import numpy as np
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from google import genai
import cohere
from datetime import datetime
from supabase import create_client, Client
from st_copy_to_clipboard import st_copy_to_clipboard

# ── Document parsers ──
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from pptx import Presentation

# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION & CONSTANTS
# ══════════════════════════════════════════════════════════════════════

load_dotenv()

DATA_DIR = "data"

GEMINI_EMBED_MODEL = "gemini-embedding-2-preview"
COHERE_EMBED_MODEL = "embed-v4.0"
GENERATION_MODEL = "gemini-2.5-flash"
MAX_CONTEXT_CHARS = 5000

os.makedirs(DATA_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
#  PAGE CONFIGURATION & CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Multi-Model RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(48, 43, 99, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .main-header h1 { color: #ffffff; font-weight: 800; font-size: 1.8rem; margin: 0; }
    .main-header p { color: rgba(255,255,255,0.65); font-size: 0.9rem; margin: 0.3rem 0 0 0; }
    
    .metric-row { display: flex; gap: 12px; margin-bottom: 1rem; }
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px; padding: 1rem; flex: 1; text-align: center;
        transition: transform 0.2s ease;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-value {
        font-size: 1.6rem; font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;
    }
    .metric-label { font-size: 0.75rem; color: rgba(255,255,255,0.5); text-transform: uppercase; margin: 0.3rem 0 0 0;}
    
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%); }
    .sidebar-section {
        background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px; padding: 1rem; margin: 0.8rem 0;
    }
    .sidebar-section h3 { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; color: rgba(255,255,255,0.4); margin: 0 0 0.6rem 0; }
    
    .status-badge { display: inline-block; padding: 0.2rem 0.7rem; border-radius: 20px; font-size: 0.72rem; font-weight: 600; }
    .status-gemini { background: rgba(66, 133, 244, 0.15); color: #4285f4; border: 1px solid rgba(66, 133, 244, 0.3); }
    .status-cohere { background: rgba(168, 85, 247, 0.15); color: #a855f7; border: 1px solid rgba(168, 85, 247, 0.3); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  STATE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "messages": [],
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_queries": 0,
        "embedding_model": "Gemini",
        "current_session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()


# ══════════════════════════════════════════════════════════════════════
#  CLIENT INITIALIZATION
# ══════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=os.getenv("gemini_api_key"))

@st.cache_resource
def get_cohere_client():
    return cohere.ClientV2(api_key=os.getenv("cohere_api_key"))

@st.cache_resource
def get_supabase_client():
    url = os.getenv("project_url")
    key = os.getenv("service_key")
    if url and key:
        return create_client(url, key)
    return None


# ══════════════════════════════════════════════════════════════════════
#  EMBEDDING LOGIC
# ══════════════════════════════════════════════════════════════════════

def embed_content(content, content_type, model_name, filename=None):
    if model_name == "Gemini":
        client = get_gemini_client()
        res = client.models.embed_content(model=GEMINI_EMBED_MODEL, contents=content)
        return res.embeddings[0].values
    else:  # Cohere
        client = get_cohere_client()
        if content_type == "text":
            res = client.embed(model=COHERE_EMBED_MODEL, texts=[content], input_type="search_document", embedding_types=["float"])
            return list(res.embeddings.float_[0])
        elif content_type == "image":
            img_path = os.path.join(DATA_DIR, filename) if filename else None
            if img_path and os.path.exists(img_path):
                ext = os.path.splitext(img_path)[1].lower().replace(".", "")
                mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp", "gif": "gif"}.get(ext, "jpeg")
                with open(img_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                res = client.embed(model=COHERE_EMBED_MODEL, images=[f"data:image/{mime};base64,{encoded}"], input_type="image", embedding_types=["float"])
                return list(res.embeddings.float_[0])
    return None

def embed_query(query_text, model_name):
    if model_name == "Gemini":
        return get_gemini_client().models.embed_content(model=GEMINI_EMBED_MODEL, contents=query_text).embeddings[0].values
    else:
        return list(get_cohere_client().embed(model=COHERE_EMBED_MODEL, texts=[query_text], input_type="search_query", embedding_types=["float"]).embeddings.float_[0])


# ══════════════════════════════════════════════════════════════════════
#  FILE PARSING
# ══════════════════════════════════════════════════════════════════════

def parse_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()

    if name.endswith(".pdf"):
        import io
        text = "".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(file_bytes)).pages)
        return "text", text.strip(), uploaded_file.name
    elif name.endswith(".docx"):
        import io
        return "text", "\n".join(p.text for p in DocxDocument(io.BytesIO(file_bytes)).paragraphs), uploaded_file.name
    elif name.endswith(".txt") or name.endswith(".md"):
        return "text", file_bytes.decode("utf-8", errors="ignore"), uploaded_file.name
    elif name.endswith((".png", ".jpg", ".jpeg", ".webp")):
        import io
        img = Image.open(io.BytesIO(file_bytes))
        save_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(save_path, "wb") as f: f.write(file_bytes)
        return "image", img, uploaded_file.name
    return None, None, uploaded_file.name


# ══════════════════════════════════════════════════════════════════════
#  INDEXING MANAGER (SUPABASE)
# ══════════════════════════════════════════════════════════════════════

def is_file_indexed(filename, model_name, supabase):
    # Check if the file exists and has an embedding for the CURRENT model
    res = supabase.table("documents").select(f"metadata, embedding_{model_name.lower()}").eq("metadata->>file", filename).execute()
    return any(row.get(f"embedding_{model_name.lower()}") is not None for row in res.data)

def index_uploaded_files(uploaded_files, model_name, progress_bar, status_text):
    supabase = get_supabase_client()
    total = len(uploaded_files)
    count = 0

    for i, file in enumerate(uploaded_files):
        fname = file.name
        progress_bar.progress((i + 1) / total, text=f"Processing {fname}...")
        
        if is_file_indexed(fname, model_name, supabase):
            status_text.markdown(f"⏩ `{fname}` already indexed for {model_name}, skipping.")
            continue

        c_type, content, _ = parse_uploaded_file(file)
        if not c_type or (c_type == "text" and not content.strip()): continue
        
        if c_type == "text":
            with open(os.path.join(DATA_DIR, fname), "w", encoding="utf-8") as f:
                f.write(content)
        
        vector = embed_content(content, c_type, model_name, filename=fname)
        if not vector: continue

        meta = {
            "file": fname, "type": c_type,
            "content": content if c_type == "text" else "Visual content",
            "weight": len(content) if c_type == "text" else 500
        }

        # For images, store the base64 string in DB to bypass Streamlit Cloud ephemeral local storage wipes
        if c_type == "image":
            import io
            buf = io.BytesIO()
            content.save(buf, format=content.format or "JPEG")
            meta["image_base64"] = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Check if the document already exists from the other model
        existing = supabase.table("documents").select("id").eq("metadata->>file", fname).execute()
        
        if existing.data and len(existing.data) > 0:
            row_id = existing.data[0]['id']
            # Update the existing row with the new model's embedding
            if model_name == "Gemini": supabase.table("documents").update({"embedding_gemini": vector, "metadata": meta}).eq("id", row_id).execute()
            else: supabase.table("documents").update({"embedding_cohere": vector, "metadata": meta}).eq("id", row_id).execute()
        else:
            # Insert brand new row
            row = {"content": meta["content"], "metadata": meta}
            if model_name == "Gemini": row["embedding_gemini"] = vector
            else: row["embedding_cohere"] = vector
            supabase.table("documents").insert(row).execute()

        count += 1
        status_text.markdown(f"✅ `{fname}` embedded!")
    
    return count


# ══════════════════════════════════════════════════════════════════════
#  DP 0/1 KNAPSACK & PIPELINE
# ══════════════════════════════════════════════════════════════════════

def optimize_context_knapsack(candidates, max_chars):
    n = len(candidates)
    if n == 0: return []
    SCALE = 10
    W = max_chars // SCALE
    weights = [max(1, item["weight"] // SCALE) for item in candidates]
    values = [item["value"] for item in candidates]

    dp = [[0.0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(1, W + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w - weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]

    chosen = []
    w = W
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            chosen.append(candidates[i-1])
            w -= weights[i-1]
    return chosen


def run_rag_pipeline(user_query, model_name):
    pipeline_meta = {"retrieval_time": 0, "generation_time": 0, "context_chunks": 0, "input_tokens": 0, "output_tokens": 0}
    candidates = []
    t0 = time.time()

    try:
        query_vec = embed_query(user_query, model_name)
        
        # Supabase Retrieval via RPC
        supabase = get_supabase_client()
        rpc_name = "match_documents_gemini" if model_name == "Gemini" else "match_documents_cohere"
        res = supabase.rpc(rpc_name, {"query_embedding": query_vec, "match_threshold": 0.0, "match_count": 10}).execute()
        for row in res.data:
            m = row["metadata"]
            dist = row.get("similarity", 0) # RPC returns distance via <->
            candidates.append({**m, "value": dist, "dist": dist})
                
    except Exception as e:
        st.warning(f"Retrieval error: {e}")

    pipeline_meta["retrieval_time"] = time.time() - t0
    chosen = optimize_context_knapsack(candidates, MAX_CONTEXT_CHARS)
    pipeline_meta["context_chunks"] = len(chosen)

    # Generate
    t1 = time.time()
    contexts, image_parts = [], []
    for item in chosen:
        if item["type"] == "text":
            contexts.append(f"From {item['file']} (Score {item['value']:.2f}):\n{item['content']}")
        elif item["type"] == "image":
            if "image_base64" in item:
                import io
                image_parts.append(Image.open(io.BytesIO(base64.b64decode(item["image_base64"]))))
            contexts.append(f"From {item['file']}: (Image attached)\n")

    prompt = user_query
    if contexts:
        prompt = "Optimized DP Context:\n---\n" + "\n---\n".join(contexts) + f"\n---\nAnswer the query: {user_query}"
    
    try:
        client = get_gemini_client()
        res = client.models.generate_content(model=GENERATION_MODEL, contents=[prompt] + image_parts)
        response_text = res.text
        if hasattr(res, 'usage_metadata') and res.usage_metadata:
            pipeline_meta["input_tokens"] = res.usage_metadata.prompt_token_count or 0
            pipeline_meta["output_tokens"] = res.usage_metadata.candidates_token_count or 0
    except Exception as e:
        response_text = f"Generation error: {e}"

    pipeline_meta["generation_time"] = time.time() - t1
    return response_text, pipeline_meta


# ══════════════════════════════════════════════════════════════════════
#  HISTORY MANAGER
# ══════════════════════════════════════════════════════════════════════

def save_chat_history():
    sid = st.session_state.current_session_id
    data = {
        "session_id": sid,
        "embedding_model": st.session_state.embedding_model,
        "total_input_tokens": st.session_state.total_input_tokens,
        "total_output_tokens": st.session_state.total_output_tokens,
        "total_queries": st.session_state.total_queries,
    }

    # Supabase History Logging
    supabase = get_supabase_client()
    supabase.table("chat_sessions").upsert(data).execute()
    # Ensure we only insert the newest messages (optimised approach: delete old, insert new)
    supabase.table("chat_messages").delete().eq("session_id", sid).execute()
    msgs_to_insert = [{"session_id": sid, "role": m["role"], "content": m["content"], "meta": m.get("meta", {})} for m in st.session_state.messages]
    if msgs_to_insert:
        supabase.table("chat_messages").insert(msgs_to_insert).execute()


def load_all_history():
    supabase = get_supabase_client()
    sessions = supabase.table("chat_sessions").select("*").order("created_at", desc=True).limit(20).execute().data
    for s in sessions:
        msgs = supabase.table("chat_messages").select("*").eq("session_id", s["session_id"]).order("id").execute().data
        s["messages"] = msgs
        s["timestamp"] = s["created_at"]
    return sessions

# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <span style="font-size: 2.5rem;">🧠</span>
        <h2 style="margin:0; font-weight:700; background:linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">Multi-Model RAG</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='sidebar-section'><h3>⚙️ Settings</h3></div>", unsafe_allow_html=True)
    
    model_choice = st.radio("Embedding Model", ["Gemini", "Cohere"], 
                            index=0 if st.session_state.embedding_model == "Gemini" else 1)
    if model_choice != st.session_state.embedding_model:
        st.session_state.embedding_model = model_choice
        st.rerun()

    st.markdown(f"""
    <span class="status-badge status-{'gemini' if model_choice=='Gemini' else 'cohere'}">{GEMINI_EMBED_MODEL if model_choice=='Gemini' else COHERE_EMBED_MODEL}</span>
    <span class="status-badge status-supa">Supabase DB</span>
    <p style="margin-top: 10px; font-size: 0.8rem; color: #aaaaaa;">
      <i>Note: You must explicitly click 'Index Uploads' whenever you switch embedding models so documents are embedded using the newly selected model.</i>
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section'><h3>📁 Upload</h3></div>", unsafe_allow_html=True)
    files = st.file_uploader(f"Index to Supabase using {st.session_state.embedding_model}", type=["txt", "md", "pdf", "docx", "png", "jpg"], accept_multiple_files=True)
    if st.button("📥 Index Uploads", use_container_width=True) and files:
        prog = st.progress(0, "Starting...")
        status = st.empty()
        count = index_uploaded_files(files, st.session_state.embedding_model, prog, status)
        if count > 0: st.success(f"Indexed {count} files using {st.session_state.embedding_model}!")
        time.sleep(2); st.rerun()

    st.markdown("<div class='sidebar-section'><h3>🕐 History</h3></div>", unsafe_allow_html=True)
    history = load_all_history()
    for h in history:
        sid = h["session_id"]
        ts = h.get("timestamp", "")[:16].replace("T", " ")
        if st.button(f"{'🟢 ' if sid==st.session_state.current_session_id else ''}{ts} ({h.get('embedding_model')})", key=f"hs_{sid}", use_container_width=True):
            if sid != st.session_state.current_session_id:
                st.session_state.messages = h.get("messages", [])
                st.session_state.total_input_tokens = h.get("total_input_tokens", 0)
                st.session_state.total_output_tokens = h.get("total_output_tokens", 0)
                st.session_state.total_queries = h.get("total_queries", 0)
                st.session_state.current_session_id = sid
                st.rerun()
                
    if st.button("🆕 New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.rerun()

# ══════════════════════════════════════════════════════════════════════
#  MAIN UI
# ══════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>🧠 Multi-Model RAG Chatbot</h1>
    <p>Gemini 2.5 Generation • Supabase Cloud Database • Dual Embeddings (Gemini/Cohere)</p>
</div>
""", unsafe_allow_html=True)

# Metrics
st.markdown(f"""
<div class="metric-row">
    <div class="metric-card"><p class="metric-value">{st.session_state.total_queries}</p><p class="metric-label">Queries Made</p></div>
    <div class="metric-card"><p class="metric-value">{st.session_state.total_input_tokens + st.session_state.total_output_tokens:,}</p><p class="metric-label">Total Tokens</p></div>
    <div class="metric-card"><p class="metric-value">Supabase</p><p class="metric-label">Active Database</p></div>
</div>
""", unsafe_allow_html=True)

for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"): st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(msg["content"])
            if "meta" in msg and msg["meta"]:
                m = msg["meta"]
                cols = st.columns([1, 1, 1, 2])
                cols[0].caption(f"⏱️ Retr: {m.get('retrieval_time', 0):.2f}s")
                cols[1].caption(f"⚡ Gen: {m.get('generation_time', 0):.2f}s")
                cols[2].caption(f"📎 Chunks: {m.get('context_chunks', 0)}")
                with cols[3]:
                    st_copy_to_clipboard(msg["content"])

if user_input := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"): st.markdown(user_input)

    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner(f"Searching Supabase with {st.session_state.embedding_model}..."):
            response_text, meta = run_rag_pipeline(user_input, st.session_state.embedding_model)
        st.markdown(response_text)
        
        cols = st.columns([1, 1, 1, 2])
        cols[0].caption(f"⏱️ Retr: {meta['retrieval_time']:.2f}s")
        cols[1].caption(f"⚡ Gen: {meta['generation_time']:.2f}s")
        cols[2].caption(f"📎 Chunks: {meta['context_chunks']}")
        with cols[3]:
            st_copy_to_clipboard(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text, "meta": meta})
    st.session_state.total_input_tokens += meta["input_tokens"]
    st.session_state.total_output_tokens += meta["output_tokens"]
    st.session_state.total_queries += 1
    
    save_chat_history()
    time.sleep(0.5) # Wait for clipboard button to flush
    st.rerun()
