# 🧠 Multi-Model RAG Chatbot (Stateless Cloud Architecture)

**🔴 Live Demo (Deployed on Streamlit Cloud):** [https://multimodal-rag-chatbot-with-dp-optimization-2341qf.streamlit.app/](https://multimodal-rag-chatbot-with-dp-optimization-2341qf.streamlit.app/)

An advanced, production-ready Retrieval-Augmented Generation (RAG) system built entirely for **serverless cloud deployment**. Features dual embedding support (Gemini & Cohere) and dynamic programming context optimization, driven entirely by a Supabase pgvector backend.

> Built with **Gemini 2.5 Flash** (Generation) • **Gemini Embedding 2 / Cohere v4.0** (Embeddings) • **Supabase pgvector** (Vector DB & History) • **Streamlit** (Web UI)

---

## 🚀 Key Features

### 1. ☁️ 100% Stateless Cloud Architecture
Ready to deploy on ephemeral servers like **Streamlit Community Cloud** or Render without losing data.
- Both text and **images** are fully encoded (Base64) and stored within the Supabase Postgres Database.
- No local hard drive dependency—if your server resets, all chat history and document intelligence survive.

### 2. 🔄 Dual Embedding Model Support
Switch between **Gemini** and **Cohere** embedding models at runtime natively from the web UI.

### 3. 🎒 0/1 Knapsack Context Optimizer (Dynamic Programming)
Uses a strict 2D Matrix DP algorithm (0/1 Knapsack) to calculate the *mathematically optimal subset* of retrieved document chunks. Maximizes total contextual relevance without exceeding Gemini's strict prompt character limits.

### 4. 🗄️ Supabase pgvector Database
Replaced local FAISS indexing with Meta's state-of-the-art vector similarity search baked directly into PostgreSQL.

### 5. 📎 Multi-Format Document Upload
Upload and index documents directly through the web interface:
- **Text**: `.txt`, `.md`
- **Documents**: `.pdf`, `.docx`
- **Presentations**: `.pptx`
- **Images**: `.png`, `.jpg`, `.jpeg`, `.webp` (Processed natively via Gemini Multimodal API)

---

## 🛠️ Installation & Setup

### 1. Clone & Set Up Virtual Environment
```bash
cd LLM
py -m venv .venv
.venv\Scripts\activate     # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Create a `.env` File (Locally)
Create a `.env` file in the project root. **Never commit this file to GitHub!**
```ini
gemini_api_key = YOUR_GEMINI_API_KEY
cohere_api_key = YOUR_COHERE_API_KEY
project_url = YOUR_SUPABASE_PROJECT_URL
service_key = YOUR_SUPABASE_SERVICE_ROLE_KEY
```

### 4. Initialize Your Supabase Database
Go to your [Supabase Dashboard](https://supabase.com/dashboard) **SQL Editor** and run the following command to create your schema and vector search RPC functions:

```sql
create extension if not exists vector;

create table documents (
  id bigserial primary key,
  content text,
  metadata jsonb,
  embedding_gemini vector(3072),
  embedding_cohere vector(1536)
);

create table chat_sessions (
    session_id text primary key,
    created_at timestamp with time zone default timezone('utc'::text, now()),
    embedding_model text,
    total_input_tokens integer default 0,
    total_output_tokens integer default 0,
    total_queries integer default 0
);

create table chat_messages (
    id bigserial primary key,
    session_id text references chat_sessions(session_id) on delete cascade,
    role text,
    content text,
    meta jsonb,
    created_at timestamp with time zone default timezone('utc'::text, now())
);

-- Search function for Gemini
create or replace function match_documents_gemini (
  query_embedding vector(3072),
  match_threshold float,
  match_count int
) returns table ( id bigint, content text, metadata jsonb, similarity float )
language sql stable as $$
  select id, content, metadata, 1 - (embedding_gemini <=> query_embedding) as similarity
  from documents where embedding_gemini is not null
  order by embedding_gemini <=> query_embedding limit match_count;
$$;

-- Search function for Cohere
create or replace function match_documents_cohere (
  query_embedding vector(1536),
  match_threshold float,
  match_count int
) returns table ( id bigint, content text, metadata jsonb, similarity float )
language sql stable as $$
  select id, content, metadata, 1 - (embedding_cohere <=> query_embedding) as similarity
  from documents where embedding_cohere is not null
  order by embedding_cohere <=> query_embedding limit match_count;
$$;
```

---

## 🧠 Web Deployment (Streamlit Cloud)

Because the architecture is entirely stateless, deployment is completely free and requires zero maintenance.

1. Ensure `.env` is listed in your `.gitignore` file.
2. Push this repository to GitHub.
3. Log into [Streamlit Community Cloud](https://share.streamlit.io).
4. Select "Create app" -> Point to your GitHub repository -> Select `app.py` as the Main file path.
5. In Streamlit Cloud, click **Advanced settings** -> **Secrets** and paste the variables from your local `.env` file.
6. Click Deploy!