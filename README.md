# 🧠 Multi-Model RAG Chatbot with DP Optimization

An advanced, production-ready Retrieval-Augmented Generation (RAG) system featuring **dual embedding model support** (Gemini & Cohere), a **Streamlit web interface**, and **Dynamic Programming context optimization**.

> Built with **Gemini 2.5 Flash** (Generation) • **Gemini Embedding 2 Preview** & **Cohere Embed v4.0** (Embeddings) • **FAISS** (Vector DB) • **Streamlit** (Web UI)

---

## 🚀 Key Features

### 1. 🔄 Dual Embedding Model Support
Switch between **Gemini** and **Cohere** embedding models at runtime via the web UI. Each model maintains its own FAISS index, so you can compare retrieval quality side-by-side.

| Feature | Gemini Embedding 2 | Cohere Embed v4.0 |
|---------|--------------------|--------------------|
| Dimensions | 3072 | 1536 |
| Multimodal | ✅ Native (PIL) | ✅ Base64 Data URL |
| Multilingual | ✅ 100+ langs | ✅ 100+ langs |
| Matryoshka Dims | ❌ | ✅ (64, 128, 256…) |
| Max Context | ~8K tokens | 128K tokens |
| Quantization | ❌ | ✅ int8 / binary |

### 2. 🌐 Streamlit Web Interface
A polished, dark-themed web dashboard with:
- **Chat interface** with real-time streaming responses
- **File upload** supporting PDF, DOCX, PPTX, TXT, MD, and images
- **Chat history** persistence across sessions
- **Token usage** tracking (input/output/total)
- **Pipeline metrics** per query (retrieval time, generation time, context chunks)

### 3. 📎 Multi-Format Document Upload
Upload and index documents directly through the web interface:
- **Text**: `.txt`, `.md`
- **Documents**: `.pdf`, `.docx`
- **Presentations**: `.pptx`
- **Images**: `.png`, `.jpg`, `.jpeg`, `.webp`

### 4. 🎒 0/1 Knapsack Context Optimizer (Dynamic Programming)
Uses a strict 2D Matrix DP algorithm (0/1 Knapsack) to calculate the *mathematically optimal subset* of retrieved chunks. Maximizes total relevance without exceeding the prompt character limit.

### 5. ⚡ Asynchronous Parallel Indexing
Implements `concurrent.futures.ThreadPoolExecutor` to embed multiple files simultaneously for fast bulk indexing.

### 6. 🗄️ FAISS Vector Database
Meta's C++-optimized vector similarity search library (`faiss-cpu`) for blazing fast nearest-neighbor retrieval.

### 7. 📊 Execution Latency Profiling
Real-time performance metrics for API requests, DB searches, DP operations, and generation — visible on every query response.

---

## 📈 Embedding Model Comparison: Gemini vs Cohere

We ran a comprehensive benchmark (`embedding_comparison.py`) comparing both models across 4 dimensions:

### Test Results Summary

| Test | Gemini | Cohere | Winner |
|------|--------|--------|--------|
| **Text Latency** | 644 ms avg | 411 ms avg | 🟣 Cohere (36% faster) |
| **Retrieval Precision@1** | 100% (5/5) | 100% (5/5) | 🟡 Tie |
| **Semantic Discrimination** | 0.204 gap | 0.335 gap | 🟣 Cohere (64% better) |
| **Multimodal Image Embedding** | ✅ 3072d | ✅ 1536d | 🟡 Tie |

### Detailed Results

#### Text Embedding Latency
| Query | Gemini | Cohere |
|-------|--------|--------|
| Hyper-Vortex Cooling Protocol | 728 ms | 498 ms |
| Golden Gate Bridge color | 592 ms | 395 ms |
| Day 4 of Project Solaris | 599 ms | 388 ms |
| Student skills and interests | 679 ms | 396 ms |
| Thermal management systems | 621 ms | 380 ms |

#### Retrieval Quality (Cosine Similarity Scores)
Both models correctly retrieved the matching document for all 5 test queries:
| Query | Gemini Score | Cohere Score |
|-------|-------------|-------------|
| HVCP Protocol | 0.796 | 0.599 |
| Golden Gate Bridge | 0.823 | 0.608 |
| Project Solaris | 0.715 | 0.556 |
| Student info | 0.718 | 0.408 |
| Thermal management | 0.663 | 0.445 |

#### Semantic Discrimination (Similar vs Dissimilar separation)
| Metric | Gemini | Cohere |
|--------|--------|--------|
| Avg Similar Pair Score | 0.769 | 0.540 |
| Avg Dissimilar Pair Score | 0.565 | 0.205 |
| **Discrimination Gap** | 0.204 | **0.335** |

> A larger gap means the model better separates relevant from irrelevant content — critical for ranking quality in RAG.

#### Cross-Modal Search (Text → Image)
| Model | Image | Cosine Similarity |
|-------|-------|-------------------|
| Gemini | framework.jpg | 0.4214 |
| Cohere | framework.jpg | 0.3549 |

### 🏆 Overall Winner: **Cohere Embed v4.0** (Score: 3.0 / 4.0)

**Cohere advantages**: Faster latency, better discrimination, Matryoshka dimensions, 128K context, quantization support.

**Gemini advantages**: Higher absolute cosine scores, native PIL image support, unified API stack with generation model.

**Bottom line**: Both achieve 100% retrieval accuracy. Cohere offers better engineering tradeoffs for production; Gemini is simpler for an all-Google stack.

---

## 📂 Project Structure

```
.
├── app.py                     # 🌐 Streamlit web application (main entry point)
├── rag_chatbot.py             # 🤖 CLI-based RAG chatbot (original)
├── embedding_comparison.py    # 📊 Benchmark: Gemini vs Cohere comparison
├── requirements.txt           # 📦 Python dependencies
├── .env                       # 🔑 API keys (gemini_api_key, cohere_api_key)
├── readme.md                  # 📖 This documentation
├── data/                      # 📁 Drop your documents and images here
│   ├── demo.txt
│   └── framework.jpg
└── faiss_data/                # 🗄️ Auto-generated vector indexes & metadata
    ├── gemini_index.bin       # FAISS index (Gemini embeddings)
    ├── gemini_metadata.json
    ├── cohere_index.bin       # FAISS index (Cohere embeddings)
    ├── cohere_metadata.json
    ├── chat_history.json      # Persistent chat history
    └── benchmark_results.json # Comparison benchmark results
```

---

## 🛠️ Installation

### 1. Clone & Create Virtual Environment
```bash
cd LLM
py -m venv .venv
.venv\Scripts\activate     
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up API Keys
Create a `.env` file in the project root:
```env
gemini_api_key = YOUR_GEMINI_API_KEY
cohere_api_key = YOUR_COHERE_API_KEY
```

Get your keys from:
- **Gemini**: [Google AI Studio](https://aistudio.google.com/apikey)
- **Cohere**: [Cohere Dashboard](https://dashboard.cohere.com/api-keys)

---

## 🧠 How to Use

### Web Interface (Recommended)
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`. Features:
- **Sidebar**: Select embedding model (Gemini/Cohere), upload files, view token stats & chat history
- **Main Area**: Chat with your documents, see retrieval metrics per query
- **File Upload**: Drag & drop PDF, DOCX, PPTX, TXT, images → auto-indexed

### CLI Mode (Original)
```bash
python rag_chatbot.py
```
Menu-driven interface:
- **Option 1**: Scan & index `data/` folder (multi-threaded)
- **Option 2**: Start conversational RAG chatbot
- **Option 3**: Exit

### Run Embedding Benchmark
```bash
python embedding_comparison.py
```
Runs 4 automated tests comparing Gemini vs Cohere across latency, retrieval, semantics, and multimodal.

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Generation** | Gemini 2.5 Flash |
| **Embeddings** | Gemini Embedding 2 Preview / Cohere Embed v4.0 |
| **Vector DB** | FAISS (faiss-cpu) |
| **Web UI** | Streamlit |
| **Context Optimization** | Dynamic Programming (0/1 Knapsack) |
| **Parallel Processing** | concurrent.futures.ThreadPoolExecutor |
| **Document Parsing** | PyPDF2, python-docx, python-pptx, Pillow |

---

## 📜 License

This project is for educational and personal use.
