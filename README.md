# Multimodal RAG Chatbot with Dynamic Programming Optimization

An advanced, production-ready Retrieval-Augmented Generation (RAG) system built with **Google Gemini 2.5 Flash**, **FAISS Vector Database**, and **Gemini Multimodal Embedding 2 Preview**.

## 🚀 Key Features

1. **Multimodal Native Processing**: 
   - No need to install heavy local vision models (like PyTorch or HuggingFace CLIP).
   - Seamlessly converts both `.txt`/`.md` documents and images (`.png`, `.jpg`, `.jpeg`) into 1408-dimensional mathematical vectors directly via the Gemini Native Multimodal Embedding API.
2. **0/1 Knapsack Context Optimizer (Dynamic Programming)**:
   - When fetching background data, we cannot pass an unlimited amount of documents to the model.
   - This app utilizes a strict 2D Matrix Dynamic Programming algorithm (the `0/1 Knapsack`) to calculate the *mathematically optimal subset* of retrieved chunks to pass to Gemini. It maximizes the total relevance score of the context without once exceeding the strict prompt character limit.
3. **Asynchronous Parallel Indexing**:
   - Implements `concurrent.futures.ThreadPoolExecutor` to index dozens of images and text files simultaneously.
4. **FAISS Vector DB**:
   - A blistering fast, C++ optimized local memory bank created by Meta, using `faiss-cpu`.
5. **Execution Latency Profiling**:
   - Features real-time `time.time()` latency timers for API requests, DB fetches, and DP matrix operations so you always know where your bottlenecks are.

## 📂 Project Structure

```
.
├── data/                  # Drop your images, text, and markdown files here!
├── faiss_data/            # Automatically stores the vector indexes and metadata
├── rag_chatbot.py         # The RAG application engine
├── readme.md              # This documentation
└── .env                   # Put your gemini_api_key in here
```

## 🛠️ Installation

1. Create a Python Virtual Environment (recommended).
2. Install the required dependencies:
```bash
pip install google-genai python-dotenv faiss-cpu numpy pillow
```
3. Set up your `.env` file at the root of the project:


## 🧠 How to Use

Simply run the main Python script:

```bash
python rag_chatbot.py
```

### The Interface
You will be prompted with a menu:
- **Option 1**: Instantly scans your `data/` folder, embeds any new text or image files entirely in parallel, and saves them to `faiss_data/`.
- **Option 2**: Starts the conversational RAG phase. Type any question. The algorithm will retrieve FAISS memories, perfectly pack the context window using the DP Matrix, and inject both text and retrieved images to the `gemini-2.5-flash` generation model!
