import os
import json
import time
import numpy as np
import faiss
from PIL import Image
from dotenv import load_dotenv
from google import genai
import concurrent.futures

# Constants
FAISS_DIR = "faiss_data"
INDEX_FILE = os.path.join(FAISS_DIR, "index.bin")
METADATA_FILE = os.path.join(FAISS_DIR, "metadata.json")
DATA_DIR = "data"
EMBEDDING_MODEL = "gemini-embedding-2-preview"
GENERATION_MODEL = "gemini-2.5-flash"
MAX_CONTEXT_CHARS = 5000  # DP knapsack capacity constraint

def get_client():
    load_dotenv()
    api_key = os.getenv("gemini_api_key")
    if not api_key:
        raise ValueError("Missing 'gemini_api_key' in .env")
    return genai.Client(api_key=api_key)

def load_or_create_index(embedding_dim=1408):
    os.makedirs(FAISS_DIR, exist_ok=True)
    try:
        if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
            index = faiss.read_index(INDEX_FILE)
            with open(METADATA_FILE, "r") as f:
                metadata = json.load(f)
            return index, metadata
    except Exception as e:
        print(f"[Warning] Could not load existing index: {e}")
        
    index = faiss.IndexFlatL2(embedding_dim)
    return index, []

# ==========================================
# ⚡ DYNAMIC PROGRAMMING: 0/1 KNAPSACK ALGORITHM for CONTEXT OPTIMIZATION
# ==========================================
def optimize_context_knapsack(candidates, max_chars):
    """
    Applies the classic Dynamic Programming 0/1 Knapsack algorithm.
    Goal: Select a subset of retrieved chunks to maximize total Relevance (value)
    without exceeding the total Character Limit (max_chars / weight).
    """
    n = len(candidates)
    if n == 0:
        return []

    # DP Matrix: dp[i][w] maps to max value using first i items and capacity w
    # We scale down weights by dividing by 10 to drastically speed up DP table allocation
    SCALE_FACTOR = 10
    W = max_chars // SCALE_FACTOR
    
    weights = [max(1, item['weight'] // SCALE_FACTOR) for item in candidates]
    values = [item['value'] for item in candidates]
    
    dp = [[0.0] * (W + 1) for _ in range(n + 1)]
    
    start_dp_time = time.time()
    for i in range(1, n + 1):
        for w in range(1, W + 1):
            if weights[i-1] <= w:
                # Max of including or excluding current chunk
                dp[i][w] = max(
                    dp[i-1][w],
                    dp[i-1][w - weights[i-1]] + values[i-1]
                )
            else:
                dp[i][w] = dp[i-1][w]
    
    # Backtrack to find chosen items
    chosen_items = []
    w = W
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            chosen_items.append(candidates[i-1])
            w -= weights[i-1]
            
    exec_time = time.time() - start_dp_time
    print(f"\n[DP Profiler] 0/1 Knapsack mathematically picked {len(chosen_items)} optimal items out of {n} in {exec_time:.5f}s.")
    return chosen_items

def embed_single_file(filename, client):
    """Worker function for threading"""
    filepath = os.path.join(DATA_DIR, filename)
    start_time = time.time()

    # --- Text Indexing ---
    if filename.lower().endswith((".txt", ".md")):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=content
        )
        vector = response.embeddings[0].values
        elapsed = time.time() - start_time
        return vector, {"file": filename, "type": "text", "content": content, "weight": len(content)}, elapsed, None

    # --- Image Indexing ---
    elif filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(filepath)
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=img
        )
        vector = response.embeddings[0].values
        elapsed = time.time() - start_time
        return vector, {"file": filename, "type": "image", "content": "Visual content", "weight": 500}, elapsed, None
        
    return None, None, 0, f"Unsupported file: {filename}"

def index_data():
    """Scans the data/ folder and indexes text files and images using multi-threading."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(FAISS_DIR, exist_ok=True)
        
        main_start = time.time()
        client = get_client()
        index, metadata = load_or_create_index()
        
        indexed_files = {item['file'] for item in metadata}
        files = [f for f in os.listdir(DATA_DIR) if f not in indexed_files]
        
        if not files:
            print("\n[Indexer] No new text or image files to index.")
            return

        print(f"\n[Indexer] Found {len(files)} new files. Embarking on parallel indexing...")
        new_embeddings = []
        new_metadata = []

        # 🚀 PARALLEL THREAD PROCESSING
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_file = {executor.submit(embed_single_file, fname, client): fname for fname in files}
            for future in concurrent.futures.as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    vector, meta, elapsed, err = future.result()
                    if err:
                        print(f"[{filename}] Skipped: {err}")
                    elif vector:
                        if index.ntotal == 0 and len(vector) != index.d:
                            index = faiss.IndexFlatL2(len(vector))
                        new_embeddings.append(vector)
                        new_metadata.append(meta)
                        print(f"[{filename}] Successfully embedded in {elapsed:.2f}s")
                except Exception as e:
                    print(f"[{filename}] Unhandled Thread Exception: {e}")

        if new_embeddings:
            embeddings_np = np.array(new_embeddings).astype("float32")
            index.add(embeddings_np)
            metadata.extend(new_metadata)
            
            faiss.write_index(index, INDEX_FILE)
            with open(METADATA_FILE, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
            process_time = time.time() - main_start
            print(f"\n[Indexer] Successfully built index of {len(new_embeddings)} files in {process_time:.2f} seconds!")
            
    except Exception as e:
        print(f"[Indexer Error]: {e}")

def chat():
    """Starts the Advanced RAG Chatbot (with Latency tracking & DP Knapsack Optimization)."""
    try:
        client = get_client()
        index, metadata = load_or_create_index()
        
        print("\n==================================================")
        print("Gemini 2.5 Flash Optimized RAG Chatbot Initialized!")
        if index.ntotal > 0:
            print(f"FAISS Vector DB loaded with {index.ntotal} documents from '{FAISS_DIR}'.")
        else:
            print("[Warning] Vector DB is empty. Run option 1 to index data.")
        print("Type 'quit' or 'exit' to end the conversation.")
        print("==================================================\n")

        chat_session = client.chats.create(model=GENERATION_MODEL)

        while True:
            try:
                user_input = input("You: ")
                if user_input.strip().lower() in ['quit', 'exit']:
                    break
                if not user_input.strip():
                    continue
                
                # -----------------------
                # 1. RAG RETRIEVAL PHASE
                # -----------------------
                t0 = time.time()
                candidates = []
                
                if index.ntotal > 0:
                    try:
                        embed_res = client.models.embed_content(
                            model=EMBEDDING_MODEL,
                            contents=user_input
                        )
                        query_vector = np.array([embed_res.embeddings[0].values]).astype("float32")
                        
                        # Search FAISS (Retrieve Top 10 broad candidates)
                        k = min(10, index.ntotal)
                        distances, indices = index.search(query_vector, k)
                        
                        for idx, dist in zip(indices[0], distances[0]):
                            if idx != -1:
                                matched_data = metadata[idx]
                                # Convert L2 Distance to a "Relevance Score (Value)" for the DP knapsack
                                # We invert distance: small distance -> large value
                                val = 1.0 / (dist + 0.001)
                                candidates.append({
                                    "file": matched_data['file'],
                                    "type": matched_data['type'],
                                    "content": matched_data['content'],
                                    "value": float(val),
                                    "weight": matched_data.get('weight', len(matched_data.get('content', ''))),
                                    "dist": float(dist)
                                })
                                
                    except Exception as e:
                        print(f"[RAG Retrieval Error]: {e}")
                
                retrieval_time = time.time() - t0
                print(f"[Profiling] Raw Embed & DB Search took {retrieval_time:.3f}s")
                
                # -----------------------
                # 2. DP CONTEXT OPTIMIZATION
                # -----------------------
                chosen_items = optimize_context_knapsack(candidates, max_chars=MAX_CONTEXT_CHARS)
                
                # -----------------------
                # 3. GENERATION PHASE
                # -----------------------
                t1 = time.time()
                contexts = []
                image_contexts = []
                
                for item in chosen_items:
                    if item['type'] == 'text':
                        contexts.append(f"Context from {item['file']} (Score {item['value']:.2f}):\n{item['content']}\n")
                    elif item['type'] == 'image':
                        img_path = os.path.join(DATA_DIR, item['file'])
                        try:
                            img = Image.open(img_path)
                            image_contexts.append(img)
                            contexts.append(f"Context from {item['file']} (Score {item['value']:.2f}): (An image has been attached to this prompt for you to see).\n")
                        except Exception as img_err:
                            print(f"Error loading retrieved image: {img_err}")
                            
                try:
                    augmented_prompt_text = user_input
                    if contexts:
                        context_str = "\n".join(contexts)
                        augmented_prompt_text = (
                            f"Below is some mathematically optimized context retrieved from my local data (using a DP 0/1 knapsack filter):\n"
                            f"----------\n{context_str}\n----------\n"
                            f"Based on the context and your knowledge, perfectly answer the user's query:\n\n"
                            f"{user_input}"
                        )
                        print(f"[System: Injected {len(chosen_items)} optimal pieces of context.]")
                    
                    message_parts = [augmented_prompt_text]
                    message_parts.extend(image_contexts)
                    
                    response = chat_session.send_message(message_parts)
                    generation_time = time.time() - t1
                    
                    print(f"Gemini: {response.text}")
                    print(f"==================================================")
                    print(f"⏱️ [Profiling] Generation Runtime: {generation_time:.3f}s | Total Pipeline Time: {retrieval_time + generation_time:.3f}s")
                    print(f"==================================================")
                except Exception as e:
                    print(f"\n[Generation Error]: {e}")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n[Unexpected Chat Error]: {e}")

    except Exception as e:
        print(f"Failed to start chatbot: {e}")

def main():
    while True:
        try:
            print("\nMenu: DP Optimized RAG")
            print("1. Index new data (MULTI-THREADED)")
            print("2. Start RAG Chatbot")
            print("3. Exit")
            choice = input("Enter choice (1-3): ")
            
            if choice == "1":
                index_data()
            elif choice == "2":
                chat()
            elif choice == "3":
                break
            else:
                print("Invalid choice.")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Main loop error: {e}")

if __name__ == "__main__":
    main()
