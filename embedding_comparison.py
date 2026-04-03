"""
╔══════════════════════════════════════════════════════════════════════╗
║  EMBEDDING MODEL BENCHMARK: Gemini vs Cohere                       ║
║  Compares: Latency, Dimensions, Retrieval Quality, Multimodal      ║
║  Models: gemini-embedding-2-preview  vs  embed-v4.0                ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import time
import base64
import json
import numpy as np
from dotenv import load_dotenv
from google import genai
from PIL import Image
import cohere

# ── Configuration ──────────────────────────────────────────────────────
load_dotenv()

GEMINI_API_KEY = os.getenv("gemini_api_key")
COHERE_API_KEY = os.getenv("cohere_api_key")

GEMINI_EMBED_MODEL = "gemini-embedding-2-preview"
COHERE_EMBED_MODEL = "embed-v4.0"

DATA_DIR = "data"

# ── Test Data ──────────────────────────────────────────────────────────
# Diverse queries to test semantic understanding across domains
TEST_QUERIES = [
    "What is the Hyper-Vortex Cooling Protocol?",
    "Tell me about the Golden Gate Bridge color",
    "What happened on day 4 of Project Solaris?",
    "What are the student's skills and interests?",
    "thermal management system for data centers",
]

# Ground-truth mapping: which file section each query should match best
GROUND_TRUTH = [
    "HVCP",       # Query 0 -> Technical doc
    "Bridge",     # Query 1 -> Golden Gate
    "Solaris",    # Query 2 -> Trip log
    "student",    # Query 3 -> About myself
    "HVCP",       # Query 4 -> Technical doc (semantic match)
]

# Document chunks (split from demo.txt for granular retrieval testing)
DOC_CHUNKS = []


def load_document_chunks():
    """Split demo.txt into logical sections for retrieval quality testing."""
    global DOC_CHUNKS
    filepath = os.path.join(DATA_DIR, "demo.txt")
    if not os.path.exists(filepath):
        print("[ERROR] demo.txt not found in data/ folder.")
        return False

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by numbered sections
    sections = []
    current = []
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped and stripped[0].isdigit() and "." in stripped[:3] and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current).strip())

    # Tag each chunk with a label for ground-truth evaluation
    labels = ["HVCP", "Bridge", "Solaris", "student"]
    for i, section in enumerate(sections):
        if section.strip():
            label = labels[i] if i < len(labels) else f"section_{i}"
            DOC_CHUNKS.append({"text": section, "label": label})

    print(f"[Setup] Loaded {len(DOC_CHUNKS)} document chunks for retrieval testing.\n")
    return True


# ══════════════════════════════════════════════════════════════════════
#  HELPER UTILITIES
# ══════════════════════════════════════════════════════════════════════

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / norm if norm > 0 else 0.0


def image_to_base64_data_url(image_path):
    """Convert an image file to a base64-encoded data URL for Cohere."""
    ext = os.path.splitext(image_path)[1].lower().replace(".", "")
    mime_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp", "gif": "gif"}
    mime = mime_map.get(ext, "jpeg")
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{mime};base64,{encoded}"


def format_table(headers, rows, col_widths=None):
    """Pretty-print a table to console."""
    if col_widths is None:
        col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]

    header_line = " │ ".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    separator = "─┼─".join("─" * w for w in col_widths)
    print(f" {header_line}")
    print(f" {separator}")
    for row in rows:
        print(" │ ".join(str(c).ljust(w) for c, w in zip(row, col_widths)))
    print()


# ══════════════════════════════════════════════════════════════════════
#  GEMINI EMBEDDING ENGINE
# ══════════════════════════════════════════════════════════════════════

class GeminiEmbedder:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("Missing 'gemini_api_key' in .env")
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = GEMINI_EMBED_MODEL
        self.name = "Gemini Embedding 2"

    def embed_text(self, text):
        """Embed a single text string. Returns (vector, latency_ms)."""
        t0 = time.perf_counter()
        response = self.client.models.embed_content(model=self.model, contents=text)
        latency = (time.perf_counter() - t0) * 1000
        return response.embeddings[0].values, latency

    def embed_texts(self, texts):
        """Embed multiple text strings. Returns (list_of_vectors, total_latency_ms)."""
        vectors = []
        total_latency = 0
        for text in texts:
            vec, lat = self.embed_text(text)
            vectors.append(vec)
            total_latency += lat
        return vectors, total_latency

    def embed_image(self, image_path):
        """Embed an image file. Returns (vector, latency_ms)."""
        img = Image.open(image_path)
        t0 = time.perf_counter()
        response = self.client.models.embed_content(model=self.model, contents=img)
        latency = (time.perf_counter() - t0) * 1000
        return response.embeddings[0].values, latency


# ══════════════════════════════════════════════════════════════════════
#  COHERE EMBEDDING ENGINE
# ══════════════════════════════════════════════════════════════════════

class CohereEmbedder:
    def __init__(self):
        if not COHERE_API_KEY:
            raise ValueError("Missing 'cohere_api_key' in .env")
        self.client = cohere.ClientV2(api_key=COHERE_API_KEY)
        self.model = COHERE_EMBED_MODEL
        self.name = "Cohere Embed v4.0"

    def embed_text(self, text, input_type="search_query"):
        """Embed a single text string. Returns (vector, latency_ms)."""
        t0 = time.perf_counter()
        response = self.client.embed(
            model=self.model,
            texts=[text],
            input_type=input_type,
            embedding_types=["float"]
        )
        latency = (time.perf_counter() - t0) * 1000
        return response.embeddings.float_[0], latency

    def embed_texts(self, texts, input_type="search_document"):
        """Embed multiple text strings. Returns (list_of_vectors, total_latency_ms)."""
        t0 = time.perf_counter()
        response = self.client.embed(
            model=self.model,
            texts=texts,
            input_type=input_type,
            embedding_types=["float"]
        )
        latency = (time.perf_counter() - t0) * 1000
        return [v for v in response.embeddings.float_], latency

    def embed_image(self, image_path):
        """Embed an image file via base64. Returns (vector, latency_ms)."""
        data_url = image_to_base64_data_url(image_path)
        t0 = time.perf_counter()
        response = self.client.embed(
            model=self.model,
            images=[data_url],
            input_type="image",
            embedding_types=["float"]
        )
        latency = (time.perf_counter() - t0) * 1000
        return response.embeddings.float_[0], latency


# ══════════════════════════════════════════════════════════════════════
#  BENCHMARK TESTS
# ══════════════════════════════════════════════════════════════════════

def benchmark_text_embedding(gemini: GeminiEmbedder, cohere_emb: CohereEmbedder):
    """
    TEST 1: Text Embedding – Latency & Dimensions
    Embeds all test queries and measures speed + vector size.
    """
    print("=" * 70)
    print("  TEST 1: TEXT EMBEDDING – LATENCY & DIMENSIONS")
    print("=" * 70)

    gemini_latencies = []
    cohere_latencies = []
    gemini_dim = 0
    cohere_dim = 0

    rows = []
    for i, query in enumerate(TEST_QUERIES):
        g_vec, g_lat = gemini.embed_text(query)
        c_vec, c_lat = cohere_emb.embed_text(query)

        gemini_latencies.append(g_lat)
        cohere_latencies.append(c_lat)
        gemini_dim = len(g_vec)
        cohere_dim = len(c_vec)

        rows.append([
            f"Q{i+1}",
            query[:45] + "..." if len(query) > 45 else query,
            f"{g_lat:.0f} ms",
            f"{c_lat:.0f} ms",
        ])

    headers = ["#", "Query", "Gemini (ms)", "Cohere (ms)"]
    format_table(headers, rows, col_widths=[3, 48, 12, 12])

    avg_g = np.mean(gemini_latencies)
    avg_c = np.mean(cohere_latencies)

    print(f"  📐 Gemini Embedding Dimensions : {gemini_dim}")
    print(f"  📐 Cohere Embedding Dimensions : {cohere_dim}")
    print(f"  ⚡ Gemini Avg Latency          : {avg_g:.1f} ms")
    print(f"  ⚡ Cohere Avg Latency          : {avg_c:.1f} ms")
    faster = "Gemini" if avg_g < avg_c else "Cohere"
    print(f"  🏆 Faster for Text Embedding   : {faster} (by {abs(avg_g - avg_c):.1f} ms)\n")

    return {
        "gemini_dim": gemini_dim,
        "cohere_dim": cohere_dim,
        "gemini_avg_latency": avg_g,
        "cohere_avg_latency": avg_c,
        "latency_winner": faster,
    }


def benchmark_retrieval_quality(gemini: GeminiEmbedder, cohere_emb: CohereEmbedder):
    """
    TEST 2: Retrieval Quality – Precision@1
    Embeds document chunks and queries, measures if top-1 retrieved chunk
    matches the ground-truth label.
    """
    print("=" * 70)
    print("  TEST 2: RETRIEVAL QUALITY – PRECISION@1 (COSINE SIMILARITY)")
    print("=" * 70)

    if not DOC_CHUNKS:
        print("  [SKIP] No document chunks loaded.\n")
        return {}

    # ── Embed all document chunks ──
    chunk_texts = [c["text"] for c in DOC_CHUNKS]
    chunk_labels = [c["label"] for c in DOC_CHUNKS]

    print("  Embedding document chunks with both models...")
    g_doc_vecs, g_doc_lat = gemini.embed_texts(chunk_texts)
    c_doc_vecs, c_doc_lat = cohere_emb.embed_texts(chunk_texts, input_type="search_document")

    print(f"  Gemini indexed {len(chunk_texts)} chunks in {g_doc_lat:.0f} ms")
    print(f"  Cohere indexed {len(chunk_texts)} chunks in {c_doc_lat:.0f} ms\n")

    # ── Query and evaluate retrieval ──
    gemini_correct = 0
    cohere_correct = 0
    rows = []

    for i, query in enumerate(TEST_QUERIES):
        expected_label = GROUND_TRUTH[i]

        # Gemini retrieval
        g_q_vec, _ = gemini.embed_text(query)
        g_sims = [cosine_similarity(g_q_vec, dv) for dv in g_doc_vecs]
        g_top_idx = int(np.argmax(g_sims))
        g_top_label = chunk_labels[g_top_idx]
        g_top_score = g_sims[g_top_idx]
        g_hit = "✅" if g_top_label == expected_label else "❌"
        if g_top_label == expected_label:
            gemini_correct += 1

        # Cohere retrieval
        c_q_vec, _ = cohere_emb.embed_text(query, input_type="search_query")
        c_sims = [cosine_similarity(c_q_vec, dv) for dv in c_doc_vecs]
        c_top_idx = int(np.argmax(c_sims))
        c_top_label = chunk_labels[c_top_idx]
        c_top_score = c_sims[c_top_idx]
        c_hit = "✅" if c_top_label == expected_label else "❌"
        if c_top_label == expected_label:
            cohere_correct += 1

        rows.append([
            f"Q{i+1}",
            query[:35] + "..." if len(query) > 35 else query,
            expected_label,
            f"{g_hit} {g_top_label} ({g_top_score:.3f})",
            f"{c_hit} {c_top_label} ({c_top_score:.3f})",
        ])

    headers = ["#", "Query", "Expected", "Gemini Result", "Cohere Result"]
    format_table(headers, rows, col_widths=[3, 38, 10, 22, 22])

    g_precision = gemini_correct / len(TEST_QUERIES) * 100
    c_precision = cohere_correct / len(TEST_QUERIES) * 100

    print(f"  🎯 Gemini Precision@1 : {gemini_correct}/{len(TEST_QUERIES)} = {g_precision:.0f}%")
    print(f"  🎯 Cohere Precision@1 : {cohere_correct}/{len(TEST_QUERIES)} = {c_precision:.0f}%")
    winner = "Gemini" if g_precision >= c_precision else "Cohere"
    if g_precision == c_precision:
        winner = "TIE"
    print(f"  🏆 Retrieval Winner   : {winner}\n")

    return {
        "gemini_precision": g_precision,
        "cohere_precision": c_precision,
        "retrieval_winner": winner,
    }


def benchmark_image_embedding(gemini: GeminiEmbedder, cohere_emb: CohereEmbedder):
    """
    TEST 3: Multimodal – Image Embedding & Cross-Modal Search
    Embeds images, then searches them using text queries.
    """
    print("=" * 70)
    print("  TEST 3: MULTIMODAL – IMAGE EMBEDDING & CROSS-MODAL SEARCH")
    print("=" * 70)

    # Find image files in data/
    image_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
    if not image_files:
        print("  [SKIP] No images found in data/ folder.\n")
        return {}

    gemini_img_results = {}
    cohere_img_results = {}

    rows = []
    for img_file in image_files:
        img_path = os.path.join(DATA_DIR, img_file)

        # Gemini image embedding
        try:
            g_vec, g_lat = gemini.embed_image(img_path)
            g_dim = len(g_vec)
            g_status = f"✅ {g_dim}d in {g_lat:.0f}ms"
            gemini_img_results[img_file] = {"vec": g_vec, "latency": g_lat, "dim": g_dim}
        except Exception as e:
            g_status = f"❌ {str(e)[:30]}"
            gemini_img_results[img_file] = None

        # Cohere image embedding
        try:
            c_vec, c_lat = cohere_emb.embed_image(img_path)
            c_dim = len(c_vec)
            c_status = f"✅ {c_dim}d in {c_lat:.0f}ms"
            cohere_img_results[img_file] = {"vec": c_vec, "latency": c_lat, "dim": c_dim}
        except Exception as e:
            c_status = f"❌ {str(e)[:30]}"
            cohere_img_results[img_file] = None

        rows.append([img_file[:25], g_status, c_status])

    headers = ["Image File", "Gemini Result", "Cohere Result"]
    format_table(headers, rows, col_widths=[25, 25, 25])

    # ── Cross-Modal Search Test ──
    # Use a text query to search against image embeddings
    cross_modal_query = "machine learning framework architecture diagram"
    print(f"  🔍 Cross-Modal Query: \"{cross_modal_query}\"")
    print()

    # Gemini cross-modal
    try:
        g_q_vec, _ = gemini.embed_text(cross_modal_query)
        for img_file, data in gemini_img_results.items():
            if data:
                sim = cosine_similarity(g_q_vec, data["vec"])
                print(f"     Gemini  │ {img_file:25s} │ Cosine Sim: {sim:.4f}")
    except Exception as e:
        print(f"     Gemini cross-modal error: {e}")

    # Cohere cross-modal
    try:
        c_q_vec, _ = cohere_emb.embed_text(cross_modal_query, input_type="search_query")
        for img_file, data in cohere_img_results.items():
            if data:
                sim = cosine_similarity(c_q_vec, data["vec"])
                print(f"     Cohere  │ {img_file:25s} │ Cosine Sim: {sim:.4f}")
    except Exception as e:
        print(f"     Cohere cross-modal error: {e}")

    print()

    gemini_success = sum(1 for v in gemini_img_results.values() if v is not None)
    cohere_success = sum(1 for v in cohere_img_results.values() if v is not None)

    return {
        "gemini_images_embedded": gemini_success,
        "cohere_images_embedded": cohere_success,
        "total_images": len(image_files),
    }


def benchmark_semantic_similarity(gemini: GeminiEmbedder, cohere_emb: CohereEmbedder):
    """
    TEST 4: Semantic Similarity Discrimination
    Tests if the model can properly distinguish similar vs dissimilar text pairs.
    A good embedding model should give HIGH scores to semantically similar pairs
    and LOW scores to dissimilar pairs.
    """
    print("=" * 70)
    print("  TEST 4: SEMANTIC SIMILARITY DISCRIMINATION")
    print("=" * 70)

    test_pairs = [
        # (text_a, text_b, expected: "high" or "low")
        ("The cooling system uses fluorocarbon liquid", "HVCP thermal management for data centers", "high"),
        ("Golden Gate Bridge is painted International Orange", "The bridge in San Francisco has a distinctive color", "high"),
        ("The team deployed solar arrays at base camp", "The cooling protocol uses MQTT-SN sensors", "low"),
        ("I am a student interested in AI and ML", "machine learning and artificial intelligence student", "high"),
        ("The atmospheric pressure was lower than baseline", "The bridge was completed ahead of schedule", "low"),
    ]

    gemini_scores = []
    cohere_scores = []
    rows = []

    for text_a, text_b, expected in test_pairs:
        g_a, _ = gemini.embed_text(text_a)
        g_b, _ = gemini.embed_text(text_b)
        g_sim = cosine_similarity(g_a, g_b)

        c_a, _ = cohere_emb.embed_text(text_a, input_type="search_document")
        c_b, _ = cohere_emb.embed_text(text_b, input_type="search_document")
        c_sim = cosine_similarity(c_a, c_b)

        gemini_scores.append((g_sim, expected))
        cohere_scores.append((c_sim, expected))

        rows.append([
            text_a[:28] + "...",
            text_b[:28] + "...",
            expected.upper(),
            f"{g_sim:.4f}",
            f"{c_sim:.4f}",
        ])

    headers = ["Text A", "Text B", "Expect", "Gemini", "Cohere"]
    format_table(headers, rows, col_widths=[31, 31, 6, 8, 8])

    # ── Discrimination Score ──
    # Good model: high expected pairs > 0.7, low expected pairs < 0.5
    def discrimination_score(scores):
        high_scores = [s for s, e in scores if e == "high"]
        low_scores = [s for s, e in scores if e == "low"]
        avg_high = np.mean(high_scores) if high_scores else 0
        avg_low = np.mean(low_scores) if low_scores else 0
        # Discrimination = gap between avg high and avg low (bigger is better)
        return avg_high, avg_low, avg_high - avg_low

    g_high, g_low, g_disc = discrimination_score(gemini_scores)
    c_high, c_low, c_disc = discrimination_score(cohere_scores)

    print(f"  Gemini │ Avg Similar: {g_high:.4f} │ Avg Dissimilar: {g_low:.4f} │ Gap: {g_disc:.4f}")
    print(f"  Cohere │ Avg Similar: {c_high:.4f} │ Avg Dissimilar: {c_low:.4f} │ Gap: {c_disc:.4f}")
    winner = "Gemini" if g_disc >= c_disc else "Cohere"
    print(f"  🏆 Better Discrimination: {winner} (larger gap = better semantic separation)\n")

    return {
        "gemini_discrimination": g_disc,
        "cohere_discrimination": c_disc,
        "discrimination_winner": winner,
    }


# ══════════════════════════════════════════════════════════════════════
#  FINAL VERDICT
# ══════════════════════════════════════════════════════════════════════

def print_final_verdict(results):
    """Aggregate all test results and declare the overall winner."""
    print("=" * 70)
    print("  ⚖️  FINAL VERDICT: GEMINI vs COHERE EMBEDDING MODELS")
    print("=" * 70)

    scores = {"Gemini": 0, "Cohere": 0}

    categories = [
        ("Text Latency", results.get("latency_winner", "TIE")),
        ("Retrieval Precision@1", results.get("retrieval_winner", "TIE")),
        ("Semantic Discrimination", results.get("discrimination_winner", "TIE")),
    ]

    # Multimodal scoring
    g_img = results.get("gemini_images_embedded", 0)
    c_img = results.get("cohere_images_embedded", 0)
    if g_img > c_img:
        categories.append(("Multimodal (Image Support)", "Gemini"))
    elif c_img > g_img:
        categories.append(("Multimodal (Image Support)", "Cohere"))
    else:
        categories.append(("Multimodal (Image Support)", "TIE"))

    rows = []
    for cat, winner in categories:
        if winner == "Gemini":
            scores["Gemini"] += 1
            badge = "🟢 Gemini"
        elif winner == "Cohere":
            scores["Cohere"] += 1
            badge = "🟣 Cohere"
        else:
            scores["Gemini"] += 0.5
            scores["Cohere"] += 0.5
            badge = "🟡 TIE"
        rows.append([cat, badge])

    headers = ["Category", "Winner"]
    format_table(headers, rows, col_widths=[35, 15])

    print(f"  📊 SCOREBOARD")
    print(f"     Gemini : {scores['Gemini']:.1f} / {len(categories)}")
    print(f"     Cohere : {scores['Cohere']:.1f} / {len(categories)}")
    print()

    # Model specs comparison
    print("  📋 MODEL SPECIFICATIONS COMPARISON")
    print("  ─" * 35)
    spec_rows = [
        ["Model Name", GEMINI_EMBED_MODEL, COHERE_EMBED_MODEL],
        ["Dimensions", str(results.get("gemini_dim", "N/A")), str(results.get("cohere_dim", "N/A"))],
        ["Avg Text Latency", f"{results.get('gemini_avg_latency', 0):.0f} ms", f"{results.get('cohere_avg_latency', 0):.0f} ms"],
        ["Retrieval Precision", f"{results.get('gemini_precision', 0):.0f}%", f"{results.get('cohere_precision', 0):.0f}%"],
        ["Discrimination Gap", f"{results.get('gemini_discrimination', 0):.4f}", f"{results.get('cohere_discrimination', 0):.4f}"],
        ["Multimodal Support", "✅ Native (PIL)", "✅ Base64 Data URL"],
        ["Multilingual", "✅ 100+ langs", "✅ 100+ langs"],
        ["Matryoshka Dims", "❌", "✅ (64, 128, 256...)"],
        ["Max Context", "~8K tokens", "128K tokens"],
        ["Quantization", "❌", "✅ int8 / binary"],
    ]
    format_table(["Spec", "Gemini", "Cohere"], spec_rows, col_widths=[22, 22, 22])

    overall = "Gemini" if scores["Gemini"] > scores["Cohere"] else "Cohere" if scores["Cohere"] > scores["Gemini"] else "TIE"

    print("  ╔══════════════════════════════════════════════════════════════╗")
    if overall == "TIE":
        print("  ║  🏆 OVERALL RESULT: IT'S A TIE!                             ║")
    else:
        emoji = "🟢" if overall == "Gemini" else "🟣"
        print(f"  ║  🏆 OVERALL WINNER: {emoji} {overall:45s}║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print("  ║                                                            ║")
    print("  ║  💡 RECOMMENDATION FOR YOUR RAG SYSTEM:                    ║")
    print("  ║                                                            ║")
    if scores["Gemini"] >= scores["Cohere"]:
        print("  ║  • Gemini excels at native multimodal embedding (images    ║")
        print("  ║    directly via PIL) — ideal for your image+text RAG.      ║")
        print("  ║  • For a unified Gemini-only stack (embed + generate),     ║")
        print("  ║    Gemini reduces complexity and API key management.       ║")
    if scores["Cohere"] >= scores["Gemini"]:
        print("  ║  • Cohere offers Matryoshka dimensions for flexible        ║")
        print("  ║    storage/speed tradeoffs and int8 quantization.          ║")
        print("  ║  • Cohere's 128K context window is superior for long docs. ║")
    print("  ║                                                            ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print()


# ══════════════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║     EMBEDDING MODEL BENCHMARK: Gemini vs Cohere                     ║")
    print("║     Testing: Latency, Retrieval, Semantics, Multimodal              ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    # Validate API keys
    if not GEMINI_API_KEY:
        print("[FATAL] Missing 'gemini_api_key' in .env file!")
        return
    if not COHERE_API_KEY:
        print("[FATAL] Missing 'cohere_api_key' in .env file!")
        return

    print(f"  ✅ Gemini API Key : {GEMINI_API_KEY[:8]}...{GEMINI_API_KEY[-4:]}")
    print(f"  ✅ Cohere API Key : {COHERE_API_KEY[:8]}...{COHERE_API_KEY[-4:]}")
    print()

    # Load test data
    if not load_document_chunks():
        return

    # Initialize embedders
    try:
        gemini = GeminiEmbedder()
        print(f"  ✅ Gemini Embedder initialized ({GEMINI_EMBED_MODEL})")
    except Exception as e:
        print(f"  ❌ Gemini init failed: {e}")
        return

    try:
        cohere_emb = CohereEmbedder()
        print(f"  ✅ Cohere Embedder initialized ({COHERE_EMBED_MODEL})")
    except Exception as e:
        print(f"  ❌ Cohere init failed: {e}")
        return

    print("\n" + "─" * 70 + "\n")

    # ── Run all benchmarks ──
    all_results = {}

    # Test 1: Text Embedding Latency & Dimensions
    r1 = benchmark_text_embedding(gemini, cohere_emb)
    all_results.update(r1)

    # Test 2: Retrieval Quality
    r2 = benchmark_retrieval_quality(gemini, cohere_emb)
    all_results.update(r2)

    # Test 3: Image Embedding & Cross-Modal Search
    r3 = benchmark_image_embedding(gemini, cohere_emb)
    all_results.update(r3)

    # Test 4: Semantic Similarity Discrimination
    r4 = benchmark_semantic_similarity(gemini, cohere_emb)
    all_results.update(r4)

    # ── Final Verdict ──
    print_final_verdict(all_results)

    # Save results to JSON
    results_file = os.path.join("faiss_data", "benchmark_results.json")
    serializable = {k: v for k, v in all_results.items() if not isinstance(v, (np.ndarray, list))}
    with open(results_file, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  📁 Raw results saved to: {results_file}")


if __name__ == "__main__":
    main()
