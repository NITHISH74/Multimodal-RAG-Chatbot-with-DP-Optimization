"""
Embedding helpers for Gemini and Cohere (text, image, query).
"""
import base64
import io

import config
from clients import get_gemini_client, get_cohere_client


def embed_text(text, model_name):
    """Embed a text chunk for indexing."""
    if model_name == "Gemini":
        res = get_gemini_client().models.embed_content(
            model=config.GEMINI_EMBED_MODEL, contents=text)
        return list(res.embeddings[0].values)
    res = get_cohere_client().embed(
        model=config.COHERE_EMBED_MODEL, texts=[text],
        input_type="search_document", embedding_types=["float"])
    return list(res.embeddings.float_[0])


def embed_image(pil_image, model_name):
    """Embed an image. Gemini takes the PIL image directly; Cohere needs a
    base64 data URL."""
    if model_name == "Gemini":
        res = get_gemini_client().models.embed_content(
            model=config.GEMINI_EMBED_MODEL, contents=pil_image)
        return list(res.embeddings[0].values)
    buf = io.BytesIO()
    fmt = (pil_image.format or "PNG").lower()
    pil_image.save(buf, format=pil_image.format or "PNG")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(fmt, "png")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    res = get_cohere_client().embed(
        model=config.COHERE_EMBED_MODEL,
        images=[f"data:image/{mime};base64,{encoded}"],
        input_type="image", embedding_types=["float"])
    return list(res.embeddings.float_[0])


# Models we populate at index time so retrieval works no matter which model
# the user later queries with (avoids the silent "0 chunks after switching
# models" trap — the crawled/uploaded row carries BOTH embeddings).
INDEX_MODELS = ("Gemini", "Cohere")


def embed_text_all(text):
    """Embed a text chunk under every index model. Returns {model: vector}.

    A model that errors (e.g. its API key isn't configured) is skipped, so
    indexing still succeeds with whatever model(s) are available.
    """
    out = {}
    for m in INDEX_MODELS:
        try:
            out[m] = embed_text(text, m)
        except Exception:
            pass
    return out


def embed_image_all(pil_image):
    """Embed an image under every index model. Returns {model: vector}.

    Same fail-soft behaviour as embed_text_all.
    """
    out = {}
    for m in INDEX_MODELS:
        try:
            out[m] = embed_image(pil_image, m)
        except Exception:
            pass
    return out


def embed_query(query_text, model_name):
    """Embed a user query for retrieval."""
    if model_name == "Gemini":
        res = get_gemini_client().models.embed_content(
            model=config.GEMINI_EMBED_MODEL, contents=query_text)
        return list(res.embeddings[0].values)
    res = get_cohere_client().embed(
        model=config.COHERE_EMBED_MODEL, texts=[query_text],
        input_type="search_query", embedding_types=["float"])
    return list(res.embeddings.float_[0])
