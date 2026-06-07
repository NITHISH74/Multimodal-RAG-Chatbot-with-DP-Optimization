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
