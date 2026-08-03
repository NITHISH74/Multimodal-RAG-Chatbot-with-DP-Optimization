"""
Embedding helpers for Gemini and Cohere (text, image, query).

Failure mode (Phase 3.5): per-model retry with exponential backoff. Errors
are reported via an optional `on_error` callback rather than silently
swallowed — the indexer surfaces them in the UI and writes them to
`st.session_state.embedding_errors` so a batch failure is visible to the
operator instead of looking like a quiet partial success.
"""
from __future__ import annotations

import base64
import io
import time

import config
from clients import get_gemini_client, get_cohere_client


def _retry(call, *, label, on_error=None):
    """Wrap a model call with up to EMBED_RETRY_MAX retries (exp backoff).

    Returns the call's value on success, None if every attempt failed.
    Errors are reported via `on_error(label, exc)` so the caller can
    surface them; we never re-raise from the indexer path.
    """
    attempts = max(0, int(getattr(config, "EMBED_RETRY_MAX", 0))) + 1
    base = float(getattr(config, "EMBED_RETRY_BASE_DELAY", 0.6))
    last = None
    for i in range(attempts):
        try:
            return call()
        except Exception as e:                          # noqa: BLE001
            last = e
            if on_error and i == 0:
                on_error(label, e)
            if i < attempts - 1:
                time.sleep(base * (2 ** i))
    if on_error:
        on_error(f"{label} (giving up)", last)
    return None


def embed_text(text, model_name, on_error=None):
    """Embed a text chunk for indexing. on_error(label, exc) is called on
    each failed attempt; returns None if every attempt failed."""
    if model_name == "Gemini":
        def _call():
            res = get_gemini_client().models.embed_content(
                model=config.GEMINI_EMBED_MODEL, contents=text)
            return list(res.embeddings[0].values)
        return _retry(_call, label=f"embed_text[{model_name}]", on_error=on_error)
    if model_name == "Cohere":
        def _call():
            res = get_cohere_client().embed(
                model=config.COHERE_EMBED_MODEL, texts=[text],
                input_type="search_document", embedding_types=["float"])
            return list(res.embeddings.float_[0])
        return _retry(_call, label=f"embed_text[{model_name}]", on_error=on_error)
    if on_error:
        on_error(f"embed_text[unknown:{model_name}]", ValueError(f"unknown model {model_name}"))
    return None


def embed_image(pil_image, model_name, on_error=None):
    """Embed an image. Gemini takes the PIL image directly; Cohere needs a
    base64 data URL. Returns None if every attempt failed."""
    if model_name == "Gemini":
        def _call():
            res = get_gemini_client().models.embed_content(
                model=config.GEMINI_EMBED_MODEL, contents=pil_image)
            return list(res.embeddings[0].values)
        return _retry(_call, label=f"embed_image[{model_name}]", on_error=on_error)
    if model_name == "Cohere":
        def _call():
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
        return _retry(_call, label=f"embed_image[{model_name}]", on_error=on_error)
    if on_error:
        on_error(f"embed_image[unknown:{model_name}]", ValueError(f"unknown model {model_name}"))
    return None


# Models we populate at index time so retrieval works no matter which model
# the user later queries with (avoids the silent "0 chunks after switching
# models" trap — the crawled/uploaded row carries BOTH embeddings).
INDEX_MODELS = ("Gemini", "Cohere")


def embed_text_all(text, on_error=None):
    """Embed a text chunk under every index model. Returns {model: vector}.

    A model that errors (e.g. its API key isn't configured) is skipped. Use
    `on_error(label, exc)` to surface failures (the UI pushes them into
    `st.session_state.embedding_errors` so the operator can see which model
    dropped a batch).
    """
    out = {}
    for m in INDEX_MODELS:
        v = embed_text(text, m, on_error=on_error)
        if v is not None:
            out[m] = v
    return out


def embed_image_all(pil_image, on_error=None):
    """Embed an image under every index model. Returns {model: vector}."""
    out = {}
    for m in INDEX_MODELS:
        v = embed_image(pil_image, m, on_error=on_error)
        if v is not None:
            out[m] = v
    return out


def embed_query(query_text, model_name, on_error=None):
    """Embed a user query for retrieval. Same retry semantics as embed_text."""
    if model_name == "Gemini":
        def _call():
            res = get_gemini_client().models.embed_content(
                model=config.GEMINI_EMBED_MODEL, contents=query_text)
            return list(res.embeddings[0].values)
        return _retry(_call, label=f"embed_query[{model_name}]", on_error=on_error)
    if model_name == "Cohere":
        def _call():
            res = get_cohere_client().embed(
                model=config.COHERE_EMBED_MODEL, texts=[query_text],
                input_type="search_query", embedding_types=["float"])
            return list(res.embeddings.float_[0])
        return _retry(_call, label=f"embed_query[{model_name}]", on_error=on_error)
    if on_error:
        on_error(f"embed_query[unknown:{model_name}]", ValueError(f"unknown model {model_name}"))
    return None
