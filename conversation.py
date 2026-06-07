"""
Conversation memory with a running summary (Phase 13.1 / 13.2).

Instead of sending the full chat history to the LLM on every turn, we keep:
  - a rolling natural-language summary of older turns, plus
  - the last few turns verbatim.

This bounds prompt growth while preserving continuity. app.py previously
sent NO history at all, so this also *adds* multi-turn memory.
"""
import config
from clients import get_gemini_client


def _recent(messages, n):
    return messages[-n:] if n > 0 else []


def build_history_block(messages, summary):
    """Compose the compact history string sent to the LLM:
    running summary + the last HISTORY_RECENT_TURNS messages verbatim."""
    parts = []
    if summary:
        parts.append(f"Conversation summary so far:\n{summary}")
    recent = _recent(messages, config.HISTORY_RECENT_TURNS * 2)  # user+assistant pairs
    if recent:
        convo = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)
        parts.append(f"Recent turns:\n{convo}")
    return "\n\n".join(parts)


def maybe_update_summary(messages, prior_summary):
    """Refresh the running summary once the conversation grows past the
    trigger length. Summarises everything except the most recent turns.

    Returns the (possibly unchanged) summary string. Fails soft: on any
    error the prior summary is returned unchanged.
    """
    if len(messages) < config.HISTORY_SUMMARY_TRIGGER:
        return prior_summary

    keep = config.HISTORY_RECENT_TURNS * 2
    older = messages[:-keep] if keep else messages
    if not older:
        return prior_summary

    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in older)
    prompt = (
        "Maintain a concise running summary of this chat for context in future "
        "turns. Merge the existing summary with the new turns. Keep it under "
        "150 words, factual, no preamble.\n\n"
        f"Existing summary:\n{prior_summary or '(none)'}\n\n"
        f"New turns:\n{transcript}\n\nUpdated summary:"
    )
    try:
        client = get_gemini_client()
        if client is None:
            return prior_summary
        res = client.models.generate_content(
            model=config.SUMMARY_MODEL, contents=[prompt])
        return (res.text or prior_summary).strip()
    except Exception:
        return prior_summary
