"""
Chat surface: message rendering, feedback controls, chat input.

The chat input itself is the only "live" widget here — it owns the
side-effect of appending to `st.session_state.messages` and running the
RAG pipeline.
"""
from __future__ import annotations

import time

import streamlit as st
from st_copy_to_clipboard import st_copy_to_clipboard

import conversation
import rag_db
from pipeline import SCHEMA_HELP, is_schema_error, run_rag_pipeline, save_chat_history


# ──────────────────────────────────────────────────────────────────────
#  Rendering helpers
# ──────────────────────────────────────────────────────────────────────
def _render_meta_chips(meta) -> None:
    cols = st.columns([1, 1, 1, 2])
    cols[0].caption(f"⏱️ {meta.get('retrieval_time', 0):.2f}s")
    cols[1].caption(f"⚡ {meta.get('generation_time', 0):.2f}s")
    cols[2].caption(f"📎 {meta.get('context_chunks', 0)}")
    if meta.get("fallback"):
        cols[3].caption("🛑 fallback")
    elif meta.get("guardrail_blocked"):
        cols[3].caption(f"🛡️ blocked ({meta['guardrail_blocked']})")
    elif meta.get("faithfulness_warning"):
        cols[3].caption(f"⚠️ low faithfulness")


def _render_dev_panel(meta) -> None:
    if not st.session_state.dev_mode:
        return
    with st.expander("🛠️ Query diagnostics", expanded=False):
        total = meta.get("retrieval_time", 0) + meta.get("generation_time", 0)
        c = st.columns(4)
        c[0].metric("Retrieval", f"{meta.get('retrieval_time', 0) * 1000:.0f} ms")
        c[1].metric("Generation", f"{meta.get('generation_time', 0) * 1000:.0f} ms")
        c[2].metric("Total", f"{total * 1000:.0f} ms")
        c[3].metric("Chunks", meta.get("context_chunks", 0))
        c2 = st.columns(3)
        c2[0].metric("Input tokens", meta.get("input_tokens", 0))
        c2[1].metric("Output tokens", meta.get("output_tokens", 0))
        c2[2].metric("Intent", meta.get("intent", "—"))
        st.caption(f"Threshold: {meta.get('threshold')} · Fallback: {meta.get('fallback')} · "
                   f"Scores: {meta.get('scores')}")


def _previous_user_question(message_index):
    for j in range(message_index - 1, -1, -1):
        msg = st.session_state.messages[j]
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def _submit_feedback(message_index, rating, comment=""):
    msg = st.session_state.messages[message_index]
    st.session_state.answer_feedback[message_index] = {
        "rating": rating, "comment": comment or "",
    }
    if rating == "down" and comment:
        st.session_state.feedback_guidance.append(comment)
    try:
        rag_db.save_feedback(
            st.session_state.current_session_id, message_index, rating,
            _previous_user_question(message_index), msg.get("content", ""),
            comment=comment, meta=msg.get("meta", {}),
            owner_id=st.session_state.owner_id or None,
        )
        st.toast("Feedback saved. Thank you.")
    except Exception as e:                                  # noqa: BLE001
        st.warning(SCHEMA_HELP if is_schema_error(e) else f"Could not save feedback: {e}")


def _render_feedback_controls(message_index):
    saved = st.session_state.answer_feedback.get(message_index)
    if saved:
        label = "Helpful" if saved.get("rating") == "up" else "Needs improvement"
        st.caption(f"Feedback: {label}")
        return
    cols = st.columns([1, 1, 4])
    if cols[0].button("Helpful", key=f"fb_up_{message_index}", use_container_width=True):
        _submit_feedback(message_index, "up")
        st.rerun()
    if cols[1].button("Improve", key=f"fb_down_{message_index}", use_container_width=True):
        st.session_state[f"fb_open_{message_index}"] = True
    if st.session_state.get(f"fb_open_{message_index}"):
        with st.form(f"fb_form_{message_index}", clear_on_submit=True):
            comment = st.text_area(
                "What should be improved?",
                placeholder="Example: answer more directly, include page references...",
                key=f"fb_comment_{message_index}",
            )
            submitted = st.form_submit_button("Save feedback")
            if submitted:
                _submit_feedback(message_index, "down", comment)
                st.session_state[f"fb_open_{message_index}"] = False
                st.rerun()


def render_feedback_summary():
    if not st.session_state.dev_mode:
        return
    try:
        stats = rag_db.load_feedback_summary()
    except Exception:                                       # noqa: BLE001
        return
    if not stats["total"]:
        return
    with st.expander("Answer feedback summary", expanded=False):
        cols = st.columns(4)
        cols[0].metric("Feedback", stats["total"])
        cols[1].metric("Helpful", stats["positive"])
        cols[2].metric("Improve", stats["negative"])
        cols[3].metric("Helpful rate", f"{stats['positive_rate'] * 100:.0f}%")


# ──────────────────────────────────────────────────────────────────────
#  Public surface
# ──────────────────────────────────────────────────────────────────────
def render_history() -> None:
    """Render all prior messages (the chat bubbles + their meta)."""
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🧠"):
                st.markdown(msg["content"])
                m = msg.get("meta") or {}
                if m:
                    _render_meta_chips(m)
                    cols = st.columns([1, 1, 1, 2])
                    with cols[3]:
                        st_copy_to_clipboard(msg["content"], key=f"copy_hist_{i}")
                    _render_dev_panel(m)
                    _render_feedback_controls(i)


def render_input_and_run() -> None:
    """The single chat_input + the per-turn pipeline run."""
    if user_input := st.chat_input("Ask about your documents…"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner(f"Searching with {st.session_state.embedding_model}…"):
                response_text, meta, _used = run_rag_pipeline(
                    user_input, st.session_state.embedding_model)
            st.markdown(response_text)
            _render_meta_chips(meta)
            cols = st.columns([1, 1, 1, 2])
            with cols[3]:
                st_copy_to_clipboard(response_text, key=f"copy_live_{len(st.session_state.messages)}")
            _render_dev_panel(meta)

        st.session_state.messages.append({"role": "assistant", "content": response_text, "meta": meta})
        st.session_state.total_input_tokens += meta.get("input_tokens", 0)
        st.session_state.total_output_tokens += meta.get("output_tokens", 0)
        st.session_state.total_queries += 1

        # Refresh the running summary, then persist.
        st.session_state.summary = conversation.maybe_update_summary(
            st.session_state.messages, st.session_state.summary)
        save_chat_history()
        time.sleep(0.3)
        st.rerun()
