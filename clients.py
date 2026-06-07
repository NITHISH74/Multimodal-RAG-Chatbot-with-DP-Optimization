"""
Cached client factories for Gemini, Cohere and Supabase.

Centralised here so every module shares one cached instance. All secrets
are read from the environment / Streamlit secrets — never hardcoded
(Phase 15).
"""
import os
import streamlit as st
from google import genai
import cohere
from supabase import create_client, Client


@st.cache_resource
def get_gemini_client():
    key = os.getenv("gemini_api_key")
    if not key:
        return None
    return genai.Client(api_key=key)


@st.cache_resource
def get_cohere_client():
    key = os.getenv("cohere_api_key")
    if not key:
        return None
    return cohere.ClientV2(api_key=key)


@st.cache_resource
def get_supabase_client() -> Client:
    """Privileged (service_role) client — used for all writes / storage.
    Runs server-side only (Phase 15)."""
    url = os.getenv("project_url")
    key = os.getenv("service_key")
    if url and key:
        return create_client(url, key)
    return None


@st.cache_resource
def get_supabase_anon_client() -> Client:
    """Least-privilege (anon) client for read paths. Returns None if no
    anon key is configured, in which case callers fall back to the service
    client (Phase 15). With RLS enabled (migration 0005) the anon key can
    only READ."""
    url = os.getenv("project_url")
    key = os.getenv("anon_key")
    if url and key:
        return create_client(url, key)
    return None
