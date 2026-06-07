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
    url = os.getenv("project_url")
    # Prefer a narrowly-scoped anon key if provided; fall back to service key.
    key = os.getenv("service_key")
    if url and key:
        return create_client(url, key)
    return None
