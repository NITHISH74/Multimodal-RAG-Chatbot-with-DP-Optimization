"""
In-app database bootstrap.

Lets the app apply its own schema (the idempotent script in
db/migrations/RUN_THIS_IN_SUPABASE.sql) via a direct Postgres connection,
so users don't have to paste SQL into the Supabase SQL Editor.

Connection note (IMPORTANT): Streamlit Cloud is IPv4-only and Supabase's
direct connection is IPv6-only, so config.SUPABASE_DB_URL must be the
**Session Pooler** URI (port 5432, IPv4). SSL is required.
"""
import os

import config
from clients import get_supabase_client

MIGRATION_SQL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "db", "migrations", "RUN_THIS_IN_SUPABASE.sql",
)


def load_migration_sql():
    """Return the consolidated, idempotent schema script (single source of truth)."""
    with open(MIGRATION_SQL_PATH, "r", encoding="utf-8") as f:
        return f.read()


def run_migrations(db_url=None):
    """Connect directly to Postgres and apply the schema script.

    Raises ValueError if no connection string is configured, or the
    underlying psycopg2 error on connection/SQL failure (so the UI can show
    exactly what went wrong — pooler/SSL/password mistakes are common).
    Returns a short success summary string.
    """
    db_url = db_url or config.SUPABASE_DB_URL
    if not db_url:
        raise ValueError(
            "No database connection string configured. Add the Supabase "
            "Session Pooler URI as the 'supabase_db_url' secret."
        )

    import psycopg2  # imported lazily so the rest of the app works without it

    sql = load_migration_sql()
    conn = psycopg2.connect(db_url, sslmode="require", connect_timeout=15)
    try:
        conn.autocommit = True  # DDL statements each commit independently
        with conn.cursor() as cur:
            cur.execute(sql)  # script has no %-placeholders; runs as one batch
    finally:
        conn.close()
    return "Schema applied successfully."


def schema_ready(supabase=None):
    """True if the V3 schema is present (documents.content_hash is queryable)."""
    supabase = supabase or get_supabase_client()
    if supabase is None:
        return False
    try:
        supabase.table("documents").select("content_hash").limit(1).execute()
        return True
    except Exception:
        return False
