"""
Copies the local Postgres database to Supabase: exact schema (via pg_dump),
all rows, keys/indexes, and sequence positions, then checks row counts.

  python scripts/migrate_db_to_supabase.py

Source: the local DB from .env (DB_HOST, DB_USER, ...).
Target: deploy/.env.production (DB_HOST, DB_USER, DB_PASSWORD, ...). If that
points at the pooler's transaction port 6543, the copy uses its session port
5432 instead.
Refuses to run if the target already has tables, so it never overwrites data.
Set PG_BIN if pg_dump isn't at E:\\PostgreSQL\\bin.
"""
import os
import subprocess
import sys
import tempfile

import psycopg2
from dotenv import dotenv_values, load_dotenv

load_dotenv()

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROD_ENV = os.path.join(ROOT_DIR, "deploy", ".env.production")
PG_DUMP = os.path.join(os.environ.get("PG_BIN", r"E:\PostgreSQL\bin"), "pg_dump.exe" if os.name == "nt" else "pg_dump")
# Keep up to this much of a table in memory while copying, spill to disk beyond.
SPOOL_BYTES = 200 * 1024 * 1024


def local_params():
    return {
        "host": os.getenv("DB_HOST", "localhost"), "port": os.getenv("DB_PORT", "5432"),
        "user": os.getenv("DB_USER", "postgres"), "password": os.getenv("DB_PASSWORD", ""),
        "dbname": os.getenv("DB_NAME", "psx_predictor"),
    }


def target_params():
    if not os.path.exists(PROD_ENV):
        sys.exit(f"Missing {PROD_ENV} - copy deploy/env.production.example and fill it in.")
    env = dotenv_values(PROD_ENV)
    missing = [k for k in ("DB_HOST", "DB_USER", "DB_PASSWORD") if not env.get(k)]
    if missing:
        sys.exit(f"deploy/.env.production is missing: {', '.join(missing)}")
    # 6543 is the pooler's transaction mode, which the API uses; the bulk copy
    # runs on the same pooler's session mode (5432) instead.
    port = env.get("DB_PORT") or "5432"
    return {
        "host": env["DB_HOST"], "port": "5432" if port == "6543" else port,
        "user": env["DB_USER"], "password": env["DB_PASSWORD"],
        "dbname": env.get("DB_NAME") or "postgres", "sslmode": env.get("DB_SSLMODE") or "require",
    }


def dump_section(section, src):
    """Schema SQL for one pg_dump section, minus what the target can't run:
    psql meta-commands (\\restrict, added by pg_dump 18), settings newer than
    the target's Postgres version, and statements about the public schema
    itself - every database already has it, and Supabase owns it."""
    env = {**os.environ, "PGPASSWORD": src["password"]}
    cmd = [PG_DUMP, "--schema-only", "--no-owner", "--no-privileges", "--schema=public",
           f"--section={section}", "-h", src["host"], "-p", str(src["port"]), "-U", src["user"], src["dbname"]]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True).stdout
    skip = ("\\", "SET transaction_timeout", "CREATE SCHEMA public", "COMMENT ON SCHEMA public")
    return "\n".join(l for l in out.splitlines() if not l.startswith(skip))


def table_names(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename")
        return [r[0] for r in cur.fetchall()]


def row_count(conn, table):
    with conn.cursor() as cur:
        cur.execute(f'SELECT COUNT(*) FROM public."{table}"')
        return cur.fetchone()[0]


def main():
    if not os.path.exists(PG_DUMP):
        sys.exit(f"pg_dump not found at {PG_DUMP} - set PG_BIN to your Postgres bin folder.")
    src_params, dst_params = local_params(), target_params()
    src = psycopg2.connect(**src_params)
    dst = psycopg2.connect(**dst_params)

    if table_names(dst):
        sys.exit(f"Target already has tables {table_names(dst)} - refusing to overwrite. Use an empty project.")
    tables = table_names(src)
    print(f"Copying {len(tables)} tables from {src_params['host']} to {dst_params['host']}")

    with dst.cursor() as cur:
        cur.execute(dump_section("pre-data", src_params))
    dst.commit()
    print("Created tables.")

    for t in tables:
        with tempfile.SpooledTemporaryFile(max_size=SPOOL_BYTES, mode="w+b") as buf:
            with src.cursor() as cur:
                cur.copy_expert(f'COPY public."{t}" TO STDOUT WITH (FORMAT csv)', buf)
            buf.seek(0)
            with dst.cursor() as cur:
                cur.copy_expert(f'COPY public."{t}" FROM STDIN WITH (FORMAT csv)', buf)
        dst.commit()
        print(f"  {t}: {row_count(dst, t):,} rows")

    with dst.cursor() as cur:
        cur.execute(dump_section("post-data", src_params))
    dst.commit()
    print("Added keys, constraints and indexes.")

    with src.cursor() as cur:
        cur.execute("SELECT schemaname, sequencename, last_value FROM pg_sequences "
                    "WHERE schemaname = 'public' AND last_value IS NOT NULL")
        sequences = cur.fetchall()
    with dst.cursor() as cur:
        for schema, seq, last in sequences:
            cur.execute("SELECT setval(%s, %s, true)", (f'{schema}."{seq}"', last))
        cur.execute("ANALYZE")
    dst.commit()
    print(f"Set {len(sequences)} sequence positions.")

    mismatches = [(t, row_count(src, t), row_count(dst, t)) for t in tables]
    mismatches = [m for m in mismatches if m[1] != m[2]]
    if mismatches:
        sys.exit(f"Row count mismatch (table, local, supabase): {mismatches}")
    print(f"Done - all {len(tables)} tables match row-for-row.")


if __name__ == "__main__":
    main()
