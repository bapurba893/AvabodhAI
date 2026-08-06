"""
main.py
-------
Entry point — run from terminal or call from your API/backend.

Usage:
    python cli.py --file path/to/doc.pdf --tenant-id t1 --org-unit-id d1
    python cli.py --dir path/to/folder --tenant-id t1 --org-unit-id d1
    python cli.py --check-db
"""
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'
import argparse
import sys

from db.database import init_db, check_db_connection, backfill_tsvector
from pipeline.orchestrator import process_single_document, process_directory
from utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Avabodh — Document summarisation pipeline"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file", type=str, help="Process a single document")
    group.add_argument("--dir",  type=str, help="Process all documents in a folder")
    group.add_argument("--check-db", action="store_true", help="Test DB connection")
    group.add_argument(
        "--backfill-tsvector", action="store_true",
        help="One-time catch-up: populate chunk_tsvector for rows that predate the full-text-search trigger. "
             "New rows are already covered automatically — only needed after first adding FTS to an existing database.",
    )
    parser.add_argument(
        "--tenant-id", type=str, default=None,
        help="Tenant to attribute processed documents to. Required unless --check-db or --backfill-tsvector.",
    )
    parser.add_argument(
        "--org-unit-id", type=str, default=None,
        help="Department/org unit within that tenant. Required unless --check-db or --backfill-tsvector.",
    )
    parser.add_argument(
        "--ground-truth", action="store_true",
        help="Mark processed documents as ground truth (retrievable by chat/search). "
             "Default is False, matching the API upload default — omit this flag to "
             "load documents that won't be retrievable until explicitly marked later.",
    )

    args = parser.parse_args()

    # ── Health check ──────────────────────────────────────────────────────
    if args.check_db:
        ok = check_db_connection()
        print("DB connection: OK" if ok else "DB connection: FAILED")
        sys.exit(0 if ok else 1)

    # ── Backfill (no tenant/org-unit needed — it fixes every row) ───────────
    if args.backfill_tsvector:
        init_db()  # make sure the trigger/column actually exist first
        total = backfill_tsvector()
        print(f"✓ Backfilled chunk_tsvector on {total} row(s).")
        sys.exit(0)

    # This is a standalone CLI tool — no gateway headers to read tenant_id
    # or org_unit_id from, so both must be passed explicitly. Refusing to
    # default either to something like "default" is deliberate: a
    # forgotten flag should fail loudly, not silently file documents
    # under the wrong tenant or department.
    missing = []
    if not args.tenant_id:
        missing.append("--tenant-id")
    if not args.org_unit_id:
        missing.append("--org-unit-id")
    if missing:
        print(f"Error: {' and '.join(missing)} required (unless using --check-db)")
        sys.exit(1)

    # ── Ensure tables exist ───────────────────────────────────────────────
    init_db()

    # ── Single file ───────────────────────────────────────────────────────
    if args.file:
        result = process_single_document(args.file, args.tenant_id, args.org_unit_id, args.ground_truth)
        if result.success and not result.skipped:
            print(f"✓ Summary saved — ID: {result.summary_id}")
        elif result.skipped:
            print(f"↩ Skipped — already summarised (ID: {result.summary_id})")
        else:
            print(f"✗ Failed: {result.error}")
            sys.exit(1)

    # ── Directory ─────────────────────────────────────────────────────────
    else:
        report = process_directory(args.tenant_id, args.org_unit_id, args.dir, args.ground_truth)
        print(
            f"\nDone — {report.succeeded} saved | "
            f"{report.skipped} skipped | {report.failed} failed"
        )
        if report.failed > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()