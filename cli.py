"""
main.py
-------
Entry point — run from terminal or call from your API/backend.

Usage:
    python main.py                          # process all docs in DOCUMENTS_DIR
    python main.py --file path/to/doc.pdf   # process one file
    python main.py --dir path/to/folder     # process a specific folder
    python main.py --check-db               # test DB connection only
"""
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'
import argparse
import sys

from db.database import init_db, check_db_connection
from pipeline.orchestrator import process_single_document, process_directory
from utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ababoth — Document summarisation pipeline"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file", type=str, help="Process a single document")
    group.add_argument("--dir",  type=str, help="Process all documents in a folder")
    group.add_argument("--check-db", action="store_true", help="Test DB connection")

    args = parser.parse_args()

    # ── Health check ──────────────────────────────────────────────────────
    if args.check_db:
        ok = check_db_connection()
        print("DB connection: OK" if ok else "DB connection: FAILED")
        sys.exit(0 if ok else 1)

    # ── Ensure tables exist ───────────────────────────────────────────────
    init_db()

    # ── Single file ───────────────────────────────────────────────────────
    if args.file:
        result = process_single_document(args.file)
        if result.success and not result.skipped:
            print(f"✓ Summary saved — ID: {result.summary_id}")
        elif result.skipped:
            print(f"↩ Skipped — already summarised (ID: {result.summary_id})")
        else:
            print(f"✗ Failed: {result.error}")
            sys.exit(1)

    # ── Directory ─────────────────────────────────────────────────────────
    else:
        report = process_directory(args.dir)
        print(
            f"\nDone — {report.succeeded} saved | "
            f"{report.skipped} skipped | {report.failed} failed"
        )
        if report.failed > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
