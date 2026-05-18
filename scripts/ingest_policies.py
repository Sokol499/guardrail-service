#!/usr/bin/env python3
"""Ingest policy chunks into ChromaDB vector store."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.logging import setup_logging, get_logger
from app.moderation.cloud.rag import PolicyRAG

logger = get_logger(__name__)


def main(force: bool = False) -> None:
    setup_logging()
    rag = PolicyRAG()
    count = rag.ingest_policies(force=force)
    logger.info("ingestion_complete", policy_count=count)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest policies into ChromaDB")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even if collection already has data",
    )
    args = parser.parse_args()
    main(force=args.force)
