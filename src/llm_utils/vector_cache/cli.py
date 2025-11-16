#!/usr/bin/env python3
"""Command-line interface for embed_cache package."""

import argparse
import json
import sys
from pathlib import Path
from typing import List

from llm_utils.vector_cache import VectorCache, estimate_cache_size, validate_model_name


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Command-line interface for embed_cache package"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Embed command
    embed_parser = subparsers.add_parser("embed", help="Generate embeddings for texts")
    embed_parser.add_argument("model", help="Model name or API URL")
    embed_parser.add_argument("--texts", nargs="+", help="Texts to embed")
    embed_parser.add_argument("--file", help="File containing texts (one per line)")
    embed_parser.add_argument("--output", help="Output file for embeddings (JSON)")
    embed_parser.add_argument(
        "--cache-db", default="embed_cache.sqlite", help="Cache database path"
    )
    embed_parser.add_argument(
        "--backend",
        choices=["vllm", "transformers", "openai", "auto"],
        default="auto",
        help="Backend to use",
    )
    embed_parser.add_argument(
        "--gpu-memory",
        type=float,
        default=0.5,
        help="GPU memory utilization for vLLM (0.0-1.0)",
    )
    embed_parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for transformers"
    )
    embed_parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )

    # Cache stats command
    stats_parser = subparsers.add_parser("stats", help="Show cache statistics")
    stats_parser.add_argument(
        "--cache-db", default="embed_cache.sqlite", help="Cache database path"
    )

    # Clear cache command
    clear_parser = subparsers.add_parser("clear", help="Clear cache")
    clear_parser.add_argument(
        "--cache-db", default="embed_cache.sqlite", help="Cache database path"
    )
    clear_parser.add_argument(
        "--confirm", action="store_true", help="Skip confirmation prompt"
    )

    # Validate model command
    validate_parser = subparsers.add_parser("validate", help="Validate model name")
    validate_parser.add_argument("model", help="Model name to validate")

    # Estimate command
    estimate_parser = subparsers.add_parser("estimate", help="Estimate cache size")
    estimate_parser.add_argument("num_texts", type=int, help="Number of texts")
    estimate_parser.add_argument(
        "--embed-dim", type=int, default=1024, help="Embedding dimension"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "embed":
            handle_embed(args)
        elif args.command == "stats":
            handle_stats(args)
        elif args.command == "clear":
            handle_clear(args)
        elif args.command == "validate":
            handle_validate(args)
        elif args.command == "estimate":
            handle_estimate(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def handle_embed(args):
    """Handle embed command."""
    # Get texts
    texts = []
    if args.texts:
        texts.extend(args.texts)

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {args.file}")

        with open(file_path, encoding="utf-8") as f:
            texts.extend([line.strip() for line in f if line.strip()])

    if not texts:
        raise ValueError("No texts provided. Use --texts or --file")

    print(f"Embedding {len(texts)} texts using model: {args.model}")

    # Initialize cache and get embeddings
    cache = VectorCache(args.model, db_path=args.cache_db)
    embeddings = cache.embeds(texts)

    print(f"Generated embeddings with shape: {embeddings.shape}")

    # Output results
    if args.output:
        output_data = {
            "texts": texts,
            "embeddings": embeddings.tolist(),
            "shape": list(embeddings.shape),
            "model": args.model,
        }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        print(f"Results saved to: {args.output}")
    else:
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Sample embedding (first 5 dims): {embeddings[0][:5].tolist()}")


def handle_stats(args):
    """Handle stats command."""
    cache_path = Path(args.cache_db)
    if not cache_path.exists():
        print(f"Cache database not found: {args.cache_db}")
        return

    cache = VectorCache("dummy", db_path=args.cache_db)
    stats = cache.get_cache_stats()

    print("Cache Statistics:")
    print(f"  Database: {args.cache_db}")
    print(f"  Total cached embeddings: {stats['total_cached']}")
    print(f"  Database size: {cache_path.stat().st_size / (1024 * 1024):.2f} MB")


def handle_clear(args):
    """Handle clear command."""
    cache_path = Path(args.cache_db)
    if not cache_path.exists():
        print(f"Cache database not found: {args.cache_db}")
        return

    if not args.confirm:
        response = input(
            f"Are you sure you want to clear cache at {args.cache_db}? [y/N]: "
        )
        if response.lower() != "y":
            print("Cancelled.")
            return

    cache = VectorCache("dummy", db_path=args.cache_db)
    stats_before = cache.get_cache_stats()
    cache.clear_cache()

    print(
        f"Cleared {stats_before['total_cached']} cached embeddings from {args.cache_db}"
    )


def handle_validate(args):
    """Handle validate command."""
    is_valid = validate_model_name(args.model)
    if is_valid:
        print(f"✓ Valid model: {args.model}")
    else:
        print(f"✗ Invalid model: {args.model}")
        sys.exit(1)


def handle_estimate(args):
    """Handle estimate command."""
    size_estimate = estimate_cache_size(args.num_texts, args.embed_dim)
    print(
        f"Estimated cache size for {args.num_texts} texts ({args.embed_dim}D embeddings): {size_estimate}"
    )


if __name__ == "__main__":
    main()
