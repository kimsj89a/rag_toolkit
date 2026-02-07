"""
RAG Toolkit - Basic Usage Example

Prerequisites:
    1. pip install -e .  (from rag_toolkit/ directory)
    2. Set RAG_LLM_API_KEY in .env or environment variable
    3. Install parser: pip install "rag-toolkit[mineru]"

Usage:
    python examples/basic_usage.py --input path/to/document.pdf
"""

import asyncio
import argparse
from pathlib import Path


async def main(input_path: str):
    from rag_toolkit import RAGClient, RAGConfig, RAG_PAPER_QUERIES, RAG_QUERY_MODES

    # ---- 1. Configuration ----
    # Option A: From environment variables (reads .env automatically)
    config = RAGConfig.from_env()

    # Option B: With custom paths
    # config = RAGConfig.with_paths(
    #     storage_dir="./my_rag_storage",
    #     output_dir="./my_rag_output",
    # )

    # Option C: Full manual configuration
    # from rag_toolkit.config import APIConfig, StorageConfig
    # config = RAGConfig(
    #     api=APIConfig(llm_api_key="sk-...", llm_model="gpt-4o-mini"),
    #     storage=StorageConfig(storage_dir="./storage", output_dir="./output"),
    # )

    # ---- 2. Index documents ----
    async with RAGClient(config=config) as rag:
        print(f"Indexing: {input_path}")
        result = await rag.index(input_path)
        print(f"Index result: {result}")

        # ---- 3. Single query ----
        answer = await rag.query(
            "What is the main topic of this document?",
            mode="mix",
        )
        print(f"\nAnswer: {answer}")

        # ---- 4. Batch query ----
        questions = [
            "What problem does this document address?",
            "What method or approach is proposed?",
            "What are the main results?",
        ]
        results = await rag.batch_query(questions, mode="mix")
        for r in results:
            print(f"\nQ: {r['query']}")
            print(f"A: {r['answer'][:200]}...")

        # ---- 5. Category-based batch query (for papers) ----
        category_results = await rag.batch_query_by_category(
            queries_by_category=RAG_PAPER_QUERIES,
            modes_by_category=RAG_QUERY_MODES,
        )
        for category, results in category_results.items():
            print(f"\n=== {category} ===")
            for r in results:
                if r["success"]:
                    print(f"  Q: {r['query'][:80]}...")
                    print(f"  A: {r['answer'][:150]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Toolkit Example")
    parser.add_argument("--input", "-i", required=True, help="Path to document")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: File not found: {args.input}")
        exit(1)

    asyncio.run(main(args.input))
