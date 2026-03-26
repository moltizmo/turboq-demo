"""Interactive CLI demo: compare Full Precision vs TurboQuant retrieval."""

import time
import sys
from benchmark import load_data
from pipeline_full import FullPrecisionPipeline
from pipeline_turboq import TurboQuantPipeline


def truncate(text: str, max_len: int = 100) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text


def main():
    print("=" * 60)
    print("  TurboQuant RAG Demo")
    print("  Full Precision vs 3-bit Compressed Retrieval")
    print("=" * 60)
    print()

    passages, _ = load_data()

    print("\nBuilding full-precision index...")
    full = FullPrecisionPipeline()
    full.build_index(passages)

    print("Building TurboQuant index...")
    turboq = TurboQuantPipeline(n_bits=3)
    turboq.build_index(passages, full.embeddings)

    mem_full = full.memory_mb()
    mem_turboq = turboq.memory_mb()

    print(f"\nReady! ({len(passages)} passages indexed)")
    print("Type a query and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            query = input("Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        # Encode query once
        q_emb = full.encode_query(query)

        # Full precision retrieval
        t0 = time.perf_counter()
        full_results = full.query_embedding(q_emb, top_k=5)
        full_ms = (time.perf_counter() - t0) * 1000

        # TurboQuant retrieval
        t0 = time.perf_counter()
        turboq_results = turboq.query(q_emb, top_k=5)
        turboq_ms = (time.perf_counter() - t0) * 1000

        print(f"\n{'='*60}")
        print("=== FULL PRECISION RESULTS ===")
        for i, r in enumerate(full_results, 1):
            print(f"  {i}. [Score: {r['score']:.3f}] {truncate(r['passage'])}")

        print(f"\n=== TURBOQ RESULTS (3-bit compressed) ===")
        for i, r in enumerate(turboq_results, 1):
            print(f"  {i}. [Score: {r['score']:.3f}] {truncate(r['passage'])}")

        reduction = mem_full / mem_turboq if mem_turboq > 0 else float("inf")
        speedup = full_ms / turboq_ms if turboq_ms > 0 else float("inf")

        print(f"\n=== PERFORMANCE COMPARISON ===")
        print(f"  Memory:  Full={mem_full:.2f} MB  TurboQ={mem_turboq:.2f} MB  Reduction={reduction:.1f}x")
        print(f"  Latency: Full={full_ms:.1f}ms   TurboQ={turboq_ms:.1f}ms    Speedup={speedup:.1f}x")
        print()


if __name__ == "__main__":
    main()
