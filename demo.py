"""Interactive CLI demo for TurboQuant RAG pipeline comparison."""

from data_loader import load_msmarco
from pipeline_full import FullPrecisionPipeline
from pipeline_turboq import TurboQPipeline


def main():
    print("=" * 60)
    print("  TurboQuant RAG Demo — MS MARCO v2.1")
    print("  Full Precision vs 4-bit Compressed Retrieval")
    print("=" * 60)

    corpus, query_data = load_msmarco(n_queries=200)

    full = FullPrecisionPipeline()
    turboq = TurboQPipeline()

    full.encode_corpus(corpus)
    turboq.compress(full.embeddings.copy())

    full_mem = full.memory_mb()
    turboq_mem = turboq.memory_mb()

    print(f"\nReady! Corpus: {len(corpus)} passages from MS MARCO v2.1")
    print("Type a query or 'q' to quit.\n")

    while True:
        try:
            query = input("Enter query (or 'q' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query or query.lower() in ("q", "quit", "exit"):
            print("Bye!")
            break

        full_results, full_lat = full.search_timed(query, k=5)
        turboq_results, turboq_lat = turboq.search_timed(query, k=5)

        print(f"\n{'='*60}")
        print("=== FULL PRECISION RESULTS ===")
        for rank, (pid, score) in enumerate(full_results, 1):
            text = corpus[pid][:120].replace("\n", " ")
            print(f"#{rank} [Score: {score:.3f}] {text}...")

        print(f"\n=== TURBOQ (4-bit compressed) ===")
        for rank, (pid, score) in enumerate(turboq_results, 1):
            text = corpus[pid][:120].replace("\n", " ")
            print(f"#{rank} [Score: {score:.3f}] {text}...")

        reduction = full_mem / turboq_mem if turboq_mem > 0 else 0
        speedup = full_lat / turboq_lat if turboq_lat > 0 else 0
        print(f"\n=== STATS ===")
        print(f"Memory:  Full={full_mem:.1f} MB  TurboQ={turboq_mem:.1f} MB  "
              f"Reduction={reduction:.1f}x")
        print(f"Latency: Full={full_lat:.1f}ms    TurboQ={turboq_lat:.1f}ms   "
              f"Speedup={speedup:.1f}x")
        print()


if __name__ == "__main__":
    main()
