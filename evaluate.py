"""Evaluation metrics for comparing full-precision and TurboQuant pipelines."""

import time
import numpy as np
from tabulate import tabulate


def precision_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Fraction of top-k retrieved that are relevant."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & relevant) / k


def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Fraction of relevant items found in top-k."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    return len(set(top_k) & relevant) / len(relevant)


def evaluate_retrieval(full_pipeline, turboq_pipeline, queries: list[str],
                       encode_fn, max_k: int = 10):
    """Run evaluation across all queries, comparing both pipelines.

    Args:
        full_pipeline: FullPrecisionPipeline instance
        turboq_pipeline: TurboQuantPipeline instance
        queries: list of query strings
        encode_fn: function that encodes a query string to embedding
        max_k: maximum k for precision/recall computation

    Returns:
        dict with metrics
    """
    precisions_full = {k: [] for k in range(1, max_k + 1)}
    precisions_turboq = {k: [] for k in range(1, max_k + 1)}
    recalls_full = {k: [] for k in range(1, max_k + 1)}
    recalls_turboq = {k: [] for k in range(1, max_k + 1)}

    full_latencies = []
    turboq_latencies = []

    full_scores_all = []
    turboq_scores_all = []

    print(f"Evaluating {len(queries)} queries...")
    for query in queries:
        q_emb = encode_fn(query)

        # Full precision
        t0 = time.perf_counter()
        full_results = full_pipeline.query_embedding(q_emb, top_k=max_k)
        full_latencies.append(time.perf_counter() - t0)

        # TurboQuant
        t0 = time.perf_counter()
        turboq_results = turboq_pipeline.query(q_emb, top_k=max_k)
        turboq_latencies.append(time.perf_counter() - t0)

        # Ground truth = full precision top-5
        relevant = {r["index"] for r in full_results[:5]}
        full_retrieved = [r["index"] for r in full_results]
        turboq_retrieved = [r["index"] for r in turboq_results]

        for k in range(1, max_k + 1):
            precisions_full[k].append(precision_at_k(full_retrieved, relevant, k))
            precisions_turboq[k].append(precision_at_k(turboq_retrieved, relevant, k))
            recalls_full[k].append(recall_at_k(full_retrieved, relevant, k))
            recalls_turboq[k].append(recall_at_k(turboq_retrieved, relevant, k))

        # Collect top-1 scores for scatter plot
        full_scores_all.append(full_results[0]["score"])
        turboq_scores_all.append(turboq_results[0]["score"])

    # Aggregate
    metrics = {
        "precision_full": {k: np.mean(v) for k, v in precisions_full.items()},
        "precision_turboq": {k: np.mean(v) for k, v in precisions_turboq.items()},
        "recall_full": {k: np.mean(v) for k, v in recalls_full.items()},
        "recall_turboq": {k: np.mean(v) for k, v in recalls_turboq.items()},
        "latency_full_ms": np.mean(full_latencies) * 1000,
        "latency_turboq_ms": np.mean(turboq_latencies) * 1000,
        "memory_full_mb": full_pipeline.memory_mb(),
        "memory_turboq_mb": turboq_pipeline.memory_mb(),
        "full_scores": np.array(full_scores_all),
        "turboq_scores": np.array(turboq_scores_all),
    }
    return metrics


def print_summary(metrics: dict):
    """Print a formatted summary table."""
    # Precision / Recall table
    rows = []
    for k in [1, 3, 5]:
        rows.append([
            f"@{k}",
            f"{metrics['precision_full'][k]:.3f}",
            f"{metrics['precision_turboq'][k]:.3f}",
            f"{metrics['recall_full'][k]:.3f}",
            f"{metrics['recall_turboq'][k]:.3f}",
        ])

    print("\n" + "=" * 60)
    print("RETRIEVAL ACCURACY")
    print("=" * 60)
    print(tabulate(rows, headers=["k", "Prec(Full)", "Prec(TurboQ)",
                                   "Recall(Full)", "Recall(TurboQ)"],
                   tablefmt="grid"))

    # Performance table
    mem_reduction = metrics["memory_full_mb"] / metrics["memory_turboq_mb"]
    latency_ratio = metrics["latency_full_ms"] / metrics["latency_turboq_ms"]

    perf_rows = [
        ["Memory (MB)", f"{metrics['memory_full_mb']:.2f}",
         f"{metrics['memory_turboq_mb']:.2f}", f"{mem_reduction:.1f}x"],
        ["Latency (ms)", f"{metrics['latency_full_ms']:.2f}",
         f"{metrics['latency_turboq_ms']:.2f}", f"{latency_ratio:.2f}x"],
    ]
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    print(tabulate(perf_rows, headers=["Metric", "Full", "TurboQ", "Ratio"],
                   tablefmt="grid"))
    print()
