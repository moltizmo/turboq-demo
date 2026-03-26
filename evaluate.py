"""IR evaluation metrics using real human-annotated relevance labels."""

import numpy as np


def precision_at_k(retrieved_ids: list[int], relevant_ids: list[int], k: int) -> float:
    retrieved_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for r in retrieved_k if r in relevant_set)
    return hits / k


def recall_at_k(retrieved_ids: list[int], relevant_ids: list[int], k: int) -> float:
    retrieved_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for r in retrieved_k if r in relevant_set)
    return hits / len(relevant_set) if relevant_set else 0.0


def mrr_at_k(retrieved_ids: list[int], relevant_ids: list[int], k: int) -> float:
    relevant_set = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids[:k], 1):
        if rid in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[int], relevant_ids: list[int], k: int) -> float:
    relevant_set = set(relevant_ids)
    dcg = sum(
        (1.0 / np.log2(rank + 1))
        for rank, rid in enumerate(retrieved_ids[:k], 1)
        if rid in relevant_set
    )
    idcg = sum(
        (1.0 / np.log2(rank + 1))
        for rank in range(1, min(len(relevant_set), k) + 1)
    )
    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(retrieved_ids: list[int], relevant_ids: list[int], k: int) -> float:
    relevant_set = set(relevant_ids)
    return 1.0 if any(r in relevant_set for r in retrieved_ids[:k]) else 0.0


def evaluate_pipeline(pipeline, query_data: list[dict], k: int = 10) -> dict:
    """Run full evaluation of a pipeline against query_data with real labels.

    Returns dict of averaged metrics.
    """
    metrics = {
        "precision@1": [], "precision@3": [], "precision@5": [], "precision@10": [],
        "recall@1": [], "recall@5": [], "recall@10": [],
        "mrr@10": [], "ndcg@10": [],
        "hit@1": [], "hit@5": [],
        "latencies_ms": [],
    }

    for qd in query_data:
        results, latency = pipeline.search_timed(qd["query"], k=k)
        retrieved_ids = [r[0] for r in results]
        rel = qd["relevant_ids"]

        metrics["precision@1"].append(precision_at_k(retrieved_ids, rel, 1))
        metrics["precision@3"].append(precision_at_k(retrieved_ids, rel, 3))
        metrics["precision@5"].append(precision_at_k(retrieved_ids, rel, 5))
        metrics["precision@10"].append(precision_at_k(retrieved_ids, rel, 10))
        metrics["recall@1"].append(recall_at_k(retrieved_ids, rel, 1))
        metrics["recall@5"].append(recall_at_k(retrieved_ids, rel, 5))
        metrics["recall@10"].append(recall_at_k(retrieved_ids, rel, 10))
        metrics["mrr@10"].append(mrr_at_k(retrieved_ids, rel, 10))
        metrics["ndcg@10"].append(ndcg_at_k(retrieved_ids, rel, 10))
        metrics["hit@1"].append(hit_at_k(retrieved_ids, rel, 1))
        metrics["hit@5"].append(hit_at_k(retrieved_ids, rel, 5))
        metrics["latencies_ms"].append(latency)

    avg = {}
    for key, vals in metrics.items():
        avg[key] = float(np.mean(vals))
    return avg
