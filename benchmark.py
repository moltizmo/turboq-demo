"""Full benchmark suite: evaluation + plots for TurboQuant RAG demo."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset

from pipeline_full import FullPrecisionPipeline
from pipeline_turboq import TurboQuantPipeline
from evaluate import evaluate_retrieval, print_summary


def setup_dark_theme():
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "axes.edgecolor": "#e94560",
        "axes.labelcolor": "#eee",
        "xtick.color": "#eee",
        "ytick.color": "#eee",
        "text.color": "#eee",
        "grid.color": "#333",
        "grid.alpha": 0.3,
        "figure.dpi": 150,
        "font.size": 11,
    })


def load_data(n_passages: int = 500, n_queries: int = 50):
    """Load SQuAD dataset and extract passages + queries."""
    print("Loading SQuAD dataset...")
    try:
        ds = load_dataset("squad", split="train")
    except Exception:
        print("SQuAD unavailable, trying BeIR/scifact...")
        ds = load_dataset("BeIR/scifact", "corpus", split="train")
        passages = [row["text"] for row in ds.select(range(min(n_passages, len(ds))))]
        queries = [p[:80] + "?" for p in passages[:n_queries]]
        return passages, queries

    # Deduplicate contexts
    seen = set()
    passages = []
    for row in ds:
        ctx = row["context"]
        if ctx not in seen and len(ctx) > 50:
            seen.add(ctx)
            passages.append(ctx)
            if len(passages) >= n_passages:
                break

    # Use questions as queries
    queries = [row["question"] for row in ds.select(range(n_queries))]
    print(f"Loaded {len(passages)} passages, {len(queries)} queries")
    return passages, queries


def plot_memory(metrics: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Full Precision\n(float32)", "TurboQuant\n(3-bit)"]
    values = [metrics["memory_full_mb"], metrics["memory_turboq_mb"]]
    colors = ["#e94560", "#0f3460"]
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="#eee", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f} MB", ha="center", va="bottom", fontweight="bold")
    ratio = metrics["memory_full_mb"] / metrics["memory_turboq_mb"]
    ax.set_title(f"Memory Usage ({ratio:.1f}x reduction)", fontweight="bold")
    ax.set_ylabel("Memory (MB)")
    fig.savefig(os.path.join(out_dir, "memory_comparison.png"), bbox_inches="tight")
    plt.close(fig)


def plot_latency(metrics: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Full Precision", "TurboQuant"]
    values = [metrics["latency_full_ms"], metrics["latency_turboq_ms"]]
    colors = ["#e94560", "#0f3460"]
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="#eee", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f} ms", ha="center", va="bottom", fontweight="bold")
    ratio = metrics["latency_full_ms"] / metrics["latency_turboq_ms"]
    ax.set_title(f"Query Latency ({ratio:.2f}x ratio)", fontweight="bold")
    ax.set_ylabel("Latency (ms)")
    fig.savefig(os.path.join(out_dir, "latency_comparison.png"), bbox_inches="tight")
    plt.close(fig)


def plot_precision_at_k(metrics: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ks = list(range(1, 11))
    full_prec = [metrics["precision_full"][k] for k in ks]
    turboq_prec = [metrics["precision_turboq"][k] for k in ks]
    ax.plot(ks, full_prec, "o-", color="#e94560", label="Full Precision", linewidth=2)
    ax.plot(ks, turboq_prec, "s--", color="#00d2ff", label="TurboQuant (3-bit)", linewidth=2)
    ax.set_xlabel("k")
    ax.set_ylabel("Precision@k")
    ax.set_title("Precision@k: Full vs TurboQuant", fontweight="bold")
    ax.set_xticks(ks)
    ax.legend()
    ax.grid(True)
    fig.savefig(os.path.join(out_dir, "precision_at_k.png"), bbox_inches="tight")
    plt.close(fig)


def plot_score_correlation(metrics: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    full_s = metrics["full_scores"]
    turboq_s = metrics["turboq_scores"]
    # Filter out NaN values
    mask = np.isfinite(full_s) & np.isfinite(turboq_s)
    full_s, turboq_s = full_s[mask], turboq_s[mask]
    ax.scatter(full_s, turboq_s,
               alpha=0.6, color="#00d2ff", s=30, edgecolors="#eee", linewidth=0.3)
    if len(full_s) > 0:
        mn = min(full_s.min(), turboq_s.min()) - 0.02
        mx = max(full_s.max(), turboq_s.max()) + 0.02
        ax.plot([mn, mx], [mn, mx], "--", color="#e94560", alpha=0.7, label="y=x")
        ax.set_xlim(mn, mx)
        ax.set_ylim(mn, mx)
    ax.set_xlabel("Full Precision Score")
    ax.set_ylabel("TurboQuant Score")
    ax.set_title("Top-1 Score Correlation", fontweight="bold")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "score_correlation.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    setup_dark_theme()
    out_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    passages, queries = load_data()

    # Build full-precision pipeline
    full = FullPrecisionPipeline()
    full.build_index(passages)

    # Build TurboQuant pipeline (reuse embeddings from full pipeline)
    turboq = TurboQuantPipeline(n_bits=3)
    turboq.build_index(passages, full.embeddings)

    # Evaluate
    metrics = evaluate_retrieval(full, turboq, queries, full.encode_query, max_k=10)

    # Print summary
    print_summary(metrics)

    # Generate plots
    print("Generating plots...")
    plot_memory(metrics, out_dir)
    plot_latency(metrics, out_dir)
    plot_precision_at_k(metrics, out_dir)
    plot_score_correlation(metrics, out_dir)
    print(f"Plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
