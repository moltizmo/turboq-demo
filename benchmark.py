"""Full benchmark: MS MARCO v2.1 with real human-annotated relevance labels."""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate

from data_loader import load_msmarco
from pipeline_full import FullPrecisionPipeline
from pipeline_turboq import TurboQPipeline
from evaluate import evaluate_pipeline


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


def run_benchmark():
    setup_dark_theme()
    out_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(out_dir, exist_ok=True)

    # Load dataset
    corpus, query_data = load_msmarco(n_queries=200)

    # Initialize pipelines
    full = FullPrecisionPipeline()
    turboq = TurboQPipeline()

    # Encode corpus (full pipeline)
    full.encode_corpus(corpus)

    # Reuse embeddings for turboq to avoid double encoding
    turboq.compress(full.embeddings.copy())

    # Memory
    full_mem = full.memory_mb()
    turboq_mem = turboq.memory_mb()
    print(f"\nMemory: Full={full_mem:.2f} MB, TurboQ={turboq_mem:.2f} MB, "
          f"Reduction={full_mem / turboq_mem:.1f}x")

    # Evaluate both pipelines
    print(f"\nEvaluating Full Precision on {len(query_data)} queries...")
    full_metrics = evaluate_pipeline(full, query_data, k=10)

    print(f"Evaluating TurboQ on {len(query_data)} queries...")
    turboq_metrics = evaluate_pipeline(turboq, query_data, k=10)

    # Results table
    metric_keys = [
        "precision@1", "precision@3", "precision@5", "precision@10",
        "recall@1", "recall@5", "recall@10",
        "mrr@10", "ndcg@10", "hit@1", "hit@5",
    ]
    table = []
    for key in metric_keys:
        table.append([key, f"{full_metrics[key]:.4f}", f"{turboq_metrics[key]:.4f}",
                       f"{turboq_metrics[key] - full_metrics[key]:+.4f}"])
    table.append(["latency (ms/query)", f"{full_metrics['latencies_ms']:.2f}",
                   f"{turboq_metrics['latencies_ms']:.2f}", ""])
    table.append(["memory (MB)", f"{full_mem:.2f}", f"{turboq_mem:.2f}",
                   f"{turboq_mem / full_mem:.1%}"])

    print("\n" + tabulate(table, headers=["Metric", "Full", "TurboQ", "Delta"],
                          tablefmt="github"))

    # Save results
    results = {
        "corpus_size": len(corpus),
        "n_queries": len(query_data),
        "full_precision": full_metrics,
        "turboq": turboq_metrics,
        "full_memory_mb": full_mem,
        "turboq_memory_mb": turboq_mem,
    }
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to benchmark_results.json")

    # Generate plots
    generate_plots(full_metrics, turboq_metrics, full_mem, turboq_mem, out_dir)

    return results


def generate_plots(full_m, turboq_m, full_mem, turboq_mem, out_dir):
    """Generate 5 benchmark plots."""
    fig_params = dict(figsize=(8, 5))

    # 1. Memory comparison
    fig, ax = plt.subplots(**fig_params)
    bars = ax.bar(["Full Precision\n(float32)", "TurboQ\n(3-bit)"],
                  [full_mem, turboq_mem], color=["#e94560", "#0f3460"],
                  width=0.5, edgecolor="#eee", linewidth=0.5)
    ax.set_ylabel("Memory (MB)")
    ratio = full_mem / turboq_mem
    ax.set_title(f"Memory Footprint ({ratio:.1f}x reduction)", fontweight="bold")
    for bar, val in zip(bars, [full_mem, turboq_mem]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f} MB", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "memory_comparison.png"), bbox_inches="tight")
    plt.close()

    # 2. Latency comparison
    fig, ax = plt.subplots(**fig_params)
    lats = [full_m["latencies_ms"], turboq_m["latencies_ms"]]
    bars = ax.bar(["Full Precision", "TurboQ"], lats,
                  color=["#e94560", "#0f3460"], width=0.5,
                  edgecolor="#eee", linewidth=0.5)
    ax.set_ylabel("Latency (ms/query)")
    ax.set_title("Query Latency Comparison", fontweight="bold")
    for bar, val in zip(bars, lats):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f} ms", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "latency_comparison.png"), bbox_inches="tight")
    plt.close()

    # 3. Precision@k line chart
    fig, ax = plt.subplots(**fig_params)
    ks = [1, 3, 5, 10]
    full_prec = [full_m[f"precision@{k}"] for k in ks]
    turboq_prec = [turboq_m[f"precision@{k}"] for k in ks]
    ax.plot(ks, full_prec, "o-", color="#e94560", label="Full Precision", linewidth=2)
    ax.plot(ks, turboq_prec, "s--", color="#00d2ff", label="TurboQ (3-bit)", linewidth=2)
    ax.set_xlabel("k")
    ax.set_ylabel("Precision@k")
    ax.set_title("Precision@k: Full vs TurboQ", fontweight="bold")
    ax.set_xticks(ks)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "precision_at_k.png"), bbox_inches="tight")
    plt.close()

    # 4. MRR and nDCG comparison
    fig, ax = plt.subplots(**fig_params)
    x = np.arange(2)
    width = 0.35
    full_vals = [full_m["mrr@10"], full_m["ndcg@10"]]
    turboq_vals = [turboq_m["mrr@10"], turboq_m["ndcg@10"]]
    bars1 = ax.bar(x - width / 2, full_vals, width, label="Full Precision", color="#e94560")
    bars2 = ax.bar(x + width / 2, turboq_vals, width, label="TurboQ", color="#0f3460")
    ax.set_ylabel("Score")
    ax.set_title("MRR@10 and nDCG@10 Comparison", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["MRR@10", "nDCG@10"])
    ax.legend()
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mrr_ndcg_comparison.png"), bbox_inches="tight")
    plt.close()

    # 5. Metric correlation scatter plot
    fig, ax = plt.subplots(figsize=(6, 6))
    labels = ["P@1", "P@3", "P@5", "P@10", "R@1", "R@5", "R@10",
              "MRR@10", "nDCG@10", "Hit@1", "Hit@5"]
    metric_keys = ["precision@1", "precision@3", "precision@5", "precision@10",
                   "recall@1", "recall@5", "recall@10",
                   "mrr@10", "ndcg@10", "hit@1", "hit@5"]
    full_vals = [full_m[k] for k in metric_keys]
    turboq_vals = [turboq_m[k] for k in metric_keys]
    ax.scatter(full_vals, turboq_vals, s=80, c="#00d2ff", edgecolors="#eee",
               linewidth=0.5, zorder=5)
    for i, label in enumerate(labels):
        ax.annotate(label, (full_vals[i], turboq_vals[i]),
                     textcoords="offset points", xytext=(8, 4), fontsize=8)
    lims = [0, max(max(full_vals), max(turboq_vals)) * 1.1]
    ax.plot(lims, lims, "--", color="#e94560", alpha=0.7, label="y=x (perfect match)")
    ax.set_xlabel("Full Precision")
    ax.set_ylabel("TurboQ")
    ax.set_title("Metric Correlation: Full vs TurboQ", fontweight="bold")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "score_correlation.png"), bbox_inches="tight")
    plt.close()

    print(f"Plots saved to {out_dir}/")


if __name__ == "__main__":
    run_benchmark()
