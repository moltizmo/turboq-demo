"""Full benchmark: MS MARCO v2.1 with multi-bit TurboQuant comparison (v3)."""

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


BIT_CONFIGS = [4, 6, 8]
COLORS = {
    "full": "#e94560",
    4: "#0f3460",
    6: "#00d2ff",
    8: "#53d769",
}
LABELS = {
    "full": "Full Precision\n(float32)",
    4: "TurboQ\n(4-bit)",
    6: "TurboQ\n(6-bit)",
    8: "TurboQ\n(8-bit)",
}
LABELS_SHORT = {
    "full": "Full Precision",
    4: "TurboQ-4bit",
    6: "TurboQ-6bit",
    8: "TurboQ-8bit",
}


def run_benchmark():
    setup_dark_theme()
    out_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(out_dir, exist_ok=True)

    # Load dataset — 500 queries
    corpus, query_data = load_msmarco(n_queries=500)

    # Initialize full-precision pipeline and encode corpus
    full = FullPrecisionPipeline()
    full.encode_corpus(corpus)
    full_mem = full.memory_mb()

    # Evaluate full precision
    print(f"\nEvaluating Full Precision on {len(query_data)} queries...")
    full_metrics = evaluate_pipeline(full, query_data, k=10)

    # Multi-bit TurboQ pipelines
    turboq_pipelines = {}
    turboq_metrics = {}
    turboq_mems = {}

    for n_bits in BIT_CONFIGS:
        print(f"\n--- TurboQ {n_bits}-bit ---")
        pipeline = TurboQPipeline(n_bits=n_bits)
        pipeline.compress(full.embeddings.copy())
        turboq_pipelines[n_bits] = pipeline
        turboq_mems[n_bits] = pipeline.memory_mb()

        print(f"Evaluating TurboQ-{n_bits}bit on {len(query_data)} queries...")
        turboq_metrics[n_bits] = evaluate_pipeline(pipeline, query_data, k=10)

    # Results table
    metric_keys = [
        "precision@1", "precision@3", "precision@5", "precision@10",
        "recall@5", "recall@10",
        "mrr@10", "ndcg@10", "hit@1", "hit@5",
    ]
    table = []
    for key in metric_keys:
        row = [key, f"{full_metrics[key]:.4f}"]
        for n_bits in BIT_CONFIGS:
            row.append(f"{turboq_metrics[n_bits][key]:.4f}")
        table.append(row)

    # Latency row
    lat_row = ["latency (ms/query)", f"{full_metrics['latencies_ms']:.2f}"]
    for n_bits in BIT_CONFIGS:
        lat_row.append(f"{turboq_metrics[n_bits]['latencies_ms']:.2f}")
    table.append(lat_row)

    # Memory row
    mem_row = ["memory (MB)", f"{full_mem:.2f}"]
    for n_bits in BIT_CONFIGS:
        mem_row.append(f"{turboq_mems[n_bits]:.2f}")
    table.append(mem_row)

    headers = ["Metric", "Full"] + [f"TurboQ-{b}bit" for b in BIT_CONFIGS]
    print("\n" + tabulate(table, headers=headers, tablefmt="github"))

    # Save results
    results = {
        "version": "v3",
        "corpus_size": len(corpus),
        "n_queries": len(query_data),
        "full_precision": full_metrics,
        "full_memory_mb": full_mem,
    }
    for n_bits in BIT_CONFIGS:
        results[f"turboq_{n_bits}bit"] = turboq_metrics[n_bits]
        results[f"turboq_{n_bits}bit_memory_mb"] = turboq_mems[n_bits]

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to benchmark_results.json")

    # Generate plots
    generate_plots(full_metrics, turboq_metrics, full_mem, turboq_mems, out_dir)

    return results


def generate_plots(full_m, turboq_m, full_mem, turboq_mems, out_dir):
    """Generate 5 benchmark plots for v3 multi-bit comparison."""
    fig_params = dict(figsize=(10, 6))

    # 1. Memory comparison — bar chart
    fig, ax = plt.subplots(**fig_params)
    labels = [LABELS["full"]] + [LABELS[b] for b in BIT_CONFIGS]
    values = [full_mem] + [turboq_mems[b] for b in BIT_CONFIGS]
    colors = [COLORS["full"]] + [COLORS[b] for b in BIT_CONFIGS]
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="#eee", linewidth=0.5)
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Memory Footprint: Full vs TurboQ (4/6/8-bit)", fontweight="bold")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f} MB", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "memory_comparison.png"), bbox_inches="tight")
    plt.close()

    # 2. Latency comparison — bar chart
    fig, ax = plt.subplots(**fig_params)
    lats = [full_m["latencies_ms"]] + [turboq_m[b]["latencies_ms"] for b in BIT_CONFIGS]
    labels_lat = [LABELS_SHORT["full"]] + [LABELS_SHORT[b] for b in BIT_CONFIGS]
    colors_lat = [COLORS["full"]] + [COLORS[b] for b in BIT_CONFIGS]
    bars = ax.bar(labels_lat, lats, color=colors_lat, width=0.5, edgecolor="#eee", linewidth=0.5)
    ax.set_ylabel("Latency (ms/query)")
    ax.set_title("Query Latency Comparison", fontweight="bold")
    for bar, val in zip(bars, lats):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f} ms", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "latency_comparison.png"), bbox_inches="tight")
    plt.close()

    # 3. MRR@10 and nDCG@10 grouped bar chart
    fig, ax = plt.subplots(**fig_params)
    x = np.arange(2)  # MRR@10, nDCG@10
    n_variants = 1 + len(BIT_CONFIGS)
    width = 0.8 / n_variants
    all_variants = ["full"] + BIT_CONFIGS

    for i, variant in enumerate(all_variants):
        if variant == "full":
            vals = [full_m["mrr@10"], full_m["ndcg@10"]]
            label = LABELS_SHORT["full"]
            color = COLORS["full"]
        else:
            vals = [turboq_m[variant]["mrr@10"], turboq_m[variant]["ndcg@10"]]
            label = LABELS_SHORT[variant]
            color = COLORS[variant]
        offset = (i - n_variants / 2 + 0.5) * width
        b = ax.bar(x + offset, vals, width, label=label, color=color)
        for bar in b:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score")
    ax.set_title("MRR@10 and nDCG@10 Comparison", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["MRR@10", "nDCG@10"])
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mrr_ndcg_comparison.png"), bbox_inches="tight")
    plt.close()

    # 4. Precision@k line chart
    fig, ax = plt.subplots(**fig_params)
    ks = [1, 3, 5, 10]
    ax.plot(ks, [full_m[f"precision@{k}"] for k in ks],
            "o-", color=COLORS["full"], label=LABELS_SHORT["full"], linewidth=2)
    for n_bits in BIT_CONFIGS:
        ax.plot(ks, [turboq_m[n_bits][f"precision@{k}"] for k in ks],
                "s--", color=COLORS[n_bits], label=LABELS_SHORT[n_bits], linewidth=2)
    ax.set_xlabel("k")
    ax.set_ylabel("Precision@k")
    ax.set_title("Precision@k: Full vs TurboQ Multi-Bit", fontweight="bold")
    ax.set_xticks(ks)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "precision_at_k.png"), bbox_inches="tight")
    plt.close()

    # 5. Accuracy vs Compression — Pareto frontier
    fig, ax = plt.subplots(figsize=(8, 6))
    from quantizer import TurboQuantizer
    dim = 384  # all-MiniLM-L6-v2 dimension

    points = []
    for n_bits in BIT_CONFIGS:
        q = TurboQuantizer(n_bits=n_bits)
        cr = q.compression_ratio(dim, 1000)
        mrr = turboq_m[n_bits]["mrr@10"]
        points.append((cr, mrr, n_bits))
        ax.scatter(cr, mrr, s=120, c=COLORS[n_bits], edgecolors="#eee",
                   linewidth=0.5, zorder=5)
        ax.annotate(f"{n_bits}-bit", (cr, mrr),
                    textcoords="offset points", xytext=(10, 5), fontsize=10)

    # Full precision: compression ratio = 1.0
    ax.scatter(1.0, full_m["mrr@10"], s=120, c=COLORS["full"], edgecolors="#eee",
               linewidth=0.5, zorder=5, marker="D")
    ax.annotate("Full (float32)", (1.0, full_m["mrr@10"]),
                textcoords="offset points", xytext=(10, 5), fontsize=10)

    # Connect Pareto frontier
    all_pts = [(1.0, full_m["mrr@10"])] + [(p[0], p[1]) for p in points]
    all_pts.sort()
    ax.plot([p[0] for p in all_pts], [p[1] for p in all_pts],
            "--", color="#e94560", alpha=0.5, linewidth=1.5)

    ax.set_xlabel("Compression Ratio (x)")
    ax.set_ylabel("MRR@10")
    ax.set_title("Accuracy vs Compression — Pareto Frontier", fontweight="bold")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_vs_compression.png"), bbox_inches="tight")
    plt.close()

    print(f"Plots saved to {out_dir}/")


if __name__ == "__main__":
    run_benchmark()
