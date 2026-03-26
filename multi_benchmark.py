"""Multi-dataset TurboQuant benchmark (v4).

Runs compression benchmarks across MS MARCO, HotpotQA, and NQ Open,
then reports per-dataset and aggregate average results with plots.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tabulate import tabulate

from data_loader import load_msmarco, load_hotpotqa, load_nq_with_wikipedia
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


DATASETS = [
    ("MS MARCO", load_msmarco),
    ("HotpotQA", load_hotpotqa),
    ("NQ Open", load_nq_with_wikipedia),
]

BIT_CONFIGS = [4, 6, 8]
VARIANT_NAMES = ["Full"] + [f"TurboQ-{b}bt" for b in BIT_CONFIGS]

COLORS = {
    "Full": "#e94560",
    "TurboQ-4bt": "#0f3460",
    "TurboQ-6bt": "#00d2ff",
    "TurboQ-8bt": "#53d769",
}


def run_dataset_benchmark(name, loader_fn):
    """Run full + TurboQ benchmarks on a single dataset."""
    print(f"\n{'='*60}")
    print(f"  DATASET: {name}")
    print(f"{'='*60}")

    corpus, query_data = loader_fn()

    # Full precision pipeline
    full = FullPrecisionPipeline()
    full.encode_corpus(corpus)
    full_mem = full.memory_mb()

    print(f"\nEvaluating Full Precision on {len(query_data)} queries...")
    full_metrics = evaluate_pipeline(full, query_data, k=10)
    full_metrics["memory_mb"] = full_mem

    # TurboQ variants
    variant_metrics = {"Full": full_metrics}
    for n_bits in BIT_CONFIGS:
        vname = f"TurboQ-{n_bits}bt"
        print(f"\n--- {vname} ---")
        pipeline = TurboQPipeline(n_bits=n_bits)
        pipeline.compress(full.embeddings.copy())
        mem = pipeline.memory_mb()

        print(f"Evaluating {vname} on {len(query_data)} queries...")
        metrics = evaluate_pipeline(pipeline, query_data, k=10)
        metrics["memory_mb"] = mem
        variant_metrics[vname] = metrics

    return {
        "corpus_size": len(corpus),
        "n_queries": len(query_data),
        "metrics": variant_metrics,
    }


def print_dataset_table(name, result):
    """Print per-dataset results table."""
    n_q = result["n_queries"]
    n_c = result["corpus_size"]
    m = result["metrics"]

    print(f"\nDataset: {name} ({n_q} queries, {n_c} passages)")
    rows = []
    for metric_key, label in [
        ("mrr@10", "MRR@10"),
        ("ndcg@10", "nDCG@10"),
        ("memory_mb", "Memory (MB)"),
        ("latencies_ms", "Latency (ms)"),
    ]:
        row = [label]
        for v in VARIANT_NAMES:
            val = m[v][metric_key]
            if metric_key in ("memory_mb", "latencies_ms"):
                row.append(f"{val:.2f}")
            else:
                row.append(f"{val:.4f}")
        rows.append(row)

    headers = ["Metric"] + VARIANT_NAMES
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def print_aggregate_table(all_results):
    """Print aggregate averages across all datasets."""
    print(f"\n{'='*68}")
    print("AGGREGATE AVERAGE (across all datasets)")
    print(f"{'='*68}")

    # Compute averages
    avg = {}
    for v in VARIANT_NAMES:
        avg[v] = {}
        for key in ("mrr@10", "ndcg@10", "memory_mb", "latencies_ms"):
            vals = [r["metrics"][v][key] for r in all_results.values()]
            avg[v][key] = np.mean(vals)

    rows = []
    for metric_key, label in [
        ("mrr@10", "Avg MRR@10"),
        ("ndcg@10", "Avg nDCG@10"),
    ]:
        row = [label]
        for v in VARIANT_NAMES:
            row.append(f"{avg[v][metric_key]:.4f}")
        rows.append(row)

    # Memory reduction ratio
    full_mem = avg["Full"]["memory_mb"]
    row = ["Avg Memory \u2193"]
    for v in VARIANT_NAMES:
        if v == "Full":
            row.append("1.0x")
        else:
            row.append(f"{full_mem / avg[v]['memory_mb']:.1f}x")
    rows.append(row)

    # Latency speedup ratio
    full_lat = avg["Full"]["latencies_ms"]
    row = ["Avg Latency \u2191"]
    for v in VARIANT_NAMES:
        if v == "Full":
            row.append("1.0x")
        else:
            row.append(f"{full_lat / avg[v]['latencies_ms']:.1f}x")
    rows.append(row)

    headers = ["Metric"] + VARIANT_NAMES
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Conclusion
    full_mrr = avg["Full"]["mrr@10"]
    print("\nCONCLUSION:")
    for n_bits in [8, 6, 4]:
        vname = f"TurboQ-{n_bits}bt"
        mrr_drop = (1 - avg[vname]["mrr@10"] / full_mrr) * 100
        mem_red = full_mem / avg[vname]["memory_mb"]
        lat_up = full_lat / avg[vname]["latencies_ms"]
        print(f"- {n_bits}-bit: {mrr_drop:.1f}% avg MRR drop, {mem_red:.1f}x memory reduction, {lat_up:.1f}x faster")

    return avg


def generate_plots(all_results, avg, out_dir):
    """Generate 4 multi-dataset plots."""
    os.makedirs(out_dir, exist_ok=True)
    dataset_names = list(all_results.keys())

    # 1. MRR@10 Heatmap
    fig, ax = plt.subplots(figsize=(10, 5))
    data = np.array([
        [all_results[ds]["metrics"][v]["mrr@10"] for v in VARIANT_NAMES]
        for ds in dataset_names
    ])
    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=data.min() - 0.05, vmax=data.max() + 0.02)
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(VARIANT_NAMES)))
    ax.set_xticklabels(VARIANT_NAMES)
    ax.set_yticks(range(len(dataset_names)))
    ax.set_yticklabels(dataset_names)
    for i in range(len(dataset_names)):
        for j in range(len(VARIANT_NAMES)):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                    color="black", fontweight="bold", fontsize=12)
    plt.colorbar(im, ax=ax, label="MRR@10")
    ax.set_title("MRR@10 Across Datasets and Quantization Levels", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "multi_mrr_heatmap.png"), bbox_inches="tight")
    plt.close()

    # 2. nDCG@10 Grouped Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(dataset_names))
    n_variants = len(VARIANT_NAMES)
    width = 0.8 / n_variants
    for i, v in enumerate(VARIANT_NAMES):
        vals = [all_results[ds]["metrics"][v]["ndcg@10"] for ds in dataset_names]
        offset = (i - n_variants / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=v, color=COLORS[v])
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("nDCG@10")
    ax.set_title("nDCG@10 Comparison Across Datasets", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names)
    ax.legend()
    ax.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "multi_ndcg_comparison.png"), bbox_inches="tight")
    plt.close()

    # 3. Memory Reduction & Latency Speedup (dual axis, averaged)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    turbo_variants = [v for v in VARIANT_NAMES if v != "Full"]
    x = np.arange(len(turbo_variants))

    full_mem = avg["Full"]["memory_mb"]
    full_lat = avg["Full"]["latencies_ms"]
    mem_reductions = [full_mem / avg[v]["memory_mb"] for v in turbo_variants]
    lat_speedups = [full_lat / avg[v]["latencies_ms"] for v in turbo_variants]

    width = 0.35
    bars1 = ax1.bar(x - width / 2, mem_reductions, width, label="Memory Reduction",
                    color="#e94560", alpha=0.85)
    ax1.set_ylabel("Memory Reduction (x)", color="#e94560")
    ax1.tick_params(axis="y", labelcolor="#e94560")

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, lat_speedups, width, label="Latency Speedup",
                    color="#00d2ff", alpha=0.85)
    ax2.set_ylabel("Latency Speedup (x)", color="#00d2ff")
    ax2.tick_params(axis="y", labelcolor="#00d2ff")

    for bar, val in zip(bars1, mem_reductions):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{val:.1f}x", ha="center", va="bottom", fontweight="bold", fontsize=10)
    for bar, val in zip(bars2, lat_speedups):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.1f}x", ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax1.set_xticks(x)
    ax1.set_xticklabels(turbo_variants)
    ax1.set_title("Avg Memory Reduction & Latency Speedup vs Full Precision", fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "multi_memory_latency.png"), bbox_inches="tight")
    plt.close()

    # 4. Average MRR@10 Summary Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_mrrs = [avg[v]["mrr@10"] for v in VARIANT_NAMES]
    colors = [COLORS[v] for v in VARIANT_NAMES]
    bars = ax.bar(VARIANT_NAMES, avg_mrrs, color=colors, width=0.5,
                  edgecolor="#eee", linewidth=0.5)
    for bar, val in zip(bars, avg_mrrs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("Average MRR@10")
    ax.set_title("Average MRR@10 Across All 3 Datasets", fontweight="bold")
    ax.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "multi_average_summary.png"), bbox_inches="tight")
    plt.close()

    print(f"Multi-dataset plots saved to {out_dir}/")


def run_multi_benchmark():
    setup_dark_theme()
    out_dir = os.path.join(os.path.dirname(__file__), "plots", "multi")

    all_results = {}
    for ds_name, loader_fn in DATASETS:
        all_results[ds_name] = run_dataset_benchmark(ds_name, loader_fn)

    # Print summary
    print(f"\n{'='*68}")
    print("MULTI-DATASET TURBOQ BENCHMARK SUMMARY")
    print(f"{'='*68}")

    for ds_name in all_results:
        print_dataset_table(ds_name, all_results[ds_name])

    avg = print_aggregate_table(all_results)

    # Generate plots
    generate_plots(all_results, avg, out_dir)

    # Save results JSON
    serializable = {}
    for ds_name, result in all_results.items():
        serializable[ds_name] = {
            "corpus_size": result["corpus_size"],
            "n_queries": result["n_queries"],
            "metrics": result["metrics"],
        }
    serializable["aggregate_average"] = {v: dict(avg[v]) for v in VARIANT_NAMES}

    with open("multi_benchmark_results.json", "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print("\nResults saved to multi_benchmark_results.json")

    return all_results, avg


if __name__ == "__main__":
    run_multi_benchmark()
