# TurboQuant RAG Demo

A demonstration of **TurboQuant**-inspired vector compression applied to Retrieval-Augmented Generation (RAG) pipelines. Compares full-precision float32 embeddings against 3-bit quantized vectors to show the memory/accuracy tradeoff.

## What is TurboQuant?

TurboQuant is Google's KV-cache compression algorithm (March 2026) that achieves:

- **6x memory reduction** in vector storage
- **8x faster retrieval** on H100 GPUs
- **Zero accuracy loss** using PolarQuant + QJL techniques

This demo implements a simplified version of TurboQuant's core principles applied to RAG embeddings rather than KV-cache.

### How It Works

**PolarQuant** decomposes each embedding vector into:
- A **magnitude** scalar (stored as float16)
- A **direction** vector quantized to 3 bits (8 levels, stored as uint8)

**QJL (Johnson-Lindenstrauss)** error correction:
- Projects quantization residuals through a random matrix
- Stores sign bits as compact error correction codes
- Reconstructs an approximation of the lost precision during dequantization

## Quick Start

```bash
pip install -r requirements.txt
python benchmark.py       # Run full benchmark + generate plots
python demo.py            # Interactive query demo
```

## Project Structure

```
turboq-demo/
├── quantizer.py           # TurboQuantizer (PolarQuant + QJL)
├── pipeline_full.py       # Full-precision FAISS pipeline
├── pipeline_turboq.py     # TurboQuant compressed pipeline
├── evaluate.py            # Precision/Recall/Latency metrics
├── benchmark.py           # End-to-end benchmark + plot generation
├── demo.py                # Interactive CLI demo
├── requirements.txt
├── README.md
└── plots/                 # Generated benchmark plots
    ├── memory_comparison.png
    ├── latency_comparison.png
    ├── precision_at_k.png
    └── score_correlation.png
```

## Components

### Full Precision Pipeline
- Generates 384-dim float32 embeddings using `all-MiniLM-L6-v2`
- Stores in a FAISS `IndexFlatIP` (inner product / cosine similarity)
- Baseline for accuracy and memory measurements

### TurboQuant Pipeline
- Compresses embeddings using PolarQuant (magnitude + 3-bit direction)
- Applies QJL sign-bit error correction
- Dequantizes on-the-fly for retrieval
- Computes cosine similarity against dequantized vectors

### Evaluation
- **Retrieval accuracy**: Precision@k and Recall@k (k=1,3,5,10)
- **Ground truth**: Full-precision top-5 results treated as relevant set
- **Latency**: Average query time in milliseconds
- **Memory**: Byte-level comparison of both representations

## Interpreting Results

- **Precision@k** measures what fraction of the top-k quantized results match the full-precision top-5. High values (>0.8) indicate minimal accuracy loss from compression.
- **Memory reduction** of ~3-5x is expected from 3-bit quantization with float16 magnitudes.
- **Score correlation** (scatter plot) should cluster tightly around the y=x diagonal, confirming that similarity scores are well-preserved.
- Latency differences depend on hardware; the quantized pipeline trades FAISS optimization for numpy-based similarity but uses less memory bandwidth.

## Dataset

Uses the [SQuAD](https://huggingface.co/datasets/squad) dataset:
- 500 deduplicated passage contexts as the corpus
- 50 questions as evaluation queries

## References

- [TurboQuant: Quantized KV Cache Compression via Progressive Quantization](https://arxiv.org/abs/2503.xxxxx) — Google Research, March 2026
- [Sentence-Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
