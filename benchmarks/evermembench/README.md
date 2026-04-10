# EverMemBench

A comprehensive benchmark for evaluating long-term memory quality in conversational AI.

## Overview

EverMemBench evaluates memory systems across three dimensions:

1. **Factual Recall** - Can the system accurately retrieve stored facts?
2. **Applied Reasoning** - Can the system reason over retrieved memories?
3. **Personalized Generalization** - Can the system generalize from memories to new contexts?

All memory systems and LLMs are evaluated under a unified standard.

## Dataset

- [EverMemBench-Dynamic on Hugging Face](https://huggingface.co/datasets/EverMind-AI/EverMemBench-Dynamic)

## Paper

- [EverMemBench: A Comprehensive Benchmark for Long-Term Memory in Conversational AI](https://arxiv.org/pdf/2602.01313)

## Quick Start

```bash
# From the project root
uv sync --group evaluation
uv run python -m evaluation.cli --dataset locomo --system evermemos --smoke
```

See the [Evaluation Guide](../../evaluation/README.md) for full details.
