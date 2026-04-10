# EvoAgentBench

The first objective benchmark for agent self-evolution.

## Overview

EvoAgentBench measures agent self-evolution capability through longitudinal growth curves rather than static snapshots. It uses controlled experiments (with vs. without evolution) to evaluate:

- **Transfer Efficiency** - How well does the agent transfer learned skills to new tasks?
- **Error Avoidance** - Does the agent learn to avoid previously encountered mistakes?
- **Skill Hit Quality** - How accurately does the agent apply acquired skills?

## Key Results

| Task | Agent + LLM | Baseline | + EverOS Skills | Delta |
| ---- | ----------- | -------- | --------------- | ----- |
| Code (Django) | OpenClaw + Qwen3.5-397B | 37% | 58% | **+21%** |
| Code (Django) | Nanobot + Qwen3.5-397B | 21% | 47% | **+26%** |
| General (GDPVAL) | OpenClaw + Qwen3.5-397B | 29% | 69% | **+40%** |
| General (GDPVAL) | OpenClaw + Qwen3.5-27B | 41% | 61% | **+20%** |

## Status

Coming soon.
