
# Toyota RAG Project - AI Agent Guidelines (AGENTS.md)

## 1. Project Context & Architecture

You are operating within an Edge AI RAG (Retrieval-Augmented Generation) system designed for Toyota factory environments.

- **Hardware:** NVIDIA Jetson Orin (64GB Unified Memory). Strict constraints on VRAM, CPU utilization, and thermal limits.
- **Goal:** Provide offline, highly accurate, and fast responses to factory operators based on technical manuals.
- **Tech Stack:** Python, LangChain, Ollama (Llama 3.1, Qwen 2.5, Gemma 3, GPT-OSS), ChromaDB, BM25, PyTorch (Cross-Encoder).
- **Evaluation:** A rigorous Google SRE-style evaluation pipeline (`evaluate.py`) is in place, tracking p95 latency, TTFT, CPU/GPU usage via asynchronous monitoring (`jtop`/`tegrastats`), and strict benchmark manifests.

## 2. Core Directive: Root Cause Analysis (RCA) First

**NEVER apply "band-aid" solutions, blind patches, or guess the fix.**
If an error occurs, latency spikes, or accuracy drops, you MUST follow this sequence:

1. **Investigate:** Write and execute diagnostic scripts (e.g., checking SQLite hashes, parsing specific log lines, measuring component execution times).
2. **Identify:** Pinpoint the exact module or logical flaw causing the issue (e.g., "The `top_k=2` limit in the retrieval step is cutting off relevant context, causing the LLM to hallucinate").
3. **Propose:** Explain the root cause to the user (Faruk) and propose the architectural or code-level fix.
4. **Execute:** Only after identifying the root cause, apply the fix.

## 3. Communication Protocol (Mandatory)

You must keep the user (Faruk) deeply informed at every step. Do not execute large refactors silently.

- **Explain Your Thoughts:** Before writing or changing code, briefly explain *what* you found and *why* you are choosing a specific solution.
- **Provide Summaries:** After executing tests or patches, summarize the results clearly (e.g., "I ran the test. The hash mismatch was a false positive caused by Chroma's SQLite background writes. I have updated the reporting script to ignore this specific hash.").
- **Highlight Trade-offs:** If a solution improves quality but increases latency, explicitly state this trade-off.

## 4. Architectural Boundaries

- **Do not break the Evaluation Suite:** Respect the `manifest.json` structure, the dual-query architecture, and the separation of `archive_old_runs` from active benchmarks.
- **Hardware Awareness:** Always consider that this runs on a Jetson Orin. Avoid logic that causes GPU memory leaks or unnecessary CPU bottlenecks.

## 5. Definition of Done

A task is only considered complete when:

1. The code is written and syntactically correct.
2. The root cause of the previous failure is provably resolved.
3. The evaluation pipeline runs successfully without unexpected side effects.
4. A clear, concise summary of the changes and their impact has been presented to the user.

## 6. Security & Confidentiality (CRITICAL)

- **Proprietary Data:** The files `user_manuel_dev.md`, `ui-user-guide.md`, and `user_manuel_dev.pdf` are strictly confidential Toyota corporate documents.
- **Zero Data Leakage:** You MUST NOT export, share, or transmit the contents of these files to any external API, external logging service, or public endpoint.
- **Handling:** Treat all manual contents, schemas, and proprietary knowledge extracted from these files as strictly local. Do not include raw proprietary text in your own internal memory summaries if requested to share context outside this sandbox.
