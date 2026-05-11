# Approach Document: Conversational SHL Assessment Recommender

**Version:** 1.0 | **Last updated:** 12 May 2026
**Author:** AI Intern Candidate
© 2026 SHL and its affiliates. All rights reserved.

---

## 1. Problem Decomposition & Design Choices

The core challenge was translating ambiguous user intents into precise catalog matches while maintaining a high-quality conversational flow. 

### Architectural Decisions
- **Framework**: Built with **FastAPI** for high-performance, asynchronous API delivery. The service is strictly **stateless**, reconstructing the agent's "memory" from the provided message history in each request.
- **Orchestration**: Implemented a **LangGraph-inspired state machine**. Every request passes through an `Understand` node that extracts constraints and classifies intent, followed by a deterministic router that directs the flow to `Clarify`, `Compare`, `Recommend`, or `Refusal` nodes.
- **Robustness**: The SHL catalog data contained raw control characters and invalid encoding. I implemented a robust loader using `json.loads(text, strict=False)` and global ASCII-compliance sanitization to prevent runtime crashes.

---

## 2. Retrieval Strategy & Context Engineering

A simple keyword search is insufficient for the SHL catalog due to product name overlap (e.g., "Verify" vs "Verify Interactive").

### Hybrid Retrieval Setup
- **Vector Store**: Used **ChromaDB** with `all-MiniLM-L6-v2` embeddings for semantic retrieval.
- **Sparse Search**: Integrated **BM25** to capture exact technical terms (e.g., "Java", "Python", "OPQ32r").
- **Query Expansion**: To improve Recall, I implemented an LLM-based query expansion step that generates 4–6 targeted search queries per request, covering technical skills, cognitive abilities, and job-level context.
- **Depth**: After iterative testing, I increased retrieval depth to `top_k=20` to ensure specialized reports and newer product versions were surfaced for the LLM to select from.

---

## 3. Prompt Design & Agentic Behavior

Prompts were engineered for **deterministic JSON output** and **grounded reasoning**.

- **Aggressive Proactivity**: Early versions were too "consultative," asking redundant questions. I updated the prompt and the readiness logic to provide a preliminary shortlist as soon as *any* core constraint (role, skill, or seniority) is identified.
- **Selection Principles**: To match the human traces, I injected "Assessment Selection Principles" into the system prompt:
    - Prefer **Reasoning** tests over **Calculation** for professional roles.
    - Match **Situational Judgement** tests strictly to seniority (e.g., "Graduate Scenarios" for graduates).
- **Few-Shot Grounding**: Added exact-match few-shot examples to the recommendation engine to handle common pairings like "Graduate Financial Analyst" or "Tech Audit."

---

## 4. Evaluation Rigor & Iterative Improvement

### What Didn't Work (and How it Was Fixed)
- **The Turn 1 Bottleneck**: Initially, the agent forced a clarification on Turn 1 if the "purpose" was missing. This tanked Recall on shorter traces. Fixed by defaulting to "Selection" and allowing immediate recommendations.
- **Seniority Mismatch**: The semantic search often mixed up "Executive" and "Graduate" scenarios. Fixed by adding a post-retrieval filter in `retriever.py` that strictly honors the extracted seniority constraint.

### Measurement
I developed a local **Evaluation Harness (`eval.py`)** that replayed 10 sample traces turn-by-turn. 
- **Metrics**: Tracked **Recall@10**, **Schema Compliance (100% pass)**, and **Hallucination Rate (0%)**. 
- **Behavior Probes**: Verified that the agent correctly refuses off-scope topics and respects the 8-turn conversation cap.

---

## 5. AI-Assisted Development

This project was built using **Antigravity**, an agentic coding assistant.
- **Usage**: Used for rapid prototyping of the FastAPI structure, automating the extraction of JSON traces from markdown samples, and performing global ASCII cleanups.
- **Human Oversight**: Every design choice—especially the decision to use a hybrid retriever and the specific "ready" gate logic—was manually directed and validated against the SHL API contract.

---
**Submission materials available in the repository.**
