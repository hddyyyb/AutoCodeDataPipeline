# System Design Document

This document explains the design decisions of AutoCodeDataPipeline and how they explicitly address the requirements of the interview assignment:  
“Local Code Repository–Based Intelligent Training Data Generation and Processing”.

## Design Goals

The goal of AutoCodeDataPipeline is to automatically generate auditable and high-quality training data from a local code repository.

Key objectives:
- Every generated sample is grounded in original source code
- Reasoning is explicit, structured, and auditable
- Training data is aligned with inference-time constraints
- The system is extensible to new repositories and domains

---

## High-Level Architecture

The system is organized as a staged pipeline:

1. Repository indexing
2. Repository understanding (domain map, rules, flows)
3. Data generation (QA and design tasks)
4. Validation and governance
5. Training-ready export and inference verification

---

## Chunk-Level Grounding

All downstream artifacts reference a stable `chunk_id` generated during repository indexing.

Each chunk records:
- File path
- Start and end line
- Original content

This provides precise and stable traceability across dataset iterations.

---

## Heuristic but Auditable Repository Understanding

Instead of constructing a full call graph, the system applies heuristics to infer:
- Architectural boundaries
- Domain entities and operations
- Candidate business flow skeletons

This tradeoff prioritizes explainability and robustness.
All inferred structures retain explicit evidence references.

---

## Reasoning Trace Design

The system intentionally avoids free-form chain-of-thought.

Instead, it generates short, structured reasoning traces:
- Stored as ordered step lists
- Each step grounded in evidence
- Minimum depth enforced:
  - QA tasks: at least 2 steps
  - Design tasks: at least 3 steps

These traces are designed for auditability and compliance.

---

## Training–Inference Alignment

The same constraints applied during dataset construction are enforced again at inference time.

Outputs that violate format, evidence, or reasoning rules are rejected or automatically repaired using trusted evidence.