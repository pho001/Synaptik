# Phase 8: CUDA Observability And Documentation Closure - Context

## Gap Closure Scope

This phase closes the remaining v1.1 milestone audit gaps from `.planning/v1.1-MILESTONE-AUDIT.md`.

## Requirements

- CUDA-06: CUDA fallback and required-mode failures remain explicit through stable reason codes for unavailable native runtime, unsupported dtype, unsupported layout, and required-but-unavailable buffer execution.
- CUDADOC-01: CUDA traces and benchmark reports expose the same accelerator evidence contract as Metal.
- CUDADOC-02: Developer documentation explains CUDA build prerequisites, capability probing, native shim troubleshooting, fallback interpretation, and Metal/CUDA ABI symmetry.
- CUDADOC-03: Source hygiene and verification gates prevent accidental commits of local CUDA build outputs, machine-local benchmark/profile artifacts, and generated native scratch files.

## Audit Gaps To Close

- Phase 8 has no plan, summary, validation, or verification artifacts yet.
- Phase 7 CUDA buffer execution is verified, but the Phase 7-to-Phase 8 observability integration is missing.
- The CUDA native runtime reportability flow is broken until traces, benchmark reports, docs, hygiene, and final verification are implemented.

## Planning Notes

- Keep CUDA support narrow unless the plan explicitly proves broader coverage.
- CPU remains the correctness oracle.
- Public `Tensor` remains logical; CUDA handles and lifetimes stay under `backend.cuda.*`.
- Do not commit local CUDA build outputs or local benchmark/profile artifacts.
- Optional native CUDA checks must pass or skip with an explicit unavailable reason.

## Suggested Plan Shape

1. Add CUDA trace/report parity for native buffer, fallback, materialization, copy timing, and storage residency evidence.
2. Stabilize CUDA fallback and required-mode reason-code coverage for unavailable native runtime, unsupported dtype/layout, and required-unavailable execution.
3. Update developer docs for build prerequisites, capability probing, native shim troubleshooting, fallback interpretation, and Metal/CUDA ABI symmetry.
4. Add source hygiene and final verification gates for CUDA native outputs, local benchmark/profile artifacts, generated scratch files, portable Java tests, and capability-gated native CUDA checks.
