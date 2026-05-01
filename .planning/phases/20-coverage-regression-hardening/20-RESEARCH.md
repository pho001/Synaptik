---
phase: 20-coverage-regression-hardening
status: complete
researched_at: 2026-05-01
---

# Phase 20 Research

## Summary

Phase 20 should harden the existing GPU coverage evidence pipeline instead of adding new accelerator operation support.
The current code already has the right foundation:

- `GpuCoverageSummary` derives backend coverage from prepare/run traces.
- `GpuCoverageRegressionGate`, `GpuCoverageGatePolicy`, and `GpuCoverageGateResult` provide a small fail-fast gate.
- `GpuHotPathCoverageTargets` defines the four v1.3 target workloads.
- Benchmark text/JSON renderers already publish most Phase 18/19 evidence fields.

The missing work is stricter target-aware gate semantics, report-visible gate results, native pass/skip evidence, and
closure docs that let milestone audit answer whether each target hot path stayed on GPU.

## Implementation Guidance

### Gate policy extension

Extend the existing gate path rather than creating a parallel checker. Add deterministic policy fields for Phase 18/19
evidence:

- minimum multi-op GPU region count,
- minimum lowered primitive count,
- minimum GPU fused subpattern count,
- required native-buffer coverage,
- allowed tensor-array, CPU fallback, CPU materialization, and device handoff budgets,
- optional expected reason evidence for targets that remain intentionally blocked.

Stable failure strings are important because tests and later audit flows should not parse prose-only benchmark output.

### Target-aware expectations

Tie expectations to `GpuHotPathCoverageTargets.defaultWorkloadNames()` and the Phase 14 target list. Do not let Phase 20
invent separate workload names. `transformer_block_hot_path` and `mlp_classifier_small` should enforce positive GPU
residency/lowering/fusion evidence. `conv2d_resnet_3x3` and `layer_norm_small` can assert explicit blocker evidence
when native support is still intentionally absent.

### Reports and native evidence

Text and JSON benchmark reports should expose enough gate inputs and gate result data that a reviewer can see why a
target passed, failed, or skipped native evidence. Portable Java tests remain mandatory. Native Metal/CUDA tasks should
remain capability-gated and record skip evidence instead of silently disappearing.

### Artifact hygiene

Local `profiles/platform/.../tuning/abc/*` files are not canonical coverage proof. Phase 20 should keep that invariant
visible in docs and source hygiene checks.

## Validation Architecture

Phase 20 validation should prove:

1. Gate policies fail on lost selected region length, lost lowered primitive count, lost fused subpattern count,
   unexpected CPU materialization, tensor-array fallback, CPU fallback, unexpected handoff, missing native buffer
   binding, and missing coverage summary.
2. Target policies are derived from the Phase 14 target list and cover transformer, MLP, conv, and normalization names.
3. Text and JSON reports render gate result and native pass/skip evidence.
4. Portable Java gates pass without requiring CUDA hardware.
5. Native Metal/CUDA evidence remains capability-gated and documented.
6. Local tuning artifacts remain unstaged.

## Risks

- Over-strict gates could fail on targets with intentionally unsupported conv/norm blockers. Mitigate with explicit
  blocker-reason expectations.
- Timing-based gates could be flaky. Keep timing out of pass/fail.
- CUDA hardware may be unavailable. Make native evidence capability-gated while keeping portable contracts hard.
- Adding report fields without tests could make milestone audit ambiguous. Require text and JSON assertions.
