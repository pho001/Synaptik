---
status: complete
phase: 15-gpu-region-internal-lowered-dag-contract
source:
  - 15-01-SUMMARY.md
  - 15-02-SUMMARY.md
  - 15-03-SUMMARY.md
  - 15-04-SUMMARY.md
started: 2026-05-01T08:01:32Z
updated: 2026-05-01T08:11:33Z
---

## Current Test

[testing complete]

## Tests

### 1. Manifest Model And Reason Vocabulary
expected: Phase 15 exposes a Java-side GPU lowered-region manifest model with stable fields for region id, backend, original ops, lowered primitives, dtype/layout/storage assumptions, fused summary, rejections, and candidate span. The stable reason vocabulary includes DAG_PRIMITIVE_UNSUPPORTED, DAG_REGION_BOUNDARY_MATERIALIZATION, DAG_CANDIDATE_SHORTENED, and DAG_FUSED_SUBPATTERN_REJECTED.
result: pass

### 2. Selected GPU Plans Carry Manifests
expected: Shared accelerator lowering builds manifests from the selected DAG, and selected Metal/CUDA partition plans expose the same manifest contract without changing native ABI.
result: pass

### 3. Trace And Benchmark Manifest Evidence
expected: Prepare/backend-selection trace carries the structured manifest; benchmark text renders a GPU Lowered Region block; benchmark JSON exposes gpuLoweredRegionManifest with stable keys; run trace contains gpuLoweredRegionId but not the full manifest.
result: pass

### 4. Documentation And Boundary Closure
expected: Docs and validation state that the manifest is Java-side metadata, Public Tensor remains logical, native Metal/CUDA ABI is unchanged, CPU Operation.OpType.FUSED remains CPU-only, final focused verification passed, and profiles/platform/.../tuning/abc/* remained unstaged.
result: pass

## Summary

total: 4
passed: 4
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

[none yet]
