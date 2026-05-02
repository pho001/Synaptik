# Phase 22 Context: Coverage Truth And Semantics Lock

**Milestone:** v1.4 Native GPU Operation Coverage Closure
**Date:** 2026-05-01
**Status:** Ready for planning/execution

## Goal

Convert the remaining GPU fallback list into auditable operation-family contracts and coverage gates before changing backend behavior.

The important correction from v1.3 is that matrix rows and stable rejection evidence are not the same thing as native execution. Phase 22 makes that distinction explicit so later phases can safely mark an operation as supported only when the backend can actually execute the lowered primitive or GPU sub-DAG.

## User Target List

- Forward reductions: `SUM`, `MEAN`, `REDUCE_MIN`, `REDUCE_MAX`
- Normalization: `LAYER_NORM`, `RMS_NORM`
- Forward SDPA, currently fallback/disabled until semantics are verified
- Loss-adjacent ops
- Conv/pool ops
- Gather/scatter/index ops
- BOOL compare outputs

## Current Evidence

- `GpuLoweringCoverageMatrix` still marks reductions as `FALLBACK`, normalization as `FALLBACK`, forward SDPA as `FALLBACK`, loss-adjacent as `UNSUPPORTED`, conv/pool as `UNSUPPORTED`, gather/scatter as `UNSUPPORTED`, and BOOL-producing compare rows as `UNSUPPORTED`.
- `AcceleratorDagNodeType` lacks forward reduction, normalization, conv/pool, gather/scatter/index, and compare BOOL-output node types. It has SDPA and backward-adjacent nodes.
- Metal native MPSGraph code already contains MPSGraph reduction helper usage for gradients and direct SDPA code paths, but planner admission remains conservative.
- CUDA native buffer execution is still much narrower than the Java-side DAG contract and needs capability-gated treatment for new primitives.
- Public `Tensor` must remain logical. Device residency and backend execution state belong in compile/prepare/execute runtime artifacts.

## Phase Decisions

1. Supported means executable. A v1.4 supported matrix row must map to a lowered DAG primitive or backend-owned GPU sub-DAG that can execute on a legal runtime path.
2. Rejection evidence remains valuable, but it does not close native coverage.
3. Metal and CUDA share contracts and reason vocabulary. Backend implementations can differ behind capability gates.
4. The first execution phase after this should be reductions, because normalization depends on reduction primitives.
5. Do not update canonical local calibration/profile artifacts as part of this phase.

## Requirements

- GPUNATIVE-01
- GPUNATIVE-02
- GPUNATIVE-03

## Verification Expectations

- Focused tests should prove the matrix/contract view classifies the listed families correctly.
- Report/gate tests should distinguish native buffer execution, tensor-array bridge execution, CPU fallback, and candidate shortening.
- Docs should state that v1.3 coverage evidence exposed these gaps, while v1.4 closes real execution coverage.
