---
phase: 15-gpu-region-internal-lowered-dag-contract
verified: 2026-05-01T08:22:55Z
status: passed
score: 5/5
requirements_verified: [GPUDAG-01, GPUDAG-02, GPUDAG-03]
human_verification: []
gaps: []
---

# Phase 15 Verification: GPU Region Internal Lowered DAG Contract

## Verdict

Passed. Phase 15 achieved its roadmap goal: selected GPU regions now have a Java-side lowered DAG manifest contract with original operation metadata, lowered primitive metadata, fused subpattern summaries, dtype/layout/storage assumptions, stable rejection/candidate-span reason codes, and trace/report rendering without changing public `Tensor` semantics or native Metal/CUDA ABI.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | A selected GPU region can be described as an internal lowered DAG, not only as one opaque operation. | Verified | `GpuLoweredRegionManifest` records ordered original ops, ordered lowered primitives, region id, backend, and selected region length. |
| 2 | Region metadata includes original op ids/types, primitive ids/types, backend id, dtype/layout/storage assumptions, fused summary, and region length. | Verified | `GpuLoweredRegionManifestTest` and `AcceleratorSubgraphLowererTest` cover bidirectional mapping, assumptions, fused summary, backend, and length. |
| 3 | Rejection metadata can point to primitive, boundary, candidate shortening, and fused subpattern reasons. | Verified | `GpuLoweringUnsupportedReason` includes `DAG_PRIMITIVE_UNSUPPORTED`, `DAG_REGION_BOUNDARY_MATERIALIZATION`, `DAG_CANDIDATE_SHORTENED`, and `DAG_FUSED_SUBPATTERN_REJECTED`; manifest tests assert all. |
| 4 | Trace/debug and benchmark reports render stable structured and compact manifest evidence. | Verified | `CompiledGraphTraceTest`, `BenchmarkSessionTest`, and `GpuCoverageSummaryTest` cover prepare trace, run trace `gpuLoweredRegionId`, text report, and JSON report contracts. |
| 5 | CPU execution, CPU fusion, public tensor semantics, and native Metal/CUDA ABI remain unchanged. | Verified | `15-SECURITY.md`, `15-VALIDATION.md`, and docs record Java-side-only metadata, public logical `Tensor`, CPU `Operation.OpType.FUSED` isolation, and unchanged native ABI. |

**Score:** 5/5 truths verified.

## Requirements Coverage

| Requirement | Status | Evidence |
|-------------|--------|----------|
| GPUDAG-01 | Satisfied | `15-01-SUMMARY.md`, `15-02-SUMMARY.md`, and `15-04-SUMMARY.md` document the multi-op/multi-primitive manifest model and selected GPU plan exposure. |
| GPUDAG-02 | Satisfied | Manifest model, lowerer construction, Metal/CUDA plan exposure, and trace/report tests cover original ops, primitives, backend, assumptions, fused summary, and selected region length. |
| GPUDAG-03 | Satisfied | Stable DAG reason enum values and candidate-span/rejection records are tested and rendered in trace/report evidence. |

**Coverage:** 3/3 requirements satisfied.

## Plan Completion

| Plan | Status | Summary |
|------|--------|---------|
| 15-01 Manifest Model And Reason Vocabulary | Complete | Added Java-side manifest records and stable DAG-level reason codes. |
| 15-02 Manifest Construction And Backend Plan Exposure | Complete | Built manifests in shared accelerator lowering and exposed them through Metal/CUDA selected plans. |
| 15-03 Trace And Report Manifest Contract | Complete | Attached manifests to prepare/backend-selection trace and rendered text/JSON report evidence. |
| 15-04 Docs Validation And Manifest Closure | Complete | Documented boundaries, ran focused verification, recorded validation evidence, and preserved artifact hygiene. |

## Automated Checks

- `./gradlew classes` - passed.
- `./gradlew test --tests backend.accelerator.lowering.GpuLoweredRegionManifestTest --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests CompiledGraphTraceTest --tests BenchmarkSessionTest --tests GpuCoverageSummaryTest --tests SourceTreeHygieneTest` - passed.
- `15-UAT.md` - complete with 4/4 tests passed and 0 issues.
- `15-SECURITY.md` - verified with `threats_open: 0`.
- `15-VALIDATION.md` - verified with `nyquist_compliant: true`, 0 gaps found, and 0 escalations.

## Human Verification

None required. Phase 15 behavior is internal accelerator manifest, trace/report, docs, and contract evidence covered by automated tests plus UAT confirmation.

## Gaps Summary

No gaps found. Phase goal achieved. Phase 15 is ready to feed Phase 16 dtype/storage residency expansion and later Phase 20 coverage gates.

## Residual Risk

Native CUDA execution remains capability-gated on this host; Phase 15 did not require native CUDA ABI or execution changes.

## Git Hygiene

Existing local profile tuning changes under `profiles/platform/mac_os_x-aarch64-oracle_corporation-16c/tuning/abc/*` remain unstaged and are not part of Phase 15.

---
*Verified: 2026-05-01T08:22:55Z*
*Verifier: Codex*
