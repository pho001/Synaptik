---
phase: 15
slug: gpu-region-internal-lowered-dag-contract
status: verified
threats_open: 0
asvs_level: 1
created: 2026-05-01
---

# Phase 15 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Java manifest metadata to native accelerator DAG | `GpuLoweredRegionManifest` wraps selected region evidence beside `AcceleratorDagSpec` and must not alter native Metal/CUDA bridge contracts. | Region ids, op ids/types, primitive ids/types, dtype/layout assumptions, reason codes |
| Prepare trace to run trace | Prepare/backend-selection trace owns structured manifests; run trace may reference only compact region id and runtime outcome evidence. | `gpuLoweredRegionManifest` in prepare reports; `gpuLoweredRegionId` in run metadata |
| Internal accelerator metadata to public tensor API | Manifest and residency evidence stay in compile/prepare/execute internals; public `Tensor` remains logical. | Internal metadata only; no public device tensor handles |
| Local verification artifacts to repository | Local tuning profile changes must remain unstaged unless intentionally promoted as canonical fixtures. | `profiles/platform/.../tuning/abc/*` local files |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-15-01 | Tampering | Native DAG boundary | mitigate | Manifest records live under `backend.accelerator.lowering`; `AcceleratorDagSpec` and native/bridge paths were not changed by Phase 15. | closed |
| T-15-02 | Repudiation | Rejection reason vocabulary | mitigate | `GpuLoweringUnsupportedReason` contains stable DAG reason constants and rejection/span records store enum values instead of free-form strings. | closed |
| T-15-03 | Elevation of privilege | Public Tensor API boundary | mitigate | Manifest classes are internal accelerator-lowering metadata; public `Tensor` remains logical and was not modified by Phase 15. | closed |
| T-15-04 | Tampering | Manifest construction | mitigate | `AcceleratorSubgraphLowerer.buildManifest` derives primitive metadata directly from `AcceleratorDagSpec.nodes()` after DAG creation. | closed |
| T-15-05 | Spoofing | Metal/CUDA manifest contract | mitigate | `PartitionPlan.gpuLoweredRegionManifest()` is the shared contract; Metal and CUDA override it and keep backend-specific detail in extensions. | closed |
| T-15-06 | Repudiation | Multi-primitive source attribution | mitigate | `AcceleratorSubgraphLowererTest` verifies `LOG_SOFTMAX` maps to `SOFTMAX` and `LOG` primitives that retain source original node id. | closed |
| T-15-07 | Information integrity | Benchmark report contract | mitigate | `BackendSelectionDecisionTrace` carries structured manifest data and JSON reports expose stable manifest fields; text rendering is supplemental. | closed |
| T-15-08 | Tampering | Run trace metadata | mitigate | `PreparedExecution` records only `gpuLoweredRegionId`; tests assert run metadata does not contain `gpuLoweredRegionManifest`. | closed |
| T-15-09 | Denial of service | Existing synthetic trace compatibility | mitigate | `BackendSelectionDecisionTrace` preserves old constructors and defaults manifest metadata to null for old/rejected decisions. | closed |
| T-15-10 | Elevation of privilege | Future public residency assumptions | mitigate | Docs explicitly state `Public Tensor remains logical; device residency lives in compile/prepare/execute runtime state.` | closed |
| T-15-11 | Tampering | Native Metal/CUDA ABI assumptions | mitigate | Docs and closure evidence state Phase 15 is Java-side metadata and native Metal/CUDA ABI is unchanged. | closed |
| T-15-12 | Information disclosure | Local tuning artifact hygiene | mitigate | Validation and final summary require `git status --short` evidence; current status leaves `profiles/platform/.../tuning/abc/*` unstaged. | closed |

*Status: open · closed*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

No accepted risks.

---

## Verification Evidence

| Check | Evidence |
|-------|----------|
| Manifest separate from native DAG | `rg -n "record GpuLoweredRegionManifest\|record GpuLoweredRegionOriginalOp\|record GpuLoweredPrimitiveManifest\|record GpuLoweredRegionValueAssumption\|record GpuLoweredRegionRejection\|record GpuLoweredRegionCandidateSpan" src/main/java/backend/accelerator/lowering src/main/java/backend/accelerator/dag/AcceleratorDagSpec.java` |
| Stable DAG reason enum | `rg -n "DAG_PRIMITIVE_UNSUPPORTED\|DAG_REGION_BOUNDARY_MATERIALIZATION\|DAG_CANDIDATE_SHORTENED\|DAG_FUSED_SUBPATTERN_REJECTED\|GpuLoweringUnsupportedReason reason" src/main/java/backend/accelerator/lowering src/test/java/backend/accelerator/lowering/GpuLoweredRegionManifestTest.java` |
| No Phase 15 native/bridge/public Tensor ABI changes | `git diff --name-only f627666..HEAD \| rg "^(src/main/native\|src/main/java/tensor/Tensor.java\|src/main/java/tensor/TensorOps.java\|src/main/java/backend/(metal\|cuda)/bridge)" \|\| true` returned no paths. |
| Lowerer and backend plan manifest contract | `rg -n "buildManifest\|dagSpec.nodes\\(\\)\|new GpuLoweredPrimitiveManifest\|GpuLoweredRegionManifest manifest\|gpuLoweredRegionManifest\\(\\)\|backendExtensions" src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java src/main/java/backend/cuda/lowering/CudaGpuPartitionPlan.java src/main/java/backend/metal/lowering/MetalPartitionPlan.java src/main/java/graph/optimizer/partition/PartitionPlan.java` |
| Trace/report manifest contract | `rg -n "gpuLoweredRegionManifest\|GPU Lowered Region\|gpuLoweredRegionId\|candidateSpan" src/main/java/tuning/benchmark/report src/main/java/graph/execution/PreparedExecution.java src/test/java/CompiledGraphTraceTest.java src/test/java/BenchmarkSessionTest.java src/test/java/GpuCoverageSummaryTest.java` |
| Docs and hygiene boundaries | `rg -n "Java-side metadata\|Public Tensor remains logical\|Native Metal/CUDA ABI unchanged\|native Metal/CUDA ABI is unchanged\|CPU Operation.OpType.FUSED remains CPU-only\|profiles/platform/.../tuning/abc/\\* remained unstaged" docs .planning/phases/15-gpu-region-internal-lowered-dag-contract` |
| Focused security verification suite | `./gradlew test --tests backend.accelerator.lowering.GpuLoweredRegionManifestTest --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests CompiledGraphTraceTest --tests BenchmarkSessionTest --tests GpuCoverageSummaryTest --tests SourceTreeHygieneTest` passed. |
| Local artifact hygiene | `git status --short` shows only `profiles/platform/.../tuning/abc/*` local tuning profile changes. |

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-05-01 | 12 | 12 | 0 | Codex |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-05-01
