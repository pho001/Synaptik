---
phase: 27
slug: conv-pool-and-bool-compare-outputs
status: verified
threats_open: 0
asvs_level: 1
created: 2026-05-02
---

# Phase 27 - Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Operation semantics -> shared GPU coverage matrix | Conv/pool and BOOL-producing op families become auditable rows before planner/backend admission. | `Operation.OpType`, operation family, status, unsupported reason, target note |
| Shared matrix -> Metal/CUDA planner diagnostics | Backend legality adapters must report stable Phase 27 reasons instead of generic missing-operation or dtype failures. | Backend, dtype, layout/rank family, capability reason |
| BOOL-producing CPU node -> GPU `WHERE` region | A CPU-produced BOOL predicate may feed a supported GPU `WHERE` region as an external input, but this must not imply native BOOL output compute. | BOOL predicate value, input role, region boundary, CPU materialization reason |
| Coverage/report evidence -> milestone proof | Reports must distinguish selected GPU region evidence from unsupported/rejected conv/pool and BOOL output compute. | Region length, lowered primitive count, fallback count, CPU exits, backend path |
| Local benchmark/profile outputs -> source control | Machine-local profile artifacts must not become canonical Phase 27 proof. | `profiles/platform/...` tuning and calibration files |

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-27-01 | Spoofing / integrity | Conv/pool support claims | mitigate | `GpuLoweringCoverageMatrix` lists every Phase 27 conv/pool target for Metal and CUDA as `UNSUPPORTED` with `CAPABILITY_MISSING`; lowerer tests assert representative planner diagnostics. | closed |
| T-27-02 | Spoofing / integrity | Native BOOL output support claims | mitigate | Compare, logical BOOL, and BOOL reduction rows remain `UNSUPPORTED` with `UNSUPPORTED_DTYPE`; docs explicitly separate external BOOL predicate residency from native BOOL-producing GPU compute. | closed |
| T-27-03 | Repudiation | Generic fallback diagnostics | mitigate | Matrix and semantics-contract tests enumerate the full Phase 27 surface so missing rows cannot silently collapse to generic `UNSUPPORTED_OPERATION` diagnostics. | closed |
| T-27-04 | Integrity | BOOL boundary hiding adjacent GPU coverage | mitigate | `PreparedExecutionBuildTest.gpuMetalWhereUsesCpuProducedComparePredicateAsExplicitBoolBoundary` proves `GT` remains a visible CPU boundary while adjacent Metal `WHERE + RELU` can still be selected. | closed |
| T-27-05 | Repudiation | Coverage report ambiguity | mitigate | `GpuCoverageSummaryTest` and `BenchmarkSessionTest` remained green; docs explain that existing report fields carry selected region length, lowered primitive count, backend path, CPU exits, and fallback evidence. | closed |
| T-27-06 | Information disclosure / API integrity | Public `Tensor` device semantics | mitigate | Phase 27 did not add public GPU tensor/device API. Residency remains in runtime state and device buffer bindings; docs keep `Tensor` logical. | closed |
| T-27-07 | Tampering / native boundary | Unsupported native ABI assumptions | mitigate | No native conv/pool or native BOOL output ABI was claimed. Support requires future DAG ABI, lowering, native execution, and parity evidence before any row can move to `SUPPORTED`. | closed |
| T-27-08 | Supply chain / integrity | Local profile artifacts as proof | mitigate | Phase verification records local benchmark/profile artifacts are not Phase 27 evidence and remain unstaged unless intentionally promoted. | closed |

*Status: open - closed*
*Disposition: mitigate (implementation required) - accept (documented risk) - transfer (third-party)*

## Accepted Risks Log

No accepted risks.

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-05-02 | 8 | 8 | 0 | Codex inline security audit |

## Audit Evidence

| Evidence | Result |
|----------|--------|
| `GpuLoweringCoverageMatrixTest`, `GpuTargetSemanticsContractTest` | Full conv/pool and BOOL output target surface has explicit Metal/CUDA rows and semantics contracts. |
| `MetalRegionLowererTest`, `CudaRegionLowererTest` | Planner diagnostics use stable Phase 27 rejection reasons for BOOL output and conv/pool targets. |
| `PreparedExecutionBuildTest.gpuMetalWhereUsesCpuProducedComparePredicateAsExplicitBoolBoundary` | External BOOL predicate boundary is visible while adjacent supported GPU region selection remains possible. |
| `Conv2dExecutionTest`, `Pool2dExecutionTest`, `BoolTensorInfrastructureTest` | CPU parity remains the correctness oracle for currently unsupported native GPU operation families. |
| `GpuCoverageSummaryTest`, `BenchmarkSessionTest` | Existing report schema continues to expose fallback, materialization, region, and backend-path evidence. |
| `./gradlew metalTest` | Metal native lane passed for the maintained Metal slice. |
| `git diff --check` | Whitespace/hygiene check passed after Phase 27 changes. |

## Security Audit 2026-05-02

| Metric | Count |
|--------|-------|
| Threats found | 8 |
| Closed | 8 |
| Open | 0 |

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-05-02
