---
phase: 24
slug: normalization-gpu-lowering
status: verified
threats_open: 0
asvs_level: 1
created: 2026-05-01
---

# Phase 24 - Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Graph normalization ops -> shared accelerator DAG | `LAYER_NORM` and `RMS_NORM` are expanded into backend-neutral reduction and elementwise primitives before backend admission. | Op type, input shape, normalized rank, epsilon, gamma/beta role, output shape |
| Shared DAG ABI -> native Metal/CUDA shims | Lowered primitives and scalar metadata cross the Java/native FFM boundary. | `AcceleratorDagNodeType`, scalar bits, node input ranks/dims, temporary buffer layout |
| Planner legality -> execution path | Matrix support must correspond to executable legal cases and explicit rejection for unsupported dtype/layout/rank/shape. | Support status, family, target, stable rejection reason |
| Device execution -> CPU correctness oracle | Legal selected-GPU normalization results are compared against CPU reference values. | FLOAT32 tensor values, tolerance, output publication boundary |
| Local benchmark/profile outputs -> source control | Machine-local profile artifacts must not become canonical evidence accidentally. | `profiles/platform/...` calibration and autotune files |

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-24-01 | Integrity | Normalization broadcast contract | mitigate | Lowering validates input rank, normalized rank, output shape, and gamma/beta tail shape before returning a DAG; `AcceleratorSubgraphLowererTest` covers valid LayerNorm/RMSNorm and invalid parameter shape rejection. | closed |
| T-24-02 | Integrity | Epsilon scalar ABI | mitigate | `AcceleratorDagNodeType.ADD_SCALAR` has stable ABI code `40`; epsilon is carried through scalar metadata and asserted by lowerer tests. | closed |
| T-24-03 | Integrity | CUDA native primitive coverage | mitigate | CUDA execution was extended for normalization-required unary/binary primitives and internal DAG temporary buffers before coverage rows moved to supported. Phase 24 aggregate tests cover CUDA lowering legality and prepared execution. | closed |
| T-24-04 | Tampering | CUDA broadcast index math | mitigate | CUDA validates suffix-broadcast dimensions before kernel launch and returns stable `unsupported broadcast` failure text for unsupported shapes. | closed |
| T-24-05 | Spoofing / integrity | Coverage matrix support claims | mitigate | `GpuLoweringCoverageMatrix`, `GpuTargetCoverageTruth`, Metal legality, and CUDA legality tests assert legal dense FLOAT32 normalization is supported/native-executable only after native primitive support exists. | closed |
| T-24-06 | Repudiation | Stable rejection vocabulary | mitigate | Metal and CUDA legality adapters preserve explicit `UNSUPPORTED_DTYPE`, `UNSUPPORTED_LAYOUT`, and `UNSUPPORTED_RANK_OR_SHAPE` normalization reasons for unsupported variants. | closed |
| T-24-07 | Integrity | Planner selection without value correctness | mitigate | `NormalizationExecutionTest` compares selected-GPU LayerNorm/RMSNorm outputs to CPU references over representative shapes and tolerance. | closed |
| T-24-08 | Repudiation | Hidden CPU materialization | mitigate | Prepared execution, trace, hot-path coverage, and regression-gate tests require legal normalization native evidence and visible CPU materialization only at true graph-output or CPU-consumer boundaries. | closed |
| T-24-09 | Supply chain / integrity | Local profile artifacts | mitigate | Phase summaries and validation record that local `profiles/platform/...` tuning artifacts are not phase evidence and remain unstaged. | closed |

*Status: open - closed*
*Disposition: mitigate (implementation required) - accept (documented risk) - transfer (third-party)*

## Accepted Risks Log

No accepted risks.

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-05-01 | 9 | 9 | 0 | Codex inline security audit |

## Audit Evidence

| Evidence | Result |
|----------|--------|
| `AcceleratorSubgraphLowererTest`, `GpuCompoundPatternDetectorTest` | Shared normalization DAG, `ADD_SCALAR`, epsilon metadata, compound summary, and invalid shape rejection covered. |
| `MetalMpsFfmBridgeTest`, `PreparedExecutionBuildTest`, `metalTest` | Metal native ADD_SCALAR / normalization-like lowered DAG and selected prepared execution passed locally. |
| `CudaRegionLowererTest`, `synaptik_cuda_graph_stub.cu` review | CUDA legality and native shim changes cover normalization primitive support, input1 shape metadata, temporary buffers, and stable unsupported broadcast rejection. |
| `GpuLoweringCoverageMatrixTest`, `MetalRegionLowererTest`, `GpuCoverageSummaryTest`, `CompiledGraphTraceTest` | Planner admission, supported rows, explicit unsupported cases, trace, and coverage reporting are covered. |
| `NormalizationExecutionTest`, `GpuHotPathCoverageTargetsTest`, `GpuCoverageRegressionGateTest`, `StandardWorkloadsTest` | CPU parity, native evidence expectations, no hidden CPU materialization gates, and representative normalization workloads are covered. |
| `git status --short profiles/platform` | Local benchmark/profile artifacts remain dirty locally but are not staged as phase evidence. |

## Security Audit 2026-05-01

| Metric | Count |
|--------|-------|
| Threats found | 9 |
| Closed | 9 |
| Open | 0 |

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-05-01
