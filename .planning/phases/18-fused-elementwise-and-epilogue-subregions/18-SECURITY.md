---
phase: 18
slug: fused-elementwise-and-epilogue-subregions
status: verified
threats_open: 0
asvs_level: 1
created: 2026-05-01
---

# Phase 18 - Security

Per-phase security contract for fused elementwise and epilogue GPU subregions.

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Public tensor API to accelerator lowering | GPU fusion metadata must stay internal to compile/prepare/execute runtime contracts. | Tensor graph operation ids, dtype/layout metadata, lowered primitive ids. |
| CPU fused backend to GPU accelerator packages | CPU fused ASM/vector internals must remain CPU-owned and must not be imported by accelerator, Metal, or CUDA packages. | CPU fused operation descriptors, generated ASM/vector execution internals. |
| GPU prepared execution to trace/report rendering | Trace and benchmark fields must expose fusion evidence without hiding fallback or CPU materialization. | Run trace attributes, coverage summaries, benchmark report JSON/text fields. |
| Backend legality gates to native execution | Metal/CUDA epilogue support must remain gated by dtype, layout, capability, and buffer policy. | Backend-specific lowering plans, rejection reasons, native capability evidence. |

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-18-01 | Tampering | Fusion metadata model | mitigate | `GpuFusionSubpatternSummary` lives under `backend.accelerator.lowering` and is documented as trace/lowering metadata, not a public operation model. Public `Tensor` API was not changed. | closed |
| T-18-02 | Tampering | GPU fusion CPU-fused boundary | mitigate | `GpuCompoundPatternDetector` rejects `Operation.OpType.FUSED` with `CPU_FUSED_OPERATION_UNSUPPORTED`; source hygiene tests reject `import backend.cpu.fused` in accelerator, Metal, and CUDA packages. | closed |
| T-18-03 | Spoofing | Region optimizer coverage evidence | mitigate | `GenericGpuRegionOptimizationPolicy` preserves selected partition node ids while adding region-internal `FUSED_ELEMENTWISE` subchain units; tests assert mixed regions are not shortened. | closed |
| T-18-04 | Repudiation | AUTO fallback observability | mitigate | Required-mode and trace/report tests keep CPU consumer materialization, accelerator buffer decision, and fallback evidence visible beside fused subpattern metadata. | closed |
| T-18-05 | Tampering | Epilogue backend legality | mitigate | Metal/CUDA legality adapters and lowerer tests gate epilogue support by dtype/layout/backend capability and assert stable rejection detail for illegal layouts. | closed |
| T-18-06 | Tampering | Epilogue CPU fused isolation | mitigate | Epilogue units use `SPECIALIZED_PRIMITIVE` plus accelerator DAG/post-op metadata and `gpu-epilogue-subregion:` trace events, not CPU fused ASM/vector internals. | closed |
| T-18-07 | Repudiation | Trace and benchmark reports | mitigate | `PreparedExecution`, `GpuCoverageSummary`, and benchmark renderers add `gpuFusedSubpattern*` fields additively beside accelerator buffer, CPU materialization, tensor-array fallback, and device handoff fields. | closed |
| T-18-08 | Tampering | CPU fused regression boundary | mitigate | `OptimizerFuseTest`, `LoweredFusedOperationBuilderTest`, and `SourceTreeHygieneTest` verify CPU fused execution still uses `Operation.OpType.FUSED` only through CPU paths and GPU packages do not import CPU fused internals. | closed |

## Evidence

| Evidence | Result |
|----------|--------|
| `rg -n "import backend.cpu.fused" src/main/java/backend/accelerator src/main/java/backend/metal src/main/java/backend/cuda` | No matches. |
| `./gradlew test --tests SourceTreeHygieneTest --tests OptimizerFuseTest --tests backend.cpu.fused.plan.LoweredFusedOperationBuilderTest --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest` | Passed. |
| `./gradlew test --tests backend.accelerator.lowering.GpuCompoundPatternDetectorTest --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.accelerator.lowering.GpuLoweredRegionManifestTest --tests graph.optimizer.region.DefaultRegionOptimizerTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests OptimizerFuseTest --tests backend.cpu.fused.plan.LoweredFusedOperationBuilderTest --tests SourceTreeHygieneTest` | Passed during 18-04 closure. |

## Accepted Risks Log

No accepted risks.

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-05-01 | 8 | 8 | 0 | Codex |

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-05-01
