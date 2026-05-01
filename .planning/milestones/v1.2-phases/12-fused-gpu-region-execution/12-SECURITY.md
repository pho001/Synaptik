---
phase: 12
slug: fused-gpu-region-execution
status: verified
threats_open: 0
asvs_level: 1
created: 2026-04-30
---

# Phase 12 - Security

Per-phase security contract: threat register, accepted risks, and audit trail for fused GPU region execution.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| CPU fused execution to GPU compound lowering | CPU `Operation.OpType.FUSED` and ASM/vector internals must not become GPU executable primitives. | Operation types, compound summaries, lowering reason codes. |
| Accelerator DAG legality to compound summaries | Compound summaries describe intent but must not bypass `AcceleratorSubgraphLowerer` DAG legality or backend adapters. | `AcceleratorDagSpec`, `AcceleratorMatMulSpec`, ordered node ids, post-ops. |
| Metal/CUDA planner to prepared executables | Selected GPU regions must preserve full supported region membership and publish traceable compound metadata. | Partition plans, lowered regions, prepared executable metadata. |
| Native buffer residency to trace/report evidence | Device-owned intermediates must not silently materialize through CPU consumer boundaries. | `DeviceBufferBinding`, storage residency, CPU materialization traces, buffer reason codes. |
| Documentation and coverage matrix to users | Docs must not overclaim CUDA parity or public GPU tensor residency. | Supported/fallback/unsupported coverage rows, backend-specific capability notes. |
| Working tree to committed artifacts | Local tuning/profile files must not be staged as phase evidence. | Local profile JSON/JSONL files under `profiles/platform/.../tuning/abc/*`. |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-12-01 | Elevation of Privilege / Tampering | GPU compound detector and matrix | mitigate | `GpuCompoundPatternDetector` rejects `Operation.OpType.FUSED` as `CPU_FUSED_OPERATION_UNSUPPORTED`; `GpuLoweringCoverageMatrix` marks `FUSED` unsupported for Metal/CUDA; detector and matrix tests assert the stable reason before backend execution. | closed |
| T-12-02 | Tampering | Accelerator DAG legality | mitigate | `AcceleratorSubgraphLowerer` builds `AcceleratorDagSpec` and optional `AcceleratorMatMulSpec` before calling `GpuCompoundPatternDetector`; Metal/CUDA legality adapters still create plans from lowered DAG results. | closed |
| T-12-03 | Repudiation / Tampering | Metal/CUDA coverage drift | mitigate | Shared matrix and detector tests assert explicit compound supported/rejected states for both backends; docs list backend-specific Metal/CUDA coverage. | closed |
| T-12-04 | Tampering | Linear+bias+activation region selection | mitigate | Metal/CUDA lowerer and prepared execution tests assert `LINEAR_BIAS_ACTIVATION` summaries contain the `LINEAR`/activation node ids and lowered DAG node types. | closed |
| T-12-05 | Repudiation | Required GPU buffer execution | mitigate | Prepared execution required-buffer-mode test throws with `REQUIRED_BUFFER_EXECUTION_UNAVAILABLE` before hidden CPU fallback when native execution cannot satisfy a supported compound path. | closed |
| T-12-06A | Repudiation | CUDA compound support claims | mitigate | CUDA tests prove the minimal `LINEAR_BIAS_ACTIVATION` and `ELEMENTWISE_CHAIN` subset through partition/prepared DAG metadata while docs state Metal and CUDA coverage is backend-specific. | closed |
| T-12-06B | Information Disclosure / Repudiation | Elementwise chain residency | mitigate | Prepared execution and buffer-binding tests assert `ADD -> RELU -> EXP` is one accelerator compound step and interior chain nodes avoid `CPU_CONSUMER` materialization under buffer binding. | closed |
| T-12-07 | Repudiation | Runtime trace metadata | mitigate | `PreparedExecution` emits `gpuCompound*` attributes for non-`NONE` summaries; `CompiledGraphTraceTest` asserts `ELEMENTWISE_CHAIN`, node ids, DAG node types, and stable reason metadata. | closed |
| T-12-08 | Repudiation / Tampering | Native buffer fallback visibility | mitigate | Metal/CUDA buffer tests assert `BUFFER_BINDING` paths keep outputs device-owned and no CPU-consumer materialization occurs for chain interiors; AUTO fallback remains visible through accelerator buffer reason codes. | closed |
| T-12-09 | Repudiation | Reduction-adjacent rejection | mitigate | `REDUCTION_ADJACENT` detector/planner diagnostics name stable reasons for `LAYER_NORM`/`RMS_NORM`; trace tests assert rejection includes `REDUCTION_ADJACENT` and `DEFERRED_FUSED_REGION`. | closed |
| T-12-10 | Tampering | CPU fused hot path independence | mitigate | Metal/CUDA `FUSED` rejection tests pass, and source hygiene check confirms accelerator, Metal, and CUDA production packages do not import `backend.cpu.fused`. | closed |
| T-12-11 | Information Disclosure / Repudiation | Documentation claims | mitigate | `docs/gpu-lowering-coverage.md`, `docs/graph-optimizer.md`, `docs/compute-flow.md`, and `docs/development.md` state public `Tensor` remains logical, residency stays in `ExecutionState`/`DeviceBufferBinding`, and Metal/CUDA coverage is backend-specific. | closed |
| T-12-12 | Tampering | Local profile artifacts | mitigate | `12-04-SUMMARY.md` records `git status --short` and confirms `profiles/platform/.../tuning/abc/*` remained unstaged; current audit index is empty for profile paths. | closed |

Note: `T-12-06` appeared in two Phase 12 plan files for distinct risks. This audit splits them into `T-12-06A` and `T-12-06B` to keep the register unambiguous.

---

## Evidence

| Threat Ref | Evidence |
|------------|----------|
| T-12-01 | `src/main/java/backend/accelerator/lowering/GpuCompoundPatternDetector.java`, `src/main/java/backend/accelerator/lowering/GpuLoweringCoverageMatrix.java`, `src/test/java/backend/accelerator/lowering/GpuCompoundPatternDetectorTest.java`, `src/test/java/backend/accelerator/lowering/GpuLoweringCoverageMatrixTest.java` |
| T-12-02 | `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java`, `src/main/java/backend/metal/lowering/MetalRegionLegalityAdapter.java`, `src/main/java/backend/cuda/lowering/CudaGpuRegionLegalityAdapter.java` |
| T-12-03 | `src/test/java/backend/accelerator/lowering/GpuCompoundPatternDetectorTest.java`, `src/test/java/backend/accelerator/lowering/GpuLoweringCoverageMatrixTest.java`, `docs/gpu-lowering-coverage.md` |
| T-12-04 | `src/test/java/backend/accelerator/lowering/AcceleratorSubgraphLowererTest.java`, `src/test/java/backend/metal/lowering/MetalRegionLowererTest.java`, `src/test/java/backend/cuda/lowering/CudaRegionLowererTest.java`, `src/test/java/PreparedExecutionBuildTest.java` |
| T-12-05 | `src/test/java/PreparedExecutionBuildTest.java` |
| T-12-06A | `src/test/java/backend/cuda/lowering/CudaRegionLowererTest.java`, `src/test/java/backend/cuda/exec/PreparedCudaExecutableBufferPolicyTest.java`, `docs/gpu-lowering-coverage.md`, `docs/development.md` |
| T-12-06B | `src/test/java/PreparedExecutionBuildTest.java`, `src/test/java/backend/metal/exec/PreparedMetalExecutableBufferBindingTest.java`, `src/test/java/backend/cuda/exec/PreparedCudaExecutableBufferPolicyTest.java` |
| T-12-07 | `src/main/java/graph/execution/PreparedExecution.java`, `src/test/java/CompiledGraphTraceTest.java` |
| T-12-08 | `src/test/java/backend/metal/exec/PreparedMetalExecutableBufferBindingTest.java`, `src/test/java/backend/cuda/exec/PreparedCudaExecutableBufferPolicyTest.java`, `src/main/java/graph/execution/PreparedExecution.java` |
| T-12-09 | `src/main/java/backend/accelerator/lowering/GpuCompoundPatternDetector.java`, `src/main/java/backend/metal/lowering/MetalPartitionSupport.java`, `src/main/java/backend/cuda/lowering/CudaGpuRegionLegalityAdapter.java`, `src/test/java/CompiledGraphTraceTest.java` |
| T-12-10 | `src/test/java/backend/metal/lowering/MetalRegionLowererTest.java`, `src/test/java/backend/cuda/lowering/CudaRegionLowererTest.java`, `src/test/java/backend/accelerator/lowering/GpuCompoundPatternDetectorTest.java`, `rg -n "import backend.cpu.fused" src/main/java/backend/accelerator src/main/java/backend/metal src/main/java/backend/cuda` |
| T-12-11 | `docs/gpu-lowering-coverage.md`, `docs/graph-optimizer.md`, `docs/compute-flow.md`, `docs/development.md` |
| T-12-12 | `.planning/phases/12-fused-gpu-region-execution/12-04-SUMMARY.md`, `git status --short`, `git diff --cached --name-only` |

---

## Accepted Risks Log

No accepted risks.

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-04-30 | 13 | 13 | 0 | Codex inline auditor |

### Security Audit 2026-04-30

| Metric | Count |
|--------|-------|
| Threats found | 13 |
| Closed | 13 |
| Open | 0 |

Verification commands:

- `./gradlew test --tests 'backend.accelerator.lowering.*' --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` - PASS
- `rg -n "import backend.cpu.fused" src/main/java/backend/accelerator src/main/java/backend/metal src/main/java/backend/cuda` - PASS, no matches
- `git diff --cached --name-only` - PASS, empty during source hygiene check

The first unquoted shell form of the accelerator-lowering test filter was rejected by zsh glob expansion before Gradle ran. The quoted Gradle filter above passed.

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-04-30
