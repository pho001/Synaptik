# Phase 08 Pattern Map

## Closest Analogs

| Target | Closest Analog | Pattern To Reuse |
|--------|----------------|------------------|
| `src/main/java/backend/cuda/bridge/CudaBridgeExecutionStats.java` | `src/main/java/backend/metal/bridge/MetalMpsBridgeExecutionStats.java` | Immutable record with fallback factory and Java-observed timing fields. |
| `src/main/java/backend/cuda/exec/PreparedCudaExecutable.java` | `src/main/java/backend/metal/exec/PreparedMetalExecutable.java` | Keep last execution diagnostics on the executable, publish fallback stats before CPU replay, and preserve required-mode throws. |
| `src/main/java/graph/execution/PreparedExecution.java` | Existing Metal block in `buildStepMetadata(...)` | Add backend-specific trace attributes after common `acceleratorBuffer*` attributes without changing public Tensor API. |
| `src/main/java/tuning/benchmark/report/AcceleratorTraceSummary.java` | Existing backend-neutral accelerator summary | Prefer backend-neutral byte/copy attributes and fall back to `metal*` attributes for compatibility. |
| `src/test/java/BenchmarkSessionTest.java` | `benchmarkSessionReportsAcceleratorEvidenceContract` and `benchmarkSessionReportsMetalMaterializationDetails` | Synthetic trace/report assertions for text and JSON output. |
| `src/test/java/backend/cuda/exec/PreparedCudaExecutableBufferPolicyTest.java` | Existing fake CUDA bridge tests | Extend fake bridge tests to assert exact reason-code and fallback path behavior without CUDA hardware. |
| `src/test/java/SourceTreeHygieneTest.java` | Existing `.planning/tmp`, `.class`, CUDA native output, benchmark persistence checks | Add narrowly scoped checks for CUDA scratch/profile artifact boundaries. |

## Concrete File Targets

- `src/main/java/backend/cuda/bridge/CudaBridgeExecutionStats.java`
- `src/main/java/backend/cuda/exec/PreparedCudaExecutable.java`
- `src/main/java/graph/execution/PreparedExecution.java`
- `src/main/java/tuning/benchmark/report/AcceleratorTraceSummary.java`
- `src/test/java/BenchmarkSessionTest.java`
- `src/test/java/backend/cuda/exec/PreparedCudaExecutableBufferPolicyTest.java`
- `src/test/java/backend/cuda/buffer/CudaAcceleratorBufferBinderTest.java`
- `src/test/java/SourceTreeHygieneTest.java`
- `docs/architecture.md`
- `docs/compute-flow.md`
- `docs/development.md`
- `docs/configuration.md`
- `docs/testing.md`
- `docs/troubleshooting.md`

## Implementation Notes

- Do not add public device tensor APIs.
- Do not move CUDA handles outside `backend.cuda.*`.
- Keep report aggregation backend-neutral where possible; add CUDA-specific attrs only for detailed trace fields.
- Preserve Metal report output strings and tests while adding CUDA parity.
- Native CUDA gate remains optional and capability-gated.
