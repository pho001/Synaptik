---
phase: 07
slug: cuda-buffer-execution-and-materialization
status: verified
threats_open: 0
asvs_level: 1
created: 2026-04-30
---

# Phase 07 - Security

Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Java runtime to CUDA native FFM bridge | Java execution code calls optional native CUDA buffer ABI through resolved FFM symbols. | Buffer handles, addresses, byte lengths, execution descriptors, native symbol availability |
| ExecutionState residency and materialization | Runtime state promotes CUDA-owned buffers and materializes them back into CPU-readable tensor storage. | Runtime tensors, device buffer bindings, storage residency markers, CPU arrays |
| Optional CUDA native build/test environment | Native CUDA verification depends on local `nvcc`, CUDA runtime, and hardware availability. | Build outputs under `build/native/cuda`, CUDA test execution capability |
| Developer documentation and reporting | Docs and validation artifacts communicate the supported CUDA scope and fallback expectations. | Coverage claims, fallback reasons, benchmark/report guidance |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-7-01 | Spoofing | CUDA FFM bridge capability detection | mitigate | `CudaFfmBridge.supportsBufferBindings()` gates support on the Java enablement flag and all required create, read, destroy, and execute-buffer symbols. Evidence: `src/main/java/backend/cuda/bridge/CudaFfmBridge.java:66`, `src/main/java/backend/cuda/bridge/CudaFfmBridge.java:525`, `src/main/java/backend/cuda/bridge/CudaFfmBridge.java:603`. | closed |
| T-7-02 | Tampering | CUDA device-to-CPU materialization | mitigate | `CudaDeviceToCpuMaterializer.materialize(...)` delegates to allocator read-back, and `CudaBufferAllocator.readToCpu(...)` performs native read before copying bytes into CPU storage. Evidence: `src/main/java/backend/cuda/buffer/CudaDeviceToCpuMaterializer.java:43`, `src/main/java/backend/cuda/buffer/CudaBufferAllocator.java:97`, `src/main/java/graph/execution/ExecutionState.java:494`. | closed |
| T-7-03 | Denial of service | CUDA native buffer resource lifecycle | mitigate | Native buffer handles are wrapped in `CudaBufferResource` and registered with the execution context for input and output allocations. Evidence: `src/main/java/backend/cuda/buffer/CudaBufferResource.java`, `src/main/java/backend/cuda/buffer/CudaAcceleratorBufferBinder.java:353`, `src/main/java/backend/cuda/buffer/CudaAcceleratorBufferBinder.java:387`. | closed |
| T-7-04 | Elevation of privilege | REQUIRED CUDA execution policy | mitigate | Buffer execution failures emit `NATIVE_BUFFER_EXECUTION_FAILED` and `requireBufferOrThrow(...)` throws before tensor-array or CPU fallback can satisfy REQUIRED mode. Evidence: `src/main/java/backend/cuda/exec/PreparedCudaExecutable.java:156`, `src/main/java/backend/cuda/exec/PreparedCudaExecutable.java:297`. | closed |
| T-7-05 | Tampering | CUDA buffer output publication | mitigate | Output bindings are reserved before execution and promoted to `DEVICE_OWNED` only after `executeBuffers(...)` succeeds. Evidence: `src/main/java/backend/cuda/buffer/CudaAcceleratorBufferBinder.java:386`, `src/main/java/backend/cuda/exec/PreparedCudaExecutable.java:154`, `src/main/java/backend/cuda/exec/PreparedCudaExecutable.java:318`. | closed |
| T-7-06 | Tampering | CPU output publication from CUDA buffers | mitigate | Graph output paths use `requireCpuReadable(..., GRAPH_OUTPUT)` and tests assert materialized CPU data matches the CPU oracle. Evidence: `src/main/java/graph/execution/ExecutionState.java:494`, `src/test/java/backend/cuda/exec/PreparedCudaExecutableBufferPolicyTest.java:76`. | closed |
| T-7-07 | Tampering | Adjacent CUDA region buffer handoff | mitigate | Compatible binding checks reject unavailable handles, non-CUDA backends, mismatched dtype/layout/shape/strides/storage offset, and unreadable access; tests cover reuse and rejection paths. Evidence: `src/main/java/backend/cuda/buffer/CudaAcceleratorBufferBinder.java:175`, `src/main/java/backend/cuda/buffer/CudaAcceleratorBufferBinder.java:423`, `src/test/java/backend/cuda/exec/PreparedCudaExecutableBufferPolicyTest.java:148`. | closed |
| T-7-08 | Repudiation | CUDA coverage documentation | mitigate | Project docs explicitly describe Phase 7 as narrow dense FLOAT32 CUDA buffer execution and state unsupported layouts/dtypes fall back visibly. Evidence: `docs/architecture.md:309`, `docs/architecture.md:453`, `docs/development.md:194`, `docs/configuration.md:349`. | closed |
| T-7-09 | Denial of service | Native CUDA verification gate | mitigate | Native CUDA tests are optional and capability-gated by `hasNvcc()`, while portable Gradle tests remain the required gate. Evidence: `build.gradle:100`, `build.gradle:119`, `build.gradle:160`, `docs/development.md:199`. | closed |

Status: closed.
Disposition: mitigate.

---

## Accepted Risks Log

No accepted risks.

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-04-30 | 9 | 9 | 0 | Codex inline auditor |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-04-30
