---
phase: 10
slug: gpu-layout-transform-and-view-path
status: verified
threats_open: 0
asvs_level: 1
created: 2026-04-30
---

# Phase 10 - Security

Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Shared layout transform planner | Backend-neutral planner classifies GPU view/materialization candidates before backend execution. | Shape, strides, storage offset, dtype, backend id, reason codes |
| Execution state and metadata-only views | Prepared execution can attach target device bindings before CPU input materialization. | Runtime node ids, storage residency, borrowed Metal/CUDA handles |
| Metal/CUDA native layout materialization | Optional native symbols perform dense layout materialization when capability and run-scoped service are available. | Native buffer handles, shape/stride arrays, storage offset, logical element count, physical byte span |
| Trace and documentation reporting | Runtime traces and docs communicate GPU residency, layout transform kind, fallback, and CUDA scope. | Execution metadata, CPU materialization traces, benchmark/report guidance |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-10-01 | Repudiation | GPU layout transform decisions | mitigate | Layout decisions use explicit transform kinds and stable reason codes for metadata-only views, dense GPU materialization, source-binding gaps, backend mismatch, unsupported metadata, and unsupported transforms. Evidence: `src/main/java/backend/accelerator/buffer/AcceleratorLayoutTransformPlanner.java:15`, `src/main/java/backend/accelerator/buffer/AcceleratorLayoutTransformPlanner.java:40`, `src/main/java/backend/accelerator/buffer/AcceleratorLayoutTransformPlanner.java:51`, `src/test/java/backend/accelerator/buffer/AcceleratorLayoutTransformPlannerTest.java:23`. | closed |
| T-10-02 | Information disclosure | Shared GPU layout request/decision records | mitigate | Shared records and descriptors carry backend-neutral layout metadata and opaque handle identity, not backend handle objects; layout ABI v2 validates rank, shape, strides, storage offset, byte lengths, and physical span. Evidence: `src/main/java/backend/accelerator/buffer/AcceleratorLayoutAbiV2Descriptor.java:11`, `src/main/java/backend/accelerator/buffer/AcceleratorLayoutAbiV2Descriptor.java:65`, `src/main/java/backend/accelerator/buffer/AcceleratorLayoutAbiV2Descriptor.java:92`. | closed |
| T-10-03 | Tampering | Metal/CUDA metadata-only alias bindings | mitigate | `viewOf(...)` creates borrowed-handle bindings over the source handle without new resource ownership, and tests assert the target view reuses the source native handle identity for both Metal and CUDA. Evidence: `src/main/java/backend/metal/buffer/MetalBufferBinding.java:35`, `src/main/java/backend/metal/buffer/MetalBufferBinding.java:47`, `src/main/java/backend/cuda/buffer/CudaBufferBinding.java:30`, `src/test/java/backend/metal/buffer/MetalBufferBindingTest.java:99`, `src/test/java/backend/cuda/buffer/CudaBufferBindingTest.java:15`. | closed |
| T-10-04 | Elevation of privilege | Pre-CPU-step device layout propagation | mitigate | `PreparedExecution` invokes `DeviceLayoutViewPropagator` before `requireCpuReadableInputs(...)`; rejected REQUIRED-mode propagation throws before CPU fallback, and host-shared view residency stays CPU-readable to avoid corrupting overlapping storage. Evidence: `src/main/java/graph/execution/PreparedExecution.java:265`, `src/main/java/graph/execution/PreparedExecution.java:271`, `src/main/java/graph/execution/DeviceLayoutViewPropagator.java:63`, `src/main/java/graph/execution/DeviceLayoutViewPropagator.java:123`, `src/test/java/graph/execution/DeviceLayoutViewPropagationTest.java:48`, `src/test/java/graph/execution/DeviceLayoutViewPropagationTest.java:98`. | closed |
| T-10-05 | Denial of service | Optional native layout materialization symbols | mitigate | Metal and CUDA layout materialization is separately capability-gated; missing symbols raise visible unsupported-layout errors and do not gate existing buffer execution. Evidence: `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java:332`, `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java:397`, `src/main/java/backend/cuda/bridge/CudaFfmBridge.java:79`, `src/main/java/backend/cuda/bridge/CudaFfmBridge.java:404`, `src/test/java/backend/metal/exec/PreparedMetalExecutableBufferBindingTest.java:101`, `src/test/java/backend/cuda/exec/PreparedCudaExecutableBufferPolicyTest.java:103`. | closed |
| T-10-06 | Tampering | Dense GPU layout materialization metadata | mitigate | Native materialization calls pass rank, shape, strides, storage offset, logical element count, physical byte span, and destination byte length from `AcceleratorLayoutAbiV2Descriptor`, preventing logical-byte-only copies for strided or offset views. Evidence: `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java:406`, `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java:410`, `src/main/java/backend/cuda/bridge/CudaFfmBridge.java:413`, `src/main/java/backend/cuda/bridge/CudaFfmBridge.java:417`, `src/main/java/backend/accelerator/buffer/AcceleratorLayoutAbiV2Descriptor.java:56`. | closed |
| T-10-07 | Repudiation | Trace evidence for GPU residency and CPU materialization | mitigate | Trace metadata records layout transform kind, accelerator reason code, storage residency, device buffer attributes, and CPU materialization boundaries; E2E tests assert metadata-only CUDA views avoid intermediate CPU consumer materialization and unsupported paths fall back visibly. Evidence: `src/main/java/graph/execution/PreparedExecution.java:432`, `src/main/java/graph/execution/PreparedExecution.java:443`, `src/main/java/graph/execution/PreparedExecution.java:496`, `src/test/java/backend/cuda/exec/CudaLayoutTransformDeviceFlowTest.java:66`, `src/test/java/backend/cuda/exec/CudaLayoutTransformDeviceFlowTest.java:118`, `src/test/java/backend/metal/MetalLayoutAwareDeviceFlowTest.java:155`, `src/test/java/backend/metal/MetalLayoutAwareDeviceFlowTest.java:218`. | closed |
| T-10-08 | Repudiation | CUDA non-dense layout support claims | mitigate | Documentation explicitly states direct non-dense CUDA compute remains conservative until Phase 11 and must preserve metadata-only views, use explicit dense materialization, or fall back with visible reason codes. Evidence: `docs/development.md:214`, `docs/native-bridges-and-blas.md:1049`, `docs/native-bridges-and-blas.md:1067`, `.planning/phases/10-gpu-layout-transform-and-view-path/10-04-SUMMARY.md:17`. | closed |

Status: closed.
Disposition: mitigate.

---

## Accepted Risks Log

No accepted risks.

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-04-30 | 8 | 8 | 0 | Codex inline auditor |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-04-30
