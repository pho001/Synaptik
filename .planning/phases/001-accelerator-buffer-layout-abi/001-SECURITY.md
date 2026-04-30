---
phase: 001
slug: accelerator-buffer-layout-abi
status: verified
threats_open: 0
asvs_level: 1
created: 2026-04-30
---

# Phase 001 - Security

Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Runtime tensor metadata -> accelerator layout ABI | Mutable runtime tensor layout facts become backend-neutral buffer metadata consumed by native-capable backends. | dtype, shape, strides, storage offset, logical element count, byte length |
| Backend binding -> ExecutionState | Backend-specific binding objects cross into shared per-run runtime state through `DeviceBufferBinding`. | backend id, layout, access mode, diagnostic native identity |
| Native handle diagnostics -> common ABI | Backend-native handle identity is represented as a diagnostic token, not as an executable common handle. | opaque diagnostic identity strings |
| ExecutionContext -> Metal binder | Runtime tensors and prepared inputs become binding requests and backend preflight decisions. | input/output layouts, existing bindings, access requirements |
| Metal buffer binding -> native bridge | Java-side binding metadata controls whether native buffer execution is invoked. | buffer handles, layout decisions, native ABI availability |
| CUDA policy -> fallback path | Required buffer mode can fail before tensor-list execution. | CUDA bridge capability and required buffer execution decision |
| Tests/docs/local verification -> repository state | Tests and docs describe diagnostic behavior and produce local build artifacts. | reason-code names, layout details, targeted Gradle commands |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-001-01-01 | Tampering | `AcceleratorBufferLayout` constructor | mitigate | `AcceleratorBufferLayout` validates non-null dtype/shape/strides/class, shape-stride length equality, non-negative offset/count/bytes, checked dtype byte length, and defensive array copies. | closed |
| T-001-01-02 | Information Disclosure | `DeviceBufferBinding.nativeHandleIdentity()` | mitigate | `DeviceBufferBinding` exposes only a string diagnostic identity; `MetalBufferBinding.nativeHandleIdentity()` emits backend/owner/storage/byte metadata rather than native handle objects, and the common ABI packages have no Metal/CUDA/FFM imports. | closed |
| T-001-01-03 | Denial of Service | `logicalByteLength` calculation | mitigate | `AcceleratorBufferLayout.byteLength(...)` uses `Math.multiplyExact(logicalElementCount, bytesPerElement(dataType))` and rejects negative element counts. | closed |
| T-001-01-04 | Repudiation | `AcceleratorBufferReasonCode` | mitigate | Stable reason-code names are preserved in `AcceleratorBufferReasonCode`, asserted in Metal/CUDA tests, and documented in `docs/compute-flow.md`. | closed |
| T-001-02-01 | Tampering | `MetalAcceleratorBufferBinder.incompatibleBindingReason` | mitigate | Existing Metal bindings are accepted only after comparing dtype, shape, strides, storage offset, logical element count, and access compatibility against the expected layout. | closed |
| T-001-02-02 | Denial of Service | `MetalBufferAllocator.createOutputBinding` | mitigate | Output allocation rejects unsupported layout classes before native allocation and allocates from checked `layout.logicalByteLength()`. | closed |
| T-001-02-03 | Repudiation | `PreparedCudaExecutable` required mode | mitigate | Required CUDA buffer mode records `REQUIRED_BUFFER_EXECUTION_UNAVAILABLE` and throws before optional tensor-list fallback can be confused with native buffer execution. | closed |
| T-001-02-04 | Elevation of Privilege | `CudaGraphBridge.supportsBufferBindings()` | accept | Accepted risk documented below: local native bridge SPI has no auth surface; Javadoc contract requires backend-owned native handles and a concrete device pointer/lifetime contract before returning true. | closed |
| T-001-03-01 | Repudiation | `PreparedMetalExecutableBufferBindingTest` | mitigate | Tests assert exact reason-code names and `layoutClass=`, `storageOffset=`, and `strides=` details for layout fallback. | closed |
| T-001-03-02 | Tampering | Output binding promotion tests | mitigate | Tests assert unsupported or failed native buffer paths do not execute buffer/tensor paths incorrectly and do not promote reserved output bindings. | closed |
| T-001-03-03 | Information Disclosure | `docs/compute-flow.md` | accept | Accepted risk documented below: docs list diagnostic field names and reason codes, not native handle values or sensitive runtime data. | closed |
| T-001-03-04 | Denial of Service | Verification commands | mitigate | Phase 1 validation and summaries use targeted Gradle filters instead of the full default `test` suite. | closed |

*Status: open / closed*
*Disposition: mitigate (implementation required) / accept (documented risk) / transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-001-01 | T-001-02-04 | `CudaGraphBridge.supportsBufferBindings()` is an internal local native bridge SPI with no auth boundary. The default is false, and implementations must only return true when they can consume the shared layout ABI with backend-owned native handles and a concrete lifetime contract. | Codex | 2026-04-30 |
| AR-001-02 | T-001-03-03 | `docs/compute-flow.md` exposes diagnostic field names and stable reason codes only; it does not publish native handle values or sensitive runtime data. | Codex | 2026-04-30 |

---

## Summary Threat Flags

All Phase 1 summaries reported no additional threat flags beyond the plan threat models.

---

## Security Audit 2026-04-30

| Metric | Count |
|--------|-------|
| Threats found | 12 |
| Closed | 12 |
| Open | 0 |

## Evidence

| Threats | Evidence | Result |
|---------|----------|--------|
| T-001-01-01, T-001-01-03 | `AcceleratorBufferLayout.java:21-44` validates required fields and copies arrays; `AcceleratorBufferLayout.java:74-80` uses checked byte-length multiplication. | PASS |
| T-001-01-02 | `DeviceBufferBinding.java:43-48` exposes an opaque diagnostic identity string; `MetalBufferBinding.java:73-79` returns backend-owned diagnostic metadata; `rg` found no Metal/CUDA/FFM imports in `backend/memory` or `backend/accelerator/buffer`. | PASS |
| T-001-01-04, T-001-03-01 | `PreparedMetalExecutableBufferBindingTest` and `PreparedCudaExecutableBufferPolicyTest` assert `INPUT_LAYOUT_UNSUPPORTED`, `OUTPUT_LAYOUT_UNSUPPORTED`, `NATIVE_BUFFER_ABI_UNAVAILABLE`, `REQUIRED_BUFFER_EXECUTION_UNAVAILABLE`, and layout diagnostic details; `docs/compute-flow.md:1900-1917` documents stable fields and reason codes. | PASS |
| T-001-02-01 | `MetalAcceleratorBufferBinder.java:327-362` compares binding availability, dtype, shape, strides, storage offset, logical element count, and access before reusing a binding. | PASS |
| T-001-02-02 | `MetalBufferAllocator.java:144-159` rejects unsupported output layouts before native allocation and uses `layout.logicalByteLength()` for native buffer allocation; `MetalBufferAllocator.java:271-275` rejects broadcast/unsupported output allocation. | PASS |
| T-001-02-03 | `PreparedCudaExecutable.java:89-105` records `REQUIRED_BUFFER_EXECUTION_UNAVAILABLE` and throws for required CUDA buffer execution; `PreparedCudaExecutableBufferPolicyTest.java:28-78` asserts both unsupported CUDA required paths. | PASS |
| T-001-02-04 | Accepted risk `AR-001-01`; `CudaGraphBridge.java:49-57` defaults `supportsBufferBindings()` to false and documents backend-owned native handle/lifetime requirements. | PASS |
| T-001-03-02 | `PreparedMetalExecutableBufferBindingTest.java:936-955` verifies unsupported layout fallback leaves no output device binding; `PreparedMetalExecutableBufferBindingTest.java:1001-1012` verifies required unsupported layouts throw before buffer allocation or execution. | PASS |
| T-001-03-03 | Accepted risk `AR-001-02`; `docs/compute-flow.md:1898-1917` lists diagnostic field names and reason-code meanings without native handle values. | PASS |
| T-001-03-04 | `001-VALIDATION.md` and Phase 1 summaries list targeted Gradle filters for layout classifier, Metal buffer, residency, allocator, and CUDA policy tests. | PASS |
| All mitigate threats | `./gradlew test --tests backend.accelerator.buffer.AcceleratorBufferLayoutClassifierTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.metal.buffer.MetalBufferAllocatorTest --tests graph.execution.ExecutionStateResidencyTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` | PASS |

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-04-30 | 12 | 12 | 0 | Codex |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-04-30
