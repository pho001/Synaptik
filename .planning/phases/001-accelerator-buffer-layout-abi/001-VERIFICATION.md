---
phase: 001-accelerator-buffer-layout-abi
verified: 2026-04-29T19:43:25Z
status: passed
score: "10/10 must-haves verified"
overrides_applied: 0
---

# Phase 1: Accelerator Buffer Layout ABI Verification Report

**Phase Goal:** Extend the shared runtime device buffer model so Metal and future CUDA backends can describe logical tensor views, including strides and storage offsets, without baking Metal-specific assumptions into common code.
**Verified:** 2026-04-29T19:43:25Z
**Status:** passed
**Re-verification:** No - initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1 | Shared runtime buffer bindings expose backend id, native handle identity, dtype, shape, strides, storage offset, logical element count, byte length, and access mode. | VERIFIED | `DeviceBufferBinding` exposes `layout()`, `accessMode()`, `nativeHandleIdentity()`, `available()`, and default `logicalByteLength()`; `AcceleratorBufferLayout` carries dtype, shape, strides, storage offset, element count, byte length, and layout class. |
| 2 | The common ABI is backend-neutral and does not expose Metal/CUDA native handle types through common buffer records. | VERIFIED | `backend.accelerator.buffer` and `backend.memory.DeviceBufferBinding` import shared layout/access types only; no `MetalBufferHandle`, CUDA handle, `MTLBuffer`, or `MemorySegment` appears in the common ABI. `Tensor` has no accelerator/device-buffer imports. |
| 3 | Layout classification distinguishes dense contiguous, zero-offset view, non-zero-offset view, permuted/strided view, broadcast zero-stride view, and unsupported layout facts. | VERIFIED | `AcceleratorBufferLayoutClass` defines exactly those six classes; `AcceleratorBufferLayoutClassifier` classifies dense, offset, strided/permuted, broadcast zero-stride, and unsupported negative-stride/element-count mismatch cases; focused tests cover each class. |
| 4 | Stable reason codes distinguish success, fallback, unsupported dtype/layout, unavailable native ABI, and required-but-unavailable buffer execution. | VERIFIED | `AcceleratorBufferReasonCode` includes existing success/fallback/dtype codes plus `INPUT_LAYOUT_UNSUPPORTED`, `NATIVE_BUFFER_ABI_UNAVAILABLE`, and `REQUIRED_BUFFER_EXECUTION_UNAVAILABLE`; `PreparedExecution` emits `acceleratorBufferReasonCode` into traces. |
| 5 | Metal bindings adapt to the shared contract without losing backend handle or access information. | VERIFIED | `MetalBufferBinding` stores `AcceleratorBufferLayout`, `MetalBufferHandle`, and `MetalBufferAccess`; maps access to `AcceleratorBufferAccessMode`; `nativeHandleIdentity()` reports backend, owner, storage mode, and bytes without exposing the handle object. |
| 6 | Request and per-input/per-output decision records carry layout metadata through canonical signatures, with no legacy compatibility constructors. | VERIFIED | `AcceleratorBufferRequest`, `AcceleratorBufferInputDecision`, and `AcceleratorBufferOutputDecision` are records with canonical layout fields; grep found no secondary constructors/overloads/adapters for these records or `MetalBufferBinding`. Strict dtype/layout list size checks are present. |
| 7 | Metal preflight reads shared layout metadata, reports precise layout classes, and preserves Phase 1 conservative fallback for non-dense/view layouts. | VERIFIED | `MetalAcceleratorBufferBinder` derives/compares `AcceleratorBufferLayout`, rejects non-`DENSE_CONTIGUOUS` classes with `INPUT_LAYOUT_UNSUPPORTED`/`OUTPUT_LAYOUT_UNSUPPORTED`, and diagnostics include `layoutClass`, `shape`, `storageOffset`, and `strides`. No native Metal strided/view execution path was added. |
| 8 | Metal allocation/materialization compares binding layout metadata rather than duplicated dtype/shape/count fields. | VERIFIED | `MetalBufferAllocator.createOutputBinding(int, AcceleratorBufferLayout)` allocates by `layout.logicalByteLength()` and rejects non-dense outputs; `readToCpu` and `MetalDeviceToCpuMaterializer.supports` compare shape, strides, storage offset, and logical element count. |
| 9 | CUDA seams can consume the shared taxonomy while native CUDA buffers remain unavailable. | VERIFIED | `CudaGraphBridge.supportsBufferBindings()` documents the shared layout ABI contract; `PreparedCudaExecutable` fails all CUDA `REQUIRE` buffer mode with `REQUIRED_BUFFER_EXECUTION_UNAVAILABLE`, including when a fake bridge advertises buffer support. |
| 10 | Existing focused Metal/CUDA buffer tests pass, with new tests covering layout metadata compatibility and review-fix regressions. | VERIFIED | Forced targeted Gradle run passed: classifier, residency, Metal executable/binder, allocator, binding, CUDA policy, and Metal FFM bridge tests. Review-fix tests cover byte-length mismatch, strict dtype lists, CUDA advertised-buffer REQUIRE failure, and Metal bridge node-id validation. |

**Score:** 10/10 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `src/main/java/backend/accelerator/buffer/AcceleratorBufferLayout.java` | Backend-neutral layout descriptor | VERIFIED | Immutable record with defensive shape/stride copies, checked byte length, dtype byte size support for FLOAT32/FLOAT64/BFLOAT16/INT32/BOOL. |
| `src/main/java/backend/accelerator/buffer/AcceleratorBufferLayoutClassifier.java` | Pure classifier | VERIFIED | Imports tensor metadata only; no backend-specific packages; produces all required layout classes. |
| `src/main/java/backend/memory/DeviceBufferBinding.java` | Shared binding ABI | VERIFIED | Exposes layout/access/native identity and default logical bytes; no Metal/CUDA native handle imports. |
| `src/main/java/backend/metal/buffer/MetalBufferBinding.java` | Metal adapter | VERIFIED | Keeps Metal handle backend-owned while all logical metadata is read via `layout()`. No shortcut logical accessors were introduced. |
| `src/main/java/backend/accelerator/buffer/AcceleratorBufferRequest.java` | Canonical layout-aware request | VERIFIED | Carries input/output layout lists and strict dtype/layout size validation. |
| `src/main/java/backend/metal/buffer/MetalAcceleratorBufferBinder.java` | Layout-aware Metal decision seam | VERIFIED | Uses shared layouts for preflight, existing binding compatibility, allocation decisions, and native ABI unavailable reason. |
| `src/main/java/backend/metal/exec/PreparedMetalExecutable.java` | Runtime request construction and execution routing | VERIFIED | Builds request layout lists from runtime tensors; preserves binding layout when converting writable outputs to readable outputs. |
| `src/main/java/backend/cuda/exec/PreparedCudaExecutable.java` | CUDA required-unavailable policy | VERIFIED | Fails REQUIRE mode before tensor-list execution and records `REQUIRED_BUFFER_EXECUTION_UNAVAILABLE`. |
| `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java` | Native bridge validation | VERIFIED | Validates binding counts, availability, node ids, dtype, and access before buffer execution. |
| `docs/compute-flow.md` | Trace/report reason docs | VERIFIED | Documents stable `acceleratorBufferReasonCode` values and layout diagnostic fields. |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `DeviceBufferBinding` | `AcceleratorBufferLayout` | `layout()` return type | WIRED | Common runtime binding exposes layout metadata directly. |
| `MetalBufferBinding` | `DeviceBufferBinding` / `AcceleratorBufferLayout` | record field and interface implementation | WIRED | Metal binding implements the shared ABI and preserves backend-owned handle details. |
| `PreparedMetalExecutable` | `AcceleratorBufferRequest` | `new AcceleratorBufferRequest(...)` with runtime input/output layouts | WIRED | External input and output layouts are populated from runtime tensors. |
| `MetalAcceleratorBufferBinder` | `AcceleratorBufferLayout` / reason taxonomy | preflight and compatibility checks | WIRED | Decisions attach layout metadata and exact reason codes. |
| `PreparedCudaExecutable` | `AcceleratorBufferReasonCode` | required unavailable decision | WIRED | CUDA required buffer mode records and throws `REQUIRED_BUFFER_EXECUTION_UNAVAILABLE`. |
| `PreparedExecution` | trace/report attrs | `lastAcceleratorBufferDecision()` | WIRED | Trace metadata includes backend, mode, execution path, reason code, reason text, prepared input flag, and counts. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|---|---|---|---|---|
| `PreparedMetalExecutable` | `externalInputLayouts`, `outputLayouts` | `context.runtimeTensorForNodeId(nodeId)` through `AcceleratorBufferLayout.fromTensor(...)` | Yes | FLOWING |
| `MetalAcceleratorBufferBinder` | input/output decision layouts | request layouts or resolved execution tensors | Yes | FLOWING |
| `MetalBufferAllocator` | output allocation byte length | `layout.logicalByteLength()` | Yes | FLOWING |
| `PreparedExecution` | accelerator trace attributes | executable `lastAcceleratorBufferDecision()` after execution | Yes | FLOWING |
| `PreparedCudaExecutable` | last CUDA buffer decision | runtime buffer mode and bridge capability | Yes | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| Focused ABI/Metal/CUDA phase tests run against current checkout | `./gradlew test --rerun-tasks --tests backend.accelerator.buffer.AcceleratorBufferLayoutClassifierTest --tests graph.execution.ExecutionStateResidencyTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.metal.buffer.MetalBufferAllocatorTest --tests backend.metal.buffer.MetalBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest` | `BUILD SUCCESSFUL in 5s`; 4 tasks executed | PASS |
| Compile gate | `./gradlew classes` | `BUILD SUCCESSFUL`; classes up to date after focused test run compiled main sources | PASS |
| Public Tensor API remains logical | `rg` for accelerator/native buffer imports under `src/main/java/tensor` | No matches | PASS |
| Common ABI avoids native handle leakage | `rg` for Metal/CUDA handles, `MTLBuffer`, and `MemorySegment` under `backend/accelerator/buffer` and `backend/memory` | No common ABI matches | PASS |
| Compatibility shims absent | `rg` for public constructors/overloads on request/decision/binding records | Only canonical record declarations found | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|---|---|---|---|---|
| ABI-01 | 001-01, 001-02, 001-03 | Runtime device buffer bindings can represent backend id, native handle identity, dtype, shape, strides, storage offset, logical element count, byte length, and access mode. | SATISFIED | `DeviceBufferBinding`, `AcceleratorBufferLayout`, and `MetalBufferBinding` expose those fields through shared layout/access/identity APIs. |
| ABI-02 | 001-01, 001-02, 001-03 | Shared accelerator buffer model is backend-neutral and reusable by Metal now and CUDA later. | SATISFIED | Common ABI has no native handle type imports; Metal consumes it now; CUDA documents the same shared layout ABI seam and uses shared reason taxonomy. |
| ABI-03 | 001-01, 001-02, 001-03 | Compatibility checks distinguish dense, zero-offset, non-zero-offset, permuted/strided, broadcast/zero-stride, and unsupported layouts. | SATISFIED | Classifier and Metal fallback tests cover all six classes; Metal diagnostics report layout class and layout facts. |
| ABI-04 | 001-01, 001-02, 001-03 | Buffer binding decisions expose stable reason codes for success, fallback, unsupported dtype/layout, unavailable native ABI, and required-but-unavailable execution. | SATISFIED | Reason enum, Metal/CUDA decisions, trace metadata, and docs cover the stable taxonomy. |

No additional Phase 1 requirements were orphaned in `.planning/REQUIREMENTS.md`.

### Code Review Fix Verification

| Finding | Status | Evidence |
|---|---|---|
| WR-01: direct layout construction could understate byte length | FIXED | `AcceleratorBufferLayout` compares constructor `logicalByteLength` with `byteLength(dataType, logicalElementCount)` and tests assert mismatch rejection. |
| WR-02: CUDA REQUIRE could still execute tensor-list path if bridge advertised buffers | FIXED | `PreparedCudaExecutable` fails all REQUIRE buffer mode before bridge execution; test with `FakeCudaBridge(true)` asserts tensor-list execution is not entered. |
| WR-03: Metal bridge did not validate binding node ids | FIXED | `MetalMpsFfmBridge.validateBufferBindings` checks input/output node ids; tests assert mismatched ids fail. |
| WR-04: request ABI allowed missing dtype lists | FIXED | `AcceleratorBufferRequest` requires dtype list sizes to match node id list sizes; tests assert both missing input and output dtype lists fail. |
| IN-01: docs promised `shape` but runtime omitted it | FIXED | `MetalAcceleratorBufferBinder.unsupportedBufferLayoutReason` includes `shape=` and docs still name `shape`. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---:|---|---|---|
| `src/main/java/backend/accelerator/buffer/AcceleratorBufferExecutionPath.java` | 23 | "not available" in enum documentation | INFO | Documentation only; not a stub or placeholder. |

### Human Verification Required

None. This phase is an internal Java ABI/runtime-contract change with focused automated coverage and no visual, interactive, or external-service behavior requiring manual UAT.

### Gaps Summary

No blocking gaps found. Residual risks are scoped to later phases: native Metal still rejects view/strided layouts by design, native CUDA buffers remain unavailable by design, and existing local profile tuning changes plus `.planning/tmp/` remain dirty/untracked but were pre-existing project hygiene items outside this phase's code changes.

---

_Verified: 2026-04-29T19:43:25Z_
_Verifier: Claude (gsd-verifier)_
