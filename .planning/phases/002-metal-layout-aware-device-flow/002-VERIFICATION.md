---
phase: 002-metal-layout-aware-device-flow
verified: 2026-04-30T05:05:00Z
status: passed
score: "12/12 must-haves verified"
overrides_applied: 0
---

# Phase 2: Metal Layout-Aware Device Flow Verification Report

**Phase Goal:** Teach Metal buffer execution to preserve legal device-owned view/layout values and avoid falling back only because an intermediate output is non-contiguous or non-zero-offset.
**Verified:** 2026-04-30T05:05:00Z
**Status:** passed

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1 | Metal buffer preflight classifies layout classes instead of rejecting all non-contiguous or offset outputs blindly. | VERIFIED | `MetalLayoutPolicy` is exercised by 002-01 tests; docs and trace tests show legal logical-view classes and explicit rejected classes. |
| 2 | Legal view-like Metal outputs can use dense physical logical-view device buffers. | VERIFIED | 002-02 allocator/materializer tests cover `ZERO_OFFSET_VIEW`, `NON_ZERO_OFFSET_VIEW`, and `PERMUTED_OR_STRIDED_VIEW` allocation/materialization through dense physical buffers. |
| 3 | `LINEAR -> RESHAPE -> PERMUTE` style flows have layout-aware Metal coverage. | VERIFIED | `MetalLayoutAwareDeviceFlowTest` and `MetalBufferTraceSmokeTest` contain layout-heavy `matmul/linear -> reshape -> permute` graphs and pass targeted test gates. |
| 4 | Native success-capable layout flow reports buffer execution and device-owned residency before CPU publication. | VERIFIED | Trace tests assert `metalExecutionPath=BUFFER_BINDING`, `storageResidency=DEVICE_OWNED`, `storageCpuCurrent=false`, and `storageDeviceCurrent=true`. |
| 5 | Unsupported broadcast zero-stride layout rejection remains visible. | VERIFIED | Tests assert `OUTPUT_LAYOUT_UNSUPPORTED` or `INPUT_LAYOUT_UNSUPPORTED` and `layoutClass=BROADCAST_ZERO_STRIDE_VIEW`/`UNSUPPORTED` in trace diagnostics. |
| 6 | CPU parity is covered for representative forward layout-aware flow. | VERIFIED | `linearReshapePermuteMatchesCpuForwardResult` compares CPU and Metal outputs with `assertArrayEquals`. |
| 7 | Forward-backward graph publishes gradients correctly or exposes visible fallback. | VERIFIED | `forwardBackwardLayoutAwareGraphPublishesGradientsWithCpuParity` compares input and weight gradients to CPU baseline and asserts `GRADIENT_PUBLICATION` or visible accelerator reason code. |
| 8 | Graph output, CPU consumer, and gradient publication materialization boundaries are observable. | VERIFIED | `MetalBufferTraceSmokeTest` covers graph output and CPU consumer reasons; `MetalLayoutAwareDeviceFlowTest` covers gradient publication/fallback visibility. |
| 9 | No native layout ABI was added in Phase 2. | VERIFIED | Docs state the no-native-layout-ABI contract and bridge tests from 002-02 prove the existing buffer ABI remains shape/dtype/access based. |
| 10 | Future native layout ABI changes must be optional-symbol/version/capability checked. | VERIFIED | `docs/metal-backend.md` documents optional-symbol/version/capability requirements for future native layout ABI additions. |
| 11 | Accelerator fallback prepared inputs are safe for native input preparation. | VERIFIED | `PreparedAcceleratorExecutable.cpuFallbackSteps()` plus `ExecutionState` allocation fixed full `metalTest` attention-slice failures. |
| 12 | Profile/tuning scratch artifacts and `.planning/tmp/` were not included in Phase 2 commits. | VERIFIED | `git status --short` shows these as unstaged/untracked only; phase commits exclude them. |

**Score:** 12/12 truths verified

## Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `src/main/java/backend/metal/buffer/MetalLayoutPolicy.java` | Metal layout policy classification | VERIFIED | Used by binder/materializer policy gates from 002-01/002-02. |
| `src/main/java/backend/metal/buffer/MetalBufferAllocator.java` | Dense physical logical-view allocation/readback | VERIFIED | 002-02 tests cover logical-view output allocation and scatter materialization. |
| `src/main/java/backend/metal/buffer/MetalDeviceToCpuMaterializer.java` | Device-to-CPU materialization support gates | VERIFIED | Supports graph output, CPU consumer, and gradient publication when binding layout/policy allow it. |
| `src/main/java/backend/accelerator/exec/PreparedAcceleratorExecutable.java` | Backend-neutral fallback metadata exposure | VERIFIED | New default `cpuFallbackSteps()` keeps per-run prepared input allocation backend-neutral. |
| `src/main/java/graph/execution/ExecutionState.java` | Runtime residency and prepared input state | VERIFIED | Allocates prepared inputs for accelerator fallback steps and marks safe CPU alias views current. |
| `src/main/java/graph/execution/PreparedExecution.java` | Root/gradient publication | VERIFIED | Handles stale semantic alias publication by publishing the actual materializable runtime root. |
| `src/test/java/backend/metal/MetalLayoutAwareDeviceFlowTest.java` | E2E CPU parity and gradient publication coverage | VERIFIED | New tests pass in targeted Java gate and native `metalTest`. |
| `src/test/java/backend/metal/MetalBufferTraceSmokeTest.java` | Trace visibility coverage | VERIFIED | New layout-aware trace tests pass and assert path, residency, materialization, and fallback metadata. |
| `docs/compute-flow.md` | Runtime trace/materialization docs | VERIFIED | Documents `DENSE_PHYSICAL_LOGICAL_VIEW`, reason codes, materialization reasons, and trace fields. |
| `docs/metal-backend.md` | Phase 2 Metal layout policy docs | VERIFIED | Documents legal/rejected layout classes and no-native-layout-ABI stance. |
| `docs/testing.md` | Focused verification commands | VERIFIED | Lists Phase 2 targeted tests, `classes`, and `metalTest`. |

## Key Link Verification

| From | To | Via | Status |
|---|---|---|---|
| `MetalLayoutAwareDeviceFlowTest` | `PreparedMetalExecutable` | `CompiledGraph.compile(...).prepare(...).execute(...)` | WIRED |
| `MetalBufferTraceSmokeTest` | `RunTrace` / `CpuMaterializationTrace` | trace metadata and materialization reason assertions | WIRED |
| `PreparedMetalExecutable` | `AcceleratorPreparedInputResolver` | fallback CPU layout plans for native external inputs | WIRED |
| `ExecutionState` | accelerator fallback metadata | `acceleratorExecutable().cpuFallbackSteps()` prepared-input allocation | WIRED |
| `docs/metal-backend.md` | `MetalLayoutPolicy` | documented policy states and rejected classes | WIRED |

## Behavioral Checks

| Command | Result | Status |
|---|---|---|
| `./gradlew test --tests backend.metal.MetalLayoutAwareDeviceFlowTest` | Passed | PASS |
| `./gradlew test --tests backend.metal.MetalBufferTraceSmokeTest` | Passed | PASS |
| `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.metal.buffer.MetalBufferAllocatorTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.metal.MetalBufferTraceSmokeTest --tests backend.metal.MetalLayoutAwareDeviceFlowTest` | Passed | PASS |
| `./gradlew classes` | Passed | PASS |
| `./gradlew metalTest --tests PreparedExecutionBuildTest.gpuMetalAttentionLikeRank4SliceCanExecuteThroughExplicitAppleShim --tests PreparedExecutionBuildTest.gpuMetalMaskedAttentionPreSoftmaxSliceCanExecuteThroughExplicitAppleShim --tests PreparedExecutionBuildTest.gpuMetalMaskedAttentionSoftmaxSliceCanExecuteThroughExplicitAppleShim --tests PreparedExecutionBuildTest.gpuMetalMaskedAttentionFullForwardSliceCanExecuteThroughExplicitAppleShim` | Passed | PASS |
| `./gradlew metalTest` | Passed | PASS |
| Phase 1 regression gate: `./gradlew test --tests backend.accelerator.buffer.AcceleratorBufferLayoutClassifierTest --tests graph.execution.ExecutionStateResidencyTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.metal.buffer.MetalBufferAllocatorTest --tests backend.metal.buffer.MetalBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest` | Passed | PASS |
| Required acceptance greps from `002-03-PLAN.md` | Passed | PASS |

## Requirements Coverage

| Requirement | Status | Evidence |
|---|---|---|
| METAL-01 | SATISFIED | Policy, binder, allocator, and trace tests prove legal view-like values no longer fall back solely because they are view-like. |
| METAL-02 | SATISFIED | Layout-heavy forward tests cover `linearReshapePermute` style graphs with CPU parity and device-owned trace assertions. |
| METAL-03 | SATISFIED | Unsupported layout rejection, native buffer ABI availability checks, no-native-layout-ABI docs, and reason-code tests remain conservative. |
| METAL-04 | SATISFIED | Materializer tests and E2E trace/parity tests cover graph output, CPU consumer, and gradient publication boundaries. |

## Code Review

| Review | Status | Notes |
|---|---|---|
| `002-REVIEW.md` | CLEAN | Standard inline review found 0 critical, 0 warning, and 0 info findings. |

## Human Verification Required

None. Phase 2 is internal runtime/test/docs work with automated Java and native Metal verification.

## Gaps Summary

No blocking gaps found.

Residual risks intentionally deferred to later phases:

- Native buffer publication parity for one layout-heavy fixture still uses tensor-array execution in the parity assertion; the trace path separately proves native buffer selection and residency.
- Region planning profitability and materialization-aware offload selection are Phase 3 scope.
- Local profile tuning files and `.planning/tmp/` remain dirty/untracked but were not part of Phase 2 output.

---
_Verified: 2026-04-30T05:05:00Z_
_Verifier: Codex (inline gsd-verifier fallback)_
