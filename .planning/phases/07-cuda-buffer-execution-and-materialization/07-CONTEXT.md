# Phase 7: CUDA Buffer Execution And Materialization - Context

**Gathered:** 2026-04-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 7 turns the Phase 6 CUDA capability seam into a narrow but real native buffer execution path. It must prove that CUDA can allocate native device buffers, execute at least one representative dense `FLOAT32` accelerator operation through the native buffer ABI, materialize CUDA-owned results back to CPU storage at graph-output and CPU-consumer boundaries, and hand device-owned CUDA buffers from one adjacent CUDA region to the next without Java array round trips when the shared layout/capability contract permits it.

This phase does not need broad CUDA operation coverage, benchmark/report parity, final CUDA documentation closure, public device tensor APIs, or persisted profile/calibration changes. Those remain Phase 8 or future milestone work.

</domain>

<decisions>
## Implementation Decisions

### Native Buffer Execution Scope
- **D-01:** Start with a narrow dense `FLOAT32` native buffer contract. `FLOAT32` data buffers and any already-supported BOOL predicate inputs may be read through existing DAG metadata, but CUDA-owned output buffers in this phase should be dense `FLOAT32`.
- **D-02:** The representative native CUDA operation should be the smallest operation that proves the full buffer path end to end. Prefer `RELU` or simple elementwise `ADD` over matmul/linear if that minimizes native complexity while still exercising external inputs, native execution, output binding, and CPU parity.
- **D-03:** Keep unsupported CUDA DAG node types visibly rejected or falling back. Do not broaden the legality adapter's claimed production coverage beyond what the native shim and Java buffer executor can actually run through buffers.
- **D-04:** Enable `CudaFfmBridge.supportsBufferBindings()` only after the Java side and native shim expose the full create/read/destroy/execute-buffer symbol set required for safe dense `FLOAT32` buffer execution.

### Native ABI And Resource Ownership
- **D-05:** Add CUDA buffer ABI functions analogous to the Metal buffer ABI, but scoped to CUDA-owned device memory: create/upload buffer, read buffer to CPU, destroy buffer, and execute partition with buffer handles.
- **D-06:** CUDA native handles, buffer access enums, allocator/materializer classes, and resource lifetime management stay under `backend.cuda.*`. Shared records such as `AcceleratorBufferRequest`, `AcceleratorBufferDecision`, and `DeviceBufferBinding` remain backend-neutral.
- **D-07:** Run-scoped CUDA resources must be registered with `ExecutionState.registerResource(...)` and released through the existing execution resource cleanup path. No global hidden CUDA buffer ownership should be introduced.
- **D-08:** Native ABI failures must become stable unavailable/failed reason codes rather than silent tensor-array replay. REQUIRED buffer mode must throw before tensor-array fallback when native buffer execution is unavailable or fails.

### Materialization Boundaries
- **D-09:** Add a CUDA `DeviceToCpuMaterializer` equivalent to the Metal materializer. It should support only active CUDA bindings that match the target tensor's dtype, shape, strides, storage offset, and logical element count.
- **D-10:** Graph output publication and CPU consumer reads must call the existing `ExecutionState.requireCpuReadable(...)` path so CUDA-owned results materialize through the standard CPU materialization trace flow instead of ad hoc tensor copying.
- **D-11:** Materialization tests should prove CPU parity for supported dense `FLOAT32` outputs and should verify a trace exists with the correct backend, residency, reason, logical byte length, and success status.

### Adjacent CUDA Handoff
- **D-12:** Adjacent CUDA regions should reuse an existing CUDA `DeviceBufferBinding` when backend id, dtype, shape/layout, access mode, and capability checks allow it. The second CUDA region should consume the device-owned binding instead of forcing Java array materialization.
- **D-13:** Handoff support may remain narrow. It is acceptable to require dense `FLOAT32`, same CUDA backend, available native buffer ABI, and supported DAG node types. Unsupported layouts, dtypes, or backend mismatches must use explicit fallback/reason codes.
- **D-14:** Tests should distinguish native buffer execution, tensor-array fallback, CPU fallback, required-unavailable, and adjacent-handoff behavior without depending on CUDA hardware in the portable gate.

### Portable And Native Verification
- **D-15:** Portable Java tests should use fake CUDA bridges/allocators where possible to verify decisions, materialization registration, resource ownership, REQUIRED-mode behavior, and no tensor-array execution on accepted buffer paths.
- **D-16:** Native CUDA tests must be capability-gated with assumptions or skipped Gradle tasks when CUDA tooling/hardware is unavailable. A local machine without `nvcc` or CUDA hardware must still pass the portable test slice cleanly.
- **D-17:** CPU remains the correctness oracle. Every native CUDA result covered in Phase 7 should have a CPU parity assertion for the same tensor graph or lowered DAG.

### the agent's Discretion
The agent may choose the exact first native operation, class names, and test split. Favor the path that delivers a real end-to-end device-buffer proof with the least native ABI churn. If implementing both `RELU` and `ADD` costs little after the first kernel is in place, doing both is acceptable, but broad operation coverage should not displace materialization and handoff proof.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Planning Scope
- `.planning/ROADMAP.md` — Phase 7 goal, success criteria, dependencies, and Phase 8 boundary.
- `.planning/REQUIREMENTS.md` — `CUDA-03`, `CUDA-04`, and `CUDA-05`, plus explicit out-of-scope limits for public device tensors, broad CUDA coverage, profile churn, and removing CPU fallback.
- `.planning/PROJECT.md` — Core value, current v1.1 state, backend symmetry constraints, and requirement that accelerator fallback/materialization remain visible.
- `.planning/STATE.md` — Current milestone status and operating notes, including unrelated dirty profile files that must not be committed.
- `.planning/phases/06-cuda-shim-and-capability-probe/06-CONTEXT.md` — Locked Phase 6 CUDA shim, capability, buffer ABI, fallback, tests, docs, and hygiene decisions.
- `.planning/phases/06-cuda-shim-and-capability-probe/06-VERIFICATION.md` — Confirms the Phase 6 seam is complete and identifies Phase 7 residual risk.

### Existing Runtime And Buffer Contracts
- `src/main/java/backend/cuda/bridge/CudaFfmBridge.java` — Current CUDA FFM symbol lookup, tensor-array execution path, optional buffer symbols, and conservative `supportsBufferBindings()`.
- `src/main/java/backend/cuda/bridge/CudaGraphBridge.java` — CUDA bridge SPI and buffer binding support contract.
- `src/main/java/backend/cuda/bridge/CudaBridgeExecutable.java` — Compiled executable metadata, external input ids, output ids, and dtype metadata used by buffer requests.
- `src/main/java/backend/cuda/exec/PreparedCudaExecutable.java` — Current CUDA prepared executable, Phase 6 buffer decision publication, REQUIRED-mode failure, and tensor-array fallback path.
- `src/main/java/backend/cuda/buffer/CudaAcceleratorBufferBinder.java` — CUDA shared ABI policy for dense supported layouts and fallback reason codes.
- `src/main/native/cuda/synaptik_cuda_graph_stub.cu` — Existing native CUDA shim source that must gain real buffer ABI support.
- `src/main/java/backend/accelerator/buffer/AcceleratorBufferRequest.java` — Backend-neutral request record to consume.
- `src/main/java/backend/accelerator/buffer/AcceleratorBufferDecision.java` — Backend-neutral decision record to publish.
- `src/main/java/backend/accelerator/buffer/AcceleratorBufferReasonCode.java` — Stable reason-code vocabulary for unavailable, unsupported, fallback, and failure paths.
- `src/main/java/backend/memory/DeviceBufferBinding.java` — Backend-neutral active device binding boundary.
- `src/main/java/backend/memory/DeviceToCpuMaterializer.java` — Materializer SPI that CUDA should implement.
- `src/main/java/graph/execution/ExecutionState.java` — Runtime residency, device binding, materializer, materialization trace, and run-resource ownership APIs.

### Metal Patterns To Mirror Carefully
- `src/main/java/backend/metal/exec/PreparedMetalExecutable.java` — Existing buffer-binding execution, fallback, resource publication, and materialization-aware execution pattern.
- `src/main/java/backend/metal/buffer/MetalAcceleratorBufferBinder.java` — Shared ABI decision/binding pattern.
- `src/main/java/backend/metal/buffer/MetalBufferAllocator.java` — Run-scoped allocator/materializer pattern for native buffers.
- `src/main/java/backend/metal/buffer/MetalDeviceToCpuMaterializer.java` — Device-to-CPU materializer support checks and materialization API shape.
- `src/main/java/backend/metal/buffer/MetalBufferBinding.java` — Backend-specific `DeviceBufferBinding` implementation pattern.
- `src/test/java/backend/metal/exec/PreparedMetalExecutableBufferBindingTest.java` — Prepared executable buffer-binding and fallback test patterns.
- `src/test/java/backend/metal/MetalLayoutAwareDeviceFlowTest.java` — End-to-end device-flow/materialization test style to adapt for CUDA where portable.

### CUDA Tests And Build Workflow
- `build.gradle` — `buildCudaGraphShim`, `cudaTest`, JVM native-access flags, and focused test task conventions.
- `scripts/build-cuda-graph-shim.sh` — Optional CUDA shim build workflow.
- `src/test/java/backend/cuda/bridge/CudaFfmBridgeTest.java` — CUDA bridge availability/capability tests to extend.
- `src/test/java/backend/cuda/buffer/CudaAcceleratorBufferBinderTest.java` — CUDA buffer decision tests.
- `src/test/java/backend/cuda/exec/PreparedCudaExecutableBufferPolicyTest.java` — REQUIRED-mode CUDA buffer policy tests.
- `src/test/java/backend/cuda/CudaAcceleratorExecutionPathTest.java` — CUDA accelerator execution/fallback coverage.
- `src/test/java/SourceTreeHygieneTest.java` — Source/build artifact hygiene checks.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `CudaFfmBridge` already resolves native CUDA shim symbols, creates a shared context, compiles DAG metadata, executes tensor-array paths, and records layered capabilities.
- `PreparedCudaExecutable` already owns bridge/context/executable lifecycle, CPU fallback steps, shared buffer decisions, and REQUIRED-mode behavior.
- `CudaAcceleratorBufferBinder` already accepts dense `FLOAT32` layout metadata and returns stable rejection reasons for unsupported dtype/layout/missing ABI.
- `ExecutionState` already supports active/reserved device bindings, per-run materializer registration, materialization traces, and resource cleanup.
- Metal buffer classes provide a working pattern for allocator, binding, materializer, prepared execution, and tests, but CUDA should not copy Metal-specific shared-memory assumptions blindly.

### Established Patterns
- Optional native capabilities are discovered up front and advanced paths are advertised only when all required symbols and Java support are present.
- Prepared accelerator executables publish `AcceleratorBufferDecision` before executing or falling back.
- CPU storage must be materialized through `ExecutionState.requireCpuReadable(...)`; backend code should not publish stale CPU tensors.
- Native/backend tests must be portable on machines without the optional accelerator and use assumptions or skipped tasks for hardware-specific paths.
- Generated native libraries and local benchmark/profile artifacts stay under ignored build/profile paths and must not be committed accidentally.

### Integration Points
- Native buffer symbols connect through `CudaFfmBridge.State`, `CudaBridgeCapabilities`, and `CudaGraphBridge.supportsBufferBindings()`.
- CUDA buffer allocation/materialization connects to `PreparedCudaExecutable.execute(...)`, `ExecutionState.attachDeviceBufferBinding(...)`, `ExecutionState.registerDeviceToCpuMaterializer(...)`, and `ExecutionState.registerResource(...)`.
- Adjacent handoff connects through `ExecutionState.deviceBufferBindingForNodeId(...)` and `writableDeviceBufferBindingForNodeId(...)` when resolving CUDA inputs/outputs.
- CPU fallback and materialization boundaries connect through `PreparedAcceleratorExecutionSupport`, CPU fallback steps, and `ExecutionContext.requireCpuReadable(...)`.

</code_context>

<specifics>
## Specific Ideas

[--auto] Selected all gray areas: Native operation scope, Native ABI/resource ownership, Materialization boundaries, Adjacent CUDA handoff, Verification strategy.

[auto] Native operation scope — Q: "Which first CUDA-native operation should prove buffer execution?" → Selected: "Smallest dense FLOAT32 operation, preferably RELU or ADD" (recommended default).

[auto] Native ABI/resource ownership — Q: "Where should CUDA buffer handles and lifetime rules live?" → Selected: "Under backend.cuda.*, with shared records remaining backend-neutral" (recommended default).

[auto] Materialization boundaries — Q: "How should CUDA-owned outputs become CPU-visible?" → Selected: "Through ExecutionState's existing DeviceToCpuMaterializer and requireCpuReadable flow" (recommended default).

[auto] Adjacent CUDA handoff — Q: "How broad should first handoff support be?" → Selected: "Narrow dense FLOAT32 same-backend handoff with explicit fallback for unsupported cases" (recommended default).

[auto] Verification strategy — Q: "How should tests handle machines without CUDA?" → Selected: "Portable fake-bridge tests plus capability-gated native cudaTest" (recommended default).

</specifics>

<deferred>
## Deferred Ideas

- CUDA trace/report parity, benchmark evidence, final docs, troubleshooting, and source hygiene closure belong to Phase 8.
- Broad CUDA operation coverage beyond the first proven native buffer operation remains future work.
- Higher-rank native shape ABI expansion and BF16/INT32/BOOL device-buffer execution remain future work unless needed for the narrow proof.
- Persisted CUDA calibration/profile updates are out of scope for Phase 7.

</deferred>

---

*Phase: 7-CUDA Buffer Execution And Materialization*
*Context gathered: 2026-04-30*
