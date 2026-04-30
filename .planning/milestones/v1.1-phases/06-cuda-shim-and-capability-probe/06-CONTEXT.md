# Phase 6: CUDA Shim And Capability Probe - Context

**Gathered:** 2026-04-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 6 brings CUDA from Java-side scaffolding to a checked-in, capability-gated native bridge seam. It adds the minimal native shim source and build/probe workflow, makes CUDA runtime availability explicit, and prepares the Java CUDA executable path to consume the shared accelerator buffer layout/access ABI for supported dense layouts. It does not need to prove broad CUDA operation coverage, native materialization, adjacent-region handoff, benchmark evidence, or final CUDA observability parity; those belong to Phases 7 and 8.

</domain>

<decisions>
## Implementation Decisions

### Native Shim And Build Workflow
- **D-01:** Add a checked-in minimal CUDA shim under `src/main/native/cuda/` with the existing CUDA graph bridge symbols as the baseline ABI: `synaptik_cuda_graph_available`, `synaptik_cuda_graph_unavailable_reason`, `synaptik_cuda_graph_create_context`, `synaptik_cuda_graph_compile_partition_f32`, `synaptik_cuda_graph_execute_partition_f32`, `synaptik_cuda_graph_destroy_context`, and `synaptik_cuda_graph_destroy_executable`.
- **D-02:** Add a CUDA build script analogous to the Metal script, expected at `scripts/build-cuda-graph-shim.sh`, producing the library under `build/native/cuda/`. The script may skip or fail with a clear diagnostic when CUDA toolkit/compiler prerequisites are absent.
- **D-03:** Add Gradle wiring for a targeted CUDA native build/probe task and an optional CUDA-focused test task. Portable `test` and `classes` must not require CUDA hardware, CUDA drivers, or `nvcc`.
- **D-04:** Document the lookup order already used by `CudaFfmBridge`: `-Dsynaptik.cuda.graph.lib`, then `SYNAPTIK_CUDA_GRAPH_LIB`, then library name `synaptik_cuda_graph`.

### Capability Probe And Reason Codes
- **D-05:** Treat CUDA availability as layered capability state, not a single boolean. The Java side should distinguish at least: native library missing, required discovery symbol missing, shim reports CUDA unavailable, context creation unavailable, compile ABI unavailable, execute ABI unavailable, buffer ABI unavailable, unsupported dtype, unsupported layout, and required buffer execution unavailable.
- **D-06:** Missing or older CUDA shims must fail gracefully into unavailable bridge/context/executable decisions. ABI mismatch should surface as an explicit unavailable reason instead of being silently collapsed into generic CPU fallback.
- **D-07:** `supportsBufferBindings()` for CUDA must remain false until every required CUDA buffer ABI symbol and Java-side binding contract needed for safe execution is present.

### Shared Accelerator Buffer ABI
- **D-08:** CUDA bridge and prepared executable seams should consume `AcceleratorBufferRequest`, `AcceleratorBufferLayout`, `AcceleratorBufferDecision`, and backend-neutral `DeviceBufferBinding` metadata, matching the shared ABI established for Metal.
- **D-09:** Keep common accelerator records backend-neutral. CUDA-specific native handles, allocator/resource lifetimes, access enums, and bridge structs belong under `backend.cuda.*`, not in shared `backend.accelerator.*` or public `Tensor` API.
- **D-10:** Phase 6 proves only the seam for supported dense CUDA layouts, primarily dense `FLOAT32` graph execution metadata. Non-dense logical-view execution, BF16/INT32/BOOL buffer execution, native materialization, and adjacent CUDA handoff remain deferred.
- **D-11:** Do not add public user-facing device tensor APIs. Backend residency stays in compile/prepare/execute runtime state and `ExecutionState` device buffer bindings.

### Execution And Fallback Policy
- **D-12:** REQUIRED buffer mode must fail before tensor-list execution when CUDA buffer execution is unavailable, preserving the existing `REQUIRED_BUFFER_EXECUTION_UNAVAILABLE` behavior and making the CUDA reason specific.
- **D-13:** AUTO/default mode may fall back to tensor-array or CPU replay, but the fallback path must publish a stable `AcceleratorBufferDecision` with backend `GPU_CUDA`, selected path, reason code, and human-readable reason.
- **D-14:** Existing CPU and Metal behavior must remain unchanged when CUDA is unavailable. CUDA work should be capability-gated and should not weaken CPU hot-path, Metal buffer-binding, or materialization safeguards.

### Tests, Docs, And Hygiene
- **D-15:** Add portable Java tests that do not require CUDA hardware: missing/unavailable native library behavior, explicit library/probe diagnostics, unsupported dtype/layout decisions, dense layout metadata acceptance, and REQUIRED-mode failure before tensor-list fallback.
- **D-16:** Optional native CUDA tests should use JUnit assumptions and the targeted CUDA Gradle task so machines without CUDA report skipped native checks rather than failures.
- **D-17:** Documentation must explain CUDA build prerequisites, build/probe commands, library lookup order, fallback interpretation, and how CUDA shares the Metal-era accelerator ABI while remaining backend-specific at native-handle boundaries.
- **D-18:** Keep local CUDA build outputs under `build/native/cuda/` or another ignored build location. Do not commit generated native artifacts, local benchmark outputs, or platform tuning/profile churn as part of Phase 6.

### the agent's Discretion
The agent may choose exact file names for CUDA buffer helper classes, native C/C++ source extension, and the Gradle task names as long as they follow existing repository conventions, mirror the Metal build/test ergonomics where practical, and keep CUDA-specific implementation details inside CUDA packages.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Planning Scope
- `.planning/ROADMAP.md` — Phase 6 goal, requirements, success criteria, dependencies, and Phase 7/8 boundaries.
- `.planning/REQUIREMENTS.md` — `CUDA-01` and `CUDA-02`, plus out-of-scope limits for broad CUDA operation coverage, public device tensors, and profile churn.
- `.planning/PROJECT.md` — Project state, core value, architecture constraints, current milestone goals, and validated v1.0 accelerator decisions.
- `.planning/STATE.md` — Current milestone status and operating notes, including existing unrelated profile changes.

### Codebase Maps
- `.planning/codebase/STACK.md` — Java 25, Gradle, FFM, Vector API, and native-access assumptions.
- `.planning/codebase/ARCHITECTURE.md` — Compile/prepare/execute lifecycle, accelerator regions, `ExecutionState`, and device buffer binding architecture.
- `.planning/codebase/INTEGRATIONS.md` — Native bridge integration points and CUDA lookup behavior.
- `.planning/codebase/TESTING.md` — JUnit/Gradle testing patterns and capability-gated native test expectations.
- `.planning/codebase/CONCERNS.md` — Native ABI mismatch, silent fallback, and local artifact hygiene risks.

### Existing CUDA And Metal Contracts
- `src/main/java/backend/cuda/bridge/CudaFfmBridge.java` — Current CUDA FFM symbol lookup, library resolution order, context/compile/execute methods, and unavailable handling.
- `src/main/java/backend/cuda/bridge/CudaGraphBridge.java` — CUDA bridge SPI and current `supportsBufferBindings()` default.
- `src/main/java/backend/cuda/exec/PreparedCudaExecutable.java` — Current CUDA prepared executable fallback, REQUIRED-mode failure, and `lastAcceleratorBufferDecision` behavior.
- `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java` — Existing native bridge pattern for optional buffer ABI symbols and buffer execution.
- `src/main/java/backend/metal/buffer/MetalAcceleratorBufferBinder.java` — Existing shared ABI decision/binding pattern to mirror conservatively for CUDA.
- `src/main/java/backend/metal/exec/PreparedMetalExecutable.java` — Existing prepared executable flow for bridge availability, buffer decision publication, REQUIRED-mode enforcement, and fallback.
- `src/main/java/backend/accelerator/buffer/AcceleratorBufferRequest.java` — Backend-neutral request record CUDA should consume.
- `src/main/java/backend/accelerator/buffer/AcceleratorBufferDecision.java` — Backend-neutral decision record CUDA should publish.
- `src/main/java/backend/memory/DeviceBufferBinding.java` — Backend-neutral device buffer descriptor boundary.

### Build, Docs, And Tests
- `build.gradle` — Current Metal native task, `nativeBuild`, `metalTest`, FFM JVM args, and test configuration pattern.
- `scripts/build-metal-mps-shim.sh` — Build-script template for optional native shim output under `build/native/...`.
- `src/test/java/backend/cuda/bridge/CudaFfmBridgeTest.java` — Existing CUDA bridge availability tests that should be expanded without requiring hardware.
- `src/test/java/backend/cuda/exec/PreparedCudaExecutableBufferPolicyTest.java` — Existing REQUIRED-mode CUDA policy tests.
- `docs/development.md` — Existing CUDA lookup documentation and native test caveats.
- `docs/configuration.md` — Runtime property/env-var documentation for CUDA bridge lookup.
- `docs/architecture.md` — Accelerator architecture and CUDA scaffolding documentation.
- `docs/metal-backend.md` — Metal buffer-binding observability and explicit note that CUDA remains capability-gated.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `CudaFfmBridge` already resolves CUDA library names, requires discovery symbols, creates contexts, compiles lowered DAG specs, and executes tensor-array `FLOAT32`/`BOOL` inputs through FFM.
- `PreparedCudaExecutable` already owns the bridge/context/executable triplet, preserves CPU fallback steps, publishes `AcceleratorBufferDecision`, and fails REQUIRED buffer mode before tensor-list execution.
- Metal's `MetalMpsFfmBridge`, `MetalAcceleratorBufferBinder`, `MetalBufferAllocator`, `MetalBufferBinding`, and `PreparedMetalExecutable` provide the closest existing pattern for optional buffer ABI symbols and shared layout decisions.
- `AcceleratorBufferLayout`, `AcceleratorBufferRequest`, `AcceleratorBufferDecision`, and `DeviceBufferBinding` already model the shared backend-neutral ABI needed by CUDA.
- `build.gradle` and `scripts/build-metal-mps-shim.sh` already show how optional native build/test commands should be wired without making the portable Java gate depend on native hardware.

### Established Patterns
- Native bridge availability is discovered once and exposed through unavailable bridge/context/executable records instead of throwing during normal fallback paths.
- Optional native symbols should gate advanced capabilities. The bridge should only advertise buffer binding when all needed native symbols are available.
- Prepared accelerator executables publish the latest accelerator buffer decision and enforce REQUIRED mode before falling back.
- Tests for optional native paths use assumptions or explicit unavailable assertions so hardware-specific checks do not break portable CI.
- Source hygiene keeps generated artifacts out of `src/`, `test/`, profiles, and `.planning/tmp/` unless intentionally updating canonical fixtures.

### Integration Points
- CUDA native source and script connect through `CudaFfmBridge.resolveLookup(...)` and a Gradle task analogous to `buildMetalMpsShim`.
- CUDA buffer capability should connect at `CudaGraphBridge.supportsBufferBindings()` and `PreparedCudaExecutable.execute(...)`.
- Shared ABI decisions should be built from runtime tensors and lowered executable input/output node ids in the same shape as Metal's `bufferRequest(...)`.
- CUDA buffer-specific handles/resources/materializers should be introduced under `backend.cuda.*` only when needed by the Phase 6 seam, with native materialization and adjacent handoff deferred.

</code_context>

<specifics>
## Specific Ideas

Auto mode selected conservative defaults. The implementation should look like the Metal native workflow where that reduces risk, but CUDA must remain explicitly capability-gated and should not claim production buffer execution until both Java and native buffer ABI pieces are present.

</specifics>

<deferred>
## Deferred Ideas

- Full CUDA device-buffer execution for a representative operation belongs to Phase 7.
- CUDA device-to-CPU materialization for graph outputs and CPU consumers belongs to Phase 7.
- Adjacent CUDA region handoff without Java array round trips belongs to Phase 7.
- CUDA trace/report parity, benchmark evidence, final docs, and source hygiene gates belong to Phase 8.
- Broader accelerator operation coverage, higher-rank native shape ABI expansion, and additional dtype buffer execution remain future milestone work unless Phase 7/8 explicitly narrow them in.

</deferred>

---

*Phase: 6-CUDA Shim And Capability Probe*
*Context gathered: 2026-04-30*
