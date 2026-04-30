# Phase 9: Native Layout ABI v2 - Context

**Gathered:** 2026-04-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 9 establishes the Metal/CUDA native layout ABI v2 contract: shared Java metadata, optional native capability/version discovery, and explicit fallback/required-mode diagnostics for non-contiguous/view-capable buffers. It does not need to make every layout execute natively yet; GPU-side layout transforms, broader lowering, and fused region execution belong to later v1.2 phases.

</domain>

<decisions>
## Implementation Decisions

### ABI Compatibility And Scope
- **D-01:** Implement layout ABI v2 as an additive contract beside the existing dense buffer ABI. Existing v1 dense `FLOAT32` Metal/CUDA buffer execution must continue to work when v2 symbols are absent.
- **D-02:** The shared Java contract should extend the current `backend.accelerator.buffer.AcceleratorBufferLayout` family rather than introduce backend-specific common records. Backend-specific native handles and lifetimes remain under `backend.metal.*` and `backend.cuda.*`.
- **D-03:** Phase 9 should carry metadata and capability decisions only. It may add native struct/array plumbing and validation, but it should not attempt GPU layout transforms, broad operation lowering, or fused GPU regions.

### Layout Metadata Shape
- **D-04:** ABI v2 metadata should include rank, full shape array, full stride array, storage offset in elements, logical element count, logical byte length, physical byte span, access mode, backend id, dtype, layout class, and native handle identity.
- **D-05:** Compute physical span explicitly from shape/strides/storage offset so planners can distinguish logical bytes from backing-storage bytes. Negative strides remain unsupported unless a backend explicitly advertises support later.
- **D-06:** Zero-stride broadcast views, non-zero-offset dense views, padded/zero-offset views, and permuted/strided views should be representable in metadata even when a backend rejects them for execution.

### Capability And Version Handshake
- **D-07:** Add optional native capability/version symbols for Metal and CUDA layout ABI v2. Absence of those symbols means "layout ABI v2 unavailable" while preserving the existing bridge availability and dense buffer capability.
- **D-08:** Java capability records should report layout ABI v2 separately from native library, runtime/context, graph execution, and v1 buffer execution. CUDA can extend `CudaBridgeCapabilities`; Metal should gain an equivalent explicit capability surface instead of relying only on boolean `supportsBufferBindings()`.
- **D-09:** Capability checks must be stable and portable: Java tests should be able to prove missing-symbol, version-mismatch, and unavailable-runtime behavior without requiring Metal or CUDA hardware.

### Fallback And Required-Mode Semantics
- **D-10:** Unsupported layout metadata, unsupported rank/dtype, physical span overflow, native ABI mismatch, and missing layout ABI v2 symbols must use stable reason codes. Add new codes rather than overloading `INPUT_NOT_CONTIGUOUS` or `OUTPUT_LAYOUT_UNSUPPORTED` when the failure is specifically ABI-version/capability related.
- **D-11:** In `AUTO`, layout ABI v2 rejection falls back visibly to the existing dense/tensor-array/CPU path according to current policy. In `REQUIRE`, it must fail before hidden tensor-array or CPU execution can satisfy the operation.
- **D-12:** Trace and benchmark fields introduced in Phase 9 should be diagnostic and backend-neutral where possible, but Phase 13 owns full coverage reporting and workload gates.

### Test And Native Rollout
- **D-13:** Start with shared Java metadata and fake/portable bridge tests, then add optional native Metal/CUDA symbol plumbing. Native checks remain capability-gated.
- **D-14:** Metal and CUDA should be kept symmetric at the contract level, but native implementation details may differ. The planner should see a common layout/capability model.
- **D-15:** Verification should include `./gradlew classes`, focused accelerator buffer/layout tests, focused Metal/CUDA bridge tests, and optional `./gradlew metalTest` / `./gradlew buildCudaGraphShim cudaTest` when local tooling is available.

### the agent's Discretion
- Exact Java type names, package placement, and native symbol names are left to the planner, as long as they preserve the shared-contract/backend-owned-handle boundary.
- The planner may choose whether ABI v2 metadata crosses native boundaries as structs, parallel arrays, or packed descriptors, provided the FFM contract is versioned, testable, and documented.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Scope
- `.planning/ROADMAP.md` — Phase 9 goal, dependencies, success criteria, and v1.2 phase sequencing.
- `.planning/REQUIREMENTS.md` — GPULAYOUT-01, GPULAYOUT-02, and GPULAYOUT-03 acceptance scope.
- `.planning/PROJECT.md` — Project-level accelerator architecture constraints and v1.2 milestone intent.

### Existing Accelerator Buffer Contract
- `src/main/java/backend/accelerator/buffer/AcceleratorBufferLayout.java` — Current shared logical layout metadata.
- `src/main/java/backend/accelerator/buffer/AcceleratorBufferRequest.java` — Current shared buffer decision input record.
- `src/main/java/backend/accelerator/buffer/AcceleratorBufferDecision.java` — Current shared execution-path decision record.
- `src/main/java/backend/accelerator/buffer/AcceleratorBufferReasonCode.java` — Stable reason-code enum to extend for ABI v2 outcomes.
- `src/test/java/backend/accelerator/buffer/AcceleratorBufferLayoutClassifierTest.java` — Existing layout-class coverage and defensive-copy/overflow tests.

### Metal Native Boundary
- `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java` — Existing Metal FFM symbol lookup, compile, buffer execution, and binding validation.
- `src/main/java/backend/metal/MetalMpsCapabilities.java` — Current Metal dtype and role capability boundary.
- `src/main/native/apple/synaptik_apple_mps_stub.m` — Objective-C native shim and current rank/dim0-dim3 ABI.
- `src/test/java/backend/metal/bridge/MetalMpsFfmBridgeTest.java` — Existing Metal bridge and buffer ABI tests.
- `docs/metal-backend.md` — Current Metal backend source map, native buffer ABI, fallback, and tests.

### CUDA Native Boundary
- `src/main/java/backend/cuda/bridge/CudaFfmBridge.java` — Existing CUDA FFM symbol lookup, capability reporting, compile, buffer execution, and binding validation.
- `src/main/java/backend/cuda/bridge/CudaBridgeCapabilities.java` — Current layered CUDA capability record.
- `src/main/java/backend/cuda/bridge/CudaBridgeCapabilityCode.java` — Current CUDA stable capability codes.
- `src/main/native/cuda/synaptik_cuda_graph_stub.cu` — CUDA native shim and current rank/dim0-dim3 buffer execution ABI.
- `src/test/java/backend/cuda/bridge/CudaFfmBridgeTest.java` — Existing CUDA bridge capability tests.

### Runtime And Docs
- `docs/development.md` — Focused Gradle commands and CUDA/Metal verification expectations.
- `docs/native-bridges-and-blas.md` — Java FFM/native ABI terminology and boundary model.
- `.planning/codebase/INTEGRATIONS.md` — Native library lookup, optional bridge behavior, and external integration map.
- `.planning/codebase/CONCERNS.md` — Native ABI fragility, generated artifact hygiene, and accelerator scaling limits.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `AcceleratorBufferLayout` already captures dtype, shape, strides, storage offset, logical element count, logical byte length, and layout class. Phase 9 should extend this model rather than starting over.
- `AcceleratorBufferLayoutClassifier` already classifies dense, zero-offset view, non-zero-offset view, permuted/strided view, broadcast zero-stride view, and unsupported layouts. This is the natural source for ABI v2 metadata decisions.
- `AcceleratorBufferDecision` and `AcceleratorBufferReasonCode` already provide backend-neutral diagnostics for AUTO/REQUIRE buffer paths.
- `CudaBridgeCapabilities` already models layered native capability state; Metal lacks an equivalent full capability record and should gain one or an equivalent explicit reporting mechanism.

### Established Patterns
- Public `Tensor` stays logical; backend residency lives in `ExecutionState` and backend buffer bindings.
- Native bridge availability is discovered through optional/local library lookup and missing symbols become unavailable records or explicit fallback rather than process crashes.
- Existing Metal/CUDA FFM compile ABI uses rank plus fixed dim0-dim3 arrays. ABI v2 should be versioned before expanding beyond that shape.
- Optional native tests skip when local tooling is unavailable; portable Java tests must still prove reason codes and fallback behavior.

### Integration Points
- Shared ABI records: `src/main/java/backend/accelerator/buffer/`.
- Metal Java bridge and native shim: `MetalMpsFfmBridge`, `MetalMpsCapabilities`, and `synaptik_apple_mps_stub.m`.
- CUDA Java bridge and native shim: `CudaFfmBridge`, `CudaBridgeCapabilities`, `CudaBridgeCapabilityCode`, and `synaptik_cuda_graph_stub.cu`.
- Tests: `AcceleratorBufferLayoutClassifierTest`, `MetalMpsFfmBridgeTest`, `CudaFfmBridgeTest`, plus focused buffer binder/allocator tests under `src/test/java/backend/{metal,cuda}/`.

</code_context>

<specifics>
## Specific Ideas

- Use Phase 9 to make layout metadata explicit and testable before later phases consume it for GPU-side view/transform execution.
- Treat "v2 available" as a capability bit with diagnostic detail, not as a blanket Metal/CUDA availability flag.
- Keep current dense paths green throughout the migration; regressions in v1 dense buffer execution should block Phase 9.

</specifics>

<deferred>
## Deferred Ideas

- GPU-side `reshape` / `permute` / `expand` execution and view residency are Phase 10 scope.
- Broader operation lowering coverage is Phase 11 scope.
- Fused GPU region execution is Phase 12 scope.
- Full coverage benchmark/regression reporting is Phase 13 scope.

</deferred>

---

*Phase: 09-native-layout-abi-v2*
*Context gathered: 2026-04-30*
