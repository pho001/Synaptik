# Phase 11: GPU Lowering Coverage Matrix - Context

**Gathered:** 2026-04-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 11 broadens Metal and CUDA operation lowering for common NN/tensor patterns and records the support contract in a checked-in coverage matrix. It should make supported GPU regions longer when layout, dtype, cost, and backend capability contracts allow it, while keeping unsupported operations visible through stable rejection reasons. It does not own fused GPU region execution internals (Phase 12) or benchmark/regression gates (Phase 13).

</domain>

<decisions>
## Implementation Decisions

### Coverage Matrix Contract
- **D-01:** Create a checked-in Metal/CUDA lowering coverage matrix that lists common operation families as `supported`, `fallback`, or `unsupported`, with the concrete reason code or unsupported reason for every non-supported entry.
- **D-02:** The matrix should cover at least matmul/linear, elementwise chains, reductions, softmax/log-softmax-style flows, normalization pieces, layout/view-adjacent nodes, and loss-adjacent operations called out by the roadmap. It should prefer honest scoped coverage over broad claims.
- **D-03:** Keep the matrix close enough to source and tests that drift is testable. Recommended default: a docs table plus a small backend-neutral coverage/support model or test helper that verifies the planner/lowerer allowlists match the documented statuses.

### Prioritization And Scope
- **D-04:** Prioritize operation patterns that unlock longer real GPU regions, not isolated one-off ops. Highest priority is preserving `matmul/linear -> layout/view -> elementwise/softmax-ish` paths that Phase 10 made layout-safe.
- **D-05:** Start from the current shared accelerator DAG and backend allowlists rather than introducing a separate second lowering architecture. Expand only where the native DAG ABI and Java prepare/execution path can prove CPU parity and visible fallback.
- **D-06:** Treat reductions, normalization, and loss-adjacent flows as matrix-first unless a narrow implementation is straightforward and testable in this phase. If a family is not implemented, it must be rejected with a stable reason and explicit matrix entry.

### Shared Lowering Model
- **D-07:** Avoid growing divergent Metal and CUDA operation lists by hand. Planning should introduce a shared operation coverage model or common helper consumed by `MetalPartitionSupport`, `CudaGpuRegionLegalityAdapter`, and tests where practical.
- **D-08:** Backend-specific native capability remains backend-owned. The shared model can classify semantic operation families and reasons, but Metal/CUDA still decide final availability based on dtype, layout ABI, native symbols, and bridge capability.
- **D-09:** Stable rejection reasons should distinguish unsupported operation, dtype, layout, rank/shape, capability missing, native ABI mismatch, and intentionally-deferred fused-region support. These reasons must surface through planner/lowering tests and traces/reports where the current metadata path supports them.

### Layout, DType, And Residency Rules
- **D-10:** Carry forward Phase 9/10 decisions: public `Tensor` stays logical; residency lives in `ExecutionState` and `DeviceBufferBinding`; metadata-only views and dense layout materialization stay separate.
- **D-11:** CUDA direct non-dense compute remains conservative unless Phase 10 metadata-only view propagation or dense materialization has made the consumer layout legal. Do not broaden CUDA lowering by silently accepting arbitrary non-dense inputs.
- **D-12:** Supported patterns should stay in GPU regions only when dtype, layout, runtime enablement, cost, and capability contracts allow it. REQUIRED mode must still fail before hidden tensor-array or CPU fallback.

### Verification Strategy
- **D-13:** Use focused portable JUnit gates first: coverage matrix status tests, Metal/CUDA legality adapter tests, lowering result tests, backend selection selected/rejected candidate tests, and trace/reason-code tests.
- **D-14:** Native Metal/CUDA tests stay capability-gated. Phase 11 should not require committing machine-local benchmark or profile artifacts.
- **D-15:** Closure must prove both selected and rejected candidates: at least one newly supported high-value pattern stays in a GPU region, and at least one unsupported family/dtype/layout/capability gap rejects visibly without CPU/Metal/CUDA fallback safeguards regressing.

### the agent's Discretion
- Exact names and file placement for a shared coverage model are left to planning, but the implementation should follow existing package ownership: backend-neutral contracts under `backend.accelerator` or `backend.partition/lowering`, backend-specific decisions under `backend.metal` and `backend.cuda`, and docs under `docs/`.
- If the planner finds the native DAG already supports a pattern but planner allowlists or tests do not, it may treat "wire and document existing support" as valid Phase 11 work.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Scope
- `.planning/ROADMAP.md` — Phase 11 goal, success criteria, dependencies, and v1.2 sequencing.
- `.planning/REQUIREMENTS.md` — GPULOWER-01, GPULOWER-02, and GPULOWER-03 acceptance scope.
- `.planning/PROJECT.md` — Accelerator architecture constraints, active v1.2 requirements, and validated Phase 9/10 decisions.
- `.planning/STATE.md` — Current milestone state and local artifact hygiene notes.

### Prior Phase Contracts
- `.planning/phases/09-native-layout-abi-v2/09-VERIFICATION.md` — Layout ABI v2 metadata/capability/fallback contract already validated.
- `.planning/phases/10-gpu-layout-transform-and-view-path/10-VERIFICATION.md` — Layout/view residency, dense materialization gates, and conservative CUDA non-dense scope already validated.
- `.planning/phases/10-gpu-layout-transform-and-view-path/10-RESEARCH.md` — Research note that broad NN/tensor operation lowering belongs to Phase 11 and direct non-dense CUDA compute remains conservative.

### Lowering And Partitioning Code
- `src/main/java/backend/partition/BackendPartitionDescriptorRegistry.java` — Registers CPU, Metal, and CUDA legality adapters/lowerers.
- `src/main/java/backend/lowering/LoweringPipeline.java` — Backend-neutral lowering orchestration.
- `src/main/java/backend/metal/lowering/MetalPartitionSupport.java` — Current Metal planner allowlist and unsupported reasons.
- `src/main/java/backend/metal/lowering/MetalRegionLegalityAdapter.java` — Metal partition candidate construction and plan creation.
- `src/main/java/backend/metal/lowering/MetalRegionLowerer.java` — Metal lowered-region family selection.
- `src/main/java/backend/cuda/lowering/CudaGpuRegionLegalityAdapter.java` — Current CUDA planner allowlist and candidate construction.
- `src/main/java/backend/cuda/lowering/CudaRegionLowerer.java` — CUDA lowered-region family selection.
- `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java` — Shared accelerator DAG lowering and matmul/linear post-op lowering.
- `src/main/java/backend/accelerator/dag/AcceleratorDagNodeType.java` — Native accelerator DAG operation ABI codes.
- `src/main/java/backend/accelerator/dag/AcceleratorPostOpType.java` — Native post-op ABI codes for matmul/linear-like plans.

### Tests And Docs
- `src/test/java/backend/metal/lowering/MetalRegionLowererTest.java` — Metal lowering coverage patterns.
- `src/test/java/backend/cuda/lowering/CudaRegionLowererTest.java` — CUDA lowering coverage patterns.
- `src/test/java/backend/metal/MetalLayoutAwareDeviceFlowTest.java` — Layout-heavy Metal flow and gradient boundary evidence.
- `src/test/java/backend/cuda/exec/CudaLayoutTransformDeviceFlowTest.java` — CUDA metadata-only view and visible fallback evidence.
- `src/test/java/CompiledGraphTraceTest.java` — Trace metadata expectations.
- `docs/native-bridges-and-blas.md` — Accelerator DAG/native ABI and Phase 10 layout transform contract.
- `docs/development.md` — Focused Gradle/native verification commands.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `AcceleratorSubgraphLowerer` already lowers shared accelerator DAG specs and maps many operation types to `AcceleratorDagNodeType`.
- `AcceleratorDagNodeType` already includes core matmul/linear, elementwise, layout, softmax, selected gradient, and SDPA op codes; this can seed the matrix.
- `AcceleratorPostOpType` already captures many matmul post-op candidates, including unary activations and binary/scalar post-ops.
- `MetalPartitionSupport` exposes `plannerUnsupportedReason(...)`, which is a useful pattern for stable rejection reasons.
- `CudaGpuRegionLegalityAdapter` currently has a CUDA allowlist but less diagnostic detail than Metal; Phase 11 can make CUDA rejection visibility more symmetric.

### Established Patterns
- Backend-neutral orchestration belongs in `backend/lowering`, `backend/partition`, and `backend/accelerator`; backend native details remain under `backend/metal` and `backend/cuda`.
- Existing tests use focused JUnit classes and real tensor/compiled graph fixtures rather than mocking `Tensor` or `CompiledGraph`.
- Optional native checks use Gradle capability gates and assumptions; portable Java tests remain mandatory.
- Local tuning/profile artifacts under `profiles/platform/.../tuning/abc/*` are dirty and must remain unstaged unless intentionally updating canonical profiles.

### Integration Points
- Planner legality: `MetalPartitionSupport.isPlannerSupported(...)` and `CudaGpuRegionLegalityAdapter.isNodeSupported(...)`.
- Structural candidates: Metal and CUDA legality adapters determine candidate node ordering, external inputs, outputs, and anchors.
- Lowering result families: `MetalRegionLowerer` and `CudaRegionLowerer` map optimizer execution units to `LoweringFamily`.
- Backend selection visibility: `DefaultBackendSelectionPolicy` and partition traces should continue to expose selected/rejected candidates.
- Trace/report closure: `PreparedExecution` and trace metadata already expose accelerator reason codes and storage residency from Phase 10.

</code_context>

<specifics>
## Specific Ideas

- Treat the coverage matrix as an executable contract: every supported/fallback/unsupported claim should have a corresponding test or source assertion.
- Prefer a small shared coverage model over duplicating allowlists in Metal and CUDA, but do not force backend-specific capability checks into common records.
- Let unsupported normalization/loss-adjacent pieces remain explicit if implementing them would pull Phase 12 fused-region or Phase 13 benchmark scope forward.

</specifics>

<deferred>
## Deferred Ideas

- Fused GPU compound execution for linear + bias + activation and elementwise chains is Phase 12 scope.
- Coverage ratios, benchmark workload comparisons, and regression gates are Phase 13 scope.
- Broad universal operation, dtype, rank, and layout support remains future accelerator coverage beyond v1.2.

</deferred>

---

*Phase: 11-gpu-lowering-coverage-matrix*
*Context gathered: 2026-04-30*
