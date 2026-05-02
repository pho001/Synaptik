# Phase 31: BOOL-Producing Metal Compute - Context

**Gathered:** 2026-05-02
**Status:** Ready for planning
**Source:** Roadmap + local code inspection

<domain>
## Phase Boundary

Phase 31 extends Metal from "BOOL can be an external predicate input" to "Metal can produce selected BOOL outputs and keep them device-resident for legal consumers." The primary user-visible flow is a mask chain such as compare/logical output feeding `WHERE` without a CPU materialization boundary.

</domain>

<decisions>
## Implementation Decisions

### Locked Scope
- Deliver `METALBOOL-01`: supported compare/logical operations can produce native device-resident `BOOL` outputs, or unsupported cases reject with stable reason codes.
- Deliver `METALBOOL-02`: supported device-resident BOOL masks can feed `WHERE` and mask-consumer lowering without CPU materialization between supported GPU-region steps.
- Deliver `METALBOOL-03`: `REDUCE_ALL` and `REDUCE_ANY` either execute natively with CPU parity evidence or remain explicit stable rejections with dtype/rank/layout reasons.
- Public `Tensor` remains logical; device residency stays in `ExecutionState`, `DeviceBufferBinding`, and trace/report evidence.
- CUDA must not inherit unsupported Metal BOOL assumptions; CUDA behavior remains capability-gated unless explicitly implemented.

### Current Code Facts
- `MetalMpsCapabilities.outputDecision(DataType.BOOL)` and `computeDecision(DataType.BOOL)` currently reject.
- `AcceleratorDagNodeType` has `WHERE` but no compare/logical/BOOL-reduction node types.
- `GpuLoweringCoverageMatrix` marks `GT`, `GE`, `LT`, `LE`, `EQ`, `NE`, `LOGICAL_AND`, `LOGICAL_OR`, `LOGICAL_NOT`, `REDUCE_ALL`, and `REDUCE_ANY` as `UNSUPPORTED_DTYPE`.
- The native Metal shim has `WHERE` support through `selectWithPredicateTensor`, but no checked-in compare/logical/reduce BOOL op lowering.
- `MetalBufferAllocator` can create BOOL predicate input buffers, and dtype ABI v3 can name `BOOL`.
- `MetalMpsFfmBridge.validateBufferBindings(...)` currently permits BOOL input binding where expected but output validation is still scoped to FLOAT32/BFLOAT16.

### the agent's Discretion
- Implement compare/logical ops through MPSGraph primitives if the local SDK exposes stable APIs.
- If `REDUCE_ALL`/`REDUCE_ANY` support is risky or unavailable, keep them rejected in Phase 31 with explicit stable reasons and tests.
- Add custom native shim branches only for scoped rank/dtype/layout combinations proven by parity tests.

</decisions>

<canonical_refs>
## Canonical References

### Planning
- `.planning/ROADMAP.md` - Phase 31 goal, success criteria, and planned waves.
- `.planning/REQUIREMENTS.md` - `METALBOOL-01`, `METALBOOL-02`, `METALBOOL-03`.
- `.planning/phases/30-bf16-metal-compute-and-output/30-VERIFICATION.md` - latest dtype ABI and scoped dtype support closure.

### Code
- `src/main/java/backend/metal/MetalMpsCapabilities.java` - role-specific dtype support and unsupported messages.
- `src/main/java/backend/accelerator/dag/AcceleratorDagNodeType.java` - native DAG ABI op codes.
- `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java` - operation-to-DAG lowering.
- `src/main/java/backend/accelerator/lowering/GpuLoweringCoverageMatrix.java` - coverage truth rows.
- `src/main/java/backend/metal/lowering/MetalPartitionSupport.java` - planner legality and stable rejection reasons.
- `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java` - Java/native dtype descriptors and buffer binding validation.
- `src/main/native/apple/synaptik_apple_mps_stub.m` - MPSGraph native operation mapping.

### Tests
- `src/test/java/backend/metal/MetalMpsCapabilitiesTest.java`
- `src/test/java/backend/metal/lowering/MetalRegionLowererTest.java`
- `src/test/java/backend/metal/bridge/MetalMpsFfmBridgeTest.java`
- `src/test/java/PreparedExecutionBuildTest.java`
- `src/test/java/GpuCoverageSummaryTest.java`
- `src/test/java/GpuHotPathCoverageTargetsTest.java`

</canonical_refs>

<specifics>
## Specific Ideas

- Treat compare ops (`GT`, `GE`, `LT`, `LE`, `EQ`, `NE`) as the first supported BOOL-producing subset.
- Treat logical ops (`LOGICAL_AND`, `LOGICAL_OR`, `LOGICAL_NOT`) as supported only for BOOL inputs and BOOL outputs.
- Use `bool_compare_where_small` as the primary coverage target once compare -> WHERE stays on Metal.
- Keep masked SDPA admission out of Phase 31 except for producing reusable BOOL mask residency evidence that Phase 34 can consume.

</specifics>

<deferred>
## Deferred Ideas

- Full masked/causal SDPA admission is Phase 34.
- INT32 indexing and gather/take mask interactions are Phase 32.
- Custom Metal kernels and router choice between MPSGraph/custom/CPU are Phase 39 unless a tiny helper is necessary for BOOL support.

</deferred>

---

*Phase: 31-bool-producing-metal-compute*
*Context gathered: 2026-05-02*
