# Roadmap: Synaptik

## Milestones

- ✅ **v1.0 Accelerator Runtime Architecture** - Phases 1-5 shipped 2026-04-30. Full archive: [v1.0-ROADMAP.md](milestones/v1.0-ROADMAP.md)
- ✅ **v1.1 CUDA Native Runtime** - Phases 6-8 shipped 2026-04-30. Full archive: [v1.1-ROADMAP.md](milestones/v1.1-ROADMAP.md)
- ✅ **v1.2 GPU Region Coverage** - Phases 9-13 shipped 2026-05-01. Full archive: [v1.2-ROADMAP.md](milestones/v1.2-ROADMAP.md)
- ✅ **v1.3 Coverage-Driven GPU Region Expansion** - Phases 14-21 shipped 2026-05-01. Full archive: [v1.3-ROADMAP.md](milestones/v1.3-ROADMAP.md)
- ✅ **v1.4 Native GPU Operation Coverage Closure** - Phases 22-28 shipped 2026-05-02. Full archive: [v1.4-ROADMAP.md](milestones/v1.4-ROADMAP.md)
- 🚧 **v1.5 Production-Grade Metal Backend Expansion** - Phases 29-39 active. Scope: broaden Metal dtype, operation, layout, training, router, and lower-copy coverage while preserving explicit fallback.

## Summary

| # | Phase | Goal | Requirements | Success Criteria |
|---|-------|------|--------------|------------------|
| 29 | Metal DType ABI And Capability Truth | Establish versioned Metal dtype ABI/capability contracts before widening native compute. | METALDTYPE-01, METALDTYPE-02, METALDTYPE-03 | 5 |
| 30 | BF16 Metal Compute And Output | Add legal BF16 Metal compute/output coverage for high-value supported op families. | METALBF16-01, METALBF16-02, METALBF16-03 | 5 |
| 31 | BOOL-Producing Metal Compute | Let Metal produce and consume device-resident BOOL masks for supported compare/logical/reduction flows. | METALBOOL-01, METALBOOL-02, METALBOOL-03 | 5 |
| 32 | INT32 Index Tensor And Gather Take Path | Add native INT32 index tensor handling plus forward gather/take Metal execution. | METALINTIDX-01, METALINTIDX-02, METALINTIDX-03 | 5 |
| 33 | GPU Layout Router And Strided Materialization | Avoid CPU exits by routing layout repair and materialization through GPU-side Metal paths where legal. | METALLAYOUT-01, METALLAYOUT-02, METALLAYOUT-03 | 5 |
| 34 | Masked And Causal SDPA | Admit verified masked and causal Metal SDPA while preserving mask semantics. | METALSDPAMASK-01, METALSDPAMASK-02, METALSDPAMASK-03 | 5 |
| 35 | Conv Pool Native Execution | Add selected Metal conv/pool native or lowered execution with parity and coverage gates. | METALCONVPOOL-01, METALCONVPOOL-02, METALCONVPOOL-03 | 5 |
| 36 | Scatter And Index Gradient Semantics | Support or explicitly reject duplicate-index scatter and index-gradient paths. | METALSCATTER-01, METALSCATTER-02, METALSCATTER-03 | 5 |
| 37 | Loss-Adjacent Metal Lowering | Lower dense and index-target loss-adjacent flows where semantics are proven. | METALLOSS-01, METALLOSS-02, METALLOSS-03 | 5 |
| 38 | Metal Training Backward Coverage | Keep gradients on Metal for v1.5-supported forward families where legal. | METALTRAIN-01, METALTRAIN-02, METALTRAIN-03 | 5 |
| 39 | Metal Backend Router And Zero-Copy Closure | Add calibrated MPSGraph/custom-kernel/CPU routing and close remaining native-copy evidence gaps. | METALROUTER-01, METALROUTER-02, METALROUTER-03 | 5 |

## Milestone Rule

v1.5 is Metal-first, but not Metal-only architecture. Shared accelerator contracts may be extended where needed, yet `GPU_CUDA` must keep explicit capability-gated behavior instead of inheriting unsupported Metal assumptions.

A Metal row can become `SUPPORTED` only when the codebase has all of: semantic contract, lowering, backend legality, native or routed execution, CPU parity evidence, trace/report evidence, and regression gate coverage. Dtype residency alone is not native dtype compute.

## Phase Details

### Phase 29: Metal DType ABI And Capability Truth

**Status:** Complete - verified 2026-05-02

**Goal:** Establish versioned Metal dtype ABI/capability contracts before widening native compute.

**Requirements:** METALDTYPE-01, METALDTYPE-02, METALDTYPE-03

**Depends on:** v1.4 Phase 28 coverage regression closure and current `MetalMpsCapabilities`.

**Success Criteria:**
1. Metal capability probes distinguish storage representation, compute legality, output legality, and per-op dtype support.
2. Native ABI versioning prevents older `_f32` shims from silently accepting BF16/BOOL/INT32/FLOAT64 paths.
3. Reports distinguish dtype residency, dtype conversion, native dtype compute, and dtype fallback.
4. FLOAT64 receives an explicit device capability decision instead of accidental support.
5. Phase 30-32 can add dtype-specific execution without reopening ABI semantics.

**Plans:**

Wave 1:
- [x] [29-01 Metal dtype capability model and reason codes](phases/29-metal-dtype-abi-and-capability-truth/29-01-PLAN.md).

Wave 2 *(blocked on Wave 1 completion)*:
- [x] [29-02 Native ABI v3 dtype descriptor and optional-symbol probes](phases/29-metal-dtype-abi-and-capability-truth/29-02-PLAN.md).

Wave 3 *(blocked on Waves 1-2 completion)*:
- [x] [29-03 Dtype trace/report and coverage truth update](phases/29-metal-dtype-abi-and-capability-truth/29-03-PLAN.md).

Wave 4 *(blocked on Waves 1-3 completion)*:
- [x] [29-04 Docs, tests, and migration closure](phases/29-metal-dtype-abi-and-capability-truth/29-04-PLAN.md).

### Phase 30: BF16 Metal Compute And Output

**Status:** Complete - verified 2026-05-02

**Goal:** Add legal BF16 Metal compute/output coverage for high-value supported op families.

**Requirements:** METALBF16-01, METALBF16-02, METALBF16-03

**Depends on:** Phase 29.

**Success Criteria:**
1. BF16 buffer binding and materialization preserve Synaptik BF16 storage semantics.
2. Supported BF16 matmul/linear and elementwise flows execute through Metal or reject per capability.
3. BF16 reductions, softmax/log-softmax, LayerNorm, and RMSNorm have parity and tolerance policy.
4. BF16 coverage gates show fewer CPU exits on representative MLP/normalization paths.
5. Existing FLOAT32 Metal and CPU paths remain stable.

**Planned waves:**
- [x] [30-01 BF16 storage/ABI/materialization path](phases/30-bf16-metal-compute-and-output/30-01-PLAN.md).
- [x] [30-02 BF16 primitive lowering and native execution](phases/30-bf16-metal-compute-and-output/30-02-PLAN.md).
- [x] [30-03 BF16 parity, tolerance, and coverage gates](phases/30-bf16-metal-compute-and-output/30-03-PLAN.md).
- [x] [30-04 Docs and regression closure](phases/30-bf16-metal-compute-and-output/30-04-PLAN.md).

### Phase 31: BOOL-Producing Metal Compute

**Goal:** Let Metal produce and consume device-resident BOOL masks for supported compare/logical/reduction flows.

**Requirements:** METALBOOL-01, METALBOOL-02, METALBOOL-03

**Depends on:** Phase 29.

**Success Criteria:**
1. Compare ops produce device-resident BOOL outputs for supported FLOAT32 inputs.
2. Logical BOOL ops and `WHERE` mask chains stay on Metal without CPU materialization.
3. `REDUCE_ALL` and `REDUCE_ANY` have supported or explicitly rejected Metal semantics.
4. Mask output dtype support is separated from numeric compute dtype support.
5. Coverage reports prove mask chains stayed device-owned.

**Planned waves:**
- [x] [31-01 BOOL output ABI and compare primitive contract](phases/31-bool-producing-metal-compute/31-01-PLAN.md).
- [x] [31-02 Logical and reduction BOOL execution](phases/31-bool-producing-metal-compute/31-02-PLAN.md).
- [x] [31-03 Mask chain residency and `WHERE` consumer gates](phases/31-bool-producing-metal-compute/31-03-PLAN.md).
- [x] [31-04 Docs, parity, and report closure](phases/31-bool-producing-metal-compute/31-04-PLAN.md).

### Phase 32: INT32 Index Tensor And Gather Take Path

**Goal:** Add native INT32 index tensor handling plus forward gather/take Metal execution.

**Requirements:** METALINTIDX-01, METALINTIDX-02, METALINTIDX-03

**Depends on:** Phase 29 and Phase 31 for mask-compatible indexing flows.

**Success Criteria:**
1. INT32 index buffers can be passed through Metal execution state and native ABI safely.
2. Forward `GATHER` and `TAKE_ALONG_AXIS` lower to Metal for scoped dtype/rank/axis cases.
3. Bounds behavior and unsupported index dtype/layout cases are visible and stable.
4. Legal index-forward regions preserve adjacent GPU producers and consumers.
5. CPU parity tests cover representative rank and axis cases.

**Planned waves:**
- [x] [32-01 INT32 ABI, residency, and planner legality](phases/32-int32-index-tensor-and-gather-take-path/32-01-PLAN.md).
- [32-02 Forward gather/take lowering and native execution](phases/32-int32-index-tensor-and-gather-take-path/32-02-PLAN.md).
- [32-03 Bounds/layout parity and visible rejection tests](phases/32-int32-index-tensor-and-gather-take-path/32-03-PLAN.md).
- [32-04 Coverage/report closure](phases/32-int32-index-tensor-and-gather-take-path/32-04-PLAN.md).

### Phase 33: GPU Layout Router And Strided Materialization

**Goal:** Avoid CPU exits by routing layout repair and materialization through GPU-side Metal paths where legal.

**Requirements:** METALLAYOUT-01, METALLAYOUT-02, METALLAYOUT-03

**Depends on:** Phase 29; should feed Phases 34-37.

**Success Criteria:**
1. Layout router classifies metadata-only, dense materialization, broadcast materialization, strided compute, and unsupported cases.
2. Legal `contiguous`, reshape-from-non-dense, zero-stride broadcast repair, and selected strided materialization run GPU-side.
3. Compute legality can consume router-produced dense or strided-compatible bindings without CPU materialization.
4. REQUIRE mode fails before hidden tensor-array or CPU fallback when layout support is missing.
5. Non-contiguous benchmark targets fail on unexpected `CPU_CONSUMER` materialization.

**Planned waves:**
- 33-01 Layout router contract and reason vocabulary.
- 33-02 Metal GPU-side materialization primitives.
- 33-03 Integration with prepared execution and region legality.
- 33-04 Coverage gates and docs.

### Phase 34: Masked And Causal SDPA

**Goal:** Admit verified masked and causal Metal SDPA while preserving mask semantics.

**Requirements:** METALSDPAMASK-01, METALSDPAMASK-02, METALSDPAMASK-03

**Depends on:** Phases 31 and 33.

**Success Criteria:**
1. BOOL mask, additive mask, causal mask, scale, rank, broadcast, and dtype semantics are locked before admission.
2. Supported masked and causal SDPA paths execute through MPSGraph/native DAG or lowered primitives.
3. CPU parity covers supported masks and causal behavior across rank-3/4 cases.
4. Unsupported mask layouts or dtypes reject with stable reason codes.
5. Transformer attention hot-path gates require native Metal evidence for supported masked/causal cases.

**Planned waves:**
- 34-01 Mask semantics and parity contract.
- 34-02 Mask conversion/lowering and native execution.
- 34-03 Causal SDPA support and rejection detail.
- 34-04 Transformer coverage closure.

### Phase 35: Conv Pool Native Execution

**Goal:** Add selected Metal conv/pool native or lowered execution with parity and coverage gates.

**Requirements:** METALCONVPOOL-01, METALCONVPOOL-02, METALCONVPOOL-03

**Depends on:** Phases 29 and 33.

**Success Criteria:**
1. Conv2D forward semantics cover NCHW, stride, padding, dilation, groups, bias, dtype, and layout gates.
2. Max/avg pool forward semantics cover tie behavior, padding, stride, and average divisor rules.
3. Supported conv/pool paths execute through MPSGraph/MPS or custom Metal kernels with CPU parity.
4. Unsupported conv/pool variants keep `CAPABILITY_MISSING` or precise shape/layout reasons.
5. Conv/pool coverage targets report reduced CPU exits and native backend path evidence.

**Planned waves:**
- 35-01 Conv/pool semantic and capability contract.
- 35-02 Conv2D forward native/lowered execution.
- 35-03 Max/avg pool forward native/lowered execution.
- 35-04 Coverage gates, docs, and profile hygiene.

### Phase 36: Scatter And Index Gradient Semantics

**Goal:** Support or explicitly reject duplicate-index scatter and index-gradient paths.

**Requirements:** METALSCATTER-01, METALSCATTER-02, METALSCATTER-03

**Depends on:** Phase 32.

**Success Criteria:**
1. Duplicate-index accumulation order/tolerance semantics are locked for Metal scatter paths.
2. `SCATTER_ADD`, `GATHER_GRAD`, and `TAKE_ALONG_AXIS_GRAD` either execute on Metal or reject with stable duplicate/bounds reasons.
3. CPU parity covers duplicate indices, repeated indices, out-of-range handling, and gradient scatter.
4. Supported scatter/index-gradient flows preserve device residency.
5. Reports distinguish forward index support from index-gradient support.

**Planned waves:**
- 36-01 Scatter/index-gradient semantics contract.
- 36-02 Native or lowered scatter execution.
- 36-03 Gradient parity and duplicate-index gates.
- 36-04 Trace/report closure.

### Phase 37: Loss-Adjacent Metal Lowering

**Goal:** Lower dense and index-target loss-adjacent flows where semantics are proven.

**Requirements:** METALLOSS-01, METALLOSS-02, METALLOSS-03

**Depends on:** Phases 31, 32, 36, and 33.

**Success Criteria:**
1. Dense NLL/CE loss variants lower to supported Metal primitives for scoped dtype/reduction cases.
2. Index-target CE/NLL support preserves INT32 target, ignore-index, class weights, bounds, and denominator semantics.
3. Unsupported loss variants keep visible reason codes instead of shortening adjacent GPU regions silently.
4. Loss-adjacent training workloads reduce CPU boundaries.
5. CPU parity covers reduction modes and ignore-index cases.

**Planned waves:**
- 37-01 Dense loss lowering contract.
- 37-02 Index-target loss semantics and native/lowered support.
- 37-03 Loss backward-adjacent integration.
- 37-04 Coverage/report and docs closure.

### Phase 38: Metal Training Backward Coverage

**Goal:** Keep gradients on Metal for v1.5-supported forward families where legal.

**Requirements:** METALTRAIN-01, METALTRAIN-02, METALTRAIN-03

**Depends on:** Phases 30-37.

**Success Criteria:**
1. Backward coverage matrix reflects every v1.5-supported forward family.
2. Gradients for BF16, BOOL-mask consumers, INT32 index flows, masked SDPA, conv/pool, and loss-adjacent ops execute or reject explicitly.
3. Training traces distinguish true gradient publication from avoidable internal CPU materialization.
4. Representative forward/backward workloads keep supported gradients on Metal.
5. CPU baseline remains the correctness oracle with focused parity tests.

**Planned waves:**
- 38-01 Backward coverage matrix and legality update.
- 38-02 Backward native/lowered execution for supported families.
- 38-03 Training trace and gradient publication gates.
- 38-04 Final training parity closure.

### Phase 39: Metal Backend Router And Zero-Copy Closure

**Goal:** Add calibrated MPSGraph/custom-kernel/CPU routing and close remaining native-copy evidence gaps.

**Requirements:** METALROUTER-01, METALROUTER-02, METALROUTER-03

**Depends on:** Phases 29-38.

**Success Criteria:**
1. Router decisions can choose MPSGraph, custom Metal kernels, tensor-array fallback, buffer binding, or CPU fallback with trace-visible reasons.
2. Cost model uses calibrated shape/dtype/layout evidence without committing local profile artifacts accidentally.
3. The current `nativeDeviceCopyNs` path is either proven as safe output-buffer write behavior or replaced by an explicit lower-copy strategy.
4. Final coverage gates fail hidden CPU exits, tensor-array replay, shorter supported regions, and native-copy regressions.
5. v1.5 milestone audit can verify every Metal support claim from tests, docs, reports, and trace evidence.

**Planned waves:**
- 39-01 Router policy and cost evidence model.
- 39-02 Custom-kernel integration point and MPSGraph routing.
- 39-03 Output-buffer/zero-copy proof or replacement strategy.
- 39-04 Final coverage regression and milestone audit readiness.

## Archived Milestones

<details>
<summary>✅ v1.0 Accelerator Runtime Architecture (Phases 1-5) - SHIPPED 2026-04-30</summary>

Archives:
- [v1.0 roadmap archive](milestones/v1.0-ROADMAP.md)
- [v1.0 requirements archive](milestones/v1.0-REQUIREMENTS.md)
- [v1.0 milestone audit](milestones/v1.0-MILESTONE-AUDIT.md)

</details>

<details>
<summary>✅ v1.1 CUDA Native Runtime (Phases 6-8) - SHIPPED 2026-04-30</summary>

Archives:
- [v1.1 roadmap archive](milestones/v1.1-ROADMAP.md)
- [v1.1 requirements archive](milestones/v1.1-REQUIREMENTS.md)
- [v1.1 milestone audit](milestones/v1.1-MILESTONE-AUDIT.md)
- [v1.1 phase artifacts](milestones/v1.1-phases/)

</details>

<details>
<summary>✅ v1.2 GPU Region Coverage (Phases 9-13) - SHIPPED 2026-05-01</summary>

Archives:
- [v1.2 roadmap archive](milestones/v1.2-ROADMAP.md)
- [v1.2 requirements archive](milestones/v1.2-REQUIREMENTS.md)
- [v1.2 milestone audit](milestones/v1.2-MILESTONE-AUDIT.md)
- [v1.2 phase artifacts](milestones/v1.2-phases/)

</details>

<details>
<summary>✅ v1.3 Coverage-Driven GPU Region Expansion (Phases 14-21) - SHIPPED 2026-05-01</summary>

Archives:
- [v1.3 roadmap archive](milestones/v1.3-ROADMAP.md)
- [v1.3 requirements archive](milestones/v1.3-REQUIREMENTS.md)
- [v1.3 milestone audit](milestones/v1.3-MILESTONE-AUDIT.md)
- [v1.3 phase artifacts](milestones/v1.3-phases/)

</details>

<details>
<summary>✅ v1.4 Native GPU Operation Coverage Closure (Phases 22-28) - SHIPPED 2026-05-02</summary>

Phases:
- [x] Phase 22: Coverage Truth And Semantics Lock - 3/3 plans complete, verified 2026-05-02
- [x] Phase 23: Forward Reductions Native Execution - 3/3 plans complete, verified 2026-05-02
- [x] Phase 24: Normalization GPU Lowering - 4/4 plans complete, verified 2026-05-01
- [x] Phase 25: Forward SDPA Semantic Enablement - 4/4 plans complete, verified 2026-05-01
- [x] Phase 26: Loss-Adjacent And Indexing GPU Coverage - 4/4 plans complete, verified 2026-05-02
- [x] Phase 27: Conv Pool And Bool Compare Outputs - 4/4 plans complete, verified 2026-05-02
- [x] Phase 28: Coverage Regression Closure - 4/4 plans complete, verified 2026-05-02

Archives:
- [v1.4 roadmap archive](milestones/v1.4-ROADMAP.md)
- [v1.4 requirements archive](milestones/v1.4-REQUIREMENTS.md)
- [v1.4 milestone audit](milestones/v1.4-MILESTONE-AUDIT.md)
- [v1.4 phase artifacts](milestones/v1.4-phases/)

</details>

## Current Status

v1.5 is active. Phase 29 Metal DType ABI And Capability Truth is ready for `$gsd-discuss-phase 29 --auto` or `$gsd-plan-phase 29 --auto`.

---
*Last updated: 2026-05-02 after v1.5 milestone planning*
