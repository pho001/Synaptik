# Roadmap: Synaptik

## Milestones

- ✅ **v1.0 Accelerator Runtime Architecture** - Phases 1-5 shipped 2026-04-30. Full archive: [v1.0-ROADMAP.md](milestones/v1.0-ROADMAP.md)
- ✅ **v1.1 CUDA Native Runtime** - Phases 6-8 shipped 2026-04-30. Full archive: [v1.1-ROADMAP.md](milestones/v1.1-ROADMAP.md)
- 🚧 **v1.2 GPU Region Coverage** - Phases 9-13 planned. Scope: non-contiguous/view Metal and CUDA paths, broader operation lowering, fused GPU regions, and coverage regression gates.

## Summary

| # | Phase | Goal | Requirements | Success Criteria |
|---|-------|------|--------------|------------------|
| 9 | Native Layout ABI v2 | Extend Metal and CUDA native bridge contracts to carry non-contiguous/view layout metadata with capability/version checks and explicit fallback. | GPULAYOUT-01, GPULAYOUT-02, GPULAYOUT-03 | 5 |
| 10 | GPU Layout Transform And View Path | Keep legal layout transforms and view-like values device-resident across compatible Metal and CUDA regions. | GPUVIEW-01, GPUVIEW-02, GPUVIEW-03 | 5 |
| 11 | GPU Lowering Coverage Matrix | Broaden Metal/CUDA operation lowering for common NN/tensor patterns and document supported/fallback/unsupported coverage. | GPULOWER-01, GPULOWER-02, GPULOWER-03 | 5 |
| 12 | Fused GPU Region Execution | Execute safe compound GPU regions for linear+bias+activation, elementwise chains, and reduction-adjacent candidates without CPU round trips. | GPUFUSE-01, GPUFUSE-02, GPUFUSE-03, GPUFUSE-04 | 5 |
| 13 | Coverage Benchmark And Regression Gate | Prove GPU coverage improvements and prevent hidden CPU exits with trace, benchmark, and regression gates. | GPUCOV-01, GPUCOV-02, GPUCOV-03 | 5 |

## Archived Milestones

<details>
<summary>✅ v1.0 Accelerator Runtime Architecture (Phases 1-5) - SHIPPED 2026-04-30</summary>

- [x] Phase 1: Accelerator Buffer Layout ABI (3/3 plans) - verified 2026-04-29
- [x] Phase 2: Metal Layout-Aware Device Flow (3/3 plans) - verified 2026-04-30
- [x] Phase 3: Materialization-Aware Region Planning (3/3 plans) - verified 2026-04-30
- [x] Phase 4: Tuning And Profile Ownership Audit (4/4 plans) - verified 2026-04-30
- [x] Phase 5: Accelerator Verification And Documentation Closure (3/3 plans) - verified 2026-04-30

Archives:
- [v1.0 roadmap archive](milestones/v1.0-ROADMAP.md)
- [v1.0 requirements archive](milestones/v1.0-REQUIREMENTS.md)
- [v1.0 milestone audit](milestones/v1.0-MILESTONE-AUDIT.md)

</details>

<details>
<summary>✅ v1.1 CUDA Native Runtime (Phases 6-8) - SHIPPED 2026-04-30</summary>

- [x] Phase 6: CUDA Shim And Capability Probe (3/3 plans) - verified 2026-04-30
- [x] Phase 7: CUDA Buffer Execution And Materialization (3/3 plans) - verified 2026-04-30
- [x] Phase 8: CUDA Observability And Documentation Closure (4/4 plans) - verified 2026-04-30

Archives:
- [v1.1 roadmap archive](milestones/v1.1-ROADMAP.md)
- [v1.1 requirements archive](milestones/v1.1-REQUIREMENTS.md)
- [v1.1 milestone audit](milestones/v1.1-MILESTONE-AUDIT.md)
- [v1.1 phase artifacts](milestones/v1.1-phases/)

</details>

## Phase Details

### Phase 9: Native Layout ABI v2

**Goal:** Extend Metal and CUDA native bridge contracts to carry non-contiguous/view layout metadata with capability/version checks and explicit fallback.

**Requirements:** GPULAYOUT-01, GPULAYOUT-02, GPULAYOUT-03

**Depends on:** v1.1 CUDA Native Runtime and v1.0 shared accelerator buffer layout ABI.

**Success Criteria:**
1. Shared Java records represent layout ABI v2 metadata: rank, shape, strides, storage offset, logical element count, physical byte span, access mode, backend id, and native handle identity.
2. Metal and CUDA bridge capability probes report layout ABI version/support without crashing when native symbols or hardware are unavailable.
3. Backend-native ABI changes are optional-symbol/version gated and do not break existing dense buffer execution.
4. Unsupported layout metadata, dtype, rank, aliasing, or ABI mismatch uses stable reason codes in AUTO and REQUIRED modes.
5. Focused portable tests cover supported metadata, missing-symbol fallback, ABI mismatch, and required-mode failure for both Metal and CUDA seams.

**Plans:**

Wave 1:
- [09-01 Shared ABI v2 Metadata Contract](phases/09-native-layout-abi-v2/09-01-PLAN.md) — creates backend-neutral descriptor/status records and physical-span tests for GPULAYOUT-01.

Wave 2 *(blocked on Wave 1 completion)*:
- [09-02 Metal/CUDA Layout ABI v2 Capability Handshake](phases/09-native-layout-abi-v2/09-02-PLAN.md) — adds optional-symbol capability/version discovery and native stub symbols for GPULAYOUT-02.

Wave 3 *(blocked on Wave 1 and Wave 2 completion)*:
- [09-03 ABI v2 Fallback And Required-Mode Diagnostics](phases/09-native-layout-abi-v2/09-03-PLAN.md) — maps unsupported metadata and ABI mismatch to stable AUTO/REQUIRE reason codes for GPULAYOUT-03.

**Cross-cutting constraints:**
- Layout ABI v2 is additive; existing dense Metal/CUDA buffer execution must keep working when v2 symbols are absent.
- Shared contracts stay backend-neutral and do not expose Metal/CUDA native handle objects or a public device tensor API.
- Capability records distinguish native library, graph ABI, v1 buffer ABI, and layout ABI v2 support.
- AUTO fallback must be visible in decisions/traces; REQUIRE must fail before hidden tensor-array or CPU fallback.
- Portable Java tests are required; native Metal/CUDA checks remain capability-gated.

**Notes:**
- Native handle lifetimes stay backend-owned; common contracts describe metadata and capability only.
- Do not force all operations to consume layout ABI v2 in this phase; establish the bridge contract first.
- Metal/CUDA native code changes must be paired with Java FFM signature and status-code tests.

### Phase 10: GPU Layout Transform And View Path

**Goal:** Keep legal layout transforms and view-like values device-resident across compatible Metal and CUDA regions.

**Requirements:** GPUVIEW-01, GPUVIEW-02, GPUVIEW-03

**Depends on:** Phase 9.

**Success Criteria:**
1. `reshape`, `permute`, `expand`, `contiguous`, alias outputs, and legal view-like graph values have backend-neutral GPU layout request/decision records.
2. Metal and CUDA can preserve device residency for supported view/layout flows without CPU materialization between compatible accelerator nodes.
3. CPU graph outputs and CPU consumers materialize correct values with CPU parity after GPU-side layout transforms or logical-view materialization.
4. Unsupported view/layout paths fall back visibly with stable reasons and do not corrupt residency state.
5. Tests cover at least one layout-heavy forward flow and one forward/backward or gradient-publication boundary where applicable.

**Plans:**

4/4 plans complete
- [10-01 Shared GPU Layout Transform Contract](phases/10-gpu-layout-transform-and-view-path/10-01-PLAN.md) - creates backend-neutral request/decision records and stable reason codes for GPUVIEW-01 and GPUVIEW-03.

Wave 2 *(blocked on Wave 1 completion)*:
- [10-02 Device Layout View Propagation](phases/10-gpu-layout-transform-and-view-path/10-02-PLAN.md) - propagates metadata-only layout/view bindings before CPU materialization for GPUVIEW-01, GPUVIEW-02, and GPUVIEW-03.

Wave 3 *(blocked on Wave 1 and Wave 2 completion)*:
- [10-03 Dense GPU Layout Materialization](phases/10-gpu-layout-transform-and-view-path/10-03-PLAN.md) - adds capability-gated Metal/CUDA dense materialization for `contiguous()` and non-contiguous-source `reshape`.

Wave 4 *(blocked on Wave 1, Wave 2, and Wave 3 completion)*:
- [10-04 Layout Transform E2E Verification Closure](phases/10-gpu-layout-transform-and-view-path/10-04-PLAN.md) - closes with layout-heavy forward, gradient publication, trace, docs, and final verification evidence.

**Cross-cutting constraints:**
- Public `Tensor` stays logical; device residency remains in `ExecutionState` and `DeviceBufferBinding`.
- Metadata-only view propagation and dense GPU materialization stay separate paths with explicit decision reason codes.
- Alias/view bindings reuse backend handles without registering duplicate native resources.
- AUTO fallback must stay visible; REQUIRE mode must fail before hidden CPU fallback.
- Direct non-dense CUDA compute remains conservative until Phase 11 unless a metadata-only view or dense materialization path has made the flow legal.

**Notes:**
- Prefer GPU-side layout transform or logical metadata preservation over Java array round trips.
- CPU remains the correctness oracle for all layout/view parity checks.
- Public `Tensor` stays logical; residency remains in execution state and buffer bindings.

### Phase 11: GPU Lowering Coverage Matrix

**Goal:** Broaden Metal/CUDA operation lowering for common NN/tensor patterns and document supported/fallback/unsupported coverage.

**Requirements:** GPULOWER-01, GPULOWER-02, GPULOWER-03

**Depends on:** Phase 9, Phase 10.

**Success Criteria:**
1. A checked-in coverage matrix lists Metal and CUDA support status for matmul/linear, elementwise chains, reductions, softmax-like flows, normalization pieces, and loss-adjacent operations.
2. Lowering support expands for the highest-value supported patterns while preserving stable rejection reasons for unsupported operations, dtypes, layouts, and capability gaps.
3. Backend selection keeps supported patterns in GPU regions when cost, layout, dtype, and capability contracts allow it.
4. Portable tests prove lowering decisions and CPU fallback safeguards for both selected and rejected candidates.
5. Documentation explains how to add new GPU-lowerable operation families without duplicating backend-neutral planning logic.

**Plans:**

4/4 plans complete

Wave 1:
- [11-01 Shared GPU Lowering Coverage Contract](phases/11-gpu-lowering-coverage-matrix/11-01-PLAN.md) - creates backend-neutral coverage statuses, operation families, stable unsupported reasons, matrix tests, and docs for GPULOWER-01/03.

Wave 2 *(blocked on Wave 1 completion)*:
- [11-02 Metal/CUDA Legality Coverage Alignment](phases/11-gpu-lowering-coverage-matrix/11-02-PLAN.md) - wires Metal and CUDA legality adapters through the shared coverage matrix while preserving backend-owned dtype/layout/capability gates.

Wave 3 *(blocked on Wave 1 and Wave 2 completion)*:
- [11-03 Softmax-Ish Lowering Expansion](phases/11-gpu-lowering-coverage-matrix/11-03-PLAN.md) - supports `LOG_SOFTMAX` as `SOFTMAX` followed by `LOG` using existing accelerator DAG primitives and adds selected/rejected candidate tests.

Wave 4 *(blocked on Wave 1, Wave 2, and Wave 3 completion)*:
- [11-04 Lowering Coverage Trace And Docs Closure](phases/11-gpu-lowering-coverage-matrix/11-04-PLAN.md) - closes with trace visibility, layout-heavy flow coverage, developer docs, focused verification commands, and profile artifact hygiene.

**Cross-cutting constraints:**
- Public `Tensor` stays logical; residency stays in `ExecutionState` and `DeviceBufferBinding`.
- Metal and CUDA share semantic coverage classifications, but backend-specific native capability, dtype, layout ABI, and bridge checks remain backend-owned.
- CUDA direct non-dense compute remains conservative unless Phase 10 metadata-only view propagation or dense materialization makes the consumer layout legal.
- Reductions, normalization, and loss-adjacent flows must be explicit matrix entries and visibly rejected unless a narrow safe implementation is added.
- Phase 12 owns fused GPU compound execution; Phase 13 owns coverage benchmark ratios and regression gates.

**Notes:**
- Prioritize operation patterns that unlock longer representative regions rather than chasing isolated low-impact ops.
- Keep operation semantics in descriptors/public tensor ops; lowering should consume existing semantics.
- Avoid benchmark/profile persistence changes in this phase.

### Phase 12: Fused GPU Region Execution

**Goal:** Execute safe compound GPU regions for linear+bias+activation, elementwise chains, and reduction-adjacent candidates without CPU round trips.

**Requirements:** GPUFUSE-01, GPUFUSE-02, GPUFUSE-03, GPUFUSE-04

**Depends on:** Phase 11.

**Success Criteria:**
1. Metal and CUDA execute at least one linear + bias + activation fused GPU region with device-owned intermediates and CPU parity.
2. Metal and CUDA execute representative elementwise-chain fused GPU regions without Java array round trips between fused operations.
3. Fused GPU region planning is backend-specific compound DAG execution and does not depend on CPU ASM/vector fused implementation internals.
4. Reduction-adjacent fusion candidates are either implemented with parity tests or rejected with coverage-matrix entries and stable unsupported reasons.
5. CPU fused execution tests continue to pass and CPU hot paths remain independent of GPU fusion changes.

**Plans:**

4/4 plans executed

Wave 1:
- [12-01 Shared GPU Compound Pattern Contract](phases/12-fused-gpu-region-execution/12-01-PLAN.md) - creates backend-neutral compound summaries, stable reason codes, CPU `FUSED` GPU rejection, and coverage matrix updates for GPUFUSE-03/GPUFUSE-04.

Wave 2 *(blocked on Wave 1 completion)*:
- [12-02 Linear Bias Activation Compound Lowering](phases/12-fused-gpu-region-execution/12-02-PLAN.md) - wires `LINEAR_BIAS_ACTIVATION` through accelerator DAG lowering, Metal/CUDA partition plans, and required-mode fallback tests for GPUFUSE-01/GPUFUSE-03.

Wave 3 *(blocked on Wave 1 and Wave 2 completion)*:
- [12-03 Elementwise Chain Compound Execution Trace](phases/12-fused-gpu-region-execution/12-03-PLAN.md) - exposes `ELEMENTWISE_CHAIN` through prepared accelerator executables, run trace attributes, and device-residency tests for GPUFUSE-02/GPUFUSE-03.

Wave 4 *(blocked on Wave 1, Wave 2, and Wave 3 completion)*:
- [12-04 Reduction Adjacent And Compound Docs Closure](phases/12-fused-gpu-region-execution/12-04-PLAN.md) - closes reduction-adjacent support/rejection, CPU fused safeguards, docs, verification summary, and hygiene evidence for GPUFUSE-01/02/03/04.

**Cross-cutting constraints:**
- GPU compound lowering extends `LoweringPipeline`, `AcceleratorSubgraphLowerer`, and Metal/CUDA region lowerers; it is not a parallel CPU-style fusion system.
- Pattern summary plus accelerator DAG are both required; summaries such as `LINEAR_BIAS_ACTIVATION`, `ELEMENTWISE_CHAIN`, and `REDUCTION_ADJACENT` must not bypass backend legality checks.
- `Operation.OpType.FUSED` remains CPU-only for Phase 12 and must reject explicitly on GPU paths.
- Supported compound target patterns must not shorten regions or materialize CPU/tensor-array intermediates between supported fused operations.
- Metal and CUDA coverage remains backend-specific and trace-visible; CUDA must not claim unsupported broad parity.

**Notes:**
- This is not a request to copy the CPU fused ASM model; GPU fusion should fit accelerator lowering and native backend capabilities.
- Start with safe, testable patterns before broadening to larger fused kernels.
- Required-mode failures must throw before hidden CPU/tensor-array fallback.

### Phase 13: Coverage Benchmark And Regression Gate

**Goal:** Prove GPU coverage improvements and prevent hidden CPU exits with trace, benchmark, and regression gates.

**Requirements:** GPUCOV-01, GPUCOV-02, GPUCOV-03

**Depends on:** Phase 9, Phase 10, Phase 11, Phase 12.

**Success Criteria:**
1. Trace and benchmark reports expose GPU coverage ratio, selected region length, rejected candidate reasons, fallback counts, CPU materialization count/reason, copy timing, storage residency, and device handoff counts for Metal and CUDA.
2. Representative transformer block, MLP, and convolution- or normalization-heavy workloads show fewer GPU-to-CPU exits or longer GPU-covered regions than the v1.1 baseline.
3. Regression gates fail when supported target workloads lose GPU coverage, add unexpected CPU materialization boundaries, or hide fallback behind tensor-array execution.
4. Native Metal/CUDA checks remain capability-gated while portable Java gates prove fallback/report contracts on machines without native tooling.
5. Documentation and hygiene checks make it clear which benchmark/profile outputs are evidence contracts versus local artifacts that must not be committed.

**Notes:**
- The milestone success metric is coverage and materialization behavior, not only raw speed.
- Do not commit machine-local benchmark/calibration output unless intentionally creating stable fixtures.
- Use targeted Gradle filters where full `./gradlew test` remains too slow.

## Coverage

| Requirement | Phase | Status |
|-------------|-------|--------|
| GPULAYOUT-01 | Phase 9 | Complete |
| GPULAYOUT-02 | Phase 9 | Complete |
| GPULAYOUT-03 | Phase 9 | Complete |
| GPUVIEW-01 | Phase 10 | Complete |
| GPUVIEW-02 | Phase 10 | Complete |
| GPUVIEW-03 | Phase 10 | Complete |
| GPULOWER-01 | Phase 11 | Complete |
| GPULOWER-02 | Phase 11 | Complete |
| GPULOWER-03 | Phase 11 | Complete |
| GPUFUSE-01 | Phase 12 | Complete |
| GPUFUSE-02 | Phase 12 | Complete |
| GPUFUSE-03 | Phase 12 | Complete |
| GPUFUSE-04 | Phase 12 | Complete |
| GPUCOV-01 | Phase 13 | Pending |
| GPUCOV-02 | Phase 13 | Pending |
| GPUCOV-03 | Phase 13 | Pending |

**Coverage:**
- v1.2 requirements: 16 total
- Mapped to phases: 16
- Unmapped: 0

---
*Roadmap updated: 2026-04-30 after Phase 12 execution*
