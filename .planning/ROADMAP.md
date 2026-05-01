# Roadmap: Synaptik

## Milestones

- ✅ **v1.0 Accelerator Runtime Architecture** - Phases 1-5 shipped 2026-04-30. Full archive: [v1.0-ROADMAP.md](milestones/v1.0-ROADMAP.md)
- ✅ **v1.1 CUDA Native Runtime** - Phases 6-8 shipped 2026-04-30. Full archive: [v1.1-ROADMAP.md](milestones/v1.1-ROADMAP.md)
- ✅ **v1.2 GPU Region Coverage** - Phases 9-13 shipped 2026-05-01. Full archive: [v1.2-ROADMAP.md](milestones/v1.2-ROADMAP.md)
- 🚧 **v1.3 Coverage-Driven GPU Region Expansion** - Phases 14-20 planned. Scope: coverage-driven hot-path triage, GPU internal lowered DAGs, dtype/storage residency, broader lowering, region-internal fusion, multi-op GPU region execution, and hard coverage gates.

## Summary

| # | Phase | Goal | Requirements | Success Criteria |
|---|-------|------|--------------|------------------|
| 14 | Coverage Gap Triage And Hot Path Targets | Use v1.2 coverage evidence to rank the CPU exits and materialization boundaries that should drive v1.3 work. | GPUTRIAGE-01, GPUTRIAGE-02, GPUTRIAGE-03 (Complete 2026-05-01) | 5 |
| 15 | GPU Region Internal Lowered DAG Contract | 1/4 | In Progress|  |
| 16 | DType And Storage Residency Expansion | Extend device residency, memory binding, slot reuse, and fallback diagnostics for BFLOAT16, INT32, and BOOL where they block GPU regions. | GPUSTORAGE-01, GPUSTORAGE-02, GPUSTORAGE-03 | 5 |
| 17 | Normalization Reduction And Loss-Adjacent Lowering | Expand or explicitly reject high-impact normalization, reduction, softmax-ish, and loss-adjacent GPU lowering gaps under a shared Metal/CUDA contract. | GPUNORM-01, GPUNORM-02, GPUNORM-03 | 5 |
| 18 | Fused Elementwise And Epilogue Subregions | Add region-internal GPU fusion for elementwise chains and linear/matmul epilogues without reusing CPU fused ASM nodes. | GPUFUSEX-01, GPUFUSEX-02, GPUFUSEX-03 | 5 |
| 19 | Multi-Op GPU Region Execution | Execute longer GPU regions containing multiple lowered operations, layout/view steps, elementwise chains, and selected softmax-ish or normalization primitives. | GPUMULTI-01, GPUMULTI-02, GPUMULTI-03 | 5 |
| 20 | Coverage Regression Hardening | Harden reports and gates so hot paths staying on GPU becomes an auditable invariant, not a performance impression. | GPUHARDEN-01, GPUHARDEN-02, GPUHARDEN-03 | 5 |

## Milestone Rule

GPU fusion is region-internal lowering/fusion, not reuse of CPU fused ASM nodes.

Partitioning selects a device-owned region; region lowering expands it into backend primitives; fusion optimizes supported subgraphs inside that lowered region. CPU `Operation.OpType.FUSED`, generated ASM, and CPU vector dispatch remain CPU backend internals.

## Archived Milestones

<details>
<summary>✅ v1.0 Accelerator Runtime Architecture (Phases 1-5) - SHIPPED 2026-04-30</summary>

- [x] Phase 1: Accelerator Buffer Layout ABI - verified 2026-04-29
- [x] Phase 2: Metal Layout-Aware Device Flow - verified 2026-04-30
- [x] Phase 3: Materialization-Aware Region Planning - verified 2026-04-30
- [x] Phase 4: Tuning And Profile Ownership Audit - verified 2026-04-30
- [x] Phase 5: Accelerator Verification And Documentation Closure - verified 2026-04-30

Archives:
- [v1.0 roadmap archive](milestones/v1.0-ROADMAP.md)
- [v1.0 requirements archive](milestones/v1.0-REQUIREMENTS.md)
- [v1.0 milestone audit](milestones/v1.0-MILESTONE-AUDIT.md)

</details>

<details>
<summary>✅ v1.1 CUDA Native Runtime (Phases 6-8) - SHIPPED 2026-04-30</summary>

- [x] Phase 6: CUDA Shim And Capability Probe - verified 2026-04-30
- [x] Phase 7: CUDA Buffer Execution And Materialization - verified 2026-04-30
- [x] Phase 8: CUDA Observability And Documentation Closure - verified 2026-04-30

Archives:
- [v1.1 roadmap archive](milestones/v1.1-ROADMAP.md)
- [v1.1 requirements archive](milestones/v1.1-REQUIREMENTS.md)
- [v1.1 milestone audit](milestones/v1.1-MILESTONE-AUDIT.md)
- [v1.1 phase artifacts](milestones/v1.1-phases/)

</details>

<details>
<summary>✅ v1.2 GPU Region Coverage (Phases 9-13) - SHIPPED 2026-05-01</summary>

- [x] Phase 9: Native Layout ABI v2 - verified 2026-04-30
- [x] Phase 10: GPU Layout Transform And View Path - verified 2026-04-30
- [x] Phase 11: GPU Lowering Coverage Matrix - verified 2026-04-30
- [x] Phase 12: Fused GPU Region Execution - verified 2026-04-30
- [x] Phase 13: Coverage Benchmark And Regression Gate - verified 2026-05-01

Archives:
- [v1.2 roadmap archive](milestones/v1.2-ROADMAP.md)
- [v1.2 requirements archive](milestones/v1.2-REQUIREMENTS.md)
- [v1.2 milestone audit](milestones/v1.2-MILESTONE-AUDIT.md)
- [v1.2 phase artifacts](milestones/v1.2-phases/)

</details>

## Phase Details

### Phase 14: Coverage Gap Triage And Hot Path Targets

**Goal:** Use v1.2 coverage evidence to rank the CPU exits and materialization boundaries that should drive v1.3 work.

**Requirements:** GPUTRIAGE-01, GPUTRIAGE-02, GPUTRIAGE-03

**Depends on:** v1.2 Phase 13 coverage reports and regression gates.

**Success Criteria:**
1. v1.2 coverage reports are parsed or summarized into a stable triage view of fallback, CPU materialization, tensor-array fallback, and device handoff reasons by backend and workload.
2. Transformer block, MLP, and conv/normalization-style workloads have deterministic coverage baselines with region length, CPU exits, and unsupported reason evidence.
3. Coverage gaps are ranked by hot-path impact and by the requirement family likely to close them: dtype/storage, lowering, fusion, multi-op execution, or regression hardening.
4. The triage output distinguishes real backend execution from CPU replay and from capability-skipped native evidence.
5. The phase produces a checked-in target list that later phases can use as source-of-truth scope instead of adding operations opportunistically.

**Plans:**

**Status:** 4/4 plans complete as of 2026-05-01.

Wave 1:
- [14-01 Coverage Gap Triage Model](phases/14-coverage-gap-triage-and-hot-path-targets/14-01-PLAN.md) - adds deterministic gap categories, records, ranking, and tests for GPUTRIAGE-01/GPUTRIAGE-03.

Wave 2 *(blocked on Wave 1 completion)*:
- [14-02 Hot Path Coverage Targets](phases/14-coverage-gap-triage-and-hot-path-targets/14-02-PLAN.md) - defines transformer, MLP, conv, and normalization target registry and suite request tests for GPUTRIAGE-02/GPUTRIAGE-03.

Wave 3 *(blocked on Wave 1 and Wave 2 completion)*:
- [14-03 Triage Report And Target List](phases/14-coverage-gap-triage-and-hot-path-targets/14-03-PLAN.md) - adds text/JSON triage reports and `14-HOT-PATH-TARGETS.md` handoff for GPUTRIAGE-01/02/03.

Wave 4 *(blocked on Wave 1, Wave 2, and Wave 3 completion)*:
- [14-04 Docs Validation And Triage Closure](phases/14-coverage-gap-triage-and-hot-path-targets/14-04-PLAN.md) - closes docs, focused verification, validation evidence, and artifact hygiene for GPUTRIAGE-01/02/03.

**Cross-cutting constraints:**
- Coverage and materialization behavior are the milestone prioritization mechanism; raw timing is supporting evidence only.
- CUDA native evidence remains capability-gated on machines without CUDA hardware.
- Do not commit local benchmark/profile artifacts unless intentionally promoted to canonical fixtures.
- Phase 14 triage ranks coverage/materialization behavior, not raw timing.
- Tensor-array fallback must stay distinct from native buffer GPU coverage.
- Target workload names must be checked against `StandardWorkloads.defaultCatalog()`.
- `14-HOT-PATH-TARGETS.md` is the source-of-truth target list for Phases 15-20.
- Local tuning profile artifacts are not canonical coverage evidence.

**Notes:**
- This phase should prevent random operation chasing. Every later implementation phase should point back to a measured hot-path gap or explicit stable rejection.

### Phase 15: GPU Region Internal Lowered DAG Contract

**Goal:** Define a GPU region as a lowered DAG with original ops, backend primitives, fused subpatterns, and stable rejection metadata.

**Requirements:** GPUDAG-01, GPUDAG-02, GPUDAG-03

**Depends on:** Phase 14.

**Success Criteria:**
1. The accelerator planner can describe a selected GPU region as a multi-op lowered DAG rather than a single operation or opaque compound label.
2. Region metadata includes original operation IDs/types, lowered primitive IDs/types, backend ID, dtype/layout assumptions, fused subpattern summaries, and selected region length.
3. Rejection metadata can point to the original op, lowered primitive, or fused subpattern that forced fallback or materialization.
4. Trace/debug output renders the internal DAG in a stable format suitable for tests and benchmark reports.
5. Existing CPU execution, CPU fusion, and public tensor semantics remain unchanged.

**Plans:**

1/4 plans executed

Wave 1:
- [15-01 Manifest Model And Reason Vocabulary](phases/15-gpu-region-internal-lowered-dag-contract/15-01-PLAN.md) - adds the Java-side lowered-region manifest records and stable DAG-level reason codes for GPUDAG-01/02/03.

Wave 2 *(blocked on Wave 1 completion)*:
- [15-02 Manifest Construction And Backend Plan Exposure](phases/15-gpu-region-internal-lowered-dag-contract/15-02-PLAN.md) - builds manifests from shared accelerator lowering and exposes them through selected Metal/CUDA plans for GPUDAG-01/02.

Wave 3 *(blocked on Wave 1 and Wave 2 completion)*:
- [15-03 Trace And Report Manifest Contract](phases/15-gpu-region-internal-lowered-dag-contract/15-03-PLAN.md) - attaches selected manifests to prepare/backend-selection trace and renders stable text/JSON report fields for GPUDAG-02/03.

Wave 4 *(blocked on Wave 1, Wave 2, and Wave 3 completion)*:
- [15-04 Docs Validation And Manifest Closure](phases/15-gpu-region-internal-lowered-dag-contract/15-04-PLAN.md) - closes docs, final focused verification, validation evidence, and artifact hygiene for GPUDAG-01/02/03.

**Cross-cutting constraints:**
- Public `Tensor` remains logical; lowered DAG state belongs to compile/prepare/execute internals.
- Metal and CUDA share the DAG metadata contract, but backend-specific primitives remain backend-owned.
- Stable reason codes are required before broadening execution so failures are diagnosable.
- The manifest is Java-side metadata and must not change the Metal or CUDA native ABI.
- Prepare/backend-selection trace is the source of truth for structured manifests; run trace should reference only compact region id/runtime outcome evidence.
- GPU fusion metadata is region-internal lowering/fusion metadata, not CPU `Operation.OpType.FUSED`.
- Local tuning profile artifacts are not canonical coverage evidence.

**Notes:**
- This is the architectural bridge between v1.2 compound summaries and v1.3 longer multi-op GPU regions.

### Phase 16: DType And Storage Residency Expansion

**Goal:** Extend device residency, memory binding, slot reuse, and fallback diagnostics for BFLOAT16, INT32, and BOOL where they block GPU regions.

**Requirements:** GPUSTORAGE-01, GPUSTORAGE-02, GPUSTORAGE-03

**Depends on:** Phase 14 and Phase 15.

**Success Criteria:**
1. Device buffer metadata, memory binding, and slot reuse can represent BFLOAT16, INT32, and BOOL values when backend capability allows residency.
2. Metal and CUDA dtype decisions report backend-specific unsupported reasons instead of silently forcing CPU materialization.
3. Supported dtype-resident values can pass through internal GPU-region boundaries without Java array round trips.
4. CPU parity tests cover true output/consumer materialization boundaries for supported non-FLOAT32/FLOAT64 flows.
5. CPU memory reuse and hot-path dispatch are not regressed by accelerator dtype residency changes.

**Cross-cutting constraints:**
- Dtype residency does not imply universal native arithmetic support; execution legality remains per operation, layout, dtype, and backend capability.
- BFLOAT16, INT32, and BOOL should be added where they close measured region gaps, not as an all-combinations promise.
- Existing FLOAT32/FLOAT64 behavior must remain stable.

**Notes:**
- This phase should focus on residency and binding first; operation-specific dtype support belongs to the lowering phases that consume it.

### Phase 17: Normalization Reduction And Loss-Adjacent Lowering

**Goal:** Expand or explicitly reject high-impact normalization, reduction, softmax-ish, and loss-adjacent GPU lowering gaps under a shared Metal/CUDA contract.

**Requirements:** GPUNORM-01, GPUNORM-02, GPUNORM-03

**Depends on:** Phase 14, Phase 15, and Phase 16.

**Success Criteria:**
1. The shared lowering coverage matrix includes layer norm, RMS norm, reductions, softmax-ish residual flows, and loss-adjacent operation families.
2. The top measured hot-path gaps in these families are implemented as GPU lowering or recorded as stable rejections with precise reasons.
3. Numerically sensitive flows preserve CPU parity with tolerances appropriate to dtype and backend capability.
4. Metal and CUDA legality remains shared at the semantic contract level while backend-specific execution and capability checks remain backend-owned.
5. Tests cover selected and rejected candidates, including dtype/layout/capability rejection and required-mode behavior.

**Cross-cutting constraints:**
- Unsupported reductions and losses must not disappear behind generic CPU replay.
- Layout and dtype legality from earlier phases must be honored by every new lowering path.
- The milestone should prefer high-impact region closure over low-impact isolated op support.

**Notes:**
- "Softmax-ish" includes residual pieces that commonly break transformer-like regions after v1.2 `LOG_SOFTMAX` coverage.

### Phase 18: Fused Elementwise And Epilogue Subregions

**Goal:** Add region-internal GPU fusion for elementwise chains and linear/matmul epilogues without reusing CPU fused ASM nodes.

**Requirements:** GPUFUSEX-01, GPUFUSEX-02, GPUFUSEX-03

**Depends on:** Phase 15 and Phase 17.

**Success Criteria:**
1. Supported elementwise chains inside a GPU region can execute as fused subpatterns without intermediate CPU materialization or Java array round trips.
2. Supported linear/matmul epilogues, including bias and activation, can lower as region-internal fused subpatterns when dtype/layout/backend gates allow it.
3. GPU fusion metadata records fused subpattern type, original operation span, lowered primitive count, and rejection reasons.
4. GPU fusion explicitly rejects CPU `Operation.OpType.FUSED` and never depends on CPU fused ASM/vector internals.
5. CPU fused execution tests continue to pass unchanged, proving GPU fusion did not invade CPU backend contracts.

**Cross-cutting constraints:**
- Fusion is an optimization inside a selected GPU region, not an alternate public operation model.
- Fused subpatterns must not shorten regions or hide fallback.
- CPU and GPU fusion can share semantic operation knowledge, but not CPU implementation internals.

**Notes:**
- This phase deepens the Phase 12 compound-region direction into a general region-internal fusion contract.

### Phase 19: Multi-Op GPU Region Execution

**Goal:** Execute longer GPU regions containing multiple lowered operations, layout/view steps, elementwise chains, and selected softmax-ish or normalization primitives.

**Requirements:** GPUMULTI-01, GPUMULTI-02, GPUMULTI-03

**Depends on:** Phase 15, Phase 16, Phase 17, and Phase 18.

**Success Criteria:**
1. A selected GPU region can execute multiple lowered operations as one device-owned runtime region with internal device handoffs.
2. Layout/view steps, dtype-resident intermediates, elementwise chains, and selected normalization or softmax-ish primitives can coexist in one supported region.
3. `ExecutionState` and device buffer bindings carry supported internal values without CPU materialization until a true CPU consumer, graph output, or gradient publication boundary.
4. Metal and CUDA use the shared planning and metadata contract while dispatching through backend-specific primitive execution routes.
5. Representative transformer block, MLP, and conv/normalization-style workloads show longer supported GPU regions or fewer CPU exits than the Phase 14 baseline.

**Cross-cutting constraints:**
- Multi-op GPU execution must stay capability-gated and must fail visibly in REQUIRED mode.
- Unsupported internal steps should split or reject regions with specific reasons rather than corrupting residency state.
- Tensor-array bridge execution must not count as native buffer GPU coverage.

**Notes:**
- This is the milestone's main integration phase: earlier contracts and coverage expansions should converge into longer device-owned execution.

### Phase 20: Coverage Regression Hardening

**Goal:** Harden reports and gates so hot paths staying on GPU becomes an auditable invariant, not a performance impression.

**Requirements:** GPUHARDEN-01, GPUHARDEN-02, GPUHARDEN-03

**Depends on:** Phase 14 through Phase 19.

**Success Criteria:**
1. Trace and benchmark reports include lowered operation count, fused subpattern count, selected region length, rejected candidate count, CPU exit count, materialization reasons, and device handoff evidence.
2. Regression gates fail when target hot paths unexpectedly shorten GPU regions, add CPU materialization, hide tensor-array fallback, or lose required lowered/fused coverage.
3. Portable Java gates prove report and fallback contracts on machines without native CUDA or Metal availability.
4. Native Metal/CUDA checks remain capability-gated and document skipped evidence clearly.
5. Documentation explains the v1.3 coverage evidence contract and the difference between canonical fixtures and local benchmark/profile artifacts.

**Cross-cutting constraints:**
- Coverage gates should be deterministic and reviewable; timing-only failures are not enough.
- Local `profiles/platform/.../tuning/abc/*` artifacts remain unstaged unless intentionally promoted.
- The final milestone audit should be able to answer whether the selected hot paths stayed on GPU.

**Notes:**
- This phase closes the loop: the target list from Phase 14 becomes an enforced regression contract.

## Coverage Check

| Requirement Category | Requirements | Phase |
|----------------------|--------------|-------|
| Coverage Gap Triage | GPUTRIAGE-01, GPUTRIAGE-02, GPUTRIAGE-03 | Phase 14 |
| GPU Region Internal Lowered DAG Contract | GPUDAG-01, GPUDAG-02, GPUDAG-03 | Phase 15 |
| DType And Storage Residency | GPUSTORAGE-01, GPUSTORAGE-02, GPUSTORAGE-03 | Phase 16 |
| Normalization Reduction And Loss-Adjacent Lowering | GPUNORM-01, GPUNORM-02, GPUNORM-03 | Phase 17 |
| Fused Elementwise And Epilogue Subregions | GPUFUSEX-01, GPUFUSEX-02, GPUFUSEX-03 | Phase 18 |
| Multi-Op GPU Region Execution | GPUMULTI-01, GPUMULTI-02, GPUMULTI-03 | Phase 19 |
| Coverage Regression Hardening | GPUHARDEN-01, GPUHARDEN-02, GPUHARDEN-03 | Phase 20 |

**Requirements:** 21 total, 21 mapped, 0 unmapped.

---
*Roadmap created: 2026-05-01 for v1.3 Coverage-Driven GPU Region Expansion*
