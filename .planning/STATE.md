---
gsd_state_version: 1.0
milestone: v1.5
milestone_name: Production-Grade Metal Backend Expansion
status: active
last_updated: "2026-05-02T11:05:00Z"
last_activity: 2026-05-02 -- Phase 38-01 backward coverage truth completed
progress:
  total_phases: 11
  completed_phases: 9
  total_plans: 44
  completed_plans: 37
  percent: 84
---

# GSD State

**Initialized:** 2026-04-29
**Project:** Synaptik
**Current focus:** Phase 37 — Loss-Adjacent Metal Lowering

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-05-02)

**Core value:** Synaptik must produce correct tensor results through a clean compiled graph architecture while letting the optimizer and tuning system choose high-performance CPU or accelerator execution without hidden regressions.

## Current Roadmap

See: `.planning/ROADMAP.md`

Completed milestones:

- v1.0 Accelerator Runtime Architecture - shipped 2026-04-30
- v1.1 CUDA Native Runtime - shipped 2026-04-30
- v1.2 GPU Region Coverage - shipped 2026-05-01
- v1.3 Coverage-Driven GPU Region Expansion - shipped 2026-05-01
- v1.4 Native GPU Operation Coverage Closure - shipped 2026-05-02

Current milestone:

- v1.5 Production-Grade Metal Backend Expansion - active

Recently completed milestone:

- v1.4 Native GPU Operation Coverage Closure - shipped 2026-05-02

## Codebase Map

Current brownfield map:

- `.planning/codebase/STACK.md`
- `.planning/codebase/INTEGRATIONS.md`
- `.planning/codebase/ARCHITECTURE.md`
- `.planning/codebase/STRUCTURE.md`
- `.planning/codebase/CONVENTIONS.md`
- `.planning/codebase/TESTING.md`
- `.planning/codebase/CONCERNS.md`

Planning agents should read the relevant codebase map documents before proposing architecture or execution changes.

## Operating Notes

- This repository has existing uncommitted profile changes under `profiles/platform/.../tuning/abc/*`; they are not part of project initialization.
- `.planning/tmp/` contains temporary verification artifacts and is not part of the initialized roadmap.
- `gsd-sdk` is available for phase/status mutations; keep manual edits scoped when SDK numbering helpers cannot infer padded phase directories.
- v1.0 and v1.1 milestone audits passed and archive files were created under `.planning/milestones/`.
- v1.1 phase execution directories were archived to `.planning/milestones/v1.1-phases/` before starting v1.2.
- v1.2 phase execution directories were archived to `.planning/milestones/v1.2-phases/` before starting the next milestone.
- v1.3 phase execution directories were archived to `.planning/milestones/v1.3-phases/` before starting the next milestone.
- v1.4 phase execution directories were archived to `.planning/milestones/v1.4-phases/` before starting v1.5.

## Phase Planning Status

| Phase | Status | Plans | Verification |
|-------|--------|-------|--------------|
| 29 — Metal DType ABI And Capability Truth | Complete | 4/4 | Verified |
| 30 — BF16 Metal Compute And Output | Complete | 4/4 | Verified |
| 31 — BOOL-Producing Metal Compute | Complete | 4/4 | Verified |
| 32 — INT32 Index Tensor And Gather Take Path | Complete | 4/4 | Verified |
| 33 — GPU Layout Router And Strided Materialization | Complete | 4/4 | Verified |
| 34 — Masked And Causal SDPA | Complete | 4/4 | Verified |
| 35 — Conv Pool Native Execution | Complete | 4/4 | Verified |
| 36 — Scatter And Index Gradient Semantics | Complete | 4/4 | Verified |
| 37 — Loss-Adjacent Metal Lowering | Complete | 4/4 | Verified |
| 38 — Metal Training Backward Coverage | In Progress | 1/4 | Pending |
| 39 — Metal Backend Router And Zero-Copy Closure | Planned | 0/4 | Pending |
| 22 — Coverage Truth And Semantics Lock | Archived | 3/3 | Verified |
| 23 — Forward Reductions Native Execution | Archived | 3/3 | Verified |
| 24 — Normalization GPU Lowering | Archived | 4/4 | Verified |
| 25 — Forward SDPA Semantic Enablement | Archived | 4/4 | Verified |
| 26 — Loss-Adjacent And Indexing GPU Coverage | Archived | 4/4 | Verified |
| 27 — Conv Pool And Bool Compare Outputs | Archived | 4/4 | Verified |
| 28 — Coverage Regression Closure | Archived | 4/4 | Verified |
| 14 — Coverage Gap Triage And Hot Path Targets | Archived | 4/4 | Verified |
| 15 — GPU Region Internal Lowered DAG Contract | Archived | 4/4 | Verified |
| 16 — DType And Storage Residency Expansion | Archived | 4/4 | Verified |
| 17 — Normalization Reduction And Loss-Adjacent Lowering | Archived | 4/4 | Verified |
| 18 — Fused Elementwise And Epilogue Subregions | Archived | 4/4 | Verified |
| 19 — Multi-Op GPU Region Execution | Archived | 5/5 | Verified |
| 20 — Coverage Regression Hardening | Archived | 4/4 | Verified |
| 21 — Milestone Verification Evidence Closure | Archived | 1/1 | Verified |

## Deferred Items

Items acknowledged and deferred at milestone close on 2026-04-30:

| Category | Item | Status |
|----------|------|--------|
| uat_gap | Phase 05 `05-UAT.md` | diagnosed; 0 pending scenarios |

## Recent Decisions

| Date | Decision | Reason |
|------|----------|--------|
| 2026-04-29 | Use native `.planning` artifacts going forward | Avoid cloning old `todo/` phase documents and give GSD a clean planning state. |
| 2026-04-29 | Treat the current roadmap as a brownfield next-milestone roadmap | The codebase already has many shipped capabilities; the next work should focus on accelerator/runtime architecture. |
| 2026-04-29 | Keep public tensors logical and backend residency in runtime state | Maintains clean API design while enabling device-owned execution. |
| 2026-04-29 | Phase 1 planning passed with no compatibility shims | The accelerator buffer ABI should migrate call sites directly to canonical layout-aware contracts. |
| 2026-04-29 | DeviceBufferBinding exposes shared layout/access/native identity | Plan 001-01 established the backend-neutral buffer ABI while keeping Metal handles backend-owned. |
| 2026-04-29 | Metal/CUDA seams consume shared layout ABI without native view execution | Plan 001-02 adapted request, decision, allocator, materializer, and CUDA policy seams while preserving conservative fallback. |
| 2026-04-29 | Phase 1 ABI closure relies on focused regression tests and trace docs | Plan 001-03 proved layout classes, conservative Metal fallback, CUDA required-unavailable behavior, and stable reason-code documentation. |
| 2026-04-29 | Phase 1 verification passed after code-review fixes | Review warnings were fixed in `c35f3fe`; verifier confirmed 10/10 must-haves and no human UAT needed. |
| 2026-04-29 | Phase 2 planning uses dense physical logical-view Metal flow | Plans keep native ABI unchanged for Phase 2, add Metal layout policy/materialization, and verify layout-heavy forward and forward-backward flows. |
| 2026-04-30 | Phase 2 execution started | Execute the verified three-wave plan sequence, beginning with Metal layout policy and binder preflight. |
| 2026-04-30 | Metal buffer preflight uses explicit layout policy | Plan 002-01 added direct dense, dense physical logical-view, and rejected policy actions while keeping output logical-view execution conservative until allocator/materializer support. |
| 2026-04-30 | Metal logical-view flow uses dense physical buffers and Java scatter materialization | Plan 002-02 enabled policy-approved logical-view outputs without native layout ABI churn. |
| 2026-04-30 | Phase 2 verification passed after runtime publication fixes | Plan 002-03 added E2E Metal parity, trace/docs coverage, and fixed accelerator fallback prepared-input allocation plus alias-view output publication. |
| 2026-04-30 | Phase 3 context locks balanced static cost planning | Materialization-aware planning should prefer longer device-owned regions on ties, use full static cost signals, summarize selected/top rejected planner decisions, and preserve CPU hot paths. |
| 2026-04-30 | Phase 3 planning completed with static materialization cost focus | Three plans cover compile-time cost summaries, backend selection/report visibility, CPU safeguards, and the Phase 4 profile/calibration-derived cost model deferral. |
| 2026-04-30 | Phase 3 execution completed targeted safeguards | Static cost traces, backend selection reports, CPU natural/fusion/BLAS guards, and Phase 4 cost-model deferral documentation are in place. |
| 2026-04-30 | Phase 4 planning completed with ownership-first waves | Four plans cover tuning knob ownership, strict profile IO, profile-derived cost factors, and benchmark read-only/doc closure. |
| 2026-04-30 | Phase 4 ownership matrix completed | Graph autotune and platform calibration candidates now validate declared knob assignments against a central ownership registry. |
| 2026-04-30 | Strict platform runtime profile IO completed | Persisted platform runtime profiles now fail loudly for unsupported schema versions and invalid accelerator buffer modes. |
| 2026-04-30 | Phase 4 runtime-derived accelerator costs completed | Backend selection now uses `PROFILE_DERIVED` summaries from `RuntimeConfig` while preserving CPU safeguards. |
| 2026-04-30 | Benchmark profile read-only boundary completed | Benchmark command kinds are explicitly read-only while full, calibration, and autotune remain the profile-writing CLI flows. |
| 2026-04-30 | Phase 4 verification passed | Verifier confirmed 28/28 must-haves across ownership, strict profile IO, runtime-derived costs, and benchmark read-only boundaries. |
| 2026-04-30 | Phase 4 security passed | Security audit closed 12/12 Phase 4 threats with `threats_open: 0`. |
| 2026-04-30 | Phase 4 Nyquist validation passed | Validation audit found 0 gaps and marked all 8 task verification rows green. |
| 2026-04-30 | Phase 5 closure verification passed | Trace/report contracts, closure workload tests, Metal buffer handoff evidence, docs, and source hygiene all passed targeted verification. |
| 2026-04-30 | Phase 6 context captured in auto mode | CUDA native shim, capability probe, shared buffer ABI seam, fallback policy, tests, docs, and hygiene decisions are ready for planning. |
| 2026-04-30 | Phase 6 planning completed | Three plans cover CUDA native shim/build workflow, Java capability diagnostics, and shared buffer ABI prepared-executable seams. |
| 2026-04-30 | Phase 6 verification passed | CUDA shim, capability diagnostics, shared buffer ABI seam, fallback reason codes, and portable tests satisfy CUDA-01/CUDA-02. |
| 2026-04-30 | Phase 7 verification passed | CUDA dense FLOAT32 native buffer execution, graph-output/CPU-consumer materialization, and adjacent CUDA handoff satisfy CUDA-03/CUDA-04/CUDA-05. |
| 2026-04-30 | v1.1 milestone audit found Phase 8 gaps | CUDA-06 and CUDADOC-01/02/03 remain pending because Phase 8 observability closure is not planned or executed yet. |
| 2026-04-30 | Phase 8 final plan evidence passed | CUDA trace/report parity, reason-code coverage, docs, hygiene gates, portable tests, and native CUDA skip status satisfy CUDA-06 and CUDADOC-01/02/03 pending phase verification. |
| 2026-04-30 | Phase 8 verification passed | Verifier confirmed 12/12 must-haves and all CUDA-06/CUDADOC-01/02/03 requirements. |
| 2026-04-30 | v1.1 milestone audit passed | Requirements, phases, integration, flows, and Nyquist validation all passed with no milestone tech debt. |
| 2026-04-30 | v1.1 milestone archived | CUDA Native Runtime archives were written under `.planning/milestones/`; fresh requirements should be defined for the next milestone. |
| 2026-04-30 | v1.2 milestone started | GPU Region Coverage will target non-contiguous/view Metal and CUDA paths, broader lowering, fused GPU regions, and coverage regression gates. |
| 2026-04-30 | Phase 9 context captured in auto mode | ABI v2 is additive, backend-neutral, optional-symbol gated, and limited to metadata/capability/fallback contracts before Phase 10 layout execution. |
| 2026-04-30 | Phase 9 planning completed | Three plans cover shared ABI v2 metadata, Metal/CUDA optional capability symbols, and ABI-specific fallback/required-mode diagnostics. |
| 2026-04-30 | Phase 9 verification passed | Shared ABI v2 metadata, Metal/CUDA capability probes, native optional symbols, CUDA ABI-v2 fallback diagnostics, Metal native tests, and portable targeted gates satisfy GPULAYOUT-01/02/03. |
| 2026-04-30 | Phase 10 verification passed | GPU layout/view transforms, metadata-only residency, dense materialization gates, native Metal parity, CUDA capability-gated checks, security, and Nyquist validation are closed. |
| 2026-04-30 | Phase 11 planning completed | Four plans cover the shared lowering coverage matrix, Metal/CUDA legality alignment, softmax-ish lowering expansion, selected/rejected candidate tests, trace/docs closure, and profile artifact hygiene. |
| 2026-04-30 | Phase 11 shared coverage contract completed | Backend-neutral Metal/CUDA coverage matrix, required family tests, stable unsupported reasons, and docs now cover GPULOWER-01/03 for downstream legality alignment. |
| 2026-04-30 | Phase 11 legality alignment completed | Metal and CUDA planner legality now consume the shared coverage matrix while CUDA exposes stable unsupported reasons and conservative non-dense layout rejection. |
| 2026-04-30 | Phase 11 softmax-ish lowering completed | LOG_SOFTMAX lowers as SOFTMAX followed by LOG for Metal/CUDA regions without adding a native ABI op code, while unsupported reductions/losses remain visible. |
| 2026-04-30 | Phase 11 trace/docs closure completed | Final portable and optional native gates passed or capability-skipped, docs describe GPU-lowerable extension workflow, and local profile artifacts remained unstaged. |
| 2026-04-30 | Phase 11 verification passed | Shared coverage matrix, Metal/CUDA legality alignment, LOG_SOFTMAX lowering, trace/docs closure, security, validation, and profile artifact hygiene satisfy GPULOWER-01/02/03. |
| 2026-04-30 | Phase 12 execution completed | GPU compound lowering now covers `LINEAR_BIAS_ACTIVATION` and `ELEMENTWISE_CHAIN`, rejects `REDUCTION_ADJACENT` and CPU `Operation.OpType.FUSED` with stable diagnostics, and records focused verification evidence. |
| 2026-04-30 | Phase 12 security passed | Security audit closed 13/13 fused GPU region threats with `threats_open: 0`. |
| 2026-04-30 | Phase 12 Nyquist validation passed | Validation audit found 0 gaps, marked all 7 task verification rows green, and kept CUDA native execution capability-gated. |
| 2026-04-30 | Phase 13 context captured in auto-routed next mode | Coverage evidence, representative workload, regression gate, and artifact hygiene decisions are ready for planning. |
| 2026-04-30 | Phase 13 planning completed | Four plans cover the coverage metric/report schema, representative workload baseline comparison, fail-fast hidden-exit regression gates, docs/native gates, validation, and local profile artifact hygiene. |
| 2026-05-01 | v1.3 milestone started | Coverage-Driven GPU Region Expansion will use v1.2 coverage evidence to expand lowered multi-op GPU regions, region-internal fusion, dtype/storage residency, and hard coverage gates. |
| 2026-05-01 | Phase 14 execution completed | GPU coverage gaps, hot-path target registry, triage report renderers, docs, and validation evidence are complete; Phase 15 is ready to plan. |
| 2026-05-01 | Phase 15 context gathered | GPU lowered-region manifest decisions are captured; Phase 15 is ready to plan. |
| 2026-05-01 | Phase 15 planning completed | Four plans cover manifest model/reason vocabulary, lowerer/backend plan construction, trace/report contracts, docs, validation, and artifact hygiene. |
| 2026-05-01 | Phase 15 verification passed | Lowered-region manifest UAT, security, Nyquist validation, and verifier report passed; GPUDAG-01/02/03 are complete and Phase 16 is ready to plan. |
| 2026-05-01 | Phase 17 planning completed | Four plans cover shared normalization/reduction/loss coverage rows, backend legality rejection detail, softmax/loss trace parity evidence, docs, validation, and profile artifact hygiene. |
| 2026-05-01 | Phase 17 execution plans completed | Phase 17 plans 17-01 through 17-04 passed focused verification, validation evidence, docs closure, and profile artifact hygiene. |
| 2026-05-01 | Phase 17 verification passed | Shared coverage rows, backend rejection detail, softmax/loss trace evidence, CPU parity, docs, code review, regression, and source hygiene passed. |
| 2026-05-01 | Phase 18 execution plans completed | GPU fused subpattern trace/report evidence, elementwise and epilogue subregions, CPU fused isolation, docs, and profile artifact hygiene passed focused gates. |
| 2026-05-01 | Phase 23 execution completed | Shared reduction DAG lowering, Metal MPSGraph reductions, CUDA dense FLOAT32 reductions, keep-dims metadata, and coverage gates passed focused verification. |
| 2026-05-01 | Phase 24 planning completed | Normalization GPU lowering plans split shared sub-DAG contract, native primitive execution, planner admission, and parity/coverage closure. |
| 2026-05-01 | Phase 24 execution completed | LayerNorm and RMSNorm now lower to shared normalization DAGs, Metal/CUDA native primitives support epsilon and broadcast execution, coverage rows are supported for legal FLOAT32 cases, and parity/coverage gates passed. |
| 2026-05-01 | Phase 24 verification passed | Security, Nyquist validation, requirement verification, focused Java gates, and native Metal gate passed; CUDA native compile remains hardware/toolchain-gated because `nvcc` is unavailable locally. |
| 2026-05-01 | Phase 25 context captured in auto mode | Forward SDPA admission is semantics-first: unmasked/masked cases are separate, Metal requires verified CPU parity before MPSGraph admission, and CUDA remains explicit support-or-rejection rather than mirrored Metal support. |
| 2026-05-01 | Phase 25 planning completed | Four plans cover SDPA semantics/parity, Metal admission or stable rejection, CUDA support-or-rejection, and transformer coverage/report closure. |
| 2026-05-01 | Phase 25-01 completed | Forward SDPA contract now exposes Phase 25 dtype/rank/shape/scale/mask boundaries, stable prefixed Metal rejection reasons, parity tests, and docs while keeping support matrix status unchanged. |
| 2026-05-01 | Phase 25-02 completed | Metal direct unmasked forward SDPA is admitted through a verified MPSGraph primitive DAG, masked SDPA remains explicitly rejected, and native executable signatures now include scalar DAG bits. |
| 2026-05-01 | Phase 25-03 completed | CUDA direct forward SDPA remains a stable `CAPABILITY_MISSING` fallback with mask, dtype, and layout diagnostics plus prepared-execution CPU parity evidence. |
| 2026-05-01 | Phase 25-04 completed | Transformer SDPA coverage gates now require Metal native evidence and expect visible CUDA `CAPABILITY_MISSING` fallback evidence; profile artifacts remain unstaged. |
| 2026-05-01 | Phase 25 verification passed | Code review was clean, verifier confirmed 5/5 must-haves and GPUSDPA-01/02/03 complete; native CUDA remains toolchain-gated because `nvcc` is unavailable. |
| 2026-05-02 | Phase 27 security and validation passed | Conv/pool and BOOL output support-or-rejection coverage has `threats_open: 0` and Nyquist-compliant automated evidence without claiming native conv/pool or native BOOL-producing GPU compute. |
| 2026-05-02 | Phase 28 planning completed | Four plans cover target policy hardening, supported-family regression gates, final coverage report evidence, and milestone audit readiness for GPUCLOSE-01/02/03. |
| 2026-05-02 | Phase 28 verification passed | Target coverage policies, hard regression gates, suite report coverage deltas, docs, Metal native gate, and artifact hygiene satisfy GPUCLOSE-01/02/03. |
| 2026-05-02 | v1.4 closure evidence normalized | Added missing Phase 22/23 verification, security, and validation artifacts plus Phase 25/26 security and Nyquist frontmatter so milestone audit can verify every active phase consistently. |
| 2026-05-02 | v1.4 milestone archived | Native GPU Operation Coverage Closure archives were written under `.planning/milestones/`; fresh requirements should be defined for the next milestone. |
| 2026-05-02 | v1.5 milestone planned | Production-Grade Metal Backend Expansion roadmap covers dtype ABI, BF16, BOOL, INT32/indexing, layout router, masked SDPA, conv/pool, scatter, loss, training, router, and zero-copy closure. |
| 2026-05-02 | Phase 29 planning completed | Four plans cover Metal dtype capability decisions, dtype ABI v3 optional probes, dtype diagnostics/report truth, docs, tests, and migration closure. |
| 2026-05-02 | Phase 29 verification passed | Metal dtype role decisions, dtype ABI v3 discovery, report-visible reason codes, docs, focused Java tests, classes, and metalTest passed without widening native compute beyond FLOAT32. |
| 2026-05-02 | Phase 30 planning completed | Four plans cover BF16 DAG/native dtype metadata, BF16 buffer/materialization, scoped primitive admission, tolerance/coverage gates, docs, and regression closure. |
| 2026-05-02 | Phase 30-01 completed | Lowered DAG nodes now carry output dtype metadata, native executable cache signatures include output dtype, Metal executable descriptors preserve output dtype, Metal buffer allocation/materialization handles raw BF16 storage, and the additive native dtype-v3 compile symbol preserves BF16 output metadata without changing the legacy FLOAT32 compile path. |
| 2026-05-02 | Phase 30-02 completed | Metal BF16 compute/output is now admitted only for scoped operation families, BF16 external inputs are role-limited to legal BF16 consumers, normalization DAG internals preserve BF16 dtype metadata, native scalar/output dtype handling is dtype-aware, and native Metal BF16 RELU/MATMUL buffer execution passed capability-gated evidence. |
| 2026-05-02 | Phase 30-03 completed | BF16 Metal parity now has exact raw storage roundtrip checks, scoped numeric tolerances for matmul/reduction/normalization/softmax, BF16-specific hot-path targets, suite report dtype residency evidence, and hard Metal BF16 gates that fail on hidden CPU or tensor-array fallback. |
| 2026-05-02 | Phase 30 verification passed | BF16 Metal docs, capability boundaries, scoped support table, troubleshooting guidance, source hygiene, focused Java tests, and native `metalTest` passed; METALBF16-01/02/03 are complete. |
| 2026-05-02 | Phase 31 planning completed | Four plans cover BOOL output ABI and compare primitive contract, native compare/logical/reduction support-or-reject execution, mask-chain residency through `WHERE`, docs, reports, and verification closure. |
| 2026-05-02 | Phase 31-01 completed | Metal compare ops now have stable accelerator DAG ABI codes and BOOL output dtype metadata, capability truth distinguishes scoped compare output from broader BOOL compute, and coverage keeps native execution capability-gated until Phase 31-02 parity lands. |
| 2026-05-02 | Phase 31-02 completed | Metal now lowers and executes native BOOL compare, logical, and REDUCE_ALL/REDUCE_ANY DAG nodes through MPSGraph buffer binding with exact BOOL byte readback, while CUDA remains explicitly unsupported for BOOL-producing compute. |
| 2026-05-02 | Phase 37-01 completed | Dense Metal `NLL_LOSS` and `CROSS_ENTROPY_LOSS` now have a locked planner contract and stable `DAG_PRIMITIVE_UNSUPPORTED` rejection until Phase 37-02 implements backend-owned lowering/execution. |
| 2026-05-02 | Phase 37-02 completed | Scoped dense `NLL_LOSS` and `CROSS_ENTROPY_LOSS` now lower to Metal DAG primitives and pass native parity; index-target loss and gradient variants remain explicit `UNSUPPORTED_INDEX_SEMANTICS` blockers. |
| 2026-05-02 | Phase 37-03 completed | Training traces now prove dense Metal loss forward remains GPU-owned without hidden internal CPU consumer materialization, while index-target loss gradients remain explicit CPU boundaries. |
| 2026-05-02 | Phase 37 verification passed | Dense loss has a separate `dense_loss_small` native Metal target and index-target `cross_entropy_small` remains a visible `UNSUPPORTED_INDEX_SEMANTICS` blocker. |
| 2026-05-02 | Phase 38 planning completed | Four plans cover backward truth/legality, supported backward execution evidence, training trace gates, and final parity/docs closure for `METALTRAIN-01..03`. |
| 2026-05-02 | Phase 38-01 completed | Backward target truth now tracks supported and unsupported training rows separately from forward support, with docs and coverage tests. |

## Recent Sessions

| Date | Completed | Stopped At | Resume |
|------|-----------|------------|--------|
| 2026-04-29 | Completed 001-01-PLAN.md | None | .planning/phases/001-accelerator-buffer-layout-abi/001-01-SUMMARY.md |
| 2026-04-29 | Completed 001-02-PLAN.md | None | .planning/phases/001-accelerator-buffer-layout-abi/001-02-SUMMARY.md |
| 2026-04-29 | Completed 001-03-PLAN.md | None | .planning/phases/001-accelerator-buffer-layout-abi/001-03-SUMMARY.md |
| 2026-04-29 | Verified Phase 1 and fixed review findings | Ready for Phase 2 planning | .planning/phases/001-accelerator-buffer-layout-abi/001-VERIFICATION.md |
| 2026-04-29 | Planned Phase 2 with 3 verified plans | Ready for Phase 2 execution | .planning/phases/002-metal-layout-aware-device-flow/002-01-PLAN.md |
| 2026-04-30 | Started Phase 2 execution | Wave 1 in progress | .planning/phases/002-metal-layout-aware-device-flow/002-01-PLAN.md |
| 2026-04-30 | Completed 002-01-PLAN.md | Ready for 002-02-PLAN.md | .planning/phases/002-metal-layout-aware-device-flow/002-01-SUMMARY.md |
| 2026-04-30 | Completed 002-02-PLAN.md | Ready for 002-03-PLAN.md | .planning/phases/002-metal-layout-aware-device-flow/002-02-SUMMARY.md |
| 2026-04-30 | Completed and verified Phase 2 | Ready for Phase 3 planning | .planning/phases/002-metal-layout-aware-device-flow/002-VERIFICATION.md |
| 2026-04-30 | Gathered Phase 3 context | Ready for Phase 3 planning | .planning/phases/003-materialization-aware-region-planning/003-CONTEXT.md |
| 2026-04-30 | Planned Phase 3 with 3 verified plans | Ready for Phase 3 execution | .planning/phases/003-materialization-aware-region-planning/003-01-PLAN.md |
| 2026-04-30 | Completed 003-01-PLAN.md | Ready for 003-02-PLAN.md | .planning/phases/003-materialization-aware-region-planning/003-01-SUMMARY.md |
| 2026-04-30 | Completed 003-02-PLAN.md | Ready for 003-03-PLAN.md | .planning/phases/003-materialization-aware-region-planning/003-02-SUMMARY.md |
| 2026-04-30 | Completed 003-03-PLAN.md | Ready for Phase 3 verification | .planning/phases/003-materialization-aware-region-planning/003-03-SUMMARY.md |
| 2026-04-30 | Planned Phase 4 with 4 ownership/cost/hygiene waves | Ready for Phase 4 execution | .planning/phases/04-tuning-and-profile-ownership-audit/04-01-PLAN.md |
| 2026-04-30 | Completed 04-01-PLAN.md | Ready for 04-02-PLAN.md | .planning/phases/04-tuning-and-profile-ownership-audit/04-01-SUMMARY.md |
| 2026-04-30 | Completed 04-02-PLAN.md | Ready for 04-03-PLAN.md | .planning/phases/04-tuning-and-profile-ownership-audit/04-02-SUMMARY.md |
| 2026-04-30 | Completed 04-03-PLAN.md | Ready for 04-04-PLAN.md | .planning/phases/04-tuning-and-profile-ownership-audit/04-03-SUMMARY.md |
| 2026-04-30 | Completed 04-04-PLAN.md | Ready for Phase 4 review and verification gates | .planning/phases/04-tuning-and-profile-ownership-audit/04-04-SUMMARY.md |
| 2026-04-30 | Completed and verified Phase 4 | Ready for Phase 4 security/validation closure or Phase 5 planning | .planning/phases/04-tuning-and-profile-ownership-audit/04-VERIFICATION.md |
| 2026-04-30 | Secured Phase 4 | Ready for `$gsd-validate-phase 04` | .planning/phases/04-tuning-and-profile-ownership-audit/04-SECURITY.md |
| 2026-04-30 | Validated Phase 4 | Ready for Phase 5 planning | .planning/phases/04-tuning-and-profile-ownership-audit/04-VALIDATION.md |
| 2026-04-30 | Completed and verified Phase 5 | Ready for milestone re-audit or completion | .planning/phases/05-accelerator-verification-and-documentation-closure/05-VERIFICATION.md |
| 2026-04-30 | Gathered Phase 6 context | Ready for Phase 6 planning | .planning/phases/06-cuda-shim-and-capability-probe/06-CONTEXT.md |
| 2026-04-30 | Planned Phase 6 with 3 verified plans | Ready for Phase 6 execution | .planning/phases/06-cuda-shim-and-capability-probe/06-01-PLAN.md |
| 2026-04-30 | Completed and verified Phase 6 | Ready for Phase 7 planning | .planning/phases/06-cuda-shim-and-capability-probe/06-VERIFICATION.md |
| 2026-04-30 | Gathered Phase 9 context | Ready for Phase 9 planning | .planning/milestones/v1.2-phases/09-native-layout-abi-v2/09-CONTEXT.md |
| 2026-04-30 | Planned Phase 9 with 3 verified plans | Ready for Phase 9 execution | .planning/milestones/v1.2-phases/09-native-layout-abi-v2/09-01-PLAN.md |
| 2026-04-30 | Completed and verified Phase 9 | Ready for Phase 10 planning | .planning/milestones/v1.2-phases/09-native-layout-abi-v2/09-VERIFICATION.md |
| 2026-04-30 | Completed and verified Phase 10 | Ready for Phase 11 context | .planning/milestones/v1.2-phases/10-gpu-layout-transform-and-view-path/10-VERIFICATION.md |
| 2026-04-30 | Gathered Phase 11 context | Ready for Phase 11 planning | .planning/milestones/v1.2-phases/11-gpu-lowering-coverage-matrix/11-CONTEXT.md |
| 2026-04-30 | Planned Phase 11 with 4 verified plans | Ready for Phase 11 execution | .planning/milestones/v1.2-phases/11-gpu-lowering-coverage-matrix/11-01-PLAN.md |
| 2026-04-30 | Completed 11-01 shared GPU lowering coverage contract | Ready for 11-02 legality alignment | .planning/milestones/v1.2-phases/11-gpu-lowering-coverage-matrix/11-01-SUMMARY.md |
| 2026-04-30 | Completed 11-02 Metal/CUDA legality coverage alignment | Ready for 11-03 softmax-ish lowering expansion | .planning/milestones/v1.2-phases/11-gpu-lowering-coverage-matrix/11-02-SUMMARY.md |
| 2026-04-30 | Completed 11-03 softmax-ish lowering expansion | Ready for 11-04 trace and docs closure | .planning/milestones/v1.2-phases/11-gpu-lowering-coverage-matrix/11-03-SUMMARY.md |
| 2026-04-30 | Completed 11-04 lowering coverage trace and docs closure | Ready for Phase 11 security/validation/verify-work | .planning/milestones/v1.2-phases/11-gpu-lowering-coverage-matrix/11-04-SUMMARY.md |
| 2026-04-30 | Completed and verified Phase 11 | Ready for Phase 12 planning | .planning/milestones/v1.2-phases/11-gpu-lowering-coverage-matrix/11-VERIFICATION.md |
| 2026-04-30 | Completed 12-04 reduction-adjacent/docs closure | Ready for Phase 12 security/validation/verify-work | .planning/milestones/v1.2-phases/12-fused-gpu-region-execution/12-04-SUMMARY.md |
| 2026-04-30 | Secured Phase 12 | Ready for `$gsd-validate-phase 12` | .planning/milestones/v1.2-phases/12-fused-gpu-region-execution/12-SECURITY.md |
| 2026-04-30 | Validated Phase 12 | Ready for `$gsd-audit-milestone` | .planning/milestones/v1.2-phases/12-fused-gpu-region-execution/12-VALIDATION.md |
| 2026-04-30 | Gathered Phase 13 context | Ready for `$gsd-plan-phase 13` | .planning/milestones/v1.2-phases/13-coverage-benchmark-and-regression-gate/13-CONTEXT.md |
| 2026-04-30 | Planned Phase 13 with 4 coverage regression waves | Ready for `$gsd-execute-phase 13` | .planning/milestones/v1.2-phases/13-coverage-benchmark-and-regression-gate/13-01-PLAN.md |
| 2026-05-01 | Completed v1.2 milestone archive | Ready for `$gsd-new-milestone` | .planning/milestones/v1.2-ROADMAP.md |
| 2026-05-01 | Planned Phase 14 with 4 coverage triage waves | Ready for `$gsd-execute-phase 14` | .planning/milestones/v1.3-phases/14-coverage-gap-triage-and-hot-path-targets/14-01-PLAN.md |
| 2026-05-01 | Gathered Phase 15 context | Ready for `$gsd-plan-phase 15` | .planning/milestones/v1.3-phases/15-gpu-region-internal-lowered-dag-contract/15-CONTEXT.md |
| 2026-05-01 | Planned Phase 15 with 4 lowered-region manifest waves | Ready for `$gsd-execute-phase 15` | .planning/milestones/v1.3-phases/15-gpu-region-internal-lowered-dag-contract/15-01-PLAN.md |
| 2026-05-01 | Completed and verified Phase 15 | Ready for `$gsd-discuss-phase 16 --auto` or `$gsd-plan-phase 16 --auto` | .planning/milestones/v1.3-phases/15-gpu-region-internal-lowered-dag-contract/15-VERIFICATION.md |
| 2026-05-01 | Planned Phase 16 with 4 dtype residency waves | Ready for `$gsd-execute-phase 16` | .planning/milestones/v1.3-phases/16-dtype-and-storage-residency-expansion/16-01-PLAN.md |
| 2026-05-01 | Completed and verified Phase 16 | Ready for `$gsd-discuss-phase 17 --auto` or `$gsd-plan-phase 17 --auto` | .planning/milestones/v1.3-phases/16-dtype-and-storage-residency-expansion/16-VERIFICATION.md |
| 2026-05-01 | Completed and verified Phase 17 | Ready for `$gsd-discuss-phase 18 --auto` or `$gsd-plan-phase 18 --auto` | .planning/milestones/v1.3-phases/17-normalization-reduction-and-loss-adjacent-lowering/17-VERIFICATION.md |
| 2026-05-02 | Completed and verified Phase 35 | Ready for Phase 36 planning | .planning/phases/35-conv-pool-native-execution/35-VERIFICATION.md |
| 2026-05-02 | Planned Phase 36 with 4 scatter/index-gradient waves | Ready for Phase 36 execution | .planning/phases/36-scatter-and-index-gradient-semantics/36-01-PLAN.md |
| 2026-05-02 | Completed 36-01 scatter/index-gradient semantics contract | Ready for 36-02 native or stable-rejection execution decision | .planning/phases/36-scatter-and-index-gradient-semantics/36-01-SUMMARY.md |
| 2026-05-02 | Completed 36-02 native or stable-rejection scatter execution | Ready for 36-03 gradient parity and duplicate-index gates | .planning/phases/36-scatter-and-index-gradient-semantics/36-02-SUMMARY.md |

---
*Last updated: 2026-05-02 after Phase 36-02 execution*

## Current Position

Phase: 36
Plan: .planning/phases/36-scatter-and-index-gradient-semantics/36-03-PLAN.md
Status: Ready for `$gsd-execute-phase 36 --wave 3`
Last activity: 2026-05-02 — 36-02 stable Metal index-write rejection path completed
