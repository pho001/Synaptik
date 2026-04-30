---
gsd_state_version: 1.0
milestone: v1.2
milestone_name: GPU Region Coverage
status: verifying
last_updated: "2026-04-30T19:56:01Z"
last_activity: 2026-04-30
progress:
  total_phases: 5
  completed_phases: 3
  total_plans: 15
  completed_plans: 15
  percent: 100
---

# GSD State

**Initialized:** 2026-04-29
**Project:** Synaptik
**Current focus:** Phase 12 — fused-gpu-region-execution

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-04-30)

**Core value:** Synaptik must produce correct tensor results through a clean compiled graph architecture while letting the optimizer and tuning system choose high-performance CPU or accelerator execution without hidden regressions.

## Current Roadmap

See: `.planning/ROADMAP.md`

Completed milestones:

- v1.0 Accelerator Runtime Architecture - shipped 2026-04-30
- v1.1 CUDA Native Runtime - shipped 2026-04-30

Current milestone:

- v1.2 GPU Region Coverage - defining requirements and roadmap for broader Metal/CUDA GPU-resident region coverage.

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

## Phase Planning Status

| Phase | Status | Plans | Verification |
|-------|--------|-------|--------------|
| 9 — Native Layout ABI v2 | Completed | 3/3 | Passed |
| 10 — GPU Layout Transform And View Path | Completed | 4/4 | Passed |
| 11 — GPU Lowering Coverage Matrix | Completed | 4/4 | Passed |
| 12 — Fused GPU Region Execution | Executed | 4/4 | Pending |
| 13 — Coverage Benchmark And Regression Gate | Planned | TBD | Pending |

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
| 2026-04-30 | Gathered Phase 9 context | Ready for Phase 9 planning | .planning/phases/09-native-layout-abi-v2/09-CONTEXT.md |
| 2026-04-30 | Planned Phase 9 with 3 verified plans | Ready for Phase 9 execution | .planning/phases/09-native-layout-abi-v2/09-01-PLAN.md |
| 2026-04-30 | Completed and verified Phase 9 | Ready for Phase 10 planning | .planning/phases/09-native-layout-abi-v2/09-VERIFICATION.md |
| 2026-04-30 | Completed and verified Phase 10 | Ready for Phase 11 context | .planning/phases/10-gpu-layout-transform-and-view-path/10-VERIFICATION.md |
| 2026-04-30 | Gathered Phase 11 context | Ready for Phase 11 planning | .planning/phases/11-gpu-lowering-coverage-matrix/11-CONTEXT.md |
| 2026-04-30 | Planned Phase 11 with 4 verified plans | Ready for Phase 11 execution | .planning/phases/11-gpu-lowering-coverage-matrix/11-01-PLAN.md |
| 2026-04-30 | Completed 11-01 shared GPU lowering coverage contract | Ready for 11-02 legality alignment | .planning/phases/11-gpu-lowering-coverage-matrix/11-01-SUMMARY.md |
| 2026-04-30 | Completed 11-02 Metal/CUDA legality coverage alignment | Ready for 11-03 softmax-ish lowering expansion | .planning/phases/11-gpu-lowering-coverage-matrix/11-02-SUMMARY.md |
| 2026-04-30 | Completed 11-03 softmax-ish lowering expansion | Ready for 11-04 trace and docs closure | .planning/phases/11-gpu-lowering-coverage-matrix/11-03-SUMMARY.md |
| 2026-04-30 | Completed 11-04 lowering coverage trace and docs closure | Ready for Phase 11 security/validation/verify-work | .planning/phases/11-gpu-lowering-coverage-matrix/11-04-SUMMARY.md |
| 2026-04-30 | Completed and verified Phase 11 | Ready for Phase 12 planning | .planning/phases/11-gpu-lowering-coverage-matrix/11-VERIFICATION.md |
| 2026-04-30 | Completed 12-04 reduction-adjacent/docs closure | Ready for Phase 12 security/validation/verify-work | .planning/phases/12-fused-gpu-region-execution/12-04-SUMMARY.md |
| 2026-04-30 | Secured Phase 12 | Ready for `$gsd-validate-phase 12` | .planning/phases/12-fused-gpu-region-execution/12-SECURITY.md |

---
*Last updated: 2026-04-30 after Phase 12 security*

## Current Position

Phase: 12 (fused-gpu-region-execution) — VERIFYING
Plan: 4 of 4
Status: Security passed; validation pending
Last activity: 2026-04-30
