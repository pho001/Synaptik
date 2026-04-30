# GSD State

**Initialized:** 2026-04-29
**Project:** Synaptik
**Current focus:** Phase 3 planned — ready to execute Materialization-Aware Region Planning.

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-04-29)

**Core value:** Synaptik must produce correct tensor results through a clean compiled graph architecture while letting the optimizer and tuning system choose high-performance CPU or accelerator execution without hidden regressions.

## Current Roadmap

See: `.planning/ROADMAP.md`

Current milestone phases:

1. Accelerator Buffer Layout ABI
2. Metal Layout-Aware Device Flow
3. Materialization-Aware Region Planning
4. Tuning And Profile Ownership Audit
5. Accelerator Verification And Documentation Closure

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
- `gsd-sdk` is not available in the current shell, so initialization artifacts were generated directly from the GSD workflow templates and current repository context.
- Continue with Phase 2 execution, building on the verified Accelerator Buffer Layout ABI.

## Phase Planning Status

| Phase | Status | Plans | Verification |
|-------|--------|-------|--------------|
| 1 — Accelerator Buffer Layout ABI | Complete | 3 of 3 plans complete | `001-VERIFICATION.md` passed 10/10 must-haves on 2026-04-29 |
| 2 — Metal Layout-Aware Device Flow | Complete | 3 of 3 plans complete | `002-VERIFICATION.md` passed 12/12 must-haves on 2026-04-30 |
| 3 — Materialization-Aware Region Planning | Planned | 3 plans | Not started |

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

---
*Last updated: 2026-04-30 after Phase 3 planning*
