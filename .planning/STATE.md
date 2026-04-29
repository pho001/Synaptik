# GSD State

**Initialized:** 2026-04-29
**Project:** Synaptik
**Current focus:** Phase 1 — Accelerator Buffer Layout ABI complete; ready to plan/execute Phase 2 — Metal Layout-Aware Device Flow

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
- Continue with Phase 2, building on the completed Accelerator Buffer Layout ABI.

## Phase Planning Status

| Phase | Status | Plans | Verification |
|-------|--------|-------|--------------|
| 1 — Accelerator Buffer Layout ABI | Complete | 3 of 3 plans complete | 001-03 targeted suite and classes passed on 2026-04-29 |

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

## Recent Sessions

| Date | Completed | Stopped At | Resume |
|------|-----------|------------|--------|
| 2026-04-29 | Completed 001-01-PLAN.md | None | .planning/phases/001-accelerator-buffer-layout-abi/001-01-SUMMARY.md |
| 2026-04-29 | Completed 001-02-PLAN.md | None | .planning/phases/001-accelerator-buffer-layout-abi/001-02-SUMMARY.md |
| 2026-04-29 | Completed 001-03-PLAN.md | None | .planning/phases/001-accelerator-buffer-layout-abi/001-03-SUMMARY.md |

---
*Last updated: 2026-04-29 after 001-03 execution*
