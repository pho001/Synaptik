# GSD State

**Initialized:** 2026-04-29
**Project:** Synaptik
**Current focus:** Phase 1 — Accelerator Buffer Layout ABI

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
- Use `$gsd-plan-phase 1` next to create the implementation plan for Phase 1.

## Recent Decisions

| Date | Decision | Reason |
|------|----------|--------|
| 2026-04-29 | Use native `.planning` artifacts going forward | Avoid cloning old `todo/` phase documents and give GSD a clean planning state. |
| 2026-04-29 | Treat the current roadmap as a brownfield next-milestone roadmap | The codebase already has many shipped capabilities; the next work should focus on accelerator/runtime architecture. |
| 2026-04-29 | Keep public tensors logical and backend residency in runtime state | Maintains clean API design while enabling device-owned execution. |

---
*State initialized: 2026-04-29*
