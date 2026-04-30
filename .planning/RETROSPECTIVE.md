# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.0 - Accelerator Runtime Architecture

**Shipped:** 2026-04-30
**Phases:** 5 | **Plans:** 16 | **Tasks:** 41

### What Was Built

- Backend-neutral accelerator buffer layout ABI for Metal now and CUDA later.
- Metal layout-aware device flow using dense physical logical-view buffers and explicit CPU materialization boundaries.
- Materialization-aware accelerator planning with selected/rejected cost summaries and CPU hot-path safeguards.
- Tuning/profile ownership split between graph autotune and platform calibration, with runtime-derived accelerator costs.
- Accelerator observability closure through report contracts, transformer-block workload coverage, Metal/CUDA gates, docs, and source hygiene.

### What Worked

- Verification gates stayed focused and fast by using targeted Gradle filters instead of default benchmark-heavy test runs.
- Keeping `Tensor` logical and moving residency into runtime state preserved the public API while enabling accelerator work.
- Explicit trace/report contracts made fallback, materialization, copy timing, and residency testable rather than advisory.

### What Was Inefficient

- Planning metadata lagged implementation twice: Phase 5 needed a late `05-VERIFICATION.md`, and roadmap/project status needed cleanup before the final audit could pass.
- Local tuning profile files remained dirty throughout the close and required repeated staging discipline.

### Patterns Established

- Backend-neutral accelerator contracts live in common packages; native handles and native ABI assumptions stay backend-owned.
- Legal logical views can use dense physical device buffers while Java owns logical scatter materialization until a versioned native layout ABI exists.
- Benchmark reports are explain artifacts and profile-read-only; autotune/calibration own persistence.
- Milestone closure should always run audit-open, milestone audit, explicit staging checks, and archive generation before deleting living requirements.

### Key Lessons

1. Treat phase-level `VERIFICATION.md` as a hard artifact gate before milestone audit.
2. Keep audit status, project status, roadmap coverage, and state files synchronized immediately after verification.
3. Guard local profile/calibration output with tests, ignore rules, and explicit `git add` paths.
4. Runtime-derived cost inputs should flow through audited `RuntimeConfig`, not local profile file reads.

### Cost Observations

- Model mix: not recorded for this milestone.
- Sessions: one extended close-out run plus phase execution sessions recorded in `STATE.md`.
- Notable: targeted verification and report-contract tests reduced the need for full-suite reruns while still preserving milestone confidence.

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | multiple | 5 | Established GSD phase verification, security, Nyquist validation, UAT diagnosis, and milestone audit as required close gates. |

### Cumulative Quality

| Milestone | Requirements | Verification | Open Threats |
|-----------|--------------|--------------|--------------|
| v1.0 | 24/24 satisfied | 5/5 phases passed | 0 |

### Top Lessons

1. Artifact consistency is part of done; implementation evidence is insufficient if audit-readable files are stale.
2. Accelerator performance work needs observability contracts as much as implementation changes.
