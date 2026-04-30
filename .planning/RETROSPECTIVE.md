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

## Milestone: v1.1 - CUDA Native Runtime

**Shipped:** 2026-04-30
**Phases:** 3 | **Plans:** 10 | **Tasks:** 28

### What Was Built

- Checked-in CUDA graph shim source with optional Gradle build/probe tasks and portable bridge diagnostics.
- Layered CUDA capability diagnostics, shared buffer ABI preflight, and REQUIRED-mode guards for dense `FLOAT32` CUDA paths.
- Native CUDA buffer resources, prepared execution binding, graph-output/CPU-consumer materialization, and adjacent CUDA handoff.
- CUDA trace and benchmark report parity for execution path, reason codes, bytes, copy timing, materialization, and storage residency.
- Developer docs, hygiene gates, portable verification, and native CUDA skip evidence for environments without `nvcc`.

### What Worked

- Capability-gated native checks let the milestone close portably while still documenting real CUDA hardware coverage gaps.
- Fake bridge tests covered CUDA reason-code and required-mode behavior without depending on local GPU availability.
- Keeping CUDA native handles under `backend.cuda.*` preserved the shared accelerator ABI and avoided public `Tensor` API drift.

### What Was Inefficient

- The milestone needed a second audit pass because Phase 6 and Phase 8 Nyquist artifacts lagged behind verification status.
- The earlier audit temporarily treated Phase 8 observability as a gap, so roadmap/state language required cleanup after Phase 8 closure.
- Local tuning profile files stayed dirty and required explicit staging discipline throughout the close.

### Patterns Established

- CUDA buffer support starts narrow: dense `FLOAT32`, same CUDA backend, available native ABI, and supported DAG node types.
- Required CUDA buffer mode throws before tensor-array or CPU fallback can hide native-buffer unavailability.
- CUDA reports use backend-neutral byte/copy/materialization fields, with compatibility fallbacks for older Metal-specific fields.
- Optional native CUDA verification is pass-or-skip by local capability; portable Java gates remain mandatory.

### Key Lessons

1. Keep validation artifacts synchronized with verification immediately, especially when a milestone audit will read both.
2. Preserve narrow runtime support boundaries until native execution, materialization, handoff, and observability are all proven together.
3. Record native-toolchain skip evidence explicitly so portable closure does not masquerade as hardware validation.
4. Continue excluding local tuning/profile output from milestone commits unless intentionally updating canonical fixtures.

### Cost Observations

- Model mix: not recorded for this milestone.
- Sessions: one extended v1.1 run covering Phase 6 through milestone close.
- Notable: portable CUDA fake-bridge coverage reduced dependency on local hardware while preserving reason-code confidence.

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | multiple | 5 | Established GSD phase verification, security, Nyquist validation, UAT diagnosis, and milestone audit as required close gates. |
| v1.1 | multiple | 3 | Added capability-gated CUDA native runtime closure with portable fake-bridge verification and explicit native skip evidence. |

### Cumulative Quality

| Milestone | Requirements | Verification | Open Threats |
|-----------|--------------|--------------|--------------|
| v1.0 | 24/24 satisfied | 5/5 phases passed | 0 |
| v1.1 | 9/9 satisfied | 3/3 phases passed | 0 |

### Top Lessons

1. Artifact consistency is part of done; implementation evidence is insufficient if audit-readable files are stale.
2. Accelerator performance work needs observability contracts as much as implementation changes.
3. Native accelerator work should separate portable correctness gates from local hardware/toolchain evidence.
