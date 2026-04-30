# Milestones

## v1.1 CUDA Native Runtime (Shipped: 2026-04-30)

**Phases completed:** 3 phases, 10 plans, 28 tasks

**Archives:**

- `milestones/v1.1-ROADMAP.md`
- `milestones/v1.1-REQUIREMENTS.md`
- `milestones/v1.1-MILESTONE-AUDIT.md`

**Key accomplishments:**

- Checked-in CUDA graph shim source with optional Gradle build/probe tasks and portable bridge diagnostics.
- Layered CUDA capability diagnostics and shared buffer ABI preflight for dense `FLOAT32` metadata with visible fallback and REQUIRED-mode guards.
- Prepared CUDA execution resolves native buffer bindings, executes accepted dense `FLOAT32` CUDA buffer decisions, and materializes device-owned graph outputs through execution state.
- Adjacent CUDA buffer handoff is covered by portable tests and docs now describe the narrow dense `FLOAT32` CUDA buffer contract.
- CUDA native-buffer execution emits trace and benchmark report evidence for path, reason code, bytes, copy timing, and storage residency.
- CUDA observability closure includes stable required-mode reason-code tests, developer docs, hygiene gates, portable verification, and native CUDA skip evidence.

---

## v1.0 Accelerator Runtime Architecture (Shipped: 2026-04-30)

**Phases completed:** 5 phases, 16 plans, 41 tasks

**Known deferred items at close:** 1 (see `STATE.md` Deferred Items)

**Archives:**

- `milestones/v1.0-ROADMAP.md`
- `milestones/v1.0-REQUIREMENTS.md`
- `milestones/v1.0-MILESTONE-AUDIT.md`

**Key accomplishments:**

- Backend-neutral accelerator layout metadata with Metal bindings migrated to shared layout/access/native identity contracts.
- Metal and CUDA accelerator seams now consume the shared layout ABI while keeping Phase 1 native execution conservative.
- Focused accelerator buffer ABI regression tests plus trace/report documentation for stable layout fallback diagnostics.
- Metal buffer preflight now classifies direct dense, dense physical logical-view, and rejected layouts with stable fake-bridge diagnostics.
- Metal now allocates dense physical buffers for legal logical-view outputs and materializes them through Java-owned layout scatter without native ABI changes.
- End-to-end Metal layout-aware tests now cover device-owned logical-view flow, visible fallback, CPU parity, gradient publication, and trace/documentation diagnostics.
- Static accelerator materialization cost summaries with compile-time selected and top rejected partition finalists
- Prepare-time accelerator cost decisions rendered in text and JSON benchmark reports without expanding runtime step traces
- CPU hot-path regression guards and architecture docs for static materialization-aware accelerator planning
- Central tuning knob ownership validation for graph autotune and platform calibration candidate spaces
- Strict platform runtime profile loading with schema and accelerator buffer validation
- Accelerator backend selection cost summaries derived from audited RuntimeConfig thresholds
- Benchmark commands now have explicit read-only profile roles, and Phase 4 ownership docs are closed.
- Benchmark reports now expose stable accelerator reason, fallback, planner, materialization, copy-time, and residency evidence through portable Java tests.
- Transformer-block workload, adjacent device-buffer handoff, native Metal trace, and CUDA required-unavailable coverage now prove the accelerator closure evidence path.
- Accelerator closure is documented, local artifact hygiene is enforced, and final targeted Java plus Metal verification passed.

---
