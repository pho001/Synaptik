# Milestones

## v1.2 GPU Region Coverage (Shipped: 2026-05-01)

**Phases completed:** 5 phases, 19 plans, 41 tasks

**Archives:**

- `milestones/v1.2-ROADMAP.md`
- `milestones/v1.2-REQUIREMENTS.md`
- `milestones/v1.2-MILESTONE-AUDIT.md`
- `milestones/v1.2-phases/`

**Key accomplishments:**

- Backend-neutral layout ABI v2 descriptor with physical-span metadata and portable tests
- Metal and CUDA layout ABI v2 capability probes with optional native symbol support
- ABI-v2-specific fallback reason codes with CUDA non-dense metadata diagnostics
- Backend-neutral GPU layout transform planner with explicit metadata-only, dense materialization, and unsupported decisions
- Metadata-only layout nodes can reuse Metal/CUDA device bindings before CPU materialization
- Optional Metal/CUDA dense layout materialization hooks with conservative runtime service routing
- Backend-neutral Metal/CUDA lowering coverage matrix with stable operation families, statuses, and unsupported reason codes
- Metal and CUDA planner legality now consume the shared lowering coverage matrix while preserving backend-owned gates
- LOG_SOFTMAX now stays in Metal/CUDA GPU regions by lowering to existing SOFTMAX and LOG DAG primitives
- Phase 11 final verification closed with trace-visible LOG_SOFTMAX support, explicit unsupported-family rejection, docs, and profile artifact hygiene
- Backend-neutral GPU compound summaries with explicit CPU FUSED rejection and stable reason codes for Metal/CUDA lowering.
- Metal and CUDA now recognize `linear + bias + activation` as one accelerator DAG-backed `LINEAR_BIAS_ACTIVATION` compound region.
- Metal and CUDA `ADD -> RELU -> EXP` regions now publish `ELEMENTWISE_CHAIN` summaries and traceable compound metadata.

---

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
