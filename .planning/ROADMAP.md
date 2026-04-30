# Roadmap: Synaptik

## Milestones

- **v1.0 Accelerator Runtime Architecture** - Phases 1-5 shipped 2026-04-30. Full archive: [v1.0-ROADMAP.md](milestones/v1.0-ROADMAP.md)
- **v1.1 CUDA Native Runtime** - Phases 6-8 planned. Scope: checked-in CUDA native runtime path, shared ABI buffer execution, materialization, handoff, observability, and documentation.

## Summary

| # | Phase | Goal | Requirements | Success Criteria |
|---|-------|------|--------------|------------------|
| 6 | CUDA Shim And Capability Probe | Add the checked-in CUDA native shim, build workflow, capability probe, and Java-side shared ABI seam. | CUDA-01, CUDA-02 | 5 |
| 7 | CUDA Buffer Execution And Materialization | Prove native CUDA device buffers can execute, materialize, and hand off across adjacent accelerator work. | CUDA-03, CUDA-04, CUDA-05 | 5 |
| 8 | CUDA Observability And Documentation Closure | Make CUDA fallback, report evidence, docs, hygiene, and final verification match the Metal-era observability contract. | CUDA-06, CUDADOC-01, CUDADOC-02, CUDADOC-03 | 5 |

## Archived Milestones

<details>
<summary>v1.0 Accelerator Runtime Architecture (Phases 1-5) - SHIPPED 2026-04-30</summary>

- [x] Phase 1: Accelerator Buffer Layout ABI (3/3 plans) - verified 2026-04-29
- [x] Phase 2: Metal Layout-Aware Device Flow (3/3 plans) - verified 2026-04-30
- [x] Phase 3: Materialization-Aware Region Planning (3/3 plans) - verified 2026-04-30
- [x] Phase 4: Tuning And Profile Ownership Audit (4/4 plans) - verified 2026-04-30
- [x] Phase 5: Accelerator Verification And Documentation Closure (3/3 plans) - verified 2026-04-30

Archives:
- [v1.0 roadmap archive](milestones/v1.0-ROADMAP.md)
- [v1.0 requirements archive](milestones/v1.0-REQUIREMENTS.md)
- [v1.0 milestone audit](milestones/v1.0-MILESTONE-AUDIT.md)

</details>

## Phase Details

### Phase 6: CUDA Shim And Capability Probe

**Goal:** Add a checked-in CUDA native shim, documented build workflow, runtime capability probe, and Java-side bridge seam that consumes the shared accelerator buffer ABI while failing gracefully on machines without CUDA.

**Requirements:** CUDA-01, CUDA-02

**Depends on:** v1.0 accelerator buffer layout ABI and existing CUDA bridge/prepared executable scaffolding.

**Success Criteria:**
1. CUDA native shim source and build instructions are checked in, with a targeted build/probe command documented.
2. Java runtime capability probing reports CUDA availability, native shim availability, and buffer execution support without crashing when CUDA is absent.
3. CUDA bridge and prepared executable seams accept shared `AcceleratorBufferLayout` / access metadata for supported dense layouts.
4. Unsupported or unavailable CUDA native paths remain capability-gated and visible through stable reason codes.
5. Focused tests cover available, unavailable, unsupported dtype/layout, and required-buffer-mode behavior without requiring CUDA hardware for the portable gate.

**Notes:**
- Keep common accelerator contracts backend-neutral; CUDA-native handles stay inside CUDA packages.
- Do not broaden operation coverage yet beyond what is needed to prove the runtime path.
- Preserve existing CPU and Metal behavior while CUDA support is unavailable.

**Planned Execution:**

Wave 1:
- `06-01` — Add checked-in CUDA native shim source, optional build/probe workflow, Gradle tasks, docs, and portable bridge tests.
- `06-02` — Add Java CUDA capability model and reason-coded bridge diagnostics while keeping buffer support conservative.

Wave 2 *(blocked on Wave 1 completion)*:
- `06-03` — Wire CUDA prepared execution to the shared accelerator buffer ABI for dense supported layouts and preserve REQUIRED-mode failure semantics.

Cross-cutting constraints:
- Default Java lifecycle tasks remain CUDA-independent.
- `supportsBufferBindings()` stays false until Java and native CUDA buffer execution are both safe to claim.
- CUDA-specific native handles and lifetimes stay under `backend.cuda.*`; shared accelerator records and public `Tensor` API remain backend-neutral.

### Phase 7: CUDA Buffer Execution And Materialization

**Goal:** Prove CUDA native device buffers can run a representative accelerator operation, materialize correct CPU-visible results, and hand device-owned buffers to adjacent CUDA work without Java array round trips.

**Requirements:** CUDA-03, CUDA-04, CUDA-05

**Depends on:** Phase 6.

**Success Criteria:**
1. CUDA buffer execution allocates native device buffers and runs at least one representative supported accelerator operation through the native path when capability checks pass.
2. CUDA graph output and CPU consumer materialization produce CPU-parity results for supported dtype/layout combinations.
3. Adjacent CUDA accelerator regions can reuse device-owned buffers without tensor-array fallback when layout and capability contracts allow it.
4. CUDA execution distinguishes native buffer execution, tensor-array fallback, CPU fallback, and required-unavailable paths in tests.
5. Native CUDA tests are capability-gated, while portable Java tests prove graceful behavior on machines without CUDA.

**Notes:**
- CPU remains the correctness oracle.
- If native CUDA coverage must start with a narrow operation set, document that explicitly and keep unsupported operations visibly falling back.
- Avoid benchmark/profile persistence changes in this phase.

### Phase 8: CUDA Observability And Documentation Closure

**Goal:** Close v1.1 by making CUDA fallback, trace/report evidence, developer docs, source hygiene, and final verification match the observability standard established for Metal.

**Requirements:** CUDA-06, CUDADOC-01, CUDADOC-02, CUDADOC-03

**Depends on:** Phase 6, Phase 7.

**Success Criteria:**
1. CUDA trace and benchmark reports expose backend, buffer path, reason code, fallback reason, prepared input usage, materialization count/reason, copy timing, and storage residency.
2. Required-mode and fallback failures use stable reason codes for unavailable native runtime, unsupported dtype, unsupported layout, and required-but-unavailable execution.
3. Developer docs explain CUDA build prerequisites, capability probing, fallback interpretation, and troubleshooting.
4. Hygiene checks prevent accidental commits of local CUDA build outputs, native scratch files, and benchmark/profile artifacts.
5. Final targeted verification covers portable Java tests, capability-gated native CUDA checks, docs checks, and no regression to CPU/Metal safeguards.

**Notes:**
- CUDA benchmark evidence is a report contract, not a request to commit measured local benchmark output.
- Any unavailable local CUDA hardware/tooling must result in explicit skipped native checks plus passing portable checks.

## Coverage

| Requirement | Phase | Status |
|-------------|-------|--------|
| CUDA-01 | Phase 6 | Complete |
| CUDA-02 | Phase 6 | Complete |
| CUDA-03 | Phase 7 | Pending |
| CUDA-04 | Phase 7 | Pending |
| CUDA-05 | Phase 7 | Pending |
| CUDA-06 | Phase 8 | Pending |
| CUDADOC-01 | Phase 8 | Pending |
| CUDADOC-02 | Phase 8 | Pending |
| CUDADOC-03 | Phase 8 | Pending |

**Coverage:**
- v1.1 requirements: 9 total
- Mapped to phases: 9
- Unmapped: 0

---
*Roadmap updated: 2026-04-30 after Phase 6 completion*
