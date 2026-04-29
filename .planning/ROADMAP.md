# Roadmap: Synaptik Accelerator Runtime Architecture

**Created:** 2026-04-29
**Scope:** Brownfield milestone for backend-neutral device buffer layout, Metal/CUDA-ready accelerator execution, region planning, tuning/profile cleanup, and verification.

## Summary

| # | Phase | Goal | Requirements | Success Criteria |
|---|-------|------|--------------|------------------|
| 1 | Accelerator Buffer Layout ABI | Extend the shared device buffer contract so accelerator backends can represent logical tensor layouts, not only dense contiguous arrays. | ABI-01, ABI-02, ABI-03, ABI-04 | 5 |
| 2 | Metal Layout-Aware Device Flow | Use the layout-aware ABI in Metal so legal view/layout values can stay device-resident across accelerator work. | METAL-01, METAL-02, METAL-03, METAL-04 | 5 |
| 3 | Materialization-Aware Region Planning | Make accelerator offload decisions account for CPU/GPU boundaries, layout fallback, and dispatch overhead. | PLAN-01, PLAN-02, PLAN-03, PLAN-04 | 5 |
| 4 | Tuning And Profile Ownership Audit | Separate graph-specific autotune decisions from platform/dtype calibration and harden profile persistence. | TUNE-01, TUNE-02, TUNE-03, TUNE-04 | 5 |
| 5 | Accelerator Verification And Documentation Closure | Prove the new flow through tests, traces, benchmarks, docs, and hygiene checks. | OBS-01, OBS-02, OBS-03, OBS-04, DOC-01, DOC-02, DOC-03, DOC-04 | 6 |

## Phase Details

### Phase 1: Accelerator Buffer Layout ABI

**Status:** Complete — verified 2026-04-29 (`001-VERIFICATION.md`, score 10/10)

**Goal:** Extend the shared runtime device buffer model so Metal and future CUDA backends can describe logical tensor views, including strides and storage offsets, without baking Metal-specific assumptions into common code.

**Requirements:** ABI-01, ABI-02, ABI-03, ABI-04

**Depends on:** Existing `backend.memory.DeviceBufferBinding`, `backend.accelerator.buffer`, `backend.metal.buffer`, and `graph.execution.ExecutionState`.

**Success Criteria:**
1. `DeviceBufferBinding` or a closely related backend-neutral record exposes layout metadata needed to describe contiguous, offset, strided, permuted, and broadcast-like logical tensors.
2. Metal bindings adapt to the shared contract without losing native handle/access information.
3. CUDA bridge/executable seams can consume the same shared contract even if native CUDA buffers remain unavailable.
4. Buffer decision reason codes distinguish unsupported layout classes clearly enough for traces and benchmark reports.
5. Existing Metal buffer binding tests and CUDA buffer policy tests pass, with new tests covering layout metadata compatibility.

**Notes:**
- Keep public `Tensor` API unchanged.
- Avoid making the common ABI depend on `MTLBuffer` or CUDA-specific handle classes.
- Preserve conservative fallback for layouts that cannot yet be executed safely.

**Plans:** 3 plans

Plans:
- [x] 001-01-PLAN.md — Create the shared accelerator buffer layout/access ABI and attach it to runtime bindings. Complete: `.planning/phases/001-accelerator-buffer-layout-abi/001-01-SUMMARY.md`
- [x] 001-02-PLAN.md — Adapt Metal and CUDA seams to the shared layout contract while preserving conservative fallback. Complete: `.planning/phases/001-accelerator-buffer-layout-abi/001-02-SUMMARY.md`
- [x] 001-03-PLAN.md — Add focused tests and trace/report documentation for layout classes and stable fallback reasons. Complete: `.planning/phases/001-accelerator-buffer-layout-abi/001-03-SUMMARY.md`

### Phase 2: Metal Layout-Aware Device Flow

**Goal:** Teach Metal buffer execution to preserve legal device-owned view/layout values and avoid falling back only because an intermediate output is non-contiguous or non-zero-offset.

**Requirements:** METAL-01, METAL-02, METAL-03, METAL-04

**Depends on:** Phase 1.

**Success Criteria:**
1. Metal buffer preflight no longer rejects every non-contiguous/offset output blindly; it classifies whether the layout can be represented, transformed on device, or must fall back.
2. `LINEAR -> RESHAPE -> PERMUTE` style regions have an implementation path that avoids CPU materialization when a safe device representation exists.
3. Native Metal ABI changes, if required, are version/capability checked before execution.
4. Device-to-CPU materialization remains correct for graph output, CPU consumer, and gradient publication reasons.
5. Tests cover native success, safe fallback, unsupported layout rejection, and CPU parity for representative forward/forward-backward graphs.

**Notes:**
- Prefer layout metadata first when native execution can consume it.
- Use explicit device contiguous transforms only when metadata views are not safe or not supported.
- Do not silently claim buffer execution when native code replays on CPU.

### Phase 3: Materialization-Aware Region Planning

**Goal:** Make `PART` and backend selection prefer profitable long device-owned flows and avoid short accelerator islands that force expensive CPU materialization.

**Requirements:** PLAN-01, PLAN-02, PLAN-03, PLAN-04

**Depends on:** Phase 1, Phase 2.

**Success Criteria:**
1. Accelerator cost model includes estimated CPU materialization cost, upload/download cost, tensor-array fallback cost, layout fallback cost, dispatch overhead, and compute work.
2. Accelerator region strategies can prefer longer legal regions when they reduce CPU/GPU boundaries.
3. Planner traces explain why a candidate was accepted, rejected, split, or left on CPU.
4. CPU natural regions and CPU fusion remain intact and are selected when accelerator offload is not profitable.
5. Benchmarks show no CPU hot-path regression for representative CPU-only and mixed CPU/GPU profiles.

**Notes:**
- This phase should not turn every supported op into a GPU op by default.
- The planner should optimize end-to-end region profitability, not only per-op support.
- Keep graph autotune able to explore competing region/offload policies.

### Phase 4: Tuning And Profile Ownership Audit

**Goal:** Review and clean up tuning/calibration ownership so graph autotune controls graph policy and platform calibration controls hardware/dtype thresholds.

**Requirements:** TUNE-01, TUNE-02, TUNE-03, TUNE-04

**Depends on:** Phase 3.

**Success Criteria:**
1. Each tuning knob is classified as graph/workload-specific, platform/dtype-specific, or obsolete.
2. Graph autotune candidate space includes the relevant accelerator region/layout/materialization policies and excludes hardware proxy knobs.
3. Platform calibration includes only hardware/dtype runtime thresholds and backend capability thresholds.
4. Profile IO validates schema/version and reports invalid accelerator buffer/layout fields instead of silently defaulting.
5. Benchmark paths remain read-only, while autotune/calibration paths are the only profile-writing flows.

**Notes:**
- Preserve historical profiles only when they remain schema-compatible or are migrated intentionally.
- Keep report output readable and traceable.
- Avoid duplicate knobs that express the same decision in multiple layers.

### Phase 5: Accelerator Verification And Documentation Closure

**Goal:** Close the milestone with focused tests, benchmark evidence, trace checks, docs, and hygiene rules proving the new accelerator flow works and remains maintainable.

**Requirements:** OBS-01, OBS-02, OBS-03, OBS-04, DOC-01, DOC-02, DOC-03, DOC-04

**Depends on:** Phase 1, Phase 2, Phase 3, Phase 4.

**Success Criteria:**
1. Trace and benchmark reports expose accelerator path, buffer mode, fallback reasons, materialization counts, copy times, and storage residency for the new flow.
2. At least one benchmark workload stresses matmul/linear, view/layout transforms, elementwise fusion, reductions, and backward/gradient publication.
3. Tests prove adjacent accelerator regions pass device buffers without Java array round trips when layout/capability contracts allow it.
4. CPU-vs-Metal correctness tests cover representative forward and forward-backward graphs.
5. Documentation explains the accelerator ABI, device-owned flow, layout/view handling, CPU materialization boundaries, tuning ownership, and fallback diagnostics.
6. Hygiene checks prevent accidental commits of local generated classes and unintentional calibration/benchmark outputs.

**Notes:**
- This is the closure phase for making the architectural shift observable.
- If native CUDA implementation is started before this phase, keep it behind capability checks and separate tests.

## Coverage

| Requirement | Phase | Status |
|-------------|-------|--------|
| ABI-01 | Phase 1 | Complete — verified in 001-VERIFICATION |
| ABI-02 | Phase 1 | Complete — verified in 001-VERIFICATION |
| ABI-03 | Phase 1 | Complete — verified in 001-VERIFICATION |
| ABI-04 | Phase 1 | Complete — verified in 001-VERIFICATION |
| METAL-01 | Phase 2 | Pending |
| METAL-02 | Phase 2 | Pending |
| METAL-03 | Phase 2 | Pending |
| METAL-04 | Phase 2 | Pending |
| PLAN-01 | Phase 3 | Pending |
| PLAN-02 | Phase 3 | Pending |
| PLAN-03 | Phase 3 | Pending |
| PLAN-04 | Phase 3 | Pending |
| TUNE-01 | Phase 4 | Pending |
| TUNE-02 | Phase 4 | Pending |
| TUNE-03 | Phase 4 | Pending |
| TUNE-04 | Phase 4 | Pending |
| OBS-01 | Phase 5 | Pending |
| OBS-02 | Phase 5 | Pending |
| OBS-03 | Phase 5 | Pending |
| OBS-04 | Phase 5 | Pending |
| DOC-01 | Phase 5 | Pending |
| DOC-02 | Phase 5 | Pending |
| DOC-03 | Phase 5 | Pending |
| DOC-04 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 24 total
- Mapped to phases: 24
- Unmapped: 0

---
*Roadmap created: 2026-04-29*
