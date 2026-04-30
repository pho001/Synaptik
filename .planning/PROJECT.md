# Synaptik

## What This Is

Synaptik is a Java tensor and compiled computation graph framework for engineers who need to build, optimize, benchmark, and extend tensor execution internals directly in Java. The project centers on explicit graph construction, staged compilation, reverse-mode autodiff, backend-aware runtime execution, calibration, and graph autotune rather than an eager-only numerical scripting model.

This is an existing brownfield codebase. v1.0 shipped the accelerator/runtime architecture hardening needed for Metal and future CUDA execution to behave as clean backend implementations with visible CPU/GPU boundary costs and minimal accidental round trips.

## Core Value

Synaptik must produce correct tensor results through a clean compiled graph architecture while letting the optimizer and tuning system choose high-performance CPU or accelerator execution without hidden regressions.

## Current State

v1.0 Accelerator Runtime Architecture shipped on 2026-04-30. Phase 8 of v1.1 closes CUDA trace/report parity, stable fallback and required-mode reason codes, developer fallback/build docs, and source hygiene gates for CUDA native outputs and local profile artifacts.

- 5 phases, 16 plans, and 41 tasks completed.
- 24/24 v1 requirements satisfied and archived in `.planning/milestones/v1.0-REQUIREMENTS.md`.
- Phase verification, security, Nyquist validation, UAT diagnosis, and milestone audit all passed.
- Backend-neutral device buffer layout ABI, Metal logical-view device flow, materialization-aware planning, tuning/profile ownership, and accelerator observability closure are now validated project state.

## Next Milestone Goals

Fresh requirements should be defined with `$gsd-new-milestone`. Likely candidates from the archived v2 backlog:

- CUDA native shim, capability probe, native materialization, and CUDA benchmark trace parity.
- Broader accelerator lowering coverage for neural-network operations and larger fused GPU kernels.
- Runtime memory scaling for BFLOAT16, INT32, BOOL slot reuse, and checked large-shape arithmetic.

## Current Milestone: v1.1 CUDA Native Runtime

**Goal:** Add a checked-in, capability-gated CUDA native runtime path that consumes the shared accelerator buffer ABI and proves CUDA buffer execution, materialization, adjacent handoff, and report evidence without weakening CPU or Metal safeguards.

**Target features:**
- Checked-in CUDA native shim, build workflow, and runtime capability probe.
- CUDA bridge/buffer binding implementation using the shared layout/access ABI.
- Native CUDA device-to-CPU materialization and adjacent region device-buffer handoff tests.
- CUDA trace and benchmark evidence aligned with the Metal accelerator report contract.

## Requirements

### Validated

- ✓ Public `Tensor` API can build typed tensor graphs with shape, stride, dtype, storage-offset, view, autodiff, and convenience compute flows — existing.
- ✓ Graph lifecycle exists as semantic graph construction, `CompiledGraph`, optimizer stages, `PreparedExecution`, and backend execution — existing.
- ✓ Optimizer stages `AR`, `CSE`, `PART`, `FUSE`, and `MEM` are implemented and documented as ordered graph transformations — existing.
- ✓ CPU backend covers the primary operation families, including elementwise, reductions, layout transforms, matmul/linear, convolution/pooling, softmax, loss functions, attention, gradients, and fused execution — existing.
- ✓ CPU fused execution can use generated ASM/vector/parallel paths governed by runtime profile knobs — existing.
- ✓ Platform calibration, graph autotune, benchmark reporting, profile persistence, and fluent tuning APIs exist — existing.
- ✓ Metal backend has native MPS bridge support, native build workflow, buffer-binding execution, device-to-CPU materialization, and trace diagnostics for accelerator paths — existing.
- ✓ CUDA backend has shared accelerator scaffolding, bridge interfaces, prepared executable policy seams, and tests for buffer-policy behavior, but no checked-in native CUDA shim yet — existing.
- ✓ Documentation exists under `docs/` for architecture, compute flow, optimizer stages, tensor API, calibration/autotune, Metal backend, native bridges, testing, and extension workflows — existing.
- ✓ `.planning/codebase/` contains a current brownfield codebase map for stack, architecture, structure, conventions, testing, integrations, and concerns — existing.
- ✓ Backend-neutral device buffer layout ABI represents shape, strides, storage offset, dtype, logical element count, byte length, access mode, backend id, and native handle identity for Metal now and CUDA later — validated in Phase 1 by `.planning/phases/001-accelerator-buffer-layout-abi/001-VERIFICATION.md`.
- ✓ Metal buffer execution can keep legal view-like layout values device-resident through dense physical logical-view buffers, visible fallback, and explicit CPU materialization boundaries — validated in Phase 2 by `.planning/phases/002-metal-layout-aware-device-flow/002-VERIFICATION.md`.
- ✓ Accelerator region planning and backend selection score static materialization cost, layout fallback cost, upload/download cost, dispatch overhead, expected compute benefit, selected candidates, and rejected finalists while preserving CPU natural/fusion/BLAS safeguards — validated in Phase 3 by `.planning/phases/003-materialization-aware-region-planning/003-VERIFICATION.md`.
- ✓ Tuning ownership separates graph/workload autotune knobs from platform/dtype calibration thresholds, strict profile IO rejects invalid schema and accelerator buffer fields, runtime-derived accelerator costs enter through `RuntimeConfig`, and benchmark commands remain profile-read-only — validated in Phase 4 by `.planning/phases/04-tuning-and-profile-ownership-audit/04-VERIFICATION.md`.
- ✓ Focused tests, traces, benchmark scenarios, documentation, and hygiene checks prove longer device-owned accelerator flows and visible CPU materialization boundaries — validated in Phase 5 by `.planning/phases/05-accelerator-verification-and-documentation-closure/05-VERIFICATION.md`.
- ✓ CUDA native shim source, optional build/probe workflow, runtime capability probe, and graceful unavailable behavior are validated — Phase 6 by `.planning/phases/06-cuda-shim-and-capability-probe/06-VERIFICATION.md`.
- ✓ CUDA bridge and prepared executable seams consume shared accelerator buffer layout/access metadata for dense supported layouts without CUDA-specific common-runtime fields — Phase 6 by `.planning/phases/06-cuda-shim-and-capability-probe/06-VERIFICATION.md`.
- ✓ CUDA dense FLOAT32 native buffer execution, graph-output/CPU-consumer materialization, and adjacent CUDA handoff are validated — Phase 7 by `.planning/phases/07-cuda-buffer-execution-and-materialization/07-VERIFICATION.md`.
- ✓ CUDA traces and benchmark reports expose the same accelerator evidence contract as Metal, with explicit fallback reason codes, docs, and source hygiene gates — validated in Phase 8 by `.planning/phases/08-cuda-observability-and-documentation-closure/08-04-SUMMARY.md`.

### Out of Scope

- Rewriting the public `Tensor` API around user-visible device objects — current direction keeps public tensors logical and puts device residency in runtime execution state.
- Replacing the CPU backend with accelerator-first execution — CPU remains the correctness baseline and a performance-critical backend.
- Implementing a full production CUDA native shim in the same first accelerator-layout phase — the shared ABI must be CUDA-ready, but native CUDA implementation can follow separately.
- Supporting every dtype/rank/layout combination on Metal immediately — capability checks must stay explicit and conservative.
- Hiding fallback behavior — accelerator fallback must remain traceable and benchmark-visible.
- Tracking arbitrary local benchmark/calibration artifacts as project state — only intentional profile fixtures or committed winner profiles should be versioned.

## Context

Synaptik is a Java 25 / Gradle project using the incubating Vector API and ASM bytecode generation. It has a substantial existing source tree under `src/main/java` and broad tests under `src/test/java`. Current codebase maps live in `.planning/codebase/` and should be read before planning major changes.

The current architectural model is already close to the desired shape:

- semantic tensors and operations are separate from compile artifacts;
- compile/prepare/execute are separate lifecycle stages;
- optimizer policy and runtime policy are separated into config/profile records;
- `ExecutionState` tracks per-run runtime tensors, storage residency, device buffer bindings, and CPU materialization traces;
- Metal and CUDA are modeled as accelerator backends under the same backend/lowering/prepare/execution architecture.

The main performance finding from recent benchmark work is that Metal buffer binding is working, but the largest remaining speedups require fewer CPU boundaries. The desired flow is:

```text
CPU input
  -> upload/materialize once when needed
  -> long device-owned accelerator region
  -> device buffer handoff to adjacent accelerator work
  -> CPU materialization only at a true CPU consumer, graph output, or gradient publication boundary
```

Recent `transformer-block medium f32` autotune and benchmark runs showed that accelerator offload and greedy accelerator region policy produce much larger gains than micro-tuning buffer binding alone. Trace diagnostics still show CPU materializations and fallback caused by non-contiguous/view outputs, especially around `LINEAR -> RESHAPE -> PERMUTE`.

The design goal is therefore not "Metal hacks" or "CUDA hacks". The goal is a clean backend-neutral accelerator storage/execution model where Metal and CUDA implement the same concepts with backend-specific native handles.

## Constraints

- **Runtime architecture**: Public `Tensor` should remain a logical graph/tensor API; backend residency belongs in compile/prepare/execute runtime state — keeps API clean and avoids leaking Metal/CUDA details into user code.
- **Correctness**: CPU execution remains the correctness oracle for accelerator work — accelerator plans must validate against CPU where feasible.
- **Performance**: Changes must not regress CPU hot paths, especially BLAS, fused ASM, vector/parallel dispatch, and memory reuse — CPU remains competitive for small and mixed workloads.
- **Traceability**: Accelerator fallback, buffer decisions, materialization reasons, copy times, and backend paths must remain visible in traces and benchmark reports — silent fallback is unacceptable.
- **Backend symmetry**: Shared accelerator abstractions should be backend-neutral where possible — Metal may implement first, but abstractions must not encode Metal-only assumptions unless explicitly named.
- **Native boundary safety**: Java FFM/native ABI changes must be versioned, capability-checked, and covered by Java-side plus native-side tests — ABI mismatch can otherwise masquerade as performance fallback.
- **Local artifact hygiene**: Calibration/benchmark outputs and hardware fingerprints should be committed only when intentionally serving as fixtures or canonical profiles — local runs must not pollute review.
- **Build environment**: JDK 25 and `jdk.incubator.vector` are required — tests and docs must keep these assumptions explicit.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Keep `Tensor` logical and put backend residency in `ExecutionState` / device buffer bindings | Avoids public API pollution and matches existing compile/prepare/execute architecture | ✓ Good |
| Treat Metal and CUDA as backend implementations of shared accelerator contracts | Prevents duplicate architecture and keeps future CUDA work aligned with Metal learnings | ✓ Good — Phase 1 ABI validated; native CUDA remains future work |
| Prioritize longer device-owned flows over buffer-binding micro-optimizations | Recent benchmarks show region/offload policy dominates zero-copy micro-gains | ✓ Good — Phase 3 static cost planning and report surfaces validated; Phase 4 now derives prepare-time accelerator costs from audited `RuntimeConfig` |
| Represent view/layout metadata in accelerator buffer ABI before broadening GPU fusion | Non-contiguous/view fallback currently breaks GPU flow and causes CPU materialization | ✓ Good — ABI validated in Phase 1; Metal layout-aware flow validated in Phase 2 |
| Keep Phase 2 native Metal ABI unchanged for logical-view flow | Dense physical buffers plus Java-owned logical materialization avoid unsafe native stride/offset ABI churn | ✓ Good — future native layout ABI must be optional-symbol/version/capability checked |
| Keep graph autotune and platform calibration separate | Graph policy is workload-specific; hardware thresholds are platform/dtype-specific | ✓ Good |
| Keep fallback observable in trace and benchmark output | Performance work must distinguish real accelerator execution from CPU replay or tensor-array copies | ✓ Good |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition**:
1. Requirements invalidated? -> Move to Out of Scope with reason.
2. Requirements validated? -> Move to Validated with phase reference.
3. New requirements emerged? -> Add to Active.
4. Decisions to log? -> Add to Key Decisions.
5. "What This Is" still accurate? -> Update if drifted.

**After each milestone**:
1. Full review of all sections.
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state.

---
*Last updated: 2026-04-30 after Phase 8 verification evidence*
