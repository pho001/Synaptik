# Synaptik

## What This Is

Synaptik is a Java tensor and compiled computation graph framework for engineers who need to build, optimize, benchmark, and extend tensor execution internals directly in Java. The project centers on explicit graph construction, staged compilation, reverse-mode autodiff, backend-aware runtime execution, calibration, and graph autotune rather than an eager-only numerical scripting model.

This is an existing brownfield codebase. v1.0 shipped the accelerator/runtime architecture hardening needed for Metal and future CUDA execution to behave as clean backend implementations with visible CPU/GPU boundary costs and minimal accidental round trips. v1.1 shipped the first checked-in CUDA native runtime path for dense `FLOAT32` buffer execution, CPU materialization, adjacent CUDA handoff, and trace/report evidence.

## Core Value

Synaptik must produce correct tensor results through a clean compiled graph architecture while letting the optimizer and tuning system choose high-performance CPU or accelerator execution without hidden regressions.

## Current State

v1.1 CUDA Native Runtime shipped on 2026-04-30. The project now has a capability-gated CUDA native shim, shared accelerator buffer ABI integration, dense `FLOAT32` CUDA buffer execution, graph-output and CPU-consumer materialization, adjacent CUDA device-buffer handoff, and CUDA trace/report/docs parity with the Metal-era observability contract.

- 8 phases, 26 plans, and 69 tasks completed across v1.0 and v1.1.
- 33/33 accelerator/runtime requirements satisfied and archived in `.planning/milestones/v1.0-REQUIREMENTS.md` and `.planning/milestones/v1.1-REQUIREMENTS.md`.
- v1.1 phase verification, Nyquist validation, milestone audit, and archival passed.
- Backend-neutral device buffer layout ABI, Metal logical-view device flow, materialization-aware planning, tuning/profile ownership, accelerator observability, and narrow CUDA native runtime execution are now validated project state.
- Real CUDA hardware/native execution remains a residual environment risk because local `nvcc` was unavailable; portable gates and capability-skip behavior passed.

## Next Milestone Goals

v1.2 should broaden GPU-resident execution coverage so realistic neural-network/tensor workloads leave Metal or CUDA less often:

- Non-contiguous/view layout support for Metal and CUDA through native layout ABI v2 or explicit GPU-side layout transforms.
- Broader accelerator lowering coverage for neural-network operations and larger fused GPU kernels.
- Fused GPU region execution for common patterns so longer graph sections stay device-owned.
- Trace and benchmark coverage metrics that make CPU materialization boundaries, fallbacks, region length, and device handoffs measurable.
- Higher-rank native shape/layout ABI support where backend runtimes support it.

## Current Milestone: v1.2 GPU Region Coverage

**Goal:** Expand Metal and CUDA from narrow buffer execution into broader GPU-resident region coverage by supporting non-contiguous/view paths, operation lowering, fused GPU regions, and coverage regression gates that minimize unnecessary GPU-to-CPU exits.

**Target features:**
- Native layout ABI v2 for Metal and CUDA, carrying rank, shape, strides, storage offset, physical byte span, access metadata, and backend capability/version checks.
- GPU-side layout transform and view paths for `reshape`, `permute`, `expand`, `contiguous`, alias outputs, and view-like graph values without CPU materialization between compatible accelerator regions.
- Operation lowering coverage matrix and implementation for common NN/tensor patterns such as matmul/linear, elementwise chains, reductions, softmax-like flows, normalization pieces, and loss-adjacent operations.
- Fused GPU region execution for safe compound patterns such as linear + bias + activation, elementwise chains, and selected reduction-adjacent flows.
- Trace, benchmark, and regression gates that report GPU coverage ratio, region length, fallback counts, CPU materialization boundaries, copy timing, and device handoffs on representative workloads.

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
- ✓ CUDA traces and benchmark reports expose the same accelerator evidence contract as Metal, with explicit fallback reason codes, docs, and source hygiene gates — validated in Phase 8 by `.planning/phases/08-cuda-observability-and-documentation-closure/08-VERIFICATION.md`.

### Active

- [ ] Native layout ABI v2 carries non-contiguous/view layout metadata across Metal and CUDA native boundaries with capability/version checks and explicit fallback.
- [ ] GPU layout transforms and view-like outputs preserve device residency across compatible Metal and CUDA regions.
- [ ] Metal and CUDA lowering cover a broader set of common NN/tensor operation patterns through a documented support matrix and stable unsupported reasons.
- [ ] Fused GPU regions execute safe compound patterns without copying CPU fused ASM internals or regressing CPU fused execution.
- [ ] Trace and benchmark reports quantify GPU region coverage, CPU materialization boundaries, fallbacks, device handoffs, and representative workload improvement.

### Out of Scope

- Rewriting the public `Tensor` API around user-visible device objects — current direction keeps public tensors logical and puts device residency in runtime execution state.
- Replacing the CPU backend with accelerator-first execution — CPU remains the correctness baseline and a performance-critical backend.
- Unlimited CUDA/Metal operation, dtype, rank, and fused-kernel coverage in one milestone — v1.2 should broaden high-value paths first and keep unsupported cases explicit.
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
| Treat Metal and CUDA as backend implementations of shared accelerator contracts | Prevents duplicate architecture and keeps future CUDA work aligned with Metal learnings | ✓ Good — Phase 1 ABI validated; v1.1 CUDA runtime now consumes the shared ABI |
| Prioritize longer device-owned flows over buffer-binding micro-optimizations | Recent benchmarks show region/offload policy dominates zero-copy micro-gains | ✓ Good — Phase 3 static cost planning and report surfaces validated; Phase 4 now derives prepare-time accelerator costs from audited `RuntimeConfig` |
| Represent view/layout metadata in accelerator buffer ABI before broadening GPU fusion | Non-contiguous/view fallback currently breaks GPU flow and causes CPU materialization | ✓ Good — ABI validated in Phase 1; Metal layout-aware flow validated in Phase 2 |
| Keep Phase 2 native Metal ABI unchanged for logical-view flow | Dense physical buffers plus Java-owned logical materialization avoid unsafe native stride/offset ABI churn | ✓ Good — future native layout ABI must be optional-symbol/version/capability checked |
| Keep graph autotune and platform calibration separate | Graph policy is workload-specific; hardware thresholds are platform/dtype-specific | ✓ Good |
| Keep fallback observable in trace and benchmark output | Performance work must distinguish real accelerator execution from CPU replay or tensor-array copies | ✓ Good |
| Keep CUDA buffer execution narrow until the native runtime path is proven | Dense `FLOAT32` coverage gives a stable correctness and observability base before broad operation expansion | ✓ Good — v1.1 validated execution, materialization, adjacent handoff, and report evidence |
| Treat native CUDA checks as capability-gated while portable Java gates remain mandatory | Development environments may not have `nvcc` or CUDA hardware, but the runtime must still fail clearly and testably | ✓ Good — v1.1 records native skip evidence and passes portable fallback/required-mode gates |
| Broaden GPU support by coverage metrics, not by claiming universal operation support | The milestone should measurably reduce CPU exits on representative workloads while preserving explicit fallback for unsupported cases | — Pending |
| Implement fused GPU regions as backend-specific region execution rather than copying CPU ASM fusion | CPU fused execution depends on JVM bytecode/vector paths; Metal and CUDA need backend-native compound DAG execution and capability gates | — Pending |

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
*Last updated: 2026-04-30 after v1.2 milestone start*
