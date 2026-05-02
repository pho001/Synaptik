# Synaptik

## What This Is

Synaptik is a Java tensor and compiled computation graph framework for engineers who need to build, optimize, benchmark, and extend tensor execution internals directly in Java. The project centers on explicit graph construction, staged compilation, reverse-mode autodiff, backend-aware runtime execution, calibration, and graph autotune rather than an eager-only numerical scripting model.

This is an existing brownfield codebase. v1.0 shipped accelerator/runtime architecture hardening, v1.1 shipped the first checked-in CUDA native runtime path for dense `FLOAT32` buffer execution, v1.2 shipped broader Metal/CUDA GPU region coverage through layout ABI v2, GPU layout/view paths, lowering coverage, compound GPU regions, and coverage regression gates, v1.3 shipped coverage-driven GPU region expansion with internal lowered DAGs, dtype/storage residency, broader lowering metadata, region-internal fusion evidence, multi-op GPU regions, and hard coverage evidence, and v1.4 shipped native/lowered GPU execution closure for high-impact operation families with support-or-rejection coverage gates.

## Core Value

Synaptik must produce correct tensor results through a clean compiled graph architecture while letting the optimizer and tuning system choose high-performance CPU or accelerator execution without hidden regressions.

## Current State

v1.4 Native GPU Operation Coverage Closure shipped and was archived on 2026-05-02. Phases 22 through 28 are complete and the v1.4 milestone audit passed. Synaptik now has target coverage truth, semantics contracts, native/lowered reduction coverage, normalization GPU lowering, verified Metal forward SDPA admission, explicit CUDA SDPA capability fallback, loss/indexing support-or-rejection diagnostics, conv/pool and BOOL-output coverage evidence, and hard regression gates that prove supported families do not silently fall back to tensor-array or CPU replay.

- 28 phases and 108 plans completed across v1.0, v1.1, v1.2, v1.3, and v1.4; v1.4 phase artifacts are archived in `.planning/milestones/v1.4-phases/`.
- 94/94 milestone requirements are satisfied and archived in `.planning/milestones/v1.0-REQUIREMENTS.md`, `.planning/milestones/v1.1-REQUIREMENTS.md`, `.planning/milestones/v1.2-REQUIREMENTS.md`, `.planning/milestones/v1.3-REQUIREMENTS.md`, and `.planning/milestones/v1.4-REQUIREMENTS.md`.
- v1.4 phase verification, security, Nyquist validation, milestone audit, and archival passed; final closure normalized Phase 22/23/25/26 evidence so all v1.4 phases are audit-readable.
- v1.3 phase verification, Nyquist validation, milestone audit, and archival passed.
- v1.2 phase verification, security, Nyquist validation, milestone audit, and archival passed.
- Backend-neutral device buffer layout ABI, Metal logical-view device flow, materialization-aware planning, tuning/profile ownership, accelerator observability, narrow CUDA native runtime execution, GPU layout/view residency, GPU lowering coverage, fused GPU compound region metadata, coverage regression gates, GPU coverage triage, lowered-region manifest contracts, dtype/storage residency, operation-family support/rejection evidence, region-internal GPU fusion, multi-op GPU region execution, native/lowered reductions, normalization lowering, Metal forward SDPA, target coverage truth, and hard v1.4 coverage regression closure are now validated project state.
- Real CUDA hardware/native execution remains a residual environment risk because local CUDA native tasks capability-skipped; portable gates and capability-skip behavior passed.
- Current milestone v1.5 is active and focuses on making the Metal backend substantially more production-grade: broader dtype compute/output, BOOL/INT32 semantics, conv/pool, masked SDPA, indexing, loss-adjacent training flows, layout repair, backend routing, and lower-copy execution.

## Current Milestone: v1.5 Production-Grade Metal Backend Expansion

**Goal:** Move Metal from high-value `FLOAT32` graph-region coverage toward a production-grade backend with broader dtype support, NN/index/loss/training coverage, GPU-side layout repair, router decisions, and lower-copy execution.

**Target features:**
- Versioned Metal dtype ABI and capability truth for `BFLOAT16`, `BOOL`, `INT32`, and explicit `FLOAT64` support/rejection.
- BF16 compute/output, BOOL-producing compute, and INT32 index tensor execution.
- Masked/causal SDPA, conv/pool, gather/scatter/index, and loss-adjacent lowering.
- GPU-side layout router for dense, strided, and broadcast materialization without CPU exits where legal.
- Training/backward coverage for v1.5-supported Metal operation families.
- Backend router across MPSGraph, custom Metal kernels, buffer binding, tensor-array fallback, and CPU fallback.
- Closure of the remaining native result-copy/zero-copy evidence gap.

## Recently Shipped Milestone: v1.4 Native GPU Operation Coverage Closure

**Status:** shipped 2026-05-02.

**Goal:** Replace the remaining measured Metal/CUDA fallback families with backend-neutral lowered DAG support and backend-specific native execution, while keeping fallback explicit for unsupported semantics.

**Target features:**
- Native/lowered Metal and CUDA support for forward reductions: `SUM`, `MEAN`, `REDUCE_MIN`, and `REDUCE_MAX`.
- Normalization lowering for `LAYER_NORM` and `RMS_NORM` using GPU-resident reduction-adjacent primitives.
- Forward SDPA execution after semantics verification for scale, mask, rank, dtype, and backward interactions.
- Loss-adjacent, gather/scatter/index, conv/pool, and BOOL compare-output coverage where it closes real CPU exits.
- Coverage gates that prove fewer CPU materialization boundaries and no hidden tensor-array replay on representative transformer, normalization, loss, indexing, and conv/pool workloads.

## Recently Shipped Milestone: v1.3 Coverage-Driven GPU Region Expansion

**Status:** shipped 2026-05-01.

**Goal:** Expand Metal/CUDA GPU execution so larger parts of realistic tensor and neural-network graphs remain device-owned through coverage-driven region lowering, internal GPU fusion, dtype/storage residency, and hard regression gates.

**Target features:**
- Coverage gap triage that uses v1.2 reports as the source of truth and selects transformer block, MLP, and conv/normalization-style hot paths.
- GPU region internal lowered DAG contract where one selected device-owned region can contain multiple original operations, backend primitives, fused subpatterns, and explicit rejection/materialization metadata.
- Dtype and storage residency expansion for BFLOAT16, INT32, and BOOL where missing memory binding currently forces CPU exits.
- Normalization, reduction, softmax-ish, and loss-adjacent lowering/rejection coverage under a shared Metal/CUDA contract.
- Region-internal fused elementwise and epilogue subregions, including elementwise chains and linear/matmul + bias + activation, without reusing CPU `Operation.OpType.FUSED`.
- Multi-op GPU region execution that combines lowered operations, layout/view steps, elementwise work, and selected softmax-ish or normalization lowering without CPU materialization between supported internal steps.
- Coverage regression hardening that fails unexpected CPU materialization, shorter hot-path GPU regions, hidden tensor-array fallback, and unreported backend exits.
- Milestone verification evidence closure that makes Phase 14, Phase 18, and Phase 20 audit evidence explicit and lets the v1.3 audit pass without stale missing-phase findings.

## Prior Shipped Milestone: v1.2 GPU Region Coverage

**Status:** shipped 2026-05-01.

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
- ✓ CUDA backend has a checked-in native shim, bridge capability probe, prepared executable policy seams, dense `FLOAT32` buffer execution, materialization, handoff, and trace/report diagnostics — validated in v1.1.
- ✓ Documentation exists under `docs/` for architecture, compute flow, optimizer stages, tensor API, calibration/autotune, Metal backend, native bridges, testing, and extension workflows — existing.
- ✓ `.planning/codebase/` contains a current brownfield codebase map for stack, architecture, structure, conventions, testing, integrations, and concerns — existing.
- ✓ Backend-neutral device buffer layout ABI represents shape, strides, storage offset, dtype, logical element count, byte length, access mode, backend id, and native handle identity for Metal now and CUDA later — validated in v1.0 archive `.planning/milestones/v1.0-ROADMAP.md`.
- ✓ Metal buffer execution can keep legal view-like layout values device-resident through dense physical logical-view buffers, visible fallback, and explicit CPU materialization boundaries — validated in v1.0 archive `.planning/milestones/v1.0-ROADMAP.md`.
- ✓ Accelerator region planning and backend selection score static materialization cost, layout fallback cost, upload/download cost, dispatch overhead, expected compute benefit, selected candidates, and rejected finalists while preserving CPU natural/fusion/BLAS safeguards — validated in v1.0 archive `.planning/milestones/v1.0-ROADMAP.md`.
- ✓ Tuning ownership separates graph/workload autotune knobs from platform/dtype calibration thresholds, strict profile IO rejects invalid schema and accelerator buffer fields, runtime-derived accelerator costs enter through `RuntimeConfig`, and benchmark commands remain profile-read-only — validated in v1.0 archive `.planning/milestones/v1.0-ROADMAP.md`.
- ✓ Focused tests, traces, benchmark scenarios, documentation, and hygiene checks prove longer device-owned accelerator flows and visible CPU materialization boundaries — validated in v1.0 archive `.planning/milestones/v1.0-ROADMAP.md`.
- ✓ CUDA native shim source, optional build/probe workflow, runtime capability probe, and graceful unavailable behavior are validated — Phase 6 by `.planning/milestones/v1.1-phases/06-cuda-shim-and-capability-probe/06-VERIFICATION.md`.
- ✓ CUDA bridge and prepared executable seams consume shared accelerator buffer layout/access metadata for dense supported layouts without CUDA-specific common-runtime fields — Phase 6 by `.planning/milestones/v1.1-phases/06-cuda-shim-and-capability-probe/06-VERIFICATION.md`.
- ✓ CUDA dense FLOAT32 native buffer execution, graph-output/CPU-consumer materialization, and adjacent CUDA handoff are validated — Phase 7 by `.planning/milestones/v1.1-phases/07-cuda-buffer-execution-and-materialization/07-VERIFICATION.md`.
- ✓ CUDA traces and benchmark reports expose the same accelerator evidence contract as Metal, with explicit fallback reason codes, docs, and source hygiene gates — validated in Phase 8 by `.planning/milestones/v1.1-phases/08-cuda-observability-and-documentation-closure/08-VERIFICATION.md`.
- ✓ Native layout ABI v2 carries non-contiguous/view layout metadata across Metal and CUDA native boundaries with capability/version checks and explicit fallback — validated in Phase 9 by `.planning/milestones/v1.2-phases/09-native-layout-abi-v2/09-VERIFICATION.md`.
- ✓ GPU layout transforms and view-like outputs preserve device residency across compatible Metal and CUDA regions, with CPU parity at graph output/CPU consumer/gradient publication boundaries — validated in Phase 10 by `.planning/milestones/v1.2-phases/10-gpu-layout-transform-and-view-path/10-VERIFICATION.md`.
- ✓ Metal and CUDA lowering cover common NN/tensor operation patterns through a checked-in support matrix with stable unsupported reasons — validated in Phase 11 by `.planning/milestones/v1.2-phases/11-gpu-lowering-coverage-matrix/11-VERIFICATION.md`.
- ✓ Fused GPU regions execute safe compound patterns without copying CPU fused ASM internals or regressing CPU fused execution — validated in Phase 12 by `.planning/milestones/v1.2-phases/12-fused-gpu-region-execution/12-VERIFICATION.md`.
- ✓ Trace and benchmark reports quantify GPU region coverage, CPU materialization boundaries, fallbacks, device handoffs, and representative workload behavior — validated in Phase 13 by `.planning/milestones/v1.2-phases/13-coverage-benchmark-and-regression-gate/13-VERIFICATION.md`.
- ✓ v1.3 coverage gap triage selects hot-path GPU exits from measured evidence, GPU regions are described as lowered DAG manifests, and `BFLOAT16`/`INT32`/`BOOL` dtype residency is represented in runtime binding, backend decisions, traces, and reports — validated through Phase 16 by `.planning/milestones/v1.3-phases/16-dtype-and-storage-residency-expansion/16-VERIFICATION.md`.
- ✓ Normalization, reduction, softmax-ish, conv, and loss-adjacent GPU lowering gaps are covered by the shared Metal/CUDA matrix with stable rejection detail, CPU parity, and trace/report evidence — validated in Phase 17 by `.planning/milestones/v1.3-phases/17-normalization-reduction-and-loss-adjacent-lowering/17-VERIFICATION.md`.
- ✓ GPU regions expose region-internal fused elementwise and linear/matmul epilogue subpatterns without reusing CPU `Operation.OpType.FUSED`, CPU fused ASM, or CPU vector dispatch internals — validated in Phase 18 by `.planning/milestones/v1.3-phases/18-fused-elementwise-and-epilogue-subregions/18-VERIFICATION.md`.
- ✓ Selected GPU regions can execute multiple lowered operations through backend-neutral planning contracts and backend-specific Metal/CUDA prepared executables while preserving true CPU materialization boundaries — validated in Phase 19 by `.planning/milestones/v1.3-phases/19-multi-op-gpu-region-execution/19-VERIFICATION.md`.
- ✓ Coverage reports and regression gates expose lowered operation counts, fused subpattern counts, selected region length, rejected candidates, CPU exits, materialization reasons, device handoffs, native capability evidence, and hard target gates — validated in Phase 20 by `.planning/milestones/v1.3-phases/20-coverage-regression-hardening/20-VERIFICATION.md`.
- ✓ v1.3 milestone verification evidence is closed: Phase 14 and Phase 18 verification reports exist, Phase 20 validation metadata is Nyquist-compliant, and the v1.3 milestone audit passes — validated in Phase 21 by `.planning/milestones/v1.3-phases/21-milestone-verification-evidence-closure/21-VERIFICATION.md` and `.planning/milestones/v1.3-MILESTONE-AUDIT.md`.
- ✓ Target coverage truth, semantics contracts, and benchmark coverage reports distinguish real native/buffer execution from rejection evidence, tensor-array bridge execution, CPU fallback, and candidate shortening for v1.4 operation families — validated in Phase 22 by `.planning/milestones/v1.4-phases/22-coverage-truth-and-semantics-lock/22-VERIFICATION.md`.
- ✓ Forward reductions `SUM`, `MEAN`, `REDUCE_MIN`, and `REDUCE_MAX` are represented as accelerator DAG primitives and execute through supported Metal/CUDA dense `FLOAT32` GPU paths with CPU parity and trace evidence — validated in Phase 23 by `.planning/milestones/v1.4-phases/23-forward-reductions-native-execution/23-VERIFICATION.md`.
- ✓ `LAYER_NORM` and `RMS_NORM` lower to GPU-resident reduction-adjacent DAGs for supported cases while preserving epsilon, gamma/beta broadcast, dtype/layout legality, and CPU parity — validated in Phase 24 by `.planning/milestones/v1.4-phases/24-normalization-gpu-lowering/24-VERIFICATION.md`.
- ✓ Forward SDPA semantics are locked and Metal admits verified unmasked forward SDPA while CUDA and unsupported SDPA variants retain stable capability-specific rejection evidence — validated in Phase 25 by `.planning/milestones/v1.4-phases/25-forward-sdpa-semantic-enablement/25-VERIFICATION.md`.
- ✓ Loss-adjacent, gather/scatter/take-along-axis, index-target, conv/pool, and BOOL-output target families expose GPU support-or-rejection coverage, residency diagnostics, and parity evidence without hidden CPU exits — validated in Phases 26 and 27 by `.planning/milestones/v1.4-phases/26-loss-adjacent-and-indexing-gpu-coverage/26-VERIFICATION.md` and `.planning/milestones/v1.4-phases/27-conv-pool-and-bool-compare-outputs/27-VERIFICATION.md`.
- ✓ v1.4 coverage regression gates harden representative transformer, normalization, loss/indexing, and conv/pool workloads with native evidence, target deltas, fallback counts, CPU materialization counts, region length, lowered primitive count, backend path, and artifact hygiene — validated in Phase 28 by `.planning/milestones/v1.4-phases/28-coverage-regression-closure/28-VERIFICATION.md`.

### Active

- [ ] v1.5 must widen Metal dtype support without conflating dtype residency with native dtype compute.
- [ ] v1.5 must add support-or-rejection execution coverage for BF16, BOOL-producing compute, INT32 indexing, masked SDPA, conv/pool, scatter/index gradients, loss-adjacent flows, and training/backward paths.
- [ ] v1.5 must keep all unsupported dtype/layout/operation combinations explicit in traces, coverage reports, and benchmark gates.
- [ ] v1.5 must add router/cost evidence for MPSGraph vs custom Metal kernels vs CPU fallback and address the remaining native result-copy evidence gap.

### Out of Scope

- Rewriting the public `Tensor` API around user-visible device objects — current direction keeps public tensors logical and puts device residency in runtime execution state.
- Replacing the CPU backend with accelerator-first execution — CPU remains the correctness baseline and a performance-critical backend.
- CUDA parity for the new v1.5 Metal coverage — v1.5 is Metal-first; shared contracts should remain backend-neutral but CUDA native implementation follows later.
- Universal Metal operation, dtype, rank, and fused-kernel coverage — v1.5 targets high-impact production gaps first and keeps unsupported cases explicit.
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

The main performance finding from accelerator benchmark work remains that buffer binding is useful, but the largest speedups require fewer CPU boundaries. v1.2 added coverage reporting and regression gates so this can now be measured directly instead of inferred from raw timing. The desired flow remains:

```text
CPU input
  -> upload/materialize once when needed
  -> long device-owned accelerator region
  -> device buffer handoff to adjacent accelerator work
  -> CPU materialization only at a true CPU consumer, graph output, or gradient publication boundary
```

Recent v1.2 work moved the codebase closer to that flow by making non-contiguous/view layout metadata, layout transforms, supported lowering, fused GPU regions, and hidden-exit coverage measurable. The design goal is still not "Metal hacks" or "CUDA hacks". The goal is a clean backend-neutral accelerator storage/execution model where Metal and CUDA implement the same concepts with backend-specific native handles.

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
| Establish native layout ABI v2 before consuming non-contiguous/view GPU paths | The bridge must carry shape/stride/storage-offset/physical-span metadata safely before broader layout execution can rely on it | ✓ Good — Phase 9 validated ABI v2 metadata, optional symbols, capability checks, and fallback diagnostics |
| Keep metadata-only view propagation separate from dense GPU materialization | Borrowed-handle views and dense transforms have different safety and capability contracts; conflating them would hide fallback and residency errors | ✓ Good — Phase 10 validated metadata-only view residency, dense materialization gates, and trace-visible fallback |
| Broaden GPU support by coverage metrics, not by claiming universal operation support | The milestone should measurably reduce CPU exits on representative workloads while preserving explicit fallback for unsupported cases | ✓ Good — Phase 13 added coverage summaries, representative workload comparison, and regression gates |
| Implement fused GPU regions as backend-specific region execution rather than copying CPU ASM fusion | CPU fused execution depends on JVM bytecode/vector paths; Metal and CUDA need backend-native compound DAG execution and capability gates | ✓ Good — Phase 12 added compound DAG summaries and kept CPU `FUSED` explicitly CPU-only |
| Treat GPU fusion as region-internal lowering/fusion, not CPU fused ASM reuse | Partitioning selects a device-owned region; region lowering expands it into backend primitives; fusion optimizes supported subgraphs inside that lowered region | ✓ Good — v1.3 validated fused subpattern evidence, multi-op GPU regions, and hard coverage gates without reusing CPU fused ASM internals |

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
*Last updated: 2026-05-02 after v1.5 milestone planning*
