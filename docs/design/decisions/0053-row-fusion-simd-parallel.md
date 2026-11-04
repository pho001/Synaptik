# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0053: Bounded row-[fusion](../../glossary.md#term-fusion) [SIMD](../../glossary.md#simd) and [synchronous](../../glossary.md#term-synchronous) workers


## Editorial reading context

CPU preparation chooses qualified vector/worker realization; the generated loop still owns time traversal. Workers receive disjoint ranges and private partials, then join before carry updates or resource reuse. A reservation accounts for participants without recursively acquiring a budget under an existing permit. SIMD means processing multiple lanes at once, not a guarantee of faster steady-state execution.

Use the [synchronous workers and run-owned reservation](../../architecture/runtime-prepare-backend-boundary.md#cpu0074-implemented-synchronous-row-region-exception) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

## Status

Accepted — 2026-10-30; implemented and MAIN accepted, CPU0074 Complete. Final [CPU](../../glossary.md#term-cpu)/[Javadoc](../../glossary.md#term-javadoc),
named integration and bounded performance evidence are recorded in the task Result below.
Functional acceptance does not qualify steady-state profitability or authorize rollout.
[`ARCHITECTURE.md`](../../../ARCHITECTURE.md) and its incorporated contracts govern;
this record explains their decision, not independent authority.

## Context and decision drivers

CPU0073 shares computed row-local values among row outputs and multiple terminal contractions/
represented [Scan](../../glossary.md#term-cumulative-scan) sums, but its [scalar](../../glossary.md#term-scalar)-only ordered override prevents [SIMD](../../glossary.md#simd)/worker realizations.
The global [aggregate floor](../../glossary.md#term-aggregate-floor) already permits rounded trees; changing global [Model](../../glossary.md#term-model) semantics is
unnecessary. The user selected the bounded extension while retaining [dtype](../../glossary.md#data-type--datatype), exact [masks](../../glossary.md#term-boolean-mask)/saved
values and distinct per-time [carry](../../glossary.md#term-carry) ADDs. Ordinary exact SUM/[MEAN](../../glossary.md#term-mean) remain separate.

Existing [vector](../../glossary.md#term-vector) math and output-column [MATMUL](../../glossary.md#matrix-multiplication--matmul) provide arithmetic reuse, but segment [carrier](../../glossary.md#term-storage-carrier)
loads do not provide arbitrary strided gathers. Existing workers synchronously join, while the
current Scan executable holds one inline permit throughout entry; nested participant acquisition
under that permit can [deadlock](../../glossary.md#term-deadlock). The solution must retain the emitted T loop, one [RunState](../../glossary.md#run-state--runstate),
predeclared scratch and current failure/close behavior without a new executor or phase framework.

## Options considered

- Keep [scalar](../../glossary.md#term-scalar)-only [fusion](../../glossary.md#term-fusion): preserves current [artifacts](../../glossary.md#term-artifact) but does not deliver the selected capability.
- Vectorize across rows: requires strided access handling and cross-row activity treatment; it is
  not selected for the primary F3/H2 reverse with mixed lengths.
- Use feature/output SIMD and bounded raw transpose staging; split rows only across synchronous
  workers with private partials. Reserve participant demand once and lend it during dispatch.<span data-glossary-navigation="true"> ([feature glossary](../../glossary.md#term-software-feature); [SIMD glossary](../../glossary.md#simd); [transpose glossary](../../glossary.md#term-transpose); [synchronous glossary](../../glossary.md#term-synchronous); [dispatch glossary](../../glossary.md#term-dispatch))</span>
- Release/reacquire inline permits around every dispatch: can be budget-safe but adds repeated
  admission/lease transitions; not selected for this current need.<span data-glossary-navigation="true"> ([Release glossary](../../glossary.md#term-release); [admission glossary](../../glossary.md#term-admission); [lease glossary](../../glossary.md#term-lease))</span>
- Introduce a new executor/cohort or mandatory reusable Job framework: unnecessary new ownership
  and lifecycle machinery; not selected. Existing scheduler metadata is explicitly tolerated.<span data-glossary-navigation="true"> ([ownership glossary](../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../glossary.md#lifecycle); [metadata glossary](../../glossary.md#term-metadata))</span>

## Decision and rationale

Choose [feature](../../glossary.md#term-software-feature)/output-[lane](../../glossary.md#term-lane) [SIMD](../../glossary.md#simd) in selected CPU0073-shaped regions, retaining full feature-K
order and preferred [target](../../glossary.md#term-target)/[dtype](../../glossary.md#data-type--datatype) species. Masked tails prevent out-of-domain loads/stores.
The primary actual F3/H2 saved-[primal](../../glossary.md#term-primal) reverse can stage W/U transposes in ten represented cells
(40 [F32](../../glossary.md#floating-point-precision-and-rounding) / 80 [F64](../../glossary.md#floating-point-precision-and-rounding) bytes before [alignment](../../glossary.md#term-alignment)). True invariants stage once per [run](../../glossary.md#term-run) under activity;
SHARED histories/changing cuts stage at the correct active time or fall back. This avoids a
general [gather](../../glossary.md#gather)/shuffle/[transpose](../../glossary.md#term-transpose) subsystem and does not promise universally profitable staging.

Worker policy stays in existing preparation unit fields, not [canonical](../../glossary.md#term-canonical-form) row [IR](../../glossary.md#term-intermediate-representation-ir). Fixed row ranges
own bounded typed partials/validity flags; the caller merges valid partials in range-index order,
then performs one separate shared-[carry](../../glossary.md#term-carry) ADD per active time. The generated outer loop alone
advances time. A run-owned nested StepDispatch connects the selected region to prebound calls.
One sufficient same-budget reservation covers caller/worker demand without concurrent math,
double charging or nested acquisitions. Join precedes consumers, carry transition and reuse;
[release](../../glossary.md#term-release) follows [quiescence](../../glossary.md#term-quiescence) on every outcome. No top-level production type or [Runtime](../../glossary.md#term-runtime) [API](../../glossary.md#term-application-programming-interface-api) is added.

## Consequences

[Scalar](../../glossary.md#term-scalar) CPU0073 and noneligible [artifacts](../../glossary.md#term-artifact) remain literal controls; only new exec-v1 realization
payloads/helper ABIs advance. Production scalar/thread-count-one defaults stay unchanged. A
standard [RNN](../../glossary.md#term-recurrent-neural-network-rnn) forward may have no terminal region and remains scalar; genuinely [fused](../../glossary.md#specialization-and-fusion) forward
and actual reverse need independent [qualification](../../glossary.md#term-qualification). The permission does not widen [dtype](../../glossary.md#data-type--datatype)/[layout](../../glossary.md#layout),
ordinary exact SUM/[MEAN](../../glossary.md#term-mean), standalone [MATMUL](../../glossary.md#matrix-multiplication--matmul)-K, [FMA](../../glossary.md#term-fused-multiply-add-fma) or public numerical modes.

Costs include bounded [transpose](../../glossary.md#term-transpose)/partial scratch, one [run](../../glossary.md#term-run)-bound context/arrays/calls and existing
Job/queue/failure/[latch](../../glossary.md#control-flow-graph) [metadata](../../glossary.md#term-metadata) per [dispatch](../../glossary.md#term-dispatch). Reserving participant demand throughout entry may
reduce [concurrency](../../glossary.md#term-concurrency) while caller-only setup/merge runs; no fairness/[throughput](../../glossary.md#term-throughput) improvement is
claimed. Small B/masked [features](../../glossary.md#term-software-feature) may be slower. [Finite](../../glossary.md#term-finite) arithmetic differs across permitted
terminal groupings, with no cross-strategy/thread-count/[device](../../glossary.md#term-device) bitwise guarantee.

Implementation must supply optimal same-realization Java/ClassFile evidence, finite per-op/dtype
tolerances plus [cancellation](../../glossary.md#term-cancellation)/special/raw checks, retained scalar bytes, and budget/[alias](../../glossary.md#aliasing-and-arena)/[preflight](../../glossary.md#term-preflight)/
interruption/close/concurrent-run tests. MAIN approved one new finite performance budget in the
task; historical experiments stay closed. Focused implementation evidence and remaining gates
are recorded in the [CPU0074 Result](../../planning/backends/cpu/tasks/0074-row-fusion-simd-parallel.md#result).
The single performance comparison completed with [nonstationary](../../glossary.md#term-nonstationary) timings and concrete [JIT](../../glossary.md#term-just-in-time-compilation-jit)
[compilation](../../glossary.md#compiler-engine-and-execution-stage-names)-pressure failures. Separate diagnosis is recommended, not a new authorized task,
frontier or measurement. No stable [ranking](../../glossary.md#term-ranking), universal speedup or steady-state parity is established.
The optimal Java reference retains twelve typed [carry](../../glossary.md#term-carry) cells across T; generated carry promotion
remains deferred without weakening the reference or granting a performance/parity guarantee.
No [commit](../../glossary.md#term-commit)/push or general [pointwise](../../glossary.md#term-pointwise) repair follows.

## Related documentation

- [CPU mechanism and exact seams](../../architecture/contracts/backend-execution.md#cpu0074-selected-row-region-execution-exception)<span data-glossary-navigation="true"> ([CPU glossary](../../glossary.md#term-cpu))</span>
- [Scan semantic exception](../../architecture/contracts/recurrent-scan.md#cpu0074-selected-terminal-grouping-exception)<span data-glossary-navigation="true"> ([Scan glossary](../../glossary.md#term-cumulative-scan))</span>
- [Run-owned reservation](../../architecture/contracts/runtime-prepare-engine.md#cpu0074-run-owned-synchronous-reservation)<span data-glossary-navigation="true"> ([Run glossary](../../glossary.md#term-run))</span>
- [Scan explanation](../../architecture/general-differentiable-scan.md#accepted-but-unimplemented-row-fusion-simd-and-workers)
- [CPU0074 brief](../../planning/backends/cpu/tasks/0074-row-fusion-simd-parallel.md)
- [CPU0073 fusion decision](0052-row-map-multi-reduction-fusion.md)<span data-glossary-navigation="true"> ([fusion glossary](../../glossary.md#term-fusion))</span>
