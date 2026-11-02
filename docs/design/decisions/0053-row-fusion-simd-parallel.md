# ADR 0053: Bounded row-fusion SIMD and synchronous workers

## Status

Accepted — 2026-10-30; implemented and MAIN accepted, CPU0074 Complete. Final CPU/Javadoc,
named integration and bounded performance evidence are recorded in the task Result below.
Functional acceptance does not qualify steady-state profitability or authorize rollout.
[`ARCHITECTURE.md`](../../../ARCHITECTURE.md) and its incorporated contracts govern;
this record explains their decision, not independent authority.

## Context and decision drivers

CPU0073 shares computed row-local values among row outputs and multiple terminal contractions/
represented Scan sums, but its scalar-only ordered override prevents SIMD/worker realizations.
The global aggregate floor already permits rounded trees; changing global Model semantics is
unnecessary. The user selected the bounded extension while retaining dtype, exact masks/saved
values and distinct per-time carry ADDs. Ordinary exact SUM/MEAN remain separate.

Existing vector math and output-column MATMUL provide arithmetic reuse, but segment carrier
loads do not provide arbitrary strided gathers. Existing workers synchronously join, while the
current Scan executable holds one inline permit throughout entry; nested participant acquisition
under that permit can deadlock. The solution must retain the emitted T loop, one RunState,
predeclared scratch and current failure/close behavior without a new executor or phase framework.

## Options considered

- Keep scalar-only fusion: preserves current artifacts but does not deliver the selected capability.
- Vectorize across rows: requires strided access handling and cross-row activity treatment; it is
  not selected for the primary F3/H2 reverse with mixed lengths.
- Use feature/output SIMD and bounded raw transpose staging; split rows only across synchronous
  workers with private partials. Reserve participant demand once and lend it during dispatch.
- Release/reacquire inline permits around every dispatch: can be budget-safe but adds repeated
  admission/lease transitions; not selected for this current need.
- Introduce a new executor/cohort or mandatory reusable Job framework: unnecessary new ownership
  and lifecycle machinery; not selected. Existing scheduler metadata is explicitly tolerated.

## Decision and rationale

Choose feature/output-lane SIMD in selected CPU0073-shaped regions, retaining full feature-K
order and preferred target/dtype species. Masked tails prevent out-of-domain loads/stores.
The primary actual F3/H2 saved-primal reverse can stage W/U transposes in ten represented cells
(40 F32 / 80 F64 bytes before alignment). True invariants stage once per run under activity;
SHARED histories/changing cuts stage at the correct active time or fall back. This avoids a
general gather/shuffle/transpose subsystem and does not promise universally profitable staging.

Worker policy stays in existing preparation unit fields, not canonical row IR. Fixed row ranges
own bounded typed partials/validity flags; the caller merges valid partials in range-index order,
then performs one separate shared-carry ADD per active time. The generated outer loop alone
advances time. A run-owned nested StepDispatch connects the selected region to prebound calls.
One sufficient same-budget reservation covers caller/worker demand without concurrent math,
double charging or nested acquisitions. Join precedes consumers, carry transition and reuse;
release follows quiescence on every outcome. No top-level production type or Runtime API is added.

## Consequences

Scalar CPU0073 and noneligible artifacts remain literal controls; only new exec-v1 realization
payloads/helper ABIs advance. Production scalar/thread-count-one defaults stay unchanged. A
standard RNN forward may have no terminal region and remains scalar; genuinely fused forward
and actual reverse need independent qualification. The permission does not widen dtype/layout,
ordinary exact SUM/MEAN, standalone MATMUL-K, FMA or public numerical modes.

Costs include bounded transpose/partial scratch, one run-bound context/arrays/calls and existing
Job/queue/failure/latch metadata per dispatch. Reserving participant demand throughout entry may
reduce concurrency while caller-only setup/merge runs; no fairness/throughput improvement is
claimed. Small B/masked features may be slower. Finite arithmetic differs across permitted
terminal groupings, with no cross-strategy/thread-count/device bitwise guarantee.

Implementation must supply optimal same-realization Java/ClassFile evidence, finite per-op/dtype
tolerances plus cancellation/special/raw checks, retained scalar bytes, and budget/alias/preflight/
interruption/close/concurrent-run tests. MAIN approved one new finite performance budget in the
task; historical experiments stay closed. Focused implementation evidence and remaining gates
are recorded in the [CPU0074 Result](../../planning/backends/cpu/tasks/0074-row-fusion-simd-parallel.md#result).
The single performance comparison completed with nonstationary timings and concrete JIT
compilation-pressure failures. Separate diagnosis is recommended, not a new authorized task,
frontier or measurement. No stable ranking, universal speedup or steady-state parity is established.
The optimal Java reference retains twelve typed carry cells across T; generated carry promotion
remains deferred without weakening the reference or granting a performance/parity guarantee.
No commit/push or general pointwise repair follows.

## Related documentation

- [CPU mechanism and exact seams](../../architecture/contracts/backend-execution.md#cpu0074-selected-row-region-execution-exception)
- [Scan semantic exception](../../architecture/contracts/recurrent-scan.md#cpu0074-selected-terminal-grouping-exception)
- [Run-owned reservation](../../architecture/contracts/runtime-prepare-engine.md#cpu0074-run-owned-synchronous-reservation)
- [Scan explanation](../../architecture/general-differentiable-scan.md#accepted-but-unimplemented-row-fusion-simd-and-workers)
- [CPU0074 brief](../../planning/backends/cpu/tasks/0074-row-fusion-simd-parallel.md)
- [CPU0073 fusion decision](0052-row-map-multi-reduction-fusion.md)
