# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0055: Generic [scalar](../../glossary.md#cpu-portable-execution-strategy) row-[invariant](../../glossary.md#term-invariant) external-cell load staging


## Editorial reading context

A row-invariant external cell has the same value across the proved loop region, so CPU generation may stage its load outside that region. Hoisting must not read an inactive or invalid input earlier than permitted. The cold proof authorizes placement; run-owned storage and unchanged arithmetic still govern execution. The old Java experiment motivates the choice but is not generator speed evidence.

Use the [invariant staging and activity boundary](../../architecture/runtime-prepare-backend-boundary.md#cpu0076-prospective-scalar-invariant-staging) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

## Status

Accepted — 2026-10-31, explicit user approval; implementation and bounded generator result
independently approved, 0 P1/P2. Final [CPU](../../glossary.md#term-cpu)/[Javadoc](../../glossary.md#term-javadoc)/named [Engine](../../glossary.md#term-engine) gates and independent documentation
review are complete in the same change.
The [architecture root](../../../ARCHITECTURE.md#core-invariants) and
[CPU0076 owning contract](../../architecture/contracts/backend-execution.md#cpu0076-prospective-invariant-external-cell-load-staging)
are authoritative. This record explains the accepted exception; the task owns executable evidence.

## Context

CPU0075 deliberately reused external cells only within active rows. A separate frozen Java
experiment moved ten genuine [invariant input](../../glossary.md#term-invariant-input)-cell reads outside the row/time loops while
retaining the actual [F32](../../glossary.md#floating-point-precision-and-rounding) reverse [Scan](../../glossary.md#general-differentiable-scan)'s arithmetic, stores and memory-backed per-time [carry](../../glossary.md#term-carry) ADDs.
In `/private/tmp/synaptik-scan-weight-lab.pLXEmb/RESULT.md`, the B256 observation is a 17.92%
[median](../../glossary.md#term-median)-time reduction; B32's 18.90% includes an outer timing-loop [C2](../../glossary.md#term-c2) [compilation](../../glossary.md#compiler-engine-and-execution-stage-names) transition.
This is bounded Java-source evidence, not a generator port, universal speedup or [C1](../../glossary.md#term-c1) repair.
That historical experiment motivated generic load placement, not an [RNN](../../glossary.md#term-recurrent-neural-network-rnn)-specific algorithm.

## Decision drivers

Preserve math/store order and activity, distinguish storage invariance from role/[shape](../../glossary.md#shape) equality,
retain accepted capability and checked fallback, bound caller/helper [liveness](../../glossary.md#term-liveness) before cuts/slots,
and invalidate only [artifacts](../../glossary.md#term-artifact) whose emitted placement changes. Avoid new [runtime](../../glossary.md#term-runtime) [ownership](../../glossary.md#term-ownership-meanings),
types, emitters, framework or entry/helper [ABI](../../glossary.md#term-application-binary-interface-abi).

## Options considered

- Keep CPU0075's within-row-only placement: safe status quo, leaves the observed opportunity.
- Special-case RNN weights/root numbers or copy lab J1 into production: not generic and cannot
  establish storage/temporal safety for other captured bodies; rejected.<span data-glossary-navigation="true"> ([RNN glossary](../../glossary.md#term-recurrent-neural-network-rnn); [weights glossary](../../glossary.md#term-weight))</span>
- Promote carries across time or rewrite SIMD/worker transport together: changes a different
  lifetime/dataflow boundary and exceeds this authorization; rejected.<span data-glossary-navigation="true"> ([SIMD glossary](../../glossary.md#simd); [lifetime glossary](../../glossary.md#term-lifetime); [dataflow glossary](../../glossary.md#term-dataflow))</span>
- Prove external-cell invariance in existing cold owners and stage scalar locals selectively:
  selected, with capability-preserving fallback and affected-only compatibility markers.<span data-glossary-navigation="true"> ([owners glossary](../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([scalar glossary](../../glossary.md#term-scalar))</span>

## Decision and rationale

Use existing ordinary and [Scan](../../glossary.md#general-differentiable-scan) [owners](../../glossary.md#term-owner) to prove source/storage/affine/temporal invariance,
whole-[lifetime](../../glossary.md#term-lifetime) [alias](../../glossary.md#aliasing-and-arena)/store safety and activity/branch dominance. An invocation-[invariant](../../glossary.md#term-invariant) cell
may remain local across T; a SHARED history cell must be reloaded at its consuming active
original time. For example, two different time slices of a saved shared [matrix](../../glossary.md#term-matrix-meanings) must not reuse
the first [slice](../../glossary.md#slice) just because both have the same [shape](../../glossary.md#shape). An inactive worker range cannot stage
body cells merely because another range has active rows.

Keep separate per-time memory-backed [carry](../../glossary.md#term-carry) ADDs, checked access, resources, helper [ABI](../../glossary.md#term-application-binary-interface-abi) and all
existing numerical semantics. Unknown/over-budget proofs retain the current checked path without
shrinking capability or strategy. The 2026-10-31 MAIN-approved, independently reviewed request
(`05529a5700d54790a0103988ee94fd1efce42a207eec73d282f43b0bab1424dc`, Pauli APPROVE, 0 P1/P2)
adds only existing [lowering](../../glossary.md#lowering) [owner](../../glossary.md#term-owner)'s [`invariantLoadStaging`](../../architecture/contracts/backend-execution.md#cpu0076-shared-cold-proof).
Its exact declaration and result contract live in the owner, not duplicated here. Private helpers
remain allowed; no other common seam, new production type, entry/helper ABI or [runtime](../../glossary.md#term-runtime) protocol.

The concrete need is identical pre-cut/preparer/[identity](../../glossary.md#term-identity-meanings)/emission selection without emitter-private
competing proofs or allocation-derived eligibility. All callers use the SAME authenticated
invocation/active-time origin scopes and structural caller/helper base facts. The pure read-only
method selects [immutable](../../glossary.md#term-immutable) ordered complete actual-consumer [view](../../glossary.md#term-view-meanings) domains, deduplicates origin/affine
equivalents and stages no unused cells. It jointly budgets staging, flags and the remaining reuse
pool; helper base zero denotes no helper. Unknown/over-budget proof keeps the checked path;
malformed [metadata](../../glossary.md#term-metadata) keeps existing exceptions. It reads no runtime storage/input and performs no
identity parsing, class generation or route/resource decision. [Binding](../../glossary.md#term-binding) owners still authenticate
temporal/activity/nonempty/access/whole-lifetime alias/write safety; worker locals do not cross ABI.

The owning contract defines selective staging markers in both identity streams while retaining
[canonical](../../glossary.md#term-canonical-form) keys, global envelope/projections and unchanged cases literally. This
qualifies ADR0054's within-row-only boundary, without rewriting its completed result or budgets.

The implemented proof is deliberately narrower than all eligible [kernel](../../glossary.md#kernel) bodies. Only complete
actually consumed fixed-[feature](../../glossary.md#term-feature) views of [rank](../../glossary.md#term-rank) at most two qualify; Scan also requires every
nonformal self-rooted [producer](../../glossary.md#term-producer) to lie in the region's row sites, terminals or finish [epilogue](../../glossary.md#term-epilogue).
Unknown multi-stage lifetimes retain checked reads. Source-authenticated scopes exclude ROW
histories, carries/banks/scratch and already-hoisted [scalar](../../glossary.md#term-scalar) roots. Zero row dependence in a
consumer view is not proof of the source's activity domain.

The two resource-review findings were repaired before source freeze. All demanded history bases,
including forwarded sources, count across the complete caller/prepublication lifetime; duplicate
producing wrappers' [carrier](../../glossary.md#term-storage-carrier)/LONG-base allowances count in both caller and helper. Independently
bounded structural floors, staged typed cells, readiness and the remaining reuse pool share the
unchanged 128-slot limit before cuts/resources/slots. Unproved/over-budget staging declines the
optimization without removing an already eligible scalar/parallel-scalar strategy. The owner
defines the exact common bases; this is not permission to grow resources or the ABI.

## Consequences

Less redundant access is possible, but longer local lifetimes can increase [compilation](../../glossary.md#compiler-engine-and-execution-stage-names)/register
pressure; a lower [bytecode](../../glossary.md#term-bytecode) load count is not a performance proof. Temporal/[alias](../../glossary.md#aliasing-and-arena) and caller/helper
proof complexity increases. Existing helper [bindings](../../glossary.md#term-binding) may preclude once-per-whole-[Scan](../../glossary.md#general-differentiable-scan) staging;
that limitation must remain explicit, not motivate a new frame or [runtime](../../glossary.md#term-runtime) [cache](../../glossary.md#term-cache).

The [CPU0076 Result](../../planning/backends/cpu/tasks/0076-row-invariant-load-staging.md#result)
records actual generator/independent-Java evidence, generic typed/temporal/helper fixtures,
raw/special/activity/empty/resource/fallback/cache coverage and independent review. Retrospective
implementation evidence: the actual [F32](../../glossary.md#floating-point-precision-and-rounding) reverse Scan stages ten [invariant](../../glossary.md#term-invariant) cells before time,
reducing row gets 22→12 without changing arithmetic, five ordered stores or twelve separate
memory-backed per-time [carry](../../glossary.md#term-carry) ADDs. Small ordinary [F64](../../glossary.md#floating-point-precision-and-rounding) staging is qualified; larger combined F64
geometry stays checked. [SIMD](../../glossary.md#simd)/nonregional controls and historical resources remain literal.

The sole released one-[JVM](../../glossary.md#term-java-virtual-machine-jvm) comparison used accepted CPU0075 literal A, actual new generated B
and unchanged original Java R. B256 medians 5301.4/4360.15/3013.225 ns give observed 17.7547%
lower B time than A, with logged primary/Entry/window stable through [sampling](../../glossary.md#term-sampling-machine-learning-and-hyperparameters). B remains 1.447×
R. B32's observed 16.4899% reduction is outer-window-[JIT](../../glossary.md#term-just-in-time-compilation-jit)-confounded, not a stationary causal
estimate. Five fresh symmetric raw-five-output checks per case passed, not every timed output.
[C1](../../glossary.md#term-c1) virtual-register failures persist; no universal stationarity, assembly, parity or C1 repair
is established. No measurement grant remains. Final owning gates passed; no broader speed,
rollout, production-default, [commit](../../glossary.md#term-commit) or push decision is made here.

## Related documentation

- [CPU boundary explanation](../../architecture/runtime-prepare-backend-boundary.md#cpu0076-prospective-scalar-invariant-staging)<span data-glossary-navigation="true"> ([CPU glossary](../../glossary.md#term-cpu))</span>
- [CPU contributor scope](../../backend-guide/cpu-backend.md#cpu0076-accepted-staging-plan-not-yet-implemented)<span data-glossary-navigation="true"> ([scope glossary](../../glossary.md#term-scope))</span>
- [Existing activity semantics](../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories)
