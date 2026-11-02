# ADR 0055: Generic scalar row-invariant external-cell load staging

## Status

Accepted — 2026-10-31, explicit user approval; implementation and bounded generator result
independently approved, 0 P1/P2. Final CPU/Javadoc/named Engine gates and independent documentation
review are complete in the same change.
The [architecture root](../../../ARCHITECTURE.md#core-invariants) and
[CPU0076 owning contract](../../architecture/contracts/backend-execution.md#cpu0076-prospective-invariant-external-cell-load-staging)
are authoritative. This record explains the accepted exception; the task owns executable evidence.

## Context

CPU0075 deliberately reused external cells only within active rows. A separate frozen Java
experiment moved ten genuine invariant input-cell reads outside the row/time loops while
retaining the actual F32 reverse Scan's arithmetic, stores and memory-backed per-time carry ADDs.
In `/private/tmp/synaptik-scan-weight-lab.pLXEmb/RESULT.md`, the B256 observation is a 17.92%
median-time reduction; B32's 18.90% includes an outer timing-loop C2 compilation transition.
This is bounded Java-source evidence, not a generator port, universal speedup or C1 repair.
That historical experiment motivated generic load placement, not an RNN-specific algorithm.

## Decision drivers

Preserve math/store order and activity, distinguish storage invariance from role/shape equality,
retain accepted capability and checked fallback, bound caller/helper liveness before cuts/slots,
and invalidate only artifacts whose emitted placement changes. Avoid new runtime ownership,
types, emitters, framework or entry/helper ABI.

## Options considered

- Keep CPU0075's within-row-only placement: safe status quo, leaves the observed opportunity.
- Special-case RNN weights/root numbers or copy lab J1 into production: not generic and cannot
  establish storage/temporal safety for other captured bodies; rejected.
- Promote carries across time or rewrite SIMD/worker transport together: changes a different
  lifetime/dataflow boundary and exceeds this authorization; rejected.
- Prove external-cell invariance in existing cold owners and stage scalar locals selectively:
  selected, with capability-preserving fallback and affected-only compatibility markers.

## Decision and rationale

Use existing ordinary and Scan owners to prove source/storage/affine/temporal invariance,
whole-lifetime alias/store safety and activity/branch dominance. An invocation-invariant cell
may remain local across T; a SHARED history cell must be reloaded at its consuming active
original time. For example, two different time slices of a saved shared matrix must not reuse
the first slice just because both have the same shape. An inactive worker range cannot stage
body cells merely because another range has active rows.

Keep separate per-time memory-backed carry ADDs, checked access, resources, helper ABI and all
existing numerical semantics. Unknown/over-budget proofs retain the current checked path without
shrinking capability or strategy. The 2026-10-31 MAIN-approved, independently reviewed request
(`05529a5700d54790a0103988ee94fd1efce42a207eec73d282f43b0bab1424dc`, Pauli APPROVE, 0 P1/P2)
adds only existing lowering owner's [`invariantLoadStaging`](../../architecture/contracts/backend-execution.md#cpu0076-shared-cold-proof).
Its exact declaration and result contract live in the owner, not duplicated here. Private helpers
remain allowed; no other common seam, new production type, entry/helper ABI or runtime protocol.

The concrete need is identical pre-cut/preparer/identity/emission selection without emitter-private
competing proofs or allocation-derived eligibility. All callers use the SAME authenticated
invocation/active-time origin scopes and structural caller/helper base facts. The pure read-only
method selects immutable ordered complete actual-consumer view domains, deduplicates origin/affine
equivalents and stages no unused cells. It jointly budgets staging, flags and the remaining reuse
pool; helper base zero denotes no helper. Unknown/over-budget proof keeps the checked path;
malformed metadata keeps existing exceptions. It reads no runtime storage/input and performs no
identity parsing, class generation or route/resource decision. Binding owners still authenticate
temporal/activity/nonempty/access/whole-lifetime alias/write safety; worker locals do not cross ABI.

The owning contract defines selective staging markers in both identity streams while retaining
canonical keys, global envelope/projections and unchanged cases literally. This
qualifies ADR0054's within-row-only boundary, without rewriting its completed result or budgets.

The implemented proof is deliberately narrower than all eligible kernel bodies. Only complete
actually consumed fixed-feature views of rank at most two qualify; Scan also requires every
nonformal self-rooted producer to lie in the region's row sites, terminals or finish epilogue.
Unknown multi-stage lifetimes retain checked reads. Source-authenticated scopes exclude ROW
histories, carries/banks/scratch and already-hoisted scalar roots. Zero row dependence in a
consumer view is not proof of the source's activity domain.

The two resource-review findings were repaired before source freeze. All demanded history bases,
including forwarded sources, count across the complete caller/prepublication lifetime; duplicate
producing wrappers' carrier/LONG-base allowances count in both caller and helper. Independently
bounded structural floors, staged typed cells, readiness and the remaining reuse pool share the
unchanged 128-slot limit before cuts/resources/slots. Unproved/over-budget staging declines the
optimization without removing an already eligible scalar/parallel-scalar strategy. The owner
defines the exact common bases; this is not permission to grow resources or the ABI.

## Consequences

Less redundant access is possible, but longer local lifetimes can increase compilation/register
pressure; a lower bytecode load count is not a performance proof. Temporal/alias and caller/helper
proof complexity increases. Existing helper bindings may preclude once-per-whole-Scan staging;
that limitation must remain explicit, not motivate a new frame or runtime cache.

The [CPU0076 Result](../../planning/backends/cpu/tasks/0076-row-invariant-load-staging.md#result)
records actual generator/independent-Java evidence, generic typed/temporal/helper fixtures,
raw/special/activity/empty/resource/fallback/cache coverage and independent review. Retrospective
implementation evidence: the actual F32 reverse Scan stages ten invariant cells before time,
reducing row gets 22→12 without changing arithmetic, five ordered stores or twelve separate
memory-backed per-time carry ADDs. Small ordinary F64 staging is qualified; larger combined F64
geometry stays checked. SIMD/nonregional controls and historical resources remain literal.

The sole released one-JVM comparison used accepted CPU0075 literal A, actual new generated B
and unchanged original Java R. B256 medians 5301.4/4360.15/3013.225 ns give observed 17.7547%
lower B time than A, with logged primary/Entry/window stable through sampling. B remains 1.447×
R. B32's observed 16.4899% reduction is outer-window-JIT-confounded, not a stationary causal
estimate. Five fresh symmetric raw-five-output checks per case passed, not every timed output.
C1 virtual-register failures persist; no universal stationarity, assembly, parity or C1 repair
is established. No measurement grant remains. Final owning gates passed; no broader speed,
rollout, production-default, commit or push decision is made here.

## Related documentation

- [CPU boundary explanation](../../architecture/runtime-prepare-backend-boundary.md#cpu0076-prospective-scalar-invariant-staging)
- [CPU contributor scope](../../backend-guide/cpu-backend.md#cpu0076-accepted-staging-plan-not-yet-implemented)
- [Existing activity semantics](../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories)
