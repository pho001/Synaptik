# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0054: Selective [scalar](../../glossary.md#term-scalar) row-[fusion](../../glossary.md#term-fusion) emission/[cache](../../glossary.md#term-cache) cutover

## Status

Accepted and implemented — 2026-10-30, after explicit user approval, independent review and
MAIN's final [CPU](../../glossary.md#term-cpu)/[Javadoc](../../glossary.md#term-javadoc) gate. [The architecture root](../../../ARCHITECTURE.md#core-invariants) and
[CPU generation/compatibility owner](../../architecture/contracts/backend-execution.md#cpu0075-selective-scalar-load-and-address-reuse)<span data-glossary-navigation="true"> ([owner glossary](../../glossary.md#term-owner))</span>
are authoritative; this record explains the decision. The [task Result](../../planning/backends/cpu/tasks/0075-row-fusion-scalar-load-address-reuse.md#result)
owns completion evidence and measurement limits.

## Context and decision drivers

CPU0073/CPU0074 retained [scalar](../../glossary.md#term-scalar) [artifacts](../../glossary.md#term-artifact) literally. The read-only CPU0074 audit reconstructed
32 external load sites over 20 origin/address cells in the actual ordinary [F32](../../glossary.md#floating-point-precision-and-rounding) F3/H2 row.
This identifies repeated reads and redundant [constant](../../glossary.md#term-constant) address work, not a measured speedup or
proof that all reads can be shared across stores. The user approved only within-row load reuse
and affine address folding/sharing, while retaining mathematics, stores, eligibility and [ABI](../../glossary.md#term-application-binary-interface-abi).
The decision must avoid invalidating unrelated caches or maintaining two production emitters.

## Options considered

- Keep literal [scalar](../../glossary.md#term-scalar) retention: leaves the diagnosed generated-[dataflow](../../glossary.md#term-dataflow) overhead unchanged.
- Maintain old and new production emitters: preserves old bytes but duplicates maintenance;
  explicitly rejected by the user.
- Replace only affected scalar row emission and append scoped [identity](../../glossary.md#term-identity-meanings) markers: selected.
- Bump the global schema or add W/U hoisting/carry promotion: invalidates unrelated artifacts
  or requires additional lifetime/semantic work; outside the approved scope.<span data-glossary-navigation="true"> ([schema glossary](../../glossary.md#term-schema); [carry glossary](../../glossary.md#term-carry); [artifacts glossary](../../glossary.md#term-artifact); [lifetime glossary](../../glossary.md#term-lifetime); [scope glossary](../../glossary.md#term-scope))</span>

## Decision and rationale

Use the existing ordinary and [Scan](../../glossary.md#term-cumulative-scan) emitter/binder [owners](../../glossary.md#term-owner) and necessary cold proof owners only.
Prove source-origin/affine-cell equality, store [epochs](../../glossary.md#term-epoch-meanings), activity and dominance before sharing;
bound live slots before assignment and recycle typed locals at last use. Unknown cases keep
unshared accesses in the replacement or existing cold fallback, not a parallel legacy emitter.
Keep [SIMD](../../glossary.md#simd) bodies, mathematical order, retained stores, resources/[ownership](../../glossary.md#term-ownership-meanings) and entry/helper ABIs.
No new production type, common seam or [runtime](../../glossary.md#term-runtime) [cache](../../glossary.md#term-cache) object is selected.

Append the owning contract's selective [scalar](../../glossary.md#term-scalar) markers in both [identity](../../glossary.md#term-identity-meanings) streams. Preserve
global [schema](../../glossary.md#term-schema), projections, [canonical](../../glossary.md#term-canonical-form) keys and prior namespaces; old affected [artifacts](../../glossary.md#term-artifact) become
stale while unaffected controls remain literal. This qualifies retention prospectively without
rewriting CPU0073/CPU0074 acceptance or their closed measurement budgets.

## Consequences

The replacement reduces generated accesses without extending capability. Proof and local
[lifetime](../../glossary.md#term-lifetime) accounting add cold complexity; checked [alias](../../glossary.md#aliasing-and-arena)/activity fallback is necessary. More
locals or fewer [bytecode](../../glossary.md#term-bytecode) instructions alone do not prove improved [machine code](../../glossary.md#term-machine-code) or performance.
Historical `.tsv`/`.gz.b64` snapshots remain [immutable](../../glossary.md#term-immutable) evidence, with affected stale tests and
unaffected literal comparisons. Only captured old bytecode may serve as a test-only old arm.
The independent Java [oracle](../../glossary.md#term-test-oracle) remains unchanged, including its disclosed wider staging/[carry](../../glossary.md#term-carry) [shape](../../glossary.md#shape).

[CPU0075](../../planning/backends/cpu/tasks/0075-row-fusion-scalar-load-address-reuse.md#result) records
targeted semantic/bytecode/[cache](../../glossary.md#term-cache) [validation](../../glossary.md#term-validation), final owning validation and the completed bounded
A/B/C experiment. Retrospective evidence is ordinary 32→20 and [Scan](../../glossary.md#term-cumulative-scan) 27→22 load sites with
unchanged arithmetic/stores/carry. Large ordinary timing ratios have unmatched [JIT](../../glossary.md#term-just-in-time-compilation-jit) states;
the [C2](../../glossary.md#term-c2)-qualified Scan comparison establishes no material speedup and leaves a Java gap.
[C1](../../glossary.md#term-c1) failures persist, including candidates; no C1 fix, proved machine-code cause, defaults change
or rollout follows. The [finite](../../glossary.md#term-finite) budget is exhausted. [The boundary explanation](../../architecture/runtime-prepare-backend-boundary.md#cpu0074-implemented-synchronous-row-region-exception)
and [CPU guide](../../backend-guide/cpu-backend.md#accepted-scalar-loadaddress-reuse-cutover)<span data-glossary-navigation="true"> ([CPU glossary](../../glossary.md#term-cpu))</span>
explain the unchanged [handoff](../../glossary.md#term-handoff). ADR0052/0053 remain historical decisions, not rewritten records.
