# ADR 0054: Selective scalar row-fusion emission/cache cutover

## Status

Accepted and implemented — 2026-10-30, after explicit user approval, independent review and
MAIN's final CPU/Javadoc gate. [The architecture root](../../../ARCHITECTURE.md#core-invariants) and
[CPU generation/compatibility owner](../../architecture/contracts/backend-execution.md#cpu0075-selective-scalar-load-and-address-reuse)
are authoritative; this record explains the decision. The [task Result](../../planning/backends/cpu/tasks/0075-row-fusion-scalar-load-address-reuse.md#result)
owns completion evidence and measurement limits.

## Context and decision drivers

CPU0073/CPU0074 retained scalar artifacts literally. The read-only CPU0074 audit reconstructed
32 external load sites over 20 origin/address cells in the actual ordinary F32 F3/H2 row.
This identifies repeated reads and redundant constant address work, not a measured speedup or
proof that all reads can be shared across stores. The user approved only within-row load reuse
and affine address folding/sharing, while retaining mathematics, stores, eligibility and ABI.
The decision must avoid invalidating unrelated caches or maintaining two production emitters.

## Options considered

- Keep literal scalar retention: leaves the diagnosed generated-dataflow overhead unchanged.
- Maintain old and new production emitters: preserves old bytes but duplicates maintenance;
  explicitly rejected by the user.
- Replace only affected scalar row emission and append scoped identity markers: selected.
- Bump the global schema or add W/U hoisting/carry promotion: invalidates unrelated artifacts
  or requires additional lifetime/semantic work; outside the approved scope.

## Decision and rationale

Use the existing ordinary and Scan emitter/binder owners and necessary cold proof owners only.
Prove source-origin/affine-cell equality, store epochs, activity and dominance before sharing;
bound live slots before assignment and recycle typed locals at last use. Unknown cases keep
unshared accesses in the replacement or existing cold fallback, not a parallel legacy emitter.
Keep SIMD bodies, mathematical order, retained stores, resources/ownership and entry/helper ABIs.
No new production type, common seam or runtime cache object is selected.

Append the owning contract's selective scalar markers in both identity streams. Preserve
global schema, projections, canonical keys and prior namespaces; old affected artifacts become
stale while unaffected controls remain literal. This qualifies retention prospectively without
rewriting CPU0073/CPU0074 acceptance or their closed measurement budgets.

## Consequences

The replacement reduces generated accesses without extending capability. Proof and local
lifetime accounting add cold complexity; checked alias/activity fallback is necessary. More
locals or fewer bytecode instructions alone do not prove improved machine code or performance.
Historical `.tsv`/`.gz.b64` snapshots remain immutable evidence, with affected stale tests and
unaffected literal comparisons. Only captured old bytecode may serve as a test-only old arm.
The independent Java oracle remains unchanged, including its disclosed wider staging/carry shape.

[CPU0075](../../planning/backends/cpu/tasks/0075-row-fusion-scalar-load-address-reuse.md#result) records
targeted semantic/bytecode/cache validation, final owning validation and the completed bounded
A/B/C experiment. Retrospective evidence is ordinary 32→20 and Scan 27→22 load sites with
unchanged arithmetic/stores/carry. Large ordinary timing ratios have unmatched JIT states;
the C2-qualified Scan comparison establishes no material speedup and leaves a Java gap.
C1 failures persist, including candidates; no C1 fix, proved machine-code cause, defaults change
or rollout follows. The finite budget is exhausted. [The boundary explanation](../../architecture/runtime-prepare-backend-boundary.md#cpu0074-implemented-synchronous-row-region-exception)
and [CPU guide](../../backend-guide/cpu-backend.md#accepted-scalar-loadaddress-reuse-cutover)
explain the unchanged handoff. ADR0052/0053 remain historical decisions, not rewritten records.
