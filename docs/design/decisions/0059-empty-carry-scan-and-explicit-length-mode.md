# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0059: Empty [carry](../../glossary.md#term-carry) and explicit length mode for [general Scan](../../glossary.md#general-differentiable-scan)

## Status

Accepted and implemented for source only by Model0046. Independent final core and wire/evidence/
docs reviews APPROVE0; exact source/caller/[guard](../../glossary.md#term-guard) cutover and actual source/compatibility checks
passed. This grants no typed execution/[AD](../../glossary.md#autograd), wider cold/wire [admission](../../glossary.md#term-admission) or complete [release](../../glossary.md#term-release) readiness.

## Context and problem

Before this cutover, [Model](../../glossary.md#term-model)'s ScanBody rejected empty [carry](../../glossary.md#term-carry); ScanAttrs required carryCount>=1 and
used rowCarryCount=0 to mean all-valid. Compiler0017 consequently proposed rejecting an otherwise
ordinary first-order [derivative](../../glossary.md#term-derivative) when every original carry is discrete. Floating emissions and
sequence inputs can have a real derivative but leave no ROW [cotangent](../../glossary.md#cotangent) carry. There can also be
only SHARED [parameter](../../glossary.md#term-parameter) accumulators in the reverse, requiring length mode with zero ROW carries.

That is a representation limit, not mathematical non-differentiability. A dummy floating carry,
fake discrete cotangent, [activation](../../glossary.md#term-activation) replay or special [unrolled](../../glossary.md#backpropagation-through-time-and-recurrent-state) reverse would hide the problem
and duplicate state/work. The delegated production [target](../../glossary.md#term-target) requires ordinary [PURE](../../glossary.md#term-pure) composition
and per-value types, including real [BOOL](../../glossary.md#data-type--datatype)/integral [primal](../../glossary.md#term-primal) dependencies; the proposed rejection
must not silently become a permanent smaller mathematical whitelist.

## Decision

Use the existing [Scan](../../glossary.md#term-cumulative-scan) representation for a true empty [carry](../../glossary.md#term-carry) and make length mode explicit.
No new [operation](../../glossary.md#operation) kind, top-level type, package, [callback](../../glossary.md#term-callback), arbitrary while or executor.

- ScanBody permits empty carry/nextCarry together; step sequences remain nonempty, and at least
  one nextCarry or emission root is required. Each existing nonempty carry still has exact stable
  shape/dtype. A carry-free scan is a fixed-T traversal of ordinary emissions, not a new map API.<span data-glossary-navigation="true"> ([shape glossary](../../glossary.md#shape); [dtype glossary](../../glossary.md#data-type--datatype); [API glossary](../../glossary.md#term-application-programming-interface-api))</span>
- ScanAttrs retains its seven-field all-valid convenience constructor. Replace the nine-field
  canonical form with <span data-glossary-navigation="true"> ([canonical form glossary](../../glossary.md#term-canonical-form))</span>`(timeCount, carryCount, sequenceCount, invariantCount, emissionCount,
  residualCount, reverse, lengthAware, rowCarryCount, sequenceRowAxes)`. Migrate every source/test
  caller; do not retain a sentinel-based nine-field alias or compatibility shim.<span data-glossary-navigation="true"> ([alias glossary](../../glossary.md#aliasing-and-arena))</span>
- Counts remain checked/nonnegative, sequenceCount>=1, carryCount+emissionCount>=1. All-valid
  requires rowCarryCount=0 and empty axes. Length mode permits rowCarryCount in [0,carryCount],
  with exactly one existing optional axis per sequence. inputCount uses lengthAware, never the
  row count; lengths remains the final actual and never a body formal.<span data-glossary-navigation="true"> ([axes glossary](../../glossary.md#term-axis))</span>
- Length-aware binding obtains the explicit static/named B from INT64[B] lengths, then validates
  every declared ROW coordinate against it. It never guesses a row from equal extents or reads
  lengths. Existing positive-ROW sources have the same meaning and descriptors. All-inactive
  traversal still skips every body operation, keeps carried values raw and emits positive zeros.<span data-glossary-navigation="true"> ([binding glossary](../../glossary.md#term-binding); [INT64 glossary](../../glossary.md#data-type--datatype); [coordinate glossary](../../glossary.md#axis-rank-scalar-vector-and-matrix); [descriptors glossary](../../glossary.md#term-descriptor))</span>
- Future typed AD selects floating roots/targets only. Compact positional maps preserve typed
  discrete primal histories and repeated/tied contributions. A reverse with no ROW cotangents
  uses zero ROW carries; if no SHARED floating accumulator is needed, it uses no carry at all.
  No discrete gradient or synthetic state is introduced. Existing ordinary cast/disconnected/
  unsupported-operation/first-order rules remain independently enforced.<span data-glossary-navigation="true"> ([AD glossary](../../glossary.md#autograd); [targets glossary](../../glossary.md#term-target); [primal glossary](../../glossary.md#term-primal); [accumulator glossary](../../glossary.md#accumulator-type); [gradient glossary](../../glossary.md#term-gradient))</span>

## Boundary and migration discipline

This decision changes source representation, not permission for every consumer. A first source
cutover must preserve independently [fail-closed](../../glossary.md#term-fail-closed) [Compiler](../../glossary.md#term-compiler) [inference](../../glossary.md#term-inference)/[AD](../../glossary.md#autograd), direct [Prepare](../../glossary.md#prepare), direct [CPU](../../glossary.md#term-cpu)
and cold/raw-[codec](../../glossary.md#term-codec) guards for newly representable topologies until each is actually qualified.
Update every old sentinel-derived mode check to explicit lengthAware; do not accidentally execute
a zero-ROW length [scan](../../glossary.md#term-cumulative-scan) as all-valid. Existing [homogeneous](../../glossary.md#term-homogeneous) nonempty sources stay accepted.

The existing v1-v5 wire formats and constructor inventories stay bounded. V4 retains its old
seven fields and nonempty-[carry](../../glossary.md#term-carry) domain; V5 retains opcode33/old tuple/roles/positive row carry.
Raw encoding, decoding and source reconstruction must independently reject newly representable
carry-free or zero-ROW length topologies before mutation/large allocation, even if [Model](../../glossary.md#term-model) accepts
them. Opcode23/33 determines the existing mode; no [boolean](../../glossary.md#data-type--datatype) wire, V6, invented [opcode](../../glossary.md#term-opcode), [alias](../../glossary.md#aliasing-and-arena) or
silent format expansion. Existing accepted file bytes remain unchanged.

The normal Scan.apply overloads and result [carrier](../../glossary.md#term-storage-carrier) are reused. finalCarry can be empty; emission
ports remain [canonical](../../glossary.md#term-canonical-form). Fixed recurrent scans, ordinary .rnn, shared [runtime](../../glossary.md#term-runtime)/slot [ownership](../../glossary.md#term-ownership-meanings) and
existing numerical algorithms remain unchanged. Generated/[cache](../../glossary.md#term-cache) [identity](../../glossary.md#term-identity-meanings) changes belong only to
later actually changed consumer facts/bytes; the source flag is not an arbitrary global bump.

## Established precedent

[JAX lax.scan](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html)<span data-glossary-navigation="true"> ([scan glossary](../../glossary.md#term-cumulative-scan))</span> permits `None` as an
empty [carry](../../glossary.md#term-carry) pytree while requiring fixed carry structure/[shape](../../glossary.md#shape)/[dtype](../../glossary.md#data-type--datatype) and compiling a static-count
body as a loop. This supports the boring representation choice; Synaptik retains its own explicit
length/activity, source/cold and ordinary-rule contracts rather than copying a framework executor.

## Acceptance and consequences

Validate both traversal modes, T/B=0/1, emission-only all-valid/length source and zero-ROW SHARED
accumulators, exact mode/count/port/[provenance](../../glossary.md#provenance)/[alias](../../glossary.md#aliasing-and-arena) facts and pre-ID rejection of malformed
headers/roles. Current fixed/nonempty behaviors, old file bytes and downstream negative guards
must remain real tests. Actual typed reverse and later [CPU](../../glossary.md#term-cpu) helper composition need independent
nonuniform-[oracle](../../glossary.md#term-test-oracle), inactive-[padding](../../glossary.md#padding), [dtype](../../glossary.md#data-type--datatype)/history, failure/resource and final workflow evidence.

Benefits: no fake state, no extra per-step work solely to satisfy an artificial header limit,
one readable source representation and one eventual [AD](../../glossary.md#autograd)/execution pipeline. Cost: a deliberate
public advanced-attrs source migration and coordinated consumer guards. This is a prerequisite,
not completion of all-[PURE](../../glossary.md#term-pure) composition, wider cold [persistence](../../glossary.md#term-persistence), higher-order AD or [release](../../glossary.md#term-release) readiness.
