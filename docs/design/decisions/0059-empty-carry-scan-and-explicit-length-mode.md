# ADR 0059: Empty carry and explicit length mode for general Scan

## Status

Accepted and implemented for source only by Model0046. Independent final core and wire/evidence/
docs reviews APPROVE0; exact source/caller/guard cutover and actual source/compatibility checks
passed. This grants no typed execution/AD, wider cold/wire admission or complete release readiness.

## Context and problem

Before this cutover, Model's ScanBody rejected empty carry; ScanAttrs required carryCount>=1 and
used rowCarryCount=0 to mean all-valid. Compiler0017 consequently proposed rejecting an otherwise
ordinary first-order derivative when every original carry is discrete. Floating emissions and
sequence inputs can have a real derivative but leave no ROW cotangent carry. There can also be
only SHARED parameter accumulators in the reverse, requiring length mode with zero ROW carries.

That is a representation limit, not mathematical non-differentiability. A dummy floating carry,
fake discrete cotangent, activation replay or special unrolled reverse would hide the problem
and duplicate state/work. The delegated production target requires ordinary PURE composition
and per-value types, including real BOOL/integral primal dependencies; the proposed rejection
must not silently become a permanent smaller mathematical whitelist.

## Decision

Use the existing Scan representation for a true empty carry and make length mode explicit.
No new operation kind, top-level type, package, callback, arbitrary while or executor.

- ScanBody permits empty carry/nextCarry together; step sequences remain nonempty, and at least
  one nextCarry or emission root is required. Each existing nonempty carry still has exact stable
  shape/dtype. A carry-free scan is a fixed-T traversal of ordinary emissions, not a new map API.
- ScanAttrs retains its seven-field all-valid convenience constructor. Replace the nine-field
  canonical form with `(timeCount, carryCount, sequenceCount, invariantCount, emissionCount,
  residualCount, reverse, lengthAware, rowCarryCount, sequenceRowAxes)`. Migrate every source/test
  caller; do not retain a sentinel-based nine-field alias or compatibility shim.
- Counts remain checked/nonnegative, sequenceCount>=1, carryCount+emissionCount>=1. All-valid
  requires rowCarryCount=0 and empty axes. Length mode permits rowCarryCount in [0,carryCount],
  with exactly one existing optional axis per sequence. inputCount uses lengthAware, never the
  row count; lengths remains the final actual and never a body formal.
- Length-aware binding obtains the explicit static/named B from INT64[B] lengths, then validates
  every declared ROW coordinate against it. It never guesses a row from equal extents or reads
  lengths. Existing positive-ROW sources have the same meaning and descriptors. All-inactive
  traversal still skips every body operation, keeps carried values raw and emits positive zeros.
- Future typed AD selects floating roots/targets only. Compact positional maps preserve typed
  discrete primal histories and repeated/tied contributions. A reverse with no ROW cotangents
  uses zero ROW carries; if no SHARED floating accumulator is needed, it uses no carry at all.
  No discrete gradient or synthetic state is introduced. Existing ordinary cast/disconnected/
  unsupported-operation/first-order rules remain independently enforced.

## Boundary and migration discipline

This decision changes source representation, not permission for every consumer. A first source
cutover must preserve independently fail-closed Compiler inference/AD, direct Prepare, direct CPU
and cold/raw-codec guards for newly representable topologies until each is actually qualified.
Update every old sentinel-derived mode check to explicit lengthAware; do not accidentally execute
a zero-ROW length scan as all-valid. Existing homogeneous nonempty sources stay accepted.

The existing v1-v5 wire formats and constructor inventories stay bounded. V4 retains its old
seven fields and nonempty-carry domain; V5 retains opcode33/old tuple/roles/positive row carry.
Raw encoding, decoding and source reconstruction must independently reject newly representable
carry-free or zero-ROW length topologies before mutation/large allocation, even if Model accepts
them. Opcode23/33 determines the existing mode; no boolean wire, V6, invented opcode, alias or
silent format expansion. Existing accepted file bytes remain unchanged.

The normal Scan.apply overloads and result carrier are reused. finalCarry can be empty; emission
ports remain canonical. Fixed recurrent scans, ordinary .rnn, shared runtime/slot ownership and
existing numerical algorithms remain unchanged. Generated/cache identity changes belong only to
later actually changed consumer facts/bytes; the source flag is not an arbitrary global bump.

## Established precedent

[JAX lax.scan](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html) permits `None` as an
empty carry pytree while requiring fixed carry structure/shape/dtype and compiling a static-count
body as a loop. This supports the boring representation choice; Synaptik retains its own explicit
length/activity, source/cold and ordinary-rule contracts rather than copying a framework executor.

## Acceptance and consequences

Validate both traversal modes, T/B=0/1, emission-only all-valid/length source and zero-ROW SHARED
accumulators, exact mode/count/port/provenance/alias facts and pre-ID rejection of malformed
headers/roles. Current fixed/nonempty behaviors, old file bytes and downstream negative guards
must remain real tests. Actual typed reverse and later CPU helper composition need independent
nonuniform-oracle, inactive-padding, dtype/history, failure/resource and final workflow evidence.

Benefits: no fake state, no extra per-step work solely to satisfy an artificial header limit,
one readable source representation and one eventual AD/execution pipeline. Cost: a deliberate
public advanced-attrs source migration and coordinated consumer guards. This is a prerequisite,
not completion of all-PURE composition, wider cold persistence, higher-order AD or release readiness.
