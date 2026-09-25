# ADR 0021: Total Recursive ACCELERATOR Numerical Floor

## Status

Accepted — 2026-09-25

Supersedes [ADR 0019](0019-explicit-numerical-profiles.md).

## Context

ADR 0019 established two graph-wide cold numerical-profile identities and deliberately limited
`ACCELERATOR` to a closed operation-row table. That table was useful while only a few Metal routes
had direct evidence, but it does not compose. Every new primitive, composite, generated gradient,
or backend route would need another exceptional row, and the same formula could receive different
meaning depending on whether it was visible or nested inside another operation.

The current Model already owns complete formulas, exact guards, contributor sets, masks, mappings,
state transitions, special-value behavior, and saved values. CPU primitive gates show one- or
two-ULP behavior except `TANH` at five ULP. Retained Metal direct-route evidence shows one ULP for
`EXP` and two for `SIGMOID`. Historical four-ULP softmax and looser ERF/GELU test tolerances are
composite-output test oracles rather than evidence for a primitive elementary-function ceiling.

A durable accelerator profile therefore needs a small construction rule that applies at every
named formula site, without a runtime policy read, a profile evaluator, a tolerance matrix, or an
operation-specific exception registry.

## Decision drivers

- Preserve the public two-value API: `STRICT_IEEE` and `ACCELERATOR`.
- Preserve every current `STRICT_IEEE` promise and freedom.
- Keep Model as the sole numerical-semantic authority and keep the selected profile graph-wide and
  cold.
- Cover every current and future Model formula without adding a new policy layer.
- Preserve exact mapping, contributors, guards, identities, divisors, selections, state, and
  publication.
- Permit accelerator-relevant `FLOAT32` primitive and aggregation freedom while rejecting reduced
  precision, algebraic substitution, term loss, and predicate tolerance.
- Make opaque vendor operations prove the same recursive set rather than receive a special waiver.
- Preserve current capabilities, routes, cache/tuning identities, schemas, wires, ABI, and exports.

## Options considered

### Keep extending the closed operation-row table

This repeats semantics, makes composite and generated-gradient coverage fragile, and requires a new
architecture decision for each operation family. Rejected.

### Add per-operation tolerances or a runtime conformance matrix

A final-output tolerance does not identify which primitive, contributor, guard, or special class
was relaxed. A runtime matrix would also add hot-path policy and a second semantic authority.
Rejected.

### Define one recursive construction contract

Exact/discrete behavior stays exact, named `FLOAT32` primitive sites receive bounded freedom, and
aggregate sites retain every declared contributor exactly once. Composite formulas inherit those
floors. Accepted.

## Decision

`STRICT_IEEE` remains the exact union of every current per-operation allowed-result set.

`ACCELERATOR` is the union of that strict set and the results constructible for `FLOAT32` by the
following recursive floors. Non-`FLOAT32` behavior remains strict.

### Exact and discrete floor

Operation kind and attributes, arity and descriptors, Shape/layout/axis mapping, contributor
membership, masks, indices, graph-RNG state transitions, traversal, casts, conversions, ordering,
stability, ties, empty-domain behavior, identities, divisors, and publication remain exact. A
selected payload is one original candidate representation. Predicates gain no epsilon, and
approximation cannot choose an index, mask, winner, ordering, or state transition.

A comparison may observe DAZ-normalized operands, and extrema may treat DAZ-normalized operands as
tied while preserving their existing NaN and signed-zero rules. The selected result is still an
original candidate. Copying, movement, classification, Boolean logic, casts, indexing, ordering,
RNG masks/state, and max-pooling selection otherwise gain no arithmetic relaxation.

### Primitive FLOAT32 floor

At each arithmetic site named by a Model formula, a subnormal operand may be read as same-signed
zero and a finite subnormal result may be flushed to either signed zero. Basic addition,
subtraction, multiplication, and division perform one round-to-nearest-even `FLOAT32` operation. A
corresponding multiply/add with no Model-observable intermediate may use one `FLOAT32` fused
multiply-add. Fusion never crosses graph nodes.

Each `POW`, logarithmic, exponential, error-function, root/reciprocal-root, sigmoid/tanh, or
formula-named activation site may return an ordinary finite result at most five monotonically
ordered `FLOAT32` representations from the correctly rounded exact mathematical result after DAZ
and before FTZ. Five is the smallest single ceiling supported by current direct-route evidence. It
is a site rule, not a final-output envelope.

Domain and special classes remain formula-derived. NaN cannot become ordinary. Ordinary finite
cannot become NaN or infinity except by genuine overflow or a domain path after DAZ. Required
infinity and zero signs remain except for DAZ/FTZ, the established extrema tie rule, or an existing
operation-specific final exact-zero publication freedom. Five ULP applies only to ordinary finite
non-subnormal references; a subnormal reference is exact or FTZ.

### Aggregate floor

After exact mapping, masking, padding, bounds, and selection determine one logical cell, every
declared contributor participates exactly once. `FLOAT32` evaluation may choose any binary tree,
reassociate, round every step to `FLOAT32`, use primitive DAZ/FTZ, and fuse only corresponding
multiply/add sites. Exact empty/point identities and mandatory divisors remain. No evaluation may
drop, duplicate, invent, pretruncate, or replace a term.

This applies to ordinary and masked reductions, sum-to-Shape, scans, overlap/scatter reductions,
statistics and norms, loss reductions, matrix/convolution/attention/recurrent contractions, and
average pooling. Extrema and winner selection use the exact/discrete floor. Existing final
exact-zero publication freedoms remain local to their named final results.

### Composite inheritance

Normalization, activation, loss, attention, pooling, convolution, random/dropout, recurrent,
visible convenience composition, and Compiler-generated gradient formulas receive no separate
error envelope. Exact guards run first. Their authoritative Model formulas recurse through the
three floors at named sites. Saved outputs, masks, indices, statistics, and state are exact stored
values. An opaque vendor selector is eligible only when its complete output set is proved to be a
subset of the recursive set.

Neither profile newly permits reduced precision, reciprocal substitution for division, algebraic
identities absent from the Model formula, cross-node contraction, term loss, hidden state changes,
or tolerance-based predicate or selection changes.

## Consequences

- No public enum value, Config shape, policy object, runtime input, capability row, backend route,
  cache/tuning identity, schema, wire, ABI, or native export changes in this decision.
- Planning and Compiler transport the exact graph-wide identity. Prepare and concrete backends may
  realize it. Runtime and Trace do not read it.
- Existing CPU behavior remains the common exact subset of both profiles. Existing Metal
  capability remains unchanged and every unsupported profile/operation pair still fails closed.
- Public Tensor and every concrete Model kind document the formula, guard, special classes, and
  family assignment needed to apply the recursive floors. Generated gradient operations use the
  same graph-wide profile as forward operations.
- ADR 0019 remains historical context for graph-wide identity, monotonicity, and fail-closed
  capability, but its closed operation-row table and rejection of a generic bounded primitive rule
  no longer govern.

## Successor Metal route-adjudication rule

This decision does not enable another Metal operation. Each future Metal operation task must first
numerically certify every candidate route against the recursive Model set. It must then record the
choice among direct MPSGraph, fused custom, and certified composition using hot-run time, dispatch
count, and temporary-memory evidence before setting capability to true. That evidence is planning
and qualification input only: it creates no runtime correctness check, dispatch-time benchmark, or
persistent tolerance matrix.

## Related documents

- [Foundational-module contract](../../architecture/contracts/foundational-modules.md#numerical-profiles)
- [Compiler and automatic-differentiation contract](../../architecture/contracts/compiler-autograd.md)
- [Backend execution contract](../../architecture/contracts/backend-execution.md)
- [ADR 0019](0019-explicit-numerical-profiles.md)
- [Model task 0030](../../planning/modules/model/tasks/0030-total-recursive-accelerator-numerical-floor.md)
