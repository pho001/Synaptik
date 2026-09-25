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

The current Model owns formulas, guards, contributor sets, masks, mappings, state transitions,
special-value behavior, and saved values. Completed Model 0031 separately closes the strict unary
baseline; this decision adds only the recursive accelerator superset. Java's scalar and Vector
`TANH` contract is at most 2.5 ULP from the exact result, while FLOAT32 Vector lanes use the
specified widen/evaluate/narrow adaptation. A power-of-two binade boundary converts that guarantee
to an ordered distance of five adjacent binary32 representation steps. The one-ULP Java
logarithmic/exponential contract and retained Metal `EXP`/`SIGMOID` observations fit within
ordered distance two.
Historical softmax and ERF/GELU test tolerances are composite or fixed-algorithm qualification
oracles, not primitive-site permissions.

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

`STRICT_IEEE` is the explicit union of every current per-operation allowed-result set, including
the completed Model 0031 unary baseline.

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

Only an irreducible `POW`, `LOG`, `LOG1P`, `EXP`, `EXPM1`, `ERF`, `SQRT`, or `TANH` site receives
an elementary allowance. Ordered binary32 distance uses the monotonic raw-bit key defined by the
normative contract: sign-set words map through unsigned complement and nonnegative words through
sign-bit flip. Distance zero is the correctly rounded exact reference, negative and positive zero
are adjacent, and the bound is inclusive.

After DAZ and before FTZ, an ordinary finite non-subnormal reference admits only
`distance(actual, reference) <= 5`; a subnormal reference is exact or FTZ. Five is a conservative
ordered-representation distance ceiling, not five ULP and not a minimality claim. It follows from
the 2.5-ULP Java scalar/Vector `TANH` guarantee: at a power-of-two binade boundary the smaller-side
spacing is half the Java ULP, so the guarantee can span an ordered distance of five adjacent
representation steps. The one-ULP logarithmic/exponential and retained Metal evidence fit within
ordered distance two.

Domain and special classes remain formula-derived. NaN cannot become ordinary. Ordinary finite
cannot become NaN or infinity except by genuine overflow or a domain path after DAZ. Required
infinity and zero signs remain except for DAZ/FTZ, the established extrema tie rule, or an existing
operation-specific final exact-zero publication freedom. Distance does not replace those checks.

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
error envelope. Exact guards run first. Every Model formula recurses through all named primitive
and aggregate sites, including constants and exponent construction:

- `RSQRT` is `SQRT` plus division; sigmoid is its exact sign guard plus the selected `EXP`, add,
  negation, and division branch.
- Exact GELU names its typed constants, root, division, `ERF`, addition, and two multiplications.
  Tanh GELU additionally builds `x^2` and `x^3` through two multiplications and names `pi`,
  `0.044715`, root/division, additions, remaining multiplications, and `TANH`. SiLU recurses
  through its stable sigmoid branch and final multiplication. None is one elementary site.
- Statistics/norms, softmax/normalization, losses, contractions, pooling, attention, dropout, and
  recurrent cells recurse through the complete primitive-site ledger in the normative contract.

Saved outputs, masks, indices, statistics, and state are exact stored values. An opaque vendor or
first-class selector is eligible only when its complete output set over its complete advertised
domain is proved to be a subset of the recursive set; names, samples, and final-output tolerances
are insufficient.

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
- [Model task 0031 strict unary baseline](../../planning/modules/model/tasks/0031-strict-unary-numerical-baseline.md)
- [Model task 0030](../../planning/modules/model/tasks/0030-total-recursive-accelerator-numerical-floor.md)
- [JDK 26 `Math` elementary-function contracts](https://docs.oracle.com/en/java/javase/26/docs/api/java.base/java/lang/Math.html)
- [JDK 26 `StrictMath` elementary-function contracts](https://docs.oracle.com/en/java/javase/26/docs/api/java.base/java/lang/StrictMath.html)
- [JDK 26 `VectorOperators` lane contracts](https://docs.oracle.com/en/java/javase/26/docs/api/jdk.incubator.vector/jdk/incubator/vector/VectorOperators.html)
- [Retained CPU unary scalar evidence](../../planning/backends/cpu/tasks/0005h-portable-unary-transcendental-and-activation-closure.md)
- [Retained CPU FLOAT32 vector evidence](../../planning/backends/cpu/tasks/0005i-float32-vector-parity-and-vector-emission-boundary.md)
