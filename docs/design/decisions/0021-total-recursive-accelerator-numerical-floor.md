# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0021: Total Recursive [ACCELERATOR](../../glossary.md#term-accelerator) Numerical Floor


## Editorial reading context

This superseded record used recursive numerical rules: a composite formula inherited allowances at its named primitive arithmetic sites. ULP measures floating spacing; ordered-bit distance counts adjacent representations and is not the same unit. Those explanations motivate the old floor, not a current graph-profile option or permission to ignore NaN, infinity or discrete selection rules.

Use the [current operation-local direction](0026-profile-free-numerical-semantics.md) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

## Status

Superseded — 2026-10-08 by [ADR 0026](0026-profile-free-numerical-semantics.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr))</span>.
Originally accepted 2026-09-25.

Originally superseded [ADR 0019](0019-explicit-numerical-profiles.md).

Historically qualified for newly admitted low-[precision](../../glossary.md#term-precision) [`ACCELERATOR`](../../glossary.md#term-accelerator) arithmetic by
[ADR 0023](0023-low-precision-accelerator-parity.md). The profile-indexed floors and every
"current" statement below describe the pre-Task-0032 baseline only. The present [Model](../../glossary.md#term-model) rule is
the [profile-free numerical-semantics contract](../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics).

## Context

[ADR](../../glossary.md#term-architecture-decision-record-adr) 0019 established two [graph](../../glossary.md#graph)-wide cold numerical-profile [identities](../../glossary.md#term-identity-meanings) and deliberately limited
[`ACCELERATOR`](../../glossary.md#term-accelerator) to a closed [operation](../../glossary.md#operation)-row table. That table was useful while only a few [Metal](../../glossary.md#term-metal) routes
had direct evidence, but it does not compose. Every new primitive, composite, generated [gradient](../../glossary.md#term-gradient),
or [backend route](../../glossary.md#backend-route) would need another exceptional row, and the same formula could receive different
meaning depending on whether it was visible or nested inside another operation.

At the time, [Model](../../glossary.md#term-model) owned formulas, guards, contributor sets, [masks](../../glossary.md#term-boolean-mask), mappings, state transitions,
special-value behavior, and saved values. Model 0031 established [backend](../../glossary.md#backend)-independent exact-
reference strict [unary](../../glossary.md#term-unary) sets for [BFLOAT16](../../glossary.md#term-bfloat16), [FLOAT32](../../glossary.md#term-float32), and [FLOAT64](../../glossary.md#term-float64); the cumulative P12/P13 cutover
extends those sets to [FLOAT16](../../glossary.md#term-float16). This decision adds only the recursive accelerator superset. For
that accelerator bound, Java's scalar and [Vector](../../glossary.md#term-vector) [`TANH`](../../glossary.md#term-tanh) contract is at most 2.5 [ULP](../../glossary.md#term-unit-in-the-last-place-ulp) from the exact
result, while FLOAT32 Vector [lanes](../../glossary.md#term-lane) use the specified widen/evaluate/narrow adaptation. A power-of-two [binade](../../glossary.md#term-binade) boundary converts that guarantee
to an ordered distance of five adjacent [binary32](../../glossary.md#floating-point-precision-and-rounding) representation steps. The one-ULP Java
logarithmic/exponential contract and retained Metal [`EXP`](../../glossary.md#term-expx)/[`SIGMOID`](../../glossary.md#term-sigmoid) observations fit within
ordered distance two.
Historical [softmax](../../glossary.md#softmax--log-softmax) and [ERF](../../glossary.md#term-erf)/[GELU](../../glossary.md#gaussian-error-linear-unit--gelu) test tolerances are composite or fixed-algorithm [qualification](../../glossary.md#term-qualification)
oracles, not primitive-site permissions.

A durable accelerator profile therefore needs a small construction rule that applies at every
named formula site, without a [runtime](../../glossary.md#term-runtime) policy read, a profile evaluator, a [tolerance](../../glossary.md#term-tolerance) [matrix](../../glossary.md#term-matrix-meanings), or an
operation-specific exception [registry](../../glossary.md#term-registry).

## Decision drivers

- Preserve the public two-value [API](../../glossary.md#term-application-programming-interface-api): `STRICT_IEEE` and [`ACCELERATOR`](../../glossary.md#term-accelerator).
- Preserve every then-current `STRICT_IEEE` promise and freedom.
- Keep Model as the sole numerical-semantic authority and keep the selected profile graph-wide and
  cold.<span data-glossary-navigation="true"> ([graph glossary](../../glossary.md#graph))</span>
- Cover every current and future [Model](../../glossary.md#term-model) formula without adding a new policy layer.
- Preserve exact mapping, contributors, guards, identities, divisors, selections, state, and
  publication.<span data-glossary-navigation="true"> ([publication glossary](../../glossary.md#term-publication))</span>
- Permit accelerator-relevant [`FLOAT32`](../../glossary.md#term-float32) primitive and aggregation freedom while rejecting reduced
  precision, algebraic substitution, term loss, and predicate tolerance.<span data-glossary-navigation="true"> ([precision glossary](../../glossary.md#term-precision); [tolerance glossary](../../glossary.md#term-tolerance))</span>
- Make [opaque](../../glossary.md#term-opaque) vendor operations prove the same recursive set rather than receive a special waiver.
- Preserve current capabilities, routes, [cache](../../glossary.md#term-cache)/tuning [identities](../../glossary.md#term-identity-meanings), [schemas](../../glossary.md#term-schema), wires, [ABI](../../glossary.md#term-application-binary-interface-abi), and exports.

## Options considered

### Keep extending the closed [operation](../../glossary.md#operation)-row table

This repeats semantics, makes composite and generated-[gradient](../../glossary.md#term-gradient) coverage fragile, and requires a new
architecture decision for each operation family. Rejected.

### Add per-[operation](../../glossary.md#operation) tolerances or a [runtime](../../glossary.md#term-runtime) [conformance](../../glossary.md#term-conformance) [matrix](../../glossary.md#term-matrix-meanings)

A final-output [tolerance](../../glossary.md#term-tolerance) does not identify which primitive, contributor, [guard](../../glossary.md#term-guard), or special class
was relaxed. A runtime matrix would also add hot-path policy and a second semantic authority.
Rejected.

### Define one recursive construction contract

Exact/discrete behavior stays exact, named [`FLOAT32`](../../glossary.md#term-float32) primitive sites receive bounded freedom, and
aggregate sites retain every declared contributor exactly once. Composite formulas inherit those
floors. Accepted.

## Decision

The following was the accepted pre-Task-0032 profile rule. It is retained to explain the
historical baseline and no longer governs current results.

`STRICT_IEEE` is the explicit union of every current per-[operation](../../glossary.md#operation) allowed-result set, including
the completed [Model](../../glossary.md#term-model) 0031 [unary](../../glossary.md#term-unary) baseline.

[`ACCELERATOR`](../../glossary.md#term-accelerator) is currently the union of that strict set and the results constructible for
[`FLOAT32`](../../glossary.md#term-float32) by the following recursive floors. Every current non-`FLOAT32` [occurrence](../../glossary.md#term-occurrence) remains strict.
[ADR](../../glossary.md#term-architecture-decision-record-adr) 0023 reserves the only future exception without activating it.

### Exact and discrete floor

[Operation](../../glossary.md#operation) kind and attributes, [arity](../../glossary.md#term-arity) and [descriptors](../../glossary.md#term-descriptor), [Shape](../../glossary.md#shape)/[layout](../../glossary.md#layout)/[axis](../../glossary.md#term-axis) mapping, contributor
membership, [masks](../../glossary.md#term-boolean-mask), indices, [graph](../../glossary.md#graph)-[RNG state](../../glossary.md#graph-rng-state--graphrngstate) transitions, traversal, casts, conversions, ordering,
stability, ties, empty-domain behavior, [identities](../../glossary.md#term-identity-meanings), divisors, and [publication](../../glossary.md#term-publication) remain exact. A
selected payload is one original candidate representation. Predicates gain no [epsilon](../../glossary.md#epsilon), and
approximation cannot choose an index, mask, winner, ordering, or state transition.

A comparison may observe [DAZ](../../glossary.md#term-daz)-normalized operands, and [extrema](../../glossary.md#saturation-pole-and-extremum) may treat DAZ-normalized operands as
tied while preserving their existing [NaN](../../glossary.md#term-nan) and [signed-zero](../../glossary.md#nan-infinity-and-signed-zero) rules. The selected result is still an
original candidate. Copying, movement, classification, [Boolean](../../glossary.md#data-type--datatype) logic, casts, indexing, ordering,
[RNG](../../glossary.md#graph-rng-state--graphrngstate) masks/state, and max-[pooling](../../glossary.md#term-pooling) selection otherwise gain no arithmetic relaxation.

### Primitive [FLOAT32](../../glossary.md#term-float32) floor

At each arithmetic site named by a [Model](../../glossary.md#term-model) formula, a [subnormal](../../glossary.md#term-subnormal) operand may be read as same-[signed
zero](../../glossary.md#term-signed-zero) and a [finite](../../glossary.md#term-finite) subnormal result may be flushed to either signed zero. Basic addition,
subtraction, multiplication, and division perform one [round-to-nearest](../../glossary.md#floating-point-precision-and-rounding)-even `FLOAT32` [operation](../../glossary.md#operation). A
corresponding multiply/add with no Model-observable intermediate may use one `FLOAT32` [fused
multiply-add](../../glossary.md#term-fused-multiply-add-fma). [Fusion](../../glossary.md#term-fusion) never crosses [graph](../../glossary.md#graph) [nodes](../../glossary.md#node).

Only an irreducible `POW`, [`LOG`](../../glossary.md#term-log), `LOG1P`, [`EXP`](../../glossary.md#term-expx), `EXPM1`, [`ERF`](../../glossary.md#term-erf), `SQRT`, or [`TANH`](../../glossary.md#term-tanh) site receives
an elementary allowance. Ordered [binary32](../../glossary.md#floating-point-precision-and-rounding) distance uses the monotonic raw-bit key defined by the
[normative](../../glossary.md#term-normative) contract: sign-set words map through unsigned complement and nonnegative words through
sign-bit flip. Distance zero is the [correctly rounded](../../glossary.md#nan-infinity-and-signed-zero) exact reference, negative and [positive zero](../../glossary.md#nan-infinity-and-signed-zero)
are adjacent, and the bound is inclusive.

After [DAZ](../../glossary.md#term-daz) and before [FTZ](../../glossary.md#term-ftz), an ordinary finite non-subnormal reference admits only
`distance(actual, reference) <= 5`; a subnormal reference is exact or FTZ. Five is a conservative
ordered-representation distance ceiling, not five [ULP](../../glossary.md#term-unit-in-the-last-place-ulp) and not a minimality claim. It follows from
the 2.5-ULP Java scalar/[Vector](../../glossary.md#term-vector) `TANH` guarantee: at a power-of-two [binade](../../glossary.md#term-binade) boundary the smaller-side
spacing is half the Java ULP, so the guarantee can span an ordered distance of five adjacent
representation steps. The one-ULP logarithmic/exponential and retained [Metal](../../glossary.md#term-metal) evidence fit within
ordered distance two.

Domain and special classes remain formula-derived. [NaN](../../glossary.md#term-nan) cannot become ordinary. Ordinary finite
cannot become NaN or [infinity](../../glossary.md#term-infinity) except by genuine [overflow](../../glossary.md#term-overflow) or a domain path after DAZ. Required
infinity and zero signs remain except for DAZ/FTZ, the established [extrema](../../glossary.md#saturation-pole-and-extremum) tie rule, or an existing
operation-specific final exact-zero [publication](../../glossary.md#term-publication) freedom. Distance does not replace those checks.

### [Aggregate floor](../../glossary.md#term-aggregate-floor)

After exact mapping, masking, [padding](../../glossary.md#padding), bounds, and selection determine one logical cell, every
declared contributor participates exactly once. [`FLOAT32`](../../glossary.md#term-float32) [evaluation](../../glossary.md#term-evaluation) may choose any [binary](../../glossary.md#term-binary-descriptor-and-metadata) tree,
reassociate, round every step to `FLOAT32`, use primitive [DAZ](../../glossary.md#term-daz)/[FTZ](../../glossary.md#term-ftz), and fuse only corresponding
multiply/add sites. Exact empty/point [identities](../../glossary.md#term-identity-meanings) and mandatory divisors remain. No evaluation may
drop, duplicate, invent, pretruncate, or replace a term.

This applies to ordinary and masked reductions, [sum-to-Shape](../../glossary.md#sum-to-shape), scans, overlap/scatter reductions,
statistics and norms, [loss](../../glossary.md#loss) reductions, [matrix](../../glossary.md#term-matrix-meanings)/[convolution](../../glossary.md#term-channels-first-convolution-layers)/attention/recurrent contractions, and
average [pooling](../../glossary.md#term-pooling). [Extrema](../../glossary.md#saturation-pole-and-extremum) and winner selection use the exact/discrete floor. Existing final
exact-zero [publication](../../glossary.md#term-publication) freedoms remain local to their named final results.

### Composite inheritance

[Normalization](../../glossary.md#term-normalization), [activation](../../glossary.md#term-activation), [loss](../../glossary.md#loss), attention, [pooling](../../glossary.md#term-pooling), [convolution](../../glossary.md#term-channels-first-convolution-layers), random/[dropout](../../glossary.md#dropout--inverted-dropout), recurrent,
visible convenience composition, and [Compiler](../../glossary.md#term-compiler)-generated [gradient](../../glossary.md#term-gradient) formulas receive no separate
[error envelope](../../glossary.md#term-error-envelope). Exact guards run first. Every [Model](../../glossary.md#term-model) formula recurses through all named primitive
and aggregate sites, including [constants](../../glossary.md#term-constant) and [exponent](../../glossary.md#term-exponent) construction:

- `RSQRT` is `SQRT` plus division; [sigmoid](../../glossary.md#term-sigmoid) is its exact sign [guard](../../glossary.md#term-guard) plus the selected [`EXP`](../../glossary.md#term-expx), add,
  negation, and division branch.
- Exact [GELU](../../glossary.md#gaussian-error-linear-unit--gelu) names its typed constants, root, division, [`ERF`](../../glossary.md#term-erf), addition, and two multiplications.
  Tanh GELU additionally builds `x^2` and `x^3` through two multiplications and names `pi`,
  `0.044715`, root/division, additions, remaining multiplications, and [`TANH`](../../glossary.md#term-tanh). SiLU recurses
  through its stable sigmoid branch and final multiplication. None is one elementary site.<span data-glossary-navigation="true"> ([SiLU glossary](../../glossary.md#sigmoid-linear-unit--silu))</span>
- Statistics/norms, softmax/normalization, losses, contractions, pooling, attention, dropout, and
  recurrent cells recurse through the complete primitive-site ledger in the normative contract.<span data-glossary-navigation="true"> ([softmax glossary](../../glossary.md#softmax--log-softmax); [normative glossary](../../glossary.md#term-normative))</span>

Saved outputs, [masks](../../glossary.md#term-boolean-mask), indices, statistics, and state are exact stored values. An [opaque](../../glossary.md#term-opaque) vendor or
first-class selector is eligible only when its complete output set over its complete advertised
domain is proved to be a subset of the recursive set; names, [samples](../../glossary.md#term-sample-machine-learning-and-hyperparameters), and final-output tolerances
are insufficient.

Neither profile newly permits reduced [precision](../../glossary.md#term-precision), reciprocal substitution for division, algebraic
[identities](../../glossary.md#term-identity-meanings) absent from the Model formula, cross-[node](../../glossary.md#node) [contraction](../../glossary.md#contraction-dimension), term loss, hidden state changes,
or [tolerance](../../glossary.md#term-tolerance)-based predicate or selection changes.

## Consequences

- No public enum value, Config shape, policy object, runtime input, capability row, backend route,
  cache/tuning identity, schema, wire, ABI, or native export changes in this decision.<span data-glossary-navigation="true"> ([runtime glossary](../../glossary.md#term-runtime); [backend route glossary](../../glossary.md#backend-route); [cache glossary](../../glossary.md#term-cache); [identity glossary](../../glossary.md#term-identity-meanings); [schema glossary](../../glossary.md#term-schema); [ABI glossary](../../glossary.md#term-application-binary-interface-abi))</span>
- Planning and Compiler transport the exact graph-wide identity. Prepare and concrete backends may
  realize it. Runtime and Trace do not read it.<span data-glossary-navigation="true"> ([Planning glossary](../../glossary.md#term-planning); [Compiler glossary](../../glossary.md#term-compiler); [graph glossary](../../glossary.md#graph); [Prepare glossary](../../glossary.md#prepare); [backends glossary](../../glossary.md#backend); [Trace glossary](../../glossary.md#trace))</span>
- Existing CPU behavior remains the common exact subset of both profiles. Existing Metal
  capability remains unchanged and every unsupported profile/operation pair still fails closed.<span data-glossary-navigation="true"> ([CPU glossary](../../glossary.md#term-cpu); [Metal glossary](../../glossary.md#term-metal); [operation glossary](../../glossary.md#operation))</span>
- Public Tensor and every concrete Model kind document the formula, guard, special classes, and
  family assignment needed to apply the recursive floors. Generated gradient operations use the
  same graph-wide profile as forward operations.<span data-glossary-navigation="true"> ([Tensor glossary](../../glossary.md#tensor); [Model glossary](../../glossary.md#term-model); [guard glossary](../../glossary.md#term-guard); [gradient glossary](../../glossary.md#term-gradient))</span>
- ADR 0019 remains historical context for graph-wide identity, monotonicity, and fail-closed
  capability, but its closed operation-row table and rejection of a generic bounded primitive rule
  no longer govern.<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr); [fail-closed glossary](../../glossary.md#term-fail-closed))</span>

## Successor [Metal](../../glossary.md#term-metal) route-adjudication rule

This decision does not enable another Metal [operation](../../glossary.md#operation). Each future Metal operation task must first
numerically certify every candidate route against the recursive [Model](../../glossary.md#term-model) set. It must then record the
choice among direct [MPSGraph](../../glossary.md#term-mpsgraph), [fused](../../glossary.md#specialization-and-fusion) custom, and certified composition using hot-[run](../../glossary.md#term-run) time, [dispatch](../../glossary.md#term-dispatch)
count, and temporary-memory evidence before setting capability to true. That evidence is [planning](../../glossary.md#term-planning)
and [qualification](../../glossary.md#term-qualification) input only: it creates no [runtime](../../glossary.md#term-runtime) correctness check, dispatch-time benchmark, or
persistent [tolerance](../../glossary.md#term-tolerance) [matrix](../../glossary.md#term-matrix-meanings).

## Related documents

- [Foundational-module contract](../../architecture/contracts/foundational-modules.md#numerical-profiles)<span data-glossary-navigation="true"> ([module glossary](../../glossary.md#term-module))</span>
- [Compiler and automatic-differentiation contract](../../architecture/contracts/compiler-autograd.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../glossary.md#term-compiler))</span>
- [Backend execution contract](../../architecture/contracts/backend-execution.md)<span data-glossary-navigation="true"> ([Backend glossary](../../glossary.md#backend))</span>
- [ADR 0019](0019-explicit-numerical-profiles.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr))</span>
- [Model task 0031 strict unary baseline](../../planning/modules/model/tasks/0031-strict-unary-numerical-baseline.md)<span data-glossary-navigation="true"> ([Model glossary](../../glossary.md#term-model); [unary glossary](../../glossary.md#term-unary))</span>
- [Model task 0030](../../planning/modules/model/tasks/0030-total-recursive-accelerator-numerical-floor.md)
- [JDK 26 `Math` elementary-function contracts](https://docs.oracle.com/en/java/javase/26/docs/api/java.base/java/lang/Math.html)<span data-glossary-navigation="true"> ([JDK glossary](../../glossary.md#term-java-development-kit-jdk))</span>
- [JDK 26 `StrictMath` elementary-function contracts](https://docs.oracle.com/en/java/javase/26/docs/api/java.base/java/lang/StrictMath.html)
- [JDK 26 `VectorOperators` lane contracts](https://docs.oracle.com/en/java/javase/26/docs/api/jdk.incubator.vector/jdk/incubator/vector/VectorOperators.html)<span data-glossary-navigation="true"> ([lane glossary](../../glossary.md#term-lane))</span>
- [Retained CPU unary scalar evidence](../../planning/backends/cpu/tasks/0005h-portable-unary-transcendental-and-activation-closure.md)<span data-glossary-navigation="true"> ([CPU glossary](../../glossary.md#term-cpu))</span>
- [Retained CPU FLOAT32 vector evidence](../../planning/backends/cpu/tasks/0005i-float32-vector-parity-and-vector-emission-boundary.md)<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../glossary.md#term-float32); [vector glossary](../../glossary.md#term-vector))</span>
