# [Model](../../../glossary.md#term-model) Capability Baseline

**Editorial reading key (later context, not a status update).**

Here ‘Model owns’ means responsibility for backend-independent meaning and valid descriptions, not possession of every tensor buffer. An application calls Tensor methods to describe operations; Compiler consumes that provenance to build graph records; Prepare and a backend supply executable realizations, and Runtime manages an invocation. A leaf has no producing expression and can be storage-free or backed by supplied values. These planning capability classes therefore do not establish execution support. The baseline contains later amendments as well as older selections; use the focused current references for the present vocabulary without applying their later choices retroactively to individual task records.

Focused current reading: [Current operation/signature reference](../../../reference/tensor/operation-foundation.md#operation-semantic-foundation); [Creation versus allocation and import](../../../reference/tensor/creation.md). The original record below retains its own scope, decisions and evidence.


## Purpose and authority

This document selects the [backend](../../../glossary.md#backend)-independent [tensor](../../../glossary.md#tensor) capabilities that Synaptik intends to expose
or represent. It is a [planning](../../../glossary.md#term-planning) baseline, not a complete tensor-library reference and not an
[architecture contract](../../../glossary.md#architecture-contract).

[`ARCHITECTURE.md`](../../../../ARCHITECTURE.md) is authoritative. In particular, [model](../../../glossary.md#term-model) owns
tensor and [operation](../../../glossary.md#operation) semantics, [compiler](../../../glossary.md#term-compiler) owns [gradient](../../../glossary.md#term-gradient) rules and backward-[graph](../../../glossary.md#graph) construction, and
backend [prepare](../../../glossary.md#prepare) owns [lowering](../../../glossary.md#lowering), [specialization](../../../glossary.md#term-specialization), [fusion](../../../glossary.md#term-fusion), and [kernel](../../../glossary.md#kernel) choice. If this plan conflicts
with that contract, implementation must stop rather than reinterpret the contract here.

The read-only `legacy/pre-rewrite` branch remains evidence for useful behavior and compatibility
tests. It is not the selection rule for the new [API](../../../glossary.md#term-application-programming-interface-api). The baseline is now chosen by semantic
coherence, usefulness for [inference](../../../glossary.md#term-inference) and training, interoperability, and the ability to specify the
behavior independently of a backend.

## Capability selection [model](../../../glossary.md#term-model)

Every candidate belongs to one of five [planning](../../../glossary.md#term-planning) classes:

| Class | Meaning |
|---|---|
| Required baseline | Needed before Synaptik can provide a useful minimal [inference](../../../glossary.md#term-inference) or training model. |
| Important next | Valuable soon after the required baseline, but not a prerequisite for the first useful system. |
| Convenience | Public spelling composed from selected semantics; it does not require a distinct [operation](../../../glossary.md#operation) kind. |
| [Compiler](../../../glossary.md#term-compiler)-only | [Backend](../../../glossary.md#backend)-independent semantics that compiler transformations may emit but public [Tensor](../../../glossary.md#tensor) does not expose. |
| Deferred or rejected | Excluded until a concrete use case and a precise semantic contract justify it. |

An operation kind in model is not end-to-end support. Public expression construction, compiler
[capture](../../../glossary.md#term-capture-meanings) and [validation](../../../glossary.md#term-validation), backend-neutral [ownership](../../../glossary.md#term-ownership-meanings) planning, backend preparation, [runtime](../../../glossary.md#term-runtime)
execution, and [conformance](../../../glossary.md#term-conformance) tests remain separate completion layers.

## Foundation hardening before new [operation](../../../glossary.md#operation) families

The capability reset identified four representation limits before task 0019. Completed tasks
0018K–0018N resolved them; the subsections below retain the selected contracts and rationale.

### Valid operations and [occurrence](../../../glossary.md#term-occurrence) [signatures](../../../glossary.md#term-signature)

Completed task 0018K now rejects an [`OperationKind`](../../../glossary.md#operationkind) paired with the wrong concrete
[`OperationAttrs`](../../../glossary.md#operationattrs) class and gives every kind family explicit input/output occurrence [cardinality](../../../glossary.md#term-cardinality).
`CompiledNode` validates those local counts after its collection checks.

The implemented direction is a small typed [operation](../../../glossary.md#operation)-signature and construction contract,
colocated with each kind family. It:

- reject a kind paired with the wrong attribute type when `Operation` is constructed;
- describe fixed or bounded input and output cardinality without a global [registry](../../../glossary.md#term-registry);
- let occurrence-aware [validation](../../../glossary.md#term-validation) check [arity](../../../glossary.md#term-arity) where ordered inputs and outputs are available;
- support [variadic](../../../glossary.md#term-variadic) operations such as concat and genuine multi-output operations; and
- avoid [backend](../../../glossary.md#backend), [lowering](../../../glossary.md#lowering), [gradient](../../../glossary.md#term-gradient), or execution [metadata](../../../glossary.md#term-metadata).

This is not permission for a [service locator](../../../glossary.md#term-service-locator), reflective registry, monolithic operation enum, or
[schema](../../../glossary.md#term-schema) framework. Family-specific validation remains close to the family; [compiler](../../../glossary.md#term-compiler) retains
[graph](../../../glossary.md#graph)-wide and operand-dependent validation.

### Multi-output [provenance](../../../glossary.md#provenance)

`CompiledNode` permits multiple outputs, and completed task 0018L now lets public [Tensor](../../../glossary.md#tensor)
provenance say that two result tensors came from the same [producer](../../../glossary.md#term-producer). Public unstack deliberately
creates independent `SELECT` operations at output index zero instead of pretending those results
share one producer.

Top-K values/indices, auxiliary [normalization](../../../glossary.md#term-normalization) statistics, [graph](../../../glossary.md#graph) random operations returning
updated random-number-generator ([RNG](../../../glossary.md#graph-rng-state--graphrngstate)) state, and attention [weights](../../../glossary.md#term-weight) now use true shared producer
provenance. The selected design is one [immutable](../../../glossary.md#term-immutable) [identity](../../../glossary.md#term-identity-meanings)-
bearing producer with an exact [operation](../../../glossary.md#operation), ordered input Tensors, and ordered output [descriptors](../../../glossary.md#term-descriptor).
Each result Tensor carries the exact producer reference plus its zero-based output position.
Output count is derived from the descriptor list. Accepted [ADR](../../../glossary.md#term-architecture-decision-record-adr) 0009 exposes one [compiler](../../../glossary.md#term-compiler)
prerequisite that completed task 0018L deliberately excluded: pre-[capture](../../../glossary.md#term-capture-meanings) [gradient](../../../glossary.md#term-gradient) formulas need
the [canonical](../../../glossary.md#term-canonical-form) exact wrapper for hidden auxiliary positions. Completed task 0025 now makes the
producer retain and retrieve each factory-created output Tensor. The same representation covers
single-output expressions at output position zero. It remains provenance, not graph [IR](../../../glossary.md#term-intermediate-representation-ir), and does
not give Tensor a graph-local [`NodeId`](../../../glossary.md#nodeid).

The reference cycle from output Tensor through provenance to producer and back to canonical
outputs is immutable, safely published, and collectable. Factory construction completes the final
output [snapshot](../../../glossary.md#term-snapshot) before returning any wrapper. Retaining one result may retain siblings, but the
cycle owns no external resource. No wrapper reconstruction, global [registry](../../../glossary.md#term-registry), weak-reference
[lifecycle](../../../glossary.md#lifecycle), gradient state, storage [ownership](../../../glossary.md#term-ownership-meanings), or [runtime](../../../glossary.md#term-runtime) resource is selected.

Unstack does not justify a true multi-output primitive. It is a public convenience that constructs
independent [scalar](../../../glossary.md#term-scalar) `select` expressions. Genuine multi-output operations use the shared producer
contract.

### Dynamic [shape](../../../glossary.md#shape) expressions

`DynamicDimension` represents one exact named [extent](../../../glossary.md#term-extent). Before the symbolic foundation and its
first adoption task, [padding](../../../glossary.md#padding) and [tiling](../../../glossary.md#tiling) preserved such a [dimension](../../../glossary.md#dimension) only for [identity](../../../glossary.md#term-identity-meanings) requests,
concat had a narrow zero-extent exception, and [convolution](../../../glossary.md#term-channels-first-convolution-layers) or [pooling](../../../glossary.md#term-pooling) could not describe ordinary
dynamic [spatial](../../../glossary.md#term-spatial-axes) results.

Completed task 0018M adds a deliberately small non-negative [symbolic extent expression](../../../glossary.md#symbolic-extent-expression) capable
of:

- a [symbol](../../../glossary.md#term-symbol) or static [constant](../../../glossary.md#term-constant);
- checked addition, including sums of different symbols;
- a checked signed constant offset so window formulas can represent values such as `N - 3` while
  retaining the requirement that every concrete result be non-negative;
- multiplication by a non-negative constant;
- floor or ceiling division by a positive constant; and
- a generated unknown extent with explicit constraints when an exact supported expression is not
  available.

The implemented compact form is a [canonical](../../../glossary.md#term-canonical-form) [linear](../../../glossary.md#term-linear-meanings) combination with positive dimension
coefficients and a signed constant offset, explicit floor and ceiling division [nodes](../../../glossary.md#node), and an
identity-based unknown with a non-negative minimum plus optional inclusive dimension upper bound.
Static folding, neutral identities, repeated-term combination, [commutative](../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks) sum equality, and local
invalid-[argument](../../../glossary.md#term-argument) checks belong with the shape value [model](../../../glossary.md#term-model).

Completed follow-up [0018M1](tasks/0018m1-dynamic-extent-adoption.md) now adopts that foundation
in pad, tile, and concat. Padding retains canonical `N + before + after`, tiling retains
`repeat * N`, and concat retains canonical sums such as `N + M`; neutral operations preserve
exact Dimension references. Completed task
[0018R](tasks/0018r-slice-and-window-public-contract-cleanup.md) keeps [slice](../../../glossary.md#slice)-selected and
window-transformed dimensions static because literal bound clamping and current window
compatibility cannot be proved from an unbound extent; it does not add a second partial dynamic
contract. Solving [graph](../../../glossary.md#graph)-wide equalities, [binding](../../../glossary.md#term-binding) [runtime](../../../glossary.md#term-runtime) sizes, and evaluating prepared/[run](../../../glossary.md#term-run)
shapes remain [compiler](../../../glossary.md#term-compiler) and later [lifecycle](../../../glossary.md#lifecycle) concerns. Deferred compiler [inference](../../../glossary.md#term-inference) alone is
insufficient because public expression results require honest Shape [metadata](../../../glossary.md#term-metadata) when they are
constructed.

### Typed [scalar](../../../glossary.md#term-scalar) values

Completed task 0018N replaces ambiguous [binary64](../../../glossary.md#floating-point-precision-and-rounding) scalar-[operation](../../../glossary.md#operation) and [padding](../../../glossary.md#padding) attributes with one
small [immutable](../../../glossary.md#term-immutable) [`ScalarValue`](../../../glossary.md#typed-scalar-value--scalarvalue) for the current six data types. It preserves exact [FLOAT64](../../../glossary.md#term-float64) and
[FLOAT32](../../../glossary.md#term-float32) bits, raw [BFLOAT16](../../../glossary.md#term-bfloat16) bits, signed [INT32](../../../glossary.md#data-type--datatype) and [INT64](../../../glossary.md#data-type--datatype) values, and [canonical](../../../glossary.md#term-canonical-form) [BOOL](../../../glossary.md#data-type--datatype). Scalar,
clamp, and padding attributes retain that value, and public [Tensor](../../../glossary.md#tensor) construction requires exact
receiver/value data-type equality without implicit conversion.

`ScalarValue` is [model](../../../glossary.md#term-model) semantic state, not a scalar Tensor, [storage carrier](../../../glossary.md#term-storage-carrier), conversion request,
or executable [constant](../../../glossary.md#term-constant). The existing `double` Tensor overloads remain exact-FLOAT64 conveniences,
and existing primitive TensorFactory scalar/full methods remain unchanged. Future [FLOAT16](../../../glossary.md#term-float16) or
cross-type conversion requires a separately specified contract rather than a generic [payload](../../../glossary.md#artifact-envelope-and-atomic-publication) or
speculative numeric category.

## Existing capability decisions

The following decisions supersede legacy parity as the default outcome. Focused cleanup tasks
apply them in task order; completed task 0018O has already finalized indexing.

| Existing capability | Decision |
|---|---|
| `fastExp` and `fastTanh` public kinds | Completed task [0018P](tasks/0018p-elementwise-semantic-cleanup.md) removed both kinds and methods atomically without [aliases](../../../glossary.md#aliasing-and-arena). Approximation route choice belongs to [backend](../../../glossary.md#backend) [prepare](../../../glossary.md#prepare). A future portable approximate [operation](../../../glossary.md#operation) would need an explicit accuracy and special-value contract. |
| Masked sum/[mean](../../../glossary.md#term-mean) [heuristic](../../../glossary.md#term-heuristic) [axis](../../../glossary.md#term-axis) mapping | Completed task [0018Q](tasks/0018q-masked-reduction-redesign.md) removed the mapper and simplified masked attributes to one [normalized axis](../../../glossary.md#normalized-axis). The public overloads remain first-class two-input SUM/MEAN occurrences because current primitives cannot safely compose masked-out [NaN](../../../glossary.md#term-nan)/Inf exclusion, dynamic [Shapes](../../../glossary.md#shape), selected counts, and [gradients](../../../glossary.md#term-gradient) without hidden [eager](../../../glossary.md#term-eager-evaluation) [constants](../../../glossary.md#term-constant) or undefined behavior. [Masks](../../../glossary.md#term-boolean-mask) must use ordinary right-aligned [broadcasting](../../../glossary.md#broadcasting) to produce exactly the input Shape; callers reshape or expand explicitly. All-false sum is zero and all-false mean is NaN. |
| `inv` | Completed task [0018P](tasks/0018p-elementwise-semantic-cleanup.md) atomically renamed the semantic kind and public method to `RECIPROCAL` and `reciprocal`, without a compatibility bridge. |
| `foldAxis` public method | Completed task [0018R](tasks/0018r-slice-and-window-public-contract-cleanup.md) historically removed the public method and helper path without an alias while retaining `WindowTransformKind.FOLD_AXIS` and `FoldAxisAttrs`. Completed [task 0023D](tasks/0023d-public-fold-axis-and-dynamic-window-transforms.md) now restores exactly `foldAxis(int,long,long)` as the generally useful overlap-summing public transformation. Its retained `long outputSize` is sufficient because current `unfold` requires its selected [extent](../../../glossary.md#term-extent) static. |
| `fromStrictFlatPrefix` and `fromCyclicFlatPrefix` | Completed task [0018S](tasks/0018s-tensor-factory-surface-cleanup.md) removed these methods and their implementation from production without aliases. Exact-[carrier](../../../glossary.md#term-storage-carrier) strict/cyclic preparation remains only in a package-private test-source [fixture](../../../glossary.md#term-fixture) helper and delegates to public flat import. |
| Primitive-array `take` | Do not stabilize it. [Canonical](../../../glossary.md#term-canonical-form) indexing accepts an index [Tensor](../../../glossary.md#tensor); any later primitive convenience must validate the axis before allocating its eager index Tensor. |
| Positive-step-only slicing | Completed task [0018R](tasks/0018r-slice-and-window-public-contract-cleanup.md) selected signed non-zero steps. Normalized `SliceAttrs` stores start plus selected length rather than an exclusive-end sentinel, the general arrays remain the primitive, a step-aware `sliceAxis` overload was added, and `flip(int... axes)` is one [`SLICE`](../../../glossary.md#slice) convenience. Positive provable [views](../../../glossary.md#term-view-meanings) remain resolved; every negative-step result is [layout](../../../glossary.md#layout)-unresolved under the current non-negative-[stride](../../../glossary.md#term-stride) [descriptor](../../../glossary.md#term-descriptor). |
| Large `Tensor` and `TensorFactory` surfaces | Keep Tensor as the public [identity](../../../glossary.md#term-identity-meanings)/fluent entry point, but continue to isolate implementation by cohesive helpers. Completed task [0018S](tasks/0018s-tensor-factory-surface-cleanup.md) narrowed TensorFactory to identity, construction, allocation, imports, constants, and integer ranges; promoted existing stateless `TensorRandoms` as the sole public explicit-source random [owner](../../../glossary.md#term-owner); and moved prefix population to test source. Class size alone does not justify another [facade](../../../glossary.md#term-facade). |

Completed task 0018P retains [`EXP`](../../../glossary.md#term-expx) and [`TANH`](../../../glossary.md#term-tanh) as portable mathematical requests without promising an
algorithm, bitwise result, approximation bound, or [backend route](../../../glossary.md#backend-route). It deliberately left the
provisional exact typed [scalar](../../../glossary.md#term-scalar) vocabulary unchanged. Completed task
[0018T](tasks/0018t-scalar-arithmetic-family-normalization.md) now selects a complete parallel
`ADD`, `SUB`, `MUL`, `DIV`, `MIN`, `MAX`, and `POW` Tensor/[binary](../../../glossary.md#term-binary-descriptor-and-metadata) and Tensor/scalar vocabulary,
retains `CLAMP` as the only distinct range kind, and removes `CLAMP_MIN`/`CLAMP_MAX` in favor of
public conveniences over scalar `MAX`/`MIN`. Pairwise public [extrema](../../../glossary.md#saturation-pole-and-extremum) use `minimum`/`maximum` while
aggregate reductions keep `min`/`max`. Completed task
[0018T1](tasks/0018t1-unary-numeric-gaps-and-floating-diagnostics.md) separately owns
floating-preserving `rsqrt`, `log1p`, and `expm1` plus fixed-[BOOL](../../../glossary.md#data-type--datatype) floating classifications.

### [Scalar](../../../glossary.md#term-scalar) arithmetic [normalization](../../../glossary.md#term-normalization)

Completed task 0018T is intentionally the complete scalar arithmetic family rather than a partial
set of new overloads. Every selected [Tensor](../../../glossary.md#tensor)-to-Tensor arithmetic relationship receives the matching
exact typed scalar form. A [`ScalarValue`](../../../glossary.md#typed-scalar-value--scalarvalue) remains an [operation](../../../glossary.md#operation) attribute, not an [eager](../../../glossary.md#term-eager-evaluation) [rank-zero](../../../glossary.md#axis-rank-scalar-vector-and-matrix)
Tensor input; the current exact receiver/value data-type check and floating-only public boundary
remain until task 0018U deliberately broadens selected integral domains.

The public vocabulary distinguishes pairwise [extrema](../../../glossary.md#saturation-pole-and-extremum) from reduction: `minimum(other-or-scalar)`
and `maximum(other-or-scalar)` select one value at each output [coordinate](../../../glossary.md#axis-rank-scalar-vector-and-matrix), whereas `min(...)` and
`max(...)` aggregate a domain. The extrema semantics propagate [NaN](../../../glossary.md#term-nan), order [infinities](../../../glossary.md#nan-infinity-and-signed-zero) normally,
select [negative zero](../../../glossary.md#nan-infinity-and-signed-zero) for minimum and [positive zero](../../../glossary.md#nan-infinity-and-signed-zero) for maximum. `CLAMP(x, lower, upper)` means one
first-class semantic request equivalent in value to `minimum(maximum(x, lower), upper)`; one-bound
clamp methods remain conveniences that create only scalar `MAX` or `MIN` [producers](../../../glossary.md#term-producer).

### [Unary](../../../glossary.md#term-unary) numeric completion and [floating classification](../../../glossary.md#floating-classification)

Completed task 0018T1 selects `RSQRT`, `LOG1P`, and `EXPM1` as first-class parameterless unary
functions rather than public compositions. Keeping the original semantic request lets later [compiler](../../../glossary.md#term-compiler) and
[backend](../../../glossary.md#backend) work preserve accuracy near zero for `log1p`/`expm1` and choose an appropriate reciprocal-
square-root implementation without exposing a [backend route](../../../glossary.md#backend-route) in [model](../../../glossary.md#term-model). These transforms preserve
the floating input type, [Shape](../../../glossary.md#shape), and [gradient](../../../glossary.md#term-gradient)-eligibility request.

Floating value classification is a separate typed family because `isFinite`, `isNaN`, and
`isInf` return [BOOL](../../../glossary.md#data-type--datatype) with false gradient eligibility. The selected public boundary accepts only
[BFLOAT16](../../../glossary.md#term-bfloat16), [FLOAT32](../../../glossary.md#term-float32), and [FLOAT64](../../../glossary.md#term-float64); integral inputs would make every answer statically trivial and are
not included without a concrete use case. Classification distinguishes all [finite](../../../glossary.md#term-finite) normal,
[subnormal](../../../glossary.md#term-subnormal), and [signed-zero](../../../glossary.md#nan-infinity-and-signed-zero) values from both [infinities](../../../glossary.md#nan-infinity-and-signed-zero) and every [NaN](../../../glossary.md#term-nan) encoding. The package uses
“classification” rather than tracing-oriented “diagnostic” [ownership](../../../glossary.md#term-ownership-meanings).

Task 0018T1's function names originally selected portable [targets](../../../glossary.md#term-target) and exact special-value classes
without an accuracy envelope. Completed Task 0031 now supplies a backend-independent strict
allowed-result baseline for every unary kind and accepted BFLOAT16/FLOAT32/FLOAT64 type. Exact
mathematical references have inclusive ordered-distance or per-type absolute/relative bounds;
reciprocal [square root](../../../glossary.md#norms-elementary-functions-and-probability-distributions), [sigmoid](../../../glossary.md#term-sigmoid), and the composite [activations](../../../glossary.md#term-activation) recurse through those primitive
sets and explicit one-round native or one-wider sites. The recursive `ACCELERATOR FLOAT32` floor
additionally applies only at the primitive sites named by the [normative](../../../glossary.md#term-normative) numerical-profile
contract.

The selected special-value classes preserve [signed zero](../../../glossary.md#term-signed-zero) for `log1p` and `expm1`; map the two
signed zeros to same-signed infinities for `rsqrt`; map [positive infinity](../../../glossary.md#nan-infinity-and-signed-zero) to [positive zero](../../../glossary.md#nan-infinity-and-signed-zero) for
`rsqrt`; map `-1` to [negative infinity](../../../glossary.md#nan-infinity-and-signed-zero) for `log1p`; map negative infinity to exactly `-1` for
`expm1`; and produce NaN for the documented out-of-domain or NaN cases. Model records those
meanings and allowed-result sets without evaluating them or selecting a backend route.

### Integral [elementwise](../../../glossary.md#term-elementwise) baseline

Completed task [0018U](tasks/0018u-integral-elementwise-arithmetic-and-comparisons.md) selects [INT32](../../../glossary.md#data-type--datatype)
and [INT64](../../../glossary.md#data-type--datatype) ADD, SUB, MUL, MIN, and MAX for both [Tensor](../../../glossary.md#tensor) operands and exact typed [scalar](../../../glossary.md#term-scalar) attributes.
All six existing comparisons accept the same integral domain. INT32/INT64 Tensor pairs
promote to INT64 when either input is INT64; same-width pairs retain their type. Floating promotion
remains unchanged, while floating/integral pairs and [BOOL](../../../glossary.md#data-type--datatype) require explicit different semantics or
an explicit cast.

Integral ADD, SUB, and MUL use fixed-width [two's-complement](../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks) modular wrap in the promoted result
type. MIN, MAX, and comparisons use signed order. Scalar values do not promote and must exactly
match the receiver type. Integral DIV, POW, range CLAMP, remainder, floor division, unsigned
values, and [saturation](../../../glossary.md#term-saturation) remain outside the selected task because they require separate zero,
[rounding](../../../glossary.md#term-rounding), [exponent](../../../glossary.md#term-exponent), or [overflow](../../../glossary.md#term-overflow) policy. Existing one-bound clamp conveniences inherit integral
MAX/MIN behavior without restoring a distinct clamp kind.

Completed task [0018U1](tasks/0018u1-integral-reductions-and-arg-min-normalization.md) selects
exact-type modular INT32/INT64 SUM and PROD, signed MIN and MAX, and bounded-domain empty
[identities](../../../glossary.md#term-identity-meanings): zero for SUM, one for PROD, the type maximum for MIN, and the type minimum for MAX.
Integral reductions do not widen. The same task adds `ARG_MIN`/`argMin` and atomically replaces
arg-max-only attributes and tie-policy naming with shared arg-[extrema](../../../glossary.md#saturation-pole-and-extremum) contracts, without [aliases](../../../glossary.md#aliasing-and-arena).
Both arg families use fixed INT64 indices and explicit first/last logical-index selection. Their
floating order selects [NaN](../../../glossary.md#term-nan) over non-NaN, orders [negative zero](../../../glossary.md#nan-infinity-and-signed-zero) below [positive zero](../../../glossary.md#nan-infinity-and-signed-zero), and orders
[infinities](../../../glossary.md#nan-infinity-and-signed-zero) normally. A statically empty selected [axis](../../../glossary.md#term-axis) is rejected; an unbound selected [extent](../../../glossary.md#term-extent) is
accepted structurally but must be positive when later bound. Keeping those decisions separate
prevents local elementwise promotion from being coupled to reduction-domain design. [Model](../../../glossary.md#term-model)
construction records these contracts without evaluating values, implementing [gradients](../../../glossary.md#term-gradient), or
claiming [compiler](../../../glossary.md#term-compiler), [backend](../../../glossary.md#backend), [runtime](../../../glossary.md#term-runtime), or execution support.

## Indexing taxonomy

Synaptik will use three [canonical](../../../glossary.md#term-canonical-form) semantic primitives:

- [`GATHER`](../../../glossary.md#gather): replace one data [axis](../../../glossary.md#term-axis) with the complete indices [Shape](../../../glossary.md#shape);
- `GATHER_ELEMENTS`: use same-[rank](../../../glossary.md#term-rank), non-axis-aligned indices and return the indices Shape; and
- `GATHER_ND`: use final-[dimension](../../../glossary.md#dimension) [coordinate](../../../glossary.md#axis-rank-scalar-vector-and-matrix) tuples with an explicit [batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-dimension count.

This follows [ONNX](../../../glossary.md#term-onnx)’s distinctions among [Gather](https://onnx.ai/onnx/operators/onnx__Gather.html),
[GatherElements](https://onnx.ai/onnx/operators/onnx__GatherElements.html), and
[GatherND](https://onnx.ai/onnx/operators/onnx__GatherND.html). Completed task
[0018O](tasks/0018o-indexing-taxonomy-and-unstack-normalization.md) finalized this vocabulary and
the public `gather` and `gatherElements` methods without compatibility [aliases](../../../glossary.md#aliasing-and-arena).

The removed reduced-rank gather is expressible by inserting a [singleton](../../../glossary.md#term-singleton) axis into its indices,
using [Gather Elements](../../../glossary.md#gather-elements), and removing that axis. No concrete public use case retained it.

The name `take(axis, indices)` is not retained. [NumPy `take`](https://numpy.org/doc/stable/reference/generated/numpy.take.html)
and [JAX `take`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.take.html) use one indices
Shape for every [slice](../../../glossary.md#slice) and can flatten when no axis is supplied, while
[PyTorch `Tensor.take`](https://docs.pytorch.org/docs/2.12/generated/torch.Tensor.take.html) is
flattened.
[NumPy `take_along_axis`](https://numpy.org/doc/stable/reference/generated/numpy.take_along_axis.html),
[JAX `take_along_axis`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.take_along_axis.html),
and [PyTorch `gather`](https://docs.pytorch.org/docs/2.12/generated/torch.gather.html) describe the
aligned-indices family. Synaptik should not assign `take` a fourth meaning.

The matching public scatter primitives are `SCATTER_ELEMENTS` and `SCATTER_ND`, with explicit
reduction semantics. The [task-0023 audit](adjoint-expressibility-audit.md) confirms that they
exactly express Gather Elements and [Gather-ND](../../../glossary.md#gather-nd) adjoints with ADD duplicate accumulation. General
rank-changing Gather has a different indices/update Shape relation. A positive static gathered
[extent](../../../glossary.md#term-extent) can compose through current one-hot selection and reduction, but an unresolved gathered
extent cannot because one-hot requires positive static depth. Completed
[task 0023B](tasks/0023b-gather-compatible-scatter-add.md) therefore implements final
`SCATTER_ADD` plus one public `scatterAdd(indices, updates, axis)` expression with exact Gather
result-Shape updates and fixed duplicate accumulation, rather than a backward-only kind or a
general scatter dimension-number language.
ONNX
[ScatterElements](https://onnx.ai/onnx/operators/onnx__ScatterElements.html) and
[ScatterND](https://onnx.ai/onnx/operators/onnx__ScatterND.html) provide the interoperability
vocabulary.

`select` remains the primitive for one [scalar](../../../glossary.md#term-scalar) coordinate, `slice` remains the primitive for
strided ranges, and unstack becomes an ordered repeated-select convenience. Each non-empty unstack
result has an independent one-output [producer](../../../glossary.md#term-producer) and [provenance](../../../glossary.md#provenance) output index zero; it is not a true
multi-output [operation](../../../glossary.md#operation).

## Prioritized selected baseline

### Required before a useful minimal [inference](../../../glossary.md#term-inference) and training system

Foundation hardening above precedes these operations.

- exact [scalar](../../../glossary.md#term-scalar) add, subtract, multiply, divide, minimum, and maximum;
- `reciprocal`, `rsqrt`, `log1p`, `expm1`, and classifications `isFinite`, `isNaN`, and `isInf`;
- signed integral add, subtract, multiply, minimum, maximum, comparisons, and reductions where an
  accumulation and overflow policy has been selected;<span data-glossary-navigation="true"> ([overflow glossary](../../../glossary.md#term-overflow))</span>
- `argMin` matching the explicit tie-policy structure of `argMax`;
- reductions over an ordered set of distinct [axes](../../../glossary.md#term-axis);
- `logSumExp`, variance, standard deviation, and L1/L2 norm semantics with explicit axes,
  retained-dimension, correction, and accumulation-type policies;<span data-glossary-navigation="true"> ([standard deviation glossary](../../../glossary.md#term-standard-deviation); [L2 norm glossary](../../../glossary.md#norms-elementary-functions-and-probability-distributions))</span>
- [vector](../../../glossary.md#term-vector), [matrix](../../../glossary.md#term-matrix-meanings), and batched `matmul`;
- a public [`linear`](../../../glossary.md#term-linear-meanings) convenience over matmul plus optional [bias](../../../glossary.md#term-bias);
- exact [GELU](../../../glossary.md#gaussian-error-linear-unit--gelu), fixed [tanh](../../../glossary.md#term-tanh)-approximation GELU, and [SiLU](../../../glossary.md#sigmoid-linear-unit--silu) as first-class floating [unary](../../../glossary.md#term-unary) [activations](../../../glossary.md#term-activation);
- layer normalization and RMS normalization with exact typed epsilon and explicit normalized
  trailing-Shape contracts;<span data-glossary-navigation="true"> ([layer normalization glossary](../../../glossary.md#layer-normalization))</span>
- exact-shape mean-squared error plus dense and index-target categorical cross-entropy from logits,
  with explicit reduction, denominator, class-axis, and index-ignore policies;<span data-glossary-navigation="true"> ([mean-squared error glossary](../../../glossary.md#mean-squared-error--mse); [target glossary](../../../glossary.md#term-target))</span>
- scaled dot-product attention as a backend-independent high-level semantic operation with mask,
  causal, and scale contracts; and
- explicit graph RNG state plus dropout as a state-consuming, state-producing operation with no
  hidden process-global generator.<span data-glossary-navigation="true"> ([graph RNG state glossary](../../../glossary.md#graph-rng-state--graphrngstate))</span>

[Batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[normalization](../../../glossary.md#term-normalization) inference is also selected for the minimal inference baseline. Its [model](../../../glossary.md#term-model)
contract is a stateless five-input [occurrence](../../../glossary.md#term-occurrence) with explicit data, scale, bias, running [mean](../../../glossary.md#term-mean), and
running [variance](../../../glossary.md#term-variance), one output, an explicit [layout](../../../glossary.md#layout)-neutral [channel](../../../glossary.md#term-channel) axis, and no hidden mode or
mutation. Task 0021B fixes its exact [Shape](../../../glossary.md#shape), type, [epsilon](../../../glossary.md#epsilon), formula, special-value, and [provenance](../../../glossary.md#provenance)
contracts before any training-time [statistic](../../../glossary.md#term-statistic) transition is specified.

Completed task [0018V](tasks/0018v-multi-axis-and-statistical-reductions.md) closes the selected
reduction frontier as one cohesive task. It fixes caller-ordered distinct normalized axes,
empty-axis point domains, exact structural [dimension](../../../glossary.md#dimension) retention, population-default non-negative
integral correction, floating-only [log-sum-exp](../../../glossary.md#log-sum-exp)/statistics/[L1](../../../glossary.md#term-l1)/[L2](../../../glossary.md#term-l2) results, and exact input/result
types. It also completes floating and [BOOL](../../../glossary.md#data-type--datatype) empty/special-value reduction policies while preserving
0018U1's exact-type modular integral reductions. The task records first-class semantics and
storage-free [Tensor](../../../glossary.md#tensor) [metadata](../../../glossary.md#term-metadata); algorithms, [gradients](../../../glossary.md#term-gradient), [compiler](../../../glossary.md#term-compiler) [validation](../../../glossary.md#term-validation), [backend](../../../glossary.md#backend) [lowering](../../../glossary.md#lowering), and
execution remain separately owned.

The former broad linear-algebra/attention row is decomposed without disturbing established tasks
0019A–0019C. Completed [task 0019](tasks/0019-matmul-semantics-and-tensor-expression.md) owns only
the first-class [`MATMUL`](../../../glossary.md#matrix-multiplication--matmul) primitive and public expression. Completed
[task 0019D](tasks/0019d-linear-convenience.md) owns the `linear` convenience. Completed
[task 0019E](tasks/0019e-scaled-dot-product-attention.md) owns [scaled dot-product attention](../../../glossary.md#scaled-dot-product-attention). Its
kind, attrs, public construction, tests, and documentation landed as one cohesive 17-path
capability rather than an unusable semantic/construction split.

The current MATMUL expression follows [rank-one](../../../glossary.md#axis-rank-scalar-vector-and-matrix) promotion/removal plus right-aligned [broadcasting](../../../glossary.md#broadcasting)
of leading batch axes.
It accepts same-category floating or signed-integral inputs through current promotion. A
[BFLOAT16](../../../glossary.md#term-bfloat16) result accumulates in [FLOAT32](../../../glossary.md#term-float32); FLOAT32/[FLOAT64](../../../glossary.md#term-float64) results accumulate in their promoted type;
integral results are the promoted-width modular sum of products. Floating [reassociation and fused
multiply-add](../../../glossary.md#rounding-sites-fma-and-aggregate-floor) are permitted without a bitwise-order guarantee. Static incompatibilities fail.
Unresolved [contraction](../../../glossary.md#contraction-dimension) equality may defer because it does not affect output Shape; unresolved
batch compatibility defers only when one exact output [extent](../../../glossary.md#term-extent) remains derivable.

`linear` is selected as `input.matmul(weight.transpose())` plus optional ADD, not a first-class
`LINEAR` [operation](../../../glossary.md#operation). Completed task 0019D adds exactly `linear(weight)` and `linear(weight, bias)`.
[Weight](../../../glossary.md#term-weight) Shape is `[outFeatures, inFeatures]`, input [rank](../../../glossary.md#term-rank) is at least one with final extent
`inFeatures`, and optional bias is exactly rank one `[outFeatures]`. It retains the current
same-category floating and signed-integral MATMUL/ADD promotion domains. Bias's sole Dimension
must structurally equal the weight's `outFeatures` Dimension, so current broadcasting needs no new
constraint. Complete prevalidation precedes intermediate allocation. The no-bias form creates
PERMUTE then MATMUL wrappers and returns MATMUL provenance; the bias form additionally creates ADD
and returns ADD provenance. [Transpose](../../../glossary.md#term-transpose) may preserve resolved [view](../../../glossary.md#term-view-meanings) layout, while MATMUL and ADD
remain unresolved. The composition keeps every primitive visible to later compiler inspection and
lets backend [prepare](../../../glossary.md#prepare) fuse a profitable pattern without placing a [fused](../../../glossary.md#specialization-and-fusion) variant in model.

Scaled dot-product attention remains a named high-level operation. Query `[...batch, L, E]`, key
`[...batch, S, E]`, and value `[...batch, S, Ev]` broadcast batch/head prefixes and produce
`[..., L, Ev]`. One kind accepts ordered `[query, key, value]` or
`[query, key, value, mask]` and produces either output slot zero alone or output plus [canonical](../../../glossary.md#term-canonical-form)
normalized weights at slot one. Its [immutable](../../../glossary.md#term-immutable) attributes [carry](../../../glossary.md#term-carry) an optional exact typed scale and
causal flag; absent scale selects `1 / sqrt(E)`, while a present scale is [finite](../../../glossary.md#term-finite), positive,
floating, and exactly matches the promoted input type. A BOOL [mask](../../../glossary.md#term-boolean-mask) broadcasts to score Shape
`[..., L, S]`, where true participates and false masks. Causal mode additionally retains positions
`j <= i`, and [softmax](../../../glossary.md#softmax--log-softmax) is over the final key axis. All-masked rows have zero weights and zero
output. Neither occurrence form has a [dropout](../../../glossary.md#dropout--inverted-dropout) [parameter](../../../glossary.md#term-parameter). [Graph](../../../glossary.md#graph) [RNG](../../../glossary.md#graph-rng-state--graphrngstate) and dropout remain owned by
task 0019B, and any later attention dropout must consume its explicit state.

The selected public receiver surface has two parallel four-overload families: output-only and
explicit output-plus-weights, each with key/value defaults, operation-specific attrs, optional
mask, and optional mask plus attrs. The public immutable attrs record is inspectable semantic
state rather than a general options framework. The two-output result retains both exact wrappers
from one shared [producer](../../../glossary.md#term-producer) rather than reconstructing weights. Query/key/value ranks are at least
two; their batch/head prefixes use exact three-way right-aligned broadcasting. [Embedding](../../../glossary.md#embedding)
equality/positivity, key/value sequence equality, batch [singleton](../../../glossary.md#term-singleton)-or-equal, and mask
singleton-or-equal obligations defer only when exact output Shape references remain derivable.
Static-zero embedding is invalid; empty query, key sequence, value-width, and batch axes follow
the task's explicit empty-domain results.

Eligible score [NaN](../../../glossary.md#term-nan) propagates through a row; positive-[infinity](../../../glossary.md#term-infinity) score ties split unit weight;
all-negative-infinity and all-masked rows produce positive-zero weights and output. Excluded
mask/causal positions are removed before score and value arithmetic, so their NaN/infinity values
do not contaminate the row. BFLOAT16/FLOAT32 accumulate in FLOAT32 and FLOAT64 accumulates in
FLOAT64; conforming [reassociation](../../../glossary.md#term-reassociation) and stable algorithms are allowed without bitwise guarantees.

`square`, scalar arithmetic overloads, `linear`, embedding, flatten, swap-axes, split, and chunk
are conveniences unless a focused task demonstrates a semantic reason for a distinct kind.
Task 0019A demonstrates that reason for GELU and SiLU: literal primitive chains encounter
infinity-times-zero at [negative infinity](../../../glossary.md#nan-infinity-and-signed-zero) and do not preserve the selected continuous extensions.
Exact GELU, its fixed tanh approximation, and SiLU therefore remain named unary semantics.
One-hot is also selected as a focused first-class encoding because composing it from the current
[eager](../../../glossary.md#term-eager-evaluation) range factory would allocate a depth-sized [host](../../../glossary.md#term-host) Tensor and obscure its rank-changing BOOL
contract. Layer/[RMS normalization](../../../glossary.md#rms-normalization) and scaled dot-product attention remain named
high-level semantics because their numerical and masking contracts must survive compiler
inspection even when a compiler can decompose them.

### [Normalization](../../../glossary.md#term-normalization) frontier

The former broad task 0021 is split by formula and [lifecycle](../../../glossary.md#lifecycle) boundary. Completed
[task 0021](tasks/0021-layer-normalization-semantics-and-tensor-expressions.md) owns deterministic
one-output [layer normalization](../../../glossary.md#layer-normalization). Completed
[task 0021A](tasks/0021a-rms-normalization-semantics-and-tensor-expressions.md) separately owns [RMS
normalization](../../../glossary.md#rms-normalization) because it uses the uncentered root-[mean](../../../glossary.md#term-mean)-square formula and selects explicit
no-scale and scale-only forms. Completed
[task 0021B](tasks/0021b-batch-normalization-inference.md) owns [batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-normalization [inference](../../../glossary.md#term-inference) with
mandatory explicit scale, [bias](../../../glossary.md#term-bias), running mean, and running [variance](../../../glossary.md#term-variance) inputs and one output. Completed
[task 0021C](tasks/0021c-batch-normalization-training-and-statistic-transition.md) owns one pure
training-time transition with the same five ordered [Tensor](../../../glossary.md#tensor) inputs, typed [momentum](../../../glossary.md#term-momentum) and [epsilon](../../../glossary.md#epsilon),
and five shared outputs: public normalized output and next running mean/variance plus hidden saved
batch mean/inverse [standard deviation](../../../glossary.md#term-standard-deviation) for later [compiler](../../../glossary.md#term-compiler) [autograd](../../../glossary.md#autograd). No variant selects a hidden
training/[evaluation](../../../glossary.md#term-evaluation) flag, process-global state, [mutable](../../../glossary.md#term-mutable) [model](../../../glossary.md#term-model) service, session, or [checkpoint](../../../glossary.md#term-checkpoint)
[owner](../../../glossary.md#term-owner).

Completed task 0021 normalizes a non-empty trailing [`Shape`](../../../glossary.md#shape) with [population variance](../../../glossary.md#population-variance) (correction zero) and
epsilon inside the [square root](../../../glossary.md#norms-elementary-functions-and-probability-distributions). One `LAYER_NORM` kind has exact no-affine `[input]` and affine
`[input, scale, bias]` one-output variants selected by distinct typed attrs classes, because the
current exact-class [signature](../../../glossary.md#term-signature) contract must not accept the invalid two-input count. Scale and bias
must exactly match the [normalized Shape](../../../glossary.md#normalized-shape). No-affine results retain input type; affine results use
ordered floating promotion. Epsilon is a [finite](../../../glossary.md#term-finite) positive [`ScalarValue`](../../../glossary.md#typed-scalar-value--scalarvalue) whose type exactly matches
the result. [BFLOAT16](../../../glossary.md#term-bfloat16)/[FLOAT32](../../../glossary.md#term-float32) accumulate in FLOAT32 and [FLOAT64](../../../glossary.md#term-float64) in FLOAT64. Static trailing
mismatches fail; unresolved equality may defer because output Shape remains exactly the input.
Saved mean/inverse-standard-deviation outputs remain compiler concerns rather than public results.

This selected trailing-Shape/population-variance model is consistent with the official
[PyTorch LayerNorm contract](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).
Its standardization formula and trailing-[axis](../../../glossary.md#term-axis) interpretation also align with official
[ONNX LayerNormalization](https://onnx.ai/onnx/operators/onnx__LayerNormalization.html), while
Synaptik deliberately omits [ONNX](../../../glossary.md#term-onnx)'s optional saved-[statistic](../../../glossary.md#term-statistic) outputs from this first public model
[operation](../../../glossary.md#operation).

Completed task 0021A selects one `RMS_NORM` kind with one
`RmsNormAttrs(normalizedShape, epsilon)` input-range signature. Its exact valid occurrences are
one-output `[input]` and `[input, scale]`; consecutive counts one and two make one bounded
signature safe, unlike layer normalization's one/three [cardinality](../../../glossary.md#term-cardinality) hole. Scale Shape must exactly
equal the non-empty trailing normalized Shape, and bias is absent. The formula is
`x / sqrt(mean(x * x) + epsilon)`: it is uncentered, always divides mean square by `N`, and has no
correction option. No-scale results retain input type; scaled results use ordered floating
promotion. Epsilon is finite, positive, exact-result-typed, and inside the square root.
BFLOAT16/FLOAT32 accumulate in FLOAT32 and FLOAT64 in FLOAT64. Static trailing mismatches fail;
unresolved equality may defer because result Shape stays exactly the input. The task adds no [saved
statistic](../../../glossary.md#saved-statistic), bias, hidden [parameter](../../../glossary.md#term-parameter), algorithm, [gradient](../../../glossary.md#term-gradient), compiler, [backend](../../../glossary.md#backend), or execution behavior.

This narrow formula and trailing-[dimension](../../../glossary.md#dimension) interpretation align with official
[PyTorch RMSNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html) and
[ONNX RMSNormalization](https://onnx.ai/onnx/operators/onnx__RMSNormalization.html), while
Synaptik deliberately does not copy their defaults, layer state, broadcast scale, or configurable
accumulation surface.

Completed task 0021B selects one `BATCH_NORM_INFERENCE` meaning with [immutable](../../../glossary.md#term-immutable) normalized [channel](../../../glossary.md#term-channel)-axis and
exact typed epsilon attributes, exact ordered inputs
`[input, scale, bias, runningMean, runningVariance]`, and one output. Input [rank](../../../glossary.md#term-rank) is at least two;
the channel axis may identify any input axis and therefore assumes no physical [NCHW](../../../glossary.md#term-nchw) or [NHWC](../../../glossary.md#term-nhwc)
[layout](../../../glossary.md#layout). Every affine/statistic operand is an exact [rank-one](../../../glossary.md#axis-rank-scalar-vector-and-matrix) per-channel [vector](../../../glossary.md#term-vector). Static channel
mismatches fail locally, while equality involving unresolved Dimensions may defer because output
Shape remains exactly the input Shape. All five operands are mandatory and use ordered floating
promotion. Epsilon is finite, positive, exact-result-typed, and added to running variance inside
the denominator square root. The supplied running variance is used directly as an estimated
per-channel variance: inference applies no correction conversion, recomputation, clamp, momentum,
or update. Negative values are not read or rejected during model construction and follow the
formula's floating square-root behavior. There is one fresh output at [provenance](../../../glossary.md#provenance) index zero and
no saved statistic, training flag, optional affine form, hidden [constant](../../../glossary.md#term-constant), or mutable state.

This five-input inference formula and explicit estimated-statistic roles align with official
[ONNX BatchNormalization](https://onnx.ai/onnx/operators/onnx__BatchNormalization.html), while
Synaptik deliberately makes the channel axis explicit, requires exact typed epsilon and exact
rank-one parameter/statistic Shapes, and separates inference from every training/state-transition
contract.

Completed task 0021C selects batch mean and biased population variance across every non-channel
axis for normalization, while the explicit next running variance uses the corresponding unbiased
estimate.
Its momentum is the new-batch [weight](../../../glossary.md#term-weight): `next = (1 - momentum) * old + momentum * batch`. A valid
non-empty channel therefore needs reduction count at least two; a zero-channel [occurrence](../../../glossary.md#term-occurrence) is
empty and evaluates no statistic. All five outputs share one [producer](../../../glossary.md#term-producer) in the order normalized
output, next running mean, next running variance, saved batch mean, and saved inverse standard
deviation. Only the first three are returned by `BatchNormTrainingResult`; saved outputs remain
producer-described auxiliary values for later compiler autograd rather than public state. This
biased-forward/unbiased-running distinction and momentum convention align with official
[PyTorch BatchNorm1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html).
The explicit running-stat transition and population-statistic formula are also comparable to
official [ONNX BatchNormalization](https://onnx.ai/onnx/operators/onnx__BatchNormalization.html),
but Synaptik deliberately keeps training separate from inference and does not copy ONNX's mode
attribute, output surface, fixed channel position, defaults, or population running-variance
update.

### [Loss](../../../glossary.md#loss) frontier

The former broad task 0022 is split at [target](../../../glossary.md#term-target) representation and denominator boundaries while
preserving established task IDs 0023–0024. Completed
[task 0022](tasks/0022-mean-squared-error-loss.md) established `NONE`, `SUM`, and [`MEAN`](../../../glossary.md#term-mean) as
explicit loss reductions through one exact-[shape](../../../glossary.md#shape)
mean-squared-error [operation](../../../glossary.md#operation). Completed
[task 0022A](tasks/0022a-dense-target-categorical-cross-entropy-with-logits.md) adds dense
floating categorical cross-entropy directly from logits with an explicit [class axis](../../../glossary.md#class-axis). Completed
[task 0022B](tasks/0022b-index-target-categorical-cross-entropy-with-logits.md) adds [INT32](../../../glossary.md#data-type--datatype)/[INT64](../../../glossary.md#data-type--datatype)
index-target categorical cross-entropy from logits, class-[axis](../../../glossary.md#term-axis) removal, an optional exact typed
[ignore index](../../../glossary.md#ignore-index), and a `MEAN` denominator equal to the non-ignored target count while preserving the
completed floating-target [dispatch](../../../glossary.md#term-dispatch).

[Mean-squared error](../../../glossary.md#mean-squared-error--mse) is the intentionally small [regression](../../../glossary.md#term-regression) baseline. [Prediction](../../../glossary.md#term-prediction) and target Shapes
must match positionally; the operation does not broadcast targets. `NONE` preserves the exact
prediction Shape, while `SUM` and `MEAN` produce a [scalar](../../../glossary.md#term-scalar). The mean denominator is the complete
logical [element](../../../glossary.md#term-element) count, including scalar count one; an empty sum is [positive zero](../../../glossary.md#nan-infinity-and-signed-zero) and an empty mean
is [NaN](../../../glossary.md#term-nan). This explicit three-mode vocabulary is comparable to official
[PyTorch MSE loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html)<span data-glossary-navigation="true"> ([MSE glossary](../../../glossary.md#mean-squared-error--mse))</span>,
but Synaptik deliberately omits defaults, weighting, and broadcast behavior from the selected
[model](../../../glossary.md#term-model) contract.

The categorical baseline consumes logits rather than probabilities so that stable [log-softmax](../../../glossary.md#softmax--log-softmax)
semantics remain one inspectable operation request. Dense targets have the exact logits Shape and
no ignore index. Index targets have the logits Shape with exactly the class axis removed; ignored
positions contribute positive zero to `NONE`/`SUM` and do not enter the `MEAN` denominator. Both
families accept [rank](../../../glossary.md#term-rank)-at-least-one floating logits and an explicit arbitrary class axis. Dense
targets are floating and promoted with logits; index targets are exact INT32 or INT64. Dense
`NONE` produces the logits Shape with the class axis removed; index `NONE` retains the exact target
Shape constrained to that non-class domain. `SUM` and `MEAN` are scalar. Dense `MEAN` divides by
the number of class-axis-removed groups, not by
the class count or target-value sum. [Dense target](../../../glossary.md#dense-target) values [carry](../../../glossary.md#term-carry) a [finite](../../../glossary.md#term-finite), non-negative,
class-normalized caller obligation, but model construction will read no values, reject none, and
does not renormalize supplied [weights](../../../glossary.md#term-weight). Its one first-class loss uses target-weighted stable
log-softmax directly from logits, including a positive-zero contribution for an exact zero target
weight. Index `MEAN` divides by the non-ignored group count; an all-ignored or otherwise empty
mean will be NaN, while the matching sum will be positive zero. Without ignore, a non-empty group
domain requires a positive class [extent](../../../glossary.md#term-extent). With ignore, zero class extent is also valid for a
non-empty all-ignored domain; every non-matching target remains invalid because no class index is
in range. Value-dependent and unresolved cases are deferred because model construction reads no
target values. These target
alternatives and reductions are comparable to official
[PyTorch cross entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy)
and [ONNX SoftmaxCrossEntropyLoss](https://onnx.ai/onnx/operators/onnx__SoftmaxCrossEntropyLoss.html)<span data-glossary-navigation="true"> ([ONNX glossary](../../../glossary.md#term-onnx))</span>,
while Synaptik deliberately uses an explicit arbitrary class axis and separate typed operation
families. Standalone probability-input cross entropy, standalone negative-[log](../../../glossary.md#term-log)-likelihood over
caller-supplied log probabilities, class/[sample](../../../glossary.md#term-sample-machine-learning-and-hyperparameters) weights, [masks](../../../glossary.md#term-boolean-mask), label smoothing, [binary](../../../glossary.md#term-binary-descriptor-and-metadata) cross
entropy, margin/[ranking](../../../glossary.md#term-ranking) losses, Kullback–Leibler divergence, connectionist temporal classification,
and a broad loss [registry](../../../glossary.md#term-registry)/options framework are not selected for this minimal frontier.

The former 0019A umbrella is split by semantic boundary. Completed task 0019A owns `gelu()`,
`geluTanhApproximation()`, and [canonical](../../../glossary.md#term-canonical-form) `silu()` without a `swish` [alias](../../../glossary.md#aliasing-and-arena). [`GELU`](../../../glossary.md#gaussian-error-linear-unit--gelu) uses
`x * Phi(x)`; `GELU_TANH_APPROXIMATION` uses the fixed conventional coefficient `0.044715`; and
`SILU` uses `x * sigmoid(x)`. All three are parameterless, floating-only, same-Shape operations
whose mathematical continuous extensions map [negative infinity](../../../glossary.md#nan-infinity-and-signed-zero) to [negative zero](../../../glossary.md#nan-infinity-and-signed-zero), preserve [signed
zero](../../../glossary.md#term-signed-zero), map [positive infinity](../../../glossary.md#nan-infinity-and-signed-zero) to positive infinity, and propagate NaN. They preserve type and
[gradient](../../../glossary.md#term-gradient)-eligibility [metadata](../../../glossary.md#term-metadata) with unresolved [layout](../../../glossary.md#layout), without defining algorithms or gradients.
[ReLU](../../../glossary.md#term-relu) is already current through completed tasks 0014C–0014D and is not duplicated by 0019A.
The former broad [dropout](../../../glossary.md#dropout--inverted-dropout)/[RNG](../../../glossary.md#graph-rng-state--graphrngstate) frontier is split at the reusable semantic boundary. Completed
[task 0019B](tasks/0019b-explicit-graph-rng-state-foundation.md) owns only explicit [graph RNG
state](../../../glossary.md#graph-rng-state--graphrngstate). Completed [task 0019B1](tasks/0019b1-explicit-graph-dropout-construction.md) owns dropout because
it consumes and produces that state rather than behaving as a deterministic [unary](../../../glossary.md#term-unary) [activation](../../../glossary.md#term-activation).

Completed [task 0019A1](tasks/0019a1-embedding-convenience.md) adds exactly
`weights.embedding(indices)`. The receiver is a [rank-two](../../../glossary.md#axis-rank-scalar-vector-and-matrix) floating
table `[vocabulary, embeddingDimension]`; indices are INT32 or INT64; the result Shape is the
complete indices Shape followed by the exact [embedding](../../../glossary.md#embedding) [dimension](../../../glossary.md#dimension). It is direct
axis-zero [Gather](../../../glossary.md#gather) composition with no EMBEDDING kind, intermediate [producer](../../../glossary.md#term-producer), or
[padding](../../../glossary.md#padding)/sparse/max-[norm](../../../glossary.md#term-norm)/frequency option. Result
gradient eligibility comes only from weights as inherited from Gather; [compiler](../../../glossary.md#term-compiler) [autograd](../../../glossary.md#autograd) owns any
scatter-add rule. Index values must be non-negative and below the table's axis-zero extent when
executed. Construction reads no values; compiler [validation](../../../glossary.md#term-validation) may reject captured [constant](../../../glossary.md#term-constant) indices,
while [backend](../../../glossary.md#backend) preparation and its [prepared executable](../../../glossary.md#prepared-executable--preparedexecutable) must preserve safe execution-time bounds
handling after dynamic extents are bound. [Runtime](../../../glossary.md#term-runtime) executes that prepared behavior without
inspecting the original Gather operation.

Completed [task 0019A2](tasks/0019a2-one-hot-encoding.md) adds exactly
`indices.oneHot(long depth)`. The receiver is INT32 or INT64,
`depth` is a positive static `long`, and one static depth axis is appended after every input axis.
Every existing indices Dimension is retained exactly, including dynamic dimensions.
The output is non-[differentiable](../../../glossary.md#derivative-and-chain-rule) [BOOL](../../../glossary.md#data-type--datatype) with `false` off and `true` at the matching [coordinate](../../../glossary.md#axis-rank-scalar-vector-and-matrix).
Negative indices and indices at least `depth` are invalid at execution; they do not wrap, clamp,
select a default, or produce an all-false row. Construction reads no index values, so later
compiler/backend/execution work must enforce that boundary safely. No axis, output-type,
on/off-value, ignore-index, sparse, or dynamic-depth configuration is selected. It is a
first-class one-input operation rather than TensorFactory construction or an [eager](../../../glossary.md#term-eager-evaluation)
range/comparison composition. The implemented contract is `OneHotKind.ONE_HOT` with
`OneHotAttrs(long depth)` and an exact one-input, one-output [signature](../../../glossary.md#term-signature).

### Explicit [graph RNG state](../../../glossary.md#graph-rng-state--graphrngstate) and [dropout](../../../glossary.md#dropout--inverted-dropout)

Completed [task 0019B](tasks/0019b-explicit-graph-rng-state-foundation.md) implements one [opaque](../../../glossary.md#term-opaque) public
[`GraphRngState`](../../../glossary.md#graph-rng-state--graphrngstate) backed privately by a storage-free [Tensor](../../../glossary.md#tensor) expression. Callers create a state with
explicit key and counter words. Both Java `long` values are interpreted as unsigned 64-bit bit
patterns: the key selects a caller-owned stream/domain and the counter is the next abstract
logical [sample](../../../glossary.md#term-sample-machine-learning-and-hyperparameters) position. Every bit pattern is valid. Equal key/counter attributes request the
same abstract position but separately constructed states retain distinct Tensor and [producer](../../../glossary.md#term-producer)
[identities](../../../glossary.md#term-identity-meanings).

`GraphRngKind.INITIAL_STATE` is a zero-input, one-output [operation](../../../glossary.md#operation) with [immutable](../../../glossary.md#term-immutable)
`GraphRngStateAttrs(long key, long counter)`. Its output is an opaque `INT64 Shape[2]` Tensor with
unresolved [layout](../../../glossary.md#layout), false [gradient](../../../glossary.md#term-gradient) eligibility, no label, no storage, and [provenance](../../../glossary.md#provenance) output index
zero. The two [lanes](../../../glossary.md#term-lane) are raw state words, not signed numerical values or a public arithmetic [API](../../../glossary.md#term-application-programming-interface-api).
State-consuming operations keep the key and advance the counter modulo `2^64` by their exact
logical draw count. Branching one state intentionally reuses a counter interval; sequential use
threads the returned next state.

The state format, operation ordering, and advancement counts are portable [model](../../../glossary.md#term-model) semantics, but
no random algorithm, key schedule, counter-to-bits function, floating conversion, or cross-[backend](../../../glossary.md#backend)
bitstream is selected. Until a portable algorithm is deliberately specified, deterministic replay
of sampled values is bounded to the same conforming prepared implementation and configuration.
Future [graph](../../../glossary.md#graph) [serialization](../../../glossary.md#term-serialization) must preserve the two raw words losslessly, but no byte encoding or
stable serialization [token](../../../glossary.md#term-token) is current.

Completed [task 0019B1](tasks/0019b1-explicit-graph-dropout-construction.md) adds training dropout as
`input.dropout(double probability, GraphRngState state)`. Its public `DropoutResult` exposes the
dropped Tensor and next state. One producer consumes ordered `[input, state]` and produces ordered
`[output, auxiliaryMask, nextState]`; the same-[Shape](../../../glossary.md#shape) [BOOL](../../../glossary.md#data-type--datatype) [mask](../../../glossary.md#term-boolean-mask) is a non-public auxiliary slot for
[compiler](../../../glossary.md#term-compiler)-owned backward construction. Dropout is floating-only, uses [finite](../../../glossary.md#term-finite) drop probability in
`[0, 1)`, scales kept values by `1 / (1 - probability)`, consumes one abstract draw per logical
[element](../../../glossary.md#term-element) even at probability zero, and advances dynamic Shapes by their bound execution count.
Empty tensors consume zero draws. [Inference](../../../glossary.md#term-inference) bypasses the operation and state advancement rather
than using a model-level training flag.

The current multi-output factory creates one indexed Tensor wrapper per output [descriptor](../../../glossary.md#term-descriptor), so one
dropout [occurrence](../../../glossary.md#term-occurrence) allocates exactly three wrappers and three Tensor IDs: public output slot zero,
short-lived non-public mask slot one, and next-state slot two. The public output and wrapped state
retain the shared producer and therefore keep the auxiliary BOOL descriptor/position available to
later compiler [capture](../../../glossary.md#term-capture-meanings) without exposing a public mask. No descriptor-only factory output or live
sibling-result [registry](../../../glossary.md#term-registry) is selected.

This graph contract is distinct from `TensorRandoms`. `TensorRandoms` eagerly consumes a
caller-owned [JDK](../../../glossary.md#term-java-development-kit-jdk) `RandomGenerator` to create [host](../../../glossary.md#term-host)-backed [leaf](../../../glossary.md#term-leaf) data. `GraphRngState` records
storage-free semantics for later graph execution. Neither owns or discovers a hidden generator.

### Sorting and top-K frontier

The former broad sorting row is split at the output/[Shape](../../../glossary.md#shape) boundary. Completed
[task 0019C](tasks/0019c-sort-and-argsort.md) owns stable full-[axis](../../../glossary.md#term-axis) `sort` and `argsort` as distinct
single-output semantics. Completed [task 0019C1](tasks/0019c1-top-k-values-and-indices.md) owns genuine
top-K because it replaces one Shape [extent](../../../glossary.md#term-extent)
with `k`, validates static or deferred selected-axis capacity, and returns values and indices from
one shared two-output [producer](../../../glossary.md#term-producer). Its detailed specification follows completed task 0019C;
established tasks 0019D and 0019E retain their IDs.

Task 0019C selects exactly `sort(axis)`, `sort(axis, descending)`, `argsort(axis)`, and
`argsort(axis, descending)`. `OrderingKind.SORT` returns values only and
`OrderingKind.ARGSORT` returns fixed-[INT64](../../../glossary.md#data-type--datatype) indices only; both use normalized
`SortAttrs(axis, descending)`, one input, and one output. Calling both methods creates distinct
producers rather than a shared multi-output sort result. Every current [DataType](../../../glossary.md#data-type--datatype) is accepted:
floating and signed-integral values use numerical order, while [BOOL](../../../glossary.md#data-type--datatype) uses `false < true`. Sort
preserves exact input type, Shape reference, and [gradient](../../../glossary.md#term-gradient)-eligibility request; argsort preserves
the exact Shape reference but is non-[differentiable](../../../glossary.md#derivative-and-chain-rule) INT64. Both leave [layout](../../../glossary.md#layout) unresolved.

Full sorting is always stable: equal values retain increasing logical input-[coordinate](../../../glossary.md#axis-rank-scalar-vector-and-matrix) order, so
argsort indices are deterministic and independent of physical layout or [backend](../../../glossary.md#backend) traversal. NaNs
form one final class after every non-[NaN](../../../glossary.md#term-nan) in both ascending and descending results; multiple NaNs
are stable and sort retains their exact input representations. [Negative zero](../../../glossary.md#nan-infinity-and-signed-zero) precedes [positive
zero](../../../glossary.md#nan-infinity-and-signed-zero) ascending and follows it descending, while [infinities](../../../glossary.md#nan-infinity-and-signed-zero) use ordinary numerical order.
Duplicates remain present. A static empty selected axis is valid, a dynamic extent is accepted
without [binding](../../../glossary.md#term-binding) because Shape is unchanged, and [scalar](../../../glossary.md#term-scalar) input rejects every axis. Stability and
NaN placement are fixed semantics rather than attributes or backend choices. This selected [API](../../../glossary.md#term-application-programming-interface-api)
uses the mainstream `sort`/`argsort`, axis, and descending vocabulary documented by
[NumPy](https://numpy.org/doc/stable/reference/generated/numpy.sort.html) and
[JAX](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sort.html) without importing their
algorithm-selection options.

Completed [task 0019C1](tasks/0019c1-top-k-values-and-indices.md) adds exactly
`topK(long k, int axis)` and
`topK(long k, int axis, boolean largest, boolean sorted)`, with largest/sorted defaults, plus
`TopKResult(values, indices)`. Focused `TopKKind.TOP_K` preserves `OrderingKind`'s exact existing
SORT/ARGSORT [signatures](../../../glossary.md#term-signature). One TOP_K producer consumes the input and describes ordered values
and indices slots zero and one, so the [carrier](../../../glossary.md#term-storage-carrier) retains the exact two wrappers and exactly two IDs
under the current factory seam. The `largest`/`sorted` names follow the conventional
[PyTorch top-k](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.topk.html) surface.
`TopKAttrs(axis, k, largest, sorted)` retains [normalized axis](../../../glossary.md#normalized-axis) and
non-negative static `k`. All current input types remain eligible. The selected set is the first
`k` items under the requested non-NaN order with NaNs last, so NaNs participate only when fewer
than `k` non-NaNs exist; equal boundary candidates prefer lower logical indices. `k == 0` is
valid; static `k > extent` fails; a dynamic selected extent defers `extent >= k` until binding;
scalar input remains invalid. A static empty axis therefore accepts only zero. The output Shape
replaces the selected [Dimension](../../../glossary.md#dimension) with static `k`. Sorted output follows requested value order;
unsorted output uses increasing original [logical index](../../../glossary.md#axis-rank-scalar-vector-and-matrix) so it remains deterministic. Values retain
input type/gradient eligibility, and indices are INT64/non-gradient; both are unresolved,
unlabeled, and storage-free. Neither task implements [evaluation](../../../glossary.md#term-evaluation), gradients, [compiler](../../../glossary.md#term-compiler) [capture](../../../glossary.md#term-capture-meanings),
[lowering](../../../glossary.md#lowering), [kernels](../../../glossary.md#kernel), or execution.

Top-K local [validation](../../../glossary.md#term-validation) checks input, normalizes axis, requires non-negative `k`, then rejects a
known static `k > extent`, before wrapper/ID allocation. Its exact failures are
`k must be non-negative: <k>` and
`k must not exceed selected static extent: k=<k>, axis=<axis>, extent=<extent>`. A dynamic extent
records the deferred inequality instead of choosing a backend-dependent result.

### [Convolution](../../../glossary.md#term-channels-first-convolution-layers) and [pooling](../../../glossary.md#term-pooling) frontier

Completed task [0025G](tasks/0025g-ncw-conv1d-composition.md) adds exactly two public [NCW](../../../glossary.md#term-ncw) Conv1d
receiver overloads as a visible composition of current operations, not as another [operation](../../../glossary.md#operation)-kind
inventory entry. Input and [rank](../../../glossary.md#term-rank)-three [weight](../../../glossary.md#term-weight) each use `EXPAND_DIMS(2)`, so the weight becomes
`[C_out, C_in/groups, 1, K_w]`; mapped
`Conv2dAttrs(1, stride, 0, padding, 1, dilation, groups)` drives the ordinary optional-[bias](../../../glossary.md#term-bias)
[`CONV2D`](../../../glossary.md#convolution--conv2d), and `SQUEEZE(2)` returns `[N, C_out, W_out]`. Floating promotion and numerical meaning
are inherited from Conv2d. Current [compiler](../../../glossary.md#term-compiler) [inference](../../../glossary.md#term-inference) and [gradients](../../../glossary.md#term-gradient) see only those existing
ordinary [nodes](../../../glossary.md#node). This [metadata](../../../glossary.md#term-metadata) convenience adds no `CONV1D`, `ConvNd`, [backend capability](../../../glossary.md#backend-capability),
execution, [fusion](../../../glossary.md#term-fusion), or performance result.

Completed task [0025H](tasks/0025h-ncdhw-conv3d-semantics-and-tensor-expressions.md) adds one
first-class grouped [NCDHW](../../../glossary.md#conv3d--ncdhw) [`CONV3D`](../../../glossary.md#conv3d--ncdhw) semantic family and exactly two [Tensor](../../../glossary.md#tensor) receiver overloads.
Input `[N, C_in, D, H, W]`, weight `[C_out, C_in/groups, K_d, K_h, K_w]`, and optional bias
`[C_out]` produce `[N, C_out, D_out, H_out, W_out]`. `Conv3dAttrs` owns exactly ten primitive
depth/height/width [stride](../../../glossary.md#term-stride), symmetric-[padding](../../../glossary.md#padding), [dilation](../../../glossary.md#term-dilation), and group fields. [Model](../../../glossary.md#term-model) validates static
[kernel](../../../glossary.md#kernel) and geometry relations, retains [canonical](../../../glossary.md#term-canonical-form) symbolic [spatial](../../../glossary.md#term-spatial-axes) [Dimensions](../../../glossary.md#dimension) and unresolved
[channel](../../../glossary.md#term-channel) relations, applies ordered floating promotion and the selected accumulation policy, and
records one canonical output with exact ordered [provenance](../../../glossary.md#provenance). Complete Compiler 0006B now consumes
that metadata for independent forward inference and final [validation](../../../glossary.md#term-validation) while leaving gradients
[fail-closed](../../../glossary.md#term-fail-closed). Compiler 0006C remains separate Draft gradient closure, and later [CPU](../../../glossary.md#term-cpu) work remains
outside this Model capability.

Completed task 0020 adds grouped [NCHW](../../../glossary.md#term-nchw) two-dimensional cross-correlation with weight-derived
static [kernels](../../../glossary.md#kernel), optional exact output-channel bias, ordered floating promotion, selected
accumulation/special-value meaning, and floor-mode static or symbolic spatial results. For dynamic
spatial [extent](../../../glossary.md#term-extent) `D`, symmetric padding `p`, dilation `d`, static kernel `k`, and stride `s`, current
construction retains the canonical equivalent of
`floor((D + 2p - (d * (k - 1) + 1)) / s) + 1`; static invalid geometry fails locally and unresolved
validity remains a compiler/[binding](../../../glossary.md#term-binding) obligation. Exact [descriptors](../../../glossary.md#term-descriptor) and attributes likewise retain
unresolved grouped-channel and bias-channel obligations. Construction records one fresh
storage-free result and exact two- or three-input provenance without [evaluation](../../../glossary.md#term-evaluation), gradients,
compiler [capture](../../../glossary.md#term-capture-meanings), algorithms, [backend](../../../glossary.md#backend) support, or execution.

Completed task [0020A](tasks/0020a-nchw-max-pool2d-semantics-and-tensor-expression.md) owns floating
NCHW max pooling. One `MAX_POOL2D` [occurrence](../../../glossary.md#term-occurrence) preserves type, [batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters), channel, and gradient-request
metadata while deriving floor or ceil static/symbolic spatial extents. Ceil mode uses the literal
symmetric padded grid and does not drop a terminal window that starts in trailing padding.
Padding [samples](../../../glossary.md#term-sample-machine-learning-and-hyperparameters) are excluded; an all-padding window returns [negative infinity](../../../glossary.md#nan-infinity-and-signed-zero). NaNs dominate,
[positive zero](../../../glossary.md#nan-infinity-and-signed-zero) is greater than [negative zero](../../../glossary.md#nan-infinity-and-signed-zero), and equal candidates select the first height-major
kernel sample. Existing `Window2dAttrs` remains specific to unfold/fold and conceptual zero
padding, so max pooling owns `MaxPool2dAttrs` in the pooling package.

Completed task [0020A1](tasks/0020a1-nchw-average-pool2d-semantics-and-tensor-expression.md)
separately owns floating NCHW average pooling through exactly
`averagePool2d(AveragePool2dAttrs)`. It extends `Pool2dKind` without weakening max pooling and uses
a dedicated attributes type with the same literal floor/ceiling padded grid. Every logical kernel
position counts: in-bounds values contribute to the numerator, padding contributes conceptual
positive zero, and the fixed divisor is the positive mathematical product of the two kernel sample
counts. Count-padding is not configurable and there is no divisor override or valid-sample mode.
An all-padding window returns positive zero; an exact-zero [finite](../../../glossary.md#term-finite) [mean](../../../glossary.md#term-mean) is negative zero only when
every divisor contribution is an in-bounds negative zero. [BFLOAT16](../../../glossary.md#term-bfloat16)/[FLOAT32](../../../glossary.md#term-float32) accumulate and divide
in FLOAT32, while [FLOAT64](../../../glossary.md#term-float64) uses FLOAT64; result type remains the exact input type. [NaN](../../../glossary.md#term-nan) propagates,
opposing [infinities](../../../glossary.md#nan-infinity-and-signed-zero) produce NaN, and a single [infinity](../../../glossary.md#term-infinity) sign is retained. Conforming summation
[reassociation](../../../glossary.md#term-reassociation) is allowed without bitwise or cross-backend [rounding](../../../glossary.md#term-rounding) [identity](../../../glossary.md#term-identity-meanings). Combining the two
pooling operations would exceed the cohesive path guardrail; sharing NCHW [coordinates](../../../glossary.md#axis-rank-scalar-vector-and-matrix) does not
make their numerical [parameters](../../../glossary.md#term-parameter) interchangeable.

Completed task [0025I](tasks/0025i-ncw-max-average-pool1d-composition.md) adds one public NCW
receiver for each of those Pool2d meanings without adding an operation kind. Each call validates
floating rank-three input and exact width geometry, then visibly constructs
`EXPAND_DIMS(axis 2) -> MAX_POOL2D or AVERAGE_POOL2D -> SQUEEZE(axis 2)`. The mapped Pool2d
attributes use [singleton](../../../glossary.md#term-singleton)-height geometry `(kernel=1, stride=1, padding=0, dilation=1)` and the
supplied width geometry. Every result is a fresh canonical squeeze output with unchanged type and
gradient eligibility, exact retained batch/channel Dimensions, derived width, unresolved [layout](../../../glossary.md#layout),
and ordered three-[producer](../../../glossary.md#term-producer) provenance. Maximum and average exceptional-value, count-padding,
accumulation, rounding, and gradient behavior are inherited through the actual Pool2d occurrence
and ordinary rank edits. There is no Pool1d [signature](../../../glossary.md#term-signature), captured node, dedicated gradient rule, or
backend capability. Backend support is only the conjunction of capability for each visible
component kind with its actual descriptors and attributes; recognizing the whole [topology](../../../glossary.md#term-topology) may
optimize it but must not report synthetic Pool1d support.

Completed task [0025J](tasks/0025j-first-class-ncdhw-max-average-pool3d-semantics.md) adds the
first-class `MAX_POOL3D` and `AVERAGE_POOL3D` Model family because exact depth windows cannot be
represented by a bounded [Shape](../../../glossary.md#shape)-independent composition. Each floating rank-five NCDHW receiver
call records one canonical one-input/one-output occurrence, retains exact type, batch/channel
Dimensions, and gradient-request metadata, and derives static or canonical-symbolic depth,
height, and width geometry. Literal ceil mode retains terminal all-padding windows.

Maximum [Pool3d](../../../glossary.md#pool3d--ncdhw-pooling) excludes padding, uses negative infinity for an all-padding window, treats NaN as
dominant, ranks positive zero above negative zero, and selects the first eligible depth-height-
width occurrence. Average Pool3d uses the fixed mathematical count-padding divisor
`kernelDepth * kernelHeight * kernelWidth`, the established floating accumulation/division and
single BFLOAT16 [narrowing](../../../glossary.md#term-narrowing) policy, and the established NaN, infinity, [signed-zero](../../../glossary.md#nan-infinity-and-signed-zero), and all-padding
rules. This completed capability is Model metadata and provenance only. Complete Compiler 0006B1
adopts both Pool3d signatures and all three Model-current 3D-window signatures for forward
capture, inference, final validation, and ordinary [Planning](../../../glossary.md#term-planning) [handoff](../../../glossary.md#term-handoff), while complete
[Model 0025K](tasks/0025k-public-ncdhw-unfold3d-and-fold3d-window-transforms.md) owns general
`unfold3d`/`fold3d`, and Complete detailed
[Compiler 0006B2](../compiler/tasks/0006b2-pool3d-and-3d-window-gradient-closure.md) owns exact
gradients. Complete [CPU 0008G1](../../backends/cpu/tasks/0008g1-portable-pool1d-composition-validation-and-pool3d-generated-execution.md)
adds the fully static resolved-layout non-gradient BFLOAT16/FLOAT32/FLOAT64 CPU execution subset.
That backend implementation does not broaden Model metadata, execute gradients or 3D window
transforms, or imply dynamic, [vector](../../../glossary.md#term-vector)/native, [fused](../../../glossary.md#specialization-and-fusion), or cross-backend support.

Completed task 0025K adds rank-specific `UNFOLD3D` and `FOLD3D` Model semantics and exactly three
public Tensor receivers. Rank-five NCDHW input unfolds to canonical rank-three columns
`[N, C * kD * kH * kW, DOut * HOut * WOut]`, with channel/kernel and output-grid coordinates
flattened depth-height-width and width-fastest. Literal floor/ceil grids, direct represented
positive-zero padding, exact typed padding, structural symbolic compatibility, explicit fold
[target](../../../glossary.md#term-target) Shape, geometric padding exclusion, and deterministic type-specific overlap addition are
fixed metadata contracts. The current inventory is 40 operation-kind families, 115 [constants](../../../glossary.md#term-constant),
137 signatures, and 213 declared public Tensor methods. Complete Compiler 0006B1 provides
capture/inference/final validation for all five signatures. Complete Compiler 0006B2 provides
their exact first-order adjoints and supported two-stage closure through ordinary public Tensor
expressions. Complete CPU 0008G1 supplies only the bounded current Pool3d CPU execution subset.

### Important shortly afterward

- diagonal convenience (`flip` was finalized by completed task 0018R as one [`SLICE`](../../../glossary.md#slice) convenience);
- [embedding](../../../glossary.md#embedding) convenience and focused [one-hot encoding](../../../glossary.md#one-hot-encoding) semantics;
- average [pooling](../../../glossary.md#term-pooling) with its separate divisor contract after completed [NCHW](../../../glossary.md#term-nchw) max pooling;
- batch normalization split into explicit one-output inference and a later training/state-
  transition occurrence with running-stat and saved-stat outputs;<span data-glossary-navigation="true"> ([batch normalization glossary](../../../glossary.md#batch-normalization--batch-normalization-inference-and-training); [occurrence glossary](../../../glossary.md#term-occurrence))</span>
- [graph](../../../glossary.md#graph) random [sampling](../../../glossary.md#term-sampling-machine-learning-and-hyperparameters) operations using the explicit [RNG](../../../glossary.md#graph-rng-state--graphrngstate)-state contract; and
- [FLOAT16](../../../glossary.md#term-float16) before [accelerator](../../../glossary.md#term-accelerator) mixed-[precision](../../../glossary.md#term-precision) [inference](../../../glossary.md#term-inference) or training is claimed.

### Explicitly deferred

- advanced linear algebra such as decompositions, eigenvalue operations, determinants, inverses,
  and general equation notation until a concrete workload selects a coherent subset;<span data-glossary-navigation="true"> ([linear glossary](../../../glossary.md#term-linear-meanings))</span>
- remainder and floor division until negative-operand, zero-divisor, and overflow semantics are
  selected;<span data-glossary-navigation="true"> ([overflow glossary](../../../glossary.md#term-overflow))</span>
- quantized, sparse, complex, unsigned, and distributed [tensors](../../../glossary.md#tensor);
- strings and arbitrary user-defined data types; and
- [backend](../../../glossary.md#backend)-specific or [fused](../../../glossary.md#specialization-and-fusion) [operation](../../../glossary.md#operation) variants in [model](../../../glossary.md#term-model).

Quantized and sparse support may later become important project areas, but neither should distort
the dense baseline before a concrete import or execution requirement exists.

## [Data type](../../../glossary.md#data-type--datatype) baseline

The current selected types remain [FLOAT64](../../../glossary.md#term-float64), [FLOAT32](../../../glossary.md#term-float32), [BFLOAT16](../../../glossary.md#term-bfloat16), [INT32](../../../glossary.md#data-type--datatype), [INT64](../../../glossary.md#data-type--datatype), and [BOOL](../../../glossary.md#data-type--datatype). BFLOAT16
remains part of that baseline. Future task 0026 is the separate [Model](../../../glossary.md#term-model)-owned addition of true
[IEEE](../../../glossary.md#term-ieee)-754 [binary16](../../../glossary.md#floating-point-precision-and-rounding) as [`FLOAT16`](../../../glossary.md#term-float16); [Metal](../../../glossary.md#term-metal), Intel, [AMD](../../../glossary.md#term-amd), [OpenBLAS](../../../glossary.md#term-openblas), and other [backends](../../../glossary.md#backend) must not introduce
that logical type ad hoc. `FLOAT16` and `BFLOAT16` remain distinct logical [`DataType`](../../../glossary.md#data-type--datatype) values even
when both use a two-byte or Java `short` [storage carrier](../../../glossary.md#term-storage-carrier). Only floating types are [differentiable](../../../glossary.md#derivative-and-chain-rule).

The initial arithmetic expansion is deliberately limited to signed INT32/INT64 operations with a
clear use in [shape](../../../glossary.md#shape), index, or model computation. BOOL remains logical rather than numeric.
Additional integer widths and unsigned types remain deferred.

### Mixed-[precision](../../../glossary.md#term-precision) contract

[Mixed precision](../../../glossary.md#term-mixed-precision) means that one [operation](../../../glossary.md#operation) may use different logical input, accumulation or other
intermediate, and output data types. Task 0026 must audit each affected operation or cohesive
family and state all three roles explicitly. [FLOAT32](../../../glossary.md#term-float32) accumulation is the expected default for
numerically sensitive [FLOAT16](../../../glossary.md#term-float16) and [BFLOAT16](../../../glossary.md#term-bfloat16) work, but an exception requires an explicit [Model](../../../glossary.md#term-model)
semantic contract; a [backend](../../../glossary.md#backend)-wide rule cannot silently replace operation semantics.

Portable [`MemorySegment`](../../../glossary.md#term-memorysegment) storage remains representation-only. It accepts the logical [data type](../../../glossary.md#data-type--datatype) to
derive byte geometry, but neither a two-byte [carrier](../../../glossary.md#term-storage-carrier) nor [`DataType`](../../../glossary.md#data-type--datatype) availability implies executable
arithmetic, a Java [Vector API](../../../glossary.md#term-vector-api) [lane](../../../glossary.md#term-lane), or backend support. The current [CPU](../../../glossary.md#term-cpu) generated-[kernel](../../../glossary.md#kernel) contract
claims [Vector](../../../glossary.md#term-vector) lanes only for [FLOAT64](../../../glossary.md#term-float64), FLOAT32, [INT32](../../../glossary.md#data-type--datatype), and [INT64](../../../glossary.md#data-type--datatype); no BFLOAT16 or FLOAT16 [Vector lane](../../../glossary.md#simd)
support is planned by [inference](../../../glossary.md#term-inference).

Every backend 16-bit route must pass exact filtering for logical data type, operation and
attributes, accumulation/intermediate/output contract, [target](../../../glossary.md#term-target) and provider capability, [layout](../../../glossary.md#layout),
numerical and determinism requirements, and resource validity. Route and workload [benchmarking](../../../glossary.md#term-benchmarking) or
compatible tuning evidence then informs selection among the remaining valid candidates.
`DataType` availability alone never advertises or selects an execution route.

## Numerical policy gates

Every [operation](../../../glossary.md#operation) task must settle the applicable rows before becoming [Ready](../../../glossary.md#term-ready). These are semantic
policies, not [backend](../../../glossary.md#backend) algorithm designs.

| Area | Required decisions |
|---|---|
| Floating [extrema](../../../glossary.md#saturation-pole-and-extremum) and comparisons | [NaN](../../../glossary.md#term-nan) propagation/order, [signed-zero](../../../glossary.md#nan-infinity-and-signed-zero) choice, [infinity](../../../glossary.md#term-infinity) behavior, and equal-value ties. |
| Reductions and matmul | Accumulation [data type](../../../glossary.md#data-type--datatype), output data type, [reassociation](../../../glossary.md#term-reassociation) allowance, empty-domain result, and deterministic guarantees. |
| Integral arithmetic | [Overflow](../../../glossary.md#term-overflow) behavior, division [rounding](../../../glossary.md#term-rounding), division by zero, and the `MIN_VALUE / -1` case where applicable. |
| Arg-reductions, sort, and top-K | First/last or stable tie policy, NaN placement, signed-zero ordering, empty [axes](../../../glossary.md#term-axis), and index data type. |
| Scatter | Out-of-bounds and negative indices, duplicate-[target](../../../glossary.md#term-target) behavior for every reduction, base-value participation, and determinism. |
| [Gather](../../../glossary.md#gather) and [slice](../../../glossary.md#slice) | Out-of-bounds and negative indices, negative-step [normalization](../../../glossary.md#term-normalization), empty extents, and deferred [validation](../../../glossary.md#term-validation) for dynamic axes. |
| [Variance](../../../glossary.md#term-variance) and [standard deviation](../../../glossary.md#term-standard-deviation) | Population/[sample](../../../glossary.md#term-sample-machine-learning-and-hyperparameters) correction, invalid divisor, accumulation type, complex exclusion, and negative round-off handling. |
| Norms and normalization | [Epsilon](../../../glossary.md#epsilon) placement/type, zero [norm](../../../glossary.md#term-norm), infinity, NaN, accumulation type, and empty normalized regions. |
| Transcendentals and [activations](../../../glossary.md#term-activation) | Domain, overflow/[underflow](../../../glossary.md#term-underflow), special values, accuracy contract, and discontinuity/subgradient convention. |
| Attention and [losses](../../../glossary.md#loss) | [Mask](../../../glossary.md#term-boolean-mask) meaning, all-masked rows, label bounds, ignored targets, reduction denominator, stability, and deterministic expectations. |
| [RNG](../../../glossary.md#graph-rng-state--graphrngstate) and [dropout](../../../glossary.md#dropout--inverted-dropout) | Completed task 0019B fixes two unsigned 64-bit state words, modular counter meaning, and [bounded replay](../../../glossary.md#bounded-replay) without a bitstream; completed task 0019B1 fixes [finite](../../../glossary.md#term-finite) drop probability `[0,1)`, inverted scaling, one draw per [element](../../../glossary.md#term-element), [auxiliary mask](../../../glossary.md#auxiliary-mask), three wrapper/ID outputs, and next-state output. |

[Model](../../../glossary.md#term-model) records the selected meaning. [Compiler](../../../glossary.md#term-compiler) owns differentiability rules and [backward graph](../../../glossary.md#backward-graph)
construction; backends may use different algorithms only when they satisfy that meaning.

## Fixed recurrent-[scan](../../../glossary.md#term-cumulative-scan) [metadata](../../../glossary.md#term-metadata)

Completed [task 0025E](tasks/0025e-fixed-recurrent-scan-semantic-family-and-tensor-expressions.md)
adds [Model](../../../glossary.md#term-model)-owned expression metadata for exactly `RNN_TANH`, `GRU_RESET_AFTER`, and [`LSTM`](../../../glossary.md#term-long-short-term-memory-lstm), with
exact `FORWARD` or `REVERSE` direction. Completed
[task 0025F](tasks/0025f-recurrent-scan-expression-namespace-correction.md) corrects only the
public placement: final field-free `model.tensor.RecurrentScan` exposes exactly six static biased
or [bias](../../../glossary.md#term-bias)-free `rnn`, `gru`, and `lstm` methods with explicit input first, and [`Tensor`](../../../glossary.md#tensor) has no
recurrent receiver method. They require fully static [time-major](../../../glossary.md#term-time-major) input, state, and packed-[weight](../../../glossary.md#term-weight)
[Shapes](../../../glossary.md#shape) plus one ordinary non-[gradient](../../../glossary.md#term-gradient) `INT64[batch]` valid-length [descriptor](../../../glossary.md#term-descriptor). [RNN](../../../glossary.md#term-recurrent-neural-network-rnn) and [GRU](../../../glossary.md#term-gated-recurrent-unit-gru) expose
dense output and final hidden state; LSTM also exposes final cell state. Each call creates one
[identity](../../../glossary.md#term-identity-meanings)-distinct flat [producer](../../../glossary.md#term-producer) and [canonical](../../../glossary.md#term-canonical-form) ordered wrappers, with no per-step intermediate
[graph](../../../glossary.md#graph).

The fixed equations and packing match the current [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) cells: RNN uses one [tanh](../../../glossary.md#term-tanh) transition; GRU is
reset-after in reset/update/candidate order; LSTM uses input/forget/candidate/output order. Model
records dense original-time-aligned results, positive-zero [padding](../../../glossary.md#padding), valid-prefix forward or
reverse traversal, and explicit final-state semantics for future execution. It validates only
descriptors and never reads [runtime](../../../glossary.md#term-runtime) valid-length values. Zero time and zero [batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters) are valid
metadata cases; runtime bounds remain an execution-layer obligation.

This is not an executable capability advertisement. Current package-private [Compiler](../../../glossary.md#term-compiler) forward-only
[inference](../../../glossary.md#term-inference) adopts and independently revalidates the family, while both backward-capable modes
reject it before [derivative](../../../glossary.md#term-derivative) Tensor allocation. No [Planning](../../../glossary.md#term-planning) capability provider advertises it, and
no [Engine](../../../glossary.md#term-engine), Runtime, [concrete backend](../../../glossary.md#concrete-backend), NN runtime-length [facade](../../../glossary.md#term-facade), or [backpropagation](../../../glossary.md#term-backpropagation)-through-time
([BPTT](../../../glossary.md#term-backpropagation-through-time-bptt)) implementation adopts it. Later execution and BPTT tasks own those boundaries. Existing
static NN sequence containers and their Java `long[]` lengths, static [unrolling](../../../glossary.md#term-unrolling), and compact-
output contracts remain unchanged. The
namespace is an advanced low-level expression seam, not a layer, [module](../../../glossary.md#term-module), execution service,
[registry](../../../glossary.md#term-registry), or [general scan](../../../glossary.md#general-differentiable-scan)-body abstraction; ordinary NN callers use `RnnSequence`, `GruSequence`,
or `LstmSequence`, and later NN runtime-length work may delegate only after [backend](../../../glossary.md#backend) adoption and
the complete executable path are available.

## [Adjoint](../../../glossary.md#term-adjoint) expressibility and missing public primitives

The [task-0023 planning audit](adjoint-expressibility-audit.md) covers every current public
semantic family and [differentiable](../../../glossary.md#derivative-and-chain-rule) input role. It selects no [operation](../../../glossary.md#operation)-specific `*_BACKWARD` kind
and no `GENUINELY_NON_EXPRESSIBLE_SEMANTIC_GAP`. Exact current compositions include typed [scalar](../../../glossary.md#term-scalar)
zero/one expanded to unresolved [Shapes](../../../glossary.md#shape), [Gather Elements](../../../glossary.md#gather-elements) through [Scatter Elements](../../../glossary.md#scatter-elements) ADD, [Gather-ND](../../../glossary.md#gather-nd)
through [Scatter-ND](../../../glossary.md#scatter-nd) ADD, and maximum-pool selection recomputed through first-index arg-maximum.
Existing [producer](../../../glossary.md#term-producer) slots supply the [dropout](../../../glossary.md#dropout--inverted-dropout) keep [mask](../../../glossary.md#term-boolean-mask), [batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-training saved statistics, and top-K
indices.

The audit selected six generally useful prerequisites. Completed
[task 0023A](tasks/0023a-binding-aware-sum-to-shape.md) now implements [binding](../../../glossary.md#term-binding)-aware [sum-to-Shape](../../../glossary.md#sum-to-shape)
as an exact [target](../../../glossary.md#term-target)-Shape variant of existing `AggregateReductionKind.SUM`, with one public
`sumToShape(Shape)` expression and no new kind. It preserves exact numeric type and eligibility,
retains unresolved right-aligned target-one-or-equal obligations, and records one-input [metadata](../../../glossary.md#term-metadata)
without binding or execution. Completed
[task 0023B](tasks/0023b-gather-compatible-scatter-add.md) implements final [Gather](../../../glossary.md#gather)-compatible
fixed-add functional scatter with one public `scatterAdd` method, exact Gather-result updates
Shape, fixed duplicate accumulation, and storage-free metadata construction. Completed
[task 0023C](tasks/0023c-slice-update-and-target-relative-crop.md) implements functional signed
multi-[axis](../../../glossary.md#term-axis) `SLICE_UPDATE` plus a target-relative [`SLICE`](../../../glossary.md#slice) crop whose exact target and prefix Shapes
may remain symbolic. It adds exactly `sliceUpdate` and `cropToShape`, uses replacement rather than
overlap addition, and changes no Shape, binding, [compiler](../../../glossary.md#term-compiler), or execution contract. Completed
[task 0023D](tasks/0023d-public-fold-axis-and-dynamic-window-transforms.md) retains [canonical](../../../glossary.md#term-canonical-form)
[rank](../../../glossary.md#term-rank)-three columns, adds the missing symbolic product for two unresolved [spatial](../../../glossary.md#term-spatial-axes) factors, restores
public general-axis fold, generalizes 2D window Shapes, and adds exact typed [padding](../../../glossary.md#padding) metadata.
Completed [task 0023E](tasks/0023e-cumulative-scan-normalization-and-product.md) atomically
normalizes the existing sum-only [scan](../../../glossary.md#term-cumulative-scan) types into one `CUM_SUM`/`CUM_PROD` family and adds two
public cumulative-product expressions. It fixes integral modular multiplication, floating product
special values, multiplicative-positive-one exclusive boundaries, and zero-length-axis meaning
without executing values or selecting [gradient](../../../glossary.md#term-gradient), compiler, [runtime](../../../glossary.md#term-runtime), or [backend](../../../glossary.md#backend) behavior. Completed
[task 0023F](tasks/0023f-scaled-dot-product-attention-weights-output.md) preserves the four
existing one-output attention methods and adds explicit same-[occurrence](../../../glossary.md#term-occurrence) output-plus-normalized-
[weights](../../../glossary.md#term-weight) construction through one shared producer. The
detailed [matrix](../../../glossary.md#term-matrix-meanings) owns formulas and policy-deferred boundary cases; this capability baseline
intentionally does not duplicate them.

The audit itself changed no Java, [API](../../../glossary.md#term-application-programming-interface-api), glossary, architecture, [Gradle](../../../glossary.md#term-gradle), [dependency](../../../glossary.md#term-dependency), backend,
runtime, or execution contract. Completed tasks 0023A–0023C change [model](../../../glossary.md#term-model) semantic/expression
metadata only. Tasks 0023D and 0023E likewise change only model values and expression metadata;
completed task 0023F is bounded to that same model-owned semantic/output construction layer. Its
focused five-suite [run](../../../glossary.md#term-run) passed 40 tests, and its single final model suite passed 1,016 tests across
127 suites. The separate documentation pass validated model [Javadoc](../../../glossary.md#term-javadoc), a runnable Java 26
output/weights metadata example, the exact one-through-two-output [signature](../../../glossary.md#term-signature), two-component result,
eight attention methods, 200-method public [Tensor](../../../glossary.md#tensor) surface, one- and two-output producer forms,
Markdown, exact 27-path [scope](../../../glossary.md#term-scope), synchronized Complete/Draft status, and whitespace without
rerunning executable Java tests.

[Task 0024](tasks/0024-model-capability-and-contract-closure-audit.md) completed the final
[planning](../../../glossary.md#term-planning)-only model capability and contract audit with a `BLOCKING_GAP` verdict. Its
[result artifact](model-capability-contract-closure-audit.md)<span data-glossary-navigation="true"> ([artifact glossary](../../../glossary.md#term-artifact))</span> inventories the exact 200-method
public Tensor surface, all 107 operation-kind [constants](../../../glossary.md#term-constant) and 127 signatures, 47 accepted
attributes classes, foundations, shared outputs, selected baseline, numerical policies, cleanup,
adjoint prerequisites, documentation, and repository [checkpoint](../../../glossary.md#term-checkpoint). Its historical verdict records
that model behavior and semantic representation were coherent while `GraphValue` Javadoc still
called the already-implemented public [mutable](../../../glossary.md#term-mutable) Tensor API planned. Completed
[task 0024A](tasks/0024a-graph-value-tensor-status-javadoc-correction.md) corrected that sole
documentation blocker to current-status wording, preserved the Java declaration and behavior,
recorded the audit closure, and passed final model-Javadoc and documentation [validation](../../../glossary.md#term-validation). The
selected model capability milestone remains historically Complete. Completed task 0025 is a
focused compiler-enabling producer foundation discovered later by [ADR](../../../glossary.md#term-architecture-decision-record-adr) 0009; it does not reopen
operation semantics or the public Tensor method inventory.

Later focused compiler-prerequisite interleaves extend the current surface without rewriting that
historical 200-method audit verdict. Completed task 0025D brings the current declared public
Tensor surface to 202 methods by adding exactly two generally useful transformations:

- `sliceByLength(starts, lengths, axes, steps)` constructs a finite normalized signed extraction
  with static requested lengths even when a selected input extent is unresolved. Model proves
  non-negative coordinates and every static upper bound, defers only the unresolved selected
  extent's upper bound, canonicalizes zero-length start to zero, and reuses existing<span data-glossary-navigation="true"> ([finite glossary](../../../glossary.md#term-finite); [extent glossary](../../../glossary.md#term-extent); [coordinates glossary](../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
  `SLICE`/`SliceAttrs` Shape/[layout](../../../glossary.md#layout)/[provenance](../../../glossary.md#provenance) semantics.
- `sliceUpdate(update, prefixShape)` places the complete update region after exact per-axis prefix
  extents. It retains the exact update Shape as `CropToShapeAttrs.targetShape`, the exact prefix
  Shape, and the exact base result Shape; a fully static fit is checked locally, while any axis
  containing an unresolved base, prefix, or update extent defers its whole fit.

Both are current Model metadata construction, not value execution. They add no kind, Shape or
[Dimension](../../../glossary.md#dimension) form, constraint object, compiler [inference](../../../glossary.md#term-inference)/[preflight](../../../glossary.md#term-preflight)/gradient rule, backend behavior,
runtime state, or architecture change. Draft Compiler 0005C owns explicit adoption of their
retained obligations.

Task 0023E's focused run passed 44 tests, and its single final model suite passed 1,008 tests across
126 suites. Its separate documentation pass validated model Javadoc, Java 26 API shape, generated
Javadoc, the 196-method public Tensor surface, Markdown, exact 33-path scope, and synchronized
Complete/Draft status. Task 0023D's focused
17-suite run passed 175 tests, and its single final model suite passed 1,008 tests across 126
suites. The separate documentation pass validated model Javadoc, a runnable Java 26 metadata
example, the 194-method public Tensor surface, Markdown, exact 33-path scope, and synchronized
Complete/Draft status. Task 0023C's focused 15-suite run passed 139 tests and its single final model suite
passed 996 tests across 126 suites. Its separate documentation pass validated model Javadoc, the
runnable Java 26 metadata example, Markdown and official references, exact 27-path scope, the
192-method public Tensor surface, and synchronized status. [Fusion](../../../glossary.md#term-fusion), adjoint construction, deferred
bounds enforcement, [lowering](../../../glossary.md#lowering), and execution remain later owning concerns.

## [Validation](../../../glossary.md#term-validation) policy

Each future family task must record selected inputs and outputs, [Shape](../../../glossary.md#shape) and data-type rules,
attribute pairing, public-versus-convenience status, numerical policy decisions, invalid-input
behavior, and cross-layer follow-up. [Model](../../../glossary.md#term-model) completion means representation and public expression
construction only. Numeric completion requires later [compiler](../../../glossary.md#term-compiler), [backend](../../../glossary.md#backend)-[conformance](../../../glossary.md#term-conformance), [runtime](../../../glossary.md#term-runtime), and
integration evidence.

Legacy tests may be adapted only after the intended behavior has been independently selected.
Tests that preserve a rejected legacy quirk are evidence for the cleanup task, not acceptance
criteria for the new baseline.
