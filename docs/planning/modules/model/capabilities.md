# Model Capability Baseline

## Purpose and authority

This document selects the backend-independent tensor capabilities that Synaptik intends to expose
or represent. It is a planning baseline, not a complete tensor-library reference and not an
architecture contract.

[`ARCHITECTURE.md`](../../../../ARCHITECTURE.md) is authoritative. In particular, model owns
tensor and operation semantics, compiler owns gradient rules and backward-graph construction, and
backend prepare owns lowering, specialization, fusion, and kernel choice. If this plan conflicts
with that contract, implementation must stop rather than reinterpret the contract here.

The read-only `legacy/pre-rewrite` branch remains evidence for useful behavior and compatibility
tests. It is not the selection rule for the new API. The baseline is now chosen by semantic
coherence, usefulness for inference and training, interoperability, and the ability to specify the
behavior independently of a backend.

## Capability selection model

Every candidate belongs to one of five planning classes:

| Class | Meaning |
|---|---|
| Required baseline | Needed before Synaptik can provide a useful minimal inference or training model. |
| Important next | Valuable soon after the required baseline, but not a prerequisite for the first useful system. |
| Convenience | Public spelling composed from selected semantics; it does not require a distinct operation kind. |
| Compiler-only | Backend-independent semantics that compiler transformations may emit but public Tensor does not expose. |
| Deferred or rejected | Excluded until a concrete use case and a precise semantic contract justify it. |

An operation kind in model is not end-to-end support. Public expression construction, compiler
capture and validation, backend-neutral ownership planning, backend preparation, runtime
execution, and conformance tests remain separate completion layers.

## Foundation hardening before new operation families

The existing model can express many legacy operations, but four representation limits must be
resolved before task 0019 expands the inventory.

### Valid operations and occurrence signatures

Completed task 0018K now rejects an `OperationKind` paired with the wrong concrete
`OperationAttrs` class and gives every kind family explicit input/output occurrence cardinality.
`CompiledNode` validates those local counts after its collection checks.

The implemented direction is a small typed operation-signature and construction contract,
colocated with each kind family. It:

- reject a kind paired with the wrong attribute type when `Operation` is constructed;
- describe fixed or bounded input and output cardinality without a global registry;
- let occurrence-aware validation check arity where ordered inputs and outputs are available;
- support variadic operations such as concat and genuine multi-output operations; and
- avoid backend, lowering, gradient, or execution metadata.

This is not permission for a service locator, reflective registry, monolithic operation enum, or
schema framework. Family-specific validation remains close to the family; compiler retains
graph-wide and operand-dependent validation.

### Multi-output provenance

`CompiledNode` already permits multiple outputs, but public Tensor provenance cannot say that two
result tensors came from the same producer. Current unstack works around that by making every
result an independent `UNSTACK` operation with an output index.

True shared producer provenance is required before top-K values/indices, auxiliary normalization
statistics, graph random operations returning updated random-number-generator (RNG) state, and
other genuine multi-output compiler operations. The selected design is one immutable identity-
bearing producer with an exact operation, ordered input Tensors, and ordered output descriptors.
Each result Tensor carries the exact producer reference plus its zero-based output position.
Output count is derived from the descriptor list; the producer never retains output Tensor
objects. The same representation covers single-output expressions at output position zero. It
must remain provenance, not graph IR, and must not give Tensor a graph-local `NodeId`.

Unstack does not justify a true multi-output primitive. It will become a public convenience that
constructs independent scalar `select` expressions. Genuine multi-output operations will use the
shared producer contract.

### Dynamic shape expressions

`DynamicDimension` represents one exact named extent. Before the symbolic foundation and its
first adoption task, padding and tiling preserved such a dimension only for identity requests,
concat had a narrow zero-extent exception, and convolution or pooling could not describe ordinary
dynamic spatial results.

Completed task 0018M adds a deliberately small non-negative symbolic extent expression capable
of:

- a symbol or static constant;
- checked addition, including sums of different symbols;
- a checked signed constant offset so window formulas can represent values such as `N - 3` while
  retaining the requirement that every concrete result be non-negative;
- multiplication by a non-negative constant;
- floor or ceiling division by a positive constant; and
- a generated unknown extent with explicit constraints when an exact supported expression is not
  available.

The implemented compact form is a canonical linear combination with positive dimension
coefficients and a signed constant offset, explicit floor and ceiling division nodes, and an
identity-based unknown with a non-negative minimum plus optional inclusive dimension upper bound.
Static folding, neutral identities, repeated-term combination, commutative sum equality, and local
invalid-argument checks belong with the shape value model.

Completed follow-up [0018M1](tasks/0018m1-dynamic-extent-adoption.md) now adopts that foundation
in pad, tile, and concat. Padding retains canonical `N + before + after`, tiling retains
`repeat * N`, and concat retains canonical sums such as `N + M`; neutral operations preserve
exact Dimension references. Completed task
[0018R](tasks/0018r-slice-and-window-public-contract-cleanup.md) keeps slice-selected and
window-transformed dimensions static because literal bound clamping and current window
compatibility cannot be proved from an unbound extent; it does not add a second partial dynamic
contract. Solving graph-wide equalities, binding runtime sizes, and evaluating prepared/run
shapes remain compiler and later lifecycle concerns. Deferred compiler inference alone is
insufficient because public expression results require honest Shape metadata when they are
constructed.

### Typed scalar values

Completed task 0018N replaces ambiguous binary64 scalar-operation and padding attributes with one
small immutable `ScalarValue` for the current six data types. It preserves exact FLOAT64 and
FLOAT32 bits, raw BFLOAT16 bits, signed INT32 and INT64 values, and canonical BOOL. Scalar,
clamp, and padding attributes retain that value, and public Tensor construction requires exact
receiver/value data-type equality without implicit conversion.

`ScalarValue` is model semantic state, not a scalar Tensor, storage carrier, conversion request,
or executable constant. The existing `double` Tensor overloads remain exact-FLOAT64 conveniences,
and existing primitive TensorFactory scalar/full methods remain unchanged. Future FLOAT16 or
cross-type conversion requires a separately specified contract rather than a generic payload or
speculative numeric category.

## Existing capability decisions

The following decisions supersede legacy parity as the default outcome. Focused cleanup tasks
apply them in task order; completed task 0018O has already finalized indexing.

| Existing capability | Decision |
|---|---|
| `fastExp` and `fastTanh` public kinds | Completed task [0018P](tasks/0018p-elementwise-semantic-cleanup.md) removed both kinds and methods atomically without aliases. Approximation route choice belongs to backend prepare. A future portable approximate operation would need an explicit accuracy and special-value contract. |
| Masked sum/mean heuristic axis mapping | Completed task [0018Q](tasks/0018q-masked-reduction-redesign.md) removed the mapper and simplified masked attributes to one normalized axis. The public overloads remain first-class two-input SUM/MEAN occurrences because current primitives cannot safely compose masked-out NaN/Inf exclusion, dynamic Shapes, selected counts, and gradients without hidden eager constants or undefined behavior. Masks must use ordinary right-aligned broadcasting to produce exactly the input Shape; callers reshape or expand explicitly. All-false sum is zero and all-false mean is NaN. |
| `inv` | Completed task [0018P](tasks/0018p-elementwise-semantic-cleanup.md) atomically renamed the semantic kind and public method to `RECIPROCAL` and `reciprocal`, without a compatibility bridge. |
| `foldAxis` public method | Completed task [0018R](tasks/0018r-slice-and-window-public-contract-cleanup.md) removed the public method and helper path without an alias. `WindowTransformKind.FOLD_AXIS` and `FoldAxisAttrs` remain stable compiler-only model semantics; task 0023 owns their first compiler-generated construction. `fold2d` remains public. |
| `fromStrictFlatPrefix` and `fromCyclicFlatPrefix` | Completed task [0018S](tasks/0018s-tensor-factory-surface-cleanup.md) removed these methods and their implementation from production without aliases. Exact-carrier strict/cyclic preparation remains only in a package-private test-source fixture helper and delegates to public flat import. |
| Primitive-array `take` | Do not stabilize it. Canonical indexing accepts an index Tensor; any later primitive convenience must validate the axis before allocating its eager index Tensor. |
| Positive-step-only slicing | Completed task [0018R](tasks/0018r-slice-and-window-public-contract-cleanup.md) selected signed non-zero steps. Normalized `SliceAttrs` stores start plus selected length rather than an exclusive-end sentinel, the general arrays remain the primitive, a step-aware `sliceAxis` overload was added, and `flip(int... axes)` is one `SLICE` convenience. Positive provable views remain resolved; every negative-step result is layout-unresolved under the current non-negative-stride descriptor. |
| Large `Tensor` and `TensorFactory` surfaces | Keep Tensor as the public identity/fluent entry point, but continue to isolate implementation by cohesive helpers. Completed task [0018S](tasks/0018s-tensor-factory-surface-cleanup.md) narrowed TensorFactory to identity, construction, allocation, imports, constants, and integer ranges; promoted existing stateless `TensorRandoms` as the sole public explicit-source random owner; and moved prefix population to test source. Class size alone does not justify another facade. |

Completed task 0018P retains `EXP` and `TANH` as portable mathematical requests without promising an
algorithm, bitwise result, approximation bound, or backend route. It deliberately left the
provisional exact typed scalar vocabulary unchanged. Completed task
[0018T](tasks/0018t-scalar-arithmetic-family-normalization.md) now selects a complete parallel
`ADD`, `SUB`, `MUL`, `DIV`, `MIN`, `MAX`, and `POW` Tensor/binary and Tensor/scalar vocabulary,
retains `CLAMP` as the only distinct range kind, and removes `CLAMP_MIN`/`CLAMP_MAX` in favor of
public conveniences over scalar `MAX`/`MIN`. Pairwise public extrema use `minimum`/`maximum` while
aggregate reductions keep `min`/`max`. Completed task
[0018T1](tasks/0018t1-unary-numeric-gaps-and-floating-diagnostics.md) separately owns
floating-preserving `rsqrt`, `log1p`, and `expm1` plus fixed-BOOL floating classifications.

### Scalar arithmetic normalization

Completed task 0018T is intentionally the complete scalar arithmetic family rather than a partial
set of new overloads. Every selected Tensor-to-Tensor arithmetic relationship receives the matching
exact typed scalar form. A `ScalarValue` remains an operation attribute, not an eager rank-zero
Tensor input; the current exact receiver/value data-type check and floating-only public boundary
remain until task 0018U deliberately broadens selected integral domains.

The public vocabulary distinguishes pairwise extrema from reduction: `minimum(other-or-scalar)`
and `maximum(other-or-scalar)` select one value at each output coordinate, whereas `min(...)` and
`max(...)` aggregate a domain. The extrema semantics propagate NaN, order infinities normally,
select negative zero for minimum and positive zero for maximum. `CLAMP(x, lower, upper)` means one
first-class semantic request equivalent in value to `minimum(maximum(x, lower), upper)`; one-bound
clamp methods remain conveniences that create only scalar `MAX` or `MIN` producers.

### Unary numeric completion and floating classification

Completed task 0018T1 selects `RSQRT`, `LOG1P`, and `EXPM1` as first-class parameterless unary
functions rather than public compositions. Keeping the original semantic request lets later compiler and
backend work preserve accuracy near zero for `log1p`/`expm1` and choose an appropriate reciprocal-
square-root implementation without exposing a backend route in model. These transforms preserve
the floating input type, Shape, and gradient-eligibility request.

Floating value classification is a separate typed family because `isFinite`, `isNaN`, and
`isInf` return BOOL with false gradient eligibility. The selected public boundary accepts only
BFLOAT16, FLOAT32, and FLOAT64; integral inputs would make every answer statically trivial and are
not included without a concrete use case. Classification distinguishes all finite normal,
subnormal, and signed-zero values from both infinities and every NaN encoding. The package uses
“classification” rather than tracing-oriented “diagnostic” ownership.

The numerical function names identify portable mathematical targets and exact special-value
classes, not a machine instruction, bitwise result, restored fast variant, or fixed model-level
ULP bound. Backend conformance must establish per-data-type accuracy tolerances before execution
support is claimed.

The selected special-value classes preserve signed zero for `log1p` and `expm1`; map the two
signed zeros to same-signed infinities for `rsqrt`; map positive infinity to positive zero for
`rsqrt`; map `-1` to negative infinity for `log1p`; map negative infinity to exactly `-1` for
`expm1`; and produce NaN for the documented out-of-domain or NaN cases. This capability records
those meanings without evaluating them or fixing rounding, payload, algorithm, or tolerance.

## Indexing taxonomy

Synaptik will use three canonical semantic primitives:

- `GATHER`: replace one data axis with the complete indices Shape;
- `GATHER_ELEMENTS`: use same-rank, non-axis-aligned indices and return the indices Shape; and
- `GATHER_ND`: use final-dimension coordinate tuples with an explicit batch-dimension count.

This follows ONNX’s distinctions among [Gather](https://onnx.ai/onnx/operators/onnx__Gather.html),
[GatherElements](https://onnx.ai/onnx/operators/onnx__GatherElements.html), and
[GatherND](https://onnx.ai/onnx/operators/onnx__GatherND.html). Completed task
[0018O](tasks/0018o-indexing-taxonomy-and-unstack-normalization.md) finalized this vocabulary and
the public `gather` and `gatherElements` methods without compatibility aliases.

The removed reduced-rank gather is expressible by inserting a singleton axis into its indices,
using Gather Elements, and removing that axis. No concrete public use case retained it.

The name `take(axis, indices)` is not retained. [NumPy `take`](https://numpy.org/doc/stable/reference/generated/numpy.take.html)
and [JAX `take`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.take.html) use one indices
Shape for every slice and can flatten when no axis is supplied, while
[PyTorch `Tensor.take`](https://docs.pytorch.org/docs/2.12/generated/torch.Tensor.take.html) is
flattened.
[NumPy `take_along_axis`](https://numpy.org/doc/stable/reference/generated/numpy.take_along_axis.html),
[JAX `take_along_axis`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.take_along_axis.html),
and [PyTorch `gather`](https://docs.pytorch.org/docs/2.12/generated/torch.gather.html) describe the
aligned-indices family. Synaptik should not assign `take` a fourth meaning.

The matching public scatter primitives are `SCATTER_ELEMENTS` and `SCATTER_ND`, with explicit
reduction semantics. Specialized fixed-add gather adjoints/compositions are not current public
model kinds. Task 0023 may later define selected compiler-generated adjoint semantics. ONNX
[ScatterElements](https://onnx.ai/onnx/operators/onnx__ScatterElements.html) and
[ScatterND](https://onnx.ai/onnx/operators/onnx__ScatterND.html) provide the interoperability
vocabulary.

`select` remains the primitive for one scalar coordinate, `slice` remains the primitive for
strided ranges, and unstack becomes an ordered repeated-select convenience. Each non-empty unstack
result has an independent one-output producer and provenance output index zero; it is not a true
multi-output operation.

## Prioritized selected baseline

### Required before a useful minimal inference and training system

Foundation hardening above precedes these operations.

- exact scalar add, subtract, multiply, divide, minimum, and maximum;
- `reciprocal`, `rsqrt`, `log1p`, `expm1`, and classifications `isFinite`, `isNaN`, and `isInf`;
- signed integral add, subtract, multiply, minimum, maximum, comparisons, and reductions where an
  accumulation and overflow policy has been selected;
- `argMin` matching the explicit tie-policy structure of `argMax`;
- reductions over an ordered set of distinct axes;
- `logSumExp`, variance, standard deviation, and L1/L2 norm semantics with explicit axes,
  retained-dimension, correction, and accumulation-type policies;
- vector, matrix, and batched `matmul`;
- a public `linear` convenience over matmul plus optional bias;
- GELU and SiLU/Swish conveniences over selected primitives;
- layer normalization and RMS normalization with explicit epsilon and normalized axes;
- dense and index-target losses needed for classification, with explicit reduction and ignore
  policies;
- scaled dot-product attention as a backend-independent high-level semantic operation with mask,
  causal, and scale contracts; and
- explicit graph RNG state plus dropout as a state-consuming, state-producing operation with no
  hidden process-global generator.

`square`, scalar arithmetic overloads, `linear`, GELU, SiLU, embedding, one-hot, flatten,
swap-axes, split, and chunk are conveniences unless a focused task demonstrates a semantic reason
for a distinct kind. Layer/RMS normalization and scaled dot-product attention remain named
high-level semantics because their numerical and masking contracts must survive compiler
inspection even when a compiler can decompose them.

### Important shortly afterward

- `cumProd` after its zero, overflow, and gradient policies are specified;
- sort, argsort, and true multi-output top-K with axis, order, stability, tie, and NaN policies;
- diagonal convenience (`flip` was finalized by completed task 0018R as one `SLICE` convenience);
- embedding and one-hot conveniences;
- convolution and max/average pooling after symbolic extent expressions are complete;
- batch normalization, including explicit training/inference statistics and auxiliary outputs;
- graph random sampling operations using the explicit RNG-state contract; and
- FLOAT16 before accelerator mixed-precision inference or training is claimed.

### Explicitly deferred

- advanced linear algebra such as decompositions, eigenvalue operations, determinants, inverses,
  and general equation notation until a concrete workload selects a coherent subset;
- remainder and floor division until negative-operand, zero-divisor, and overflow semantics are
  selected;
- quantized, sparse, complex, unsigned, and distributed tensors;
- strings and arbitrary user-defined data types; and
- backend-specific or fused operation variants in model.

Quantized and sparse support may later become important project areas, but neither should distort
the dense baseline before a concrete import or execution requirement exists.

## Data type baseline

The current selected types remain FLOAT64, FLOAT32, BFLOAT16, INT32, INT64, and BOOL. FLOAT16 is
an important planned addition for accelerator mixed precision, but it is not required to begin
linear algebra and must not be implied by the current `short` BFLOAT16 carrier. Only floating
types are differentiable.

The initial arithmetic expansion is deliberately limited to signed INT32/INT64 operations with a
clear use in shape, index, or model computation. BOOL remains logical rather than numeric.
Additional integer widths and unsigned types remain deferred.

## Numerical policy gates

Every operation task must settle the applicable rows before becoming Ready. These are semantic
policies, not backend algorithm designs.

| Area | Required decisions |
|---|---|
| Floating extrema and comparisons | NaN propagation/order, signed-zero choice, infinity behavior, and equal-value ties. |
| Reductions and matmul | Accumulation data type, output data type, reassociation allowance, empty-domain result, and deterministic guarantees. |
| Integral arithmetic | Overflow behavior, division rounding, division by zero, and the `MIN_VALUE / -1` case where applicable. |
| Arg-reductions, sort, and top-K | First/last or stable tie policy, NaN placement, signed-zero ordering, empty axes, and index data type. |
| Scatter | Out-of-bounds and negative indices, duplicate-target behavior for every reduction, base-value participation, and determinism. |
| Gather and slice | Out-of-bounds and negative indices, negative-step normalization, empty extents, and deferred validation for dynamic axes. |
| Variance and standard deviation | Population/sample correction, invalid divisor, accumulation type, complex exclusion, and negative round-off handling. |
| Norms and normalization | Epsilon placement/type, zero norm, infinity, NaN, accumulation type, and empty normalized regions. |
| Transcendentals and activations | Domain, overflow/underflow, special values, accuracy contract, and discontinuity/subgradient convention. |
| Attention and losses | Mask meaning, all-masked rows, label bounds, ignored targets, reduction denominator, stability, and deterministic expectations. |
| RNG and dropout | State format, state advancement, reproducibility boundary, probability endpoints, output scaling, and multi-output contract. |

Model records the selected meaning. Compiler owns differentiability rules and backward graph
construction; backends may use different algorithms only when they satisfy that meaning.

## Compiler-generated semantic operations

The model may contain backend-independent kinds needed only after compiler transformations. They
are not automatically public Tensor methods. The selected compiler-only set includes:

- specialized gather/scatter adjoints when a composition would lose required semantics;
- retained compiler-only `FOLD_AXIS` semantics as the adjoint of single-axis unfold;
- slice and window backward operations;
- reduction-extrema, softmax, log-softmax, attention, normalization, and loss adjoints when a
  focused compiler task demonstrates that primitive composition is insufficient; and
- saved-statistic or auxiliary-output operations backed by the shared multi-output contract.

Gradient formulas and the choice to emit any of these belong to compiler. `FUSED` is never a
model semantic operation; fusion belongs to backend prepare.

## Validation policy

Each future family task must record selected inputs and outputs, Shape and data-type rules,
attribute pairing, public-versus-convenience status, numerical policy decisions, invalid-input
behavior, and cross-layer follow-up. Model completion means representation and public expression
construction only. Numeric completion requires later compiler, backend-conformance, runtime, and
integration evidence.

Legacy tests may be adapted only after the intended behavior has been independently selected.
Tests that preserve a rejected legacy quirk are evidence for the cleanup task, not acceptance
criteria for the new baseline.
