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
| `foldAxis` public method | Completed task [0018R](tasks/0018r-slice-and-window-public-contract-cleanup.md) historically removed the public method and helper path without an alias while retaining `WindowTransformKind.FOLD_AXIS` and `FoldAxisAttrs`. Completed [task 0023D](tasks/0023d-public-fold-axis-and-dynamic-window-transforms.md) now restores exactly `foldAxis(int,long,long)` as the generally useful overlap-summing public transformation. Its retained `long outputSize` is sufficient because current `unfold` requires its selected extent static. |
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

### Integral elementwise baseline

Completed task [0018U](tasks/0018u-integral-elementwise-arithmetic-and-comparisons.md) selects INT32
and INT64 ADD, SUB, MUL, MIN, and MAX for both Tensor operands and exact typed scalar attributes.
All six existing comparisons accept the same integral domain. INT32/INT64 Tensor pairs
promote to INT64 when either input is INT64; same-width pairs retain their type. Floating promotion
remains unchanged, while floating/integral pairs and BOOL require explicit different semantics or
an explicit cast.

Integral ADD, SUB, and MUL use fixed-width two's-complement modular wrap in the promoted result
type. MIN, MAX, and comparisons use signed order. Scalar values do not promote and must exactly
match the receiver type. Integral DIV, POW, range CLAMP, remainder, floor division, unsigned
values, and saturation remain outside the selected task because they require separate zero,
rounding, exponent, or overflow policy. Existing one-bound clamp conveniences inherit integral
MAX/MIN behavior without restoring a distinct clamp kind.

Completed task [0018U1](tasks/0018u1-integral-reductions-and-arg-min-normalization.md) selects
exact-type modular INT32/INT64 SUM and PROD, signed MIN and MAX, and bounded-domain empty
identities: zero for SUM, one for PROD, the type maximum for MIN, and the type minimum for MAX.
Integral reductions do not widen. The same task adds `ARG_MIN`/`argMin` and atomically replaces
arg-max-only attributes and tie-policy naming with shared arg-extrema contracts, without aliases.
Both arg families use fixed INT64 indices and explicit first/last logical-index selection. Their
floating order selects NaN over non-NaN, orders negative zero below positive zero, and orders
infinities normally. A statically empty selected axis is rejected; an unbound selected extent is
accepted structurally but must be positive when later bound. Keeping those decisions separate
prevents local elementwise promotion from being coupled to reduction-domain design. Model
construction records these contracts without evaluating values, implementing gradients, or
claiming compiler, backend, runtime, or execution support.

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
reduction semantics. The [task-0023 audit](adjoint-expressibility-audit.md) confirms that they
exactly express Gather Elements and Gather-ND adjoints with ADD duplicate accumulation. General
rank-changing Gather has a different indices/update Shape relation. A positive static gathered
extent can compose through current one-hot selection and reduction, but an unresolved gathered
extent cannot because one-hot requires positive static depth. Completed
[task 0023B](tasks/0023b-gather-compatible-scatter-add.md) therefore implements final
`SCATTER_ADD` plus one public `scatterAdd(indices, updates, axis)` expression with exact Gather
result-Shape updates and fixed duplicate accumulation, rather than a backward-only kind or a
general scatter dimension-number language.
ONNX
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
- exact GELU, fixed tanh-approximation GELU, and SiLU as first-class floating unary activations;
- layer normalization and RMS normalization with exact typed epsilon and explicit normalized
  trailing-Shape contracts;
- exact-shape mean-squared error plus dense and index-target categorical cross-entropy from logits,
  with explicit reduction, denominator, class-axis, and index-ignore policies;
- scaled dot-product attention as a backend-independent high-level semantic operation with mask,
  causal, and scale contracts; and
- explicit graph RNG state plus dropout as a state-consuming, state-producing operation with no
  hidden process-global generator.

Batch-normalization inference is also selected for the minimal inference baseline. Its model
contract is a stateless five-input occurrence with explicit data, scale, bias, running mean, and
running variance, one output, an explicit layout-neutral channel axis, and no hidden mode or
mutation. Task 0021B fixes its exact Shape, type, epsilon, formula, special-value, and provenance
contracts before any training-time statistic transition is specified.

Completed task [0018V](tasks/0018v-multi-axis-and-statistical-reductions.md) closes the selected
reduction frontier as one cohesive task. It fixes caller-ordered distinct normalized axes,
empty-axis point domains, exact structural dimension retention, population-default non-negative
integral correction, floating-only log-sum-exp/statistics/L1/L2 results, and exact input/result
types. It also completes floating and BOOL empty/special-value reduction policies while preserving
0018U1's exact-type modular integral reductions. The task records first-class semantics and
storage-free Tensor metadata; algorithms, gradients, compiler validation, backend lowering, and
execution remain separately owned.

The former broad linear-algebra/attention row is decomposed without disturbing established tasks
0019A–0019C. Completed [task 0019](tasks/0019-matmul-semantics-and-tensor-expression.md) owns only
the first-class `MATMUL` primitive and public expression. Completed
[task 0019D](tasks/0019d-linear-convenience.md) owns the `linear` convenience. Completed
[task 0019E](tasks/0019e-scaled-dot-product-attention.md) owns scaled dot-product attention. Its
kind, attrs, public construction, tests, and documentation landed as one cohesive 17-path
capability rather than an unusable semantic/construction split.

The current MATMUL expression follows rank-one promotion/removal plus right-aligned broadcasting
of leading batch axes.
It accepts same-category floating or signed-integral inputs through current promotion. A
BFLOAT16 result accumulates in FLOAT32; FLOAT32/FLOAT64 results accumulate in their promoted type;
integral results are the promoted-width modular sum of products. Floating reassociation and fused
multiply-add are permitted without a bitwise-order guarantee. Static incompatibilities fail.
Unresolved contraction equality may defer because it does not affect output Shape; unresolved
batch compatibility defers only when one exact output extent remains derivable.

`linear` is selected as `input.matmul(weight.transpose())` plus optional ADD, not a first-class
`LINEAR` operation. Completed task 0019D adds exactly `linear(weight)` and `linear(weight, bias)`.
Weight Shape is `[outFeatures, inFeatures]`, input rank is at least one with final extent
`inFeatures`, and optional bias is exactly rank one `[outFeatures]`. It retains the current
same-category floating and signed-integral MATMUL/ADD promotion domains. Bias's sole Dimension
must structurally equal the weight's `outFeatures` Dimension, so current broadcasting needs no new
constraint. Complete prevalidation precedes intermediate allocation. The no-bias form creates
PERMUTE then MATMUL wrappers and returns MATMUL provenance; the bias form additionally creates ADD
and returns ADD provenance. Transpose may preserve resolved view layout, while MATMUL and ADD
remain unresolved. The composition keeps every primitive visible to later compiler inspection and
lets backend prepare fuse a profitable pattern without placing a fused variant in model.

Scaled dot-product attention remains a named high-level operation. Query `[...batch, L, E]`, key
`[...batch, S, E]`, and value `[...batch, S, Ev]` broadcast batch/head prefixes and produce
`[..., L, Ev]`. One kind accepts ordered `[query, key, value]` or
`[query, key, value, mask]` and produces exactly one output. Its immutable attributes carry an
optional exact typed scale and causal flag; absent scale selects `1 / sqrt(E)`, while a present
scale is finite, positive, floating, and exactly matches the promoted input type. A BOOL mask
broadcasts to score Shape `[..., L, S]`, where true participates and false masks. Causal mode
additionally retains positions `j <= i`, and softmax is over the final key axis.
All-masked rows have zero weights and zero output. The initial operation has one output, exposes
no attention weights, and has no dropout parameter. Graph RNG and dropout remain owned by task
0019B; initial attention has no technical dependency on that task, and any later attention
dropout must consume its explicit state.

The selected public receiver surface is exactly four overloads: key/value with defaults,
key/value with operation-specific attrs, key/value/mask with defaults, and key/value/mask with
attrs. The public immutable attrs record is inspectable semantic state rather than a general
options framework. Query/key/value ranks are at least two; their batch/head prefixes use exact
three-way right-aligned broadcasting. Embedding equality/positivity, key/value sequence equality,
batch singleton-or-equal, and mask singleton-or-equal obligations defer only when exact output
Shape references remain derivable. Static-zero embedding is invalid; empty query, key sequence,
value-width, and batch axes follow the task's explicit empty-domain results.

Eligible score NaN propagates through a row; positive-infinity score ties split unit weight;
all-negative-infinity and all-masked rows produce positive-zero weights and output. Excluded
mask/causal positions are removed before score and value arithmetic, so their NaN/infinity values
do not contaminate the row. BFLOAT16/FLOAT32 accumulate in FLOAT32 and FLOAT64 accumulates in
FLOAT64; conforming reassociation and stable algorithms are allowed without bitwise guarantees.

`square`, scalar arithmetic overloads, `linear`, embedding, flatten, swap-axes, split, and chunk
are conveniences unless a focused task demonstrates a semantic reason for a distinct kind.
Task 0019A demonstrates that reason for GELU and SiLU: literal primitive chains encounter
infinity-times-zero at negative infinity and do not preserve the selected continuous extensions.
Exact GELU, its fixed tanh approximation, and SiLU therefore remain named unary semantics.
One-hot is also selected as a focused first-class encoding because composing it from the current
eager range factory would allocate a depth-sized host Tensor and obscure its rank-changing BOOL
contract. Layer/RMS normalization and scaled dot-product attention remain named
high-level semantics because their numerical and masking contracts must survive compiler
inspection even when a compiler can decompose them.

### Normalization frontier

The former broad task 0021 is split by formula and lifecycle boundary. Completed
[task 0021](tasks/0021-layer-normalization-semantics-and-tensor-expressions.md) owns deterministic
one-output layer normalization. Completed
[task 0021A](tasks/0021a-rms-normalization-semantics-and-tensor-expressions.md) separately owns RMS
normalization because it uses the uncentered root-mean-square formula and selects explicit
no-scale and scale-only forms. Completed
[task 0021B](tasks/0021b-batch-normalization-inference.md) owns batch-normalization inference with
mandatory explicit scale, bias, running mean, and running variance inputs and one output. Completed
[task 0021C](tasks/0021c-batch-normalization-training-and-statistic-transition.md) owns one pure
training-time transition with the same five ordered Tensor inputs, typed momentum and epsilon,
and five shared outputs: public normalized output and next running mean/variance plus hidden saved
batch mean/inverse standard deviation for later compiler autograd. No variant selects a hidden
training/evaluation flag, process-global state, mutable model service, session, or checkpoint
owner.

Completed task 0021 normalizes a non-empty trailing `Shape` with population variance (correction zero) and
epsilon inside the square root. One `LAYER_NORM` kind has exact no-affine `[input]` and affine
`[input, scale, bias]` one-output variants selected by distinct typed attrs classes, because the
current exact-class signature contract must not accept the invalid two-input count. Scale and bias
must exactly match the normalized Shape. No-affine results retain input type; affine results use
ordered floating promotion. Epsilon is a finite positive `ScalarValue` whose type exactly matches
the result. BFLOAT16/FLOAT32 accumulate in FLOAT32 and FLOAT64 in FLOAT64. Static trailing
mismatches fail; unresolved equality may defer because output Shape remains exactly the input.
Saved mean/inverse-standard-deviation outputs remain compiler concerns rather than public results.

This selected trailing-Shape/population-variance model is consistent with the official
[PyTorch LayerNorm contract](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).
Its standardization formula and trailing-axis interpretation also align with official
[ONNX LayerNormalization](https://onnx.ai/onnx/operators/onnx__LayerNormalization.html), while
Synaptik deliberately omits ONNX's optional saved-statistic outputs from this first public model
operation.

Completed task 0021A selects one `RMS_NORM` kind with one
`RmsNormAttrs(normalizedShape, epsilon)` input-range signature. Its exact valid occurrences are
one-output `[input]` and `[input, scale]`; consecutive counts one and two make one bounded
signature safe, unlike layer normalization's one/three cardinality hole. Scale Shape must exactly
equal the non-empty trailing normalized Shape, and bias is absent. The formula is
`x / sqrt(mean(x * x) + epsilon)`: it is uncentered, always divides mean square by `N`, and has no
correction option. No-scale results retain input type; scaled results use ordered floating
promotion. Epsilon is finite, positive, exact-result-typed, and inside the square root.
BFLOAT16/FLOAT32 accumulate in FLOAT32 and FLOAT64 in FLOAT64. Static trailing mismatches fail;
unresolved equality may defer because result Shape stays exactly the input. The task adds no saved
statistic, bias, hidden parameter, algorithm, gradient, compiler, backend, or execution behavior.

This narrow formula and trailing-dimension interpretation align with official
[PyTorch RMSNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html) and
[ONNX RMSNormalization](https://onnx.ai/onnx/operators/onnx__RMSNormalization.html), while
Synaptik deliberately does not copy their defaults, layer state, broadcast scale, or configurable
accumulation surface.

Completed task 0021B selects one `BATCH_NORM_INFERENCE` meaning with immutable normalized channel-axis and
exact typed epsilon attributes, exact ordered inputs
`[input, scale, bias, runningMean, runningVariance]`, and one output. Input rank is at least two;
the channel axis may identify any input axis and therefore assumes no physical NCHW or NHWC
layout. Every affine/statistic operand is an exact rank-one per-channel vector. Static channel
mismatches fail locally, while equality involving unresolved Dimensions may defer because output
Shape remains exactly the input Shape. All five operands are mandatory and use ordered floating
promotion. Epsilon is finite, positive, exact-result-typed, and added to running variance inside
the denominator square root. The supplied running variance is used directly as an estimated
per-channel variance: inference applies no correction conversion, recomputation, clamp, momentum,
or update. Negative values are not read or rejected during model construction and follow the
formula's floating square-root behavior. There is one fresh output at provenance index zero and
no saved statistic, training flag, optional affine form, hidden constant, or mutable state.

This five-input inference formula and explicit estimated-statistic roles align with official
[ONNX BatchNormalization](https://onnx.ai/onnx/operators/onnx__BatchNormalization.html), while
Synaptik deliberately makes the channel axis explicit, requires exact typed epsilon and exact
rank-one parameter/statistic Shapes, and separates inference from every training/state-transition
contract.

Completed task 0021C selects batch mean and biased population variance across every non-channel
axis for normalization, while the explicit next running variance uses the corresponding unbiased
estimate.
Its momentum is the new-batch weight: `next = (1 - momentum) * old + momentum * batch`. A valid
non-empty channel therefore needs reduction count at least two; a zero-channel occurrence is
empty and evaluates no statistic. All five outputs share one producer in the order normalized
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

### Loss frontier

The former broad task 0022 is split at target representation and denominator boundaries while
preserving established task IDs 0023–0024. Completed
[task 0022](tasks/0022-mean-squared-error-loss.md) established `NONE`, `SUM`, and `MEAN` as
explicit loss reductions through one exact-shape
mean-squared-error operation. Completed
[task 0022A](tasks/0022a-dense-target-categorical-cross-entropy-with-logits.md) adds dense
floating categorical cross-entropy directly from logits with an explicit class axis. Completed
[task 0022B](tasks/0022b-index-target-categorical-cross-entropy-with-logits.md) adds INT32/INT64
index-target categorical cross-entropy from logits, class-axis removal, an optional exact typed
ignore index, and a `MEAN` denominator equal to the non-ignored target count while preserving the
completed floating-target dispatch.

Mean-squared error is the intentionally small regression baseline. Prediction and target Shapes
must match positionally; the operation does not broadcast targets. `NONE` preserves the exact
prediction Shape, while `SUM` and `MEAN` produce a scalar. The mean denominator is the complete
logical element count, including scalar count one; an empty sum is positive zero and an empty mean
is NaN. This explicit three-mode vocabulary is comparable to official
[PyTorch MSE loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html),
but Synaptik deliberately omits defaults, weighting, and broadcast behavior from the selected
model contract.

The categorical baseline consumes logits rather than probabilities so that stable log-softmax
semantics remain one inspectable operation request. Dense targets have the exact logits Shape and
no ignore index. Index targets have the logits Shape with exactly the class axis removed; ignored
positions contribute positive zero to `NONE`/`SUM` and do not enter the `MEAN` denominator. Both
families accept rank-at-least-one floating logits and an explicit arbitrary class axis. Dense
targets are floating and promoted with logits; index targets are exact INT32 or INT64. Dense
`NONE` produces the logits Shape with the class axis removed; index `NONE` retains the exact target
Shape constrained to that non-class domain. `SUM` and `MEAN` are scalar. Dense `MEAN` divides by
the number of class-axis-removed groups, not by
the class count or target-value sum. Dense target values carry a finite, non-negative,
class-normalized caller obligation, but model construction will read no values, reject none, and
does not renormalize supplied weights. Its one first-class loss uses target-weighted stable
log-softmax directly from logits, including a positive-zero contribution for an exact zero target
weight. Index `MEAN` divides by the non-ignored group count; an all-ignored or otherwise empty
mean will be NaN, while the matching sum will be positive zero. Without ignore, a non-empty group
domain requires a positive class extent. With ignore, zero class extent is also valid for a
non-empty all-ignored domain; every non-matching target remains invalid because no class index is
in range. Value-dependent and unresolved cases are deferred because model construction reads no
target values. These target
alternatives and reductions are comparable to official
[PyTorch cross entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy)
and [ONNX SoftmaxCrossEntropyLoss](https://onnx.ai/onnx/operators/onnx__SoftmaxCrossEntropyLoss.html),
while Synaptik deliberately uses an explicit arbitrary class axis and separate typed operation
families. Standalone probability-input cross entropy, standalone negative-log-likelihood over
caller-supplied log probabilities, class/sample weights, masks, label smoothing, binary cross
entropy, margin/ranking losses, Kullback–Leibler divergence, connectionist temporal classification,
and a broad loss registry/options framework are not selected for this minimal frontier.

The former 0019A umbrella is split by semantic boundary. Completed task 0019A owns `gelu()`,
`geluTanhApproximation()`, and canonical `silu()` without a `swish` alias. `GELU` uses
`x * Phi(x)`; `GELU_TANH_APPROXIMATION` uses the fixed conventional coefficient `0.044715`; and
`SILU` uses `x * sigmoid(x)`. All three are parameterless, floating-only, same-Shape operations
whose mathematical continuous extensions map negative infinity to negative zero, preserve signed
zero, map positive infinity to positive infinity, and propagate NaN. They preserve type and
gradient-eligibility metadata with unresolved layout, without defining algorithms or gradients.
ReLU is already current through completed tasks 0014C–0014D and is not duplicated by 0019A.
The former broad dropout/RNG frontier is split at the reusable semantic boundary. Completed
[task 0019B](tasks/0019b-explicit-graph-rng-state-foundation.md) owns only explicit graph RNG
state. Completed [task 0019B1](tasks/0019b1-explicit-graph-dropout-construction.md) owns dropout because
it consumes and produces that state rather than behaving as a deterministic unary activation.

Completed [task 0019A1](tasks/0019a1-embedding-convenience.md) adds exactly
`weights.embedding(indices)`. The receiver is a rank-two floating
table `[vocabulary, embeddingDimension]`; indices are INT32 or INT64; the result Shape is the
complete indices Shape followed by the exact embedding dimension. It is direct
axis-zero Gather composition with no EMBEDDING kind, intermediate producer, or
padding/sparse/max-norm/frequency option. Result
gradient eligibility comes only from weights as inherited from Gather; compiler autograd owns any
scatter-add rule. Index values must be non-negative and below the table's axis-zero extent when
executed. Construction reads no values; compiler validation may reject captured constant indices,
while backend preparation and its prepared executable must preserve safe execution-time bounds
handling after dynamic extents are bound. Runtime executes that prepared behavior without
inspecting the original Gather operation.

Completed [task 0019A2](tasks/0019a2-one-hot-encoding.md) adds exactly
`indices.oneHot(long depth)`. The receiver is INT32 or INT64,
`depth` is a positive static `long`, and one static depth axis is appended after every input axis.
Every existing indices Dimension is retained exactly, including dynamic dimensions.
The output is non-differentiable BOOL with `false` off and `true` at the matching coordinate.
Negative indices and indices at least `depth` are invalid at execution; they do not wrap, clamp,
select a default, or produce an all-false row. Construction reads no index values, so later
compiler/backend/execution work must enforce that boundary safely. No axis, output-type,
on/off-value, ignore-index, sparse, or dynamic-depth configuration is selected. It is a
first-class one-input operation rather than TensorFactory construction or an eager
range/comparison composition. The implemented contract is `OneHotKind.ONE_HOT` with
`OneHotAttrs(long depth)` and an exact one-input, one-output signature.

### Explicit graph RNG state and dropout

Completed [task 0019B](tasks/0019b-explicit-graph-rng-state-foundation.md) implements one opaque public
`GraphRngState` backed privately by a storage-free Tensor expression. Callers create a state with
explicit key and counter words. Both Java `long` values are interpreted as unsigned 64-bit bit
patterns: the key selects a caller-owned stream/domain and the counter is the next abstract
logical sample position. Every bit pattern is valid. Equal key/counter attributes request the
same abstract position but separately constructed states retain distinct Tensor and producer
identities.

`GraphRngKind.INITIAL_STATE` is a zero-input, one-output operation with immutable
`GraphRngStateAttrs(long key, long counter)`. Its output is an opaque `INT64 Shape[2]` Tensor with
unresolved layout, false gradient eligibility, no label, no storage, and provenance output index
zero. The two lanes are raw state words, not signed numerical values or a public arithmetic API.
State-consuming operations keep the key and advance the counter modulo `2^64` by their exact
logical draw count. Branching one state intentionally reuses a counter interval; sequential use
threads the returned next state.

The state format, operation ordering, and advancement counts are portable model semantics, but
no random algorithm, key schedule, counter-to-bits function, floating conversion, or cross-backend
bitstream is selected. Until a portable algorithm is deliberately specified, deterministic replay
of sampled values is bounded to the same conforming prepared implementation and configuration.
Future graph serialization must preserve the two raw words losslessly, but no byte encoding or
stable serialization token is current.

Completed [task 0019B1](tasks/0019b1-explicit-graph-dropout-construction.md) adds training dropout as
`input.dropout(double probability, GraphRngState state)`. Its public `DropoutResult` exposes the
dropped Tensor and next state. One producer consumes ordered `[input, state]` and produces ordered
`[output, auxiliaryMask, nextState]`; the same-Shape BOOL mask is a non-public auxiliary slot for
compiler-owned backward construction. Dropout is floating-only, uses finite drop probability in
`[0, 1)`, scales kept values by `1 / (1 - probability)`, consumes one abstract draw per logical
element even at probability zero, and advances dynamic Shapes by their bound execution count.
Empty tensors consume zero draws. Inference bypasses the operation and state advancement rather
than using a model-level training flag.

The current multi-output factory creates one indexed Tensor wrapper per output descriptor, so one
dropout occurrence allocates exactly three wrappers and three Tensor IDs: public output slot zero,
short-lived non-public mask slot one, and next-state slot two. The public output and wrapped state
retain the shared producer and therefore keep the auxiliary BOOL descriptor/position available to
later compiler capture without exposing a public mask. No descriptor-only factory output or live
sibling-result registry is selected.

This graph contract is distinct from `TensorRandoms`. `TensorRandoms` eagerly consumes a
caller-owned JDK `RandomGenerator` to create host-backed leaf data. `GraphRngState` records
storage-free semantics for later graph execution. Neither owns or discovers a hidden generator.

### Sorting and top-K frontier

The former broad sorting row is split at the output/Shape boundary. Completed
[task 0019C](tasks/0019c-sort-and-argsort.md) owns stable full-axis `sort` and `argsort` as distinct
single-output semantics. Completed [task 0019C1](tasks/0019c1-top-k-values-and-indices.md) owns genuine
top-K because it replaces one Shape extent
with `k`, validates static or deferred selected-axis capacity, and returns values and indices from
one shared two-output producer. Its detailed specification follows completed task 0019C;
established tasks 0019D and 0019E retain their IDs.

Task 0019C selects exactly `sort(axis)`, `sort(axis, descending)`, `argsort(axis)`, and
`argsort(axis, descending)`. `OrderingKind.SORT` returns values only and
`OrderingKind.ARGSORT` returns fixed-INT64 indices only; both use normalized
`SortAttrs(axis, descending)`, one input, and one output. Calling both methods creates distinct
producers rather than a shared multi-output sort result. Every current DataType is accepted:
floating and signed-integral values use numerical order, while BOOL uses `false < true`. Sort
preserves exact input type, Shape reference, and gradient-eligibility request; argsort preserves
the exact Shape reference but is non-differentiable INT64. Both leave layout unresolved.

Full sorting is always stable: equal values retain increasing logical input-coordinate order, so
argsort indices are deterministic and independent of physical layout or backend traversal. NaNs
form one final class after every non-NaN in both ascending and descending results; multiple NaNs
are stable and sort retains their exact input representations. Negative zero precedes positive
zero ascending and follows it descending, while infinities use ordinary numerical order.
Duplicates remain present. A static empty selected axis is valid, a dynamic extent is accepted
without binding because Shape is unchanged, and scalar input rejects every axis. Stability and
NaN placement are fixed semantics rather than attributes or backend choices. This selected API
uses the mainstream `sort`/`argsort`, axis, and descending vocabulary documented by
[NumPy](https://numpy.org/doc/stable/reference/generated/numpy.sort.html) and
[JAX](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sort.html) without importing their
algorithm-selection options.

Completed [task 0019C1](tasks/0019c1-top-k-values-and-indices.md) adds exactly
`topK(long k, int axis)` and
`topK(long k, int axis, boolean largest, boolean sorted)`, with largest/sorted defaults, plus
`TopKResult(values, indices)`. Focused `TopKKind.TOP_K` preserves `OrderingKind`'s exact existing
SORT/ARGSORT signatures. One TOP_K producer consumes the input and describes ordered values
and indices slots zero and one, so the carrier retains the exact two wrappers and exactly two IDs
under the current factory seam. The `largest`/`sorted` names follow the conventional
[PyTorch top-k](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.topk.html) surface.
`TopKAttrs(axis, k, largest, sorted)` retains normalized axis and
non-negative static `k`. All current input types remain eligible. The selected set is the first
`k` items under the requested non-NaN order with NaNs last, so NaNs participate only when fewer
than `k` non-NaNs exist; equal boundary candidates prefer lower logical indices. `k == 0` is
valid; static `k > extent` fails; a dynamic selected extent defers `extent >= k` until binding;
scalar input remains invalid. A static empty axis therefore accepts only zero. The output Shape
replaces the selected Dimension with static `k`. Sorted output follows requested value order;
unsorted output uses increasing original logical index so it remains deterministic. Values retain
input type/gradient eligibility, and indices are INT64/non-gradient; both are unresolved,
unlabeled, and storage-free. Neither task implements evaluation, gradients, compiler capture,
lowering, kernels, or execution.

Top-K local validation checks input, normalizes axis, requires non-negative `k`, then rejects a
known static `k > extent`, before wrapper/ID allocation. Its exact failures are
`k must be non-negative: <k>` and
`k must not exceed selected static extent: k=<k>, axis=<axis>, extent=<extent>`. A dynamic extent
records the deferred inequality instead of choosing a backend-dependent result.

### Convolution and pooling frontier

Completed task 0020 adds grouped NCHW two-dimensional cross-correlation with weight-derived
static kernels, optional exact output-channel bias, ordered floating promotion, selected
accumulation/special-value meaning, and floor-mode static or symbolic spatial results. For dynamic
spatial extent `D`, symmetric padding `p`, dilation `d`, static kernel `k`, and stride `s`, current
construction retains the canonical equivalent of
`floor((D + 2p - (d * (k - 1) + 1)) / s) + 1`; static invalid geometry fails locally and unresolved
validity remains a compiler/binding obligation. Exact descriptors and attributes likewise retain
unresolved grouped-channel and bias-channel obligations. Construction records one fresh
storage-free result and exact two- or three-input provenance without evaluation, gradients,
compiler capture, algorithms, backend support, or execution.

Completed task [0020A](tasks/0020a-nchw-max-pool2d-semantics-and-tensor-expression.md) owns floating
NCHW max pooling. One `MAX_POOL2D` occurrence preserves type, batch, channel, and gradient-request
metadata while deriving floor or ceil static/symbolic spatial extents. Ceil mode uses the literal
symmetric padded grid and does not drop a terminal window that starts in trailing padding.
Padding samples are excluded; an all-padding window returns negative infinity. NaNs dominate,
positive zero is greater than negative zero, and equal candidates select the first height-major
kernel sample. Existing `Window2dAttrs` remains specific to unfold/fold and conceptual zero
padding, so max pooling owns `MaxPool2dAttrs` in the pooling package.

Completed task [0020A1](tasks/0020a1-nchw-average-pool2d-semantics-and-tensor-expression.md)
separately owns floating NCHW average pooling through exactly
`averagePool2d(AveragePool2dAttrs)`. It extends `Pool2dKind` without weakening max pooling and uses
a dedicated attributes type with the same literal floor/ceiling padded grid. Every logical kernel
position counts: in-bounds values contribute to the numerator, padding contributes conceptual
positive zero, and the fixed divisor is the positive mathematical product of the two kernel sample
counts. Count-padding is not configurable and there is no divisor override or valid-sample mode.
An all-padding window returns positive zero; an exact-zero finite mean is negative zero only when
every divisor contribution is an in-bounds negative zero. BFLOAT16/FLOAT32 accumulate and divide
in FLOAT32, while FLOAT64 uses FLOAT64; result type remains the exact input type. NaN propagates,
opposing infinities produce NaN, and a single infinity sign is retained. Conforming summation
reassociation is allowed without bitwise or cross-backend rounding identity. Combining the two
pooling operations would exceed the cohesive path guardrail; sharing NCHW coordinates does not
make their numerical parameters interchangeable.

### Important shortly afterward

- diagonal convenience (`flip` was finalized by completed task 0018R as one `SLICE` convenience);
- embedding convenience and focused one-hot encoding semantics;
- average pooling with its separate divisor contract after completed NCHW max pooling;
- batch normalization split into explicit one-output inference and a later training/state-
  transition occurrence with running-stat and saved-stat outputs;
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
| RNG and dropout | Completed task 0019B fixes two unsigned 64-bit state words, modular counter meaning, and bounded replay without a bitstream; completed task 0019B1 fixes finite drop probability `[0,1)`, inverted scaling, one draw per element, auxiliary mask, three wrapper/ID outputs, and next-state output. |

Model records the selected meaning. Compiler owns differentiability rules and backward graph
construction; backends may use different algorithms only when they satisfy that meaning.

## Adjoint expressibility and missing public primitives

The [task-0023 planning audit](adjoint-expressibility-audit.md) covers every current public
semantic family and differentiable input role. It selects no operation-specific `*_BACKWARD` kind
and no `GENUINELY_NON_EXPRESSIBLE_SEMANTIC_GAP`. Exact current compositions include typed scalar
zero/one expanded to unresolved Shapes, Gather Elements through Scatter Elements ADD, Gather-ND
through Scatter-ND ADD, and maximum-pool selection recomputed through first-index arg-maximum.
Existing producer slots supply the dropout keep mask, batch-training saved statistics, and top-K
indices.

The audit selected six generally useful prerequisites. Completed
[task 0023A](tasks/0023a-binding-aware-sum-to-shape.md) now implements binding-aware sum-to-Shape
as an exact target-Shape variant of existing `AggregateReductionKind.SUM`, with one public
`sumToShape(Shape)` expression and no new kind. It preserves exact numeric type and eligibility,
retains unresolved right-aligned target-one-or-equal obligations, and records one-input metadata
without binding or execution. Completed
[task 0023B](tasks/0023b-gather-compatible-scatter-add.md) implements final Gather-compatible
fixed-add functional scatter with one public `scatterAdd` method, exact Gather-result updates
Shape, fixed duplicate accumulation, and storage-free metadata construction. Completed
[task 0023C](tasks/0023c-slice-update-and-target-relative-crop.md) implements functional signed
multi-axis `SLICE_UPDATE` plus a target-relative `SLICE` crop whose exact target and prefix Shapes
may remain symbolic. It adds exactly `sliceUpdate` and `cropToShape`, uses replacement rather than
overlap addition, and changes no Shape, binding, compiler, or execution contract. Completed
[task 0023D](tasks/0023d-public-fold-axis-and-dynamic-window-transforms.md) retains canonical
rank-three columns, adds the missing symbolic product for two unresolved spatial factors, restores
public general-axis fold, generalizes 2D window Shapes, and adds exact typed padding metadata.
Completed [task 0023E](tasks/0023e-cumulative-scan-normalization-and-product.md) atomically
normalizes the existing sum-only scan types into one `CUM_SUM`/`CUM_PROD` family and adds two
public cumulative-product expressions. It fixes integral modular multiplication, floating product
special values, multiplicative-positive-one exclusive boundaries, and zero-length-axis meaning
without executing values or selecting gradient, compiler, runtime, or backend behavior. Draft task
0023F retains same-occurrence attention weights. The
detailed matrix owns formulas and policy-deferred boundary cases; this capability baseline
intentionally does not duplicate them.

The audit itself changed no Java, API, glossary, architecture, Gradle, dependency, backend,
runtime, or execution contract. Completed tasks 0023A–0023C change model semantic/expression
metadata only. Tasks 0023D and 0023E likewise change only model values and expression metadata.
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
192-method public Tensor surface, and synchronized status. Fusion, adjoint construction, deferred
bounds enforcement, lowering, and execution remain later owning concerns.

## Validation policy

Each future family task must record selected inputs and outputs, Shape and data-type rules,
attribute pairing, public-versus-convenience status, numerical policy decisions, invalid-input
behavior, and cross-layer follow-up. Model completion means representation and public expression
construction only. Numeric completion requires later compiler, backend-conformance, runtime, and
integration evidence.

Legacy tests may be adapted only after the intended behavior has been independently selected.
Tests that preserve a rejected legacy quirk are evidence for the cleanup task, not acceptance
criteria for the new baseline.
