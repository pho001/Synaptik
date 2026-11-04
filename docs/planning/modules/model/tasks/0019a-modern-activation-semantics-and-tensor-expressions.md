# Task 0019A: Modern [Activation](../../../../glossary.md#term-activation) Semantics and [Tensor](../../../../glossary.md#tensor) Expressions

**Editorial reading key (current context, not a status update).**

This completed record covers Model expression construction, not activation execution. Unary means
one input; elementwise means positions are transformed independently. The caller selects exact GELU,
fixed tanh-approximation GELU, or SiLU through a Tensor method; `TensorUnaryExpressions.apply`
validates metadata and delegates derived construction to `TensorFactory`. Descriptors describe type,
Shape and gradient eligibility; provenance records the selected operation and exact receiver. No
element values cross this construction boundary. For example, `input.gelu()` records one GELU
producer, not computed activation values; its fresh, storage-free result owns no result buffer.

The historical three-type scope and deferred numerical details remain historical. Follow the current
[unary
reference](../../../../reference/tensor/elementwise.md#unary-numeric-transforms-and-floating-classifications)
for expanded floating-type coverage and Model-owned primitive-site numerical rules. Model defines
meaning; execution providers realize supported routes rather than redefine it, and semantic
responsibility is not storage ownership. Exact and tanh GELU remain distinct targets. Negative
infinity maps to negative zero, signed zero is preserved, and NaN remains NaN—not a promise that
literal primitive composition preserves these boundaries.

## Status

Complete

## Goal

Add exact [Gaussian error linear unit](../../../../glossary.md#gaussian-error-linear-unit--gelu) ([GELU](../../../../glossary.md#gaussian-error-linear-unit--gelu)), the conventional [tanh](../../../../glossary.md#term-tanh) GELU approximation, and
[sigmoid linear unit](../../../../glossary.md#sigmoid-linear-unit--silu) ([SiLU](../../../../glossary.md#sigmoid-linear-unit--silu)) as one cohesive [backend](../../../../glossary.md#backend)-independent [unary](../../../../glossary.md#term-unary) [activation](../../../../glossary.md#term-activation) capability. The
task introduces three parameterless first-class semantic kinds and three zero-[argument](../../../../glossary.md#term-argument) public
[`Tensor`](../../../../glossary.md#tensor) methods while preserving the current floating unary [descriptor](../../../../glossary.md#term-descriptor), [producer](../../../../glossary.md#term-producer), [provenance](../../../../glossary.md#provenance),
and failure contracts.

This task records storage-free [model](../../../../glossary.md#term-model) [metadata](../../../../glossary.md#term-metadata) only. It does not evaluate activations, choose an
algorithm, define an accuracy [tolerance](../../../../glossary.md#term-tolerance) or [gradient](../../../../glossary.md#term-gradient) rule, [capture](../../../../glossary.md#term-capture-meanings) a [graph](../../../../glossary.md#graph), lower an [operation](../../../../glossary.md#operation), or
claim [compiler](../../../../glossary.md#term-compiler), backend, [runtime](../../../../glossary.md#term-runtime), or end-to-end support.

## Why the former 0019A frontier is split

The former row combined three independently reviewable capability boundaries. [GELU](../../../../glossary.md#gaussian-error-linear-unit--gelu) and [SiLU](../../../../glossary.md#sigmoid-linear-unit--silu) are
same-[shape](../../../../glossary.md#shape) floating [unary](../../../../glossary.md#term-unary) [activations](../../../../glossary.md#term-activation) and fit the existing unary family. [Embedding](../../../../glossary.md#embedding) is a [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)
table [lookup](../../../../glossary.md#term-lookup) that can be expressed by [canonical](../../../../glossary.md#term-canonical-form) [axis](../../../../glossary.md#term-axis)-zero [gather](../../../../glossary.md#gather). [One-hot encoding](../../../../glossary.md#one-hot-encoding) instead changes
[rank](../../../../glossary.md#term-rank) and result type and cannot use the unary construction contract. Combining their kinds,
public APIs, [validation](../../../../glossary.md#term-validation), tests, and documentation would exceed normal task granularity and obscure
three different [ownership](../../../../glossary.md#term-ownership-meanings) decisions.

Task 0019A therefore owns only modern activations. Draft task 0019A1 owns embedding, and Draft task
0019A2 owns one-hot. Existing Draft tasks 0019B through 0019E retain their identifiers, meanings,
and order after these inserted follow-ups.

## [Mental model](../../../../glossary.md#term-mental-model) and examples

All three [activations](../../../../glossary.md#term-activation) transform each floating input [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) independently and retain the input
[Shape](../../../../glossary.md#shape):

```text
GELU exact:       x * Phi(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
GELU tanh target: 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
SiLU:             x * sigmoid(x) = x / (1 + exp(-x))
```

For conceptual [finite](../../../../glossary.md#term-finite) inputs:

```text
input Shape [3], FLOAT32: [-1.0, 0.0, 1.0]
gelu() target:             about [-0.158655, 0.0, 0.841345]
silu() target:             about [-0.268941, 0.0, 0.731059]
```

The decimal values explain mathematical meaning only. [Model](../../../../glossary.md#term-model) construction neither reads those
inputs nor computes the results, and this task does not select rounded expected bits or an
execution [tolerance](../../../../glossary.md#term-tolerance).

## [Scope](../../../../glossary.md#term-scope)

- Add exactly [`GELU`](../../../../glossary.md#gaussian-error-linear-unit--gelu), `GELU_TANH_APPROXIMATION`, and `SILU` to `UnaryElementwiseKind`.
- Keep all three parameterless with the existing exact [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), one-input, one-output
  unary signature.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [signature glossary](../../../../glossary.md#term-signature))</span>
- Add exactly `Tensor.gelu()`, `Tensor.geluTanhApproximation()`, and `Tensor.silu()`.
- Reuse `TensorUnaryExpressions.apply` unchanged in behavior, expanding only its documented kind
  count.
- Accept only BFLOAT16, FLOAT32, and FLOAT64 and preserve exact input type, Shape reference, and
  gradient-eligibility request.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [Shape glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Fix the mathematical and special-value policies below without prescribing an implementation.
- Construct one fresh unresolved-layout result with exact one-input provenance, no label, and no
  storage for every valid call.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance))</span>
- Extend the focused unary semantic and public-expression tests plus the exact public API
  inventories.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Finalize Javadocs, Tensor/Compile API references, glossary, capabilities, and planning records
  through a mandatory separate clean-context documentation pass.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- embedding, padding indices, one-hot, dropout, graph random-number-generator state, sorting,
  top-K, linear, attention, normalization, losses, convolution, or pooling<span data-glossary-navigation="true"> ([embedding glossary](../../../../glossary.md#embedding); [padding glossary](../../../../glossary.md#padding); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [graph glossary](../../../../glossary.md#graph); [linear glossary](../../../../glossary.md#term-linear-meanings); [normalization glossary](../../../../glossary.md#term-normalization); [losses glossary](../../../../glossary.md#loss); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [pooling glossary](../../../../glossary.md#term-pooling))</span>
- ReLU, sigmoid, tanh, or any other already implemented unary method; completed tasks
  0014C–0014D own the current <span data-glossary-navigation="true"> ([ReLU glossary](../../../../glossary.md#term-relu); [unary glossary](../../../../glossary.md#term-unary))</span>`RELU`/`relu()` contract, which this task must not duplicate or alter
- a `swish` alias, an approximation enum or attribute, string-valued modes, configurable GELU
  coefficients, SiLU beta, or additional activation variants<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [GELU glossary](../../../../glossary.md#gaussian-error-linear-unit--gelu); [SiLU glossary](../../../../glossary.md#sigmoid-linear-unit--silu); [activation glossary](../../../../glossary.md#term-activation))</span>
- composition into [ERF](../../../../glossary.md#term-erf), [TANH](../../../../glossary.md#term-tanh), [SIGMOID](../../../../glossary.md#term-sigmoid), [scalar](../../../../glossary.md#term-scalar), [binary](../../../../glossary.md#term-binary-descriptor-and-metadata), or WHERE [producer](../../../../glossary.md#term-producer) chains
- eager value reads, host-storage allocation, result storage, resolved result layout, mutation,
  or numerical evaluation<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [host glossary](../../../../glossary.md#term-host); [layout glossary](../../../../glossary.md#layout); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- a fixed unit-in-the-last-place bound, relative-error tolerance, rounding mode, NaN payload,
  machine instruction, polynomial/rational evaluation scheme, or backend route<span data-glossary-navigation="true"> ([tolerance glossary](../../../../glossary.md#term-tolerance); [rounding glossary](../../../../glossary.md#term-rounding); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [backend route glossary](../../../../glossary.md#backend-route))</span>
- compiler capture, canonicalization, decomposition, constant folding, fusion, autograd, gradient
  formulas, subgradient conventions, or backward operations<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [decomposition glossary](../../../../glossary.md#term-decomposition); [constant folding glossary](../../../../glossary.md#term-constant-folding); [fusion glossary](../../../../glossary.md#term-fusion); [autograd glossary](../../../../glossary.md#autograd); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- capability providers, backend lowering, kernels, runtime execution, conformance, integration,
  or end-to-end support<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel); [runtime glossary](../../../../glossary.md#term-runtime); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- changes to [operation](../../../../glossary.md#operation) attributes/[signatures](../../../../glossary.md#term-signature), [`DataType`](../../../../glossary.md#data-type--datatype), Shape/layout/descriptor,
  producer/provenance/factory foundations, Gradle, another module, <span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>`ARCHITECTURE.md`, or focused
  architecture documentation

## Exact semantic and [API](../../../../glossary.md#term-application-programming-interface-api) contract

### Kinds, attributes, [signatures](../../../../glossary.md#term-signature), and public methods

Extend the existing enum with exactly:

```java
GELU,
GELU_TANH_APPROXIMATION,
SILU
```

Every kind continues to use the family's one stable signature:

```java
OperationSignature.fixed(NoOperationAttrs.class, 1, 1)
```

Every [operation](../../../../glossary.md#operation) uses exactly `NoOperationAttrs.INSTANCE`. The [tanh](../../../../glossary.md#term-tanh) approximation is a separate
semantic kind, not an attribute variant of [`GELU`](../../../../glossary.md#gaussian-error-linear-unit--gelu), because it names a different portable function
with different [finite](../../../../glossary.md#term-finite) results and no tunable state. Keeping it typed lets [compiler](../../../../glossary.md#term-compiler) and [backend](../../../../glossary.md#backend)
work preserve the caller's selected [target](../../../../glossary.md#term-target) without string inspection. Add no [activation](../../../../glossary.md#term-activation)-specific
attribute type or signature variant.

Add exactly these public methods:

```java
public Tensor gelu()
public Tensor geluTanhApproximation()
public Tensor silu()
```

`gelu()` selects the exact error-function definition. `geluTanhApproximation()` selects the fixed
formula above; it is not permission for a backend to substitute an arbitrary fast GELU.
`silu()` is the sole public spelling. Do not add `swish`, because two names for the same exact
parameterless semantic would add [API](../../../../glossary.md#term-application-programming-interface-api) surface without adding meaning or compatibility value.

### First-class rather than composed semantics

All three are first-class [unary](../../../../glossary.md#term-unary) kinds. Literal expression composition is not semantically
equivalent at important special values: straightforward `x * sigmoid(x)` and
`x * (1 + erf(...))` chains encounter `infinity * zero` at [negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero) and may produce [NaN](../../../../glossary.md#term-nan)
instead of the selected limiting [signed zero](../../../../glossary.md#term-signed-zero). Composition would also hide the caller's exact
versus [tanh](../../../../glossary.md#term-tanh) [GELU](../../../../glossary.md#gaussian-error-linear-unit--gelu) choice behind a larger [producer](../../../../glossary.md#term-producer) chain and constrain later accuracy-preserving
[compiler](../../../../glossary.md#term-compiler)/[backend](../../../../glossary.md#backend) treatment. First-class meaning fixes the mathematical [target](../../../../glossary.md#term-target) while leaving
[decomposition](../../../../glossary.md#term-decomposition), [fusion](../../../../glossary.md#term-fusion), and algorithm choice to compiler and backend [owners](../../../../glossary.md#term-owner).

### [Data type](../../../../glossary.md#data-type--datatype), [Shape](../../../../glossary.md#shape), [layout](../../../../glossary.md#layout), and [gradient](../../../../glossary.md#term-gradient) eligibility

Each method accepts exactly [BFLOAT16](../../../../glossary.md#term-bfloat16), [FLOAT32](../../../../glossary.md#term-float32), or [FLOAT64](../../../../glossary.md#term-float64). [INT32](../../../../glossary.md#data-type--datatype), [INT64](../../../../glossary.md#data-type--datatype), and [BOOL](../../../../glossary.md#data-type--datatype) fail through the
existing [unary](../../../../glossary.md#term-unary) helper message; no implicit cast is inserted.

Every result retains:

- the exact input [`DataType`](../../../../glossary.md#data-type--datatype);
- the exact [immutable](../../../../glossary.md#term-immutable) input `Shape` reference, including scalar, zero-sized, named-symbolic, and
  expression dimensions; and<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [dimensions glossary](../../../../glossary.md#dimension))</span>
- the exact input `requiresGrad` value as [model](../../../../glossary.md#term-model)-level gradient eligibility only.

Every result uses `Optional.empty()` layout even when the input layout is resolved. Construction
does not claim a [derivative](../../../../glossary.md#term-derivative), differentiability at every [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), a subgradient convention, or
[compiler](../../../../glossary.md#term-compiler) [autograd](../../../../glossary.md#autograd) support. Compiler-owned gradient rules may later consume the preserved
eligibility [metadata](../../../../glossary.md#term-metadata).

### Numerical and special-value meaning

The mathematical definitions are evaluated over the represented floating type by a future
execution [owner](../../../../glossary.md#term-owner). They select these portable special-value classes:

| Kind | [Negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero) | [Negative zero](../../../../glossary.md#nan-infinity-and-signed-zero) | [Positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) | [Positive infinity](../../../../glossary.md#nan-infinity-and-signed-zero) | [NaN](../../../../glossary.md#term-nan) |
|---|---|---|---|---|---|
| [`GELU`](../../../../glossary.md#gaussian-error-linear-unit--gelu) | negative zero | negative zero | positive zero | positive infinity | NaN |
| `GELU_TANH_APPROXIMATION` | negative zero | negative zero | positive zero | positive infinity | NaN |
| `SILU` | negative zero | negative zero | positive zero | positive infinity | NaN |

The [infinity](../../../../glossary.md#term-infinity) entries are the continuous extensions of the selected functions, not a required
literal order of primitive floating operations. [Finite](../../../../glossary.md#term-finite) [overflow](../../../../glossary.md#term-overflow), [underflow](../../../../glossary.md#term-underflow), [subnormal](../../../../glossary.md#term-subnormal) handling,
[rounding](../../../../glossary.md#term-rounding), and [NaN payload](../../../../glossary.md#nan-infinity-and-signed-zero) propagation follow future per-data-type execution and [conformance](../../../../glossary.md#term-conformance)
contracts. This task promises neither bitwise equality nor a fixed accuracy bound and chooses no
[backend](../../../../glossary.md#backend) algorithm.

### Result [metadata](../../../../glossary.md#term-metadata) and [provenance](../../../../glossary.md#provenance)

Every successful method creates:

- one fresh `TensorDescriptor` with exact input type and Shape, empty layout, and unchanged
  gradient eligibility;<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- one [`Operation`](../../../../glossary.md#operation) with the exact selected [unary](../../../../glossary.md#term-unary) kind and `NoOperationAttrs.INSTANCE`;
- one fresh single-output [`TensorProducer`](../../../../glossary.md#producer--tensorproducer) retaining exactly `[this]` and the exact result
  descriptor; and<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- one fresh unlabeled, storage-free [Tensor](../../../../glossary.md#tensor) with output index zero and one newly allocated ID.

Inputs and all input metadata, provenance, labels, storage, and IDs remain unchanged. Repeated
calls create distinct Tensor and [producer](../../../../glossary.md#term-producer) [identities](../../../../glossary.md#term-identity-meanings). Chaining records the immediately preceding
Tensor as the exact input. No [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form) or [eager](../../../../glossary.md#term-eager-evaluation) simplification occurs.

### [Validation](../../../../glossary.md#term-validation) order, messages, and ID effects

The public zero-[argument](../../../../glossary.md#term-argument) methods delegate directly to `TensorUnaryExpressions.apply(this, kind)`.
That helper retains this exact order:

1. null-check `input`, then `kind`, with the [parameter](../../../../glossary.md#term-parameter) name as message;
2. read the [descriptor](../../../../glossary.md#term-descriptor) type and require a floating type;
3. construct the result descriptor;
4. construct the exact parameterless [operation](../../../../glossary.md#operation); and
5. delegate exactly once to `TensorFactory.createDerived`.

The existing task-owned type failure remains exact:

```text
input must be a floating data type, but was <dataType>
```

Every null/type failure before step 5 consumes no [Tensor](../../../../glossary.md#tensor) ID. Only the final factory delegation may
allocate an ID; identifier exhaustion retains the existing exact message and behavior. Focused
tests inspect the shared ID counter only for this existing concrete failure-side-effect risk.

### [Operation](../../../../glossary.md#operation)-[signature](../../../../glossary.md#term-signature) compatibility

The existing [unary](../../../../glossary.md#term-unary) signature remains the sole family variant and accepts all nineteen enum
[constants](../../../../glossary.md#term-constant) through exact [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs) class matching. `Operation` and [`TensorProducer`](../../../../glossary.md#producer--tensorproducer)
continue to validate the one-input, one-output [occurrence](../../../../glossary.md#term-occurrence). Add no [registry](../../../../glossary.md#term-registry), [schema](../../../../glossary.md#term-schema), reflective
[dispatch](../../../../glossary.md#term-dispatch), [backend](../../../../glossary.md#backend) support, result facts, or [gradient](../../../../glossary.md#term-gradient) [metadata](../../../../glossary.md#term-metadata) to the signature.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model master plan](../master-plan.md)

## Architecture constraints

- Work remains entirely inside `modules/model` plus its documentation/[planning](../../../../glossary.md#term-planning) records.
- [`Tensor`](../../../../glossary.md#tensor) remains public [mutable](../../../../glossary.md#term-mutable) [API](../../../../glossary.md#term-application-programming-interface-api) state and is not [graph](../../../../glossary.md#graph) [IR](../../../../glossary.md#term-intermediate-representation-ir).
- [`Operation`](../../../../glossary.md#operation) and unary kinds record backend-independent meaning only and expose no support,
  lowering, algorithm, or execution route.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering))</span>
- Package direction remains `model.tensor -> model.operation.elementwise.unary`; the operation
  package must not import Tensor, graph, compiler, backend, or runtime state.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Compiler later owns capture, graph-wide validation, canonicalization/decomposition, and gradient
  construction. Backend prepare owns lowering, numerical implementation, fusion, and kernel route.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [validation glossary](../../../../glossary.md#term-validation); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [decomposition glossary](../../../../glossary.md#term-decomposition); [gradient glossary](../../../../glossary.md#term-gradient); [prepare glossary](../../../../glossary.md#prepare); [fusion glossary](../../../../glossary.md#term-fusion); [kernel glossary](../../../../glossary.md#kernel))</span>
- No architecture, [dependency](../../../../glossary.md#term-dependency), [lifecycle](../../../../glossary.md#lifecycle), [module](../../../../glossary.md#term-module)-boundary, [Gradle](../../../../glossary.md#term-gradle), or cross-module change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.operation`
- `io.github.pho001.synaptik.model.operation.elementwise.unary`
- `io.github.pho001.synaptik.model.tensor`

Packages added or changed:

- No package is added. The existing unary operation package gains three enum constants, and the
  existing Tensor package gains three facade methods.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [operation glossary](../../../../glossary.md#operation); [constants glossary](../../../../glossary.md#term-constant); [Tensor glossary](../../../../glossary.md#tensor); [facade glossary](../../../../glossary.md#term-facade))</span>

Type placement:

- `io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind` — existing
  public owner of parameterless floating-preserving unary mathematical and activation identities.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([activation glossary](../../../../glossary.md#term-activation))</span>
- `io.github.pho001.synaptik.model.tensor.TensorUnaryExpressions` — existing package-private
  shared floating-unary validation and derived-construction boundary.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- `io.github.pho001.synaptik.model.tensor.Tensor` — existing public fluent [API](../../../../glossary.md#term-application-programming-interface-api) [owner](../../../../glossary.md#term-owner).

Tests mirror the production packages whose typed and package-private surfaces they inspect.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/unary/UnaryElementwiseKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorUnaryExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/elementwise/unary/UnaryElementwiseKindTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorUnaryElementwiseTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java` — update only
  the exact public Tensor API inventory and method count from 157 to 160.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java` —
  update only the shared exact public Tensor method count from 157 to 160.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java` —
  update only the stale shared exact public Tensor method count from 157 to 160; preserve every
  MATMUL-specific assertion and behavior.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>

Documentation/[planning](../../../../glossary.md#term-planning):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless the final implementation makes an existing statement inaccurate:
Training [API](../../../../glossary.md#term-application-programming-interface-api); [scalar](../../../../glossary.md#term-scalar)/[binary](../../../../glossary.md#term-binary-descriptor-and-metadata)/where/cast/indexing/factory/[producer](../../../../glossary.md#term-producer)/[provenance](../../../../glossary.md#provenance)/[signature](../../../../glossary.md#term-signature) contracts;
architecture/ADRs/tests; [conformance](../../../../glossary.md#term-conformance)/integration; [Gradle](../../../../glossary.md#term-gradle); other [modules](../../../../glossary.md#term-module).

## Maximum [scope](../../../../glossary.md#term-scope)

At most three production, five test, and seven documentation/[planning](../../../../glossary.md#term-planning) files: exactly 15 paths.
`TensorUnaryExpressions.java` changes only for its kind-count documentation.
`Tensor.java` changes only for the three exact methods, required import/reference text, and its
[Javadoc](../../../../glossary.md#term-javadoc). `TensorMatmulExpressionTest.java` changes only for its stale total public [Tensor](../../../../glossary.md#tensor) method
count. No new Java type or package is permitted. Stop for a sixteenth path, another source or test
contract, any executable behavior outside [model](../../../../glossary.md#term-model), cross-[module](../../../../glossary.md#term-module) work, or architecture change.

## [Javadoc](../../../../glossary.md#term-javadoc) and documentation requirements

- Update unary kind, helper, and Tensor Javadocs for the three exact semantics, formula choices,
  floating domain, Shape/type/eligibility/layout retention, special values, provenance, freshness,
  validation, failure side effects, and owning-layer boundaries.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [Tensor glossary](../../../../glossary.md#tensor); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance); [validation glossary](../../../../glossary.md#term-validation))</span>
- Document every new public method with meaningful description, `@return`, all applicable
  `@throws`, and no invented execution or [gradient](../../../../glossary.md#term-gradient) promise.
- Update Tensor API's current inventory, unary table, formulas, special-value table, and metadata
  example or add a focused planned-versus-current example that observes only model metadata.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [metadata glossary](../../../../glossary.md#term-metadata); [model glossary](../../../../glossary.md#term-model))</span>
- Update Compile API only enough to list these current expressions while preserving the explicit
  planned capture/decomposition/gradient boundary. Do not claim compiler support.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [capture glossary](../../../../glossary.md#term-capture-meanings); [decomposition glossary](../../../../glossary.md#term-decomposition); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Update the glossary's unary-family inventory and define GELU and SiLU only if the final prose
  uses them as reusable project terms; distinguish SiLU from the omitted <span data-glossary-navigation="true"> ([GELU glossary](../../../../glossary.md#gaussian-error-linear-unit--gelu); [SiLU glossary](../../../../glossary.md#sigmoid-linear-unit--silu))</span>`swish` [alias](../../../../glossary.md#aliasing-and-arena).
- Keep capabilities, task, master plan, and roadmap synchronized. Record reasoned no-change
  conclusions for Training API, related contracts, architecture/tests, conformance/integration,
  Gradle, and other modules.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

## Acceptance criteria

- `UnaryElementwiseKind` contains exactly nineteen values, adding only [`GELU`](../../../../glossary.md#gaussian-error-linear-unit--gelu),
  `GELU_TANH_APPROXIMATION`, and `SILU` with the unchanged exact no-attributes one-input,
  one-output signature.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>
- [Tensor](../../../../glossary.md#tensor) adds exactly `gelu()`, `geluTanhApproximation()`, and `silu()`; its exact public method
  count becomes 160, and no `swish` or configurable [activation](../../../../glossary.md#term-activation) overload exists.
- Exact GELU, fixed tanh-approximation GELU, and exact SiLU meanings and special-value classes
  match this specification without composition producers or algorithm promises.<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([tanh glossary](../../../../glossary.md#term-tanh); [SiLU glossary](../../../../glossary.md#sigmoid-linear-unit--silu))</span>
- Every floating type and Shape state preserves exact type/Shape/eligibility metadata, leaves
  layout unresolved, and creates exact parameterless one-input, output-index-zero provenance.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [metadata glossary](../../../../glossary.md#term-metadata); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance))</span>
- Every valid call is fresh, unlabeled, storage-free, and non-canonicalized; inputs remain
  unchanged.
- INT32, INT64, and BOOL fail with the existing exact message and no ID consumption. Null helper
  tests retain input-then-kind order; only final factory exhaustion may consume/fail an ID.<span data-glossary-navigation="true"> ([INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Tests validate kinds, signatures, API shape, descriptor/provenance/freshness, and failures
  without pretending to execute activation values.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- No embedding, one-hot, swish alias, attributes, gradients, compiler/backend/runtime behavior,
  dependencies, Gradle, architecture, or other-module work lands.<span data-glossary-navigation="true"> ([embedding glossary](../../../../glossary.md#embedding); [alias glossary](../../../../glossary.md#aliasing-and-arena); [gradients glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>
- Focused tests, the final model suite after Java stabilizes, Javadoc, documentation/link, exact
  15-path scope, status, formatting, newline, fence, and whitespace checks pass. If the first
  final suite exposes only a stale total-method inventory outside the original focused paths, one
  authorized scope-repair rerun may follow that exact inventory correction and must be recorded.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- A separate clean-context documentation pass finalizes all authorized documentation and records
  reused Java evidence plus reasoned no-change conclusions.
- Task 0019A becomes Complete only after all evidence is recorded. Task 0019 remains Complete;
  0019A1, 0019A2, and existing 0019B through 0019E remain Draft without detailed specifications.

## Tests / [validation](../../../../glossary.md#term-validation)

Required focused command:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKindTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorUnaryElementwiseTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorBinaryArithmeticTest
```

After executable Java stabilizes, normally [run](../../../../glossary.md#term-run) exactly one final [model](../../../../glossary.md#term-model) suite:

```bash
./gradlew :modules:model:test
```

The first final suite exposed the stale total public [Tensor](../../../../glossary.md#tensor) method count in
`TensorMatmulExpressionTest`. The authorized 15-path [scope](../../../../glossary.md#term-scope) repair changes only 157 to 160 there;
run one necessary final model-suite rerun afterward and record that reason. Do not perform any
additional model-suite rerun.

The separate documentation pass runs after final [Javadoc](../../../../glossary.md#term-javadoc) edits:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

Also validate local Markdown links and anchors, balanced fences, terminology, generated Javadoc,
final newlines, trailing whitespace, exact 15 paths, package placement, exact nineteen-kind and
160-method surfaces, forbidden [aliases](../../../../glossary.md#aliasing-and-arena), synchronized status, the model frontier, and the absence
of detailed 0019A1-or-later specifications. The documentation pass reuses the successful Java
evidence and must not rerun either Java command unless it changes executable Java behavior or
records a concrete cross-check reason.

Repository-wide validation is deferred to the selected-modern-[operation](../../../../glossary.md#operation)-family [checkpoint](../../../../glossary.md#term-checkpoint) after
task 0022 and to [CI](../../../../glossary.md#term-continuous-integration-ci). This task changes one existing family inside one [module](../../../../glossary.md#term-module) and does not alter a
[dependency](../../../../glossary.md#term-dependency), architecture boundary, shared build configuration, or multiple modules.

## Dependencies

- Tasks 0005–0006 and 0018K: operation kinds, typed attributes, exact family-owned signatures,
  and occurrence-cardinality validation.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [signatures glossary](../../../../glossary.md#term-signature); [occurrence glossary](../../../../glossary.md#term-occurrence); [cardinality glossary](../../../../glossary.md#term-cardinality); [validation glossary](../../../../glossary.md#term-validation))</span>
- Tasks 0011–0013 and 0018L: public Tensor construction, identity, producer, and indexed
  provenance contracts.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [identity glossary](../../../../glossary.md#term-identity-meanings); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- Tasks 0014C–0014D: the parameterless unary semantic family and public floating-unary
  construction boundary.
- Task 0018P: final portable [unary](../../../../glossary.md#term-unary) vocabulary and removal of uncontracted fast variants.

## Follow-up tasks

- Draft task 0019A1 owns `weights.embedding(indices)` as a rank-two floating embedding-table
  convenience that validates its public table contract and delegates to exact<span data-glossary-navigation="true"> ([rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [embedding glossary](../../../../glossary.md#embedding))</span>
  `weights.gather(indices, 0)` provenance. It will add no padding index, embedding kind, or gradient
  rule; compiler may reject captured constant bounds failures, while backend preparation and the
  prepared executable own safe handling of bound execution-time index values. Runtime does not
  inspect the original Gather operation.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [padding glossary](../../../../glossary.md#padding); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [constant glossary](../../../../glossary.md#term-constant); [backend glossary](../../../../glossary.md#backend); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable); [Runtime glossary](../../../../glossary.md#term-runtime); [Gather glossary](../../../../glossary.md#gather); [operation glossary](../../../../glossary.md#operation))</span>
- Draft task 0019A2 owns `indices.oneHot(long depth)` as a trailing-axis, BOOL-valued first-class
  encoding with <span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>`false` off, `true` on, false gradient eligibility, positive static depth, and
  all-false rows for negative or too-large indices. It will add no output-type, axis, or on/off
  configuration and no eager TensorFactory depth vector. Its selected Draft semantic shape is<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [vector glossary](../../../../glossary.md#term-vector); [shape glossary](../../../../glossary.md#shape))</span>
  `OneHotKind.ONE_HOT` plus `OneHotAttrs(long depth)` with a fixed one-input, one-output [signature](../../../../glossary.md#term-signature).
- Existing tasks 0019B through 0019E retain their Draft [identities](../../../../glossary.md#term-identity-meanings) and scopes.
- In particular, unchanged Draft task 0019B remains the sole owner of dropout because dropout
  requires explicit state-consuming/state-producing graph RNG semantics rather than another
  deterministic unary activation.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [graph glossary](../../../../glossary.md#graph); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [unary glossary](../../../../glossary.md#term-unary); [activation glossary](../../../../glossary.md#term-activation))</span>
- No detailed follow-up specification is created by this [planning](../../../../glossary.md#term-planning) task.

## Architecture impact

Expected impact: None.

The task extends [model](../../../../glossary.md#term-model)-owned [backend](../../../../glossary.md#backend)-independent [unary](../../../../glossary.md#term-unary) semantics and the public [Tensor](../../../../glossary.md#tensor) expression
surface without changing [module](../../../../glossary.md#term-module) [ownership](../../../../glossary.md#term-ownership-meanings), [dependency direction](../../../../glossary.md#java-platform-and-build-tools), [lifecycle](../../../../glossary.md#lifecycle) contracts, or backend
responsibility. If implementation requires any architecture change, another package/type, or a
different semantic boundary, stop and report before editing architecture or expanding [scope](../../../../glossary.md#term-scope).

## Implementation prompt

Use this prompt in a separate agentic task/thread with clean context:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, documentation/planning rules, roadmap, model capabilities/master
plan, tasks 0014C–0014D/0018K/0018L/0018P/0019A, Tensor and Compile APIs, glossary, and every
affected or review-only source/test named by task 0019A in full.

Implement task 0019A exactly. Add only first-class exact GELU, fixed tanh-approximation GELU, and
SiLU parameterless unary semantics plus gelu(), geluTanhApproximation(), and silu(). Preserve the
exact current floating-unary validation, metadata, producer/provenance, freshness, and ID-side-
effect contracts. Add no composition, swish alias, attribute/configuration, embedding, one-hot,
gradient, compiler/backend/runtime behavior, dependency, Gradle, architecture change, or later
spec. Stay within the exact fifteen paths and stop on scope or architecture conflict. Do not
commit or push.

Run the focused tests and the final model suite after executable Java stabilizes. The authorized
15-path repair permits one necessary rerun only if the first suite fails solely on the stale
MATMUL test's total Tensor method count; record that reason and perform no additional model rerun.
Then hand the actual diff and exact evidence to a separate clean-context documentation-focused
agent in the same overall change. That agent must inspect final source/tests, finalize permitted
Javadocs/Tensor API/Compile API/glossary/capability/planning documents, run model Javadoc and
documentation/scope checks, and must not repeat successful Java tests unless executable behavior
changes or it records a concrete reason.

Mark 0019A Complete only after both passes succeed. Leave 0019A1, 0019A2, and 0019B–0019E Draft
without detailed specifications.
```

## Local decisions

- GELU defaults to the exact error-function target; the tanh approximation is available only by
  the explicit <span data-glossary-navigation="true"> ([GELU glossary](../../../../glossary.md#gaussian-error-linear-unit--gelu); [target glossary](../../../../glossary.md#term-target); [tanh glossary](../../../../glossary.md#term-tanh))</span>`geluTanhApproximation()` spelling.
- Exact and tanh GELU are separate parameterless kinds because they are different fixed functions,
  not one kind with runtime configuration.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime))</span>
- GELU and SiLU are first-class because literal primitive chains do not preserve the selected
  negative-infinity limit and would erase useful semantic identity.<span data-glossary-navigation="true"> ([SiLU glossary](../../../../glossary.md#sigmoid-linear-unit--silu); [infinity glossary](../../../../glossary.md#term-infinity); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `silu` is [canonical](../../../../glossary.md#term-canonical-form) and no `swish` [alias](../../../../glossary.md#aliasing-and-arena) is added.
- [Model](../../../../glossary.md#term-model) retains [gradient](../../../../glossary.md#term-gradient) eligibility but defines no gradient or subgradient rule.

## Known limitations

- No execution owner, per-type accuracy tolerance, correct-rounding promise, or backend support is
  included.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [tolerance glossary](../../../../glossary.md#term-tolerance); [rounding glossary](../../../../glossary.md#term-rounding); [backend glossary](../../../../glossary.md#backend))</span>
- No configurable approximation, coefficients, activation parameters, or compatibility aliases
  are included.<span data-glossary-navigation="true"> ([activation glossary](../../../../glossary.md#term-activation); [parameters glossary](../../../../glossary.md#term-parameter); [aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
- [Gradient](../../../../glossary.md#term-gradient) eligibility does not imply that [compiler](../../../../glossary.md#term-compiler) [autograd](../../../../glossary.md#autograd) can differentiate these operations.
- [Embedding](../../../../glossary.md#embedding) and one-hot remain separate Draft follow-ups.

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) context: clean documentation/planning task `/root/plan_0019a`.

- Read the architecture contract and focused architecture index/module/dependency/lifecycle/
  training documents; documentation rules and General/Planning/Example profiles; planning guide,
  roadmap, model capability baseline/master plan; completed semantic, Tensor, factory, indexing,
  provenance, signature-hardening, reset, and MATMUL task history; current relevant source/tests;
  Tensor/Compile/Training API references; glossary; and current public API inventories.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [lifecycle glossary](../../../../glossary.md#lifecycle); [model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor); [provenance glossary](../../../../glossary.md#provenance); [signature glossary](../../../../glossary.md#term-signature); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [Compile glossary](../../../../glossary.md#compile))</span>
- Confirmed the current unary family has sixteen parameterless one-input/one-output kinds, all
  using exact <span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary))</span>[`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), and the shared helper already provides the required floating
  validation, descriptor, provenance, freshness, and no-ID local-failure behavior.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Confirmed Tensor currently has 157 public methods, so three additions produce the exact 160-
  method surface and require only the two existing inventory/count updates named above.
- Confirmed literal GELU and SiLU primitive chains do not preserve the selected negative-infinity
  limiting result because they encounter zero multiplied by negative infinity. This supplies the
  semantic reason for first-class kinds despite the earlier provisional convenience label.<span data-glossary-navigation="true"> ([GELU glossary](../../../../glossary.md#gaussian-error-linear-unit--gelu); [SiLU glossary](../../../../glossary.md#sigmoid-linear-unit--silu); [infinity glossary](../../../../glossary.md#term-infinity); [negative infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Confirmed embedding maps exactly to axis-zero Gather for a rank-two table, whereas one-hot needs
  distinct rank-changing, fixed-BOOL semantics; they are separate Draft rows without detailed
  specifications.<span data-glossary-navigation="true"> ([embedding glossary](../../../../glossary.md#embedding); [axis glossary](../../../../glossary.md#term-axis); [Gather glossary](../../../../glossary.md#gather); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [rank glossary](../../../../glossary.md#term-rank); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Planning-only validation found exactly the four authorized paths: capabilities, model master
  plan, this new task, and roadmap. Source/build/architecture/other-module scans found no changed
  path. Status scans found exactly one Ready model task and one matching Ready row in each queue;
  0019A1, 0019A2, and 0019B–0019E remain Draft, with no detailed later specification.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- A targeted Ruby Markdown check resolved all 345 local links and fragments across the four
  changed files. Fence parity, final-newline, trailing-whitespace, and exact-path checks passed.
  `git diff --check` passed with no output; the new untracked task separately passed newline,
  whitespace, and fence checks because ordinary diff checks do not include untracked files.
- Implementation, Java tests, Javadoc generation, and executable documentation validation remain
  deferred to the required implementation and documentation contexts.

Implementation context: clean task `/root/task_0019a_implementation`.

- The focused command in this task's validation section passed before the authorized [scope](../../../../glossary.md#term-scope) repair:
  `BUILD SUCCESSFUL`, with three tasks executed.
- The first final `./gradlew :modules:model:test` ran 758 tests and failed only because
  `TensorMatmulExpressionTest` line 69 retained the stale total public Tensor method count of 157.
  No activation behavior failed. The user authorized expanding the exact scope from 14 to 15 paths
  for that count-only repair.<span data-glossary-navigation="true"> ([activation glossary](../../../../glossary.md#term-activation))</span>
- After changing only that stale count to 160, the necessary final
  `./gradlew :modules:model:test` rerun passed: `BUILD SUCCESSFUL`, three actionable tasks, two
  executed and one up-to-date. Executable Java did not change afterward.

Documentation context: clean task `/root/task_0019a_implementation/docs_0019a` using the General,
[API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), Planning, and Example profiles.

- Independently read the required architecture, documentation, planning, API, glossary, source,
  tests, and directly relevant unary/provenance/signature contracts and inspected the actual
  shared-tree diff.
- Finalized the three production Javadocs and the Tensor/Compile API, glossary, capabilities, task,
  master-plan, and roadmap content. The public references now distinguish exact GELU, the fixed
  tanh approximation, and canonical SiLU; document formulas and function-level continuous
  extensions; and separate current metadata construction from planned compiler capture,
  decomposition, gradient construction, backend preparation, and execution.<span data-glossary-navigation="true"> ([tanh glossary](../../../../glossary.md#term-tanh); [canonical glossary](../../../../glossary.md#term-canonical-form); [metadata glossary](../../../../glossary.md#term-metadata); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [decomposition glossary](../../../../glossary.md#term-decomposition); [gradient glossary](../../../../glossary.md#term-gradient); [backend glossary](../../../../glossary.md#backend))</span>
- `./gradlew :modules:model:javadoc` passed after final Javadoc review: `BUILD SUCCESSFUL`, two
  actionable tasks, one executed and one up-to-date.
- A targeted Ruby Markdown check resolved all 509 local links and fragments across the seven
  documentation/planning paths. Separate format checks passed final-newline, trailing-whitespace,
  and balanced-fence checks for all seven paths.
- Source/scope checks found exactly nineteen unary kinds, exactly 160 public Tensor methods, and
  exactly the three new zero-argument methods. Production scans found no <span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument))</span>`swish`, embedding,
  one-hot, configurable activation attributes, or later-task implementation/specification.
- Final status checks found task 0019 and task 0019A Complete; 0019A1, 0019A2, and 0019B–0019E
  remain Draft without detailed specifications; no model task is currently Ready.
- Final combined scope is exactly 15 paths: three production, five tests, and seven documentation/
  planning paths. `git diff --check` passed with no output; the untracked task separately passed
  the same whitespace/newline checks.
- Training API requires no change because this task preserves eligibility metadata but adds no
  gradient rule, training workflow, or trainable-state contract. Related scalar, binary, where,
  cast, indexing, factory, producer, provenance, and signature documents require no change because
  their shared construction and cardinality contracts remain unchanged. Architecture documents,
  ADRs, and architecture tests require no change because module ownership and dependency rules do
  not change. Backend conformance and integration tests require no change because no executable
  behavior or backend support is added. Gradle and other modules require no change because the
  task adds no dependency, build configuration, or cross-module API.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [producer glossary](../../../../glossary.md#term-producer); [cardinality glossary](../../../../glossary.md#term-cardinality); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle))</span>

## Implementation notes

- Added only [`GELU`](../../../../glossary.md#gaussian-error-linear-unit--gelu), `GELU_TANH_APPROXIMATION`, and `SILU` to the existing parameterless unary
  family and only <span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary))</span>`gelu()`, `geluTanhApproximation()`, and `silu()` to [Tensor](../../../../glossary.md#tensor).
- Reused the existing floating-unary helper without executable modification; its Javadoc count is
  now nineteen and its validation, metadata, provenance, freshness, and ID-side-effect contracts
  remain unchanged.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [validation glossary](../../../../glossary.md#term-validation); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance))</span>
- Repaired the shared MATMUL API-inventory test only after the first final suite exposed its stale
  count, as explicitly authorized. No MATMUL-specific assertion changed.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- No Java behavior changed during the documentation-focused pass.

## Completion summary

- Completed changes: added first-class exact GELU, fixed tanh-approximation GELU, and SiLU semantic
  identities plus their three public storage-free Tensor expression methods, tests, Javadocs, and
  synchronized public/planning documentation.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([GELU glossary](../../../../glossary.md#gaussian-error-linear-unit--gelu); [tanh glossary](../../../../glossary.md#term-tanh); [SiLU glossary](../../../../glossary.md#sigmoid-linear-unit--silu); [Tensor glossary](../../../../glossary.md#tensor); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: `UnaryElementwiseKind.java`, `TensorUnaryExpressions.java`,
  `Tensor.java`; `UnaryElementwiseKindTest.java`, `TensorUnaryElementwiseTest.java`, `TensorTest.java`,
  `TensorBinaryArithmeticTest.java`, `TensorMatmulExpressionTest.java`; `tensor-api.md`,
  `compile-api.md`, `glossary.md`, [model](../../../../glossary.md#term-model) `capabilities.md`, this task, model `master-plan.md`, and
  `roadmap.md`.
- Tests and validation: reused the successful focused and repaired final model-test evidence above;
  final model Javadoc, 509 local-link/anchor checks, formatting, exact-scope, surface, terminology,
  status, forbidden-feature, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope); [feature glossary](../../../../glossary.md#term-software-feature))</span>`git diff --check` checks passed.
- Documentation-agent review: completed in clean context
  `/root/task_0019a_implementation/docs_0019a` under the required profiles.
- Documentation impact: Tensor API, Compile API, glossary, capabilities, task, master plan, and
  roadmap are synchronized with current model construction and planned downstream ownership.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Javadoc review: all three affected production contracts are detailed and consistent; no further
  correction was required after independent review.
- Glossary impact: added reusable GELU and SiLU definitions and the exact-versus-approximation and
  canonical-SiLU/no-alias distinctions.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Unresolved issues: None.
- Follow-up required: None for task 0019A. Separate 0019A1, 0019A2, and 0019B–0019E work remains
  Draft and outside this task.

Status: Complete
