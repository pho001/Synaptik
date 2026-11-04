# Task 0004: [Compiler](../../../../glossary.md#term-compiler)-owned Pre-[capture](../../../../glossary.md#term-capture-meanings) [Autograd](../../../../glossary.md#autograd) and [Graph](../../../../glossary.md#graph) [Compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names)

> **Editorial reading key — not part of the historical record.**
>
> Compiler calls preflight and ordinary Tensor-operation constructors to build derivative
> expressions before assigning graph-local identities. A cotangent is an incoming output
> sensitivity, not a stored Tensor gradient. For a real scalar illustration, let $x\in\mathbb{R}$
> be the input, let $s\in\mathbb{R}$ be the seed and let $f:\mathbb{R}\to\mathbb{R}$ be
> $f(x)=x^2$. The reverse contribution is $g=s f'(x)=2sx$. At $x=3$ and $s=2$, $g=12$: the seed
> scales sensitivity, not the forward value. This illustration does not replace the recorded
> floating-point rules. One combined capture records forward and derivative roots; no numerical
> backend execution occurs during this construction.
>
> Current reading: [pre-capture differentiation and functional requests](../../../../reference/compiler/autograd.md#current-package-private-pre-capture-autograd); [structural capture and exact leaf identities](../../../../reference/compiler/capture.md#current-package-private-structural-capture).

## Status

Complete

## Goal

Implement the first bounded [compiler](../../../../glossary.md#term-compiler)-owned [reverse-mode](../../../../glossary.md#primal-adjoint-and-differentiation-seed) automatic-differentiation path over the
original [Tensor](../../../../glossary.md#tensor) expression [DAG](../../../../glossary.md#term-directed-acyclic-graph-dag), then [capture](../../../../glossary.md#term-capture-meanings) the requested forward results and generated
first-order [gradients](../../../../glossary.md#term-gradient) together exactly once.

The task must [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation) before allocating any [derivative](../../../../glossary.md#term-derivative) Tensor [identity](../../../../glossary.md#term-identity-meanings) when the requested
backward [slice](../../../../glossary.md#slice) contains an unsupported [occurrence](../../../../glossary.md#term-occurrence), output role, attributes variant, data-type
relationship, or derivative policy. A successful backward-capable request constructs formulas
only with ordinary public `Tensor` operations, explicitly registers storage-free logical-splat
[constants](../../../../glossary.md#term-constant), assigns [graph](../../../../glossary.md#graph)-local identities once during one phase-aware combined capture, validates
the combined graph, and applies only the exact transformations already selected by Compiler
0003–0003B.

This task also establishes the smallest package-private [scalar](../../../../glossary.md#term-scalar)-[objective](../../../../glossary.md#term-objective) and gradient-[target](../../../../glossary.md#term-target)
contract needed to prove the pipeline. It does not establish the future public [compile](../../../../glossary.md#compile),
[publication](../../../../glossary.md#term-publication), arbitrary-[seed](../../../../glossary.md#term-seed), or higher-derivative APIs.

## [Scope](../../../../glossary.md#term-scope)

### Mandatory obsolete-source cleanup

Before creating or modifying any implementation Java file, verify that these exact six paths exist
and are reported as untracked (`??`) by `git status --short -- <paths>`:

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradExpansion.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradGraph.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/BackwardGraphBuilder.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ElementwiseAdjoints.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/IndexingRandomAdjoints.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ReductionLayoutAdjoints.java`

They are obsolete pre-[planning](../../../../glossary.md#term-planning) [prototypes](../../../../glossary.md#term-prototype) located under the [compiler](../../../../glossary.md#term-compiler) production source root, so
leaving them present would make [Gradle](../../../../glossary.md#term-gradle) [compile](../../../../glossary.md#compile) unauthorized code. After the exact path/status
check succeeds, delete all six as the first implementation cleanup step and verify their absence
before compiling or testing.

The cleanup is removal-only. Do not read them for design evidence, copy, rename, move, stage,
adapt, salvage, quote, or reintroduce any content from them. They are not repository history,
legacy capability evidence, architecture, accepted design, or an implementation starting point.
If any path is missing, tracked, modified in Git rather than untracked, or replaced by a different
filesystem type at implementation start, stop and report the status mismatch before touching
implementation files.

### Package-private [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) request

Add one package-private stateless `GraphCompiler` entry point with this exact [parameter](../../../../glossary.md#term-parameter) list:

```java
static GraphCompilation compile(
        CompileMode mode,
        List<Tensor> forwardOutputs,
        Optional<AutogradPreflight.FirstOrderRequest> firstOrderRequest,
        CompileTimeConstantGraph.Ingress forwardConstants,
        GraphOptimizationConfig optimizationConfig)
```

Do not wrap these parameters in a request aggregate or add another entry type. Their types and
order are fixed by this task. The implementation must not add a public [facade](../../../../glossary.md#term-facade) or aggregate.
[Validation](../../../../glossary.md#term-validation) is deterministic and checks the [arguments](../../../../glossary.md#term-argument) in declaration order before [graph](../../../../glossary.md#graph)
construction.

`forwardOutputs` is a non-null, non-empty ordered [snapshot](../../../../glossary.md#term-snapshot). Its [Tensor](../../../../glossary.md#tensor) references must be
[identity](../../../../glossary.md#term-identity-meanings)-unique and must resolve to distinct logical forward values, preserving [Compiler](../../../../glossary.md#term-compiler) 0001
behavior. The complete original [producer](../../../../glossary.md#term-producer) inventory starts from every forward output, not only the
backward [objective](../../../../glossary.md#term-objective), because [capture](../../../../glossary.md#term-capture-meanings) phase classification covers the complete mode-selected
request.

The mode [matrix](../../../../glossary.md#term-matrix-meanings) is exact:

| [`CompileMode`](../../../../glossary.md#compile-mode--compilemode) | First-order request | Behavior |
|---|---|---|
| `FORWARD_ONLY` | Must be absent. | Inventory and phase-aware capture the requested forward [DAG](../../../../glossary.md#term-directed-acyclic-graph-dag), perform no [derivative](../../../../glossary.md#term-derivative) [preflight](../../../../glossary.md#term-preflight) or construction, then validate and optimize the forward-only graph-stage result. |
| `FORWARD_AND_BACKWARD` | Must be present. | Preflight and construct the first-order backward [slice](../../../../glossary.md#slice), then use the combined pipeline. |
| `TRAINING_STEP` | Must be present. | Perform exactly the same work as `FORWARD_AND_BACKWARD`; add no [optimizer](../../../../glossary.md#term-optimizer), optimizer-update [node](../../../../glossary.md#node), session, schedule, preparation, or execution state. |

No mode is inferred from Tensor [metadata](../../../../glossary.md#term-metadata). A mode/request mismatch fails before derivative
construction.

### First-order [objective](../../../../glossary.md#term-objective), [target](../../../../glossary.md#term-target), and [seed](../../../../glossary.md#term-seed) contract

`AutogradPreflight.FirstOrderRequest` is a package-private [immutable](../../../../glossary.md#term-immutable) request containing:

```java
record FirstOrderRequest(Tensor objective, List<Tensor> targets)
```

Its exact contract is:

- `objective` is one exact [Tensor](../../../../glossary.md#tensor) reference present in `forwardOutputs`;
- the objective has [scalar](../../../../glossary.md#term-scalar) [`Shape`](../../../../glossary.md#shape), one floating [data type](../../../../glossary.md#data-type--datatype), and `requiresGrad == true`;
- `targets` is a non-null, non-empty ordered immutable snapshot of identity-unique exact Tensor
  references;<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- each target occurs in the original objective ancestry, has the same floating data type as the
  objective along the selected supported path, and has `requiresGrad == true`;
- a target may be a [provenance](../../../../glossary.md#provenance)-free [leaf](../../../../glossary.md#term-leaf) or an intermediate original Tensor;
- the objective itself may be a target;
- each target must be reachable through at least one differentiable input-role path selected by
  this task's matrix; an unrelated or condition-only target is rejected rather than assigned an
  invented zero;<span data-glossary-navigation="true"> ([differentiable glossary](../../../../glossary.md#derivative-and-chain-rule); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- target order determines result-role order and is otherwise not a [graph](../../../../glossary.md#graph) traversal order; and
- targets are not traversal stop points, because another requested target may be upstream of an
  intermediate target.

The only current seed is an implicit [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) positive one with the objective's exact floating
data type. There is no caller-supplied seed Tensor. The [compiler](../../../../glossary.md#term-compiler) creates that seed only after
[preflight](../../../../glossary.md#term-preflight) succeeds as a provenance-free, storage-free, non-[gradient](../../../../glossary.md#term-gradient) Tensor leaf and explicitly
registers its exact logical-splat value. [BFLOAT16](../../../../glossary.md#term-bfloat16) one uses exact bits `0x3F80`; [FLOAT32](../../../../glossary.md#term-float32) and [FLOAT64](../../../../glossary.md#term-float64)
use exact positive `1.0` in their respective types.

Arbitrary objectives, non-scalar objectives, explicit [seeds](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings), [vector](../../../../glossary.md#term-vector)-[Jacobian](../../../../glossary.md#term-jacobian) products,
disconnected-target zero policy, public target/[publication](../../../../glossary.md#term-publication) requests, and multiple [derivative](../../../../glossary.md#term-derivative)
orders remain Compiler 0006 work.

### Original forward inventory and [fail-closed](../../../../glossary.md#term-fail-closed) [preflight](../../../../glossary.md#term-preflight)

Add `AutogradPreflight` as one package-private stateless [owner](../../../../glossary.md#term-owner) of:

- a deterministic iterative inventory of exact original [Tensor](../../../../glossary.md#tensor) and [`TensorProducer`](../../../../glossary.md#producer--tensorproducer) [identities](../../../../glossary.md#term-identity-meanings);
- [producer](../../../../glossary.md#term-producer) postorder and reverse-postorder [views](../../../../glossary.md#term-view-meanings);
- exact producer inputs, every [canonical](../../../../glossary.md#term-canonical-form) output wrapper and output index, exact [`Operation`](../../../../glossary.md#operation),
  attributes runtime type and value, and descriptor facts;<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- [objective](../../../../glossary.md#term-objective)-to-[target](../../../../glossary.md#term-target) [differentiable](../../../../glossary.md#derivative-and-chain-rule)-role reachability; and
- the closed rule-selection decision for every occurrence/output/input role required by the
  request.

Traversal must use object [identity](../../../../glossary.md#term-reference-identity), not [`TensorId`](../../../../glossary.md#tensorid), record equality, operation equality, structural
hashing, labels, storage, or captured [graph](../../../../glossary.md#graph) IDs. It must be iterative so a deep expression chain
does not consume the Java call stack. Structurally equal but separately invoked producers remain
distinct occurrences. Repeated exact input positions remain repeated.

Preflight proceeds in this order:

1. validate the top-level mode, forward-output, optional request, ingress, and optimization
   arguments without creating a Tensor;<span data-glossary-navigation="true"> ([arguments glossary](../../../../glossary.md#term-argument))</span>
2. inventory the complete original forward DAG and validate the request's objective/target
   membership;<span data-glossary-navigation="true"> ([DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag))</span>
3. derive, by exact Tensor identity, which objective ancestry branches can reach at least one
   requested target through differentiable roles;
4. inspect every required producer occurrence, exact output role, input role, kind, attributes
   variant and value, descriptor, data-type relationship, Shape relationship, and required
   derivative policy;<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [derivative glossary](../../../../glossary.md#term-derivative))</span>
5. reject the first unsupported fact with deterministic occurrence, output-role, input-role,
   operation-kind, attributes, and reason context; and
6. only after the complete selected slice succeeds, create the implicit seed, derivative
   constants, or any formula Tensor.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice); [seed glossary](../../../../glossary.md#term-seed); [constants glossary](../../../../glossary.md#term-constant))</span>

An unsupported branch that cannot reach any requested target is not differentiated and does not
require a rule. A non-differentiable role stops that route. An unsupported [occurrence](../../../../glossary.md#term-occurrence) on a
selected objective-to-target route rejects the complete request. Preflight is not a replacement
for [Compiler](../../../../glossary.md#term-compiler) 0002 [inference](../../../../glossary.md#term-inference) and [validation](../../../../glossary.md#term-validation); post-[capture](../../../../glossary.md#term-capture-meanings) verification remains authoritative.

Known preflight rejection must consume no new `TensorId`. Once construction begins, a later public
Tensor-construction, capture, inference, validation, or optimization failure may consume IDs.
Identifiers are never rolled back or reused.

### Closed `SUPPORTED_0004` [derivative](../../../../glossary.md#term-derivative) [matrix](../../../../glossary.md#term-matrix-meanings)

Only the rows below are supported. Each selected [occurrence](../../../../glossary.md#term-occurrence) must have its current exact
family-owned [signature](../../../../glossary.md#term-signature), output count, attributes variant, [descriptor](../../../../glossary.md#term-descriptor) relationship, and a floating
[gradient](../../../../glossary.md#term-gradient) path. `g` denotes the accumulated [cotangent](../../../../glossary.md#cotangent) of the selected output, `x`/`left`/`right`
denote exact original inputs, `y` denotes the [producer](../../../../glossary.md#term-producer)'s [canonical](../../../../glossary.md#term-canonical-form) exact output wrapper, and
`B_input(v)` is `v.sumToShape(input.descriptor().shape())`.

| Family and exact variant | Additional [preflight](../../../../glossary.md#term-preflight) [guard](../../../../glossary.md#term-guard) | Formula for selected [differentiable](../../../../glossary.md#derivative-and-chain-rule) input roles |
|---|---|---|
| `BinaryArithmeticKind.ADD` with `NoOperationAttrs.INSTANCE` | Left, right, and output have one exact equal floating type; mixed floating promotion is rejected. | `B_left(g)`, `B_right(g)` |
| `BinaryArithmeticKind.SUB` with `NoOperationAttrs.INSTANCE` | Same-type guard as ADD. | `B_left(g)`, `B_right(g.neg())` |
| `BinaryArithmeticKind.MUL` with `NoOperationAttrs.INSTANCE` | Same-type guard as ADD. | `B_left(g.mul(right))`, `B_right(g.mul(left))` |
| `ScalarElementwiseKind.ADD` or `SUB` with exact `ScalarValueAttrs` | Input, output, and [scalar](../../../../glossary.md#term-scalar) attribute have one exact equal floating type. | `g` |
| `ScalarElementwiseKind.MUL` with exact `ScalarValueAttrs` | Same exact-type guard as scalar ADD/SUB. | `g.mul(attrs.value())` |
| `WhereSelectionKind.WHERE` with `NoOperationAttrs.INSTANCE` | Condition is exact [BOOL](../../../../glossary.md#data-type--datatype); both branches and output have one exact equal floating type, so no branch promotion/cotangent cast is required. | True branch: `B_true(Tensor.where(condition, g, Z_y))`; false branch: `B_false(Tensor.where(condition, Z_y, g))`; condition is non-differentiable. |
| `CastKind.CAST` with exact `CastAttrs` | Source and [target](../../../../glossary.md#term-target) are the same exact floating type. | `g` |
| `UnaryElementwiseKind.NEG` with `NoOperationAttrs.INSTANCE` | Input/output exact same floating type and [Shape](../../../../glossary.md#shape). | `g.neg()` |
| `UnaryElementwiseKind.EXP` with `NoOperationAttrs.INSTANCE` | Input/output exact same floating type and Shape. | `g.mul(y)` |
| `UnaryElementwiseKind.EXPM1` with `NoOperationAttrs.INSTANCE` | Input/output exact same floating type and Shape. | `g.mul(y.add(O_y))` |
| `UnaryElementwiseKind.SIGMOID` with `NoOperationAttrs.INSTANCE` | Input/output exact same floating type and Shape. | `g.mul(y).mul(O_y.sub(y))` |
| `UnaryElementwiseKind.TANH` with `NoOperationAttrs.INSTANCE` | Input/output exact same floating type and Shape. | `g.mul(O_y.sub(y.mul(y)))` |
| `AggregateReductionKind.SUM` with `NoOperationAttrs.INSTANCE` | One exact floating input/output type; this is full reduction only. | Restore every removed [axis](../../../../glossary.md#term-axis) as [extent](../../../../glossary.md#term-extent) one, then expand to `x.shape()`. |
| `AggregateReductionKind.SUM` with exact `AxisReductionAttrs` | One exact floating input/output type and the recorded [normalized axis](../../../../glossary.md#normalized-axis)/keep-[dimensions](../../../../glossary.md#dimension) facts. | If the axis was removed, `expandDims` at that axis; then expand to `x.shape()`. |
| `AggregateReductionKind.SUM` with exact `MultiAxisReductionAttrs` | One exact floating input/output type and ordered distinct normalized axes; an empty list is a point reduction. | If dimensions were removed, insert the selected axes in ascending axis order; then expand to `x.shape()`. Empty axes return `g`. |
| `CumulativeScanKind.CUM_SUM` with exact `CumulativeScanAttrs(axis, exclusive, reverse)` | Input/output exact same floating type and Shape. | `g.cumSum(axis, exclusive, !reverse)` |
| `ContiguousKind.CONTIGUOUS` with `NoOperationAttrs.INSTANCE` | Input/output exact same floating type and Shape. | `g` |
| `ShapeTransformKind.RESHAPE` with exact `TargetShapeAttrs` | Input/output exact same floating type and the current forward [element](../../../../glossary.md#term-element)-count contract. | `g.reshape(x.shape())` |
| `ShapeTransformKind.EXPAND` with exact `TargetShapeAttrs` | Input/output exact same floating type and current locally proved expand relationship. | `B_x(g)` |
| `AxisTransformKind.EXPAND_DIMS` with exact `AxisTransformAttrs` | Input/output exact same floating type and recorded normalized result axis. | `g.squeeze(axis)` |
| `AxisTransformKind.SQUEEZE` with exact `AxisTransformAttrs` | Input/output exact same floating type and recorded normalized input axis. | `g.expandDims(axis)` |
| `AxisTransformKind.PERMUTE` with exact `PermutationAttrs` | Input/output exact same floating type and a complete [permutation](../../../../glossary.md#permutation). | `g.permute(inversePermutation)` |

`O_y` and `Z_y` are the [compiler](../../../../glossary.md#term-compiler)'s explicit exact typed logical one and positive-zero expressions
expanded to `y.shape()`. They are not [host](../../../../glossary.md#term-host)-storage values, [eager](../../../../glossary.md#term-eager-evaluation) factory [constants](../../../../glossary.md#term-constant), descriptor
[inference](../../../../glossary.md#term-inference), labels, or scalar literals hidden from [capture](../../../../glossary.md#term-capture-meanings).

The formulas may reuse exact forward [Tensor](../../../../glossary.md#tensor) wrappers, including `producer.output(outputIndex)`.
They must never reconstruct an equal output wrapper from a descriptor. The selected first matrix
does not need a hidden auxiliary output, but the traversal and capture contracts must retain every
output role so later named rule families can consume [dropout](../../../../glossary.md#dropout--inverted-dropout) [masks](../../../../glossary.md#term-boolean-mask) and saved [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[normalization](../../../../glossary.md#term-normalization)
statistics without another pipeline redesign.

### Explicitly deferred and non-[differentiable](../../../../glossary.md#derivative-and-chain-rule) work

[Preflight](../../../../glossary.md#term-preflight) rejects every [operation](../../../../glossary.md#operation) or variant not listed in `SUPPORTED_0004`; it does not fall
through to a generic rule. The following [ownership](../../../../glossary.md#term-ownership-meanings) split is explicit:

| Category | Decision |
|---|---|
| Mixed-floating [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) arithmetic or WHERE branch promotion | Deferred. Backward would require a selected cross-floating [cotangent](../../../../glossary.md#cotangent) conversion. |
| Cross-floating `CAST` | Deferred to [Compiler](../../../../glossary.md#term-compiler) 0004B's explicit conversion policy. Floating-to-integral, integral-to-floating, integral, and [BOOL](../../../../glossary.md#data-type--datatype) roles are non-differentiable. |
| Binary/[scalar](../../../../glossary.md#term-scalar) `DIV` and `POW`; `RECIPROCAL`, [`LOG`](../../../../glossary.md#term-log), `LOG1P`, `SQRT`, `RSQRT` | Deferred to 0004B because zero, invalid-domain, singular, [infinity](../../../../glossary.md#term-infinity), [NaN](../../../../glossary.md#term-nan), and exceptional-value [derivative](../../../../glossary.md#term-derivative) behavior is unresolved. |
| Binary/scalar `MIN`/`MAX`, `CLAMP`, `ABS`, `RELU`, `FLOOR`, `CEIL`, `SIGN` | Deferred to 0004B because tie, endpoint, discontinuity, [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero), NaN, or subgradient behavior is unresolved. |
| [`ERF`](../../../../glossary.md#term-erf), [GELU](../../../../glossary.md#gaussian-error-linear-unit--gelu) variants, `SILU`, and other exact-composition [unary](../../../../glossary.md#term-unary) additions | Deferred to 0004A; this task does not expand its closed core [matrix](../../../../glossary.md#term-matrix-meanings) or select coefficient handling. |
| SUM with `MaskedReductionAttrs` or `SumToShapeAttrs`; [MEAN](../../../../glossary.md#term-mean), PROD, reduction [extrema](../../../../glossary.md#saturation-pole-and-extremum), advanced/statistical/[norm](../../../../glossary.md#term-norm) reductions, [softmax](../../../../glossary.md#softmax--log-softmax)/[log-softmax](../../../../glossary.md#softmax--log-softmax), and [normalization](../../../../glossary.md#term-normalization) | Deferred to 0004A when policy-free and otherwise 0004B. Empty-domain, tie, zero-norm, singular, and exceptional policies are not selected here. |
| `CUM_PROD` | Deferred. This task selects no product special-value derivative behavior. |
| [Matrix multiplication](../../../../glossary.md#matrix-multiplication--matmul), attention, [convolution](../../../../glossary.md#term-channels-first-convolution-layers), [pooling](../../../../glossary.md#term-pooling), [losses](../../../../glossary.md#loss), indexing/scatter, slicing/pad/tile/composition/windows, ordering/top-k, [dropout](../../../../glossary.md#dropout--inverted-dropout), and [batch normalization](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training) | Deferred to later exact-rule or policy tasks. Current public primitives and [canonical](../../../../glossary.md#term-canonical-form) auxiliary outputs remain evidence, not authorization to widen 0004. |
| Comparisons, BOOL logic and classification, BOOL reductions, arg-extrema/index outputs, one-hot indices, and [graph RNG state](../../../../glossary.md#graph-rng-state--graphrngstate) | Non-differentiable roles. They may occur on an unselected route or serve as fixed conditions, but no cotangent propagates through them. |
| Unknown/custom [`OperationKind`](../../../../glossary.md#operationkind), wrong attributes class, wrong [cardinality](../../../../glossary.md#term-cardinality), missing canonical output, or [descriptor](../../../../glossary.md#term-descriptor) contradiction | Deterministic preflight failure. |

Compiler 0004A may add only separately specified exact-composition formulas after this pipeline is
complete. Compiler 0004B must select explicit derivative policies before adding any
tie/discontinuity/singularity/cross-floating-dependent rule. Neither follow-up may silently widen
this matrix.

### Named [gradient](../../../../glossary.md#term-gradient) rules and deterministic reverse accumulation

Add exactly these named stateless package-private rule [owners](../../../../glossary.md#term-owner):

- `ElementwiseGradientRules` — binary/scalar arithmetic, WHERE, same-type cast, and the selected
  unary formulas;<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [scalar glossary](../../../../glossary.md#term-scalar); [unary glossary](../../../../glossary.md#term-unary))</span>
- `ReductionGradientRules` — ordinary SUM and CUM_SUM; and
- `LayoutGradientRules` — [contiguous](../../../../glossary.md#term-contiguous), reshape, expand, [rank](../../../../glossary.md#term-rank) edits, and [permutation](../../../../glossary.md#permutation).

[Dispatch](../../../../glossary.md#term-dispatch) is by typed [`OperationKind`](../../../../glossary.md#operationkind)/attributes tests and exhaustive closed switches. Do not use
strings, reflection, service loading, a public or [mutable](../../../../glossary.md#term-mutable) [registry](../../../../glossary.md#term-registry), `Map<OperationKind, ...>`,
generic symbolic algebra, direct [`Operation`](../../../../glossary.md#operation)/`CompiledNode` construction, or [backend](../../../../glossary.md#backend) dispatch.
Rules call only current public [`Tensor`](../../../../glossary.md#tensor) methods. They do not perform [graph capture](../../../../glossary.md#capture-graph-ports-and-reconstruction), [inference](../../../../glossary.md#term-inference),
numerical execution, storage access, or [lowering](../../../../glossary.md#lowering).

`FirstOrderAutograd` owns one deterministic reverse traversal over the successful [preflight](../../../../glossary.md#term-preflight) plan.
During one request it may retain only ephemeral [identity](../../../../glossary.md#term-identity-meanings)-based maps for:

- exact Tensor to ordered incoming contribution list;
- exact Tensor to accumulated gradient Tensor; and
- request-local generated zero/one leaves and their explicit splat [bindings](../../../../glossary.md#term-binding).

The traversal starts with the implicit [seed](../../../../glossary.md#term-seed) contribution to the [objective](../../../../glossary.md#term-objective). It processes [producers](../../../../glossary.md#term-producer)
in deterministic reverse postorder after all downstream contributions for the selected output
role are known. For each processed role it accumulates contributions with left-associated
ordinary `Tensor.add` in insertion order. It visits [differentiable](../../../../glossary.md#derivative-and-chain-rule) input positions in operation
order and appends each contribution immediately, so a repeated operand receives repeated
contributions in position order. Consumer ordering follows the deterministic original [producer](../../../../glossary.md#term-producer)
postorder. It does not sort by Tensor ID, kind name, [hash](../../../../glossary.md#term-hash), label, or [structural equality](../../../../glossary.md#term-structural-equality).

Distinct [targets](../../../../glossary.md#term-target) may resolve to the exact same accumulated gradient Tensor. The expansion returns
one ordered target-to-gradient role per request target and does not manufacture identity,
reshape, add-zero, or copy expressions to make roles distinct.

All maps and Tensor references used for backward construction are discarded after combined
[capture](../../../../glossary.md#term-capture-meanings). They
are not stored in `Tensor`, `TensorProvenance`, [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel),
`GraphCompilation`, [runtime](../../../../glossary.md#term-runtime) state, or a global [cache](../../../../glossary.md#term-cache).

### [Derivative](../../../../glossary.md#term-derivative) [constants](../../../../glossary.md#term-constant)

Generated [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) and positive one are exact [BFLOAT16](../../../../glossary.md#term-bfloat16), [FLOAT32](../../../../glossary.md#term-float32), or [FLOAT64](../../../../glossary.md#term-float64) values matching
the formula type. Each base constant is:

- a [provenance](../../../../glossary.md#provenance)-free [Tensor](../../../../glossary.md#tensor) made with public `TensorFactory.create`;
- [rank](../../../../glossary.md#term-rank) zero with unresolved [layout](../../../../glossary.md#layout) and `requiresGrad == false`;
- free of label and [host storage](../../../../glossary.md#host-storage); and
- bound explicitly through `CompileTimeConstantGraph.Binding` and `Splat`.

Within one expansion, a request-local helper may reuse at most one base zero and one base one per
floating [data type](../../../../glossary.md#data-type--datatype), in deterministic first-use order. [Shape](../../../../glossary.md#shape)-specific constants are public
`expand(Shape)` expressions from those base leaves. Equal caller-supplied and generated splats are
not inferred to be the same source. Merge forward ingress [bindings](../../../../glossary.md#term-binding) first in caller order and
generated derivative bindings second in deterministic creation order, rejecting an [identity](../../../../glossary.md#term-identity-meanings)
collision.

No constant is inferred from factory history, host storage, [descriptor](../../../../glossary.md#term-descriptor)/layout facts, label,
provenance absence, Shape, zero [extent](../../../../glossary.md#term-extent), or numerical [evaluation](../../../../glossary.md#term-evaluation). The [compiler](../../../../glossary.md#term-compiler) reads no Tensor
[payload](../../../../glossary.md#artifact-envelope-and-atomic-publication).

### One phase-aware combined [capture](../../../../glossary.md#term-capture-meanings)

Extend `GraphCapture` with one package-private combined entry that receives:

- ordered forward outputs;
- ordered [target](../../../../glossary.md#term-target)/[gradient](../../../../glossary.md#term-gradient) [Tensor](../../../../glossary.md#tensor) roles;
- the [identity](../../../../glossary.md#term-identity-meanings)-based inventory of every original forward [producer](../../../../glossary.md#term-producer);
- the merged explicit [constant](../../../../glossary.md#term-constant) ingress; and
- no captured [graph](../../../../glossary.md#graph) or placeholder Tensor.

The combined traversal is iterative and assigns each [`NodeId`](../../../../glossary.md#nodeid) and [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) exactly once in
deterministic input-first/depth-first postorder. Every declared producer output position becomes
a [graph value](../../../../glossary.md#graph-value), including hidden or [opaque](../../../../glossary.md#term-opaque) positions. A producer whose exact identity is in the
original inventory receives `GraphPhase.FORWARD`; every generated producer receives
`GraphPhase.BACKWARD`. [Leaf](../../../../glossary.md#term-leaf) values have no phase. A positional `backwardStartIndex` is forbidden.

The graph-output boundary is:

1. every distinct requested forward value in forward request order; then
2. each gradient value not already present, at its first target-role [occurrence](../../../../glossary.md#term-occurrence).

Duplicate gradient roots and a gradient already present at the forward boundary are not repeated.
The combined capture retains, for each target role, the exact graph-output ordinal to which it
resolved. This positional role [metadata](../../../../glossary.md#term-metadata), not a Tensor retained after capture, remaps roles through
later [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form) and optimization.

Existing `GraphCapture.capture` entries remain package-private and behaviorally compatible for
[Compiler](../../../../glossary.md#term-compiler) 0001/0003B callers and tests. They may delegate to a shared implementation, but they
continue to classify every producer as `FORWARD` and preserve their current duplicate-output
failure contract.

### [Immutable](../../../../glossary.md#term-immutable) mode-neutral [graph](../../../../glossary.md#graph)-stage result

Add package-private immutable `GraphCompilation` containing:

```java
record GraphCompilation(
        CompileMode mode,
        ValidatedGraph validatedGraph,
        List<ValueId> forwardOutputs,
        List<GradientResultRole> gradientResults)
```

`GradientResultRole` is an immutable nested record containing the exact requested [target](../../../../glossary.md#term-target)'s
[`TensorId`](../../../../glossary.md#tensorid) and its final [gradient](../../../../glossary.md#term-gradient) [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot). The list order equals request target order. Distinct
target IDs may name the same gradient value. No [Tensor](../../../../glossary.md#tensor) reference is retained.

The constructor snapshots lists and validates:

- every forward and gradient value exists and is a graph output;
- forward outputs are [identity](../../../../glossary.md#term-identity-meanings)-unique by `ValueId` and preserve the requested prefix;
- the graph output boundary equals the forward prefix followed by stable first-occurrence
  de-duplication of gradient values not already present;<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- no gradient role exists in `FORWARD_ONLY`;
- backward-capable modes have the non-empty role list produced by their request; and
- each role has non-null target and gradient IDs.

`GraphCompilation` is the mode-neutral package-private result of this internal graph stage.
`FORWARD_ONLY` contains a forward-only validated graph, no `GraphPhase.BACKWARD` [nodes](../../../../glossary.md#node), and an
empty `gradientResults` list. `FORWARD_AND_BACKWARD` and `TRAINING_STEP` may contain the combined
forward/[backward graph](../../../../glossary.md#backward-graph) and non-empty target-specific gradient roles.

The exact `ValidatedGraph` retains the final `CompileTimeConstantGraph`, constraints, [constants](../../../../glossary.md#term-constant),
bindable inputs, structural graph, graph boundaries, and per-node phase facts. `GraphCompilation`
is not the later public or cross-package aggregate `CompileArtifacts`: it adds no [publication](../../../../glossary.md#term-publication)
[binding](../../../../glossary.md#term-binding), [publication plan](../../../../glossary.md#publication-plan), logical-memory/[partition](../../../../glossary.md#partition) plan, physical constant plan, diagnostics
bundle, [backend](../../../../glossary.md#backend) choice, schedule, prepared state, or executable [artifact](../../../../glossary.md#term-artifact).

### Mode-selected [graph](../../../../glossary.md#graph) [inference](../../../../glossary.md#term-inference) and exact optimization

The [immutable](../../../../glossary.md#term-immutable) graph captured for the selected mode is the sole optimization unit. In
backward-capable modes that unit is the combined forward/[backward graph](../../../../glossary.md#backward-graph). Preserve this exact
one-shot order:

1. one phase-aware [capture](../../../../glossary.md#term-capture-meanings), combined for backward-capable modes;
2. `CapturedGraphInference.inferAndValidate` on the captured [constant](../../../../glossary.md#term-constant) sidecar;
3. mandatory dense `GraphCanonicalization` and [validation](../../../../glossary.md#term-validation);
4. if optional optimization is disabled, return;
5. existing seven-rule exact arithmetic rewriting once;
6. existing bounded [BOOL](../../../../glossary.md#data-type--datatype)/signed-integral logical-splat folding once;
7. whole-graph sidecar-aware [DCE](../../../../glossary.md#term-dead-code-elimination-dce) once;
8. exact phase-local [CSE](../../../../glossary.md#term-common-subexpression-elimination-cse) once;
9. whole-graph sidecar-aware cleanup DCE once; and
10. return the final validated result, remapping forward/gradient roles by preserved graph-output
    ordinal after every ID rebuild.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient))</span>

Every changed candidate is revalidated through [Compiler](../../../../glossary.md#term-compiler) 0002 before the next pass consumes it.
An unchanged helper result is not redundantly validated. No fixed point or extra scan is added.

The completed package-private helper names may remain `Forward*` to keep this already large
pipeline migration bounded, but their Javadocs and behavior must no longer use the name as a
phase filter:

- arithmetic rewriting may consider FORWARD or BACKWARD nodes while retaining every existing
  0003A guard: the closed seven rules, exact descriptor equality, <span data-glossary-navigation="true"> ([nodes glossary](../../../../glossary.md#node); [guard glossary](../../../../glossary.md#term-guard); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>`requiresGrad == false`,
  one non-graph-output result, and no new algebra;
- constant folding may consider either phase while retaining the exact 0003B operation/type
  matrix, all graph-output/multi-output exclusions, exact typed evaluation, and bounded work;<span data-glossary-navigation="true"> ([constant folding glossary](../../../../glossary.md#term-constant-folding); [operation glossary](../../../../glossary.md#operation); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- DCE derives liveness only from the complete selected graph output boundary and retained
  dependencies; a BACKWARD node is not automatically live merely because of its phase;<span data-glossary-navigation="true"> ([liveness glossary](../../../../glossary.md#term-liveness))</span>
- CSE may consider both phases, but `GraphPhase` remains in the complete expression key and
  candidates never merge across phases; graph-output producers remain excluded; and<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span>
- constant sidecar roles, graph-output order, complete multi-output occurrences, exact operation
  and descriptor references, and phase facts survive every rebuild.

This task authorizes no new rewrite, floating/[BFLOAT16](../../../../glossary.md#term-bfloat16) [evaluation](../../../../glossary.md#term-evaluation), cast folding, [reassociation](../../../../glossary.md#term-reassociation),
relaxed arithmetic, approximate rule, constant [materialization](../../../../glossary.md#materialization), or cross-phase CSE.

## Out of [scope](../../../../glossary.md#term-scope)

- any public [compiler](../../../../glossary.md#term-compiler), [gradient](../../../../glossary.md#term-gradient), [compile](../../../../glossary.md#compile)-request, compile-result, [publication](../../../../glossary.md#term-publication), or [artifact](../../../../glossary.md#term-artifact) type
- `CompileArtifacts`, `CompileConfig`, public objective/target/seed selection, gradient
  publication bindings, or engine integration<span data-glossary-navigation="true"> ([objective glossary](../../../../glossary.md#term-objective); [target glossary](../../../../glossary.md#term-target); [seed glossary](../../../../glossary.md#term-seed); [bindings glossary](../../../../glossary.md#term-binding); [engine glossary](../../../../glossary.md#term-engine))</span>
- non-scalar objectives, explicit cotangent seeds, Jacobian construction, higher-order
  differentiation, create-graph flags, derivative-order state, or differentiating generated
  gradients<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [cotangent glossary](../../../../glossary.md#cotangent); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [Jacobian glossary](../../../../glossary.md#term-jacobian))</span>
- every [derivative](../../../../glossary.md#term-derivative) rule or attributes variant outside `SUPPORTED_0004`
- selecting a tie, endpoint, discontinuity, singularity, invalid-domain, NaN/infinity,
  zero-count/zero-norm, or cross-floating conversion derivative policy<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [norm glossary](../../../../glossary.md#term-norm))</span>
- [optimizer](../../../../glossary.md#term-optimizer) semantics, optimizer-update [graph](../../../../glossary.md#graph) [nodes](../../../../glossary.md#node), [parameter](../../../../glossary.md#term-parameter) mutation, [`TrainingSession`](../../../../glossary.md#training-session--trainingsession), or
  training execution
- `Tensor.gradient`, `Tensor.backward`, mutable gradient fields, model-owned rules, a runtime
  tape, ThreadLocal compilation state, global caches, registries, service locators, plugins, or
  reflection/string dispatch<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [model glossary](../../../../glossary.md#term-model); [runtime glossary](../../../../glossary.md#term-runtime); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- placeholder [Tensors](../../../../glossary.md#tensor) for [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), conversion from captured graph values back to Tensor
  expressions, a second algebra, or direct compiler construction of model IR nodes for formulas<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- reconstructing a [canonical](../../../../glossary.md#term-canonical-form) [producer](../../../../glossary.md#term-producer) output wrapper, changing [`TensorProducer`](../../../../glossary.md#producer--tensorproducer),
  `TensorFactory`, provenance, Tensor methods, result carriers, model operation semantics, or
  model tests<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [operation glossary](../../../../glossary.md#operation))</span>
- reading or materializing host/device storage, evaluating Tensor values, buffers, planning,
  lowering, backend work, prepare, runtime, engine, execution, or trace events<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [device glossary](../../../../glossary.md#term-device); [buffers glossary](../../../../glossary.md#term-buffer); [planning glossary](../../../../glossary.md#term-planning); [lowering glossary](../../../../glossary.md#lowering); [backend glossary](../../../../glossary.md#backend); [prepare glossary](../../../../glossary.md#prepare); [trace glossary](../../../../glossary.md#trace))</span>
- new arithmetic rewrites, new constant-fold rows, pass iteration, cross-phase CSE, or approximate
  math<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse))</span>
- architecture, ADR, module dependency, Gradle/build, Java-version, architecture-test,
  backend-conformance, or integration-test changes<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- implementing Compiler 0004A, 0004B, 0005, or 0006 or creating their detailed specifications
- reading, copying, adapting, moving, renaming, staging, or treating the six removal-only obsolete
  prototype Java files as design authority; their required deletion is the sole permitted action
  on those paths<span data-glossary-navigation="true"> ([prototype glossary](../../../../glossary.md#term-prototype))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially compiler ownership,
  pre-capture autograd, combined capture, phases, constant facts, validation, and lifecycle<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [capture glossary](../../../../glossary.md#term-capture-meanings); [autograd glossary](../../../../glossary.md#autograd); [constant glossary](../../../../glossary.md#term-constant); [validation glossary](../../../../glossary.md#term-validation); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [ADR 0009](../../../../design/decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [Superseded ADR 0005](../../../../design/decisions/0005-training-combined-forward-backward-graph.md)
- [Autograd strategy](../../../../design/notes/autograd-strategy.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Compiler master plan](../master-plan.md)
- [Compiler 0001](0001-tensor-expression-graph-capture.md)
- [Compiler 0002](0002-captured-graph-inference-and-validation.md)
- [Compiler 0003](0003-canonicalization-and-forward-optimization.md)
- [Compiler 0003A](0003a-exact-arithmetic-rewriting.md)
- [Compiler 0003B](0003b-compile-time-constants-and-constant-folding.md)
- [Model master plan](../../model/master-plan.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model capabilities](../../model/capabilities.md)
- [Adjoint expressibility audit](../../model/adjoint-expressibility-audit.md)<span data-glossary-navigation="true"> ([Adjoint glossary](../../../../glossary.md#term-adjoint))</span>
- [Model contract-closure audit](../../model/model-capability-contract-closure-audit.md)
- [Model 0025](../../model/tasks/0025-canonical-tensor-producer-outputs.md)
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `ARCHITECTURE.md` is authoritative. The accepted pre-capture design already covers this task;
  no architecture or ADR change is expected.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- Compiler owns reverse traversal, derivative-rule selection, accumulation, phase-aware capture,
  validation, and exact combined optimization.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [derivative glossary](../../../../glossary.md#term-derivative); [validation glossary](../../../../glossary.md#term-validation))</span>
- Model owns Tensor operations, producer occurrence identity, canonical output wrappers,
  descriptors, operation vocabulary, and immutable compiled graph values. It owns no derivative
  rule or backward lifecycle.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor); [producer glossary](../../../../glossary.md#term-producer); [occurrence glossary](../../../../glossary.md#term-occurrence); [identity glossary](../../../../glossary.md#term-identity-meanings); [canonical glossary](../../../../glossary.md#term-canonical-form); [descriptors glossary](../../../../glossary.md#term-descriptor); [operation glossary](../../../../glossary.md#operation); [immutable glossary](../../../../glossary.md#term-immutable); [compiled graph glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Compiler formulas use only public Tensor operations and explicit storage-free leaves. Captured
  graph IDs are assigned once, after formulas exist.<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
- Compiler result state is immutable, backend-neutral, and contains no runtime, prepare, concrete
  backend, physical storage, or execution object.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
- Existing dependency direction remains unchanged. No architecture test is required because no
  dependency rule changes.<span data-glossary-navigation="true"> ([dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [architecture test glossary](../../../../glossary.md#term-architecture-test); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- If implementation needs a model API, public compiler API, additional derivative policy,
  different graph phase model, second algebra, or path outside the maximum scope, stop and report
  the exact conflict instead of inventing a contract.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [graph phase glossary](../../../../glossary.md#graph-phase); [scope glossary](../../../../glossary.md#term-scope))</span>

## Package impact

Existing package used:

- `io.github.pho001.synaptik.compiler` — the cohesive package-private [compiler](../../../../glossary.md#term-compiler) front-end boundary.

Packages added or changed:

- No package is added. No public package surface changes.

Type placement:

- `io.github.pho001.synaptik.compiler.AutogradPreflight` — owns the nested package-private
  first-order request, original identity inventory, target-path analysis, and complete preflight
  plan.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [target glossary](../../../../glossary.md#term-target); [preflight glossary](../../../../glossary.md#term-preflight))</span>
- `io.github.pho001.synaptik.compiler.ElementwiseGradientRules` — owns the selected typed
  elementwise formulas.<span data-glossary-navigation="true"> ([elementwise glossary](../../../../glossary.md#term-elementwise))</span>
- `io.github.pho001.synaptik.compiler.ReductionGradientRules` — owns the selected typed reduction
  and scan formulas.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- `io.github.pho001.synaptik.compiler.LayoutGradientRules` — owns the selected typed logical
  layout-transform formulas.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>
- `io.github.pho001.synaptik.compiler.FirstOrderAutograd` — owns reverse traversal,
  identity-keyed contribution accumulation, implicit seed, derivative constants, and ordered
  target/gradient construction results.<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed); [derivative glossary](../../../../glossary.md#term-derivative); [constants glossary](../../../../glossary.md#term-constant); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- `io.github.pho001.synaptik.compiler.GraphCapture` — remains the sole Tensor/producer
  identity-to-graph-ID boundary and gains combined phases, role ordinals, and output
  de-duplication.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [producer glossary](../../../../glossary.md#term-producer); [graph glossary](../../../../glossary.md#graph))</span>
- `io.github.pho001.synaptik.compiler.GraphCompilation` — owns the mode-neutral immutable
  validated graph-stage result, forward boundary, and optional target-specific gradient roles;
  it is distinct from later <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>`CompileArtifacts`.
- `io.github.pho001.synaptik.compiler.GraphCompiler` — owns exact mode routing and the
  capture/infer/optimize/result orchestration without adding a request aggregate.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- Existing `ForwardGraphOptimization`, exact rewriting/folding, DCE, and CSE helpers remain
  package-private and are generalized only as specified for a backward-capable combined graph.<span data-glossary-navigation="true"> ([DCE glossary](../../../../glossary.md#term-dead-code-elimination-dce); [CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse))</span>

Tests mirror the production package to exercise package-private contracts.

## Affected files

Expected production files:

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradPreflight.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ElementwiseGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ReductionGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/LayoutGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/FirstOrderAutograd.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCompilation.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCompiler.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCapture.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ForwardGraphOptimization.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ForwardExactArithmeticRewriting.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ForwardConstantFolding.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ForwardDeadCodeElimination.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ForwardCommonSubexpressionElimination.java`

Expected test files:

- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/AutogradPreflightTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GradientRulesTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/FirstOrderAutogradTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GraphCompilerTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GraphCaptureTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/ForwardGraphOptimizationTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/ForwardExactArithmeticRewritingTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/ForwardConstantFoldingTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/ForwardDeadCodeEliminationTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/ForwardCommonSubexpressionEliminationTest.java`

Expected documentation and [planning](../../../../glossary.md#term-planning) files:

- `docs/api/compile-api.md`
- `docs/api/tensor-api.md`
- `docs/api/training-api.md`
- `docs/design/notes/autograd-strategy.md`
- `docs/user-guide/autograd.md`
- `docs/glossary.md`
- `docs/planning/modules/compiler/tasks/0004-compiler-owned-pre-capture-autograd-and-combined-graph-compilation.md`
- `docs/planning/modules/compiler/master-plan.md`
- `docs/planning/roadmap.md`

Required removal-only pre-existing untracked paths:

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradExpansion.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradGraph.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/BackwardGraphBuilder.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ElementwiseAdjoints.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/IndexingRandomAdjoints.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ReductionLayoutAdjoints.java`

## Maximum [scope](../../../../glossary.md#term-scope)

This task has a 38 touched-path ceiling:

- at most the 32 expected tracked create/modify paths listed above; plus
- exactly the six pre-existing untracked production-source paths listed under required
  removal-only paths, which must only be verified and deleted.

The higher-than-normal path count is justified by one indivisible [invariant](../../../../glossary.md#term-invariant) explicitly assigned to
[Compiler](../../../../glossary.md#term-compiler) 0004 by the roadmap: [preflight](../../../../glossary.md#term-preflight) must cover the complete selected backward [slice](../../../../glossary.md#slice) before any
formula [Tensor](../../../../glossary.md#tensor) exists; formula results and generated splats must enter the same [identity](../../../../glossary.md#term-identity-meanings)-aware
[capture](../../../../glossary.md#term-capture-meanings); combined output roles must survive every ID-rebuilding pass; and the completed
forward-only [optimizer](../../../../glossary.md#term-optimizer) must become exact for a backward-capable combined [graph](../../../../glossary.md#graph) before a result is
returned.
Splitting combined optimization into a 0004C would leave an invalid intermediate contract and is
explicitly rejected by the compiler master plan and roadmap. The task remains bounded by one
[module](../../../../glossary.md#term-module), one closed rule [matrix](../../../../glossary.md#term-matrix-meanings), package-private types, and the exact path inventory above.

If another tracked file, type, test, documentation path, [operation](../../../../glossary.md#operation) rule, [model](../../../../glossary.md#term-model) change, or build
change is required, stop and propose a follow-up or task-spec revision before implementation.
If any cleanup path would require content modification, migration, staging, or preservation rather
than deletion, stop instead of exceeding its removal-only authorization.

## Acceptance criteria

### Request and [preflight](../../../../glossary.md#term-preflight)

- The package-private compile entry validates the exact mode/request matrix and preserves
  declaration-order null/failure behavior.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- The first-order request supports exactly one scalar floating objective, one implicit exact
  typed unit seed, and an ordered identity-unique non-empty target list.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [objective glossary](../../../../glossary.md#term-objective); [seed glossary](../../../../glossary.md#term-seed); [identity glossary](../../../../glossary.md#term-identity-meanings); [target glossary](../../../../glossary.md#term-target))</span>
- Inventory covers every original forward producer and every canonical output role by exact
  object identity without recursion.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity))</span><span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Target reachability prunes branches unrelated to every requested target and stops
  non-differentiable roles.<span data-glossary-navigation="true"> ([differentiable glossary](../../../../glossary.md#derivative-and-chain-rule))</span>
- The complete selected slice is preflighted before the first seed, constant, or formula Tensor
  is created.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice); [constant glossary](../../../../glossary.md#term-constant); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Every unsupported kind/attrs/type/role/policy failure reports deterministic contextual
  diagnostics and consumes no new Tensor ID.
- Unknown [operation](../../../../glossary.md#operation) families [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation); no fallback or [registry](../../../../glossary.md#term-registry) exists.

### Rule [matrix](../../../../glossary.md#term-matrix-meanings) and accumulation

- Tests exercise every `SUPPORTED_0004` row, attributes variant, input role, exact same-type guard,
  broadcasting reversal, rank restoration, inverse permutation, and CUM_SUM mode combination.<span data-glossary-navigation="true"> ([guard glossary](../../../../glossary.md#term-guard); [broadcasting glossary](../../../../glossary.md#broadcasting); [rank glossary](../../../../glossary.md#term-rank); [permutation glossary](../../../../glossary.md#permutation))</span>
- Tests cover [BFLOAT16](../../../../glossary.md#term-bfloat16), [FLOAT32](../../../../glossary.md#term-float32), and [FLOAT64](../../../../glossary.md#term-float64) [seed](../../../../glossary.md#term-seed)/zero/one bit/type behavior where applicable.
- Mixed-floating binary/WHERE, cross-floating cast, SumToShape-as-forward-SUM, masked SUM, and
  representative 0004A/0004B/non-differentiable families fail or stop exactly as specified.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [differentiable glossary](../../../../glossary.md#derivative-and-chain-rule))</span>
- Repeated operands receive repeated contributions in input-position order.
- Shared subexpressions and multiple consumers accumulate in deterministic reverse traversal and
  left-associated `Tensor.add` order.
- A [target](../../../../glossary.md#term-target) equal to the [objective](../../../../glossary.md#term-objective) receives the implicit unit seed.
- An intermediate target does not stop propagation needed by an upstream target.
- Distinct targets can return the exact same [gradient](../../../../glossary.md#term-gradient) [Tensor](../../../../glossary.md#tensor) without an [identity](../../../../glossary.md#term-identity-meanings) [node](../../../../glossary.md#node).
- Rule [owners](../../../../glossary.md#term-owner) are exactly the three named `*GradientRules` components and use only current public
  Tensor methods.

### [Constants](../../../../glossary.md#term-constant), [capture](../../../../glossary.md#term-capture-meanings), and result

- Every derivative zero/one base is storage-free, non-gradient, provenance-free, explicitly
  registered as an exact logical splat, and never inferred.<span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative); [provenance glossary](../../../../glossary.md#provenance); [logical splat glossary](../../../../glossary.md#logical-splat))</span>
- Forward ingress order and generated constant order are stable, and all fixed sources are absent
  from `bindableInputs()`.
- One combined capture for a backward-capable request assigns graph-local IDs once, preserves
  every producer output slot, classifies original producers FORWARD and generated producers
  BACKWARD by exact identity, and retains no Tensor/producer reference in the immutable final
  result.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [producer glossary](../../../../glossary.md#term-producer); [Tensor glossary](../../../../glossary.md#tensor); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Forward outputs preserve request order. Gradient graph outputs use stable first-occurrence
  de-duplication, while every target retains its own ordered role.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Multiple [targets](../../../../glossary.md#term-target) may map to one final [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), and no manufactured [identity](../../../../glossary.md#term-identity-meanings)/copy [node](../../../../glossary.md#node) appears.
- `GraphCompilation` retains exact mode, final `ValidatedGraph`, forward boundary,
  target-[`TensorId`](../../../../glossary.md#tensorid)/[gradient](../../../../glossary.md#term-gradient)-`ValueId` roles, constants, bindable inputs, constraints, and phase
  facts through its components.
- `GraphCompilation` remains package-private, mode-neutral graph-stage state and is neither named
  nor shaped as the later `CompileArtifacts` aggregate.
- `FORWARD_ONLY` creates no [seed](../../../../glossary.md#term-seed), gradient constant, gradient role, or BACKWARD node, and its
  `GraphCompilation.gradientResults()` is empty.
- `FORWARD_AND_BACKWARD` and `TRAINING_STEP` produce equal graph semantics for equal requests; the
  latter contains no optimizer work.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer))</span>

### Combined optimization

- Disabled optional optimization still canonicalizes and validates the graph selected by the mode
  and remaps every applicable result role correctly.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
- Enabled optimization uses exactly rewrite, fold, whole-graph DCE, phase-local CSE, whole-graph
  DCE once each in the specified order.<span data-glossary-navigation="true"> ([DCE glossary](../../../../glossary.md#term-dead-code-elimination-dce); [CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse))</span>
- Existing exact rewrite and fold matrices are unchanged and may consider either phase only under
  all existing non-phase guards.<span data-glossary-navigation="true"> ([matrices glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Whole-graph DCE removes an unused BACKWARD occurrence and keeps every dependency of a live
  forward or gradient output, including multi-output producers and constant roles.<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [dependency glossary](../../../../glossary.md#term-dependency); [gradient glossary](../../../../glossary.md#term-gradient); [constant glossary](../../../../glossary.md#term-constant))</span>
- CSE merges equal eligible occurrences within FORWARD and within BACKWARD, never across phases,
  and never merges graph-output producers.
- Every changed candidate is revalidated through [Compiler](../../../../glossary.md#term-compiler) 0002; unchanged candidates are not.
- Graph output order, role ordinals, exact constants, constraints, operations, descriptors,
  phases, and immutable snapshots survive all rebuilds.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- No new algebra, folding row, fixed point, approximate math, storage read, or physical
  materialization is present.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>

### Boundaries, documentation, and completion

- No public declaration is added to `modules/compiler`; reflection/source tests lock the intended
  package-private surface.
- `GraphCompiler` exposes only the specified direct package-private [parameter](../../../../glossary.md#term-parameter) list, returns
  `GraphCompilation`, and adds no request aggregate, alternate entry/result owner, or alias with
  the rejected combined-prefixed names.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- No model, config, planning, trace, runtime, prepare, backend, engine, Gradle, architecture,
  conformance, or integration source/test changes occur.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [planning glossary](../../../../glossary.md#term-planning); [trace glossary](../../../../glossary.md#trace); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [engine glossary](../../../../glossary.md#term-engine); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- The six obsolete prototype paths were confirmed present and untracked before implementation,
  deleted before any implementation Java edit or compiler invocation, and are absent from the
  source tree at final status.<span data-glossary-navigation="true"> ([prototype glossary](../../../../glossary.md#term-prototype); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- No prototype content was copied, adapted, renamed, moved, staged, cited as design evidence, or
  reintroduced under another path.
- All affected production Javadocs fully document ownership, inputs, results, failure modes,
  identity, ordering, immutability, constants, phases, and no-side-effect boundaries.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [identity glossary](../../../../glossary.md#term-identity-meanings); [constants glossary](../../../../glossary.md#term-constant))</span>
- Compile API, Tensor API, Training API, autograd strategy, user guide, glossary, task, compiler
  master plan, and roadmap consistently distinguish current implementation from planned public
  publication, policy extensions, artifacts, and higher derivatives.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Tensor glossary](../../../../glossary.md#tensor); [autograd glossary](../../../../glossary.md#autograd); [publication glossary](../../../../glossary.md#term-publication); [artifacts glossary](../../../../glossary.md#term-artifact); [derivatives glossary](../../../../glossary.md#term-derivative))</span>
- Documentation explicitly records reviewed no-change conclusions for `ARCHITECTURE.md`, focused
  architecture documents, ADRs, model/config APIs, dependency/architecture tests,
  backend-conformance, integration tests, Gradle, and other modules.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [modules glossary](../../../../glossary.md#term-module))</span>
- A separate clean-context documentation-focused agent pass finalizes affected Javadocs,
  explanatory documentation, glossary, planning status/evidence, terminology, links, examples,
  and no-change conclusions in the same overall change without repeating successful Java tests
  unless it changes executable behavior or records a concrete reason.

## Tests / [validation](../../../../glossary.md#term-validation)

Before the first implementation edit, record:

```bash
git status --short -- \
  modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradExpansion.java \
  modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradGraph.java \
  modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/BackwardGraphBuilder.java \
  modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ElementwiseAdjoints.java \
  modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/IndexingRandomAdjoints.java \
  modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ReductionLayoutAdjoints.java
```

The recorded output must contain exactly six `??` entries for those exact paths. Delete them
without reading or migrating their contents. Before the first [compiler](../../../../glossary.md#term-compiler) invocation and again at
final status, verify:

```bash
for path in \
  modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradExpansion.java \
  modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradGraph.java \
  modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/BackwardGraphBuilder.java \
  modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ElementwiseAdjoints.java \
  modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/IndexingRandomAdjoints.java \
  modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ReductionLayoutAdjoints.java
do
  test ! -e "$path"
done
```

Implementation-focused validation:

```bash
./gradlew :modules:compiler:test
git diff --check
```

The final compiler suite must include focused evidence for:

- zero-ID side effects across every [preflight](../../../../glossary.md#term-preflight) rejection class;
- deep iterative inventory, deep reverse traversal, and deep backward-capable combined [capture](../../../../glossary.md#term-capture-meanings);
- complete supported/deferred/non-[differentiable](../../../../glossary.md#derivative-and-chain-rule) [matrix](../../../../glossary.md#term-matrix-meanings) locks;
- repeated operands, shared consumers, intermediate and duplicate-[gradient](../../../../glossary.md#term-gradient) [targets](../../../../glossary.md#term-target);
- exact hidden-output inventory and [producer](../../../../glossary.md#term-producer) [identity](../../../../glossary.md#term-identity-meanings) classification;
- generated splat bits/types, source roles, and no storage [inference](../../../../glossary.md#term-inference);
- all three modes and mode/request mismatch ordering;
- exact `GraphCompiler`/`GraphCompilation` package-private signatures and absence of a request
  aggregate or alternate entry/result type;<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>
- per-[node](../../../../glossary.md#node) phases and [graph](../../../../glossary.md#graph)-output/role de-duplication;
- inference/constraint regeneration after backward-capable combined capture and every changed
  pass;
- rewrite/fold phase eligibility without matrix [widening](../../../../glossary.md#term-widening);
- whole-graph [DCE](../../../../glossary.md#term-dead-code-elimination-dce) and phase-local [CSE](../../../../glossary.md#term-common-subexpression-elimination-cse); and
- immutability, deterministic IDs/order, no mutation, and no public compiler declarations.

Documentation-focused pass:

```bash
./gradlew :modules:compiler:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
```

If the repository Markdown validator is not present, recreate or use the established equivalent
that checks every local Markdown link and anchor, balanced fences, final newlines, and trailing
whitespace, and record the exact command.

The documentation pass also inspects generated compiler [Javadoc](../../../../glossary.md#term-javadoc), source/[API](../../../../glossary.md#term-application-programming-interface-api) terminology, exact
task/master/roadmap statuses, the 38 touched-path ceiling (32 tracked create/modify plus six
removal-only), the initial cleanup evidence, final [prototype](../../../../glossary.md#term-prototype) absence, and the no-change
boundaries.

Repository-wide validation is deferred to the compiler transformation/[autograd](../../../../glossary.md#autograd) capability
[checkpoint](../../../../glossary.md#term-checkpoint) after Compiler 0004B and to [CI](../../../../glossary.md#term-continuous-integration-ci). This task changes one [module](../../../../glossary.md#term-module) and no [dependency](../../../../glossary.md#term-dependency),
architecture, build, or public cross-module contract.

## Dependencies

- [Model](../../../../glossary.md#term-model) 0025 — Complete; supplies exact [canonical](../../../../glossary.md#term-canonical-form) [Tensor](../../../../glossary.md#tensor) wrappers for every [producer](../../../../glossary.md#term-producer) output.
- [Compiler](../../../../glossary.md#term-compiler) 0001 — Complete; supplies deterministic Tensor-expression [graph capture](../../../../glossary.md#capture-graph-ports-and-reconstruction).
- Compiler 0002 — Complete; supplies authoritative captured-[graph](../../../../glossary.md#graph) [inference](../../../../glossary.md#term-inference) and [validation](../../../../glossary.md#term-validation).
- Compiler 0003 — Complete; supplies [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form) and bounded optimization orchestration.
- Compiler 0003A — Complete; supplies the closed seven-rule exact rewriting [matrix](../../../../glossary.md#term-matrix-meanings).
- Compiler 0003B — Complete; supplies explicit logical-splat ingress, exact bounded folding, and
  source-role sidecars.
- Config 0002 — Complete; supplies [`CompileMode`](../../../../glossary.md#compile-mode--compilemode) and [`GraphOptimizationConfig`](../../../../glossary.md#graph-optimization-configuration--graphoptimizationconfig).
- [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0009 — Accepted; selects compiler-owned [pre-capture Tensor-expression autograd](../../../../glossary.md#pre-capture-tensor-expression-autograd).
- The current public Tensor operations named in `SUPPORTED_0004` — Complete and source-verified.

No unresolved [dependency](../../../../glossary.md#term-dependency) blocks this task.

## Follow-up tasks

- Compiler 0004A — add separately specified policy-free exact-composition gradient rules without
  changing the one-capture lifecycle.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [gradient glossary](../../../../glossary.md#term-gradient); [capture glossary](../../../../glossary.md#term-capture-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Compiler 0004B — select explicit tie, discontinuity, singularity, exceptional-value, empty
  domain, and cross-floating conversion policies before adding policy-dependent rules.
- Compiler 0005 — add publication, planning orchestration, and immutable public/cross-package
  compile artifacts while transporting the exact mode-selected graph-stage result, applicable
  gradient roles, constraints, and constant facts.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [planning glossary](../../../../glossary.md#term-planning); [immutable glossary](../../../../glossary.md#term-immutable); [compile artifacts glossary](../../../../glossary.md#compile-artifacts); [graph glossary](../../../../glossary.md#graph); [constant glossary](../../../../glossary.md#term-constant))</span>
- Compiler 0006 — define public functional objective/target/seed requests and higher-order
  derivative lifecycle/order representation.<span data-glossary-navigation="true"> ([objective glossary](../../../../glossary.md#term-objective); [target glossary](../../../../glossary.md#term-target); [seed glossary](../../../../glossary.md#term-seed); [derivative glossary](../../../../glossary.md#term-derivative))</span>

Do not create detailed specifications for these tasks until the frontier advances.

## Architecture impact

Expected impact: None.

The authoritative architecture, focused architecture documents, and accepted [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0009 already
specify this [ownership](../../../../glossary.md#term-ownership-meanings) and [lifecycle](../../../../glossary.md#lifecycle). Dependencies and [graph phase](../../../../glossary.md#graph-phase) vocabulary do not change.

If implementation requires architecture, ADR, [dependency](../../../../glossary.md#term-dependency), [model](../../../../glossary.md#term-model) [graph](../../../../glossary.md#graph), phase, public [API](../../../../glossary.md#term-application-programming-interface-api), or
cross-[module](../../../../glossary.md#term-module) changes, stop and report the issue.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository.

Read:
- AGENTS.md
- ARCHITECTURE.md
- docs/planning/planning-guide.md
- docs/planning/roadmap.md
- docs/planning/modules/compiler/master-plan.md
- docs/planning/modules/compiler/tasks/0004-compiler-owned-pre-capture-autograd-and-combined-graph-compilation.md

Implement Compiler task 0004 exactly as specified. Before any implementation Java edit, verify the
six exact obsolete compiler-source prototype paths are present and untracked, then delete them as
the first cleanup step. Do not read, copy, adapt, rename, move, stage, preserve, or treat their
contents as design authority, and verify their absence before compiling or testing. Do not
implement out-of-scope rules or public APIs. Use the exact planned package-private names
`GraphCompiler`, `GraphCompilation`, and `GraphCompilerTest`; retain the specified compile
parameter list and do not add a request aggregate or another entry/result type. Keep
`GraphCompilation` distinct from later `CompileArtifacts`. Stop and report any prototype-status,
architecture, policy, or maximum-scope conflict. Do not commit or push.

After executable code and the compiler module test pass are stable, hand the same working tree and
recorded test evidence to a separate clean-context documentation-focused agent. That pass must
follow docs/developer-guide/documentation-rules.md, finalize all affected Javadocs, API/design/user
documentation, glossary, planning status/evidence, and no-change conclusions in this same overall
change, and run compiler Javadoc plus documentation validation. It must not repeat the successful
Java suite unless it changes executable behavior or records a concrete reason.

At the end, update this task with local decisions, limitations, validation evidence,
implementation notes, completion summary, and final status. Mark Complete only after every
acceptance criterion and the documentation pass succeed.
```

## Local decisions

- Keep the general entry and [graph](../../../../glossary.md#graph)-stage result package-private as the specified `GraphCompiler`
  and `GraphCompilation`; retain the direct compile parameter list rather than adding a request
  aggregate, facade, registry, or alternate result type.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [parameter glossary](../../../../glossary.md#term-parameter); [facade glossary](../../../../glossary.md#term-facade); [registry glossary](../../../../glossary.md#term-registry))</span>
- Separate a Tensor-allocation-free inventory/preflight plan from formula expansion. Preflight
  uses the closed <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [preflight glossary](../../../../glossary.md#term-preflight))</span>`SUPPORTED_0004` matrix, and successful expansion uses only ordinary public
  Tensor operations plus explicitly registered derivative splats.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- Preserve deterministic identity semantics with compile-local identity maps and ordered
  contribution lists. Accumulate repeated and shared contributions with ordinary <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>`Tensor.add`.
- Capture requested forward outputs and gradient roots in one phase-aware traversal. Preserve
  every requested target role while de-duplicating only the combined graph output boundary by
  stable first occurrence.<span data-glossary-navigation="true"> ([Capture glossary](../../../../glossary.md#term-capture-meanings); [gradient glossary](../../../../glossary.md#term-gradient); [target glossary](../../../../glossary.md#term-target); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Run inference/validation, canonicalization, exact rewrite/fold, whole-graph DCE, phase-local
  CSE, and final DCE as one bounded mode-selected graph pipeline. Revalidate each changed
  candidate.<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run); [inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [DCE glossary](../../../../glossary.md#term-dead-code-elimination-dce); [CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse))</span>

## Known limitations

- The supported [derivative](../../../../glossary.md#term-derivative) [matrix](../../../../glossary.md#term-matrix-meanings) is exactly `SUPPORTED_0004`; all other operations, output
  roles, attribute variants, mixed-floating cotangent conversions, and unselected derivative
  policies fail closed.<span data-glossary-navigation="true"> ([cotangent glossary](../../../../glossary.md#cotangent); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- Requests are limited to one requested scalar floating objective, ordered identity-unique
  targets, one implicit exact typed rank-zero unit seed, and first-order derivatives.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [objective glossary](../../../../glossary.md#term-objective); [identity glossary](../../../../glossary.md#term-identity-meanings); [targets glossary](../../../../glossary.md#term-target); [rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [seed glossary](../../../../glossary.md#term-seed))</span>
- `TRAINING_STEP` currently constructs the same combined [graph](../../../../glossary.md#graph) as `FORWARD_AND_BACKWARD`; it adds
  no optimizer update, training state, prepare behavior, backend work, or runtime execution.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Public [compile](../../../../glossary.md#compile) requests, [publication](../../../../glossary.md#term-publication)/[planning](../../../../glossary.md#term-planning) orchestration, `CompileArtifacts`, explicit seeds,
  higher derivatives, and policy-dependent formulas remain with Compiler 0004A, 0004B, 0005, and
  0006 as specified.<span data-glossary-navigation="true"> ([seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [Compiler glossary](../../../../glossary.md#term-compiler))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Before any implementation Java read or edit, the implementation context ran the exact
  six-path `git status --short -- ...` command and observed exactly six `??` entries. It deleted
  those obsolete prototypes without reading, copying, adapting, moving, or staging their
  contents. The exact absence loop passed before the first Gradle invocation and again during the
  final documentation audit.<span data-glossary-navigation="true"> ([prototypes glossary](../../../../glossary.md#term-prototype); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- The focused ten-class [compiler](../../../../glossary.md#term-compiler) suite passed. The final
  `./gradlew :modules:compiler:test` run passed 118 tests with zero failures, errors, or skips;
  Gradle reported 13 actionable tasks, one executed and 12 up-to-date.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- The separate clean-context documentation pass
  `/root/implement_compiler_0004_precapture/compiler_0004_docs` changed no executable Java
  tokens and deliberately did not repeat the successful Java suite. Comment-stripped hashes for
  all 13 authorized production Java paths matched the hashes recorded before the Javadoc pass.<span data-glossary-navigation="true"> ([hashes glossary](../../../../glossary.md#term-hash))</span><span data-glossary-navigation="true"> ([tokens glossary](../../../../glossary.md#term-token))</span>
- `./gradlew :modules:compiler:javadoc` passed, and the generated compiler [Javadoc](../../../../glossary.md#term-javadoc) was inspected.
- `python3 /tmp/validate_synaptik_markdown.py` passed every repository-local Markdown link and
  anchor, balanced fence, final newline, and trailing-whitespace check.
- `git diff --check` passed. The final path audit found exactly 32 tracked create/modify paths
  plus the six required removal-only prototype paths, for the specified 38 touched-path ceiling,
  with no out-of-scope path.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- Repository-wide validation remains deferred to the Compiler 0004B transformation/autograd
  capability checkpoint and CI, as specified by the validation tiers.<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [CI glossary](../../../../glossary.md#term-continuous-integration-ci))</span>

## Implementation notes

- Added seven focused package-private production types for preflight, the three named rule
  families, formula expansion, compilation ownership, and the immutable graph-stage result.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [immutable glossary](../../../../glossary.md#term-immutable); [graph glossary](../../../../glossary.md#graph))</span>
- Extended phase-aware capture and the existing exact optimization components so one
  backward-capable request is captured and optimized as a complete combined graph. DCE is
  whole-graph; CSE remains phase-local.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [DCE glossary](../../../../glossary.md#term-dead-code-elimination-dce); [CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse))</span>
- Added or extended ten compiler test classes covering the closed matrix, preflight side-effect
  boundary, deep iterative traversal, modes, identities, constants, phases, output/role
  de-duplication, and combined optimization.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [constants glossary](../../../../glossary.md#term-constant))</span>
- Finalized Javadocs for all 13 affected production paths and synchronized the Compile, Tensor,
  and Training API documents, autograd design note and user guide, glossary, task, compiler master
  plan, and roadmap.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [autograd glossary](../../../../glossary.md#autograd))</span>
- Reviewed `ARCHITECTURE.md`, focused architecture documents, ADR 0009, model/config APIs,
  architecture and dependency tests, backend conformance, integration tests, Gradle
  configuration, and other modules. No change was needed: the implementation realizes the
  already accepted ownership and lifecycle, changes no dependency or public cross-module
  contract, and adds no backend/runtime/execution behavior.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [model glossary](../../../../glossary.md#term-model); [dependency glossary](../../../../glossary.md#term-dependency); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module); [lifecycle glossary](../../../../glossary.md#lifecycle); [runtime glossary](../../../../glossary.md#term-runtime))</span>

## Completion summary

Completed [compiler](../../../../glossary.md#term-compiler)-owned bounded first-order pre-[capture](../../../../glossary.md#term-capture-meanings) [autograd](../../../../glossary.md#autograd) and one phase-aware combined
[graph](../../../../glossary.md#graph) [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) path. The change preserves the exact package-private boundary, closed [derivative](../../../../glossary.md#term-derivative)
[matrix](../../../../glossary.md#term-matrix-meanings), exact typed splats, deterministic [identity](../../../../glossary.md#term-identity-meanings)/order behavior, and one-shot exact optimization
contract. All specified source, test, [Javadoc](../../../../glossary.md#term-javadoc), [API](../../../../glossary.md#term-application-programming-interface-api), design, user, glossary, and [planning](../../../../glossary.md#term-planning) work is
complete, and all required implementation and documentation [validation](../../../../glossary.md#term-validation) passed.

Status: Complete
