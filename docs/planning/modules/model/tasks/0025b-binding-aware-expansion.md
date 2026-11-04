# Task 0025B: [Binding](../../../../glossary.md#term-binding)-Aware Expansion

## Status

Complete

## Goal

Broaden the existing `EXPAND` [model](../../../../glossary.md#term-model) contract so public expansion can retain an exact [target](../../../../glossary.md#term-target)
[`Shape`](../../../../glossary.md#shape) when right-aligned compatibility depends on unresolved [Dimensions](../../../../glossary.md#dimension).

```text
input Shape + exact target Shape
  -> prove locally when possible
     -> equal pair or static source singleton: accept
     -> incompatible fully static pair: reject
     -> pair containing an unresolved Dimension: retain for later proof
        -> required binding proof: source extent == 1 or source extent == target extent
```

This task changes only Model-owned expression construction and semantic documentation. It adds no
public [API](../../../../glossary.md#term-application-programming-interface-api), [operation](../../../../glossary.md#operation) kind, attributes type, Shape or constraint type, [binding](../../../../glossary.md#term-binding) implementation, or
[compiler](../../../../glossary.md#term-compiler) behavior. Compiler adoption is a separate prerequisite step inside Draft
[Compiler 0005B](../../compiler/master-plan.md#task-list).

## [Scope](../../../../glossary.md#term-scope)

- Preserve exactly the two current public methods:

  ```java
  public Tensor expand(long... requestedShape)
  public Tensor expand(Shape targetShape)
  ```

- Preserve `ShapeTransformKind.EXPAND` and `TargetShapeAttrs(targetShape)` as the sole expansion
  semantic spelling.
- Broaden only aligned compatibility that current [immutable](../../../../glossary.md#term-immutable) [Shape](../../../../glossary.md#shape) [metadata](../../../../glossary.md#term-metadata) cannot decide.
- Preserve [target](../../../../glossary.md#term-target) [rank](../../../../glossary.md#term-rank) greater than or equal to input rank and right [alignment](../../../../glossary.md#term-alignment).
- Continue to accept structurally equal aligned Dimensions and a statically known source
  singleton.<span data-glossary-navigation="true"> ([Dimensions glossary](../../../../glossary.md#dimension))</span>
- Continue to reject the first fully static incompatible aligned pair before Tensor identity
  allocation with the existing exception type and message.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Accept an aligned pair when at least one Dimension is unresolved and neither existing local
  success rule proves compatibility.
- Retain the exact target `Shape` reference in the [descriptor](../../../../glossary.md#term-descriptor) and `TargetShapeAttrs`.
- Define the retained concrete obligation as:

  ```text
  source extent == 1 OR source extent == target extent
  ```

- Keep new leading target [axes](../../../../glossary.md#term-axis) as implicit-[singleton](../../../../glossary.md#term-singleton) expansion without an aligned deferred pair.
- Preserve all current fully resolved static zero-[stride](../../../../glossary.md#term-stride) [view](../../../../glossary.md#term-view-meanings) geometry.
- Keep [layout](../../../../glossary.md#layout) unresolved whenever compatibility or geometry is unresolved.
- Preserve exact [data type](../../../../glossary.md#data-type--datatype), `requiresGrad`, producer/provenance, output index, label/storage
  absence, freshness, validation order, unaffected messages, and identifier side effects.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [validation glossary](../../../../glossary.md#term-validation))</span>
- Add focused tests for the complete static, symbolic, expression, zero, leading-axis, layout, and
  ID matrix.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Finalize affected Javadocs, Tensor/Compile API text, glossary text, and planning records through
  the required separate documentation-focused pass.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- a public `broadcastTo`, `broadcast_to`, `expandTo`, or second semantic spelling
- another Tensor method, overload, static facade, factory entry, operation kind, enum constant,
  attributes record, result carrier, alias, or deprecation bridge<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [facade glossary](../../../../glossary.md#term-facade); [operation glossary](../../../../glossary.md#operation); [constant glossary](../../../../glossary.md#term-constant); [carrier glossary](../../../../glossary.md#term-storage-carrier); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- changes to [`Shape`](../../../../glossary.md#shape), [`Dimension`](../../../../glossary.md#dimension), `StaticDimension`, `DynamicDimension`,
  `ExpressionDimension`, `DimensionExpression`, `DimensionExpressions`, `ShapeBroadcast`,
  `LayoutDescriptor`, `LayoutKind`, `TensorDescriptor`, `TensorFactory`, [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), or
  `TensorProvenance`
- a public or Model-owned binding, constraint, predicate, proof, assignment, substitution, or
  unification type<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [binding glossary](../../../../glossary.md#term-binding))</span>
- assigning values to unresolved Dimensions, replacing the exact target Dimension, inferring a
  result extent, or manufacturing layout/runtime facts<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [extent glossary](../../../../glossary.md#term-extent); [layout glossary](../../../../glossary.md#layout); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- changing reshape, expand-dimensions, squeeze, tile, ordinary broadcasting, or sum-to-Shape
  behavior<span data-glossary-navigation="true"> ([broadcasting glossary](../../../../glossary.md#broadcasting); [sum-to-Shape glossary](../../../../glossary.md#sum-to-shape))</span>
- [compiler](../../../../glossary.md#term-compiler) [inference](../../../../glossary.md#term-inference), `DeferredGraphConstraint`, graph-predicate proof, autograd preflight,
  gradient formulas, graph capture, canonicalization, optimization, publication, or compile
  artifacts<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [autograd glossary](../../../../glossary.md#autograd); [preflight glossary](../../../../glossary.md#term-preflight); [gradient glossary](../../../../glossary.md#term-gradient); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [publication glossary](../../../../glossary.md#term-publication); [compile artifacts glossary](../../../../glossary.md#compile-artifacts))</span>
- implementing or specifying Compiler 0005B; that row remains Draft without a task file
- prepare, planning, runtime, backend, engine, training, execution, materialization, storage
  aliasing, value repetition, or kernels<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare); [planning glossary](../../../../glossary.md#term-planning); [backend glossary](../../../../glossary.md#backend); [engine glossary](../../../../glossary.md#term-engine); [materialization glossary](../../../../glossary.md#materialization); [aliasing glossary](../../../../glossary.md#term-aliasing); [kernels glossary](../../../../glossary.md#kernel))</span>
- architecture contracts or explanatory architecture pages, ADRs, architecture tests,
  backend-conformance tests, integration tests, dependencies, Gradle, or Java version changes<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- capability-baseline changes: the existing selected expand capability and Model ownership remain
  unchanged<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

## Exact behavior contract

### Existing semantic and public surface

The implementation keeps:

```text
ShapeTransformKind.EXPAND
  + TargetShapeAttrs(exactTargetShape)
  + ordered input [source]
  + one output at index zero
```

`ShapeTransformKind` retains its exact enum [constants](../../../../glossary.md#term-constant), order, and one-input/one-output
`TargetShapeAttrs` [signature](../../../../glossary.md#term-signature). `TargetShapeAttrs` retains exactly one non-null `Shape targetShape`
component and performs no input-aware [validation](../../../../glossary.md#term-validation). [`Tensor`](../../../../glossary.md#tensor) retains exactly two public `expand`
methods. There is no `broadcastTo` [alias](../../../../glossary.md#aliasing-and-arena) or second [operation](../../../../glossary.md#operation) representation.

The package-private `TensorExpandExpressions` helper remains final, field-free, non-instantiable,
and limited to its existing six methods:

```java
static Tensor apply(Tensor input, long[] requestedShape)
static Tensor apply(Tensor input, Shape targetShape)
private static void validateExpansion(Shape inputShape, Shape targetShape)
private static Optional<LayoutDescriptor> resolveViewLayout(
        TensorDescriptor inputDescriptor, Shape targetShape)
private static long[] deriveExpandedStrides(
        Shape inputShape, LayoutDescriptor inputLayout, Shape targetShape)
private static Tensor create(
        Tensor input,
        TensorDescriptor inputDescriptor,
        Shape targetShape,
        Optional<LayoutDescriptor> resultLayout)
```

Do not add a field, nested type, helper method, overload, or separate compatibility object.

### [Validation](../../../../glossary.md#term-validation) order and diagnostics

`apply(Tensor, long[])` preserves this order:

1. null-check `input` with message `input`;
2. null-check `requestedShape` with message `requestedShape`;
3. read the exact input [descriptor](../../../../glossary.md#term-descriptor) once and its [Shape](../../../../glossary.md#shape) once;
4. construct the [target](../../../../glossary.md#term-target) with `Shape.of(requestedShape)`;
5. validate target [rank](../../../../glossary.md#term-rank) and aligned compatibility;
6. resolve [layout](../../../../glossary.md#layout) only when all required geometry is available;
7. construct the result once.

`Shape.of` continues to copy the raw request semantics, treat an empty array as [scalar](../../../../glossary.md#term-scalar) Shape,
accept zero, and reject the first negative literal with:

```text
Static dimension size must be non-negative: <value>
```

`apply(Tensor, Shape)` preserves this order:

1. null-check `input` with message `input`;
2. null-check `targetShape` with message `targetShape`;
3. read the exact input descriptor once and its Shape once;
4. validate target rank and aligned compatibility;
5. resolve layout only when all required geometry is available;
6. construct the result once.

Target rank below input rank still fails with:

```text
expand target rank <targetRank> must be at least input rank <inputRank>
```

After computing `rankOffset = targetRank - inputRank`, inspect aligned input [axes](../../../../glossary.md#term-axis) in increasing
input-axis order. The first pair whose [Dimensions](../../../../glossary.md#dimension) are both static, unequal, and whose source is
not static one still fails with:

```text
cannot expand input shape <inputShape> to target shape <targetShape> at target axis <targetAxis>
```

The task changes no null, raw-size, rank, fully static incompatibility, arithmetic-[overflow](../../../../glossary.md#term-overflow), or
identifier-exhaustion message. Every local failure remains before the sole factory delegation and
consumes no [Tensor](../../../../glossary.md#tensor) ID.

### Aligned compatibility [matrix](../../../../glossary.md#term-review-matrix)

An unresolved [Dimension](../../../../glossary.md#dimension) is any `Dimension` whose `isStatic()` result is false, including named
dynamic Dimensions, symbolic expression Dimensions, and constrained unknown Dimensions.

| Source Dimension | [Target](../../../../glossary.md#term-target) Dimension | [Model](../../../../glossary.md#term-model) construction | Retained [binding](../../../../glossary.md#term-binding) rule |
|---|---|---|---|
| Structurally equal | Structurally equal | Accept | Already proved equal |
| Static `1` | Any static or unresolved target | Accept | Source [singleton](../../../../glossary.md#term-singleton) already proved |
| Static non-`1` | Unequal static | Reject immediately | Contradiction is fully known |
| Unresolved | Unequal static | Accept deferred | Source must bind to `1` or target |
| Static non-`1` | Unresolved | Accept deferred | Target must bind equal to source |
| Unresolved | Structurally unequal unresolved | Accept deferred | Source must bind to `1` or equal target |

[Structural equality](../../../../glossary.md#term-structural-equality) remains the complete Model-local equality proof. Equal named symbols, equal
symbolic formulas, and the same constrained-unknown [identity](../../../../glossary.md#term-identity-meanings) are therefore accepted without a
new obligation. Distinct names, structurally unequal formulas, different unknown identities, and
static/unresolved mixtures are accepted but not equated.

The binding rule is directional:

```text
source == 1 OR source == target
```

It is not symmetric ordinary [broadcasting](../../../../glossary.md#broadcasting). A target singleton does not itself prove that an
unresolved source can shrink; it defers the requirement that the source eventually bind to one.
A statically known source `2` still cannot expand to static target `1`.

### [Shape](../../../../glossary.md#shape) examples

The examples describe construction [metadata](../../../../glossary.md#term-metadata) and later proof, not execution.

| Input Shape | [Target](../../../../glossary.md#term-target) Shape | Construction result | Later requirement |
|---|---|---|---|
| `[2, 3]` | `[2, 3]` | Accept | Exact equality |
| `[1, 3]` | `[4, 3]` | Accept | Source [singleton](../../../../glossary.md#term-singleton) |
| `[1, 3]` | `[0, 3]` | Accept | Source singleton; result [axis](../../../../glossary.md#term-axis) is empty |
| `[2, 3]` | `[4, 3]` | Reject at target axis `0` | Fully static contradiction |
| `[2, 3]` | `[1, 3]` | Reject at target axis `0` | Fully static shrink |
| `[0, 3]` | `[1, 3]` | Reject at target axis `0` | Empty source is not singleton |
| `[N, 3]` | `[N, 3]` | Accept | [Structural equality](../../../../glossary.md#term-structural-equality) |
| `[N, 3]` | `[4, 3]` | Accept deferred | `N == 1 OR N == 4` |
| `[4, 3]` | `[N, 3]` | Accept deferred | `N == 4` |
| `[N, 3]` | `[M, 3]` | Accept deferred | `N == 1 OR N == M` |
| `[N + 1, 3]` | `[M, 3]` | Accept deferred | `N + 1 == 1 OR N + 1 == M` |
| `[N, 3]` | `[B, N, 3]` | Accept | `B` is a new leading axis; aligned `N` is equal |

For the final row, no source/target pair exists for leading axis `B`; it is implicit-singleton
expansion and needs no deferred aligned pair. Its numeric geometry remains unresolved because
`B` is unresolved.

[Scalar](../../../../glossary.md#term-scalar) input continues to expand to any target Shape. A non-scalar input cannot use a lower-[rank](../../../../glossary.md#term-rank)
target. Same-Shape, repeated, and nested calls remain fresh explicit occurrences.

### [Layout](../../../../glossary.md#layout) policy

Preserve the existing resolved layout exactly when the current facts are sufficient:

- [target](../../../../glossary.md#term-target) [Shape](../../../../glossary.md#shape) is fully static;
- input layout is resolved, which already requires compatible fully static input geometry; and
- the aligned compatibility pass has no unresolved pair.

In that case:

- every new leading target [axis](../../../../glossary.md#term-axis) receives [stride](../../../../glossary.md#term-stride) zero;
- a changed statically known source-[singleton](../../../../glossary.md#term-singleton) axis receives stride zero;
- every unchanged aligned axis preserves the exact source stride, including an existing zero or
  non-canonical stride;<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- the exact input [storage offset](../../../../glossary.md#term-storage-offset) is retained;
- the result is marked as a logical [view](../../../../glossary.md#term-view-meanings); and
- current `LayoutDescriptor` kind/span/[overflow](../../../../glossary.md#term-overflow) behavior is unchanged.

Return unresolved layout when any required fact is unavailable, including:

- an unresolved target [Dimension](../../../../glossary.md#dimension);
- an unresolved source Dimension;
- an accepted deferred aligned compatibility pair; or
- absent input layout.

Do not choose a stride from a later alternative, bind a Dimension, replace an unresolved [extent](../../../../glossary.md#term-extent),
assume a [materialization](../../../../glossary.md#materialization), or manufacture a [runtime](../../../../glossary.md#term-runtime)/layout fact. Logical view [metadata](../../../../glossary.md#term-metadata) continues
to attach no [host storage](../../../../glossary.md#host-storage) and promises neither physical [aliasing](../../../../glossary.md#term-aliasing) nor zero-copy execution.

### [Descriptor](../../../../glossary.md#term-descriptor), [ownership](../../../../glossary.md#term-ownership-meanings), [metadata](../../../../glossary.md#term-metadata), and ID effects

Every success preserves the current construction:

1. one `TensorDescriptor` with exact input [`DataType`](../../../../glossary.md#data-type--datatype), exact target Shape reference, selected
   resolved/unresolved layout, and exact input `requiresGrad`;
2. one `TargetShapeAttrs` retaining that same [target](../../../../glossary.md#term-target) [Shape](../../../../glossary.md#shape) reference;
3. one `Operation(ShapeTransformKind.EXPAND, attrs)`;
4. one [producer](../../../../glossary.md#term-producer) with exact ordered input `[input]` and one output descriptor;
5. [canonical](../../../../glossary.md#term-canonical-form) output [Tensor](../../../../glossary.md#tensor) at [provenance](../../../../glossary.md#provenance) index zero;
6. one fresh Tensor ID, absent label, and absent [host storage](../../../../glossary.md#host-storage).

The input [identity](../../../../glossary.md#term-identity-meanings), descriptor, Shape, [layout](../../../../glossary.md#layout), label, provenance, storage association, storage
[liveness](../../../../glossary.md#term-liveness), and values remain unchanged. Early [validation](../../../../glossary.md#term-validation) and layout-arithmetic failures consume no
ID. Successful construction consumes exactly one fresh ID. Existing terminal identifier
exhaustion remains the final factory failure, with no [rollback](../../../../glossary.md#term-rollback) or ID reuse.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Backend glossary](../../../../glossary.md#backend))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Roadmap](../../../roadmap.md)
- [Model master plan](../master-plan.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model capabilities](../capabilities.md)
- [Shape and Dimension model](0002-shape-and-dimension-model.md)<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension))</span>
- [Reshape and expand semantics](0017c-reshape-and-expand-semantics.md)<span data-glossary-navigation="true"> ([Reshape and expand semantics glossary](../../../../glossary.md#reshape-and-expand-semantics))</span>
- [Expand Tensor expressions](0017d1-expand-tensor-expressions.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- [Symbolic extent expressions](0018m-symbolic-extent-expressions.md)<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent))</span>
- [Binding-aware sum-to-Shape](0023a-binding-aware-sum-to-shape.md)<span data-glossary-navigation="true"> ([Binding glossary](../../../../glossary.md#term-binding); [sum-to-Shape glossary](../../../../glossary.md#sum-to-shape))</span>
- [Canonical TensorProducer outputs](0025-canonical-tensor-producer-outputs.md)<span data-glossary-navigation="true"> ([Canonical glossary](../../../../glossary.md#term-canonical-form); [TensorProducer glossary](../../../../glossary.md#producer--tensorproducer))</span>
- [Portable floating semantics](0025a-portable-floating-comparison-extrema-and-clamp-semantics.md)
- [Compiler master plan](../../compiler/master-plan.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>
- [Captured-graph inference](../../compiler/tasks/0002-captured-graph-inference-and-validation.md)<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [inference glossary](../../../../glossary.md#term-inference))</span>
- [Exact-composition gradient rules](../../compiler/tasks/0004a-exact-composition-gradient-rule-extensions.md)<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient))</span>
- [Compiler 0005A](../../compiler/tasks/0005a-derivative-policy-and-elementwise-activation-gradient-completion.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Model owns public Tensor expression semantics, immutable descriptors, operation attributes, and
  pre-capture producer/provenance construction.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [descriptors glossary](../../../../glossary.md#term-descriptor); [capture glossary](../../../../glossary.md#term-capture-meanings); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- [Tensor](../../../../glossary.md#tensor) remains public [model](../../../../glossary.md#term-model) state and is not [graph](../../../../glossary.md#graph) [IR](../../../../glossary.md#term-intermediate-representation-ir).
- [`Operation`](../../../../glossary.md#operation) remains backend-neutral and exposes no binding, backend support, route, or
  executable state.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [binding glossary](../../../../glossary.md#term-binding))</span>
- Model may reject a contradiction provable from immutable Shape values and retain an unresolved
  obligation, but it must not solve or bind that obligation.
- Compiler owns graph-wide inference, validation, deferred graph predicates, autograd preflight,
  and gradient construction.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation); [autograd glossary](../../../../glossary.md#autograd); [preflight glossary](../../../../glossary.md#term-preflight); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- The existing [compiler](../../../../glossary.md#term-compiler) predicate vocabulary already demonstrates the later [shape](../../../../glossary.md#shape) of the proof:

  ```text
  AnyOf(
      DimensionEqual(source, StaticDimension(1)),
      DimensionEqual(source, target))
  ```

  This is downstream evidence only. This task neither creates that predicate nor changes
  Compiler.
- Planning and prepare own materialization/lowering handoffs; concrete backends own route
  selection and implementation; runtime executes prepared work.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [materialization glossary](../../../../glossary.md#materialization); [lowering glossary](../../../../glossary.md#lowering); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- No [module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency), package boundary, [architecture contract](../../../../glossary.md#architecture-contract), or [lifecycle](../../../../glossary.md#lifecycle) [ownership](../../../../glossary.md#term-ownership-meanings) changes.

## Package impact

No package is added, moved, or renamed.

Existing packages used:

- `io.github.pho001.synaptik.model.operation.layout`
- `io.github.pho001.synaptik.model.tensor`

Type placement:

- `...operation.layout.ShapeTransformKind` continues to own the sole `EXPAND` semantic [identity](../../../../glossary.md#term-identity-meanings).
- `...operation.layout.TargetShapeAttrs` continues to own the exact [immutable](../../../../glossary.md#term-immutable) [target](../../../../glossary.md#term-target) [Shape](../../../../glossary.md#shape).
- `...tensor.TensorExpandExpressions` continues to own package-private input-aware validation,
  layout derivation, and expression construction.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [layout glossary](../../../../glossary.md#layout))</span>
- `...tensor.Tensor` remains the public fluent [facade](../../../../glossary.md#term-facade).

No public surface or package map changes.

## Affected files

Production and [Javadoc](../../../../glossary.md#term-javadoc) (4):

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/ShapeTransformKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/TargetShapeAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorExpandExpressions.java`

Tests (1):

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorExpandExpressionTest.java`

Documentation and [planning](../../../../glossary.md#term-planning) (7):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/modules/compiler/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless repository evidence contradicts this specification:

- [Shape](../../../../glossary.md#shape)/[Dimension](../../../../glossary.md#dimension)/DimensionExpressions and LayoutDescriptor foundations
- TensorDescriptor, TensorFactory, [TensorProducer](../../../../glossary.md#producer--tensorproducer), TensorProvenance, and identifier allocation
- reshape, expand-dimensions/squeeze, [broadcasting](../../../../glossary.md#broadcasting), and [sum-to-Shape](../../../../glossary.md#sum-to-shape) contracts
- current [Compiler](../../../../glossary.md#term-compiler) `LayoutInference`, `ReductionNormalizationInference`,
  `DeferredGraphConstraint`, `AutogradPreflight`, and [gradient](../../../../glossary.md#term-gradient) rules
- model capabilities, Training/Runtime APIs, architecture/ADRs/tests, conformance/integration
  tests, Gradle/Java 26 configuration, dependencies, and all other modules<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Runtime glossary](../../../../glossary.md#term-runtime); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

At most the exact 12 paths listed above may change: four existing [Model](../../../../glossary.md#term-model) production/[Javadoc](../../../../glossary.md#term-javadoc) paths,
one focused Model test, and seven documentation/[planning](../../../../glossary.md#term-planning) paths.

Stop before changing a thirteenth path, adding a public/helper type or method, changing the package
map, editing [compiler](../../../../glossary.md#term-compiler) Java/tests, changing capabilities, architecture, dependencies, [Gradle](../../../../glossary.md#term-gradle), or
creating any Compiler 0005B–0005E/0006 task file. If implementation evidence requires another
path, update this task through a planning decision before proceeding.

## [Javadoc](../../../../glossary.md#term-javadoc) and documentation requirements

- Update all four affected production Javadocs to describe local proof, deferred aligned
  compatibility, exact target retention, and unresolved layout without claiming binding or
  execution.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [layout glossary](../../../../glossary.md#layout))</span>
- Preserve meaningful `@param`, `@return`, and `@throws` coverage for both public methods and
  helper methods.
- Keep the current validation order, exact unaffected messages, no-ID failures, metadata,
  producer/provenance, storage/label absence, freshness, and identifier exhaustion visible.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [metadata glossary](../../../../glossary.md#term-metadata); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- Update Tensor API examples and failure summaries from “unprovable dynamic pairs reject” to the
  selected source-one-or-equal deferred rule.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- Update [Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api) current-status text precisely:
  - [structural capture](../../../../glossary.md#term-structural-capture) can retain the broadened [Model](../../../../glossary.md#term-model) [occurrence](../../../../glossary.md#term-occurrence);
  - current [compiler](../../../../glossary.md#term-compiler) `LayoutInference` does not yet adopt [binding](../../../../glossary.md#term-binding)-aware EXPAND;
  - Compiler 0005B owns the later deferred constraint and [gradient](../../../../glossary.md#term-gradient)/[preflight](../../../../glossary.md#term-preflight) integration; and
  - no concrete binding, [lowering](../../../../glossary.md#lowering), [backend](../../../../glossary.md#backend), or execution support is claimed.
- Update the glossary's existing expand/broadcasting/Dimension/deferred-constraint wording only
  where needed. Do not add a second term or synonym for expansion.<span data-glossary-navigation="true"> ([broadcasting glossary](../../../../glossary.md#broadcasting); [Dimension glossary](../../../../glossary.md#dimension))</span>
- Keep completed task history intact. On completion, synchronize Model 0025B as Complete. Compiler
  0005B–0005E/0006 remain Draft without detailed specifications.
- Record reasoned no-change conclusions for capabilities, related Javadocs, Training/Runtime APIs,
  architecture/ADRs/tests, conformance/integration, Gradle/Java 26, dependencies, and other
  modules.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

## Acceptance criteria

- Exactly two public `expand` overloads remain; public [Tensor](../../../../glossary.md#tensor) method count remains 200.
- `ShapeTransformKind` [constants](../../../../glossary.md#term-constant)/order/[signature](../../../../glossary.md#term-signature), `TargetShapeAttrs` shape, and the exact helper
  surface remain unchanged.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>
- No `broadcastTo`, second semantic spelling, new kind, attrs, Shape/constraint type, alias,
  package, or facade exists.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [facade glossary](../../../../glossary.md#term-facade))</span>
- Exact null/raw/rank/static-incompatibility validation order, exception types/messages, and no-ID
  behavior remain unchanged.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [validation glossary](../../../../glossary.md#term-validation))</span>
- Equal pairs and static source-[singleton](../../../../glossary.md#term-singleton) pairs remain accepted.
- Every aligned pair containing at least one unresolved Dimension is accepted unless structural
  equality or static source one has already proved it.<span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension); [structural equality glossary](../../../../glossary.md#term-structural-equality))</span>
- Every unequal fully static pair whose source is not one still fails immediately at the first
  aligned target axis.
- New leading [target](../../../../glossary.md#term-target) [axes](../../../../glossary.md#term-axis) remain valid without an aligned deferred pair.
- The exact target Shape reference is retained in descriptor and attributes for raw-normalized and
  exact-Shape requests.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Fully resolved static dense, offset, strided, broadcast, scalar, zero-extent, identity, leading,
  and singleton cases retain their exact current stride, offset, kind, view, span, and overflow
  behavior.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [extent glossary](../../../../glossary.md#term-extent); [identity glossary](../../../../glossary.md#term-identity-meanings); [stride glossary](../../../../glossary.md#term-stride); [view glossary](../../../../glossary.md#term-view-meanings); [overflow glossary](../../../../glossary.md#term-overflow))</span>
- Every unresolved compatibility or geometry case has unresolved layout; no dimension, stride,
  materialization, or runtime fact is manufactured.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [materialization glossary](../../../../glossary.md#materialization); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Every success retains exact type and eligibility, one exact EXPAND [producer](../../../../glossary.md#term-producer), `[input]`,
  provenance index zero, freshness, absent label/storage, and exactly one new ID without mutating
  input state.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- Focused tests cover named, expression, constrained-unknown, static/unresolved,
  unresolved/static, unresolved/unresolved, equal unresolved, leading unresolved, static
  contradiction, zero, layout, validation order, message, and ID cases.
- No compiler Java/test behavior changes; Compiler 0005B remains Draft and is recorded as the
  adoption owner.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [owner glossary](../../../../glossary.md#term-owner))</span>
- Exact 12-path scope, package placement, documentation pass, validation evidence, and planning
  status synchronization all pass before task completion.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [planning glossary](../../../../glossary.md#term-planning); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Focused implementation validation:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.tensor.TensorExpandExpressionTest \
  --tests io.github.pho001.synaptik.model.operation.layout.ShapeTransformSemanticsTest
```

After executable Java stabilizes, [run](../../../../glossary.md#term-run) exactly one final [Model](../../../../glossary.md#term-model) suite:

```bash
./gradlew :modules:model:test
```

The implementation pass records exact test/suite counts and hands that evidence to the
documentation-focused pass. The documentation pass does not rerun successful Java tests unless it
changes executable Java behavior or records a concrete reason.

Documentation pass after final Javadocs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

Also validate:

- every changed Markdown relative link and heading anchor;
- balanced fences, final newlines, and trailing whitespace;
- exact two-method public expand surface and unchanged 200-method public [Tensor](../../../../glossary.md#tensor) count;
- exact six-method field-free helper and unchanged [operation](../../../../glossary.md#operation)/attrs surface;
- exact 12 touched paths and only Model Java/test plus documentation/[planning](../../../../glossary.md#term-planning) changes;
- Model 0025B Complete [synchronization](../../../../glossary.md#term-synchronization) in its task, Model master plan, and roadmap;
- [Compiler](../../../../glossary.md#term-compiler) 0005B–0005E/0006 Draft synchronization;
- no Compiler 0005B or later task file;
- no capability, compiler Java/test, Gradle, dependency, architecture, ADR, architecture-test,
  conformance, integration, runtime, prepare, backend, engine, or training edit.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [engine glossary](../../../../glossary.md#term-engine))</span>

Repository-wide validation is deferred to the first-order [gradient](../../../../glossary.md#term-gradient) capability [checkpoint](../../../../glossary.md#term-checkpoint) and [CI](../../../../glossary.md#term-continuous-integration-ci).
This focused Model task changes no [dependency](../../../../glossary.md#term-dependency), build, architecture boundary, or executable [module](../../../../glossary.md#term-module)
outside `modules/model`.

## Required separate documentation [handoff](../../../../glossary.md#term-handoff)

After implementation and final [Model](../../../../glossary.md#term-model) test evidence, hand the actual diff to a separate
documentation-focused agent or thread with clean context in the same overall change.

The handoff must include:

- this task specification and exact goal;
- the four affected production contracts and focused test;
- the exact implementation diff;
- exact focused/final Model test commands and results;
- the source-one-or-source-equal [binding](../../../../glossary.md#term-binding) rule and complete behavior [matrix](../../../../glossary.md#term-matrix-meanings);
- [validation](../../../../glossary.md#term-validation) order/messages, [layout](../../../../glossary.md#layout) policy, [metadata](../../../../glossary.md#term-metadata)/[provenance](../../../../glossary.md#provenance), and ID effects;
- the current [Compiler](../../../../glossary.md#term-compiler) non-adoption boundary and Compiler 0005B follow-up;
- expected [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api), glossary, task/master/roadmap edits;
- the unchanged capability conclusion; and
- all documentation, [scope](../../../../glossary.md#term-scope), status, and whitespace checks.

The documentation agent must independently inspect source/tests and the downstream compiler
evidence, finalize affected Javadocs and explanatory/[planning](../../../../glossary.md#term-planning) text, check glossary impact, and
record files reviewed, changes made, reused evidence, commands/results, limitations, and
unresolved issues. Task 0025B cannot become Complete before that pass finishes.

## Dependencies

- [Model](../../../../glossary.md#term-model) 0002: static, named dynamic, [scalar](../../../../glossary.md#term-scalar), zero, [structural equality](../../../../glossary.md#term-structural-equality), and [immutable](../../../../glossary.md#term-immutable) [Shape](../../../../glossary.md#shape) rules.
- Model 0017C: the sole EXPAND [identity](../../../../glossary.md#term-identity-meanings) and exact `TargetShapeAttrs` semantic representation.
- Model 0017D1: the two public expand methods, helper, current validation/messages, zero-stride
  layout, metadata, provenance, and ID contract being broadened.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [stride glossary](../../../../glossary.md#term-stride); [layout glossary](../../../../glossary.md#layout); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance))</span>
- Model 0018M: expression [Dimensions](../../../../glossary.md#dimension) and unresolved Shape vocabulary without [binding](../../../../glossary.md#term-binding).
- Model 0023A: precedent for accepting only statically provable compatibility failures while
  retaining an exact target Shape for later compiler/binding proof.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Model 0025 and 0025A: completed current Model/Compiler-enabling frontier and preserved producer
  and forward-semantics history.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>
- Compiler 0001–0005A: read-only downstream evidence for capture, typed deferred predicates,
  autograd, and the current first-order frontier.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [autograd glossary](../../../../glossary.md#autograd))</span>

All implementation prerequisites are Complete.

## Follow-up tasks

- Compiler 0005B remains Draft and must adopt binding-aware EXPAND during its existing reduction,
  scan, softmax, statistics, and normalization gradient frontier. It owns:<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [scan glossary](../../../../glossary.md#term-cumulative-scan); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [normalization glossary](../../../../glossary.md#term-normalization); [gradient glossary](../../../../glossary.md#term-gradient))</span>
  - [graph](../../../../glossary.md#graph) [inference](../../../../glossary.md#term-inference) that proves, rejects, or retains each aligned
    `source == 1 OR source == target` predicate;
  - deterministic deferred-constraint subject/order;
  - preflight and generated-gradient consistency for binding-dependent EXPAND and
    sum-to-Shape inversion; and<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [sum-to-Shape glossary](../../../../glossary.md#sum-to-shape))</span>
  - focused [compiler](../../../../glossary.md#term-compiler) [validation](../../../../glossary.md#term-validation).
- Compiler 0005C–0005E and 0006 remain Draft in their existing order without detailed task files.
- Concrete binding, prepare/backend lowering, materialization, execution, and conformance remain
  with their later owners.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [materialization glossary](../../../../glossary.md#materialization); [conformance glossary](../../../../glossary.md#term-conformance))</span>

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None.

If implementation requires [Model](../../../../glossary.md#term-model)-owned constraints or [binding](../../../../glossary.md#term-binding), a new public method/kind/attrs
type, [compiler](../../../../glossary.md#term-compiler) Java changes, a package/[dependency](../../../../glossary.md#term-dependency) change, or architecture-contract edits, stop and
report the conflict rather than expanding this task.

## Implementation prompt

Use this prompt in a separate clean-context task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/developer-guide/documentation-rules.md,
docs/planning/planning-guide.md, docs/planning/roadmap.md, the Model and Compiler master plans,
and docs/planning/modules/model/tasks/0025b-binding-aware-expansion.md in full. Read the completed
Shape/dynamic/expand/sum-to-Shape and current compiler tasks linked by 0025B. Inspect every exact
affected and review-only source/test named by the task.

Implement Model task 0025B exactly within its 12 authorized paths. Broaden only existing EXPAND:
accept aligned compatibility involving an unresolved Dimension, retain the exact target Shape,
and keep layout unresolved until compatibility and geometry are known. Preserve fully static
rejection/messages, resolved zero-stride geometry, public/helper/semantic surfaces, metadata,
provenance, freshness, validation order, and ID effects. Add no broadcastTo spelling, public
method, kind, attrs, Shape/constraint type, binding, compiler behavior, or later task spec. Stop
on an architecture, completed-contract, package, affected-file, or maximum-scope conflict.

Run focused tests while developing and exactly one final Model suite after executable Java
stabilizes. Then hand the actual diff and exact Java evidence to a separate clean-context
documentation-focused agent in the same overall change. That agent must inspect final
source/tests and current compiler evidence, finalize affected Javadocs, Tensor/Compile APIs,
glossary and planning status, record the reasoned capabilities no-change conclusion, and run
Javadoc/Markdown/scope/status/whitespace validation without repeating successful Java tests
unless executable behavior changes or a concrete reason is recorded.

Do not mark 0025B Complete until implementation, tests, the independent documentation pass, every
acceptance criterion, and final evidence succeed. Keep Compiler 0005B–0005E and 0006 Draft
without detailed task specifications.
```

## Local decisions

- Broadened the existing directional EXPAND contract rather than adding `broadcastTo` or another
  semantic spelling.
- Reused exact source and target Dimensions as the later proof inputs; no Model constraint object
  is needed.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [Dimensions glossary](../../../../glossary.md#dimension); [Model glossary](../../../../glossary.md#term-model))</span>
- Selected immediate failure only for a fully static contradiction. Any aligned pair involving an
  unresolved Dimension is representable without choosing a bound result because the caller's
  exact target Shape is already retained.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>
- Kept leading target axes outside the aligned-pair obligation because they expand from implicit
  source singletons.<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis))</span>
- Kept layout conservative: current static resolved geometry remains exact, while every
  unresolved compatibility or geometry case stays unresolved.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>
- Assigned compiler adoption to existing Draft Compiler 0005B because that task already owns
  binding-dependent sum-to-Shape and the gradient family that consumes expansion.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [binding glossary](../../../../glossary.md#term-binding); [sum-to-Shape glossary](../../../../glossary.md#sum-to-shape); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Left `docs/planning/modules/model/capabilities.md` unchanged because it already selects expand
  and assigns graph-wide/binding proof outside Model; this task changes the completeness of an
  existing semantic contract, not the selected capability inventory.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>

## Known limitations

- At task completion, [Model](../../../../glossary.md#term-model) construction will represent [binding](../../../../glossary.md#term-binding)-aware EXPAND, but current [Compiler](../../../../glossary.md#term-compiler)
  `LayoutInference` will not yet accept and retain its unresolved aligned obligation.
- Current autograd/preflight support does not yet close binding-dependent EXPAND and inverse
  sum-to-Shape cases. Compiler 0005B owns that adoption.<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [preflight glossary](../../../../glossary.md#term-preflight); [sum-to-Shape glossary](../../../../glossary.md#sum-to-shape))</span>
- No current concrete dimension binding, execution, materialization, or backend implementation is
  introduced.<span data-glossary-navigation="true"> ([dimension glossary](../../../../glossary.md#dimension); [materialization glossary](../../../../glossary.md#materialization); [backend glossary](../../../../glossary.md#backend))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- The implementation context
  `/root/implement_model_0025b_expand` ran the focused command exactly as specified. It passed
  22 tests: 14 `TensorExpandExpressionTest` tests plus 8
  `ShapeTransformSemanticsTest` tests, with zero failures, errors, or skipped tests.
- After executable Java stabilized, that context ran `./gradlew :modules:model:test` exactly once.
  The final Model report passed 1,019 tests across 127 suites with zero failures, errors, or
  skipped tests.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- The separate documentation context
  `/root/implement_model_0025b_expand/docs_0025b` reused those successful Java results and did not
  repeat either test command. It ran `./gradlew :modules:model:javadoc`; compilation and Javadoc
  completed with <span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`BUILD SUCCESSFUL`.
- The Tensor API example was compiled with Java 26 against final Model classes and executed. Its
  exact output matched the documented Shape, <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Shape glossary](../../../../glossary.md#shape))</span>`[0, 0, 1]` strides, offset, layout kind, span, view,
  EXPAND provenance, input identity, exact dynamic target identity, unresolved layout, and absent
  host storage.<span data-glossary-navigation="true"> ([strides glossary](../../../../glossary.md#term-stride); [layout glossary](../../../../glossary.md#layout); [view glossary](../../../../glossary.md#term-view-meanings); [provenance glossary](../../../../glossary.md#provenance); [identity glossary](../../../../glossary.md#term-identity-meanings); [target glossary](../../../../glossary.md#term-target); [host storage glossary](../../../../glossary.md#host-storage))</span>
- `javap` confirmed exactly 200 public [Tensor](../../../../glossary.md#tensor) methods and exactly two public `expand` overloads.
  It also confirmed that `TensorExpandExpressions` remains final and field-free with exactly six
  declared methods, `ShapeTransformKind` remains ordered `RESHAPE`, `EXPAND` with the same
  signature method, and <span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>`TargetShapeAttrs` remains the one-component record surface.
- Generated Javadoc inspection confirmed the right-aligned source-one-or-source-equal obligation,
  fully static rejection including zero-to-one, exact-target ownership, no Model binding or
  deferred constraint, and binding-dependent unresolved-layout wording. The package-private
  helper source was inspected directly because the standard public Javadoc task does not publish
  package-private types.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [binding glossary](../../../../glossary.md#term-binding))</span>
- `python3 /tmp/validate_synaptik_markdown.py` passed with 12 Markdown files and 695
  repository-local links checked. Heading anchors, relative links, fences, final newlines, and
  Markdown structure passed.
- `git diff --check` passed. The union of tracked and untracked changed paths is exactly the 12
  authorized paths: four Model production/Javadoc files, the focused Model test, and seven
  documentation/planning files.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Source and test review confirmed the complete aligned matrix: structural equality, static
  source singleton, named Dimensions, expression Dimensions, constrained unknowns,
  static/unresolved, unresolved/static, unresolved/unresolved, unresolved leading axes, fully
  static contradictions, and ordinary zero behavior. It also confirmed exact target identity,
  resolved stride/offset/view behavior, unresolved binding-dependent layout, metadata,
  provenance index zero, freshness, input non-mutation, failure messages/order, no-ID failures,
  one-ID successes, and terminal identifier exhaustion.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [structural equality glossary](../../../../glossary.md#term-structural-equality); [singleton glossary](../../../../glossary.md#term-singleton); [Dimensions glossary](../../../../glossary.md#dimension); [axes glossary](../../../../glossary.md#term-axis); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- [Compiler](../../../../glossary.md#term-compiler) source review confirmed current `LayoutInference` still rejects a non-equal,
  non-static-source-singleton unresolved EXPAND pair. Model creates no
  `DeferredGraphConstraint`; Draft Compiler 0005B owns the later
  `AnyOf(DimensionEqual(source, 1), DimensionEqual(source, target))` adoption. Existing
  `SUM_TO_SHAPE` behavior is unchanged.
- Planning review confirmed Model 0025B Complete in this task, the Model master plan, and the
  roadmap. Compiler 0005B–0005E and 0006 remain Draft, and no detailed task file exists for any of
  those rows.

Reasoned no-change conclusions:

- `docs/planning/modules/model/capabilities.md` remains unchanged because EXPAND was already a
  selected Model capability and its ownership boundary was already correct; this task improves
  representability within that capability rather than selecting a new one.
- The Training API remains unchanged because this task adds no training request, differentiable
  role, saved value, optimizer, or session behavior. No binding-dependent gradient claim is made.<span data-glossary-navigation="true"> ([differentiable glossary](../../../../glossary.md#derivative-and-chain-rule); [optimizer glossary](../../../../glossary.md#term-optimizer); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Related reshape, expand-dimensions, squeeze, tile, ordinary broadcasting, and sum-to-Shape
  contracts remain unchanged because this task alters only the existing one-input directional
  EXPAND validation rule.<span data-glossary-navigation="true"> ([broadcasting glossary](../../../../glossary.md#broadcasting); [sum-to-Shape glossary](../../../../glossary.md#sum-to-shape))</span>
- Runtime, Prepare, Backend, Engine, and tracing documentation remains unchanged because no
  prepared state, binding implementation, materialization, alias guarantee, lowering, execution,
  or trace event changed.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Backend glossary](../../../../glossary.md#backend); [Engine glossary](../../../../glossary.md#term-engine); [materialization glossary](../../../../glossary.md#materialization); [alias glossary](../../../../glossary.md#aliasing-and-arena); [lowering glossary](../../../../glossary.md#lowering); [trace glossary](../../../../glossary.md#trace))</span>
- `ARCHITECTURE.md`, focused architecture pages, ADRs, and architecture tests remain unchanged
  because Model and Compiler ownership, module boundaries, and dependencies did not change.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
- Backend-conformance and integration tests remain unchanged because this task constructs Model
  metadata only and adds no executable backend or end-to-end behavior.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>
- Gradle, Java 26 configuration, dependency declarations, and all other modules remain unchanged
  because no build or cross-module contract changed.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Implementation notes

- Changed only `TensorExpandExpressions.validateExpansion` executable logic: it now rejects an
  unequal aligned pair only when both dimensions are static and the source extent is not one.
  Validation order, diagnostics, allocation point, and every other helper statement remain
  unchanged.<span data-glossary-navigation="true"> ([dimensions glossary](../../../../glossary.md#dimension); [extent glossary](../../../../glossary.md#term-extent); [Validation glossary](../../../../glossary.md#term-validation))</span>
- Expanded the focused test to cover all unresolved Dimension pairings, leading axes, static and
  zero contradictions, exact target and layout state, metadata/provenance, input state, validation
  ordering, messages, and Tensor-ID effects.<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis); [target glossary](../../../../glossary.md#term-target); [layout glossary](../../../../glossary.md#layout); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Finalized all four affected production Javadocs, the Tensor API executable example and
  summaries, the Compile API non-adoption boundary, glossary distinctions, and planning status.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [planning glossary](../../../../glossary.md#term-planning))</span>
- Preserved exactly two public expand overloads, the 200-method Tensor surface, the six-method
  field-free helper, enum order/signature, record surface, exact target ownership, one-output
  provenance, resolved layout behavior, and all out-of-scope layers.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [scope glossary](../../../../glossary.md#term-scope))</span>
- Added no new public or semantic spelling, type, predicate, constraint, binding, compiler
  behavior, gradient rule, materialization, lowering, execution, dependency, or build change.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [compiler glossary](../../../../glossary.md#term-compiler); [gradient glossary](../../../../glossary.md#term-gradient); [materialization glossary](../../../../glossary.md#materialization); [lowering glossary](../../../../glossary.md#lowering); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Completion summary

Completed [binding](../../../../glossary.md#term-binding)-aware [Model](../../../../glossary.md#term-model) expansion within the exact 12 authorized paths. Any aligned pair
containing an unresolved [Dimension](../../../../glossary.md#dimension) is now representable with the exact [target](../../../../glossary.md#term-target) [Shape](../../../../glossary.md#shape) and unresolved
[layout](../../../../glossary.md#layout), while [structural equality](../../../../glossary.md#term-structural-equality), static source-[singleton](../../../../glossary.md#term-singleton) acceptance, right [alignment](../../../../glossary.md#term-alignment), leading
[axes](../../../../glossary.md#term-axis), and fully static rejection remain intact. Model retains the later
source-one-or-source-equal requirement without creating a constraint or binding dimensions.

Files changed: the exact four Model production/[Javadoc](../../../../glossary.md#term-javadoc) files, one focused Model test, and seven
documentation/[planning](../../../../glossary.md#term-planning) files listed under [Affected files](#affected-files).

[Validation](../../../../glossary.md#term-validation): the focused 22-test command and final 1,019-test/127-suite Model [run](../../../../glossary.md#term-run) passed in the
implementation context; the independent documentation context passed Model Javadoc, the compiled
and executed [API](../../../../glossary.md#term-application-programming-interface-api) example, generated-Javadoc inspection, Markdown with 695 local links, [bytecode](../../../../glossary.md#term-bytecode)
surface checks, exact 12-path [scope](../../../../glossary.md#term-scope), planning-status checks, and `git diff --check` without
repeating successful Java tests.

Documentation review: all four affected Javadocs, [Tensor](../../../../glossary.md#tensor) API, [Compile](../../../../glossary.md#compile) API, glossary, this task,
both relevant master plans, and roadmap required updates. Capabilities, Training/[Runtime](../../../../glossary.md#term-runtime) APIs,
related [operation](../../../../glossary.md#operation) contracts, architecture/ADRs/tests, [conformance](../../../../glossary.md#term-conformance)/integration, [Gradle](../../../../glossary.md#term-gradle)/Java 26,
dependencies, and other [modules](../../../../glossary.md#term-module) required no change for the reasons recorded above.

Unresolved issues: None within Model task 0025B.

Required follow-up: Draft [Compiler](../../../../glossary.md#term-compiler) 0005B must adopt the [occurrence](../../../../glossary.md#term-occurrence)-owned
source-one-or-source-equal predicate and align compiler [inference](../../../../glossary.md#term-inference), [preflight](../../../../glossary.md#term-preflight), and [gradient](../../../../glossary.md#term-gradient)
construction. Compiler 0005B–0005E and 0006 remain Draft without detailed task specifications.

Status: Complete
