# Task 0016F1: Masked Sum and [Mean](../../../../glossary.md#term-mean) [Tensor](../../../../glossary.md#tensor) Expressions

**Editorial reading key (later context, not a status update).**

The application passes a data Tensor, axis and BOOL mask; the helper resolves the historical ordered mask-to-input-axis mapping and records exact [input, mask] references. It does not reshape stored bytes, count selected elements or divide a mean. A mask excludes positions rather than multiplying them by zero, which matters for exceptional floating values. The linked current API uses right-aligned masks and a different empty-mean rule; read it as later navigation, not as an assertion that this task already selected those behaviors.

Focused current reading: [Current masked-reduction contract](../../../../reference/tensor/masked-index-reductions.md#masked-sum-and-mean-expressions). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Expose the completed masked-reduction semantics through public, [backend](../../../../glossary.md#backend)-independent
`Tensor.sum(axis, mask)` and `Tensor.mean(axis, mask)` expressions. Resolve each [mask](../../../../glossary.md#term-boolean-mask) [dimension](../../../../glossary.md#dimension) to
one ordered input [axis](../../../../glossary.md#term-axis) with deterministic local [Shape](../../../../glossary.md#shape) reasoning, remove the normalized reduction
axis from the result, and record one first-class masked [operation](../../../../glossary.md#operation) with exact ordered [provenance](../../../../glossary.md#provenance)
`[input, mask]`.

This task constructs expression [metadata](../../../../glossary.md#term-metadata) only. It does not align storage, reshape or expand a
mask, inspect values, perform a reduction, count selected values, divide a [mean](../../../../glossary.md#term-mean), create [gradient](../../../../glossary.md#term-gradient)
rules, [capture](../../../../glossary.md#term-capture-meanings) a [graph](../../../../glossary.md#graph), or execute backend work.

## [Scope](../../../../glossary.md#term-scope)

- Add exactly two public instance methods to [`Tensor`](../../../../glossary.md#tensor): `sum(int, Tensor)` and
  `mean(int, Tensor)`.
- Add one package-private final `TensorMaskedReductionExpressions` helper with one construction
  entry and private local Shape-mapping implementation.
- Accept only floating input and exact [BOOL](../../../../glossary.md#data-type--datatype) [mask](../../../../glossary.md#term-boolean-mask) data types, without promotion or conversion.
- Normalize the caller [axis](../../../../glossary.md#term-axis) exactly once through the input [Shape](../../../../glossary.md#shape).
- Require mask [rank](../../../../glossary.md#term-rank) not to exceed input rank.
- Resolve every mask [dimension](../../../../glossary.md#dimension) to one distinct, strictly increasing input-axis position.
- Treat equal immutable Dimensions as compatible and a static mask singleton as compatible with
  any input Dimension; reject every relationship that cannot be proved locally.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [singleton glossary](../../../../glossary.md#term-singleton))</span>
- Prefer a valid mapping that includes the reduction axis. Then minimize total positional
  displacement and use lexicographic axis order as the final deterministic tie-break.
- Support [scalar](../../../../glossary.md#term-scalar), static, zero-[extent](../../../../glossary.md#term-extent), and locally provable dynamic Shape relationships.
- Remove the normalized input axis, retaining every other exact Dimension reference in order.
- Preserve exact input [DataType](../../../../glossary.md#data-type--datatype) and `requiresGrad`, leave layout unresolved, and attach no label or
  host storage.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [host storage glossary](../../../../glossary.md#host-storage))</span>
- Construct exact `MaskedReductionAttrs(normalizedAxis, mapping)`, exact SUM or MEAN Operation,
  and exact ordered two-input provenance <span data-glossary-navigation="true"> ([MEAN glossary](../../../../glossary.md#term-mean); [Operation glossary](../../../../glossary.md#operation); [provenance glossary](../../../../glossary.md#provenance))</span>`[input, mask]`.
- Delegate [identity](../../../../glossary.md#term-identity-meanings) allocation exactly once to `TensorFactory.createDerived` and return a fresh
  Tensor for every valid call.
- Update the Tensor reflection test and add one focused masked-reduction expression test.
- Finalize Javadocs, Tensor API, Compile API, glossary, task evidence, master plan, and roadmap
  through the required independent documentation pass.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- a full/all-[axes](../../../../glossary.md#term-axis) masked form, `keepDimensions`, multiple axes, named axes, weights, or another
  overload<span data-glossary-navigation="true"> ([weights glossary](../../../../glossary.md#term-weight))</span>
- masked product, min, max, all, any, arg-max, [loss](../../../../glossary.md#loss), attention, indexing, or generic masking
- a public mapping resolver, mapping object, candidate, score, strategy, policy, builder, factory,
  registry, service, cache, or test hook<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [cache glossary](../../../../glossary.md#term-cache))</span>
- ordinary right-aligned [broadcasting](../../../../glossary.md#broadcasting) as a replacement for the ordered [mask](../../../../glossary.md#term-boolean-mask) mapping
- a legacy fixed numeric score penalty, mutable candidate list, reshape/expand expression, or
  hidden <span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable))</span>`where + zeros + sum/divide` [operation](../../../../glossary.md#operation) composition
- value or storage access, mask materialization, zero filling, counting, aggregation, division,
  allocation, copying, mutation, or output storage<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>
- numerical accumulation, [NaN](../../../../glossary.md#term-nan), [infinity](../../../../glossary.md#term-infinity), [signed zero](../../../../glossary.md#term-signed-zero), [precision](../../../../glossary.md#term-precision), [overflow](../../../../glossary.md#term-overflow), or empty-axis execution
- [gradients](../../../../glossary.md#term-gradient), [derivative](../../../../glossary.md#term-derivative) rules, mask gradients, [autograd](../../../../glossary.md#autograd), [optimizer](../../../../glossary.md#term-optimizer), or training execution
- cast insertion, DataType promotion, symbolic constraints/bindings, graph-wide inference, graph
  capture, canonicalization, common-subexpression elimination, or compiler lowering<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [bindings glossary](../../../../glossary.md#term-binding); [graph glossary](../../../../glossary.md#graph); [inference glossary](../../../../glossary.md#term-inference); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [common-subexpression elimination glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [compiler glossary](../../../../glossary.md#term-compiler); [lowering glossary](../../../../glossary.md#lowering))</span>
- changes to DataType, Dimension, Shape, ShapeBroadcast, TensorDescriptor, TensorProvenance,
  TensorFactory, Operation, reduction kinds/attributes, ordinary reductions, or their tests<span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension); [Shape glossary](../../../../glossary.md#shape))</span>
- planning, prepare, runtime, backend, engine, tracing, ONNX, dependencies, Gradle, architecture,
  another module, or a detailed task-0016G specification<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [engine glossary](../../../../glossary.md#term-engine); [ONNX glossary](../../../../glossary.md#term-onnx); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model master plan](../master-plan.md)
- [Task 0001](0001-data-type-model.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0006](0006-operation-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0016A](0016a-reduction-semantic-kinds-and-attributes.md)
- [Task 0016B](0016b-sum-mean-and-product-tensor-expressions.md)
- [Task 0016F](0016f-masked-reduction-semantics-and-axis-mapping.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes exactly the [axis](../../../../glossary.md#term-axis)-removing fluent forms
`sum(axis, mask)` and `mean(axis, mask)`. It requires floating input and a [BOOL](../../../../glossary.md#data-type--datatype) [mask](../../../../glossary.md#term-boolean-mask) whose [rank](../../../../glossary.md#term-rank) is
not greater than the input rank. Its mapping is more expressive than ordinary right-aligned
[broadcasting](../../../../glossary.md#broadcasting): mask `[batch, time]` aligns to axes `[0, 1]` of input
`[batch, time, features]`, behaving structurally like `[batch, time, 1]` when reducing `time`.
Legacy prefers placements involving the reduced axis, then uses positional scoring.

Legacy constructs reshape, expand, where, zero/one, sum, clamp, and divide expressions. Those
mechanisms are not copied. The new [model](../../../../glossary.md#term-model) already owns first-class `MaskedReductionAttrs`, so this
task records one SUM or [MEAN](../../../../glossary.md#term-mean) semantic [operation](../../../../glossary.md#operation) and ordered `[input, mask]` [provenance](../../../../glossary.md#provenance). Task 0016F
fixed false-value exclusion, true-count mean denominator, and zero for no selected values. Later
[compiler](../../../../glossary.md#term-compiler)/[backend](../../../../glossary.md#backend) work owns [lowering](../../../../glossary.md#lowering) and execution of that meaning.

## Architecture constraints

- [`Tensor`](../../../../glossary.md#tensor) remains public [mutable](../../../../glossary.md#term-mutable) [API](../../../../glossary.md#term-application-programming-interface-api) state, not an [IR](../../../../glossary.md#term-intermediate-representation-ir) [node](../../../../glossary.md#node) or executable value.
- Mapping resolution is deterministic local [Shape](../../../../glossary.md#shape) algebra inside `modules/model`; it creates no
  symbolic constraint, graph state, layout, stride, materialization, or backend plan.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [stride glossary](../../../../glossary.md#term-stride); [materialization glossary](../../../../glossary.md#materialization); [backend glossary](../../../../glossary.md#backend))</span>
- A mask mapping is an ordered injection from every mask dimension to input axes. Omitted input
  axes are implicit broadcast dimensions.<span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask))</span>
- Dimension compatibility is locally provable only when the Dimensions are equal or the mask
  Dimension is exactly `StaticDimension(1)`. A non-singleton mask dimension never expands an
  input singleton. Different dynamic symbols and dynamic/static non-singleton pairs are rejected.<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton))</span>
- Candidate selection uses this ordered priority:
  1. a mapping containing the normalized reduction [axis](../../../../glossary.md#term-axis) beats every mapping that omits it;
  2. lower `sum(abs(mappedAxis[i] - i))` wins;
  3. lexicographically smaller mapped-axis lists win.
- Because mapped positions are strictly increasing, the result is stable and independent of
  collection iteration order. A scalar mask resolves to the empty mapping.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- The resolver must use bounded dynamic programming or memoized search over Shape positions. It
  must not materialize every candidate mapping, retain global state, or use the legacy magic
  numeric noncoverage penalty.
- The result removes the [normalized axis](../../../../glossary.md#normalized-axis) and retains every unaffected [Dimension](../../../../glossary.md#dimension) reference.
- Result type and gradient eligibility come only from the floating input. BOOL mask metadata never
  requests or propagates gradients.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [BOOL glossary](../../../../glossary.md#data-type--datatype); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- `MaskedReductionAttrs` receives the normalized axis and resolved immutable mapping. Provenance
  order is exact <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [Provenance glossary](../../../../glossary.md#provenance))</span>`[input, mask]`; neither value is duplicated in attributes.
- Every result is fresh, unlabeled, storage-free, unresolved-[layout](../../../../glossary.md#layout), and allocated only through
  `TensorFactory.createDerived`.
- Package direction remains `model.tensor -> model.operation.reduction`, datatype, and shape. The
  reduction package must not import Tensor.
- Stop if implementation requires another production type, public Shape API, attribute change,
  numerical behavior, gradient rule, dependency, or architecture decision.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns public Tensor expression construction, local
  Shape mapping/derivation, descriptor creation, provenance, and the derived factory seam.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Shape glossary](../../../../glossary.md#shape); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- `io.github.pho001.synaptik.model.operation.reduction` — supplies SUM/MEAN and immutable masked
  attributes.<span data-glossary-navigation="true"> ([MEAN glossary](../../../../glossary.md#term-mean); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- `io.github.pho001.synaptik.model.shape` — supplies immutable Dimensions, Shapes, singleton
  recognition, and axis normalization.<span data-glossary-navigation="true"> ([Dimensions glossary](../../../../glossary.md#dimension); [singleton glossary](../../../../glossary.md#term-singleton); [axis glossary](../../../../glossary.md#term-axis); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- `io.github.pho001.synaptik.model.datatype` — supplies floating and [BOOL](../../../../glossary.md#data-type--datatype) categories.

No package is added.

Type placement:

- `io.github.pho001.synaptik.model.tensor.Tensor` — exact two public fluent overloads.
- `io.github.pho001.synaptik.model.tensor.TensorMaskedReductionExpressions` — package-private
  masked validation, mapping resolution, result Shape derivation, and construction boundary.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- `TensorMaskedReductionTest` — same-package focused contract and behavior test.

## Required contract

### Public [Tensor](../../../../glossary.md#tensor) surface

Add exactly:

```java
public Tensor sum(int axis, Tensor mask)
public Tensor mean(int axis, Tensor mask)
```

Each method is a public instance method that is non-static and non-synchronized. It delegates
exactly once to `TensorMaskedReductionExpressions.apply(this, mask, exactKind, axis)`. Neither
calls another public reduction overload.

### Helper surface

Create one final package-private, non-record helper with no fields or nested types, one private
zero-[argument](../../../../glossary.md#term-argument) constructor, and exactly one package-private static entry:

```java
static Tensor apply(
        Tensor input,
        Tensor mask,
        AggregateReductionKind kind,
        int axis)
```

Private static methods may implement kind/type [validation](../../../../glossary.md#term-validation), dynamic-programming mapping,
compatibility, result-[shape](../../../../glossary.md#shape) derivation, and final construction. Add no other package-private,
protected, or public method, overload, state, [cache](../../../../glossary.md#term-cache), strategy, service, resolver type, or test hook.

### [Validation](../../../../glossary.md#term-validation) and construction order

`apply` performs exactly:

1. `Objects.requireNonNull(input, "input")`.
2. `Objects.requireNonNull(mask, "mask")`.
3. `Objects.requireNonNull(kind, "kind")`.
4. Accept only SUM or [MEAN](../../../../glossary.md#term-mean); otherwise throw `IllegalArgumentException` with exact message
   `kind must be SUM or MEAN, but was <kind>`.
5. Require floating input; otherwise throw `IllegalArgumentException` with exact message
   `input must be a floating data type, but was <dataType>`.
6. Require exact [BOOL](../../../../glossary.md#data-type--datatype) [mask](../../../../glossary.md#term-boolean-mask); otherwise throw `IllegalArgumentException` with exact message
   `mask must have BOOL data type, but was <dataType>`.
7. Normalize [`axis`](../../../../glossary.md#term-axis) exactly once through `inputShape.normalizeAxis(axis)`.
8. Reject mask [rank](../../../../glossary.md#term-rank) greater than input rank with `IllegalArgumentException` and exact message
   `mask rank must not exceed input rank: mask=<maskRank>, input=<inputRank>`.
9. Resolve the deterministic mapping. If none exists, throw `IllegalArgumentException` with exact
   message `mask shape <maskShape> cannot be aligned to input shape <inputShape> for reduction axis <normalizedAxis>`.
10. Derive the axis-removing result [Shape](../../../../glossary.md#shape).
11. Construct attributes, descriptor, Operation, provenance, and the derived Tensor in the order
    specified below.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [Operation glossary](../../../../glossary.md#operation))</span>

Every failure before final factory delegation consumes no [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings). Do not inspect [layout](../../../../glossary.md#layout),
storage, values, [element](../../../../glossary.md#term-element) count, labels, existing [provenance](../../../../glossary.md#provenance), or [runtime](../../../../glossary.md#term-runtime) state during validation.

### Mapping resolution

For [mask](../../../../glossary.md#term-boolean-mask) [rank](../../../../glossary.md#term-rank) `m` and input rank `n`, produce a list `p` of length `m` satisfying:

- `0 <= p[0] < ... < p[m - 1] < n`;
- mask [Dimension](../../../../glossary.md#dimension) `i` is equal to input Dimension `p[i]`, or mask Dimension `i` is a static
  singleton;
- every mask dimension appears exactly once; omitted input [axes](../../../../glossary.md#term-axis) broadcast implicitly.

The [scalar](../../../../glossary.md#term-scalar) mask has the sole mapping `[]`. Equal-rank compatible [Shapes](../../../../glossary.md#shape) have the sole mapping
`[0, ..., n - 1]`. Static zero extents match equal zero extents, and a static [singleton](../../../../glossary.md#term-singleton) may align
to a zero or dynamic input Dimension. Dynamic Dimensions match only equal [canonical](../../../../glossary.md#term-canonical-form) symbols unless
the mask side is a static singleton.

Select among valid mappings using the three-level priority defined under Architecture constraints.
Examples:

- input `[batch, time, features]`, mask `[batch, time]`, reduce axis `1` -> `[0, 1]`;
- input `[batch, time, features]`, mask `[time, features]`, reduce axis `1` -> `[1, 2]`;
- input `[N, N, N]`, mask `[N]`, reduce axis `1` -> `[1]`;
- input `[2, 3, 4]`, mask `[2]`, reduce axis `1` -> `[0]` because no compatible mapping covers
  axis `1`;
- scalar mask -> `[]`.

The implementation uses polynomially bounded dynamic-programming or memoized Shape-position
states and deterministic reconstruction. It may allocate constructor-local arrays/lists but must
not store state, enumerate a retained list of all candidates, score with a magic penalty, mutate a
Shape, or call `ShapeBroadcast.broadcast` as a substitute for mapping.

### Result and [provenance](../../../../glossary.md#provenance) construction

Remove exactly the normalized input [axis](../../../../glossary.md#term-axis). Allocate one [rank](../../../../glossary.md#term-rank)-minus-one [Dimension](../../../../glossary.md#dimension) array, copy every
unaffected Dimension in order by exact reference, and call `Shape.ofDimensions` once. Rank one
produces [canonical](../../../../glossary.md#term-canonical-form) `Shape.scalar()`.

Construct in order:

1. `MaskedReductionAttrs(normalizedAxis, mapping)`;
2. one unresolved `TensorDescriptor` with exact input [DataType](../../../../glossary.md#data-type--datatype), derived [Shape](../../../../glossary.md#shape), and exact input
   `requiresGrad`;
3. one `Operation(kind, attrs)` retaining the exact attributes reference;
4. one `TensorProvenance(operation, List.of(input, mask))` retaining exact ordered references;
5. one `TensorFactory.createDerived(descriptor, Optional.empty(), provenance)` call.

The output is fresh, unlabeled, storage-free, and unresolved-[layout](../../../../glossary.md#layout). Input and [mask](../../../../glossary.md#term-boolean-mask) [descriptors](../../../../glossary.md#term-descriptor),
Shapes, labels, provenance, storage associations, and contents remain unchanged. SUM and [MEAN](../../../../glossary.md#term-mean)
record their fixed task-0016F meanings but perform no value work here.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorMaskedReductionExpressions.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMaskedReductionTest.java`

Documentation/[planning](../../../../glossary.md#term-planning):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless inconsistent: Training [API](../../../../glossary.md#term-application-programming-interface-api), capabilities, [Shape](../../../../glossary.md#shape)/[Dimension](../../../../glossary.md#dimension),
ordinary reduction helpers/tests, task-0016F semantic contracts/Javadocs, focused architecture,
ADRs/tests, [conformance](../../../../glossary.md#term-conformance)/integration tests, and [Gradle](../../../../glossary.md#term-gradle) configuration.

## Maximum [scope](../../../../glossary.md#term-scope)

At most two production files, two tests, and six documentation/[planning](../../../../glossary.md#term-planning) files: ten paths total.
[`Tensor`](../../../../glossary.md#tensor) and `TensorTest` may change only for the exact two public methods, imports/Javadocs,
reflection expectations, [signatures](../../../../glossary.md#term-signature), and non-[synchronization](../../../../glossary.md#term-synchronization) assertions. Do not modify
`TensorReductionExpressions`, [Shape](../../../../glossary.md#shape) contracts, masked attributes, or existing focused reduction
tests. Stop beyond this scope or if another type/file, public mapping [API](../../../../glossary.md#term-application-programming-interface-api), numerical policy,
storage access, [gradient](../../../../glossary.md#term-gradient) rule, [compiler](../../../../glossary.md#term-compiler) behavior, [dependency](../../../../glossary.md#term-dependency), or architecture change is needed.
Do not create task 0016G.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Document both public methods with mask mental model, mapping preference, axis removal, type
  requirements, result metadata, exact provenance, fixed masked numerical meaning, and deferred
  execution/gradient work.<span data-glossary-navigation="true"> ([mental model glossary](../../../../glossary.md#term-mental-model))</span><span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Document every parameter, returned Tensor, null/type/axis/rank/alignment/identity-exhaustion
  failure, ownership, and no-mutation behavior.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [Tensor glossary](../../../../glossary.md#tensor); [rank glossary](../../../../glossary.md#term-rank); [alignment glossary](../../../../glossary.md#term-alignment); [identity glossary](../../../../glossary.md#term-identity-meanings); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Document the helper type, private constructor, package-private entry, and every private method
  with validation order, mapping invariants, dynamic/singleton compatibility, tie-breaking,
  allocation/identity side effects, and failures.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [singleton glossary](../../../../glossary.md#term-singleton))</span>
- Include the concrete `[batch, time]` to `[batch, time, features]` example and explain every [axis](../../../../glossary.md#term-axis).
- Review Tensor, Dimension, Shape, MaskedReductionAttrs, AggregateReductionKind, descriptor,
  provenance, factory, and ordinary reduction Javadocs; change only permitted files or stop.<span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension); [Shape glossary](../../../../glossary.md#shape); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>

## Acceptance criteria

- Exactly two public Tensor methods and one helper are added with no other public/package-private
  API or production type.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Both public methods delegate once to the exact helper entry and semantic kind.
- Exact null/kind/input-type/mask-type/axis/rank/alignment order, exception types, messages, and
  no-ID-consumption behavior hold.<span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask); [rank glossary](../../../../glossary.md#term-rank); [alignment glossary](../../../../glossary.md#term-alignment))</span>
- Mapping handles scalar, equal-rank, left/right/noncontiguous placements, singleton Dimensions,
  zero extents, equal dynamic symbols, ambiguous repeated Dimensions, negative axes, and
  incompatible rank/dimensions deterministically.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [singleton glossary](../../../../glossary.md#term-singleton))</span>
- Mapping includes the reduction axis when possible, then minimizes displacement, then resolves
  ties lexicographically; no magic penalty or ordinary right-aligned shortcut defines semantics.
- Result [Shape](../../../../glossary.md#shape) removes one [axis](../../../../glossary.md#term-axis) and preserves every unaffected [Dimension](../../../../glossary.md#dimension) reference.
- Results have exact input type/eligibility, empty layout/label/storage, fresh identity, exact
  masked attributes, and ordered <span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>`[input, mask]` [provenance](../../../../glossary.md#provenance).
- Inputs and masks remain unchanged; no value, storage, gradient, graph, compiler, backend, build,
  or architecture behavior is added.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend))</span>
- Focused/aggregate tests, Javadoc, root tests, reflection/javap/bytecode/import/scope and docs
  validation pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope); [validation glossary](../../../../glossary.md#term-validation))</span>
- A separate clean-context docs agent finalizes permitted Javadocs/APIs/glossary/planning and
  records reasoned no-change conclusions.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- 0016F1 becomes Complete only after both passes; 0016G remains Draft without a specification.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorMaskedReductionTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorNumericReductionTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.reduction.MaskedReductionAttrsTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test covers exact helper/public surface and delegation; every validation precedence
and message; all floating input types and [BOOL](../../../../glossary.md#data-type--datatype)-only [mask](../../../../glossary.md#term-boolean-mask); positive/negative [axes](../../../../glossary.md#term-axis); [scalar](../../../../glossary.md#term-scalar)/static/
zero/dynamic [Shapes](../../../../glossary.md#shape); mapping examples and all preference/tie levels; result Shape references;
[descriptor](../../../../glossary.md#term-descriptor)/attributes/[provenance](../../../../glossary.md#provenance); freshness/nesting; ID side effects; and unchanged input/mask
[metadata](../../../../glossary.md#term-metadata), storage, contents, and prior reduction behavior.

Manually inspect `javap -p -c -s`, reflection, imports, and source for exact methods, one public
delegation, one axis [normalization](../../../../glossary.md#term-normalization), bounded local mapping states, deterministic reconstruction,
one result Shape construction, exact attributes, ordered provenance, one createDerived call, and
absence of ShapeBroadcast substitution, retained candidates, magic penalty, values/storage,
[gradient](../../../../glossary.md#term-gradient), cross-layer types, [registry](../../../../glossary.md#term-registry)/service, [dependency](../../../../glossary.md#term-dependency), or build change. Validate generated
[Javadoc](../../../../glossary.md#term-javadoc), APIs/glossary, links/anchors/fences/whitespace, exact ten paths, synchronized statuses,
and absence of a task-0016G specification.

## Dependencies

- 0001 supplies floating and [BOOL](../../../../glossary.md#data-type--datatype) DataTypes.
- 0002 supplies [immutable](../../../../glossary.md#term-immutable) static/dynamic [Dimensions](../../../../glossary.md#dimension), [Shapes](../../../../glossary.md#shape), and [axis](../../../../glossary.md#term-axis) [normalization](../../../../glossary.md#term-normalization).
- 0006–0007 supply [Operation](../../../../glossary.md#operation) and TensorDescriptor.
- 0011–0013 supply [Tensor](../../../../glossary.md#tensor), central [identity](../../../../glossary.md#term-identity-meanings) allocation, [provenance](../../../../glossary.md#provenance), and createDerived.
- 0016A supplies SUM/[MEAN](../../../../glossary.md#term-mean) identities; 0016B establishes ordinary axis-removing result behavior.
- 0016F supplies MaskedReductionAttrs and fixed masked SUM/MEAN meaning.

## Follow-up tasks

- 0016G remains Draft for cumulative-sum semantic kind and attributes.
- [Compiler](../../../../glossary.md#term-compiler) tasks own [capture](../../../../glossary.md#term-capture-meanings), [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form), [Shape](../../../../glossary.md#shape) revalidation, and [autograd](../../../../glossary.md#autograd) expansion.
- Backend/config/conformance tasks own mask alignment interpretation, value selection, true-count
  mean denominator, all-false zero results, numerical accumulation, storage, kernels, and
  cross-backend parity.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [mask glossary](../../../../glossary.md#term-boolean-mask); [alignment glossary](../../../../glossary.md#term-alignment); [mean glossary](../../../../glossary.md#term-mean); [kernels glossary](../../../../glossary.md#kernel))</span>

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. This task composes existing [model](../../../../glossary.md#term-model)-owned [Tensor](../../../../glossary.md#tensor), [Shape](../../../../glossary.md#shape), reduction semantic,
[descriptor](../../../../glossary.md#term-descriptor), [provenance](../../../../glossary.md#provenance), and factory contracts without executable or cross-layer state. Stop if an
architecture change is required.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0002/0006/0007/0011/0012/0013/0016A/0016B/0016F/
0016F1, Tensor API, Compile API, Training API, glossary, current DataType/Dimension/Shape/
TensorDescriptor/Tensor/TensorFactory/TensorProvenance/Operation/AggregateReductionKind/
MaskedReductionAttrs contracts and tests, ordinary reduction helper/tests, and Java 26 Gradle.

Implement task 0016F1 exactly. Modify Tensor.java and add package-private final
TensorMaskedReductionExpressions.java. Update TensorTest only for the exact two-method API and add
TensorMaskedReductionTest. Add exactly sum(axis,mask) and mean(axis,mask), each delegating once to
the shared helper and exact kind.

The helper validates input/mask/kind, floating/BOOL types, normalizes the axis once, and resolves
an ordered injective mask-dimension-to-input-axis mapping. Equal Dimensions and mask singletons
are compatible. Prefer mappings covering the reduction axis, then minimum positional displacement,
then lexicographic order, using bounded DP/memoized Shape-position states without retained all-
candidate enumeration or a magic penalty. Remove the result axis, preserve unaffected Dimension
references, create exact MaskedReductionAttrs, unchanged input type/eligibility, unresolved layout,
ordered [input,mask] provenance, and one createDerived call. Every result is fresh.

Do not inspect values/storage, compose where/zeros/ordinary reductions, modify Shape or existing
reduction contracts, define gradients, capture graphs, add overloads/types, or introduce compiler/
runtime/backend behavior. Stop beyond ten paths or on architecture uncertainty.

Run all task validation, then hand the actual diff/evidence to a separate clean-context docs agent
in the same change. It must inspect source/tests/generated Javadoc, finalize permitted Javadocs/
Tensor API/Compile API/glossary/planning, record related-contract/capability/Training API/
architecture no-change conclusions, and rerun validation.

Update task 0016F1, model master plan, and roadmap only for planning status/evidence. Do not mark
0016F1 Complete until both passes succeed. Leave 0016G Draft without a specification. Do not
commit or push.
```

## Local decisions

- Keep exactly the two legacy-compatible axis-removing fluent overloads; do not speculate about
  masked full or retained-axis forms.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>
- Represent one [masked reduction](../../../../glossary.md#masked-reduction) directly with existing attributes and `[input, mask]` provenance
  instead of manufacturing a model-time subgraph of where/constant/ordinary operations.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [model glossary](../../../../glossary.md#term-model); [constant glossary](../../../../glossary.md#term-constant))</span>
- Keep mapping resolution package-private with its sole consumer; one use does not justify a
  public Shape or policy API.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Replace the legacy magic noncoverage penalty with explicit ordered preferences: cover the
  reduced axis when possible, minimize displacement, then choose lexicographically.
- Accept a static mask singleton against any input Dimension, including dynamic or zero extent;
  otherwise require exact immutable Dimension equality so no symbolic constraint is invented.<span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask); [singleton glossary](../../../../glossary.md#term-singleton); [Dimension glossary](../../../../glossary.md#dimension); [extent glossary](../../../../glossary.md#term-extent); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Preserve input gradient eligibility as model intent while deferring gradient rules. The BOOL
  mask never contributes eligibility.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Use bounded dynamic programming or memoized states to avoid retaining an exponential candidate
  collection while keeping deterministic behavior for ambiguous repeated dimensions.

## Known limitations

- No values, storage [alignment](../../../../glossary.md#term-alignment), [mask](../../../../glossary.md#term-boolean-mask) [materialization](../../../../glossary.md#materialization), numerical reduction, or [gradients](../../../../glossary.md#term-gradient).
- [Axis](../../../../glossary.md#term-axis)-removing floating SUM/[MEAN](../../../../glossary.md#term-mean) only; no full, retained-axis, weighted, or other masked family.
- Dynamic compatibility is limited to equal symbols or a static mask [singleton](../../../../glossary.md#term-singleton).
- No [compiler](../../../../glossary.md#term-compiler) [capture](../../../../glossary.md#term-capture-meanings)/[canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form)/[autograd](../../../../glossary.md#autograd), [ONNX](../../../../glossary.md#term-onnx)/[backend](../../../../glossary.md#backend) support, or execution.

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) reviewed the [architecture contract](../../../../glossary.md#architecture-contract) and focused architecture docs; documentation/planning
rules; roadmap; [model](../../../../glossary.md#term-model) capabilities/master plan; tasks 0001, 0002, 0006, 0007, 0011–0013, 0016A,
0016B, and 0016F; current [DataType](../../../../glossary.md#data-type--datatype)/[Dimension](../../../../glossary.md#dimension)/[Shape](../../../../glossary.md#shape)/[Tensor](../../../../glossary.md#tensor)/[descriptor](../../../../glossary.md#term-descriptor)/factory/[provenance](../../../../glossary.md#provenance)/reduction
contracts and tests; Tensor/[Compile](../../../../glossary.md#compile)/Training APIs; glossary; and Java 26 [Gradle](../../../../glossary.md#term-gradle) configuration.

The legacy branch was read directly. It confirms the two public overloads, floating/[BOOL](../../../../glossary.md#data-type--datatype) and [rank](../../../../glossary.md#term-rank)
requirements, [axis](../../../../glossary.md#term-axis) removal, negative-axis [normalization](../../../../glossary.md#term-normalization), ordered non-right-aligned placement,
preferred reduced-axis coverage, padded-sequence `[batch,time]` mapping, false exclusion,
true-count [mean](../../../../glossary.md#term-mean) denominator, and all-false zero result. Legacy [mutable](../../../../glossary.md#term-mutable) candidates, magic scoring,
reshape/expand, where/[constants](../../../../glossary.md#term-constant), callbacks, storage, [runtime](../../../../glossary.md#term-runtime) state, [lowering](../../../../glossary.md#lowering), and [kernels](../../../../glossary.md#kernel) are
excluded or reassigned.

Planning selected two public methods and one package-private helper. Existing model contracts
suffice; no package, [dependency](../../../../glossary.md#term-dependency), foundational contract, or architecture change is required.

Pre-implementation planning validation after synchronizing this task, the model master plan, and
roadmap:

- `git diff --check` passed.
- The trailing-whitespace [scan](../../../../glossary.md#term-cumulative-scan) returned no matches across the three planning files.
- All 169 relative Markdown links across the three planning files resolve locally.
- Markdown fence counts are balanced: eight in this task, two in the master plan, and zero in the
  roadmap.
- All 21 required task-specification headings are present.
- At that planning checkpoint, task/master/roadmap statuses consistently identified 0016F1 as
  Ready and 0016G as Draft.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [Ready glossary](../../../../glossary.md#term-ready))</span>
- No detailed task-0016G specification exists.
- Repository scope is exactly this task, the model master plan, and roadmap; no Java, API,
  architecture, Gradle, or other file changed during planning.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

Implementation validation and the independent clean-context documentation-focused review are now
complete.

Implementation and focused validation:

- `./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorMaskedReductionTest --tests io.github.pho001.synaptik.model.tensor.TensorTest --tests io.github.pho001.synaptik.model.tensor.TensorNumericReductionTest --tests io.github.pho001.synaptik.model.operation.reduction.MaskedReductionAttrsTest`
  passed: 44 tests across the four requested classes, with zero failures, errors, or skips.
- `./gradlew :modules:model:test :modules:model:javadoc test --rerun-tasks` passed. The model suite
  reports 417 tests with zero failures, errors, or skips; root `test` completed successfully; and
  model Javadoc generated without warnings or errors.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- `javap -p -c -s` confirmed exactly the two new public descriptors. Each public method performs
  one call to `TensorMaskedReductionExpressions.apply`, with exact `SUM` or `MEAN`, and the helper
  bytecode confirms one axis normalization, bounded rank-indexed dynamic-programming arrays,
  deterministic reconstruction, one <span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode))</span>`Shape.ofDimensions` result construction, ordered
  `List.of(input, mask)` provenance, and one `TensorFactory.createDerived` call.
- Reflection tests confirm the helper is package-private final with no fields or nested types, one
  private constructor, one package-private entry, four private methods, and no extra public or
  protected API. `Tensor` has exactly 72 declared public methods and both new methods are public,
  non-static, and non-synchronized.
- Source/import inspection found no `ShapeBroadcast`, retained complete-candidate collection,
  magic penalty, value/storage access, gradient implementation, compiler/runtime/backend type,
  registry, service, dependency, or build change.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [registry glossary](../../../../glossary.md#term-registry))</span>

Independent documentation-focused review:

- Clean one-shot Codex documentation context `019f3234-136e-7860-b3af-e99441339b90`, launched by
  `/root` after the collaboration thread-limit fallback, applied General style, API and Javadoc
  style, Planning style, and Example format. It independently inspected the final implementation
  diff, all four changed Java paths, focused and aggregate test evidence, generated Javadoc,
  Tensor/Compile/Training APIs, glossary, model capabilities/master plan/roadmap, Shape,
  Dimension, Tensor, descriptor, factory, provenance, ordinary reduction, aggregate kind, and
  masked-attribute contracts.
- The affected `Tensor` and `TensorMaskedReductionExpressions` Javadocs are complete and accurate:
  every parameter and result is documented, caller-visible failure conditions and identity side
  effects are stated, ownership/no-mutation boundaries are explicit, and the concrete<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [identity glossary](../../../../glossary.md#term-identity-meanings); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
  `[batch, time]` to `[batch, time, features]` mapping explains every axis. No additional Java
  documentation edit was required.
- `docs/api/tensor-api.md` now documents public masked sum/mean construction, deterministic mapping,
  metadata/provenance, failures, deferred behavior, and one complete verified example.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>
  `docs/api/compile-api.md` now lists masked construction among current expression inputs while
  preserving compiler capture and transformation as planned. <span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings))</span>`docs/glossary.md` now reflects the
  implemented masked-reduction and mask-to-input mapping boundaries.<span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask))</span>
- Training API requires no change because no training, autograd, gradient publication, optimizer,
  or execution contract changed. Model capabilities require no change because the selected legacy
  capability and its deferred compiler/backend responsibilities were already represented.<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [publication glossary](../../../../glossary.md#term-publication); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- Architecture, focused architecture explanations, ADRs, and architecture tests require no change
  because module ownership and dependency direction are unchanged. DataType, Dimension, Shape,
  ShapeBroadcast, TensorDescriptor, TensorFactory, TensorProvenance, Operation, aggregate semantic
  kinds/attributes, ordinary and arg-max reductions, and their tests remain accurate unchanged
  because this task only composes them through a new package-private helper and two Tensor methods.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [Operation glossary](../../../../glossary.md#operation))</span>
- Backend conformance and integration tests require no change because there is no numerical,
  lowering, backend, or end-to-end execution behavior. Gradle requires no change because no
  dependency, source set, language level, preview feature, task, or module was added. No other
  module is affected.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [preview feature glossary](../../../../glossary.md#term-preview-feature))</span>

Documentation and final validation:

- `./gradlew :modules:model:javadoc` and the aggregate Gradle command above passed; generated
  `Tensor.html` contains both new method anchors and rendered contracts.
- The documented example was [run](../../../../glossary.md#term-run) through the Java 26 source launcher against
  `modules/model/build/classes/java/main`; it exited zero and produced exactly the documented
  shapes, axis, mapping, provenance-order, and metadata output. An earlier JShell run produced the
  same output but its process could not flush macOS preferences inside the sandbox; the source
  launcher removed that environmental limitation.
- The final local Markdown link-and-anchor check examined 257 links across the six permitted
  documentation/planning files and found zero errors. The first checker attempt used unavailable
  Ruby `filter_map`; a corrected compatibility form then exposed a checker-only slash-anchor slug
  mismatch, and the final GitHub-style slug check passed.
- Fence, terminology, trailing-whitespace, and status checks passed. `git diff --check` passed.
  The final worktree scope is exactly the authorized ten paths: two production files, two tests,
  and six documentation/planning files.
- Task, model master plan, and roadmap statuses consistently mark 0016F1 Complete. Task 0016G
  remains Draft, and no `0016g-*.md` task specification exists.

## Implementation notes

- Added only `Tensor.sum(int, Tensor)` and `Tensor.mean(int, Tensor)`, each delegating once to the
  exact package-private helper entry and aggregate kind.
- Added `TensorMaskedReductionExpressions` with ordered validation, bounded dynamic programming,
  deterministic reduced-axis/displacement/lexicographic preference, exact axis-removing Shape
  derivation, and one derived Tensor construction.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [axis glossary](../../../../glossary.md#term-axis); [Shape glossary](../../../../glossary.md#shape); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Added focused coverage for surface, validation order/messages, ID side effects, floating/BOOL
  eligibility, scalar/static/zero/dynamic mapping, every tie-break level, reference preservation,
  metadata, provenance, freshness, nesting, and unchanged inputs. Updated Tensor reflection
  expectations only for the two methods.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [scalar glossary](../../../../glossary.md#term-scalar); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance))</span>
- Finalized only the six authorized documentation/planning paths. No Java declaration, behavior,
  or test was changed during this documentation-focused review.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Completion summary

- Completed changes: public masked sum/mean expression construction with deterministic ordered
  Shape mapping, axis-removing metadata, and exact two-input provenance; focused tests and public
  documentation are complete.<span data-glossary-navigation="true"> ([mean glossary](../../../../glossary.md#term-mean); [Shape glossary](../../../../glossary.md#shape); [axis glossary](../../../../glossary.md#term-axis); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance))</span>
- Files changed or created: exactly the two production, two test, and six documentation/planning
  paths listed under Affected files.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: all focused, model aggregate, Javadoc, root, reflection, bytecode, example,
  link/anchor, terminology, fence, whitespace, scope, and status checks passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [model glossary](../../../../glossary.md#term-model); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: clean one-shot Codex context
  `019f3234-136e-7860-b3af-e99441339b90`, launched by `/root` after the collaboration thread-limit
  fallback, completed the independent pass using the API/Javadoc and Planning profiles plus
  Example format.
- Documentation impact: [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api), [Compile](../../../../glossary.md#compile) API, glossary, task, master plan, and roadmap finalized.
- [Javadoc](../../../../glossary.md#term-javadoc) review: affected Tensor and helper Javadocs are complete; no further edit required.
- Glossary impact: masked reduction and mask-to-input axis mapping now describe current public
  construction while preserving deferred numerical and cross-layer work.<span data-glossary-navigation="true"> ([masked reduction glossary](../../../../glossary.md#masked-reduction); [mask glossary](../../../../glossary.md#term-boolean-mask))</span>
- Unresolved issues: None.
- Follow-up required: None. Task 0016G remains a separate Draft planning frontier without a
  detailed specification.

Status: Complete
