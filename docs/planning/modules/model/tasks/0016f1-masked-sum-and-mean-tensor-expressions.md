# Task 0016F1: Masked Sum and Mean Tensor Expressions

## Status

Complete

## Goal

Expose the completed masked-reduction semantics through public, backend-independent
`Tensor.sum(axis, mask)` and `Tensor.mean(axis, mask)` expressions. Resolve each mask dimension to
one ordered input axis with deterministic local Shape reasoning, remove the normalized reduction
axis from the result, and record one first-class masked operation with exact ordered provenance
`[input, mask]`.

This task constructs expression metadata only. It does not align storage, reshape or expand a
mask, inspect values, perform a reduction, count selected values, divide a mean, create gradient
rules, capture a graph, or execute backend work.

## Scope

- Add exactly two public instance methods to `Tensor`: `sum(int, Tensor)` and
  `mean(int, Tensor)`.
- Add one package-private final `TensorMaskedReductionExpressions` helper with one construction
  entry and private local Shape-mapping implementation.
- Accept only floating input and exact BOOL mask data types, without promotion or conversion.
- Normalize the caller axis exactly once through the input Shape.
- Require mask rank not to exceed input rank.
- Resolve every mask dimension to one distinct, strictly increasing input-axis position.
- Treat equal immutable Dimensions as compatible and a static mask singleton as compatible with
  any input Dimension; reject every relationship that cannot be proved locally.
- Prefer a valid mapping that includes the reduction axis. Then minimize total positional
  displacement and use lexicographic axis order as the final deterministic tie-break.
- Support scalar, static, zero-extent, and locally provable dynamic Shape relationships.
- Remove the normalized input axis, retaining every other exact Dimension reference in order.
- Preserve exact input DataType and `requiresGrad`, leave layout unresolved, and attach no label or
  host storage.
- Construct exact `MaskedReductionAttrs(normalizedAxis, mapping)`, exact SUM or MEAN Operation,
  and exact ordered two-input provenance `[input, mask]`.
- Delegate identity allocation exactly once to `TensorFactory.createDerived` and return a fresh
  Tensor for every valid call.
- Update the Tensor reflection test and add one focused masked-reduction expression test.
- Finalize Javadocs, Tensor API, Compile API, glossary, task evidence, master plan, and roadmap
  through the required independent documentation pass.

## Out of scope

- a full/all-axes masked form, `keepDimensions`, multiple axes, named axes, weights, or another
  overload
- masked product, min, max, all, any, arg-max, loss, attention, indexing, or generic masking
- a public mapping resolver, mapping object, candidate, score, strategy, policy, builder, factory,
  registry, service, cache, or test hook
- ordinary right-aligned broadcasting as a replacement for the ordered mask mapping
- a legacy fixed numeric score penalty, mutable candidate list, reshape/expand expression, or
  hidden `where + zeros + sum/divide` operation composition
- value or storage access, mask materialization, zero filling, counting, aggregation, division,
  allocation, copying, mutation, or output storage
- numerical accumulation, NaN, infinity, signed zero, precision, overflow, or empty-axis execution
- gradients, derivative rules, mask gradients, autograd, optimizer, or training execution
- cast insertion, DataType promotion, symbolic constraints/bindings, graph-wide inference, graph
  capture, canonicalization, common-subexpression elimination, or compiler lowering
- changes to DataType, Dimension, Shape, ShapeBroadcast, TensorDescriptor, TensorProvenance,
  TensorFactory, Operation, reduction kinds/attributes, ordinary reductions, or their tests
- planning, prepare, runtime, backend, engine, tracing, ONNX, dependencies, Gradle, architecture,
  another module, or a detailed task-0016G specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
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
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes exactly the axis-removing fluent forms
`sum(axis, mask)` and `mean(axis, mask)`. It requires floating input and a BOOL mask whose rank is
not greater than the input rank. Its mapping is more expressive than ordinary right-aligned
broadcasting: mask `[batch, time]` aligns to axes `[0, 1]` of input
`[batch, time, features]`, behaving structurally like `[batch, time, 1]` when reducing `time`.
Legacy prefers placements involving the reduced axis, then uses positional scoring.

Legacy constructs reshape, expand, where, zero/one, sum, clamp, and divide expressions. Those
mechanisms are not copied. The new model already owns first-class `MaskedReductionAttrs`, so this
task records one SUM or MEAN semantic operation and ordered `[input, mask]` provenance. Task 0016F
fixed false-value exclusion, true-count mean denominator, and zero for no selected values. Later
compiler/backend work owns lowering and execution of that meaning.

## Architecture constraints

- `Tensor` remains public mutable API state, not an IR node or executable value.
- Mapping resolution is deterministic local Shape algebra inside `modules/model`; it creates no
  symbolic constraint, graph state, layout, stride, materialization, or backend plan.
- A mask mapping is an ordered injection from every mask dimension to input axes. Omitted input
  axes are implicit broadcast dimensions.
- Dimension compatibility is locally provable only when the Dimensions are equal or the mask
  Dimension is exactly `StaticDimension(1)`. A non-singleton mask dimension never expands an
  input singleton. Different dynamic symbols and dynamic/static non-singleton pairs are rejected.
- Candidate selection uses this ordered priority:
  1. a mapping containing the normalized reduction axis beats every mapping that omits it;
  2. lower `sum(abs(mappedAxis[i] - i))` wins;
  3. lexicographically smaller mapped-axis lists win.
- Because mapped positions are strictly increasing, the result is stable and independent of
  collection iteration order. A scalar mask resolves to the empty mapping.
- The resolver must use bounded dynamic programming or memoized search over Shape positions. It
  must not materialize every candidate mapping, retain global state, or use the legacy magic
  numeric noncoverage penalty.
- The result removes the normalized axis and retains every unaffected Dimension reference.
- Result type and gradient eligibility come only from the floating input. BOOL mask metadata never
  requests or propagates gradients.
- `MaskedReductionAttrs` receives the normalized axis and resolved immutable mapping. Provenance
  order is exact `[input, mask]`; neither value is duplicated in attributes.
- Every result is fresh, unlabeled, storage-free, unresolved-layout, and allocated only through
  `TensorFactory.createDerived`.
- Package direction remains `model.tensor -> model.operation.reduction`, datatype, and shape. The
  reduction package must not import Tensor.
- Stop if implementation requires another production type, public Shape API, attribute change,
  numerical behavior, gradient rule, dependency, or architecture decision.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns public Tensor expression construction, local
  Shape mapping/derivation, descriptor creation, provenance, and the derived factory seam.
- `io.github.pho001.synaptik.model.operation.reduction` — supplies SUM/MEAN and immutable masked
  attributes.
- `io.github.pho001.synaptik.model.shape` — supplies immutable Dimensions, Shapes, singleton
  recognition, and axis normalization.
- `io.github.pho001.synaptik.model.datatype` — supplies floating and BOOL categories.

No package is added.

Type placement:

- `io.github.pho001.synaptik.model.tensor.Tensor` — exact two public fluent overloads.
- `io.github.pho001.synaptik.model.tensor.TensorMaskedReductionExpressions` — package-private
  masked validation, mapping resolution, result Shape derivation, and construction boundary.
- `TensorMaskedReductionTest` — same-package focused contract and behavior test.

## Required contract

### Public Tensor surface

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
zero-argument constructor, and exactly one package-private static entry:

```java
static Tensor apply(
        Tensor input,
        Tensor mask,
        AggregateReductionKind kind,
        int axis)
```

Private static methods may implement kind/type validation, dynamic-programming mapping,
compatibility, result-shape derivation, and final construction. Add no other package-private,
protected, or public method, overload, state, cache, strategy, service, resolver type, or test hook.

### Validation and construction order

`apply` performs exactly:

1. `Objects.requireNonNull(input, "input")`.
2. `Objects.requireNonNull(mask, "mask")`.
3. `Objects.requireNonNull(kind, "kind")`.
4. Accept only SUM or MEAN; otherwise throw `IllegalArgumentException` with exact message
   `kind must be SUM or MEAN, but was <kind>`.
5. Require floating input; otherwise throw `IllegalArgumentException` with exact message
   `input must be a floating data type, but was <dataType>`.
6. Require exact BOOL mask; otherwise throw `IllegalArgumentException` with exact message
   `mask must have BOOL data type, but was <dataType>`.
7. Normalize `axis` exactly once through `inputShape.normalizeAxis(axis)`.
8. Reject mask rank greater than input rank with `IllegalArgumentException` and exact message
   `mask rank must not exceed input rank: mask=<maskRank>, input=<inputRank>`.
9. Resolve the deterministic mapping. If none exists, throw `IllegalArgumentException` with exact
   message `mask shape <maskShape> cannot be aligned to input shape <inputShape> for reduction axis <normalizedAxis>`.
10. Derive the axis-removing result Shape.
11. Construct attributes, descriptor, Operation, provenance, and the derived Tensor in the order
    specified below.

Every failure before final factory delegation consumes no Tensor identity. Do not inspect layout,
storage, values, element count, labels, existing provenance, or runtime state during validation.

### Mapping resolution

For mask rank `m` and input rank `n`, produce a list `p` of length `m` satisfying:

- `0 <= p[0] < ... < p[m - 1] < n`;
- mask Dimension `i` is equal to input Dimension `p[i]`, or mask Dimension `i` is a static
  singleton;
- every mask dimension appears exactly once; omitted input axes broadcast implicitly.

The scalar mask has the sole mapping `[]`. Equal-rank compatible Shapes have the sole mapping
`[0, ..., n - 1]`. Static zero extents match equal zero extents, and a static singleton may align
to a zero or dynamic input Dimension. Dynamic Dimensions match only equal canonical symbols unless
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

### Result and provenance construction

Remove exactly the normalized input axis. Allocate one rank-minus-one Dimension array, copy every
unaffected Dimension in order by exact reference, and call `Shape.ofDimensions` once. Rank one
produces canonical `Shape.scalar()`.

Construct in order:

1. `MaskedReductionAttrs(normalizedAxis, mapping)`;
2. one unresolved `TensorDescriptor` with exact input DataType, derived Shape, and exact input
   `requiresGrad`;
3. one `Operation(kind, attrs)` retaining the exact attributes reference;
4. one `TensorProvenance(operation, List.of(input, mask))` retaining exact ordered references;
5. one `TensorFactory.createDerived(descriptor, Optional.empty(), provenance)` call.

The output is fresh, unlabeled, storage-free, and unresolved-layout. Input and mask descriptors,
Shapes, labels, provenance, storage associations, and contents remain unchanged. SUM and MEAN
record their fixed task-0016F meanings but perform no value work here.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorMaskedReductionExpressions.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMaskedReductionTest.java`

Documentation/planning:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless inconsistent: Training API, capabilities, Shape/Dimension,
ordinary reduction helpers/tests, task-0016F semantic contracts/Javadocs, focused architecture,
ADRs/tests, conformance/integration tests, and Gradle configuration.

## Maximum scope

At most two production files, two tests, and six documentation/planning files: ten paths total.
`Tensor` and `TensorTest` may change only for the exact two public methods, imports/Javadocs,
reflection expectations, signatures, and non-synchronization assertions. Do not modify
`TensorReductionExpressions`, Shape contracts, masked attributes, or existing focused reduction
tests. Stop beyond this scope or if another type/file, public mapping API, numerical policy,
storage access, gradient rule, compiler behavior, dependency, or architecture change is needed.
Do not create task 0016G.

## Javadoc requirements

- Document both public methods with mask mental model, mapping preference, axis removal, type
  requirements, result metadata, exact provenance, fixed masked numerical meaning, and deferred
  execution/gradient work.
- Document every parameter, returned Tensor, null/type/axis/rank/alignment/identity-exhaustion
  failure, ownership, and no-mutation behavior.
- Document the helper type, private constructor, package-private entry, and every private method
  with validation order, mapping invariants, dynamic/singleton compatibility, tie-breaking,
  allocation/identity side effects, and failures.
- Include the concrete `[batch, time]` to `[batch, time, features]` example and explain every axis.
- Review Tensor, Dimension, Shape, MaskedReductionAttrs, AggregateReductionKind, descriptor,
  provenance, factory, and ordinary reduction Javadocs; change only permitted files or stop.

## Acceptance criteria

- Exactly two public Tensor methods and one helper are added with no other public/package-private
  API or production type.
- Both public methods delegate once to the exact helper entry and semantic kind.
- Exact null/kind/input-type/mask-type/axis/rank/alignment order, exception types, messages, and
  no-ID-consumption behavior hold.
- Mapping handles scalar, equal-rank, left/right/noncontiguous placements, singleton Dimensions,
  zero extents, equal dynamic symbols, ambiguous repeated Dimensions, negative axes, and
  incompatible rank/dimensions deterministically.
- Mapping includes the reduction axis when possible, then minimizes displacement, then resolves
  ties lexicographically; no magic penalty or ordinary right-aligned shortcut defines semantics.
- Result Shape removes one axis and preserves every unaffected Dimension reference.
- Results have exact input type/eligibility, empty layout/label/storage, fresh identity, exact
  masked attributes, and ordered `[input, mask]` provenance.
- Inputs and masks remain unchanged; no value, storage, gradient, graph, compiler, backend, build,
  or architecture behavior is added.
- Focused/aggregate tests, Javadoc, root tests, reflection/javap/bytecode/import/scope and docs
  validation pass.
- A separate clean-context docs agent finalizes permitted Javadocs/APIs/glossary/planning and
  records reasoned no-change conclusions.
- 0016F1 becomes Complete only after both passes; 0016G remains Draft without a specification.

## Tests / validation

Run before and after documentation review:

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
and message; all floating input types and BOOL-only mask; positive/negative axes; scalar/static/
zero/dynamic Shapes; mapping examples and all preference/tie levels; result Shape references;
descriptor/attributes/provenance; freshness/nesting; ID side effects; and unchanged input/mask
metadata, storage, contents, and prior reduction behavior.

Manually inspect `javap -p -c -s`, reflection, imports, and source for exact methods, one public
delegation, one axis normalization, bounded local mapping states, deterministic reconstruction,
one result Shape construction, exact attributes, ordered provenance, one createDerived call, and
absence of ShapeBroadcast substitution, retained candidates, magic penalty, values/storage,
gradient, cross-layer types, registry/service, dependency, or build change. Validate generated
Javadoc, APIs/glossary, links/anchors/fences/whitespace, exact ten paths, synchronized statuses,
and absence of a task-0016G specification.

## Dependencies

- 0001 supplies floating and BOOL DataTypes.
- 0002 supplies immutable static/dynamic Dimensions, Shapes, and axis normalization.
- 0006–0007 supply Operation and TensorDescriptor.
- 0011–0013 supply Tensor, central identity allocation, provenance, and createDerived.
- 0016A supplies SUM/MEAN identities; 0016B establishes ordinary axis-removing result behavior.
- 0016F supplies MaskedReductionAttrs and fixed masked SUM/MEAN meaning.

## Follow-up tasks

- 0016G remains Draft for cumulative-sum semantic kind and attributes.
- Compiler tasks own capture, canonicalization, Shape revalidation, and autograd expansion.
- Backend/config/conformance tasks own mask alignment interpretation, value selection, true-count
  mean denominator, all-false zero results, numerical accumulation, storage, kernels, and
  cross-backend parity.

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. This task composes existing model-owned Tensor, Shape, reduction semantic,
descriptor, provenance, and factory contracts without executable or cross-layer state. Stop if an
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
  masked full or retained-axis forms.
- Represent one masked reduction directly with existing attributes and `[input, mask]` provenance
  instead of manufacturing a model-time subgraph of where/constant/ordinary operations.
- Keep mapping resolution package-private with its sole consumer; one use does not justify a
  public Shape or policy API.
- Replace the legacy magic noncoverage penalty with explicit ordered preferences: cover the
  reduced axis when possible, minimize displacement, then choose lexicographically.
- Accept a static mask singleton against any input Dimension, including dynamic or zero extent;
  otherwise require exact immutable Dimension equality so no symbolic constraint is invented.
- Preserve input gradient eligibility as model intent while deferring gradient rules. The BOOL
  mask never contributes eligibility.
- Use bounded dynamic programming or memoized states to avoid retaining an exponential candidate
  collection while keeping deterministic behavior for ambiguous repeated dimensions.

## Known limitations

- No values, storage alignment, mask materialization, numerical reduction, or gradients.
- Axis-removing floating SUM/MEAN only; no full, retained-axis, weighted, or other masked family.
- Dynamic compatibility is limited to equal symbols or a static mask singleton.
- No compiler capture/canonicalization/autograd, ONNX/backend support, or execution.

## Validation evidence

Planning reviewed the architecture contract and focused architecture docs; documentation/planning
rules; roadmap; model capabilities/master plan; tasks 0001, 0002, 0006, 0007, 0011–0013, 0016A,
0016B, and 0016F; current DataType/Dimension/Shape/Tensor/descriptor/factory/provenance/reduction
contracts and tests; Tensor/Compile/Training APIs; glossary; and Java 26 Gradle configuration.

The legacy branch was read directly. It confirms the two public overloads, floating/BOOL and rank
requirements, axis removal, negative-axis normalization, ordered non-right-aligned placement,
preferred reduced-axis coverage, padded-sequence `[batch,time]` mapping, false exclusion,
true-count mean denominator, and all-false zero result. Legacy mutable candidates, magic scoring,
reshape/expand, where/constants, callbacks, storage, runtime state, lowering, and kernels are
excluded or reassigned.

Planning selected two public methods and one package-private helper. Existing model contracts
suffice; no package, dependency, foundational contract, or architecture change is required.

Pre-implementation planning validation after synchronizing this task, the model master plan, and
roadmap:

- `git diff --check` passed.
- The trailing-whitespace scan returned no matches across the three planning files.
- All 169 relative Markdown links across the three planning files resolve locally.
- Markdown fence counts are balanced: eight in this task, two in the master plan, and zero in the
  roadmap.
- All 21 required task-specification headings are present.
- At that planning checkpoint, task/master/roadmap statuses consistently identified 0016F1 as
  Ready and 0016G as Draft.
- No detailed task-0016G specification exists.
- Repository scope is exactly this task, the model master plan, and roadmap; no Java, API,
  architecture, Gradle, or other file changed during planning.

Implementation validation and the independent clean-context documentation-focused review are now
complete.

Implementation and focused validation:

- `./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorMaskedReductionTest --tests io.github.pho001.synaptik.model.tensor.TensorTest --tests io.github.pho001.synaptik.model.tensor.TensorNumericReductionTest --tests io.github.pho001.synaptik.model.operation.reduction.MaskedReductionAttrsTest`
  passed: 44 tests across the four requested classes, with zero failures, errors, or skips.
- `./gradlew :modules:model:test :modules:model:javadoc test --rerun-tasks` passed. The model suite
  reports 417 tests with zero failures, errors, or skips; root `test` completed successfully; and
  model Javadoc generated without warnings or errors.
- `javap -p -c -s` confirmed exactly the two new public descriptors. Each public method performs
  one call to `TensorMaskedReductionExpressions.apply`, with exact `SUM` or `MEAN`, and the helper
  bytecode confirms one axis normalization, bounded rank-indexed dynamic-programming arrays,
  deterministic reconstruction, one `Shape.ofDimensions` result construction, ordered
  `List.of(input, mask)` provenance, and one `TensorFactory.createDerived` call.
- Reflection tests confirm the helper is package-private final with no fields or nested types, one
  private constructor, one package-private entry, four private methods, and no extra public or
  protected API. `Tensor` has exactly 72 declared public methods and both new methods are public,
  non-static, and non-synchronized.
- Source/import inspection found no `ShapeBroadcast`, retained complete-candidate collection,
  magic penalty, value/storage access, gradient implementation, compiler/runtime/backend type,
  registry, service, dependency, or build change.

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
  effects are stated, ownership/no-mutation boundaries are explicit, and the concrete
  `[batch, time]` to `[batch, time, features]` mapping explains every axis. No additional Java
  documentation edit was required.
- `docs/api/tensor-api.md` now documents public masked sum/mean construction, deterministic mapping,
  metadata/provenance, failures, deferred behavior, and one complete verified example.
  `docs/api/compile-api.md` now lists masked construction among current expression inputs while
  preserving compiler capture and transformation as planned. `docs/glossary.md` now reflects the
  implemented masked-reduction and mask-to-input mapping boundaries.
- Training API requires no change because no training, autograd, gradient publication, optimizer,
  or execution contract changed. Model capabilities require no change because the selected legacy
  capability and its deferred compiler/backend responsibilities were already represented.
- Architecture, focused architecture explanations, ADRs, and architecture tests require no change
  because module ownership and dependency direction are unchanged. DataType, Dimension, Shape,
  ShapeBroadcast, TensorDescriptor, TensorFactory, TensorProvenance, Operation, aggregate semantic
  kinds/attributes, ordinary and arg-max reductions, and their tests remain accurate unchanged
  because this task only composes them through a new package-private helper and two Tensor methods.
- Backend conformance and integration tests require no change because there is no numerical,
  lowering, backend, or end-to-end execution behavior. Gradle requires no change because no
  dependency, source set, language level, preview feature, task, or module was added. No other
  module is affected.

Documentation and final validation:

- `./gradlew :modules:model:javadoc` and the aggregate Gradle command above passed; generated
  `Tensor.html` contains both new method anchors and rendered contracts.
- The documented example was run through the Java 26 source launcher against
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
  derivation, and one derived Tensor construction.
- Added focused coverage for surface, validation order/messages, ID side effects, floating/BOOL
  eligibility, scalar/static/zero/dynamic mapping, every tie-break level, reference preservation,
  metadata, provenance, freshness, nesting, and unchanged inputs. Updated Tensor reflection
  expectations only for the two methods.
- Finalized only the six authorized documentation/planning paths. No Java declaration, behavior,
  or test was changed during this documentation-focused review.

## Completion summary

- Completed changes: public masked sum/mean expression construction with deterministic ordered
  Shape mapping, axis-removing metadata, and exact two-input provenance; focused tests and public
  documentation are complete.
- Files changed or created: exactly the two production, two test, and six documentation/planning
  paths listed under Affected files.
- Tests and validation: all focused, model aggregate, Javadoc, root, reflection, bytecode, example,
  link/anchor, terminology, fence, whitespace, scope, and status checks passed.
- Documentation-agent review: clean one-shot Codex context
  `019f3234-136e-7860-b3af-e99441339b90`, launched by `/root` after the collaboration thread-limit
  fallback, completed the independent pass using the API/Javadoc and Planning profiles plus
  Example format.
- Documentation impact: Tensor API, Compile API, glossary, task, master plan, and roadmap finalized.
- Javadoc review: affected Tensor and helper Javadocs are complete; no further edit required.
- Glossary impact: masked reduction and mask-to-input axis mapping now describe current public
  construction while preserving deferred numerical and cross-layer work.
- Unresolved issues: None.
- Follow-up required: None. Task 0016G remains a separate Draft planning frontier without a
  detailed specification.

Status: Complete
