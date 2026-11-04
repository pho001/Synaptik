# Task 0017G: [Slice](../../../../glossary.md#slice) Semantics

**Editorial reading key (later context, not a status update).**

A slice selects coordinate ranges; its step advances logical coordinates, whereas a layout stride measures storage-element spacing. The original positive-step attributes describe normalized bounds and distinct axes, not input values. Raw negative bounds require an input extent and therefore belong to expression construction. Here Model defines a typed meaning for later construction code; this semantic-family task neither receives stored operands nor evaluates them. Its completion is not evidence of Compiler capture, derivatives or backend execution.

Focused current reading: [Current slicing reference](../../../../reference/tensor/slicing.md#slice-expressions). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Define the typed, [backend](../../../../glossary.md#backend)-independent semantic [identity](../../../../glossary.md#term-identity-meanings) and [immutable](../../../../glossary.md#term-immutable) normalized [parameters](../../../../glossary.md#term-parameter) for a
general positive-step [tensor](../../../../glossary.md#tensor) [slice](../../../../glossary.md#slice).

The general [operation](../../../../glossary.md#operation) may constrain any ordered set of distinct input [axes](../../../../glossary.md#term-axis). A future single-axis
public convenience uses the same operation with one parameter entry; it is not a separate
semantic kind. This task defines meaning and parameter invariants only. Public request syntax,
negative index and axis [normalization](../../../../glossary.md#term-normalization), defaults, [Shape](../../../../glossary.md#shape)/[layout](../../../../glossary.md#layout) derivation, [provenance](../../../../glossary.md#provenance), [gradients](../../../../glossary.md#term-gradient),
[materialization](../../../../glossary.md#materialization), [compiler](../../../../glossary.md#term-compiler) behavior, and execution remain in later tasks and layers.

## [Scope](../../../../glossary.md#term-scope)

- Add one public `SliceKind` enum implementing [`OperationKind`](../../../../glossary.md#operationkind) with exactly [`SLICE`](../../../../glossary.md#slice).
- Add one public `SliceAttrs` record implementing [`OperationAttrs`](../../../../glossary.md#operationattrs) with exactly, in order,
  `List<Long> starts`, `List<Long> ends`, `List<Integer> axes`, and `List<Long> steps`.
- Define index `i` across the four parallel lists as [half-open interval](../../../../glossary.md#term-half-open-interval)
  `[starts[i], ends[i])`, advanced by `steps[i]`, on normalized input [axis](../../../../glossary.md#term-axis) `axes[i]`.
- Require equal list sizes, non-null elements, non-negative normalized starts/ends/axes, unique
  axes, and strictly positive steps.<span data-glossary-navigation="true"> ([elements glossary](../../../../glossary.md#term-element))</span>
- Accept empty lists as a normalized [identity](../../../../glossary.md#term-identity-meanings) slice that constrains no axes.
- Accept every non-negative start/end relationship structurally; later Shape-aware construction
  decides result extent and empty-slice policy.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [extent glossary](../../../../glossary.md#term-extent))</span>
- Preserve entry order and store one [immutable](../../../../glossary.md#term-immutable) [snapshot](../../../../glossary.md#term-snapshot) of each list after [validation](../../../../glossary.md#term-validation).
- Document exact [`Operation`](../../../../glossary.md#operation) composition and future single-axis convenience as one step-one entry.
- Add one focused same-package semantic test for both production contracts.
- Keep production in the existing `model.operation.layout` package.
- Finalize Javadocs, Tensor API, glossary, task evidence, master plan, and roadmap through the
  mandatory independent documentation pass during implementation.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- public `Tensor.slice`, `sliceAxis`, another Tensor method, overload, factory, builder, expression
  helper, or task-0017H implementation<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- raw primitive-array requests, nullable/default axes or steps, implicit axes, implicit step one,
  ellipsis, omitted bounds, open-ended ranges, or a Python slice object<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis))</span>
- normalizing negative axes, starts, or ends against an input Shape; clamping bounds; binding a
  dynamic dimension; or storing raw negative request values<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [binding glossary](../../../../glossary.md#term-binding); [dimension glossary](../../../../glossary.md#dimension))</span>
- calculating output extents, deciding public empty-[slice](../../../../glossary.md#slice) policy, or checking arithmetic [overflow](../../../../glossary.md#term-overflow)
- negative/zero steps, reverse slicing, repeated axes, axis sorting, canonicalization, or merging
  adjacent slices<span data-glossary-navigation="true"> ([canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- input Tensor, Shape, Dimension, DataType, TensorDescriptor, label, identity, provenance, host
  storage, gradient state, or <span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance); [host storage glossary](../../../../glossary.md#host-storage); [gradient glossary](../../../../glossary.md#term-gradient))</span>`TensorFactory.createDerived`
- LayoutDescriptor, stride multiplication, storage offset, view/alias state, copy choice,
  contiguity, materialization, or physical storage access<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [storage offset glossary](../../../../glossary.md#term-storage-offset); [view glossary](../../../../glossary.md#term-view-meanings); [alias glossary](../../../../glossary.md#aliasing-and-arena); [contiguity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [materialization glossary](../../../../glossary.md#materialization))</span>
- slice backward/scatter, autograd, graph capture, compiler, planning, prepare, runtime, backend,
  engine, trace, ONNX, training, or execution behavior<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [engine glossary](../../../../glossary.md#term-engine); [trace glossary](../../../../glossary.md#trace); [ONNX glossary](../../../../glossary.md#term-onnx))</span>
- factories, registries, parsers, visitors, maps, string dispatch, reflective discovery, arity,
  result, cost, fusion, backend-support, route, or kernel metadata<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [arity glossary](../../../../glossary.md#term-arity); [fusion glossary](../../../../glossary.md#term-fusion); [kernel glossary](../../../../glossary.md#kernel); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- changing existing Java/tests, dependencies, [Gradle](../../../../glossary.md#term-gradle), architecture, another [module](../../../../glossary.md#term-module), or later specs

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Backend glossary](../../../../glossary.md#backend))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model master plan](../master-plan.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0005](0005-operation-semantic-foundation.md)
- [Task 0006](0006-operation-model.md)
- [Task 0017E](0017e-axis-transform-semantics.md)
- [Task 0017F1](0017f1-expand-dimensions-and-squeeze-tensor-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The baseline requires general [`slice`](../../../../glossary.md#slice) and single-[axis](../../../../glossary.md#term-axis) convenience. The read-only
`legacy/pre-rewrite` branch represents both with one `SLICE` [operation](../../../../glossary.md#operation):

```java
Tensor slice(int[] starts, int[] ends, int[] axes, int[] steps)
Tensor sliceAxis(int axis, int fromInclusive, int toExclusive)
```

The legacy builder accepts positive steps, normalizes negative axes/bounds against a concrete
input [shape](../../../../glossary.md#shape), clamps bounds, rejects duplicate axes, derives a strided [view](../../../../glossary.md#term-view-meanings), and implements
`sliceAxis` with one axis and step one. Its operation stores parallel normalized arrays plus a
derived output shape.

The new [model](../../../../glossary.md#term-model) preserves those useful capabilities while using [immutable](../../../../glossary.md#term-immutable) lists, `long`
[coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) compatible with current Shape [dimensions](../../../../glossary.md#dimension)/[layout](../../../../glossary.md#layout) [strides](../../../../glossary.md#term-stride), and no derived output Shape
inside semantic attributes. Task 0017H owns input-dependent [normalization](../../../../glossary.md#term-normalization), clamping, output Shape,
layout geometry, [provenance](../../../../glossary.md#provenance), and public methods. Legacy [mutable](../../../../glossary.md#term-mutable) arrays, immediate storage [views](../../../../glossary.md#view),
[graph](../../../../glossary.md#graph) builders, [gradients](../../../../glossary.md#term-gradient), traits, [lowering](../../../../glossary.md#lowering), [kernels](../../../../glossary.md#kernel), and [runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) behavior are not copied.

## Architecture constraints

- Production remains in `modules/model`, which owns [backend](../../../../glossary.md#backend)-neutral [operation](../../../../glossary.md#operation) semantics.
- `SliceKind` identifies logical meaning only, not a Tensor, graph occurrence, descriptor,
  layout, materialization instruction, executable, or backend route.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [graph glossary](../../../../glossary.md#graph); [occurrence glossary](../../../../glossary.md#term-occurrence); [descriptor glossary](../../../../glossary.md#term-descriptor); [layout glossary](../../../../glossary.md#layout); [materialization glossary](../../../../glossary.md#materialization); [backend route glossary](../../../../glossary.md#backend-route))</span>
- `SliceAttrs` stores normalized intrinsic parameters, never input-dependent facts or raw request
  syntax.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter))</span>
- List index `i` couples one start, end, axis, and step. Order is retained even though compiler
  canonicalization may later reorder distinct-axis constraints.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- Bounds are half-open. Starts/ends/axes are normalized non-negative values, axes are unique, and
  steps are strictly positive.<span data-glossary-navigation="true"> ([half-open glossary](../../../../glossary.md#term-half-open-interval))</span>
- The record performs no [dimension](../../../../glossary.md#dimension) [lookup](../../../../glossary.md#term-lookup), clamping, [extent](../../../../glossary.md#term-extent) calculation, or start/end comparison.
- Empty parallel lists are a valid identity description. Task 0017H decides whether a public
  no-entry request is exposed.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Single-[axis](../../../../glossary.md#term-axis) convenience has no separate kind or attributes type.
- Generic `Operation` remains an open kind/attributes pair and does not validate family pairing,
  arity, rank, Shape, layout, gradients, or backend support.<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [rank glossary](../../../../glossary.md#term-rank); [Shape glossary](../../../../glossary.md#shape); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Package direction is `model.operation.layout -> model.operation + java.base` only.
- Stop if implementation needs input Shape/Tensor state, another type, existing Java changes, a
  dependency, or an architecture decision.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

No package is added or moved.

- `io.github.pho001.synaptik.model.operation.layout.SliceKind` — public semantic [identity](../../../../glossary.md#term-identity-meanings).
- `io.github.pho001.synaptik.model.operation.layout.SliceAttrs` — public immutable normalized
  parallel parameters.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [parameters glossary](../../../../glossary.md#term-parameter))</span>
- `SliceSemanticsTest` — same-package structural, [validation](../../../../glossary.md#term-validation), [ownership](../../../../glossary.md#term-ownership-meanings), and composition test.

The existing [layout](../../../../glossary.md#layout)-[operation](../../../../glossary.md#operation) package owns cohesive layout/[view](../../../../glossary.md#term-view-meanings) semantic requests. No root
fallback, generic utility package, or migration is introduced.

## Required contract

### Semantic kind vocabulary

Create exactly:

```java
public enum SliceKind implements OperationKind {
    SLICE
}
```

The enum declares no project field, explicit constructor, method, nested type, per-[constant](../../../../glossary.md#term-constant) body,
[alias](../../../../glossary.md#aliasing-and-arena), [arity](../../../../glossary.md#term-arity), bounds, [axes](../../../../glossary.md#term-axis), steps, result, [layout](../../../../glossary.md#layout), or [backend](../../../../glossary.md#backend) [metadata](../../../../glossary.md#term-metadata). Inherited `Enum.name()`
satisfies `OperationKind.name()`.

Document one logical input and a same-[rank](../../../../glossary.md#term-rank) result. Entry `i` selects input [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) starting at
`starts[i]`, advancing by `steps[i]`, and remaining below `ends[i]` along `axes[i]`. Unlisted axes
retain their full logical coordinate range. This type calculates no [Shape](../../../../glossary.md#shape) and creates no [view](../../../../glossary.md#term-view-meanings).

Do not add `SLICE_AXIS`; future single-axis convenience is one [`SLICE`](../../../../glossary.md#slice) entry with step one.

### Normalized [slice](../../../../glossary.md#slice) attributes

Create exactly:

```java
public record SliceAttrs(
        List<Long> starts,
        List<Long> ends,
        List<Integer> axes,
        List<Long> steps) implements OperationAttrs
```

The record has exactly four components in that order, one [canonical](../../../../glossary.md#term-canonical-form) constructor, four explicit
documented accessors, and generated object methods. Add no array overload, factory, builder,
range/entry type, [rank](../../../../glossary.md#term-rank), size method, output [Shape](../../../../glossary.md#shape), [layout](../../../../glossary.md#layout), [cache](../../../../glossary.md#term-cache), nested type, or extra [API](../../../../glossary.md#term-application-programming-interface-api)/state.

Constructor [validation](../../../../glossary.md#term-validation) order is exact:

1. null-check `starts`, `ends`, [`axes`](../../../../glossary.md#term-axis), and `steps`, in order, with component-name messages;
2. reject unequal sizes with `IllegalArgumentException` and exact message
   `starts, ends, axes, and steps must have matching sizes`;
3. create one constructor-local `HashSet<Integer>` for seen axes;
4. inspect entries in ascending index order;
5. null-check start, end, axis, and step, in order, with exact messages `starts[<index>]`,
   `ends[<index>]`, `axes[<index>]`, and `steps[<index>]`;
6. reject negative start with `starts[<index>] must be non-negative: <value>`;
7. reject negative end with `ends[<index>] must be non-negative: <value>`;
8. reject negative axis with `axes[<index>] must be non-negative: <value>`;
9. reject the first repeated axis with
   `axes contains duplicate axis <value> at index <index>`;
10. reject non-positive step with `steps[<index>] must be positive: <value>`;
11. after all validation, assign `List.copyOf` snapshots to starts, ends, axes, and steps in that
    order, exactly once each.

The record retains values/order but not caller list [identity](../../../../glossary.md#term-identity-meanings). Caller mutation cannot affect it and
accessor mutation fails. It stores no `HashSet` or derived state.

`Long.MAX_VALUE` is structurally valid for bounds/steps and `Integer.MAX_VALUE` for an axis because
no input rank exists here. A start may equal or exceed its end. This does not promise public API
acceptance; task 0017H owns Shape-aware bounds, [extent](../../../../glossary.md#term-extent), empty-result, and [overflow](../../../../glossary.md#term-overflow) policy.

Record equality/hashing use all four ordered lists. Generated text is diagnostic, not
[serialization](../../../../glossary.md#term-serialization), request syntax, [compiler](../../../../glossary.md#term-compiler) [canonical form](../../../../glossary.md#term-canonical-form), [ONNX](../../../../glossary.md#term-onnx) mapping, or [backend](../../../../glossary.md#backend) [dispatch](../../../../glossary.md#term-dispatch).

### Typed composition and single-[axis](../../../../glossary.md#term-axis) convenience

Document exact composition:

```java
SliceAttrs attrs = new SliceAttrs(starts, ends, axes, steps);
Operation operation = new Operation(SliceKind.SLICE, attrs);
```

[Operation](../../../../glossary.md#operation) retains exact references. Do not use [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), add a factory/validator, or
change `Operation`.

Document future single-axis convenience as semantically equivalent to:

```java
new SliceAttrs(
        List.of(fromInclusive),
        List.of(toExclusive),
        List.of(normalizedAxis),
        List.of(1L))
```

This assumes input-dependent [normalization](../../../../glossary.md#term-normalization) already occurred. Task 0017H owns actual [signatures](../../../../glossary.md#term-signature),
[validation](../../../../glossary.md#term-validation), raw negative syntax, and construction.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/SliceKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/SliceAttrs.java`

Test:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/layout/SliceSemanticsTest.java`

Documentation/[planning](../../../../glossary.md#term-planning):

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless inconsistency requires stopping: [Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api), Training API,
capabilities, [Operation](../../../../glossary.md#operation) foundations, [Shape](../../../../glossary.md#shape)/[Dimension](../../../../glossary.md#dimension), [layout](../../../../glossary.md#layout) contracts, [Tensor](../../../../glossary.md#tensor)/[graph](../../../../glossary.md#graph) contracts,
architecture/ADRs/tests, [conformance](../../../../glossary.md#term-conformance)/integration tests, [Gradle](../../../../glossary.md#term-gradle), dependencies, and other [modules](../../../../glossary.md#term-module).

## Maximum [scope](../../../../glossary.md#term-scope)

At most two production files, one focused test, and five documentation/[planning](../../../../glossary.md#term-planning) files: eight paths.
If a ninth path, another Java type/test, existing Java edit, [dependency](../../../../glossary.md#term-dependency), build, or architecture
change is required, stop and report. Do not create task 0017H.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Document enum/type/[constant](../../../../glossary.md#term-constant), record/constructor/components, and all explicit accessors.
- Explain parallel [half-open](../../../../glossary.md#term-half-open-interval) semantics with [Shape](../../../../glossary.md#shape) `[3, 6]`, starts `[0, 1]`, ends `[3, 6]`, [axes](../../../../glossary.md#term-axis)
  `[0, 1]`, steps `[1, 2]`: selected columns are `1, 3, 5`, without claiming execution exists.
- Explain why [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) use `long` and axes use `int`.
- Explain snapshots, order-sensitive value semantics, empty [identity](../../../../glossary.md#term-identity-meanings), and list [ownership](../../../../glossary.md#term-ownership-meanings).
- Explain normalized attributes versus raw negative request syntax and deferred [rank](../../../../glossary.md#term-rank)/bound checks.
- Explain why single-axis convenience is not another kind.
- Document [validation](../../../../glossary.md#term-validation) order, [parameters](../../../../glossary.md#term-parameter), results, null/value failures, and deferred behavior.
- Explain that no Tensor, Shape calculation, layout/view, storage, materialization, gradient,
  compiler, backend, ONNX, or execution behavior is defined.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [layout glossary](../../../../glossary.md#layout); [view glossary](../../../../glossary.md#term-view-meanings); [materialization glossary](../../../../glossary.md#materialization); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [ONNX glossary](../../../../glossary.md#term-onnx))</span>
- Review related contract Javadocs and record why they remain accurate or stop on discrepancy.

## Acceptance criteria

- Exact one-[constant](../../../../glossary.md#term-constant) public enum and exact four-component public record with no extra [API](../../../../glossary.md#term-application-programming-interface-api)/state.
- Empty, one-[axis](../../../../glossary.md#term-axis), and multi-axis values are accepted and preserve values/order.
- Every specified invalid container, element, size, coordinate, axis, duplicate, and step fails
  with exact type/message/precedence.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Four snapshots occur only after [validation](../../../../glossary.md#term-validation); caller/accessor mutation cannot alter state.
- Extreme values and start/end relationships remain [Shape](../../../../glossary.md#shape)-independent.
- Exact [Operation](../../../../glossary.md#operation) composition retains references; no generic validator is added.
- Single-axis is documented as one step-one entry; no `SLICE_AXIS` or [Tensor](../../../../glossary.md#tensor) method is added.
- No Shape/[layout](../../../../glossary.md#layout)/[provenance](../../../../glossary.md#provenance)/storage/[gradient](../../../../glossary.md#term-gradient)/cross-layer behavior or existing Java edit appears.
- Javadocs, Tensor API, glossary, independent docs review, evidence, and statuses are complete.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.layout.SliceSemanticsTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test verifies exact enum/record/[API](../../../../glossary.md#term-application-programming-interface-api) [shape](../../../../glossary.md#shape); component order/types; empty, one-[axis](../../../../glossary.md#term-axis), and
general examples; validation failures/precedence; list snapshots/immutability; value semantics;
extreme values; unconstrained start/end relations; duplicate axes; exact [Operation](../../../../glossary.md#operation) references;
step-one convenience; and no forbidden members/imports.

Inspect `javap -p -c -s`, reflection, imports, and source. Confirm two production types, one local
`HashSet`, four post-validation `List.copyOf` calls, no retained helper state, and no forbidden
layer types. Validate generated [Javadoc](../../../../glossary.md#term-javadoc), API/glossary terminology, links/anchors/fences/whitespace,
exact eight paths, synchronized status, and no task-0017H specification.

## Dependencies

- 0005 supplies [`OperationKind`](../../../../glossary.md#operationkind) and [`OperationAttrs`](../../../../glossary.md#operationattrs).
- 0006 supplies generic [immutable](../../../../glossary.md#term-immutable) [`Operation`](../../../../glossary.md#operation) composition.
- 0002 supplies long-[dimension](../../../../glossary.md#dimension)/int-[axis](../../../../glossary.md#term-axis) terminology but is not a production [dependency](../../../../glossary.md#term-dependency).
- 0017E–0017F1 establish the package and semantic/expression split without production coupling.

## Follow-up tasks

- 0017H remains Draft for public requests, array ownership/defaults if selected, negative
  normalization, clamping, output Shape/LayoutDescriptor, provenance, and descriptor construction.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [normalization glossary](../../../../glossary.md#term-normalization); [Shape glossary](../../../../glossary.md#shape); [provenance glossary](../../../../glossary.md#provenance); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- [Compiler](../../../../glossary.md#term-compiler) later owns [identity](../../../../glossary.md#term-identity-meanings)/[slice](../../../../glossary.md#slice)-chain [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form) and [graph](../../../../glossary.md#graph) constraints.
- Compiler-generated/training semantics later own slice-backward scatter construction.
- [Planning](../../../../glossary.md#term-planning)/[backend](../../../../glossary.md#backend) [prepare](../../../../glossary.md#prepare) later own [materialization](../../../../glossary.md#materialization) and concrete [lowering](../../../../glossary.md#lowering).

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. These [model](../../../../glossary.md#term-model)-owned semantic values stay in the existing package and
[dependency direction](../../../../glossary.md#java-platform-and-build-tools). Stop if implementation needs input state, dependencies, generic [Operation](../../../../glossary.md#operation)
changes, or architecture changes.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0002/0005/0006/0017E/0017F1/0017G, Tensor API, Compile API,
Training API, glossary, current OperationKind/OperationAttrs/Operation and layout-operation
semantic contracts/tests, Shape/LayoutDescriptor terminology, and Java 26 Gradle configuration.

Implement task 0017G exactly. Add only SliceKind.java, SliceAttrs.java, and
SliceSemanticsTest.java under io.github.pho001.synaptik.model.operation.layout.

SliceKind contains exactly SLICE. SliceAttrs contains exactly List<Long> starts, List<Long> ends,
List<Integer> axes, and List<Long> steps in order. Validate exact matching sizes, indexed non-null
elements, non-negative normalized starts/ends/unique axes, and positive steps with specified
order/messages, then store four immutable snapshots. Empty lists and unconstrained start/end
relations are valid. Document parallel half-open semantics, exact pairing, and single-axis
convenience as one step-one entry, not another kind.

Do not add Tensor methods, raw arrays/defaults/normalization, Shape/result/layout/provenance logic,
negative/reverse steps, gradients, graph/compiler/planning/runtime/backend/ONNX behavior,
factories, dependencies, build/architecture changes, existing Java edits, or later specs. Stop
beyond eight paths or on architecture uncertainty.

Run all specified validation, then hand actual diff/evidence to a separate clean-context docs
agent in the same change. It must inspect source/tests/Javadoc, finalize permitted Javadocs/Tensor
API/glossary/planning, record no-change conclusions, and rerun validation.

Update task 0017G, model master plan, and roadmap only for status/evidence. Do not mark Complete
until both passes succeed. Leave 0017H Draft without a specification. Do not commit or push.
```

## Local decisions

- One [`SLICE`](../../../../glossary.md#slice) covers general and single-[axis](../../../../glossary.md#term-axis) forms because convenience changes no mathematics.
- Four [immutable](../../../../glossary.md#term-immutable) parallel lists preserve general capability without [mutable](../../../../glossary.md#term-mutable) primitive arrays.
- Bounds/steps use `long` to match [Shape](../../../../glossary.md#shape)/[layout](../../../../glossary.md#layout) geometry; axes remain `int`.
- Only normalized non-negative values enter attributes; raw negative syntax belongs to 0017H.
- Axes are unique but entry order is retained rather than sorted.
- Bounds are not compared here, keeping [extent](../../../../glossary.md#term-extent) and zero-extent policy Shape-aware in 0017H.
- Empty lists are a structural [identity](../../../../glossary.md#term-identity-meanings); 0017H decides whether public [API](../../../../glossary.md#term-application-programming-interface-api) exposes them.

## Known limitations

- No raw normalization, rank validation, clamping, extent calculation, arithmetic proof, or view
  geometry exists.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [rank glossary](../../../../glossary.md#term-rank); [validation glossary](../../../../glossary.md#term-validation); [extent glossary](../../../../glossary.md#term-extent); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- No Tensor method, gradient scatter, compiler behavior, materialization, lowering, execution, or
  ONNX mapping exists.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [materialization glossary](../../../../glossary.md#materialization); [lowering glossary](../../../../glossary.md#lowering); [ONNX glossary](../../../../glossary.md#term-onnx))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Clean implementation context `/root/implement_model_0017g` added only the two production types
  and focused test specified by this task before handing the actual uncommitted diff to the
  independent documentation context.
- Clean documentation context `/root/implement_model_0017g/review_model_0017g_docs` applied the
  General, API/Javadoc, Planning, and Example profiles. It read the architecture and focused
  boundary documents, documentation/planning rules, prerequisite tasks, model plan and capability
  baseline, API/glossary sources, final production/test source, generated Javadoc, and actual diff.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Planning glossary](../../../../glossary.md#term-planning); [model glossary](../../../../glossary.md#term-model))</span>
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.layout.SliceSemanticsTest` — `BUILD SUCCESSFUL`; the
  focused report contains 14 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; XML report aggregation contains 543 tests
  with zero failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`; generated pages contain the enum,
  constant, record components, canonical constructor, explicit accessors, parameters, results,
  failures, ownership, example, and cross-layer exclusions.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [canonical glossary](../../../../glossary.md#term-canonical-form); [parameters glossary](../../../../glossary.md#term-parameter); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- `./gradlew test` — `BUILD SUCCESSFUL`; the root [lifecycle](../../../../glossary.md#lifecycle) reported 36 actionable tasks.
- `javap -p -c -s` plus reflection tests confirm one exact enum constant; one exact four-component
  record in starts/ends/axes/steps order; four private final list fields; one constructor-local<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis))</span>
  `HashSet`; deterministic validation; exactly four post-validation `List.copyOf` calls; four
  direct accessors; generated record value methods; and no retained helper state or extra API.
- Source/import inspection confirms package direction only to `model.operation` and `java.base`.
  No Tensor, Shape, layout value, storage, graph, compiler, planning, prepare, runtime, backend,
  training, or ONNX type enters production state.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [ONNX glossary](../../../../glossary.md#term-onnx))</span>
- The documentation pass retained the complete implementation-drafted Javadocs unchanged and
  finalized Tensor API and glossary coverage for parallel half-open semantics, the conceptual
  Shape <span data-glossary-navigation="true"> ([half-open glossary](../../../../glossary.md#term-half-open-interval))</span>`[3, 6]` example selecting columns `1`, `3`, `5`, long coordinates/int axes, immutable
  snapshots, order-sensitive value semantics, empty identity, raw-versus-normalized boundaries,
  exact <span data-glossary-navigation="true"> ([coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [immutable glossary](../../../../glossary.md#term-immutable); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>[`Operation`](../../../../glossary.md#operation) composition, one-entry step-one convenience, and all deferred behavior.
- A targeted local Markdown file-and-GitHub-heading checker resolved all 319 links and anchors in
  the five changed documentation/planning files with zero errors. Fence counts are balanced
  (`150`, `0`, `14`, `2`, `0`), trailing-whitespace scans found no matches, and every file ends
  with a newline.
- Final path inventory contains exactly the permitted eight paths: two production sources, one
  focused test, Tensor API, glossary, this task, model master plan, and roadmap. Task/master/roadmap
  identify 0017G as Complete; 0017H remains Draft and no task-0017H specification exists.
- Compile API remains accurate unchanged because no public Tensor slice expression or compiler
  capture exists. Training API remains accurate unchanged because this task defines no gradient,
  autograd, optimizer, or training behavior. The capability baseline already lists general and
  single-axis slice capability and distinguishes model semantics from later public/executable
  layers, so no update is needed.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [slice glossary](../../../../glossary.md#slice); [capture glossary](../../../../glossary.md#term-capture-meanings); [gradient glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- Existing [`OperationKind`](../../../../glossary.md#operationkind), [`OperationAttrs`](../../../../glossary.md#operationattrs), `Operation`, Shape/Dimension, LayoutDescriptor,
  Tensor/layout-operation/graph Javadocs and contracts remain accurate because the new family
  composes them without changing their signatures, invariants, or ownership. No architecture
  document, ADR, architecture test, backend-conformance test, integration test, Gradle file,
  dependency, existing unrelated Java file, other module, or later task specification changed
  because no boundary, dependency, backend, executable, or end-to-end behavior changed.<span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension); [signatures glossary](../../../../glossary.md#term-signature); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [architecture test glossary](../../../../glossary.md#term-architecture-test); [conformance glossary](../../../../glossary.md#term-conformance); [integration test glossary](../../../../glossary.md#term-integration-test); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module))</span>
- `git diff --check` passed with no whitespace errors after final documentation and planning
  synchronization.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>

## Implementation notes

- Added `SliceKind` with exactly [`SLICE`](../../../../glossary.md#slice) and `SliceAttrs` with exactly the ordered starts, ends,
  axes, and steps list components in the existing layout-operation package.<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis); [layout glossary](../../../../glossary.md#layout); [operation glossary](../../../../glossary.md#operation))</span>
- `SliceAttrs` validates containers, pairing, indexed elements, normalized values, distinct axes,
  and positive steps in the specified precedence, then snapshots all four lists. Empty lists,
  extreme non-negative values, and every non-negative start/end relationship remain valid.<span data-glossary-navigation="true"> ([elements glossary](../../../../glossary.md#term-element))</span>
- Added one focused same-package test covering exact surface shape, validation messages and
  precedence, immutability, value semantics, identity and general examples, exact Operation
  composition, step-one convenience, and forbidden-state absence.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [validation glossary](../../../../glossary.md#term-validation); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Tensor API and glossary now describe the implemented semantic contract and conceptual parallel
  example while explicitly retaining public Tensor construction and every Shape/layout/
  provenance/cross-layer behavior for task 0017H or later owning layers.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [provenance glossary](../../../../glossary.md#provenance))</span>
- Synchronized this task, model master plan, and roadmap at Complete without creating the 0017H
  specification.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>

## Completion summary

- Completed changes: Added the exact backend-neutral positive-step slice kind and immutable
  normalized parallel attributes, focused semantic tests, complete API/glossary documentation,
  and synchronized planning evidence.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [immutable glossary](../../../../glossary.md#term-immutable); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: Exactly the eight paths listed under Affected files.
- Tests and validation: Focused 14-test suite, 543-test model suite, model Javadoc, root test
  lifecycle, bytecode/reflection/import/source checks, generated-Javadoc review, 319 local links
  and anchors, fences/whitespace/newlines, exact scope/status checks, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [lifecycle glossary](../../../../glossary.md#lifecycle); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>`git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0017g/review_model_0017g_docs` independently reviewed final source,
  tests, generated Javadoc, related contracts, and the actual diff under the required profiles.
- Documentation impact: [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api) and glossary now define current [slice](../../../../glossary.md#slice) semantics and the
  `[3, 6]` parallel [half-open](../../../../glossary.md#term-half-open-interval) example while keeping public construction in Draft task 0017H.
- Javadoc review: SliceKind/SliceAttrs type, member, constructor, component, accessor, ownership,
  validation, example, and exclusion contracts are complete and accurate; related Javadocs remain
  accurate unchanged.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Glossary impact: Added the reusable Slice term and updated normalized-axis, OperationAttrs,
  OperationKind, status, and kind/attributes/operation distinctions.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [OperationAttrs glossary](../../../../glossary.md#operationattrs); [OperationKind glossary](../../../../glossary.md#operationkind); [operation glossary](../../../../glossary.md#operation))</span>
- Architecture impact: None.
- Unresolved issues: None within task [scope](../../../../glossary.md#term-scope); known limitations are intentional layer boundaries.
- Follow-up required: None for task 0017G. Task 0017H remains Draft without a specification.

Status: Complete
