# Task 0018A: [Scalar Select](../../../../glossary.md#scalar-select) Semantics

**Editorial reading key (later context, not a status update).**

Scalar select fixes one coordinate on an axis and removes that axis: selecting axis one, index two from [2, 3, 4] describes [2, 4]. A normalized scalar index is intrinsic attributes state, unlike a gather indices Tensor or a Boolean WHERE condition. Input-dependent bounds cannot be proved by this record alone. Here Model defines a typed meaning for later construction code; this semantic-family task neither receives stored operands nor evaluates them. Its completion is not evidence of Compiler capture, derivatives or backend execution.

Focused current reading: [Current scalar-selection reference](../../../../reference/tensor/gather.md#scalar-select-expressions). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Define the typed, [backend](../../../../glossary.md#backend)-independent semantic [identity](../../../../glossary.md#term-identity-meanings) and [immutable](../../../../glossary.md#term-immutable) normalized [parameters](../../../../glossary.md#term-parameter) for
selecting one [scalar](../../../../glossary.md#term-scalar) [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) from one [tensor](../../../../glossary.md#tensor) [axis](../../../../glossary.md#term-axis).

[Scalar select](../../../../glossary.md#scalar-select) fixes one coordinate on one existing source axis and removes that axis from the
logical result. For example, selecting axis `1`, index `2` from [shape](../../../../glossary.md#shape) `[2, 3, 4]` later produces
shape `[2, 4]`. This [operation](../../../../glossary.md#operation) is distinct from [elementwise](../../../../glossary.md#term-elementwise) conditional `WHERE` and from [gather](../../../../glossary.md#gather)
operations whose indices are themselves tensors.

This task defines meaning and intrinsic normalized parameters only. Public request syntax, input
Shape [validation](../../../../glossary.md#term-validation), negative axis/index [normalization](../../../../glossary.md#term-normalization), result Shape and [layout](../../../../glossary.md#layout) derivation, Tensor
[provenance](../../../../glossary.md#provenance), [gradients](../../../../glossary.md#term-gradient), [compiler](../../../../glossary.md#term-compiler) behavior, [lowering](../../../../glossary.md#lowering), and execution remain later responsibilities.

## [Scope](../../../../glossary.md#term-scope)

- Add one public `SelectKind` enum implementing [`OperationKind`](../../../../glossary.md#operationkind) with exactly `SELECT`.
- Add one public `SelectAttrs` record implementing [`OperationAttrs`](../../../../glossary.md#operationattrs) with exactly `int axis` and
  `long index`, in that order.
- Require both stored values to be normalized and non-negative.
- Document that one logical input [axis](../../../../glossary.md#term-axis) is fixed to one [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) and removed from the result.
- Document the exact `SELECT` plus `SelectAttrs` composition without adding a compatibility
  validator.
- Add one focused same-package structural, [validation](../../../../glossary.md#term-validation), value-semantics, and composition test.
- Introduce the cohesive package `io.github.pho001.synaptik.model.operation.index`.
- Finalize Javadocs, Tensor API terminology, glossary, task evidence, master plan, and roadmap
  through the mandatory independent documentation pass during implementation.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- public `Tensor.select`, another Tensor method, overload, factory, expression helper, or task
  0018B implementation<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- input Tensor, input Shape, rank lookup, negative request normalization, bounds checking, static
  versus dynamic selected-dimension policy, result Shape, descriptor, or provenance<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [rank glossary](../../../../glossary.md#term-rank); [lookup glossary](../../../../glossary.md#term-lookup); [normalization glossary](../../../../glossary.md#term-normalization); [dimension glossary](../../../../glossary.md#dimension); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- result layout, offset/stride calculation, view or alias state, storage access, allocation,
  copying, materialization, or value selection<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [stride glossary](../../../../glossary.md#term-stride); [view glossary](../../../../glossary.md#term-view-meanings); [alias glossary](../../../../glossary.md#aliasing-and-arena); [materialization glossary](../../../../glossary.md#materialization))</span>
- tensor-valued indices, gather, gather-axis, take, take-along-axis, gather-ND, scatter, or their
  parameter contracts<span data-glossary-navigation="true"> ([gather glossary](../../../../glossary.md#gather); [axis glossary](../../../../glossary.md#term-axis); [gather-ND glossary](../../../../glossary.md#gather-nd); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- using `WHERE`, `UNSTACK`, or [`SLICE`](../../../../glossary.md#slice) as a replacement for the first-class `SELECT` meaning
- data-type validation, output type, gradient rules, graph capture, canonicalization,
  decomposition, compiler, planning, prepare, runtime, backend, engine, trace, ONNX, training, or
  execution behavior<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [gradient glossary](../../../../glossary.md#term-gradient); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [decomposition glossary](../../../../glossary.md#term-decomposition); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [engine glossary](../../../../glossary.md#term-engine); [trace glossary](../../../../glossary.md#trace); [ONNX glossary](../../../../glossary.md#term-onnx))</span>
- factories, registries, parsers, maps, string dispatch, reflection discovery, arity, cost,
  fusion, backend-support, route, or kernel metadata<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [arity glossary](../../../../glossary.md#term-arity); [fusion glossary](../../../../glossary.md#term-fusion); [kernel glossary](../../../../glossary.md#kernel); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- changing existing Java/tests, dependencies, Gradle, architecture, another module, or creating a
  task-0018B specification<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

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
- [Task 0015E](0015e-where-selection-semantic-kind.md)
- [Task 0017G](0017g-slice-semantics.md)
- [Task 0017K](0017k-tensor-composition-semantics.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The selected capability baseline requires [scalar](../../../../glossary.md#term-scalar)-index `select`. The read-only legacy code
represents it as an [axis](../../../../glossary.md#term-axis) plus one scalar [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), removes the selected axis from the logical
[shape](../../../../glossary.md#shape), and uses it as one building block for unstack.

The new contract retains that capability but stores normalized coordinates using current [model](../../../../glossary.md#term-model)
conventions: an `int` axis and a `long` index compatible with long-valued Shape [dimensions](../../../../glossary.md#dimension). The
semantic record does not store an input or output Shape because bounds and [rank](../../../../glossary.md#term-rank) are properties of
one [operation](../../../../glossary.md#operation) [occurrence](../../../../glossary.md#term-occurrence), not intrinsic operation [parameters](../../../../glossary.md#term-parameter).

Legacy [mutable](../../../../glossary.md#term-mutable) [graph](../../../../glossary.md#graph) [nodes](../../../../glossary.md#node), immediate [view](../../../../glossary.md#term-view-meanings) construction, storage offsets, [gradients](../../../../glossary.md#term-gradient), traits,
[lowering](../../../../glossary.md#lowering), [kernels](../../../../glossary.md#kernel), and [runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) coupling are capability evidence only and are not copied.
Task 0018B will own the public [Tensor](../../../../glossary.md#tensor) request, input-dependent [normalization](../../../../glossary.md#term-normalization), Shape/[layout](../../../../glossary.md#layout) rules,
[descriptor](../../../../glossary.md#term-descriptor) construction, and [provenance](../../../../glossary.md#provenance).

## Architecture constraints

- Production remains in `modules/model`, which owns [backend](../../../../glossary.md#backend)-independent [operation](../../../../glossary.md#operation) meaning.
- `SelectKind` identifies mathematical indexing meaning only, not an occurrence, Tensor, graph
  node, descriptor, executable, view, materialization plan, kernel, or backend route.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [Tensor glossary](../../../../glossary.md#tensor); [graph glossary](../../../../glossary.md#graph); [node glossary](../../../../glossary.md#node); [descriptor glossary](../../../../glossary.md#term-descriptor); [view glossary](../../../../glossary.md#term-view-meanings); [materialization glossary](../../../../glossary.md#materialization); [kernel glossary](../../../../glossary.md#kernel); [backend route glossary](../../../../glossary.md#backend-route))</span>
- `SelectAttrs.axis` is already normalized and non-negative; it stores no rank and cannot prove
  that the axis exists.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [axis glossary](../../../../glossary.md#term-axis))</span>
- `SelectAttrs.index` is already normalized and non-negative; it stores no selected extent and
  cannot prove that the coordinate exists.<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Scalar select has one logical input and removes exactly one selected axis, but this task neither
  validates nor constructs the result Shape.<span data-glossary-navigation="true"> ([Scalar select glossary](../../../../glossary.md#scalar-select); [Shape glossary](../../../../glossary.md#shape))</span>
- `SELECT` pairs with `SelectAttrs`. Generic `Operation` remains an open kind/attributes pair and
  does not enforce family pairing, arity, rank, bounds, result Shape, gradients, or backend support.<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- `SELECT` is distinct from conditional `WHERE`, individually indexed `UNSTACK`, general [`SLICE`](../../../../glossary.md#slice),
  and every tensor-index gather form.<span data-glossary-navigation="true"> ([gather glossary](../../../../glossary.md#gather))</span>
- Package direction is `model.operation.index -> model.operation + java.base` only.
- Stop if implementation requires Tensor, Shape, layout, provenance, another production type,
  another test, dependency, architecture change, or cross-layer behavior.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Add one cohesive [operation](../../../../glossary.md#operation)-family package:

```text
io.github.pho001.synaptik.model.operation.index
  SelectKind
  SelectAttrs
```

The focused test uses the same package. This package will later own typed [gather](../../../../glossary.md#gather) and functional
scatter semantic contracts; it must not depend on public [Tensor](../../../../glossary.md#tensor) or [compiled graph](../../../../glossary.md#compiled-graph--compiledgraphmodel) packages.

## Required contract

### Semantic kind

Create exactly:

```java
public enum SelectKind implements OperationKind {
    SELECT
}
```

The enum declares no project field, explicit constructor, method, nested type, per-[constant](../../../../glossary.md#term-constant) body,
[alias](../../../../glossary.md#aliasing-and-arena), [arity](../../../../glossary.md#term-arity), [axis](../../../../glossary.md#term-axis), index, [Shape](../../../../glossary.md#shape), result, [layout](../../../../glossary.md#layout), cost, or [backend](../../../../glossary.md#backend) [metadata](../../../../glossary.md#term-metadata). Inherited
`Enum.name()` satisfies `OperationKind.name()`.

[Javadoc](../../../../glossary.md#term-javadoc) must explain that `SELECT` fixes one [scalar](../../../../glossary.md#term-scalar) [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) on one source axis and removes that
axis from the logical result. It must distinguish scalar index selection from conditional `WHERE`
and [tensor](../../../../glossary.md#tensor)-index [gather](../../../../glossary.md#gather) without claiming Shape [validation](../../../../glossary.md#term-validation) or execution.

### Normalized attributes

Create exactly:

```java
public record SelectAttrs(int axis, long index) implements OperationAttrs
```

The record has exactly two components in that order, one [canonical](../../../../glossary.md#term-canonical-form) constructor, two explicit
documented accessors, and record-generated `equals`, [`hashCode`](../../../../glossary.md#hash-codes-and-content-digests), and `toString`. Add no factory,
overload, builder, input/output [Shape](../../../../glossary.md#shape), [rank](../../../../glossary.md#term-rank), [dimension](../../../../glossary.md#dimension) [extent](../../../../glossary.md#term-extent), normalized flag, result [descriptor](../../../../glossary.md#term-descriptor),
[layout](../../../../glossary.md#layout), nested type, helper, or extra state/[API](../../../../glossary.md#term-application-programming-interface-api).

Constructor [validation](../../../../glossary.md#term-validation) order is exact:

1. reject a negative [`axis`](../../../../glossary.md#term-axis) with `IllegalArgumentException` and exact message
   `axis must be non-negative: <axis>`;
2. reject a negative `index` with `IllegalArgumentException` and exact message
   `index must be non-negative: <index>`;
3. assign both primitive values unchanged.

Zero, `Integer.MAX_VALUE`, and `Long.MAX_VALUE` are structurally valid. Rank and bounds checks are
deferred because this value has no input Shape.

The explicit accessors must document normalized non-negative meaning, inputs, output values, and
the absence of input-dependent validation.

### Typed composition

The valid composition is explicit:

```java
SelectAttrs attrs = new SelectAttrs(1, 2L);
Operation operation = new Operation(SelectKind.SELECT, attrs);
```

For a later input with [shape](../../../../glossary.md#shape) `[2, 3, 4]`, this means fixing [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `2` on [axis](../../../../glossary.md#term-axis) `1`; the later
[Tensor](../../../../glossary.md#tensor)-expression task derives result shape `[2, 4]`. This task adds no compatibility validator
to [`Operation`](../../../../glossary.md#operation), no factory, and no result [inference](../../../../glossary.md#term-inference).

## Affected files

Expected implementation change:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/SelectKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/SelectAttrs.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/index/SelectSemanticsTest.java`
- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task file
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most the eight paths listed above.

If implementation requires another production type, another test, a [Tensor](../../../../glossary.md#tensor) or [Shape](../../../../glossary.md#shape) change,
[dependency](../../../../glossary.md#term-dependency), build change, focused architecture change, or more than eight paths, stop and propose
a follow-up task. Do not trade away [validation](../../../../glossary.md#term-validation) or documentation to fit the limit.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Document every public type, enum [constant](../../../../glossary.md#term-constant), record component/accessor, and [canonical](../../../../glossary.md#term-canonical-form) constructor.
- Explain [scalar select](../../../../glossary.md#scalar-select) using both words and the concrete `[2, 3, 4]`, [axis](../../../../glossary.md#term-axis) `1`, index `2` example.
- Define `axis` and `index` as normalized and zero-based.
- Explain why [rank](../../../../glossary.md#term-rank)/bounds/result [Shape](../../../../glossary.md#shape) are not validated by the attributes.
- Distinguish `SELECT` from `WHERE`, `UNSTACK`, [`SLICE`](../../../../glossary.md#slice), and [tensor](../../../../glossary.md#tensor)-index [gather](../../../../glossary.md#gather).
- Do not promise view construction, aliasing, gradients, compiler capture, backend support, or
  execution.<span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings); [aliasing glossary](../../../../glossary.md#term-aliasing); [gradients glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [backend glossary](../../../../glossary.md#backend))</span>

## Acceptance criteria

- `SelectKind` is a public enum implementing [`OperationKind`](../../../../glossary.md#operationkind) with exactly `SELECT`.
- `SelectKind` adds no project-declared fields, methods, constructors, or nested types.
- `SelectAttrs` is a public record implementing [`OperationAttrs`](../../../../glossary.md#operationattrs) with exactly [`axis`](../../../../glossary.md#term-axis) then `index`.
- Negative axis/index failures use the exact order, exception type, and messages specified above.
- Valid primitive values are retained unchanged and record value semantics remain generated.
- Exact `SELECT` plus `SelectAttrs` composition works through the unchanged [`Operation`](../../../../glossary.md#operation) record.
- No existing Java contract or test changes.
- No Tensor, Shape/result, layout, provenance, graph, gradient, compiler, runtime, backend, or
  execution behavior is introduced.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance); [graph glossary](../../../../glossary.md#graph); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>
- Javadoc, Tensor API, glossary, planning evidence, master plan, and roadmap are independently
  reviewed and synchronized.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>
- All [validation](../../../../glossary.md#term-validation) passes and the final diff contains exactly the permitted paths.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run):

```bash
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.operation.index.SelectSemanticsTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test must verify:

- exact enum [constant](../../../../glossary.md#term-constant) count, name, order, and [`OperationKind`](../../../../glossary.md#operationkind) implementation;
- absence of project-declared enum state/[API](../../../../glossary.md#term-application-programming-interface-api)/nested types;
- exact record status, component names/order/types, interface, and absence of extra instance state;
- zero and primitive maximum values;
- exact negative-[axis](../../../../glossary.md#term-axis) then negative-index validation and messages;
- generated equality, hashing, and diagnostic value semantics;
- exact `Operation(SelectKind.SELECT, SelectAttrs)` reference/value composition;
- the [operation](../../../../glossary.md#operation) contracts contain no [Tensor](../../../../glossary.md#tensor), [Shape](../../../../glossary.md#shape), [layout](../../../../glossary.md#layout), [graph](../../../../glossary.md#graph), [backend](../../../../glossary.md#backend), or [runtime](../../../../glossary.md#term-runtime) imports.

Manually inspect `javap`/reflection output, imports, generated [Javadoc](../../../../glossary.md#term-javadoc) pages, Markdown links,
anchors, code fences, trailing whitespace, task/master/roadmap status, exact eight-path [scope](../../../../glossary.md#term-scope), and
absence of a task-0018B specification.

## Dependencies

- Task 0002 defines current [axis](../../../../glossary.md#term-axis)/[Shape](../../../../glossary.md#shape) terminology used by the documentation.
- Task 0005 defines [`OperationKind`](../../../../glossary.md#operationkind) and [`OperationAttrs`](../../../../glossary.md#operationattrs).
- Task 0006 defines the open typed [`Operation`](../../../../glossary.md#operation) pair.
- Task 0018B depends on this task for public [Tensor](../../../../glossary.md#tensor) expression construction.

## Follow-up tasks

- 0018B: public scalar select Tensor expression, normalization, Shape/layout derivation, and
  provenance.<span data-glossary-navigation="true"> ([scalar select glossary](../../../../glossary.md#scalar-select); [normalization glossary](../../../../glossary.md#term-normalization); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance))</span>
- 0018C–0018J: focused [tensor](../../../../glossary.md#tensor)-index [gather](../../../../glossary.md#gather) and functional-scatter semantic/expression pairs.

Do not create any follow-up specification during this task.

## Architecture impact

Expected impact: None.

The task fills the existing [model](../../../../glossary.md#term-model)-owned [operation](../../../../glossary.md#operation) vocabulary. If implementation requires a new
architecture rule or conflicts with `ARCHITECTURE.md`, stop and report the issue.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0002/0005/0006/0015E/0017G/0017K/0018A, Tensor API,
Compile API, Training API, glossary, current OperationKind/OperationAttrs/Operation and related
selection/layout semantic contracts/tests, Shape terminology, and Java 26 Gradle configuration.

Implement task 0018A exactly. Add only SelectKind.java, SelectAttrs.java, and
SelectSemanticsTest.java under io.github.pho001.synaptik.model.operation.index.

SelectKind contains exactly SELECT and no project state/methods/nested types/metadata. SelectAttrs
contains exactly normalized non-negative int axis then long index, validates in the exact order
with exact messages, retains valid primitives unchanged, and exposes explicit documented
accessors. Document scalar axis removal, the [2,3,4]/axis-1/index-2 -> [2,4] example, exact typed
pairing, and distinctions from WHERE, UNSTACK, SLICE, and tensor-index gather.

Do not add Tensor methods, Shape/rank/bounds/result/layout/provenance logic, gather/scatter types,
gradients, graph/compiler/planning/runtime/backend behavior, factories, dependencies, build or
architecture changes, existing Java edits, or later specs. Stop beyond eight paths or on
architecture uncertainty.

Run all specified validation, then hand the actual diff/evidence to a separate clean-context docs
agent in the same change. It must inspect source/tests/generated Javadoc, finalize permitted
Javadocs/Tensor API/glossary/planning, record Compile API/Training API/capability/architecture and
related-contract no-change conclusions, and rerun validation.

Update task 0018A, model master plan, and roadmap only for planning status/evidence. Do not mark
Complete until both passes succeed. Leave 0018B Draft without a specification. Do not commit/push.
```

## Local decisions

- [Scalar select](../../../../glossary.md#scalar-select) is represented by the first-class `SelectKind.SELECT` identity rather than by
  reusing conditional <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>`WHERE`, individually indexed `UNSTACK`, general [`SLICE`](../../../../glossary.md#slice), or a tensor-index
  gather meaning.<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor); [gather glossary](../../../../glossary.md#gather))</span>
- `SelectAttrs` stores a normalized `int axis` followed by a normalized `long index`. The widths
  match current rank/axis positions and long-valued Shape dimensions without storing an input
  Shape or duplicating input-dependent rank and bounds facts.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [axis glossary](../../../../glossary.md#term-axis); [Shape glossary](../../../../glossary.md#shape); [dimensions glossary](../../../../glossary.md#dimension))</span>
- Both non-negative checks occur in the canonical record constructor, axis first, and the explicit
  accessors return the stored primitive fields directly. Generated record object methods remain
  the value-semantics contract.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- The submitted production Javadocs were retained unchanged after independent review because they
  already document every public type, constant, component, constructor, accessor, failure,
  example, family distinction, and deferred boundary required by this task.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>

## Known limitations

- This task provides semantic identity and normalized intrinsic parameters only. It intentionally
  provides no public <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [parameters glossary](../../../../glossary.md#term-parameter))</span>`Tensor.select`, raw negative normalization, rank or bounds check, result
  Shape/layout, view geometry, provenance, value access, gradient, graph/compiler behavior,
  materialization, backend lowering, or execution. Task 0018B remains Draft for the public
  expression boundary.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [rank glossary](../../../../glossary.md#term-rank); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [view glossary](../../../../glossary.md#term-view-meanings); [provenance glossary](../../../../glossary.md#provenance); [gradient glossary](../../../../glossary.md#term-gradient); [graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [materialization glossary](../../../../glossary.md#materialization); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `/root/implement_model_0018a` created exactly `SelectKind`,
  `SelectAttrs`, and `SelectSemanticsTest`. Independent documentation context
  `/root/implement_model_0018a/review_model_0018a_docs` inspected the actual uncommitted diff,
  implementation evidence, source, tests, generated Javadoc, related contracts, APIs, glossary,
  planning state, and build configuration before finalizing documentation in the same change.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning))</span>
- The documentation pass applied General plus API/Javadoc style and Example format to the two
  production Javadocs, Tensor API, and glossary, and Planning style to this task, the model master
  plan, and roadmap.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Tensor glossary](../../../../glossary.md#tensor); [model glossary](../../../../glossary.md#term-model))</span>
- Reviewed in full: `AGENTS.md`, `ARCHITECTURE.md`, the current architecture plan, overview,
  lifecycle, module-boundary, dependency-rule, and runtime/prepare/backend-boundary explanations;
  documentation rules and General/API-Javadoc/Planning/Example profiles; planning guide and
  roadmap; model capabilities and master plan; tasks 0002, 0005, 0006, 0015E, 0017G, 0017K, and
  0018A; Tensor, Compile, and Training API references; glossary; both new production sources and
  the focused test; OperationKind, OperationAttrs, Operation, WHERE, SLICE, UNSTACK, Shape, and
  related semantic contracts/tests; generated model Javadoc; and Java 26 root/model Gradle
  configuration.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [Compile glossary](../../../../glossary.md#compile); [SLICE glossary](../../../../glossary.md#slice); [Shape glossary](../../../../glossary.md#shape); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- `SelectKind` review confirmed exactly `SELECT`, [`OperationKind`](../../../../glossary.md#operationkind) implementation, no project
  field, method, explicit constructor, per-constant body, or nested type, and no metadata beyond
  enum compiler machinery. <span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [metadata glossary](../../../../glossary.md#term-metadata); [compiler glossary](../../../../glossary.md#term-compiler))</span>`SelectAttrs` review confirmed exactly private final `int axis` then
  `long index`, one public `(int, long)` canonical constructor, explicit direct accessors,
  axis-first validation, exact failure messages, unchanged assignments, generated equality,
  hashing, and diagnostic text, and no nested or extra state.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.index.SelectSemanticsTest` — `BUILD SUCCESSFUL`; XML
  reports 9 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; XML aggregation reports 638 tests across
  75 suites with zero failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL` without Javadoc errors. Generated pages
  contain the type, enum constant, record components, canonical constructor, explicit accessors,
  exact failures, normalized semantics, `[2, 3, 4]`/[axis](../../../../glossary.md#term-axis)-`1`/index-`2` to `[2, 4]` example,
  SELECT/attributes pairing, family distinctions, and deferred boundaries.
- `./gradlew test` — `BUILD SUCCESSFUL`; the repository lifecycle completed with 36 actionable
  tasks and no failing task.
- The implementation handoff also recorded one intentionally parallel model/root Gradle attempt
  that collided in shared test-result files (<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff))</span>`EOFException` and a missing
  `in-progress-results-generic.bin`). Required sequential reruns of both lifecycles passed; this
  superseded result is shared-build-output concurrency evidence, not a product-test failure.<span data-glossary-navigation="true"> ([concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- `javap -classpath modules/model/build/classes/java/main -p -c -s` confirmed the exact enum and
  record structures, `(int, long)` constructor, axis-first/index-second checks, direct field
  assignments/accessors, and generated object methods. Focused reflection tests independently
  confirm visibility, interfaces, component/member order, absence of extra project API/state, and
  exact typed [`Operation`](../../../../glossary.md#operation) composition.
- Production import inspection found only `OperationKind` in `SelectKind` and [`OperationAttrs`](../../../../glossary.md#operationattrs) in
  `SelectAttrs`. Source, bytecode, and focused-test review found no Tensor, Shape, data-type,
  layout, provenance, graph, compiler, planning, prepare, runtime, backend, training, or execution
  dependency or behavior.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance); [graph glossary](../../../../glossary.md#graph))</span>
- Tensor API and glossary now document the current semantic values, normalized axis/index,
  scalar-axis removal, conceptual example, exact composition, validation boundary, and
  distinctions from WHERE, UNSTACK, SLICE, and tensor-index gather while keeping public<span data-glossary-navigation="true"> ([normalized axis glossary](../../../../glossary.md#normalized-axis); [scalar glossary](../../../../glossary.md#term-scalar); [gather glossary](../../../../glossary.md#gather))</span>
  `Tensor.select` and cross-layer work planned.
- Local Markdown validation passed for the five changed documentation/planning files: 346 local
  links and 87 anchors resolve, including the new scalar-select heading and glossary link. All 210
  code-fence markers are balanced. Trailing-whitespace scans found no matches, all eight files end
  with a newline, and `git diff --check` passed.
- Final scope contains exactly the authorized eight paths: two new production sources, one focused
  test, Tensor API, glossary, this task, model master plan, and roadmap. Task, master plan, and
  roadmap consistently mark 0018A Complete. Task 0018B remains Draft, and no task-0018B
  specification exists.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- `docs/api/compile-api.md` remains accurate unchanged because 0018A adds no public Tensor
  expression, compiler entry point, capture, inference, canonicalization, graph conversion, or
  compile artifact. <span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [artifact glossary](../../../../glossary.md#term-artifact))</span>`docs/api/training-api.md` remains accurate unchanged because no gradient,
  autograd, optimizer, parameter, session, or training behavior changed. The capability baseline
  already lists scalar-index select and correctly separates model semantics, public Tensor API,
  compiler, planning, backend prepare, runtime, and tests, so it required no status change.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- OperationKind, OperationAttrs, and Operation remain accurate because the new values implement
  their open typed roles without changing generic compatibility enforcement. WHERE remains ternary
  elementwise choice, UNSTACK remains an individually indexed logical multi-result output, SLICE
  remains same-rank half-open interval selection, and Shape/layout/Tensor/provenance/graph
  contracts remain unchanged because this task derives no result or occurrence state.<span data-glossary-navigation="true"> ([elementwise glossary](../../../../glossary.md#term-elementwise); [rank glossary](../../../../glossary.md#term-rank); [half-open interval glossary](../../../../glossary.md#term-half-open-interval); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, and architecture tests remain
  unchanged because the work stays inside model-owned backend-independent semantics and changes no
  module boundary, dependency direction, or lifecycle. Backend-conformance and integration tests
  remain unchanged because there is no backend or end-to-end behavior. Java 26 Gradle files,
  dependencies, other modules, and later task specifications remain unchanged because the task
  needs no build, dependency, cross-module, or follow-up implementation change.<span data-glossary-navigation="true"> ([dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Implementation notes

- Added the exact one-value `SelectKind` vocabulary and exact two-component `SelectAttrs` record in
  `model.operation.index`, plus one focused structural, validation, value-semantics, distinction,
  composition, and dependency-boundary test.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Finalized the Tensor API, glossary, and synchronized planning evidence through the independent
  documentation pass. No production Javadoc correction was necessary.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## Completion summary

- Completed changes: Implemented and documented first-class scalar-axis SELECT meaning and
  immutable normalized axis/index attributes without public Tensor or cross-layer behavior.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [axis glossary](../../../../glossary.md#term-axis); [immutable glossary](../../../../glossary.md#term-immutable); [normalized axis glossary](../../../../glossary.md#normalized-axis); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Files changed or created: Two production Java files, one focused test, Tensor API, glossary,
  this task specification, model master plan, and implementation roadmap.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: Focused 9-test suite, all 638 model tests across 75 suites, generated model
  Javadoc, full repository tests, javap/reflection/import/source/generated-page review, Markdown
  link/anchor/fence/whitespace checks, exact-scope/status checks, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0018a/review_model_0018a_docs` completed the required independent pass in
  the same overall change.
- Documentation impact: Tensor API and glossary now explain scalar select, normalized parameters,
  axis removal, exact pairing, family distinctions, and current-versus-planned boundaries. Compile
  API, Training API, capabilities, architecture/ADRs/tests, conformance/integration material,
  Gradle, dependencies, other modules, and later tasks remain accurate unchanged for the recorded
  reasons.<span data-glossary-navigation="true"> ([scalar select glossary](../../../../glossary.md#scalar-select); [parameters glossary](../../../../glossary.md#term-parameter); [Compile glossary](../../../../glossary.md#compile); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>
- Javadoc review: Complete for both production types, the enum constant, record components,
  canonical constructor, accessors, failures, example, distinctions, and exclusions; no correction
  was required.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Glossary impact: Added scalar-select terminology and aligned operation-kind and attribute
  inventories with the implemented semantic family.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- Unresolved issues: None.
- Follow-up required: None for task 0018A. Task 0018B remains Draft without a detailed
  specification.

Status: Complete
