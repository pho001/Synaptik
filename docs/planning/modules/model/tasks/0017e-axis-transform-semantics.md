# Task 0017E: [Axis](../../../../glossary.md#term-axis)-Transform Semantics

## Status

Complete

## Goal

Define typed, [backend](../../../../glossary.md#backend)-independent semantic [identities](../../../../glossary.md#term-identity-meanings) and [immutable](../../../../glossary.md#term-immutable) normalized attributes for
[axis](../../../../glossary.md#term-axis) [permutation](../../../../glossary.md#permutation), [singleton](../../../../glossary.md#term-singleton)-axis insertion, and singleton-axis removal.

This task defines [operation](../../../../glossary.md#operation) meaning only. Public [Tensor](../../../../glossary.md#tensor) methods, negative-axis [normalization](../../../../glossary.md#term-normalization),
input-[rank](../../../../glossary.md#term-rank) and singleton [validation](../../../../glossary.md#term-validation), result [Shape](../../../../glossary.md#shape)/[layout](../../../../glossary.md#layout) derivation, [provenance](../../../../glossary.md#provenance), [gradients](../../../../glossary.md#term-gradient),
[materialization](../../../../glossary.md#materialization), [compiler](../../../../glossary.md#term-compiler) behavior, and execution remain in later tasks and layers.

## [Scope](../../../../glossary.md#term-scope)

- Add one public `AxisTransformKind` enum implementing [`OperationKind`](../../../../glossary.md#operationkind).
- Define exactly `PERMUTE`, `EXPAND_DIMS`, and `SQUEEZE`, in that order.
- Add one public `PermutationAttrs` record implementing [`OperationAttrs`](../../../../glossary.md#operationattrs) with exactly one
  `List<Integer> axes` component.
- Define `axes[outputAxis]` as the normalized input [axis](../../../../glossary.md#term-axis) placed at that output position.
- Require `axes` to be a complete normalized [permutation](../../../../glossary.md#permutation) of `[0, rank)` where [rank](../../../../glossary.md#term-rank) is list size.
- Accept an empty permutation as the [scalar](../../../../glossary.md#term-scalar) [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [identity](../../../../glossary.md#term-identity-meanings) permutation.
- Validate list and [elements](../../../../glossary.md#term-element) deterministically, then store one [immutable](../../../../glossary.md#term-immutable) `List.copyOf` [snapshot](../../../../glossary.md#term-snapshot).
- Add one public `AxisTransformAttrs` record implementing `OperationAttrs` with exactly one
  non-negative normalized `int axis` component.
- Use `AxisTransformAttrs` for `EXPAND_DIMS` insertion position and `SQUEEZE` removal position.
- Document exact valid kind/attributes pairings without adding generic compatibility [validation](../../../../glossary.md#term-validation).
- Document that [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `transpose()` is a future convenience over
  `PERMUTE + PermutationAttrs([1, 0])`, not a fourth semantic kind.
- Add one focused same-package semantic test for all three cohesive production types.
- Keep all production types in the existing `model.operation.layout` package.
- Finalize Javadocs, Tensor API semantic reference, glossary, task evidence, model master plan, and
  roadmap through the mandatory independent documentation pass.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- public `Tensor.permute`, [`transpose`](../../../../glossary.md#term-transpose), `expandDims`, `squeeze`, another Tensor method, expression
  helper, factory, or task-0017F implementation<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- a `TRANSPOSE` enum constant, transpose attributes, matrix-only metadata, or backend transpose
  alias<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [metadata glossary](../../../../glossary.md#term-metadata); [backend glossary](../../../../glossary.md#backend); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- raw varargs, negative-axis normalization, input rank, input Shape, input Dimension, output Shape,
  result descriptor, layout, stride, offset, view flag, Tensor identity, label, storage, or
  provenance<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [normalization glossary](../../../../glossary.md#term-normalization); [rank glossary](../../../../glossary.md#term-rank); [Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [descriptor glossary](../../../../glossary.md#term-descriptor); [layout glossary](../../../../glossary.md#layout); [stride glossary](../../../../glossary.md#term-stride); [view glossary](../../../../glossary.md#term-view-meanings); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance))</span>
- validating that a permutation length equals an eventual input rank; the immutable list validates
  itself as a complete permutation of its own length only<span data-glossary-navigation="true"> ([permutation glossary](../../../../glossary.md#permutation); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- validating insertion against `[0, inputRank]`, removal against `[0, inputRank)`, or checking that
  a squeezed input dimension is a static singleton<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton))</span>
- automatic squeeze of every singleton axis, multi-axis insertion/removal, ellipsis, names, or a
  numeric sentinel
- changing Shape, Dimension, LayoutDescriptor, LayoutKind, TensorDescriptor, Tensor,
  TensorFactory, TensorProvenance, existing layout-operation semantics, Operation foundations, or
  their tests<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- operation factories, registries, parsers, visitors, aliases, string dispatch, reflection
  discovery, maps, services, arity/result/cost/fusion metadata, backend support, or routes<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [arity glossary](../../../../glossary.md#term-arity); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- gradients, inverse permutation construction, autograd, compiler-generated operations, optimizer,
  or training behavior<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [compiler glossary](../../../../glossary.md#term-compiler); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- graph capture/canonicalization, planning materialization, prepare, runtime, backend storage or
  execution, engine, tracing, ONNX mapping, or conformance behavior<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [planning glossary](../../../../glossary.md#term-planning); [materialization glossary](../../../../glossary.md#materialization); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [ONNX glossary](../../../../glossary.md#term-onnx); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- another [module](../../../../glossary.md#term-module), [dependency](../../../../glossary.md#term-dependency), [Gradle](../../../../glossary.md#term-gradle)/build change, architecture change, or task-0017F specification

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
- [Task 0017C](0017c-reshape-and-expand-semantics.md)
- [Task 0017D1](0017d1-expand-tensor-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The selected capability baseline includes:

- `permute(int...)` for arbitrary complete [axis](../../../../glossary.md#term-axis) reordering;
- [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) parameterless `transpose()` as convenience for [permutation](../../../../glossary.md#permutation) `[1, 0]`;
- `expandDims(int)` for inserting one [singleton](../../../../glossary.md#term-singleton) [dimension](../../../../glossary.md#dimension); and
- `squeeze(int)` for removing one selected singleton dimension.

The read-only `legacy/pre-rewrite` branch represents these as `PERMUTE`, `EXPAND_DIMS`, and
`SQUEEZE` operations. `transpose()` validates [rank](../../../../glossary.md#term-rank) two and delegates to `permute(1, 0)`, so it does
not introduce a separate [operation](../../../../glossary.md#operation) [identity](../../../../glossary.md#term-identity-meanings). Legacy operation [descriptors](../../../../glossary.md#term-descriptor) retain normalized axes;
their builders own negative-axis [normalization](../../../../glossary.md#term-normalization), input-rank checks, [Shape](../../../../glossary.md#shape)/[stride](../../../../glossary.md#term-stride) derivation,
storage [views](../../../../glossary.md#view), and [gradient](../../../../glossary.md#term-gradient) inverses.

This task preserves only typed semantic vocabulary and [immutable](../../../../glossary.md#term-immutable) normalized [parameters](../../../../glossary.md#term-parameter). It does
not copy legacy [mutable](../../../../glossary.md#term-mutable) arrays, [graph](../../../../glossary.md#graph) builders, storage [aliasing](../../../../glossary.md#term-aliasing), gradient callbacks, operation
traits, [compiler](../../../../glossary.md#term-compiler)/[lowering](../../../../glossary.md#lowering) code, [kernels](../../../../glossary.md#kernel), or [runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) behavior.

## Architecture constraints

- Production stays in `modules/model`, which owns [backend](../../../../glossary.md#backend)-independent [operation](../../../../glossary.md#operation) semantics.
- Semantic types store normalized intrinsic [parameters](../../../../glossary.md#term-parameter) only, never an input [Tensor](../../../../glossary.md#tensor) or [Shape](../../../../glossary.md#shape).
- `PERMUTE` reorders axes according to an output-to-input mapping but performs no Shape/layout
  calculation here.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>
- `EXPAND_DIMS` inserts one [singleton](../../../../glossary.md#term-singleton) [axis](../../../../glossary.md#term-axis) at a normalized output position.
- `SQUEEZE` removes one selected singleton input axis; this semantic value cannot prove the
  eventual dimension is one.<span data-glossary-navigation="true"> ([dimension glossary](../../../../glossary.md#dimension))</span>
- `transpose()` remains a public convenience to be built by task 0017F from PERMUTE semantics.
- Generic `Operation` stays an open non-null kind/attributes pair and does not enforce family
  pairings, arity, rank, Shape, layout, gradient, or backend rules.<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [rank glossary](../../../../glossary.md#term-rank); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Package direction is `model.operation.layout -> model.operation + java.base` only.
- No tensor, shape, layout-value, storage, graph, compiler, planning, prepare, runtime, backend,
  engine, trace, or training dependency may be introduced.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [trace glossary](../../../../glossary.md#trace); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Stop if implementation needs input-dependent validation, another semantic type, dependency, or
  architecture change.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>

## Package impact

No package is added or moved.

- `io.github.pho001.synaptik.model.operation.layout.AxisTransformKind` — semantic vocabulary.
- `io.github.pho001.synaptik.model.operation.layout.PermutationAttrs` — immutable complete
  output-to-input axis permutation.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [permutation glossary](../../../../glossary.md#permutation))</span>
- `io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs` — normalized single-axis
  position shared by insertion/removal meanings.
- `AxisTransformSemanticsTest` — same-package focused structural and semantic test.

The existing [layout](../../../../glossary.md#layout)-[operation](../../../../glossary.md#operation) package already owns [contiguous](../../../../glossary.md#term-contiguous), reshape, and expand meanings; these
[axis](../../../../glossary.md#term-axis)-[coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) transformations are cohesive with that [ownership](../../../../glossary.md#term-ownership-meanings).

## Required contract

### Semantic kind vocabulary

Create exactly:

```java
public enum AxisTransformKind implements OperationKind {
    PERMUTE,
    EXPAND_DIMS,
    SQUEEZE
}
```

The enum declares no project field, constructor, method, nested type, [alias](../../../../glossary.md#aliasing-and-arena), [arity](../../../../glossary.md#term-arity), [axis](../../../../glossary.md#term-axis), [Shape](../../../../glossary.md#shape),
[layout](../../../../glossary.md#layout), result [metadata](../../../../glossary.md#term-metadata), or per-[constant](../../../../glossary.md#term-constant) class body. Inherited enum names are diagnostic text only.

Document these meanings:

| Kind | One-input semantic meaning | Attributes |
|---|---|---|
| `PERMUTE` | output axis `i` corresponds to input axis `axes[i]`; logical values keep their [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) association under reordered axes | `PermutationAttrs` |
| `EXPAND_DIMS` | insert one [extent](../../../../glossary.md#term-extent)-one axis at the normalized output position while preserving logical values | `AxisTransformAttrs` |
| `SQUEEZE` | remove one selected extent-one input axis while preserving logical values | `AxisTransformAttrs` |

Do not add [`TRANSPOSE`](../../../../glossary.md#term-transpose). A later [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) convenience explicitly creates
`PERMUTE + PermutationAttrs(List.of(1, 0))`.

### [Permutation](../../../../glossary.md#permutation) attributes

Create exactly:

```java
public record PermutationAttrs(List<Integer> axes) implements OperationAttrs
```

It has exactly one component, one public [canonical](../../../../glossary.md#term-canonical-form) constructor, one explicit documented accessor,
and record-generated object methods. It adds no array overload, factory, inverse, [rank](../../../../glossary.md#term-rank) accessor,
[lookup](../../../../glossary.md#term-lookup) method, nested type, or cached state.

Constructor [validation](../../../../glossary.md#term-validation) order:

1. null-check [`axes`](../../../../glossary.md#term-axis) with exact message `axes`;
2. create constructor-local `boolean[] seen` with length `axes.size()`;
3. inspect [elements](../../../../glossary.md#term-element) in ascending index order;
4. null-check each element with exact message `axes[<index>]`;
5. reject a negative value with exact message
   `axes[<index>] must be non-negative: <value>`;
6. reject a value at least permutation rank with exact message
   `axes[<index>] must be less than permutation rank <rank>: <value>`;
7. reject the first duplicate with exact message
   `axes contains duplicate axis <value> at index <index>`;
8. only after all validation, assign `axes = List.copyOf(axes)` exactly once.

Range plus uniqueness makes the list a complete permutation. Empty list is valid [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)
[identity](../../../../glossary.md#term-identity-meanings). Store no `seen` array or inverse. Preserve order and integer values; do not promise list
object [identity](../../../../glossary.md#term-reference-identity).

`axes()` returns the [immutable](../../../../glossary.md#term-immutable) [snapshot](../../../../glossary.md#term-snapshot). Element `i` names the normalized input axis used as output
axis `i`.

### Single-[axis](../../../../glossary.md#term-axis) attributes

Create exactly:

```java
public record AxisTransformAttrs(int axis) implements OperationAttrs
```

It has one component, one [canonical](../../../../glossary.md#term-canonical-form) constructor, one explicit documented accessor, and no other
state/[API](../../../../glossary.md#term-application-programming-interface-api). Reject a negative axis with exact message:

```text
axis must be non-negative: <axis>
```

Retain every non-negative value unchanged. This record does not know a [rank](../../../../glossary.md#term-rank): task 0017F later
normalizes a raw axis and validates whether it is an insertion position or existing input axis.

### Typed composition

Document exactly these valid pairings:

```java
new Operation(AxisTransformKind.PERMUTE, permutationAttrs)
new Operation(AxisTransformKind.EXPAND_DIMS, axisTransformAttrs)
new Operation(AxisTransformKind.SQUEEZE, axisTransformAttrs)
```

[Operation](../../../../glossary.md#operation) retains exact kind and attributes references. Do not use [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), add a
compatibility validator/factory, or change Operation. Family pairings and one-input context are
documented, not generically enforced.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/AxisTransformKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/PermutationAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/AxisTransformAttrs.java`

Test:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/layout/AxisTransformSemanticsTest.java`

Documentation/[planning](../../../../glossary.md#term-planning):

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without changing unless an inconsistency requires stopping:

- Compile API, Training API, capabilities, existing operation/layout contracts, architecture/ADRs,
  architecture/conformance/integration tests, Gradle, dependencies, and other modules.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [operation glossary](../../../../glossary.md#operation); [layout glossary](../../../../glossary.md#layout); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

At most three production files, one focused test, and five documentation/[planning](../../../../glossary.md#term-planning) files: nine
paths total. If another Java type/test, public method, existing Java edit, [dependency](../../../../glossary.md#term-dependency), build or
architecture change, or tenth path is required, stop and report. Do not create task 0017F.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Document the enum, every constant, both records, constructors, components, and explicit
  accessors completely.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>
- Explain output-to-input [permutation](../../../../glossary.md#permutation) order with a concrete example such as `[1, 0, 2]`.
- Explain insertion versus removal [axis](../../../../glossary.md#term-axis) position and why [rank](../../../../glossary.md#term-rank) [validation](../../../../glossary.md#term-validation) is deferred.
- Explain [scalar](../../../../glossary.md#term-scalar) empty permutation and [immutable](../../../../glossary.md#term-immutable) list [ownership](../../../../glossary.md#term-ownership-meanings).
- Document exact validation order, [parameters](../../../../glossary.md#term-parameter), results, null failures, and value failures.
- Explain [transpose](../../../../glossary.md#term-transpose) as future [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) PERMUTE convenience, not a distinct kind.
- Explain that these values define no Tensor, Shape, layout, storage view, gradient, compiler,
  backend, or execution behavior.<span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#view))</span><span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend))</span>
- Independently review OperationKind, OperationAttrs, Operation, existing layout-operation
  semantics, and axis terminology; record why unchanged Javadocs remain accurate or stop on an
  out-of-scope discrepancy.<span data-glossary-navigation="true"> ([OperationKind glossary](../../../../glossary.md#operationkind); [OperationAttrs glossary](../../../../glossary.md#operationattrs); [Operation glossary](../../../../glossary.md#operation); [scope glossary](../../../../glossary.md#term-scope))</span>

## Acceptance criteria

- Enum is public, implements [OperationKind](../../../../glossary.md#operationkind), and has exactly the three ordered [constants](../../../../glossary.md#term-constant).
- Enum adds no project field, method, constructor, nested type, [metadata](../../../../glossary.md#term-metadata), or [TRANSPOSE](../../../../glossary.md#term-transpose) constant.
- PermutationAttrs is exactly one-component public record with deterministic validation and one
  immutable snapshot.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [immutable glossary](../../../../glossary.md#term-immutable); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Empty, identity, and reordered complete permutations are accepted; null, negative, out-of-range,
  and duplicate values fail with exact type/message/precedence.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Caller list mutation cannot affect stored [axes](../../../../glossary.md#term-axis); accessor mutation fails.
- AxisTransformAttrs is exactly one-component public record accepting all non-negative values and
  rejecting negatives with exact message.
- Explicit typed pairings retain exact attribute references and no generic validator is added.
- No [Tensor](../../../../glossary.md#tensor)/[Shape](../../../../glossary.md#shape)/[layout](../../../../glossary.md#layout)/[provenance](../../../../glossary.md#provenance)/cross-layer behavior or existing Java contract changes.
- Complete Javadocs, Tensor API, glossary, independent documentation review, and synchronized
  planning evidence/status are finished before marking Complete.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.layout.AxisTransformSemanticsTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test must verify exact enum order/[API](../../../../glossary.md#term-application-programming-interface-api) [shape](../../../../glossary.md#shape); exact record components/accessors/API;
empty, [identity](../../../../glossary.md#term-identity-meanings), [transpose](../../../../glossary.md#term-transpose), and general permutations; output-to-input order; list [snapshot](../../../../glossary.md#term-snapshot) and
immutability; every validation failure and precedence; all non-negative single [axes](../../../../glossary.md#term-axis); record value
semantics; exact [Operation](../../../../glossary.md#operation) composition/reference retention; absence of TRANSPOSE/extra members;
and no forbidden imports or dependencies.

Inspect `javap -p -c -s`, reflection, imports, and source. Confirm exactly three production types,
specified enum/record shapes, one constructor-local seen array, one List.copyOf, no retained helper
state, and no [Tensor](../../../../glossary.md#tensor)/Shape/[layout](../../../../glossary.md#layout)/storage/[graph](../../../../glossary.md#graph)/[compiler](../../../../glossary.md#term-compiler)/[planning](../../../../glossary.md#term-planning)/[prepare](../../../../glossary.md#prepare)/[runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend)/training
types. Validate generated [Javadoc](../../../../glossary.md#term-javadoc), API/glossary terminology, links, anchors, fences, whitespace,
exact nine-path [scope](../../../../glossary.md#term-scope), synchronized statuses, and absence of a task-0017F specification.

## Dependencies

- Task 0005 supplies [OperationKind](../../../../glossary.md#operationkind) and [OperationAttrs](../../../../glossary.md#operationattrs).
- Task 0006 supplies [immutable](../../../../glossary.md#term-immutable) generic [Operation](../../../../glossary.md#operation) composition.
- Task 0002 supplies [axis](../../../../glossary.md#term-axis)/[Shape](../../../../glossary.md#shape) terminology for documentation but is not a production [dependency](../../../../glossary.md#term-dependency).
- Tasks 0017C–0017D1 establish the existing layout-operation package and semantic/expression layer
  split without becoming production dependencies of these new types.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>

## Follow-up tasks

- 0017F remains Draft for public permute/transpose/expandDims/squeeze expressions, raw axis
  normalization, Shape and layout derivation, provenance, and descriptor construction.<span data-glossary-navigation="true"> ([transpose glossary](../../../../glossary.md#term-transpose); [axis glossary](../../../../glossary.md#term-axis); [normalization glossary](../../../../glossary.md#term-normalization); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- [Compiler](../../../../glossary.md#term-compiler) later owns [identity](../../../../glossary.md#term-identity-meanings)/inverse-chain [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form) and [graph](../../../../glossary.md#graph) constraints.
- [Planning](../../../../glossary.md#term-planning)/[backend](../../../../glossary.md#backend) [prepare](../../../../glossary.md#prepare) later own [materialization](../../../../glossary.md#materialization) and concrete [lowering](../../../../glossary.md#lowering).
- Training/compiler-generated semantics later own inverse-transform [gradient](../../../../glossary.md#term-gradient) expressions.

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. The [model](../../../../glossary.md#term-model) already owns [backend](../../../../glossary.md#backend)-neutral [operation](../../../../glossary.md#operation) semantics, and the new
types remain within its existing [layout](../../../../glossary.md#layout)-operation package and [dependency direction](../../../../glossary.md#java-platform-and-build-tools).

If implementation needs [Tensor](../../../../glossary.md#tensor)/[Shape](../../../../glossary.md#shape)/layout state, another [dependency](../../../../glossary.md#term-dependency), generic Operation changes,
or architecture changes, stop and report.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0002/0005/0006/0017C/0017D1/0017E, Tensor API, Compile API,
Training API, glossary, current OperationKind/OperationAttrs/Operation and layout-operation
semantic contracts/tests, and Java 26 Gradle configuration.

Implement task 0017E exactly. Add only AxisTransformKind.java, PermutationAttrs.java,
AxisTransformAttrs.java, and AxisTransformSemanticsTest.java for Java code/tests under
io.github.pho001.synaptik.model.operation.layout.

The enum contains exactly PERMUTE, EXPAND_DIMS, SQUEEZE in order with no project state/methods/
nested types/metadata. PermutationAttrs is exactly immutable List<Integer> axes using output-to-
input order, validates a complete normalized permutation with exact order/messages, accepts empty
scalar permutation, and snapshots once. AxisTransformAttrs contains exactly non-negative normalized
int axis with exact validation/message. Document exact kind/attribute pairings and transpose as
PERMUTE [1,0], not a separate kind.

Do not add Tensor methods, Shape/layout/result/provenance logic, raw-axis normalization, gradients,
graph/compiler/planning/runtime/backend behavior, factories/registries, dependencies, build/
architecture changes, existing Java edits, or later specs. Stop beyond nine paths or on
architecture uncertainty.

Run every specified focused/aggregate test, Javadoc, javap/reflection/import/manual,
documentation/link/whitespace/scope/status check. Then hand the actual diff/evidence to a separate
clean-context documentation agent in the same change. It must inspect source/tests/generated
Javadoc, finalize permitted Javadocs/Tensor API/glossary/planning, record related-contract/
capability/Compile API/Training API/architecture no-change conclusions, and rerun validation.

Update task 0017E, model master plan, and roadmap only for planning status/evidence. Do not mark
0017E Complete until both passes succeed. Leave 0017F Draft without a specification. Do not commit
or push.
```

## Local decisions

- `transpose()` has no semantic kind because legacy and planned API behavior make it a rank-two
  convenience over <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>`PERMUTE [1, 0]`.
- [Permutation](../../../../glossary.md#permutation) order is explicitly output-to-input: `axes[i]` identifies the input axis occupying
  output axis `i`.
- PermutationAttrs validates completeness against its own list size, making the semantic value
  normalized and self-consistent without input Shape state.
- An empty [axes](../../../../glossary.md#term-axis) list is the [scalar](../../../../glossary.md#term-scalar) [identity](../../../../glossary.md#term-identity-meanings) permutation, consistent with [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [Shape](../../../../glossary.md#shape) support.
- AxisTransformAttrs is shared because both insertion and removal need one normalized non-negative
  axis; the kind supplies the different meaning and later Tensor construction supplies rank rules.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [rank glossary](../../../../glossary.md#term-rank))</span>
- Negative raw axes remain public request syntax for task 0017F and never enter semantic attrs.

## Known limitations

- These values cannot validate an eventual input rank, inserted-axis bound, squeezed singleton, or
  result Shape/layout.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [axis glossary](../../../../glossary.md#term-axis); [singleton glossary](../../../../glossary.md#term-singleton); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout))</span>
- No public Tensor construction, transpose convenience, gradient inverse, compiler capture,
  canonicalization, materialization, backend lowering, runtime execution, or ONNX mapping exists.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [transpose glossary](../../../../glossary.md#term-transpose); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [materialization glossary](../../../../glossary.md#materialization); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [runtime glossary](../../../../glossary.md#term-runtime); [ONNX glossary](../../../../glossary.md#term-onnx))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) reviewed the [architecture contract](../../../../glossary.md#architecture-contract) and focused [lifecycle](../../../../glossary.md#lifecycle)/[module](../../../../glossary.md#term-module)/[dependency](../../../../glossary.md#term-dependency)/[runtime](../../../../glossary.md#term-runtime)-
boundary explanations; documentation and planning rules; roadmap; [model](../../../../glossary.md#term-model) capabilities/master plan;
tasks 0002, 0005, 0006, 0017C, and 0017D1; current [operation](../../../../glossary.md#operation) and [layout](../../../../glossary.md#layout)-operation contracts/tests;
[Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile)/Training APIs and glossary; and the Java 26 root/model [Gradle](../../../../glossary.md#term-gradle) configuration.

The read-only legacy branch confirmed three semantic [identities](../../../../glossary.md#term-identity-meanings), complete normalized [permutation](../../../../glossary.md#permutation),
single-[axis](../../../../glossary.md#term-axis) insertion/removal, [transpose](../../../../glossary.md#term-transpose) delegation to `[1, 0]`, [view](../../../../glossary.md#term-view-meanings)-oriented [Shape](../../../../glossary.md#shape)/[stride](../../../../glossary.md#term-stride)
behavior, and inverse [gradient](../../../../glossary.md#term-gradient) operations. Coupled legacy arrays, [graph](../../../../glossary.md#graph)/storage/gradient builders,
traits, [lowering](../../../../glossary.md#lowering), [kernels](../../../../glossary.md#kernel), runtime, and [backends](../../../../glossary.md#backend) were excluded.

Implementation and independent documentation validation:

- Implementation context `/root/implement_model_0017e` added exactly the three production types
  and focused semantic test. Clean documentation context
  `/root/implement_model_0017e/docs_review_0017e` independently read the required architecture,
  documentation and planning profiles, plans and predecessor tasks, APIs, glossary, final source
  and tests, generated Javadoc, XML reports, bytecode, build configuration, and complete live diff.
  It applied General and API/Javadoc style to Java, Tensor API, and glossary; Planning style to
  this task, the model master plan, and roadmap; and Example format to the output-to-input example.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Independent source and generated-Javadoc review found all three submitted production Javadocs
  complete unchanged. They document exact kind/attributes pairings, output-to-input order with
  `[1, 0, 2]`, the empty scalar permutation, immutable snapshot ownership, validation order and
  exact failures, insertion versus removal position, deferred rank/singleton checks, transpose as
  future <span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [immutable glossary](../../../../glossary.md#term-immutable); [snapshot glossary](../../../../glossary.md#term-snapshot); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [rank glossary](../../../../glossary.md#term-rank); [singleton glossary](../../../../glossary.md#term-singleton))</span>`PERMUTE [1, 0]`, and the Tensor/Shape/layout/storage/provenance/gradient/compiler/backend/
  execution boundaries.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.layout.AxisTransformSemanticsTest` —
  `BUILD SUCCESSFUL`; XML records 11 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; 64 XML suites record 508 tests with zero
  failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`; generated pages contain the enum and
  every constant, both record components/constructors/accessors, pairings, ownership, validation,
  result semantics, transpose boundary, and cross-layer exclusions.
- `./gradlew test` — `BUILD SUCCESSFUL`; the root lifecycle completed 36 actionable tasks
  without failure.
- `javap -p -c -s` confirms exact ordered enum [constants](../../../../glossary.md#term-constant) and [compiler](../../../../glossary.md#term-compiler) enum machinery only;
  `PermutationAttrs` has one `List` field, constructor-local `boolean[]`, indexed validation in the
  specified order, one `List.copyOf`, a direct accessor, and generated record value methods; and
  `AxisTransformAttrs` has one `int` field, one non-negative check, a direct accessor, and generated
  record value methods. Focused reflection tests independently verify the same API shape and exact
  Operation reference retention.
- Production imports are only [`OperationKind`](../../../../glossary.md#operationkind) or [`OperationAttrs`](../../../../glossary.md#operationattrs) plus `List` and `Objects`.
  Source/bytecode review found no Tensor, Shape, layout value, storage, provenance, gradient,
  graph, compiler, planning, prepare, runtime, backend, engine, trace, or training dependency or
  behavior.<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [trace glossary](../../../../glossary.md#trace))</span>
- Tensor API now presents the three semantic kinds and both attributes values as current, gives a
  concrete `[1, 0, 2]` output-to-input example, and keeps all public Tensor construction and
  input-dependent/cross-layer behavior planned. The glossary adds reusable axis-transform and
  permutation terminology with the same status and boundaries.
- A targeted Markdown validator resolved 246 local file links and heading anchors across the five
  changed documentation/planning files. Backtick fences are balanced (138 Tensor API, fourteen
  task, two master plan; none in glossary or roadmap), no tilde fences or trailing whitespace are
  present, every changed file ends with a newline, and `git diff --check` passes.
- Exact scope is the nine authorized paths: three production files, one focused test, Tensor API,
  glossary, this task, model master plan, and roadmap. Task 0017E, the master-plan row/current
  status/notes, and roadmap frontier/table are synchronized as Complete. Task 0017F remains Draft,
  and no detailed task-0017F specification exists. No commit or push was performed.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [commit glossary](../../../../glossary.md#term-commit))</span>
- `OperationKind`, `OperationAttrs`, and `Operation` remain accurate unchanged because the new
  values conform to their existing open typed contracts and generic Operation intentionally does
  not enforce pairings. Existing contiguous and target-shape operation semantics remain distinct
  and unchanged because they describe parameterless canonical geometry or exact target Shapes,
  not axis-coordinate transforms. Existing axis terminology remains accurate and is extended only
  to name the new normalized positions and permutation order.<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [target glossary](../../../../glossary.md#term-target); [canonical glossary](../../../../glossary.md#term-canonical-form); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- `capabilities.md` already inventories permute/transpose/expand-dimensions/squeeze capability and
  the layer split. Compile API remains accurate because no public expression, capture, inference,
  canonicalization, artifact, or materialization behavior was added. Training API remains accurate
  because no gradient, autograd, optimizer, parameter, publication, or session behavior changed.<span data-glossary-navigation="true"> ([dimensions glossary](../../../../glossary.md#dimension); [capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [artifact glossary](../../../../glossary.md#term-artifact); [materialization glossary](../../../../glossary.md#materialization); [autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer); [parameter glossary](../../../../glossary.md#term-parameter); [publication glossary](../../../../glossary.md#term-publication))</span>
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests, backend-
  conformance material, and integration tests remain accurate unchanged because module ownership,
  dependency direction, lifecycle, backend behavior, and end-to-end behavior did not change.
  Java 26 Gradle configuration, dependencies, other modules, and unrelated tests also remain
  unchanged because the task adds only model-owned semantic values.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>

## Implementation notes

- Added exact `PERMUTE`, `EXPAND_DIMS`, and `SQUEEZE` semantic identities in the existing
  operation-layout package with no extra project API or state.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [layout glossary](../../../../glossary.md#layout); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Added immutable complete output-to-input permutation attributes and normalized non-negative
  single-axis insertion/removal attributes with exact validation and ownership contracts.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [permutation glossary](../../../../glossary.md#permutation); [axis glossary](../../../../glossary.md#term-axis); [validation glossary](../../../../glossary.md#term-validation); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Added the focused eleven-test suite for enum/record shape, scalar/identity/transpose/general
  permutations, output-to-input order, snapshot immutability, every failure and precedence,
  representative axes, record semantics, exact Operation composition, and forbidden state.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [identity glossary](../../../../glossary.md#term-identity-meanings); [transpose glossary](../../../../glossary.md#term-transpose); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Finalized Tensor API, glossary, task evidence, model master plan, and roadmap in the mandatory
  independent documentation context; the submitted production Javadocs needed no correction.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [model glossary](../../../../glossary.md#term-model))</span>
- Added no Tensor method, Shape/layout/result/provenance behavior, raw-axis normalization,
  gradient, compiler/planning/prepare/runtime/backend behavior, dependency, build change, or
  architecture change.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [normalization glossary](../../../../glossary.md#term-normalization); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Completion summary

- Completed changes: Implemented and documented exact axis permutation, singleton-axis insertion,
  and selected singleton-axis removal semantic values.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [permutation glossary](../../../../glossary.md#permutation); [singleton glossary](../../../../glossary.md#term-singleton))</span>
- Files changed or created: Exactly three production Java files, one focused test, Tensor API,
  glossary, this task, model master plan, and roadmap.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: Focused tests passed 11/11; all 508 model tests across 64 suites, generated
  model Javadoc, root tests, bytecode/reflection/import/source/generated-Javadoc checks, 246
  Markdown link/anchor checks, fence/whitespace/newline checks, exact-scope/status checks, and<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>
  `git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0017e/docs_review_0017e` completed the independent pass using General,
  API/Javadoc, Planning, and Example profiles.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Tensor API and glossary present axis-transform semantic values as current
  while public Tensor methods and all input-dependent and cross-layer behavior remain planned.
- Javadoc review: All three new Javadocs are complete unchanged; related operation, layout-
  operation, capability, API, architecture, build, test, and cross-layer contracts remain accurate
  for the reasons recorded above.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [layout glossary](../../../../glossary.md#layout))</span>
- Glossary impact: Implementation status and reusable axis-transform, permutation, normalized-axis,
  kind, and attributes distinctions now include the completed semantics.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0017E. Task 0017F remains Draft without a detailed
  specification.

Status: Complete
