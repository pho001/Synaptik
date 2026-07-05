# Task 0017E: Axis-Transform Semantics

## Status

Complete

## Goal

Define typed, backend-independent semantic identities and immutable normalized attributes for
axis permutation, singleton-axis insertion, and singleton-axis removal.

This task defines operation meaning only. Public Tensor methods, negative-axis normalization,
input-rank and singleton validation, result Shape/layout derivation, provenance, gradients,
materialization, compiler behavior, and execution remain in later tasks and layers.

## Scope

- Add one public `AxisTransformKind` enum implementing `OperationKind`.
- Define exactly `PERMUTE`, `EXPAND_DIMS`, and `SQUEEZE`, in that order.
- Add one public `PermutationAttrs` record implementing `OperationAttrs` with exactly one
  `List<Integer> axes` component.
- Define `axes[outputAxis]` as the normalized input axis placed at that output position.
- Require `axes` to be a complete normalized permutation of `[0, rank)` where rank is list size.
- Accept an empty permutation as the scalar rank-zero identity permutation.
- Validate list and elements deterministically, then store one immutable `List.copyOf` snapshot.
- Add one public `AxisTransformAttrs` record implementing `OperationAttrs` with exactly one
  non-negative normalized `int axis` component.
- Use `AxisTransformAttrs` for `EXPAND_DIMS` insertion position and `SQUEEZE` removal position.
- Document exact valid kind/attributes pairings without adding generic compatibility validation.
- Document that rank-two `transpose()` is a future convenience over
  `PERMUTE + PermutationAttrs([1, 0])`, not a fourth semantic kind.
- Add one focused same-package semantic test for all three cohesive production types.
- Keep all production types in the existing `model.operation.layout` package.
- Finalize Javadocs, Tensor API semantic reference, glossary, task evidence, model master plan, and
  roadmap through the mandatory independent documentation pass.

## Out of scope

- public `Tensor.permute`, `transpose`, `expandDims`, `squeeze`, another Tensor method, expression
  helper, factory, or task-0017F implementation
- a `TRANSPOSE` enum constant, transpose attributes, matrix-only metadata, or backend transpose
  alias
- raw varargs, negative-axis normalization, input rank, input Shape, input Dimension, output Shape,
  result descriptor, layout, stride, offset, view flag, Tensor identity, label, storage, or
  provenance
- validating that a permutation length equals an eventual input rank; the immutable list validates
  itself as a complete permutation of its own length only
- validating insertion against `[0, inputRank]`, removal against `[0, inputRank)`, or checking that
  a squeezed input dimension is a static singleton
- automatic squeeze of every singleton axis, multi-axis insertion/removal, ellipsis, names, or a
  numeric sentinel
- changing Shape, Dimension, LayoutDescriptor, LayoutKind, TensorDescriptor, Tensor,
  TensorFactory, TensorProvenance, existing layout-operation semantics, Operation foundations, or
  their tests
- operation factories, registries, parsers, visitors, aliases, string dispatch, reflection
  discovery, maps, services, arity/result/cost/fusion metadata, backend support, or routes
- gradients, inverse permutation construction, autograd, compiler-generated operations, optimizer,
  or training behavior
- graph capture/canonicalization, planning materialization, prepare, runtime, backend storage or
  execution, engine, tracing, ONNX mapping, or conformance behavior
- another module, dependency, Gradle/build change, architecture change, or task-0017F specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0005](0005-operation-semantic-foundation.md)
- [Task 0006](0006-operation-model.md)
- [Task 0017C](0017c-reshape-and-expand-semantics.md)
- [Task 0017D1](0017d1-expand-tensor-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The selected capability baseline includes:

- `permute(int...)` for arbitrary complete axis reordering;
- rank-two parameterless `transpose()` as convenience for permutation `[1, 0]`;
- `expandDims(int)` for inserting one singleton dimension; and
- `squeeze(int)` for removing one selected singleton dimension.

The read-only `legacy/pre-rewrite` branch represents these as `PERMUTE`, `EXPAND_DIMS`, and
`SQUEEZE` operations. `transpose()` validates rank two and delegates to `permute(1, 0)`, so it does
not introduce a separate operation identity. Legacy operation descriptors retain normalized axes;
their builders own negative-axis normalization, input-rank checks, Shape/stride derivation,
storage views, and gradient inverses.

This task preserves only typed semantic vocabulary and immutable normalized parameters. It does
not copy legacy mutable arrays, graph builders, storage aliasing, gradient callbacks, operation
traits, compiler/lowering code, kernels, or runtime/backend behavior.

## Architecture constraints

- Production stays in `modules/model`, which owns backend-independent operation semantics.
- Semantic types store normalized intrinsic parameters only, never an input Tensor or Shape.
- `PERMUTE` reorders axes according to an output-to-input mapping but performs no Shape/layout
  calculation here.
- `EXPAND_DIMS` inserts one singleton axis at a normalized output position.
- `SQUEEZE` removes one selected singleton input axis; this semantic value cannot prove the
  eventual dimension is one.
- `transpose()` remains a public convenience to be built by task 0017F from PERMUTE semantics.
- Generic `Operation` stays an open non-null kind/attributes pair and does not enforce family
  pairings, arity, rank, Shape, layout, gradient, or backend rules.
- Package direction is `model.operation.layout -> model.operation + java.base` only.
- No tensor, shape, layout-value, storage, graph, compiler, planning, prepare, runtime, backend,
  engine, trace, or training dependency may be introduced.
- Stop if implementation needs input-dependent validation, another semantic type, dependency, or
  architecture change.

## Package impact

No package is added or moved.

- `io.github.pho001.synaptik.model.operation.layout.AxisTransformKind` — semantic vocabulary.
- `io.github.pho001.synaptik.model.operation.layout.PermutationAttrs` — immutable complete
  output-to-input axis permutation.
- `io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs` — normalized single-axis
  position shared by insertion/removal meanings.
- `AxisTransformSemanticsTest` — same-package focused structural and semantic test.

The existing layout-operation package already owns contiguous, reshape, and expand meanings; these
axis-coordinate transformations are cohesive with that ownership.

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

The enum declares no project field, constructor, method, nested type, alias, arity, axis, Shape,
layout, result metadata, or per-constant class body. Inherited enum names are diagnostic text only.

Document these meanings:

| Kind | One-input semantic meaning | Attributes |
|---|---|---|
| `PERMUTE` | output axis `i` corresponds to input axis `axes[i]`; logical values keep their coordinate association under reordered axes | `PermutationAttrs` |
| `EXPAND_DIMS` | insert one extent-one axis at the normalized output position while preserving logical values | `AxisTransformAttrs` |
| `SQUEEZE` | remove one selected extent-one input axis while preserving logical values | `AxisTransformAttrs` |

Do not add `TRANSPOSE`. A later rank-two convenience explicitly creates
`PERMUTE + PermutationAttrs(List.of(1, 0))`.

### Permutation attributes

Create exactly:

```java
public record PermutationAttrs(List<Integer> axes) implements OperationAttrs
```

It has exactly one component, one public canonical constructor, one explicit documented accessor,
and record-generated object methods. It adds no array overload, factory, inverse, rank accessor,
lookup method, nested type, or cached state.

Constructor validation order:

1. null-check `axes` with exact message `axes`;
2. create constructor-local `boolean[] seen` with length `axes.size()`;
3. inspect elements in ascending index order;
4. null-check each element with exact message `axes[<index>]`;
5. reject a negative value with exact message
   `axes[<index>] must be non-negative: <value>`;
6. reject a value at least permutation rank with exact message
   `axes[<index>] must be less than permutation rank <rank>: <value>`;
7. reject the first duplicate with exact message
   `axes contains duplicate axis <value> at index <index>`;
8. only after all validation, assign `axes = List.copyOf(axes)` exactly once.

Range plus uniqueness makes the list a complete permutation. Empty list is valid rank-zero
identity. Store no `seen` array or inverse. Preserve order and integer values; do not promise list
object identity.

`axes()` returns the immutable snapshot. Element `i` names the normalized input axis used as output
axis `i`.

### Single-axis attributes

Create exactly:

```java
public record AxisTransformAttrs(int axis) implements OperationAttrs
```

It has one component, one canonical constructor, one explicit documented accessor, and no other
state/API. Reject a negative axis with exact message:

```text
axis must be non-negative: <axis>
```

Retain every non-negative value unchanged. This record does not know a rank: task 0017F later
normalizes a raw axis and validates whether it is an insertion position or existing input axis.

### Typed composition

Document exactly these valid pairings:

```java
new Operation(AxisTransformKind.PERMUTE, permutationAttrs)
new Operation(AxisTransformKind.EXPAND_DIMS, axisTransformAttrs)
new Operation(AxisTransformKind.SQUEEZE, axisTransformAttrs)
```

Operation retains exact kind and attributes references. Do not use `NoOperationAttrs`, add a
compatibility validator/factory, or change Operation. Family pairings and one-input context are
documented, not generically enforced.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/AxisTransformKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/PermutationAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/AxisTransformAttrs.java`

Test:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/layout/AxisTransformSemanticsTest.java`

Documentation/planning:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without changing unless an inconsistency requires stopping:

- Compile API, Training API, capabilities, existing operation/layout contracts, architecture/ADRs,
  architecture/conformance/integration tests, Gradle, dependencies, and other modules.

## Maximum scope

At most three production files, one focused test, and five documentation/planning files: nine
paths total. If another Java type/test, public method, existing Java edit, dependency, build or
architecture change, or tenth path is required, stop and report. Do not create task 0017F.

## Javadoc requirements

- Document the enum, every constant, both records, constructors, components, and explicit
  accessors completely.
- Explain output-to-input permutation order with a concrete example such as `[1, 0, 2]`.
- Explain insertion versus removal axis position and why rank validation is deferred.
- Explain scalar empty permutation and immutable list ownership.
- Document exact validation order, parameters, results, null failures, and value failures.
- Explain transpose as future rank-two PERMUTE convenience, not a distinct kind.
- Explain that these values define no Tensor, Shape, layout, storage view, gradient, compiler,
  backend, or execution behavior.
- Independently review OperationKind, OperationAttrs, Operation, existing layout-operation
  semantics, and axis terminology; record why unchanged Javadocs remain accurate or stop on an
  out-of-scope discrepancy.

## Acceptance criteria

- Enum is public, implements OperationKind, and has exactly the three ordered constants.
- Enum adds no project field, method, constructor, nested type, metadata, or TRANSPOSE constant.
- PermutationAttrs is exactly one-component public record with deterministic validation and one
  immutable snapshot.
- Empty, identity, and reordered complete permutations are accepted; null, negative, out-of-range,
  and duplicate values fail with exact type/message/precedence.
- Caller list mutation cannot affect stored axes; accessor mutation fails.
- AxisTransformAttrs is exactly one-component public record accepting all non-negative values and
  rejecting negatives with exact message.
- Explicit typed pairings retain exact attribute references and no generic validator is added.
- No Tensor/Shape/layout/provenance/cross-layer behavior or existing Java contract changes.
- Complete Javadocs, Tensor API, glossary, independent documentation review, and synchronized
  planning evidence/status are finished before marking Complete.

## Tests / validation

Run before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.layout.AxisTransformSemanticsTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test must verify exact enum order/API shape; exact record components/accessors/API;
empty, identity, transpose, and general permutations; output-to-input order; list snapshot and
immutability; every validation failure and precedence; all non-negative single axes; record value
semantics; exact Operation composition/reference retention; absence of TRANSPOSE/extra members;
and no forbidden imports or dependencies.

Inspect `javap -p -c -s`, reflection, imports, and source. Confirm exactly three production types,
specified enum/record shapes, one constructor-local seen array, one List.copyOf, no retained helper
state, and no Tensor/Shape/layout/storage/graph/compiler/planning/prepare/runtime/backend/training
types. Validate generated Javadoc, API/glossary terminology, links, anchors, fences, whitespace,
exact nine-path scope, synchronized statuses, and absence of a task-0017F specification.

## Dependencies

- Task 0005 supplies OperationKind and OperationAttrs.
- Task 0006 supplies immutable generic Operation composition.
- Task 0002 supplies axis/Shape terminology for documentation but is not a production dependency.
- Tasks 0017C–0017D1 establish the existing layout-operation package and semantic/expression layer
  split without becoming production dependencies of these new types.

## Follow-up tasks

- 0017F remains Draft for public permute/transpose/expandDims/squeeze expressions, raw axis
  normalization, Shape and layout derivation, provenance, and descriptor construction.
- Compiler later owns identity/inverse-chain canonicalization and graph constraints.
- Planning/backend prepare later own materialization and concrete lowering.
- Training/compiler-generated semantics later own inverse-transform gradient expressions.

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. The model already owns backend-neutral operation semantics, and the new
types remain within its existing layout-operation package and dependency direction.

If implementation needs Tensor/Shape/layout state, another dependency, generic Operation changes,
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
  convenience over `PERMUTE [1, 0]`.
- Permutation order is explicitly output-to-input: `axes[i]` identifies the input axis occupying
  output axis `i`.
- PermutationAttrs validates completeness against its own list size, making the semantic value
  normalized and self-consistent without input Shape state.
- An empty axes list is the scalar identity permutation, consistent with rank-zero Shape support.
- AxisTransformAttrs is shared because both insertion and removal need one normalized non-negative
  axis; the kind supplies the different meaning and later Tensor construction supplies rank rules.
- Negative raw axes remain public request syntax for task 0017F and never enter semantic attrs.

## Known limitations

- These values cannot validate an eventual input rank, inserted-axis bound, squeezed singleton, or
  result Shape/layout.
- No public Tensor construction, transpose convenience, gradient inverse, compiler capture,
  canonicalization, materialization, backend lowering, runtime execution, or ONNX mapping exists.

## Validation evidence

Planning reviewed the architecture contract and focused lifecycle/module/dependency/runtime-
boundary explanations; documentation and planning rules; roadmap; model capabilities/master plan;
tasks 0002, 0005, 0006, 0017C, and 0017D1; current operation and layout-operation contracts/tests;
Tensor/Compile/Training APIs and glossary; and the Java 26 root/model Gradle configuration.

The read-only legacy branch confirmed three semantic identities, complete normalized permutation,
single-axis insertion/removal, transpose delegation to `[1, 0]`, view-oriented Shape/stride
behavior, and inverse gradient operations. Coupled legacy arrays, graph/storage/gradient builders,
traits, lowering, kernels, runtime, and backends were excluded.

Implementation and independent documentation validation:

- Implementation context `/root/implement_model_0017e` added exactly the three production types
  and focused semantic test. Clean documentation context
  `/root/implement_model_0017e/docs_review_0017e` independently read the required architecture,
  documentation and planning profiles, plans and predecessor tasks, APIs, glossary, final source
  and tests, generated Javadoc, XML reports, bytecode, build configuration, and complete live diff.
  It applied General and API/Javadoc style to Java, Tensor API, and glossary; Planning style to
  this task, the model master plan, and roadmap; and Example format to the output-to-input example.
- Independent source and generated-Javadoc review found all three submitted production Javadocs
  complete unchanged. They document exact kind/attributes pairings, output-to-input order with
  `[1, 0, 2]`, the empty scalar permutation, immutable snapshot ownership, validation order and
  exact failures, insertion versus removal position, deferred rank/singleton checks, transpose as
  future `PERMUTE [1, 0]`, and the Tensor/Shape/layout/storage/provenance/gradient/compiler/backend/
  execution boundaries.
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
- `javap -p -c -s` confirms exact ordered enum constants and compiler enum machinery only;
  `PermutationAttrs` has one `List` field, constructor-local `boolean[]`, indexed validation in the
  specified order, one `List.copyOf`, a direct accessor, and generated record value methods; and
  `AxisTransformAttrs` has one `int` field, one non-negative check, a direct accessor, and generated
  record value methods. Focused reflection tests independently verify the same API shape and exact
  Operation reference retention.
- Production imports are only `OperationKind` or `OperationAttrs` plus `List` and `Objects`.
  Source/bytecode review found no Tensor, Shape, layout value, storage, provenance, gradient,
  graph, compiler, planning, prepare, runtime, backend, engine, trace, or training dependency or
  behavior.
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
  and no detailed task-0017F specification exists. No commit or push was performed.
- `OperationKind`, `OperationAttrs`, and `Operation` remain accurate unchanged because the new
  values conform to their existing open typed contracts and generic Operation intentionally does
  not enforce pairings. Existing contiguous and target-shape operation semantics remain distinct
  and unchanged because they describe parameterless canonical geometry or exact target Shapes,
  not axis-coordinate transforms. Existing axis terminology remains accurate and is extended only
  to name the new normalized positions and permutation order.
- `capabilities.md` already inventories permute/transpose/expand-dimensions/squeeze capability and
  the layer split. Compile API remains accurate because no public expression, capture, inference,
  canonicalization, artifact, or materialization behavior was added. Training API remains accurate
  because no gradient, autograd, optimizer, parameter, publication, or session behavior changed.
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests, backend-
  conformance material, and integration tests remain accurate unchanged because module ownership,
  dependency direction, lifecycle, backend behavior, and end-to-end behavior did not change.
  Java 26 Gradle configuration, dependencies, other modules, and unrelated tests also remain
  unchanged because the task adds only model-owned semantic values.

## Implementation notes

- Added exact `PERMUTE`, `EXPAND_DIMS`, and `SQUEEZE` semantic identities in the existing
  operation-layout package with no extra project API or state.
- Added immutable complete output-to-input permutation attributes and normalized non-negative
  single-axis insertion/removal attributes with exact validation and ownership contracts.
- Added the focused eleven-test suite for enum/record shape, scalar/identity/transpose/general
  permutations, output-to-input order, snapshot immutability, every failure and precedence,
  representative axes, record semantics, exact Operation composition, and forbidden state.
- Finalized Tensor API, glossary, task evidence, model master plan, and roadmap in the mandatory
  independent documentation context; the submitted production Javadocs needed no correction.
- Added no Tensor method, Shape/layout/result/provenance behavior, raw-axis normalization,
  gradient, compiler/planning/prepare/runtime/backend behavior, dependency, build change, or
  architecture change.

## Completion summary

- Completed changes: Implemented and documented exact axis permutation, singleton-axis insertion,
  and selected singleton-axis removal semantic values.
- Files changed or created: Exactly three production Java files, one focused test, Tensor API,
  glossary, this task, model master plan, and roadmap.
- Tests and validation: Focused tests passed 11/11; all 508 model tests across 64 suites, generated
  model Javadoc, root tests, bytecode/reflection/import/source/generated-Javadoc checks, 246
  Markdown link/anchor checks, fence/whitespace/newline checks, exact-scope/status checks, and
  `git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0017e/docs_review_0017e` completed the independent pass using General,
  API/Javadoc, Planning, and Example profiles.
- Documentation impact: Tensor API and glossary present axis-transform semantic values as current
  while public Tensor methods and all input-dependent and cross-layer behavior remain planned.
- Javadoc review: All three new Javadocs are complete unchanged; related operation, layout-
  operation, capability, API, architecture, build, test, and cross-layer contracts remain accurate
  for the reasons recorded above.
- Glossary impact: Implementation status and reusable axis-transform, permutation, normalized-axis,
  kind, and attributes distinctions now include the completed semantics.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0017E. Task 0017F remains Draft without a detailed
  specification.

Status: Complete
