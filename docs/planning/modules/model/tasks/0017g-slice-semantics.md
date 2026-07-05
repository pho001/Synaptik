# Task 0017G: Slice Semantics

## Status

Complete

## Goal

Define the typed, backend-independent semantic identity and immutable normalized parameters for a
general positive-step tensor slice.

The general operation may constrain any ordered set of distinct input axes. A future single-axis
public convenience uses the same operation with one parameter entry; it is not a separate
semantic kind. This task defines meaning and parameter invariants only. Public request syntax,
negative index and axis normalization, defaults, Shape/layout derivation, provenance, gradients,
materialization, compiler behavior, and execution remain in later tasks and layers.

## Scope

- Add one public `SliceKind` enum implementing `OperationKind` with exactly `SLICE`.
- Add one public `SliceAttrs` record implementing `OperationAttrs` with exactly, in order,
  `List<Long> starts`, `List<Long> ends`, `List<Integer> axes`, and `List<Long> steps`.
- Define index `i` across the four parallel lists as half-open interval
  `[starts[i], ends[i])`, advanced by `steps[i]`, on normalized input axis `axes[i]`.
- Require equal list sizes, non-null elements, non-negative normalized starts/ends/axes, unique
  axes, and strictly positive steps.
- Accept empty lists as a normalized identity slice that constrains no axes.
- Accept every non-negative start/end relationship structurally; later Shape-aware construction
  decides result extent and empty-slice policy.
- Preserve entry order and store one immutable snapshot of each list after validation.
- Document exact `Operation` composition and future single-axis convenience as one step-one entry.
- Add one focused same-package semantic test for both production contracts.
- Keep production in the existing `model.operation.layout` package.
- Finalize Javadocs, Tensor API, glossary, task evidence, master plan, and roadmap through the
  mandatory independent documentation pass during implementation.

## Out of scope

- public `Tensor.slice`, `sliceAxis`, another Tensor method, overload, factory, builder, expression
  helper, or task-0017H implementation
- raw primitive-array requests, nullable/default axes or steps, implicit axes, implicit step one,
  ellipsis, omitted bounds, open-ended ranges, or a Python slice object
- normalizing negative axes, starts, or ends against an input Shape; clamping bounds; binding a
  dynamic dimension; or storing raw negative request values
- calculating output extents, deciding public empty-slice policy, or checking arithmetic overflow
- negative/zero steps, reverse slicing, repeated axes, axis sorting, canonicalization, or merging
  adjacent slices
- input Tensor, Shape, Dimension, DataType, TensorDescriptor, label, identity, provenance, host
  storage, gradient state, or `TensorFactory.createDerived`
- LayoutDescriptor, stride multiplication, storage offset, view/alias state, copy choice,
  contiguity, materialization, or physical storage access
- slice backward/scatter, autograd, graph capture, compiler, planning, prepare, runtime, backend,
  engine, trace, ONNX, training, or execution behavior
- factories, registries, parsers, visitors, maps, string dispatch, reflective discovery, arity,
  result, cost, fusion, backend-support, route, or kernel metadata
- changing existing Java/tests, dependencies, Gradle, architecture, another module, or later specs

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
- [Task 0017E](0017e-axis-transform-semantics.md)
- [Task 0017F1](0017f1-expand-dimensions-and-squeeze-tensor-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The baseline requires general `slice` and single-axis convenience. The read-only
`legacy/pre-rewrite` branch represents both with one `SLICE` operation:

```java
Tensor slice(int[] starts, int[] ends, int[] axes, int[] steps)
Tensor sliceAxis(int axis, int fromInclusive, int toExclusive)
```

The legacy builder accepts positive steps, normalizes negative axes/bounds against a concrete
input shape, clamps bounds, rejects duplicate axes, derives a strided view, and implements
`sliceAxis` with one axis and step one. Its operation stores parallel normalized arrays plus a
derived output shape.

The new model preserves those useful capabilities while using immutable lists, `long`
coordinates compatible with current Shape dimensions/layout strides, and no derived output Shape
inside semantic attributes. Task 0017H owns input-dependent normalization, clamping, output Shape,
layout geometry, provenance, and public methods. Legacy mutable arrays, immediate storage views,
graph builders, gradients, traits, lowering, kernels, and runtime/backend behavior are not copied.

## Architecture constraints

- Production remains in `modules/model`, which owns backend-neutral operation semantics.
- `SliceKind` identifies logical meaning only, not a Tensor, graph occurrence, descriptor,
  layout, materialization instruction, executable, or backend route.
- `SliceAttrs` stores normalized intrinsic parameters, never input-dependent facts or raw request
  syntax.
- List index `i` couples one start, end, axis, and step. Order is retained even though compiler
  canonicalization may later reorder distinct-axis constraints.
- Bounds are half-open. Starts/ends/axes are normalized non-negative values, axes are unique, and
  steps are strictly positive.
- The record performs no dimension lookup, clamping, extent calculation, or start/end comparison.
- Empty parallel lists are a valid identity description. Task 0017H decides whether a public
  no-entry request is exposed.
- Single-axis convenience has no separate kind or attributes type.
- Generic `Operation` remains an open kind/attributes pair and does not validate family pairing,
  arity, rank, Shape, layout, gradients, or backend support.
- Package direction is `model.operation.layout -> model.operation + java.base` only.
- Stop if implementation needs input Shape/Tensor state, another type, existing Java changes, a
  dependency, or an architecture decision.

## Package impact

No package is added or moved.

- `io.github.pho001.synaptik.model.operation.layout.SliceKind` — public semantic identity.
- `io.github.pho001.synaptik.model.operation.layout.SliceAttrs` — public immutable normalized
  parallel parameters.
- `SliceSemanticsTest` — same-package structural, validation, ownership, and composition test.

The existing layout-operation package owns cohesive layout/view semantic requests. No root
fallback, generic utility package, or migration is introduced.

## Required contract

### Semantic kind vocabulary

Create exactly:

```java
public enum SliceKind implements OperationKind {
    SLICE
}
```

The enum declares no project field, explicit constructor, method, nested type, per-constant body,
alias, arity, bounds, axes, steps, result, layout, or backend metadata. Inherited `Enum.name()`
satisfies `OperationKind.name()`.

Document one logical input and a same-rank result. Entry `i` selects input coordinates starting at
`starts[i]`, advancing by `steps[i]`, and remaining below `ends[i]` along `axes[i]`. Unlisted axes
retain their full logical coordinate range. This type calculates no Shape and creates no view.

Do not add `SLICE_AXIS`; future single-axis convenience is one `SLICE` entry with step one.

### Normalized slice attributes

Create exactly:

```java
public record SliceAttrs(
        List<Long> starts,
        List<Long> ends,
        List<Integer> axes,
        List<Long> steps) implements OperationAttrs
```

The record has exactly four components in that order, one canonical constructor, four explicit
documented accessors, and generated object methods. Add no array overload, factory, builder,
range/entry type, rank, size method, output Shape, layout, cache, nested type, or extra API/state.

Constructor validation order is exact:

1. null-check `starts`, `ends`, `axes`, and `steps`, in order, with component-name messages;
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

The record retains values/order but not caller list identity. Caller mutation cannot affect it and
accessor mutation fails. It stores no `HashSet` or derived state.

`Long.MAX_VALUE` is structurally valid for bounds/steps and `Integer.MAX_VALUE` for an axis because
no input rank exists here. A start may equal or exceed its end. This does not promise public API
acceptance; task 0017H owns Shape-aware bounds, extent, empty-result, and overflow policy.

Record equality/hashing use all four ordered lists. Generated text is diagnostic, not
serialization, request syntax, compiler canonical form, ONNX mapping, or backend dispatch.

### Typed composition and single-axis convenience

Document exact composition:

```java
SliceAttrs attrs = new SliceAttrs(starts, ends, axes, steps);
Operation operation = new Operation(SliceKind.SLICE, attrs);
```

Operation retains exact references. Do not use `NoOperationAttrs`, add a factory/validator, or
change `Operation`.

Document future single-axis convenience as semantically equivalent to:

```java
new SliceAttrs(
        List.of(fromInclusive),
        List.of(toExclusive),
        List.of(normalizedAxis),
        List.of(1L))
```

This assumes input-dependent normalization already occurred. Task 0017H owns actual signatures,
validation, raw negative syntax, and construction.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/SliceKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/SliceAttrs.java`

Test:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/layout/SliceSemanticsTest.java`

Documentation/planning:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless inconsistency requires stopping: Compile API, Training API,
capabilities, Operation foundations, Shape/Dimension, layout contracts, Tensor/graph contracts,
architecture/ADRs/tests, conformance/integration tests, Gradle, dependencies, and other modules.

## Maximum scope

At most two production files, one focused test, and five documentation/planning files: eight paths.
If a ninth path, another Java type/test, existing Java edit, dependency, build, or architecture
change is required, stop and report. Do not create task 0017H.

## Javadoc requirements

- Document enum/type/constant, record/constructor/components, and all explicit accessors.
- Explain parallel half-open semantics with Shape `[3, 6]`, starts `[0, 1]`, ends `[3, 6]`, axes
  `[0, 1]`, steps `[1, 2]`: selected columns are `1, 3, 5`, without claiming execution exists.
- Explain why coordinates use `long` and axes use `int`.
- Explain snapshots, order-sensitive value semantics, empty identity, and list ownership.
- Explain normalized attributes versus raw negative request syntax and deferred rank/bound checks.
- Explain why single-axis convenience is not another kind.
- Document validation order, parameters, results, null/value failures, and deferred behavior.
- Explain that no Tensor, Shape calculation, layout/view, storage, materialization, gradient,
  compiler, backend, ONNX, or execution behavior is defined.
- Review related contract Javadocs and record why they remain accurate or stop on discrepancy.

## Acceptance criteria

- Exact one-constant public enum and exact four-component public record with no extra API/state.
- Empty, one-axis, and multi-axis values are accepted and preserve values/order.
- Every specified invalid container, element, size, coordinate, axis, duplicate, and step fails
  with exact type/message/precedence.
- Four snapshots occur only after validation; caller/accessor mutation cannot alter state.
- Extreme values and start/end relationships remain Shape-independent.
- Exact Operation composition retains references; no generic validator is added.
- Single-axis is documented as one step-one entry; no `SLICE_AXIS` or Tensor method is added.
- No Shape/layout/provenance/storage/gradient/cross-layer behavior or existing Java edit appears.
- Javadocs, Tensor API, glossary, independent docs review, evidence, and statuses are complete.

## Tests / validation

Run before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.layout.SliceSemanticsTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test verifies exact enum/record/API shape; component order/types; empty, one-axis, and
general examples; validation failures/precedence; list snapshots/immutability; value semantics;
extreme values; unconstrained start/end relations; duplicate axes; exact Operation references;
step-one convenience; and no forbidden members/imports.

Inspect `javap -p -c -s`, reflection, imports, and source. Confirm two production types, one local
`HashSet`, four post-validation `List.copyOf` calls, no retained helper state, and no forbidden
layer types. Validate generated Javadoc, API/glossary terminology, links/anchors/fences/whitespace,
exact eight paths, synchronized status, and no task-0017H specification.

## Dependencies

- 0005 supplies `OperationKind` and `OperationAttrs`.
- 0006 supplies generic immutable `Operation` composition.
- 0002 supplies long-dimension/int-axis terminology but is not a production dependency.
- 0017E–0017F1 establish the package and semantic/expression split without production coupling.

## Follow-up tasks

- 0017H remains Draft for public requests, array ownership/defaults if selected, negative
  normalization, clamping, output Shape/LayoutDescriptor, provenance, and descriptor construction.
- Compiler later owns identity/slice-chain canonicalization and graph constraints.
- Compiler-generated/training semantics later own slice-backward scatter construction.
- Planning/backend prepare later own materialization and concrete lowering.

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. These model-owned semantic values stay in the existing package and
dependency direction. Stop if implementation needs input state, dependencies, generic Operation
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

- One `SLICE` covers general and single-axis forms because convenience changes no mathematics.
- Four immutable parallel lists preserve general capability without mutable primitive arrays.
- Bounds/steps use `long` to match Shape/layout geometry; axes remain `int`.
- Only normalized non-negative values enter attributes; raw negative syntax belongs to 0017H.
- Axes are unique but entry order is retained rather than sorted.
- Bounds are not compared here, keeping extent and zero-extent policy Shape-aware in 0017H.
- Empty lists are a structural identity; 0017H decides whether public API exposes them.

## Known limitations

- No raw normalization, rank validation, clamping, extent calculation, arithmetic proof, or view
  geometry exists.
- No Tensor method, gradient scatter, compiler behavior, materialization, lowering, execution, or
  ONNX mapping exists.

## Validation evidence

- Clean implementation context `/root/implement_model_0017g` added only the two production types
  and focused test specified by this task before handing the actual uncommitted diff to the
  independent documentation context.
- Clean documentation context `/root/implement_model_0017g/review_model_0017g_docs` applied the
  General, API/Javadoc, Planning, and Example profiles. It read the architecture and focused
  boundary documents, documentation/planning rules, prerequisite tasks, model plan and capability
  baseline, API/glossary sources, final production/test source, generated Javadoc, and actual diff.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.layout.SliceSemanticsTest` — `BUILD SUCCESSFUL`; the
  focused report contains 14 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; XML report aggregation contains 543 tests
  with zero failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`; generated pages contain the enum,
  constant, record components, canonical constructor, explicit accessors, parameters, results,
  failures, ownership, example, and cross-layer exclusions.
- `./gradlew test` — `BUILD SUCCESSFUL`; the root lifecycle reported 36 actionable tasks.
- `javap -p -c -s` plus reflection tests confirm one exact enum constant; one exact four-component
  record in starts/ends/axes/steps order; four private final list fields; one constructor-local
  `HashSet`; deterministic validation; exactly four post-validation `List.copyOf` calls; four
  direct accessors; generated record value methods; and no retained helper state or extra API.
- Source/import inspection confirms package direction only to `model.operation` and `java.base`.
  No Tensor, Shape, layout value, storage, graph, compiler, planning, prepare, runtime, backend,
  training, or ONNX type enters production state.
- The documentation pass retained the complete implementation-drafted Javadocs unchanged and
  finalized Tensor API and glossary coverage for parallel half-open semantics, the conceptual
  Shape `[3, 6]` example selecting columns `1`, `3`, `5`, long coordinates/int axes, immutable
  snapshots, order-sensitive value semantics, empty identity, raw-versus-normalized boundaries,
  exact `Operation` composition, one-entry step-one convenience, and all deferred behavior.
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
  layers, so no update is needed.
- Existing `OperationKind`, `OperationAttrs`, `Operation`, Shape/Dimension, LayoutDescriptor,
  Tensor/layout-operation/graph Javadocs and contracts remain accurate because the new family
  composes them without changing their signatures, invariants, or ownership. No architecture
  document, ADR, architecture test, backend-conformance test, integration test, Gradle file,
  dependency, existing unrelated Java file, other module, or later task specification changed
  because no boundary, dependency, backend, executable, or end-to-end behavior changed.
- `git diff --check` passed with no whitespace errors after final documentation and planning
  synchronization.

## Implementation notes

- Added `SliceKind` with exactly `SLICE` and `SliceAttrs` with exactly the ordered starts, ends,
  axes, and steps list components in the existing layout-operation package.
- `SliceAttrs` validates containers, pairing, indexed elements, normalized values, distinct axes,
  and positive steps in the specified precedence, then snapshots all four lists. Empty lists,
  extreme non-negative values, and every non-negative start/end relationship remain valid.
- Added one focused same-package test covering exact surface shape, validation messages and
  precedence, immutability, value semantics, identity and general examples, exact Operation
  composition, step-one convenience, and forbidden-state absence.
- Tensor API and glossary now describe the implemented semantic contract and conceptual parallel
  example while explicitly retaining public Tensor construction and every Shape/layout/
  provenance/cross-layer behavior for task 0017H or later owning layers.
- Synchronized this task, model master plan, and roadmap at Complete without creating the 0017H
  specification.

## Completion summary

- Completed changes: Added the exact backend-neutral positive-step slice kind and immutable
  normalized parallel attributes, focused semantic tests, complete API/glossary documentation,
  and synchronized planning evidence.
- Files changed or created: Exactly the eight paths listed under Affected files.
- Tests and validation: Focused 14-test suite, 543-test model suite, model Javadoc, root test
  lifecycle, bytecode/reflection/import/source checks, generated-Javadoc review, 319 local links
  and anchors, fences/whitespace/newlines, exact scope/status checks, and `git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0017g/review_model_0017g_docs` independently reviewed final source,
  tests, generated Javadoc, related contracts, and the actual diff under the required profiles.
- Documentation impact: Tensor API and glossary now define current slice semantics and the
  `[3, 6]` parallel half-open example while keeping public construction in Draft task 0017H.
- Javadoc review: SliceKind/SliceAttrs type, member, constructor, component, accessor, ownership,
  validation, example, and exclusion contracts are complete and accurate; related Javadocs remain
  accurate unchanged.
- Glossary impact: Added the reusable Slice term and updated normalized-axis, OperationAttrs,
  OperationKind, status, and kind/attributes/operation distinctions.
- Architecture impact: None.
- Unresolved issues: None within task scope; known limitations are intentional layer boundaries.
- Follow-up required: None for task 0017G. Task 0017H remains Draft without a specification.

Status: Complete
