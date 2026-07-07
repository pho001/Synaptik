# Task 0017M: Unfold and Fold Semantics

## Status

Complete

## Goal

Define typed, backend-independent semantic identities and immutable normalized parameters for
single-axis sliding-window unfold and its scatter-add fold, rank-four NCHW two-dimensional unfold,
and rank-three-column to rank-four NCHW fold.

Single-axis unfold replaces one source-axis extent with a window-position extent and appends the
window extent as the last result axis. Single-axis fold removes that final window axis and
scatter-adds its windows into an explicit target extent. Two-dimensional unfold materializes NCHW
image windows as canonical im2col columns. Two-dimensional fold performs the inverse-shaped
col2im accumulation into an explicit NCHW output shape; overlapping contributions are summed
rather than averaged.

This task defines meaning and intrinsic parameters only. Public Tensor methods, request
normalization, input/result validation, Shape arithmetic, descriptors, provenance, gradients,
materialization, compiler behavior, lowering, kernels, and numerical execution remain later
responsibilities.

## Scope

- Add one public `WindowTransformKind` enum implementing `OperationKind` with exactly
  `UNFOLD_AXIS`, `FOLD_AXIS`, `UNFOLD2D`, and `FOLD2D`, in that order.
- Add one public `UnfoldAxisAttrs` record implementing `OperationAttrs` with exactly normalized
  non-negative `int axis`, positive `long size`, and positive `long step`, in that order.
- Add one public `FoldAxisAttrs` record implementing `OperationAttrs` with exactly normalized
  non-negative `int axis`, non-negative `long outputSize`, and positive `long step`, in that order.
- Add one public `Window2dAttrs` record implementing `OperationAttrs` with exact long-valued
  kernel, stride, symmetric-padding, and dilation geometry plus `boolean ceilMode`.
- Add one public `Fold2dAttrs` record implementing `OperationAttrs` with exactly `Shape outputShape`
  and `Window2dAttrs window`, in that order.
- Define `UNFOLD_AXIS` as general no-padding, no-dilation one-axis window materialization.
- Define `FOLD_AXIS` as the shape-adjoint scatter-add of general-axis unfold, using the input's
  final dimension as window size and an explicit target extent for the restored axis.
- Define `UNFOLD2D` as rank-four NCHW to rank-three canonical im2col semantics parameterized by
  `Window2dAttrs`.
- Define `FOLD2D` as rank-three canonical columns accumulated into the explicit output Shape using
  the same two-dimensional window geometry.
- Document exact kind/attribute pairings without adding a generic compatibility validator.
- Add one focused same-package semantic test for all five production contracts.
- Keep production in the existing `model.operation.layout` package.
- Finalize Javadocs, Tensor API semantic reference, capability baseline, glossary, task evidence,
  master plan, and roadmap through the mandatory independent documentation pass.

## Out of scope

- public `Tensor.unfold`, `Tensor.foldAxis`, `Tensor.unfold2d`, `Tensor.fold2d`, static facade
  methods, overloads, convenience defaults, expression helpers, factories, or task-0017N
  implementation; task 0017N is required to plan all four public expressions
- input Tensor, input Shape, input/result DataType, rank validation, axis normalization from a raw
  negative request, result Shape derivation, window-count calculation, compatibility validation,
  or arithmetic overflow checks
- validating that unfold size fits a selected source dimension, that a two-dimensional effective
  kernel fits padded input, or that fold columns match output batch/channels/window count
- asymmetric padding, per-side padding, padding modes other than zero for unfold, negative or zero
  kernel/stride/dilation, negative padding, reverse windows, adaptive windows, or multiple general
  unfold axes
- treating im2col or col2im as additional operation identities; they are explanatory aliases for
  `UNFOLD2D` and `FOLD2D`
- storing input/output rank, inferred output spatial sizes, effective kernel, column count,
  DataType, descriptor, layout, Tensor, provenance, label, storage, gradient, or backend facts
- view/alias semantics, strides, offsets, resolved LayoutDescriptor, allocation, copying,
  materialization, zero filling, overlap buffers, division by overlap count, or value execution
- gradient-rule construction, scatter-add value execution, fold backward construction, autograd
  traversal, compiler insertion of `FOLD_AXIS`, optimizer behavior, or training execution
- graph capture, canonicalization, decomposition, fusion, planning, prepare, runtime, backend
  lowering/kernels, engine, trace, ONNX mapping, or conformance behavior
- factories, registries, visitors, parsers, maps, string dispatch, reflection discovery, arity,
  result-kind, cost, fusion, backend-support, route, or kernel metadata
- changing existing Java/tests, dependencies, Gradle, architecture, another module, or creating a
  task-0017N specification

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
- [Task 0017G](0017g-slice-semantics.md)
- [Task 0017I](0017i-pad-and-tile-semantics.md)
- [Task 0017K](0017k-tensor-composition-semantics.md)
- [Task 0017L](0017l-tensor-composition-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The selected baseline requires all three legacy public window transformations:

```java
Tensor unfold(int axis, int size, int step)
Tensor unfold2d(Window2dOptions options)
Tensor fold2d(int[] outputShape, Window2dOptions options)
```

The new public capability planned for task 0017N is:

```java
Tensor foldAxis(int axis, long outputSize, long step)
```

The raw axis may use the same negative-index normalization convention as other public axis APIs;
the normalized semantic attributes remain non-negative. The final input dimension supplies window
size, while outputSize explicitly selects the restored target extent.

The read-only `legacy/pre-rewrite` branch represents them as distinct `UNFOLD_AXIS`, `UNFOLD2D`,
and `FOLD2D` operation meanings. Its two-dimensional window value carries kernel height/width,
stride height/width, symmetric padding height/width, dilation height/width, and ceil mode.

Legacy has no public general `foldAxis` operation. Its `UNFOLD_AXIS` gradient nevertheless performs
the same adjoint behavior by slicing every window, padding it back to the selected input position,
and adding overlapping contributions. The new semantic vocabulary names that concrete requirement
as `FOLD_AXIS` so later compiler-generated autograd and backend work can represent it directly
without expanding it into a large pad/add graph. Unlike the legacy surface, the new capability
also deliberately exposes this symmetric operation through public `Tensor.foldAxis(...)` in task
0017N. Task 0023 separately owns later compiler-generated use of the same semantic identity.

Legacy single-axis unfold accepts every supported data type and appends the window dimension.
Legacy two-dimensional unfold/fold accepts floating data, uses NCHW image geometry and canonical
rank-three columns, and uses the formulas and accumulation behavior documented below. Those facts
are capability evidence. This task does not copy legacy classes, mutable arrays, graph builders,
gradient callbacks, shape-rule helpers, materializers, lowering metadata, kernels, or runtime
coupling.

The new semantic values use `long` geometry because current `Shape` dimensions are long-valued.
Task 0017N will own public syntax, raw-axis normalization, static/dynamic Shape policy, checked
arithmetic, result descriptors, provenance, and gradient eligibility.

## Architecture constraints

- Production remains in `modules/model`, which owns backend-neutral operation semantics and Shape.
- `WindowTransformKind` describes mathematical/layout-adjacent meaning only, not an occurrence,
  graph node, Tensor, executable, materialization plan, kernel, or backend route.
- `UnfoldAxisAttrs.axis` is already normalized and non-negative. The record stores no rank and
  cannot prove that the axis exists or that the window fits.
- `FoldAxisAttrs.axis` is already normalized in the target rank. Its `outputSize` disambiguates the
  restored target extent; its input window size is read later from the final input dimension and
  is deliberately not duplicated in the attributes.
- `Window2dAttrs` is intrinsic symmetric two-dimensional window geometry. It contains no input or
  output Shape and performs no formula evaluation.
- `Fold2dAttrs.outputShape` is the explicit normalized logical result requested by fold. The record
  stores the exact immutable Shape reference and does not validate rank or compatibility.
- `Fold2dAttrs.window` reuses the exact immutable `Window2dAttrs` reference so unfold/fold geometry
  has one value contract rather than duplicated scalar fields.
- `UNFOLD_AXIS` pairs only with `UnfoldAxisAttrs`; `FOLD_AXIS` pairs only with `FoldAxisAttrs`;
  `UNFOLD2D` pairs only with `Window2dAttrs`; `FOLD2D` pairs only with `Fold2dAttrs`.
- Generic `Operation` remains an open kind/attributes pair and does not enforce these pairings,
  arity, rank, Shape, data type, gradients, materialization, or backend support.
- Package direction is `model.operation.layout -> model.operation + model.shape + java.base` only.
- Stop if implementation requires Tensor/provenance/layout changes, another type, another test,
  dependency, architecture change, or cross-layer behavior.

## Package impact

No package is added or moved.

- `io.github.pho001.synaptik.model.operation.layout.WindowTransformKind` — exact semantic identities
  for general-axis unfold, NCHW unfold2d, and NCHW fold2d.
- `io.github.pho001.synaptik.model.operation.layout.UnfoldAxisAttrs` — normalized one-axis window
  parameters.
- `io.github.pho001.synaptik.model.operation.layout.FoldAxisAttrs` — normalized target axis,
  restored extent, and window step for scatter-add folding.
- `io.github.pho001.synaptik.model.operation.layout.Window2dAttrs` — reusable immutable symmetric
  NCHW window geometry.
- `io.github.pho001.synaptik.model.operation.layout.Fold2dAttrs` — explicit fold result Shape plus
  its two-dimensional window geometry.
- `WindowTransformSemanticsTest` — same-package structural, validation, ownership, and typed-pairing
  test.

Window transforms remain cohesive with existing reshape, axis-transform, slice, pad/tile, and
composition semantics in `model.operation.layout`. The Shape import in `Fold2dAttrs` expresses a
logical result contract owned by model; it introduces no compiler, runtime, or backend dependency.

## Required contract

### Semantic kind vocabulary

Create exactly:

```java
public enum WindowTransformKind implements OperationKind {
    UNFOLD_AXIS,
    FOLD_AXIS,
    UNFOLD2D,
    FOLD2D
}
```

The enum adds no project field, explicit constructor, method, nested type, per-constant body,
alias, arity, geometry, Shape, DataType, result, layout, materialization, cost, or backend metadata.

Document exact meanings:

| Kind | Logical meaning | Attributes |
|---|---|---|
| `UNFOLD_AXIS` | one input; sliding windows along one normalized axis; selected extent becomes window-position count and window size is appended as the final axis | `UnfoldAxisAttrs` |
| `FOLD_AXIS` | one window tensor; final window dimension is scatter-added back along one normalized target axis with an explicit restored extent | `FoldAxisAttrs` |
| `UNFOLD2D` | one rank-four NCHW input; materialized rank-three `[N, C * kernelHeight * kernelWidth, outputHeight * outputWidth]` columns | `Window2dAttrs` |
| `FOLD2D` | one rank-three canonical column input; contributions accumulated into one explicit rank-four NCHW result Shape | `Fold2dAttrs` |

The enum does not validate or calculate any of these Shapes and does not execute values.

### Single-axis unfold attributes

Create exactly:

```java
public record UnfoldAxisAttrs(int axis, long size, long step) implements OperationAttrs
```

The record has exactly three components in that order, one public canonical constructor, three
explicit documented accessors, and record-generated object methods. Add no rank, input extent,
window count, result Shape, raw axis, factory, builder, overload, nested type, or extra state/API.

Constructor validation order is exact:

1. reject negative `axis` with `IllegalArgumentException` and exact message
   `axis must be non-negative: <axis>`;
2. reject zero or negative `size` with exact message `size must be positive: <size>`;
3. reject zero or negative `step` with exact message `step must be positive: <step>`.

Retain every valid value unchanged, including `Integer.MAX_VALUE` axis and `Long.MAX_VALUE` size or
step. Rank, axis bounds, selected-dimension staticity, size-fit checks, window-count arithmetic,
and overflow remain task-0017N responsibilities.

For static selected extent `D`, later result construction uses the floor-mode count
`floor((D - size) / step) + 1` after proving `size <= D`. The selected input axis is replaced by
that count and a new final axis of extent `size` is appended. This explanatory formula is not
implemented in this record.

### Single-axis fold attributes

Create exactly:

```java
public record FoldAxisAttrs(int axis, long outputSize, long step) implements OperationAttrs
```

The record has exactly three components in that order, one public canonical constructor, three
explicit documented accessors, and record-generated object methods. Add no rank, input Shape,
window size, window count, result Shape, factory, builder, overload, nested type, or extra
state/API.

Constructor validation order is exact:

1. reject negative `axis` with `IllegalArgumentException` and exact message
   `axis must be non-negative: <axis>`;
2. reject negative `outputSize` with exact message
   `outputSize must be non-negative: <outputSize>`;
3. reject zero or negative `step` with exact message `step must be positive: <step>`.

Retain every valid value unchanged, including zero outputSize, `Integer.MAX_VALUE` axis, and
`Long.MAX_VALUE` outputSize or step. Zero is structurally consistent with the current Shape model;
later construction decides whether a particular input window tensor is compatible. Rank, axis
bounds, input rank, final-window-dimension size, number-of-windows compatibility, uncovered target
positions, and arithmetic overflow remain deferred.

`outputSize` is required because window count, window size, and step do not uniquely determine the
original source extent when trailing positions were not covered. Later construction interprets
the input's final dimension as the window size, removes that dimension, restores the selected axis
to `outputSize`, and scatter-adds every window element at `windowIndex * step + offset`. Overlaps
sum; valid target positions receiving no contribution remain zero. The attributes perform none of
that Shape or value behavior.

### Shared two-dimensional window attributes

Create exactly:

```java
public record Window2dAttrs(
        long kernelHeight,
        long kernelWidth,
        long strideHeight,
        long strideWidth,
        long paddingHeight,
        long paddingWidth,
        long dilationHeight,
        long dilationWidth,
        boolean ceilMode) implements OperationAttrs
```

The record has exactly nine components in that order, one public canonical constructor, nine
explicit documented accessors, and record-generated object methods. Add no asymmetric sides,
input/output Shape, effective-kernel cache, output-size method, defaults, factory, fluent modifier,
builder, padding mode, DataType, nested type, or extra state/API.

Constructor validation order and exact failures are:

1. `kernelHeight` must be positive: `kernelHeight must be positive: <value>`;
2. `kernelWidth` must be positive: `kernelWidth must be positive: <value>`;
3. `strideHeight` must be positive: `strideHeight must be positive: <value>`;
4. `strideWidth` must be positive: `strideWidth must be positive: <value>`;
5. `paddingHeight` must be non-negative:
   `paddingHeight must be non-negative: <value>`;
6. `paddingWidth` must be non-negative: `paddingWidth must be non-negative: <value>`;
7. `dilationHeight` must be positive: `dilationHeight must be positive: <value>`;
8. `dilationWidth` must be positive: `dilationWidth must be positive: <value>`.

Retain all valid long values and `ceilMode` unchanged, including zero padding and
`Long.MAX_VALUE`. Perform no checked multiplication/addition or Shape validation here.

Document the later static-Shape formulas without implementing them:

```text
effectiveKernel = dilation * (kernel - 1) + 1
numerator       = input + 2 * padding - effectiveKernel
output          = floor(numerator / stride) + 1       when ceilMode is false
output          = ceil(numerator / stride) + 1        when ceilMode is true
```

Task 0017N must use checked long arithmetic, require the effective kernel to fit the padded
dimension, and define the resulting descriptor. Padding positions sampled outside the source are
conceptual zeros for `UNFOLD2D`. This task stores semantics only and performs no sampling.

### Fold attributes

Create exactly:

```java
public record Fold2dAttrs(Shape outputShape, Window2dAttrs window)
        implements OperationAttrs
```

The record has exactly two components in that order, one public canonical constructor, two
explicit documented accessors, and record-generated object methods. Validation order is exact:

1. null-check `outputShape` with `Objects.requireNonNull` and exact message `outputShape`;
2. null-check `window` with `Objects.requireNonNull` and exact message `window`.

Retain and return the exact immutable references. Accept every current Shape category, including
scalar, non-rank-four, zero-extent, and dynamic Shapes, at this structural layer. Task 0017N must
decide the supported public fold boundary and prove rank-four NCHW/static compatibility before
construction. Do not copy, reconstruct, canonicalize, or compare the Shape against window
geometry in this record.

`FOLD2D` interprets its rank-three input as canonical columns. Multiple column entries targeting
the same output coordinate are added. Uncovered output positions remain zero. No overlap-count
division is performed. These are semantic statements, not model-level value execution.

### Typed composition

Document exactly these valid pairings:

```java
Operation axisUnfold =
        new Operation(WindowTransformKind.UNFOLD_AXIS, unfoldAxisAttrs);
Operation axisFold =
        new Operation(WindowTransformKind.FOLD_AXIS, foldAxisAttrs);
Operation imageUnfold =
        new Operation(WindowTransformKind.UNFOLD2D, window2dAttrs);
Operation imageFold =
        new Operation(WindowTransformKind.FOLD2D, fold2dAttrs);
```

`Operation` retains exact kind and attribute references. Do not use `NoOperationAttrs`, pair a
kind with another window-family attribute, add a compatibility validator/factory/registry, or
modify `Operation`.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/WindowTransformKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/UnfoldAxisAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/FoldAxisAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/Window2dAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/Fold2dAttrs.java`

Test:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/layout/WindowTransformSemanticsTest.java`

Documentation/planning:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless an inconsistency requires stopping: Compile API, Training API,
Shape/Operation/Tensor/TensorProvenance/graph/layout contracts, existing operation
families/tests, architecture/ADRs/tests, conformance/integration tests, Gradle, dependencies, and
other modules.

## Maximum scope

At most five production files, one focused test, and six documentation/planning files: twelve
paths. The twelve-path scope is deliberate because `FOLD_AXIS` is a cohesive adjoint semantic that
requires its own typed attributes and adds an intentional public capability to `capabilities.md`.
If another Java type/test, existing Java edit, dependency, build/architecture change, or thirteenth
path is required, stop and report. Do not create task 0017N.

## Javadoc requirements

- Document the enum type, all four enum constants, every record type/component, canonical
  constructor, and every explicit accessor.
- Explain `UNFOLD_AXIS` with input Shape `[2, 5, 3]`, axis 1, size 3, step 1, producing conceptual
  Shape `[2, 3, 3, 3]`: the selected extent becomes three window positions and the appended final
  extent is the window size.
- Explain that general-axis unfold has no padding/dilation/image assumptions and materializes
  windows rather than promising a view.
- Explain `FOLD_AXIS` as the scatter-add adjoint with unfolded Shape `[3, 3]`, axis 0,
  `outputSize=5`, and step 1, producing conceptual Shape `[5]`; use concrete window values to show
  overlap summation and explain why outputSize cannot always be inferred.
- Explain that `FOLD_AXIS` supports both the public task-0017N Tensor expression and later
  compiler-generated autograd/backend work, with one shared semantic identity.
- Explain `UNFOLD2D` with NCHW Shape `[1, 1, 3, 3]`, kernel 2x2, unit stride, zero padding, unit
  dilation, floor mode, producing conceptual columns Shape `[1, 4, 4]`.
- Explain `FOLD2D` for those columns and explicit output Shape `[1, 1, 3, 3]`, including that the
  center receives four contributions while corners receive one; no overlap averaging occurs.
- Define NCHW, im2col, col2im, effective kernel, symmetric padding, dilation, stride, floor mode,
  and ceil mode at first use or link to glossary entries added by the documentation pass.
- Include the output-size formulas, units, checked-arithmetic deferral, and behavior of conceptual
  zero-padding positions.
- Explain normalized axis versus raw negative public syntax and why rank/bounds are deferred.
- Explain `Fold2dAttrs` exact reference retention and why structural construction accepts Shapes
  that task 0017N will reject at the public expression boundary.
- Explain no Tensor construction, Shape calculation, values, storage, resolved layout, gradients,
  graph/compiler/planning/prepare/runtime/backend/ONNX behavior, or execution.
- Review Shape, Operation, existing layout semantic, TensorProvenance, and Tensor terminology
  Javadocs; record unchanged reasons or stop on discrepancy.

## Acceptance criteria

- `WindowTransformKind` is the exact public four-constant ordered vocabulary with no extra project
  state/API.
- `UnfoldAxisAttrs` has exactly the specified record shape, validation order/messages, explicit
  accessors, generated value behavior, and accepts valid extreme values.
- `FoldAxisAttrs` has exactly the specified record shape, validation order/messages, explicit
  accessors, generated value behavior, and duplicates no window-size/input-shape state.
- `Window2dAttrs` has exactly nine components in order, exact validation/messages, explicit
  accessors, generated value behavior, and performs no arithmetic or Shape validation.
- `Fold2dAttrs` has exactly two components in order, exact null validation, exact reference
  retention, explicit accessors, and no rank/compatibility validation.
- Exact valid kind/attribute pairings compose through unchanged `Operation` and retain references.
- Focused tests verify enum order/surface, record components/order/types, fields/constructors/
  methods, valid values, every failure type/message/precedence, reference retention, generated
  equality/hash/toString, typed composition, and absence of extra API/state.
- Tests prove no production array/list/map/registry, Tensor, provenance, layout, compiler, runtime,
  backend, execution, or reflection-discovery state was introduced.
- Complete Javadocs explain all public contracts, formulas, examples, deferred checks, parameters,
  results, and failures without claiming execution.
- Tensor API, capability baseline, and glossary describe the four semantic values as current and
  keep all four public unfold/fold Tensor expressions planned for 0017N. They distinguish public
  `foldAxis` construction from later task-0023 compiler-generated use of the same semantic kind.
- Compile API, Training API, focused architecture docs, and existing component
  Javadocs receive reasoned no-change conclusions unless an inconsistency requires stopping.
- Task, master plan, and roadmap status/evidence are synchronized only after all validation passes.
- A separate clean-context documentation agent finalizes Javadocs, Tensor API, glossary, planning
  evidence, terminology, links, examples, and no-change conclusions in the same overall change.
- The final diff contains exactly the twelve permitted paths, `git diff --check` passes, task
  0017M is Complete, task 0017N remains Draft, and no task-0017N specification exists.

## Tests / validation

Run implementation validation:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.layout.WindowTransformSemanticsTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Inspect compiled contracts:

```bash
javap -classpath modules/model/build/classes/java/main \
  io.github.pho001.synaptik.model.operation.layout.WindowTransformKind \
  io.github.pho001.synaptik.model.operation.layout.UnfoldAxisAttrs \
  io.github.pho001.synaptik.model.operation.layout.FoldAxisAttrs \
  io.github.pho001.synaptik.model.operation.layout.Window2dAttrs \
  io.github.pho001.synaptik.model.operation.layout.Fold2dAttrs
```

Manual validation must confirm:

- exactly four enum constants in order and no project members beyond compiler-generated enum API;
- exact record component order/types, one canonical constructor per record, explicit accessors,
  no additional public methods, and no extra instance/static project fields;
- exact validation order, exception types, messages, valid extrema, generated object methods, and
  Fold2d reference identity;
- exact imports and no Tensor, descriptor, layout, provenance, graph, compiler, planning, runtime,
  backend, service, registry, map, reflection-discovery, dependency, or build coupling;
- generated Javadoc contains types, constants, components, constructors, accessors, examples,
  formulas, `@param`, `@return`, and `@throws` details required above;
- every local Markdown link/anchor resolves, code fences are balanced, files end with newlines,
  and changed/untracked files have no trailing whitespace;
- final inventory is exactly twelve paths, package placement matches this plan, 0017M is Complete,
  0017N is Draft, and no `0017n-*.md` exists.

The documentation agent must rerun the focused test, model test, model Javadoc, root test,
documentation/link/format checks, inventory/status checks, and `git diff --check` against the final
tree. Record exact commands, results, test counts when available, its agent/thread identity, and
reasoned no-change conclusions in this task before completion.

## Dependencies

- Task 0002: Shape and dimension model
- Task 0005: Operation semantic foundation
- Task 0006: Operation model
- Task 0017I: Pad and tile semantics for related long-valued geometry conventions
- Task 0017K: Tensor composition semantics for current layout-family kind/attribute conventions
- Task 0017L: completed current public layout-expression frontier and established 0017M as next

## Follow-up tasks

- Task 0017N: public `unfold`, `foldAxis`, `unfold2d`, and `fold2d` Tensor expressions — required
  next Draft frontier; create its detailed specification only after task 0017M is Complete.
- Task 0023: authorize compiler-generated `FOLD_AXIS` construction for autograd while reusing the
  task-0017M semantic contract rather than defining another fold identity.
- Later planning, prepare, backend, ONNX, and execution tasks may consume these semantic values but
  are not authorized here.

## Architecture impact

Expected impact: None.

This task adds model-owned backend-neutral semantic vocabulary only. It does not alter module
ownership, dependency direction, compile/prepare/run separation, backend-owned lowering, runtime
boundaries, or the architecture contract. If implementation requires such a change, stop and
report the issue.

## Implementation prompt

Use this prompt in a separate agentic task/thread with clean context:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0002/0005/0006/0017I/0017K/0017L/0017M, Tensor API,
Compile API, Training API, glossary, current OperationKind/OperationAttrs/Operation/Shape and
layout-operation semantic contracts/tests, and Java 26 Gradle configuration.

Implement task 0017M exactly. Add only WindowTransformKind.java, UnfoldAxisAttrs.java,
FoldAxisAttrs.java, Window2dAttrs.java, Fold2dAttrs.java, and WindowTransformSemanticsTest.java for
Java code/tests under io.github.pho001.synaptik.model.operation.layout.

The enum contains exactly UNFOLD_AXIS, FOLD_AXIS, UNFOLD2D, FOLD2D. UnfoldAxisAttrs contains
exactly normalized non-negative int axis plus positive long size and step. FoldAxisAttrs contains
exactly normalized non-negative int axis plus non-negative long outputSize and positive long step,
takes window size from the eventual input's final dimension, and names the scatter-add operation
that task 0017N will expose publicly and task 0023 may generate for autograd. Window2dAttrs contains
exactly the nine ordered long/boolean kernel, stride, symmetric-padding, dilation, and ceil-mode
components with exact
validation order/messages and no arithmetic. Fold2dAttrs contains exactly Shape outputShape and
Window2dAttrs window, null-checks in order, and retains exact references. Document exact typed
pairings, general-axis unfold/fold, NCHW im2col, and overlap-summing col2im.

Do not add Tensor methods, Shape/result/data-type validation, formulas as executable helpers,
layouts/provenance/gradients, value behavior, compiler/planning/prepare/runtime/backend/ONNX work,
factories/registries, dependencies, build/architecture changes, existing Java edits, or later
specs. Stop beyond twelve paths or on architecture uncertainty.

Run all specified focused/aggregate tests, Javadoc, javap/reflection/import/manual,
documentation/link/whitespace/scope/status checks. Then hand the actual diff/evidence to a separate
clean-context documentation agent in the same change. It must inspect source/tests/generated
Javadoc, finalize permitted Javadocs/Tensor API/glossary/planning, record related-contract,
Compile API, Training API, and architecture no-change conclusions, and rerun validation. It must
retain the intentional public `foldAxis` addition in capabilities.md and distinguish it from
task-0023 compiler generation.

Update task 0017M, model master plan, and roadmap only for planning status/evidence. Do not mark
0017M Complete until both passes succeed. Leave 0017N Draft without a specification. Do not commit
or push.
```

## Local decisions

- The four operations remain one cohesive window-transform family because the paired attributes
  distinguish general-axis and NCHW geometry without a registry or generic family validator.
- `FOLD_AXIS` is one shared semantic identity for both the public expression planned by task
  0017N and compiler-generated unfold-adjoint use planned by task 0023. The attributes do not
  encode which consumer constructed it.
- General-axis fold reads window size from the eventual input's final dimension rather than
  duplicating it in `FoldAxisAttrs`; explicit `outputSize` preserves trailing uncovered extent.
- `Window2dAttrs` is shared exact geometry. `Fold2dAttrs` nests its exact reference with an exact
  immutable output Shape reference rather than copying fields or validating input-dependent
  compatibility structurally.
- Formula text remains documentation only. Task 0017N owns checked arithmetic, Shape/rank/axis
  validation, compatibility, descriptors, provenance, and public request normalization.

## Known limitations

- No public unfold/fold Tensor expression, result Shape or data-type rule, provenance, gradient,
  compiler capture or autograd construction, materialization, lowering, backend/ONNX mapping, or
  execution is implemented by this task.
- Structurally valid extreme axes and geometry can be incompatible with a future input. The
  semantic records intentionally lack the input context required to reject those combinations.
- `Fold2dAttrs` accepts every current Shape category; task 0017N must establish the narrower public
  rank-four NCHW/static boundary and column compatibility.

## Validation evidence

- Clean implementation context `/root/implement_model_0017m` added exactly the five production
  contracts and one focused test specified by this task before handing the shared uncommitted tree
  to independent documentation context `/root/implement_model_0017m/docs_0017m`.
- The documentation context applied General, API/Javadoc, Planning, and Example profiles. It read
  the architecture and focused boundary documents, documentation/planning rules, capability
  baseline, prerequisite and neighboring tasks, Tensor/Compile/Training APIs, glossary, final
  source/test, generated Javadoc, related Operation/Shape/Tensor/provenance/layout contracts, Java
  26 build configuration, and the actual diff.
- The documentation pass retained the complete Javadocs for `UnfoldAxisAttrs`, `Window2dAttrs`,
  and `Fold2dAttrs`, and refined `WindowTransformKind` and `FoldAxisAttrs` to state the conceptual
  `[3, 3]` window input and `[5]` fold result Shapes explicitly. It finalized the Tensor API and
  glossary with exact typed pairings, normalized general-axis unfold/fold, final-dimension window
  size, explicit target extent, NCHW im2col, overlap-summing col2im, formulas as documentation,
  validation/ownership, and current-versus-planned boundaries.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.layout.WindowTransformSemanticsTest` — `BUILD
  SUCCESSFUL`; the XML report contains 12 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; 73 XML suites contain 613 tests with zero
  failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL` without warnings. Generated pages were
  inspected for all five public types, four constants, record components, canonical constructors,
  explicit accessors, parameter/result/failure contracts, formulas, examples, reference ownership,
  and cross-layer exclusions.
- `./gradlew test` — `BUILD SUCCESSFUL`; the repository lifecycle reported 36 actionable tasks
  with no failing task. The documentation-tree run had one executed and 35 up-to-date; the final
  synchronized-tree rerun had all 36 up-to-date.
- `javap -classpath modules/model/build/classes/java/main -p -c -s` for all five production types
  confirmed exact enum order, exact record component fields and constructor signatures, explicit
  accessors, generated record methods, validation precedence, direct reference retention, and no
  extra project member or state. The focused reflection tests independently verify the same API
  surface, value semantics, extrema, failure messages, pairings, and forbidden-state absence.
- Production import and source inspection found only `OperationKind`, `OperationAttrs`, `Shape`,
  and `java.util.Objects` where applicable. No Tensor, descriptor, resolved-layout, provenance,
  graph, compiler, planning, prepare, runtime, backend, service, registry, map, reflection-
  discovery, ONNX, training, dependency, or build coupling was introduced.
- The targeted local Markdown checker resolved all 362 links, including heading anchors, across
  the six changed documentation/planning files with zero errors. Code fences are balanced, every
  permitted path ends with a newline, trailing-whitespace scans found no matches, and
  `git diff --check` passed.
- Final inventory contains exactly the twelve permitted paths: five production sources, one
  focused test, Tensor API, glossary, capabilities, this task, model master plan, and roadmap.
  Package placement matches the plan; task 0017N remains Draft and no `0017n-*.md` exists.
- Compile API remains accurate unchanged because there is no public unfold/fold Tensor expression,
  compiler entry point, capture, inference, canonicalization, autograd construction, or artifact
  behavior. Training API remains accurate unchanged because no gradient object, optimizer,
  parameter, training session, or training execution changed.
- `OperationKind`, `OperationAttrs`, `Operation`, `Shape`/`Dimension`, `TensorDescriptor`,
  `Tensor`, `TensorProvenance`, graph records, `LayoutDescriptor`, and adjacent layout semantic and
  expression contracts remain accurate unchanged because task 0017M implements new immutable
  semantic values without altering those representations or constructing results.
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, and architecture tests remain
  unchanged because module ownership, dependency direction, and compile/prepare/run boundaries did
  not change. Backend-conformance and integration tests remain unchanged because there is no
  backend or end-to-end behavior. Java 26 Gradle configuration, dependencies, existing unrelated
  Java/tests, other modules, and later task specifications remain unchanged for the same reason.

## Implementation notes

- Added exact `UNFOLD_AXIS`, `FOLD_AXIS`, `UNFOLD2D`, and `FOLD2D` semantic identities plus their
  four typed immutable attribute records in the existing layout-operation package.
- Constructors validate only intrinsic normalized component invariants in the specified order,
  retain valid extrema, and add no arithmetic, Tensor, Shape compatibility, registry, or cross-
  layer behavior. `Fold2dAttrs` null-checks and retains exact immutable references.
- Added one focused same-package structural/validation/composition test covering exact public
  shapes, validation messages and precedence, value semantics, typed pairings, reference identity,
  and forbidden dependency/state absence.
- Finalized Javadocs, Tensor API, glossary, capability addition, and synchronized planning
  evidence without modifying Compile API, Training API, architecture material, or creating task
  0017N.

## Completion summary

- Completed changes: implemented and documented exact backend-neutral single-axis unfold/fold and
  NCHW two-dimensional unfold/fold semantic identities and immutable intrinsic parameters.
- Files changed or created: exactly the twelve paths listed under Affected files.
- Tests and validation: focused 12/12, model 613/613 across 73 suites, model Javadoc, root tests,
  javap/reflection/import/source/generated-page checks, 362 local link/anchor checks, fence/
  whitespace/final-newline checks, exact scope/status checks, and `git diff --check` passed.
- Documentation-agent review: clean context `/root/implement_model_0017m/docs_0017m` completed the
  required independent pass using General, API/Javadoc, Planning, and Example profiles.
- Documentation impact: Tensor API and glossary now explain all four current semantic values,
  exact pairings, examples, formulas, ownership, validation, and deferred public/compiler/backend
  boundaries. Capabilities retains the intentional public `foldAxis` addition and distinguishes it
  from task-0023 compiler generation.
- Javadoc review: all five types, four constants, four canonical constructors, seventeen explicit
  accessors, components, results, failures, examples, ownership, and exclusions are complete;
  two Shape-wording refinements were made. Related contracts remain accurate unchanged.
- Glossary impact: added implemented window-transform terminology and aligned normalized-axis,
  operation-kind, and operation-attributes inventories.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0017M. Task 0017N remains Draft without a specification.

Status: Complete
