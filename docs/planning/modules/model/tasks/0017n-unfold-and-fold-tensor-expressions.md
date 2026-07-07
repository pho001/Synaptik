# Task 0017N: Unfold and Fold Tensor Expressions

## Status

Complete

## Goal

Add the public storage-free Tensor expression surface for general-axis unfold and fold, NCHW
two-dimensional unfold into canonical columns, and two-dimensional fold back into an explicit NCHW
Shape.

The four methods validate only locally provable model facts, derive immutable result Shapes, create
unresolved descriptors, attach exact typed operation provenance, and return fresh Tensor identities.
They do not read values, allocate result storage, perform scatter-add, materialize windows, define
gradient rules, capture a compiled graph, or execute kernels.

## Scope

- Add exactly four public instance methods to Tensor:
  - `unfold(int axis, long size, long step)`;
  - `foldAxis(int axis, long outputSize, long step)`;
  - `unfold2d(Window2dAttrs window)`;
  - `fold2d(Shape outputShape, Window2dAttrs window)`.
- Add one package-private final field-free `TensorWindowExpressions` helper in the tensor package.
- Normalize general-axis raw axes locally and construct exact `UnfoldAxisAttrs` or `FoldAxisAttrs`.
- Support `unfold` for every current DataType and `foldAxis` for floating or integral DataTypes.
- Require a static selected/window-count axis for general-axis operations while preserving exact
  unaffected Dimension references, including dynamic dimensions.
- Support zero-sized public `foldAxis` output only for a zero window-count dimension.
- Require floating input for `unfold2d` and `fold2d`.
- Require rank-four NCHW input for `unfold2d`, with static channel/height/width dimensions while
  preserving the exact batch Dimension, including a dynamic batch.
- Require rank-three canonical columns and an explicit rank-four NCHW output Shape for `fold2d`.
- Validate exact batch, channel-window, and window-count compatibility using checked long
  arithmetic and the completed `Window2dAttrs` geometry.
- Leave every result layout unresolved because all four operations describe materialized output,
  not alias-view geometry.
- Preserve exact input DataType and gradient eligibility in every result descriptor.
- Create exact one-input provenance and delegate final construction once to
  `TensorFactory.createDerived` with no label or storage.
- Add one focused same-package expression test and update TensorTest only for exact public API
  shape.
- Finalize affected Javadocs, Tensor API, Compile API status, glossary, task evidence, master plan,
  and roadmap through the mandatory independent documentation pass.

## Out of scope

- executing window extraction, padding reads, scatter-add, overlap accumulation, zero filling,
  im2col, col2im, or any numerical value behavior
- allocating, attaching, copying, aliasing, slicing, or inspecting host/device storage; resolved
  LayoutDescriptor, strides, offsets, materialization policy, or liveness
- gradient rules, backward traversal, compiler-generated `FOLD_AXIS`, task-0023 implementation,
  autograd, optimizer, or training execution
- graph capture, CompiledNode/GraphValue construction, ID grouping, compiler passes,
  canonicalization, decomposition, fusion, planning, prepare, runtime, backend lowering/kernels,
  engine, trace, ONNX, or conformance behavior
- adding window defaults, overloads, builders, `Window2dOptions`, asymmetric padding, other padding
  modes, multiple general axes, reverse windows, adaptive windows, or im2col/col2im aliases
- permitting BOOL foldAxis accumulation, integral unfold2d/fold2d, dynamic transformed spatial or
  channel dimensions, symbolic constraint creation, or runtime shape binding
- changing declarations, logic, imports, or tests in completed WindowTransformKind,
  UnfoldAxisAttrs, FoldAxisAttrs, Window2dAttrs, or Fold2dAttrs; their Javadocs may receive only
  the authorized current-versus-planned temporal correction for completed public expressions
- changing Shape, TensorDescriptor, TensorFactory, TensorProvenance, Operation, or another
  existing Java contract/test
- dependencies, Gradle/build options, preview/incubator features, architecture changes, another
  module, unrelated documentation, task-0018 specification, or later implementation

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
- [Task 0001](0001-data-type-model.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0003](0003-layout-descriptor-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0017M](0017m-unfold-and-fold-semantics.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only legacy branch exposes:

```java
Tensor unfold(int axis, int size, int step)
Tensor unfold2d(Window2dOptions options)
Tensor fold2d(int[] outputShape, Window2dOptions options)
```

Task 0017M preserved these three meanings with long-valued model geometry and intentionally added
the symmetric public/compiler-facing `FOLD_AXIS` semantic. This task exposes the new public form:

```java
Tensor foldAxis(int axis, long outputSize, long step)
```

Legacy evidence establishes general-axis output ordering, NCHW im2col column ordering, symmetric
padding, stride/dilation/ceil-mode formulas, floating two-dimensional restrictions, and overlap-
summing fold behavior. The new code is implemented from scratch against current Shape, descriptor,
provenance, and package contracts. Legacy mutable arrays, graph builders, gradient callbacks,
storage/materialization, compiler/lowering code, and kernels are not copied.

## Architecture constraints

- Production remains in `modules/model`, which owns public Tensor expressions and backend-neutral
  semantic metadata.
- Tensor remains public mutable API state and does not become an IR node.
- Every method constructs metadata only. The model does not execute window values or allocate
  physical result buffers.
- Result layout remains unresolved even for fully static Shapes. These operations request
  materialized logical arrangements rather than proving alias views.
- The helper may use temporary arrays and checked arithmetic during local Shape derivation but
  stores no index, cache, service, registry, graph, or runtime state.
- General-axis normalization is relative to the input rank for unfold and to target rank
  `inputRank - 1` for foldAxis. Negative foldAxis axes therefore never address the final input
  window dimension.
- `FOLD_AXIS` is one semantic identity shared by its public expression and later task-0023 compiler
  generation. This task does not implement compiler insertion or gradient behavior.
- `Window2dAttrs` is both the exact semantic geometry and the public immutable geometry value; no
  duplicate options type is introduced.
- Package direction remains tensor -> datatype + shape + operation/layout + java.base.
- Stop if implementation requires changing a completed semantic/Shape/descriptor/factory contract,
  another production helper, another test, dependency, architecture decision, or cross-layer code.

## Package impact

No package is added or moved.

- `io.github.pho001.synaptik.model.tensor.Tensor` — four new public expression delegates only.
- `io.github.pho001.synaptik.model.tensor.TensorWindowExpressions` — package-private stateless
  validation, Shape derivation, semantic composition, provenance, and derived construction.
- `TensorWindowExpressionTest` — same-package behavioral and structural tests for all four methods.
- `TensorTest` — exact public Tensor API reflection inventory only.

The operation semantic contracts remain in `model.operation.layout`; no semantic type moves into
the public tensor package.

## Required contract

### Public Tensor surface

Add exactly:

```java
public Tensor unfold(int axis, long size, long step)
public Tensor foldAxis(int axis, long outputSize, long step)
public Tensor unfold2d(Window2dAttrs window)
public Tensor fold2d(Shape outputShape, Window2dAttrs window)
```

Each method delegates exactly once to its matching package-private helper entry and performs no
validation, arithmetic, operation construction, or factory call itself. Add no overload, static
facade, default geometry, alias, or convenience method.

### Helper surface

Create one field-free package-private final class with one private zero-argument constructor and
exactly these fourteen static methods:

```java
static Tensor unfoldAxis(Tensor input, int axis, long size, long step)
static Tensor foldAxis(Tensor input, int axis, long outputSize, long step)
static Tensor unfold2d(Tensor input, Window2dAttrs window)
static Tensor fold2d(Tensor input, Shape outputShape, Window2dAttrs window)
private static void validateNumeric(DataType dataType, String operation)
private static void validateFloating(DataType dataType, String operation)
private static int normalizeAxis(int axis, int rank)
private static long requireStaticSize(Shape shape, int axis, String operation, String dimension)
private static Shape unfoldAxisShape(Shape inputShape, UnfoldAxisAttrs attrs)
private static Shape foldAxisShape(Shape inputShape, FoldAxisAttrs attrs)
private static Shape unfold2dShape(Shape inputShape, Window2dAttrs window)
private static void validateFold2dShape(Shape inputShape, Shape outputShape, Window2dAttrs window)
private static long windowOutputSize(
        long inputSize,
        long kernel,
        long padding,
        long stride,
        long dilation,
        boolean ceilMode,
        String operation,
        String dimension)
private static Tensor create(Tensor input, Shape resultShape, Operation operation)
```

Use no method reference or lambda that generates a synthetic helper method. Add no field, nested
type, generic utility, public/package-private extra method, cache, or state.

### General-axis unfold

Validation and construction order is exact:

1. null-check input with message `input`;
2. require input rank at least one with message `unfold requires rank at least 1`;
3. normalize raw axis against input Shape using the current positive/negative axis convention;
4. construct `UnfoldAxisAttrs(normalizedAxis, size, step)`, delegating exact size/step validation;
5. require the selected Dimension to be static, otherwise fail with
   `unfold requires static selected dimension at axis <axis>`;
6. reject size greater than selected extent with
   `unfold size <size> exceeds selected dimension <extent>`;
7. calculate window count as `((extent - size) / step) + 1` using checked addition;
8. derive result Shape, Operation, unresolved descriptor, provenance, and fresh Tensor.

Every current DataType is accepted. Preserve exact DataType and requiresGrad. Replace only the
selected Dimension with a new StaticDimension(windowCount), preserve every other Dimension by
exact reference, and append a new StaticDimension(size). Input Shape `[2,5,3]`, axis 1, size 3,
step 1 produces `[2,3,3,3]`.

Zero selected extent cannot accept positive size and therefore fails the size-fit check. A step
larger than the remaining traversal is valid and produces one window when size fits.

### General-axis fold

Validation and construction order is exact:

1. null-check input with message `input`;
2. require input rank at least two with message `foldAxis requires rank at least 2`;
3. define target rank as `inputRank - 1` and normalize raw axis against that target rank;
4. construct `FoldAxisAttrs(normalizedAxis, outputSize, step)`, delegating exact output/step
   validation;
5. accept only floating or integral input; reject BOOL with
   `foldAxis requires floating or integral input: BOOL`;
6. require the selected window-count Dimension and final window-size Dimension to be static;
7. require positive final window size;
8. validate outputSize/window-count/window-size/step compatibility;
9. derive result Shape, Operation, unresolved descriptor, provenance, and fresh Tensor.

Use exact dynamic/static failures:

```text
foldAxis requires static window-count dimension at axis <axis>
foldAxis requires a positive static final window dimension
```

For outputSize zero, accept exactly zero windows and reject any non-zero window count with:

```text
foldAxis window count <actual> does not match output size and window geometry: expected=0
```

For positive outputSize, require windowSize <= outputSize, otherwise fail with:

```text
foldAxis window size <windowSize> exceeds output size <outputSize>
```

Then calculate expected windows as `((outputSize - windowSize) / step) + 1` and require exact
equality with the selected input extent. Mismatch uses:

```text
foldAxis window count <actual> does not match output size and window geometry: expected=<expected>
```

Remove the final input window Dimension, replace the selected window-count Dimension with new
StaticDimension(outputSize), and preserve every other target Dimension by exact reference.
Accept FLOAT64, FLOAT32, BFLOAT16, INT32, and INT64; reject BOOL. Preserve exact DataType and
requiresGrad. This constructs scatter-add semantics but performs no addition.

### Shared axis normalization

`normalizeAxis(axis, rank)` follows Shape semantics exactly: negative values add rank once; values
outside `[-rank, rank-1]` throw:

```text
Axis <axis> is outside shape rank <rank>
```

Use long intermediate normalization so Integer.MIN_VALUE cannot overflow. The helper is necessary
for foldAxis because its target rank excludes the final window dimension.

### Two-dimensional unfold

Validation and construction order is exact:

1. null-check input, then window, with messages `input` and `window`;
2. require input rank four with `unfold2d requires rank-4 NCHW input`;
3. require floating input through `validateFloating`;
4. preserve exact batch Dimension, which may be static or dynamic;
5. require static channel, height, and width Dimensions;
6. derive checked window channels and output spatial/window counts;
7. create exact `UNFOLD2D` Operation with the supplied Window2dAttrs reference;
8. create unresolved descriptor, one-input provenance, and fresh Tensor.

Static-dimension failures use:

```text
unfold2d requires static channel dimension at axis 1
unfold2d requires static height dimension at axis 2
unfold2d requires static width dimension at axis 3
```

The result Shape is:

```text
[exact batch Dimension,
 channel * kernelHeight * kernelWidth,
 outputHeight * outputWidth]
```

All arithmetic is checked. Input `[1,1,3,3]` with a 2x2 unit-stride, zero-padding, unit-dilation,
floor-mode window produces `[1,4,4]`. Preserve exact floating DataType and requiresGrad.

### Two-dimensional fold

Validation and construction order is exact:

1. null-check input, outputShape, then window with exact parameter-name messages;
2. require rank-three input with `fold2d requires rank-3 canonical column input`;
3. require rank-four outputShape with `fold2d outputShape must be rank-4 NCHW`;
4. require floating input;
5. require structural equality between input and output batch Dimensions, including matching
   dynamic symbols, otherwise fail with
   `fold2d output batch dimension must match column batch dimension`;
6. require static input column-channel and column-count Dimensions;
7. require static output channel, height, and width Dimensions;
8. calculate and validate expected channel-window and window-count extents;
9. construct exact Fold2dAttrs with the exact outputShape/window references;
10. create unresolved descriptor retaining the exact outputShape, one-input provenance, and fresh
    Tensor.

Static-dimension failures follow:

```text
fold2d requires static column-channel dimension at axis 1
fold2d requires static column-count dimension at axis 2
fold2d requires static output channel dimension at axis 1
fold2d requires static output height dimension at axis 2
fold2d requires static output width dimension at axis 3
```

Channel mismatch uses:

```text
fold2d column-channel dimension <actual> does not match output channels and kernel geometry: expected=<expected>
```

Window-count mismatch uses:

```text
fold2d column count <actual> does not match output shape and window geometry: expected=<expected>
```

Preserve exact floating DataType and requiresGrad. The output descriptor retains the exact supplied
Shape reference. Fold overlap summation is semantic only and is not executed here.

### Floating and numeric validation

`validateFloating` accepts exactly FLOAT64, FLOAT32, and BFLOAT16. Failure is:

```text
<operation> requires floating input: <dataType>
```

`validateNumeric` accepts floating and integral types. Failure is:

```text
<operation> requires floating or integral input: <dataType>
```

The operation argument is private constant text supplied only as `foldAxis`, `unfold2d`, or
`fold2d`; do not expose string dispatch or a registry.

### Window output-size calculation

`windowOutputSize` implements the task-0017M formula with checked long arithmetic:

```text
effectiveKernel = dilation * (kernel - 1) + 1
paddedInput     = inputSize + 2 * padding
numerator       = paddedInput - effectiveKernel
```

If numerator is negative, fail with:

```text
<operation> effective kernel does not fit padded <dimension>
```

Floor mode returns `numerator / stride + 1`. Ceil mode performs quotient/remainder rounding and
checked addition; do not use `numerator + stride - 1`, which may overflow. Kernel, stride,
padding, and dilation were already validated by Window2dAttrs; this helper performs no duplicate
parameter validation.

### Common result construction

Every successful call creates exactly one Operation, one TensorDescriptor with Optional.empty()
layout, one TensorProvenance with ordered inputs `[input]`, and invokes
`TensorFactory.createDerived(descriptor, Optional.empty(), provenance)` exactly once.

Return a fresh unlabeled, storage-free Tensor on every valid call, including identical repeated
requests. Do not inspect input label, provenance, host storage, storage liveness/content, or input
layout. Preserve input DataType and requiresGrad exactly. Do not return the input or canonicalize
nested/equivalent expressions.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorWindowExpressions.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorWindowExpressionTest.java`

Documentation/planning:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Authorized Javadoc-only temporal corrections:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/WindowTransformKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/UnfoldAxisAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/FoldAxisAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/Window2dAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/Fold2dAttrs.java`

Review without modification unless inconsistency requires stopping: Training API, capabilities,
all completed operation/Shape/descriptor/factory/provenance contracts and tests except the five
explicit Javadoc-only corrections above, focused
architecture/ADRs/tests, backend conformance, integration tests, Gradle/dependencies, other
modules, and task 0023.

## Maximum scope

Exactly two implementation production files, two test files, six documentation/planning files,
and the five authorized Javadoc-only semantic files: fifteen paths. The user explicitly expanded
the original ten-path limit only to correct stale task-0017N temporal wording in those five
completed semantic contracts. If any declaration, logic, import, semantic test, capabilities,
dependency, build/architecture change, sixteenth path, or task-0018 specification is required,
stop and report.

## Javadoc requirements

- Add complete Javadoc for all four public Tensor methods and the helper type, constructor, and all
  fourteen methods.
- Define window, window position, target rank, scatter-add, overlap, NCHW, im2col, col2im,
  symmetric padding, dilation, stride, effective kernel, floor mode, and ceil mode for newcomers.
- Document every parameter, units in logical elements, normalized/raw axis behavior, accepted
  DataTypes, dynamic/static rules, result Shape, unresolved layout, provenance, fresh identity,
  return value, and exact failure/overflow conditions.
- Include concrete numeric Shape examples for all four methods and explain each result dimension.
- Include a concrete foldAxis overlap calculation, for example windows `[1,2,3]`, `[2,3,4]`,
  `[3,4,5]` with step one producing accumulated `[1,4,9,8,5]`.
- Include the checked two-dimensional formulas and explain why ceil division avoids the common
  overflow-prone `numerator + stride - 1` expression.
- Explain that public foldAxis and later compiler-generated FOLD_AXIS share one semantic kind but
  this task adds no gradient rule or compiler behavior.
- Explain that all four expressions are metadata only, leave layout unresolved, attach no storage,
  and do not execute values.
- Review Javadocs for TensorDescriptor, Shape/Dimension, all five task-0017M contracts, Operation,
  TensorProvenance, and TensorFactory; record why unchanged wording remains accurate or stop on an
  out-of-scope discrepancy.

## Acceptance criteria

- Tensor adds exactly the four specified public methods with exact signatures and one delegation
  each; its total declared public method inventory is updated from 92 to 96.
- TensorWindowExpressions is final, package-private, field-free, has one private constructor and
  exactly fourteen declared methods without synthetic additions.
- Every validation follows the specified order, exception type, and exact message.
- General-axis unfold accepts all six DataTypes and derives exact Shape/provenance without values.
- foldAxis accepts five numeric DataTypes, rejects BOOL, normalizes against target rank, handles the
  zero-output/zero-window case, validates exact count geometry, and derives exact Shape/provenance.
- unfold2d/fold2d accept exactly floating input, validate static non-batch geometry with checked
  arithmetic, preserve a dynamic batch when structurally compatible, and derive/retain exact Shapes.
- Every result retains exact DataType/requiresGrad, has unresolved layout, no label/storage, exact
  one-input provenance, exact semantic kind/attrs, and a fresh TensorId.
- Focused tests cover reflection/API/helper shape, all DataTypes, negative axes, static/dynamic
  dimensions, zero dimensions, floor/ceil geometry, padding/stride/dilation, overflow, mismatch
  messages, exact reference retention, repeated freshness, and attached-storage non-interference.
- No completed semantic/foundational Java declaration, logic, import, or test changes; the five
  authorized semantic files contain Javadoc-only current-versus-planned corrections.
- Tensor API moves all four methods from planned to current and includes newcomer-readable worked
  examples; Compile API records current expressions without claiming capture/execution; glossary
  covers new terms and public-versus-compiler FOLD_AXIS distinction.
- Training API, capabilities, architecture docs/tests, conformance/integration, Gradle, other
  modules, and task 0023 receive reasoned no-change conclusions.
- Independent clean-context documentation pass finalizes Javadocs/docs/planning and reruns all
  validation in the same overall change.
- Final diff contains exactly fifteen paths, `git diff --check` passes, 0017N is Complete, 0018 remains
  Draft without a detailed specification, and no task-0018 specification exists.

## Tests / validation

Run:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorWindowExpressionTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Inspect compiled contracts:

```bash
javap -classpath modules/model/build/classes/java/main \
  io.github.pho001.synaptik.model.tensor.Tensor \
  io.github.pho001.synaptik.model.tensor.TensorWindowExpressions
```

Manual validation must confirm:

- exact four-method public addition, total 96 public Tensor methods, one delegation each, and no
  overload/alias;
- exact helper modifiers, zero fields, one private constructor, fourteen declared methods, no
  synthetic/lambda-generated methods, and no additional public/package-private surface;
- exact validation order/messages and checked arithmetic, including Integer.MIN_VALUE axes,
  Long.MAX_VALUE geometry, ceil rounding, and zero foldAxis output;
- exact result descriptor/provenance/reference/identity behavior and no storage/layout inspection;
- exact imports and no graph/compiler/planning/prepare/runtime/backend/ONNX/service/registry/
  reflection-discovery/dependency/build coupling;
- generated Javadoc contains every required type/method/parameter/result/failure/formula/example;
- every Markdown link/anchor resolves, fences balance, files end with newline, and changed/untracked
  files have no trailing whitespace;
- final inventory is exactly fifteen paths, package placement matches this plan, 0017N is Complete,
  0018 is Draft, and no `0018-*.md` task specification exists.

The documentation agent must rerun focused/model/root tests, model Javadoc, javap/reflection/
bytecode/import/source/generated-doc checks, Markdown/link/format checks, exact inventory/status
checks, and `git diff --check` against the final synchronized tree. Record commands, results, test
counts, agent identity, and reasoned no-change conclusions before completion.

## Dependencies

- Task 0001: Data type model
- Task 0002: Shape and dimension model
- Task 0003: Layout descriptor model
- Task 0007: Tensor descriptor model
- Task 0011: Public Tensor skeleton
- Task 0012: Tensor factory
- Task 0013: Tensor provenance skeleton
- Task 0017M: Unfold and fold semantics

## Follow-up tasks

- Task 0018: Indexing and scatter operations — next Draft frontier; create its detailed
  specification only after 0017N is Complete.
- Task 0023 later owns compiler-generated FOLD_AXIS use for autograd and reuses this public
  expression's completed semantic kind; it does not block 0017N.
- Compiler, planning, prepare, backend, ONNX, gradient, and execution work remain later tasks.

## Architecture impact

Expected impact: None.

This task adds model-owned public expression metadata only. It preserves Tensor/IR separation,
module dependency direction, compile/prepare/run separation, backend-owned lowering, runtime
boundaries, and the authoritative architecture contract. Stop if implementation requires changing
any of those decisions.

## Implementation prompt

Use this prompt in a separate clean-context agentic task:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0002/0003/0007/0011/0012/0013/0017M/0017N, Tensor API,
Compile API, Training API, glossary, current Tensor/Shape/Dimension/descriptor/factory/provenance,
window semantic contracts/tests, related tensor-expression helpers/tests, and Java 26 Gradle.

Implement task 0017N exactly. Modify Tensor.java and add package-private final
TensorWindowExpressions.java. Update TensorTest only for exact API shape and add
TensorWindowExpressionTest. Add exactly unfold(axis,size,step), foldAxis(axis,outputSize,step),
unfold2d(window), and fold2d(outputShape,window).

Follow the task's exact null/rank/axis/type/static-dimension/geometry/compatibility validation order
and messages. Use checked long arithmetic and overflow-safe ceil division. Preserve provable
dynamic Dimensions, exact DataType/requiresGrad, unresolved layout, exact one-input provenance,
and one createDerived call. Do not inspect/execute values or storage, derive resolved layouts, add
gradient/compiler/backend behavior, modify completed contracts, add overloads/types/dependencies,
change architecture/build, or create later specs. Apart from Javadoc-only temporal corrections in
the five explicitly authorized task-0017M semantic Java files, stop beyond fifteen paths or on
uncertainty.

Run all specified tests, Javadoc, javap/reflection/bytecode/import/manual, documentation/link/
whitespace/scope/status checks. Then hand the actual diff/evidence to a separate clean-context docs
agent in the same change. It must inspect source/tests/generated Javadoc, finalize permitted
Javadocs/Tensor API/Compile API/glossary/planning, record Training API/capabilities/architecture/
related-contract no-change conclusions, and rerun validation.

Update task 0017N, model master plan, and roadmap only for planning status/evidence. Do not mark
Complete until both passes succeed. Leave 0018 Draft without a specification. Do not commit/push.
```

## Local decisions

- Public `unfold` accepts every current DataType because this model boundary describes window
  materialization without arithmetic on values. `foldAxis` accepts floating and integral input but
  rejects BOOL because its semantic result is overlap-summing scatter-add. NCHW unfold/fold remain
  floating-only.
- `foldAxis` normalizes raw syntax against target rank `inputRank - 1`, after excluding the final
  input window dimension. It therefore cannot address that final dimension.
- Static input-dependent dimensions are required wherever the current Shape model cannot express a
  symbolic window count or compatibility constraint. Exact unaffected and batch Dimension
  references remain preserved where local proof is possible.
- Floor and ceil window counts use checked quotient/remainder arithmetic. Ceil mode deliberately
  avoids `numerator + stride - 1`, which can overflow even when the quotient is representable.
- Every result remains storage-free and layout-unresolved. No locally valid request is simplified
  to the input, and repeated identical calls receive fresh Tensor identities.
- The same FOLD_AXIS semantic identity serves current public `foldAxis` construction and future
  compiler-generated unfold adjoints. Task 0023 still owns the latter; this task adds no gradient
  or compiler behavior.
- Explicit user authorization expanded the original ten-path limit to exactly fifteen paths only
  so the five completed task-0017M semantic Java files could receive Javadoc-only temporal
  corrections. Their declarations, imports, logic, and tests remain unchanged.

## Known limitations

- The expressions calculate metadata only. They do not materialize windows, sample padding,
  scatter-add values, zero-fill uncovered output, allocate or attach storage, or execute kernels.
- General-axis transformed dimensions and NCHW channel/spatial geometry must be statically known;
  this task creates no symbolic constraints or runtime Shape binding.
- Two-dimensional geometry is NCHW, symmetric-padding, single-window-configuration only. There are
  no defaults, builders, asymmetric padding modes, aliases, or integral NCHW forms.
- Gradient rules, compiler capture/canonicalization, compiler-generated FOLD_AXIS, planning,
  materialization, backend/ONNX lowering, and execution remain future owning-layer work.

## Validation evidence

- The implementation pass and independent clean-context documentation pass both completed in the
  same uncommitted tree. The final documentation agent context was
  `/root/implement_model_0017n/docs_0017n_resumed`; it read the architecture contract, focused
  architecture documents, documentation rules and General/API-Javadoc/Planning/Example profiles,
  planning guide/roadmap, model plans and capability baseline, prerequisite/task specifications,
  API/glossary documents, affected source/tests, related contracts, and Java 26 Gradle setup.
- `./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorWindowExpressionTest`
  passed all 16 focused tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` passed all 629 tests across 74 suites with zero failures, errors,
  or skips. `./gradlew :modules:model:javadoc` succeeded without warnings, and generated public
  Tensor plus all five semantic-contract pages contain the current signatures, parameters,
  returns, failures, examples, formulas, and current-versus-planned boundary. The package-private
  helper is intentionally outside the public Javadoc task output; all fourteen source Javadocs
  were reviewed directly.
- `./gradlew test` completed the 36-task root lifecycle successfully, including the model test
  result and all currently populated modules; no backend, conformance, or integration production
  behavior was added.
- `javap -p -c -s` plus a compiled reflection helper confirmed 96 declared public Tensor methods;
  the four exact new signatures each delegate once. `TensorWindowExpressions` is package-private
  final and field-free, with one private constructor, exactly fourteen declared methods, and zero
  synthetic methods. Bytecode contains one final `TensorFactory.createDerived` call in the shared
  construction path and checked `Math` arithmetic without lambda-generated methods.
- A separately compiled and executed copy of `TensorWindowExpressionExample` printed
  `Shape[2, 3, 3, 3]`, `Shape[5]`, `Shape[1, 4, 4]`, `Shape[1, 1, 3, 3]`, `UNFOLD_AXIS`, `FOLD2D`,
  `true`, and `true`, exactly as documented.
- Source/import/manual inspection confirmed exact package placement, no legacy overload/options,
  no helper fields, no graph/compiler/planning/prepare/runtime/backend/ONNX/service/registry or
  reflection-discovery coupling, unresolved result layouts, one-input provenance, and no storage
  inspection. The five semantic Java diffs contain Javadoc changes only.
- The targeted Markdown checker resolved 370 local links, including 108 anchors, across the six
  changed documentation/planning files with zero errors. All 242 code-fence markers are balanced;
  every authorized path ends with a newline and has no trailing whitespace.
- `git diff --check` passed. The combined changed/untracked inventory is exactly the fifteen paths
  authorized above. Task 0017N, the model master plan, and the roadmap agree on Complete; task
  0018 remains Draft and no `0018*.md` task specification exists.

## Implementation notes

- `Tensor` exposes only the four specified instance methods and delegates all validation and
  construction to the field-free helper. The helper centralizes source/target-rank normalization,
  static-dimension extraction, checked general-axis and two-dimensional Shape arithmetic,
  canonical-column validation, and final descriptor/provenance construction.
- `TensorWindowExpressionTest` covers API/helper reflection shape, every DataType boundary,
  positive and negative axes, floor/ceil/padding/stride/dilation geometry, static/dynamic and zero
  dimensions, overflow, exact failures, reference retention, identity freshness, and attached-
  storage non-interference. `TensorTest` changes only the exact public method inventory.
- The independent documentation pass finalized public/helper and semantic Javadocs, Tensor API,
  Compile API, glossary, task evidence, model master plan, and roadmap. It applied the General,
  API/Javadoc, Planning, and Example documentation profiles.
- Training API remains accurate unchanged because this task adds no gradient, optimizer, or
  training behavior. Capabilities remains unchanged by explicit scope and because it is the broad
  capability baseline rather than task-status evidence. TensorDescriptor, Shape/Dimension,
  Operation, TensorProvenance, TensorFactory, related expression helpers, and completed semantic
  declarations remain accurate because the new methods compose their existing contracts without
  changing them.
- Architecture documents/ADRs/tests remain unchanged because no module boundary, dependency, or
  lifecycle decision changed. Backend conformance and integration tests remain unchanged because
  there is no backend or end-to-end execution behavior. Java 26 Gradle configuration,
  dependencies, other modules, and task 0023 also remain unchanged.

## Completion summary

- Completed changes: added four public storage-free unfold/fold Tensor expressions, the exact
  field-free fourteen-method helper, focused tests, public API inventory updates, complete
  Javadocs/API/glossary documentation, five authorized temporal Javadoc corrections, and
  synchronized planning evidence.
- Files changed or created: exactly the two implementation production files, two test files, six
  documentation/planning files, and five Javadoc-only semantic files listed under Affected files.
- Tests and validation: focused 16/16, model 629/629 across 74 suites, model Javadoc, root tests,
  javap/reflection/bytecode/import/source/generated-doc checks, executable example, 370 local
  link checks including 108 anchors, fence/newline/whitespace checks, exact fifteen-path inventory,
  synchronized status, no-0018-spec check, and `git diff --check` all passed.
- Unresolved issues: none for task 0017N.
- Required follow-up: none for task 0017N. Task 0018 remains the next Draft frontier without a
  specification; task 0023 later owns compiler-generated FOLD_AXIS and autograd behavior.

Status: Complete
