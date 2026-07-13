# Task 0023D: Public Fold Axis and Dynamic Window Transforms

## Status

Complete

## Goal

Close the window-transform gaps proved by the completed adjoint-expressibility audit without
adding operation-specific backward kinds.

This task restores the retained general-axis overlap-add meaning as a public `Tensor.foldAxis`
expression and generalizes the existing canonical NCHW `unfold2d`/`fold2d` pair to exact dynamic
channel and spatial Shapes. It also adds an exact typed padding-value variant of `UNFOLD2D`, so a
caller can materialize out-of-domain window positions as values such as negative infinity rather
than being limited to the existing conceptual positive-zero variant.

Mental model:

```text
general axis
  [..., windowCount, ..., windowSize]
    -- foldAxis(axis, outputSize, step) --> [..., outputSize, ...]
    overlapping window values are summed

NCHW 2D windows
  [N, C, H, W]
    -- unfold2d --> [N, C * kernelHeight * kernelWidth,
                         outputHeight * outputWidth]
    -- fold2d(targetShape, window) --> [N, C, H, W]

dynamic outputHeight * outputWidth
  is retained as a symbolic Dimension product, not flattened into an unnamed unknown
```

The task preserves the existing rank-three canonical im2col/col2im representation. It adds the
minimum missing symbolic product form instead of introducing a second non-flattened window format
whose Shape, ordering, compiler use, and public naming would diverge from the completed contract.

## Scope

- Extend the sealed `DimensionExpression` hierarchy with one canonical symbolic product form.
- Add `DimensionExpressions.multiply(Dimension, Dimension)` while preserving the existing
  constant-factor overload.
- Canonicalize symbolic products by folding static values, preserving zero/one identities,
  flattening nested products, combining structurally equal factors, and making factor-map order
  non-semantic.
- Add one public immutable `Unfold2dAttrs` record containing exact `Window2dAttrs` and exact typed
  `ScalarValue` padding metadata.
- Extend only `WindowTransformKind.UNFOLD2D` with the additional `Unfold2dAttrs` one-input,
  one-output signature. Preserve every existing kind, enum order, and signature variant.
- Restore exactly one public method:

  ```java
  public Tensor foldAxis(int axis, long outputSize, long step)
  ```

- Add exactly one configurable-padding overload:

  ```java
  public Tensor unfold2d(Window2dAttrs window, ScalarValue paddingValue)
  ```

- Preserve the existing `unfold(int,long,long)`, `unfold2d(Window2dAttrs)`, and
  `fold2d(Shape,Window2dAttrs)` signatures and their existing static behavior.
- Restore the historical locally validated `foldAxis` construction using the existing
  `FOLD_AXIS` and `FoldAxisAttrs` contracts.
- Generalize both `unfold2d` paths to dynamic channel, height, and width Dimensions while retaining
  the exact batch Dimension and canonical rank-three columns.
- Generalize `fold2d` to structurally exact dynamic column/target compatibility using the same
  symbolic formulas.
- Preserve unresolved result layouts, exact input metadata, exact one-input producers, output
  index zero, one-ID success, and no-ID local failures.
- Update every current global public-Tensor method-count lock from 192 to 194.
- Finalize Javadocs, Tensor and Compile API explanations, glossary terminology, capability status,
  this task, the model master plan, and the roadmap in a separate clean documentation pass.

## Out of scope

- a new backward, convolution-transpose, pooling-backward, im2col, col2im, patch, or maximum-pool
  index operation kind
- changing or removing `UNFOLD_AXIS`, `FOLD_AXIS`, `UNFOLD2D`, or `FOLD2D`
- changing existing `Window2dAttrs`, `FoldAxisAttrs`, or `Fold2dAttrs` record components,
  constructor validation, equality, hashing, or accessors
- changing the existing conceptual-positive-zero meaning of direct `UNFOLD2D + Window2dAttrs`
- replacing the canonical rank-three column representation with a rank-five/rank-six window
  format or adding a second public window result carrier
- arbitrary symbolic multiplication, division by a Dimension, subtraction, min/max, constraint
  solving, binding, evaluation, or a general algebra simplifier beyond the exact product contract
- public construction of a raw `DimensionExpression.Product`; construction remains through
  `DimensionExpressions`
- integral or Boolean `unfold2d`/`fold2d`, implicit scalar conversion, a default negative-infinity
  value, padding modes, asymmetric padding, or per-position padding Tensors
- changing general-axis unfold staticity or widening `FoldAxisAttrs.outputSize` beyond `long`
- value access, storage materialization, scatter-add execution, padding execution, algorithms,
  kernels, numerical tolerances, or performance promises
- gradients, adjoint rules, compiler capture/adoption, saved-value lifetime, graph rewriting,
  planning, prepare, runtime, backend, engine, or training behavior
- dependencies, Gradle/build changes, architecture/ADR changes, another module, unrelated refactors,
  task 0023E implementation, or a detailed 0023E specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Adjoint expressibility audit](../adjoint-expressibility-audit.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Glossary](../../../../glossary.md)
- completed tasks [0017M](0017m-unfold-and-fold-semantics.md),
  [0017N](0017n-unfold-and-fold-tensor-expressions.md),
  [0018M](0018m-symbolic-extent-expressions.md),
  [0018R](0018r-slice-and-window-public-contract-cleanup.md), and
  [0023](0023-adjoint-expressibility-audit.md)

## Architecture constraints

- `modules/model` owns the symbolic Shape values, backend-independent window semantics, public
  Tensor expressions, descriptors, and pre-capture producer metadata added here.
- Tensor remains public mutable API state and is not an IR node. Every valid call returns a fresh
  storage-free metadata Tensor; no method evaluates a value.
- `Operation` and window attributes contain no backend support, route, kernel, cost, fusion,
  planning, prepared, runtime, or execution state.
- Symbolic Dimension products are immutable model values. They do not bind, evaluate, inspect
  graph values, or move Shape solving into the model.
- `FOLD_AXIS` is one general public semantic identity. Later compiler-generated use must reuse this
  identity; this task adds no gradient rule or compiler construction.
- Configurable `UNFOLD2D` padding is an exact typed scalar semantic parameter, not backend fill
  policy. Backend prepare later chooses an implementation without changing the meaning.
- Existing direct `UNFOLD2D + Window2dAttrs` retains conceptual positive-zero padding. The new
  attributes variant does not silently migrate or rewrite completed producers.
- Result layout remains unresolved because window materialization and overlap accumulation do not
  prove alias-view geometry.
- No dependency or module-boundary change is authorized. If exact dynamic columns cannot be
  represented within these contracts, stop rather than add an unknown Shape or a cross-layer
  constraint service.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.shape` — symbolic Dimension expression values and factories.
- `io.github.pho001.synaptik.model.operation.layout` — typed window kinds and attributes.
- `io.github.pho001.synaptik.model.tensor` — public Tensor facade and package-private expression
  construction.

Packages added or moved: None.

Type placement:

- `DimensionExpression.Product` remains nested in the sealed Shape-expression hierarchy because it
  is one symbolic extent form, not a window-specific concept.
- `DimensionExpressions.multiply(Dimension, Dimension)` remains the sole validated public
  construction boundary for the product form.
- `Unfold2dAttrs` lives beside `Window2dAttrs` and `WindowTransformKind` because it is exact
  intrinsic `UNFOLD2D` semantic metadata.
- `TensorWindowExpressions` continues to own all public window-expression validation and
  descriptor/producer construction; no second window helper is added.

## Required contracts

### Canonical symbolic product

Add `DimensionExpression.Product` as the fifth permitted expression form. It stores exactly:

```java
Map<Dimension, Long> factors
long coefficient
```

The immutable factor map is non-empty, has non-null Dimension keys, non-null strictly positive
exponents, and order-independent value semantics. The coefficient is strictly positive. The
package-owned constructor validates `factors`, every key/value, exponents, then coefficient and
stores one `Map.copyOf` snapshot. It exposes documented `factors()` and `coefficient()` accessors
and exact structural `equals`, `hashCode`, and readable mathematical diagnostic text.

Add exactly this public factory overload:

```java
public static Dimension multiply(Dimension left, Dimension right)
```

Its exact behavior is:

- null-check left then right;
- return static zero if either operand is static zero;
- return the exact opposing reference for static one;
- use checked multiplication for two static operands;
- fold every other static operand into a positive checked coefficient;
- flatten nested canonical products and combine structurally equal factors with checked exponent
  addition;
- return the exact sole factor reference when coefficient and exponent are both one;
- otherwise return one `ExpressionDimension(Product)`.

Update existing `multiply(Dimension,long)` only so a newly introduced Product remains canonical:
zero/one behavior is unchanged, and a positive factor multiplies the Product coefficient with
checked arithmetic rather than wrapping the complete Product as a linear term. Every pre-0023D
input retains its previous result contract.

The product form does not distribute across sums, cancel terms, infer bounds, or evaluate. For
example, `multiply(Hout, Wout)` is exact structural metadata for the flattened column count, while
concrete binding remains later work.

### Configurable 2D unfold attributes

Add exactly:

```java
public record Unfold2dAttrs(Window2dAttrs window, ScalarValue paddingValue)
        implements OperationAttrs
```

The compact constructor null-checks `window` then `paddingValue` with those exact messages and
retains both exact references. It performs no DataType compatibility, Shape, window-count,
padding-coordinate, value, or execution validation. Record-generated equality, hashing, and
diagnostic text use both components in order. Explicit documented accessors return the exact
references.

`paddingValue` supplies every sample outside the logical unpadded input, including symmetric
padding and any terminal ceil-grid sample beyond that padded extent. The record accepts every
current ScalarValue structurally because it has no input Tensor; the Tensor helper later requires
an exact floating DataType match.

### Operation signatures

Preserve enum order and add no kind. Exact signatures become:

```text
UNFOLD_AXIS + UnfoldAxisAttrs: one input, one output
FOLD_AXIS   + FoldAxisAttrs:   one input, one output
UNFOLD2D    + Window2dAttrs:   one input, one output
UNFOLD2D    + Unfold2dAttrs:   one input, one output
FOLD2D      + Fold2dAttrs:     one input, one output
```

The `WindowTransformKind.UNFOLD2D.signatures()` list order is `Window2dAttrs` first and
`Unfold2dAttrs` second. Every other signature list remains bytecode-equivalent.

### Public Tensor surface

Add exactly:

```java
public Tensor foldAxis(int axis, long outputSize, long step)
public Tensor unfold2d(Window2dAttrs window, ScalarValue paddingValue)
```

Each method delegates once to the matching `TensorWindowExpressions` overload and owns no
validation, arithmetic, operation construction, factory call, alias, or default. Preserve the
existing three public window methods unchanged. The declared public Tensor method count becomes
exactly 194.

### General-axis fold

Restore the completed task-0017N contract without widening it:

1. null-check input;
2. require input rank at least two;
3. normalize the raw axis against target rank `inputRank - 1`;
4. construct exact `FoldAxisAttrs(axis, outputSize, step)`;
5. accept floating and integral input, reject BOOL;
6. require the selected window-count Dimension and final window-size Dimension to be static;
7. require the final window size positive;
8. validate exact zero-output or positive-output count geometry with checked arithmetic;
9. remove the final Dimension, replace the selected count with `StaticDimension(outputSize)`, and
   preserve every unaffected Dimension reference;
10. construct exact `FOLD_AXIS`, unresolved descriptor, `[input]` producer, output index zero, and
    one fresh Tensor.

Reuse the task-0017N exception types, validation order, and exact messages. Accept FLOAT64,
FLOAT32, BFLOAT16, INT32, and INT64. `requiresGrad` and DataType are retained exactly. This is
functional overlap summation metadata and never mutates the input.

### Dynamic canonical 2D unfold

Both `unfold2d` overloads:

- null-check in parameter order;
- require rank-four NCHW and floating input;
- preserve the exact batch Dimension;
- accept static, named dynamic, expression, and constrained-unknown channel/height/width
  Dimensions;
- reject only statically provable invalid geometry or checked arithmetic overflow;
- derive:

  ```text
  effectiveKernel = dilation * (kernel - 1) + 1
  outputSpatial = round((input + 2 * padding - effectiveKernel) / stride) + 1
  channelWindows = inputChannels * kernelHeight * kernelWidth
  windowCount = outputHeight * outputWidth
  result = [batch, channelWindows, windowCount]
  ```

- use existing linear/floor/ceiling forms plus the new canonical product; and
- create one exact `UNFOLD2D` occurrence and one fresh unresolved-layout result.

The original overload pairs directly with the exact supplied `Window2dAttrs` and retains its
conceptual positive-zero padding meaning. The new overload additionally null-checks
`paddingValue`, requires `paddingValue.dataType()` to equal the input DataType exactly, constructs
one `Unfold2dAttrs` retaining both supplied references, and preserves every raw scalar bit.

Exact type failure:

```text
unfold2d paddingValue data type must match input data type: paddingValue=<type>, input=<type>
```

No validation failure consumes a Tensor ID. Every successful call consumes exactly one.

### Dynamic canonical 2D fold

Preserve the public signature and validation prefix: input/output/window null checks, rank-three
canonical columns, rank-four NCHW target, floating type, and exact batch-Dimension equality.

Derive expected column channels and expected column count from the exact target Shape and window
using the same canonical formulas as unfold. Static invalid target geometry still fails locally.
Require exact structural equality for both complete expected Dimensions:

```text
actual columnChannels == targetChannels * kernelHeight * kernelWidth
actual columnCount    == outputHeight * outputWidth
```

Reuse the current mismatch-message forms with Dimension diagnostics in place of static longs.
Unprovable unrelated symbols are rejected rather than silently recorded as an unnamed equality
constraint. A Tensor returned by a matching dynamic `unfold2d` therefore composes with `fold2d`
exactly, while an arbitrary unresolved column Shape must state the same structural formulas.

The result retains the exact supplied output Shape reference, input floating DataType and
`requiresGrad`, unresolved layout, exact `FOLD2D/Fold2dAttrs`, `[input]` producer, output index
zero, and one fresh ID.

## Affected files

Production Java:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/shape/DimensionExpression.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/shape/DimensionExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/Unfold2dAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/WindowTransformKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/Window2dAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/FoldAxisAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/Fold2dAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorWindowExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/shape/DimensionExpressionsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationSignatureTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/layout/WindowTransformSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorWindowExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBatchNormInferenceExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorDenseCategoricalCrossEntropyWithLogitsExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorIndexCategoricalCrossEntropyWithLogitsExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLayerNormExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLinearExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMeanSquaredErrorExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorRmsNormExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorSlicePlacementExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorSumToShapeExpressionTest.java`

The twelve files after `TensorTest` receive count-only changes from 192 to 194.

Documentation and planning:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a contradiction requires stopping: Shape/Dimension public
contracts outside the two listed Shape files, DataType/ScalarValue, TensorDescriptor,
TensorFactory, TensorProducer/TensorProvenance, Operation, convolution/pooling contracts and tests,
Training/Runtime APIs, architecture/ADRs/tests, Gradle/dependencies, other modules, backend
conformance, integration tests, completed task history, and later tasks.

## Maximum scope

This task may create or modify exactly 33 paths: nine production files, seventeen tests, and seven
documentation/planning files. The count includes thirteen existing global Tensor-surface locks;
those are mechanical count-only edits except for `TensorTest`.

This is an explicit atomic exception to the normal 18-path guardrail. The symbolic product,
dynamic rank-three column Shape, configurable padding attributes, public overloads, restored
foldAxis, signatures, tests, and documentation form one contract. Splitting them would either
publish unused nonlinear Shape syntax, expose dynamic unfold without an exact fold, or temporarily
leave public/signature inventories inconsistent.

Do not use this allowance for unrelated cleanup. Stop if a thirty-fourth path, another type or
test, a completed-contract migration beyond the named files, another module, dependency, Gradle,
architecture change, or detailed 0023E specification is required.

## Javadoc requirements

- Document Product purpose, immutable factors/coefficient, canonicalization, identity behavior,
  overflow, equality, diagnostics, lack of binding/evaluation, and every accessor/factory input,
  result, and failure.
- Document `Unfold2dAttrs` exact padding region, exact typed bits, reference retention, accepted
  structural values, operation pairing, and absence of input-aware validation.
- Update all affected window Javadocs from compiler-only/static wording to the exact current public
  and dynamic boundaries without rewriting unchanged declarations.
- Document raw versus normalized fold axes, target rank, overlap summation, zero-output geometry,
  accepted types, Shape derivation, provenance, unresolved layout, freshness, ID effects, and
  failures.
- Document the direct positive-zero and explicit typed-padding `unfold2d` variants distinctly.
- Explain canonical im2col order and symbolic channel/window products with a concrete dynamic
  example such as `[N,C,H,W]` and a 3-by-3 kernel.
- Explain that configurable padding includes ceil-grid tail positions and preserves exact NaN,
  infinity, and signed-zero bits without selecting max-pool or backend behavior.
- Explain exact structural dynamic `fold2d` compatibility and why unrelated symbols are rejected.
- Review unchanged Shape, ScalarValue, descriptor, producer/provenance, convolution, pooling, and
  public window contracts; record reasoned no-change conclusions or stop on an out-of-scope
  discrepancy.

## Acceptance criteria

- The sealed expression hierarchy contains exactly the previous four forms plus Product.
- Product and both multiply overloads satisfy the exact surface, validation, canonicalization,
  value, immutability, reference, overflow, and diagnostic contracts.
- `Unfold2dAttrs` has exactly the two ordered components and no additional instance state or public
  API beyond record members and documented accessors.
- Window kind order is unchanged and only UNFOLD2D gains the exact second signature variant.
- Tensor adds exactly `foldAxis(int,long,long)` and
  `unfold2d(Window2dAttrs,ScalarValue)`; the public count is exactly 194.
- Restored foldAxis matches the retained semantics and historical completed validation contract,
  accepts five numeric types, rejects BOOL, preserves unaffected Dimensions, and creates one fresh
  result without execution.
- Existing static unfold2d/fold2d behavior and direct positive-zero attributes remain unchanged.
- Both unfold2d variants derive exact static or symbolic rank-three canonical columns without an
  unnamed unknown Dimension.
- Configurable unfold2d retains exact Window2dAttrs/ScalarValue references and bits and rejects a
  padding/input type mismatch before ID allocation.
- Fold2d accepts exact matching dynamic canonical columns, rejects structurally unprovable
  mismatch, and retains the exact target Shape.
- Every successful expression preserves the specified DataType/requiresGrad, unresolved layout,
  one producer, ordered `[input]`, output index zero, no label/storage, and one fresh ID.
- Failures follow exact order/messages and consume no ID. No values or storage are inspected.
- Focused tests, exactly one final model suite after executable stabilization, model Javadoc,
  runnable Java 26 example, Markdown/link/anchor checks, exact 33-path scope, status, and
  `git diff --check` pass.
- A separate clean documentation-focused agent finalizes every affected Javadoc, API explanation,
  glossary term, planning record, and validation result without repeating successful Java tests
  unless executable behavior changes or a concrete risk is recorded.
- Task 0023D becomes Complete only after both passes. Task 0023E remains Draft without a detailed
  specification. Architecture, dependencies, Gradle, other modules, compiler/runtime/backend, and
  completed unrelated contracts remain unchanged.

## Tests / validation

During development run the focused affected contracts as needed:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.shape.DimensionExpressionsTest \
  --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest \
  --tests io.github.pho001.synaptik.model.operation.layout.WindowTransformSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorWindowExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorBatchNormInferenceExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorBinaryArithmeticTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorDenseCategoricalCrossEntropyWithLogitsExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorIndexCategoricalCrossEntropyWithLogitsExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorLayerNormExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorLinearExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorMatmulExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorMeanSquaredErrorExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorRmsNormExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorScaledDotProductAttentionExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorSlicePlacementExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorSumToShapeExpressionTest
```

After executable Java stabilizes, run exactly once:

```bash
./gradlew :modules:model:test
```

Tests must cover exact reflection/surface shape, product static/identity/nested/commutative/
repeated-factor/overflow/value/diagnostic behavior, record signatures, every DataType, foldAxis
axes/zero/count/overflow, static and dynamic floor/ceil 2D geometry, exact configurable scalar
bits/type mismatch, structural fold compatibility, result metadata, producer/provenance, storage
non-interference, freshness, and ID side effects.

The separate documentation pass reuses successful Java evidence unless it changes executable
behavior. After final Javadocs it runs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

It also compiles/runs one Java 26 metadata example, verifies generated Javadoc and the 194-method
Tensor surface, validates all changed Markdown links/anchors/fences/newlines/terminology, confirms
exact 33-path scope and no 0023E specification, and records no-change conclusions.

Repository-wide tests are deferred to the next recorded model capability checkpoint or CI because
this task changes one module without dependencies, Gradle, module boundaries, or architecture.

## Dependencies

- Task 0002 and completed task 0018M for Shape, Dimension, and canonical symbolic extent values.
- Tasks 0005–0007 and 0018K for typed operation/signature construction.
- Tasks 0011–0013 and 0018L for Tensor descriptors, identity, producers, and provenance.
- Tasks 0017M–0017N for retained window meanings and the historical foldAxis contract.
- Task 0018N for exact typed scalar values.
- Task 0018R for the current signed-slice/window public boundary and retained FOLD_AXIS semantics.
- Tasks 0020–0020A1 for the dynamic convolution/pooling geometry that proves the window need.
- Task 0023 and its completed audit matrix. Tasks 0023A–0023C are complete preceding frontiers.

## Follow-up tasks

- Task 0023E remains Draft for cumulative-product scan semantics and Tensor construction.
- Task 0023F remains Draft for same-occurrence attention weights.
- Task 0024 remains Draft and depends on all task-0023 follow-ups.
- Later compiler work may compose these public transformations into adjoints and own binding,
  capture, saved-value lifetime, gradient accumulation, and optimization.
- Later planning/prepare/backend work owns algorithms, materialization, kernels, execution, and
  performance.

## Architecture impact

Expected impact: None.

The task extends model-owned Shape values and public backend-independent Tensor semantics without
changing ownership or dependencies. If implementation requires a solver, compiler constraint
object, runtime binding, backend policy, another module, or authoritative architecture edit, stop
and report the decision instead of proceeding.

## Implementation prompt

Use this prompt in a separate clean agentic task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, documentation/planning rules, roadmap, model capabilities/master
plan, tasks 0002/0017M/0017N/0018M/0018N/0018R/0020–0020A1/0023–0023D, Tensor and Compile APIs,
glossary, and every affected/review-only source/test named by task 0023D in full.

Implement task 0023D exactly inside its 33 authorized paths. Add the canonical symbolic Dimension
product, exact typed-padding UNFOLD2D variant, public foldAxis restoration, and dynamic canonical
rank-three unfold2d/fold2d Shape construction. Preserve every existing kind, static window
contract, producer/provenance rule, and architecture boundary. Add no alternate window format,
backward kind, compiler adoption, value execution, dependency, Gradle, architecture change, or
later task. Stop on architecture, dependency, completed-contract, validation-order, affected-file,
or maximum-scope conflict.

Run focused tests while developing and exactly one final model suite after executable Java
stabilizes. Then hand the actual diff and exact Java evidence to a separate clean-context
documentation-focused agent in the same overall change. That agent must independently inspect
final source/tests, finalize Javadocs, Tensor/Compile APIs, glossary, capability/task/master/
roadmap status and documentation validation, and reuse successful Java evidence unless executable
behavior changes or it records a concrete reason.

Do not mark 0023D Complete until both passes and every acceptance criterion succeed. Leave 0023E
and every later task Draft without a detailed specification.
```

## Local decisions

- Preserve canonical rank-three im2col/col2im Shapes. A reusable symbolic product is smaller and
  more coherent than introducing a second public rank-five/rank-six window representation.
- Product canonicalization uses a coefficient plus factor/exponent map so multiplication is
  order-independent and nested/repeated products remain structurally stable.
- Keep the existing zero-padding UNFOLD2D attributes variant unchanged. An explicit typed-padding
  overload is additive and does not migrate completed producer metadata.
- Require exact structural dynamic fold compatibility. This model task does not invent an equality
  constraint registry for unrelated symbols.
- Restore only the historical long-output-size foldAxis contract. Current unfold still requires a
  static selected extent, so no dynamic output-size attribute is needed for its exact adjoint.
- The 33-path scope is an intentional atomic exception dominated by thirteen mandatory public
  Tensor count locks.

## Known limitations

- Product expressions retain symbolic multiplication but do not bind or evaluate it.
- Dynamic formulas can retain a validity obligation such as a non-negative numerator; this task
  does not prove it before concrete binding.
- Fold2d rejects unrelated unresolved column symbols even if an external caller knows they will
  later bind to equal products.
- Window transforms remain metadata only. No current backend is promised to execute the new
  variant, and no compiler rule is added.
- Configurable padding is a single exact scalar, not a padding mode or Tensor-valued boundary.

## Validation evidence

- The implementation context `/root/implement_0023d` ran the exact focused 17-suite command in
  this task and passed 175 tests. After executable Java stabilized, that context ran exactly one
  final `./gradlew :modules:model:test`; it passed 1,008 tests across 126 suites with zero failures,
  errors, or skips. Documentation context `/root/implement_0023d/docs_0023d` reused that evidence
  because it changed only Javadocs and Markdown after the final suite; it did not rerun Java tests.
- Documentation context `/root/implement_0023d/docs_0023d` applied the General, API/Javadoc,
  Planning, and Example documentation profiles. It independently reviewed the architecture and
  planning contracts, actual diff, all nine production paths, the four behavior-defining tests,
  count-lock tests, Tensor/Compile API references, glossary, capabilities, task, master plan, and
  roadmap. It finalized all nine production Javadocs and all seven authorized documentation and
  planning files.
- `./gradlew :modules:model:javadoc` passed after final Javadoc edits (`BUILD SUCCESSFUL`; two
  executed tasks). Generated pages exist for Product, both Dimension factory contracts,
  `Unfold2dAttrs`, all window kinds/attributes, `Tensor`, and the construction helper. A targeted
  `rg` generated-page scan found `foldAxis`, the typed `unfold2d` overload, Product,
  `Unfold2dAttrs`, and canonical-symbolic wording. A later redundant Javadoc invocation inside a
  combined final command could not open the Gradle user-home lock under the filesystem sandbox;
  rerunning `./gradlew :modules:model:javadoc` with the approved Gradle permission passed with both
  tasks up to date. After the final constant-product Javadoc clarification, the approved command
  passed again with compileJava and Javadoc executed successfully.
- `javac --release 26 -cp modules/model/build/classes/java/main -d
  /tmp/synaptik-0023d-example /tmp/TensorWindowExpressionExample.java && java -cp
  modules/model/build/classes/java/main:/tmp/synaptik-0023d-example
  TensorWindowExpressionExample` passed. Its exact output was `Shape[2, 3, 3, 3]`, `Shape[5]`,
  `Shape[1, 4, 4]`, `Shape[1, 1, 3, 3]`, then four `true` lines confirming exact padding-reference
  retention, dynamic `9 * C`, exact dynamic fold target retention, unresolved layout, and absent
  result storage.
- The first compile of `/tmp/Task0023dSurfaceCheck.java` failed only because the validation helper
  declared an invariant generic `List<Class<?>>` for a captured class-token list. After correcting
  that helper-only type to `List<? extends Class<?>>`, the same `javac --release 26` and `java`
  check passed. It confirmed exactly 194 declared public Tensor methods; exact public `foldAxis`
  and typed `unfold2d` signatures; public `multiply(Dimension,Dimension)`; Product in the sealed
  hierarchy; ordered UNFOLD2D attribute variants `[Window2dAttrs, Unfold2dAttrs]`; and exact
  `Unfold2dAttrs` components `window:Window2dAttrs,paddingValue:ScalarValue`.
- The first `python3 /tmp/validate_synaptik_markdown.py` run identified a checker-only mismatch in
  GitHub slug handling for adjacent spaces around punctuation. After correcting that temporary
  validator, the same command passed. The final run after planning-status edits covered 208
  Markdown files, 3,462 local links, 205 local anchors, 2,606 fence markers, final newlines, and
  trailing whitespace. Targeted terminology scans found no stale compiler-only `foldAxis`,
  static-only 2D-window, four-form symbolic-expression, or single-UNFOLD2D-signature claim.
- `{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u` produced exactly
  the authorized 33 paths: nine production Java files, seventeen tests, and seven documentation/
  planning files. `git diff --check` passed. Status checks confirm task 0023D and its master-plan/
  roadmap rows are Complete, task 0023E and later rows remain Draft, and no 0023E task
  specification exists.
- No-change review concluded that Shape/Dimension contracts outside Product and its factories,
  `ScalarValue`, `TensorDescriptor`, `TensorFactory`, producer/provenance, convolution/pooling,
  Training/Runtime APIs, architecture/ADRs/tests, Gradle/dependencies, backend conformance,
  integration tests, other modules, completed task history, and later tasks remain accurate.
  The change adds model-owned immutable formulas, attributes, and storage-free expression
  construction only; it changes no descriptor ownership, scalar representation, producer model,
  numerical algorithm, compiler capture, gradient, runtime, backend, execution, dependency, or
  architecture contract.

## Implementation notes

- Added `DimensionExpression.Product` and public canonical Dimension-to-Dimension multiplication;
  the existing constant-factor path now scales a Product coefficient without nesting it as a
  linear term.
- Added `Unfold2dAttrs` and the second ordered UNFOLD2D signature while preserving the direct
  `Window2dAttrs` conceptual-positive-zero variant and every kind order.
- Restored public `foldAxis`, added exact typed-padding `unfold2d`, and generalized canonical
  rank-three 2D window formulas and exact fold compatibility to unresolved Dimensions.
- Updated all global public-Tensor surface locks from 192 to 194. No executable Java change
  occurred after the recorded final model suite.

## Completion summary

Completed changes: added canonical symbolic Dimension products, exact typed-padding UNFOLD2D
metadata, restored public general-axis overlap fold, and dynamic canonical rank-three
unfold2d/fold2d Shape construction with exact structural compatibility.

Files changed or created: exactly the nine production, seventeen test, and seven documentation/
planning paths listed under Affected files.

Tests or validation performed: focused 17-suite run (175 tests) and one final model suite (1,008
tests across 126 suites) from the implementation context; model Javadoc; runnable Java 26 metadata
example; Java 26 reflection and generated-Javadoc/API-surface checks; Markdown links, anchors,
fences, newlines, whitespace, and terminology; exact 33-path inventory; synchronized status; no
0023E specification; and `git diff --check` all passed.

Documentation-agent review: clean context `/root/implement_0023d/docs_0023d` independently
finalized affected Javadocs, Tensor/Compile API explanations, glossary, capability baseline, task,
master plan, roadmap, and documentation evidence without rerunning successful Java tests.

Documentation impact: Tensor and Compile API references now distinguish public foldAxis, direct
positive-zero versus exact typed-padding unfold, canonical symbolic products, dynamic 2D formulas,
and exact structural fold compatibility while preserving current-versus-planned boundaries.

Javadoc review: all nine authorized production paths are current, including the unchanged-behavior
`Window2dAttrs`, `FoldAxisAttrs`, and `Fold2dAttrs` declarations.

Glossary impact: symbolic extent expressions and window transforms now define canonical symbolic
products, both UNFOLD2D attribute variants, public foldAxis, dynamic canonical columns, and
structural fold compatibility.

Unresolved issues: None.

Required follow-up: None for task 0023D. Task 0023E remains the next Draft planning frontier
without a detailed specification.

Status: Complete
