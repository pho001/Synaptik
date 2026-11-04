# Task 0023D: Public Fold [Axis](../../../../glossary.md#term-axis) and Dynamic Window Transforms

**Editorial reading key (current context, not a status update).**

This record concerns Task 0023D’s Model-only construction boundary; its method counts and deferred
compiler/backend work are historical, not current capability descriptions. The caller supplies a
Tensor, window attributes, and optionally a typed padding scalar; the Tensor helper validates these
descriptions and returns a fresh storage-free expression recording its exact input Tensor reference
and its own producing occurrence. It neither
reads samples nor owns a result buffer. Model defines the meaning; execution providers realize
supported routes without redefining it. Semantic responsibility is not storage ownership.
`unfold2d` exposes samples as canonical rank-three columns; fold adds overlapping contributions rather
than undoing extraction or averaging. For a one-channel three-by-three image and two-by-two windows
at unit stride, folding unchanged columns counts the center four times. Symbolic products describe
unresolved extents without binding them; dynamic fold requires structurally matching formulas.
Direct padding means positive zero; explicit padding retains same-typed scalar bits, including NaNs,
infinities, and signed zero, also for ceil-grid tails. Follow [current window-transform
expressions](../../../../reference/tensor/windows.md#window-transform-expressions) for these
contracts and their distinction from later compiler/provider capabilities.

## Status

Complete

## Goal

Close the window-transform gaps proved by the completed [adjoint](../../../../glossary.md#term-adjoint)-expressibility audit without
adding [operation](../../../../glossary.md#operation)-specific backward kinds.

This task restores the retained general-[axis](../../../../glossary.md#term-axis) overlap-add meaning as a public `Tensor.foldAxis`
expression and generalizes the existing [canonical](../../../../glossary.md#term-canonical-form) [NCHW](../../../../glossary.md#term-nchw) `unfold2d`/`fold2d` pair to exact dynamic
[channel](../../../../glossary.md#term-channel) and [spatial](../../../../glossary.md#term-spatial-axes) [Shapes](../../../../glossary.md#shape). It also adds an exact typed [padding](../../../../glossary.md#padding)-value variant of `UNFOLD2D`, so a
caller can materialize out-of-domain window positions as values such as [negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero) rather
than being limited to the existing conceptual positive-zero variant.

[Mental model](../../../../glossary.md#term-mental-model):

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

The task preserves the existing [rank](../../../../glossary.md#term-rank)-three canonical [im2col](../../../../glossary.md#term-im2col)/[col2im](../../../../glossary.md#term-col2im) representation. It adds the
minimum missing symbolic product form instead of introducing a second non-flattened window format
whose Shape, ordering, [compiler](../../../../glossary.md#term-compiler) use, and public naming would diverge from the completed contract.

## [Scope](../../../../glossary.md#term-scope)

- Extend the [sealed](../../../../glossary.md#term-sealed) `DimensionExpression` hierarchy with one [canonical](../../../../glossary.md#term-canonical-form) symbolic product form.
- Add `DimensionExpressions.multiply(Dimension, Dimension)` while preserving the existing
  constant-factor overload.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>
- Canonicalize symbolic products by folding static values, preserving zero/one identities,
  flattening nested products, combining structurally equal factors, and making factor-map order
  non-semantic.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span>
- Add one public [immutable](../../../../glossary.md#term-immutable) `Unfold2dAttrs` record containing exact `Window2dAttrs` and exact typed
  [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) [padding](../../../../glossary.md#padding) [metadata](../../../../glossary.md#term-metadata).
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
  `fold2d(Shape,Window2dAttrs)` [signatures](../../../../glossary.md#term-signature) and their existing static behavior.
- Restore the historical locally validated `foldAxis` construction using the existing
  `FOLD_AXIS` and `FoldAxisAttrs` contracts.
- Generalize both `unfold2d` paths to dynamic channel, height, and width Dimensions while retaining
  the exact batch Dimension and canonical rank-three columns.<span data-glossary-navigation="true"> ([channel glossary](../../../../glossary.md#term-channel); [Dimensions glossary](../../../../glossary.md#dimension); [batch Dimension glossary](../../../../glossary.md#batch-dimension); [rank glossary](../../../../glossary.md#term-rank))</span>
- Generalize `fold2d` to structurally exact dynamic column/target compatibility using the same
  symbolic formulas.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- Preserve unresolved result layouts, exact input metadata, exact one-input producers, output
  index zero, one-ID success, and no-ID local failures.<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout))</span>
- Update every current global public-[Tensor](../../../../glossary.md#tensor) method-count lock from 192 to 194.
- Finalize Javadocs, Tensor and Compile API explanations, glossary terminology, capability status,
  this task, the model master plan, and the roadmap in a separate clean documentation pass.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- a new backward, convolution-transpose, pooling-backward, im2col, col2im, patch, or maximum-pool
  index operation kind<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [transpose glossary](../../../../glossary.md#term-transpose); [pooling glossary](../../../../glossary.md#term-pooling); [im2col glossary](../../../../glossary.md#term-im2col); [col2im glossary](../../../../glossary.md#term-col2im); [operation glossary](../../../../glossary.md#operation))</span>
- changing or removing `UNFOLD_AXIS`, `FOLD_AXIS`, `UNFOLD2D`, or `FOLD2D`
- changing existing `Window2dAttrs`, `FoldAxisAttrs`, or `Fold2dAttrs` record components,
  constructor validation, equality, hashing, or accessors<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- changing the existing conceptual-positive-zero meaning of direct `UNFOLD2D + Window2dAttrs`
- replacing the canonical rank-three column representation with a rank-five/rank-six window
  format or adding a second public window result carrier<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [rank glossary](../../../../glossary.md#term-rank); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- arbitrary symbolic multiplication, division by a Dimension, subtraction, min/max, constraint
  solving, binding, evaluation, or a general algebra simplifier beyond the exact product contract<span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension); [binding glossary](../../../../glossary.md#term-binding); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- public construction of a raw `DimensionExpression.Product`; construction remains through
  `DimensionExpressions`
- integral or [Boolean](../../../../glossary.md#data-type--datatype) `unfold2d`/`fold2d`, implicit scalar conversion, a default negative-infinity
  value, padding modes, asymmetric padding, or per-position padding Tensors<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [infinity glossary](../../../../glossary.md#term-infinity); [padding glossary](../../../../glossary.md#padding); [Tensors glossary](../../../../glossary.md#tensor))</span>
- changing general-[axis](../../../../glossary.md#term-axis) unfold staticity or [widening](../../../../glossary.md#term-widening) `FoldAxisAttrs.outputSize` beyond `long`
- value access, storage materialization, scatter-add execution, padding execution, algorithms,
  kernels, numerical tolerances, or performance promises<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [kernels glossary](../../../../glossary.md#kernel))</span>
- gradients, adjoint rules, compiler capture/adoption, saved-value lifetime, graph rewriting,
  planning, prepare, runtime, backend, engine, or training behavior<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient); [adjoint glossary](../../../../glossary.md#term-adjoint); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [lifetime glossary](../../../../glossary.md#term-lifetime); [graph glossary](../../../../glossary.md#graph); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [engine glossary](../../../../glossary.md#term-engine))</span>
- dependencies, Gradle/build changes, architecture/ADR changes, another module, unrelated refactors,
  task 0023E implementation, or a detailed 0023E specification<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model master plan](../master-plan.md)
- [Adjoint expressibility audit](../adjoint-expressibility-audit.md)<span data-glossary-navigation="true"> ([Adjoint glossary](../../../../glossary.md#term-adjoint))</span>
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Glossary](../../../../glossary.md)
- completed tasks [0017M](0017m-unfold-and-fold-semantics.md),
  [0017N](0017n-unfold-and-fold-tensor-expressions.md),
  [0018M](0018m-symbolic-extent-expressions.md),
  [0018R](0018r-slice-and-window-public-contract-cleanup.md), and
  [0023](0023-adjoint-expressibility-audit.md)

## Architecture constraints

- `modules/model` owns the symbolic Shape values, backend-independent window semantics, public
  Tensor expressions, descriptors, and pre-capture producer metadata added here.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [backend glossary](../../../../glossary.md#backend); [Tensor glossary](../../../../glossary.md#tensor); [descriptors glossary](../../../../glossary.md#term-descriptor); [capture glossary](../../../../glossary.md#term-capture-meanings); [producer glossary](../../../../glossary.md#term-producer); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Tensor remains public mutable API state and is not an IR node. Every valid call returns a fresh
  storage-free metadata Tensor; no method evaluates a value.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [node glossary](../../../../glossary.md#node))</span>
- [`Operation`](../../../../glossary.md#operation) and window attributes contain no backend support, route, kernel, cost, fusion,
  planning, prepared, runtime, or execution state.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [fusion glossary](../../../../glossary.md#term-fusion); [planning glossary](../../../../glossary.md#term-planning); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Symbolic Dimension products are immutable model values. They do not bind, evaluate, inspect
  graph values, or move Shape solving into the model.<span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension); [immutable glossary](../../../../glossary.md#term-immutable); [model glossary](../../../../glossary.md#term-model); [graph glossary](../../../../glossary.md#graph))</span>
- `FOLD_AXIS` is one general public semantic identity. Later compiler-generated use must reuse this
  identity; this task adds no gradient rule or compiler construction.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [compiler glossary](../../../../glossary.md#term-compiler); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Configurable `UNFOLD2D` padding is an exact typed scalar semantic parameter, not backend fill
  policy. Backend prepare later chooses an implementation without changing the meaning.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [scalar glossary](../../../../glossary.md#term-scalar); [parameter glossary](../../../../glossary.md#term-parameter); [prepare glossary](../../../../glossary.md#prepare))</span>
- Existing direct `UNFOLD2D + Window2dAttrs` retains conceptual positive-zero padding. The new
  attributes variant does not silently migrate or rewrite completed producers.
- Result layout remains unresolved because window materialization and overlap accumulation do not
  prove alias-view geometry.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [materialization glossary](../../../../glossary.md#materialization); [alias glossary](../../../../glossary.md#aliasing-and-arena); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- No dependency or module-boundary change is authorized. If exact dynamic columns cannot be
  represented within these contracts, stop rather than add an unknown Shape or a cross-layer
  constraint service.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.shape` — symbolic [Dimension](../../../../glossary.md#dimension) expression values and factories.
- `io.github.pho001.synaptik.model.operation.layout` — typed window kinds and attributes.
- `io.github.pho001.synaptik.model.tensor` — public Tensor facade and package-private expression
  construction.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [facade glossary](../../../../glossary.md#term-facade))</span>

Packages added or moved: None.

Type placement:

- `DimensionExpression.Product` remains nested in the sealed Shape-expression hierarchy because it
  is one symbolic extent form, not a window-specific concept.<span data-glossary-navigation="true"> ([sealed glossary](../../../../glossary.md#term-sealed); [Shape glossary](../../../../glossary.md#shape); [extent glossary](../../../../glossary.md#term-extent))</span>
- `DimensionExpressions.multiply(Dimension, Dimension)` remains the sole validated public
  construction boundary for the product form.
- `Unfold2dAttrs` lives beside `Window2dAttrs` and `WindowTransformKind` because it is exact
  intrinsic `UNFOLD2D` semantic [metadata](../../../../glossary.md#term-metadata).
- `TensorWindowExpressions` continues to own all public window-expression validation and
  descriptor/producer construction; no second window helper is added.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [descriptor glossary](../../../../glossary.md#term-descriptor); [producer glossary](../../../../glossary.md#term-producer))</span>

## Required contracts

### [Canonical](../../../../glossary.md#term-canonical-form) symbolic product

Add `DimensionExpression.Product` as the fifth permitted expression form. It stores exactly:

```java
Map<Dimension, Long> factors
long coefficient
```

The [immutable](../../../../glossary.md#term-immutable) factor map is non-empty, has non-null [Dimension](../../../../glossary.md#dimension) keys, non-null strictly positive
exponents, and order-independent value semantics. The coefficient is strictly positive. The
package-owned constructor validates `factors`, every key/value, exponents, then coefficient and
stores one `Map.copyOf` [snapshot](../../../../glossary.md#term-snapshot). It exposes documented `factors()` and `coefficient()` accessors
and exact structural `equals`, [`hashCode`](../../../../glossary.md#hash-codes-and-content-digests), and readable mathematical diagnostic text.

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
- return the exact sole factor reference when coefficient and [exponent](../../../../glossary.md#term-exponent) are both one;
- otherwise return one `ExpressionDimension(Product)`.

Update existing `multiply(Dimension,long)` only so a newly introduced Product remains canonical:
zero/one behavior is unchanged, and a positive factor multiplies the Product coefficient with
checked arithmetic rather than wrapping the complete Product as a [linear](../../../../glossary.md#term-linear-meanings) term. Every pre-0023D
input retains its previous result contract.

The product form does not distribute across sums, cancel terms, infer bounds, or evaluate. For
example, `multiply(Hout, Wout)` is exact structural [metadata](../../../../glossary.md#term-metadata) for the flattened column count, while
concrete [binding](../../../../glossary.md#term-binding) remains later work.

### Configurable 2D unfold attributes

Add exactly:

```java
public record Unfold2dAttrs(Window2dAttrs window, ScalarValue paddingValue)
        implements OperationAttrs
```

The compact constructor null-checks `window` then `paddingValue` with those exact messages and
retains both exact references. It performs no [DataType](../../../../glossary.md#data-type--datatype) compatibility, [Shape](../../../../glossary.md#shape), window-count,
[padding](../../../../glossary.md#padding)-[coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), value, or execution [validation](../../../../glossary.md#term-validation). Record-generated equality, hashing, and
diagnostic text use both components in order. Explicit documented accessors return the exact
references.

`paddingValue` supplies every [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) outside the logical unpadded input, including symmetric
padding and any terminal ceil-grid sample beyond that padded [extent](../../../../glossary.md#term-extent). The record accepts every
current [ScalarValue](../../../../glossary.md#typed-scalar-value--scalarvalue) structurally because it has no input [Tensor](../../../../glossary.md#tensor); the Tensor helper later requires
an exact floating DataType match.

### [Operation](../../../../glossary.md#operation) [signatures](../../../../glossary.md#term-signature)

Preserve enum order and add no kind. Exact signatures become:

```text
UNFOLD_AXIS + UnfoldAxisAttrs: one input, one output
FOLD_AXIS   + FoldAxisAttrs:   one input, one output
UNFOLD2D    + Window2dAttrs:   one input, one output
UNFOLD2D    + Unfold2dAttrs:   one input, one output
FOLD2D      + Fold2dAttrs:     one input, one output
```

The `WindowTransformKind.UNFOLD2D.signatures()` list order is `Window2dAttrs` first and
`Unfold2dAttrs` second. Every other signature list remains [bytecode](../../../../glossary.md#term-bytecode)-equivalent.

### Public [Tensor](../../../../glossary.md#tensor) surface

Add exactly:

```java
public Tensor foldAxis(int axis, long outputSize, long step)
public Tensor unfold2d(Window2dAttrs window, ScalarValue paddingValue)
```

Each method delegates once to the matching `TensorWindowExpressions` overload and owns no
[validation](../../../../glossary.md#term-validation), arithmetic, [operation](../../../../glossary.md#operation) construction, factory call, [alias](../../../../glossary.md#aliasing-and-arena), or default. Preserve the
existing three public window methods unchanged. The declared public Tensor method count becomes
exactly 194.

### General-[axis](../../../../glossary.md#term-axis) fold

Restore the completed task-0017N contract without [widening](../../../../glossary.md#term-widening) it:

1. null-check input;
2. require input [rank](../../../../glossary.md#term-rank) at least two;
3. normalize the raw axis against [target](../../../../glossary.md#term-target) rank `inputRank - 1`;
4. construct exact `FoldAxisAttrs(axis, outputSize, step)`;
5. accept floating and integral input, reject [BOOL](../../../../glossary.md#data-type--datatype);
6. require the selected window-count [Dimension](../../../../glossary.md#dimension) and final window-size Dimension to be static;
7. require the final window size positive;
8. validate exact zero-output or positive-output count geometry with checked arithmetic;
9. remove the final Dimension, replace the selected count with `StaticDimension(outputSize)`, and
   preserve every unaffected Dimension reference;
10. construct exact `FOLD_AXIS`, unresolved [descriptor](../../../../glossary.md#term-descriptor), `[input]` producer, output index zero, and
    one fresh Tensor.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [Tensor glossary](../../../../glossary.md#tensor))</span>

Reuse the task-0017N exception types, [validation](../../../../glossary.md#term-validation) order, and exact messages. Accept [FLOAT64](../../../../glossary.md#term-float64),
[FLOAT32](../../../../glossary.md#term-float32), [BFLOAT16](../../../../glossary.md#term-bfloat16), [INT32](../../../../glossary.md#data-type--datatype), and [INT64](../../../../glossary.md#data-type--datatype). `requiresGrad` and [DataType](../../../../glossary.md#data-type--datatype) are retained exactly. This is
functional overlap summation [metadata](../../../../glossary.md#term-metadata) and never mutates the input.

### Dynamic [canonical](../../../../glossary.md#term-canonical-form) 2D unfold

Both `unfold2d` overloads:

- null-check in [parameter](../../../../glossary.md#term-parameter) order;
- require [rank](../../../../glossary.md#term-rank)-four [NCHW](../../../../glossary.md#term-nchw) and floating input;
- preserve the exact [batch Dimension](../../../../glossary.md#batch-dimension);
- accept static, named dynamic, expression, and constrained-unknown channel/height/width
  Dimensions;<span data-glossary-navigation="true"> ([channel glossary](../../../../glossary.md#term-channel); [Dimensions glossary](../../../../glossary.md#dimension))</span>
- reject only statically provable invalid geometry or checked arithmetic [overflow](../../../../glossary.md#term-overflow);
- derive:

  ```text
  effectiveKernel = dilation * (kernel - 1) + 1
  outputSpatial = round((input + 2 * padding - effectiveKernel) / stride) + 1
  channelWindows = inputChannels * kernelHeight * kernelWidth
  windowCount = outputHeight * outputWidth
  result = [batch, channelWindows, windowCount]
  ```

- use existing [linear](../../../../glossary.md#term-linear-meanings)/floor/ceiling forms plus the new canonical product; and
- create one exact `UNFOLD2D` [occurrence](../../../../glossary.md#term-occurrence) and one fresh unresolved-[layout](../../../../glossary.md#layout) result.

The original overload pairs directly with the exact supplied `Window2dAttrs` and retains its
conceptual positive-zero [padding](../../../../glossary.md#padding) meaning. The new overload additionally null-checks
`paddingValue`, requires `paddingValue.dataType()` to equal the input [DataType](../../../../glossary.md#data-type--datatype) exactly, constructs
one `Unfold2dAttrs` retaining both supplied references, and preserves every raw [scalar](../../../../glossary.md#term-scalar) bit.

Exact type failure:

```text
unfold2d paddingValue data type must match input data type: paddingValue=<type>, input=<type>
```

No [validation](../../../../glossary.md#term-validation) failure consumes a [Tensor](../../../../glossary.md#tensor) ID. Every successful call consumes exactly one.

### Dynamic [canonical](../../../../glossary.md#term-canonical-form) 2D fold

Preserve the public [signature](../../../../glossary.md#term-signature) and [validation](../../../../glossary.md#term-validation) prefix: input/output/window null checks, [rank](../../../../glossary.md#term-rank)-three
canonical columns, rank-four [NCHW](../../../../glossary.md#term-nchw) [target](../../../../glossary.md#term-target), floating type, and exact [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[Dimension](../../../../glossary.md#dimension) equality.

Derive expected column channels and expected column count from the exact target [Shape](../../../../glossary.md#shape) and window
using the same canonical formulas as unfold. Static invalid target geometry still fails locally.
Require exact [structural equality](../../../../glossary.md#term-structural-equality) for both complete expected Dimensions:

```text
actual columnChannels == targetChannels * kernelHeight * kernelWidth
actual columnCount    == outputHeight * outputWidth
```

Reuse the current mismatch-message forms with Dimension diagnostics in place of static longs.
Unprovable unrelated symbols are rejected rather than silently recorded as an unnamed equality
constraint. A [Tensor](../../../../glossary.md#tensor) returned by a matching dynamic `unfold2d` therefore composes with `fold2d`
exactly, while an arbitrary unresolved column Shape must state the same structural formulas.

The result retains the exact supplied output Shape reference, input floating [DataType](../../../../glossary.md#data-type--datatype) and
`requiresGrad`, unresolved [layout](../../../../glossary.md#layout), exact `FOLD2D/Fold2dAttrs`, `[input]` [producer](../../../../glossary.md#term-producer), output index
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

Documentation and [planning](../../../../glossary.md#term-planning):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a contradiction requires stopping: [Shape](../../../../glossary.md#shape)/[Dimension](../../../../glossary.md#dimension) public
contracts outside the two listed Shape files, [DataType](../../../../glossary.md#data-type--datatype)/[ScalarValue](../../../../glossary.md#typed-scalar-value--scalarvalue), TensorDescriptor,
TensorFactory, [TensorProducer](../../../../glossary.md#producer--tensorproducer)/TensorProvenance, [Operation](../../../../glossary.md#operation), [convolution](../../../../glossary.md#term-channels-first-convolution-layers)/[pooling](../../../../glossary.md#term-pooling) contracts and tests,
Training/[Runtime](../../../../glossary.md#term-runtime) APIs, architecture/ADRs/tests, [Gradle](../../../../glossary.md#term-gradle)/dependencies, other [modules](../../../../glossary.md#term-module), [backend](../../../../glossary.md#backend)
[conformance](../../../../glossary.md#term-conformance), integration tests, completed task history, and later tasks.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify exactly 33 paths: nine production files, seventeen tests, and seven
documentation/[planning](../../../../glossary.md#term-planning) files. The count includes thirteen existing global [Tensor](../../../../glossary.md#tensor)-surface locks;
those are mechanical count-only edits except for `TensorTest`.

This is an explicit [atomic](../../../../glossary.md#resource-transaction-and-rollback) exception to the normal 18-path guardrail. The symbolic product,
dynamic [rank](../../../../glossary.md#term-rank)-three column [Shape](../../../../glossary.md#shape), configurable [padding](../../../../glossary.md#padding) attributes, public overloads, restored
foldAxis, [signatures](../../../../glossary.md#term-signature), tests, and documentation form one contract. Splitting them would either
publish unused [nonlinear](../../../../glossary.md#term-nonlinear) Shape syntax, expose dynamic unfold without an exact fold, or temporarily
leave public/signature inventories inconsistent.

Do not use this allowance for unrelated cleanup. Stop if a thirty-fourth path, another type or
test, a completed-contract migration beyond the named files, another [module](../../../../glossary.md#term-module), [dependency](../../../../glossary.md#term-dependency), [Gradle](../../../../glossary.md#term-gradle),
architecture change, or detailed 0023E specification is required.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Document Product purpose, immutable factors/coefficient, canonicalization, identity behavior,
  overflow, equality, diagnostics, lack of binding/evaluation, and every accessor/factory input,
  result, and failure.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [identity glossary](../../../../glossary.md#term-identity-meanings); [overflow glossary](../../../../glossary.md#term-overflow); [binding glossary](../../../../glossary.md#term-binding); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- Document `Unfold2dAttrs` exact padding region, exact typed bits, reference retention, accepted
  structural values, operation pairing, and absence of input-aware validation.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [validation glossary](../../../../glossary.md#term-validation))</span>
- Update all affected window Javadocs from compiler-only/static wording to the exact current public
  and dynamic boundaries without rewriting unchanged declarations.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler))</span>
- Document raw versus normalized fold axes, target rank, overlap summation, zero-output geometry,
  accepted types, Shape derivation, provenance, unresolved layout, freshness, ID effects, and
  failures.<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis); [target glossary](../../../../glossary.md#term-target); [rank glossary](../../../../glossary.md#term-rank); [Shape glossary](../../../../glossary.md#shape); [provenance glossary](../../../../glossary.md#provenance); [layout glossary](../../../../glossary.md#layout))</span>
- Document the direct positive-zero and explicit typed-[padding](../../../../glossary.md#padding) `unfold2d` variants distinctly.
- Explain canonical im2col order and symbolic channel/window products with a concrete dynamic
  example such as <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [im2col glossary](../../../../glossary.md#term-im2col); [channel glossary](../../../../glossary.md#term-channel))</span>`[N,C,H,W]` and a 3-by-3 [kernel](../../../../glossary.md#kernel).
- Explain that configurable padding includes ceil-grid tail positions and preserves exact NaN,
  infinity, and signed-zero bits without selecting max-pool or backend behavior.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [backend glossary](../../../../glossary.md#backend))</span>
- Explain exact structural dynamic `fold2d` compatibility and why unrelated symbols are rejected.
- Review unchanged Shape, ScalarValue, descriptor, producer/provenance, convolution, pooling, and
  public window contracts; record reasoned no-change conclusions or stop on an out-of-scope
  discrepancy.<span data-glossary-navigation="true"> ([ScalarValue glossary](../../../../glossary.md#typed-scalar-value--scalarvalue); [descriptor glossary](../../../../glossary.md#term-descriptor); [producer glossary](../../../../glossary.md#term-producer); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [pooling glossary](../../../../glossary.md#term-pooling); [scope glossary](../../../../glossary.md#term-scope))</span>

## Acceptance criteria

- The [sealed](../../../../glossary.md#term-sealed) expression hierarchy contains exactly the previous four forms plus Product.
- Product and both multiply overloads satisfy the exact surface, validation, canonicalization,
  value, immutability, reference, overflow, and diagnostic contracts.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [overflow glossary](../../../../glossary.md#term-overflow))</span>
- `Unfold2dAttrs` has exactly the two ordered components and no additional instance state or public
  API beyond record members and documented accessors.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Window kind order is unchanged and only UNFOLD2D gains the exact second [signature](../../../../glossary.md#term-signature) variant.
- [Tensor](../../../../glossary.md#tensor) adds exactly `foldAxis(int,long,long)` and
  `unfold2d(Window2dAttrs,ScalarValue)`; the public count is exactly 194.
- Restored foldAxis matches the retained semantics and historical completed validation contract,
  accepts five numeric types, rejects BOOL, preserves unaffected Dimensions, and creates one fresh
  result without execution.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [Dimensions glossary](../../../../glossary.md#dimension))</span>
- Existing static unfold2d/fold2d behavior and direct positive-zero attributes remain unchanged.
- Both unfold2d variants derive exact static or symbolic rank-three canonical columns without an
  unnamed unknown Dimension.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Configurable unfold2d retains exact Window2dAttrs/ScalarValue references and bits and rejects a
  padding/input type mismatch before ID allocation.<span data-glossary-navigation="true"> ([ScalarValue glossary](../../../../glossary.md#typed-scalar-value--scalarvalue); [padding glossary](../../../../glossary.md#padding))</span>
- Fold2d accepts exact matching dynamic canonical columns, rejects structurally unprovable
  mismatch, and retains the exact target Shape.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [Shape glossary](../../../../glossary.md#shape))</span>
- Every successful expression preserves the specified DataType/requiresGrad, unresolved layout,
  one producer, ordered <span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [layout glossary](../../../../glossary.md#layout); [producer glossary](../../../../glossary.md#term-producer))</span>`[input]`, output index zero, no label/storage, and one fresh ID.
- Failures follow exact order/messages and consume no ID. No values or storage are inspected.
- Focused tests, exactly one final model suite after executable stabilization, model Javadoc,
  runnable Java 26 example, Markdown/link/anchor checks, exact 33-path scope, status, and<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
  `git diff --check` pass.
- A separate clean documentation-focused agent finalizes every affected Javadoc, API explanation,
  glossary term, planning record, and validation result without repeating successful Java tests
  unless executable behavior changes or a concrete risk is recorded.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Task 0023D becomes Complete only after both passes. Task 0023E remains Draft without a detailed
  specification. Architecture, dependencies, Gradle, other modules, compiler/runtime/backend, and
  completed unrelated contracts remain unchanged.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

During development [run](../../../../glossary.md#term-run) the focused affected contracts as needed:

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

Tests must cover exact reflection/surface [shape](../../../../glossary.md#shape), product static/[identity](../../../../glossary.md#term-identity-meanings)/nested/[commutative](../../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks)/
repeated-factor/[overflow](../../../../glossary.md#term-overflow)/value/diagnostic behavior, record [signatures](../../../../glossary.md#term-signature), every [DataType](../../../../glossary.md#data-type--datatype), foldAxis
[axes](../../../../glossary.md#term-axis)/zero/count/overflow, static and dynamic floor/ceil 2D geometry, exact configurable [scalar](../../../../glossary.md#term-scalar)
bits/type mismatch, structural fold compatibility, result [metadata](../../../../glossary.md#term-metadata), [producer](../../../../glossary.md#term-producer)/[provenance](../../../../glossary.md#provenance), storage
non-interference, freshness, and ID side effects.

The separate documentation pass reuses successful Java evidence unless it changes executable
behavior. After final Javadocs it runs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

It also compiles/runs one Java 26 metadata example, verifies generated [Javadoc](../../../../glossary.md#term-javadoc) and the 194-method
[Tensor](../../../../glossary.md#tensor) surface, validates all changed Markdown links/anchors/fences/newlines/terminology, confirms
exact 33-path [scope](../../../../glossary.md#term-scope) and no 0023E specification, and records no-change conclusions.

Repository-wide tests are deferred to the next recorded [model](../../../../glossary.md#term-model) capability [checkpoint](../../../../glossary.md#term-checkpoint) or [CI](../../../../glossary.md#term-continuous-integration-ci) because
this task changes one [module](../../../../glossary.md#term-module) without dependencies, [Gradle](../../../../glossary.md#term-gradle), module boundaries, or architecture.

## Dependencies

- Task 0002 and completed task 0018M for [Shape](../../../../glossary.md#shape), [Dimension](../../../../glossary.md#dimension), and [canonical](../../../../glossary.md#term-canonical-form) symbolic [extent](../../../../glossary.md#term-extent) values.
- Tasks 0005–0007 and 0018K for typed [operation](../../../../glossary.md#operation)/[signature](../../../../glossary.md#term-signature) construction.
- Tasks 0011–0013 and 0018L for [Tensor](../../../../glossary.md#tensor) [descriptors](../../../../glossary.md#term-descriptor), [identity](../../../../glossary.md#term-identity-meanings), [producers](../../../../glossary.md#term-producer), and [provenance](../../../../glossary.md#provenance).
- Tasks 0017M–0017N for retained window meanings and the historical foldAxis contract.
- Task 0018N for exact typed [scalar](../../../../glossary.md#term-scalar) values.
- Task 0018R for the current signed-[slice](../../../../glossary.md#slice)/window public boundary and retained FOLD_AXIS semantics.
- Tasks 0020–0020A1 for the dynamic [convolution](../../../../glossary.md#term-channels-first-convolution-layers)/[pooling](../../../../glossary.md#term-pooling) geometry that proves the window need.
- Task 0023 and its completed audit [matrix](../../../../glossary.md#term-matrix-meanings). Tasks 0023A–0023C are complete preceding frontiers.

## Follow-up tasks

- Task 0023E remains Draft for cumulative-product [scan](../../../../glossary.md#term-cumulative-scan) semantics and [Tensor](../../../../glossary.md#tensor) construction.
- Task 0023F remains Draft for same-[occurrence](../../../../glossary.md#term-occurrence) attention [weights](../../../../glossary.md#term-weight).
- Task 0024 remains Draft and depends on all task-0023 follow-ups.
- Later compiler work may compose these public transformations into adjoints and own binding,
  capture, saved-value lifetime, gradient accumulation, and optimization.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [binding glossary](../../../../glossary.md#term-binding); [capture glossary](../../../../glossary.md#term-capture-meanings); [lifetime glossary](../../../../glossary.md#term-lifetime); [gradient accumulation glossary](../../../../glossary.md#term-gradient-accumulation))</span>
- Later planning/prepare/backend work owns algorithms, materialization, kernels, execution, and
  performance.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [materialization glossary](../../../../glossary.md#materialization); [kernels glossary](../../../../glossary.md#kernel))</span>

## Architecture impact

Expected impact: None.

The task extends [model](../../../../glossary.md#term-model)-owned [Shape](../../../../glossary.md#shape) values and public [backend](../../../../glossary.md#backend)-independent [Tensor](../../../../glossary.md#tensor) semantics without
changing [ownership](../../../../glossary.md#term-ownership-meanings) or dependencies. If implementation requires a solver, [compiler](../../../../glossary.md#term-compiler) constraint
object, [runtime](../../../../glossary.md#term-runtime) [binding](../../../../glossary.md#term-binding), backend policy, another [module](../../../../glossary.md#term-module), or authoritative architecture edit, stop
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
  more coherent than introducing a second public rank-five/rank-six window representation.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [rank glossary](../../../../glossary.md#term-rank); [im2col glossary](../../../../glossary.md#term-im2col); [col2im glossary](../../../../glossary.md#term-col2im); [Shapes glossary](../../../../glossary.md#shape))</span>
- Product canonicalization uses a coefficient plus factor/exponent map so multiplication is
  order-independent and nested/repeated products remain structurally stable.<span data-glossary-navigation="true"> ([canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [exponent glossary](../../../../glossary.md#term-exponent))</span>
- Keep the existing zero-padding UNFOLD2D attributes variant unchanged. An explicit typed-padding
  overload is additive and does not migrate completed producer metadata.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [producer glossary](../../../../glossary.md#term-producer); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Require exact structural dynamic fold compatibility. This model task does not invent an equality
  constraint registry for unrelated symbols.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [registry glossary](../../../../glossary.md#term-registry))</span>
- Restore only the historical long-output-size foldAxis contract. Current unfold still requires a
  static selected extent, so no dynamic output-size attribute is needed for its exact adjoint.<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent); [adjoint glossary](../../../../glossary.md#term-adjoint))</span>
- The 33-path scope is an intentional atomic exception dominated by thirteen mandatory public
  Tensor count locks.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [Tensor glossary](../../../../glossary.md#tensor))</span>

## Known limitations

- Product expressions retain symbolic multiplication but do not bind or evaluate it.
- Dynamic formulas can retain a validity obligation such as a non-negative numerator; this task
  does not prove it before concrete binding.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- Fold2d rejects unrelated unresolved column symbols even if an external caller knows they will
  later bind to equal products.
- Window transforms remain metadata only. No current backend is promised to execute the new
  variant, and no compiler rule is added.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [backend glossary](../../../../glossary.md#backend); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Configurable [padding](../../../../glossary.md#padding) is a single exact [scalar](../../../../glossary.md#term-scalar), not a padding mode or [Tensor](../../../../glossary.md#tensor)-valued boundary.

## [Validation](../../../../glossary.md#term-validation) evidence

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
  planning files.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Planning glossary](../../../../glossary.md#term-planning))</span>
- `./gradlew :modules:model:javadoc` passed after final [Javadoc](../../../../glossary.md#term-javadoc) edits (`BUILD SUCCESSFUL`; two
  executed tasks). Generated pages exist for Product, both Dimension factory contracts,
  `Unfold2dAttrs`, all window kinds/attributes, [`Tensor`](../../../../glossary.md#tensor), and the construction helper. A targeted
  `rg` generated-page scan found `foldAxis`, the typed `unfold2d` overload, Product,
  `Unfold2dAttrs`, and canonical-symbolic wording. A later redundant Javadoc invocation inside a
  combined final command could not open the Gradle user-home lock under the filesystem sandbox;
  rerunning <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [Gradle glossary](../../../../glossary.md#term-gradle))</span>`./gradlew :modules:model:javadoc` with the approved Gradle permission passed with both
  tasks up to date. After the final constant-product Javadoc clarification, the approved command
  passed again with compileJava and Javadoc executed successfully.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>
- `javac --release 26 -cp modules/model/build/classes/java/main -d
  /tmp/synaptik-0023d-example /tmp/TensorWindowExpressionExample.java && java -cp
  modules/model/build/classes/java/main:/tmp/synaptik-0023d-example
  TensorWindowExpressionExample` passed. Its exact output was `Shape[2, 3, 3, 3]`, `Shape[5]`,
  `Shape[1, 4, 4]`, `Shape[1, 1, 3, 3]`, then four `true` lines confirming exact padding-reference
  retention, dynamic <span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding))</span>`9 * C`, exact dynamic fold target retention, unresolved layout, and absent
  result storage.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [layout glossary](../../../../glossary.md#layout))</span>
- The first [compile](../../../../glossary.md#compile) of `/tmp/Task0023dSurfaceCheck.java` failed only because the validation helper
  declared an invariant generic <span data-glossary-navigation="true"> ([invariant glossary](../../../../glossary.md#term-invariant))</span>`List<Class<?>>` for a captured class-token list. After correcting
  that helper-only type to <span data-glossary-navigation="true"> ([token glossary](../../../../glossary.md#term-token))</span>`List<? extends Class<?>>`, the same `javac --release 26` and `java`
  check passed. It confirmed exactly 194 declared public Tensor methods; exact public `foldAxis`
  and typed `unfold2d` [signatures](../../../../glossary.md#term-signature); public `multiply(Dimension,Dimension)`; Product in the sealed
  hierarchy; ordered UNFOLD2D attribute variants <span data-glossary-navigation="true"> ([sealed glossary](../../../../glossary.md#term-sealed))</span>`[Window2dAttrs, Unfold2dAttrs]`; and exact
  `Unfold2dAttrs` components `window:Window2dAttrs,paddingValue:ScalarValue`.
- The first `python3 /tmp/validate_synaptik_markdown.py` run identified a checker-only mismatch in
  GitHub slug handling for adjacent spaces around punctuation. After correcting that temporary
  validator, the same command passed. The final run after planning-status edits covered 208
  Markdown files, 3,462 local links, 205 local anchors, 2,606 fence markers, final newlines, and
  trailing whitespace. Targeted terminology scans found no stale compiler-only <span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [compiler glossary](../../../../glossary.md#term-compiler))</span>`foldAxis`,
  static-only 2D-window, four-form symbolic-expression, or single-UNFOLD2D-signature claim.
- `{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u` produced exactly
  the authorized 33 paths: nine production Java files, seventeen tests, and seven documentation/
  planning files. `git diff --check` passed. Status checks confirm task 0023D and its master-plan/
  roadmap rows are Complete, task 0023E and later rows remain Draft, and no 0023E task
  specification exists.
- No-change review concluded that [Shape](../../../../glossary.md#shape)/[Dimension](../../../../glossary.md#dimension) contracts outside Product and its factories,
  [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue), `TensorDescriptor`, `TensorFactory`, producer/provenance, convolution/pooling,
  Training/Runtime APIs, architecture/ADRs/tests, Gradle/dependencies, backend conformance,
  integration tests, other modules, completed task history, and later tasks remain accurate.
  The change adds model-owned immutable formulas, attributes, and storage-free expression
  construction only; it changes no descriptor ownership, scalar representation, producer model,
  numerical algorithm, compiler capture, gradient, runtime, backend, execution, dependency, or
  architecture contract.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [pooling glossary](../../../../glossary.md#term-pooling); [Runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [modules glossary](../../../../glossary.md#term-module); [model glossary](../../../../glossary.md#term-model); [immutable glossary](../../../../glossary.md#term-immutable); [descriptor glossary](../../../../glossary.md#term-descriptor); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [scalar glossary](../../../../glossary.md#term-scalar); [capture glossary](../../../../glossary.md#term-capture-meanings); [gradient glossary](../../../../glossary.md#term-gradient); [dependency glossary](../../../../glossary.md#term-dependency); [architecture contract glossary](../../../../glossary.md#architecture-contract))</span>

## Implementation notes

- Added `DimensionExpression.Product` and public canonical Dimension-to-Dimension multiplication;
  the existing constant-factor path now scales a Product coefficient without nesting it as a
  linear term.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [Dimension glossary](../../../../glossary.md#dimension); [constant glossary](../../../../glossary.md#term-constant); [linear glossary](../../../../glossary.md#term-linear-meanings))</span>
- Added `Unfold2dAttrs` and the second ordered UNFOLD2D [signature](../../../../glossary.md#term-signature) while preserving the direct
  `Window2dAttrs` conceptual-positive-zero variant and every kind order.
- Restored public `foldAxis`, added exact typed-[padding](../../../../glossary.md#padding) `unfold2d`, and generalized canonical
  rank-three 2D window formulas and exact fold compatibility to unresolved Dimensions.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank))</span>
- Updated all global public-Tensor surface locks from 192 to 194. No executable Java change
  occurred after the recorded final model suite.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [model glossary](../../../../glossary.md#term-model))</span>

## Completion summary

Completed changes: added [canonical](../../../../glossary.md#term-canonical-form) symbolic [Dimension](../../../../glossary.md#dimension) products, exact typed-[padding](../../../../glossary.md#padding) UNFOLD2D
[metadata](../../../../glossary.md#term-metadata), restored public general-[axis](../../../../glossary.md#term-axis) overlap fold, and dynamic canonical [rank](../../../../glossary.md#term-rank)-three
unfold2d/fold2d [Shape](../../../../glossary.md#shape) construction with exact structural compatibility.

Files changed or created: exactly the nine production, seventeen test, and seven documentation/
[planning](../../../../glossary.md#term-planning) paths listed under Affected files.

Tests or [validation](../../../../glossary.md#term-validation) performed: focused 17-suite [run](../../../../glossary.md#term-run) (175 tests) and one final [model](../../../../glossary.md#term-model) suite (1,008
tests across 126 suites) from the implementation context; model [Javadoc](../../../../glossary.md#term-javadoc); runnable Java 26 metadata
example; Java 26 reflection and generated-Javadoc/[API](../../../../glossary.md#term-application-programming-interface-api)-surface checks; Markdown links, anchors,
fences, newlines, whitespace, and terminology; exact 33-path inventory; synchronized status; no
0023E specification; and `git diff --check` all passed.

Documentation-agent review: clean context `/root/implement_0023d/docs_0023d` independently
finalized affected Javadocs, [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile) API explanations, glossary, capability baseline, task,
master plan, roadmap, and documentation evidence without rerunning successful Java tests.

Documentation impact: Tensor and Compile API references now distinguish public foldAxis, direct
positive-zero versus exact typed-padding unfold, canonical symbolic products, dynamic 2D formulas,
and exact structural fold compatibility while preserving current-versus-planned boundaries.

Javadoc review: all nine authorized production paths are current, including the unchanged-behavior
`Window2dAttrs`, `FoldAxisAttrs`, and `Fold2dAttrs` declarations.

Glossary impact: symbolic [extent](../../../../glossary.md#term-extent) expressions and window transforms now define canonical symbolic
products, both UNFOLD2D attribute variants, public foldAxis, dynamic canonical columns, and
structural fold compatibility.

Unresolved issues: None.

Required follow-up: None for task 0023D. Task 0023E remains the next Draft planning frontier
without a detailed specification.

Status: Complete
