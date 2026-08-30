# Task 0025K: Public NCDHW unfold3d and fold3d window transforms

## Status

Complete

## Goal

Add the smallest general public three-dimensional window algebra needed for exact NCDHW Pool3d
adjoints without adding pooling-specific backward operations:

```text
[N, C, D, H, W]
  -- unfold3d(window[, paddingValue]) -->
[N, C * kD * kH * kW, D_out * H_out * W_out]
  -- fold3d([N, C, D, H, W], window) -->
[N, C, D, H, W]
```

The rank-three column representation extends the current canonical NCHW `unfold2d`/`fold2d`
contract by one spatial axis. Direct unfold uses conceptual typed positive-zero padding; the
explicit overload retains one exact typed padding value, including negative infinity for later
maximum-pooling winner reconstruction. Fold excludes out-of-range padded coordinates and
overlap-adds every in-range contribution into the exact caller-supplied NCDHW target Shape.

This task owns Model semantics, local Shape validation, public Tensor construction, and canonical
one-input provenance only. It creates no Pool3d gradient, Compiler adoption, backend capability,
materialization, execution, or performance claim.

## Scope

- Add public immutable `Window3dAttrs` for depth-height-width kernel, stride, symmetric padding,
  dilation, and literal ceil-mode geometry.
- Add public immutable `Unfold3dAttrs(Window3dAttrs window, ScalarValue paddingValue)` for exact
  typed out-of-domain samples.
- Add public immutable `Fold3dAttrs(Shape outputShape, Window3dAttrs window)` for the exact NCDHW
  fold target.
- Append `UNFOLD3D` and `FOLD3D` to `WindowTransformKind` without changing the existing four
  constants, their order, or their signatures.
- Add exactly these public Tensor methods:

  ```java
  public Tensor unfold3d(Window3dAttrs window)
  public Tensor unfold3d(Window3dAttrs window, ScalarValue paddingValue)
  public Tensor fold3d(Shape outputShape, Window3dAttrs window)
  ```

- Extend the existing package-private, final, field-free `TensorWindowExpressions` owner with
  rank-five NCDHW construction and exact rank-three column compatibility validation.
- Accept BFLOAT16, FLOAT32, and FLOAT64 only, matching current `unfold2d`/`fold2d`.
- Preserve exact static or canonical-symbolic depth, height, and width geometry, symmetric
  padding, dilation, stride, and literal floor/ceil grids.
- Fix the canonical column Shape, flattening order, out-of-domain padding meaning, fold exclusion,
  overlap addition, positive-zero initialization, validation order, allocation effects, and
  canonical provenance below.
- Update exact operation/signature inventories and every current global public-Tensor method-count
  lock from 210 to 213.
- Finalize affected Javadocs, Tensor and Compile API explanations, glossary terminology,
  capabilities, and planning in a separate clean documentation-focused context before completion.

## Out of scope

- Pool3d gradient formulas, derivative policy, Compiler 0006B1 or 0006B2 implementation, saved
  maximum indices, unpooling, a pooling-backward kind, or another pooling-specific primitive.
- CPU 0008G1 or any backend lowering, capability advertisement, algorithm, generated code,
  materialization, workspace, Runtime behavior, execution, conformance, integration, or
  performance evidence.
- A public or private generic `WindowNd`, `PoolNd`, arbitrary-rank transform, dynamic-rank API,
  geometry arrays, rank-polymorphic attributes, broad window facade, manager, or utility.
- Reusing or modifying `MaxPool3dAttrs` or `AveragePool3dAttrs` as general window metadata; their
  excluded-padding extrema and fixed-divisor average policies remain pooling-specific.
- Changing the existing `UNFOLD_AXIS`, `FOLD_AXIS`, `UNFOLD2D`, or `FOLD2D` declarations,
  signatures, attributes, Shape ordering, padding meanings, numerical semantics, or public APIs.
- Integral or BOOL 3D transforms, FLOAT16, implicit conversion, promotion, output-type override,
  asymmetric intrinsic padding, padding modes, Tensor-valued padding, valid-sample counting, or
  overlap averaging.
- A second rank-five/rank-seven window carrier, an unflattened public patch tensor, an inferred
  fold target Shape, or unnamed equality constraints for unrelated unresolved Dimensions.
- Value reads, eager evaluation, host allocation, attached result storage, resolved result layout,
  mutation, alias promises, or input materialization.
- Changes to Shape/Dimension expression types, ScalarValue representation, TensorFactory seams,
  Pool3d/Pool2d/Conv3d behavior, dependencies, Gradle, architecture contracts, ADRs, architecture
  tests, other modules, or detailed specifications for Compiler 0006B1/0006B2, CPU 0008G1, or
  Model 0026.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Model master plan](../master-plan.md)
- [Model capabilities](../capabilities.md)
- [Public foldAxis and dynamic 2D windows](0023d-public-fold-axis-and-dynamic-window-transforms.md)
- [NCHW Max Pool2d](0020a-nchw-max-pool2d-semantics-and-tensor-expression.md)
- [NCHW Average Pool2d](0020a1-nchw-average-pool2d-semantics-and-tensor-expression.md)
- [First-class NCDHW Pool3d](0025j-first-class-ncdhw-max-average-pool3d-semantics.md)
- [Compiler Pool2d gradients](../../compiler/tasks/0005d-attention-convolution-pooling-and-loss-gradient-completion.md)
- [CPU overlap fold](../../../backends/cpu/tasks/0006b2-portable-overlap-fold.md)

## Architecture constraints

- Work stays in `modules/model` plus directly affected documentation and planning. Model owns
  immutable backend-independent operation meanings, attributes, Tensor descriptors, public
  expression construction, and pre-capture producer metadata.
- Tensor remains public mutable API state, not graph IR. Every successful call returns one fresh
  storage-free canonical Tensor wrapper; no method evaluates a value.
- Window kinds and attributes contain no Tensor reference except the established immutable Shape
  and ScalarValue semantic values, and no graph, compiler, planning, backend, prepared, runtime,
  kernel, route, storage, or execution state.
- Direct positive-zero padding and explicit typed padding are semantic extraction values, not
  backend fill policies. Fold padding exclusion is geometric and never inferred by comparing
  column values with a padding scalar.
- Result layout remains unresolved because materialized windows and overlap accumulation do not
  establish alias-view geometry.
- Every derived occurrence uses the existing factory-atomic canonical-output seam. A returned
  Tensor is exactly producer `output(0)` with output index zero.
- Compiler owns Pool3d forward adoption, retained binding proof, gradient construction, and
  higher-order closure. Backends own algorithms and execution. This task must not move either
  responsibility into Model.
- No architecture, dependency, module-boundary, lifecycle, build, conformance, or integration
  contract changes. Stop if implementation requires one.

## Package impact

Existing packages used and changed:

- `io.github.pho001.synaptik.model.operation.layout` — rank-specific immutable window meanings and
  attributes.
- `io.github.pho001.synaptik.model.tensor` — established public Tensor facade and package-private
  window-expression construction owner.
- `io.github.pho001.synaptik.model.shape` — existing static and canonical symbolic Dimension
  arithmetic; no new expression form.
- `io.github.pho001.synaptik.model.datatype` — existing floating-type and exact ScalarValue
  contracts.

Packages added or moved: None.

Type placement:

- `...operation.layout.Window3dAttrs` — reusable rank-specific three-dimensional window geometry,
  distinct from pooling-specific attrs.
- `...operation.layout.Unfold3dAttrs` — exact typed-padding `UNFOLD3D` metadata beside the existing
  2D counterpart.
- `...operation.layout.Fold3dAttrs` — exact target Shape and geometry for `FOLD3D`.
- `...operation.layout.WindowTransformKind` — existing window identity/signature owner, extended
  by the two rank-specific meanings.
- `...tensor.TensorWindowExpressions` — existing cohesive owner for input-aware window validation,
  Shape derivation, descriptors, operations, and factory delegation.
- `...tensor.Tensor` — established fluent public receiver surface.

Tests continue to mirror the production packages. No generic helper package or second Tensor
facade is added.

## Required contracts

### Rank-specific attributes

Add exactly:

```java
public record Window3dAttrs(
        long kernelDepth,
        long kernelHeight,
        long kernelWidth,
        long strideDepth,
        long strideHeight,
        long strideWidth,
        long paddingDepth,
        long paddingHeight,
        long paddingWidth,
        long dilationDepth,
        long dilationHeight,
        long dilationWidth,
        boolean ceilMode) implements OperationAttrs

public record Unfold3dAttrs(
        Window3dAttrs window,
        ScalarValue paddingValue) implements OperationAttrs

public record Fold3dAttrs(
        Shape outputShape,
        Window3dAttrs window) implements OperationAttrs
```

`Window3dAttrs` validates components in declaration order. Kernel, stride, and dilation values
must be positive; padding values must be non-negative. Failures use the established exact forms:

```text
<component> must be positive: <value>
<component> must be non-negative: <value>
```

It performs no input-aware rank, Shape, type, fit, product, or arithmetic validation. In
particular, it does not precompute the three-factor kernel volume.

`Unfold3dAttrs` null-checks `window` then `paddingValue`, retains both exact references, and
performs no input-aware type compatibility. The padding scalar supplies every sampled coordinate
outside the logical unpadded NCDHW input, including leading/trailing symmetric padding and every
terminal ceil-grid coordinate beyond the padded extent. Its exact type and primitive bits are
semantic metadata.

`Fold3dAttrs` null-checks `outputShape` then `window`, retains both exact immutable references, and
performs no rank, type, geometry, or column-compatibility validation.

Each record has only its declared components and ordinary record value semantics. Add no builder,
defaults, arrays, nested types, normalization method, pooling conversion method, or mutable state.

### Kinds and signatures

Append the new constants after the existing four values. Exact enum order becomes:

```text
UNFOLD_AXIS
FOLD_AXIS
UNFOLD2D
FOLD2D
UNFOLD3D
FOLD3D
```

Exact signature pairings are:

```text
UNFOLD_AXIS + UnfoldAxisAttrs: one input, one output
FOLD_AXIS   + FoldAxisAttrs:   one input, one output
UNFOLD2D    + Window2dAttrs:   one input, one output
UNFOLD2D    + Unfold2dAttrs:   one input, one output
FOLD2D      + Fold2dAttrs:     one input, one output
UNFOLD3D    + Window3dAttrs:   one input, one output
UNFOLD3D    + Unfold3dAttrs:   one input, one output
FOLD3D      + Fold3dAttrs:     one input, one output
```

`UNFOLD3D.signatures()` lists direct `Window3dAttrs` first and `Unfold3dAttrs` second. Existing
signature lists remain unchanged. The Model compiled inventory becomes exactly 40 operation-kind
families, 115 constants, and 137 signatures; tests must derive and lock the new window-family
surface rather than altering unrelated family contracts.

### Public Tensor surface

Add exactly the three methods listed in Scope. Each delegates once to the matching
`TensorWindowExpressions` overload and owns no validation, Shape arithmetic, operation
construction, factory call, default, or alias. The declared public Tensor method count becomes
exactly 213. Add only public names `unfold3d` and `fold3d`; `unfold3d` has exactly the two specified
overloads and `fold3d` exactly one.

### NCDHW geometry and canonical column Shape

Input is rank-five NCDHW `[N, C, D, H, W]`. For each spatial input Dimension `X`, positive kernel
sample count `k`, non-negative symmetric padding per side `p`, positive dilation `d`, and positive
stride `s`:

```text
effectiveKernel = d * (k - 1) + 1
numerator       = X + 2 * p - effectiveKernel
floor output    = floor(numerator / s) + 1
ceil output     = ceil(numerator / s) + 1
```

All literal constants and static arithmetic use checked signed-`long` operations. Static negative
numerators fail. For unresolved Dimensions, use the existing canonical `addConstant`, matching
floor/ceiling divide, and `addConstant(1)` forms; retain the non-negative-numerator obligation for
later Compiler/binding proof. Derive and validate depth, then height, then width. Literal ceil mode
never decrements a terminal window, including an all-padding window.

The canonical unfold result Shape is exactly:

```text
[N, C * kernelDepth * kernelHeight * kernelWidth,
    outputDepth * outputHeight * outputWidth]
```

Retain the exact input batch Dimension reference. Form both products through successive existing
`DimensionExpressions.multiply` calls in written order; do not precompute `kernelDepth *
kernelHeight * kernelWidth` as a host `long`. A static or symbolic product fails only under the
existing Dimension representability rules. The result is rank three even though the source is
rank five; no alternate patch Shape is exposed.

### Deterministic coordinate ordering

For output-grid coordinates `(od, oh, ow)` and kernel coordinates `(kd, kh, kw)`, define:

```text
id = od * strideDepth  - paddingDepth  + kd * dilationDepth
ih = oh * strideHeight - paddingHeight + kh * dilationHeight
iw = ow * strideWidth  - paddingWidth  + kw * dilationWidth

q = (((c * kernelDepth + kd) * kernelHeight + kh) * kernelWidth + kw)
p = ((od * outputHeight + oh) * outputWidth + ow)
```

The column coordinate is `[n, q, p]`. Thus channel is outermost in `q`, then kernel depth, kernel
height, and kernel width; width is fastest. In `p`, output depth is outermost, then output height,
then output width; width is fastest. Unfold uses increasing `(n, c, kd, kh, kw, od, oh, ow)`
canonical row-major column order as defined by `[n,q,p]`. This fixed structural order is the later
Pool3d maximum first-winner basis and must not be replaced by height-first, width-first, or
backend-dependent ordering.

The direct `unfold3d(window)` supplies exact represented positive zero for every coordinate where
`id`, `ih`, or `iw` is outside its unpadded input extent. The typed overload supplies the exact
retained `paddingValue` instead. In-range samples retain their exact represented input values.
Unfold performs no arithmetic on sampled values.

### Exact NCDHW fold compatibility

`fold3d` consumes rank-three canonical columns and an explicit rank-five NCDHW target Shape.
Preserve the null/rank/type/batch validation prefix specified below. Derive:

```text
expectedColumnChannels = targetC * kernelDepth * kernelHeight * kernelWidth
expectedColumnCount    = targetDOut * targetHOut * targetWOut
```

where each target output-grid Dimension uses the exact same target spatial extent, geometry, and
literal floor/ceil formula as unfold. Require complete structural equality with input Dimensions
one and two. Matching dynamic columns produced by `unfold3d` compose exactly; unrelated unresolved
symbols fail rather than creating an unnamed equality constraint. The result descriptor and
`Fold3dAttrs` both retain the exact supplied `outputShape` reference.

For each logical column value `[n,q,p]`, decode the same `(c,kd,kh,kw)` and `(od,oh,ow)` order and
compute `(id,ih,iw)` above. Contribute exactly once to `[n,c,id,ih,iw]` only when all three spatial
coordinates are in bounds. Every padded or terminal ceil-tail coordinate is excluded
geometrically regardless of its column value. Fold has no padding scalar and never compares a
value to decide inclusion.

### Fold overlap addition

Every target coordinate begins at exact represented positive zero. A target with no in-range
contribution retains that zero. Multiple logical column positions mapping to the same target are
distinct contributions and are added in increasing canonical flattened input order `[n,q,p]`.
No reassociation, tree reduction, averaging, compensation, saturation, or hidden widening is part
of the semantic contract.

Type-specific addition is:

- FLOAT64: sequential IEEE-754 binary64 addition after each contribution;
- FLOAT32: sequential IEEE-754 binary32 addition after each contribution; and
- BFLOAT16: expand the represented accumulator and next operand to FLOAT32, add once in FLOAT32,
  then narrow to BFLOAT16 after every logical contribution.

NaN payload, sign, and quiet/signaling preservation remain unspecified, but contribution order,
represented positive-zero initialization, exceptional-value class, signed-zero result, and
per-contribution BFLOAT16 narrowing follow the contract above. This matches the current portable
`FOLD2D` realization and gives later Pool3d gradients one unambiguous overlap-add meaning without
introducing a pooling primitive.

### Validation, failure, and allocation order

Direct `unfold3d` validates in this order:

1. null-check `input`, then `window`;
2. require rank-five NCDHW input;
3. require BFLOAT16, FLOAT32, or FLOAT64;
4. derive checked/canonical depth, height, then width geometry;
5. derive channel-kernel and flattened-grid Dimensions;
6. construct the exact descriptor and `UNFOLD3D + Window3dAttrs` operation; and
7. delegate once with exact ordered inputs `[input]`.

Typed `unfold3d` inserts `paddingValue` null checking after `window`, then after rank and floating
eligibility requires exact `paddingValue.dataType() == input.dataType()`, before geometry. It
constructs `UNFOLD3D + Unfold3dAttrs` retaining both supplied references.

`fold3d` validates in this order:

1. null-check `input`, `outputShape`, then `window`;
2. require rank-three canonical-column input;
3. require rank-five NCDHW output Shape;
4. require BFLOAT16, FLOAT32, or FLOAT64 input;
5. require exact structural equality of input and target batch Dimensions;
6. derive and compare the complete channel-kernel Dimension;
7. derive depth, height, then width target-grid geometry;
8. derive and compare the complete flattened-grid Dimension;
9. construct exact `Fold3dAttrs`, descriptor, and `FOLD3D` operation; and
10. delegate once with exact ordered inputs `[input]`.

Use family-specific messages matching current 2D forms with `unfold3d` or `fold3d`, including:

```text
unfold3d requires rank-5 NCDHW input
unfold3d requires floating input: <type>
unfold3d paddingValue data type must match input data type: paddingValue=<type>, input=<type>
unfold3d effective kernel does not fit padded <depth|height|width>

fold3d requires rank-3 canonical column input
fold3d outputShape must be rank-5 NCDHW
fold3d requires floating input: <type>
fold3d output batch dimension must match column batch dimension
fold3d column-channel dimension <actual> does not match output channels and kernel geometry: expected=<expected>
fold3d column count <actual> does not match output shape and window geometry: expected=<expected>
fold3d effective kernel does not fit padded <depth|height|width>
```

Null messages are parameter names. Existing checked-arithmetic, Dimension canonicalization,
descriptor, operation-signature, factory, and identifier-exhaustion failures retain their current
types and messages. Every locally decidable failure precedes the factory call and consumes no
Tensor ID, producer, descriptor wrapper, or output wrapper. Every success consumes exactly one
fresh ID and creates exactly one one-input/one-output producer occurrence.

### Result metadata and provenance

Every successful result:

- preserves the exact input DataType and `requiresGrad` value;
- has unresolved layout, no label, and no host storage;
- owns a fresh descriptor, operation occurrence, producer, ID, and canonical output wrapper;
- records the exact kind and attributes variant, exact ordered inputs `[input]`, and output index
  zero; and
- is exactly the object returned by its producer's `output(0)`.

Direct unfold retains the exact `Window3dAttrs` reference as operation attrs. Typed unfold retains
the exact window and scalar references inside one new `Unfold3dAttrs`. Fold retains the exact
target Shape and window references inside one new `Fold3dAttrs`. Repeated equal requests remain
identity-distinct and never inspect or mutate input layout, label, provenance, storage, or values.

## Affected files

Production/Javadoc, exactly six paths:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/Window3dAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/Unfold3dAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/Fold3dAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/WindowTransformKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorWindowExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Behavior and signature tests, exactly three paths:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationSignatureTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/layout/WindowTransformSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorWindowExpressionTest.java`

Existing exact public-Tensor method-count owners, exactly eighteen paths, change only `210 -> 213`
except `TensorTest` also adds the exact two public method names and window reflection checks:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/RecurrentScanExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBatchNormInferenceExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorConv1dExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorConv3dExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorDenseCategoricalCrossEntropyWithLogitsExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorIndexCategoricalCrossEntropyWithLogitsExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLayerNormExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLinearExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMaxPool3dExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMeanSquaredErrorExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorPool1dExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorRmsNormExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorSlicePlacementExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorSumToShapeExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`

Documentation/planning, exactly seven paths:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review-only with a reasoned no-change conclusion unless a contradiction requires stopping:
`ARCHITECTURE.md`, current architecture pages and ADRs, Runtime and Training API pages, existing
2D/general-axis window, Pool1d/Pool2d/Pool3d, Conv3d, ScalarValue, Shape/Dimension,
operation/signature, Tensor/factory/producer/provenance contracts and tests, Compiler 0005D and
current gradient algebra, Compiler/CPU production and tests, backend conformance, integration,
architecture tests, Gradle/build files, dependencies, and all other modules.

## Maximum scope

Hard ceiling: exactly 34 paths — six production/Javadoc, twenty-one Model test, and seven
documentation/planning paths listed above.

This is an explicit atomic exception to the normal 18-path guardrail. Three rank-specific attrs,
two kind/signature meanings, three public methods, one shared construction owner, exact behavior
tests, eighteen mandatory global method-count locks, and the documentation pass must agree in one
compilable change. Splitting them would leave signatures, public inventory, or the mutually
inverse unfold/fold contract temporarily inconsistent.

The eighteen count-owner files are mechanical count-only edits except for `TensorTest` as stated.
Do not use this allowance for unrelated cleanup. Stop before a thirty-fifth path, another source or
test type, a different existing helper, any non-count change in the other seventeen count owners,
architecture, dependency, Gradle, cross-module implementation, or a later detailed task file.

## Javadoc and documentation requirements

- Fully document all three attrs records, both new kinds, all helper methods, and all three Tensor
  receivers: purpose, NCDHW/rank-three Shapes, geometry, flattening order, direct versus typed
  padding, exact reference retention, fold exclusion/addition, supported types, metadata,
  provenance, validation, allocation effects, and lifecycle boundaries.
- Every public record component, constructor parameter, method parameter, result, nullability,
  constraint, and expected failure has meaningful `@param`, `@return`, and `@throws` coverage.
- Update the WindowTransformKind overview without weakening or rewriting the existing 2D and
  general-axis contracts. Explain why rank-specific 3D kinds are added instead of a `WindowNd`.
- Tensor API adds the three methods, exact formulas and Shapes, a coordinate-order table, direct
  positive-zero versus exact typed-padding example, dynamic NCDHW example, fold overlap example,
  and current-Model versus planned Compiler/backend boundary.
- Compile API records current Model `UNFOLD3D`/`FOLD3D` metadata and their planned use by Compiler
  0006B2 without claiming current Compiler capture, inference, gradients, higher-order support, or
  execution.
- Review glossary entries for NCDHW, window transform, im2col/col2im terminology, padding,
  dilation, literal ceil grids, canonical columns, overlap fold, and positive zero. Extend only
  reusable distinctions; use clear three-dimensional wording such as volumetric columns rather
  than inventing an unexplained acronym.
- Synchronize capabilities, this task, Model master plan, and roadmap. During implementation keep
  0025K Ready or In progress; mark it Complete only after all Java and documentation evidence
  passes. Model 0026, Compiler 0006B1/0006B2, and CPU 0008G1 remain Draft without detailed task
  files.
- Record reasoned no-change conclusions for Runtime/Training APIs, related contracts,
  architecture/ADRs/tests, conformance/integration, Gradle, downstream source/tests, and other
  modules.

## Acceptance criteria

- The three records expose exactly the declared components, validation order, exact-reference
  retention, and structural-only responsibilities.
- WindowTransformKind has exactly six constants in the required order. Only the new kinds gain the
  exact new signatures, with direct UNFOLD3D attrs before explicit-padding attrs.
- Tensor adds exactly three declarations and no alias/default/builder/namespace; the exact public
  declared-method count is 213.
- Both unfold forms accept only BFLOAT16/FLOAT32/FLOAT64 rank-five NCDHW input, derive exact static
  floor, literal-ceil, terminal all-padding, and canonical-symbolic rank-three columns, and retain
  exact batch/type/gradient metadata.
- Direct unfold uses represented positive zero. Typed unfold retains the exact matching
  ScalarValue reference and raw bits and can represent BFLOAT16/FLOAT32/FLOAT64 negative infinity
  without a pooling-specific kind.
- Canonical `q` and `p` flattening and depth-height-width/kernel ordering match the formulas in this
  task for static and unresolved Shapes.
- Fold accepts only exact structurally compatible floating rank-three columns and rank-five target
  Shape, retains that exact Shape, excludes every out-of-range padded/ceil-tail coordinate, and
  overlap-adds in canonical input order with the exact type-specific positive-zero and addition
  policy.
- Static geometry/products use checked arithmetic; unresolved geometry uses only existing
  canonical Dimension expressions; no host kernel-volume product or unnamed equality constraint
  is introduced.
- Null/type/rank/batch/channel/grid/fit/overflow failures occur in exact order before factory
  allocation and consume no Tensor ID. Every success consumes one ID and creates one fresh exact
  one-input/output-zero canonical occurrence.
- Existing general-axis and 2D window behavior, Pool3d semantics, Shape/Dimension algebra,
  ScalarValue, factory/provenance, and all architecture boundaries remain unchanged.
- Focused tests, exactly one final Model suite after executable stabilization, final Model
  Javadoc, runnable Java 26 metadata example, reflection/`javap`, operation and Tensor inventory,
  Markdown headings/links/anchors/fences/newlines/terminology, exact 34 paths, status/order/
  dependencies/task-file absence, and `git diff --check` pass.
- A separate clean documentation-focused context reuses successful Java evidence, independently
  finalizes all affected Javadocs/docs/glossary/planning, and records no-change conclusions before
  completion.
- No gradient, Compiler, backend, Runtime, architecture, dependency, Gradle, detailed later task,
  or generic WindowNd/PoolNd implementation is present.

## Tests / validation

Focused implementation validation:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest \
  --tests io.github.pho001.synaptik.model.operation.layout.WindowTransformSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorWindowExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorTest
```

After executable Java stabilizes, run exactly once:

```bash
./gradlew :modules:model:test
```

Tests must cover record reflection/validation/reference retention, enum order and signatures,
public receiver shape and count locks, every accepted/rejected DataType, static and symbolic
floor/ceil geometry, terminal all-padding windows, product overflow/zero cases, exact coordinate
flattening, padding bits, fold structural compatibility, exclusion/addition meaning, exact target
Shape, metadata, freshness, producer/output identity, and ID deltas. Numerical fold tests in Model
lock inspectable semantics and Javadocs rather than pretending Model evaluates values.

The separate documentation-focused pass reuses successful Java evidence unless it changes
executable behavior. After final Javadoc edits it runs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

It also:

- compiles and runs one Java 26 metadata example covering both unfold variants, fold, Shape,
  attrs, provenance, exact reference retention, unresolved layout, and absent storage;
- uses reflection and `javap` for the three records, six-kind enum, exact signatures, three Tensor
  declarations, helper visibility/field-free shape, 213-method Tensor count, and 40/115/137
  operation inventory;
- validates changed Markdown headings, links, anchors, fences, final newlines, trailing whitespace,
  terminology, examples, and current-versus-planned claims;
- confirms exactly the 34 authorized paths and package placement;
- confirms 0025K is the sole detailed Ready/In-progress Model task until completion, 0026 remains
  Draft without a task file, and Compiler 0006B1/0006B2 and CPU 0008G1 remain Draft without task
  files; and
- confirms the exact cross-plan order `0025I -> 0025J -> 0025K -> Compiler 0006B1 -> Compiler
  0006B2 -> CPU 0008G1 -> CPU 0008H`.

Repository-wide validation is deferred to the next recorded capability checkpoint and CI. No
Compiler, CPU, architecture, backend-conformance, or integration suite is required because this
task changes one module without dependencies, shared build configuration, backend behavior, or an
architecture boundary.

## Dependencies

- Complete Model 0023D supplies the public/dynamic canonical rank-three `unfold2d`/`fold2d`
  contract, exact typed padding attributes, symbolic products, validation ordering, and canonical
  provenance precedent.
- Complete Model 0020A/0020A1 supply excluded-padding maximum and fixed-count average Pool2d
  geometry/numerical policies.
- Complete Model 0025J supplies first-class NCDHW Pool3d geometry, depth-height-width first-winner
  order, current floating types, and the exact downstream need.
- Complete Compiler 0005D and current `PoolingGradientRules` demonstrate the generally useful
  public 2D unfold/fold algebra needed to reconstruct exact maximum winners and distribute
  fixed-count average cotangents.
- Complete CPU 0006B2 supplies source-backed represented positive-zero and type-specific overlap
  addition precedent for current FOLD2D. This task does not add CPU execution.
- Existing Model operation/signature, ScalarValue, Shape/Dimension, Tensor descriptor/factory,
  canonical producer/output, and provenance contracts are stable.

## Follow-up tasks

- Compiler 0006B1 remains Draft without a detailed task file. It owns forward capture/inference/
  final validation for `MAX_POOL3D` and `AVERAGE_POOL3D` plus fail-closed backward before any
  derivative allocation.
- Compiler 0006B2 remains Draft without a detailed task file after 0025K and 0006B1. It owns exact
  Pool3d gradients through these public transforms: average distributes the cotangent divided by
  the logical `kernelDepth * kernelHeight * kernelWidth`; maximum unfolds exact input with typed
  negative-infinity padding, separately reconstructs in-bounds eligibility, matches the exact
  same-occurrence Pool3d output including NaN and signed zero, selects the first depth-height-width
  candidate, routes the cotangent, and fold3d overlap-adds to the exact input Shape.
- Compiler 0006B2 must preserve Pool3d's specified accumulator/division/narrowing, exceptional-
  value, padding, first-winner, and occurrence-identity contracts and must update the source-backed
  derivative and higher-order inventories. This task selects no derivative formula implementation.
- CPU 0008G1 remains Draft without a detailed task file after both Compiler tasks. It owns exact
  Pool1d composition validation and first-class Pool3d generated execution, including its own
  algorithm, Class-File, conformance, and performance evidence.
- Model 0026 remains Draft without a detailed task file for FLOAT16 and mixed precision.

Do not create another detailed specification during implementation.

## Architecture impact

Expected impact: None.

This task extends existing Model-owned rank-specific window semantics and public Tensor metadata
construction without changing module ownership, dependency direction, lifecycle, or architecture.
If implementation requires an architecture edit, cross-layer constraint service, new Shape
expression, backend policy, compiler behavior, another module, or scope beyond the exact ceiling,
stop and report the conflict.

## Implementation prompt

### Clean implementation task prompt

```text
Work in /Users/phujka/IdeaProjects/Synaptik without commit or push. Do not use a GSD workflow.

Read root AGENTS.md, ARCHITECTURE.md, current architecture plan, planning guide/roadmap, Model
master/capabilities, task 0025K, completed 0023D/0020A/0020A1/0025J and Compiler 0005D, current
window/Pool3d/Tensor/gradient source and tests, and the exact affected/review-only contracts named
by task 0025K.

Implement task 0025K exactly inside its 34 authorized paths. Add only rank-specific Window3dAttrs,
Unfold3dAttrs, Fold3dAttrs, appended UNFOLD3D/FOLD3D signatures, the three exact Tensor receivers,
and matching TensorWindowExpressions construction. Preserve canonical rank-three columns,
depth-height-width/kernel order, literal floor/ceil geometry, exact padding, overlap addition,
validation/ID order, and canonical provenance. Add no WindowNd/PoolNd, gradient, Compiler/backend,
build, dependency, architecture, or later-task work. Stop on any conflict or scope overflow.

Run focused validation while developing and exactly one final Model suite after executable Java
stabilizes. Do not edit the seven documentation/planning paths except implementation evidence in
this task; production Javadocs may be provisional for the required separate clean documentation
pass. Hand the actual stabilized diff and exact Java evidence to that context. Do not mark 0025K
Complete until both passes and every acceptance criterion succeed.
```

### Clean documentation task prompt

```text
In a distinct clean documentation-focused context, review the stabilized task-0025K diff and exact
implementation evidence. Work in the same Synaptik change without commit or push and without a GSD
workflow.

Read root AGENTS.md, ARCHITECTURE.md, documentation rules and General/API-Javadoc/Planning/Example
profiles, task 0025K, Model master/capabilities, affected source/tests/generated Javadoc,
Tensor/Compile/Runtime/Training APIs, glossary, completed 0023D/0020A/0020A1/0025J/Compiler 0005D,
current Pool3d gradient algebra, and downstream Compiler 0006B1/0006B2 and CPU 0008G1 rows.

Independently finalize the six affected production Javadocs and exactly the seven documentation/
planning paths authorized by task 0025K. Explain exact NCDHW/rank-three Shape and coordinate order,
direct versus typed padding, fold exclusion and type-specific overlap addition, canonical fresh
provenance, and current Model versus planned Compiler/backend boundaries. Record reasoned no-change
conclusions for Runtime/Training APIs, related contracts, architecture/ADRs/tests, conformance/
integration, Gradle, downstream source/tests, and other modules.

Reuse successful Java test evidence unless executable Java changes or a concrete risk requires a
rerun. Run final Model Javadoc, the Java 26 metadata example, reflection/javap and inventory checks,
Markdown headings/links/anchors/fences/newlines/terminology, exact 34-path and package checks,
status/dependency/order/later-task-file absence, and git diff --check. Mark 0025K Complete only
when every criterion passes; otherwise return the exact incomplete follow-up.
```

## Local decisions

- Canonical rank-three columns extend the completed 2D public contract. A second rank-six/rank-
  eight carrier would give the same logical data a divergent Shape and ordering contract without
  a current need.
- New rank-specific kinds and attrs are required. Reusing Pool3d attrs would make general window
  transforms inherit pooling numerical meaning; reusing 2D attrs cannot represent depth. A
  generic WindowNd abstraction is not justified by this one concrete rank-three extension.
- Direct positive-zero and explicit typed-padding unfold variants mirror current UNFOLD2D. The
  typed variant is the general capability that later represents maximum-pool negative-infinity
  padding without a pooling-specific primitive.
- Fold retains an explicit exact target Shape because canonical column dimensions do not identify
  trailing uncovered target coordinates. Structural equality, not an unnamed Model constraint,
  is required for unresolved columns.
- Fold contribution order and type-specific addition follow the current portable FOLD2D
  realization so signed zero, exceptional values, and BFLOAT16 per-contribution narrowing are not
  left ambiguous for later Pool3d gradients.
- Successive Dimension multiplication avoids a false host-`long` kernel-volume requirement while
  still rejecting an actually unrepresentable result Dimension through existing Shape rules.
- The 34-path ceiling is an atomic exception dominated by eighteen mandatory Tensor count locks.

## Known limitations

- Only floating rank-five NCDHW source/target tensors with symmetric intrinsic padding are
  included. There is no NDHWC, asymmetric intrinsic padding, FLOAT16, integral, BOOL, sparse,
  quantized, or arbitrary-rank form.
- Dynamic Dimensions retain exact formulas and non-negative-numerator obligations but do not bind
  or evaluate them. Fold rejects unrelated unresolved column symbols.
- Window transforms remain metadata only. No current Compiler or backend support is implied by
  Model construction, and no numerical value is materialized in this task.
- Direct unfold has only positive-zero padding; any other padding uses the exact typed overload.
  Fold always excludes out-of-range coordinates and has no configurable padding contribution.
- Repository-wide validation remains deferred to the next capability checkpoint or CI.

## Validation evidence

- Implementation Java evidence was reused without rerunning tests because this documentation pass
  changed no executable Java. The stabilized `modules/model/build/test-results/test/TEST-*.xml`
  artifacts from 2026-08-30 11:16:28 CEST record the final
  `./gradlew :modules:model:test` result: 138 suites, 1,103 tests, zero skipped, zero failures, and
  zero errors. The reports include `OperationSignatureTest`, `WindowTransformSemanticsTest`,
  `TensorWindowExpressionTest`, and `TensorTest`; executable Java did not change afterward.
- `./gradlew :modules:model:javadoc` passed after the documentation review with
  `BUILD SUCCESSFUL`; `compileJava` and `javadoc` were up to date against the stabilized sources.
- The Java 26 `/tmp/Task0025KMetadataExample.java` compiled against
  `modules/model/build/classes/java/main` and ran successfully. It printed direct and typed column
  Shapes `Shape[1, 8, 8]`, fold target `Shape[1, 1, 3, 3, 3]`, and `metadata-ok` after checking
  both unfold variants, fold, exact Shape/window/padding references, operation kinds, canonical
  producer output, unresolved layout, and absent storage.
- `/tmp/Task0025KInventoryCheck.java` compiled and ran successfully. Reflection reported exactly
  `families=40 constants=115 signatures=137 tensorMethods=213`, verified the six-kind window enum
  order, two UNFOLD3D and one FOLD3D signatures, three record component surfaces, three exact
  Tensor declarations, and the package-private final field-free helper shape.
- `javap -public` confirmed the three record constructors, public enum constants `UNFOLD3D` and
  `FOLD3D`, and exactly the two `unfold3d` plus one `fold3d` Tensor declarations.
- `ruby /tmp/task0025k_markdown_check.rb` passed all seven changed documentation/planning files:
  local Markdown targets and anchors resolved, fences were balanced, and final newlines were
  present. The focused terminology/example review confirmed NCDHW expansion at first use,
  canonical volumetric-column Shape and flattening, literal floor/ceil grids, direct positive-zero
  versus typed padding, fold exclusion/addition, symbolic obligations, and current-versus-planned
  claims.
- The exact-scope comparison against `/tmp/task0025k_expected_paths.txt` passed with 34 paths:
  six production/Javadoc, twenty-one Model tests, and seven documentation/planning paths. All new
  production types match the task package map.
- Later-task and ordering checks passed: Model 0025I, 0025J, and 0025K are Complete in order;
  Compiler 0006B1 and 0006B2, CPU 0008G1 and 0008H, and Model 0026 remain Draft; no detailed task
  file exists for Model 0026, Compiler 0006B1/0006B2, or CPU 0008G1. The next cross-plan frontier
  is Compiler 0006B1, without a detailed specification.
- Downstream absence/import scans found no UNFOLD3D/FOLD3D or 3D-window-attribute production/test
  adoption in Compiler, Runtime, Training, CPU, or shared testing. `PoolingGradientRules` remains
  the unchanged 2D source-backed precedent; Compiler 0006B1/0006B2 and CPU 0008G1 remain planning
  rows only.
- Runtime and Training API pages require no change because this task adds no prepared/run contract,
  execution state, optimizer, or training orchestration API. Architecture pages, ADRs, and
  architecture tests require no change because ownership, dependency direction, lifecycle, and
  module boundaries are unchanged. Backend conformance and integration tests require no change
  because no backend capability or end-to-end execution was added. Gradle/build files and
  dependencies require no change because all implementation stays inside the existing Model
  module surface. Downstream production/tests and every other module require no change because
  Compiler adoption, gradients, and CPU execution are explicitly deferred.
- Final changed-Markdown validation, exact-scope/status scans, downstream absence scans, and
  `git diff --check` passed after this completion record was written.

## Implementation notes

- Added immutable `Window3dAttrs`, `Unfold3dAttrs`, and `Fold3dAttrs`; appended UNFOLD3D/FOLD3D
  without changing earlier enum order or signatures; and added the exact three Tensor receivers
  through the existing `TensorWindowExpressions` construction owner.
- Static and canonical-symbolic depth-height-width geometry, canonical rank-three columns,
  structural fold compatibility, exact metadata/reference retention, validation-before-allocation,
  and one-input/output-zero provenance match the task contract.
- The clean documentation-focused pass independently reviewed all six production Javadocs and
  finalized the Tensor API, Compile API, glossary, capabilities, task, Model master plan, and
  roadmap. It changed no executable Java and therefore reused the successful Model test evidence.

## Completion summary

- Completed changes: public NCDHW `unfold3d`/`fold3d` Model algebra, exact attributes/signatures,
  three Tensor receivers, validation/metadata construction, tests/count locks, finalized Javadocs,
  API/glossary explanations, and synchronized planning.
- Files changed or created: exactly 34 authorized paths: six production/Javadoc, twenty-one Model
  tests, and seven documentation/planning files.
- Tests and validation: reused the stabilized final Model result of 1,103 tests across 138 suites
  with zero failures/errors; final Model Javadoc, runnable Java 26 metadata example,
  reflection/inventory, `javap`, Markdown, scope/status/order/absence scans, and whitespace checks
  passed.
- Documentation-agent review: completed in this mandatory separate clean documentation context;
  the six production Javadocs and all seven authorized documentation/planning paths were reviewed
  and finalized.
- Documentation impact: Tensor and Compile API pages now distinguish current Model metadata from
  Draft Compiler 0006B1/0006B2 and CPU 0008G1; Runtime and Training APIs need no change for the
  ownership reasons recorded above.
- Javadoc review: all six changed production files accurately document parameters, results,
  failures, Shapes, padding, fold addition, metadata/provenance, and lifecycle boundaries.
- Glossary impact: the window-transform entry now covers NCDHW volumetric columns, 3D flattening,
  typed padding, literal-ceil exclusion, and type-specific overlap fold without inventing a new
  acronym.
- Unresolved issues: None.
- Follow-up required: None. Compiler 0006B1 remains the next Draft frontier and does not yet have
  a detailed task specification.

Status: Complete
