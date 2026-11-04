# Task 0025K: Public [NCDHW](../../../../glossary.md#conv3d--ncdhw) unfold3d and fold3d window transforms

**Editorial reading key (current context, not a status update).**

This record covers Model-only construction for BFLOAT16, FLOAT32 and FLOAT64, not today’s broader
routes. NCDHW names logical batch, channel, depth, height and width axes; canonical columns flatten
channel/kernel positions and spatial window positions, with width fastest. The caller supplies
window geometry, an optional same-typed padding scalar, and an explicit fold target Shape. Tensor
receivers delegate to TensorWindowExpressions: descriptions and input references cross this
boundary, not sampled values. Fresh storage-free results record a producer but neither execute nor
own result buffers; Model’s semantic responsibility is not storage ownership.

Unfold describes extraction with positive-zero or exact typed padding; fold excludes out-of-range
coordinates geometrically and adds overlaps in canonical order, without averaging. For a one-channel
three-by-three-by-three target and two-by-two-by-two unit-stride windows without padding, the center
receives eight contributions. Follow the [current window
contract](../../../../reference/tensor/windows.md#window-transform-expressions) for typed
arithmetic, special-value boundaries, and later Compiler/backend descriptions, including FP8
support. Those current descriptions do not expand this historical task’s scope: providers realize
supported routes rather than redefine Model meaning.

**Overlap reading illustration.** Let a one-channel real volume have three positions on each spatial
axis and use size-two windows at stride one without padding. For a target coordinate $p$, let $W(p)$
be the extracted sample occurrences that map back to it; fold computes $f(p)=\sum_{w\in W(p)}c_w$,
where $c_w$ is that column occurrence's value. The center has eight occurrences, so unit columns
give eight, not one. This explains overlap membership rather than authorizing a different floating
addition order.

## Status

Complete

## Goal

Add the smallest general public three-dimensional window algebra needed for exact [NCDHW](../../../../glossary.md#conv3d--ncdhw) [Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling)
adjoints without adding [pooling](../../../../glossary.md#term-pooling)-specific backward operations:

```text
[N, C, D, H, W]
  -- unfold3d(window[, paddingValue]) -->
[N, C * kD * kH * kW, D_out * H_out * W_out]
  -- fold3d([N, C, D, H, W], window) -->
[N, C, D, H, W]
```

The [rank](../../../../glossary.md#term-rank)-three column representation extends the current [canonical](../../../../glossary.md#term-canonical-form) [NCHW](../../../../glossary.md#term-nchw) `unfold2d`/`fold2d`
contract by one [spatial axis](../../../../glossary.md#spatial-axes-channels-and-window-sampling). Direct unfold uses conceptual typed positive-zero [padding](../../../../glossary.md#padding); the
explicit overload retains one exact typed padding value, including [negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero) for later
maximum-pooling winner reconstruction. Fold excludes out-of-range padded [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) and
overlap-adds every in-range contribution into the exact caller-supplied NCDHW [target](../../../../glossary.md#term-target) [Shape](../../../../glossary.md#shape).

This task owns [Model](../../../../glossary.md#term-model) semantics, local Shape [validation](../../../../glossary.md#term-validation), public [Tensor](../../../../glossary.md#tensor) construction, and canonical
one-input [provenance](../../../../glossary.md#provenance) only. It creates no Pool3d [gradient](../../../../glossary.md#term-gradient), [Compiler](../../../../glossary.md#term-compiler) adoption, [backend capability](../../../../glossary.md#backend-capability),
[materialization](../../../../glossary.md#materialization), execution, or performance claim.

## [Scope](../../../../glossary.md#term-scope)

- Add public [immutable](../../../../glossary.md#term-immutable) `Window3dAttrs` for depth-height-width kernel, stride, symmetric padding,
  dilation, and literal ceil-mode geometry.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [stride glossary](../../../../glossary.md#term-stride); [padding glossary](../../../../glossary.md#padding); [dilation glossary](../../../../glossary.md#term-dilation))</span>
- Add public immutable `Unfold3dAttrs(Window3dAttrs window, ScalarValue paddingValue)` for exact
  typed out-of-domain samples.<span data-glossary-navigation="true"> ([samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- Add public immutable `Fold3dAttrs(Shape outputShape, Window3dAttrs window)` for the exact NCDHW
  fold target.<span data-glossary-navigation="true"> ([NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [target glossary](../../../../glossary.md#term-target))</span>
- Append `UNFOLD3D` and `FOLD3D` to `WindowTransformKind` without changing the existing four
  constants, their order, or their signatures.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [signatures glossary](../../../../glossary.md#term-signature))</span>
- Add exactly these public [Tensor](../../../../glossary.md#tensor) methods:

  ```java
  public Tensor unfold3d(Window3dAttrs window)
  public Tensor unfold3d(Window3dAttrs window, ScalarValue paddingValue)
  public Tensor fold3d(Shape outputShape, Window3dAttrs window)
  ```

- Extend the existing package-private, final, field-free `TensorWindowExpressions` owner with
  rank-five NCDHW construction and exact rank-three column compatibility validation.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [rank glossary](../../../../glossary.md#term-rank); [validation glossary](../../../../glossary.md#term-validation))</span>
- Accept [BFLOAT16](../../../../glossary.md#term-bfloat16), [FLOAT32](../../../../glossary.md#term-float32), and [FLOAT64](../../../../glossary.md#term-float64) only, matching current `unfold2d`/`fold2d`.
- Preserve exact static or canonical-symbolic depth, height, and width geometry, symmetric
  padding, dilation, stride, and literal floor/ceil grids.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Fix the canonical column Shape, flattening order, out-of-domain padding meaning, fold exclusion,
  overlap addition, positive-zero initialization, validation order, allocation effects, and
  canonical provenance below.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [provenance glossary](../../../../glossary.md#provenance))</span>
- Update exact operation/signature inventories and every current global public-Tensor method-count
  lock from 210 to 213.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- Finalize affected Javadocs, Tensor and Compile API explanations, glossary terminology,
  capabilities, and planning in a separate clean documentation-focused context before completion.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Pool3d gradient formulas, derivative policy, Compiler 0006B1 or 0006B2 implementation, saved
  maximum indices, unpooling, a pooling-backward kind, or another pooling-specific primitive.<span data-glossary-navigation="true"> ([Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling); [gradient glossary](../../../../glossary.md#term-gradient); [derivative glossary](../../../../glossary.md#term-derivative); [Compiler glossary](../../../../glossary.md#term-compiler); [pooling glossary](../../../../glossary.md#term-pooling))</span>
- CPU 0008G1 or any backend lowering, capability advertisement, algorithm, generated code,
  materialization, workspace, Runtime behavior, execution, conformance, integration, or
  performance evidence.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [materialization glossary](../../../../glossary.md#materialization); [Runtime glossary](../../../../glossary.md#term-runtime); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- A public or private generic `WindowNd`, `PoolNd`, arbitrary-rank transform, dynamic-rank API,
  geometry arrays, rank-polymorphic attributes, broad window facade, manager, or utility.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [facade glossary](../../../../glossary.md#term-facade))</span>
- Reusing or modifying `MaxPool3dAttrs` or `AveragePool3dAttrs` as general window metadata; their
  excluded-padding extrema and fixed-divisor average policies remain pooling-specific.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [padding glossary](../../../../glossary.md#padding); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>
- Changing the existing `UNFOLD_AXIS`, `FOLD_AXIS`, `UNFOLD2D`, or `FOLD2D` declarations,
  signatures, attributes, Shape ordering, padding meanings, numerical semantics, or public APIs.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [Shape glossary](../../../../glossary.md#shape))</span>
- Integral or BOOL 3D transforms, FLOAT16, implicit conversion, promotion, output-type override,
  asymmetric intrinsic padding, padding modes, Tensor-valued padding, valid-sample counting, or
  overlap averaging.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [FLOAT16 glossary](../../../../glossary.md#term-float16); [Tensor glossary](../../../../glossary.md#tensor); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- A second rank-five/rank-seven window carrier, an unflattened public patch tensor, an inferred
  fold target Shape, or unnamed equality constraints for unrelated unresolved Dimensions.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [target glossary](../../../../glossary.md#term-target); [Dimensions glossary](../../../../glossary.md#dimension))</span>
- Value reads, eager evaluation, host allocation, attached result storage, resolved result layout,
  mutation, alias promises, or input materialization.<span data-glossary-navigation="true"> ([eager evaluation glossary](../../../../glossary.md#term-eager-evaluation); [host glossary](../../../../glossary.md#term-host); [layout glossary](../../../../glossary.md#layout); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Changes to Shape/Dimension expression types, ScalarValue representation, TensorFactory seams,
  Pool3d/Pool2d/Conv3d behavior, dependencies, Gradle, architecture contracts, ADRs, architecture
  tests, other modules, or detailed specifications for Compiler 0006B1/0006B2, CPU 0008G1, or
  Model 0026.<span data-glossary-navigation="true"> ([ScalarValue glossary](../../../../glossary.md#typed-scalar-value--scalarvalue); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module); [Model glossary](../../../../glossary.md#term-model))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Roadmap](../../../roadmap.md)
- [Model master plan](../master-plan.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model capabilities](../capabilities.md)
- [Public foldAxis and dynamic 2D windows](0023d-public-fold-axis-and-dynamic-window-transforms.md)
- [NCHW Max Pool2d](0020a-nchw-max-pool2d-semantics-and-tensor-expression.md)<span data-glossary-navigation="true"> ([NCHW glossary](../../../../glossary.md#term-nchw))</span>
- [NCHW Average Pool2d](0020a1-nchw-average-pool2d-semantics-and-tensor-expression.md)
- [First-class NCDHW Pool3d](0025j-first-class-ncdhw-max-average-pool3d-semantics.md)<span data-glossary-navigation="true"> ([NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling))</span>
- [Compiler Pool2d gradients](../../compiler/tasks/0005d-attention-convolution-pooling-and-loss-gradient-completion.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- [CPU overlap fold](../../../backends/cpu/tasks/0006b2-portable-overlap-fold.md)<span data-glossary-navigation="true"> ([CPU overlap fold glossary](../../../../glossary.md#cpu-overlap-fold))</span>

## Architecture constraints

- Work stays in `modules/model` plus directly affected documentation and planning. Model owns
  immutable backend-independent operation meanings, attributes, Tensor descriptors, public
  expression construction, and pre-capture producer metadata.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [Model glossary](../../../../glossary.md#term-model); [immutable glossary](../../../../glossary.md#term-immutable); [backend glossary](../../../../glossary.md#backend); [operation glossary](../../../../glossary.md#operation); [Tensor glossary](../../../../glossary.md#tensor); [descriptors glossary](../../../../glossary.md#term-descriptor); [capture glossary](../../../../glossary.md#term-capture-meanings); [producer glossary](../../../../glossary.md#term-producer); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Tensor remains public mutable API state, not graph IR. Every successful call returns one fresh
  storage-free canonical Tensor wrapper; no method evaluates a value.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [graph glossary](../../../../glossary.md#graph); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Window kinds and attributes contain no Tensor reference except the established immutable Shape
  and ScalarValue semantic values, and no graph, compiler, planning, backend, prepared, runtime,
  kernel, route, storage, or execution state.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [ScalarValue glossary](../../../../glossary.md#typed-scalar-value--scalarvalue); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [kernel glossary](../../../../glossary.md#kernel))</span>
- Direct positive-zero padding and explicit typed padding are semantic extraction values, not
  backend fill policies. Fold padding exclusion is geometric and never inferred by comparing
  column values with a padding scalar.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Result layout remains unresolved because materialized windows and overlap accumulation do not
  establish alias-view geometry.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [alias glossary](../../../../glossary.md#aliasing-and-arena); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- Every derived occurrence uses the existing factory-atomic canonical-output seam. A returned
  Tensor is exactly producer <span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>`output(0)` with output index zero.
- Compiler owns Pool3d forward adoption, retained binding proof, gradient construction, and
  higher-order closure. Backends own algorithms and execution. This task must not move either
  responsibility into Model.<span data-glossary-navigation="true"> ([Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling); [binding glossary](../../../../glossary.md#term-binding); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- No architecture, dependency, module-boundary, lifecycle, build, conformance, or integration
  contract changes. Stop if implementation requires one.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module); [lifecycle glossary](../../../../glossary.md#lifecycle); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Package impact

Existing packages used and changed:

- `io.github.pho001.synaptik.model.operation.layout` — rank-specific immutable window meanings and
  attributes.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- `io.github.pho001.synaptik.model.tensor` — established public Tensor facade and package-private
  window-expression construction owner.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- `io.github.pho001.synaptik.model.shape` — existing static and canonical symbolic Dimension
  arithmetic; no new expression form.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [Dimension glossary](../../../../glossary.md#dimension))</span>
- `io.github.pho001.synaptik.model.datatype` — existing floating-type and exact ScalarValue
  contracts.<span data-glossary-navigation="true"> ([ScalarValue glossary](../../../../glossary.md#typed-scalar-value--scalarvalue))</span>

Packages added or moved: None.

Type placement:

- `...operation.layout.Window3dAttrs` — reusable rank-specific three-dimensional window geometry,
  distinct from pooling-specific attrs.<span data-glossary-navigation="true"> ([pooling glossary](../../../../glossary.md#term-pooling))</span>
- `...operation.layout.Unfold3dAttrs` — exact typed-[padding](../../../../glossary.md#padding) `UNFOLD3D` metadata beside the existing
  2D counterpart.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>
- `...operation.layout.Fold3dAttrs` — exact [target](../../../../glossary.md#term-target) [Shape](../../../../glossary.md#shape) and geometry for `FOLD3D`.
- `...operation.layout.WindowTransformKind` — existing window identity/signature owner, extended
  by the two rank-specific meanings.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [signature glossary](../../../../glossary.md#term-signature))</span>
- `...tensor.TensorWindowExpressions` — existing cohesive owner for input-aware window validation,
  Shape derivation, descriptors, operations, and factory delegation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- `...tensor.Tensor` — established fluent public receiver surface.

Tests continue to mirror the production packages. No generic helper package or second [Tensor](../../../../glossary.md#tensor)
[facade](../../../../glossary.md#term-facade) is added.

## Required contracts

### [Rank](../../../../glossary.md#term-rank)-specific attributes

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

`Window3dAttrs` validates components in declaration order. [Kernel](../../../../glossary.md#kernel), [stride](../../../../glossary.md#term-stride), and [dilation](../../../../glossary.md#term-dilation) values
must be positive; [padding](../../../../glossary.md#padding) values must be non-negative. Failures use the established exact forms:

```text
<component> must be positive: <value>
<component> must be non-negative: <value>
```

It performs no input-aware rank, [Shape](../../../../glossary.md#shape), type, fit, product, or arithmetic [validation](../../../../glossary.md#term-validation). In
particular, it does not precompute the three-factor kernel volume.

`Unfold3dAttrs` null-checks `window` then `paddingValue`, retains both exact references, and
performs no input-aware type compatibility. The padding [scalar](../../../../glossary.md#term-scalar) supplies every sampled [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)
outside the logical unpadded [NCDHW](../../../../glossary.md#conv3d--ncdhw) input, including leading/trailing symmetric padding and every
terminal ceil-grid coordinate beyond the padded [extent](../../../../glossary.md#term-extent). Its exact type and primitive bits are
semantic [metadata](../../../../glossary.md#term-metadata).

`Fold3dAttrs` null-checks `outputShape` then `window`, retains both exact [immutable](../../../../glossary.md#term-immutable) references, and
performs no rank, type, geometry, or column-compatibility validation.

Each record has only its declared components and ordinary record value semantics. Add no builder,
defaults, arrays, nested types, [normalization](../../../../glossary.md#term-normalization) method, [pooling](../../../../glossary.md#term-pooling) conversion method, or [mutable](../../../../glossary.md#term-mutable) state.

### Kinds and [signatures](../../../../glossary.md#term-signature)

Append the new [constants](../../../../glossary.md#term-constant) after the existing four values. Exact enum order becomes:

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
signature lists remain unchanged. The [Model](../../../../glossary.md#term-model) compiled inventory becomes exactly 40 [operation](../../../../glossary.md#operation)-kind
families, 115 constants, and 137 signatures; tests must derive and lock the new window-family
surface rather than altering unrelated family contracts.

### Public [Tensor](../../../../glossary.md#tensor) surface

Add exactly the three methods listed in [Scope](../../../../glossary.md#term-scope). Each delegates once to the matching
`TensorWindowExpressions` overload and owns no [validation](../../../../glossary.md#term-validation), [Shape](../../../../glossary.md#shape) arithmetic, [operation](../../../../glossary.md#operation)
construction, factory call, default, or [alias](../../../../glossary.md#aliasing-and-arena). The declared public Tensor method count becomes
exactly 213. Add only public names `unfold3d` and `fold3d`; `unfold3d` has exactly the two specified
overloads and `fold3d` exactly one.

### [NCDHW](../../../../glossary.md#conv3d--ncdhw) geometry and [canonical](../../../../glossary.md#term-canonical-form) column [Shape](../../../../glossary.md#shape)

Input is [rank](../../../../glossary.md#term-rank)-five NCDHW `[N, C, D, H, W]`. For each [spatial](../../../../glossary.md#term-spatial-axes) input [Dimension](../../../../glossary.md#dimension) `X`, positive [kernel](../../../../glossary.md#kernel)
[sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) count `k`, non-negative symmetric [padding](../../../../glossary.md#padding) per side `p`, positive [dilation](../../../../glossary.md#term-dilation) `d`, and positive
[stride](../../../../glossary.md#term-stride) `s`:

```text
effectiveKernel = d * (k - 1) + 1
numerator       = X + 2 * p - effectiveKernel
floor output    = floor(numerator / s) + 1
ceil output     = ceil(numerator / s) + 1
```

All literal [constants](../../../../glossary.md#term-constant) and static arithmetic use checked signed-`long` operations. Static negative
numerators fail. For unresolved Dimensions, use the existing canonical `addConstant`, matching
floor/ceiling divide, and `addConstant(1)` forms; retain the non-negative-numerator obligation for
later [Compiler](../../../../glossary.md#term-compiler)/[binding](../../../../glossary.md#term-binding) proof. Derive and validate depth, then height, then width. Literal ceil mode
never decrements a terminal window, including an all-padding window.

The canonical unfold result Shape is exactly:

```text
[N, C * kernelDepth * kernelHeight * kernelWidth,
    outputDepth * outputHeight * outputWidth]
```

Retain the exact input [batch Dimension](../../../../glossary.md#batch-dimension) reference. Form both products through successive existing
`DimensionExpressions.multiply` calls in written order; do not precompute `kernelDepth *
kernelHeight * kernelWidth` as a [host](../../../../glossary.md#term-host) `long`. A static or symbolic product fails only under the
existing Dimension representability rules. The result is rank three even though the source is
rank five; no alternate patch Shape is exposed.

### Deterministic [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) ordering

For output-grid coordinates `(od, oh, ow)` and [kernel](../../../../glossary.md#kernel) coordinates `(kd, kh, kw)`, define:

```text
id = od * strideDepth  - paddingDepth  + kd * dilationDepth
ih = oh * strideHeight - paddingHeight + kh * dilationHeight
iw = ow * strideWidth  - paddingWidth  + kw * dilationWidth

q = (((c * kernelDepth + kd) * kernelHeight + kh) * kernelWidth + kw)
p = ((od * outputHeight + oh) * outputWidth + ow)
```

The column coordinate is `[n, q, p]`. Thus [channel](../../../../glossary.md#term-channel) is outermost in `q`, then kernel depth, kernel
height, and kernel width; width is fastest. In `p`, output depth is outermost, then output height,
then output width; width is fastest. Unfold uses increasing `(n, c, kd, kh, kw, od, oh, ow)`
[canonical](../../../../glossary.md#term-canonical-form) [row-major](../../../../glossary.md#term-row-major) column order as defined by `[n,q,p]`. This fixed structural order is the later
[Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling) maximum first-winner basis and must not be replaced by height-first, width-first, or
[backend](../../../../glossary.md#backend)-dependent ordering.

The direct `unfold3d(window)` supplies exact represented [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) for every coordinate where
`id`, `ih`, or `iw` is outside its unpadded input [extent](../../../../glossary.md#term-extent). The typed overload supplies the exact
retained `paddingValue` instead. In-range [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) retain their exact represented input values.
Unfold performs no arithmetic on sampled values.

### Exact [NCDHW](../../../../glossary.md#conv3d--ncdhw) fold compatibility

`fold3d` consumes [rank](../../../../glossary.md#term-rank)-three [canonical](../../../../glossary.md#term-canonical-form) columns and an explicit rank-five NCDHW [target](../../../../glossary.md#term-target) [Shape](../../../../glossary.md#shape).
Preserve the null/rank/type/[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) [validation](../../../../glossary.md#term-validation) prefix specified below. Derive:

```text
expectedColumnChannels = targetC * kernelDepth * kernelHeight * kernelWidth
expectedColumnCount    = targetDOut * targetHOut * targetWOut
```

where each target output-grid [Dimension](../../../../glossary.md#dimension) uses the exact same target [spatial](../../../../glossary.md#term-spatial-axes) [extent](../../../../glossary.md#term-extent), geometry, and
literal floor/ceil formula as unfold. Require complete [structural equality](../../../../glossary.md#term-structural-equality) with input Dimensions
one and two. Matching dynamic columns produced by `unfold3d` compose exactly; unrelated unresolved
symbols fail rather than creating an unnamed equality constraint. The result [descriptor](../../../../glossary.md#term-descriptor) and
`Fold3dAttrs` both retain the exact supplied `outputShape` reference.

For each logical column value `[n,q,p]`, decode the same `(c,kd,kh,kw)` and `(od,oh,ow)` order and
compute `(id,ih,iw)` above. Contribute exactly once to `[n,c,id,ih,iw]` only when all three spatial
[coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) are in bounds. Every padded or terminal ceil-tail coordinate is excluded
geometrically regardless of its column value. Fold has no [padding](../../../../glossary.md#padding) [scalar](../../../../glossary.md#term-scalar) and never compares a
value to decide inclusion.

### Fold overlap addition

Every [target](../../../../glossary.md#term-target) [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) begins at exact represented [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero). A target with no in-range
contribution retains that zero. Multiple logical column positions mapping to the same target are
distinct contributions and are added in increasing [canonical](../../../../glossary.md#term-canonical-form) flattened input order `[n,q,p]`.
No [reassociation](../../../../glossary.md#term-reassociation), tree reduction, averaging, compensation, [saturation](../../../../glossary.md#term-saturation), or hidden [widening](../../../../glossary.md#term-widening) is part
of the semantic contract.

Type-specific addition is:

- [FLOAT64](../../../../glossary.md#term-float64): sequential [IEEE](../../../../glossary.md#term-ieee)-754 [binary64](../../../../glossary.md#floating-point-precision-and-rounding) addition after each contribution;
- [FLOAT32](../../../../glossary.md#term-float32): sequential IEEE-754 [binary32](../../../../glossary.md#floating-point-precision-and-rounding) addition after each contribution; and
- BFLOAT16: expand the represented accumulator and next operand to FLOAT32, add once in FLOAT32,
  then narrow to BFLOAT16 after every logical contribution.<span data-glossary-navigation="true"> ([accumulator glossary](../../../../glossary.md#accumulator-type))</span>

[NaN payload](../../../../glossary.md#nan-infinity-and-signed-zero), sign, and quiet/signaling preservation remain unspecified, but contribution order,
represented positive-zero initialization, exceptional-value class, [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) result, and
per-contribution [BFLOAT16](../../../../glossary.md#term-bfloat16) [narrowing](../../../../glossary.md#term-narrowing) follow the contract above. This matches the current portable
`FOLD2D` realization and gives later [Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling) [gradients](../../../../glossary.md#term-gradient) one unambiguous overlap-add meaning without
introducing a [pooling](../../../../glossary.md#term-pooling) primitive.

### [Validation](../../../../glossary.md#term-validation), failure, and allocation order

Direct `unfold3d` validates in this order:

1. null-check `input`, then `window`;
2. require [rank](../../../../glossary.md#term-rank)-five [NCDHW](../../../../glossary.md#conv3d--ncdhw) input;
3. require [BFLOAT16](../../../../glossary.md#term-bfloat16), [FLOAT32](../../../../glossary.md#term-float32), or [FLOAT64](../../../../glossary.md#term-float64);
4. derive checked/[canonical](../../../../glossary.md#term-canonical-form) depth, height, then width geometry;
5. derive [channel](../../../../glossary.md#term-channel)-[kernel](../../../../glossary.md#kernel) and flattened-grid [Dimensions](../../../../glossary.md#dimension);
6. construct the exact [descriptor](../../../../glossary.md#term-descriptor) and `UNFOLD3D + Window3dAttrs` [operation](../../../../glossary.md#operation); and
7. delegate once with exact ordered inputs `[input]`.

Typed `unfold3d` inserts `paddingValue` null checking after `window`, then after rank and floating
eligibility requires exact `paddingValue.dataType() == input.dataType()`, before geometry. It
constructs `UNFOLD3D + Unfold3dAttrs` retaining both supplied references.

`fold3d` validates in this order:

1. null-check `input`, `outputShape`, then `window`;
2. require rank-three canonical-column input;
3. require rank-five NCDHW output [Shape](../../../../glossary.md#shape);
4. require BFLOAT16, FLOAT32, or FLOAT64 input;
5. require exact [structural equality](../../../../glossary.md#term-structural-equality) of input and [target](../../../../glossary.md#term-target) [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) Dimensions;
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

Null messages are [parameter](../../../../glossary.md#term-parameter) names. Existing checked-arithmetic, Dimension [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form),
descriptor, operation-[signature](../../../../glossary.md#term-signature), factory, and identifier-exhaustion failures retain their current
types and messages. Every locally decidable failure precedes the factory call and consumes no
[Tensor](../../../../glossary.md#tensor) ID, [producer](../../../../glossary.md#term-producer), descriptor wrapper, or output wrapper. Every success consumes exactly one
fresh ID and creates exactly one one-input/one-output producer [occurrence](../../../../glossary.md#term-occurrence).

### Result [metadata](../../../../glossary.md#term-metadata) and [provenance](../../../../glossary.md#provenance)

Every successful result:

- preserves the exact input [DataType](../../../../glossary.md#data-type--datatype) and `requiresGrad` value;
- has unresolved [layout](../../../../glossary.md#layout), no label, and no [host storage](../../../../glossary.md#host-storage);
- owns a fresh [descriptor](../../../../glossary.md#term-descriptor), [operation](../../../../glossary.md#operation) [occurrence](../../../../glossary.md#term-occurrence), [producer](../../../../glossary.md#term-producer), ID, and [canonical](../../../../glossary.md#term-canonical-form) output wrapper;
- records the exact kind and attributes variant, exact ordered inputs `[input]`, and output index
  zero; and
- is exactly the object returned by its producer's `output(0)`.

Direct unfold retains the exact `Window3dAttrs` reference as operation attrs. Typed unfold retains
the exact window and [scalar](../../../../glossary.md#term-scalar) references inside one new `Unfold3dAttrs`. Fold retains the exact
[target](../../../../glossary.md#term-target) [Shape](../../../../glossary.md#shape) and window references inside one new `Fold3dAttrs`. Repeated equal requests remain
[identity](../../../../glossary.md#term-identity-meanings)-distinct and never inspect or mutate input layout, label, provenance, storage, or values.

## Affected files

Production/[Javadoc](../../../../glossary.md#term-javadoc), exactly six paths:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/Window3dAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/Unfold3dAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/Fold3dAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/WindowTransformKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorWindowExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Behavior and [signature](../../../../glossary.md#term-signature) tests, exactly three paths:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationSignatureTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/layout/WindowTransformSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorWindowExpressionTest.java`

Existing exact public-[Tensor](../../../../glossary.md#tensor) method-count [owners](../../../../glossary.md#term-owner), exactly eighteen paths, change only `210 -> 213`
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

Documentation/[planning](../../../../glossary.md#term-planning), exactly seven paths:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review-only with a reasoned no-change conclusion unless a contradiction requires stopping:
`ARCHITECTURE.md`, current architecture pages and ADRs, [Runtime](../../../../glossary.md#term-runtime) and Training [API](../../../../glossary.md#term-application-programming-interface-api) pages, existing
2D/general-[axis](../../../../glossary.md#term-axis) window, Pool1d/Pool2d/[Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling), [Conv3d](../../../../glossary.md#conv3d--ncdhw), [ScalarValue](../../../../glossary.md#typed-scalar-value--scalarvalue), [Shape](../../../../glossary.md#shape)/[Dimension](../../../../glossary.md#dimension),
[operation](../../../../glossary.md#operation)/signature, Tensor/factory/[producer](../../../../glossary.md#term-producer)/[provenance](../../../../glossary.md#provenance) contracts and tests, [Compiler](../../../../glossary.md#term-compiler) 0005D and
current [gradient](../../../../glossary.md#term-gradient) algebra, Compiler/[CPU](../../../../glossary.md#term-cpu) production and tests, [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance), integration,
architecture tests, [Gradle](../../../../glossary.md#term-gradle)/build files, dependencies, and all other [modules](../../../../glossary.md#term-module).

## Maximum [scope](../../../../glossary.md#term-scope)

Hard ceiling: exactly 34 paths — six production/[Javadoc](../../../../glossary.md#term-javadoc), twenty-one [Model](../../../../glossary.md#term-model) test, and seven
documentation/[planning](../../../../glossary.md#term-planning) paths listed above.

This is an explicit [atomic](../../../../glossary.md#resource-transaction-and-rollback) exception to the normal 18-path guardrail. Three [rank](../../../../glossary.md#term-rank)-specific attrs,
two kind/[signature](../../../../glossary.md#term-signature) meanings, three public methods, one shared construction [owner](../../../../glossary.md#term-owner), exact behavior
tests, eighteen mandatory global method-count locks, and the documentation pass must agree in one
compilable change. Splitting them would leave signatures, public inventory, or the mutually
inverse unfold/fold contract temporarily inconsistent.

The eighteen count-owner files are mechanical count-only edits except for `TensorTest` as stated.
Do not use this allowance for unrelated cleanup. Stop before a thirty-fifth path, another source or
test type, a different existing helper, any non-count change in the other seventeen count owners,
architecture, [dependency](../../../../glossary.md#term-dependency), [Gradle](../../../../glossary.md#term-gradle), cross-[module](../../../../glossary.md#term-module) implementation, or a later detailed task file.

## [Javadoc](../../../../glossary.md#term-javadoc) and documentation requirements

- Fully document all three attrs records, both new kinds, all helper methods, and all three Tensor
  receivers: purpose, NCDHW/rank-three Shapes, geometry, flattening order, direct versus typed
  padding, exact reference retention, fold exclusion/addition, supported types, metadata,
  provenance, validation, allocation effects, and lifecycle boundaries.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [rank glossary](../../../../glossary.md#term-rank); [Shapes glossary](../../../../glossary.md#shape); [padding glossary](../../../../glossary.md#padding); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance); [validation glossary](../../../../glossary.md#term-validation); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Every public record component, constructor parameter, method parameter, result, nullability,
  constraint, and expected failure has meaningful <span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [nullability glossary](../../../../glossary.md#term-nullability))</span>`@param`, `@return`, and `@throws` coverage.
- Update the WindowTransformKind overview without weakening or rewriting the existing 2D and
  general-axis contracts. Explain why rank-specific 3D kinds are added instead of a <span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>`WindowNd`.
- Tensor API adds the three methods, exact formulas and Shapes, a coordinate-order table, direct
  positive-zero versus exact typed-padding example, dynamic NCDHW example, fold overlap example,
  and current-Model versus planned Compiler/backend boundary.<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [Compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend))</span>
- [Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api) records current [Model](../../../../glossary.md#term-model) `UNFOLD3D`/`FOLD3D` metadata and their planned use by Compiler
  0006B2 without claiming current Compiler capture, inference, gradients, higher-order support, or
  execution.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Review glossary entries for NCDHW, window transform, im2col/col2im terminology, padding,
  dilation, literal ceil grids, canonical columns, overlap fold, and positive zero. Extend only
  reusable distinctions; use clear three-dimensional wording such as volumetric columns rather
  than inventing an unexplained acronym.<span data-glossary-navigation="true"> ([window transform glossary](../../../../glossary.md#window-transform); [im2col glossary](../../../../glossary.md#term-im2col); [col2im glossary](../../../../glossary.md#term-col2im); [dilation glossary](../../../../glossary.md#term-dilation); [canonical glossary](../../../../glossary.md#term-canonical-form); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Synchronize capabilities, this task, Model master plan, and roadmap. During implementation keep
  0025K Ready or In progress; mark it Complete only after all Java and documentation evidence
  passes. Model 0026, Compiler 0006B1/0006B2, and CPU 0008G1 remain Draft without detailed task
  files.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Record reasoned no-change conclusions for Runtime/Training APIs, related contracts,
  architecture/ADRs/tests, conformance/integration, Gradle, downstream source/tests, and other
  modules.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

## Acceptance criteria

- The three records expose exactly the declared components, validation order, exact-reference
  retention, and structural-only responsibilities.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- WindowTransformKind has exactly six constants in the required order. Only the new kinds gain the
  exact new signatures, with direct UNFOLD3D attrs before explicit-padding attrs.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [signatures glossary](../../../../glossary.md#term-signature); [padding glossary](../../../../glossary.md#padding))</span>
- Tensor adds exactly three declarations and no alias/default/builder/namespace; the exact public
  declared-method count is 213.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Both unfold forms accept only BFLOAT16/FLOAT32/FLOAT64 rank-five NCDHW input, derive exact static
  floor, literal-ceil, terminal all-padding, and canonical-symbolic rank-three columns, and retain
  exact batch/type/gradient metadata.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [rank glossary](../../../../glossary.md#term-rank); [NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [gradient glossary](../../../../glossary.md#term-gradient); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Direct unfold uses represented positive zero. Typed unfold retains the exact matching
  ScalarValue reference and raw bits and can represent BFLOAT16/FLOAT32/FLOAT64 negative infinity
  without a pooling-specific kind.<span data-glossary-navigation="true"> ([positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [ScalarValue glossary](../../../../glossary.md#typed-scalar-value--scalarvalue); [raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [negative infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [pooling glossary](../../../../glossary.md#term-pooling))</span>
- [Canonical](../../../../glossary.md#term-canonical-form) `q` and `p` flattening and depth-height-width/kernel ordering match the formulas in this
  task for static and unresolved Shapes.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [Shapes glossary](../../../../glossary.md#shape))</span>
- Fold accepts only exact structurally compatible floating rank-three columns and rank-five target
  Shape, retains that exact Shape, excludes every out-of-range padded/ceil-tail coordinate, and
  overlap-adds in canonical input order with the exact type-specific positive-zero and addition
  policy.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Static geometry/products use checked arithmetic; unresolved geometry uses only existing
  canonical Dimension expressions; no host kernel-volume product or unnamed equality constraint
  is introduced.<span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension); [host glossary](../../../../glossary.md#term-host))</span>
- Null/type/rank/batch/channel/grid/fit/overflow failures occur in exact order before factory
  allocation and consume no Tensor ID. Every success consumes one ID and creates one fresh exact
  one-input/output-zero canonical occurrence.<span data-glossary-navigation="true"> ([channel glossary](../../../../glossary.md#term-channel); [overflow glossary](../../../../glossary.md#term-overflow); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Existing general-axis and 2D window behavior, Pool3d semantics, Shape/Dimension algebra,
  ScalarValue, factory/provenance, and all architecture boundaries remain unchanged.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling); [provenance glossary](../../../../glossary.md#provenance))</span>
- Focused tests, exactly one final Model suite after executable stabilization, final Model
  Javadoc, runnable Java 26 metadata example, reflection/<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`javap`, operation and Tensor inventory,
  Markdown headings/links/anchors/fences/newlines/terminology, exact 34 paths, status/order/
  dependencies/task-file absence, and <span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>`git diff --check` pass.
- A separate clean documentation-focused context reuses successful Java evidence, independently
  finalizes all affected Javadocs/docs/glossary/planning, and records no-change conclusions before
  completion.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- No gradient, Compiler, backend, Runtime, architecture, dependency, Gradle, detailed later task,
  or generic WindowNd/PoolNd implementation is present.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [Runtime glossary](../../../../glossary.md#term-runtime); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Focused implementation validation:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest \
  --tests io.github.pho001.synaptik.model.operation.layout.WindowTransformSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorWindowExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorTest
```

After executable Java stabilizes, [run](../../../../glossary.md#term-run) exactly once:

```bash
./gradlew :modules:model:test
```

Tests must cover record reflection/validation/reference retention, enum order and [signatures](../../../../glossary.md#term-signature),
public receiver [shape](../../../../glossary.md#shape) and count locks, every accepted/rejected [DataType](../../../../glossary.md#data-type--datatype), static and symbolic
floor/ceil geometry, terminal all-[padding](../../../../glossary.md#padding) windows, product [overflow](../../../../glossary.md#term-overflow)/zero cases, exact [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)
flattening, padding bits, fold structural compatibility, exclusion/addition meaning, exact [target](../../../../glossary.md#term-target)
Shape, [metadata](../../../../glossary.md#term-metadata), freshness, [producer](../../../../glossary.md#term-producer)/output [identity](../../../../glossary.md#term-identity-meanings), and ID deltas. Numerical fold tests in [Model](../../../../glossary.md#term-model)
lock inspectable semantics and Javadocs rather than pretending Model evaluates values.

The separate documentation-focused pass reuses successful Java evidence unless it changes
executable behavior. After final [Javadoc](../../../../glossary.md#term-javadoc) edits it runs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

It also:

- compiles and runs one Java 26 metadata example covering both unfold variants, fold, Shape,
  attrs, provenance, exact reference retention, unresolved layout, and absent storage;<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [layout glossary](../../../../glossary.md#layout))</span>
- uses reflection and `javap` for the three records, six-kind enum, exact signatures, three Tensor
  declarations, helper visibility/field-free shape, 213-method Tensor count, and 40/115/137
  operation inventory;<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [operation glossary](../../../../glossary.md#operation))</span>
- validates changed Markdown headings, links, anchors, fences, final newlines, trailing whitespace,
  terminology, examples, and current-versus-planned claims;
- confirms exactly the 34 authorized paths and package placement;
- confirms 0025K is the sole detailed Ready/In-progress Model task until completion, 0026 remains
  Draft without a task file, and Compiler 0006B1/0006B2 and CPU 0008G1 remain Draft without task
  files; and<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- confirms the exact cross-plan order `0025I -> 0025J -> 0025K -> Compiler 0006B1 -> Compiler
  0006B2 -> CPU 0008G1 -> CPU 0008H`.

Repository-wide validation is deferred to the next recorded capability [checkpoint](../../../../glossary.md#term-checkpoint) and [CI](../../../../glossary.md#term-continuous-integration-ci). No
[Compiler](../../../../glossary.md#term-compiler), [CPU](../../../../glossary.md#term-cpu), architecture, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), or integration suite is required because this
task changes one [module](../../../../glossary.md#term-module) without dependencies, shared build configuration, backend behavior, or an
architecture boundary.

## Dependencies

- Complete [Model](../../../../glossary.md#term-model) 0023D supplies the public/dynamic [canonical](../../../../glossary.md#term-canonical-form) [rank](../../../../glossary.md#term-rank)-three `unfold2d`/`fold2d`
  contract, exact typed padding attributes, symbolic products, validation ordering, and canonical
  provenance precedent.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [validation glossary](../../../../glossary.md#term-validation); [provenance glossary](../../../../glossary.md#provenance))</span>
- Complete Model 0020A/0020A1 supply excluded-padding maximum and fixed-count average Pool2d
  geometry/numerical policies.
- Complete Model 0025J supplies first-class NCDHW Pool3d geometry, depth-height-width first-winner
  order, current floating types, and the exact downstream need.<span data-glossary-navigation="true"> ([NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling))</span>
- Complete [Compiler](../../../../glossary.md#term-compiler) 0005D and current `PoolingGradientRules` demonstrate the generally useful
  public 2D unfold/fold algebra needed to reconstruct exact maximum winners and distribute
  fixed-count average cotangents.
- Complete CPU 0006B2 supplies source-backed represented positive-zero and type-specific overlap
  addition precedent for current FOLD2D. This task does not add CPU execution.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- Existing Model operation/signature, ScalarValue, Shape/Dimension, Tensor descriptor/factory,
  canonical producer/output, and provenance contracts are stable.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [signature glossary](../../../../glossary.md#term-signature); [ScalarValue glossary](../../../../glossary.md#typed-scalar-value--scalarvalue); [Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [Tensor descriptor glossary](../../../../glossary.md#tensor-descriptor); [producer glossary](../../../../glossary.md#term-producer))</span>

## Follow-up tasks

- Compiler 0006B1 remains Draft without a detailed task file. It owns forward capture/inference/
  final validation for <span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation))</span>`MAX_POOL3D` and `AVERAGE_POOL3D` plus fail-closed backward before any
  derivative allocation.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- Compiler 0006B2 remains Draft without a detailed task file after 0025K and 0006B1. It owns exact
  Pool3d gradients through these public transforms: average distributes the cotangent divided by
  the logical <span data-glossary-navigation="true"> ([Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling); [gradients glossary](../../../../glossary.md#term-gradient); [cotangent glossary](../../../../glossary.md#cotangent))</span>`kernelDepth * kernelHeight * kernelWidth`; maximum unfolds exact input with typed
  negative-infinity padding, separately reconstructs in-bounds eligibility, matches the exact
  same-occurrence Pool3d output including NaN and signed zero, selects the first depth-height-width
  candidate, routes the cotangent, and fold3d overlap-adds to the exact input Shape.<span data-glossary-navigation="true"> ([infinity glossary](../../../../glossary.md#term-infinity); [padding glossary](../../../../glossary.md#padding); [occurrence glossary](../../../../glossary.md#term-occurrence); [NaN glossary](../../../../glossary.md#term-nan); [signed zero glossary](../../../../glossary.md#term-signed-zero); [Shape glossary](../../../../glossary.md#shape))</span>
- Compiler 0006B2 must preserve Pool3d's specified accumulator/division/narrowing, exceptional-
  value, padding, first-winner, and occurrence-identity contracts and must update the source-backed
  derivative and higher-order inventories. This task selects no derivative formula implementation.<span data-glossary-navigation="true"> ([accumulator glossary](../../../../glossary.md#accumulator-type); [narrowing glossary](../../../../glossary.md#term-narrowing); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- CPU 0008G1 remains Draft without a detailed task file after both Compiler tasks. It owns exact
  Pool1d composition validation and first-class Pool3d generated execution, including its own
  algorithm, Class-File, conformance, and performance evidence.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Pool1d composition glossary](../../../../glossary.md#pool1d-composition); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- [Model](../../../../glossary.md#term-model) 0026 remains Draft without a detailed task file for [FLOAT16](../../../../glossary.md#term-float16) and [mixed precision](../../../../glossary.md#term-mixed-precision).

Do not create another detailed specification during implementation.

## Architecture impact

Expected impact: None.

This task extends existing [Model](../../../../glossary.md#term-model)-owned [rank](../../../../glossary.md#term-rank)-specific window semantics and public [Tensor](../../../../glossary.md#tensor) [metadata](../../../../glossary.md#term-metadata)
construction without changing [module](../../../../glossary.md#term-module) [ownership](../../../../glossary.md#term-ownership-meanings), [dependency direction](../../../../glossary.md#java-platform-and-build-tools), [lifecycle](../../../../glossary.md#lifecycle), or architecture.
If implementation requires an architecture edit, cross-layer constraint service, new [Shape](../../../../glossary.md#shape)
expression, [backend](../../../../glossary.md#backend) policy, [compiler](../../../../glossary.md#term-compiler) behavior, another module, or [scope](../../../../glossary.md#term-scope) beyond the exact ceiling,
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
  a current need.<span data-glossary-navigation="true"> ([Canonical glossary](../../../../glossary.md#term-canonical-form); [rank glossary](../../../../glossary.md#term-rank); [carrier glossary](../../../../glossary.md#term-storage-carrier); [Shape glossary](../../../../glossary.md#shape))</span>
- New rank-specific kinds and attrs are required. Reusing Pool3d attrs would make general window
  transforms inherit pooling numerical meaning; reusing 2D attrs cannot represent depth. A
  generic WindowNd abstraction is not justified by this one concrete rank-three extension.<span data-glossary-navigation="true"> ([Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling); [pooling glossary](../../../../glossary.md#term-pooling))</span>
- Direct positive-zero and explicit typed-padding unfold variants mirror current UNFOLD2D. The
  typed variant is the general capability that later represents maximum-pool negative-infinity
  padding without a pooling-specific primitive.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [infinity glossary](../../../../glossary.md#term-infinity))</span>
- Fold retains an explicit exact target Shape because canonical column dimensions do not identify
  trailing uncovered target coordinates. Structural equality, not an unnamed Model constraint,
  is required for unresolved columns.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [Structural equality glossary](../../../../glossary.md#term-structural-equality); [Model glossary](../../../../glossary.md#term-model))</span>
- Fold contribution order and type-specific addition follow the current portable FOLD2D
  realization so signed zero, exceptional values, and BFLOAT16 per-contribution narrowing are not
  left ambiguous for later Pool3d gradients.<span data-glossary-navigation="true"> ([signed zero glossary](../../../../glossary.md#term-signed-zero); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [narrowing glossary](../../../../glossary.md#term-narrowing); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Successive [Dimension](../../../../glossary.md#dimension) multiplication avoids a false [host](../../../../glossary.md#term-host)-`long` kernel-volume requirement while
  still rejecting an actually unrepresentable result Dimension through existing Shape rules.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel))</span>
- The 34-path ceiling is an [atomic](../../../../glossary.md#resource-transaction-and-rollback) exception dominated by eighteen mandatory [Tensor](../../../../glossary.md#tensor) count locks.

## Known limitations

- Only floating rank-five NCDHW source/target tensors with symmetric intrinsic padding are
  included. There is no NDHWC, asymmetric intrinsic padding, FLOAT16, integral, BOOL, sparse,
  quantized, or arbitrary-rank form.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [target glossary](../../../../glossary.md#term-target); [tensors glossary](../../../../glossary.md#tensor); [padding glossary](../../../../glossary.md#padding); [FLOAT16 glossary](../../../../glossary.md#term-float16); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Dynamic Dimensions retain exact formulas and non-negative-numerator obligations but do not bind
  or evaluate them. Fold rejects unrelated unresolved column symbols.<span data-glossary-navigation="true"> ([Dimensions glossary](../../../../glossary.md#dimension))</span>
- Window transforms remain metadata only. No current Compiler or backend support is implied by
  Model construction, and no numerical value is materialized in this task.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [Compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [Model glossary](../../../../glossary.md#term-model))</span>
- Direct unfold has only positive-zero padding; any other padding uses the exact typed overload.
  Fold always excludes out-of-range coordinates and has no configurable padding contribution.<span data-glossary-navigation="true"> ([coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Repository-wide [validation](../../../../glossary.md#term-validation) remains deferred to the next capability [checkpoint](../../../../glossary.md#term-checkpoint) or [CI](../../../../glossary.md#term-continuous-integration-ci).

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation Java evidence was reused without rerunning tests because this documentation pass
  changed no executable Java. The stabilized `modules/model/build/test-results/test/TEST-*.xml`
  artifacts from 2026-08-30 11:16:28 CEST record the final<span data-glossary-navigation="true"> ([artifacts glossary](../../../../glossary.md#term-artifact))</span>
  `./gradlew :modules:model:test` result: 138 suites, 1,103 tests, zero skipped, zero failures, and
  zero errors. The reports include `OperationSignatureTest`, `WindowTransformSemanticsTest`,
  `TensorWindowExpressionTest`, and `TensorTest`; executable Java did not change afterward.
- `./gradlew :modules:model:javadoc` passed after the documentation review with
  `BUILD SUCCESSFUL`; `compileJava` and [`javadoc`](../../../../glossary.md#term-javadoc) were up to date against the stabilized sources.
- The Java 26 `/tmp/Task0025KMetadataExample.java` compiled against
  `modules/model/build/classes/java/main` and ran successfully. It printed direct and typed column
  Shapes <span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape))</span>`Shape[1, 8, 8]`, fold [target](../../../../glossary.md#term-target) `Shape[1, 1, 3, 3, 3]`, and `metadata-ok` after checking
  both unfold variants, fold, exact Shape/window/padding references, operation kinds, canonical
  producer output, unresolved layout, and absent storage.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [operation glossary](../../../../glossary.md#operation); [canonical glossary](../../../../glossary.md#term-canonical-form); [producer glossary](../../../../glossary.md#term-producer); [layout glossary](../../../../glossary.md#layout))</span>
- `/tmp/Task0025KInventoryCheck.java` compiled and ran successfully. Reflection reported exactly
  `families=40 constants=115 signatures=137 tensorMethods=213`, verified the six-kind window enum
  order, two UNFOLD3D and one FOLD3D signatures, three record component surfaces, three exact
  Tensor declarations, and the package-private final field-free helper shape.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>
- `javap -public` confirmed the three record constructors, public enum [constants](../../../../glossary.md#term-constant) `UNFOLD3D` and
  `FOLD3D`, and exactly the two `unfold3d` plus one `fold3d` [Tensor](../../../../glossary.md#tensor) declarations.
- `ruby /tmp/task0025k_markdown_check.rb` passed all seven changed documentation/planning files:
  local Markdown targets and anchors resolved, fences were balanced, and final newlines were
  present. The focused terminology/example review confirmed NCDHW expansion at first use,
  canonical volumetric-column Shape and flattening, literal floor/ceil grids, direct positive-zero
  versus typed padding, fold exclusion/addition, symbolic obligations, and current-versus-planned
  claims.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [NCDHW glossary](../../../../glossary.md#conv3d--ncdhw))</span>
- The exact-[scope](../../../../glossary.md#term-scope) comparison against `/tmp/task0025k_expected_paths.txt` passed with 34 paths:
  six production/Javadoc, twenty-one Model tests, and seven documentation/planning paths. All new
  production types match the task package map.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- Later-task and ordering checks passed: Model 0025I, 0025J, and 0025K are Complete in order;
  Compiler 0006B1 and 0006B2, CPU 0008G1 and 0008H, and Model 0026 remain Draft; no detailed task
  file exists for Model 0026, Compiler 0006B1/0006B2, or CPU 0008G1. The next cross-plan frontier
  is Compiler 0006B1, without a detailed specification.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Downstream absence/import scans found no UNFOLD3D/FOLD3D or 3D-window-attribute production/test
  adoption in Compiler, Runtime, Training, CPU, or shared testing. <span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>`PoolingGradientRules` remains
  the unchanged 2D source-backed precedent; Compiler 0006B1/0006B2 and CPU 0008G1 remain planning
  rows only.
- Runtime and Training API pages require no change because this task adds no prepared/run contract,
  execution state, optimizer, or training orchestration API. Architecture pages, ADRs, and
  architecture tests require no change because ownership, dependency direction, lifecycle, and
  module boundaries are unchanged. Backend conformance and integration tests require no change
  because no backend capability or end-to-end execution was added. Gradle/build files and
  dependencies require no change because all implementation stays inside the existing Model
  module surface. Downstream production/tests and every other module require no change because
  Compiler adoption, gradients, and CPU execution are explicitly deferred.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [run glossary](../../../../glossary.md#term-run); [optimizer glossary](../../../../glossary.md#term-optimizer); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [lifecycle glossary](../../../../glossary.md#lifecycle); [module glossary](../../../../glossary.md#term-module); [Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [backend capability glossary](../../../../glossary.md#backend-capability); [Gradle glossary](../../../../glossary.md#term-gradle); [Model module glossary](../../../../glossary.md#model-module-buffer-and-scan-distinctions); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Final changed-Markdown validation, exact-scope/status scans, downstream absence scans, and
  `git diff --check` passed after this completion record was written.

## Implementation notes

- Added [immutable](../../../../glossary.md#term-immutable) `Window3dAttrs`, `Unfold3dAttrs`, and `Fold3dAttrs`; appended UNFOLD3D/FOLD3D
  without changing earlier enum order or signatures; and added the exact three Tensor receivers
  through the existing <span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [Tensor glossary](../../../../glossary.md#tensor))</span>`TensorWindowExpressions` construction [owner](../../../../glossary.md#term-owner).
- Static and canonical-symbolic depth-height-width geometry, canonical rank-three columns,
  structural fold compatibility, exact metadata/reference retention, validation-before-allocation,
  and one-input/output-zero provenance match the task contract.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [rank glossary](../../../../glossary.md#term-rank); [metadata glossary](../../../../glossary.md#term-metadata); [validation glossary](../../../../glossary.md#term-validation); [provenance glossary](../../../../glossary.md#provenance))</span>
- The clean documentation-focused pass independently reviewed all six production Javadocs and
  finalized the Tensor API, Compile API, glossary, capabilities, task, Model master plan, and
  roadmap. It changed no executable Java and therefore reused the successful Model test evidence.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [Model glossary](../../../../glossary.md#term-model))</span>

## Completion summary

- Completed changes: public [NCDHW](../../../../glossary.md#conv3d--ncdhw) `unfold3d`/`fold3d` Model algebra, exact attributes/signatures,
  three Tensor receivers, validation/metadata construction, tests/count locks, finalized Javadocs,
  API/glossary explanations, and synchronized planning.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [signatures glossary](../../../../glossary.md#term-signature); [Tensor glossary](../../../../glossary.md#tensor); [validation glossary](../../../../glossary.md#term-validation); [metadata glossary](../../../../glossary.md#term-metadata); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: exactly 34 authorized paths: six production/Javadoc, twenty-one Model
  tests, and seven documentation/planning files.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Tests and validation: reused the stabilized final Model result of 1,103 tests across 138 suites
  with zero failures/errors; final Model Javadoc, runnable Java 26 metadata example,
  reflection/inventory, `javap`, Markdown, scope/status/order/absence scans, and whitespace checks
  passed.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: completed in this mandatory separate clean documentation context;
  the six production Javadocs and all seven authorized documentation/planning paths were reviewed
  and finalized.
- Documentation impact: Tensor and Compile API pages now distinguish current Model metadata from
  Draft Compiler 0006B1/0006B2 and CPU 0008G1; Runtime and Training APIs need no change for the
  ownership reasons recorded above.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [Compiler glossary](../../../../glossary.md#term-compiler); [CPU glossary](../../../../glossary.md#term-cpu); [Runtime glossary](../../../../glossary.md#term-runtime); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Javadoc review: all six changed production files accurately document parameters, results,
  failures, Shapes, padding, fold addition, metadata/provenance, and lifecycle boundaries.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter); [Shapes glossary](../../../../glossary.md#shape); [padding glossary](../../../../glossary.md#padding); [provenance glossary](../../../../glossary.md#provenance); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Glossary impact: the window-transform entry now covers NCDHW volumetric columns, 3D flattening,
  typed padding, literal-ceil exclusion, and type-specific overlap fold without inventing a new
  acronym.
- Unresolved issues: None.
- Follow-up required: None. Compiler 0006B1 remains the next Draft frontier and does not yet have
  a detailed task specification.

Status: Complete
