# Task 0017N: [Unfold and Fold](../../../../glossary.md#window-transform) [Tensor](../../../../glossary.md#tensor) Expressions

**Editorial reading key (later context, not a status update).**

Unfold exposes sliding windows; fold adds window contributions into target coordinates, so overlapping windows make fold more than an inverse reshape. The application supplies axis/window/target parameters, and the helper derives result descriptions without sampling padding or adding overlaps. The same forward semantic primitive can later be used by Compiler in a derivative, but this task installs no derivative rule or saved-value lifecycle. The application calls the public Tensor method; its model helper validates available descriptions and returns a fresh storage-free expression with exact input references. It neither copies input values nor calculates the result. Compiler consumes this provenance later; actual storage and execution are separate downstream concerns.

**Window reading example.** Let an exact real sequence have integer length $D=5$, window width $w=3$, and step $s=1$. The number of fitting windows is $m=\lfloor(D-w)/s\rfloor+1=3$. Unfold exposes $(a,b,c)$, $(b,c,d)$ and $(c,d,e)$; folding those unchanged windows by addition gives $(a,2b,3c,2d,e)$. Each coefficient counts appearances, so fold is not generally an identity. This conceptual example is not execution evidence.

Focused current reading: [Current window-transform explanation](../../../../reference/tensor/windows.md#window-transform-expressions). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Add the public storage-free [Tensor](../../../../glossary.md#tensor) expression surface for general-[axis](../../../../glossary.md#term-axis) [unfold and fold](../../../../glossary.md#window-transform), [NCHW](../../../../glossary.md#term-nchw)
two-dimensional unfold into [canonical](../../../../glossary.md#term-canonical-form) columns, and two-dimensional fold back into an explicit NCHW
[Shape](../../../../glossary.md#shape).

The four methods validate only locally provable [model](../../../../glossary.md#term-model) facts, derive [immutable](../../../../glossary.md#term-immutable) result Shapes, create
unresolved [descriptors](../../../../glossary.md#term-descriptor), attach exact typed [operation](../../../../glossary.md#operation) [provenance](../../../../glossary.md#provenance), and return fresh Tensor [identities](../../../../glossary.md#term-identity-meanings).
They do not read values, allocate result storage, perform scatter-add, materialize windows, define
[gradient](../../../../glossary.md#term-gradient) rules, [capture](../../../../glossary.md#term-capture-meanings) a [compiled graph](../../../../glossary.md#compiled-graph--compiledgraphmodel), or execute [kernels](../../../../glossary.md#kernel).

## [Scope](../../../../glossary.md#term-scope)

- Add exactly four public instance methods to [Tensor](../../../../glossary.md#tensor):
  - `unfold(int axis, long size, long step)`;
  - `foldAxis(int axis, long outputSize, long step)`;
  - `unfold2d(Window2dAttrs window)`;
  - `fold2d(Shape outputShape, Window2dAttrs window)`.
- Add one package-private final field-free `TensorWindowExpressions` helper in the tensor package.
- Normalize general-[axis](../../../../glossary.md#term-axis) raw axes locally and construct exact `UnfoldAxisAttrs` or `FoldAxisAttrs`.
- Support `unfold` for every current [DataType](../../../../glossary.md#data-type--datatype) and `foldAxis` for floating or integral DataTypes.
- Require a static selected/window-count axis for general-axis operations while preserving exact
  unaffected Dimension references, including dynamic dimensions.
- Support zero-sized public `foldAxis` output only for a zero window-count [dimension](../../../../glossary.md#dimension).
- Require floating input for `unfold2d` and `fold2d`.
- Require [rank](../../../../glossary.md#term-rank)-four [NCHW](../../../../glossary.md#term-nchw) input for `unfold2d`, with static channel/height/width dimensions while
  preserving the exact batch Dimension, including a dynamic batch.<span data-glossary-navigation="true"> ([channel glossary](../../../../glossary.md#term-channel); [batch Dimension glossary](../../../../glossary.md#batch-dimension); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- Require rank-three [canonical](../../../../glossary.md#term-canonical-form) columns and an explicit rank-four NCHW output [Shape](../../../../glossary.md#shape) for `fold2d`.
- Validate exact batch, channel-window, and window-count compatibility using checked long
  arithmetic and the completed `Window2dAttrs` geometry.
- Leave every result layout unresolved because all four operations describe materialized output,
  not alias-view geometry.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [alias glossary](../../../../glossary.md#aliasing-and-arena); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- Preserve exact input DataType and [gradient](../../../../glossary.md#term-gradient) eligibility in every result [descriptor](../../../../glossary.md#term-descriptor).
- Create exact one-input [provenance](../../../../glossary.md#provenance) and delegate final construction once to
  `TensorFactory.createDerived` with no label or storage.
- Add one focused same-package expression test and update TensorTest only for exact public API
  shape.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Finalize affected Javadocs, Tensor API, Compile API status, glossary, task evidence, master plan,
  and roadmap through the mandatory independent documentation pass.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- executing window extraction, padding reads, scatter-add, overlap accumulation, zero filling,
  im2col, col2im, or any numerical value behavior<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [im2col glossary](../../../../glossary.md#term-im2col); [col2im glossary](../../../../glossary.md#term-col2im))</span>
- allocating, attaching, copying, aliasing, slicing, or inspecting host/device storage; resolved
  LayoutDescriptor, strides, offsets, materialization policy, or liveness<span data-glossary-navigation="true"> ([aliasing glossary](../../../../glossary.md#term-aliasing); [host glossary](../../../../glossary.md#term-host); [device glossary](../../../../glossary.md#term-device); [strides glossary](../../../../glossary.md#term-stride); [materialization glossary](../../../../glossary.md#materialization); [liveness glossary](../../../../glossary.md#term-liveness))</span>
- [gradient](../../../../glossary.md#term-gradient) rules, backward traversal, [compiler](../../../../glossary.md#term-compiler)-generated `FOLD_AXIS`, task-0023 implementation,
  autograd, optimizer, or training execution<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- graph capture, CompiledNode/GraphValue construction, ID grouping, compiler passes,
  canonicalization, decomposition, fusion, planning, prepare, runtime, backend lowering/kernels,
  engine, trace, ONNX, or conformance behavior<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [decomposition glossary](../../../../glossary.md#term-decomposition); [fusion glossary](../../../../glossary.md#term-fusion); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel); [engine glossary](../../../../glossary.md#term-engine); [trace glossary](../../../../glossary.md#trace); [ONNX glossary](../../../../glossary.md#term-onnx); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- adding window defaults, overloads, builders, `Window2dOptions`, asymmetric padding, other padding
  modes, multiple general axes, reverse windows, adaptive windows, or im2col/col2im aliases<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis); [aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
- permitting BOOL foldAxis accumulation, integral unfold2d/fold2d, dynamic transformed spatial or
  channel dimensions, symbolic constraint creation, or runtime shape binding<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [spatial glossary](../../../../glossary.md#term-spatial-axes); [channel glossary](../../../../glossary.md#term-channel); [dimensions glossary](../../../../glossary.md#dimension); [shape glossary](../../../../glossary.md#shape); [binding glossary](../../../../glossary.md#term-binding))</span>
- changing declarations, logic, imports, or tests in completed WindowTransformKind,
  UnfoldAxisAttrs, FoldAxisAttrs, Window2dAttrs, or Fold2dAttrs; their Javadocs may receive only
  the authorized current-versus-planned temporal correction for completed public expressions
- changing Shape, TensorDescriptor, TensorFactory, TensorProvenance, Operation, or another
  existing Java contract/test<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation))</span>
- dependencies, Gradle/build options, preview/incubator features, architecture changes, another
  module, unrelated documentation, task-0018 specification, or later implementation<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [features glossary](../../../../glossary.md#term-software-feature); [module glossary](../../../../glossary.md#term-module))</span>

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
- [Task 0001](0001-data-type-model.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0003](0003-layout-descriptor-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0017M](0017m-unfold-and-fold-semantics.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only legacy branch exposes:

```java
Tensor unfold(int axis, int size, int step)
Tensor unfold2d(Window2dOptions options)
Tensor fold2d(int[] outputShape, Window2dOptions options)
```

Task 0017M preserved these three meanings with long-valued [model](../../../../glossary.md#term-model) geometry and intentionally added
the symmetric public/[compiler](../../../../glossary.md#term-compiler)-facing `FOLD_AXIS` semantic. This task exposes the new public form:

```java
Tensor foldAxis(int axis, long outputSize, long step)
```

Legacy evidence establishes general-[axis](../../../../glossary.md#term-axis) output ordering, [NCHW](../../../../glossary.md#term-nchw) [im2col](../../../../glossary.md#term-im2col) column ordering, symmetric
[padding](../../../../glossary.md#padding), [stride](../../../../glossary.md#term-stride)/[dilation](../../../../glossary.md#term-dilation)/ceil-mode formulas, floating two-dimensional restrictions, and overlap-
summing fold behavior. The new code is implemented from scratch against current [Shape](../../../../glossary.md#shape), [descriptor](../../../../glossary.md#term-descriptor),
[provenance](../../../../glossary.md#provenance), and package contracts. Legacy [mutable](../../../../glossary.md#term-mutable) arrays, [graph](../../../../glossary.md#graph) builders, [gradient](../../../../glossary.md#term-gradient) callbacks,
storage/[materialization](../../../../glossary.md#materialization), compiler/[lowering](../../../../glossary.md#lowering) code, and [kernels](../../../../glossary.md#kernel) are not copied.

## Architecture constraints

- Production remains in `modules/model`, which owns public Tensor expressions and backend-neutral
  semantic metadata.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- [Tensor](../../../../glossary.md#tensor) remains public [mutable](../../../../glossary.md#term-mutable) [API](../../../../glossary.md#term-application-programming-interface-api) state and does not become an [IR](../../../../glossary.md#term-intermediate-representation-ir) [node](../../../../glossary.md#node).
- Every method constructs metadata only. The model does not execute window values or allocate
  physical result buffers.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [buffers glossary](../../../../glossary.md#term-buffer))</span>
- Result layout remains unresolved even for fully static Shapes. These operations request
  materialized logical arrangements rather than proving alias views.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [views glossary](../../../../glossary.md#term-view-meanings))</span>
- The helper may use temporary arrays and checked arithmetic during local Shape derivation but
  stores no index, cache, service, registry, graph, or runtime state.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [registry glossary](../../../../glossary.md#term-registry); [graph glossary](../../../../glossary.md#graph); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- General-[axis](../../../../glossary.md#term-axis) [normalization](../../../../glossary.md#term-normalization) is relative to the input [rank](../../../../glossary.md#term-rank) for unfold and to [target](../../../../glossary.md#term-target) rank
  `inputRank - 1` for foldAxis. Negative foldAxis axes therefore never address the final input
  window dimension.<span data-glossary-navigation="true"> ([dimension glossary](../../../../glossary.md#dimension))</span>
- `FOLD_AXIS` is one semantic identity shared by its public expression and later task-0023 compiler
  generation. This task does not implement compiler insertion or gradient behavior.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [compiler glossary](../../../../glossary.md#term-compiler); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- `Window2dAttrs` is both the exact semantic geometry and the public immutable geometry value; no
  duplicate options type is introduced.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- Package direction remains tensor -> datatype + [shape](../../../../glossary.md#shape) + [operation](../../../../glossary.md#operation)/[layout](../../../../glossary.md#layout) + java.base.
- Stop if implementation requires changing a completed semantic/Shape/descriptor/factory contract,
  another production helper, another test, dependency, architecture decision, or cross-layer code.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

No package is added or moved.

- `io.github.pho001.synaptik.model.tensor.Tensor` — four new public expression delegates only.
- `io.github.pho001.synaptik.model.tensor.TensorWindowExpressions` — package-private stateless
  validation, Shape derivation, semantic composition, provenance, and derived construction.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Shape glossary](../../../../glossary.md#shape); [provenance glossary](../../../../glossary.md#provenance))</span>
- `TensorWindowExpressionTest` — same-package behavioral and structural tests for all four methods.
- `TensorTest` — exact public [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api) reflection inventory only.

The [operation](../../../../glossary.md#operation) semantic contracts remain in `model.operation.layout`; no semantic type moves into
the public tensor package.

## Required contract

### Public [Tensor](../../../../glossary.md#tensor) surface

Add exactly:

```java
public Tensor unfold(int axis, long size, long step)
public Tensor foldAxis(int axis, long outputSize, long step)
public Tensor unfold2d(Window2dAttrs window)
public Tensor fold2d(Shape outputShape, Window2dAttrs window)
```

Each method delegates exactly once to its matching package-private helper entry and performs no
[validation](../../../../glossary.md#term-validation), arithmetic, [operation](../../../../glossary.md#operation) construction, or factory call itself. Add no overload, static
[facade](../../../../glossary.md#term-facade), default geometry, [alias](../../../../glossary.md#aliasing-and-arena), or convenience method.

### Helper surface

Create one field-free package-private final class with one private zero-[argument](../../../../glossary.md#term-argument) constructor and
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
type, generic utility, public/package-private extra method, [cache](../../../../glossary.md#term-cache), or state.

### General-[axis](../../../../glossary.md#term-axis) unfold

[Validation](../../../../glossary.md#term-validation) and construction order is exact:

1. null-check input with message `input`;
2. require input [rank](../../../../glossary.md#term-rank) at least one with message `unfold requires rank at least 1`;
3. normalize raw axis against input [Shape](../../../../glossary.md#shape) using the current positive/negative axis convention;
4. construct `UnfoldAxisAttrs(normalizedAxis, size, step)`, delegating exact size/step validation;
5. require the selected [Dimension](../../../../glossary.md#dimension) to be static, otherwise fail with
   `unfold requires static selected dimension at axis <axis>`;
6. reject size greater than selected [extent](../../../../glossary.md#term-extent) with
   `unfold size <size> exceeds selected dimension <extent>`;
7. calculate window count as `((extent - size) / step) + 1` using checked addition;
8. derive result Shape, [Operation](../../../../glossary.md#operation), unresolved [descriptor](../../../../glossary.md#term-descriptor), [provenance](../../../../glossary.md#provenance), and fresh [Tensor](../../../../glossary.md#tensor).

Every current [DataType](../../../../glossary.md#data-type--datatype) is accepted. Preserve exact DataType and requiresGrad. Replace only the
selected Dimension with a new StaticDimension(windowCount), preserve every other Dimension by
exact reference, and append a new StaticDimension(size). Input Shape `[2,5,3]`, axis 1, size 3,
step 1 produces `[2,3,3,3]`.

Zero selected extent cannot accept positive size and therefore fails the size-fit check. A step
larger than the remaining traversal is valid and produces one window when size fits.

### General-[axis](../../../../glossary.md#term-axis) fold

[Validation](../../../../glossary.md#term-validation) and construction order is exact:

1. null-check input with message `input`;
2. require input [rank](../../../../glossary.md#term-rank) at least two with message `foldAxis requires rank at least 2`;
3. define [target](../../../../glossary.md#term-target) rank as `inputRank - 1` and normalize raw axis against that target rank;
4. construct `FoldAxisAttrs(normalizedAxis, outputSize, step)`, delegating exact output/step
   validation;
5. accept only floating or integral input; reject [BOOL](../../../../glossary.md#data-type--datatype) with
   `foldAxis requires floating or integral input: BOOL`;
6. require the selected window-count [Dimension](../../../../glossary.md#dimension) and final window-size Dimension to be static;
7. require positive final window size;
8. validate outputSize/window-count/window-size/step compatibility;
9. derive result [Shape](../../../../glossary.md#shape), [Operation](../../../../glossary.md#operation), unresolved [descriptor](../../../../glossary.md#term-descriptor), [provenance](../../../../glossary.md#provenance), and fresh [Tensor](../../../../glossary.md#tensor).

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
equality with the selected input [extent](../../../../glossary.md#term-extent). Mismatch uses:

```text
foldAxis window count <actual> does not match output size and window geometry: expected=<expected>
```

Remove the final input window Dimension, replace the selected window-count Dimension with new
StaticDimension(outputSize), and preserve every other target Dimension by exact reference.
Accept [FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32), [BFLOAT16](../../../../glossary.md#term-bfloat16), [INT32](../../../../glossary.md#data-type--datatype), and [INT64](../../../../glossary.md#data-type--datatype); reject BOOL. Preserve exact [DataType](../../../../glossary.md#data-type--datatype) and
requiresGrad. This constructs scatter-add semantics but performs no addition.

### Shared [axis](../../../../glossary.md#term-axis) [normalization](../../../../glossary.md#term-normalization)

`normalizeAxis(axis, rank)` follows [Shape](../../../../glossary.md#shape) semantics exactly: negative values add [rank](../../../../glossary.md#term-rank) once; values
outside `[-rank, rank-1]` throw:

```text
Axis <axis> is outside shape rank <rank>
```

Use long intermediate normalization so Integer.MIN_VALUE cannot [overflow](../../../../glossary.md#term-overflow). The helper is necessary
for foldAxis because its [target](../../../../glossary.md#term-target) rank excludes the final window [dimension](../../../../glossary.md#dimension).

### Two-dimensional unfold

[Validation](../../../../glossary.md#term-validation) and construction order is exact:

1. null-check input, then window, with messages `input` and `window`;
2. require input [rank](../../../../glossary.md#term-rank) four with `unfold2d requires rank-4 NCHW input`;
3. require floating input through `validateFloating`;
4. preserve exact [batch Dimension](../../../../glossary.md#batch-dimension), which may be static or dynamic;
5. require static [channel](../../../../glossary.md#term-channel), height, and width [Dimensions](../../../../glossary.md#dimension);
6. derive checked window channels and output [spatial](../../../../glossary.md#term-spatial-axes)/window counts;
7. create exact `UNFOLD2D` [Operation](../../../../glossary.md#operation) with the supplied Window2dAttrs reference;
8. create unresolved [descriptor](../../../../glossary.md#term-descriptor), one-input [provenance](../../../../glossary.md#provenance), and fresh [Tensor](../../../../glossary.md#tensor).

Static-dimension failures use:

```text
unfold2d requires static channel dimension at axis 1
unfold2d requires static height dimension at axis 2
unfold2d requires static width dimension at axis 3
```

The result [Shape](../../../../glossary.md#shape) is:

```text
[exact batch Dimension,
 channel * kernelHeight * kernelWidth,
 outputHeight * outputWidth]
```

All arithmetic is checked. Input `[1,1,3,3]` with a 2x2 unit-[stride](../../../../glossary.md#term-stride), zero-[padding](../../../../glossary.md#padding), unit-[dilation](../../../../glossary.md#term-dilation),
floor-mode window produces `[1,4,4]`. Preserve exact floating [DataType](../../../../glossary.md#data-type--datatype) and requiresGrad.

### Two-dimensional fold

[Validation](../../../../glossary.md#term-validation) and construction order is exact:

1. null-check input, outputShape, then window with exact [parameter](../../../../glossary.md#term-parameter)-name messages;
2. require [rank](../../../../glossary.md#term-rank)-three input with `fold2d requires rank-3 canonical column input`;
3. require rank-four outputShape with `fold2d outputShape must be rank-4 NCHW`;
4. require floating input;
5. require structural equality between input and output batch Dimensions, including matching
   dynamic symbols, otherwise fail with<span data-glossary-navigation="true"> ([structural equality glossary](../../../../glossary.md#term-structural-equality); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
   `fold2d output batch dimension must match column batch dimension`;
6. require static input column-[channel](../../../../glossary.md#term-channel) and column-count [Dimensions](../../../../glossary.md#dimension);
7. require static output channel, height, and width Dimensions;
8. calculate and validate expected channel-window and window-count extents;
9. construct exact Fold2dAttrs with the exact outputShape/window references;
10. create unresolved descriptor retaining the exact outputShape, one-input provenance, and fresh
    Tensor.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [Tensor glossary](../../../../glossary.md#tensor))</span>

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

Preserve exact floating [DataType](../../../../glossary.md#data-type--datatype) and requiresGrad. The output [descriptor](../../../../glossary.md#term-descriptor) retains the exact supplied
[Shape](../../../../glossary.md#shape) reference. Fold overlap summation is semantic only and is not executed here.

### Floating and numeric [validation](../../../../glossary.md#term-validation)

`validateFloating` accepts exactly [FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32), and [BFLOAT16](../../../../glossary.md#term-bfloat16). Failure is:

```text
<operation> requires floating input: <dataType>
```

`validateNumeric` accepts floating and integral types. Failure is:

```text
<operation> requires floating or integral input: <dataType>
```

The [operation](../../../../glossary.md#operation) [argument](../../../../glossary.md#term-argument) is private [constant](../../../../glossary.md#term-constant) text supplied only as `foldAxis`, `unfold2d`, or
`fold2d`; do not expose string [dispatch](../../../../glossary.md#term-dispatch) or a [registry](../../../../glossary.md#term-registry).

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

Floor mode returns `numerator / stride + 1`. Ceil mode performs quotient/remainder [rounding](../../../../glossary.md#term-rounding) and
checked addition; do not use `numerator + stride - 1`, which may [overflow](../../../../glossary.md#term-overflow). [Kernel](../../../../glossary.md#kernel), [stride](../../../../glossary.md#term-stride),
[padding](../../../../glossary.md#padding), and [dilation](../../../../glossary.md#term-dilation) were already validated by Window2dAttrs; this helper performs no duplicate
[parameter](../../../../glossary.md#term-parameter) [validation](../../../../glossary.md#term-validation).

### Common result construction

Every successful call creates exactly one [Operation](../../../../glossary.md#operation), one TensorDescriptor with Optional.empty()
[layout](../../../../glossary.md#layout), one TensorProvenance with ordered inputs `[input]`, and invokes
`TensorFactory.createDerived(descriptor, Optional.empty(), provenance)` exactly once.

Return a fresh unlabeled, storage-free [Tensor](../../../../glossary.md#tensor) on every valid call, including identical repeated
requests. Do not inspect input label, [provenance](../../../../glossary.md#provenance), [host storage](../../../../glossary.md#host-storage), storage [liveness](../../../../glossary.md#term-liveness)/content, or input
layout. Preserve input [DataType](../../../../glossary.md#data-type--datatype) and requiresGrad exactly. Do not return the input or canonicalize
nested/equivalent expressions.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorWindowExpressions.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorWindowExpressionTest.java`

Documentation/[planning](../../../../glossary.md#term-planning):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Authorized [Javadoc](../../../../glossary.md#term-javadoc)-only temporal corrections:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/WindowTransformKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/UnfoldAxisAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/FoldAxisAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/Window2dAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/Fold2dAttrs.java`

Review without modification unless inconsistency requires stopping: Training [API](../../../../glossary.md#term-application-programming-interface-api), capabilities,
all completed [operation](../../../../glossary.md#operation)/[Shape](../../../../glossary.md#shape)/[descriptor](../../../../glossary.md#term-descriptor)/factory/[provenance](../../../../glossary.md#provenance) contracts and tests except the five
explicit Javadoc-only corrections above, focused
architecture/ADRs/tests, [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance), integration tests, [Gradle](../../../../glossary.md#term-gradle)/dependencies, other
[modules](../../../../glossary.md#term-module), and task 0023.

## Maximum [scope](../../../../glossary.md#term-scope)

Exactly two implementation production files, two test files, six documentation/[planning](../../../../glossary.md#term-planning) files,
and the five authorized [Javadoc](../../../../glossary.md#term-javadoc)-only semantic files: fifteen paths. The user explicitly expanded
the original ten-path limit only to correct stale task-0017N temporal wording in those five
completed semantic contracts. If any declaration, logic, import, semantic test, capabilities,
[dependency](../../../../glossary.md#term-dependency), build/architecture change, sixteenth path, or task-0018 specification is required,
stop and report.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Add complete Javadoc for all four public Tensor methods and the helper type, constructor, and all
  fourteen methods.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- Define window, window position, target rank, scatter-add, overlap, NCHW, im2col, col2im,
  symmetric padding, dilation, stride, effective kernel, floor mode, and ceil mode for newcomers.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [rank glossary](../../../../glossary.md#term-rank); [NCHW glossary](../../../../glossary.md#term-nchw); [im2col glossary](../../../../glossary.md#term-im2col); [col2im glossary](../../../../glossary.md#term-col2im); [padding glossary](../../../../glossary.md#padding); [dilation glossary](../../../../glossary.md#term-dilation); [stride glossary](../../../../glossary.md#term-stride); [kernel glossary](../../../../glossary.md#kernel))</span>
- Document every parameter, units in logical elements, normalized/raw axis behavior, accepted
  DataTypes, dynamic/static rules, result Shape, unresolved layout, provenance, fresh identity,
  return value, and exact failure/overflow conditions.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [elements glossary](../../../../glossary.md#term-element); [axis glossary](../../../../glossary.md#term-axis); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance); [identity glossary](../../../../glossary.md#term-identity-meanings); [overflow glossary](../../../../glossary.md#term-overflow))</span>
- Include concrete numeric [Shape](../../../../glossary.md#shape) examples for all four methods and explain each result [dimension](../../../../glossary.md#dimension).
- Include a concrete foldAxis overlap calculation, for example windows `[1,2,3]`, `[2,3,4]`,
  `[3,4,5]` with step one producing accumulated `[1,4,9,8,5]`.
- Include the checked two-dimensional formulas and explain why ceil division avoids the common
  overflow-prone `numerator + stride - 1` expression.
- Explain that public foldAxis and later compiler-generated FOLD_AXIS share one semantic kind but
  this task adds no gradient rule or compiler behavior.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Explain that all four expressions are metadata only, leave layout unresolved, attach no storage,
  and do not execute values.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>
- Review Javadocs for TensorDescriptor, Shape/Dimension, all five task-0017M contracts, Operation,
  TensorProvenance, and TensorFactory; record why unchanged wording remains accurate or stop on an
  out-of-scope discrepancy.<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation); [scope glossary](../../../../glossary.md#term-scope))</span>

## Acceptance criteria

- Tensor adds exactly the four specified public methods with exact signatures and one delegation
  each; its total declared public method inventory is updated from 92 to 96.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [signatures glossary](../../../../glossary.md#term-signature))</span>
- TensorWindowExpressions is final, package-private, field-free, has one private constructor and
  exactly fourteen declared methods without synthetic additions.
- Every [validation](../../../../glossary.md#term-validation) follows the specified order, exception type, and exact message.
- General-[axis](../../../../glossary.md#term-axis) unfold accepts all six DataTypes and derives exact [Shape](../../../../glossary.md#shape)/[provenance](../../../../glossary.md#provenance) without values.
- foldAxis accepts five numeric DataTypes, rejects BOOL, normalizes against target rank, handles the
  zero-output/zero-window case, validates exact count geometry, and derives exact Shape/provenance.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [target glossary](../../../../glossary.md#term-target); [rank glossary](../../../../glossary.md#term-rank))</span>
- unfold2d/fold2d accept exactly floating input, validate static non-batch geometry with checked
  arithmetic, preserve a dynamic batch when structurally compatible, and derive/retain exact Shapes.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- Every result retains exact DataType/requiresGrad, has unresolved layout, no label/storage, exact
  one-input provenance, exact semantic kind/attrs, and a fresh TensorId.<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [layout glossary](../../../../glossary.md#layout); [TensorId glossary](../../../../glossary.md#tensorid))</span>
- Focused tests cover reflection/API/helper shape, all DataTypes, negative axes, static/dynamic
  dimensions, zero dimensions, floor/ceil geometry, padding/stride/dilation, overflow, mismatch
  messages, exact reference retention, repeated freshness, and attached-storage non-interference.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [dimensions glossary](../../../../glossary.md#dimension); [padding glossary](../../../../glossary.md#padding); [stride glossary](../../../../glossary.md#term-stride); [dilation glossary](../../../../glossary.md#term-dilation); [overflow glossary](../../../../glossary.md#term-overflow))</span>
- No completed semantic/foundational Java declaration, logic, import, or test changes; the five
  authorized semantic files contain Javadoc-only current-versus-planned corrections.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Tensor API moves all four methods from planned to current and includes newcomer-readable worked
  examples; Compile API records current expressions without claiming capture/execution; glossary
  covers new terms and public-versus-compiler FOLD_AXIS distinction.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [capture glossary](../../../../glossary.md#term-capture-meanings); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Training API, capabilities, architecture docs/tests, conformance/integration, Gradle, other
  modules, and task 0023 receive reasoned no-change conclusions.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>
- Independent clean-context documentation pass finalizes Javadocs/docs/planning and reruns all
  validation in the same overall change.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Final diff contains exactly fifteen paths, `git diff --check` passes, 0017N is Complete, 0018 remains
  Draft without a detailed specification, and no task-0018 specification exists.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run):

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
  overload/alias;<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- exact helper modifiers, zero fields, one private constructor, fourteen declared methods, no
  synthetic/lambda-generated methods, and no additional public/package-private surface;
- exact validation order/messages and checked arithmetic, including Integer.MIN_VALUE axes,
  Long.MAX_VALUE geometry, ceil rounding, and zero foldAxis output;<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis); [rounding glossary](../../../../glossary.md#term-rounding))</span>
- exact result [descriptor](../../../../glossary.md#term-descriptor)/[provenance](../../../../glossary.md#provenance)/reference/[identity](../../../../glossary.md#term-identity-meanings) behavior and no storage/[layout](../../../../glossary.md#layout) inspection;
- exact imports and no graph/compiler/planning/prepare/runtime/backend/ONNX/service/registry/
  reflection-discovery/dependency/build coupling;<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [ONNX glossary](../../../../glossary.md#term-onnx); [registry glossary](../../../../glossary.md#term-registry); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- generated [Javadoc](../../../../glossary.md#term-javadoc) contains every required type/method/[parameter](../../../../glossary.md#term-parameter)/result/failure/formula/example;
- every Markdown link/anchor resolves, fences balance, files end with newline, and changed/untracked
  files have no trailing whitespace;
- final inventory is exactly fifteen paths, package placement matches this plan, 0017N is Complete,
  0018 is Draft, and no `0018-*.md` task specification exists.

The documentation agent must rerun focused/[model](../../../../glossary.md#term-model)/root tests, model Javadoc, javap/reflection/
[bytecode](../../../../glossary.md#term-bytecode)/import/source/generated-doc checks, Markdown/link/format checks, exact inventory/status
checks, and `git diff --check` against the final synchronized tree. Record commands, results, test
counts, agent identity, and reasoned no-change conclusions before completion.

## Dependencies

- Task 0001: [Data type](../../../../glossary.md#data-type--datatype) [model](../../../../glossary.md#term-model)
- Task 0002: [Shape](../../../../glossary.md#shape) and [dimension](../../../../glossary.md#dimension) model
- Task 0003: [Layout](../../../../glossary.md#layout) [descriptor](../../../../glossary.md#term-descriptor) model
- Task 0007: [Tensor descriptor](../../../../glossary.md#tensor-descriptor) model
- Task 0011: Public [Tensor](../../../../glossary.md#tensor) skeleton
- Task 0012: [Tensor factory](../../../../glossary.md#tensor-factory)
- Task 0013: Tensor [provenance](../../../../glossary.md#provenance) skeleton
- Task 0017M: [Unfold and fold](../../../../glossary.md#window-transform) semantics

## Follow-up tasks

- Task 0018: Indexing and scatter operations — next Draft frontier; create its detailed
  specification only after 0017N is Complete.
- Task 0023 later owns compiler-generated FOLD_AXIS use for autograd and reuses this public
  expression's completed semantic kind; it does not block 0017N.<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd))</span>
- [Compiler](../../../../glossary.md#term-compiler), [planning](../../../../glossary.md#term-planning), [prepare](../../../../glossary.md#prepare), [backend](../../../../glossary.md#backend), [ONNX](../../../../glossary.md#term-onnx), [gradient](../../../../glossary.md#term-gradient), and execution work remain later tasks.

## Architecture impact

Expected impact: None.

This task adds [model](../../../../glossary.md#term-model)-owned public expression [metadata](../../../../glossary.md#term-metadata) only. It preserves [Tensor](../../../../glossary.md#tensor)/[IR](../../../../glossary.md#term-intermediate-representation-ir) separation,
[module](../../../../glossary.md#term-module) [dependency direction](../../../../glossary.md#java-platform-and-build-tools), [compile](../../../../glossary.md#compile)/[prepare](../../../../glossary.md#prepare)/[run](../../../../glossary.md#term-run) separation, [backend-owned lowering](../../../../glossary.md#backend-owned-lowering), [runtime](../../../../glossary.md#term-runtime)
boundaries, and the authoritative [architecture contract](../../../../glossary.md#architecture-contract). Stop if implementation requires changing
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
  materialization without arithmetic on values. <span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [model glossary](../../../../glossary.md#term-model); [materialization glossary](../../../../glossary.md#materialization))</span>`foldAxis` accepts floating and integral input but
  rejects BOOL because its semantic result is overlap-summing scatter-add. NCHW unfold/fold remain
  floating-only.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [NCHW glossary](../../../../glossary.md#term-nchw))</span>
- `foldAxis` normalizes raw syntax against [target](../../../../glossary.md#term-target) [rank](../../../../glossary.md#term-rank) `inputRank - 1`, after excluding the final
  input window dimension. It therefore cannot address that final dimension.<span data-glossary-navigation="true"> ([dimension glossary](../../../../glossary.md#dimension))</span>
- Static input-dependent dimensions are required wherever the current Shape model cannot express a
  symbolic window count or compatibility constraint. Exact unaffected and batch Dimension
  references remain preserved where local proof is possible.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [batch Dimension glossary](../../../../glossary.md#batch-dimension))</span>
- Floor and ceil window counts use checked quotient/remainder arithmetic. Ceil mode deliberately
  avoids `numerator + stride - 1`, which can [overflow](../../../../glossary.md#term-overflow) even when the quotient is representable.
- Every result remains storage-free and layout-unresolved. No locally valid request is simplified
  to the input, and repeated identical calls receive fresh Tensor identities.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [Tensor glossary](../../../../glossary.md#tensor))</span>
- The same FOLD_AXIS semantic [identity](../../../../glossary.md#term-identity-meanings) serves current public `foldAxis` construction and future
  compiler-generated unfold adjoints. Task 0023 still owns the latter; this task adds no gradient
  or compiler behavior.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Explicit user authorization expanded the original ten-path limit to exactly fifteen paths only
  so the five completed task-0017M semantic Java files could receive Javadoc-only temporal
  corrections. Their declarations, imports, logic, and tests remain unchanged.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## Known limitations

- The expressions calculate metadata only. They do not materialize windows, sample padding,
  scatter-add values, zero-fill uncovered output, allocate or attach storage, or execute kernels.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [padding glossary](../../../../glossary.md#padding); [kernels glossary](../../../../glossary.md#kernel))</span>
- General-axis transformed dimensions and NCHW channel/spatial geometry must be statically known;
  this task creates no symbolic constraints or runtime Shape binding.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [dimensions glossary](../../../../glossary.md#dimension); [NCHW glossary](../../../../glossary.md#term-nchw); [channel glossary](../../../../glossary.md#term-channel); [spatial glossary](../../../../glossary.md#term-spatial-axes); [runtime glossary](../../../../glossary.md#term-runtime); [Shape glossary](../../../../glossary.md#shape); [binding glossary](../../../../glossary.md#term-binding))</span>
- Two-dimensional geometry is NCHW, symmetric-padding, single-window-configuration only. There are
  no defaults, builders, asymmetric padding modes, aliases, or integral NCHW forms.<span data-glossary-navigation="true"> ([aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Gradient rules, compiler capture/canonicalization, compiler-generated FOLD_AXIS, planning,
  materialization, backend/ONNX lowering, and execution remain future owning-layer work.<span data-glossary-navigation="true"> ([Gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [planning glossary](../../../../glossary.md#term-planning); [materialization glossary](../../../../glossary.md#materialization); [backend glossary](../../../../glossary.md#backend); [ONNX glossary](../../../../glossary.md#term-onnx); [lowering glossary](../../../../glossary.md#lowering))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- The implementation pass and independent clean-context documentation pass both completed in the
  same uncommitted tree. The final documentation agent context was
  `/root/implement_model_0017n/docs_0017n_resumed`; it read the architecture contract, focused
  architecture documents, documentation rules and General/API-Javadoc/Planning/Example profiles,
  planning guide/roadmap, model plans and capability baseline, prerequisite/task specifications,
  API/glossary documents, affected source/tests, related contracts, and Java 26 Gradle setup.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Planning glossary](../../../../glossary.md#term-planning); [model glossary](../../../../glossary.md#term-model); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- `./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorWindowExpressionTest`
  passed all 16 focused tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` passed all 629 tests across 74 suites with zero failures, errors,
  or skips. `./gradlew :modules:model:javadoc` succeeded without warnings, and generated public
  Tensor plus all five semantic-contract pages contain the current signatures, parameters,
  returns, failures, examples, formulas, and current-versus-planned boundary. The package-private
  helper is intentionally outside the public Javadoc task output; all fourteen source Javadocs
  were reviewed directly.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [signatures glossary](../../../../glossary.md#term-signature); [parameters glossary](../../../../glossary.md#term-parameter))</span>
- `./gradlew test` completed the 36-task root lifecycle successfully, including the model test
  result and all currently populated modules; no backend, conformance, or integration production
  behavior was added.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [modules glossary](../../../../glossary.md#term-module); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- `javap -p -c -s` plus a compiled reflection helper confirmed 96 declared public Tensor methods;
  the four exact new signatures each delegate once. `TensorWindowExpressions` is package-private
  final and field-free, with one private constructor, exactly fourteen declared methods, and zero
  synthetic methods. Bytecode contains one final <span data-glossary-navigation="true"> ([Bytecode glossary](../../../../glossary.md#term-bytecode))</span>`TensorFactory.createDerived` call in the shared
  construction path and checked `Math` arithmetic without lambda-generated methods.
- A separately compiled and executed copy of `TensorWindowExpressionExample` printed
  `Shape[2, 3, 3, 3]`, `Shape[5]`, `Shape[1, 4, 4]`, `Shape[1, 1, 3, 3]`, `UNFOLD_AXIS`, `FOLD2D`,
  `true`, and `true`, exactly as documented.
- Source/import/manual inspection confirmed exact package placement, no legacy overload/options,
  no helper fields, no graph/compiler/planning/prepare/runtime/backend/ONNX/service/registry or
  reflection-discovery coupling, unresolved result layouts, one-input provenance, and no storage
  inspection. The five semantic Java diffs contain Javadoc changes only.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [ONNX glossary](../../../../glossary.md#term-onnx); [registry glossary](../../../../glossary.md#term-registry); [layouts glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance))</span>
- The targeted Markdown checker resolved 370 local links, including 108 anchors, across the six
  changed documentation/planning files with zero errors. All 242 code-fence markers are balanced;
  every authorized path ends with a newline and has no trailing whitespace.
- `git diff --check` passed. The combined changed/untracked inventory is exactly the fifteen paths
  authorized above. Task 0017N, the model master plan, and the roadmap agree on Complete; task
  0018 remains Draft and no `0018*.md` task specification exists.

## Implementation notes

- [`Tensor`](../../../../glossary.md#tensor) exposes only the four specified instance methods and delegates all validation and
  construction to the field-free helper. The helper centralizes source/target-rank normalization,
  static-dimension extraction, checked general-axis and two-dimensional Shape arithmetic,
  canonical-column validation, and final descriptor/provenance construction.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [target glossary](../../../../glossary.md#term-target); [rank glossary](../../../../glossary.md#term-rank); [normalization glossary](../../../../glossary.md#term-normalization); [dimension glossary](../../../../glossary.md#dimension); [axis glossary](../../../../glossary.md#term-axis); [Shape glossary](../../../../glossary.md#shape); [canonical glossary](../../../../glossary.md#term-canonical-form); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- `TensorWindowExpressionTest` covers API/helper reflection shape, every DataType boundary,
  positive and negative axes, floor/ceil/padding/stride/dilation geometry, static/dynamic and zero
  dimensions, overflow, exact failures, reference retention, identity freshness, and attached-
  storage non-interference. <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [DataType glossary](../../../../glossary.md#data-type--datatype); [padding glossary](../../../../glossary.md#padding); [stride glossary](../../../../glossary.md#term-stride); [dilation glossary](../../../../glossary.md#term-dilation); [overflow glossary](../../../../glossary.md#term-overflow); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>`TensorTest` changes only the exact public method inventory.
- The independent documentation pass finalized public/helper and semantic Javadocs, Tensor API,
  Compile API, glossary, task evidence, model master plan, and roadmap. It applied the General,
  API/Javadoc, Planning, and Example documentation profiles.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Planning glossary](../../../../glossary.md#term-planning))</span>
- Training API remains accurate unchanged because this task adds no gradient, optimizer, or
  training behavior. Capabilities remains unchanged by explicit scope and because it is the broad
  capability baseline rather than task-status evidence. TensorDescriptor, Shape/Dimension,
  Operation, TensorProvenance, TensorFactory, related expression helpers, and completed semantic
  declarations remain accurate because the new methods compose their existing contracts without
  changing them.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [optimizer glossary](../../../../glossary.md#term-optimizer); [scope glossary](../../../../glossary.md#term-scope); [Operation glossary](../../../../glossary.md#operation))</span>
- Architecture documents/ADRs/tests remain unchanged because no module boundary, dependency, or
  lifecycle decision changed. Backend conformance and integration tests remain unchanged because
  there is no backend or end-to-end execution behavior. Java 26 Gradle configuration,
  dependencies, other modules, and task 0023 also remain unchanged.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [lifecycle glossary](../../../../glossary.md#lifecycle); [Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle))</span>

## Completion summary

- Completed changes: added four public storage-free unfold/fold Tensor expressions, the exact
  field-free fourteen-method helper, focused tests, public API inventory updates, complete
  Javadocs/API/glossary documentation, five authorized temporal Javadoc corrections, and
  synchronized planning evidence.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: exactly the two implementation production files, two test files, six
  documentation/planning files, and five Javadoc-only semantic files listed under Affected files.
- Tests and validation: focused 16/16, model 629/629 across 74 suites, model Javadoc, root tests,
  javap/reflection/bytecode/import/source/generated-doc checks, executable example, 370 local
  link checks including 108 anchors, fence/newline/whitespace checks, exact fifteen-path inventory,
  synchronized status, no-0018-spec check, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [model glossary](../../../../glossary.md#term-model); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>`git diff --check` all passed.
- Unresolved issues: none for task 0017N.
- Required follow-up: none for task 0017N. Task 0018 remains the next Draft frontier without a
  specification; task 0023 later owns compiler-generated FOLD_AXIS and autograd behavior.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [autograd glossary](../../../../glossary.md#autograd))</span>

Status: Complete
