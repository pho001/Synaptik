# Task 0025H: [NCDHW](../../../../glossary.md#conv3d--ncdhw) [Conv3d](../../../../glossary.md#conv3d--ncdhw) Semantics and [Tensor](../../../../glossary.md#tensor) Expressions

**Editorial reading key (current context, not a status update).**

This record’s Complete status concerns Model construction, not downstream execution; its Draft
Compiler boundaries and inventory counts are historical. NCDHW means batch, channel, depth, height,
width. The caller invokes the input Tensor’s conv3d method with weight, optional bias, and immutable
geometry/group attributes; the Model helper validates metadata and returns a storage-free
expression. Provenance retains exact input and attribute references, not copied numerical buffers.
For example, two groups with four input channels give each output channel access to two input
channels; cross-correlation does not reverse the kernel.

For current geometry and deferred obligations, follow [Grouped NCDHW Conv3d
expressions](../../../../reference/tensor/convolution.md#grouped-ncdhw-conv3d-expressions):
unresolved dimensions are not proven valid bindings. For current compiler responsibilities,
qualified execution descriptions, and arithmetic boundaries, follow [Numerical and provenance
policy](../../../../reference/tensor/convolution.md#numerical-and-provenance-policy). Model
semantics defines meaning; execution providers realize supported routes rather than redefine it.
Semantic responsibility is not buffer ownership: this expression neither executes nor owns result
storage. Conceptual padded positive zero still multiplies infinite weights; permitted reassociation
does not promise bitwise-identical results.

## Status

Complete

## Goal

Add one first-class grouped three-dimensional cross-correlation semantic [operation](../../../../glossary.md#operation) and the natural
public [Tensor](../../../../glossary.md#tensor) expressions for channels-first [NCDHW](../../../../glossary.md#conv3d--ncdhw) tensors. The [Model](../../../../glossary.md#term-model) capability consists of one
[rank](../../../../glossary.md#term-rank)-specific [immutable](../../../../glossary.md#term-immutable) attribute value, one [`CONV3D`](../../../../glossary.md#conv3d--ncdhw) operation kind, and exactly two receiver
methods for unbiased and biased construction. It defines [metadata](../../../../glossary.md#term-metadata), [Shape](../../../../glossary.md#shape), floating numerical
meaning, and [provenance](../../../../glossary.md#provenance) only; later [Compiler](../../../../glossary.md#term-compiler), [backend](../../../../glossary.md#backend), and execution work remain separate.

## [Scope](../../../../glossary.md#term-scope)

- Add [immutable](../../../../glossary.md#term-immutable) public
  `Conv3dAttrs(strideDepth, strideHeight, strideWidth, paddingDepth, paddingHeight, paddingWidth,
  dilationDepth, dilationHeight, dilationWidth, groups)` with exact declaration-order validation
  and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`defaults()` equal to `(1, 1, 1, 0, 0, 0, 1, 1, 1, 1)`.
- Add public `Conv3dKind.CONV3D` with exactly one immutable [signature](../../../../glossary.md#term-signature) accepting `Conv3dAttrs`, two
  or three ordered inputs, and exactly one output.
- Add exactly these public receiver methods to [`Tensor`](../../../../glossary.md#tensor):

  ```java
  public Tensor conv3d(Tensor weight, Conv3dAttrs attrs)

  public Tensor conv3d(Tensor weight, Tensor bias, Conv3dAttrs attrs)
  ```

- Add one package-private, final, field-free `TensorConv3dExpressions` construction [owner](../../../../glossary.md#term-owner).
- Accept floating input `[N, C_in, D, H, W]`, [weight](../../../../glossary.md#term-weight)
  `[C_out, C_in/groups, K_d, K_h, K_w]`, and optional [bias](../../../../glossary.md#term-bias) `[C_out]`; derive
  `[N, C_out, D_out, H_out, W_out]`.
- Validate nulls, floating roles and promotion, exact ranks, positive static kernel extents,
  locally provable group/channel/bias relations, and checked static or symbolic spatial geometry
  before creating an operation, producer, descriptor result, or Tensor identity.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [channel glossary](../../../../glossary.md#term-channel); [spatial glossary](../../../../glossary.md#term-spatial-axes); [operation glossary](../../../../glossary.md#operation); [producer glossary](../../../../glossary.md#term-producer); [descriptor glossary](../../../../glossary.md#term-descriptor); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Retain the exact operation attributes, ordered Tensor inputs, derived output descriptor, one
  canonical output wrapper, and output-index-zero provenance.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [provenance glossary](../../../../glossary.md#provenance))</span>
- Define grouped cross-correlation, promotion, accumulation, conceptual padding, special-value,
  reassociation, and determinism meaning as the rank-three-spatial extension of current Conv2d.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [reassociation glossary](../../../../glossary.md#term-reassociation); [rank glossary](../../../../glossary.md#term-rank); [Conv2d glossary](../../../../glossary.md#convolution--conv2d))</span>
- Test the exact kind/attribute/public surface, Shapes, local/deferred relations, validation order
  and messages, effects, freshness, canonical provenance, and checked arithmetic.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape))</span>
- Finalize affected Javadocs, Tensor/Compile APIs, glossary, Model capabilities, and planning in a
  separate clean documentation-focused context before completion.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [Model glossary](../../../../glossary.md#term-model); [planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- `ConvNd`, a dynamic-rank convolution, public or private geometry arrays, options arrays, nullable
  bias, default-attribute overloads, aliases, static namespaces, or additional public entry points.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [bias glossary](../../../../glossary.md#term-bias); [aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Conv1d or Conv2d changes, decomposition into existing operations, kernel unrolling, generated
  code, eager evaluation, or any claim that current operations can express Conv3d with bounded
  graph size independent of spatial extents.<span data-glossary-navigation="true"> ([Conv2d glossary](../../../../glossary.md#convolution--conv2d); [decomposition glossary](../../../../glossary.md#term-decomposition); [kernel glossary](../../../../glossary.md#kernel); [unrolling glossary](../../../../glossary.md#term-unrolling); [eager evaluation glossary](../../../../glossary.md#term-eager-evaluation); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [graph glossary](../../../../glossary.md#graph); [spatial glossary](../../../../glossary.md#term-spatial-axes))</span>
- Asymmetric intrinsic padding, padding modes, transposed, deformable, causal-specialized,
  depthwise-specific, separable, quantized, sparse, or channels-last convolution.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [channels-last glossary](../../../../glossary.md#spatial-axes-channels-and-window-sampling))</span>
- Compiler capture, inference, final validation, constraints, graph transformations, gradients,
  gradient policy, or derivative formulas. Compiler 0006B remains the next separate Draft forward
  consumer and explicit fail-closed backward boundary; Compiler 0006C remains the separate Draft
  adjoint-expressibility and gradient-closure task.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation); [gradients glossary](../../../../glossary.md#term-gradient); [derivative glossary](../../../../glossary.md#term-derivative); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [adjoint glossary](../../../../glossary.md#term-adjoint))</span>
- Planning, Prepare, Runtime, Engine, backend, algorithm selection, lowering, fusion, allocation,
  capability advertisement, conformance, integration, or execution implementation.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [fusion glossary](../../../../glossary.md#term-fusion); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- A neural-network layer, [module](../../../../glossary.md#term-module), [parameters](../../../../glossary.md#term-parameter), initializer, factory recipe, state, or training [API](../../../../glossary.md#term-application-programming-interface-api).
- Changes to TensorFactory, TensorProducer, TensorProvenance, Operation, OperationSignature,
  Shape/Dimension arithmetic, DataType promotion, PAD/window semantics, or existing convolution
  kinds and helpers.<span data-glossary-navigation="true"> ([TensorProducer glossary](../../../../glossary.md#producer--tensorproducer); [Operation glossary](../../../../glossary.md#operation); [Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [DataType glossary](../../../../glossary.md#data-type--datatype); [PAD glossary](../../../../glossary.md#padding))</span>
- Detailed specifications for Model 0026, Compiler 0006B/0006C, CPU 0008/0008A, Engine 0004, or
  NN 0025.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [CPU glossary](../../../../glossary.md#term-cpu); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Legacy source, package structure, [dependency direction](../../../../glossary.md#java-platform-and-build-tools), or implementation reuse.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Roadmap](../../../roadmap.md)
- [Model capabilities](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model master plan](../master-plan.md)
- [Operation signature hardening](0018k-operation-signature-and-construction-hardening.md)<span data-glossary-navigation="true"> ([Operation signature glossary](../../../../glossary.md#operation-signature))</span>
- [Shared multi-output Tensor provenance](0018l-shared-multi-output-tensor-provenance.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [provenance glossary](../../../../glossary.md#provenance))</span>
- [Symbolic extent expressions](0018m-symbolic-extent-expressions.md)<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent))</span>
- [Dynamic-extent adoption](0018m1-dynamic-extent-adoption.md)
- [Typed scalar value contract](0018n-typed-scalar-value-contract.md)<span data-glossary-navigation="true"> ([Typed scalar value glossary](../../../../glossary.md#typed-scalar-value--scalarvalue))</span>
- [Multi-axis and statistical reductions](0018v-multi-axis-and-statistical-reductions.md)<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>
- [Matmul semantics and Tensor expression](0019-matmul-semantics-and-tensor-expression.md)
- [NCHW Conv2d semantics and Tensor expressions](0020-nchw-conv2d-semantics-and-tensor-expressions.md)<span data-glossary-navigation="true"> ([NCHW glossary](../../../../glossary.md#term-nchw); [Conv2d glossary](../../../../glossary.md#convolution--conv2d))</span>
- [Canonical TensorProducer outputs](0025-canonical-tensor-producer-outputs.md)<span data-glossary-navigation="true"> ([Canonical glossary](../../../../glossary.md#term-canonical-form); [TensorProducer glossary](../../../../glossary.md#producer--tensorproducer))</span>
- [NCW Conv1d composition](0025g-ncw-conv1d-composition.md)<span data-glossary-navigation="true"> ([NCW glossary](../../../../glossary.md#term-ncw); [Conv1d composition glossary](../../../../glossary.md#conv1d-composition))</span>
- [Compiler master plan](../../compiler/master-plan.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>

## Architecture constraints

- Work stays in Model and directly affected documentation/planning. Model owns immutable Tensor,
  operation, descriptor, Shape, and provenance semantics; it does not capture graphs, infer a
  compiled graph, construct gradients, choose execution, or depend on downstream modules.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [planning glossary](../../../../glossary.md#term-planning); [descriptor glossary](../../../../glossary.md#term-descriptor); [Shape glossary](../../../../glossary.md#shape); [provenance glossary](../../../../glossary.md#provenance); [capture glossary](../../../../glossary.md#term-capture-meanings); [graphs glossary](../../../../glossary.md#graph); [compiled graph glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel); [gradients glossary](../../../../glossary.md#term-gradient); [modules glossary](../../../../glossary.md#term-module))</span>
- [`Tensor`](../../../../glossary.md#tensor) remains the public fluent receiver facade. The helper depends only on existing Model
  datatype, operation, Shape, descriptor, factory, and provenance contracts.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade))</span>
- [`CONV3D`](../../../../glossary.md#conv3d--ncdhw) is one ordinary flat one-output semantic operation. It has no region, subgraph,
  decomposition, hidden auxiliary, backend payload, execution state, or generated implementation.<span data-glossary-navigation="true"> ([decomposition glossary](../../../../glossary.md#term-decomposition); [backend glossary](../../../../glossary.md#backend))</span>
- Every successful call creates one [immutable](../../../../glossary.md#term-immutable) [`Operation`](../../../../glossary.md#operation), one [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), one descriptor,
  and the producer-owned canonical output wrapper through the existing <span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>`TensorFactory` seam.
- Ordered inputs are exact Tensor references. Attributes are retained by exact reference.
  Provenance remains immutable and output-index zero.
- `Conv3dAttrs` is rank-specific. A common helper may remain private only if it reflects an
  already-proved local implementation responsibility; this task adds no shared public ConvNd
  abstraction and changes no completed Conv2d contract.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [Conv2d glossary](../../../../glossary.md#convolution--conv2d))</span>
- Compiler owns independent forward inference, proof of deferred relations, and gradients.
  Prepare/backend work owns algorithms, lowering, allocation, and materialization. Runtime
  receives only prepared work.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [inference glossary](../../../../glossary.md#term-inference); [Prepare glossary](../../../../glossary.md#prepare); [lowering glossary](../../../../glossary.md#lowering); [materialization glossary](../../../../glossary.md#materialization); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- No architecture, module-boundary, dependency, lifecycle, build, or architecture-test rule
  changes.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.datatype`
- `io.github.pho001.synaptik.model.shape`
- `io.github.pho001.synaptik.model.operation`
- `io.github.pho001.synaptik.model.operation.convolution`
- `io.github.pho001.synaptik.model.tensor`

Packages changed:

- add the [rank](../../../../glossary.md#term-rank)-three-[spatial](../../../../glossary.md#term-spatial-axes) kind and [immutable](../../../../glossary.md#term-immutable) attributes to
  `io.github.pho001.synaptik.model.operation.convolution`;
- add the package-private expression [owner](../../../../glossary.md#term-owner) and two receiver overloads to
  `io.github.pho001.synaptik.model.tensor`;
- add no package and widen no existing helper or factory seam.

Type placement:

- `...operation.convolution.Conv3dAttrs` owns only validated depth/height/width stride, symmetric
  padding, dilation, and groups.<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [padding glossary](../../../../glossary.md#padding); [dilation glossary](../../../../glossary.md#term-dilation))</span>
- `...operation.convolution.Conv3dKind` owns only the exact [operation](../../../../glossary.md#operation) name and [signature](../../../../glossary.md#term-signature).
- `...tensor.TensorConv3dExpressions` owns local validation, Shape derivation, descriptor and
  operation construction, and delegation to the existing derived-output factory seam.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Shape glossary](../../../../glossary.md#shape); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- `...tensor.Tensor` exposes the two natural single-output receiver entries parallel to [Conv2d](../../../../glossary.md#convolution--conv2d).

## Exact semantic and expression contract

### [Signature](../../../../glossary.md#term-signature) and input order

`Conv3dKind.CONV3D.signatures()` is exactly:

```java
List.of(OperationSignature.inputRange(Conv3dAttrs.class, 2, 3, 1))
```

The unbiased [occurrence](../../../../glossary.md#term-occurrence) has ordered inputs `[input, weight]`. The biased occurrence has ordered
inputs `[input, weight, bias]`. Both have one output at index zero. No other attribute type,
[arity](../../../../glossary.md#term-arity), output count, or signature is accepted.

### [Shapes](../../../../glossary.md#shape) and geometry

```text
input:   [N, C_in, D, H, W]
weight:  [C_out, C_in/groups, K_d, K_h, K_w]
bias:    [C_out]                                      optional
result:  [N, C_out, D_out, H_out, W_out]

effectiveKernel_x = dilation_x * (K_x - 1) + 1
numerator_x       = X + 2 * padding_x - effectiveKernel_x
X_out             = floor(numerator_x / stride_x) + 1

x in depth, height, width order
```

`K_d`, `K_h`, and `K_w` must each be statically known and positive. Literal arithmetic uses
checked signed `long` operations. For a static input [extent](../../../../glossary.md#term-extent), a negative numerator fails locally;
zero produces one output position. For an unresolved input extent, compute the checked [constant](../../../../glossary.md#term-constant)
offset `2 * padding_x - effectiveKernel_x`, then use the existing [canonical](../../../../glossary.md#term-canonical-form)
`DimensionExpressions.addConstant`, `floorDivide`, and `addConstant(..., 1)` sequence. The future
concrete [binding](../../../../glossary.md#term-binding) retains the obligation that each numerator is non-negative.

The result reuses the exact input [batch Dimension](../../../../glossary.md#batch-dimension) reference and [weight](../../../../glossary.md#term-weight) output-[channel](../../../../glossary.md#term-channel) [Dimension](../../../../glossary.md#dimension)
reference. Static input and output channels must each be divisible by `groups`. When input
channels and weight channels per group are both static, checked
`weightChannelsPerGroup * groups` must equal input channels. When present [bias](../../../../glossary.md#term-bias) length and output
channels are both static, they must be equal. Any relation containing an unresolved Dimension is
deferred and remains observable through exact inputs, [descriptors](../../../../glossary.md#term-descriptor), and attributes; [Model](../../../../glossary.md#term-model) creates
no constraint object and guesses no binding.

### Attributes

`Conv3dAttrs` record components, in declaration and accessor order, are:

```text
strideDepth
strideHeight
strideWidth
paddingDepth
paddingHeight
paddingWidth
dilationDepth
dilationHeight
dilationWidth
groups
```

The [canonical](../../../../glossary.md#term-canonical-form) constructor validates in that same order with exact messages:

```text
strideDepth must be positive: <value>
strideHeight must be positive: <value>
strideWidth must be positive: <value>
paddingDepth must be non-negative: <value>
paddingHeight must be non-negative: <value>
paddingWidth must be non-negative: <value>
dilationDepth must be positive: <value>
dilationHeight must be positive: <value>
dilationWidth must be positive: <value>
groups must be positive: <value>
```

All components are primitive `long`; the record implements only [`OperationAttrs`](../../../../glossary.md#operationattrs), has no nested
type, performs no geometry calculation, and exposes no arrays or [mutable](../../../../glossary.md#term-mutable) state. `defaults()`
returns a fresh or value-equivalent `(1, 1, 1, 0, 0, 0, 1, 1, 1, 1)` instance.

### Values and numerical policy

[Conv3d](../../../../glossary.md#conv3d--ncdhw) means grouped [NCDHW](../../../../glossary.md#conv3d--ncdhw) cross-correlation and does not reverse the stored [kernel](../../../../glossary.md#kernel). Output
[channel](../../../../glossary.md#term-channel) `o` selects its [contiguous](../../../../glossary.md#term-contiguous) input-channel group. For each output [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), products are
conceptually visited in increasing logical input-channel, kernel-depth, kernel-height, then
kernel-width order. The optional [bias](../../../../glossary.md#term-bias) for `o` participates exactly once in the accumulation
domain before final output conversion.

Input, [weight](../../../../glossary.md#term-weight), and present bias must each be floating. Promotion processes input and weight first,
then present bias, using the current floating hierarchy. [FLOAT64](../../../../glossary.md#term-float64) output accumulates in FLOAT64;
[FLOAT32](../../../../glossary.md#term-float32) and [BFLOAT16](../../../../glossary.md#term-bfloat16) output accumulate in FLOAT32, with final conversion for BFLOAT16.

Out-of-range [spatial](../../../../glossary.md#term-spatial-axes) coordinates are conceptual positive-zero input [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) and participate in
ordinary [IEEE](../../../../glossary.md#term-ieee)-754 multiplication, including zero multiplied by [infinity](../../../../glossary.md#term-infinity). [NaN, infinity, and
signed zero](../../../../glossary.md#nan-infinity-and-signed-zero) otherwise follow ordinary multiplication and addition. An empty input-channel
[contraction](../../../../glossary.md#contraction-dimension) starts at [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) before optional bias. Empty [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) or output-channel [axes](../../../../glossary.md#term-axis) are
valid. [Reassociation and fused multiply-add](../../../../glossary.md#rounding-sites-fma-and-aggregate-floor) are permitted, so neither fixed summation order nor
bitwise cross-[backend](../../../../glossary.md#backend)-identical [rounding](../../../../glossary.md#term-rounding) is promised. These rules define represented meaning;
[Model](../../../../glossary.md#term-model) reads no [tensor](../../../../glossary.md#tensor) values and chooses no algorithm.

### Result [descriptor](../../../../glossary.md#term-descriptor) and [provenance](../../../../glossary.md#provenance)

After all local [validation](../../../../glossary.md#term-validation) succeeds, construct exactly:

1. one `TensorDescriptor` with promoted data type, exact derived NCDHW Shape, unresolved layout,
   and <span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout))</span>`requiresGrad` equal to the logical OR of input, [weight](../../../../glossary.md#term-weight), and present [bias](../../../../glossary.md#term-bias);
2. one `Operation(Conv3dKind.CONV3D, attrs)` retaining the exact `attrs` reference; and
3. one derived Tensor through the existing single-output factory seam with empty label, no host
   storage, ordered exact inputs, and output index zero.<span data-glossary-navigation="true"> ([host storage glossary](../../../../glossary.md#host-storage))</span>

The result is the exact [canonical](../../../../glossary.md#term-canonical-form) object returned by its [producer](../../../../glossary.md#term-producer)'s `output(0)`. Its provenance
returns the same [operation](../../../../glossary.md#operation), exact ordered inputs, exact output descriptor, and index zero.
Equivalent calls create [identity](../../../../glossary.md#term-identity-meanings)-distinct operations, producers, descriptors, wrappers, and IDs.
Every success consumes exactly one fresh [Tensor](../../../../glossary.md#tensor) ID.

### Local versus deferred relations

[Model](../../../../glossary.md#term-model) proves only relations whose participating [Dimensions](../../../../glossary.md#dimension) are static at construction:

- positive [kernel](../../../../glossary.md#kernel) depth, height, and width are always local because kernel extents must be static;
- group divisibility is local only for a static [channel](../../../../glossary.md#term-channel) [extent](../../../../glossary.md#term-extent);
- grouped [weight](../../../../glossary.md#term-weight)/input equality is local only when both channel extents are static;
- [bias](../../../../glossary.md#term-bias)/output equality is local only when both extents are static;
- padded-kernel fit is local only for a static input [spatial](../../../../glossary.md#term-spatial-axes) extent.

All other relations are deferred to [Compiler](../../../../glossary.md#term-compiler) [inference](../../../../glossary.md#term-inference)/final [validation](../../../../glossary.md#term-validation) and eventual [binding](../../../../glossary.md#term-binding).
Deferral is acceptance of unresolved [metadata](../../../../glossary.md#term-metadata), not proof that an arbitrary future binding is
valid. Model adds no compiler constraint, inference, binding, [gradient](../../../../glossary.md#term-gradient), or execution behavior.

### [Validation](../../../../glossary.md#term-validation) order, messages, and effects

Each overload null-checks in declaration order: `input`, [`weight`](../../../../glossary.md#term-weight), optional [`bias`](../../../../glossary.md#term-bias), then `attrs`,
using the [parameter](../../../../glossary.md#term-parameter) name as the `NullPointerException` message. The helper then validates:

1. input, weight, then present-bias floating eligibility;
2. input/weight and then present-bias floating promotion;
3. input [rank](../../../../glossary.md#term-rank) five, weight rank five, then present-bias rank one;
4. positive static [kernel](../../../../glossary.md#kernel) depth, then height, then width;
5. static input-channel divisibility, output-channel divisibility, grouped weight/input equality,
   then bias/output equality;<span data-glossary-navigation="true"> ([channel glossary](../../../../glossary.md#term-channel))</span>
6. checked output depth, then height, then width geometry;
7. [descriptor](../../../../glossary.md#term-descriptor), [operation](../../../../glossary.md#operation), [producer](../../../../glossary.md#term-producer), and [canonical](../../../../glossary.md#term-canonical-form) output creation.

Task-owned expression failures use these exact messages:

```text
conv3d input must have a floating data type, but was <dataType>
conv3d weight must have a floating data type, but was <dataType>
conv3d bias must have a floating data type, but was <dataType>
conv3d input rank must be 5: <rank>
conv3d weight rank must be 5: <rank>
conv3d bias rank must be 1: <rank>
conv3d kernel depth must be static: <dimension>
conv3d kernel depth must be positive: <dimension>
conv3d kernel height must be static: <dimension>
conv3d kernel height must be positive: <dimension>
conv3d kernel width must be static: <dimension>
conv3d kernel width must be positive: <dimension>
conv3d input channels must be divisible by groups: channels=<dimension>, groups=<groups>
conv3d output channels must be divisible by groups: channels=<dimension>, groups=<groups>
conv3d weight channels per group do not match input channels: weight=<dimension>, groups=<groups>, input=<dimension>
conv3d bias length must match output channels: bias=<dimension>, output=<dimension>
conv3d effective kernel does not fit padded depth: input=<dimension>, effectiveKernel=<value>, padding=<value>
conv3d effective kernel does not fit padded height: input=<dimension>, effectiveKernel=<value>, padding=<value>
conv3d effective kernel does not fit padded width: input=<dimension>, effectiveKernel=<value>, padding=<value>
```

Invalid attributes fail when their record is constructed and therefore consume no [Tensor](../../../../glossary.md#tensor) ID.
Every helper-local null, semantic, or arithmetic failure occurs before operation/producer creation
and consumes no ID. Successful factory allocation consumes one ID. Existing identifier-exhaustion
behavior propagates without [rollback](../../../../glossary.md#term-rollback) or reuse. No path allocates storage, mutates an input,
creates a label, or consumes an ID speculatively before local validation completes.

## Public surface

- Add exactly one public enum type, one public record type, and two public [Tensor](../../../../glossary.md#tensor) receiver methods.
- `Conv3dKind` has exactly one enum [constant](../../../../glossary.md#term-constant) and implements only [`OperationKind`](../../../../glossary.md#operationkind).
- `Conv3dAttrs` implements only [`OperationAttrs`](../../../../glossary.md#operationattrs).
- `TensorConv3dExpressions` is package-private, final, field-free, non-instantiable, and exposes no
  public member.
- Public Tensor declared-method count changes from 204 to 206.
- Add no `ConvNd`, nullable-bias overload, default-attrs overload, factory method, builder, options
  object, array geometry, output record, or auxiliary output.<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias))</span>

## Affected files

Expected production source (4):

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/convolution/Conv3dAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/convolution/Conv3dKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorConv3dExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Expected [Model](../../../../glossary.md#term-model) tests (17):

- add `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/convolution/Conv3dSemanticsTest.java`;
- add `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorConv3dExpressionTest.java`;
- update the exact [Tensor](../../../../glossary.md#tensor) public-surface locks in `TensorTest`, `TensorConv1dExpressionTest`,
  `RecurrentScanExpressionTest`,
  `TensorBatchNormInferenceExpressionTest`, `TensorBinaryArithmeticTest`,
  `TensorDenseCategoricalCrossEntropyWithLogitsExpressionTest`,
  `TensorIndexCategoricalCrossEntropyWithLogitsExpressionTest`, `TensorLayerNormExpressionTest`,
  `TensorLinearExpressionTest`, `TensorMatmulExpressionTest`, `TensorMeanSquaredErrorExpressionTest`,
  `TensorRmsNormExpressionTest`, `TensorScaledDotProductAttentionExpressionTest`,
  `TensorSlicePlacementExpressionTest`, and `TensorSumToShapeExpressionTest`, changing only the
  expected count from 204 to 206 except for direct Conv3d signature/name assertions owned by<span data-glossary-navigation="true"> ([Conv3d glossary](../../../../glossary.md#conv3d--ncdhw))</span>
  `TensorTest` and `TensorConv3dExpressionTest`.

Expected documentation/[planning](../../../../glossary.md#term-planning) (7):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless inaccurate: [Runtime](../../../../glossary.md#term-runtime) and Training APIs; Conv1d/[Conv2d](../../../../glossary.md#convolution--conv2d), [padding](../../../../glossary.md#padding)/window,
[operation](../../../../glossary.md#operation)/[signature](../../../../glossary.md#term-signature), TensorFactory/[producer](../../../../glossary.md#term-producer)/[provenance](../../../../glossary.md#provenance), [Shape](../../../../glossary.md#shape)/[Dimension](../../../../glossary.md#dimension), [DataType](../../../../glossary.md#data-type--datatype) promotion, and
[gradient](../../../../glossary.md#term-gradient) contracts; architecture/ADRs/tests; [Compiler](../../../../glossary.md#term-compiler) source/tests/master and Draft task state;
[conformance](../../../../glossary.md#term-conformance)/integration; [Gradle](../../../../glossary.md#term-gradle); other [modules](../../../../glossary.md#term-module).

## Maximum [scope](../../../../glossary.md#term-scope)

Exactly 28 paths maximum: four production, seventeen [Model](../../../../glossary.md#term-model) test, and seven documentation/[planning](../../../../glossary.md#term-planning)
paths. No production or test path outside Model may change. Stop for path 28, another public type
or method, an existing semantic-helper modification, architecture, [Gradle](../../../../glossary.md#term-gradle), [dependency](../../../../glossary.md#term-dependency), [compiler](../../../../glossary.md#term-compiler),
[backend](../../../../glossary.md#backend), [runtime](../../../../glossary.md#term-runtime), generated-code, [conformance](../../../../glossary.md#term-conformance), or integration work.

This is one cohesive Model capability: kind, [immutable](../../../../glossary.md#term-immutable) attributes, both natural [Tensor](../../../../glossary.md#tensor) forms,
local semantics, [canonical](../../../../glossary.md#term-canonical-form) [provenance](../../../../glossary.md#provenance), tests, public locks, Javadocs, APIs, glossary, capability
status, and planning must land together.

## [Javadoc](../../../../glossary.md#term-javadoc) and documentation requirements

- Fully document `Conv3dAttrs`, its [canonical](../../../../glossary.md#term-canonical-form) constructor and `defaults()`, `Conv3dKind`, the
  package-private helper, and both Tensor methods. Every parameter, result, nullability rule,
  constraint, failure, arithmetic/ID effect, and ownership rule receives the required tags.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [parameter glossary](../../../../glossary.md#term-parameter); [nullability glossary](../../../../glossary.md#term-nullability); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Tensor API explains NCDHW and weight/bias Shapes, axis order, grouped cross-correlation, formula,
  static versus symbolic relations, symmetric padding, numerical policy, and one complete biased
  grouped metadata example.<span data-glossary-navigation="true"> ([NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [weight glossary](../../../../glossary.md#term-weight); [bias glossary](../../../../glossary.md#term-bias); [Shapes glossary](../../../../glossary.md#shape); [axis glossary](../../../../glossary.md#term-axis); [padding glossary](../../../../glossary.md#padding); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- [Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api) states that [`CONV3D`](../../../../glossary.md#conv3d--ncdhw) is Model-current but Compiler forward adoption and final proof
  remain Draft 0006B, with backward-capable requests required to stay fail-closed until that task;
  it makes no inference, gradient, execution, or capability claim.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [inference glossary](../../../../glossary.md#term-inference); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Glossary defines Conv3d/NCDHW consistently and distinguishes first-class Conv3d from composed
  Conv1d and first-class Conv2d while reusing group, dilation, effective-kernel, cross-correlation,
  and conceptual-padding terms.<span data-glossary-navigation="true"> ([Conv2d glossary](../../../../glossary.md#convolution--conv2d); [dilation glossary](../../../../glossary.md#term-dilation); [kernel glossary](../../../../glossary.md#kernel))</span>
- Model capabilities records the Model semantic/API capability only and names Compiler 0006B,
  0006C, and later CPU work as absent downstream consumers.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- Synchronize this task and Model master to Complete only after implementation, documentation,
  and validation pass. Preserve 0025G Complete; keep 0026 and Compiler 0006B/0006C Draft without
  detailed specifications.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Record reasoned no-change conclusions for Runtime/Training APIs, related contracts,
  architecture/tests, Compiler planning/source/tests, conformance/integration, Gradle, and other
  modules.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [planning glossary](../../../../glossary.md#term-planning); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

## Acceptance criteria

- `Conv3dAttrs` has exactly ten primitive-long components in the specified order, implements only
  [`OperationAttrs`](../../../../glossary.md#operationattrs), validates exact order/messages, and supplies exact defaults.
- `Conv3dKind.CONV3D` is the only constant and exposes exactly the immutable two-to-three-input,
  one-output <span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [immutable glossary](../../../../glossary.md#term-immutable))</span>`Conv3dAttrs` [signature](../../../../glossary.md#term-signature).
- Exactly two [`conv3d`](../../../../glossary.md#conv3d--ncdhw) Tensor receiver overloads exist and public Tensor method count is 206. No
  alias, nullable-bias form, default-attrs form, static namespace, ConvNd, options, or arrays exist.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [alias glossary](../../../../glossary.md#aliasing-and-arena); [bias glossary](../../../../glossary.md#term-bias))</span>
- Static and symbolic D/H/W derivation, exact N/C-out Dimension reference retention, positive
  static kernels, checked arithmetic, group/channel/bias relations, and deferred unresolved
  relations match this task in exact axis and validation order.<span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension); [kernels glossary](../../../../glossary.md#kernel); [channel glossary](../../../../glossary.md#term-channel); [axis glossary](../../../../glossary.md#term-axis); [validation glossary](../../../../glossary.md#term-validation))</span>
- Every ordered floating input/weight/bias combination promotes correctly; BOOL and integral
  roles fail. Numerical meaning is documented and tested as metadata without Model evaluation.<span data-glossary-navigation="true"> ([weight glossary](../../../../glossary.md#term-weight); [BOOL glossary](../../../../glossary.md#data-type--datatype); [metadata glossary](../../../../glossary.md#term-metadata); [Model glossary](../../../../glossary.md#term-model); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- Each success creates one fresh ordinary `CONV3D` occurrence and canonical output with exact
  ordered provenance, output index zero, promoted descriptor, unresolved layout, empty label and
  storage, requires-grad OR, and exactly one consumed ID. Every local failure consumes none.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [canonical glossary](../../../../glossary.md#term-canonical-form); [provenance glossary](../../../../glossary.md#provenance); [descriptor glossary](../../../../glossary.md#term-descriptor); [layout glossary](../../../../glossary.md#layout))</span>
- No Conv1d/Conv2d, padding/window, Shape/Dimension, promotion, operation/signature, factory,
  provenance, compiler, gradient, backend, runtime, execution, architecture, dependency, or build
  behavior changes.<span data-glossary-navigation="true"> ([Conv2d glossary](../../../../glossary.md#convolution--conv2d); [padding glossary](../../../../glossary.md#padding); [Shape glossary](../../../../glossary.md#shape); [operation glossary](../../../../glossary.md#operation); [compiler glossary](../../../../glossary.md#term-compiler); [gradient glossary](../../../../glossary.md#term-gradient); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Focused tests, exactly one final Model suite, Javadoc/docs checks, exact 28 paths/public surface,
  single Ready/Complete Model frontier, dependency/order consistency, later-spec absence, and
  whitespace validation pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Ready glossary](../../../../glossary.md#term-ready))</span>
- A separate clean documentation-focused pass finalizes affected Javadocs/docs and records the
  required no-change conclusions before completion.

## Tests / [validation](../../../../glossary.md#term-validation)

Focused development validation:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.convolution.Conv3dSemanticsTest --tests io.github.pho001.synaptik.model.tensor.TensorConv3dExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorConv2dExpressionTest --tests io.github.pho001.synaptik.model.shape.DimensionExpressionsTest --tests io.github.pho001.synaptik.model.datatype.DataTypePromotionTest --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest --tests io.github.pho001.synaptik.model.tensor.TensorProducerTest --tests io.github.pho001.synaptik.model.tensor.TensorProvenanceTest --tests io.github.pho001.synaptik.model.tensor.TensorFactoryTest
```

After executable Java stabilizes, exactly once:

```bash
./gradlew :modules:model:test
```

Documentation pass after final Javadocs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

Also validate local Markdown links/anchors, balanced fences, final newlines, trailing whitespace,
exact paths/packages/[signatures](../../../../glossary.md#term-signature)/counts, absence of ConvNd and geometry arrays, task/master/roadmap
status consistency, exactly one [Ready](../../../../glossary.md#term-ready) [Model](../../../../glossary.md#term-model) frontier while implementation is pending, hard
0025G-before-0025H-before-[Compiler](../../../../glossary.md#term-compiler)-0006B order, Compiler 0006B/0006C and Model 0026 Draft status,
absence of detailed task files for those later rows, and empty staging. Repository-wide tests are
deferred to the next recorded cross-[module](../../../../glossary.md#term-module) [checkpoint](../../../../glossary.md#term-checkpoint) or [CI](../../../../glossary.md#term-continuous-integration-ci) because this task changes one module
without [dependency](../../../../glossary.md#term-dependency) or architecture boundaries.

Generate Javadocs only after executable Java stabilizes. Do not repeat a successful Model suite
in the documentation context unless that pass changes executable Java or finds a concrete reason.

Validation tier: task-level Model validation plus documentation validation. Architecture,
[backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), integration, and repository-wide suites are not required because no boundary,
backend, end-to-end, shared build, or multi-module executable behavior changes.

## Dependencies

Hard prerequisites, all Complete:

- Model 0018K–0018N, including operation signatures, immutable provenance, symbolic Dimension
  arithmetic/adoption, and typed semantic attributes;<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [signatures glossary](../../../../glossary.md#term-signature); [immutable glossary](../../../../glossary.md#term-immutable); [provenance glossary](../../../../glossary.md#provenance); [Dimension glossary](../../../../glossary.md#dimension))</span>
- [Model](../../../../glossary.md#term-model) 0018V and 0019 for portable floating accumulation/promotion vocabulary;
- Model 0020 for the completed grouped [NCHW](../../../../glossary.md#term-nchw) [Conv2d](../../../../glossary.md#convolution--conv2d) contract used as the [rank](../../../../glossary.md#term-rank) extension [oracle](../../../../glossary.md#term-test-oracle);
- Model 0025 and 0025G for canonical producer outputs and the settled rank-specific convolution
  program.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [producer glossary](../../../../glossary.md#term-producer); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers))</span>

The [architecture contract](../../../../glossary.md#architecture-contract), [planning](../../../../glossary.md#term-planning) guide, Model master plan, and roadmap remain controlling.
Legacy may be consulted read-only for selected observable capability evidence but supplies no
architecture, packages, dependencies, or implementation.

## Follow-up tasks

- Compiler 0006B remains the next separate Draft consumer. It independently infers and
  final-validates forward NCDHW descriptors and deferred relations, treats each <span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>[`CONV3D`](../../../../glossary.md#conv3d--ncdhw) producer
  as one ordinary flat node, and rejects backward-capable requests containing it before derivative
  allocation.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [node glossary](../../../../glossary.md#node); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- Compiler 0006C remains a separate Draft gradient-closure proof after 0006B. It may implement
  input/weight/bias adjoints only if group isolation, dilation/padding, overlap accumulation,
  symbolic Shape, and higher-order closure are expressible through the current public Tensor
  algebra; otherwise it must first select the smallest separately planned Model prerequisite.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [weight glossary](../../../../glossary.md#term-weight); [bias glossary](../../../../glossary.md#term-bias); [dilation glossary](../../../../glossary.md#term-dilation); [padding glossary](../../../../glossary.md#padding); [Shape glossary](../../../../glossary.md#shape); [Tensor glossary](../../../../glossary.md#tensor); [Model glossary](../../../../glossary.md#term-model))</span>
- CPU 0008A may add direct Conv3d execution only after Compiler 0006B and the CPU 0008 portable
  Conv2d foundation. Compiler 0006C does not block forward execution.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Conv2d glossary](../../../../glossary.md#convolution--conv2d))</span>
- Engine 0004 and NN 0025 remain later consumers. Model 0026 remains the separate future FLOAT16
  semantic foundation.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [FLOAT16 glossary](../../../../glossary.md#term-float16))</span>

Do not create any follow-up detailed specification in this task.

## Implementation prompt

Implement [Model](../../../../glossary.md#term-model) task 0025H exactly as specified. Read the repository instructions, architecture,
[planning](../../../../glossary.md#term-planning) guide, this task, completed 0020/0025/0025G contracts, current [convolution](../../../../glossary.md#term-channels-first-convolution-layers)/[Shape](../../../../glossary.md#shape)/promotion/
factory/[provenance](../../../../glossary.md#provenance) source and tests, and applicable documentation profiles in full. Add only
`Conv3dAttrs`, `Conv3dKind.CONV3D`, package-private `TensorConv3dExpressions`, and the two specified
[Tensor](../../../../glossary.md#tensor) receiver overloads. Preserve exact [NCDHW](../../../../glossary.md#conv3d--ncdhw)/[weight](../../../../glossary.md#term-weight)/[bias](../../../../glossary.md#term-bias) Shapes, static/symbolic checked
geometry, local/deferred relations, floating numerical meaning, [validation](../../../../glossary.md#term-validation) order/messages/effects,
[canonical](../../../../glossary.md#term-canonical-form) one-output provenance, and the 28-path ceiling. Add no ConvNd, arrays/options, [compiler](../../../../glossary.md#term-compiler),
[gradient](../../../../glossary.md#term-gradient), [backend](../../../../glossary.md#backend), [runtime](../../../../glossary.md#term-runtime), execution, [decomposition](../../../../glossary.md#term-decomposition), or generated-code behavior. [Run](../../../../glossary.md#term-run) focused and
final Model validation once at the defined tiers. Then use a distinct clean documentation-focused
context to finalize Javadocs/APIs/glossary/capabilities/planning and required no-change conclusions
without repeating successful Java tests. Stop and request clarification if implementation reveals
an architecture uncertainty. Do not [commit](../../../../glossary.md#term-commit), stage, or push unless separately authorized.

## Local decisions

- Conv3d remains one rank-specific first-class Model occurrence. Its exact public description uses
  NCDHW axis order, weight <span data-glossary-navigation="true"> ([Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [rank glossary](../../../../glossary.md#term-rank); [Model glossary](../../../../glossary.md#term-model); [occurrence glossary](../../../../glossary.md#term-occurrence); [NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [axis glossary](../../../../glossary.md#term-axis); [weight glossary](../../../../glossary.md#term-weight))</span>`[C_out, C_in/groups, K_d, K_h, K_w]`, optional [bias](../../../../glossary.md#term-bias) `[C_out]`, and the
  ten primitive `Conv3dAttrs` fields in [stride](../../../../glossary.md#term-stride), [padding](../../../../glossary.md#padding), [dilation](../../../../glossary.md#term-dilation), groups order.
- Static relations fail locally and symbolic spatial formulas retain their unresolved obligations.
  Exact inputs, descriptors, attributes, producer output index zero, and canonical wrapper
  identity are the only retained provenance mechanism; no new constraint or factory abstraction
  was introduced.<span data-glossary-navigation="true"> ([spatial glossary](../../../../glossary.md#term-spatial-axes); [descriptors glossary](../../../../glossary.md#term-descriptor); [producer glossary](../../../../glossary.md#term-producer); [canonical glossary](../../../../glossary.md#term-canonical-form); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance))</span>
- The Compile API distinguishes the 132-fingerprint current Model inventory from the preceding
  131-fingerprint Compiler forward inventory and 128-fingerprint first-order closure. Conv3d does
  not enter either Compiler inventory in this task.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [fingerprint glossary](../../../../glossary.md#term-fingerprint); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- The documentation pass applied the General, API/Javadoc, Planning, and Example profiles. The
  Tensor API includes one Java 26 biased grouped metadata example; the glossary adds the reusable
  Conv3d/NCDHW distinction without duplicating existing group, dilation, effective-kernel,
  cross-correlation, or conceptual-padding definitions.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Planning glossary](../../../../glossary.md#term-planning); [Tensor glossary](../../../../glossary.md#tensor); [metadata glossary](../../../../glossary.md#term-metadata); [kernel glossary](../../../../glossary.md#kernel))</span>

## Known limitations

- Kernel depth, height, and width must be statically positive. Input spatial Dimensions and
  channel or bias relations may remain unresolved, but a later concrete value must satisfy every
  retained relation.<span data-glossary-navigation="true"> ([Kernel glossary](../../../../glossary.md#kernel); [spatial glossary](../../../../glossary.md#term-spatial-axes); [Dimensions glossary](../../../../glossary.md#dimension); [channel glossary](../../../../glossary.md#term-channel); [bias glossary](../../../../glossary.md#term-bias))</span>
- Compiler 0006B remains the next separate Draft forward consumer, and Compiler 0006C remains the
  separate Draft gradient-closure decision. Neither has a detailed specification. Model 0026 also
  remains Draft without a detailed specification.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [gradient glossary](../../../../glossary.md#term-gradient); [Model glossary](../../../../glossary.md#term-model))</span>
- Repository-wide, architecture, conformance, and integration suites remain deferred under the
  recorded single-module validation tier. This documentation pass reused the stabilized Model
  test evidence because it changed only Javadoc and Markdown, not executable Java behavior.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [module glossary](../../../../glossary.md#term-module); [validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- The implementation context ran the focused command from this task and passed 69 tests. After
  executable Java stabilized, it ran the single final `./gradlew :modules:model:test`; 133 suites
  and 1,069 tests passed with zero failures, errors, or skips. Clean documentation context
  `/root` reused those results and did not repeat either Java suite.
- Documentation context `/root` independently reviewed the complete 28-path diff, new source and
  tests, affected Javadocs, completed tasks 0020/0025/0025G, Shape/Dimension, promotion,
  producer/provenance/factory, and Compiler inventory boundaries. It changed no executable Java.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- `./gradlew :modules:model:javadoc` passed after final Javadoc edits with two executed tasks.
  Rendered <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`Conv3dAttrs.html`, `Conv3dKind.html`, and both `Tensor.conv3d` sections in
  `Tensor.html` contain the reviewed Shape, attributes, numerical, provenance, parameter, result,
  and failure contracts. The package-private helper source Javadoc was reviewed directly.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter))</span>
- The documented Java 26 example compiled and ran against [Model](../../../../glossary.md#term-model) classes. It printed
  `Shape[2, 6, 4, 7, 4]`, [`FLOAT64`](../../../../glossary.md#term-float64), `true`, [`CONV3D`](../../../../glossary.md#conv3d--ncdhw), `true`, and `true`, confirming the exact
  documented metadata, attributes reference, and ordered provenance.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>
- Reflection reported exactly 206 public [Tensor](../../../../glossary.md#tensor) methods, exactly two `conv3d` overloads, ten
  `Conv3dAttrs` components, and one `Conv3dKind` [constant](../../../../glossary.md#term-constant). `javap -public` showed exactly the two
  receiver signatures. <span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>`javap -private` showed the package-private final helper with no fields,
  two package-private `apply` methods, and only private construction helpers. Import and manual
  scans found no Compiler, Prepare, Runtime, or backend references, no <span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>`ConvNd`, and no geometry
  arrays in the four production paths.
- The targeted Markdown validator passed seven files, 1,029 local links including 316 target
  anchors, balanced fences, final newlines, carriage-return absence, and trailing-whitespace
  checks. The exact-scope validator passed 28 authorized paths: four production, seventeen Model
  tests, and seven documentation/planning files. The staging area is empty.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [scope glossary](../../../../glossary.md#term-scope); [planning glossary](../../../../glossary.md#term-planning))</span>
- Status and ordering checks preserve 0025G Complete, set 0025H Complete, keep Model 0026 and
  Compiler 0006B/0006C Draft, show no Model task Ready or In progress, preserve the hard
  0025G-before-0025H-before-Compiler-0006B sequence, and find no detailed later specification.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
  `git diff --check` passed on the final combined change.
- Runtime and Training APIs remain unchanged because this task adds only Model expression
  metadata and no API owned by either document. Conv1d/Conv2d, padding/window,
  operation/signature, TensorFactory/producer/provenance, Shape/Dimension, promotion, and existing
  gradient contracts remain unchanged because Conv3d composes their established value contracts
  without modifying them.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [padding glossary](../../../../glossary.md#padding); [operation glossary](../../../../glossary.md#operation); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- `ARCHITECTURE.md`, focused architecture pages, ADRs, and architecture tests remain unchanged
  because ownership, dependencies, and lifecycle rules did not change. Compiler source/tests and
  master-plan status remain unchanged because 0006B/0006C still own later adoption and closure.
  Backend conformance, integration, Gradle/build configuration, and other modules remain unchanged
  because this task adds no behavior in those areas.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

## Implementation notes

- Added public `Conv3dAttrs`, `Conv3dKind.CONV3D`, package-private field-free
  `TensorConv3dExpressions`, and exactly two `Tensor.conv3d` receiver overloads. Seventeen Model
  tests cover semantic attributes, exact public surface, validation, static/symbolic Shapes,
  promotion, effects, and canonical provenance; fifteen existing public-count locks now expect
  206 methods.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [validation glossary](../../../../glossary.md#term-validation); [Shapes glossary](../../../../glossary.md#shape); [canonical glossary](../../../../glossary.md#term-canonical-form); [provenance glossary](../../../../glossary.md#provenance))</span>
- The independent documentation pass finalized the three new type/helper Javadocs, both Tensor
  method Javadocs, Tensor and Compile APIs, glossary, Model capabilities, this task, Model master
  plan, and roadmap. It preserved the implementation context's initial 24-path scope and expanded
  it only by the four missing explanatory documents to the exact authorized 28 paths.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [scope glossary](../../../../glossary.md#term-scope))</span>

## Completion summary

- Completed changes: Added first-class grouped NCDHW Conv3d Model semantics, immutable ten-field
  attributes, biased and unbiased Tensor construction, static/symbolic geometry, floating
  numerical policy, canonical provenance, focused tests/public locks, complete Javadocs, and
  synchronized explanatory/planning documentation.<span data-glossary-navigation="true"> ([NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [Model glossary](../../../../glossary.md#term-model); [immutable glossary](../../../../glossary.md#term-immutable); [Tensor glossary](../../../../glossary.md#tensor); [canonical glossary](../../../../glossary.md#term-canonical-form); [provenance glossary](../../../../glossary.md#provenance); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: Exactly 28 authorized paths: four production, seventeen Model test,
  and seven documentation/planning paths.
- Tests and validation: Reused passing focused 69-test and final 133-suite/1,069-test Model
  evidence; passed Model Javadoc, rendered inspection, Java 26 example, reflection/<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`javap`/import/
  manual checks, 1,029-link/316-anchor Markdown validation, exact scope/status/order/staging checks,
  and <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check`.
- Documentation-agent review: Clean context `/root` independently finalized all affected
  Javadocs, APIs, glossary/capability impact, planning evidence, and status.
- Documentation impact: Tensor and Compile APIs, glossary, Model capabilities, task, Model master
  plan, and roadmap now describe the current Model capability and exact planned Compiler boundary.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: `Conv3dAttrs`, its constructor/defaults, `Conv3dKind`, the package-private helper,
  and both Tensor methods have complete, rendered contracts; related existing Javadocs remain
  accurate unchanged.
- Glossary impact: Added Conv3d/NCDHW and updated inventory wording to distinguish first-class
  Conv3d from composed Conv1d and first-class Conv2d.<span data-glossary-navigation="true"> ([Conv2d glossary](../../../../glossary.md#convolution--conv2d))</span>
- Unresolved issues: None within task 0025H.
- Follow-up required: Compiler 0006B remains the next separate Draft forward consumer; Compiler
  0006C remains the separate Draft gradient closure.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient))</span>

Status: Complete
