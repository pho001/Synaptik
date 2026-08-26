# Task 0025H: NCDHW Conv3d Semantics and Tensor Expressions

## Status

Complete

## Goal

Add one first-class grouped three-dimensional cross-correlation semantic operation and the natural
public Tensor expressions for channels-first NCDHW tensors. The Model capability consists of one
rank-specific immutable attribute value, one `CONV3D` operation kind, and exactly two receiver
methods for unbiased and biased construction. It defines metadata, Shape, floating numerical
meaning, and provenance only; later Compiler, backend, and execution work remain separate.

## Scope

- Add immutable public
  `Conv3dAttrs(strideDepth, strideHeight, strideWidth, paddingDepth, paddingHeight, paddingWidth,
  dilationDepth, dilationHeight, dilationWidth, groups)` with exact declaration-order validation
  and `defaults()` equal to `(1, 1, 1, 0, 0, 0, 1, 1, 1, 1)`.
- Add public `Conv3dKind.CONV3D` with exactly one immutable signature accepting `Conv3dAttrs`, two
  or three ordered inputs, and exactly one output.
- Add exactly these public receiver methods to `Tensor`:

  ```java
  public Tensor conv3d(Tensor weight, Conv3dAttrs attrs)

  public Tensor conv3d(Tensor weight, Tensor bias, Conv3dAttrs attrs)
  ```

- Add one package-private, final, field-free `TensorConv3dExpressions` construction owner.
- Accept floating input `[N, C_in, D, H, W]`, weight
  `[C_out, C_in/groups, K_d, K_h, K_w]`, and optional bias `[C_out]`; derive
  `[N, C_out, D_out, H_out, W_out]`.
- Validate nulls, floating roles and promotion, exact ranks, positive static kernel extents,
  locally provable group/channel/bias relations, and checked static or symbolic spatial geometry
  before creating an operation, producer, descriptor result, or Tensor identity.
- Retain the exact operation attributes, ordered Tensor inputs, derived output descriptor, one
  canonical output wrapper, and output-index-zero provenance.
- Define grouped cross-correlation, promotion, accumulation, conceptual padding, special-value,
  reassociation, and determinism meaning as the rank-three-spatial extension of current Conv2d.
- Test the exact kind/attribute/public surface, Shapes, local/deferred relations, validation order
  and messages, effects, freshness, canonical provenance, and checked arithmetic.
- Finalize affected Javadocs, Tensor/Compile APIs, glossary, Model capabilities, and planning in a
  separate clean documentation-focused context before completion.

## Out of scope

- `ConvNd`, a dynamic-rank convolution, public or private geometry arrays, options arrays, nullable
  bias, default-attribute overloads, aliases, static namespaces, or additional public entry points.
- Conv1d or Conv2d changes, decomposition into existing operations, kernel unrolling, generated
  code, eager evaluation, or any claim that current operations can express Conv3d with bounded
  graph size independent of spatial extents.
- Asymmetric intrinsic padding, padding modes, transposed, deformable, causal-specialized,
  depthwise-specific, separable, quantized, sparse, or channels-last convolution.
- Compiler capture, inference, final validation, constraints, graph transformations, gradients,
  gradient policy, or derivative formulas. Compiler 0006B remains the next separate Draft forward
  consumer and explicit fail-closed backward boundary; Compiler 0006C remains the separate Draft
  adjoint-expressibility and gradient-closure task.
- Planning, Prepare, Runtime, Engine, backend, algorithm selection, lowering, fusion, allocation,
  capability advertisement, conformance, integration, or execution implementation.
- A neural-network layer, module, parameters, initializer, factory recipe, state, or training API.
- Changes to TensorFactory, TensorProducer, TensorProvenance, Operation, OperationSignature,
  Shape/Dimension arithmetic, DataType promotion, PAD/window semantics, or existing convolution
  kinds and helpers.
- Detailed specifications for Model 0026, Compiler 0006B/0006C, CPU 0008/0008A, Engine 0004, or
  NN 0025.
- Legacy source, package structure, dependency direction, or implementation reuse.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Model capabilities](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Operation signature hardening](0018k-operation-signature-and-construction-hardening.md)
- [Shared multi-output Tensor provenance](0018l-shared-multi-output-tensor-provenance.md)
- [Symbolic extent expressions](0018m-symbolic-extent-expressions.md)
- [Dynamic-extent adoption](0018m1-dynamic-extent-adoption.md)
- [Typed scalar value contract](0018n-typed-scalar-value-contract.md)
- [Multi-axis and statistical reductions](0018v-multi-axis-and-statistical-reductions.md)
- [Matmul semantics and Tensor expression](0019-matmul-semantics-and-tensor-expression.md)
- [NCHW Conv2d semantics and Tensor expressions](0020-nchw-conv2d-semantics-and-tensor-expressions.md)
- [Canonical TensorProducer outputs](0025-canonical-tensor-producer-outputs.md)
- [NCW Conv1d composition](0025g-ncw-conv1d-composition.md)
- [Compiler master plan](../../compiler/master-plan.md)

## Architecture constraints

- Work stays in Model and directly affected documentation/planning. Model owns immutable Tensor,
  operation, descriptor, Shape, and provenance semantics; it does not capture graphs, infer a
  compiled graph, construct gradients, choose execution, or depend on downstream modules.
- `Tensor` remains the public fluent receiver facade. The helper depends only on existing Model
  datatype, operation, Shape, descriptor, factory, and provenance contracts.
- `CONV3D` is one ordinary flat one-output semantic operation. It has no region, subgraph,
  decomposition, hidden auxiliary, backend payload, execution state, or generated implementation.
- Every successful call creates one immutable `Operation`, one `TensorProducer`, one descriptor,
  and the producer-owned canonical output wrapper through the existing `TensorFactory` seam.
- Ordered inputs are exact Tensor references. Attributes are retained by exact reference.
  Provenance remains immutable and output-index zero.
- `Conv3dAttrs` is rank-specific. A common helper may remain private only if it reflects an
  already-proved local implementation responsibility; this task adds no shared public ConvNd
  abstraction and changes no completed Conv2d contract.
- Compiler owns independent forward inference, proof of deferred relations, and gradients.
  Prepare/backend work owns algorithms, lowering, allocation, and materialization. Runtime
  receives only prepared work.
- No architecture, module-boundary, dependency, lifecycle, build, or architecture-test rule
  changes.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.datatype`
- `io.github.pho001.synaptik.model.shape`
- `io.github.pho001.synaptik.model.operation`
- `io.github.pho001.synaptik.model.operation.convolution`
- `io.github.pho001.synaptik.model.tensor`

Packages changed:

- add the rank-three-spatial kind and immutable attributes to
  `io.github.pho001.synaptik.model.operation.convolution`;
- add the package-private expression owner and two receiver overloads to
  `io.github.pho001.synaptik.model.tensor`;
- add no package and widen no existing helper or factory seam.

Type placement:

- `...operation.convolution.Conv3dAttrs` owns only validated depth/height/width stride, symmetric
  padding, dilation, and groups.
- `...operation.convolution.Conv3dKind` owns only the exact operation name and signature.
- `...tensor.TensorConv3dExpressions` owns local validation, Shape derivation, descriptor and
  operation construction, and delegation to the existing derived-output factory seam.
- `...tensor.Tensor` exposes the two natural single-output receiver entries parallel to Conv2d.

## Exact semantic and expression contract

### Signature and input order

`Conv3dKind.CONV3D.signatures()` is exactly:

```java
List.of(OperationSignature.inputRange(Conv3dAttrs.class, 2, 3, 1))
```

The unbiased occurrence has ordered inputs `[input, weight]`. The biased occurrence has ordered
inputs `[input, weight, bias]`. Both have one output at index zero. No other attribute type,
arity, output count, or signature is accepted.

### Shapes and geometry

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
checked signed `long` operations. For a static input extent, a negative numerator fails locally;
zero produces one output position. For an unresolved input extent, compute the checked constant
offset `2 * padding_x - effectiveKernel_x`, then use the existing canonical
`DimensionExpressions.addConstant`, `floorDivide`, and `addConstant(..., 1)` sequence. The future
concrete binding retains the obligation that each numerator is non-negative.

The result reuses the exact input batch Dimension reference and weight output-channel Dimension
reference. Static input and output channels must each be divisible by `groups`. When input
channels and weight channels per group are both static, checked
`weightChannelsPerGroup * groups` must equal input channels. When present bias length and output
channels are both static, they must be equal. Any relation containing an unresolved Dimension is
deferred and remains observable through exact inputs, descriptors, and attributes; Model creates
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

The canonical constructor validates in that same order with exact messages:

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

All components are primitive `long`; the record implements only `OperationAttrs`, has no nested
type, performs no geometry calculation, and exposes no arrays or mutable state. `defaults()`
returns a fresh or value-equivalent `(1, 1, 1, 0, 0, 0, 1, 1, 1, 1)` instance.

### Values and numerical policy

Conv3d means grouped NCDHW cross-correlation and does not reverse the stored kernel. Output
channel `o` selects its contiguous input-channel group. For each output coordinate, products are
conceptually visited in increasing logical input-channel, kernel-depth, kernel-height, then
kernel-width order. The optional bias for `o` participates exactly once in the accumulation
domain before final output conversion.

Input, weight, and present bias must each be floating. Promotion processes input and weight first,
then present bias, using the current floating hierarchy. FLOAT64 output accumulates in FLOAT64;
FLOAT32 and BFLOAT16 output accumulate in FLOAT32, with final conversion for BFLOAT16.

Out-of-range spatial coordinates are conceptual positive-zero input samples and participate in
ordinary IEEE-754 multiplication, including zero multiplied by infinity. NaN, infinity, and
signed zero otherwise follow ordinary multiplication and addition. An empty input-channel
contraction starts at positive zero before optional bias. Empty batch or output-channel axes are
valid. Reassociation and fused multiply-add are permitted, so neither fixed summation order nor
bitwise cross-backend-identical rounding is promised. These rules define represented meaning;
Model reads no tensor values and chooses no algorithm.

### Result descriptor and provenance

After all local validation succeeds, construct exactly:

1. one `TensorDescriptor` with promoted data type, exact derived NCDHW Shape, unresolved layout,
   and `requiresGrad` equal to the logical OR of input, weight, and present bias;
2. one `Operation(Conv3dKind.CONV3D, attrs)` retaining the exact `attrs` reference; and
3. one derived Tensor through the existing single-output factory seam with empty label, no host
   storage, ordered exact inputs, and output index zero.

The result is the exact canonical object returned by its producer's `output(0)`. Its provenance
returns the same operation, exact ordered inputs, exact output descriptor, and index zero.
Equivalent calls create identity-distinct operations, producers, descriptors, wrappers, and IDs.
Every success consumes exactly one fresh Tensor ID.

### Local versus deferred relations

Model proves only relations whose participating Dimensions are static at construction:

- positive kernel depth, height, and width are always local because kernel extents must be static;
- group divisibility is local only for a static channel extent;
- grouped weight/input equality is local only when both channel extents are static;
- bias/output equality is local only when both extents are static;
- padded-kernel fit is local only for a static input spatial extent.

All other relations are deferred to Compiler inference/final validation and eventual binding.
Deferral is acceptance of unresolved metadata, not proof that an arbitrary future binding is
valid. Model adds no compiler constraint, inference, binding, gradient, or execution behavior.

### Validation order, messages, and effects

Each overload null-checks in declaration order: `input`, `weight`, optional `bias`, then `attrs`,
using the parameter name as the `NullPointerException` message. The helper then validates:

1. input, weight, then present-bias floating eligibility;
2. input/weight and then present-bias floating promotion;
3. input rank five, weight rank five, then present-bias rank one;
4. positive static kernel depth, then height, then width;
5. static input-channel divisibility, output-channel divisibility, grouped weight/input equality,
   then bias/output equality;
6. checked output depth, then height, then width geometry;
7. descriptor, operation, producer, and canonical output creation.

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

Invalid attributes fail when their record is constructed and therefore consume no Tensor ID.
Every helper-local null, semantic, or arithmetic failure occurs before operation/producer creation
and consumes no ID. Successful factory allocation consumes one ID. Existing identifier-exhaustion
behavior propagates without rollback or reuse. No path allocates storage, mutates an input,
creates a label, or consumes an ID speculatively before local validation completes.

## Public surface

- Add exactly one public enum type, one public record type, and two public Tensor receiver methods.
- `Conv3dKind` has exactly one enum constant and implements only `OperationKind`.
- `Conv3dAttrs` implements only `OperationAttrs`.
- `TensorConv3dExpressions` is package-private, final, field-free, non-instantiable, and exposes no
  public member.
- Public Tensor declared-method count changes from 204 to 206.
- Add no `ConvNd`, nullable-bias overload, default-attrs overload, factory method, builder, options
  object, array geometry, output record, or auxiliary output.

## Affected files

Expected production source (4):

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/convolution/Conv3dAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/convolution/Conv3dKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorConv3dExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Expected Model tests (17):

- add `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/convolution/Conv3dSemanticsTest.java`;
- add `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorConv3dExpressionTest.java`;
- update the exact Tensor public-surface locks in `TensorTest`, `TensorConv1dExpressionTest`,
  `RecurrentScanExpressionTest`,
  `TensorBatchNormInferenceExpressionTest`, `TensorBinaryArithmeticTest`,
  `TensorDenseCategoricalCrossEntropyWithLogitsExpressionTest`,
  `TensorIndexCategoricalCrossEntropyWithLogitsExpressionTest`, `TensorLayerNormExpressionTest`,
  `TensorLinearExpressionTest`, `TensorMatmulExpressionTest`, `TensorMeanSquaredErrorExpressionTest`,
  `TensorRmsNormExpressionTest`, `TensorScaledDotProductAttentionExpressionTest`,
  `TensorSlicePlacementExpressionTest`, and `TensorSumToShapeExpressionTest`, changing only the
  expected count from 204 to 206 except for direct Conv3d signature/name assertions owned by
  `TensorTest` and `TensorConv3dExpressionTest`.

Expected documentation/planning (7):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless inaccurate: Runtime and Training APIs; Conv1d/Conv2d, padding/window,
operation/signature, TensorFactory/producer/provenance, Shape/Dimension, DataType promotion, and
gradient contracts; architecture/ADRs/tests; Compiler source/tests/master and Draft task state;
conformance/integration; Gradle; other modules.

## Maximum scope

Exactly 28 paths maximum: four production, seventeen Model test, and seven documentation/planning
paths. No production or test path outside Model may change. Stop for path 28, another public type
or method, an existing semantic-helper modification, architecture, Gradle, dependency, compiler,
backend, runtime, generated-code, conformance, or integration work.

This is one cohesive Model capability: kind, immutable attributes, both natural Tensor forms,
local semantics, canonical provenance, tests, public locks, Javadocs, APIs, glossary, capability
status, and planning must land together.

## Javadoc and documentation requirements

- Fully document `Conv3dAttrs`, its canonical constructor and `defaults()`, `Conv3dKind`, the
  package-private helper, and both Tensor methods. Every parameter, result, nullability rule,
  constraint, failure, arithmetic/ID effect, and ownership rule receives the required tags.
- Tensor API explains NCDHW and weight/bias Shapes, axis order, grouped cross-correlation, formula,
  static versus symbolic relations, symmetric padding, numerical policy, and one complete biased
  grouped metadata example.
- Compile API states that `CONV3D` is Model-current but Compiler forward adoption and final proof
  remain Draft 0006B, with backward-capable requests required to stay fail-closed until that task;
  it makes no inference, gradient, execution, or capability claim.
- Glossary defines Conv3d/NCDHW consistently and distinguishes first-class Conv3d from composed
  Conv1d and first-class Conv2d while reusing group, dilation, effective-kernel, cross-correlation,
  and conceptual-padding terms.
- Model capabilities records the Model semantic/API capability only and names Compiler 0006B,
  0006C, and later CPU work as absent downstream consumers.
- Synchronize this task and Model master to Complete only after implementation, documentation,
  and validation pass. Preserve 0025G Complete; keep 0026 and Compiler 0006B/0006C Draft without
  detailed specifications.
- Record reasoned no-change conclusions for Runtime/Training APIs, related contracts,
  architecture/tests, Compiler planning/source/tests, conformance/integration, Gradle, and other
  modules.

## Acceptance criteria

- `Conv3dAttrs` has exactly ten primitive-long components in the specified order, implements only
  `OperationAttrs`, validates exact order/messages, and supplies exact defaults.
- `Conv3dKind.CONV3D` is the only constant and exposes exactly the immutable two-to-three-input,
  one-output `Conv3dAttrs` signature.
- Exactly two `conv3d` Tensor receiver overloads exist and public Tensor method count is 206. No
  alias, nullable-bias form, default-attrs form, static namespace, ConvNd, options, or arrays exist.
- Static and symbolic D/H/W derivation, exact N/C-out Dimension reference retention, positive
  static kernels, checked arithmetic, group/channel/bias relations, and deferred unresolved
  relations match this task in exact axis and validation order.
- Every ordered floating input/weight/bias combination promotes correctly; BOOL and integral
  roles fail. Numerical meaning is documented and tested as metadata without Model evaluation.
- Each success creates one fresh ordinary `CONV3D` occurrence and canonical output with exact
  ordered provenance, output index zero, promoted descriptor, unresolved layout, empty label and
  storage, requires-grad OR, and exactly one consumed ID. Every local failure consumes none.
- No Conv1d/Conv2d, padding/window, Shape/Dimension, promotion, operation/signature, factory,
  provenance, compiler, gradient, backend, runtime, execution, architecture, dependency, or build
  behavior changes.
- Focused tests, exactly one final Model suite, Javadoc/docs checks, exact 28 paths/public surface,
  single Ready/Complete Model frontier, dependency/order consistency, later-spec absence, and
  whitespace validation pass.
- A separate clean documentation-focused pass finalizes affected Javadocs/docs and records the
  required no-change conclusions before completion.

## Tests / validation

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
exact paths/packages/signatures/counts, absence of ConvNd and geometry arrays, task/master/roadmap
status consistency, exactly one Ready Model frontier while implementation is pending, hard
0025G-before-0025H-before-Compiler-0006B order, Compiler 0006B/0006C and Model 0026 Draft status,
absence of detailed task files for those later rows, and empty staging. Repository-wide tests are
deferred to the next recorded cross-module checkpoint or CI because this task changes one module
without dependency or architecture boundaries.

Generate Javadocs only after executable Java stabilizes. Do not repeat a successful Model suite
in the documentation context unless that pass changes executable Java or finds a concrete reason.

Validation tier: task-level Model validation plus documentation validation. Architecture,
backend-conformance, integration, and repository-wide suites are not required because no boundary,
backend, end-to-end, shared build, or multi-module executable behavior changes.

## Dependencies

Hard prerequisites, all Complete:

- Model 0018K–0018N, including operation signatures, immutable provenance, symbolic Dimension
  arithmetic/adoption, and typed semantic attributes;
- Model 0018V and 0019 for portable floating accumulation/promotion vocabulary;
- Model 0020 for the completed grouped NCHW Conv2d contract used as the rank extension oracle;
- Model 0025 and 0025G for canonical producer outputs and the settled rank-specific convolution
  program.

The architecture contract, planning guide, Model master plan, and roadmap remain controlling.
Legacy may be consulted read-only for selected observable capability evidence but supplies no
architecture, packages, dependencies, or implementation.

## Follow-up tasks

- Compiler 0006B remains the next separate Draft consumer. It independently infers and
  final-validates forward NCDHW descriptors and deferred relations, treats each `CONV3D` producer
  as one ordinary flat node, and rejects backward-capable requests containing it before derivative
  allocation.
- Compiler 0006C remains a separate Draft gradient-closure proof after 0006B. It may implement
  input/weight/bias adjoints only if group isolation, dilation/padding, overlap accumulation,
  symbolic Shape, and higher-order closure are expressible through the current public Tensor
  algebra; otherwise it must first select the smallest separately planned Model prerequisite.
- CPU 0008A may add direct Conv3d execution only after Compiler 0006B and the CPU 0008 portable
  Conv2d foundation. Compiler 0006C does not block forward execution.
- Engine 0004 and NN 0025 remain later consumers. Model 0026 remains the separate future FLOAT16
  semantic foundation.

Do not create any follow-up detailed specification in this task.

## Implementation prompt

Implement Model task 0025H exactly as specified. Read the repository instructions, architecture,
planning guide, this task, completed 0020/0025/0025G contracts, current convolution/Shape/promotion/
factory/provenance source and tests, and applicable documentation profiles in full. Add only
`Conv3dAttrs`, `Conv3dKind.CONV3D`, package-private `TensorConv3dExpressions`, and the two specified
Tensor receiver overloads. Preserve exact NCDHW/weight/bias Shapes, static/symbolic checked
geometry, local/deferred relations, floating numerical meaning, validation order/messages/effects,
canonical one-output provenance, and the 28-path ceiling. Add no ConvNd, arrays/options, compiler,
gradient, backend, runtime, execution, decomposition, or generated-code behavior. Run focused and
final Model validation once at the defined tiers. Then use a distinct clean documentation-focused
context to finalize Javadocs/APIs/glossary/capabilities/planning and required no-change conclusions
without repeating successful Java tests. Stop and request clarification if implementation reveals
an architecture uncertainty. Do not commit, stage, or push unless separately authorized.

## Local decisions

- Conv3d remains one rank-specific first-class Model occurrence. Its exact public description uses
  NCDHW axis order, weight `[C_out, C_in/groups, K_d, K_h, K_w]`, optional bias `[C_out]`, and the
  ten primitive `Conv3dAttrs` fields in stride, padding, dilation, groups order.
- Static relations fail locally and symbolic spatial formulas retain their unresolved obligations.
  Exact inputs, descriptors, attributes, producer output index zero, and canonical wrapper
  identity are the only retained provenance mechanism; no new constraint or factory abstraction
  was introduced.
- The Compile API distinguishes the 132-fingerprint current Model inventory from the preceding
  131-fingerprint Compiler forward inventory and 128-fingerprint first-order closure. Conv3d does
  not enter either Compiler inventory in this task.
- The documentation pass applied the General, API/Javadoc, Planning, and Example profiles. The
  Tensor API includes one Java 26 biased grouped metadata example; the glossary adds the reusable
  Conv3d/NCDHW distinction without duplicating existing group, dilation, effective-kernel,
  cross-correlation, or conceptual-padding definitions.

## Known limitations

- Kernel depth, height, and width must be statically positive. Input spatial Dimensions and
  channel or bias relations may remain unresolved, but a later concrete value must satisfy every
  retained relation.
- Compiler 0006B remains the next separate Draft forward consumer, and Compiler 0006C remains the
  separate Draft gradient-closure decision. Neither has a detailed specification. Model 0026 also
  remains Draft without a detailed specification.
- Repository-wide, architecture, conformance, and integration suites remain deferred under the
  recorded single-module validation tier. This documentation pass reused the stabilized Model
  test evidence because it changed only Javadoc and Markdown, not executable Java behavior.

## Validation evidence

- The implementation context ran the focused command from this task and passed 69 tests. After
  executable Java stabilized, it ran the single final `./gradlew :modules:model:test`; 133 suites
  and 1,069 tests passed with zero failures, errors, or skips. Clean documentation context
  `/root` reused those results and did not repeat either Java suite.
- Documentation context `/root` independently reviewed the complete 28-path diff, new source and
  tests, affected Javadocs, completed tasks 0020/0025/0025G, Shape/Dimension, promotion,
  producer/provenance/factory, and Compiler inventory boundaries. It changed no executable Java.
- `./gradlew :modules:model:javadoc` passed after final Javadoc edits with two executed tasks.
  Rendered `Conv3dAttrs.html`, `Conv3dKind.html`, and both `Tensor.conv3d` sections in
  `Tensor.html` contain the reviewed Shape, attributes, numerical, provenance, parameter, result,
  and failure contracts. The package-private helper source Javadoc was reviewed directly.
- The documented Java 26 example compiled and ran against Model classes. It printed
  `Shape[2, 6, 4, 7, 4]`, `FLOAT64`, `true`, `CONV3D`, `true`, and `true`, confirming the exact
  documented metadata, attributes reference, and ordered provenance.
- Reflection reported exactly 206 public Tensor methods, exactly two `conv3d` overloads, ten
  `Conv3dAttrs` components, and one `Conv3dKind` constant. `javap -public` showed exactly the two
  receiver signatures. `javap -private` showed the package-private final helper with no fields,
  two package-private `apply` methods, and only private construction helpers. Import and manual
  scans found no Compiler, Prepare, Runtime, or backend references, no `ConvNd`, and no geometry
  arrays in the four production paths.
- The targeted Markdown validator passed seven files, 1,029 local links including 316 target
  anchors, balanced fences, final newlines, carriage-return absence, and trailing-whitespace
  checks. The exact-scope validator passed 28 authorized paths: four production, seventeen Model
  tests, and seven documentation/planning files. The staging area is empty.
- Status and ordering checks preserve 0025G Complete, set 0025H Complete, keep Model 0026 and
  Compiler 0006B/0006C Draft, show no Model task Ready or In progress, preserve the hard
  0025G-before-0025H-before-Compiler-0006B sequence, and find no detailed later specification.
  `git diff --check` passed on the final combined change.
- Runtime and Training APIs remain unchanged because this task adds only Model expression
  metadata and no API owned by either document. Conv1d/Conv2d, padding/window,
  operation/signature, TensorFactory/producer/provenance, Shape/Dimension, promotion, and existing
  gradient contracts remain unchanged because Conv3d composes their established value contracts
  without modifying them.
- `ARCHITECTURE.md`, focused architecture pages, ADRs, and architecture tests remain unchanged
  because ownership, dependencies, and lifecycle rules did not change. Compiler source/tests and
  master-plan status remain unchanged because 0006B/0006C still own later adoption and closure.
  Backend conformance, integration, Gradle/build configuration, and other modules remain unchanged
  because this task adds no behavior in those areas.

## Implementation notes

- Added public `Conv3dAttrs`, `Conv3dKind.CONV3D`, package-private field-free
  `TensorConv3dExpressions`, and exactly two `Tensor.conv3d` receiver overloads. Seventeen Model
  tests cover semantic attributes, exact public surface, validation, static/symbolic Shapes,
  promotion, effects, and canonical provenance; fifteen existing public-count locks now expect
  206 methods.
- The independent documentation pass finalized the three new type/helper Javadocs, both Tensor
  method Javadocs, Tensor and Compile APIs, glossary, Model capabilities, this task, Model master
  plan, and roadmap. It preserved the implementation context's initial 24-path scope and expanded
  it only by the four missing explanatory documents to the exact authorized 28 paths.

## Completion summary

- Completed changes: Added first-class grouped NCDHW Conv3d Model semantics, immutable ten-field
  attributes, biased and unbiased Tensor construction, static/symbolic geometry, floating
  numerical policy, canonical provenance, focused tests/public locks, complete Javadocs, and
  synchronized explanatory/planning documentation.
- Files changed or created: Exactly 28 authorized paths: four production, seventeen Model test,
  and seven documentation/planning paths.
- Tests and validation: Reused passing focused 69-test and final 133-suite/1,069-test Model
  evidence; passed Model Javadoc, rendered inspection, Java 26 example, reflection/`javap`/import/
  manual checks, 1,029-link/316-anchor Markdown validation, exact scope/status/order/staging checks,
  and `git diff --check`.
- Documentation-agent review: Clean context `/root` independently finalized all affected
  Javadocs, APIs, glossary/capability impact, planning evidence, and status.
- Documentation impact: Tensor and Compile APIs, glossary, Model capabilities, task, Model master
  plan, and roadmap now describe the current Model capability and exact planned Compiler boundary.
- Javadoc review: `Conv3dAttrs`, its constructor/defaults, `Conv3dKind`, the package-private helper,
  and both Tensor methods have complete, rendered contracts; related existing Javadocs remain
  accurate unchanged.
- Glossary impact: Added Conv3d/NCDHW and updated inventory wording to distinguish first-class
  Conv3d from composed Conv1d and first-class Conv2d.
- Unresolved issues: None within task 0025H.
- Follow-up required: Compiler 0006B remains the next separate Draft forward consumer; Compiler
  0006C remains the separate Draft gradient closure.

Status: Complete
