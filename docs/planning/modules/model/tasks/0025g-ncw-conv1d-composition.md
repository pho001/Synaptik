# Task 0025G: NCW Conv1d Composition

## Status

Complete

## Goal

Add public one-dimensional grouped cross-correlation for tensors in NCW order: batch, channel,
width. The API must be a rank-specific convenience over current Model operations, not a new
semantic operation. Every successful call therefore returns the exact visible composition:

```text
NCW input   --EXPAND_DIMS(axis 2)--\
                                      CONV2D --SQUEEZE(axis 2)--> NCW result
1-D weight --EXPAND_DIMS(axis 2)--/
optional bias --------------------/
```

The two inserted axes are singleton height axes. The resulting `CONV2D` occurrence remains the
single owner of convolution meaning, Compiler inference and gradients, and later Conv2d
execution.

## Scope

- Add immutable public `Conv1dAttrs(stride, padding, dilation, groups)` with declaration-order
  validation and `defaults()` equal to `(1, 0, 1, 1)`.
- Add exactly these public receiver methods to `Tensor`:

  ```java
  public Tensor conv1d(Tensor weight, Conv1dAttrs attrs)

  public Tensor conv1d(Tensor weight, Tensor bias, Conv1dAttrs attrs)
  ```

- Add one package-private, final, field-free `TensorConv1dExpressions` construction owner.
- Accept floating input `[N, C_in, W]`, weight `[C_out, C_in/groups, K_w]`, and optional bias
  `[C_out]`; derive `[N, C_out, W_out]`.
- Validate the rank-one-dimensional contract locally before constructing the visible composition:
  nulls, floating roles, promotion, ranks, positive static kernel width, group divisibility,
  grouped input/weight compatibility, bias length, and checked static/symbolic output geometry.
- Expand input and weight at axis `2`, map the rank-one geometry to
  `Conv2dAttrs(1, stride, 0, padding, 1, dilation, groups)`, call the existing biased or unbiased
  `conv2d`, and squeeze the Conv2d height axis `2`.
- Preserve current Conv2d grouped cross-correlation, conceptual padding, floating promotion,
  accumulation, special-value, reassociation, and determinism policy.
- Test exact Shapes, validation/effects, public surface, four-producer provenance, canonical
  wrappers, freshness, and static/symbolic Dimension relations.
- Finalize affected Javadocs, Tensor/Compile APIs, glossary, Model capabilities, and planning in a
  separate clean documentation-focused context before completion.

## Out of scope

- A `CONV1D` kind, signature, producer, attributes occurrence, compiler inventory row, captured
  node, backend operation, or capability advertisement.
- `Conv3d`, public or private `ConvNd`, arbitrary rank, geometry arrays, asymmetric intrinsic
  padding, padding modes, transposed convolution, causal convolution, depthwise-specific APIs,
  separable convolution, quantized convolution, or another layout such as NWC.
- A neural-network layer, module, parameter, initializer, factory recipe, state, or training API.
- Compiler, autograd, graph verification, Planning, Prepare, Runtime, Engine, backend, lowering,
  fusion, generated code, execution, conformance, or integration implementation.
- Hidden fusion, producer elision, expand/squeeze cancellation, direct construction of a Conv2d
  producer, a synthetic provenance wrapper, or a specialized factory seam.
- Changes to `Conv2dKind`, `Conv2dAttrs`, `TensorConv2dExpressions`, rank-editing operations,
  TensorFactory, Shape/Dimension arithmetic, current convolution-gradient formulas, or operation
  signatures.
- A detailed specification for 0025H, 0026, Compiler 0006B, CPU 0008/0008A, Engine 0004, or NN
  0025.
- Legacy source, package structure, dependency direction, or implementation reuse.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Model capabilities](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Expand-dimensions and squeeze expressions](0017f1-expand-dimensions-and-squeeze-tensor-expressions.md)
- [NCHW Conv2d semantics and Tensor expressions](0020-nchw-conv2d-semantics-and-tensor-expressions.md)
- [Canonical TensorProducer outputs](0025-canonical-tensor-producer-outputs.md)
- [Compiler convolution gradients](../../compiler/tasks/0005d-attention-convolution-pooling-and-loss-gradient-completion.md)

## Architecture constraints

- Work stays in Model and directly affected documentation/planning. Model owns immutable Tensor
  expression metadata; it does not capture graphs, select execution, or depend on downstream
  modules.
- `Tensor` remains the public fluent receiver facade. The helper direction is Tensor to existing
  operation/datatype/shape contracts, preserving acyclic package dependencies.
- The constructed graph is exactly two `EXPAND_DIMS` occurrences, one `CONV2D` occurrence, and one
  `SQUEEZE` occurrence. No code path may replace, conceal, tag, or fuse that structure.
- Each derived Tensor is the canonical wrapper created by its existing factory call. Provenance
  remains per-producer, ordered, immutable, and output-index zero.
- `Conv1dAttrs` is a rank-specific immutable semantic-parameter value. It implements
  `OperationAttrs`, but no `OperationKind` accepts or retains it: construction translates it to a
  fresh `Conv2dAttrs`, which is the exact attributes value on the sole convolution producer.
- Compiler owns capture, deferred Shape proof, gradients, and legal rewrites. Backend Prepare owns
  algorithms, lowering, fusion, allocation, and materialization. Runtime receives only prepared
  work.
- No architecture, module-boundary, dependency, lifecycle, build, or architecture-test rule
  changes.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.datatype`
- `io.github.pho001.synaptik.model.shape`
- `io.github.pho001.synaptik.model.operation`
- `io.github.pho001.synaptik.model.operation.convolution`
- `io.github.pho001.synaptik.model.operation.layout`
- `io.github.pho001.synaptik.model.tensor`

Packages added or changed:

- change `io.github.pho001.synaptik.model.operation.convolution` only by adding the rank-one
  immutable parameter record;
- change `io.github.pho001.synaptik.model.tensor` only by adding the package-private composition
  owner and two receiver overloads;
- add no package and widen no existing type or factory seam.

Type placement:

- `...operation.convolution.Conv1dAttrs` owns only validated one-axis stride, symmetric padding,
  dilation, and group parameters.
- `...tensor.TensorConv1dExpressions` owns rank-specific prevalidation, exact parameter mapping,
  and calls to the existing public Tensor compositions.
- `...tensor.Tensor` exposes the two natural single-output receiver entries, parallel to current
  `conv2d` and `linear` composition entries.
- Existing `Conv2dKind`, `Conv2dAttrs`, `AxisTransformKind`, `AxisTransformAttrs`, and their
  expression owners remain unchanged and own every actual producer.

## Exact semantic and expression contract

### Shapes and geometry

```text
input:   [N, C_in, W]
weight:  [C_out, C_in/groups, K_w]
bias:    [C_out]                              optional
result:  [N, C_out, W_out]

effectiveKernel = dilation * (K_w - 1) + 1
numerator       = W + 2 * padding - effectiveKernel
W_out           = floor(numerator / stride) + 1
```

`K_w` must be statically known and positive. Literal arithmetic is checked in signed `long`.
A static negative numerator fails locally; zero yields one output position. For unresolved `W`,
use the same canonical `DimensionExpressions.addConstant`, `floorDivide`, and `addConstant`
sequence as Conv2d width derivation. The future concrete binding retains the obligation that the
numerator is non-negative.

The result reuses the exact input batch Dimension and weight output-channel Dimension references.
Static input and output channels must each be divisible by `groups`. When both sides are static,
`weightChannelsPerGroup * groups == inputChannels` is checked with exact multiplication. Static
bias length must equal static output channels. Relations involving an unresolved Dimension are
deferred exactly as for Conv2d and remain visible through the expanded descriptors and mapped
Conv2d attributes.

### Attributes and mapping

`Conv1dAttrs` components are validated in declaration order:

```text
stride must be positive: <value>
padding must be non-negative: <value>
dilation must be positive: <value>
groups must be positive: <value>
```

Every call creates this exact mapping:

```java
new Conv2dAttrs(1, attrs.stride(), 0, attrs.padding(), 1, attrs.dilation(), attrs.groups())
```

The new `Conv2dAttrs` is retained by exact reference on the Conv2d operation. `Conv1dAttrs` is not
retained by a producer because there is no Conv1d operation occurrence.

### Values and numerical policy

Conv1d means grouped NCW cross-correlation. It does not reverse the stored kernel. Output channel
`o` selects its contiguous channel group and sums products across the group's input channels and
kernel width positions. The optional bias for `o` participates once. Out-of-range width positions
are conceptual positive-zero inputs and still participate in ordinary floating multiplication.

Input, weight, and present bias must each be floating. Promotion processes input and weight, then
bias. FLOAT64 output accumulates in FLOAT64; FLOAT32 and BFLOAT16 output accumulate in FLOAT32,
with final BFLOAT16 conversion. NaN, infinity, and signed zero follow the current Conv2d meaning,
including conceptual zero multiplied by infinity. Empty channel contraction begins with positive
zero before optional bias. Empty batch or output-channel axes are valid. Reassociation and fused
multiply-add are allowed, so bitwise or cross-backend-identical rounding is not promised.

### Exact visible provenance

Successful construction occurs in this order:

1. `input.expandDims(2)` produces `[N, C_in, 1, W]` with `EXPAND_DIMS` and
   `AxisTransformAttrs(2)`.
2. `weight.expandDims(2)` produces `[C_out, C_in/groups, 1, K_w]` with a distinct
   `EXPAND_DIMS` and `AxisTransformAttrs(2)`.
3. The biased or unbiased `conv2d` call produces `[N, C_out, 1, W_out]` with mapped
   `Conv2dAttrs` and ordered inputs `[expandedInput, expandedWeight]` or
   `[expandedInput, expandedWeight, bias]`. Bias is not expanded.
4. `conv2dResult.squeeze(2)` produces `[N, C_out, W_out]` with `SQUEEZE` and
   `AxisTransformAttrs(2)`.

The returned Tensor's immediate producer is `SQUEEZE`, not `CONV2D`. Each success consumes exactly
four Tensor IDs and creates four identity-distinct one-output producers and canonical wrappers.
Equivalent calls remain fresh. Final type and gradient eligibility are the Conv2d promoted type
and logical OR over original input, weight, and optional bias. Final label and host storage are
empty, and final layout is unresolved because current Conv2d construction produces unresolved
layout.

### Validation and effects

Each overload null-checks in declaration order: `input`, `weight`, optional `bias`, then `attrs`,
using the parameter name as the `NullPointerException` message. Before the first composition call,
the helper validates in this order:

1. input, weight, then present-bias floating eligibility;
2. input/weight and then present-bias floating promotion;
3. input rank three, weight rank three, then present-bias rank one;
4. positive static kernel width;
5. static input-channel divisibility, output-channel divisibility, grouped weight/input equality,
   then bias/output equality;
6. checked effective-kernel and output-width geometry; and
7. construct the mapped `Conv2dAttrs`.

Task-owned failures use `conv1d` and the rank-one role names:

```text
conv1d input must have a floating data type, but was <dataType>
conv1d weight must have a floating data type, but was <dataType>
conv1d bias must have a floating data type, but was <dataType>
conv1d input rank must be 3: <rank>
conv1d weight rank must be 3: <rank>
conv1d bias rank must be 1: <rank>
conv1d kernel width must be static: <dimension>
conv1d kernel width must be positive: <dimension>
conv1d input channels must be divisible by groups: channels=<dimension>, groups=<groups>
conv1d output channels must be divisible by groups: channels=<dimension>, groups=<groups>
conv1d weight channels per group do not match input channels: weight=<dimension>, groups=<groups>, input=<dimension>
conv1d bias length must match output channels: bias=<dimension>, output=<dimension>
conv1d effective kernel does not fit padded width: input=<dimension>, effectiveKernel=<value>, padding=<value>
```

Every task-owned validation failure occurs before an ID is consumed. After composition begins,
existing rank-editing, Conv2d, factory, and ID-exhaustion effects remain visible: a later delegated
failure may leave earlier successful intermediate IDs consumed. Do not add rollback, hidden
preallocation, direct producer construction, or duplicated rank-editing layout implementation to
make the composition appear atomic.

## Affected files

Expected production source (3):

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/convolution/Conv1dAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorConv1dExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Expected Model tests (16):

- add `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/convolution/Conv1dAttrsTest.java`
- add `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorConv1dExpressionTest.java`
- update the exact Tensor public-surface locks in `TensorTest`, `RecurrentScanExpressionTest`,
  `TensorBatchNormInferenceExpressionTest`, `TensorBinaryArithmeticTest`,
  `TensorDenseCategoricalCrossEntropyWithLogitsExpressionTest`,
  `TensorIndexCategoricalCrossEntropyWithLogitsExpressionTest`, `TensorLayerNormExpressionTest`,
  `TensorLinearExpressionTest`, `TensorMatmulExpressionTest`, `TensorMeanSquaredErrorExpressionTest`,
  `TensorRmsNormExpressionTest`, `TensorScaledDotProductAttentionExpressionTest`,
  `TensorSlicePlacementExpressionTest`, and `TensorSumToShapeExpressionTest`, changing only the
  expected method count from 202 to 204 except for the direct `conv1d` signature/name assertions
  owned by `TensorTest` and `TensorConv1dExpressionTest`.

Expected documentation/planning (8):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`
- `docs/planning/backends/cpu/master-plan.md` — planning-stage-only correction of the one stale
  sentence that called completed CPU 0007F2 Ready; implementation must preserve all other CPU
  evidence and wording.

Review unchanged unless inaccurate: Runtime and Training APIs; Conv2d/window/rank-editing,
operation/signature, TensorFactory/producer/provenance, Shape/Dimension and gradient contracts;
architecture/ADRs/tests; conformance/integration; Gradle; other modules.

## Maximum scope

Exactly 27 paths maximum: three production, sixteen Model test, and eight
documentation/planning paths. No production or test path outside Model may change. The CPU master
plan may contain only the planning-stage `Ready` to `Complete` correction already named above.
Stop for path 28, another type/test/document, an existing-helper modification, architecture,
Gradle, dependency, or cross-module executable work.

This is one cohesive Model capability despite four visible producers: the public rank-one
contract, attributes, prevalidation, exact existing-operation composition, public locks, tests,
and documentation must land together.

## Javadoc and documentation requirements

- Fully document `Conv1dAttrs`, its canonical constructor and `defaults()`, the package-private
  helper, and both Tensor methods. Every parameter, result, nullability rule, failure, and
  arithmetic/ID effect receives the required tags.
- Tensor API explains NCW, Shapes, mapping, groups, formula, symbolic relations, numerical policy,
  exact four-node provenance, and a complete biased grouped metadata example.
- Compile API states that generic capture, current Conv2d inference and Compiler 0005D gradients
  see ordinary expand/Conv2d/squeeze nodes; it makes no fusion, execution, or capability claim.
- Glossary distinguishes Conv1d composition from first-class Conv2d and planned Conv3d while
  reusing cross-correlation, group, dilation, effective-kernel, and conceptual-padding terms.
- Model capabilities distinguishes the current public convenience from a new operation-kind
  inventory entry.
- Synchronize this task and Model master to Complete only after implementation, documentation,
  and validation pass. Keep 0025H and 0026 Draft without detailed specifications.
- Record reasoned no-change conclusions for Runtime/Training APIs, related contracts,
  architecture/tests, conformance/integration, Gradle, and other modules.

## Acceptance criteria

- `Conv1dAttrs` has exactly four record components, implements `OperationAttrs`, validates in the
  specified order/messages, and returns exact defaults.
- Exactly two `conv1d` Tensor receiver overloads exist; public Tensor method count is 204. There is
  no alias, static namespace, nullable-bias form, default-attrs overload, or public ConvNd.
- Static and symbolic Shape derivation, exact Dimension-reference retention, checked geometry,
  group/channel/bias relations, and deferred unresolved relations match this task.
- Every floating ordered width combination promotes correctly; BOOL and integral roles fail.
  Numerical meaning is documented/tested as metadata without Model value evaluation.
- Provenance is exactly `EXPAND_DIMS(input,2)`, `EXPAND_DIMS(weight,2)`, mapped `CONV2D`, then
  `SQUEEZE(2)`, with optional bias direct at Conv2d input two. Four IDs/producers/wrappers,
  freshness, storage/layout/label, requires-grad OR, and delegated-failure effects pass.
- Current Compiler capture/inference/gradient applicability follows only from existing operation
  support. No Compiler inventory, source, test, or capability changes.
- No `CONV1D`, Conv3d, compiler/backend/runtime/NN behavior, generated code, hidden fusion,
  architecture, dependency, or build work is added.
- Focused tests, exactly one final Model suite, Javadoc/docs checks, exact 27 paths/public surface,
  statuses/order/dependencies, no 0025H spec, and whitespace validation pass.
- A separate clean documentation-focused pass finalizes Javadocs/docs and records no-change
  conclusions before completion.

## Tests / validation

Focused development validation:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.convolution.Conv1dAttrsTest --tests io.github.pho001.synaptik.model.tensor.TensorConv1dExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorConv2dExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorRankEditingExpressionTest --tests io.github.pho001.synaptik.model.shape.DimensionExpressionsTest --tests io.github.pho001.synaptik.model.datatype.DataTypePromotionTest --tests io.github.pho001.synaptik.model.tensor.TensorProducerTest --tests io.github.pho001.synaptik.model.tensor.TensorProvenanceTest
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
exact paths/packages/signatures/counts, no `CONV1D` or ConvNd production symbol, task/master/roadmap
status consistency, hard 0025G-before-0025H ordering, and absence of a 0025H task file. Repository-
wide tests are deferred to the next recorded cross-module checkpoint or CI because this task
changes one module without dependency or architecture boundaries.

Generated-code validation is explicitly non-applicable. Model builds immutable metadata and the
ordinary four-node expression only; it generates no Java source, Class-File, executable kernel,
or hot loop. Generated/direct oracle, bytecode inspection, and performance evidence belong to the
later CPU execution tasks.

## Dependencies

- Completed [Model 0020](0020-nchw-conv2d-semantics-and-tensor-expressions.md) supplies grouped
  NCHW cross-correlation, mapped geometry, floating policy, Shape relations, and optional bias.
- Completed [Model 0017F1](0017f1-expand-dimensions-and-squeeze-tensor-expressions.md) supplies the
  exact public singleton-axis producers and canonical wrappers used on both sides of Conv2d.
- Completed [Model 0025](0025-canonical-tensor-producer-outputs.md) supplies canonical producer
  output wrappers and immutable provenance identity.
- Complete Compiler 0004A rank-edit gradients and
  [Compiler 0005D](../../compiler/tasks/0005d-attention-convolution-pooling-and-loss-gradient-completion.md)
  supply the current downstream gradient coverage inherited by the visible composition; this
  task changes neither contract.
- Model 0025F was Complete when implementation began. Task 0025G was then the next unfinished
  Model row and sole Ready frontier, so no out-of-order exception was used.

## Follow-up tasks

- Model 0025H remains Draft without a detailed task specification. It depends on completed 0025G
  and separately owns first-class grouped NCDHW Conv3d because current operations cannot express
  it with graph size independent of spatial extents.
- Compiler 0006B follows Model 0025H for Conv3d forward adoption. Conv1d needs no Compiler task or
  inventory row because it is ordinary existing-operation composition.
- CPU 0008 later implements Conv2d execution; CPU 0008A then validates this visible Conv1d
  composition end to end and adds Conv3d execution. Fusion remains later CPU 0008B–0008E work.
- Engine 0004 and NN 0025 remain separate lifecycle and layer owners after the ordered
  Model/Compiler/CPU prerequisites.
- Asymmetric padding, other layouts, transposed/depthwise-specific APIs, and mixed precision need
  separately justified future tasks.

Do not create another detailed specification during implementation.

## Architecture impact

Expected impact: None.

This task uses existing public Model expressions and adds a bounded public convenience with
rank-specific parameters. It changes no semantic-kind inventory, ownership, package dependency,
graph representation, lifecycle, or executable boundary. If implementation requires any such
change, stop and report the conflict.

## Implementation prompt

```text
You are the clean-context implementation agent for Synaptik Model task 0025G in
/Users/phujka/IdeaProjects/Synaptik. Do not use GSD. Do not commit, stage, or push.

Read AGENTS.md, ARCHITECTURE.md, the focused architecture and documentation rules/profiles,
planning guide/roadmap, Model capabilities/master plan, task 0025G, completed tasks 0017F1, 0020,
0025, and 0025F, and the complete current Conv2d/rank-edit/Tensor/provenance source, tests,
Javadocs, APIs, glossary, and Compiler 0005D boundary before editing.

Implement task 0025G exactly inside 27 paths. Preserve the planning-only CPU 0007F2 status
correction and every other CPU evidence line. Update all exact Tensor method locks 202 -> 204 up
front. Build the literal input-expand, weight-expand, Conv2d, squeeze composition; do not add a
CONV1D kind, direct producer, hidden fusion, ConvNd, compiler/backend/runtime/NN work, or another
detailed task. Stop on architecture uncertainty, scope overflow, another path, existing-helper
change, Gradle, dependency, or cross-module executable work.

Run focused validation and exactly one final Model suite after Java stabilizes. Then hand the
actual diff and exact Java evidence to a separate clean documentation-focused agent in the same
overall change. That agent must independently finalize affected Javadocs, Tensor/Compile APIs,
glossary, capabilities, and planning; record reasoned no-change conclusions; reuse successful
Java evidence unless executable code changes; and run final Javadoc/documentation checks.

Keep 0025G Ready until every implementation, documentation, and validation criterion passes.
Then mark the task/master/roadmap entry Complete while 0025H and 0026 remain Draft without task
files. Update validation evidence, implementation notes, and completion summary before reporting
Status: Complete.
```

## Local decisions

- `Tensor.conv1d` is selected because this is a natural single-output algebraic composition with
  the receiver as input, matching `conv2d` and `linear`. A domain namespace is unnecessary.
- Bias presence uses two overloads. There is no nullable bias, bias flag, default-attributes
  overload, or alias.
- Axis `2` is the inserted and removed singleton height axis. Inserting at axis `3` would model a
  singleton width and move convolution over the wrong spatial axis.
- `Conv1dAttrs` remains rank-specific and implements the semantic-parameter marker, but the actual
  producer retains only a freshly mapped `Conv2dAttrs`. This makes inspection truthful: there is
  no Conv1d operation whose attributes could retain the convenience value.
- Rank-specific validation occurs before composition so callers receive `conv1d` diagnostics and
  task-owned failures consume no ID. Once the first existing operation succeeds, ordinary
  delegated partial-ID effects remain visible rather than being hidden by an atomic facade.
- Current Compiler rank-edit and Conv2d inference/gradient rules apply structurally without
  knowing the `conv1d` Java method name. No Compiler change or test is warranted.
- Symmetric padding is intrinsic. Asymmetric padding remains explicit `PAD` composition rather
  than a silent change to the completed Conv2d contract.

## Known limitations

- Kernel width must be statically positive; only input width and channel/bias relations may remain
  symbolic under the current Shape contract.
- Construction records meaning and provenance only. It does not evaluate values, bind unresolved
  extents, lower, prepare, allocate storage, choose an algorithm, fuse, or execute.
- Execution remains unavailable until the ordered Conv2d CPU route is implemented; no current
  backend capability follows from this Model API.
- Final layout is unresolved even when original operands have resolved layouts because the
  existing Conv2d result layout is unresolved.
- The visible four-node composition may allocate earlier intermediate IDs before a later
  delegated layout, factory, or ID-space failure. No rollback is promised.

## Validation evidence

The implementation context supplied the final executable evidence. Its focused task command
passed, and its single final `./gradlew :modules:model:test` run passed 1,055 tests with zero
failures, errors, or skips. No executable Java changed afterward: the documentation context added
only Javadoc and Markdown. The documentation pass therefore reused those results as required by
the planning and documentation rules and did not rerun Java tests.

The mandatory separate clean documentation-focused context was `/root`. It applied the General,
API/Javadoc, Planning, and Example profiles and independently reviewed the complete implementation
diff, all three production paths, all sixteen Model test paths, rendered affected public Javadoc,
Tensor and Compile APIs, glossary, Model capabilities, planning status, completed Model
0017F1/0020/0025/0025F contracts, and the relevant Compiler inference and convolution-gradient
boundaries.

- `./gradlew :modules:model:javadoc` passed after the final Javadoc edits. Inspection of generated
  `Conv1dAttrs.html` and `Tensor.html` confirmed complete constructor, parameter, return, failure,
  Shape, provenance, numerical-policy, and identifier-effect contracts for the record and exactly
  two public receiver overloads. The package-private helper is intentionally absent from public
  generated Javadoc; its source Javadoc was reviewed in full.
- The documented Java 26 biased grouped example compiled and ran against Model classes. It printed
  `Shape[2, 6, 4]`, `FLOAT64`, `SQUEEZE`, `CONV2D`, and expanded weight
  `Shape[6, 2, 1, 3]`, confirming the documented metadata and visible producer boundary.
- `javap -public` showed exactly the two required `conv1d` methods and exactly 204 public Tensor
  methods. `javap -p`, source/reflection tests, import scans, and manual provenance review confirmed
  four record components, a final package-private field-free helper, exact two overloads, the two
  distinct axis-2 expansions, mapped `Conv2dAttrs(1, stride, 0, padding, 1, dilation, groups)`,
  direct optional bias, and axis-2 squeeze. Production/test scans found no `CONV1D` or `ConvNd`.
- A read-only documentation check validated 893 local Markdown links including target anchors,
  balanced fences, final newlines, no carriage returns, and no trailing whitespace across the
  eight documentation/planning paths. `git diff --check` passed.
- Combined tracked/untracked scope inspection found exactly 27 permitted paths: three production,
  sixteen Model tests, and eight documentation/planning paths. The staging area is empty. No
  0025H specification exists. Task/master/roadmap mark 0025G Complete; 0025H and 0026 remain Draft
  without detailed specifications. The CPU master-plan diff remains the single requested 0007F2
  `Ready` to `Complete` correction and preserves every other CPU evidence line.
- Final independent diff review found no stale execution, fusion, backend, or performance claim,
  no architecture or dependency change, and no work outside task 0025G.

## Implementation notes

- Added immutable `Conv1dAttrs` and a bounded `TensorConv1dExpressions` construction owner. The
  two Tensor receiver methods validate the NCW contract and visibly construct
  `EXPAND_DIMS(2) -> EXPAND_DIMS(2) -> CONV2D -> SQUEEZE(2)` with optional direct bias.
- The input becomes `[N, C_in, 1, W]`; rank-three weight becomes
  `[C_out, C_in/groups, 1, K_w]`; `Conv1dAttrs` maps to
  `Conv2dAttrs(1, stride, 0, padding, 1, dilation, groups)`. Floating promotion and numerical
  meaning are inherited from Conv2d, while the final Tensor is the canonical squeeze output.
- Focused tests cover attributes, API shape, static/symbolic dimensions, grouping, bias,
  promotion, failures and ID effects, exact provenance, canonical wrappers, and freshness. The
  fourteen existing exact Tensor method-count locks now expect 204.
- Tensor API now contains the complete current NCW contract and biased grouped metadata example.
  Compile API explains that generic capture, inference, and gradients reuse ordinary rank-edit and
  Conv2d nodes. The glossary distinguishes the composition from first-class Conv2d and future
  Conv3d, and Model capabilities records no new operation-kind inventory entry.
- Runtime and Training APIs remain accurate unchanged because this task adds no runtime or
  training surface. Existing Conv2d, window, rank-editing, operation/signature, Tensor factory and
  provenance, Shape/Dimension, and gradient contracts remain accurate because the new API composes
  them without changing their meanings. Architecture/ADRs/tests, conformance/integration, Gradle,
  dependencies, and other modules require no change because no boundary, build, executable, or
  cross-module behavior changed.

## Completion summary

- Completed changes: Added the exact NCW Conv1d convenience, immutable geometry, local validation,
  four-producer composition, public locks/tests, complete Javadocs and explanatory documentation,
  and synchronized completion status.
- Files changed or created: Exactly 27 permitted paths: three production, sixteen Model test, and
  eight documentation/planning paths.
- Tests and validation: Reused the implementation context's passing focused command and final
  1,055-test Model suite; final Model Javadoc, rendered inspection, Java 26 example,
  reflection/`javap`/import/manual structural checks, 893 local links/anchors, fences/newlines/
  whitespace, exact scope/status/staging checks, and `git diff --check` passed.
- Documentation-agent review: Clean documentation context `/root` independently finalized all
  affected Javadocs, APIs, glossary/capabilities impact, planning evidence, and status.
- Documentation impact: Tensor and Compile APIs, glossary, capabilities, task, master plan, and
  roadmap now explain the current composition and its boundaries; the sole CPU status correction
  is preserved.
- Javadoc review: `Conv1dAttrs`, its constructor/defaults, the helper and all helper methods, and
  both Tensor methods have complete accurate contracts; directly related Javadocs remain accurate
  unchanged.
- Glossary impact: Added `Conv1d composition` to distinguish the rank-specific convenience from
  first-class Conv2d and planned Conv3d while reusing existing convolution terminology.
- Unresolved issues: None within task scope.
- Follow-up required: None for 0025G. Model 0025H and 0026 remain Draft without specifications.

Status: Complete
