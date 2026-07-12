# Task 0020: NCHW Conv2d Semantics and Tensor Expressions

## Status

Complete

## Goal

Add one first-class backend-independent `CONV2D` model operation and exactly two public receiver
expressions for grouped NCHW two-dimensional convolution, with and without bias. The result must
carry truthful static or symbolic spatial extents, promoted floating type, exact ordered
provenance, and the complete selected numerical meaning without performing value work.

This focused task replaces only the convolution portion of the former broad 0020 frontier. Max
and average pooling remain Draft follow-up 0020A because their literal window, padding-value,
average-divisor, ceil-mode, empty, and all-padding-window decisions are distinct.

## Scope

- Add `Conv2dKind.CONV2D` with input range two through three and exactly one output.
- Add immutable public `Conv2dAttrs(strideHeight, strideWidth, paddingHeight, paddingWidth,
  dilationHeight, dilationWidth, groups)` using `long` components.
- Add `Conv2dAttrs.defaults()` returning stride one, zero symmetric padding, dilation one, and one
  group.
- Add exactly these receiver methods:

  ```java
  public Tensor conv2d(Tensor weight, Conv2dAttrs attrs)
  public Tensor conv2d(Tensor weight, Tensor bias, Conv2dAttrs attrs)
  ```

- Use one package-private stateless construction helper and one final `TensorFactory.createDerived`
  call for each successful expression.
- Require rank-four NCHW input `[N, C_in, H, W]` and rank-four weight
  `[C_out, C_in/groups, K_h, K_w]`; optional bias is rank one `[C_out]`.
- Require floating input, weight, and present bias. Promote input and weight with
  `DataTypePromotion.promoteFloating`, then promote present bias with the intermediate result.
- Require statically known, positive weight kernel height and width. Preserve exact input batch
  and weight output-channel Dimension references in result `[N, C_out, H_out, W_out]`.
- Validate locally provable group/channel and bias/channel relations; retain unresolved relations
  as compiler/binding obligations represented by exact input descriptors and attributes.
- Derive static and dynamic spatial output Dimensions with the exact floor-mode policy below.
- Produce one fresh unlabeled, storage-free result with unresolved layout, gradient-request OR
  across all actual inputs, and output index zero from exact ordered inputs `[input, weight]` or
  `[input, weight, bias]`.
- Document cross-correlation, padding, promotion, accumulation, empty, special-value,
  determinism, Shape, metadata, and lifecycle boundaries.
- Update focused API/glossary/planning documentation and public-Tensor inventory locks.

## Out of scope

- max pooling, average pooling, global/adaptive pooling, ceil-mode convolution, transposed
  convolution, depthwise convenience, causal/one-dimensional/three-dimensional convolution, or
  channels-last layouts
- asymmetric or automatic `same`/`valid` padding, padding Tensor inputs, configurable padding
  values, runtime strides/dilations/groups, or a broad options framework
- integral, BOOL, quantized, sparse, complex, unsigned, or FLOAT16 inputs; implicit casts, output
  type overrides, or accumulator options
- dynamic kernel extents: the current positive-coefficient Dimension expression model cannot
  truthfully subtract an unresolved dilated kernel from an unresolved input extent
- value reads, eager evaluation, host allocation, result storage, resolved layout, mutation, or
  input materialization
- gradients, adjoints, backward-operation kinds, saved values, trainable-parameter ownership,
  compiler capture, graph-wide constraint solving, decomposition, fusion, or optimization
- algorithms such as im2col/GEMM/direct/Winograd/FFT, kernel selection, backend capabilities,
  lowering, prepare, runtime, execution, conformance, or integration
- changes to `Window2dAttrs`, unfold/fold, pad, MATMUL, attention, `ShapeBroadcast`,
  `DimensionExpression`, `DataTypePromotion`, Tensor provenance/factory seams, architecture,
  dependencies, Gradle, or another module

## Public and operation contracts

### Attributes and signature

`Conv2dAttrs` validates components in declaration order. Strides, dilations, and groups are
positive; padding components are non-negative. Exact constructor failures are:

```text
strideHeight must be positive: <value>
strideWidth must be positive: <value>
paddingHeight must be non-negative: <value>
paddingWidth must be non-negative: <value>
dilationHeight must be positive: <value>
dilationWidth must be positive: <value>
groups must be positive: <value>
```

The record owns intrinsic convolution geometry only. Kernel extents come from weight Shape, and
bias presence comes from the operation occurrence's input count. It owns no Tensor, Shape,
DataType, layout, storage, gradient, compiler, backend, or execution state. `CONV2D` accepts two
or three inputs and exactly one output; no second kind or bias flag is added.

### Shape and dynamic extents

For each spatial axis, let input extent be `D`, static positive weight kernel be `k`, symmetric
padding per side be `p`, positive dilation be `d`, and positive stride be `s`:

```text
effectiveKernel = d * (k - 1) + 1
numerator       = D + 2 * p - effectiveKernel
output          = floor(numerator / s) + 1
```

All literal arithmetic uses checked `long` operations. For static `D`, reject negative numerator
before division; zero numerator yields one output position. For unresolved `D`, construct the
canonical equivalent through existing `DimensionExpressions.addConstant`, `floorDivide`, and
`addConstant` calls. Do not require a static spatial input, create an identity-based unknown, or
change the symbolic-expression model. The unresolved result retains the binding obligation that
the numerator is non-negative. Neutral arithmetic must preserve exact Dimension references when
the existing canonicalization contract does so.

Input batch `N` and weight output-channel `C_out` are retained by exact reference. Kernel
Dimensions must be static and positive. Static input channels must be divisible by `groups`;
static output channels must be divisible by `groups`; and when both input channels and weight
channels-per-group are static, `weightChannelsPerGroup * groups == inputChannels` must hold using
checked multiplication. When both bias length and output channels are static they must match.
Any corresponding relation involving an unresolved Dimension is deferred; the helper must not
invent a result constraint or reject a truthful result whose exact selected Dimensions are known.

Exact task-owned Shape failures are:

```text
conv2d input rank must be 4: <rank>
conv2d weight rank must be 4: <rank>
conv2d bias rank must be 1: <rank>
conv2d kernel height must be static: <dimension>
conv2d kernel width must be static: <dimension>
conv2d kernel height must be positive: <dimension>
conv2d kernel width must be positive: <dimension>
conv2d input channels must be divisible by groups: channels=<dimension>, groups=<groups>
conv2d output channels must be divisible by groups: channels=<dimension>, groups=<groups>
conv2d weight channels per group do not match input channels: weight=<dimension>, groups=<groups>, input=<dimension>
conv2d bias length must match output channels: bias=<dimension>, output=<dimension>
conv2d effective kernel does not fit padded height: input=<dimension>, effectiveKernel=<value>, padding=<value>
conv2d effective kernel does not fit padded width: input=<dimension>, effectiveKernel=<value>, padding=<value>
```

### Data type, values, and numerical policy

Input, weight, and present bias must each be floating. Role-specific failures are:

```text
conv2d input must have a floating data type, but was <dataType>
conv2d weight must have a floating data type, but was <dataType>
conv2d bias must have a floating data type, but was <dataType>
```

`CONV2D` means NCHW grouped cross-correlation, not mathematical kernel reversal. For output
channel `o`, its group selects the corresponding contiguous `C_in/groups` input channels. Each
output element is the optional bias for `o` plus the sum of pairwise input/weight products over
that group and kernel height/width in increasing logical index meaning. Out-of-range spatial
coordinates are conceptual positive-zero input values and participate in ordinary floating
multiplication; no value is read outside the input.

The promoted type is the output type. FLOAT64 output accumulates in FLOAT64. FLOAT32 and BFLOAT16
output accumulate in FLOAT32, with final conversion for BFLOAT16. Bias participates in that
accumulation domain exactly once. Reassociation and fused multiply-add are allowed, so fixed
traversal order, bitwise equality, and cross-backend identical rounding are not promised; later
conformance owns tolerances. IEEE-754 NaN, infinity, and signed-zero behavior follows ordinary
multiplication/addition, including conceptual zero multiplied by an infinite weight. An empty
channel contraction yields positive zero before optional bias. Empty batch or output-channel
axes are valid; zero output spatial extent is not produced by valid static geometry under this
formula, while unresolved geometry is checked when bound.

### Validation and construction order

The helper validates before its sole factory call in this order:

1. null-check input, weight, present bias, and attrs in parameter order;
2. validate input, weight, then present-bias floating eligibility;
3. promote input/weight and then present bias;
4. validate input rank, weight rank, then present bias rank;
5. require static positive kernel height, then width;
6. validate static input-channel divisibility, output-channel divisibility, grouped weight/input
   compatibility, then bias/output compatibility;
7. derive height then width with checked geometry and existing Dimension expressions;
8. create the descriptor and exact operation;
9. delegate once with exact ordered inputs.

Null messages are parameter names. `DataTypePromotion` and checked-arithmetic failures retain
their existing messages. Every failure before final delegation consumes no Tensor ID, producer,
or result wrapper. A successful call consumes exactly one ID and creates one producer.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Capabilities](../capabilities.md)
- [Master plan](../master-plan.md)
- [Signature hardening](0018k-operation-signature-and-construction-hardening.md)
- [Symbolic extents](0018m-symbolic-extent-expressions.md)
- [Dynamic extent adoption](0018m1-dynamic-extent-adoption.md)
- [Typed scalar](0018n-typed-scalar-value-contract.md)
- [Multi-axis/statistical reductions](0018v-multi-axis-and-statistical-reductions.md)
- [MATMUL](0019-matmul-semantics-and-tensor-expression.md)
- [Attention](0019e-scaled-dot-product-attention.md)

## Architecture constraints

- Work stays in model plus its documentation/planning. Tensor remains public mutable state, not
  graph IR.
- Convolution operation types record backend-independent meaning only; operation packages do not
  import Tensor, graph, compiler, runtime, prepare, or backend types.
- Direction is tensor helper to convolution operation/datatype/shape. Packages remain acyclic.
- Compiler owns capture, binding/deferred proof, legal decomposition, gradients, adjoints, and
  saved values. Backend prepare owns conforming algorithms, lowering, kernels, and
  materialization. Runtime receives prepared work without original operations on its hot path.
- No architecture, dependency, lifecycle, focused-architecture, Gradle, or cross-module change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.datatype`
- `io.github.pho001.synaptik.model.shape`
- `io.github.pho001.synaptik.model.operation`
- `io.github.pho001.synaptik.model.tensor`

Package added:

- `io.github.pho001.synaptik.model.operation.convolution` — public convolution identity and
  intrinsic immutable attributes only.

Type placement:

- `...operation.convolution.Conv2dKind` — exact operation identity/signature owner.
- `...operation.convolution.Conv2dAttrs` — public inspectable intrinsic geometry/group semantics.
- `...tensor.TensorConv2dExpressions` — package-private validation, Shape, descriptor, and
  provenance construction owner.
- `...tensor.Tensor` — established public fluent receiver facade.

Tests mirror production packages where package-private access is required.

## Affected files

Production (4):

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/convolution/Conv2dKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/convolution/Conv2dAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorConv2dExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests (7):

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/convolution/Conv2dSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorConv2dExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java` — exact two
  signatures and public count 175 to 177.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
  — count only, 175 to 177.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
  — count only, 175 to 177.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLinearExpressionTest.java`
  — count only, 175 to 177.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressionTest.java`
  — count only, 175 to 177.

Documentation/planning (7):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless inaccurate: Runtime/Training APIs; Shape/Dimension expression,
layout/window/pad, MATMUL/attention, operation/signature, Tensor/provenance contracts;
architecture/ADRs/tests; conformance/integration; Gradle; other modules.

## Maximum scope

Exactly 18 paths maximum: four production, seven test, and seven documentation/planning paths.
`Tensor.java` changes only one import, two methods/Javadocs, and its operation inventory. Five
existing tests change only the stated signatures/counts. Stop for path 19, another type/test/doc,
an existing-helper change, architecture, Gradle, or cross-module work.

This is one cohesive vertical model capability inside the planning guide guardrail. Kind, attrs,
public construction, symbolic Shape derivation, promotion/numerical meaning, API locks, and docs
must agree in one compilable state. Pooling would exceed the guardrail and introduces independent
semantic decisions, so it remains 0020A Draft.

## Javadoc and documentation requirements

- Fully document kind, attrs/defaults, helper, and both Tensor methods: NCHW/cross-correlation
  meaning, input/weight/bias Shape, dynamic extent formula, group rules, type/accumulation,
  padding/special/empty policy, metadata/provenance, freshness, validation/failures, and lifecycle
  boundaries.
- Every parameter, result, and expected failure has complete `@param`, `@return`, and `@throws`
  tags as applicable.
- Tensor API gets a Shape/attribute table, grouped example, one dynamic-height example, numerical
  policy, and current-model versus planned compiler/execution boundary.
- Compile API records current convolution expression metadata and future compiler-owned binding,
  capture, decomposition, and gradients without claiming compiler support.
- Review glossary terms NCHW, convolution, cross-correlation, kernel, dilation, group, and
  effective kernel; add or refine only terms needed by the public explanation.
- Synchronize capabilities/task/master/roadmap: keep 0020 Ready during implementation, then mark
  it Complete only after all criteria pass; keep 0020A and 0021–0024 Draft, exactly one detailed
  post-0019E task, and no Ready model frontier after completion.
- Record reasoned no-change conclusions for Runtime/Training APIs, related contracts,
  architecture, conformance/integration, Gradle, and other modules.

## Acceptance criteria

- Exact kind, attrs/defaults, two-to-three-input/one-output signature, and two receiver methods
  exist; public Tensor method count is 177.
- Static/dynamic Shape derivation, exact Dimension reference retention, checked geometry,
  grouped-channel/bias constraints, and deferred unresolved obligations match this task.
- All floating ordered width combinations promote correctly; BOOL/integral roles fail; selected
  accumulation and conversion policies are documented and tested as semantic metadata without
  evaluation.
- NCHW grouped cross-correlation, conceptual padding, NaN/infinity/signed-zero, empty contraction,
  reassociation/tolerance/determinism policies are explicit.
- Validation order, exact task-owned failures, no-ID failures, one-ID success, freshness,
  storage/layout/label, gradient-request OR, and exact ordered provenance pass.
- No pooling, gradients, compiler, algorithm, backend/runtime, architecture, dependency, or build
  work is added.
- Focused tests, exactly one final model suite, Javadoc/docs/link/anchor/fence/newline/whitespace,
  exact 18 paths/packages/public surface/statuses, and `git diff --check` pass.
- A separate clean documentation-focused pass reuses Java evidence, finalizes Javadocs/docs, and
  records reasoned no-change conclusions before completion.

## Tests / validation

Focused development validation:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.convolution.Conv2dSemanticsTest --tests io.github.pho001.synaptik.model.tensor.TensorConv2dExpressionTest --tests io.github.pho001.synaptik.model.shape.DimensionExpressionsTest --tests io.github.pho001.synaptik.model.datatype.DataTypePromotionTest --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest --tests io.github.pho001.synaptik.model.tensor.TensorProducerTest --tests io.github.pho001.synaptik.model.tensor.TensorProvenanceTest
```

After executable Java stabilizes, exactly once:

```bash
./gradlew :modules:model:test
```

The focused tests cover attributes/signature, overloads, all static/dynamic spatial cases,
kernel/group/channel/bias constraints, floating promotion, validation/ID effects,
metadata/provenance/freshness, and numerical contracts as meaning/Javadoc without value execution.

Documentation pass after final Javadoc:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

Also validate local Markdown links/anchors, fences, terminology, examples, generated Javadoc,
newlines/whitespace, exact paths/packages, public count/signatures, the required completion
frontier state, and no 0020A-or-later detailed spec. Repository validation is deferred to the selected-modern-
operations checkpoint after 0022 or CI because no repository-wide contract changes.

## Dependencies

- 0001–0002 and 0018M–0018M1: DataType, Shape, Dimensions, canonical symbolic arithmetic.
- 0005–0007, 0011–0013, and 0018K–0018L: operation/signature,
  Tensor/descriptor/factory/provenance.
- 0018N: exact typed scalar and conceptual positive-zero vocabulary precedent.
- 0018V: final selected reduction/accumulation policy precedent.
- 0019: floating product-sum promotion, accumulation, special-value, and determinism precedent.

Tasks 0017M–0017N and 0019E are table-order/history references, not technical dependencies.

## Follow-up tasks

- 0020A remains Draft for cohesive NCHW max/average Pool2d semantics and expressions. It must
  select literal window/dilation scope, padding value, average divisor, ceil-mode terminal-window,
  empty, all-padding-window, and max tie/special-value contracts before becoming Ready.
- Compiler later owns capture, binding/constraints, legal decomposition, gradients, adjoints, and
  saved values.
- Backend/conformance/runtime/integration later own algorithms, tolerances, lowering, kernels,
  storage, prepared execution, and numerical evidence.

Do not create another detailed specification during implementation.

## Architecture impact

Expected impact: None.

If this task requires architecture, dependency, lifecycle, focused-architecture, Gradle,
cross-module, or scope changes, stop and report the issue.

## Implementation prompt

```text
Work in Synaptik without commit or push. Read AGENTS.md, ARCHITECTURE.md, focused architecture,
documentation/planning rules and profiles, roadmap, model capabilities/master, completed
operation/signature/Tensor/provenance/symbolic-extent/typed-scalar/reduction/MATMUL/attention and
layout-window tasks, current related source/tests/APIs/glossary, and task 0020.

Implement task 0020 exactly inside 18 paths. Update every global public Tensor inventory/count
175 -> 177 up front. Preserve all contracts and stop on architecture uncertainty, scope overflow,
another type/test/document, existing-helper change, Gradle, or cross-module work.

Run focused validation and exactly one final model suite after Java stabilizes. Hand the actual
diff and Java evidence to a separate clean documentation-focused agent in the same change; it
finalizes Javadocs, Tensor/Compile APIs, glossary, capabilities/planning and docs checks while
reusing Java evidence. Keep 0020 Ready until all criteria pass, then mark it Complete while 0020A+
remain Draft with no next Ready model task.
```

## Documentation-agent handoff

Provide this task, the complete diff, exact focused/final Java evidence and post-test Java-change
state, API/Shape/type/group/bias/numerical/provenance policies, seven documentation paths, and
validation requirements. The clean agent reads repository instructions, architecture, rules and
General/API-Javadoc/Planning/Example profiles, task, source/tests/generated Javadoc,
Tensor/Compile/Runtime/Training APIs, glossary/planning, and directly related contracts. It
finalizes documentation and records reasoned no-change conclusions without repeating successful
Java tests absent executable change, stale evidence, or a concrete risk.

## Local decisions

- One `CONV2D` kind with a two-to-three-input signature represents bias presence by occurrence
  input count; no second kind or bias flag was added.
- Grouped convolution means NCHW cross-correlation with contiguous channel groups and no kernel
  reversal. Kernel extents come from weight Shape and remain statically positive because current
  symbolic extents cannot truthfully represent subtraction of an unresolved dilated kernel.
- Static relations fail locally in the specified order. Relations involving an unresolved channel,
  bias, or spatial input Dimension remain compiler/binding obligations retained by exact
  descriptors and attributes.
- Conceptual padding is positive zero and participates in ordinary multiplication, including with
  infinity. The selected promotion, accumulation, empty-contraction, special-value,
  reassociation, and determinism policies match the task contract without claiming evaluation.
- The documentation pass used General, API/Javadoc, Planning, and Example profiles. It added one
  glossary entry for the reusable convolution/cross-correlation/NCHW/group/dilation/effective-
  kernel distinction and refined `Kernel` to distinguish a weight kernel from an executable
  backend kernel.
- Runtime and Training API pages remain unchanged because this task adds model expression metadata
  only: runtime receives prepared work, while training/compiler later own gradients and graph
  expansion. Related Shape/Dimension, layout/window/pad, MATMUL/attention,
  operation/signature/Tensor/provenance contracts remain accurate and unchanged because the new
  helper composes them without changing their behavior.
- Architecture, focused architecture, ADRs, architecture tests, backend conformance, integration
  tests, Gradle, dependencies, other modules, and task 0020A-or-later specifications remain
  unchanged because no boundary, executable behavior, build contract, or downstream detailed
  design changed.

## Known limitations

- Dynamic kernel height and width are rejected; only input spatial extents may remain symbolic.
- Unresolved group/channel, bias/output-channel, and spatial-fit relations are recorded only by
  exact descriptors and attributes; no current compiler or binding API proves them.
- Current construction records numerical meaning but does not read values, evaluate convolution,
  construct gradients, capture or decompose a graph, choose an algorithm/backend, allocate
  storage, lower, prepare, or execute.
- Reassociation and fused multiply-add are permitted, so fixed traversal order, bitwise equality,
  and identical cross-backend rounding are deliberately not promised.
- Pooling remains Draft task 0020A. Repository-wide validation remains deferred to the selected-
  modern-operations checkpoint after 0022 or CI as planned.

## Validation evidence

- Implementation context `/root/task_0020_implementation` ran the exact focused command. Its
  first attempt exposed a compile-test failure, which was corrected before final evidence. The
  final focused run passed `BUILD SUCCESSFUL`: 7 suites, 41 tests, 0 failures, 0 errors, 0 skips.
- After executable Java stabilized, that implementation context ran exactly one final
  `./gradlew :modules:model:test`; it passed `BUILD SUCCESSFUL`: 109 suites, 860 tests, 0 failures,
  0 errors, 0 skips. Executable Java did not change afterward. Clean documentation context
  `/root/task_0020_implementation/task_0020_docs` reused this evidence and did not repeat either
  Java suite.
- The documentation context ran `./gradlew :modules:model:javadoc` after final Javadoc; it passed
  `BUILD SUCCESSFUL` with two actionable tasks, one executed and one up-to-date. Generated pages
  for `Conv2dKind`, `Conv2dAttrs`, and both `Tensor.conv2d` overloads contain the reviewed
  contracts.
- The complete grouped metadata example was compiled with Java against
  `modules/model/build/classes/java/main` and run. It printed the documented FLOAT64 type,
  `Shape[2, 6, 7, 7]`, true gradient eligibility, true unresolved-layout/storage-free check,
  `CONV2D`, and true exact ordered-input check.
- A targeted local Markdown checker passed 580 links including 157 heading anchors across the
  seven documentation/planning paths. Fence balance, terminology, examples, non-empty files,
  final newlines, and generated-Javadoc content checks passed.
- The final path audit found exactly 18 authorized paths: 4 production, 7 tests, and 7
  documentation/planning paths. Production packages, the exact kind/attrs/defaults surface, the
  two receiver signatures, and the public Tensor method count of 177 were verified from compiled
  classes. Task 0020 is Complete; 0020A and 0021–0024 are Draft; no model task is Ready; task 0020
  is the only detailed post-0019E specification; and no 0020A-or-later detailed spec exists.
- `git diff --check` passed on the final combined change.

## Implementation notes

- Added public `Conv2dKind.CONV2D`, immutable `Conv2dAttrs`, one package-private stateless
  `TensorConv2dExpressions` helper, and exactly two public `Tensor.conv2d` receiver methods.
- Successful construction performs one final `TensorFactory.createDerived` call and creates one
  fresh output. It preserves exact batch/output-channel Dimensions, derives checked static or
  canonical symbolic spatial Dimensions, applies ordered floating promotion, propagates the
  actual-input gradient-request OR, and records exact ordered provenance at output index zero.
- Focused tests cover surface/signature, attribute validation, static/dynamic Shape cases,
  deferred obligations, promotion and role failures, exact task-owned diagnostics, ID effects,
  checked arithmetic, freshness, metadata, and provenance. Existing Tensor inventory locks now
  expect 177 methods.
- The independent documentation pass finalized the four affected production Javadocs plus Tensor
  API, Compile API, glossary, capabilities, task, master plan, and roadmap. It changed no
  executable Java behavior.

## Completion summary

- Completed grouped NCHW Conv2d semantic identity, intrinsic immutable attributes, exact
  static/symbolic result Shape derivation, two public biased/unbiased expression methods,
  validation, promotion/numerical policy, fresh metadata/provenance construction, focused tests,
  public-surface locks, Javadocs, API/glossary documentation, and planning synchronization.
- Changed exactly the 18 paths listed by this task; no extra type, test, document, module,
  architecture, dependency, Gradle, runtime, training, backend, conformance, or integration change
  was introduced.
- Focused 41-test and final 860-test model evidence passed in the implementation context. The
  separate documentation context passed model Javadoc, the compiled example, 580-link/157-anchor
  Markdown validation, generated-content/fence/newline/scope/surface/status checks, and final
  whitespace validation.
- Unresolved issues: none. Required follow-up: none for task 0020; Draft task 0020A separately owns
  pooling decisions, and later compiler/backend/runtime tasks own their documented boundaries.

Status: Complete
