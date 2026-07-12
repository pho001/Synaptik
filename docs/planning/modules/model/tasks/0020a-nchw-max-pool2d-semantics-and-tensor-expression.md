# Task 0020A: NCHW Max Pool2d Semantics and Tensor Expression

## Status

Complete

## Goal

Add one first-class backend-independent `MAX_POOL2D` operation and exactly one public receiver
expression for floating rank-four NCHW tensors. The result must retain exact batch/channel
dimensions, derive truthful static or symbolic spatial dimensions, and record the selected
padding, ceil-mode, extrema, metadata, and provenance semantics without evaluating values.

This task is the max-pooling half of the former combined 0020A row. Average pooling is Draft
follow-up 0020A1 because its divisor, padding-count, accumulation, and invalid-divisor policies
are independent of max ordering, NaN, tie, signed-zero, and empty-window semantics.

## Scope

- Add `Pool2dKind.MAX_POOL2D` with exactly one input and one output.
- Add immutable public `MaxPool2dAttrs(kernelHeight, kernelWidth, strideHeight, strideWidth,
  paddingHeight, paddingWidth, dilationHeight, dilationWidth, ceilMode)` using `long` geometry.
- Add exactly `public Tensor maxPool2d(MaxPool2dAttrs attrs)`.
- Add one package-private stateless construction helper and one final
  `TensorFactory.createDerived` call for each successful expression.
- Require floating rank-four NCHW input `[N, C, H, W]` and preserve exact input type, batch,
  channel, and gradient-request metadata.
- Derive floor- or ceil-mode static and symbolic spatial extents with checked arithmetic.
- Treat padding positions as excluded from the maximum; an all-padding window returns negative
  infinity in the input type.
- Select exact NaN, infinity, signed-zero, and equal-value tie semantics.
- Produce one fresh unlabeled, storage-free result with unresolved layout and exact one-input,
  output-index-zero provenance.
- Update focused API, glossary, capability, and planning documentation plus public-Tensor
  inventory locks.

## Out of scope

- average, global, adaptive, one-dimensional, three-dimensional, or channels-last pooling
- max-pool indices, multiple outputs, unpooling, configurable padding values, asymmetric or
  automatic padding, runtime geometry, or a broad options framework
- integral, BOOL, quantized, sparse, complex, unsigned, or FLOAT16 input; casts, promotion,
  output-type overrides, or accumulation options
- PyTorch/ONNX terminal-window removal in ceil mode; this task selects the literal symmetric
  padded window grid defined below so symbolic output extents remain exact
- reuse or modification of `Window2dAttrs`; it owns unfold/fold layout-transform geometry and
  conceptual zero padding, not pooling extrema semantics
- value reads, eager evaluation, host allocation, result storage, resolved layout, mutation, or
  input materialization
- gradients, adjoints, saved indices, backward kinds, compiler capture, graph-wide constraint
  solving, canonicalization, decomposition, fusion, or optimization
- algorithms, backend capabilities, lowering, prepare, runtime, execution, conformance, or
  integration
- changes to Conv2d, unfold/fold, pad, aggregate reductions, symbolic-expression foundations,
  promotion, provenance/factory seams, architecture, dependencies, Gradle, or another module

## Public and operation contracts

### Attributes and signature

`MaxPool2dAttrs` validates components in declaration order. Kernel, stride, and dilation
components are positive; padding components are non-negative. Exact constructor failures are:

```text
kernelHeight must be positive: <value>
kernelWidth must be positive: <value>
strideHeight must be positive: <value>
strideWidth must be positive: <value>
paddingHeight must be non-negative: <value>
paddingWidth must be non-negative: <value>
dilationHeight must be positive: <value>
dilationWidth must be positive: <value>
```

`ceilMode` is retained unchanged. The record owns intrinsic max-pooling geometry only and owns no
Tensor, Shape, DataType, layout, storage, gradient, compiler, backend, or execution state.
`MAX_POOL2D` accepts only `MaxPool2dAttrs`, exactly one input, and exactly one output.

### NCHW Shape and spatial windows

For each spatial axis, let input extent be `D`, positive kernel sample count be `k`, symmetric
padding per side be `p`, positive dilation be `d`, and positive stride be `s`:

```text
effectiveKernel = d * (k - 1) + 1
numerator       = D + 2 * p - effectiveKernel
floor output    = floor(numerator / s) + 1
ceil output     = ceil(numerator / s) + 1
```

All literal arithmetic uses checked `long` operations. For static `D`, reject a negative
numerator before division. For unresolved `D`, use existing `DimensionExpressions.addConstant`,
then `floorDivide` or `ceilingDivide`, then `addConstant`; the result retains the binding
obligation that the numerator is non-negative. Do not require static spatial input, create an
unknown extent, clamp a negative numerator to zero, or modify the symbolic-expression model.

The output Shape is `[N, C, H_out, W_out]`. Preserve the exact input `N` and `C` Dimension
references. Floor mode includes every window in the floor grid. Ceil mode includes every window
in the literal ceiling grid, including a final window whose start lies entirely in trailing
padding; there is no terminal-window decrement. Along an axis, output position `o` starts at
`o * s - p` and samples positions `start + r * d` for `r` from zero through `k - 1`.

Exact task-owned Shape failures are:

```text
maxPool2d input rank must be 4: <rank>
maxPool2d effective kernel does not fit padded height: input=<dimension>, effectiveKernel=<value>, padding=<value>
maxPool2d effective kernel does not fit padded width: input=<dimension>, effectiveKernel=<value>, padding=<value>
```

Static zero batch or channel extents are valid and make the result empty while retaining spatial
metadata. A static zero height or width is valid only when its padding makes the corresponding
numerator non-negative; resulting windows are all-padding where no sampled coordinate is in
bounds. Valid static geometry always produces at least one spatial position. Unresolved geometry
is accepted with the same non-negative-numerator binding obligation.

### Data type, extrema, and numerical policy

The sole input must be BFLOAT16, FLOAT32, or FLOAT64. The result retains the exact input type; no
promotion or accumulation type applies. The exact type failure is:

```text
maxPool2d input must have a floating data type, but was <dataType>
```

Padding positions do not participate. If a window has one or more in-bounds sampled positions:

- any NaN is greater for selection purposes than every non-NaN, so NaN propagates;
- otherwise ordinary numerical maximum applies, with positive zero greater than negative zero;
- infinities use ordinary numerical order; and
- equal candidates select the first sampled input in increasing kernel-height then kernel-width
  order.

Multiple NaNs use that same first-sample tie rule. NaN payload/sign and signaling preservation
remain unspecified, matching the current floating-reduction policy; signed-zero selection is
exact. An all-padding window has no selected input and returns exact negative infinity in the
input type. These tie rules are semantic evidence for later compiler-owned gradients; this task
creates no gradient rule or indices output. The value contract fixes results independently of
physical traversal but does not promise a particular algorithm.

### Validation and construction order

The helper validates before its sole factory call in this order:

1. null-check `input`, then `attrs`;
2. validate input floating eligibility;
3. validate input rank;
4. derive height, then width, including checked effective-kernel, padding, numerator, and
   floor/ceil arithmetic;
5. create the result descriptor and exact operation; and
6. delegate once with exact ordered input `[input]`.

Null messages are parameter names. Checked-arithmetic and existing symbolic-expression failures
retain their existing messages. Every failure before final delegation consumes no Tensor ID,
producer, or wrapper. A successful call consumes exactly one ID and creates one producer.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Capabilities](../capabilities.md)
- [Master plan](../master-plan.md)
- [Unfold/fold semantics](0017m-unfold-and-fold-semantics.md)
- [Unfold/fold expressions](0017n-unfold-and-fold-tensor-expressions.md)
- [Signature hardening](0018k-operation-signature-and-construction-hardening.md)
- [Shared provenance](0018l-shared-multi-output-tensor-provenance.md)
- [Symbolic extents](0018m-symbolic-extent-expressions.md)
- [Typed scalar](0018n-typed-scalar-value-contract.md)
- [Reduction policies](0018v-multi-axis-and-statistical-reductions.md)
- [NCHW Conv2d](0020-nchw-conv2d-semantics-and-tensor-expressions.md)

## Architecture constraints

- Work stays in model plus its documentation/planning. Tensor remains public mutable state, not
  graph IR.
- Pooling operation types record backend-independent meaning only and do not import Tensor,
  graph, compiler, runtime, prepare, or backend types.
- Package direction is tensor helper to pooling operation/datatype/shape; packages remain acyclic.
- Compiler owns capture, binding/deferred proof, gradients, adjoints, and saved indices. Backend
  prepare owns conforming algorithms, lowering, kernels, and materialization. Runtime receives
  prepared work without original operations on its hot path.
- No architecture, dependency, lifecycle, focused-architecture, Gradle, or cross-module change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.datatype`
- `io.github.pho001.synaptik.model.shape`
- `io.github.pho001.synaptik.model.operation`
- `io.github.pho001.synaptik.model.tensor`

Package added:

- `io.github.pho001.synaptik.model.operation.pooling` — public pooling identities and
  operation-specific immutable attributes only.

Type placement:

- `...operation.pooling.Pool2dKind` — pooling identity/signature owner, initially MAX only.
- `...operation.pooling.MaxPool2dAttrs` — public inspectable max-window geometry.
- `...tensor.TensorMaxPool2dExpressions` — package-private validation, Shape, descriptor, and
  provenance construction owner.
- `...tensor.Tensor` — established public fluent receiver facade.

Tests mirror production packages where package-private access is required.

## Affected files

Production (4):

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/pooling/Pool2dKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/pooling/MaxPool2dAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorMaxPool2dExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests (7):

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/pooling/Pool2dSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMaxPool2dExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java` — exact
  signature and public count 177 to 178.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
  — count only, 177 to 178.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
  — count only, 177 to 178.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLinearExpressionTest.java`
  — count only, 177 to 178.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressionTest.java`
  — count only, 177 to 178.

Documentation/planning (7):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless inaccurate: Runtime/Training APIs; Conv2d, unfold/fold, pad, reduction,
Shape/Dimension expression, operation/signature, Tensor/provenance contracts; architecture/ADRs/
tests; conformance/integration; Gradle; other modules.

## Maximum scope

Exactly 18 paths maximum: four production, seven test, and seven documentation/planning paths.
`Tensor.java` changes only one import, one method/Javadoc, and its operation inventory. Five
existing tests change only the stated signature/count. Stop for path 19, another type/test/doc,
an existing-helper change, architecture, Gradle, or cross-module work.

This is one cohesive vertical model capability. Combining average pooling would add a second
attribute/numerical contract and exceed the established path guardrail; 0020A1 remains Draft.

## Javadoc and documentation requirements

- Fully document kind, attrs, helper, and Tensor method: NCHW window meaning, Shape formulas,
  literal ceil grid, padding exclusion, extrema/special/empty policy, metadata/provenance,
  validation/failures, and lifecycle boundaries.
- Every parameter, result, and expected failure has complete `@param`, `@return`, and `@throws`
  tags as applicable.
- Tensor API gets a geometry table, static floor/ceil and symbolic examples, extrema policy, and
  current-model versus planned compiler/execution boundary.
- Compile API records current max-pooling metadata and future compiler-owned binding, capture,
  gradient/index-saving, and decomposition without claiming compiler support.
- Review glossary terms NCHW, pooling, window, padding, dilation, effective kernel, and ceil mode;
  add or refine only reusable distinctions needed by the public explanation.
- Synchronize capability/task/master/roadmap statuses: 0020 and 0020A are Complete; 0020A1 plus
  0021–0024 remain Draft with no detailed future specs, and no model task is Ready.
- Record reasoned no-change conclusions for Runtime/Training APIs, related contracts,
  architecture, conformance/integration, Gradle, and other modules.

## Acceptance criteria

- Exact kind, attrs, one-input/one-output signature, and one receiver method exist; public Tensor
  method count is 178.
- Static and symbolic floor/ceil Shape derivation, exact N/C reference retention, checked
  geometry, and deferred unresolved obligation match this task.
- BFLOAT16/FLOAT32/FLOAT64 retain exact type; BOOL/integral fail; requiresGrad is retained,
  layout is unresolved, and result is fresh, unlabeled, and storage-free.
- Padding exclusion, all-padding negative infinity, NaN propagation/payload selection,
  infinity, signed-zero, and first logical-kernel tie policies are explicit and API-locked without
  evaluation.
- Validation order, exact task-owned failures, no-ID failures, one-ID success, and exact
  one-input/output-zero provenance pass.
- No average pooling, indices output, gradients, compiler, algorithm, backend/runtime,
  architecture, dependency, or build work is added.
- Focused tests, exactly one final model suite, Javadoc/docs/link/anchor/fence/newline/whitespace,
  exact 18 paths/packages/public surface/statuses, and `git diff --check` pass.
- A separate clean documentation-focused pass reuses Java evidence, finalizes Javadocs/docs, and
  records reasoned no-change conclusions before completion.

## Tests / validation

Focused development validation:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.pooling.Pool2dSemanticsTest --tests io.github.pho001.synaptik.model.tensor.TensorMaxPool2dExpressionTest --tests io.github.pho001.synaptik.model.shape.DimensionExpressionsTest --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest --tests io.github.pho001.synaptik.model.tensor.TensorProducerTest --tests io.github.pho001.synaptik.model.tensor.TensorProvenanceTest
```

After executable Java stabilizes, exactly once:

```bash
./gradlew :modules:model:test
```

The focused tests cover attributes/signature, public surface, all static/symbolic floor/ceil
cases, literal terminal windows, type/rank/geometry failures, metadata, validation/ID effects,
freshness/provenance, and numerical contracts as meaning/Javadoc without value execution.

Documentation pass after final Javadoc:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

Also validate local Markdown links/anchors, fences, terminology, examples, generated Javadoc,
newlines/whitespace, exact paths/packages, public count/signature, statuses, and absence of a
detailed 0020A1-or-later specification. Repository validation is deferred to the selected-modern-
operations checkpoint after 0022 or CI because no repository-wide contract changes.

## Dependencies

- 0001–0002 and 0018M–0018M1: DataType, Shape, Dimensions, canonical symbolic arithmetic.
- 0005–0007, 0011–0013, and 0018K–0018L: operation/signature,
  Tensor/descriptor/factory/provenance.
- 0016A–0016D, 0018U1, and 0018V: floating extrema, NaN/signed-zero, empty-domain, and
  deterministic logical-order precedents.
- 0017M–0017N: NCHW window vocabulary and the explicit reason not to reuse `Window2dAttrs`.
- 0018N: exact typed infinity and signed-zero representation vocabulary.
- 0020: NCHW naming, checked static/symbolic spatial formula, metadata, and lifecycle precedent.

## Follow-up tasks

- 0020A1 remains Draft for average pooling with its own attributes, divisor/count-padding,
  accumulation, special-value, empty/all-padding, and validation contracts. It may extend
  `Pool2dKind` but must not weaken or alias `MaxPool2dAttrs`.
- Compiler later owns capture, binding/constraints, gradients, adjoints, and saved max indices.
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
operation/signature/Tensor/provenance/symbolic-extent/typed-scalar/reduction/window/Conv2d tasks,
current related source/tests/APIs/glossary, and task 0020A.

Implement task 0020A exactly inside 18 paths. Update every global public Tensor inventory/count
177 -> 178 up front. Preserve all contracts and stop on architecture uncertainty, scope overflow,
another type/test/document, existing-helper change, Gradle, or cross-module work.

Run focused validation and exactly one final model suite after Java stabilizes. Hand the actual
diff and Java evidence to a separate clean documentation-focused agent in the same change; it
finalizes Javadocs, Tensor/Compile APIs, glossary, capabilities/planning and documentation checks
while reusing Java evidence. Keep 0020A Ready until all criteria pass, then mark it Complete while
0020A1 and 0021–0024 remain Draft with no next detailed specification.
```

## Documentation-agent handoff

Provide this task, complete diff, exact focused/final Java evidence and post-test Java-change
state, public/Shape/type/ceil/extrema/padding/provenance policies, seven documentation paths, and
validation requirements. The clean agent reads repository instructions, architecture, rules and
General/API-Javadoc/Planning/Example profiles, task, source/tests/generated Javadoc,
Tensor/Compile/Runtime/Training APIs, glossary/planning, and directly related contracts. It
finalizes documentation and records reasoned no-change conclusions without repeating successful
Java tests absent executable change, stale evidence, or a concrete risk.

## Local decisions

- Max pooling uses its own `MaxPool2dAttrs` rather than `Window2dAttrs`: identical coordinate
  fields do not make conceptual-zero window transforms and excluded-padding extrema the same
  contract.
- Literal ceil mode retains the complete ceiling grid without a terminal-window decrement. This
  keeps static and symbolic formulas identical and allows an all-padding terminal window.
- The operation exposes no indices result. First-logical-sample ties fix forward meaning and
  future gradient evidence without selecting a current gradient or saved-index representation.

## Known limitations

- This task implements model semantics and public expression metadata only. Compiler capture,
  binding proof, gradients/adjoints, saved indices, backend lowering, algorithms, prepared
  execution, conformance, and integration remain planned in their owning layers.
- Average pooling remains Draft task 0020A1 because divisor and padding-count semantics require a
  separate contract.

## Validation evidence

- Implementation context `/root/task_0020a_implementation` ran the specified focused Gradle
  command after correcting its test fixture: passed, 33 tests. The same context then ran exactly
  one final `./gradlew :modules:model:test`: passed, 870 tests across 111 suites. Executable Java
  did not change after that final suite; the documentation pass reused this evidence and did not
  repeat Java tests.
- Documentation-focused clean context
  `/root/task_0020a_implementation/task_0020a_docs` applied the General, API/Javadoc, Planning,
  and Example profiles. It reviewed the complete task diff; all four production files; both
  focused tests; `TensorTest` and count locks; Tensor, Compile, Runtime, and Training API pages;
  glossary and planning state; and related Conv2d, window-transform, reduction, Shape-expression,
  operation/signature, Tensor/provenance, architecture, and lifecycle contracts.
- The documentation pass finalized Javadocs in the four task-owned production paths and updated
  the Tensor API with the geometry table, static floor/ceil and symbolic examples, extrema policy,
  and lifecycle boundary. It updated the Compile API with current metadata and planned compiler
  ownership, and refined reusable NCHW pooling/window/padding/dilation/effective-kernel/ceil-mode
  glossary distinctions.
- Documentation validation: `./gradlew :modules:model:javadoc` passed. The static Tensor API
  example compiled and ran with the documented two output Shapes. Local Markdown links and
  anchors, fences, final newlines, whitespace, generated Javadoc content, exact 18 paths,
  production/test packages, public Tensor count 178, `MAX_POOL2D` one-input/one-output signature,
  synchronized statuses, and absence of detailed 0020A1-or-later specifications were checked and
  passed. `git diff --check` passed.
- Runtime and Training API pages require no change: they describe planned prepared execution and
  autograd/training boundaries, while this task adds no runtime or training API. Related Conv2d,
  unfold/fold, pad, reduction, Shape/Dimension-expression, operation/signature, and
  Tensor/provenance contracts remain accurate and unchanged because max pooling owns a new
  operation-specific type and construction helper. Architecture documents/tests, backend
  conformance, integration tests, Gradle, and other modules require no change because there is no
  dependency, lifecycle, backend, execution, build, or cross-module change.

## Implementation notes

- Added `Pool2dKind.MAX_POOL2D`, exact immutable `MaxPool2dAttrs`, one package-private construction
  helper, and exactly `Tensor.maxPool2d(MaxPool2dAttrs)`.
- Construction accepts exactly BFLOAT16/FLOAT32/FLOAT64 rank-four NCHW input, retains exact type,
  batch/channel references and gradient request, derives checked static or canonical symbolic
  floor/ceil spatial Dimensions, and creates one fresh unresolved-layout, unlabeled, storage-free
  output with exact one-input/output-zero provenance.
- Public and planning documentation records excluded padding, all-padding negative infinity, NaN
  dominance, positive-zero ordering, ordinary infinity behavior, and first height-major kernel
  sample ties without claiming evaluation or downstream lifecycle support.

## Completion summary

- Completed changes: added NCHW maximum-pooling semantics, attributes, public Tensor expression,
  focused tests, complete Javadocs, public API explanations, glossary distinctions, and
  synchronized capability/planning status.
- Files changed or created: exactly four production, seven test, and seven documentation/planning
  paths listed in this specification.
- Tests and validation: reused focused 33-test and final 870-test/111-suite implementation
  evidence; documentation Javadoc, runnable example, links/anchors/fences/newlines/whitespace,
  generated output, scope/public/signature/status/no-spec checks, and `git diff --check` passed.
- Documentation-agent review: completed in clean context
  `/root/task_0020a_implementation/task_0020a_docs` using the required profiles.
- Documentation impact: Tensor API, Compile API, glossary, capabilities, task, master plan, and
  roadmap finalized; Runtime/Training and related-contract no-change conclusions are recorded.
- Javadoc review: all four production paths reviewed and finalized; generated model Javadoc
  passed.
- Glossary impact: added reusable NCHW pooling window distinctions without redefining ordinary
  programming terms.
- Unresolved issues: None.
- Follow-up required: None for task 0020A; Draft 0020A1 remains future average-pooling work.

Status: Complete
