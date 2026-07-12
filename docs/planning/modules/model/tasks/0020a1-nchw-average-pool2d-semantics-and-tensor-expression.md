# Task 0020A1: NCHW Average Pool2d Semantics and Tensor Expression

## Status

Complete

## Goal

Add one first-class backend-independent `AVERAGE_POOL2D` operation and exactly one public receiver
expression for floating rank-four NCHW tensors. The result must retain exact batch/channel
dimensions, derive truthful static or symbolic spatial dimensions, and record one fixed
count-padding average, accumulation, special-value, metadata, and provenance contract without
evaluating values.

This task completes the average-pooling half of the former combined pooling frontier. It follows
completed task 0020A's NCHW coordinate and literal floor/ceiling grid, but owns a distinct
attributes type because average divisor and accumulation meaning are not max-extrema parameters.

## Scope

- Extend `Pool2dKind` with `AVERAGE_POOL2D`, accepting only `AveragePool2dAttrs`, exactly one input,
  and exactly one output. Preserve `MAX_POOL2D` and its signature unchanged.
- Add immutable public `AveragePool2dAttrs(kernelHeight, kernelWidth, strideHeight, strideWidth,
  paddingHeight, paddingWidth, dilationHeight, dilationWidth, ceilMode)` using `long` geometry.
- Add exactly `public Tensor averagePool2d(AveragePool2dAttrs attrs)`; add no short alias.
- Add one package-private stateless `TensorAveragePool2dExpressions` helper and one final
  `TensorFactory.createDerived` call for each successful expression.
- Require floating rank-four NCHW input `[N, C, H, W]` and preserve exact input type, batch,
  channel, and gradient-request metadata.
- Derive floor- or ceil-mode static and symbolic spatial extents with the exact completed-0020A
  literal-grid formulas and checked arithmetic.
- For every output window, use the fixed divisor `kernelHeight * kernelWidth`. Every logical kernel
  position counts once. An out-of-bounds position contributes conceptual positive zero to the
  numerator and one count to the denominator.
- Select exact accumulation, empty/all-padding, NaN, infinity, signed-zero, and determinism
  policies.
- Produce one fresh unlabeled, storage-free result with unresolved layout and exact one-input,
  output-index-zero provenance.
- Update focused API, glossary, capability, and planning documentation plus public-Tensor
  inventory locks.

## Out of scope

- maximum, global, adaptive, one-dimensional, three-dimensional, or channels-last pooling
- configurable count-padding, excluded-padding averages, divisor overrides, asymmetric or
  automatic padding, padding Tensor inputs, configurable padding values, runtime geometry, or a
  broad options framework
- integral, BOOL, quantized, sparse, complex, unsigned, FLOAT16, or mixed-type input; casts,
  promotion, output-type overrides, or accumulator options
- terminal-window removal in ceil mode; average pooling uses task 0020A's literal symmetric
  padded grid, including a terminal all-padding window
- reuse or modification of `MaxPool2dAttrs` or `Window2dAttrs`; their extrema and conceptual
  window-transform meanings are not average-divisor semantics
- value reads, eager evaluation, host allocation, result storage, resolved layout, mutation, or
  input materialization
- gradients, adjoints, saved values, backward kinds, compiler capture, graph-wide constraint
  solving, canonicalization, decomposition, fusion, or optimization
- summation or division algorithms, compensated summation, backend capabilities, lowering,
  prepare, runtime, execution, tolerances, conformance, or integration
- changes to Conv2d, max pooling, unfold/fold, pad, aggregate reductions, symbolic-expression
  foundations, promotion, provenance/factory seams, architecture, dependencies, Gradle, or
  another module
- later task specifications, normalization, loss, compiler-generated semantic work, adaptive or
  global pooling

## Public and operation contracts

### Public surface, attributes, and signature

The sole new receiver method is:

```java
public Tensor averagePool2d(AveragePool2dAttrs attrs)
```

The full word `average` is the canonical public spelling. Do not add `avgPool2d`, overloads,
defaults, static entries, or compatibility aliases.

`AveragePool2dAttrs` validates components in declaration order. Kernel, stride, and dilation
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

`ceilMode` is retained unchanged. The record owns intrinsic average-pooling geometry only. The
fixed count-padding/divisor policy is part of `AVERAGE_POOL2D` meaning and is not configurable
state, so the record has no `countIncludePad`, `divisorOverride`, or padding-value component. It
owns no Tensor, Shape, DataType, layout, storage, gradient, compiler, backend, or execution state.

`Pool2dKind` retains exact declaration order `MAX_POOL2D`, `AVERAGE_POOL2D`. Each kind has one
family-owned fixed one-input/one-output signature paired only with its own attributes class. A max
attrs value must not construct average pooling, and an average attrs value must not construct max
pooling.

### NCHW Shape and literal spatial windows

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
unknown extent, clamp a negative numerator to zero, add a terminal-window correction, or modify
the symbolic-expression model.

The output Shape is `[N, C, H_out, W_out]`. Preserve the exact input `N` and `C` Dimension
references. Along an axis, output position `o` starts at `o * s - p` and samples positions
`start + r * d` for `r` from zero through `k - 1`. Floor mode includes every window in the floor
grid. Ceil mode includes every window in the literal ceiling grid, including a final window whose
start lies entirely in trailing padding. These formulas and coordinates intentionally match
completed task 0020A; only the value policy differs.

Exact task-owned Shape failures are:

```text
averagePool2d input rank must be 4: <rank>
averagePool2d effective kernel does not fit padded height: input=<dimension>, effectiveKernel=<value>, padding=<value>
averagePool2d effective kernel does not fit padded width: input=<dimension>, effectiveKernel=<value>, padding=<value>
```

Static zero batch or channel extents are valid and make the result empty while retaining spatial
metadata. A static zero height or width is valid only when padding makes the corresponding
numerator non-negative. Valid static geometry always produces at least one spatial position;
some or all windows may contain no in-bounds samples. Unresolved geometry is accepted with the
same non-negative-numerator binding obligation.

### Data type, divisor, and numerical policy

The sole input must be BFLOAT16, FLOAT32, or FLOAT64. The result retains the exact input type; no
input promotion occurs. The exact type failure is:

```text
averagePool2d input must have a floating data type, but was <dataType>
```

For each output window, there are exactly `kernelHeight * kernelWidth` logical kernel positions;
dilation changes their coordinates, not their count. Every in-bounds position contributes its
input value to the numerator. Every out-of-bounds position contributes conceptual exact positive
zero. Every logical position contributes one to the denominator, so the divisor is always the
positive mathematical product `kernelHeight * kernelWidth`. Padding therefore contributes zero
to the numerator and contributes to the denominator. No runtime count, valid-sample count,
divisor override, or zero-divisor case exists. The mathematical product need not be materialized
as a new `long` attribute during model construction.

BFLOAT16 and FLOAT32 inputs accumulate and divide in FLOAT32; FLOAT64 input accumulates and divides
in FLOAT64. BFLOAT16 converts the final accumulator result to BFLOAT16. The operation's selected
meaning is the accumulator-domain sum divided once by the exact positive divisor. Reassociation,
tree reduction, and other conforming summation order are permitted, so fixed traversal order,
bitwise equality, NaN payload/sign preservation, and cross-backend identical rounding are not
promised. Later backend conformance owns accuracy tolerances; this task does not select an
algorithm or evaluate values.

The observable special-value classes are fixed:

- any in-bounds NaN makes the result NaN;
- positive and negative infinity both present in one window make the result NaN;
- otherwise any positive infinity makes the result positive infinity and any negative infinity
  makes it negative infinity;
- conceptual padding never introduces NaN or infinity;
- an exact-zero finite mean is negative zero only when every divisor contribution is an in-bounds
  negative zero; cancellation, any positive zero, or any conceptual padding produces positive
  zero; and
- an all-padding window returns exact positive zero in the result type.

Empty batch/channel outputs contain no values. A spatial window is never divided by zero because
both kernel counts are positive. The special-value and zero rules make observable results
independent of physical traversal; finite non-zero rounding remains subject to the permitted
reassociation and later conformance tolerance.

### Validation and construction order

The helper validates before its sole factory call in this order:

1. null-check `input`, then `attrs`;
2. validate input floating eligibility;
3. validate input rank;
4. derive height, then width, including checked effective-kernel, padding, numerator, and
   floor/ceil arithmetic;
5. create the result descriptor and exact `AVERAGE_POOL2D` operation; and
6. delegate once with exact ordered input `[input]`.

Null messages are parameter names. Attribute construction validates its fields before the helper
is called. Checked-arithmetic and existing symbolic-expression failures retain their existing
messages. There is no separate divisor validation or divisor-overflow message. Every failure
before final delegation consumes no Tensor ID, producer, or wrapper. A successful call consumes
exactly one Tensor ID, creates one producer, and creates one output wrapper.

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
- [NCHW max pooling](0020a-nchw-max-pool2d-semantics-and-tensor-expression.md)

## Architecture constraints

- Work stays in model plus its documentation/planning. Tensor remains public mutable state, not
  graph IR.
- Pooling operation types record backend-independent meaning only and do not import Tensor,
  graph, compiler, runtime, prepare, or backend types.
- Package direction is tensor helper to pooling operation/datatype/shape; packages remain acyclic.
- Compiler owns capture, binding/deferred proof, gradients, adjoints, and saved values. Backend
  prepare owns conforming algorithms, lowering, kernels, tolerance satisfaction, and
  materialization. Runtime receives prepared work without original operations on its hot path.
- No architecture, dependency, lifecycle, focused-architecture, Gradle, or cross-module change.

## Package impact

Existing packages used and changed:

- `io.github.pho001.synaptik.model.operation.pooling` — extend the existing public pooling kind
  and add average-specific immutable attributes without weakening max pooling.
- `io.github.pho001.synaptik.model.datatype`
- `io.github.pho001.synaptik.model.shape`
- `io.github.pho001.synaptik.model.operation`
- `io.github.pho001.synaptik.model.tensor`

Packages added: None.

Type placement:

- `...operation.pooling.Pool2dKind` — existing pooling identity/signature owner extended with
  exact `AVERAGE_POOL2D` pairing.
- `...operation.pooling.AveragePool2dAttrs` — public inspectable average-window geometry; its
  dedicated type prevents max extrema and average divisor contracts from being interchanged.
- `...tensor.TensorAveragePool2dExpressions` — package-private validation, Shape, descriptor, and
  provenance construction owner.
- `...tensor.Tensor` — established public fluent receiver facade.

Tests mirror production packages where package-private access is required.

## Affected files

Production (4):

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/pooling/Pool2dKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/pooling/AveragePool2dAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorAveragePool2dExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests (7):

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/pooling/Pool2dSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorAveragePool2dExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java` — exact
  signature and public count 178 to 179.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
  — count only, 178 to 179.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
  — count only, 178 to 179.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLinearExpressionTest.java`
  — count only, 178 to 179.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressionTest.java`
  — count only, 178 to 179.

Documentation/planning (7):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless inaccurate: Runtime/Training APIs; Conv2d, max pooling, unfold/fold, pad,
reduction, Shape/Dimension-expression, operation/signature, Tensor/factory/provenance contracts;
architecture/ADRs/tests; conformance/integration; Gradle; other modules.

## Maximum scope

Exactly 18 paths maximum: four production, seven test, and seven documentation/planning paths.
`Tensor.java` changes only one import, one method/Javadoc, and its operation inventory. Five
existing tests change only the stated signature/count. `Pool2dSemanticsTest` may cover both exact
attrs pairings while preserving all max assertions. Stop for path 19, another type/test/document,
an existing-helper change, architecture, Gradle, or cross-module work.

This is one cohesive vertical model capability. Kind, average-specific attrs, public construction,
symbolic Shape derivation, fixed divisor/numerical meaning, API locks, and documentation must agree
in one compilable state.

## Javadoc and documentation requirements

- Fully document kind, attrs, helper, and Tensor method: canonical public spelling, NCHW window
  meaning, Shape formulas, literal ceil grid, fixed count-padding divisor, accumulation,
  special/empty policy, metadata/provenance, validation/failures, and lifecycle boundaries.
- Every parameter, result, and expected failure has complete `@param`, `@return`, and `@throws`
  tags as applicable.
- Tensor API gets a geometry/divisor table, static floor/ceil and symbolic examples, one numerical
  example distinguishing count-padding from valid-sample averaging, special-value policy, and
  current-model versus planned compiler/execution boundary.
- Compile API records current average-pooling metadata and future compiler-owned binding,
  capture, legal decomposition, and gradients without claiming compiler support.
- Review glossary terms NCHW, pooling, window, padding, dilation, effective kernel, ceil mode,
  count-padding, divisor, and accumulation; add or refine only reusable distinctions needed by the
  public explanation.
- Synchronize capability/task/master/roadmap statuses: 0020 and 0020A remain Complete; keep
  0020A1 Ready during implementation and mark it Complete only after every acceptance criterion
  passes; 0021–0024 remain Draft without detailed specifications.
- Record reasoned no-change conclusions for Runtime/Training APIs, related contracts,
  architecture, conformance/integration, Gradle, and other modules.

## Acceptance criteria

- Exact `AVERAGE_POOL2D` kind, dedicated attrs, one-input/one-output signature, and one canonical
  receiver method exist; max pooling remains unchanged; public Tensor method count is 179.
- Static and symbolic floor/ceil Shape derivation, exact N/C reference retention, checked
  geometry, literal terminal windows, and deferred unresolved obligation match this task and
  completed 0020A.
- BFLOAT16/FLOAT32/FLOAT64 retain exact type; BOOL/integral fail; selected accumulator types,
  fixed kernel-count divisor, conceptual padding zeros, and one final division are documented and
  API-locked without evaluation.
- NaN, opposing/single infinity, exact signed-zero selection, all-padding positive zero, empty
  axes, reassociation, rounding, and determinism policies are explicit.
- Requires-grad is retained exactly; layout is unresolved; result is fresh, unlabeled, and
  storage-free.
- Validation order, exact task-owned failures, no-ID failures, one-ID/one-producer/one-wrapper
  success, and exact one-input/output-zero provenance pass.
- No count-padding flag, valid-sample divisor, divisor override, alias/overload, max/window attrs
  weakening, gradient, compiler, algorithm, backend/runtime, architecture, dependency, build, or
  later-spec work is added.
- Focused tests, exactly one final model suite, Javadoc/docs/link/anchor/fence/newline/whitespace,
  exact 18 paths/packages/public surface/statuses, and `git diff --check` pass.
- A separate clean documentation-focused pass reuses Java evidence, finalizes Javadocs/docs, and
  records reasoned no-change conclusions before completion.

## Tests / validation

Focused development validation:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.pooling.Pool2dSemanticsTest --tests io.github.pho001.synaptik.model.tensor.TensorAveragePool2dExpressionTest --tests io.github.pho001.synaptik.model.shape.DimensionExpressionsTest --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest --tests io.github.pho001.synaptik.model.tensor.TensorProducerTest --tests io.github.pho001.synaptik.model.tensor.TensorProvenanceTest
```

After executable Java stabilizes, exactly once:

```bash
./gradlew :modules:model:test
```

The focused tests cover both exact attrs/signature pairings, public surface, all static/symbolic
floor/ceil cases, literal terminal/all-padding windows, type/rank/geometry failures, accumulator/
divisor/special-value contracts as inspectable meaning and Javadoc, metadata, validation/ID
effects, freshness, and provenance without value execution.

Documentation pass after final Javadoc:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

Also validate local Markdown links and anchors, fences, terminology, examples, generated Javadoc,
final newlines and whitespace, exact paths/packages, public count/signature, exact kind/signature
pairing, synchronized statuses, and absence of detailed 0021-or-later specifications. Repository
validation is deferred to the selected-modern-operations checkpoint after 0022 or CI because no
repository-wide contract changes.

## Dependencies

- 0001–0002 and 0018M–0018M1: DataType, Shape, Dimensions, canonical symbolic arithmetic.
- 0005–0007, 0011–0013, and 0018K–0018L: operation/signature,
  Tensor/descriptor/factory/provenance.
- 0016A–0016D, 0018U1, and 0018V: floating accumulation, NaN/infinity/signed-zero, empty-domain,
  and reassociation precedents.
- 0017M–0017N: NCHW window vocabulary and the reason not to reuse `Window2dAttrs`.
- 0018N: exact typed positive-zero and floating special-value vocabulary.
- 0020: NCHW naming, checked static/symbolic spatial formula, metadata, and lifecycle precedent.
- 0020A: `Pool2dKind`, literal floor/ceil grid, operation-specific attrs, public-surface, and
  one-input provenance precedent.

## Follow-up tasks

- Task 0021 remains Draft for normalization operations and must not be specified or implemented
  here.
- Compiler later owns capture, dynamic binding/constraint proof, gradients, adjoints, and any
  legal decomposition preserving this fixed divisor and special-value meaning.
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
operation/signature/Tensor/provenance/symbolic-extent/typed-scalar/reduction/window/Conv2d/max-
pooling tasks, current related source/tests/APIs/glossary, and task 0020A1.

Implement task 0020A1 exactly inside 18 paths. Update every global public Tensor inventory/count
178 -> 179 up front. Preserve completed max pooling and all architecture contracts; stop on
architecture uncertainty, scope overflow, another type/test/document, existing-helper change,
Gradle, cross-module work, or a need to alter the selected fixed count-padding contract.

Run focused validation and exactly one final model suite after Java stabilizes. Hand the actual
diff and Java evidence to a separate clean documentation-focused agent in the same change; it
finalizes Javadocs, Tensor/Compile APIs, glossary, capabilities/planning and documentation checks
while reusing Java evidence. Keep 0020A1 Ready until all criteria pass, then mark it Complete while
0021–0024 remain Draft with no next detailed specification.
```

## Documentation-agent handoff

Provide this task, complete diff, exact focused/final Java evidence and post-test Java-change
state, public/Shape/type/ceil/divisor/padding/special-value/provenance policies, seven documentation
paths, and validation requirements. The clean agent reads repository instructions, architecture,
rules and General/API-Javadoc/Planning/Example profiles, task, source/tests/generated Javadoc,
Tensor/Compile/Runtime/Training APIs, glossary/planning, and directly related contracts. It
finalizes documentation and records reasoned no-change conclusions without repeating successful
Java tests absent executable change, stale evidence, or a concrete risk.

## Local decisions

- One `AVERAGE_POOL2D` kind with one dedicated `AveragePool2dAttrs` record prevents average
  divisor/accumulation semantics from weakening completed max-pooling extrema semantics.
- Count-padding is fixed operation meaning rather than configurable attributes. The mathematical
  divisor is not materialized with checked `long` multiplication during model construction,
  because positive kernel counts already establish a positive mathematical product and model
  construction evaluates no window.
- Static and symbolic result geometry follows completed max pooling exactly, including literal
  ceil-grid terminal all-padding windows. Only the eventual value policy differs.
- The documentation pass applied the General, API/Javadoc, Planning, and Example profiles. It
  refined the reusable NCHW pooling glossary entry with count-padding, fixed-divisor,
  accumulation-domain, and average-versus-valid-sample distinctions.
- Runtime and Training API pages remain unchanged because this task adds model expression metadata
  only. Runtime will receive prepared work, while compiler/training later own graph capture,
  binding proof, gradients, and backward construction.
- Related Conv2d, max-pooling, unfold/fold, pad, reduction, Shape/Dimension-expression,
  operation/signature, Tensor/factory/provenance, architecture, ADR, architecture-test,
  backend-conformance, integration, Java 26 Gradle, dependency, and other-module contracts remain
  accurate unchanged because this task composes their current model boundaries without changing
  their APIs, behavior, ownership, build, or executable layers.

## Known limitations

- Only floating rank-four NCHW input and static geometry attributes are current. Input spatial
  extents may be symbolic, but their non-negative numerator obligation is not currently bound or
  proved by a compiler.
- Current construction records result metadata, numerical meaning, and provenance only. It does
  not read values, evaluate pooling, construct gradients, capture or decompose a graph, select a
  backend algorithm, allocate result storage, lower, prepare, or execute.
- Permitted summation reassociation means finite non-zero bitwise results and identical
  cross-backend rounding are deliberately not promised. Later conformance work must select and
  validate tolerances.
- Repository-wide validation remains deferred to the selected-modern-operations checkpoint after
  task 0022 or CI, as planned.

## Validation evidence

- Implementation context `/root/task_0020a1_implementation` ran the exact focused command. It
  passed `BUILD SUCCESSFUL`; three Gradle tasks executed. The same context then ran exactly one
  final `./gradlew :modules:model:test`; it passed `BUILD SUCCESSFUL` with three actionable tasks,
  one executed and two up-to-date. No executable Java changed afterward.
- Clean documentation context
  `/root/task_0020a1_implementation/documentation_finalization` reused both successful Java
  results and did not rerun Java tests. It independently reviewed the final four production
  contracts, seven tests, Tensor/Compile/Runtime/Training APIs, glossary, capabilities, task,
  master plan, roadmap, and related pooling/window/reduction/Shape/provenance boundaries.
- The documentation context ran `./gradlew :modules:model:javadoc` after final Javadoc. It passed
  `BUILD SUCCESSFUL` with two actionable tasks, both executed. Generated pages for `Pool2dKind`,
  `AveragePool2dAttrs`, and `Tensor.averagePool2d` contain the reviewed fixed-divisor,
  accumulation/division, special-value, metadata, failure, and lifecycle contracts.
- A targeted local Markdown checker passed 593 links including 160 heading anchors across all
  seven documentation/planning paths. Fence balance and final-newline checks also passed. Manual
  terminology and example review confirmed the literal floor/ceil and symbolic Shape formulas,
  the `0.25` count-padding example versus `1.0` valid-sample averaging, and current-versus-planned
  compiler/execution boundaries.
- The final scope and surface audit found exactly 18 authorized paths: four production, seven
  tests, and seven documentation/planning paths. Production packages remain the planned pooling
  and tensor packages; compiled `Tensor` has exactly 179 public methods and exactly one
  `averagePool2d(AveragePool2dAttrs)` receiver; compiled pooling types expose
  `AVERAGE_POOL2D` and the exact nine-component attributes constructor. No `avgPool2d`,
  configuration flag/override, later detailed specification, cross-module import, or executable
  Java change was introduced.
- Final planning audit confirmed tasks 0020, 0020A, and 0020A1 Complete; tasks 0021–0024 Draft;
  no model task Ready; and no detailed 0021-or-later specification. `git diff --check` passed on
  the final combined change.

## Implementation notes

- Added `Pool2dKind.AVERAGE_POOL2D`, immutable `AveragePool2dAttrs`, one package-private stateless
  `TensorAveragePool2dExpressions` helper, and exactly one public canonical
  `Tensor.averagePool2d(AveragePool2dAttrs)` receiver method.
- Successful construction performs one final `TensorFactory.createDerived` call, preserving exact
  type, batch/channel Dimensions, gradient request, and one-input/output-zero provenance while
  deriving checked static or canonical symbolic spatial dimensions.
- Focused tests lock the exact kind/attrs/signature surface, canonical receiver spelling, public
  method count 179, static/symbolic literal floor/ceil geometry, local failures and identifier
  effects, metadata, freshness, and provenance without value evaluation.
- The independent documentation pass finalized the affected production Javadocs, Tensor and
  Compile API references, glossary, capabilities, task, master plan, and roadmap without changing
  executable Java behavior.

## Completion summary

- Completed changes: implemented fixed-count NCHW average-pooling semantics, dedicated geometry
  attributes, exact static/symbolic result Shape derivation, one canonical public expression,
  validation, metadata/provenance construction, tests, public-surface locks, Javadocs, API and
  glossary documentation, and planning synchronization.
- Files changed or created: exactly the authorized 18 paths: four production, seven tests, and
  seven documentation/planning paths.
- Tests and validation: focused and final model test evidence passed in the implementation
  context; final documentation validation is recorded above.
- Documentation-agent review: clean context
  `/root/task_0020a1_implementation/documentation_finalization` independently finalized the
  required documentation and Javadocs while reusing Java evidence.
- Documentation impact: Tensor and Compile APIs, glossary, capabilities, task, master plan, and
  roadmap finalized; Runtime/Training and related contracts remain accurate unchanged for the
  reasons recorded above.
- Javadoc review: all four affected production types/members reviewed and finalized.
- Glossary impact: the existing pooling entry now defines count-padding, fixed divisor,
  accumulation domain, and valid-sample contrast.
- Unresolved issues: None.
- Follow-up required: None for task 0020A1; tasks 0021–0024 remain Draft, and later compiler,
  backend, runtime, conformance, and integration work owns the documented boundaries.

Status: Complete
