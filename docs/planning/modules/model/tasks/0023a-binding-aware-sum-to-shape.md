# Task 0023A: Binding-Aware Sum-to-Shape

## Status

Complete

## Goal

Add one backend-independent public reduction form that sums a numeric Tensor to an exact target
`Shape`. The target Shape describes the result directly, while future binding determines which
right-aligned axes were singleton-expanded and therefore require summation.

```text
numeric input + target Shape
  -> AggregateReductionKind.SUM + SumToShapeAttrs(targetShape)
     -> exact target Shape, type, eligibility, and one-input provenance
        -> later proof: every aligned target extent is one or equals the input extent
```

This is a generally useful public transformation, not a MATMUL-, attention-, gradient-, or
backend-specific unbroadcast operation. It closes only the model-vocabulary gap proven by the
[adjoint expressibility audit](../adjoint-expressibility-audit.md); compiler and later autograd
work remain separately owned.

## Scope

- Add public immutable `SumToShapeAttrs` with exactly one `Shape targetShape` component.
- Extend only `AggregateReductionKind.SUM` with the exact one-input/one-output
  `SumToShapeAttrs` signature; add no operation kind or enum constant.
- Add exactly one public Tensor method:

  ```java
  public Tensor sumToShape(Shape targetShape)
  ```

- Add package-private field-free `TensorSumToShapeExpressions` for deterministic validation,
  descriptor, operation, producer, and provenance construction.
- Accept current numeric types BFLOAT16, FLOAT32, FLOAT64, INT32, and INT64; reject BOOL.
- Define right-aligned target compatibility, leading-axis reduction, binding-dependent aligned
  reduction, exact target-Shape retention, numerical/empty behavior, metadata, failures,
  provenance, identifiers, and lifecycle boundaries.
- Add focused semantic/expression tests and change every exact public Tensor method-count inventory
  from 188 to 189.
- Finalize Javadocs, Tensor/Compile APIs, glossary, capabilities, and planning records through the
  required separate clean-context documentation pass.

## Out of scope

- a new `SUM_TO_SHAPE`, `UNBROADCAST`, `REDUCE_TO_SHAPE`, MATMUL-backward, attention-backward, or
  other operation kind
- changing the existing full-, single-axis, multi-axis, or masked SUM behavior, signatures,
  messages, Shapes, numerical policies, producer contracts, or public methods
- adding a target Tensor, axes, `keepDimensions`, raw `long...` target sizes, a public attrs
  overload, aliases, convenience names, builders, registries, or constraint objects
- changing `Shape`, `Dimension`, `DimensionExpression`, `DimensionExpressions`, `ShapeBroadcast`,
  `TensorDescriptor`, `TensorFactory`, `TensorProducer`, or `TensorProvenance`
- adopting this operation in MATMUL, attention, binary expressions, compiler-generated adjoints,
  graph capture, canonicalization, or optimization
- symbolic binding, concrete-axis resolution, runtime Shape values, execution-time validation,
  value aggregation, algorithms, accumulation buffers, kernels, lowering, fusion, or backend
  support
- BOOL reduction, mean-to-Shape, product-to-Shape, arbitrary reshape, broadcasting the input to a
  larger Shape, or changing data type during the operation
- gradients, autograd traversal, compiler, planning, prepare, runtime, engine, training, another
  module, dependencies, Gradle, architecture documents/tests, conformance, integration, or later
  tasks

## Exact semantic and public contract

### Existing SUM kind and new attributes

`AggregateReductionKind` retains its exact existing enum constants and order. No new kind is
introduced. Its signature variants become:

```text
SUM:
  NoOperationAttrs                         1 input, 1 output
  AxisReductionAttrs                       1 input, 1 output
  MultiAxisReductionAttrs                  1 input, 1 output
  MaskedReductionAttrs                     2 inputs, 1 output
  SumToShapeAttrs                          1 input, 1 output

MEAN:
  NoOperationAttrs                         1 input, 1 output
  AxisReductionAttrs                       1 input, 1 output
  MultiAxisReductionAttrs                  1 input, 1 output
  MaskedReductionAttrs                     2 inputs, 1 output
```

Every other aggregate signature remains unchanged. The new SUM signature is appended after all
existing SUM variants so existing declaration order remains stable. `MEAN` and every other kind
must reject `SumToShapeAttrs` through current exact attributes-class matching.

`SumToShapeAttrs` is a public record in the reduction package with exactly:

```java
Shape targetShape
```

Its canonical constructor performs exactly:

```java
targetShape = Objects.requireNonNull(targetShape, "targetShape");
```

The exact immutable Shape reference is retained. The record performs no input-aware
compatibility, axis, binding, type, descriptor, provenance, compiler, or execution validation.
It deliberately does not reuse layout-family `TargetShapeAttrs`: reshape/expand target geometry
and reduction-to-Shape semantics remain separate typed operation variants.

### Public method and helper surface

`Tensor` adds exactly:

```java
public Tensor sumToShape(Shape targetShape)
```

The method delegates exactly once to `TensorSumToShapeExpressions.apply(this, targetShape)` and
does not compose ordinary public `sum`, `reshape`, or another Tensor expression.

The package-private helper is one final non-record class with no fields, no nested types, one
private zero-argument constructor, and exactly these methods:

```java
static Tensor apply(Tensor input, Shape targetShape)
private static void validateInput(Tensor input)
private static void validateCompatibility(Shape inputShape, Shape targetShape)
private static Tensor create(Tensor input, Shape targetShape)
```

It adds no reusable public validator, Shape utility, binding object, or factory overload.

### Validation order and diagnostics

`apply` performs local validation in exactly this order:

1. `Objects.requireNonNull(input, "input")`;
2. `Objects.requireNonNull(targetShape, "targetShape")`;
3. read and validate the input data type;
4. read the exact input Shape;
5. reject target rank greater than input rank;
6. inspect right-aligned target axes in increasing target-axis order and reject the first
   statically incompatible pair;
7. construct attrs, descriptor, operation, producer, provenance, and Tensor.

BOOL fails before Shape compatibility with:

```text
input must have a numeric data type for SUM, but was BOOL
```

A target rank greater than input rank fails with:

```text
sumToShape target rank must not exceed input rank: input=<inputRank>, target=<targetRank>
```

For target axis `t`, the aligned input axis is:

```text
i = inputRank - targetRank + t
```

If both aligned Dimensions are static, the pair is locally compatible exactly when their extents
are equal or the target extent is one. The first incompatible pair fails with:

```text
sumToShape incompatible dimension at target axis <t> (input axis <i>): input=<inputDimension>, target=<targetDimension>
```

An equal Dimension pair, a static target singleton, or any pair involving an unresolved Dimension
is accepted locally. The latter records no separate constraint object; the operation's source and
target Shapes carry the later compatibility obligation. All local failures occur before factory
delegation and consume no Tensor identifier.

### Right-aligned reduction and binding obligation

At concrete binding, target rank must remain at most input rank. Every leading input axis without
an aligned target axis is reduced and removed. For each aligned pair:

- equal bound extents preserve the coordinate without reduction;
- target bound extent one accepts any non-negative input extent, sums that input axis, and retains
  one target position; and
- every other pair is invalid and must be rejected before execution.

For example:

```text
input Shape  = [2, 3, 4]
target Shape =    [3, 1]

input axis 0 has no target axis -> reduce and remove
input axis 1 equals target 3    -> preserve coordinates
input axis 2 maps to target 1   -> reduce and retain one position
result Shape                     -> [3, 1]
```

For a binding-dependent example:

```text
input Shape  = [8, 4]
target Shape = [X, 4]
```

Construction accepts the expression. Later binding must prove `X == 1` or `X == 8`. `X == 1`
reduces input axis zero; `X == 8` preserves it; any other value is invalid. Two unequal unresolved
aligned Dimensions are also accepted because the exact target Shape is already known and later
binding can prove target-one-or-equal without choosing a result Dimension during model
construction.

Scalar target Shape reduces every input axis. Scalar input accepts only scalar target. Equal
input and target Shapes are valid and still create a fresh explicit SUM occurrence rather than
returning the input or canonicalizing the expression.

### Numerical and empty-domain semantics

For each target coordinate, the result sums exactly the source coordinates selected by the rules
above. The operation inherits ordinary `AggregateReductionKind.SUM` semantics without defining a
new accumulation policy:

- INT32 and INT64 retain their input type and use fixed-width modular addition;
- floating values retain input type and the current SUM NaN, infinity, signed-zero, reassociation,
  and rounding contract;
- an actually reduced empty source domain produces the current SUM empty identity, numeric
  positive zero; and
- a coordinate with no reduced axis is the corresponding input value, not an arithmetic rewrite.

Target Shape may contain zero, named dynamic, or expression Dimensions. A static source zero
aligned to target one is valid and describes an empty sum yielding one zero result at that target
coordinate. A target zero aligned to source zero is equal and describes no output coordinates.
No element count is calculated during construction.

### Descriptor, producer, provenance, and identifiers

Success creates in this order:

1. one `SumToShapeAttrs` retaining the exact caller target Shape;
2. one unresolved `TensorDescriptor` with exact input data type, exact target Shape reference, and
   unchanged `requiresGrad` metadata;
3. one `Operation(AggregateReductionKind.SUM, attrs)`;
4. one producer with exact ordered inputs `[input]` and one output descriptor;
5. output-index-zero provenance and one fresh, unlabeled, storage-free Tensor.

The input Tensor is unchanged. The result has no resolved layout even for equal Shapes, no label,
no storage, and no additional output. Every successful call creates a fresh producer and consumes
one Tensor identifier. Current identifier-exhaustion behavior remains unchanged and no partial
state is rolled back.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Capabilities](../capabilities.md)
- [Master plan](../master-plan.md)
- [Adjoint expressibility audit](../adjoint-expressibility-audit.md)
- [Task 0023](0023-adjoint-expressibility-audit.md)
- [Shape and dimension model](0002-shape-and-dimension-model.md)
- [Operation hardening](0018k-operation-signature-and-construction-hardening.md)
- [Shared producer provenance](0018l-shared-multi-output-tensor-provenance.md)
- [Symbolic extent expressions](0018m-symbolic-extent-expressions.md)
- [Integral reductions](0018u1-integral-reductions-and-arg-min-normalization.md)
- [Multi-axis reductions](0018v-multi-axis-and-statistical-reductions.md)
- [MATMUL](0019-matmul-semantics-and-tensor-expression.md)
- [Scaled dot-product attention](0019e-scaled-dot-product-attention.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Work stays inside model-owned backend-independent operation semantics, Tensor metadata,
  pre-capture producer/provenance construction, and directly affected docs/planning files.
- `AggregateReductionKind.SUM` remains semantic metadata and exposes no backend support, bound
  axis set, algorithm, kernel, execution route, or runtime state.
- Model construction may reject provable static incompatibility and retain unresolved
  target-one-or-equal obligations. Compiler or later concrete binding owns proof before execution.
- Compiler later owns adjoint graph construction and may emit this public semantic variant;
  this task implements neither compiler use nor gradient rules.
- Runtime hot paths must not consume `Operation` or `CompiledNode`; backend prepare owns lowering,
  specialization, fusion, and concrete reduction implementation.
- No architecture, ADR, architecture-test, dependency, Gradle, cross-module, compiler, runtime,
  prepare, backend, training, conformance, or integration change is authorized. Stop if the
  existing Shape, signature, producer, or descriptor foundation cannot express this contract.

## Package impact

Existing packages changed:

- `io.github.pho001.synaptik.model.operation.reduction`
- `io.github.pho001.synaptik.model.tensor`

No package is added, moved, or renamed.

Type placement:

- `...operation.reduction.AggregateReductionKind` owns the existing SUM semantic identity and its
  exact structural variants.
- `...operation.reduction.SumToShapeAttrs` owns the intrinsic immutable target Shape for the new
  SUM variant.
- `...tensor.TensorSumToShapeExpressions` owns package-private operand validation and expression
  construction.
- `...tensor.Tensor` remains the public fluent facade.

## Affected files

Production (4):

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/reduction/AggregateReductionKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/reduction/SumToShapeAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorSumToShapeExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests (14):

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationSignatureTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/reduction/ReductionSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorSumToShapeExpressionTest.java`
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

The eleven existing Tensor tests change only the exact global method inventory from 188 to 189
and, for `TensorTest`, add the exact `sumToShape(Shape)` method to the public surface lock. They
receive no unrelated changes.

Documentation/planning (7):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless inaccurate: current Shape/Dimension/ShapeBroadcast, descriptor, operation,
factory, producer/provenance, ordinary reduction, MATMUL, attention, Training/Runtime APIs,
architecture/ADRs/tests, conformance/integration, Gradle, dependencies, and other modules.

## Maximum scope

Exactly 25 paths: four production, fourteen tests, and seven documentation/planning paths. This
cohesive public capability exceeds the usual 18-path guardrail under the user's standing automatic
higher-path authorization because one semantic variant, one facade method/helper, eleven existing
global Tensor-count locks, the repository-wide operation-signature matrix, focused coverage, and
mandatory documentation must change atomically. The signature matrix was added after the first
final model run proved that every new family signature must update this completed global lock.
Stop for path 26, another public type/method, another helper/test/document, a foundation edit,
compiler adoption, later-task work, cross-module work, architecture/Gradle change, or unrelated
cleanup. If live repository evidence changes another inventory path, update this task before
implementation and stop pending renewed authorization above 25.

## Javadoc and documentation requirements

- Add complete Javadocs for `SumToShapeAttrs`, the SUM variant, helper, helper methods, and public
  Tensor method, including target/source alignment, dynamic obligation, numerical/empty behavior,
  metadata, provenance/IDs, failures, and lifecycle ownership.
- Apply General, API/Javadoc, Planning, and Example profiles as relevant, with complete `@param`,
  `@return`, and expected `@throws` text.
- Tensor API moves binding-aware sum-to-Shape from planned to current, adds static and dynamic
  examples, exact public signature, Shape/type rules, producer metadata, and lifecycle boundary.
- Compile API records current model metadata and the later binding-proof obligation; it must not
  claim compiler capture, autograd use, lowering, or execution is implemented.
- Glossary adds or updates `sum-to-Shape`, distinguishing it from arbitrary reshape, ordinary
  fixed-axis SUM, broadcasting, and compiler-generated adjoint use.
- Capabilities/task/master/roadmap remain synchronized: 0023 and 0023A become/stay Complete only
  after implementation, 0023B is the next Draft frontier without a detailed specification, and
  0023C–0024 remain Draft.
- Review the official PyTorch
  [`sum_to_size`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.sum_to_size.html)
  page only as terminology/comparison evidence, not as design authority; repository contracts
  remain primary.
- Record reasoned no-change conclusions for related existing Javadocs, Training/Runtime APIs,
  architecture/ADRs/tests, dependencies, Gradle, conformance/integration, other modules, and later
  tasks.

## Acceptance criteria

- No new operation kind or enum constant; exact existing `AggregateReductionKind` order remains.
- SUM alone appends exact fixed `SumToShapeAttrs` 1/1 signature; MEAN and every other signature
  remain exact and reject the new attrs class.
- `SumToShapeAttrs` has exactly one non-null `Shape targetShape` component, explicit documented
  accessor, exact reference retention, generated record value semantics, and no extra API/state.
- Exactly one new public `sumToShape(Shape)` method; public Tensor count is 189; no alias, raw-size,
  axes, keep-dimension, attrs, or target-Tensor overload.
- Helper has the exact field-free four-method surface and private constructor described above.
- Exact null/type/rank/aligned-static validation order, exception types/messages, and no-ID local
  failures.
- Exact right-aligned semantics: leading input axes reduce, equal bound aligned axes preserve,
  target-one axes reduce and retain, and every other concrete pair is invalid.
- Equal, singleton, zero, scalar, lower-rank, named/expression, unresolved/static, and
  unresolved/unresolved cases follow the selected contract and retain exact target Shape.
- Numeric domain and SUM numerical/empty semantics remain exact; BOOL is rejected; no type
  promotion or value/storage inspection occurs.
- Result has exact input type and eligibility, exact target Shape reference, unresolved layout,
  no label/storage, exact SUM/attrs, ordered `[input]` producer, output index zero, one fresh ID,
  and unchanged input.
- Exact 25-path/package scope; no changes to completed reduction behavior, ShapeBroadcast,
  MATMUL/attention helpers, compiler, execution layers, architecture, build, or later tasks.
- Separate clean documentation-focused pass and all required validation/evidence complete before
  status Complete.

## Tests / validation

Focused implementation command:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest \
  --tests io.github.pho001.synaptik.model.operation.reduction.ReductionSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorSumToShapeExpressionTest \
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
  --tests io.github.pho001.synaptik.model.tensor.TensorScaledDotProductAttentionExpressionTest
```

After executable Java stabilizes, run exactly one final model suite:

```bash
./gradlew :modules:model:test
```

Focused coverage must verify exact attrs/signature shape and rejection, public/helper surfaces,
static leading and aligned reductions, equal-shape freshness, scalar target/input, zero extents,
target reference retention, all accepted unresolved pair categories, deterministic first static
mismatch, BOOL/rank/null failures, no-ID local failures, exact descriptor, operation, producer,
provenance, storage/label absence, input immutability, and public count 189. Tests inspect metadata
only and do not claim value execution.

Documentation pass after final Javadocs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

Also validate Markdown links/anchors/fences/final newlines/trailing whitespace, official reference,
examples, exact 25 paths, package placement, signatures/count 189, synchronized status, 0023B
Draft/no spec, and absence of compiler/runtime/backend/build changes. Reuse successful Java
evidence unless executable Java changes.

Repository-wide validation is deferred to the capability checkpoint after task 0023F and to CI;
this single-module task changes no dependency, build, or architecture boundary.

## Dependencies

- 0001–0002: numeric types and immutable static/symbolic Shape/Dimension contracts.
- 0005–0007, 0011–0013: typed operations, descriptors, Tensor, identity, factory, and provenance.
- 0016A–0016D and 0018V: SUM family, numerical/empty policy, axis and multi-axis reduction
  construction.
- 0018K: exact family-owned signatures; 0018L: producer/output-index provenance.
- 0018M–0018M1: symbolic extent expressions retained without model-side binding.
- 0018U–0018U1: selected signed-integral arithmetic/reduction semantics.
- 0019 and 0019E: current deferred singleton-or-equal MATMUL/attention batch obligations.
- 0023: completed adjoint expressibility audit selecting this exact general public gap.

All dependencies are Complete.

## Follow-up tasks

- 0023B remains the next Draft frontier for Gather-compatible axis scatter-add; do not create its
  detailed specification during this task.
- 0023C–0023F remain concise Draft public-capability rows.
- Compiler/autograd work later consumes this semantic variant and owns target binding proof,
  gradient construction, canonicalization, and optimization.
- Task 0024 remains Draft and depends through completed 0023F.

## Architecture impact

Expected impact: None. Stop if implementation needs a new operation kind, Shape constraint type,
runtime Shape value, compiler/prepare/backend contract, dependency, architecture update, or work
outside the exact model and documentation scope.

## Implementation prompt

Use this prompt in a separate clean-context task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/developer-guide/documentation-rules.md,
docs/planning/planning-guide.md, docs/planning/modules/model/capabilities.md, the model master plan,
roadmap, completed tasks 0016A–0016D/0018K–0018M1/0018U–0019E/0023, and task 0023A in full.
Inspect every affected and review-only source/test named by task 0023A.

Implement task 0023A exactly inside its 25 authorized paths. Add binding-aware sum-to-Shape only
as a new exact AggregateReductionKind.SUM attribute variant and one public Tensor expression; add
no kind, alias, Shape/binding foundation, compiler adoption, execution behavior, or later task.
Stop on architecture, dependency, completed-contract, validation-order, affected-file, or maximum-
scope conflict.

Run focused tests while developing and exactly one final model suite after executable Java
stabilizes. Then hand the actual diff and exact Java evidence to a separate clean-context
documentation-focused agent in the same overall change. That agent must independently inspect
final source/tests, finalize Javadocs, Tensor/Compile APIs, glossary, capability/task/master/roadmap
status and documentation validation, and reuse successful Java evidence unless executable behavior
changes or it records a concrete reason.

Do not mark 0023A Complete until both passes and every acceptance criterion succeed. Leave 0023B
and every later task Draft without a detailed specification.
```

## Documentation-agent handoff

The implementation agent must hand over this task, the actual implementation/test diff, exact
Java evidence, SUM signature change, target-Shape/right-aligned/binding semantics, validation and
ID order, numeric/empty policy, metadata/provenance behavior, architecture constraints, expected
Tensor/Compile API and glossary changes, official comparison link, existing-Javadoc review list,
and every documentation/scope/status command. The documentation agent must inspect final source
and tests rather than rely on the summary.

## Local decisions

- Reused `AggregateReductionKind.SUM` and appended `SumToShapeAttrs` after its four existing
  signature variants; no new kind or enum constant was needed.
- Kept `SumToShapeAttrs` to exactly one non-null `Shape targetShape` component. Input-aware
  compatibility remains in the field-free package-private Tensor helper.
- Accepted every aligned pair involving an unresolved Dimension. Construction rejects only a
  fully static pair that is neither equal nor target-one; later binding validation owns proof.
- Added `OperationSignatureTest` to the authorized scope after the first final model run exposed
  its repository-wide family-signature matrix. This recorded correction changed the task from 24
  to exactly 25 paths before implementation continued.
- Used the official PyTorch `sum_to_size` page only to compare terminology. Synaptik retains its
  independently specified exact-Shape, metadata, validation, and lifecycle contract.

## Known limitations

- Model construction does not bind dimensions, resolve a concrete reduction-axis set, inspect
  values, aggregate, construct gradients, capture a graph, lower, or execute.
- Compiler adoption, target-one-or-equal proof at concrete binding, adjoint construction,
  canonicalization, and optimization remain future work.
- Runtime, prepare, engine, training, backend, conformance, and integration layers do not yet
  consume this semantic variant.
- Repository-wide validation remains deferred to the capability checkpoint after task 0023F and
  to CI, as planned for this single-module task.

## Validation evidence

- The initial 13-suite focused command passed 126 tests.
- The first final model run failed only because the completed global `OperationSignatureTest`
  matrix did not yet include the new SUM attributes class. After the affected-file specification
  was corrected and that exact lock was updated, the 14-suite focused command passed 131 tests.
- The replacement final `./gradlew :modules:model:test` passed 977 tests. Executable Java did not
  change after that run, so the independent documentation pass reused this evidence as required.
- Manual `javap`, reflection, import, and source inspection confirmed the exact helper/record/enum
  surfaces, signature order, one public `sumToShape(Shape)` method, single delegation, and public
  Tensor count 189.
- Independent clean-context Javadocs completed without changing executable Java behavior.
  `./gradlew :modules:model:javadoc` passed with `BUILD SUCCESSFUL`; generated public pages contain
  `SumToShapeAttrs`, SUM's target-Shape semantics, and `Tensor.sumToShape(Shape)` with parameters,
  result, failures, metadata, and lifecycle boundaries. The package-private helper remains
  intentionally absent from public generated Javadoc and was validated in source and bytecode.
- The exact Tensor API example compiled and ran under the repository's Java 26 model classes. It
  printed `Shape[3, 1]`, exact target/input reference checks `true`, output index `0`, dynamic
  `Shape[X, 4]`, and unresolved-layout `true`, matching the documented metadata output.
- The targeted Markdown validator passed all seven changed documentation/planning files, 717 local
  links, required anchors including the new binding-aware section, balanced fences, and final
  newlines. The official PyTorch page was reviewed only for the documented naming comparison.
- Exact-scope validation passed exactly 25 task paths plus the two declared untracked task-0023
  baseline artifacts. Status checks confirmed 0023 and 0023A Complete, 0023B–0023F and 0024 Draft,
  and no 0023B detailed specification. No compiler, runtime, training, backend, testing-layer,
  architecture, dependency, or build path changed. `git diff --check` passed.

## Implementation notes

- Implementation context `/root/implement_0023a` completed production and test work. Independent
  clean documentation context `/root/implement_0023a/docs_0023a` read the architecture,
  documentation profiles, planning contracts, audit, task, final source/tests, and affected APIs
  before finalizing Javadocs and documentation.
- Four production paths implement the semantic variant and public construction. Fourteen tests
  cover the exact signature, validation, unresolved/static Shape matrix, identity discipline,
  descriptor/producer/provenance metadata, helper/public surface, and eleven global count locks.
- Seven documentation/planning paths finalize Tensor and Compile APIs, glossary terminology,
  capability state, task evidence, model master plan, and roadmap.
- Related Shape/binding foundations, ordinary reductions, MATMUL/attention construction, Compile
  and Training public contracts, Runtime API, architecture/ADRs/tests, dependencies, Java 26
  Gradle configuration, conformance/integration suites, other modules, and later tasks were
  reviewed and remain unchanged because this task adds only model-owned semantic/expression
  metadata and crosses no dependency or execution boundary.

## Completion summary

- Completed changes: added binding-aware exact-target-Shape SUM metadata and one public
  `sumToShape(Shape)` expression with deterministic static validation, deferred unresolved binding
  obligations, exact descriptor/provenance construction, and no new operation kind.
- Files changed or created: exactly 25 authorized task paths (four production, fourteen tests, and
  seven documentation/planning paths), distinct from the two declared task-0023 baseline files.
- Tests and validation: reused the passed 131-test focused and 977-test final model evidence;
  final model Javadoc, generated pages, runnable Java 26 metadata example, Markdown links/anchors/
  fences/newlines, package/signature/helper/public-count/status/no-later-spec, exact-scope, and
  whitespace checks passed.
- Documentation-agent review: clean context `/root/implement_0023a/docs_0023a`; Complete.
- Documentation impact: all four affected production Javadocs, Tensor API, Compile API, glossary,
  capabilities, this task, model master plan, and roadmap are finalized. Training/Runtime APIs and
  architecture documentation require no change for the model-only reasons recorded above.
- Javadoc review: `AggregateReductionKind`, `SumToShapeAttrs`,
  `TensorSumToShapeExpressions`, and `Tensor.sumToShape(Shape)` are complete and accurate. Related
  Shape, descriptor, factory, producer/provenance, ordinary reduction, MATMUL, and attention
  Javadocs remain accurate without modification.
- Glossary impact: sum-to-Shape is current and distinguished from reshape, fixed-axis SUM,
  broadcasting, and possible future compiler-generated adjoint use.
- Unresolved issues: None within task scope. Binding proof, gradients, capture, lowering,
  execution, and backend support are intentional future-layer limitations.
- Follow-up required: Draft task 0023B and later planned work only; no follow-up is required to
  complete task 0023A.

Status: Complete
