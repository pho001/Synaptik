# Task 0022: Mean-Squared-Error Loss

## Status

Complete

## Goal

Add the first intentionally selected loss operation: one backend-independent mean-squared-error
(MSE) meaning and one public Tensor expression with explicit `NONE`, `SUM`, or `MEAN` reduction.
The task establishes reusable typed loss-reduction vocabulary without implementing categorical
losses, gradients, training coordination, compiler behavior, or execution.

Mental model:

```text
prediction + exact-shape target + explicit reduction
  -> one loss producer
     -> elementwise squared error or one reduced scalar
```

## Scope

- Add public `LossKind.MEAN_SQUARED_ERROR` in a focused model-owned loss package.
- Add public `LossReduction` with exactly `NONE`, `SUM`, and `MEAN`, in that order.
- Add immutable public `MeanSquaredErrorAttrs(LossReduction reduction)`.
- Give the operation exactly one fixed two-input/one-output signature.
- Add exactly this fluent receiver method:

  ```java
  public Tensor meanSquaredError(Tensor target, LossReduction reduction)
  ```

- Add one package-private, final, field-free `TensorLossExpressions` construction helper.
- Fix ordered roles, Shape/type validation, result metadata, formula, denominator and empty-domain
  policy, special-value classes, provenance, identifier effects, diagnostics, and validation order.
- Finalize affected Javadocs, Tensor and Compile API references, glossary impact, and planning
  records in the mandatory clean documentation-focused handoff.

## Out of scope

- dense-target or index-target categorical cross-entropy, caller-supplied probability or
  log-probability losses, negative-log-likelihood as a separate operation, binary cross-entropy,
  Kullback–Leibler divergence, margin/ranking, connectionist temporal classification, or any other
  loss
- class axes, class/sample/element weights, masks, ignore index, label smoothing, normalization
  factors other than the selected logical element count, or a loss registry/options framework
- broadcasting, implicit expansion, implicit casts, nullable/default reduction, a default-mean
  overload, static facade, alias, primitive reduction spelling, or eager constants
- value evaluation, storage reads, tolerances, numerical algorithm/pass/tree choice, gradients,
  adjoints, autograd traversal, compiler capture/validation/decomposition/fusion, backend support,
  lowering, kernels, prepare, runtime, publication, or execution
- parameter/module/optimizer/training-session/step/checkpoint ownership or mutation
- integral, BOOL, FLOAT16, complex, sparse, quantized, or unsigned prediction/target/result data
- changes to existing arithmetic, reduction, Shape, promotion, Tensor factory, producer,
  provenance, compiler/runtime/training APIs, architecture, dependencies, Gradle, or another module
- detailed specifications or implementation for tasks 0022A–0022B or 0023–0024

## Public and operation contracts

### Kind, typed attributes, and signature

`LossKind` contains exactly `MEAN_SQUARED_ERROR`. Its exact ordered signature is:

```java
OperationSignature.fixed(MeanSquaredErrorAttrs.class, 2, 1)
```

One occurrence has ordered inputs `[prediction, target]` and ordered output `[loss]`. The receiver
is prediction at input position zero. There is no hidden input, saved output, sibling value, or
variable cardinality.

`LossReduction` is a public immutable enum vocabulary with exactly:

```java
NONE
SUM
MEAN
```

The enum does not itself implement `OperationAttrs`; it is a typed configuration component reused
by family-specific immutable attributes. It carries no string parser, default, denominator,
axis, mask, ignore value, Tensor, or executable behavior.

`MeanSquaredErrorAttrs` has exactly one component:

```java
LossReduction reduction
```

Its compact constructor null-checks the component with `NullPointerException("reduction")` and
retains the exact enum reference. The record contains no input, output Shape, denominator, data
type, algorithm, gradient, graph, compiler, backend, or runtime state. No `ScalarValue` is needed:
this operation has no scalar semantic parameter beyond the typed reduction enum.

### Public receiver and target Shape

The public surface is exactly:

```java
public Tensor meanSquaredError(Tensor target, LossReduction reduction)
```

Prediction and target must have equal rank and positionally compatible Dimensions. For each axis:

- structurally equal Dimensions pass;
- unequal static Dimensions fail locally;
- a non-structurally-equal pair involving an unresolved Dimension is accepted with an equality
  obligation for later compiler capture/binding; and
- no singleton, scalar, leading-axis, right-aligned, or other broadcasting is accepted.

This exact-shape target policy prevents silent replication from changing the selected `MEAN`
denominator. When `NONE` is selected, the result retains the exact prediction Shape reference.
The target Shape object is never substituted into the result. When `SUM` or `MEAN` is selected,
the result uses the shared `Shape.scalar()` reference.

Exact Shape failures are:

```text
meanSquaredError target rank must equal prediction rank: prediction=<predictionRank>, target=<targetRank>
meanSquaredError target dimension mismatch at axis <axis>: prediction=<predictionDimension>, target=<targetDimension>
```

### Formula, reduction, denominator, and empty domains

After converting both input values to the selected result type and computation format, the
unreduced loss at logical coordinate `i` is:

```text
d_i = prediction_i - target_i
loss_i = d_i * d_i
```

Let `E` be the complete logical element count of the compatible prediction/target Shape after all
unresolved Dimensions are bound:

```text
NONE: output_i = loss_i                         Shape = prediction Shape
SUM:  output   = sum_i(loss_i)                  Shape = scalar
MEAN: output   = sum_i(loss_i) / E              Shape = scalar
```

`E` includes every logical coordinate and is not a batch-only or leading-axis count. A scalar has
`E = 1`. A Shape with any zero extent has `E = 0`; no elementwise subtraction or square is then
evaluated. `NONE` returns an empty result, `SUM` returns positive zero, and `MEAN` returns NaN for
the zero-over-zero empty mean. Construction does not need a static element count and does not
store `E` in attributes. Dynamic empty/non-empty selection is an execution-time consequence of
the bound Shape, not a model-construction branch.

For prediction `[1, 2, 4]` and target `[1, 4, 1]`, the unreduced losses are `[0, 4, 9]`, `SUM` is
`13`, and `MEAN` is `13 / 3`. This is a mathematical illustration; Tensor construction reads no
values. The explicit reduction vocabulary is comparable to official
[PyTorch MSE loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html),
but this task deliberately selects no default, weighting, or broadcast behavior.

### Data types, result type, and accumulation

Prediction and target must each be BFLOAT16, FLOAT32, or FLOAT64. Derive the result type exactly
once with:

```java
DataTypePromotion.promoteFloating(predictionType, targetType)
```

Both logical operands participate in arithmetic as that promoted type. Every reduction mode
returns that same result type; `MEAN` does not widen the public result. No cast producer or hidden
conversion Tensor is inserted.

BFLOAT16 and FLOAT32 results perform subtraction, multiplication, summation, and division in
FLOAT32. FLOAT64 results perform them in FLOAT64. Final values are rounded to the result format.
Equal-or-wider intermediates, stable summation, compensation, vectorization, parallelization,
fused operations, and reassociation are allowed when later conformance tolerances and the exact
special-value classes below are preserved. Narrower computation, a fixed traversal/tree, bitwise
identity, NaN payload/sign preservation, and identical finite rounding across backends are not
promised.

Exact type failures, checked in input order, are:

```text
meanSquaredError prediction must have a floating data type, but was <dataType>
meanSquaredError target must have a floating data type, but was <dataType>
```

### NaN, infinity, signed zero, overflow, reassociation, and determinism

Construction reads no values. For every evaluated element and reduction:

- a NaN prediction or target produces NaN squared error;
- equal-sign infinities subtract to NaN and therefore produce NaN squared error;
- one infinity and one finite value, or opposite-sign infinities, produce positive infinity after
  squaring;
- every exact zero difference, including results from any signed-zero pairing, squares to positive
  zero;
- finite subtraction or squaring may overflow to positive infinity or underflow to positive zero
  in the selected computation format;
- `SUM`/`MEAN` propagates any NaN; otherwise any positive infinity makes the reduced result
  positive infinity; an all-finite exact-zero domain reduces to positive zero before mean
  division; and
- empty `SUM` is positive zero while empty `MEAN` is NaN as specified above.

The abstract formula is non-negative for every non-NaN element. An implementation must not use an
algebraic expansion such as `p*p - 2*p*t + t*t` when it changes the required infinity/NaN classes.
Reduction traversal is not fixed, and permitted reassociation may change finite rounding within
later tolerance. Separately constructed equal requests remain distinct producers; there is no
interning, identity determinism, or bitwise-result guarantee.

### Result metadata, layout, gradients, provenance, and identifiers

The result descriptor uses the promoted result type, selected Shape, unresolved layout, and
`requiresGrad = prediction.requiresGrad() || target.requiresGrad()`. This eligibility is metadata
only and does not define or promise a gradient rule. The result is fresh, unlabeled, and
storage-free. Neither input nor its descriptor, label, storage, gradient flag, ID, Shape,
provenance, or layout is mutated.

Every success creates one `MeanSquaredErrorAttrs`, one `Operation`, one `TensorProducer`, one
output descriptor, one provenance value at output index zero, one Tensor wrapper, and one Tensor
ID. Provenance records exact ordered inputs `[prediction, target]`. No primitive arithmetic or
reduction producers are decomposed beneath the loss occurrence.

Every known local validation failure occurs before factory delegation and consumes no Tensor ID,
producer, provenance, or wrapper. Factory identifier exhaustion retains the existing single-output
factory behavior; no rollback or new identifier policy is introduced.

### Validation and construction order

`TensorLossExpressions.meanSquaredError(...)` performs exactly:

1. null-check prediction, target, then reduction;
2. validate prediction floating type, then target floating type;
3. require equal rank;
4. validate static/deferred Dimension compatibility in increasing logical-axis order;
5. promote prediction and target types in occurrence order;
6. construct `MeanSquaredErrorAttrs`;
7. select the exact prediction Shape for `NONE` or shared `Shape.scalar()` for `SUM`/`MEAN`;
8. construct the unresolved result descriptor, `Operation`, and delegate exactly once to the
   single-output derived Tensor factory path.

Null messages are `prediction`, `target`, and `reduction`. The public receiver cannot be null at a
normal Java instance call, but the package-private helper retains explicit prediction validation
and tests lock it. The attributes constructor defensively repeats reduction null validation.
Existing promotion, descriptor, operation, producer, and identifier failures retain their current
messages.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Capabilities](../capabilities.md)
- [Master plan](../master-plan.md)
- [Operation signatures](0018k-operation-signature-and-construction-hardening.md)
- [Shared producer provenance](0018l-shared-multi-output-tensor-provenance.md)
- [Typed scalar contract](0018n-typed-scalar-value-contract.md)
- [Multi-axis and statistical reductions](0018v-multi-axis-and-statistical-reductions.md)
- [Softmax semantics](0016i-softmax-semantic-kinds-and-attributes.md)
- [Softmax Tensor expressions](0016j-softmax-tensor-expressions.md)

## Architecture constraints

- `modules/model` owns only backend-independent loss semantics, Tensor metadata, and immutable
  pre-capture provenance. Tensor remains public mutable API state and is not graph IR.
- Operation types may consume foundational data-type values but must not import Tensor, compiler,
  training, runtime, prepare, engine, graph state, or backend types.
- Compiler owns graph capture, proof of deferred Shape equality, gradients, autograd/backward
  construction, decomposition, canonicalization, and optimization.
- Backend prepare owns algorithm, lowering, specialization, fusion, kernel, and tolerance
  satisfaction. Runtime executes already-prepared work and must not consume `Operation` in its hot
  path.
- The training extension may later consume loss outputs but does not own this model semantic
  operation. This task adds no session, optimizer, parameter, or state dependency.
- No architecture, explanatory architecture, ADR, dependency, Gradle, cross-module, conformance,
  or integration change is authorized. Stop if the contract cannot be represented by the existing
  single-output producer and descriptor foundations.

## Package impact

Existing packages used and changed:

- `io.github.pho001.synaptik.model.tensor`
- `io.github.pho001.synaptik.model.datatype`
- `io.github.pho001.synaptik.model.shape`
- `io.github.pho001.synaptik.model.operation`

Package added:

- `io.github.pho001.synaptik.model.operation.loss` — cohesive public loss semantics and typed
  reduction/configuration values. It is distinct from aggregate reduction because a loss combines
  ordered prediction/target roles before optionally reducing.

Type placement:

- `...operation.loss.LossKind` — family-owned semantic identity and signature.
- `...operation.loss.LossReduction` — shared public loss-only reduction vocabulary.
- `...operation.loss.MeanSquaredErrorAttrs` — immutable MSE reduction parameters.
- `...tensor.TensorLossExpressions` — package-private validation, promotion, Shape, descriptor,
  and provenance construction owner, deliberately named for reuse by tasks 0022A–0022B without
  pre-implementing them.
- `...tensor.Tensor` — established public fluent receiver facade.

Tests mirror production packages where package-private helper access is needed.

## Affected files

Production (5):

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/loss/LossKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/loss/LossReduction.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/loss/MeanSquaredErrorAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorLossExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests (10):

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/loss/LossSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMeanSquaredErrorExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBatchNormInferenceExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLayerNormExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLinearExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorRmsNormExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressionTest.java`

The eight existing inventory tests change only their exact public Tensor method count from 185 to
186. They must not receive unrelated assertion or behavior changes.

Documentation/planning (7):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless inaccurate: Runtime and Training APIs; related arithmetic, reduction,
Shape, promotion, signature, factory, producer, provenance, normalization, and indexing contracts;
architecture/ADRs/tests; conformance/integration; Gradle; dependencies; and other modules.

## Maximum scope

Exactly 22 actual paths: five production, ten tests, and seven documentation/planning paths. This
exceeds the planning guide's normal 18-path guardrail under the user's standing authorization
because the cohesive capability necessarily adds three public loss semantic types, one bounded
construction helper, one Tensor method, two focused tests, updates all eight established exact
public-count inventories atomically, and completes the mandatory seven-path documentation handoff.
Splitting those inventory or documentation updates would leave the public API locks broken or the
capability incomplete. Stop for path 23, another public method/type, another test/document, a
factory/foundation modification, any task-0022A+ implementation, cross-module work, architecture or
Gradle change, or unrelated cleanup.

## Javadoc and documentation requirements

- Fully document the kind, reduction enum, attributes, helper, and Tensor method: exact input
  roles/order, target Shape/no-broadcast policy, formulas, reductions, denominator, empty domains,
  types/computation, special values, metadata, IDs, failures, and lifecycle boundaries.
- Every method/constructor input, non-void result, and expected failure has meaningful `@param`,
  `@return`, and `@throws` text under the API/Javadoc profile.
- Tensor API adds the exact signature, reduction/output table, `[1,2,4]` versus `[1,4,1]` example,
  no-broadcast/deferred-Shape rules, computation and special values, provenance/IDs, and current
  model versus planned compiler/backend/runtime/training boundaries.
- Compile API records MSE as current model metadata only; capture, deferred equality proof,
  gradients, decomposition, validation, lowering, and execution remain planned.
- Review glossary terms loss, loss reduction, mean-squared error, prediction, target, and mean
  denominator; update only reusable distinctions.
- Record reasoned no-change conclusions for Runtime and Training APIs, related contracts,
  architecture/ADRs/tests, conformance/integration, Gradle, dependencies, and other modules.

## Acceptance criteria

- `LossKind` has exactly one kind and exact `MeanSquaredErrorAttrs` fixed 2/1 signature; incompatible
  attributes and occurrence counts fail through existing contracts.
- `LossReduction` has exactly `NONE`, `SUM`, `MEAN`; attrs retain the exact non-null value and no
  parser/default/options surface exists.
- Tensor exposes exactly one new public method with the signature in this task; public Tensor
  method count is exactly 186 and every existing inventory lock is synchronized.
- Ordered inputs are exactly `[prediction, target]`; result provenance uses output index zero and
  one fresh producer/ID on success.
- BFLOAT16/FLOAT32/FLOAT64 pairs promote through current floating promotion; all other inputs fail
  in exact order and no cast is inserted.
- Equal ranks and exact/static-or-deferred positional Dimension compatibility are enforced; target
  broadcasting is absent.
- `NONE` retains the exact prediction Shape; `SUM`/`MEAN` use `Shape.scalar()`; every layout is
  unresolved and gradient eligibility is the input logical OR.
- Formula, accumulation, denominator, scalar/empty behavior, NaN/infinity/signed-zero/overflow,
  reassociation, and determinism match this task without construction-time value reads.
- Null/type/rank/Dimension failures have exact messages and occur before factory delegation with no
  ID consumption.
- No gradient, compiler, backend, runtime, training-session, weights/mask/ignore/smoothing,
  categorical loss, registry, architecture, dependency, or Gradle behavior is added.
- The exact 22-path maximum and package placement hold.
- A separate clean documentation-focused agent pass finalizes affected Javadocs, Tensor/Compile
  APIs, glossary impact, capabilities/task/master/roadmap records, and documentation validation in
  the same overall change.
- Task 0022 becomes `Complete` only after implementation, focused/final validation, clean docs
  handoff, evidence, completion summary, and synchronized status. Tasks 0022A–0022B and 0023–0024
  remain Draft without detailed specifications.

## Tests / validation

Implementation-focused validation while developing:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.operation.loss.LossSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorMeanSquaredErrorExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorBatchNormInferenceExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorBinaryArithmeticTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorLayerNormExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorLinearExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorMatmulExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorRmsNormExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorScaledDotProductAttentionExpressionTest
```

Final Java validation after executable code stabilizes, exactly once:

```bash
./gradlew :modules:model:test
```

Documentation-focused pass after final Javadocs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

The documentation pass also runs targeted Markdown local-link/anchor/fence/final-newline/trailing-
whitespace checks; verifies the official PyTorch link and the numerical example; audits exact
22-path scope, package placement, exact public signature/count, no detailed 0022A–0024 specs, and
status synchronization; and records reused Java evidence. It does not rerun successful Java tests
unless it changes executable Java behavior or records a concrete risk.

Repository-wide validation is deferred to the model capability checkpoint after task 0022B and CI
because this is a single-module metadata-construction task with no dependency, architecture,
shared-build, or executable backend change.

## Dependencies

- 0001–0002: floating types, promotion, Shape, Dimension, static/symbolic extents, rank, and scalar
  Shape.
- 0005–0007 and 0011–0013: operation, signature, descriptor, Tensor, factory, and provenance
  foundations.
- 0018K: exact attrs-class signature and occurrence-cardinality validation.
- 0018L: unified producer and output-index provenance.
- 0018N: exact typed configuration precedent; this task confirms no scalar parameter is needed.
- 0018V: floating computation, reduction, empty-domain, special-value, and determinism precedent.

All dependencies are Complete.

## Follow-up tasks

- Draft task 0022A later owns dense-target categorical cross-entropy directly from logits with an
  explicit class axis and the shared loss reductions. It has no detailed specification yet.
- Draft task 0022B later owns INT32/INT64 index-target logits cross-entropy, class-axis removal,
  optional ignore index, value-bound obligations, and the non-ignored `MEAN` denominator. It has no
  detailed specification yet.
- Task 0023 later owns selected compiler-generated backward semantic operations, not autograd
  traversal itself. This task defines no MSE gradient.
- Compiler, prepare/runtime, concrete backends, and the training extension later own their existing
  capture, proof, autograd, lowering, execution, and coordination responsibilities.
- Task 0024 remains the final model capability-selection audit.

## Architecture impact

Expected impact: None.

The task adds one model-owned pure semantic operation and a focused operation-family package under
the existing model boundary. If implementation requires a new dependency, graph-local identity,
hidden state, compiler/runtime/backend/training type, mutable service, or architecture update, stop
and report the conflict.

## Implementation prompt

Use this prompt in a separate agentic task/thread with clean context:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, and
docs/planning/modules/model/tasks/0022-mean-squared-error-loss.md. Implement that task exactly as
specified. Do not implement categorical losses, gradients/autograd, compiler/backend/runtime/
training behavior, or later tasks. Stop and report any architecture, dependency, or maximum-scope
conflict. Do not commit or push unless separately authorized.

After Java implementation and recorded focused/final model-test evidence, hand the final diff and
evidence to a separate documentation-focused agent/thread with clean context. That pass must follow
docs/developer-guide/documentation-rules.md, inspect source/tests, finalize Javadocs, Tensor and
Compile APIs, glossary and planning records, run Javadoc/Markdown/scope/status/whitespace checks,
and reuse successful Java evidence unless executable Java changes.

Update this task's decisions, limitations, evidence, notes, completion summary, and status. Do not
mark Complete before the documentation pass and every acceptance criterion finishes.
```

## Documentation-agent handoff

Provide the clean documentation context with this task, the complete diff, focused/final Java
evidence, exact two-input/one-output roles, reduction/denominator/empty and special-value contracts,
Shape/type/layout/gradient/provenance/ID effects, architecture and 22-path limits, expected
Tensor/Compile/glossary/planning impact, official PyTorch link, and required Javadoc/Markdown/
scope/status evidence. It must inspect source/tests independently and record reasoned no-change
conclusions rather than relying on the handoff summary.

## Local decisions

- `TensorLossExpressions` rejects only unequal static/static Dimension pairs. Structural equality
  passes and every unequal pair involving an unresolved Dimension remains a later equality
  obligation; no broadcast representation is introduced.
- `NONE` retains the exact prediction Shape object, while both reduced forms reuse
  `Shape.scalar()`. This preserves the requested role asymmetry without storing Shape or element
  count in attributes.
- `LossReduction` remains loss-only configuration rather than aggregate-reduction attributes.
  `MeanSquaredErrorAttrs` is the sole operation attributes type and retains only that exact enum
  value.
- The documentation example is explicitly mathematical because current model construction reads
  no values. It uses prediction `[1, 2, 4]` and target `[1, 4, 1]`, yielding losses `[0, 4, 9]`,
  sum `13`, and mean `13 / 3`.

## Known limitations

- Only exact-shape BFLOAT16/FLOAT32/FLOAT64 mean-squared error is current. There is no target
  broadcasting, weighting, mask, ignore index, smoothing, categorical loss, default reduction,
  or integral/BOOL loss domain.
- `requiresGrad` is eligibility metadata only. Gradient/adjoint construction, graph capture,
  deferred-Shape proof, decomposition, lowering, numerical execution, and training coordination
  remain planned in their owning layers.
- Numerical semantics permit reassociation and equal-or-wider intermediates subject to future
  conformance tolerances and the fixed special-value classes; traversal, bitwise identity, NaN
  payload/sign retention, and identical finite rounding are not promised.
- Repository-wide validation remains deferred to the model loss-capability checkpoint after task
  0022B and CI, as specified for this single-module metadata task.

## Validation evidence

- Implementation context `/root/task_0022_implementation` ran the prescribed focused Gradle
  selection containing `LossSemanticsTest`, `TensorMeanSquaredErrorExpressionTest`, `TensorTest`,
  and all eight exact-count inventory suites: `BUILD SUCCESSFUL`. It then ran
  `./gradlew :modules:model:test` exactly once after executable Java stabilized: `BUILD SUCCESSFUL
  in 1s`, three actionable tasks, one executed and two up-to-date. The documentation context
  reused this evidence because it changed Javadoc and Markdown only after that run.
- An earlier implementation-development `compileTestJava` failure came from a test lambda that
  captured a mutated local. The implementation context corrected the test before both successful
  runs above; it is not an unresolved product issue.
- Documentation context `/root/task_0022_implementation/task_0022_docs` applied the General,
  API/Javadoc, API-reference, Example, and Planning profiles. It independently reviewed the final
  five production paths, ten test paths, operation/signature, floating promotion, Shape/Dimension,
  descriptor/factory/producer/provenance foundations, Tensor/Compile/Runtime/Training API
  boundaries, glossary, capability baseline, master plan, roadmap, and this task.
- `./gradlew :modules:model:javadoc`: `BUILD SUCCESSFUL in 2s`; two actionable tasks executed.
  This compiled the final Javadoc-only production edits and generated model Javadoc successfully.
- The targeted Ruby Markdown checker resolved 636 local links including 171 anchors across all
  seven affected documentation/planning files. Its first invocation failed before checking links
  because the installed Ruby lacks `Array#filter_map`; the compatible replacement then passed.
  Targeted checks also reported balanced Markdown fences, final newlines present, and no trailing
  whitespace.
- `curl -L --fail --silent --show-error --head` returned `HTTP/2 200` for the official PyTorch MSE
  reference. An independent arithmetic check produced `losses=[0,4,9] sum=13
  mean=4.33333333333`, matching `13 / 3`.
- The final scope audit found exactly 22 paths: five production, ten tests, and seven
  documentation/planning paths. `javap` found exactly 186 public declared Tensor methods and the
  sole exact receiver signature
  `Tensor meanSquaredError(Tensor, LossReduction)`. It also confirmed package-private, final,
  field-free `TensorLossExpressions` with exactly one package entry and its two private validators.
  Source/package checks confirmed the three loss types under `operation.loss`, exact enum
  vocabularies, fixed 2-input/1-output signature, and no detailed task files for 0022A, 0022B,
  0023, or 0024.
- Final status audit synchronized task 0022 to `Complete` in this task, the model master plan, and
  roadmap. Tasks 0022A–0022B and 0023–0024 remain `Draft`; no later detailed specification exists.
- Final targeted Markdown checks and `git diff --check` passed after status/evidence
  synchronization.

## Implementation notes

- Added exactly `LossKind.MEAN_SQUARED_ERROR`, `LossReduction.NONE/SUM/MEAN`, and immutable
  `MeanSquaredErrorAttrs(reduction)` with the fixed exact two-input/one-output signature.
- Added field-free package-private `TensorLossExpressions` and exactly one public receiver method.
  Construction validates nulls, floating types, rank, and positional static Dimensions before
  promotion and factory delegation; records ordered `[prediction, target]`; selects exact
  none/scalar Shape metadata; combines gradient eligibility by logical OR; and creates one fresh
  producer, ID, and output-index-zero provenance per success.
- Added focused semantic and expression tests and changed only the expected public Tensor count
  from 185 to 186 in the eight existing inventory tests.
- Finalized Javadocs in all five authorized production paths and added the Tensor API expression
  and semantic references, Compile API model/planned boundary, reusable glossary definitions, and
  synchronized capability/task/master/roadmap records.
- Runtime API required no change because no prepared execution or run behavior exists. Training
  API required no change because the loss is model metadata and adds no session, parameter,
  optimizer, step, or gradient contract.
- Existing arithmetic/reduction, Shape/promotion, signature/factory, producer/provenance,
  normalization/indexing, Runtime/Training, architecture/ADR/test, conformance/integration,
  Gradle/dependency, and other-module contracts remain accurate: the implementation composes
  their existing boundaries without changing them. Compile API changed only to expose the new
  current model metadata and explicitly preserve planned compiler responsibilities.

## Completion summary

- Completed changes: added the selected exact-shape MSE semantic vocabulary, attributes,
  construction helper, sole Tensor receiver, tests, Javadocs, API references, glossary terms, and
  synchronized planning records.
- Files changed or created: exactly five production, ten test, and seven documentation/planning
  paths; 22 total.
- Tests and validation: reused successful prescribed focused and one final model-test run;
  independently passed model Javadoc, Markdown links/anchors/fences/newlines/whitespace, official-
  reference, numerical-example, exact-scope/package/surface/status, and final diff checks.
- Documentation-agent review: completed in clean context
  `/root/task_0022_implementation/task_0022_docs` under the selected documentation profiles.
- Documentation impact: Tensor and Compile API references and planning records now distinguish
  current MSE metadata from planned compiler/backend/runtime/training behavior.
- Javadoc review: finalized all five affected production paths; no executable Java changed during
  the documentation pass.
- Glossary impact: added reusable loss, loss-reduction, prediction/target, and MSE/mean-denominator
  distinctions and integrated the new kind, attributes, and provenance into existing entries.
- Unresolved issues: None.
- Follow-up required: None for task 0022. Draft tasks 0022A–0022B retain categorical-loss work;
  later compiler, backend, runtime, training, and checkpoint work remains separately planned.

Status: Complete
