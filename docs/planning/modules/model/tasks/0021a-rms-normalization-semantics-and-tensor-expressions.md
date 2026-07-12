# Task 0021A: RMS Normalization Semantics and Tensor Expressions

## Status

Complete

## Goal

Add one first-class backend-independent `RMS_NORM` meaning and exactly two public receiver
expressions: one without an affine operand and one with explicit elementwise scale. Each operation
normalizes a non-empty trailing slice through its uncentered mean square, adds an exact typed
positive epsilon inside the square root, preserves the input Shape, and records one output without
evaluating values.

This is an intentionally narrow root-mean-square normalization baseline. It keeps scale visible as
an optional logical input, adds no bias, and does not import framework layer state, defaults,
broadcast options, accumulator options, or training behavior into the model API.

## Scope

- Add `RmsNormKind.RMS_NORM` in the existing normalization operation package.
- Add one immutable public `RmsNormAttrs(normalizedShape, epsilon)` value shared safely by the
  contiguous valid input counts one and two.
- Give `RMS_NORM` exactly one family-owned input-range signature accepting one or two inputs and
  exactly one output.
- Add exactly two public receiver methods and no static entry:

  ```java
  public Tensor rmsNorm(Shape normalizedShape, ScalarValue epsilon)
  public Tensor rmsNorm(Shape normalizedShape, Tensor scale, ScalarValue epsilon)
  ```

- Add one package-private stateless `TensorRmsNormExpressions` helper and exactly one final
  `TensorFactory.createDerived` delegation per successful call.
- Normalize the trailing dimensions described by the exact supplied `normalizedShape`; require an
  optional scale Shape exactly equal to it.
- Require floating input and scale, apply ordered floating promotion for the scale form, and
  require epsilon to have the exact result type.
- Select exact uncentered-mean-square, epsilon, accumulation, empty, special-value, signed-zero,
  overflow, reassociation, and determinism policies.
- Produce one fresh unlabeled, storage-free, unresolved-layout result with exact ordered
  provenance and output index zero.
- Update focused Javadocs, Tensor and Compile API references, glossary, capability baseline,
  master plan, roadmap, task evidence, and public-surface tests in the same overall change.

## Out of scope

- bias, shift, scale-plus-bias, bias-only, broadcast scale, implicit scale constants, parameter
  initialization, modules/layers, parameters, trainable ownership, or a general options object
- layer, batch, group, instance, local-response, Lp, weight, spectral, partial RMS, or other
  normalization; task 0021 remains complete and tasks 0021B–0021C remain later
- a public axis or axis array, arbitrary non-trailing axes, inferred normalized Shape, scalar or
  empty normalized Shape, default epsilon, untyped binary64 epsilon, or epsilon Tensor input
- centered mean, variance, standard deviation, subtraction of a mean, correction parameter,
  correction other than the intrinsic divisor `N`, epsilon outside the square root, denominator
  clamp/sentinel, or configurable result/accumulator type
- saved mean square, root mean square, inverse root mean square, or any auxiliary output
- training/evaluation mode, running statistics, momentum, mutation, hidden flags, or stateful
  services
- integral, BOOL, FLOAT16, quantized, sparse, complex, or unsigned inputs; implicit or explicit
  cast-producer insertion
- value reads, eager evaluation, host allocation, result storage, resolved layout, input mutation,
  or materialization
- algorithms, compensated summation, fixed traversal order, gradients, adjoints, compiler capture,
  graph-wide constraints, binding, decomposition, fusion, backend support, lowering, prepare,
  runtime, execution, tolerances, conformance, or integration
- changes to existing layer normalization, softmax, reductions, typed scalars, Shape,
  operation-signature, producer/factory/provenance foundations, architecture, dependencies,
  Gradle, other modules, or detailed specifications for later tasks

## Public and operation contracts

### Semantic kind, attributes, and signature

`RmsNormKind` contains exactly `RMS_NORM`. It does not extend `LayerNormKind`, reuse
`LayerNormAttrs`, or change any existing normalization family. The kind accepts exactly:

| Attributes type | Ordered inputs | Ordered outputs |
|---|---|---|
| `RmsNormAttrs` | `[input]` or `[input, scale]` | `[output]` |

The family declares this stable signature:

```java
OperationSignature.inputRange(RmsNormAttrs.class, 1, 2, 1)
```

The range is intentional and signature-safe: one and two are both valid consecutive counts, with
the optional second position having exactly the scale role. Zero, three, or any larger input count
is invalid. This differs from layer normalization, whose valid counts one and three contain an
invalid two-input hole and therefore require two exact attribute classes. Do not weaken or change
`OperationSignature`, create a registry, reuse either layer-normalization attributes class, or add
a second kind for scale.

`RmsNormAttrs` contains exactly:

```java
Shape normalizedShape
ScalarValue epsilon
```

It null-checks `normalizedShape`, then `epsilon`; requires `normalizedShape.rank() > 0`; requires
floating epsilon; and requires epsilon to be finite and strictly greater than positive zero in its
exact BFLOAT16, FLOAT32, or FLOAT64 representation. It retains both immutable references
unchanged. Exact intrinsic failures are:

```text
NullPointerException("normalizedShape")
NullPointerException("epsilon")
IllegalArgumentException("normalizedShape rank must be positive")
IllegalArgumentException("epsilon must have a floating data type, but was <dataType>")
IllegalArgumentException("epsilon must be finite and positive: <epsilon>")
```

Positive zero, negative zero, negative finite values, either infinity, and every NaN encoding fail
the final check. The record stores no Tensor, scale-presence flag, input rank, axes, result,
layout, gradient, compiler, backend, or execution state. Input cardinality, not duplicated
attributes, identifies whether scale is present.

### Public surface and scale policy

The no-scale method records exact ordered input `[input]`. The scale method records
`[input, scale]`. Scale is a required explicit operand in that overload; there is no nullable
operand, `Optional<Tensor>`, hidden all-ones Tensor, attribute-held scalar scale, bias, alias, or
attrs-taking public overload.

The methods construct model expressions, not layer objects. They do not create, initialize, own,
register, or mutate trainable parameters. There is no `double` epsilon convenience, default
epsilon, public static entry, or raw-axis overload.

### Normalized Shape, scale Shape, and deferred constraints

For input Shape `[D0, ..., D(R-1)]` and normalized Shape `[N0, ..., N(K-1)]`:

- require `K > 0` and `K <= R`;
- normalized axis `j` corresponds to input axis `R - K + j`;
- equal static extents pass and unequal static extents fail locally;
- structurally equal dynamic or expression Dimensions pass;
- any other pair involving an unresolved Dimension is accepted with an equality obligation for
  later compiler capture/binding because the result Shape is still exactly derivable; and
- the result retains the exact input Shape reference and every exact Dimension reference.

Exact task-owned rank and static mismatch failures are:

```text
rmsNorm normalized rank must not exceed input rank: normalizedRank=<K>, inputRank=<R>
rmsNorm normalized dimension mismatch at normalized axis <j>: input=<inputDimension>, normalized=<normalizedDimension>
```

The scale Shape must be structurally equal to `normalizedShape`; ordinary broadcastability is not
enough. This includes exact rank and static, dynamic, and expression structure. Failure is:

```text
rmsNorm scale Shape must equal normalizedShape: scale=<scaleShape>, normalizedShape=<normalizedShape>
```

A corresponding static zero in the input and normalized Shape is valid. It makes the entire
result empty, so no divisor or root mean square is evaluated. The same rule applies if an
unresolved normalized extent later binds to zero. A zero leading extent also makes the result
empty. Scalar input is invalid because the required positive normalized rank cannot fit rank zero.
No Shape or caller array is copied.

### Formula, correction, and epsilon placement

For one non-empty trailing slice containing `N` values `x_i`, define the uncentered mean square and
root mean square as:

```text
meanSquare = sum_i(x_i * x_i) / N
rms        = sqrt(meanSquare + epsilon)
normalized_i = x_i / rms
```

The no-scale result is `normalized_i`. The scale form is:

```text
output_i = normalized_i * scale_i
```

Scale coordinates span the normalized Shape in row-major logical coordinate order and are reused
for every leading slice. Epsilon is added to the uncentered mean square inside the square root.
There is no centering and no variance. Statistical correction is not applicable: the mean square
always divides by the population count `N`, never `N - c`, and there is no correction attribute.
There is no epsilon outside the root, denominator clamp, sentinel, or replacement norm.

For `[1, 2, 3]` and FLOAT64 epsilon `1e-5`, the mean square is `14 / 3`, the root is approximately
`2.1602492140182963`, and the no-scale result is approximately
`[0.4629095539120194, 0.9258191078240388, 1.3887286617360581]`. A scale value `2` at the final
coordinate yields approximately `2.7774573234721163` there. This example illustrates the
mathematical contract only; model construction does not evaluate it.

The formula and trailing-dimension interpretation align with the official
[PyTorch RMSNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html) and
[ONNX RMSNormalization](https://onnx.ai/onnx/operators/onnx__RMSNormalization.html) contracts.
Synaptik deliberately requires an explicit typed epsilon, exact scale Shape, and no bias or
operator-configurable stash type, rather than copying those broader surfaces.

### Data types, promotion, and accumulation

No-scale RMS normalization accepts BFLOAT16, FLOAT32, or FLOAT64 input and retains its exact type.
The scale form requires both inputs to be floating and derives result type with
`DataTypePromotion.promoteFloating(inputType, scaleType)` in ordered input occurrence order.
Promotion changes result metadata only and inserts no cast.

Exact type failures are:

```text
rmsNorm input must have a floating data type, but was <dataType>
rmsNorm scale must have a floating data type, but was <dataType>
rmsNorm epsilon data type must match result data type: epsilon=<epsilonType>, result=<resultType>
```

Epsilon must exactly equal the derived result type. The normalized computation treats input as
the result type before multiplication and accumulation. BFLOAT16/FLOAT32 results square and sum in
FLOAT32; FLOAT64 results square and sum in FLOAT64. Division and optional scale multiplication
occur in the result type. A BFLOAT16 no-scale result therefore accumulates in FLOAT32 and converts
the final normalized value to BFLOAT16. No narrower accumulator or hidden model result type is
allowed.

For finite values, the semantic target is the stated real formula rounded to the result format. A
conforming implementation may use equal-or-wider intermediates, reassociate, parallelize, or
choose another evaluation strategy only if it preserves the exact special-value classes below
and satisfies later conformance tolerances. Model selects no algorithm. Bitwise equality, a fixed
reduction traversal, NaN payload/sign preservation, and cross-backend identical rounding are not
promised.

### NaN, infinity, signed zero, overflow, and determinism

- Any NaN in a non-empty normalized slice makes `meanSquare`, `rms`, and every normalized value in
  that slice NaN. Optional scale cannot suppress that NaN.
- With no NaN and at least one positive or negative infinity, `meanSquare` and `rms` are positive
  infinity. Each finite input then produces same-signed zero, while each infinite input produces
  NaN from infinity divided by infinity.
- For an all-finite slice, `meanSquare` is non-negative and positive epsilon makes `rms` strictly
  positive. Positive and negative zero inputs therefore remain positive and negative zero,
  respectively, before scale. A finite constant slice is not centered: a nonzero constant remains
  a nonzero normalized value, and an all-zero slice preserves each input zero sign.
- Finite squaring or summation may overflow to positive infinity in the selected accumulator. In
  that case every still-finite numerator produces same-signed zero. Other finite overflow and
  underflow follow the selected accumulator and result formats.
- In the scale form, NaN scale produces NaN at that coordinate. Infinite scale times a normalized
  zero produces NaN; infinite scale times a nonzero finite normalized value produces the
  corresponding signed infinity. Finite scale multiplication follows ordinary floating sign
  rules, including signed zero. No bias addition follows.
- Empty results contain no values and expose no mean square, denominator, NaN, infinity, or signed-
  zero result.

These policies make special-value classes independent of physical traversal. Finite nonzero
rounding remains subject to permitted reassociation and later tolerance. Separately constructed
equal requests remain distinct producers; there is no interning or canonicalization.

### Result metadata, provenance, and identity effects

The descriptor uses the exact or promoted result type, the exact input Shape reference,
unresolved layout, and `requiresGrad` equal to input eligibility for no-scale or logical OR of
input and scale eligibility for scaled RMS normalization. Every result is fresh, unlabeled, and
storage-free. Inputs and their descriptors, IDs, labels, storage, and provenance are not mutated.

Each successful call creates one `Operation`, one `TensorProducer`, one output descriptor, one
Tensor wrapper, and one Tensor ID. Provenance output index is zero and ordered inputs are exactly
`[input]` or `[input, scale]`. No saved statistic, hidden scale, sibling descriptor, wrapper, or ID
is created. Future compiler-generated saved values or adjoints do not change this producer's one-
output contract.

### Validation and construction order

`TensorRmsNormExpressions` exposes exactly two package-private entries matching the public methods.
The no-scale entry validates in this order:

1. null-check `input`, `normalizedShape`, then `epsilon`;
2. validate input floating eligibility;
3. require positive normalized rank and validate it against input rank;
4. validate trailing static compatibility in normalized-axis order;
5. validate epsilon exact result type;
6. construct `RmsNormAttrs`, descriptor, operation, and delegate once.

The scale entry validates in this order:

1. null-check `input`, `normalizedShape`, `scale`, then `epsilon`;
2. validate input, then scale floating eligibility;
3. require positive normalized rank and validate it against input rank;
4. validate trailing static compatibility in normalized-axis order;
5. validate exact scale Shape;
6. promote input and scale in occurrence order;
7. validate epsilon exact result type;
8. construct `RmsNormAttrs`, descriptor, operation, and delegate once.

Null failures use the exact parameter names. The attributes constructor repeats its intrinsic
non-null/rank/epsilon validation defensively. Existing Shape, promotion, descriptor, operation,
producer, and identifier failures retain their current messages. Every local failure happens
before factory delegation and consumes no Tensor ID, producer, or wrapper.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Capabilities](../capabilities.md)
- [Master plan](../master-plan.md)
- [Layer normalization](0021-layer-normalization-semantics-and-tensor-expressions.md)
- [Operation signatures](0018k-operation-signature-and-construction-hardening.md)
- [Shared producer provenance](0018l-shared-multi-output-tensor-provenance.md)
- [Typed scalar values](0018n-typed-scalar-value-contract.md)
- [Multi-axis and statistical reductions](0018v-multi-axis-and-statistical-reductions.md)

## Architecture constraints

- Work stays in model plus its documentation/planning. `Tensor` remains public mutable API state,
  not graph intermediate representation.
- RMS-normalization operation types contain backend-independent semantic parameters only and do
  not import Tensor, graph, compiler, training, runtime, prepare, or backend types.
- Package direction is tensor helper to normalization operation/datatype/shape. No reverse edge,
  registry, service, layer owner, parameter owner, or state owner is introduced.
- Compiler owns capture, deferred equality proof, binding validation, gradients, saved values,
  adjoints, and legal decomposition. Backend prepare owns algorithms, lowering, specialization,
  fusion, kernels, and tolerance satisfaction. Runtime executes prepared work without consuming
  `Operation` on its hot path.
- No architecture, focused-architecture, dependency, lifecycle, Gradle, or cross-module change.

## Package impact

Existing packages used and changed:

- `io.github.pho001.synaptik.model.operation.normalization` — add distinct RMS-normalization
  semantics beside unchanged softmax and layer-normalization families.
- `io.github.pho001.synaptik.model.datatype`
- `io.github.pho001.synaptik.model.shape`
- `io.github.pho001.synaptik.model.operation`
- `io.github.pho001.synaptik.model.tensor`

Packages added: None.

Type placement:

- `...operation.normalization.RmsNormKind` — public semantic identity and exact contiguous input-
  range signature.
- `...operation.normalization.RmsNormAttrs` — public trailing-Shape and exact typed epsilon
  parameters shared by the two safe cardinalities.
- `...tensor.TensorRmsNormExpressions` — package-private validation, promotion, Shape, descriptor,
  and provenance owner.
- `...tensor.Tensor` — established public fluent receiver facade.

Tests mirror production packages where package-private access is required.

## Affected files

Production (4):

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/normalization/RmsNormKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/normalization/RmsNormAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorRmsNormExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests (8):

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/normalization/RmsNormSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorRmsNormExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java` — exact
  signatures and public count 181 to 183.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLayerNormExpressionTest.java`
  — count only, 181 to 183.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
  — count only, 181 to 183.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
  — count only, 181 to 183.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLinearExpressionTest.java`
  — count only, 181 to 183.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressionTest.java`
  — count only, 181 to 183.

Documentation/planning (7):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless inaccurate: Runtime/Training APIs; layer normalization, softmax,
reductions/statistics, typed-scalar, Shape, signature, Tensor/factory/producer/provenance, and
other operation contracts; architecture/ADRs/tests; conformance/integration; Gradle; other modules.

## Maximum scope

Exactly 19 paths maximum: four production, eight test, and seven documentation/planning paths. The
two new public methods require six existing public-count assertions to advance together, while one
input-range signature avoids a redundant scale-specific attributes class. Stop for path 20,
another production type/test/document, an existing-helper change, architecture, Gradle,
cross-module work, bias, saved output, state/mode behavior, or a later task specification.

## Javadoc and documentation requirements

- Fully document kind, attrs, helper, and both Tensor methods: uncentered formula, trailing Shape,
  scale role, exact epsilon placement/type, accumulation, empty/special-value behavior,
  metadata/provenance/ID effects, validation/failures, and lifecycle boundaries.
- Every parameter, non-void result, and expected failure has complete `@param`, `@return`, and
  `@throws` tags.
- Tensor API gets both exact signatures, an operand/Shape table, the finite `[1, 2, 3]` example,
  a static mismatch, symbolic deferral, empty/special policies, and current-model versus planned
  compiler/execution boundary.
- Compile API records RMS-normalization metadata as current after implementation and leaves
  capture, constraint proof, saved values, gradients, decomposition, lowering, and execution
  planned.
- Review glossary terms normalization, RMS normalization, normalized Shape, uncentered mean
  square, root mean square, epsilon, scale, accumulator, producer, and saved statistic. Add or
  refine only reusable distinctions needed by the public explanation.
- Synchronize statuses: 0021 remains Complete; 0021A is Ready during implementation and becomes
  Complete only after all criteria pass; 0021B–0021C and 0022–0024 remain Draft without detailed
  specifications.
- Record reasoned no-change conclusions for Runtime/Training APIs, related contracts,
  architecture, conformance/integration, Gradle, and other modules.

## Acceptance criteria

- Exact `RMS_NORM`, one safe `RmsNormAttrs` input-range signature, and two canonical receiver
  methods exist; layer normalization and softmax remain unchanged and public Tensor count is 183.
- Non-empty trailing normalized Shape, exact scale Shape, static mismatch rejection, symbolic
  equality deferral, exact input Shape retention, and empty-result rules match this task.
- No-scale exact type retention and scale-form ordered promotion pass; epsilon is exact-result-
  typed, floating, finite, and positive; selected accumulator types are API-locked without value
  evaluation.
- Uncentered mean square divided by `N`, epsilon inside the root, no correction option, scale-only
  formula, NaN/infinity/signed-zero, overflow, reassociation, rounding, and determinism policies
  are explicit and internally consistent.
- Requires-grad, unresolved layout, freshness, no label/storage/mutation, exact input order, one
  output/index zero, and exact one-ID/producer/wrapper effects pass.
- Validation order and task-owned messages pass; every pre-factory failure consumes no ID.
- No bias, broadcast scale, axes/options/default epsilon, hidden constants, saved outputs,
  state/mode, gradient/compiler/algorithm/backend/runtime, architecture, dependency, build,
  cross-module, or later-spec work is added.
- Focused tests, exactly one final model suite, Javadoc/docs/link/anchor/fence/newline/whitespace,
  official-link, exact 19-path/package/public-surface/status audits, and `git diff --check` pass.
- A separate clean documentation-focused pass reuses Java evidence, finalizes Javadocs/docs, and
  records reasoned no-change conclusions before completion.

## Tests / validation

Focused development validation:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.normalization.RmsNormSemanticsTest --tests io.github.pho001.synaptik.model.tensor.TensorRmsNormExpressionTest --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest --tests io.github.pho001.synaptik.model.datatype.ScalarValueTest --tests io.github.pho001.synaptik.model.shape.ShapeTest --tests io.github.pho001.synaptik.model.tensor.TensorProducerTest --tests io.github.pho001.synaptik.model.tensor.TensorProvenanceTest
```

After executable Java stabilizes, exactly once:

```bash
./gradlew :modules:model:test
```

Focused tests cover exact attrs pairing and input range, public surface, both formulas as
inspectable semantic contracts, static/symbolic/empty Shapes, scale Shape, promotion,
accumulation/special-value policy, validation/ID effects, metadata, freshness, and exact
one-output provenance without value execution.

Documentation pass after final Javadoc:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

The documentation-focused pass also runs the repository's existing or targeted Markdown
link/anchor/fence/final-newline checks, verifies the PyTorch and ONNX official links, audits exact
19-path scope and package placement, checks public count/signatures and synchronized statuses, and
records reused Java evidence. Repository-wide validation is deferred to the selected modern-
operation checkpoint after task 0022 and CI because this task changes one module without
dependencies, architecture, shared build configuration, or executable backend behavior.

## Dependencies

- 0001–0002: DataType, Shape, Dimension, static/symbolic, rank, and element-count contracts.
- 0005–0007 and 0011–0013: operation, descriptor, Tensor, factory, and provenance foundations.
- 0018K: exact attributes-class signatures and inclusive occurrence cardinality.
- 0018L: producer/output-index provenance, used here in one-output form.
- 0018N: exact typed scalar representation for epsilon.
- 0018V: accumulation, empty-domain, special-value, reassociation, and statistical vocabulary
  precedents; RMS normalization deliberately has no correction parameter.
- 0021: trailing normalized-Shape, deferred equality, typed epsilon, and metadata precedents,
  without reusing layer-normalization attrs or centered formula.

All dependencies are Complete.

## Follow-up tasks

- 0021B (Draft) — batch-normalization inference with explicit running mean/variance and affine
  inputs, one output, and no hidden training flag, mutation, or saved output.
- 0021C (Draft) — batch-normalization training as an explicit state transition after inference:
  batch statistics, running-stat inputs/outputs, and saved statistics use genuine multi-output
  provenance without a stateful service.
- 0022–0024 remain Draft at their established IDs. Task 0023 owns any RMS-normalization adjoint or
  compiler-generated saved-statistic meaning justified by compiler implementation.

## Architecture impact

Expected impact: None.

The task adds model-owned semantic values and public expression metadata inside existing package
and dependency boundaries. If implementation requires hidden mode/state, bias, a cross-module
type, architecture/dependency change, or runtime/backend behavior, stop and report the conflict.

## Implementation prompt

Use this prompt in a separate agentic task/thread with clean context:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, and
docs/planning/modules/model/tasks/0021a-rms-normalization-semantics-and-tensor-expressions.md.
Implement that task exactly as specified and do not implement out-of-scope or later normalization
work. Stop and report any architecture, dependency, or maximum-scope conflict. Do not commit or
push unless separately authorized.

After Java implementation and recorded module-test evidence, hand the final diff and evidence to
a separate documentation-focused agent/thread with clean context. That pass must follow
docs/developer-guide/documentation-rules.md, inspect source/tests, finalize Javadocs, Tensor and
Compile APIs, glossary and planning records, run Javadoc/Markdown/scope/status/whitespace checks,
and reuse successful Java evidence unless executable Java changes.

Update this task's local decisions, limitations, evidence, notes, completion summary, and status.
Do not mark Complete before the documentation pass and all acceptance criteria finish.
```

## Documentation-agent handoff

Provide the clean documentation context with this task, the complete implementation diff, exact
focused/final Java commands and results, affected RMS-normalization semantics/public methods,
architecture and 19-path constraints, expected Tensor/Compile API and glossary/planning impact,
official links, and required Javadoc/Markdown/scope/status/whitespace evidence. It must inspect the
actual source/tests, not rely only on the summary, and record reasoned no-change conclusions.

## Local decisions

- Kept RMS normalization first-class and distinct from layer normalization: one `RMS_NORM` kind,
  one `RmsNormAttrs(normalizedShape, epsilon)` value, and one safe inclusive one-to-two-input
  signature represent exact `[input]` and `[input, scale]` occurrences.
- Selected only receiver methods with explicit typed epsilon. The no-scale form retains input type;
  the scaled form promotes input then scale and requires epsilon to match that result exactly.
- Required exact structural scale Shape equality rather than broadcasting. Trailing static input
  mismatches fail locally, while equality involving unresolved Dimensions defers because result
  Shape remains exactly the input Shape.
- Fixed the uncentered population mean-square formula, epsilon inside the square root, FLOAT32
  accumulation for BFLOAT16/FLOAT32 results, FLOAT64 accumulation for FLOAT64, and explicit
  empty/special-value/reassociation boundaries without selecting an algorithm.
- Retained the current one-output expression model: one fresh producer, descriptor, wrapper, and
  Tensor ID with output index zero and exact ordered provenance. No hidden scale, bias, saved
  statistic, sibling output, or execution state is created.

## Known limitations

- Construction records semantic metadata and reads no values. Compiler capture, deferred-
  constraint representation/proof, saved values, gradients or adjoints, legal decomposition,
  lowering, backend preparation, tolerance enforcement, and runtime execution remain planned.
- Fixed traversal, bitwise identity, NaN payload/sign retention, and cross-backend identical
  finite rounding are not promised; later conformance work must select tolerances consistent with
  the documented special-value classes.
- Bias, broadcast scale, defaults, state/mode behavior, saved outputs, and other normalization
  families remain deliberately absent. Tasks 0021B–0021C and 0022–0024 remain Draft without
  detailed specifications.
- Repository-wide validation remains deferred to the selected modern-operation checkpoint after
  task 0022 or CI because this change affects only model and its documentation without dependency,
  architecture, shared-build, backend, or execution changes.

## Validation evidence

- Implementation context `/root/task_0021a_implementation` ran the exact focused command listed in
  this task; it passed `BUILD SUCCESSFUL`. Its first broader module-suite attempt exposed only that
  the exact-name `TensorTest` surface set had not yet listed `rmsNorm`; after correcting that
  expected set, the focused `TensorTest` passed. The same context then ran exactly one final stable
  `./gradlew :modules:model:test`; 908 tests passed with `BUILD SUCCESSFUL`. No executable Java
  behavior changed afterward.
- Clean documentation context `/root/task_0021a_implementation/task_0021a_docs` applied the General,
  API/Javadoc, Planning, glossary, and example profiles. It independently reviewed all four
  production contracts, all eight tests, Tensor/Compile/Runtime/Training APIs, glossary,
  capabilities, task, master plan, roadmap, focused architecture, and related layer-normalization,
  softmax, reduction/statistics, typed-scalar, Shape, signature, factory, producer, and provenance
  contracts.
- `./gradlew :modules:model:javadoc` passed `BUILD SUCCESSFUL` after final Javadoc edits. Generated
  pages contain `RmsNormKind`, `RmsNormAttrs`, and both exact `Tensor.rmsNorm` signatures and their
  reviewed parameter, result, failure, formula, metadata, and lifecycle contracts.
- A targeted Markdown checker resolved 607 local links, including 165 heading anchors, across all
  seven documentation/planning paths with zero failures. Separate checks found balanced backtick
  and tilde fences, final newlines, and no trailing whitespace. Manual recalculation confirmed the
  FLOAT64 `[1, 2, 3]`, epsilon `1e-5` example and scaled final coordinate.
- The official PyTorch RMSNorm URL resolves to the current PyTorch RMSNorm reference, and the
  official ONNX RMSNormalization URL resolves to the current ONNX operator reference.
- Compiled-surface inspection found exactly 183 public `Tensor` methods and exactly these two
  signatures: `rmsNorm(Shape, ScalarValue)` and `rmsNorm(Shape, Tensor, ScalarValue)`. Source and
  test inspection confirmed the sole `RMS_NORM` enum value, exact
  `OperationSignature.inputRange(RmsNormAttrs.class, 1, 2, 1)`, two package-private helper entries,
  and no static public entry or forbidden alias.
- Final inventory found exactly 19 authorized paths: four production, eight tests, and seven
  documentation/planning paths. Production types remain in the planned normalization/tensor
  packages; no forbidden cross-module import, architecture/dependency/build edit, later detailed
  specification, bias, saved output, or executable compiler/backend/runtime addition exists.
- Runtime API remains accurate unchanged because no prepared execution, storage, state, or run
  behavior was added. Training API remains accurate unchanged because gradient eligibility is
  metadata only and no gradient rule, parameter, optimizer, saved statistic, or training mode was
  added. Compile API was updated because current RMS metadata is compiler-visible, while capture,
  proof, saved values, gradients, decomposition, lowering, and execution remain planned.
- Related layer-normalization, softmax, reduction/statistics, typed-scalar, Shape, signature,
  factory, producer, and provenance contracts remain accurate unchanged because RMS uses their
  established boundaries without changing them. Architecture/ADRs/tests, conformance/integration,
  Gradle, and other modules remain unchanged because no ownership, dependency, build, backend, or
  end-to-end behavior changed.
- Final status audit confirmed tasks 0021 and 0021A Complete; tasks 0021B–0021C and 0022–0024
  Draft; no model task Ready; and no detailed 0021B/0021C or 0022–0024 specification. Final
  `git diff --check` passed with no output.

## Implementation notes

- Added `RmsNormKind`, `RmsNormAttrs`, and package-private `TensorRmsNormExpressions`, then exposed
  exactly two fluent receiver methods in `Tensor`.
- Focused semantics tests lock the kind/signature, typed epsilon, formula and special-value policy.
  Expression tests lock public surface, type promotion, Shape validation and symbolic deferral,
  empty results, metadata, validation order, freshness, identity use, and exact one-output
  provenance. Six existing surface tests advanced their exact public count from 181 to 183.
- The independent documentation pass finalized all affected Javadocs, added the complete public
  Tensor explanation and compiler lifecycle boundary, refined reusable glossary distinctions, and
  synchronized capability/master/roadmap status without changing executable Java.

## Completion summary

- Completed changes: added distinct RMS-normalization semantics and exact no-scale/scale-only
  Tensor expression construction with the specified formula, Shape/type/epsilon policies,
  metadata, validation, freshness, and one-output provenance contracts.
- Files changed or created: exactly 19 authorized paths comprising four production Java files,
  eight model tests, and seven API/glossary/planning documents listed under Affected files.
- Tests and validation: the exact focused command and final 908-test model suite passed in the
  implementation context; the documentation context reused those stable results, and model
  Javadoc, 607-link/165-anchor Markdown, official-link, generated-page, public-surface, package,
  scope, status, fence/newline/whitespace, terminology, example, and `git diff --check` validation
  passed.
- Documentation review: Tensor and Compile APIs, glossary, capabilities, task, master plan,
  roadmap, and all affected Javadocs are final. Runtime/Training APIs, related contracts,
  architecture/ADRs/tests, conformance/integration, Gradle, and other modules need no change for
  the reasons recorded above.
- Unresolved issues: None within task scope.
- Required follow-up: None for task 0021A. Draft tasks 0021B–0021C and 0022–0024 remain future
  work under their established ownership.

Status: Complete
