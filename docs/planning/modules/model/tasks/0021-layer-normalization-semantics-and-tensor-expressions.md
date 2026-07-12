# Task 0021: Layer Normalization Semantics and Tensor Expressions

## Status

Complete

## Goal

Add one first-class backend-independent `LAYER_NORM` meaning and two public receiver expressions:
one without affine inputs and one with explicit scale and bias. The operation normalizes each
non-empty trailing slice described by a non-empty `Shape`, uses population variance and an exact
typed positive epsilon, preserves the input Shape, and records one output without evaluating
values or exposing compiler-owned saved statistics.

This is the first dependency-ordered part of the former broad normalization frontier. RMS
normalization follows because it has a different second-moment formula and affine policy. Batch
normalization remains later because inference consumes explicit running statistics, while training
also transitions running statistics and exposes saved statistics through genuine multi-output
producer semantics.

## Scope

- Add `LayerNormKind.LAYER_NORM` in the existing normalization operation package.
- Add immutable public `LayerNormAttrs(normalizedShape, epsilon)` for one-input normalization and
  `AffineLayerNormAttrs(normalizedShape, epsilon)` for the exact three-input affine form.
- Give `LAYER_NORM` exactly two family-owned signatures: one input/one output with
  `LayerNormAttrs`, and three inputs/one output with `AffineLayerNormAttrs`.
- Add exactly two public receiver methods and no static entry:

  ```java
  public Tensor layerNorm(Shape normalizedShape, ScalarValue epsilon)
  public Tensor layerNorm(
          Shape normalizedShape, Tensor scale, Tensor bias, ScalarValue epsilon)
  ```

- Add one package-private stateless `TensorLayerNormExpressions` helper and exactly one final
  `TensorFactory.createDerived` delegation per successful call.
- Normalize the trailing dimensions whose exact contract is supplied by `normalizedShape`; the
  result retains the exact input Shape reference.
- Require floating input and affine operands, apply floating promotion in ordered input occurrence
  order, and require epsilon to have the exact promoted result type.
- Select exact population-variance, epsilon, accumulation, empty, special-value, signed-zero,
  reassociation, and determinism policies.
- Produce one fresh unlabeled, storage-free, unresolved-layout result with exact ordered
  provenance and output index zero.
- Update focused API, glossary, capability, master-plan, roadmap, task, Javadoc, and public-surface
  tests in the same overall change.

## Out of scope

- RMS, batch, group, instance, local-response, Lp, weight, spectral, or other normalization
- a public axis array, first-axis integer, arbitrary non-trailing axes, inferred normalized Shape,
  scalar/empty normalized Shape, or a general normalization options object
- scale-only, bias-only, broadcast affine operands, implicit affine constants, affine defaults,
  parameter initialization, modules/layers, parameters, stateful services, or trainable ownership
- saved mean, variance, standard deviation, inverse standard deviation, or any other auxiliary
  output; the public result is one Tensor from one one-output producer
- training/evaluation mode, running statistics, momentum, state mutation, or hidden mode flags
- correction other than zero, epsilon outside the square root, epsilon Tensor input, untyped
  binary64 epsilon, epsilon defaults, or output/accumulator configuration
- integral, BOOL, FLOAT16, quantized, sparse, complex, or unsigned inputs; explicit cast producers
- value reads, eager evaluation, host allocation, result storage, resolved layout, mutation, or
  materialization
- algorithms, compensated summation, exact traversal order, gradients, adjoints, compiler capture,
  binding, graph-wide constraint solving, decomposition, fusion, backend support, lowering,
  prepare, runtime, execution, tolerances, conformance, or integration
- changes to softmax, reductions, typed scalars, Shape foundations, producer/factory contracts,
  architecture, dependencies, Gradle, other modules, or detailed specifications for later tasks

## Public and operation contracts

### Semantic kind, attributes, and signatures

`LayerNormKind` contains exactly `LAYER_NORM`. Its declaration does not absorb or rename
`SoftmaxKind`; softmax and layer normalization remain distinct families in the same cohesive
package.

The kind accepts exactly these variants, in this order:

| Attributes type | Ordered inputs | Ordered outputs |
|---|---|---|
| `LayerNormAttrs` | `[input]` | `[output]` |
| `AffineLayerNormAttrs` | `[input, scale, bias]` | `[output]` |

Both records contain exactly:

```java
Shape normalizedShape
ScalarValue epsilon
```

They null-check `normalizedShape`, then `epsilon`; require `normalizedShape.rank() > 0`; require
floating epsilon; and require epsilon to be finite and strictly greater than positive zero in its
own exact BFLOAT16, FLOAT32, or FLOAT64 representation. They retain both immutable references
unchanged. Exact intrinsic failures are:

```text
NullPointerException("normalizedShape")
NullPointerException("epsilon")
IllegalArgumentException("normalizedShape rank must be positive")
IllegalArgumentException("epsilon must have a floating data type, but was <dataType>")
IllegalArgumentException("epsilon must be finite and positive: <epsilon>")
```

Negative zero, positive zero, negative values, infinities, and every NaN encoding fail the last
check. The records own only operation parameters. They store no input, affine Tensor, rank-derived
axis list, result, layout, gradient, compiler, backend, or execution state.

Two attribute classes are intentional. The current exact-class signature contract cannot express
the disjoint valid input counts one and three with one attributes class without incorrectly
accepting two inputs. Do not weaken `OperationSignature`, use an input range, add a registry, or
split affine layer normalization into a second semantic kind.

### Public surface and affine policy

The no-affine method records exact ordered input `[input]`. The affine method records
`[input, scale, bias]`; scale and bias are both required. There is no two-input scale-only form,
nullable operand, optional wrapper, hidden ones/zeros construction, or compatibility alias.
Callers that want scale without bias may supply an explicit zero bias Tensor, keeping every
operand visible to compiler capture.

These receiver methods are model expression constructors, not layer objects. They do not create,
initialize, own, register, or mutate parameters. There is no public static `Tensor.layerNorm`,
attrs-taking overload, raw `int[]` axes overload, `double` epsilon convenience, or default epsilon.

### Normalized Shape and deferred constraints

For input Shape `[D0, ..., D(R-1)]` and normalized Shape `[N0, ..., N(K-1)]`:

- require `K > 0` and `K <= R`;
- normalized axis `j` corresponds to input axis `R - K + j`;
- equal static extents pass and unequal static extents fail locally;
- structurally equal dynamic or expression Dimensions pass;
- any other pair involving an unresolved Dimension is accepted with an equality obligation for
  compiler capture/binding, because the output Shape is still exactly derivable; and
- the result retains the exact input Shape reference, including every Dimension reference.

The task-owned rank and static mismatch failures are:

```text
layerNorm normalized rank must not exceed input rank: normalizedRank=<K>, inputRank=<R>
layerNorm normalized dimension mismatch at normalized axis <j>: input=<inputDimension>, normalized=<normalizedDimension>
```

The affine scale and bias Shapes must each be structurally equal to `normalizedShape`; ordinary
broadcastability is deliberately insufficient. These checks include scalar, rank, static, dynamic,
and expression structure and use exact failures:

```text
layerNorm scale Shape must equal normalizedShape: scale=<scaleShape>, normalizedShape=<normalizedShape>
layerNorm bias Shape must equal normalizedShape: bias=<biasShape>, normalizedShape=<normalizedShape>
```

No Shape object or caller array is copied. A static zero in `normalizedShape` is valid when the
corresponding input extent is zero; the whole input/result then has zero elements, so no
normalization slice produces an output value and no zero divisor is evaluated. The same rule
applies when an unresolved normalized extent later binds to zero. A zero leading extent also makes
the result empty. Scalar input is invalid because a positive normalized rank cannot fit rank zero.

### Formula, correction, and epsilon

For one non-empty trailing slice containing `N` values `x_i`, use population correction zero:

```text
mean     = sum_i(x_i) / N
variance = sum_i((x_i - mean) * (x_i - mean)) / N
standardized_i = (x_i - mean) / sqrt(variance + epsilon)
```

The no-affine result is `standardized_i`. The affine result is:

```text
output_i = standardized_i * scale_i + bias_i
```

Scale and bias coordinates span the normalized Shape in row-major logical coordinate order and
are reused for every leading slice. Epsilon is added to population variance inside the square
root. There is no sample correction, variance clamp, denominator sentinel, norm replacement,
or epsilon outside the root.

This trailing-Shape and biased-variance contract follows the mainstream functional layer-
normalization model documented by [PyTorch LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).
The first normalized axis and standardization equations align with the official
[ONNX LayerNormalization operator](https://onnx.ai/onnx/operators/onnx__LayerNormalization.html),
but Synaptik deliberately keeps saved mean/inverse-standard-deviation outputs and optional bias
outside this first public model contract.

### Data types, promotion, and accumulation

No-affine layer normalization accepts BFLOAT16, FLOAT32, or FLOAT64 input and retains its exact
type. Affine layer normalization requires all three inputs to be floating and computes the result
type by applying `DataTypePromotion.promoteFloating` to `input` and `scale`, then that result and
`bias`, preserving ordered occurrence semantics. Promotion records result metadata only and does
not insert casts or read values.

Exact type failures are:

```text
layerNorm input must have a floating data type, but was <dataType>
layerNorm scale must have a floating data type, but was <dataType>
layerNorm bias must have a floating data type, but was <dataType>
layerNorm epsilon data type must match result data type: epsilon=<epsilonType>, result=<resultType>
```

The exact epsilon type must equal the derived result type. The standardized computation treats
the input as the result type before accumulation. BFLOAT16/FLOAT32 results compute mean and
variance in FLOAT32; FLOAT64 results compute them in FLOAT64. Affine multiply/add occurs in the
result type. A BFLOAT16 no-affine result therefore accumulates in FLOAT32 and converts the final
standardized value to BFLOAT16. No narrower accumulator or hidden model-level result type is
allowed.

For finite values, the semantic target is the stated real formula, rounded to the result format;
an implementation may use equal-or-wider intermediates, reassociate, parallelize, or use another
conforming evaluation strategy only when it satisfies later conformance tolerance and the exact
policies below. Model does not select an algorithm. Bitwise equality, fixed reduction traversal, NaN
payload/sign preservation, and cross-backend identical rounding are not promised.

### NaN, infinity, signed zero, and deterministic classes

- Any NaN or either infinity anywhere in a non-empty normalized slice makes every standardized
  value in that slice NaN. Affine multiplication or addition does not suppress that NaN.
- Every finite constant slice, including any mixture of positive and negative zero, has positive-
  zero variance and exact positive-zero standardized values before affine transformation.
- For a nonconstant finite slice, variance is non-negative; epsilon is positive, so the
  denominator is positive. Accumulator overflow may make a centered value or denominator
  infinite: finite divided by positive infinity produces signed zero, while an indeterminate
  infinity divided by infinity produces NaN. Other finite overflow/underflow follows the selected
  accumulator and result formats.
- A no-affine exact zero is positive zero for a constant slice. In the affine form, a NaN scale or
  bias produces NaN at that coordinate; infinite scale times standardized zero produces NaN;
  infinite scale times a nonzero finite standardized value produces the corresponding signed
  infinity; finite product plus one infinite bias retains that bias infinity; and opposing
  infinities in the final addition produce NaN. Otherwise zero sign follows the selected floating
  multiply/add operations.
- Empty results contain no values and therefore expose no mean, variance, NaN, infinity, or signed-
  zero result.

These rules make special-value class behavior independent of physical traversal. Finite non-zero
rounding remains subject to permitted reassociation and later tolerance. Separately constructed
equal requests remain distinct producers; there is no interning or canonicalization.

### Result metadata, provenance, and identity effects

The result descriptor uses the promoted result type, exact input Shape reference, unresolved
layout, and `requiresGrad` equal to input eligibility for the no-affine form or the logical OR of
input, scale, and bias eligibility for the affine form. Every result is fresh, unlabeled, and
storage-free. Inputs and their labels, storage, provenance, IDs, and descriptors are not mutated.

Each success creates one `Operation`, one `TensorProducer`, one output descriptor, one Tensor
wrapper, and one Tensor ID. Provenance output index is zero and inputs are exactly `[input]` or
`[input, scale, bias]`. No saved statistic, sibling result, hidden descriptor, or additional ID is
created. A later compiler may derive saved values or an adjoint, but that does not retroactively
change this producer's output count.

### Validation and construction order

`TensorLayerNormExpressions` exposes exactly two package-private entries corresponding to the two
public methods. The no-affine entry validates in this order:

1. null-check `input`, `normalizedShape`, then `epsilon`;
2. validate input floating eligibility;
3. require positive normalized rank and validate it against input rank;
4. validate trailing static compatibility in normalized-axis order;
5. validate epsilon exact result type;
6. construct `LayerNormAttrs`, descriptor, operation, and delegate once.

The affine entry validates in this order:

1. null-check `input`, `normalizedShape`, `scale`, `bias`, then `epsilon`;
2. validate input, scale, then bias floating eligibility;
3. require positive normalized rank and validate it against input rank;
4. validate trailing static compatibility in normalized-axis order;
5. validate scale Shape, then bias Shape;
6. promote input/scale/bias in occurrence order;
7. validate epsilon exact result type;
8. construct `AffineLayerNormAttrs`, descriptor, operation, and delegate once.

Null messages are parameter names. The attribute constructors then repeat their intrinsic
non-null/rank/epsilon validation defensively before retaining values. Existing Shape, promotion,
descriptor, operation, producer, and identifier failures retain their messages. Every local
failure occurs before factory delegation and consumes no Tensor ID, producer, or wrapper.

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
- [Operation signatures](0018k-operation-signature-and-construction-hardening.md)
- [Shared producer provenance](0018l-shared-multi-output-tensor-provenance.md)
- [Typed scalar values](0018n-typed-scalar-value-contract.md)
- [Multi-axis and statistical reductions](0018v-multi-axis-and-statistical-reductions.md)
- [Matmul](0019-matmul-semantics-and-tensor-expression.md)
- [Explicit-state dropout](0019b1-explicit-graph-dropout-construction.md)
- [Scaled dot-product attention](0019e-scaled-dot-product-attention.md)
- [NCHW Conv2d](0020-nchw-conv2d-semantics-and-tensor-expressions.md)
- [NCHW average pooling](0020a1-nchw-average-pool2d-semantics-and-tensor-expression.md)

## Architecture constraints

- Work stays in model plus its documentation/planning. `Tensor` remains public mutable API state,
  not graph intermediate representation.
- Normalization operation types record backend-independent semantic parameters only and do not
  import Tensor, graph, compiler, training, runtime, prepare, or backend types.
- Package direction is tensor helper to normalization operation/datatype/shape; no reverse edge,
  service, registry, or state owner is introduced.
- Compiler owns capture, deferred equality proof, binding validation, gradients, saved values,
  adjoints, and legal decomposition. Backend prepare owns conforming algorithms, lowering,
  specialization, fusion, kernels, and tolerance satisfaction. Runtime executes prepared work
  without consuming `Operation` on its hot path.
- No architecture, focused-architecture, dependency, lifecycle, Gradle, or cross-module change.

## Package impact

Existing packages used and changed:

- `io.github.pho001.synaptik.model.operation.normalization` — add layer-normalization semantics
  beside, without changing, softmax semantics.
- `io.github.pho001.synaptik.model.datatype`
- `io.github.pho001.synaptik.model.shape`
- `io.github.pho001.synaptik.model.operation`
- `io.github.pho001.synaptik.model.tensor`

Packages added: None.

Type placement:

- `...operation.normalization.LayerNormKind` — public family-owned semantic identity/signatures.
- `...operation.normalization.LayerNormAttrs` — public no-affine trailing-Shape/epsilon parameters.
- `...operation.normalization.AffineLayerNormAttrs` — public affine trailing-Shape/epsilon
  parameters whose distinct class selects exact three-input cardinality.
- `...tensor.TensorLayerNormExpressions` — package-private validation, promotion, Shape, descriptor,
  and provenance construction owner.
- `...tensor.Tensor` — established public fluent receiver facade.

Tests mirror production packages where package-private access is required.

## Affected files

Production (5):

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/normalization/LayerNormKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/normalization/LayerNormAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/normalization/AffineLayerNormAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorLayerNormExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests (7):

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/normalization/LayerNormSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLayerNormExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java` — exact
  signatures and public count 179 to 181.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
  — count only, 179 to 181.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
  — count only, 179 to 181.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLinearExpressionTest.java`
  — count only, 179 to 181.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressionTest.java`
  — count only, 179 to 181.

Documentation/planning (7):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless inaccurate: Runtime/Training APIs; softmax, reductions, typed-scalar,
Shape, operation/signature, Tensor/factory/producer/provenance, matmul, attention, dropout,
convolution, and pooling contracts; architecture/ADRs/tests; conformance/integration; Gradle;
other modules.

## Maximum scope

Exactly 19 paths maximum: five production, seven test, and seven documentation/planning paths.
This one-path exception above the normal 18-path capability guardrail is required because the
existing exact-class signature contract needs distinct public no-affine and affine attributes to
represent valid input counts one and three without accepting two; keeping both forms in one
semantic task prevents an unusable partial public capability. `Tensor.java` changes only imports,
two methods/Javadocs, and its operation inventory. Five existing tests change only the stated
signatures/count. Stop for path 20, another type/test/document, an existing-helper change,
architecture, Gradle, cross-module work, saved outputs, or a later task specification.

## Javadoc and documentation requirements

- Fully document kind, both attrs records, helper, and Tensor methods: trailing normalized Shape,
  formula/correction/epsilon, affine roles, Shape constraints, promotion/accumulation, empty and
  special values, metadata/provenance/ID effects, validation/failures, and lifecycle boundaries.
- Every parameter, non-void result, and expected failure has complete `@param`, `@return`, and
  `@throws` tags.
- Tensor API gets the two exact signatures, a Shape/operand table, one finite numerical example,
  one static mismatch example, symbolic deferral, empty/special policy, and current-model versus
  planned compiler/execution boundary.
- Compile API records layer-normalization metadata as current after implementation and leaves
  capture, constraint proof, saved values, gradients, legal decomposition, and execution planned.
- Review glossary terms normalization, layer normalization, normalized Shape, affine transform,
  epsilon, population variance, accumulator, producer, and saved statistic; add/refine only
  reusable distinctions required by the public explanation.
- Synchronize planning status: 0021 is Ready during implementation and Complete only after every
  criterion passes; 0021A–0021C and 0022–0024 remain Draft without detailed specifications.
- Record reasoned no-change conclusions for Runtime/Training APIs, related contracts,
  architecture, conformance/integration, Gradle, and other modules.

## Acceptance criteria

- Exact `LAYER_NORM`, both attributes/signature variants, and two canonical receiver methods exist;
  softmax remains unchanged and public Tensor method count is 181.
- Non-empty trailing normalized Shape, exact scale/bias Shape, static mismatch rejection, symbolic
  equality deferral, exact input Shape retention, and empty-result rules match this task.
- No-affine exact type retention and affine ordered promotion pass; epsilon is exact-result-typed,
  floating, finite, and positive; selected accumulator types are API-locked without evaluation.
- Population correction zero, epsilon placement, affine formula, NaN/infinity/signed-zero,
  overflow, reassociation, rounding, and determinism policies are explicit and consistent.
- Requires-grad, unresolved layout, freshness, no label/storage, no mutation, exact input order,
  one output/index zero, and exact one-ID/producer/wrapper effects pass.
- Validation order and exact task-owned messages pass; every pre-factory failure consumes no ID.
- No axes/options overload, default/untyped epsilon, partial affine form, hidden constants, saved
  outputs, state/mode, gradient/compiler/algorithm/backend/runtime, architecture, dependency,
  build, cross-module, or later-spec work is added.
- Focused tests, exactly one final model suite, Javadoc/docs/link/anchor/fence/newline/whitespace,
  official-link, exact 19 paths/packages/public surface/statuses, and `git diff --check` pass.
- A separate clean documentation-focused pass reuses Java evidence, finalizes Javadocs/docs, and
  records reasoned no-change conclusions before completion.

## Tests / validation

Focused development validation:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.normalization.LayerNormSemanticsTest --tests io.github.pho001.synaptik.model.tensor.TensorLayerNormExpressionTest --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest --tests io.github.pho001.synaptik.model.datatype.ScalarValueTest --tests io.github.pho001.synaptik.model.shape.ShapeTest --tests io.github.pho001.synaptik.model.tensor.TensorProducerTest --tests io.github.pho001.synaptik.model.tensor.TensorProvenanceTest
```

After executable Java stabilizes, exactly once:

```bash
./gradlew :modules:model:test
```

Focused tests cover exact attrs pairing/cardinality, public surface, intrinsic epsilon validation,
both formulas as inspectable semantic contracts, static/symbolic/empty Shapes, affine Shape,
promotion/accumulator/special-value policy, validation/ID effects, metadata, freshness, and exact
provenance without value execution.

Documentation pass after final Javadoc:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

The documentation-focused pass also runs the repository's existing or targeted Markdown
link/anchor/fence/final-newline checks, verifies both official documentation links, audits exact
19-path scope and package placement, checks public method count/signatures and synchronized
statuses, and records reused Java evidence. Repository-wide validation is deferred to the selected
modern-operation checkpoint after task 0022 and CI because this task changes one module without
dependencies, architecture, shared build configuration, or executable backend behavior.

## Dependencies

- 0001–0002: DataType, Shape, Dimension, axis/rank/static/symbolic value contracts.
- 0005–0007 and 0011–0013: operation, descriptor, Tensor, factory, and provenance foundations.
- 0018K: exact attributes-class signatures and occurrence cardinality.
- 0018L: shared producer/output-index provenance, used here in its one-output form.
- 0018N: exact typed scalar representation for epsilon.
- 0018V: population variance, accumulation, empty, correction, and special-value precedents.
- 0019, 0019E, 0019B1, and 0020–0020A1: current promotion, high-level semantic, explicit-state,
  validation, and numerical-policy precedents.

All dependencies are Complete.

## Follow-up tasks

- 0021A (Draft) — RMS normalization as a distinct one-output high-level semantic operation. It
  uses root-mean-square rather than centered variance and will independently decide its explicit
  scale-only/no-affine surface; it does not reuse layer-normalization attrs.
- 0021B (Draft) — batch-normalization inference with explicit running mean/variance and affine
  inputs, one output, no hidden training flag, mutation, or saved outputs.
- 0021C (Draft) — batch-normalization training as an explicit state transition after inference
  semantics: batch statistics, running-stat inputs/outputs, and saved statistics use genuine
  multi-output provenance without a stateful service.
- 0022–0024 remain Draft at their established IDs. Task 0023 owns any normalization adjoint or
  compiler-generated saved-statistic semantic operation justified by compiler implementation.

## Architecture impact

Expected impact: None.

The task adds model-owned semantic values and public expression metadata inside existing package
and dependency boundaries. If implementation requires hidden mode/state, a cross-module type,
architecture/dependency change, or runtime/backend behavior, stop and report the conflict.

## Implementation prompt

Use this prompt in a separate agentic task/thread with clean context:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, and
docs/planning/modules/model/tasks/0021-layer-normalization-semantics-and-tensor-expressions.md.
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
focused/final Java commands and results, affected layer-normalization semantics/public methods,
architecture and 19-path constraints, expected Tensor/Compile API and glossary/planning impact,
official links, and required Javadoc/Markdown/scope/status/whitespace evidence. It must inspect the
actual source/tests, not rely only on the summary, and record reasoned no-change conclusions.

## Local decisions

- Implemented the planned two exact attributes classes rather than weakening signature matching:
  `LayerNormAttrs` selects one input and `AffineLayerNormAttrs` selects exactly three.
- Reused the task's ordered floating promotion and exact-typed epsilon rules without inserting
  casts or hidden constants. The result always retains the exact input Shape reference.
- Kept all deferred unresolved-dimension equality as metadata obligation only. No model constraint
  object, saved output, gradient rule, or execution mechanism was introduced.
- Documentation uses “normalized Shape” for the exact trailing Shape and keeps it distinct from a
  normalized integer axis. The glossary adds only reusable layer-normalization distinctions.

## Known limitations

- Model construction records layer-normalization meaning and metadata but does not evaluate
  values. Compiler capture, deferred-constraint proof, saved-statistic lifetime, gradients or
  adjoints, and legal decomposition remain planned.
- Backend preparation, algorithms, numerical tolerances, kernels, and runtime execution remain
  outside this task. Finite non-zero results therefore have no current execution evidence.
- RMS normalization, batch-normalization inference/training, loss operations, and compiler-
  generated normalization semantics remain Draft tasks 0021A–0023 without detailed specs.

## Validation evidence

- Implementation context `/root/task_0021_implementation` ran the required focused command. Its
  first run failed only because one new assertion expected the abbreviated diagnostic `input=3`
  while the established `Dimension` rendering is `input=StaticDimension[size=3]`. After correcting
  that test expectation, the exact required focused command passed 42 tests with `BUILD
  SUCCESSFUL`.
- After executable Java stabilized, implementation context `/root/task_0021_implementation` ran
  exactly one `./gradlew :modules:model:test`; it passed with `BUILD SUCCESSFUL`. No executable
  Java changed afterward. Documentation-only Javadoc edits do not invalidate that evidence.
- Documentation context `/root/task_0021_implementation/task_0021_docs` applied General, API and
  Javadoc, Planning, and Example profiles; independently reviewed the architecture contract,
  planning contracts, task, complete implementation diff, source, tests, public APIs, and glossary;
  and finalized the five affected production Javadocs plus all seven documentation/planning paths.
- `./gradlew :modules:model:javadoc` passed after final Javadoc edits with `BUILD SUCCESSFUL`.
- A targeted Ruby check passed for all seven changed Markdown files: every local link target and
  heading anchor resolved, backtick/tilde fences were balanced, and every file had a final newline.
- Official-link checks passed with HTTP 200 for both the PyTorch LayerNorm and ONNX
  LayerNormalization pages.
- Exact-scope audit combined tracked and untracked paths and passed with exactly the prescribed 19
  paths. Package/import review found only the planned normalization, tensor, datatype, shape,
  operation, and JDK dependencies; no new package or cross-module edge exists.
- `javap` public-surface audit passed with 181 public declared Tensor methods and exactly two
  non-static `layerNorm` signatures. Source and test inspection confirmed no static entry,
  axes/options/default-epsilon overload, partial affine form, saved output, or hidden sibling.
- Status audit confirmed task 0021 and its synchronized master-plan/roadmap rows are Complete;
  0021A–0021C and 0022–0024 remain Draft and have no detailed specification files.
- `git diff --check` passed on the final combined change. No repository-wide suite was run because
  this single-module task changes no dependency, architecture, shared build, or executable backend
  contract; the modern-operation checkpoint after 0022 and CI retain that responsibility.

## Implementation notes

- Added `LayerNormKind.LAYER_NORM` with ordered exact signatures
  `LayerNormAttrs: 1 -> 1` and `AffineLayerNormAttrs: 3 -> 1`.
- Added defensive intrinsic Shape/epsilon attributes validation and one package-private Tensor
  construction helper with the specified validation order, type promotion, exact Shape checks,
  descriptor derivation, factory delegation, freshness, and provenance.
- Added the two exact public receiver methods. Focused tests cover signatures, attributes,
  formulas, static/symbolic/empty Shapes, promotion, metadata, validation order and diagnostics,
  identity effects, freshness, and exact one-output provenance. Existing public-count tests were
  updated from 179 to 181 only.
- Tensor API now contains exact signatures, an operand/Shape table, a finite `[1, 2, 3]` example,
  static mismatch, symbolic deferral, empty/special-value rules, provenance, and lifecycle
  boundaries. Compile API identifies layer-normalization metadata as current while keeping
  capture, proof, saved values, gradients, decomposition, lowering, and execution planned.
- Runtime and Training APIs remain unchanged because this task adds no prepared/run contract,
  executable state, training mode, parameter ownership, gradient API, or optimizer behavior.
  Related softmax, reduction, typed-scalar, Shape, operation/signature, Tensor/factory/producer,
  matmul, attention, dropout, convolution, and pooling contracts remain accurate because their
  public behavior did not change. Architecture/ADRs/tests remain unchanged because ownership and
  dependency rules did not change. Backend conformance and integration remain unchanged because no
  executable behavior exists. Gradle, other modules, and later task specs remain unchanged because
  the capability stays within `modules/model` and its existing build/package boundaries.

## Completion summary

- Completed changes: added exact no-affine and affine layer-normalization semantics, public Tensor
  expression construction, validation, result/provenance contracts, focused tests, complete
  Javadocs, API reference, glossary distinctions, and synchronized planning status.
- Files changed or created: exactly the five production, seven test, and seven
  documentation/planning paths listed in this task; no additional path changed.
- Tests and validation: required focused command passed 42 tests after one corrected diagnostic
  expectation; exactly one final model suite passed; final model Javadoc, Markdown
  links/anchors/fences/newlines, official links, package/scope/public-surface/status audits, and
  `git diff --check` passed.
- Documentation-agent review: completed by clean context
  `/root/task_0021_implementation/task_0021_docs` using the required profiles and reused Java test
  evidence because it changed documentation only.
- Documentation impact: Tensor API, Compile API, glossary, capability baseline, master plan,
  roadmap, and this task are finalized. Runtime/Training APIs and related/architecture/execution/
  build documentation need no change for the reasons recorded above.
- Javadoc review: finalized kind, both attributes records, package-private helper, both Tensor
  methods, and Tensor's operation inventory; Javadoc generation passes.
- Glossary impact: added layer normalization, normalized Shape, population variance, epsilon,
  affine transform, accumulator type, and saved statistic as reusable public distinctions.
- Unresolved issues: None.
- Follow-up required: None for task 0021. Draft tasks 0021A–0024 remain separately scoped future
  work.

Status: Complete
