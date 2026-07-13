# Task 0023B: Gather-Compatible Scatter-Add

## Status

Complete

## Goal

Add one backend-independent public functional scatter-add whose indices and updates have exactly
the Shape relationship produced by current `GATHER`. The operation starts from a base `data`
Tensor, maps the complete indices Shape back into one selected data axis, adds every update to its
addressed result coordinate, and preserves the exact data Shape:

```text
data Shape    = data before axis + selected extent + data after axis
indices Shape = arbitrary index Shape
updates Shape = data before axis + complete indices Shape + data after axis
result Shape  = exact data Shape
```

This is a generally useful public functional update for repeated-index accumulation, bucketed
addition, embedding-table accumulation, and reversing a Gather selection. It closes the precise
dynamic-extent gap proven by the
[adjoint expressibility audit](../adjoint-expressibility-audit.md) without introducing a
`GATHER_BACKWARD` kind, a general scatter dimension-number language, or compiler behavior.

## Scope

- Append exactly one `SCATTER_ADD` constant to existing `AxisScatterKind`, preserving
  `SCATTER_ELEMENTS` as the first constant and preserving all existing behavior.
- Pair `SCATTER_ADD` with existing `IndexAxisAttrs` and the exact fixed three-input/one-output
  signature.
- Update `IndexAxisAttrs` Javadoc only so it accurately covers `GATHER`, `GATHER_ELEMENTS`, and
  the new `SCATTER_ADD`; do not change its declaration, validation, components, or bytecode
  behavior.
- Add exactly one public Tensor method:

  ```java
  public Tensor scatterAdd(Tensor indices, Tensor updates, int axis)
  ```

- Extend existing package-private field-free `TensorAxisScatterExpressions`; add no helper type.
- Require exact INT32 or INT64 indices, exact matching data/update type, and current numeric data
  types BFLOAT16, FLOAT32, FLOAT64, INT32, or INT64; reject BOOL.
- Define exact Gather-compatible Shape validation, fixed addition, duplicate accumulation,
  metadata, failure order, provenance, and identifier effects.
- Preserve both existing `scatterElements` overloads, all `ScatterReduction` values, Scatter-ND,
  Gather, Gather Elements, Gather-ND, and embedding behavior unchanged.
- Update every exact public Tensor method-count inventory from 189 to 190.
- Finalize Javadocs, Tensor/Compile APIs, glossary, capabilities, and planning records through the
  mandatory separate clean-context documentation pass.

## Out of scope

- restoring historical `SCATTER_AXIS_ADD`, the removed reduced-rank scatter meaning, a
  `GATHER_BACKWARD` kind, aliases, migration shims, deprecated methods, or multiple spellings
- configurable replacement, MUL, MIN, or MAX behavior for the new Shape relationship; callers
  continue to use `scatterElements` or `scatterNd` where their respective Shape contracts apply
- changing `SCATTER_ELEMENTS`, `ScatterElementsAttrs`, `ScatterReduction`, `SCATTER_ND`,
  `ScatterNdAttrs`, or their public methods, validation, numerical boundaries, or Shapes
- changing `AxisGatherKind`, `TensorAxisGatherExpressions`, Gather validation, embedding,
  Gather-ND, One Hot, Shape, Dimension, TensorDescriptor, TensorFactory, TensorProducer, or
  TensorProvenance
- accepting BOOL data, promoting data/update types, converting updates, broadcasting updates,
  inferring an axis, adding a default axis, or accepting raw primitive indices
- reading index or update values, checking concrete bounds, normalizing index values, detecting
  duplicates, allocating/copying storage, mutating the base Tensor, or executing addition
- defining a broad JAX/XLA-style scatter dimension-number contract, coordinate grids, ranges,
  index tuples, batching attributes, builders, registries, factories, or another abstraction
- gradients, adjoint construction, compiler capture, graph traversal, planning, prepare, runtime,
  engine, backend, kernels, atomic implementation, execution ordering, or determinism guarantees
- dependencies, Gradle, architecture documents/tests, conformance/integration tests, another
  module, unrelated refactors, or later tasks

## Exact semantic and public contract

### Final axis-scatter vocabulary

`AxisScatterKind` contains exactly these constants in this order:

```java
SCATTER_ELEMENTS,
SCATTER_ADD
```

The existing constant remains first so its enum order and identity stay stable. The new final
`SCATTER_ADD` name is the additive functional counterpart of current rank-changing `GATHER`:

```text
GATHER            IndexAxisAttrs   [data, indices]          -> gathered values
SCATTER_ADD       IndexAxisAttrs   [data, indices, updates] -> data-shaped accumulated result

GATHER_ELEMENTS   IndexAxisAttrs   [data, indices]          -> aligned gathered values
SCATTER_ELEMENTS  ScatterElementsAttrs
                                  [data, indices, updates]  -> data-shaped configured result
```

This deliberately does not restore historical `SCATTER_AXIS_ADD`. Before task 0018O, that name
was paired with the operation then called `GATHER_AXIS`; current task 0018O renamed that selected
public gather meaning to final `GATHER`. Historical `SCATTER_ADD` instead belonged to a removed
reduced-rank gather meaning. The new contract reuses the concise public name only for the final
Gather-compatible meaning, with no compatibility alias or claim that the removed semantics
return.

Family-owned signatures are exact:

```text
SCATTER_ELEMENTS  ScatterElementsAttrs  3 inputs, 1 output
SCATTER_ADD       IndexAxisAttrs         3 inputs, 1 output
```

`SCATTER_ADD` accepts no `ScatterElementsAttrs`; `SCATTER_ELEMENTS` accepts no `IndexAxisAttrs`.
`OperationSignatureTest` must lock the two variants separately rather than treating the enum as a
single-signature family.

### Public method and helper surface

`Tensor` adds exactly:

```java
public Tensor scatterAdd(Tensor indices, Tensor updates, int axis) {
    return TensorAxisScatterExpressions.scatterAdd(this, indices, updates, axis);
}
```

It adds no overload, reduction argument, static form, target Shape, target Tensor, raw indices,
alias, or convenience spelling.

`TensorAxisScatterExpressions` remains one final package-private non-record class with no fields,
no nested types, and one private zero-argument constructor. Its exact methods become:

```java
static Tensor scatterAdd(Tensor data, Tensor indices, Tensor updates, int axis)
static Tensor scatterElements(Tensor data, Tensor indices, Tensor updates, int axis)
static Tensor scatterElements(
        Tensor data,
        Tensor indices,
        Tensor updates,
        int axis,
        ScatterReduction reduction)
private static void validateIndexType(
        String operation, TensorDescriptor indicesDescriptor)
private static void validateMatchingDataType(
        String operation,
        TensorDescriptor dataDescriptor,
        TensorDescriptor updatesDescriptor)
private static void validateAddDataType(TensorDescriptor dataDescriptor)
private static Shape gatherResultShape(
        Shape dataShape, Shape indicesShape, int normalizedAxis)
private static void validateScatterElementsShape(
        Shape dataShape,
        Shape indicesShape,
        Shape updatesShape,
        int normalizedAxis)
private static Tensor create(
        Tensor data,
        Tensor indices,
        Tensor updates,
        TensorDescriptor dataDescriptor,
        TensorDescriptor updatesDescriptor,
        Operation operation)
```

The existing common `create` method changes only its private construction seam from separate
kind/attributes parameters to one already validated exact `Operation`, allowing both typed
variants without accepting a broad public attrs input. Both existing `scatterElements` paths
retain their current delegation and behavior.

### Gather-compatible Shape relationship

Let data rank be `R`, normalized axis be `A`, and indices rank be `Q`. The required updates rank is
`R - 1 + Q`, and its Dimensions are exactly:

```text
data.dimensions[0:A]
    + indices.dimensions[0:Q]
    + data.dimensions[A + 1:R]
```

Every Dimension reference used to construct the expected Shape is the exact corresponding data or
indices Dimension reference. Validation compares the complete updates Shape structurally with
this expected Shape; it does not broadcast, bind symbols, compare element counts, or replace
Dimensions.

Examples:

```text
data     [2, 3, 4]
axis      1
indices  [5, 6]
updates  [2, 5, 6, 4]
result   [2, 3, 4]
```

```text
data     [batch, vocabulary, width]
axis      1
indices  [tokens]
updates  [batch, tokens, width]
result   exact data Shape
```

Scalar indices remove the selected axis from the updates Shape. Zero-element and unresolved
indices Shapes are accepted when the exact structural updates Shape matches. An unresolved or
zero selected data extent is also representable because construction does not inspect index
values; eventual valid execution still requires every index to be in bounds. No special case
returns `data`, including an updates domain known to be empty.

### Coordinate and addition meaning

For an update coordinate split as:

```text
[dataPrefixCoordinates, indicesCoordinates, dataSuffixCoordinates]
```

the corresponding result coordinate is:

```text
[dataPrefixCoordinates,
 indices[indicesCoordinates],
 dataSuffixCoordinates]
```

The conceptual result begins with every base `data` value. Each update is added to its addressed
coordinate. Updates whose index values select the same result coordinate all accumulate with the
base value. This fixed addition is intrinsic to `SCATTER_ADD`; no reduction value is stored or
accepted.

INT32 and INT64 addition retains exact type and uses the selected fixed-width modular arithmetic
contract. Floating addition retains exact type, permits reassociation of a duplicate group, and
does not promise a NaN payload, intermediate precision, traversal order, bitwise result,
cross-backend determinism, or a particular atomic algorithm. An empty updates domain leaves every
base value mathematically unchanged. Model construction records this meaning but reads and adds no
values.

Index values are not normalized during model construction. Eventual valid execution requires each
value to be in bounds for the selected data extent; negative and out-of-range values are invalid
rather than wrapped, clamped, or ignored. Bounds and duplicate grouping are value-aware later
obligations, while duplicate indices are semantically valid because addition combines them.

### Validation order and diagnostics

`scatterAdd` performs local validation in exactly this order:

1. `Objects.requireNonNull(data, "data")`;
2. `Objects.requireNonNull(indices, "indices")`;
3. `Objects.requireNonNull(updates, "updates")`;
4. read descriptors in data, indices, updates order;
5. require indices type INT32 or INT64;
6. require exact data/update type equality;
7. reject BOOL data;
8. read exact data Shape and normalize the raw axis exactly once through `Shape.normalizeAxis`;
9. derive the expected Gather result Shape exactly once from data, indices Shape, and normalized
   axis;
10. require exact structural equality with updates Shape; and
11. construct `IndexAxisAttrs`, `Operation`, result descriptor, producer, provenance, and Tensor.

Exact failures are:

```text
scatterAdd indices data type must be INT32 or INT64: <dataType>
scatterAdd updates data type must match data: expected=<dataType>, actual=<dataType>
scatterAdd data type must be numeric: BOOL
scatterAdd updates shape must match gather result shape: expected=<shape>, actual=<shape>
```

Axis failure remains the exact `Shape.normalizeAxis` `IndexOutOfBoundsException`. Type failures
precede axis failure; axis failure precedes updates-Shape failure. Every local failure occurs
before `TensorFactory.createDerived` and consumes no Tensor identifier.

### Descriptor, producer, provenance, and identifiers

Success creates in this order:

1. one `IndexAxisAttrs` with the normalized data axis;
2. one `Operation(AxisScatterKind.SCATTER_ADD, attrs)`;
3. one unresolved `TensorDescriptor` retaining the exact data type and exact data Shape reference,
   with `requiresGrad` equal to data/update eligibility OR;
4. one producer with exact ordered inputs `[data, indices, updates]` and one output descriptor; and
5. output-index-zero provenance plus one fresh unlabeled, storage-free Tensor.

Integral descriptors remain non-differentiable under their existing invariant. Index eligibility
never contributes. The input Tensors are unchanged. Every successful call creates a fresh producer
and consumes one Tensor identifier, even for empty updates or a zero-valued base. Current
identifier-exhaustion behavior remains unchanged and no partial state is rolled back.

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
- [Task 0018C](0018c-axis-gather-semantics.md)
- [Task 0018D](0018d-axis-gather-tensor-expressions.md)
- [Task 0018G](0018g-axis-scatter-semantics.md)
- [Task 0018H](0018h-axis-scatter-tensor-expressions.md)
- [Task 0018O](0018o-indexing-taxonomy-and-unstack-normalization.md)
- [Task 0018U](0018u-integral-elementwise-arithmetic-and-comparisons.md)
- [Task 0023](0023-adjoint-expressibility-audit.md)
- [Task 0023A](0023a-binding-aware-sum-to-shape.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Work stays inside model-owned backend-independent indexing semantics, Tensor metadata,
  pre-capture producer/provenance construction, and directly affected docs/planning files.
- `SCATTER_ADD` means functional mathematical accumulation only. It stores no Tensor, Shape,
  bound index, duplicate group, algorithm, atomicity, backend support, kernel, execution route, or
  runtime state.
- Model construction owns only locally provable descriptor/type/Shape validation. Compiler or
  later concrete binding and execution own value bounds and executable realization.
- Compiler later may use this public semantic primitive while constructing Gather adjoints; this
  task adds no gradient rule, traversal, graph capture, or compiler behavior.
- Runtime hot paths must not consume `Operation` or `CompiledNode`; backend prepare owns lowering,
  specialization, fusion, and concrete accumulation implementation.
- No architecture, ADR, architecture-test, dependency, Gradle, cross-module, compiler, runtime,
  prepare, backend, training, conformance, or integration change is authorized. Stop if current
  index attributes, signatures, descriptors, producers, or Shapes cannot express this contract.

## Package impact

Existing packages changed:

- `io.github.pho001.synaptik.model.operation.index`
- `io.github.pho001.synaptik.model.tensor`

No package or public type is added, moved, or renamed.

Type placement:

- `...operation.index.AxisScatterKind` owns both selected one-axis functional scatter identities.
- `...operation.index.IndexAxisAttrs` remains the minimal normalized-axis value shared by Gather,
  Gather Elements, and fixed Gather-compatible scatter-add.
- `...tensor.TensorAxisScatterExpressions` remains the single package-private owner of local axis-
  scatter expression construction.
- `...tensor.Tensor` remains the public fluent facade.

## Affected files

Production (4):

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/AxisScatterKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/IndexAxisAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorAxisScatterExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests (15):

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationSignatureTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/index/AxisScatterSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorAxisScatterExpressionTest.java`
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
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorSumToShapeExpressionTest.java`

Documentation/planning (7):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless inaccurate: Axis Gather, Gather-ND, Scatter-ND, Scatter reduction,
descriptor/factory/producer/provenance, Shape, arithmetic, Training/Runtime APIs,
architecture/ADRs/tests, conformance/integration, Gradle, dependencies, other modules, and later
tasks.

## Maximum scope

Exactly 26 paths: four production, fifteen tests, and seven documentation/planning paths. The
cohesive capability exceeds the usual 18-path guardrail under the user's standing automatic
higher-path authorization because one semantic constant, one facade/helper path, twelve existing
global Tensor-count locks, two focused family tests, the global signature matrix, and mandatory
documentation must change atomically. Stop for path 27, another public type/method, another
helper/test/document, an existing Gather/Scatter-ND contract edit, later-task work, cross-module
work, architecture/Gradle change, or unrelated cleanup. If live repository evidence changes an
inventory path, update this Ready task before implementation without exceeding 26.

## Javadoc and documentation requirements

- Finalize complete Javadocs for both `AxisScatterKind` constants, `IndexAxisAttrs`, the helper and
  all its methods, and public `Tensor.scatterAdd`, including Shape/coordinate meaning, duplicate
  addition, numeric domain, metadata, provenance/IDs, failures, and lifecycle ownership.
- Apply General, API/Javadoc, Planning, and Example profiles as relevant, with complete `@param`,
  `@return`, and expected `@throws` text.
- Tensor API moves Gather-compatible scatter-add from planned to current, adds exact signature,
  static/dynamic/scalar/empty examples, coordinate and type rules, distinction from Scatter
  Elements/Scatter-ND, producer metadata, and lifecycle boundary.
- Compile API records the current model semantic and later bounds/adjoint obligation without
  claiming graph capture, autograd use, lowering, or execution is implemented.
- Glossary adds `Scatter Add`, cross-links Gather and Scatter Elements, records the exact Shape
  formula, and updates `IndexAxisAttrs` and operation-family inventories.
- Capabilities/task/master/roadmap remain synchronized: 0023–0023B become/stay Complete only after
  implementation, 0023C is the next Draft frontier without a detailed specification, and
  0023D–0024 remain Draft.
- Use official [ONNX Gather](https://onnx.ai/onnx/operators/onnx__Gather.html),
  [ONNX Scatter Elements](https://onnx.ai/onnx/operators/onnx__ScatterElements.html),
  [PyTorch `scatter_add`](https://docs.pytorch.org/docs/stable/generated/torch.scatter_add.html),
  and [JAX `lax.scatter_add`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scatter_add.html)
  only as terminology/comparison evidence. Explain that ONNX/PyTorch aligned scatter differs,
  while JAX's general dimension-number API is intentionally broader than this focused contract.
- Record reasoned no-change conclusions for related existing Javadocs, Training/Runtime APIs,
  architecture/ADRs/tests, dependencies, Gradle, conformance/integration, other modules, and later
  tasks.

## Acceptance criteria

- `AxisScatterKind` has exactly `SCATTER_ELEMENTS`, then appended `SCATTER_ADD`; no historical
  `SCATTER_AXIS_ADD`, backward kind, alias, field, nested type, or other constant.
- Existing Scatter Elements signature and behavior remain exact. `SCATTER_ADD` alone accepts exact
  `IndexAxisAttrs` with three inputs and one output; exact cross-family attrs rejection is tested.
- `IndexAxisAttrs` declaration/validation/value behavior remains bytecode-equivalent apart from
  Javadoc and now accurately documents the new pairing.
- Exactly one new public `scatterAdd(Tensor, Tensor, int)` method; public Tensor count is 190; no
  overload, reduction parameter, static method, raw index, target Shape, or alias.
- Axis-scatter helper remains field-free with the exact nine-method surface and private
  constructor; no new helper/type or Gather-helper dependency.
- Exact null/index-type/update-type/numeric/axis/Shape validation order, messages, exception types,
  and no-ID local failures.
- Expected updates Shape is exactly data prefix plus complete indices Shape plus data suffix;
  static, zero, scalar, named/expression, and unresolved Dimensions preserve the selected
  structural contract without broadcasting or binding.
- Fixed addition starts from data and accumulates every addressed update, including duplicates;
  integral modular and floating non-bitwise boundaries are documented; no value work occurs.
- Result retains exact data Shape/type, data/update eligibility OR, unresolved layout, no
  label/storage, exact `SCATTER_ADD`/axis attrs, ordered `[data, indices, updates]` producer,
  output index zero, one fresh ID, and unchanged inputs.
- Existing Scatter Elements, Gather families, Scatter-ND, count locks, and operation-signature
  matrix remain covered without unrelated behavior changes.
- Exact 26-path/package scope; no compiler, execution, architecture, dependency, build, or later-
  task work.
- Separate clean documentation-focused pass and all required validation/evidence complete before
  status Complete.

## Tests / validation

Focused implementation command:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest \
  --tests io.github.pho001.synaptik.model.operation.index.AxisScatterSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorAxisScatterExpressionTest \
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
  --tests io.github.pho001.synaptik.model.tensor.TensorScaledDotProductAttentionExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorSumToShapeExpressionTest
```

After executable Java stabilizes, run exactly one final model suite:

```bash
./gradlew :modules:model:test
```

Focused coverage must verify enum order and exact signatures, cross-attrs rejection, public/helper
surfaces, method count 190, static/dynamic/expression/scalar/zero indices Shapes, exact expected
updates Shape, first deterministic mismatch, all numeric types, BOOL/index/update-type failures,
axis normalization/failure, null order, no-ID failures, duplicate-add semantics as metadata,
descriptor, operation, producer, provenance, storage/label absence, freshness, and unchanged inputs.
Tests inspect metadata only and do not claim value execution.

Documentation pass after final Javadocs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

Also validate Markdown links/anchors/fences/final newlines/trailing whitespace, official
references, examples, exact 26 paths, package placement, signatures/count 190, synchronized
status, 0023C Draft/no spec, absence of removed `SCATTER_AXIS_ADD`, and absence of compiler/runtime/
backend/build changes. Reuse successful Java evidence unless executable Java changes.

Repository-wide validation is deferred to the capability checkpoint after task 0023F and to CI;
this single-module task changes no dependency, build, or architecture boundary.

## Dependencies

- 0001–0002: numeric types and immutable static/symbolic Shape/Dimension contracts.
- 0005–0007 and 0011–0013: typed operations, descriptors, Tensor, identity, factory, and
  provenance.
- 0018C–0018D: final Gather Shape and public expression contracts.
- 0018G–0018H: axis-scatter semantic vocabulary, reduction semantics, and shared helper behavior.
- 0018K: exact family-owned signatures; 0018L: producer/output-index provenance.
- 0018O: final indexing taxonomy and removal of provisional fixed-add meanings.
- 0018U: selected signed-integral arithmetic semantics.
- 0023: completed audit proving the unresolved-Gather-extent gap and selecting a general public
  primitive rather than a backward kind.
- 0023A: completed first audit-selected prerequisite and current public Tensor count baseline.

All dependencies are Complete.

## Follow-up tasks

- 0023C remains the next Draft frontier for signed slice placement plus target-relative dynamic
  crop; do not create its detailed specification during this task.
- 0023D–0023F remain concise Draft public-capability rows.
- Compiler/autograd work later may use `SCATTER_ADD` for Gather/embedding data adjoints and owns
  graph construction, bounds proof, accumulation, canonicalization, and optimization.
- Task 0024 remains Draft and depends through completed 0023F.

## Architecture impact

Expected impact: None. Stop if implementation needs general scatter dimension numbers, another
attributes/helper type, Shape/binding changes, compiler/prepare/backend contracts, dependency,
architecture update, or work outside the exact model and documentation scope.

## Implementation prompt

Use this prompt in a separate clean-context task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, documentation/planning rules, model capabilities/master plan,
roadmap, completed tasks 0018C–0018D/0018G–0018H/0018K–0018L/0018O/0018U/0023–0023A, task
0023B, and every affected/review-only source/test named there in full.

Implement task 0023B exactly inside its 26 authorized paths. Add only final Gather-compatible
`SCATTER_ADD` and one public `Tensor.scatterAdd(indices, updates, axis)` expression. Preserve
Scatter Elements, Scatter-ND, Gather families, Shapes, producer/provenance, and every architecture
boundary. Add no historical alias, configurable reduction, general scatter language, compiler
adoption, value execution, or later task. Stop on architecture, dependency, completed-contract,
validation-order, affected-file, or maximum-scope conflict.

Run focused tests while developing and exactly one final model suite after executable Java
stabilizes. Then hand the actual diff and exact Java evidence to a separate clean-context
documentation-focused agent in the same overall change. That agent must independently inspect
final source/tests, finalize Javadocs, Tensor/Compile APIs, glossary, capability/task/master/
roadmap status and documentation validation, and reuse successful Java evidence unless executable
behavior changes or it records a concrete reason.

Do not mark 0023B Complete until both passes and every acceptance criterion succeed. Leave 0023C
and every later task Draft without a detailed specification.
```

## Documentation-agent handoff

The implementation agent must hand over this task, actual implementation/test diff, exact Java
evidence, final naming/history distinction, kind/signature change, Gather-compatible Shape and
coordinate rules, numeric/duplicate/bounds boundaries, validation and ID order, metadata/
provenance behavior, architecture constraints, expected Tensor/Compile API and glossary changes,
official comparison links, existing-Javadoc review list, and every documentation/scope/status
command. The documentation agent must inspect final source and tests rather than rely on the
summary.

## Local decisions

- Appended final `SCATTER_ADD` after `SCATTER_ELEMENTS` and paired it only with unchanged
  `IndexAxisAttrs`; the historical `SCATTER_AXIS_ADD` and removed reduced-rank meaning remain
  absent.
- Added exactly one public `scatterAdd(Tensor indices, Tensor updates, int axis)` method and kept
  all axis-scatter construction in the existing field-free helper. The helper has exactly nine
  methods plus its private constructor.
- Derived the required updates Shape locally from exact Dimension references rather than calling
  the Gather helper, binding symbols, or introducing a general scatter-dimension mapping.
- Kept fixed addition intrinsic to the kind. There is no reduction attribute or overload, and
  every duplicate target includes the base value plus all addressed updates.

## Known limitations

- Model construction is descriptor-only: it does not read indices or updates, enforce eventual
  index bounds, perform addition, choose accumulation order, construct gradients, capture a graph,
  lower to a backend, or execute work.
- Signed-integral addition is fixed-width modular. Floating addition may be reassociated and has no
  bitwise-order guarantee.
- Compiler/autograd adoption for Gather or embedding adjoints remains later work, as do value-aware
  bounds enforcement and all compiler, prepare, runtime, and backend support.
- Tasks 0023C–0023F and 0024 remain Draft; no 0023C detailed specification was created.

## Validation evidence

- The focused implementation command passed 124 tests across 15 suites with zero failures,
  errors, or skips.
- After executable Java stabilized, the single final `./gradlew :modules:model:test` run passed
  981 tests across 125 suites with zero failures, errors, or skips. The separate documentation
  pass changed Javadocs and Markdown only, so it reused this evidence without rerunning Java tests.
- `./gradlew :modules:model:javadoc` completed successfully after final Javadocs.
- The complete Scatter Add Java example compiled and ran against the built model classes. It
  produced the documented static, symbolic, scalar-index, and zero-index Shapes, normalized
  operation attributes, and exact ordered provenance checks.
- Reflection and `javap` confirmed exactly 190 declared public Tensor methods, the sole exact
  `scatterAdd(Tensor, Tensor, int)` signature, enum order `[SCATTER_ELEMENTS, SCATTER_ADD]`, exact
  fixed signatures, nine helper methods, zero helper fields, and unchanged one-component
  `IndexAxisAttrs`.
- Markdown local links and anchors, balanced fences, official references, final newlines,
  synchronized status, 0023C Draft/no specification, and `git diff --check` passed. `git status`
  confirmed the exact authorized 26 paths: four production, fifteen tests, and seven
  documentation/planning paths.
- Source/package and path inspection confirmed no `SCATTER_AXIS_ADD` production symbol and no
  compiler, runtime, backend, architecture, ADR, dependency, Gradle, conformance, integration,
  other-module, or later-task change. Repository-wide tests remain deferred to the 0023F
  capability checkpoint and CI as specified.

## Implementation notes

- `AxisScatterKind` now has exact signatures `SCATTER_ELEMENTS`/`ScatterElementsAttrs` and
  `SCATTER_ADD`/`IndexAxisAttrs`, each fixed at three inputs and one output. Cross-family
  attributes remain invalid.
- `TensorAxisScatterExpressions.scatterAdd` checks nulls, index type, exact update type, numeric
  domain, normalized axis, and exact Gather-compatible Shape in the specified order before final
  identity allocation.
- Valid results retain exact data Shape/type, combine data/update gradient eligibility, leave
  layout unresolved, omit label/storage, and record one fresh output-index-zero producer with
  ordered exact inputs `[data, indices, updates]`.
- The separate documentation-focused pass reviewed and finalized the four affected Javadocs,
  Tensor and Compile API references, glossary, capability baseline, task, master plan, and roadmap.
  Existing Gather, Gather-ND, Scatter-ND, Shape, descriptor/factory/producer/provenance,
  Training/Runtime API, architecture/ADR/test, conformance/integration, Gradle, dependency, other-
  module, and later-task contracts remained accurate and required no changes.

## Completion summary

- Added final Gather-compatible `SCATTER_ADD` semantics and one public `Tensor.scatterAdd` metadata
  expression with exact validation, Shape, numeric, duplicate-accumulation, result, identity, and
  provenance contracts.
- Extended the existing axis-scatter helper and focused semantic/expression/signature tests while
  preserving all existing Scatter Elements and related indexing behavior.
- Finalized all affected Javadocs, Tensor and Compile API references, glossary terminology,
  capability baseline, task evidence, master plan, and roadmap. No architecture documentation,
  tests outside the model scope, or other API documentation required a change.
- All task-level implementation and documentation validation passed; no unresolved issue or
  follow-up is required for 0023B. Tasks 0023C–0023F and 0024 remain Draft.

Status: Complete
