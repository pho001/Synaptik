# Task 0023F: Scaled Dot-Product Attention Weights Output

## Status

Complete

## Goal

Add one explicit public scaled-dot-product-attention result form that returns the ordinary
attention output and the normalized attention weights from the same semantic occurrence.

The completed API must preserve all four existing chainable
`scaledDotProductAttention(...)` methods and their one-output producer behavior. It adds four
clearly named `scaledDotProductAttentionWithWeights(...)` overloads returning one shallowly
immutable `ScaledDotProductAttentionResult`. The carrier exposes output slot zero and weights slot
one from one exact shared producer, so callers and later compiler work never need to recompute
weights and accidentally lose the operation's selected masking and floating-special-value
semantics.

## Rationale and mental model

The existing operation already defines both intermediate weights and the final output:

```text
query  [..., L, E]
key    [..., S, E]  -> scores/weights [..., L, S]
value  [..., S, Ev] -> output         [..., L, Ev]

scores  = (query @ transpose(key)) * scale
weights = maskedSoftmax(scores, final axis S)
output  = weights @ value
```

Reconstructing `weights` as a second public expression is not equivalent. The current attention
contract gives all-masked and all-negative-infinity rows positive-zero weights, splits unit weight
across eligible positive-infinity ties, and excludes masked score/value special values before
arithmetic. Existing general softmax and mask composition cannot reproduce every selected case.

The public model therefore becomes:

```text
one attention producer
  operation: SCALED_DOT_PRODUCT_ATTENTION(attrs)
  inputs:    [query, key, value] or [query, key, value, mask]
  outputs:
    slot 0: output  [..., L, Ev]
    slot 1: weights [..., L, S]

ScaledDotProductAttentionResult
  output  -> exact Tensor wrapper for slot 0
  weights -> exact Tensor wrapper for slot 1
```

The existing one-output methods remain the conventional fluent path:

```java
Tensor next = query.scaledDotProductAttention(key, value).add(residual);
```

Callers request the auxiliary result only when needed:

```java
ScaledDotProductAttentionResult attention =
        query.scaledDotProductAttentionWithWeights(key, value);
Tensor next = attention.output().add(residual);
Tensor weights = attention.weights();
```

These examples describe expression metadata, not eager computation, compiler capture, gradients,
backend support, or execution.

## Scope

- Preserve the existing `SCALED_DOT_PRODUCT_ATTENTION` kind and
  `ScaledDotProductAttentionAttrs` without adding a second semantic identity or attributes type.
- Widen the kind's one exact stable signature to accept the existing three-to-four ordered inputs
  with an output-count range of one through two.
- Preserve all four existing public one-output `scaledDotProductAttention(...)` signatures,
  validation, metadata, numerical meaning, freshness, producer shape, and ID effects.
- Add public `ScaledDotProductAttentionResult(Tensor output, Tensor weights)` under the Tensor
  package.
- Add exactly four public receiver methods:

  ```java
  ScaledDotProductAttentionResult scaledDotProductAttentionWithWeights(
          Tensor key, Tensor value)
  ScaledDotProductAttentionResult scaledDotProductAttentionWithWeights(
          Tensor key, Tensor value, ScaledDotProductAttentionAttrs attrs)
  ScaledDotProductAttentionResult scaledDotProductAttentionWithWeights(
          Tensor key, Tensor value, Tensor mask)
  ScaledDotProductAttentionResult scaledDotProductAttentionWithWeights(
          Tensor key, Tensor value, Tensor mask, ScaledDotProductAttentionAttrs attrs)
  ```

- Reuse the existing package-private attention helper for all validation and Shape derivation.
- Construct the two public outputs with one call to existing
  `TensorFactory.createDerivedOutputs(...)` and one exact producer.
- Keep the output descriptor exactly equal to the existing one-output descriptor contract.
- Give weights the exact score Shape, promoted attention data type, unresolved layout, and
  query/key gradient-request OR.
- Retain exact ordered inputs, exact attrs reference for attrs-bearing overloads, exact producer
  reference in both wrappers, and output indices zero then one.
- Update all thirteen global public-Tensor inventory locks from 196 to 200.
- Finalize affected Javadocs, Tensor and Compile API explanations, glossary terminology,
  capability status, task evidence, master plan, and roadmap through a separate clean-context
  documentation pass in the same overall change.

## Out of scope

- changing, removing, deprecating, or secretly making multi-output any existing
  `scaledDotProductAttention(...)` method
- a new attention kind, a weights-only kind, a second attributes type, an output-selection
  attribute, hidden outputs, a nullable result, or an optional weights component
- recomputing weights with MATMUL, mask, WHERE, SOFTMAX, or another attention occurrence
- returning scores, log weights, dropout masks, RNG state, caches, saved values, query/key/value
  projections, or another auxiliary output
- attention dropout, training mode, hidden/global randomness, `GraphRngState`, grouped-query,
  sparse, packed, cached, additive-bias, offset-causal, or quantized attention
- changing query/key/value promotion, rank checks, batch broadcasting, mask broadcasting,
  explicit-scale matching, causal eligibility, Shape references, or validation messages/order
- changing all-masked, NaN, infinity, signed-zero, accumulation, empty-axis, stability, or
  determinism semantics selected by task 0019E
- an output-only `Tensor` convenience, result-carrier chaining facade, destructuring facility,
  builder, callback, consumer, tuple hierarchy, generic multi-output API, or producer lookup API
- gradient rules, backward kinds, autograd traversal, compiler adoption, graph capture,
  canonicalization, decomposition, fusion, saved-value lifetime, or accumulation
- value or storage inspection, attention execution, algorithms, materialization, kernels, backend
  support, prepare, runtime, engine, training, ONNX, conformance, or integration behavior
- changes to Operation, OperationKind, OperationAttrs, OperationSignature, TensorDescriptor,
  Shape, Dimension, DataType, ScalarValue, TensorFactory, TensorProducer, TensorProvenance, or
  storage foundations
- dependencies, Gradle/build changes, architecture/ADR/tests, another module, unrelated refactors,
  task 0024 implementation, or a detailed task-0024 specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Task 0001](0001-data-type-model.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0005](0005-operation-semantic-foundation.md)
- [Task 0006](0006-operation-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0016I](0016i-softmax-semantic-kinds-and-attributes.md)
- [Task 0016J](0016j-softmax-tensor-expressions.md)
- [Task 0018K](0018k-operation-signature-and-construction-hardening.md)
- [Task 0018L](0018l-shared-multi-output-tensor-provenance.md)
- [Task 0018N](0018n-typed-scalar-value-contract.md)
- [Task 0018Q](0018q-masked-reduction-redesign.md)
- [Task 0019](0019-matmul-semantics-and-tensor-expression.md)
- [Task 0019E](0019e-scaled-dot-product-attention.md)
- [Task 0023](0023-adjoint-expressibility-audit.md)
- [Adjoint expressibility result](../adjoint-expressibility-audit.md)
- [Task 0023E](0023e-cumulative-scan-normalization-and-product.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/model` owns backend-independent attention meaning, output descriptors, and public
  Tensor expression construction. It does not own gradients, compiler traversal, execution,
  kernels, backend support, or saved-value lifetime.
- `Tensor` remains public mutable API state and is not an intermediate-representation node.
- The normalized weights are a generally useful public result and an exact output of the same
  operation occurrence. They are not a compiler-private backward kind or a reconstructed sibling.
- The existing kind remains the single semantic identity because one-output and two-output calls
  request the same attention mathematics. One family-owned `OperationSignature` accepts the exact
  input range and both output counts; producer descriptor count selects the occurrence shape.
- The kind still declares exactly one signature for the exact attributes class. This preserves
  the `OperationKind.signatureFor(...)` invariant that duplicate attribute-class variants are
  malformed.
- Existing public methods create exactly one output descriptor and one wrapper. They do not
  allocate, retain, or hide a weights output.
- New methods create exactly two descriptors and two wrappers through one producer. Both wrappers
  retain that exact producer and expose output indices zero and one.
- The output descriptor is unchanged: promoted query/key/value type, exact output Shape,
  unresolved layout, and query/key/value `requiresGrad` OR.
- The weights descriptor uses the same promoted attention type, exact score Shape, unresolved
  layout, and query/key `requiresGrad` OR. Value and BOOL mask do not affect weights and therefore
  do not contribute to its eligibility metadata.
- Both outputs are unlabeled and storage-free. The carrier owns no state or behavior beyond exact
  wrapper references and ordinary record value semantics.
- The existing numerical contract applies to the exposed weights exactly: excluded entries are
  positive zero; no-eligible and all-eligible-negative-infinity rows are positive zero; eligible
  NaN makes eligible weights NaN; positive-infinity ties split unit weight equally; otherwise
  finite eligible weights use the selected stable-softmax meaning and total one ideally.
- Local validation completes before output descriptors, producer, wrappers, or IDs are allocated.
  Existing failures retain their messages and consume no ID.
- A successful two-output call consumes two monotonically allocated Tensor IDs in slot order. If
  identifier exhaustion occurs after slot zero, that already allocated ID remains consumed and no
  partial result carrier is returned, matching `TensorFactory.createDerivedOutputs(...)`.
- Generic Operation, producer, provenance, and factory foundations remain unchanged. The existing
  multi-output contract is sufficient and must be reused directly.
- Runtime hot paths never consume Operation or producer metadata. No architecture boundary,
  dependency direction, lifecycle ownership, or backend-selection contract changes.
- Stop if implementation requires another semantic kind, another public method, a generic tuple,
  another production type, a foundation change, another module, or an architecture decision.

## Package impact

Existing packages used:

```text
io.github.pho001.synaptik.model.operation.attention
  Attention semantic identity, typed attributes, and occurrence signatures.

io.github.pho001.synaptik.model.tensor
  Public Tensor facade/result carrier and package-private attention expression construction.
```

Type placement:

- `io.github.pho001.synaptik.model.tensor.ScaledDotProductAttentionResult` — public carrier for the
  two exact Tensor wrappers returned by one attention occurrence.
- `io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionKind` — existing
  owner of the one signature accepting both valid output counts.
- `io.github.pho001.synaptik.model.tensor.TensorScaledDotProductAttentionExpressions` — existing
  package-private owner of validation, Shapes, descriptors, and factory delegation.
- `io.github.pho001.synaptik.model.tensor.Tensor` — existing public receiver facade.

No package is added or moved.

## Required contracts

### Operation signature

`ScaledDotProductAttentionKind` still contains exactly one enum constant and no new public member.
Its immutable singleton signature list becomes exactly:

```java
List.of(new OperationSignature(
        ScaledDotProductAttentionAttrs.class,
        3,
        4,
        1,
        2))
```

The output range preserves every current one-output occurrence and admits only the new
same-operation two-output form. Zero or more than two outputs remain invalid. Input order remains
`[query, key, value]` or `[query, key, value, mask]` for both.

### Public result carrier

Add exactly:

```java
public record ScaledDotProductAttentionResult(Tensor output, Tensor weights)
```

Its compact constructor null-checks `output` then `weights` with `Objects.requireNonNull` and exact
messages `output` and `weights`. It retains both exact immutable wrapper references and adds no
factory, helper, alias, list view, producer lookup, descriptor validation, or chaining method.

The record's Javadoc explains slot order, Shapes, type, eligibility difference, exact shared
producer, and its lack of execution, storage, gradient-rule, or lifecycle ownership.

### Public Tensor methods

Preserve the existing four `scaledDotProductAttention(...)` overloads exactly. Add exactly the
four `scaledDotProductAttentionWithWeights(...)` signatures listed in Scope. Default forms create
the same fresh `ScaledDotProductAttentionAttrs(Optional.empty(), false)` values as their existing
one-output counterparts. Attrs-bearing forms retain the exact caller attrs reference.

Each method delegates exactly once to the matching package-private helper entry. There is no
public static form, boolean shortcut, primitive-scale form, nullable mask, `Optional<Tensor>`, or
overload returning weights alone.

### Shared helper and construction

Retain the existing field-free final helper and private constructor. Preserve its two current
package-private `apply(...)` methods and add exactly two package-private `applyWithWeights(...)`
entries, one unmasked and one masked. The helper may refactor private implementation methods but
must not add another type or expose another package-private/public method.

All four construction families use the exact existing null, type, promotion, rank, contraction,
batch, mask, and scale validation order and messages. Score and output Shapes are derived once per
call from the same selected batch Dimension references.

The one-output path remains:

```java
TensorFactory.createDerived(outputDescriptor, Optional.empty(), operation, inputs)
```

The two-output path delegates exactly once:

```java
List<Tensor> outputs = TensorFactory.createDerivedOutputs(
        operation,
        inputs,
        List.of(outputDescriptor, weightsDescriptor));
return new ScaledDotProductAttentionResult(outputs.get(0), outputs.get(1));
```

No second Operation or producer is constructed. No existing output is wrapped again.

### Descriptor and identity contract

For query `[..., L, E]`, key `[..., S, E]`, and value `[..., S, Ev]` after exact three-way batch
broadcast:

| Slot | Result | Data type | Shape | `requiresGrad` |
|---|---|---|---|---|
| 0 | output | promoted query/key/value type | `[..., L, Ev]` | query OR key OR value |
| 1 | weights | promoted query/key/value type | `[..., L, S]` | query OR key |

Both layouts are unresolved. The two outer Shape objects may differ, but they reuse the exact
selected batch Dimensions and exact query-sequence reference; weights additionally retains the
exact key-sequence reference, while output retains the exact value-embedding reference.

Both Tensors retain exact inputs and the same exact `Operation`, `TensorProducer`, and attrs
references. Slot indices are exactly zero and one. Repeated valid calls create distinct producers,
wrappers, and IDs without changing any input or attrs.

## Affected files

Production — exactly four paths:

- modify `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/attention/ScaledDotProductAttentionKind.java`
- add `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/ScaledDotProductAttentionResult.java`
- modify `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressions.java`
- modify `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests — exactly sixteen paths:

- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationSignatureTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/attention/ScaledDotProductAttentionSemanticsTest.java`
- add `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/ScaledDotProductAttentionResultTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressionTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBatchNormInferenceExpressionTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorDenseCategoricalCrossEntropyWithLogitsExpressionTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorIndexCategoricalCrossEntropyWithLogitsExpressionTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLayerNormExpressionTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLinearExpressionTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMeanSquaredErrorExpressionTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorRmsNormExpressionTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorSlicePlacementExpressionTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorSumToShapeExpressionTest.java`

Documentation and planning — exactly seven paths:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a contradiction requires stopping: `ARCHITECTURE.md`, focused
architecture documentation, Training and Runtime APIs, `ScaledDotProductAttentionAttrs`,
Operation foundations, TensorFactory, producer/provenance contracts, existing result carriers,
Shape/Dimension/DataType/ScalarValue contracts, MATMUL/softmax/mask/dropout contracts,
architecture tests, Gradle, other modules, conformance, integration, and completed task history.

## Maximum scope

This task may create or modify exactly the 27 authorized paths above: four production paths,
sixteen test paths, and seven documentation/planning paths. The larger-than-normal scope is
deliberate because four new public methods activate thirteen independent exact Tensor surface
locks and the semantic signature, result carrier, shared helper, focused tests, Javadocs, and API
documentation must land atomically.

Stop if a twenty-eighth path, another production or test type, another public method, another
operation kind, a foundation change, another module, dependency, Gradle, architecture change,
compiler adoption, execution behavior, or detailed task 0024 specification is required.

## Javadoc requirements

- Update `ScaledDotProductAttentionKind` to explain its one-output and two-output occurrence
  structures without implying hidden outputs in the old path.
- Fully document `ScaledDotProductAttentionResult`, both components, constructor null failures,
  exact slot order, shared producer, descriptor distinctions, and non-ownership boundaries.
- Update the helper Javadoc and every affected package-private entry with complete parameters,
  result, Shape/type/eligibility/provenance meaning, nullability, failures, and ID side effects.
- Add complete Javadoc for all four public Tensor methods, including inputs, defaults, mask and
  attrs behavior, result slots, special-value meaning by reference to the operation contract,
  exact failure conditions, and identifier exhaustion.
- Review existing attrs, TensorFactory, TensorProducer, TensorProvenance, TopKResult, and
  BatchNormTrainingResult Javadocs. Record why they remain accurate without changes or stop on a
  contradiction.

## Acceptance criteria

- `ScaledDotProductAttentionKind` still has exactly one constant and exposes exactly one signature
  with input range 3–4 and output range 1–2 as specified above.
- `ScaledDotProductAttentionAttrs` declaration and behavior remain unchanged.
- `ScaledDotProductAttentionResult` has exactly two non-null Tensor components, exact reference
  retention, ordinary record value semantics, and no additional public API.
- All four existing `scaledDotProductAttention(...)` methods retain exact signatures,
  construction, one-output producer shape, validation, numerical contract, and one-ID behavior.
- Exactly four `scaledDotProductAttentionWithWeights(...)` methods exist and no other public Tensor
  method is added or removed.
- Default, attrs, mask, and mask-plus-attrs variants retain exact inputs/defaults/references and
  produce one two-output occurrence.
- Output and weights descriptors, Shapes, exact Dimension references, types, layouts,
  `requiresGrad`, labels, and storage match the required table.
- Both results retain one exact shared producer and operation with output indices zero and one;
  repeated calls are fresh.
- Existing validation order/messages remain exact and every local failure consumes no ID.
- Successful two-output calls consume two IDs in slot order; factory exhaustion returns no partial
  carrier and retains existing non-rollback behavior.
- Focused tests cover masked/unmasked construction, defaults/attrs, dynamic/static/zero Shapes,
  promotion, eligibility differences, numerical weight meanings, exact references, freshness,
  nulls, validation failures, signature rejection, and result-carrier API shape.
- All thirteen public Tensor inventory locks move from 196 to 200 with no unrelated surface
  change.
- Focused validation and exactly one final model suite pass after executable Java stabilizes.
- A separate clean-context documentation-focused pass finalizes every affected Javadoc, Tensor
  and Compile APIs, glossary, capability/task/master/roadmap status, examples, links, and
  formatting without repeating successful Java tests unless executable behavior changes or a
  concrete risk is recorded.
- Final inventory contains exactly the authorized 27 paths; `git diff --check` passes; 0023F is
  Complete only after both passes; task 0024 remains Draft without a detailed specification.
- Architecture, dependencies, Gradle, other modules, compiler/runtime/backend behavior, and
  completed unrelated contracts remain unchanged.

## Tests / validation

Implementation-focused tests while developing:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest \
  --tests io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.ScaledDotProductAttentionResultTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorScaledDotProductAttentionExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorTest
```

Exactly one final model suite after executable Java stabilizes:

```bash
./gradlew :modules:model:test
```

The implementation agent records exact suite/test counts and hands the actual diff plus evidence
to the documentation agent. The documentation pass does not rerun successful Java tests unless it
changes executable Java behavior or records a concrete reason.

Documentation pass:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

The documentation agent also validates Java 26 reflection/generated Javadoc for the exact kind
signatures, result record, eight total attention methods, exact 200-method Tensor surface, both
producer forms, local Markdown links/anchors/fences/final newlines, exact 27-path scope,
synchronized status, and absence of a detailed task-0024 specification. A small runnable
metadata example must demonstrate retrieving `output()` and `weights()` from one result without
claiming execution.

Repository-wide tests are deferred to the task-0024 model capability-selection checkpoint or CI
because this task changes only `modules/model` and no dependency, build, architecture, or
cross-module contract.

## Dependencies

- Tasks 0018K and 0018L for family-owned occurrence signatures and exact shared multi-output
  producer/provenance construction.
- Task 0019E for the completed attention operation, attrs, validation, Shapes, numerical semantics,
  and four existing one-output receiver methods.
- Task 0023 and its final audit matrix for the proof that recomputation cannot preserve every
  current attention-weight boundary and that same-occurrence weights are generally useful.
- Task 0023E is the completed immediately preceding frontier.

## Follow-up tasks

- Task 0024 remains Draft and depends on completion of all six task-0023 follow-ups. Do not create
  its detailed specification in this task.
- Later compiler work may capture weights slot one when building attention adjoints and owns
  saved-value lifetime, cotangent accumulation, traversal, and optimization.
- Later planning/prepare/backend work owns fused or decomposed implementation, materialization,
  kernels, execution, accuracy validation, and performance.

## Architecture impact

Expected impact: None.

This task exposes a second public result already defined by backend-independent attention
semantics and uses the completed model-owned multi-output producer contract. It changes no module
ownership, dependency, lifecycle, runtime hot path, or backend-selection rule. Stop and report if
implementation reveals a required architecture change.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, documentation/planning rules, model capabilities/master plan,
roadmap, completed tasks 0016I–0016J/0018K–0018L/0018N/0018Q/0019/0019E/0023–0023E, task
0023F, Tensor and Compile APIs, glossary, and every affected/review-only source/test named by task
0023F in full.

Implement task 0023F exactly inside its 27 authorized paths. Preserve all four existing one-output
attention methods and add only the four explicit WithWeights forms, one exact public result
carrier, the widened one-through-two-output occurrence signature, and one shared two-output
producer. Preserve every attention validation, numerical, Shape, metadata, producer/provenance,
and architecture contract.
Add no kind, recomputation, hidden output, gradient/compiler adoption, execution behavior,
dependency, Gradle, architecture change, or later task. Stop on architecture, dependency,
completed-contract, validation-order, affected-file, or maximum-scope conflict.

Run focused tests while developing and exactly one final model suite after executable Java
stabilizes. Then hand the actual diff and exact Java evidence to a separate clean-context
documentation-focused agent in the same overall change. That agent must independently inspect
final source/tests, finalize Javadocs, Tensor/Compile APIs, glossary, capability/task/master/
roadmap status and documentation validation, and reuse successful Java evidence unless executable
behavior changes or it records a concrete reason.

Do not mark 0023F Complete until both passes and every acceptance criterion succeed. Leave task
0024 Draft without a detailed specification.
```

## Local decisions

- Preserve fluent one-output attention as the default path and use an explicit `WithWeights`
  method family only when the caller needs both results.
- Use one public operation-specific result record, consistently with existing top-K and
  batch-normalization multi-output APIs, rather than a generic tuple or output list.
- Keep one attention kind and widen its single structural signature to output range one through
  two because `OperationKind` requires one unique signature per exact attributes class and the
  semantic operation is unchanged.
- Expose normalized post-mask/post-causal weights, not raw scores or independently recomputed
  softmax values.
- Give weights query/key eligibility only, while retaining query/key/value eligibility for output.
- Preserve all existing validation and numerical policy by deriving both descriptors in the one
  existing helper and delegating once to the completed shared-output factory.

## Known limitations

- This task constructs metadata and executes no attention or softmax values.
- The weights carrier adds no fluent forwarding API; callers explicitly select `output()` before
  continuing an ordinary Tensor chain.
- No log weights, scores, dropout mask, cache, RNG state, or other auxiliary result is exposed.
- The model records gradient eligibility but defines no attention gradient rule or saved-value
  lifetime.
- Dynamic constraints remain deferred exactly as in task 0019E; this task adds no binding or
  constraint representation.

## Validation evidence

Implementation context: `/root/implement_0023f`.

- The exact prescribed focused command passed 40 tests across its five selected suites with no
  failures, errors, or skips.
- After executable Java stabilized, the single final `./gradlew :modules:model:test` run passed
  1,016 tests across 127 suites with no failures, errors, or skips. `git diff --check` passed
  immediately afterward. The documentation context changed Javadoc and documentation only, so it
  reused this evidence and did not rerun Java tests.

Documentation context: `/root/implement_0023f/document_0023f`, applying the General,
API/Javadoc, Planning, and Example profiles.

- Independently reviewed the repository and architecture instructions, focused architecture
  pages, documentation/planning rules and selected profiles, roadmap, capability baseline, master
  plan, this task, final production/test diff, Tensor/Compile/Training/Runtime APIs, glossary,
  attention attrs and semantics, factory/producer/provenance foundations, existing result
  carriers, and directly relevant completed task contracts. No architecture or executable
  contradiction was found.
- Finalized Javadocs in all four production paths. They now distinguish exact one-output
  occurrences from explicit two-output occurrences; document slot order, Shapes, promoted type,
  gradient eligibility, exact shared producer/operation/attributes/inputs/descriptors, validation,
  identifier exhaustion, selected special values, and lifecycle boundaries; and make no compiler,
  gradient, backend, or execution claim. No executable statement changed.
- Finalized `docs/api/tensor-api.md`, `docs/api/compile-api.md`, and `docs/glossary.md`. The Tensor
  API includes an output-slot table and a runnable Java 26 metadata example retrieving
  `output()` and `weights()` from one result. The Compile API describes compiler-visible metadata
  while leaving traversal, capture, constraints, saved-value lifetime, gradients, and execution
  planned. The glossary adds the reusable attention-weight term and updates current attention
  status.
- `./gradlew :modules:model:javadoc` passed with `BUILD SUCCESSFUL`; two tasks executed. Generated
  Javadoc contains the one-through-two-output kind contract, the exact result record, and all four
  `scaledDotProductAttentionWithWeights` forms. An initial generated-page scan used an inapplicable
  module-qualified output prefix after the successful reflection check and reported missing paths;
  the corrected scan used the actual generated paths and passed.
- `javac --release 26 -cp modules/model/build/classes/java/main -d
  /tmp/synaptik-attention-weights-doc-example /tmp/AttentionWeightsMetadataExample.java` and the
  matching `java` command passed. Output was exact Shapes `Shape[2, 4, 10]` and
  `Shape[2, 4, 6]`, types `FLOAT32,FLOAT32`, eligibility `true,false`, shared producer `true`,
  slots `0,1`, exact attributes reference `true`, and three ordered inputs.
- The standalone Java 26 reflection/construction check compiled and passed. It confirmed the sole
  attention kind signature has input range 3–4 and output range 1–2; the public record has exactly
  `output` and `weights` Tensor components; Tensor exposes exactly eight attention methods and 200
  total declared public methods; and ordinary versus explicit calls create exact one-output and
  shared two-output producer forms.
- `python3 /tmp/validate_synaptik_markdown.py` passed across 210 tracked and untracked Markdown
  files, 3,535 local links, 207 heading anchors, and 2,662 fence markers, including final-newline
  and trailing-whitespace checks. Final inventory validation found exactly the authorized 27
  paths: four production, sixteen tests, and seven documentation/planning paths. Status checks
  found 0023F Complete and 0024 Draft with no detailed task-0024 file. Final
  `git diff --check` passed.
- Training and Runtime APIs remain accurate without edits because this task adds no training
  workflow, gradient rule, prepared execution, schedule, storage, residency, resource, or run
  contract. `ScaledDotProductAttentionAttrs`, `TensorFactory`, `TensorProducer`,
  `TensorProvenance`, `TopKResult`, and `BatchNormTrainingResult` Javadocs remain accurate because
  the new result reuses their existing semantic-attributes and shared-output foundations without
  changing those contracts. Architecture pages/tests, Gradle/dependencies, other modules,
  backend conformance, and integration tests remain unchanged because there is no module boundary,
  dependency, build, compiler, backend, runtime, or executable behavior change.

## Implementation notes

Widened the existing attention kind's sole occurrence signature from exactly one output to one or
two outputs. Preserved all four existing one-output receiver paths and added four explicit
`WithWeights` paths plus the shallow two-component public result. The helper derives both
descriptors after the existing complete validation and delegates once to the existing shared-
output factory; output and normalized weights therefore retain one exact producer at slots zero
and one without recomputation or a hidden output.

## Completion summary

- Completed changes: preserved exact one-output attention and added explicit same-occurrence
  output-plus-normalized-weights construction with exact descriptor, provenance, validation, and
  identifier behavior.
- Files changed or created: exactly the authorized 27 paths — four production, sixteen tests, and
  seven documentation/planning files.
- Tests and validation: reused the passing 40-test focused run and sole 1,016-test/127-suite final
  model run; model Javadoc, runnable Java 26 example, reflection/construction, generated Javadoc,
  Markdown, exact scope, status/no-task-0024, newline/whitespace, and `git diff --check` validation
  passed.
- Documentation-agent review: completed independently in
  `/root/implement_0023f/document_0023f` with the General, API/Javadoc, Planning, and Example
  profiles.
- Documentation impact: Tensor and Compile APIs, glossary, capability baseline, task, master plan,
  and roadmap describe the implemented contract and its current-versus-planned boundaries.
- Javadoc review: all four affected production paths were reviewed and finalized; no executable
  behavior changed after the recorded Java tests.
- Glossary impact: added attention weight and updated scaled dot-product attention to distinguish
  one-output and explicit same-occurrence two-output construction.
- Unresolved issues: None.
- Follow-up required: None for 0023F. Task 0024 remains Draft without a detailed specification.

Status: Complete
