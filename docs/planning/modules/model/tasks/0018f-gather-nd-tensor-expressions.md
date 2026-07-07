# Task 0018F: Gather-ND Tensor Expressions

## Status

Complete

## Goal

Add public model-level Gather-ND expressions with exact index-type, rank, batch-prefix,
tuple-depth, and result-Shape validation.

The zero-batch convenience and explicit-batch method both consume ordered logical inputs
`[data, indices]`, preserve data metadata, leave result layout unresolved, and record exact
`GATHER_ND` provenance. They construct expression metadata only and never read index values.

## Scope

- Add exactly two public instance methods to `Tensor`:
  - `Tensor gatherNd(Tensor indices)`
  - `Tensor gatherNd(Tensor indices, int batchDimensions)`
- Add one field-free package-private final `TensorGatherNdExpressions` helper.
- Give the helper exactly two package-private entries and six private methods specified below.
- Make the no-batch entry delegate exactly to the explicit entry with `batchDimensions = 0`.
- Require indices data type to be exactly `INT32` or `INT64`.
- Require indices rank at least one and batch count smaller than both indices and data rank.
- Require every shared leading batch Dimension to be structurally equal.
- Require the final indices Dimension to be statically known and use its positive extent as tuple
  depth.
- Require tuple depth no greater than `dataRank - batchDimensions`.
- Derive exact result Shape as indices prefix without its final tuple-depth Dimension followed by
  the unindexed data suffix.
- Produce canonical scalar Shape when both result parts are empty.
- Preserve exact data type and gradient eligibility; always leave result layout unresolved.
- Construct exact `GatherNdKind.GATHER_ND`, `GatherNdAttrs`, ordered `[data, indices]` provenance,
  no label or storage, and one final `TensorFactory.createDerived` call.
- Keep every valid request explicit and fresh.
- Update `TensorTest` only for the two-method public API expansion and add one focused expression
  test.
- Apply Javadoc-only current-status corrections to `GatherNdKind` and `GatherNdAttrs`; do not
  change their declarations or behavior.
- Finalize Tensor API, Compile API, glossary, task evidence, master plan, and roadmap through the
  mandatory independent documentation pass.

## Out of scope

- gather axis, scalar select, scatter-ND, axis scatter, masks, slices, or another operation family
- primitive-array Gather-ND conveniences, tuple objects, collection inputs, default indices, or
  another overload
- accepting floating/BOOL indices, implicit conversion, inspecting whether numeric values are
  integral, or converting `INT64` to `INT32`
- reading, normalizing, clamping, or bounds-checking index values; negative index-value policy is
  not model metadata construction
- dynamic tuple depth, symbolic tuple-length constraints, mismatched dynamic batch symbols,
  broadcasting batch Dimensions, or graph-wide Shape inference
- resolved result layout, view/alias geometry, storage access, materialization, device state, or
  backend route selection
- modifying Gather-ND semantic declarations/behavior, Shape, Dimension, DataType,
  TensorDescriptor, TensorFactory, TensorProvenance, Operation, or unrelated completed contracts
- gradients, repeated-index accumulation, scatter-ND backward, graph capture, canonicalization,
  compiler, planning, prepare, runtime, backend, engine, trace, ONNX implementation, training, or
  execution behavior
- another production helper/type, dependency, Gradle/build option, architecture change, another
  module, or task-0018G specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Documentation profile index](../../../../developer-guide/documentation/README.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)
- [Example format](../../../../developer-guide/documentation/example-format.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Task 0001](0001-data-type-model.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0018C](0018c-axis-gather-semantics.md)
- [Task 0018D](0018d-axis-gather-tensor-expressions.md)
- [Task 0018E](0018e-gather-nd-semantics.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only legacy branch exposes zero-batch and explicit-batch Gather-ND. It requires numeric
indices, validates shared batch Shapes, reads tuple depth from the final indices extent, and derives
the result from the indices prefix plus unindexed data suffix.

The new model tightens index data type to exact `INT32` or `INT64`, supports current static/dynamic
Dimension values conservatively, and uses canonical scalar Shape rather than legacy `[1]` when the
formula produces rank zero. It does not copy legacy value access, graph builders, gradient
callbacks, scatter implementations, traits, lowering, kernels, or runtime/backend behavior.

## Architecture constraints

- `Tensor` remains public mutable API state, not IR. The methods create fresh storage-free
  expressions through the existing derived-factory seam.
- Kind and attributes come only from completed task 0018E.
- The helper may inspect immutable descriptors and Shapes, but never values or host/device storage.
- Index type is explicit metadata and is validated locally. Index bounds require values and remain
  deferred.
- Dynamic Dimensions pass only through existing structural equality. Tuple depth must be static
  because it controls which axes exist in the result Shape.
- Gather-ND is value-reordering/materialization semantics, so result layout is always unresolved.
- Type and gradient eligibility come only from data; indices do not contribute eligibility.
- Compiler owns capture/canonicalization; compiler/training owns backward scatter; later layers own
  lowering, materialization, and execution.
- No dependency, package ownership, or module boundary changes are authorized.

## Package impact

No package is added or moved.

- `io.github.pho001.synaptik.model.tensor.Tensor` — receives two Gather-ND methods.
- `io.github.pho001.synaptik.model.tensor.TensorGatherNdExpressions` — owns local validation,
  Shape derivation, and semantic construction.
- `TensorGatherNdExpressionTest` — mirrors `model.tensor` for focused validation.
- `TensorTest` — changes only exact public API inventory/reflection assertions.
- `GatherNdKind` and `GatherNdAttrs` remain in `model.operation.index`; only stale temporal and
  validation-boundary Javadoc wording may change.

## Required contract

### Public Tensor methods

Add exactly:

```java
public Tensor gatherNd(Tensor indices) {
    return TensorGatherNdExpressions.gatherNd(this, indices);
}

public Tensor gatherNd(Tensor indices, int batchDimensions) {
    return TensorGatherNdExpressions.gatherNd(this, indices, batchDimensions);
}
```

Each method contains one return statement and one matching helper call, is non-static and
non-synchronized, and performs no direct validation or field access.

### Helper shape

Create one package-private final, field-free class with one private zero-argument constructor and
exactly these eight static methods:

```java
static Tensor gatherNd(Tensor data, Tensor indices)
static Tensor gatherNd(Tensor data, Tensor indices, int batchDimensions)
private static void validateIndexType(TensorDescriptor indicesDescriptor)
private static void validateBatchDimensions(
        Shape dataShape, Shape indicesShape, int batchDimensions)
private static void validateBatchPrefix(
        Shape dataShape, Shape indicesShape, int batchDimensions)
private static int tupleDepth(
        Shape dataShape, Shape indicesShape, int batchDimensions)
private static Shape resultShape(
        Shape dataShape, Shape indicesShape, int batchDimensions, int tupleDepth)
private static Tensor create(
        Tensor data,
        Tensor indices,
        TensorDescriptor dataDescriptor,
        Shape resultShape,
        GatherNdAttrs attrs)
```

Add no field, nested type, alternate constructor, overload, cache, mutable state, or extra method.

### Default delegation

The two-argument helper contains exactly:

```java
return gatherNd(data, indices, 0);
```

It performs no independent validation. Both public forms therefore share errors, Shape,
provenance, layout, identity, and side effects.

### Explicit validation order

The explicit helper performs exactly:

1. null-check `data` with message `data`;
2. null-check `indices` with message `indices`;
3. read exact data descriptor once;
4. read exact indices descriptor once;
5. call `validateIndexType` once;
6. read exact data Shape once;
7. read exact indices Shape once;
8. reject indices rank zero with `IllegalArgumentException` and exact message
   `gatherNd indices rank must be at least 1`;
9. construct one `GatherNdAttrs(batchDimensions)`, preserving its exact negative-value failure;
10. call `validateBatchDimensions` once;
11. call `validateBatchPrefix` once;
12. call `tupleDepth` once;
13. call `resultShape` once;
14. call `create` once.

All local failures precede result identifier allocation.

`validateIndexType` accepts only `INT32` and `INT64`; otherwise it throws
`IllegalArgumentException` with exact message:

```text
gatherNd indices data type must be INT32 or INT64: <actual>
```

### Batch validation

After non-negative validation by `GatherNdAttrs`, `validateBatchDimensions` requires:

- `batchDimensions < indicesShape.rank()`, otherwise exact message
  `gatherNd batchDimensions must be less than indices rank: batchDimensions=<B>, indicesRank=<Q>`;
- `batchDimensions < dataShape.rank()`, otherwise exact message
  `gatherNd batchDimensions must be less than data rank: batchDimensions=<B>, dataRank=<R>`.

`validateBatchPrefix` compares Dimensions at axes `[0, B)` in increasing order. On first mismatch,
throw exact message:

```text
gatherNd batch dimension at axis <axis> must match data: expected=<dataDimension>, actual=<indicesDimension>
```

Structural equality supports identical static sizes and equal dynamic symbols. No broadcasting or
constraint generation is allowed.

### Tuple depth

`tupleDepth` reads the final indices Dimension exactly once.

- If it is dynamic, throw `IllegalArgumentException` with exact message
  `gatherNd tuple depth must be statically known`.
- Let its static long extent be `depth` and `maximum = dataRank - batchDimensions`.
- If `depth < 1` or `depth > maximum`, throw exact message
  `gatherNd tuple depth must be in [1, data rank - batchDimensions]: depth=<depth>, maximum=<maximum>`.
- Return the validated depth as `int`; the upper bound proves conversion safe.

### Result Shape

`resultShape` creates one Dimension array ordered as:

1. exact indices Dimensions `[0, indicesRank - 1)`, excluding only tuple-depth Dimension;
2. exact data Dimensions `[batchDimensions + tupleDepth, dataRank)`.

Use checked rank addition when allocating. Call `Shape.ofDimensions` once. Empty parts produce
canonical scalar Shape.

Examples:

- data `[2, 3, 4]`, indices `[5, 2]`, `B=0`, `K=2` gives `[5, 4]`;
- data `[2, 3, 4]`, indices `[2, 5, 1]`, `B=1`, `K=1` gives `[2, 5, 4]`;
- data `[2, 3]`, indices `[2]`, `B=0`, `K=2` gives scalar `[]`;
- data `[N, 3, 4]`, indices `[N, M, 1]`, `B=1`, `K=1` gives `[N, M, 4]` while retaining exact
  `N`, `M`, and suffix Dimension references.

### Result metadata

`create` constructs exactly:

```java
TensorDescriptor descriptor = new TensorDescriptor(
        dataDescriptor.dataType(), resultShape, Optional.empty(), dataDescriptor.requiresGrad());
Operation operation = new Operation(GatherNdKind.GATHER_ND, attrs);
TensorProvenance provenance = new TensorProvenance(operation, List.of(data, indices));
return TensorFactory.createDerived(descriptor, Optional.empty(), provenance);
```

Every current data type is accepted for data. Results preserve exact data type/eligibility, have
unresolved layout, absent label/storage, exact attrs, ordered provenance, and fresh identity.

## Affected files

Expected implementation change:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorGatherNdExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/GatherNdKind.java`
  — Javadoc only
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/GatherNdAttrs.java`
  — Javadoc only
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorGatherNdExpressionTest.java`
- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task file
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

This task may create or modify at most the twelve paths listed above. The two semantic files allow
Javadoc-only current-status corrections; declarations and behavior must remain bytecode-equivalent.

If implementation requires another production type/helper, semantic behavior change, another
test, capability baseline, architecture, dependency, build, another module, or more than twelve
paths, stop and propose a follow-up task.

## Javadoc requirements

- Document both public methods and all helper methods/constructor.
- Explain validation order, index type, ranks, batch prefix, static tuple depth, result formula,
  dynamic equality, and all examples.
- Explain unresolved layout, exact type/eligibility, provenance, fresh identity, and no storage.
- State that index values/bounds are never inspected.
- Correct GatherNdKind/GatherNdAttrs wording from future input-aware boundary to current public
  boundary, without declaration or behavior changes.
- Distinguish axis gather and scatter-ND and avoid promising gradients/execution/backend support.

## Acceptance criteria

- Tensor adds exactly two public non-static, non-synchronized Gather-ND methods.
- Each delegates once; default uses exact batch count zero through helper delegation.
- Helper is package-private/final/field-free with one private constructor and exactly eight methods.
- Validation follows exact order, exception types, and messages.
- Only INT32/INT64 indices pass; values are not read.
- Batch prefix, static positive tuple depth, and result Shape rules match specification.
- Canonical scalar and exact Dimension reference retention are verified.
- Results preserve data metadata, unresolved layout, exact GATHER_ND/attrs, `[data, indices]`, no
  label/storage, and fresh identity.
- Semantic declaration/behavior remains unchanged; only authorized Javadoc wording changes.
- Documentation/planning are independently reviewed and synchronized; unrelated docs receive
  reasoned no-change conclusions.
- All validation passes with exactly twelve paths.

## Tests / validation

Run:

```bash
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.tensor.TensorGatherNdExpressionTest
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Manually verify `javap`, reflection, one-call delegation, helper shape,
semantic bytecode equivalence apart from documentation, imports, generated Javadoc, executable
example, Markdown links/anchors/fences/whitespace, exact twelve paths, synchronized status, and no
task-0018G specification.

## Dependencies

- Tasks 0001/0002 provide DataType, Shape, Dimension, and structural equality.
- Tasks 0011–0013 provide Tensor, derived identity, and provenance.
- Task 0018E provides exact Gather-ND semantics and normalized batch attributes.

## Follow-up tasks

- 0018G: axis scatter semantics and reduction policy.
- 0018I: Scatter-ND semantics and reduction policy.

Do not create follow-up specifications during this task.

## Architecture impact

Expected impact: None.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0002/0007/0011/0012/0013/0018C/0018D/0018E/0018F,
Tensor API, Compile API, Training API, glossary, current DataType/Dimension/Shape/TensorDescriptor/
Tensor/TensorFactory/TensorProvenance/Operation/GatherNdKind/GatherNdAttrs contracts and tests, and
Java 26 Gradle configuration.

Implement task 0018F exactly. Modify Tensor.java, add package-private final
TensorGatherNdExpressions.java, update TensorTest only for two methods, and add
TensorGatherNdExpressionTest. Add exactly gatherNd(Tensor) and gatherNd(Tensor,int).

The field-free helper has exactly eight methods. Follow exact null/type/rank/batch-prefix/static-
tuple-depth validation order/messages. Derive indices-prefix plus data-suffix Shape, including
canonical scalar. Preserve data type/eligibility, unresolved layout, exact GATHER_ND/attrs and
[data, indices] provenance, and one createDerived call. Never inspect index values.

Permit Javadoc-only current-status corrections in GatherNdKind/GatherNdAttrs; do not change their
declarations or behavior. Do not add scatter, gradients, compiler/runtime/backend behavior,
dependencies, build/architecture changes, or later specs. Stop beyond twelve paths.

Run all validation, then hand actual diff/evidence to a separate clean-context docs agent to
finalize permitted Javadocs/Tensor API/Compile API/glossary/planning and rerun validation.
Update task/master/roadmap only after both passes. Leave 0018G Draft without a spec. Do not commit.
```

## Local decisions

- The public no-batch method remains a one-call Tensor delegation, and the helper's two-argument
  entry delegates exactly once to the explicit entry with batch count zero. This keeps one
  validation and construction path without inventing a default attributes singleton.
- The exact eight-method, field-free helper owns only local descriptor and Shape work. It reads
  each descriptor once, validates metadata in the specified order, calls `Shape.ofDimensions`
  once for the result, and performs one final `TensorFactory.createDerived` call.
- Shared batch compatibility uses existing structural `Dimension.equals` semantics. Equal dynamic
  symbols pass even when represented by distinct immutable objects; different symbols fail rather
  than creating a compiler-style constraint.
- Static positive tuple depth remains an occurrence-specific final indices Dimension. The result
  preserves exact Dimension references from the indices prefix and data suffix, and the existing
  empty `Shape.ofDimensions` path supplies canonical scalar identity.
- `GatherNdKind` and `GatherNdAttrs` received only temporal Javadoc corrections from “later” or
  “future” input-aware construction to the current public Tensor boundary. Comparison against the
  pre-documentation `javap` baselines proves declaration and executable-bytecode equivalence.
- The Tensor API uses one complete executable example covering zero-batch, batched, scalar, and
  dynamic-Dimension cases. It exposes metadata and provenance only and makes no value, bounds,
  gradient, compiler, backend, materialization, or execution claim.

## Known limitations

- Gather-ND construction does not read, normalize, clamp, or bounds-check index values, including
  negative or out-of-range coordinates.
- Tuple depth must be statically known and positive. The model creates no symbolic tuple-length or
  dynamic-rank constraint.
- Shared batch Dimensions must already be structurally equal. Broadcasting and graph-wide symbol
  equivalence remain outside this local model boundary.
- Every result layout is unresolved and no storage is attached. Values, physical aliases,
  materialization, gradients or scatter-ND backward, graph capture/canonicalization, ONNX mapping,
  backend lowering, and execution remain owned by later layers and tasks.

## Validation evidence

- Clean implementation context `/root/implement_model_0018f` added the two public Tensor methods,
  exact helper, and focused tests, then handed the actual shared-tree diff and passing pre-handoff
  evidence to independent documentation context
  `/root/implement_model_0018f/review_model_0018f_docs`. The documentation pass used General plus
  API/Javadoc style for Java and API contracts, Planning style for this task/master/roadmap, and
  Example format for the complete Gather-ND example.
- The documentation context read `AGENTS.md`, the complete architecture contract and focused
  current/overview/lifecycle/module-boundary/dependency explanations, documentation workflow and
  selected profiles, planning guide and roadmap, model capabilities/master plan, tasks
  0001/0002/0007/0011/0012/0013/0018C/0018D/0018E/0018F, Tensor/Compile/Training API references,
  glossary, Java 26 root/model Gradle configuration, final implementation/tests, related model
  contracts, generated Javadoc, and the complete diff before editing.
- Related-contract review covered DataType, Dimension variants, Shape, TensorDescriptor, Tensor,
  TensorFactory, TensorProvenance, Operation, GatherNdKind, GatherNdAttrs, axis-gather semantics,
  and focused tests. Their behavior remains unchanged; only Tensor's new API/Javadocs and the two
  explicitly authorized semantic temporal Javadocs changed.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorGatherNdExpressionTest` — `BUILD SUCCESSFUL`; XML
  reports 10 tests, zero failures, errors, or skips.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorTest` — `BUILD SUCCESSFUL`; XML reports 14 tests,
  zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; XML aggregation reports 694 tests across
  81 suites with zero failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`. Generated `Tensor`, `GatherNdKind`, and
  `GatherNdAttrs` pages contain both overloads, exact parameters/results/failures, validation and
  Shape rules, current semantic boundary, provenance/metadata ownership, and explicit deferred
  value/gradient/compiler/backend/execution behavior. Package-private helper Javadocs were
  reviewed in source.
- `./gradlew test` — `BUILD SUCCESSFUL`; the repository lifecycle reports 36 actionable tasks
  with no failing task.
- `javap -classpath modules/model/build/classes/java/main -p -c -s` confirmed two one-call public
  delegates; a package-private final helper with no fields, one private constructor, and exactly
  eight methods; default-to-zero delegation; exact null/type/rank/attrs/batch-prefix/tuple/result/
  create order; `Math.addExact`; one result array; one `Shape.ofDimensions`; exact descriptor,
  operation, provenance, and final `createDerived` construction. Focused reflection tests confirm
  the same API and modifier shape.
- `cmp -s` of fresh `javap -p -c -s` output against `/tmp/gather-kind-after.txt` and
  `/tmp/gather-attrs-after.txt` passed, proving the Javadoc-only semantic edits leave declarations
  and executable behavior bytecode-equivalent.
- The documented `GatherNdExpressionExample` compiled with Java 26 against model main classes and
  ran successfully. It printed `[5, 4]`, `[2, 5, 4]`, canonical scalar identity, `[N, M, 4]`,
  `GatherNdAttrs[batchDimensions=1]`, exact dynamic/suffix Dimension retention, ordered exact
  provenance, and combined type/eligibility/unresolved-layout/no-label/no-storage facts.
- Source/import/manual inspection found only the expected local model/JDK dependencies and no
  value or storage read, graph/compiler/planning/prepare/runtime/backend/training dependency, new
  package, build option, reflection, registry, or service state. The focused storage test proves
  input arrays and attached storage remain untouched while the result is storage-free.
- The targeted Markdown validator resolved 417 local links, including 121 heading anchors, across
  Tensor API, Compile API, glossary, this task, model master plan, and roadmap with zero errors.
  Backtick fence counts were 218, 4, 4, 16, 2, and 0 respectively and therefore balanced; no
  tilde fences or trailing whitespace were found, and every changed path has a final newline.
- Final scope review found exactly the twelve authorized paths: Tensor, the new helper, the two
  Javadoc-only semantic contracts, TensorTest, the new focused test, Tensor API, Compile API,
  glossary, this task, model master plan, and roadmap. Task 0018G remains a Draft row and no
  task-0018G specification exists. `git diff --check` passed.
- Compile API changed only to list the current model expression input and explicitly retains graph
  capture, inference, optimization, artifacts, materialization planning, and engine behavior as
  planned. Training API remains accurate unchanged because no gradient rule/object, autograd,
  parameter, optimizer, publication, session, or training execution behavior changed.
- `capabilities.md` remains accurate unchanged because it already inventories Gather-ND with batch
  dimensions, exact integral index tensors, and separate model/public/compiler/backend/runtime
  support layers. `ARCHITECTURE.md`, focused architecture docs, ADRs, architecture tests,
  backend-conformance tests, integration tests, Gradle/dependencies, other modules, and later task
  specs remain unchanged because this task changes no ownership, dependency rule, backend
  behavior, executable end-to-end behavior, or build requirement.

## Implementation notes

- Added exactly `Tensor.gatherNd(Tensor)` and `Tensor.gatherNd(Tensor, int)` as one-call public
  instance delegations.
- Added the field-free package-private `TensorGatherNdExpressions` with exact metadata validation,
  structural batch-prefix checks, static tuple-depth validation, Shape derivation, and one final
  storage-free derived construction.
- Added the ten-test focused suite and expanded Tensor's exact public API assertions by two.
- Finalized Tensor/helper and semantic Javadocs, Tensor and Compile API current-status references,
  glossary terminology, the complete executable example, and synchronized planning evidence.

## Completion summary

- Completed changes: Implemented and documented public zero-batch and explicit-batch Gather-ND
  metadata construction with exact type/rank/batch/tuple/Shape validation and ordered provenance.
- Files changed or created: Exactly the twelve authorized production, test, API, glossary, task,
  master-plan, and roadmap paths.
- Tests and validation: Focused 10-test and 14-test suites, all 694 model tests across 81 suites,
  model Javadoc, root tests, bytecode/reflection/import/source/generated-page review, executable
  Java 26 example, 417-link/121-anchor checks, fence/whitespace/newline checks, exact scope/status
  and no-0018G-spec checks, semantic bytecode equivalence, and `git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0018f/review_model_0018f_docs` completed the mandatory independent pass
  with General, API/Javadoc, Planning, and Example profiles.
- Documentation impact: Tensor API, Compile API, and glossary now describe current Gather-ND
  expression construction while index-value bounds, gradients, compiler behavior, materialization,
  lowering, backend behavior, and execution remain separately owned.
- Javadoc review: Both public methods, the helper type/constructor/eight methods, Tensor type, and
  two authorized semantic timing corrections are final. Related model contracts remain accurate
  unchanged for the reasons recorded above.
- Glossary impact: Gather-ND now distinguishes semantic values from current input-aware Tensor
  construction, structural dynamic equality, static tuple depth, exact Dimension retention,
  canonical scalar results, and deferred value/executable behavior.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0018F. Task 0018G remains the next Draft frontier without a
  detailed specification.

Status: Complete
