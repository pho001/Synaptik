# Task 0018J: Scatter-ND Tensor Expression

## Status

Complete

## Goal

Add public, locally validated Tensor-expression construction for functional tuple-index
`SCATTER_ND`.

The three public overloads consume ordered `[data, indices, updates]`, share one explicit
validation path, preserve exact data Shape/type, leave layout unresolved, and record exact
Scatter-ND semantics without reading index or update values.

## Scope

- Add exactly three public instance methods to `Tensor`:
  - `Tensor scatterNd(Tensor indices, Tensor updates)`
  - `Tensor scatterNd(Tensor indices, Tensor updates, ScatterReduction reduction)`
  - `Tensor scatterNd(Tensor indices, Tensor updates, ScatterReduction reduction,
    int batchDimensions)`
- Add one field-free package-private final `TensorScatterNdExpressions` helper with exactly the
  eleven methods specified below.
- Delegate the shortest overload with `ScatterReduction.NONE` and zero batch Dimensions.
- Delegate the reduction overload with the exact supplied reduction and zero batch Dimensions.
- Require exact `INT32`/`INT64` indices and exact matching data/update types.
- Permit `NONE` for every current data type; permit arithmetic reductions for floating/integral
  data and reject them for BOOL.
- Require indices rank at least one, valid normalized batch count, structurally equal batch prefix,
  static positive tuple depth, and exact updates Shape.
- Preserve exact data Shape/type and data/update gradient-eligibility OR in one unresolved result.
- Construct exact `SCATTER_ND`/attributes and ordered `[data, indices, updates]` provenance, then
  call `TensorFactory.createDerived` once with no label/storage.
- Keep every valid request fresh and metadata-only.
- Update `TensorTest` only for the three-method API expansion and add one focused expression test.
- Permit Javadoc-only current-status corrections to `ScatterNdKind` and `ScatterNdAttrs`; their
  declarations and behavior must not change.
- Finalize Tensor API, Compile API, glossary, task/master/roadmap through the mandatory independent
  documentation pass.

## Out of scope

- another Scatter-ND overload, primitive/collection indices, default indices, factories, static
  methods, or caller-supplied result Shape/type
- axis scatter changes, Gather-ND changes, gather/scatter backward, masks, slices, or other families
- floating/BOOL indices, conversion, promotion, mixed data/update types, or output type selection
- dynamic tuple depth, symbolic tuple constraints, broadcast batch Dimensions, or graph-wide
  inference
- reading/normalizing/clamping/bounds-checking index values or detecting `NONE` duplicate targets
- applying writes/reductions, in-place mutation, numerical order, overflow, NaN/signed-zero,
  reproducibility, atomics, or backend numeric policy
- resolved layout, aliases, storage access, materialization, device state, or backend route
- rejecting floating gradient eligibility for reductions without a current backward rule;
  eligibility metadata is not a backward-support promise
- modifying foundational/semantic behavior, dependencies, Gradle, architecture, another module,
  or creating task 0019 specifications
- gradients, graph capture, compiler, planning, prepare, runtime, backend, engine, trace, ONNX,
  training, or execution behavior

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
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
- [Task 0018E](0018e-gather-nd-semantics.md)
- [Task 0018F](0018f-gather-nd-tensor-expressions.md)
- [Task 0018G](0018g-axis-scatter-semantics.md)
- [Task 0018H](0018h-axis-scatter-tensor-expressions.md)
- [Task 0018I](0018i-scatter-nd-semantics.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only legacy branch exposes replacement with zero batch count, explicit zero-batch
reduction, and explicit reduction plus batch count. It validates tuple-index Shape relationships,
matching types, and arithmetic-reduction restrictions before graph construction.

The new boundary narrows indices consistently to `INT32`/`INT64`, uses canonical scalar updates
instead of legacy `[1]`, preserves immutable Dimension references, and never treats null as
`NONE`. Legacy value access, mutable Shapes, graph builders, gradient callbacks, traits, lowering,
kernels, and runtime/backend behavior are not copied.

## Architecture constraints

- `Tensor` remains public mutable API state, not IR. Methods create fresh storage-free expressions
  through the existing derived-factory seam.
- Kind, attributes, and reductions come only from completed tasks 0018G and 0018I.
- The helper may inspect immutable descriptors/Shapes, never values or storage.
- Index type, matching update type, ranks, batch prefix, tuple depth, and updates Shape are locally
  decidable. Index bounds and duplicates are not.
- Result always preserves exact data Shape/type and has unresolved layout.
- Gradient eligibility is data/update OR; indices do not contribute it.
- Compiler/training owns backward construction. Later layers own value-aware validation,
  lowering, materialization, and execution.
- No dependency, package, or module-boundary change is authorized.

## Package impact

No package is added or moved.

- `io.github.pho001.synaptik.model.tensor.Tensor` — three public Scatter-ND overloads.
- `io.github.pho001.synaptik.model.tensor.TensorScatterNdExpressions` — local validation, expected
  updates Shape, semantics, and provenance.
- `TensorScatterNdExpressionTest` — same-package focused coverage.
- `TensorTest` — exact public API inventory/reflection changes only.
- `ScatterNdKind` and `ScatterNdAttrs` — Javadoc-only temporal updates.

## Required contract

### Public Tensor methods

Add exactly:

```java
public Tensor scatterNd(Tensor indices, Tensor updates) {
    return TensorScatterNdExpressions.scatterNd(this, indices, updates);
}

public Tensor scatterNd(
        Tensor indices, Tensor updates, ScatterReduction reduction) {
    return TensorScatterNdExpressions.scatterNd(this, indices, updates, reduction);
}

public Tensor scatterNd(
        Tensor indices,
        Tensor updates,
        ScatterReduction reduction,
        int batchDimensions) {
    return TensorScatterNdExpressions.scatterNd(
            this, indices, updates, reduction, batchDimensions);
}
```

Each is one matching return/delegate, non-static, non-synchronized, with no field access or direct
validation.

### Helper shape

Create one package-private final field-free class, one private zero-argument constructor, and
exactly these eleven static methods:

```java
static Tensor scatterNd(Tensor data, Tensor indices, Tensor updates)
static Tensor scatterNd(
        Tensor data, Tensor indices, Tensor updates, ScatterReduction reduction)
static Tensor scatterNd(
        Tensor data,
        Tensor indices,
        Tensor updates,
        ScatterReduction reduction,
        int batchDimensions)
private static void validateIndexType(TensorDescriptor indicesDescriptor)
private static void validateMatchingDataType(
        TensorDescriptor dataDescriptor, TensorDescriptor updatesDescriptor)
private static void validateReductionDataType(
        TensorDescriptor dataDescriptor, ScatterReduction reduction)
private static void validateBatchDimensions(
        Shape dataShape, Shape indicesShape, int batchDimensions)
private static void validateBatchPrefix(
        Shape dataShape, Shape indicesShape, int batchDimensions)
private static int tupleDepth(
        Shape dataShape, Shape indicesShape, int batchDimensions)
private static Shape expectedUpdatesShape(
        Shape dataShape, Shape indicesShape, int batchDimensions, int tupleDepth)
private static Tensor create(
        Tensor data,
        Tensor indices,
        Tensor updates,
        TensorDescriptor dataDescriptor,
        TensorDescriptor updatesDescriptor,
        ScatterNdAttrs attrs)
```

Add no field, nested type, alternate constructor, overload, cache, mutable state, or extra method.

### Delegation

The shortest helper contains exactly:

```java
return scatterNd(data, indices, updates, ScatterReduction.NONE, 0);
```

The reduction helper contains exactly:

```java
return scatterNd(data, indices, updates, reduction, 0);
```

Neither performs independent validation.

### Explicit validation order

The five-argument helper performs exactly:

1. null-check data, indices, updates, reduction in order with parameter-name messages;
2. read data, indices, updates descriptors once each in order;
3. validate index type;
4. validate exact matching data/update type;
5. validate reduction/data-type compatibility;
6. read exact data, indices, updates Shapes once each in order;
7. reject rank-zero indices with
   `scatterNd indices rank must be at least 1`;
8. create one `ScatterNdAttrs(batchDimensions, reduction)`;
9. validate batch count fit;
10. validate shared batch prefix;
11. read/validate tuple depth once;
12. derive expected updates Shape once;
13. require exact updates equality or throw
    `scatterNd updates shape must equal indices prefix plus data suffix: expected=<expected>, actual=<actual>`;
14. call create once.

All failures precede result identity allocation.

### Type validation

Indices accept only `INT32`/`INT64`; otherwise:

```text
scatterNd indices data type must be INT32 or INT64: <actual>
```

Update mismatch is:

```text
scatterNd updates data type must match data: expected=<dataType>, actual=<updatesType>
```

BOOL with non-`NONE` reduction is:

```text
scatterNd BOOL data supports only NONE reduction: <reduction>
```

All floating/integral types accept all reductions; BOOL accepts `NONE` only.

### Batch and tuple validation

Validate batch count against indices rank first, then data rank:

```text
scatterNd batchDimensions must be less than indices rank: batchDimensions=<B>, indicesRank=<Q>
scatterNd batchDimensions must be less than data rank: batchDimensions=<B>, dataRank=<R>
```

Compare each shared batch Dimension in increasing order; mismatch is:

```text
scatterNd batch dimension at axis <axis> must match data: expected=<dataDimension>, actual=<indicesDimension>
```

The final indices Dimension must be static. Dynamic failure is:

```text
scatterNd tuple depth must be statically known
```

Static depth must lie in `[1, data rank - batchDimensions]`; otherwise:

```text
scatterNd tuple depth must be in [1, data rank - batchDimensions]: depth=<depth>, maximum=<maximum>
```

### Expected updates Shape

Derive exact Shape as indices prefix without final tuple depth followed by data suffix beginning at
`batchDimensions + tupleDepth`. Preserve every exact Dimension reference and use `Math.addExact`
for rank addition. Empty prefix and suffix produce canonical scalar Shape.

Examples inherited from task 0018I:

- data `[2,3,4]`, indices `[5,2]`, `B=0`, updates `[5,4]`;
- data `[2,3,4]`, indices `[2,5,1]`, `B=1`, updates `[2,5,4]`;
- data `[2,3]`, indices `[2]`, `B=0`, scalar updates `[]`.

### Common result construction

Create exactly one descriptor with exact data type, exact data Shape reference, empty layout, and
data/update eligibility OR; one Operation with `ScatterNdKind.SCATTER_ND` and exact attributes;
one provenance with exact ordered `[data, indices, updates]`; and one final
`TensorFactory.createDerived(descriptor, Optional.empty(), provenance)` call. Add no label or
storage. Every success is fresh.

## Affected files

Expected implementation change:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorScatterNdExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/ScatterNdKind.java`
  — Javadoc only
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/ScatterNdAttrs.java`
  — Javadoc only
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScatterNdExpressionTest.java`
- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task file
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

At most the twelve paths above. The bound combines one public expression family, two semantic
Javadoc temporal updates, public API documentation, and synchronized planning. Stop if another
production type/test, semantic behavior change, capability edit, dependency, build/architecture
change, another module, or a thirteenth path is required.

## Javadoc requirements

- Document all public/helper methods with parameters, results, constraints, ownership, identity
  side effects, and failures.
- Define data, indices, updates, batch prefix, tuple depth, target, reduction, and duplicates.
- Explain delegation defaults, exact validation order, formula, and all three examples.
- Explain type/reduction rules, exact data Shape/type retention, unresolved layout, and eligibility.
- Explain that values/storage/bounds/duplicates/writes/reductions are not inspected/executed.
- Correct stale task-0018J temporal wording in semantic Javadocs only; behavior stays unchanged.
- Promise no gradients, compiler capture, numeric order, backend support, or execution.

## Acceptance criteria

- Exactly three public Tensor methods and exact one-call delegates are added.
- Helper has exact class/constructor/field/eleven-method surface and two exact delegation bodies.
- Explicit path follows exact validation/construction order and messages.
- Type/reduction, rank, batch, prefix, tuple-depth, and updates-Shape rules are exact.
- Valid result retains exact data Shape/type, eligibility OR, unresolved layout, no label/storage,
  exact semantics/provenance, and fresh identity.
- No values/storage/bounds/duplicates are inspected and no writes/reductions execute.
- Both semantic types remain behaviorally bytecode-equivalent after Javadoc-only changes.
- Tensor/Compile API, glossary, task/master/roadmap are independently finalized. Training API,
  capabilities, architecture, and related contracts receive reasoned no-change conclusions.
- All validation passes with exactly twelve changed paths.

## Tests / validation

Run:

```bash
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.tensor.TensorScatterNdExpressionTest
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests cover exact API/helper/delegation; null/type/reduction/rank/batch/prefix/tuple/Shape
order and messages; all allowed/rejected data/index types and reductions; scalar/static/dynamic
Shapes; exact descriptor/provenance/fresh identity; and absence of value/storage/bounds/duplicate
inspection.

Manually inspect `javap -p -c -s`, reflection/source/imports, helper method count, generated
Javadoc, runnable example, Markdown links/anchors/fences/whitespace, exact scope/status, semantic
bytecode equivalence, and absence of a task-0019 specification.

## Dependencies

- 0001/0002: DataType, Shape, Dimension, static/dynamic facts.
- 0007/0011/0012/0013: descriptor, Tensor, identity, provenance.
- 0018F: completed Gather-ND validation/Shape pattern.
- 0018G/0018H: reduction type rules and functional-scatter result boundary.
- 0018I: exact Scatter-ND semantic kind and attributes.

## Follow-up tasks

- 0019: linear algebra and attention operations, to be planned only after this task completes.
- 0023: compiler-generated backward semantics.

Do not create a follow-up specification here.

## Architecture impact

Expected impact: None. Stop if a new rule, dependency, or cross-layer service is required.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0002/0007/0011/0012/0013/0018F/0018G/0018H/0018I/
0018J, Tensor API, Compile API, Training API, glossary, current Tensor/Shape/Dimension/descriptor/
factory/provenance, Gather-ND and scatter contracts/helpers/tests, and Java 26 Gradle.

Implement task 0018J exactly. Modify Tensor.java and add package-private final
TensorScatterNdExpressions.java. Update TensorTest only for exact three-method API expansion and
add TensorScatterNdExpressionTest. Add exactly three scatterNd overloads.

The field-free helper has exactly eleven methods. Follow exact null/descriptor/index-type/
data-update-type/reduction/rank/batch-prefix/tuple-depth/updates-Shape/construction order and
messages. Default overloads delegate with NONE/B=0 or supplied reduction/B=0. Preserve exact data
Shape/type, data/update eligibility OR, unresolved layout, exact SCATTER_ND attributes, ordered
[data, indices, updates] provenance, and one createDerived call. Never inspect values.

Permit Javadoc-only current-status corrections in ScatterNdKind and ScatterNdAttrs; do not change
declarations/behavior. Do not add gradients, compiler/runtime/backend behavior, dependencies,
build/architecture changes, or later specs. Stop beyond twelve paths or on uncertainty.

Run all validation, then hand actual diff/evidence to a separate clean-context docs agent. It must
inspect source/tests/Javadoc, finalize permitted Javadocs/Tensor API/Compile API/glossary/planning,
record no-change conclusions, and rerun validation.

Update 0018J, master plan, and roadmap only for status/evidence. Do not mark Complete before both
passes. Leave 0019 Draft without a specification. Do not commit/push.
```

## Local decisions

- Three overloads preserve baseline capability without a separate batch-only overload.
- Defaults delegate directly to the full path; null never means `NONE`.
- Validation mirrors Gather-ND, then compares updates against the derived Gather-ND result Shape.
- Result keeps exact data Shape/type with unresolved layout and eligibility OR.
- Bounds and duplicate validity remain value-aware responsibilities.

## Known limitations

- Index values, negative-index policy, bounds, and duplicates are not inspected.
- Numeric order, overflow, NaN/signed-zero, atomics, backend support, and execution are undefined.
- No gradient rule, graph capture, compiler/ONNX/planning/prepare/runtime/backend work is added.

## Validation evidence

Planning validation completed against repository rules, architecture, planning style, capability
baseline, completed Gather-ND/axis-scatter/Scatter-ND contracts, current Tensor patterns, and
read-only legacy capability/tests. Implementation validation then passed the focused Scatter-ND
suite with 10 tests, the Tensor API suite with 14 tests, and the full model suite with 735 tests
across 85 suites and no failures, errors, or skipped tests. Model Javadoc and the root test task
also passed.

Independent documentation review ran in clean context
`/root/implement_model_0018j/review_model_0018j_docs`. It read the architecture and focused
architecture documents, documentation rules with the General, API/Javadoc, Planning, and Example
profiles, planning guide and roadmap, model capability/master plans, prerequisite and adjacent
tasks, current Tensor/Compile/Training APIs, glossary, implementation, tests, generated Javadoc,
and Java 26 build configuration. The review finalized production Javadocs, Tensor and Compile API
status, glossary terminology, and synchronized planning evidence without changing semantic
behavior.

Post-documentation validation repeated the focused, Tensor API, full-model, model-Javadoc, and
root-test tasks. `javap` and reflection confirmed exactly three public methods, one final
package-private field-free helper, one private constructor, eleven static methods, exact delegate
bytecode, and the required explicit construction path. Semantic bytecode hashes remained
`1189d3dce3752ddfd96f9f8f2205bb8a8ed78bc102b7aeb6de88732b31beca4c` for
`ScatterNdKind` and
`06eb287e0f6685e97bc42c2731cb77c2ac0f667f70a8e75839f293333eae0c22` for
`ScatterNdAttrs`. Import/source/generated-page checks and the Java 26 example passed with the
documented zero-batch, batched, scalar-update, attribute, metadata, provenance, and freshness
output. Markdown 442-link/134-anchor and fence checks, whitespace/final-newline, exact twelve-path,
synchronized-status, and no-task-0019-specification checks also passed.

Training API needs no change because no gradient construction or training behavior was added.
Capabilities need no change because this task implements an already-recorded model capability.
Foundational, Gather-ND, axis-scatter, Scatter-ND semantic, and reduction contracts need no
behavioral change because this task consumes their current APIs; only the two explicitly permitted
semantic Javadocs needed temporal corrections. Architecture/ADR/architecture-test changes are
unnecessary because module boundaries and dependencies did not change. Backend conformance and
integration tests are unnecessary because no backend or executable behavior was added. Java 26
Gradle/dependencies, other modules, and later tasks remain unchanged because the implementation is
model-local and task 0019 stays Draft without a specification.

## Implementation notes

Implemented exactly three public overloads and the field-free eleven-method helper. The shared
path now performs the specified type, reduction, rank, batch-prefix, tuple-depth, and updates-Shape
validation before creating one fresh storage-free result with exact metadata and provenance.
Focused tests cover public/helper surfaces, delegation, validation ordering/messages, accepted and
rejected domains, Shape formulas, result metadata, provenance, freshness, value/storage
non-inspection, and identity exhaustion.

## Completion summary

Completed changes:

- added the three public `Tensor.scatterNd` overloads and one exact shared implementation helper;
- added focused expression tests and updated the exact public Tensor API inventory;
- corrected current semantic Javadocs without changing semantic declarations or bytecode; and
- finalized Tensor API, Compile API, glossary, task, model master plan, and roadmap documentation.

Files changed or created: exactly the twelve paths listed under [Affected files](#affected-files).
Validation: all automated and manual checks recorded in [Validation evidence](#validation-evidence)
passed. Unresolved issues: none. Required follow-up: none for task 0018J; task 0019 remains Draft
and may be planned separately.

Status: Complete
