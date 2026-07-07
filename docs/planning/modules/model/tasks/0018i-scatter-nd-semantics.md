# Task 0018I: Scatter-ND Semantics

## Status

Complete

## Goal

Define the typed, backend-independent semantic identity and immutable parameters for functional
tuple-index `SCATTER_ND`.

Scatter-ND consumes ordered logical inputs `[data, indices, updates]`. The final indices Dimension
is tuple depth `K`; each tuple addresses `K` data axes after a shared leading batch prefix. Updates
have the same Shape that Gather-ND would read:

```text
updates.shape == indices.shape[:-1]
                 + data.shape[batchDimensions + K:]
```

The conceptual result starts from `data`, applies the selected `ScatterReduction`, has exactly the
data Shape, and does not mutate data in place. This task defines meaning and intrinsic parameters
only; input-aware validation and public Tensor construction remain task 0018J.

## Scope

- Add one public `ScatterNdKind` enum implementing `OperationKind` with exactly `SCATTER_ND`.
- Add one public `ScatterNdAttrs` record implementing `OperationAttrs` with exactly
  `int batchDimensions` followed by `ScatterReduction reduction`.
- Require non-negative normalized batch count and explicit non-null reduction in exact order.
- Reuse the completed `ScatterReduction` enum without changing or duplicating it.
- Define exact ordered `[data, indices, updates]` roles.
- Define final-indices-Dimension tuple depth, shared batch prefix, indexed data axes, untouched
  data suffix, exact updates-Shape formula, data-shaped result, and functional non-mutation.
- Document replacement/addition/multiplication/maximum/minimum through the shared reduction values.
- Document `NONE` duplicate target tuples as invalid according to the existing reduction contract.
- Pair `SCATTER_ND` explicitly with `ScatterNdAttrs` without adding a compatibility validator.
- Add one same-package focused structural, validation, value-semantics, distinction, and
  composition test.
- Keep production in `io.github.pho001.synaptik.model.operation.index`.
- Finalize Javadocs, Tensor API, glossary, task evidence, master plan, and roadmap through the
  mandatory independent documentation pass.

## Out of scope

- public `Tensor.scatterNd` methods, default overloads, expression helper, factory, or task-0018J
- axis scatter, Gather-ND expression changes, gather backward, masks, slices, or another family
- storing data, indices, updates, ranks, Shapes, tuple depth, data types, descriptors, layout,
  requiresGrad, provenance, label, storage, or backend facts
- storing tuple depth in attributes; it remains occurrence-specific final indices Shape data
- validating batch count against ranks, shared batch Dimensions, tuple depth, updates Shape, data
  and update types, index type, result descriptor, or reduction/type compatibility
- reading index/update values, normalizing index values, bounds checks, negative-index policy,
  duplicate detection, write/reduction execution, or in-place mutation
- implicit default reduction, null-as-`NONE`, default constructor, singleton, separate zero-batch
  kind, sentinel batch count, factory, builder, registry, parser, visitor, or map
- numerical order, overflow, NaN, signed-zero, reproducibility, atomics, empty-domain behavior,
  gradient policy, or backend support
- graph capture, compiler, planning, prepare, runtime, backend, engine, trace, ONNX implementation,
  training, gradients, or execution behavior
- changing existing Java/tests, `ScatterReduction`, dependencies, Gradle, architecture, another
  module, or creating a task-0018J specification

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
- [Task 0005](0005-operation-semantic-foundation.md)
- [Task 0006](0006-operation-model.md)
- [Task 0018E](0018e-gather-nd-semantics.md)
- [Task 0018F](0018f-gather-nd-tensor-expressions.md)
- [Task 0018G](0018g-axis-scatter-semantics.md)
- [Task 0018H](0018h-axis-scatter-tensor-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only legacy branch represents ONNX-style ScatterND with one `SCATTER_ND` identity, one
reduction, and a non-negative `batchDims`. Its public API exposes zero-batch replacement, zero-
batch explicit reduction, and explicit reduction plus batch count.

Legacy Shape behavior uses the final indices extent as tuple depth, requires shared batch
Dimensions, and requires updates Shape equal to the Gather-ND result Shape. The conceptual result
always has data Shape. Replacement rejects duplicate tuple targets; arithmetic reductions combine
duplicates. The current model uses canonical rank-zero scalar Shape instead of legacy `[1]` where
the updates formula is scalar.

The new semantic value uses descriptive `batchDimensions`, explicit non-null reduction, and typed
immutable contracts. Legacy mutable Shapes, graph builders, gradient callbacks, traits, lowering,
kernels, and runtime/backend behavior are not copied.

## Architecture constraints

- Production remains in `modules/model`, which owns backend-neutral operation meaning.
- `ScatterNdKind` identifies mathematical tuple-scatter semantics only, not an occurrence, Tensor,
  graph node, descriptor, executable, kernel, or backend route.
- Ordered inputs are semantically `[data, indices, updates]`; attributes store none of them.
- `ScatterNdAttrs.batchDimensions` is already normalized and non-negative. It stores no ranks and
  cannot prove input compatibility.
- Tuple depth remains the final indices Dimension and must not be duplicated in attributes.
- The shared `ScatterReduction` is exact semantic vocabulary, not an algorithm or backend route.
- Result and updates Shape formulas are explanatory contracts; these types inspect no Shape.
- Generic `Operation` remains an open kind/attributes pair and validates no family pairing, arity,
  ranks, Shapes, types, bounds, duplicates, gradients, or backend support.
- Package direction remains `model.operation.index -> model.operation + java.base`.
- Stop if implementation requires Tensor, Shape, DataType, provenance, another production type,
  another test, dependency, architecture change, or cross-layer behavior.

## Package impact

No package is added or moved.

- `io.github.pho001.synaptik.model.operation.index.ScatterNdKind` — sole tuple-index scatter
  semantic identity.
- `io.github.pho001.synaptik.model.operation.index.ScatterNdAttrs` — normalized batch count plus
  explicit shared reduction.
- `ScatterNdSemanticsTest` — same-package structural, validation, composition, and boundary test.

The existing index package remains cohesive and independent of public Tensor and graph packages.

## Required contract

### Semantic kind

Create exactly:

```java
public enum ScatterNdKind implements OperationKind {
    SCATTER_ND
}
```

The enum declares no project field, explicit constructor, method, nested type, per-constant body,
alias, arity, batch count, tuple depth, reduction, Shape, type, result, cost, fusion, route, or
backend metadata. Inherited `Enum.name()` satisfies `OperationKind.name()`.

Javadoc must explain ordered inputs, functional non-mutation, final-Dimension tuple depth, batch
prefix, indexed axes, untouched suffix, updates formula, data-shaped result, reduction meaning,
duplicate-target boundary, and distinctions from Gather-ND and axis scatter.

### Immutable attributes

Create exactly:

```java
public record ScatterNdAttrs(
        int batchDimensions,
        ScatterReduction reduction) implements OperationAttrs
```

The record has exactly two components in that order, one canonical constructor, two explicit
documented accessors, and record-generated `equals`, `hashCode`, and `toString`. Add no tuple
depth, rank, Shape, input, result, default constructor, factory, builder, nested type, or extra API.

Constructor validation order and behavior are exact:

1. if `batchDimensions < 0`, throw `IllegalArgumentException` with exact message
   `batchDimensions must be non-negative: <batchDimensions>`;
2. require non-null `reduction` with `Objects.requireNonNull(reduction, "reduction")`;
3. otherwise retain both values unchanged.

Zero and `Integer.MAX_VALUE` batch counts are structurally valid because no ranks are present.
Every reduction is structurally valid. Null never means `NONE`.

### Typed composition

Compose explicitly:

```java
ScatterNdAttrs attrs =
        new ScatterNdAttrs(1, ScatterReduction.ADD);
Operation operation =
        new Operation(ScatterNdKind.SCATTER_ND, attrs);
```

The exact attributes reference is retained. Generic Operation does not enforce pairing. Add no
operation factory, default attributes, compatibility validator, or matrix.

### Shape terminology and examples

For data rank `R`, indices rank `Q`, batch count `B`, and final indices extent `K`:

- `Q >= 1`;
- `0 <= B < Q` and `B <= R`;
- leading data and indices Dimensions `[0, B)` match;
- `1 <= K <= R - B`;
- each tuple indexes data axes `[B, B + K)`;
- data axes `[B + K, R)` form the untouched suffix copied as one update slice;
- `updates.shape == indices.shape[0:Q-1] + data.shape[B+K:R]`;
- result Shape is exactly `data.shape`.

These are task-0018J input-aware rules, not constructor checks here.

Use these non-executable examples:

- data `[2, 3, 4]`, indices `[5, 2]`, `B=0`, `K=2`, updates `[5, 4]`, result `[2, 3, 4]`;
- data `[2, 3, 4]`, indices `[2, 5, 1]`, `B=1`, `K=1`, updates `[2, 5, 4]`, result `[2, 3, 4]`;
- data `[2, 3]`, indices `[2]`, `B=0`, `K=2`, canonical scalar updates `[]`, result `[2, 3]`.

No production code here stores operands, computes Shapes, reads values, or executes writes.

## Affected files

Expected implementation change:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/ScatterNdKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/ScatterNdAttrs.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/index/ScatterNdSemanticsTest.java`
- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task file
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

At most the eight paths above. Stop if implementation requires another production type/test,
existing Java edit, public Tensor behavior, Compile API/capabilities edit, dependency, build or
architecture change, another module, or a ninth path.

## Javadoc requirements

- Document every public type, constant, component/accessor, and canonical constructor.
- Define data, indices, updates, tuple depth, batch prefix, indexed axes, untouched suffix, target,
  reduction, and duplicate target before relying on them.
- Include the exact formula and all three Shape examples, including canonical scalar updates.
- Explain why tuple depth is not stored and why ranks/Shapes are deferred to task 0018J.
- Explain exact validation order, explicit reduction, and zero-batch convention.
- Explain `NONE` duplicate invalidity without promising detection here.
- Distinguish Scatter-ND from Gather-ND, axis scatter, and in-place mutation.
- Promise no gradients, compiler capture, backend support, bounds, numeric order, or execution.

## Acceptance criteria

- Exact one-constant public `ScatterNdKind` implements `OperationKind` with no project API/state.
- Exact two-component public `ScatterNdAttrs` implements `OperationAttrs` in required order.
- Negative-batch then null-reduction validation uses exact exception types/messages.
- Every valid batch/reduction is retained; record value methods remain generated.
- Exact kind/attributes composition works for every reduction through unchanged `Operation`.
- Javadocs define roles, formulas, examples, reduction/duplicate boundaries, and deferred checks.
- No Tensor, Shape, DataType, descriptor, provenance, gradient, graph, compiler, backend, or
  execution behavior is introduced, and existing `ScatterReduction` is unchanged.
- Tensor API, glossary, task/master/roadmap are independently finalized. Compile/Training API,
  capabilities, architecture, and related contracts receive reasoned no-change conclusions.
- All validation passes with exactly eight changed paths.

## Tests / validation

Run:

```bash
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.operation.index.ScatterNdSemanticsTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests cover exact enum/record structure and interfaces; component order; zero, ordinary,
and maximum batch values with every reduction; exact negative-first and null validation; generated
value semantics; exact composition/reference retention; distinctions from Gather-ND/axis scatter;
and absence of extra kinds, APIs, state, or cross-layer imports.

Manually inspect `javap -p -c -s`, reflection/source/imports, generated Javadoc, Markdown links,
anchors, examples, fences, whitespace, exact eight-path scope, synchronized status, unchanged
`ScatterReduction`, and absence of a task-0018J specification.

## Dependencies

- 0005 and 0006 define semantic foundation and open Operation pair.
- 0018E defines Gather-ND tuple/batch terminology and immutable batch parameter precedent.
- 0018G defines the shared exact reduction vocabulary and duplicate-target meaning.
- 0018H distinguishes completed one-axis scatter from tuple-index scatter.

## Follow-up tasks

- 0018J: public Scatter-ND Tensor expressions with type/rank/batch/tuple/Shape validation.
- 0023: compiler-generated backward semantics.

Do not create a follow-up specification here.

## Architecture impact

Expected impact: None. Stop if a new rule or cross-module dependency is required.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0005/0006/0018E/0018F/0018G/0018H/0018I, Tensor API,
Compile API, Training API, glossary, current OperationKind/OperationAttrs/Operation/GatherNdKind/
GatherNdAttrs/ScatterReduction and related index-family contracts/tests, and Java 26 Gradle.

Implement task 0018I exactly. Add only ScatterNdKind.java, ScatterNdAttrs.java, and
ScatterNdSemanticsTest.java under io.github.pho001.synaptik.model.operation.index.

ScatterNdKind contains exactly SCATTER_ND. ScatterNdAttrs contains exactly non-negative normalized
int batchDimensions then non-null ScatterReduction reduction, with exact validation order/messages
and explicit documented accessors. Reuse ScatterReduction unchanged. Document ordered
[data, indices, updates], final-Dimension tuple depth, shared batch prefix, indexed axes, untouched
suffix, updates formula/examples, exact data-shaped functional result, reduction meanings, and
invalid NONE duplicates without input-aware validation.

Do not add Tensor methods, Shape/DataType/result/provenance validation, new reduction types,
values/execution, gradients, graph/compiler/planning/runtime/backend behavior, dependencies,
build/architecture changes, existing Java edits, or later specs. Stop beyond eight paths or on
architecture uncertainty.

Run all validation, then hand actual diff/evidence to a separate clean-context docs agent. It must
inspect source/tests/Javadoc, finalize permitted Javadocs/Tensor API/glossary/planning, record
Compile API/Training API/capability/architecture and related-contract no-change conclusions, and
rerun validation.

Update 0018I, master plan, and roadmap only for status/evidence. Do not mark Complete before both
passes. Leave 0018J Draft without a specification. Do not commit/push.
```

## Local decisions

- One kind represents tuple-index scatter; reduction and batch count are explicit attributes.
- Structural parameter precedes reduction, matching axis-scatter attribute style.
- Tuple depth remains final indices Shape data, not duplicated attributes.
- Shared `ScatterReduction` is reused unchanged; null is never a default.
- `NONE` duplicate tuples are invalid but detection is value-aware and deferred.
- Independent documentation review found both initial production Javadocs substantively complete.
  The final coordinating audit then requested two small source-Javadoc clarifications: define a
  target and duplicate target tuples locally, and explicitly name task 0018J as the owner of
  operand-dependent rank/Shape validation. No declaration or behavior changed.

## Known limitations

- No ranks, Shapes, types, bounds, duplicates, descriptors, or provenance are validated/created.
- No numeric order, gradient, compiler, ONNX, backend, runtime, or execution behavior is defined.
- Public Scatter-ND remains task 0018J.

## Validation evidence

- Clean implementation context `/root/implement_model_0018i` added exactly `ScatterNdKind`,
  `ScatterNdAttrs`, and `ScatterNdSemanticsTest`. Independent documentation context
  `/root/implement_model_0018i/review_model_0018i_docs` inspected the shared-tree diff, final
  source/test, generated Javadoc, related contracts, APIs, glossary, planning state, and Java 26
  build configuration before finalizing documentation in the same overall change.
- The documentation pass applied General plus API/Javadoc style to the production Javadocs,
  Tensor API, and glossary; Planning style to this task, the model master plan, and roadmap; and
  Example format to the conceptual Shape examples. The final coordinating audit refined only the
  two production Javadocs to define target/duplicate-target terminology and explicitly name task
  0018J as the future owner of operand-dependent rank/Shape checks. Tensor API and glossary now
  explain current Scatter-ND semantics while public Tensor construction remains Draft task 0018J.
- Reviewed architecture and process material included `AGENTS.md`, `ARCHITECTURE.md`, the focused
  current architecture, overview, lifecycle, module-boundary, and dependency explanations;
  documentation rules and General/API-Javadoc/Planning/Example profiles; planning guide and
  roadmap; model capabilities/master plan; tasks 0005, 0006, and 0018E–0018I; Tensor, Compile,
  and Training API references; glossary; Java 26 root/settings/model Gradle configuration; final
  implementation/tests; generated model Javadoc; and the actual complete diff.
- Related-contract review covered `OperationKind`, `OperationAttrs`, `Operation`, `GatherNdKind`,
  `GatherNdAttrs`, `ScatterReduction`, `AxisScatterKind`, `ScatterElementsAttrs`, aggregate/scan
  reductions, their focused tests, and current Tensor/provenance boundaries. The new values
  compose the open semantic contracts without changing any existing declaration or behavior.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.index.ScatterNdSemanticsTest` — `BUILD SUCCESSFUL`;
  XML reports 9 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; XML aggregation reports 725 tests across
  84 suites with zero failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`; both generated public pages contain the
  enum/constant, record components, canonical constructor, explicit accessors, exact failures,
  ordered roles, both formula notations, all three examples, reductions, duplicate boundary,
  family distinctions, and exclusions.
- `./gradlew test` — `BUILD SUCCESSFUL`; all 36 root lifecycle tasks completed or were up-to-date
  with no failing task.
- `javap -classpath modules/model/build/classes/java/main -p -c -s` confirmed exactly one enum
  constant plus compiler-generated enum machinery. The record has exactly the two specified
  fields in order, negative-batch-first then non-null-reduction validation, two direct accessors,
  and generated record value methods. Focused reflection tests confirm the same API and state.
- Source, import, and `jdeps` inspection found only the permitted local `OperationKind`/
  `OperationAttrs` contracts and `java.base`. No Tensor, DataType, Shape, layout, provenance,
  graph, compiler, planning, prepare, runtime, backend, training, or execution dependency or
  behavior was introduced. `ScatterReduction` is unchanged.
- Targeted Markdown validation resolved 425 local links, including 131 heading anchors, across
  the five changed documentation/planning files with zero errors. Fence counts were 228, 4, 12,
  2, and 0 and therefore balanced. Trailing-whitespace scans found no matches, all eight paths
  have final newlines, and `git diff --check` passes.
- Final changed-path inventory contains exactly the eight authorized paths: two production
  contracts, one focused test, Tensor API, glossary, this task, model master plan, and roadmap.
  Task/master-plan/roadmap status is synchronized as Complete. Task 0018J remains Draft, and no
  task-0018J specification exists.
- Compile API remains accurate unchanged because this task adds no Tensor expression, graph
  capture, inference, validation, optimization, compile artifact, or engine behavior. Training
  API remains accurate unchanged because no gradient, autograd, parameter, optimizer,
  publication, session, or training execution behavior changed. The capability baseline already
  inventories Scatter-ND with batch dimensions, the shared five reductions, exact integral index
  types, and distinct support layers, so it required no edit.
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests,
  backend-conformance tests, integration tests, Gradle/dependencies, and other modules remain
  accurate unchanged because the task stays inside model-owned semantics and changes no module
  boundary, dependency rule, backend behavior, executable end-to-end behavior, or build
  requirement. Existing Operation, Gather-ND, axis-scatter, ScatterReduction, aggregate/scan,
  Tensor, provenance, and other operation-family contracts remain accurate unchanged for the
  related-contract reasons above.

## Implementation notes

- Added `ScatterNdKind` with exactly `SCATTER_ND` and no project-declared behavior or metadata.
- Added `ScatterNdAttrs(int batchDimensions, ScatterReduction reduction)` with exact validation
  order, direct accessors, and generated record value semantics while reusing unchanged
  `ScatterReduction`.
- Added nine focused tests covering exact structure, retained values, exact failures, value
  semantics, typed composition, family distinctions, and dependency boundaries.
- Finalized Tensor API, glossary terminology/inventories, and synchronized planning
  evidence/status through the independent documentation pass. A final coordinating audit added
  only the two source-Javadoc clarifications described above; declarations and behavior remain
  unchanged.

## Completion summary

- Completed changes: Implemented and documented functional tuple-index `SCATTER_ND` semantics and
  immutable normalized batch-count plus explicit shared-reduction attributes without public
  Tensor or cross-layer behavior.
- Files changed or created: Exactly the eight authorized production, test, API, glossary, task,
  master-plan, and roadmap paths.
- Tests and validation: Focused 9-test and all 725-model-test/84-suite runs, model Javadoc, root
  tests, javap/reflection/import/source/generated-page review, 425-link/131-anchor checks,
  fence/whitespace/newline checks, exact scope/status and no-0018J-spec checks, and
  `git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0018i/review_model_0018i_docs` completed the mandatory independent pass
  with General, API/Javadoc, Planning, and Example profiles.
- Documentation impact: Tensor API and glossary now define current Scatter-ND roles, tuple/batch
  model, updates formula, examples, reduction meanings, duplicate boundary, and distinctions,
  while public Tensor construction and every value/executable concern remain planned. Compile
  API, Training API, capabilities, architecture/ADRs/tests, conformance/integration material,
  Gradle, dependencies, other modules, and later tasks remain accurate unchanged for the recorded
  reasons.
- Javadoc review: Both new public types, the enum constant, record components, canonical
  constructor, accessors, failures, formulas, examples, and exclusions are complete; no correction
  to declarations or behavior was required. The final wording explicitly defines target and
  duplicate target tuples and names task 0018J for operand-dependent rank/Shape validation.
- Glossary impact: Added reusable Scatter-ND terminology and synchronized current
  `OperationKind`/`OperationAttrs` inventories without implying a public expression.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0018I. Task 0018J remains Draft without a detailed
  specification.

Status: Complete
