# Task 0019A1: Embedding Convenience

## Status

Complete

## Goal

Add exactly one public `Tensor` convenience:

```java
public Tensor embedding(Tensor indices)
```

The receiver is the embedding-weight table. A successful call is exactly one validated
axis-zero `AxisGatherKind.GATHER` occurrence with ordered inputs `[weights, indices]`; it does not
introduce an `EMBEDDING` kind, an intermediate Tensor, or a second producer.

This task makes the common lookup-table spelling explicit while keeping compiler capture,
automatic differentiation (autograd), lowering, bounds enforcement, and execution attached to
the existing Gather semantic.

## Rationale and mental model

An embedding table is a rank-two floating Tensor:

```text
weights: [vocabulary, embeddingSize]
indices: [index axes...]
result:  [index axes..., embeddingSize]
```

This is exactly Gather on weight axis zero. Naming the composition improves the public API but
does not add a new computation for the compiler or backends to recognize.

### Newcomer example

Given floating weights of Shape `[10, 4]` and INT64 indices of Shape `[2, 3]`:

```java
Tensor embedded = weights.embedding(indices);
```

the result Shape is `[2, 3, 4]`. The two index axes are retained first and the exact weight
Dimension at axis one is appended. The result operation remains `GATHER` with
`new IndexAxisAttrs(0)` and exact provenance `[weights, indices]`. This example describes model
metadata; it does not read the six index values or the selected weight rows.

## Scope

- Add exactly `public Tensor embedding(Tensor indices)` to `Tensor`.
- Add exactly one package-private static `embedding(Tensor weights, Tensor indices)` method to
  the existing field-free `TensorAxisGatherExpressions` helper.
- Accept only a rank-two weight receiver with `BFLOAT16`, `FLOAT32`, or `FLOAT64` data type.
- Accept only exact `INT32` or `INT64` indices of any rank, including a scalar Shape.
- Validate the embedding-specific receiver contract, then delegate directly to the existing
  package-private `gather(weights, indices, 0)` method.
- Preserve the exact existing Gather operation, attributes, result metadata, producer, provenance,
  identity-allocation, and value-bounds boundaries.
- Add focused tests and update every current exact public/helper inventory affected by one method.
- Finalize public Javadoc, Tensor API, Compile API, glossary impact, and planning records through a
  mandatory separate clean-context documentation pass.

## Out of scope

- an `EMBEDDING` operation kind, attributes type, signature, alias, registry entry, compiler
  decomposition, or fused model semantic
- weight ranks other than two; integral, BOOL, FLOAT16, quantized, sparse, or complex weights
- a padding index, sparse-gradient flag, maximum-norm option, frequency scaling, negative-index
  wrapping, clamping, default row, or configurable table axis
- a mutable embedding-table abstraction, parameter wrapper, initializer, eager value lookup,
  host-storage allocation, result storage, resolved result layout, or input mutation
- gradient rules, scatter-add adjoints, backward graph construction, optimizer behavior, or
  changes to `requiresGrad` eligibility contracts
- compiler capture or constant propagation, backend lowering or kernels, prepare-time proof,
  runtime bounds checks, execution, backend conformance, or integration support
- changes to `AxisGatherKind`, `IndexAxisAttrs`, `Operation`, `OperationSignature`, `DataType`,
  `Shape`, `Dimension`, `TensorDescriptor`, `TensorFactory`, `TensorProducer`,
  `TensorProvenance`, Gradle, another module, `ARCHITECTURE.md`, or focused architecture docs
- one-hot encoding, dropout/RNG, sorting/top-K, linear, attention, or any later task

## Exact API and delegation design

### Receiver and argument roles

Add exactly:

```java
public Tensor embedding(Tensor indices)
```

The receiver is named `weights` inside the helper and is ordered producer input zero. `indices`
is ordered producer input one. Add no static form and no overload. Place the public method next to
the axis-gather surface, immediately after `gather(Tensor, int)` and before `gatherElements`.

Weights must have exact rank two:

```text
axis 0 = vocabulary extent
axis 1 = embeddingSize
```

The accepted weight types are exactly the current floating types `BFLOAT16`, `FLOAT32`, and
`FLOAT64`. Indices are exactly `INT32` or `INT64`; their `requiresGrad` is necessarily false under
the current descriptor contract.

### Direct Gather delegation

`Tensor.embedding(indices)` is a one-call facade over
`TensorAxisGatherExpressions.embedding(this, indices)`. After the validation below, the helper
must return exactly:

```java
gather(weights, indices, 0)
```

Do not call the public `weights.gather(...)` method and do not construct a descriptor, operation,
producer, Tensor, or eager index carrier in `embedding`. The existing helper call constructs the
sole result. There is no intermediate public call result and no second ID allocation.

The final operation is exactly:

```java
new Operation(AxisGatherKind.GATHER, new IndexAxisAttrs(0))
```

Its family-owned signature remains the existing exact two-input, one-output signature.

## Shape and result metadata

For weights Shape `[vocabulary, embeddingSize]`, the result Shape is:

```text
indices.shape + [embeddingSize]
```

This follows the existing Gather replacement formula at axis zero. The result:

- retains every indices `Dimension` reference exactly and in order;
- appends the exact weight axis-one `Dimension` reference;
- omits the weight vocabulary `Dimension` from the result;
- accepts static, named dynamic, and expression Dimensions without binding or rewriting them;
- accepts scalar indices, producing rank-one Shape `[embeddingSize]`;
- accepts zero extents structurally, including zero vocabulary or zero embedding size;
- creates one new non-scalar Shape value through the existing Gather path;
- has the exact weight data type and `requiresGrad` value;
- always has unresolved layout, no label, and no host storage; and
- leaves weights and indices, including their descriptor, label, storage, provenance, and ID,
  unchanged.

`requiresGrad` comes only from weights because Gather preserves data-input eligibility. Indices do
not contribute eligibility. This descriptor fact is not a gradient rule: compiler autograd later
owns any repeated-index scatter-add construction.

Each valid call creates one fresh single-output `TensorProducer`, one provenance object at output
index zero, and one newly allocated Tensor ID. The producer retains the exact existing operation,
the exact ordered input references `[weights, indices]`, and the exact sole result descriptor.
Repeated valid calls return identity-distinct results and producers.

## Validation order, exact messages, and ID effects

`TensorAxisGatherExpressions.embedding(weights, indices)` performs this exact order:

1. `Objects.requireNonNull(weights, "weights")`;
2. `Objects.requireNonNull(indices, "indices")`;
3. require exact weight rank two;
4. require a floating weight data type;
5. call the existing `validateIndexType("embedding", indices.descriptor())`;
6. delegate exactly once to `gather(weights, indices, 0)`.

The unchanged Gather method repeats its own null and index-type checks before Shape construction.
Do not bypass or refactor those checks in this task. They cannot allocate an ID and preserve one
canonical Gather construction path; the embedding-specific precheck exists to provide the exact
public convenience failure name before delegation.

Use these exact task-owned messages:

```text
embedding weights rank must be 2: actual=<rank>
embedding weights data type must be BFLOAT16, FLOAT32, or FLOAT64: <type>
embedding indices data type must be INT32 or INT64: <type>
```

Null failures use exact messages `weights` and `indices`. Rank validation precedes weight-type
validation, which precedes index-type validation. The only embedding-specific Shape validation is
exact weight rank two. There is no indices-rank restriction and no static extent restriction;
rank two makes axis zero valid, and the unchanged Gather helper owns result-Shape construction.

Every null, rank, type, or local metadata failure occurs before the sole final
`TensorFactory.createDerived` delegation and consumes no Tensor ID. The successful path allocates
exactly one ID. Identifier exhaustion retains the existing exact message
`tensor identifier space exhausted`; there is no rollback or alternative allocation path.
Focused tests may inspect the shared ID counter for this concrete no-ID-on-validation-failure risk.

## Index values and bounds ownership

Model construction validates index representation, not index contents. It reads neither host
storage nor values, so even eager constant indices are not inspected here. A valid executed lookup
requires each index value `i` to satisfy:

```text
0 <= i < bound vocabulary extent
```

Negative and out-of-range values are invalid. They do not wrap, clamp, select a padding/default
row, or produce zero. Because the retained operation is ordinary Gather, this is not an
embedding-only execution branch: future Gather compiler/backend support must apply its common
safe bounds contract. Compiler validation may reject captured constant invalid indices when it
can prove them. After dynamic extents are bound, backend preparation or the prepared executable
must prove or check bounds and fail safely rather than perform an out-of-bounds read. The exact
future execution exception/status belongs to the later execution contract and is not invented by
this model task. Runtime executes prepared behavior and must not inspect the original `Operation`.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Planning guide](../../../planning-guide.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)

## Architecture constraints

- Work remains inside the model-owned public Tensor and existing Gather-construction boundary plus
  its documentation/planning records.
- `Tensor` remains public mutable API state and is not graph IR.
- `Operation` retains backend-independent `GATHER` meaning only; it gains no embedding identity,
  support metadata, route, kernel, or execution state.
- Package direction remains `model.tensor -> model.operation.index`, datatype, shape, and layout.
- Compiler owns capture, graph-wide validation, constant analysis, autograd, and backward-graph
  construction. Backend prepare owns lowering, fusion, specialization, bounds strategy, and kernel
  choice. Runtime executes prepared work only.
- No architecture, dependency, lifecycle, module-boundary, build, or cross-module change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor`
- `io.github.pho001.synaptik.model.datatype`
- `io.github.pho001.synaptik.model.shape`
- `io.github.pho001.synaptik.model.operation.index`

Packages added or changed:

- No package is added. The existing `model.tensor` package gains one public facade method and one
  package-private helper method.

Type placement:

- `io.github.pho001.synaptik.model.tensor.Tensor` — existing owner of the public fluent convenience.
- `io.github.pho001.synaptik.model.tensor.TensorAxisGatherExpressions` — existing package-private
  owner of Gather validation, Shape derivation, and final derived-Tensor construction.

Tests mirror the production package because they inspect package-private helper and producer
contracts.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorAxisGatherExpressions.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorEmbeddingExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorAxisGatherExpressionTest.java`
  — update only the exact helper method inventory for `embedding`.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java` — add the exact
  public signature/name inventory and change the total public method count from 160 to 161.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
  — change only the shared total public Tensor method count from 160 to 161.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
  — change only the shared total public Tensor method count from 160 to 161.

Documentation/planning:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task file
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless the final implementation makes a current statement inaccurate:
Training API; axis-gather semantic Javadocs; DataType, Shape/Dimension, TensorDescriptor,
TensorFactory/createDerived, producer/provenance, operation-signature, and related expression
contracts; architecture/ADRs/tests; conformance/integration; Gradle; other modules.

## Maximum scope

At most two production, five test, and seven documentation/planning files: exactly 14 paths.
`Tensor.java` changes only for the exact method and its Javadoc. The axis-gather helper changes
only for `embedding`, its documentation, and the helper inventory it necessarily changes. The
three historical API-count tests change only `160` to `161` plus `TensorTest`'s exact name and
signature inventory.

Stop before a fifteenth path, a new production helper/type, any semantic/attribute/foundation
edit, another existing test edit, cross-module work, or architecture/build change.

## Javadoc and explanatory documentation requirements

- Add complete Javadoc to `Tensor.embedding` and the helper `embedding` method, including receiver
  and argument roles, exact signature, accepted types/ranks, Shape formula and Dimension identity,
  metadata, delegation, producer/provenance/ID behavior, validation messages/order, value-bounds
  boundary, failures, and cross-layer exclusions.
- Update the helper type Javadoc only as needed to include its embedding validation/composition
  role without weakening the existing Gather contract.
- Add the `[10, 4]` weights and `[2, 3]` indices newcomer example to the Tensor API. Explain that
  `[2, 3, 4]` is metadata and that no row values were read.
- Update Compile API only to keep the current public-expression inventory and planned compiler/
  autograd/bounds boundary accurate. Do not claim compiler support.
- Do not change Training API: its current compiler-owned autograd and backend-independent training
  boundary remains accurate because this convenience adds no parameter, gradient, optimizer,
  publication, session, or training-execution contract. Record that no-change conclusion.
- Update the existing Gather glossary entry or add an `Embedding` entry only if the independent
  documentation pass finds the reusable convenience distinction needs its own navigation term.
- Keep capabilities, task, master plan, and roadmap synchronized. Record reasoned no-change
  conclusions for Training API and every reviewed unchanged contract.

## Acceptance criteria

- Exactly one public `embedding(Tensor)` method exists; the public Tensor method count is 161.
- Weights are rank two and exactly floating; indices are exactly INT32/INT64 and may have any Shape.
- Result Shape is the complete indices Shape plus the exact weight axis-one Dimension, including
  scalar, zero, static, named dynamic, and expression-dimension cases.
- Validation follows the exact order/messages and every local failure consumes no ID.
- The public method delegates once to the helper; the helper delegates once directly to existing
  `gather(weights, indices, 0)` and constructs no intermediate Tensor or public call.
- Every success is exactly one `GATHER`/axis-zero/two-input/one-output occurrence, one producer,
  provenance output index zero, and one fresh ID, with exact `[weights, indices]` references.
- Result type and `requiresGrad` come only from weights; layout is unresolved; label and storage are
  absent; inputs are unchanged.
- Tests establish metadata and construction behavior without eager value execution or inventing a
  gradient/backend/runtime contract.
- Negative/out-of-range values remain unread during construction and are documented as invalid at
  execution, with no wrap/clamp/padding/default behavior and safe enforcement deferred to the
  future ordinary Gather execution contract.
- No `EMBEDDING` kind, padding/sparse/max-norm/frequency option, mutable table, gradient rule,
  compiler/backend/runtime implementation, dependency, Gradle, architecture, or other-module work
  lands.
- Focused tests, exactly one final model test after Java stability, final model Javadoc,
  documentation/link/scope/status/formatting checks, and `git diff --check` pass.
- A separate clean-context documentation pass finalizes all authorized documentation, reuses the
  final Java evidence, and records its context and no-change conclusions.
- 0019 and 0019A remain Complete; 0019A1 becomes Complete only after all evidence; 0019A2,
  0019B–0019E, and later tasks remain Draft without detailed specifications.

## Tests / validation

Required focused command during implementation:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorEmbeddingExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorAxisGatherExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorTest
```

After executable Java stabilizes, run exactly one final model suite:

```bash
./gradlew :modules:model:test
```

Focused tests cover exact public/helper surfaces and one-call delegation; rank/type/null ordering
and messages; no-ID local failures; all weight/index types; scalar, zero, static, named dynamic,
and expression Dimensions; exact descriptor/provenance/producer/freshness; one-ID success; input
non-mutation; and absence of an embedding kind or value inspection.

The mandatory separate clean-context documentation pass receives and reuses the successful final
model-test evidence. It does not rerun Java tests unless it changes executable Java behavior or
records a concrete risk. After final Javadoc edits it runs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

It also validates local Markdown links and anchors, balanced fences, terminology, generated
Javadoc, final newlines, trailing whitespace, exact 14-path scope, package placement, the
161-method and seven-helper-method inventories, exact Ready/Complete/Draft synchronization, one
Ready model frontier, and absence of a task-0019A2 or later detailed specification.

Repository-wide validation is deferred to the selected modern-operation capability checkpoint
after task 0022 and to CI. This is a task-tier, single-module convenience with no dependency,
architecture, build, or cross-module change.

## Dependencies

- Completed task 0018K supplies exact operation signatures and local occurrence validation.
- Completed task 0018L supplies the one-producer/output-index provenance contract.
- Completed task 0018O supplies the final canonical `GATHER` taxonomy and public/helper path.
- Completed tasks 0001, 0002, 0007, and 0013 supply DataType, exact Dimension/Shape retention,
  descriptor eligibility, derived construction, identity, and provenance foundations.

## Follow-up tasks

- 0019A2: focused one-hot encoding semantics. It remains Draft without a detailed specification.
- 0023: compiler-generated semantic operations may later own Gather adjoints; it remains Draft and
  does not implement autograd traversal in this task.

Do not create either specification during task 0019A1.

## Architecture impact

Expected impact: None.

This task adds a model-owned public convenience over an existing model-owned semantic occurrence.
If implementation requires a new semantic identity, cross-module dependency, or different
compiler/backend ownership, stop and report the concrete conflict.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, focused module/dependency/lifecycle/training architecture docs,
docs/developer-guide/documentation-rules.md and applicable profiles, the planning guide and
roadmap, model capabilities/master plan, completed foundation/provenance/signature/indexing tasks,
completed tasks 0019/0019A, this task, Tensor/Compile/Training APIs, glossary, and the current
Tensor, TensorAxisGatherExpressions, Gather semantics/attributes, Shape/Dimension, DataType,
TensorDescriptor, TensorFactory/createDerived, producer/provenance, signatures, tests, and API
inventories.

Implement task 0019A1 exactly. Add only public weights.embedding(indices) and the existing
axis-gather helper's embedding validation/delegation method. Preserve one ordinary axis-zero
GATHER occurrence, exact [weights, indices] provenance, one producer, one ID, weight-only
requiresGrad, exact indices Dimensions plus embedding Dimension, and unresolved layout. Follow
the specified validation order/messages and no-ID failures.

Do not add an EMBEDDING kind, another helper/type/overload, padding/sparse/max-norm/frequency
options, eager value access, a mutable table, gradient/compiler/backend/runtime behavior,
dependencies, build/architecture changes, other modules, or later specs. Stop beyond the exact
14 paths or on an architecture/scope conflict.

Run the focused command and exactly one final model suite after Java stability. Then hand the
actual diff, task, affected API/behavior, architecture constraints, drafted Javadocs/docs, and
exact Java evidence to a separate clean-context documentation-focused agent in the same overall
change. That pass must independently inspect source/tests, finalize permitted Javadocs, Tensor
API, Compile API, glossary impact, capabilities/task/master/roadmap, run final model Javadoc and
documentation validation, reuse Java evidence unless executable behavior changes, and record
reasoned no-change conclusions.

Update task evidence, implementation notes, completion summary, and synchronized status only
after both passes finish. Keep 0019/0019A Complete and 0019A2 plus 0019B-0019E/later Draft without
detailed specs. Mark Complete only when every acceptance criterion passes.
```

## Separate documentation handoff

The implementation agent must provide the documentation-focused clean context with:

- this task and the exact final implementation/test diff;
- the affected `Tensor.embedding` and helper behavior, existing Gather semantics, and exact Java
  test command/results;
- architecture constraints and the explicit no-new-kind/no-cross-layer boundary;
- drafted Javadocs and expected Tensor API, Compile API, glossary, capabilities, task, master, and
  roadmap impact; and
- required Javadoc, generated-page, example, Markdown, terminology, scope, status, and whitespace
  checks.

The documentation pass applies General plus API/Javadoc style to public/explanatory content,
Planning style to planning records, and Example format to the newcomer example. It must name its
clean context and record reviewed files, changes, commands/results, limitations, glossary outcome,
and reasoned no-change conclusions. The task remains incomplete until this evidence lands in the
same overall change.

## Local decisions

- Kept `embedding` as a thin public convenience and package-private validation boundary over the
  existing Gather construction path. The helper performs the task-owned receiver checks and then
  returns `gather(weights, indices, 0)` directly; no descriptor, producer, or Tensor is created by
  the convenience itself.
- Added a glossary `Embedding` entry because the public convenience is a reusable navigation term
  whose distinction from a separate operation kind and from configurable training-framework
  embedding layers is useful beyond one API paragraph.
- Placed the complete `[10, 4]` by `[2, 3]` metadata example inside the existing axis-gather API
  section. It demonstrates Shape, operation, attributes, ordered inputs, output index, eligibility,
  and storage boundaries without implying eager row selection.
- Kept execution ownership phrased as the ordinary Gather contract: compiler constant analysis
  may reject provably invalid captured indices; preparation or prepared execution must enforce
  bounds after binding. The model convenience itself does not select an exception or status.

## Known limitations

- Model construction intentionally accepts stored negative and out-of-range index values because
  it does not inspect values. Such values are invalid for later ordinary Gather execution.
- No padding index, sparse-gradient flag, maximum-norm option, frequency scaling, negative-index
  wrapping, configurable table axis, gradient rule, compiler capture, backend lowering, or
  execution support is included.
- Repository-wide validation remains deferred to the selected modern-operation checkpoint after
  task 0022 and CI. This task changed one module without changing dependencies, architecture, or
  shared build configuration.

## Validation evidence

- Implementation context ran the exact focused command:

  ```bash
  ./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorEmbeddingExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorAxisGatherExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorTest
  ```

  It passed. Focused coverage included the exact facade/helper surface, validation order and
  messages, no-ID failures, type/rank/Shape cases, exact Dimension retention, producer/provenance
  identity, input non-mutation, and construction-time non-inspection of stored invalid values.
- After executable Java stabilized, the implementation context ran exactly one final command:

  ```bash
  ./gradlew :modules:model:test
  ```

  It passed with 94 suites and 765 tests, with zero failures, errors, or skips. `git diff --check`
  also passed after Java stability. The documentation context reused this evidence and did not
  rerun Java tests because it changed only Javadocs and Markdown afterward.
- Separate clean-context documentation review identity:
  `/root/task_0019a1_documentation`. It applied General style, API/Javadoc style, Planning style,
  and Example format. It independently read the implementation and tests; finalized both
  production Javadocs and all seven authorized documentation/planning paths; and reviewed the
  final Tensor, Gather, descriptor, producer/provenance, Shape/Dimension, operation-signature,
  compiler, training, architecture, testing, and build boundaries.
- Documentation context ran:

  ```bash
  ./gradlew :modules:model:javadoc
  ```

  It passed on OpenJDK 26.0.1 (`BUILD SUCCESSFUL`, two executed tasks). The generated public
  `Tensor.html` contains the `embedding(Tensor)` signature, direct ordinary axis-zero Gather
  explanation, accepted types/Shape, value-bound boundary, and unsupported options. The
  package-private helper Javadoc was checked in source because the default public Javadoc task
  does not emit that package-private type.
- A Java 26 compile-and-run check of the documented `EmbeddingExpressionExample` passed and
  printed exactly `Shape[2, 3, 4]`, `GATHER`, `IndexAxisAttrs[axis=0]`, two true ordered-input
  identity checks, output index `0`, true weight eligibility, and true empty layout/label/storage.
- `javap`, reflection, import, and source checks passed. They confirmed exactly one public
  `Tensor.embedding(Tensor)`, one package-private static helper, 161 public Tensor methods, seven
  helper methods, the required `AxisGatherKind` imports, the one-call public delegation, and the
  exact `return gather(weights, indices, 0)` helper delegation.
- Final documentation validation checked local Markdown targets and heading anchors, balanced
  fences, final newlines, trailing whitespace, terminology, the generated page, and the runnable
  example. It also confirmed exactly 14 changed paths; no architecture/build/other-module path;
  0019, 0019A, and 0019A1 Complete; 0019A2 and 0019B–0019E/later Draft; no later detailed task
  specification; no Ready model task; and a final successful `git diff --check`.
- Training API no-change conclusion: the page already assigns global autograd to the compiler and
  backend-specific lowering to backend owners. Embedding adds no parameter, gradient, optimizer,
  publication, session, or training-execution contract, so changing that page would be inaccurate
  scope expansion.
- Architecture and focused architecture documentation required no change because module ownership,
  dependency direction, lifecycle, and runtime hot-path rules are unchanged. Architecture tests,
  backend conformance, and integration tests required no change because no boundary, backend, or
  end-to-end behavior changed. Compile API required only an inventory and planned-ownership
  clarification; it does not claim compiler support. Gradle and other modules required no change.

## Implementation notes

- Added `Tensor.embedding(Tensor)` immediately after `gather` and added exactly one package-private
  `TensorAxisGatherExpressions.embedding(weights, indices)` method.
- The helper checks weights null, indices null, weight rank two, floating weight type, and exact
  index type in the specified order, then delegates directly to existing axis-zero Gather.
- Added focused embedding tests and updated only the specified helper/public inventories and three
  historical public-method counts. No executable Java changed after the final model test.
- Final documentation explains receiver/indices/result roles, exact Shape and Dimension identity,
  one ordinary producer/output-index-zero/ID occurrence, weight-only metadata, unresolved result
  layout, value-bound ownership, unsupported options, and planned compiler/autograd/backend/runtime
  responsibilities.

## Completion summary

- Completed changes: added the validated rank-two floating embedding convenience as exactly one
  ordinary axis-zero Gather occurrence, plus focused tests, complete Javadocs, API/glossary
  explanation, and synchronized planning status.
- Files changed or created: two production Java files, five model test files, Tensor API, Compile
  API, glossary, model capabilities, this task, model master plan, and roadmap; exactly 14 paths.
- Tests and validation: focused tests and the final 94-suite/765-test model run passed in the
  implementation context; final model Javadoc, Java 26 example/metadata checks, generated-page,
  `javap`, reflection, import/source, Markdown, scope/status, and whitespace checks passed in the
  documentation context.
- Documentation-agent review: `/root/task_0019a1_documentation` completed the mandatory targeted
  independent pass without changing executable behavior or rerunning Java tests.
- Documentation impact: Tensor and Compile API, glossary, capabilities, task, master plan, and
  roadmap finalized. Training API, architecture documentation, test guides, and other-module docs
  remain accurate without changes for the reasons recorded above.
- Javadoc review: both affected production contracts are complete; public generated Javadoc passed.
- Glossary impact: added `Embedding` to distinguish the convenience from a semantic kind or
  configurable training-layer abstraction.
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
