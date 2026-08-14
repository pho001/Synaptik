# Task 0007: Embedding Layer

## Status

Complete

## Goal

Add one final public `Embedding` module that owns exactly one positive fully static rank-two
floating table `Parameter` and constructs indexed lookup metadata only through the existing
`Tensor.embedding` convenience. The layer retains caller-supplied state, exposes its stable
parameter handle, and remains mode-insensitive without introducing an embedding operation,
initialization policy, padding behavior, or execution claim.

## Scope

- Add final public `io.github.pho001.synaptik.nn.layers.Embedding` extending the existing
  `io.github.pho001.synaptik.nn.module.Module`.
- Add exactly this public constructor and these methods:

  ```java
  public Embedding(Tensor weight)

  public Parameter weight()

  public Tensor forward(Tensor indices)
  ```

- Declare exactly one parameter under local name `weight`. It is the lookup table with exact
  Shape orientation `[vocabularySize, embeddingSize]`; axis zero contains rows selected by an
  index value, and axis one is appended to every result as the embedding Dimension.
- Require the supplied weight to be non-null, floating, gradient-eligible, rank two, fully static,
  and strictly positive on both axes. Retain its exact Tensor reference through the existing
  `Parameter` declaration. Do not require a label, host storage, provenance absence, dense or
  resolved layout, or a particular Tensor identity.
- Validate supplied construction in exact order: null weight, floating data type, gradient
  eligibility, rank two, fully static Shape, positive vocabulary extent at axis zero, then
  positive embedding extent at axis one. Complete validation before declaring the parameter.
  Constructor validation creates no Tensor, expression, storage, random draw, or Tensor identity
  and never mutates the supplied value.
- `weight()` returns the exact stable wrapper declared by the layer. Add no setter or layer-local
  replacement facade; the existing public schema-compatible `Parameter.replace(Tensor)` remains
  the only public update capability.
- `forward(indices)` first rejects null with message `indices`, reads the current weight binding
  exactly once, and returns exactly `currentWeight.embedding(indices)`. Do not reproduce rank,
  index-type, Shape, operation, descriptor, producer, provenance, bounds, or identity logic in NN.
- Forward inherits Model's current contract: indices must use exact `INT32` or `INT64` and may
  have any rank, including scalar; result Shape is the complete indices Shape followed by the
  exact current weight axis-one Dimension; result type and gradient eligibility come only from
  the current weight; and the sole occurrence is ordinary axis-zero `GATHER` with ordered
  `[weight, indices]` provenance.
- Forward construction reads no index value. Negative or out-of-range values remain invalid for
  eventual ordinary Gather execution; the layer adds no wrapping, clamping, default row, or
  padding interpretation.
- `Embedding` is mode-insensitive. Its forward method accepts no `ForwardContext`, does not inspect
  `mode()`, and constructs the same Model expression contract in training and evaluation modes.
- A compatible `Parameter.replace` affects the next forward call. Earlier Tensor references and
  already constructed expressions retain the prior exact table. The replacement schema preserves
  exact declaration-time data type and structural Shape plus gradient eligibility; replacement
  remains individual and not thread-safe.
- Add focused tests for the exact public surface, supplied-state validation and side effects,
  one-parameter ownership, direct Model delegation and provenance, scalar and shaped indices,
  mode irrelevance, inherited Model failures, and replacement snapshots.
- Add complete type, constructor, method, and package Javadocs. After executable work and focused
  validation, use a separate documentation-focused clean context to finalize those Javadocs,
  glossary impact, planning evidence, no-change conclusions, and documentation validation in the
  same overall change.

## Out of scope

- An initialized constructor, default parameter Tensor, default data type, default or retained
  random source, layer-owned distribution, caller-selectable initializer object, initializer
  registry, or reuse of Linear-specific Glorot/Kaiming fan policy as an embedding default.
- A padding index, guaranteed-zero padding row, gradient mask, post-update projection, sparse
  gradient, maximum-norm option, frequency scaling, negative-index wrapping, clamping, or default
  row. The current Model convenience represents none of these, and this task must not simulate
  them through eager storage mutation or an NN-owned update hook.
- Vocabulary or embedding-size accessors, parameter-Shape introspection, lazy or dynamic state,
  zero-sized table axes, multiple tables, sharding, quantization, integral/BOOL parameters, frozen
  parameters, mixed parameter types, or a configurable table axis.
- A generic `Module.forward`, layer interface, abstract layer base, unary module contract,
  `Sequential`, block, or stateless functional facade.
- Any `EmbeddingAttrs`, `EMBEDDING` operation kind, semantic signature, Tensor overload, Gather
  change, bounds implementation, eager lookup, host-storage read, result storage, numerical
  output assertion, or input mutation.
- Gradient rules, repeated-index scatter-add construction, optimizer behavior, parameter groups,
  checkpoints, state dictionaries, serialization, compiler capture or optimization, backend
  lowering, runtime/prepare/Engine behavior, execution, or end-to-end support.
- Gradle, dependency, architecture-contract, ADR, explanatory-architecture, architecture-test,
  Model, Training production, CPU, global-roadmap, or unrelated refactoring changes.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md): NN ownership, Tensor invariants,
  dependency direction, optimizer/training lifecycle, and testing requirements.
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md).
- [Module boundaries](../../../../architecture/module-boundaries.md).
- [Dependency rules](../../../../architecture/dependency-rules.md).
- [Training graph](../../../../architecture/training-graph.md).
- [NN master plan](../master-plan.md).
- [Planning guide](../../../planning-guide.md).
- [Completed NN task 0006](0006-layer-normalization-layer.md) and its foundation dependencies.
- [Completed Model task 0019A1: Embedding convenience](../../../modules/model/tasks/0019a1-embedding-convenience.md).
- [Tensor API embedding convenience](../../../../api/tensor-api.md#embedding-convenience).

## Architecture constraints

- `extensions/nn` continues to depend only on `modules/model`; it must not import training,
  compiler, runtime, prepare, engine, or concrete backend code.
- NN owns the stable table parameter and typed forward composition. Model remains the sole owner
  of generic axis-zero Gather meaning, index representation, Shape/result metadata, operation
  attributes, provenance, and construction-time validation. Training remains downstream and owns
  optimizer algorithms and update orchestration.
- Tensor identity, descriptor, and provenance remain immutable. `Embedding` may retain one mutable
  NN parameter binding but must not mutate a Tensor, its storage, or its gradient lifecycle.
- Construction and forward create only module state around a supplied Tensor or storage-free Model
  expression metadata. They must not imply compiler capture, gradient availability, bounds
  enforcement, backend support, numerical evaluation, storage residency, or execution.
- Train/eval mode remains NN composition metadata. This mode-insensitive layer must not add an
  unused context argument or execution-state dependency.
- Preserve unrelated worktree changes exactly, including the dirty CPU master/task and global
  roadmap. This task does not authorize edits outside its exact six paths.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.nn.module` — existing `Module` and `Parameter` ownership,
  discovery, and replacement contracts.
- `io.github.pho001.synaptik.model.tensor` — existing `Tensor.embedding` construction and current
  Tensor descriptor/provenance contracts.
- `io.github.pho001.synaptik.model.shape` and `.datatype` — existing immutable Shape, static
  Dimension, and floating/index data-type contracts.
- `io.github.pho001.synaptik.model.operation.index` — test inspection of the existing ordinary
  `GATHER` and `IndexAxisAttrs(0)` occurrence only; production must not construct them directly.

Packages added or changed:

- `io.github.pho001.synaptik.nn.layers` — add one concrete stateful lookup layer beside `Linear`
  and `LayerNorm`; no package or layer hierarchy is introduced.

Type placement:

- `io.github.pho001.synaptik.nn.layers.Embedding` — public stateful table layer; it belongs with
  concrete layers rather than generic module ownership, stateless initialization policy, or Model
  operation semantics.
- `io.github.pho001.synaptik.nn.layers.EmbeddingTest` — exact public surface, supplied state,
  validation, forward/provenance, mode, inherited failures, and replacement coverage.

## Affected files

Expected production files:

- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/package-info.java`.
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/Embedding.java`.

Expected test file:

- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/EmbeddingTest.java`.

Documentation and planning files:

- `docs/glossary.md` — extend the existing NN-module and Embedding entries with the current
  stateful owner while preserving the existing Model/Gather and execution boundaries.
- `docs/planning/extensions/nn/master-plan.md`.
- this task specification.

Reviewed unchanged unless implementation makes a current statement inaccurate:

- `ARCHITECTURE.md`, ADR 0007, module/dependency/training-graph explanations, and architecture
  tests — the task realizes the existing NN layer responsibility without changing a dependency,
  owner, or lifecycle boundary.
- Tensor and Compile APIs, Model capability/master plans, and Model task 0019A1 — the layer
  delegates to the unchanged current Model convenience and adds no Model or compiler capability.
- Training API — recursive discovery and individual parameter replacement remain unchanged; no
  optimizer, gradient publication, padding-update, or training workflow is added.
- Public API status, user training documentation, conformance/integration suites, Gradle,
  compiler/runtime/prepare/Engine, backends, other modules, CPU planning, and the global roadmap —
  none owns this bounded Model-composition wrapper.

## Maximum scope

This task may create or modify at most:

- two production Java files;
- one NN test file; and
- the three documentation/planning files listed above.

That is an exact six-path maximum. If another type, test owner, API document, module, dependency,
architecture test, Model helper, initializer API, padding/update mechanism, or executable
integration path is required, stop and propose a focused follow-up rather than expanding this
task.

## Acceptance criteria

- Final public `Embedding` extends `Module` and declares exactly the one constructor and two
  methods listed in Scope, with no additional public or protected member.
- Construction retains the exact caller-supplied Tensor and validates null, floating type,
  gradient eligibility, rank, fully static Shape, positive vocabulary extent, and positive
  embedding extent in the specified order before parameter declaration and without Tensor-ID,
  allocation, storage, value, or expression side effects.
- Direct and recursive discovery expose exactly one parameter named `weight`; `weight()` returns
  that exact stable wrapper, and the wrapper's value is the exact supplied table.
- `forward` null-checks indices, reads the current weight once, and delegates exactly to that
  Tensor's `embedding(indices)` method. It does not construct an operation, Shape, descriptor,
  producer, provenance, bound check, or intermediate Tensor itself.
- Tests cover both exact index types, scalar and multi-axis indices, all three floating table
  types, result Shape/type/eligibility/layout metadata, one ordinary `GATHER` with
  `IndexAxisAttrs(0)`, exact `[weight, indices]` inputs, output index zero, and fresh result
  identity without evaluating values.
- Tests cover null indices and inherited Model index-type and identifier-exhaustion failures
  without duplicating the exhaustive Model 0019A1 matrix or changing its messages/order.
- Training and evaluation mode produce the same composition contract; no context overload,
  generic Module forward method, hidden state, or mode-dependent operation appears.
- A compatible replacement appears only in later forward provenance; an earlier expression keeps
  the old exact table, the wrapper remains stable, and incompatible type/Shape/eligibility
  replacement stays rejected by the existing `Parameter` schema.
- No initialized constructor, initializer call, padding behavior, new semantic operation,
  gradient/update rule, optimizer, checkpoint, serialization, compiler, runtime, backend,
  execution, build, dependency, architecture, CPU, roadmap, or unrelated change enters the diff.
- Public and package Javadocs cover table orientation, state schema and name, exact retained
  references, absence of a layer-owned initializer/padding policy, Model delegation, result
  metadata, mode irrelevance, replacement snapshots, nullability, validation and side effects,
  inherited failures, thread safety, and no-execution boundaries with complete tags.
- A separate clean-context documentation pass independently finalizes Javadoc, package docs,
  glossary wording, planning evidence, generated Javadoc, local links/anchors, exact scope,
  reasoned no-change conclusions, and whitespace before the task becomes Complete.

## Tests / validation

Implementation pass runs the focused test while developing:

```bash
./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.layers.EmbeddingTest
```

After executable Java stabilizes, run exactly one final NN suite:

```bash
./gradlew :extensions:nn:test
```

The focused test owns the exact declared API, state schema/name/identity, validation order and
side effects, direct Model producer/provenance delegation, mode irrelevance, inherited failures,
and replacement snapshots. Existing Model 0019A1 tests remain the authoritative exhaustive
embedding semantic/validation matrix; do not repeat them or claim numerical execution.

The separate documentation pass runs after final Javadoc edits:

```bash
./gradlew :extensions:nn:javadoc
git diff --check
```

It also validates changed Markdown links and anchors, balanced fences, final newlines, trailing
whitespace, generated `Embedding` and package pages, exact six-path scope, package placement,
public surface, forbidden imports, 0001–0007 Complete, 0008–0011 Draft, no Ready NN task, and
absence of detailed NN 0008 or later task files. It preserves unrelated CPU/global-roadmap
changes and may reuse final NN test evidence when executable Java has not changed.

Repository-wide and architecture-test validation are deferred to the NN capability checkpoint or
CI. This task changes one existing model-only module, no build edge, dependency rule, architecture
boundary, shared Gradle contract, backend behavior, or end-to-end execution path.

## Dependencies

- NN 0001–0006 are Complete.
- Model task 0019A1 and the exact `Tensor.embedding(Tensor)` convenience are Complete.
- Existing `Module`, `Parameter`, Shape, data-type, producer, and provenance contracts are
  Complete and provide every state and forward fact used here.
- Accepted ADR 0007 and the model-only NN dependency direction remain unchanged.
- The user-authorized NN parallel exception recorded in the NN master plan remains in force; the
  six task paths do not overlap the active dirty CPU planning work.

## Follow-up tasks

- NN 0008: BatchNorm remains Draft only; it owns the first explicit parameter/buffer and
  train/eval state-transition layer.
- NN 0009: Dropout remains Draft only; it owns explicit `GraphRngState` threading and evaluation
  bypass without hidden RNG.
- NN 0010: state dictionary and checkpoint semantics remain Draft only.
- NN 0011: a narrow unary forward contract and `Sequential` remain Draft only and require their
  concrete consumer.
- A future embedding-specific initializer or padding/update policy requires a concrete consumer
  and an independently specified contract. It must not retrofit hidden storage mutation or infer
  padding semantics from ordinary Gather.
- `extensions/training` later consumes the stable discovered `Parameter`; this task adds no
  optimizer or session behavior. Numerical end-to-end validation waits for the required compiled
  execution and backend coverage.

## Architecture impact

Expected impact: None.

This task implements the existing architecture allowance for stateful NN layers composed from
Model semantics. If implementation requires a new dependency, architecture rule, Model change,
initializer policy, padding/update mechanism, generic layer abstraction, or executable behavior,
stop and report the conflict rather than editing architecture within this task.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Work in the Synaptik repository without commit or push. Do not use any GSD skill or workflow.
Read AGENTS.md, ARCHITECTURE.md, the current architecture index, planning and documentation rules,
the NN master plan and completed tasks 0001–0006, Model task 0019A1, current Module/Parameter/
existing-layer contracts, Tensor.embedding and its focused tests, and
docs/planning/extensions/nn/tasks/0007-embedding-layer.md in full.

Implement task 0007 exactly inside its six authorized paths. Preserve all unrelated worktree
changes exactly, especially CPU planning and the global roadmap. Stop and report any architecture
uncertainty, scope overflow, or need for another public type, dependency, initializer, padding
mechanism, or file.

Run the focused test and one final NN suite after executable Java stabilizes. Then hand the actual
diff, exact Java evidence, and task contract to a separate documentation-focused clean context in
the same overall change. That pass finalizes package/type Javadocs, glossary and planning
evidence, generated Javadoc, Markdown, scope, no-change conclusions, and whitespace without
repeating successful Java tests unless executable behavior changes. Mark Complete only after
every criterion passes.
```

## Documentation-agent handoff

Give the separate documentation-focused agent this task, the complete implementation diff, exact
focused/final NN evidence and whether Java changed afterward, the exact public surface, table
validation order, parameter and replacement behavior, forward producer/provenance and train/eval
evidence, padding/initializer exclusions, and the exact six authorized paths.

That agent independently reads the repository instructions, architecture contract and index,
documentation rules plus General/API-Javadoc/Planning profiles, ADR 0007, this task, final source
and tests, generated Javadoc, Tensor/Compile/Training APIs, glossary, NN master/task history, Model
0019A1, `Tensor.embedding`, Gather/index contracts, `Module`, `Parameter`, and current layers. It
finalizes only package/type Javadoc, glossary, this task, and the NN master plan. It records
reasoned no-change conclusions for architecture/ADRs/tests, Tensor/Compile/Training/public APIs,
Model plans/contracts, conformance/integration, Gradle, execution layers, backends, other modules,
CPU work, and the global roadmap.

The documentation pass reuses successful Java evidence unless executable Java changes or it
records a concrete reason to rerun it. It records its clean-context identifier, files/topics
reviewed, commands/results, glossary impact, limitations, and unresolved issues before completion.

## Local decisions

- The public surface is deliberately only supplied-state construction, `weight()`, and
  `forward(indices)`. A caller can construct a table explicitly with Model factories or the
  existing generic NN normal/uniform initializers, so no current consumer justifies baking one
  distribution, default source, or data type into this layer.
- Existing Glorot and Kaiming methods are explicitly defined for Linear weights in
  `[outFeatures, inFeatures]` orientation. Treating vocabulary size and embedding size as those
  fans would silently invent an embedding initialization policy, so this task does not reuse them.
- The table parameter is named `weight`, matching the receiver terminology of
  `Tensor.embedding` and the stable layer parameter convention. Its exact Shape is
  `[vocabularySize, embeddingSize]` and both axes are positive and fully static.
- Forward is mode-insensitive and delegates directly to Model without an unused
  `ForwardContext`. It performs only an indices null check before reading the current table once.
- Padding-row behavior is deferred. The current Model operation is ordinary Gather and has no
  padding attribute. Initial zeroing alone would not preserve a row across public parameter
  replacement or future optimizer updates, while enforcing it would require hidden mutation,
  gradient/update policy, or a new semantic contract outside this task.

## Known limitations

- Only caller-supplied positive fully static rank-two floating state is supported; there is no
  initialized, lazy, dynamic, zero-axis, sharded, quantized, or sparse table form.
- There is no padding index or invariant zero row. Every eventual index must satisfy the ordinary
  Gather bound `0 <= index < vocabularySize`.
- Parameter replacement is individual and not thread-safe. Callers must coordinate replacement
  with forward construction when one stable table snapshot matters.
- Forward constructs expression metadata and proves no compiler capture, gradient rule, repeated-
  index accumulation, bounds enforcement, backend coverage, numerical value, or execution.
- There is no checkpoint, serialization, device/dtype conversion, optimizer, or training-session
  convenience.

## Validation evidence

- Planning context `/root/nn_0007_planning` read the repository instructions, architecture and NN/
  training boundaries, planning/documentation rules, NN master and tasks 0001–0006, Model master
  and task 0019A1, current NN/Model source and tests, Tensor/Compile/Training APIs, glossary,
  dependency enforcement, and Java 26 Gradle configuration before fixing this API and scope.
- Planning inspection confirmed that Model exposes no `EmbeddingAttrs` or embedding operation:
  `Tensor.embedding` validates a rank-two floating receiver and exact INT32/INT64 indices, then
  delegates to one ordinary `GATHER` with `IndexAxisAttrs(0)`. It offers no padding option and
  reads no values. Existing parameter replacement preserves only exact type, structural Shape,
  and gradient eligibility.
- Targeted validation passed for both planning paths: every repository-local Markdown target and
  the Tensor API `#embedding-convenience` anchor resolved; fenced blocks were balanced; both files
  had terminal newlines and no trailing whitespace. Scope inspection found exactly the NN master
  plan and this new task file, one Ready master row paired with this Ready task, four concise Draft
  rows for 0008–0011, and no 0008–0011 task files. Whole-worktree `git diff --check` and the
  untracked-file `git diff --no-index --check` passed; unrelated CPU and global-roadmap changes
  remained outside the NN planning diff.
- Clean implementation context `/root/nn_0007_implementation` read the required architecture,
  planning, documentation-profile, completed NN 0001–0006, Model 0019A1, final Tensor/Gather,
  module/parameter/layer, API/glossary, test, dependency, and Java 26 Gradle contracts before
  editing. It found no architecture, final-Model-API, package-placement, dependency, or six-path
  scope conflict.
- The final focused command after executable Java stabilized,
  `./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.layers.EmbeddingTest`, passed
  with one suite and 8 tests, zero failures, errors, or skips. Coverage includes exact final public
  surface, sole state identity/path, every constructor validation stage and no-ID effects, all
  three floating table types, both exact index types, scalar and multi-axis indices, exact
  ordinary-Gather producer/provenance and result metadata, freshness, mode irrelevance, inherited
  index/null/identifier failures, and replacement snapshots.
- The sole authoritative final `./gradlew :extensions:nn:test` passed after source and tests
  stabilized (`BUILD SUCCESSFUL`, five actionable tasks: one executed and four up-to-date). XML
  reports contain 12 suites and 61 tests with zero failures, errors, or skips. No executable Java
  or test changed afterward in this implementation context.
- Preliminary `./gradlew :extensions:nn:javadoc` passed after the final executable edit (`BUILD
  SUCCESSFUL`, three actionable tasks: one executed and two up-to-date). Generated
  `Embedding.html` and `layers/package-summary.html` exist and render the planned public members,
  orientation, delegation, replacement, padding exclusion, and no-execution boundaries. This is
  implementation-draft evidence only; the separate documentation context still owns final
  Javadoc editing, generation, and generated-page inspection.
- `javap -public` and an independent Java 26 reflection program confirmed public final
  `Embedding extends Module` with exactly `Embedding(Tensor)`, `weight()`, and
  `forward(Tensor)`, and no other declared public or protected member. Manual source inspection
  confirmed null-check-first forward order, one current-binding read, and exact direct
  `return currentWeight.embedding(indices)` delegation.
- Production-import and dependency scans found only Model, existing NN module, and JDK imports;
  `extensions/nn` retains its sole Model project dependency. The implementation adds no
  initializer call, operation/attributes type, padding/update mechanism, generic layer contract,
  execution dependency, or other module edge.
- Preliminary documentation-impact review found no task-owned change necessary for
  `ARCHITECTURE.md`, ADR 0007, focused architecture documents/tests, Tensor/Compile/Training APIs,
  Model capabilities/master/task 0019A1, conformance/integration, Gradle, compiler/runtime/
  prepare/Engine, backends, other modules, CPU planning, or the global roadmap. The layer realizes
  the existing NN owner and delegates to unchanged Model metadata without adding a semantic kind,
  gradient/update, bounds, backend, or execution contract. These preliminary conclusions were
  handed to the mandatory documentation context for independent verification and glossary review.
- Independent clean documentation context `/root/nn_0007_docs` read the repository instructions,
  architecture contract and focused architecture explanations, planning/documentation rules and
  profiles, ADR 0007, NN master/task history, Model master/capabilities/task 0019A1, final
  Tensor/Gather/Shape/DataType and NN module/parameter/layer sources and tests, API references,
  glossary, dependency rules, and Java 26 Gradle configuration. It found no executable Java or
  test defect and changed only task-owned Javadoc and prose; executable source and tests remained
  unchanged, so it reused the stable focused and final NN test evidence above.
- Final `./gradlew :extensions:nn:javadoc` passed (`BUILD SUCCESSFUL`, three actionable tasks: two
  executed and one up-to-date). Inspection of generated `Embedding.html` and
  `layers/package-summary.html` confirmed the exact public members and rendered ownership,
  validation/order, result/delegation, replacement-snapshot, padding/initializer exclusion,
  mode-insensitivity, and declarative no-execution contracts.
- Final `javap -public` and an independent Java 26 reflection check confirmed final public
  `Embedding extends Module`, exactly `Embedding(Tensor)`, `weight()`, and `forward(Tensor)`, and
  no other declared public or protected API. Production import and Gradle inspection confirmed
  only Model, existing NN module, and JDK imports and the unchanged sole Model project dependency.
- Targeted Markdown validation resolved all local paths and anchors and found balanced fences in
  the glossary, NN master plan, and task. Scope/status inspection found exactly the six task paths
  among preserved unrelated dirty CPU/global-roadmap work, NN 0001–0007 Complete, 0008–0011
  Draft, no Ready NN task, and no detailed 0008–0011 task files. Final newline, trailing-
  whitespace, tracked/untracked diff, and whole-worktree `git diff --check` gates passed.
- Final no-change review confirmed no update is warranted for `ARCHITECTURE.md`, the current
  architecture explanations, ADR 0007, architecture tests, Tensor/Compile/Training APIs, Model
  capabilities/master/task 0019A1, conformance/integration tests, Gradle, compiler/runtime/
  prepare/Engine, backends, other modules, training graph, CPU planning, the global roadmap, or
  later NN tasks. The change uses the established NN state owner and unchanged Model convenience;
  it adds no boundary, kind, numerical, gradient/update, compiler, backend, or execution contract.

## Implementation notes

- Added final public `io.github.pho001.synaptik.nn.layers.Embedding` with exactly one supplied-state
  constructor, stable `weight` parameter accessor, and mode-insensitive typed forward method.
  Construction completes the specified null, floating, gradient, rank, static-Shape, vocabulary,
  and embedding-extent validation before declaring the sole parameter.
- Forward rejects null indices before reading the current binding once and returning that exact
  table's existing `embedding(indices)` expression. Focused tests lock the ordinary
  `GATHER`/`IndexAxisAttrs(0)` occurrence, exact `[weight, indices]` inputs, output index zero,
  result metadata, fresh identity, inherited failures, and old/new replacement snapshots without
  reading or asserting numerical values.
- Finalized the required type/method and layers-package Javadocs plus the existing Embedding and
  NN-module glossary entries. The glossary includes a small `[10, 4]` table and `[2, 3]` indices
  metadata example and preserves the Model/Gather versus NN state-ownership boundary. No
  initialized constructor, padding policy, hidden mutation, buffer, RNG, optimizer, checkpoint,
  compiler/backend/runtime behavior, build change, or unrelated refactor was added.

## Completion summary

- Completed changes: Implemented and documented the exact supplied-state Embedding layer,
  constructor schema/order, sole stable parameter, direct Model delegation, mode-insensitivity,
  compatible-replacement snapshots, and current glossary definition/example.
- Files changed or created: `Embedding.java`, `layers/package-info.java`, `EmbeddingTest.java`,
  `docs/glossary.md`, the NN master plan, and this task record: exactly six task paths.
- Tests and validation: Reused the stable final focused one-suite/8-test and authoritative full NN
  12-suite/61-test results, both with zero failures, errors, or skips. Final NN Javadoc,
  generated-page inspection, `javap`, reflection, manual forwarding, import/dependency,
  Markdown, exact-scope/status, final-newline, trailing-whitespace, and diff checks passed.
- Documentation-agent review: Complete in independent context `/root/nn_0007_docs`; it found no
  executable defect and made no executable Java or test change.
- Documentation impact: Finalized affected public/package Javadocs and both relevant glossary
  entries. The reasoned no-change set is recorded in Validation evidence.
- Unresolved issues: None for task 0007.
- Required follow-up: None. NN 0008–0011 remain Draft and have no detailed task specifications.

Status: Complete
