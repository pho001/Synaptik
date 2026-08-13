# Task 0001: Module, Parameter, Buffer, and Forward-Context Foundation

## Status

Complete

## Goal

Create `extensions:nn` as the stateful neural-network composition foundation. It must let a
module directly declare named trainable parameters and persistent buffers over existing immutable
`Tensor` expressions, retain a local training/evaluation mode, and expose an immutable mode
snapshot for a concrete layer's own typed forward method.

## Scope

- Add and include the `:extensions:nn` Gradle project with only an implementation dependency on
  `:modules:model`.
- Add `io.github.pho001.synaptik.nn.module.Module`, `Parameter`, `Buffer`, `ForwardMode`, and
  `ForwardContext` as the narrow public foundation surface.
- Make `Module` the direct owner/declarer of unique local parameter and buffer names and retain
  its local train/eval mode. Provide protected declaration mechanisms and public direct-state and
  mode access only as needed by the task's tests and Javadoc.
- Make `Parameter` and `Buffer` named wrappers around an exact non-null current `Tensor` binding,
  not `Tensor` subtypes. A parameter is trainable by its type; a buffer is persistent and not an
  optimizer target.
- Define the initial binding lifecycle explicitly: declaration captures one exact current Tensor,
  reads return that exact reference, and this task exposes no replacement, update, optimizer, or
  checkpoint API. Later work must introduce any rebinding atomically with its validation and
  concurrency contract.
- Provide immutable training/evaluation `ForwardContext` values derived from the local mode. A
  context is NN composition metadata only: it is not a Tensor, graph node, compile request,
  runtime state, backend resource, or execution handle.
- Add focused NN tests for direct declaration/name validation, exact Tensor-reference retention,
  parameter-versus-buffer classification, local mode/context behavior, and absence of a generic
  `Module.forward(...)` contract.
- Atomically add `implementation(project(":extensions:nn"))` to
  `extensions/training/build.gradle.kts`, because the existing conditional architecture test
  requires the downstream edge as soon as NN is included.
- Strengthen the NN/training architecture test to prove the included NN project is model-only and
  that training, not NN, owns the one-way extension edge.
- Add complete Javadoc for every public production type and member, then hand the complete change
  to an independent documentation-focused context for targeted finalization.

## Out of scope

- Any layer, block, functional convenience, or generic universal `forward` signature.
- Child-module registration, recursive traversal, recursive train/eval propagation, cycles,
  shared children, or hierarchical state names.
- Parameter or buffer replacement, optimizer algorithms, parameter groups, training sessions,
  gradient publication, autograd construction, or `Tensor` gradient state.
- Checkpoints, state dictionaries, serialization, initialization policies, or file formats.
- CPU, Engine, compiler, runtime, prepare, backend, device-storage, kernel-selection, or numerical
  execution behavior.
- Changes to `ARCHITECTURE.md`, explanatory architecture documents, or
  `docs/planning/roadmap.md`.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md): `extensions/nn`, dependency rules,
  optimizer/training lifecycle, and testing requirements.
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md).
- [Dependency rules](../../../../architecture/dependency-rules.md).
- [Training graph](../../../../architecture/training-graph.md).
- [NN master plan](../master-plan.md).
- [Planning Guide](../../../planning-guide.md).

## Architecture constraints

- `extensions/nn` may depend on `modules/model` and must not depend on training, compiler,
  runtime, prepare, engine, or concrete backends.
- `modules/model` remains generic and must not depend on NN; `Tensor` identity, descriptor, and
  expression provenance remain immutable and contain no gradient or trainable lifecycle state.
- NN owns declaration, direct ownership, and train/eval forward behavior. Training later owns
  optimizer algorithms and update orchestration; it consumes NN declarations without making NN
  depend back on training.
- A forward context must not introduce backend residency, physical buffers, run state, or a
  compile/prepare/run lifecycle.
- Do not use legacy source as an implementation input. It is read-only capability evidence only.
- This task is the recorded parallel exception to the global CPU frontier. It must remain confined
  to the files below and must not modify the active CPU work or global roadmap.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model` — existing public `Tensor` contract only.
- `io.github.pho001.synaptik.testing.architecture` — existing dependency-contract test.

Packages added or changed:

- `io.github.pho001.synaptik.nn.module` — the entire initial public NN surface; this cohesive
  package owns direct state declaration and forward-mode contracts.

Type placement:

- `io.github.pho001.synaptik.nn.module.Module` — abstract direct-state owner with no generic
  forward method, preserving each future layer's typed forward signature.
- `io.github.pho001.synaptik.nn.module.Parameter` — named, trainable current-Tensor binding.
- `io.github.pho001.synaptik.nn.module.Buffer` — named, persistent non-trainable current-Tensor
  binding.
- `io.github.pho001.synaptik.nn.module.ForwardMode` — closed training/evaluation vocabulary.
- `io.github.pho001.synaptik.nn.module.ForwardContext` — immutable snapshot of `ForwardMode`.
- `io.github.pho001.synaptik.testing.architecture.NnTrainingDependencyContractTest` — repository
  architecture enforcement; it remains outside the NN module.

## Affected files

Expected build and architecture files:

- `settings.gradle.kts` — include `:extensions:nn` beside the other extensions.
- `extensions/nn/build.gradle.kts` — new module build with `:modules:model` only.
- `extensions/training/build.gradle.kts` — add the required downstream NN dependency.
- `testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/NnTrainingDependencyContractTest.java` — enforce the exact NN model-only and training-to-NN Gradle edges.

Expected NN production files:

- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Module.java`.
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Parameter.java`.
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Buffer.java`.
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/ForwardMode.java`.
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/ForwardContext.java`.

Expected NN test files:

- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/module/ModuleTest.java`.
- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/module/ParameterAndBufferTest.java`.
- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/module/ForwardContextTest.java`.

Planning files synchronized by this planning task:

- `docs/planning/extensions/nn/master-plan.md`.
- this task specification.

## Maximum scope

This task may create or modify at most:

- ten Java source and test files; and
- five build, planning, and architecture-enforcement files listed above.

If direct state cannot be implemented within this limit, or requires a recursive tree,
replacement lifecycle, checkpoint contract, another module dependency, or architecture change,
stop and propose the smallest follow-up task.

## Acceptance criteria

- `:extensions:nn` is included in Gradle, compiles on the repository Java toolchain, and declares
  only `implementation(project(":modules:model"))`.
- `extensions/training` declares `implementation(project(":extensions:nn"))`; NN's build script
  does not mention training or an execution/backend module.
- `Module`, `Parameter`, `Buffer`, `ForwardMode`, and `ForwardContext` exist exactly in the mapped
  public package, and no NN root-package catch-all API is added.
- A `Module` can declare and retain direct named parameters and buffers; null, blank, and duplicate
  direct names fail deterministically without silently replacing state. The chosen duplicate domain
  (one shared direct namespace or distinct parameter/buffer namespaces) is explicitly documented,
  tested, and recorded as a local decision.
- `Parameter` and `Buffer` retain and return the exact declared non-null `Tensor` object. Neither
  is a `Tensor` subtype, and this task exposes no value replacement API.
- A parameter is distinguishable as trainable from a buffer without adding a mutable gradient,
  optimizer, or backend concern to `Tensor`.
- A module exposes only local training/evaluation state and an immutable corresponding forward
  context. No generic `forward` method is declared on `Module`; concrete future layers choose
  their own typed forward signatures.
- The focused NN tests verify every foundation invariant above, and the architecture test rejects
  a missing model dependency, an NN-to-training edge, or a missing training-to-NN edge.
- Every public NN API has meaningful Javadoc with complete parameter, result, and failure
  documentation. The separate documentation-focused pass independently reviews the final API,
  explanatory-document need, glossary impact, links, examples, and generated Javadoc in this same
  overall change.
- No CPU/Engine/compiler/runtime/prepare/backend source, global roadmap, architecture contract,
  layer, optimizer, checkpoint, or numerical execution behavior changes.

## Tests / validation

Implementation pass runs:

```bash
./gradlew :extensions:nn:test
./gradlew :testing:architecture-tests:test
./gradlew test
```

The repository-wide command is required because this task changes included projects and an
inter-module Gradle dependency. Record the focused results first and do not repeat successful
Java suites in the documentation pass unless executable Java changes afterward.

Documentation pass runs after final Javadoc edits:

```bash
./gradlew :extensions:nn:javadoc
git diff --check
```

The documentation pass also verifies all Markdown links and anchors changed by the planning
artifacts, confirms the type placement against the package map, checks generated Javadoc pages,
and records a glossary update or a specific no-change conclusion. Repository-wide validation is
not deferred because the Gradle project graph changes; CI remains the final independent gate.

## Dependencies

- Completed `modules/model` public Tensor/provenance contracts.
- Accepted ADR 0007 and its existing conditional NN/training architecture test.
- Explicit user authorization and the recorded NN-master-plan parallel exception while CPU 0006D
  is active.

## Follow-up tasks

- NN 0002 (required, Draft until 0001 is Complete): child-module ownership, deterministic
  recursive traversal, hierarchical names, and recursive mode propagation.
- NN 0003 (required, Draft until 0002 is Complete): validated parameter/buffer binding
  replacement and its concurrency/checkpoint interaction.
- NN 0004 (required, Draft until 0001's public API is stable): first `Linear` layer over existing
  `Tensor.linear` semantics; numerical end-to-end execution is separately deferred.
- Checkpoint/state-dictionary, normalization, dropout, and training tasks remain future and must
  not be detailed before their preceding dependencies stabilize.

## Architecture impact

Expected impact: None.

This task implements existing NN ownership and dependency rules. If implementation needs to alter
those rules, add a Tensor gradient/trainable lifecycle, or introduce an execution-layer dependency,
stop and report the conflict instead of editing architecture documents.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, docs/developer-guide/documentation-rules.md, and docs/planning/extensions/nn/tasks/0001-module-parameter-buffer-and-forward-context-foundation.md in full. Implement that task exactly as specified. Do not implement out-of-scope work, modify active CPU files, commit, or push. Stop and report any architecture or scope conflict.

After executable implementation and validation, hand the complete diff and exact test evidence to a separate documentation-focused clean context. That pass must follow the documentation rules, independently finalize affected Javadoc, explanatory-document and glossary impact, and documentation validation in the same overall change. It must not rerun successful Java suites unless executable behavior changed or it records a concrete risk.

Update the task with implementation notes, local decisions, validation evidence including the documentation-agent pass, and completion summary. Do not mark it Complete before that pass finishes.
```

## Local decisions

- Parameters and buffers share one exact local name namespace. A null name fails with
  `NullPointerException`; a blank name or a name already used by either direct-state type fails
  with `IllegalArgumentException`. Names are retained exactly rather than trimmed or normalized.
- `Module` exposes protected `parameter(name, value)` and `buffer(name, value)` declaration
  methods; its public direct-state accessors are declaration-ordered immutable snapshots from
  `parameters()` and `buffers()`. `Parameter` and `Buffer` expose only `name()` and `value()`.
- `Module` starts in local `TRAINING` mode. `train()` and `eval()` change only that local mode;
  `forwardContext()` creates an immutable `ForwardContext` snapshot with no child propagation.

## Known limitations

- Direct state only: no child modules, traversal, hierarchical names, or mode propagation.
- Current Tensor bindings are declaration-time references only; no rebinding, update, checkpoint,
  serialization, or concurrency semantics are exposed.
- No layers or numerical execution exist, so this task validates ownership and expression
  composition contracts rather than evaluated results.

## Validation evidence

- `./gradlew :extensions:nn:test :testing:architecture-tests:test` — passed (2026-08-13): NN
  foundation tests and the strengthened dependency contract compiled and passed.
- `./gradlew test` — passed (2026-08-13): repository-wide included-project and dependency-graph
  validation completed successfully; 60 actionable tasks, 2 executed, 58 up-to-date.
- `./gradlew :testing:architecture-tests:test` — passed (2026-08-13) after strengthening the
  exact sole-NN-project-dependency assertion.
- `git diff --check` — passed (2026-08-13) on the shared worktree after implementation changes.
- Documentation-focused context `/root/nn_0001_docs` independently finalized the affected API
  Javadoc and planning evidence (2026-08-13). It reused the implementation's successful Java
  evidence because its edits did not alter executable behavior.
- `./gradlew :extensions:nn:javadoc` — passed (2026-08-13) after the final Javadoc edits.
- Markdown local-link, heading-anchor, fenced-code-block, package-map, and terminology review —
  passed (2026-08-13). The NN glossary entry was updated from planned tree behavior to the current
  direct-state boundary.
- Architecture contract/explanatory pages, ADR 0007, training graph, and public Tensor API —
  reviewed with no change (2026-08-13): they already state the correct ownership and
  dependency direction; the new API neither changes Tensor semantics nor implements training,
  execution, or a module tree.
- `git diff --check` — passed (2026-08-13) after documentation finalization.

## Implementation notes

- Added the model-only `:extensions:nn` Gradle project and atomically added the downstream
  `extensions:training -> extensions:nn` implementation dependency.
- Added `Module`, `Parameter`, `Buffer`, `ForwardMode`, and `ForwardContext` under the planned
  `io.github.pho001.synaptik.nn.module` public package with focused direct-state, identity,
  lifecycle-boundary, and no-generic-forward tests.
- Strengthened the architecture dependency test to require the included NN project, its exact
  sole `modules:model` project edge, all listed forbidden NN edges, and the downstream training
  edge.

## Completion summary

Created the model-only `extensions:nn` module and its direct-state `Module`, `Parameter`,
`Buffer`, `ForwardMode`, and immutable `ForwardContext` API. The module preserves exact
declaration-time Tensor references, one shared local state-name namespace, and local train/eval
mode without a generic forward contract or execution behavior. The downstream training dependency
and architecture enforcement are present.

The independent documentation-focused pass finalized the API Javadoc, updated the glossary's
current-versus-planned NN boundary, verified the required planning/documentation evidence, and
recorded reasoned no-change conclusions for Tensor and architecture/training documents. Focused
and repository-wide Java validation from implementation remains applicable; final NN Javadoc,
documentation checks, and whitespace validation passed.

Status: Complete
