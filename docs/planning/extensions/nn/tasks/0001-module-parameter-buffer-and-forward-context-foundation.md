# Task 0001: [Module](../../../../glossary.md#term-module), [Parameter](../../../../glossary.md#term-parameter), [Buffer](../../../../glossary.md#term-buffer), and Forward-Context Foundation

## Status

Complete

## Goal

Create `extensions:nn` as the stateful neural-network composition foundation. It must let a
[module](../../../../glossary.md#term-module) directly declare named [trainable parameters](../../../../glossary.md#trainable-parameter) and persistent [buffers](../../../../glossary.md#term-buffer) over existing [immutable](../../../../glossary.md#term-immutable)
[`Tensor`](../../../../glossary.md#tensor) expressions, retain a local training/[evaluation](../../../../glossary.md#term-evaluation) mode, and expose an immutable mode
[snapshot](../../../../glossary.md#term-snapshot) for a concrete layer's own typed forward method.

## [Scope](../../../../glossary.md#term-scope)

- Add and include the `:extensions:nn` [Gradle](../../../../glossary.md#term-gradle) project with only an implementation [dependency](../../../../glossary.md#term-dependency) on
  `:modules:model`.
- Add `io.github.pho001.synaptik.nn.module.Module`, [`Parameter`](../../../../glossary.md#term-parameter), [`Buffer`](../../../../glossary.md#term-buffer), `ForwardMode`, and
  `ForwardContext` as the narrow public foundation surface.
- Make [`Module`](../../../../glossary.md#term-module) the direct owner/declarer of unique local parameter and buffer names and retain
  its local train/eval mode. Provide protected declaration mechanisms and public direct-state and
  mode access only as needed by the task's tests and Javadoc.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [eval mode glossary](../../../../glossary.md#inference-and-evaluation); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Make `Parameter` and `Buffer` named wrappers around an exact non-null current [`Tensor`](../../../../glossary.md#tensor) binding,
  not <span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>`Tensor` subtypes. A parameter is trainable by its type; a buffer is persistent and not an
  optimizer target.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [target glossary](../../../../glossary.md#term-target))</span>
- Define the initial binding lifecycle explicitly: declaration captures one exact current Tensor,
  reads return that exact reference, and this task exposes no replacement, update, optimizer, or
  checkpoint API. Later work must introduce any rebinding atomically with its validation and
  concurrency contract.<span data-glossary-navigation="true"> ([captures glossary](../../../../glossary.md#term-capture-meanings))</span><span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [validation glossary](../../../../glossary.md#term-validation); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- Provide [immutable](../../../../glossary.md#term-immutable) training/[evaluation](../../../../glossary.md#term-evaluation) `ForwardContext` values derived from the local mode. A
  context is NN composition metadata only: it is not a Tensor, graph node, compile request,
  runtime state, backend resource, or execution handle.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [metadata glossary](../../../../glossary.md#term-metadata); [graph glossary](../../../../glossary.md#graph); [node glossary](../../../../glossary.md#node); [compile glossary](../../../../glossary.md#compile); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>
- Add focused NN tests for direct declaration/name validation, exact Tensor-reference retention,
  parameter-versus-buffer classification, local mode/context behavior, and absence of a generic
  `Module.forward(...)` contract.
- Atomically add `implementation(project(":extensions:nn"))` to
  `extensions/training/build.gradle.kts`, because the existing conditional architecture test
  requires the downstream edge as soon as NN is included.<span data-glossary-navigation="true"> ([architecture test glossary](../../../../glossary.md#term-architecture-test))</span>
- Strengthen the NN/training architecture test to prove the included NN project is model-only and
  that training, not NN, owns the one-way extension edge.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- Add complete Javadoc for every public production type and member, then hand the complete change
  to an independent documentation-focused context for targeted finalization.

## Out of [scope](../../../../glossary.md#term-scope)

- Any layer, block, functional convenience, or generic universal `forward` [signature](../../../../glossary.md#term-signature).
- Child-module registration, recursive traversal, recursive train/eval propagation, cycles,
  shared children, or hierarchical state names.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
- Parameter or buffer replacement, optimizer algorithms, parameter groups, training sessions,
  gradient publication, autograd construction, or <span data-glossary-navigation="true"> ([Parameter glossary](../../../../glossary.md#term-parameter); [buffer glossary](../../../../glossary.md#term-buffer); [optimizer glossary](../../../../glossary.md#term-optimizer); [publication glossary](../../../../glossary.md#term-publication); [autograd glossary](../../../../glossary.md#autograd))</span>[`Tensor`](../../../../glossary.md#tensor) [gradient](../../../../glossary.md#term-gradient) state.
- [Checkpoints](../../../../glossary.md#term-checkpoint), state dictionaries, [serialization](../../../../glossary.md#term-serialization), initialization policies, or file formats.
- CPU, Engine, compiler, runtime, prepare, backend, device-storage, kernel-selection, or numerical
  execution behavior.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [device glossary](../../../../glossary.md#term-device); [kernel glossary](../../../../glossary.md#kernel))</span>
- Changes to `ARCHITECTURE.md`, explanatory architecture documents, or
  `docs/planning/roadmap.md`.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>: `extensions/nn`, dependency rules,
  optimizer/training lifecycle, and testing requirements.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [optimizer glossary](../../../../glossary.md#term-optimizer); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module))</span>.
- [Dependency rules](../../../../architecture/dependency-rules.md).
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>.
- [NN master plan](../master-plan.md)<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>.
- [Planning Guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>.

## Architecture constraints

- `extensions/nn` may depend on `modules/model` and must not depend on training, compiler,
  runtime, prepare, engine, or concrete backends.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [backends glossary](../../../../glossary.md#backend))</span>
- `modules/model` remains generic and must not depend on [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [`Tensor`](../../../../glossary.md#tensor) identity, descriptor, and
  expression provenance remain immutable and contain no gradient or trainable lifecycle state.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [immutable glossary](../../../../glossary.md#term-immutable); [gradient glossary](../../../../glossary.md#term-gradient); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- NN owns declaration, direct ownership, and train/eval forward behavior. Training later owns
  optimizer algorithms and update orchestration; it consumes NN declarations without making NN
  depend back on training.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- A forward context must not introduce backend residency, physical buffers, run state, or a
  compile/prepare/run lifecycle.<span data-glossary-navigation="true"> ([residency glossary](../../../../glossary.md#residency); [buffers glossary](../../../../glossary.md#term-buffer); [run state glossary](../../../../glossary.md#run-state--runstate); [compile glossary](../../../../glossary.md#compile); [run glossary](../../../../glossary.md#term-run))</span>
- Do not use legacy source as an implementation input. It is read-only capability evidence only.
- This task is the recorded parallel exception to the global CPU frontier. It must remain confined
  to the files below and must not modify the active CPU work or global roadmap.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model` — existing public [`Tensor`](../../../../glossary.md#tensor) contract only.
- `io.github.pho001.synaptik.testing.architecture` — existing [dependency](../../../../glossary.md#term-dependency)-contract test.

Packages added or changed:

- `io.github.pho001.synaptik.nn.module` — the entire initial public NN surface; this cohesive
  package owns direct state declaration and forward-mode contracts.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [forward-mode glossary](../../../../glossary.md#primal-adjoint-and-differentiation-seed))</span>

Type placement:

- `io.github.pho001.synaptik.nn.module.Module` — abstract direct-state owner with no generic
  forward method, preserving each future layer's typed forward signature.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [signature glossary](../../../../glossary.md#term-signature))</span>
- `io.github.pho001.synaptik.nn.module.Parameter` — named, trainable current-Tensor [binding](../../../../glossary.md#term-binding).
- `io.github.pho001.synaptik.nn.module.Buffer` — named, persistent non-trainable current-Tensor
  binding.
- `io.github.pho001.synaptik.nn.module.ForwardMode` — closed training/[evaluation](../../../../glossary.md#term-evaluation) vocabulary.
- `io.github.pho001.synaptik.nn.module.ForwardContext` — [immutable](../../../../glossary.md#term-immutable) [snapshot](../../../../glossary.md#term-snapshot) of `ForwardMode`.
- `io.github.pho001.synaptik.testing.architecture.NnTrainingDependencyContractTest` — repository
  architecture enforcement; it remains outside the NN module.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>

## Affected files

Expected build and architecture files:

- `settings.gradle.kts` — include `:extensions:nn` beside the other extensions.
- `extensions/nn/build.gradle.kts` — new [module](../../../../glossary.md#term-module) build with `:modules:model` only.
- `extensions/training/build.gradle.kts` — add the required downstream [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [dependency](../../../../glossary.md#term-dependency).
- `testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/NnTrainingDependencyContractTest.java` — enforce the exact [NN model](../../../../glossary.md#model-module-buffer-and-scan-distinctions)-only and training-to-NN [Gradle](../../../../glossary.md#term-gradle) edges.

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

[Planning](../../../../glossary.md#term-planning) files synchronized by this planning task:

- `docs/planning/extensions/nn/master-plan.md`.
- this task specification.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most:

- ten Java source and test files; and
- five build, [planning](../../../../glossary.md#term-planning), and architecture-enforcement files listed above.

If direct state cannot be implemented within this limit, or requires a recursive tree,
replacement [lifecycle](../../../../glossary.md#lifecycle), [checkpoint](../../../../glossary.md#term-checkpoint) contract, another [module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency), or architecture change,
stop and propose the smallest follow-up task.

## Acceptance criteria

- `:extensions:nn` is included in Gradle, compiles on the repository Java toolchain, and declares
  only <span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [toolchain glossary](../../../../glossary.md#term-toolchain))</span>`implementation(project(":modules:model"))`.
- `extensions/training` declares `implementation(project(":extensions:nn"))`; NN's build script
  does not mention training or an execution/backend module.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [backend glossary](../../../../glossary.md#backend))</span>
- [`Module`](../../../../glossary.md#term-module), [`Parameter`](../../../../glossary.md#term-parameter), [`Buffer`](../../../../glossary.md#term-buffer), `ForwardMode`, and `ForwardContext` exist exactly in the mapped
  public package, and no NN root-package catch-all API is added.
- A `Module` can declare and retain direct named parameters and buffers; null, blank, and duplicate
  direct names fail deterministically without silently replacing state. The chosen duplicate domain
  (one shared direct namespace or distinct parameter/buffer namespaces) is explicitly documented,
  tested, and recorded as a local decision.
- `Parameter` and `Buffer` retain and return the exact declared non-null [`Tensor`](../../../../glossary.md#tensor) object. Neither
  is a `Tensor` subtype, and this task exposes no value replacement [API](../../../../glossary.md#term-application-programming-interface-api).
- A parameter is distinguishable as trainable from a buffer without adding a mutable gradient,
  optimizer, or backend concern to <span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [gradient glossary](../../../../glossary.md#term-gradient); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>`Tensor`.
- A module exposes only local training/evaluation state and an immutable corresponding forward
  context. No generic <span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation); [immutable glossary](../../../../glossary.md#term-immutable))</span>`forward` method is declared on `Module`; concrete future layers choose
  their own typed forward signatures.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>
- The focused NN tests verify every foundation invariant above, and the architecture test rejects
  a missing model dependency, an NN-to-training edge, or a missing training-to-NN edge.<span data-glossary-navigation="true"> ([invariant glossary](../../../../glossary.md#term-invariant); [architecture test glossary](../../../../glossary.md#term-architecture-test); [model glossary](../../../../glossary.md#term-model); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Every public NN API has meaningful Javadoc with complete parameter, result, and failure
  documentation. The separate documentation-focused pass independently reviews the final API,
  explanatory-document need, glossary impact, links, examples, and generated Javadoc in this same
  overall change.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- No CPU/Engine/compiler/runtime/prepare/backend source, global roadmap, architecture contract,
  layer, optimizer, checkpoint, or numerical execution behavior changes.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [architecture contract glossary](../../../../glossary.md#architecture-contract); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation pass runs:

```bash
./gradlew :extensions:nn:test
./gradlew :testing:architecture-tests:test
./gradlew test
```

The repository-wide command is required because this task changes included projects and an
inter-[module](../../../../glossary.md#term-module) [Gradle](../../../../glossary.md#term-gradle) [dependency](../../../../glossary.md#term-dependency). Record the focused results first and do not repeat successful
Java suites in the documentation pass unless executable Java changes afterward.

Documentation pass runs after final [Javadoc](../../../../glossary.md#term-javadoc) edits:

```bash
./gradlew :extensions:nn:javadoc
git diff --check
```

The documentation pass also verifies all Markdown links and anchors changed by the [planning](../../../../glossary.md#term-planning)
[artifacts](../../../../glossary.md#term-artifact), confirms the type placement against the package map, checks generated Javadoc pages,
and records a glossary update or a specific no-change conclusion. Repository-wide validation is
not deferred because the Gradle project [graph](../../../../glossary.md#graph) changes; [CI](../../../../glossary.md#term-continuous-integration-ci) remains the final independent gate.

## Dependencies

- Completed `modules/model` public [Tensor](../../../../glossary.md#tensor)/[provenance](../../../../glossary.md#provenance) contracts.
- Accepted [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0007 and its existing conditional [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/training [architecture test](../../../../glossary.md#term-architecture-test).
- Explicit user authorization and the recorded NN-master-plan parallel exception while CPU 0006D
  is active.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>

## Follow-up tasks

- NN 0002 (required, Draft until 0001 is Complete): child-module ownership, deterministic
  recursive traversal, hierarchical names, and recursive mode propagation.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- NN 0003 (required, Draft until 0002 is Complete): validated parameter/buffer binding
  replacement and its concurrency/checkpoint interaction.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [buffer glossary](../../../../glossary.md#term-buffer); [binding glossary](../../../../glossary.md#term-binding); [concurrency glossary](../../../../glossary.md#term-concurrency); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0004 (required, Draft until 0001's public [API](../../../../glossary.md#term-application-programming-interface-api) is stable): first [`Linear`](../../../../glossary.md#term-linear-meanings) layer over existing
  `Tensor.linear` semantics; numerical end-to-end execution is separately deferred.
- Checkpoint/state-dictionary, normalization, dropout, and training tasks remain future and must
  not be detailed before their preceding dependencies stabilize.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>

## Architecture impact

Expected impact: None.

This task implements existing [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [ownership](../../../../glossary.md#term-ownership-meanings) and [dependency](../../../../glossary.md#term-dependency) rules. If implementation needs to alter
those rules, add a [Tensor](../../../../glossary.md#tensor) [gradient](../../../../glossary.md#term-gradient)/trainable [lifecycle](../../../../glossary.md#lifecycle), or introduce an execution-layer dependency,
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

- [Parameters](../../../../glossary.md#term-parameter) and [buffers](../../../../glossary.md#term-buffer) share one exact local name namespace. A null name fails with
  `NullPointerException`; a blank name or a name already used by either direct-state type fails
  with `IllegalArgumentException`. Names are retained exactly rather than trimmed or normalized.
- [`Module`](../../../../glossary.md#term-module) exposes protected `parameter(name, value)` and `buffer(name, value)` declaration
  methods; its public direct-state accessors are declaration-ordered immutable snapshots from
  `parameters()` and `buffers()`. `Parameter` and `Buffer` expose only `name()` and `value()`.
- `Module` starts in local `TRAINING` mode. `train()` and `eval()` change only that local mode;
  `forwardContext()` creates an [immutable](../../../../glossary.md#term-immutable) `ForwardContext` [snapshot](../../../../glossary.md#term-snapshot) with no child propagation.

## Known limitations

- Direct state only: no child [modules](../../../../glossary.md#term-module), traversal, hierarchical names, or mode propagation.
- Current Tensor bindings are declaration-time references only; no rebinding, update, checkpoint,
  serialization, or concurrency semantics are exposed.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [bindings glossary](../../../../glossary.md#term-binding); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [serialization glossary](../../../../glossary.md#term-serialization); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- No layers or numerical execution exist, so this task validates ownership and expression
  composition contracts rather than evaluated results.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- `./gradlew :extensions:nn:test :testing:architecture-tests:test` — passed (2026-08-13): NN
  foundation tests and the strengthened dependency contract compiled and passed.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- `./gradlew test` — passed (2026-08-13): repository-wide included-project and dependency-graph
  validation completed successfully; 60 actionable tasks, 2 executed, 58 up-to-date.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
- `./gradlew :testing:architecture-tests:test` — passed (2026-08-13) after strengthening the
  exact sole-NN-project-dependency assertion.
- `git diff --check` — passed (2026-08-13) on the shared worktree after implementation changes.
- Documentation-focused context `/root/nn_0001_docs` independently finalized the affected API
  Javadoc and planning evidence (2026-08-13). It reused the implementation's successful Java
  evidence because its edits did not alter executable behavior.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>
- `./gradlew :extensions:nn:javadoc` — passed (2026-08-13) after the final [Javadoc](../../../../glossary.md#term-javadoc) edits.
- Markdown local-link, heading-anchor, fenced-code-block, package-map, and terminology review —
  passed (2026-08-13). The NN glossary entry was updated from planned tree behavior to the current
  direct-state boundary.
- Architecture contract/explanatory pages, ADR 0007, training graph, and public Tensor API —
  reviewed with no change (2026-08-13): they already state the correct ownership and
  dependency direction; the new API neither changes Tensor semantics nor implements training,
  execution, or a module tree.<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [training graph glossary](../../../../glossary.md#training-graph); [Tensor glossary](../../../../glossary.md#tensor); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [module glossary](../../../../glossary.md#term-module))</span>
- `git diff --check` — passed (2026-08-13) after documentation finalization.

## Implementation notes

- Added the [model](../../../../glossary.md#term-model)-only `:extensions:nn` [Gradle](../../../../glossary.md#term-gradle) project and atomically added the downstream
  `extensions:training -> extensions:nn` implementation [dependency](../../../../glossary.md#term-dependency).
- Added [`Module`](../../../../glossary.md#term-module), [`Parameter`](../../../../glossary.md#term-parameter), [`Buffer`](../../../../glossary.md#term-buffer), `ForwardMode`, and `ForwardContext` under the planned
  `io.github.pho001.synaptik.nn.module` public package with focused direct-state, identity,
  lifecycle-boundary, and no-generic-forward tests.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Strengthened the architecture dependency test to require the included NN project, its exact
  sole <span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>`modules:model` project edge, all listed forbidden NN edges, and the downstream training
  edge.

## Completion summary

Created the [model](../../../../glossary.md#term-model)-only `extensions:nn` [module](../../../../glossary.md#term-module) and its direct-state `Module`, [`Parameter`](../../../../glossary.md#term-parameter),
[`Buffer`](../../../../glossary.md#term-buffer), `ForwardMode`, and [immutable](../../../../glossary.md#term-immutable) `ForwardContext` [API](../../../../glossary.md#term-application-programming-interface-api). The module preserves exact
declaration-time [Tensor](../../../../glossary.md#tensor) references, one shared local state-name namespace, and local train/[eval
mode](../../../../glossary.md#inference-and-evaluation) without a generic forward contract or execution behavior. The downstream training [dependency](../../../../glossary.md#term-dependency)
and architecture enforcement are present.

The independent documentation-focused pass finalized the API [Javadoc](../../../../glossary.md#term-javadoc), updated the glossary's
current-versus-planned [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) boundary, verified the required [planning](../../../../glossary.md#term-planning)/documentation evidence, and
recorded reasoned no-change conclusions for Tensor and architecture/training documents. Focused
and repository-wide Java [validation](../../../../glossary.md#term-validation) from implementation remains applicable; final NN Javadoc,
documentation checks, and whitespace validation passed.

Status: Complete
