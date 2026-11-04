# Task 0003: Validated [Parameter](../../../../glossary.md#term-parameter) and [Buffer](../../../../glossary.md#term-buffer) [Binding](../../../../glossary.md#term-binding) Replacement

## Status

Complete

## Goal

Allow a concrete [module](../../../../glossary.md#term-module) to replace the current [Tensor](../../../../glossary.md#tensor) of one of its own direct [parameters](../../../../glossary.md#term-parameter) or
[buffers](../../../../glossary.md#term-buffer) without mutating Tensor, exposing a public setter, changing module-tree structure, or
introducing [optimizer](../../../../glossary.md#term-optimizer)/[checkpoint](../../../../glossary.md#term-checkpoint) behavior.

## [Scope](../../../../glossary.md#term-scope)

- Change [`Parameter`](../../../../glossary.md#term-parameter) and [`Buffer`](../../../../glossary.md#term-buffer) so each preserves its declaration-time local name and wrapper
  identity while retaining one replaceable current non-null <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>[`Tensor`](../../../../glossary.md#tensor) reference.
- Add exactly these protected final, direct-state methods to [`Module`](../../../../glossary.md#term-module):

  ```java
  protected final void replaceParameter(String name, Tensor value)
  protected final void replaceBuffer(String name, Tensor value)
  ```

  Each method [targets](../../../../glossary.md#term-target) only a direct declaration of its matching kind on the receiving module;
  `name` is a local name, never a dot path. A subclass changes a retained binding through its
  declaring module, for example <span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>`replaceParameter("weight", nextWeight)`. It must not mutate a
  `Parameter` or `Buffer` directly.
- Make the replacement sequence deterministic: reject a null `name`, then a null `value`, then
  reject a name that does not identify a direct declaration of the requested kind. Only after all
  validation succeeds may the binding's current reference change. A name occupied by a direct
  buffer is not a parameter, and vice versa.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Preserve the existing declaration order, shared direct-name namespace, child ownership, mode,
  and recursive dot paths. Successful replacement changes neither maps/lists nor their ordering.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Define value observation precisely. `Parameter.value()` and `Buffer.value()` return the exact
  Tensor currently bound at that call. Previously returned Tensor references and Tensor
  expressions already constructed from them stay unchanged; Tensor's immutable identity and
  provenance are never modified. Direct and recursive discovery snapshots are structural
  snapshots containing the same <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [provenance glossary](../../../../glossary.md#provenance))</span>`Parameter`/`Buffer` wrapper objects, so wrappers obtained from a
  snapshot expose their subsequently current binding.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Add no descriptor, dtype, shape, layout, provenance, gradient-eligibility, or storage
  compatibility check beyond non-nullity. <span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [dtype glossary](../../../../glossary.md#data-type--datatype); [shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient))</span>`Module` has no declared binding schema, and model
  supports unresolved/dynamic Tensor descriptors; freezing a descriptor from the initial value
  would invent a layer policy and reject valid future module contracts. A later concrete layer or
  checkpoint task owns any shape/schema validation it needs.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [model glossary](../../../../glossary.md#term-model); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Specify that this API has no version counter or historical-value/snapshot carrier, no batch or
  tree-wide replacement transaction, and no concurrency guarantee. It is not thread-safe:
  callers must externally synchronize concurrent replacement, declaration, traversal, mode, and
  forward construction when a consistent multi-binding view matters. One successful call changes
  only one binding; a failed call changes none.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [carrier glossary](../../../../glossary.md#term-storage-carrier); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [transaction glossary](../../../../glossary.md#term-transaction); [concurrency glossary](../../../../glossary.md#term-concurrency); [thread-safe glossary](../../../../glossary.md#term-thread-safe); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- Update all affected public and contract-relevant Javadocs and focused tests. Use an independent
  documentation-focused agent/thread after implementation to finalize Javadoc, glossary impact,
  and this planning record.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Public, protected, or package-external setters/[rebind](../../../../glossary.md#capture-graph-ports-and-reconstruction) methods on [`Parameter`](../../../../glossary.md#term-parameter) or [`Buffer`](../../../../glossary.md#term-buffer).
- Replacement by recursive path, parent-mediated child replacement, replacement by an arbitrary
  wrapper object, detach/reparent/rename, or module-tree mutation.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
- A batch/state-dictionary replacement API, transaction/rollback across multiple bindings,
  versioning, locking, visibility guarantees, or concurrent update protocol.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [transaction glossary](../../../../glossary.md#term-transaction); [rollback glossary](../../../../glossary.md#term-rollback); [bindings glossary](../../../../glossary.md#term-binding))</span>
- Checkpoints, serialization, optimizer algorithms, parameter groups, training sessions,
  gradients, compiled update graphs, or any <span data-glossary-navigation="true"> ([Checkpoints glossary](../../../../glossary.md#term-checkpoint); [serialization glossary](../../../../glossary.md#term-serialization); [optimizer glossary](../../../../glossary.md#term-optimizer); [gradients glossary](../../../../glossary.md#term-gradient); [graphs glossary](../../../../glossary.md#graph))</span>`extensions/training` [API](../../../../glossary.md#term-application-programming-interface-api).
- Layers, initializers, [RNG](../../../../glossary.md#graph-rng-state--graphrngstate), generic `forward` signatures, Tensor-operation semantics, numerical
  evaluation, compiler, runtime, prepare, Engine, or backend work.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [Tensor glossary](../../../../glossary.md#tensor); [operation glossary](../../../../glossary.md#operation); [evaluation glossary](../../../../glossary.md#term-evaluation); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend))</span>
- Gradle/dependency changes, architecture-contract/ADR/explanatory-architecture updates,
  architecture-test edits, global-roadmap edits, or unrelated refactoring.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>: immutable Tensor identity/provenance,
  NN ownership, extension direction, and testing discipline.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [Tensor glossary](../../../../glossary.md#tensor); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module))</span>.
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>.
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>.
- [NN master plan](../master-plan.md).
- [Task 0001](0001-module-parameter-buffer-and-forward-context-foundation.md) and
  [Task 0002](0002-module-tree-ownership-and-recursive-mode-propagation.md).
- [Planning Guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>.
- [Documentation rules](../../../../developer-guide/documentation-rules.md) and its
  [Planning profile](../../../../developer-guide/documentation/planning-style.md).

## Architecture constraints

- `extensions/nn` continues to depend only on `modules/model`. It must not import training,
  compiler, runtime, prepare, engine, or a concrete backend.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
- [`Module`](../../../../glossary.md#term-module) owns declaration and controlled replacement of direct module state. [`Parameter`](../../../../glossary.md#term-parameter) and
  [`Buffer`](../../../../glossary.md#term-buffer) remain named state carriers, not Tensor subtypes and not optimizer APIs. Training stays
  downstream and may later consume the stable wrapper identities.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- Replacement selects a direct wrapper through the receiving module's existing local registry.
  It never replaces a <span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry))</span>`Parameter`/`Buffer` object, a map entry, or any [Tensor](../../../../glossary.md#tensor) internals.
- Tensor remains public model state with immutable identity, descriptor, and expression
  provenance. Rebinding only determines which distinct Tensor a later <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [immutable glossary](../../../../glossary.md#term-immutable); [identity glossary](../../../../glossary.md#term-identity-meanings); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>`value()` call returns.
- The existing [mutable](../../../../glossary.md#term-mutable) `Module` API has no thread-safety guarantee. This task must document that
  replacement adds no locking, atomic tree state, cross-binding consistency, or Java-memory-model
  visibility promise for unsynchronized callers.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [binding glossary](../../../../glossary.md#term-binding))</span>
- The authorized NN parallel exception is implementation-order only. Do not touch dirty CPU files,
  the global roadmap, build graph, architecture documents, or dependency enforcement.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [CPU glossary](../../../../glossary.md#term-cpu); [graph glossary](../../../../glossary.md#graph); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.nn.module` — owns the complete public module-state and controlled
  binding-replacement contract.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [binding glossary](../../../../glossary.md#term-binding))</span>
- `io.github.pho001.synaptik.model.tensor` — supplies the existing [`Tensor`](../../../../glossary.md#tensor) references only; no
  model source changes are allowed.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>

Packages added or changed:

- No package is added. Only the existing `io.github.pho001.synaptik.nn.module` contracts change.

Type placement:

- `io.github.pho001.synaptik.nn.module.Module` — the only public/protected owner of named direct
  replacement and its target-kind validation.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [target glossary](../../../../glossary.md#term-target); [validation glossary](../../../../glossary.md#term-validation))</span>
- `io.github.pho001.synaptik.nn.module.Parameter` — keeps a package-internal mutation primitive
  callable only by the module package; exposes no replacement method outside that ownership path.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- `io.github.pho001.synaptik.nn.module.Buffer` — keeps the equivalent package-internal primitive
  for persistent non-trainable state; exposes no replacement method outside that ownership path.
- `io.github.pho001.synaptik.nn.module.ModuleTest` — tests direct registration and replacement
  validation/rollback without adding a public test seam.<span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback))</span>
- `io.github.pho001.synaptik.nn.module.ModuleTreeTest` — tests direct versus recursive wrapper
  snapshots and descendant-path boundaries after replacement.
- `io.github.pho001.synaptik.nn.module.ParameterAndBufferTest` — tests wrapper identity, current
  exact-value observation, and absence of public/protected setters.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>

## Affected files

Expected production files:

- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Module.java`.
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Parameter.java`.
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Buffer.java`.

Expected test files:

- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/module/ModuleTest.java`.
- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/module/ModuleTreeTest.java`.
- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/module/ParameterAndBufferTest.java`.

[Planning](../../../../glossary.md#term-planning) files synchronized by this task:

- `docs/planning/extensions/nn/master-plan.md`.
- this task specification.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most:

- three production Java files;
- three [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) test files; and
- these two [planning](../../../../glossary.md#term-planning) files.

If implementation needs a new public [carrier](../../../../glossary.md#term-storage-carrier) type, a new package/[module](../../../../glossary.md#term-module)/build [dependency](../../../../glossary.md#term-dependency),
[descriptor](../../../../glossary.md#term-descriptor)-[schema](../../../../glossary.md#term-schema) [validation](../../../../glossary.md#term-validation), [checkpoint](../../../../glossary.md#term-checkpoint)/state-dictionary [API](../../../../glossary.md#term-application-programming-interface-api), locking/[concurrency](../../../../glossary.md#term-concurrency) mechanism,
architecture-test update, or more files, stop and propose a separate follow-up task.

## Acceptance criteria

- [`Module`](../../../../glossary.md#term-module) exposes exactly the two protected final void direct-binding methods named in Scope;
  they accept a local name and a new <span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [Scope glossary](../../../../glossary.md#term-scope))</span>[`Tensor`](../../../../glossary.md#tensor), and no public `Module` replacement method exists.
- [`Parameter`](../../../../glossary.md#term-parameter) and [`Buffer`](../../../../glossary.md#term-buffer) have no public or protected setter, rebind, replace, or update method.
  Their names and wrapper identities remain stable, and only <span data-glossary-navigation="true"> ([rebind glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction))</span>`Module` invokes their
  package-internal current-value replacement implementation.
- Both replacement methods validate `name`, then `value`, then the requested direct kind. Null
  name/value failures are `NullPointerException`; a missing name or a direct name registered for
  the other kind is `IllegalArgumentException`. Every failure leaves the target's current Tensor,
  registry contents, orders, child ownership, and mode unchanged.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [registry glossary](../../../../glossary.md#term-registry); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- A successful parameter/buffer replacement retains the exact supplied Tensor reference without
  copy or evaluation. <span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation))</span>`value()` immediately returns that exact reference in ordinary
  single-threaded use. The old Tensor object and expressions formed with it remain unchanged.
- Direct `parameters()`/`buffers()` and recursive maps preserve their current order, paths,
  unmodifiable structural-snapshot semantics, and exact wrapper identities after replacement.
  An already captured collection/map still contains that same wrapper; calling its <span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>`value()` sees
  the wrapper's current binding rather than a historical deep snapshot.
- A module may replace only its direct state. It cannot target a child or descendant by dot path;
  a parent must not receive a new recursive replacement API. Direct parameter and buffer names
  remain in their shared namespace, and replacement does not alter <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>`children()`, `train()`,
  `eval()`, `mode()`, or an already-created `ForwardContext`.
- Javadoc documents mutation ownership, direct-only target scope, validation/exception order,
  exact reference and old-expression semantics, structural versus value snapshot behavior, lack
  of versioning/checkpoint semantics, and lack of thread safety.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [validation glossary](../../../../glossary.md#term-validation); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Focused tests cover both kinds' success paths; exact wrapper/current-Tensor identity; preserved
  direct/recursive ordering and paths; old Tensor/forward-expression stability; invalid kind,
  missing name, null name, null value, and no-change failures; direct-only descendant rejection;
  and reflection-level absence of public/protected binding setters. Existing NN 0001/0002 tests
  remain valid except for deliberate assertions updated for the new contract.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- A separate documentation-focused agent pass has finalized affected documentation, Javadoc, and
  glossary impact in this same overall change.

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation pass runs:

```bash
./gradlew :extensions:nn:test
```

This single-[module](../../../../glossary.md#term-module) [API](../../../../glossary.md#term-application-programming-interface-api) change neither changes module boundaries nor the [Gradle](../../../../glossary.md#term-gradle) [graph](../../../../glossary.md#graph). Existing
architecture enforcement already covers the [model](../../../../glossary.md#term-model)-only [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) edge, so no architecture-test edit or
repeat is required. Repository-wide validation is deferred to the NN capability [checkpoint](../../../../glossary.md#term-checkpoint) or [CI](../../../../glossary.md#term-continuous-integration-ci).

Documentation pass runs after final [Javadoc](../../../../glossary.md#term-javadoc) edits:

```bash
./gradlew :extensions:nn:javadoc
git diff --check
```

The documentation pass verifies task/master-plan links and anchors, generated Javadoc, package
placement, current terminology against the glossary, and that no unlisted [CPU](../../../../glossary.md#term-cpu) or global-roadmap
change entered the diff. It reuses the implementation test result unless it changes executable
Java behavior; if it does, it reruns the focused NN test and records that reason.

## Dependencies

- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0001 and NN 0002 are Complete.
- Existing [model](../../../../glossary.md#term-model) [`Tensor`](../../../../glossary.md#tensor) immutability and descriptor contracts, ADR 0007, and the current
  model-only NN dependency edge remain accepted.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [dependency edge glossary](../../../../glossary.md#term-dependency-edge))</span>
- The user-authorized NN parallel exception recorded in the NN master plan remains in force.

## Follow-up tasks

- A later checkpoint/state-dictionary task must define persistent snapshots, loading/validation,
  file format, and any multi-binding atomicity; it must not infer those contracts from this task.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [validation glossary](../../../../glossary.md#term-validation); [binding glossary](../../../../glossary.md#term-binding); [atomicity glossary](../../../../glossary.md#term-atomicity))</span>
- `extensions/training` must define optimizer ownership, update sequencing, and synchronization
  over these stable wrappers when it reaches its own frontier.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0004: explicit [eager](../../../../glossary.md#term-eager-evaluation) initializers over caller-owned random sources.
- NN 0005: [`Linear`](../../../../glossary.md#term-linear-meanings) over stabilized [module](../../../../glossary.md#term-module) state and NN 0004 initializers.

## Architecture impact

Expected impact: None.

This task preserves the existing extension boundary and [Tensor](../../../../glossary.md#tensor) immutability contract. If a [binding](../../../../glossary.md#term-binding)
replacement needs [descriptor](../../../../glossary.md#term-descriptor) [schema](../../../../glossary.md#term-schema) checks, cross-binding [atomicity](../../../../glossary.md#term-atomicity), [checkpoint](../../../../glossary.md#term-checkpoint) behavior,
[optimizer](../../../../glossary.md#term-optimizer) [ownership](../../../../glossary.md#term-ownership-meanings), or a [dependency](../../../../glossary.md#term-dependency) change, stop and report the required architecture or
separate-task decision.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository.

Read:
- AGENTS.md
- ARCHITECTURE.md
- docs/planning/planning-guide.md
- docs/planning/extensions/nn/tasks/0003-validated-parameter-and-buffer-binding-replacement.md

Implement this task exactly as specified. Do not implement out-of-scope items or commit/push.
Stop and report any architecture or scope conflict.

After code implementation and focused module validation, hand the resulting diff and recorded test
evidence to a separate documentation-focused agent or thread with clean context. That targeted pass
must follow docs/developer-guide/documentation-rules.md and finalize affected documentation,
Javadoc, glossary impact, and documentation validation in the same overall change. It must not
repeat successful Java tests unless it changes executable behavior or records a concrete reason.

At the end, update this task with implementation notes, validation evidence including the
documentation-agent pass, completion summary, and final status. Do not mark the task Complete
before that pass finishes.
```

## Local decisions

- The direct state-name is the replacement selector. It is the existing stable module-owned
  identity and prevents callers from applying a foreign or stale wrapper to a module; wrapper
  identity itself remains stable in all direct and recursive maps.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `void` return avoids implying a value snapshot, old-value ownership transfer, or versioned
  update receipt. Callers that need the old exact Tensor retain it before replacement.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Non-nullity is the sole validation of a new Tensor. The existing module abstraction declares no
  descriptor schema and must not create one implicitly from its initial value.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [descriptor glossary](../../../../glossary.md#term-descriptor); [schema glossary](../../../../glossary.md#term-schema))</span>
- Replacement is deliberately individual rather than transactional. A later checkpoint or
  training task may define a multi-binding consistency contract without retrofitting it here.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [binding glossary](../../../../glossary.md#term-binding))</span>

## Known limitations

- No [checkpoint](../../../../glossary.md#term-checkpoint), [serialization](../../../../glossary.md#term-serialization), state dictionary, version, or [rollback](../../../../glossary.md#term-rollback) representation exists.
- No synchronization, atomic multi-binding view, or unsynchronized cross-thread visibility
  guarantee exists; external coordination is required.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [binding glossary](../../../../glossary.md#term-binding); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- A layer needing shape/dtype compatibility must validate its own forward contract until a later
  layer-specific or checkpoint schema task defines a reusable policy.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [dtype glossary](../../../../glossary.md#data-type--datatype); [schema glossary](../../../../glossary.md#term-schema))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation pass: `./gradlew :extensions:nn:test` passed on 2026-08-13. Its focused coverage
proves successful direct replacement, exact wrapper/current-[Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings), structural [snapshot](../../../../glossary.md#term-snapshot)
observation, validation order and no-change failures, direct-only descendant rejection, and the
absence of public/protected wrapper replacement APIs.

Documentation-focused pass `/root/nn_0003_docs` independently reviewed the final [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0003 diff,
current NN source and focused tests, `AGENTS.md`, `ARCHITECTURE.md`, the architecture index,
[ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0007, [dependency](../../../../glossary.md#term-dependency) rules, training-[graph](../../../../glossary.md#graph) explanation, NN 0001/0002 records, the NN master
plan, and the glossary. It applied the General, [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), [Planning](../../../../glossary.md#term-planning), and Example profiles.

The pass finalized the [`Module`](../../../../glossary.md#term-module), [`Parameter`](../../../../glossary.md#term-parameter), and [`Buffer`](../../../../glossary.md#term-buffer) Javadocs. They now state direct-only
Module [ownership](../../../../glossary.md#term-ownership-meanings), deterministic name-then-value-then-kind validation, exact-reference and
old-expression behavior, structural-versus-value snapshot behavior, absence of [descriptor](../../../../glossary.md#term-descriptor) [schema](../../../../glossary.md#term-schema)
validation, and the absence of version, [transaction](../../../../glossary.md#term-transaction), [checkpoint](../../../../glossary.md#term-checkpoint), [optimizer](../../../../glossary.md#term-optimizer), or thread-safety
contracts. The glossary was updated because the existing NN entry incorrectly described [binding](../../../../glossary.md#term-binding)
replacement as future work. No Tensor API, architecture document, ADR, dependency test, or
training-graph change is needed: this task preserves [immutable](../../../../glossary.md#term-immutable) Tensor identity/[provenance](../../../../glossary.md#provenance) and the
existing `model -> nn -> training` direction without adding training or execution behavior.

Documentation validation: `./gradlew :extensions:nn:javadoc` passed on 2026-08-13. Generated
Javadoc was inspected for the replacement, snapshot, validation, and limitation contracts. A
targeted local Markdown file, relative-link, heading-anchor, and fenced-code check passed for the
changed NN master plan, task record, and glossary. `git diff --check` passed with no output. The
implementation test evidence was reused because this documentation pass changed no executable
behavior.

## Implementation notes

The isolated implementation pass added the two protected final direct-[binding](../../../../glossary.md#term-binding) replacement methods
to [`Module`](../../../../glossary.md#term-module), package-internal wrapper mutation primitives, and focused tests for [identity](../../../../glossary.md#term-identity-meanings),
structural snapshots, [validation](../../../../glossary.md#term-validation) order, failure non-mutation, and direct-only boundaries.

The required independent documentation-focused pass finalized the affected Javadocs, corrected
the glossary's current-versus-planned replacement boundary, and synchronized this [planning](../../../../glossary.md#term-planning) record
and master-plan status.

## Completion summary

- Completed changes: Added protected direct [`Module`](../../../../glossary.md#term-module) replacement for current parameter and buffer
  bindings while preserving wrapper identity, local names, registry order, child ownership, mode,
  and old Tensor-expression provenance.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [buffer glossary](../../../../glossary.md#term-buffer); [bindings glossary](../../../../glossary.md#term-binding); [identity glossary](../../../../glossary.md#term-identity-meanings); [registry glossary](../../../../glossary.md#term-registry); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Tensor glossary](../../../../glossary.md#tensor); [provenance glossary](../../../../glossary.md#provenance))</span>
- Files changed or created: `Module.java`, `Parameter.java`, `Buffer.java`, three focused NN test
  classes, the NN master plan, this task record, and the NN glossary entry.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Tests and [validation](../../../../glossary.md#term-validation): Implementation `./gradlew :extensions:nn:test` passed; documentation
  `./gradlew :extensions:nn:javadoc` passed; Markdown links/anchors/fences and generated Javadoc
  were checked; <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`git diff --check` passed.
- Documentation impact: Finalized API Javadocs and updated the glossary. Tensor API, architecture
  documents, ADR 0007, dependency tests, and training-graph documentation require no change
  because no architecture, dependency, Tensor, training, or execution contract changed.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [dependency glossary](../../../../glossary.md#term-dependency); [graph glossary](../../../../glossary.md#graph))</span>
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
