# Task 0003: Validated Parameter and Buffer Binding Replacement

## Status

Complete

## Goal

Allow a concrete module to replace the current Tensor of one of its own direct parameters or
buffers without mutating Tensor, exposing a public setter, changing module-tree structure, or
introducing optimizer/checkpoint behavior.

## Scope

- Change `Parameter` and `Buffer` so each preserves its declaration-time local name and wrapper
  identity while retaining one replaceable current non-null `Tensor` reference.
- Add exactly these protected final, direct-state methods to `Module`:

  ```java
  protected final void replaceParameter(String name, Tensor value)
  protected final void replaceBuffer(String name, Tensor value)
  ```

  Each method targets only a direct declaration of its matching kind on the receiving module;
  `name` is a local name, never a dot path. A subclass changes a retained binding through its
  declaring module, for example `replaceParameter("weight", nextWeight)`. It must not mutate a
  `Parameter` or `Buffer` directly.
- Make the replacement sequence deterministic: reject a null `name`, then a null `value`, then
  reject a name that does not identify a direct declaration of the requested kind. Only after all
  validation succeeds may the binding's current reference change. A name occupied by a direct
  buffer is not a parameter, and vice versa.
- Preserve the existing declaration order, shared direct-name namespace, child ownership, mode,
  and recursive dot paths. Successful replacement changes neither maps/lists nor their ordering.
- Define value observation precisely. `Parameter.value()` and `Buffer.value()` return the exact
  Tensor currently bound at that call. Previously returned Tensor references and Tensor
  expressions already constructed from them stay unchanged; Tensor's immutable identity and
  provenance are never modified. Direct and recursive discovery snapshots are structural
  snapshots containing the same `Parameter`/`Buffer` wrapper objects, so wrappers obtained from a
  snapshot expose their subsequently current binding.
- Add no descriptor, dtype, shape, layout, provenance, gradient-eligibility, or storage
  compatibility check beyond non-nullity. `Module` has no declared binding schema, and model
  supports unresolved/dynamic Tensor descriptors; freezing a descriptor from the initial value
  would invent a layer policy and reject valid future module contracts. A later concrete layer or
  checkpoint task owns any shape/schema validation it needs.
- Specify that this API has no version counter or historical-value/snapshot carrier, no batch or
  tree-wide replacement transaction, and no concurrency guarantee. It is not thread-safe:
  callers must externally synchronize concurrent replacement, declaration, traversal, mode, and
  forward construction when a consistent multi-binding view matters. One successful call changes
  only one binding; a failed call changes none.
- Update all affected public and contract-relevant Javadocs and focused tests. Use an independent
  documentation-focused agent/thread after implementation to finalize Javadoc, glossary impact,
  and this planning record.

## Out of scope

- Public, protected, or package-external setters/rebind methods on `Parameter` or `Buffer`.
- Replacement by recursive path, parent-mediated child replacement, replacement by an arbitrary
  wrapper object, detach/reparent/rename, or module-tree mutation.
- A batch/state-dictionary replacement API, transaction/rollback across multiple bindings,
  versioning, locking, visibility guarantees, or concurrent update protocol.
- Checkpoints, serialization, optimizer algorithms, parameter groups, training sessions,
  gradients, compiled update graphs, or any `extensions/training` API.
- Layers, initializers, RNG, generic `forward` signatures, Tensor-operation semantics, numerical
  evaluation, compiler, runtime, prepare, Engine, or backend work.
- Gradle/dependency changes, architecture-contract/ADR/explanatory-architecture updates,
  architecture-test edits, global-roadmap edits, or unrelated refactoring.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md): immutable Tensor identity/provenance,
  NN ownership, extension direction, and testing discipline.
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md).
- [Dependency rules](../../../../architecture/dependency-rules.md).
- [Training graph](../../../../architecture/training-graph.md).
- [NN master plan](../master-plan.md).
- [Task 0001](0001-module-parameter-buffer-and-forward-context-foundation.md) and
  [Task 0002](0002-module-tree-ownership-and-recursive-mode-propagation.md).
- [Planning Guide](../../../planning-guide.md).
- [Documentation rules](../../../../developer-guide/documentation-rules.md) and its
  [Planning profile](../../../../developer-guide/documentation/planning-style.md).

## Architecture constraints

- `extensions/nn` continues to depend only on `modules/model`. It must not import training,
  compiler, runtime, prepare, engine, or a concrete backend.
- `Module` owns declaration and controlled replacement of direct module state. `Parameter` and
  `Buffer` remain named state carriers, not Tensor subtypes and not optimizer APIs. Training stays
  downstream and may later consume the stable wrapper identities.
- Replacement selects a direct wrapper through the receiving module's existing local registry.
  It never replaces a `Parameter`/`Buffer` object, a map entry, or any Tensor internals.
- Tensor remains public model state with immutable identity, descriptor, and expression
  provenance. Rebinding only determines which distinct Tensor a later `value()` call returns.
- The existing mutable `Module` API has no thread-safety guarantee. This task must document that
  replacement adds no locking, atomic tree state, cross-binding consistency, or Java-memory-model
  visibility promise for unsynchronized callers.
- The authorized NN parallel exception is implementation-order only. Do not touch dirty CPU files,
  the global roadmap, build graph, architecture documents, or dependency enforcement.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.nn.module` — owns the complete public module-state and controlled
  binding-replacement contract.
- `io.github.pho001.synaptik.model.tensor` — supplies the existing `Tensor` references only; no
  model source changes are allowed.

Packages added or changed:

- No package is added. Only the existing `io.github.pho001.synaptik.nn.module` contracts change.

Type placement:

- `io.github.pho001.synaptik.nn.module.Module` — the only public/protected owner of named direct
  replacement and its target-kind validation.
- `io.github.pho001.synaptik.nn.module.Parameter` — keeps a package-internal mutation primitive
  callable only by the module package; exposes no replacement method outside that ownership path.
- `io.github.pho001.synaptik.nn.module.Buffer` — keeps the equivalent package-internal primitive
  for persistent non-trainable state; exposes no replacement method outside that ownership path.
- `io.github.pho001.synaptik.nn.module.ModuleTest` — tests direct registration and replacement
  validation/rollback without adding a public test seam.
- `io.github.pho001.synaptik.nn.module.ModuleTreeTest` — tests direct versus recursive wrapper
  snapshots and descendant-path boundaries after replacement.
- `io.github.pho001.synaptik.nn.module.ParameterAndBufferTest` — tests wrapper identity, current
  exact-value observation, and absence of public/protected setters.

## Affected files

Expected production files:

- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Module.java`.
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Parameter.java`.
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Buffer.java`.

Expected test files:

- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/module/ModuleTest.java`.
- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/module/ModuleTreeTest.java`.
- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/module/ParameterAndBufferTest.java`.

Planning files synchronized by this task:

- `docs/planning/extensions/nn/master-plan.md`.
- this task specification.

## Maximum scope

This task may create or modify at most:

- three production Java files;
- three NN test files; and
- these two planning files.

If implementation needs a new public carrier type, a new package/module/build dependency,
descriptor-schema validation, checkpoint/state-dictionary API, locking/concurrency mechanism,
architecture-test update, or more files, stop and propose a separate follow-up task.

## Acceptance criteria

- `Module` exposes exactly the two protected final void direct-binding methods named in Scope;
  they accept a local name and a new `Tensor`, and no public `Module` replacement method exists.
- `Parameter` and `Buffer` have no public or protected setter, rebind, replace, or update method.
  Their names and wrapper identities remain stable, and only `Module` invokes their
  package-internal current-value replacement implementation.
- Both replacement methods validate `name`, then `value`, then the requested direct kind. Null
  name/value failures are `NullPointerException`; a missing name or a direct name registered for
  the other kind is `IllegalArgumentException`. Every failure leaves the target's current Tensor,
  registry contents, orders, child ownership, and mode unchanged.
- A successful parameter/buffer replacement retains the exact supplied Tensor reference without
  copy or evaluation. `value()` immediately returns that exact reference in ordinary
  single-threaded use. The old Tensor object and expressions formed with it remain unchanged.
- Direct `parameters()`/`buffers()` and recursive maps preserve their current order, paths,
  unmodifiable structural-snapshot semantics, and exact wrapper identities after replacement.
  An already captured collection/map still contains that same wrapper; calling its `value()` sees
  the wrapper's current binding rather than a historical deep snapshot.
- A module may replace only its direct state. It cannot target a child or descendant by dot path;
  a parent must not receive a new recursive replacement API. Direct parameter and buffer names
  remain in their shared namespace, and replacement does not alter `children()`, `train()`,
  `eval()`, `mode()`, or an already-created `ForwardContext`.
- Javadoc documents mutation ownership, direct-only target scope, validation/exception order,
  exact reference and old-expression semantics, structural versus value snapshot behavior, lack
  of versioning/checkpoint semantics, and lack of thread safety.
- Focused tests cover both kinds' success paths; exact wrapper/current-Tensor identity; preserved
  direct/recursive ordering and paths; old Tensor/forward-expression stability; invalid kind,
  missing name, null name, null value, and no-change failures; direct-only descendant rejection;
  and reflection-level absence of public/protected binding setters. Existing NN 0001/0002 tests
  remain valid except for deliberate assertions updated for the new contract.
- A separate documentation-focused agent pass has finalized affected documentation, Javadoc, and
  glossary impact in this same overall change.

## Tests / validation

Implementation pass runs:

```bash
./gradlew :extensions:nn:test
```

This single-module API change neither changes module boundaries nor the Gradle graph. Existing
architecture enforcement already covers the model-only NN edge, so no architecture-test edit or
repeat is required. Repository-wide validation is deferred to the NN capability checkpoint or CI.

Documentation pass runs after final Javadoc edits:

```bash
./gradlew :extensions:nn:javadoc
git diff --check
```

The documentation pass verifies task/master-plan links and anchors, generated Javadoc, package
placement, current terminology against the glossary, and that no unlisted CPU or global-roadmap
change entered the diff. It reuses the implementation test result unless it changes executable
Java behavior; if it does, it reruns the focused NN test and records that reason.

## Dependencies

- NN 0001 and NN 0002 are Complete.
- Existing model `Tensor` immutability and descriptor contracts, ADR 0007, and the current
  model-only NN dependency edge remain accepted.
- The user-authorized NN parallel exception recorded in the NN master plan remains in force.

## Follow-up tasks

- A later checkpoint/state-dictionary task must define persistent snapshots, loading/validation,
  file format, and any multi-binding atomicity; it must not infer those contracts from this task.
- `extensions/training` must define optimizer ownership, update sequencing, and synchronization
  over these stable wrappers when it reaches its own frontier.
- NN 0004: explicit eager initializers over caller-owned random sources.
- NN 0005: `Linear` over stabilized module state and NN 0004 initializers.

## Architecture impact

Expected impact: None.

This task preserves the existing extension boundary and Tensor immutability contract. If a binding
replacement needs descriptor schema checks, cross-binding atomicity, checkpoint behavior,
optimizer ownership, or a dependency change, stop and report the required architecture or
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
  identity itself remains stable in all direct and recursive maps.
- `void` return avoids implying a value snapshot, old-value ownership transfer, or versioned
  update receipt. Callers that need the old exact Tensor retain it before replacement.
- Non-nullity is the sole validation of a new Tensor. The existing module abstraction declares no
  descriptor schema and must not create one implicitly from its initial value.
- Replacement is deliberately individual rather than transactional. A later checkpoint or
  training task may define a multi-binding consistency contract without retrofitting it here.

## Known limitations

- No checkpoint, serialization, state dictionary, version, or rollback representation exists.
- No synchronization, atomic multi-binding view, or unsynchronized cross-thread visibility
  guarantee exists; external coordination is required.
- A layer needing shape/dtype compatibility must validate its own forward contract until a later
  layer-specific or checkpoint schema task defines a reusable policy.

## Validation evidence

Implementation pass: `./gradlew :extensions:nn:test` passed on 2026-08-13. Its focused coverage
proves successful direct replacement, exact wrapper/current-Tensor identity, structural snapshot
observation, validation order and no-change failures, direct-only descendant rejection, and the
absence of public/protected wrapper replacement APIs.

Documentation-focused pass `/root/nn_0003_docs` independently reviewed the final NN 0003 diff,
current NN source and focused tests, `AGENTS.md`, `ARCHITECTURE.md`, the architecture index,
ADR 0007, dependency rules, training-graph explanation, NN 0001/0002 records, the NN master
plan, and the glossary. It applied the General, API/Javadoc, Planning, and Example profiles.

The pass finalized the `Module`, `Parameter`, and `Buffer` Javadocs. They now state direct-only
Module ownership, deterministic name-then-value-then-kind validation, exact-reference and
old-expression behavior, structural-versus-value snapshot behavior, absence of descriptor schema
validation, and the absence of version, transaction, checkpoint, optimizer, or thread-safety
contracts. The glossary was updated because the existing NN entry incorrectly described binding
replacement as future work. No Tensor API, architecture document, ADR, dependency test, or
training-graph change is needed: this task preserves immutable Tensor identity/provenance and the
existing `model -> nn -> training` direction without adding training or execution behavior.

Documentation validation: `./gradlew :extensions:nn:javadoc` passed on 2026-08-13. Generated
Javadoc was inspected for the replacement, snapshot, validation, and limitation contracts. A
targeted local Markdown file, relative-link, heading-anchor, and fenced-code check passed for the
changed NN master plan, task record, and glossary. `git diff --check` passed with no output. The
implementation test evidence was reused because this documentation pass changed no executable
behavior.

## Implementation notes

The isolated implementation pass added the two protected final direct-binding replacement methods
to `Module`, package-internal wrapper mutation primitives, and focused tests for identity,
structural snapshots, validation order, failure non-mutation, and direct-only boundaries.

The required independent documentation-focused pass finalized the affected Javadocs, corrected
the glossary's current-versus-planned replacement boundary, and synchronized this planning record
and master-plan status.

## Completion summary

- Completed changes: Added protected direct `Module` replacement for current parameter and buffer
  bindings while preserving wrapper identity, local names, registry order, child ownership, mode,
  and old Tensor-expression provenance.
- Files changed or created: `Module.java`, `Parameter.java`, `Buffer.java`, three focused NN test
  classes, the NN master plan, this task record, and the NN glossary entry.
- Tests and validation: Implementation `./gradlew :extensions:nn:test` passed; documentation
  `./gradlew :extensions:nn:javadoc` passed; Markdown links/anchors/fences and generated Javadoc
  were checked; `git diff --check` passed.
- Documentation impact: Finalized API Javadocs and updated the glossary. Tensor API, architecture
  documents, ADR 0007, dependency tests, and training-graph documentation require no change
  because no architecture, dependency, Tensor, training, or execution contract changed.
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
