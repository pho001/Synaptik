# Task 0002: [Module](../../../../glossary.md#term-module)-Tree [Ownership](../../../../glossary.md#term-ownership-meanings) and Recursive Mode Propagation

> **Editorial reading context — not a status update.** Parent ownership here is permanent
> membership in an NN module tree, not storage ownership. An application-built parent registers
> an exact child; discovery walks that tree to produce stable names such as encoder.weight.
> Mode propagation changes descendants' forward-behavior choices but performs no gradient
> calculation or parameter update. Discovery snapshots contain wrappers, so later wrapper reads
> can observe replacement; they are not detached value snapshots. Today's
> [module topology explanation](../../../../reference/nn/composition.md#current-nn-typed-model-composition-contract)
> and [state export ordering](../../../../reference/nn/state.md#current-nn-state-dictionary-contract)
> explain these related but distinct structures.

## Status

Complete

## Goal

Extend the [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [module](../../../../glossary.md#term-module) foundation with exclusively owned named child modules, deterministic
hierarchical discovery of [parameters](../../../../glossary.md#term-parameter) and [buffers](../../../../glossary.md#term-buffer), and all-or-nothing recursive training/[evaluation](../../../../glossary.md#term-evaluation)
mode changes. Preserve the direct-state [API](../../../../glossary.md#term-application-programming-interface-api) and [immutable](../../../../glossary.md#term-immutable) [Tensor](../../../../glossary.md#tensor) [binding](../../../../glossary.md#term-binding) [lifecycle](../../../../glossary.md#lifecycle) established by
NN 0001.

## [Scope](../../../../glossary.md#term-scope)

- Extend [`Module`](../../../../glossary.md#term-module) with protected final child registration:
  `protected final <T extends Module> T child(String name, T child)`.
- Make parameters, buffers, and children share one direct local-name namespace. Registration
  order remains declaration order across each category and child names are local path segments.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter); [buffers glossary](../../../../glossary.md#term-buffer))</span>
- Retain the existing direct `parameters()` and `buffers()` APIs. Add public snapshot access to
  direct children and to recursive state as insertion-ordered, unmodifiable maps:<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>
  `children()`, `parametersRecursively()`, and `buffersRecursively()`.
- Define a recursive state path as local names joined from the receiving root by `.`. Direct state
  uses its local name; a descendant state uses, for example, `encoder.layer1.weight`. Traverse
  depth-first: for each module visit its direct parameters, then direct buffers, then children in
  child-registration order. Descendant entries follow that same preorder. Parameters and buffers
  remain separate maps, so their equal paths are impossible because every module has a shared
  namespace.
- Enforce exclusive permanent tree ownership. A child must be non-null, non-blank-named, not the
  registering module, have no parent, and not be an ancestor of the registering module. Reject
  duplicate names and all failed registrations before modifying either module. There is no detach,
  rename, reparent, or shared-child API.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Make `train()` and `eval()` apply to the receiving module and every reachable owned descendant.
  Before changing any mode, perform an identity-based full-tree preflight that rejects a repeated
  module identity as an internal invariant violation. Only after preflight succeeds, assign the
  requested mode to the collected modules in deterministic preorder. Thus a defensive malformed
  cycle/shared-child failure cannot expose a partially propagated requested mode.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [preflight glossary](../../../../glossary.md#term-preflight); [invariant glossary](../../../../glossary.md#term-invariant))</span>
- Add focused tests for registration order and dot paths, direct-versus-recursive snapshots,
  namespace collisions, null/blank/self/already-owned/ancestor rejection with rollback, snapshot
  isolation, descendant mode propagation, and defensive all-or-nothing traversal failure through
  a test-local reflective corruption seam if ordinary API construction cannot form a cycle.<span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback))</span>
- Update `Module` Javadoc for its new ownership, traversal, mode, snapshot, and failure contracts.
  Perform the required independent documentation-focused finalization after executable work.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Binding replacement, locking, concurrent mutation semantics, checkpoint/state dictionaries,
  serialization, parameter groups, optimizer algorithms, gradients, or training sessions.<span data-glossary-navigation="true"> ([Binding glossary](../../../../glossary.md#term-binding); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [serialization glossary](../../../../glossary.md#term-serialization); [parameter glossary](../../../../glossary.md#term-parameter); [optimizer glossary](../../../../glossary.md#term-optimizer); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Layers, generic `forward` signatures, initializers, random sources, functional conveniences,
  and Tensor expression evaluation.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [Tensor glossary](../../../../glossary.md#tensor); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- Any Gradle project/dependency change, architecture-contract/ADR/explanatory-architecture update,
  architecture-test update, CPU/Engine/compiler/runtime/prepare/backend work, or global-roadmap
  update.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>: NN ownership, immutable Tensor identity,
  one-way extension dependencies, and testing discipline.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [immutable glossary](../../../../glossary.md#term-immutable); [Tensor glossary](../../../../glossary.md#tensor); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module))</span>.
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>.
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>.
- [NN master plan](../master-plan.md).
- [Task 0001](0001-module-parameter-buffer-and-forward-context-foundation.md).
- [Planning Guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>.

## Architecture constraints

- `extensions/nn` continues to depend only on `modules/model`; it must not import training,
  compiler, runtime, prepare, engine, or a concrete backend.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
- [`Module`](../../../../glossary.md#term-module) owns nesting, named state discovery, and train/eval metadata; training remains the
  downstream owner of optimizer algorithms and update orchestration.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- [`Tensor`](../../../../glossary.md#tensor), [`Parameter`](../../../../glossary.md#term-parameter), and [`Buffer`](../../../../glossary.md#term-buffer) keep NN 0001's immutable exact-binding contract. This task
  must not add replacement, gradient, optimizer, execution, backend, or runtime state.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [immutable glossary](../../../../glossary.md#term-immutable); [binding glossary](../../../../glossary.md#term-binding); [gradient glossary](../../../../glossary.md#term-gradient); [backend glossary](../../../../glossary.md#backend))</span>
- Module mode is composition [metadata](../../../../glossary.md#term-metadata), not per-[run](../../../../glossary.md#term-run) [mutable](../../../../glossary.md#term-mutable) execution state. `ForwardContext`
  remains an immutable local mode snapshot; no recursive context object is introduced.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- The authorized parallel exception is implementation-order only. Do not edit dirty CPU files,
  the global roadmap, build graph, architecture documents, or dependency enforcement for this
  model-only API extension.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [graph glossary](../../../../glossary.md#graph); [dependency glossary](../../../../glossary.md#term-dependency); [model glossary](../../../../glossary.md#term-model); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.nn.module` — owns the complete public module tree, named state, and
  forward-mode contract.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [forward-mode glossary](../../../../glossary.md#primal-adjoint-and-differentiation-seed))</span>
- `io.github.pho001.synaptik.model.tensor` — used only by the already-existing [`Parameter`](../../../../glossary.md#term-parameter) and
  [`Buffer`](../../../../glossary.md#term-buffer) [bindings](../../../../glossary.md#term-binding); no [model](../../../../glossary.md#term-model) type changes.

Packages added or changed:

- No package is added. Only `io.github.pho001.synaptik.nn.module.Module` changes.

Type placement:

- `io.github.pho001.synaptik.nn.module.Module` — the only owning type for child registration,
  recursive state traversal, and recursive mode propagation; no new facade or traversal utility
  is needed.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade))</span>
- `io.github.pho001.synaptik.nn.module.ModuleTest` — preserves NN 0001 direct-state regression
  coverage and adds shared-namespace behavior.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [regression glossary](../../../../glossary.md#term-regression))</span>
- `io.github.pho001.synaptik.nn.module.ModuleTreeTest` — isolates child ownership, traversal, and
  recursive-mode behavior without making a production test seam public.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

## Affected files

Expected production file:

- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Module.java`.

Expected test files:

- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/module/ModuleTest.java`.
- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/module/ModuleTreeTest.java`.

[Planning](../../../../glossary.md#term-planning) files synchronized by this task:

- `docs/planning/extensions/nn/master-plan.md`.
- this task specification.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most:

- one production Java file;
- two [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) test files; and
- these two [planning](../../../../glossary.md#term-planning) files.

If implementation needs a new public [carrier](../../../../glossary.md#term-storage-carrier) type, another [module](../../../../glossary.md#term-module)/build [dependency](../../../../glossary.md#term-dependency), an
architecture-test change, [binding](../../../../glossary.md#term-binding) mutation, [checkpoint](../../../../glossary.md#term-checkpoint) format, or more files, stop and propose a
separate follow-up task.

## Acceptance criteria

- `child(name, child)` is protected, final, generic enough to return the concrete child type, and
  installs exactly one permanently owned child only after every validation succeeds.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Parameters, buffers, and children occupy one shared direct namespace. Null and blank names,
  duplicate names across all three categories, null children, self-registration, reparenting,
  repeated attachment, and attempted ancestor cycles fail deterministically with no observable
  registry or ownership mutation.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- `children()` returns an insertion-ordered, unmodifiable snapshot of direct child name to child
  mappings. Existing direct <span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>`parameters()` and `buffers()` retain their direct-only declaration-
  order semantics and unmodifiable snapshot behavior.
- `parametersRecursively()` and `buffersRecursively()` return independent insertion-ordered,
  unmodifiable snapshots from the receiving root. They use the exact existing [`Parameter`](../../../../glossary.md#term-parameter)/
  [`Buffer`](../../../../glossary.md#term-buffer) instances and the specified depth-first dot-path order.
- `train()` and `eval()` set every owned descendant, including the receiver, to the requested
  `ForwardMode`; previously obtained `ForwardContext` values remain unchanged. A defensive
  repeated-identity preflight fails before any requested mode assignment.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [preflight glossary](../../../../glossary.md#term-preflight))</span>
- No generic `Module.forward(...)` method, Tensor subtype/mutation, binding replacement,
  checkpoint, optimizer, execution API, Gradle/dependency change, or architecture change is
  introduced.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [binding glossary](../../../../glossary.md#term-binding); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [optimizer glossary](../../../../glossary.md#term-optimizer); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Focused NN tests observe every contract above. Existing NN 0001 tests remain valid except for
  deliberate assertions updated to reflect the now-shared child namespace.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Every changed public [`Module`](../../../../glossary.md#term-module) member has complete meaningful Javadoc, including parameter,
  return, failure, ordering, snapshot, ownership, and thread-safety semantics where relevant.
  A separate documentation-focused agent finalizes Javadoc, planning wording, glossary impact,
  links, and generated Javadoc in the same overall change.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning))</span>

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

The documentation pass verifies the task/master-plan links and anchors, Javadoc pages, package
placement, terminology against the glossary, and that no unlisted [CPU](../../../../glossary.md#term-cpu) or global-roadmap changes
entered this task's diff. It reuses the implementation test result unless executable Java changes
after that result.

## Dependencies

- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0001 is Complete.
- Existing [model](../../../../glossary.md#term-model) [Tensor](../../../../glossary.md#tensor) contracts and [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0007 remain accepted.
- The user-authorized NN parallel exception recorded in the NN master plan remains in force.

## Follow-up tasks

- NN 0003: controlled parameter and buffer binding replacement, including its validation and
  concurrency/checkpoint interaction.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [buffer glossary](../../../../glossary.md#term-buffer); [binding glossary](../../../../glossary.md#term-binding); [validation glossary](../../../../glossary.md#term-validation); [concurrency glossary](../../../../glossary.md#term-concurrency); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- A later checkpoint/state-dictionary task: persistence over the stable recursive maps; do not
  infer a file format here.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence))</span>
- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0004: explicit [eager](../../../../glossary.md#term-eager-evaluation) initializers, independently of tree traversal.
- NN 0005: [`Linear`](../../../../glossary.md#term-linear-meanings) over the stabilized [ownership](../../../../glossary.md#term-ownership-meanings) and initializer contracts.

## Architecture impact

Expected impact: None.

This task realizes the existing [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0007 responsibility that [`Module`](../../../../glossary.md#term-module) declares and traverses
state. It changes no module edge or [dependency](../../../../glossary.md#term-dependency) rule; therefore architecture-test and architecture
documentation changes are intentionally unnecessary. If implementation needs an architectural
change, stop and report it rather than editing those documents.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository. Read AGENTS.md, ARCHITECTURE.md,
docs/planning/planning-guide.md, docs/developer-guide/documentation-rules.md, and
docs/planning/extensions/nn/tasks/0002-module-tree-ownership-and-recursive-mode-propagation.md
in full. Implement that task exactly; do not touch dirty CPU work, the global roadmap, or
out-of-scope behavior. Stop and report any architecture or scope conflict. Do not commit or push.

After module validation, hand the resulting diff and exact test evidence to a separate
documentation-focused clean context. That pass must finalize affected Javadoc, planning text,
glossary impact, and documentation validation under the documentation rules without repeating
successful Java tests unless executable behavior changes. Update the task evidence and completion
summary only after that pass; do not mark it Complete first.
```

## Local decisions

- One shared direct namespace covers parameters, buffers, and children. This prevents ambiguous
  state paths and makes a future checkpoint namespace stable. Local names must not contain
  {@code .}, which is reserved exclusively as the recursive path separator.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Child ownership is exclusive and permanent. A module can have zero or one parent and a child
  cannot be reused, removed, renamed, or reparented in this task.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [module glossary](../../../../glossary.md#term-module))</span>
- Recursive maps, not a new public state-entry type, carry qualified names while preserving the
  existing <span data-glossary-navigation="true"> ([carry glossary](../../../../glossary.md#term-carry))</span>[`Parameter`](../../../../glossary.md#term-parameter) and [`Buffer`](../../../../glossary.md#term-buffer) APIs. The maps are defensive insertion-ordered snapshots.
- State traversal is depth-first with direct parameters before direct buffers before descendants.
- Recursive mode changes preflight identity uniqueness before setting a mode, yielding all-or-
  nothing requested-mode propagation even for a defensively detected malformed tree.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>

## Known limitations

- The API is not thread-safe. Concurrent declarations, traversals, or mode changes require caller
  synchronization; this task does not introduce locks or a concurrent snapshot contract.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [thread-safe glossary](../../../../glossary.md#term-thread-safe); [synchronization glossary](../../../../glossary.md#term-synchronization); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Parent [ownership](../../../../glossary.md#term-ownership-meanings) is intentionally internal to [`Module`](../../../../glossary.md#term-module); there is no public parent lookup or
  mutable tree-management API.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- No public construction path can create a cycle or shared child. The defensive preflight exists
  only to retain fail-fast, non-partial behavior if an invariant is corrupted outside the API.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [invariant glossary](../../../../glossary.md#term-invariant))</span>
- Recursive state maps are discovery-only. They neither replace bindings nor serialize or restore
  values.<span data-glossary-navigation="true"> ([bindings glossary](../../../../glossary.md#term-binding))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation pass: `./gradlew :extensions:nn:test` passed after the dot-name correction on
2026-08-13 (5 actionable tasks: 3 executed, 2 up-to-date). Documentation review identified that
permitting `.` in a local name could make distinct child structures yield the same recursive path;
the implementation now rejects `.` at every declaration entry point. The focused suite covers
direct and recursive insertion order, dot-qualified paths, reserved-separator rejection, shared
namespace collisions, invalid child registration [rollback](../../../../glossary.md#term-rollback), [snapshot](../../../../glossary.md#term-snapshot) isolation, recursive mode
propagation, and reflective repeated-[identity](../../../../glossary.md#term-identity-meanings) all-or-nothing failure.

Implementation also ran `git diff --check` after the correction; it passed with no output.

Architecture-test execution is intentionally not repeated: this task changes neither the [Gradle](../../../../glossary.md#term-gradle)
[graph](../../../../glossary.md#graph) nor a [dependency](../../../../glossary.md#term-dependency) boundary, and the task validation section records that existing enforcement
covers the unchanged [model](../../../../glossary.md#term-model)-only edge. Repository-wide validation remains deferred to the [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)
capability [checkpoint](../../../../glossary.md#term-checkpoint) or [CI](../../../../glossary.md#term-continuous-integration-ci).

Documentation-focused pass `/root/nn_0002_docs` independently reviewed the final combined NN
0002 diff against `AGENTS.md`, `ARCHITECTURE.md`, [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0007, training-graph and dependency-rule
explanations, the NN 0001/0002 records, the current NN public source, and focused tests. It used
the General, [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), [Planning](../../../../glossary.md#term-planning), and Example documentation profiles. It found and required the
dot-name repair before completion because `.` is the recursive-path separator; after the repair,
all declaration entry points reject it and the focused implementation suite was rerun successfully.

The pass finalized [`Module`](../../../../glossary.md#term-module) Javadoc and updated the NN glossary entry for permanent tree
[ownership](../../../../glossary.md#term-ownership-meanings), shared local namespace, reserved dot separator, deterministic [immutable](../../../../glossary.md#term-immutable) recursive
snapshots, and [preflight](../../../../glossary.md#term-preflight)-[atomic](../../../../glossary.md#resource-transaction-and-rollback) mode propagation. It made no architecture-document, ADR,
training-graph, [Tensor](../../../../glossary.md#tensor) API, or architecture-test change: NN 0002 realizes the accepted existing
NN ownership boundary, retains the existing model-only dependency, and introduces no Tensor,
[compiler](../../../../glossary.md#term-compiler), training, or dependency-rule contract. It also made no example change because the
glossary's `encoder.layer1.weight` path is descriptive and agrees with the tested API.

Documentation validation: `./gradlew :extensions:nn:javadoc` passed on 2026-08-13 (3 actionable
tasks: 1 executed, 2 up-to-date); generated `Module` Javadoc was inspected for the changed
ownership, path, snapshot, failure, and thread-safety contracts. A targeted local Markdown
file/anchor and fence check of the changed NN planning files and glossary passed. `git diff --check`
passed with no output. The implementation's final `./gradlew :extensions:nn:test` evidence above
was reused; this documentation pass made no executable Java change.

## Implementation notes

[`Module`](../../../../glossary.md#term-module) now owns a permanently attached child tree with one shared direct namespace, direct and
recursive [immutable](../../../../glossary.md#term-immutable) discovery snapshots, and [identity](../../../../glossary.md#term-identity-meanings)-preflighted recursive mode propagation.
Local names reserve `.` as the path separator, preventing recursive-path collisions. The
implementation changed only the planned [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) production and test files; no [dependency](../../../../glossary.md#term-dependency),
architecture, [CPU](../../../../glossary.md#term-cpu), [Engine](../../../../glossary.md#term-engine), [compiler](../../../../glossary.md#term-compiler), [runtime](../../../../glossary.md#term-runtime), [prepare](../../../../glossary.md#prepare), [backend](../../../../glossary.md#backend), [binding](../../../../glossary.md#term-binding)-replacement, [checkpoint](../../../../glossary.md#term-checkpoint),
[optimizer](../../../../glossary.md#term-optimizer), layer, or execution work was added.

## Completion summary

- Completed changes: Added permanent exclusive child ownership, collision-free shared local names,
  deterministic immutable recursive parameter/buffer discovery, and preflight-atomic recursive
  train/eval propagation to <span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [immutable glossary](../../../../glossary.md#term-immutable); [parameter glossary](../../../../glossary.md#term-parameter); [buffer glossary](../../../../glossary.md#term-buffer); [preflight glossary](../../../../glossary.md#term-preflight); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>[`Module`](../../../../glossary.md#term-module).
- Files changed or created: `Module.java`, `ModuleTest.java`, `ModuleTreeTest.java`, the NN master
  plan, this task record, and the NN glossary entry.
- Tests and [validation](../../../../glossary.md#term-validation): Implementation `./gradlew :extensions:nn:test` passed after the final
  dot-name repair (5 actionable tasks: 3 executed, 2 up-to-date); implementation `git diff --check`
  passed. Documentation `./gradlew :extensions:nn:javadoc` passed (3 actionable tasks: 1
  executed, 2 up-to-date); changed Markdown links, anchors, fences, generated Javadoc, and final
  whitespace were checked.
- Documentation-agent review: `/root/nn_0002_docs` independently finalized the API/Javadoc and
  planning documentation under the General, API/Javadoc, Planning, and Example profiles.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Updated the glossary and NN planning status; architecture docs, ADR 0007,
  dependency rules, training graph, Tensor API, and architecture tests require no change because
  no architecture or dependency boundary changed.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [dependency glossary](../../../../glossary.md#term-dependency); [training graph glossary](../../../../glossary.md#training-graph); [Tensor glossary](../../../../glossary.md#tensor))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: `Module` documents local-name separator reservation, exclusive ownership,
  deterministic traversal, immutable snapshots, preflight-atomic propagation, and non-thread-safe
  caller synchronization.<span data-glossary-navigation="true"> ([thread-safe glossary](../../../../glossary.md#term-thread-safe); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- Glossary impact: Updated the existing [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) entry; no new reusable term was introduced.
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
