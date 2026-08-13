# Task 0002: Module-Tree Ownership and Recursive Mode Propagation

## Status

Complete

## Goal

Extend the NN module foundation with exclusively owned named child modules, deterministic
hierarchical discovery of parameters and buffers, and all-or-nothing recursive training/evaluation
mode changes. Preserve the direct-state API and immutable Tensor binding lifecycle established by
NN 0001.

## Scope

- Extend `Module` with protected final child registration:
  `protected final <T extends Module> T child(String name, T child)`.
- Make parameters, buffers, and children share one direct local-name namespace. Registration
  order remains declaration order across each category and child names are local path segments.
- Retain the existing direct `parameters()` and `buffers()` APIs. Add public snapshot access to
  direct children and to recursive state as insertion-ordered, unmodifiable maps:
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
  rename, reparent, or shared-child API.
- Make `train()` and `eval()` apply to the receiving module and every reachable owned descendant.
  Before changing any mode, perform an identity-based full-tree preflight that rejects a repeated
  module identity as an internal invariant violation. Only after preflight succeeds, assign the
  requested mode to the collected modules in deterministic preorder. Thus a defensive malformed
  cycle/shared-child failure cannot expose a partially propagated requested mode.
- Add focused tests for registration order and dot paths, direct-versus-recursive snapshots,
  namespace collisions, null/blank/self/already-owned/ancestor rejection with rollback, snapshot
  isolation, descendant mode propagation, and defensive all-or-nothing traversal failure through
  a test-local reflective corruption seam if ordinary API construction cannot form a cycle.
- Update `Module` Javadoc for its new ownership, traversal, mode, snapshot, and failure contracts.
  Perform the required independent documentation-focused finalization after executable work.

## Out of scope

- Binding replacement, locking, concurrent mutation semantics, checkpoint/state dictionaries,
  serialization, parameter groups, optimizer algorithms, gradients, or training sessions.
- Layers, generic `forward` signatures, initializers, random sources, functional conveniences,
  and Tensor expression evaluation.
- Any Gradle project/dependency change, architecture-contract/ADR/explanatory-architecture update,
  architecture-test update, CPU/Engine/compiler/runtime/prepare/backend work, or global-roadmap
  update.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md): NN ownership, immutable Tensor identity,
  one-way extension dependencies, and testing discipline.
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md).
- [Dependency rules](../../../../architecture/dependency-rules.md).
- [Training graph](../../../../architecture/training-graph.md).
- [NN master plan](../master-plan.md).
- [Task 0001](0001-module-parameter-buffer-and-forward-context-foundation.md).
- [Planning Guide](../../../planning-guide.md).

## Architecture constraints

- `extensions/nn` continues to depend only on `modules/model`; it must not import training,
  compiler, runtime, prepare, engine, or a concrete backend.
- `Module` owns nesting, named state discovery, and train/eval metadata; training remains the
  downstream owner of optimizer algorithms and update orchestration.
- `Tensor`, `Parameter`, and `Buffer` keep NN 0001's immutable exact-binding contract. This task
  must not add replacement, gradient, optimizer, execution, backend, or runtime state.
- Module mode is composition metadata, not per-run mutable execution state. `ForwardContext`
  remains an immutable local mode snapshot; no recursive context object is introduced.
- The authorized parallel exception is implementation-order only. Do not edit dirty CPU files,
  the global roadmap, build graph, architecture documents, or dependency enforcement for this
  model-only API extension.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.nn.module` — owns the complete public module tree, named state, and
  forward-mode contract.
- `io.github.pho001.synaptik.model.tensor` — used only by the already-existing `Parameter` and
  `Buffer` bindings; no model type changes.

Packages added or changed:

- No package is added. Only `io.github.pho001.synaptik.nn.module.Module` changes.

Type placement:

- `io.github.pho001.synaptik.nn.module.Module` — the only owning type for child registration,
  recursive state traversal, and recursive mode propagation; no new facade or traversal utility
  is needed.
- `io.github.pho001.synaptik.nn.module.ModuleTest` — preserves NN 0001 direct-state regression
  coverage and adds shared-namespace behavior.
- `io.github.pho001.synaptik.nn.module.ModuleTreeTest` — isolates child ownership, traversal, and
  recursive-mode behavior without making a production test seam public.

## Affected files

Expected production file:

- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Module.java`.

Expected test files:

- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/module/ModuleTest.java`.
- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/module/ModuleTreeTest.java`.

Planning files synchronized by this task:

- `docs/planning/extensions/nn/master-plan.md`.
- this task specification.

## Maximum scope

This task may create or modify at most:

- one production Java file;
- two NN test files; and
- these two planning files.

If implementation needs a new public carrier type, another module/build dependency, an
architecture-test change, binding mutation, checkpoint format, or more files, stop and propose a
separate follow-up task.

## Acceptance criteria

- `child(name, child)` is protected, final, generic enough to return the concrete child type, and
  installs exactly one permanently owned child only after every validation succeeds.
- Parameters, buffers, and children occupy one shared direct namespace. Null and blank names,
  duplicate names across all three categories, null children, self-registration, reparenting,
  repeated attachment, and attempted ancestor cycles fail deterministically with no observable
  registry or ownership mutation.
- `children()` returns an insertion-ordered, unmodifiable snapshot of direct child name to child
  mappings. Existing direct `parameters()` and `buffers()` retain their direct-only declaration-
  order semantics and unmodifiable snapshot behavior.
- `parametersRecursively()` and `buffersRecursively()` return independent insertion-ordered,
  unmodifiable snapshots from the receiving root. They use the exact existing `Parameter`/
  `Buffer` instances and the specified depth-first dot-path order.
- `train()` and `eval()` set every owned descendant, including the receiver, to the requested
  `ForwardMode`; previously obtained `ForwardContext` values remain unchanged. A defensive
  repeated-identity preflight fails before any requested mode assignment.
- No generic `Module.forward(...)` method, Tensor subtype/mutation, binding replacement,
  checkpoint, optimizer, execution API, Gradle/dependency change, or architecture change is
  introduced.
- Focused NN tests observe every contract above. Existing NN 0001 tests remain valid except for
  deliberate assertions updated to reflect the now-shared child namespace.
- Every changed public `Module` member has complete meaningful Javadoc, including parameter,
  return, failure, ordering, snapshot, ownership, and thread-safety semantics where relevant.
  A separate documentation-focused agent finalizes Javadoc, planning wording, glossary impact,
  links, and generated Javadoc in the same overall change.

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

The documentation pass verifies the task/master-plan links and anchors, Javadoc pages, package
placement, terminology against the glossary, and that no unlisted CPU or global-roadmap changes
entered this task's diff. It reuses the implementation test result unless executable Java changes
after that result.

## Dependencies

- NN 0001 is Complete.
- Existing model Tensor contracts and ADR 0007 remain accepted.
- The user-authorized NN parallel exception recorded in the NN master plan remains in force.

## Follow-up tasks

- NN 0003: controlled parameter and buffer binding replacement, including its validation and
  concurrency/checkpoint interaction.
- A later checkpoint/state-dictionary task: persistence over the stable recursive maps; do not
  infer a file format here.
- NN 0004: explicit eager initializers, independently of tree traversal.
- NN 0005: `Linear` over the stabilized ownership and initializer contracts.

## Architecture impact

Expected impact: None.

This task realizes the existing ADR 0007 responsibility that `Module` declares and traverses
state. It changes no module edge or dependency rule; therefore architecture-test and architecture
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
  {@code .}, which is reserved exclusively as the recursive path separator.
- Child ownership is exclusive and permanent. A module can have zero or one parent and a child
  cannot be reused, removed, renamed, or reparented in this task.
- Recursive maps, not a new public state-entry type, carry qualified names while preserving the
  existing `Parameter` and `Buffer` APIs. The maps are defensive insertion-ordered snapshots.
- State traversal is depth-first with direct parameters before direct buffers before descendants.
- Recursive mode changes preflight identity uniqueness before setting a mode, yielding all-or-
  nothing requested-mode propagation even for a defensively detected malformed tree.

## Known limitations

- The API is not thread-safe. Concurrent declarations, traversals, or mode changes require caller
  synchronization; this task does not introduce locks or a concurrent snapshot contract.
- Parent ownership is intentionally internal to `Module`; there is no public parent lookup or
  mutable tree-management API.
- No public construction path can create a cycle or shared child. The defensive preflight exists
  only to retain fail-fast, non-partial behavior if an invariant is corrupted outside the API.
- Recursive state maps are discovery-only. They neither replace bindings nor serialize or restore
  values.

## Validation evidence

Implementation pass: `./gradlew :extensions:nn:test` passed after the dot-name correction on
2026-08-13 (5 actionable tasks: 3 executed, 2 up-to-date). Documentation review identified that
permitting `.` in a local name could make distinct child structures yield the same recursive path;
the implementation now rejects `.` at every declaration entry point. The focused suite covers
direct and recursive insertion order, dot-qualified paths, reserved-separator rejection, shared
namespace collisions, invalid child registration rollback, snapshot isolation, recursive mode
propagation, and reflective repeated-identity all-or-nothing failure.

Implementation also ran `git diff --check` after the correction; it passed with no output.

Architecture-test execution is intentionally not repeated: this task changes neither the Gradle
graph nor a dependency boundary, and the task validation section records that existing enforcement
covers the unchanged model-only edge. Repository-wide validation remains deferred to the NN
capability checkpoint or CI.

Documentation-focused pass `/root/nn_0002_docs` independently reviewed the final combined NN
0002 diff against `AGENTS.md`, `ARCHITECTURE.md`, ADR 0007, training-graph and dependency-rule
explanations, the NN 0001/0002 records, the current NN public source, and focused tests. It used
the General, API/Javadoc, Planning, and Example documentation profiles. It found and required the
dot-name repair before completion because `.` is the recursive-path separator; after the repair,
all declaration entry points reject it and the focused implementation suite was rerun successfully.

The pass finalized `Module` Javadoc and updated the NN glossary entry for permanent tree
ownership, shared local namespace, reserved dot separator, deterministic immutable recursive
snapshots, and preflight-atomic mode propagation. It made no architecture-document, ADR,
training-graph, Tensor API, or architecture-test change: NN 0002 realizes the accepted existing
NN ownership boundary, retains the existing model-only dependency, and introduces no Tensor,
compiler, training, or dependency-rule contract. It also made no example change because the
glossary's `encoder.layer1.weight` path is descriptive and agrees with the tested API.

Documentation validation: `./gradlew :extensions:nn:javadoc` passed on 2026-08-13 (3 actionable
tasks: 1 executed, 2 up-to-date); generated `Module` Javadoc was inspected for the changed
ownership, path, snapshot, failure, and thread-safety contracts. A targeted local Markdown
file/anchor and fence check of the changed NN planning files and glossary passed. `git diff --check`
passed with no output. The implementation's final `./gradlew :extensions:nn:test` evidence above
was reused; this documentation pass made no executable Java change.

## Implementation notes

`Module` now owns a permanently attached child tree with one shared direct namespace, direct and
recursive immutable discovery snapshots, and identity-preflighted recursive mode propagation.
Local names reserve `.` as the path separator, preventing recursive-path collisions. The
implementation changed only the planned NN production and test files; no dependency,
architecture, CPU, Engine, compiler, runtime, prepare, backend, binding-replacement, checkpoint,
optimizer, layer, or execution work was added.

## Completion summary

- Completed changes: Added permanent exclusive child ownership, collision-free shared local names,
  deterministic immutable recursive parameter/buffer discovery, and preflight-atomic recursive
  train/eval propagation to `Module`.
- Files changed or created: `Module.java`, `ModuleTest.java`, `ModuleTreeTest.java`, the NN master
  plan, this task record, and the NN glossary entry.
- Tests and validation: Implementation `./gradlew :extensions:nn:test` passed after the final
  dot-name repair (5 actionable tasks: 3 executed, 2 up-to-date); implementation `git diff --check`
  passed. Documentation `./gradlew :extensions:nn:javadoc` passed (3 actionable tasks: 1
  executed, 2 up-to-date); changed Markdown links, anchors, fences, generated Javadoc, and final
  whitespace were checked.
- Documentation-agent review: `/root/nn_0002_docs` independently finalized the API/Javadoc and
  planning documentation under the General, API/Javadoc, Planning, and Example profiles.
- Documentation impact: Updated the glossary and NN planning status; architecture docs, ADR 0007,
  dependency rules, training graph, Tensor API, and architecture tests require no change because
  no architecture or dependency boundary changed.
- Javadoc review: `Module` documents local-name separator reservation, exclusive ownership,
  deterministic traversal, immutable snapshots, preflight-atomic propagation, and non-thread-safe
  caller synchronization.
- Glossary impact: Updated the existing NN entry; no new reusable term was introduced.
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
