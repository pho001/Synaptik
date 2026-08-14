# Task 0010: State Dictionary and Checkpoint Contract

## Status

Complete

## Goal

Add one deterministic in-memory state-dictionary contract for a complete `Module` tree. A module
exports an immutable structural snapshot of its parameter and buffer bindings, and strictly loads
another dictionary only after validating the complete path, state-kind, data-type, Shape, and
parameter-gradient schema. An ordinary validation failure changes no binding.

Mental model:

```text
owned Module tree
  -> deterministic parameter/buffer traversal
  -> immutable StateDictionary of exact Tensor references

candidate StateDictionary + target Module tree
  -> collect complete target structure
  -> validate every path and compatible binding
  -> install exact Tensor references in target traversal order
```

This task calls the in-memory value a state dictionary. It establishes the object boundary that a
future checkpoint codec may consume, but it does not define a checkpoint file, byte format,
schema version, storage provider, or serialization API.

## Scope

- Add one public immutable `StateDictionary` value containing an ordered immutable list of
  `StateEntry` values. Its public constructor accepts a list so callers and future adapters can
  assemble a candidate dictionary without gaining direct buffer-mutation access.
- Add one public immutable `StateEntry` value containing exactly a relative qualified `path`, a
  `StateKind`, and the exact current `Tensor` reference. The Tensor's immutable descriptor supplies
  the entry's data type, structural Shape, and gradient-eligibility fact; do not duplicate those
  facts in another schema carrier.
- Add public `StateKind` with exactly `PARAMETER` and `BUFFER`. Kind is part of the state schema;
  an entry cannot satisfy a target of the other kind merely because its Tensor is compatible.
- Add exactly these public final methods to `Module`:

  ```java
  public final StateDictionary stateDictionary()
  public final void loadStateDictionary(StateDictionary dictionary)
  ```

- Export state in the existing deterministic depth-first module order. Each visited module emits
  its direct parameters in declaration order, then its direct buffers in declaration order, then
  its child subtrees in child-registration order. Paths are relative to the receiving module and
  use the existing reserved `.` separator.
- Make export a value snapshot rather than another wrapper-discovery snapshot. Each entry retains
  the exact `Tensor` returned by its wrapper at export time. A later successful replacement does
  not change an earlier dictionary, while an earlier parameter/buffer discovery map continues to
  retain wrappers that observe the new binding.
- Make a `StateDictionary` structurally immutable. Defensively copy the supplied entry list,
  preserve its encounter order, reject null entries, and reject duplicate paths before the value
  is constructed. Do not copy, evaluate, materialize, detach, or mutate an entry Tensor.
- Validate `StateEntry.path` against the existing recursive-path grammar: it is non-null and
  non-blank, has one or more non-blank local-name segments, and has no leading, trailing, or
  consecutive `.` separator. Preserve the exact accepted text; do not trim or normalize names.
- Make load strict. The candidate must contain exactly one entry for every target parameter and
  buffer and no other path. Reordered candidate entries are accepted because path, not list
  position, identifies state. The target's traversal order controls compatibility validation and
  installation.
- For every matched target, validate kind first, then exact `DataType`, then structural
  `Shape.equals`. For a parameter also require incoming `requiresGrad == true`, matching the
  permanent parameter contract. Do not compare or freeze layout, host storage, provenance, label,
  or Tensor identity.
- For a buffer, compare incoming data type and structural Shape with the target buffer's exact
  current binding at load start. Do not compare gradient eligibility: `Buffer` is excluded from
  optimizer discovery by its wrapper kind, and current `BatchNorm` may validly bind a
  gradient-eligible next-statistic expression. This task does not retrofit a declaration-time
  schema or public update API onto `Buffer`.
- Complete all ordinary validation before changing any binding. After successful validation,
  install entries in target traversal order through the already owning wrapper operations. The
  prevalidated parameter call and non-null package-owned buffer assignment are non-throwing by
  construction under the current immutable descriptor contracts, so no rollback log or public
  unchecked mutation primitive is needed.
- Preserve exact `Parameter` and `Buffer` wrapper identities, local names, recursive paths,
  declaration order, child ownership, and mode. Tensors and expressions obtained before load stay
  unchanged; only later wrapper reads and later forward construction observe installed bindings.
- Support empty modules. Export returns an empty dictionary; loading another empty dictionary
  succeeds without side effects; loading any non-empty dictionary into an empty module fails as
  unexpected state.
- Preserve the existing explicit-stack traversal and repeated-module-identity defense. Export or
  load of a malformed shared/cyclic tree fails before returning a snapshot or installing state.
- Document that `Module`, state export, and load remain non-thread-safe. Atomic load means that
  ordinary schema validation finishes before any binding changes in caller-coordinated use. It is
  not a linearizable operation, a lock, a simultaneous snapshot visible to racing readers, or a
  Java-memory-model visibility guarantee.
- Add focused API, ordering, immutability, schema, strict-load, successful-install, atomic-failure,
  empty/deep/malformed-tree, wrapper/expression, and boundary tests.
- Finalize affected Javadocs, module-package documentation, the Training API state boundary,
  glossary terminology, and planning evidence in the required separate documentation-focused
  clean context before implementation becomes Complete.

## Out of scope

- Bytes, files, streams, paths, channels, codecs, parsers, writers, repositories, stores, object
  serialization, JSON, protocol buffers, archives, compression, encryption, checksums, manifests,
  schema-version numbers, migrations, compatibility negotiation, or atomic file publication.
- A `Checkpoint` production type, checkpoint facade, load/save service, generic manager, registry,
  plugin, reflective discovery, annotation scanning, or hidden global state.
- Tensor-value evaluation, host-storage reads or copies, device transfer, materialization,
  detachment, cloning, casting, reshaping, layout conversion, backend storage, or execution.
- Optimizer state, gradients, gradient publication, parameter groups, random-source state,
  caller-threaded `GraphRngState`, forward mode, training session/step state, compiler artifacts,
  prepared execution, runtime state, or backend state.
- Partial, permissive, best-effort, path-prefix, parameter-only, buffer-only, recursive-subtree,
  rename/remap, ignore-missing, ignore-unexpected, conversion, or load-report modes.
- A separate schema descriptor, entry builder, dictionary builder, loader/report/result type,
  options object, format identifier, version field, map facade, or mutable dictionary API.
- Public buffer replacement, public/protected unchecked parameter replacement, arbitrary wrapper
  mutation, a protected batch primitive for subclasses, rollback callbacks, transaction receipts,
  version counters, compare-and-set, locks, synchronization, or concurrent snapshot guarantees.
- Changing `Parameter.replace(Tensor)`, protected direct `Module` replacement, `Buffer`'s direct
  transition semantics, `BatchNorm`'s sequential forward transition, or any existing layer API.
- New Tensor, Shape, DataType, operation, compiler, runtime, prepare, Engine, backend, optimizer,
  training, conformance, or integration behavior.
- Task 0011's unary composition and `Sequential` work or a detailed task-0011 specification.
- Gradle/dependency, architecture-contract, ADR, architecture-test, global-roadmap, CPU, legacy, or
  unrelated refactoring changes.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Training API](../../../../api/training-api.md)
- [Planning guide](../../../planning-guide.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [NN master plan](../master-plan.md)
- [Task 0002: Module-tree ownership and recursive mode propagation](0002-module-tree-ownership-and-recursive-mode-propagation.md)
- [Task 0003: Validated parameter and buffer binding replacement](0003-validated-parameter-and-buffer-binding-replacement.md)
- [Task 0004A: Parameter update and traversal hardening](0004a-parameter-update-and-traversal-hardening.md)
- [Task 0008: Batch normalization layer](0008-batch-normalization-layer.md)
- [Task 0009: Dropout layer](0009-dropout-layer.md)

## Architecture constraints

- `extensions/nn` owns module-declared parameters, persistent buffers, module-tree traversal, and
  this in-memory state transfer. It continues to depend only on `modules/model`.
- NN must not import Training, Compiler, Runtime, Prepare, Engine, CPU, Metal, CUDA, or another
  backend. Training remains downstream and may later coordinate when state is saved or loaded.
- Tensor identity, descriptor, and provenance remain immutable. Export and load retain exact
  Tensor references and change only the current binding held by stable NN wrappers.
- Parameter schema remains owned by `Parameter`: exact declaration-time data type, structural
  Shape, and `requiresGrad == true`. State load must not create a competing or weaker parameter
  validation rule.
- `Buffer` remains optimizer-excluded persistent module state without a public replacement API or
  declaration-time schema. State load may use the target's current data type and Shape as the
  strict target schema without changing protected layer-owned transition behavior.
- The existing collision-free module namespace and qualified paths are authoritative. State load
  identifies targets only through complete relative paths and kind, never reflection, field names,
  class names, wrapper equality, or positional list matching.
- The strict public load method is the only new mutation surface. It validates a complete owned
  tree and does not expose a reusable unchecked assignment primitive to callers or subclasses.
- In-memory atomicity does not require rollback because every ordinary caller-visible failure is
  discovered before installation and current post-validation assignments cannot reject. If that
  premise becomes false during implementation, stop and report the conflict instead of adding a
  hidden bypass or partial rollback design.
- Planning is non-authoritative. If implementation reveals a conflict with `ARCHITECTURE.md`, a
  need for another module dependency, or a need to define persistent checkpoint transport, stop
  before editing and request an explicit decision.

## Public API

All new public/protected surface is exact:

| Type | Signature | Visibility | Contract |
|---|---|---|---|
| `StateKind` | `PARAMETER`, `BUFFER` | `public enum` | Distinguishes trainable and persistent non-trainable binding roles; enum names are in-memory Java identities, not wire tokens. |
| `StateEntry` | `StateEntry(String path, StateKind kind, Tensor value)` | `public record` canonical constructor | Validates path, kind, and value in component order and retains the exact Tensor reference. |
| `StateEntry` | `path()`, `kind()`, `value()` | generated public accessors | Return the exact immutable components; `value()` performs no copy or evaluation. |
| `StateDictionary` | `StateDictionary(List<StateEntry> entries)` | `public record` canonical constructor | Defensively copies in encounter order and rejects null entries or duplicate paths. |
| `StateDictionary` | `entries()` | generated public accessor | Returns the unmodifiable ordered structural snapshot; never `null`. |
| `Module` | `StateDictionary stateDictionary()` | `public final` | Captures exact current Tensor references in deterministic combined parameter/buffer tree order. |
| `Module` | `void loadStateDictionary(StateDictionary dictionary)` | `public final` | Strictly validates the complete target tree before installing exact candidate Tensor references. |

Records retain their standard generated value equality, hashing, and diagnostic `toString`.
Those methods and enum names are not serialization or persistent compatibility contracts. No
other public/protected constructor, method, field, nested type, overload, option, or mutation
surface is added.

## State model, ordering, and ownership

The state dictionary is an immutable shallow snapshot:

- the entry list and each entry's path/kind association are immutable;
- the exported list order is root direct parameters, root direct buffers, then each child subtree
  recursively in registration order;
- each Tensor reference is retained exactly at export or candidate-entry construction time;
- Tensor descriptor and provenance remain immutable under Model ownership, while the Tensor's
  separately mutable borrowed host-storage association keeps its existing Model lifecycle;
- no wrapper, module, parent link, mode, random state, optimizer state, or historical value is
  retained by an entry; and
- an exported dictionary can be loaded into another independently constructed module tree only if
  the complete strict target schema matches.

For example, a root with parameter `weight`, buffer `step`, and child `encoder` whose direct state
is parameter `scale` then buffer `runningMean` exports:

```text
weight                    PARAMETER
step                      BUFFER
encoder.scale             PARAMETER
encoder.runningMean       BUFFER
```

The order is observable and deterministic, but load remains path-keyed. A caller-constructed
dictionary containing the same four unique entries in another order is compatible and installs in
the target order above.

`StateEntry` does not duplicate `DataType`, `Shape`, or `requiresGrad` components. Its exact Tensor
has one immutable `TensorDescriptor`, so duplicating schema would create mismatch states without
adding information. Layout is intentionally omitted from load compatibility because existing
parameter replacement permits layout changes and a buffer binding may move between resolved and
unresolved expressions. Storage, label, provenance, and identity are likewise payload facts, not
binding schema.

## Construction validation

### `StateEntry`

1. Reject null `path` with `NullPointerException("path")`.
2. Reject a blank path or any empty/blank dot-separated segment with
   `IllegalArgumentException`. Preserve accepted path text exactly.
3. Reject null `kind` with `NullPointerException("kind")`.
4. Reject null `value` with `NullPointerException("value")`.
5. Retain all three exact values without Tensor inspection, allocation, evaluation, or mutation.

Path validation accepts exactly names that could be produced by the current Module namespace. It
does not test that a target module currently contains the path.

### `StateDictionary`

1. Reject null `entries` with `NullPointerException("entries")`.
2. Traverse the supplied list in encounter order. Reject a null entry with an index-bearing
   `NullPointerException` and reject the first repeated path with `IllegalArgumentException`.
3. Only after complete validation, retain `List.copyOf(entries)` as the ordered immutable
   structural snapshot.

Duplicate rejection belongs at dictionary construction, before a value can be supplied to a
module. Because `StateEntry` is already valid, dictionary construction creates no Tensor or module
side effect.

## Strict load validation and atomic installation

`loadStateDictionary` performs these phases in exact order:

1. Reject null `dictionary` with `NullPointerException("dictionary")`.
2. Traverse the complete target tree once with the existing explicit-stack, repeated-identity
   defense. Capture each target's qualified path, kind, exact wrapper, and exact current Tensor at
   load start. No binding changes during this phase.
3. Index the already unique candidate entries by exact path.
4. In target traversal order, reject the first target path absent from the candidate as missing.
5. In candidate encounter order, reject the first path absent from the target as unexpected.
6. In target traversal order, validate each matched entry:
   - exact `StateKind`;
   - exact `DataType` identity;
   - structural `Shape.equals`; and
   - for a parameter only, incoming `requiresGrad == true`.
7. After every target validates, traverse the prepared target/entry pairs in target order. Call
   the existing final `Parameter.replace` for a parameter and the existing package-owned
   `Buffer.replaceValue` for a buffer. Both receive the exact non-null candidate Tensor.
8. Return normally after all assignments. Do not change mode, state order, ownership, or a Tensor.

Missing state takes precedence over unexpected state so a target's required schema is reported
first; kind then data type then Shape then parameter gradient eligibility is the per-path
compatibility order. Failures use `IllegalArgumentException` with the category and exact path plus
expected/actual facts where applicable. Message prose is diagnostic and not a wire contract.

All lookup and compatibility failures occur before step 7 and therefore leave every exact old
binding current. Step 7 is non-throwing by construction for ordinary program behavior: parameters
have already passed the same immutable descriptor checks used by `replace`, and buffers require
only a non-null exact reference. No new unchecked wrapper method, rollback map, catch-and-restore
path, or partially successful result exists.

This atomicity assumes the caller externally excludes concurrent declaration, ownership
corruption, replacement, load, mode changes, and forward construction. The sequential assignments
are not promised to appear simultaneously to a racing reader. JVM-fatal conditions such as
resource exhaustion are outside the ordinary validation atomicity guarantee and do not justify a
public rollback protocol.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.nn.module` — owns module state, paths, dictionary values, strict
  validation, and installation.
- `io.github.pho001.synaptik.model.tensor`, `.datatype`, and `.shape` — supply unchanged immutable
  Tensor descriptor facts.

No package is added. Type placement:

- `io.github.pho001.synaptik.nn.module.StateKind` — binding-role vocabulary colocated with
  `Parameter` and `Buffer`.
- `io.github.pho001.synaptik.nn.module.StateEntry` — one qualified in-memory binding snapshot.
- `io.github.pho001.synaptik.nn.module.StateDictionary` — immutable ordered aggregate and duplicate
  boundary.
- `io.github.pho001.synaptik.nn.module.Module` — sole owner of tree export, strict validation, and
  multi-binding installation.
- `io.github.pho001.synaptik.nn.module.StateDictionaryTest` — focused same-package contract tests,
  including package-owned buffer installation effects without a public test seam.

## Affected files

Expected production files:

- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Module.java`.
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/package-info.java` (new).
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/StateKind.java` (new).
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/StateEntry.java` (new).
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/StateDictionary.java` (new).

Expected test file:

- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/module/StateDictionaryTest.java` (new).

Expected documentation and planning files:

- `docs/api/training-api.md` — replace the no-checkpoint/no-batch statement with the exact current
  in-memory dictionary and strict load boundary while keeping optimizer and persistent formats
  planned.
- `docs/glossary.md` — update the existing NN module/parameter/buffer entry and add a reusable state
  dictionary/checkpoint distinction.
- `docs/planning/extensions/nn/master-plan.md`.
- this task specification.

## Maximum scope

Implementation may create or modify exactly the ten paths listed above: five production Java
files, one focused NN test file, two explanatory documentation files, and two planning files. If
implementation needs another test owner, `Parameter.java`, `Buffer.java`, a layer edit, a second
schema/report type, a public buffer method, a codec/format, a dependency/build/architecture change,
or an eleventh path, stop and propose a focused follow-up rather than expanding this task.

## Acceptance criteria

- The exact public API table is implemented with three final immutable public state types and only
  the two new public final Module methods. No generic loader, manager, service, builder, options,
  result, report, map facade, format, version, or public/protected unchecked mutation appears.
- `StateEntry` validates and retains exactly path, kind, and Tensor. Path grammar matches current
  relative module paths, accepted text is not normalized, and every failure has no Tensor/module
  side effect.
- `StateDictionary` preserves supplied encounter order in an unmodifiable defensive copy and
  rejects null entries and the first duplicate exact path. Empty dictionaries are valid.
- Export proves exact combined ordering across root/direct state and multiple nested children,
  exact qualified paths/kinds/Tensor references, stable output for an empty module, deep-tree
  traversal without call-stack dependence, and repeated-identity failure before a result.
- An exported dictionary is a value snapshot: later parameter or buffer replacement leaves its
  entries unchanged. Existing discovery snapshots retain stable wrappers that observe successful
  load, and Tensors/expressions captured before load remain unchanged.
- Strict load accepts candidate list reordering but requires the exact complete path set. Tests
  cover null dictionary, missing path, unexpected path, duplicate construction, and parameter/
  buffer kind mismatch with no target mutation.
- Compatibility tests cover exact data-type mismatch, structural Shape mismatch, parameter
  `requiresGrad == false`, and atomic failure at a later target after earlier targets are valid.
  Every ordinary failure leaves all parameter and buffer bindings exactly unchanged.
- Compatibility accepts different Tensor identity, label, layout, storage presence, and
  provenance when kind/data type/Shape and parameter gradient eligibility match. Buffer gradient
  eligibility may differ, preserving current BatchNorm state semantics.
- Successful load retains every exact candidate Tensor, installs in target traversal order,
  preserves all wrapper/name/path/order/child/mode identities, and causes no Tensor creation,
  copy, evaluation, mutation, identifier allocation, or expression rewrite.
- Empty-to-empty load succeeds; empty/non-empty mismatches fail strictly. Malformed repeated tree
  identity fails before installation. Existing parameter, buffer, traversal, mode, BatchNorm, and
  Dropout tests remain unchanged and pass.
- Javadocs and module package documentation define snapshot depth, ownership, schema, exact
  ordering, strict failures, side-effect order, atomicity meaning, threading limits, old-expression
  behavior, and persistence exclusions with complete `@param`, `@return`, and `@throws` tags.
- Training API and glossary distinguish implemented in-memory state transfer from future
  persistent checkpoints and continue to exclude optimizer/session/RNG/execution state.
- A separate clean-context documentation pass finalizes all affected Javadoc, explanatory text,
  glossary and planning evidence, generated Javadoc, Markdown, exact scope/status, and whitespace
  before the implementation is Complete.
- No architecture/ADR/test, Gradle/dependency, Model/Tensor/Compile API, Training implementation,
  layer, compiler/runtime/prepare/Engine/backend, conformance/integration, CPU, global-roadmap,
  task-0011 specification, legacy, or unrelated path enters the implementation diff.

## Tests / validation

Validation tier: task validation for the single affected `extensions/nn` module plus a targeted
documentation pass. This task changes no module edge, backend behavior, or end-to-end execution.

Implementation runs focused tests while developing:

```text
./gradlew :extensions:nn:test \
  --tests io.github.pho001.synaptik.nn.module.StateDictionaryTest
```

After executable Java and tests stabilize, run the affected module once as final Java evidence:

```text
./gradlew :extensions:nn:test
```

The separate documentation-focused pass reuses that successful evidence unless executable Java
changes afterward. After final Javadoc edits it runs:

```text
./gradlew :extensions:nn:javadoc
```

Final implementation/documentation validation also checks:

- reflection and `javap -public` for the exact three-type/two-Module-method public surface;
- production imports and the unchanged Model-only NN dependency;
- local Markdown paths, links, heading anchors, balanced fences, terminal newlines, and trailing
  whitespace in the four documentation/planning files;
- exactly the ten task-owned implementation paths;
- tasks 0001–0009 Complete, exactly task 0010 Ready/In progress/Complete as appropriate, task 0011
  a concise Draft row with no detailed specification, and no second Ready NN task;
- `git diff --check`; and
- `git diff --no-index --check /dev/null <path>` for every untracked new file.

Repository-wide, architecture, backend-conformance, integration, Model, Compiler, CPU, and root
tests are deferred to the NN capability checkpoint or CI. Existing architecture enforcement
already locks the unchanged Model-only dependency, and focused NN tests own the changed behavior.

## Dependencies

- NN tasks 0001–0009 are Complete.
- Tasks 0002 and 0004A provide collision-free deterministic paths, iterative repeated-identity-
  defended traversal, stable wrappers, and public schema-compatible parameter replacement.
- Task 0008 establishes a concrete persistent-buffer consumer whose next expressions may be
  gradient eligible without becoming optimizer targets.
- Model `Tensor`, `TensorDescriptor`, `DataType`, and `Shape` provide immutable exact schema facts.
- ADR 0007 and the existing architecture test provide the accepted `model -> nn -> training`
  dependency direction.
- No persistent storage, Engine, execution, compiler, backend, or optimizer prerequisite exists
  because this task transfers in-memory Tensor references only.

## Follow-up tasks

- NN 0011 remains Draft and separately owns unary Tensor composition and `Sequential`. It may
  consume ordinary modules but must not broaden or reinterpret state-dictionary loading.
- A future checkpoint persistence task may define a versioned codec/storage/file contract only
  after a concrete consumer and materialized-value lifecycle are known. It must consume this
  in-memory schema without treating enum names, record `toString`, Tensor provenance, or Java
  object serialization as a wire format. Do not create its detailed specification now.
- `extensions/training` may later coordinate state-dictionary timing with optimizer/session state,
  but optimizer state remains a separate training-owned contract and must not be inserted into
  this NN dictionary.

## Documentation and no-change review

Document profiles:

- Java/package Javadoc: General plus API/Javadoc.
- Training API: General plus API/Javadoc.
- glossary: General reference style.
- task/master plan: General plus Planning.

Required documentation changes are the five new/changed production Javadocs, the new module
package documentation, Training API state/checkpoint wording, the glossary distinction, and
synchronized NN planning records.

The completion summary must record these reasoned no-change conclusions:

- `ARCHITECTURE.md`, focused architecture documents, and ADR 0007 remain accurate because NN
  already owns module parameter/buffer state and dependency direction does not change.
- `docs/api/tensor-api.md`, `docs/api/compile-api.md`, Model source/plans/capabilities, and related
  Tensor/Shape/DataType Javadocs remain accurate because no model semantic, descriptor, capture,
  gradient, or execution behavior changes.
- `docs/architecture/training-graph.md` remains accurate because no optimizer, session, gradient
  publication, compiled update, or execution orchestration is added.
- `Parameter.java` remains accurate because its individual public replacement still provides no
  checkpoint or transaction; Module now owns the separate strict batch contract. `Buffer.java`
  remains accurate because it gains no public update or declaration schema.
- Concrete layer Javadocs/tests remain accurate: BatchNorm's forward transition is still
  sequential and non-atomic, while Dropout's caller-owned graph RNG state is absent from module
  state.
- Build files and architecture tests remain accurate because no dependency changes. Backend
  conformance and integration tests remain unnecessary because no numerical execution or
  end-to-end lifecycle changes.
- The global roadmap, CPU work, other modules, legacy branch, and task 0011 remain untouched.

## Architecture impact

Expected impact: None.

This task implements the architecture's existing NN ownership of parameters, persistent buffers,
and module-tree traversal. It changes no module owner, dependency direction, Tensor invariant, or
execution lifecycle. If implementation requires a persistent checkpoint owner, optimizer state,
another module edge, architecture rule, or unsafe public mutation seam, stop and report the exact
conflict instead of editing architecture within this task.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/developer-guide/documentation-rules.md, and
docs/planning/extensions/nn/tasks/0010-state-dictionary-and-checkpoint-contract.md in full.
Read the task's directly linked NN/module/Model/training contracts and final source/tests before
editing.

Implement task 0010 exactly as specified within its ten authorized paths. Do not use GSD, do not
implement task 0011 or persistent serialization, and do not commit or push. Preserve every
unrelated worktree change exactly. Stop and report any architecture uncertainty, maximum-scope
overflow, or failure of the specified validate-before-install design to remain non-throwing.

Run the focused StateDictionary test and one final NN module test after executable Java stabilizes.
Then hand the actual diff and exact test evidence to a separate clean documentation-focused
context. That pass follows the documentation rules and selected profiles, independently finalizes
Javadocs, module package documentation, Training API, glossary, planning evidence, and all
documentation/surface/scope/status/whitespace checks in the same overall change without repeating
successful Java tests unless executable behavior changes or a concrete risk is recorded.

Update this task's decisions, limitations, evidence, notes, completion summary, and final status
only after every implementation and documentation criterion passes.
```

## Documentation-agent handoff

After Java/tests stabilize, give the clean documentation context:

- this task and exact ten-path limit;
- the final diff and exact focused/final NN command results and counts;
- the public API table, path/order/schema/ownership decisions, strict validation order, atomicity
  boundary, Buffer gradient exception, wrapper/expression semantics, and persistence exclusions;
- the relevant architecture/ADR, documentation rules and profiles, final Module/Parameter/Buffer/
  layer source and tests, Model descriptor contracts, Training API, glossary, dependency test, and
  NN planning history;
- the mandate to finalize all affected documentation and reasoned no-change conclusions without
  changing executable behavior;
- generated-Javadoc, reflection/`javap`, imports/dependency, Markdown, exact-scope/status,
  whitespace/newline, and task-0011-absence gates; and
- the required completion-summary and `Status` format from `AGENTS.md`.

## Local decisions

- `Module` owns both export and strict load. It already owns the complete private tree registries
  and can install through existing wrapper operations, so another loader/service or widened
  protected mutation API would add indirection without a boundary.
- Use three small public immutable values: `StateKind`, `StateEntry`, and `StateDictionary`. A raw
  `Map<String, Tensor>` would lose parameter/buffer kind, while a separate schema/report hierarchy
  has no current consumer.
- Keep path in `StateEntry` and accept a list in `StateDictionary`. This preserves deterministic
  order and lets dictionary construction detect duplicate paths before load; a Map constructor
  could not observe duplicates.
- Make strictness unconditional. A report/result type and permissive options are unnecessary when
  the current requirement is exact all-or-nothing state installation.
- Reuse immutable Tensor descriptor facts rather than duplicate data type, Shape, or gradient
  fields in an entry. Load deliberately excludes layout, storage, provenance, label, and identity.
- Parameter compatibility is the existing permanent schema. Buffer compatibility uses the
  target's current data type and Shape and ignores gradient eligibility, preserving BatchNorm's
  valid gradient-eligible next-statistic bindings without inventing a universal buffer schema.
- Candidate list order is preserved but is not compatibility. Paths identify entries, while the
  target's established traversal order controls diagnostics and installation.
- Validate the complete tree before installing. Current immutable descriptors and existing final
  wrapper operations make the installation phase non-throwing for ordinary behavior, so no
  rollback or unchecked assignment primitive is justified.
- Keep persistent checkpoints deferred. A Tensor-reference snapshot says nothing about evaluated
  bytes, host/device materialization, schema versions, durability, or atomic file replacement.

## Known limitations

- The dictionary is shallow and in-memory. It retains Tensor objects, not portable values or a
  durable checkpoint, and Tensor host-storage lifetime remains governed by the existing Model
  contract.
- Buffer data type and Shape compatibility are relative to the target's current binding because
  generic Buffer has no declaration schema. Direct protected buffer transitions remain capable of
  changing that current schema outside this strict load operation.
- The contract is strict only; it cannot partially load, rename, convert, ignore, or report a set
  of mismatches.
- Export and load are not thread-safe or linearizable. Callers must coordinate them with module
  declaration, replacement, mode changes, and forward construction.
- Successful installation is sequential and offers no simultaneous view to racing readers.
- JVM-fatal failure is outside ordinary validate-before-install atomicity.
- Optimizer state, graph RNG state, mode, session state, and evaluated running-statistic bytes are
  not represented.

## Validation evidence

Planning context `/root/nn_0010_planning` read the repository instructions, architecture contract
and index, planning guide and roadmap, documentation rules and General/Planning/API-Javadoc
profiles, ADR 0007, NN master plan and completed task history, final Module/Parameter/Buffer and
layer source/tests, Model descriptor/Shape/data-type contracts, Training API and training-graph
boundary, glossary terminology, build edges, and NN/training architecture enforcement. A read-only
legacy tree scan found no selected module state-dictionary/checkpoint capability worth carrying
forward; no legacy architecture or source was used.

The planning pass selected the exact public API and validate-before-install algorithm recorded
above and found no architecture blocker. Current private Module registries plus existing final
wrapper operations are sufficient; no new public/protected unchecked batch primitive is required.

Planning validation passed:

- the repository-local link/anchor check resolved every local Markdown target in the two planning
  files;
- fence balance, terminal newlines, and trailing-whitespace checks passed;
- exactly one NN task file has `Ready` status and exactly one NN master-plan row is Ready;
- task 0011 remains a concise Draft row and has no detailed specification;
- exact changed-path inspection contains only the NN master plan and this new task;
- whole-worktree `git diff --check` passed; and
- `git diff --no-index --check /dev/null
  docs/planning/extensions/nn/tasks/0010-state-dictionary-and-checkpoint-contract.md` passed.

No Java test, Javadoc, architecture, repository, conformance, or integration command was run for
this planning-only change. Implementation and documentation evidence remain empty until their
separate task contexts complete.

Implementation context `/root/nn_0010_implementation` added the exact planned three public state
values, the two final Module methods, package documentation, and one focused contract suite within
the six authorized executable paths. The first focused compile exposed and corrected one test-only
parenthesis error before any test ran. The stabilized focused command
`./gradlew :extensions:nn:test --tests
io.github.pho001.synaptik.nn.module.StateDictionaryTest` then passed 1 suite and 15 tests with zero
failures, errors, or skips. It covers exact API/finality, entry/list validation and immutability,
combined deterministic export, shallow value snapshots, reordered strict load, validation order,
zero-mutation failures, Buffer gradient treatment, exact-reference installation, stable wrappers,
old expressions, empty and deep trees, and repeated-identity defense.

After executable Java stabilized, implementation context `/root/nn_0010_implementation` ran the
sole authoritative `./gradlew :extensions:nn:test` command. It passed 16 suites and 98 tests with
zero failures, errors, or skips. No executable Java or test changed afterward.

Preliminary implementation validation also passed:

- `./gradlew :extensions:nn:javadoc` completed successfully after one sandbox-denied Gradle-cache
  lock attempt was rerun with the already approved Gradle permission; the documentation context
  must run final Javadoc again after its edits.
- `javap -public` over `StateKind`, `StateEntry`, `StateDictionary`, and `Module` showed exactly the
  three planned final state types, record-generated value/accessor methods, and the two new final
  Module methods with no extra public state API.
- The focused reflection test independently locks finality, record components, enum order, and the
  two Module signatures.
- Production import and forbidden-vocabulary scans found only Model Tensor plus `java.util`
  dependencies and no Training, compiler, runtime, prepare, Engine, backend, serializer, codec,
  file-I/O, reflective-discovery, or checkpoint production surface.
- `extensions/nn/build.gradle.kts` remains the unchanged one-edge Model-only dependency.
- Exact changed/untracked-path inspection found only the eight currently applicable task-owned
  implementation/planning paths. Training API and glossary remain reserved for the documentation
  pass; no architecture, build, layer, other module, task-0011, or unrelated path changed.
- `git diff --check` passed. New-file whitespace checks produced no diagnostics; terminal-newline
  checks passed.

Independent documentation context `/root/nn_0010_docs` reviewed the final implementation and
tests against the architecture, task, preceding NN contracts, Model descriptor contracts, and
current training boundary. It found no executable, atomicity, architecture, or scope blocker and
changed no executable Java or tests. The context finalized only task-authorized Javadocs/package
documentation, Training API, glossary, and planning records.

The documentation pass selected the General, API/Javadoc, Planning, and Example profiles. It
finalized exact path/order/reference ownership, record construction and accessor semantics,
strict missing-before-unexpected and per-target validation order, target-current Buffer schema,
the intentional Buffer gradient exception, stable-wrapper installation, old-expression behavior,
caller-coordinated validate-before-install atomicity, threading/race limits, and no Tensor-copy/
execution boundary. The Training API and glossary now distinguish this shallow in-memory module
state from future persistent checkpoint codecs and separately owned optimizer/session/RNG state;
both include the same concise nested root/child ordering and strict-load example.

Final documentation validation:

- `./gradlew :extensions:nn:javadoc` passed after all Javadoc edits (`BUILD SUCCESSFUL`, three
  actionable tasks: two executed and one up-to-date). Generated `Module`, `StateDictionary`,
  `StateEntry`, `StateKind`, and module-package pages were inspected for ordering, exact-reference,
  validation, ownership, atomicity, concurrency, failure, and persistence/execution boundaries.
- `javap -public` over `StateKind`, `StateEntry`, `StateDictionary`, and `Module` showed exactly the
  planned enum constants, record constructors/components/standard members, and the two new final
  Module state methods. A standalone Java 26 reflection check independently passed finality,
  record-component order/types, enum order, exact Module signatures, and absence of another
  public Module state method.
- Two preliminary JShell reflection attempts printed the expected finality/components and exact
  Module state-method assertions, but JShell then reported a sandboxed macOS Preferences flush
  failure on exit. The successful standalone compiled reflection check supersedes that
  environment-only cleanup failure.
- A targeted repository-local Markdown checker passed the Training API, glossary, NN master plan,
  and this task: four files and 345 local links, with every target/anchor resolved, balanced
  fences, terminal newlines, and no trailing whitespace. Its first draft incorrectly collapsed
  consecutive spaces while modeling GitHub heading slugs and reported false missing glossary
  anchors; correcting that rule produced the passing result without a documentation edit. A
  compact final rerun initially used an unavailable `Enumerator#filter_map` on the installed Ruby
  and failed before checking content; replacing it with an equivalent supported loop produced the
  same passing 345-link result.
- Production import and Gradle inspection found only Model and JDK dependencies and the unchanged
  sole `implementation(project(":modules:model"))` edge. Exact scope contains the five production
  paths, focused test, Training API, glossary, NN master plan, and this task: exactly ten paths.
  Concurrent CPU master-plan and task-planning paths were separately visible in the shared
  worktree and were preserved unchanged by this context.
- Status inspection found NN 0001–0010 Complete, task 0011 a concise Draft row with no detailed
  specification, and no Ready NN task. Final tracked/untracked newline/whitespace checks,
  `git diff --no-index --check` for every new task path, and whole-worktree `git diff --check`
  passed.

The documentation context reused the implementation context's final focused one-suite/15-test and
authoritative NN 16-suite/98-test results because only Javadoc comments and Markdown changed after
those runs. No Java test suite was repeated.

## Implementation notes

- Added `StateKind`, `StateEntry`, and `StateDictionary` as final immutable public values in the
  existing module-ownership package. Constructors preserve exact accepted values, validate in the
  specified order, and retain no mutable caller list.
- Added one unified explicit-stack state-binding snapshot in `Module`. Export converts that stable
  target snapshot to exact Tensor entries; load captures it once, validates missing, unexpected,
  kind, data type, Shape, and parameter gradient eligibility before installing in target order.
- Kept installation inside existing owning operations: `Parameter.replace` and package-owned
  `Buffer.replaceValue`. No public buffer mutation, unchecked assignment, loader/service, codec,
  persistent format, optimizer state, execution behavior, dependency, or architecture change was
  introduced.
- Added the focused same-package state-dictionary suite. Existing executable Java and tests were
  otherwise unchanged.

## Completion summary

- Completed changes: Added the exact immutable state values, deterministic combined Module export,
  strict complete validate-before-install load, focused coverage, finalized public/package
  Javadocs, Training API boundary, glossary definition/example, and synchronized planning records.
- Files changed or created: Exactly the five production Java paths, one focused test path,
  Training API, glossary, NN master plan, and this task record.
- Tests and validation: Reused the stabilized focused one-suite/15-test and authoritative NN
  16-suite/98-test results, both with zero failures, errors, or skips. Final Javadoc/generated-page,
  `javap`, independent reflection, Markdown, imports/dependency, exact ten-path scope, status,
  newline, trailing-whitespace, no-index, and `git diff --check` gates passed.
- Documentation-agent review: Independent clean context `/root/nn_0010_docs` completed the
  mandatory General/API-Javadoc/Planning/Example review without changing executable behavior.
- Documentation impact: Training API and glossary now define the implemented shallow in-memory
  state dictionary and distinguish it from future persistent checkpoints and separate optimizer,
  session, graph-RNG, compiler, runtime, and backend state.
- Javadoc review: Finalized all affected type, constructor, component/accessor, Module-method, and
  package contracts; final generated Javadoc passed and was inspected.
- Glossary impact: Updated the existing NN module/parameter/buffer entry and added reusable state
  dictionary/checkpoint terminology plus a concise nested ordering/load example.
- Unresolved issues: None.
- Follow-up required: None for task 0010. Task 0011 remains Draft without a detailed specification.

Status: Complete
