# Task 0010: State Dictionary and [Checkpoint](../../../../glossary.md#term-checkpoint) Contract

> **Editorial reading context — not a status update.** The application asks Module to export
> exact current Tensor references under complete state paths. The dictionary copies its list,
> not the Tensor values or their backing storage. Loading first validates every target path
> and schema, then installs exact references; it does not calculate numbers or create a file.
> “Atomic” here concerns validation-before-install under application coordination, not a
> simultaneous view for racing readers. Compare today's [state-dictionary contract](../../../../reference/nn/state.md#current-nn-state-dictionary-contract)
> with [values-only checkpoint capture](../../../../reference/checkpoint/values.md#values-only-recovery):
> the latter separately materializes values and validates a persistent schema.

## Status

Complete

## Goal

Add one deterministic in-memory state-dictionary contract for a complete [`Module`](../../../../glossary.md#term-module) tree. A module
exports an [immutable](../../../../glossary.md#term-immutable) structural [snapshot](../../../../glossary.md#term-snapshot) of its [parameter](../../../../glossary.md#term-parameter) and [buffer](../../../../glossary.md#term-buffer) [bindings](../../../../glossary.md#term-binding), and strictly loads
another dictionary only after validating the complete path, state-kind, data-type, [Shape](../../../../glossary.md#shape), and
parameter-[gradient](../../../../glossary.md#term-gradient) [schema](../../../../glossary.md#term-schema). An ordinary [validation](../../../../glossary.md#term-validation) failure changes no binding.

[Mental model](../../../../glossary.md#term-mental-model):

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
future [checkpoint](../../../../glossary.md#term-checkpoint) [codec](../../../../glossary.md#term-codec) may consume, but it does not define a checkpoint file, byte format,
schema version, storage provider, or [serialization](../../../../glossary.md#term-serialization) [API](../../../../glossary.md#term-application-programming-interface-api).

## [Scope](../../../../glossary.md#term-scope)

- Add one public [immutable](../../../../glossary.md#term-immutable) `StateDictionary` value containing an ordered immutable list of
  `StateEntry` values. Its public constructor accepts a list so callers and future adapters can
  assemble a candidate dictionary without gaining direct buffer-mutation access.
- Add one public immutable `StateEntry` value containing exactly a relative qualified `path`, a
  `StateKind`, and the exact current [`Tensor`](../../../../glossary.md#tensor) reference. The Tensor's immutable descriptor supplies
  the entry's data type, structural Shape, and gradient-eligibility fact; do not duplicate those
  facts in another schema carrier.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [data type glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient); [schema glossary](../../../../glossary.md#term-schema); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Add public `StateKind` with exactly [`PARAMETER`](../../../../glossary.md#term-parameter) and [`BUFFER`](../../../../glossary.md#term-buffer). Kind is part of the state schema;
  an entry cannot satisfy a target of the other kind merely because its Tensor is compatible.
- Add exactly these public final methods to [`Module`](../../../../glossary.md#term-module):

  ```java
  public final StateDictionary stateDictionary()
  public final void loadStateDictionary(StateDictionary dictionary)
  ```

- Export state in the existing deterministic depth-first module order. Each visited module emits
  its direct parameters in declaration order, then its direct buffers in declaration order, then
  its child subtrees in child-registration order. Paths are relative to the receiving module and
  use the existing reserved `.` separator.
- Make export a value snapshot rather than another wrapper-discovery snapshot. Each entry retains
  the exact <span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>`Tensor` returned by its wrapper at export time. A later successful replacement does
  not change an earlier dictionary, while an earlier parameter/buffer discovery map continues to
  retain wrappers that observe the new binding.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- Make a `StateDictionary` structurally immutable. Defensively copy the supplied entry list,
  preserve its encounter order, reject null entries, and reject duplicate paths before the value
  is constructed. Do not copy, evaluate, materialize, detach, or mutate an entry Tensor.
- Validate `StateEntry.path` against the existing recursive-path grammar: it is non-null and
  non-blank, has one or more non-blank local-name segments, and has no leading, trailing, or
  consecutive `.` separator. Preserve the exact accepted text; do not trim or normalize names.
- Make load strict. The candidate must contain exactly one entry for every target parameter and
  buffer and no other path. Reordered candidate entries are accepted because path, not list
  position, identifies state. The target's traversal order controls compatibility validation and
  installation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- For every matched [target](../../../../glossary.md#term-target), validate kind first, then exact [`DataType`](../../../../glossary.md#data-type--datatype), then structural
  `Shape.equals`. For a parameter also require incoming `requiresGrad == true`, matching the
  permanent parameter contract. Do not compare or freeze layout, host storage, provenance, label,
  or Tensor identity.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [host storage glossary](../../../../glossary.md#host-storage); [provenance glossary](../../../../glossary.md#provenance); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- For a buffer, compare incoming data type and structural Shape with the target buffer's exact
  current binding at load start. Do not compare gradient eligibility: `Buffer` is excluded from
  optimizer discovery by its wrapper kind, and current <span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer))</span>`BatchNorm` may validly bind a
  gradient-eligible next-statistic expression. This task does not retrofit a declaration-time
  schema or public update API onto <span data-glossary-navigation="true"> ([statistic glossary](../../../../glossary.md#term-statistic); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>`Buffer`.
- Complete all ordinary validation before changing any binding. After successful validation,
  install entries in target traversal order through the already owning wrapper operations. The
  prevalidated parameter call and non-null package-owned buffer assignment are non-throwing by
  construction under the current immutable descriptor contracts, so no rollback log or public
  unchecked mutation primitive is needed.<span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback); [log glossary](../../../../glossary.md#term-log))</span>
- Preserve exact `Parameter` and `Buffer` wrapper identities, local names, recursive paths,
  declaration order, child ownership, and mode. Tensors and expressions obtained before load stay
  unchanged; only later wrapper reads and later forward construction observe installed bindings.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Support empty modules. Export returns an empty dictionary; loading another empty dictionary
  succeeds without side effects; loading any non-empty dictionary into an empty module fails as
  unexpected state.
- Preserve the existing explicit-stack traversal and repeated-module-identity defense. Export or
  load of a malformed shared/cyclic tree fails before returning a snapshot or installing state.
- Document that `Module`, state export, and load remain non-thread-safe. Atomic load means that
  ordinary schema validation finishes before any binding changes in caller-coordinated use. It is
  not a linearizable operation, a lock, a simultaneous snapshot visible to racing readers, or a
  Java-memory-model visibility guarantee.<span data-glossary-navigation="true"> ([thread-safe glossary](../../../../glossary.md#term-thread-safe); [Atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [operation glossary](../../../../glossary.md#operation); [model glossary](../../../../glossary.md#term-model))</span>
- Add focused API, ordering, immutability, schema, strict-load, successful-install, atomic-failure,
  empty/deep/malformed-tree, wrapper/expression, and boundary tests.
- Finalize affected Javadocs, module-package documentation, the Training API state boundary,
  glossary terminology, and planning evidence in the required separate documentation-focused
  clean context before implementation becomes Complete.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Bytes, files, streams, paths, channels, codecs, parsers, writers, repositories, stores, object
  serialization, JSON, protocol buffers, archives, compression, encryption, checksums, manifests,
  schema-version numbers, migrations, compatibility negotiation, or atomic file publication.<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization); [JSON glossary](../../../../glossary.md#term-json); [schema glossary](../../../../glossary.md#term-schema); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [publication glossary](../../../../glossary.md#term-publication))</span>
- A [`Checkpoint`](../../../../glossary.md#term-checkpoint) production type, checkpoint facade, load/save service, generic manager, registry,
  plugin, reflective discovery, annotation scanning, or hidden global state.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade); [registry glossary](../../../../glossary.md#term-registry); [plugin glossary](../../../../glossary.md#term-plugin))</span>
- Tensor-value evaluation, host-storage reads or copies, device transfer, materialization,
  detachment, cloning, casting, reshaping, layout conversion, backend storage, or execution.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [evaluation glossary](../../../../glossary.md#term-evaluation); [host glossary](../../../../glossary.md#term-host); [device glossary](../../../../glossary.md#term-device); [materialization glossary](../../../../glossary.md#materialization); [layout glossary](../../../../glossary.md#layout); [backend glossary](../../../../glossary.md#backend))</span>
- Optimizer state, gradients, gradient publication, parameter groups, random-source state,
  caller-threaded <span data-glossary-navigation="true"> ([Optimizer state glossary](../../../../glossary.md#term-optimizer-state); [gradients glossary](../../../../glossary.md#term-gradient); [parameter glossary](../../../../glossary.md#term-parameter))</span>[`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate), forward mode, training session/step state, compiler artifacts,
  prepared execution, runtime state, or backend state.<span data-glossary-navigation="true"> ([forward mode glossary](../../../../glossary.md#term-forward-mode); [training session glossary](../../../../glossary.md#training-session--trainingsession); [compiler glossary](../../../../glossary.md#term-compiler); [artifacts glossary](../../../../glossary.md#term-artifact); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Partial, permissive, best-effort, path-prefix, parameter-only, buffer-only, recursive-subtree,
  rename/remap, ignore-missing, ignore-unexpected, conversion, or load-report modes.
- A separate schema descriptor, entry builder, dictionary builder, loader/report/result type,
  options object, format identifier, version field, map facade, or mutable dictionary API.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- Public buffer replacement, public/protected unchecked parameter replacement, arbitrary wrapper
  mutation, a protected batch primitive for subclasses, rollback callbacks, transaction receipts,
  version counters, compare-and-set, locks, synchronization, or concurrent snapshot guarantees.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [rollback glossary](../../../../glossary.md#term-rollback); [transaction glossary](../../../../glossary.md#term-transaction); [compare-and-set glossary](../../../../glossary.md#term-compare-and-set-cas); [synchronization glossary](../../../../glossary.md#term-synchronization); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Changing `Parameter.replace(Tensor)`, protected direct [`Module`](../../../../glossary.md#term-module) replacement, [`Buffer`](../../../../glossary.md#term-buffer)'s direct
  transition semantics, `BatchNorm`'s sequential forward transition, or any existing layer [API](../../../../glossary.md#term-application-programming-interface-api).
- New Tensor, Shape, DataType, operation, compiler, runtime, prepare, Engine, backend, optimizer,
  training, conformance, or integration behavior.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [DataType glossary](../../../../glossary.md#data-type--datatype); [operation glossary](../../../../glossary.md#operation); [prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [optimizer glossary](../../../../glossary.md#term-optimizer); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Task 0011's [unary](../../../../glossary.md#term-unary) composition and `Sequential` work or a detailed task-0011 specification.
- Gradle/dependency, architecture-contract, ADR, architecture-test, global-roadmap, CPU, legacy, or
  unrelated refactoring changes.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [CPU glossary](../../../../glossary.md#term-cpu))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [Training API](../../../../api/training-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [NN master plan](../master-plan.md)<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- [Task 0002: Module-tree ownership and recursive mode propagation](0002-module-tree-ownership-and-recursive-mode-propagation.md)<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- [Task 0003: Validated parameter and buffer binding replacement](0003-validated-parameter-and-buffer-binding-replacement.md)<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [buffer glossary](../../../../glossary.md#term-buffer); [binding glossary](../../../../glossary.md#term-binding))</span>
- [Task 0004A: Parameter update and traversal hardening](0004a-parameter-update-and-traversal-hardening.md)
- [Task 0008: Batch normalization layer](0008-batch-normalization-layer.md)<span data-glossary-navigation="true"> ([Batch normalization glossary](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training))</span>
- [Task 0009: Dropout layer](0009-dropout-layer.md)<span data-glossary-navigation="true"> ([Dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>

## Architecture constraints

- `extensions/nn` owns module-declared parameters, persistent buffers, module-tree traversal, and
  this in-memory state transfer. It continues to depend only on <span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>`modules/model`.
- NN must not import Training, Compiler, Runtime, Prepare, Engine, CPU, Metal, CUDA, or another
  backend. Training remains downstream and may later coordinate when state is saved or loaded.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Compiler glossary](../../../../glossary.md#term-compiler); [Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal); [CUDA glossary](../../../../glossary.md#term-cuda); [backend glossary](../../../../glossary.md#backend); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Tensor identity, descriptor, and provenance remain immutable. Export and load retain exact
  Tensor references and change only the current binding held by stable NN wrappers.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [identity glossary](../../../../glossary.md#term-identity-meanings); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [immutable glossary](../../../../glossary.md#term-immutable); [binding glossary](../../../../glossary.md#term-binding))</span>
- [Parameter](../../../../glossary.md#term-parameter) [schema](../../../../glossary.md#term-schema) remains owned by `Parameter`: exact declaration-time data type, structural
  Shape, and <span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape))</span>`requiresGrad == true`. State load must not create a competing or weaker parameter
  validation rule.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- [`Buffer`](../../../../glossary.md#term-buffer) remains optimizer-excluded persistent module state without a public replacement API or
  declaration-time schema. State load may use the target's current data type and Shape as the
  strict target schema without changing protected layer-owned transition behavior.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [target glossary](../../../../glossary.md#term-target))</span>
- The existing collision-free module namespace and qualified paths are authoritative. State load
  identifies targets only through complete relative paths and kind, never reflection, field names,
  class names, wrapper equality, or positional list matching.
- The strict public load method is the only new mutation surface. It validates a complete owned
  tree and does not expose a reusable unchecked assignment primitive to callers or subclasses.
- In-memory atomicity does not require rollback because every ordinary caller-visible failure is
  discovered before installation and current post-validation assignments cannot reject. If that
  premise becomes false during implementation, stop and report the conflict instead of adding a
  hidden bypass or partial rollback design.<span data-glossary-navigation="true"> ([atomicity glossary](../../../../glossary.md#term-atomicity); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- [Planning](../../../../glossary.md#term-planning) is non-authoritative. If implementation reveals a conflict with `ARCHITECTURE.md`, a
  need for another module dependency, or a need to define persistent checkpoint transport, stop
  before editing and request an explicit decision.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>

## Public [API](../../../../glossary.md#term-application-programming-interface-api)

All new public/protected surface is exact:

| Type | [Signature](../../../../glossary.md#term-signature) | Visibility | Contract |
|---|---|---|---|
| `StateKind` | [`PARAMETER`](../../../../glossary.md#term-parameter), [`BUFFER`](../../../../glossary.md#term-buffer) | `public enum` | Distinguishes trainable and persistent non-trainable [binding](../../../../glossary.md#term-binding) roles; enum names are in-memory Java [identities](../../../../glossary.md#term-identity-meanings), not wire [tokens](../../../../glossary.md#term-token). |
| `StateEntry` | `StateEntry(String path, StateKind kind, Tensor value)` | `public record` [canonical](../../../../glossary.md#term-canonical-form) constructor | Validates path, kind, and value in component order and retains the exact [Tensor](../../../../glossary.md#tensor) reference. |
| `StateEntry` | `path()`, `kind()`, `value()` | generated public accessors | Return the exact [immutable](../../../../glossary.md#term-immutable) components; `value()` performs no copy or [evaluation](../../../../glossary.md#term-evaluation). |
| `StateDictionary` | `StateDictionary(List<StateEntry> entries)` | `public record` canonical constructor | Defensively copies in encounter order and rejects null entries or duplicate paths. |
| `StateDictionary` | `entries()` | generated public accessor | Returns the unmodifiable ordered structural [snapshot](../../../../glossary.md#term-snapshot); never `null`. |
| [`Module`](../../../../glossary.md#term-module) | `StateDictionary stateDictionary()` | `public final` | [Captures](../../../../glossary.md#term-capture-meanings) exact current Tensor references in deterministic combined parameter/buffer tree order. |
| `Module` | `void loadStateDictionary(StateDictionary dictionary)` | `public final` | Strictly validates the complete [target](../../../../glossary.md#term-target) tree before installing exact candidate Tensor references. |

Records retain their standard generated value equality, hashing, and diagnostic `toString`.
Those methods and enum names are not [serialization](../../../../glossary.md#term-serialization) or persistent compatibility contracts. No
other public/protected constructor, method, field, nested type, overload, option, or mutation
surface is added.

## State [model](../../../../glossary.md#term-model), ordering, and [ownership](../../../../glossary.md#term-ownership-meanings)

The state dictionary is an [immutable](../../../../glossary.md#term-immutable) shallow [snapshot](../../../../glossary.md#term-snapshot):

- the entry list and each entry's path/kind association are immutable;
- the exported list order is root direct parameters, root direct buffers, then each child subtree
  recursively in registration order;
- each [Tensor](../../../../glossary.md#tensor) reference is retained exactly at export or candidate-entry construction time;
- Tensor descriptor and provenance remain immutable under Model ownership, while the Tensor's
  separately mutable borrowed host-storage association keeps its existing Model lifecycle;<span data-glossary-navigation="true"> ([Tensor descriptor glossary](../../../../glossary.md#tensor-descriptor); [mutable glossary](../../../../glossary.md#term-mutable); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [host glossary](../../../../glossary.md#term-host); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- no wrapper, module, parent link, mode, random state, optimizer state, or historical value is
  retained by an entry; and<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [optimizer state glossary](../../../../glossary.md#term-optimizer-state))</span>
- an exported dictionary can be loaded into another independently constructed module tree only if
  the complete strict target schema matches.

For example, a root with [parameter](../../../../glossary.md#term-parameter) [`weight`](../../../../glossary.md#term-weight), [buffer](../../../../glossary.md#term-buffer) `step`, and child `encoder` whose direct state
is parameter `scale` then buffer `runningMean` exports:

```text
weight                    PARAMETER
step                      BUFFER
encoder.scale             PARAMETER
encoder.runningMean       BUFFER
```

The order is observable and deterministic, but load remains path-keyed. A caller-constructed
dictionary containing the same four unique entries in another order is compatible and installs in
the [target](../../../../glossary.md#term-target) order above.

`StateEntry` does not duplicate [`DataType`](../../../../glossary.md#data-type--datatype), [`Shape`](../../../../glossary.md#shape), or `requiresGrad` components. Its exact Tensor
has one immutable `TensorDescriptor`, so duplicating [schema](../../../../glossary.md#term-schema) would create mismatch states without
adding information. [Layout](../../../../glossary.md#layout) is intentionally omitted from load compatibility because existing
parameter replacement permits layout changes and a buffer [binding](../../../../glossary.md#term-binding) may move between resolved and
unresolved expressions. Storage, label, [provenance](../../../../glossary.md#provenance), and [identity](../../../../glossary.md#term-identity-meanings) are likewise [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) facts, not
binding schema.

## Construction [validation](../../../../glossary.md#term-validation)

### `StateEntry`

1. Reject null `path` with `NullPointerException("path")`.
2. Reject a blank path or any empty/blank dot-separated segment with
   `IllegalArgumentException`. Preserve accepted path text exactly.
3. Reject null `kind` with `NullPointerException("kind")`.
4. Reject null `value` with `NullPointerException("value")`.
5. Retain all three exact values without [Tensor](../../../../glossary.md#tensor) inspection, allocation, [evaluation](../../../../glossary.md#term-evaluation), or mutation.

Path [validation](../../../../glossary.md#term-validation) accepts exactly names that could be produced by the current [Module](../../../../glossary.md#term-module) namespace. It
does not test that a [target](../../../../glossary.md#term-target) module currently contains the path.

### `StateDictionary`

1. Reject null `entries` with `NullPointerException("entries")`.
2. Traverse the supplied list in encounter order. Reject a null entry with an index-bearing
   `NullPointerException` and reject the first repeated path with `IllegalArgumentException`.
3. Only after complete [validation](../../../../glossary.md#term-validation), retain `List.copyOf(entries)` as the ordered immutable
   structural snapshot.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>

Duplicate rejection belongs at dictionary construction, before a value can be supplied to a
[module](../../../../glossary.md#term-module). Because `StateEntry` is already valid, dictionary construction creates no [Tensor](../../../../glossary.md#tensor) or module
[side effect](../../../../glossary.md#term-side-effect).

## Strict load [validation](../../../../glossary.md#term-validation) and [atomic](../../../../glossary.md#resource-transaction-and-rollback) installation

`loadStateDictionary` performs these phases in exact order:

1. Reject null `dictionary` with `NullPointerException("dictionary")`.
2. Traverse the complete target tree once with the existing explicit-stack, repeated-identity
   defense. Capture each target's qualified path, kind, exact wrapper, and exact current Tensor at
   load start. No binding changes during this phase.<span data-glossary-navigation="true"> ([Capture glossary](../../../../glossary.md#term-capture-meanings))</span>
3. Index the already unique candidate entries by exact path.
4. In [target](../../../../glossary.md#term-target) traversal order, reject the first target path absent from the candidate as missing.
5. In candidate encounter order, reject the first path absent from the target as unexpected.
6. In target traversal order, validate each matched entry:
   - exact `StateKind`;
   - exact [`DataType`](../../../../glossary.md#data-type--datatype) [identity](../../../../glossary.md#term-identity-meanings);
   - structural `Shape.equals`; and
   - for a [parameter](../../../../glossary.md#term-parameter) only, incoming `requiresGrad == true`.
7. After every target validates, traverse the prepared target/entry pairs in target order. Call
   the existing final `Parameter.replace` for a parameter and the existing package-owned
   `Buffer.replaceValue` for a [buffer](../../../../glossary.md#term-buffer). Both receive the exact non-null candidate [Tensor](../../../../glossary.md#tensor).
8. Return normally after all assignments. Do not change mode, state order, [ownership](../../../../glossary.md#term-ownership-meanings), or a Tensor.

Missing state takes precedence over unexpected state so a target's required [schema](../../../../glossary.md#term-schema) is reported
first; kind then [data type](../../../../glossary.md#data-type--datatype) then [Shape](../../../../glossary.md#shape) then parameter [gradient](../../../../glossary.md#term-gradient) eligibility is the per-path
compatibility order. Failures use `IllegalArgumentException` with the category and exact path plus
expected/actual facts where applicable. Message prose is diagnostic and not a wire contract.

All [lookup](../../../../glossary.md#term-lookup) and compatibility failures occur before step 7 and therefore leave every exact old
[binding](../../../../glossary.md#term-binding) current. Step 7 is non-throwing by construction for ordinary program behavior: parameters
have already passed the same [immutable](../../../../glossary.md#term-immutable) [descriptor](../../../../glossary.md#term-descriptor) checks used by `replace`, and buffers require
only a non-null exact reference. No new unchecked wrapper method, [rollback](../../../../glossary.md#term-rollback) map, catch-and-restore
path, or partially successful result exists.

This [atomicity](../../../../glossary.md#term-atomicity) assumes the caller externally excludes concurrent declaration, ownership
corruption, replacement, load, mode changes, and forward construction. The sequential assignments
are not promised to appear simultaneously to a racing reader. [JVM](../../../../glossary.md#term-java-virtual-machine-jvm)-fatal conditions such as
resource exhaustion are outside the ordinary validation atomicity guarantee and do not justify a
public rollback protocol.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.nn.module` — owns module state, paths, dictionary values, strict
  validation, and installation.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [validation glossary](../../../../glossary.md#term-validation))</span>
- `io.github.pho001.synaptik.model.tensor`, `.datatype`, and `.shape` — supply unchanged immutable
  Tensor descriptor facts.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [Tensor descriptor glossary](../../../../glossary.md#tensor-descriptor))</span>

No package is added. Type placement:

- `io.github.pho001.synaptik.nn.module.StateKind` — [binding](../../../../glossary.md#term-binding)-role vocabulary colocated with
  [`Parameter`](../../../../glossary.md#term-parameter) and [`Buffer`](../../../../glossary.md#term-buffer).
- `io.github.pho001.synaptik.nn.module.StateEntry` — one qualified in-memory binding [snapshot](../../../../glossary.md#term-snapshot).
- `io.github.pho001.synaptik.nn.module.StateDictionary` — immutable ordered aggregate and duplicate
  boundary.
- `io.github.pho001.synaptik.nn.module.Module` — sole owner of tree export, strict validation, and
  multi-binding installation.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
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

Expected documentation and [planning](../../../../glossary.md#term-planning) files:

- `docs/api/training-api.md` — replace the no-checkpoint/no-batch statement with the exact current
  in-memory dictionary and strict load boundary while keeping optimizer and persistent formats
  planned.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- `docs/glossary.md` — update the existing NN module/parameter/buffer entry and add a reusable state
  dictionary/checkpoint distinction.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [module glossary](../../../../glossary.md#term-module); [parameter glossary](../../../../glossary.md#term-parameter); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- `docs/planning/extensions/nn/master-plan.md`.
- this task specification.

## Maximum [scope](../../../../glossary.md#term-scope)

Implementation may create or modify exactly the ten paths listed above: five production Java
files, one focused [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) test file, two explanatory documentation files, and two [planning](../../../../glossary.md#term-planning) files. If
implementation needs another test [owner](../../../../glossary.md#term-owner), `Parameter.java`, `Buffer.java`, a layer edit, a second
[schema](../../../../glossary.md#term-schema)/report type, a public [buffer](../../../../glossary.md#term-buffer) method, a [codec](../../../../glossary.md#term-codec)/format, a [dependency](../../../../glossary.md#term-dependency)/build/architecture change,
or an eleventh path, stop and propose a focused follow-up rather than expanding this task.

## Acceptance criteria

- The exact public API table is implemented with three final immutable public state types and only
  the two new public final Module methods. No generic loader, manager, service, builder, options,
  result, report, map facade, format, version, or public/protected unchecked mutation appears.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [immutable glossary](../../../../glossary.md#term-immutable); [Module glossary](../../../../glossary.md#term-module); [facade glossary](../../../../glossary.md#term-facade))</span>
- `StateEntry` validates and retains exactly path, kind, and Tensor. Path grammar matches current
  relative module paths, accepted text is not normalized, and every failure has no Tensor/module
  side effect.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [side effect glossary](../../../../glossary.md#term-side-effect))</span>
- `StateDictionary` preserves supplied encounter order in an unmodifiable defensive copy and
  rejects null entries and the first duplicate exact path. Empty dictionaries are valid.<span data-glossary-navigation="true"> ([defensive copy glossary](../../../../glossary.md#term-defensive-copy))</span>
- Export proves exact combined ordering across root/direct state and multiple nested children,
  exact qualified paths/kinds/Tensor references, stable output for an empty module, deep-tree
  traversal without call-stack dependence, and repeated-identity failure before a result.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- An exported dictionary is a value snapshot: later parameter or buffer replacement leaves its
  entries unchanged. Existing discovery snapshots retain stable wrappers that observe successful
  load, and Tensors/expressions captured before load remain unchanged.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- Strict load accepts candidate list reordering but requires the exact complete path set. Tests
  cover null dictionary, missing path, unexpected path, duplicate construction, and parameter/
  buffer kind mismatch with no target mutation.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- Compatibility tests cover exact data-type mismatch, structural [Shape](../../../../glossary.md#shape) mismatch, [parameter](../../../../glossary.md#term-parameter)
  `requiresGrad == false`, and atomic failure at a later target after earlier targets are valid.
  Every ordinary failure leaves all parameter and buffer bindings exactly unchanged.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [bindings glossary](../../../../glossary.md#term-binding))</span>
- Compatibility accepts different Tensor identity, label, layout, storage presence, and
  provenance when kind/data type/Shape and parameter gradient eligibility match. Buffer gradient
  eligibility may differ, preserving current BatchNorm state semantics.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance); [data type glossary](../../../../glossary.md#data-type--datatype); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Successful load retains every exact candidate Tensor, installs in target traversal order,
  preserves all wrapper/name/path/order/child/mode identities, and causes no Tensor creation,
  copy, evaluation, mutation, identifier allocation, or expression rewrite.<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- Empty-to-empty load succeeds; empty/non-empty mismatches fail strictly. Malformed repeated tree
  identity fails before installation. Existing parameter, buffer, traversal, mode, BatchNorm, and
  Dropout tests remain unchanged and pass.<span data-glossary-navigation="true"> ([Dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- Javadocs and module package documentation define snapshot depth, ownership, schema, exact
  ordering, strict failures, side-effect order, atomicity meaning, threading limits, old-expression
  behavior, and persistence exclusions with complete <span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [schema glossary](../../../../glossary.md#term-schema); [atomicity glossary](../../../../glossary.md#term-atomicity); [persistence glossary](../../../../glossary.md#term-persistence))</span>`@param`, `@return`, and `@throws` tags.
- Training API and glossary distinguish implemented in-memory state transfer from future
  persistent checkpoints and continue to exclude optimizer/session/RNG/execution state.<span data-glossary-navigation="true"> ([checkpoints glossary](../../../../glossary.md#term-checkpoint); [optimizer glossary](../../../../glossary.md#term-optimizer); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- A separate clean-context documentation pass finalizes all affected Javadoc, explanatory text,
  glossary and planning evidence, generated Javadoc, Markdown, exact scope/status, and whitespace
  before the implementation is Complete.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning); [scope glossary](../../../../glossary.md#term-scope))</span>
- No architecture/ADR/test, Gradle/dependency, Model/Tensor/Compile API, Training implementation,
  layer, compiler/runtime/prepare/Engine/backend, conformance/integration, CPU, global-roadmap,
  task-0011 specification, legacy, or unrelated path enters the implementation diff.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [Model glossary](../../../../glossary.md#term-model); [Compile glossary](../../../../glossary.md#compile); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [CPU glossary](../../../../glossary.md#term-cpu))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Validation tier: task validation for the single affected `extensions/nn` [module](../../../../glossary.md#term-module) plus a targeted
documentation pass. This task changes no module edge, [backend](../../../../glossary.md#backend) behavior, or end-to-end execution.

Implementation runs focused tests while developing:

```text
./gradlew :extensions:nn:test \
  --tests io.github.pho001.synaptik.nn.module.StateDictionaryTest
```

After executable Java and tests stabilize, [run](../../../../glossary.md#term-run) the affected module once as final Java evidence:

```text
./gradlew :extensions:nn:test
```

The separate documentation-focused pass reuses that successful evidence unless executable Java
changes afterward. After final [Javadoc](../../../../glossary.md#term-javadoc) edits it runs:

```text
./gradlew :extensions:nn:javadoc
```

Final implementation/documentation validation also checks:

- reflection and `javap -public` for the exact three-type/two-Module-method public surface;
- production imports and the unchanged [Model](../../../../glossary.md#term-model)-only [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [dependency](../../../../glossary.md#term-dependency);
- local Markdown paths, links, heading anchors, balanced fences, terminal newlines, and trailing
  whitespace in the four documentation/planning files;<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- exactly the ten task-owned implementation paths;
- tasks 0001–0009 Complete, exactly task 0010 Ready/In progress/Complete as appropriate, task 0011
  a concise Draft row with no detailed specification, and no second Ready NN task;<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- `git diff --check`; and
- `git diff --no-index --check /dev/null <path>` for every untracked new file.

Repository-wide, architecture, backend-[conformance](../../../../glossary.md#term-conformance), integration, Model, [Compiler](../../../../glossary.md#term-compiler), [CPU](../../../../glossary.md#term-cpu), and root
tests are deferred to the NN capability [checkpoint](../../../../glossary.md#term-checkpoint) or [CI](../../../../glossary.md#term-continuous-integration-ci). Existing architecture enforcement
already locks the unchanged Model-only dependency, and focused NN tests own the changed behavior.

## Dependencies

- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) tasks 0001–0009 are Complete.
- Tasks 0002 and 0004A provide collision-free deterministic paths, iterative repeated-identity-
  defended traversal, stable wrappers, and public schema-compatible parameter replacement.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- Task 0008 establishes a concrete persistent-buffer consumer whose next expressions may be
  gradient eligible without becoming optimizer targets.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [gradient glossary](../../../../glossary.md#term-gradient); [optimizer glossary](../../../../glossary.md#term-optimizer); [targets glossary](../../../../glossary.md#term-target))</span>
- [Model](../../../../glossary.md#term-model) [`Tensor`](../../../../glossary.md#tensor), `TensorDescriptor`, [`DataType`](../../../../glossary.md#data-type--datatype), and [`Shape`](../../../../glossary.md#shape) provide [immutable](../../../../glossary.md#term-immutable) exact [schema](../../../../glossary.md#term-schema) facts.
- [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0007 and the existing [architecture test](../../../../glossary.md#term-architecture-test) provide the accepted `model -> nn -> training`
  dependency direction.<span data-glossary-navigation="true"> ([dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>
- No persistent storage, Engine, execution, compiler, backend, or optimizer prerequisite exists
  because this task transfers in-memory Tensor references only.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend))</span>

## Follow-up tasks

- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0011 remains Draft and separately owns [unary](../../../../glossary.md#term-unary) [Tensor composition](../../../../glossary.md#tensor-composition) and `Sequential`. It may
  consume ordinary modules but must not broaden or reinterpret state-dictionary loading.<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span>
- A future checkpoint persistence task may define a versioned codec/storage/file contract only
  after a concrete consumer and materialized-value lifecycle are known. It must consume this
  in-memory schema without treating enum names, record <span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [persistence glossary](../../../../glossary.md#term-persistence); [codec glossary](../../../../glossary.md#term-codec); [lifecycle glossary](../../../../glossary.md#lifecycle); [schema glossary](../../../../glossary.md#term-schema))</span>`toString`, Tensor provenance, or Java
  object serialization as a wire format. Do not create its detailed specification now.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [provenance glossary](../../../../glossary.md#provenance); [serialization glossary](../../../../glossary.md#term-serialization); [wire format glossary](../../../../glossary.md#term-wire-format))</span>
- `extensions/training` may later coordinate state-dictionary timing with optimizer/session state,
  but optimizer state remains a separate training-owned contract and must not be inserted into
  this NN dictionary.<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [optimizer glossary](../../../../glossary.md#term-optimizer); [optimizer state glossary](../../../../glossary.md#term-optimizer-state))</span>

## Documentation and no-change review

Document profiles:

- Java/package [Javadoc](../../../../glossary.md#term-javadoc): General plus [API](../../../../glossary.md#term-application-programming-interface-api)/Javadoc.
- Training API: General plus API/Javadoc.
- glossary: General reference style.
- task/master plan: General plus [Planning](../../../../glossary.md#term-planning).

Required documentation changes are the five new/changed production Javadocs, the new [module](../../../../glossary.md#term-module)
package documentation, Training API state/[checkpoint](../../../../glossary.md#term-checkpoint) wording, the glossary distinction, and
synchronized [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) planning records.

The completion summary must record these reasoned no-change conclusions:

- `ARCHITECTURE.md`, focused architecture documents, and ADR 0007 remain accurate because NN
  already owns module parameter/buffer state and dependency direction does not change.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [parameter glossary](../../../../glossary.md#term-parameter); [buffer glossary](../../../../glossary.md#term-buffer); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>
- `docs/api/tensor-api.md`, `docs/api/compile-api.md`, Model source/plans/capabilities, and related
  Tensor/Shape/DataType Javadocs remain accurate because no model semantic, descriptor, capture,
  gradient, or execution behavior changes.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor); [Shape glossary](../../../../glossary.md#shape); [DataType glossary](../../../../glossary.md#data-type--datatype); [descriptor glossary](../../../../glossary.md#term-descriptor); [capture glossary](../../../../glossary.md#term-capture-meanings); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- `docs/architecture/training-graph.md` remains accurate because no optimizer, session, gradient
  publication, compiled update, or execution orchestration is added.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [publication glossary](../../../../glossary.md#term-publication))</span>
- `Parameter.java` remains accurate because its individual public replacement still provides no
  checkpoint or transaction; Module now owns the separate strict batch contract. <span data-glossary-navigation="true"> ([transaction glossary](../../../../glossary.md#term-transaction); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>`Buffer.java`
  remains accurate because it gains no public update or declaration schema.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>
- Concrete layer Javadocs/tests remain accurate: BatchNorm's forward transition is still
  sequential and non-atomic, while Dropout's caller-owned graph RNG state is absent from module
  state.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [Dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [graph RNG state glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- Build files and architecture tests remain accurate because no dependency changes. Backend
  conformance and integration tests remain unnecessary because no numerical execution or
  end-to-end lifecycle changes.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- The global roadmap, [CPU](../../../../glossary.md#term-cpu) work, other modules, legacy branch, and task 0011 remain untouched.

## Architecture impact

Expected impact: None.

This task implements the architecture's existing [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [ownership](../../../../glossary.md#term-ownership-meanings) of [parameters](../../../../glossary.md#term-parameter), persistent [buffers](../../../../glossary.md#term-buffer),
and [module](../../../../glossary.md#term-module)-tree traversal. It changes no module [owner](../../../../glossary.md#term-owner), [dependency direction](../../../../glossary.md#java-platform-and-build-tools), [Tensor](../../../../glossary.md#tensor) [invariant](../../../../glossary.md#term-invariant), or
execution [lifecycle](../../../../glossary.md#lifecycle). If implementation requires a persistent [checkpoint](../../../../glossary.md#term-checkpoint) owner, [optimizer state](../../../../glossary.md#term-optimizer-state),
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

## Documentation-agent [handoff](../../../../glossary.md#term-handoff)

After Java/tests stabilize, give the clean documentation context:

- this task and exact ten-path limit;
- the final diff and exact focused/final [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) command results and counts;
- the public API table, path/order/schema/ownership decisions, strict validation order, atomicity
  boundary, Buffer gradient exception, wrapper/expression semantics, and persistence exclusions;<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [schema glossary](../../../../glossary.md#term-schema); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [validation glossary](../../../../glossary.md#term-validation); [atomicity glossary](../../../../glossary.md#term-atomicity); [Buffer glossary](../../../../glossary.md#term-buffer); [gradient glossary](../../../../glossary.md#term-gradient); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- the relevant architecture/ADR, documentation rules and profiles, final Module/Parameter/Buffer/
  layer source and tests, Model descriptor contracts, Training API, glossary, dependency test, and
  NN planning history;<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Module glossary](../../../../glossary.md#term-module); [Parameter glossary](../../../../glossary.md#term-parameter); [Model glossary](../../../../glossary.md#term-model); [descriptor glossary](../../../../glossary.md#term-descriptor); [dependency glossary](../../../../glossary.md#term-dependency); [planning glossary](../../../../glossary.md#term-planning))</span>
- the mandate to finalize all affected documentation and reasoned no-change conclusions without
  changing executable behavior;
- generated-[Javadoc](../../../../glossary.md#term-javadoc), reflection/`javap`, imports/dependency, Markdown, exact-scope/status,
  whitespace/newline, and task-0011-absence gates; and<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- the required completion-summary and `Status` format from `AGENTS.md`.

## Local decisions

- [`Module`](../../../../glossary.md#term-module) owns both export and strict load. It already owns the complete private tree registries
  and can install through existing wrapper operations, so another loader/service or widened
  protected mutation API would add indirection without a boundary.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Use three small public [immutable](../../../../glossary.md#term-immutable) values: `StateKind`, `StateEntry`, and `StateDictionary`. A raw
  `Map<String, Tensor>` would lose parameter/buffer kind, while a separate schema/report hierarchy
  has no current consumer.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [buffer glossary](../../../../glossary.md#term-buffer); [schema glossary](../../../../glossary.md#term-schema))</span>
- Keep path in `StateEntry` and accept a list in `StateDictionary`. This preserves deterministic
  order and lets dictionary construction detect duplicate paths before load; a Map constructor
  could not observe duplicates.
- Make strictness unconditional. A report/result type and permissive options are unnecessary when
  the current requirement is exact all-or-nothing state installation.
- Reuse immutable Tensor descriptor facts rather than duplicate data type, Shape, or gradient
  fields in an entry. Load deliberately excludes layout, storage, provenance, label, and identity.<span data-glossary-navigation="true"> ([Tensor descriptor glossary](../../../../glossary.md#tensor-descriptor); [data type glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Parameter compatibility is the existing permanent schema. Buffer compatibility uses the
  target's current data type and Shape and ignores gradient eligibility, preserving BatchNorm's
  valid gradient-eligible next-statistic bindings without inventing a universal buffer schema.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [statistic glossary](../../../../glossary.md#term-statistic); [bindings glossary](../../../../glossary.md#term-binding))</span>
- Candidate list order is preserved but is not compatibility. Paths identify entries, while the
  target's established traversal order controls diagnostics and installation.
- Validate the complete tree before installing. Current immutable descriptors and existing final
  wrapper operations make the installation phase non-throwing for ordinary behavior, so no
  rollback or unchecked assignment primitive is justified.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- Keep persistent checkpoints deferred. A Tensor-reference snapshot says nothing about evaluated
  bytes, host/device materialization, schema versions, durability, or atomic file replacement.<span data-glossary-navigation="true"> ([checkpoints glossary](../../../../glossary.md#term-checkpoint); [Tensor glossary](../../../../glossary.md#tensor); [snapshot glossary](../../../../glossary.md#term-snapshot); [host glossary](../../../../glossary.md#term-host); [device glossary](../../../../glossary.md#term-device); [materialization glossary](../../../../glossary.md#materialization); [durability glossary](../../../../glossary.md#term-durability); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>

## Known limitations

- The dictionary is shallow and in-memory. It retains Tensor objects, not portable values or a
  durable checkpoint, and Tensor host-storage lifetime remains governed by the existing Model
  contract.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [host glossary](../../../../glossary.md#term-host); [lifetime glossary](../../../../glossary.md#term-lifetime); [Model glossary](../../../../glossary.md#term-model))</span>
- Buffer data type and Shape compatibility are relative to the target's current binding because
  generic Buffer has no declaration schema. Direct protected buffer transitions remain capable of
  changing that current schema outside this strict load operation.<span data-glossary-navigation="true"> ([Buffer glossary](../../../../glossary.md#term-buffer); [data type glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [target glossary](../../../../glossary.md#term-target); [binding glossary](../../../../glossary.md#term-binding); [schema glossary](../../../../glossary.md#term-schema); [operation glossary](../../../../glossary.md#operation))</span>
- The contract is strict only; it cannot partially load, rename, convert, ignore, or report a set
  of mismatches.
- Export and load are not thread-safe or linearizable. Callers must coordinate them with module
  declaration, replacement, mode changes, and forward construction.<span data-glossary-navigation="true"> ([thread-safe glossary](../../../../glossary.md#term-thread-safe); [module glossary](../../../../glossary.md#term-module))</span>
- Successful installation is sequential and offers no simultaneous [view](../../../../glossary.md#term-view-meanings) to racing readers.
- [JVM](../../../../glossary.md#term-java-virtual-machine-jvm)-fatal failure is outside ordinary validate-before-install [atomicity](../../../../glossary.md#term-atomicity).
- Optimizer state, graph RNG state, mode, session state, and evaluated running-statistic bytes are
  not represented.<span data-glossary-navigation="true"> ([Optimizer state glossary](../../../../glossary.md#term-optimizer-state); [graph RNG state glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [statistic glossary](../../../../glossary.md#term-statistic))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) context `/root/nn_0010_planning` read the repository instructions, [architecture contract](../../../../glossary.md#architecture-contract)
and index, planning guide and roadmap, documentation rules and General/Planning/[API](../../../../glossary.md#term-application-programming-interface-api)-[Javadoc](../../../../glossary.md#term-javadoc)
profiles, [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0007, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) master plan and completed task history, final [Module](../../../../glossary.md#term-module)/[Parameter](../../../../glossary.md#term-parameter)/[Buffer](../../../../glossary.md#term-buffer) and
layer source/tests, [Model](../../../../glossary.md#term-model) [descriptor](../../../../glossary.md#term-descriptor)/[Shape](../../../../glossary.md#shape)/data-type contracts, Training API and training-[graph](../../../../glossary.md#graph)
boundary, glossary terminology, build edges, and NN/training architecture enforcement. A read-only
legacy tree [scan](../../../../glossary.md#term-cumulative-scan) found no selected module state-dictionary/[checkpoint](../../../../glossary.md#term-checkpoint) capability worth carrying
forward; no legacy architecture or source was used.

The planning pass selected the exact public API and validate-before-install algorithm recorded
above and found no architecture blocker. Current private Module registries plus existing final
wrapper operations are sufficient; no new public/protected unchecked [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) primitive is required.

Planning validation passed:

- the repository-local link/anchor check resolved every local Markdown target in the two planning
  files;
- fence balance, terminal newlines, and trailing-whitespace checks passed;
- exactly one NN task file has [`Ready`](../../../../glossary.md#term-ready) status and exactly one NN master-plan row is Ready;
- task 0011 remains a concise Draft row and has no detailed specification;
- exact changed-path inspection contains only the NN master plan and this new task;
- whole-worktree `git diff --check` passed; and
- `git diff --no-index --check /dev/null
  docs/planning/extensions/nn/tasks/0010-state-dictionary-and-checkpoint-contract.md` passed.

No Java test, Javadoc, architecture, repository, [conformance](../../../../glossary.md#term-conformance), or integration command was [run](../../../../glossary.md#term-run) for
this planning-only change. Implementation and documentation evidence remain empty until their
separate task contexts complete.

Implementation context `/root/nn_0010_implementation` added the exact planned three public state
values, the two final Module methods, package documentation, and one focused contract suite within
the six authorized executable paths. The first focused [compile](../../../../glossary.md#compile) exposed and corrected one test-only
parenthesis error before any test ran. The stabilized focused command
`./gradlew :extensions:nn:test --tests
io.github.pho001.synaptik.nn.module.StateDictionaryTest` then passed 1 suite and 15 tests with zero
failures, errors, or skips. It covers exact API/finality, entry/list validation and immutability,
combined deterministic export, shallow value snapshots, reordered strict load, validation order,
zero-mutation failures, Buffer [gradient](../../../../glossary.md#term-gradient) treatment, exact-reference installation, stable wrappers,
old expressions, empty and deep trees, and repeated-[identity](../../../../glossary.md#term-identity-meanings) defense.

After executable Java stabilized, implementation context `/root/nn_0010_implementation` ran the
sole authoritative `./gradlew :extensions:nn:test` command. It passed 16 suites and 98 tests with
zero failures, errors, or skips. No executable Java or test changed afterward.

Preliminary implementation validation also passed:

- `./gradlew :extensions:nn:javadoc` completed successfully after one sandbox-denied Gradle-cache
  lock attempt was rerun with the already approved Gradle permission; the documentation context
  must run final Javadoc again after its edits.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [cache glossary](../../../../glossary.md#term-cache))</span>
- `javap -public` over `StateKind`, `StateEntry`, `StateDictionary`, and `Module` showed exactly the
  three planned final state types, record-generated value/accessor methods, and the two new final
  Module methods with no extra public state API.
- The focused reflection test independently locks finality, record components, enum order, and the
  two Module signatures.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>
- Production import and forbidden-vocabulary scans found only Model [Tensor](../../../../glossary.md#tensor) plus `java.util`
  dependencies and no Training, compiler, runtime, prepare, Engine, backend, serializer, codec,
  file-I/O, reflective-discovery, or checkpoint production surface.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [codec glossary](../../../../glossary.md#term-codec))</span>
- `extensions/nn/build.gradle.kts` remains the unchanged one-edge Model-only [dependency](../../../../glossary.md#term-dependency).
- Exact changed/untracked-path inspection found only the eight currently applicable task-owned
  implementation/planning paths. Training API and glossary remain reserved for the documentation
  pass; no architecture, build, layer, other module, task-0011, or unrelated path changed.
- `git diff --check` passed. New-file whitespace checks produced no diagnostics; terminal-newline
  checks passed.

Independent documentation context `/root/nn_0010_docs` reviewed the final implementation and
tests against the architecture, task, preceding NN contracts, Model descriptor contracts, and
current training boundary. It found no executable, [atomicity](../../../../glossary.md#term-atomicity), architecture, or [scope](../../../../glossary.md#term-scope) blocker and
changed no executable Java or tests. The context finalized only task-authorized Javadocs/package
documentation, Training API, glossary, and planning records.

The documentation pass selected the General, API/Javadoc, Planning, and Example profiles. It
finalized exact path/order/reference [ownership](../../../../glossary.md#term-ownership-meanings), record construction and accessor semantics,
strict missing-before-unexpected and per-[target](../../../../glossary.md#term-target) validation order, target-current Buffer [schema](../../../../glossary.md#term-schema),
the intentional Buffer gradient exception, stable-wrapper installation, old-expression behavior,
caller-coordinated validate-before-install atomicity, threading/[race](../../../../glossary.md#term-race) limits, and no Tensor-copy/
execution boundary. The Training API and glossary now distinguish this shallow in-memory module
state from future persistent checkpoint codecs and separately owned [optimizer](../../../../glossary.md#term-optimizer)/session/[RNG state](../../../../glossary.md#graph-rng-state--graphrngstate);
both include the same concise nested root/child ordering and strict-load example.

Final documentation validation:

- `./gradlew :extensions:nn:javadoc` passed after all Javadoc edits (`BUILD SUCCESSFUL`, three
  actionable tasks: two executed and one up-to-date). Generated `Module`, `StateDictionary`,
  `StateEntry`, `StateKind`, and module-package pages were inspected for ordering, exact-reference,
  validation, ownership, atomicity, concurrency, failure, and persistence/execution boundaries.<span data-glossary-navigation="true"> ([concurrency glossary](../../../../glossary.md#term-concurrency); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- `javap -public` over `StateKind`, `StateEntry`, `StateDictionary`, and `Module` showed exactly the
  planned enum constants, record constructors/components/standard members, and the two new final
  Module state methods. A standalone Java 26 reflection check independently passed finality,
  record-component order/types, enum order, exact Module signatures, and absence of another
  public Module state method.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant))</span>
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
  sole <span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>`implementation(project(":modules:model"))` edge. Exact scope contains the five production
  paths, focused test, Training API, glossary, NN master plan, and this task: exactly ten paths.
  Concurrent CPU master-plan and task-planning paths were separately visible in the shared
  worktree and were preserved unchanged by this context.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
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
  specified order, and retain no mutable caller list.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- Added one unified explicit-stack state-[binding](../../../../glossary.md#term-binding) [snapshot](../../../../glossary.md#term-snapshot) in [`Module`](../../../../glossary.md#term-module). Export converts that stable
  target snapshot to exact Tensor entries; load captures it once, validates missing, unexpected,
  kind, data type, Shape, and parameter gradient eligibility before installing in target order.<span data-glossary-navigation="true"> ([captures glossary](../../../../glossary.md#term-capture-meanings))</span><span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [Tensor glossary](../../../../glossary.md#tensor); [data type glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [parameter glossary](../../../../glossary.md#term-parameter); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Kept installation inside existing owning operations: `Parameter.replace` and package-owned
  `Buffer.replaceValue`. No public buffer mutation, unchecked assignment, loader/service, codec,
  persistent format, optimizer state, execution behavior, dependency, or architecture change was
  introduced.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [codec glossary](../../../../glossary.md#term-codec); [optimizer state glossary](../../../../glossary.md#term-optimizer-state); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Added the focused same-package state-dictionary suite. Existing executable Java and tests were
  otherwise unchanged.

## Completion summary

- Completed changes: Added the exact immutable state values, deterministic combined Module export,
  strict complete validate-before-install load, focused coverage, finalized public/package
  Javadocs, Training API boundary, glossary definition/example, and synchronized planning records.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [Module glossary](../../../../glossary.md#term-module); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: Exactly the five production Java paths, one focused test path,
  Training API, glossary, NN master plan, and this task record.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Tests and validation: Reused the stabilized focused one-suite/15-test and authoritative NN
  16-suite/98-test results, both with zero failures, errors, or skips. Final Javadoc/generated-page,<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
  `javap`, independent reflection, Markdown, imports/dependency, exact ten-path scope, status,
  newline, trailing-whitespace, no-index, and <span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` gates passed.
- Documentation-agent review: Independent clean context `/root/nn_0010_docs` completed the
  mandatory General/API-Javadoc/Planning/Example review without changing executable behavior.
- Documentation impact: Training API and glossary now define the implemented shallow in-memory
  state dictionary and distinguish it from future persistent checkpoints and separate optimizer,
  session, graph-RNG, compiler, runtime, and backend state.<span data-glossary-navigation="true"> ([checkpoints glossary](../../../../glossary.md#term-checkpoint); [optimizer glossary](../../../../glossary.md#term-optimizer); [graph glossary](../../../../glossary.md#graph); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>
- Javadoc review: Finalized all affected type, constructor, component/accessor, Module-method, and
  package contracts; final generated Javadoc passed and was inspected.
- Glossary impact: Updated the existing NN module/parameter/buffer entry and added reusable state
  dictionary/checkpoint terminology plus a concise nested ordering/load example.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- Unresolved issues: None.
- Follow-up required: None for task 0010. Task 0011 remains Draft without a detailed specification.

Status: Complete
