# Task 0005A: Automatic Input Discovery and Compute Convenience

## Status

Complete

## Goal

Replace the two ordinary one-shot `Engine.forward(...)` overloads that require an explicit input
list with four `Engine.compute(...)` overloads that discover the request's logical input [Tensors](../../../../glossary.md#tensor)
from the requested Tensor expression closure. The call then uses [Compiler](../../../../glossary.md#term-compiler)'s final ordered
caller-[bindable input](../../../../glossary.md#bindable-input) [metadata](../../../../glossary.md#term-metadata) to select only the required discovered leaves, runs the existing
fresh one-shot [lifecycle](../../../../glossary.md#lifecycle), and returns [detached](../../../../glossary.md#detached-model-results) [immutable](../../../../glossary.md#term-immutable) [host](../../../../glossary.md#term-host) values.

This is an [Engine](../../../../glossary.md#term-engine) convenience change, not a Compiler [liveness](../../../../glossary.md#term-liveness) repair. Traversing immutable Tensor
expression [provenance](../../../../glossary.md#provenance) to build one transient [leaf](../../../../glossary.md#term-leaf) inventory is distinct from traversing compiler
[intermediate representation](../../../../glossary.md#term-intermediate-representation-ir) ([IR](../../../../glossary.md#intermediate-representation-dag-and-ssa)), reconstructing [graph](../../../../glossary.md#graph) liveness, or retaining Tensor state in a
compiled or prepared [artifact](../../../../glossary.md#term-artifact).

## Required reading

Read these files and contracts in full in the clean implementation context:

- root `AGENTS.md`, [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), and the
  [current architecture index](../../../../architecture/current-architecture-plan.md);
- the [planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>, [roadmap](../../../roadmap.md),
  [Engine master plan](../master-plan.md), and this task;
- completed [Engine](../../../../glossary.md#term-engine) tasks [0003](0003-typed-logical-input-binding-and-published-result-access.md),
  [0004](0004-explicit-host-materialization-boundary.md), and
  [0005](0005-one-shot-forward-convenience.md);
- completed [Compiler](../../../../glossary.md#term-compiler) tasks
  [0006B3](../../compiler/tasks/0006b3-public-constant-free-complete-compile-entry.md),
  [0006B4](../../compiler/tasks/0006b4-stable-caller-input-tensor-identity-bindings.md), and
  [0006B5](../../compiler/tasks/0006b5-published-compile-time-constant-descriptor-closure.md);
- current `Engine`, `AdvancedEngine`, `CompiledGraph`, [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), [`RunResult`](../../../../glossary.md#run-result--runresult),
  [`HostTensorValue`](../../../../glossary.md#host-tensor-value--hosttensorvalue), package documentation, focused Engine tests, and Engine integration tests;
- current [Model](../../../../glossary.md#term-model) [`Tensor`](../../../../glossary.md#tensor), [`TensorId`](../../../../glossary.md#tensorid), `TensorProvenance`, [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), `TensorDescriptor`,
  and host-storage contracts;<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host))</span>
- current Compiler `GraphCompilationPort`, `CompileArtifacts`, `CompileConstantPlan`, and final
  graph-input contracts plus their focused tests;<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
- the current Public, Compile, Runtime, Training, Tensor, and glossary documentation relevant to
  expression provenance, binding, one-shot execution, and host materialization; and<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime); [provenance glossary](../../../../glossary.md#provenance); [binding glossary](../../../../glossary.md#term-binding); [materialization glossary](../../../../glossary.md#materialization))</span>
- [documentation rules](../../../../developer-guide/documentation-rules.md) with the General,
  Planning, API/Javadoc, and Example profiles.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

`ARCHITECTURE.md` is authoritative. Stop before editing if implementation requires a new
[dependency direction](../../../../glossary.md#java-platform-and-build-tools), an inward public-contract change, Compiler [IR](../../../../glossary.md#term-intermediate-representation-ir) access from Engine, or a
different architecture decision.

## [Scope](../../../../glossary.md#term-scope)

- Remove the two public `Engine.forward(...)` overloads introduced by task 0005. Do not retain
  deprecated methods, aliases, adapters, or alternate names for them.<span data-glossary-navigation="true"> ([aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Add exactly four public `Engine.compute(...)` overloads: singleton and ordered-output forms,
  each with an unbounded convenience and an explicit aggregate-byte-bound form.<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton))</span>
- Preserve task 0005's one Engine admission, fresh compile/prepare/run/materialize/cleanup,
  forward-only settings, output order, aggregate preflight-before-copy, failure precedence,
  cleanup suppression, concurrency, and no-cache behavior.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [admission glossary](../../../../glossary.md#term-admission); [compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run); [preflight glossary](../../../../glossary.md#term-preflight); [concurrency glossary](../../../../glossary.md#term-concurrency); [cache glossary](../../../../glossary.md#term-cache))</span>
- Build one identity-safe transient inventory of provenance-free Tensor leaves reachable from the
  requested output Tensor expression closure.<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance); [Tensor glossary](../../../../glossary.md#tensor))</span>
- After [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names), treat the final ordered `CompileConstantPlan.bindableInputBindings()` as
  projected through `CompiledGraph.inputs()` as authoritative. Select matching inventory Tensors
  in that order and pass only those Tensors to the unchanged explicit-input run machinery.
- Discard the inventory and every selected Tensor reference after the synchronous call. No
  compiled handle, prepared handle, result, Engine field, cache, or inward artifact may retain
  them.<span data-glossary-navigation="true"> ([synchronous glossary](../../../../glossary.md#term-synchronous); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Finalize affected Javadocs and explanatory documentation through the required distinct clean
  documentation-focused pass.

## Exact public [API](../../../../glossary.md#term-application-programming-interface-api)

The existing public final [`Engine`](../../../../glossary.md#term-engine) must expose exactly these one-shot methods:

```java
public HostTensorValue compute(Tensor output);

public HostTensorValue compute(Tensor output, long maximumTotalBytes);

public List<HostTensorValue> compute(List<Tensor> outputs);

public List<HostTensorValue> compute(List<Tensor> outputs, long maximumTotalBytes);
```

The two overloads without a caller byte limit delegate to their corresponding bounded overload
with `Long.MAX_VALUE`. They do not bypass checked aggregate arithmetic, the per-value [JVM](../../../../glossary.md#term-java-virtual-machine-jvm)
`byte[]` ceiling, static-[Shape](../../../../glossary.md#shape) requirements, resolved-[layout](../../../../glossary.md#layout) requirements, storage [validation](../../../../glossary.md#term-validation), or
any inward bound.

No public or protected `forward(...)` overload remains. No new public type, request, options,
builder, result [carrier](../../../../glossary.md#term-storage-carrier), field, constructor, nested type, or exception hierarchy is authorized.
The [singleton](../../../../glossary.md#term-singleton) methods return the existing [`HostTensorValue`](../../../../glossary.md#host-tensor-value--hosttensorvalue); the ordered methods return an
[immutable](../../../../glossary.md#term-immutable) requested-order `List<HostTensorValue>`.

Ordinary [signatures](../../../../glossary.md#term-signature) expose only Engine, [Model](../../../../glossary.md#term-model), and [JDK](../../../../glossary.md#term-java-development-kit-jdk) types. They expose no [Compiler](../../../../glossary.md#term-compiler), [Planning](../../../../glossary.md#term-planning),
[Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), [backend](../../../../glossary.md#backend), representation, storage, [arena](../../../../glossary.md#term-arena), [`MemorySegment`](../../../../glossary.md#term-memorysegment), [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), or advanced
handle type.

## [Tensor](../../../../glossary.md#tensor)-expression [leaf](../../../../glossary.md#term-leaf) inventory

### Meaning

For this task, the requested **[Tensor](../../../../glossary.md#tensor) expression closure** is the set of exact Tensor objects
reachable by starting at each requested output and repeatedly following
`Tensor.provenance().producer().inputs()`. A **[provenance](../../../../glossary.md#provenance)-free [leaf](../../../../glossary.md#term-leaf)** is a reachable Tensor whose
`provenance()` is empty. [Parameters](../../../../glossary.md#term-parameter) and persistent [buffers](../../../../glossary.md#term-buffer) are not special execution categories:
when their current Tensor [bindings](../../../../glossary.md#term-binding) are reachable provenance-free leaves, they participate exactly
like every other leaf and their current [host](../../../../glossary.md#term-host)-storage associations remain caller-owned.

This walk observes [Model](../../../../glossary.md#term-model) expression [metadata](../../../../glossary.md#term-metadata) only. It does not inspect [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel),
`CompiledNode`, `GraphValue`, a [Planning](../../../../glossary.md#term-planning) [artifact](../../../../glossary.md#term-artifact), a [prepared schedule](../../../../glossary.md#prepared-schedule--preparedschedule), or [Runtime](../../../../glossary.md#term-runtime) state. It does
not decide whether a leaf is live, [constant](../../../../glossary.md#term-constant), bindable, optimized away, published, or executable.
Those decisions remain with [Compiler](../../../../glossary.md#term-compiler) and later [lifecycle](../../../../glossary.md#lifecycle) [owners](../../../../glossary.md#term-owner).

### Inventory algorithm

Under the admitted call and before or around the single [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names):

1. Start from the [immutable](../../../../glossary.md#term-immutable) requested-output [snapshot](../../../../glossary.md#term-snapshot) in encounter order.
2. Traverse iteratively, not recursively, so expression depth does not consume the Java call
   stack.
3. Track visited [Tensors](../../../../glossary.md#tensor) by exact object [identity](../../../../glossary.md#term-reference-identity), not `equals`, [`hashCode`](../../../../glossary.md#hash-codes-and-content-digests), label, descriptor,
   numeric <span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor))</span>[`TensorId`](../../../../glossary.md#tensorid) assumptions, or [producer](../../../../glossary.md#term-producer) structure.
4. For a Tensor with [provenance](../../../../glossary.md#provenance), visit the exact ordered Tensor references returned by
   `TensorProvenance.inputs()`. Multi-output siblings and shared subexpressions may reach the same
   input repeatedly; identity tracking visits it once.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
5. For a provenance-free Tensor, record its exact `TensorId` and exact Tensor reference. Distinct
   leaves with equal descriptors remain distinct; repeated reachability of one exact leaf yields
   one inventory entry.<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span>
6. Treat a repeated `TensorId` associated with a different exact Tensor object as an impossible
   Model identity inconsistency and fail before storage access. Do not choose one arbitrarily.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>

The inventory's traversal order is not caller-input order and carries no [liveness](../../../../glossary.md#term-liveness) meaning.

### Authoritative compiled selection

After the one forward-only [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) constructs the ordinary `CompiledGraph`, iterate
`CompiledGraph.inputs()` in its exact final order. For each entry:

1. find the exact inventoried [leaf](../../../../glossary.md#term-leaf) with the same [`TensorId`](../../../../glossary.md#tensorid);
2. fail with an `IllegalStateException` identifying the missing `TensorId` if no such reachable
   leaf exists;
3. preserve the exact [Tensor](../../../../glossary.md#tensor) reference without reconstructing it from [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), descriptor, or
   graph position; and
4. append it to the selected input list in `CompiledGraph.inputs()` order.

Ignore inventory leaves that have no final compiled [binding](../../../../glossary.md#term-binding). This is filtering by an
authoritative [Compiler](../../../../glossary.md#term-compiler) result, not [Engine](../../../../glossary.md#term-engine) [liveness](../../../../glossary.md#term-liveness) [inference](../../../../glossary.md#term-inference). Do not require inventory size to
equal binding count, inspect final [nodes](../../../../glossary.md#node) or [graph](../../../../glossary.md#graph) outputs, or attempt to explain why Compiler did
or did not publish a binding. The current audited Compiler pipeline normally binds every reachable
[provenance](../../../../glossary.md#provenance)-free caller leaf because [capture](../../../../glossary.md#term-capture-meanings) includes only the requested closure, exact rewriting
cannot discard a distinct bindable, folding consumes [constants](../../../../glossary.md#term-constant) only, [common-subexpression
elimination](../../../../glossary.md#term-common-subexpression-elimination-cse) requires identical remapped inputs, and [dead-code elimination](../../../../glossary.md#term-dead-code-elimination-dce) retains complete output
dependencies. Those facts justify the seam but are not reimplemented by Engine.

Pass only the selected ordered Tensor list into the existing task-0003 binding/[run](../../../../glossary.md#term-run) [operation](../../../../glossary.md#operation).
That operation remains responsible for [descriptor](../../../../glossary.md#term-descriptor) equality, one synchronized [host](../../../../glossary.md#term-host)-storage
[snapshot](../../../../glossary.md#term-snapshot) per selected Tensor, liveness/accessibility and geometry checks, [borrowing](../../../../glossary.md#term-borrowing), [Runtime](../../../../glossary.md#term-runtime)
execution, and cleanup. Do not inspect `hostStorage()` for an inventoried leaf that Compiler did
not select.

## Compute [lifecycle](../../../../glossary.md#lifecycle) and result semantics

After successful [Engine](../../../../glossary.md#term-engine) [admission](../../../../glossary.md#term-admission), the bounded [singleton](../../../../glossary.md#term-singleton) overload validates non-null `output`,
then the non-negative byte limit. The bounded ordered overload validates and snapshots non-null,
non-empty, non-null-[element](../../../../glossary.md#term-element), exact-object-[identity](../../../../glossary.md#term-identity-meanings)-unique `outputs` in the existing task-0005
order, then validates the non-negative byte limit. The no-limit overloads delegate with
`Long.MAX_VALUE` and otherwise have identical behavior.

Each bounded call performs exactly once:

```text
validate and snapshot outputs
  -> inventory reachable provenance-free Tensor leaves transiently
  -> compile the requested outputs with the current ordinary FORWARD_ONLY settings
  -> select matching leaves in final CompiledGraph.inputs() order
  -> prepare
  -> run through the unchanged explicit-input binding path
  -> validate every forward publication and aggregate canonical byte count
  -> materialize each occurrence in output order
  -> close temporary result and borrows
  -> discard transient Tensor inventory and return detached value(s)
```

The single form is the exact singleton [specialization](../../../../glossary.md#term-specialization) of the ordered form. Internal sharing must
not acquire a second Engine admission. The implementation may inventory immediately before or
immediately after [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names), but storage access and [run](../../../../glossary.md#term-run) [binding](../../../../glossary.md#term-binding) must use only the final ordered
selection described above.

[Publication](../../../../glossary.md#term-publication) identity/order, [alias](../../../../glossary.md#aliasing-and-arena) handling, [canonical](../../../../glossary.md#term-canonical-form) encoding, checked byte counts, the
`Integer.MAX_VALUE` ceiling, aggregate `Math.addExact`, complete [preflight](../../../../glossary.md#term-preflight) before the first copy,
one copy per [publication occurrence](../../../../glossary.md#publication-occurrence), [immutable](../../../../glossary.md#term-immutable) return list, and [detached](../../../../glossary.md#detached-model-results) post-close value
semantics remain exactly those of task 0005.

The current reusable advanced path remains unchanged:

```text
compile -> prepare -> run(preparedExecution, explicit input Tensors)
```

Its explicit input list is intentional for reusable [prepared execution](../../../../glossary.md#prepared-execution--preparedexecution) and is not deprecated by
this convenience.

## Failure, cleanup, and [concurrency](../../../../glossary.md#term-concurrency)

- [Engine](../../../../glossary.md#term-engine) [admission](../../../../glossary.md#term-admission) remains first and retains the stable closed-Engine failure precedence.
- Structural output and byte-limit [validation](../../../../glossary.md#term-validation) precede [provenance](../../../../glossary.md#provenance) traversal and inward work.
- A Model identity inconsistency or missing authoritative compiled binding fails before any host
  storage snapshot or borrow.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [identity glossary](../../../../glossary.md#term-identity-meanings); [binding glossary](../../../../glossary.md#term-binding); [host storage glossary](../../../../glossary.md#host-storage); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Compiler failures precede Prepare; Prepare precedes binding/run; run precedes publication and
  byte preflight; complete preflight precedes copying.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run); [publication glossary](../../../../glossary.md#term-publication); [preflight glossary](../../../../glossary.md#term-preflight))</span>
- Preserve the first unchecked failure or `Error` object. Attempt cleanup exactly as task 0005
  specifies, suppressing only distinct cleanup failures and never self-suppressing.
- No partial value or list is returned.
- One admission spans traversal, compile, selection, prepare, run, all copying, and cleanup.
  Engine close waits for an admitted call, and concurrent calls retain isolated Runtime states.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- No inventory, Tensor, provenance, storage, compiled result, or prepared result is cached across
  calls. No asynchronous work, retry, fallback, hidden state, or global registry is added.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [asynchronous glossary](../../../../glossary.md#term-asynchronous); [registry glossary](../../../../glossary.md#term-registry))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Compiler bindable-input pruning, liveness normalization, new graph transforms, pass-order
  changes, <span data-glossary-navigation="true"> ([liveness glossary](../../../../glossary.md#term-liveness); [normalization glossary](../../../../glossary.md#term-normalization); [graph glossary](../../../../glossary.md#graph))</span>[`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) remapping, or changes to `CompileConstantPlan`/`CompileArtifacts`.
- Traversal of [compiler](../../../../glossary.md#term-compiler) [IR](../../../../glossary.md#term-intermediate-representation-ir), [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel), final nodes, graph outputs, partitions,
  logical-memory plans, prepared schedules, or Runtime state from Engine.<span data-glossary-navigation="true"> ([nodes glossary](../../../../glossary.md#node); [Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Any execution method on [`Tensor`](../../../../glossary.md#tensor); `Tensor.compute`, `Tensor.execute`, `Tensor.backward`, gradient
  fields, hidden execution state, or a Model/NN-to-Engine dependency.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [Model glossary](../../../../glossary.md#term-model); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- `ModelExecutor`, `Predictor`, another service/facade/manager, implicit process-global Engine,
  cache, reusable hidden prepared state, or automatic model/session ownership.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade); [cache glossary](../../../../glossary.md#term-cache); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Changes to `Model.forward(input)`: it remains pure Tensor [DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) construction.
- Backward convenience, seed/target discovery, gradient policy, Training behavior, optimizer
  work, or Engine 0006 implementation.<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed); [target glossary](../../../../glossary.md#term-target); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- CPU source-only/zero-node execution. This task does not broaden CPU's rejection of zero-node or
  pass-through artifacts and does not implement CPU 0010H.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [artifacts glossary](../../../../glossary.md#term-artifact))</span>
- Cross-backend transfer, mixed-backend schedule composition, tuning, persistence, streaming,
  typed arrays, caller destinations, or another byte-limit policy.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- Changes to Model, Compiler, Config, Planning, Prepare, Runtime, CPU, NN, Training, build files,
  dependencies, architecture files, ADRs, architecture-test source, or backend conformance.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially immutable Tensor provenance,
  Compiler capture/artifacts, Runtime run ownership, and Engine composition<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [Tensor glossary](../../../../glossary.md#tensor); [provenance glossary](../../../../glossary.md#provenance); [Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [artifacts glossary](../../../../glossary.md#term-artifact); [Runtime glossary](../../../../glossary.md#term-runtime); [run glossary](../../../../glossary.md#term-run); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Engine glossary](../../../../glossary.md#term-engine))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Runtime/Prepare/backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
- [ADR 0011](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>

## Architecture constraints

- Model owns immutable Tensor identity, descriptor, and expression provenance. The transient walk
  consumes that public Model contract without moving execution into Model.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [immutable glossary](../../../../glossary.md#term-immutable); [Tensor glossary](../../../../glossary.md#tensor); [identity glossary](../../../../glossary.md#term-identity-meanings); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- Compiler alone owns capture, inference, optimization, liveness, graph-local IDs, and final
  bindable-input classification. Engine consumes final binding metadata without interpreting it.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [liveness glossary](../../../../glossary.md#term-liveness); [graph glossary](../../../../glossary.md#graph); [Engine glossary](../../../../glossary.md#term-engine); [binding glossary](../../../../glossary.md#term-binding); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Engine owns public lifecycle orchestration, Tensor-to-logical-input selection, storage snapshot
  coordination, output materialization, and failure/lifetime behavior.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [snapshot glossary](../../../../glossary.md#term-snapshot); [materialization glossary](../../../../glossary.md#materialization); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), and [CPU](../../../../glossary.md#term-cpu) retain their existing responsibilities and public contracts.
- Tensor/provenance references exist only in local synchronous Engine state and are never retained
  by <span data-glossary-navigation="true"> ([synchronous glossary](../../../../glossary.md#term-synchronous))</span>`CompiledGraph`, [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), [`RunResult`](../../../../glossary.md#run-result--runresult), [Compiler](../../../../glossary.md#term-compiler) [artifacts](../../../../glossary.md#term-artifact), or a [cache](../../../../glossary.md#term-cache).
- Runtime hot-path prohibitions remain intact because provenance traversal ends before prepared
  execution and Runtime sees only already selected borrowed representations.<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>
- No [dependency direction](../../../../glossary.md#java-platform-and-build-tools), [module](../../../../glossary.md#term-module) boundary, authoritative contract, or build structure changes.

## Package impact

Existing package changed:

- `io.github.pho001.synaptik.engine` replaces two ordinary one-shot method shapes and adds only
  package-private implementation support inside the existing lifecycle owner.<span data-glossary-navigation="true"> ([shapes glossary](../../../../glossary.md#shape); [lifecycle glossary](../../../../glossary.md#lifecycle); [owner glossary](../../../../glossary.md#term-owner))</span>

Packages added or moved: none.

Type placement:

- `io.github.pho001.synaptik.engine.Engine` owns the four public overloads because it already owns
  ordinary standard composition and one-shot lifecycle entry.
- `io.github.pho001.synaptik.engine.AdvancedEngine` remains the package-private implementation and
  admission owner used by ordinary Engine. It may own narrow inventory/selection helpers but gains
  no advanced public API.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [Engine glossary](../../../../glossary.md#term-engine); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Existing `CompiledGraph.Input` remains the final ordered ordinary binding metadata; no new
  Engine input-plan type is needed.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [metadata glossary](../../../../glossary.md#term-metadata))</span>

## Affected files

Exact implementation allowlist:

[Engine](../../../../glossary.md#term-engine) production and [Javadoc](../../../../glossary.md#term-javadoc) (three paths):

1. `modules/engine/src/main/java/io/github/pho001/synaptik/engine/Engine.java`.
2. `modules/engine/src/main/java/io/github/pho001/synaptik/engine/AdvancedEngine.java`.
3. `modules/engine/src/main/java/io/github/pho001/synaptik/engine/package-info.java`.

Focused tests (four paths):

4. `modules/engine/src/test/java/io/github/pho001/synaptik/engine/EngineTypedLifecycleTest.java`.
5. `modules/engine/src/test/java/io/github/pho001/synaptik/engine/api/EngineTypedPublicShapeTest.java`.
6. `modules/engine/src/test/java/io/github/pho001/synaptik/engine/api/AdvancedEnginePublicShapeTest.java`.
7. `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineTypedLifecycleIntegrationTest.java`.

Explanatory documentation (four paths):

8. `docs/api/public-api.md`.
9. `docs/api/compile-api.md`.
10. `docs/api/runtime-api.md`.
11. `docs/glossary.md`.

[Planning](../../../../glossary.md#term-planning)/status consistency (four paths):

12. This task.
13. `docs/planning/modules/engine/master-plan.md`.
14. `docs/planning/roadmap.md`.
15. `docs/planning/modules/engine/tasks/0006-one-shot-scalar-objective-backward-convenience.md`.

Review without modification: all other Engine production/tests; [Model](../../../../glossary.md#term-model), [Compiler](../../../../glossary.md#term-compiler), Config,
Planning, [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), [CPU](../../../../glossary.md#term-cpu), [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context), and Training source/tests; [Tensor](../../../../glossary.md#tensor) and Training [API](../../../../glossary.md#term-application-programming-interface-api) pages;
architecture and [ADR](../../../../glossary.md#term-architecture-decision-record-adr) files; [Gradle](../../../../glossary.md#term-gradle)/build files; architecture tests; [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance); Engine
0006 implementation; and CPU 0010H. Record reasoned no-change conclusions in the documentation
pass.

## Maximum [scope](../../../../glossary.md#term-scope)

At most the exact fifteen paths above: three [Engine](../../../../glossary.md#term-engine) production/[Javadoc](../../../../glossary.md#term-javadoc) paths, four focused test
paths, four explanatory documentation paths, and four [planning](../../../../glossary.md#term-planning)/status-consistency paths. No new
production or test file is authorized. If implementation requires another path, inward contract,
[dependency](../../../../glossary.md#term-dependency), or architecture change, stop and replan.

## Acceptance criteria

- [`Engine`](../../../../glossary.md#term-engine) exposes exactly the four specified `compute(...)` overloads and no public/protected
  `forward(...)` overload or compatibility [alias](../../../../glossary.md#aliasing-and-arena).
- The no-limit overloads delegate to bounded behavior with `Long.MAX_VALUE`; individual JVM array
  ceilings and checked byte arithmetic remain enforced.<span data-glossary-navigation="true"> ([JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm))</span>
- The requested closure is traversed iteratively and by exact Tensor object identity. Shared
  subexpressions and repeated leaves are visited safely; distinct equal-descriptor leaves remain
  distinct.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity))</span><span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Only provenance-free leaves are inventoried. Derived Tensors are never passed as caller inputs,
  and parameters/buffers require no special Engine category.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [parameters glossary](../../../../glossary.md#term-parameter); [buffers glossary](../../../../glossary.md#term-buffer))</span>
- `CompiledGraph.inputs()` is the sole ordering and selection authority after compilation.
  Unselected inventory entries are ignored, missing bindings fail, and Engine performs no graph
  liveness inference or Compiler IR traversal.<span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [graph glossary](../../../../glossary.md#graph); [liveness glossary](../../../../glossary.md#term-liveness); [inference glossary](../../../../glossary.md#term-inference); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Storage is read and [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) only for selected [leaves](../../../../glossary.md#term-leaf) and in final [Compiler](../../../../glossary.md#term-compiler) [binding](../../../../glossary.md#term-binding) order.
- Transient Tensor/provenance state does not escape the synchronous call or enter a handle,
  result, field, cache, artifact, or Runtime object.<span data-glossary-navigation="true"> ([synchronous glossary](../../../../glossary.md#term-synchronous); [cache glossary](../../../../glossary.md#term-cache); [artifact glossary](../../../../glossary.md#term-artifact); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Task 0005's one-admission lifecycle, fresh compile/prepare/run, publication/output order,
  aggregate preflight-before-copy, exact-limit behavior, per-value ceiling, copy ordering,
  detached results, cleanup/suppression, close waiting, concurrent isolation, and no-cache rules
  remain covered.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [lifecycle glossary](../../../../glossary.md#lifecycle); [compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run); [publication glossary](../../../../glossary.md#term-publication); [preflight glossary](../../../../glossary.md#term-preflight); [detached glossary](../../../../glossary.md#detached-model-results))</span>
- The advanced and reusable ordinary `run(preparedExecution, explicitInputs)` APIs and behavior
  remain unchanged.
- CPU zero-node/source-only behavior remains unchanged. A provenance-free output may be
  inventoried and compiled, but current CPU preparation may still reject the zero-node artifact.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [node glossary](../../../../glossary.md#node))</span>
- Public-[shape](../../../../glossary.md#shape) tests reject the removed `forward(...)` methods, require all four `compute(...)`
  signatures, and preserve every unrelated ordinary/advanced declaration.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>
- Real `Engine.standard()` integration covers singleton and ordered derived outputs whose
  reachable caller-backed leaves are discovered automatically, shared/repeated leaves, output
  order, exact and failed aggregate bounds, detached access, and caller-owned arenas.<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton); [arenas glossary](../../../../glossary.md#term-arena))</span>
- Javadocs document automatic discovery, transient provenance traversal, Compiler-authoritative
  selection, aggregate byte units, no-limit semantics, ownership, lifecycle, failures, limits,
  and the reusable explicit-input alternative.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Public/Compile/Runtime API and glossary wording clearly distinguish Model expression traversal
  from Compiler IR/liveness and present <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Model glossary](../../../../glossary.md#term-model))</span>`compute` as current only after implementation completes.
- No non-allowlisted path changes. Task/master/roadmap status agree before completion; Engine
  0006 remains Draft and depends on completed 0005A plus CPU 0010H.

## Tests / [validation](../../../../glossary.md#term-validation)

This is task-tier [Engine](../../../../glossary.md#term-engine) validation. During implementation [run](../../../../glossary.md#term-run) focused tests as needed, then run
one final pass after executable Java stabilizes:

```bash
./gradlew :modules:engine:test
./gradlew :testing:integration-tests:test --tests io.github.pho001.synaptik.testing.integration.EngineTypedLifecycleIntegrationTest
./gradlew :testing:architecture-tests:test --tests io.github.pho001.synaptik.testing.architecture.EngineCompositionContractTest
```

Do not run [Compiler](../../../../glossary.md#term-compiler), [Model](../../../../glossary.md#term-model), [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), or [CPU](../../../../glossary.md#term-cpu) suites: this task changes no executable
contract in those [modules](../../../../glossary.md#term-module). Do not run [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance). Repository-wide validation remains
deferred to Engine 0008/[CI](../../../../glossary.md#term-continuous-integration-ci) because dependencies, architecture boundaries, shared build
configuration, and inward contracts do not change.

The distinct clean documentation-focused pass reuses stable Java evidence unless it changes
executable Java behavior, then runs:

```bash
./gradlew :modules:engine:javadoc
git diff --check
```

Also validate generated [Javadoc](../../../../glossary.md#term-javadoc), exact public [signatures](../../../../glossary.md#term-signature) and erasures, distinct-package usage,
local Markdown links/anchors, unique headings, balanced fences, trailing whitespace, LF/final
newlines, terminology, current-versus-planned claims, exact fifteen-path [scope](../../../../glossary.md#term-scope), task/master/
roadmap/Engine-0006 status consistency, absence of stale ordinary `Engine.forward(...)` [API](../../../../glossary.md#term-application-programming-interface-api)
claims, and preservation of all pre-existing dirty paths.

## Dependencies

- [Engine](../../../../glossary.md#term-engine) 0003 typed logical [binding](../../../../glossary.md#term-binding) and explicit-input reusable execution — Complete.
- Engine 0004 [detached](../../../../glossary.md#detached-model-results) [host](../../../../glossary.md#term-host) [materialization](../../../../glossary.md#materialization) — Complete.
- Engine 0005 one-shot forward [lifecycle](../../../../glossary.md#lifecycle) and aggregate [preflight](../../../../glossary.md#term-preflight) — Complete.
- [Compiler](../../../../glossary.md#term-compiler) 0006B3 complete [integration port](../../../../glossary.md#term-integration-port) — Complete.
- Compiler 0006B4 stable final caller-input [`TensorId`](../../../../glossary.md#tensorid) bindings — Complete and sufficient.
- Current [Model](../../../../glossary.md#term-model) [immutable](../../../../glossary.md#term-immutable) [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings)/[provenance](../../../../glossary.md#provenance) contracts — Complete.
- [Runtime](../../../../glossary.md#term-runtime) 0015 and [CPU](../../../../glossary.md#term-cpu) 0010G host-materialization prerequisites — Complete.

No additional Compiler [liveness](../../../../glossary.md#term-liveness) task is a [dependency](../../../../glossary.md#term-dependency). CPU 0010H is not a dependency because this
task neither materializes producerless [constants](../../../../glossary.md#term-constant) nor broadens zero-[node](../../../../glossary.md#node) CPU execution.

This task may execute before CPU 0010H as a user-authorized sequential ordering exception. Its
Java work is Engine-only and changes only the ordinary forward convenience surface. It must not
[run](../../../../glossary.md#term-run) concurrently with CPU 0010H documentation/status finalization because both tasks may update
shared roadmap or explanatory files. Engine 0006 still waits for both completed 0005A and CPU
0010H.

## Follow-up tasks

- Engine 0006 remains Draft. Its backward convenience must reuse this task's transient
  expression-leaf inventory and authoritative compiled-input selection seam and must not accept or
  reconstruct an explicit input list.<span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf))</span>
- CPU 0010H remains separately Ready and owns source-only published-constant physical
  materialization without changing this task's ordinary compute surface.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Ready glossary](../../../../glossary.md#term-ready); [constant glossary](../../../../glossary.md#term-constant); [materialization glossary](../../../../glossary.md#materialization))</span>
- [Engine](../../../../glossary.md#term-engine) 0007 remains Draft for optional [autotuning](../../../../glossary.md#autotuning--model-autotuning) composition.
- Engine 0008 remains Draft for the [lifecycle](../../../../glossary.md#lifecycle) capability [checkpoint](../../../../glossary.md#term-checkpoint).

## Architecture impact

Expected impact: None.

The architecture already makes [Tensor](../../../../glossary.md#tensor) [immutable](../../../../glossary.md#term-immutable) expression state and [Engine](../../../../glossary.md#term-engine) the public [lifecycle](../../../../glossary.md#lifecycle)
orchestrator. A transient pre-[compile](../../../../glossary.md#compile) [Model](../../../../glossary.md#term-model)-expression walk supplies exact caller Tensor objects;
[Compiler](../../../../glossary.md#term-compiler) still exclusively decides final bindable membership/order and retains no Tensor. [Runtime](../../../../glossary.md#term-runtime)
still executes only prepared schedules. No architecture page, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), [dependency](../../../../glossary.md#term-dependency), or [architecture
test](../../../../glossary.md#term-architecture-test) change is required. Stop and report if implementation evidence contradicts this conclusion.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
Work in /Users/phujka/IdeaProjects/Synaptik on Engine task 0005A. Do not use GSD, commit, or push.

Read AGENTS.md, ARCHITECTURE.md, docs/architecture/current-architecture-plan.md,
docs/planning/planning-guide.md, the Engine master plan, and
docs/planning/modules/engine/tasks/0005a-automatic-input-discovery-and-compute-convenience.md in
full. Read the final Engine 0003–0005, Compiler 0006B3–0006B5, Model Tensor/provenance, and current
Engine source/tests/API documentation. Inspect and preserve the complete dirty worktree.

Implement exactly the fifteen-path allowlist. Replace the two explicit-input Engine.forward
overloads with the four specified Engine.compute overloads. Inventory reachable provenance-free
Tensor leaves transiently by exact object identity, compile once, select only matching leaves in
final CompiledGraph.inputs() order, and reuse the existing explicit-input run machinery. Preserve
one admission, fresh lifecycle, publication order, complete aggregate preflight before copying,
cleanup, concurrency, per-value ceilings, and checked arithmetic. Add no Compiler pass, IR
traversal, retained Tensor state, cache, Tensor execution, ModelExecutor, Predictor, backward
behavior, inward change, build/dependency change, or zero-node CPU claim. Stop on architecture or
scope conflict.

Use apply_patch. Run the specified task-tier Java validation once after executable code
stabilizes. Then hand the same diff and exact evidence to a distinct clean documentation-focused
context to finalize Javadocs, API guides, glossary impact, planning evidence, and documentation
validation without repeating stable Java suites unless executable behavior changes. Mark Complete
only after all gates pass.
```

## Local decisions

- Use `compute` because the method performs execution and [materialization](../../../../glossary.md#materialization), while `Model.forward`
  remains pure Tensor expression construction.
- Replace rather than [alias](../../../../glossary.md#aliasing-and-arena) `forward` so the ordinary API does not retain an explicit-input
  one-shot path or confuse graph construction with execution.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [graph glossary](../../../../glossary.md#graph))</span>
- Inventory leaves by [Tensor](../../../../glossary.md#tensor) object [identity](../../../../glossary.md#term-reference-identity), but join to [Compiler](../../../../glossary.md#term-compiler) output by [immutable](../../../../glossary.md#term-immutable) [`TensorId`](../../../../glossary.md#tensorid);
  these are the existing identities owned by each side of the boundary.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span>
- Let `CompiledGraph.inputs()` provide the exact final order and membership. Inventory traversal
  order is deliberately irrelevant.
- Reuse the existing explicit-input run implementation after selection so descriptor, storage,
  borrow, Runtime, and cleanup behavior has one owner.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [descriptor glossary](../../../../glossary.md#term-descriptor); [Runtime glossary](../../../../glossary.md#term-runtime); [owner glossary](../../../../glossary.md#term-owner))</span>
- Keep overloads without a caller budget as direct `Long.MAX_VALUE` delegations rather than adding
  a default-limit object or policy.

## Known limitations

- Every compute call still compiles and prepares from scratch; repeated work should use the
  reusable compile/prepare/run API with explicit inputs.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Current execution/materialization remains CPU-only and limited to supported non-empty prepared
  graphs, data types, static Shapes, and resolved layouts.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [CPU glossary](../../../../glossary.md#term-cpu); [graphs glossary](../../../../glossary.md#graph); [Shapes glossary](../../../../glossary.md#shape); [layouts glossary](../../../../glossary.md#layout))</span>
- CPU zero-node/source-only behavior is unchanged; automatic discovery does not make a leaf-only
  output executable.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [leaf glossary](../../../../glossary.md#term-leaf))</span>
- The aggregate limit covers returned canonical payload bytes only, and each payload remains
  subject to the JVM array ceiling and available heap.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [heap glossary](../../../../glossary.md#term-heap))</span>
- Input associations are snapshotted per selected Tensor, not atomically across leaves, and Engine
  does not synchronize caller payload mutation.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Engine glossary](../../../../glossary.md#term-engine))</span>
- No backward, target/seed discovery, tuning, cache, cross-backend transfer, mixed schedule,
  streaming, typed-array, or persistence behavior is included.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [seed glossary](../../../../glossary.md#term-seed); [cache glossary](../../../../glossary.md#term-cache); [backend glossary](../../../../glossary.md#backend); [persistence glossary](../../../../glossary.md#term-persistence))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Reused stable executable evidence from clean implementation context
  `01a0afef-3f95-78d1-b56e-d44aaeaf002c`; documentation context
  `01a0b0e8-da36-7c71-aaac-96c36f24e09c` changed no Java executable token and therefore did not
  repeat the successful suites:<span data-glossary-navigation="true"> ([token glossary](../../../../glossary.md#term-token))</span>
  - `./gradlew :modules:engine:test` passed 35 tests with 0 failures, 0 errors, and 0 skips.
  - The first focused integration [run](../../../../glossary.md#term-run) had 1 of 5 tests fail because the new [fixture](../../../../glossary.md#term-fixture) used `ADD`,
    for which the current CPU standard composition had no hard-eligible backend owner. Production
    code was unchanged; the fixture was corrected to supported contiguous expressions.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend); [owner glossary](../../../../glossary.md#term-owner); [contiguous glossary](../../../../glossary.md#term-contiguous))</span>
  - Corrected
    `./gradlew :testing:integration-tests:test --tests io.github.pho001.synaptik.testing.integration.EngineTypedLifecycleIntegrationTest`
    passed 5 tests with 0 failures, 0 errors, and 0 skips.
  - `./gradlew :testing:architecture-tests:test --tests io.github.pho001.synaptik.testing.architecture.EngineCompositionContractTest`
    was successful/up-to-date; its report contains 1 passing test.
  - Implementation `git diff --check` passed.
- Documentation context ran `./gradlew :modules:engine:javadoc`; it completed successfully with
  14 actionable tasks (1 executed, 13 up-to-date). Generated `Engine.html` contains exactly four
  `compute(...)` entries and no `forward(...)` entry.
- `javap -public` and `javap -p` confirmed the exact four `Engine.compute(...)` signatures,
  absence of an ordinary public/protected <span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>`forward(...)`, the expected package-private/private
  compute helpers, and the unchanged seven-member advanced surface. A
  distinct-package minimal `Engine.standard()` / `engine.compute(output)` fixture compiled with
  `javac` against the current [Engine](../../../../glossary.md#term-engine) and [Model](../../../../glossary.md#term-model) classes.
- A targeted seven-Markdown checker passed local file links and anchors, generated-anchor
  uniqueness, balanced fences, and LF/final-newline rules. Targeted scans confirmed current-facing
  API/glossary text contains no ordinary <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>`Engine.forward(...)` claim; remaining references are
  historical task-0005 or replacement-instruction evidence. The rejected Compiler liveness
  follow-up does not reappear in the allowed documentation/status set.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [liveness glossary](../../../../glossary.md#term-liveness))</span>
- Exact scope inspection reports all and only the fifteen authorized task paths as the task's
  production/test/documentation/status set. The documentation pass edited only the seven Markdown
  paths listed in the completion summary; all unrelated dirty work was preserved.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- Final [hashes](../../../../glossary.md#term-hash) remained exactly:
  - `Engine.java`: `2df5af3190a7148e73173df62858a25db15f74707435be7e6b161eafd087599c`.
  - `AdvancedEngine.java`: `041d3d98617a725cc7b06dc4ca5eb96e22bc2efa6b180dc770e90fb67e859327`.
  - `package-info.java`: `168d9778d7f0fbb60abb050381a6daf188293ca4435d516156bb40d0fa6cd287`.
  - `EngineTypedLifecycleTest.java`:
    `e70d4a40f3300f33b39dad9231ee2ee0ed43fa54844218ac600f035222a587a8`.
  - `EngineTypedPublicShapeTest.java`:
    `69d022f52aa0f5f6714d823005cdc6051ee3711e506075150f9d749155b777a5`.
  - `AdvancedEnginePublicShapeTest.java`:
    `193721e85d9dd9039ff7b7280d6d5fa79526f346c4c5d82f02ccb9aeaf476a3f`.
  - `EngineTypedLifecycleIntegrationTest.java`:
    `c411353b0a57302d8736dc2dc8a2b5c16c1a4409193e15836e0ca7160dadc09c`.
- Final `git diff --check` passed after documentation stabilization.

## Implementation notes

- Clean implementation context `01a0afef-3f95-78d1-b56e-d44aaeaf002c` replaced the two public
  explicit-input `forward(...)` methods with exactly four `compute(...)` overloads. The no-limit
  forms delegate with `Long.MAX_VALUE`.
- The implementation walks Model Tensor-expression provenance iteratively with exact object
  identity, inventories only provenance-free leaves, joins those leaves by <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity))</span><span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor); [provenance glossary](../../../../glossary.md#provenance))</span>[`TensorId`](../../../../glossary.md#tensorid) to final
  `CompiledGraph.inputs()` membership and order, and then reuses the unchanged explicit-input run
  machinery. No Tensor/provenance state escapes the synchronous call and no cache was added.<span data-glossary-navigation="true"> ([synchronous glossary](../../../../glossary.md#term-synchronous); [cache glossary](../../../../glossary.md#term-cache))</span>
- The first focused integration [run](../../../../glossary.md#term-run) failed one of five tests because its new [fixture](../../../../glossary.md#term-fixture) used `ADD`,
  for which the current CPU standard composition had no hard-eligible backend owner. This was a
  test-only unsupported assumption; production code was unchanged. The fixture was corrected to
  supported contiguous expressions while preserving automatic discovery, repeated/shared leaf,
  output order, aggregate-limit, immutable-result, caller-ownership, and detached-value coverage.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend); [owner glossary](../../../../glossary.md#term-owner); [contiguous glossary](../../../../glossary.md#term-contiguous); [immutable glossary](../../../../glossary.md#term-immutable); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [detached glossary](../../../../glossary.md#detached-model-results))</span>
- Clean documentation context `01a0b0e8-da36-7c71-aaac-96c36f24e09c` independently reviewed the
  final implementation and tests, finalized the Public, Compile, Runtime, glossary, task, master,
  and roadmap text, and changed no Java file. The existing three changed production Javadocs were
  already complete and accurate, so their implementation hashes and executable tokens remain
  unchanged.<span data-glossary-navigation="true"> ([hashes glossary](../../../../glossary.md#term-hash))</span><span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime); [tokens glossary](../../../../glossary.md#term-token))</span>

## Completion summary

- Completed changes: Replaced the ordinary explicit-input one-shot `forward(...)` surface with
  exactly four automatic-discovery `compute(...)` overloads. The implementation inventories
  reachable provenance-free leaves transiently by exact Tensor identity, lets final<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf); [identity glossary](../../../../glossary.md#term-reference-identity))</span><span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
  `CompiledGraph.inputs()` select membership/order by [`TensorId`](../../../../glossary.md#tensorid), and reuses the explicit-input
  lifecycle without Compiler IR traversal, liveness reconstruction, retained Tensor state, or a
  cache.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [Compiler glossary](../../../../glossary.md#term-compiler); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [liveness glossary](../../../../glossary.md#term-liveness); [cache glossary](../../../../glossary.md#term-cache))</span>
- Files changed or created: the exact fifteen-path allowlist. Documentation context edited only
  `docs/api/public-api.md`, `docs/api/compile-api.md`, `docs/api/runtime-api.md`,
  `docs/glossary.md`, this task, `docs/planning/modules/engine/master-plan.md`, and
  `docs/planning/roadmap.md`. This planning consistency correction additionally edits only the
  status/dependency wording in<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [dependency glossary](../../../../glossary.md#term-dependency))</span>
  `docs/planning/modules/engine/tasks/0006-one-shot-scalar-objective-backward-convenience.md`;
  it left all three production Java and four test paths unchanged.
- Tests and validation: Reused 35 passing Engine tests, corrected 5 passing integration tests, and
  1 passing focused architecture test from the implementation context. Documentation context
  passed Engine Javadoc, generated-page inspection, <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Engine glossary](../../../../glossary.md#term-engine); [architecture test glossary](../../../../glossary.md#term-architecture-test); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`javap`, distinct-package compilation,
  Markdown/link/anchor/heading/fence/newline checks, terminology/current-status scans, exact-scope
  and hash checks, and final <span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [scope glossary](../../../../glossary.md#term-scope); [hash glossary](../../../../glossary.md#term-hash))</span>`git diff --check`.
- Documentation-agent review: Clean context `01a0b0e8-da36-7c71-aaac-96c36f24e09c` applied the
  General, API/Javadoc, Planning, and Example profiles. Public API documentation now includes the
  requested one-shot Tensor-expression and conceptual NN Model-expression examples and states
  that <span data-glossary-navigation="true"> ([NN Model glossary](../../../../glossary.md#model-module-buffer-and-scan-distinctions))</span>`Model.forward(input)` constructs the expression while `Engine.compute(output)` executes.
- Documentation impact: Public, Compile, Runtime, and glossary text now documents automatic leaf
  discovery, Compiler-authoritative selection, caller storage ownership, bounded/no-limit byte
  semantics, current CPU limitations, detached results, and the explicit reusable alternative.
  Task/master/roadmap status is synchronized: 0005A is Complete, CPU 0010H remains Ready, and
  Engine 0006 remains Draft.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [CPU glossary](../../../../glossary.md#term-cpu); [detached glossary](../../../../glossary.md#detached-model-results); [Ready glossary](../../../../glossary.md#term-ready))</span>
- Javadoc review: All three changed production Javadocs already provided meaningful type/method
  descriptions and complete parameter, return, failure, lifecycle, ownership, nullability, limit,
  and reusable-alternative contracts. No Javadoc edit was needed, so all Java hashes and executable
  tokens remain exactly those handed off by implementation.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [nullability glossary](../../../../glossary.md#term-nullability); [tokens glossary](../../../../glossary.md#term-token))</span>
- Glossary impact: Updated the existing one-shot-forward entry rather than adding a competing
  term; it now defines the four `compute(...)` overloads and distinguishes Model provenance
  traversal from Compiler IR/liveness.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- No-change conclusions: Training [API](../../../../glossary.md#term-application-programming-interface-api) and [Tensor](../../../../glossary.md#tensor) API already state that `Model.forward(input)`
  constructs ordinary Tensor expressions and does not execute, so no edit was needed. Compile,
  Training, Tensor, provenance, `CompiledGraph`, [`RunResult`](../../../../glossary.md#run-result--runresult), and [`HostTensorValue`](../../../../glossary.md#host-tensor-value--hosttensorvalue) contracts need
  no source change because 0005A composes them without changing their ownership or semantics.
  Architecture/ADRs, build files, dependencies, architecture tests, backend conformance, and all
  Compiler/Prepare/Runtime/CPU/NN source and plans remain unchanged because no boundary or inward
  contract changed. Historical Engine 0005 evidence remains truthful at its completion time.
  Engine 0006 stays Draft; only its status/dependency wording was synchronized to record completed
  0005A and separately Ready CPU 0010H. The rejected Compiler liveness follow-up does not
  reappear; no Compiler liveness repair is required.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Prepare glossary](../../../../glossary.md#prepare); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Unresolved issues: None within task 0005A. Current CPU zero-node/source-only execution remains
  unchanged and is owned by Ready CPU 0010H; scalar-objective backward convenience remains Draft
  Engine 0006.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [scalar glossary](../../../../glossary.md#term-scalar); [objective glossary](../../../../glossary.md#term-objective))</span>
- Follow-up required: None for task 0005A.

Status: Complete
