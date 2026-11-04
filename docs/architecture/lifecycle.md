# [Lifecycle](../glossary.md#lifecycle)

This document explains the [compile](../glossary.md#compile), [prepare](../glossary.md#prepare), [run](../glossary.md#term-run), and training lifecycles defined by [`ARCHITECTURE.md`](../../ARCHITECTURE.md). The contract remains authoritative.

The ordinary public lifecycle is runnable through `Engine.standard()` for fixed [CPU](../glossary.md#term-cpu) [ownership](../glossary.md#term-ownership-meanings) and
through `Engine.builder()` for explicit CPU/[Metal](../glossary.md#term-metal) ownership. Builder preparation composes
single-[owner](../glossary.md#term-owner) or mixed-owner partitions through deterministic owner-indexed representations and
explicit bidirectional transfer for all seven [model](../glossary.md#term-model) data types over exact [rank](../glossary.md#term-rank)-0..16 static
[canonical](../glossary.md#term-canonical-form) or positive-[stride](../glossary.md#term-stride) non-overlapping storage [layouts](../glossary.md#layout). The public Training [module](../glossary.md#term-module) now
layers one [optimizer](../glossary.md#term-optimizer)/session owner over that same [Engine](../glossary.md#term-engine) compile, prepare, run, [publication](../glossary.md#term-publication), and
[materialization](../glossary.md#materialization) path. [CUDA](../glossary.md#term-cuda) execution remains planned.
The [roadmap](../planning/roadmap.md) records delivery status.

The short reading model is [compile versus prepare versus run](../glossary.md#compile-versus-prepare-versus-run):
describe meaning and owners, realize reusable work, then supply this invocation's values.
**Cold** work is setup/[binding](../glossary.md#term-binding) outside the repeated numerical loop; the **[hot path](../glossary.md#term-hot-path)** is the work
repeated during execution. **[Admission](../glossary.md#term-admission)** is permission to enter an [operation](../glossary.md#operation) after lifecycle and
input checks. A **[preflight](../glossary.md#term-preflight)** checks the facts available before a later [side effect](../glossary.md#term-side-effect); **[fail-closed](../glossary.md#term-fail-closed)**
means an unsupported or unproved case rejects rather than guessing a route or silently changing it.

### Participants at the public boundary

The application calls `Model.compute` or the explicit Engine APIs. In the Model workflow, NN's
execution component is the caller of Engine; Engine in turn calls Compiler, shared Prepare and
Runtime. Compiler receives output expressions and descriptors, Prepare receives immutable
recipes, and Runtime receives a prepared schedule plus this invocation's concrete inputs.
These are handoffs of different information, not repeated execution of the same stage.

Input storage is not transferred by compilation. During execution its owner keeps it live and
quiescent until `RunResult.close()`. Model's convenience detaches outputs and closes that result
internally; explicit Engine users do so themselves. Supplied builder parameters are copied into
private Model state before any of these runs. The
[ownership table](../explanation/ownership.md#which-call-borrows-or-copies) separates these cases,
including storage-free leaves and heap Tensors that require no deterministic close.


## State across the [lifecycle](../glossary.md#lifecycle)

Synaptik deliberately separates three kinds of state:

- The **[compile](../glossary.md#compile)-time [graph](../glossary.md#graph)** is the [immutable](../glossary.md#term-immutable) [`CompiledGraphModel`](../glossary.md#compiled-graph--compiledgraphmodel) and its associated `CompileArtifacts`. It contains graph semantics, [backend ownership](../glossary.md#backend-ownership), logical memory requirements, and [publication](../glossary.md#term-publication) [bindings](../glossary.md#term-binding).
- **[Prepared execution](../glossary.md#prepared-execution--preparedexecution)** is immutable reusable runtime-ready state: prepared partitions,
  executable recipes, memory geometry, a schedule, and any immutable persistent prepared
  resources. Its logical recipe is immutable, while its explicit close state controls the finite
  lifetime of those resources.<span data-glossary-navigation="true"> ([ready glossary](../glossary.md#term-ready); [finite glossary](../glossary.md#term-finite); [lifetime glossary](../glossary.md#term-lifetime))</span>
- **Per-[run](../glossary.md#term-run) [mutable](../glossary.md#term-mutable) state** is exactly one [`RunState`](../glossary.md#run-state--runstate) for each active complete logical invocation.
  It covers every backend partition in that heterogeneous run and tracks its logical slots,
  resources, validity, and residency without sharing mutable state with another run.<span data-glossary-navigation="true"> ([backend glossary](../glossary.md#backend); [partition glossary](../glossary.md#partition); [heterogeneous glossary](../glossary.md#term-heterogeneous); [residency glossary](../glossary.md#residency))</span>

The [compiler](../glossary.md#term-compiler) does not create physical [buffers](../glossary.md#term-buffer) or executable units. [Runtime](../glossary.md#term-runtime) does not revisit graph transformations or implementation selection.

## [Compile](../glossary.md#compile) [lifecycle](../glossary.md#lifecycle)

[NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) builder construction precedes [compilation](../glossary.md#compiler-engine-and-execution-stage-names): copy supplied raw [parameter](../glossary.md#term-parameter) [leaves](../glossary.md#term-leaf) once, initialize
fresh automatic layers, then build one ordinary [prototype](../glossary.md#term-prototype) expression through the frozen Sequential.
Cold [Model](../glossary.md#term-model).forward applies that same chain to an exact-[descriptor](../glossary.md#term-descriptor) expression; it creates no [Engine](../glossary.md#term-engine)
and does not change cached outputs. Compute/train compile those fixed outputs once per setup and
reuse positional [bindings](../glossary.md#term-binding) across [batches](../glossary.md#term-batch-machine-learning-and-hyperparameters). The selected default [Conv2d](../glossary.md#convolution--conv2d)/pool/flatten and static
recurrent DAGs persist through the same cold source path below. Unsupported saves touch no file and do not
freeze or invalidate the Model. Definition/[graph](../glossary.md#graph)/load use the bounded source path below. The
[owned Model contract](contracts/extensions-training.md#owned-tensor-models)<span data-glossary-navigation="true"> ([owned Model glossary](../glossary.md#model-module-buffer-and-scan-distinctions))</span> owns this distinction.

Before ordinary compilation, integration callers may use the bounded cold source path defined by
the [Compiler lifecycle contract](contracts/compiler-autograd.md#compile-lifecycle):

```text
ordinary forward Tensor expressions
  -> one raw capture + exact input-aligned borrowed leaves
  -> complete static source verification
  -> optional cold reconstruction with positional descriptor-compatible substitutions
  -> ordinary Tensor expressions -> ordinary compile below
```

For example, [capture](../glossary.md#term-capture-meanings) `x.matmul(w.permute(1,0)).add(b)`, then reconstruct it with a fresh input and
the same [weight](../glossary.md#term-weight) and [bias](../glossary.md#term-bias). Reconstruction builds each stored [occurrence](../glossary.md#term-occurrence) once through its public
[Tensor](../glossary.md#tensor) constructor, so shared operands remain shared and later [autograd](../glossary.md#autograd) sees ordinary [provenance](../glossary.md#provenance).

An **occurrence** is one particular use, not just an [operation](../glossary.md#operation) name: two separate additions are
two [producer](../glossary.md#term-producer) occurrences even if they use the same formula. **Provenance** records which producer
and output position created a derived Tensor. Capture translates these source relationships into
graph-local [identities](../glossary.md#term-identity-meanings); it does not calculate the Tensor's values.

A replacement can itself be a produced expression; its upstream computation remains attached and
is checked by later ordinary compile/autograd, not widened source [admission](../glossary.md#term-admission). Position associates
inputs, so exchanging same-descriptor weights cannot be detected.

The exact [Compiler source slice](contracts/compiler-autograd.md#compile-lifecycle) includes the
original eight kinds, the selected default [CNN](../glossary.md#term-convolutional-neural-network-cnn)/static recurrent compositions and fixed flat scans.
The [owning v3 contract](contracts/extensions-training.md#canonical-graph-model-version-3) preserves
v1/v2 bytes for old sources and defines multi-output [normalization](../glossary.md#term-normalization). For example, a static [GRU](../glossary.md#term-gated-recurrent-unit-gru)
remains its SELECT/[GATHER](../glossary.md#gather)/[SLICE](../glossary.md#slice)/CONCAT and arithmetic [DAG](../glossary.md#term-directed-acyclic-graph-dag), whereas an advanced RecurrentScan remains
one producer. Reconstruction calls each public family constructor once and retains every [canonical](../glossary.md#term-canonical-form)
output wrapper, including siblings that are not roots. Cold scan [persistence](../glossary.md#term-persistence) grants no execution.
It neither optimizes nor closes raw [layouts](../glossary.md#layout). An optimized, layout-closed [artifact](../glossary.md#term-artifact) graph is not a
reconstructible source merely because it has the same operations. The [borrowed](../glossary.md#ownership-borrowing-and-lease) source sidecar
owns no storage or [close lifecycle](../glossary.md#ownership-borrowing-and-lease); it is not retained in graph/artifact state. Only after complete
[preflight](../glossary.md#term-preflight) does reconstruction use a call-local [identity](../glossary.md#term-identity-meanings) table, discarded on return or failure.
No storage read/allocation, [backend](../glossary.md#backend) discovery, preparation, execution or [runtime](../glossary.md#term-runtime) graph is added.
See the [cold source API example](../reference/compiler/cold-source.md#current-cold-source-capture-and-reconstruction)<span data-glossary-navigation="true"> ([API glossary](../glossary.md#term-application-programming-interface-api))</span>.

```text
forward Tensor outputs
  -> if backward is requested:
     - fail-closed operation/attribute/policy preflight
     - reverse accumulation through ordinary public Tensor operations
     - combined forward + gradient Tensor expression DAG
  -> one phase-aware GraphCapture
  -> topological sort
  -> producer/use index
  -> shape and data type inference
  -> validation
  -> canonicalization
  -> combined-graph optimization
     - dead-code elimination
     - phase-local common subexpression elimination
     - constant folding
     - algebraic simplification
  -> final validation
  -> publication binding
  -> backend intent propagation
  -> capability analysis
  -> partition scoring
  -> ownership decision
  -> maximal same-owner partitioning
  -> logical memory/materialization requirements
  -> CompileArtifacts
```

`CompileArtifacts` packages the [immutable](../glossary.md#term-immutable) graph, planned partitions, [logical memory plan](../glossary.md#logical-memory-plan), [publication plan](../glossary.md#publication-plan), and [compile diagnostics](../glossary.md#compile-diagnostics). Compile does not create prepared schedules, executable units, physical [buffers](../glossary.md#term-buffer), runtime workspaces, concrete [kernel](../glossary.md#kernel) routes, or backend-specific executable graphs.

The [compiler](../glossary.md#term-compiler) vocabulary describes different structural jobs. A **[topological order](../glossary.md#term-topological-order)** places
producers before their consumers. **[Canonicalization](../glossary.md#occurrence-identity-and-canonical-form)** chooses a consistent representation,
including graph-local numbering, without being numerical normalization. **[Dead-code elimination](../glossary.md#term-dead-code-elimination-dce)**
removes unneeded computation; **common subexpression elimination** shares eligible repeated
expressions; **[constant folding](../glossary.md#term-constant-folding)** evaluates admitted compile-time [constants](../glossary.md#term-constant). Their exact guards,
not the names alone, determine which changes preserve meaning.

Before [publication](../glossary.md#term-publication) and [planning](../glossary.md#term-planning), final static-result closure resolves proven materialized logical
results, then propagates exact input-relative [views](../glossary.md#term-view-meanings) in topological order. [Scalar SELECT](../glossary.md#scalar-select) uses the
existing indexing [inference](../glossary.md#term-inference-meanings); positive-step [coordinate](../glossary.md#axis-rank-scalar-vector-and-matrix) SLICE, [target-relative crop](../glossary.md#target-relative-crop), and [shape](../glossary.md#shape)/[axis](../glossary.md#term-axis)
views use layout inference. For example, a `[2,6]` [matrix](../glossary.md#term-matrix-meanings) product followed by a two-column gate
slice retains row [stride](../glossary.md#term-stride) `6`, including a nonzero column-start offset, rather than acquiring a
[contiguous](../glossary.md#term-contiguous) copy. Negative-step, empty, dynamic, and otherwise unproved extractions stay unresolved.
After exact view inference, fully static validated RESHAPE with resolved noncontiguous input and
an unproved view may instead select canonical non-view logical output. This permits downstream
view propagation without changing raw source inference or inserting a consumer copy. Existing
resolved/offset views stay exact; physical [materialization](../glossary.md#materialization) remains separately backend-qualified.
These descriptor facts leave physical storage, [lowering](../glossary.md#lowering), and execution with [Prepare](../glossary.md#prepare) and backends;
they do not widen the separate cold source boundary above.

Autograd preflight checks every backward-reachable producer occurrence and its exact attributes
before constructing a [derivative](../glossary.md#term-derivative) expression. The compiler then uses exact Tensor [identity](../glossary.md#term-reference-identity) only
for temporary contribution accumulation and combines contributions with ordinary `Tensor.add`.
[Seeds](../glossary.md#commit-release-distribution-log-and-seed-meanings) and derivative constants are storage-free leaves registered explicitly as logical splats.
The one capture call receives the original forward-producer identity set so it can assign
`FORWARD` or `BACKWARD` per [node](../glossary.md#node) while assigning graph-local IDs only once. Full inference happens
after capture; a failure at that or a later stage may therefore consume temporary Tensor IDs,
which remain [opaque](../glossary.md#term-opaque) and non-reusable.

[Partition](../glossary.md#partition) scoring chooses [backend ownership](../glossary.md#backend-ownership), not concrete implementation. See [Partition Scoring](partition-scoring.md).

## [Prepare](../glossary.md#prepare) [lifecycle](../glossary.md#lifecycle)

```text
CompileArtifacts
  -> validate partition coverage
  -> project partition-scoped facts and resolved prepare inputs
  -> backend analysis, lowering, specialization, fusion, and kernel selection
  -> exact shared buffer/workspace declarations
  -> build BackendPartitionAnalysis[]
  -> assign stable slots and build PreparedMemoryPlan
  -> backend finalization against assigned slots
  -> build PreparedPartition[] and PreparedExecutable[]
  -> build PreparedSchedule
  -> validate prepared memory and schedule
  -> PreparedExecution
```

Prepare creates [`BackendPartitionAnalysis`](../glossary.md#backend-partition-analysis-backendpartitionanalysis), [`PreparedPartition`](../glossary.md#prepared-partition--preparedpartition), [`PreparedExecutable`](../glossary.md#prepared-executable--preparedexecutable),
[`PreparedMemoryPlan`](../glossary.md#prepared-memory-plan--preparedmemoryplan), [`PreparedSchedule`](../glossary.md#prepared-schedule--preparedschedule), and [`PreparedExecution`](../glossary.md#prepared-execution--preparedexecution). Shared
Prepare owns projection, orchestration, exact resource declarations, slot assignment, and
[validation](../glossary.md#term-validation). Concrete [backends](../glossary.md#backend) own deterministic analysis/[lowering](../glossary.md#lowering)/route choice, retain the
selected plan opaquely, and construct executables only during finalization after slot assignment.
Finalization may acquire an [immutable](../glossary.md#term-immutable) persistent [prepared resource](../glossary.md#prepared-resource--preparedresource), such as a compiled native
executable, only through the transactional [ownership](../glossary.md#term-ownership-meanings) [handoff](../glossary.md#term-handoff): the backend rolls back before
return, Prepare rolls back after return, and the completed `PreparedExecution` becomes the sole
[owner](../glossary.md#term-owner). [Runtime](../glossary.md#term-runtime)'s owner and [lease](../glossary.md#term-lease) lifecycle are current. The finalizer return and Prepare
[transaction](../glossary.md#term-transaction) described here are current through Prepare 0006; current [CPU](../glossary.md#term-cpu) preparation contributes
an empty persistent-resource list.

The resulting Runtime `PreparedExecution` retains the exact assigned memory plan and exact
same-plan schedule. `PreparedSchedule` is an ordered list of step occurrences: optional
representation creation is first, executable and explicit [buffer](../glossary.md#term-buffer)-transfer occurrences follow,
and publications form the dense final suffix. Repeating an executable or transfer [occurrence](../glossary.md#term-occurrence)
means repeated work, not another ownership occurrence. No additional aggregate sits between a
schedule occurrence and its recipe.

See [Runtime, Prepare, and Backend Boundary](runtime-prepare-backend-boundary.md) for the exact ownership split.

## [Run](../glossary.md#term-run) [lifecycle](../glossary.md#lifecycle)

```text
PreparedExecutionRunner.run(PreparedExecution, callerInputs)
  -> acquire one prepared-execution run lease
  -> create one RunState for this complete logical run
     - bind caller inputs as borrowed representations
     - invoke the optional first representation-creation recipe
  -> cold-bind every executable, transfer, and publication occurrence
  -> traverse the PreparedSchedule.Step occurrences in order
     - execute prepared executable occurrences
     - perform explicit buffer-transfer/materialization occurrences
     - publish the dense final suffix
  -> RunResult
     - owns the complete RunState lease until close
  -> release the prepared-execution run lease before the synchronous call returns
```

The stateless [`PreparedExecutionRunner`](../glossary.md#prepared-execution-runner--preparedexecutionrunner) owns [synchronous](../glossary.md#term-synchronous) run orchestration and creates one
isolated [`RunState`](../glossary.md#run-state--runstate) per call. Run executes prepared work. It must not perform [graph optimization](../glossary.md#forward-graph-optimization),
[autograd](../glossary.md#autograd) construction,
[compiler](../glossary.md#term-compiler) passes, [backend](../glossary.md#backend) discovery, backend-specific [lowering](../glossary.md#lowering), or [kernel](../glossary.md#kernel) selection. [Runtime](../glossary.md#term-runtime) owns
logical per-[run state](../glossary.md#run-state--runstate) and cleanup orchestration; concrete backends own physical representation
classes and their allocation, [release](../glossary.md#term-release), transfer, and access mechanics.

The cold [binding](../glossary.md#term-binding) step is the only boundary where [heterogeneous](../glossary.md#term-heterogeneous) backend representation types are
checked dynamically. Executable and transfer binding create backend-owned typed objects with
direct references; Runtime [publication](../glossary.md#term-publication) binding retains the selected representation directly.
Execution therefore needs no map [lookup](../glossary.md#term-lookup), reflection, string [dispatch](../glossary.md#term-dispatch), [graph](../glossary.md#graph) inspection, service
lookup, or repeated unsafe cast.

Each synchronous run first acquires a [lease](../glossary.md#term-lease) on its [prepared execution](../glossary.md#prepared-execution--preparedexecution). Once close begins, no new
lease is admitted. Existing runs finish without close waiting for them; the last lease performs
deferred reverse cleanup when close raced an active run. Thus close is deterministic without an
unbounded wait, and the prepared recipe never moves native [compilation](../glossary.md#compiler-engine-and-execution-stage-names) onto the run path. This
Runtime lifecycle is current. [Engine](../glossary.md#term-engine) prepared handles now own exactly one inward Runtime
execution. Their outward [synchronization](../glossary.md#term-synchronization) covers delegate retrieval and the inward close
transition, not a complete run; a delegate-first run still arbitrates at Runtime's unique lease
authority. Once outward closure is observable, no later retrieval can admit work. Engine adds
neither another lease nor a waiting close.

## Current public [Engine](../glossary.md#term-engine) [lifecycle](../glossary.md#lifecycle)

The ordinary reusable prepared path is:

```text
Tensor outputs -> engine.compile(...) -> engine.prepare(...) -> prepared owner
exact-ID inputs in any order -> engine.run(prepared, inputs) -> leased RunResult
fresh values in compiled order -> engine.runValues(prepared, values) -> leased RunResult
exact publication occurrence -> result.materialize(...) -> detached HostTensorValue
```

One Engine prepared handle borrows the composition [lifetime](../glossary.md#term-lifetime) and owns its exact inward preparation;
[Runtime](../glossary.md#term-runtime) alone owns persistent-resource and [run](../glossary.md#term-run)-[lease](../glossary.md#term-lease) authority. The final `CompiledGraph.inputs()`
list authenticates [identity](../glossary.md#term-identity-meanings) membership for run and positional order for runValues. Fresh values
require fully static exact [descriptors](../glossary.md#term-descriptor), including [layout](../glossary.md#layout) presence and requiresGrad. Neither path
mutates [prototypes](../glossary.md#term-prototype), recompiles, prepares, queries providers/registries or adds per-[element](../glossary.md#term-element) work.
Before any borrow, both [snapshot](../glossary.md#term-snapshot) associations once and validate storage. runValues additionally
uses checked [canonical](../glossary.md#term-canonical-form) static element/byte geometry for absent layouts, or checked referenced spans
for resolved layouts. Unknown/[overflow](../glossary.md#term-overflow) geometry rejects only positional eligibility.

Both calls check Engine [admission](../glossary.md#term-admission), prepared nullness/exact [owner](../glossary.md#term-owner) and exact inward delegate closure
before inputs. This intentionally changes former direct-run precedence: closed preparation wins
null/malformed/dead inputs with `prepared execution is closed` and zero borrows. Runtime still
arbitrates any later close/run [race](../glossary.md#term-race) without an outward lock held through execution.

Each run has isolated [mutable](../glossary.md#term-mutable) state and a result retaining caller-input borrows until result close.
A prepared close does not wait for already leased runs or close returned results. [Materialization](../glossary.md#materialization)
remains usable while result and Engine are open. [Metadata](../glossary.md#term-metadata) survives close; [detached](../glossary.md#detached-model-results) values survive
result/prepared/Engine/input-storage closure. Engine shutdown waits for admitted operations, then
closes results, ordinary/advanced preparations and integrations in their reverse [publication](../glossary.md#term-publication) or
registration order, with [rollback](../glossary.md#term-rollback), suppression and failure replay.

[`InferenceSession`](../glossary.md#former-inference-session--inferencesession)/`Engine.session` are removed; migrate to explicit [prepare](../glossary.md#prepare)/run and recompile.
No new owner/wrapper or lease protocol replaces them.
`Engine.compute(...)` is a different lifetime choice: it discovers reachable expression leaves,
then freshly compiles, prepares, runs, materializes, closes the temporary result, and closes the
temporary preparation during every call. `Engine.backward(...)` does the same for one [scalar
objective](../glossary.md#loss) and explicit [gradient](../glossary.md#term-gradient) [targets](../glossary.md#term-target). Neither convenience installs gradient state on [Tensor](../glossary.md#tensor).

## Public training [lifecycle](../glossary.md#lifecycle)

`TrainingSession.open(engine, parameters, objective, optimizer)` snapshots the live source's
ordered paths and exact [Tensor](../glossary.md#tensor)/storage [identities](../glossary.md#term-identity-meanings), compiles the [scalar
objective](../glossary.md#loss) with those [parameters](../glossary.md#term-parameter) as ordered first-[gradient](../glossary.md#term-gradient) [targets](../glossary.md#term-target) once, and calls
`engine.prepare(compiledGraph)` once. It borrows [Engine](../glossary.md#term-engine) and the live source, owns the resulting
Engine [`PreparedExecution`](../glossary.md#prepared-execution--preparedexecution), a private native [cotangent seed](../glossary.md#primal-adjoint-and-differentiation-seed), primitive [optimizer](../glossary.md#term-optimizer)/accumulation staging
[buffers](../glossary.md#term-buffer), counters, and persistent optimizer slots.

```text
open: snapshot parameters -> Engine.compile once -> Engine.prepare once
run:  bind captured parameters + seed + caller inputs -> Engine.run or Engine.runValues
      -> validate/materialize objective and target-indexed gradients
      -> stage and validate every update -> atomically publish parameters/state/counters
```

Every later [`run`](../glossary.md#term-run) uses the same [compiled graph](../glossary.md#compiled-graph--compiledgraphmodel) and [prepared execution](../glossary.md#prepared-execution--preparedexecution). Training neither
calls [Runtime](../glossary.md#term-runtime)/[Prepare](../glossary.md#prepare) nor reconstructs [publication](../glossary.md#term-publication) mapping: `CompiledGraph.inputs()` supplies
input [identity](../glossary.md#term-identity-meanings)/order and each gradient publication's `targetIndex` maps to the snapshotted
parameter position. Only one Training [operation](../glossary.md#operation) is admitted at a time. Close rejects new work,
waits for admitted Training work, closes the prepared execution and private [seed](../glossary.md#term-seed) [arena](../glossary.md#term-arena) once, and
never closes the [borrowed](../glossary.md#ownership-borrowing-and-lease) Engine, source, or parameter storage. [Detached](../glossary.md#detached-model-results) `TrainingStep` and
`TrainingState` values outlive all of those [owners](../glossary.md#term-owner).

`run` accepts exact external TensorIds in arbitrary order. `runValues` instead takes stored values
in `session.inputs()` order, including targets but excluding captured parameters and the private
seed. Exact full static [descriptors](../glossary.md#term-descriptor) must match; expressions are not executed or rebound. Fresh
storage is borrowed through the [synchronous](../glossary.md#term-synchronous) call and internal result closure only. Both paths
use the same optimizer [transaction](../glossary.md#term-transaction) and reject nonempty overlapping inputs, even repeated values
that generic Engine positional [binding](../glossary.md#term-binding) would allow. For example, a fresh [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) and target can
replace yesterday's input values without replacing the [graph](../glossary.md#graph), parameters or [optimizer history](../glossary.md#term-optimizer-history).

[Checkpoint](../glossary.md#term-checkpoint) does not add a step to this execution lifecycle. Its [model](../glossary.md#term-model) [API](../glossary.md#term-application-programming-interface-api) can save one complete
[quiescent](../glossary.md#term-quiescent) [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) state through public Engine [materialization](../glossary.md#materialization), inspect the bounded file, then load it
into a fully bound recreated target through NN's strict whole-tree installation. The caller owns
[quiescence](../glossary.md#term-quiescence) and versioned configuration/input fingerprints; v1 does not infer source [alias](../glossary.md#aliasing-and-arena) identity.
The [opaque](../glossary.md#term-opaque) [artifact](../glossary.md#term-artifact) API provides [atomic](../glossary.md#resource-transaction-and-rollback) live visibility, not power-loss [durability](../glossary.md#term-durability). [Exact training
continuation](../glossary.md#exact-training-continuation) would additionally
require Training-owned state and the caller's next-input and external-state [handoff](../glossary.md#term-handoff), coordinated
by a separately gated optional adapter rather than by NN or the model-only project. This is a
planned [ownership](../glossary.md#term-ownership-meanings) flow, not a current runnable training [resume](../glossary.md#schema-serialization-and-checkpoint) sequence. See the [checkpoint
boundary decision](../design/decisions/0031-model-checkpoint-extension-boundary.md) and
[artifact envelope decision](../design/decisions/0032-checkpoint-artifact-envelope.md)<span data-glossary-navigation="true"> ([artifact envelope glossary](../glossary.md#term-artifact-envelope))</span> and
[model schema decision](../design/decisions/0033-model-candidate-schema.md)<span data-glossary-navigation="true"> ([schema glossary](../glossary.md#term-schema))</span>. The
[public model checkpoint decision](../design/decisions/0034-public-model-checkpoint.md)
records the bound-target and alias policy.

The [accepted CP0005B whole-model lifecycle](contracts/extensions-training.md#whole-model-definition-and-composition)
is implemented and reviewed (Complete, 2026-10-15), separately from values-only recovery.
Under caller quiescence, NN description and public
values [capture](../glossary.md#term-capture-meanings) may each read state; no combined [snapshot](../glossary.md#term-snapshot) transaction is promised. [Dry sizing](../glossary.md#term-dry-sizing) and
[residual](../glossary.md#scan-residual-port-and-history-demand) checks precede bounded definition encoding/[hash](../glossary.md#term-hash); public capture's projected [preflight](../glossary.md#term-preflight)
enforces outer bounds before Engine work or bulk value copies. Load preflights outer [framing](../glossary.md#term-framing) before
any copy/projection, independently validates configuration/descriptors/alias bytes, reconstructs an
unpublished descriptor-only NN candidate and invokes public values target-load. Only the complete
[heap](../glossary.md#term-heap)-backed eval model is returned, with no Engine/native [owner](../glossary.md#term-owner) or [autograd](../glossary.md#autograd). For example, closing
the source Arena and Engine after successful save cannot invalidate a later loaded model's values;
its forward expression still needs an admitted caller Engine route. Empty identity models retain
semantics without promising zero-operation Engine execution. Default load rejects a stored input
[digest](../glossary.md#term-digest); explicit verified identity must match presence/bytes. File durability policy is unchanged.

The first update domain is [scalar](../glossary.md#term-scalar) [`FLOAT32`](../glossary.md#term-float32)/[`FLOAT64`](../glossary.md#term-float64) objectives and connected, fully static,
dense-[contiguous](../glossary.md#term-contiguous), offset-zero non-[view](../glossary.md#term-view-meanings) parameters in exact-capacity writable shareable native
[host storage](../glossary.md#host-storage). Unsupported Engine preparation/execution—including a [Metal](../glossary.md#term-metal) or mixed plan outside
current [backend](../glossary.md#backend) coverage—fails before optimizer mutation. Training branches on no [backend
identity](../glossary.md#backend-identity--backendid).

`AdvancedEngine.takeOwnership(...)` is the lower-level [CPU](../glossary.md#term-cpu)-only explicit-composition surface. Its
caller supplies one supported CPU integration and transfers ownership to Engine. Ordinary builder
composition can assemble registered CPU/Metal mixed-owner schedules; neither surface discovers
backends.

## Explicit [backend composition](../glossary.md#backend-composition)

The public composition surface is an `AutoCloseable Engine.Builder`. Its concrete
`takeOwnership(...)` overloads accept opened [CPU](../glossary.md#term-cpu) and [Metal](../glossary.md#term-metal) integrations. [Ownership](../glossary.md#term-ownership-meanings) transfers at
each non-null method entry, so the builder—not the caller—owns [rollback](../glossary.md#term-rollback) even when registration
fails. Successful `build()` moves the complete ordered [registry](../glossary.md#term-registry) into one [Engine](../glossary.md#term-engine); unsuccessful
construction and builder close [release](../glossary.md#term-release) accepted integrations in reverse registration order.

```text
backend-owned configuration -> backend opens integration and native state
                            -> Engine.Builder takes ownership
                            -> capture BackendId + capability provider + availability snapshot
                            -> Engine.compile selects one owner identity per operation
                            -> Engine.prepare validates all owners/transfers, then runs one shared
                               Prepare transaction and captures per-occurrence outer adapters
                            -> Engine run/materialize use those captured direct references
                            -> Runtime executes the immutable prepared schedule
```

The capability provider reference and point-in-time availability [snapshot](../glossary.md#term-snapshot) are captured once for
each registration and reused for every [compile](../glossary.md#compile) by that Engine. Equal [backend](../glossary.md#backend) IDs are rejected.
Compile-time [Planning](../glossary.md#term-planning) receives the [immutable](../glossary.md#term-immutable) registration-ordered inputs and records only
[`BackendId`](../glossary.md#backend-identity--backendid) ownership; it receives no Engine registry or live integration.

Ordinary cold preparation requires a non-empty plan and resolves every [partition](../glossary.md#partition) [owner](../glossary.md#term-owner) against the
frozen registry. Before any backend analysis, it rejects a missing owner or unsupported directed
transfer pair. Shared [Prepare](../glossary.md#prepare) then analyzes and finalizes every partition, assigns one logical slot
plus one deterministic representation position per participating owner, collects backend physical
creation contributions, and assembles explicit execution, transfer, and [publication](../glossary.md#term-publication) occurrences.
Current CPU/Metal transfers cover all seven carriers over exact [rank](../glossary.md#term-rank)-0..16 static [canonical](../glossary.md#term-canonical-form) or
positive-[stride](../glossary.md#term-stride) non-overlapping [layouts](../glossary.md#layout) with checked physical spans; there is no retry, owner
substitution, conversion, [heap](../glossary.md#term-heap) staging, or CPU fallback.

One outward prepared Engine handle keeps direct non-owning adapter references in caller-input and
[publication occurrence](../glossary.md#publication-occurrence) order while Engine owns the integrations. [Run](../glossary.md#term-run) uses each input adapter to
wrap caller [host storage](../glossary.md#host-storage), and each result publication uses its captured adapter for backend-owned
[host](../glossary.md#term-host) [materialization](../glossary.md#materialization). Neither [operation](../glossary.md#operation) re-queries the registry. Engine shutdown closes retained
results first, then ordinary and advanced prepared handles in reverse
preparation-publication order, then integrations, so no captured reference can outlive its owner.

`prepareTuned(...)` accepts one eligible single-owner CPU plan or exact [singleton](../glossary.md#term-singleton)-NEG Metal plan.
Metal Phase 1 measures its custom-[kernel](../glossary.md#kernel) and [MPSGraph](../glossary.md#term-mpsgraph) routes; Phase 2 checks one complete candidate
fixed to the selected route. Mixed ownership, multiple partitions, and MPSGraph-only Metal plans
remain ineligible. A safe-[heuristic](../glossary.md#term-heuristic) fallback always prepares through the already-selected adapter;
it never substitutes CPU for Metal or otherwise changes ownership.

Engine shutdown rejects new work, waits for admitted Engine operations, closes retained results in
reverse successful-run publication order, then ordinary and advanced prepared handles in
reverse preparation-publication order, and finally backend integrations in reverse
registration order. Closing a prepared handle rejects later runs without
closing an already returned result. A run that already acquired [Runtime](../glossary.md#term-runtime)'s prepared [lease](../glossary.md#term-lease) may
finish; close does not wait for it, and deferred prepared-resource cleanup occurs when the final
lease [releases](../glossary.md#commit-release-distribution-log-and-seed-meanings). The registry map has no post-prepare execution role, and no capability query,
availability check, reflection, or `ServiceLoader` [lookup](../glossary.md#term-lookup) enters cold-bound Runtime execution.
The Engine-handle adapter calls described above occur outside Runtime.

`Engine.standard()` remains the current CPU-only convenience and constructs its fresh CPU
integration through the same ownership path. Metal owns its configuration and native open
operation; Engine accepts the opened integration and does not parse a [native library](../glossary.md#term-native-library) path or
select a Metal [device](../glossary.md#term-device). See
[ADR 0015](../design/decisions/0015-explicit-engine-backend-composition.md)<span data-glossary-navigation="true"> ([ADR glossary](../glossary.md#term-architecture-decision-record-adr))</span> for the alternatives and
consequences.

## Planned fixed recurrent [scan](../glossary.md#term-fixed-recurrent-scan) through the [lifecycle](../glossary.md#lifecycle)

The fixed recurrent scan is a current ordinary [Model](../glossary.md#term-model) expression whose `INT64[batch]`
valid-length values are modeled as ordinary inputs while every [Shape](../glossary.md#shape) remains fully static. Model
construction and forward-only [Compiler](../glossary.md#term-compiler) adoption are implemented; public execution and [backend](../glossary.md#backend)
realization are not. The complete lifecycle reads left to right as follows:

```text
one flat multi-output TensorProducer
  -> one flat CompiledNode
  -> one ordinary capability and ownership decision
  -> one backend analysis and exact resource declaration
  -> one finalized reusable PreparedExecutable
  -> one cold-bound invocation per RunState
  -> one backend-internal bounded recurrent loop
```

Current Model fixes `RNN_TANH`, `GRU_RESET_AFTER`, and [`LSTM`](../glossary.md#term-long-short-term-memory-lstm), `FORWARD` or `REVERSE`, ordered inputs and
outputs, static [descriptor](../glossary.md#term-descriptor) rules, and dense original-time-aligned zero-filled padded results.
Generic [capture](../glossary.md#term-capture-meanings) preserves the [occurrence](../glossary.md#term-occurrence) as one [node](../glossary.md#node), and current Compiler forward-only [inference](../glossary.md#term-inference-meanings)
independently revalidates its fully static descriptors. Both backward-capable modes reject a
complete forward inventory containing recurrence before [derivative](../glossary.md#term-derivative) [Tensor](../glossary.md#tensor) allocation. Later
[Planning](../glossary.md#term-planning), [Prepare](../glossary.md#prepare), [Engine](../glossary.md#term-engine), [Runtime](../glossary.md#term-runtime), and backend work owns the remaining execution arrows. Planning
will treat the node as one ordinary capability query. Shared Prepare will
project its static facts and assign backend-declared resources without receiving a loop body.
Engine will cold-bind the typed logical inputs, including the runtime length Tensor, and Runtime
will invoke the prepared bound action without interpreting recurrence.

The [concrete backend](../glossary.md#concrete-backend) validates the complete length [vector](../glossary.md#term-vector) before mutating any output
representation. It traverses only each row's valid prefix, writes exact [positive zero](../glossary.md#nan-infinity-and-signed-zero) at padded
[coordinates](../glossary.md#axis-rank-scalar-vector-and-matrix), returns explicit final hidden and LSTM cell states, and skips recurrent arithmetic
for invalid coordinates. `REVERSE` traverses `L[b]-1 .. 0`, not the padded suffix. A zero-length
row returns its initial states semantically; a zero-time input requires all lengths to be zero.

One occurrence and one prepared transition remain [constant](../glossary.md#term-constant) in [graph](../glossary.md#graph) size as `time` grows. A
backend-internal row/time loop does not create nested graph [identity](../glossary.md#term-identity-meanings). Physical active-row
compaction and [backpropagation through time](../glossary.md#term-backpropagation-through-time-bptt) ([BPTT](../glossary.md#backpropagation-through-time-and-recurrent-state)) remain later [owner](../glossary.md#term-owner)-specific decisions. See
[ADR 0012](../design/decisions/0012-fixed-recurrent-scan-without-regions.md)<span data-glossary-navigation="true"> ([ADR glossary](../glossary.md#term-architecture-decision-record-adr))</span>.

Cold semantic [persistence](../glossary.md#term-persistence) is a separate branch before capability/[ownership](../glossary.md#term-ownership-meanings) selection. The
[v3 contract](contracts/extensions-training.md#canonical-graph-model-version-3) permits capture,
copy/save/load and [rebind](../glossary.md#capture-graph-ports-and-reconstruction) as one flat multi-output node. For example, an advanced list Model
declares `[sequence, validLengths, initialHidden]`
as external [prototypes](../glossary.md#term-prototype) and [weights](../glossary.md#term-weight) as named [leaves](../glossary.md#term-leaf), saves without an Engine, then cold-loads and
rebinds fresh exact-descriptor inputs. Only descriptors persist for those inputs, not length or
state prototype bytes. Static time/[batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) may be zero. Descriptor-only [preflight](../glossary.md#term-preflight) never reads length
values or tests `[0,T]`; execution must do that later. A root at finalHidden or LSTM finalCell
retains all [canonical](../glossary.md#term-canonical-form) sibling ports but never serializes a carrier, prepared loop or runtime state.
Trying compute/train on that cold scan is not a persistence workflow and remains unsupported.

## State scenario

Suppose one [compiled graph](../glossary.md#compiled-graph--compiledgraphmodel) is prepared once for [CPU](../glossary.md#term-cpu) and then [run](../glossary.md#term-run) concurrently twice with different
input values. The [immutable](../glossary.md#term-immutable) `CompileArtifacts` and [`PreparedExecution`](../glossary.md#prepared-execution--preparedexecution) are shared. Each invocation
has one distinct [`RunState`](../glossary.md#run-state--runstate), [borrowed](../glossary.md#ownership-borrowing-and-lease) input [bindings](../glossary.md#term-binding), and isolated run-owned [buffers](../glossary.md#term-buffer)/workspaces.
Storing either run's [residency](../glossary.md#residency) in the prepared recipe would mix [mutable](../glossary.md#term-mutable) invocation state into
shared state; selecting another CPU route during either run would repeat a [prepare](../glossary.md#prepare)-time decision
in the [hot path](../glossary.md#term-hot-path).

## Training [lifecycle](../glossary.md#lifecycle)

Before [compilation](../glossary.md#compiler-engine-and-execution-stage-names), an `extensions/nn` [module](../glossary.md#term-module) tree supplies forward behavior. Each module declares
its trainable [`Parameter`](../glossary.md#term-parameter) values and persistent [`Buffer`](../glossary.md#term-buffer) values; `train()` or `eval()` propagates
the selected mode to children. This is not [optimizer](../glossary.md#term-optimizer) work: for example, a [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters)-[normalization](../glossary.md#term-normalization)
layer can select its training or [inference](../glossary.md#term-inference) forward behavior before [gradients](../glossary.md#term-gradient) exist. The generic
[Tensor](../glossary.md#tensor) operations used by that forward pass remain owned by `modules/model`.

The [compile mode](../glossary.md#compile-mode--compilemode) determines how much expression and [graph](../glossary.md#graph) work the [compiler](../glossary.md#term-compiler) constructs:

- `FORWARD_ONLY` compiles forward computation only.
- `FORWARD_AND_BACKWARD` constructs backward Tensor expressions from the original forward
  expression and captures a combined forward and backward graph once.<span data-glossary-navigation="true"> ([backward graph glossary](../glossary.md#backward-graph))</span>
- `TRAINING_STEP` represents the training-step direction in which optimizer updates may also become graph operations.

In the initial backward-capable lifecycle, `FORWARD_AND_BACKWARD` and `TRAINING_STEP` both build
the combined expression before [capture](../glossary.md#term-capture-meanings); `TRAINING_STEP` does not yet add optimizer-update graph
work. Combined optimization and [planning](../glossary.md#term-planning) therefore see the entire [immutable](../glossary.md#term-immutable) graph. [Prepare](../glossary.md#prepare) may
still expose separate forward and backward schedules or one training-step schedule.

See [Training Graph](training-graph.md)<span data-glossary-navigation="true"> ([Training Graph glossary](../glossary.md#training-graph))</span> for the graph [model](../glossary.md#term-model) and optimization rationale.

## [Optimizer](../glossary.md#term-optimizer) and training-step [lifecycle](../glossary.md#lifecycle)

The initial lifecycle keeps the optimizer as a [backend](../glossary.md#backend)-agnostic step after the prepared forward/backward execution:

```text
compile:
  forward Tensor expression DAG
  -> compiler-owned autograd through ordinary Tensor expressions
  -> one phase-aware capture of forward outputs and gradient roots
  -> infer, validate, optimize, and revalidate the immutable combined graph
  -> CompileArtifacts

run:
  forward/backward prepared execution
  -> publish gradients
  -> optimizer.step()
```

A later architecture version may represent the optimizer update in the [graph](../glossary.md#graph), but only after an explicit architecture update where required:

```text
compile:
  forward + backward + optimizer update graph
  -> optimize
  -> partition scoring
  -> CompileArtifacts

prepare:
  backend prepare may fuse optimizer update routes

run:
  training-step schedule
```

In both forms, `modules/training` owns optimizer algorithms and training orchestration over a
generic [live parameter source](../glossary.md#live-parameter-source--trainingparameters). [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) supplies `Module.trainingParameters()` without moving wrapper
[ownership](../glossary.md#term-ownership-meanings) into Training. Live membership [validation](../glossary.md#term-validation) precedes [run](../glossary.md#term-run)/state/restore and every existing
complete-[view](../glossary.md#term-view-meanings) [precommit](../glossary.md#term-precommit) check; Training separately checks exact storage and [finite](../glossary.md#term-finite) values. A
pre-session native copy preserves the same source by changing only storage associations.
Training remains independent of concrete backends. Backend-
specific optimizer execution belongs to backend [prepare](../glossary.md#prepare) and [kernels](../glossary.md#kernel).

## Numerical semantics across the [lifecycle](../glossary.md#lifecycle)

```text
Model semantics -> Planning occurrence query -> CompileArtifacts -> PrepareContext
                -> qualified backend route -> PreparedExecution -> RunState
```

There is no [Engine](../glossary.md#term-engine) numerical selection or cold profile transport. [Model](../glossary.md#term-model) fixes each [operation](../glossary.md#operation)
family's exact and named-arithmetic-site meaning. [Planning](../glossary.md#term-planning) queries provider capability for the
complete [occurrence](../glossary.md#term-occurrence); [Compiler](../glossary.md#term-compiler) preserves guarded rewrites and [immutable](../glossary.md#term-immutable) occurrence facts. [Prepare](../glossary.md#prepare)
uses qualified [backend](../glossary.md#backend) routes and non-profile compatibility [identities](../glossary.md#term-identity-meanings), then [Runtime](../glossary.md#term-runtime) executes the
prepared result without [tolerance](../glossary.md#term-tolerance), [certificate](../glossary.md#formal-proof-certificate-and-trusted-assumptions), route, or capability [lookup](../glossary.md#term-lookup). For example, a
[BFLOAT16](../glossary.md#term-bfloat16) raw move preserves its stored word, while a following named arithmetic primitive may
use its permitted input [DAZ](../glossary.md#term-daz). Neither step changes the other's boundary. [Metal](../glossary.md#term-metal) retains [MPSGraph](../glossary.md#term-mpsgraph)
for eligible [FLOAT32](../glossary.md#term-float32)-only steps and the fixed custom program for low-containing partitions. Its
current program is [schema](../glossary.md#term-schema) 20 with a 124-byte header and tuning [identity](../glossary.md#term-identity-meanings) 36; [CPU](../glossary.md#term-cpu) generator schema
is 70, and native [ABI](../glossary.md#term-application-binary-interface-abi) 7 retains thirteen exports. Schema 19/identity 29 and CPU schema 68 are
historical, incompatible inputs.
