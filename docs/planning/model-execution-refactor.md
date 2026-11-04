# [Model](../glossary.md#term-model) execution and [persistence](../glossary.md#term-persistence) refactor program

## Reading context: two meanings of Model

The [foundational Model module](../reference/tensor/model-overview.md) defines Tensor
expressions and their meaning. The [NN Model object](../reference/nn/owned-model.md) is the
application-facing object with compute, configured training, save/load and close operations.
This completed program connected that workflow to existing compilation, execution and
checkpoint owners; it did not make the foundational module depend on NN.

For a concrete call, the application invokes `model.compute(input)`: Model executes the
captured expression using the current input values and returns a copied host result.
Its private parameter state is not the application's original initialization Tensor.
Read [the ownership walkthrough](../explanation/ownership.md) and
[the complete save/close/load chapter](../tutorials/save-close-load.md) for the current workflow.
The dated phases below preserve the implementation history, not a new task authorization.

## Purpose and status

Coordinate one [model](../glossary.md#term-model)-facing compute/configured-train/save/load workflow for manually constructed
and file-restored [Tensor](../glossary.md#tensor) [graphs](../glossary.md#graph). A [directed acyclic graph](../glossary.md#term-directed-acyclic-graph-dag) ([DAG](../glossary.md#intermediate-representation-dag-and-ssa)) records ordinary Tensor operations,
not Java callbacks. This plan and [ADR0039](../design/decisions/0039-portable-graph-model-persistence.md)
are non-authoritative. All selected phases are Complete, including NN0028 accepted 2026-10-19;
their bounded root/scoped cutovers are enacted. No new task is launched by this completed program.

The separately selected [Checkpoint0009 activation-source successor](extensions/checkpoint/tasks/0009-owned-model-activation-source-and-persistence.md)<span data-glossary-navigation="true"> ([activation glossary](../glossary.md#term-activation))</span>
is Complete, accepted 2026-10-19 after Archimedes final APPROVE and main's integration/native root
gates. It is not another phase of this program. [Checkpoint](../glossary.md#term-checkpoint)/[Compiler](../glossary.md#term-compiler)/[NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) frontiers: None.
The eight-kind/selective-v2 capability is current; the original [slice](../glossary.md#slice) and all phases stay Complete.

P2 incorporates all five product choices accepted on 2026-10-16: common Model-facing tensor
operations with generic Java forward retained; [lazy](../glossary.md#term-lazy-evaluation) model-owned [Engine](../glossary.md#term-engine).standard or explicit [borrowed](../glossary.md#ownership-borrowing-and-lease)
Engine; owned working [weights](../glossary.md#term-weight); targeted source migration; and a fixed explicit first graph [signature](../glossary.md#term-signature).
The prior P2 packet was independently approved; the user subsequently approved removing the thin
[InferenceSession](../glossary.md#former-inference-session--inferencesession) in favor of existing Engine [PreparedExecution](../glossary.md#prepared-execution--preparedexecution) plus [Engine.runValues](../glossary.md#fresh-value-binding--enginerunvalues).
Independent Astra/xhigh review approved the plan amendment. Engine0021 implementation is now
Complete after independent Class C approval and the native-enabled integrated test/[Javadoc](../glossary.md#term-javadoc) gate.
Training0007 is Complete after Pascal's final independent APPROVE and the coordinator gate.
Authorized Training frontier: None. [Compiler0008](modules/compiler/tasks/0008-cold-source-capture-and-reconstruction.md)
is Complete after final independent code/docs APPROVE and the coordinator Class C checkpoint.
Authorized Compiler frontier: None; [Checkpoint0008](extensions/checkpoint/tasks/0008-generic-graph-value-persistence-and-nn-glue-cutover.md)
is Complete after Halley's final APPROVE and coordinator acceptance on 2026-10-17. Selected-program Checkpoint
frontier: None; [Training0008](extensions/training/tasks/0008-fresh-value-training-session-runs.md) is
Complete after Euclid's definitive code/docs APPROVE and main's source-matched root gate (2026-10-19);
[NN0028](extensions/nn/tasks/0028-owned-model-compute-training-and-persistence.md) is Complete, accepted 2026-10-19;
its brief owns definitive review and coordinator-gate evidence. Authorized NN frontier: None.

Launch verification: CP0005D and P1/Engine0020 are Complete, including their recorded independent
reviews and [validation](../glossary.md#term-validation). The coordinator seeded the current-main [planning](../glossary.md#term-planning) packet into HEAD
`9e7e8aa15e0833a1bc91263a7add99a80d88cd92` plus the integrated uncommitted P1 [snapshot](../glossary.md#term-snapshot).
Those inherited changes are not this amendment's diff. The pinned snapshot includes reviewed P2;
all nine allowed documents matched at launch. Coordinator verified main HEAD/base unchanged,
dependencies Complete and no active conflicts. The user then authorized Engine0021 implementation
in a clean context. Its brief records final review and integrated validation. Training0007 is now
Complete, as are Compiler0008, Checkpoint0008, Training0008 and NN0028.
Completed task histories and evidence remain intact.

## Phases and [owner](../glossary.md#term-owner) [DAG](../glossary.md#term-directed-acyclic-graph-dag)

Arrows below are actual capability dependencies, not table-order authorization. Engine0021,
Training0007/0008 and Compiler0008 are Complete; authorized Training frontier is None,
and [Compiler](../glossary.md#term-compiler) frontier is None.
Checkpoint0008 is Complete after final independent approval and the coordinator gate; selected-program [Checkpoint](../glossary.md#term-checkpoint)
frontier: None. Training0008's brief owns Euclid's final APPROVE and main's accepted root-gate evidence;
NN0028 is Complete. All selected program phases are Complete; authorized [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) frontier: None.

| Owner/task | Capability | Depends on / integration order |
|---|---|---|
| P0 CP0005D; P1 Engine0020 | Completed [planning](../glossary.md#term-planning)/review and [backend](../glossary.md#backend)/native [layout](../glossary.md#layout) | Complete; no reopening of numerical/native campaigns. |
| [Engine](../glossary.md#term-engine) [0021](modules/engine/tasks/0021-prepared-execution-fresh-value-binding.md) | Sole ordinary prepared owner, positional fresh values and [atomic](../glossary.md#resource-transaction-and-rollback) consumer migration — Complete | Coordinated authority, independent implementation review and native-enabled integrated gate passed. |
| [Training0007](extensions/training/tasks/0007-generic-live-parameter-admission.md) — Complete | NN-independent live [admission](../glossary.md#term-admission)/native copy, relocation and NN adapter | Final independent APPROVE and coordinator native-enabled gate passed; evidence in brief. |
| [Compiler0008](modules/compiler/tasks/0008-cold-source-capture-and-reconstruction.md) — Complete | Bounded cold raw-source [API](../glossary.md#term-application-programming-interface-api) and root/compiler authority cutover | Final independent code/docs APPROVE and coordinator Class C checkpoint passed; evidence in brief. |
| [Checkpoint0008](extensions/checkpoint/tasks/0008-generic-graph-value-persistence-and-nn-glue-cutover.md) — Complete | Generic [graph](../glossary.md#graph)/value [persistence](../glossary.md#term-persistence), physical split and retained old-v1 NN readers | Engine0021 + Training0007 + Compiler0008 + CP0005B + accepted P2 Complete; Halley final APPROVE and coordinator source-matched test/[Javadoc](../glossary.md#term-javadoc) gate passed; evidence in brief. |
| [Training0008](extensions/training/tasks/0008-fresh-value-training-session-runs.md) — Complete | Fresh stored values in existing [TrainingSession](../glossary.md#training-session--trainingsession).inputs() order, excluding [parameters](../glossary.md#term-parameter)/private [seed](../glossary.md#term-seed) | Engine0021 + Training0007; Checkpoint0008 serial join Complete; Euclid final APPROVE and main source-matched root gate passed; evidence in brief. |
| [NN0028](extensions/nn/tasks/0028-owned-model-compute-training-and-persistence.md) — Complete | Owned ordinary [Model](../glossary.md#term-model) and manual/file integration | Accepted 2026-10-19 after definitive independent approval and coordinator native-enabled root gate; evidence in brief. |

Training0007 and Compiler0008 could be independent code branches, but this packet authorizes no
parallel writers. Parallel group: None. Current common base: main
`97c3bbe7a0ee893a2ba76b5d8e90449c033cadd4`, including all five completed NN0028 prerequisites.
The NN0028 brief records coordinator-supplied predecessor verification and protected user files;
reverify if the base changes. Serial order remains Engine0021 -> Training0007 -> Compiler0008
-> Checkpoint0008 -> Training0008 -> NN0028; it does not add an algorithm [dependency](../glossary.md#term-dependency) on Training.
Conflicts: None active; future shared root/scoped, NN state [lifecycle](../glossary.md#lifecycle) and Training/Checkpoint
consumer cutovers serialize. Integration owner and shared-document owner: model-execution
coordinator. Each worker validates its affected [modules](../glossary.md#term-module) once; each boundary-changing integrated
capability gets one coordinator repository test/Javadoc checkpoint, not duplicate reviewer reruns.
Checkpoint0008's brief records all seven integration fields, Pauli's approval and coordinator [Ready](../glossary.md#term-ready) verification dated
2026-10-17 (source/base/dependencies/conflicts/frontier); main retains exclusive shared-document writer delegation.
Training0008's bounded root/owning Training lifecycle cutover, implementation, independent review
and source-matched coordinator gate are Complete. The accepted [ADR](../glossary.md#term-architecture-decision-record-adr) does not override current
authority. NN0028 established its bounded root/owning NN lifecycle cutover before dependent Java.
Independent review and coordinator acceptance are complete. Model/Training [autotuning](../glossary.md#autotuning--model-autotuning) stays excluded.

Engine0021 delivered prepared [ownership](../glossary.md#term-ownership-meanings) and fresh [binding](../glossary.md#term-binding) with mechanical Training/benchmark/test
migration. Training0007 completed generic live admission, native copy and the atomic reverse-edge/
consumer cutover. Its brief records final approval and coordinator [validation](../glossary.md#term-validation). It did not wait on recurrent recipes,
fixed-[scan](../glossary.md#term-cumulative-scan) [backpropagation through time](../glossary.md#term-backpropagation-through-time-bptt) ([BPTT](../glossary.md#backpropagation-through-time-and-recurrent-state)), Data or all [operation](../glossary.md#operation) codecs.

## Selected public surface

[Model](../glossary.md#term-model) declarations are implemented in completed NN0028; Checkpoint0008 implements the
bounded GraphCheckpoint boundary and physical/[NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) split, now Complete after final review/coordinator gate. Engine0021 delivered
[prepared execution](../glossary.md#prepared-execution--preparedexecution), and Training0007 implements only generic live [admission](../glossary.md#term-admission)/native copy plus NN's
adapter. Exact responsibilities, failures,
[ownership](../glossary.md#term-ownership-meanings) and examples are in [ADR0039's API decision](../design/decisions/0039-portable-graph-model-persistence.md#selected-api-and-type-feasibility)<span data-glossary-navigation="true"> ([API glossary](../glossary.md#term-application-programming-interface-api))</span>.

- Preserve `Model<I,O>`, `Model.define`, `ModelDefinition`, `ModelForward` and `O forward(I)`.
  Add tensor-only instance `compute(List<Tensor>) -> List<HostTensorValue>`,
  `train(List<Tensor>, List<Tensor>) -> TrainingStep`, configuration, save and close.
  Return types do not masquerade as arbitrary `O`.
- Normal `Model.defineTensor(inputSignature, definition)` and `Model.load(Path)` return
  `Model<Tensor,Tensor>`, with `compute(Tensor)` and `train(Tensor,Tensor)` conveniences. The factory
  captures forward/state once and discards that callback; symbolic forward reconstructs the owned
  source only on an explicit cold call. No user leaf inventory or second Model is required.
  Advanced <span data-glossary-navigation="true"> ([callback glossary](../glossary.md#term-callback))</span>`Model.graph(inputs, outputs, namedLeaves)` and `Model.loadGraph(Path)` return
  `Model<List<Tensor>,List<Tensor>>`. Private implementations supply the tensor capability;
  existing arbitrary Java models remain forward-only unless they opt into an explicit tensor boundary. No unchecked cast, mandatory public subtype or Java-record codec.<span data-glossary-navigation="true"> ([codec glossary](../glossary.md#term-codec))</span>
- `configureTraining(targetPrototypes, objectiveBuilder, optimizer)` fixes a scalar objective
  once over owned outputs and target prototypes; <span data-glossary-navigation="true"> ([scalar objective glossary](../glossary.md#loss); [target glossary](../glossary.md#term-target); [prototypes glossary](../glossary.md#term-prototype))</span>`train(inputs, targets)` means RESET_AND_STEP.
  Existing zero-argument <span data-glossary-navigation="true"> ([argument glossary](../glossary.md#term-argument))</span>`Module.train()` remains only a mode toggle.
- `Model.load(path[, limits]).withEngine(engine)` and application-built models use the same
  `Model<I,O>.withEngine(Engine) -> Model<I,O>`, returning the exact receiver, not a copied
  model/request/executor. It borrows that Engine for compute and training; existing explicit
  borrowed-Engine factory/load overloads remain. <span data-glossary-navigation="true"> ([borrowed glossary](../glossary.md#ownership-borrowing-and-lease))</span>`compute(batch)` is terminal and [detached](../glossary.md#detached-model-results).
- Engine0021 removed [`InferenceSession`](../glossary.md#former-inference-session--inferencesession)/`Engine.session` and their session-only delegates, retaining
  existing Engine [`PreparedExecution`](../glossary.md#prepared-execution--preparedexecution) as sole ordinary public prepared [owner](../glossary.md#term-owner). It implements
  `Engine.runValues(PreparedExecution, List<Tensor>) -> RunResult` in compiled input order.
  `Engine.run(prepared, inputs)` keeps exact-TensorId/arbitrary-order matching; both methods adopt
  former session-strength close admission, an explicit direct-run failure-precedence change.<span data-glossary-navigation="true"> ([TensorId glossary](../glossary.md#tensorid); [run glossary](../glossary.md#term-run))</span>
- Generic Training adds [`TrainingParameters`](../glossary.md#live-parameter-source--trainingparameters) (ordered entries plus live validation);
  NN adds <span data-glossary-navigation="true"> ([validation glossary](../glossary.md#term-validation))</span>`Module.trainingParameters()` as the adapter factory, not another public adapter class.
- Generic [Checkpoint](../glossary.md#term-checkpoint) adds `checkpoint.graph.GraphCheckpoint` with nested `Source`, [`Leaf`](../glossary.md#term-leaf),
  and `Role`: data-only ordinary Tensor boundary/roles for persistence and cold copying.
  Compiler0008 implemented only its bounded source capture/reconstruction boundary, not another
  graph representation; Checkpoint0008's bounded generic source API is Complete after final gates.<span data-glossary-navigation="true"> ([capture glossary](../glossary.md#term-structural-capture))</span><span data-glossary-navigation="true"> ([persistence glossary](../glossary.md#term-persistence); [graph glossary](../glossary.md#graph))</span>

The costs are explicit: a [tensor](../glossary.md#tensor)-capability check on generic Model methods, detached tensor rather
than arbitrary-record execution results, one live Training interface, and one Checkpoint source
aggregate for the NN-free [module](../glossary.md#term-module) boundary. None is a second model/runner hierarchy. No new
InferenceResult is needed: current [Engine](../glossary.md#term-engine) has [RunResult](../glossary.md#run-result--runresult) and detached [HostTensorValue](../glossary.md#host-tensor-value--hosttensorvalue).

## [Ownership](../glossary.md#term-ownership-meanings) and source [lifecycle](../glossary.md#lifecycle)

[Graph](../glossary.md#graph) construction fixes ordered input [prototypes](../glossary.md#term-prototype), ordinary output expressions and every reachable
[PARAMETER](../glossary.md#term-parameter)/[BUFFER](../glossary.md#term-buffer)/[CONSTANT](../glossary.md#term-constant) [leaf](../glossary.md#term-leaf) before compute. Save-before-first-compute therefore works; an unknown
Java forward is not captured implicitly by save. Load already has this source/[signature](../glossary.md#term-signature) and opens
no [Engine](../glossary.md#term-engine), performs no discovery and needs no future [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters).

External supplied [weights](../glossary.md#term-weight) are copied once by [identity](../glossary.md#term-identity-meanings) into private leaves and the source [DAG](../glossary.md#term-directed-acyclic-graph-dag) is
reconstructed against those leaves before the [model](../glossary.md#term-model) is published. Never attach copied storage to
the caller's original [Tensor](../glossary.md#tensor) or retain an output graph still referring to it. Model-owned/loaded
values need no second ownership-only copy. The first [API](../glossary.md#term-application-programming-interface-api) does not need an exclusive-transfer
overload; an explicit future transfer can avoid acquisition copying but must transfer the real
[lifetime](../glossary.md#term-lifetime) [owner](../glossary.md#term-owner), not assert ownership of a [borrowed](../glossary.md#ownership-borrowing-and-lease) Tensor.

[Inference](../glossary.md#term-inference), training and save all read the same owned working leaves. [Heap](../glossary.md#term-heap)-to-native [admission](../glossary.md#term-admission) may
perform one exact storage copy before [training session](../glossary.md#training-session--trainingsession) creation; it is not a second working model
or per-batch copy. Close old inference preparations/[TrainingSession](../glossary.md#training-session--trainingsession) before storage replacement,
retain Tensor [aliases](../glossary.md#aliasing-and-arena), rebuild [bindings](../glossary.md#term-binding) explicitly, and preserve native-only admission and
[atomic](../glossary.md#resource-transaction-and-rollback)/[precommit](../glossary.md#term-precommit) checks. Failed admission leaves prior bytes/bindings intact; failed preparation
publishes no handle.

New graph signatures are fully static: [32,784] accepts fresh compatible [32,784], rejects [64,784].
No automatic recapture, [compilation](../glossary.md#compiler-engine-and-execution-stage-names) or preparation on mismatch. Mode or parameter/wrapper/storage
binding changes invalidate cached execution and fail before use; no silent stale graph. Explicitly
construct a replacement graph model for changed source/signature. Same-storage trained byte updates
do not invalidate it. Existing ModelSpecification/DeclarativeModel/v1 flexible [rank](../glossary.md#term-rank)/batch semantics
remain supported and are not silently tightened.

Normal Engine creation is [lazy](../glossary.md#term-lazy-evaluation) through Engine.standard(), currently [CPU](../glossary.md#term-cpu)-only; no discovery/global
default. [Target](../glossary.md#term-target) withEngine and explicit Engine overloads borrow and never close the Engine.
Binding freezes at the first compute/configureTraining preparation setup after [argument](../glossary.md#term-argument)/source
checks and before Engine creation/native admission/[compile](../glossary.md#compile)/[prepare](../glossary.md#prepare); failed setup remains frozen.
Before freeze an idle open model can change its borrowed selection; after freeze even the same
Engine rejects. Lifecycle admission precedes arguments; null and closed candidates reject without
changing selection. The [exact error order and retry policy](../design/decisions/0039-portable-graph-model-persistence.md#target-model-engine-binding)
also covers configureTraining after inference: its explicit storage transition uses the same Engine,
never rebinding or automatic reprepare after rebinding. Load/save alone create no Engine.
Model.close rejects new operations, waits
for admitted work, closes TrainingSession/inference preparations, owned storage and its created Engine with
attempt-all cleanup. Compute returns [detached](../glossary.md#detached-model-results) values and internally closes [RunResult](../glossary.md#run-result--runresult); values and
TrainingStep survive model close. Low-level RunResult still borrows inputs until result close and
is closed by Engine shutdown, not by prepared close. High-level Model compute/train hides preparation;
no new session/executor/runner [facade](../glossary.md#term-facade) replaces the removed type. TrainingSession remains because
[optimizer](../glossary.md#term-optimizer)/accumulation state, counters, busy admission and atomic [commit](../glossary.md#term-commit) are real Training ownership.

First high-level lifecycle admits one compute/train/save/configure/withEngine [operation](../glossary.md#operation) at a time (busy rejects);
close waits. External callers must exclude mutation/closure of input storage for the whole call and
must not mutate exposed working state while the model is usable. Training additionally forbids
input/state overlap and retains all live path/order/wrapper/Tensor/storage checks before commit.
This is not a promise to make inherited [Module](../glossary.md#term-module) mutators [thread-safe](../glossary.md#term-thread-safe).

## Current-to-[target](../glossary.md#term-target) map

P1 locations, Training0007's physical/[admission](../glossary.md#term-admission) migration and Checkpoint0008's generic/[NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) split
below are Complete, including Checkpoint0008's independent approval/coordinator [validation](../glossary.md#term-validation). NN0028's
selected [Model](../glossary.md#term-model) [lifecycle](../glossary.md#lifecycle) is also Complete. Historical [planning](../glossary.md#term-planning)
directories and completed evidence stay where they are.

| Source/symbols | Selected target and migration |
|---|---|
| Old [backends](../glossary.md#backend)/* and native/[metal](../glossary.md#term-metal)-macos-[arm64](../glossary.md#native-packaging-and-platform-vocabulary) | P1 completed: [modules](../glossary.md#term-module)/backend/{cpu,metal,cuda,openblas-provider}, Metal native at modules/backend/metal/macos-arm64; :backends:* [identities](../glossary.md#term-identity-meanings)/packages/[ABI](../glossary.md#term-application-binary-interface-abi) unchanged. |
| Removed [InferenceSession](../glossary.md#former-inference-session--inferencesession) / [Engine](../glossary.md#term-engine).session / session.[run](../glossary.md#term-run) | Engine0021 delivered Engine.[prepare](../glossary.md#prepare) → existing [PreparedExecution](../glossary.md#prepared-execution--preparedexecution); Engine.run(prepared, exactInputs), or explicit [Engine.runValues](../glossary.md#fresh-value-binding--enginerunvalues)(prepared, compiledOrderValues); no compatibility [facade](../glossary.md#term-facade). |
| [TrainingSession](../glossary.md#training-session--trainingsession)'s former inferenceSession field/open/run/isClosed/close | Engine0021 migrated to Engine PreparedExecution/prepare/run/isClosed/close; public Training [API](../glossary.md#term-application-programming-interface-api) and [optimizer](../glossary.md#term-optimizer) [transaction](../glossary.md#term-transaction) unchanged. |
| extensions/training | Training0007 moved to modules/training; retained `:extensions:training` via projectDir and existing `training` Java package. |
| TrainingSession / TrainingParameter / [TrainingParameterStorage](../glossary.md#native-parameter-copy--trainingparameterstorage) | Training0007 replaces NN types with live [TrainingParameters](../glossary.md#live-parameter-source--trainingparameters) and migrates callers; no NN-taking compatibility overload inside generic Training. |
| extensions/[checkpoint](../glossary.md#term-checkpoint) [artifact](../glossary.md#term-artifact) package (six files) | modules/checkpoint, unchanged `checkpoint.artifact`; preserve `:extensions:checkpoint` via projectDir. |
| Nine checkpoint.model classes | Move to extensions/nn under `io.github.pho001.synaptik.nn.checkpoint`; retain old file readers and declarative meaning, not forwarding compatibility classes. |
| GraphCompilationPort / GraphCapture / [CompiledGraphModel](../glossary.md#compiled-graph--compiledgraphmodel) | Compiler0008 Complete: same-traversal input sidecar over validated raw source, no optimization/[layout](../glossary.md#layout) closure; ordinary constructor reconstruction accepts [descriptor](../glossary.md#term-descriptor)-compatible symbolic replacements. |
| [NN Model](../glossary.md#model-module-buffer-and-scan-distinctions) / Module / DeclarativeModel | Stay extensions/nn; generic forward and flexible declarative APIs retained; new private [tensor](../glossary.md#tensor) implementation composes [owners](../glossary.md#term-owner) after reverse-edge removal. |
| [ONNX](../glossary.md#term-onnx) | Remains an extension marker; no importer/exporter promised. |

Concrete old-to-new imports/calls and consumer dependencies are recorded in
[ADR0039 migration](../design/decisions/0039-portable-graph-model-persistence.md#source-migration-and-retired-code).
The nine-class move is the unavoidable package exception: placing NN semantics in a package named
generic checkpoint would obscure the new [owner](../glossary.md#term-owner). Do not create a compatibility project.
Engine0021 separately removed a public type/factory, a deliberate source/[binary](../glossary.md#term-binary-descriptor-and-metadata) break; the brief
includes all Java consumers, API-[shape](../glossary.md#shape) tests, benchmark and test hooks. Engine's outward prepared
wrapper and [Runtime](../glossary.md#term-runtime)'s inward unique [lease](../glossary.md#term-lease) boundary remain intentionally separate.

<a id="target-dependencies-not-current-permission"></a>

## Coordinated [dependency](../glossary.md#term-dependency) cutover

An arrow means “depends on”; public [signature](../glossary.md#term-signature) types require [API](../glossary.md#term-application-programming-interface-api) visibility.
Training0007 enacts only Training `api(Model, Engine)` and [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) `api(Model, Training)` now.
Checkpoint0008 now enacts the table's generic [Checkpoint](../glossary.md#term-checkpoint) and NN [Engine](../glossary.md#term-engine)/Checkpoint edges in the root
and owning scoped contract. Its final review/coordinator gate passed; NN0028's bounded [owned Model](../glossary.md#model-module-buffer-and-scan-distinctions)
workflow is Complete. Unselected workflow extensions remain future and require their own authority.

| [Owner](../glossary.md#term-owner) | Selected direct project edges |
|---|---|
| Generic Training | api([Model](../glossary.md#term-model), Engine); no NN/Checkpoint/[Compiler](../glossary.md#term-compiler)/[Runtime](../glossary.md#term-runtime)/[Prepare](../glossary.md#prepare)/tuning/[backend](../glossary.md#backend) edge. |
| Generic Checkpoint | api(Model), implementation(Compiler); no NN/Training/Engine edge needed for [host](../glossary.md#term-host)-only [graph](../glossary.md#graph) values. Old Engine-materializing NN save glue moves to NN. |
| NN | api(Model, Training, Checkpoint, Engine); no Compiler/Runtime/Prepare/backend edge. |
| Engine and concrete backends | Current edges unchanged; no reverse NN/Training/Checkpoint dependency. |

Remove Training/Checkpoint→NN before or atomically with NN→Training/Checkpoint. Update exact edge
and role-aware architecture tests, not broad [modules](../glossary.md#term-module)-directory exemptions. Root/scoped [ownership](../glossary.md#term-ownership-meanings)
must be cut over in the implementing capability before code; [ADR](../glossary.md#term-architecture-decision-record-adr) approval alone permits no edge.

## Compatibility, [lifetime](../glossary.md#term-lifetime) and [validation](../glossary.md#term-validation)

For both ordinary [run](../glossary.md#term-run) methods: [Engine](../glossary.md#term-engine) [admission](../glossary.md#term-admission) first, non-null/exact-[owner](../glossary.md#term-owner) prepared validation,
then closed-prepared gate, then input inspection. Engine0021 changed the former direct-run ordering:
closed prepared now wins invalid-input failures without [borrowing](../glossary.md#term-borrowing). Exact-ID/order semantics remain.
Fresh values require exact full static [descriptors](../glossary.md#term-descriptor) and
live accessible capacity-checked storage, including checked [canonical](../glossary.md#term-canonical-form) geometry for unresolved
[layout](../glossary.md#layout), with no [prototype](../glossary.md#term-prototype) mutation or per-[batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) provider/[compile](../glossary.md#compile)/[prepare](../glossary.md#prepare) work. Existing adapters
are reused. [Runtime](../glossary.md#term-runtime) alone arbitrates close/run [leases](../glossary.md#term-lease); returned results survive prepared close,
and Engine shutdown remains results → preparations → integrations with [rollback](../glossary.md#term-rollback)/failure retention.

Retain ArtifactFile v1, [model](../glossary.md#term-model).v1, whole-model.v1, their readers and declarative behavior. Model.load (or advanced multi-input/output Model.loadGraph)
is the new [graph](../glossary.md#graph)-model.v1 entry; old ModelCheckpoint.load (moved import) remains the flexible
DeclarativeModel entry. No silent old-format-to-fixed-[signature](../glossary.md#term-signature) conversion. New model-only save
stores source [DAG](../glossary.md#term-directed-acyclic-graph-dag) and working values, not [optimizer state](../glossary.md#term-optimizer-state), Java callbacks, [RNG](../glossary.md#graph-rng-state--graphrngstate)/cursor continuation,
compiled [artifacts](../glossary.md#term-artifact) or [device](../glossary.md#term-device) plans. Optional durable Training [resume](../glossary.md#schema-serialization-and-checkpoint) remains separately selected.

The first closed graph [codec](../glossary.md#term-codec) [slice](../glossary.md#slice) is ADD, MUL, NEG, [MATMUL](../glossary.md#matrix-multiplication--matmul) and PERMUTE with exact [1,0]
attributes (ordinary [Linear](../glossary.md#term-linear-meanings)), with
multiple distinct output roots; unknown kinds/attributes reject. All seven raw value [dtypes](../glossary.md#data-type--datatype),
[signed zero](../glossary.md#term-signed-zero), [NaN](../glossary.md#term-nan) payloads and exact [alias](../glossary.md#aliasing-and-arena) groups persist independently of [backend](../glossary.md#backend)/training support.
Canonical multi-output reconstruction is an [invariant](../glossary.md#term-invariant), not permission to admit every multi-output
[operation](../glossary.md#operation) in the first slice. Zero-[node](../glossary.md#node) graphs may be data, but current Engine empty-plan rejection
remains explicit. No fake operation is inserted to hide it.

Hold NN0027/CP0005C/Training0006/adapter expansion as optional older proposals, not prerequisites.
No broad performance, profile, [certificate](../glossary.md#formal-proof-certificate-and-trusted-assumptions), proof, five-[fork](../glossary.md#term-fork-meanings) or numerical campaign. Compiler0008
validated bounded malformed source and forward/[autograd](../glossary.md#autograd) round trips; Checkpoint0008 [persistence](../glossary.md#term-persistence)/alias
validation is Complete. NN0028
joins manual/file fresh compute, configured train, save-before-compute, trained re-save, [ownership](../glossary.md#term-ownership-meanings)/
close failures and old-v1 flexible behavior. Those implementation tests and the coordinator gate passed.

Original P2 [planning](../glossary.md#term-planning) used source/[API](../glossary.md#term-application-programming-interface-api) reasoning, Markdown validation and an allowed-write inventory,
without implementation. Engine0021's subsequent implementation, independent review and native-enabled
test/[Javadoc](../glossary.md#term-javadoc) gate are recorded in its brief. Compiler0008's brief records its final independent
approval and successful coordinator [checkpoint](../glossary.md#term-checkpoint), including the bounded root/[compiler](../glossary.md#term-compiler) exception.
Checkpoint0008's bounded Compiler edge and generic persistence are Complete after Halley's final
APPROVE and the coordinator source-matched root gate. Training0008 and NN0028 are Complete; unselected Model capabilities still require their own cutovers;
no broad benchmark or native-build campaign is implied.
