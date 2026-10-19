# Model execution and persistence refactor program

## Purpose and status

Coordinate one model-facing compute/configured-train/save/load workflow for manually constructed
and file-restored Tensor graphs. A directed acyclic graph (DAG) records ordinary Tensor operations,
not Java callbacks. This plan and [ADR0039](../design/decisions/0039-portable-graph-model-persistence.md)
are non-authoritative. Engine0021, Training0007/0008, Compiler0008 and Checkpoint0008 enacted their bounded root/scoped cutovers; remaining module
and Model changes below are still planned and require their own coordinated authority.

P2 incorporates all five product choices accepted on 2026-10-16: common Model-facing tensor
operations with generic Java forward retained; lazy model-owned Engine.standard or explicit borrowed
Engine; owned working weights; targeted source migration; and a fixed explicit first graph signature.
The prior P2 packet was independently approved; the user subsequently approved removing the thin
InferenceSession in favor of existing Engine PreparedExecution plus Engine.runValues.
Independent Astra/xhigh review approved the plan amendment. Engine0021 implementation is now
Complete after independent Class C approval and the native-enabled integrated test/Javadoc gate.
Training0007 is Complete after Pascal's final independent APPROVE and the coordinator gate.
Authorized Training frontier: None. [Compiler0008](modules/compiler/tasks/0008-cold-source-capture-and-reconstruction.md)
is Complete after final independent code/docs APPROVE and the coordinator Class C checkpoint.
Authorized Compiler frontier: None; [Checkpoint0008](extensions/checkpoint/tasks/0008-generic-graph-value-persistence-and-nn-glue-cutover.md)
is Complete after Halley's final APPROVE and coordinator acceptance on 2026-10-17. Authorized Checkpoint
frontier: None; [Training0008](extensions/training/tasks/0008-fresh-value-training-session-runs.md) is
Complete after Euclid's definitive code/docs APPROVE and main's source-matched root gate (2026-10-19);
NN0028 is the next planning candidate, still Draft. The user approved fluent
same-instance Model.withEngine binding; it is a target for NN0028, not current Model API.

Launch verification: CP0005D and P1/Engine0020 are Complete, including their recorded independent
reviews and validation. The coordinator seeded the current-main planning packet into HEAD
`9e7e8aa15e0833a1bc91263a7add99a80d88cd92` plus the integrated uncommitted P1 snapshot.
Those inherited changes are not this amendment's diff. The pinned snapshot includes reviewed P2;
all nine allowed documents matched at launch. Coordinator verified main HEAD/base unchanged,
dependencies Complete and no active conflicts. The user then authorized Engine0021 implementation
in a clean context. Its brief records final review and integrated validation. Training0007 is now
Complete, as are Compiler0008, Checkpoint0008 and Training0008; NN0028 remains Draft.
Completed task histories and evidence remain intact.

## Phases and owner DAG

Arrows below are actual capability dependencies, not table-order authorization. Engine0021,
Training0007/0008 and Compiler0008 are Complete; authorized Training frontier is None,
and Compiler frontier is None.
Checkpoint0008 is Complete after final independent approval and the coordinator gate; Checkpoint
frontier: None. Training0008's brief owns Euclid's final APPROVE and main's accepted root-gate evidence;
NN0028 is the next planning candidate, still Draft.

| Owner/task | Capability | Depends on / integration order |
|---|---|---|
| P0 CP0005D; P1 Engine0020 | Completed planning/review and backend/native layout | Complete; no reopening of numerical/native campaigns. |
| Engine [0021](modules/engine/tasks/0021-prepared-execution-fresh-value-binding.md) | Sole ordinary prepared owner, positional fresh values and atomic consumer migration — Complete | Coordinated authority, independent implementation review and native-enabled integrated gate passed. |
| [Training0007](extensions/training/tasks/0007-generic-live-parameter-admission.md) — Complete | NN-independent live admission/native copy, relocation and NN adapter | Final independent APPROVE and coordinator native-enabled gate passed; evidence in brief. |
| [Compiler0008](modules/compiler/tasks/0008-cold-source-capture-and-reconstruction.md) — Complete | Bounded cold raw-source API and root/compiler authority cutover | Final independent code/docs APPROVE and coordinator Class C checkpoint passed; evidence in brief. |
| [Checkpoint0008](extensions/checkpoint/tasks/0008-generic-graph-value-persistence-and-nn-glue-cutover.md) — Complete | Generic graph/value persistence, physical split and retained old-v1 NN readers | Engine0021 + Training0007 + Compiler0008 + CP0005B + accepted P2 Complete; Halley final APPROVE and coordinator source-matched test/Javadoc gate passed; evidence in brief. |
| [Training0008](extensions/training/tasks/0008-fresh-value-training-session-runs.md) — Complete | Fresh stored values in existing TrainingSession.inputs() order, excluding parameters/private seed | Engine0021 + Training0007; Checkpoint0008 serial join Complete; Euclid final APPROVE and main source-matched root gate passed; evidence in brief. |
| NN0028 — Draft | Owned ordinary Model graph workflow and manual/file integration | Engine0021 + Training0008 + Checkpoint0008; same weights for compute/train/save. |

Training0007 and Compiler0008 could be independent code branches, but this packet authorizes no
parallel writers. Parallel group: None. Current common base: main
`7386249479ccda991b96a7e27bd5ed0279ca1ae3`, including Complete Engine0021/Training0007/Compiler0008/Checkpoint0008.
The Training0008 brief records coordinator-supplied predecessor verification and protected user files;
reverify if the base changes. Serial order remains Engine0021 -> Training0007 -> Compiler0008
-> Checkpoint0008 -> Training0008 -> NN0028; it does not add an algorithm dependency on Training.
Conflicts: None active; future shared root/scoped, NN state lifecycle and Training/Checkpoint
consumer cutovers serialize. Integration owner and shared-document owner: model-execution
coordinator. Each worker validates its affected modules once; each boundary-changing integrated
capability gets one coordinator repository test/Javadoc checkpoint, not duplicate reviewer reruns.
Checkpoint0008's brief records all seven integration fields, Pauli's approval and coordinator Ready verification dated
2026-10-17 (source/base/dependencies/conflicts/frontier); main retains exclusive shared-document writer delegation.
Training0008's bounded root/owning Training lifecycle cutover, implementation, independent review
and source-matched coordinator gate are Complete. The accepted ADR does not override current
authority. No NN0028 API or Model/Training autotuning is included.

Engine0021 delivered prepared ownership and fresh binding with mechanical Training/benchmark/test
migration. Training0007 completed generic live admission, native copy and the atomic reverse-edge/
consumer cutover. Its brief records final approval and coordinator validation. It did not wait on recurrent recipes,
fixed-scan backpropagation through time (BPTT), Data or all operation codecs.

## Selected public surface

Model declarations remain design targets, **not implemented APIs**; Checkpoint0008 implements the
bounded GraphCheckpoint boundary and physical/NN split, now Complete after final review/coordinator gate. Engine0021 delivered
prepared execution, and Training0007 implements only generic live admission/native copy plus NN's
adapter. Exact responsibilities, failures,
ownership and examples are in [ADR0039's API decision](../design/decisions/0039-portable-graph-model-persistence.md#selected-api-and-type-feasibility).

- Preserve `Model<I,O>`, `Model.define`, `ModelDefinition`, `ModelForward` and `O forward(I)`.
  Add tensor-only instance `compute(List<Tensor>) -> List<HostTensorValue>`,
  `train(List<Tensor>, List<Tensor>) -> TrainingStep`, configuration, save and close.
  Return types do not masquerade as arbitrary `O`.
- Normal `Model.defineTensor(inputSignature, definition)` and `Model.load(Path)` return
  `Model<Tensor,Tensor>`, with `compute(Tensor)` and `train(Tensor,Tensor)` conveniences. The factory
  captures forward/state once and discards that callback; symbolic forward reconstructs the owned
  source only on an explicit cold call. No user leaf inventory or second Model is required.
  Advanced `Model.graph(inputs, outputs, namedLeaves)` and `Model.loadGraph(Path)` return
  `Model<List<Tensor>,List<Tensor>>`. Private implementations supply the tensor capability;
  existing arbitrary Java models remain forward-only unless they opt into an explicit tensor boundary. No unchecked cast, mandatory public subtype or Java-record codec.
- `configureTraining(targetPrototypes, objectiveBuilder, optimizer)` fixes a scalar objective
  once over owned outputs and target prototypes; `train(inputs, targets)` means RESET_AND_STEP.
  Existing zero-argument `Module.train()` remains only a mode toggle.
- `Model.load(path[, limits]).withEngine(engine)` and application-built models use the same
  target `Model<I,O>.withEngine(Engine) -> Model<I,O>`, returning the exact receiver, not a copied
  model/request/executor. It borrows that Engine for compute and training; existing explicit
  borrowed-Engine factory/load overloads remain. `compute(batch)` is terminal and detached.
- Engine0021 removed `InferenceSession`/`Engine.session` and their session-only delegates, retaining
  existing Engine `PreparedExecution` as sole ordinary public prepared owner. It implements
  `Engine.runValues(PreparedExecution, List<Tensor>) -> RunResult` in compiled input order.
  `Engine.run(prepared, inputs)` keeps exact-TensorId/arbitrary-order matching; both methods adopt
  former session-strength close admission, an explicit direct-run failure-precedence change.
- Generic Training adds `TrainingParameters` (ordered entries plus live validation);
  NN adds `Module.trainingParameters()` as the adapter factory, not another public adapter class.
- Generic Checkpoint adds `checkpoint.graph.GraphCheckpoint` with nested `Source`, `Leaf`,
  and `Role`: data-only ordinary Tensor boundary/roles for persistence and cold copying.
  Compiler0008 implemented only its bounded source capture/reconstruction boundary, not another
  graph representation; Checkpoint0008's bounded generic source API is Complete after final gates.

The costs are explicit: a tensor-capability check on generic Model methods, detached tensor rather
than arbitrary-record execution results, one live Training interface, and one Checkpoint source
aggregate for the NN-free module boundary. None is a second model/runner hierarchy. No new
InferenceResult is needed: current Engine has RunResult and detached HostTensorValue.

## Ownership and source lifecycle

Graph construction fixes ordered input prototypes, ordinary output expressions and every reachable
PARAMETER/BUFFER/CONSTANT leaf before compute. Save-before-first-compute therefore works; an unknown
Java forward is not captured implicitly by save. Load already has this source/signature and opens
no Engine, performs no discovery and needs no future batch.

External supplied weights are copied once by identity into private leaves and the source DAG is
reconstructed against those leaves before the model is published. Never attach copied storage to
the caller's original Tensor or retain an output graph still referring to it. Model-owned/loaded
values need no second ownership-only copy. The first API does not need an exclusive-transfer
overload; an explicit future transfer can avoid acquisition copying but must transfer the real
lifetime owner, not assert ownership of a borrowed Tensor.

Inference, training and save all read the same owned working leaves. Heap-to-native admission may
perform one exact storage copy before training session creation; it is not a second working model
or per-batch copy. Close old inference preparations/TrainingSession before storage replacement,
retain Tensor aliases, rebuild bindings explicitly, and preserve native-only admission and
atomic/precommit checks. Failed admission leaves prior bytes/bindings intact; failed preparation
publishes no handle.

New graph signatures are fully static: [32,784] accepts fresh compatible [32,784], rejects [64,784].
No automatic recapture, compilation or preparation on mismatch. Mode or parameter/wrapper/storage
binding changes invalidate cached execution and fail before use; no silent stale graph. Explicitly
construct a replacement graph model for changed source/signature. Same-storage trained byte updates
do not invalidate it. Existing ModelSpecification/DeclarativeModel/v1 flexible rank/batch semantics
remain supported and are not silently tightened.

Normal Engine creation is lazy through Engine.standard(), currently CPU-only; no discovery/global
default. Target withEngine and explicit Engine overloads borrow and never close the Engine.
Binding freezes at the first compute/configureTraining preparation setup after argument/source
checks and before Engine creation/native admission/compile/prepare; failed setup remains frozen.
Before freeze an idle open model can change its borrowed selection; after freeze even the same
Engine rejects. Lifecycle admission precedes arguments; null and closed candidates reject without
changing selection. The [exact target error order and retry policy](../design/decisions/0039-portable-graph-model-persistence.md#target-model-engine-binding)
also covers configureTraining after inference: its explicit storage transition uses the same Engine,
never rebinding or automatic reprepare after rebinding. Load/save alone create no Engine.
Model.close rejects new operations, waits
for admitted work, closes TrainingSession/inference preparations, owned storage and its created Engine with
attempt-all cleanup. Compute returns detached values and internally closes RunResult; values and
TrainingStep survive model close. Low-level RunResult still borrows inputs until result close and
is closed by Engine shutdown, not by prepared close. High-level Model compute/train hides preparation;
no new session/executor/runner facade replaces the removed type. TrainingSession remains because
optimizer/accumulation state, counters, busy admission and atomic commit are real Training ownership.

First high-level lifecycle admits one compute/train/save/configure/withEngine operation at a time (busy rejects);
close waits. External callers must exclude mutation/closure of input storage for the whole call and
must not mutate exposed working state while the model is usable. Training additionally forbids
input/state overlap and retains all live path/order/wrapper/Tensor/storage checks before commit.
This is not a promise to make inherited Module mutators thread-safe.

## Current-to-target map

P1 locations, Training0007's physical/admission migration and Checkpoint0008's generic/NN split
below are Complete, including Checkpoint0008's independent approval/coordinator validation. Model
semantic moves remain future. Historical planning
directories and completed evidence stay where they are.

| Source/symbols | Selected target and migration |
|---|---|
| Old backends/* and native/metal-macos-arm64 | P1 completed: modules/backend/{cpu,metal,cuda,openblas-provider}, Metal native at modules/backend/metal/macos-arm64; :backends:* identities/packages/ABI unchanged. |
| Removed InferenceSession / Engine.session / session.run | Engine0021 delivered Engine.prepare → existing PreparedExecution; Engine.run(prepared, exactInputs), or explicit Engine.runValues(prepared, compiledOrderValues); no compatibility facade. |
| TrainingSession's former inferenceSession field/open/run/isClosed/close | Engine0021 migrated to Engine PreparedExecution/prepare/run/isClosed/close; public Training API and optimizer transaction unchanged. |
| extensions/training | Training0007 moved to modules/training; retained `:extensions:training` via projectDir and existing `training` Java package. |
| TrainingSession / TrainingParameter / TrainingParameterStorage | Training0007 replaces NN types with live TrainingParameters and migrates callers; no NN-taking compatibility overload inside generic Training. |
| extensions/checkpoint artifact package (six files) | modules/checkpoint, unchanged `checkpoint.artifact`; preserve `:extensions:checkpoint` via projectDir. |
| Nine checkpoint.model classes | Move to extensions/nn under `io.github.pho001.synaptik.nn.checkpoint`; retain old file readers and declarative meaning, not forwarding compatibility classes. |
| GraphCompilationPort / GraphCapture / CompiledGraphModel | Compiler0008 Complete: same-traversal input sidecar over validated raw source, no optimization/layout closure; ordinary constructor reconstruction accepts descriptor-compatible symbolic replacements. |
| NN Model / Module / DeclarativeModel | Stay extensions/nn; generic forward and flexible declarative APIs retained; new private tensor implementation composes owners after reverse-edge removal. |
| ONNX | Remains an extension marker; no importer/exporter promised. |

Concrete old-to-new imports/calls and consumer dependencies are recorded in
[ADR0039 migration](../design/decisions/0039-portable-graph-model-persistence.md#source-migration-and-retired-code).
The nine-class move is the unavoidable package exception: placing NN semantics in a package named
generic checkpoint would obscure the new owner. Do not create a compatibility project.
Engine0021 separately removed a public type/factory, a deliberate source/binary break; the brief
includes all Java consumers, API-shape tests, benchmark and test hooks. Engine's outward prepared
wrapper and Runtime's inward unique lease boundary remain intentionally separate.

<a id="target-dependencies-not-current-permission"></a>

## Coordinated dependency cutover

An arrow means “depends on”; public signature types require API visibility.
Training0007 enacts only Training `api(Model, Engine)` and NN `api(Model, Training)` now.
Checkpoint0008 now enacts the table's generic Checkpoint and NN Engine/Checkpoint edges in the root
and owning scoped contract. Its final review/coordinator gate passed; Model/Training
workflow extensions remain future and gain no permission from this cutover.

| Owner | Selected direct project edges |
|---|---|
| Generic Training | api(Model, Engine); no NN/Checkpoint/Compiler/Runtime/Prepare/tuning/backend edge. |
| Generic Checkpoint | api(Model), implementation(Compiler); no NN/Training/Engine edge needed for host-only graph values. Old Engine-materializing NN save glue moves to NN. |
| NN | api(Model, Training, Checkpoint, Engine); no Compiler/Runtime/Prepare/backend edge. |
| Engine and concrete backends | Current edges unchanged; no reverse NN/Training/Checkpoint dependency. |

Remove Training/Checkpoint→NN before or atomically with NN→Training/Checkpoint. Update exact edge
and role-aware architecture tests, not broad modules-directory exemptions. Root/scoped ownership
must be cut over in the implementing capability before code; ADR approval alone permits no edge.

## Compatibility, lifetime and validation

For both ordinary run methods: Engine admission first, non-null/exact-owner prepared validation,
then closed-prepared gate, then input inspection. Engine0021 changed the former direct-run ordering:
closed prepared now wins invalid-input failures without borrowing. Exact-ID/order semantics remain.
Fresh values require exact full static descriptors and
live accessible capacity-checked storage, including checked canonical geometry for unresolved
layout, with no prototype mutation or per-batch provider/compile/prepare work. Existing adapters
are reused. Runtime alone arbitrates close/run leases; returned results survive prepared close,
and Engine shutdown remains results → preparations → integrations with rollback/failure retention.

Retain ArtifactFile v1, model.v1, whole-model.v1, their readers and declarative behavior. Model.load (or advanced multi-input/output Model.loadGraph)
is the new graph-model.v1 entry; old ModelCheckpoint.load (moved import) remains the flexible
DeclarativeModel entry. No silent old-format-to-fixed-signature conversion. New model-only save
stores source DAG and working values, not optimizer state, Java callbacks, RNG/cursor continuation,
compiled artifacts or device plans. Optional durable Training resume remains separately selected.

The first closed graph codec slice is ADD, MUL, NEG, MATMUL and PERMUTE with exact [1,0]
attributes (ordinary Linear), with
multiple distinct output roots; unknown kinds/attributes reject. All seven raw value dtypes,
signed zero, NaN payloads and exact alias groups persist independently of backend/training support.
Canonical multi-output reconstruction is an invariant, not permission to admit every multi-output
operation in the first slice. Zero-node graphs may be data, but current Engine empty-plan rejection
remains explicit. No fake operation is inserted to hide it.

Hold NN0027/CP0005C/Training0006/adapter expansion as optional older proposals, not prerequisites.
No broad performance, profile, certificate, proof, five-fork or numerical campaign. Compiler0008
validated bounded malformed source and forward/autograd round trips; Checkpoint0008 persistence/alias
validation is Complete. NN0028
joins manual/file fresh compute, configured train, save-before-compute, trained re-save, ownership/
close failures and old-v1 flexible behavior. Those downstream implementation tests remain planned.

Original P2 planning used source/API reasoning, Markdown validation and an allowed-write inventory,
without implementation. Engine0021's subsequent implementation, independent review and native-enabled
test/Javadoc gate are recorded in its brief. Compiler0008's brief records its final independent
approval and successful coordinator checkpoint, including the bounded root/compiler exception.
Checkpoint0008's bounded Compiler edge and generic persistence are Complete after Halley's final
APPROVE and the coordinator source-matched root gate. Training0008 is also Complete; remaining Model capabilities still require their own cutovers;
no broad benchmark or native-build campaign is implied.
