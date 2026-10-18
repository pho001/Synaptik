# ADR 0039: One tensor-model execution and persistence path

## Status

2026-10-17: **accepted product choices; Engine0021, Training0007, Compiler0008 and Checkpoint0008 Complete**.
The user explicitly authorized the prepared-owner cutover after independent Astra/xhigh packet
approval. Root and owning Engine/Training contracts enact that bounded cutover. Independent final
Class C approval and the coordinator's native-enabled integrated test/Javadoc gate passed.
The user also approved fluent Engine binding on the future ordinary Model; its precise target
lifecycle is recorded below. Training0007 enacts generic live admission and the physical Training/
NN dependency cutover, accepted after Pascal's definitive APPROVE and the coordinator gate.
Its brief owns evidence. Authorized Training frontier: None. On 2026-10-17,
[Compiler0008](../../planning/modules/compiler/tasks/0008-cold-source-capture-and-reconstruction.md)
is Complete: bounded source API and coordinated root/compiler authority implemented, Einstein final
code/docs APPROVE and coordinator Class C checkpoint passed. Its brief owns evidence.
Authorized Compiler frontier: None; [Checkpoint0008](../../planning/extensions/checkpoint/tasks/0008-generic-graph-value-persistence-and-nn-glue-cutover.md)
is Complete after Halley's definitive final APPROVE and the coordinator's source-matched test/Javadoc
gate on 2026-10-17. Authorized Checkpoint frontier: None; Training0008 is the next planning candidate,
still Draft; NN0028 remains Draft. Checkpoint0008 enacts the bounded root/owning-contract,
physical split and role-specific architecture-test cutover in the same change, before relying on
reversed edges. Final independent code/docs review and coordinator validation passed; its brief owns evidence.
High-level Model/withEngine remains TARGET, not implemented.
This record does not override [root architecture](../../../ARCHITECTURE.md) or its scoped
contracts, and does not authorize the remaining cutovers.

## Context and decision drivers

Manually built and file-restored models need the same ordinary compute/configured-train/save/load
calls. Reuse Tensor provenance, Compiler automatic differentiation, Engine prepared execution and existing
Training commits. Persist source operations and values, not Java callbacks or device plans.

Source checks against the pre-Engine0021 current-main/P1 snapshot established:

- NN Model<I,O> is arbitrary Java composition; Module.train() changes mode only.
  Parameter.replace keeps the wrapper but does not update already-constructed expressions.
- Before Engine0021, InferenceSession.run required exact compiled TensorId/descriptor membership. AdvancedEngine's
  runOrdinaryOpen already snapshots storage, validates it, then borrows through direct adapters.
  EngineTypedLifecycleTest covers reuse, close precedence and independent results.
  Direct Engine.run currently checks prepared closure later than the session path; unification
  must explicitly change that precedence rather than claim all behavior is identical.
- HostTensorValue is detached and survives Engine close. There is no current InferenceResult type.
- TrainingSession, TrainingParameter and TrainingParameterStorage are the three NN-coupled files.
  Live path/order/wrapper/Tensor/storage checks and late precommit checks are required;
  TrainingSessionFailureContractTest proves late replacement rejection and rollback.
- Native copy preserves represented bits and Tensor identity but mutates storage association:
  it is safe only after acquiring private state, never on an externally borrowed source graph.
- Checkpoint has six opaque artifact files and nine NN-specific model files.
  ModelStateRestoration preserves raw payloads and aliases; whole-model.v1 really retains flexible
  Linear/LayerNorm ranks and convolution batch/spatial constraints.
- GraphCapture and CompiledGraphModel provide the existing graph, but no public cold reconstruction.
  Gradle and NnTrainingDependencyContractTest/CheckpointDependencyContractTest enforce the old
  edges; isolated nnApiConsumer/trainingApiConsumer source sets expose migration obligations.

## Accepted product decisions

1. Common Model-facing tensor compute/configured-train/save/load for both origins; keep existing
   generic Model<I,O>/arbitrary Java forward. No mandatory TensorModel, LoadedGraph, Runner or
   ModelExecutor hierarchy; no arbitrary Java callback/record serialization.
2. Lazy model-owned Engine.standard() for normal use, currently fresh CPU composition. Explicit
   Engine is borrowed, never closed by the model. Model.load(path[, limits]).withEngine(engine)
   configures that exact same ordinary Model before preparation; application-built models use the
   same method. Load is cold and opens no Engine or discovery.
3. One model-owned working weight set feeds compute, train and save. Acquire external values once
   when necessary without changing their Tensor/storage; no ownership-only duplication of loaded
   values. One exact native-admission copy may still be necessary before training.
4. Targeted source migration is permitted for generic Training/Checkpoint separation and the
   subsequently approved thin Engine session removal, not blanket redesign or compatibility
   frameworks. Retain old formats/readers and declarative behavior.
5. First new graph signature is explicit/static: [32,784] accepts fresh [32,784] and rejects
   [64,784], with no automatic recapture/recompile/prepare. Static Java unrolling is ordinary DAG,
   not a recurrent recipe/BPTT prerequisite.

## Options and rationale

Keep the status quo for existing flexible declarative models, but do not extend per-network recipes
as a prerequisite for ordinary graph persistence. Reject per-batch forward reconstruction as normal
execution, serialized CompileArtifacts, a second public model hierarchy and merging Training into NN.

Select fixed tensor-list execution methods on the existing Model class. This deliberately separates
symbolic Java output O from materialized values. It avoids unchecked casts and an I/O codec/adapter
framework. The cost is that arbitrary Model<I,O> is not automatically executable or serializable:
ordinary NN users select a Tensor signature once through defineTensor; advanced arbitrary-record
models may explicitly select their Tensor boundary. Both use the same ordinary Model workflow as load.

The approved Engine amendment chooses existing Engine PreparedExecution over retaining a thin
InferenceSession or renaming it to another executor/runner. One public prepared owner reduces
lifecycle/API duplication; Model's high-level compute/train hides preparation anyway. The cost is
explicit source/binary migration of session users and the direct-run precedence change below.
TrainingSession is retained: optimizer history, accumulation, counters and transaction/busy state
are substantive ownership, not a redundant inference wrapper. Engine PreparedExecution versus
Runtime PreparedExecution is also retained: outward owner/adapters versus inward unique lease
and prepared resources are separate boundaries. Advanced lifecycle remains supported.

For Model binding, select a fluent in-place configuration method over a copied model, request
builder, compute wrapper or global Engine/session. It preserves one graph and weight owner and
keeps compute terminal. The cost is a strict pre-preparation freeze, including failed preparation
attempts; callers cannot switch engines to retry after that boundary.

## Selected API and type feasibility

**Design declarations only; not implemented or runnable current APIs.**
Model remains abstract, extends Module and gains AutoCloseable. Existing Model factories and forward
signatures are unchanged. Types below use their existing owning packages unless stated otherwise.

```java
// On Model<I,O>; tensor capability is independent of I/O:
public Model<I, O> withEngine(Engine engine);
public HostTensorValue compute(Tensor input);
public TrainingStep train(Tensor input, Tensor target);
public void configureTraining(TensorDescriptor targetSignature,
    BiFunction<Tensor, Tensor, Tensor> objectiveBuilder, Optimizer optimizer);
public List<HostTensorValue> compute(List<Tensor> inputs);
public List<HostTensorValue> compute(List<Tensor> inputs, long maximumTotalBytes);
public void configureTraining(List<Tensor> targetPrototypes,
    BiFunction<List<Tensor>, List<Tensor>, Tensor> objectiveBuilder, Optimizer optimizer);
public TrainingStep train(List<Tensor> inputs, List<Tensor> targets);
public void save(Path path) throws IOException;
public void save(Path path, ArtifactLimits limits) throws IOException;
public void close();

// Static Model factories (defineTensor also has an explicit borrowedEngine overload):
public static Model<Tensor, Tensor> defineTensor(
    TensorDescriptor inputSignature, ModelDefinition<Tensor, Tensor> definition);
public static Model<List<Tensor>, List<Tensor>> graph(
    List<Tensor> inputs, List<Tensor> outputs, List<GraphCheckpoint.Leaf> namedLeaves);
public static Model<List<Tensor>, List<Tensor>> graph(
    List<Tensor> inputs, List<Tensor> outputs, List<GraphCheckpoint.Leaf> namedLeaves,
    Engine borrowedEngine);
public static Model<Tensor, Tensor> load(Path path) throws IOException;
public static Model<Tensor, Tensor> load(Path path, ArtifactLimits limits) throws IOException;
public static Model<Tensor, Tensor> load(
    Path path, ArtifactLimits limits, Engine borrowedEngine) throws IOException;
```

Private implementations specialize Model<Tensor,Tensor> or Model<List<Tensor>,List<Tensor>>;
the existing abstract base need not narrow I or O. No cast from O occurs: compute(Tensor) returns
HostTensorValue and compute(List<Tensor>) returns List<HostTensorValue>, never O. Single-input/output
methods validate arity. Advanced static loadGraph(Path) (and limits/borrowed-Engine overload) returns
Model<List<Tensor>,List<Tensor>> for multi-boundary files; ordinary load rejects those with a loadGraph hint.
The advanced graph implementation's forward(List<Tensor>) constructs ordinary symbolic outputs through cold source
reconstruction with the supplied descriptor-compatible input leaves and the same owned state.
That advanced symbolic call does not execute or change the cached source; compute never calls it
per batch. Existing Model.define and custom subclasses keep their exact forward behavior.
Their new tensor methods fail with UnsupportedOperationException before Engine/file work unless
they provide the tensor capability; their default close owns nothing and is a no-op.

Normal defineTensor invokes the typed definition once, creates its declared prototype internally,
and constructs forward once to initialize deferred layers and establish the source before returning.
It discovers registered state itself. Callback-supplied state is borrowed until privately acquired;
copy/reconstruct and strict installation into the adopted child wrappers precede publication. Original
caller Tensors/stores are never rebound. Internally owned reconstructed/loaded state is adopted,
not copied again. Definition/forward side effects are not generally rollbackable; failed construction
publishes no Model, closes newly owned resources and does not claim to undo arbitrary callback effects.
The NEW defineTensor discards its definition/forward callbacks after prototype source construction.
Its symbolic forward(Tensor) reconstructs from the frozen owned source/leaves on an explicit cold
call, just like the loaded single-Tensor model. It never invokes the old closure, even if that closure
captured a pre-acquisition weight Tensor. Compute/train/save use the same owned source without
per-batch reconstruction. Callers must not mutate adopted child modules. Legacy Model.define retains
its existing callback behavior unchanged; this one-shot policy is only the new defineTensor contract.

For an advanced existing Model<MyInput,MyOutput>, call its forward once using explicit prototype Tensors,
select the Tensor outputs from MyOutput and enumerate named state leaves; pass those to Model.graph.
No MyInput/MyOutput adapter, callback retention, reflective record discovery or record restoration
is required. Normal load returns Model<Tensor,Tensor> like defineTensor; loadGraph matches the advanced graph factory.
This is source-level feasible through ordinary concrete generic specialization; no javac experiment
or unchecked-cast assumption is needed. Compile-only API consumers must witness it at implementation.

### Necessary boundary types, not a new model hierarchy

- `training.TrainingParameters`: `List<Entry> entries()`, `void validateUnchanged()`;
  nested `Entry(String path, Tensor tensor)`. Membership is immutable for a borrow, but validation
  reads the live owner. NN's new `Module.trainingParameters()` returns a private adapter retaining
  exact ordered paths/wrappers/Tensors. Training owns storage/descriptor/finiteness validation.
  This boundary and the NN adapter are implemented by Training0007. Training has exactly public
  Model/Engine edges; NN now also exports Checkpoint/Engine for the moved persistence signatures.
  GraphCheckpoint below is Complete in Checkpoint0008 after final review/coordinator gate.
- `checkpoint.graph.GraphCheckpoint`: NN-free cold persistence owner with nested
  `Source(List<Tensor> inputs, List<Tensor> outputs, List<Leaf> leaves)`,
  `Leaf(String name, Role role, Tensor tensor)` and enum PARAMETER/BUFFER/CONSTANT.
  Source is ordinary Tensor references plus roles, not another graph IR, executable or lifecycle
  owner. Public `copy(Source)`, `save(Path, Source, ArtifactLimits)`,
  `load(Path, ArtifactLimits)` and `rebind(Source, List<Tensor>)` respectively acquire detached
  host state, persist it, restore it, or construct symbolic outputs. Save/load declare IOException.
  Copy/rebind are cold, never a batch loop.
- Compiler0008 implements on existing GraphCompilationPort `captureSource(List<Tensor>)` returning nested
  `SourceCapture(CompiledGraphModel graph, List<Tensor> leaves)`, with leaves aligned to graph.inputs,
  and `reconstructSource(CompiledGraphModel, List<Tensor>) -> List<Tensor>`.
  This integration-only sidecar supplies exact source identity; compiled artifacts still retain no
  Tensor. GraphCheckpoint hides Compiler types from NN. No Compiler registry or second DAG.

Compiler0008 source reconciliation (2026-10-17): capture returns the **raw** graph, validated by
existing inference without optimization or logical-layout closure, and exact leaves collected when
that same traversal assigns graph inputs. Reconstruction accepts full-descriptor-compatible
ordinary Tensor expressions as positional replacements, not only provenance-free leaves: this is
required for cold symbolic forward/composition. Their upstream provenance remains ordinary input
to later compile/autograd, not added source-codec coverage. Captured sidecar leaves remain unique
provenance-free identities. Repeated replacement references are explicit substitution; output
positions remain ordered and ordinary compile still rejects duplicate resolved outputs. Preserve
resolved/unresolved descriptors exactly; do not strip final compiled layouts to force reconstruction.
Compiler checks logical metadata only; Checkpoint's host-value admission below remains separate.

Checkpoint0008 source reconciliation (2026-10-17): declared input order is not capture's first-encounter
order, which also includes named state. Associate by exact Tensor identity; retain unused declared
prototype slots using existing graph constructors and Compiler's unused-input support, without saved
batch values. Unreachable named registrations still reject. Host reconstruction retains the original
descriptor through TensorFactory.create and compatible storage, never resolves an absent layout merely
to call allocate. These are consumer consequences of current source APIs, not additional Compiler scope.

These are the minimum two cross-module data/admission boundaries plus a Compiler capture sidecar;
no public execution result, signature builder, transfer token or generic adapter framework is added.
Private graph Model uses existing Parameter/Buffer wrappers. Dotted state paths use private
state-namespace Module children only where needed to preserve inherited final stateDictionary/
parametersRecursively behavior; these children have no forward recipe or execution role.
Require canonical Module traversal order and reject prefix/name collisions; do not silently rename
paths or keep a second authoritative weight map. CONSTANT leaves remain outside trainable state.
This private namespace cost is necessary because existing traversal methods are final.

## Source, shape, state and ownership

Graph factory requires a complete source DAG and explicit ordered input prototypes. Every reachable
non-input leaf has exactly one role; repeated named aliases of one Tensor retain one leaf/store,
while equal-valued distinct leaves stay distinct. Reject unreachable registrations, stale outputs
referring to replaced state, conflicting roles, unsupported views and incomplete values.
requiresGrad alone never establishes a parameter role. First graph source uses static shapes and
canonical non-view host values; descriptors retain requiresGrad and resolved/unresolved layout
distinctions. Storage capacity is validated from canonical static geometry when layout is unresolved.
This is separate from, and does not narrow, existing flexible declarative/v1 behavior.

Acquisition copies each external state identity once into fresh private host leaves, then reconstructs
the ordinary expressions against those leaves before publication. The caller's original Tensor,
storage association and bytes are unchanged. No retained original-weight graph hides behind the
owned values. The fully restored load result is adopted internally without another ownership copy.
No public transfer API is needed for this first capability; any later exclusive transfer must name
and transfer the actual resource owner. Old generic Model.forward ownership is unchanged.

Save before first compute works because graph construction or load already establishes source and
signature. Save does not run Java forward, infer a signature from the last batch or create an Engine.
Only host-readable graph values are in this first path; device-only source leaves reject rather than
trigger hidden execution. Old NN checkpoint save still uses public Engine materialization.

Mode and state guards compare the source-bound mode plus current path/order/wrapper/Tensor/storage
identities before compute, configured train, symbolic forward or save. A changed mode/binding closes
cached inference preparations/TrainingSession and puts the graph capability into an invalid state; the call rejects, without
recapture or a silent reset of optimizer state. Build an explicit replacement graph Model from the
new source to proceed. Existing Parameter.replace/loadStateDictionary still have their current
borrowed-reference semantics; they are not the new model's ownership-acquisition API. Never train or
save newly borrowed replacement state via the old graph. Ordinary successful Training updates change
bytes in the same owned stores and remain valid for both compute and save.

Module.train() and eval() remain mode toggles. Both graph and defineTensor construction select evaluation mode before the guarded source is fixed
(defineTensor sets its adopted tree to eval before the single forward); it does not promise to infer which
ForwardContext a caller used. configureTraining does not call Module.train() or reinterpret a
mode-sensitive source. Its objectiveBuilder runs once on owned output expressions and explicit
target prototypes; it must be pure, return one supported scalar FLOAT32/FLOAT64 objective and retain
no runtime callback role. Training-mode BatchNorm buffer updates and advancing dropout RNG are not
added. A mode-specific DAG must be explicitly constructed as such.

configureTraining may succeed once, before the first train; after success repeat configuration
rejects rather than discarding optimizer history. Failed attempts may be explicitly retried under
the Engine freeze below. It validates the configuration, acquires shared native storage where
needed, and constructs the objective/session against the final owned state. Native admission closes
any inference preparation before rebinding storage; preserve Tensor identity and exact alias groups.
Already-native admitted owned values are not copied. Heap values may need one exact copy.
A failed copy leaves old associations/bytes intact; preparation failure closes unpublished resources
and leaves no active Training session. Subsequent explicit retry may reuse already-admitted storage,
never recopy merely for ownership. Inference can prepare once again after this explicit transition,
not after every batch/update.

Training's native-only domain, duplicate registered Tensor/overlapping-store rejection, complete
live pre-run/precommit validation, optimizer algorithms, slots/counters and failure atomicity remain.
Repeated DAG use of one uniquely registered weight is not duplicate registration; aliased parameter
paths that current Training rejects still reject, never split to evade checks. Training0008 adds
`runValues(List<Tensor>, GradientMode)` for non-parameter values in fixed non-parameter compiled
order; existing run keeps its exact-identity meaning. Both share the same transaction.

## Engine, result and exclusion lifecycle

### Target Model Engine binding

This subsection is a TARGET for NN0028, not current Model API or a change to Engine0021.
`withEngine(Engine)` returns `this` by reference, with the same generic Model type, graph, wrappers,
state and ownership. It only selects the borrowed Engine used by both compute and configured
training. It does not copy state, move memory, create an executor/request, compile, prepare or run.
`compute(batch)` remains terminal and returns detached `HostTensorValue`; there is no
`compute().setEngine`, request builder, execution wrapper or public/global engine/session hierarchy.

Before the first preparation attempt, an open idle tensor-capable Model may select or replace a
borrowed Engine, including selecting the same one again. It never closes the previous or new
borrowed Engine. With no explicit selection, execution lazily creates one model-owned
`Engine.standard()`, currently CPU-only; there is no discovery or global default. Construction,
load, save and explicit cold symbolic forward do not create it. Existing explicit borrowed-Engine
factory/load overloads remain as declared above (including defineTensor and loadGraph variants);
they establish the same selection and ownership, not a second binding mechanism.

Admission checks a previously selected Engine's closure first, then Model closing/closure, then
busy state, before inspecting arguments. An admitted withEngine checks tensor capability, then
the binding freeze, then rejects null with `NullPointerException` and a closed candidate with
`IllegalStateException`. Closed/closing/busy/frozen failures are `IllegalStateException`;
unsupported generic Model capability is `UnsupportedOperationException`. Rejection leaves the
selection unchanged and releases admission. Even the same Engine is rejected after freeze;
a closed selected Engine is not replaceable, and is never replaced automatically.
`Model.close()` bypasses ordinary Engine/capability/busy admission: it establishes closure, waits
for admitted work and attempts all owned cleanup regardless of the selected Engine's state.
Repeated/concurrent close replays the retained cleanup result, including failure; a closed borrowed
Engine must never prevent release of model-owned resources.

Freeze occurs when the first admitted compute or configureTraining has passed its argument and
live-source checks and enters preparation setup, **before** lazy Engine creation, native admission,
compilation or preparation. No engine/configuration argument can then rebind it. Freeze survives
Engine-creation, objective-builder, native-copy, compile or prepare failure; an explicit retry uses
the same selected Engine or still-uncreated standard choice. Invalid arguments rejected before
setup do not freeze. A train without successful configuration rejects without preparation.
This early, irreversible boundary prevents a partial failed setup from permitting a different Engine.

configureTraining may follow inference and may explicitly replace heap storage with native storage
under the existing transition above, closing/rebuilding affected preparations on the **same** Engine.
It does not reopen binding. There is no automatic reprepare after rebinding; rebinding is possible
only before any setup. One successful training configuration is final; failed configuration may
retry on the frozen choice with already-admitted storage. Model.close closes its lazily created
Engine only; a borrowed Engine remains caller-owned. Load never inspects backend availability.

### Implemented Engine boundary and target Model results

Engine0021 removed the former `InferenceSession`/`Engine.session` and session-only Engine/AdvancedEngine
delegation/registration branches. The implemented Engine surface uses existing `Engine.prepare(graph)` and adds
`Engine.runValues(PreparedExecution, List<Tensor>) -> RunResult`. The owner-bound prepared graph's
inputs authenticate positional order; no new public type/token is introduced. Unlike exact-ID,
arbitrary-order `Engine.run(prepared, inputs)`, IDs may be fresh. Validate exact count and full
static descriptor (dtype, shape, layout, requiresGrad), present/live/current-thread-accessible
storage and capacity before any borrow. Resolved layout uses its checked referenced span;
unresolved layout uses checked canonical static element/byte geometry, rejecting overflow/unknown
geometry. That added unresolved-layout check is scoped to runValues, not a silent run change.
Snapshot associations once and reuse prepared metadata/direct adapters and isolated Runtime state;
no prototype replacement, synthetic Tensor, extra per-element work, capture, compile, prepare,
provider/registry lookup or discovery per call.
At high level, public input order is mapped once to compiled order; parameters/constants fill their
own precomputed positions. Same-shape input values may differ; [64,784] never specializes [32,784].

Both ordinary run methods use Engine admission, non-null/exact-owner prepared validation, exact
inward delegate/closed gate, then inputs. This deliberately changes direct run's failure precedence:
Previously input validation/borrowing could precede closed-prepared discovery; now closed prepared
wins null/malformed/dead inputs without borrowing, as the former session did. Null/foreign prepared
validation also precedes input-list errors. Exact-ID membership/order and open-valid-handle
diagnostics remain. Runtime's unique nonwaiting lease authority still arbitrates a subsequent close.
Publish exactly one outward prepared registration; rollback failed prepare/wrapper/publication.
Engine shutdown waits for admitted operations then closes results before preparations before
integrations, preserving reverse order, attempt-all cleanup, suppression and retained failures.

Model compute materializes all ordered outputs under an aggregate bound (default Long.MAX_VALUE,
like Engine.compute), closes every RunResult before returning, and returns detached HostTensorValue
objects. Compute results and TrainingStep remain readable after model/input/Engine close.
Low-level RunResult remains independently owned: prepared close does not close it; Engine close does,
and caller input storage must stay usable/unmodified until result close. No new result lifetime.

The first high-level model admits one operation at a time; a competing compute/train/save/configure/withEngine
fails busy before argument inspection. Close rejects new work, waits for admitted work, closes
TrainingSession/inference preparations before owned native arenas and finally its owned Engine, attempting all cleanup and retaining
failures. Borrowed Engine closure causes subsequent work to fail, never opens a replacement.
Caller input/state mutation and closure are excluded for each whole operation; outside mutation of
owned working state is forbidden. Training additionally rejects overlapping input/parameter regions.
Underlying Engine preparations retain their concurrent isolated-run capability; this convenience adds
no concurrent training/inference promise or synchronization to inherited Module mutators.

## Source migration and retired code

Selected direct edges: generic Training api(Model,Engine); generic Checkpoint api(Model) plus
implementation(Compiler); NN api(Model,Engine,Training,Checkpoint). New graph values are host-only,
so generic Checkpoint needs no Engine edge. The nine moved NN glue classes retain their Engine use.
Preserve `:extensions:training` and `:extensions:checkpoint` project IDs through projectDir to
modules/training and modules/checkpoint, plus current training/checkpoint.artifact packages.

| Existing source/build call | Target migration |
|---|---|
| Engine.session(graph) → InferenceSession | Engine.prepare(graph) → existing engine.PreparedExecution; remove type/factory, no shim. |
| InferenceSession.run(originalInputs) | engine.run(prepared, originalInputs), preserving exact-ID/arbitrary-order binding; fresh values explicitly use engine.runValues(prepared, compiledOrderValues). |
| InferenceSession.compiledGraph/isClosed/close | Existing prepared.compiledGraph/isClosed/close; independently owned results are not closed by prepared close. |
| TrainingSession inferenceSession field/constructor/open/executeAndDecode/isClosed/beginOperation/close | Engine0021 mechanical Engine PreparedExecution/prepare/run migration; preserve public TrainingSession, transaction, prepared-before-seed cleanup and failed-open rollback. |
| CpuLifecycleBenchmark helper signatures; EngineMixedOwnerTestAccess reflection; Engine typed/advanced public-shape and lifecycle/integration tests | Engine + PreparedExecution calls; migrate reflective target and API assertions, including var-typed Engine.session calls; no benchmark execution or backend redesign. |
| TrainingSession.open(engine, module, objective, optimizer[, groups]) | TrainingSession.open(engine, module.trainingParameters(), objective, optimizer[, groups]); remove Module overloads atomically with consumers. |
| TrainingParameterStorage.copyToSharedNative(module, arena) | copyToSharedNative(module.trainingParameters(), arena); advanced caller still owns Arena/exclusion. |
| trainingApiConsumer imports NN Module through Training alone | Generic consumer accepts TrainingParameters with Training-only classpath; NN caller explicitly depends on NN and calls its adapter. |
| checkpoint.model.ModelCheckpoint / ModelCheckpointInspection / InputArtifactFingerprint imports | nn.checkpoint equivalents, with an explicit :extensions:nn dependency for old NN checkpoint users. Method behavior/file readers retained. |
| ModelArtifactCapture/Candidate/Codec, ModelStateRestoration, WholeModelComposition/DefinitionCodec | Move privately with the other three types to NN; old wire IDs/defaults/preflight/publication preserved. |
| ArtifactFile/Envelope/Limits users | Same checkpoint.artifact imports and :extensions:checkpoint dependency; no NN transitivity after split. |
| Model.define(...).forward(input), DeclarativeModel and ModelSpecification | Unchanged. New graph opt-in uses explicit Tensor boundary; no implicit fixed-batch conversion of old files. |

No compatibility layer/project. Engine0021 removed InferenceSession and Engine.session plus
Engine.runSession, AdvancedEngine.openInferenceSession/runSessionOrdinary/unregister(InferenceSession)
and session-only branches atomically with all consumers. External session users must migrate and
recompile; this is a deliberate source/binary break, not an additive-only API claim. Training fixture
session() accessors returning TrainingSession remain. Training0007 retires old NN-taking
Training declarations/imports in its coordinated migration; Checkpoint0008 retires moved locations.
Do not delete algorithms,
readers, generic Model, declarative types or completed history. The superseded outputs(actualInputs)-
first proposal was never implemented and has no production code to delete.

## Planned usage — not current runnable code

Goal: ordinary NN construction, save before compute, fresh batches, training and load without
manual forward, leaf enumeration or a second Model. Imports are Model/Linear from nn.module/
nn.layers, Tensor/TensorDescriptor/TensorFactory from model.tensor, Shape/LayoutDescriptor/DataType
from their Model packages, Sgd from training, and ordinary Java Path/Optional. Proposed Model calls
are marked by this heading; existing Tensor factories and operations supply the values.

```java
var inputShape = Shape.of(32, 784);
var inputSignature = new TensorDescriptor(DataType.FLOAT32, inputShape,
    Optional.of(LayoutDescriptor.contiguous(inputShape)), false);
var weightShape = Shape.of(1, 784);
var weightDescriptor = new TensorDescriptor(DataType.FLOAT32, weightShape,
    Optional.of(LayoutDescriptor.contiguous(weightShape)), true);
// allocate supplies zero-filled host values; weight is external/borrowed.
Tensor weight = TensorFactory.allocate(weightDescriptor);
Tensor firstBatch = TensorFactory.allocate(inputSignature);
Tensor secondBatch = TensorFactory.allocate(inputSignature); // fresh identity
var targetShape = Shape.of(32, 1);
var targetSignature = new TensorDescriptor(DataType.FLOAT32, targetShape,
    Optional.of(LayoutDescriptor.contiguous(targetShape)), false);
Tensor targets = TensorFactory.allocate(targetSignature);
Path path = Path.of("linear.graph");
try (Model<Tensor, Tensor> model = Model.defineTensor(inputSignature, topology -> {
    Linear linear = topology.addModule("linear", new Linear(weight));
    return linear::forward;
})) {
    model.save(path);                          // source/signature known; no Engine
    HostTensorValue first = model.compute(firstBatch); // zero [32,1]; lazy Engine
    HostTensorValue second = model.compute(secondBatch); // same preparation, fresh input
    model.configureTraining(targetSignature, (prediction, target) -> {
        Tensor difference = prediction.sub(target);
        return difference.mul(difference).mean();
    }, new Sgd(0.01));
    TrainingStep step = model.train(firstBatch, targets); // zero loss, zero update
    model.save(path);                          // saves the same working weights
}
try (Model<Tensor, Tensor> restored = Model.load(path)) {
    HostTensorValue value = restored.compute(secondBatch); // zero [32,1]
}
// Optional explicit binding: proposed API; caller owns the Engine.
try (Engine engine = Engine.standard()) {
    try (Model<Tensor, Tensor> restored = Model.load(path).withEngine(engine)) {
        HostTensorValue value = restored.compute(secondBatch); // same ordinary Model, zero [32,1]
    } // closes model resources, not engine
}
```

The explicit signature is declared once, before any batch. Values/results are zero because both
weights and inputs are zero; use nonzero inputs/targets to observe learned updates. External weight
bytes and its storage association remain unchanged even for a nonzero update. Detached results
survive model close. A fresh [64,784] input rejects; no hidden recompilation.
`Model.load(path, limits).withEngine(engine)` uses the same binding policy. Calling withEngine
after the first preparation setup rejects, even if preparation failed; configure it beforehand.

Advanced graph(inputs,outputs,namedLeaves) is optional for raw Tensor graphs, not mandatory NN UX.
Old whole-model.v1 still loads through moved ModelCheckpoint.load into a flexible DeclarativeModel.
Model-only save does not resume optimizer slots, RNG, cursor, scheduler or scaler; configureTraining
is supplied anew after load. No automatic exact-continuation claim.

## Coordinated authority cutovers and validation

Current root and scoped rules win. Each implementation packet updates its exact authority,
explanations, ADR and affected architecture tests together **before code relies on the new rule**:

1. Engine0021: root Core lifecycle; runtime-prepare-engine modules/engine, Public reusable inference
   session (replace), Construction failure and closure, Run lifecycle; extensions-training
   Optimizer/training lifecycle owner/open/run references. Review root ownership/dependencies/testing
   and Training/NN rules without changing edges. Preserve explicit composition, mixed-owner routing,
   Engine versus Runtime prepared ownership and unique leases. Update existing public/runtime/Training
   API and lifecycle/user guides, glossary and Javadocs in the same cutover (exact files in 0021).
   The Engine0021 change annotates ADR0017's status with narrow supersession, preserving historical
   rationale; reconcile ADR0018's owner reference, not optimizer decisions. Update public-shape/
   lifecycle consumers and retain role-aware EngineCompositionContractTest and
   NnTrainingDependencyContractTest edges; no broad exemptions. This bounded Engine0021 cutover is
   Complete; it did not authorize the later cutovers on its own.
2. Training0007: root Repository layout/Module-ownership routing/Dependency rules/Testing
   requirements and extensions-training Training/NN/Optimizer-training lifecycle: generic live
   admission, native-copy ownership, remove Training→NN before NN→Training. The
   [source-verified brief](../../planning/extensions/training/tasks/0007-generic-live-parameter-admission.md)
   records the coordinated implementation and validation. Root/scoped authority, not this plan,
   now enacts only this bounded generic admission/dependency/layout cutover.
3. Compiler0008: root Explicit non-goals permits only a temporary graph-local ValueId→Tensor table
   inside cold validated source reconstruction; compiler-autograd modules/compiler and Compile
   lifecycle own family-constructor reconstruction, and Compile artifacts excludes source sidecars
   and Tensor retention. No gradient map exception, fabricated
   intermediate placeholders, Model execution or public generic derived-output constructor.
   The 2026-10-17 implementation enacts this bounded cold boundary only, with complete preflight
   before family construction and table disposal on return/failure. Ordinary compile/autograd
   stays unchanged; no dependency edge or downstream Model/Checkpoint capability is enacted.
   Final independent code/docs APPROVE and coordinator Class C checkpoint complete this cutover.
4. Checkpoint0008: root layout/ownership/dependencies/testing and extensions-training Checkpoint/
   Whole-model definition and composition move NN glue and authorize Checkpoint→Compiler cold
   implementation access; no reverse Compiler edge. First codecs: ADD/MUL/NEG plus MATMUL/PERMUTE with exact permutation [1,0] for ordinary Linear, multiple distinct
   roots, all seven raw value types, strict bounds/roles/aliases; unknown families reject.
5. Training0008/NN0028: their owning lifecycle contracts authorize fresh Training values and Model
   convenience/ownership, retaining old declarative behavior and Engine composition authority.
   NN0028 must test borrowed Engine closed first, followed by complete model-owned resource cleanup
   and repeated/concurrent Model.close failure replay.

Keep ArtifactFile v1 and old model.v1/whole-model.v1 readers. New graph-model.v1 uses explicit input
descriptors, leaf roles/identities, operation attributes and canonical raw values; inputs are not
saved batches. Validate complete data before publication; reuse force/revalidate/one-atomic-move.
Persist signed zero/NaN payloads without normalization, reject invalid BOOL bytes before import.
Finite closed codecs are sufficient; preserve canonical sibling-output identity whenever admitted.
Backend/gradient support remains independent; zero-node execution remains explicitly unsupported.

The [program](../../planning/model-execution-refactor.md#phases-and-owner-dag) owns the concrete DAG
and final coordinator signoff. The [Engine0021 brief](../../planning/modules/engine/tasks/0021-prepared-execution-fresh-value-binding.md)
records implementation, independent approval and the coordinator's successful cross-module gate.
Training0007 is Complete: generic live admission/native copy and its NN adapter passed final
independent review and the coordinator boundary gate. Compiler0008's bounded API/authority cutover
is Complete after final independent code/docs APPROVE and the coordinator checkpoint; authorized
Compiler and Checkpoint frontiers: None. Checkpoint0008 is Complete after Halley's final APPROVE
and the coordinator root gate; its brief owns evidence. Training0008 is next planning candidate,
still Draft; NN0028 stays Draft.
Glossary/Javadocs describe the bounded Training and Compiler cutovers. Model.withEngine,
Model compute/configured train/save/load and other APIs in
this ADR remain planned; no Model lifecycle is enacted here.
