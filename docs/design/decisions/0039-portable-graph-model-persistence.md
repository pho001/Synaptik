# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0039: One [tensor](../../glossary.md#tensor)-[model](../../glossary.md#term-model) execution and [persistence](../../glossary.md#term-persistence) path


## Editorial reading context

The application constructs an owned Model; supplied parameter leaves have no producing expression, and their values are copied into private Model state during acquisition. The original application Tensors do not become Model-owned resources. Model delegates compilation/execution to Engine and updates to Training, reusing preparations rather than creating a second executor. Saving source/state is separate from serializing native prepared work.

Use the [current Model and prepared execution workflow](../../user-guide/running-models.md) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

## Status

2026-10-17: **accepted product choices; Engine0021, Training0007, Compiler0008 and Checkpoint0008 Complete**.
The user explicitly authorized the prepared-[owner](../../glossary.md#term-owner) cutover after independent Astra/xhigh packet
approval. Root and owning [Engine](../../glossary.md#term-engine)/Training contracts enact that bounded cutover. Independent final
Class C approval and the coordinator's native-enabled integrated test/[Javadoc](../../glossary.md#term-javadoc) gate passed.
The user also approved fluent Engine [binding](../../glossary.md#term-binding) on the ordinary [Model](../../glossary.md#term-model); its implemented
[lifecycle](../../glossary.md#lifecycle) is recorded below. Training0007 enacts generic live [admission](../../glossary.md#term-admission) and the physical Training/
[NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [dependency](../../glossary.md#term-dependency) cutover, accepted after Pascal's definitive APPROVE and the coordinator gate.
Its brief owns evidence. Authorized Training frontier: None. On 2026-10-17,
[Compiler0008](../../planning/modules/compiler/tasks/0008-cold-source-capture-and-reconstruction.md)
is Complete: bounded source [API](../../glossary.md#term-application-programming-interface-api) and coordinated root/[compiler](../../glossary.md#term-compiler) authority implemented, Einstein final
code/docs APPROVE and coordinator Class C [checkpoint](../../glossary.md#term-checkpoint) passed. Its brief owns evidence.
Authorized Compiler frontier: None; [Checkpoint0008](../../planning/extensions/checkpoint/tasks/0008-generic-graph-value-persistence-and-nn-glue-cutover.md)
is Complete after Halley's definitive final APPROVE and the coordinator's source-matched test/Javadoc
gate on 2026-10-17. Authorized Checkpoint frontier: None. On 2026-10-19,
[Training0008](../../planning/extensions/training/tasks/0008-fresh-value-training-session-runs.md)
is Complete after Euclid's definitive code/docs APPROVE, no findings, and main's source-matched
root gate; its brief owns evidence. [NN0028](../../planning/extensions/nn/tasks/0028-owned-model-compute-training-and-persistence.md)
is Complete, accepted 2026-10-19 after definitive independent approval and the coordinator's
source-matched native-enabled root gate; evidence is in its brief. Authorized NN frontier: None.
Checkpoint0008 enacts the bounded root/owning-contract,
physical split and role-specific architecture-test cutover in the same change, before relying on
reversed edges. Final independent code/docs review and coordinator [validation](../../glossary.md#term-validation) passed; its brief owns evidence.
High-level [owned Model](../../glossary.md#model-module-buffer-and-scan-distinctions)/withEngine and all selected program phases are Complete.
This record does not override [root architecture](../../../ARCHITECTURE.md) or its scoped
contracts, and does not authorize the remaining cutovers.

Amended — 2026-10-20: selected [CNN](../../glossary.md#term-convolutional-neural-network-cnn)/static recurrent [persistence](../../glossary.md#term-persistence) and cold fixed scans share one
closed v3 format. Joint acceptance and validation evidence belong to the CP0010/CP0011 task briefs.

## Context and decision drivers

Manually built and file-restored models need the same ordinary compute/configured-train/save/load
calls. Reuse [Tensor](../../glossary.md#tensor) [provenance](../../glossary.md#provenance), [Compiler](../../glossary.md#term-compiler) [automatic differentiation](../../glossary.md#autograd), [Engine](../../glossary.md#term-engine) [prepared execution](../../glossary.md#prepared-execution--preparedexecution) and existing
Training [commits](../../glossary.md#commit-release-distribution-log-and-seed-meanings). Persist source operations and values, not Java callbacks or [device](../../glossary.md#term-device) plans.

Source checks against the pre-Engine0021 current-main/P1 [snapshot](../../glossary.md#term-snapshot) established:

- NN Model<I,O> is arbitrary Java composition; Module.train() changes mode only.
  Parameter.replace keeps the wrapper but does not update already-constructed expressions.<span data-glossary-navigation="true"> ([NN Model glossary](../../glossary.md#model-module-buffer-and-scan-distinctions); [Module glossary](../../glossary.md#term-module); [Parameter glossary](../../glossary.md#term-parameter))</span>
- Before Engine0021, InferenceSession.run required exact compiled TensorId/descriptor membership. AdvancedEngine's
  runOrdinaryOpen already snapshots storage, validates it, then borrows through direct adapters.
  EngineTypedLifecycleTest covers reuse, close precedence and independent results.
  Direct Engine.run currently checks prepared closure later than the session path; unification
  must explicitly change that precedence rather than claim all behavior is identical.<span data-glossary-navigation="true"> ([InferenceSession glossary](../../glossary.md#former-inference-session--inferencesession); [run glossary](../../glossary.md#term-run); [TensorId glossary](../../glossary.md#tensorid); [descriptor glossary](../../glossary.md#term-descriptor))</span>
- [HostTensorValue](../../glossary.md#host-tensor-value--hosttensorvalue) is [detached](../../glossary.md#detached-model-results) and survives Engine close. There is no current InferenceResult type.
- TrainingSession, TrainingParameter and TrainingParameterStorage are the three NN-coupled files.
  Live path/order/wrapper/Tensor/storage checks and late precommit checks are required;
  TrainingSessionFailureContractTest proves late replacement rejection and rollback.<span data-glossary-navigation="true"> ([TrainingSession glossary](../../glossary.md#training-session--trainingsession); [TrainingParameterStorage glossary](../../glossary.md#native-parameter-copy--trainingparameterstorage); [NN glossary](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [precommit glossary](../../glossary.md#term-precommit); [rollback glossary](../../glossary.md#term-rollback))</span>
- Native copy preserves represented bits and Tensor identity but mutates storage association:
  it is safe only after acquiring private state, never on an externally borrowed source graph.<span data-glossary-navigation="true"> ([represented bits glossary](../../glossary.md#term-represented-bits); [identity glossary](../../glossary.md#term-identity-meanings); [borrowed glossary](../../glossary.md#ownership-borrowing-and-lease); [graph glossary](../../glossary.md#graph))</span>
- Checkpoint has six opaque artifact files and nine NN-specific model files.
  ModelStateRestoration preserves raw payloads and aliases; whole-model.v1 really retains flexible
  Linear/LayerNorm ranks and convolution batch/spatial constraints.<span data-glossary-navigation="true"> ([Checkpoint glossary](../../glossary.md#term-checkpoint); [opaque glossary](../../glossary.md#term-opaque); [artifact glossary](../../glossary.md#term-artifact); [model glossary](../../glossary.md#term-model); [aliases glossary](../../glossary.md#aliasing-and-arena); [Linear glossary](../../glossary.md#term-linear-meanings); [convolution glossary](../../glossary.md#term-channels-first-convolution-layers); [batch glossary](../../glossary.md#term-batch-machine-learning-and-hyperparameters); [spatial glossary](../../glossary.md#term-spatial-axes))</span>
- GraphCapture and CompiledGraphModel provide the existing graph, but no public cold reconstruction.
  Gradle and NnTrainingDependencyContractTest/CheckpointDependencyContractTest enforce the old
  edges; isolated nnApiConsumer/trainingApiConsumer source sets expose migration obligations.<span data-glossary-navigation="true"> ([CompiledGraphModel glossary](../../glossary.md#compiled-graph--compiledgraphmodel); [cold reconstruction glossary](../../glossary.md#term-cold-reconstruction); [Gradle glossary](../../glossary.md#term-gradle))</span>

## Accepted product decisions

1. Common Model-facing tensor compute/configured-train/save/load for both origins; keep existing
   generic Model<I,O>/arbitrary Java forward. No mandatory TensorModel, LoadedGraph, Runner or
   ModelExecutor hierarchy; no arbitrary Java callback/record serialization.<span data-glossary-navigation="true"> ([Model glossary](../../glossary.md#term-model); [tensor glossary](../../glossary.md#tensor); [callback glossary](../../glossary.md#term-callback); [serialization glossary](../../glossary.md#term-serialization))</span>
2. Lazy model-owned Engine.standard() for normal use, currently fresh CPU composition. Explicit
   Engine is borrowed, never closed by the model. Model.load(path[, limits]).withEngine(engine)
   configures that exact same ordinary Model before preparation; application-built models use the
   same method. Load is cold and opens no Engine or discovery.<span data-glossary-navigation="true"> ([Lazy glossary](../../glossary.md#term-lazy-evaluation); [Engine glossary](../../glossary.md#term-engine); [CPU glossary](../../glossary.md#term-cpu); [borrowed glossary](../../glossary.md#ownership-borrowing-and-lease))</span>
3. One model-owned working weight set feeds compute, train and save. Acquire external values once
   when necessary without changing their Tensor/storage; no ownership-only duplication of loaded
   values. One exact native-admission copy may still be necessary before training.<span data-glossary-navigation="true"> ([weight glossary](../../glossary.md#term-weight); [ownership glossary](../../glossary.md#term-ownership-meanings); [admission glossary](../../glossary.md#term-admission))</span>
4. Targeted source migration is permitted for generic Training/Checkpoint separation and the
   subsequently approved thin Engine session removal, not blanket redesign or compatibility
   frameworks. Retain old formats/readers and declarative behavior.<span data-glossary-navigation="true"> ([Checkpoint glossary](../../glossary.md#term-checkpoint))</span>
5. First new graph signature is explicit/static: [32,784] accepts fresh [32,784] and rejects
   [64,784], with no automatic recapture/recompile/prepare. Static Java unrolling is ordinary DAG,
   not a recurrent recipe/BPTT prerequisite.<span data-glossary-navigation="true"> ([graph glossary](../../glossary.md#graph); [signature glossary](../../glossary.md#term-signature); [prepare glossary](../../glossary.md#prepare); [unrolling glossary](../../glossary.md#term-unrolling); [DAG glossary](../../glossary.md#term-directed-acyclic-graph-dag); [BPTT glossary](../../glossary.md#term-backpropagation-through-time-bptt))</span>

## Options and rationale

Keep the status quo for existing flexible declarative models, but do not extend per-network recipes
as a prerequisite for ordinary [graph](../../glossary.md#graph) [persistence](../../glossary.md#term-persistence). Reject per-[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) forward reconstruction as normal
execution, serialized CompileArtifacts, a second public [model](../../glossary.md#term-model) hierarchy and merging Training into [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context).

Select fixed [tensor](../../glossary.md#tensor)-list execution methods on the existing Model class. This deliberately separates
symbolic Java output O from materialized values. It avoids unchecked casts and an I/O [codec](../../glossary.md#term-codec)/adapter
framework. The cost is that arbitrary Model<I,O> is not automatically executable or serializable:
ordinary NN users select a Tensor [signature](../../glossary.md#term-signature) once through defineTensor; advanced arbitrary-record
models may explicitly select their Tensor boundary. Both use the same ordinary Model workflow as load.

The approved [Engine](../../glossary.md#term-engine) amendment chooses existing Engine [PreparedExecution](../../glossary.md#prepared-execution--preparedexecution) over retaining a thin
[InferenceSession](../../glossary.md#former-inference-session--inferencesession) or renaming it to another executor/runner. One public prepared [owner](../../glossary.md#term-owner) reduces
[lifecycle](../../glossary.md#lifecycle)/[API](../../glossary.md#term-application-programming-interface-api) duplication; Model's high-level compute/train hides preparation anyway. The cost is
explicit source/[binary](../../glossary.md#term-binary-descriptor-and-metadata) migration of session users and the direct-[run](../../glossary.md#term-run) precedence change below.
[TrainingSession](../../glossary.md#training-session--trainingsession) is retained: [optimizer history](../../glossary.md#term-optimizer-history), accumulation, counters and [transaction](../../glossary.md#term-transaction)/busy state
are substantive [ownership](../../glossary.md#term-ownership-meanings), not a redundant [inference](../../glossary.md#term-inference) wrapper. Engine PreparedExecution versus
[Runtime](../../glossary.md#term-runtime) PreparedExecution is also retained: outward owner/adapters versus inward unique [lease](../../glossary.md#term-lease)
and prepared resources are separate boundaries. Advanced lifecycle remains supported.

For Model [binding](../../glossary.md#term-binding), select a fluent in-place configuration method over a copied model, request
builder, compute wrapper or global Engine/session. It preserves one graph and [weight](../../glossary.md#term-weight) owner and
keeps compute terminal. The cost is a strict pre-preparation freeze, including failed preparation
attempts; callers cannot switch engines to retry after that boundary.

## Selected [API](../../glossary.md#term-application-programming-interface-api) and type feasibility

**Implemented and accepted in NN0028, Complete on 2026-10-19.**
[Model](../../glossary.md#term-model) remains abstract, extends [Module](../../glossary.md#term-module) and gains AutoCloseable. Existing Model factories and forward
[signatures](../../glossary.md#term-signature) are unchanged. Types below use their existing owning packages unless stated otherwise.

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

Private implementations specialize Model<[Tensor](../../glossary.md#tensor),Tensor> or Model<List<Tensor>,List<Tensor>>;
the existing abstract base need not narrow I or O. No cast from O occurs: compute(Tensor) returns
[HostTensorValue](../../glossary.md#host-tensor-value--hosttensorvalue) and compute(List<Tensor>) returns List<HostTensorValue>, never O. Single-input/output
methods validate [arity](../../glossary.md#term-arity). Advanced static loadGraph(Path) (and limits/[borrowed](../../glossary.md#ownership-borrowing-and-lease)-[Engine](../../glossary.md#term-engine) overload) returns
Model<List<Tensor>,List<Tensor>> for multi-boundary files; ordinary load rejects those with a loadGraph hint.
The advanced [graph](../../glossary.md#graph) implementation's forward(List<Tensor>) constructs ordinary symbolic outputs through cold source
reconstruction with the supplied [descriptor](../../glossary.md#term-descriptor)-compatible input [leaves](../../glossary.md#term-leaf) and the same owned state.
That advanced symbolic call does not execute or change the cached source; compute never calls it
per [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters). Existing Model.define and custom subclasses keep their exact forward behavior.
Their new tensor methods fail with UnsupportedOperationException before Engine/file work unless
they provide the tensor capability; their default close owns nothing and is a no-op.

Normal defineTensor invokes the typed definition once, creates its declared [prototype](../../glossary.md#term-prototype) internally,
and constructs forward once to initialize deferred layers and establish the source before returning.
It discovers registered state itself. [Callback](../../glossary.md#term-callback)-supplied state is borrowed until privately acquired;
copy/reconstruct and strict installation into the adopted child wrappers precede [publication](../../glossary.md#term-publication). Original
caller Tensors/stores are never rebound. Internally owned reconstructed/loaded state is adopted,
not copied again. Definition/forward side effects are not generally rollbackable; failed construction
publishes no Model, closes newly owned resources and does not claim to undo arbitrary callback effects.
The NEW defineTensor discards its definition/forward callbacks after prototype source construction.
Its symbolic forward(Tensor) reconstructs from the frozen owned source/leaves on an explicit cold
call, just like the loaded single-Tensor model. It never invokes the old closure, even if that closure
captured a pre-acquisition [weight](../../glossary.md#term-weight) Tensor. Compute/train/save use the same owned source without
per-batch reconstruction. Callers must not mutate adopted child modules. Legacy Model.define retains
its existing callback behavior unchanged; this one-shot policy is only the new defineTensor contract.

For an advanced existing Model<MyInput,MyOutput>, call its forward once using explicit prototype Tensors,
select the Tensor outputs from MyOutput and enumerate named state leaves; pass those to Model.graph.
No MyInput/MyOutput adapter, callback retention, reflective record discovery or record restoration
is required. Normal load returns Model<Tensor,Tensor> like defineTensor; loadGraph matches the advanced graph factory.
This is source-level feasible through ordinary concrete generic [specialization](../../glossary.md#term-specialization); no javac experiment
or unchecked-cast assumption is needed. [Compile](../../glossary.md#compile)-only API consumers must witness it at implementation.

### Necessary boundary types, not a new [model](../../glossary.md#term-model) hierarchy

- `training.TrainingParameters`: `List<Entry> entries()`, `void validateUnchanged()`;
  nested `Entry(String path, Tensor tensor)`. Membership is immutable for a borrow, but validation
  reads the live owner. NN's new <span data-glossary-navigation="true"> ([immutable glossary](../../glossary.md#term-immutable); [validation glossary](../../glossary.md#term-validation))</span>`Module.trainingParameters()` returns a private adapter retaining
  exact ordered paths/wrappers/Tensors. Training owns storage/descriptor/finiteness validation.
  This boundary and the NN adapter are implemented by Training0007. Training has exactly public
  Model/Engine edges; NN now also exports Checkpoint/Engine for the moved persistence signatures.
  GraphCheckpoint below is Complete in Checkpoint0008 after final review/coordinator gate.<span data-glossary-navigation="true"> ([Engine glossary](../../glossary.md#term-engine))</span>
- `checkpoint.graph.GraphCheckpoint`: [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)-free cold [persistence](../../glossary.md#term-persistence) [owner](../../glossary.md#term-owner) with nested
  `Source(List<Tensor> inputs, List<Tensor> outputs, List<Leaf> leaves)`,
  `Leaf(String name, Role role, Tensor tensor)` and enum PARAMETER/BUFFER/CONSTANT.
  Source is ordinary Tensor references plus roles, not another graph IR, executable or lifecycle
  owner. Public <span data-glossary-navigation="true"> ([IR glossary](../../glossary.md#term-intermediate-representation-ir); [lifecycle glossary](../../glossary.md#lifecycle))</span>`copy(Source)`, `save(Path, Source, ArtifactLimits)`,
  `load(Path, ArtifactLimits)` and `rebind(Source, List<Tensor>)` respectively acquire detached
  host state, persist it, restore it, or construct symbolic outputs. Save/load declare IOException.
  Copy/rebind are cold, never a batch loop.<span data-glossary-navigation="true"> ([detached glossary](../../glossary.md#detached-model-results); [rebind glossary](../../glossary.md#capture-graph-ports-and-reconstruction))</span>
- Compiler0008 implements on existing GraphCompilationPort `captureSource(List<Tensor>)` returning nested
  `SourceCapture(CompiledGraphModel graph, List<Tensor> leaves)`, with leaves aligned to graph.inputs,
  and `reconstructSource(CompiledGraphModel, List<Tensor>) -> List<Tensor>`.
  This integration-only sidecar supplies exact source identity; compiled artifacts still retain no
  Tensor. GraphCheckpoint hides Compiler types from NN. No Compiler registry or second DAG.<span data-glossary-navigation="true"> ([artifacts glossary](../../glossary.md#term-artifact); [registry glossary](../../glossary.md#term-registry); [DAG glossary](../../glossary.md#term-directed-acyclic-graph-dag))</span>

Compiler0008 source reconciliation (2026-10-17): [capture](../../glossary.md#term-capture-meanings) returns the **raw** [graph](../../glossary.md#graph), validated by
existing [inference](../../glossary.md#term-inference) without optimization or logical-[layout](../../glossary.md#layout) closure, and exact [leaves](../../glossary.md#term-leaf) collected when
that same traversal assigns graph inputs. Reconstruction accepts full-[descriptor](../../glossary.md#term-descriptor)-compatible
ordinary [Tensor](../../glossary.md#tensor) expressions as positional replacements, not only [provenance](../../glossary.md#provenance)-free leaves: this is
required for cold symbolic forward/composition. Their upstream provenance remains ordinary input
to later [compile](../../glossary.md#compile)/[autograd](../../glossary.md#autograd), not added source-[codec](../../glossary.md#term-codec) coverage. Captured sidecar leaves remain unique
provenance-free [identities](../../glossary.md#term-identity-meanings). Repeated replacement references are explicit substitution; output
positions remain ordered and ordinary compile still rejects duplicate resolved outputs. Preserve
resolved/unresolved descriptors exactly; do not strip final compiled layouts to force reconstruction.
[Compiler](../../glossary.md#term-compiler) checks logical [metadata](../../glossary.md#term-metadata) only; [Checkpoint](../../glossary.md#term-checkpoint)'s [host](../../glossary.md#term-host)-value [admission](../../glossary.md#term-admission) below remains separate.

Checkpoint0008 source reconciliation (2026-10-17): declared input order is not capture's first-encounter
order, which also includes named state. Associate by exact Tensor [identity](../../glossary.md#term-reference-identity); retain unused declared
[prototype](../../glossary.md#term-prototype) slots using existing graph constructors and Compiler's unused-input support, without saved
[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) values. Unreachable named registrations still reject. Host reconstruction retains the original
descriptor through TensorFactory.create and compatible storage, never resolves an absent layout merely
to call allocate. These are consumer consequences of current source APIs, not additional Compiler [scope](../../glossary.md#term-scope).

These are the minimum two cross-[module](../../glossary.md#term-module) data/admission boundaries plus a Compiler capture sidecar;
no public execution result, [signature](../../glossary.md#term-signature) builder, transfer [token](../../glossary.md#term-token) or generic adapter framework is added.
Private graph Model uses existing [Parameter](../../glossary.md#term-parameter)/[Buffer](../../glossary.md#term-buffer) wrappers. Dotted state paths use private
state-namespace Module children only where needed to preserve inherited final stateDictionary/
parametersRecursively behavior; these children have no forward recipe or execution role.
Require [canonical](../../glossary.md#term-canonical-form) Module traversal order and reject prefix/name collisions; do not silently rename
paths or keep a second authoritative [weight](../../glossary.md#term-weight) map. [CONSTANT](../../glossary.md#term-constant) leaves remain outside trainable state.
This private namespace cost is necessary because existing traversal methods are final.

## Source, [shape](../../glossary.md#shape), state and [ownership](../../glossary.md#term-ownership-meanings)

[Graph](../../glossary.md#graph) factory requires a complete source [DAG](../../glossary.md#term-directed-acyclic-graph-dag) and explicit ordered input [prototypes](../../glossary.md#term-prototype). Every reachable
non-input [leaf](../../glossary.md#term-leaf) has exactly one role; repeated named [aliases](../../glossary.md#aliasing-and-arena) of one [Tensor](../../glossary.md#tensor) retain one leaf/store,
while equal-valued distinct leaves stay distinct. Reject unreachable registrations, stale outputs
referring to replaced state, conflicting roles, unsupported [views](../../glossary.md#term-view-meanings) and incomplete values.
requiresGrad alone never establishes a [parameter](../../glossary.md#term-parameter) role. First graph source uses static shapes and
[canonical](../../glossary.md#term-canonical-form) non-view [host](../../glossary.md#term-host) values; [descriptors](../../glossary.md#term-descriptor) retain requiresGrad and resolved/unresolved [layout](../../glossary.md#layout)
distinctions. Storage capacity is validated from canonical static geometry when layout is unresolved.
This is separate from, and does not narrow, existing flexible declarative/v1 behavior.

Acquisition copies each external state [identity](../../glossary.md#term-identity-meanings) once into fresh private host leaves, then reconstructs
the ordinary expressions against those leaves before [publication](../../glossary.md#term-publication). The caller's original Tensor,
storage association and bytes are unchanged. No retained original-[weight](../../glossary.md#term-weight) graph hides behind the
owned values. The fully restored load result is adopted internally without another ownership copy.
No public transfer [API](../../glossary.md#term-application-programming-interface-api) is needed for this first capability; any later exclusive transfer must name
and transfer the actual [resource owner](../../glossary.md#ownership-borrowing-and-lease). Old generic [Model](../../glossary.md#term-model).forward ownership is unchanged.

Save before first compute works because graph construction or load already establishes source and
[signature](../../glossary.md#term-signature). Save does not [run](../../glossary.md#term-run) Java forward, infer a signature from the last [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) or create an [Engine](../../glossary.md#term-engine).
Only host-readable graph values are in this first path; [device](../../glossary.md#term-device)-only source leaves reject rather than
trigger hidden execution. Old [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [checkpoint](../../glossary.md#term-checkpoint) save still uses public Engine [materialization](../../glossary.md#materialization).

Mode and state guards compare the source-bound mode plus current path/order/wrapper/Tensor/storage
identities before compute, configured train, symbolic forward or save. A changed mode/[binding](../../glossary.md#term-binding) closes
cached [inference](../../glossary.md#term-inference) preparations/[TrainingSession](../../glossary.md#training-session--trainingsession) and puts the graph capability into an invalid state; the call rejects, without
recapture or a silent reset of [optimizer state](../../glossary.md#term-optimizer-state). Build an explicit replacement graph Model from the
new source to proceed. Existing Parameter.replace/loadStateDictionary still have their current
[borrowed](../../glossary.md#ownership-borrowing-and-lease)-reference semantics; they are not the new model's ownership-acquisition API. Never train or
save newly borrowed replacement state via the old graph. Ordinary successful Training updates change
bytes in the same owned stores and remain valid for both compute and save.

[Module](../../glossary.md#term-module).train() and eval() remain mode toggles. Both graph and defineTensor construction select [evaluation](../../glossary.md#term-evaluation) mode before the guarded source is fixed
(defineTensor sets its adopted tree to eval before the single forward); it does not promise to infer which
ForwardContext a caller used. configureTraining does not call Module.train() or reinterpret a
mode-sensitive source. Its objectiveBuilder runs once on owned output expressions and explicit
[target](../../glossary.md#term-target) prototypes; it must be pure, return one supported [scalar](../../glossary.md#term-scalar) [FLOAT32](../../glossary.md#term-float32)/[FLOAT64](../../glossary.md#term-float64) [objective](../../glossary.md#term-objective) and retain
no [runtime](../../glossary.md#term-runtime) [callback](../../glossary.md#term-callback) role. Training-mode BatchNorm [buffer](../../glossary.md#term-buffer) updates and advancing [dropout](../../glossary.md#dropout--inverted-dropout) [RNG](../../glossary.md#graph-rng-state--graphrngstate) are not
added. A mode-specific DAG must be explicitly constructed as such.

configureTraining may succeed once, before the first train; after success repeat configuration
rejects rather than discarding [optimizer history](../../glossary.md#term-optimizer-history). Failed attempts may be explicitly retried under
the Engine freeze below. It validates the configuration, acquires shared native storage where
needed, and constructs the objective/session against the final owned state. Native [admission](../../glossary.md#term-admission) closes
any inference preparation before rebinding storage; preserve Tensor identity and exact alias groups.
Already-native admitted owned values are not copied. [Heap](../../glossary.md#term-heap) values may need one exact copy.
A failed copy leaves old associations/bytes intact; preparation failure closes unpublished resources
and leaves no active [Training session](../../glossary.md#training-session--trainingsession). Subsequent explicit retry may reuse already-admitted storage,
never recopy merely for ownership. Inference can [prepare](../../glossary.md#prepare) once again after this explicit transition,
not after every batch/update.

Training's native-only domain, duplicate registered Tensor/overlapping-store rejection, complete
live pre-run/[precommit](../../glossary.md#term-precommit) [validation](../../glossary.md#term-validation), [optimizer](../../glossary.md#term-optimizer) algorithms, slots/counters and [failure atomicity](../../glossary.md#term-failure-atomicity) remain.
Repeated DAG use of one uniquely registered weight is not duplicate registration; aliased parameter
paths that current Training rejects still reject, never split to evade checks. The bounded
Training0008 cutover admits
`TrainingStep runValues(List<Tensor> values, GradientMode mode)` for fresh stored values in exactly
`TrainingSession.inputs()` order, excluding captured parameters and the private [seed](../../glossary.md#term-seed). Existing run
keeps exact-identity/arbitrary-order binding and its diagnostics. Both share the same [transaction](../../glossary.md#term-transaction);
Training's nonempty overlap rejection still applies, including repeated supplied values. The root
Core [lifecycle](../../glossary.md#lifecycle) and owning Training contract enact this admission; this [ADR](../../glossary.md#term-architecture-decision-record-adr) is not its authority.

## [Engine](../../glossary.md#term-engine), result and exclusion [lifecycle](../../glossary.md#lifecycle)

<a id="target-model-engine-binding"></a>

### [Model](../../glossary.md#term-model) [Engine](../../glossary.md#term-engine) [binding](../../glossary.md#term-binding)

NN0028 implements this binding under the owning [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) contract; it does not change Engine0021.
`withEngine(Engine)` returns `this` by reference, with the same generic Model type, [graph](../../glossary.md#graph), wrappers,
state and [ownership](../../glossary.md#term-ownership-meanings). It only selects the [borrowed](../../glossary.md#ownership-borrowing-and-lease) Engine used by both compute and configured
training. It does not copy state, move memory, create an executor/request, [compile](../../glossary.md#compile), [prepare](../../glossary.md#prepare) or [run](../../glossary.md#term-run).
`compute(batch)` remains terminal and returns [detached](../../glossary.md#detached-model-results) [`HostTensorValue`](../../glossary.md#host-tensor-value--hosttensorvalue); there is no
`compute().setEngine`, request builder, execution wrapper or public/global engine/session hierarchy.

Before the first preparation attempt, an open idle [tensor](../../glossary.md#tensor)-capable Model may select or replace a
borrowed Engine, including selecting the same one again. It never closes the previous or new
borrowed Engine. With no explicit selection, execution lazily creates one model-owned
`Engine.standard()`, currently [CPU](../../glossary.md#term-cpu)-only; there is no discovery or global default. Construction,
load, save and explicit cold symbolic forward do not create it. Existing explicit borrowed-Engine
factory/load overloads remain as declared above (including defineTensor and loadGraph variants);
they establish the same selection and ownership, not a second binding mechanism.

[Admission](../../glossary.md#term-admission) checks a previously selected Engine's closure first, then Model closing/closure, then
busy state, before inspecting [arguments](../../glossary.md#term-argument). An admitted withEngine checks tensor capability, then
the binding freeze, then rejects null with `NullPointerException` and a closed candidate with
`IllegalStateException`. Closed/closing/busy/frozen failures are `IllegalStateException`;
unsupported generic Model capability is `UnsupportedOperationException`. Rejection leaves the
selection unchanged and [releases](../../glossary.md#commit-release-distribution-log-and-seed-meanings) admission. Even the same Engine is rejected after freeze;
a closed selected Engine is not replaceable, and is never replaced automatically.
`Model.close()` bypasses ordinary Engine/capability/busy admission: it establishes closure, waits
for admitted work and attempts all owned cleanup regardless of the selected Engine's state.
Repeated/concurrent close replays the retained cleanup result, including failure; a closed borrowed
Engine must never prevent [release](../../glossary.md#term-release) of model-owned resources.

Freeze occurs when the first admitted compute or configureTraining has passed its argument and
live-source checks and enters preparation setup, **before** [lazy](../../glossary.md#term-lazy-evaluation) Engine creation, native admission,
[compilation](../../glossary.md#compiler-engine-and-execution-stage-names) or preparation. No engine/configuration argument can then [rebind](../../glossary.md#capture-graph-ports-and-reconstruction) it. Freeze survives
Engine-creation, [objective](../../glossary.md#term-objective)-builder, native-copy, compile or prepare failure; an explicit retry uses
the same selected Engine or still-uncreated standard choice. Invalid arguments rejected before
setup do not freeze. A train without successful configuration rejects without preparation.
This early, irreversible boundary prevents a partial failed setup from permitting a different Engine.

configureTraining may follow [inference](../../glossary.md#term-inference) and may explicitly replace [heap](../../glossary.md#term-heap) storage with native storage
under the existing transition above, closing/rebuilding affected preparations on the **same** Engine.
It does not reopen binding. There is no automatic reprepare after rebinding; rebinding is possible
only before any setup. One successful training configuration is final; failed configuration may
retry on the frozen choice with already-admitted storage. Model.close closes its lazily created
Engine only; a borrowed Engine remains caller-owned. Load never inspects [backend](../../glossary.md#backend) availability.

### Implemented [Engine](../../glossary.md#term-engine) boundary and [Model](../../glossary.md#term-model) results

Engine0021 removed the former [`InferenceSession`](../../glossary.md#former-inference-session--inferencesession)/`Engine.session` and session-only Engine/AdvancedEngine
delegation/registration branches. The implemented Engine surface uses existing `Engine.prepare(graph)` and adds
`Engine.runValues(PreparedExecution, List<Tensor>) -> RunResult`. The [owner](../../glossary.md#term-owner)-bound prepared [graph](../../glossary.md#graph)'s
inputs authenticate positional order; no new public type/[token](../../glossary.md#term-token) is introduced. Unlike exact-ID,
arbitrary-order `Engine.run(prepared, inputs)`, IDs may be fresh. Validate exact count and full
static [descriptor](../../glossary.md#term-descriptor) ([dtype](../../glossary.md#data-type--datatype), [shape](../../glossary.md#shape), [layout](../../glossary.md#layout), requiresGrad), present/live/current-thread-accessible
storage and capacity before any borrow. Resolved layout uses its checked [referenced span](../../glossary.md#referenced-element-span);
unresolved layout uses checked [canonical](../../glossary.md#term-canonical-form) static [element](../../glossary.md#term-element)/byte geometry, rejecting [overflow](../../glossary.md#term-overflow)/unknown
geometry. That added unresolved-layout check is scoped to runValues, not a silent [run](../../glossary.md#term-run) change.
[Snapshot](../../glossary.md#term-snapshot) associations once and reuse prepared [metadata](../../glossary.md#term-metadata)/direct adapters and isolated [Runtime](../../glossary.md#term-runtime) state;
no [prototype](../../glossary.md#term-prototype) replacement, synthetic [Tensor](../../glossary.md#tensor), extra per-element work, [capture](../../glossary.md#term-capture-meanings), [compile](../../glossary.md#compile), [prepare](../../glossary.md#prepare),
provider/[registry](../../glossary.md#term-registry) [lookup](../../glossary.md#term-lookup) or discovery per call.
At high level, public input order is mapped once to compiled order; [parameters](../../glossary.md#term-parameter)/[constants](../../glossary.md#term-constant) fill their
own precomputed positions. Same-shape input values may differ; [64,784] never specializes [32,784].

Both ordinary run methods use Engine [admission](../../glossary.md#term-admission), non-null/exact-owner prepared [validation](../../glossary.md#term-validation), exact
inward delegate/closed gate, then inputs. This deliberately changes direct run's failure precedence:
Previously input validation/[borrowing](../../glossary.md#term-borrowing) could precede closed-prepared discovery; now closed prepared
wins null/malformed/dead inputs without borrowing, as the former session did. Null/foreign prepared
validation also precedes input-list errors. Exact-ID membership/order and open-valid-handle
diagnostics remain. Runtime's unique nonwaiting [lease](../../glossary.md#term-lease) authority still arbitrates a subsequent close.
Publish exactly one outward prepared registration; [rollback](../../glossary.md#term-rollback) failed prepare/wrapper/[publication](../../glossary.md#term-publication).
Engine shutdown waits for admitted operations then closes results before preparations before
integrations, preserving reverse order, attempt-all cleanup, suppression and retained failures.

Model compute materializes all ordered outputs under an aggregate bound (default Long.MAX_VALUE,
like Engine.compute), closes every [RunResult](../../glossary.md#run-result--runresult) before returning, and returns [detached](../../glossary.md#detached-model-results) [HostTensorValue](../../glossary.md#host-tensor-value--hosttensorvalue)
objects. Compute results and TrainingStep remain readable after model/input/Engine close.
Low-level RunResult remains independently owned: prepared close does not close it; Engine close does,
and caller input storage must stay usable/unmodified until result close. No new result [lifetime](../../glossary.md#term-lifetime).

The first high-level model admits one [operation](../../glossary.md#operation) at a time; a competing compute/train/save/configure/withEngine
fails busy before [argument](../../glossary.md#term-argument) inspection. Close rejects new work, waits for admitted work, closes
[TrainingSession](../../glossary.md#training-session--trainingsession)/[inference](../../glossary.md#term-inference) preparations before owned native [arenas](../../glossary.md#term-arena) and finally its owned Engine, attempting all cleanup and retaining
failures. [Borrowed](../../glossary.md#ownership-borrowing-and-lease) Engine closure causes subsequent work to fail, never opens a replacement.
Caller input/state mutation and closure are excluded for each whole operation; outside mutation of
owned working state is forbidden. Training additionally rejects overlapping input/parameter regions.
Underlying Engine preparations retain their concurrent isolated-run capability; this convenience adds
no concurrent training/inference promise or [synchronization](../../glossary.md#term-synchronization) to inherited [Module](../../glossary.md#term-module) mutators.

## Source migration and retired code

Selected direct edges: generic Training api([Model](../../glossary.md#term-model),[Engine](../../glossary.md#term-engine)); generic [Checkpoint](../../glossary.md#term-checkpoint) api(Model) plus
implementation([Compiler](../../glossary.md#term-compiler)); [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) api(Model,Engine,Training,Checkpoint). New [graph](../../glossary.md#graph) values are [host](../../glossary.md#term-host)-only,
so generic Checkpoint needs no Engine edge. The nine moved NN glue classes retain their Engine use.
Preserve `:extensions:training` and `:extensions:checkpoint` project IDs through projectDir to
[modules](../../glossary.md#term-module)/training and modules/checkpoint, plus current training/checkpoint.[artifact](../../glossary.md#term-artifact) packages.

| Existing source/build call | [Target](../../glossary.md#term-target) migration |
|---|---|
| Engine.session(graph) → [InferenceSession](../../glossary.md#former-inference-session--inferencesession) | Engine.[prepare](../../glossary.md#prepare)(graph) → existing engine.[PreparedExecution](../../glossary.md#prepared-execution--preparedexecution); remove type/factory, no shim. |
| InferenceSession.[run](../../glossary.md#term-run)(originalInputs) | engine.run(prepared, originalInputs), preserving exact-ID/arbitrary-order [binding](../../glossary.md#term-binding); fresh values explicitly use engine.runValues(prepared, compiledOrderValues). |
| InferenceSession.compiledGraph/isClosed/close | Existing prepared.compiledGraph/isClosed/close; independently owned results are not closed by prepared close. |
| [TrainingSession](../../glossary.md#training-session--trainingsession) inferenceSession field/constructor/open/executeAndDecode/isClosed/beginOperation/close | Engine0021 mechanical Engine PreparedExecution/prepare/run migration; preserve public TrainingSession, [transaction](../../glossary.md#term-transaction), prepared-before-[seed](../../glossary.md#term-seed) cleanup and failed-open [rollback](../../glossary.md#term-rollback). |
| CpuLifecycleBenchmark helper [signatures](../../glossary.md#term-signature); EngineMixedOwnerTestAccess reflection; Engine typed/advanced public-[shape](../../glossary.md#shape) and [lifecycle](../../glossary.md#lifecycle)/integration tests | Engine + PreparedExecution calls; migrate reflective target and [API](../../glossary.md#term-application-programming-interface-api) assertions, including var-typed Engine.session calls; no benchmark execution or [backend](../../glossary.md#backend) redesign. |
| TrainingSession.open(engine, module, [objective](../../glossary.md#term-objective), [optimizer](../../glossary.md#term-optimizer)[, groups]) | TrainingSession.open(engine, module.trainingParameters(), objective, optimizer[, groups]); remove Module overloads atomically with consumers. |
| [TrainingParameterStorage](../../glossary.md#native-parameter-copy--trainingparameterstorage).copyToSharedNative(module, [arena](../../glossary.md#term-arena)) | copyToSharedNative(module.trainingParameters(), arena); advanced caller still owns Arena/exclusion. |
| trainingApiConsumer imports NN Module through Training alone | Generic consumer accepts [TrainingParameters](../../glossary.md#live-parameter-source--trainingparameters) with Training-only [classpath](../../glossary.md#term-classpath); NN caller explicitly depends on NN and calls its adapter. |
| checkpoint.model.ModelCheckpoint / ModelCheckpointInspection / InputArtifactFingerprint imports | nn.checkpoint equivalents, with an explicit :extensions:nn [dependency](../../glossary.md#term-dependency) for old NN checkpoint users. Method behavior/file readers retained. |
| ModelArtifactCapture/Candidate/[Codec](../../glossary.md#term-codec), ModelStateRestoration, WholeModelComposition/DefinitionCodec | Move privately with the other three types to NN; old wire IDs/defaults/[preflight](../../glossary.md#term-preflight)/[publication](../../glossary.md#term-publication) preserved. |
| ArtifactFile/Envelope/Limits users | Same checkpoint.artifact imports and :extensions:checkpoint dependency; no NN transitivity after split. |
| Model.define(...).forward(input), DeclarativeModel and ModelSpecification | Unchanged. New graph opt-in uses explicit [Tensor](../../glossary.md#tensor) boundary; no implicit fixed-[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) conversion of old files. |

No compatibility layer/project. Engine0021 removed InferenceSession and Engine.session plus
Engine.runSession, AdvancedEngine.openInferenceSession/runSessionOrdinary/unregister(InferenceSession)
and session-only branches atomically with all consumers. External session users must migrate and
recompile; this is a deliberate source/[binary](../../glossary.md#term-binary-descriptor-and-metadata) break, not an additive-only API claim. Training [fixture](../../glossary.md#term-fixture)
session() accessors returning TrainingSession remain. Training0007 retires old NN-taking
Training declarations/imports in its coordinated migration; Checkpoint0008 retires moved locations.
Do not delete algorithms,
readers, generic Model, declarative types or completed history. The superseded outputs(actualInputs)-
first proposal was never implemented and has no production code to delete.

## [Owned Model](../../glossary.md#model-module-buffer-and-scan-distinctions) usage

Goal: ordinary [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) construction, save before compute, fresh [batches](../../glossary.md#term-batch-machine-learning-and-hyperparameters), training and load without
manual forward, [leaf](../../glossary.md#term-leaf) enumeration or a second [Model](../../glossary.md#term-model). Imports are Model/[Linear](../../glossary.md#term-linear-meanings) from nn.[module](../../glossary.md#term-module)/
nn.layers, [Tensor](../../glossary.md#tensor)/TensorDescriptor/TensorFactory from model.tensor, [Shape](../../glossary.md#shape)/LayoutDescriptor/[DataType](../../glossary.md#data-type--datatype)
from their Model packages, Sgd/TrainingStep from training, [HostTensorValue](../../glossary.md#host-tensor-value--hosttensorvalue)/[Engine](../../glossary.md#term-engine) from engine, and
ordinary Java Path/Optional. The implemented Model calls compose those existing public [owners](../../glossary.md#term-owner).

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
// Optional explicit binding: caller owns the Engine.
try (Engine engine = Engine.standard()) {
    try (Model<Tensor, Tensor> restored = Model.load(path).withEngine(engine)) {
        HostTensorValue value = restored.compute(secondBatch); // same ordinary Model, zero [32,1]
    } // closes model resources, not engine
}
```

The explicit [signature](../../glossary.md#term-signature) is declared once, before any batch. Values/results are zero because both
[weights](../../glossary.md#term-weight) and inputs are zero; use nonzero inputs/[targets](../../glossary.md#term-target) to observe learned updates. External weight
bytes and its storage association remain unchanged even for a nonzero update. [Detached](../../glossary.md#detached-model-results) results
survive model close. A fresh [64,784] input rejects; no hidden recompilation.
`Model.load(path, limits).withEngine(engine)` uses the same [binding](../../glossary.md#term-binding) policy. Calling withEngine
after the first preparation setup rejects, even if preparation failed; configure it beforehand.

Advanced [graph](../../glossary.md#graph)(inputs,outputs,namedLeaves) is optional for raw Tensor graphs, not mandatory NN UX.
Old whole-model.v1 still loads through moved ModelCheckpoint.load into a flexible DeclarativeModel.
Model-only save does not [resume](../../glossary.md#schema-serialization-and-checkpoint) [optimizer](../../glossary.md#term-optimizer) slots, [RNG](../../glossary.md#graph-rng-state--graphrngstate), cursor, scheduler or scaler; configureTraining
is supplied anew after load. No automatic exact-continuation claim.

## Coordinated authority cutovers and [validation](../../glossary.md#term-validation)

### [CNN](../../glossary.md#term-convolutional-neural-network-cnn) and recurrent source successor

Accepted 2026-10-20: [CP0010](../../planning/extensions/checkpoint/tasks/0010-fluent-cnn-and-static-recurrent-graph-persistence.md)
and [CP0011](../../planning/extensions/checkpoint/tasks/0011-flat-recurrent-scan-persistence.md)
form one Class C packet and completion gate. The root routes source [admission](../../glossary.md#term-admission) to
[Compiler lifecycle](../../architecture/contracts/compiler-autograd.md#compile-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../glossary.md#lifecycle))</span> and the
wire and [normalization](../../glossary.md#term-normalization) to one owning
[graph-model v3 heading](../../architecture/contracts/extensions-training.md#canonical-graph-model-version-3)<span data-glossary-navigation="true"> ([graph glossary](../../glossary.md#graph))</span>.
The [recurrent purity amendment](0012-fixed-recurrent-scan-without-regions.md#purity-state-and-serialization)
permits cold semantic [persistence](../../glossary.md#term-persistence) without serializing prepared/native/[runtime](../../glossary.md#term-runtime) state.

Before this amendment, the builder constructed ordinary [convolution](../../glossary.md#term-channels-first-convolution-layers), [pooling](../../glossary.md#term-pooling), flatten and static
recurrent DAGs that eight-kind save admission rejected. Fixed scans exposed a separate cold-source
gap: [capture](../../glossary.md#term-capture-meanings) retained [canonical](../../glossary.md#term-canonical-form) sibling ports, but reconstruction, normalization and [topology](../../glossary.md#term-topology)
assumed one output. These gaps belonged to the existing [Compiler](../../glossary.md#term-compiler)/[Checkpoint](../../glossary.md#term-checkpoint) boundary, not to
new [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) recipes, APIs, types, [modules](../../glossary.md#term-module) or a generic serializer [registry](../../glossary.md#term-registry).

The selected source [slice](../../glossary.md#slice) uses the exact Compiler contract, biased default convolutions and
the six fixed [scan](../../glossary.md#term-cumulative-scan) overloads. All-family admission and per-network [serialization](../../glossary.md#term-serialization) were rejected: both
would widen [scope](../../glossary.md#term-scope) without solving canonical source reconstruction. An intermediate single-output
v3 followed by another scan format was also rejected. One v3 instead carries ordered
result [descriptors](../../glossary.md#term-descriptor) and implicit prefix-sum values for every [producer](../../glossary.md#term-producer). Producer-aware iterative
topology retains unused sibling ports without accepting dead producers or recursive traversal.
Cold [inference](../../glossary.md#term-inference) remains raw; [layout](../../glossary.md#layout) closure and optimized [artifacts](../../glossary.md#term-artifact) are not substitutes.

Smallest-version selection preserves byte-identical saves for old five-kind/eight-kind sources.
Appending new operations under v1/v2 would change published formats; writing every source as v3
would unnecessarily exclude old readers. New readers retain exact old tuples, and old readers
reject v3. The wire fields live only in the owning contract, not in this [ADR](../../glossary.md#term-architecture-decision-record-adr). ArtifactFile [framing](../../glossary.md#term-framing)
and old `model.v1`/`whole-model.v1` readers are unchanged.

The cost is a [finite](../../glossary.md#term-finite) supported slice plus explicit multi-output bookkeeping and malformed-wire
coverage. Compiler [preflight](../../glossary.md#term-preflight) remains descriptor-only, including [GATHER](../../glossary.md#gather) indices and runtime length
inputs; named [host](../../glossary.md#term-host) state is imported only after complete wire/source [validation](../../glossary.md#term-validation). Reconstruction
calls a scan constructor once, retaining every canonical wrapper even when only finalCell is a
root. Neither a length [prototype](../../glossary.md#term-prototype)'s bytes nor a scan result-[carrier](../../glossary.md#term-storage-carrier) object is persistent state.

The selected user workflow saves a default CNN or static recurrent [Model](../../glossary.md#term-model) before compute, closes
it, cold-loads it and uses fresh exact-descriptor [batches](../../glossary.md#term-batch-machine-learning-and-hyperparameters). Training is freshly configured where
current adjoints permit; [Conv3d](../../glossary.md#conv3d--ncdhw) remains inference-only. Advanced list Models may cold-persist
scans with external [INT64](../../glossary.md#data-type--datatype) lengths and [rebind](../../glossary.md#capture-graph-ports-and-reconstruction) them, but cannot execute or train scans. There is no
[backend](../../glossary.md#backend) scan adoption, [BPTT](../../glossary.md#term-backpropagation-through-time-bptt), [optimizer](../../glossary.md#term-optimizer) [resume](../../glossary.md#schema-serialization-and-checkpoint) or automatic continuation claim.

The coordinated contract/ADR cutover preceded dependent Java work. Validation covers golden wire
and malformed data, old byte compatibility, canonical multi-output [identity](../../glossary.md#term-identity-meanings), CNN/static recurrent
forward and independent [SGD](../../glossary.md#term-stochastic-gradient-descent-sgd) oracles, and cold scan roundtrips. The joint task briefs retain the
independent review evidence and coordinator's source-matched acceptance gate; this ADR does not
mark those tasks complete.

### [Activation](../../glossary.md#term-activation) source successor

Accepted 2026-10-19: the separately authorized
[Checkpoint0009](../../planning/extensions/checkpoint/tasks/0009-owned-model-activation-source-and-persistence.md)
adds only RELU/[TANH](../../glossary.md#term-tanh)/[SIGMOID](../../glossary.md#term-sigmoid) to the completed first [graph](../../glossary.md#graph) [slice](../../glossary.md#slice). This successor does not reopen
the completed [model](../../glossary.md#term-model)-execution program or supersede its [lifecycle](../../glossary.md#lifecycle)/[dependency](../../glossary.md#term-dependency) decisions.
[Compiler](../../glossary.md#term-compiler) owns cold [admission](../../glossary.md#term-admission)/reconstruction; [Checkpoint](../../glossary.md#term-checkpoint) owns the wire, as routed by the root
and enacted in their existing scoped contracts before dependent Java changes.

The existing five-kind limit prevented owned [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) construction because acquisition already calls
GraphCheckpoint.copy. Compiler-only [widening](../../glossary.md#term-widening) was unsafe: the old encoder defaulted to PERMUTE.
The [atomic](../../glossary.md#resource-transaction-and-rollback) change therefore uses explicit [fail-closed](../../glossary.md#term-fail-closed) kind/attributes/[arity](../../glossary.md#term-arity) admission on encoding
and explicit family reconstruction, without new public APIs, types, [kernels](../../glossary.md#kernel) or build edges.

The compatibility drivers are preserving existing bytes and keeping each version closed. Appending
activation IDs under v1 would change its contract; writing v2 for every graph would unnecessarily
exclude old readers from five-kind sources. Select the smallest sufficient version instead:
five-kind and zero-[node](../../glossary.md#node) sources retain byte-identical v1, while any added activation selects
graph-model.v2/SYNGRF02/version 2 with appended opcodes 6/7/8. All other fields, values and bounds
are unchanged. V2-only overhead is version admission logic, not a second graph representation.
Old readers reject v2; new readers accept both exact tuples and downgrade five-kind-only v2 on
re-save. ArtifactFile/model.v1/whole-model.v1 compatibility remains independent and unchanged.

The cost is a deliberately [finite](../../glossary.md#term-finite) eight-kind capability, not general [operation](../../glossary.md#operation) [serialization](../../glossary.md#term-serialization).
Tests preserve v1 golden bytes, add independent v2 [opcode](../../glossary.md#term-opcode)/header fixtures and exact-size bounds,
and compare a three-activation two-[Linear](../../glossary.md#term-linear-meanings) [MLP](../../glossary.md#term-multilayer-perceptron-mlp) against ordinary [Tensor](../../glossary.md#tensor)/[Engine](../../glossary.md#term-engine) forward and [SGD](../../glossary.md#term-stochastic-gradient-descent-sgd)
oracles. Independent code/documentation review and the coordinator's native-enabled gate remain
required before acceptance; the task brief records their status and implementation evidence.

### Completed first [slice](../../glossary.md#slice) cutovers

Current root and scoped rules win. Each implementation packet updates its exact authority,
explanations, [ADR](../../glossary.md#term-architecture-decision-record-adr) and affected architecture tests together **before code relies on the new rule**:

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
   Complete; it did not authorize the later cutovers on its own.<span data-glossary-navigation="true"> ([runtime glossary](../../glossary.md#term-runtime); [prepare glossary](../../glossary.md#prepare); [engine glossary](../../glossary.md#term-engine); [inference glossary](../../glossary.md#term-inference); [Run glossary](../../glossary.md#term-run); [Optimizer glossary](../../glossary.md#term-optimizer); [owner glossary](../../glossary.md#term-owner); [ownership glossary](../../glossary.md#term-ownership-meanings); [leases glossary](../../glossary.md#term-lease); [shape glossary](../../glossary.md#shape))</span>
2. Training0007: root Repository layout/Module-ownership routing/Dependency rules/Testing
   requirements and extensions-training Training/NN/Optimizer-training lifecycle: generic live
   admission, native-copy ownership, remove Training→NN before NN→Training. The<span data-glossary-navigation="true"> ([Dependency glossary](../../glossary.md#term-dependency))</span>
   [source-verified brief](../../planning/extensions/training/tasks/0007-generic-live-parameter-admission.md)
   records the coordinated implementation and validation. Root/scoped authority, not this plan,
   now enacts only this bounded generic admission/dependency/layout cutover.<span data-glossary-navigation="true"> ([validation glossary](../../glossary.md#term-validation))</span>
3. Compiler0008: root Explicit non-goals permits only a temporary graph-local ValueId→Tensor table
   inside cold validated source reconstruction; compiler-autograd modules/compiler and Compile
   lifecycle own family-constructor reconstruction, and Compile artifacts excludes source sidecars
   and Tensor retention. No gradient map exception, fabricated
   intermediate placeholders, Model execution or public generic derived-output constructor.
   The 2026-10-17 implementation enacts this bounded cold boundary only, with complete preflight
   before family construction and table disposal on return/failure. Ordinary compile/autograd
   stays unchanged; no dependency edge or downstream Model/Checkpoint capability is enacted.
   Final independent code/docs APPROVE and coordinator Class C checkpoint complete this cutover.<span data-glossary-navigation="true"> ([ValueId glossary](../../glossary.md#logical-valueid-versus-physical-memory-slot); [Tensor glossary](../../glossary.md#tensor); [autograd glossary](../../glossary.md#autograd); [Compile glossary](../../glossary.md#compile); [Compile artifacts glossary](../../glossary.md#compile-artifacts); [preflight glossary](../../glossary.md#term-preflight); [dependency edge glossary](../../glossary.md#term-dependency-edge))</span>
4. Checkpoint0008: root layout/ownership/dependencies/testing and extensions-training Checkpoint/
   Whole-model definition and composition move NN glue and authorize Checkpoint→Compiler cold
   implementation access; no reverse Compiler edge. First codecs: ADD/MUL/NEG plus MATMUL/PERMUTE with exact permutation [1,0] for ordinary Linear, multiple distinct
   roots, all seven raw value types, strict bounds/roles/aliases; unknown families reject.<span data-glossary-navigation="true"> ([MATMUL glossary](../../glossary.md#matrix-multiplication--matmul); [permutation glossary](../../glossary.md#permutation); [Linear glossary](../../glossary.md#term-linear-meanings); [aliases glossary](../../glossary.md#aliasing-and-arena))</span>
5. Training0008: coordinate root Core lifecycle and the owning modules/training and Optimizer/training
   lifecycle, focused explanations and this ADR before code relies on fresh admission. This bounded
   authority cutover is Complete after Euclid's final independent APPROVE and main's source-matched
   root gate (2026-10-19). Engine rules and dependency edges remain unchanged. NN0028 separately
   coordinates Model convenience/ownership, retaining old declarative behavior and Engine composition authority.
   Completed NN0028 tests cover borrowed Engine closed first, complete model-owned resource cleanup
   and repeated/concurrent Model.close failure replay.<span data-glossary-navigation="true"> ([coordinate glossary](../../glossary.md#axis-rank-scalar-vector-and-matrix); [borrowed glossary](../../glossary.md#ownership-borrowing-and-lease))</span>

Keep ArtifactFile v1 and old [model](../../glossary.md#term-model).v1/whole-model.v1 readers. New [graph](../../glossary.md#graph)-model.v1 uses explicit input
[descriptors](../../glossary.md#term-descriptor), [leaf](../../glossary.md#term-leaf) roles/[identities](../../glossary.md#term-identity-meanings), [operation](../../glossary.md#operation) attributes and [canonical](../../glossary.md#term-canonical-form) raw values; inputs are not
saved [batches](../../glossary.md#term-batch-machine-learning-and-hyperparameters). Validate complete data before [publication](../../glossary.md#term-publication); reuse force/revalidate/one-[atomic](../../glossary.md#resource-transaction-and-rollback)-move.
Persist [signed zero](../../glossary.md#term-signed-zero)/[NaN](../../glossary.md#term-nan) payloads without [normalization](../../glossary.md#term-normalization), reject invalid [BOOL](../../glossary.md#data-type--datatype) bytes before import.
[Finite](../../glossary.md#term-finite) closed codecs are sufficient; preserve canonical sibling-output [identity](../../glossary.md#term-identity-meanings) whenever admitted.
[Backend](../../glossary.md#backend)/[gradient](../../glossary.md#term-gradient) support remains independent; zero-[node](../../glossary.md#node) execution remains explicitly unsupported.

The [program](../../planning/model-execution-refactor.md#phases-and-owner-dag) owns the concrete [DAG](../../glossary.md#term-directed-acyclic-graph-dag)
and final coordinator signoff. The [Engine0021 brief](../../planning/modules/engine/tasks/0021-prepared-execution-fresh-value-binding.md)
records implementation, independent approval and the coordinator's successful cross-[module](../../glossary.md#term-module) gate.
Training0007 is Complete: generic live [admission](../../glossary.md#term-admission)/native copy and its [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) adapter passed final
independent review and the coordinator boundary gate. Compiler0008's bounded [API](../../glossary.md#term-application-programming-interface-api)/authority cutover
is Complete after final independent code/docs APPROVE and the coordinator [checkpoint](../../glossary.md#term-checkpoint); authorized
[Compiler](../../glossary.md#term-compiler) and Checkpoint frontiers: None. Checkpoint0008 is Complete after Halley's final APPROVE
and the coordinator root gate; its brief owns evidence. Training0008 is Complete after Euclid's
definitive APPROVE and main's root gate (2026-10-19); authorized Training frontier: None.
NN0028 is Complete; its brief owns final approval and coordinator acceptance evidence.
The bounded root/owning NN contract enacts its [lifecycle](../../glossary.md#lifecycle); this ADR itself never supplies authority.
