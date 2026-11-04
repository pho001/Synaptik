# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0038: Stored declarative models with [inference](../../glossary.md#term-inference) or owned training [resume](../../glossary.md#schema-serialization-and-checkpoint)

## Status

Accepted [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) decision — 2026-10-14; final independent review approved and coordinator verified NN0026
[Ready](../../glossary.md#term-ready). Accepted CP-only whole-[model](../../glossary.md#term-model) decision — 2026-10-14; final independent contract APPROVE
and coordinator verification authorized CP0005B implementation. CP0005B is Complete (2026-10-15),
with independent code/documentation approval, [module](../../glossary.md#term-module) tests, final [Javadoc](../../glossary.md#term-javadoc) and coordinator gates.
Future recurrent/Training/adapter decisions
remain Proposed, not accepted contracts or implementation authorization.
Replaces the Proposed direction in [ADR 0037](0037-stored-training-checkpoint-adapter.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr))</span>,
which is withdrawn as the primary product, not as evidence of useful low-level foundations.
The NN [scope](../../glossary.md#term-scope) is accepted and coordinated with the root and owning NN contract. NN0026
is Complete with implementation independently reviewed and coordinator-validated.
CP0005B is Complete; no successor is authorized Ready. NN0027 semantic
recurrent reconstruction and CP0005C [persistence](../../glossary.md#term-persistence) adoption are Draft rows only; Training0006/adapter0006 remain
Draft with Proposed decisions. No TrainingRun or adapter acceptance.

## Context

The user needs a usable [model](../../glossary.md#term-model) from a file, then a choice between [inference](../../glossary.md#term-inference) and resumed training.
Published ModelCheckpoint persists values against a caller-recreated [topology](../../glossary.md#term-topology); proposed 0037 added
TrainingState but still required manual model/[objective](../../glossary.md#term-objective) construction, digests and native [Arena](../../glossary.md#term-arena) work.
That cannot deliver the selected primary experience.

[NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)'s `Model<I,O>` can contain arbitrary Java forward code and record-shaped inputs/outputs.
Existing `ModelDefinition` is a [callback](../../glossary.md#term-callback), not serializable model data. `Sequential` orders [unary](../../glossary.md#term-unary)
[Tensor](../../glossary.md#tensor) layers; ModuleFactory creates fresh layers but retains no reconstructable specification.
[TrainingSession](../../glossary.md#training-session--trainingsession) requires native [parameters](../../glossary.md#term-parameter), compiles/prepares once, and closes its inward session
and [cotangent](../../glossary.md#cotangent) Arena, not the [borrowed](../../glossary.md#ownership-borrowing-and-lease) model, [Engine](../../glossary.md#term-engine) or parameter Arena.

## Decision drivers

- Load supported models without lambdas, restore callbacks, manual construction or [digest](../../glossary.md#term-digest) plumbing.
- Keep semantic reconstruction in [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context), recipe/[run](../../glossary.md#term-run) [lifecycle](../../glossary.md#lifecycle) in Training, codecs/files downstream.
- Preserve existing explicit APIs, ArtifactFile v1 and published `model.v1`.
- Reconstruct actual attributes, wiring and sharing, not merely compare a configuration [hash](../../glossary.md#term-hash).
- Make resource [ownership](../../glossary.md#term-ownership) and [fail-closed](../../glossary.md#term-fail-closed) capability limits observable and testable.

## Options considered

1. Retain 0037 as the primary product. Useful explicit foundation, but fails the automatic-[model](../../glossary.md#term-model) UX.
2. Serialize Java callbacks/objects, instantiate class names reflectively or use a plugin registry.
   Reject: unavailable application code, unsafe/nonportable reconstruction and wrong owner boundaries.<span data-glossary-navigation="true"> ([plugin glossary](../../glossary.md#term-plugin); [registry glossary](../../glossary.md#term-registry); [owner glossary](../../glossary.md#term-owner))</span>
3. Put all model/loss/session logic in Checkpoint, or add a general ModelExecutor/transaction framework.
   Reject: bypasses NN/Training and invents an execution/persistence framework.<span data-glossary-navigation="true"> ([loss glossary](../../glossary.md#loss); [Checkpoint glossary](../../glossary.md#term-checkpoint); [transaction glossary](../../glossary.md#term-transaction); [NN glossary](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [persistence glossary](../../glossary.md#term-persistence))</span>
4. Add a closed NN declarative specification, whole-model persistence and an optional adapter
   composing a Training-owned recipe and run. Propose this bounded option.<span data-glossary-navigation="true"> ([run glossary](../../glossary.md#term-run))</span>

<a id="nn-decision-selected-for-review"></a>

## Accepted [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) decision

NN owns a new [immutable](../../glossary.md#term-immutable) `ModelSpecification` and `DeclarativeModel extends Model<Tensor, Tensor>`
in `io.github.pho001.synaptik.nn.module`; the NN [API](../../glossary.md#term-application-programming-interface-api) is implemented and accepted.
The specification describes a closed Sequential tree, actual attributes, derived input/output
constraints and explicit [parameter](../../glossary.md#term-parameter) [Tensor](../../glossary.md#tensor)-[identity](../../glossary.md#term-identity-meanings) groups. NN validates and reconstructs forward
semantics using [Model](../../glossary.md#term-model) only. It does not encode bytes, assign persistent versions or perform I/O.
The existing [callback](../../glossary.md#term-callback) ModelDefinition is unchanged. Unknown custom code and arbitrary Java records
remain explicit application reconstruction, never routine automatic restore.

The [owning NN contract](../../architecture/contracts/extensions-training.md#closed-declarative-models)
selects nested Sequential, [Linear](../../glossary.md#term-linear-meanings), affine LayerNorm, [Embedding](../../glossary.md#embedding), Conv1d/2d/3d and closed
RELU/[TANH](../../glossary.md#term-tanh)/[SIGMOID](../../glossary.md#term-sigmoid), across all existing floating parameter types. It preserves supplied Linear and
LayerNorm input promotion, automatic Linear/[convolution](../../glossary.md#term-channels-first-convolution-layers) exact-type guards, integral Embedding
input and compatible variable leading/[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) [Dimensions](../../glossary.md#dimension). No saved batch example defines a [signature](../../glossary.md#term-signature).
The [NN0026 API packet](../../planning/extensions/nn/tasks/0026-declarative-model-specification.md#exact-public-api)
freezes all constructors, typed step records, [Activation](../../glossary.md#term-activation) and the bound Linear.fromState factory.
Description reads current [bindings](../../glossary.md#term-binding) under caller [quiescence](../../glossary.md#term-quiescence); it never trusts cached [alias](../../glossary.md#aliasing-and-arena) groups
after legal parameter replacement or strict loading. Numeric nested paths have no new root prefix.

Reconstruction creates one [descriptor](../../glossary.md#term-descriptor)-only Tensor per validated alias group or [singleton](../../glossary.md#term-singleton), binds
NN-owned layers and returns a fully bound [evaluation](../../glossary.md#term-evaluation) skeleton for existing strict state load.
Existing TensorFactory.create and supplied-state constructors allocate no [weight](../../glossary.md#term-weight) storage;
convolution reservation-load tests prove initialization-free binding. NN0026's focused tests now
extend that evidence across its [matrix](../../glossary.md#term-matrix-meanings); coordinator [validation](../../glossary.md#term-validation) has passed.
No random/throwaway weights or native [ownership](../../glossary.md#term-ownership-meanings) is needed.
Current [Checkpoint](../../glossary.md#term-checkpoint) [target](../../glossary.md#term-target)-load semantics can consume the skeleton's shared references unchanged.
[Inference](../../glossary.md#term-inference) aliases preserve exact Tensor [identity](../../glossary.md#term-reference-identity), not equal-byte sharing or physical overlaps;
Training duplicate-identity/overlap rejection remains unchanged.

Choosing a closed value tree avoids generic [node](../../glossary.md#node) IDs/references and [graph](../../glossary.md#graph)-validation machinery.
4096 occurrences/depth 64 bound cold tree processing, not [backend](../../glossary.md#backend) sizes or capability. Input-specific
composition checks use existing Model [metadata](../../glossary.md#term-metadata) operations, not a new [shape](../../glossary.md#shape) solver. Empty/activation-
only trees are semantically eligible without a generic zero-node [Engine](../../glossary.md#term-engine) execution promise.
Rejected alternatives for this NN [scope](../../glossary.md#term-scope) are rejecting all aliases (unnecessary given exact-reference
strict load), capturing arbitrary callbacks (no reconstruction meaning), and [eager](../../glossary.md#term-eager-evaluation) initialization
then replacement (unnecessary weight-sized allocations). The cost is a deliberately bounded tree
matrix and new typed layer configuration access, with focused semantic/allocation/API tests required.
No [dependency](../../glossary.md#term-dependency) rule changes; the existing exact [NN Model](../../glossary.md#model-module-buffer-and-scan-distinctions)-only architecture [guard](../../glossary.md#term-guard) remains sufficient.

<a id="checkpoint-decision-selected-for-review"></a>

## Accepted [Checkpoint](../../glossary.md#term-checkpoint) decision

Accepted CP [scope](../../glossary.md#term-scope) only — 2026-10-14, after final independent approval of all 13 packet docs (no
P1/P2; 276 links) and coordinator base/[dependency](../../glossary.md#term-dependency)/contract/frontier/conflict verification recorded
in the CP0005B brief. The user selected whole-[model](../../glossary.md#term-model) [persistence](../../glossary.md#term-persistence) first,
then recurrent reconstruction before simple Training [resume](../../glossary.md#schema-serialization-and-checkpoint). The coordinated
[whole-model contract](../../architecture/contracts/extensions-training.md#whole-model-definition-and-composition)
freezes the exact ModelCheckpoint overloads, failure types, encoded defaults and [canonical](../../glossary.md#term-canonical-form) fields.
Use [Module](../../glossary.md#term-module) input recognized only as Sequential or DeclarativeModel so an ordinary supported
Sequential can be saved without manual conversion; custom roots fail before [Engine](../../glossary.md#term-engine)/file work.
Short save/load need no caller [digest](../../glossary.md#term-digest). Advanced overloads use ArtifactLimits and one [immutable](../../glossary.md#term-immutable)
32-byte InputArtifactFingerprint, requiring expected presence and bytes on explicit load; short
load rejects stored input [identity](../../glossary.md#term-identity-meanings) rather than treating it as verified. No new public image wrapper.

The new kind is whole-model.v1: canonical SYNDEF01 version-one definition/[schema](../../glossary.md#term-schema) plus unchanged
model.v1 entries prefixed model/. The definition is a prefix closed eight-kind Sequence tree with
all [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) attributes, [Linear](../../glossary.md#term-linear-meanings) [guard](../../glossary.md#term-guard) policy, exact [ScalarValue](../../glossary.md#typed-scalar-value--scalarvalue) bits and canonical numeric-path identity
groups. No IDs/edge tables/[graph](../../glossary.md#graph)-cycle fields are needed: arbitrary edges and cycles are
unrepresentable in the closed value tree.
Configuration [SHA-256](../../glossary.md#term-sha-256) covers the exact definition; complete ordered [descriptors](../../glossary.md#term-descriptor) and [alias](../../glossary.md#aliasing-and-arena) raw bytes
must independently agree before private NN reconstruction and public [target](../../glossary.md#term-target)-load can return a model.
Use only public [capture](../../glossary.md#term-capture-meanings)/inspect/canonicalStateBytes/load for the values component.

Defaults are 256 MiB complete file/aggregate, 4 MiB [manifest](../../glossary.md#term-manifest), 16384 entries, 64-byte kind/key,
64 MiB single [payload](../../glossary.md#artifact-envelope-and-atomic-publication). A fixed 1 MiB definition and NN 4096-[occurrence](../../glossary.md#term-occurrence)/depth-64 ceiling also apply
to advanced limits. These are encoded bounds, not peak-[heap](../../glossary.md#term-heap) budgets. Dry-size/check definition and
[residual](../../glossary.md#scan-residual-port-and-history-demand) [metadata](../../glossary.md#term-metadata), encode at most 1 MiB, [hash](../../glossary.md#term-hash), then public capture with exact projected limits
preflights actual component/outer bounds before Engine or bulk values. Recheck before prefixed
copies; no duplicate private parser or new validator. Description and values capture may each read
state under caller [quiescence](../../glossary.md#term-quiescence), not a combined [snapshot](../../glossary.md#term-snapshot) [transaction](../../glossary.md#term-transaction). File [publication](../../glossary.md#term-publication) stays intact.

This choice rejects DeclarativeModel-only save (unnecessary user reconstruction), raw untyped
fingerprints (ambiguous advanced identity), a public [validation](../../glossary.md#term-validation) credential/zero-copy image (weaker
publication), and fresh generic persistence/validator machinery (duplicates [owner](../../glossary.md#term-owner) boundaries).
Its cost is bounded defensive copies and independent repeated component validation, not a
large-model heap guarantee. Load returns heap-backed eval state without Engine/[autograd](../../glossary.md#autograd)/[compile](../../glossary.md#compile),
native [ownership](../../glossary.md#term-ownership-meanings) or throwaway initialization; identity trees have no generic zero-[node](../../glossary.md#node) promise.
Unknown recurrent IDs [fail closed](../../glossary.md#admission-preflight-and-fail-closed-validation) now. NN0027 owns only Model-only semantic description/reconstruction
using existing static Rnn/Gru/Lstm capabilities; CP0005C owns explicit versioned recurrent definition
wire/whole-save/load adoption. Initial-state/output selection, format version and Training [gradient](../../glossary.md#term-gradient)
[admission](../../glossary.md#term-admission) are future decisions before detailed briefs. Neither waits on optimized [scan](../../glossary.md#term-cumulative-scan)/[backend](../../glossary.md#backend)
[BPTT](../../glossary.md#term-backpropagation-through-time-bptt)/Data, and CP0005B v1 claims no recurrent files. Ordinary [unrolled](../../glossary.md#backpropagation-through-time-and-recurrent-state) differentiation depends
on its constituent operations; only fixed RecurrentScan BPTT is explicitly blocked here.

## Proposed Training and adapter decision

Training owns `TrainingRecipe` and `TrainingRun` in its existing public package.
The recipe describes a supported [scalar](../../glossary.md#term-scalar) [loss](../../glossary.md#loss), options, named input/[target](../../glossary.md#term-target) static [dtype](../../glossary.md#data-type--datatype)/[Shapes](../../glossary.md#shape) and
[binding](../../glossary.md#term-binding) meaning, modes/effective contexts and [gradient](../../glossary.md#term-gradient) mode. TrainingState alone encodes saved
[optimizer](../../glossary.md#term-optimizer)/groups; fresh [run](../../glossary.md#term-run) creation receives optimizer/groups separately. Recipe semantics,
[objective](../../glossary.md#term-objective) construction and reusable typed [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) binding are Training responsibilities.
A run exclusively owns its recreated [model](../../glossary.md#term-model), shared native [parameter](../../glossary.md#term-parameter) [Arena](../../glossary.md#term-arena), placeholders and
[TrainingSession](../../glossary.md#training-session--trainingsession), [borrowing](../../glossary.md#term-borrowing) the supplied [Engine](../../glossary.md#term-engine). The optional [persistence](../../glossary.md#term-persistence) adapter does not become
a runner. Its only direct project edges are public Training + [Checkpoint](../../glossary.md#term-checkpoint); [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Model/Engine are transitive.

### Proposed user experience

Conceptual [API](../../glossary.md#term-application-programming-interface-api), not runnable today. Assume a stored supported [FLOAT32](../../glossary.md#term-float32) dense [model](../../glossary.md#term-model) accepting
`[8,3]` and producing `[8,2]`, and a training [artifact](../../glossary.md#term-artifact) with a saved [scalar](../../glossary.md#term-scalar) [mean](../../glossary.md#term-mean)-square [loss](../../glossary.md#loss)
with SUM/MEAN reduction and FLOAT32 [target](../../glossary.md#term-target) `[8,2]`. The [Engine](../../glossary.md#term-engine) and caller [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) storage already exist.

```java
var model = ModelCheckpoint.load(modelPath);
var prediction = engine.compute(model.forward(input));

var inferenceModel = TrainingCheckpoint.loadModel(trainingPath);
var predictionFromTrainingFile = engine.compute(inferenceModel.forward(input));

try (var run = TrainingCheckpoint.resume(trainingPath, engine)) {
    var step = run.trainBatch(input, target);
    TrainingCheckpoint.save(trainingPath, run);
}
```

Loads return independently [heap](../../glossary.md#term-heap)-backed models in [inference](../../glossary.md#term-inference) mode without an [Arena](../../glossary.md#term-arena) [lifetime](../../glossary.md#term-lifetime).
[Parameter](../../glossary.md#term-parameter) [Tensors](../../glossary.md#tensor) remain [gradient](../../glossary.md#term-gradient)-eligible; [evaluation](../../glossary.md#term-evaluation) mode does not disable gradients.
Ordinary Engine.compute is forward-only, performs no [autograd](../../glossary.md#autograd) and returns [detached](../../glossary.md#detached-model-results) [HostTensorValue](../../glossary.md#host-tensor-value--hosttensorvalue)
data readable after Engine closure, not a closeable [run](../../glossary.md#term-run) [owner](../../glossary.md#term-owner). [Resume](../../glossary.md#schema-serialization-and-checkpoint) builds one [objective](../../glossary.md#term-objective)/session;
a changed batch [Shape](../../glossary.md#shape) fails, not an implicit per-batch recompile. The example restores stored state,
not a data cursor or an advancing [random stream](../../glossary.md#graph-rng-state--graphrngstate). Save also has a supported model-only overload.
Advanced explicit envelope, target, [fingerprint](../../glossary.md#term-fingerprint), limits and session APIs remain available.

### Initial support and exact [scope](../../glossary.md#term-scope) decision

The [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [matrix](../../glossary.md#term-matrix-meanings), [aliases](../../glossary.md#aliasing-and-arena) and [descriptor](../../glossary.md#term-descriptor) skeleton are implemented and validated. Branches/contextual adapters,
BatchNorm/[Dropout](../../glossary.md#dropout--inverted-dropout), recurrent states and generic I/O remain excluded. [Backend](../../glossary.md#backend) execution is separately
admitted; [Conv3d](../../glossary.md#conv3d--ncdhw) [gradients](../../glossary.md#term-gradient) and fixed RecurrentScan [BPTT](../../glossary.md#term-backpropagation-through-time-bptt) remain [fail-closed](../../glossary.md#term-fail-closed). Ordinary static [unrolling](../../glossary.md#term-unrolling)
depends on normal constituent-[operation](../../glossary.md#operation) adjoints, not a blanket recurrent-[autograd](../../glossary.md#autograd) ban. The closed initial tree has no
reused execution [nodes](../../glossary.md#node). A future execution [DAG](../../glossary.md#term-directed-acyclic-graph-dag) is a separate choice, not [inference](../../glossary.md#term-inference) from [ownership](../../glossary.md#term-ownership-meanings).
`model.v1` alone stores per-path values, not source [identity](../../glossary.md#term-identity-meanings); the selected whole-[model](../../glossary.md#term-model) contract carries
NN alias meaning without altering the current values-only format. Recurrent policy remains future.

### [Artifact](../../glossary.md#term-artifact) composition and proposed Training [validation](../../glossary.md#term-validation)

[Checkpoint](../../glossary.md#term-checkpoint)'s selected review contract owns [canonical](../../glossary.md#term-canonical-form) bounded encoding and persistent versions. Use distinct whole-[model](../../glossary.md#term-model) kind
`whole-model.v1`: `definition/schema` plus prefixed `model.v1` entries under `model/`.
Definition bytes contain real configuration/wiring/sharing; Checkpoint computes configuration
[SHA-256](../../glossary.md#term-sha-256) from canonical encoding automatically. Semantic compatibility still checks supported
version, type, attributes and [signatures](../../glossary.md#term-signature). SHA is neither [authentication](../../glossary.md#term-authentication) nor approval evidence.
Use public ModelCheckpoint [capture](../../glossary.md#term-capture-meanings)/inspect/canonicalStateBytes/load for the values component,
never private model-[codec](../../glossary.md#term-codec) parsing. Public whole-model envelope operations support downstream reuse
without temporary files or another trusted-image [certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions), using existing ArtifactEnvelope
and distinctly named ModelCheckpoint whole-model capture/load operations, not a new image wrapper.
High-level load rejects old values-only files with a missing-definition diagnostic directing callers
to the existing [target](../../glossary.md#term-target)-load [API](../../glossary.md#term-application-programming-interface-api); it does not guess [topology](../../glossary.md#term-topology) from [weight](../../glossary.md#term-weight) [Shapes](../../glossary.md#shape).

Use distinct `training-model.v1` for whole-model component + Training-owned recipe +
TrainingState. The unpublished 0037 `training-checkpoint.v1` proposal is replaced, not claimed
same-file-v1 compatible. ArtifactFile v1 and `model.v1` bytes/APIs remain untouched.
Retain 0037's explicit [optimizer](../../glossary.md#term-optimizer) IDs/raw coefficients, ordered groups, logical/master/slots/pending
payloads and counters as the state-codec foundation; Training public constructors and restore
remain the validators. Recipe wire IDs/limits and Training defaults remain decisions before [Ready](../../glossary.md#term-ready);
CP0005B defaults are frozen separately, not acceptance of adapter outer limits.

Before publishing any model/[run](../../glossary.md#term-run), compare definition-predicted complete [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) state [descriptors](../../glossary.md#term-descriptor) and
[alias](../../glossary.md#aliasing-and-arena) topology to independently validated raw values; training additionally compares the COMPLETE
[parameter](../../glossary.md#term-parameter) path/type/Shape/logical-byte set with TrainingState. Decode bounded known recipe data,
load the private NN skeleton, copy native parameters, apply modes/contexts, build the [scalar](../../glossary.md#term-scalar)
[FLOAT32](../../glossary.md#term-float32)/[FLOAT64](../../glossary.md#term-float64) [objective](../../glossary.md#term-objective) (explicit FLOAT32 cast for low [predictions](../../glossary.md#term-prediction) when required), open the
ordinary session, compare canonical parameters and call restore. No new pure Training validator.

TrainingCheckpoint.save captures one coherent [quiescent](../../glossary.md#term-quiescent) run through a narrow Training-owned
checkpoint exclusion boundary spanning model capture and session.state; that boundary is not a
general [transaction](../../glossary.md#term-transaction)/participant API. Freeze this [owner](../../glossary.md#term-owner) bridge before implementation.
Return a run only after complete owner validation. Failure discards the private candidate and
closes session before parameter/placeholder [Arenas](../../glossary.md#term-arena); attempt all cleanup and retain suppressed
failures. Never expose a native-backed model that can silently outlive the run. Prefer only [detached](../../glossary.md#detached-model-results)
[inference](../../glossary.md#term-inference) snapshots; any advanced live access must be explicitly run-[lifetime](../../glossary.md#term-lifetime)-bound.

Safe default ArtifactLimits apply to outer and projected components, with configurable advanced
overloads. [Preflight](../../glossary.md#term-preflight) nested counts/lengths and aggregate/file sizes before allocations/copies.
Input-artifact fingerprints retain owner verification: the short overload requires absent input
[digest](../../glossary.md#term-digest); present digests require a typed explicit expected-[identity](../../glossary.md#term-identity-meanings) overload and fail by default.
No [tokenizer](../../glossary.md#frozen-whitespace-tokenizer) parsing, automatic Text loading, generic LoadOptions or trust in a stored digest.

NN input/output constraints preserve compatible variable leading/[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) [Dimensions](../../glossary.md#dimension) as selected
in the owning NN contract; [Embedding](../../glossary.md#embedding) input is integral and output floating, not [homogeneous](../../glossary.md#term-homogeneous) [dtype](../../glossary.md#data-type--datatype).
Training signatures are independently static/dense/offset-zero for one prepared session.
TrainingRun binds each fresh batch's [borrowed](../../glossary.md#ownership-borrowing-and-lease) [host storage](../../glossary.md#host-storage) to stable placeholder [Tensor](../../glossary.md#tensor) identities
with replaceHostStorage, validates exact [layout](../../glossary.md#layout)/offset/[view](../../glossary.md#term-view-meanings) and capacity (not just type/Shape),
and clears associations in finally with clearHostStorage. Resolve roles once from session compiled
[metadata](../../glossary.md#term-metadata)/placeholder identity; no fresh IDs passed directly, positional mapping authority or
hot-path string [lookup](../../glossary.md#term-lookup). Run [admission](../../glossary.md#term-admission) excludes batch/save/close; borrowed inputs are never closed.

## Consequences

- Automatic restore is possible only for the approved declarative matrix; custom code retains the
  explicit workflow. A new format is not retroactive whole-model reconstruction of old value files.<span data-glossary-navigation="true"> ([matrix glossary](../../glossary.md#term-matrix-meanings); [model glossary](../../glossary.md#term-model))</span>
- Resource close order, checkpoint exclusion and descriptor skeletons require focused proof.
  Encoded limits do not promise peak-heap budgets or large-model capacity.<span data-glossary-navigation="true"> ([checkpoint glossary](../../glossary.md#term-checkpoint); [descriptor glossary](../../glossary.md#term-descriptor); [heap glossary](../../glossary.md#term-heap))</span>
- No automatic cursor/epoch/RNG/scheduler/scaler restoration, changing per-step dropout stream,
  BatchNorm step-aligned training-buffer recovery or cross-device exact continuation is claimed.<span data-glossary-navigation="true"> ([epoch glossary](../../glossary.md#term-epoch-meanings); [RNG glossary](../../glossary.md#graph-rng-state--graphrngstate); [dropout glossary](../../glossary.md#dropout--inverted-dropout); [buffer glossary](../../glossary.md#term-buffer); [device glossary](../../glossary.md#term-device))</span>
- Reuse file checksums and atomic-live publication; no authentication, universal power-loss
  guarantee, saved backend/device/compiled artifacts or automatic backend discovery.<span data-glossary-navigation="true"> ([atomic glossary](../../glossary.md#resource-transaction-and-rollback); [publication glossary](../../glossary.md#term-publication); [authentication glossary](../../glossary.md#term-authentication); [loss glossary](../../glossary.md#loss); [backend glossary](../../glossary.md#backend); [artifacts glossary](../../glossary.md#term-artifact))</span>
- NN ownership and implementation are accepted; CP-only whole-model contract/explanations are
  accepted after final independent approval. No dependency/build
  edge changes or new dependency test is needed. Before any downstream Ready decision, separately
  coordinate its owning contract, explanations, ADR and applicable architecture/integration tests.
  CP0005B is Complete on accepted packet base <span data-glossary-navigation="true"> ([NN glossary](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [ownership glossary](../../glossary.md#term-ownership-meanings); [dependency glossary](../../glossary.md#term-dependency); [Ready glossary](../../glossary.md#term-ready); [coordinate glossary](../../glossary.md#axis-rank-scalar-vector-and-matrix); [ADR glossary](../../glossary.md#term-architecture-decision-record-adr))</span>`7d65f660051becc9bc6adc6f08af155b08096b59`,
  with independent reviews and recorded module/Javadoc/integration evidence in its brief. Neither this
  acceptance nor NN accepts future recurrent/Training/adapter contracts or opens their build gates.<span data-glossary-navigation="true"> ([module glossary](../../glossary.md#term-module); [Javadoc glossary](../../glossary.md#term-javadoc))</span>
- Selected serial order, with each capability in clean implementation and independent review:
  NN0026 Complete → Checkpoint0005B → NN0027 semantics → Checkpoint0005C persistence
  → Training0006 → optional Checkpoint0006. NN0027/CP0005C have no detailed speculative briefs;
  the split follows NN meaning versus Checkpoint bytes ownership, not mechanical API/test staging.
  Each owner runs focused/module validation once; integration owner runs one source-matched
  cross-module repository test/Javadoc checkpoint. No performance campaign.<span data-glossary-navigation="true"> ([persistence glossary](../../glossary.md#term-persistence); [API glossary](../../glossary.md#term-application-programming-interface-api); [owner glossary](../../glossary.md#term-owner); [validation glossary](../../glossary.md#term-validation))</span>

## Related documentation

- [NN 0026](../../planning/extensions/nn/tasks/0026-declarative-model-specification.md)<span data-glossary-navigation="true"> ([NN glossary](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- [Checkpoint 0005B](../../planning/extensions/checkpoint/tasks/0005b-whole-model-save-load.md)<span data-glossary-navigation="true"> ([Checkpoint glossary](../../glossary.md#term-checkpoint))</span>
- [Training 0006](../../planning/extensions/training/tasks/0006-supported-recipe-and-owned-run.md)
- [Revised Checkpoint 0006](../../planning/extensions/checkpoint/tasks/0006-training-checkpoint-adapter.md)
- [Root authority](../../../ARCHITECTURE.md#authority-incorporation-and-precedence)
- [Dependency rules](../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../glossary.md#term-dependency))</span>
- [Owning extensions contract](../../architecture/contracts/extensions-training.md)
- [Lifecycle explanation](../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../glossary.md#lifecycle))</span>
- [Published model API](../../api/model-checkpoint-api.md)<span data-glossary-navigation="true"> ([model glossary](../../glossary.md#term-model); [API glossary](../../glossary.md#term-application-programming-interface-api))</span>
