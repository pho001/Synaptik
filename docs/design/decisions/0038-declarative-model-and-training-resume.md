# ADR 0038: Stored declarative models with inference or owned training resume

## Status

Accepted NN decision — 2026-10-14; final independent review approved and coordinator verified NN0026
Ready. Accepted CP-only whole-model decision — 2026-10-14; final independent contract APPROVE
and coordinator verification authorize CP0005B Ready, not an implementation result. Future recurrent/Training/adapter decisions
remain Proposed, not accepted contracts or implementation authorization.
Replaces the Proposed direction in [ADR 0037](0037-stored-training-checkpoint-adapter.md),
which is withdrawn as the primary product, not as evidence of useful low-level foundations.
The NN scope is accepted and coordinated with the root and owning NN contract. NN0026
is Complete with implementation independently reviewed and coordinator-validated.
CP0005B is Ready, the sole authorized Checkpoint frontier. NN0027 semantic
recurrent reconstruction and CP0005C persistence adoption are Draft rows only; Training0006/adapter0006 remain
Draft with Proposed decisions. No TrainingRun or adapter acceptance.

## Context

The user needs a usable model from a file, then a choice between inference and resumed training.
Published ModelCheckpoint persists values against a caller-recreated topology; proposed 0037 added
TrainingState but still required manual model/objective construction, digests and native Arena work.
That cannot deliver the selected primary experience.

NN's `Model<I,O>` can contain arbitrary Java forward code and record-shaped inputs/outputs.
Existing `ModelDefinition` is a callback, not serializable model data. `Sequential` orders unary
Tensor layers; ModuleFactory creates fresh layers but retains no reconstructable specification.
TrainingSession requires native parameters, compiles/prepares once, and closes its inward session
and cotangent Arena, not the borrowed model, Engine or parameter Arena.

## Decision drivers

- Load supported models without lambdas, restore callbacks, manual construction or digest plumbing.
- Keep semantic reconstruction in NN, recipe/run lifecycle in Training, codecs/files downstream.
- Preserve existing explicit APIs, ArtifactFile v1 and published `model.v1`.
- Reconstruct actual attributes, wiring and sharing, not merely compare a configuration hash.
- Make resource ownership and fail-closed capability limits observable and testable.

## Options considered

1. Retain 0037 as the primary product. Useful explicit foundation, but fails the automatic-model UX.
2. Serialize Java callbacks/objects, instantiate class names reflectively or use a plugin registry.
   Reject: unavailable application code, unsafe/nonportable reconstruction and wrong owner boundaries.
3. Put all model/loss/session logic in Checkpoint, or add a general ModelExecutor/transaction framework.
   Reject: bypasses NN/Training and invents an execution/persistence framework.
4. Add a closed NN declarative specification, whole-model persistence and an optional adapter
   composing a Training-owned recipe and run. Propose this bounded option.

<a id="nn-decision-selected-for-review"></a>

## Accepted NN decision

NN owns a new immutable `ModelSpecification` and `DeclarativeModel extends Model<Tensor, Tensor>`
in `io.github.pho001.synaptik.nn.module`; the NN API is implemented and accepted.
The specification describes a closed Sequential tree, actual attributes, derived input/output
constraints and explicit parameter Tensor-identity groups. NN validates and reconstructs forward
semantics using Model only. It does not encode bytes, assign persistent versions or perform I/O.
The existing callback ModelDefinition is unchanged. Unknown custom code and arbitrary Java records
remain explicit application reconstruction, never routine automatic restore.

The [owning NN contract](../../architecture/contracts/extensions-training.md#closed-declarative-models)
selects nested Sequential, Linear, affine LayerNorm, Embedding, Conv1d/2d/3d and closed
RELU/TANH/SIGMOID, across all existing floating parameter types. It preserves supplied Linear and
LayerNorm input promotion, automatic Linear/convolution exact-type guards, integral Embedding
input and compatible variable leading/batch Dimensions. No saved batch example defines a signature.
The [NN0026 API packet](../../planning/extensions/nn/tasks/0026-declarative-model-specification.md#exact-public-api)
freezes all constructors, typed step records, Activation and the bound Linear.fromState factory.
Description reads current bindings under caller quiescence; it never trusts cached alias groups
after legal parameter replacement or strict loading. Numeric nested paths have no new root prefix.

Reconstruction creates one descriptor-only Tensor per validated alias group or singleton, binds
NN-owned layers and returns a fully bound evaluation skeleton for existing strict state load.
Existing TensorFactory.create and supplied-state constructors allocate no weight storage;
convolution reservation-load tests prove initialization-free binding. NN0026's focused tests now
extend that evidence across its matrix; coordinator validation has passed.
No random/throwaway weights or native ownership is needed.
Current Checkpoint target-load semantics can consume the skeleton's shared references unchanged.
Inference aliases preserve exact Tensor identity, not equal-byte sharing or physical overlaps;
Training duplicate-identity/overlap rejection remains unchanged.

Choosing a closed value tree avoids generic node IDs/references and graph-validation machinery.
4096 occurrences/depth 64 bound cold tree processing, not backend sizes or capability. Input-specific
composition checks use existing Model metadata operations, not a new shape solver. Empty/activation-
only trees are semantically eligible without a generic zero-node Engine execution promise.
Rejected alternatives for this NN scope are rejecting all aliases (unnecessary given exact-reference
strict load), capturing arbitrary callbacks (no reconstruction meaning), and eager initialization
then replacement (unnecessary weight-sized allocations). The cost is a deliberately bounded tree
matrix and new typed layer configuration access, with focused semantic/allocation/API tests required.
No dependency rule changes; the existing exact NN Model-only architecture guard remains sufficient.

<a id="checkpoint-decision-selected-for-review"></a>

## Accepted Checkpoint decision

Accepted CP scope only — 2026-10-14, after final independent approval of all 13 packet docs (no
P1/P2; 276 links) and coordinator base/dependency/contract/frontier/conflict verification recorded
in the CP0005B brief. The user selected whole-model persistence first,
then recurrent reconstruction before simple Training resume. The coordinated
[whole-model contract](../../architecture/contracts/extensions-training.md#whole-model-definition-and-composition)
freezes the exact ModelCheckpoint overloads, failure types, encoded defaults and canonical fields.
Use Module input recognized only as Sequential or DeclarativeModel so an ordinary supported
Sequential can be saved without manual conversion; custom roots fail before Engine/file work.
Short save/load need no caller digest. Advanced overloads use ArtifactLimits and one immutable
32-byte InputArtifactFingerprint, requiring expected presence and bytes on explicit load; short
load rejects stored input identity rather than treating it as verified. No new public image wrapper.

The new kind is whole-model.v1: canonical SYNDEF01 version-one definition/schema plus unchanged
model.v1 entries prefixed model/. The definition is a prefix closed eight-kind Sequence tree with
all NN attributes, Linear guard policy, exact ScalarValue bits and canonical numeric-path identity
groups. No IDs/edge tables/graph-cycle fields are needed: arbitrary edges and cycles are
unrepresentable in the closed value tree.
Configuration SHA-256 covers the exact definition; complete ordered descriptors and alias raw bytes
must independently agree before private NN reconstruction and public target-load can return a model.
Use only public capture/inspect/canonicalStateBytes/load for the values component.

Defaults are 256 MiB complete file/aggregate, 4 MiB manifest, 16384 entries, 64-byte kind/key,
64 MiB single payload. A fixed 1 MiB definition and NN 4096-occurrence/depth-64 ceiling also apply
to advanced limits. These are encoded bounds, not peak-heap budgets. Dry-size/check definition and
residual metadata, encode at most 1 MiB, hash, then public capture with exact projected limits
preflights actual component/outer bounds before Engine or bulk values. Recheck before prefixed
copies; no duplicate private parser or new validator. Description and values capture may each read
state under caller quiescence, not a combined snapshot transaction. File publication stays intact.

This choice rejects DeclarativeModel-only save (unnecessary user reconstruction), raw untyped
fingerprints (ambiguous advanced identity), a public validation credential/zero-copy image (weaker
publication), and fresh generic persistence/validator machinery (duplicates owner boundaries).
Its cost is bounded defensive copies and independent repeated component validation, not a
large-model heap guarantee. Load returns heap-backed eval state without Engine/autograd/compile,
native ownership or throwaway initialization; identity trees have no generic zero-node promise.
Unknown recurrent IDs fail closed now. NN0027 owns only Model-only semantic description/reconstruction
using existing static Rnn/Gru/Lstm capabilities; CP0005C owns explicit versioned recurrent definition
wire/whole-save/load adoption. Initial-state/output selection, format version and Training gradient
admission are future decisions before detailed briefs. Neither waits on optimized scan/backend
BPTT/Data, and CP0005B v1 claims no recurrent files. Ordinary unrolled differentiation depends
on its constituent operations; only fixed RecurrentScan BPTT is explicitly blocked here.

## Proposed Training and adapter decision

Training owns `TrainingRecipe` and `TrainingRun` in its existing public package.
The recipe describes a supported scalar loss, options, named input/target static dtype/Shapes and
binding meaning, modes/effective contexts and gradient mode. TrainingState alone encodes saved
optimizer/groups; fresh run creation receives optimizer/groups separately. Recipe semantics,
objective construction and reusable typed batch binding are Training responsibilities.
A run exclusively owns its recreated model, shared native parameter Arena, placeholders and
TrainingSession, borrowing the supplied Engine. The optional persistence adapter does not become
a runner. Its only direct project edges are public Training + Checkpoint; NN/Model/Engine are transitive.

### Proposed user experience

Conceptual API, not runnable today. Assume a stored supported FLOAT32 dense model accepting
`[8,3]` and producing `[8,2]`, and a training artifact with a saved scalar mean-square loss
with SUM/MEAN reduction and FLOAT32 target `[8,2]`. The Engine and caller batch storage already exist.

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

Loads return independently heap-backed models in inference mode without an Arena lifetime.
Parameter Tensors remain gradient-eligible; evaluation mode does not disable gradients.
Ordinary Engine.compute is forward-only, performs no autograd and returns detached HostTensorValue
data readable after Engine closure, not a closeable run owner. Resume builds one objective/session;
a changed batch Shape fails, not an implicit per-batch recompile. The example restores stored state,
not a data cursor or an advancing random stream. Save also has a supported model-only overload.
Advanced explicit envelope, target, fingerprint, limits and session APIs remain available.

### Initial support and exact scope decision

The NN matrix, aliases and descriptor skeleton are implemented and validated. Branches/contextual adapters,
BatchNorm/Dropout, recurrent states and generic I/O remain excluded. Backend execution is separately
admitted; Conv3d gradients and fixed RecurrentScan BPTT remain fail-closed. Ordinary static unrolling
depends on normal constituent-operation adjoints, not a blanket recurrent-autograd ban. The closed initial tree has no
reused execution nodes. A future execution DAG is a separate choice, not inference from ownership.
`model.v1` alone stores per-path values, not source identity; the selected whole-model contract carries
NN alias meaning without altering the current values-only format. Recurrent policy remains future.

### Artifact composition and proposed Training validation

Checkpoint's selected review contract owns canonical bounded encoding and persistent versions. Use distinct whole-model kind
`whole-model.v1`: `definition/schema` plus prefixed `model.v1` entries under `model/`.
Definition bytes contain real configuration/wiring/sharing; Checkpoint computes configuration
SHA-256 from canonical encoding automatically. Semantic compatibility still checks supported
version, type, attributes and signatures. SHA is neither authentication nor approval evidence.
Use public ModelCheckpoint capture/inspect/canonicalStateBytes/load for the values component,
never private model-codec parsing. Public whole-model envelope operations support downstream reuse
without temporary files or another trusted-image certificate, using existing ArtifactEnvelope
and distinctly named ModelCheckpoint whole-model capture/load operations, not a new image wrapper.
High-level load rejects old values-only files with a missing-definition diagnostic directing callers
to the existing target-load API; it does not guess topology from weight Shapes.

Use distinct `training-model.v1` for whole-model component + Training-owned recipe +
TrainingState. The unpublished 0037 `training-checkpoint.v1` proposal is replaced, not claimed
same-file-v1 compatible. ArtifactFile v1 and `model.v1` bytes/APIs remain untouched.
Retain 0037's explicit optimizer IDs/raw coefficients, ordered groups, logical/master/slots/pending
payloads and counters as the state-codec foundation; Training public constructors and restore
remain the validators. Recipe wire IDs/limits and Training defaults remain decisions before Ready;
CP0005B defaults are frozen separately, not acceptance of adapter outer limits.

Before publishing any model/run, compare definition-predicted complete NN state descriptors and
alias topology to independently validated raw values; training additionally compares the COMPLETE
parameter path/type/Shape/logical-byte set with TrainingState. Decode bounded known recipe data,
load the private NN skeleton, copy native parameters, apply modes/contexts, build the scalar
FLOAT32/FLOAT64 objective (explicit FLOAT32 cast for low predictions when required), open the
ordinary session, compare canonical parameters and call restore. No new pure Training validator.

TrainingCheckpoint.save captures one coherent quiescent run through a narrow Training-owned
checkpoint exclusion boundary spanning model capture and session.state; that boundary is not a
general transaction/participant API. Freeze this owner bridge before implementation.
Return a run only after complete owner validation. Failure discards the private candidate and
closes session before parameter/placeholder Arenas; attempt all cleanup and retain suppressed
failures. Never expose a native-backed model that can silently outlive the run. Prefer only detached
inference snapshots; any advanced live access must be explicitly run-lifetime-bound.

Safe default ArtifactLimits apply to outer and projected components, with configurable advanced
overloads. Preflight nested counts/lengths and aggregate/file sizes before allocations/copies.
Input-artifact fingerprints retain owner verification: the short overload requires absent input
digest; present digests require a typed explicit expected-identity overload and fail by default.
No tokenizer parsing, automatic Text loading, generic LoadOptions or trust in a stored digest.

NN input/output constraints preserve compatible variable leading/batch Dimensions as selected
in the owning NN contract; Embedding input is integral and output floating, not homogeneous dtype.
Training signatures are independently static/dense/offset-zero for one prepared session.
TrainingRun binds each fresh batch's borrowed host storage to stable placeholder Tensor identities
with replaceHostStorage, validates exact layout/offset/view and capacity (not just type/Shape),
and clears associations in finally with clearHostStorage. Resolve roles once from session compiled
metadata/placeholder identity; no fresh IDs passed directly, positional mapping authority or
hot-path string lookup. Run admission excludes batch/save/close; borrowed inputs are never closed.

## Consequences

- Automatic restore is possible only for the approved declarative matrix; custom code retains the
  explicit workflow. A new format is not retroactive whole-model reconstruction of old value files.
- Resource close order, checkpoint exclusion and descriptor skeletons require focused proof.
  Encoded limits do not promise peak-heap budgets or large-model capacity.
- No automatic cursor/epoch/RNG/scheduler/scaler restoration, changing per-step dropout stream,
  BatchNorm step-aligned training-buffer recovery or cross-device exact continuation is claimed.
- Reuse file checksums and atomic-live publication; no authentication, universal power-loss
  guarantee, saved backend/device/compiled artifacts or automatic backend discovery.
- NN ownership and implementation are accepted; CP-only whole-model contract/explanations are
  accepted after final independent approval. No dependency/build
  edge changes or new dependency test is needed. Before any downstream Ready decision, separately
  coordinate its owning contract, explanations, ADR and applicable architecture/integration tests.
  CP0005B is Ready after coordinator verification, with source f892 plus accepted packet as its
  execution base. The launcher records the final contract commit after integration. Neither this
  acceptance nor NN accepts future recurrent/Training/adapter contracts or opens their build gates.
- Selected serial order, with each capability in clean implementation and independent review:
  NN0026 Complete → Checkpoint0005B → NN0027 semantics → Checkpoint0005C persistence
  → Training0006 → optional Checkpoint0006. NN0027/CP0005C have no detailed speculative briefs;
  the split follows NN meaning versus Checkpoint bytes ownership, not mechanical API/test staging.
  Each owner runs focused/module validation once; integration owner runs one source-matched
  cross-module repository test/Javadoc checkpoint. No performance campaign.

## Related documentation

- [NN 0026](../../planning/extensions/nn/tasks/0026-declarative-model-specification.md)
- [Checkpoint 0005B](../../planning/extensions/checkpoint/tasks/0005b-whole-model-save-load.md)
- [Training 0006](../../planning/extensions/training/tasks/0006-supported-recipe-and-owned-run.md)
- [Revised Checkpoint 0006](../../planning/extensions/checkpoint/tasks/0006-training-checkpoint-adapter.md)
- [Root authority](../../../ARCHITECTURE.md#authority-incorporation-and-precedence)
- [Dependency rules](../../../ARCHITECTURE.md#dependency-rules)
- [Owning extensions contract](../../architecture/contracts/extensions-training.md)
- [Lifecycle explanation](../../architecture/lifecycle.md)
- [Published model API](../../api/model-checkpoint-api.md)
