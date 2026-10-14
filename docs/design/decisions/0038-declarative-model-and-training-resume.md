# ADR 0038: Stored declarative models with inference or owned training resume

## Status

Proposed — 2026-10-14. Not an accepted contract or implementation authorization.
Replaces the Proposed direction in [ADR 0037](0037-stored-training-checkpoint-adapter.md),
which is withdrawn as the primary product, not as evidence of useful low-level foundations.
All new/revised tasks remain Draft pending the support matrix and coordinated contract cutover.

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

## Proposed decision

NN owns a new immutable `ModelSpecification` and `DeclarativeModel extends Model<Tensor, Tensor>`
in `io.github.pho001.synaptik.nn.module`; names are proposed, not existing.
The specification describes owned layer instances, actual attributes, Tensor input/output signatures,
ordered connections and explicit parameter alias groups. NN validates and reconstructs forward
semantics using Model only. It does not encode bytes, assign persistent versions or perform I/O.
The existing callback ModelDefinition is unchanged. Unknown custom code and arbitrary Java records
remain explicit application reconstruction, never routine automatic restore.

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

Propose Tensor-to-Tensor nested Sequential with Linear, LayerNorm, Embedding, Conv1d/2d/3d
and closed RELU/TANH/SIGMOID activation steps using existing Model semantics;
capture resolved dimensions, bias policy, epsilon type/bits, convolution groups/stride/padding/
dilation/layout and all other forward-affecting attributes. This is a meaningful multi-layer
subset, not a promise about all NN models. NN owns typed access to missing configuration facts:
do not reverse-engineer private fields or infer attrs from weight Shapes alone.
Declarative branches and explicit context-sensitive adapters are named successors, not detailed
new tasks. BatchNorm/Dropout, recurrent typed states and arbitrary generic I/O are initially excluded.
Approve or revise this exact matrix before Ready; backend execution remains separately admitted.
Conv3d inference may load, but training gradients and recurrent BPTT remain fail-closed.

Inference sharing must be described explicitly and either recreated faithfully by NN or rejected
explicitly; `model.v1` alone stores per-path values, not source identity. Training refuses repeated
parameter Tensor identities across recursive parameter paths or overlapping regions; never split
or duplicate shared weights to evade that rule. Multiple forward uses of a uniquely owned layer
are not that rejected alias case: child ownership is a tree, execution wiring may be a DAG.
Decide whether the initial NN matrix implements inference aliases or rejects all
nontrivial sharing. Descriptor-only bound skeleton construction is preferred, subject to focused
proof against layer/state loading contracts; avoid random initialization and throwaway weight arrays.

### Proposed artifact composition and validation

Checkpoint owns canonical bounded encoding and persistent versions. Use distinct whole-model kind
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
remain the validators. Exact recipe wire IDs/limits and defaults are decisions before Ready.

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

NN input/output constraints must preserve supported leading dimensions rather than freezing an
inference batch size from a save example; exact variable-batch policy is a Draft decision.
Embedding input is integral and output floating, not a homogeneous-dtype signature.
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
- Before Ready, cut over root ownership/dependencies and the owning extensions contract, affected
  explanations, ADR and architecture tests. No normative/build edits are part of this proposal.
- Four cohesive capabilities, each clean implementation and independent review:
  NN 0026; Checkpoint 0005B; Training 0006; optional Checkpoint 0006. NN is the shared prerequisite;
  the two owner capabilities join at the adapter. No speculative mechanical cascade.
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
