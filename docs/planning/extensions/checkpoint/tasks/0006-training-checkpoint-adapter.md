# Task 0006: Whole-[model](../../../../glossary.md#term-model) TrainingCheckpoint save, [inference](../../../../glossary.md#term-inference) load and [resume](../../../../glossary.md#schema-serialization-and-checkpoint)

## Status

Draft — revised primary product, not implementation authorization.
Replaces the [snapshot](../../../../glossary.md#term-snapshot)-only direction of withdrawn Proposed [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0037 with Proposed ADR 0038.
Support [matrix](../../../../glossary.md#term-review-matrix), [owner](../../../../glossary.md#term-owner) bridges, default limits and exact wire decisions must stabilize before [Ready](../../../../glossary.md#term-ready).

## Change class

Class C implementation — optional downstream [module](../../../../glossary.md#term-module), persistent kind and cross-[owner](../../../../glossary.md#term-owner) recovery.
This [planning](../../../../glossary.md#term-planning) revision changes no [normative](../../../../glossary.md#term-normative) contract, build, Java or test.

## Goal

Save one supported stored [model](../../../../glossary.md#term-model) + Training-owned recipe + TrainingState, then load a usable model
for [inference](../../../../glossary.md#term-inference) or [resume](../../../../glossary.md#schema-serialization-and-checkpoint) a Training-owned [run](../../../../glossary.md#term-run) automatically. No routine lambda, model reconstruction,
ArtifactEnvelope, config [digest](../../../../glossary.md#term-digest) or [Arena](../../../../glossary.md#term-arena) plumbing. Retain explicit advanced APIs.

## [Scope](../../../../glossary.md#term-scope) and non-goals

Optional `extensions/training-checkpoint` has exactly two direct public project edges:
`api(project(":extensions:training"))` and `api(project(":extensions:checkpoint"))`.
Use published [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/[Model](../../../../glossary.md#term-model)/[Engine](../../../../glossary.md#term-engine) transitively; no reverse or lower/[backend](../../../../glossary.md#backend) edge.
Training owns recipe semantics, [objective](../../../../glossary.md#term-objective) construction, [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) [binding](../../../../glossary.md#term-binding) and [run](../../../../glossary.md#term-run) [lifecycle](../../../../glossary.md#lifecycle); [Checkpoint](../../../../glossary.md#term-checkpoint)
owns codecs/file [publication](../../../../glossary.md#term-publication); NN owns model definition and forward reconstruction.

No arbitrary Java [serialization](../../../../glossary.md#term-serialization)/[callback](../../../../glossary.md#term-callback) [capture](../../../../glossary.md#term-capture-meanings)/reflection/class-name factory/[registry](../../../../glossary.md#term-registry), general
transactions/participants, ModelExecutor, automatic backend discovery, saved [device](../../../../glossary.md#term-device)/compiled
formats or new pure Training validator. Exclude caller cursor/[epoch](../../../../glossary.md#term-epoch-meanings)/external [RNG](../../../../glossary.md#graph-rng-state--graphrngstate)/scheduler/scaler.
Restore stored state, not complete automatic exact continuation, changing [dropout](../../../../glossary.md#dropout--inverted-dropout) stream,
BatchNorm training-[buffer](../../../../glossary.md#term-buffer) [alignment](../../../../glossary.md#term-alignment), [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt) or [Conv3d](../../../../glossary.md#conv3d--ncdhw) [gradients](../../../../glossary.md#term-gradient).

## Contracts

- [Root authority](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
  [scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>,
  [module ownership](../../../../../ARCHITECTURE.md#module-ownership-routing)<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>,
  [dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span> and
  [testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Owning scoped headings: [extensions/nn](../../../../architecture/contracts/extensions-training.md#extensionsnn),
  [extensions/training](../../../../architecture/contracts/extensions-training.md#extensionstraining),
  [extensions/checkpoint](../../../../architecture/contracts/extensions-training.md#extensionscheckpoint)<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>,
  [Optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)<span data-glossary-navigation="true"> ([Optimizer glossary](../../../../glossary.md#term-optimizer); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>.
- [Proposed ADR 0038](../../../../design/decisions/0038-declarative-model-and-training-resume.md).

The current contracts do not authorize the new product. Before [Ready](../../../../glossary.md#term-ready), [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) root + owning
extensions contract + affected architecture explanations + [ADR](../../../../glossary.md#term-architecture-decision-record-adr) + architecture tests as applicable.
Stop on missing/ambiguous contracts; this proposal does not override them.

## Dependencies and integration

- Depends on: NN0026 Complete; NN0027; Checkpoint0005B/0005C; Training0006; completed Checkpoint0002/0004/0004A/
  0005/0005A and Training0004. Remaining new predecessors are Draft and NOT satisfied.
- Conflicts with: adapter wire/[API](../../../../glossary.md#term-application-programming-interface-api)/build, owning contracts and shared [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training/[Checkpoint](../../../../glossary.md#term-checkpoint) docs.
- Parallel group: None; separate clean implementation contexts, serialized contract cutover.
- Common base revision: [planning](../../../../glossary.md#term-planning) base `52733fcfd32d278b651302e308487e0b6c6afa81`; refresh before [Ready](../../../../glossary.md#term-ready).
- Integration order: separate contract gates → NN0026 → CP0005B → NN0027 → CP0005C → Training0006
  → adapter 0006 → independent review → coordinator acceptance. No mechanical task split.
- Integration [validation](../../../../glossary.md#term-validation): once-only source-matched repository checkpoint specified below.
- Shared-document integration [owner](../../../../glossary.md#term-owner): NN/Training/Checkpoint coordinator.
- [Authorized frontier](../../../../glossary.md#term-authorized-frontier): None. No implementation or build inclusion is authorized by this brief.

## Proposed types and high-level [API](../../../../glossary.md#term-application-programming-interface-api)

In `io.github.pho001.synaptik.training.checkpoint`: public final `TrainingCheckpoint` and nested
[immutable](../../../../glossary.md#term-immutable) advanced [`Snapshot`](../../../../glossary.md#term-snapshot); private `TrainingArtifactCodec`, `TrainingArtifactComposition`
and `TrainingParameterMatch`. Reuse TrainingRecipe/TrainingRun and ArtifactEnvelope, no live image
wrapper or duplicate [validation](../../../../glossary.md#term-validation) credential. Final [signature](../../../../glossary.md#term-signature) freeze is a [Ready](../../../../glossary.md#term-ready) condition.

Conceptual, not existing APIs:

```text
TrainingCheckpoint.save(Path, TrainingRun): void throws IOException
TrainingCheckpoint.loadModel(Path): DeclarativeModel throws IOException
TrainingCheckpoint.resume(Path, Engine): TrainingRun throws IOException
TrainingCheckpoint.read(Path, ArtifactLimits): Snapshot throws IOException
```

Typed expected-input-[identity](../../../../glossary.md#term-identity-meanings) and ArtifactLimits overloads serve advanced callers without generic
LoadOptions. Short overloads require absent input [digest](../../../../glossary.md#term-digest); present [fingerprint](../../../../glossary.md#term-fingerprint) rejects until supplied
expected identity is verified. No automatic [tokenizer](../../../../glossary.md#frozen-whitespace-tokenizer) load (Text remains Draft).
Advanced [detached](../../../../glossary.md#detached-model-results) save/read/[model](../../../../glossary.md#term-model)-component/TrainingState access and explicit [target](../../../../glossary.md#term-target)/session
restore remain available; do not turn successful read into a restorable-session [certificate](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions).
Existing published ArtifactFile and ModelCheckpoint explicit APIs are unchanged.

## Stored [model](../../../../glossary.md#term-model) and recipe

Initial proposal is [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0026's [Tensor](../../../../glossary.md#tensor)-to-Tensor nested Sequential with [Linear](../../../../glossary.md#term-linear-meanings), LayerNorm, [Embedding](../../../../glossary.md#embedding),
Conv1d/2d/3d and closed RELU/[TANH](../../../../glossary.md#term-tanh)/[SIGMOID](../../../../glossary.md#term-sigmoid) steps; exact [matrix](../../../../glossary.md#term-matrix-meanings) remains a decision before [Ready](../../../../glossary.md#term-ready).
[Capture](../../../../glossary.md#term-capture-meanings) actual attrs and wiring, state [descriptors](../../../../glossary.md#term-descriptor) and explicit sharing; never infer ModelForward
captures. NN input constraints need not freeze [inference](../../../../glossary.md#term-inference) [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) size. Training recipe separately
records STATIC dense/offset-zero input/[target](../../../../glossary.md#term-target) [dtype](../../../../glossary.md#data-type--datatype)/[Shapes](../../../../glossary.md#shape), roles/[binding](../../../../glossary.md#term-binding) meaning, selected [loss](../../../../glossary.md#loss)/
options/reduction and modes/effective contexts/[gradient](../../../../glossary.md#term-gradient) mode. [Scalar objective](../../../../glossary.md#loss) is [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64);
low [predictions](../../../../glossary.md#term-prediction) cast explicitly as the approved loss requires. Saved [optimizer](../../../../glossary.md#term-optimizer) alone is insufficient.
TrainingState alone stores optimizer/default/groups; do not duplicate them in recipe encoding.
Unknown custom/generic record models use existing explicit reconstruction, not routine [resume](../../../../glossary.md#schema-serialization-and-checkpoint).

Inference sharing must reconstruct source [identity](../../../../glossary.md#term-identity-meanings) or reject it explicitly. Training rejects repeated
[parameter](../../../../glossary.md#term-parameter) Tensor identities across paths/overlapping regions; never silently duplicate shared
[weights](../../../../glossary.md#term-weight). A uniquely owned layer called multiple times is a different execution-[DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) case.
Current Training capability, [compiler](../../../../glossary.md#term-compiler)/[backend](../../../../glossary.md#backend) gradients and context limits remain [fail-closed](../../../../glossary.md#term-fail-closed).

## Proposed [canonical](../../../../glossary.md#term-canonical-form) wire and retained foundation

Use new kind `training-model.v1` over UNCHANGED ArtifactFile v1 [framing](../../../../glossary.md#term-framing)/[publication](../../../../glossary.md#term-publication). Prefix a
validated `whole-model.v1` component under `whole-model/`, plus `recipe/schema` and Training
[schema](../../../../glossary.md#term-schema)/slot entries. [Checkpoint](../../../../glossary.md#term-checkpoint) 0005B defines whole-[model](../../../../glossary.md#term-model) canonical definition bytes and config
SHA automatically. Reconstruct components by namespace only; delegate public whole-model operations
and model.v1 inspect/canonicalStateBytes/load, NEVER parse private model [metadata](../../../../glossary.md#term-metadata).
Published `model.v1` remains untouched. Unpublished 0037 `training-checkpoint.v1` is replaced,
not same-file-v1 compatible; no migration promise to an unimplemented wire.

Retain 0037's raw TrainingState [codec](../../../../glossary.md#term-codec) foundation: BIG_ENDIAN, strict [UTF-8](../../../../glossary.md#term-utf-8) paths without
[normalization](../../../../glossary.md#term-normalization), explicit [SGD](../../../../glossary.md#term-stochastic-gradient-descent-sgd)=1/[Adam](../../../../glossary.md#adaptive-optimizer--adam-adamw)=2/[AdamW](../../../../glossary.md#adaptive-optimizer--adam-adamw)=3 IDs; raw [binary64](../../../../glossary.md#floating-point-precision-and-rounding) [optimizer](../../../../glossary.md#term-optimizer) coefficients in accessor
order, ordered groups, execution/optimizer-step/pending counters, ordered static [parameter](../../../../glossary.md#term-parameter) schema.
[Dtype](../../../../glossary.md#data-type--datatype) IDs 1 [FLOAT64](../../../../glossary.md#term-float64), 2 [FLOAT32](../../../../glossary.md#term-float32), 3 [BFLOAT16](../../../../glossary.md#term-bfloat16), 7 [FLOAT16](../../../../glossary.md#term-float16). For ordinal N (eight-digit decimal), keys
`training/parameter/N/{logical,master,first,pending}` plus `second` iff adaptive slot is present.
Logical [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) uses its dtype width; masters/slots/pending use [F64](../../../../glossary.md#floating-point-precision-and-rounding) for F64, [F32](../../../../glossary.md#floating-point-precision-and-rounding) otherwise.
Preserve signed zeros, low logical/master distinctions and raw slot/pending bytes exactly.
Structural constructors permit zero-size values; session restore decides actual trainability.

Before [Ready](../../../../glossary.md#term-ready) freeze magic/version/record order for revised Training schema and recipe [loss](../../../../glossary.md#loss) IDs/
options/[signatures](../../../../glossary.md#term-signature)/context records; no generic caller-object encoding. Do not invent a duplicate
[certificate](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions) or pure validator. Public defensive TrainingState/ParameterState/[ParameterGroup](../../../../glossary.md#parameter-group--parametergroup)
constructors establish structure; algorithm slot validity/finiteness/group membership/order and
coefficient [narrowing](../../../../glossary.md#term-narrowing)/session compatibility remain Training.restore authority.

## [Validation](../../../../glossary.md#term-validation), [capture](../../../../glossary.md#term-capture-meanings) and recovery

Bound/[checksum](../../../../glossary.md#term-checksum) outer file first; [preflight](../../../../glossary.md#term-preflight) checked outer AND projected whole-[model](../../../../glossary.md#term-model)/model [framing](../../../../glossary.md#term-framing),
[schema](../../../../glossary.md#term-schema)/[payload](../../../../glossary.md#artifact-envelope-and-atomic-publication)/aggregate sizes and all ArtifactLimits before component or ParameterState clones.
Bound every nested count/[rank](../../../../glossary.md#term-rank)/path/string by remaining bytes/minimum record size and array ceilings.
Defaults must be conservative and configurable; [canonical](../../../../glossary.md#term-canonical-form) encoded limits are not peak-[heap](../../../../glossary.md#term-heap) budgets.
Reject missing/duplicate/extra keys, unknown versions/IDs, malformed/trailing data and [truncation](../../../../glossary.md#term-truncation-toward-zero).
Retain existing checked ArtifactFormatException/I/O taxonomy; [owner](../../../../glossary.md#term-owner)/[lifecycle](../../../../glossary.md#lifecycle) errors stay owner errors.

Before save/read returns any [detached](../../../../glossary.md#detached-model-results) [snapshot](../../../../glossary.md#term-snapshot), compare COMPLETE model [parameter](../../../../glossary.md#term-parameter) paths, PARAMETER
kind, [dtype](../../../../glossary.md#data-type--datatype)/[Shape](../../../../glossary.md#shape) and independent canonical logical bytes with TrainingState, including no extras.
[Buffers](../../../../glossary.md#term-buffer) stay [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/model-owned with all seven raw model types. Definition must predict the whole NN
[descriptor](../../../../glossary.md#term-descriptor)/[alias](../../../../glossary.md#aliasing-and-arena) set; config SHA is not [authentication](../../../../glossary.md#term-authentication), approval or a replacement for semantic checks.
High-level save holds TrainingRun's narrow [checkpoint](../../../../glossary.md#term-checkpoint) exclusion across model capture + session.state;
equality cannot prove [quiescence](../../../../glossary.md#term-quiescence). No old open-session [binding](../../../../glossary.md#term-binding) replacement or capture [race](../../../../glossary.md#term-race).

Recovery order is fixed:

1. Read/bound/validate all components, definition/recipe meaning, fingerprints and raw-value coherence.
2. NN reconstructs a private descriptor-bound model; public whole-model load installs heap state
   BEFORE any session opens, preserving sharing or rejecting unsupported aliases.
3. TrainingRun owns it exclusively, allocates/copies shared native parameter storage and applies
   saved modes/contexts. Build placeholders and supported objective from final parameter Tensors.<span data-glossary-navigation="true"> ([objective glossary](../../../../glossary.md#term-objective); [Tensors glossary](../../../../glossary.md#tensor))</span>
4. Open ONE ordinary TrainingSession with saved optimizer/groups; compare canonical parameters
   and call ordinary restore through Training's private-run admission/restore bridge.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [admission glossary](../../../../glossary.md#term-admission))</span>
5. Return only the complete run. Failure closes/discards everything acquired; session closes before
   parameter/placeholder Arenas, all cleanup attempted and failures retained/suppressed.

[Engine](../../../../glossary.md#term-engine) is [borrowed](../../../../glossary.md#ownership-borrowing-and-lease), never closed by load/[resume](../../../../glossary.md#schema-serialization-and-checkpoint)/[run](../../../../glossary.md#term-run). TrainingRun.trainBatch validates actual dense
dtype/Shape/[layout](../../../../glossary.md#layout)/offset/[view](../../../../glossary.md#term-view-meanings)/storage and binds borrowed fresh [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) storage to stable placeholder
IDs through replaceHostStorage; roles resolve once from compiled [metadata](../../../../glossary.md#term-metadata), associations clear in
finally. No hot-path string [dispatch](../../../../glossary.md#term-dispatch)/new input mapping authority, per-batch [compile](../../../../glossary.md#compile) or input closure.
Shape changes fail for fixed prepared recipe; an advancing [graph](../../../../glossary.md#graph)-[RNG](../../../../glossary.md#graph-rng-state--graphrngstate) stream is not provided.
Run/native model never escapes as indefinitely usable state. loadModel instead returns independent
heap-backed eval model, with no [arena](../../../../glossary.md#term-arena) or [TrainingSession](../../../../glossary.md#training-session--trainingsession); [inference](../../../../glossary.md#term-inference) uses forward-only Engine.compute.
High-level model-only load rejects legacy values-only files with a clear existing-[target](../../../../glossary.md#term-target)-load hint.

## Acceptance, documentation and [validation](../../../../glossary.md#term-validation)

Tests cover high-level no-[callback](../../../../glossary.md#term-callback) [MLP](../../../../glossary.md#term-multilayer-perceptron-mlp) [inference](../../../../glossary.md#term-inference) and [CPU](../../../../glossary.md#term-cpu) next-input [resume](../../../../glossary.md#schema-serialization-and-checkpoint) matching uninterrupted
admitted route; all four Training [dtypes](../../../../glossary.md#data-type--datatype), [optimizers](../../../../glossary.md#term-optimizer)/mixed groups/raw slots/counters/pending;
complete [descriptor](../../../../glossary.md#term-descriptor)/raw-byte/[alias](../../../../glossary.md#aliasing-and-arena)/recipe/[fingerprint](../../../../glossary.md#term-fingerprint) mismatch; corruption/nested/outer/projected
bounds; [immutable](../../../../glossary.md#term-immutable) values after source close; failed candidate preserving published state; native
resource cleanup at every failure and [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) [Engine](../../../../glossary.md#term-engine)/input [lifetime](../../../../glossary.md#term-lifetime); reusable static [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) [binding](../../../../glossary.md#term-binding).
Add bounded actual [Metal](../../../../glossary.md#term-metal) [smoke](../../../../glossary.md#term-smoke-test) with source-matched library and report absence/skips honestly;
no cross-[device](../../../../glossary.md#term-device) exactness or broad performance claim.

Before [Ready](../../../../glossary.md#term-ready) [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) root + owning extensions contract + explanations + ADR0038 acceptance
and architecture tests; include optional project and exact two api edges only in implementation.
Update CheckpointDependencyContractTest's absent-adapter assertion/exact edges and forbid reverse
edges; retain [model](../../../../glossary.md#term-model)-only exact three-edge check. Add durable end-to-end integration [fixture](../../../../glossary.md#term-fixture).
Clean implementation + independent targeted Class C [API](../../../../glossary.md#term-application-programming-interface-api)/docs review. Update user examples,
model/training API/[Javadoc](../../../../glossary.md#term-javadoc) and targeted glossary terms; shared-doc coordinator integrates once.

Future worker: `./gradlew :extensions:training-checkpoint:test :extensions:training-checkpoint:javadoc`.
Coordinator once after integration:
`SYNAPTIK_METAL_TEST_LIBRARY=<source-matched-library> ./gradlew test javadoc --max-workers=2 --console=plain`.
Reuse [owner](../../../../glossary.md#term-owner) evidence; independent reviewer does not rerun successful Java suites.
[Planning](../../../../glossary.md#term-planning) now: local links/anchors, brief ≤200 lines/15KB, `git diff --check`; no Java/build edits/tests.

## Result

Empty until implementation. Record changes/files/[validation](../../../../glossary.md#term-validation)/docs/limitations/follow-up and required
completion status; this [planning](../../../../glossary.md#term-planning) revision marks no code implemented.
