# Extensions and training contract

> This scoped contract is incorporated by the [authoritative architecture root](../../../ARCHITECTURE.md).
> It is normative only within the scope stated below and has no independent authority. Root global
> invariants and dependency rules apply everywhere and take precedence. Any overlap, contradiction,
> missing applicable scope, or ambiguity requires an explicit architecture update; do not choose
> between contracts silently.

## Scope

This contract owns NN, Training, model-only Checkpoint, the optional future Training Checkpoint
adapter, and ONNX responsibilities plus the optimizer/training lifecycle. It does not own Model
semantics, Compiler autograd, concrete backend optimizer routes, or Engine materialization.

## `extensions/nn`

`extensions/nn` owns the stateful neural-network composition layer. It defines model modules,
their trainable parameters and persistent buffers, training/evaluation mode, and the forward
context needed to apply that mode consistently through a module tree.

Allowed:

- `Module`
- `Parameter`
- `Buffer`
- module-owned parameter and buffer traversal
- training/evaluation mode propagation
- forward-context contracts
- neural-network layers, blocks, and functional conveniences composed from model semantics

Forbidden:

- optimizer algorithms
- optimizer update orchestration
- autograd construction
- backend storage access
- backend kernel selection
- concrete backend dependencies

`extensions/nn` depends on `modules/model` for tensor and operation semantics. It must not make
the model depend on neural-network layers or stateful module ownership.

### Closed declarative models

NN owns immutable semantic model specifications and reconstruction of a closed Tensor-to-Tensor
Sequential tree. This is the selected NN0026 contract, not a claim that its API is implemented.
The specification contains no live Tensor, callback, Java class name, backend, storage handle,
persistent version, or bytes. Existing generic Model/ModelDefinition and Sequential APIs remain
supported. Description never captures arbitrary Model.define code or custom module subclasses.

The supported tree has nested Sequential, Linear, affine LayerNorm, Embedding, Conv1d/2d/3d,
and closed RELU/TANH/SIGMOID steps. Parameter types are all four existing floating types:
FLOAT64, FLOAT32, BFLOAT16 and FLOAT16. There is no homogeneous-input-type shortcut:

| Step | Retained configuration and accepted input/output |
|---|---|
| Sequential | Ordered children; each receives its predecessor's exact result once. Empty returns the exact input. |
| Linear | Positive resolved in/out features, bias presence, parameter type, and whether constructed with automatic input guards. Supplied-state form retains Model promotion and contraction checks; automatic form retains exact configured input type and positive static final feature extent. Output preserves leading Dimensions and replaces the final extent with out-features. |
| LayerNorm | Positive fully static normalized trailing Shape, parameter type and exact ScalarValue epsilon type/bits; mandatory scale then bias. Input promotion, trailing matching and result-arithmetic epsilon checks remain Model-owned, including rejection when promotion makes retained epsilon incompatible. Output Shape is the input Shape. |
| Embedding | Positive vocabulary/embedding sizes and floating table type; no padding/frozen row. INT32/INT64 input of any rank; output appends embedding width and has table type. Index-value bounds remain the existing Model/backend obligation. |
| Conv1d/2d/3d | Positive resolved in/out channels and kernel extents, bias, exact parameter/input type and complete existing Conv1dAttrs/Conv2dAttrs/Conv3dAttrs (stride, symmetric padding, dilation, groups). Fixed NCW/NCHW/NCDHW layout; ranks 3/4/5, positive static matching channels and existing geometry guards. Batch/spatial Dimensions may vary compatibly; output geometry remains Model-owned. |
| Activation | One closed kind, delegating to the matching Model Tensor operation; floating input, same Shape/type, existing gradient and numerical semantics. |

Input/output constraints are derived by composing these rules, never saved from an example batch.
Compatible variable leading Dimensions/ranks for Linear and LayerNorm, arbitrary index Shape for
Embedding, and variable convolution batch/spatial Dimensions remain supported. Dynamic dimensions
are accepted only where the existing corresponding layer/Model contract accepts them. Description
requires all lazy parameters already bound; it does not run forward to initialize them. Initializers,
seeds, source labels/layouts/provenance and modes are not reconstruction configuration: selected
steps are mode-insensitive, reconstructed models start in evaluation mode, and state values load
separately. BatchNorm, Dropout, recurrence, branches, reused execution nodes and custom I/O are out.
An empty/activation-only tree is semantically eligible; NN promises neither generic zero-operation
Engine execution nor a newly admitted backend route, dtype, layout or gradient (including Conv3d).

The public boundary is ModelSpecification and final DeclarativeModel extending Model<Tensor,Tensor>
in nn.module. A specification has one closed Sequence root, immutable typed step records and
explicit AliasGroup path lists. ModelSpecification.describe(Sequential) describes an ordinary
supported tree without taking ownership; DeclarativeModel.describe() describes its current tree.
Both read current bindings under caller-owned quiescence, not creation-time alias metadata.
Each supported concrete layer exposes one typed specification() step snapshot; public Activation
in nn.layers supplies closed activation construction. No mutable-layer reflection or registry.
ModelSpecification.stateDescriptors() exposes immutable (path, kind, dataType, Shape, requiresGrad)
entries derived from configuration in exact Module.stateDictionary traversal order, without Tensors.
ModelSpecification.outputDescriptor(TensorDescriptor) checks composed input constraints and returns
the corresponding Model result metadata without evaluation, storage allocation or backend admission.

The root is represented directly by DeclarativeModel's indexed children, not by a named wrapper.
Root children are "0", "1", etc.; nested children extend the same dot-separated paths. Parameters
remain weight then optional bias, or LayerNorm scale then bias. No buffers occur in this matrix.
Numeric encounter order, not lexical sorting, defines traversal. A closed value tree carries no
node IDs, arbitrary connections or references: wiring/ownership cycles and dangling references
are unrepresentable through its safe public constructors. Reusing a step value means independent
occurrences, never shared module identity. NN validates nulls, supported kinds, positive dimensions,
parameter schema, epsilon, geometry and checked element/byte/count arithmetic before returning a
specification or skeleton. Composition compatibility is checked for a supplied input descriptor
through existing storage-free Model metadata operations on the cold path, and on each forward;
there is no generic shape solver or proof that a tree admits every possible input. Runtime-dependent
checks remain with their current owners; description needs no speculative input or fixed batch.
NN bounds each root to 4096 step occurrences (including sequences) and depth 64 (root depth 1),
checks limits before recursive reconstruction/copy, and uses stack-safe validation. Shape and
convolution array-count constraints retain their existing owner rules. These bounds protect cold
tree processing, not backend capability or runtime shape sizes. Future wire parsers separately
own unknown IDs/fields, canonical ordering, string/rank/length limits and aggregate/file-byte
preflight before allocating decoded objects; safe Java tree constructors do not validate raw bytes.

An alias group means repeated exact parameter Tensor identity at distinct state paths, not equal
values or overlapping physical storage. Groups have at least two distinct existing parameter paths,
equal type/Shape/gradient schema, disjoint membership, members in state order, and groups ordered
by first member. Unlisted paths are independent singletons. NN rejects invalid/noncanonical groups.
Description recomputes groups by exact current identity; independent equal-valued parameters stay
independent. Legal Parameter.replace or strict state loading may join/split groups; the next describe
must reflect that change. No physical-overlap topology, view layout or source expression is recreated.
Training's duplicate-identity/overlap rejection is unchanged; aliases are not split to evade it.

DeclarativeModel.reconstruct(ModelSpecification) returns a private-candidate, fully bound descriptor
skeleton, not a value-loaded executable model. It creates one storage-free, provenance-free,
gradient-eligible Tensor per alias group or singleton, with the derived static parameter descriptor,
and installs the exact same reference at every group path through NN-owned constructors/strict load.
Linear/Embedding/LayerNorm accept supplied descriptor Tensors; convolutions bind reservations via
strict load before any forward. Linear.fromState(Tensor, Optional<Tensor>, boolean automaticInputGuards)
creates a fully bound layer retaining the selected supplied-state or automatic guard policy, without
calling initialization. It is not a new lazy-initialization path or a change to existing constructors.
No random source, weight-sized zero/one array, throwaway initialized weights, evaluation, native
resource or new Engine path is permitted. Existing public stateDictionary/loadStateDictionary remain
the value bridge: a later recovery owner sees these fully bound alias references and can load one
fresh value Tensor per target identity group without changing its existing target-load contract.
NN reconstruction validates before returning; failure exposes no partial candidate and never mutates
the described source. Caller quiescence still owns consistency; NN adds no lock or transaction.

## `extensions/training`

`extensions/training` owns optimizer algorithms and training orchestration over parameters
declared by `extensions/nn` modules.

Allowed:

- `Optimizer`
- `Sgd`
- `Adam`
- `AdamW`
- `ParameterGroup`
- `TrainingSession`
- `TrainingStep`
- `TrainingParameterStorage` for an explicit pre-session copy into caller-owned shareable native host memory
- immutable in-memory optimizer/session state snapshots
- an explicit dependency on the public Engine facade for the session lifecycle
- backend-neutral access to caller-owned Model host storage within the supported optimizer domain

Forbidden:

- dependency on Runtime, Prepare, tuning, or concrete backends
- use of non-public Engine implementation state
- a second compiler, preparer, scheduler, runner, publication model, or backend adapter
- `MetalOptimizerBridge`
- `CudaOptimizerBridge`
- `CpuOptimizerBridge`
- backend-specific optimizer execution
- backend kernel selection

Training depends on `extensions/nn`, `modules/model`, and `modules/engine`, not the reverse. Because
its public signatures name `Module`, `Tensor`, `DataType`, `Shape`, Engine, and Engine metadata,
all three project edges are published Gradle `api` dependencies rather than hidden implementation
dependencies. The public Engine edge is the explicit Class C composition edge needed to reuse one
compile/Prepare/Runtime path; it does not transfer optimizer ownership into Engine or authorize
Training to inspect backend selection or representations. `train()` and `eval()` mode are module
forward-behavior concerns owned by `extensions/nn`; an optimizer neither selects nor changes that
mode.

Training owns backend-neutral optimizer mathematics over supported caller-owned host parameter
storage, not backend-specific optimizer execution. A later fused optimizer route belongs to
backend prepare and kernels and requires a separate architecture decision.

## `extensions/checkpoint`

`extensions/checkpoint` is the model-only recovery owner downstream of NN and the public
Engine facade. Its artifact package owns a version-one, single-regular-file envelope for opaque
bytes only; that package assigns no model meaning to a payload. The
canonical big-endian file has a fixed magic/version header, SHA-256 manifest digest, sorted strict
ASCII names, explicit bounded lengths, and SHA-256 per-payload digests. Readers reject malformed,
noncanonical, unsupported, over-limit, truncated, or trailing data before returning a detached
immutable image. Callers supply bounds for file and manifest bytes, entry count, name bytes,
individual payload bytes, and aggregate payload bytes. Readers bound the manifest length before
allocating its array, then check aggregate length arithmetic before allocating payloads. Its direct
public project dependencies are exactly `extensions/nn`, `modules/engine`, and `modules/model`.
NN retains module-state meaning and installation; Engine retains public host materialization.
Checkpoint must not inspect backend representations or reach through Engine into Compiler,
Runtime, Prepare, tuning, or concrete backends.

Its model package owns a separate version-one metadata codec, a detached candidate, and a public
`ModelCheckpoint` file and model-envelope facade; it is not a state installer. Under caller-owned
quiescence, capture calls
`Module.stateDictionary()` exactly once; an unbound or incomplete NN state fails there. Before
any Engine call or Checkpoint-owned payload-sized allocation, capture validates exact NN paths,
32-byte caller-supplied SHA-256 configuration and optional input-artifact fingerprints, static
Shape, checked canonical byte length, entry and payload limits, aggregate bytes including every
alias, manifest bytes, and final file bytes. It deduplicates exact Tensor identity in first-
encounter order only for one `Engine.compute(List<Tensor>, limit)` call. Empty state makes no
Engine call. It checks returned data type, Shape, and byte length before copying detached values
into the opaque envelope. No live Tensor, backend storage, or Engine handle escapes the candidate.
The caller defines what its versioned configuration and optional input-artifact digests cover;
Checkpoint cannot serialize arbitrary Java model code or infer tokenizer/configuration meaning.

The canonical model metadata payload has key `schema` and encodes, in big-endian order: eight
ASCII bytes `SYNMOD01`, signed 32-bit version `1`, 32 configuration-digest bytes, one input-
digest presence byte (`0` or `1`), optional 32 input-digest bytes, signed 32-bit entry count,
then ordered entries. Each entry is a signed 32-bit positive UTF-8 path byte length, exact path
bytes, one StateKind ID, one DataType ID, signed 32-bit nonnegative Shape rank, that many signed
64-bit nonnegative extents, and signed 64-bit expected canonical payload length. Paths use NN's
nonblank dot-segment grammar with valid Unicode; decoding requires strict UTF-8 and retains the
exact text without normalization. Kind IDs are `1` parameter and `2` buffer. DataType IDs are
`1` FLOAT64, `2` FLOAT32, `3` BFLOAT16, `4` INT32, `5` INT64, `6` BOOL, and `7` FLOAT16. IDs do
not derive from enum order or names. No padding or trailing bytes are permitted. SHA-256 of these
exact metadata bytes is the model schema fingerprint, so it binds both caller digests and every
ordered path, kind, type, Shape, and length.

The artifact kind is exactly `model.v1`; each state payload key is `state/` followed by its
zero-based, eight-digit ASCII decimal ordinal (`state/00000000` through `state/99999999`). NN
paths occur only in metadata, never in artifact keys or filesystem paths. Decoder acceptance
requires exactly one `schema` and one correctly keyed, correctly sized payload per ordered entry,
with no extra or missing entry, repeated path, unknown wire ID, negative extent, malformed UTF-8,
noncanonical length, or trailing metadata. A candidate is fully detached and immutable. The
configured aggregate limit bounds schema plus every state's canonical output bytes, including
aliased paths; it does not bound Engine workspace, source detached values, defensive copies, or
peak heap. The envelope's own v1 framing and checksums remain unchanged. No cross-call cache,
implicit initialization, or model-defining code reconstruction is provided. Public inspection
fully validates file framing, schema, payload correspondence, and canonical BOOL bytes before
returning detached immutable typed descriptors and defensive digest copies. Load rereads and
revalidates the file, checks exact caller configuration and optional input-digest presence and
bytes, and rejects an unbound target by requiring a complete NN state snapshot before
installation. It validates every path, kind, type, Shape, gradient requirement, and target
Tensor-identity alias group before constructing a complete candidate dictionary. The v1 wire
stores one value per path and cannot prove source Tensor identity: each fully bound target alias
group must have identical stored bytes, then receives one fresh dense-contiguous host Tensor per
group. A group containing a parameter receives a gradient-eligible Tensor; buffer-only groups do
not. Canonical big-endian bits are restored into data-type-compatible native-order heap carriers
without numeric conversion or NaN normalization. NN alone performs the one strict whole-tree
installation attempt after all checkpoint validation and reconstruction. Save and load require
caller-owned quiescence; a prior inspection is not trusted for later load.

The public model-envelope composition boundary implements the scope approved for Checkpoint
0005A alongside the file-based facade. `ModelCheckpoint` exposes these static operations using
existing public types:

- `capture(Module, Engine, byte[] configDigest, byte[] inputDigest, ArtifactLimits)` returns
  `ArtifactEnvelope` and declares checked `ArtifactFormatException` for semantic decode failure.
- `inspect(ArtifactEnvelope, ArtifactLimits)` returns `ModelCheckpointInspection` and declares
  checked `ArtifactFormatException`.
- `canonicalStateBytes(ArtifactEnvelope, ArtifactLimits)` returns `Map<String, ByteBuffer>` and
  declares checked `ArtifactFormatException`.
- `load(ArtifactEnvelope, Module, byte[] expectedConfigDigest, byte[] expectedInputDigest,
  ArtifactLimits)` returns `void` and declares checked `ArtifactFormatException`.

These operations must reuse the current internal capture, model codec, and restoration machinery,
including `ModelArtifactCodec.decode` and its preflight of every `ArtifactLimits` dimension before
schema copying. Standalone envelopes have not passed `ArtifactFile.read`: validate kind, schema,
exact payload correspondence, canonical BOOL bytes, and all limits, including the equivalent v1
manifest and complete file size, on each call. An in-memory envelope has no encoded file checksum
to verify; file framing and integrity validation remain the file reader's responsibility.
Capture returns only a fully validated detached envelope. No live Module, Tensor, Engine, or
borrowed storage lifetime escapes. Limits retain their existing meaning: aggregate canonical
bytes include schema and every aliased path, not defensive copies, workspace, or peak heap.

Canonical bytes include every parameter and buffer under its exact NN path in model metadata
order, not lexical path order or artifact-key order. The map structure is immutable and each
call returns fresh independent read-only buffers positioned at zero, with exact payload limits
and `BIG_ENDIAN` byte order. Payload bytes are immutable; buffer position, limit, and byte-order
settings remain caller-mutable and must not affect another call or path view. All seven stored
data types retain their raw logical bits, including signed zero and NaN payloads; no conversion
or numerical normalization is authorized. Inspection supplies descriptors, not a validation
credential: canonical-byte access and load independently validate the supplied envelope.

Envelope load must validate the complete model and expected digests before delegating one strict
NN installation attempt, preserving the fully bound target and target-defined alias policies.
The file API must delegate to the same semantic machinery while retaining rereads on every
inspect/load, existing checked `IOException` signatures, failure types and validation ordering,
and the existing save publication policy. The internal `ModelArtifactCandidate`,
`ModelArtifactCodec`, `ModelArtifactCapture`, and `ModelStateRestoration` remain package-private.
This boundary adds no public model-image wrapper, schema, codec, registry, temporary-file
composition pipeline, backend access, Training validator, or shared transaction.

For a trusted existing local directory and quiescent caller-owned target, save writes a newly
created sibling temporary file, forces that file through its writing channel, closes, reopens, and
fully validates it against the source image, then attempts exactly one `ATOMIC_MOVE`. It may
replace an existing regular target; it rejects a symlink or directory target before work. There
is no copy/non-atomic fallback or portable exclusive no-replace promise. A successful move gives
atomic live visibility of the prior complete file (or absence for a new target) or the new complete
file, not a cross-platform guarantee that the rename survives power loss. A failed move may have
provider-dependent target state: save reports failure, never deliberately deletes the target,
and callers must inspect it. Pre-move failures attempt to clean the owned temporary file. An explicit
unsupported-atomic-move result permits attempted temporary cleanup; other failed move attempts leave the
temporary path untouched because their outcome is uncertain. No filesystem operation follows a
successful move. The envelope is not encryption or adversarial authentication.

Checkpoint must not depend on Training or Text. Model, NN, Training, Engine, and other inward
modules must not depend on Checkpoint. The direct public Model edge is limited to checkpoint
Tensor descriptors and restoration; it does not authorize backend storage access or a second
execution pipeline.

A separate optional `extensions/training-checkpoint` adapter is reserved for durable recovery.
Its selected first product is Model + TrainingState only, with a saved-state restoration
guarantee, not complete automatic exact continuation. A concrete downstream need for the current
model-envelope API is comparison of the complete model parameter descriptors and raw logical
bytes against TrainingState before restore/publication. External input position/cursor, RNG,
Module modes, effective forward contexts, scheduler, and scaler remain application-owned.
Broader exact-continuation claims still require all affecting state under the lifecycle below.
The adapter may depend on Training and Checkpoint, but is neither included in
Gradle settings nor implemented until its later adapter gate. Training keeps optimizer/session
state meaning, and the caller and other state owners keep their own validation and restore
authority under the optimizer/training lifecycle below. Neither Training nor Checkpoint gains a
reverse dependency or a generic persistence or state-installation API from this reservation.
Model-envelope implementation does not select a training wire format or open the Checkpoint 0006
adapter build gate; 0006 remains Draft.

## `extensions/onnx`

`extensions/onnx` owns ONNX import/export and mapping.

It must not be part of runtime hot path.

Allowed:

- ONNX import
- ONNX export
- ONNX-to-model mapping
- model-to-ONNX mapping

Forbidden:

- runtime execution
- backend-specific lowering
- kernel selection
- runtime residency


## Optimizer/training lifecycle

The initial public lifecycle is:

```text
open:
  deterministic Module parameter path/wrapper/Tensor/storage snapshot
  -> one scalar-objective first-order Engine compile
  -> one Engine inference-session preparation
  -> TrainingSession

run:
  validate the complete module/input/storage view before execution
  -> reuse the prepared forward/backward execution
  -> validate and detach the objective and every target-indexed gradient occurrence
  -> stage gradient accumulation, parameter values, and optimizer slots
  -> validate the complete parameter view again
  -> commit all parameter bytes, optimizer slots, counters, and accumulation state
```

Rules:

- `extensions/nn` owns module-declared trainable parameters, persistent buffers, and train/eval
  forward behavior.
- A `TrainingState` snapshot restores only its detached optimizer/session state: exact optimizer
  configuration and ordered parameter schema, logical and master parameter bytes, optimizer slots,
  pending gradient sums, and execution, optimizer-step, and accumulation counters. It does not
  contain persistent NN buffers, Module modes, caller inputs or progress, graph random state, or
  external scheduler/scaler state. `Module.stateDictionary()` retains exact Tensor references and
  is not a detached value snapshot; `TrainingState` supplies detached parameter bytes within its
  domain, while a recovery owner must separately detach NN buffer values and any parameter values
  outside that domain. Each Module has a local mode field; recursive `train()`
  and `eval()` set the whole owned tree, while `StateDictionary` omits modes.
  A supplied immutable `ForwardContext` controls a mode-sensitive forward even when it differs
  from that module's local mode. An exact continuation therefore requires the effective
  objective-construction contexts, not merely saved Module modes, to be known and reproducible.
  Training-mode `BatchNorm.forward` installs symbolic next-buffer expressions during graph
  construction. Neither a shallow dictionary nor a successful session run establishes a
  step-aligned detached value for those expressions; that workflow is currently ineligible for
  exact continuation until a supported capture/publication boundary proves the values belong to
  the completed step.
- An exact continuation claim for the currently supported session means the same admitted
  computation route resumes at the next caller-selected input with all state that can affect that
  run restored. The caller owns an explicit handoff for the next input position (including sampler
  cursor and epoch when applicable), external random-generator state, and any program-owned
  scheduler or scaler state. Those participants are not inferred from Training counters. If a
  participant cannot be captured, validated, and restored, the result is a partial restore or
  restart, not exact continuation. No built-in scheduler or scaler state exists today. A durable
  exact artifact may include caller-owned participants only where each has a supported versioned,
  bounded schema and codec plus owner validation and restore; arbitrary caller objects have no
  generic exact-resume encoding.
- Graph random state is an explicit Model expression chosen when the objective is built.
  `GraphRngState.initial(key, counter)` creates a storage-free fixed expression, not a mutable
  stream. The current `TrainingSession` compiles that objective once and has no per-run graph-RNG
  input/output handoff; repeated runs do not automatically advance dropout state. A caller may
  record the key/counter and rebuild an equivalent graph before opening a new session, but must
  not claim a changing per-step dropout stream within the current fixed session. Exact
  continuation does not promise bitwise-identical results across devices, backends, versions, or
  execution routes.
- No new public Training value or API is required to capture the existing detached
  `TrainingState`; caller-owned progress and random state remain explicit, separately typed
  handoff data under the caller's schema. A changing graph-random stream requires a separately
  designed per-run handoff before that claim can be widened.
- NN state installation and Training restore currently validate only their own boundaries and
  do not form a cross-owner atomic transaction. `Module.loadStateDictionary(...)` replaces Tensor
  bindings and is forbidden while a Training session over that tree is open: the session captured
  fixed wrapper, Tensor, storage, and objective identities at open. For a coordinated exact
  recovery, the caller quiesces published state and prepares a distinct, isolated, unpublished
  replacement tree and session. Before mutating any published object, it identifies all required
  participants: detached NN values or a bounded model artifact, per-module modes, Training state,
  effective forward contexts, fixed-objective construction inputs, and typed caller-owned state.
  It rejects missing or unsupported participants and validates every available detached fact,
  including schemas, exposed values, and declared route requirements. A shallow
  `StateDictionary` is not detached value evidence. The caller stages external progress, random,
  scheduler, or scaler state so it can be handed off under exclusion without changing published
  state during candidate preparation.

  On the private tree, install NN bindings and modes first, revalidating any model artifact during
  load. Prepare supported native parameter storage if needed, then construct the objective from
  those final Tensors and open the replacement session. Before restore and publication, compare
  the candidate's canonical logical parameter bytes with the supplied Training state's logical
  bytes at every stable path. This comparison
  may be made from already comparable detached values before open, or from the new private
  session's `state()` snapshot after open; `ModelCheckpoint.inspect` alone exposes no value bytes.
  Session-specific compatibility and payload validation may then occur through
  `TrainingSession.restore`; it need not be duplicated in a pure pre-open Training validator.
  Validate the resulting candidate and caller-state handoff completely, then publish the whole
  replacement only after success under caller-owned exclusion. On ordinary failure, close and
  discard the private candidate without publishing it. The old session need not be closed before
  a distinct private candidate is prepared; the caller owns its quiescence and retirement during
  the handoff. No end-to-end rollback of an already closed old session, general transaction API,
  or simultaneous visibility to readers outside the exclusion is promised. If a participant
  cannot be staged and handed off safely, do not claim exact recovery. Training 0004 must test
  both rejection before publication and the successful next-input handoff. A later Checkpoint
  adapter may persist only supported, validated participant schemas and must not invent an
  NN-to-Training dependency.
- Before opening a session, a caller may explicitly ask Training to copy a fully initialized
  Module tree's current parameters into a caller-owned shareable Arena. Training validates the
  complete recursive parameter set and source storage before replacing any Tensor association,
  copies exact represented bytes into fresh, aligned, exact-capacity writable native segments,
  and installs the associations in path order without replacing Parameter or Tensor identities.
  Source storage may be heap or native and read-only; duplicate Tensor identities and overlapping
  source parameter regions are rejected. Ordinary installation failure restores earlier exact
  associations. The caller owns both source and destination lifetimes, must keep the destination
  Arena alive through any session, and must give the copy exclusive Module/storage use. Copying
  while a session is open is forbidden. This does not change the native-only capture domain of
  `TrainingSession.open` or introduce a backend transfer.
- A `TrainingSession` borrows exactly one caller-owned Engine and one caller-owned Module tree. It
  owns exactly one Engine `InferenceSession`, one optimizer-state set, and its counters.
- Session construction snapshots recursive parameter paths in Module traversal order. Every
  wrapper, Tensor identity, and non-overlapping storage occurrence remains fixed until close;
  callers grant exclusive module/parameter-storage use and must not replace bindings or storage
  associations while the session is open.
- The public session supports a scalar `FLOAT32` or `FLOAT64` objective and connected, non-empty
  `FLOAT64`, `FLOAT32`, `BFLOAT16`, or `FLOAT16` parameters with fully static, dense-contiguous,
  offset-zero, non-view descriptors and exact-capacity writable native host storage accessible to
  every thread that operates on the session. Every parameter has a strictly positive element count
  and finite initial values; rejection occurs during capture before cotangent-seed allocation,
  compilation, or preparation. BFLOAT16 and FLOAT16 parameters retain private FLOAT32 masters and
  publish one narrowed logical value only after a successful update. Parameter storage is updated
  in place so compiled Tensor identities remain stable. JVM-heap, read-only,
  confined-to-another-thread, device-only, empty, integral, BOOL, and non-finite parameter storage
  are outside the update domain and fail before compile or mutation.
- Compiler input occurrence order and publication target indices are the only mapping authorities.
  Parameters bind internally by captured Tensor identity; each other bindable input is supplied
  exactly once per run. A missing, duplicate, foreign, descriptor-incompatible, absent, dead,
  inaccessible, overlapping, or parameter-aliasing storage fails before optimizer mutation.
- `Sgd`, `Adam`, and `AdamW` are immutable built-in optimizer configurations. The exact default
  and ordered parameter groups resolve one algorithm per path at open. All parameters update on
  every successful global step through one atomic commit, with no backend or per-element
  algorithm lookup. `Sgd` owns immutable learning-rate, momentum, dampening,
  weight-decay, and Nesterov configuration; its session owns one zero-initialized momentum slot
  per parameter. Coefficients are finite; momentum is in `[0, 1)`, dampening in `[0, 1]`, and
  weight decay is non-negative. Nesterov requires positive momentum and exactly zero dampening.
  Coupled weight decay first produces `adjusted = gradient + weightDecay * parameter`. The first
  momentum step stores `adjusted` without dampening. Later steps store
  `momentum * previous + (1 - dampening) * adjusted`. Ordinary momentum selects that new slot;
  Nesterov selects `adjusted + momentum * updatedMomentum`. The parameter candidate is
  `parameter - learningRate * selectedGradient`. FLOAT32, BFLOAT16, and FLOAT16 parameter sessions
  validate the same invariants after coefficient narrowing, including rejecting positive
  Nesterov momentum that narrows to zero, and use FLOAT32 master, momentum, accumulation, and
  candidate arithmetic. A successful low update narrows the validated master candidate once to
  the declared logical type. FLOAT64 uses the configured binary64 values.
- `Adam` and `AdamW` require finite positive learning rate and epsilon, finite beta1 and beta2
  in `[0, 1)`, and finite non-negative weight decay. FLOAT32, BFLOAT16, and FLOAT16 assignments
  narrow coefficients once to binary32 at open and reject a rate or epsilon that becomes zero,
  a beta that becomes one, or any out-of-range or non-finite coefficient. Both have zero-initialized
  first and second moments in session state. At one-based global step `t`, they calculate
  `m = beta1 * oldM + (1 - beta1) * g` and `v = beta2 * oldV + (1 - beta2) * g * g`,
  divide them by `1 - beta1^t` and `1 - beta2^t`, and subtract
  `learningRate * correctedM / (sqrt(correctedV) + epsilon)`. Adam first adds
  `weightDecay * masterParameter` to `g` (coupled decay). AdamW leaves `g` unchanged and first
  multiplies the master parameter by `1 - learningRate * weightDecay` (decoupled decay).
  Accumulate-and-step uses the pending gradient sum, not its mean. Bias corrections are validated
  outside element loops; all moments and arithmetic intermediates must be finite and second
  moments non-negative.
- A reset-and-step run uses only that run's gradients and clears older accumulation on success.
  An accumulate run atomically adds the run's gradients without updating parameters or optimizer
  step count. An accumulate-and-step run applies the pending sum plus that run's gradients and
  clears it on success. `zeroGrad` clears only pending accumulation.
- Optimizer step numbers start at one and advance exactly once after each successful parameter
  update. Execution numbers advance after every successful forward/backward run, including an
  accumulate-only run. Failed runs advance neither.
- Run/materialization, gradient decoding, candidate calculation, and complete precommit
  validation precede all mutation. Current parameters, decoded gradients, accumulated gradients,
  optimizer slots, every arithmetic intermediate, and parameter candidates must remain finite in
  the parameter precision. NaN, either infinity, or represented overflow is an ordinary rejected
  failure; either signed zero is finite and supported. A non-finite detached objective alone is
  not optimizer state and does not invalidate a run whose required gradients and optimizer
  arithmetic are finite. Restore rejects a non-finite parameter, optimizer slot, or accumulation
  payload. Supported native parameter stores make ordered path-order primitive writes
  non-throwing after validation. Any ordinary validation, execution, materialization, non-finite,
  overflow, or optimizer failure leaves parameter bytes, optimizer slots, counters, and pending
  gradients unchanged. Caller violation of exclusive storage use and fatal virtual-machine
  failure are outside that atomicity guarantee.
- Every operation first attempts lifecycle admission without inspecting its arguments. Admission
  observes Engine closure first, then session closing/closure, then an already-admitted busy
  operation; only the admitted operation validates its arguments inside a `finally`-guarded
  operation lifetime. Close atomically rejects later admission, waits for the admitted operation
  without holding an Engine or Runtime lock, closes all owned resources once, and makes
  repeated/concurrent close replay the retained cleanup result. An operation admitted before
  session close begins may finish normally before cleanup.
- `TrainingStep` and immutable in-memory session-state snapshots own detached canonical bytes and
  remain readable after session or Engine close. Logical parameter payloads use their declared
  width; master, first-slot, optional second-moment, and accumulated-gradient payloads use
  FLOAT32 width for BFLOAT16,
  FLOAT16, and FLOAT32 parameters and FLOAT64 width for FLOAT64 parameters. A restore validates the
  complete optimizer kind/configuration, path/order/schema, counters, payload finiteness, exact
  slot presence and widths, and non-negative Adam/AdamW second moments before installing any state.
  SGD has no second-moment payload. The existing momentum payload remains the first-slot payload
  for every algorithm, with a neutral first-moment alias. The default optimizer accessor returns
  `Optimizer`, including Adam and AdamW defaults, rather than only `Sgd`.
- CPU, Metal, and supported mixed Engine preparations use this same lifecycle. Training never
  branches on backend identity. This lifecycle does not independently prove or widen low-precision
  Metal graph capability. Unsupported compile, prepare, execution, publication, or materialization
  fails before optimizer mutation.
- `FORWARD_ONLY` performs no autograd. `FORWARD_AND_BACKWARD` and the initial `TRAINING_STEP` use
  the same combined pre-capture forward/backward construction; the initial `TRAINING_STEP` adds no
  optimizer-update graph work.

A later architecture version may compile optimizer updates into the graph. That change requires a
separate explicit decision and does not weaken the current concrete-backend dependency prohibition.
