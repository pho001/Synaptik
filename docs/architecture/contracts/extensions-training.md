# Extensions and training contract

> This scoped contract is incorporated by the [authoritative architecture root](../../../ARCHITECTURE.md).
> It is normative only within the scope stated below and has no independent authority. Root global
> invariants and dependency rules apply everywhere and take precedence. Any overlap, contradiction,
> missing applicable scope, or ambiguity requires an explicit architecture update; do not choose
> between contracts silently.

## Scope

This contract owns NN, Training, generic graph/value Checkpoint, the optional future Training Checkpoint
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
- backend storage access (public Model host storage for owned Models and checkpoint restoration is permitted)
- backend kernel selection
- concrete backend dependencies

`extensions/nn` has exactly public Model, Training, Checkpoint and Engine project dependencies. Model supplies tensor
and operation semantics; Training supplies the generic `TrainingParameters` admission boundary.
`Module.trainingParameters()` returns a private live adapter retaining the root and exact ordered
paths, Parameter wrappers and Tensors. Creation rejects unbound reservations. Its immutable entries
snapshot is not validation: `validateUnchanged()` re-traverses the live root and rejects any count,
order, path, wrapper or Tensor change. It deliberately does not freeze storage associations or
bytes, so an explicit native copy does not invalidate its own source. NN still owns no optimizer
algorithm or update orchestration. Owned Models delegate updates to TrainingSession and execution
to public Engine preparations; Checkpoint supplies cold graph/value persistence. The retained NN
persistence signatures below remain supported. Model must not depend on NN.

### Owned tensor Models

Model remains abstract and generic and implements AutoCloseable. Generic/custom/declarative
forward behavior is unchanged; unsupported tensor operations throw UnsupportedOperationException
before Engine or file work, and default close is a no-op. defineTensor returns `Model<Tensor,Tensor>`;
graph/loadGraph return `Model<List<Tensor>,List<Tensor>>`. Single load rejects multiple boundaries.
Only private concrete specializations and package-private source/execution state belong in nn.module.

defineTensor seals its definition once, selects eval before one prototype forward (including
deferred initialization), then discards callbacks. Registered wrappers establish PARAMETER/BUFFER
roles, never requiresGrad alone; remaining reachable stored leaves receive deterministic collision-free
CONSTANT names. Every registered state must be reachable. Graph/load state paths must follow canonical
Module traversal order. Dotted namespaces contain no forward recipe. Constants are outside stateDictionary.
GraphCheckpoint.copy acquires each external identity once and reconstructs against private leaves;
strict state installation preserves wrappers, paths and aliases without mutating caller Tensors or stores.
Load adopts final heap state without another copy. Callback side effects are not rolled back.
Source admission remains static ADD/MUL/NEG/MATMUL/PERMUTE[1,0]; raw persistence supports seven dtypes,
independently of execution. Old model.v1/whole-model.v1 readers and flexible forward remain separate.

All declared static slots, including unused ones, require exact descriptors (layout presence and
requiresGrad included). Compile/session input order determines a cached positional binding plan;
surviving batch/target slots and owned leaves fill it. Unmapped objective inputs reject. Compute/train
reuse public runValues: no per-batch source rebuild, rebind, prototype mutation, copy or provider lookup.
Compute checks every publication and aggregate byte bound before materializing any ordered occurrence,
closes RunResult on every path, and returns detached HostTensorValue values (default Long.MAX_VALUE).
Batch storage is borrowed for the whole synchronous call. TrainingStep is detached too.

withEngine returns the same receiver and only selects a borrowed Engine, never closing it. Admission
checks selected-Engine closure, Model closing/closure, busy, then capability. withEngine then checks
freeze before null/closed candidate validation. Selection remains unchanged on failure. Valid compute
or configure arguments and live-source checks freeze binding before Engine creation, callback, native
copy, compile or prepare, even on failure; same-Engine rebinding then rejects. Without explicit selection,
setup lazily creates a model-owned CPU-only Engine.standard, with no discovery/global Engine.
Construction/load/save/cold forward never create an Engine.

configureTraining succeeds once, keeps mode unchanged, and calls a pure objective builder once per
explicit attempt, retaining no callback. The scalar FLOAT32/FLOAT64 objective uses current Compiler
support but is not persisted. TrainingSession receives Module.trainingParameters. Close inference
preparations before TrainingParameterStorage.copyToSharedNative installs parameter storage in a
model-owned shared Arena without changing Tensor identity. Copy failure preserves old associations;
successful native admission is not repeated on setup retry. Failed session setup cleans unpublished
owners and permits explicit frozen-Engine retry. Training retains unresolved-layout, empty, nonfinite,
alias/overlap rejection and RESET_AND_STEP transaction semantics; NN never converts or splits state.

Before compute/train/configure/cold forward/save, guard every module mode and ordered
path/wrapper/Tensor/storage association. Mutation invalidates the capability and closes cached
owners, without recapture, adopting replacement state or resetting optimizer history. Same-store
trained bytes stay valid. Inherited Module mutators remain caller-coordinated.
Only one high-level operation is admitted; busy rejects before arguments. Close bypasses admission,
rejects new work, waits for active work, then closes session/preparation, native owners and owned
Engine, attempting all cleanup, suppressing distinct failures and replaying the retained result
on repeated/concurrent close, even when a borrowed Engine closed first.

Save uses the fixed GraphCheckpoint source and current owned leaves only, including before compute
and after training. Explicit ArtifactLimits remain. Short forms use 256 MiB file/aggregate,
4 MiB manifest, 16384 entries, 64-byte keys and 64 MiB payload ceilings. No optimizer/RNG/cursor
resume or Model/Training autotuning is implied. Zero-node source can persist; Engine's empty-plan
rejection remains unchanged.

### Closed declarative models

NN owns immutable semantic model specifications and reconstruction of a closed Tensor-to-Tensor
Sequential tree. The NN0026 API is implemented, independently reviewed and coordinator-validated.
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

### Retained NN checkpoint APIs

The nine old model persistence classes now live in `nn.checkpoint`, without compatibility
forwarders. The three public types remain ModelCheckpoint, ModelCheckpointInspection and
InputArtifactFingerprint. The following old-v1 rules belong to NN; they do not restrict the new
generic graph format or transfer NN semantics into Checkpoint.

Its model package owns a separate version-one metadata codec, a detached candidate, and a public
`ModelCheckpoint` file and model-envelope facade; it is not a state installer. For the existing
values-only operations, under caller-owned quiescence, capture calls
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

These relocated NN APIs retain their existing semantics, readers, bounds and failures. They use
Checkpoint's opaque artifact package and public Engine materialization, never Compiler or backend
internals. Their package is `nn.checkpoint`; no old-package forwarders remain.

### Whole-model definition and composition

This is the accepted CP0005B contract (2026-10-14), independently approved and coordinator-verified
for implementation. CP0005B is Complete (2026-10-15): implementation, independent code and
documentation reviews, module/Javadoc and coordinator gates passed. NN's closed declarative
contract above owns meaning; NN's relocated `nn.checkpoint` package owns the
following old persistent representation and public composition. Generic Checkpoint owns only the
opaque envelope and the separate graph-model.v1 boundary below. Training and recurrent extensions
remain Proposed. Existing values-only methods, `model.v1`, ArtifactFile v1, and ArtifactEnvelope's
defensive-copy boundary remain unchanged. The relocation adds only the root-declared NN public Checkpoint/Engine edges; no Java-object
serialization or second execution pipeline is added.

#### Whole-model public API and failures

Add only these static overloads to existing `nn.checkpoint.ModelCheckpoint` (types use their
existing owning packages). File methods declare `IOException`; envelope methods declare
`ArtifactFormatException`, the checked IOException subtype. These signatures are accepted:

```text
void save(Path, Module, Engine)
void save(Path, Module, Engine, ArtifactLimits)
void save(Path, Module, Engine, InputArtifactFingerprint, ArtifactLimits)
DeclarativeModel load(Path)
DeclarativeModel load(Path, ArtifactLimits)
DeclarativeModel load(Path, InputArtifactFingerprint, ArtifactLimits)
ArtifactEnvelope captureWholeModel(Module, Engine, ArtifactLimits)
ArtifactEnvelope captureWholeModel(Module, Engine, InputArtifactFingerprint, ArtifactLimits)
DeclarativeModel loadWholeModel(ArtifactEnvelope, ArtifactLimits)
DeclarativeModel loadWholeModel(ArtifactEnvelope, InputArtifactFingerprint, ArtifactLimits)
```

The Module source is recognized only as Sequential or final DeclarativeModel; use respectively
ModelSpecification.describe(source) or source.describe(). Reject custom roots/children and
unbound state before Engine or filesystem work. A save checks its required arguments and captures
before calling ArtifactFile.save; no temporary component files. Entire-call caller quiescence
spans description and public values capture. Each may read state: values capture's existing
stateDictionary-once rule is not a combined whole-model snapshot, lock or transaction promise.

The only new public type is final `io.github.pho001.synaptik.nn.checkpoint.InputArtifactFingerprint`:
public constructor `InputArtifactFingerprint(byte[] bytes)`, public `byte[] bytes()`, and value
`boolean equals(Object)`/`int hashCode()`. Constructor and accessor copy exactly 32 SHA-256 bytes; null rejects
with NullPointerException, wrong length with IllegalArgumentException. Equality uses all 32 bytes.
It is expected application-verified identity, not verification performed by Checkpoint, a trust
credential or authentication. Fingerprint overloads require a non-null value. Overloads without
it save absent input digest and require absence on load; explicit load requires presence and exact
bytes. A mismatch, including stored presence on default load, throws ArtifactFormatException with
an explicit-expected-identity diagnostic. No generic options, registry or input-artifact parsing.

Required null arguments throw NullPointerException. Unsupported/unbound sources, source schema
and preflight failures retain existing IllegalArgumentException/IllegalStateException and checked
arithmetic ArithmeticException. Engine unchecked failures propagate. Malformed/unsupported stored
data, incoherent descriptors/config/aliases, input mismatch and decode bounds/overflow are
ArtifactFormatException (wrap owner validation failures with their cause). File access/publication
failures remain IOException; fatal VM failures are not converted. Tensor-identifier exhaustion
remains IllegalStateException. A values-only file fails high-level load with a missing-definition
message directing callers to existing target-load, never inferred topology or silent upconversion.

#### Whole-model limits and preflight

Short file overloads use `ArtifactLimits(268435456L, 4194304, 16384, 64, 67108864, 268435456L)`:
complete file/aggregate 256 MiB, manifest 4 MiB, entries 16384, ASCII kind/key 64 bytes, single
payload 64 MiB; MiB is 1048576 bytes. Advanced overloads reuse existing ArtifactLimits, but every
definition is at most 1048576 bytes even with larger limits. NN's 4096 occurrences/depth 64 still
apply (root counts once at depth 1). These are encoded bounds, not peak-heap/workspace promises.

Dry-size the canonical definition with checked arithmetic before creating its byte array. Let D
be that size and n the predicted inner model entry count (stateDescriptors.size() + 1 for schema).
The inner manifest is `66 + 58*(n-1)` bytes. Outer kind length is 14 instead of 8, each inner key
gains the 6-byte `model/` prefix, and the new 17-byte `definition/schema` key adds 61 bytes:
manifest delta is `67 + 6*n`, file delta `D + 67 + 6*n`, aggregate delta D, entry delta 1.
Check outer kind/definition key lengths, D against both definition and single-payload ceilings,
and checked residuals before invoking public capture. Pass projected limits with file minus file
delta, manifest minus manifest delta, entries minus 1, key bytes minus 6, unchanged maxPayloadBytes,
and aggregate minus D. Reject residual file <48, manifest <8, negative counts/bytes or unusable
key ceilings safely before constructing ArtifactLimits. With separately checked outer keys/D,
these residuals are faithful equivalence to the outer framing, not a new public validator or a
duplicate private schema/ordinal parser. The feasible capture order is: dry-size D and check outer
definition/residual metadata bounds; encode the already bounded definition (at most 1 MiB); hash
those exact bytes; pass that digest and projected limits to public values capture, whose actual
component preflight precedes Engine and bulk value-sized allocations/copies. The bounded definition
allocation is metadata, not bulk values; no new public validator/streaming proof layer is required.
After capture, check complete outer sizes/coherence again before prefixed payload copies; caller
quiescence makes the described count stable. Validate final correspondence anew.

On envelope load, preflight all six outer encoded limits using entry sizes/read-only views and
checked `48 + manifest + aggregate` before any definition/payload copy or projected envelope
construction. Check exact outer keys/kind and definition ceiling, then apply the equivalent
projected limits and public component validation. File load uses ArtifactFile.read first; its
framing preflight already precedes file payload allocation. Raw nested counts/lengths are bounded
by remaining definition bytes and derived state counts before arrays/lists/strings are allocated.
In particular positive LayerNorm rank requires room for `8*rank` extent bytes plus its typed
scalar before allocating long[]; no backend rank-16 restriction belongs in this format.

#### Canonical definition version 1

All fields are big-endian, with no padding, optional extensions or trailing bytes. `i32`/`i64`
are signed 32/64-bit fields; counts are nonnegative unless specified positive. `u8` is one byte.
Flags accept exactly 0 or 1. The payload is eight ASCII magic bytes `SYNDEF01`, i32 version 1,
one root node (must be Sequence), then i32 alias-group count and those groups. Nodes use prefix
order: one u8 kind ID then precisely the fields below; Sequence children follow immediately.
No IDs, edge tables, cycles, saved batch signatures or redundant output descriptors are encoded:
the closed value tree represents wiring, and NN derives input/output constraints from these fields.

| ID / kind | Fields after kind, in exact order |
|---|---|
| 1 Sequence | i32 child count; each child node in numeric encounter order. |
| 2 Linear | i64 inFeatures; i64 outFeatures; u8 bias; u8 parameter DataType; u8 automaticInputGuards. |
| 3 LayerNorm | u8 parameter DataType; positive i32 normalized rank; positive i64 extents in axis order; ScalarValue epsilon. |
| 4 Embedding | i64 vocabularySize; i64 embeddingSize; u8 parameter DataType. |
| 5 Conv1d | i64 inChannels, outChannels, kernelWidth; u8 bias; u8 parameter DataType; i64 stride, padding, dilation, groups. |
| 6 Conv2d | i64 inChannels, outChannels, kernelHeight, kernelWidth; u8 bias; u8 parameter DataType; i64 strideHeight, strideWidth, paddingHeight, paddingWidth, dilationHeight, dilationWidth, groups. |
| 7 Conv3d | i64 inChannels, outChannels, kernelDepth, kernelHeight, kernelWidth; u8 bias; u8 parameter DataType; i64 strideDepth, strideHeight, strideWidth, paddingDepth, paddingHeight, paddingWidth, dilationDepth, dilationHeight, dilationWidth, groups. |
| 8 Activation | u8 activation ID: 1 RELU, 2 TANH, 3 SIGMOID. |

Feature/channel/vocabulary/kernel/stride/dilation/group fields are positive element counts;
padding is nonnegative elements per side, not total padding. NN's existing geometry, divisibility,
epsilon, element/byte arithmetic and Java-array constraints apply through its public constructors.
DataType IDs are the unchanged model.v1 assignments: 1 FLOAT64, 2 FLOAT32, 3 BFLOAT16, 4 INT32,
5 INT64, 6 BOOL, 7 FLOAT16. Parameter fields permit only 1/2/3/7. No enum ordinal or class name.
ScalarValue is u8 DataType followed by raw bits at declared logical width: FLOAT64/INT64 8 bytes,
FLOAT32/INT32 4, BFLOAT16/FLOAT16 2, BOOL 1 (only 0/1). Floating decoders use raw-bit Model
factories (no conversion/NaN normalization); integral fields are two's-complement. LayerNorm's
epsilon must satisfy NN: finite positive, parameter type except FLOAT16 uses FLOAT32 arithmetic.
Thus integral/BOOL/FLOAT16 epsilon encodings are invalid here even though their scalar codec is
defined. Unknown dtype/activation/node/version values fail closed, including all recurrent IDs.
Recurrent adoption needs an explicit future versioned format decision, never a registry.

Canonical empty root/no-alias example is 21 bytes:
`53594e4445463031 00000001 01 00000000 00000000` (magic, version, Sequence, children, groups).
It is semantic identity, not evidence of an executable zero-node backend route.

Each alias group is positive i32 member count (at least 2), then members as positive i32 UTF-8
byte length plus exact strict UTF-8 path bytes. Paths must equal derived numeric dot-separated
state paths without a root prefix or leading-zero child index. No normalization is permitted.
Members follow NN state traversal order, groups ascend by first member; disjoint existing paths
must have identical kind/type/Shape/gradient schema. Unlisted paths are singleton identities;
equal bytes never imply sharing. Bound group/member totals by derived state count and remaining
bytes before allocation. Reject duplicate/reordered/unknown paths, malformed UTF-8, noncanonical
flags, count/depth/rank/size/overflow violations and trailing bytes before model publication.

#### Whole-model coherence and publication

Outer kind is exactly `whole-model.v1`, with `definition/schema` and every unchanged `model.v1`
entry prefixed `model/` (`model/schema`, `model/state/00000000`, ...); no other entries. Compute
configuration SHA-256 from exact canonical definition bytes, not caller input. Project the values
component using existing ArtifactEnvelope defensive copies, then use public ModelCheckpoint
capture/inspect/canonicalStateBytes/load only: no private ordinal parser or component temp files.
Independently validate complete ordered value descriptors/lengths against NN-derived parameter
descriptors (all gradient-eligible), configuration digest, exact expected input presence/bytes,
and byte equality for every declared alias group before reconstruction/value installation returns
anything. Do not trust an inspection result as a validation credential. On capture check the same
coherence before returning an envelope. Hashes bind configuration but do not prove authenticity.

Reconstruct only a private NN descriptor skeleton, then public target-load values; return one
independent heap-backed evaluation DeclarativeModel only after full validation and installation.
No RNG, throwaway weights, autograd, compile, native ownership, source mutation or weakened
publication. No Engine is required for load. Semantic empty/identity trees are allowed with
no Engine call for empty state, not a generic zero-node execution promise. Save reuses existing
ArtifactFile force/revalidate/one-atomic-move policy unchanged. Failure exposes no partial model.

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


<a id="extensionstraining"></a>

## `modules/training`

`modules/training` (unchanged Gradle project `:extensions:training` and Java packages) owns generic
live parameter admission, optimizer algorithms and training orchestration independently of NN.

Allowed:

- `Optimizer`
- `Sgd`
- `Adam`
- `AdamW`
- `ParameterGroup`
- `TrainingSession`
- `TrainingParameters` with immutable ordered `Entry(path, tensor)` membership and live-owner
  `validateUnchanged()`; Training snapshots and validates admission before seed/compile/prepare
- `TrainingStep`
- `TrainingParameterStorage` for an explicit pre-session copy into caller-owned shareable native host memory
- immutable in-memory optimizer/session state snapshots
- an explicit dependency on the public Engine facade for the session lifecycle
- fresh stored external values through `TrainingSession.runValues`, reusing public `Engine.runValues`
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

Training has exactly `api` project dependencies on `modules/model` and `modules/engine`, with no
NN dependency, import, NN-taking compatibility overload, or test-only back-edge. Its public
signatures name `TrainingParameters`, `Tensor`, `DataType`, `Shape`, Engine and Engine metadata.
NN consumes Training's live admission, explicit native copy and TrainingSession APIs. The public Engine edge is the explicit
Class C composition edge needed to reuse one
compile/Prepare/Runtime path; it does not transfer optimizer ownership into Engine or authorize
Training to inspect backend selection or representations. `train()` and `eval()` mode are module
forward-behavior concerns owned by `extensions/nn`; an optimizer neither selects nor changes that
mode.

Training owns backend-neutral optimizer mathematics over supported caller-owned host parameter
storage, not backend-specific optimizer execution. A later fused optimizer route belongs to
backend prepare and kernels and requires a separate architecture decision.

## `extensions/checkpoint`

`modules/checkpoint` retains Gradle identity `:extensions:checkpoint` via settings mapping.
Its artifact package owns a version-one, single-regular-file envelope for opaque
bytes only; that package assigns no model meaning to a payload. The
canonical big-endian file has a fixed magic/version header, SHA-256 manifest digest, sorted strict
ASCII names, explicit bounded lengths, and SHA-256 per-payload digests. Readers reject malformed,
noncanonical, unsupported, over-limit, truncated, or trailing data before returning a detached
immutable image. Callers supply bounds for file and manifest bytes, entry count, name bytes,
individual payload bytes, and aggregate payload bytes. Readers bound the manifest length before
allocating its array, then check aggregate length arithmetic before allocating payloads.
It has exactly `api(Model)` and `implementation(Compiler)` project dependencies. Compiler access
uses only GraphCompilationPort's cold capture/reconstruction boundary. No Compiler type appears
in public GraphCheckpoint signatures. No NN, Training, Engine, Runtime, Prepare, tuning or backend
dependency is permitted, including tests. NN alone consumes both generic Checkpoint and Engine
for owned Model composition and the relocated old persistence APIs; other inward modules do not depend on Checkpoint.

### Generic graph source and values

GraphCheckpoint is a final cold facade with only nested Source(inputs, outputs, leaves),
Leaf(name, role, tensor), Role { PARAMETER, BUFFER, CONSTANT }, and copy(Source),
save(Path, Source, ArtifactLimits), load(Path, ArtifactLimits), rebind(Source, List<Tensor>).
Save/load declare IOException. Records snapshot lists and null-check in declaration/list order;
Tensors are borrowed ordinary model objects, not another graph, executor or closeable owner.
Operations revalidate. Inputs are ordered unique provenance-free prototypes, possibly unused or
empty. Outputs are nonempty and distinct. Named provenance-free state is disjoint from inputs;
every reachable non-input leaf is named and every named identity is reachable. Names use strict
UTF-8, nonblank dot segments without normalization, duplicates or prefix collisions; registration
order is retained. Aliases mean exact Tensor identity and identical role, not equal bytes or
physical overlap. PARAMETER requires floating gradient eligibility. BUFFER and CONSTANT preserve
requiresGrad without inferring roles.

Capture uses Compiler captureSource exactly once and associates its first-encounter inputs by
exact Tensor identity, never by declared position. Existing immutable Model graph records normalize
file-local IDs and retain unused signature slots; no second graph representation or Tensor
reconstruction table belongs here. Compiler retains source admission and call-local reconstruction
ownership, unchanged: static FORWARD ADD/MUL/NEG/MATMUL with NoOperationAttrs and rank-two
PERMUTE[1,0], no optimization or layout closure. Preserve full descriptors, occurrences, sharing
and ordered roots; all seven dtypes where Model permits (no integral/BOOL NEG). Zero-node data
is not new Engine capability. Rebind checks metadata then replacement count/full descriptors,
retains named state and permits produced/repeated substitutions with opaque upstream provenance;
its immutable ordinary outputs may repeat, which ordinary compilation still rejects.

Copy/save read named identities only. Check static canonical geometry even for unresolved layouts;
present leaf layouts must equal zero-offset non-view contiguous geometry. Intermediate views are
metadata. Snapshot each hostStorage association once; admit every state's type, capacity, liveness
and thread access, including empty state, before BOOL checks or bulk copying. Read-only heap/native
segments are allowed; absent/device-only/dead/inaccessible/short state rejects. Caller excludes
mutation, rebinding and closure; no lock, hidden execution or ownership transfer. Read native-order
unaligned integer lanes at widths 8 (FLOAT64/INT64), 4 (FLOAT32/INT32), 2 (BFLOAT16/FLOAT16), 1
(BOOL). Preserve all raw bits including signed zero and NaN payload/sign; BOOL must be 0/1. Ignore
excess capacity and copy distinct overlapping Tensors independently. Use matching primitive heap
carriers and MemorySegmentStorage with TensorFactory.create retaining the original descriptor,
not layout-closing allocate. Copy creates storage-free prototypes and copies each state identity
once, then delegates reconstruction. Load adopts final heap state without a second ownership copy,
original graph/state or saved input storage. Labels/native owners are not persisted.

Copy/save order: metadata, checked geometry and (save) encoded bounds, all storage admission,
BOOL/raw copy, complete result/image, then ArtifactFile.save. Source/unsupported/bounds failures are
IllegalArgumentException; sizing overflow is ArithmeticException (retain Compiler wrapping),
absent/dead/inaccessible state is IllegalStateException. Rebind has no value gate. Load performs
ArtifactFile.read, complete wire/keys/descriptors/aliases/lengths/BOOL validation, descriptor-only
candidate leaves and Compiler SourceCapture preflight, host import, reconstructSource, publication.
Stored semantic/overflow errors become ArtifactFormatException with cause; I/O, identifier
exhaustion and VM failures propagate. Never publish partial candidates or mutate source bindings.

### Canonical graph-model version 1

ArtifactFile v1 kind is graph-model.v1: exactly schema plus state/00000000… per unique named
identity. Big-endian signed i32/i64 and unsigned u8, nonnegative counts, flags exactly 0/1,
no padding, extensions or trailing bytes. Fixed IDs never use enum ordinals or class names.

| Schema order | Exact fields |
|---|---|
| Header | Eight ASCII bytes SYNGRF01, i32 version 1. |
| Inputs | i32 I, I descriptors in declared order, including unused slots; ValueIds 0… I−1, no payload. |
| State | i32 S, then descriptor, u8 role (1 PARAMETER, 2 BUFFER, 3 CONSTANT), i64 byte length per first named identity; ValueIds I… I+S−1, payload ordinal is state index. |
| Registrations | i32 A, then positive i32 strict UTF-8 name length, exact bytes, i32 state index in registration order; all states referenced and first new indices are 0,1,…; repeats are aliases. |
| Nodes | i32 N, then u8 phase=1, u8 op (1 ADD, 2 MUL, 3 NEG, 4 MATMUL, 5 PERMUTE), fixed-arity i32 preceding input ValueIds (two for ADD/MUL/MATMUL, one otherwise); PERMUTE adds i32 2, i32 1, i32 0; output descriptor. NodeId n, output ValueId I+S+n. |
| Outputs | Positive i32 O and O distinct ordered i32 ValueIds. |
| Descriptor | u8 dtype (1 FLOAT64, 2 FLOAT32, 3 BFLOAT16, 4 INT32, 5 INT64, 6 BOOL, 7 FLOAT16), u8 requiresGrad, i32 rank, rank nonnegative i64 extents, u8 layout-present; when present, i64 element offset, u8 view, rank nonnegative i64 strides; derive kind/span via LayoutDescriptor.of. |

Reject forward/dangling/cyclic references, dead nodes, unreachable state, noncanonical aliases,
unknown IDs/phases/attrs, keys, lengths and BOOL. Compiler owns family/descriptor validation;
normalized recapture must match stored topology in capture order, retaining unused inputs.
Payloads are exact row-major bits, shape-count × width; no TensorIds, batches or runtime state.
Save/load require explicit ArtifactLimits, with no new defaults or restriction of old NN trees.
Bound schema by maxPayloadBytes (≤Integer.MAX_VALUE−8), all counts/ranks/strings by remaining bytes
before allocation, S≤100000000, S+1≤maxEntries and checked I+S+N≤Integer.MAX_VALUE. Traversal is
iterative with no backend rank-16 bound. For schema D and unique state bytes B: manifest
M=72+58*S, aggregate=D+B, file=48+M+D+B, entries=1+S, kind/key ceiling≥14. Check all six limits
and individual payloads with checked sums before encoding or bulk copies. These bound encoded
size, not peak heap. Copy/rebind have no implicit file ceiling; checked collection/geometry/JVM
limits apply and copy preflights each unique payload≤Integer.MAX_VALUE−8. Complete save image
validation precedes unchanged ArtifactFile force/revalidate/one-atomic-move publication and its
cleanup/failure semantics; this is not authentication or portable power-loss durability.

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
  ordered generic live parameter path/Tensor/storage snapshot
  -> one scalar-objective first-order Engine compile
  -> one Engine PreparedExecution preparation
  -> TrainingSession

run:
  validate the complete live source/input/storage view before execution
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
  adapter may persist only supported, validated participant schemas. NN's live-parameter adapter
  edge grants no recovery coordination or optimizer ownership.
- Before opening a session, a caller may explicitly ask Training to copy a generic live source's
  current parameters into a caller-owned shareable Arena. A zero-byte Arena shareability probe
  retains its existing failure precedence. Training validates the complete ordered parameter set
  and source storage before destination parameter allocation or replacing any Tensor association,
  copies exact represented bytes into fresh, aligned, exact-capacity writable native segments,
  and installs the associations in path order without replacing Parameter or Tensor identities.
  Source storage may be heap or native and read-only; duplicate Tensor identities and overlapping
  source parameter regions are rejected. Ordinary installation failure restores earlier exact
  associations. The caller owns both source and destination lifetimes, must keep the destination
  Arena alive through any session, and must give the copy exclusive source/storage use. Membership
  is revalidated before installation; successful copy preserves the same live source's membership.
  Copying
  while a session is open is forbidden. This does not change the native-only capture domain of
  `TrainingSession.open` or introduce a backend transfer.
- A `TrainingSession` borrows exactly one caller-owned Engine and one caller-owned live
  `TrainingParameters` source. It
  owns exactly one Engine `PreparedExecution`, one optimizer-state set, and its counters. It opens
  that owner with `Engine.prepare` and executes exact-ID inputs with `Engine.run`, or fresh stored
  values with `Engine.runValues`. Both share one private optimizer transaction. Close releases
  the prepared owner before the private cotangent-seed arena, preserving failed-open rollback and
  update transactions.
- Session construction snapshots nonempty, unique, nonblank paths and exact Tensors in caller
  encounter order, never sorted; null sources, lists, entries, paths or Tensors reject. The source
  validates live membership at capture and every complete-view validation site, including before
  run/state/restore and precommit for accumulation, update and restore. NN adapters additionally
  retain wrappers. Training owns descriptor, exact storage identity/schema/liveness/accessibility/
  capacity and finiteness checks. Every Tensor identity and non-overlapping storage occurrence
  of the captured parameters remains fixed until close; callers grant exclusive source/parameter
  storage use and must not replace membership, bindings or storage associations while the session
  is open.
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
  Parameters and the private seed bind internally by captured Tensor identity; callers fill every
  external bindable position according to the selected admission mode below. Incompatible
  descriptors or absent, dead, inaccessible, insufficient, nonempty overlapping or parameter-aliasing
  storage fail before optimizer mutation.
- `run` requires exact external TensorIds in arbitrary list order, retaining its existing checks
  and diagnostics. `runValues` requires exactly the external positions returned by `inputs()` in
  relative final compiled order, excluding parameters and the private seed, including targets and
  every other external bindable leaf. Fresh IDs are accepted only with exact full static
  descriptors (dtype, shape, layout presence/geometry and requiresGrad). Values are stored values,
  not expressions to execute. Training inserts fixed parameters/seed using its captured binding
  plan and delegates checked static storage geometry to public Engine positional admission; it
  does not recapture, recompile, prepare, copy or rebind prototypes. Old `run` retains its
  layout-span-or-zero capacity checks. Caller external storage remains borrowed and exclusively
  usable through the synchronous call and internal result closure, never retained for a later run.
  Both paths reject nonempty external/external, external/parameter and seed overlap in compiled
  order, including the same nonempty value in two positions. Engine's repeated-value permission
  does not relax Training's rule; zero-byte storage remains exempt from overlap checks.
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
