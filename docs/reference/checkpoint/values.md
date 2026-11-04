# Values-only [checkpoint](../../glossary.md#term-checkpoint) contracts

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/graph/GraphCheckpoint.java)

Values-only recovery replaces a compatible [target](../../glossary.md#term-target)'s named [parameter](../../glossary.md#term-parameter) and [buffer](../../glossary.md#term-buffer) [bindings](../../glossary.md#term-binding); it
does not reconstruct arbitrary [model](../../glossary.md#term-model)-building code. An **[alias](../../glossary.md#aliasing-and-arena) group** contains paths referring
to the same exact [Tensor](../../glossary.md#tensor), not merely equal numbers. A configuration [digest](../../glossary.md#term-digest) identifies the caller's
chosen model configuration, while the [schema](../../glossary.md#term-schema) describes each stored state entry.

## Detached bytes versus live bindings

The application lends a quiescent Module and Engine to ModelCheckpoint capture/save. NN exports
current parameter/buffer references; Checkpoint calls Engine to obtain admitted values and
copies their canonical bytes into an envelope. The returned envelope owns no live Module,
Engine or Arena resource. Load performs the reverse value installation into the application's
compatible target, but creates fresh heap Tensors instead of retaining source storage. Existing
expressions keep their earlier Tensor dependencies, so do not replace bindings underneath an
open Training session. [The examples](values-examples.md) demonstrate source closure and exact
raw-bit comparison; [recovery coordination](../training/recovery.md) explains why a private tree
is safer than mutating the active session's tree.

<a id="values-only-recovery"></a>
## Values-only recovery

The remaining sections describe the unchanged advanced values-only [API](../../glossary.md#term-application-programming-interface-api), including its caller
digests, fully bound [target](../../glossary.md#term-target), failure types and target-defined [aliases](../../glossary.md#aliasing-and-arena).

```text
quiescent Module --one NN snapshot + public Engine.compute--> detached model.v1/v2 envelope
envelope --independent validation per call--> descriptors / canonical path bytes / NN load
model.v1/v2 file --ArtifactFile integrity checks + model validation--> descriptors / NN load
```

The arrows show who owns each stage: [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) exports/installs state and its `nn.checkpoint` package
validates old [model](../../glossary.md#term-model) [metadata](../../glossary.md#term-metadata) and bytes; [Engine](../../glossary.md#term-engine) materializes [capture](../../glossary.md#term-capture-meanings)/save-time values. Loading does not use an Engine.

<a id="core-contracts"></a>
## Core contracts

The file-free operations use existing public `ArtifactEnvelope`, `ArtifactLimits`, and
`ModelCheckpointInspection` types:

- `capture(Module, Engine, configDigest, inputDigest, ArtifactLimits)` returns a fully detached,
  semantically validated `model.v1` or [FP8](../../glossary.md#term-fp8) `model.v2` envelope without changing source [bindings](../../glossary.md#term-binding).
- `inspect(ArtifactEnvelope, ArtifactLimits)` returns ordered descriptors and defensive digests,
  not value bytes or a trusted-validation credential.
- `canonicalStateBytes(ArtifactEnvelope, ArtifactLimits)` returns every parameter and buffer's
  raw logical bytes under its exact Unicode NN path in metadata order, not lexical path order.
- `load(ArtifactEnvelope, Module, expectedConfigDigest, expectedInputDigest, ArtifactLimits)`
  validates and reconstructs the complete candidate before one NN strict installation attempt.

Every [operation](../../glossary.md#operation) independently runs [model](../../glossary.md#term-model) decode and all six [artifact](../../glossary.md#term-artifact) ceilings. A standalone
envelope need not have passed a file reader; equivalent [manifest](../../glossary.md#term-manifest) and complete-file limits still
apply before [schema](../../glossary.md#term-schema) copying. It has no encoded file [checksum](../../glossary.md#term-checksum) to verify. `ArtifactFile.read` owns
file [framing](../../glossary.md#term-framing) and [integrity](../../glossary.md#term-integrity) checks; constructing an [opaque](../../glossary.md#term-opaque) envelope alone proves no model validity.

`save(Path, Module, Engine, configDigest, inputDigest, ArtifactLimits)` snapshots the complete
state once through public [Engine](../../glossary.md#term-engine) [materialization](../../glossary.md#materialization) and publishes the smallest sufficient v1/v2 envelope.
Fully bound native and matching primitive-array [heap](../../glossary.md#term-heap) leaves, including previously loaded state,
can be saved directly on the validated [CPU](../../glossary.md#term-cpu) route; callers need no expression wrappers.
`inspect(Path, ArtifactLimits)`
validates the whole file, including checksums, the model schema, every [payload](../../glossary.md#artifact-envelope-and-atomic-publication), and [canonical](../../glossary.md#term-canonical-form)
[`BOOL`](../../glossary.md#data-type--datatype) bytes (`0` or `1`). It returns ordered typed path, role, [data type](../../glossary.md#data-type--datatype), [Shape](../../glossary.md#shape), and byte-length
[descriptors](../../glossary.md#term-descriptor) plus defensive [digest](../../glossary.md#term-digest) copies. The inspection owns no file or live [module](../../glossary.md#term-module)/Engine data.
`load(Path, Module, expectedConfigDigest, expectedInputDigest, ArtifactLimits)` rereads and
revalidates its path; it does not trust a prior inspection of a [mutable](../../glossary.md#term-mutable) file.

Both caller digests are [SHA-256](../../glossary.md#term-sha-256) byte arrays of exactly 32 bytes. The configuration [fingerprint](../../glossary.md#term-fingerprint)
must include the application's versioned model configuration and intended sharing [topology](../../glossary.md#term-topology).
The optional input-artifact fingerprint is `null` when absent; presence and all bytes must match
exactly at load. [Checkpoint](../../glossary.md#term-checkpoint) does not interpret either fingerprint or serialize model-defining Java
code. The first version stores one value per state path and cannot prove source [Tensor](../../glossary.md#tensor) [identity](../../glossary.md#term-identity-meanings).

The [target](../../glossary.md#term-target) must already be fully bound. `Module.stateDictionary()` is called before installation
to reject [lazy](../../glossary.md#term-lazy-evaluation)/unbound groups, even though [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)'s in-memory loader can initialize those groups.
The recreated target's exact Tensor-identity groups define [aliases](../../glossary.md#aliasing-and-arena). Every path in one group must
have byte-identical stored values; one fresh dense-[contiguous](../../glossary.md#term-contiguous) [host](../../glossary.md#term-host) Tensor is then installed for
that group. A group containing a [parameter](../../glossary.md#term-parameter) is [gradient](../../glossary.md#term-gradient)-eligible; [buffer](../../glossary.md#term-buffer)-only groups are not.
All nine current data types use compatible primitive-array heap carriers and native [byte order](../../glossary.md#text-encoding-and-file-boundaries).
FP8 uses `byte[]`, [BFLOAT16](../../glossary.md#term-bfloat16)/[FLOAT16](../../glossary.md#term-float16) use `short[]`, and floating values restore by [raw bits](../../glossary.md#storage-mapping-contiguity-and-injectivity),
without conversion or [NaN](../../glossary.md#term-nan) [normalization](../../glossary.md#term-normalization).
[Rank-zero](../../glossary.md#axis-rank-scalar-vector-and-matrix) shapes contain one [element](../../glossary.md#term-element); a zero [extent](../../glossary.md#term-extent) produces an empty payload.

<a id="inputs-outputs-and-ownership"></a>
## Inputs, outputs, and [ownership](../../glossary.md#term-ownership-meanings)

The caller owns the [module](../../glossary.md#term-module), [Engine](../../glossary.md#term-engine), file path, and trusted local parent directory. Save and load
require [quiescence](../../glossary.md#term-quiescence) of [bindings](../../glossary.md#term-binding) and relevant storage for their whole calls, as does file-free [capture](../../glossary.md#term-capture-meanings).
Source storage must remain live and accessible until capture/save returns. Envelopes, inspections,
and [canonical](../../glossary.md#term-canonical-form) [views](../../glossary.md#term-view-meanings) retain no Module, [Tensor](../../glossary.md#tensor), Engine, or source [Arena](../../glossary.md#term-arena) resource handles and remain
readable after source Arena and Engine closure. Empty state still requires a non-null Engine,
but makes no Engine call, so that Engine may already be closed. The returned inspection
is [immutable](../../glossary.md#term-immutable) and [detached](../../glossary.md#detached-model-results); its digests are copies and its state list cannot be changed. Loaded
[heap](../../glossary.md#term-heap) storage has ordinary garbage-collected [lifetime](../../glossary.md#term-lifetime), with no [Checkpoint](../../glossary.md#term-checkpoint)-owned arena or close
[operation](../../glossary.md#operation). [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)'s wrappers remain, but their Tensor bindings change after a successful load.

The canonical map's structure and [payload](../../glossary.md#artifact-envelope-and-atomic-publication) bytes are immutable. Each path in each call gets a
fresh read-only `ByteBuffer` with position zero, exact payload capacity/limit, and `BIG_ENDIAN`
order. Its position, limit, and byte-order setting are caller-[mutable](../../glossary.md#term-mutable); this is not deep cursor
immutability or a [thread-safe](../../glossary.md#term-thread-safe) shared cursor. Moving one view cannot affect another call or path,
including an [alias](../../glossary.md#aliasing-and-arena). Use a duplicate or fresh map when a consumer may have moved a cursor.

Both load overloads require a fully bound [target](../../glossary.md#term-target) with no open [Training session](../../glossary.md#training-session--trainingsession) over that tree.
That session captured fixed bindings; replacing them would invalidate it. The caller enforces
this restriction and exclusion—Checkpoint does not inspect Training. For recovery, use a distinct,
unpublished replacement tree rather than mutate an active session's tree.

<a id="lifecycle-and-invariants"></a>
## [Lifecycle](../../glossary.md#lifecycle) and invariants

Before [Engine](../../glossary.md#term-engine) work or [payload](../../glossary.md#artifact-envelope-and-atomic-publication)-sized allocation, [capture](../../glossary.md#term-capture-meanings) preflights caller fingerprints, original
ordered paths/kinds/types/[Shapes](../../glossary.md#shape), checked byte lengths, and entry, payload, aggregate, [manifest](../../glossary.md#term-manifest),
and final-file limits. Only afterward does it deduplicate original [Tensor](../../glossary.md#tensor) [identities](../../glossary.md#term-identity-meanings) in
first-encounter order. Internally, each unique [provenance](../../glossary.md#provenance)-free [leaf](../../glossary.md#term-leaf) becomes a raw-preserving
[`CONTIGUOUS`](../../glossary.md#term-contiguous) request; an existing expression stays an expression. These requests normalize
save-time outputs, not [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [bindings](../../glossary.md#term-binding) or [schema](../../glossary.md#term-schema) [descriptors](../../glossary.md#term-descriptor). One public `Engine.compute` call
materializes nonempty state and checks returned type, Shape, and byte length. Empty state makes
no Engine call, even with a closed Engine. Each path still receives a separate payload.

This raw movement preserves signed zeros, [subnormals](../../glossary.md#term-subnormal), [infinities](../../glossary.md#nan-infinity-and-signed-zero), and [NaN payload](../../glossary.md#nan-infinity-and-signed-zero)/sign bits;
it uses neither [identity](../../glossary.md#term-identity-meanings) arithmetic nor casts. The original seven-[carrier](../../glossary.md#term-storage-carrier) test coverage remains
historical evidence; [FP8](../../glossary.md#term-fp8) additionally uses exact raw byte [lanes](../../glossary.md#term-lane). This includes native and
matching [heap](../../glossary.md#term-heap) carriers, [scalar](../../glossary.md#term-scalar) and zero-[extent](../../glossary.md#term-extent) state, and mixed leaves/expressions. It is not a
generic promise that Engine prepares provenance-free outputs, or a [backend capability](../../glossary.md#backend-capability) expansion.

[Checkpoint](../../glossary.md#term-checkpoint) validates all file bytes, digests, paths, kinds, data types, Shapes, [alias](../../glossary.md#aliasing-and-arena) values, and
reconstructed candidates before one NN whole-tree installation attempt. Ordinary [validation](../../glossary.md#term-validation) or
reconstruction failure changes no [target](../../glossary.md#term-target) binding. There is no cross-[owner](../../glossary.md#term-owner) [transaction](../../glossary.md#term-transaction) or
concurrent-reader [snapshot](../../glossary.md#term-snapshot) guarantee. Save reuses the [artifact](../../glossary.md#term-artifact) [API](../../glossary.md#term-application-programming-interface-api)'s sibling temporary-file,
force, reopen, validation, and one [atomic](../../glossary.md#resource-transaction-and-rollback)-move path. Successful return gives atomic live
visibility of a complete prior or new file, not portable power-[loss](../../glossary.md#loss) [durability](../../glossary.md#term-durability) of the rename.
A failed move can have provider-dependent target state; inspect the target before retrying.


---

[Previous topic](models.md) · [Next topic](values-examples.md)
