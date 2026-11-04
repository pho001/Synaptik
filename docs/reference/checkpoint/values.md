# Values-only checkpoint contracts

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/graph/GraphCheckpoint.java)

Values-only recovery replaces a compatible target's named parameter and buffer bindings; it
does not reconstruct arbitrary model-building code. An **alias group** contains paths referring
to the same exact Tensor, not merely equal numbers. A configuration digest identifies the caller's
chosen model configuration, while the schema describes each stored state entry.

<a id="values-only-recovery"></a>
## Values-only recovery

The remaining sections describe the unchanged advanced values-only API, including its caller
digests, fully bound target, failure types and target-defined aliases.

```text
quiescent Module --one NN snapshot + public Engine.compute--> detached model.v1/v2 envelope
envelope --independent validation per call--> descriptors / canonical path bytes / NN load
model.v1/v2 file --ArtifactFile integrity checks + model validation--> descriptors / NN load
```

The arrows show who owns each stage: NN exports/installs state and its `nn.checkpoint` package
validates old model metadata and bytes; Engine materializes capture/save-time values. Loading does not use an Engine.

<a id="core-contracts"></a>
## Core contracts

The file-free operations use existing public `ArtifactEnvelope`, `ArtifactLimits`, and
`ModelCheckpointInspection` types:

- `capture(Module, Engine, configDigest, inputDigest, ArtifactLimits)` returns a fully detached,
  semantically validated `model.v1` or FP8 `model.v2` envelope without changing source bindings.
- `inspect(ArtifactEnvelope, ArtifactLimits)` returns ordered descriptors and defensive digests,
  not value bytes or a trusted-validation credential.
- `canonicalStateBytes(ArtifactEnvelope, ArtifactLimits)` returns every parameter and buffer's
  raw logical bytes under its exact Unicode NN path in metadata order, not lexical path order.
- `load(ArtifactEnvelope, Module, expectedConfigDigest, expectedInputDigest, ArtifactLimits)`
  validates and reconstructs the complete candidate before one NN strict installation attempt.

Every operation independently runs model decode and all six artifact ceilings. A standalone
envelope need not have passed a file reader; equivalent manifest and complete-file limits still
apply before schema copying. It has no encoded file checksum to verify. `ArtifactFile.read` owns
file framing and integrity checks; constructing an opaque envelope alone proves no model validity.

`save(Path, Module, Engine, configDigest, inputDigest, ArtifactLimits)` snapshots the complete
state once through public Engine materialization and publishes the smallest sufficient v1/v2 envelope.
Fully bound native and matching primitive-array heap leaves, including previously loaded state,
can be saved directly on the validated CPU route; callers need no expression wrappers.
`inspect(Path, ArtifactLimits)`
validates the whole file, including checksums, the model schema, every payload, and canonical
`BOOL` bytes (`0` or `1`). It returns ordered typed path, role, data type, Shape, and byte-length
descriptors plus defensive digest copies. The inspection owns no file or live module/Engine data.
`load(Path, Module, expectedConfigDigest, expectedInputDigest, ArtifactLimits)` rereads and
revalidates its path; it does not trust a prior inspection of a mutable file.

Both caller digests are SHA-256 byte arrays of exactly 32 bytes. The configuration fingerprint
must include the application's versioned model configuration and intended sharing topology.
The optional input-artifact fingerprint is `null` when absent; presence and all bytes must match
exactly at load. Checkpoint does not interpret either fingerprint or serialize model-defining Java
code. The first version stores one value per state path and cannot prove source Tensor identity.

The target must already be fully bound. `Module.stateDictionary()` is called before installation
to reject lazy/unbound groups, even though NN's in-memory loader can initialize those groups.
The recreated target's exact Tensor-identity groups define aliases. Every path in one group must
have byte-identical stored values; one fresh dense-contiguous host Tensor is then installed for
that group. A group containing a parameter is gradient-eligible; buffer-only groups are not.
All nine current data types use compatible primitive-array heap carriers and native byte order.
FP8 uses `byte[]`, BFLOAT16/FLOAT16 use `short[]`, and floating values restore by raw bits,
without conversion or NaN normalization.
Rank-zero shapes contain one element; a zero extent produces an empty payload.

<a id="inputs-outputs-and-ownership"></a>
## Inputs, outputs, and ownership

The caller owns the module, Engine, file path, and trusted local parent directory. Save and load
require quiescence of bindings and relevant storage for their whole calls, as does file-free capture.
Source storage must remain live and accessible until capture/save returns. Envelopes, inspections,
and canonical views retain no Module, Tensor, Engine, or source Arena resource handles and remain
readable after source Arena and Engine closure. Empty state still requires a non-null Engine,
but makes no Engine call, so that Engine may already be closed. The returned inspection
is immutable and detached; its digests are copies and its state list cannot be changed. Loaded
heap storage has ordinary garbage-collected lifetime, with no Checkpoint-owned arena or close
operation. NN's wrappers remain, but their Tensor bindings change after a successful load.

The canonical map's structure and payload bytes are immutable. Each path in each call gets a
fresh read-only `ByteBuffer` with position zero, exact payload capacity/limit, and `BIG_ENDIAN`
order. Its position, limit, and byte-order setting are caller-mutable; this is not deep cursor
immutability or a thread-safe shared cursor. Moving one view cannot affect another call or path,
including an alias. Use a duplicate or fresh map when a consumer may have moved a cursor.

Both load overloads require a fully bound target with no open Training session over that tree.
That session captured fixed bindings; replacing them would invalidate it. The caller enforces
this restriction and exclusion—Checkpoint does not inspect Training. For recovery, use a distinct,
unpublished replacement tree rather than mutate an active session's tree.

<a id="lifecycle-and-invariants"></a>
## Lifecycle and invariants

Before Engine work or payload-sized allocation, capture preflights caller fingerprints, original
ordered paths/kinds/types/Shapes, checked byte lengths, and entry, payload, aggregate, manifest,
and final-file limits. Only afterward does it deduplicate original Tensor identities in
first-encounter order. Internally, each unique provenance-free leaf becomes a raw-preserving
`CONTIGUOUS` request; an existing expression stays an expression. These requests normalize
save-time outputs, not NN bindings or schema descriptors. One public `Engine.compute` call
materializes nonempty state and checks returned type, Shape, and byte length. Empty state makes
no Engine call, even with a closed Engine. Each path still receives a separate payload.

This raw movement preserves signed zeros, subnormals, infinities, and NaN payload/sign bits;
it uses neither identity arithmetic nor casts. The original seven-carrier test coverage remains
historical evidence; FP8 additionally uses exact raw byte lanes. This includes native and
matching heap carriers, scalar and zero-extent state, and mixed leaves/expressions. It is not a
generic promise that Engine prepares provenance-free outputs, or a backend capability expansion.

Checkpoint validates all file bytes, digests, paths, kinds, data types, Shapes, alias values, and
reconstructed candidates before one NN whole-tree installation attempt. Ordinary validation or
reconstruction failure changes no target binding. There is no cross-owner transaction or
concurrent-reader snapshot guarantee. Save reuses the artifact API's sibling temporary-file,
force, reopen, validation, and one atomic-move path. Successful return gives atomic live
visibility of a complete prior or new file, not portable power-loss durability of the rename.
A failed move can have provider-dependent target state; inspect the target before retrying.


---

[Previous topic](models.md) · [Next topic](values-examples.md)
