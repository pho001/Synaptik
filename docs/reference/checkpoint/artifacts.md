<a id="checkpoint-artifact-file-api"></a>
# Opaque checkpoint artifact files

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/graph/GraphCheckpoint.java)

An **opaque payload** is a byte sequence whose meaning this API deliberately does not know.
The **envelope** groups named payloads; the **manifest** describes their names, sizes and digests.
A higher-level **schema** says how those bytes represent a graph or model. File integrity checks
can detect bytes that do not match their recorded digests, but do not prove who created the file.

<a id="purpose-and-mental-model"></a>
## Purpose and mental model

The artifact API stores named opaque byte payloads, not a model or training session. An
`ArtifactEnvelope` is an immutable in-memory image; `ArtifactFile` validates or publishes its
version-one file form. The [extensions contract](../../architecture/contracts/extensions-training.md#extensionscheckpoint)
owns the boundary, and [ADR 0032](../../design/decisions/0032-checkpoint-artifact-envelope.md)
explains the format and publication choice.

NN retains a public [model checkpoint API](models.md), now in `nn.checkpoint`, that maps state to
opaque entries via public Engine materialization. It is separate from this opaque artifact API:
the artifact package still assigns no meaning to payloads. Its old model schema is described by the
[owning NN contract](../../architecture/contracts/extensions-training.md#retained-nn-checkpoint-apis) and
[ADR 0033](../../design/decisions/0033-model-candidate-schema.md).

<a id="core-contracts"></a>
## Core contracts

`ArtifactEnvelope` has one ASCII kind and zero or more entries. Entry keys are unique and sorted
canonically on construction. A kind is one nonempty segment; entry keys can use `/` between
nonempty segments. Each segment contains only ASCII letters, digits, `_`, `-`, or `.` and cannot
equal `.` or `..`. Payloads are byte-exact and have no interpretation in this API. The wire file
contains a fixed magic/version header, SHA-256 manifest digest, length-prefixed manifest, and
contiguous payloads with individual SHA-256 digests. Unknown versions, invalid names or order,
bad lengths or digests, truncation, and trailing bytes are rejected.

`ArtifactLimits` is mandatory on both read and save. Its six ceilings cover total file bytes,
manifest bytes, number of entries, ASCII kind/key bytes, bytes in one payload, and aggregate
payload bytes. Payloads use Java byte arrays, so one cannot exceed `Integer.MAX_VALUE - 8` bytes.
The reader bounds the encoded manifest length before allocating its array and checks aggregate
arithmetic before payload allocation; callers should choose limits appropriate to their
application memory budget.

<a id="inputs-outputs-and-ownership"></a>
## Inputs, outputs, and ownership

Entry construction copies a caller's byte array. An entry exposes a fresh read-only `ByteBuffer`
view with independent position on each call; moving a view does not change the entry. The image's
entry list is immutable. `ArtifactFile.read` returns only after the complete image passes
validation. A file path and directory remain caller-owned; the API does not lock out other writers.

<a id="lifecycle-and-invariants"></a>
## Lifecycle and invariants

Save needs an existing trusted local parent directory and a quiescent target. It rejects an
existing symlink or directory target before writing; a regular target may be replaced. It writes
a new sibling temporary file, forces its contents through the writing file channel, closes it,
reopens it, and verifies both complete format and exact bytes against the source. It then attempts
one atomic move. There is no non-atomic fallback. Successful return gives atomic live visibility:
a concurrent reader may see the prior complete file, absence if the target was initially missing,
or the new complete file, not an intentionally partial final file.
Java NIO does not portably guarantee that the rename survives power loss. A failed move can leave
the target in a provider-dependent state; inspect it before retrying or making recovery decisions.
Save never deliberately deletes the target and performs no filesystem work after a successful
move. It attempts to clean its owned temporary file on failures before the move and after an explicit
unsupported-atomic-move result. Another move failure leaves that temporary path untouched; the
file may remain and need caller-directed inspection or later cleanup.

<a id="complete-example"></a>
## Complete example

The example creates a local temporary directory as its trusted parent. The input is one two-byte
payload under `weights`:

```java
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactEnvelope;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactFile;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

ArtifactLimits limits = new ArtifactLimits(4096, 2048, 4, 64, 1024, 2048);
ArtifactEnvelope image = new ArtifactEnvelope("example.v1", List.of(
        new ArtifactEnvelope.Entry("weights", new byte[] {1, 2})));
Path trustedDirectory = Files.createTempDirectory("artifact-example");
Path target = trustedDirectory.resolve("checkpoint.art");
ArtifactFile.save(target, image, limits);
ArtifactEnvelope checked = ArtifactFile.read(target, limits);
int payloadBytes = checked.entries().getFirst().size(); // 2
```

The final result is one validated `example.v1` file with a two-byte opaque payload. It proves
neither Tensor compatibility nor power-loss persistence of the rename. If the provider rejects
the atomic move, `save` throws `IOException`; the caller inspects the target rather than assuming
the old file is still present.

<a id="failures-and-limitations"></a>
## Failures and limitations

`ArtifactFormatException` reports malformed, corrupt, unsupported, or over-limit images. Other
`IOException`s report access, force, validation I/O, and move failures. This is not encryption,
adversarial authentication, a remote-filesystem guarantee, or training
resume. There is no cross-process no-replace or concurrent-target-mutation promise.

<a id="related-api-and-architecture"></a>
## Related API and architecture

- [Checkpoint module boundary](../../architecture/module-boundaries.md#extensionscheckpoint)
- [Public model checkpoint API](models.md)
- [Checkpoint lifecycle explanation](../../architecture/lifecycle.md#public-training-lifecycle)

The artifact package and `:extensions:checkpoint` project ID are unchanged; sources now live in
`modules/checkpoint`. Its separate [generic graph API](graph.md) stores cold Tensor
sources and host values without NN or Engine. Existing opaque file bytes/publication are unchanged.
The graph codec now selects `graph-model.v1` through `graph-model.v6`; v6 adds maximum/average
Pool3d (`SYNGRF06`, opcodes 34/35), including otherwise-qualified inline bodies. This is a
graph-schema extension, not a change to opaque ArtifactFile v1 framing, bounds or publication.
Unchanged graph sources keep their smallest byte-identical v1–v5 format. See the
[graph API's v6 record](graph-formats.md#complete-pool3d-geometry-in-v6) for geometry and
metadata-preflight details. No optimizer or session state is stored.
