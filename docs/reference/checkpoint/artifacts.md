<a id="checkpoint-artifact-file-api"></a>
# [Opaque](../../glossary.md#term-opaque) [checkpoint](../../glossary.md#term-checkpoint) [artifact](../../glossary.md#term-artifact) files

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/graph/GraphCheckpoint.java)

An **opaque [payload](../../glossary.md#artifact-envelope-and-atomic-publication)** is a byte sequence whose meaning this [API](../../glossary.md#term-application-programming-interface-api) deliberately does not know.
The **envelope** groups named payloads; the **[manifest](../../glossary.md#term-manifest)** describes their names, sizes and digests.
A higher-level **[schema](../../glossary.md#term-schema)** says how those bytes represent a [graph](../../glossary.md#graph) or [model](../../glossary.md#term-model). File [integrity](../../glossary.md#term-integrity) checks
can detect bytes that do not match their recorded digests, but do not prove who created the file.

## What crosses the file boundary?

The application calls `ArtifactEnvelope.Entry` with a byte array; the entry copies those bytes,
so later source-array mutation cannot change the envelope. The application then calls
`ArtifactFile.save(path, envelope, limits)`; ArtifactFile owns the temporary file it creates, not
the application's directory or final path. A later `read` returns a validated detached image
with no open-file lease. The payload remains opaque: neither call knows whether it encodes a
Tensor, a layer definition or something else. [GraphCheckpoint](graph.md) and
[NN values-only recovery](values.md) supply those higher-level meanings. Successful atomic
visibility is not a promise that a rename survives power loss.

<a id="purpose-and-mental-model"></a>
## Purpose and [mental model](../../glossary.md#term-mental-model)

The [artifact](../../glossary.md#term-artifact) [API](../../glossary.md#term-application-programming-interface-api) stores named [opaque](../../glossary.md#term-opaque) byte payloads, not a [model](../../glossary.md#term-model) or [training session](../../glossary.md#training-session--trainingsession). An
`ArtifactEnvelope` is an [immutable](../../glossary.md#term-immutable) in-memory image; `ArtifactFile` validates or publishes its
version-one file form. The [extensions contract](../../architecture/contracts/extensions-training.md#extensionscheckpoint)
owns the boundary, and [ADR 0032](../../design/decisions/0032-checkpoint-artifact-envelope.md)
explains the format and [publication](../../glossary.md#term-publication) choice.

[NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) retains a public [model checkpoint API](models.md), now in `nn.checkpoint`, that maps state to
opaque entries via public [Engine](../../glossary.md#term-engine) [materialization](../../glossary.md#materialization). It is separate from this opaque artifact API:
the artifact package still assigns no meaning to payloads. Its old model [schema](../../glossary.md#term-schema) is described by the
[owning NN contract](../../architecture/contracts/extensions-training.md#retained-nn-checkpoint-apis) and
[ADR 0033](../../design/decisions/0033-model-candidate-schema.md).

<a id="core-contracts"></a>
## Core contracts

`ArtifactEnvelope` has one [ASCII](../../glossary.md#term-ascii) kind and zero or more entries. Entry keys are unique and sorted
canonically on construction. A kind is one nonempty segment; entry keys can use `/` between
nonempty segments. Each segment contains only ASCII letters, digits, `_`, `-`, or `.` and cannot
equal `.` or `..`. Payloads are byte-exact and have no interpretation in this [API](../../glossary.md#term-application-programming-interface-api). The wire file
contains a fixed magic/version header, [SHA-256](../../glossary.md#term-sha-256) [manifest](../../glossary.md#term-manifest) [digest](../../glossary.md#term-digest), length-prefixed manifest, and
[contiguous](../../glossary.md#term-contiguous) payloads with individual SHA-256 digests. Unknown versions, invalid names or order,
bad lengths or digests, truncation, and trailing bytes are rejected.

`ArtifactLimits` is mandatory on both read and save. Its six ceilings cover total file bytes,
manifest bytes, number of entries, ASCII kind/key bytes, bytes in one [payload](../../glossary.md#artifact-envelope-and-atomic-publication), and aggregate
payload bytes. Payloads use Java byte arrays, so one cannot exceed `Integer.MAX_VALUE - 8` bytes.
The reader bounds the encoded manifest length before allocating its array and checks aggregate
arithmetic before payload allocation; callers should choose limits appropriate to their
application memory budget.

<a id="inputs-outputs-and-ownership"></a>
## Inputs, outputs, and [ownership](../../glossary.md#term-ownership-meanings)

Entry construction copies a caller's byte array. An entry exposes a fresh read-only `ByteBuffer`
[view](../../glossary.md#term-view-meanings) with independent position on each call; moving a view does not change the entry. The image's
entry list is [immutable](../../glossary.md#term-immutable). `ArtifactFile.read` returns only after the complete image passes
[validation](../../glossary.md#term-validation). A file path and directory remain caller-owned; the [API](../../glossary.md#term-application-programming-interface-api) does not lock out other writers.

<a id="lifecycle-and-invariants"></a>
## [Lifecycle](../../glossary.md#lifecycle) and invariants

Save needs an existing trusted local parent directory and a [quiescent](../../glossary.md#term-quiescent) [target](../../glossary.md#term-target). It rejects an
existing symlink or directory target before writing; a regular target may be replaced. It writes
a new sibling temporary file, forces its contents through the writing file [channel](../../glossary.md#term-channel), closes it,
reopens it, and verifies both complete format and exact bytes against the source. It then attempts
one [atomic move](../../glossary.md#term-atomic-move). There is no non-[atomic](../../glossary.md#resource-transaction-and-rollback) fallback. Successful return gives atomic live visibility:
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
[payload](../../glossary.md#artifact-envelope-and-atomic-publication) under [`weights`](../../glossary.md#term-weight):

```java
// The entry copies its source bytes; save owns only its staging file, not this directory.
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

The final result is one validated `example.v1` file with a two-byte [opaque](../../glossary.md#term-opaque) payload. It proves
neither [Tensor](../../glossary.md#tensor) compatibility nor power-loss [persistence](../../glossary.md#term-persistence) of the rename. If the provider rejects
the [atomic move](../../glossary.md#term-atomic-move), `save` throws `IOException`; the caller inspects the [target](../../glossary.md#term-target) rather than assuming
the old file is still present.

<a id="failures-and-limitations"></a>
## Failures and limitations

`ArtifactFormatException` reports malformed, corrupt, unsupported, or over-limit images. Other
`IOException`s report access, force, [validation](../../glossary.md#term-validation) I/O, and move failures. This is not encryption,
adversarial [authentication](../../glossary.md#term-authentication), a remote-filesystem guarantee, or training
[resume](../../glossary.md#schema-serialization-and-checkpoint). There is no cross-process no-replace or concurrent-[target](../../glossary.md#term-target)-mutation promise.

<a id="related-api-and-architecture"></a>
## Related [API](../../glossary.md#term-application-programming-interface-api) and architecture

- [Checkpoint module boundary](../../architecture/module-boundaries.md#extensionscheckpoint)
- [Public model checkpoint API](models.md)
- [Checkpoint lifecycle explanation](../../architecture/lifecycle.md#public-training-lifecycle)

The [artifact](../../glossary.md#term-artifact) package and `:extensions:checkpoint` project ID are unchanged; sources now live in
`modules/checkpoint`. Its separate [generic graph API](graph.md) stores cold [Tensor](../../glossary.md#tensor)
sources and [host](../../glossary.md#term-host) values without [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) or [Engine](../../glossary.md#term-engine). Existing [opaque](../../glossary.md#term-opaque) file bytes/[publication](../../glossary.md#term-publication) are unchanged.
The graph codec selects the smallest sufficient `graph-model.v1` through `graph-model.v7`.
V6 adds maximum/average Pool3d (`SYNGRF06`, opcodes 34/35); FP8 anywhere in the supported
source selects v7. These graph-schema extensions do not change opaque ArtifactFile v1 framing,
bounds or publication. Old sources retain their smallest sufficient byte-identical format.
See [v6 Pool3d metadata](graph-formats.md#complete-pool3d-geometry-in-v6) and
[v7 FP8 metadata](graph-formats.md#fp8-graph-persistence); neither stores optimizer/session state.
