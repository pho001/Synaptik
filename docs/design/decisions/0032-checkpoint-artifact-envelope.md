# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0032: Bounded [opaque](../../glossary.md#term-opaque) [artifact](../../glossary.md#term-artifact) and [atomic live publication](../../glossary.md#artifact-envelope-and-atomic-publication)

## Status

Accepted — 2026-10-13, after independent Class C review. The [architecture root](../../../ARCHITECTURE.md#module-ownership-routing)
and [owning extensions contract](../../architecture/contracts/extensions-training.md#extensionscheckpoint)
are authoritative; this record explains the choice.

## Context

[Checkpoint](../../glossary.md#term-checkpoint) has a [model](../../glossary.md#term-model)-only [module](../../glossary.md#term-module) boundary but no file format. Model and optional training
adapters will eventually need a common way to [carry](../../glossary.md#term-carry) [opaque](../../glossary.md#term-opaque) byte payloads without moving [Tensor](../../glossary.md#tensor)
meaning, [backend](../../glossary.md#backend) access, or restore policy into the [artifact](../../glossary.md#term-artifact) layer. The first format must reject
corruption and unsafe resource demands before a caller can use an image. Java NIO can force bytes
written through a file [channel](../../glossary.md#term-channel) and request an [atomic move](../../glossary.md#term-atomic-move), but it does not portably prove that a
rename survives power [loss](../../glossary.md#loss) or that a failed move left the [target](../../glossary.md#term-target) unchanged.

## Decision drivers

- Make version-one bytes deterministic and reviewable without compression or semantic codecs.
- Bound untrusted encoded lengths before size-driven allocation and reject noncanonical or corrupt
  files completely.
- Preserve prior-complete-or-new live visibility (including prior absence) on successful publication
  without a weaker fallback.<span data-glossary-navigation="true"> ([publication glossary](../../glossary.md#term-publication))</span>
- State the actual Java NIO failure and [durability](../../glossary.md#term-durability) limits.

## Options considered

| Option | Advantage | Rejection or cost |
|---|---|---|
| Serialize Java objects directly | Little [codec](../../glossary.md#term-codec) work | Unbounded or unstable semantics and unsafe object reconstruction. |
| Stream arbitrary compressed payloads | Smaller files | Decompression and streaming limits complicate the first reviewable contract. |
| Single uncompressed bounded envelope with [atomic move](../../glossary.md#term-atomic-move) | Explicit bytes, limits, and [publication](../../glossary.md#term-publication) boundary | Payloads remain limited to [JVM](../../glossary.md#term-java-virtual-machine-jvm) byte arrays; semantic adapters come later. |
| Copy or ordinary rename when atomic move is unavailable | More filesystem coverage | Can expose a partial [target](../../glossary.md#term-target) and silently weaken the selected guarantee. |

The bounded envelope is selected; the fallback is rejected.

## Decision

Version one uses one regular file: eight-byte `SYNART01` magic, [big-endian](../../glossary.md#text-encoding-and-file-boundaries) 32-bit version `1`,
big-endian 32-bit [manifest](../../glossary.md#term-manifest) length, 32-byte [SHA-256](../../glossary.md#term-sha-256) manifest [digest](../../glossary.md#term-digest), then the manifest and
[contiguous](../../glossary.md#term-contiguous) payloads. The manifest encodes a length-prefixed [ASCII](../../glossary.md#term-ascii) kind, entry count, and each
sorted entry's length-prefixed ASCII key, signed 64-bit [payload](../../glossary.md#artifact-envelope-and-atomic-publication) length, and 32-byte SHA-256 payload
digest. Names contain only ASCII letters, digits, `_`, `-`, `.`, and `/` between entry-key
segments; the kind is one segment. Empty segments and `.` or `..` segments are forbidden. No
[padding](../../glossary.md#padding), compression, or trailing bytes are permitted. Callers provide file, manifest, count,
name, single-payload, and aggregate-payload limits. Decoding bounds the encoded manifest length
before allocating its array, checks aggregate length arithmetic before payload allocation, and
returns only a fully checked [detached](../../glossary.md#detached-model-results) [immutable](../../glossary.md#term-immutable) image.

Saving creates a sibling temporary file in an existing trusted local directory, writes and
`FileChannel.force(true)`s it, closes and reopens it, and streams a full [validation](../../glossary.md#term-validation) against the
source image before one `ATOMIC_MOVE`. An existing regular [target](../../glossary.md#term-target) may be replaced; a symlink or
directory target is refused before work. Success means [atomic](../../glossary.md#resource-transaction-and-rollback) live visibility of the prior complete
file (or absence) or the new complete file. There is no directory force or cross-platform
power-[loss](../../glossary.md#loss) guarantee. On failed move, the target's state is provider-dependent and the caller must
inspect it; the implementation never deliberately deletes that target. It attempts to clean its
owned temporary file after a failure before the move attempt, or after an explicit
unsupported-atomic-move result.
Other move failures leave the temporary path untouched because the result is uncertain; the file
may remain. A successful move is followed by no filesystem work. External concurrent mutation
is outside the supported trusted-directory, [quiescent](../../glossary.md#term-quiescent)-target use.

## Rationale

The fixed [manifest](../../glossary.md#term-manifest) permits exact file-length checks before [payload](../../glossary.md#artifact-envelope-and-atomic-publication) allocation. Per-payload
digests detect accidental corruption, while [canonical](../../glossary.md#term-canonical-form) order makes equal images byte-identical.
They do not authenticate against an adversary able to rewrite both bytes and digests. One [atomic
move](../../glossary.md#term-atomic-move) avoids deliberately publishing a partial [target](../../glossary.md#term-target). `FileChannel.force` applies to the
temporary file's [channel](../../glossary.md#term-channel) writes, not portably to the later directory-entry change.

## Consequences

### Positive

Later adapters can supply semantic payloads without adding [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context), [Engine](../../glossary.md#term-engine), or Training concepts to
the [artifact](../../glossary.md#term-artifact) [codec](../../glossary.md#term-codec). Readers receive no partial image; around a successful save, ordinary live
readers see the prior complete file (or absence) or the new complete version.

### Negative and risks

One [payload](../../glossary.md#artifact-envelope-and-atomic-publication) cannot exceed the conservative [JVM](../../glossary.md#term-java-virtual-machine-jvm) byte-array ceiling. Large caller-selected bounds
can still consume substantial memory. A provider can refuse [atomic](../../glossary.md#resource-transaction-and-rollback) replacement, and a failed
move may have changed the [target](../../glossary.md#term-target). Power-[loss](../../glossary.md#loss) recovery and remote-filesystem guarantees are not
claimed. A semantic [model](../../glossary.md#term-model) [schema](../../glossary.md#term-schema) and restore [transaction](../../glossary.md#term-transaction) remain future work.

### Migration, testing, and follow-up

There is no previous [checkpoint](../../glossary.md#term-checkpoint) file to migrate. Focused tests cover deterministic bytes,
[canonical](../../glossary.md#term-canonical-form) rejection, limits, checksums, exact lengths, temporary cleanup, [atomic](../../glossary.md#resource-transaction-and-rollback)-move refusal,
and uncertain provider results. Checkpoint 0003–0004 must define [model](../../glossary.md#term-model) meaning and strict restore
separately; optional training continuation remains separately gated.

## Related documentation

- [Artifact API](../../api/checkpoint-artifact-api.md)<span data-glossary-navigation="true"> ([Artifact glossary](../../glossary.md#term-artifact); [API glossary](../../glossary.md#term-application-programming-interface-api))</span>
- [Checkpoint module boundary](../../architecture/module-boundaries.md#extensionscheckpoint)<span data-glossary-navigation="true"> ([Checkpoint glossary](../../glossary.md#term-checkpoint); [module glossary](../../glossary.md#term-module))</span>
- [Checkpoint 0002 brief](../../planning/extensions/checkpoint/tasks/0002-bounded-artifact-envelope.md)
- [Subsequent internal model-candidate schema decision](0033-model-candidate-schema.md)<span data-glossary-navigation="true"> ([model glossary](../../glossary.md#term-model); [schema glossary](../../glossary.md#term-schema))</span> — adds
  model meaning outside this unchanged opaque envelope; public model save/load is still pending.<span data-glossary-navigation="true"> ([opaque glossary](../../glossary.md#term-opaque))</span>
