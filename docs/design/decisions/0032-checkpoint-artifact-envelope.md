# ADR 0032: Bounded opaque artifact and atomic live publication

## Status

Accepted — 2026-10-13, after independent Class C review. The [architecture root](../../../ARCHITECTURE.md#module-ownership-routing)
and [owning extensions contract](../../architecture/contracts/extensions-training.md#extensionscheckpoint)
are authoritative; this record explains the choice.

## Context

Checkpoint has a model-only module boundary but no file format. Model and optional training
adapters will eventually need a common way to carry opaque byte payloads without moving Tensor
meaning, backend access, or restore policy into the artifact layer. The first format must reject
corruption and unsafe resource demands before a caller can use an image. Java NIO can force bytes
written through a file channel and request an atomic move, but it does not portably prove that a
rename survives power loss or that a failed move left the target unchanged.

## Decision drivers

- Make version-one bytes deterministic and reviewable without compression or semantic codecs.
- Bound untrusted encoded lengths before size-driven allocation and reject noncanonical or corrupt
  files completely.
- Preserve prior-complete-or-new live visibility (including prior absence) on successful publication
  without a weaker fallback.
- State the actual Java NIO failure and durability limits.

## Options considered

| Option | Advantage | Rejection or cost |
|---|---|---|
| Serialize Java objects directly | Little codec work | Unbounded or unstable semantics and unsafe object reconstruction. |
| Stream arbitrary compressed payloads | Smaller files | Decompression and streaming limits complicate the first reviewable contract. |
| Single uncompressed bounded envelope with atomic move | Explicit bytes, limits, and publication boundary | Payloads remain limited to JVM byte arrays; semantic adapters come later. |
| Copy or ordinary rename when atomic move is unavailable | More filesystem coverage | Can expose a partial target and silently weaken the selected guarantee. |

The bounded envelope is selected; the fallback is rejected.

## Decision

Version one uses one regular file: eight-byte `SYNART01` magic, big-endian 32-bit version `1`,
big-endian 32-bit manifest length, 32-byte SHA-256 manifest digest, then the manifest and
contiguous payloads. The manifest encodes a length-prefixed ASCII kind, entry count, and each
sorted entry's length-prefixed ASCII key, signed 64-bit payload length, and 32-byte SHA-256 payload
digest. Names contain only ASCII letters, digits, `_`, `-`, `.`, and `/` between entry-key
segments; the kind is one segment. Empty segments and `.` or `..` segments are forbidden. No
padding, compression, or trailing bytes are permitted. Callers provide file, manifest, count,
name, single-payload, and aggregate-payload limits. Decoding bounds the encoded manifest length
before allocating its array, checks aggregate length arithmetic before payload allocation, and
returns only a fully checked detached immutable image.

Saving creates a sibling temporary file in an existing trusted local directory, writes and
`FileChannel.force(true)`s it, closes and reopens it, and streams a full validation against the
source image before one `ATOMIC_MOVE`. An existing regular target may be replaced; a symlink or
directory target is refused before work. Success means atomic live visibility of the prior complete
file (or absence) or the new complete file. There is no directory force or cross-platform
power-loss guarantee. On failed move, the target's state is provider-dependent and the caller must
inspect it; the implementation never deliberately deletes that target. It attempts to clean its
owned temporary file after a failure before the move attempt, or after an explicit
unsupported-atomic-move result.
Other move failures leave the temporary path untouched because the result is uncertain; the file
may remain. A successful move is followed by no filesystem work. External concurrent mutation
is outside the supported trusted-directory, quiescent-target use.

## Rationale

The fixed manifest permits exact file-length checks before payload allocation. Per-payload
digests detect accidental corruption, while canonical order makes equal images byte-identical.
They do not authenticate against an adversary able to rewrite both bytes and digests. One atomic
move avoids deliberately publishing a partial target. `FileChannel.force` applies to the
temporary file's channel writes, not portably to the later directory-entry change.

## Consequences

### Positive

Later adapters can supply semantic payloads without adding NN, Engine, or Training concepts to
the artifact codec. Readers receive no partial image; around a successful save, ordinary live
readers see the prior complete file (or absence) or the new complete version.

### Negative and risks

One payload cannot exceed the conservative JVM byte-array ceiling. Large caller-selected bounds
can still consume substantial memory. A provider can refuse atomic replacement, and a failed
move may have changed the target. Power-loss recovery and remote-filesystem guarantees are not
claimed. A semantic model schema and restore transaction remain future work.

### Migration, testing, and follow-up

There is no previous checkpoint file to migrate. Focused tests cover deterministic bytes,
canonical rejection, limits, checksums, exact lengths, temporary cleanup, atomic-move refusal,
and uncertain provider results. Checkpoint 0003–0004 must define model meaning and strict restore
separately; optional training continuation remains separately gated.

## Related documentation

- [Artifact API](../../api/checkpoint-artifact-api.md)
- [Checkpoint module boundary](../../architecture/module-boundaries.md#extensionscheckpoint)
- [Checkpoint 0002 brief](../../planning/extensions/checkpoint/tasks/0002-bounded-artifact-envelope.md)
- [Subsequent internal model-candidate schema decision](0033-model-candidate-schema.md) — adds
  model meaning outside this unchanged opaque envelope; public model save/load is still pending.
