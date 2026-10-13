# Task 0002: Bounded artifact envelope and atomic publication

## Status

Complete. Implemented from corrected policy base `7261d11d`, after the Ready frontier
verified at `6482eac7`. Checkpoint 0001 is Complete; 0003–0006 remain Draft and unauthorized.

## Change class

Class C — first persistent wire format, public artifact API, resource bounds, and filesystem
publication/failure semantics.

## Goal

Implement one reusable, bounded, versioned file envelope for opaque payloads. Later model and
optional training adapters supply semantic payloads; this task owns neither NN state meaning nor
Engine materialization.

## Selected v1 policy

- One regular file, fixed magic and format version 1; canonical big-endian metadata; strict ASCII
  logical kind/entry keys, sorted entries, explicit lengths, no padding or compression. Payloads
  are opaque bytes. A fixed header checks the manifest with an unkeyed SHA-256 digest; the manifest
  records each payload length and SHA-256 digest. Reject duplicate/noncanonical names, `.`/`..`, empty
  segments, unknown versions, malformed fields, truncation, and trailing bytes.
- Public caller-supplied limits bound file and manifest bytes, entry count, key length, each
  payload, and aggregate payload bytes. The reader bounds encoded lengths before allocation and
  checks aggregate arithmetic before payload allocation. An individual payload cannot exceed the
  JVM byte-array ceiling. The returned image is detached and immutable; construction
  defensively captures caller bytes and access exposes read-only views.
- Save into a newly created sibling temporary file in an existing trusted local directory. Write
  and `FileChannel.force(true)` the temporary file; close, reopen, and fully validate the image;
  then use one `ATOMIC_MOVE` to the requested target. No copy or non-atomic rename fallback. A
  pre-existing regular target may be replaced; a symlink or directory target is rejected before
  work. The Java filesystem provider may still refuse atomic replacement, in which case save
  fails without a weaker fallback. External concurrent mutation of the target/directory is out
  of scope; do not promise exclusive no-replace semantics.
- Guarantee atomic *live visibility* only when the move succeeds: a reader may see the prior
  complete file (or absence if no target existed) or the new complete file, never a partially
  written final file. The file's contents are forced before publication on a local filesystem,
  but Java NIO does not portably guarantee that the rename's directory entry survives a power
  loss. Do not claim full crash durability, try to
  manufacture it with `FileChannel.force` on a directory, or add a platform-specific fallback.
  On move failure, never deliberately delete the target; clean only owned temporary files when
  safe, and tell the caller to inspect the target if the provider's result is uncertain.
- This is an initial storage contract, not encryption, authentication against an adversary who
  can rewrite both data and hashes, model schema, or training resume. The path is caller-owned;
  only a trusted directory and quiescent target are supported.

The replace policy is deliberate: [Java `Files.move`](https://docs.oracle.com/en/java/javase/26/docs/api/java.base/java/nio/file/Files.html#move(java.nio.file.Path,java.nio.file.Path,java.nio.file.CopyOption...))
with `ATOMIC_MOVE` ignores other move options and does not portably guarantee no-replace behavior
when the target exists. [Java `FileChannel.force`](https://docs.oracle.com/en/java/javase/26/docs/api/java.base/java/nio/channels/FileChannel.html#force(boolean))
only guarantees flushing changes made through that channel, not a separate rename. The format is kept
small and uncompressed so exact bytes, bounds, and corruption refusal are reviewable before
semantic adapters are added.

## Scope

- `io.github.pho001.synaptik.checkpoint.artifact.ArtifactEnvelope` with nested immutable entry
  value, `ArtifactLimits`, `ArtifactFile` (public read/save), and `ArtifactFormatException`.
- Package-private `ArtifactWire` for deterministic encoding/decoding and `ArtifactPublication`
  for the temporary-file, force, validate, and atomic-move lifecycle. Additional private nested
  helpers only when they keep these responsibilities focused.
- Tests of canonical roundtrip/determinism, zero-length payloads, ordering, bounds and overflow,
  every corruption class, exact-file-length checks, existing-target replacement, temporary
  cleanup, and failure after/without atomic publication. Keep a narrow test seam only if needed
  to simulate an unsupported move or force operation reliably.
- Update the root ownership summary and the one extensions scoped contract, ADR 0032, focused
  explanatory/API documentation, and affected glossary/Javadocs.

## Non-goals

- Tensor/Shape/DataType or optimizer metadata, ModelCheckpoint, TrainingCheckpoint, NN loading,
  Engine execution/materialization, tokenizer parsing, compression, encryption, generic object
  serialization, streaming payloads larger than a JVM byte array, or migrations from v1.
- A recovery service, registry, implicit backend discovery, or best-effort non-atomic mode.

## Contracts

- [`ARCHITECTURE.md` — Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence), [Core invariants](../../../../../ARCHITECTURE.md#core-invariants), [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing), [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules), [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements): preserve the six-contract model and Checkpoint's downstream ownership.
- [`extensions-training.md` — Scope](../../../../architecture/contracts/extensions-training.md#scope), [`extensions/checkpoint`](../../../../architecture/contracts/extensions-training.md#extensionscheckpoint): amend only this owning scoped contract for the new envelope and publication lifecycle.
- [`runtime-prepare-engine.md` — `modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine): no Engine or backend access in the artifact layer.

## Dependencies and integration

- Depends on: Checkpoint 0001 (Complete); selected v1 policy above.
- Conflicts with: Checkpoint 0003–0006 and concurrent edits to the artifact format/Checkpoint contract.
- Parallel group: None.
- Common base revision: N/A (serial work from `6482eac7`).
- Integration order: 0002 before 0003; no parallel integration.
- Integration validation: `./gradlew :extensions:checkpoint:check :extensions:checkpoint:javadoc`.
- Shared-document integration owner: 0002 implementation owner; no concurrent writers.

## Files and symbols

- `extensions/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/artifact/` — the
  six named types and exact v1 file/ownership contract above.
- `extensions/checkpoint/src/test/java/io/github/pho001/synaptik/checkpoint/artifact/` — focused
  format, bounds, corruption, and publication tests.
- `ARCHITECTURE.md`, `docs/architecture/contracts/extensions-training.md`,
  `docs/architecture/{module-boundaries,lifecycle}.md`,
  `docs/design/decisions/0032-checkpoint-artifact-envelope.md`, `docs/glossary.md` — current
  contract and explanation.
- This brief, Checkpoint master plan, roadmap — status and evidence.

## Acceptance criteria

- Two equal envelopes produce byte-identical files; roundtrip preserves kind, names, and exact
  payload bytes. Unsupported versions, any malformed/truncated/trailing encoding, duplicate or
  unsafe names, checksum mismatch, and configured limit violations fail before returning an image.
- No reader allocation is driven by an unchecked file length; no partially validated image is
  returned. A single payload and aggregate are bounded with checked arithmetic.
- Publication never exposes a partial final file: validation/force failure before rename leaves
  an existing target unchanged and cleans its own temp; success shows the full new image.
  Unsupported atomic move never triggers a non-atomic fallback. A failed move has the documented
  provider-dependent target state; no post-rename operation or false crash-durability claim is
  introduced.
- No NN/Engine/Training concepts or new Gradle edges appear in the artifact package. Javadocs,
  ADR, contract, examples, and tests describe the exact supported guarantee and limitations.

## Validation

Worker validation:

```bash
./gradlew :extensions:checkpoint:check :extensions:checkpoint:javadoc
```

Integration/repository validation: reuse the focused gate after the independent review if no
executable change is made. No repository-wide suite is required here: no dependency/build edge or
cross-module behavior changes; Checkpoint 0004 owns the end-to-end recovery checkpoint.

## Documentation and review impact

- Public API, durable format, and failure modes require a clean Class C implementation context
  and independent targeted architecture/documentation review. Read the general, architecture,
  ADR, API, and glossary documentation profiles before writing their document types.
- Every new public/contract method and type needs detailed Javadoc for bounds, ownership,
  nullability, result, mutation, and failure conditions.

## Result

The opaque v1 artifact API, bounded deterministic wire reader/writer, streamed temporary-file
validation, file force, and single atomic-move publication are implemented with focused corruption,
limits, immutability, replacement, and failure tests. Root/scoped contracts, explanations, ADR
0032, API reference, glossary, and affected Javadocs are current. The implementation-owner gate
`./gradlew :extensions:checkpoint:check :extensions:checkpoint:javadoc` passed (15 tests,
no skips or failures). The existing architecture dependency test has only its stale
empty-project wording updated; `./gradlew :testing:architecture-tests:test --tests
io.github.pho001.synaptik.testing.architecture.CheckpointDependencyContractTest` passed
(one test, no skips or failures).

Independent Class C architecture/documentation review completed. It found post-move temporary
cleanup, which was corrected and verified by the 15-test gate, then signed off in a targeted
second pass. That pass also confirmed the amended Javadocs and failure claims. Markdown links
and anchors (478 local targets), Javadoc generation, and `git diff --check` passed. No model or
training schema and no Gradle edge were added.
