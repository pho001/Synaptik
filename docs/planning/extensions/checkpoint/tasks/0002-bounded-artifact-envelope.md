# Task 0002: Bounded [artifact envelope and atomic publication](../../../../glossary.md#artifact-envelope-and-atomic-publication)

> **Editorial reading context — not a status update.** The application supplies named byte
> arrays to an envelope; the envelope copies those arrays, but does not know whether they
> encode weights, text or anything else. A manifest inventories payloads and their integrity
> digests; a model schema is a different layer that explains their meaning. File publication
> makes a complete file visible at one move, not necessarily durable through power loss.
> The application still controls the target directory and concurrent access. Read the current
> [byte ownership](../../../../reference/checkpoint/artifacts.md#inputs-outputs-and-ownership) and
> [publication lifecycle](../../../../reference/checkpoint/artifacts.md#lifecycle-and-invariants)
> for those two independent boundaries.

## Status

Complete. Implemented from corrected policy base `7261d11d`, after the [Ready](../../../../glossary.md#term-ready) frontier
verified at `6482eac7`. [Checkpoint](../../../../glossary.md#term-checkpoint) 0001 is Complete; 0003–0006 remain Draft and unauthorized.

## Change class

Class C — first persistent [wire format](../../../../glossary.md#term-wire-format), public [artifact](../../../../glossary.md#term-artifact) [API](../../../../glossary.md#term-application-programming-interface-api), resource bounds, and filesystem
[publication](../../../../glossary.md#term-publication)/failure semantics.

## Goal

Implement one reusable, bounded, versioned file envelope for [opaque](../../../../glossary.md#term-opaque) payloads. Later [model](../../../../glossary.md#term-model) and
optional training adapters supply semantic payloads; this task owns neither [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) state meaning nor
[Engine](../../../../glossary.md#term-engine) [materialization](../../../../glossary.md#materialization).

## Selected v1 policy

- One regular file, fixed magic and format version 1; canonical big-endian metadata; strict ASCII
  logical kind/entry keys, sorted entries, explicit lengths, no padding or compression. Payloads
  are opaque bytes. A fixed header checks the manifest with an unkeyed SHA-256 digest; the manifest
  records each payload length and SHA-256 digest. Reject duplicate/noncanonical names, <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [big-endian glossary](../../../../glossary.md#text-encoding-and-file-boundaries); [metadata glossary](../../../../glossary.md#term-metadata); [ASCII glossary](../../../../glossary.md#term-ascii); [padding glossary](../../../../glossary.md#padding); [opaque glossary](../../../../glossary.md#term-opaque); [manifest glossary](../../../../glossary.md#term-manifest); [SHA-256 glossary](../../../../glossary.md#term-sha-256); [digest glossary](../../../../glossary.md#term-digest); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>`.`/`..`, empty
  segments, unknown versions, malformed fields, truncation, and trailing bytes.<span data-glossary-navigation="true"> ([truncation glossary](../../../../glossary.md#term-truncation-toward-zero))</span>
- Public caller-supplied limits bound file and manifest bytes, entry count, key length, each
  payload, and aggregate payload bytes. The reader bounds encoded lengths before allocation and
  checks aggregate arithmetic before payload allocation. An individual payload cannot exceed the
  JVM byte-array ceiling. The returned image is detached and immutable; construction
  defensively captures caller bytes and access exposes read-only views.<span data-glossary-navigation="true"> ([captures glossary](../../../../glossary.md#term-capture-meanings))</span><span data-glossary-navigation="true"> ([JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [detached glossary](../../../../glossary.md#detached-model-results); [immutable glossary](../../../../glossary.md#term-immutable); [views glossary](../../../../glossary.md#term-view-meanings))</span>
- Save into a newly created sibling temporary file in an existing trusted local directory. Write
  and `FileChannel.force(true)` the temporary file; close, reopen, and fully validate the image;
  then use one `ATOMIC_MOVE` to the requested target. No copy or non-atomic rename fallback. A
  pre-existing regular target may be replaced; a symlink or directory target is rejected before
  work. The Java filesystem provider may still refuse atomic replacement, in which case save
  fails without a weaker fallback. External concurrent mutation of the target/directory is out
  of scope; do not promise exclusive no-replace semantics.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- Guarantee [atomic](../../../../glossary.md#resource-transaction-and-rollback) *live visibility* only when the move succeeds: a reader may see the prior
  complete file (or absence if no target existed) or the new complete file, never a partially
  written final file. The file's contents are forced before publication on a local filesystem,
  but Java NIO does not portably guarantee that the rename's directory entry survives a power
  loss. Do not claim full crash durability, try to
  manufacture it with <span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [loss glossary](../../../../glossary.md#loss); [crash durability glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>`FileChannel.force` on a directory, or add a platform-specific fallback.
  On move failure, never deliberately delete the target; clean only owned temporary files when
  safe, and tell the caller to inspect the target if the provider's result is uncertain.
- This is an initial storage contract, not encryption, authentication against an adversary who
  can rewrite both data and hashes, model schema, or training resume. The path is caller-owned;
  only a trusted directory and quiescent target are supported.<span data-glossary-navigation="true"> ([hashes glossary](../../../../glossary.md#term-hash))</span><span data-glossary-navigation="true"> ([authentication glossary](../../../../glossary.md#term-authentication); [model glossary](../../../../glossary.md#term-model); [schema glossary](../../../../glossary.md#term-schema); [resume glossary](../../../../glossary.md#schema-serialization-and-checkpoint); [quiescent glossary](../../../../glossary.md#term-quiescent))</span>

The replace policy is deliberate: [Java `Files.move`](https://docs.oracle.com/en/java/javase/26/docs/api/java.base/java/nio/file/Files.html#move(java.nio.file.Path,java.nio.file.Path,java.nio.file.CopyOption...))
with `ATOMIC_MOVE` ignores other move options and does not portably guarantee no-replace behavior
when the [target](../../../../glossary.md#term-target) exists. [Java `FileChannel.force`](https://docs.oracle.com/en/java/javase/26/docs/api/java.base/java/nio/channels/FileChannel.html#force(boolean))
only guarantees flushing changes made through that [channel](../../../../glossary.md#term-channel), not a separate rename. The format is kept
small and uncompressed so exact bytes, bounds, and corruption refusal are reviewable before
semantic adapters are added.

## [Scope](../../../../glossary.md#term-scope)

- `io.github.pho001.synaptik.checkpoint.artifact.ArtifactEnvelope` with nested immutable entry
  value, <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>`ArtifactLimits`, `ArtifactFile` (public read/save), and `ArtifactFormatException`.
- Package-private `ArtifactWire` for deterministic encoding/decoding and `ArtifactPublication`
  for the temporary-file, force, validate, and atomic-move lifecycle. Additional private nested
  helpers only when they keep these responsibilities focused.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Tests of canonical roundtrip/determinism, zero-length payloads, ordering, bounds and overflow,
  every corruption class, exact-file-length checks, existing-target replacement, temporary
  cleanup, and failure after/without atomic publication. Keep a narrow test seam only if needed
  to simulate an unsupported move or force operation reliably.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [overflow glossary](../../../../glossary.md#term-overflow); [target glossary](../../../../glossary.md#term-target); [atomic publication glossary](../../../../glossary.md#term-atomic-publication); [operation glossary](../../../../glossary.md#operation))</span>
- Update the root ownership summary and the one extensions scoped contract, ADR 0032, focused
  explanatory/API documentation, and affected glossary/Javadocs.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Non-goals

- Tensor/Shape/DataType or optimizer metadata, ModelCheckpoint, TrainingCheckpoint, NN loading,
  Engine execution/materialization, tokenizer parsing, compression, encryption, generic object
  serialization, streaming payloads larger than a JVM byte array, or migrations from v1.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Shape glossary](../../../../glossary.md#shape); [DataType glossary](../../../../glossary.md#data-type--datatype); [optimizer glossary](../../../../glossary.md#term-optimizer); [metadata glossary](../../../../glossary.md#term-metadata); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Engine glossary](../../../../glossary.md#term-engine); [materialization glossary](../../../../glossary.md#materialization); [tokenizer glossary](../../../../glossary.md#frozen-whitespace-tokenizer); [serialization glossary](../../../../glossary.md#term-serialization); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm))</span>
- A recovery service, [registry](../../../../glossary.md#term-registry), implicit [backend](../../../../glossary.md#backend) discovery, or best-effort non-[atomic](../../../../glossary.md#resource-transaction-and-rollback) mode.

## Contracts

- [`ARCHITECTURE.md` — Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence), [Core invariants](../../../../../ARCHITECTURE.md#core-invariants), [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>, [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>, [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements): preserve the six-contract [model](../../../../glossary.md#term-model) and [Checkpoint](../../../../glossary.md#term-checkpoint)'s downstream [ownership](../../../../glossary.md#term-ownership-meanings).
- [`extensions-training.md` — Scope](../../../../architecture/contracts/extensions-training.md#scope)<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope))</span>, [`extensions/checkpoint`](../../../../architecture/contracts/extensions-training.md#extensionscheckpoint): amend only this owning scoped contract for the new envelope and [publication](../../../../glossary.md#term-publication) [lifecycle](../../../../glossary.md#lifecycle).
- [`runtime-prepare-engine.md` — `modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine): no [Engine](../../../../glossary.md#term-engine) or [backend](../../../../glossary.md#backend) access in the [artifact](../../../../glossary.md#term-artifact) layer.

## Dependencies and integration

- Depends on: [Checkpoint](../../../../glossary.md#term-checkpoint) 0001 (Complete); selected v1 policy above.
- Conflicts with: Checkpoint 0003–0006 and concurrent edits to the [artifact](../../../../glossary.md#term-artifact) format/Checkpoint contract.
- Parallel group: None.
- Common base revision: N/A (serial work from `6482eac7`).
- Integration order: 0002 before 0003; no parallel integration.
- Integration [validation](../../../../glossary.md#term-validation): `./gradlew :extensions:checkpoint:check :extensions:checkpoint:javadoc`.
- Shared-document integration [owner](../../../../glossary.md#term-owner): 0002 implementation owner; no concurrent writers.

## Files and symbols

- `extensions/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/artifact/` — the
  six named types and exact v1 file/ownership contract above.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- `extensions/checkpoint/src/test/java/io/github/pho001/synaptik/checkpoint/artifact/` — focused
  format, bounds, corruption, and publication tests.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>
- `ARCHITECTURE.md`, `docs/architecture/contracts/extensions-training.md`,
  `docs/architecture/{module-boundaries,lifecycle}.md`,
  `docs/design/decisions/0032-checkpoint-artifact-envelope.md`, `docs/glossary.md` — current
  contract and explanation.
- This brief, [Checkpoint](../../../../glossary.md#term-checkpoint) master plan, roadmap — status and evidence.

## Acceptance criteria

- Two equal envelopes produce byte-identical files; roundtrip preserves kind, names, and exact
  payload bytes. Unsupported versions, any malformed/truncated/trailing encoding, duplicate or
  unsafe names, checksum mismatch, and configured limit violations fail before returning an image.<span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [checksum glossary](../../../../glossary.md#term-checksum))</span>
- No reader allocation is driven by an unchecked file length; no partially validated image is
  returned. A single payload and aggregate are bounded with checked arithmetic.
- Publication never exposes a partial final file: validation/force failure before rename leaves
  an existing target unchanged and cleans its own temp; success shows the full new image.
  Unsupported atomic move never triggers a non-atomic fallback. A failed move has the documented
  provider-dependent target state; no post-rename operation or false crash-durability claim is
  introduced.<span data-glossary-navigation="true"> ([Publication glossary](../../../../glossary.md#term-publication); [validation glossary](../../../../glossary.md#term-validation); [target glossary](../../../../glossary.md#term-target); [atomic move glossary](../../../../glossary.md#term-atomic-move); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [operation glossary](../../../../glossary.md#operation); [durability glossary](../../../../glossary.md#term-durability))</span>
- No NN/Engine/Training concepts or new Gradle edges appear in the artifact package. Javadocs,
  ADR, contract, examples, and tests describe the exact supported guarantee and limitations.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Engine glossary](../../../../glossary.md#term-engine); [Gradle glossary](../../../../glossary.md#term-gradle); [artifact glossary](../../../../glossary.md#term-artifact); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker validation:

```bash
./gradlew :extensions:checkpoint:check :extensions:checkpoint:javadoc
```

Integration/repository validation: reuse the focused gate after the independent review if no
executable change is made. No repository-wide suite is required here: no [dependency](../../../../glossary.md#term-dependency)/build edge or
cross-[module](../../../../glossary.md#term-module) behavior changes; [Checkpoint](../../../../glossary.md#term-checkpoint) 0004 owns the end-to-end recovery checkpoint.

## Documentation and review impact

- Public API, durable format, and failure modes require a clean Class C implementation context
  and independent targeted architecture/documentation review. Read the general, architecture,
  ADR, API, and glossary documentation profiles before writing their document types.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- Every new public/contract method and type needs detailed Javadoc for bounds, ownership,
  nullability, result, mutation, and failure conditions.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [nullability glossary](../../../../glossary.md#term-nullability))</span>

## Result

The [opaque](../../../../glossary.md#term-opaque) v1 [artifact](../../../../glossary.md#term-artifact) [API](../../../../glossary.md#term-application-programming-interface-api), bounded deterministic wire reader/writer, streamed temporary-file
[validation](../../../../glossary.md#term-validation), file force, and single [atomic](../../../../glossary.md#resource-transaction-and-rollback)-move [publication](../../../../glossary.md#term-publication) are implemented with focused corruption,
limits, immutability, replacement, and failure tests. Root/scoped contracts, explanations, [ADR](../../../../glossary.md#term-architecture-decision-record-adr)
0032, API reference, glossary, and affected Javadocs are current. The implementation-[owner](../../../../glossary.md#term-owner) gate
`./gradlew :extensions:checkpoint:check :extensions:checkpoint:javadoc` passed (15 tests,
no skips or failures). The existing architecture [dependency](../../../../glossary.md#term-dependency) test has only its stale
empty-project wording updated; `./gradlew :testing:architecture-tests:test --tests
io.github.pho001.synaptik.testing.architecture.CheckpointDependencyContractTest` passed
(one test, no skips or failures).

Independent Class C architecture/documentation review completed. It found post-move temporary
cleanup, which was corrected and verified by the 15-test gate, then signed off in a targeted
second pass. That pass also confirmed the amended Javadocs and failure claims. Markdown links
and anchors (478 local [targets](../../../../glossary.md#term-target)), [Javadoc](../../../../glossary.md#term-javadoc) generation, and `git diff --check` passed. No [model](../../../../glossary.md#term-model) or
training [schema](../../../../glossary.md#term-schema) and no [Gradle](../../../../glossary.md#term-gradle) edge were added.
