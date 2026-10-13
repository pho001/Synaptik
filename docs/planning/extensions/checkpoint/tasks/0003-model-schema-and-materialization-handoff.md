# Task 0003: Model schema and Engine materialization handoff

## Status

Complete. Frontier was verified at `e96fa174`: Checkpoint 0002 is Complete with focused validation,
documentation and independent Class C review; 0001 is Complete. No other Checkpoint writer is
active. Existing NN state export and Engine ordered bounded compute are present. Checkpoint 0004–
0007 remain Draft and are not authorized by this brief.

## Change class

Class C — durable model-state schema and a cross-module NN-to-Engine materialization boundary.

## Goal

Turn a complete, already-bound NN module-state snapshot into a bounded, detached model artifact
candidate using only the public Engine facade. Define and validate its canonical version-one
model metadata; do not yet publish files or install state.

## Selected policy

- Use internal `checkpoint.model` types in this task. Its capture/codec is package-private; the
  public model save/inspect/load facade belongs to 0004. Keep the existing two direct Gradle edges
  to NN and Engine. Do not add a direct Model edge or expose Model/NN/Engine types in a new public
  Checkpoint signature in 0003.
- The caller supplies an exact 32-byte SHA-256 fingerprint of its own versioned canonical model
  configuration, and optionally an exact 32-byte fingerprint of an input artifact. The caller is
  responsible for what those digests cover; Checkpoint cannot serialize an arbitrary Java lambda
  or infer external tokenizer/configuration semantics. Preserve both as immutable bytes.
- The canonical model metadata is a versioned big-endian byte sequence with its own magic,
  explicit fixed wire IDs for every current `DataType` and `StateKind`, ordered entries, strict
  UTF-8 length-prefixed exact NN state paths, fully static Shape extents, and expected canonical
  payload byte lengths. Reject malformed Unicode, duplicate paths, unknown IDs, negative extents,
  malformed lengths, and trailing bytes. Its SHA-256 is the model schema fingerprint and covers
  both caller digests and every ordered path/kind/type/Shape/length. Never use enum ordinal,
  `toString()`, a class name, or a Java hash code as a persistent wire identity.
- Put metadata under the artifact key `schema` and state bytes under fixed-width ASCII ordinal
  keys such as `state/00000000`. NN paths are metadata, never filesystem paths or artifact keys;
  NN permits names outside the artifact key grammar. Require artifact kind `model.v1` and exact
  one-to-one metadata/ordinal payload correspondence. The existing artifact envelope supplies
  byte-exact digests and canonical file framing; this task does not alter its v1 wire format.
- Capture `Module.stateDictionary()` once under caller-owned quiescence. Preflight path/metadata,
  entry, per-payload, aggregate, manifest, and final file-size bounds with checked arithmetic
  before Engine computation or Checkpoint-owned payload-sized allocation. Reject nonstatic, overflowed, or oversized
  state. Empty state is legal and makes no Engine call. Deduplicate repeated exact Tensor objects
  in first-encounter order for `Engine.compute(List<Tensor>, limit)`; retain every distinct NN
  state path and payload in the candidate. The final aggregate bound counts every path, including
  aliases, not only unique Engine outputs. Verify returned type, Shape, and byte length before
  encoding. No direct host-storage inspection, backend access, implicit initialization, or live
  Tensor reference may escape in the result.
- Treat returned detached `HostTensorValue`s as source values; copying into the opaque envelope
  may use more peak memory than the encoded-byte limit. Document that the limit bounds canonical
  output bytes, not Engine workspace or peak heap. No cross-call cache or hidden registry.

## Scope

- Package-private `ModelArtifactCodec`, `ModelArtifactCandidate`, and `ModelArtifactCapture`
  under `extensions/checkpoint/.../checkpoint/model/`; add only focused helpers needed for those
  responsibilities. Decode/validate model metadata and payload correspondence now, but do not
  recreate Tensors or mutate NN state.
- Focused tests for exact F32/F64/low/integer/BOOL bytes where executable support exists, empty
  model, aliases, Unicode paths, noncanonical metadata, all dtype/kind wire IDs, fingerprints,
  bounds, Engine failure with no candidate, and detached lifetime. Use minimal test modules and
  small supported CPU computations; do not infer backend support from `DataType` membership.
- Update the owning extensions scoped contract and focused explanatory/API documentation for
  the new internal model-candidate behavior. Keep artifact API documentation accurate and do not
  present model file save/load as implemented. A significant schema decision needs an ADR.

## Non-goals

- Public `ModelCheckpoint` save/inspect/load API, file publication, model reconstruction,
  deferred/unbound lazy model capture, strict NN installation, Training/Text dependency, generic
  serializer, optimizer state, or a new Engine materialization path. 0004 owns user-facing model
  persistence and strict restore; any NN change needed for unbound target inspection is separate.

## Contracts

- [`ARCHITECTURE.md` — Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence), [Core invariants](../../../../../ARCHITECTURE.md#core-invariants), [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing), [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules), [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements): preserve the authority index, model-only ownership and existing two Checkpoint edges.
- [`extensions-training.md` — Scope](../../../../architecture/contracts/extensions-training.md#scope), [`extensions/nn`](../../../../architecture/contracts/extensions-training.md#extensionsnn), [`extensions/checkpoint`](../../../../architecture/contracts/extensions-training.md#extensionscheckpoint): NN owns state meaning and traversal; Checkpoint owns detached model artifact meaning, not installation.
- [`runtime-prepare-engine.md` — `modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine): use only public Engine host materialization and its ownership/failure contract.
- [`foundational-modules.md` — `modules/model`](../../../../architecture/contracts/foundational-modules.md#modulesmodel): Tensor descriptor/Shape/DataType meaning remains in Model; no reverse dependency.

## Dependencies and integration

- Depends on: Checkpoint 0002 (Complete), current NN state export, current Engine bounded compute.
- Conflicts with: Checkpoint 0004–0007 and concurrent model schema or owning-contract edits.
- Parallel group: None.
- Common base revision: N/A (serial work from `e96fa174`).
- Integration order: 0003 before 0004; no parallel integration.
- Integration validation: `./gradlew :extensions:checkpoint:check :extensions:checkpoint:javadoc` plus focused architecture dependency test if its assertion changes.
- Shared-document integration owner: 0003 implementation owner; no concurrent writers.

## Files and symbols

- `extensions/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/model/` — internal schema codec, parsed candidate and capture handoff.
- `extensions/checkpoint/src/test/java/io/github/pho001/synaptik/checkpoint/model/` — focused behavior, corruption, bounds and lifetime tests.
- `extensions/checkpoint/build.gradle.kts` — test-only CPU vector-module JVM flag for real Engine compute; no new project edge or shared build change.
- `docs/architecture/contracts/extensions-training.md`, focused architecture/API explanation,
  schema ADR and this brief/master/roadmap — current contract and evidence.

## Acceptance criteria

- Equal metadata and state produce identical model envelope bytes and schema fingerprint;
  changing config/input digest, ordered path/kind/type/Shape/length changes the fingerprint.
- Arbitrary valid NN paths round-trip exactly without becoming artifact keys; malformed or
  noncanonical metadata, missing/extra/duplicate ordinal payloads, and unsupported IDs fail
  before returning a candidate. Raw payload bytes and source Tensor identity never leak mutably.
- Checkpoint-owned payload-sized allocation and every Engine call follow caller-limit and checked
  aggregate/file preflight. NN's own `stateDictionary()` snapshot precedes that preflight and is
  not bounded by this task. Repeated Tensor identity is computed once yet represented at each path;
  empty state skips Engine. Failure returns no partial candidate and changes no NN binding or Engine
  ownership.
- No new Gradle edge, backend shortcut, public persistence facade, or implicit model init appears.
  Docs and Javadocs distinguish detached candidate from future save/load.

## Validation

Worker: `./gradlew :extensions:checkpoint:check :extensions:checkpoint:javadoc`.
Integration: focused architecture dependency test if changed, then `git diff --check`; no
repository-wide gate unless dependency/shared-build behavior actually changes. The only build
configuration change is module-local and test-only; the existing two project edges are unchanged.

## Documentation and review impact

Class C schema and cross-module handoff require a clean implementation context and independent
targeted architecture/documentation review. Review relevant API/Javadocs and glossary terms;
apply the general, architecture, ADR, API, and planning documentation profiles as applicable.

## Result

Implementation and documentation completed from `63708bc8`. Internal `ModelArtifactCodec`, `ModelArtifactCapture`, and `ModelArtifactCandidate` now
preflight and capture one complete NN snapshot using public bounded Engine compute, encode fixed
v1 IDs and strict UTF-8 paths, preserve both caller digests, and validate detached metadata and
ordinal payload correspondence. Six new model tests cover all current type IDs and executable
CPU materialization, kind IDs, empty state, aliases, Unicode, corruption, fingerprint sensitivity,
bounds, failure, and lifetime. The test-only vector module flag is necessary for real CPU compute
in this module's test JVM; production dependencies remain exactly NN and Engine.

Validation: `./gradlew :extensions:checkpoint:test --tests '*ModelArtifactTest' --offline`
passed 7/7; `./gradlew :extensions:checkpoint:check :extensions:checkpoint:javadoc` passed
22/22 tests and Javadoc. The focused architecture dependency assertion did not change because
there is no new project edge. `git diff --check` is recorded in the final worktree handoff.
The owning scoped contract, root routing summary, architecture/API explanations, glossary,
Javadocs, and ADR 0033 were updated. Public model save/inspect/load and state installation remain
unimplemented by design. Independent Class C architecture/documentation review identified an
early entry-proportional allocation before complete preflight. The implementation was corrected
with a lazy descriptor view and incremental bounds checks, and a virtual-million-state regression
was added. A second independent targeted review signed off the correction; Markdown links and
anchors, whitespace, and `git diff --check` passed. This closes 0003, but does not itself
authorize implementation of Draft 0004.
