# Task 0003: [Model](../../../../glossary.md#term-model) [schema](../../../../glossary.md#term-schema) and [Engine](../../../../glossary.md#term-engine) [materialization](../../../../glossary.md#materialization) [handoff](../../../../glossary.md#term-handoff)

## Status

Complete. Frontier was verified at `e96fa174`: [Checkpoint](../../../../glossary.md#term-checkpoint) 0002 is Complete with focused [validation](../../../../glossary.md#term-validation),
documentation and independent Class C review; 0001 is Complete. No other Checkpoint writer is
active. Existing [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) state export and [Engine](../../../../glossary.md#term-engine) ordered bounded compute are present. Checkpoint 0004–
0007 remain Draft and are not authorized by this brief.

## Change class

Class C — durable [model](../../../../glossary.md#term-model)-state [schema](../../../../glossary.md#term-schema) and a cross-[module](../../../../glossary.md#term-module) [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)-to-[Engine](../../../../glossary.md#term-engine) [materialization](../../../../glossary.md#materialization) boundary.

## Goal

Turn a complete, already-bound [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [module](../../../../glossary.md#term-module)-state [snapshot](../../../../glossary.md#term-snapshot) into a bounded, [detached](../../../../glossary.md#detached-model-results) [model](../../../../glossary.md#term-model) [artifact](../../../../glossary.md#term-artifact)
candidate using only the public [Engine](../../../../glossary.md#term-engine) [facade](../../../../glossary.md#term-facade). Define and validate its [canonical](../../../../glossary.md#term-canonical-form) version-one
model [metadata](../../../../glossary.md#term-metadata); do not yet publish files or install state.

## Selected policy

- Use internal `checkpoint.model` types in this task. Its capture/codec is package-private; the
  public model save/inspect/load facade belongs to 0004. Keep the existing two direct Gradle edges
  to NN and Engine. Do not add a direct Model edge or expose Model/NN/Engine types in a new public
  Checkpoint signature in 0003.<span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec); [model glossary](../../../../glossary.md#term-model); [facade glossary](../../../../glossary.md#term-facade); [Gradle glossary](../../../../glossary.md#term-gradle); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Engine glossary](../../../../glossary.md#term-engine); [Checkpoint glossary](../../../../glossary.md#term-checkpoint); [signature glossary](../../../../glossary.md#term-signature))</span>
- The caller supplies an exact 32-byte SHA-256 fingerprint of its own versioned canonical model
  configuration, and optionally an exact 32-byte fingerprint of an input artifact. The caller is
  responsible for what those digests cover; Checkpoint cannot serialize an arbitrary Java lambda
  or infer external tokenizer/configuration semantics. Preserve both as immutable bytes.<span data-glossary-navigation="true"> ([SHA-256 glossary](../../../../glossary.md#term-sha-256); [fingerprint glossary](../../../../glossary.md#term-fingerprint); [canonical glossary](../../../../glossary.md#term-canonical-form); [tokenizer glossary](../../../../glossary.md#frozen-whitespace-tokenizer); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- The canonical model metadata is a versioned big-endian byte sequence with its own magic,
  explicit fixed wire IDs for every current <span data-glossary-navigation="true"> ([big-endian glossary](../../../../glossary.md#text-encoding-and-file-boundaries))</span>[`DataType`](../../../../glossary.md#data-type--datatype) and `StateKind`, ordered entries, strict
  UTF-8 length-prefixed exact NN state paths, fully static Shape extents, and expected canonical
  payload byte lengths. Reject malformed Unicode, duplicate paths, unknown IDs, negative extents,
  malformed lengths, and trailing bytes. Its SHA-256 is the model schema fingerprint and covers
  both caller digests and every ordered path/kind/type/Shape/length. Never use enum ordinal,<span data-glossary-navigation="true"> ([UTF-8 glossary](../../../../glossary.md#term-utf-8); [Shape glossary](../../../../glossary.md#shape); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [Unicode glossary](../../../../glossary.md#term-unicode))</span>
  `toString()`, a class name, or a Java [hash code](../../../../glossary.md#term-hash-code) as a persistent wire [identity](../../../../glossary.md#term-identity-meanings).
- Put [metadata](../../../../glossary.md#term-metadata) under the [artifact](../../../../glossary.md#term-artifact) key [`schema`](../../../../glossary.md#term-schema) and state bytes under fixed-width ASCII ordinal
  keys such as <span data-glossary-navigation="true"> ([ASCII glossary](../../../../glossary.md#term-ascii))</span>`state/00000000`. NN paths are metadata, never filesystem paths or artifact keys;
  NN permits names outside the artifact key grammar. Require artifact kind `model.v1` and exact
  one-to-one metadata/ordinal payload correspondence. The existing artifact envelope supplies
  byte-exact digests and canonical file framing; this task does not alter its v1 wire format.<span data-glossary-navigation="true"> ([artifact envelope glossary](../../../../glossary.md#term-artifact-envelope); [framing glossary](../../../../glossary.md#term-framing); [wire format glossary](../../../../glossary.md#term-wire-format))</span>
- [Capture](../../../../glossary.md#term-capture-meanings) `Module.stateDictionary()` once under caller-owned quiescence. Preflight path/metadata,
  entry, per-payload, aggregate, manifest, and final file-size bounds with checked arithmetic
  before Engine computation or Checkpoint-owned payload-sized allocation. Reject nonstatic, overflowed, or oversized
  state. Empty state is legal and makes no Engine call. Deduplicate repeated exact Tensor objects
  in first-encounter order for <span data-glossary-navigation="true"> ([quiescence glossary](../../../../glossary.md#term-quiescence); [Preflight glossary](../../../../glossary.md#term-preflight); [manifest glossary](../../../../glossary.md#term-manifest); [Tensor glossary](../../../../glossary.md#tensor))</span>`Engine.compute(List<Tensor>, limit)`; retain every distinct NN
  state path and payload in the candidate. The final aggregate bound counts every path, including
  aliases, not only unique Engine outputs. Verify returned type, Shape, and byte length before
  encoding. No direct host-storage inspection, backend access, implicit initialization, or live
  Tensor reference may escape in the result.<span data-glossary-navigation="true"> ([aliases glossary](../../../../glossary.md#aliasing-and-arena); [host glossary](../../../../glossary.md#term-host); [backend glossary](../../../../glossary.md#backend))</span>
- Treat returned [detached](../../../../glossary.md#detached-model-results) [`HostTensorValue`](../../../../glossary.md#host-tensor-value--hosttensorvalue)s as source values; copying into the opaque envelope
  may use more peak memory than the encoded-byte limit. Document that the limit bounds canonical
  output bytes, not Engine workspace or peak heap. No cross-call cache or hidden registry.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [heap glossary](../../../../glossary.md#term-heap); [cache glossary](../../../../glossary.md#term-cache); [registry glossary](../../../../glossary.md#term-registry))</span>

## [Scope](../../../../glossary.md#term-scope)

- Package-private `ModelArtifactCodec`, `ModelArtifactCandidate`, and `ModelArtifactCapture`
  under `extensions/checkpoint/.../checkpoint/model/`; add only focused helpers needed for those
  responsibilities. Decode/validate model metadata and payload correspondence now, but do not
  recreate Tensors or mutate NN state.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [metadata glossary](../../../../glossary.md#term-metadata); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [Tensors glossary](../../../../glossary.md#tensor); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Focused tests for exact F32/F64/low/integer/BOOL bytes where executable support exists, empty
  model, aliases, Unicode paths, noncanonical metadata, all dtype/kind wire IDs, fingerprints,
  bounds, Engine failure with no candidate, and detached lifetime. Use minimal test modules and
  small supported CPU computations; do not infer backend support from <span data-glossary-navigation="true"> ([F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [BOOL glossary](../../../../glossary.md#data-type--datatype); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [Unicode glossary](../../../../glossary.md#term-unicode); [dtype glossary](../../../../glossary.md#data-type--datatype); [Engine glossary](../../../../glossary.md#term-engine); [detached glossary](../../../../glossary.md#detached-model-results); [lifetime glossary](../../../../glossary.md#term-lifetime); [modules glossary](../../../../glossary.md#term-module); [CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend))</span>[`DataType`](../../../../glossary.md#data-type--datatype) membership.
- Update the owning extensions scoped contract and focused explanatory/API documentation for
  the new internal model-candidate behavior. Keep artifact API documentation accurate and do not
  present model file save/load as implemented. A significant schema decision needs an ADR.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [artifact glossary](../../../../glossary.md#term-artifact); [schema glossary](../../../../glossary.md#term-schema); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>

## Non-goals

- Public `ModelCheckpoint` save/inspect/load API, file publication, model reconstruction,
  deferred/unbound lazy model capture, strict NN installation, Training/Text dependency, generic
  serializer, optimizer state, or a new Engine materialization path. 0004 owns user-facing model
  persistence and strict restore; any NN change needed for unbound target inspection is separate.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [publication glossary](../../../../glossary.md#term-publication); [model glossary](../../../../glossary.md#term-model); [lazy glossary](../../../../glossary.md#term-lazy-evaluation); [capture glossary](../../../../glossary.md#term-capture-meanings); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [dependency glossary](../../../../glossary.md#term-dependency); [optimizer state glossary](../../../../glossary.md#term-optimizer-state); [Engine glossary](../../../../glossary.md#term-engine); [materialization glossary](../../../../glossary.md#materialization); [persistence glossary](../../../../glossary.md#term-persistence); [target glossary](../../../../glossary.md#term-target))</span>

## Contracts

- [`ARCHITECTURE.md` — Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence), [Core invariants](../../../../../ARCHITECTURE.md#core-invariants), [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>, [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules), [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements): preserve the authority index, [model](../../../../glossary.md#term-model)-only [ownership](../../../../glossary.md#term-ownership-meanings) and existing two [Checkpoint](../../../../glossary.md#term-checkpoint) edges.
- [`extensions-training.md` — Scope](../../../../architecture/contracts/extensions-training.md#scope)<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope))</span>, [`extensions/nn`](../../../../architecture/contracts/extensions-training.md#extensionsnn), [`extensions/checkpoint`](../../../../architecture/contracts/extensions-training.md#extensionscheckpoint): [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) owns state meaning and traversal; Checkpoint owns [detached](../../../../glossary.md#detached-model-results) model [artifact](../../../../glossary.md#term-artifact) meaning, not installation.
- [`runtime-prepare-engine.md` — `modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine): use only public [Engine](../../../../glossary.md#term-engine) [host](../../../../glossary.md#term-host) [materialization](../../../../glossary.md#materialization) and its ownership/failure contract.
- [`foundational-modules.md` — `modules/model`](../../../../architecture/contracts/foundational-modules.md#modulesmodel): [Tensor descriptor](../../../../glossary.md#tensor-descriptor)/[Shape](../../../../glossary.md#shape)/[DataType](../../../../glossary.md#data-type--datatype) meaning remains in Model; no reverse [dependency](../../../../glossary.md#term-dependency).

## Dependencies and integration

- Depends on: [Checkpoint](../../../../glossary.md#term-checkpoint) 0002 (Complete), current [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) state export, current [Engine](../../../../glossary.md#term-engine) bounded compute.
- Conflicts with: Checkpoint 0004–0007 and concurrent [model](../../../../glossary.md#term-model) [schema](../../../../glossary.md#term-schema) or owning-contract edits.
- Parallel group: None.
- Common base revision: N/A (serial work from `e96fa174`).
- Integration order: 0003 before 0004; no parallel integration.
- Integration [validation](../../../../glossary.md#term-validation): `./gradlew :extensions:checkpoint:check :extensions:checkpoint:javadoc` plus focused architecture [dependency](../../../../glossary.md#term-dependency) test if its assertion changes.
- Shared-document integration [owner](../../../../glossary.md#term-owner): 0003 implementation owner; no concurrent writers.

## Files and symbols

- `extensions/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/model/` — internal [schema](../../../../glossary.md#term-schema) [codec](../../../../glossary.md#term-codec), parsed candidate and [capture](../../../../glossary.md#term-capture-meanings) [handoff](../../../../glossary.md#term-handoff).
- `extensions/checkpoint/src/test/java/io/github/pho001/synaptik/checkpoint/model/` — focused behavior, corruption, bounds and [lifetime](../../../../glossary.md#term-lifetime) tests.
- `extensions/checkpoint/build.gradle.kts` — test-only [CPU](../../../../glossary.md#term-cpu) [vector](../../../../glossary.md#term-vector)-[module](../../../../glossary.md#term-module) [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) flag for real [Engine](../../../../glossary.md#term-engine) compute; no new project edge or shared build change.
- `docs/architecture/contracts/extensions-training.md`, focused architecture/API explanation,
  schema ADR and this brief/master/roadmap — current contract and evidence.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>

## Acceptance criteria

- Equal metadata and state produce identical model envelope bytes and schema fingerprint;
  changing config/input digest, ordered path/kind/type/Shape/length changes the fingerprint.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [model glossary](../../../../glossary.md#term-model); [schema glossary](../../../../glossary.md#term-schema); [fingerprint glossary](../../../../glossary.md#term-fingerprint); [digest glossary](../../../../glossary.md#term-digest); [Shape glossary](../../../../glossary.md#shape))</span>
- Arbitrary valid NN paths round-trip exactly without becoming artifact keys; malformed or
  noncanonical metadata, missing/extra/duplicate ordinal payloads, and unsupported IDs fail
  before returning a candidate. Raw payload bytes and source Tensor identity never leak mutably.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [round-trip glossary](../../../../glossary.md#schema-serialization-and-checkpoint); [artifact glossary](../../../../glossary.md#term-artifact); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [Tensor glossary](../../../../glossary.md#tensor); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Checkpoint-owned payload-sized allocation and every Engine call follow caller-limit and checked
  aggregate/file preflight. NN's own <span data-glossary-navigation="true"> ([Checkpoint glossary](../../../../glossary.md#term-checkpoint); [Engine glossary](../../../../glossary.md#term-engine); [preflight glossary](../../../../glossary.md#term-preflight))</span>`stateDictionary()` snapshot precedes that preflight and is
  not bounded by this task. Repeated Tensor identity is computed once yet represented at each path;
  empty state skips Engine. Failure returns no partial candidate and changes no NN binding or Engine
  ownership.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [binding glossary](../../../../glossary.md#term-binding); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- No new Gradle edge, backend shortcut, public persistence facade, or implicit model init appears.
  Docs and Javadocs distinguish detached candidate from future save/load.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [backend glossary](../../../../glossary.md#backend); [persistence glossary](../../../../glossary.md#term-persistence); [facade glossary](../../../../glossary.md#term-facade); [detached glossary](../../../../glossary.md#detached-model-results))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker: `./gradlew :extensions:checkpoint:check :extensions:checkpoint:javadoc`.
Integration: focused architecture [dependency](../../../../glossary.md#term-dependency) test if changed, then `git diff --check`; no
repository-wide gate unless dependency/shared-build behavior actually changes. The only build
configuration change is [module](../../../../glossary.md#term-module)-local and test-only; the existing two project edges are unchanged.

## Documentation and review impact

Class C [schema](../../../../glossary.md#term-schema) and cross-[module](../../../../glossary.md#term-module) [handoff](../../../../glossary.md#term-handoff) require a clean implementation context and independent
targeted architecture/documentation review. Review relevant [API](../../../../glossary.md#term-application-programming-interface-api)/Javadocs and glossary terms;
apply the general, architecture, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), API, and [planning](../../../../glossary.md#term-planning) documentation profiles as applicable.

## Result

Implementation and documentation completed from `63708bc8`. Internal `ModelArtifactCodec`, `ModelArtifactCapture`, and `ModelArtifactCandidate` now
[preflight](../../../../glossary.md#term-preflight) and [capture](../../../../glossary.md#term-capture-meanings) one complete [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [snapshot](../../../../glossary.md#term-snapshot) using public bounded [Engine](../../../../glossary.md#term-engine) compute, encode fixed
v1 IDs and strict [UTF-8](../../../../glossary.md#term-utf-8) paths, preserve both caller digests, and validate [detached](../../../../glossary.md#detached-model-results) [metadata](../../../../glossary.md#term-metadata) and
ordinal [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) correspondence. Six new [model](../../../../glossary.md#term-model) tests cover all current type IDs and executable
[CPU](../../../../glossary.md#term-cpu) [materialization](../../../../glossary.md#materialization), kind IDs, empty state, [aliases](../../../../glossary.md#aliasing-and-arena), [Unicode](../../../../glossary.md#term-unicode), corruption, [fingerprint](../../../../glossary.md#term-fingerprint) [sensitivity](../../../../glossary.md#term-sensitivity),
bounds, failure, and [lifetime](../../../../glossary.md#term-lifetime). The test-only [vector](../../../../glossary.md#term-vector) [module](../../../../glossary.md#term-module) flag is necessary for real CPU compute
in this module's test [JVM](../../../../glossary.md#term-java-virtual-machine-jvm); production dependencies remain exactly NN and Engine.

[Validation](../../../../glossary.md#term-validation): `./gradlew :extensions:checkpoint:test --tests '*ModelArtifactTest' --offline`
passed 7/7; `./gradlew :extensions:checkpoint:check :extensions:checkpoint:javadoc` passed
22/22 tests and [Javadoc](../../../../glossary.md#term-javadoc). The focused architecture [dependency](../../../../glossary.md#term-dependency) assertion did not change because
there is no new project edge. `git diff --check` is recorded in the final worktree [handoff](../../../../glossary.md#term-handoff).
The owning scoped contract, root routing summary, architecture/[API](../../../../glossary.md#term-application-programming-interface-api) explanations, glossary,
Javadocs, and [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0033 were updated. Public model save/inspect/load and state installation remain
unimplemented by design. Independent Class C architecture/documentation review identified an
early entry-proportional allocation before complete preflight. The implementation was corrected
with a [lazy](../../../../glossary.md#term-lazy-evaluation) [descriptor](../../../../glossary.md#term-descriptor) [view](../../../../glossary.md#term-view-meanings) and incremental bounds checks, and a virtual-million-state [regression](../../../../glossary.md#term-regression)
was added. A second independent targeted review signed off the correction; Markdown links and
anchors, whitespace, and `git diff --check` passed. This closes 0003, but does not itself
authorize implementation of Draft 0004.
