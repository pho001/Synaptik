# Task 0008: Generic [graph](../../../../glossary.md#graph)/value [persistence](../../../../glossary.md#term-persistence) and [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) glue cutover

> **Editorial reading context — not a status update.** Generic Checkpoint handles ordinary
> Tensor source descriptions and named host values; NN's relocated glue handles module-specific
> definitions and state paths. Compiler supplies bounded semantic capture/reconstruction;
> Checkpoint integrates it with encoding, limits and publication rather than interpreting or
> executing operations itself. A Source borrows Tensor references; copy acquires private named
> values, rebind substitutes inputs without reading their storage, and load reconstructs cold
> expressions. See today's [copy/rebind boundary](../../../../reference/checkpoint/graph.md#inputs-outputs-and-identity)
> and [NN consumer](../../../../reference/checkpoint/graph.md#owned-nn-model-consumer)
> for the split between semantic authority and integration responsibility.

## Status and change class

Complete — coordinator accepted 2026-10-17 after Halley's final APPROVE and root gate. Class C: [persistence](../../../../glossary.md#term-persistence)/[ownership](../../../../glossary.md#term-ownership-meanings)/public-edge cutover; clean execution and independent review. Authorized [Checkpoint](../../../../glossary.md#term-checkpoint) frontier: None.

## Goal and [scope](../../../../glossary.md#term-scope)

Deliver cold [Tensor](../../../../glossary.md#tensor) source/value copy/save/load/[rebind](../../../../glossary.md#capture-graph-ports-and-reconstruction); atomically split six [artifact](../../../../glossary.md#term-artifact) and nine [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)-glue files, preserving old readers.
Use existing Tensor/[graph](../../../../glossary.md#graph) infrastructure. Exclude Training0008/NN0028, [Model](../../../../glossary.md#term-model) execution/withEngine, broader codecs/[gradients](../../../../glossary.md#term-gradient),
[optimizer](../../../../glossary.md#term-optimizer)/native/[device](../../../../glossary.md#term-device)/[backend](../../../../glossary.md#backend)/codegen work, performance/numerical campaigns and compatibility wrappers.
No [Compiler](../../../../glossary.md#term-compiler) changes: report missing-predecessor blockers. Size exception: the [atomic](../../../../glossary.md#resource-transaction-and-rollback) wire/[owner](../../../../glossary.md#term-owner) split needs this ~17 KB packet;
splitting would separate wire [admission](../../../../glossary.md#term-admission), source association and [dependency](../../../../glossary.md#term-dependency) cutover. No duplicate old-v1 specification.

## Contracts and coordinated cutover

Read [root scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>: Repository layout; [Module](../../../../glossary.md#term-module)-[ownership](../../../../glossary.md#term-ownership-meanings) routing; [Dependency](../../../../glossary.md#term-dependency) rules; Testing requirements; Core invariants.
Owning [extensions-training](../../../../architecture/contracts/extensions-training.md): extensions/nn; Closed declarative models;
extensions/[checkpoint](../../../../glossary.md#term-checkpoint); Whole-[model](../../../../glossary.md#term-model) definition and composition; Whole-model public [API](../../../../glossary.md#term-application-programming-interface-api) and failures; Whole-model limits and
[preflight](../../../../glossary.md#term-preflight); [Canonical](../../../../glossary.md#term-canonical-form) definition version 1; Whole-model coherence and [publication](../../../../glossary.md#term-publication).
Read-only [compiler-autograd](../../../../architecture/contracts/compiler-autograd.md)<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd))</span>: modules/[compiler](../../../../glossary.md#term-compiler); [Compile artifacts](../../../../glossary.md#compile-artifacts); [Compile](../../../../glossary.md#compile) [lifecycle](../../../../glossary.md#lifecycle).
[ADR0039](../../../../design/decisions/0039-portable-graph-model-persistence.md): Selected API and type feasibility / Necessary boundary types;
Source, [shape](../../../../glossary.md#shape), state and ownership; Source migration and retired code; Coordinated authority cutovers and [validation](../../../../glossary.md#term-validation).
Read the [program](../../../model-execution-refactor.md). Source and contracts override historical tasks; report missing/ambiguous authority.

Current authority forbids [target](../../../../glossary.md#term-target) edges. BEFORE code relies on them, update the named root headings/index and only owning
extensions-training contract: move [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [persistence](../../../../glossary.md#term-persistence) rules to NN, retain [artifacts](../../../../glossary.md#term-artifact) under relocated Checkpoint, add [graph](../../../../glossary.md#graph) rules.
[Coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) explanations, ADR0039 and architecture tests; Compiler contract unchanged. Checkpoint: api(Model),
implementation(Compiler), no NN/Training/[Engine](../../../../glossary.md#term-engine). NN: retain api(Model,Training), add api(Checkpoint,Engine) for moved
PUBLIC [signatures](../../../../glossary.md#term-signature). No reverse edge/broad exemption. Stop on uncertainty beyond this accepted target.

## Dependencies and integration

- Depends on: Engine0021, Training0007, Compiler0008, Checkpoint0005B and accepted P2 Complete; accepted ADR0039 [target](../../../../glossary.md#term-target).
- Conflicts with: shared root/owning contract, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) state [ownership](../../../../glossary.md#term-ownership-meanings), [Checkpoint](../../../../glossary.md#term-checkpoint) wire/[API](../../../../glossary.md#term-application-programming-interface-api); Training0008/NN0028 inactive Draft.
- Parallel group: None; sole writer, no concurrent writes.
- Common base revision: d235c0d3b89d6f2eaad4c80a59fd1c5091ea2851 (main = origin/main).
- Integration order: Engine0021(done) → Training0007(done) → Compiler0008(done) → Checkpoint0008 → Training0008 → NN0028.
- Integration [validation](../../../../glossary.md#term-validation): commands below, then one coordinator repository test/[Javadoc](../../../../glossary.md#term-javadoc) join.
- Shared-document integration [owner](../../../../glossary.md#term-owner): main [model](../../../../glossary.md#term-model)-execution coordinator; exclusive-writer delegation only.

2026-10-17: Pauli definitive APPROVE; no findings/required revisions; ~17 KB [atomic](../../../../glossary.md#resource-transaction-and-rollback)-size exception accepted.
Coordinator [Ready](../../../../glossary.md#term-ready) verification: all dependencies above Complete with independent reviews/gates; supplied evidence includes
[Compiler](../../../../glossary.md#term-compiler) 307 tests/Javadoc and root test/Javadoc 5m1s (architecture 12, integration 131/one [fixture](../../../../glossary.md#term-fixture) skip), not rerun.
Pre-Ready frozen 2256 files unchanged; validated 2255-file predecessor executable [hashes](../../../../glossary.md#term-hash) match; only five allowed [planning](../../../../glossary.md#term-planning) deltas.
Both protected deltas/native [dylib](../../../../glossary.md#term-dylib) unchanged; main = origin/main at the common base; conflicts inactive, no other writer.
Coordinator authorizes Checkpoint0008 as sole Ready Checkpoint frontier. Main launches clean implementation;
authority cutover remains implementation-future; Training0008/NN0028 remain Draft.
Never edit/stage/revert THIRD_PARTY_NOTICES.md or
[modules](../../../../glossary.md#term-module)/[backend](../../../../glossary.md#backend)/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuOrdinaryPointwiseCastCleanJavaOracle.java.

## Production types and migration

Packages below are relative to io.github.pho001.synaptik; existing nested types move with their [owners](../../../../glossary.md#term-owner).

| Types | Destination / responsibility |
|---|---|
| ArtifactEnvelope, ArtifactFile, ArtifactFormatException, ArtifactLimits, ArtifactPublication, ArtifactWire | Six files to [modules](../../../../glossary.md#term-module)/[checkpoint](../../../../glossary.md#term-checkpoint), unchanged checkpoint.[artifact](../../../../glossary.md#term-artifact) package/behavior. |
| ModelCheckpoint, ModelCheckpointInspection, InputArtifactFingerprint; ModelArtifactCapture, ModelArtifactCandidate, ModelArtifactCodec, ModelStateRestoration, WholeModelComposition, WholeModelDefinitionCodec | Nine files from checkpoint.[model](../../../../glossary.md#term-model) to extensions/nn, package nn.checkpoint; first three public, six package-private. |
| GraphCheckpoint with public nested Source, [Leaf](../../../../glossary.md#term-leaf), Role | New checkpoint.[graph](../../../../glossary.md#graph) [facade](../../../../glossary.md#term-facade)/data boundary in modules/checkpoint. |
| GraphSource, GraphValueStorage, GraphArtifactCodec | New package-private final classes in checkpoint.graph: [identity](../../../../glossary.md#term-identity-meanings)/[metadata](../../../../glossary.md#term-metadata) association, raw [host](../../../../glossary.md#term-host) copying, closed wire. [Codec](../../../../glossary.md#term-codec) may nest private Reader/Sink. No generalized codec or second graph representation. |

Preserve :extensions:checkpoint through settings projectDir=file("modules/checkpoint"); move build/artifact tests, remove old
source/build location. Move four model-package test files to nn.checkpoint and ModelCheckpointEnvelopeApiTest to nn.checkpoint.api;
retain [Engine](../../../../glossary.md#term-engine)/[vector](../../../../glossary.md#term-vector) test configuration in [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context), no generic Engine test edge. Migrate WholeModelCheckpointEngineIntegrationTest/
PrivateTrainingRestoreIntegrationTest imports; their build already declares NN. External old-model users need nn.checkpoint and
explicit NN [dependency](../../../../glossary.md#term-dependency); artifact imports/project ID unchanged. Preserve old APIs/wire/defaults/failures via the linked contract.

## [API](../../../../glossary.md#term-application-programming-interface-api) and source association

Only these new public declarations, nested records/enum inside final GraphCheckpoint:

~~~java
public record Source(List<Tensor> inputs, List<Tensor> outputs, List<Leaf> leaves) {}
public record Leaf(String name, Role role, Tensor tensor) {}
public enum Role { PARAMETER, BUFFER, CONSTANT }
public static Source copy(Source source);
public static void save(Path target, Source source, ArtifactLimits limits) throws IOException;
public static Source load(Path source, ArtifactLimits limits) throws IOException;
public static List<Tensor> rebind(Source source, List<Tensor> inputs);
~~~

Records null-check/[snapshot](../../../../glossary.md#term-snapshot) in declaration/list order; borrow [Tensors](../../../../glossary.md#tensor), no [DAG](../../../../glossary.md#term-directed-acyclic-graph-dag)/executor/[close lifecycle](../../../../glossary.md#ownership-borrowing-and-lease). Operations revalidate.
Outputs nonempty/distinct; inputs ordered unique [provenance](../../../../glossary.md#provenance)-free [prototypes](../../../../glossary.md#term-prototype), possibly empty.
Named [leaves](../../../../glossary.md#term-leaf): provenance-free, disjoint from inputs; every reachable non-input leaf named, every named [identity](../../../../glossary.md#term-identity-meanings) reachable.
Names: unique strict [UTF-8](../../../../glossary.md#term-utf-8)/nonblank dot segments ([NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) path grammar), no [normalization](../../../../glossary.md#term-normalization)/prefix collision; keep registration order.
[Aliases](../../../../glossary.md#aliasing-and-arena) share exact Tensor [identity](../../../../glossary.md#term-reference-identity) and role, never merely bytes/overlap. [PARAMETER](../../../../glossary.md#term-parameter) needs floating requiresGrad=true;
[BUFFER](../../../../glossary.md#term-buffer)/[CONSTANT](../../../../glossary.md#term-constant) retain requiresGrad, never infer role from it.

GraphCompilationPort.captureSource(outputs) [captures](../../../../glossary.md#term-capture-meanings) once; [graph](../../../../glossary.md#graph).inputs includes state in first-encounter order.
Map by EXACT Tensor identity, never zip with Source.inputs. Keep unused declared [signature](../../../../glossary.md#term-signature) slots without payloads;
only unreachable NAMED registrations reject. Existing public [CompiledGraphModel](../../../../glossary.md#compiled-graph--compiledgraphmodel)/GraphValue/CompiledNode and Compiler0008 support this.
All source [nodes](../../../../glossary.md#node) are static FORWARD ADD/MUL/NEG/[MATMUL](../../../../glossary.md#matrix-multiplication--matmul) with [NoOperationAttrs](../../../../glossary.md#nooperationattrs), or [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) PERMUTE[1,0].
Preserve [descriptors](../../../../glossary.md#term-descriptor)/sharing/occurrences/output order. All seven [dtypes](../../../../glossary.md#data-type--datatype) where [Model](../../../../glossary.md#term-model) permits (not integral/[BOOL](../../../../glossary.md#data-type--datatype) NEG).
No optimization/[layout](../../../../glossary.md#layout) closure; zero-node data adds no [Engine](../../../../glossary.md#term-engine) execution support.

[rebind](../../../../glossary.md#capture-graph-ports-and-reconstruction) validates [metadata](../../../../glossary.md#term-metadata) then replacement count/full descriptors; substitutes explicit slots, retains named state.
Produced/repeated substitutions keep [opaque](../../../../glossary.md#term-opaque) upstream provenance; return [immutable](../../../../glossary.md#term-immutable) ordinary outputs (possibly repeated,
which ordinary [compile](../../../../glossary.md#compile) still rejects). No storage work. copy creates storage-free prototypes, copies each state identity once
and delegates reconstruction to [Compiler](../../../../glossary.md#term-compiler). load has no second [ownership](../../../../glossary.md#term-ownership-meanings) copy, original-state graph or [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) storage retained;
returned Tensor storage is ordinary [mutable](../../../../glossary.md#term-mutable) [heap](../../../../glossary.md#term-heap) state. Only Compiler owns the call-local [ValueId](../../../../glossary.md#logical-valueid-versus-physical-memory-slot)-to-Tensor reconstruction table.

## Storage and failure ordering

Read [Tensor](../../../../glossary.md#tensor)/TensorDescriptor/TensorFactory, HostTensorStorage/MemorySegmentStorage and LayoutDescriptor APIs.
copy/save read named [leaves](../../../../glossary.md#term-leaf) only; checked [canonical](../../../../glossary.md#term-canonical-form) geometry sizes unresolved [layouts](../../../../glossary.md#layout) too. Present leaf layouts must equal
zero-offset non-[view](../../../../glossary.md#term-view-meanings) [contiguous](../../../../glossary.md#term-contiguous) geometry. Intermediate PERMUTE views are [metadata](../../../../glossary.md#term-metadata). Preserve layout presence/requiresGrad.
[Snapshot](../../../../glossary.md#term-snapshot) hostStorage once per [identity](../../../../glossary.md#term-identity-meanings); check type/capacity/[liveness](../../../../glossary.md#term-liveness)/thread access, even for empty values.
Allow read-only [heap](../../../../glossary.md#term-heap)/native; reject absent/[device](../../../../glossary.md#term-device)-only, dead/inaccessible, short/noncanonical state.
Caller excludes mutation/rebinding/closure; no lock, [ownership](../../../../glossary.md#term-ownership-meanings) transfer or hidden execution.

Read native-order logical [lanes](../../../../glossary.md#term-lane) using unaligned integer-width access: 8 bytes [FLOAT64](../../../../glossary.md#term-float64)/[INT64](../../../../glossary.md#data-type--datatype), 4 [FLOAT32](../../../../glossary.md#term-float32)/[INT32](../../../../glossary.md#data-type--datatype),
2 [BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT16](../../../../glossary.md#term-float16), 1 [BOOL](../../../../glossary.md#data-type--datatype). HostTensorStorage itself promises no [byte order](../../../../glossary.md#text-encoding-and-file-boundaries); callers supply this representation.
Preserve all [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity), including [NaN payload](../../../../glossary.md#nan-infinity-and-signed-zero)/sign and [signed zero](../../../../glossary.md#term-signed-zero). Reject BOOL outside 0/1 before import, never normalize.
Ignore excess capacity; distinct overlapping Tensors are copied independently. Use matching primitive heap carriers and
MemorySegmentStorage with TensorFactory.create(originalDescriptor, Optional.empty(), Optional.of(storage)), not layout-closing
allocate. On load, final [descriptor](../../../../glossary.md#term-descriptor)-only candidate leaves may first undergo SourceCapture [preflight](../../../../glossary.md#term-preflight), then receive private
storage; no extra [weight](../../../../glossary.md#term-weight) wrappers or intermediate placeholders. Labels and native [owners](../../../../glossary.md#term-owner) are not persisted.

Nulls fail in declaration order. copy/save: metadata → geometry/(save) encoded bounds → all storage [admission](../../../../glossary.md#term-admission) →
BOOL [validation](../../../../glossary.md#term-validation)/raw copy → complete result/image → (save) ArtifactFile.save. Source/unsupported/bounds: IllegalArgumentException;
sizing [overflow](../../../../glossary.md#term-overflow): ArithmeticException (retain [Compiler](../../../../glossary.md#term-compiler) wrapping); absent/dead/inaccessible state: IllegalStateException.
[rebind](../../../../glossary.md#capture-graph-ports-and-reconstruction) checks metadata then replacements, no value gate.
load: ArtifactFile.read → complete wire/keys/descriptors/[aliases](../../../../glossary.md#aliasing-and-arena)/lengths/BOOL checks → descriptor-only leaves/Compiler
SourceCapture preflight → [host](../../../../glossary.md#term-host) import → reconstructSource → [publication](../../../../glossary.md#term-publication). Stored semantic/overflow errors become
ArtifactFormatException with cause; I/O stays IOException. Identifier exhaustion/VM failures propagate. Never publish partial
results, change source [bindings](../../../../glossary.md#term-binding) or trust earlier inspection. Preserve moved old-[facade](../../../../glossary.md#term-facade) ordering unchanged.

## [graph](../../../../glossary.md#graph)-[model](../../../../glossary.md#term-model).v1 wire and bounds

Ratify this bounded encoding in the owning contract during cutover. Reuse ArtifactFile v1: kind graph-model.v1,
one [schema](../../../../glossary.md#term-schema) and state/00000000… per unique named [identity](../../../../glossary.md#term-identity-meanings). No extra keys. [Big-endian](../../../../glossary.md#text-encoding-and-file-boundaries) i32/i64 signed, u8 unsigned;
counts nonnegative, flags exactly 0/1, no [padding](../../../../glossary.md#padding)/extensions/trailing bytes. IDs are fixed, never enum ordinals/class names.

| Schema order | Fields |
|---|---|
| Header | Eight [ASCII](../../../../glossary.md#term-ascii) bytes SYNGRF01; i32 version=1. |
| Inputs | i32 I; I [descriptors](../../../../glossary.md#term-descriptor) in declared order, unused slots included. Implicit ValueIds 0… I−1; no [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication). |
| State identities | i32 S; records in first named-identity order: descriptor, u8 role (1 [PARAMETER](../../../../glossary.md#term-parameter), 2 [BUFFER](../../../../glossary.md#term-buffer), 3 [CONSTANT](../../../../glossary.md#term-constant)), i64 byte length. ValueIds I… I+S−1; payload ordinal=state index. |
| Registrations | i32 A; records in caller order: positive i32 [UTF-8](../../../../glossary.md#term-utf-8) name length, exact bytes, i32 state index. Every state referenced; first new indices encounter 0,1,…; repeats encode [aliases](../../../../glossary.md#aliasing-and-arena). |
| [Nodes](../../../../glossary.md#node) | i32 N; [topological](../../../../glossary.md#term-topological-order) records: u8 phase=1; u8 op (1 ADD, 2 MUL, 3 NEG, 4 [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), 5 PERMUTE); fixed-[arity](../../../../glossary.md#term-arity) i32 input ValueIds (two for ADD/MUL/MATMUL, one otherwise); PERMUTE alone adds i32 [axis](../../../../glossary.md#term-axis)-count=2, i32 1, i32 0; output descriptor. [NodeId](../../../../glossary.md#nodeid) n, output [ValueId](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) I+S+n. |
| Outputs | Positive i32 O; O distinct ordered i32 ValueIds. |
| Descriptor | u8 [dtype](../../../../glossary.md#data-type--datatype) (1 [FLOAT64](../../../../glossary.md#term-float64), 2 [FLOAT32](../../../../glossary.md#term-float32), 3 [BFLOAT16](../../../../glossary.md#term-bfloat16), 4 [INT32](../../../../glossary.md#data-type--datatype), 5 [INT64](../../../../glossary.md#data-type--datatype), 6 [BOOL](../../../../glossary.md#data-type--datatype), 7 [FLOAT16](../../../../glossary.md#term-float16)); u8 requiresGrad; i32 [rank](../../../../glossary.md#term-rank); rank nonnegative i64 extents; u8 [layout](../../../../glossary.md#layout)-present; if present, i64 [element](../../../../glossary.md#term-element) offset, u8 [view](../../../../glossary.md#term-view-meanings), rank nonnegative i64 [strides](../../../../glossary.md#term-stride). Derive kind/span via LayoutDescriptor.of. |

Nodes reference preceding file-local ordinals only. Reject cycles/dangling/forward edges, dead nodes, unreachable named state,
unknown attrs/phases, duplicate outputs/noncanonical aliases. [Compiler](../../../../glossary.md#term-compiler) owns family/descriptor [validation](../../../../glossary.md#term-validation).
Node order is [capture](../../../../glossary.md#term-capture-meanings) order; decode compares normalized recapture with stored [topology](../../../../glossary.md#term-topology), retaining unused inputs.
No TensorIds/[batches](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)/[runtime](../../../../glossary.md#term-runtime) state. Payloads: exact [row-major](../../../../glossary.md#term-row-major) bits, [shape](../../../../glossary.md#shape)-count × width; one per identity.

No new defaults: save/load require ArtifactLimits; old whole-model defaults/tree bounds unchanged, not graph restrictions.
Schema ≤maxPayloadBytes (already ≤Integer.MAX_VALUE−8); bound counts/ranks/strings by remaining bytes before allocation.
S≤100000000, S+1≤maxEntries; checked I+S+N fits i32. Iterative traversal; no [backend](../../../../glossary.md#backend) rank-16 limit.
For schema D, unique state bytes B: [manifest](../../../../glossary.md#term-manifest) M=72+58*S; aggregate=D+B; file=48+M+D+B; entries=1+S.
Kind/key ceiling ≥14. Check all SIX limits and individual payloads with checked sums BEFORE encoding/bulk copies;
file reader already preflights [framing](../../../../glossary.md#term-framing) allocation. These are encoded, not peak-[heap](../../../../glossary.md#term-heap) bounds. copy/[rebind](../../../../glossary.md#capture-graph-ports-and-reconstruction) have no implicit
file ceiling; checked collection/geometry/[JVM](../../../../glossary.md#term-java-virtual-machine-jvm) limits apply, copy preflights each unique payload ≤Integer.MAX_VALUE−8.
No generic validation layer. Complete save image validation precedes unchanged ArtifactFile force/revalidate/one-[atomic](../../../../glossary.md#resource-transaction-and-rollback)-move
[publication](../../../../glossary.md#term-publication), with its existing failure/cleanup semantics; not [authentication](../../../../glossary.md#term-authentication) or portable power-[loss](../../../../glossary.md#loss) [durability](../../../../glossary.md#term-durability).

## Acceptance and [validation](../../../../glossary.md#term-validation)

- Split/acyclic edges and isolated consumers [compile](../../../../glossary.md#compile); old reader bytes/defaults/[aliases](../../../../glossary.md#aliasing-and-arena)/failure order remain unchanged.
- Roundtrip/re-save: seven raw types, scalar/empty, all roles/aliases/distinct equal values, sharing/multiple roots,
  Linear MATMUL/PERMUTE/ADD, descriptors/detached lifetime. Differing input/encounter order, state encountered first,
  unused inputs and ignored prototype storage; copied/loaded graphs retain no original state.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [Linear glossary](../../../../glossary.md#term-linear-meanings); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [descriptors glossary](../../../../glossary.md#term-descriptor); [detached glossary](../../../../glossary.md#detached-model-results); [lifetime glossary](../../../../glossary.md#term-lifetime); [prototype glossary](../../../../glossary.md#term-prototype))</span>
- Malformed/bound tests: all six exact/below limits, unknown IDs/version, UTF-8/flags/counts/ranks/overflow/truncation/trailing
  data, references/order/aliases/keys/lengths, BOOL, unsupported/unreadable state, late failures/no mutation/publication.<span data-glossary-navigation="true"> ([UTF-8 glossary](../../../../glossary.md#term-utf-8); [overflow glossary](../../../../glossary.md#term-overflow); [truncation glossary](../../../../glossary.md#term-truncation-toward-zero); [BOOL glossary](../../../../glossary.md#data-type--datatype); [publication glossary](../../../../glossary.md#term-publication))</span>
- rebind accepts produced/repeated substitutions, retains upstream provenance/state, rejects full-descriptor/arity mismatch.
  One small GraphCheckpointEngineIntegrationTest proves cold load then explicit Engine forward/ordinary autograd equivalence,
  Linear and two fresh same-signature inputs. Retain old flexible-batch and private restore integrations; no new Model API.<span data-glossary-navigation="true"> ([rebind glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [provenance glossary](../../../../glossary.md#provenance); [arity glossary](../../../../glossary.md#term-arity); [autograd glossary](../../../../glossary.md#autograd); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [Model glossary](../../../../glossary.md#term-model); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

Add GraphCheckpointTest, GraphArtifactCodecTest, GraphValueStorageTest in [checkpoint](../../../../glossary.md#term-checkpoint).[graph](../../../../glossary.md#graph). Update exact
CheckpointDependencyContractTest/NnTrainingDependencyContractTest/EngineCompositionContractTest, retaining role-specific checks.
Add isolated checkpointApiConsumer (only Checkpoint); extend nnApiConsumer (only [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)) to compile moved public [signatures](../../../../glossary.md#term-signature).
Wire consumer [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) into [architecture test](../../../../glossary.md#term-architecture-test); declare [modules](../../../../glossary.md#term-module)/checkpoint/src/**/*.java file inputs. Assert mapping,
absent old sources/packages, exact api/implementation edges, no NN/[Engine](../../../../glossary.md#term-engine)/Training imports even in generic tests and no
[Compiler](../../../../glossary.md#term-compiler) types in public GraphCheckpoint signatures. Training-only consumer remains. No broader exemptions.

Worker: focused development checks, then final affected-module/[Javadoc](../../../../glossary.md#term-javadoc) once:

~~~bash
./gradlew :extensions:checkpoint:test :extensions:nn:test :extensions:checkpoint:javadoc :extensions:nn:javadoc --max-workers=2 --console=plain
./gradlew :testing:architecture-tests:test --tests '*CheckpointDependencyContractTest' --tests '*NnTrainingDependencyContractTest' --tests '*EngineCompositionContractTest' --max-workers=2 --console=plain
./gradlew :testing:integration-tests:test --tests '*GraphCheckpointEngineIntegrationTest' --tests '*WholeModelCheckpointEngineIntegrationTest' --tests '*PrivateTrainingRestoreIntegrationTest' --max-workers=2 --console=plain
~~~

Coordinator: one source-matched ./gradlew test javadoc --max-workers=2 --console=plain join, existing [Metal](../../../../glossary.md#term-metal) [dylib](../../../../glossary.md#term-dylib)/configuration,
no rebuild. Reviewer reuses evidence unless executable changes/concrete findings invalidate it; final Javadoc after final edits.
No extra [backend](../../../../glossary.md#backend)/codegen/performance gates. Planner: links/anchors/size/allowed diff/git diff --check only; no [Gradle](../../../../glossary.md#term-gradle)/native work.

## Documentation and review impact

Apply documentation-rules/general/[planning](../../../../glossary.md#term-planning) profiles, and affected type profiles at implementation.
Update docs/architecture/{[module](../../../../glossary.md#term-module)-boundaries,[dependency](../../../../glossary.md#term-dependency)-rules,overview}.md, docs/developer-guide/repository-[layout](../../../../glossary.md#layout).md,
docs/api/{[checkpoint](../../../../glossary.md#term-checkpoint)-[artifact](../../../../glossary.md#term-artifact)-api,[model](../../../../glossary.md#term-model)-checkpoint-api}.md; add docs/api/[graph](../../../../glossary.md#graph)-checkpoint-api.md. Migrate maintained examples
in docs/user-guide/{declarative-models,running-models}.md and docs/api/training-api.md. Review glossary checkpoint/raw-source/
[alias](../../../../glossary.md#aliasing-and-arena) terms and new/moved Javadocs; synchronize ADR0039/planning indexes. Show cold save/load versus flexible old readers;
preserve historical tasks, no [TARGET](../../../../glossary.md#term-target) Model [API](../../../../glossary.md#term-application-programming-interface-api) as implemented. Planning approval/[Ready](../../../../glossary.md#term-ready) verification complete; independent code/docs
approval and coordinator gate precede Complete. Report extra production types/policy, never silently expand.

## Result

2026-10-17: coordinator accepted Complete; Halley definitive APPROVE covers code/tests/docs/[Javadoc](../../../../glossary.md#term-javadoc)/glossary, zero outstanding.
Delivered GraphCheckpoint Source/[Leaf](../../../../glossary.md#term-leaf)/Role copy/save/load/[rebind](../../../../glossary.md#capture-graph-ports-and-reconstruction), GraphSource [identity](../../../../glossary.md#term-identity-meanings) association,
GraphValueStorage raw [host](../../../../glossary.md#term-host) copying and GraphArtifactCodec [graph](../../../../glossary.md#graph)-[model](../../../../glossary.md#term-model).v1; no extra production types.
Moved six [artifact](../../../../glossary.md#term-artifact) files/build/tests to [modules](../../../../glossary.md#term-module)/[checkpoint](../../../../glossary.md#term-checkpoint) and nine [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) files/five test files to nn.checkpoint;
settings retains :extensions:checkpoint; no forwarders, old readers/defaults/failures preserved.
Root/owning contract, architecture tests/isolated consumers, ADR0039, explanations/[API](../../../../glossary.md#term-application-programming-interface-api)/user/[layout](../../../../glossary.md#layout) guides,
Javadocs and targeted glossary updated; [Compiler](../../../../glossary.md#term-compiler) cold API/contract unchanged; Model/Training additions stay Draft.
Worker session 84552: `./gradlew :extensions:checkpoint:test :extensions:nn:test :extensions:checkpoint:javadoc :extensions:nn:javadoc --max-workers=2 --console=plain` — exit 0, 3s; Checkpoint 36, NN 375, zero failures/errors/skips; Javadocs passed after final edits.
Worker session 4428: `./gradlew :testing:architecture-tests:test --tests '*CheckpointDependencyContractTest' --tests '*NnTrainingDependencyContractTest' --tests '*EngineCompositionContractTest' --max-workers=2 --console=plain` — exit 0, 1s; 4 cases and isolated Checkpoint/NN/Training consumers passed.
Worker session 23989: `./gradlew :testing:integration-tests:test --tests '*GraphCheckpointEngineIntegrationTest' --tests '*WholeModelCheckpointEngineIntegrationTest' --tests '*PrivateTrainingRestoreIntegrationTest' --max-workers=2 --console=plain` — exit 0, 1s; 6 cases, zero failures/errors/skips.
Coordinator gate: `SYNAPTIK_METAL_TEST_LIBRARY=/private/tmp/synaptik-checkpoint-leaf.oCW897/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc :testing:architecture-tests:test --rerun :testing:integration-tests:test --rerun --no-build-cache --no-configuration-cache --max-workers=2 --console=plain` — exit 0, BUILD SUCCESSFUL, 4s; 107 tasks (2 executed, 105 up-to-date).
Fresh architecture 12/0 skipped; full integration 132/1 skipped, zero failures/errors; both actual [Metal](../../../../glossary.md#term-metal) training cases passed.
Integration skip: host [CPU](../../../../glossary.md#term-cpu) [fixture](../../../../glossary.md#term-fixture) had no eligible local workload alternatives. Existing verified Metal [dylib](../../../../glossary.md#term-dylib) reused, no rebuild.
Current 19-project reports: 4188 tests, zero failures/errors, 41 skipped; not all reports freshly executed:
CPU 30 and Metal 10 opt-in skips are cached; integration contributes one fixture skip. Worker Checkpoint/NN tests/Javadocs reused.
Coordinator verified all 2266 files identical across gate and both protected user changes unchanged; main/origin/main remain d235c0d3b89d6f2eaad4c80a59fd1c5091ea2851.
All original 57 Checkpoint cases retained; six artifact files byte-identical and nine NN algorithms identical ignoring package/Javadocs.
Final completion edits are [metadata](../../../../glossary.md#term-metadata)-only in the five coordinator-authorized files; links/whitespace and outside-[scope](../../../../glossary.md#term-scope)/protected [hashes](../../../../glossary.md#term-hash) checked.
No unresolved issue or required follow-up for this capability. Checkpoint frontier: None; Training0008 is next [planning](../../../../glossary.md#term-planning) candidate, still Draft; NN0028/Model/withEngine remain future.
Status: Complete
