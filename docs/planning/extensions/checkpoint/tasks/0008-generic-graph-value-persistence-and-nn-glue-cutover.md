# Task 0008: Generic graph/value persistence and NN glue cutover

## Status and change class

Complete — coordinator accepted 2026-10-17 after Halley's final APPROVE and root gate. Class C: persistence/ownership/public-edge cutover; clean execution and independent review. Authorized Checkpoint frontier: None.

## Goal and scope

Deliver cold Tensor source/value copy/save/load/rebind; atomically split six artifact and nine NN-glue files, preserving old readers.
Use existing Tensor/graph infrastructure. Exclude Training0008/NN0028, Model execution/withEngine, broader codecs/gradients,
optimizer/native/device/backend/codegen work, performance/numerical campaigns and compatibility wrappers.
No Compiler changes: report missing-predecessor blockers. Size exception: the atomic wire/owner split needs this ~17 KB packet;
splitting would separate wire admission, source association and dependency cutover. No duplicate old-v1 specification.

## Contracts and coordinated cutover

Read [root scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts): Repository layout; Module-ownership routing; Dependency rules; Testing requirements; Core invariants.
Owning [extensions-training](../../../../architecture/contracts/extensions-training.md): extensions/nn; Closed declarative models;
extensions/checkpoint; Whole-model definition and composition; Whole-model public API and failures; Whole-model limits and
preflight; Canonical definition version 1; Whole-model coherence and publication.
Read-only [compiler-autograd](../../../../architecture/contracts/compiler-autograd.md): modules/compiler; Compile artifacts; Compile lifecycle.
[ADR0039](../../../../design/decisions/0039-portable-graph-model-persistence.md): Selected API and type feasibility / Necessary boundary types;
Source, shape, state and ownership; Source migration and retired code; Coordinated authority cutovers and validation.
Read the [program](../../../model-execution-refactor.md). Source and contracts override historical tasks; report missing/ambiguous authority.

Current authority forbids target edges. BEFORE code relies on them, update the named root headings/index and only owning
extensions-training contract: move NN persistence rules to NN, retain artifacts under relocated Checkpoint, add graph rules.
Coordinate explanations, ADR0039 and architecture tests; Compiler contract unchanged. Checkpoint: api(Model),
implementation(Compiler), no NN/Training/Engine. NN: retain api(Model,Training), add api(Checkpoint,Engine) for moved
PUBLIC signatures. No reverse edge/broad exemption. Stop on uncertainty beyond this accepted target.

## Dependencies and integration

- Depends on: Engine0021, Training0007, Compiler0008, Checkpoint0005B and accepted P2 Complete; accepted ADR0039 target.
- Conflicts with: shared root/owning contract, NN state ownership, Checkpoint wire/API; Training0008/NN0028 inactive Draft.
- Parallel group: None; sole writer, no concurrent writes.
- Common base revision: d235c0d3b89d6f2eaad4c80a59fd1c5091ea2851 (main = origin/main).
- Integration order: Engine0021(done) → Training0007(done) → Compiler0008(done) → Checkpoint0008 → Training0008 → NN0028.
- Integration validation: commands below, then one coordinator repository test/Javadoc join.
- Shared-document integration owner: main model-execution coordinator; exclusive-writer delegation only.

2026-10-17: Pauli definitive APPROVE; no findings/required revisions; ~17 KB atomic-size exception accepted.
Coordinator Ready verification: all dependencies above Complete with independent reviews/gates; supplied evidence includes
Compiler 307 tests/Javadoc and root test/Javadoc 5m1s (architecture 12, integration 131/one fixture skip), not rerun.
Pre-Ready frozen 2256 files unchanged; validated 2255-file predecessor executable hashes match; only five allowed planning deltas.
Both protected deltas/native dylib unchanged; main = origin/main at the common base; conflicts inactive, no other writer.
Coordinator authorizes Checkpoint0008 as sole Ready Checkpoint frontier. Main launches clean implementation;
authority cutover remains implementation-future; Training0008/NN0028 remain Draft.
Never edit/stage/revert THIRD_PARTY_NOTICES.md or
modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuOrdinaryPointwiseCastCleanJavaOracle.java.

## Production types and migration

Packages below are relative to io.github.pho001.synaptik; existing nested types move with their owners.

| Types | Destination / responsibility |
|---|---|
| ArtifactEnvelope, ArtifactFile, ArtifactFormatException, ArtifactLimits, ArtifactPublication, ArtifactWire | Six files to modules/checkpoint, unchanged checkpoint.artifact package/behavior. |
| ModelCheckpoint, ModelCheckpointInspection, InputArtifactFingerprint; ModelArtifactCapture, ModelArtifactCandidate, ModelArtifactCodec, ModelStateRestoration, WholeModelComposition, WholeModelDefinitionCodec | Nine files from checkpoint.model to extensions/nn, package nn.checkpoint; first three public, six package-private. |
| GraphCheckpoint with public nested Source, Leaf, Role | New checkpoint.graph facade/data boundary in modules/checkpoint. |
| GraphSource, GraphValueStorage, GraphArtifactCodec | New package-private final classes in checkpoint.graph: identity/metadata association, raw host copying, closed wire. Codec may nest private Reader/Sink. No generalized codec or second graph representation. |

Preserve :extensions:checkpoint through settings projectDir=file("modules/checkpoint"); move build/artifact tests, remove old
source/build location. Move four model-package test files to nn.checkpoint and ModelCheckpointEnvelopeApiTest to nn.checkpoint.api;
retain Engine/vector test configuration in NN, no generic Engine test edge. Migrate WholeModelCheckpointEngineIntegrationTest/
PrivateTrainingRestoreIntegrationTest imports; their build already declares NN. External old-model users need nn.checkpoint and
explicit NN dependency; artifact imports/project ID unchanged. Preserve old APIs/wire/defaults/failures via the linked contract.

## API and source association

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

Records null-check/snapshot in declaration/list order; borrow Tensors, no DAG/executor/close lifecycle. Operations revalidate.
Outputs nonempty/distinct; inputs ordered unique provenance-free prototypes, possibly empty.
Named leaves: provenance-free, disjoint from inputs; every reachable non-input leaf named, every named identity reachable.
Names: unique strict UTF-8/nonblank dot segments (NN path grammar), no normalization/prefix collision; keep registration order.
Aliases share exact Tensor identity and role, never merely bytes/overlap. PARAMETER needs floating requiresGrad=true;
BUFFER/CONSTANT retain requiresGrad, never infer role from it.

GraphCompilationPort.captureSource(outputs) captures once; graph.inputs includes state in first-encounter order.
Map by EXACT Tensor identity, never zip with Source.inputs. Keep unused declared signature slots without payloads;
only unreachable NAMED registrations reject. Existing public CompiledGraphModel/GraphValue/CompiledNode and Compiler0008 support this.
All source nodes are static FORWARD ADD/MUL/NEG/MATMUL with NoOperationAttrs, or rank-two PERMUTE[1,0].
Preserve descriptors/sharing/occurrences/output order. All seven dtypes where Model permits (not integral/BOOL NEG).
No optimization/layout closure; zero-node data adds no Engine execution support.

rebind validates metadata then replacement count/full descriptors; substitutes explicit slots, retains named state.
Produced/repeated substitutions keep opaque upstream provenance; return immutable ordinary outputs (possibly repeated,
which ordinary compile still rejects). No storage work. copy creates storage-free prototypes, copies each state identity once
and delegates reconstruction to Compiler. load has no second ownership copy, original-state graph or batch storage retained;
returned Tensor storage is ordinary mutable heap state. Only Compiler owns the call-local ValueId-to-Tensor reconstruction table.

## Storage and failure ordering

Read Tensor/TensorDescriptor/TensorFactory, HostTensorStorage/MemorySegmentStorage and LayoutDescriptor APIs.
copy/save read named leaves only; checked canonical geometry sizes unresolved layouts too. Present leaf layouts must equal
zero-offset non-view contiguous geometry. Intermediate PERMUTE views are metadata. Preserve layout presence/requiresGrad.
Snapshot hostStorage once per identity; check type/capacity/liveness/thread access, even for empty values.
Allow read-only heap/native; reject absent/device-only, dead/inaccessible, short/noncanonical state.
Caller excludes mutation/rebinding/closure; no lock, ownership transfer or hidden execution.

Read native-order logical lanes using unaligned integer-width access: 8 bytes FLOAT64/INT64, 4 FLOAT32/INT32,
2 BFLOAT16/FLOAT16, 1 BOOL. HostTensorStorage itself promises no byte order; callers supply this representation.
Preserve all raw bits, including NaN payload/sign and signed zero. Reject BOOL outside 0/1 before import, never normalize.
Ignore excess capacity; distinct overlapping Tensors are copied independently. Use matching primitive heap carriers and
MemorySegmentStorage with TensorFactory.create(originalDescriptor, Optional.empty(), Optional.of(storage)), not layout-closing
allocate. On load, final descriptor-only candidate leaves may first undergo SourceCapture preflight, then receive private
storage; no extra weight wrappers or intermediate placeholders. Labels and native owners are not persisted.

Nulls fail in declaration order. copy/save: metadata → geometry/(save) encoded bounds → all storage admission →
BOOL validation/raw copy → complete result/image → (save) ArtifactFile.save. Source/unsupported/bounds: IllegalArgumentException;
sizing overflow: ArithmeticException (retain Compiler wrapping); absent/dead/inaccessible state: IllegalStateException.
rebind checks metadata then replacements, no value gate.
load: ArtifactFile.read → complete wire/keys/descriptors/aliases/lengths/BOOL checks → descriptor-only leaves/Compiler
SourceCapture preflight → host import → reconstructSource → publication. Stored semantic/overflow errors become
ArtifactFormatException with cause; I/O stays IOException. Identifier exhaustion/VM failures propagate. Never publish partial
results, change source bindings or trust earlier inspection. Preserve moved old-facade ordering unchanged.

## graph-model.v1 wire and bounds

Ratify this bounded encoding in the owning contract during cutover. Reuse ArtifactFile v1: kind graph-model.v1,
one schema and state/00000000… per unique named identity. No extra keys. Big-endian i32/i64 signed, u8 unsigned;
counts nonnegative, flags exactly 0/1, no padding/extensions/trailing bytes. IDs are fixed, never enum ordinals/class names.

| Schema order | Fields |
|---|---|
| Header | Eight ASCII bytes SYNGRF01; i32 version=1. |
| Inputs | i32 I; I descriptors in declared order, unused slots included. Implicit ValueIds 0… I−1; no payload. |
| State identities | i32 S; records in first named-identity order: descriptor, u8 role (1 PARAMETER, 2 BUFFER, 3 CONSTANT), i64 byte length. ValueIds I… I+S−1; payload ordinal=state index. |
| Registrations | i32 A; records in caller order: positive i32 UTF-8 name length, exact bytes, i32 state index. Every state referenced; first new indices encounter 0,1,…; repeats encode aliases. |
| Nodes | i32 N; topological records: u8 phase=1; u8 op (1 ADD, 2 MUL, 3 NEG, 4 MATMUL, 5 PERMUTE); fixed-arity i32 input ValueIds (two for ADD/MUL/MATMUL, one otherwise); PERMUTE alone adds i32 axis-count=2, i32 1, i32 0; output descriptor. NodeId n, output ValueId I+S+n. |
| Outputs | Positive i32 O; O distinct ordered i32 ValueIds. |
| Descriptor | u8 dtype (1 FLOAT64, 2 FLOAT32, 3 BFLOAT16, 4 INT32, 5 INT64, 6 BOOL, 7 FLOAT16); u8 requiresGrad; i32 rank; rank nonnegative i64 extents; u8 layout-present; if present, i64 element offset, u8 view, rank nonnegative i64 strides. Derive kind/span via LayoutDescriptor.of. |

Nodes reference preceding file-local ordinals only. Reject cycles/dangling/forward edges, dead nodes, unreachable named state,
unknown attrs/phases, duplicate outputs/noncanonical aliases. Compiler owns family/descriptor validation.
Node order is capture order; decode compares normalized recapture with stored topology, retaining unused inputs.
No TensorIds/batches/runtime state. Payloads: exact row-major bits, shape-count × width; one per identity.

No new defaults: save/load require ArtifactLimits; old whole-model defaults/tree bounds unchanged, not graph restrictions.
Schema ≤maxPayloadBytes (already ≤Integer.MAX_VALUE−8); bound counts/ranks/strings by remaining bytes before allocation.
S≤100000000, S+1≤maxEntries; checked I+S+N fits i32. Iterative traversal; no backend rank-16 limit.
For schema D, unique state bytes B: manifest M=72+58*S; aggregate=D+B; file=48+M+D+B; entries=1+S.
Kind/key ceiling ≥14. Check all SIX limits and individual payloads with checked sums BEFORE encoding/bulk copies;
file reader already preflights framing allocation. These are encoded, not peak-heap bounds. copy/rebind have no implicit
file ceiling; checked collection/geometry/JVM limits apply, copy preflights each unique payload ≤Integer.MAX_VALUE−8.
No generic validation layer. Complete save image validation precedes unchanged ArtifactFile force/revalidate/one-atomic-move
publication, with its existing failure/cleanup semantics; not authentication or portable power-loss durability.

## Acceptance and validation

- Split/acyclic edges and isolated consumers compile; old reader bytes/defaults/aliases/failure order remain unchanged.
- Roundtrip/re-save: seven raw types, scalar/empty, all roles/aliases/distinct equal values, sharing/multiple roots,
  Linear MATMUL/PERMUTE/ADD, descriptors/detached lifetime. Differing input/encounter order, state encountered first,
  unused inputs and ignored prototype storage; copied/loaded graphs retain no original state.
- Malformed/bound tests: all six exact/below limits, unknown IDs/version, UTF-8/flags/counts/ranks/overflow/truncation/trailing
  data, references/order/aliases/keys/lengths, BOOL, unsupported/unreadable state, late failures/no mutation/publication.
- rebind accepts produced/repeated substitutions, retains upstream provenance/state, rejects full-descriptor/arity mismatch.
  One small GraphCheckpointEngineIntegrationTest proves cold load then explicit Engine forward/ordinary autograd equivalence,
  Linear and two fresh same-signature inputs. Retain old flexible-batch and private restore integrations; no new Model API.

Add GraphCheckpointTest, GraphArtifactCodecTest, GraphValueStorageTest in checkpoint.graph. Update exact
CheckpointDependencyContractTest/NnTrainingDependencyContractTest/EngineCompositionContractTest, retaining role-specific checks.
Add isolated checkpointApiConsumer (only Checkpoint); extend nnApiConsumer (only NN) to compile moved public signatures.
Wire consumer compilation into architecture test; declare modules/checkpoint/src/**/*.java file inputs. Assert mapping,
absent old sources/packages, exact api/implementation edges, no NN/Engine/Training imports even in generic tests and no
Compiler types in public GraphCheckpoint signatures. Training-only consumer remains. No broader exemptions.

Worker: focused development checks, then final affected-module/Javadoc once:

~~~bash
./gradlew :extensions:checkpoint:test :extensions:nn:test :extensions:checkpoint:javadoc :extensions:nn:javadoc --max-workers=2 --console=plain
./gradlew :testing:architecture-tests:test --tests '*CheckpointDependencyContractTest' --tests '*NnTrainingDependencyContractTest' --tests '*EngineCompositionContractTest' --max-workers=2 --console=plain
./gradlew :testing:integration-tests:test --tests '*GraphCheckpointEngineIntegrationTest' --tests '*WholeModelCheckpointEngineIntegrationTest' --tests '*PrivateTrainingRestoreIntegrationTest' --max-workers=2 --console=plain
~~~

Coordinator: one source-matched ./gradlew test javadoc --max-workers=2 --console=plain join, existing Metal dylib/configuration,
no rebuild. Reviewer reuses evidence unless executable changes/concrete findings invalidate it; final Javadoc after final edits.
No extra backend/codegen/performance gates. Planner: links/anchors/size/allowed diff/git diff --check only; no Gradle/native work.

## Documentation and review impact

Apply documentation-rules/general/planning profiles, and affected type profiles at implementation.
Update docs/architecture/{module-boundaries,dependency-rules,overview}.md, docs/developer-guide/repository-layout.md,
docs/api/{checkpoint-artifact-api,model-checkpoint-api}.md; add docs/api/graph-checkpoint-api.md. Migrate maintained examples
in docs/user-guide/{declarative-models,running-models}.md and docs/api/training-api.md. Review glossary checkpoint/raw-source/
alias terms and new/moved Javadocs; synchronize ADR0039/planning indexes. Show cold save/load versus flexible old readers;
preserve historical tasks, no TARGET Model API as implemented. Planning approval/Ready verification complete; independent code/docs
approval and coordinator gate precede Complete. Report extra production types/policy, never silently expand.

## Result

2026-10-17: coordinator accepted Complete; Halley definitive APPROVE covers code/tests/docs/Javadoc/glossary, zero outstanding.
Delivered GraphCheckpoint Source/Leaf/Role copy/save/load/rebind, GraphSource identity association,
GraphValueStorage raw host copying and GraphArtifactCodec graph-model.v1; no extra production types.
Moved six artifact files/build/tests to modules/checkpoint and nine NN files/five test files to nn.checkpoint;
settings retains :extensions:checkpoint; no forwarders, old readers/defaults/failures preserved.
Root/owning contract, architecture tests/isolated consumers, ADR0039, explanations/API/user/layout guides,
Javadocs and targeted glossary updated; Compiler cold API/contract unchanged; Model/Training additions stay Draft.
Worker session 84552: `./gradlew :extensions:checkpoint:test :extensions:nn:test :extensions:checkpoint:javadoc :extensions:nn:javadoc --max-workers=2 --console=plain` — exit 0, 3s; Checkpoint 36, NN 375, zero failures/errors/skips; Javadocs passed after final edits.
Worker session 4428: `./gradlew :testing:architecture-tests:test --tests '*CheckpointDependencyContractTest' --tests '*NnTrainingDependencyContractTest' --tests '*EngineCompositionContractTest' --max-workers=2 --console=plain` — exit 0, 1s; 4 cases and isolated Checkpoint/NN/Training consumers passed.
Worker session 23989: `./gradlew :testing:integration-tests:test --tests '*GraphCheckpointEngineIntegrationTest' --tests '*WholeModelCheckpointEngineIntegrationTest' --tests '*PrivateTrainingRestoreIntegrationTest' --max-workers=2 --console=plain` — exit 0, 1s; 6 cases, zero failures/errors/skips.
Coordinator gate: `SYNAPTIK_METAL_TEST_LIBRARY=/private/tmp/synaptik-checkpoint-leaf.oCW897/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc :testing:architecture-tests:test --rerun :testing:integration-tests:test --rerun --no-build-cache --no-configuration-cache --max-workers=2 --console=plain` — exit 0, BUILD SUCCESSFUL, 4s; 107 tasks (2 executed, 105 up-to-date).
Fresh architecture 12/0 skipped; full integration 132/1 skipped, zero failures/errors; both actual Metal training cases passed.
Integration skip: host CPU fixture had no eligible local workload alternatives. Existing verified Metal dylib reused, no rebuild.
Current 19-project reports: 4188 tests, zero failures/errors, 41 skipped; not all reports freshly executed:
CPU 30 and Metal 10 opt-in skips are cached; integration contributes one fixture skip. Worker Checkpoint/NN tests/Javadocs reused.
Coordinator verified all 2266 files identical across gate and both protected user changes unchanged; main/origin/main remain d235c0d3b89d6f2eaad4c80a59fd1c5091ea2851.
All original 57 Checkpoint cases retained; six artifact files byte-identical and nine NN algorithms identical ignoring package/Javadocs.
Final completion edits are metadata-only in the five coordinator-authorized files; links/whitespace and outside-scope/protected hashes checked.
No unresolved issue or required follow-up for this capability. Checkpoint frontier: None; Training0008 is next planning candidate, still Draft; NN0028/Model/withEngine remain future.
Status: Complete
