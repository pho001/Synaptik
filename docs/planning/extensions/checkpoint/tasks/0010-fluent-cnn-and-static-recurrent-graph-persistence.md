# Task CP0010 Fluent CNN and static recurrent graph persistence

## Status

Complete jointly with CP0011 on 2026-10-20 after independent final APPROVE and main's focused
and native-enabled root gates. Main accepted the user-approved source/wire/contracts before Java
and verified completed predecessors, base and inactive conflicts. Authorized frontier: None.

## Change class

Class C: atomic Compiler cold-source/Checkpoint wire capability consumed by NN.
Clean implementation and independent review required.

## Goal

Save an existing fluent convolutional neural network (CNN) or statically unrolled recurrent Model,
load it without an Engine, then use ordinary compute and freshly configured training where current
adjoints permit. Include CP0011's cold flat-scan persistence, not scan execution/training. Persist
the existing Tensor directed acyclic graph (DAG), not layer recipes.

## Scope and source inventory

Builder slice: Linear, RELU/TANH/SIGMOID, Conv1d/2d/3d, MaxPool2d, flatten, RNN/GRU/LSTM,
lastTimeStep. No per-network serializers.
The table maps builder composition to missing operations. CP0010 and [CP0011](0011-flat-recurrent-scan-persistence.md)
are ONE execution packet and completion gate; CP0011 is a non-independent scan supplement. CP0010's
foundation alone admits no RecurrentScan; the supplement supplies admission/wires 20..22.

| Current source | Missing kinds and exact composition |
|---|---|
| Linear and activations | Already admitted: MATMUL, weight PERMUTE[1,0], optional ADD and activations. |
| Conv1d | TensorConv1dExpressions uses input/weight EXPAND_DIMS(2), biased CONV2D, SQUEEZE(2); no CONV1D kind. |
| Conv2d/3d | Biased CONV2D/CONV3D, input/weight/bias order; unit stride/dilation/groups, zero padding. |
| MaxPool2d/flatten | MAX_POOL2D stride=kernel, zero padding, unit dilation, floor; RESHAPE preserves axis zero, checked remaining product. |
| Static sequences | SELECT(0,t), GATHER(0) input/state rows, existing linear/activations; RESHAPE hidden to [1,B,H], CONCAT(0) to TBF [T,B,H]. |
| GRU/LSTM cells | Rank-two last-axis SLICE, normalized start/length, step 1; GRU uses SUB in candidate + update*(hidden-candidate). |
| Stacks/lastTimeStep | TBF feeds next layer; SELECT(0,T-1) yields [B,H]. LSTM returns hidden steps, not serialized result objects. |

Model.fromComposition composes once in eval. OwnedModelSource.inventory assigns registered
PARAMETER/BUFFER roles, then collision-free __constant names to reachable leaves. Static sequences
emit GATHER even for all-valid rows: INT64 non-gradient row indices and floating [B,H] zero h/c
are CONSTANTs outside stateDictionary. Persist reachable leaves only, shared identities/roles and
distinct equal-valued constants. Preserve static signatures, full descriptors, paths/order/raw bits.
Builder NCW/NCHW/NCDHW or TBF inputs are positive floating; TBF flatten yields [T,B*H], not [B,T*H].

Builder constructors produce one output per node; MaxPool2d has no hidden index port and static
recurrent cell results are separate expressions. GraphCapture already retains every canonical
producer output. Replace GraphSource's getFirst-only remap with ordered remapping of every output;
do not equate node number with value number. Admission stays closed by kind and exact arity.

## Accepted source and wire policy

No new public API, production type or dependency. Extend GraphCompilationPort's existing methods
via SourceGraphReconstruction.preflight/reconstruct and extend GraphArtifactCodec atomically.
Revise unpublished graph-model.v3 jointly for multi-output normalization/topology and scans;
no intermediate one-output v3, later scan format or published v1/v2 field changes.

Accepted ArtifactFile v1 kind graph-model.v3, magic SYNGRF03, i32 version 3 retains header sections,
IDs 1..8 and descriptor/role/dtype encodings. Each v3 node is u8 phase=1, u8 opcode, input IDs,
attributes, positive i32 Q, then Q ordered output descriptors. CONCAT and scans put i32 input count
immediately after opcode; other kinds retain fixed input arity. All IDs are i32. NodeId n outputs
are ValueIds I+S+sum(Q[k], k<n)+j for j=0..Q[n]-1; output IDs are implicit, never serialized twice.
Inputs precede the node's first output; checked I+S+sum(Q) <= Integer.MAX_VALUE.
Q=1 for wires 1..19; CP0011 alone specifies the scan records for 20..22. Fields below follow input
IDs and precede Q/descriptors, except CONCAT's count. Existing PERMUTE fields remain unchanged.

| Wire and kind | Inputs | Attribute bytes and accepted admission |
|---|---|---|
| 9 SUB | 2 | None; NoOperationAttrs.INSTANCE, floating operands. Rebuild a.sub(b). |
| 10 RESHAPE | 1 | i32 target rank then i64 extents; TargetShapeAttrs, positive fully static target, equal checked element count. Rebuild a.reshape(shape). |
| 11 EXPAND_DIMS; 12 SQUEEZE | 1 | i32 axis; AxisTransformAttrs, axis 2 only. Rebuild a.expandDims(2)/squeeze(2). |
| 13 SELECT | 1 | i32 axis, i64 index; SelectAttrs, axis 0, rank-three input, in-bounds static index. Rebuild a.select(axis,index). |
| 14 GATHER | 2 | i32 axis; IndexAxisAttrs, axis 0, rank-two data and rank-one INT64 non-gradient indices. Rebuild data.gather(indices,axis); ordinary execution retains index-value bounds checks. |
| 15 SLICE | 1 | i32 entry count then ordered (i64 start, i64 length, i32 axis, i64 step); SliceAttrs restricted to one nonempty last-axis entry on rank two, step 1. Rebuild a.sliceByLength(starts,lengths,axes,steps), never reverse-normalize ends. |
| 16 CONCAT | Positive variadic | Immediately after opcode: i32 input count then that many preceding IDs, then i32 axis; CompositionAxisAttrs restricted to axis 0 and rank-three inputs. Rebuild Tensor.concat(axis,inputs). |
| 17 CONV2D | 3 | Seven i64 Conv2dAttrs components in declaration order; defaults (1,1,0,0,1,1,1) only. Rebuild input.conv2d(weight,bias,attrs). |
| 18 CONV3D | 3 | Ten i64 Conv3dAttrs components in declaration order; defaults (1,1,1,0,0,0,1,1,1,1) only. Rebuild input.conv3d(weight,bias,attrs). |
| 19 MAX_POOL2D | 1 | Eight i64 MaxPool2dAttrs components in declaration order, then u8 ceilMode; positive kernels, stride=kernel, padding=0, dilation=1, flag=0. Rebuild a.maxPool2d(attrs). |

Only builder geometry is selected; unbiased convolutions/other axes/geometries remain unsupported.
Model guards/inference apply; never infer attrs from layer names/payloads.
Select the smallest sufficient version: old five kinds retain byte-identical v1, old eight-kind
graphs retain byte-identical v2, any added kind selects v3. Accept exact kind/magic/version tuples;
v1 admits 1..5, v2 1..8; joint v3 admits only 1..22 with exact packet-specific arities. CP0010's
foundation implements 1..19, not scan capability; joint acceptance requires 1..22. A v3 containing
only older kinds re-saves as v1/v2. Old readers
reject v3; retain all old graph and separate model.v1/whole-model.v1 readers and ArtifactFile framing.

Replace GraphArtifactCodec.canonicalTopology's value-boundary=node assumption with a call-local
value-to-producer/port index and iterative producer-level DFS/postorder, never Java-stack recursion.
Visiting any output visits its producer once and retains all sibling descriptors, even unused ones;
reject dead producers, not unused
sibling ports. Preserve declared unused inputs, root order and normalized recapture equality.

Use CapturedGraphInference, never StaticResultLogicalLayoutClosure: GATHER checks descriptors,
not index storage; SELECT/SLICE derive views only from present layouts; CONCAT stays unresolved;
RESHAPE derives a view only from present contiguous geometry, otherwise remaining unresolved.
Preserve exact descriptors/zero deferred obligations,
without raw layout closure, optimized-graph substitution or forced copies. Test unresolved raw
MATMUL→SLICE/RESHAPE, CONCAT→SELECT/RESHAPE and storage-free GATHER.

## Approval checklist and contracts

- [x] User approved this exact source slice, narrow attributes and fixed three-input convolution signatures.
- [x] User approved joint v3 node record, prefix-sum IDs, ordered Q descriptors, CONCAT count, CP0011 scan wires, smallest-version selection and fail-closed policy.
- [x] User accepted coordinated cutover before Java: root [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts), Compiler [Compile lifecycle](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle), NN [Owned tensor Models](../../../../architecture/contracts/extensions-training.md#owned-tensor-models), Checkpoint [Generic graph source and values](../../../../architecture/contracts/extensions-training.md#generic-graph-source-and-values), [Canonical graph-model version 1](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-1) and [version 2](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-2), plus a new owning v3 heading.
- [x] Independent clean joint planning review of revised normalization/wire and CP0011; iterative traversal and raw RESHAPE wording corrected. Not source/wire/user approval.
- [x] Main verified approvals/dependencies/base/conflicts before Ready; contract stage precedes Java.

Preserve root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants), [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules), [Explicit non-goals](../../../../../ARCHITECTURE.md#explicit-non-goals) and [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements); Model [modules/model](../../../../architecture/contracts/foundational-modules.md#modulesmodel) owns canonical producer wrappers. Stop on contract ambiguity.

## Dependencies and integration

- Depends on: Compiler0008/0009, Checkpoint0008/0009, Training0008, NN0028/0029 Complete; CPU0027 published baseline; joint approvals accepted.
- Conflicts with: Compiler source admission/reconstruction, Checkpoint wire, NN source ownership, authoritative/shared docs; serialize all.
- Parallel group: None.
- Common base revision: main==origin/main `8d5357e9a8698721a2252cb2ca3cb94150e705e8`.
- Integration order: accepted joint contracts/ADRs → common foundation → scan extension → independent clean review → main gate. One execution packet; neither brief becomes Complete alone. Git commits may stage work, not advertise partial v3; completed-task commit/push follows joint acceptance. No separate release machinery or parallel schema writer.
- Integration validation: focused integration/architecture checks below, then main's single source-matched repository test/Javadoc checkpoint.
- Shared-document integration owner: main; exclusive writer delegation only.

## Files and symbols

Paths abbreviate io/github/pho001/synaptik; no added/moved types.

- modules/compiler/src/main/java/.../compiler/{SourceGraphReconstruction,GraphCompilationPort}.java: admission/reconstruction/Javadoc; existing CapturedGraphInference and family inference remain the raw-descriptor owners.
- modules/checkpoint/src/main/java/.../checkpoint/graph/{GraphArtifactCodec,GraphSource,GraphCheckpoint}.java: opcode/version/read/write/dry sizing, all-output remap, producer-aware canonicalTopology and Javadoc. Reuse GraphValueStorage unchanged.
- extensions/nn/src/main/java/.../nn/module/{ModelBuilder,Model,OwnedModelSource,ModelExecutionState}.java: admission Javadoc/tests, no lifecycle rewrite.
- Focused tests: Compiler SourceGraphCaptureTest/SourceGraphReconstructionTest; Checkpoint GraphCheckpointTest/GraphArtifactCodecTest; NN ModelBuilderTest/ModelBuilderExecutionTest; integration FluentModelBuilderIntegrationTest/GraphCheckpointEngineIntegrationTest. Replace the explicit supported-CNN/LSTM save rejections, retaining unsupported-case coverage.

## Acceptance and validation

- Compiler: each constructor, sharing/canonical wrappers, full descriptors, opaque substitutions, late-invalid/dead-node preflight before producers; no storage reads. CP0011 covers scan ports; load creates no Engine.
- Checkpoint: independent golden fields/counts; byte-identical v1/v2; v3 roundtrip/re-save; malformed tuples/opcodes/attrs/arity/Q/IDs/truncations/trailing bytes, dead producers, aliases, BOOL/raw bits. Bound counts/ranks by remaining bytes before allocation; check prefix-sum overflow, not I+S+N alone. Dry Sink traverses identical fields before schema allocation; all six limits precede host admission/copy. Envelope formulas remain M=72+58*S, aggregate=D+B, file=48+M+D+B, entries=1+S with D including Q/all descriptors. Atomic publication and prior target/source bindings survive failure; preflight precedes host import and canonical recapture precedes return.
- Small FLOAT32 Conv1d/2d/3d, pool/flatten/projection and static RNN/GRU/LSTM fixtures, including two-layer stacks/all-time/last-step outputs: save before compute → close → cold load → fresh-batch outputs → fresh objective/SGD update for Conv1d/2d/static recurrence. Independent equations/SGD oracles compare all outputs/parameter updates; save trained values and reload. Conv3d inference roundtrips, training still rejects. Preserve role/alias/raw-bit/signature/layout cases and Training alias/overlap rejection. CP0011 scan tests stay cold.
- Preserve cold forward/rebind, live guards, detached results, borrowed Engine and cached preparations; no per-batch rebuild. Unsupported save leaves compute/train usable, without Engine/file work.

Completed worker/module, main integration and repository validation commands and outcomes are recorded
once in Result below; independent review reuses successful executable evidence.

## Exclusions and documentation

No all-family/Embedding/LayerNorm/Data campaign, dynamic dimensions/control-flow capture, new public
types/registry/reflection/callback serializer, Checkpoint→NN edge, alternate pipeline, Conv3d adjoints,
scan BPTT, backend/tuning changes or performance campaign. No objective/optimizer/cursor/RNG resume;
configureTraining is fresh only.
CP0011 lists the coordinated contract/ADR/explanation/API/guide cutover for the joint packet.
Javadocs and glossary Owned Model source/Fluent Model builder/Fixed recurrent scan reviewed;
no new term or changed term meaning. Contract/ADR/explanation/API/guide cutover, implementation
and independent final review completed under main-owned integration.

## Result

Joint CP0010/CP0011 accepted 2026-10-20: bounded kinds 9..22, all-output normalization,
producer-aware iterative topology and v3 Q/prefix records implemented. Old v1/v2 bytes, smallest-format
selection, readers and functional implementations retained; obsolete single-output assumptions and
supported-save rejections replaced. No new production API/type/dependency.

Changed Java/test paths below abbreviate `io/github/pho001/synaptik` as `...`; brace lists name exact files.

- `modules/compiler/src/main/java/.../compiler/{SourceGraphReconstruction,GraphCompilationPort}.java`; `modules/compiler/src/test/java/.../compiler/{SourceGraphCaptureTest,SourceGraphCapabilityTest}.java` (Capability new).
- `modules/checkpoint/src/main/java/.../checkpoint/graph/{GraphSource,GraphArtifactCodec,GraphCheckpoint}.java`; new `modules/checkpoint/src/test/java/.../checkpoint/graph/GraphArtifactV3Test.java`.
- `extensions/nn/src/main/java/.../nn/module/{Model,ModelBuilder}.java`; `extensions/nn/src/test/java/.../nn/module/{ModelBuilderTest,ModelBuilderExecutionTest,OwnedModelSourceTest}.java`.
- `testing/integration-tests/src/test/java/.../testing/integration/{FluentModelBuilderIntegrationTest,GraphCheckpointEngineIntegrationTest}.java`.

Documentation finalized independently: `ARCHITECTURE.md`; compiler-autograd/extensions-training/recurrent-scan
contracts; lifecycle/module-boundaries/overview; ADR0012/0039; graph-checkpoint-api/public-api/running-models;
targeted glossary and six planning records. All affected Javadocs reviewed; final reviewer changed comments
only. General/architecture/decision/API/user-guide profiles applied, plus general/planning for status recording.
No term meaning changed; protected notice/CPU oracle hashes unchanged.

- Kepler worker: `./gradlew :modules:compiler:test :modules:compiler:javadoc :extensions:checkpoint:test :extensions:checkpoint:javadoc :extensions:nn:test :extensions:nn:javadoc --max-workers=2 --console=plain` passed, 24s, 43 tasks (7 executed, 36 up-to-date).
- After final Compiler RESHAPE-overflow translation/regression: `./gradlew :modules:compiler:test :modules:compiler:javadoc :extensions:checkpoint:test --tests '*GraphArtifact*Test' --tests '*GraphCheckpointTest' --max-workers=2 --console=plain` passed, 3s, 18 tasks (6 executed, 12 up-to-date); Compiler 321 and filtered Checkpoint 26 tests, zero failures/errors/skips.
- Main: `./gradlew :testing:architecture-tests:test :testing:integration-tests:test --tests '*FluentModelBuilderIntegrationTest' --tests '*GraphCheckpointEngineIntegrationTest' --max-workers=2 --console=plain` passed, 8s, 44 tasks (9 executed, 35 up-to-date); 13 architecture and 10 integration tests, zero failures/errors/skips.
- Independent Archimedes final APPROVE: no unresolved executable finding; 484 local links/anchors across 13 affected documents, whitespace and protected hashes passed. Successful worker evidence reused, no test reruns by review/status contexts.
- Clean status-recording pass: 263 local links/anchors across the six owned planning records and trailing-whitespace checks passed; scoped `git diff --check` passed. No protected files or executable behavior changed.
- First root `./gradlew test javadoc --max-workers=2 --console=plain` failed after 5m36s solely in four Metal Exp/Sigmoid native tests: `Path.of(System.getenv("SYNAPTIK_METAL_TEST_LIBRARY"))` received an unset environment value. CPU passed; no backend source fix.
- Corrected main gate: `SYNAPTIK_METAL_TEST_LIBRARY=/Users/phujka/IdeaProjects/Synaptik/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc --no-configuration-cache --max-workers=2 --console=plain` passed, exit 0, 31s, 107 tasks (22 executed, 85 up-to-date). Metal tests executed and passed; unchanged successful CPU evidence reused up-to-date. All root tests/Javadocs passed. Current XML aggregates 643 suites/4307 tests, zero failures/errors, 41 skipped (4266 successful results), including up-to-date reuse, not 4307 fresh executions. Full affected suites: Compiler 321, Checkpoint 46, NN 430, zero failures/errors/skips.

Limitations: scans remain cold-only; no backend execution/BPTT, Conv3d adjoints or optimizer resume.
No unresolved issue or required follow-up; no successor authorized. Seven production hashes matched
the independently reviewed versions at the final gate. Size exception: atomic joint source/wire acceptance
and backward compatibility retained with compact evidence, rather than a mechanical split.

Status: Complete
