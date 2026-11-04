# Task CP0010 Fluent [CNN](../../../../glossary.md#term-convolutional-neural-network-cnn) and static recurrent [graph](../../../../glossary.md#graph) [persistence](../../../../glossary.md#term-persistence)

> **Editorial reading context — not a status update.** The application may use layer recipes,
> but this task stores their resulting Tensor dependency graph, not a recipe interpreter.
> Reachable zero initial states and row-index Tensors are named constants, distinct from NN
> trainable parameters. Load reconstructs expressions and private state without Engine;
> compute or newly configured training is a later call with its own capability checks.
> Multi-output producers must retain every canonical output identity, not just the requested
> root. Today's [selected CNN/recurrent format](../../../../reference/checkpoint/graph-formats.md#selected-cnn-and-recurrent-persistence)
> explains that source slice, while the [builder](../../../../reference/nn/model-builder.md#fluent-model-builder)
> explains the layer-to-expression side.

## Status

Complete jointly with CP0011 on 2026-10-20 after independent final APPROVE and main's focused
and native-enabled root gates. Main accepted the user-approved source/wire/contracts before Java
and verified completed predecessors, base and inactive conflicts. [Authorized frontier](../../../../glossary.md#term-authorized-frontier): None.

## Change class

Class C: [atomic](../../../../glossary.md#resource-transaction-and-rollback) [Compiler](../../../../glossary.md#term-compiler) cold-source/[Checkpoint](../../../../glossary.md#term-checkpoint) wire capability consumed by [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context).
Clean implementation and independent review required.

## Goal

Save an existing fluent [convolutional neural network](../../../../glossary.md#term-convolutional-neural-network-cnn) ([CNN](../../../../glossary.md#neural-network-families)) or statically [unrolled](../../../../glossary.md#backpropagation-through-time-and-recurrent-state) recurrent [Model](../../../../glossary.md#term-model),
load it without an [Engine](../../../../glossary.md#term-engine), then use ordinary compute and freshly configured training where current
adjoints permit. Include CP0011's cold flat-scan [persistence](../../../../glossary.md#term-persistence), not scan execution/training. Persist
the existing [Tensor](../../../../glossary.md#tensor) [directed acyclic graph](../../../../glossary.md#term-directed-acyclic-graph-dag) ([DAG](../../../../glossary.md#intermediate-representation-dag-and-ssa)), not layer recipes.

## [Scope](../../../../glossary.md#term-scope) and source inventory

Builder slice: [Linear](../../../../glossary.md#term-linear-meanings), RELU/[TANH](../../../../glossary.md#term-tanh)/[SIGMOID](../../../../glossary.md#term-sigmoid), Conv1d/2d/3d, MaxPool2d, flatten, [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn)/[GRU](../../../../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../../../../glossary.md#term-long-short-term-memory-lstm),
lastTimeStep. No per-network serializers.
The table maps builder composition to missing operations. CP0010 and [CP0011](0011-flat-recurrent-scan-persistence.md)
are ONE execution packet and completion gate; CP0011 is a non-independent scan supplement. CP0010's
foundation alone admits no RecurrentScan; the supplement supplies [admission](../../../../glossary.md#term-admission)/wires 20..22.

| Current source | Missing kinds and exact composition |
|---|---|
| Linear and [activations](../../../../glossary.md#term-activation) | Already admitted: [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), [weight](../../../../glossary.md#term-weight) PERMUTE[1,0], optional ADD and activations. |
| Conv1d | TensorConv1dExpressions uses input/weight EXPAND_DIMS(2), biased [CONV2D](../../../../glossary.md#convolution--conv2d), SQUEEZE(2); no CONV1D kind. |
| Conv2d/3d | Biased CONV2D/[CONV3D](../../../../glossary.md#conv3d--ncdhw), input/weight/[bias](../../../../glossary.md#term-bias) order; unit [stride](../../../../glossary.md#term-stride)/[dilation](../../../../glossary.md#term-dilation)/groups, zero [padding](../../../../glossary.md#padding). |
| MaxPool2d/flatten | MAX_POOL2D stride=[kernel](../../../../glossary.md#kernel), zero padding, unit dilation, floor; RESHAPE preserves [axis](../../../../glossary.md#term-axis) zero, checked remaining product. |
| Static sequences | SELECT(0,t), [GATHER](../../../../glossary.md#gather)(0) input/state rows, existing linear/activations; RESHAPE hidden to [1,B,H], CONCAT(0) to [TBF](../../../../glossary.md#term-tbf) [T,B,H]. |
| GRU/LSTM cells | [Rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) last-axis SLICE, normalized start/length, step 1; GRU uses SUB in candidate + update*(hidden-candidate). |
| Stacks/lastTimeStep | TBF feeds next layer; SELECT(0,T-1) yields [B,H]. LSTM returns hidden steps, not serialized result objects. |

[Model](../../../../glossary.md#term-model).fromComposition composes once in eval. OwnedModelSource.inventory assigns registered
[PARAMETER](../../../../glossary.md#term-parameter)/[BUFFER](../../../../glossary.md#term-buffer) roles, then collision-free __constant names to reachable [leaves](../../../../glossary.md#term-leaf). Static sequences
emit GATHER even for all-valid rows: [INT64](../../../../glossary.md#data-type--datatype) non-[gradient](../../../../glossary.md#term-gradient) row indices and floating [B,H] zero h/c
are [CONSTANTs](../../../../glossary.md#term-constant) outside stateDictionary. Persist reachable leaves only, shared [identities](../../../../glossary.md#term-identity-meanings)/roles and
distinct equal-valued constants. Preserve static [signatures](../../../../glossary.md#term-signature), full [descriptors](../../../../glossary.md#term-descriptor), paths/order/[raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity).
Builder [NCW](../../../../glossary.md#term-ncw)/[NCHW](../../../../glossary.md#term-nchw)/[NCDHW](../../../../glossary.md#conv3d--ncdhw) or TBF inputs are positive floating; TBF flatten yields [T,B*H], not [B,T*H].

Builder constructors produce one output per [node](../../../../glossary.md#node); MaxPool2d has no hidden index port and static
recurrent cell results are separate expressions. GraphCapture already retains every [canonical](../../../../glossary.md#term-canonical-form)
[producer](../../../../glossary.md#term-producer) output. Replace GraphSource's getFirst-only remap with ordered remapping of every output;
do not equate node number with value number. Admission stays closed by kind and exact [arity](../../../../glossary.md#term-arity).

## Accepted source and wire policy

No new public [API](../../../../glossary.md#term-application-programming-interface-api), production type or [dependency](../../../../glossary.md#term-dependency). Extend GraphCompilationPort's existing methods
via SourceGraphReconstruction.[preflight](../../../../glossary.md#term-preflight)/reconstruct and extend GraphArtifactCodec atomically.
Revise unpublished [graph](../../../../glossary.md#graph)-[model](../../../../glossary.md#term-model).v3 jointly for multi-output [normalization](../../../../glossary.md#term-normalization)/[topology](../../../../glossary.md#term-topology) and scans;
no intermediate one-output v3, later scan format or published v1/v2 field changes.

Accepted ArtifactFile v1 kind graph-model.v3, magic SYNGRF03, i32 version 3 retains header sections,
IDs 1..8 and [descriptor](../../../../glossary.md#term-descriptor)/role/[dtype](../../../../glossary.md#data-type--datatype) encodings. Each v3 [node](../../../../glossary.md#node) is u8 phase=1, u8 [opcode](../../../../glossary.md#term-opcode), input IDs,
attributes, positive i32 Q, then Q ordered output descriptors. CONCAT and scans put i32 input count
immediately after opcode; other kinds retain fixed input [arity](../../../../glossary.md#term-arity). All IDs are i32. [NodeId](../../../../glossary.md#nodeid) n outputs
are ValueIds I+S+sum(Q[k], k<n)+j for j=0..Q[n]-1; output IDs are implicit, never serialized twice.
Inputs precede the node's first output; checked I+S+sum(Q) <= Integer.MAX_VALUE.
Q=1 for wires 1..19; CP0011 alone specifies the scan records for 20..22. Fields below follow input
IDs and precede Q/descriptors, except CONCAT's count. Existing PERMUTE fields remain unchanged.

| Wire and kind | Inputs | Attribute bytes and accepted [admission](../../../../glossary.md#term-admission) |
|---|---|---|
| 9 SUB | 2 | None; [NoOperationAttrs](../../../../glossary.md#nooperationattrs).INSTANCE, floating operands. Rebuild a.sub(b). |
| 10 RESHAPE | 1 | i32 [target](../../../../glossary.md#term-target) [rank](../../../../glossary.md#term-rank) then i64 extents; TargetShapeAttrs, positive fully static target, equal checked [element](../../../../glossary.md#term-element) count. Rebuild a.reshape([shape](../../../../glossary.md#shape)). |
| 11 EXPAND_DIMS; 12 SQUEEZE | 1 | i32 [axis](../../../../glossary.md#term-axis); AxisTransformAttrs, axis 2 only. Rebuild a.expandDims(2)/squeeze(2). |
| 13 SELECT | 1 | i32 axis, i64 index; SelectAttrs, axis 0, rank-three input, in-bounds static index. Rebuild a.select(axis,index). |
| 14 [GATHER](../../../../glossary.md#gather) | 2 | i32 axis; IndexAxisAttrs, axis 0, [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) data and [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [INT64](../../../../glossary.md#data-type--datatype) non-[gradient](../../../../glossary.md#term-gradient) indices. Rebuild data.gather(indices,axis); ordinary execution retains index-value bounds checks. |
| 15 [SLICE](../../../../glossary.md#slice) | 1 | i32 entry count then ordered (i64 start, i64 length, i32 axis, i64 step); SliceAttrs restricted to one nonempty last-axis entry on rank two, step 1. Rebuild a.sliceByLength(starts,lengths,axes,steps), never reverse-normalize ends. |
| 16 CONCAT | Positive [variadic](../../../../glossary.md#term-variadic) | Immediately after opcode: i32 input count then that many preceding IDs, then i32 axis; CompositionAxisAttrs restricted to axis 0 and rank-three inputs. Rebuild [Tensor](../../../../glossary.md#tensor).concat(axis,inputs). |
| 17 [CONV2D](../../../../glossary.md#convolution--conv2d) | 3 | Seven i64 Conv2dAttrs components in declaration order; defaults (1,1,0,0,1,1,1) only. Rebuild input.conv2d([weight](../../../../glossary.md#term-weight),[bias](../../../../glossary.md#term-bias),attrs). |
| 18 [CONV3D](../../../../glossary.md#conv3d--ncdhw) | 3 | Ten i64 Conv3dAttrs components in declaration order; defaults (1,1,1,0,0,0,1,1,1,1) only. Rebuild input.conv3d(weight,bias,attrs). |
| 19 MAX_POOL2D | 1 | Eight i64 MaxPool2dAttrs components in declaration order, then u8 ceilMode; positive [kernels](../../../../glossary.md#kernel), [stride](../../../../glossary.md#term-stride)=[kernel](../../../../glossary.md#kernel), [padding](../../../../glossary.md#padding)=0, [dilation](../../../../glossary.md#term-dilation)=1, flag=0. Rebuild a.maxPool2d(attrs). |

Only builder geometry is selected; unbiased convolutions/other axes/geometries remain unsupported.
Model guards/[inference](../../../../glossary.md#term-inference-meanings) apply; never infer attrs from layer names/payloads.
Select the smallest sufficient version: old five kinds retain byte-identical v1, old eight-kind
graphs retain byte-identical v2, any added kind selects v3. Accept exact kind/magic/version tuples;
v1 admits 1..5, v2 1..8; joint v3 admits only 1..22 with exact packet-specific arities. CP0010's
foundation implements 1..19, not scan capability; joint acceptance requires 1..22. A v3 containing
only older kinds re-saves as v1/v2. Old readers
reject v3; retain all old graph and separate model.v1/whole-model.v1 readers and ArtifactFile [framing](../../../../glossary.md#term-framing).

Replace GraphArtifactCodec.canonicalTopology's value-boundary=node assumption with a call-local
value-to-[producer](../../../../glossary.md#term-producer)/port index and iterative producer-level DFS/postorder, never Java-stack recursion.
Visiting any output visits its producer once and retains all sibling descriptors, even unused ones;
reject dead producers, not unused
sibling ports. Preserve declared unused inputs, root order and normalized recapture equality.

Use CapturedGraphInference, never StaticResultLogicalLayoutClosure: GATHER checks descriptors,
not index storage; SELECT/SLICE derive [views](../../../../glossary.md#term-view-meanings) only from present [layouts](../../../../glossary.md#layout); CONCAT stays unresolved;
RESHAPE derives a view only from present [contiguous](../../../../glossary.md#term-contiguous) geometry, otherwise remaining unresolved.
Preserve exact descriptors/zero deferred obligations,
without raw layout closure, optimized-graph substitution or forced copies. Test unresolved raw
[MATMUL](../../../../glossary.md#matrix-multiplication--matmul)→SLICE/RESHAPE, CONCAT→SELECT/RESHAPE and storage-free GATHER.

## Approval checklist and contracts

- [x] User approved this exact source slice, narrow attributes and fixed three-input [convolution](../../../../glossary.md#term-channels-first-convolution-layers) [signatures](../../../../glossary.md#term-signature).
- [x] User approved joint v3 [node](../../../../glossary.md#node) record, prefix-sum IDs, ordered Q [descriptors](../../../../glossary.md#term-descriptor), CONCAT count, CP0011 scan wires, smallest-version selection and [fail-closed](../../../../glossary.md#term-fail-closed) policy.
- [x] User accepted coordinated cutover before Java: root [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [normative glossary](../../../../glossary.md#term-normative))</span>, [Compiler](../../../../glossary.md#term-compiler) [Compile lifecycle](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [Owned tensor Models](../../../../architecture/contracts/extensions-training.md#owned-tensor-models)<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor))</span>, [Checkpoint](../../../../glossary.md#term-checkpoint) [Generic graph source and values](../../../../architecture/contracts/extensions-training.md#generic-graph-source-and-values)<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>, [Canonical graph-model version 1](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-1) and [version 2](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-2), plus a new owning v3 heading.
- [x] Independent clean joint [planning](../../../../glossary.md#term-planning) review of revised [normalization](../../../../glossary.md#term-normalization)/wire and CP0011; iterative traversal and raw RESHAPE wording corrected. Not source/wire/user approval.
- [x] Main verified approvals/dependencies/base/conflicts before [Ready](../../../../glossary.md#term-ready); contract stage precedes Java.

Preserve root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants), [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>, [Explicit non-goals](../../../../../ARCHITECTURE.md#explicit-non-goals) and [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements); [Model](../../../../glossary.md#term-model) [modules/model](../../../../architecture/contracts/foundational-modules.md#modulesmodel)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span> owns [canonical](../../../../glossary.md#term-canonical-form) [producer](../../../../glossary.md#term-producer) wrappers. Stop on contract ambiguity.

## Dependencies and integration

- Depends on: Compiler0008/0009, Checkpoint0008/0009, Training0008, NN0028/0029 Complete; CPU0027 published baseline; joint approvals accepted.
- Conflicts with: [Compiler](../../../../glossary.md#term-compiler) source [admission](../../../../glossary.md#term-admission)/reconstruction, [Checkpoint](../../../../glossary.md#term-checkpoint) wire, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) source [ownership](../../../../glossary.md#term-ownership-meanings), authoritative/shared docs; serialize all.
- Parallel group: None.
- Common base revision: main==origin/main `8d5357e9a8698721a2252cb2ca3cb94150e705e8`.
- Integration order: accepted joint contracts/ADRs → common foundation → scan extension → independent clean review → main gate. One execution packet; neither brief becomes Complete alone. Git [commits](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) may stage work, not advertise partial v3; completed-task [commit](../../../../glossary.md#term-commit)/push follows joint acceptance. No separate [release](../../../../glossary.md#term-release) machinery or parallel [schema](../../../../glossary.md#term-schema) writer.
- Integration [validation](../../../../glossary.md#term-validation): focused integration/architecture checks below, then main's single source-matched repository test/[Javadoc](../../../../glossary.md#term-javadoc) checkpoint.
- Shared-document integration [owner](../../../../glossary.md#term-owner): main; exclusive writer delegation only.

## Files and symbols

Paths abbreviate io/github/pho001/synaptik; no added/moved types.

- [modules](../../../../glossary.md#term-module)/[compiler](../../../../glossary.md#term-compiler)/src/main/java/.../compiler/{SourceGraphReconstruction,GraphCompilationPort}.java: [admission](../../../../glossary.md#term-admission)/reconstruction/[Javadoc](../../../../glossary.md#term-javadoc); existing CapturedGraphInference and family [inference](../../../../glossary.md#term-inference-meanings) remain the raw-[descriptor](../../../../glossary.md#term-descriptor) [owners](../../../../glossary.md#term-owner).
- modules/[checkpoint](../../../../glossary.md#term-checkpoint)/src/main/java/.../checkpoint/[graph](../../../../glossary.md#graph)/{GraphArtifactCodec,GraphSource,GraphCheckpoint}.java: [opcode](../../../../glossary.md#term-opcode)/version/read/write/[dry sizing](../../../../glossary.md#term-dry-sizing), all-output remap, [producer](../../../../glossary.md#term-producer)-aware canonicalTopology and Javadoc. Reuse GraphValueStorage unchanged.
- extensions/nn/src/main/java/.../nn/module/{ModelBuilder,[Model](../../../../glossary.md#term-model),OwnedModelSource,ModelExecutionState}.java: admission Javadoc/tests, no [lifecycle](../../../../glossary.md#lifecycle) rewrite.
- Focused tests: Compiler SourceGraphCaptureTest/SourceGraphReconstructionTest; Checkpoint GraphCheckpointTest/GraphArtifactCodecTest; [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) ModelBuilderTest/ModelBuilderExecutionTest; integration FluentModelBuilderIntegrationTest/GraphCheckpointEngineIntegrationTest. Replace the explicit supported-[CNN](../../../../glossary.md#term-convolutional-neural-network-cnn)/[LSTM](../../../../glossary.md#term-long-short-term-memory-lstm) save rejections, retaining unsupported-case coverage.

## Acceptance and [validation](../../../../glossary.md#term-validation)

- [Compiler](../../../../glossary.md#term-compiler): each constructor, sharing/[canonical](../../../../glossary.md#term-canonical-form) wrappers, full [descriptors](../../../../glossary.md#term-descriptor), [opaque](../../../../glossary.md#term-opaque) substitutions, late-invalid/dead-[node](../../../../glossary.md#node) [preflight](../../../../glossary.md#term-preflight) before [producers](../../../../glossary.md#term-producer); no storage reads. CP0011 covers scan ports; load creates no [Engine](../../../../glossary.md#term-engine).
- [Checkpoint](../../../../glossary.md#term-checkpoint): independent golden fields/counts; byte-identical v1/v2; v3 roundtrip/re-save; malformed tuples/opcodes/attrs/[arity](../../../../glossary.md#term-arity)/Q/IDs/truncations/trailing bytes, dead producers, [aliases](../../../../glossary.md#aliasing-and-arena), [BOOL](../../../../glossary.md#data-type--datatype)/[raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity). Bound counts/ranks by remaining bytes before allocation; check prefix-sum [overflow](../../../../glossary.md#term-overflow), not I+S+N alone. Dry Sink traverses identical fields before [schema](../../../../glossary.md#term-schema) allocation; all six limits precede [host](../../../../glossary.md#term-host) [admission](../../../../glossary.md#term-admission)/copy. Envelope formulas remain M=72+58*S, aggregate=D+B, file=48+M+D+B, entries=1+S with D including Q/all descriptors. [Atomic publication](../../../../glossary.md#term-atomic-publication) and prior [target](../../../../glossary.md#term-target)/source [bindings](../../../../glossary.md#term-binding) survive failure; preflight precedes host import and canonical recapture precedes return.
- Small [FLOAT32](../../../../glossary.md#term-float32) Conv1d/2d/3d, pool/flatten/projection and static [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn)/[GRU](../../../../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../../../../glossary.md#term-long-short-term-memory-lstm) fixtures, including two-layer stacks/all-time/last-step outputs: save before compute → close → cold load → fresh-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) outputs → fresh [objective](../../../../glossary.md#term-objective)/[SGD](../../../../glossary.md#term-stochastic-gradient-descent-sgd) update for Conv1d/2d/static recurrence. Independent equations/SGD oracles compare all outputs/[parameter](../../../../glossary.md#term-parameter) updates; save trained values and reload. [Conv3d](../../../../glossary.md#conv3d--ncdhw) [inference](../../../../glossary.md#term-inference) roundtrips, training still rejects. Preserve role/alias/raw-bit/[signature](../../../../glossary.md#term-signature)/[layout](../../../../glossary.md#layout) cases and Training alias/overlap rejection. CP0011 scan tests stay cold.
- Preserve cold forward/[rebind](../../../../glossary.md#capture-graph-ports-and-reconstruction), live guards, [detached](../../../../glossary.md#detached-model-results) results, [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) Engine and cached preparations; no per-batch rebuild. Unsupported save leaves compute/train usable, without Engine/file work.

Completed worker/[module](../../../../glossary.md#term-module), main integration and repository validation commands and outcomes are recorded
once in Result below; independent review reuses successful executable evidence.

## Exclusions and documentation

No all-family/[Embedding](../../../../glossary.md#embedding)/LayerNorm/Data campaign, dynamic [dimensions](../../../../glossary.md#dimension)/control-flow [capture](../../../../glossary.md#term-capture-meanings), new public
types/[registry](../../../../glossary.md#term-registry)/reflection/[callback](../../../../glossary.md#term-callback) serializer, [Checkpoint](../../../../glossary.md#term-checkpoint)→[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) edge, alternate pipeline, [Conv3d](../../../../glossary.md#conv3d--ncdhw) adjoints,
scan [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt), [backend](../../../../glossary.md#backend)/tuning changes or performance campaign. No [objective](../../../../glossary.md#term-objective)/[optimizer](../../../../glossary.md#term-optimizer)/cursor/[RNG](../../../../glossary.md#graph-rng-state--graphrngstate) [resume](../../../../glossary.md#schema-serialization-and-checkpoint);
configureTraining is fresh only.
CP0011 lists the coordinated contract/[ADR](../../../../glossary.md#term-architecture-decision-record-adr)/explanation/[API](../../../../glossary.md#term-application-programming-interface-api)/guide cutover for the joint packet.
Javadocs and glossary [Owned Model source](../../../../glossary.md#owned-model-source)/[Fluent Model builder](../../../../glossary.md#fluent-model-builder)/Fixed recurrent scan reviewed;
no new term or changed term meaning. Contract/ADR/explanation/API/guide cutover, implementation
and independent final review completed under main-owned integration.

## Result

Joint CP0010/CP0011 accepted 2026-10-20: bounded kinds 9..22, all-output [normalization](../../../../glossary.md#term-normalization),
[producer](../../../../glossary.md#term-producer)-aware iterative [topology](../../../../glossary.md#term-topology) and v3 Q/prefix records implemented. Old v1/v2 bytes, smallest-format
selection, readers and functional implementations retained; obsolete single-output assumptions and
supported-save rejections replaced. No new production [API](../../../../glossary.md#term-application-programming-interface-api)/type/[dependency](../../../../glossary.md#term-dependency).

Changed Java/test paths below abbreviate `io/github/pho001/synaptik` as `...`; brace lists name exact files.

- `modules/compiler/src/main/java/.../compiler/{SourceGraphReconstruction,GraphCompilationPort}.java`; `modules/compiler/src/test/java/.../compiler/{SourceGraphCaptureTest,SourceGraphCapabilityTest}.java` (Capability new).
- `modules/checkpoint/src/main/java/.../checkpoint/graph/{GraphSource,GraphArtifactCodec,GraphCheckpoint}.java`; new `modules/checkpoint/src/test/java/.../checkpoint/graph/GraphArtifactV3Test.java`.
- `extensions/nn/src/main/java/.../nn/module/{Model,ModelBuilder}.java`; `extensions/nn/src/test/java/.../nn/module/{ModelBuilderTest,ModelBuilderExecutionTest,OwnedModelSourceTest}.java`.
- `testing/integration-tests/src/test/java/.../testing/integration/{FluentModelBuilderIntegrationTest,GraphCheckpointEngineIntegrationTest}.java`.

Documentation finalized independently: `ARCHITECTURE.md`; [compiler](../../../../glossary.md#term-compiler)-[autograd](../../../../glossary.md#autograd)/extensions-training/recurrent-scan
contracts; [lifecycle](../../../../glossary.md#lifecycle)/[module](../../../../glossary.md#term-module)-boundaries/overview; ADR0012/0039; [graph](../../../../glossary.md#graph)-[checkpoint](../../../../glossary.md#term-checkpoint)-api/public-api/running-models;
targeted glossary and six [planning](../../../../glossary.md#term-planning) records. All affected Javadocs reviewed; final reviewer changed comments
only. General/architecture/decision/API/user-guide profiles applied, plus general/planning for status recording.
No term meaning changed; protected notice/[CPU](../../../../glossary.md#term-cpu) [oracle](../../../../glossary.md#term-test-oracle) [hashes](../../../../glossary.md#term-hash) unchanged.

- Kepler worker: `./gradlew :modules:compiler:test :modules:compiler:javadoc :extensions:checkpoint:test :extensions:checkpoint:javadoc :extensions:nn:test :extensions:nn:javadoc --max-workers=2 --console=plain` passed, 24s, 43 tasks (7 executed, 36 up-to-date).
- After final Compiler RESHAPE-[overflow](../../../../glossary.md#term-overflow) translation/[regression](../../../../glossary.md#term-regression): `./gradlew :modules:compiler:test :modules:compiler:javadoc :extensions:checkpoint:test --tests '*GraphArtifact*Test' --tests '*GraphCheckpointTest' --max-workers=2 --console=plain` passed, 3s, 18 tasks (6 executed, 12 up-to-date); Compiler 321 and filtered Checkpoint 26 tests, zero failures/errors/skips.
- Main: `./gradlew :testing:architecture-tests:test :testing:integration-tests:test --tests '*FluentModelBuilderIntegrationTest' --tests '*GraphCheckpointEngineIntegrationTest' --max-workers=2 --console=plain` passed, 8s, 44 tasks (9 executed, 35 up-to-date); 13 architecture and 10 integration tests, zero failures/errors/skips.
- Independent Archimedes final APPROVE: no unresolved executable finding; 484 local links/anchors across 13 affected documents, whitespace and protected hashes passed. Successful worker evidence reused, no test reruns by review/status contexts.
- Clean status-recording pass: 263 local links/anchors across the six owned planning records and trailing-whitespace checks passed; scoped `git diff --check` passed. No protected files or executable behavior changed.
- First root `./gradlew test javadoc --max-workers=2 --console=plain` failed after 5m36s solely in four [Metal](../../../../glossary.md#term-metal) [Exp](../../../../glossary.md#term-expx)/[Sigmoid](../../../../glossary.md#term-sigmoid) native tests: `Path.of(System.getenv("SYNAPTIK_METAL_TEST_LIBRARY"))` received an unset environment value. CPU passed; no [backend](../../../../glossary.md#backend) source fix.
- Corrected main gate: `SYNAPTIK_METAL_TEST_LIBRARY=/Users/phujka/IdeaProjects/Synaptik/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc --no-configuration-cache --max-workers=2 --console=plain` passed, exit 0, 31s, 107 tasks (22 executed, 85 up-to-date). Metal tests executed and passed; unchanged successful CPU evidence reused up-to-date. All root tests/Javadocs passed. Current XML aggregates 643 suites/4307 tests, zero failures/errors, 41 skipped (4266 successful results), including up-to-date reuse, not 4307 fresh executions. Full affected suites: Compiler 321, Checkpoint 46, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 430, zero failures/errors/skips.

Limitations: scans remain cold-only; no backend execution/[BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt), [Conv3d](../../../../glossary.md#conv3d--ncdhw) adjoints or [optimizer](../../../../glossary.md#term-optimizer) [resume](../../../../glossary.md#schema-serialization-and-checkpoint).
No unresolved issue or required follow-up; no successor authorized. Seven production hashes matched
the independently reviewed versions at the final gate. Size exception: [atomic](../../../../glossary.md#resource-transaction-and-rollback) joint source/wire acceptance
and backward compatibility retained with compact evidence, rather than a mechanical split.

Status: Complete
