# Task CP0011 Flat recurrent scan persistence

## Status

Complete jointly with CP0010 on 2026-10-20 after independent final APPROVE and main's focused
and native-enabled root gates; not an independently completed capability. Main verified approved
scope, predecessors, base and inactive conflicts. Authorized frontier: None; execution/BPTT excluded.

## Change class

Class C: cold Compiler/Checkpoint boundary and durable wire contract.

## Goal and scope

Copy/save/load/rebind RecurrentScan as one flat node with all canonical outputs, independently of
backend execution. [CP0010](0010-fluent-cnn-and-static-recurrent-graph-persistence.md) owns the common v3 record/normalization;
this supplement adds only RNN_TANH, GRU_RESET_AFTER and LSTM, biased/unbiased, FORWARD/REVERSE.
Together they are ONE execution packet/completion gate. No body/region/callback/general serializer,
registry, new public wrapper/type, module or second graph.

## Verified source and boundary

Paths abbreviate the repository's io/github/pho001/synaptik package prefix.

- `modules/model/src/main/java/.../model/tensor/{RecurrentScan,RecurrentScanResult,LstmRecurrentScanResult}.java`: six overloads createDerivedOutputs once; carriers retain canonical slots, not persisted objects.
- `modules/compiler/src/main/java/.../compiler/{GraphCapture,SourceGraphReconstruction,RecurrentScanInference,GraphCompilationPort}.java`: capture retains all outputIndex slots; joint cold reconstruction now retains all admitted producer outputs. Scan inference reads no lengths and has no deferred obligations.
- `modules/checkpoint/src/main/java/.../checkpoint/graph/{GraphSource,GraphArtifactCodec,GraphCheckpoint,GraphValueStorage}.java`: joint remap/codec retain all outputs and use producer-aware topology; GraphCapture and GraphValueStorage remain unchanged.
- `extensions/nn/src/main/java/.../nn/module/{Model,OwnedModelSource,ModelExecutionState}.java`: Model.graph copies Source; loadGraph adopts it; cold forward rebinds. Lists have no floating-only guard; builder still uses one floating input/static unrolling.

Source.inputs can declare sequence, INT64 non-gradient lengths and initial states as unique
provenance-free prototypes, disjoint from named state. Copy/load recreate storage-free prototypes;
only named host values are copied. Fixtures declare runtime lengths as input, not a frozen CONSTANT
or inferred prototype value. Joint cold admission now includes the bounded scan slice below.

## Accepted scan wire supplement

Use CP0010's graph-model.v3/SYNGRF03/version 3 and ordered Q-descriptor record, not a v4.
Append these explicit u8 opcodes; do not use enum ordinals. After opcode: i32 input count, ordered
i32 preceding ValueIds, u8 direction (1 FORWARD, 2 REVERSE), i32 Q, Q descriptors. No bias flag:
the exact input count selects the public overload. No lengths, loop count or body attribute.

| Opcode and kind | Ordered inputs without optional trailing bias | Allowed count | Ordered outputs and Q |
|---|---|---|---|
| 20 RNN_TANH | input, validLengths, initialHidden, inputWeight, hiddenWeight | 5 or 6 | outputs, finalHidden; Q=2 |
| 21 GRU_RESET_AFTER | input, validLengths, initialHidden, inputWeight, hiddenWeight | 5 or 6 | outputs, finalHidden; Q=2 |
| 22 LSTM | input, validLengths, initialHidden, initialCell, inputWeight, hiddenWeight | 6 or 7 | outputs, finalHidden, finalCell; Q=3 |

Call RecurrentScan.rnn/gru/lstm once with direction last; map all returned components to ValueIds
in the post-preflight call-local table. Never rebuild siblings. Retain one producer when only
finalCell is consumed, downstream operands repeat ports or roots reorder slots; never merge scans.

Preserve RecurrentScanInference: common floating dtype, input [T,B,I], lengths INT64[B] requiresGrad=false,
states [B,H], weights [G*H,I]/[G*H,H], optional bias [G*H], G=1/3/4; I,H positive, T,B may be zero.
Outputs are [T,B,H], [B,H] and optional [B,H], unresolved layout and the existing floating-role OR
requiresGrad. Full descriptors, ordered signatures and leaf roles remain exact. Preflight checks
descriptors only; runtime length bounds [0,T] belong to execution, not cold persistence.

## Contracts and approval checklist

The coordinated cutover now permits cold semantic persistence under recurrent
[Purity and lifecycle ownership](../../../../architecture/contracts/recurrent-scan.md#purity-and-lifecycle-ownership)
by reference to the owning [v3 contract](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-3), without runtime artifacts or execution capability.

- [x] User approved six overloads, runtime-length input, canonical slots and joint v3 wires 1..22. CP0010 foundation alone admits only 1..19, not scans.
- [x] Update root [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts) routing; read/preserve CP0010's exact root/global and Model headings.
- [x] Update recurrent [Purity and lifecycle ownership](../../../../architecture/contracts/recurrent-scan.md#purity-and-lifecycle-ownership) to permit cold semantic persistence by Checkpoint reference, not prepared/runtime artifacts. Retain Fixed recurrent scan without graph regions and its Fixed family and current Model surface, Static Shape and runtime-value boundary, Valid lengths traversal and outputs, Performance and migration invariants headings.
- [x] Update Compiler [Compile lifecycle](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle), extensions [Owned tensor Models](../../../../architecture/contracts/extensions-training.md#owned-tensor-models) and [Generic graph source and values](../../../../architecture/contracts/extensions-training.md#generic-graph-source-and-values); one owning Canonical graph-model version 3 heading there. Preserve CP0010's exact v1/v2 wire fields, bounds/raw bits/atomic publication/smallest-version policy; qualify v2 selection to the old eight-kind slice; fail closed on unknowns.
- [x] Coordinate ADR0012 [Purity state and serialization](../../../../design/decisions/0012-fixed-recurrent-scan-without-regions.md#purity-state-and-serialization), ADR0039 [Coordinated authority cutovers and validation](../../../../design/decisions/0039-portable-graph-model-persistence.md#coordinated-authority-cutovers-and-validation), lifecycle.md's Planned fixed recurrent scan through the lifecycle, module-boundaries.md's Model/Compiler/Checkpoint/owned-Model sections, graph-checkpoint-api.md, public-api.md and running-models.md.
- [x] Java/Javadocs implemented; independent final code/docs APPROVE and main's joint validation gates passed; evidence in CP0010 Result.
- [x] Independent joint planning review with CP0010; iterative traversal, raw RESHAPE and duplicate-ID wording clarified. Not source/wire/user approval; earlier one-output review remains insufficient.
- [x] Main verified approvals/dependencies/base/conflicts before Ready; contract stage precedes Java.

## Dependencies and integration

- Depends on: CP0010's completed predecessors plus Model0025E/0025F/Compiler0006A; joint approvals accepted. Foundation-before-scan is internal order, not a separately Complete task. No backend/BPTT/held-NN/continuation/campaign prerequisite.
- Conflicts with: shared Compiler source/Checkpoint schema/NN admission/contracts/docs writers; serialize.
- Parallel group: None.
- Common base revision: main==origin/main `8d5357e9a8698721a2252cb2ca3cb94150e705e8`.
- Integration order: joint contracts/ADRs → common foundation → scan addition → independent implementation/docs review → main gate. Both complete together; Git staging commits are not completed-task acceptance or format publication. Normal completed-task commit/push follows the joint gate; no extra release infrastructure.
- Integration validation: CP0010's combined architecture/integration and repository test/Javadoc gate, once.
- Shared-document integration owner: main; exclusive delegation only. Joint packet Complete; authorized frontier None.

## Acceptance and validation

- Compiler SourceGraphCaptureTest/SourceGraphReconstructionTest: 3 kinds × bias/no bias × both directions; nonzero-only/reordered roots/consumers, shared siblings/repeated ports, distinct equal scans, opaque/repeated substitutions/pass-through aliases. Assert producer.output(j), descriptors/recapture. Stored roots stay distinct; rebind may alias results.
- Checkpoint GraphCheckpointTest/GraphArtifactCodecTest: cold copy/save/load/re-save; golden scan/Q/prefix-sum bytes; input-to-output and state aliases; unused siblings versus dead producers; raw roles/bits, v1/v2 byte identity/downgrade; T=0/B=0 cold cases.
- Reject malformed input/Q counts, direction, descriptors, op/version/tuple, negative/oversized counts, self/forward/dangling references, duplicate value definitions or stored roots, truncation/trailing bytes. Repeated input references, shared role Tensors and repeated port uses remain valid. Bound counts/cumulative values; full late-invalid/dead-node preflight precedes producer construction/host import. Dry sizing/six limits precede storage admission; failure preserves target/source.
- Capture/preflight/rebind read no storage, even dead/inaccessible prototypes. Named-state reads follow metadata/size preflight. Prototype lengths never affect schema/payloads; loaded lengths are storage-free and rebind to fresh exact descriptors. No Engine on cold roundtrip.
- NN OwnedModelSourceTest: Model.graph → save → close → loadGraph → cold forward/re-save with sequence/INT64 runtime lengths/optional states, output order/state aliases, no Engine. Single Model.load still rejects multiple boundaries; no builder redirection.

Completed validation commands and outcomes are recorded only in [CP0010 Result](0010-fluent-cnn-and-static-recurrent-graph-persistence.md#result), including module/Javadoc, focused integration/architecture and native-enabled root gates.

## Limitations and review impact

No backend executes scans; BPTT stays fail-closed. Persistence grants no compute/train support and
needs no runnable scan test. No prepared/native/backend artifacts, runtime state/prototype values,
RNG/optimizer/cursor or automatic continuation.
CP0010 retains builder tests/no Conv3d gradients. Independent review/docs completed; glossary
Fixed recurrent scan/Owned Model source introduce no new term meaning. Main integrates shared docs.

## Result

Completed with CP0010 on 2026-10-20: flat scan copy/save/load/rebind preserves canonical sibling
ports and runtime-length input boundaries without backend execution. [CP0010 Result](0010-fluent-cnn-and-static-recurrent-graph-persistence.md#result)
is the single home for changed files, documentation/Javadoc review, validation and final approval.
Scans remain cold-only; no BPTT, optimizer resume or new term meaning. No unresolved issue or
required follow-up; authorized frontier None.

Status: Complete
