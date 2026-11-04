# Task CP0011 Flat recurrent scan [persistence](../../../../glossary.md#term-persistence)

> **Editorial reading context — not a status update.** A flat fixed scan describes one
> recurrent producer with dense sequence and final-state output positions. Saving one selected
> position must not turn the sibling positions into different producers during load or rebind.
> Declared sequence, lengths and initial-state inputs are prototypes: their descriptors cross
> the cold boundary, not a saved runtime batch. Only registered named state contributes value
> payloads. Read today's [fixed-scan persistence example and restrictions](../../../../reference/checkpoint/graph-formats.md#selected-cnn-and-recurrent-persistence)
> alongside [NN runtime-length construction](../../../../reference/nn/recurrent.md#nn-runtime-length-recurrent-composition).
> Neither link converts this joint persistence supplement into historical execution or training evidence.

## Status

Complete jointly with CP0010 on 2026-10-20 after independent final APPROVE and main's focused
and native-enabled root gates; not an independently completed capability. Main verified approved
[scope](../../../../glossary.md#term-scope), predecessors, base and inactive conflicts. [Authorized frontier](../../../../glossary.md#term-authorized-frontier): None; execution/[BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt) excluded.

## Change class

Class C: cold [Compiler](../../../../glossary.md#term-compiler)/[Checkpoint](../../../../glossary.md#term-checkpoint) boundary and durable wire contract.

## Goal and [scope](../../../../glossary.md#term-scope)

Copy/save/load/[rebind](../../../../glossary.md#capture-graph-ports-and-reconstruction) RecurrentScan as one flat [node](../../../../glossary.md#node) with all [canonical](../../../../glossary.md#term-canonical-form) outputs, independently of
[backend](../../../../glossary.md#backend) execution. [CP0010](0010-fluent-cnn-and-static-recurrent-graph-persistence.md) owns the common v3 record/[normalization](../../../../glossary.md#term-normalization);
this supplement adds only RNN_TANH, GRU_RESET_AFTER and [LSTM](../../../../glossary.md#term-long-short-term-memory-lstm), biased/unbiased, FORWARD/REVERSE.
Together they are ONE execution packet/completion gate. No body/region/[callback](../../../../glossary.md#term-callback)/general serializer,
[registry](../../../../glossary.md#term-registry), new public wrapper/type, [module](../../../../glossary.md#term-module) or second [graph](../../../../glossary.md#graph).

## Verified source and boundary

Paths abbreviate the repository's io/github/pho001/synaptik package prefix.

- `modules/model/src/main/java/.../model/tensor/{RecurrentScan,RecurrentScanResult,LstmRecurrentScanResult}.java`: six overloads createDerivedOutputs once; carriers retain [canonical](../../../../glossary.md#term-canonical-form) slots, not persisted objects.
- `modules/compiler/src/main/java/.../compiler/{GraphCapture,SourceGraphReconstruction,RecurrentScanInference,GraphCompilationPort}.java`: [capture](../../../../glossary.md#term-capture-meanings) retains all outputIndex slots; joint [cold reconstruction](../../../../glossary.md#term-cold-reconstruction) now retains all admitted [producer](../../../../glossary.md#term-producer) outputs. Scan [inference](../../../../glossary.md#term-inference-meanings) reads no lengths and has no deferred obligations.
- `modules/checkpoint/src/main/java/.../checkpoint/graph/{GraphSource,GraphArtifactCodec,GraphCheckpoint,GraphValueStorage}.java`: joint remap/[codec](../../../../glossary.md#term-codec) retain all outputs and use producer-aware [topology](../../../../glossary.md#term-topology); GraphCapture and GraphValueStorage remain unchanged.
- `extensions/nn/src/main/java/.../nn/module/{Model,OwnedModelSource,ModelExecutionState}.java`: [Model](../../../../glossary.md#term-model).[graph](../../../../glossary.md#graph) copies Source; loadGraph adopts it; cold forward rebinds. Lists have no floating-only [guard](../../../../glossary.md#term-guard); builder still uses one floating input/static [unrolling](../../../../glossary.md#term-unrolling).

Source.inputs can declare sequence, [INT64](../../../../glossary.md#data-type--datatype) non-[gradient](../../../../glossary.md#term-gradient) lengths and initial states as unique
[provenance](../../../../glossary.md#provenance)-free [prototypes](../../../../glossary.md#term-prototype), disjoint from named state. Copy/load recreate storage-free prototypes;
only named [host](../../../../glossary.md#term-host) values are copied. Fixtures declare [runtime](../../../../glossary.md#term-runtime) lengths as input, not a frozen [CONSTANT](../../../../glossary.md#term-constant)
or inferred prototype value. Joint cold [admission](../../../../glossary.md#term-admission) now includes the bounded scan slice below.

## Accepted scan wire supplement

Use CP0010's [graph](../../../../glossary.md#graph)-[model](../../../../glossary.md#term-model).v3/SYNGRF03/version 3 and ordered Q-[descriptor](../../../../glossary.md#term-descriptor) record, not a v4.
Append these explicit u8 opcodes; do not use enum ordinals. After [opcode](../../../../glossary.md#term-opcode): i32 input count, ordered
i32 preceding ValueIds, u8 direction (1 FORWARD, 2 REVERSE), i32 Q, Q descriptors. No [bias](../../../../glossary.md#term-bias) flag:
the exact input count selects the public overload. No lengths, loop count or body attribute.

| Opcode and kind | Ordered inputs without optional trailing bias | Allowed count | Ordered outputs and Q |
|---|---|---|---|
| 20 RNN_TANH | input, validLengths, initialHidden, inputWeight, hiddenWeight | 5 or 6 | outputs, finalHidden; Q=2 |
| 21 GRU_RESET_AFTER | input, validLengths, initialHidden, inputWeight, hiddenWeight | 5 or 6 | outputs, finalHidden; Q=2 |
| 22 [LSTM](../../../../glossary.md#term-long-short-term-memory-lstm) | input, validLengths, initialHidden, initialCell, inputWeight, hiddenWeight | 6 or 7 | outputs, finalHidden, finalCell; Q=3 |

Call RecurrentScan.rnn/gru/lstm once with direction last; map all returned components to ValueIds
in the post-[preflight](../../../../glossary.md#term-preflight) call-local table. Never rebuild siblings. Retain one [producer](../../../../glossary.md#term-producer) when only
finalCell is consumed, downstream operands repeat ports or roots reorder slots; never merge scans.

Preserve RecurrentScanInference: common floating [dtype](../../../../glossary.md#data-type--datatype), input [T,B,I], lengths [INT64](../../../../glossary.md#data-type--datatype)[B] requiresGrad=false,
states [B,H], [weights](../../../../glossary.md#term-weight) [G*H,I]/[G*H,H], optional bias [G*H], G=1/3/4; I,H positive, T,B may be zero.
Outputs are [T,B,H], [B,H] and optional [B,H], unresolved [layout](../../../../glossary.md#layout) and the existing floating-role OR
requiresGrad. Full descriptors, ordered [signatures](../../../../glossary.md#term-signature) and [leaf](../../../../glossary.md#term-leaf) roles remain exact. Preflight checks
descriptors only; [runtime](../../../../glossary.md#term-runtime) length bounds [0,T] belong to execution, not cold [persistence](../../../../glossary.md#term-persistence).

## Contracts and approval checklist

The coordinated cutover now permits cold semantic [persistence](../../../../glossary.md#term-persistence) under recurrent
[Purity and lifecycle ownership](../../../../architecture/contracts/recurrent-scan.md#purity-and-lifecycle-ownership)<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
by reference to the owning [v3 contract](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-3), without [runtime](../../../../glossary.md#term-runtime) [artifacts](../../../../glossary.md#term-artifact) or execution capability.

- [x] User approved six overloads, runtime-length input, [canonical](../../../../glossary.md#term-canonical-form) slots and joint v3 wires 1..22. CP0010 foundation alone admits only 1..19, not scans.
- [x] Update root [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [normative glossary](../../../../glossary.md#term-normative))</span> routing; read/preserve CP0010's exact root/global and [Model](../../../../glossary.md#term-model) headings.
- [x] Update recurrent [Purity and lifecycle ownership](../../../../architecture/contracts/recurrent-scan.md#purity-and-lifecycle-ownership) to permit cold semantic persistence by [Checkpoint](../../../../glossary.md#term-checkpoint) reference, not prepared/runtime artifacts. Retain Fixed recurrent scan without [graph](../../../../glossary.md#graph) regions and its Fixed family and current Model surface, Static [Shape](../../../../glossary.md#shape) and runtime-value boundary, Valid lengths traversal and outputs, Performance and migration invariants headings.
- [x] Update [Compiler](../../../../glossary.md#term-compiler) [Compile lifecycle](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>, extensions [Owned tensor Models](../../../../architecture/contracts/extensions-training.md#owned-tensor-models)<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor))</span> and [Generic graph source and values](../../../../architecture/contracts/extensions-training.md#generic-graph-source-and-values); one owning Canonical graph-model version 3 heading there. Preserve CP0010's exact v1/v2 wire fields, bounds/[raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity)/[atomic publication](../../../../glossary.md#term-atomic-publication)/smallest-version policy; qualify v2 selection to the old eight-kind slice; [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation) on unknowns.
- [x] Coordinate ADR0012 [Purity state and serialization](../../../../design/decisions/0012-fixed-recurrent-scan-without-regions.md#purity-state-and-serialization)<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization))</span>, ADR0039 [Coordinated authority cutovers and validation](../../../../design/decisions/0039-portable-graph-model-persistence.md#coordinated-authority-cutovers-and-validation), [lifecycle](../../../../glossary.md#lifecycle).md's Planned fixed recurrent scan through the lifecycle, [module](../../../../glossary.md#term-module)-boundaries.md's Model/Compiler/Checkpoint/owned-Model sections, graph-checkpoint-api.md, public-api.md and running-models.md.
- [x] Java/Javadocs implemented; independent final code/docs APPROVE and main's joint [validation](../../../../glossary.md#term-validation) gates passed; evidence in CP0010 Result.
- [x] Independent joint [planning](../../../../glossary.md#term-planning) review with CP0010; iterative traversal, raw RESHAPE and duplicate-ID wording clarified. Not source/wire/user approval; earlier one-output review remains insufficient.
- [x] Main verified approvals/dependencies/base/conflicts before [Ready](../../../../glossary.md#term-ready); contract stage precedes Java.

## Dependencies and integration

- Depends on: CP0010's completed predecessors plus Model0025E/0025F/Compiler0006A; joint approvals accepted. Foundation-before-scan is internal order, not a separately Complete task. No [backend](../../../../glossary.md#backend)/[BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt)/held-[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/continuation/campaign prerequisite.
- Conflicts with: shared [Compiler](../../../../glossary.md#term-compiler) source/[Checkpoint](../../../../glossary.md#term-checkpoint) [schema](../../../../glossary.md#term-schema)/NN [admission](../../../../glossary.md#term-admission)/contracts/docs writers; serialize.
- Parallel group: None.
- Common base revision: main==origin/main `8d5357e9a8698721a2252cb2ca3cb94150e705e8`.
- Integration order: joint contracts/ADRs → common foundation → scan addition → independent implementation/docs review → main gate. Both complete together; Git staging [commits](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) are not completed-task acceptance or format [publication](../../../../glossary.md#term-publication). Normal completed-task [commit](../../../../glossary.md#term-commit)/push follows the joint gate; no extra [release](../../../../glossary.md#term-release) infrastructure.
- Integration [validation](../../../../glossary.md#term-validation): CP0010's combined architecture/integration and repository test/[Javadoc](../../../../glossary.md#term-javadoc) gate, once.
- Shared-document integration [owner](../../../../glossary.md#term-owner): main; exclusive delegation only. Joint packet Complete; [authorized frontier](../../../../glossary.md#term-authorized-frontier) None.

## Acceptance and [validation](../../../../glossary.md#term-validation)

- [Compiler](../../../../glossary.md#term-compiler) SourceGraphCaptureTest/SourceGraphReconstructionTest: 3 kinds × [bias](../../../../glossary.md#term-bias)/no bias × both directions; nonzero-only/reordered roots/consumers, shared siblings/repeated ports, distinct equal scans, [opaque](../../../../glossary.md#term-opaque)/repeated substitutions/pass-through [aliases](../../../../glossary.md#aliasing-and-arena). Assert [producer](../../../../glossary.md#term-producer).output(j), [descriptors](../../../../glossary.md#term-descriptor)/recapture. Stored roots stay distinct; [rebind](../../../../glossary.md#capture-graph-ports-and-reconstruction) may alias results.
- [Checkpoint](../../../../glossary.md#term-checkpoint) GraphCheckpointTest/GraphArtifactCodecTest: cold copy/save/load/re-save; golden scan/Q/prefix-sum bytes; input-to-output and state aliases; unused siblings versus dead producers; raw roles/bits, v1/v2 byte [identity](../../../../glossary.md#term-identity-meanings)/downgrade; T=0/B=0 cold cases.
- Reject malformed input/Q counts, direction, descriptors, op/version/tuple, negative/oversized counts, self/forward/dangling references, duplicate value definitions or stored roots, truncation/trailing bytes. Repeated input references, shared role [Tensors](../../../../glossary.md#tensor) and repeated port uses remain valid. Bound counts/cumulative values; full late-invalid/dead-[node](../../../../glossary.md#node) [preflight](../../../../glossary.md#term-preflight) precedes producer construction/[host](../../../../glossary.md#term-host) import. [Dry sizing](../../../../glossary.md#term-dry-sizing)/six limits precede storage [admission](../../../../glossary.md#term-admission); failure preserves [target](../../../../glossary.md#term-target)/source.
- [Capture](../../../../glossary.md#term-capture-meanings)/preflight/rebind read no storage, even dead/inaccessible [prototypes](../../../../glossary.md#term-prototype). Named-state reads follow [metadata](../../../../glossary.md#term-metadata)/size preflight. Prototype lengths never affect [schema](../../../../glossary.md#term-schema)/payloads; loaded lengths are storage-free and rebind to fresh exact descriptors. No [Engine](../../../../glossary.md#term-engine) on cold roundtrip.
- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) OwnedModelSourceTest: [Model](../../../../glossary.md#term-model).[graph](../../../../glossary.md#graph) → save → close → loadGraph → cold forward/re-save with sequence/[INT64](../../../../glossary.md#data-type--datatype) [runtime](../../../../glossary.md#term-runtime) lengths/optional states, output order/state aliases, no Engine. Single Model.load still rejects multiple boundaries; no builder redirection.

Completed validation commands and outcomes are recorded only in [CP0010 Result](0010-fluent-cnn-and-static-recurrent-graph-persistence.md#result), including [module](../../../../glossary.md#term-module)/[Javadoc](../../../../glossary.md#term-javadoc), focused integration/architecture and native-enabled root gates.

## Limitations and review impact

No [backend](../../../../glossary.md#backend) executes scans; [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt) stays [fail-closed](../../../../glossary.md#term-fail-closed). [Persistence](../../../../glossary.md#term-persistence) grants no compute/train support and
needs no runnable scan test. No prepared/native/backend [artifacts](../../../../glossary.md#term-artifact), [runtime](../../../../glossary.md#term-runtime) state/[prototype](../../../../glossary.md#term-prototype) values,
[RNG](../../../../glossary.md#graph-rng-state--graphrngstate)/[optimizer](../../../../glossary.md#term-optimizer)/cursor or automatic continuation.
CP0010 retains builder tests/no [Conv3d](../../../../glossary.md#conv3d--ncdhw) [gradients](../../../../glossary.md#term-gradient). Independent review/docs completed; glossary
Fixed recurrent scan/[Owned Model source](../../../../glossary.md#owned-model-source) introduce no new term meaning. Main integrates shared docs.

## Result

Completed with CP0010 on 2026-10-20: flat scan copy/save/load/[rebind](../../../../glossary.md#capture-graph-ports-and-reconstruction) preserves [canonical](../../../../glossary.md#term-canonical-form) sibling
ports and [runtime](../../../../glossary.md#term-runtime)-length input boundaries without [backend](../../../../glossary.md#backend) execution. [CP0010 Result](0010-fluent-cnn-and-static-recurrent-graph-persistence.md#result)
is the single home for changed files, documentation/[Javadoc](../../../../glossary.md#term-javadoc) review, [validation](../../../../glossary.md#term-validation) and final approval.
Scans remain cold-only; no [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt), [optimizer](../../../../glossary.md#term-optimizer) [resume](../../../../glossary.md#schema-serialization-and-checkpoint) or new term meaning. No unresolved issue or
required follow-up; [authorized frontier](../../../../glossary.md#term-authorized-frontier) None.

Status: Complete
