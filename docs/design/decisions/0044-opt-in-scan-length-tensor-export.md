# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0044: Opt-in [scan](../../glossary.md#term-cumulative-scan) length [Tensor](../../glossary.md#tensor) export

## Status

Accepted — 2026-10-21. The user approved the exact narrow [API](../../glossary.md#term-application-programming-interface-api) and implementation [scope](../../glossary.md#term-scope) with
“ok,pustse do toho”. Contract cutover and Java implementation are enacted; NN0022A is Complete
after independent APPROVE and main's source-matched architecture/root [checkpoint](../../glossary.md#term-checkpoint) on 2026-10-21.
This record explains, but does not override, the [architecture contract](../../glossary.md#architecture-contract).

## Context

Data already declares valid counts for [borrowed](../../glossary.md#ownership-borrowing-and-lease) padded [tensors](../../glossary.md#tensor). Existing [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [scan](../../glossary.md#term-cumulative-scan) Models accept
ordinary values and [runtime](../../glossary.md#term-runtime) [INT64](../../glossary.md#data-type--datatype) lengths through public [Engine](../../glossary.md#term-engine) execution. [Host](../../glossary.md#term-host) [metadata](../../glossary.md#term-metadata) is not
itself a [graph](../../glossary.md#graph) input, and requiring NN to consume Data objects would reverse the allowed [dependency](../../glossary.md#term-dependency).
The current floating-only ModelBuilder has no [Embedding](../../glossary.md#embedding) recipe, so a count export cannot also
deliver a fluent Text IDs → Embedding → scan workflow.

## Decision drivers

Preserve [metadata](../../glossary.md#term-metadata)-only construction, exact [descriptors](../../glossary.md#term-descriptor) and row order; make allocation explicit;
reuse existing [Model](../../glossary.md#term-model)/[Engine](../../glossary.md#term-engine) [owners](../../glossary.md#term-owner); add only the selected public method without new types or edges.

## Options considered

- Keep manual application count materialization: no API change, but repeats the same descriptor
  and copy step for every padded numeric consumer.<span data-glossary-navigation="true"> ([materialization glossary](../../glossary.md#materialization); [API glossary](../../glossary.md#term-application-programming-interface-api); [descriptor glossary](../../glossary.md#term-descriptor))</span>
- Export explicitly from the existing Data record: one bounded allocation without [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) coupling.
- Accept Data objects in NN, cache mutable exports, or introduce a runner: adds dependency,
  mutation or lifecycle obligations unnecessary for this capability.<span data-glossary-navigation="true"> ([cache glossary](../../glossary.md#term-cache); [mutable glossary](../../glossary.md#term-mutable); [dependency glossary](../../glossary.md#term-dependency); [lifecycle glossary](../../glossary.md#lifecycle))</span>

## Decision

Add exactly `public Tensor PaddedSequenceBatch.toScanLengthsTensor()` under the
[owning Data contract](../../architecture/contracts/extensions-training.md#extensionsdata).
Check RIGHT first and TIME_MAJOR next; reject either mismatch with IllegalArgumentException
before [Tensor](../../glossary.md#tensor) creation, allocation or storage access. On success, allocate through public
TensorFactory.allocate and fill public [Model](../../glossary.md#term-model) [host storage](../../glossary.md#host-storage) from counts in original row order.
Every call returns a fresh ordinary [heap](../../glossary.md#term-heap) [canonical](../../glossary.md#term-canonical-form) [contiguous](../../glossary.md#term-contiguous) offset-zero non-[view](../../glossary.md#term-view-meanings),
[provenance](../../glossary.md#provenance)-free non-[gradient](../../glossary.md#term-gradient) [INT64](../../glossary.md#data-type--datatype) `[B]` [leaf](../../glossary.md#term-leaf), including fresh empty storage for B=0.

Construction and accessors retain their existing contracts. Export never inspects values storage,
copies/reorders/repackages values, clears gradient flags, validates numeric [backend](../../glossary.md#backend) support or
caches [mutable](../../glossary.md#term-mutable) output. Applications bind ordinary Tensors through existing [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/[Engine](../../glossary.md#term-engine) APIs;
there is no native allocation, [device](../../glossary.md#term-device) transfer, close protocol or additional [owner](../../glossary.md#term-owner).

## Rationale

The record already knows the side, [axes](../../glossary.md#term-axis) and [immutable](../../glossary.md#term-immutable) counts needed to reject incompatible
representations. For RIGHT/TIME_MAJOR counts `[3,1,0]`, export yields those exact three values;
the application can supply them beside [canonical](../../glossary.md#term-canonical-form) [FLOAT32](../../glossary.md#term-float32) `[3,3,F]` values to a [scan](../../glossary.md#term-cumulative-scan) [Model](../../glossary.md#term-model).
LEFT counts cannot be passed as prefixes without moving values, so rejection prevents silently
executing the wrong interval. Fresh storage prevents one consumer's mutations leaking to another.

## Consequences

There is an explicit [O(B)](../../glossary.md#memory-management-atomic-counters-and-complexity) [heap](../../glossary.md#term-heap) allocation/count-copy cost per export. Export can succeed for
storage-free values and is not execution [admission](../../glossary.md#term-admission): callers still satisfy exact [descriptors](../../glossary.md#term-descriptor) and
storage [lifetime](../../glossary.md#term-lifetime). Existing [Model](../../glossary.md#term-model) and [Engine](../../glossary.md#term-engine) preparations remain the only execution path; Data
retains only its Model [dependency](../../glossary.md#term-dependency) and [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) gains no Data/Text edge.

The first integration proves [canonical](../../glossary.md#term-canonical-form) [FLOAT32](../../glossary.md#term-float32) numeric [inference](../../glossary.md#term-inference) and prepared reuse across length
patterns. Text [Embedding](../../glossary.md#embedding) composition, LEFT/BATCH_MAJOR scans, [BPTT](../../glossary.md#term-backpropagation-through-time-bptt) and broader batching stay out
of [scope](../../glossary.md#term-scope). No predecessor brief is rewritten. Tests must cover failure ordering, fresh [identity](../../glossary.md#term-identity-meanings) and
storage, mutation isolation, empty B and unchanged [metadata](../../glossary.md#term-metadata) geometry; external integration and
architecture/root [validation](../../glossary.md#term-validation) precede task completion. Independent Class C review is required before acceptance.

## Related documentation

- [Root ownership routing](../../../ARCHITECTURE.md#module-ownership-routing)<span data-glossary-navigation="true"> ([ownership glossary](../../glossary.md#term-ownership-meanings))</span>
- [Data boundary explanation](../../architecture/module-boundaries.md#extensionsdata)
- [NN0022A execution brief](../../planning/extensions/nn/tasks/0022a-padded-numeric-sequence-inference.md)
- [Earlier padded Data/Text decision](0043-whitespace-text-and-padded-integral-batches.md)
