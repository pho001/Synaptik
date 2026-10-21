# ADR 0044: Opt-in scan length Tensor export

## Status

Accepted — 2026-10-21. The user approved the exact narrow API and implementation scope with
“ok,pustse do toho”. Contract cutover and Java implementation are enacted; NN0022A is Complete
after independent APPROVE and main's source-matched architecture/root checkpoint on 2026-10-21.
This record explains, but does not override, the architecture contract.

## Context

Data already declares valid counts for borrowed padded tensors. Existing NN scan Models accept
ordinary values and runtime INT64 lengths through public Engine execution. Host metadata is not
itself a graph input, and requiring NN to consume Data objects would reverse the allowed dependency.
The current floating-only ModelBuilder has no Embedding recipe, so a count export cannot also
deliver a fluent Text IDs → Embedding → scan workflow.

## Decision drivers

Preserve metadata-only construction, exact descriptors and row order; make allocation explicit;
reuse existing Model/Engine owners; add only the selected public method without new types or edges.

## Options considered

- Keep manual application count materialization: no API change, but repeats the same descriptor
  and copy step for every padded numeric consumer.
- Export explicitly from the existing Data record: one bounded allocation without NN coupling.
- Accept Data objects in NN, cache mutable exports, or introduce a runner: adds dependency,
  mutation or lifecycle obligations unnecessary for this capability.

## Decision

Add exactly `public Tensor PaddedSequenceBatch.toScanLengthsTensor()` under the
[owning Data contract](../../architecture/contracts/extensions-training.md#extensionsdata).
Check RIGHT first and TIME_MAJOR next; reject either mismatch with IllegalArgumentException
before Tensor creation, allocation or storage access. On success, allocate through public
TensorFactory.allocate and fill public Model host storage from counts in original row order.
Every call returns a fresh ordinary heap canonical contiguous offset-zero non-view,
provenance-free non-gradient INT64 `[B]` leaf, including fresh empty storage for B=0.

Construction and accessors retain their existing contracts. Export never inspects values storage,
copies/reorders/repackages values, clears gradient flags, validates numeric backend support or
caches mutable output. Applications bind ordinary Tensors through existing NN/Engine APIs;
there is no native allocation, device transfer, close protocol or additional owner.

## Rationale

The record already knows the side, axes and immutable counts needed to reject incompatible
representations. For RIGHT/TIME_MAJOR counts `[3,1,0]`, export yields those exact three values;
the application can supply them beside canonical FLOAT32 `[3,3,F]` values to a scan Model.
LEFT counts cannot be passed as prefixes without moving values, so rejection prevents silently
executing the wrong interval. Fresh storage prevents one consumer's mutations leaking to another.

## Consequences

There is an explicit O(B) heap allocation/count-copy cost per export. Export can succeed for
storage-free values and is not execution admission: callers still satisfy exact descriptors and
storage lifetime. Existing Model and Engine preparations remain the only execution path; Data
retains only its Model dependency and NN gains no Data/Text edge.

The first integration proves canonical FLOAT32 numeric inference and prepared reuse across length
patterns. Text Embedding composition, LEFT/BATCH_MAJOR scans, BPTT and broader batching stay out
of scope. No predecessor brief is rewritten. Tests must cover failure ordering, fresh identity and
storage, mutation isolation, empty B and unchanged metadata geometry; external integration and
architecture/root validation precede task completion. Independent Class C review is required before acceptance.

## Related documentation

- [Root ownership routing](../../../ARCHITECTURE.md#module-ownership-routing)
- [Data boundary explanation](../../architecture/module-boundaries.md#extensionsdata)
- [NN0022A execution brief](../../planning/extensions/nn/tasks/0022a-padded-numeric-sequence-inference.md)
- [Earlier padded Data/Text decision](0043-whitespace-text-and-padded-integral-batches.md)
