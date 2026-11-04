# Data, Text and NN public boundaries

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)

> **Scope:** Model construction, Compiler admission, backend execution, and checkpoint persistence are separate gates. Execution-status sentences retained in this detailed material record earlier qualification slices; they are not an exhaustive current support matrix. The [current backend matrix](../../backend-guide/production-support.md) and its incorporated contracts supersede narrower historical CPU/Metal lists. Package-private implementation sketches are not supported application API.

<a id="current-public-contracts"></a>
## Current public contracts

The implemented `extensions:data` surface pairs existing Tensor values with immutable host
`ValidSequenceLengths`, logical `SequenceAxisOrder` and explicit `PaddingSide` in a
`PaddedSequenceBatch` record, replacing the unpublished right-only type without an alias.
Its values builder requires explicit order, side and counts, copies counts at the setter, and
validates geometry without inspecting Tensor contents. Direct record construction remains available.
`LongSequenceBatcher` instead allocates one fresh INT64 heap Tensor and pads ordered long-array
rows to their maximum length, on either side and in either axis order, without truncation.
Neither path binds runtime length inputs or changes recurrent-scan semantics; see the
[Data sequence batching API](../data/sequences.md) for construction, ownership and failures.

The implemented `extensions:text` surface contains final `WhitespaceTokenizer` and
`TextSequenceBatcher` in one flat package. The tokenizer freezes ordered ordinary IDs from 2,
reserves padding (PAD)=0 and unknown (UNK)=1, and splits exact strings using
`Character.isWhitespace(int)`. The adapter requires explicit padding side and axis order,
encodes each row once and delegates all numeric batching to Data. Text exports only Data,
with Model transitive; NN/Engine composition belongs to the caller. PAD selects an ordinary
Embedding row, not a mask or a frozen zero row. See the [Text API](../text/tokenization.md) for a complete example.

The implemented `extensions:nn` declarative boundary adds `ModelSpecification` and
`DeclarativeModel` in `nn.module`, plus closed `Activation` in `nn.layers`. Describe a
bound supported Sequential, reconstruct an evaluation-mode descriptor-only skeleton, then
use inherited strict state loading to supply values. The specification exposes immutable
typed steps, canonical current Tensor-identity groups, ordered `stateDescriptors()` and
input-specific `outputDescriptor(...)`; it captures neither arbitrary Model callbacks nor
bytes. Six parameterized layer families and Activation expose typed `specification()`
snapshots; `Linear.fromState(Tensor, Optional<Tensor>, boolean)` binds exact supplied state
while retaining supplied promotion or automatic input guards. Existing APIs stay supported.
`Module.trainingParameters()` now exposes Training's generic live admission boundary. Its private
adapter retains and re-traverses the root's exact ordered paths, wrappers and Tensors, while
Training owns storage/value validation. Open/copy callers pass this adapter instead of Module;
the same source survives explicit native copying. Training is physically `modules/training`
with unchanged `:extensions:training`/Java identities and exactly public Model/Engine edges;
NN exports Model/Training/Checkpoint/Engine. Owned Models provide high-level execution and
`withEngine`; the fluent builder composes existing layers without changing those owners.
See the [generic Training example](../training/parameters.md#generic-live-parameter-admission) and
the [NN composition reference](../nn/declarative.md#current-nn-declarative-composition-contract)
and [complete reconstruction example](../../user-guide/declarative-models.md). Whole-model files use
the separate ModelCheckpoint workflow; automatic Training-owned resume is not added by the builder.


---

[Previous topic](../compatibility.md) · [Next topic](../model-trace.md)
