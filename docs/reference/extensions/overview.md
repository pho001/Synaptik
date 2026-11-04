# Data, Text and [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) public boundaries

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)

These extensions meet at ordinary [Tensors](../../glossary.md#tensor) but do different jobs: Data arranges numeric inputs,
Text maps text fragments to integer identifiers, and NN (neural networks) owns layers and their
named state. Preparing a [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) does not execute a network, and constructing a network expression
does not train it. The application explicitly connects those stages.

<a id="current-public-contracts"></a>
## Current public contracts

The implemented `extensions:data` surface pairs existing [Tensor](../../glossary.md#tensor) values with [immutable](../../glossary.md#term-immutable) [host](../../glossary.md#term-host)
`ValidSequenceLengths`, logical `SequenceAxisOrder` and explicit `PaddingSide` in a
`PaddedSequenceBatch` record, replacing the unpublished right-only type without an [alias](../../glossary.md#aliasing-and-arena).
Its values builder requires explicit order, side and counts, copies counts at the setter, and
validates geometry without inspecting Tensor contents. Direct record construction remains available.
`LongSequenceBatcher` instead allocates one fresh [INT64](../../glossary.md#data-type--datatype) [heap](../../glossary.md#term-heap) Tensor and pads ordered long-array
rows to their maximum length, on either side and in either [axis](../../glossary.md#term-axis) order, without truncation.
Neither path binds [runtime](../../glossary.md#term-runtime) length inputs or changes recurrent-[scan](../../glossary.md#term-fixed-recurrent-scan) semantics; see the
[Data sequence batching API](../data/sequences.md) for construction, [ownership](../../glossary.md#term-ownership-meanings) and failures.

The implemented `extensions:text` surface contains final `WhitespaceTokenizer` and
`TextSequenceBatcher` in one flat package. The [tokenizer](../../glossary.md#frozen-whitespace-tokenizer) freezes ordered ordinary IDs from 2,
reserves [padding](../../glossary.md#padding) ([PAD](../../glossary.md#frozen-whitespace-tokenizer))=0 and unknown ([UNK](../../glossary.md#frozen-whitespace-tokenizer))=1, and splits exact strings using
`Character.isWhitespace(int)`. The adapter requires explicit padding side and axis order,
encodes each row once and delegates all numeric batching to Data. Text exports only Data,
with [Model](../../glossary.md#term-model) transitive; [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/[Engine](../../glossary.md#term-engine) composition belongs to the caller. PAD selects an ordinary
[Embedding](../../glossary.md#embedding) row, not a [mask](../../glossary.md#term-boolean-mask) or a frozen zero row. See the [Text API](../text/tokenization.md) for a complete example.

The implemented `extensions:nn` declarative boundary adds `ModelSpecification` and
`DeclarativeModel` in `nn.module`, plus closed [`Activation`](../../glossary.md#term-activation) in `nn.layers`. Describe a
bound supported Sequential, reconstruct an [evaluation](../../glossary.md#term-evaluation)-mode [descriptor](../../glossary.md#term-descriptor)-only skeleton, then
use inherited strict state loading to supply values. The specification exposes immutable
typed steps, [canonical](../../glossary.md#term-canonical-form) current Tensor-[identity](../../glossary.md#term-identity-meanings) groups, ordered `stateDescriptors()` and
input-specific `outputDescriptor(...)`; it [captures](../../glossary.md#term-capture-meanings) neither arbitrary Model callbacks nor
bytes. Six parameterized layer families and Activation expose typed `specification()`
snapshots; `Linear.fromState(Tensor, Optional<Tensor>, boolean)` binds exact supplied state
while retaining supplied promotion or automatic input guards. Existing APIs stay supported.
`Module.trainingParameters()` now exposes Training's generic live [admission](../../glossary.md#term-admission) boundary. Its private
adapter retains and re-traverses the root's exact ordered paths, wrappers and Tensors, while
Training owns storage/value [validation](../../glossary.md#term-validation). Open/copy callers pass this adapter instead of [Module](../../glossary.md#term-module);
the same source survives explicit native copying. Training is physically `modules/training`
with unchanged `:extensions:training`/Java identities and exactly public Model/Engine edges;
NN exports Model/Training/[Checkpoint](../../glossary.md#term-checkpoint)/Engine. Owned Models provide high-level execution and
`withEngine`; the fluent builder composes existing layers without changing those [owners](../../glossary.md#term-owner).
See the [generic Training example](../training/parameters.md#generic-live-parameter-admission) and
the [NN composition reference](../nn/declarative.md#current-nn-declarative-composition-contract)
and [complete reconstruction example](../../user-guide/declarative-models.md). Whole-model files use
the separate ModelCheckpoint workflow; automatic Training-owned [resume](../../glossary.md#schema-serialization-and-checkpoint) is not added by the builder.


---

[Previous topic](../compatibility.md) · [Next topic](../model-trace.md)
