# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0043: Whitespace text and padded integral [batches](../../glossary.md#term-batch-machine-learning-and-hyperparameters)

## Status

Accepted — user approval and main authority approval, 2026-10-20. Data and Text are implemented;
the final repository [checkpoint](../../glossary.md#term-checkpoint) is tracked in the task brief. This [ADR](../../glossary.md#term-architecture-decision-record-adr) explains the
[authoritative root](../../../ARCHITECTURE.md), not a separate authority.

## Context and decision drivers

Accepted Data0001/Data0002 provide a narrow [Model](../../glossary.md#term-model)-only extension and [borrowed](../../glossary.md#ownership-borrowing-and-lease) right-padded [host](../../glossary.md#term-host)
[metadata](../../glossary.md#term-metadata). The next concrete consumer needs frozen [token](../../glossary.md#term-token) IDs and left/right host [padding](../../glossary.md#padding), not a
[tokenizer](../../glossary.md#frozen-whitespace-tokenizer) training/[persistence](../../glossary.md#term-persistence) program. Reusable numeric preparation must stay outside [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) and
must not create another execution or storage-[lifetime](../../glossary.md#term-lifetime) [owner](../../glossary.md#term-owner).

The drivers are a minimal complete public workflow, explicit padding geometry, [immutable](../../glossary.md#term-immutable) ID
meaning, unchanged [Embedding](../../glossary.md#embedding) semantics, checked bounded allocation and enforceable dependencies.

## Options considered

- Retain right-only metadata and duplicate left padding in Text: leaves two padding owners and
  cannot represent the selected consumer with the shared Data value.<span data-glossary-navigation="true"> ([metadata glossary](../../glossary.md#term-metadata); [padding glossary](../../glossary.md#padding))</span>
- Add a compatibility family beside the unpublished API: avoids source migration but leaves
  redundant types without a supported-release compatibility need.<span data-glossary-navigation="true"> ([API glossary](../../glossary.md#term-application-programming-interface-api); [release glossary](../../glossary.md#term-release))</span>
- Start with interfaces, BPE/training/artifacts and a general input pipeline: addresses broader
  future needs but adds unselected lifecycle, format and policy decisions.<span data-glossary-navigation="true"> ([BPE glossary](../../glossary.md#term-byte-pair-encoding-bpe); [artifacts glossary](../../glossary.md#term-artifact); [lifecycle glossary](../../glossary.md#lifecycle))</span>
- Select a concrete frozen whitespace tokenizer plus one Data padding owner and an explicit
  unpublished source migration. This is the accepted option.<span data-glossary-navigation="true"> ([frozen whitespace tokenizer glossary](../../glossary.md#frozen-whitespace-tokenizer); [owner glossary](../../glossary.md#term-owner))</span>

## Decision

Data owns `PaddingSide { LEFT, RIGHT }`, `PaddedSequenceBatch` and `LongSequenceBatcher` in its
existing flat package. Migrate `RightPaddedSequenceBatch` directly, without an [alias](../../glossary.md#aliasing-and-arena). Existing
[metadata](../../glossary.md#term-metadata) geometry/[borrowing](../../glossary.md#term-borrowing) behavior survives; lengths become position-independent counts, with
side selecting valid prefix or suffix. [Axis](../../glossary.md#term-axis) order remains explicit and independent of side.

The integral batcher requires explicit [padding](../../glossary.md#padding) value, side and order, pads long[] rows to [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)
maximum without [truncation](../../glossary.md#term-truncation-toward-zero), and creates one [contiguous](../../glossary.md#term-contiguous) non-[gradient](../../glossary.md#term-gradient) [INT64](../../glossary.md#data-type--datatype) [heap](../../glossary.md#term-heap) [Tensor](../../glossary.md#tensor) through
public [Model](../../glossary.md#term-model) allocation. Check [element](../../glossary.md#term-element) geometry incrementally before batch-sized metadata/output;
the element [carrier](../../glossary.md#term-storage-carrier) limit is Integer.MAX_VALUE and bytes use checked long arithmetic, not an int
byte cap. Fill the allocated destination directly, with no full-size staging/import recopy.
Callers coordinate mutation during [synchronous](../../glossary.md#term-synchronous) batching; no transactional [snapshot](../../glossary.md#term-snapshot) is promised.

Text gets exactly `api(project(":extensions:data"))`; Model is transitive, not a direct edge.
The flat concrete `WhitespaceTokenizer` copies ordered ordinary [tokens](../../glossary.md#term-token), reserves [PAD](../../glossary.md#frozen-whitespace-tokenizer)=0/[UNK](../../glossary.md#frozen-whitespace-tokenizer)=1,
assigns IDs from 2 and splits code-point runs using Character.isWhitespace(int). Preserve case,
punctuation and exact non-normalized [UTF-16](../../glossary.md#term-utf-16). `TextSequenceBatcher` encodes and delegates padding
once to Data. Neither Text nor Data depends on [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/[Engine](../../glossary.md#term-engine)/[backends](../../glossary.md#backend); NN gains no Data/Text edge.

No [normalization](../../glossary.md#term-normalization), [BOS](../../glossary.md#term-bos)/[EOS](../../glossary.md#term-eos), decoding, [BPE](../../glossary.md#term-byte-pair-encoding-bpe), vocabulary training, [artifact](../../glossary.md#term-artifact)/[fingerprint](../../glossary.md#term-fingerprint) or [persistence](../../glossary.md#term-persistence)
is selected. Frozen ordinary vocabulary strings have no magic special-token spelling. Broader
float/[sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters) Data and training/artifact Text plans remain deferred Draft.

## Rationale and scenario

For vocabulary `["Hello", "world"]`, encoding `"Hello world"` gives `[2,3]`, while `"missing"`
gives `[1]`. Left/[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)-major adaptation yields values `[[2,3],[0,1]]`, counts `[2,1]`
and LEFT. Data owns this geometry; Text owns the ID meaning. Choosing RIGHT yields `[1,0]` for
the second row without changing counts. The [Text API example](../../api/text-api.md#complete-example)<span data-glossary-navigation="true"> ([API glossary](../../glossary.md#term-application-programming-interface-api))</span>
composes these implemented APIs with unchanged [Embedding](../../glossary.md#embedding) and public [Engine](../../glossary.md#term-engine).

The external integration project composes unchanged Embedding and public
`engine.compute(embedding.forward(batch.values()))`. Its direct table-row [oracle](../../glossary.md#term-test-oracle) must cover
all side/order pairs and a distinct nonzero [PAD](../../glossary.md#padding) row. Moving this numerical test into Text would
require forbidden [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Engine edges; adding PAD row masking would change Embedding semantics.

## Consequences and migration

The unpublished source migration changes constructors/imports and requires explicit side in the
values Builder. The sole Data0003 worker migrates source/tests and the isolated architecture consumer.
Completed predecessor briefs stay historical. Direct [metadata](../../glossary.md#term-metadata) construction still inspects no
storage and admits long-extreme [Shapes](../../glossary.md#shape); allocation restrictions apply only to the batcher.

Text's new [module](../../glossary.md#term-module) and edge require [dependency](../../glossary.md#term-dependency)/namespace fixtures and a Text-only consumer proving
transitive Data/[Model](../../glossary.md#term-model) exposure. Integration receives Text only as a test dependency. Main owns
serialized shared-document integration, review coordination and the final root gate. The sole
Text0001 worker owns registration/enforcement and external numerical test plumbing; workers own
focused module [validation](../../glossary.md#term-validation), followed by independent targeted review and shared-doc finalization
after each worker stops.

LEFT metadata is not accepted by the current [scan](../../glossary.md#term-cumulative-scan) contract: scans remain right-valid-prefix only.
No implicit [host](../../glossary.md#term-host)-length [runtime](../../glossary.md#term-runtime) [binding](../../glossary.md#term-binding), [BPTT](../../glossary.md#term-backpropagation-through-time-bptt) extension, native [lifetime](../../glossary.md#term-lifetime) or execution pipeline is
introduced. Frozen vocabulary is deliberately less capable than a trained/persisted [tokenizer](../../glossary.md#frozen-whitespace-tokenizer);
same-sized vocabularies need not have the same ID mapping, and no [persistence](../../glossary.md#term-persistence) compatibility claim
is made here.

## Related documentation

- [Owning Data contract](../../architecture/contracts/extensions-training.md#extensionsdata)
  and [owning Text contract](../../architecture/contracts/extensions-training.md#extensionstext).
- [Module boundaries](../../architecture/module-boundaries.md#extensionsdata)<span data-glossary-navigation="true"> ([Module glossary](../../glossary.md#term-module))</span>.
- [Data0003](../../planning/extensions/data/tasks/0003-padded-integral-sequence-batching.md)
  and [Text0001](../../planning/extensions/text/tasks/0001-whitespace-tokenizer-and-data-batching.md).
- [ADR 0042](0042-data-extension-boundary.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr))</span> remains the historical Data-only foundation decision;
  this decision extends its selected capability and separately authorizes Text.
